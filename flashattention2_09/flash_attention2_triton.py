import triton
import triton.language as tl
import torch

@triton.jit
def flash_attention_fwd(
    # input: batch * sequcence * dim
    # Poiters
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    # Strides
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_lr,
    # Shape
    num_rows, D: tl.constexpr,
    # tensors
    tile_size_qr: tl.constexpr,  tile_size_kvr: tl.constexpr,
    scale,
    is_casual: tl.constexpr):
    
    q_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    q_offset = q_index * tile_size_qr + tl.arange(0, tile_size_qr)
    
    q_block_ptr = tl.make_block_ptr(
        # Mnemonics: Pizza Slices Should Only be Ordered. 
        #1. Ptr
        q_ptr + batch_index * stride_qb,
        #2. Shape
        shape = (num_rows, D),
        #3. Strides
        strides = (stride_qq, stride_qd),
        #4. Offsets
        offsets = (q_index * tile_size_qr, 0),
        #5. Block size
        block_shape = (tile_size_qr, D),
        #6. order
        order = (1, 0))
    
    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch_index * stride_kb,
        shape = (num_rows, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape = (tile_size_kvr, D),
        order = (1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch_index * stride_vb,
        shape = (num_rows, D),
        strides = (stride_vv, stride_vd),
        offsets = (0,0),
        block_shape = (tile_size_kvr, D),
        order = (1, 0)
    )

    q_tile = tl.load(q_block_ptr, boundary_check = (0, 1), padding_option="zero")

    # Recurring update variable: m = previous_maximum, l = rowsum(P), o = output
    m = tl.full((tile_size_qr,), value = float("-inf"), dtype = tl.float32)
    l = tl.zeros((tile_size_qr,), dtype = tl.float32)
    o = tl.zeros((tile_size_qr, D), dtype = tl.float32)
    
    for i in range(0, num_rows, tile_size_kvr):
        k_offset = i + tl.arange(0, tile_size_kvr)

        k_tile = tl.load(k_block_ptr, boundary_check = (0, 1), padding_option="zero")
        v_tile = tl.load(v_block_ptr, boundary_check = (0, 1), padding_option="zero")
    
        score = tl.dot(q_tile, tl.trans(k_tile))
        score = score * scale
    
        # For boundary checking the k padding
        valid_k =  q_offset < num_rows
    
        if is_casual:
          casual_mask = (
              q_offset[:, None] >= k_offset [None, : ]
          )
          mask = (casual_mask & valid_k[None, :])
    
        else:
          mask = valid_k[None, :]
        
        score = tl.where(mask, score, float("-inf"))
    
        max_ = tl.max(score, axis=-1)
        maximum = tl.maximum(max_, m)
        
        
        P = tl.exp(score-maximum[:, None])
        row_sum = tl.sum(P, axis=-1)
        
        alpha = tl.exp(m - maximum)
        
        l = alpha * l + row_sum
        o = alpha[:, None] * o + tl.dot(P, v_tile.to(tl.float32))
        
        m = maximum
    
        k_block_ptr = tl.advance(k_block_ptr, (tile_size_kvr, 0))
        v_block_ptr = tl.advance(v_block_ptr, (tile_size_kvr, 0))

    o = o / l[:, None]
    lse = m + tl.log(l)

    out_block_ptr = tl.make_block_ptr(
        out_ptr + batch_index * stride_ob,
        shape = (num_rows, D),
        strides = (stride_oo, stride_od),
        offsets = (q_index * tile_size_qr, 0),
        block_shape = (tile_size_qr, D),
        order = (1, 0))
    
    lse_block_ptr = tl.make_block_ptr(
        lse_ptr + batch_index * stride_lb,
        shape = (num_rows,),
        strides = (stride_lr,),
        offsets = (q_index * tile_size_qr,),
        block_shape = (tile_size_qr,),
        order = (0,),)

    tl.store(out_block_ptr, o, boundary_check = (0,1))
    tl.store(lse_block_ptr, lse, boundary_check = (0,))


@torch.compile
def flash_attention_bwd(
    Q, K, V, O, L,
    dO):
  Q, K, V, O, L = Q.float(), K.float(), V.float(), O.float(), L.float()

  batch, context_length, d = Q.shape
  scale = 1/(d ** 0.5)
  # precompute
  D = torch.sum(dO *O, axis=-1)
  # recompute score
  S = torch.matmul(Q, K.transpose(-1, -2)) * scale
  # mask
  causal_mask = torch.tril(
      torch.ones(
          (context_length, context_length),
          dtype=torch.bool,
          device=Q.device
      )
  )

  S = S.masked_fill(~causal_mask, float("-inf"))

  # recompute P
  P = torch.exp(S - L.unsqueeze(-1))

  # gradient
  dV = torch.matmul(P.transpose(-1, -2), dO)
  dP = torch.matmul(dO, V.transpose(-1, -2))
  dS = P * (dP - D.unsqueeze(-1))
  dQ = torch.matmul(dS, K) * scale
  dK = torch.matmul(dS.transpose(-1, -2), Q) * scale

  return dQ, dK, dV


class FlashAttention2(torch.autograd.Function):

  @staticmethod
  def forward(
        ctx,
        Q, K, V,
        tile_size_r
      ):

    B, context_length, d = Q.shape

    LSE = torch.empty((B, context_length), dtype=torch.float32, device=Q.device)
    O = torch.empty_like(Q, dtype=torch.float32)
    scale = 1 / (d ** 0.5)

    num_tiles = triton.cdiv(context_length, tile_size_r)
    grid = (num_tiles, B)
    flash_attention_fwd[grid](
        Q, K, V, O, LSE,
        # Strides
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1),
        # Shape
        context_length, d,
        # tensors
        tile_size_r,  tile_size_r,
        scale,
        is_casual = True
    )

    ctx.save_for_backward(Q, K, V, O, LSE)
    ctx.scale = scale

    return O

  @staticmethod
  def backward(ctx, dO):

    Q, K, V, O, L = ctx.saved_tensors

    dQ, dK, dV = flash_attention_bwd(Q, K, V, O, L, dO)

    return dQ, dK, dV, None

