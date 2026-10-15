# python module
import math
import os
import random
from contextlib import contextmanager
import time
from pathlib import Path
import csv

# torch and numpy imports
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tokenizers import Tokenizer

# megatron-core imports
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformer_from_scratch_01.config import (
    N_MODEL,
    S_MODEL,
    M_MODEL,
    L_MODEL,
    XL_MODEL,
)
# repo modules
from megatron_07.model_block import build_llama_model
from megatron_07.distributed_helper import wrap_model_for_distributed_training, build_optimizer
from megatron_07.simple_tokenizer import SimpleTokenizer
# from megatron_07.dataset_helper import build_dataloader


#dataset
DATASET_FILEPATH = "megatron_07/megatron_2b"
TOKENIZER_PATH = "tokenizer_02/fineweb_bpe_32k.json"
CACHE_PATH = "megatron_07/cache"

MICRO_BATCH_SIZE = 2
NUM_MICROBATCHES = 8
CONTEXT_LENGTH = 1024
VOCAB_SIZE = 32000

NUM_ITERATIONS = 122071

SAVE_INTERVAL = 1000
GENERATION_INTERVAL = 1000
RESUME = True
MODEL_SIZE = "l"
CHECKPOINT_DIR = f"megatron_07/checkpoints_{MODEL_SIZE}"
LOG_DIR = "megatron_07/logs"
LOG_FILE = f"{LOG_DIR}/training_{MODEL_SIZE}.csv"

MAX_NEW_TOKENS = 1024

# Parallelism
TP_SIZE = 2
PP_SIZE = 1
WORLD_SIZE = 2
assert WORLD_SIZE % (TP_SIZE * PP_SIZE) == 0
DP_SIZE = WORLD_SIZE // (TP_SIZE * PP_SIZE)

# Optimizer
LR = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 10
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

SEED = 1337

MASTER_ADDR = "localhost"
MASTER_PORT = "29500"

MODEL_SIZES = {
    "n": N_MODEL,
    "s": S_MODEL,
    "m": M_MODEL,
    "l": L_MODEL,
    "xl": XL_MODEL,
}
_CAUSAL_MASK = None
VERBOSE_TIMING = True

@contextmanager
def time_block(name="Code block", rank=0):
    # CUDA is async: synchronize before and after so the timing is honest
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        # if VERBOSE_TIMING and rank == 0:
            # print(f"[{name}] took {time.perf_counter() - start:.4f} seconds", flush=True)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=PP_SIZE,
    )

def log_training_metrics(metrics):
    os.makedirs(LOG_DIR, exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(metrics.keys()),
        )

        if not file_exists or os.path.getsize(LOG_FILE) == 0:
            writer.writeheader()

        writer.writerow(metrics)
        
def cleanup():
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def get_lr(it):
    """Linear warmup + cosine decay."""
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    progress = (it - WARMUP_ITERS) / max(1, NUM_ITERATIONS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def get_causal_mask(device):
    global _CAUSAL_MASK
    if _CAUSAL_MASK is None:
        # True = masked out
        _CAUSAL_MASK = torch.triu(
            torch.ones(1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH,
                       dtype=torch.bool, device=device),
            diagonal=1,
        )
    return _CAUSAL_MASK

def build_dataloader():
    tokenizer = SimpleTokenizer(TOKENIZER_PATH)
    assert tokenizer.vocab_size == VOCAB_SIZE, (
        f"tokenizer vocab {tokenizer.vocab_size} != VOCAB_SIZE {VOCAB_SIZE}"
    )

    config = GPTDatasetConfig(
        random_seed=SEED,
        sequence_length=CONTEXT_LENGTH,
        blend=([DATASET_FILEPATH], None),   # single dataset, no weights
        split="98,1,1",
        path_to_cache=CACHE_PATH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
    )

    # Sizes are NUMBER OF SAMPLES (sequences), not tokens.
    train_samples = NUM_ITERATIONS * NUM_MICROBATCHES * MICRO_BATCH_SIZE * DP_SIZE
    sizes = [train_samples, 1_000, None]

    # Every rank builds/loads the datasets (rank 0 builds the index first).
    # TP ranks must see identical data, so no TP-dependent logic here.
    train_ds, _valid_ds, _test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, sizes, lambda: True, config
    ).build()

    # GPTDataset already handles shuffling internally, so shuffle=False.
    # The sampler splits data across DP ranks; TP ranks in the same DP
    # group share the same dp_rank and therefore get identical batches.
    sampler = DistributedSampler(
        train_ds,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=False,
        drop_last=True,
    )

    return DataLoader(
        train_ds,
        batch_size=MICRO_BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )


def forward_step_func(data_iterator, model):
    batch = next(data_iterator)

    # GPTDataset returns a dict.
    with time_block("Loading Batch"):
        x = batch["tokens"].cuda(non_blocking=True).long()
        y = batch["labels"].cuda(non_blocking=True).long()
    loss_mask = batch["loss_mask"].cuda(non_blocking=True).float()

    batch_size = x.shape[0]
    position_ids = (
        torch.arange(CONTEXT_LENGTH, device=x.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    attention_mask = get_causal_mask(x.device)

    

    with time_block("Forward pass"):
        output = model(
            input_ids=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=y,
        )

    def loss_func(output_tensor):
        losses = output_tensor.float()
        loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return loss, {"lm loss": loss.detach()}

    return output, loss_func

def save_checkpoint(model, optimizer, completed_steps, rank):
    """Write one checkpoint shard per rank, then publish a complete step."""
    folder = Path(CHECKPOINT_DIR) / f"step_{completed_steps:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rank_{rank}.pt"
    tmp = folder / f"rank_{rank}.pt.tmp"

    torch.save({
        "completed_steps": completed_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, path)
    dist.barrier()

    if rank == 0:
        latest = Path(CHECKPOINT_DIR) / "latest.txt"
        latest_tmp = Path(CHECKPOINT_DIR) / "latest.txt.tmp"
        latest_tmp.write_text(str(completed_steps))
        os.replace(latest_tmp, latest)
        print(f"Saved checkpoint at step {completed_steps}", flush=True)
    dist.barrier()


def load_checkpoint(model, optimizer, rank):
    """All ranks agree on the most recently completed checkpoint."""
    latest = Path(CHECKPOINT_DIR) / "latest.txt"
    step = int(latest.read_text().strip()) if rank == 0 and latest.exists() else 0
    step_tensor = torch.tensor([step], device=f"cuda:{rank}", dtype=torch.long)
    dist.broadcast(step_tensor, src=0)
    step = int(step_tensor.item())
    if step == 0:
        return 0

    path = Path(CHECKPOINT_DIR) / f"step_{step:08d}" / f"rank_{rank}.pt"
    checkpoint = torch.load(path, map_location=f"cuda:{rank}", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if rank == 0:
        print(f"Resumed from step {step}", flush=True)
    return step


@torch.no_grad()
def test_generation(model, tokenizer, rank):
    """Greedy generation, compatible with TP-sharded or full-vocab logits."""
    model.eval()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_src_rank = parallel_state.get_tensor_model_parallel_src_rank()
    device = torch.device(f"cuda:{rank}")

    eos_id = getattr(tokenizer, "eod", None)
    if eos_id is None:
        eos_id = tokenizer._tok.token_to_id("<|endoftext|>")

    prompts = ["There is a", "Once upon a time"]
    for prompt in prompts:
        prompt_ids = tokenizer._tok.encode(prompt, add_special_tokens=False).ids
        generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        max_new = min(MAX_NEW_TOKENS, CONTEXT_LENGTH - len(prompt_ids))

        for _ in range(max_new):
            seq_len = generated.size(1)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            attention_mask = torch.triu(
                torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool),
                diagonal=1,
            )
            logits = model(
                input_ids=generated,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None,
            )[:, -1, :]

            if logits.size(-1) == VOCAB_SIZE:
                # Model has already gathered the vocabulary.
                next_id = logits.argmax(dim=-1).view(1, 1)
            elif logits.size(-1) * tp_size == VOCAB_SIZE:
                # Each TP rank holds a contiguous vocabulary shard.
                local_value, local_index = logits.max(dim=-1)
                local_index += parallel_state.get_tensor_model_parallel_rank() * logits.size(-1)
                all_values = [torch.empty_like(local_value) for _ in range(tp_size)]
                all_indices = [torch.empty_like(local_index) for _ in range(tp_size)]
                dist.all_gather(all_values, local_value, group=tp_group)
                dist.all_gather(all_indices, local_index, group=tp_group)
                winner = torch.stack(all_values).argmax(dim=0)
                next_id = torch.stack(all_indices).gather(0, winner.unsqueeze(0)).view(1, 1)
            else:
                raise ValueError(f"Unexpected vocabulary dimension: {logits.shape}")

            # Ensure all TP ranks append exactly the same token.
            dist.broadcast(next_id, src=tp_src_rank, group=tp_group)
            generated = torch.cat([generated, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break

        if rank == 0:
            response_ids = generated[0, len(prompt_ids):].tolist()
            print("\n" + "=" * 60)
            print(f"PROMPT: {prompt}")
            print(f"GENERATED TOKENS: {len(response_ids)}")
            print(f"RESPONSE: {tokenizer._tok.decode(response_ids, skip_special_tokens=True)}")
            print("=" * 60, flush=True)
    model.train()


def train(rank, world_size):
    setup(rank, world_size)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # Must run before model init so TP-region RNG states are set up.
    model_parallel_cuda_manual_seed(SEED)

    model = build_llama_model(MODEL_SIZES[MODEL_SIZE], VOCAB_SIZE, TP_SIZE, PP_SIZE, CONTEXT_LENGTH)
    model.cuda()
    total_params = sum(p.numel() for p in model.parameters())
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    
    local_params = sum(p.numel() for p in model.parameters())
    # Count replicated parameters only once
    unique_local_params = sum(
        p.numel()
        for p in model.parameters()
        if getattr(p, "tensor_model_parallel", False) or tp_rank == 0
    )
    
    total_params = torch.tensor(
        unique_local_params,
        device="cuda",
        dtype=torch.long,
    )
    
    dist.all_reduce(
        total_params,
        op=dist.ReduceOp.SUM,
        group=tp_group,
    )
    if rank == 0:
        print("*" * 32)
        print(f"GPU 0 parameters: {local_params:,}")
        print(f"Full model parameters: {total_params.item():,}")
        print(f"Model size: {total_params.item() / 1e6:.2f}M")
        print("*" * 32)
    model = wrap_model_for_distributed_training(model)

    optimizer = build_optimizer(model, LR, MIN_LR, WEIGHT_DECAY, GRAD_CLIP)
    
    
    start_iteration = load_checkpoint(model, optimizer, rank) if RESUME else 0

    train_dataloader = build_dataloader()
    train_iterator = iter(train_dataloader)
    # Recreate the original sample position; each step consumes NUM_MICROBATCHES.
    for _ in range(start_iteration * NUM_MICROBATCHES):
        next(train_iterator)

    forward_backward_func = get_forward_backward_func()

    for iteration in range(start_iteration, NUM_ITERATIONS):
        lr = get_lr(iteration)

        for group in optimizer.param_groups:
            group["lr"] = lr

        # Clear BOTH Megatron-DDP's grad buffers and the optimizer's grads.
        model.zero_grad_buffer()
        optimizer.zero_grad()

        losses_per_microbatch = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=[model],                      # must be a list
            num_microbatches=NUM_MICROBATCHES,
            seq_length=CONTEXT_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            decoder_seq_length=CONTEXT_LENGTH,
            forward_only=False,
        )
        with time_block("Calculating Gradients"):
            # DP grad reduction + replicated-param (e.g. norm) grad sync.
            finalize_model_grads([model])

        with time_block("Backward pass"):
            update_successful, grad_norm, num_zeros = optimizer.step()

        with time_block("Loss"):
            # Average loss over microbatches, then over the DP group.
            # mean_loss = torch.stack([l["lm loss"] for l in losses_per_microbatch]).mean()
            mean_loss = torch.stack([l["lm loss"] for l in losses_per_microbatch]).sum()
            # print(f"Loss: {mean_loss:0.5f}")

        with time_block("Synchronize"):
            dist.all_reduce(
                mean_loss,
                op=dist.ReduceOp.AVG,
                group=parallel_state.get_data_parallel_group(),
            )

        if rank == 0:
            metrics = {
                "iteration": iteration + 1,
                "loss": mean_loss.item(),
                "learning_rate": lr,
                "grad_norm": (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                ),
                "step_ok": bool(update_successful),
                "gpu_memory_gb": (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                ),
            }
            log_training_metrics(metrics)
            print(
                f"iteration {iteration:4d} | lr {lr:.2e} | "
                f"loss {mean_loss.item():.4f} | "
                f"grad_norm {grad_norm} | "
                f"step_ok {update_successful}",
                flush=True,
            )

        if (iteration + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, iteration + 1, rank)
        if (iteration +1)% GENERATION_INTERVAL==0:
            tokenizer = SimpleTokenizer(TOKENIZER_PATH)
            dist.barrier()
            test_generation(model=model,tokenizer=tokenizer,rank=rank,)
            dist.barrier()
            

    if NUM_ITERATIONS > start_iteration and NUM_ITERATIONS % SAVE_INTERVAL != 0:
        save_checkpoint(model, optimizer, NUM_ITERATIONS, rank)

    tokenizer = SimpleTokenizer(TOKENIZER_PATH)

    dist.barrier()

    test_generation(model=model,tokenizer=tokenizer,rank=rank)

    dist.barrier()

    cleanup()



if __name__ == "__main__":
    # train()
    mp.spawn(
        train,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )