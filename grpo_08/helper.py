from __future__ import annotations
from typing import Callable
import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_ids, response_ids, pad_token_id, device):
    """Build shifted causal-LM tensors directly from sampled token IDs."""
    if len(prompt_ids) != len(response_ids) or not prompt_ids:
        raise ValueError('Expected equally sized nonempty prompt/response ID lists')
    sequences = [p + r for p, r in zip(prompt_ids, response_ids)]
    if any(not p or not r for p, r in zip(prompt_ids, response_ids)):
        raise ValueError('Prompt and response must both contain tokens')
    length = max(len(s) - 1 for s in sequences)
    bsz = len(sequences)
    inputs = torch.full((bsz, length), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, length), pad_token_id, dtype=torch.long)
    mask = torch.zeros((bsz, length), dtype=torch.bool)
    attention_mask = torch.zeros((bsz, length), dtype=torch.long)
    for i, (p, r, seq) in enumerate(zip(prompt_ids, response_ids, sequences)):
        n = len(seq) - 1
        inputs[i, :n] = torch.tensor(seq[:-1], dtype=torch.long)
        labels[i, :n] = torch.tensor(seq[1:], dtype=torch.long)
        attention_mask[i, :n] = 1
        mask[i, len(p)-1:n] = True
    return {
        'input_ids': inputs.to(device), 'labels': labels.to(device),
        'response_mask': mask.to(device), 'attention_mask': attention_mask.to(device),
    }


def get_response_log_probs(model, input_ids, labels, attention_mask, return_token_entropy=False):
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
    # Compute only target-token log probs without materializing a second vocab-sized tensor.
    log_probs = -F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
    result = {'log_probs': log_probs}
    if return_token_entropy:
        # Entropy requires full vocabulary probabilities and costs substantial memory.
        log_p = F.log_softmax(logits, dim=-1)
        result['token_entropy'] = -(log_p.exp() * log_p).sum(-1)
    return result


def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError('Responses and ground truths differ in length')
    entries = [reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)]
    rewards = torch.tensor([x['reward'] for x in entries], dtype=torch.float32)
    return rewards, {
        'mean_reward': rewards.mean().item(),
        'format_reward': sum(x['format_reward'] for x in entries) / len(entries),
    }


def compute_group_normalized_rewards(raw_rewards, group_size, advantage_eps=1e-6):
    if group_size < 2 or raw_rewards.numel() % group_size:
        raise ValueError('Group size must be >= 2 and divide number of responses')
    groups = raw_rewards.reshape(-1, group_size)
    means = groups.mean(-1, keepdim=True)
    stds = groups.std(-1, keepdim=True, correction=1)
    advantages = ((groups - means) / (stds + advantage_eps)).reshape(-1)
    return advantages, {'group_mean': means.flatten().tolist(),
                        'nonconstant_group_fraction': (stds.squeeze(-1) > 0).float().mean().item()}


def compute_policy_gradient_loss(advantages, policy_log_probs):
    # One on-policy optimization step per freshly sampled batch; no importance ratio.
    return -advantages[:, None] * policy_log_probs


def aggregate_loss_across_microbatch(per_token_loss, response_mask):
    mask = response_mask.to(per_token_loss.dtype)
    return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompts, ground_truths, group_size, max_new_tokens, temperature, device):
    if len(prompts) != len(ground_truths):
        raise ValueError('Prompts and ground truths differ in length')
    model.eval()
    repeated_prompts, texts, truths, all_prompt_ids, all_response_ids = [], [], [], [], []
    for prompt, truth in zip(prompts, ground_truths):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        encoded = torch.tensor([ids], dtype=torch.long, device=device)
        # Override disabled training cache only for generation.
        outputs = model.generate(
            input_ids=encoded, attention_mask=torch.ones_like(encoded),
            do_sample=True, temperature=temperature, num_return_sequences=group_size,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        for output in outputs:
            response = output[len(ids):].tolist()
            if tokenizer.eos_token_id in response:
                response = response[:response.index(tokenizer.eos_token_id) + 1]
            if not response: