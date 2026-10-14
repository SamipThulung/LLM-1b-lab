import json
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'allenai/OLMo-2-0425-1B'


def load_model(device, model_path=None):
    source = model_path or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep trainable parameters FP32; BF16 autocast is applied during forward passes.
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32, attn_implementation='sdpa')
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer


def load_dataset(path, limit=None):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows if limit is None else rows[:limit]


def get_ground_truth(example):
    if 'ground_truth' in example:
        return str(example['ground_truth'])
    answer = str(example['answer'])
    return answer.rsplit('####', 1)[-1].strip() if '####' in answer else answer.strip()


def make_prompt(question):
    return (
        'A conversation between User and Assistant. The Assistant solves math problems. '
        'The reasoning is enclosed in <think> </think> and the final answer in <answer> </answer>.\n'
        'User: There are 15 trees and then 21 trees. How many were planted?\n'
        'Assistant: <think> 21 - 15 = 6. </think> <answer> 6 </answer>\n'
        f'User: {question}\nAssistant: <think>'
    )


def save_checkpoint(model, tokenizer, optimizer, step, checkpoint_dir='checkpoints'):
    path = os.path.join(checkpoint_dir, f'step_{step}')
    model_dir = os.path.join(path, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    state = {
        'step': step,
        'optimizer': optimizer.state_dict(),
        'python_rng': random.getstate(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, os.path.join(path, 'optimizer.pt'))
    print(f'Checkpoint saved: {path}')


def load_checkpoint(checkpoint_path, device, learning_rate=1e-5):
    model, tokenizer = load_model(device=device, model_path=os.path.join(checkpoint_path, 'model'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    state = torch.load(os.path.join(checkpoint_path, 'optimizer.pt'), map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return model, tokenizer, optimizer, int(state['step'])
