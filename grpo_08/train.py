import csv
import os
import random
import torch

from grpo_08.helper import generate_rollouts, grpo_train_step, grpo_val_step
from grpo_08.load_model import (load_model, load_dataset, get_ground_truth,
                                make_prompt, save_checkpoint, load_checkpoint)
from grpo_08.dan_hendrycks_parser import r1_zero_reward_fn

# Set device exactly once here. All helpers receive this device explicitly.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 102
NUM_STEPS = 10
GROUP_SIZE = 8
PROMPTS_PER_STEP = 2
GRADIENT_ACCUMULATION_STEPS = 16
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 5
TRAIN_PATH = "dataset_03/gsmk_train.jsonl"
CHECKPOINT_DIR = "checkpoints"
LOG_PATH = "training_metrics.csv"
RESUME_FROM = None

VAL_PATH = "dataset_03/gsmk_test.jsonl"
VAL_SUBSET_SIZE = 8
VAL_LOG = "validation_metrics.csv"


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if RESUME_FROM:
        model, tokenizer, optimizer, start_step = load_checkpoint(RESUME_FROM, DEVICE, LEARNING_RATE)
    else:
        model, tokenizer = load_model(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      betas=(0.9, 0.95), weight_decay=0.0)
        start_step = 0

    examples = load_dataset(TRAIN_PATH, limit=6400)
    if len(examples) < PROMPTS_PER_STEP:
        raise ValueError('Not enough training examples')
    if (PROMPTS_PER_STEP * GROUP_SIZE) % GRADIENT_ACCUMULATION_STEPS:
        raise ValueError('Batch size must be divisible by accumulation steps')

    val_set = load_dataset(VAL_PATH)
    if not val_set:
        raise ValueError('Validation dataset is empty')
    rng = random.Random(SEED)
    val_indices = rng.sample(range(len(val_set)), min(VAL_SUBSET_SIZE, len(val_set)))
    val_subset = [val_set[i] for i in val_indices]
    val_prompts = [make_prompt(ex['question']) for ex in val_subset]
    val_truths = [get_ground_truth(ex) for ex in val_subset]

    if start_step == 0 or not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'loss', 'reward', 'format_reward',
                                    'grad_norm', 'nonconstant_group_fraction'])

    if start_step == 0 or not os.path.exists(VAL_LOG):
        with open(VAL_LOG, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'accuracy', 'reward', 'format_reward', 'num_examples'])

    print(f'Device: {DEVICE}; total steps: {NUM_STEPS}; start: {start_step}')
    for step in range(start_step, NUM_STEPS):
        selected = random.sample(examples, PROMPTS_PER_STEP)
        prompts = [make_prompt(ex['question']) for ex in selected]
        truths = [get_ground_truth(ex) for ex in selected]
        _, responses, repeated_truths, prompt_ids, response_ids = generate_rollouts(
            model, tokenizer, prompts, truths, GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE, DEVICE)

        loss, metrics = grpo_train_step(
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM, reward_fn=r1_zero_reward_fn,
            rollout_responses=responses, repeated_ground_truths=repeated_truths,
            prompt_token_ids=prompt_ids, response_token_ids=response_ids,
            group_size=GROUP_SIZE, device=DEVICE,
            use_bf16=(DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()),
        )
        completed = step + 1
        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([
                completed, loss, metrics['mean_reward'], metrics['format_reward'],
                metrics['grad_norm'], metrics['nonconstant_group_fraction'],
            ])
        print(f"Step {completed}/{NUM_STEPS} | loss {loss:.4f} | "
              f"reward {metrics['mean_reward']:.3f} | "
              f"format {metrics['format_reward']:.3f} | "
              f"grad_norm {metrics['grad_norm']:.3f} | "
              f"informative_groups {metrics['nonconstant_group_fraction']:.2%}")
        print('Example response:', responses[0][:1000])

        if completed % SAVE_EVERY == 0:
            print(f'Running validation at step {completed}...')

            _, val_responses, val_repeated_truths, _, _ = generate_rollouts(
                model, tokenizer, val_prompts, val_truths, 1, MAX_NEW_TOKENS, TEMPERATURE, DEVICE)

            val_metrics = grpo_val_step(
                model=model, reward_fn=r1_zero_reward_fn,
                rollout_responses=val_responses, repeated_ground_truths=val_repeated_truths,
            )

            with open(VAL_LOG, 'a', newline='') as f:
                csv.writer(f).writerow([
                    completed, val_metrics['accuracy'], val_metrics['mean_reward'],
                    val_metrics['format_reward'], val_metrics['num_examples'],
                ])

            print(f"Validation step {completed} | "
                  f"accuracy {val_metrics['accuracy']:.3f} | "
                  f"reward {val_metrics['mean_reward']:.3f} | "
                  f"format {val_metrics['format_reward']:.3f}")

            save_checkpoint(model, tokenizer, optimizer, completed, CHECKPOINT_DIR)

    final_dir = os.path.join(CHECKPOINT_DIR, 'final_model')
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print('Final model saved:', final_dir)


if __name__ == '__main__':
    main()