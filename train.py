"""
OrigamiRL — GRPO Training Script
Code-as-policy: model generates complete fold sequence, gets terminal reward.

Usage:
    python train.py
    python train.py --model unsloth/Qwen2.5-7B-Instruct --epochs 3 --output origami-grpo
"""
import argparse
import json
import copy
import random
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unsloth/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--n_generations', type=int, default=8)
    parser.add_argument('--max_folds', type=int, default=8)
    parser.add_argument('--output', default='origami-grpo')
    parser.add_argument('--level', type=int, default=1, help='Target difficulty level (1-3)')
    parser.add_argument('--dry_run', action='store_true', help='Test reward function without training')
    return parser.parse_args()


def build_dataset(env, level: int = 1, max_folds: int = 8) -> list[dict]:
    """
    Build a training dataset of prompts from available targets.
    Each item: {'prompt': str, 'target_name': str}
    Repeats each target multiple times to give enough training steps.
    """
    all_names = env.available_targets()

    # Filter by level; fall back to all targets if none match
    level_names = [
        name for name in all_names
        if env._targets[name].get('level', 1) == level
    ]
    if not level_names:
        level_names = all_names

    items = []
    for name in level_names:
        obs = env.reset(target_name=name)
        prompt = obs['prompt']
        items.append({'prompt': prompt, 'target_name': name})

    # Repeat each target 10x; ensure at least 50 examples
    repeat = max(10, (50 + len(items) - 1) // len(items))
    items = items * repeat

    random.shuffle(items)
    return items


def make_reward_fn(env_template, max_folds: int):
    """
    Returns a reward function compatible with trl GRPOTrainer.

    Signature: reward_fn(completions, prompts=None, **kwargs) -> list[float]

    For each completion:
    1. Clone the environment (fresh paper state)
    2. Reset to the target embedded in the prompt (use target_name from kwargs if available)
    3. Execute the completion as a fold sequence
    4. Return the total reward
    """
    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        target_names = kwargs.get('target_names', [None] * len(completions))

        for completion, target_name in zip(completions, target_names):
            try:
                env = env_template.clone()
                env.reset(target_name=target_name)
                _, reward_dict, _, _ = env.step(completion)
                rewards.append(float(reward_dict['total']))
            except Exception:
                rewards.append(-0.1)

        return rewards

    return reward_fn


def make_detailed_reward_fns(env_template, max_folds: int) -> list:
    """
    Returns a list of reward functions, one per reward component.
    Used for detailed W&B logging of each component separately.
    Components: kawasaki, maekawa, blb, progress, economy, completion
    """
    components = ['kawasaki', 'maekawa', 'blb', 'progress', 'economy', 'completion']

    def make_component_fn(component: str):
        def component_fn(completions, prompts=None, **kwargs):
            rewards = []
            target_names = kwargs.get('target_names', [None] * len(completions))

            for completion, target_name in zip(completions, target_names):
                try:
                    env = env_template.clone()
                    env.reset(target_name=target_name)
                    _, reward_dict, _, _ = env.step(completion)
                    rewards.append(float(reward_dict.get(component, 0.0)))
                except Exception:
                    rewards.append(0.0)

            return rewards

        component_fn.__name__ = f'reward_{component}'
        return component_fn

    return [make_component_fn(c) for c in components]


def main():
    args = parse_args()

    # Import here to allow dry_run without GPU
    from env.environment import OrigamiEnvironment

    env = OrigamiEnvironment(mode='code_as_policy', max_steps=args.max_folds)

    # Build dataset
    dataset_items = build_dataset(env, level=args.level, max_folds=args.max_folds)
    print(f"Dataset: {len(dataset_items)} examples from level-{args.level} targets")
    print(f"Targets: {env.available_targets()}")

    # Dry run: test reward function without loading model
    if args.dry_run:
        reward_fn = make_reward_fn(env, args.max_folds)
        test_completions = [
            '<folds>[{"instruction": "Valley fold along horizontal center", "from": [0, 0.5], "to": [1, 0.5], "assignment": "V"}]</folds>',
            '<folds>[{"instruction": "Invalid fold", "from": [0.3, 0.3], "to": [0.7, 0.7], "assignment": "V"}]</folds>',
            'this is not valid JSON at all',
        ]
        target_names = [dataset_items[0]['target_name']] * 3
        rewards = reward_fn(test_completions, target_names=target_names)
        print(f"\nDry run rewards: {rewards}")
        print("Dry run complete — reward function works.")
        return

    # Load model via unsloth
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run: pip install unsloth")
        print("Or run with --dry_run to test the reward function without a model.")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    # Convert dataset to HuggingFace Dataset format
    from datasets import Dataset

    # GRPOTrainer expects 'prompt' column and optionally others.
    # We embed target_name in the dataset so the reward fn can use it.
    hf_dataset = Dataset.from_list(dataset_items)

    # Build main reward function
    reward_fn = make_reward_fn(env, args.max_folds)

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=512,
        num_generations=args.n_generations,
        temperature=1.0,
        logging_steps=1,
        report_to="wandb",
        run_name="origami-grpo",
    )

    # GRPOTrainer passes all dataset columns as kwargs to reward_funcs.
    # The 'target_name' column arrives as a list (one per completion in the batch).
    def wrapped_reward_fn(completions, target_name=None, **kwargs):
        """Wrapper that extracts target_name from batch columns."""
        target_names = target_name if isinstance(target_name, list) else [target_name] * len(completions)
        return reward_fn(completions, target_names=target_names)

    trainer = GRPOTrainer(
        model=model,
        config=config,
        train_dataset=hf_dataset,
        reward_funcs=[wrapped_reward_fn],
        tokenizer=tokenizer,
    )

    print(f"\nStarting GRPO training...")
    print(f"  Model: {args.model}")
    print(f"  Level: {args.level} targets")
    print(f"  Epochs: {args.epochs}")
    print(f"  Generations per prompt: {args.n_generations}")
    print(f"  Output: {args.output}/")

    trainer.train()

    # Save
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}/")


if __name__ == '__main__':
    main()
