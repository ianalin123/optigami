"""
Origami GRPO Training Script

Usage (Colab with T4/A100):
    python trainer/train.py

Or in a notebook:
    %run trainer/train.py

Requires: unsloth, trl>=0.22.2, transformers>=4.56.2, trackio, datasets
"""

import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainer.prompts import build_prompt, SYSTEM_PROMPT, get_task_target_ratio, get_task_max_folds
from trainer.rewards import code_valid, physically_valid, fold_quality, set_task_config

try:
    from engine.materials import get_material
    Material = type(get_material("paper"))  # get the Material class
except ImportError:
    from trainer.mock_env import Material
    def get_material(name):
        return Material()

# ============================================================================
# Config
# ============================================================================

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 4

# Start with the simplest task
TASK_NAME = "half_fold"

# GRPO hyperparameters (from 2048 reference, adapted for origami)
LEARNING_RATE = 2e-4
MAX_STEPS = 600
NUM_GENERATIONS = 2
TEMPERATURE = 1.0
BATCH_SIZE = 1
GRAD_ACCUM = 1
DATASET_SIZE = 1000  # replicated prompt dataset


def main():
    # ========================================================================
    # 1. Load model with Unsloth
    # ========================================================================
    from unsloth import FastLanguageModel

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ========================================================================
    # 2. Apply LoRA adapters
    # ========================================================================
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ========================================================================
    # 3. Build prompt and dataset
    # ========================================================================
    user_prompt = build_prompt(TASK_NAME)
    target_ratio = get_task_target_ratio(TASK_NAME)
    max_folds = get_task_max_folds(TASK_NAME)

    # Configure reward functions with task parameters
    set_task_config(
        width=1.0,
        height=1.0,
        material=get_material("paper"),
        target_ratio=target_ratio,
        max_folds=max_folds,
    )

    # Create replicated prompt dataset (same pattern as 2048)
    from datasets import Dataset

    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
    ] * DATASET_SIZE)

    # Calculate prompt token length for max_completion_length
    prompt_tokens = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
    )
    max_prompt_length = len(prompt_tokens) + 1
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
    print(f"Prompt tokens: {max_prompt_length}, Max completion: {max_completion_length}")

    # ========================================================================
    # 4. Test inference before training
    # ========================================================================
    print("\n=== Pre-training inference test ===")
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=TEMPERATURE,
        max_new_tokens=min(512, max_completion_length),
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    # ========================================================================
    # 5. Configure GRPO training
    # ========================================================================
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=TEMPERATURE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=MAX_STEPS,
        save_steps=100,
        report_to="trackio",
        output_dir="outputs",
    )

    # ========================================================================
    # 6. Create trainer and start training
    # ========================================================================
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            code_valid,          # Reward 1: valid Python?
            physically_valid,    # Reward 2: physically possible folds?
            fold_quality,        # Reward 3: how good is the solution?
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\n=== Starting GRPO training: {TASK_NAME} ===")
    print(f"Steps: {MAX_STEPS}, Generations: {NUM_GENERATIONS}, LR: {LEARNING_RATE}")
    trainer.train()

    # ========================================================================
    # 7. Post-training inference
    # ========================================================================
    print("\n=== Post-training inference ===")
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=TEMPERATURE,
        max_new_tokens=min(1024, max_completion_length),
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    # ========================================================================
    # 8. Save model (optional)
    # ========================================================================
    save_path = "outputs/origami-fold-lora"
    print(f"\nSaving LoRA adapter to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")


if __name__ == "__main__":
    main()
