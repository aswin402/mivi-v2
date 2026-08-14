#!/usr/bin/env python3
"""
MIVI-V2 Multi-LoRA Fine-Tuning Pipeline
Trains lightweight LoRA specialist adapters (< 20 MB each) for Qwen2.5-0.5B:
1. reasoner  — outputs/loras/mivi-reasoner
2. tools     — outputs/loras/mivi-tools
3. coder     — outputs/loras/mivi-coder
4. debugger  — outputs/loras/mivi-debugger
"""

import argparse
import json
import os
import sys

SPECIALIST_DATASETS = {
    "reasoner": "datasets/mivi_reasoner_dataset.jsonl",
    "tools": "datasets/mivi_tools_dataset.jsonl",
    "coder": "datasets/mivi_coder_dataset.jsonl",
    "debugger": "datasets/mivi_debugger_dataset.jsonl",
}

def train_specialist(
    specialist: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_base_dir: str = "outputs/loras",
    max_seq_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = 250,
):
    if specialist not in SPECIALIST_DATASETS:
        raise ValueError(f"Unknown specialist '{specialist}'. Available: {list(SPECIALIST_DATASETS.keys())}")

    dataset_path = SPECIALIST_DATASETS[specialist]
    output_dir = os.path.join(output_base_dir, f"mivi-{specialist}")

    print("=" * 60)
    print(f"🚀 Training MIVI Specialist LoRA: '{specialist.upper()}'")
    print(f"Base Model:  {base_model}")
    print(f"Dataset:     {dataset_path}")
    print(f"Output Dir:  {output_dir}")
    print(f"LoRA Rank:   {lora_rank} (Alpha: {lora_alpha})")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("❌ Unsloth/Transformers dependencies not found.")
        print("Run on Google Colab or install with:")
        print("pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("pip install trl peft accelerate bitsandbytes datasets transformers")
        sys.exit(1)

    # 1. Load Base Model in 4-bit
    print("📥 Loading base model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Add LoRA Adapters
    print(f"🔧 Attaching LoRA adapter matrix for {specialist}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load & Ingest Dataset
    print(f"📄 Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    print(f"✅ Loaded {len(dataset)} training examples.")

    # 4. Trainer Configuration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=max_steps,
        ),
    )

    # 5. Train
    print(f"🔥 Training LoRA adapter '{specialist}'...")
    trainer.train()

    # 6. Save Adapter Only (< 25 MB)
    print(f"💾 Saving LoRA adapter checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 Successfully trained and saved specialist LoRA: mivi-{specialist} (< 25 MB)!")

def main():
    parser = argparse.ArgumentParser(description="MIVI-V2 Multi-LoRA Specialist Trainer")
    parser.add_argument(
        "--specialist",
        type=str,
        default="all",
        choices=["reasoner", "tools", "coder", "debugger", "all"],
        help="Specialist persona to train"
    )
    parser.add_argument("--steps", type=int, default=250, help="Max training steps per adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    args = parser.parse_args()

    specialists = ["reasoner", "tools", "coder", "debugger"] if args.specialist == "all" else [args.specialist]
    for s in specialists:
        train_specialist(specialist=s, base_model=args.base_model, max_steps=args.steps)

if __name__ == "__main__":
    main()
