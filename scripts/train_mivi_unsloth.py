#!/usr/bin/env python3
"""
MIVI-V2 Sub-1B Unsloth QLoRA Fine-Tuning Script
Fine-tunes the MIVI default model (openbmb/MiniCPM5-1B; also works for
Qwen2.5/Qwen3 small models) on Colab Free Tier (T4 GPU).
Memory target: < 2.5 GB VRAM during training | < 700 MB RAM during GGUF inference.
Dataset: scripts/build_agentic_sft.py output (OpenAI messages format).
"""

import os
import sys
import json
import argparse

def train(
    base_model: str = "openbmb/MiniCPM5-1B",
    dataset_path: str = "datasets/mivi_agentic_sft.jsonl",
    output_dir: str = "outputs/mivi-minicpm5-agent",
    max_seq_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    # 175-row dataset / effective batch 16 = ~11 steps per epoch.
    # 60 steps ~= 5 epochs — enough for LoRA to absorb the patterns without
    # memorizing; raise only if the agent eval plateaus early.
    max_steps: int = 60,
    export_gguf: bool = True
):
    print("=" * 60)
    print("🚀 MIVI-V2 Unsloth QLoRA Fine-Tuning Pipeline")
    print(f"Base Model: {base_model}")
    print(f"Dataset:    {dataset_path}")
    print(f"LoRA Rank:  {lora_rank} (Alpha: {lora_alpha})")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import load_dataset, Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("❌ Unsloth/Transformers dependencies not found.")
        print("To run on Google Colab, install with:")
        print("pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("pip install --no-deps trl peft accelerate bitsandbytes")
        sys.exit(1)

    # 1. Load Base Model with 4-bit Quantization
    print("📥 Loading base model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,
        max_seq_length = max_seq_length,
        dtype = None, # Auto-detect (Float16 for T4)
        load_in_4bit = True,
    )

    # 2. Configure PEFT / LoRA Adapters
    print("🔧 Configuring LoRA adapters for all linear projection modules...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Optimized by Unsloth
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Load & Format Dataset
    print(f"📄 Ingesting training dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    def normalize_for_template(messages):
        """Chat templates expect arguments as a dict; the dataset stores the
        OpenAI wire format (JSON string). Convert before apply_chat_template."""
        out = []
        for m in messages:
            if "tool_calls" in m:
                m = dict(m)
                m["tool_calls"] = [
                    dict(tc, function=dict(tc["function"],
                         arguments=json.loads(tc["function"]["arguments"])))
                    for tc in m["tool_calls"]
                ]
            out.append(m)
        return out

    formatted_texts = []
    for item in raw_data:
        if "prompt" in item and "completion" in item:
            # Serving-format rows (round 2): byte-exact rendered prompt,
            # completion-style training — no chat template.
            formatted_texts.append({"text": item["prompt"] + item["completion"]})
            continue
        messages = normalize_for_template(item["messages"])
        # Apply standard ChatML template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append({"text": text})

    dataset = Dataset.from_list(formatted_texts)
    print(f"✅ Loaded {len(dataset)} training samples.")

    # 4. Set up SFTTrainer
    print("⚡ Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = 10,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none",
        ),
    )

    # 5. Train Model
    print("🚀 Starting fine-tuning session...")
    trainer_stats = trainer.train()
    print(f"✅ Training completed in {trainer_stats.metrics.get('train_runtime', 0):.2f} seconds.")

    # 6. Save LoRA Adapters
    print(f"💾 Saving LoRA adapter checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Export to GGUF (Q4_K_M)
    if export_gguf:
        print("📦 Exporting merged model to GGUF (Q4_K_M format)...")
        gguf_output = f"{output_dir}/mivi-0.5b-tool-q4_k_m"
        model.save_pretrained_gguf(
            gguf_output,
            tokenizer,
            quantization_method = "q4_k_m"
        )
        print(f"🎉 Successfully exported GGUF model: {gguf_output}-unsloth.Q4_K_M.gguf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIVI-V2 Sub-1B Unsloth Fine-Tuning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model identifier")
    parser.add_argument("--dataset", type=str, default="datasets/mivi_sub1b_tuning_dataset.jsonl", help="Dataset path")
    parser.add_argument("--output", type=str, default="outputs/mivi-0.5b-tool-expert", help="Output directory")
    parser.add_argument("--steps", type=int, default=250, help="Maximum training steps")
    parser.add_argument("--no-gguf", action="store_true", help="Skip GGUF export")
    
    args = parser.parse_args()
    train(
        base_model=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_steps=args.steps,
        export_gguf=not args.no_gguf
    )
