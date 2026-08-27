#!/usr/bin/env python3
"""
MIVI-V2 Unsloth QLoRA fine-tuning entry point.

This script is used by the high-VRAM LFM2.5 master run as well as the smaller
serving-format experiments. It deliberately fails fast without CUDA: a CPU
fallback can make a long Colab job look hung for hours.
"""

import os
import sys
import json
import argparse
from typing import Optional

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
    max_steps: int = 60,
    export_gguf: bool = True,
    dataset_num_proc: int = 2,
    dataloader_num_workers: int = 2,
    gradient_checkpointing: bool = True,
    save_steps: int = 100,
    resume_from_checkpoint: Optional[str] = None,
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
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:
        print("❌ Unsloth/Transformers dependencies not found.")
        print("To run on Google Colab, install with:")
        print("pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("pip install --no-deps trl peft accelerate bitsandbytes")
        sys.exit(1)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. This training path is intentionally GPU-only; "
            "check Colab Runtime > Change runtime type > T4/L4/A100 GPU."
        )

    gpu = torch.cuda.get_device_properties(0)
    effective_batch = batch_size * gradient_accumulation_steps
    print(f"🖥️ GPU: {gpu.name} ({gpu.total_memory / 2**30:.1f} GiB)")
    print(f"📈 Effective batch: {effective_batch} samples")
    print(f"📏 Token positions/step before packing: {effective_batch * max_seq_length:,}")
    print(f"🔁 Steps: {max_steps} | checkpoint every {save_steps} steps")
    if gpu.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

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
        use_gradient_checkpointing = "unsloth" if gradient_checkpointing else False,
        random_state = 3407,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

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
        dataset_num_proc = dataset_num_proc,
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
            dataloader_num_workers = dataloader_num_workers,
            dataloader_pin_memory = True,
            gradient_checkpointing = gradient_checkpointing,
            save_strategy = "steps",
            save_steps = save_steps,
            save_total_limit = 2,
        ),
    )

    # Do not spend adapter capacity learning the user prompt/tool inventory.
    if any("<|im_start|>user" in row["text"] for row in formatted_texts):
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )
        print("🎯 Response-only loss masking enabled.")

    # 5. Train Model
    print("🚀 Starting fine-tuning session...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"✅ Training completed in {trainer_stats.metrics.get('train_runtime', 0):.2f} seconds.")

    # 6. Save LoRA Adapters
    print(f"💾 Saving LoRA adapter checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Export to GGUF (Q4_K_M)
    if export_gguf:
        print("📦 Exporting merged model to GGUF (Q4_K_M format)...")
        gguf_output = f"{output_dir}/mivi-master-q4_k_m"
        model.save_pretrained_gguf(
            gguf_output,
            tokenizer,
            quantization_method = "q4_k_m"
        )
        print(f"🎉 Successfully exported GGUF model: {gguf_output}-unsloth.Q4_K_M.gguf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIVI-V2 Sub-1B Unsloth Fine-Tuning")
    parser.add_argument("--model", "--base_model", dest="model", type=str, default="openbmb/MiniCPM5-1B", help="Base model identifier")
    parser.add_argument("--dataset", "--dataset_path", dest="dataset", type=str, default="datasets/mivi_serving_sft.jsonl", help="Dataset path")
    parser.add_argument("--output", "--output_dir", dest="output", type=str, default="outputs/mivi-minicpm5-r2", help="Output directory")
    parser.add_argument("--steps", "--max_steps", dest="steps", type=int, default=25, help="Maximum training steps")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset-procs", type=int, default=2, help="Dataset preprocessing workers")
    parser.add_argument("--loader-workers", type=int, default=2, help="Training dataloader workers")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", help="Save VRAM at the cost of speed")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", help="Use the faster high-VRAM path")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--export_gguf", type=str, default="1", help="Export GGUF (1/0/true/false)")
    parser.add_argument("--no-gguf", action="store_true", help="Skip GGUF export")
    
    args = parser.parse_args()
    export_gguf = not args.no_gguf and args.export_gguf.lower() in ("1", "true", "yes")
    train(
        base_model=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.steps,
        dataset_num_proc=args.dataset_procs,
        dataloader_num_workers=args.loader_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
        export_gguf=export_gguf
    )
