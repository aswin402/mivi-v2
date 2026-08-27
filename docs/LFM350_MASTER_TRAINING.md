# 🚀 MIVI Turbo Master Base Model — Colab Training Guide (High-VRAM V3)

> **Model Target:** `LiquidAI/LFM2.5-350M`
> **Platform:** Google Colab (Free Tier - T4 GPU, 15 GB VRAM)
> **Dataset:** **20,000 samples** across 10 agentic categories (XML `<tool_call>` format)
> **GPU Acceleration:** `batch_size=32`, `grad_accum=2` (Effective Batch = 64)
> **Loss Masking:** Unsloth Response-Only Masking (`train_on_responses_only`)
> **Training Time:** ~3.5–5 minutes (600 optimizer steps, 38,400 sample views)
> **Output:** `mivi-lfm350-master` GGUF (`Q4_K_M`, ~229 MB)

---

## 📋 Step 1: Set Runtime to T4 GPU
1. Go to **Runtime** → **Change runtime type**
2. Select **T4 GPU** → Click **Save**

---

## 📋 Step 2: Copy-Paste These 5 Cells (Run in Order)

### 🔹 Cell 1: Install Dependencies (~30 seconds)
```python
%%capture
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ["PATH"]
!uv pip install --system unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers sentencepiece protobuf
print("✅ Environment ready via uv!")
```

---

### 🔹 Cell 2: Clone Repo & Build 20,000-Sample Master Dataset (~4 seconds)
```python
import os, pathlib, subprocess

REPO = 'https://github.com/aswin402/mivi-v2.git'
BRANCH = 'feat/verifier-sandbox'

if not pathlib.Path('mivi-v2').exists():
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO], check=True)
os.chdir('/content/mivi-v2')
subprocess.run(['git', 'pull', 'origin', BRANCH], check=True)

# Build 20,000-sample Master Dataset (10 categories, XML <tool_call> format)
subprocess.run([
    'python3', 'scripts/build_15k_agentic_dataset.py',
    '--total', '20000', '--fast',
    '--out', 'datasets/mivi_master_15k_sft.jsonl'
], check=True)
DATASET = 'datasets/mivi_master_15k_sft.jsonl'
print('✅ Master Dataset ready:', DATASET, '| rows:', sum(1 for _ in open(DATASET)))
```

---

### 🔹 Cell 3: Define High-Throughput Training Function
```python
import os, json, glob, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import files

def train_and_export_master(
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_master_15k_sft.jsonl",
    output_dir="outputs/mivi-lfm350-master",
    max_steps=600,
    batch_size=32,       # ⚡ 4x larger batch: fills GPU VRAM, 5x faster
    grad_accum=2,        # ⚡ Effective batch = 64
    lr=2.5e-4
):
    print("=" * 60)
    print(f"🚀 MIVI Turbo Master Training (High-VRAM Mode)")
    print(f"📁 Model:       {model_name}")
    print(f"📦 Dataset:     {dataset_path}")
    print(f"⚡ GPU:         {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"🔄 Steps:       {max_steps} (Effective Batch = {batch_size * grad_accum})")
    print(f"📊 Exposures:   {max_steps * batch_size * grad_accum} sample views")
    print("=" * 60)

    # 1. Load base model in 4-bit with 512 context
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Add LoRA adapters to all linear projections
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    formatted = []
    for item in raw:
        if "prompt" in item and "completion" in item:
            formatted.append({"text": item["prompt"] + item["completion"] + "<|im_end|>\n"})

    dataset = Dataset.from_list(formatted)
    print(f"📄 Loaded {len(dataset)} training samples")

    # 4. High-Throughput SFT Trainer (Batch 32)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,     # ⚡ Batch 32
            gradient_accumulation_steps=grad_accum,    # ⚡ Grad Accum 2
            warmup_steps=20,
            max_steps=max_steps,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    # 5. Response-Only Loss Masking
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("🔥 Training started with GPU acceleration (~3.5–5 minutes)...")
    stats = trainer.train()
    print(f"✅ Training done in {stats.metrics.get('train_runtime', 0):.1f}s!")

    # 6. Export merged GGUF
    print(f"📦 Exporting Q4_K_M GGUF...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(f"🎉 GGUF export complete: {output_dir}")
```

---

### 🔹 Cell 4: Train! (~3.5–5 minutes)
```python
train_and_export_master(
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_master_15k_sft.jsonl",
    output_dir="outputs/mivi-lfm350-master",
    max_steps=600,        # 600 steps × 64 = 38,400 sample views (~1.92 epochs)
    batch_size=32,        # ⚡ Uses ~6-8 GB VRAM, 5x faster
    grad_accum=2,
    lr=2.5e-4
)
```

---

### 🔹 Cell 5: Download the GGUF
```python
import glob, os
from google.colab import files

master_dir = '/content/mivi-v2/outputs/mivi-lfm350-master'
ggufs = glob.glob(f'{master_dir}/**/*.gguf', recursive=True)

if not ggufs:
    all_ggufs = glob.glob('/content/mivi-v2/outputs/**/*.gguf', recursive=True)
    if all_ggufs:
        all_ggufs.sort(key=os.path.getmtime, reverse=True)
        ggufs = [all_ggufs[0]]

print('📦 GGUF files:', ggufs)
for g in ggufs:
    print(f'⬇️ Downloading {g}...')
    files.download(g)
```

---

## 📋 Step 3: Local Benchmarking (After Download)

```bash
# 1. Copy downloaded model
cp ~/Downloads/*LFM2.5-350M*.gguf models/new/LFM2.5-350M.Q4_K_M.gguf

# 2. Run 26-test deep evaluation
python3 scripts/deep_model_test.py

# 3. Run 11-agentic-workflow benchmark
python3 scripts/run_11_eval.py
```
