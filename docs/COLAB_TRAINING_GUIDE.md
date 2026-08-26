# Google Colab Fine-Tuning Guide — MIVI-V2 (Round 2: Serving-Format SFT)

> **Platform:** Google Colab (Free T4 GPU, 16 GB VRAM)  
> **Estimated Training Time:** ~1–2 minutes per model (fast execution)  
> **Target Models:**  
> 1. `openbmb/MiniCPM5-1B` (Dense 1.08B, Apache 2.0, hybrid `<think>`)  
> 2. `LiquidAI/LFM2.5-350M` (Hybrid 350M, 438 MB inference RAM, 88 tok/s)  
> **Dataset:** `datasets/mivi_serving_sft.jsonl` (Round 2 byte-exact serving prompts & grammar-exact tool completions)  
> **Output:** `mivi-minicpm5-r2` & `mivi-lfm350-r2` GGUF models (`Q4_K_M`)

---

## 🎯 Step 1: Open Google Colab & Enable T4 GPU

1. Open **[Google Colab](https://colab.research.google.com/)** in your browser.
2. Create a new notebook (or open `notebooks/train_agentic_colab.ipynb`).
3. In Colab's top menu bar, click **Runtime** $\rightarrow$ **Change runtime type**.
4. Set **Hardware accelerator** to **T4 GPU** and click **Save**.

---

## 📋 Step 2: Paste and Run Cells Step-by-Step

### 🔹 Cell 1: Fast Install with `uv` (~30 seconds)
```python
%%capture
# 1. Install Astral uv package manager
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ["PATH"]

# 2. Install Unsloth and ML dependencies
!uv pip install --system unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers sentencepiece protobuf
print("✅ Environment ready via uv!")
```

---

### 🔹 Cell 2: Clone Repository & Build Dataset (~10 seconds)
```python
import os, pathlib, subprocess

REPO = 'https://github.com/aswin402/mivi-v2.git'
BRANCH = 'feat/verifier-sandbox'

if not pathlib.Path('mivi-v2').exists():
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO], check=True)
os.chdir('/content/mivi-v2')
subprocess.run(['git', 'pull', 'origin', BRANCH], check=True)

# Build the Round 2 serving SFT dataset
subprocess.run(['python3', 'scripts/build_serving_sft.py', '--out', 'datasets/mivi_serving_sft.jsonl'], check=True)
DATASET = 'datasets/mivi_serving_sft.jsonl'
print('✅ Dataset ready:', DATASET, '| rows:', sum(1 for _ in open(DATASET)))
```

---

### 🔹 Cell 3: Direct In-Kernel Trainer Function
```python
import os, json, glob, torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import files

def train_and_export(
    model_name="openbmb/MiniCPM5-1B",
    dataset_path="datasets/mivi_serving_sft.jsonl",
    output_dir="outputs/mivi-minicpm5-r2",
    max_steps=30,
    batch_size=2,
    grad_accum=2,
    lr=2e-4
):
    print("=" * 60)
    print(f"🚀 Training: {model_name}")
    print(f"📁 Output:   {output_dir}")
    print(f"⚡ GPU:      {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    print("=" * 60)

    # 1. Load base model in 4-bit with 4k context
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Add LoRA adapters
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

    # 3. Load serving-format dataset (prompt + completion)
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    formatted = []
    for item in raw:
        if "prompt" in item and "completion" in item:
            formatted.append({"text": item["prompt"] + item["completion"]})
        elif "messages" in item:
            formatted.append({"text": tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)})

    dataset = Dataset.from_list(formatted)
    print(f"📄 Training samples loaded: {len(dataset)}")

    # 4. Trainer with full GPU utilization
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    print("🔥 Starting training loop...")
    stats = trainer.train()
    print(f"✅ Training finished in {stats.metrics.get('train_runtime', 0):.2f}s!")

    # 5. Export merged GGUF
    print(f"📦 Exporting merged model to Q4_K_M GGUF...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(f"🎉 GGUF export complete in: {output_dir}")
```

---

### 🔹 Cell 4: Train MiniCPM5-1B (~1–2 minutes)
```python
train_and_export(
    model_name="openbmb/MiniCPM5-1B",
    dataset_path="datasets/mivi_serving_sft.jsonl",
    output_dir="outputs/mivi-minicpm5-r2",
    max_steps=30
)
```

---

### 🔹 Cell 5: Train LFM2.5-350M (~1 minute)
```python
train_and_export(
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_serving_sft.jsonl",
    output_dir="outputs/mivi-lfm350-r2",
    max_steps=25
)
```

---

### 🔹 Cell 6: Download the Exported GGUFs
```python
import glob
from google.colab import files

ggufs = glob.glob('/content/mivi-v2/outputs/**/*.gguf', recursive=True)
print('📦 Found GGUFs:', ggufs)
for g in ggufs:
    print(f'⬇️ Downloading {g}...')
    files.download(g)
```

---

## 💻 Step 3: Local Setup & Evaluation (After Download)

### 1. Move Downloaded Models into `models/`
```bash
cp ~/Downloads/*minicpm5*.gguf models/
cp ~/Downloads/*lfm350*.gguf models/
```

### 2. Benchmark Models Against 11 Agent Workflows
```bash
# Test MiniCPM5-1B Round 2
MIVI_RUNTIME_MODE=worker-eco \
MIVI_REASONER_MODEL=models/mivi-minicpm5-r2.Q4_K_M.gguf \
MIVI_CODER_MODEL=models/mivi-minicpm5-r2.Q4_K_M.gguf \
python3 scripts/eval_agent_workflows.py

# Test LFM2.5-350M Round 2
MIVI_RUNTIME_MODE=worker-eco \
MIVI_REASONER_MODEL=models/mivi-lfm350-r2.Q4_K_M.gguf \
MIVI_CODER_MODEL=models/mivi-lfm350-r2.Q4_K_M.gguf \
python3 scripts/eval_agent_workflows.py
```

### 3. Success Criteria:
- **Baseline to beat:** Qwen3-1.7B default = **7/11**.
- **Round-2 Target:** **$\ge$ 9/11**.
- The highest scoring base will become the MoE foundation model for Phase 17 MixLoRA specialists.
