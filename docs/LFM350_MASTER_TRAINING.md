# LFM2.5-350M Master Agentic Training Guide (Google Colab)

> **Target Model:** `LiquidAI/LFM2.5-350M`  
> **Platform:** Google Colab Free Tier (T4 GPU, 16 GB VRAM)  
> **Dataset:** `datasets/mivi_lfm_serving_master.jsonl` (280 balanced samples across 6 agentic pillars)  
> **Training Time:** ~1.5–2 minutes (60 steps)  
> **Output:** `mivi-lfm350-master` GGUF (`Q4_K_M`, ~229 MB)

---

## 🚀 Step 1: Set Runtime in Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. In the top menu, go to **Runtime** $\rightarrow$ **Change runtime type**.
3. Select **T4 GPU** under Hardware accelerator and click **Save**.

---

## 📋 Step 2: Colab Notebook Cells (Run in Order)

### 🔹 Cell 1: Super-Fast Install with `uv` (~30 seconds)
```python
%%capture
# 1. Install Astral uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:" + os.environ["PATH"]

# 2. Fast install Unsloth, Unsloth Zoo, and ML dependencies
!uv pip install --system unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers sentencepiece protobuf
print("✅ Environment ready via uv!")
```

---

### 🔹 Cell 2: Clone Repo & Build Master Dataset (~10 seconds)
```python
import os, pathlib, subprocess

REPO = 'https://github.com/aswin402/mivi-v2.git'
BRANCH = 'feat/verifier-sandbox'

if not pathlib.Path('mivi-v2').exists():
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO], check=True)
os.chdir('/content/mivi-v2')
subprocess.run(['git', 'pull', 'origin', BRANCH], check=True)

# Build Master SFT Dataset (280 balanced samples across all 6 agentic pillars)
subprocess.run(['python3', 'scripts/generate_agentic_lfm_dataset.py'], check=True)
DATASET = 'datasets/mivi_lfm_serving_master.jsonl'
print('✅ Master Dataset ready:', DATASET, '| rows:', sum(1 for _ in open(DATASET)))
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
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_lfm_serving_master.jsonl",
    output_dir="outputs/mivi-lfm350-master",
    max_steps=60,
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

### 🔹 Cell 4: Run Training (~2 minutes)
```python
train_and_export(
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_lfm_serving_master.jsonl",
    output_dir="outputs/mivi-lfm350-master",
    max_steps=60
)
```

---

### 🔹 Cell 5: Download Exported Model to Your Computer
```python
import glob, os
from google.colab import files

# Specifically target the new master model
master_dir = '/content/mivi-v2/outputs/mivi-lfm350-master'
ggufs = glob.glob(f'{master_dir}/**/*.gguf', recursive=True)

if not ggufs:
    # Fallback: get the single newest .gguf across outputs
    all_ggufs = glob.glob('/content/mivi-v2/outputs/**/*.gguf', recursive=True)
    if all_ggufs:
        all_ggufs.sort(key=os.path.getmtime, reverse=True)
        ggufs = [all_ggufs[0]]

print('📦 Target Master GGUF:', ggufs)
for g in ggufs:
    print(f'⬇️ Downloading {g}...')
    files.download(g)
```

---

## 💻 Step 3: Local Benchmarking (After Download)

Once the file is downloaded to your machine:

```bash
# 1. Copy model into models directory
cp ~/Downloads/*lfm350-master*.gguf models/mivi-lfm350-master.Q4_K_M.gguf

# 2. Run the 11-agentic-workflow benchmark
MIVI_RUNTIME_MODE=worker-eco \
MIVI_REASONER_MODEL=models/mivi-lfm350-master.Q4_K_M.gguf \
MIVI_CODER_MODEL=models/mivi-lfm350-master.Q4_K_M.gguf \
python3 scripts/eval_agent_workflows.py
```
