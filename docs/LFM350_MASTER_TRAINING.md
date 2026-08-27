# 🚀 MIVI Master Base Model — Colab Training Guide (V2)

> **Model Target:** `LiquidAI/LFM2.5-350M`
> **Platform:** Google Colab (Free Tier - T4 GPU, 15 GB VRAM)
> **Dataset:** **15,000 samples** across 10 agentic categories (XML `<tool_call>` format)
> **Loss Masking:** Unsloth Response-Only Masking (`train_on_responses_only`)
> **Training Time:** ~5–8 minutes (500 optimizer steps)
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
print("✅ Environment ready!")
```

---

### 🔹 Cell 2: Clone Repo & Build Dataset (~3 seconds)
```python
import os, pathlib, subprocess

REPO = 'https://github.com/aswin402/mivi-v2.git'
BRANCH = 'feat/verifier-sandbox'

if not pathlib.Path('mivi-v2').exists():
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO], check=True)
os.chdir('/content/mivi-v2')
subprocess.run(['git', 'pull', 'origin', BRANCH], check=True)

# Build 15,000-sample Master Dataset (10 categories, XML <tool_call> format)
subprocess.run([
    'python3', 'scripts/build_15k_agentic_dataset.py',
    '--total', '15000', '--fast',
    '--out', 'datasets/mivi_master_15k_sft.jsonl'
], check=True)
DATASET = 'datasets/mivi_master_15k_sft.jsonl'
print('✅ Master Dataset ready:', DATASET, '| rows:', sum(1 for _ in open(DATASET)))
```

---

### 🔹 Cell 3: Define Training Function
```python
import os, json, glob, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from google.colab import files

def train_and_export_master(
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_master_15k_sft.jsonl",
    output_dir="outputs/mivi-lfm350-master",
    max_steps=500,
    batch_size=8,
    grad_accum=2,
    lr=2e-4
):
    print("=" * 60)
    print(f"🚀 MIVI Master Base Model Training")
    print(f"📁 Model:    {model_name}")
    print(f"📦 Dataset:  {dataset_path}")
    print(f"⚡ GPU:      {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
    print(f"🔄 Steps:    {max_steps} (effective batch={batch_size * grad_accum})")
    print("=" * 60)

    # 1. Load base model in 4-bit with 512 context
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
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

    # 3. Load dataset (prompt + completion format)
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    formatted = []
    for item in raw:
        if "prompt" in item and "completion" in item:
            formatted.append({"text": item["prompt"] + item["completion"] + "<|im_end|>\n"})
        elif "messages" in item:
            formatted.append({"text": tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)})

    dataset = Dataset.from_list(formatted)
    print(f"📄 Loaded {len(dataset)} training samples")

    # 4. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=15,
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

    # 5. Response-Only Loss Masking (only train on assistant responses)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("🔥 Training started...")
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
    print(f"🎉 GGUF export done: {output_dir}")
```

---

### 🔹 Cell 4: Train! (~5–8 minutes)
```python
train_and_export_master(
    model_name="LiquidAI/LFM2.5-350M",
    dataset_path="datasets/mivi_master_15k_sft.jsonl",
    output_dir="outputs/mivi-lfm350-master",
    max_steps=500,        # 500 steps × 16 effective batch = 8,000 sample views
    batch_size=8,
    grad_accum=2,
    lr=2e-4
)
```

---

### 🔹 Cell 5: Download Model
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
cp ~/Downloads/*LFM2.5-350M*.gguf models/new/LFM2.5-350M.Q4_K_M_new.gguf

# 2. Run 26-test deep evaluation
python3 scripts/deep_model_test.py

# 3. Run 11-workflow benchmark
MIVI_RUNTIME_MODE=worker-eco \
MIVI_REASONER_MODEL=models/new/LFM2.5-350M.Q4_K_M_new.gguf \
MIVI_CODER_MODEL=models/new/LFM2.5-350M.Q4_K_M_new.gguf \
python3 scripts/eval_agent_workflows.py

# Target: ≥ 9/11 pass rate
```

---

## 📊 What Changed from V1

| Parameter | V1 (Old) | V2 (New) | Why |
|---|---|---|---|
| **Completion format** | JSON `{"tool_calls":[...]}` | XML `<tool_call>{...}</tool_call>` | Matches server inference format |
| **Data categories** | 4 (bash, job, web, chat) | 10 (+ weather, file, tool_select, identity, agent_protocol, coding) | Comprehensive coverage |
| **max_steps** | 120 | 500 | 53% dataset coverage vs 12.8% |
| **warmup_steps** | 5 | 15 | Smoother convergence at 500 steps |
| **logging_steps** | 10 | 25 | Less noise in output |
| **Negative samples** | 1,000 | 2,000 | Better "when NOT to call tools" |
| **Identity samples** | 200 | 1,000 | Stronger MIVI identity lock |
| **Bash commands** | 17 unique | 50+ unique | Better command disambiguation |
