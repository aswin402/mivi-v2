# 🚀 Google Colab Fine-Tuning Guide — MIVI Turbo Master Model (High-VRAM V3)

> **Platform:** Google Colab (Free T4 GPU, 15 GB VRAM)  
> **Estimated Training Time:** 30–120 minutes on a real T4/L4 GPU; longer means inspect the preflight
> **Target Model:** `LiquidAI/LFM2.5-350M` (Hybrid 350M, 438 MB inference RAM, 88 tok/s)  
> **Dataset:** `datasets/mivi_master_15k_sft.jsonl` (20,000 samples across 10 agentic categories with XML `<tool_call>` format)  
> **Loss Masking:** Unsloth Response-Only Masking (`train_on_responses_only`)  
> **Output:** `mivi-lfm350-master` GGUF (`Q4_K_M`, ~229 MB)

---

## 🎯 Step 1: Open Google Colab & Enable T4 GPU

1. Open **[Google Colab](https://colab.research.google.com/)** in your browser.
2. Create a new notebook.
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

# 2. Fast install Unsloth, Unsloth Zoo, HuggingFace datasets and ML dependencies
!uv pip install --system unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers sentencepiece protobuf
print("✅ Environment ready via uv!")
```

---

### 🔹 Cell 1b: Confirm the GPU after installing dependencies
```python
import subprocess, torch
assert torch.cuda.is_available(), "CUDA is missing: switch Colab to a T4/L4/A100 runtime."
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 2**30:.1f} GiB")
subprocess.run(["nvidia-smi"], check=False)
```

If this cell does not show a CUDA GPU, stop here. A CPU run is the main reason this job can take hours.

---

### 🔹 Cell 2: Clone Repository & Build 20,000-Sample Master Dataset (~4 seconds)
```python
import os, pathlib, subprocess

REPO = 'https://github.com/aswin402/mivi-v2.git'
BRANCH = "main"

if not pathlib.Path('mivi-v2').exists():
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO], check=True)
os.chdir('/content/mivi-v2')
subprocess.run(['git', 'pull', 'origin', BRANCH], check=True)

# Build 20,000 Sample Master Dataset (10 Categories, XML <tool_call> format)
subprocess.run(['python3', 'scripts/build_15k_agentic_dataset.py', '--total', '20000', '--fast', '--out', 'datasets/mivi_master_15k_sft.jsonl'], check=True)
DATASET = 'datasets/mivi_master_15k_sft.jsonl'
print('✅ Master Dataset ready:', DATASET, '| rows:', sum(1 for _ in open(DATASET)))
```

---

### 🔹 Cell 3: Run the canonical trainer script
```python
# 1,000 optimizer steps; effective batch = 8 x 8 = 64.
# The trainer fails fast if Colab is accidentally running on CPU.
!python3 scripts/train_mivi_unsloth.py --model LiquidAI/LFM2.5-350M --dataset datasets/mivi_master_15k_sft.jsonl --output outputs/mivi-lfm350-master --steps 1000 --max-seq-length 512 --batch-size 8 --grad-accum 8 --lr 2.5e-4 --dataset-procs 2 --loader-workers 2 --save-steps 100 --gradient-checkpointing
```

---

### 🔹 Cell 4: Verify progress and resume if needed
```python
# A fresh run was started in Cell 3. Check that the step counter is moving:
!nvidia-smi

# If Colab disconnects after a checkpoint, resume from the latest directory, e.g.:
# !python3 scripts/train_mivi_unsloth.py --model LiquidAI/LFM2.5-350M --dataset datasets/mivi_master_15k_sft.jsonl --output outputs/mivi-lfm350-master --steps 1000 --max-seq-length 512 --batch-size 8 --grad-accum 8 --lr 2.5e-4 --dataset-procs 2 --loader-workers 2 --save-steps 100 --gradient-checkpointing --resume outputs/mivi-lfm350-master/checkpoint-500
```

---

### 🔹 Cell 5: Download the Master Model to Your Machine
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

print('📦 Target Master GGUF:', ggufs)
for g in ggufs:
    print(f'⬇️ Downloading {g}...')
    files.download(g)
```

---

## If training is slow, appears stuck, or runs out of memory

- The job is 1,000 optimizer steps × 64 samples = 64,000 sample views. With `max_seq_length=512`, the upper bound is 32.8M token positions before padding/truncation. With batch 8 and gradient checkpointing, one hour can be normal for this 1,000-step T4 run; use the step counter and GPU utilization to judge progress.
- The trainer may remove rows where the response marker was truncated: your log removed 3,412 rows and trained on 16,588. This is not an OOM; it means the 512-token limit cut off the assistant response. Do not increase sequence length on the T4 unless you reduce batch size further.
- The training cell must print the GPU name, effective batch, and checkpoint interval before model loading. If it does not, it is not using the updated script.
- If the loss/step counter does not advance for 10 minutes, run `!nvidia-smi` in another cell and inspect GPU utilization. Do not start a second training process.
- The T4 profile uses batch 8 with gradient checkpointing. If it still runs out of memory, keep the effective batch with `--batch-size 4 --grad-accum 16 --gradient-checkpointing`. This is slower but safer.
- Checkpoints are written every 100 steps. Resume with the commented command in Cell 4 after a disconnect.

## 💻 Step 3: Local Benchmarking (After Download)

```bash
# The master dataset teaches XML <tool_call>, so benchmark it in Hermes mode.
export MIVI_TOOL_FORMAT=hermes

# 1. Copy downloaded model to models/new/
cp ~/Downloads/*q4_k_m*.gguf models/new/LFM2.5-350M.Q4_K_M.gguf

# 2. Run the 26-test deep evaluation
python3 scripts/deep_model_test.py

# 3. Run the 11-agentic-workflow benchmark
python3 scripts/run_11_eval.py
```
