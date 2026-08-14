# Google Colab Fine-Tuning Guide: MIVI Sub-1B Engine

> **Target Platform:** Google Colab Free Tier (NVIDIA T4 GPU, 16 GB VRAM)  
> **Estimated Training Time:** ~15–20 minutes  
> **Peak VRAM Usage:** < 2.5 GB  
> **Output:** `mivi-0.5b-tool-expert-unsloth.Q4_K_M.gguf` (~460 MB)

---

## Step 1: Open Google Colab

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload Notebook** and choose `notebooks/train_mivi_unsloth.ipynb` from this repository.
3. In the top menu, go to **Runtime** $\rightarrow$ **Change runtime type** and select **T4 GPU** (Hardware accelerator).

---

## Step 2: Upload the Training Dataset

1. Generate the local dataset by running:
   ```bash
   python3 scripts/prepare_mivi_dataset.py
   ```
2. In Colab's left sidebar, click the **Files** (folder icon) $\rightarrow$ **Upload to session storage**.
3. Upload `datasets/mivi_sub1b_tuning_dataset.jsonl` (or rename to `mivi_sub1b_tuning_dataset.jsonl`).

---

## Step 3: Run the Training Cells

1. **Install Unsloth & Dependencies** (Cell 1):
   Installs `unsloth`, `trl`, `peft`, `accelerate`, and `bitsandbytes`.
2. **Load Model & LoRA Configuration** (Cells 2 & 3):
   Loads `Qwen/Qwen2.5-0.5B-Instruct` in 4-bit and attaches LoRA adapters to all attention and feed-forward projection layers (`r = 16`, `lora_alpha = 32`).
3. **Train with SFTTrainer** (Cell 5):
   Executes 250 steps of cosine-scheduled fine-tuning. Training loss will drop steadily below ~0.8.
4. **Export GGUF** (Cell 6):
   Unsloth automatically merges the LoRA adapter weights into the base model and quantizes to `Q4_K_M` GGUF.

---

## Step 4: Download and Place Model in MIVI-V2

1. After Cell 6 completes, find `mivi-0.5b-tool-expert-unsloth.Q4_K_M.gguf` in the Colab file explorer.
2. Download the `.gguf` file to your local computer.
3. Move the file to MIVI-V2's `models/` directory:
   ```bash
   mv ~/Downloads/mivi-0.5b-tool-expert-unsloth.Q4_K_M.gguf models/mivi-0.5b-tool-q4_k_m.gguf
   ```
4. Start MIVI-V2:
   ```bash
   MIVI_REASONER_MODEL=models/mivi-0.5b-tool-q4_k_m.gguf MIVI_CODER_MODEL=models/mivi-0.5b-tool-q4_k_m.gguf cargo run --release -- serve
   ```
