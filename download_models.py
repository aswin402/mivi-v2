import os
import sys

def download_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    print("=========================================================")
    print(" 🚀 MIVI-V2 MODEL DOWNLOADER")
    print(" Downloading Reasoner, Coder, and Vision Model Suite...")
    print("=========================================================\n")

    downloads = [
        {
            "name": "Qwen 2.5 0.5B Instruct (Coder)",
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "filename": "qwen2.5-0.5b-instruct-q2_k.gguf",
            "target": os.path.join(models_dir, "qwen2.5-0.5b-instruct-q2_k.gguf")
        },
        {
            "name": "Qwen 2.5 0.5B Instruct Q4_K_M (Default Coder/Tool Model)",
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "target": os.path.join(models_dir, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        },
        {
            "name": "Llama 3.2 1B Instruct (Reasoner)",
            "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "filename": "Llama-3.2-1B-Instruct-IQ3_M.gguf",
            "target": os.path.join(models_dir, "Llama-3.2-1B-Instruct-IQ3_M.gguf")
        },
        {
            "name": "Qwen3 0.6B Q4_K_M (Thinking Reasoner Candidate)",
            "repo_id": "Antigma/Qwen3-0.6B-GGUF",
            "filename": "qwen3-0.6b-q4_k_m.gguf",
            "target": os.path.join(models_dir, "qwen3-0.6b-q4_k_m.gguf")
        },
        {
            "name": "MiniCPM-V 4.6 (Vision Model)",
            "repo_id": "ggml-org/MiniCPM-V-4.6-GGUF",
            "filename": "MiniCPM-V-4.6-Q4_K_M.gguf",
            "target": os.path.join(models_dir, "MiniCPM-V-4.6-Q4_K_M.gguf")
        },
        {
            "name": "MiniCPM-V 4.6 Vision Projector (mmproj)",
            "repo_id": "ggml-org/MiniCPM-V-4.6-GGUF",
            "filename": "mmproj-MiniCPM-V-4.6-Q8_0.gguf",
            "target": os.path.join(models_dir, "mmproj-MiniCPM-V-4.6-Q8_0.gguf")
        },
        {
            "name": "Cactus Compute Needle 26M (Sub-2ms Router)",
            "repo_id": "Cactus-Compute/needle",
            "filename": "model.safetensors",
            "target": os.path.join(models_dir, "needle-model.safetensors")
        }
    ]

    for item in downloads:
        if os.path.exists(item["target"]):
            print(f"[SKIP] '{item['name']}' already exists at '{item['target']}'.")
            continue

        print(f"[DOWNLOADING] {item['name']} from HF repo '{item['repo_id']}'...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                local_dir=models_dir
            )
            size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
            print(f"[SUCCESS] Saved '{item['filename']}' ({size_mb:.2f} MB)\n")
        except Exception as e:
            print(f"[ERROR] Failed to download {item['name']}: {e}\n")

if __name__ == "__main__":
    download_models()
