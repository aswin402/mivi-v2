#!/usr/bin/env python3
"""
MIVI-V2 Colab Remote Runner
Automates dependency installation, dataset transfer, training execution,
and model download over the active tmate SSH session.
"""

import base64
import os
import sys
import time
import pexpect

TMATE_HOST = os.environ.get("MIVI_COLAB_SSH", "VWpmQmNjQ67XbbQg3N6x2Qh7X@sfo2.tmate.io")

def run_remote_command(child, cmd, wait_secs=2, expect_pattern=None, timeout=300):
    print(f"\n[LOCAL -> COLAB] Executing: {cmd}")
    child.sendline(cmd)
    if expect_pattern:
        child.expect(expect_pattern, timeout=timeout)
    else:
        time.sleep(wait_secs)

def transfer_file_base64(child, local_path, remote_path):
    print(f"\n[LOCAL -> COLAB] Transferring {local_path} -> {remote_path}...")
    with open(local_path, "rb") as f:
        data = f.read()
    b64_str = base64.b64encode(data).decode('utf-8')
    
    # Split into chunks of 10,000 characters to prevent buffer overflow
    chunk_size = 10000
    chunks = [b64_str[i:i+chunk_size] for i in range(0, len(b64_str), chunk_size)]
    
    run_remote_command(child, f"rm -f {remote_path}.b64", wait_secs=1)
    for i, chunk in enumerate(chunks):
        run_remote_command(child, f"cat << 'EOF' >> {remote_path}.b64\n{chunk}\nEOF", wait_secs=1)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  Sent chunk {i+1}/{len(chunks)}...")
            
    run_remote_command(child, f"python3 -c \"import base64; open('{remote_path}', 'wb').write(base64.b64decode(open('{remote_path}.b64').read().replace('\\n', '')))\"", wait_secs=2)
    run_remote_command(child, f"rm -f {remote_path}.b64", wait_secs=1)
    run_remote_command(child, f"ls -lh {remote_path}", wait_secs=2)
    print(f"✅ Transfer complete: {remote_path}")

def main():
    print("=" * 60)
    print(f"🚀 Connecting to Colab via: {TMATE_HOST}")
    print("=" * 60)

    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {TMATE_HOST}"
    child = pexpect.spawn(ssh_cmd, encoding='utf-8', timeout=600)
    child.logfile = sys.stdout

    time.sleep(2)
    child.send("q") # Dismiss tmate splash screen
    time.sleep(1)

    # 1. Check GPU
    run_remote_command(child, "nvidia-smi", wait_secs=3)

    # 2. Install Unsloth and training dependencies
    print("\n📦 Installing Unsloth and ML dependencies on Colab...")
    run_remote_command(
        child,
        "pip install --upgrade --no-cache-dir \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\" trl peft accelerate bitsandbytes datasets transformers",
        wait_secs=5
    )

    # 3. Create workspace and transfer files
    run_remote_command(child, "mkdir -p /content/mivi_train", wait_secs=1)
    run_remote_command(child, "cd /content/mivi_train", wait_secs=1)

    transfer_file_base64(child, "scripts/train_mivi_unsloth.py", "/content/mivi_train/train_mivi_unsloth.py")
    transfer_file_base64(child, "datasets/mivi_sub1b_tuning_dataset.jsonl", "/content/mivi_train/mivi_sub1b_tuning_dataset.jsonl")

    # 4. Start Unsloth fine-tuning
    print("\n🔥 Starting MIVI 0.5B Unsloth QLoRA Fine-Tuning...")
    train_cmd = "python3 train_mivi_unsloth.py --dataset mivi_sub1b_tuning_dataset.jsonl --output-dir outputs/mivi-0.5b-tool-expert --max-steps 250"
    run_remote_command(child, train_cmd, wait_secs=5)

    print("\n⏳ Training is now executing on Colab T4 GPU! Monitoring log output...")
    # Keep session active and monitor until completion
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nDetaching monitor. Session remains active on Colab.")

if __name__ == "__main__":
    main()
