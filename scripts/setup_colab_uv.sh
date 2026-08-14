#!/usr/bin/env bash
set -euo pipefail

echo "========================================================="
echo "🚀 MIVI-V2 Colab Fast Setup (Powered by uv & Rust tooling)"
echo "========================================================="

# 1. Install uv (Rust package manager)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Fast Install Unsloth and AI dependencies via uv
echo "⚡ Installing AI training dependencies with uv..."
uv pip install --system --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --system trl peft accelerate bitsandbytes datasets transformers orjson ruff sentencepiece protobuf

echo "========================================================="
echo "✅ Environment Ready in seconds with Rust-powered uv!"
echo "========================================================="
