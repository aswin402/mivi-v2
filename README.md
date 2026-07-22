# 🚀 MIVI-V2: Ultra-Compact Low-Resource Pure Rust Local AI Engine

[![Version](https://img.shields.io/badge/version-v0.0.1-brightgreen.svg)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![RAM Footprint](https://img.shields.io/badge/Idle%20RAM-%3C%2012%20MB-purple.svg)]()
[![Active Footprint](https://img.shields.io/badge/Active%20RAM-~180%20MB-green.svg)]()

**MIVI-V2 (v0.0.1)** is a **100% Pure Rust, Small Model Logic (SML) local AI engine** designed to run advanced reasoning, coding, vision analysis, RAG, and multi-agent coordination on low-spec hardware with an active RAM footprint of **< 12 MB idle RAM** and **~180 MB active process RAM**.

It acts as an ultra-fast, zero-overhead, OpenAI-compatible local AI backend for autonomous AI agents including **Hermes Agent**, **OpenCode Agent**, **AutoGen**, **CrewAI**, **VS Code (Continue.dev)**, and **Cursor IDE**.

---

## 🌟 Key Features in v0.0.1

* 🦀 **100% Pure Rust Architecture:** Zero Python runtime, zero PyTorch/transformers memory bloat, and zero virtual environment dependencies.
* ⚡ **High-Speed Async Axum REST Server:** OpenAI-compatible API listening on `http://localhost:8000/v1` for `/v1/chat/completions` and `/v1/models`.
* 🧠 **Specialized SLM Multi-Agent Engine:**
  * 🧠 **Reasoner & Orchestrator:** `Llama-3.2-1B-Instruct-IQ3_M` (Meta)
  * 💻 **Coder Engine:** `Qwen-2.5-0.5B-Instruct-Q2_K` (Alibaba)
  * 👁️ **Vision Specialist:** `MiniCPM-V-4.6-Q4_K_M` + `mmproj-Q8_0` (OpenBMB)
* ⚙️ **Multi-Language Double-Loop Verifier:** Generates, executes, and auto-corrects code across **Python, JavaScript, TypeScript, Rust, and C++** in local runtimes until code passes cleanly.
* ⚡ **FlashAttention & KV Cache Quantization:** Pre-configured with `-fa on` FlashAttention and `-ctk q8_0 -ctv q8_0` 8-bit Key-Value cache quantization for ultra-fast token generation on low-end CPUs/GPUs.
* 🔎 **TurboVec RAG Engine:** Workspace code chunking and retrieval (< 1 MB RAM footprint) with automatic code pollution protection.
* 🧠 **Zero-Overhead Semantic Cache:** Token-set Jaccard similarity cache for instant **< 0.001s responses** on repeat or rephrased queries.
* 📊 **Fine-Tuning Dataset Generator:** Automatically logs verified execution pairs into `dataset/verified_pairs.jsonl` for SFT fine-tuning.

---

## 📁 Repository Structure

```text
mivi-v2/
├── Cargo.toml                  # Cargo Package Manifest
├── README.md                   # Main Documentation
├── CHANGELOG.md                # Release History & Version v0.0.1 Notes
├── download_models.py          # One-Click GGUF Model Weights Downloader
├── bin/                        # Pre-compiled native llama.cpp release binaries
├── docs/                       # Comprehensive System Documentation
│   ├── ARCHITECTURE.md         # Internal Engine Architecture & Workflow
│   ├── AGENTS_GUIDE.md         # Connecting Hermes, OpenCode, VS Code, Cursor
│   └── API_REFERENCE.md        # OpenAI REST API Specification
├── models/                     # GGUF Model Checkpoints (.gitignore managed)
└── src/
    ├── main.rs                 # Entry point & CLI mode dispatcher
    ├── lib.rs                  # Exported Rust Library Crate
    ├── brain.rs                # EdgeBrain llama-cli Process Wrapper
    ├── verifier.rs             # Multi-Language Double-Loop Terminal Verifier
    ├── rag.rs                  # TurboVec Workspace Code Indexer & RAG
    ├── cache.rs                # Semantic Token-Set Prompt Cache
    ├── router.rs               # Sub-millisecond Needle Intent Router
    ├── logger.rs               # Verified SFT Dataset Logger
    ├── orchestrator.rs         # Multi-Agent Planner & Sequential Executor
    ├── server.rs               # Axum Async REST API Server (Port 8000)
    ├── cli.rs                  # Interactive Terminal Chat UI
    └── audit.rs                # End-to-End System Health Diagnostic
```

---

## 🚀 Quick Start

### 1. Installation & Build

Ensure you have Rust (1.75+) installed:

```bash
git clone https://github.com/aswin402/mivi-v2.git
cd mivi-v2

# Build optimized release binary
cargo build --release
```

### 2. Download Model Weights

Fetch the ultra-compact GGUF model suite using the downloader:

```bash
uv run --with huggingface_hub python3 download_models.py
```

### 3. Usage Commands

```bash
# 1. Run End-to-End System Health Audit
./target/release/mivi audit

# 2. Start OpenAI-Compatible API Server (port 8000)
./target/release/mivi serve

# 3. Launch Interactive Terminal Chat CLI
./target/release/mivi cli

# 4. Execute a single code task
./target/release/mivi task "Write a python script calculating Fibonacci numbers"
```

---

## 🤖 Connecting External Agents (Hermes, OpenCode, VS Code)

Configure your agent environment or client to use MIVI-V2 as your local backend:

* **Base URL:** `http://localhost:8000/v1`
* **API Key:** `local`
* **Models Available:** `mivi-v2`, `qwen-2.5-0.5b`, `llama-3.2-1b`, `minicpm-v-4.6`

See [docs/AGENTS_GUIDE.md](docs/AGENTS_GUIDE.md) for full configuration guides.

---

## 📚 Documentation

* 📖 [Architecture Guide](docs/ARCHITECTURE.md)
* 🤖 [External Agents Integration Guide](docs/AGENTS_GUIDE.md)
* 📡 [REST API Reference](docs/API_REFERENCE.md)
* 📜 [Changelog](CHANGELOG.md)

---

## 📄 License
MIT License. Free for open-source and commercial use.
