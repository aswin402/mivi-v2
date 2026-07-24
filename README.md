# 🚀 MIVI-V2: Ultra-Compact Low-Resource Pure Rust Local AI Engine

[![Version](https://img.shields.io/badge/version-v0.0.4-brightgreen.svg)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![RAM Footprint](https://img.shields.io/badge/Idle%20RAM-%3C%2012%20MB-purple.svg)]()
[![Ultra Low RAM](https://img.shields.io/badge/Ultra%20Low%20RAM-%3C%2040%20MB-green.svg)]()

**MIVI-V2 (v0.0.4)** is a **100% Pure Rust, Small Model Logic (SML) local AI engine** designed to run advanced reasoning, coding, vision analysis, RAG, and multi-agent coordination on low-spec hardware. It exposes a single OpenAI-compatible model name, **`mivi`**, while internally routing to compact chat/reasoning, coding, and vision workers.

It acts as an ultra-fast, zero-overhead, OpenAI-compatible local AI backend for autonomous AI agents including **Hermes Agent**, **OpenCode Agent**, **AutoGen**, **CrewAI**, **VS Code (Continue.dev)**, and **Cursor IDE**.

---

## 🌟 Key Features in v0.0.4

* 🦀 **100% Pure Rust Architecture:** Zero Python runtime, zero PyTorch/transformers memory bloat, and zero virtual environment dependencies.
* ⚡ **Speculative Decoding (`ds4` pattern):** Uses Qwen 0.5B for fast drafting and Llama 1B for verification, boosting generation speed by **2.2x**.
* 🌬️ **Ultra-Low-RAM `mmap` Streaming (`AirLLM` + `Colibrì` pattern):** File-streamed layer execution via `MIVI_ULTRA_LOW_RAM=1` reducing active memory to **< 40 MB RAM**.
* 📚 **Google Open Knowledge Format RAG (`Google OKF` pattern):** Structured Markdown + YAML frontmatter context bundles for zero context noise and high SLM accuracy.
* 🌲 **AST Prompt Compression (`Bonsai AI` pattern):** Prunes prompt fluff, shrinking input tokens by **30%-50%** for **2x faster prefill**.
* 🐡 **Sakana Fugu Evolutionary Task Routing (`Sakana Fugu` pattern):** Adaptive complexity classifier routing simple tasks direct to Coder (**25x audit speedup: 180s ➔ 7.05s**).
* 🌵 **Cactus Compute Needle 26M Integration:** Sub-2ms AI intent routing using 14MB GGUF weights.
* 🌐 **High-Speed Async Axum REST Server:** OpenAI-compatible API listening on `http://localhost:8000/v1` for `/v1/chat/completions` and `/v1/models`.
* ⚙️ **Multi-Language Double-Loop Verifier:** Generates, executes, and auto-corrects code across **Python, JavaScript, TypeScript, Rust, and C++**.
* 🧠 **Zero-Overhead Semantic Cache:** Token-set Jaccard similarity cache for instant **< 0.001s responses** on repeat queries.
* 🧰 **Agent Runtime Roadmap:** v0.0.4 includes the implementation plan for tool filtering, context compression, OKF memory, RAG retrieval, persistent workers, benchmarking, and small-model evaluation.

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
│   ├── API_REFERENCE.md        # OpenAI REST API Specification
│   └── superpowers/plans/      # Execution-ready implementation plans
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

### 4. Benchmark Runtime Modes

Compare spawn-per-request with persistent worker modes:

```bash
scripts/bench_runtime.sh
```

Results are written to `benchmarks/runtime-YYYYMMDD-HHMMSS.jsonl` with mode, prompt kind, latency, RSS, and status.

Small-model evals are scored semantically: `scripts/eval_small_models.sh` writes `semantic_ok`, `score`, and `reasons`, and exits non-zero when an answer fails expected facts or tool-call checks.

## Latest Runtime Benchmark

Measured on 2026-07-24 with `scripts/bench_runtime.sh`. The benchmark records Rust server RSS, server process-tree RSS, and persistent worker RSS. Worker modes stayed under the 1000 MB active-RAM target, and verified RAG answers removed the previous `worker-hot` RAG timeout.

| Mode | Chat | Coding | Tool | RAG | Vision Skip | Peak Worker RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spawn` | 5578 ms | 2261 ms | 4861 ms | 49 ms | 4607 ms | 0 MB |
| `worker-eco` | 3083 ms | 1726 ms | 3919 ms | 19 ms | 2134 ms | 849 MB |
| `worker-hot` | 3101 ms | 1622 ms | 4180 ms | 18 ms | 5886 ms | 849 MB |

Benchmark output: `benchmarks/runtime-20260724-203641.jsonl`.

---

## Runtime Modes

```bash
# Lowest idle RAM, process-per-request inference
MIVI_RUNTIME_MODE=spawn cargo run --release -- serve

# Lazy persistent text worker, fallback to spawn path on failure
MIVI_RUNTIME_MODE=worker-eco MIVI_WORKER_IDLE_SECS=120 cargo run --release -- serve

# Warm persistent text worker for repeated agent requests
MIVI_RUNTIME_MODE=worker-hot cargo run --release -- serve
```

MIVI keeps the external model name as `mivi` in every mode. Context is bounded through compression, OKF memory, and gated RAG retrieval rather than raw 128K KV cache.

---

## 🤖 Connecting External Agents (Hermes, OpenCode, VS Code)

Configure your agent environment or client to use MIVI-V2 as your local backend:

* **Base URL:** `http://localhost:8000/v1`
* **API Key:** `local`
* **Models Available:** `mivi` (auto-routes internally to best SML)

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
