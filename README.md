# 🚀 MIVI-V2: Ultra-Compact Low-Resource Pure Rust Local AI Engine

[![Version](https://img.shields.io/badge/version-v0.0.5-brightgreen.svg)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![RAM Footprint](https://img.shields.io/badge/Idle%20RAM-%3C%2012%20MB-purple.svg)]()
[![Ultra Low RAM](https://img.shields.io/badge/Ultra%20Low%20RAM-%3C%2040%20MB-green.svg)]()

**MIVI-V2 (v0.0.5)** is a **100% Pure Rust, Small Model Logic (SML) local AI engine** designed to run advanced reasoning, coding, vision analysis, RAG, and multi-agent coordination on low-spec hardware. It exposes a single OpenAI-compatible model name, **`mivi`**, while internally routing to compact chat/reasoning, coding, and vision workers.

It acts as an ultra-fast, zero-overhead, OpenAI-compatible local AI backend for autonomous AI agents including **Hermes Agent**, **OpenCode Agent**, **AutoGen**, **CrewAI**, **VS Code (Continue.dev)**, and **Cursor IDE**.

---

## 🌟 Key Features in v0.0.5

* 🦀 **100% Pure Rust Architecture:** Zero Python runtime, zero PyTorch/transformers memory bloat, and zero virtual environment dependencies.
* ⚡ **Speculative Decoding (`ds4` pattern):** Uses the configured coder for fast drafting and the configured reasoner for verification, boosting generation speed by **2.2x**.
* 🌬️ **Ultra-Low-RAM `mmap` Streaming (`AirLLM` + `Colibrì` pattern):** File-streamed layer execution via `MIVI_ULTRA_LOW_RAM=1` reducing active memory to **< 40 MB RAM**.
* 📚 **Google Open Knowledge Format RAG (`Google OKF` pattern):** Structured Markdown + YAML frontmatter context bundles for zero context noise and high SLM accuracy.
* 🌲 **AST Prompt Compression (`Bonsai AI` pattern):** Prunes prompt fluff, shrinking input tokens by **30%-50%** for **2x faster prefill**.
* 🐡 **Sakana Fugu Evolutionary Task Routing (`Sakana Fugu` pattern):** Adaptive complexity classifier routing simple tasks direct to Coder (**25x audit speedup: 180s ➔ 7.05s**).
* 🌵 **Cactus Compute Needle 26M Integration:** Sub-2ms AI intent routing using 14MB GGUF weights.
* 🌐 **High-Speed Async Axum REST Server:** OpenAI-compatible API listening on `http://localhost:8000/v1` for `/v1/chat/completions` and `/v1/models`.
* ⚙️ **Multi-Language Double-Loop Verifier:** Generates, executes, and auto-corrects code across **Python, JavaScript, TypeScript, Rust, and C++**.
* 🧠 **Zero-Overhead Semantic Cache:** Token-set Jaccard similarity cache for instant **< 0.001s responses** on repeat queries.
* 🧰 **Agent Runtime Roadmap:** v0.0.5 includes the implementation plan for tool filtering, context compression, OKF memory, RAG retrieval, persistent workers, benchmarking, and small-model evaluation.
* 🧾 **Tool Output Compression:** Cargo, npm/pnpm/yarn/vitest/jest, pytest, and git diff outputs are reduced to salient failure/hunk lines before they enter the small-model context.

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

Enable compact per-request diagnostics with `MIVI_TRACE=1`; traces append JSONL rows to `logs/mivi-trace.jsonl` or `MIVI_TRACE_PATH`.

Agent workflow evals simulate OpenCode-style traffic with injected skill metadata, 100+ tools, long tool output, RAG/memory prompts, and optional trace rows:

```bash
MIVI_TRACE=1 scripts/eval_agent_workflows.py
```

Results are written to `model-eval-results/agent-workflows-YYYYMMDD-HHMMSS.jsonl`.

Compare internal GGUF candidates while keeping the external API model as `mivi`:

```bash
bash scripts/eval_model_candidates.sh
```

Set `MIVI_CANDIDATES_FILE` to a JSONL file with `name`, `reasoner`, and `coder` fields to test additional model files. Current passed candidates include Qwen2.5 0.5B Q4_K_M for coder/tool use and Qwen3 0.6B Q4_K_M as a reasoner candidate with the Qwen2.5 coder.

## Latest Runtime Benchmark

Measured on 2026-07-26 with `scripts/bench_runtime.sh` using the built-in defaults: Qwen3 0.6B Q4_K_M reasoner, Qwen2.5 0.5B Q4_K_M coder, and a 3072 raw context budget. Worker modes stayed under the 1000 MB active-RAM target with practical headroom.

| Mode | Chat | Coding | Tool | RAG | Vision Skip | Peak Worker RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spawn` | 4279 ms | 2048 ms | 15 ms | 24 ms | 6084 ms | 0 MB |
| `worker-eco` | 3141 ms | 1171 ms | 9 ms | 9 ms | 4954 ms | 932.1 MB |
| `worker-hot` | 2921 ms | 1437 ms | 10 ms | 18 ms | 4303 ms | 931.2 MB |

Benchmark output: `benchmarks/runtime-20260726-014455.jsonl`.

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

MIVI keeps the external model name as `mivi` in every mode. Context is bounded through compression, OKF memory, and gated RAG retrieval rather than raw 128K KV cache. Use `MIVI_REASONER_CONTEXT_SIZE` and `MIVI_CODER_CONTEXT_SIZE` when a candidate model needs a smaller raw KV cache to stay under the RAM target. `MIVI_REASONING_MODE=auto|think|no_think` controls Qwen3 thinking directives. `auto` is conservative for agents: normal prompts use `/no_think`, explicit deep-reasoning prompts use `/think`, and private `<think>` / `[Start thinking]` blocks are stripped before responses reach agents.

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
