# Changelog

All notable changes to the **MIVI-V2** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.0.1] - 2026-07-22

### 🚀 Initial Public Release — Pure Rust Local AI Engine

#### Added
* 🦀 **Pure Rust Core Engine**: Completely migrated the core orchestration engine from Python to 100% Pure Rust (`tokio`, `axum`, `serde`), reducing idle memory footprint to **< 12 MB RAM**.
* 🌐 **Axum REST API Server**: Added an OpenAI-compatible REST server listening on `http://localhost:8000/v1` supporting `/v1/chat/completions`, `/v1/models`, and root health checks.
* ⚙️ **Multi-Language Double-Loop Verifier**: Enhanced `CompilerVerifier` to support compiling and executing generated code across 5 programming languages:
  * 🐍 **Python** (`python3`)
  * 🟨 **JavaScript** (`node`)
  * 📜 **TypeScript** (`bun` / `node` fallback)
  * 🦀 **Rust** (`rustc` temporary compilation & execution)
  * ⚡ **C / C++** (`g++` temporary compilation & execution)
* 👁️ **MiniCPM-V 4.6 Vision Integration**: Added multimodal vision support in `EdgeBrain` (`query_vision()`) and REST API handling for image URLs using `MiniCPM-V-4.6-Q4_K_M` + `mmproj-MiniCPM-V-4.6-Q8_0`.
* 🗺️ **Multi-Step Sequential Orchestrator**: Integrated `Llama-3.2-1B` to generate structured JSON step plans and `Qwen-2.5-0.5B` to execute steps sequentially, feeding previous step outputs forward.
* ⚡ **FlashAttention & KV Cache Acceleration**: Configured `llama-cli` invocation arguments with `-fa on` (FlashAttention) and `-ctk q8_0 -ctv q8_0` (8-bit Key-Value cache quantization) with an 8,192 token context window.
* 🧠 **Token-Set Semantic Cache**: Implemented a Jaccard token-set similarity prompt cache ($\ge 0.85$ threshold) for sub-millisecond responses on repeat or rephrased queries.
* 🔎 **TurboVec RAG Engine**: Workspace file indexer with line-range chunking and commented snippet formatting to prevent LLM code generation pollution.
* 📊 **Fine-Tuning Dataset Logger**: Auto-appends verified prompt-code-terminal execution pairs into `dataset/verified_pairs.jsonl`.
* 📥 **One-Click Downloader**: Added `download_models.py` for downloading the complete GGUF model suite (`Llama 3.2 1B`, `Qwen 2.5 0.5B`, and `MiniCPM-V 4.6` + `mmproj`).
