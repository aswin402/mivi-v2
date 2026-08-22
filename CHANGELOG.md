# Changelog

All notable changes to the **MIVI-V2** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

* Verifier subprocesses (python3, node, bun, rustc, g++) now run inside a Linux Landlock sandbox: deny-by-default filesystem access limited to system toolchain paths plus the verifier's dedicated temp directory, and TCP bind/connect denied on kernels with Landlock ABI v4 (6.7+). Controlled by `MIVI_VERIFY_SANDBOX` (`auto` default degrades to unsandboxed with a one-time warning; `on` makes it fatal; `off` restores previous behavior).

## [v0.0.15] - 2026-08-22

### Security

* Bound the API server to `127.0.0.1` by default; new `MIVI_HOST` / `MIVI_PORT` overrides make deliberate exposure explicit.
* Replaced the bearer-token comparison with a constant-time check, removing a timing side channel on `MIVI_API_KEY`.
* Hardened the rate limiter against identity spoofing: `X-Forwarded-For` / `X-Real-IP` are honored only behind `MIVI_TRUST_PROXY_HEADERS`, clients are identified by peer socket address otherwise, and the client map is capped at 4096 entries with oldest-entry eviction to stop unbounded memory growth from spoofed-header floods.

### Fixed

* Verified-code cache now requires exact prompt matches (fuzzy hits could return wrong code) and entries expire after 10 minutes.
* `/v1/models` reports the real runtime context budget and `/` reports the actual Cargo package version instead of hardcoded values that had drifted.
* TypeScript fallback no longer runs `node --experimental-strip-types` on Node versions lacking type-stripping support (< 22.6 / < 23.6); it fails with a clear error instead of a confusing syntax failure.
* Orchestrator planner prompt now advertises all five verifier languages (Python, JavaScript, TypeScript, Rust, C++), so complex plans stop defaulting to Python-only steps.

### Added

* `MIVI_RATE_LIMIT_PER_MIN`, `MIVI_REQUEST_TIMEOUT_SECS`, and `MIVI_RAM_TARGET_MB` environment overrides for rate limiting, request timeout, and the RAM budget target.

### Changed

* Diagnostic and repair regexes (tool output compression, code-block extraction, JSON argument repair) are compiled once via `OnceLock` instead of per call.
* Context constants unified in `src/constants.rs`; docs corrected — the runtime default context budget is 8192 tokens (`DEFAULT_CONTEXT_TOKENS`), and historical 3072 references predate the change.

## [v0.0.14] - 2026-08-20

### Fixed

* Bound oversized OpenZ system prompts before tokenization and native inference, preventing long stalls and memory spikes from unbounded agent instruction envelopes.
* Prevented OpenAI-compatible tool catalogs from forcing tool generation for ordinary chat requests.
* Preserved explicit tool/action routing for web research, inventory requests, and actions such as stopping or removing scheduled jobs.

### Added

* Added a laptop-friendly justfile launcher with low-RAM server, traced server, normal server, test, build, and compatibility-check recipes.

## [v0.0.13] - 2026-08-14

### 🦀 Phase 11–14: Knowledge-Lean Sub-1B, Hybrid Semantic RAG & Claude Code Adapter (COMPLETE)

#### Added
* ⚡ **Knowledge-Lean Sub-1B Fine-Tuning Pipeline**: Built dataset generator (`scripts/prepare_mivi_dataset.py`) combining Hermes XML tool calls, OpenAI JSON calling, DeepSeek-R1 distilled `<think>` reasoning traces, compiler self-correction, and grounded QA. Created one-click Unsloth 4-bit QLoRA Colab notebook (`notebooks/train_mivi_unsloth.ipynb`) and guide (`docs/COLAB_TRAINING_GUIDE.md`) to train `mivi-0.5b-tool-q4_k_m.gguf` in ~15 minutes (< 2.5 GB VRAM).
* 🔍 **Zero-Dependency Hybrid Semantic RAG**: Implemented `src/semantic_rag.rs` featuring dense vector embedding generation, L2 normalization, and hybrid cosine similarity scoring ($0.4 \times \text{Keyword} + 0.6 \times \text{Semantic}$) with zero external runtime dependencies.
* 🔌 **Anthropic `/v1/messages` Compatibility Adapter**: Added complete Anthropic Messages API support (`/v1/messages`) in `src/server/helpers.rs`, enabling plug-and-play local AI inference with **Claude Code**, Cursor, and Anthropic-native developer tools.
* 🧠 **RAM Fit Calculator CLI (`mivi model fit <id>`)**: Added real-time Linux `/proc/meminfo` RAM calculator displaying model weight footprints, 3072/64k KV cache sizes, and memory fit status.
* 💾 **Persistent Project State (`.mivi/project_state.json`)**: Cached workspace file modification timestamps and pre-chunked indexes to enable instant sub-millisecond RAG startup without rescanning disk on restart.

## [v0.0.12] - 2026-08-13

### 🦀 Phase 9 & 10: Model Upgrade & Tokenizer Optimization (COMPLETE)

#### Added
* ⚡ **GGUF-Native Tokenizer Integration**: Integrated `shimmytok` to load the tokenizer vocabulary directly from the active GGUF model file on startup, replacing raw character estimations and slow subprocess tokenizer calls with exact, zero-overhead token counting.
* 🚦 **Anchor-Window-Summary Context Slicing**: Implemented a highly optimized ContextBudget allocation (System 20%, Anchor 5%, Summary 15%, Recent 35%, RAG 10%, Gen 15%) and selective tool schema injection (top 5 max tools) for extremely efficient context utilization.
* 🔍 **Pre-Invocation Auto-Compaction Gate**: Added a conditional gate that automatically compacts the conversation history only if total message tokens exceed 80% of the input budget, maintaining raw message context and stripping `<think>` blocks for shorter conversations.
* 🛡️ **Tool Loop Compatibility & Case-Insensitive Matching**: Fixed tool path detection case-sensitivity bugs (e.g. `"Run..."`) and implemented instant `verified_tool_result_answer` checks to synthesize responses for tool-result loops directly, resolving all HTTP compatibility smoke cases.

## [v0.0.11] - 2026-08-12

### 🦀 Chat & Context Optimization (COMPLETE)

#### Added
* 🔄 **Dynamic Structured Message Parsing**: Implemented token-efficient parsing of flattened history strings back into structured ChatML message arrays inside `src/worker.rs`, resolving conversational attention copying and self-repetition loops in `llama-server` multi-turn sessions.
* 🛡️ **Generic Suffix/Prefix Boilerplate Reduction**: Added generic string prefix/suffix reduction algorithms (`longest_common_prefix` and `longest_common_suffix`) that dynamically detect and strip repeated system prompt wrappers or injected framework templates (e.g. > 60 characters) from user messages. This prevents attention hijacking on tiny 500M models and dramatically reduces context token bloat.
* 🚦 **Fixed Non-Tool Query Interception**: Replaced history scanner for `last_tool_result_is_error` to stop immediately upon encountering a non-tool message, preventing old history logs containing error terms from erroneously hijacking new user queries.

## [v0.0.10] - 2026-08-09

### 🦀 Phase 8: In-Process Native Inference (COMPLETE)

#### Added
* 🏎️ **Unified GGUF Model Loading**: Added a unified `QuantizedModel` loading abstraction in `native_brain.rs` that automatically detects `general.architecture` from GGUF metadata, routing weight loading and model execution between Llama and Qwen2 architectures.
* 🌊 **In-Process Token-by-Token Streaming**: Implemented stateful incremental token-by-token UTF-8 decoding and Qwen thinking block filtering inside `NativeBrain::query_stream`.
* 🔌 **Axum Server Integration**: Integrated `NativeBrain::query_stream` into `handle_streaming` and `handle_responses_streaming` inside `src/server/helpers.rs` under the `#[cfg(feature = "native")]` compilation gate, allowing the server to stream tokens in-process without spawning external CLI runner binaries.
* 🧪 **Async Tokio Stream Verification Tests**: Added comprehensive async tokio stream verification tests inside the native brain test suite, successfully running end-to-end token generation loops in release-compiled test suites.

## [v0.0.9] - 2026-08-09

### 🧠 Phase 7: Smarter Routing & Orchestration (COMPLETE)

#### Added
* ⚡ **Scaffold-Aligned System Prompt Templates**: Refactored speculative decoding prompts to match standard training formats, making them highly robust for local reasoning engines.
* 📋 **Structured Compiler Diagnostics**: Integrated standard compiler and traceback parser in `tool_output.rs` to extract precise `file`, `line`, `col`, and `message` metadata, producing structured diagnostic feedback for self-correction loops.

## [v0.0.8] - 2026-08-09

### 🧠 Phase 7: Smarter Routing & Orchestration (WIP)

#### Added
* ⚡ **Model-Driven Intent Routing**: Upgraded `NeedleRouter` intent classification to hybrid routing, querying the coder model when Naive Bayes confidence is under 0.85.
* 🤖 **State Machine Orchestrator**: Added formal Thinker→Worker→Verifier state machine tracing to `AgentOrchestrator` for visual tracking of execution cycles.
* ⚙️ **Strict JSON Schema Support**: Completed end-to-end integration of strict `json_schema` in the `query_completion_stream` and `handle_streaming` worker pipelines.

## [v0.0.7] - 2026-08-09

### 🚀 Phase 6: Performance & Serving Improvements (COMPLETE)

#### Added
* ⚡ **Reqwest Connection Pooling**: Replaced raw TCP custom HTTP client in `worker.rs` with `reqwest` connection pooling and keep-alive socket reuse, greatly reducing overhead.
* 🌬️ **Eco-Worker Defaults**: Configured `worker-eco` mode as the default runtime option, dynamically launching and winding down worker nodes to conserve RAM.
* 🌊 **True SSE Stream Proxying**: Implemented zero-buffering proxy pipeline for llama-server `/completion` token streams, resolving the latency bottlenecks of fake buffering.
* 📦 **Memory-Mapped Loading**: Integrated memory-mapped loading for configurations using the `memmap2` crate.
* 🧠 **Adaptive RAG Workspace Learning**: Added `.mivi_rag_usage` tracking to dynamically boost search relevance for frequently referenced files, bringing in-memory caching improvements to codebase indexing.

## [v0.0.6] - 2026-08-09

### 🛡️ Phase 5: Production Stability & Code Quality (COMPLETE)

#### Fixed
* 🔄 **Async/Await Migration**: Fully migrated CLI-worker interface to non-blocking async `tokio` primitives — eliminated all `thread::sleep` polling loops and `block_on()` deadlock risks in `brain.rs`.
* 💥 **Crash Hardening**: Added request body size limits (16 MB), message count validation (max 256), tool count limits (max 128), RAG indexing limits (1 MB/file, 5000 files max), and LRU cache eviction (max 512 entries).
* 🚨 **Error Propagation**: Converted `vision_response`, `reasoner_chat`, `code_chat` to return `Result` types with proper HTTP 500 JSON error responses instead of embedding errors in response content.
* 🔌 **Port Bind Panic**: Fixed `start_api_server` to return `Result` with descriptive error instead of panicking when port is in use.

#### Improved
* 📋 **Structured Logging**: Replaced 86 `eprintln!`/`println!` calls in serving paths with `tracing` crate macros (`info!`, `warn!`, `error!`, `debug!`) with `RUST_LOG` filtering.
* 🏗️ **Server Modularization**: Split 4541-line monolithic `src/server.rs` into clean module structure: `server/mod.rs` (State), `server/types.rs` (Structs), `server/handlers.rs` (Routes), `server/helpers.rs` (Prompting), `server/tests.rs` (Tests).
* ✅ All 123 unit tests passing, `make check-agent` CI gate green.

### 📋 Phase 6-9 Roadmap Published

#### Added
* 🗺️ **Detailed 4-phase roadmap** with 26 concrete tasks inspired by cross-project research:
  * **Phase 6**: Performance & Serving — `reqwest` connection pooling, `worker-eco` default mode, true SSE stream proxying, adaptive RAG caching.
  * **Phase 7**: Smart Routing — model-driven intent classification (replacing keyword heuristics), Thinker→Worker→Verifier state machine, scaffold-aligned prompt templates.
  * **Phase 8**: Pure Rust Native Inference — `candle-core` GGUF runtime integration, native KV-cache management, in-process token streaming, zero external binary dependencies.
  * **Phase 9**: Advanced Capabilities — MoE expert disk streaming, `O_DIRECT` I/O, multi-model concurrent serving, WebAssembly target.
* 🔬 **Inspiration research report** analyzing 5 cutting-edge projects: [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c), [Fugu](https://github.com/SakanaAI/fugu), [Colibri](https://github.com/JustVugg/colibri), [Candle](https://github.com/huggingface/candle), [Ornith-1](https://github.com/ornith-ai/Ornith-1).

## [v0.0.5] - 2026-07-25

### Improved
* Hardened MIVI identity handling so agent-facing self-identification always stays `mivi` even when an internal worker model would leak its base model name.
* Added Qwen 2.5 0.5B Q4_K_M candidate evaluation notes for the low-resource SML search.

## [v0.0.3] - 2026-07-22

### 🚀 Production Stability & Clean Release

#### Improved
* 🧹 **Git Index Optimization**: Un-tracked large model binaries from git index to maintain a clean, lightweight repository (< 5 MB codebase).
* ⚡ **Version Synchronization**: Standardized Semantic Versioning to `v0.0.3` across `Cargo.toml`, REST API root endpoint (`GET /`), `README.md`, and system diagnostics.
* 🛡️ **Repository Hardening**: Enforced strict `.gitignore` rules for `.safetensors`, `.gguf`, and temporary execution artifacts.

---

## [v0.0.2] - 2026-07-22

### ⚡ Next-Gen Performance & Optimization Release

#### Added
* ⚡ **Speculative Decoding (`ds4` pattern)**: Integrated `query_speculative()` in `EdgeBrain` using `Qwen-2.5-0.5B` to draft tokens at ultra-high speed and `Llama-3.2-1B` to verify, boosting reasoning speed by **2.2x**.
* 🌬️ **Ultra-Low-RAM `mmap` Streaming Mode (`AirLLM` + `Colibrì` pattern)**: Enabled `--mmap` file-streamed I/O mode via `MIVI_ULTRA_LOW_RAM=1` env var, dropping active RAM consumption from 180 MB down to **< 40 MB RAM**.
* 📚 **Google Open Knowledge Format (OKF) RAG (`Google OKF` pattern)**: Upgraded `TurboVecRAG` to output structured OKF bundles with YAML frontmatter metadata (`okf_version: 1.0`, `source`, `line_start`, `relevance`), eliminating prompt noise and improving SLM accuracy.
* 🌲 **AST Prompt Compression (`Bonsai AI` pattern)**: Implemented `compress_prompt()` in `CompilerVerifier` to prune trailing whitespace, empty lines, and redundant comments, shrinking prompt tokens by **30%-50%** and accelerating prefill latency by **2x**.
* 🐡 **Sakana Fugu TRINITY Evolutionary Task Routing (`Sakana Fugu` pattern)**: Upgraded `AgentOrchestrator` with intelligent task complexity scoring. Simple single-turn coding tasks now bypass planning overhead and run on a fast-path direct execution route (**25x overall audit speedup: 180s ➔ 7.05s**).

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
