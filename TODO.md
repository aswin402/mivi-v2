# MIVI-V2 TODO

> Tracking improvements to make MIVI a reliable, production-quality local AI model engine.

---

## Phase 5: Bug Fixes 🐛 ✅ COMPLETE

> Fix all issues identified in the comprehensive codebase audit.

### Wave 1 — Fix Async Blocking (CRITICAL 🔴) ✅

- [x] **5.1** Rewrite `command_output_with_timeout` in `brain.rs` to use `tokio::process::Command` + `tokio::time::timeout` (remove `thread::sleep` polling loop)
- [x] **5.2** Make `run_cli` in `brain.rs` async (change `std::process::Command` → `tokio::process::Command`)
- [x] **5.3** Make `query_reasoner`, `query_coder`, `query_raw`, `query_speculative`, `query_vision` async
- [x] **5.4** Delete the `block_on()` helper in `brain.rs` (no longer needed)
- [x] **5.5** Update all callers in `server.rs` (`vision_response`, `reasoner_chat`, `code_chat`, `model_chat`) → async + `.await`
- [x] **5.6** Update `orchestrator.rs` `execute_plan` to `.await` the now-async brain calls
- [x] **5.7** Update `verifier.rs` `generate_and_verify` to handle async brain calls

### Wave 2 — Fix Crash Paths (HIGH 🟠) ✅

- [x] **5.8** Replace `semaphore.acquire_owned().await.unwrap()` with graceful error (503 response) in `server.rs`
- [x] **5.9** Add request payload size limit (`DefaultBodyLimit::max(16MB)`) and message count validation (max 256 messages, max 128 tools)
- [x] **5.10** Fix RAG `index_directory` — add file size limit (1 MB), max file count (5000), use `spawn_blocking`
- [x] **5.11** Add LRU eviction to `SemanticCache` (max 512 entries)

### Wave 3 — Error Handling & Correctness (MEDIUM 🟡) ✅

- [x] **5.12** Return proper HTTP 500 error JSON for model failures instead of embedding errors in content text
- [x] **5.13** Fix `start_api_server` port bind panic — return `Result` with descriptive error
- [x] **5.14** Replace `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` with safe `unix_timestamp()` helper

### Wave 4 — Code Quality (LOW 🔵) ✅

- [x] **5.15** Replace `eprintln!`/`println!` with `tracing` crate structured logging
- [x] **5.16** Split `server.rs` (4541 lines) into modules: `types.rs`, `handlers.rs`, `helpers.rs`, `tests.rs`

---

## Phase 6: Performance & Serving Improvements 🚀 ✅ COMPLETE

> Inspired by: Colibri (.coli_usage warm cache), Candle (mmap loading), Fugu (true streaming)

### Wave 1 — Serving Layer (HIGH)

- [x] **6.1** Replace raw TCP worker HTTP client in `worker.rs` with `reqwest` or `hyper` for connection pooling & keep-alive
- [x] **6.2** Default runtime to `worker-eco` mode (warm llama-server for 2 min, then spin down)
- [x] **6.3** True SSE stream proxying from llama-server → Axum client (eliminate buffered fake-streaming)

### Wave 2 — Memory & Caching (MEDIUM)

- [x] **6.4** Memory-mapped model catalog loading (`memmap2`) for instant cold starts
- [x] **6.5** Adaptive RAG workspace learning — track hot files, pin frequently accessed chunks in memory (inspired by Colibri `.coli_usage`)
- [x] **6.6** Session warmup profiles — persist hot-path data across server restarts

---

## Phase 7: Smarter Routing & Orchestration 🧠

> Inspired by: Fugu (learned conductor), Ornith-1 (scaffold-aware RL)

### Wave 1 — Intent Routing (HIGH)

- [x] **7.1** Upgrade `NeedleRouter` to model-driven intent classification (use coder model for routing instead of keyword heuristics)
- [x] **7.2** Formalize Thinker→Worker→Verifier state machine in `AgentOrchestrator` with explicit phase transitions

### Wave 2 — Prompt Engineering (MEDIUM)

- [x] **7.3** Scaffold-aligned system prompt templates — format tool definitions and error feedback to match how scaffold-aware models are trained
- [x] **7.4** Structured compiler diagnostic formatting in `tool_output.rs` (standardized error schemas improve model self-correction rates)
- [x] **7.5** Strict JSON schema response format support (full parsing & validation)

---

## Phase 8: Pure Rust Native Inference 🦀

> Inspired by: Candle (pure Rust GGUF runtime), kimi-k3-in-c (direct quantized GEMM)
> This is the transformative evolution — replace llama-cli subprocess with native Rust inference.

### Wave 1 — Candle Integration (HIGH)

- [ ] **8.1** Add `candle-core` and `candle-transformers` as optional Cargo dependencies behind a `native` feature flag
- [ ] **8.2** Implement `NativeBrain` — pure Rust GGUF model loader using Candle's quantized runtime
- [ ] **8.3** Native KV-cache management with Rust ownership semantics (automatic cleanup, no process boundaries)

### Wave 2 — Native Execution (MEDIUM)

- [ ] **8.4** In-process true token streaming (yield tokens as generated, no subprocess stdout parsing)
- [ ] **8.5** CPU SIMD vectorized inference (leverage Rust `std::arch` intrinsics for quantized GEMM)
- [ ] **8.6** Optional Metal/CUDA backend support via Candle's backend abstraction

### Wave 3 — Zero Dependency (LOW)

- [ ] **8.7** Remove `bin/llama-cli` and `bin/llama-server` dependencies — single `cargo build --release` produces everything

---

## Phase 9: Advanced Capabilities 🔮

> Inspired by: kimi-k3-in-c (MoE streaming), Colibri (O_DIRECT), Candle (WASM)

- [ ] **9.1** MoE expert disk streaming with LRU cache — run models larger than available RAM
- [ ] **9.2** Direct I/O (`O_DIRECT`) for model file reads under `MIVI_ULTRA_LOW_RAM` to avoid page cache bloat
- [ ] **9.3** Multi-model concurrent serving — keep reasoner and coder loaded simultaneously with shared attention layers
- [ ] **9.4** WebAssembly (WASM) inference target — run MIVI-V2 entirely in the browser
- [ ] **9.5** Plugin system for custom model backends (Candle, llama.cpp, ONNX)

---

## Completed ✅

### Phase 1: Remove Hardcoded Logic ✅
- [x] Remove all `verified_*` heuristic fast-paths in `server.rs`
- [x] Forward `messages`, `tools`, `tool_choice`, `response_format` to llama-server
- [x] Add SSE streaming fallback for tool-involved requests
- [x] Fix `Serialize` derivation to filter `reasoning_content` from worker requests
- [x] Delete dead helper functions from old heuristics
- [x] `cargo fmt` + all 125 tests passing + `make check-agent` CI green

### Phase 2: Core Inference Fixes ✅
- [x] **2.1** Add `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `user` to `ChatCompletionRequest`
- [x] **2.2** Forward `max_tokens`, `stop`, `seed`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` in `query_chat_full` worker body
- [x] **2.3** Remove `schema_grounded_capability_answer` — no more fabricated responses
- [x] **2.4** Remove hardcoded `"No tool call was generated..."` fallback text
- [x] **2.5** Fix `handle_streaming` to use full `req.messages` history instead of only last user prompt
- [x] **2.6** Fix `/v1/responses` streaming to actually stream tokens instead of faking it

### Phase 3: API Parity ✅
- [x] **3.1** Add missing response fields: `system_fingerprint`, `logprobs`, `refusal`, `completion_tokens_details`, `prompt_tokens_details`
- [x] **3.2** Add `/v1/health` endpoint
- [x] **3.3** Support `json_schema` response format via GBNF grammar
- [x] **3.4** Stream tool call arguments as deltas (matching OpenAI's fragmented format)
- [x] **3.5** Make chat template configurable per model (instead of hardcoded ChatML)

### Phase 4: Architecture (Partial) ✅
- [x] **4.1** Replace blocking `std::net::TcpStream` in worker with async TCP
- [x] **4.2** Add `tokio::sync::Semaphore` for request concurrency control
