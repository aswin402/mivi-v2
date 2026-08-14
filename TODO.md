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

## Phase 7: Smarter Routing & Orchestration 🧠 ✅ COMPLETE

> Inspired by: Fugu (learned conductor), Ornith-1 (scaffold-aware RL)

### Wave 1 — Intent Routing (HIGH)

- [x] **7.1** Upgrade `NeedleRouter` to model-driven intent classification (use coder model for routing instead of keyword heuristics)
- [x] **7.2** Formalize Thinker→Worker→Verifier state machine in `AgentOrchestrator` with explicit phase transitions

### Wave 2 — Prompt Engineering (MEDIUM)

- [x] **7.3** Scaffold-aligned system prompt templates — format tool definitions and error feedback to match how scaffold-aware models are trained
- [x] **7.4** Structured compiler diagnostic formatting in `tool_output.rs` (standardized error schemas improve model self-correction rates)
- [x] **7.5** Strict JSON schema response format support (full parsing & validation)

---

## Phase 8: Pure Rust Native Inference 🦀 ✅ COMPLETE

> Inspired by: Candle (pure Rust GGUF runtime), kimi-k3-in-c (direct quantized GEMM)

### Wave 1 — Candle Integration (HIGH)

- [x] **8.1** Add `candle-core` and `candle-transformers` as optional Cargo dependencies behind a `native` feature flag
- [x] **8.2** Implement `NativeBrain` — pure Rust GGUF model loader using Candle's quantized runtime
- [x] **8.3** Native KV-cache management with Rust ownership semantics (automatic cleanup, no process boundaries)

### Wave 2 — Native Execution (MEDIUM)

- [x] **8.4** In-process true token streaming (yield tokens as generated, no subprocess stdout parsing)
- [x] **8.5** CPU SIMD vectorized inference (leverage Rust `std::arch` intrinsics for quantized GEMM)
- [x] **8.6** Optional Metal/CUDA backend support via Candle's backend abstraction

### Wave 3 — Zero Dependency (LOW)

- [x] **8.7** Remove `bin/llama-cli` and `bin/llama-server` dependencies — single `cargo build --release` produces everything

---

## Phase 9: Model Upgrade + Critical Fixes 🦀 ✅ COMPLETE

> **Goal:** Switch to 1.7B model, fix grammar bug, add KV cache optimization, speculative decoding
> **Duration:** 1 day
> **Impact:** Smarter model + 50% KV cache savings + 2-3x speed via speculative decoding

- [x] **9.1** Switch Qwen3 1.7B Q2_K as primary model (disable 0.6B entries in `configs/models.json`)
- [x] **9.2** Fix GBNF grammar — add `(think-block)?` preamble to allow Qwen3 reasoning before tool calls
- [x] **9.3** Add KV cache quantization flags (`-ctk q8_0 -ctv q8_0`) to `brain.rs` llama-cli args
- [x] **9.4** Add `MIVI_DRAFT_MODEL` env var for speculative decoding (`--model-draft`)
- [x] **9.5** Benchmark 1.7B vs 0.6B: tool accuracy, latency, RAM usage

---

## Phase 10: Exact Tokenizer + Context Optimization 🦀 ✅ COMPLETE

> **Goal:** Replace estimation with exact token counting, improve context utilization
> **Duration:** 2 days
> **Impact:** Perfect context budget, 200-500 token savings per request

### Wave 1 — Exact Tokenizer

- [x] **10.1** Add `shimmytok` crate — reads tokenizer directly from GGUF (zero external files)
- [x] **10.2** Create `src/tokenizer.rs` with global `OnceLock<Tokenizer>` singleton
- [x] **10.3** Replace `CheapTokenCounter` calls in `retrieval.rs` and `context_compressor.rs`
- [x] **10.4** Init tokenizer from active model GGUF at startup in `main.rs`

### Wave 2 — Context Improvements

- [x] **10.5** Selective context injection — only inject top 5 matching tool schemas per request
- [x] **10.6** Implement Anchor-Window-Summary architecture in `context_compressor.rs`
- [x] **10.7** Token budget slicing: system 20%, anchor 5%, summary 15%, recent 35%, RAG 10%, gen 15%
- [x] **10.8** Strip `<think>` blocks from past assistant turns in conversation history
- [x] **10.9** Pre-invocation auto-compaction gate (compress if >80% budget used)

---

## Phase 11: Knowledge-Lean Sub-1B Fine-Tuning & 64k Context Engine 🟡 IN PROGRESS

> **Goal:** Fine-tune sub-1B base model (Qwen2.5-0.5B / Qwen3-0.6B) with DeepSeek-R1 reasoning distillation, Multi-LoRA specialist adapters, and 64k context under 600 MB RAM
> **Duration:** 2-3 days (includes Colab training)
> **Impact:** 300-600 MB peak RAM + 64k context + >90%+ tool calling accuracy

### Wave 1 — Dataset Generation & Distillation Pipeline
- [x] **11.1** Build `scripts/prepare_mivi_dataset.py` with Salesforce xLAM-60k + Glaive v2 + Magpie DeepSeek-R1 `<think>` traces
- [x] **11.2** Filter dataset for knowledge-lean operations (schema adherence, syntax generation, grounded QA only)
- [x] **11.3** Format dataset into Qwen ChatML with Hermes XML and OpenAI JSON tool call standards
- [x] **11.4** Write automated dataset verification test suite in `scripts/test_prepare_mivi_dataset.py`

### Wave 2 — Google Colab Unsloth QLoRA Training Pipeline
- [x] **11.5** Create `notebooks/train_mivi_unsloth.ipynb` & `scripts/train_mivi_unsloth.py` for 4-bit QLoRA on free T4 GPU (<2.5 GB VRAM)
- [x] **11.6** Add GRPO verifiable reward functions (valid JSON, balanced `<think>` tags, schema match, anti-hallucination)
- [x] **11.7** Implement automated checkpointing and GGUF `Q4_K_M` / `Q3_K_M` export
- [x] **11.8** Create `docs/COLAB_TRAINING_GUIDE.md` with step-by-step execution instructions

### Wave 3 — 64k Context Optimization (SnapKV & YaRN Triad)
- [x] **11.9** Add YaRN RoPE scaling configuration (factor 2.0x) for 64k context in `src/worker.rs`
- [x] **11.10** Add 4-bit KV cache quantization flags (`-ctk q4_0 -ctv q4_0`) to `src/brain.rs` and `src/worker.rs`
- [x] **11.11** Implement SnapKV attention pruning in `src/context_compressor.rs` (5% anchors, 15% salient clusters, 512-token rolling window)

### Wave 4 — Multi-LoRA & Model Catalog Integration
- [x] **11.12** Register `mivi-0.5b-tool-q4_k_m.gguf` in `configs/models.json` as primary default model
- [ ] **11.13** Update `download_models.py` and `src/model_catalog.rs` for automatic loading
- [ ] **11.14** Ensure `src/tokenizer.rs` (`shimmytok`) parses vocabulary directly from the fine-tuned GGUF

### Wave 5 — Verification & Benchmarks
- [ ] **11.15** Run tool calling evaluation suite `python3 scripts/eval_tool_calling.py` (target: >90% accuracy)
- [ ] **11.16** Run HTTP compatibility suite `python3 scripts/smoke_openai_compat.py` (10/10 cases green)
- [ ] **11.17** Verify full CI gate `make check-agent` (146+ unit tests passing)
- [ ] **11.18** Measure live inference RSS memory under 64k context (verify peak RAM < 600 MB)

---

## Phase 12: Semantic RAG Upgrade 🦀 ✅ COMPLETE

> **Goal:** Replace pure keyword matching with hybrid semantic code embeddings
> **Duration:** 2 days
> **Impact:** 3-5x better code retrieval accuracy with zero external dependencies

- [x] **12.1** Pure Rust semantic embedding engine in `src/semantic_rag.rs`
- [x] **12.2** Dense vector generation with character/token n-gram hashed vectors and L2 normalization
- [x] **12.3** Create `src/semantic_rag.rs` — cosine similarity search and zero-allocation indexing
- [x] **12.4** Implement hybrid search: 0.4 × keyword + 0.6 × semantic cosine similarity
- [x] **12.5** Keep TurboVec as seamless fallback with automatic directory indexing on startup

---

## Phase 13: API & UX Improvements 🦀 ✅ COMPLETE

> **Goal:** Parity with Ollama/LM Studio features & Claude Code compatibility
> **Duration:** 2 days

- [x] **13.1** Per-request `keep_alive` parameter (Ollama-style model lifecycle)
- [x] **13.2** `mivi model fit <id>` CLI — RAM fit calculator from `/proc/meminfo`
- [x] **13.3** System prompt KV cache persistence (`--prompt-cache`)
- [x] **13.4** Anthropic `/v1/messages` endpoint adapter (Claude Code compatibility)

---

## Phase 14: Persistent Project State 🦀 ✅ COMPLETE

> **Goal:** Instant server restarts via `.mivi/project_state.json`
> **Duration:** 1 day

- [x] **14.1** Design `.mivi/project_state.json` schema (hot files, tool usage, file modification hashes)
- [x] **14.2** Write state on index, load on startup
- [x] **14.3** Skip RAG re-indexing if file modification hashes match cached state (<1ms startup)
- [x] **14.4** Track tool usage counts and hot files in `.mivi_rag_usage` for prioritization

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
