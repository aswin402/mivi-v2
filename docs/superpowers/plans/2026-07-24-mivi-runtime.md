# MIVI Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-RAM MIVI runtime that makes `mivi` reliable for external AI agents through better context selection, tool filtering, verified execution, memory, persistent workers, benchmarking, and small-model evaluation.

**Architecture:** Keep the Rust Axum server as the single OpenAI-compatible facade and expose only `model: "mivi"`. Add small, testable runtime modules behind the facade: tool filtering, context compression, OKF memory, RAG retrieval, persistent worker management, and benchmarks. Keep large models lazy-loaded and make the default path favor one active text worker plus lazy vision.

**Tech Stack:** Rust 2021, Axum, serde/serde_json, tokio, llama.cpp binaries in `bin/`, GGUF models in `models/`, local filesystem OKF memory, existing verifier and RAG modules.

## Global Constraints

- External clients must use only `model: "mivi"`.
- Target max active RAM: 1000 MB on low-end CPU, integrated GPU, or no GPU.
- Prefer one active text worker; vision must be lazy-loaded only when image input appears.
- Do not add network services beyond local loopback workers.
- 128K context is an effective context target using retrieval, memory, summaries, and compression; do not require a tiny model to hold raw 128K tokens in KV cache.
- All runtime changes must have Rust unit tests and at least one local benchmark command.

---

## File Structure

- Create: `src/runtime.rs` - runtime configuration, mode selection, and shared request budget types.
- Create: `src/tool_filter.rs` - ranks and trims provided OpenAI tools before model prompting.
- Create: `src/context_compressor.rs` - converts long agent history into protected recent turns, tool observations, summaries, and retrievable facts.
- Create: `src/okf_memory.rs` - reads and writes OKF Markdown memory bundles with YAML-like frontmatter.
- Create: `src/retrieval.rs` - merges OKF memory, workspace RAG, and recent context into a bounded prompt pack.
- Create: `src/worker.rs` - manages persistent `llama-server` processes and health checks.
- Create: `scripts/bench_runtime.sh` - measures spawn path vs worker path latency and memory.
- Create: `docs/model-evals/small-model-matrix.md` - records small-model candidates and test results.
- Modify: `src/server.rs` - call tool filter, context compressor, retrieval packer, and optional worker path.
- Modify: `src/lib.rs` - export new modules.
- Modify: `README.md` - document runtime modes, benchmark command, and roadmap status.

## Task 1: Runtime Configuration

**Files:**
- Create: `src/runtime.rs`
- Modify: `src/lib.rs`
- Test: `src/runtime.rs`

**Interfaces:**
- Produces: `RuntimeConfig::from_env() -> RuntimeConfig`
- Produces: `RuntimeMode::{Spawn, WorkerEco, WorkerHot}`
- Produces: `ContextBudget { max_input_tokens, recent_turn_tokens, retrieved_tokens, memory_tokens, tool_tokens }`

- [x] **Step 1: Write failing tests** for default mode, env override, and invalid env fallback.
- [x] **Step 2: Run** `cargo test runtime` and confirm tests fail because `src/runtime.rs` does not exist.
- [x] **Step 3: Implement** `RuntimeConfig::from_env()` with defaults: `Spawn`, 4096 raw context, 120 second idle worker timeout, 1000 MB RAM target.
- [x] **Step 4: Export** `pub mod runtime;` from `src/lib.rs`.
- [x] **Step 5: Run** `cargo test runtime` and commit `feat: add runtime configuration`.

## Task 2: Tool Filtering

**Files:**
- Create: `src/tool_filter.rs`
- Modify: `src/server.rs`
- Test: `src/tool_filter.rs`, `src/server.rs`

**Interfaces:**
- Consumes: latest normalized user prompt from `src/server.rs`.
- Produces: `filter_tools(prompt: &str, tools: &[ToolDefinition], max_tools: usize) -> Vec<ToolDefinition>`.
- Produces: `tool_score(prompt: &str, tool_name: &str, description: &str) -> f32`.

- [x] **Step 1: Write tests** where 133 OpenCode tools are reduced to exact matching tools plus a small generic fallback set.
- [x] **Step 2: Write tests** where no explicit tool request keeps tools out of the model prompt.
- [x] **Step 3: Run** `cargo test tool_filter server` and confirm tests fail.
- [x] **Step 4: Implement** token overlap scoring for name, description, and parameter keys.
- [x] **Step 5: Add hard triggers** for exact tool names and phrases like `use tool`, `call function`, `read file`, `edit file`, and `run command`.
- [x] **Step 6: Wire** `filter_tools()` into `generate_tool_calls()` so tool prompts never include all 133 tools unless `tool_choice` requires it.
- [x] **Step 7: Run** `cargo test tool_filter server` and commit `feat: filter agent tools before prompting`.

## Task 3: Context Compression

**Files:**
- Create: `src/context_compressor.rs`
- Modify: `src/server.rs`
- Test: `src/context_compressor.rs`, `src/server.rs`

**Interfaces:**
- Consumes: `Vec<ChatMessage>`.
- Produces: `CompressedContext { system: String, protected_recent: Vec<String>, tool_observations: Vec<String>, summary: String }`.
- Produces: `compress_context(messages: &[ChatMessage], budget: ContextBudget) -> CompressedContext`.

- [x] **Step 1: Write tests** proving latest user message, latest assistant answer, tool results, and explicit instructions are preserved.
- [x] **Step 2: Write tests** proving old greetings and repeated injected skill text are dropped.
- [x] **Step 3: Run** `cargo test context_compressor server` and confirm tests fail.
- [x] **Step 4: Implement** deterministic compression: keep system identity, latest real user prompt, last two turns, tool observations, code blocks, and errors.
- [x] **Step 5: Add extension point** for future model summarization without calling a model in v1.
- [x] **Step 6: Wire** compressed context into normal chat and streaming paths.
- [x] **Step 7: Run** `cargo test context_compressor server` and commit `feat: compress agent context deterministically`.

## Task 4: OKF Memory

**Files:**
- Create: `src/okf_memory.rs`
- Create: `memory/.gitkeep`
- Test: `src/okf_memory.rs`

**Interfaces:**
- Produces: `OkfMemory { id, title, kind, tags, body }`.
- Produces: `load_memory_dir(path: &Path) -> Result<Vec<OkfMemory>, String>`.
- Produces: `write_memory(path: &Path, memory: &OkfMemory) -> Result<(), String>`.

- [x] **Step 1: Write tests** for parsing Markdown with frontmatter fields `id`, `title`, `type`, and `tags`.
- [x] **Step 2: Write tests** rejecting files missing `type`, matching OKF typed knowledge rules.
- [x] **Step 3: Run** `cargo test okf_memory` and confirm tests fail.
- [x] **Step 4: Implement** a small frontmatter parser using line scanning; avoid new dependencies.
- [x] **Step 5: Add write support** for verified user preferences, project facts, and reusable tool notes.
- [x] **Step 6: Run** `cargo test okf_memory` and commit `feat: add OKF memory store`.

## Task 5: RAG Retrieval Pack

**Files:**
- Create: `src/retrieval.rs`
- Modify: `src/orchestrator.rs`
- Modify: `src/server.rs`
- Test: `src/retrieval.rs`, `src/orchestrator.rs`

**Interfaces:**
- Consumes: `CompressedContext`, `OkfMemory`, existing `TurboVecRag`.
- Produces: `RetrievalPack { prompt: String, sources: Vec<String>, estimated_tokens: usize }`.
- Produces: `build_retrieval_pack(query: &str, compressed: &CompressedContext, budget: ContextBudget) -> RetrievalPack`.

- [x] **Step 1: Write tests** proving project/codebase prompts include workspace RAG.
- [x] **Step 2: Write tests** proving simple chat does not get polluted by code chunks.
- [x] **Step 3: Run** `cargo test retrieval orchestrator` and confirm tests fail.
- [x] **Step 4: Implement** source ordering: user instruction, recent turn, tool observations, OKF memory, workspace RAG.
- [x] **Step 5: Enforce** per-source token budgets and expose source labels for debugging.
- [x] **Step 6: Run** `cargo test retrieval orchestrator` and commit `feat: build bounded retrieval packs`.

## Task 6: Persistent Workers

**Files:**
- Create: `src/worker.rs`
- Modify: `src/brain.rs`
- Modify: `src/model_process.rs`
- Modify: `src/server.rs`
- Test: `src/worker.rs`

**Interfaces:**
- Consumes: `RuntimeConfig`.
- Produces: `WorkerManager::ensure_text_worker(&self) -> Result<WorkerEndpoint, String>`.
- Produces: `WorkerManager::stop_idle_workers(&self) -> Result<(), String>`.
- Produces: fallback to existing `llama-cli` path on worker failure.

- [x] **Step 1: Write tests** for generated `llama-server` command arguments without launching a real worker.
- [x] **Step 2: Write tests** for worker states: stopped, starting, ready, failed, idle-stopped.
- [x] **Step 3: Run** `cargo test worker` and confirm tests fail.
- [x] **Step 4: Implement** one text worker first using `bin/llama-server` on `127.0.0.1`.
- [x] **Step 5: Add** `worker-eco` mode: lazy start and idle stop after `MIVI_WORKER_IDLE_SECS`.
- [x] **Step 6: Add** `worker-hot` mode: keep one text worker warm.
- [x] **Step 7: Keep vision** on the current lazy CLI path until text worker is stable.
- [x] **Step 8: Add fallback** to spawn-per-request if worker health check fails.
- [x] **Step 9: Run** `cargo test worker` and commit `feat: add persistent text worker manager`.

## Task 7: Benchmark Script

**Files:**
- Create: `scripts/bench_runtime.sh`
- Modify: `README.md`

**Interfaces:**
- Produces: shell script that records first token latency, total latency, RSS, mode, model, and prompt type.

- [x] **Step 1: Add benchmark prompts** for chat, coding, tool call, RAG question, and vision skip case.
- [x] **Step 2: Measure spawn mode** with `MIVI_RUNTIME_MODE=spawn`.
- [x] **Step 3: Measure worker eco mode** with `MIVI_RUNTIME_MODE=worker-eco`.
- [x] **Step 4: Measure worker hot mode** with `MIVI_RUNTIME_MODE=worker-hot`.
- [x] **Step 5: Save results** to `benchmarks/runtime-YYYYMMDD-HHMMSS.jsonl`.
- [x] **Step 6: Document** the exact benchmark command in README.
- [x] **Step 7: Commit** `chore: add runtime benchmark script`.

## Task 8: Small Model Evaluation

**Files:**
- Create: `docs/model-evals/small-model-matrix.md`
- Create: `scripts/eval_small_models.sh`

**Interfaces:**
- Produces: repeatable eval rows for chat, coding, reasoning, tool calling, instruction following, and context handling.

- [x] **Step 1: Record baseline models**: current Llama 3.2 1B, Qwen 2.5 0.5B Coder, MiniCPM-V 4.6.
- [ ] **Step 2: Test candidate text models** only after runtime path is stable.
- [x] **Step 3: Prioritize candidates**: Qwen 2.5/3 small instruct, SmolLM small instruct, TinyLlama-class models, and any model with native tool/JSON strength under RAM budget.
- [x] **Step 4: Score** quality, latency, RAM, tool-call JSON validity, and failure mode.
- [x] **Step 5: Keep the winner** behind `model: "mivi"`; do not expose internal model names.
- [x] **Step 6: Commit** `docs: add small model evaluation matrix`.

## Task 9: Release Gate

**Files:**
- Modify: `README.md`
- Modify: `docs/API_REFERENCE.md`
- Modify: `docs/AGENTS_GUIDE.md`

**Interfaces:**
- Produces: release notes for runtime modes and agent usage.

- [x] **Step 1: Run** `cargo fmt --check`.
- [x] **Step 2: Run** `cargo test`.
- [x] **Step 3: Run** `cargo build --release`.
- [x] **Step 4: Run** `scripts/bench_runtime.sh`.
- [x] **Step 5: Start server** and verify `/v1/models` returns only `mivi`.
- [x] **Step 6: Test OpenCode** with a plain chat prompt, a coding prompt, and a forced tool prompt.
- [x] **Step 7: Update docs** with measured numbers, not estimates.
- [x] **Step 8: Commit** `docs: publish runtime validation results`.

## Todo Plan

- [x] v0.0.4 docs/version release: version bump, README update, runtime implementation plan.
- [x] Runtime config module.
- [x] Tool filtering module.
- [x] Context compression module.
- [x] OKF memory module.
- [x] Retrieval pack module.
- [x] Persistent text worker.
- [x] Benchmark script.
- [x] Small-model eval scripts and matrix.
- [x] Docs release gate with measured results.
- [x] Typed command-output compression for agent tool observations.

## Additional Needed Work

- Add prompt injection filtering for agent-supplied skill/tool metadata before it reaches the small model.
- Add strict JSON repair and validation for tool calls before returning OpenAI-compatible responses.
- Add execution audit logs for verified code paths and tool decisions.
- Add per-request debug traces behind `MIVI_TRACE=1` so OpenCode failures can be diagnosed without printing huge prompts.
- Add memory privacy controls before long-term OKF memory is enabled by default.

## Self-Review

- Spec coverage: runtime, tool filtering, context compression, OKF memory, RAG retrieval, persistent workers, benchmark script, small-model testing, and extra needed work are covered.
- Placeholder scan: no task depends on undefined future work; each task has concrete files, interfaces, tests, and commands.
- Type consistency: shared types are introduced in `src/runtime.rs` and consumed by later modules.
