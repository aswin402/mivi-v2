# AGENTS.md

Guidance for working in the **MIVI-V2** repository — a 100% Pure Rust, low-resource local AI engine that exposes a single OpenAI-compatible model name (`mivi`) while internally routing to small GGUF models via llama.cpp binaries.

## Essential commands

```bash
# Build / test / format (Rust)
cargo build --release
cargo test                     # all tests are inline #[cfg(test)] unit modules; no tests/ dir
cargo fmt --check              # CI enforces formatting (run `cargo fmt` before pushing)
cargo clippy                   # optional, not in CI

# Run the server (default mode; `serve` is also the default subcommand)
cargo run --release -- serve

# CLI subcommands
cargo run --release -- audit          # end-to-end health audit (actually invokes the models)
cargo run --release -- cli            # interactive chat
cargo run --release -- chat           # chat with orchestrator router
cargo run --release -- task "prompt"  # single code task (plan -> generate -> verify -> execute)
cargo run --release -- model list                  # show internal model catalog
cargo run --release -- model inspect <internal-id> # detail one catalog entry

# Python script unit tests (run from scripts/)
python3 -m unittest test_check_agent_compat.py test_smoke_openai_compat.py \
    test_eval_agent_workflows.py test_score_eval.py test_eval_tool_calling.py \
    test_prepare_mivi_dataset.py

# One-command compatibility gate (this is what CI runs)
make check-agent          # = scripts/check_agent_compat.py --live off
make check-agent-live     # = --live auto (adds HTTP smoke checks if a server is reachable)

# HTTP smoke tests against a running server
python3 scripts/smoke_openai_compat.py

# Benchmarks / evals (all write JSONL results; exit non-zero on failure)
scripts/bench_runtime.sh                       # runtime mode comparison -> benchmarks/runtime-*.jsonl
scripts/eval_small_models.sh                   # small-model semantic evals -> model-eval-results/
MIVI_TRACE=1 scripts/eval_agent_workflows.py   # simulated OpenCode agent traffic
bash scripts/eval_model_candidates.sh          # internal GGUF candidate comparison
```

CI (`.github/workflows/agent-compat.yml`) runs `scripts/check_agent_compat.py --live off`, which sequentially runs: python unit tests → `cargo test --quiet` → `cargo fmt --check` → `cargo build --release`.

Model weights (~3.3 GB total) and llama.cpp binaries are **gitignored** — `models/*.gguf`, `bin/`, `*.jsonl`, `memory.db`. Download models with `uv run --with huggingface_hub python3 download_models.py`. A dev machine needs `bin/llama-cli` + shared libs and the GGUF files in `models/` for anything except the pure-Rust unit tests.

## Repository layout

- `src/main.rs` — subcommand dispatcher: `serve` (default), `audit`, `cli`, `chat`, `task`, `model`.
- `src/lib.rs` — module list; adding a module requires adding it here.
- `src/server.rs` — **5800+ lines**; the OpenAI-compatible API surface. Almost all agent-facing logic lives here (not in a framework router): tool-call parsing/validation, verified fast-path answers, streaming, usage accounting, `/v1/responses` mapping.
- `src/brain.rs` — `EdgeBrain`: subprocess wrapper around `llama-cli` / `llama-mtmd-cli`; response cleaning; Qwen3 think-mode directives.
- `src/worker.rs` — `WorkerManager`: persistent `llama-server` worker for `worker-eco`/`worker-hot` modes (port 18080 by default).
- `src/runtime.rs` — `RuntimeConfig` / `ContextBudget` from env.
- `src/router.rs` — `NeedleRouter`: keyword/heuristic intent classifier (`CHAT`/`VISION`/`CODE`/`MULTI_STEP`).
- `src/orchestrator.rs` — `AgentOrchestrator::execute_plan`: routes conversational prompts to the reasoner; otherwise plans (simple → direct coder fast-path; complex → reasoner-generated JSON plan) and runs `CompilerVerifier::generate_and_verify` per step.
- `src/verifier.rs` — double-loop code generation/execution/correction (python3, node/bun, rustc, g++), up to 3 fix attempts.
- `src/context_compressor.rs` — compresses long histories into recent turns + typed observations before prompting.
- `src/tool_filter.rs` — selects a small relevant subset from 100+ tool payloads (tag/score based).
- `src/tool_output.rs` — minifies build/test/git output to salient error lines.
- `src/retrieval.rs` — assembles the `RetrievalPack` (compressed context + OKF memory + gated RAG) under a token budget.
- `src/okf_memory.rs` — OKF (Open Knowledge Format) memory: `memory/*.md` files with `---` frontmatter (`id`, `title`, `type`, optional `tags`).
- `src/rag.rs` — `TurboVecRAG`: keyword line-scoring over the workspace (25-line chunks, skip-lists generated artifacts).
- `src/trace.rs` + `MIVI_TRACE=1` — per-request JSONL trace rows appended to `logs/mivi-trace.jsonl` (or `MIVI_TRACE_PATH`).
- `src/model_catalog.rs` — typed model catalog loader; `configs/models.json` drives `mivi model list/inspect` and token-counter config.
- `configs/models.json` — internal model catalog (external name `mivi` + reasoner/coder/vision entries with `enabled` flags).
- `configs/capabilities.json` — tool taxonomy/aliases, error markers, error-category priority (used by tool summary and trace logic).
- `docs/` — `ARCHITECTURE.md` (partially stale: says 8192 default context; the runtime default is 3072), `AGENTS_GUIDE.md` (external-agent integration, not agent rules), `API_REFERENCE.md`, `OPENZ_INTEGRATION.md`.
- `scripts/` — Python eval/smoke/compat tooling with matching `test_*.py` unit tests; `bench_runtime.sh`, `eval_small_models.sh`, `eval_model_candidates.sh`.

## Request flow (server)

`handle_chat_completions` → `complete_chat_non_stream` (or streaming path) → ordered checks:

1. `validate_response_format` (json_object supported; strict json_schema rejected).
2. `verified_tool_result_answer` — synthesize an answer directly from the trailing tool result (no model call) when the conversation is a tool-result loop.
3. Tool involvement → `generate_tool_calls` (validates calls against the *selected* tool set, repairs malformed argument strings only when they normalize to valid JSON, rejects unknown tools and repeated failed calls).
4. `extract_content` → vision path if an image is attached.
5. Verified fast-path answers (no model load): tool inventory → identity → memory → reasoning → RAG.
6. Otherwise route by requested model: `coder` / `reasoner` / direct-reasoner intent / orchestrator.

There is a parallel `/v1/responses` endpoint (`handle_responses`) that maps Responses API requests onto the chat pipeline. Streaming exists for both. Usage metadata is included; token counts default to a cheap estimator (`CheapTokenCounter`) unless `MIVI_TOKENIZER_CMD` (+ optional `MIVI_TOKENIZER_MODEL`) points to a llama.cpp tokenizer.

## Environment variables (all `MIVI_*`)

| Var | Effect |
| --- | --- |
| `MIVI_RUNTIME_MODE` | `spawn` (default) \| `worker-eco` \| `worker-hot` |
| `MIVI_CONTEXT_BUDGET` | max input tokens (default 3072; floors at 1024). `ContextBudget` derives recent/retrieval/memory/tool slices |
| `MIVI_WORKER_IDLE_SECS` | worker idle timeout (default 120) |
| `MIVI_WORKER_PORT` | worker server port (default 18080) |
| `MIVI_REASONER_MODEL` / `MIVI_CODER_MODEL` / `MIVI_VISION_MODEL` / `MIVI_VISION_PROJECTOR` | GGUF path overrides |
| `MIVI_REASONER_CONTEXT_SIZE` / `MIVI_CODER_CONTEXT_SIZE` | per-model context override (min 1024) |
| `MIVI_REASONING_MODE` | `auto` (default) \| `think` \| `no_think` — prepends `/think` or `/no_think` to reasoner prompts; `auto` is conservative for agents |
| `MIVI_ULTRA_LOW_RAM` | `1`/`true` → `-ngl 0`, reduced contexts, `--mmap` |
| `MIVI_CLI_TIMEOUT_SECS` | llama-cli subprocess timeout (default 180) |
| `MIVI_TRACE` / `MIVI_TRACE_PATH` | enable JSONL request traces |
| `MIVI_TOKENIZER_CMD` / `MIVI_TOKENIZER_MODEL` | tokenizer-backed usage counts (falls back to estimator) |
| `MIVI_AGENT_REASONING_SUMMARY` | set to `0`/`false`/`off`/`no` to disable the `reasoning_content` summary (on by default) |
| `MIVI_API_KEY` | Bearer Authorization key for API auth (disabled/public if unset) |
| `MIVI_MAX_CONCURRENT_REQUESTS` | Max concurrent requests allowed to be handled by the server (default 2) |
| `MIVI_SMOKE_BASE_URL`, `MIVI_EVAL_SERVER_URL`, `MIVI_EVAL_TIMEOUT` | script-side URLs/timeouts |


## Conventions

- **External model name is always `mivi`.** Never configure agents with `qwen`/`llama`/`minicpm`; internal models are an implementation detail. The server also honors `coder`/`reasoner` as requested model names (documented as internal-testing only).
- All Rust tests are inline `#[cfg(test)] mod tests` at the bottom of each file — no `tests/` directory. Env-var-dependent tests serialize with a shared `env_lock()` (`OnceLock<Mutex<()>>`) because `std::env` is process-global and `cargo test` runs tests in parallel.
- Python scripts are stdlib-only (`urllib`, `unittest`) — no pip dependencies, so tests run in CI without installs.
- Naming: snake_case functions/modules, PascalCase types, `mivi`-prefixed env vars, JSONL output files stamped `YYYYMMDD-HHMMSS` under `model-eval-results/` or `benchmarks/`.
- Qwen3 thinking blocks (`<think>…</think>`, `[start thinking]…`) must be stripped before responses reach agents (`strip_think_blocks` in brain.rs, plus streaming strip in model_process.rs). If you add a new way to surface model output, reuse those.
- Tool-call output is validated against the *selected* tool set, not the full request payload. Keep that invariant when editing `generate_tool_calls`.

## Gotchas

- `server.rs` is the monolith; unrelated-looking behavior (verified answers, tool heuristics, trace metadata) is all in there. Search it before assuming a feature lives elsewhere.
- `*.jsonl` is globally gitignored, so benchmark/eval outputs never show up in `git status` — that is expected.
- `docs/ARCHITECTURE.md` is stale in places (context sizes, component list). Prefer `README.md`, `docs/AGENTS_GUIDE.md`, and the code itself.
- The vision model (`minicpm-vision`) is `enabled: false` in `configs/models.json` by default, but `EdgeBrain::query_vision` uses `MIVI_VISION_MODEL`/`MIVI_VISION_PROJECTOR` paths directly, independent of the catalog flag.
- `main.rs` indexes the current working directory into RAG at startup (`orchestrator.rag.index_directory`), so startup behavior depends on the working directory.
- Worker modes spawn `llama-server` subprocesses (~850–950 MB RSS); keep RAM targets in mind when testing runtime modes on shared machines.
- `scripts/check_agent_compat.py --live on` is the full gate (adds HTTP smoke + trace evals); CI uses `--live off`. Trace-backed evals need the server started with `MIVI_TRACE=1`.
