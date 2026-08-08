# How Local LLM Engines Do OpenAI-Compatible APIs + Tool Calling Right

**Date:** 2026-08-08
**Scope:** llama.cpp (`llama-server`), Ollama, vLLM — OpenAI-compatible serving + tool calling with small local models.
**Goal:** Turn MIVI-V2 into a reliable OpenAI-compatible agent backend (no hardcoded answers, real tool calls, valid streaming).

---

## 1. The reference engines and what they do

### 1.1 llama.cpp `llama-server` — *already bundled inside MIVI-V2 as `bin/llama-server`*

From `tools/server/README.md` (master, fetched 2026-08-08):

- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/responses` (Responses API), `/v1/embeddings`, `/v1/messages` (Anthropic), all with SSE streaming.
- **Native tool/function calling** in `/v1/chat/completions`:
  > "OpenAI-style function calling is supported with the `--jinja` flag… See our Function calling docs for more details, supported native tool call styles (generic tool call style is used as fallback)."
  - Request fields: `tools`, `tool_choice`, `parallel_tool_calls`, `parse_tool_calls`.
  - **The chat template (Jinja) renders the tool schemas into the exact prompt format the model was trained with** — that is the entire trick. `--jinja` is enabled by default; `--chat-template` / `--chat-template-file` override.
- **Schema-constrained JSON output**:
  > "The `response_format` parameter supports both plain JSON output (e.g. `{"type": "json_object"}`) and schema-constrained JSON (e.g. `{"type": "json_schema", "schema": {...}}`)."
  - Constraint is implemented as a **grammar** — the sampler can only emit tokens that keep the JSON valid. CLI equivalents: `--json-schema`, `--grammar`, `--grammar-file` (GBNF).
- **Reasoning**: `--reasoning-format deepseek` extracts thoughts into `message.reasoning_content`; `chat_template_kwargs` (e.g. `{"enable_thinking": false}`) per request; `/v1/chat/completions/control` to end reasoning early.
- **Multi-model router mode** (start with no `-m`): one server serves many GGUF files; request `"model"` field routes; models load/unload on demand; presets via INI.
- **Built-in agent tools**: `--tools all` exposes `read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_datetime, get_info` (local-file-system agent tools; note `--cors-origins` restricted to localhost for safety).
- **Performance**: multi-slot parallel decoding (`-np`), continuous batching (default on), prompt caching (`--cache-prompt` default on), speculative decoding (`--spec-draft-model`), `--sleep-idle-seconds`, KV cache quant `-ctk/-ctv`, flash attention `-fa`.
- **Streaming is always SSE** (`data: ...` frames ending with `[DONE]`), including `delta.tool_calls` for tool calls. There is no "return a JSON object instead of a stream" path.

### 1.2 Ollama

- Native `/v1/chat/completions` OpenAI compatibility (since Feb 2024) and **tool support since Jul 2024**:
  > "Supported models will now answer with a `tool_calls` response. Tool responses can be provided via messages with the `tool` role."
- Tool support is **opt-in per model**: models advertise a `tools` capability; Ollama's docs list supported models (Llama 3.1, Mistral Nemo, Firefunction v2, Command-R+, Qwen2.5 family, …). The server refuses/blocks tools for models that don't declare the capability.
- Same backend lineage as llama.cpp: tool schemas are rendered into the prompt by the model's own chat template (Hermes-style for Qwen2.5).
- `tool_choice` + `tools` are passed straight through the OpenAI-compatible endpoint; streaming works.

### 1.3 vLLM

- **Tool calling is implemented as schema-constrained decoding** (structured outputs), not free-form prompting:
  > "`tool_choice='required'` … vLLM will use structured outputs to ensure the response matches the tool parameter object defined by the JSON schema in the `tools` parameter."
  > "Named function [tool_choice] … Arguments are guaranteed to be valid JSON conforming to the function's parameter schema."
- Backends: **xgrammar** (default), guidance (llguidance), outlines, lm-format-enforcer. XGrammar-2 (MLC): "ensured **100% valid tool-calling formats**" and is integrated into vLLM, SGLang, TensorRT-LLM, MLC-LLM.
- Structured output API: `response_format` with JSON schema, or `structured_outputs` (`json`, `choice`, `regex`, `grammar`).
- Their guidance note matches the tiny-model reality: *"We recommend ensuring that the expected output format / schema is specified in the prompt … so the model's intended generation is aligned with the schema it's being forced to generate."*

### 1.4 The distilled philosophy (what every engine above shares)

1. **Never hand-roll the prompt format.** The GGUF's embedded Jinja chat template renders system/user/tool messages in the exact format the model was trained on (Qwen2.5/3 → `<tool_call>` blocks; Hermes-style).
2. **Don't trust sampling for structure — constrain it.** Grammar / JSON-schema (or xgrammar-style) decoding makes malformed tool arguments *impossible*, which is precisely what 0.5B–0.6B models need.
3. **Only offer tools to models that were trained for them** (capability flag), and fall back gracefully to plain chat.
4. **Streaming is always SSE**, with `delta.tool_calls`; there is no valid "plain JSON in place of a stream" response.
5. **Serve inference from a long-lived batched server** (multi-slot, prompt cache, continuous batching) instead of spawning a process per request.

---

## 2. What MIVI-V2 does instead (evidence from code)

| Concern | Reference engines | MIVI-V2 current |
|---|---|---|
| Prompt format | Model's own Jinja template from GGUF | Hand-rolled `<|im_start|>` wrapping + fake `{"name":…,"arguments":…}` block inserted *before the last user turn* + agent-contract noise (`build_chat_prompt`, `src/server.rs:872`; `wrap_agent_prompt`/`agent_contract_prompt_for_tools`) |
| Tool call reliability | Grammar/schema-constrained decoding | Free-form JSON parsing of raw model output; empty output → `content: ""`, zero tool calls (`generate_tool_calls`, `src/server.rs:2928`) |
| Determinism | Temperature 0 / constrained decoding | Heuristic string matching that bypasses the model entirely and returns canned answers for a handful of phrases (`verified_reasoning_answer`, `verified_tool_call_from_request`, `verified_rag_answer_from_prompt`, `src/server.rs:1278–2335`) |
| Streaming | SSE `data:` frames + `[DONE]`, `delta.tool_calls` | Non-stream fallback returns a plain `Json(...)` object when `has_tools` but no tool call was generated (`tool_text_fallback`, `src/server.rs:3414–3446`) |
| Inference hosting | Long-lived multi-slot server | Default `spawn` mode: one `llama-cli` process per request with a raw prompt file, no grammar, no template (`brain.rs::query_raw`, `src/brain.rs:412`) |
| Worker mode | — | `llama-server` is spawned (`src/worker.rs:101`) but only ever called with a bare two-message body — **`tools`, `tool_choice`, `response_format`, `stream` are never passed to it** (`query_chat`, `src/worker.rs:146`) |

### 2.1 Root cause of "model is not working as expected"

The 0.5B/0.6B models *are* tool-capable:

- `models/qwen3-0.6b-q4_k_m.gguf` and `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` both carry Jinja chat templates that instruct tool calls as:
  `<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call><|im_end|>`
  (verified by inspecting GGUF metadata on disk).
- Qwen2.5-0.5B-Instruct officially supports function calling (Hermes-style chat template) via vLLM (`--enable-auto-tool-choice --tool-call-parser hermes`), Ollama, and transformers (Qwen2.5 release notes, 2024-09).

So the failures are **not the model** — they are:

1. **Wrong prompt format.** MIVI builds a ChatML-ish string by hand that only *resembles* the trained format and buries it under agent-contract/skills-injection noise. The model was never trained to answer that exact prompt shape.
2. **No output constraint.** Even with a right prompt, a 0.6B model will occasionally emit prose, stray tokens, or nothing. Engines make this *structurally impossible* with grammar/JSON-schema constraints; MIVI just parses and gives up.
3. **Hardcoded fast paths that hide the failure.** `verified_*` functions answer a few tuned phrases without touching the model (and hallucinate for everything else), so the tests pass while real usage breaks.
4. **Broken protocol fallbacks.** Empty text and non-SSE streaming responses break OpenAI-compatible clients.

---

## 3. Recommended architecture: make MIVI-V2 the best it can be

**Core idea:** stop reimplementing the OpenAI protocol around llama.cpp. MIVI already ships `bin/llama-server`; delegate protocol + tool parsing + constrained decoding + streaming to it, and keep MIVI as the thin layer that adds what llama-server lacks: the single `mivi` model id hiding 3 tiny models, RAG, agent tool *execution*, tracing, evals, auth.

### Phase 1 — Delete hardcoded paths, delegate to llama-server (highest impact)

1. **Remove every `verified_*` heuristic** (`src/server.rs:1278–2335`): `verified_reasoning_answer`, `verified_tool_call_from_request`, `verified_rag_answer_from_prompt`, `verified_tool_result_answer`, canned identity answers. Every request goes through the model. (User directive: no hardcoded things.)
2. **Make worker mode the default** and stop using `llama-cli` per request in the chat path. Keep `llama-cli` only for one-off CLI/task/audit invocations.
3. **Pass the OpenAI request straight through to llama-server**:
   - `tools` + `tool_choice` (default `auto`, `required` when the request asks) + `parallel_tool_calls`.
   - `response_format: {"type":"json_schema","schema":<function.parameters>}` when a tool call is expected → grammar-guaranteed valid arguments JSON.
   - `stream: true` proxied as SSE (each `data:` chunk forwarded verbatim; never synthesize a JSON-object response).
   - Let llama-server's Jinja template render the system/user/tool history — **delete `build_chat_prompt`'s hand-rolled wrapping and the agent-contract injection** for model-bound requests (keep agent framing only as a plain system-prompt line, if at all).
4. **Streaming fallback fix**: if the model still produces no tool call and no content, stream one `content` chunk (e.g. "I wasn't able to produce a response.") + `[DONE]` as SSE. Never return a bare JSON object.
5. **Non-stream empty fix**: retry once with tools stripped and a minimal prompt; if still empty, return a real error message with `finish_reason: "stop"`.

### Phase 2 — Reliability engineering for tiny models

6. **Constraint over hope**: rely on llama-server's `response_format: json_schema` + `tool_choice` (the vLLM/XGrammar approach — 100% valid tool-call formats).
7. **Sampling hygiene**: `temperature 0.1–0.2` (already), `seed`, `min_p 0.05`, `top_p 0.9` for tool paths; `--samplers` order; no `available-skills`/skills-injection noise inside the prompt.
8. **Capability gating**: mark which bundled models are tool-capable (`qwen2.5-0.5b` and `qwen3-0.6b` both are, per their templates); if a request needs tools and the active model isn't tool-capable, fall back to chat with a note (Ollama's pattern) — never a hardcoded answer.
9. **Retry ladder for tool calls**: attempt 1 = full context + `tool_choice:auto`; attempt 2 = trimmed context (only 2–3 most likely tools), `tool_choice:required`, `response_format: json_schema` of the union of parameter schemas; attempt 3 = plain chat without tools. Each attempt is real model output — no strings.

### Phase 3 — Speed and concurrency (make it feel good)

10. **Multi-slot llama-server**: `-np 2–4` (currently `-np 1`), continuous batching (default), prompt caching (default), KV `q8_0` (already set) → concurrent chat + tool calls with far lower latency than per-request `llama-cli` spawn.
11. **Speculative decoding**: draft with `qwen2.5-0.5b` → verify with `qwen3-0.6b` (`--spec-draft-model`), replacing the current hand-rolled `query_speculative` double-inference.
12. **Router mode**: run one `llama-server` without `-m`, register reasoner/coder/vision via `--models-dir`; the `mivi` model id maps to a routing choice, and each underlying model is served from a hot slot (models unload when idle).
13. **Vision**: llama-server serves the MiniCPM-V GGUF with the OpenAI-compatible multimodal API (`image_url` content type) — retire the bespoke `llama-minicpmv-cli` path for the server.

### Phase 4 — Cleanup (bugs already identified)

14. **Stop blocking RAG index of CWD at startup** (`src/main.rs:62-66`) — make it opt-in (`MIVI_INDEX_DIR`), background, or bounded.
15. Fix docs/const mismatch: `MIVI_CHAT_SYSTEM_PROMPT` is a compile-time const, not an env override (update AGENTS.md row / remove).
16. Tighten the external model surface: `/v1/models` should advertise only `mivi`; keep `coder`/`reasoner` internal or fully document them.
17. Re-run the full gate after each phase: `cargo test`, `python3 -m unittest` (4 files), `cargo fmt --check`, `make check-agent`, live probes.

### What MIVI keeps that the reference engines don't have

- One OpenAI-compatible model id (`mivi`) that internally routes chat/tool/vision to the best of three tiny models.
- Agent tool *execution* with per-tool filtering (`tool_filter.rs`) and RAG retrieval (`rag.rs`, `retrieval.rs`) — llama-server only *generates* calls; MIVI executes them.
- Tracing (`trace.rs`), eval harness (`scripts/eval_agent_workflows.py`), and the CLI subcommands.

The cleanest long-term shape: **MIVI = thin OpenAI-compatible proxy (routing, RAG, tool execution, tracing, auth) in front of one multi-slot llama-server router** — the same division of labor Ollama and vLLM use, without reimplementing their protocol layer.

---

## 4. Sources

- llama.cpp `tools/server/README.md` (master): OpenAI-compatible endpoints, tool calling, `response_format` json_schema, streaming SSE, router mode, built-in tools, spec decoding — https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp function calling docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Ollama: OpenAI compatibility (2024-02) — https://ollama.com/blog/openai-compatibility ; Tool support (2024-07) — https://ollama.com/blog/tool-support ; docs — https://docs.ollama.com/api/openai-compatibility ; `qwen2.5:0.5b-instruct` model page (tools tab) — https://ollama.com/library/qwen2.5:0.5b-instruct
- vLLM: Tool calling — https://docs.vllm.ai/en/latest/features/tool_calling.html ; Structured outputs — https://docs.vllm.ai/en/latest/features/structured_outputs.html
- XGrammar-2 (MLC): "100% valid tool-calling formats" — https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation
- Qwen2.5 release notes: tool calling via vLLM (`--tool-call-parser hermes`), Ollama, transformers — https://qwenlm.github.io/blog/qwen2.5
- Local verification: GGUF chat templates of `models/qwen3-0.6b-q4_k_m.gguf` and `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` contain `<tool_call>{"name":…,"arguments":…}</tool_call>` tool-call instructions.
