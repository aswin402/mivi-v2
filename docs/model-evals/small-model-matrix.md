# Small Model Evaluation Matrix

MIVI exposes only `model: mivi` to agents. Internal model swaps must be judged by runtime behavior, not by model name marketing.

## Baseline

| Role | Current Model | File | Purpose | Status |
| --- | --- | --- | --- | --- |
| Chat/reasoning | Qwen3 0.6B Q4_K_M | `models/qwen3-0.6b-q4_k_m.gguf` | General chat, reasoning, agent instructions | Active default |
| Coding | Qwen 2.5 0.5B Instruct Q4_K_M | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Code drafting, tool JSON, and verifier loop | Active default |
| Vision | MiniCPM-V 4.6 Q4_K_M + mmproj | `models/MiniCPM-V-4.6-Q4_K_M.gguf` | Image understanding | Lazy baseline |

## Candidate Rules

- Keep total active runtime under 1000 MB RAM.
- Prefer models with stable instruction following and valid JSON/tool-call behavior.
- Do not expose candidate names through `/v1/models`; clients still see only `mivi`.
- Runtime improvements must be tested before model replacement.

## Scorecard

| Candidate | Quant | RAM RSS MB | Chat | Coding | Reasoning | Tool JSON | Context | RAG | Latency ms | Decision | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Llama 3.2 1B Instruct | IQ3_M | 0 spawn / 849 worker | pass | n/a | pass | pass via tool prompt | pass | pass | 33-3824 | keep | Final eval `model-eval-results/small-model-20260724-232007.jsonl`; Cargo-cache maintenance prompts now use a verified safe playbook before model fallback. |
| Qwen 2.5 0.5B Instruct | Q2_K | included in spawn verifier | n/a | pass | n/a | n/a | n/a | n/a | 1309 | keep | Coding passed with verifier repair fallback; output executed and printed `5`. |
| Qwen 2.5 0.5B Instruct | Q4_K_M | pending RSS | pass | pass | pass | pass | pass | pass | 0-1754 eval rows | passed candidate | `model-candidates-20260725-220000.jsonl`; passed all agent workflow and small-model eval rows after v0.0.5 identity and deterministic single-tool guards. |
| Qwen3 0.6B | Q4_K_M | 935 MB at 3K / 996 MB at 4K / 1240 MB at 8K direct CLI; 992 MB worker | pass | n/a | pass | pass via guard | pass | pass | 0-6066 eval rows at 3K | active reasoner default | `model-candidates-20260726-013901.jsonl`; passed as reasoner with Qwen2.5 0.5B Q4 coder at `MIVI_CONTEXT_BUDGET=3072`. Do not use as coder yet: all-text candidate failed coding by outputting `Hello, World!` instead of `5`. |
| Qwen 2.5 1.5B Instruct | Q2_K | 878 spawn peak observed | timeout | n/a | pass via verified guard | fail | pass | pass | 45848 total candidate run | reject | `model-candidates-20260725-204613.jsonl`; timed out on agent chat/tool-json under 30s and produced malformed weather tool JSON in small eval. |
| Qwen 3 small instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | pending | candidate | Prioritize JSON/tool strength. |
| SmolLM small instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | pending | candidate | Check 128K/effective context behavior. |
| TinyLlama-class instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | pending | candidate | Only keep if tool calling is stable. |

## Eval Prompts

1. Chat: `Say who you are in one short sentence.`
2. Coding: `Write Python code that prints the sum of 2 and 3.`
3. Reasoning: `A tool failed because Cargo cache is corrupted. Explain the safest fix in two steps.`
4. Tool JSON: `Use the get_weather tool for Paris.`
5. Context: `Using the project memory, what model name should agents call?`
6. Tool Shell: `Run npm test.` with `bash` plus 100 irrelevant tools
7. RAG: `In this codebase, what module handles intent routing?`

## Latest Baseline Eval

Measured on 2026-07-24 with `scripts/eval_small_models.sh` in `spawn` mode after verified answer guards, compact RAG evidence extraction, and semantic scoring.

| Kind | Result | Latency | Notes |
| --- | --- | ---: | --- |
| Chat | pass | 3824 ms | Semantic score `1.0`; identifies as MIVI and keeps external model name stable. |
| Coding | pass | 1309 ms | Semantic score `1.0`; verified Python execution produced `5`. |
| Reasoning | pass | 35 ms | Semantic score `1.0`; Cargo-cache corruption prompts return a verified two-step repair and warn not to delete project manifests. |
| Tool JSON | pass | 3312 ms | Semantic score `1.0`; produced one valid `get_weather` tool call with `city: Paris`. |
| Tool Shell | pass | 33 ms | Semantic score `1.0`; selected `bash` with `{"cmd":"npm test"}` from 101 tools. |
| Context | pass | 34 ms | Semantic score `1.0`; verified memory answer returns external model name `mivi`. |
| RAG | pass | 37 ms | Semantic score `1.0`; verified RAG answer returns `src/router.rs` / `NeedleRouter::classify_intent`. |

Raw output: `model-eval-results/small-model-20260724-232007.jsonl`.

## Pass Criteria

- `scripts/eval_small_models.sh` writes `semantic_ok`, `score`, and `reasons` for each row and exits non-zero on semantic failures unless `MIVI_EVAL_ALLOW_FAILURES=1` is set.
- Tool JSON must parse without repair for forced tool prompts.
- 100-tool shell eval must select the shell tool and preserve the requested command, not hallucinate a different command.
- Coding output must pass the existing verifier.
- Plain chat must not route to the code verifier.
- RAG answers must cite project-relevant modules when workspace context is requested.
- Worker mode must beat spawn mode on repeated text prompts before becoming default.

## Agent Workflow Eval

Measured on 2026-07-25 with `MIVI_TRACE=1 python3 scripts/eval_agent_workflows.py` against the local `mivi` server. All 8 OpenCode-style workflow rows passed, including injected skill metadata, 100+ tool payloads, long tool output summary, RAG, memory, and trace validation.

Raw output: `model-eval-results/agent-workflows-20260725-190308.jsonl`.

## Candidate Runner

Use `bash scripts/eval_model_candidates.sh` to start MIVI per candidate, run agent workflow evals plus small-model evals, and write `model-eval-results/model-candidates-YYYYMMDD-HHMMSS.jsonl`. Set `MIVI_CANDIDATES_FILE` to a JSONL file with `name`, `reasoner`, and `coder` fields for additional GGUF candidates.

## Candidate Comparison Run

Measured on 2026-07-25 with `bash scripts/eval_model_candidates.sh`. Baseline `Llama-3.2-1B-Instruct-IQ3_M` reasoner plus `Qwen2.5-0.5B-Instruct-Q2_K` coder passed both eval suites.

Summary: `model-eval-results/model-candidates-20260725-200233.jsonl`.
Agent workflows: `model-eval-results/agent-workflows-20260725-200234.jsonl`.
Small-model eval: `model-eval-results/small-model-20260725-200253.jsonl`.

## Candidate Timeout Guard

`MIVI_CLI_TIMEOUT_SECS` bounds each `llama-cli` subprocess. The Qwen 2.5 1.5B Q2_K candidate used `cli_timeout_secs: 30` and was rejected because tool JSON was slow/malformed despite fitting under the rough 1000 MB process RSS target.

## Qwen 0.5B Q4_K_M Candidate

Measured on 2026-07-25 with `MIVI_CANDIDATES_FILE=/tmp/mivi-qwen05-q4-candidates.jsonl bash scripts/eval_model_candidates.sh`.

Summary: `model-eval-results/model-candidates-20260725-214644.jsonl`.
Agent workflows: `model-eval-results/agent-workflows-20260725-214645.jsonl`.
Small-model eval: `model-eval-results/small-model-20260725-214651.jsonl`.

Final v0.0.5 result: `model-candidates-20260725-220000.jsonl` passed all candidate checks in 9173 ms total. Agent workflows: `agent-workflows-20260725-220005.jsonl`. Small-model eval: `small-model-20260725-220008.jsonl`. Earlier raw runs showed identity leakage and malformed JSON were the weak points; v0.0.5 handles those with verified identity and deterministic single-tool guards.

## Qwen3 0.6B Q4_K_M Reasoner Candidate

Measured on 2026-07-25 with `MIVI_CANDIDATES_FILE=/tmp/mivi-qwen3-candidates.jsonl bash scripts/eval_model_candidates.sh` after downloading `models/qwen3-0.6b-q4_k_m.gguf` from `Antigma/Qwen3-0.6B-GGUF`.

Initial summary: `model-eval-results/model-candidates-20260725-232346.jsonl`.
4K context summary: `model-eval-results/model-candidates-20260725-234518.jsonl`.
3K context summary: `model-eval-results/model-candidates-20260726-013901.jsonl`.
Promoted default summary: `model-eval-results/model-candidates-20260726-014313.jsonl`.

Results:

| Candidate | Role | Result | Notes |
| --- | --- | --- | --- |
| `qwen3-06b-reasoner-qwen25q4-coder` | Qwen3 reasoner + Qwen2.5 Q4 coder | pass | Agent workflows and small-model evals fully passed. Long tool-output reasoning was the slowest row at 7072 ms. |
| `qwen3-06b-reasoner-qwen25q4-coder-4k` | Qwen3 reasoner + Qwen2.5 Q4 coder, `MIVI_CONTEXT_BUDGET=4096` | pass | Agent workflows and small-model evals fully passed in `model-candidates-20260725-234518.jsonl`; long tool-output reasoning was the slowest row at 6522 ms. |
| `qwen3-06b-reasoner-qwen25q4-coder-3k` | Qwen3 reasoner + Qwen2.5 Q4 coder, `MIVI_CONTEXT_BUDGET=3072` | pass | Agent workflows and small-model evals fully passed in `model-candidates-20260726-013901.jsonl`; direct CLI peak RSS dropped to 935080 KB. |
| `default-qwen3-qwen25q4` | Built-in default split | pass | Agent workflows and small-model evals fully passed in `model-candidates-20260726-014313.jsonl`; total run 17965 ms. |
| `qwen3-06b-q4-alltext` | Qwen3 reasoner + Qwen3 coder | fail | Chat, reasoning, tools, context, and RAG passed, but coding failed by producing `Hello, World!` instead of printing `5`. |

Decision: keep Qwen3 0.6B Q4_K_M as the next thinking/reasoner candidate, but keep Qwen2.5 0.5B Q4_K_M as coder/tool model. Before making Qwen3 the default reasoner, measure RSS in `spawn`, `worker-eco`, and `worker-hot`.

## Qwen3 Reasoner Runtime Benchmark

Measured on 2026-07-26 with the built-in defaults: Qwen3 0.6B Q4_K_M reasoner, Qwen2.5 0.5B Q4_K_M coder, and 3072 raw context budget.

Benchmark output: `benchmarks/runtime-20260726-014455.jsonl`.

| Mode | Chat | Coding | Tool | RAG | Vision Skip | Peak Process Tree RSS | Peak Worker RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spawn` | 4279 ms | 2048 ms | 15 ms | 24 ms | 6084 ms | 7.3 MB server-side only | 0 MB |
| `worker-eco` | 3141 ms | 1171 ms | 9 ms | 9 ms | 4954 ms | 939.5 MB | 932.1 MB |
| `worker-hot` | 2921 ms | 1437 ms | 10 ms | 18 ms | 4303 ms | 938.5 MB | 931.2 MB |

Direct `llama-cli` peak RSS for Qwen3 0.6B Q4_K_M:

| Raw context | Peak RSS | Decision |
| ---: | ---: | --- |
| 8192 | 1239716 KB | over target |
| 4096 | 995756 KB | barely under target |
| 3072 | 935080 KB | recommended cap |

Decision: Qwen3 0.6B Q4_K_M is behaviorally good as reasoner. Use a 3072 raw context cap for practical RAM headroom; 4096 is too close to the 1000 MB ceiling and 8192 exceeds it. Keep MIVI's effective 128K through context compression, OKF memory, and RAG. `q4_k_s` and `q4_0` were tested for disk-size/RAM savings, but they peaked slightly above 1000 MB at 4096 context and were rejected locally.

## Qwen3 Reasoning Mode Control

Measured on 2026-07-26 with `MIVI_REASONING_MODE=auto bash scripts/eval_model_candidates.sh` after adding conservative Qwen3 reasoning directives and thought stripping.

Summary: `model-eval-results/model-candidates-20260726-015412.jsonl`.
Agent workflows: `model-eval-results/agent-workflows-20260726-015421.jsonl`.
Small-model eval: `model-eval-results/small-model-20260726-015428.jsonl`.

Result: all rows passed with zero `<think>`, `</think>`, `[Start thinking]`, or `[End thinking]` leakage. `auto` uses `/no_think` for normal agent prompts and reserves `/think` for explicit deep-reasoning prompts; manual overrides are `MIVI_REASONING_MODE=think` and `MIVI_REASONING_MODE=no_think`.
