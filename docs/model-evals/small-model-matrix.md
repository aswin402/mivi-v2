# Small Model Evaluation Matrix

MIVI exposes only `model: mivi` to agents. Internal model swaps must be judged by runtime behavior, not by model name marketing.

## Baseline

| Role | Current Model | File | Purpose | Status |
| --- | --- | --- | --- | --- |
| Chat/reasoning | Llama 3.2 1B Instruct IQ3_M | `models/Llama-3.2-1B-Instruct-IQ3_M.gguf` | General chat, reasoning, agent instructions | Active baseline |
| Coding | Qwen 2.5 0.5B Instruct Q2_K | `models/qwen2.5-0.5b-instruct-q2_k.gguf` | Code drafting and verifier loop | Active baseline |
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
| Qwen 2.5 small instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | pending | candidate | Test after runtime path is stable. |
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
