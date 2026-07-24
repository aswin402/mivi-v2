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

| Candidate | Quant | RAM RSS MB | Chat | Coding | Reasoning | Tool JSON | Context | Latency ms | Decision | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Llama 3.2 1B Instruct | IQ3_M | pending | pending | pending | pending | pending | pending | pending | baseline | Current reasoner |
| Qwen 2.5 0.5B Instruct | Q2_K | pending | pending | pending | pending | pending | pending | pending | baseline | Current coder |
| Qwen 2.5 small instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | candidate | Test after worker benchmarks |
| Qwen 3 small instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | candidate | Prioritize JSON/tool strength |
| SmolLM small instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | candidate | Check 128K/effective context behavior |
| TinyLlama-class instruct | GGUF low-bit | pending | pending | pending | pending | pending | pending | pending | candidate | Only keep if tool calling is stable |

## Eval Prompts

1. Chat: `Say who you are in one short sentence.`
2. Coding: `Write Python code that prints the sum of 2 and 3.`
3. Reasoning: `A tool failed because Cargo cache is corrupted. Explain the safest fix in two steps.`
4. Tool JSON: `Use the get_weather tool for Paris.`
5. Context: `Using the project memory, what model name should agents call?`
6. RAG: `In this codebase, what module handles intent routing?`

## Pass Criteria

- Tool JSON must parse without repair for forced tool prompts.
- Coding output must pass the existing verifier.
- Plain chat must not route to the code verifier.
- RAG answers must cite project-relevant modules when workspace context is requested.
- Worker mode must beat spawn mode on repeated text prompts before becoming default.
