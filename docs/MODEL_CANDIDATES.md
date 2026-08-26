# SLM Candidate Research & Benchmark — 2026-08-26

Target profile: <2B parameters, <1000 MB RAM, fast, stable, strong tool calling,
agentic, fine-tune candidate, MoE candidate, thinking support.

Methods: internet research (HF model cards) + local measurement. Local numbers
measured on Ryzen 7 7730U (16 threads), llama.cpp b10075, Q4_K_M/Q4_0 quants,
4096 ctx, `-t 8`, 250-token generation. Agentic score = our
`scripts/eval_agent_workflows.py` suite (11 graded workflows) served through the
full MIVI pipeline (`scripts/eval_model_candidates.sh`).


| Model | Params | File size | Model RSS | tok/s | Agentic | Notes |
|---|---|---|---|---|---|---|
| **MiniCPM5-1B** (openbmb) | 1.08B dense | 656 MB | 1,124 MB | **26.4** | **6/11** | apache-2.0; standard LlamaForCausalLM; hybrid `<think>` toggle; 131k ctx; XML tool calls |
| **LFM2.5-1.2B-Instruct** (LiquidAI) | 1.17B hybrid | 697 MB | 1,259 MB | 21.7 | 5/11 | beats Qwen3-1.7B on paper (BFCLv3 49.1 vs 46.3); custom LFM1.0 license; DSpark 296M drafter = 2.1× spec-decode |
| **Qwen3.5-0.8B** (unsloth GGUF) | 0.8B | 508 MB | 903 MB | 23.1* | 4/11 | *slow through MIVI pipeline (62 s/workflow avg — long `<think>` traces); timed out once |
| **Granite-4.0-h-350m** (IBM) | 350M hybrid Mamba | 212 MB | **494 MB** | **44.0** | 4/11 | fastest + lightest; apache-2.0; tool calling supported |
| **LFM2.5-350M** (LiquidAI) | 350M hybrid | 219 MB | **438 MB** | **88.2** | **6/11** | ties 1B-class agentic at ⅓ RAM; LFM1.0 license: free incl. commercial use under $10M annual revenue, derivatives OK (same license + notices) |

| Qwen2.5-0.5B (old default) | 0.5B | ~400 MB | 599 MB | 39.1 | 3/11 | reference baseline |

Agentic workflow matrix (PASS/fail):

| Workflow | MiniCPM5 | LFM2.5 | Qwen3.5 | Granite | Qwen3-1.7B |
|---|---|---|---|---|---|
| tool-json / tool-shell-100 / web-research-url | ✅ | ✅ | ✅ | ✅ | ✅ |
| chat-injected (identity) | ✅ | ✅ | ✅ | ✅ | ✅ |
| long-tool-output | ✅ | ✅ | ❌ | ❌ | ✅ |
| memory-model-name | ✅ | ❌ | ❌ | ❌ | ✅ |
| rag-router | ❌ | ❌ | ❌ | ❌ | ✅ |
| coding-verified / stop-scheduled-job / trace-* | ❌ | ❌ | ❌ | ❌ | ❌ |

### LFM1.0 license (checked 2026-08-26)

Free for individuals and organizations under **$10M annual revenue**, including
commercial use and Derivative Works (finetunes, MoE upcycles) — redistributed
under the same license with attribution notices. Entities ≥ $10M revenue need a
separate agreement from Liquid AI. This clears LFM2.5 as a finetune/MoE base
for this project; MiniCPM5-1B (apache-2.0) remains the zero-condition default.
| Qwen3-1.7B Q2_K (current default) | 1.7B | 778 MB | 1,255 MB | 19.1 | **7/11** | slowest of the set; Q2_K quant |

Semantic eval (`eval_small_models.sh`): all candidates 4/8 with *identical*
failure reasons — those cases test MIVI pipeline behaviors, not model quality;
the suite does not discriminate candidates.

## Researched but not locally tested

| Model | Verdict for our constraints |
|---|---|
| **LFM2.5-8B-A1B** | **Best MoE candidate**: 8.3B total / 1.5B active → active-parameter RAM near our budget; needs ~5 GB disk for Q4. Test when disk/RAM budget allows |
| **LFM2.5-1.2B-Thinking** | Thinking variant of LFM2.5; same footprint as 1.2B-Instruct; swap candidate for reasoning role |
| **LFM2.5-350M** | Measured 2026-08-26: 6/11 agentic at 438 MB / 88 tok/s — the ultra-light tier default; beats Granite-350m on every axis |
| **LFM2.5-1.2B-Instruct-DSpark** (296M) | Speculative-decoding drafter; pairs with 1.2B for ~2.1× decode at identical outputs — worth wiring into our worker mode (`--model-draft`) |
| **GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking** | Community thinking finetune of MiniCPM5-1B; evidence MiniCPM5 finetunes well; evaluate vs base after our own LoRA |
| **Supra2-100M-Instruct / Supra2-Medium** | 100M-class: too small for reliable tool calls (Granite-350M already borderline); skip for agent role |
| **min-spark-1.1 / cRia-LM-75M / distilgpt2 / gpt2** | Sub-100M/legacy; no tool-calling training; not candidates for the agent role |
| **all-MiniLM-L6-v2** | Embedding model (384-dim) — not a chat candidate, but a strong candidate to upgrade `/v1/embeddings` quality over our pure-Rust hash embeddings |
| **MiniCPM-V-4.6** | Vision model; already integrated (disabled by default) |
| **Qwen3.8-27B** (DFlash2 / NVFP4-MTP variants) | 27B — violates <2B budget. The variants are speculative-decoding/quant repackagings, not distinct models |
| **Ornith-1.5-35B-A3B** | 35B MoE / 3B active — violates <2B total; active-size RAM ~3B-class. Out of budget; note as far-future MoE reference |

## Recommendations

1. **Default agent tier**: keep Qwen3-1.7B (7/11) — still the agentic leader.
2. **Best new candidate**: **MiniCPM5-1B** — 6/11 agentic, fastest dense model
   (26.4 tok/s), apache-2.0, standard Llama arch (any finetune stack works),
   hybrid thinking toggle. Best **fine-tune candidate** for Phase 17 LoRA.
3. **Speed/size tier**: **Granite-4.0-h-350m** — 44 tok/s at 494 MB for
   RAM-critical deployments (4/11 agentic = same as 0.5B at 2× speed, half RAM).
4. **Thinking role**: LFM2.5-1.2B-Thinking or MiniCPM5-1B think mode.
5. **MoE path**: LFM2.5-8B-A1B (1.5B active) is the only in-spirit MoE option.
6. **Speculative decoding**: wire LFM2.5 DSpark drafter via existing
   `MIVI_DRAFT_MODEL` plumbing for ~2× decode on the LFM2.5 pair.
7. **Embeddings upgrade candidate**: all-MiniLM-L6-v2 for `/v1/embeddings`.

Raw evidence: `model-eval-results/model-candidates-20260826-012700.jsonl`,
`agent-workflows-2026082{6}-*.jsonl`, server logs
`model-eval-results/{candidate}.server.log`.

## Finetune round 1 results (2026-08-26)

Both bases trained (QLoRA r=16, 60 steps, 182 rows) and evaluated through the
full pipeline:

| Model | Pre-FT | Post-FT | Verdict |
|---|---|---|---|
| MiniCPM5-1B | 6/11 | 6/11 | neutral |
| LFM2.5-350M | 6/11 | **5/11** | slight regression (lost long-tool-output) |
| Qwen3-1.7B default | 7/11 | — | still leads |

**Key finding — the behaviors WERE learned, but the serving format doesn't match:**

- stop-scheduled-job: both finetuned models now emit the RIGHT decision
  (`remove_job` with `id: 1`) — but as raw JSON text in `content`
  (`{"tool_calls":[...]}`) instead of the format MIVI's grammar-constrained
  tool path parses, so `tool_calls` arrives empty.
- coding-verified: models emit code + output but with `**Response:**` /
  `**Output:**` headings instead of the pipeline's `**Verified Terminal
  Output:**`.

**Root cause: train/serve prompt-format mismatch.** We finetuned on raw chat
template; MIVI serves with the agent-contract prompt + grammar-constrained
tool format. Round 2 must render training rows exactly as
`build_chat_prompt` + `generate_tool_calls` do (same agent contract, same
tool JSON block, same expected assistant output shape).

Also fix in data: use consistent tool-call `id` values (models learned to put
the function name in `id`).
