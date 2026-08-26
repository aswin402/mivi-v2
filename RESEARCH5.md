# MIVI-V2 Research 5 — Finetune → Specialist → MoE Pipeline

> **Date:** August 26, 2026
> **Goal:** Validate and refine the plan — finetune both selected bases
> (MiniCPM5-1B, LFM2.5-350M) for thinking / reasoning / chat-English /
> tool calls / orchestrator / agentic behaviors → pick the winner →
> build MoE specialists on top of it.

---

## 1. Verdict on the plan

**The idea is sound and matches published state-of-the-art.** The exact
pipeline — dense finetune → specialist experts → MoE — is an active research
area with multiple proven recipes (below). Two refinements based on the
research:

1. **You don't need weight-level MoE to start.** Adapter-level MoE
   (LoRA experts + a learned router on a *frozen* dense base) trains on a
   single free Colab GPU and captures most of the specialization benefit.
   Weight-level upcycling is phase 2, after the LoRA-MoE proves routing value.
2. **"Orchestrator" is a trainable skill too.** The orchestrator's JSON
   step-planning output is just another format — add plan-generation examples
   to the SFT set (prompt → JSON plan with steps/tools), same as tool calls.

## 2. What each finetune category means concretely

| Category | Training signal | Source data |
|---|---|---|
| Thinking / reasoning | `<think>…</think>` traces before answers (MiniCPM5 supports hybrid toggle) | Distill from MiniCPM5's own think mode on our prompts; keep short (2-4 sentences) to protect latency |
| Chat / English | Concise correct English Q&A | Small anchor set (already in `build_agentic_sft.py`); expand with everyday prompts |
| Tool calls | Single-call selection **with distractors**; exact argument extraction | `build_agentic_sft.py` tool_selection rows; scale variety (more tools, more phrasings) |
| Tool-result use | Aggregate "Tool results" summaries naming each tool + salient facts; error-line extraction | tool_result_summary / error_summary rows; add more tool combos |
| Orchestrator | prompt → JSON step plan (`{"steps":[{"description","lang"}]}`) matching orchestrator.rs's planner schema | Synthesize from verified pairs: decompose multi-step tasks into plans whose steps verify |
| RAG grounding | Answer only from provided context + cite source file | rag_grounded rows with more files/topics |
| Identity / stability | Deterministic persona; never leak internal names | identity rows (already deterministic server-side too) |

## 3. Prior art — inspirations and what to steal

### 3.1 PESC — Parameter-Efficient Sparsity Crafting (EMNLP 2024, arXiv:2401.02731)
**The closest published match to our plan.** Recipe: take a dense model,
upcycle FFNs into experts initialized from the dense weights, attach LoRA to
each expert, train with instruction data + load-balancing loss. Result: MoE
that beats the dense base on instruction tuning with ~same active params.
**Steal:** expert-init-from-dense + per-expert LoRA + load-balancing loss.

### 3.2 MixLoRA (TUDB-Labs, arXiv:2404.15159)
**Build a LoRA-based MoE on a SINGLE GPU with the dense model frozen.** Router
+ multiple LoRA experts per FFN layer, trained with standard SFT data. This is
the free-Colab-feasible version of our MoE phase — no continued pretraining
required. Has a pip package (`mixlora`).
**Steal:** this is our Phase C implementation shortcut. Freeze the finetuned
base, train MixLoRA experts per category (tools/code/chat/RAG).

### 3.3 ESFT — Expert-Specialized Fine-Tuning ("Let the Expert Stick to His Last", EMNLP 2024)
For an existing MoE: identify which expert most handles a task, finetune ONLY
that expert, freeze everything else. Preserves general ability while adding
specialization at a fraction of the cost.
**Steal:** after upcycling, train each specialist category into its own expert
instead of full-model finetunes.

### 3.4 OLMoE-1B-7B (AllenAI, apache-2.0, ICLR 2025)
**The existence proof for our size class**: 7B total / **1B active**, 64 experts
top-8, fully open (weights, data, code). Matches RESEARCH3 §11's vision
(~100M active/token). Their ablations: fine-grained experts (more, smaller)
+ higher top-k beat coarse experts; router load-balancing is critical.
**Also directly usable**: OLMoE-1B-7B Q4 is ~4.7 GB — over our RAM budget for
weights, but a candidate "big sibling" tier for worker-hot on 8 GB machines.
**Steal:** fine-grained-experts design; their open data pipeline as SFT
reference; load-balance-loss weighting.

### 3.5 Lory (arXiv:2405.03133)
Fully differentiable MoE for autoregressive pretraining — causal-MergeToker
style expert fusion. Relevant later for the from-scratch path in RESEARCH3 §11
(V4); not needed for the adapter path.

### 3.6 Branch-Train-MiX (Meta, 2024)
Finetune N copies of a dense model on N disjoint domains (one per expert),
then merge into an MoE and train the router. Compute-heavy (N full finetunes)
but the cleanest "specialist experts" story. Our LoRA variant: N LoRA finetunes
on domain splits (tools/code/chat/RAG), then MixLoRA-style merge + router —
achievable on Colab.

### 3.7 Alignment with our own docs
- RESEARCH.md §DeepSeek note: "MoE proves specialization works — when tiny GGUF
  MoE models appear they'd be perfect for MIVI." That moment is now: OLMoE and
  MixLoRA make it buildable.
- RESEARCH3 §11 (Sparse MoE, 500M/100M-active, 2-of-8 + shared expert, ~200 MB)
  is the destination. The pipeline in this doc is the road there:
  **finetune dense → MixLoRA specialists → (optional) PESC-style weight upcycle
  → GGUF → Candle MoE inference.**

## 4. Refined pipeline

```
Stage 0 (done)   Dataset: datasets/mivi_agentic_sft.jsonl (175 rows) + eval harness
Stage 1          SFT both bases (MiniCPM5-1B, LFM2.5-350M) on Colab T4, 60 steps
                 -> pick winner by `just agent-eval` (target >= 8/11)
Stage 2          Expand SFT set for the winner's weak workflows; retrain
                 -> target >= 9/11
Stage 3          MixLoRA specialist experts on the frozen winner
                 (tools / code / chat / RAG / planner routers)
                 -> specialist routing visible in traces; score >= 10/11
Stage 4          PESC-style weight upcycle (optional): FFN -> experts from
                 dense weights, brief continued training w/ load-balance loss
Stage 5          GGUF export; Candle/llama.cpp MoE serving; make it `mivi`
```

## 5. Category → expert mapping (Stage 3)

| Expert | Trained on | Routed when |
|---|---|---|
| tool-caller | tool_selection rows | request has tools + tool intent |
| summarizer | tool_result_summary + error_summary | tool results in context |
| coder | verified pairs | code intent / coder role |
| planner | JSON step plans | MULTI_STEP intent |
| chat (shared/frozen) | english_chat + identity | everything else (default expert) |

Router input = hidden state at each FFN layer (MixLoRA) or request-level intent
(our orchestrator already classifies — a hybrid: intent-conditional expert
biasing is a valid simplification to start).

## 6. Risks

- **Catastrophic forgetting** of chat ability during specialist training —
  mitigate: freeze base (MixLoRA does), keep chat anchor rows in every mix.
- **Router collapse** (all traffic to one expert) — use load-balancing loss
  (both MixLoRA and OLMoE do).
- **350M capacity ceiling** — if LFM2.5-350M plateaus at 6-7/11 after SFT,
  that's the parameter limit, not the data; switch MoE base to MiniCPM5-1B.
- **GGUF MoE export**: llama.cpp supports MoE GGUFs (Mixtral-style); MixLoRA
  experts need merging into FFN weights before export — budget a day for the
  conversion tooling.
