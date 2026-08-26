# MIVI-V2 — Master Plan

> **Last updated:** 2026-08-26 · **Release:** v0.0.17 · **Branch:** `feat/verifier-sandbox`
> Companion docs: `docs/MODEL_CANDIDATES.md` (model benchmarks) · `RESEARCH5.md`
> (finetune→MoE prior art) · `TODO.md` (engineering backlog) · `CHANGELOG.md`

---

## 1. The Goal

A 100% Pure Rust local AI engine that runs a **useful AI agent backend on any
device under ~1 GB RAM**, exposed as a single OpenAI-compatible model name
(`mivi`), powered by a **self-finetuned small language model** that is:

- best-in-class at **tool calling** and **agentic** workflows
- good at **thinking / reasoning** and **natural English chat**
- **fast** and **stable**
- ultimately a **fine-tuned dense base → specialist MoE** built by us

## 2. Where we are (v0.0.17 — all shipped)

### Engineering ✅
- OpenAI + Anthropic + Responses + Embeddings API surface, streaming, auth,
  rate limits, Landlock-sandboxed verifier
- Adaptive SemanticCache (hit-rate-driven capacity tuner), KV `--cache-reuse`
  (13.6× turn-2 speedup), `mivi doctor`, `MIVI_RUNTIME_MODE=auto`
- Codebase decomposed: `helpers.rs` 6,185 → 211 lines; reasoning + native-model
  modules extracted; 219 lib tests green
- Agent testing one command away: `just serve-traced / smoke / agent-eval /
  agent-opencode / agent-claude`
- Cross-mode output determinism enforced in the `--live on` CI gate
- LoRA adapter plumbing (`MIVI_LORA_ADAPTERS`) in spawn + worker modes

### Model work ✅
- Benchmarked 7 models through the full pipeline (`docs/MODEL_CANDIDATES.md`)
- Default tier: **Qwen3-1.7B Q2_K** (7/11 agent workflows)
- Ultra-low tier: **LFM2.5-350M** (6/11 at 438 MB, 88 tok/s) registered +
  doctor-recommended; LFM1.0 license verified OK under $10M revenue
- Deterministic identity fast path (server-side, model-independent)

### Finetune pipeline 🟡 (in progress)
- **Round 1 done (neutral result, root-caused):** QLoRA r=16, 60 steps, 182
  chat-template rows on both bases → behaviors learned (right tool, right args)
  but served in the wrong wrapper. Root cause: train/serve prompt-format
  mismatch. Documented in `docs/MODEL_CANDIDATES.md`.
- **Round 2 built, not yet trained:** serving-format dataset (byte-exact
  `mivi debug-prompt` renders, grammar-exact minified tool calls with string
  args, `**Verified Terminal Output:**` headings) + completion-style trainer
  support. 6 surgical rows, 25/20 steps.

## 3. The pipeline (stages, owners, status)

```
Stage 1  SFT both bases on Colab T4            [YOU — GPU]        🟡 round 2 pending
Stage 2  Score GGUFs, pick winner              [ME — local]       ⏳ after Stage 1
Stage 3  Dataset expansion for weak workflows  [ME — local]       (if winner < 9/11)
Stage 4  MixLoRA specialist experts on frozen winner              ⏳ blocked by Stage 2
Stage 5  PESC-style weight upcycle (optional)  [Colab, bigger]    ⏳ blocked by Stage 4
Stage 6  GGUF export + make it `mivi` default  [ME + YOU]         ⏳
```

**Success metric at every stage:** `just agent-eval` score.
Baseline: Qwen3-1.7B default = **7/11**. Round-2 target: **≥9/11**.

## 4. What YOU need to do next (Colab, ~30 min)

1. `git push` is done — repo is current on `feat/verifier-sandbox`
2. Open `notebooks/train_agentic_colab.ipynb` in Colab (T4 GPU)
3. Cell order: install → clone+datasets (add the `build_serving_sft.py` line,
   set `DATASET = "datasets/mivi_serving_sft.jsonl"`) → trainer →
   `train_model("openbmb/MiniCPM5-1B", "outputs/mivi-minicpm5-r2", max_steps=25)`
   → `train_model("LiquidAI/LFM2.5-350M", "outputs/mivi-lfm350-r2", max_steps=20)`
   → export GGUF (`save_pretrained_gguf`, q4_k_m) → download
4. Bring both GGUFs back to `models/` and say the word

## 5. What I do next (local, no GPU needed)

| # | Task | Status |
|---|---|---|
| 1 | Score round-2 GGUFs through the full eval, pick MoE base | waiting on you |
| 2 | If <9/11: expand serving dataset (more tool combos, planner depth, RAG variety) and iterate | ready to go |
| 3 | Stage-4 prep: MixLoRA experiment plan + expert/category mapping (drafted in RESEARCH5 §5) | drafted |
| 4 | DSpark speculative decoding quick win (LFM2.5 pair, `MIVI_DRAFT_MODEL` exists) | pending |
| 5 | Embeddings upgrade candidate: all-MiniLM-L6-v2 for `/v1/embeddings` | pending |
| 6 | Document candle/Qwen3-Q2_K native-engine limitation in catalog + doctor | pending |
| 7 | Ship v0.0.18 with the finetuned winner as (optional) default tier | after Stage 2 |

## 6. Decision log

| Decision | Rationale |
|---|---|
| Finetune base #1 = MiniCPM5-1B | apache-2.0, standard Llama arch, best dense tok/s, hybrid thinking |
| Finetune base #2 = LFM2.5-350M | best score-per-MB (6/11 @ 438 MB), fastest (88 tok/s), fits MoE budget after upcycle |
| Default tier stays Qwen3-1.7B | 7/11 — unbeaten until a finetune wins |
| QLoRA, not full finetune | free T4 constraint; 175→182 rows is adapter-scale |
| Round-2 = serving-format completion training | round 1 proved behaviors transfer but wrappers don't |
| MoE = MixLoRA first, PESC later | adapter-MoE runs on free Colab; weight-upcycle only after routing value is proven |

## 7. Known limitations (accepted, documented)

- Candle native engine cannot load Qwen3 Q2_K GGUFs → 1.7B runs via worker modes
- `mivi serve --port` flag parsing landed in v0.0.17; older binaries ignore it
- 350M-class models sit near the tool-calling floor — hard multi-step workflows
  (coding-verified, trace-*) likely need the 1B base or MoE stage
- Free Colab sessions cap ~4 h; each training run is 10–15 min, well within it
