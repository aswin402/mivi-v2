# 🤖 External AI Agents Integration Guide

This guide explains how to connect external AI agents and IDE tools to **MIVI-V2**.

---

## 🌐 Endpoint Details

* **Base URL:** `http://localhost:8000/v1`
* **API Key:** `local` (or any non-empty string)
* **Chat Endpoint:** `http://localhost:8000/v1/chat/completions`
* **Models Endpoint:** `http://localhost:8000/v1/models`

### Model

External agents see only one model: **`mivi`**.

Internally, MIVI auto-routes your request to the right SML (Small Model Logic):
- **Chat/QA** → Llama-3.2-1B (reasoner)
- **Code generation** → Qwen-2.5-0.5B (coder) via orchestrator
- **Vision/image** → MiniCPM-V-4.6 (multimodal)

You never need to specify these. Just use `mivi`.

## Runtime Recommendations

Use `spawn` first when testing a new machine because it keeps idle RAM lowest and uses the existing `llama-cli` path:

```bash
MIVI_RUNTIME_MODE=spawn cargo run --release -- serve
```

Use `worker-eco` for agent sessions where repeated chat/reasoning calls matter but RAM still matters:

```bash
MIVI_RUNTIME_MODE=worker-eco MIVI_WORKER_IDLE_SECS=120 cargo run --release -- serve
```

Use `worker-hot` only when you want the fastest repeated text responses and accept the persistent model memory cost:

```bash
MIVI_RUNTIME_MODE=worker-hot cargo run --release -- serve
```

Agent behavior notes:

- Keep model name set to `mivi`; do not configure `llama`, `qwen`, or `minicpm` in the external agent.
- Large tool lists are filtered before prompting so OpenCode-style 100+ tool payloads do not flood the tiny model.
- Long histories are compressed into recent turns, typed tool observations, errors, code blocks, OKF memory, and gated RAG context.
- Noisy command output is minified by tool type before prompting, keeping build/test failures visible without flooding context.
- Store durable project facts in `memory/*.md` using OKF frontmatter with `id`, `title`, `type`, and optional `tags`.

## Latest Runtime Benchmark

Measured on 2026-07-24 with `scripts/bench_runtime.sh`. The benchmark records Rust server RSS, server process-tree RSS, and persistent worker RSS. Worker modes stayed under the 1000 MB active-RAM target, and verified RAG answers removed the previous `worker-hot` RAG timeout.

| Mode | Chat | Coding | Tool | RAG | Vision Skip | Peak Worker RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spawn` | 5578 ms | 2261 ms | 4861 ms | 49 ms | 4607 ms | 0 MB |
| `worker-eco` | 3083 ms | 1726 ms | 3919 ms | 19 ms | 2134 ms | 849 MB |
| `worker-hot` | 3101 ms | 1622 ms | 4180 ms | 18 ms | 5886 ms | 849 MB |

Benchmark output: `benchmarks/runtime-20260724-203641.jsonl`.

Small-model evals are scored semantically: `scripts/eval_small_models.sh` writes `semantic_ok`, `score`, and `reasons`, and exits non-zero when an answer fails expected facts or tool-call checks.

---

## 🛠️ Integration Configurations

### 1. OpenCode / Claude Code / Hermes Agent / AutoGen / CrewAI

Set the following environment variables:

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="local"
export DEFAULT_MODEL="mivi"
```

Or in your `opencode.jsonc`:

```json
{
  "providers": [
    {
      "id": "mivi",
      "name": "MIVI Local",
      "endpoint": "http://localhost:8000/v1/chat/completions",
      "apiKey": "local"
    }
  ],
  "model": {
    "provider": "mivi",
    "model": "mivi"
  }
}
```

### 2. VS Code (Continue.dev Extension)

Add to `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "MIVI Pure Rust AI",
      "provider": "openai",
      "model": "mivi",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "local"
    }
  ]
}
```

### 3. OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="mivi",
    messages=[
        {"role": "user", "content": "Write a python script printing Hello MIVI!"}
    ]
)

print(response.choices[0].message.content)
```

### 4. cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mivi",
    "messages": [
      {"role": "user", "content": "What is Rust programming?"}
    ]
  }'
```

---

## 🔍 Internal Architecture (for reference)

```
External Agent → mivi (single model endpoint)
                        │
              ┌─────────┴──────────┐
              │  NeedleRouter       │  < 2ms intent classification
              │  .classify_intent() │
              └─────────┬──────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   CHAT/QA        CODE/VISION      MULTI_STEP
         │              │              │
         ▼              ▼              ▼
   Llama-1B      Qwen-0.5B      Orchestrator
   (reasoner)    (coder)        (plan → execute → verify)
```

All small models are internal. External agents never need to know about them.
