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
