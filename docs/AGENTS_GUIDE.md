# 🤖 External AI Agents Integration Guide

This guide explains how to connect external AI agents and IDE tools to **MIVI-V2**.

---

## 🌐 Endpoint Details

* **Base URL:** `http://localhost:8000/v1`
* **API Key:** `local` (or any non-empty string)
* **OpenAI Chat Endpoint:** `http://localhost:8000/v1/chat/completions`
* **Models Endpoint:** `http://localhost:8000/v1/models`

---

## 🛠️ Integration Configuration Examples

### 1. VS Code (Continue.dev Extension)

Add the following to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "MIVI-V2 Pure Rust AI",
      "provider": "openai",
      "model": "mivi-v2",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "local"
    },
    {
      "title": "Qwen 2.5 Coder (MIVI-V2)",
      "provider": "openai",
      "model": "qwen-2.5-0.5b",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "local"
    }
  ]
}
```

---

### 2. OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="mivi-v2",
    messages=[
        {"role": "user", "content": "Write a python script printing Hello MIVI-V2!"}
    ]
)

print(response.choices[0].message.content)
```

---

### 3. OpenCode Agent / Hermes Agent / AutoGen / CrewAI

Set the following environment variables in your agent execution shell:

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="local"
export DEFAULT_MODEL="mivi-v2"
```

---

### 4. cURL Direct REST Testing

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mivi-v2",
    "messages": [
      {"role": "user", "content": "Write a python script calculating Fibonacci numbers"}
    ]
  }'
```
