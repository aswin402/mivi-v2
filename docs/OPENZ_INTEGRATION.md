# OpenZ Integration

This guide configures OpenZ to use MIVI as a local OpenAI-compatible model backend.

## 1. Start MIVI

Use spawn mode first for the lowest idle RAM while testing OpenZ:

```bash
MIVI_RUNTIME_MODE=spawn cargo run --release -- serve
```

For longer OpenZ agent sessions, use the lazy worker mode:

```bash
MIVI_RUNTIME_MODE=worker-eco MIVI_WORKER_IDLE_SECS=120 cargo run --release -- serve
```

MIVI listens on:

```text
http://127.0.0.1:8000/v1
```

OpenZ must call only this external model name:

```text
mivi
```

Do not configure OpenZ with internal model names such as Qwen, Llama, MiniCPM, or coder/reasoner aliases.

## 2. OpenZ Provider Values

Use these values in OpenZ if it supports OpenAI-compatible providers:

| Field | Value |
| --- | --- |
| Provider type | `openai-compatible` or `openai` chat-completions compatible |
| Base URL | `http://127.0.0.1:8000/v1` |
| Chat endpoint | `http://127.0.0.1:8000/v1/chat/completions` |
| API key | `local` |
| Model | `mivi` |
| Streaming | enabled |
| Tool calling | enabled |
| Vision | enabled only if OpenZ can send OpenAI-style `image_url` content |
| Context limit to advertise | `128000` logical tokens |
| Practical raw context | controlled by `MIVI_CONTEXT_BUDGET`, default `3072` |

MIVI provides the 128K agent experience through compression, OKF memory, gated RAG, and tool-output reduction, not by keeping a full 128K KV cache loaded on low-RAM devices.

## 3. Generic OpenZ JSON Example

If OpenZ accepts JSON provider definitions, start from [`configs/openz-mivi.example.json`](../configs/openz-mivi.example.json) or use this shape and rename keys to match OpenZ's exact schema:

```json
{
  "providers": {
    "mivi": {
      "type": "openai-compatible",
      "baseURL": "http://127.0.0.1:8000/v1",
      "apiKey": "local",
      "models": {
        "mivi": {
          "id": "mivi",
          "name": "MIVI Local",
          "context": 128000,
          "maxOutput": 4096,
          "supportsTools": true,
          "supportsVision": true,
          "supportsStreaming": true
        }
      }
    }
  },
  "defaultModel": "mivi"
}
```

If OpenZ uses environment variables instead, set:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local"
export OPENAI_MODEL="mivi"
export DEFAULT_MODEL="mivi"
```

## 4. Required OpenZ Behavior

OpenZ should send requests to MIVI using OpenAI Chat Completions format:

```json
{
  "model": "mivi",
  "stream": true,
  "messages": [
    { "role": "user", "content": "hii" }
  ],
  "tools": []
}
```

For tool use, OpenZ should pass its available tools in the request `tools` array. MIVI filters large tool lists and validates generated tool calls against the provided schemas.

For images, OpenZ must send real image data as an OpenAI-style content part. Clipboard error text is not enough:

```json
{
  "model": "mivi",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What is in this image?" },
        { "type": "image_url", "image_url": { "url": "file:///tmp/screenshot.png" } }
      ]
    }
  ]
}
```

## 5. Smoke Tests

Models endpoint:

```bash
curl http://127.0.0.1:8000/v1/models
```

Chat endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mivi","stream":false,"messages":[{"role":"user","content":"Say your model name"}]}'
```

Tool inventory check:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mivi","stream":false,"messages":[{"role":"user","content":"what tools do you have"}],"tools":[{"type":"function","function":{"name":"bash","description":"Run shell commands","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}}]}'
```

Expected: MIVI answers that OpenZ/OpenAI-compatible client provided callable tools in this request.

Run the HTTP compatibility smoke suite against a running MIVI server:

```bash
scripts/smoke_openai_compat.py
```

It checks `/v1/models`, Chat Completions, streaming usage chunks, tool calls, `/v1/responses`, URL research tool selection, the full tool-result follow-up loop, and multi-tool result aggregation. Use `MIVI_SMOKE_BASE_URL` when testing a non-default port or host.

## 6. Troubleshooting

If OpenZ says the model does not support images, the issue is OpenZ model metadata. Mark `mivi` as vision-capable/multimodal in OpenZ so it sends `image_url` content to MIVI.

If OpenZ asks what MCPs exist and MIVI cannot name MCP servers, check whether OpenZ exposes MCP server names in tool names or descriptions. MIVI can only see the tool schemas sent in the request.

If OpenZ uses the OpenAI Responses API, point it at MIVI's `/v1/responses` compatibility endpoint. For older OpenZ versions, Chat Completions remains the most tested path.
