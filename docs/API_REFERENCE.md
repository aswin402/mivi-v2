# 📡 MIVI-V2 REST API Specification

MIVI-V2 exposes an OpenAI-compatible REST API built with Axum listening on `http://localhost:8000`.

---

## 📍 Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service status and version health check |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions endpoint |

---

## 1. `GET /`

Returns overall server health and operational metrics.

**Response:**
```json
{
  "openai_endpoint": "/v1/chat/completions",
  "ram_footprint": "< 12 MB RAM",
  "service": "MIVI-V2 Pure Rust High-Speed AI Engine",
  "status": "online",
  "version": "0.0.4"
}
```

---

## 2. `GET /v1/models`

Returns available models in OpenAI list format.

**Response:**
```json
{
  "object": "list",
  "data": [
    { "id": "mivi", "object": "model", "created": 1742600000, "owned_by": "mivi" }
  ]
}
```

> **Note:** Internal SMLs (qwen-2.5-0.5b, llama-3.2-1b, minicpm-v-4.6) exist but are not exposed. MIVI auto-routes `mivi` requests to the right model internally.

## Runtime Modes

MIVI supports three runtime modes through environment variables:

| Variable | Values | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `MIVI_RUNTIME_MODE` | `spawn`, `worker-eco`, `worker-hot` | `spawn` | Select process-per-request or persistent text worker mode |
| `MIVI_CONTEXT_BUDGET` | integer tokens, minimum `1024` | `4096` | Sets the bounded prompt pack budget |
| `MIVI_WORKER_IDLE_SECS` | positive integer seconds | `120` | Idle sleep/stop budget for worker modes |
| `MIVI_WORKER_PORT` | local TCP port | `18080` | Internal `llama-server` worker port |

`spawn` is the safest low-RAM mode. `worker-eco` lazy-starts one local text worker and falls back to `llama-cli` if the worker fails. `worker-hot` keeps the text worker warm for lower repeated-request latency. Vision stays lazy-loaded.

MIVI also filters large agent tool lists, compresses noisy agent context, reads OKF memory from `memory/`, and gates workspace RAG so normal chat is not polluted by code chunks.

## Latest Runtime Benchmark

Measured on 2026-07-24 with `scripts/bench_runtime.sh`. RSS currently records the Rust MIVI server process only; worker child model RSS needs a follow-up benchmark improvement.

| Mode | Chat | Coding | Tool | RAG | Vision Skip |
| --- | ---: | ---: | ---: | ---: | ---: |
| `spawn` | 6297 ms | 6355 ms | 4605 ms | 41468 ms | 8728 ms |
| `worker-eco` | 3649 ms | 1289 ms | 4155 ms | 49786 ms | 10935 ms |
| `worker-hot` | 3454 ms | 1622 ms | 4642 ms | 38920 ms | 7383 ms |

Benchmark output: `benchmarks/runtime-20260724-160116.jsonl`.

---

## 3. `POST /v1/chat/completions`

Generates chat completions or verified code execution.

**Request Body:**
```json
{
  "model": "mivi",
  "messages": [
    { "role": "user", "content": "Write a python line printing Hello World" }
  ]
}
```

**Multimodal Vision Request Body:**
```json
{
  "model": "mivi",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Describe the contents of this image" },
        { "type": "image_url", "image_url": { "url": "/path/to/screenshot.png" } }
      ]
    }
  ]
}
```

> MIVI auto-detects multimodal input and routes to vision model internally.

**Response Body:**
```json
{
  "id": "chatcmpl-v2-1784699136",
  "object": "chat.completion",
  "created": 1784699136,
  "model": "mivi",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "```python\nprint(\"Hello World\")\n```"
      },
      "finish_reason": "stop"
    }
  ]
}
```
