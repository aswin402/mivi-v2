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
| `MIVI_TRACE` | `1`, `true`, `yes`, `on` | off | Enables compact per-request JSONL traces |
| `MIVI_TRACE_PATH` | filesystem path | `logs/mivi-trace.jsonl` | Overrides trace output file |

`spawn` is the safest low-RAM mode. `worker-eco` lazy-starts one local text worker and falls back to `llama-cli` if the worker fails. `worker-hot` keeps the text worker warm for lower repeated-request latency. Vision stays lazy-loaded.

MIVI also filters large agent tool lists, compresses noisy agent context, minifies command/tool outputs, validates returned tool calls against the selected tools, records optional compact JSONL traces, reads OKF memory from `memory/`, and gates workspace RAG so normal chat is not polluted by code chunks.

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
