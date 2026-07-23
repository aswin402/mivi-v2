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
