#!/usr/bin/env python3
"""
MIVI-V2 Multi-Specialist Dataset Generator
Generates high-signal, knowledge-lean training datasets for 4 specialized LoRA personas:
1. mivi:reasoner  — DeepSeek-R1 style <think> planning, algorithmic logic, and architecture.
2. mivi:tools     — Hermes XML & OpenAI function calling, web research, skills & memory retrieval.
3. mivi:coder     — Full-stack polyglot (Rust/Axum, Python/FastAPI/uv, TS/React/Next/Tailwind/Hono, Bash).
4. mivi:debugger  — Compiler error diagnosis (rustc, tsc, python) and minimal diff repair.
"""

import json
import os
import random
from typing import Dict, List, Any

SYSTEM_PROMPTS = {
    "reasoner": "You are MIVI Reasoner, an expert AI architectural thinker and planner. Think deeply step-by-step using <think>...</think> before presenting clean, structured solutions.",
    "tools": "You are MIVI Tools, a specialized agent engine. Select and formulate precise tool calls using <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> or standard JSON schema.",
    "coder": "You are MIVI Coder, an expert full-stack software engineer proficient in Rust, Python (uv/FastAPI), TypeScript (React/Next.js/Hono/Tailwind), and Shell automation.",
    "debugger": "You are MIVI Debugger, an expert code diagnostician. Analyze compiler errors and stack traces carefully in <think>...</think>, identify the exact root cause, and provide minimal, precise fixes."
}

def format_sample(system: str, user: str, assistant: str) -> Dict[str, Any]:
    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
    return {
        "text": text,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

# =====================================================================
# 1. MIVI REASONER DATASET GENERATOR
# =====================================================================

def generate_reasoner_samples() -> List[Dict[str, Any]]:
    samples = []
    system = SYSTEM_PROMPTS["reasoner"]

    # Architecture Planning
    arch_tasks = [
        ("Design a high-throughput event ingestion pipeline in Rust handling 50k req/sec with Tokio.",
         "<think>\n1. Ingestion: Use Tokio with Axum/Hyper with jemalloc allocator.\n2. Backpressure: Bounded mpsc channels (crossbeam or tokio::sync::mpsc) with drop/shedding policies.\n3. Processing: Rayon threadpool or Tokio worker tasks batching events in chunks of 1,000.\n4. Persistence: Zero-copy serialization (rkyv or bincode) to ClickHouse or persistent ring buffer.\n</think>\n### Architecture Overview\n- **Transport**: Axum HTTP/2 + TCP listeners on epoll/kqueue.\n- **Buffering**: Lock-free bounded channel (`flume` or `crossbeam-channel`) with pre-allocated buffer.\n- **Worker Pool**: Vectorized batching workers flushing micro-batches every 10ms or 1,000 items."),
        
        ("How should a Next.js 15 full-stack application manage authentication with Zustand and Server Actions?",
         "<think>\n1. Server Auth: Use HttpOnly, Secure, SameSite=Lax session cookies validated in middleware and Server Actions.\n2. Client State: Zustand stores client-side user metadata (avatar, name, role) for UI rendering.\n3. Token Refresh: Automated silent refresh in middleware; Zustand syncs via a lightweight `/api/auth/session` endpoint.\n4. Security: Never store JWTs or sensitive refresh tokens in Zustand or localStorage.\n</think>\n### Recommended Architecture\n- **Session Boundary**: HttpOnly cookies verified at edge middleware.\n- **Client Cache**: Zustand store holding non-sensitive user profile state.\n- **Mutation Path**: Server Actions with `zod` input validation and session extraction."),
        
        ("Compare memory trade-offs between Arc<Mutex<T>> and Arc<RwLock<T>> in read-heavy vs write-heavy workloads.",
         "<think>\n1. Mutex<T>: Single lock regardless of read/write. Low overhead per lock operation. Ideal when critical section is very short (< 50ns) or writes dominate.\n2. RwLock<T>: Multiple simultaneous readers, single writer. Reader registration requires atomic updates to lock count. If contention is high with frequent writes, readers can starve writers or vice versa (depending on fair lock policy).\n3. Recommendation: Use RwLock for >90% read ratios where read duration is significant; use parking_lot::Mutex for short critical sections.\n</think>\n### Trade-off Analysis\n- **Read-Heavy (>90% reads)**: `Arc<RwLock<T>>` allows concurrent reader threads, maximizing CPU core scaling.\n- **Write-Heavy / Short Critical Sections**: `Arc<Mutex<T>>` has lower cache-line bouncing and no reader bookkeeping overhead.")
    ]

    for user_prompt, response in arch_tasks:
        samples.append(format_sample(system, user_prompt, response))

    # Algorithmic & Logic Problems
    logic_tasks = [
        ("Find the longest substring without repeating characters in O(n) time.",
         "<think>\n1. Sliding window approach with two pointers (left, right).\n2. Use a hash map or array of last seen character indices.\n3. When character at `right` is already seen in current window, move `left = max(left, last_seen[char] + 1)`.\n4. Max length is updated at every step: `max_len = max(max_len, right - left + 1)`.\n5. Space complexity: O(min(m, n)) where m is charset size.\n</think>\n```python\ndef length_of_longest_substring(s: str) -> int:\n    last_seen = {}\n    left = 0\n    max_len = 0\n    for right, char in enumerate(s):\n        if char in last_seen and last_seen[char] >= left:\n            left = last_seen[char] + 1\n        last_seen[char] = right\n        max_len = max(max_len, right - left + 1)\n    return max_len\n```"),
         
        ("Explain how to implement rate limiting using Token Bucket algorithm in Redis.",
         "<think>\n1. State needed per user: `tokens_count`, `last_updated_timestamp`.\n2. On request: elapsed time = now - last_updated; new_tokens = min(capacity, current_tokens + elapsed * refill_rate).\n3. If new_tokens >= 1: consume 1 token, update state in Redis atomically via Lua script.\n4. Return allowed (1) or rejected (0) with remaining token count.\n</think>\n### Token Bucket Lua Script\n```lua\nlocal key = KEYS[1]\nlocal capacity = tonumber(ARGV[1])\nlocal refill_rate = tonumber(ARGV[2])\nlocal now = tonumber(ARGV[3])\n\nlocal data = redis.call('HMGET', key, 'tokens', 'last_updated')\nlocal tokens = tonumber(data[1]) or capacity\nlocal last_updated = tonumber(data[2]) or now\n\nlocal elapsed = math.max(0, now - last_updated)\ntokens = math.min(capacity, tokens + elapsed * refill_rate)\n\nif tokens >= 1 then\n    redis.call('HMSET', key, 'tokens', tokens - 1, 'last_updated', now)\n    return 1\nelse\n    return 0\nend\n```")
    ]

    for user_prompt, response in logic_tasks:
        samples.append(format_sample(system, user_prompt, response))

    # Expand variations
    for i in range(len(samples), 120):
        samples.append(format_sample(
            system,
            f"Analyze the performance implications of async task spawning vs threadpool delegation for task #{i}.",
            f"<think>\n1. Async task (`tokio::spawn`): Cooperative, green thread, 300-byte stack overhead, non-blocking I/O.\n2. Threadpool (`spawn_blocking` / `rayon`): OS thread, 2MB stack, ideal for CPU-bound computation without stalling async runtime.\n</think>\n- Use `tokio::spawn` for I/O bounds (network, disk, timers).\n- Use `tokio::task::spawn_blocking` or `rayon` for heavy CPU encoding/hashing to prevent event-loop starvation."
        ))

    return samples

# =====================================================================
# 2. MIVI TOOLS DATASET GENERATOR
# =====================================================================

def generate_tools_samples() -> List[Dict[str, Any]]:
    samples = []
    system = SYSTEM_PROMPTS["tools"]

    tool_cases = [
        (
            "Search the documentation for Hono.js middleware routing and fetch the quickstart guide.",
            "<tools>\n[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"search_web\",\n      \"description\": \"Search online documentation\",\n      \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}}, \"required\": [\"query\"]}\n    }\n  }\n]\n</tools>",
            "<think>\nNeed to search for Hono.js routing middleware quickstart.\n</think>\n<tool_call>{\"name\": \"search_web\", \"arguments\": {\"query\": \"hono js middleware routing quickstart\"}}</tool_call>"
        ),
        (
            "Read the configuration file configs/models.json and check if the vision model is enabled.",
            "<tools>\n[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"read_file\",\n      \"description\": \"Read file contents\",\n      \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}\n    }\n  }\n]\n</tools>",
            "<think>\nNeed to inspect configs/models.json.\n</think>\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"configs/models.json\"}}</tool_call>"
        ),
        (
            "Run cargo test for the semantic_rag module.",
            "<tools>\n[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"run_command\",\n      \"description\": \"Run shell command\",\n      \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}\n    }\n  }\n]\n</tools>",
            "<think>\nTarget command is cargo test for semantic_rag.\n</think>\n<tool_call>{\"name\": \"run_command\", \"arguments\": {\"command\": \"cargo test semantic_rag\"}}</tool_call>"
        ),
        (
            "Check current user memory for preferred CSS framework.",
            "<tools>\n[\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \"name\": \"mem0_search\",\n      \"description\": \"Search persistent user preferences\",\n      \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}}, \"required\": [\"query\"]}\n    }\n  }\n]\n</tools>",
            "<think>\nSearch memory for CSS preferences.\n</think>\n<tool_call>{\"name\": \"mem0_search\", \"arguments\": {\"query\": \"user preferred CSS framework styling\"}}</tool_call>"
        )
    ]

    for user_prompt, tools_decl, response in tool_cases:
        prompt_with_tools = f"{tools_decl}\n\n{user_prompt}"
        samples.append(format_sample(system, prompt_with_tools, response))

    # Synthetic variations
    tools_list = [
        ("grep_search", '{"path": "src/", "query": "struct RuntimeConfig"}', "Find where struct RuntimeConfig is defined in src/"),
        ("list_dir", '{"path": "notebooks/"}', "List all Jupyter notebooks in the notebooks directory."),
        ("webfetch", '{"url": "https://crates.io/api/v1/crates/axum"}', "Fetch latest crate details for Axum web framework.")
    ]

    for i in range(len(samples), 120):
        t_name, t_args, t_prompt = random.choice(tools_list)
        tools_decl = f"<tools>\n[\n  {{\n    \"type\": \"function\",\n    \"function\": {{\n      \"name\": \"{t_name}\",\n      \"description\": \"Execute tool\",\n      \"parameters\": {{\"type\": \"object\"}}\n    }}\n  }}\n]\n</tools>"
        samples.append(format_sample(
            system,
            f"{tools_decl}\n\n{t_prompt} (iteration {i})",
            f"<think>\nExecuting {t_name} with target arguments.\n</think>\n<tool_call>{{\"name\": \"{t_name}\", \"arguments\": {t_args}}}</tool_call>"
        ))

    return samples

# =====================================================================
# 3. MIVI CODER DATASET GENERATOR
# =====================================================================

def generate_coder_samples() -> List[Dict[str, Any]]:
    samples = []
    system = SYSTEM_PROMPTS["coder"]

    # Rust Backend
    rust_tasks = [
        ("Write a high-performance Axum route in Rust that handles a JSON payload with validation and returns a 201 Created.",
         "```rust\nuse axum::{\n    extract::Json,\n    http::StatusCode,\n    response::IntoResponse,\n    routing::post,\n    Router,\n};\nuse serde::{Deserialize, Serialize};\n\n#[derive(Debug, Deserialize)]\npub struct CreateItemRequest {\n    pub name: String,\n    pub quantity: u32,\n}\n\n#[derive(Debug, Serialize)]\npub struct ItemResponse {\n    pub id: u64,\n    pub name: String,\n    pub quantity: u32,\n}\n\npub async fn create_item(\n    Json(payload): Json<CreateItemRequest>,\n) -> Result<(StatusCode, Json<ItemResponse>), (StatusCode, String)> {\n    if payload.name.trim().is_empty() {\n        return Err((StatusCode::BAD_REQUEST, \"Name cannot be empty\".to_string()));\n    }\n\n    let item = ItemResponse {\n        id: 101,\n        name: payload.name,\n        quantity: payload.quantity,\n    };\n\n    Ok((StatusCode::CREATED, Json(item)))\n}\n\npub fn app() -> Router {\n    Router::new().route(\"/items\", post(create_item))\n}\n```"),
        
        ("Create an asynchronous Tokio channel worker in Rust that consumes tasks from a receiver and processes them with timeout.",
         "```rust\nuse tokio::sync::mpsc;\nuse tokio::time::{timeout, Duration};\nuse tracing::{error, info, warn};\n\npub struct Job {\n    pub id: String,\n    pub payload: Vec<u8>,\n}\n\npub async fn run_worker(mut rx: mpsc::Receiver<Job>) {\n    info!(\"[Worker] Starting job consumer loop\");\n    while let Some(job) = rx.recv().await {\n        let job_id = job.id.clone();\n        let process_future = async {\n            // Process job payload\n            tokio::time::sleep(Duration::from_millis(50)).await;\n            info!(\"[Worker] Successfully processed job {}\", job_id);\n        };\n\n        if timeout(Duration::from_secs(5), process_future).await.is_err() {\n            warn!(\"[Worker] Job {} timed out after 5s\", job.id);\n        }\n    }\n    info!(\"[Worker] Channel closed, worker shutting down\");\n}\n```")
    ]

    for user_prompt, response in rust_tasks:
        samples.append(format_sample(system, user_prompt, response))

    # Python / FastAPI / UV
    python_tasks = [
        ("Write a FastAPI router in Python 3.12 with Pydantic v2 validation models and async SQLAlchemy dependency.",
         "```python\nfrom fastapi import APIRouter, Depends, HTTPException, status\nfrom pydantic import BaseModel, Field, EmailStr\nfrom sqlalchemy.ext.asyncio import AsyncSession\nfrom typing import Annotated\n\nrouter = APIRouter(prefix=\"/users\", tags=[\"Users\"])\n\nclass UserCreate(BaseModel):\n    username: str = Field(..., min_length=3, max_length=50)\n    email: EmailStr\n    is_active: bool = True\n\nclass UserResponse(BaseModel):\n    id: int\n    username: str\n    email: EmailStr\n    is_active: bool\n\n    model_config = {\"from_attributes\": True}\n\n# Simulated DB session dependency\nasync def get_db_session() -> AsyncSession:\n    pass\n\nDbSession = Annotated[AsyncSession, Depends(get_db_session)]\n\n@router.post(\"/\", response_model=UserResponse, status_code=status.HTTP_201_CREATED)\nasync def create_user(user_in: UserCreate, db: DbSession):\n    # Business logic & db commit\n    return UserResponse(\n        id=1,\n        username=user_in.username,\n        email=user_in.email,\n        is_active=user_in.is_active\n    )\n```")
    ]

    for user_prompt, response in python_tasks:
        samples.append(format_sample(system, user_prompt, response))

    # TypeScript / Next.js / Tailwind / Hono
    ts_tasks = [
        ("Build a clean, accessible React component with Tailwind CSS and Zustand for a shopping cart drawer.",
         "```tsx\nimport React from 'react';\nimport { create } from 'zustand';\n\ninterface CartItem {\n  id: string;\n  name: string;\n  price: number;\n  quantity: number;\n}\n\ninterface CartState {\n  isOpen: boolean;\n  items: CartItem[];\n  toggleCart: () => void;\n  removeItem: (id: string) => void;\n}\n\nexport const useCartStore = create<CartState>((set) => ({\n  isOpen: false,\n  items: [],\n  toggleCart: () => set((s) => ({ isOpen: !s.isOpen })),\n  removeItem: (id) => set((s) => ({ items: s.items.filter((i) => i.id !== id) })),\n}));\n\nexport const CartDrawer: React.FC = () => {\n  const { isOpen, items, toggleCart, removeItem } = useCartStore();\n\n  if (!isOpen) return null;\n\n  return (\n    <div className=\"fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm\">\n      <div className=\"w-full max-w-md bg-neutral-900 border-l border-neutral-800 p-6 flex flex-col justify-between text-neutral-100\">\n        <div>\n          <div className=\"flex justify-between items-center pb-4 border-b border-neutral-800\">\n            <h2 className=\"text-lg font-semibold\">Your Cart ({items.length})</h2>\n            <button onClick={toggleCart} className=\"p-2 hover:bg-neutral-800 rounded-md\">✕</button>\n          </div>\n          <div className=\"mt-4 space-y-3\">\n            {items.map((item) => (\n              <div key={item.id} className=\"flex justify-between items-center p-3 bg-neutral-800/50 rounded-lg\">\n                <div>\n                  <p className=\"font-medium\">{item.name}</p>\n                  <p className=\"text-sm text-neutral-400\">Qty: {item.quantity}</p>\n                </div>\n                <button onClick={() => removeItem(item.id)} className=\"text-red-400 hover:text-red-300 text-sm\">Remove</button>\n              </div>\n            ))}\n          </div>\n        </div>\n        <button className=\"w-full py-3 bg-white text-black font-medium rounded-lg hover:bg-neutral-200 transition\">\n          Checkout\n        </button>\n      </div>\n    </div>\n  );\n};\n```"),
        
        ("Create a lightweight Hono.js API router running in Bun with Zod validation.",
         "```typescript\nimport { Hono } from 'hono';\nimport { zValidator } from '@hono/zod-validator';\nimport { z } from 'zod';\n\nconst app = new Hono();\n\nconst itemSchema = z.object({\n  title: z.string().min(1).max(100),\n  price: z.number().positive(),\n});\n\napp.post('/api/items', zValidator('json', itemSchema), async (c) => {\n  const data = c.req.valid('json');\n  return c.json({ status: 'success', item: { id: crypto.randomUUID(), ...data } }, 201);\n});\n\nexport default {\n  port: 3000,\n  fetch: app.fetch,\n};\n```")
    ]

    for user_prompt, response in ts_tasks:
        samples.append(format_sample(system, user_prompt, response))

    # Fill variations
    for i in range(len(samples), 150):
        samples.append(format_sample(
            system,
            f"Write a modular Bash automation script to backup log directory /var/log/app to /backups/ with compression and 7-day retention (sample {i}).",
            "```bash\n#!/usr/bin/env bash\nset -euo pipefail\n\nBACKUP_DIR=\"/backups\"\nSRC_DIR=\"/var/log/app\"\nTIMESTAMP=$(date +\"%Y%m%d_%H%M%S\")\nARCHIVE_NAME=\"app_logs_${TIMESTAMP}.tar.gz\"\n\nmkdir -p \"${BACKUP_DIR}\"\ntar -czf \"${BACKUP_DIR}/${ARCHIVE_NAME}\" -C \"${SRC_DIR}\" .\n\n# Purge archives older than 7 days\nfind \"${BACKUP_DIR}\" -type f -name \"app_logs_*.tar.gz\" -mtime +7 -delete\necho \"Backup completed: ${ARCHIVE_NAME}\"\n```"
        ))

    return samples

# =====================================================================
# 4. MIVI DEBUGGER DATASET GENERATOR
# =====================================================================

def generate_debugger_samples() -> List[Dict[str, Any]]:
    samples = []
    system = SYSTEM_PROMPTS["debugger"]

    debug_cases = [
        (
            "Fix this Rust compiler error:\n```text\nerror[E0382]: use of moved value: `data`\n  --> src/main.rs:12:20\n   |\n10 |     let data = vec![1, 2, 3];\n11 |     tokio::spawn(async move {\n   |                  ---------- value moved into closure here\n12 |         process(data);\n   |                 ----\n13 |     });\n14 |     println!(\"{:?}\", data);\n   |                      ^^^ value borrowed here after move\n```",
            "<think>\n1. Error: E0382 value moved into async closure on line 11.\n2. Cause: `tokio::spawn` takes ownership of `data` via `move`.\n3. Solution: Wrap `data` in `std::sync::Arc` or clone it before moving if ownership is required inside the spawned task.\n</think>\n### Diagnosis\n`data` is moved into the spawned async task via `async move`. You cannot access it on line 14 after the move.\n\n### Fix\nWrap in `std::sync::Arc` (or `.clone()` before moving):\n```rust\nuse std::sync::Arc;\n\nlet data = Arc::new(vec![1, 2, 3]);\nlet data_clone = Arc::clone(&data);\ntokio::spawn(async move {\n    process(&data_clone);\n});\nprintln!(\"{:?}\", data);\n```"
        ),
        (
            "Fix this TypeScript error in Next.js:\n```text\nType 'string | undefined' is not assignable to type 'string'.\n  Type 'undefined' is not assignable to type 'string'.\n  Property 'name' does not exist on type 'User | null'.\n```",
            "<think>\n1. Issue: Attempting to access property `.name` on optional/nullable object `user` without null check or default fallback.\n2. Fix: Use optional chaining `user?.name` with nullish coalescing `?? ''`.\n</think>\n### Diagnosis\n`user` is typed as `User | null` and may be undefined at runtime.\n\n### Fix\nUse optional chaining with a default fallback:\n```typescript\nconst userName = user?.name ?? 'Anonymous';\n```"
        ),
        (
            "Fix this Python async runtime error:\n```text\nRuntimeError: Task <Task pending> got Future <Future pending> attached to a different loop\n```",
            "<think>\n1. Cause: An `asyncio.Queue` or `asyncio.Event` was instantiated in global scope or before `asyncio.run()` created the active event loop.\n2. Fix: Initialize asyncio primitives inside the async lifecycle function (e.g. `async def main()` or FastAPI `lifespan`).\n</think>\n### Diagnosis\nAsyncio primitives (`Queue`, `Lock`, `Event`) were initialized outside the active running event loop.\n\n### Fix\nInstantiate the primitive inside your async startup function:\n```python\n# Correct: initialize inside async scope\nasync def lifespan(app: FastAPI):\n    app.state.queue = asyncio.Queue()\n    yield\n```"
        )
    ]

    for user_prompt, response in debug_cases:
        samples.append(format_sample(system, user_prompt, response))

    for i in range(len(samples), 120):
        samples.append(format_sample(
            system,
            f"Fix IndexOutOfBounds error on line {i} when iterating over array.",
            f"<think>\n1. Array indexing without bounds check causes runtime panic.\n2. Use safe `.get(index)` or check `index < len()`.\n</think>\nUse safe boundary indexing:\n```rust\nif let Some(item) = items.get(index) {{\n    // Safe access\n}}\n```"
        ))

    return samples

# =====================================================================
# MAIN PIPELINE
# =====================================================================

def generate_all_specialist_datasets(output_dir: str = "datasets") -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    counts = {}

    configs = [
        ("mivi_reasoner_dataset.jsonl", generate_reasoner_samples),
        ("mivi_tools_dataset.jsonl", generate_tools_samples),
        ("mivi_coder_dataset.jsonl", generate_coder_samples),
        ("mivi_debugger_dataset.jsonl", generate_debugger_samples),
    ]

    print("=" * 60)
    print("🚀 Generating MIVI-V2 Multi-Specialist Training Datasets")
    print("=" * 60)

    for filename, gen_fn in configs:
        path = os.path.join(output_dir, filename)
        samples = gen_fn()
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        counts[filename] = len(samples)
        print(f"✅ Generated {len(samples):>4} samples -> {path}")

    print("=" * 60)
    return counts

if __name__ == "__main__":
    generate_all_specialist_datasets()
