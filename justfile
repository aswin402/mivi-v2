# MIVI-V2 local commands
# Default server profile is laptop-friendly and keeps model/tool prompts bounded.

default:
    @just --list

serve:
    MIVI_ULTRA_LOW_RAM=1 MIVI_THREADS=1 MIVI_CONTEXT_BUDGET=1024 MIVI_MAX_CONCURRENT_REQUESTS=1 MIVI_MODEL_CACHE_MAX=1 cargo run --release -- serve

serve-traced:
    MIVI_TRACE=1 MIVI_ULTRA_LOW_RAM=1 MIVI_THREADS=1 MIVI_CONTEXT_BUDGET=1024 MIVI_MAX_CONCURRENT_REQUESTS=1 MIVI_MODEL_CACHE_MAX=1 cargo run --release -- serve

serve-normal:
    cargo run --release -- serve

build:
    cargo build --release

test:
    cargo test

check:
    make check-agent

# HTTP smoke tests against a running server (just serve first)
smoke:
    python3 scripts/smoke_openai_compat.py

# Graded simulated agent traffic against a running traced server
agent-eval:
    MIVI_TRACE=1 python3 scripts/eval_agent_workflows.py

# Drive real OpenCode against local mivi (just serve first)
agent-opencode prompt="write a python script that prints hello":
    OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_API_KEY=local opencode --model mivi "{{prompt}}"

# Drive real Claude Code against local mivi via the /v1/messages adapter
agent-claude prompt="write a python script that prints hello":
    ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_API_KEY=local claude --model mivi -p "{{prompt}}"
