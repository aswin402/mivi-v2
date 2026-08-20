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
