#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p benchmarks
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="benchmarks/runtime-${STAMP}.jsonl"
SERVER_PORT="${MIVI_BENCH_SERVER_PORT:-8000}"
WORKER_PORT_BASE="${MIVI_BENCH_WORKER_PORT_BASE:-18180}"

PROMPTS=(
  "chat|Say hello in one short sentence."
  "coding|Write Python code that prints the sum of 2 and 3."
  "tool|Use the get_weather tool for Paris."
  "rag|In this project codebase, what module handles routing intent?"
  "vision-skip|Describe how vision requests should be routed when no image is attached."
)

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

wait_for_server() {
  local url="http://127.0.0.1:${SERVER_PORT}/v1/models"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

rss_kb() {
  local pid="$1"
  ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1}'
}

descendant_pids() {
  local parent="$1"
  local child
  pgrep -P "$parent" 2>/dev/null || true
  for child in $(pgrep -P "$parent" 2>/dev/null || true); do
    descendant_pids "$child"
  done
}

tree_rss_kb() {
  local root_pid="$1"
  local total=0
  local pid rss
  for pid in "$root_pid" $(descendant_pids "$root_pid"); do
    rss="$(rss_kb "$pid")"
    if [[ -n "${rss:-}" ]]; then
      total="$((total + rss))"
    fi
  done
  echo "$total"
}

worker_rss_kb() {
  local worker_port="$1"
  local total=0
  local pid rss cmdline
  for pid in $(pgrep -f 'llama-server' 2>/dev/null || true); do
    cmdline="$(tr '\000' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"--port ${worker_port}"* ]]; then
      rss="$(rss_kb "$pid")"
      if [[ -n "${rss:-}" ]]; then
        total="$((total + rss))"
      fi
    fi
  done
  echo "$total"
}

request_payload() {
  local kind="$1"
  local prompt="$2"
  if [[ "$kind" == "tool" ]]; then
    cat <<JSON
{"model":"mivi","messages":[{"role":"user","content":$(printf '%s' "$prompt" | json_escape)}],"stream":false,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
JSON
  else
    cat <<JSON
{"model":"mivi","messages":[{"role":"user","content":$(printf '%s' "$prompt" | json_escape)}],"stream":false}
JSON
  fi
}

run_mode() {
  local mode="$1"
  local worker_port="$2"
  cleanup
  SERVER_PID=""

  echo "[bench] starting mode=${mode} server_port=${SERVER_PORT} worker_port=${worker_port}" >&2
  MIVI_RUNTIME_MODE="$mode" \
  MIVI_WORKER_PORT="$worker_port" \
  MIVI_WORKER_IDLE_SECS="${MIVI_WORKER_IDLE_SECS:-30}" \
  cargo run --release -- serve >/tmp/mivi-bench-${mode}.log 2>&1 &
  SERVER_PID="$!"
  wait_for_server

  for item in "${PROMPTS[@]}"; do
    local kind="${item%%|*}"
    local prompt="${item#*|}"
    local payload
    payload="$(request_payload "$kind" "$prompt")"
    local start end elapsed_ms response_file status server_rss server_tree_rss worker_rss
    response_file="/tmp/mivi-bench-response-${mode}-${kind}.json"
    start="$(date +%s%3N)"
    status="ok"
    if ! curl -fsS --max-time "${MIVI_BENCH_TIMEOUT:-180}" \
      "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$payload" >"$response_file"; then
      status="error"
    fi
    end="$(date +%s%3N)"
    elapsed_ms="$((end - start))"
    server_rss="$(rss_kb "$SERVER_PID")"
    server_tree_rss="$(tree_rss_kb "$SERVER_PID")"
    worker_rss="$(worker_rss_kb "$worker_port")"
    python3 -c 'import json,sys; print(json.dumps({"mode":sys.argv[1],"kind":sys.argv[2],"elapsed_ms":int(sys.argv[3]),"server_rss_kb":int(sys.argv[4] or 0),"server_tree_rss_kb":int(sys.argv[5] or 0),"worker_rss_kb":int(sys.argv[6] or 0),"status":sys.argv[7]}))' \
      "$mode" "$kind" "$elapsed_ms" "${server_rss:-0}" "${server_tree_rss:-0}" "${worker_rss:-0}" "$status" >>"$OUT"
  done

  cleanup
}

cargo build --release >/dev/null
run_mode spawn "$((WORKER_PORT_BASE))"
run_mode worker-eco "$((WORKER_PORT_BASE + 1))"
run_mode worker-hot "$((WORKER_PORT_BASE + 2))"

echo "$OUT"
