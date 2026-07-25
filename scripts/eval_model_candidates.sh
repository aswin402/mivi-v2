#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="model-eval-results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/model-candidates-$(date +%Y%m%d-%H%M%S).jsonl"
SERVER_URL="${MIVI_EVAL_SERVER_URL:-http://127.0.0.1:8000/v1/chat/completions}"
TRACE_PATH="${MIVI_TRACE_PATH:-logs/mivi-trace.jsonl}"
ALLOW_FAILURES="${MIVI_EVAL_ALLOW_FAILURES:-0}"
FAILED=0
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

write_row() {
  local candidate="$1"
  local role="$2"
  local reasoner="$3"
  local coder="$4"
  local agent_out="$5"
  local small_out="$6"
  local elapsed_ms="$7"
  local ok="$8"
  python3 - "$candidate" "$role" "$reasoner" "$coder" "$agent_out" "$small_out" "$elapsed_ms" "$ok" <<'PYROW' >>"$OUT"
import json, sys
candidate, role, reasoner, coder, agent_out, small_out, elapsed_ms, ok = sys.argv[1:]
print(json.dumps({
    "candidate": candidate,
    "role": role,
    "reasoner_model": reasoner,
    "coder_model": coder,
    "agent_eval": agent_out,
    "small_eval": small_out,
    "elapsed_ms": int(elapsed_ms),
    "ok": ok == "1",
}))
PYROW
}

wait_for_server() {
  local deadline=$((SECONDS + ${MIVI_CANDIDATE_BOOT_TIMEOUT:-45}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

candidate_lines() {
  if [[ -n "${MIVI_CANDIDATES_FILE:-}" ]]; then
    cat "$MIVI_CANDIDATES_FILE"
  else
    python3 - <<'PYCANDIDATE'
import json
print(json.dumps({
    "name": "default-qwen3-qwen25q4",
    "role": "default",
    "reasoner": "models/qwen3-0.6b-q4_k_m.gguf",
    "coder": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
}))
PYCANDIDATE
  fi
}

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  name="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["name"])' "$line")"
  role="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("role", "candidate"))' "$line")"
  reasoner="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("reasoner", ""))' "$line")"
  coder="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("coder", ""))' "$line")"
  cli_timeout="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("cli_timeout_secs", "180"))' "$line")"

  if [[ -n "$reasoner" && ! -f "$reasoner" ]]; then
    write_row "$name" "$role" "$reasoner" "$coder" "missing reasoner model" "" 0 0
    FAILED=1
    continue
  fi
  if [[ -n "$coder" && ! -f "$coder" ]]; then
    write_row "$name" "$role" "$reasoner" "$coder" "" "missing coder model" 0 0
    FAILED=1
    continue
  fi

  cleanup
  SERVER_PID=""
  start_ms="$(date +%s%3N)"
  MIVI_TRACE=1 \
  MIVI_TRACE_PATH="$TRACE_PATH" \
  MIVI_REASONER_MODEL="${reasoner:-models/qwen3-0.6b-q4_k_m.gguf}" \
  MIVI_CODER_MODEL="${coder:-models/qwen2.5-0.5b-instruct-q4_k_m.gguf}" \
  MIVI_CLI_TIMEOUT_SECS="$cli_timeout" \
  cargo run --release -- serve >"$OUT_DIR/${name}.server.log" 2>&1 &
  SERVER_PID="$!"

  if ! wait_for_server; then
    end_ms="$(date +%s%3N)"
    write_row "$name" "$role" "$reasoner" "$coder" "server boot timeout" "" "$((end_ms - start_ms))" 0
    FAILED=1
    continue
  fi

  ok=1
  agent_eval=""
  small_eval=""
  if ! agent_eval="$(MIVI_EVAL_ALLOW_FAILURES=0 MIVI_TRACE=1 MIVI_TRACE_PATH="$TRACE_PATH" python3 scripts/eval_agent_workflows.py --url "$SERVER_URL" --trace-path "$TRACE_PATH")"; then
    ok=0
  fi
  if ! small_eval="$(MIVI_EVAL_ALLOW_FAILURES=0 MIVI_EVAL_SERVER_URL="$SERVER_URL" bash scripts/eval_small_models.sh)"; then
    ok=0
  fi

  end_ms="$(date +%s%3N)"
  write_row "$name" "$role" "$reasoner" "$coder" "$agent_eval" "$small_eval" "$((end_ms - start_ms))" "$ok"
  if [[ "$ok" != "1" ]]; then
    FAILED=1
  fi
  cleanup
  SERVER_PID=""
done < <(candidate_lines)

echo "$OUT"
if [[ "$FAILED" != "0" && "$ALLOW_FAILURES" != "1" ]]; then
  exit 1
fi
