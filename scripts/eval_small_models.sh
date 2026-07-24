#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="model-eval-results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/small-model-$(date +%Y%m%d-%H%M%S).jsonl"
SERVER_URL="${MIVI_EVAL_SERVER_URL:-http://127.0.0.1:8000/v1/chat/completions}"

PROMPTS=(
  "chat|Say who you are in one short sentence."
  "coding|Write Python code that prints the sum of 2 and 3."
  "reasoning|A tool failed because Cargo cache is corrupted. Explain the safest fix in two steps."
  "tool-json|Use the get_weather tool for Paris."
  "context|Using the project memory, what model name should agents call?"
  "rag|In this codebase, what module handles intent routing?"
)

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

payload_for() {
  local kind="$1"
  local prompt="$2"
  if [[ "$kind" == "tool-json" ]]; then
    cat <<JSON
{"model":"mivi","messages":[{"role":"user","content":$(printf '%s' "$prompt" | json_escape)}],"stream":false,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
JSON
  else
    cat <<JSON
{"model":"mivi","messages":[{"role":"user","content":$(printf '%s' "$prompt" | json_escape)}],"stream":false}
JSON
  fi
}

for item in "${PROMPTS[@]}"; do
  kind="${item%%|*}"
  prompt="${item#*|}"
  start="$(date +%s%3N)"
  response="$(curl -fsS --max-time "${MIVI_EVAL_TIMEOUT:-180}" "$SERVER_URL" -H 'Content-Type: application/json' -d "$(payload_for "$kind" "$prompt")" || true)"
  end="$(date +%s%3N)"
  python3 -c 'import json,sys; print(json.dumps({"kind":sys.argv[1],"elapsed_ms":int(sys.argv[2]),"ok":bool(sys.argv[3]),"response":sys.argv[3][:2000]}))' \
    "$kind" "$((end - start))" "$response" >>"$OUT"
done

echo "$OUT"
