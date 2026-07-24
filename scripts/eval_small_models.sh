#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="model-eval-results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/small-model-$(date +%Y%m%d-%H%M%S).jsonl"
SERVER_URL="${MIVI_EVAL_SERVER_URL:-http://127.0.0.1:8000/v1/chat/completions}"
FAILED=0

PROMPTS=(
  "chat|Say who you are in one short sentence."
  "coding|Write Python code that prints the sum of 2 and 3."
  "reasoning|A tool failed because Cargo cache is corrupted. Explain the safest fix in two steps."
  "tool-json|Use the get_weather tool for Paris."
  "tool-shell|Run npm test."
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
  elif [[ "$kind" == "tool-shell" ]]; then
    python3 -c 'import json,sys; prompt=sys.argv[1]; tools=[{"type":"function","function":{"name":"bash","description":"Run a shell command in the project terminal","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]; tools += [{"type":"function","function":{"name":f"irrelevant_tool_{i}","description":"Unrelated plugin action","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}} for i in range(100)]; print(json.dumps({"model":"mivi","messages":[{"role":"user","content":prompt}],"stream":False,"tools":tools}))' "$prompt"
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
  score_json="$(python3 scripts/score_eval.py "$kind" "$response")"
  python3 -c 'import json,sys; score=json.loads(sys.argv[4]); print(json.dumps({"kind":sys.argv[1],"elapsed_ms":int(sys.argv[2]),"http_ok":bool(sys.argv[3]),"semantic_ok":bool(score["semantic_ok"]),"score":float(score["score"]),"reasons":score["reasons"],"content":score["content"],"response":sys.argv[3][:2000]}))' \
    "$kind" "$((end - start))" "$response" "$score_json" >>"$OUT"
  if ! python3 -c 'import json,sys; raise SystemExit(0 if json.loads(sys.argv[1])["semantic_ok"] else 1)' "$score_json"; then
    FAILED=1
  fi
done

echo "$OUT"
if [[ "$FAILED" != "0" && "${MIVI_EVAL_ALLOW_FAILURES:-0}" != "1" ]]; then
  exit 1
fi
