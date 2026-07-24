#!/usr/bin/env python3
import json
import sys


def extract_message(response_text):
    if not response_text:
        return "", []
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text, []
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "", []
    return msg.get("content") or "", msg.get("tool_calls") or []


def score_eval(kind, response_text):
    content, tool_calls = extract_message(response_text)
    text = content.lower()
    reasons = []

    if kind == "chat":
        if "mivi" not in text:
            reasons.append("missing mivi identity")
    elif kind == "coding":
        if "verified terminal output" not in text or "5" not in text:
            reasons.append("missing verified output 5")
    elif kind == "reasoning":
        if "cargo fetch" not in text:
            reasons.append("missing cargo fetch")
        if "do not delete project `cargo.toml` or `cargo.lock`" not in text:
            reasons.append("missing manifest safety warning")
    elif kind == "tool-json":
        if len(tool_calls) != 1:
            reasons.append("expected one tool call")
        else:
            fn = tool_calls[0].get("function", {})
            if fn.get("name") != "get_weather":
                reasons.append("wrong tool name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
                reasons.append("tool arguments not json")
            if args.get("city") != "Paris":
                reasons.append("missing city Paris")
    elif kind == "context":
        if "mivi" not in text:
            reasons.append("missing external model name mivi")
    elif kind == "rag":
        if "router" not in text and "src/router.rs" not in text:
            reasons.append("missing router source")
        if "qwen" in text:
            reasons.append("confused model name with routing module")
    else:
        reasons.append(f"unknown eval kind {kind}")

    return {
        "semantic_ok": not reasons,
        "score": 1.0 if not reasons else 0.0,
        "reasons": reasons,
        "content": content[:1000],
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: score_eval.py KIND RESPONSE_JSON")
    print(json.dumps(score_eval(sys.argv[1], sys.argv[2])))


if __name__ == "__main__":
    main()
