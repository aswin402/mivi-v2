#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SERVER_URL = os.environ.get(
    "MIVI_EVAL_SERVER_URL", "http://127.0.0.1:8000/v1/chat/completions"
)
DEFAULT_TIMEOUT = float(os.environ.get("MIVI_EVAL_TIMEOUT", "180"))

WORKFLOWS = [
    "chat-injected",
    "coding-verified",
    "tool-json",
    "tool-shell-100",
    "long-tool-output",
    "rag-router",
    "memory-model-name",
    "trace-tool-shell",
]


def make_tool(name, description, properties=None, required=None):
    properties = properties or {"value": {"type": "string"}}
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


def large_agent_tools(total=120):
    tools = [
        make_tool("bash", "Run a shell command in the project terminal", {"cmd": {"type": "string"}}, ["cmd"]),
        make_tool("read", "Read a file from the workspace", {"path": {"type": "string"}}, ["path"]),
        make_tool("apply_patch", "Edit files by applying a patch", {"patch": {"type": "string"}}, ["patch"]),
        make_tool("grep", "Search text in the workspace", {"pattern": {"type": "string"}}, ["pattern"]),
        make_tool("get_weather", "Get weather for a city", {"city": {"type": "string"}}, ["city"]),
    ]
    for index in range(max(0, total - len(tools))):
        tools.append(make_tool(f"irrelevant_tool_{index}", "Unrelated plugin or agent action"))
    return tools


def payload_for(kind):
    if kind == "chat-injected":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "<available-skills>Use read apply_patch bash and many tools</available-skills>",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<user-prompt-submit-hook>{\"hookSpecificOutput\":{\"additionalContext\":\"tool metadata\"}}</user-prompt-submit-hook>",
                        },
                        {"type": "text", "text": "Say who you are in one short sentence."},
                    ],
                },
            ],
            "tools": large_agent_tools(),
        }
    if kind == "coding-verified":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "Write Python code that prints the sum of 2 and 3."}],
        }
    if kind == "tool-json":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "Use the get_weather tool for Paris."}],
            "tools": [make_tool("get_weather", "Get weather for a city", {"city": {"type": "string"}}, ["city"])],
        }
    if kind in {"tool-shell-100", "trace-tool-shell"}:
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "Run npm test."}],
            "tools": large_agent_tools(),
        }
    if kind == "long-tool-output":
        noise = "\n".join(f"compiling filler crate {index}" for index in range(80))
        return {
            "model": "mivi",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Run cargo test."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bash",
                            "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"cmd": "cargo test"})},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bash",
                    "content": f"cargo test\n{noise}\nerror[E0425]: cannot find value `x` in this scope\ntest result: FAILED. 0 passed; 1 failed\nfinal filler line",
                },
                {"role": "user", "content": "Summarize the failure in one sentence."},
            ],
            "tools": large_agent_tools(),
        }
    if kind == "rag-router":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "In this codebase, what module handles intent routing?"}],
        }
    if kind == "memory-model-name":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "Using the project memory, what model name should agents call?"}],
        }
    raise ValueError(f"unknown workflow kind {kind}")


def extract_message(response_text):
    if not response_text:
        return "", []
    try:
        data = json.loads(response_text)
        message = data["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return response_text, []
    return message.get("content") or "", message.get("tool_calls") or []


def parse_tool_arguments(tool_call):
    raw = (tool_call.get("function") or {}).get("arguments") or "{}"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return {}, f"tool arguments not json: {exc}"


def trace_has(trace_rows, kind=None, route=None):
    for row in trace_rows:
        if kind is not None and row.get("kind") != kind:
            continue
        if route is not None and row.get("route") != route:
            continue
        return True
    return False


def score_workflow(kind, response_text, trace_rows):
    content, tool_calls = extract_message(response_text)
    text = content.lower()
    reasons = []

    if kind == "chat-injected":
        if tool_calls:
            reasons.append("plain injected chat produced tool calls")
        if "mivi" not in text:
            reasons.append("missing mivi identity")
    elif kind == "coding-verified":
        if "verified terminal output" not in text or "5" not in text:
            reasons.append("missing verified coding output")
    elif kind == "tool-json":
        if len(tool_calls) != 1:
            reasons.append("expected one tool call")
        else:
            fn = tool_calls[0].get("function", {})
            if fn.get("name") != "get_weather":
                reasons.append("wrong weather tool name")
            args, error = parse_tool_arguments(tool_calls[0])
            if error:
                reasons.append(error)
            if args.get("city") != "Paris":
                reasons.append("missing city Paris")
    elif kind in {"tool-shell-100", "trace-tool-shell"}:
        if len(tool_calls) != 1:
            reasons.append("expected one shell tool call")
        else:
            fn = tool_calls[0].get("function", {})
            if fn.get("name") not in {"bash", "shell", "exec_command"}:
                reasons.append("wrong shell tool name")
            args, error = parse_tool_arguments(tool_calls[0])
            if error:
                reasons.append(error)
            command = (args.get("cmd") or args.get("command") or "").lower()
            if "npm test" not in command:
                reasons.append("missing npm test command")
        if kind == "trace-tool-shell":
            if not trace_has(trace_rows, "request"):
                reasons.append("missing request trace row")
            if not trace_has(trace_rows, "tool_generation"):
                reasons.append("missing tool generation trace row")
            if not trace_has(trace_rows, "final_response", "tool_calls"):
                reasons.append("missing tool_calls final trace row")
    elif kind == "long-tool-output":
        salient_failure = (
            "error[e0425]" in text
            or "cannot find value" in text
            or "undefined variable" in text
            or "unable to find the value" in text
            or "failed" in text
        )
        if not salient_failure:
            reasons.append("missing salient tool failure")
        if "final filler line" in text:
            reasons.append("leaked low-value tool filler")
    elif kind == "rag-router":
        if "router" not in text and "src/router.rs" not in text:
            reasons.append("missing router source")
        if "qwen" in text:
            reasons.append("confused model name with routing module")
    elif kind == "memory-model-name":
        if "mivi" not in text:
            reasons.append("missing external model name mivi")
    else:
        reasons.append(f"unknown workflow kind {kind}")

    return {
        "ok": not reasons,
        "score": 1.0 if not reasons else 0.0,
        "reasons": reasons,
        "content": content[:1000],
        "tool_calls": tool_calls[:4],
    }


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_trace_rows(path, start_size):
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(start_size)
        rows = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"kind": "invalid_trace_row", "raw": line[:240]})
        return rows


def output_path():
    out_dir = Path("model-eval-results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return out_dir / f"agent-workflows-{stamp}.jsonl"


def run_workflows(url, timeout, kinds, trace_path):
    out = output_path()
    failed = False
    for kind in kinds:
        payload = payload_for(kind)
        start_size = trace_path.stat().st_size if trace_path and trace_path.exists() else 0
        start = time.time()
        response_text = ""
        http_ok = False
        error = None
        try:
            response_text = post_json(url, payload, timeout)
            http_ok = True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = str(exc)
        elapsed_ms = int((time.time() - start) * 1000)
        trace_rows = load_trace_rows(trace_path, start_size) if trace_path else []
        score = score_workflow(kind, response_text, trace_rows)
        ok = http_ok and score["ok"]
        failed = failed or not ok
        row = {
            "kind": kind,
            "elapsed_ms": elapsed_ms,
            "http_ok": http_ok,
            "ok": ok,
            "score": score["score"],
            "reasons": score["reasons"] + ([error] if error else []),
            "content": score["content"],
            "tool_calls": score["tool_calls"],
            "trace_rows": len(trace_rows),
            "response_preview": response_text[:2000],
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    return out, failed


def main():
    parser = argparse.ArgumentParser(description="Run MIVI agent workflow evals against a live server.")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--kinds", default=",".join(WORKFLOWS), help="Comma-separated workflow kinds")
    parser.add_argument("--trace-path", default=os.environ.get("MIVI_TRACE_PATH", "logs/mivi-trace.jsonl"))
    parser.add_argument("--allow-failures", action="store_true", default=os.environ.get("MIVI_EVAL_ALLOW_FAILURES") == "1")
    args = parser.parse_args()

    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    trace_path = Path(args.trace_path) if args.trace_path else None
    out, failed = run_workflows(args.url, args.timeout, kinds, trace_path)
    print(out)
    if failed and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
