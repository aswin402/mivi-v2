#!/usr/bin/env python3
"""HTTP smoke tests for MIVI's OpenAI-compatible agent surface.

Run against an already-started server:
  scripts/smoke_openai_compat.py

Environment:
  MIVI_SMOKE_BASE_URL   default http://127.0.0.1:8000/v1
  MIVI_SMOKE_TIMEOUT    default 30
  MIVI_SMOKE_CASES      comma list, default all cases
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("MIVI_SMOKE_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_TIMEOUT = float(os.environ.get("MIVI_SMOKE_TIMEOUT", "30"))
DEFAULT_CASES = [
    "models",
    "chat-usage",
    "chat-stream-usage",
    "tool-call",
    "responses",
    "web-research-tool",
    "tool-result-loop",
    "tool-error-loop",
    "unmatched-tool-result",
    "multi-tool-result-loop",
]


def make_tool(name, description, properties=None, required=None):
    schema = {"type": "object", "properties": properties or {}}
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


def webfetch_tool():
    return make_tool(
        "webfetch",
        "Fetch a URL from the web",
        {"url": {"type": "string"}},
        ["url"],
    )


def shell_tool():
    return make_tool(
        "bash",
        "Run a shell command in the project terminal",
        {"cmd": {"type": "string"}},
        ["cmd"],
    )


def payload_for(case):
    if case == "chat-usage":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "Say your external model name."}],
        }
    if case == "chat-stream-usage":
        return {
            "model": "mivi",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Say your external model name."}],
        }
    if case == "tool-call":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [{"role": "user", "content": "Run npm test."}],
            "tools": [shell_tool()],
        }
    if case == "responses":
        return {
            "model": "mivi",
            "stream": False,
            "input": "Say your external model name.",
        }
    if case == "web-research-tool":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Research https://hono.dev/ and tell me about it.",
                }
            ],
            "tools": [webfetch_tool(), shell_tool()],
        }
    if case == "tool-result-loop":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Research https://hono.dev/ and tell me about it.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_webfetch",
                            "type": "function",
                            "function": {
                                "name": "webfetch",
                                "arguments": json.dumps({"url": "https://hono.dev/"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_webfetch",
                    "content": (
                        '---\n'
                        'title: "Hono - Web framework built on Web Standards"\n'
                        'description: Fast, lightweight, built on Web Standards.\n'
                        '---\n'
                        'Hono is a small, simple, and ultrafast web framework for JavaScript runtimes.'
                    ),
                },
            ],
            "tools": [webfetch_tool()],
        }
    if case == "tool-error-loop":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Research https://hono.dev/"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_webfetch",
                            "type": "function",
                            "function": {
                                "name": "webfetch",
                                "arguments": json.dumps({"url": "https://hono.dev/"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_webfetch",
                    "content": {"error": "network timeout", "message": "connection timed out"},
                },
            ],
            "tools": [webfetch_tool()],
        }
    if case == "unmatched-tool-result":
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
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"cmd": "cargo test"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_missing",
                    "content": "test result: ok",
                },
            ],
            "tools": [shell_tool()],
        }
    if case == "multi-tool-result-loop":
        return {
            "model": "mivi",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Research Hono and run tests."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_webfetch",
                            "type": "function",
                            "function": {
                                "name": "webfetch",
                                "arguments": json.dumps({"url": "https://hono.dev/"}),
                            },
                        },
                        {
                            "id": "call_bash",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"cmd": "cargo test"}),
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_webfetch",
                    "content": (
                        '---\n'
                        'title: "Hono"\n'
                        'description: Web framework built on Web Standards.\n'
                        '---\n'
                        'Hono is a fast web framework.'
                    ),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bash",
                    "content": "test result: ok. 152 passed",
                },
            ],
            "tools": [webfetch_tool(), shell_tool()],
        }
    raise ValueError(f"unknown smoke case: {case}")


def request_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def request_get_json(url, timeout):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_sse(url, payload, timeout):
    raw = request_json(url, payload, timeout)
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            events.append(data)
    return events


def usage_is_valid(usage):
    return (
        isinstance(usage, dict)
        and isinstance(usage.get("prompt_tokens"), int)
        and isinstance(usage.get("completion_tokens"), int)
        and isinstance(usage.get("total_tokens"), int)
        and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    )


def first_choice(data):
    return data.get("choices", [{}])[0]


def score_case(case, result):
    reasons = []
    if case == "models":
        ids = [item.get("id") for item in result.get("data", [])]
        if result.get("object") != "list":
            reasons.append("models object is not list")
        if "mivi" not in ids:
            reasons.append("missing mivi model id")
    elif case == "chat-usage":
        choice = first_choice(result)
        content = ((choice.get("message") or {}).get("content") or "").lower()
        if "mivi" not in content:
            reasons.append("chat response missing mivi identity")
        if not usage_is_valid(result.get("usage")):
            reasons.append("chat response usage missing or invalid")
    elif case == "chat-stream-usage":
        if "[DONE]" not in result:
            reasons.append("stream missing DONE marker")
        usage_events = [event for event in result if isinstance(event, dict) and "usage" in event]
        if not usage_events:
            reasons.append("stream missing usage chunk")
        elif not usage_is_valid(usage_events[-1].get("usage")):
            reasons.append("stream usage invalid")
    elif case == "tool-call":
        calls = (first_choice(result).get("message") or {}).get("tool_calls") or []
        if len(calls) != 1:
            reasons.append("expected one tool call")
        else:
            fn = calls[0].get("function") or {}
            if fn.get("name") not in {"bash", "shell", "exec_command"}:
                reasons.append("wrong shell tool name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
                reasons.append("tool arguments not json")
            if "npm test" not in (args.get("cmd") or args.get("command") or ""):
                reasons.append("missing npm test command")
        if not usage_is_valid(result.get("usage")):
            reasons.append("tool response usage missing or invalid")
    elif case == "responses":
        if result.get("object") != "response":
            reasons.append("responses object is not response")
        texts = []
        for item in result.get("output", []):
            for part in item.get("content", []):
                texts.append(part.get("text", ""))
        if "mivi" not in "\n".join(texts).lower():
            reasons.append("responses output missing mivi identity")
        if not usage_is_valid(result.get("usage")):
            reasons.append("responses usage missing or invalid")
    elif case == "web-research-tool":
        calls = (first_choice(result).get("message") or {}).get("tool_calls") or []
        if len(calls) != 1:
            reasons.append("expected one web tool call")
        else:
            fn = calls[0].get("function") or {}
            if fn.get("name") != "webfetch":
                reasons.append("wrong web tool name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
                reasons.append("web tool arguments not json")
            if args.get("url") != "https://hono.dev/":
                reasons.append("missing expected URL")
    elif case == "tool-result-loop":
        content = ((first_choice(result).get("message") or {}).get("content") or "").lower()
        if "hono" not in content:
            reasons.append("tool result summary missing page title")
        if "framework" not in content and "web standards" not in content:
            reasons.append("tool result summary missing fetched content")
        if (first_choice(result).get("message") or {}).get("tool_calls"):
            reasons.append("tool result loop repeated tool call")
        if first_choice(result).get("finish_reason") == "tool_calls":
            reasons.append("tool result loop finished with tool_calls")
        if not usage_is_valid(result.get("usage")):
            reasons.append("tool result loop usage missing or invalid")
    elif case == "tool-error-loop":
        content = ((first_choice(result).get("message") or {}).get("content") or "").lower()
        if "webfetch" not in content:
            reasons.append("tool error summary missing tool name")
        if "timeout" not in content:
            reasons.append("tool error summary missing timeout category")
        if not usage_is_valid(result.get("usage")):
            reasons.append("tool error usage missing or invalid")
    elif case == "unmatched-tool-result":
        content = ((first_choice(result).get("message") or {}).get("content") or "").lower()
        if "protocol issue" not in content:
            reasons.append("unmatched result missing protocol issue")
        if "call_missing" not in content:
            reasons.append("unmatched result missing tool_call_id")
        if not usage_is_valid(result.get("usage")):
            reasons.append("unmatched result usage missing or invalid")
    elif case == "multi-tool-result-loop":
        content = ((first_choice(result).get("message") or {}).get("content") or "").lower()
        if "tool results" not in content:
            reasons.append("multi-tool result missing aggregate heading")
        if "hono" not in content:
            reasons.append("multi-tool result missing web summary")
        if "bash" not in content or "152 passed" not in content:
            reasons.append("multi-tool result missing shell summary")
        if not usage_is_valid(result.get("usage")):
            reasons.append("multi-tool result usage missing or invalid")
    else:
        reasons.append(f"unknown case {case}")
    return {"ok": not reasons, "reasons": reasons}


def run_case(base_url, case, timeout):
    base_url = base_url.rstrip("/")
    if case == "models":
        return request_get_json(f"{base_url}/models", timeout)
    if case == "responses":
        raw = request_json(f"{base_url}/responses", payload_for(case), timeout)
        return json.loads(raw)
    if case == "chat-stream-usage":
        return request_sse(f"{base_url}/chat/completions", payload_for(case), timeout)
    raw = request_json(f"{base_url}/chat/completions", payload_for(case), timeout)
    return json.loads(raw)


def parse_cases(text):
    if not text:
        return list(DEFAULT_CASES)
    cases = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [case for case in cases if case not in DEFAULT_CASES]
    if unknown:
        raise ValueError(f"unknown smoke cases: {', '.join(unknown)}")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MIVI OpenAI-compatible HTTP smoke tests")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--cases", default=os.environ.get("MIVI_SMOKE_CASES", ",".join(DEFAULT_CASES)))
    args = parser.parse_args(argv)

    try:
        cases = parse_cases(args.cases)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2

    overall_ok = True
    for case in cases:
        try:
            result = run_case(args.base_url, case, args.timeout)
            score = score_case(case, result)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            score = {"ok": False, "reasons": [f"request failed: {exc}"]}
        overall_ok = overall_ok and score["ok"]
        print(json.dumps({"case": case, **score}, sort_keys=True))

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
