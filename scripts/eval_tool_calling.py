#!/usr/bin/env python3
"""Tool Call Evaluation Suite for MIVI.

Evaluates:
1. Tool call extraction and argument parsing.
2. Tool selection accuracy (filtering out irrelevant tools).
3. Parameter constraint and JSON Schema compliance.
4. Tool choice enforcement (tool_choice).

Usage:
  python3 scripts/eval_tool_calling.py
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SERVER_URL = os.environ.get(
    "MIVI_EVAL_SERVER_URL", "http://127.0.0.1:8000/v1/chat/completions"
)
DEFAULT_TIMEOUT = float(os.environ.get("MIVI_EVAL_TIMEOUT", "180"))
TRACE_PATH = os.environ.get("MIVI_TRACE_PATH", "logs/mivi-trace.jsonl")


def make_tool(name, description, properties=None, required=None):
    properties = properties or {}
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


# Test Cases
TEST_CASES = []


def register_test(name, prompt, tools, tool_choice=None, expected_tool=None, val_fn=None):
    TEST_CASES.append({
        "name": name,
        "prompt": prompt,
        "tools": tools,
        "tool_choice": tool_choice,
        "expected_tool": expected_tool,
        "val_fn": val_fn,
    })


# 1. Weather check (required arguments check)
register_test(
    name="weather-required-args",
    prompt="Check the current weather in Paris, France using Celsius unit.",
    tools=[
        make_tool(
            "get_weather",
            "Get current weather for a city",
            {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            required=["city", "country"],
        )
    ],
    expected_tool="get_weather",
    val_fn=lambda args: (
        "city" in args
        and "paris" in str(args["city"]).lower()
        and "country" in args
        and "france" in str(args["country"]).lower()
    ),
)

# 2. Tool choice enforcement
register_test(
    name="tool-choice-enforcement",
    prompt="Look up weather information for New York City.",
    tools=[
        make_tool(
            "get_weather",
            "Get weather details",
            {"city": {"type": "string"}},
            required=["city"],
        ),
        make_tool(
            "search_web",
            "Search online for info",
            {"query": {"type": "string"}},
            required=["query"],
        ),
    ],
    tool_choice={"type": "function", "function": {"name": "search_web"}},
    expected_tool="search_web",
    val_fn=lambda args: "query" in args and any(term in str(args["query"]).lower() for term in ["new york", "weather", "nyc"]),
)

# 3. User profile creation with nested structure
register_test(
    name="profile-schema-compliance",
    prompt="Create user profile for a user named Alice who is 25 years old.",
    tools=[
        make_tool(
            "create_profile",
            "Create a new user profile",
            {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "interests": {"type": "array", "items": {"type": "string"}},
            },
            required=["name", "age"],
        )
    ],
    expected_tool="create_profile",
    val_fn=lambda args: (
        "name" in args
        and str(args["name"]).lower() == "alice"
        and "age" in args
        and int(args["age"]) == 25
    ),
)

# 4. Math calculator (integer/number parameter constraints)
register_test(
    name="math-param-constraints",
    prompt="Add 15 and 35 together.",
    tools=[
        make_tool(
            "calculator",
            "Perform basic math",
            {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
            },
            required=["a", "b", "op"],
        )
    ],
    expected_tool="calculator",
    val_fn=lambda args: (
        "a" in args
        and float(args["a"]) == 15.0
        and "b" in args
        and float(args["b"]) == 35.0
        and args.get("op") == "add"
    ),
)


def read_trace_events():
    if not os.path.exists(TRACE_PATH):
        return []
    events = []
    with open(TRACE_PATH, "r") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def run_eval(server_url, timeout):
    print(f"Starting Tool Call Eval Suite against: {server_url}")
    print(f"Total Test Cases: {len(TEST_CASES)}\n")

    results = []
    passed_count = 0

    # Clear trace log before running to isolate trace reads
    if os.path.exists(TRACE_PATH):
        try:
            os.remove(TRACE_PATH)
        except Exception:
            pass

    for tc in TEST_CASES:
        name = tc["name"]
        print(f"Running test: {name:.<40} ", end="", flush=True)

        payload = {
            "model": "mivi",
            "messages": [{"role": "user", "content": tc["prompt"]}],
            "tools": tc["tools"],
            "stream": False,
        }
        if tc["tool_choice"]:
            payload["tool_choice"] = tc["tool_choice"]

        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            server_url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                elapsed = (time.time() - start_time) * 1000
                res_body = f.read().decode()
                res_data = json.loads(res_body)
        except Exception as e:
            print(f"\033[91mFAILED\033[0m (Error: {e})")
            results.append({
                "name": name,
                "ok": False,
                "elapsed_ms": 0,
                "reasons": [f"HTTP request failed: {e}"],
            })
            continue

        # Parse choices and tool calls
        try:
            choices = res_data.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
        except Exception as e:
            print(f"\033[91mFAILED\033[0m (Malformed response JSON: {e})")
            results.append({
                "name": name,
                "ok": False,
                "elapsed_ms": elapsed,
                "reasons": [f"Malformed response payload: {e}"],
            })
            continue

        reasons = []
        # Assertions
        if not tool_calls:
            reasons.append("Model did not generate any tool call")
        else:
            first_call = tool_calls[0]
            func = first_call.get("function", {})
            actual_name = func.get("name")
            raw_args = func.get("arguments", "{}")

            if actual_name != tc["expected_tool"]:
                reasons.append(
                    f"Expected tool '{tc['expected_tool']}', got '{actual_name}'"
                )

            try:
                parsed_args = json.loads(raw_args)
                if tc["val_fn"] and not tc["val_fn"](parsed_args):
                    reasons.append(
                        f"Arguments validation failed for: {parsed_args}"
                    )
            except json.JSONDecodeError:
                reasons.append(f"Arguments are not valid JSON: '{raw_args}'")

        # Read trace metadata if available
        trace_events = read_trace_events()
        self_corrected = False
        for ev in trace_events:
            # Check if there was self-correction (retries > 0 or rejected tool calls)
            if ev.get("kind") == "tool_generation" and ev.get("rejected_tool_calls", 0) > 0:
                self_corrected = True

        ok = len(reasons) == 0
        if ok:
            passed_count += 1
            corr_str = " (Self-Corrected)" if self_corrected else ""
            print(f"\033[92mPASSED\033[0m in {elapsed:.0f}ms{corr_str}")
        else:
            print(f"\033[91mFAILED\033[0m in {elapsed:.0f}ms")
            for r in reasons:
                print(f"  - {r}")

        results.append({
            "name": name,
            "ok": ok,
            "elapsed_ms": elapsed,
            "self_corrected": self_corrected,
            "reasons": reasons,
        })

    # Write report to model-eval-results/
    out_dir = Path("model-eval-results")
    out_dir.mkdir(exist_ok=True)
    report_file = out_dir / f"tool-calling-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(report_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 50)
    print("TOOL CALL EVALUATION SCORECARD")
    print("=" * 50)
    print(f"Passed: {passed_count}/{len(TEST_CASES)} ({passed_count/len(TEST_CASES)*100:.1f}%)")
    print(f"Report saved to: {report_file}")
    print("=" * 50)

    return passed_count == len(TEST_CASES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Call Evaluation Suite")
    parser.add_argument(
        "--url", default=DEFAULT_SERVER_URL, help="Completions endpoint URL"
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout"
    )
    args = parser.parse_args()

    success = run_eval(args.url, args.timeout)
    sys.exit(0 if success else 1)
