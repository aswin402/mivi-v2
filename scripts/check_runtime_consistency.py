#!/usr/bin/env python3
"""Cross-runtime-mode consistency check for MIVI.

Inspired by kimi-k3-in-c's byte-identical guarantee: the same seeded prompt
served through different runtime modes (`spawn` vs `worker-eco`/`worker-hot`)
must produce identical content and tool calls. Memory placement may change
speed; it must never change model semantics.

Starts a MIVI server per mode (subprocess), sends one deterministic request,
and diffs the outputs. Writes JSONL results to
model-eval-results/runtime-consistency-YYYYMMDD-HHMMSS.jsonl and exits
non-zero on any mismatch or server failure.

Usage:
    python3 scripts/test_runtime_consistency.py \
        --binary target/release/mivi --modes spawn,worker-eco

Stdlib only (urllib, unittest) so it runs in CI without installs.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

DEFAULT_PORT = 8210
DEFAULT_WORKER_PORT_BASE = 18210
DEFAULT_PROMPT = "Reply with exactly one word: the capital of France."
RESULTS_DIR = os.environ.get(
    "MIVI_CONSISTENCY_RESULTS_DIR", "model-eval-results"
)


def build_payload(prompt, seed=42, max_tokens=24):
    """Deterministic request: greedy sampling pinned by an explicit seed."""
    return {
        "model": "mivi",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": max_tokens,
    }


def extract_output(response):
    """Reduce an OpenAI-shaped response to its comparable semantic payload."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = [
        (
            (tool.get("function") or {}).get("name"),
            (tool.get("function") or {}).get("arguments"),
        )
        for tool in (message.get("tool_calls") or [])
    ]
    return {"content": message.get("content") or "", "tool_calls": tool_calls}


def outputs_match(a, b):
    return a == b


def wait_for_server(url, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.25)
    return False


def query_server(base_url, payload, timeout_s):
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_mode(binary, mode, port, worker_port, prompt, timeout_s):
    """Start the server in one runtime mode, ask the prompt, tear it down."""
    env = dict(os.environ)
    env.update(
        {
            "MIVI_RUNTIME_MODE": mode,
            "MIVI_PORT": str(port),
            "MIVI_WORKER_PORT": str(worker_port),
            "MIVI_WORKER_IDLE_SECS": "30",
            # Identical context budget across modes keeps prompt assembly equal.
            "MIVI_CONTEXT_BUDGET": str(os.environ.get("MIVI_CONTEXT_BUDGET", "4096")),
            "MIVI_REASONING_MODE": "no_think",
        }
    )
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [binary, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        if not wait_for_server(base_url, timeout_s):
            raise RuntimeError(f"server in mode {mode} did not become ready")
        started = time.time()
        response = query_server(base_url, build_payload(prompt), timeout_s)
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            "mode": mode,
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "output": extract_output(response),
        }
    finally:
        # Kill the whole process group: worker modes own llama-server children.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify identical model output across MIVI runtime modes"
    )
    parser.add_argument("--binary", default="target/release/mivi")
    parser.add_argument("--modes", default="spawn,worker-eco")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--worker-port-base", type=int, default=DEFAULT_WORKER_PORT_BASE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"runtime-consistency-{stamp}.jsonl")

    rows = []
    for index, mode in enumerate(modes):
        print(f"[consistency] running mode={mode}", flush=True)
        row = {
            "mode": mode,
            "ok": False,
            "elapsed_ms": None,
            "output": None,
            "error": None,
        }
        try:
            result = run_mode(
                args.binary,
                mode,
                args.port,
                args.worker_port_base + index,
                args.prompt,
                args.timeout,
            )
            row.update(result)
        except Exception as exc:  # noqa: BLE001 - report and continue
            row["error"] = str(exc)
        rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    failed = [row for row in rows if not row["ok"]]
    consistent = len(rows) >= 2 and all(
        outputs_match(rows[0]["output"], row["output"]) for row in rows[1:]
    )

    for row in rows:
        status = "ok" if row["ok"] else f"FAILED ({row['error']})"
        print(f"[consistency] {row['mode']}: {status} ({row['elapsed_ms']} ms)")
        print(f"  output: {json.dumps(row['output'])}")

    print(f"[consistency] results: {results_path}")
    if failed:
        print("[consistency] FAIL: at least one mode errored")
        return 1
    if not consistent:
        print("[consistency] FAIL: outputs differ across runtime modes")
        return 1
    print("[consistency] PASS: byte-identical outputs across all modes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
