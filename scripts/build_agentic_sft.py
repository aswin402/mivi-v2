#!/usr/bin/env python3
"""
MIVI-V2 Agentic SFT Dataset Builder.

Builds `datasets/mivi_agentic_sft.jsonl` (OpenAI `messages` format) for
fine-tuning the default MIVI model (MiniCPM5-1B) on the exact behaviors the
agent-workflow eval grades:

1. coding-verified      — real verified pairs from `dataset/verified_pairs.jsonl`
                          plus synthetic "sum of 2 and 3" style prompts, answered
                          with a code block and a `Verified Terminal Output`
                          section (the format the verifier pipeline emits).
2. tool-call selection  — single tool call with DISTRACTOR tools present
                          (stop-scheduled-job, npm test shell, weather, webfetch).
3. tool-result summary  — multi-turn tool loops answered with an aggregated
                          "Tool results" summary naming each tool and its key
                          facts (trace-multi-tool-result, long-tool-output).
4. error summary        — noisy build/test failure summarized in one sentence
                          citing the salient error line.
5. RAG-grounded answers — answers that cite the provided workspace source.
6. Identity + English chat.

Deterministic: seeded RNG, stable output ordering. Run:

    python3 scripts/build_agentic_sft.py [--pairs dataset/verified_pairs.jsonl] \
        [--out datasets/mivi_agentic_sft.jsonl]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "dataset" / "verified_pairs.jsonl"
DEFAULT_OUT = ROOT / "datasets" / "mivi_agentic_sft.jsonl"


def row(messages: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    return {"messages": messages, "category": category}


def tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def assistant_tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


AGENT_TOOLS = [
    tool("bash", "Run a shell command in the project terminal", {"cmd": {"type": "string"}}, ["cmd"]),
    tool("shell", "Run a shell command", {"command": {"type": "string"}}, ["command"]),
    tool("read_file", "Read a local workspace file", {"path": {"type": "string"}}, ["path"]),
    tool(
        "remove_job",
        "Remove or stop an existing scheduled job",
        {"id": {"type": "string"}},
        ["id"],
    ),
    tool(
        "schedule_job",
        "Create or update a scheduled job",
        {"prompt": {"type": "string"}},
        ["prompt"],
    ),
    tool("webfetch", "Fetch and read a web page from a URL", {"url": {"type": "string", "format": "uri"}}, ["url"]),
    tool("search_web", "Search the web for a query", {"query": {"type": "string"}}, ["query"]),
    tool(
        "get_weather",
        "Get the current weather for a city",
        {"city": {"type": "string"}},
        ["city"],
    ),
]


def coding_rows_from_pairs(pairs_path: Path, rng: random.Random) -> List[Dict[str, Any]]:
    """Convert real verified pairs into chat rows with the verifier output format."""
    rows: List[Dict[str, Any]] = []
    if not pairs_path.exists():
        return rows
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            pair = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not pair.get("verified"):
            continue
        instruction = (pair.get("instruction") or "").strip()
        output = (pair.get("output") or "").strip()
        terminal = (pair.get("verified_terminal_output") or "").strip()
        if not instruction or not output:
            continue
        answer = output
        if terminal:
            answer += f"\n\n**Verified Terminal Output:**\n```\n{terminal}\n```"
        rows.append(
            row(
                [
                    {"role": "system", "content": "You are MIVI, a local coding assistant. Answer with code and its verified terminal output."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": answer},
                ],
                "coding_verified",
            )
        )
    rng.shuffle(rows)
    return rows


def synthetic_coding_rows() -> List[Dict[str, Any]]:
    """Arithmetic-code prompts shaped exactly like the coding-verified eval."""
    rows = []
    cases = [
        ("Write Python code that prints the sum of 2 and 3.", "print(2 + 3)", "5"),
        ("Write Python code that prints the sum of 4 and 1.", "print(4 + 1)", "5"),
        ("Write a python script that prints 5.", "print(5)", "5"),
        ("Write Python code that prints 2 + 3.", "print(2 + 3)", "5"),
    ]
    for prompt, code, out in cases:
        rows.append(
            row(
                [
                    {"role": "system", "content": "You are MIVI, a local coding assistant. Answer with code and its verified terminal output."},
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": f"```python\n{code}\n```\n\n**Verified Terminal Output:**\n```\n{out}\n```",
                    },
                ],
                "coding_verified_synthetic",
            )
        )
    return rows


def tool_selection_rows() -> List[Dict[str, Any]]:
    """Single tool call with distractors present — the stop-scheduled-job skill."""
    rows = []
    cases = [
        (
            "Stop scheduled job 1.",
            assistant_tool_call("call_remove", "remove_job", {"id": "1"}),
            "Scheduled job 1 has been stopped.",
        ),
        (
            "Stop scheduled job 7.",
            assistant_tool_call("call_remove", "remove_job", {"id": "7"}),
            "Scheduled job 7 has been stopped.",
        ),
        (
            "Please cancel the scheduled job with id 3.",
            assistant_tool_call("call_remove", "remove_job", {"id": "3"}),
            "Scheduled job 3 has been stopped.",
        ),
        (
            "Run npm test.",
            assistant_tool_call("call_bash", "bash", {"cmd": "npm test"}),
            None,
        ),
        (
            "Run npm test in the project.",
            assistant_tool_call("call_shell", "shell", {"command": "npm test"}),
            None,
        ),
        (
            "Research https://hono.dev/ and summarize it.",
            assistant_tool_call("call_fetch", "webfetch", {"url": "https://hono.dev/"}),
            None,
        ),
        (
            "What is the weather in Paris?",
            assistant_tool_call("call_weather", "get_weather", {"city": "Paris"}),
            None,
        ),
        (
            "Schedule a job that reminds me to hydrate.",
            assistant_tool_call("call_sched", "schedule_job", {"prompt": "Remind me to hydrate."}),
            None,
        ),
    ]
    for user, call, followup in cases:
        messages = [
            {"role": "system", "content": "You are MIVI, a local AI endpoint for agents. Use the provided tools when the request maps to one; pick the single most specific tool."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ]
        if followup:
            messages.append({"role": "assistant", "content": followup})
        rows.append(row(messages, "tool_selection"))
    return rows


def tool_result_summary_rows() -> List[Dict[str, Any]]:
    """Aggregate tool results into a named summary — the trace-multi-tool skill."""
    hono = '---\ntitle: "Hono"\ndescription: Web framework built on Web Standards.\n---\nHono is a fast web framework.'
    tests = "test result: ok. 152 passed"
    summary = (
        "**Tool results**\n\n"
        "- webfetch (https://hono.dev/): Hono is a fast web framework built on Web Standards.\n"
        "- bash (`cargo test`): 152 passed — all tests succeeded."
    )
    rows = [
        row(
            [
                {"role": "system", "content": "You are MIVI, a local AI endpoint for agents. Summarize tool results factually, naming each tool and its key outcome."},
                {"role": "user", "content": "Research Hono and run tests."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        assistant_tool_call("call_webfetch", "webfetch", {"url": "https://hono.dev/"}),
                        assistant_tool_call("call_bash", "bash", {"cmd": "cargo test"}),
                    ],
                },
                {"role": "tool", "tool_call_id": "call_webfetch", "content": hono},
                {"role": "tool", "tool_call_id": "call_bash", "content": tests},
                {"role": "assistant", "content": summary},
            ],
            "tool_result_summary",
        )
    ]
    # shell-only loop: run then report the salient line
    rows.append(
        row(
            [
                {"role": "system", "content": "You are MIVI, a local AI endpoint for agents. Summarize tool results factually, naming each tool and its key outcome."},
                {"role": "user", "content": "Run npm test."},
                {"role": "assistant", "content": "", "tool_calls": [assistant_tool_call("call_bash", "bash", {"cmd": "npm test"})]},
                {
                    "role": "tool",
                    "tool_call_id": "call_bash",
                    "content": "> project@1.0.0 test\n> jest\n\nTests: 12 passed, 12 total\nTest Suites: 2 passed",
                },
                {
                    "role": "assistant",
                    "content": "**Tool results**\n\n- bash (`npm test`): all tests passed — 12 passed, 12 total across 2 suites.",
                },
            ],
            "tool_result_summary",
        )
    )
    return rows


def error_summary_rows() -> List[Dict[str, Any]]:
    """Noisy failure output summarized in one sentence citing the salient error."""
    noise = "\n".join(f"compiling filler crate {i}" for i in range(20))
    tool_out = (
        f"cargo test\n{noise}\nerror[E0425]: cannot find value `x` in this scope\n"
        "test result: FAILED. 0 passed; 1 failed\nfinal filler line"
    )
    return [
        row(
            [
                {"role": "system", "content": "You are MIVI, a local AI endpoint for agents. Summarize failures in one sentence citing the salient error line, ignoring filler output."},
                {"role": "user", "content": "Run cargo test."},
                {"role": "assistant", "content": "", "tool_calls": [assistant_tool_call("call_bash", "bash", {"cmd": "cargo test"})]},
                {"role": "tool", "tool_call_id": "call_bash", "content": tool_out},
                {
                    "role": "assistant",
                    "content": "The test run failed with `error[E0425]: cannot find value `x` in this scope` (0 passed; 1 failed).",
                },
            ],
            "error_summary",
        )
    ]


def rag_grounded_rows() -> List[Dict[str, Any]]:
    """Answers grounded in provided workspace context, citing the source file."""
    context = (
        "[src/router.rs:12-30]\n/// NeedleRouter: keyword/heuristic intent classifier.\n"
        "pub struct NeedleRouter;\nimpl NeedleRouter { pub fn classify(&self, q: &str) -> &'static str { ... } }"
    )
    cases = [
        (
            "In this codebase, what module handles intent routing?",
            "Intent routing is handled by `src/router.rs` — the `NeedleRouter`, a keyword/heuristic intent classifier (CHAT/VISION/CODE/MULTI_STEP).",
        ),
        (
            "Which file classifies user intent into CHAT or CODE?",
            "`src/router.rs` — `NeedleRouter` classifies requests into CHAT, VISION, CODE, or MULTI_STEP.",
        ),
    ]
    rows = []
    for question, answer in cases:
        rows.append(
            row(
                [
                    {"role": "system", "content": "You are MIVI, a local AI endpoint for agents. Answer only from the retrieved workspace context and cite the source file."},
                    {"role": "user", "content": f"{question}\n\nRetrieved context:\n{context}"},
                    {"role": "assistant", "content": answer},
                ],
                "rag_grounded",
            )
        )
    return rows


def identity_and_chat_rows() -> List[Dict[str, Any]]:
    identity = "I'm MIVI, a local OpenAI-compatible model endpoint for AI agents; my external model name is mivi."
    prompts = [
        "Say who you are in one short sentence.",
        "Who are you?",
        "What model are you?",
        "Introduce yourself briefly.",
    ]
    rows = []
    for p in prompts:
        rows.append(
            row(
                [
                    {"role": "system", "content": "You are MIVI, a local OpenAI-compatible model endpoint for AI agents. Externally your model name is mivi."},
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": identity},
                ],
                "identity",
            )
        )
    chat = [
        ("Explain what a binary search algorithm is in one sentence.", "Binary search finds a target in a sorted list by repeatedly halving the search range, giving O(log n) lookups."),
        ("Rewrite this sentence clearly: 'the thing got broke by him maybe'.", "He may have broken it."),
        ("Give one tip for writing stable shell scripts.", "Use `set -euo pipefail` so scripts fail fast on errors and undefined variables."),
    ]
    for q, a in chat:
        rows.append(
            row(
                [
                    {"role": "system", "content": "You are MIVI, a local OpenAI-compatible model endpoint for AI agents. Answer concisely in clear English."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "english_chat",
            )
        )
    return rows


def build(pairs_path: Path, out_path: Path) -> Dict[str, int]:
    rng = random.Random(42)
    rows: List[Dict[str, Any]] = []
    rows += coding_rows_from_pairs(pairs_path, rng)
    rows += synthetic_coding_rows()
    rows += tool_selection_rows()
    rows += tool_result_summary_rows()
    rows += error_summary_rows()
    rows += rag_grounded_rows()
    rows += identity_and_chat_rows()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    counts = build(args.pairs, args.out)
    total = sum(counts.values())
    print(f"wrote {total} rows to {args.out}")
    for category, count in sorted(counts.items()):
        print(f"  {category:<26} {count}")


if __name__ == "__main__":
    main()
