#!/usr/bin/env python3
"""
MIVI-V2 Serving-Format SFT Dataset Builder (finetune round 2).

Round 1 lesson: finetuning on the raw chat template taught the right behaviors
but the wrong output wrapper — MIVI serves through `build_chat_prompt` (agent
contract + <tools> block) and parses grammar-constrained
`{"tool_calls":[...]}` output.

This builder emits `{"prompt", "completion"}` rows where:

- `prompt`     = the byte-exact rendered serving prompt (produced by
                 `mivi debug-prompt <request.json>`, stored as
                 `datasets/serving_requests/*.rendered.txt`)
- `completion` = the exact target the serving path expects:
                 - tool path: minified grammar-exact
                   `{"tool_calls":[{"id":...,"type":"function","function":{
                     "name":...,"arguments":{...string values...}}}]}`
                 - coding: code block + `**Verified Terminal Output:**`

Train completion-style (prompt + completion, no chat template) — see
`train_mivi_unsloth.py --serving-format`.

Usage:
    python3 scripts/build_serving_sft.py \
        [--req-dir datasets/serving_requests] \
        [--out datasets/mivi_serving_sft.jsonl]
"""

import argparse
import json
from typing import Dict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQ_DIR = ROOT / "datasets" / "serving_requests"
DEFAULT_OUT = ROOT / "datasets" / "mivi_serving_sft.jsonl"

# name -> (target completion, category)
TARGETS = {
    "stop_job": (
        '{"tool_calls":[{"id":"call_1","type":"function","function":'
        '{"name":"remove_job","arguments":{"id":"1"}}}]}',
        "tool_call",
    ),
    "stop_job_7": (
        '{"tool_calls":[{"id":"call_1","type":"function","function":'
        '{"name":"remove_job","arguments":{"id":"7"}}}]}',
        "tool_call",
    ),
    "npm_test": (
        '{"tool_calls":[{"id":"call_1","type":"function","function":'
        '{"name":"bash","arguments":{"cmd":"npm test"}}}]}',
        "tool_call",
    ),
    "weather": (
        '{"tool_calls":[{"id":"call_1","type":"function","function":'
        '{"name":"get_weather","arguments":{"city":"Paris"}}}]}',
        "tool_call",
    ),
    "research": (
        '{"tool_calls":[{"id":"call_1","type":"function","function":'
        '{"name":"webfetch","arguments":{"url":"https://hono.dev/"}}}]}',
        "tool_call",
    ),
    "coding_sum": (
        "```python\nprint(2 + 3)\n```\n\n**Verified Terminal Output:**\n```\n5\n```",
        "coding_verified",
    ),
}


def strip_banner(rendered: str) -> str:
    """debug-prompt output includes the startup banner on stdout; keep only
    from the first chat-template marker onward."""
    idx = rendered.find("<|im_start|>")
    return rendered[idx:] if idx >= 0 else rendered


def build(req_dir: Path, out_path: Path) -> Dict[str, int]:
    rows = []
    counts: Dict[str, int] = {}
    for req_file in sorted(req_dir.glob("*.rendered.txt")):
        name = req_file.name.replace(".rendered.txt", "")
        if name not in TARGETS:
            continue
        completion, category = TARGETS[name]
        prompt = strip_banner(req_file.read_text())
        if not prompt.endswith("\n"):
            prompt += "\n"
        rows.append({"prompt": prompt, "completion": completion, "category": category})
        counts[category] = counts.get(category, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--req-dir", type=Path, default=DEFAULT_REQ_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    counts = build(args.req_dir, args.out)
    total = sum(counts.values())
    print(f"wrote {total} rows to {args.out}")
    for category, count in sorted(counts.items()):
        print(f"  {category:<20} {count}")


if __name__ == "__main__":
    main()
