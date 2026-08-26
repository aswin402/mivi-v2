#!/usr/bin/env python3
"""Unit tests for scripts/build_agentic_sft.py (stdlib only, CI-safe)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_agentic_sft as builder


class BuildAgenticSftTests(unittest.TestCase):
    def test_build_produces_all_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sft.jsonl"
            counts = builder.build(Path("/nonexistent/verified_pairs.jsonl"), out)
            rows = [json.loads(l) for l in out.read_text().splitlines()]

            for category in (
                "coding_verified_synthetic",
                "tool_selection",
                "tool_result_summary",
                "error_summary",
                "rag_grounded",
                "identity",
                "english_chat",
                "planner",
            ):
                self.assertIn(category, counts, f"missing category {category}")
                self.assertGreater(counts[category], 0)
            self.assertEqual(sum(counts.values()), len(rows))

    def test_every_row_is_valid_openai_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sft.jsonl"
            builder.build(Path("/nonexistent"), out)
            for line in out.read_text().splitlines():
                r = json.loads(line)
                self.assertIn("messages", r)
                self.assertIn("category", r)
                roles = [m["role"] for m in r["messages"]]
                self.assertEqual(roles[0], "system")
                self.assertIn("user", roles)
                self.assertEqual(roles[-1], "assistant")
                for m in r["messages"]:
                    if "tool_calls" in m:
                        for tc in m["tool_calls"]:
                            json.loads(tc["function"]["arguments"])  # args are valid JSON

    def test_stop_job_examples_pick_remove_not_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sft.jsonl"
            builder.build(Path("/nonexistent"), out)
            stop_rows = [
                json.loads(l)
                for l in out.read_text().splitlines()
                if '"remove_job"' in l
            ]
            self.assertGreaterEqual(len(stop_rows), 3)
            for r in stop_rows:
                calls = r["messages"][-2]["tool_calls"] if "tool_calls" in r["messages"][-2] else r["messages"][2]["tool_calls"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["name"], "remove_job")

    def test_verified_pairs_are_converted_with_output_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "verified_pairs.jsonl"
            pairs.write_text(
                json.dumps(
                    {
                        "instruction": "print hello",
                        "language": "python",
                        "output": "```python\nprint('hello')\n```",
                        "verified_terminal_output": "hello",
                        "verified": True,
                    }
                )
                + "\n"
                + json.dumps({"instruction": "broken", "verified": False})
                + "\n"
            )
            out = Path(tmp) / "sft.jsonl"
            counts = builder.build(pairs, out)
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            coding = [r for r in rows if r["category"] == "coding_verified"]
            self.assertEqual(len(coding), 1)  # unverified pair skipped
            self.assertIn("Verified Terminal Output", coding[0]["messages"][-1]["content"])
            self.assertIn("hello", coding[0]["messages"][-1]["content"])

    def test_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            builder.build(Path("/nonexistent"), a)
            builder.build(Path("/nonexistent"), b)
            self.assertEqual(a.read_text(), b.read_text())


if __name__ == "__main__":
    unittest.main()
