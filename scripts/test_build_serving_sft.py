#!/usr/bin/env python3
"""Unit tests for scripts/build_serving_sft.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_serving_sft as builder


def write_rendered(directory: Path, name: str, text: str):
    (directory / f"{name}.rendered.txt").write_text(
        "banner noise\nmore noise\n" + text
    )


class BuildServingSftTests(unittest.TestCase):
    def test_build_emits_prompt_completion_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp)
            write_rendered(req, "stop_job", "<|im_start|>system\nagent contract<|im_end|>\n<|im_start|>user\nStop scheduled job 1.\n<|im_start|>assistant\n")
            write_rendered(req, "coding_sum", "<|im_start|>assistant\n")
            out = Path(tmp) / "sft.jsonl"
            counts = builder.build(req, out)
            rows = [json.loads(l) for l in out.read_text().splitlines()]

            self.assertEqual(counts, {"tool_call": 1, "coding_verified": 1})
            for r in rows:
                self.assertIn("prompt", r)
                self.assertIn("completion", r)
                self.assertTrue(r["prompt"].startswith("<|im_start|>"))  # banner stripped
                if "system" in r["prompt"]:
                    self.assertIn("agent contract", r["prompt"].lower())

    def test_tool_call_target_is_grammar_exact_with_string_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp)
            write_rendered(req, "stop_job", "<|im_start|>assistant\n")
            out = Path(tmp) / "sft.jsonl"
            builder.build(req, out)
            row = json.loads(out.read_text().splitlines()[0])
            target = json.loads(row["completion"])
            call = target["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "remove_job")
            self.assertEqual(call["function"]["arguments"], {"id": "1"})  # string, not int
            self.assertIsInstance(call["function"]["arguments"]["id"], str)

    def test_coding_target_uses_verified_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp)
            write_rendered(req, "coding_sum", "<|im_start|>assistant\n")
            out = Path(tmp) / "sft.jsonl"
            builder.build(req, out)
            row = json.loads(out.read_text().splitlines()[0])
            self.assertIn("**Verified Terminal Output:**", row["completion"])
            self.assertIn("5", row["completion"])

    def test_unknown_rendered_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp)
            write_rendered(req, "unknown_case", "<|im_start|>assistant\n")
            out = Path(tmp) / "sft.jsonl"
            counts = builder.build(req, out)
            self.assertEqual(counts, {})
            self.assertEqual(out.read_text(), "")


if __name__ == "__main__":
    unittest.main()
