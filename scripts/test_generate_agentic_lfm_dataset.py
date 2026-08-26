#!/usr/bin/env python3
import json
import unittest
from pathlib import Path
import tempfile
import subprocess
import sys

class TestGenerateAgenticLfmDataset(unittest.TestCase):
    def test_dataset_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_serv = Path(tmpdir) / "serv.jsonl"
            tmp_chat = Path(tmpdir) / "chat.jsonl"
            res = subprocess.run([
                sys.executable,
                "scripts/generate_agentic_lfm_dataset.py",
                "--out-serving", str(tmp_serv),
                "--out-chatml", str(tmp_chat),
                "--multiplier", "2"
            ], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            self.assertTrue(tmp_serv.exists())
            self.assertTrue(tmp_chat.exists())

            serv_lines = [json.loads(line) for line in tmp_serv.read_text().splitlines() if line.strip()]
            chat_lines = [json.loads(line) for line in tmp_chat.read_text().splitlines() if line.strip()]

            self.assertEqual(len(serv_lines), len(chat_lines))
            self.assertGreaterEqual(len(serv_lines), 50)

            for s in serv_lines:
                self.assertIn("prompt", s)
                self.assertIn("completion", s)
                self.assertIn("category", s)
                self.assertIn("<|im_start|>", s["prompt"])
                self.assertIn("<|im_end|>", s["prompt"])

if __name__ == "__main__":
    unittest.main()
