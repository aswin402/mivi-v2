#!/usr/bin/env python3
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from generate_specialist_datasets import (
    generate_all_specialist_datasets,
    generate_reasoner_samples,
    generate_tools_samples,
    generate_coder_samples,
    generate_debugger_samples,
)

class TestGenerateSpecialistDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="mivi_test_datasets_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generate_reasoner_samples(self):
        samples = generate_reasoner_samples()
        self.assertGreaterEqual(len(samples), 50)
        for s in samples:
            self.assertIn("text", s)
            self.assertIn("<|im_start|>system", s["text"])
            self.assertIn("<think>", s["text"])

    def test_generate_tools_samples(self):
        samples = generate_tools_samples()
        self.assertGreaterEqual(len(samples), 50)
        for s in samples:
            self.assertIn("text", s)
            self.assertIn("<tool_call>", s["text"])

    def test_generate_coder_samples(self):
        samples = generate_coder_samples()
        self.assertGreaterEqual(len(samples), 50)
        for s in samples:
            self.assertIn("text", s)
            self.assertIn("```", s["text"])

    def test_generate_debugger_samples(self):
        samples = generate_debugger_samples()
        self.assertGreaterEqual(len(samples), 50)
        for s in samples:
            self.assertIn("text", s)
            self.assertIn("<think>", s["text"])

    def test_generate_all_specialist_datasets(self):
        counts = generate_all_specialist_datasets(self.tmp_dir)
        self.assertIn("mivi_reasoner_dataset.jsonl", counts)
        self.assertIn("mivi_tools_dataset.jsonl", counts)
        self.assertIn("mivi_coder_dataset.jsonl", counts)
        self.assertIn("mivi_debugger_dataset.jsonl", counts)

        for filename, count in counts.items():
            filepath = os.path.join(self.tmp_dir, filename)
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
                self.assertEqual(len(lines), count)

if __name__ == "__main__":
    unittest.main()
