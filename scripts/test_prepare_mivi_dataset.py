#!/usr/bin/env python3
"""
Unit tests for scripts/prepare_mivi_dataset.py
Verifies dataset generation, JSON structure, tag balance, and tool schema compliance.
"""

import json
import unittest
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))
from prepare_mivi_dataset import (
    build_dataset,
    generate_tool_calling_samples,
    generate_compiler_self_correction_samples,
    generate_context_grounded_qa_samples,
    MIVI_TOOLS
)

class TestPrepareMiviDataset(unittest.TestCase):
    def test_mivi_tools_schemas_are_valid(self):
        """Verify all MIVI tool declarations have required OpenAI function schema keys."""
        self.assertGreater(len(MIVI_TOOLS), 0)
        for tool in MIVI_TOOLS:
            self.assertEqual(tool["type"], "function")
            func = tool["function"]
            self.assertIn("name", func)
            self.assertIn("description", func)
            self.assertIn("parameters", func)
            self.assertEqual(func["parameters"]["type"], "object")
            self.assertIn("properties", func["parameters"])
            self.assertIn("required", func["parameters"])

    def test_tool_calling_samples_have_balanced_tags(self):
        """Verify Hermes XML format has balanced <think> and <tool_call> tags."""
        samples = generate_tool_calling_samples()
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("messages", s)
            self.assertIn("category", s)
            messages = s["messages"]
            self.assertGreaterEqual(len(messages), 2)
            
            if s["category"] == "tool_calling_hermes":
                assistant_content = messages[-1]["content"]
                self.assertIn("<think>", assistant_content)
                self.assertIn("</think>", assistant_content)
                self.assertIn("<tool_call>", assistant_content)
                self.assertIn("</tool_call>", assistant_content)
                
                # Check tool_call JSON validity
                tool_call_match = assistant_content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                parsed = json.loads(tool_call_match)
                self.assertIn("name", parsed)
                self.assertIn("arguments", parsed)

    def test_code_correction_samples_structure(self):
        """Verify compiler error self-correction samples have valid multi-turn flow."""
        samples = generate_compiler_self_correction_samples()
        self.assertGreater(len(samples), 0)
        for s in samples:
            messages = s["messages"]
            self.assertEqual(len(messages), 5) # system, user, assistant_buggy, user_error, assistant_fixed
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            self.assertEqual(messages[2]["role"], "assistant")
            self.assertEqual(messages[3]["role"], "user")
            self.assertEqual(messages[4]["role"], "assistant")
            self.assertIn("<think>", messages[4]["content"])
            self.assertIn("```", messages[4]["content"])

    def test_grounded_qa_anti_hallucination(self):
        """Verify context-grounded Q&A samples instruct strict adherence to context."""
        samples = generate_context_grounded_qa_samples()
        self.assertGreater(len(samples), 0)
        for s in samples:
            messages = s["messages"]
            self.assertIn("Use ONLY the following context", messages[0]["content"])
            self.assertIn("<think>", messages[-1]["content"])

    def test_build_dataset_creates_requested_number_of_samples(self):
        """Verify build_dataset generates correct sample count and serializes properly."""
        dataset = build_dataset(total_samples=100)
        self.assertEqual(len(dataset), 100)
        for entry in dataset:
            serialized = json.dumps(entry)
            deserialized = json.loads(serialized)
            self.assertIn("messages", deserialized)

if __name__ == "__main__":
    unittest.main()
