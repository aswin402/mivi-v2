#!/usr/bin/env python3
"""Unit tests for check_runtime_consistency.py (stdlib unittest only)."""

import unittest

from check_runtime_consistency import (
    build_payload,
    extract_output,
    outputs_match,
)


class BuildPayloadTests(unittest.TestCase):
    def test_payload_is_deterministic(self):
        self.assertEqual(
            build_payload("hello world"),
            build_payload("hello world"),
        )

    def test_pins_greedy_sampling_and_seed(self):
        payload = build_payload("hello", seed=7, max_tokens=12)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["top_p"], 1)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["max_tokens"], 12)
        self.assertFalse(payload["stream"])


class ExtractOutputTests(unittest.TestCase):
    def test_extracts_content_and_tool_calls(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Paris",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"cmd":"ls"}',
                                }
                            }
                        ],
                    }
                }
            ]
        }
        self.assertEqual(
            extract_output(response),
            {"content": "Paris", "tool_calls": [("bash", '{"cmd":"ls"}')]},
        )

    def test_missing_choices_yields_empty_output(self):
        self.assertEqual(
            extract_output({}),
            {"content": "", "tool_calls": []},
        )


class OutputsMatchTests(unittest.TestCase):
    def test_identical_outputs_match(self):
        a = {"content": "Paris.", "tool_calls": []}
        self.assertTrue(outputs_match(a, dict(a)))

    def test_content_difference_fails(self):
        a = {"content": "Paris.", "tool_calls": []}
        b = {"content": "paris.", "tool_calls": []}
        self.assertFalse(outputs_match(a, b))

    def test_tool_call_difference_fails(self):
        a = {"content": "", "tool_calls": [("bash", "{}")]}
        b = {"content": "", "tool_calls": [("shell", "{}")]}
        self.assertFalse(outputs_match(a, b))


if __name__ == "__main__":
    unittest.main()
