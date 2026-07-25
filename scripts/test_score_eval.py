#!/usr/bin/env python3
import json
import unittest

from score_eval import score_eval


def response(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": msg}]})


class ScoreEvalTests(unittest.TestCase):
    def test_rag_rejects_qwen_confusion(self):
        result = score_eval("rag", response("The module is Qwen."))
        self.assertFalse(result["semantic_ok"])
        self.assertIn("confused model name with routing module", result["reasons"])

    def test_tool_json_requires_expected_call_and_city(self):
        result = score_eval(
            "tool-json",
            response(
                tool_calls=[{
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                }],
            ),
        )
        self.assertTrue(result["semantic_ok"])

    def test_reasoning_requires_manifest_safety_warning(self):
        result = score_eval("reasoning", response("Run cargo fetch. Delete Cargo.toml."))
        self.assertFalse(result["semantic_ok"])
        self.assertIn("missing manifest safety warning", result["reasons"])


    def test_thought_leakage_is_rejected(self):
        result = score_eval("chat", response("Start thinking private End thinking I am MIVI."))
        self.assertFalse(result["semantic_ok"])
        self.assertIn("thought leakage", result["reasons"])

    def test_shell_tool_requires_npm_test_command(self):
        result = score_eval(
            "tool-shell",
            response(
                tool_calls=[{
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"cmd": "npm test"})},
                }],
            ),
        )
        self.assertTrue(result["semantic_ok"])

    def test_context_requires_mivi_model_name(self):
        result = score_eval("context", response("Agents should call mivi."))
        self.assertTrue(result["semantic_ok"])


if __name__ == "__main__":
    unittest.main()
