#!/usr/bin/env python3
import json
import unittest

import smoke_openai_compat as smoke


class OpenAICompatSmokeTests(unittest.TestCase):
    def test_parse_cases_rejects_unknown_case(self):
        with self.assertRaises(ValueError):
            smoke.parse_cases("models,missing-case")

    def test_chat_usage_score_requires_valid_usage(self):
        result = {
            "choices": [{"message": {"content": "I am MIVI."}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

        self.assertTrue(smoke.score_case("chat-usage", result)["ok"])

    def test_chat_usage_score_rejects_bad_total(self):
        result = {
            "choices": [{"message": {"content": "I am MIVI."}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99},
        }

        scored = smoke.score_case("chat-usage", result)
        self.assertFalse(scored["ok"])
        self.assertIn("chat response usage missing or invalid", scored["reasons"])

    def test_stream_score_requires_done_and_usage(self):
        events = [
            {"choices": [{"delta": {"content": "MIVI"}, "finish_reason": None}]},
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            "[DONE]",
        ]

        self.assertTrue(smoke.score_case("chat-stream-usage", events)["ok"])

    def test_web_research_payload_exposes_webfetch_tool(self):
        payload = smoke.payload_for("web-research-tool")
        names = [tool["function"]["name"] for tool in payload["tools"]]

        self.assertIn("webfetch", names)
        self.assertIn("https://hono.dev/", payload["messages"][0]["content"])

    def test_web_research_score_requires_url_argument(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "webfetch",
                            "arguments": json.dumps({"url": "https://hono.dev/"}),
                        }
                    }]
                }
            }]
        }

        self.assertTrue(smoke.score_case("web-research-tool", response)["ok"])

    def test_tool_call_score_requires_usage_and_npm_test(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"cmd": "npm test"}),
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
        }

        self.assertTrue(smoke.score_case("tool-call", response)["ok"])

    def test_tool_result_loop_payload_includes_assistant_call_and_tool_result(self):
        payload = smoke.payload_for("tool-result-loop")
        roles = [message["role"] for message in payload["messages"]]

        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(payload["messages"][2]["tool_call_id"], "call_webfetch")
        self.assertIn("Hono", payload["messages"][2]["content"])

    def test_tool_result_loop_score_requires_summary_content(self):
        response = {
            "choices": [{"message": {"content": "Page: Hono. Fast web framework for JavaScript runtimes."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        self.assertTrue(smoke.score_case("tool-result-loop", response)["ok"])

    def test_tool_error_loop_score_accepts_error_summary(self):
        response = {
            "choices": [{"message": {"content": "Tool `webfetch` returned timeout: connection timed out"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        self.assertTrue(smoke.score_case("tool-error-loop", response)["ok"])

    def test_unmatched_tool_result_score_accepts_protocol_issue(self):
        response = {
            "choices": [{"message": {"content": "Tool result protocol issue: no matching assistant tool call for `call_missing`."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        self.assertTrue(smoke.score_case("unmatched-tool-result", response)["ok"])

    def test_multi_tool_result_score_requires_both_summaries(self):
        response = {
            "choices": [{"message": {"content": "Tool results:\n- Page: Hono. Web framework.\n- Tool `bash` returned: test result: ok. 152 passed"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 12, "total_tokens": 24},
        }

        self.assertTrue(smoke.score_case("multi-tool-result-loop", response)["ok"])


if __name__ == "__main__":
    unittest.main()
