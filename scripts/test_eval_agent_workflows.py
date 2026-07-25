#!/usr/bin/env python3
import json
import unittest

import eval_agent_workflows as workflows


def response(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": msg}]})


class AgentWorkflowEvalTests(unittest.TestCase):
    def test_large_tool_payload_contains_shell_and_many_irrelevant_tools(self):
        payload = workflows.payload_for("tool-shell-100")
        names = [tool["function"]["name"] for tool in payload["tools"]]

        self.assertIn("bash", names)
        self.assertGreaterEqual(len(names), 100)
        self.assertIn("irrelevant_tool_80", names)

    def test_injected_chat_payload_keeps_latest_real_prompt(self):
        payload = workflows.payload_for("chat-injected")
        parts = payload["messages"][-1]["content"]

        self.assertEqual(parts[-1]["text"], "Say who you are in one short sentence.")
        self.assertIn("user-prompt-submit-hook", parts[0]["text"])

    def test_tool_shell_score_requires_valid_npm_test_call(self):
        result = workflows.score_workflow(
            "tool-shell-100",
            response(
                tool_calls=[{
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"cmd": "npm test"})},
                }]
            ),
            [],
        )

        self.assertTrue(result["ok"])

    def test_invalid_tool_score_rejects_unselected_tool(self):
        result = workflows.score_workflow(
            "tool-shell-100",
            response(
                tool_calls=[{
                    "type": "function",
                    "function": {"name": "delete_everything", "arguments": "{}"},
                }]
            ),
            [],
        )

        self.assertFalse(result["ok"])
        self.assertIn("wrong shell tool name", result["reasons"])

    def test_trace_score_accepts_request_and_final_rows(self):
        trace_rows = [
            {"kind": "request", "has_tool_involvement": True, "tools_in_request": 120},
            {"kind": "tool_generation", "selected_tools": ["bash"], "rejected_tool_calls": 1},
            {"kind": "final_response", "route": "tool_calls"},
        ]

        result = workflows.score_workflow("trace-tool-shell", response(tool_calls=[{"type":"function","function":{"name":"bash","arguments":json.dumps({"cmd":"npm test"})}}]), trace_rows)

        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
