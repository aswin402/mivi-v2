#!/usr/bin/env python3
import json
import unittest

import eval_tool_calling as etc


def response(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": msg}]})


class ToolCallEvalTests(unittest.TestCase):
    def test_make_tool_schema_structure(self):
        tool = etc.make_tool(
            "test_tool",
            "A test tool",
            {"param1": {"type": "string"}},
            required=["param1"],
        )
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "test_tool")
        self.assertEqual(tool["function"]["parameters"]["required"], ["param1"])

    def test_val_fn_weather_matches_exact_args(self):
        tc = next(t for t in etc.TEST_CASES if t["name"] == "weather-required-args")
        args_valid = {"city": "Paris", "country": "France", "unit": "celsius"}
        args_invalid = {"city": "London", "country": "France"}

        self.assertTrue(tc["val_fn"](args_valid))
        self.assertFalse(tc["val_fn"](args_invalid))

    def test_val_fn_profile_compliance(self):
        tc = next(t for t in etc.TEST_CASES if t["name"] == "profile-schema-compliance")
        args_valid = {"name": "Alice", "age": 25}
        args_invalid = {"name": "Bob", "age": 25}

        self.assertTrue(tc["val_fn"](args_valid))
        self.assertFalse(tc["val_fn"](args_invalid))


if __name__ == "__main__":
    unittest.main()
