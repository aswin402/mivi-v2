#!/usr/bin/env python3
import unittest

import check_agent_compat as check


class AgentCompatCheckTests(unittest.TestCase):
    def test_default_plan_runs_local_checks_and_skips_live_when_server_down(self):
        plan = check.build_plan(live=False, eval_live=False)
        names = [step.name for step in plan]

        self.assertEqual(names, [
            "python-tests",
            "rust-tests",
            "rustfmt",
            "release-build",
        ])

    def test_live_plan_adds_smoke_and_eval_checks(self):
        plan = check.build_plan(live=True, eval_live=True)
        names = [step.name for step in plan]

        self.assertIn("live-smoke", names)
        self.assertIn("live-consistency", names)
        self.assertIn("live-agent-eval", names)
        self.assertLess(names.index("release-build"), names.index("live-smoke"))
        self.assertLess(names.index("release-build"), names.index("live-consistency"))

    def test_probe_server_returns_false_for_unreachable_port(self):
        self.assertFalse(check.server_is_reachable("http://127.0.0.1:9/v1", timeout=0.05))

    def test_live_eval_url_is_derived_from_custom_base_url(self):
        plan = check.build_plan(live=True, eval_live=True, base_url="http://127.0.0.1:9000/v1")
        eval_step = next(step for step in plan if step.name == "live-agent-eval")

        self.assertIn("http://127.0.0.1:9000/v1/chat/completions", eval_step.cmd)

    def test_auto_live_runs_smoke_but_not_trace_eval_by_default(self):
        live, eval_live = check.choose_live_checks("auto", "auto", reachable=True)

        self.assertTrue(live)
        self.assertFalse(eval_live)

    def test_explicit_live_eval_runs_when_server_reachable(self):
        live, eval_live = check.choose_live_checks("auto", "on", reachable=True)

        self.assertTrue(live)
        self.assertTrue(eval_live)

    def test_makefile_check_agent_target_uses_runner(self):
        makefile = (check.ROOT / "Makefile").read_text()

        self.assertIn("check-agent:", makefile)
        self.assertIn("scripts/check_agent_compat.py --live off", makefile)

    def test_github_ci_uses_runner(self):
        workflow = (check.ROOT / ".github" / "workflows" / "agent-compat.yml").read_text()

        self.assertIn("scripts/check_agent_compat.py --live off", workflow)


if __name__ == "__main__":
    unittest.main()
