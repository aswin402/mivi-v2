#!/usr/bin/env python3
"""Run MIVI agent compatibility checks from one command.

Default: local-only checks that do not require a running server.
Use --live auto|on|off to control HTTP smoke checks against MIVI.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.environ.get("MIVI_SMOKE_BASE_URL", "http://127.0.0.1:8000/v1")


def chat_url_for(base_url):
    return os.environ.get("MIVI_EVAL_SERVER_URL", f"{base_url.rstrip('/')}/chat/completions")


@dataclass(frozen=True)
class Step:
    name: str
    cmd: tuple[str, ...]
    cwd: Path = ROOT
    env: dict[str, str] | None = None


def server_is_reachable(base_url, timeout=0.5):
    url = base_url.rstrip('/') + '/models'
    request = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            return response.status == 200 and data.get('object') == 'list'
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def build_plan(live=False, eval_live=False, base_url=DEFAULT_BASE_URL, trace_path=None):
    steps = [
        Step(
            'python-tests',
            (
                'python3',
                '-m',
                'unittest',
                'test_check_agent_compat.py',
                'test_smoke_openai_compat.py',
                'test_eval_agent_workflows.py',
                'test_score_eval.py',
                'test_eval_tool_calling.py',
                'test_prepare_mivi_dataset.py',
            ),
            ROOT / 'scripts',
        ),
        Step('rust-tests', ('cargo', 'test', '--quiet')),
        Step('rustfmt', ('cargo', 'fmt', '--check')),
        Step('release-build', ('cargo', 'build', '--release', '--quiet')),
    ]

    if live:
        steps.append(Step('live-smoke', ('python3', 'scripts/smoke_openai_compat.py', '--base-url', base_url)))
    if eval_live:
        trace_path = trace_path or os.environ.get('MIVI_TRACE_PATH', 'logs/mivi-trace.jsonl')
        env = {'MIVI_TRACE': '1', 'MIVI_TRACE_PATH': trace_path}
        steps.append(
            Step(
                'live-agent-eval',
                (
                    'python3',
                    'scripts/eval_agent_workflows.py',
                    '--url',
                    chat_url_for(base_url),
                    '--trace-path',
                    trace_path,
                    '--kinds',
                    'trace-multi-tool-result',
                ),
                ROOT,
                env,
            )
        )
        steps.append(
            Step(
                'live-tool-calling-eval',
                (
                    'python3',
                    'scripts/eval_tool_calling.py',
                    '--url',
                    chat_url_for(base_url),
                ),
                ROOT,
                env,
            )
        )
    return steps


def run_step(step):
    env = os.environ.copy()
    if step.env:
        env.update(step.env)
    print(f"==> {step.name}: {' '.join(step.cmd)}", flush=True)
    return subprocess.run(step.cmd, cwd=step.cwd, env=env).returncode


def parse_live_mode(value):
    value = value.lower()
    if value not in {'auto', 'on', 'off'}:
        raise argparse.ArgumentTypeError('mode must be auto, on, or off')
    return value


def choose_live_checks(live_mode, live_eval_mode, reachable):
    live = reachable if live_mode == 'auto' else live_mode == 'on'
    if live_eval_mode == 'on':
        return live, live
    if live_eval_mode == 'off':
        return live, False
    return live, False


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run MIVI agent compatibility checks')
    parser.add_argument('--live', type=parse_live_mode, default=os.environ.get('MIVI_CHECK_LIVE', 'auto'))
    parser.add_argument('--live-eval', type=parse_live_mode, default=os.environ.get('MIVI_CHECK_LIVE_EVAL', 'auto'), help='Run trace-backed live evals. Use on only when the server was started with MIVI_TRACE=1.')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--trace-path', default=os.environ.get('MIVI_TRACE_PATH', 'logs/mivi-trace.jsonl'))
    parser.add_argument('--skip-release-build', action='store_true')
    args = parser.parse_args(argv)

    reachable = server_is_reachable(args.base_url)
    live, eval_live = choose_live_checks(args.live, args.live_eval, reachable)
    if args.live == 'on' and not reachable:
        print(f"live server unreachable at {args.base_url}", file=sys.stderr)
        return 2
    if args.live_eval == 'on' and not live:
        print(f"live eval requires a reachable server at {args.base_url}", file=sys.stderr)
        return 2

    steps = build_plan(live=live, eval_live=eval_live, base_url=args.base_url, trace_path=args.trace_path)
    if args.skip_release_build:
        steps = [step for step in steps if step.name != 'release-build']

    failed = []
    for step in steps:
        code = run_step(step)
        if code != 0:
            failed.append((step.name, code))
            break

    if args.live == 'auto' and not live:
        print(f"==> live checks skipped: no server at {args.base_url}")
    if args.live_eval == 'auto' and live and not eval_live:
        print("==> live trace eval skipped: start server with MIVI_TRACE=1 and pass --live-eval on")

    if failed:
        name, code = failed[0]
        print(f"FAILED {name} exit={code}", file=sys.stderr)
        return code or 1

    print('OK agent compatibility checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
