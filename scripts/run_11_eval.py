#!/usr/bin/env python3
"""Start MIVI server with the new model, run 11-workflow eval, and print results."""
import os, sys, time, json, subprocess, urllib.request, signal

MODEL = "models/new/LFM2.5-350M.Q4_K_M.gguf"
PORT = 8146
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/v1/models"
TRACE_PATH = "logs/mivi-trace-eval.jsonl"

env = os.environ.copy()
env["MIVI_RUNTIME_MODE"] = "worker-eco"
env["MIVI_REASONER_MODEL"] = MODEL
env["MIVI_CODER_MODEL"] = MODEL
env["MIVI_TOOL_MODEL"] = MODEL
env["MIVI_TRACE"] = "1"
env["MIVI_TRACE_PATH"] = TRACE_PATH
env["RAYON_NUM_THREADS"] = "2"
env["MIVI_CLI_THREADS"] = "2"

print(f"🚀 Starting MIVI server on port {PORT} with model: {MODEL}...")
server = subprocess.Popen(
    ["target/release/mivi", "serve", "--port", str(PORT)],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    preexec_fn=os.setsid
)

try:
    ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            req = urllib.request.Request(HEALTH)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    if not ready:
        print("❌ Server failed to start")
        sys.exit(1)

    print("✅ Server ready! Running 11-agentic-workflow evaluation...\n")
    
    eval_env = os.environ.copy()
    eval_env["MIVI_EVAL_SERVER_URL"] = URL
    eval_env["MIVI_TRACE_PATH"] = TRACE_PATH
    eval_env["MIVI_EVAL_TIMEOUT"] = "180"

    res = subprocess.run(
        [sys.executable, "scripts/eval_agent_workflows.py"],
        env=eval_env, capture_output=True, text=True
    )
    
    out_file = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    print(f"📄 Eval result file: {out_file}\n")
    
    if os.path.exists(out_file):
        total = 0
        passed = 0
        with open(out_file) as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                total += 1
                ok = d.get("ok", False)
                if ok: passed += 1
                mark = "✅" if ok else "❌"
                kind = d.get("kind", "unknown")
                score = d.get("score", 0.0)
                reasons = d.get("reasons", [])
                elapsed = d.get("elapsed_ms", 0)
                detail = f"score={score:.2f} ({elapsed}ms)" if ok else f"reasons={reasons}"
                print(f"  {mark} {kind:<25} {detail}")
                
        print(f"\n{'='*60}")
        print(f"🏆 11-WORKFLOW BENCHMARK RESULT: {passed}/{total} PASSED ({passed/total*100:.1f}%)")
        print(f"{'='*60}")
    else:
        print("Raw output:", res.stdout)
        print("Raw stderr:", res.stderr)

finally:
    print("\n🧹 Shutting down server...")
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except Exception:
        pass
    server.wait()
