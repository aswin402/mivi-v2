#!/usr/bin/env python3
"""Deep model testing: probes parameter binding, tool selection, coding, identity, and edge cases."""
import os, sys, time, json, subprocess, urllib.request, signal, glob

# Auto-detect target model in models/new/
new_models = glob.glob("models/new/*.gguf")
if not new_models:
    print("❌ No GGUF models found in models/new/")
    sys.exit(1)

# Pick the newest one if multiple
new_models.sort(key=os.path.getmtime, reverse=True)
MODEL = new_models[0]
print(f"🎯 Testing model: {MODEL} (size: {os.path.getsize(MODEL)/1e6:.1f} MB, modified: {time.ctime(os.path.getmtime(MODEL))})")

PORT = 8145
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/v1/models"

def chat(messages, tools=None, model="mivi"):
    body = {"model": model, "stream": False, "messages": messages}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read())
            msg = r["choices"][0]["message"]
            return msg.get("content", ""), msg.get("tool_calls", [])
    except Exception as e:
        return f"ERROR: {e}", []

def tool(name, desc, props, req_fields):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req_fields}}}

def parse_args(tc):
    a = tc.get("function", {}).get("arguments", "{}")
    if isinstance(a, str):
        try: return json.loads(a)
        except: return {"_raw": a}
    return a

results = []

def test(category, name, check_fn, messages, tools=None):
    content, tool_calls = chat(messages, tools)
    ok, detail = check_fn(content, tool_calls)
    status = "✅" if ok else "❌"
    results.append({
        "category": category,
        "name": name,
        "ok": ok,
        "detail": detail,
        "content": content[:300] if content else "",
        "tool_calls": [{"name": tc.get("function",{}).get("name"), "args": parse_args(tc)} for tc in tool_calls]
    })
    print(f"  {status} {name}: {detail}")

# ============================== TOOLS ==============================

BASH_TOOL = tool("bash", "Run a shell command", {"cmd": {"type": "string"}}, ["cmd"])
REMOVE_JOB = tool("remove_job", "Remove a scheduled job", {"id": {"type": "string"}}, ["id"])
SCHEDULE_JOB = tool("schedule_job", "Create a scheduled job", {"prompt": {"type": "string"}}, ["prompt"])
WEBFETCH = tool("webfetch", "Fetch a web page", {"url": {"type": "string", "format": "uri"}}, ["url"])
READ_FILE = tool("read_file", "Read a file", {"path": {"type": "string"}}, ["path"])
WRITE_FILE = tool("write_file", "Write a file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
GET_WEATHER = tool("get_weather", "Get weather for a city", {"city": {"type": "string"}}, ["city"])
SEARCH_WEB = tool("search_web", "Search the web", {"query": {"type": "string"}}, ["query"])
CALCULATOR = tool("calculator", "Evaluate math expression", {"expr": {"type": "string"}}, ["expr"])

def run_all_tests():
    # ===== 1. PARAMETER BINDING =====
    print("\n🔬 1. PARAMETER BINDING TESTS")

    test("param_binding", "bash: npm test", 
        lambda c, tc: (len(tc)==1 and parse_args(tc[0]).get("cmd","")=="npm test", f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Run npm test."}], [BASH_TOOL, READ_FILE])

    test("param_binding", "bash: cargo build --release",
        lambda c, tc: (len(tc)==1 and "cargo build --release" in parse_args(tc[0]).get("cmd",""), f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Run cargo build --release."}], [BASH_TOOL, READ_FILE])

    test("param_binding", "bash: git status",
        lambda c, tc: (len(tc)==1 and "git status" in parse_args(tc[0]).get("cmd",""), f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Run git status."}], [BASH_TOOL])

    test("param_binding", "bash: python3 -m unittest",
        lambda c, tc: (len(tc)==1 and "python3 -m unittest" in parse_args(tc[0]).get("cmd",""), f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Execute python3 -m unittest."}], [BASH_TOOL, READ_FILE])

    test("param_binding", "remove_job: id=1",
        lambda c, tc: (len(tc)==1 and str(parse_args(tc[0]).get("id",""))=="1", f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Stop scheduled job 1."}], [REMOVE_JOB, SCHEDULE_JOB, READ_FILE])

    test("param_binding", "remove_job: id=42",
        lambda c, tc: (len(tc)==1 and str(parse_args(tc[0]).get("id",""))=="42", f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Cancel job 42 immediately."}], [REMOVE_JOB, SCHEDULE_JOB])

    test("param_binding", "remove_job: id=job_99",
        lambda c, tc: (len(tc)==1 and parse_args(tc[0]).get("id","")=="job_99", f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Remove job job_99 from the scheduler."}], [REMOVE_JOB])

    test("param_binding", "webfetch: hono.dev",
        lambda c, tc: (len(tc)==1 and "hono.dev" in parse_args(tc[0]).get("url",""), f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Research https://hono.dev/ and summarize it."}], [WEBFETCH, SEARCH_WEB, READ_FILE])

    test("param_binding", "webfetch: docs.rs/tokio",
        lambda c, tc: (len(tc)==1 and "docs.rs/tokio" in parse_args(tc[0]).get("url",""), f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Fetch the documentation at https://docs.rs/tokio."}], [WEBFETCH, READ_FILE])

    test("param_binding", "get_weather: Paris",
        lambda c, tc: (len(tc)==1 and parse_args(tc[0]).get("city","")=="Paris", f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "What's the weather in Paris?"}], [GET_WEATHER, SEARCH_WEB])

    test("param_binding", "get_weather: Tokyo",
        lambda c, tc: (len(tc)==1 and parse_args(tc[0]).get("city","")=="Tokyo", f"args={parse_args(tc[0]) if tc else 'none'}"),
        [{"role": "user", "content": "Get the weather for Tokyo."}], [GET_WEATHER])

    # ===== 2. TOOL SELECTION =====
    print("\n🔬 2. TOOL SELECTION TESTS")

    test("tool_select", "select bash from 5 tools",
        lambda c, tc: (len(tc)==1 and tc[0].get("function",{}).get("name")=="bash", f"tool={tc[0].get('function',{}).get('name') if tc else 'none'}"),
        [{"role": "user", "content": "Run ls -la."}], [BASH_TOOL, READ_FILE, WRITE_FILE, GET_WEATHER, WEBFETCH])

    test("tool_select", "select webfetch not search_web",
        lambda c, tc: (len(tc)==1 and tc[0].get("function",{}).get("name")=="webfetch", f"tool={tc[0].get('function',{}).get('name') if tc else 'none'}"),
        [{"role": "user", "content": "Fetch the page at https://unsloth.ai/docs."}], [WEBFETCH, SEARCH_WEB, READ_FILE, BASH_TOOL])

    test("tool_select", "select search_web not webfetch",
        lambda c, tc: (len(tc)==1 and tc[0].get("function",{}).get("name")=="search_web", f"tool={tc[0].get('function',{}).get('name') if tc else 'none'}"),
        [{"role": "user", "content": "Search the web for Rust async runtime comparisons."}], [WEBFETCH, SEARCH_WEB, READ_FILE])

    test("tool_select", "select read_file",
        lambda c, tc: (len(tc)==1 and tc[0].get("function",{}).get("name")=="read_file", f"tool={tc[0].get('function',{}).get('name') if tc else 'none'}"),
        [{"role": "user", "content": "Read the file at src/main.rs."}], [BASH_TOOL, READ_FILE, WRITE_FILE, WEBFETCH])

    # ===== 3. NEGATIVE / CHAT TESTS =====
    print("\n🔬 3. NEGATIVE / CHAT TESTS (Should NOT produce tool calls)")

    test("negative", "greeting: hello",
        lambda c, tc: (len(tc)==0 and len(c) > 5, f"tool_calls={len(tc)}, content_len={len(c)}"),
        [{"role": "user", "content": "Hello, how are you today?"}], [BASH_TOOL, READ_FILE, GET_WEATHER])

    test("negative", "identity: what is your name",
        lambda c, tc: (len(tc)==0 and "mivi" in c.lower(), f"tool_calls={len(tc)}, has_mivi={'mivi' in c.lower()}"),
        [{"role": "user", "content": "What is your name?"}], [BASH_TOOL, READ_FILE])

    test("negative", "knowledge: explain TCP vs UDP",
        lambda c, tc: (len(tc)==0 and len(c) > 20, f"tool_calls={len(tc)}, content_len={len(c)}"),
        [{"role": "user", "content": "Explain the difference between TCP and UDP."}], [BASH_TOOL, WEBFETCH])

    test("negative", "thanks response",
        lambda c, tc: (len(tc)==0 and len(c) > 5, f"tool_calls={len(tc)}, content_len={len(c)}"),
        [{"role": "user", "content": "Thanks for your help!"}], [BASH_TOOL, READ_FILE])

    # ===== 4. CODING TESTS =====
    print("\n🔬 4. CODING / VERIFIED OUTPUT TESTS")

    test("coding", "sum 2+3 with verified output",
        lambda c, tc: ("5" in c and "verified terminal output" in c.lower(), f"has_5={'5' in c}, has_verified={'verified terminal output' in c.lower()}"),
        [{"role": "user", "content": "Write Python code that prints the sum of 2 and 3."}])

    test("coding", "reverse string 'hello'",
        lambda c, tc: ("olleh" in c, f"has_olleh={'olleh' in c}"),
        [{"role": "user", "content": "Write Python code to reverse the string 'hello'."}])

    test("coding", "product 7*8",
        lambda c, tc: ("56" in c, f"has_56={'56' in c}"),
        [{"role": "user", "content": "Write Python code that prints the product of 7 and 8."}])

    # ===== 5. IDENTITY TESTS =====
    print("\n🔬 5. IDENTITY TESTS")

    test("identity", "who are you",
        lambda c, tc: ("mivi" in c.lower() and len(tc)==0, f"has_mivi={'mivi' in c.lower()}, tool_calls={len(tc)}"),
        [{"role": "user", "content": "Who are you?"}])

    test("identity", "what model are you",
        lambda c, tc: ("mivi" in c.lower() and "qwen" not in c.lower() and "llama" not in c.lower(), f"has_mivi={'mivi' in c.lower()}, leaked={'qwen' in c.lower() or 'llama' in c.lower()}"),
        [{"role": "user", "content": "What model are you running?"}])

    # ===== 6. EDGE CASES =====
    print("\n🔬 6. EDGE CASE TESTS")

    test("edge", "multi-step: read then run",
        lambda c, tc: (len(tc) >= 1, f"tool_calls={len(tc)}"),
        [{"role": "user", "content": "Read Cargo.toml and then run cargo test."}], [BASH_TOOL, READ_FILE])

    test("edge", "no tools provided, coding question",
        lambda c, tc: (len(tc)==0 and len(c) > 10, f"tool_calls={len(tc)}, content_len={len(c)}"),
        [{"role": "user", "content": "Write a Python hello world script."}])

# ============================== MAIN ==============================

env = os.environ.copy()
env["MIVI_RUNTIME_MODE"] = "worker-eco"
env["MIVI_REASONER_MODEL"] = MODEL
env["MIVI_CODER_MODEL"] = MODEL
env["MIVI_TOOL_MODEL"] = MODEL
env["MIVI_TRACE"] = "0"
env["RAYON_NUM_THREADS"] = "2"
env["MIVI_CLI_THREADS"] = "2"

print(f"🚀 Starting MIVI server with {MODEL} on port {PORT}...")
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

    print("✅ Server ready! Running deep model evaluation...\n")
    run_all_tests()

    # Summary
    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"pass": 0, "fail": 0}
        if r["ok"]:
            cats[cat]["pass"] += 1
        else:
            cats[cat]["fail"] += 1

    total_pass = sum(c["pass"] for c in cats.values())
    total_fail = sum(c["fail"] for c in cats.values())
    total = total_pass + total_fail

    print(f"\n{'='*60}")
    print(f"🏆 DEEP MODEL TEST RESULTS: {total_pass}/{total} PASSED ({total_pass/total*100:.1f}%)")
    print(f"{'='*60}")
    for cat, counts in cats.items():
        t = counts["pass"] + counts["fail"]
        print(f"  {cat:<20} {counts['pass']}/{t} ({counts['pass']/t*100:.0f}%)")
    print(f"{'='*60}")

    # Dump failures for analysis
    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\n🔍 FAILURE DETAILS:")
        for f in failures:
            print(f"  ❌ [{f['category']}] {f['name']}: {f['detail']}")
            if f["tool_calls"]:
                print(f"     tool_calls: {f['tool_calls']}")
            if f["content"]:
                print(f"     content: {repr(f['content'][:150])}")

    # Save full results
    os.makedirs("model-eval-results", exist_ok=True)
    with open("model-eval-results/deep-test-latest.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n📁 Full results saved to model-eval-results/deep-test-latest.json")

finally:
    print("\n🧹 Shutting down server...")
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except Exception:
        pass
    server.wait()
