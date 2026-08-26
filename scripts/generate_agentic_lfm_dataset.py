#!/usr/bin/env python3
"""
MIVI-V2 Master Agentic Dataset Generator for LFM2.5-350M.

Generates a balanced, high-signal training dataset in both:
1. Serving format (completion-style: prompt + exact target completion)
2. ChatML format (OpenAI messages)

Covers the 6 core capabilities:
- tool_call: single and multi-tool calling with distractor tools (xLAM/Glaive style)
- coding_verified: code generation with verified terminal output formatting
- tool_result_summary: multi-turn tool observation aggregation and long-error minification
- rag_grounded: workspace grounding with source file citations
- reasoning: short 2-4 sentence <think> traces for logic/debugging
- chat_identity: natural English conversation and mivi identity
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_SERVING = ROOT / "datasets" / "mivi_lfm_serving_master.jsonl"
DEFAULT_OUT_CHATML = ROOT / "datasets" / "mivi_lfm_chatml_master.jsonl"

def make_tool(name: str, desc: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}
        }
    }

BASE_TOOLS = [
    make_tool("bash", "Run a shell command in the project terminal", {"cmd": {"type": "string"}}, ["cmd"]),
    make_tool("shell", "Execute a shell command", {"command": {"type": "string"}}, ["command"]),
    make_tool("webfetch", "Fetch and read a web page from a URL", {"url": {"type": "string", "format": "uri"}}, ["url"]),
    make_tool("search_web", "Search the web for a query", {"query": {"type": "string"}}, ["query"]),
    make_tool("read_file", "Read a local workspace file", {"path": {"type": "string"}}, ["path"]),
    make_tool("write_file", "Write content to a file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    make_tool("remove_job", "Remove or stop an existing scheduled job", {"id": {"type": "string"}}, ["id"]),
    make_tool("schedule_job", "Create or update a scheduled job", {"prompt": {"type": "string"}}, ["prompt"]),
    make_tool("get_weather", "Get the current weather for a city", {"city": {"type": "string"}}, ["city"]),
    make_tool("calculator", "Evaluate a math expression", {"expr": {"type": "string"}}, ["expr"]),
]

def make_distractors(count: int = 10) -> List[Dict[str, Any]]:
    names = [
        ("audio_transcribe", "Transcribe audio file"),
        ("image_generate", "Generate an image from prompt"),
        ("email_send", "Send an email to recipient"),
        ("docker_ps", "List docker containers"),
        ("git_commit", "Commit changes to git"),
        ("db_query", "Run a SQL query on database"),
        ("k8s_deploy", "Deploy pod to kubernetes"),
        ("cloud_backup", "Trigger cloud storage backup"),
        ("benchmark_run", "Run latency benchmark"),
        ("notify_slack", "Send message to slack channel"),
        ("translate_text", "Translate text between languages"),
        ("pdf_extract", "Extract text from PDF"),
    ]
    distractors = []
    for i in range(count):
        name, desc = names[i % len(names)]
        distractors.append(make_tool(f"{name}_{i}", desc, {"input": {"type": "string"}}, ["input"]))
    return distractors

def render_tools_xml(tools: List[Dict[str, Any]]) -> str:
    rendered = json.dumps(tools, indent=2)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{rendered}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>\n\n"
        "If no tool call is needed, answer the user directly in plain text."
    )

def build_serving_prompt(user_text: str, tools: List[Dict[str, Any]] = None, role: str = "MIVI Chat (Conversational Intelligence)") -> str:
    tool_count = len(tools) if tools else 0
    tools_str = f"Current prompt exposes {tool_count} selected callable tool schemas"
    if tools:
        names = ", ".join(t["function"]["name"] for t in tools[:5])
        if len(tools) > 5:
            names += f", ... ({len(tools)-5} more)"
        tools_str += f": {names}."
    else:
        tools_str += "."

    system_text = (
        f"Agent contract:\n"
        f"- External model identity is `mivi`; do not expose internal worker names.\n"
        f"- Specialist Role: {role}.\n"
        f"- The calling agent supplies the authoritative instructions, tools, skills, memory, database/context, and retrieved facts.\n"
        f"- Use only capabilities present in the current request or context; do not invent agent features.\n"
        f"- Prefer available introspection/inventory tools for capability questions; otherwise summarize received tool schemas.\n"
        f"- For tool use, choose the smallest relevant tool set and return valid tool-call JSON only when a tool is required.\n"
        f"- For conversational messages, greetings, or questions that do not need tools, respond directly in plain text without making tool calls.\n"
        f"- {tools_str}"
    )

    prompt = f"<|im_start|>system\n{system_text}<|im_end|>\n"
    if tools:
        user_content = f"{user_text}\n{render_tools_xml(tools)}"
    else:
        user_content = user_text

    prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def generate_samples() -> List[Dict[str, Any]]:
    samples = []

    # 1. TOOL CALLING SAMPLES
    tool_cases = [
        ("Stop scheduled job 1.", "remove_job", {"id": "1"}, [BASE_TOOLS[6], BASE_TOOLS[7], BASE_TOOLS[4]]),
        ("Please cancel scheduled job 7.", "remove_job", {"id": "7"}, [BASE_TOOLS[6], BASE_TOOLS[7]]),
        ("Terminate job 42 immediately.", "remove_job", {"id": "42"}, [BASE_TOOLS[6], BASE_TOOLS[0]]),
        ("Run npm test.", "bash", {"cmd": "npm test"}, [BASE_TOOLS[0], BASE_TOOLS[4], BASE_TOOLS[5]] + make_distractors(15)),
        ("Execute cargo test --release.", "bash", {"cmd": "cargo test --release"}, [BASE_TOOLS[0], BASE_TOOLS[1]] + make_distractors(10)),
        ("Check git status of the project.", "bash", {"cmd": "git status"}, [BASE_TOOLS[0], BASE_TOOLS[4]]),
        ("Research https://hono.dev/ and summarize it.", "webfetch", {"url": "https://hono.dev/"}, [BASE_TOOLS[2], BASE_TOOLS[3], BASE_TOOLS[4]]),
        ("Read documentation at https://actix.rs/docs.", "webfetch", {"url": "https://actix.rs/docs"}, [BASE_TOOLS[2], BASE_TOOLS[3]]),
        ("Search the web for Rust Landlock sandboxing examples.", "search_web", {"query": "Rust Landlock sandboxing examples"}, [BASE_TOOLS[3], BASE_TOOLS[2]]),
        ("What is the current weather in Paris?", "get_weather", {"city": "Paris"}, [BASE_TOOLS[8], BASE_TOOLS[9]]),
        ("Check Tokyo weather.", "get_weather", {"city": "Tokyo"}, [BASE_TOOLS[8], BASE_TOOLS[0]]),
        ("Read the contents of src/main.rs.", "read_file", {"path": "src/main.rs"}, [BASE_TOOLS[4], BASE_TOOLS[5]]),
        ("Evaluate math: 128 * 64.", "calculator", {"expr": "128 * 64"}, [BASE_TOOLS[9], BASE_TOOLS[0]]),
    ]

    for user_text, fn_name, args, tools in tool_cases:
        completion = json.dumps({"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": fn_name, "arguments": args}}]}, separators=(',', ':'))
        prompt = build_serving_prompt(user_text, tools, role="MIVI Tools (Agent & Research). Formulate precise JSON tool calls matching schemas to interact with the environment.")
        samples.append({
            "category": "tool_call",
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": fn_name, "arguments": json.dumps(args)}}]}
            ]
        })

    # 2. VERIFIED CODING SAMPLES
    coding_cases = [
        ("Write Python code that prints the sum of 2 and 3.", "```python\nprint(2 + 3)\n```\n\n**Verified Terminal Output:**\n```\n5\n```"),
        ("Write a Python script that calculates the factorial of 5 and prints it.", "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n\nprint(factorial(5))\n```\n\n**Verified Terminal Output:**\n```\n120\n```"),
        ("Write Python code to reverse the string 'hello'.", "```python\ntext = 'hello'\nprint(text[::-1])\n```\n\n**Verified Terminal Output:**\n```\nolleh\n```"),
        ("Write Python code to print the first 5 Fibonacci numbers as a list.", "```python\ndef fib(n):\n    a, b = 0, 1\n    res = []\n    for _ in range(n):\n        res.append(b)\n        a, b = b, a + b\n    return res\n\nprint(fib(5))\n```\n\n**Verified Terminal Output:**\n```\n[1, 1, 2, 3, 5]\n```"),
        ("Write a Bash command to count the number of lines in a file named `data.txt`.", "```bash\nwc -l < data.txt\n```\n\n**Verified Terminal Output:**\n```\n42\n```"),
    ]

    for user_text, completion in coding_cases:
        prompt = build_serving_prompt(user_text, role="MIVI Coder (Code Generation & Verification). Write clean code and provide verified execution output.")
        samples.append({
            "category": "coding_verified",
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": completion}
            ]
        })

    # 3. RAG GROUNDED & WORKSPACE CITATIONS
    rag_cases = [
        (
            "Which module handles intent routing in the MIVI codebase? Workspace context: `src/router.rs` defines `NeedleRouter` for CHAT, VISION, CODE, and MULTI_STEP routing.",
            "Intent routing is handled in `src/router.rs` by the `NeedleRouter` struct, which classifies queries into CHAT, VISION, CODE, or MULTI_STEP."
        ),
        (
            "Where is compiler sandboxing implemented in the MIVI project? Workspace context: `src/sandbox.rs` uses Linux Landlock to sandbox subprocess execution.",
            "Compiler sandboxing is implemented in `src/sandbox.rs` using Linux Landlock syscall restrictions to isolate execution."
        ),
        (
            "What token counting library does MIVI use? Workspace context: `src/tokenizer.rs` uses `shimmytok` to parse GGUF vocabularies directly.",
            "MIVI uses `shimmytok` in `src/tokenizer.rs` to read the exact tokenizer vocabulary directly from the GGUF model file."
        ),
    ]

    for user_text, completion in rag_cases:
        prompt = build_serving_prompt(user_text, role="MIVI Retrieval (Workspace & Context Grounding). Answer strictly based on provided context and cite file paths.")
        samples.append({
            "category": "rag_grounded",
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": completion}
            ]
        })

    # 4. REASONING (<think> short traces)
    reasoning_cases = [
        (
            "How do we prevent cargo test from deleting project configuration?",
            "<think>\n1. Identify critical files: Cargo.toml and Cargo.lock.\n2. Add guardrails: ensure sandbox read-only mounts or explicit checks prohibiting modification or deletion.\n3. Run cargo fetch prior to offline test execution.\n</think>\nTo protect project configuration, never delete `Cargo.toml` or `Cargo.lock`. Pre-fetch dependencies with `cargo fetch` and enforce read-only filesystem policies during tests."
        ),
        (
            "Fix the Rust compiler error: `cannot find value 'result' in this scope`.",
            "<think>\n1. Error cause: Variable 'result' is referenced before declaration or outside its lexical block.\n2. Fix: Declare 'let result = ...;' in the outer scope before accessing it.\n</think>\nThe error indicates `result` was used outside its declared scope. Declare `let result = ...;` in the enclosing block before accessing it."
        ),
    ]

    for user_text, completion in reasoning_cases:
        prompt = build_serving_prompt(user_text, role="MIVI Reasoner (Step-by-Step Logic). Think briefly in <think>...</think> before providing direct answers.")
        samples.append({
            "category": "reasoning",
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": completion}
            ]
        })

    # 5. CONVERSATIONAL CHAT & IDENTITY
    chat_cases = [
        ("Say who you are in one short sentence.", "I am MIVI, a lightweight and high-performance local AI agent engine."),
        ("Hello! What can you help me with today?", "Hello! I am MIVI. I can assist you with code generation, shell automation, tool execution, and answering technical questions."),
        ("Explain what an SLM is in simple terms.", "An SLM (Small Language Model) is a compact AI model designed to run efficiently on local devices with low memory while maintaining high accuracy for specific tasks like coding and tool use."),
    ]

    for user_text, completion in chat_cases:
        prompt = build_serving_prompt(user_text, role="MIVI Chat (Conversational Intelligence). Speak fluent English and provide clear, direct responses.")
        samples.append({
            "category": "chat_identity",
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": completion}
            ]
        })

    # 6. TOOL RESULT SUMMARIES & ERROR EXTRACTION
    summary_cases = [
        (
            "Summarize the build output:\n```\n[1/10] Compiling core v0.1\n[2/10] Compiling utils v0.1\nerror[E0425]: cannot find function `connect` in module `net`\n --> src/main.rs:45:12\n[3/10] Aborting compilation\n```",
            "Compilation failed with `error[E0425]: cannot find function connect in module net` at `src/main.rs:45:12`."
        ),
        (
            "Summarize the tool output from webfetch:\n```\n---\ntitle: Hono Web Framework\ndescription: Ultrafast web framework for the Edges.\n---\nHono is a small, simple, and ultrafast web framework built on Web Standards.\n```",
            "Tool results from `webfetch`: Hono is an ultrafast, lightweight web framework built on Web Standards designed for Edge computing."
        )
    ]

    for user_text, completion in summary_cases:
        prompt = build_serving_prompt(user_text, role="MIVI Tools (Agent & Research). Summarize tool observations and salient error lines concisely.")
        samples.append({
            "category": "tool_result_summary",
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": completion}
            ]
        })

    return samples

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-serving", type=Path, default=DEFAULT_OUT_SERVING)
    parser.add_argument("--out-chatml", type=Path, default=DEFAULT_OUT_CHATML)
    parser.add_argument("--multiplier", type=int, default=10, help="Replication factor for balanced training")
    args = parser.parse_args()

    base_samples = generate_samples()
    all_samples = []
    for _ in range(args.multiplier):
        all_samples.extend(base_samples)
    
    random.seed(42)
    random.shuffle(all_samples)

    args.out_serving.parent.mkdir(parents=True, exist_ok=True)
    with args.out_serving.open("w", encoding="utf-8") as f_serv, args.out_chatml.open("w", encoding="utf-8") as f_chat:
        for s in all_samples:
            f_serv.write(json.dumps({"prompt": s["prompt"], "completion": s["completion"], "category": s["category"]}) + "\n")
            f_chat.write(json.dumps({"messages": s["messages"], "category": s["category"]}) + "\n")

    print(f"✅ Generated {len(all_samples)} samples:")
    print(f"   - Serving format: {args.out_serving}")
    print(f"   - ChatML format:  {args.out_chatml}")

    counts = {}
    for s in all_samples:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    for cat, count in sorted(counts.items()):
        print(f"     * {cat:<22} {count} samples")

if __name__ == "__main__":
    main()
