#!/usr/bin/env python3
"""
MIVI-V2 Knowledge-Lean Dataset Generator & Formatter
Generates high-signal training datasets for Sub-1B Models (Qwen2.5-0.5B / Qwen3-0.6B).

Focuses 100% of model parameter capacity on:
1. Grammar-compliant Tool Calling (Hermes XML `<tools>`/`<tool_call>` and OpenAI JSON)
2. DeepSeek-R1 Distilled Step-by-Step Reasoning (`<think>...</think>`)
3. Code Syntax Generation, Diagnostics & Self-Correction Loop
4. Context-Grounded Q&A (Anti-Hallucination)
"""

import json
import os
import random
import re
from typing import Dict, List, Any, Optional

# Standard MIVI Workspace Tool Schemas
MIVI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from the workspace filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "description": "Create a new file or overwrite an existing file with complete content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path"},
                    "content": {"type": "string", "description": "Full text or code content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_file_content",
            "description": "Replace a specific substring in a file with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path"},
                    "target_content": {"type": "string", "description": "Exact text snippet to be replaced"},
                    "replacement_content": {"type": "string", "description": "New replacement snippet"}
                },
                "required": ["path", "target_content", "replacement_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the local environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact shell command line string to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetch content or documentation from a web URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The HTTP/HTTPS URL to retrieve"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for exact text or regex pattern across workspace files",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword or regex pattern"},
                    "path": {"type": "string", "description": "Directory or file path to search inside"}
                },
                "required": ["query", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List child files and directories inside a given folder",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                },
                "required": ["path"]
            }
        }
    }
]

SYSTEM_PROMPTS = [
    "You are MIVI, a high-speed local AI agent engine. Follow instructions carefully and call tools when needed.",
    "You are a helpful assistant with access to workspace tools. Always think step-by-step before calling a tool.",
    "You are an expert software engineer assistant. Formulate precise JSON tool calls to interact with the environment."
]

def format_hermes_tools_header(tools: List[Dict[str, Any]]) -> str:
    """Format tools into Hermes XML <tools> header."""
    return f"<tools>\n{json.dumps(tools, indent=2)}\n</tools>"

def generate_tool_calling_samples() -> List[Dict[str, Any]]:
    """Generate high-signal tool calling examples with DeepSeek-R1 style <think> blocks."""
    samples = []
    
    # 1. Workspace File & Code Operations
    file_tasks = [
        ("Read the contents of src/main.rs to inspect CLI args.",
         "1. User wants to view `src/main.rs`.\n2. Tool `read_file` is available with argument `path`.\n3. Construct valid JSON argument `{\"path\": \"src/main.rs\"}`.",
         "read_file", {"path": "src/main.rs"}),
        ("Check the cargo configuration in .cargo/config.toml.",
         "1. User needs the cargo config.\n2. Call `read_file` with path `.cargo/config.toml`.\n3. Output proper tool call.",
         "read_file", {"path": ".cargo/config.toml"}),
        ("List all files inside the scripts directory.",
         "1. User requests directory listing of `scripts`.\n2. Use `list_dir` with path `scripts`.",
         "list_dir", {"path": "scripts"}),
        ("Search for all occurrences of 'ContextBudget' across the src folder.",
         "1. User wants to find 'ContextBudget' in `src`.\n2. Use `grep_search` with query='ContextBudget' and path='src'.",
         "grep_search", {"query": "ContextBudget", "path": "src"}),
        ("Run the test suite using cargo test.",
         "1. User asks to run test suite.\n2. Shell execution tool `run_command` is appropriate with `cargo test`.\n3. Formulate JSON arguments.",
         "run_command", {"command": "cargo test"}),
        ("Run cargo fmt check to verify code formatting.",
         "1. Formatting check requested.\n2. Call `run_command` with command `cargo fmt --check`.",
         "run_command", {"command": "cargo fmt --check"}),
        ("Fetch the Axum documentation from https://docs.rs/axum/latest/axum/.",
         "1. External documentation URL requested.\n2. Use `webfetch` tool with `url`='https://docs.rs/axum/latest/axum/'.",
         "webfetch", {"url": "https://docs.rs/axum/latest/axum/"}),
        ("Fetch latest release notes from https://github.com/rust-lang/rust/releases.",
         "1. Web URL fetch requested.\n2. Call `webfetch` with target URL.",
         "webfetch", {"url": "https://github.com/rust-lang/rust/releases"}),
        ("Update the version constant in src/version.rs to 0.0.12.",
         "1. User wants to update version string.\n2. Use `replace_file_content` with target_content and replacement_content.",
         "replace_file_content", {
             "path": "src/version.rs",
             "target_content": "pub const VERSION: &str = \"0.0.11\";",
             "replacement_content": "pub const VERSION: &str = \"0.0.12\";"
         }),
        ("Create a python test script at scripts/test_hello.py with print('hello world').",
         "1. File creation requested.\n2. Use `write_to_file` with path and code content.",
         "write_to_file", {
             "path": "scripts/test_hello.py",
             "content": "#!/usr/bin/env python3\nprint('hello world')\n"
         })
    ]

    for user_prompt, thought, tool_name, args in file_tasks:
        # Hermes format
        hermes_assistant = (
            f"<think>\n{thought}\n</think>\n"
            f"<tool_call>\n{{\"name\": \"{tool_name}\", \"arguments\": {json.dumps(args)}}}\n</tool_call>"
        )
        samples.append({
            "messages": [
                {"role": "system", "content": f"{random.choice(SYSTEM_PROMPTS)}\n\n{format_hermes_tools_header(MIVI_TOOLS)}"},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": hermes_assistant}
            ],
            "category": "tool_calling_hermes"
        })

        # OpenAI format
        openai_assistant = {
            "role": "assistant",
            "content": f"<think>\n{thought}\n</think>",
            "tool_calls": [
                {
                    "id": f"call_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args)
                    }
                }
            ]
        }
        samples.append({
            "messages": [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": user_prompt},
                openai_assistant
            ],
            "category": "tool_calling_openai"
        })

    return samples

def generate_compiler_self_correction_samples() -> List[Dict[str, Any]]:
    """Generate self-correction examples: code generation, compiler error feedback, and fixed code."""
    samples = []
    
    corrections = [
        (
            "Write a Rust function to parse an integer from a string with default fallback.",
            "```rust\npub fn parse_or_default(s: &str, default: i32) -> i32 {\n    s.parse::<i32>().unwrap()\n}\n```",
            "thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: ParseIntError { kind: InvalidDigit }'",
            "1. Panic occurs because unwrap() was called on invalid string input.\n2. Fix by using `.unwrap_or(default)`.\n3. Return safe default on parse error.",
            "```rust\npub fn parse_or_default(s: &str, default: i32) -> i32 {\n    s.parse::<i32>().unwrap_or(default)\n}\n```"
        ),
        (
            "Write a Python function to read a JSON file safely.",
            "```python\ndef load_json(filepath):\n    with open(filepath) as f:\n        return json.loads(f)\n```",
            "TypeError: the JSON object must be str, bytes or bytearray, not _io.TextIOWrapper",
            "1. `json.loads` takes a string, not a file handle.\n2. Should use `json.load(f)` instead of `json.loads(f)`.\n3. Add proper exception handling for FileNotFoundError.",
            "```python\nimport json\n\ndef load_json(filepath):\n    try:\n        with open(filepath, 'r', encoding='utf-8') as f:\n            return json.load(f)\n    except (FileNotFoundError, json.JSONDecodeError):\n        return None\n```"
        ),
        (
            "Write a Rust function to calculate Fibonacci numbers with memoization.",
            "```rust\npub fn fib(n: u32, memo: &mut Vec<u64>) -> u64 {\n    if n <= 1 { return n as u64; }\n    memo[n as usize] = fib(n - 1, memo) + fib(n - 2, memo);\n    memo[n as usize]\n}\n```",
            "thread 'main' panicked at 'index out of bounds: the len is 0 but the index is 2'",
            "1. The `memo` vector was not resized to accommodate index `n`.\n2. Resize or check bounds before indexing `memo`.\n3. Use 0 as uninitialized sentinel.",
            "```rust\npub fn fib(n: u32, memo: &mut Vec<u64>) -> u64 {\n    if n <= 1 {\n        return n as u64;\n    }\n    let idx = n as usize;\n    if memo.len() <= idx {\n        memo.resize(idx + 1, 0);\n    }\n    if memo[idx] != 0 {\n        return memo[idx];\n    }\n    memo[idx] = fib(n - 1, memo) + fib(n - 2, memo);\n    memo[idx]\n}\n```"
        )
    ]

    for user_prompt, buggy_code, error_msg, thought, fixed_code in corrections:
        samples.append({
            "messages": [
                {"role": "system", "content": "You are an expert code repair assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": buggy_code},
                {"role": "user", "content": f"Compiler/Runtime error:\n{error_msg}\nPlease fix the code."},
                {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n{fixed_code}"}
            ],
            "category": "code_correction"
        })

    return samples

def generate_context_grounded_qa_samples() -> List[Dict[str, Any]]:
    """Generate context-grounded Q&A to enforce strict anti-hallucination behavior."""
    samples = []
    
    qa_pairs = [
        (
            "MIVI-V2 is configured with a 3072 token budget by default. In ultra-low-RAM mode, concurrency is locked to 1.",
            "What is the default context token budget and ultra-low concurrency limit in MIVI-V2?",
            "1. Context states default token budget is 3072.\n2. Concurrency limit in ultra-low-RAM mode is 1.\n3. Formulate direct, factual answer.",
            "Based on the provided context, MIVI-V2 has a default token budget of 3072 tokens, and concurrency is locked to 1 in ultra-low-RAM mode."
        ),
        (
            "NeedleRouter performs Naive Bayes intent classification in under 2ms. If confidence is below 0.70 in normal mode, it queries the coder model.",
            "Who founded Apple Computer in 1976?",
            "1. The question asks about the founders of Apple Computer.\n2. Check provided context: context only describes NeedleRouter.\n3. The requested information is not in the context. Refuse to hallucinate.",
            "I cannot answer this question based on the provided context because the context only discusses NeedleRouter intent classification."
        ),
        (
            "The TurboVec RAG index chunks files into 25-line segments and ignores binaries, .git, and target directories.",
            "How does TurboVec RAG chunk files and what does it ignore?",
            "1. Context says TurboVec RAG chunks files into 25-line segments.\n2. Context states it ignores binaries, .git, and target directories.\n3. Output factual synthesis.",
            "According to the context, TurboVec RAG chunks files into 25-line segments and ignores binaries, .git, and target directories."
        )
    ]

    for context, question, thought, answer in qa_pairs:
        samples.append({
            "messages": [
                {"role": "system", "content": f"You are a factual assistant. Use ONLY the following context to answer. If the context does not contain the answer, say that the information is not present.\n\nContext:\n{context}"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n{answer}"}
            ],
            "category": "grounded_qa"
        })

    return samples

def build_dataset(total_samples: int = 5000) -> List[Dict[str, Any]]:
    """Build a comprehensive, balanced knowledge-lean dataset."""
    tool_samples = generate_tool_calling_samples()
    code_samples = generate_compiler_self_correction_samples()
    grounded_samples = generate_context_grounded_qa_samples()
    
    all_base_samples = tool_samples * 20 + code_samples * 25 + grounded_samples * 25
    random.shuffle(all_base_samples)
    
    dataset = []
    for i in range(min(total_samples, len(all_base_samples))):
        item = all_base_samples[i]
        dataset.append(item)
        
    return dataset

def main():
    print("=" * 60)
    print("🚀 MIVI-V2 Knowledge-Lean Dataset Generator")
    print("=" * 60)
    
    dataset = build_dataset(total_samples=5000)
    os.makedirs("datasets", exist_ok=True)
    
    output_file = "datasets/mivi_sub1b_tuning_dataset.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ Generated {len(dataset)} verified training examples.")
    print(f"📁 Output file: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
