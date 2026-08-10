# MIVI-V2 Research 2 — Implementation Blueprint

> **Purpose:** Detailed actionable plan synthesizing ALL research into concrete implementation tasks  
> **Date:** August 10, 2026  
> **Status:** Ready to execute  

---

## 1. The Big Picture — What We're Building

### Current State
```
MIVI v0.0.10 — working but limited:
├── Qwen3 0.6B Q4_K_M (462 MB) — 67% tool calling accuracy
├── Keyword-based RAG (TurboVecRAG) — misses semantic meaning
├── CheapTokenCounter — estimates tokens, often wrong
├── Heuristic NeedleRouter — double-classifies on low confidence
├── No grammar constraints — model can output anything, including garbage
└── Think blocks leak in streams — agents see raw <think> tags
```

### Target State
```
MIVI v1.0.0 — production-ready agent backend under 1 GB:
├── Qwen3 1.7B Q2_K fine-tuned (580 MB) — 85%+ tool calling
├── Grammar-constrained decoding — 95%+ valid JSON tool calls
├── MiniLM semantic RAG (90 MB) — 3-5x better code retrieval
├── Cactus Needle native router (14 MB) — <2ms intent classification
├── rustbpe exact tokenizer (2 MB) — perfect context budget
├── Clean streaming — zero think-block leaks
└── Total RAM: ~756 MB ✅ under 1 GB
```

### What Changes
| Component | Before | After | Why |
|---|---|---|---|
| Primary model | 0.6B Q4 (462 MB) | 1.7B Q2_K FT (580 MB) | 2.8x smarter, +118 MB |
| Tool calling | Hope-and-parse (67%) | Grammar-forced (95%+) | Eliminates hallucination |
| RAG | Keyword matching | Semantic embeddings | Finds related code, not just exact words |
| Token counting | Estimation (~15% error) | Exact BPE counting | Perfect context utilization |
| Routing | Heuristic + double-classify | Needle 26M native | <2ms, single-pass |
| Streaming | Think blocks leak | Clean stripping | Agent compatibility |

---

## 2. Implementation Phases — Detailed

### Phase 9: Critical Fixes & Model Upgrade 🔴

> **Goal:** Fix the 3 critical bugs and upgrade the primary model  
> **Duration:** 3-5 days  
> **Impact:** Tool calling 67% → 85%, clean streaming, -2s latency

---

#### Task 9.1: Fix Think Block Leak in Streaming

**Problem:** `<think>` tags appear in SSE stream when they're the first token in a chunk.

**Files to edit:**
- [`src/model_process.rs`](file:///home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi-v2/src/model_process.rs) — `strip_thinking_from_stream_line()` at line 43

**Current behavior:**
```
Stream chunk 1: "<think>"       → should be suppressed, but gets through
Stream chunk 2: "reasoning..."  → suppressed (skipping=true)
Stream chunk 3: "</think>Answer" → "Answer" output correctly
```

**Fix:**
```rust
pub(crate) fn strip_thinking_from_stream_line(
    line: &str,
    skipping: &mut bool,
) -> Option<String> {
    // FIX: Handle case where line starts with <think> as first token
    let trimmed = line.trim();
    
    // If we encounter <think> at any position (including start), begin skipping
    if trimmed.starts_with("<think>") || trimmed.contains("<think>") {
        *skipping = true;
        // Check if there's content AFTER </think> on the same line
        if let Some(pos) = trimmed.find("</think>") {
            *skipping = false;
            let after = &trimmed[pos + 8..];
            if after.trim().is_empty() {
                return None;
            }
            return Some(after.to_string());
        }
        return None;
    }
    
    if *skipping {
        if let Some(pos) = trimmed.find("</think>") {
            *skipping = false;
            let after = &trimmed[pos + 8..];
            if after.trim().is_empty() {
                return None;
            }
            return Some(after.to_string());
        }
        return None;
    }
    
    Some(line.to_string())
}
```

**Tests to add:**
```rust
#[test]
fn strip_think_at_start_of_stream() {
    let mut skipping = false;
    // First chunk is "<think>" — must be suppressed
    assert_eq!(
        strip_thinking_from_stream_line("<think>", &mut skipping),
        None
    );
    assert!(skipping);
}

#[test]
fn strip_think_inline_with_content() {
    let mut skipping = false;
    assert_eq!(
        strip_thinking_from_stream_line("<think>reasoning</think>Answer", &mut skipping),
        Some("Answer".to_string())
    );
    assert!(!skipping);
}
```

**Effort:** 1 hour  
**Verify:** `cargo test`, then manually test streaming with `curl`

---

#### Task 9.2: Download & Benchmark Qwen3 1.7B Q2_K

**Steps:**
```bash
# 1. Download the model (~580 MB)
huggingface-cli download Qwen/Qwen3-1.7B-GGUF \
  qwen3-1.7b-q2_k.gguf \
  --local-dir models/

# 2. Verify file
ls -lh models/qwen3-1.7b-q2_k.gguf

# 3. Quick test with llama-cli (if available)
./bin/llama-cli -m models/qwen3-1.7b-q2_k.gguf \
  -p "What is 2+2?" -n 50

# 4. Benchmark against current 0.6B
# Test 1: Simple chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mivi","messages":[{"role":"user","content":"What is 2+2?"}]}'

# Test 2: Tool calling
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mivi","messages":[{"role":"user","content":"What is the weather in Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'

# Test 3: Code generation
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mivi","messages":[{"role":"user","content":"Write a Python fibonacci function with memoization"}]}'
```

**Config change to `configs/models.json`:**
```json
{
  "external_model": "mivi",
  "models": [
    {
      "id": "qwen3-17b-reasoner",
      "role": "reasoner",
      "backend": "llama-cli",
      "path": "models/qwen3-1.7b-q2_k.gguf",
      "context_tokens": 4096,
      "ram_mb_estimate": 700,
      "enabled": true,
      "notes": "Upgraded reasoner. 2.8x more parameters than 0.6B. Q2_K fits under 1GB RAM."
    },
    {
      "id": "qwen3-06b-coder",
      "role": "coder",
      "backend": "llama-cli",
      "path": "models/qwen3-0.6b-q4_k_m.gguf",
      "context_tokens": 4096,
      "ram_mb_estimate": 540,
      "enabled": true,
      "notes": "Fast coder for simple tasks. Swapped in when NeedleRouter classifies as simple CODE."
    },
    {
      "id": "minicpm-vision",
      "role": "vision",
      "backend": "vision-cli",
      "path": "models/MiniCPM-V-4.6-Q4_K_M.gguf",
      "context_tokens": 4096,
      "ram_mb_estimate": 900,
      "enabled": false,
      "notes": "Vision worker. Disabled by default."
    }
  ]
}
```

**Effort:** 30 minutes (download) + 1 hour (benchmarking)  
**Verify:** Compare latency, quality, RAM usage vs current 0.6B

---

#### Task 9.3: Eliminate Double Intent Classification

**Problem:** `NeedleRouter` calls the model twice when confidence < 0.84, adding ~2s latency.

**Files to edit:**
- [`src/router.rs`](file:///home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi-v2/src/router.rs)

**Fix approach:**
1. Lower the confidence threshold from 0.84 to 0.65
2. If still low confidence, default to CHAT (the safest fallback) instead of re-classifying
3. Log the low-confidence case for monitoring

```rust
// In NeedleRouter::classify()
// Before: if confidence < 0.84 { re-classify with different prompt }
// After:  if confidence < 0.65 { default to CHAT, log warning }

if confidence < 0.65 {
    tracing::warn!(
        confidence = %confidence,
        prompt_preview = %&prompt[..prompt.len().min(100)],
        "Low confidence classification, defaulting to CHAT"
    );
    return Ok(Intent::Chat);
}
```

**Effort:** 30 minutes  
**Impact:** -2s latency on simple queries (5.6s → 3.6s)

---

#### Task 9.4: Fix 45 Compiler Warnings

**Steps:**
```bash
# List all warnings
cargo build --release 2>&1 | grep "warning:"

# Fix unused imports in handlers.rs
# Fix unused variables in helpers.rs
# Run cargo fmt
cargo fmt

# Verify
cargo build --release 2>&1 | grep -c "warning:"
# Should output: 0
```

**Effort:** 30 minutes

---

### Phase 10: Grammar-Constrained Tool Calling 🔴

> **Goal:** Force the model to output valid JSON when tools are provided  
> **Duration:** 2-3 days  
> **Impact:** Tool calling accuracy 67% → 90%+ regardless of model

---

#### Task 10.1: Implement GBNF Grammar for Tool Calls

**What is GBNF?**
GBNF (GGML BNF) is a grammar format that llama.cpp supports. It constrains the model's output to match a specific pattern. The model physically cannot generate invalid tokens.

**Grammar for OpenAI-style tool calls:**
```gbnf
# File: configs/tool_call.gbnf
# Forces model output to be a valid tool call JSON

root        ::= tool-call
tool-call   ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws object ws "}"
object      ::= "{" ws "}" | "{" ws pair (ws "," ws pair)* ws "}"
pair        ::= string ws ":" ws value
value       ::= string | number | "true" | "false" | "null" | object | array
array       ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
string      ::= "\"" chars "\""
chars       ::= "" | char chars
char        ::= [^"\\] | "\\" escape
escape      ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t"
number      ::= "-"? [0-9]+ ("." [0-9]+)?
ws          ::= [ \t\n]*
```

**Dynamic grammar generation from tool schemas:**

When the client sends tools in the request, MIVI should generate a grammar that matches ONLY those specific tools:

```rust
// src/server/grammar.rs (NEW FILE)

/// Generate a GBNF grammar from the tool definitions in the request
pub fn generate_tool_grammar(tools: &[Tool]) -> String {
    let mut grammar = String::new();
    
    // Root: must be one of the defined tool calls
    grammar.push_str("root ::= ");
    let tool_names: Vec<String> = tools.iter()
        .map(|t| format!("tool-{}", t.function.name.replace("-", "_")))
        .collect();
    grammar.push_str(&tool_names.join(" | "));
    grammar.push('\n');
    
    // Each tool: specific name + its argument schema
    for tool in tools {
        let safe_name = tool.function.name.replace("-", "_");
        grammar.push_str(&format!(
            "tool-{name} ::= \"{{\" ws \"\\\"name\\\"\" ws \":\" ws \"\\\"{}\\\"\" ws \",\" ws \"\\\"arguments\\\"\" ws \":\" ws {name}-args ws \"}}\"\n",
            tool.function.name,
            name = safe_name
        ));
        
        // Generate argument schema grammar from the tool's parameters
        if let Some(params) = &tool.function.parameters {
            grammar.push_str(&generate_params_grammar(&safe_name, params));
        }
    }
    
    // Common rules
    grammar.push_str(COMMON_GRAMMAR_RULES);
    grammar
}

const COMMON_GRAMMAR_RULES: &str = r#"
string ::= "\"" chars "\""
chars ::= "" | char chars
char ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t"
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \t\n]*
"#;
```

**Files to create/edit:**
- `src/server/grammar.rs` — NEW: grammar generation logic
- `src/server/mod.rs` — add `mod grammar;`
- `src/server/helpers.rs` — modify `generate_tool_calls()` at line 1806 to pass grammar to model
- `src/worker.rs` — add `grammar` field to worker request body
- `src/brain.rs` — add `--grammar-file` flag to llama-cli invocation

**Worker request body change:**
```rust
// In worker.rs query_chat_full(), add grammar to the JSON body:
if let Some(grammar) = &request.grammar {
    body["grammar"] = serde_json::Value::String(grammar.clone());
}
```

**Brain CLI change:**
```rust
// In brain.rs run_cli(), add grammar flag:
if let Some(grammar_path) = &opts.grammar_path {
    cmd.arg("--grammar-file").arg(grammar_path);
}
```

**Effort:** 2-3 days  
**Impact:** THE single highest-impact change. Tool calling goes from "hoping" to "guaranteed valid JSON."

---

#### Task 10.2: Grammar Integration in Native Candle Path

For the `NativeBrain` (Candle), implement logit masking:

```rust
// src/native_brain.rs — add grammar-aware sampling

use crate::server::grammar::GrammarState;

/// During token sampling, mask logits that would violate the grammar
fn sample_with_grammar(
    logits: &Tensor,
    grammar_state: &mut GrammarState,
    temperature: f64,
) -> Result<u32> {
    // 1. Get valid next tokens from grammar state
    let valid_tokens = grammar_state.valid_next_tokens();
    
    // 2. Mask invalid tokens to -infinity
    let masked_logits = mask_invalid_tokens(logits, &valid_tokens)?;
    
    // 3. Apply temperature and sample
    let token = sample_top_p(&masked_logits, temperature, 0.9)?;
    
    // 4. Advance grammar state
    grammar_state.advance(token);
    
    Ok(token)
}
```

**Effort:** 1 week (more complex than CLI path)  
**Note:** Can be deferred — CLI/worker path covers most use cases

---

### Phase 11: Fine-Tuning Pipeline 🟠

> **Goal:** Fine-tune Qwen3 1.7B specifically for MIVI's tool calling needs  
> **Duration:** 1-2 days  
> **Impact:** Tool calling 75% → 85%

---

#### Task 11.1: Create Data Preparation Script

```python
# scripts/prepare_finetune_data.py
"""
Prepare training data for MIVI tool-calling fine-tuning.
Combines public datasets with MIVI-specific examples.
"""
import json
from datasets import load_dataset

def main():
    all_examples = []
    
    # ─── Source 1: Glaive Function Calling v2 ───
    print("Loading Glaive Function Calling v2...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    for row in ds:
        parsed = parse_glaive_to_chatml(row)
        if parsed and validate_tool_call_json(parsed):
            all_examples.append(parsed)
    print(f"  → {len(all_examples)} valid examples from Glaive")
    
    # ─── Source 2: xLAM Function Calling ───
    print("Loading xLAM Function Calling...")
    ds2 = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    count_before = len(all_examples)
    for row in ds2:
        parsed = parse_xlam_to_chatml(row)
        if parsed and validate_tool_call_json(parsed):
            all_examples.append(parsed)
    print(f"  → {len(all_examples) - count_before} valid examples from xLAM")
    
    # ─── Source 3: MIVI-specific examples ───
    print("Adding MIVI-specific examples...")
    mivi_examples = generate_mivi_specific_examples()
    all_examples.extend(mivi_examples)
    print(f"  → {len(mivi_examples)} MIVI-specific examples")
    
    # ─── Filter & Deduplicate ───
    all_examples = deduplicate(all_examples)
    print(f"\nTotal: {len(all_examples)} examples")
    
    # ─── Split train/eval ───
    eval_size = min(500, len(all_examples) // 10)
    eval_set = all_examples[:eval_size]
    train_set = all_examples[eval_size:]
    
    # ─── Save ───
    save_jsonl(train_set, "data/tool_calling_train.jsonl")
    save_jsonl(eval_set, "data/tool_calling_eval.jsonl")
    print(f"Saved: {len(train_set)} train, {len(eval_set)} eval")

def generate_mivi_specific_examples():
    """
    Generate training examples using the ACTUAL tool schemas
    that OpenCode, Continue.dev, and Cursor send to MIVI.
    """
    examples = []
    
    # OpenCode agent tools
    opencode_tools = [
        {"name": "read_file", "description": "Read file contents", 
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "write_file", "description": "Write to file",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "bash", "description": "Execute bash command",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "grep", "description": "Search for pattern in files",
         "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}},
    ]
    
    # Generate diverse examples for each tool
    prompts_and_calls = [
        ("Read the contents of main.rs", "read_file", '{"path": "src/main.rs"}'),
        ("Show me the Cargo.toml file", "read_file", '{"path": "Cargo.toml"}'),
        ("Write a hello world program to test.py", "write_file", 
         '{"path": "test.py", "content": "print(\\"Hello, World!\\")"}'),
        ("Run the test suite", "bash", '{"command": "cargo test"}'),
        ("Build the project in release mode", "bash", '{"command": "cargo build --release"}'),
        ("Find all TODO comments in the codebase", "grep", '{"pattern": "TODO", "path": "src/"}'),
        ("Search for the function handle_request", "grep", '{"pattern": "fn handle_request", "path": "src/"}'),
        # ... generate 500-2000 diverse examples
    ]
    
    for prompt, tool_name, args in prompts_and_calls:
        examples.append({
            "messages": [
                {"role": "system", "content": f"You have access to tools: {json.dumps(opencode_tools)}"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{hash(prompt) % 10000}", "type": "function",
                     "function": {"name": tool_name, "arguments": args}}
                ]}
            ]
        })
    
    return examples

def validate_tool_call_json(example):
    """Ensure all tool call arguments are valid JSON."""
    try:
        for msg in example.get("messages", []):
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    json.loads(tc["function"]["arguments"])
        return True
    except (json.JSONDecodeError, KeyError):
        return False

if __name__ == "__main__":
    main()
```

**Effort:** 2-3 hours

---

#### Task 11.2: Google Colab Training Notebook

```python
# MIVI_FineTune_ToolCalling.ipynb (Run on Google Colab Free)

# ─── Cell 1: Install ───
!pip install -q --upgrade unsloth unsloth_zoo

# ─── Cell 2: Load Model ───
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-1.7B",
    max_seq_length = 4096,
    load_in_4bit = True,
)

# ─── Cell 3: Configure LoRA ───
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                        # Rank — 32 is sweet spot for tool calling
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,
    lora_dropout = 0.05,
    use_gradient_checkpointing = "unsloth",
)

print(f"Trainable params: {model.print_trainable_parameters()}")

# ─── Cell 4: Load Dataset ───
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "tool_calling_train.jsonl",
    "eval": "tool_calling_eval.jsonl",
})

# ─── Cell 5: Format for Chat Template ───
def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# ─── Cell 6: Train ───
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["eval"],
    args = TrainingArguments(
        output_dir = "mivi-qwen3-17b-toolcall",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        warmup_ratio = 0.1,
        logging_steps = 10,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        fp16 = True,
        load_best_model_at_end = True,
    ),
    max_seq_length = 4096,
)

trainer.train()

# ─── Cell 7: Export to GGUF Q2_K ───
model.save_pretrained_gguf(
    "mivi-qwen3-17b-agent",
    tokenizer,
    quantization_method = "q2_k"  # Matches our <1GB target
)

# ─── Cell 8: Also export Q4_K_M for comparison ───
model.save_pretrained_gguf(
    "mivi-qwen3-17b-agent-q4",
    tokenizer,
    quantization_method = "q4_k_m"  # For users with >1GB RAM
)

# ─── Cell 9: Download ───
from google.colab import files
files.download("mivi-qwen3-17b-agent/mivi-qwen3-17b-agent-Q2_K.gguf")
```

**Training time:** ~30 minutes on T4  
**Cost:** $0 (Google Colab free tier)

---

#### Task 11.3: Apply ThinkingCap Early-Exit Concept

When creating training data, add examples that train the model to stop thinking early:

```json
{
  "messages": [
    {"role": "user", "content": "Read the file main.rs"},
    {"role": "assistant", "content": "<think>User wants to read a file. Tool: read_file.</think>",
     "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\": \"main.rs\"}"}}]}
  ]
}
```

Notice the thinking block is **very short** — just 1 sentence. This trains the model to:
1. Think briefly (identify the tool)
2. Stop thinking immediately
3. Output the tool call

**Result:** ~50% fewer reasoning tokens = ~50% faster inference

---

### Phase 12: rustbpe Integration 🟠

> **Goal:** Replace CheapTokenCounter with exact BPE tokenization  
> **Duration:** 1-2 days  
> **Impact:** Perfect context budget management

---

#### Task 12.1: Add rustbpe Dependency

```toml
# Cargo.toml
[dependencies]
# ... existing deps ...
bpe = { version = "0.1", optional = true }  # karpathy/rustbpe or similar

[features]
native = ["candle-core", "candle-transformers", "candle-nn", "tokenizers"]
exact-tokenizer = ["bpe"]  # Feature flag for exact tokenization
```

#### Task 12.2: Implement Exact Token Counter

```rust
// src/tokenizer.rs (NEW FILE)

use std::path::Path;

/// Exact BPE token counter using rustbpe
pub struct ExactTokenCounter {
    // BPE vocabulary loaded from the model's tokenizer
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    vocab: std::collections::HashMap<Vec<u8>, u32>,
}

impl ExactTokenCounter {
    pub fn from_tokenizer_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Load the tokenizer.json that comes with the GGUF model
        let data = std::fs::read_to_string(path)?;
        let tokenizer: serde_json::Value = serde_json::from_str(&data)?;
        
        // Parse merges and vocabulary
        // ... implementation using rustbpe internals
        
        Ok(Self { merges, vocab })
    }
    
    pub fn count_tokens(&self, text: &str) -> usize {
        // Apply BPE encoding and count resulting tokens
        let tokens = self.encode(text);
        tokens.len()
    }
    
    fn encode(&self, text: &str) -> Vec<u32> {
        // BPE encoding algorithm
        // ... implementation
        vec![]
    }
}

// Implement the existing TokenCounter trait
impl crate::server::types::TokenCounter for ExactTokenCounter {
    fn count(&self, text: &str) -> usize {
        self.count_tokens(text)
    }
}
```

#### Task 12.3: Wire Into Server

```rust
// In src/server/mod.rs — AppState initialization

let token_counter: Box<dyn TokenCounter> = if cfg!(feature = "exact-tokenizer") {
    // Use exact BPE tokenizer from model's tokenizer.json
    let tokenizer_path = model_config.tokenizer_path
        .as_ref()
        .map(Path::new)
        .unwrap_or(Path::new("models/qwen2.5-0.5b-tokenizer.json"));
    
    match ExactTokenCounter::from_tokenizer_json(tokenizer_path) {
        Ok(counter) => {
            tracing::info!("Using exact BPE tokenizer from {}", tokenizer_path.display());
            Box::new(counter)
        }
        Err(e) => {
            tracing::warn!("Failed to load exact tokenizer: {}, falling back to estimator", e);
            Box::new(CheapTokenCounter)
        }
    }
} else {
    Box::new(CheapTokenCounter)
};
```

**Effort:** 1-2 days  
**Verify:** Compare token counts against `llama-tokenize` output

---

### Phase 13: Semantic RAG with MiniLM 🟡

> **Goal:** Replace keyword-based RAG with semantic embedding search  
> **Duration:** 1 week  
> **Impact:** 3-5x better code retrieval quality

---

#### Task 13.1: Embed MiniLM in Candle

MiniLM-L6-v2 is a 6-layer transformer encoder. Can be loaded via Candle:

```rust
// src/embeddings.rs (NEW FILE)

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

pub struct SemanticEmbedder {
    model: BertModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl SemanticEmbedder {
    /// Load MiniLM-L6-v2 from ONNX or safetensors
    pub fn load(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let tokenizer = tokenizers::Tokenizer::from_file(
            format!("{}/tokenizer.json", model_path)
        )?;
        
        let config = Config::from_file(format!("{}/config.json", model_path))?;
        let vb = VarBuilder::from_safetensors(
            format!("{}/model.safetensors", model_path),
            candle_core::DType::F32,
            &device,
        )?;
        
        let model = BertModel::load(vb, &config)?;
        Ok(Self { model, tokenizer, device })
    }
    
    /// Encode text into 384-dimensional embedding vector
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?
            .unsqueeze(0)?;
        let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)?
            .unsqueeze(0)?;
        
        let output = self.model.forward(&input_ids, &attention_mask)?;
        
        // Mean pooling over token embeddings
        let mean_pooled = (output.sum(1)? / output.dim(1)? as f64)?;
        
        Ok(mean_pooled.squeeze(0)?.to_vec1::<f32>()?)
    }
    
    /// Compute cosine similarity between two embeddings
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-8)
    }
}
```

#### Task 13.2: Upgrade TurboVecRAG

```rust
// In src/rag.rs — add semantic mode

impl TurboVecRAG {
    /// Semantic search using MiniLM embeddings
    pub fn semantic_search(
        &self,
        query: &str,
        embedder: &SemanticEmbedder,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let query_embedding = embedder.embed(query).unwrap();
        
        let mut scored: Vec<(String, f32)> = self.chunks.iter()
            .map(|chunk| {
                let chunk_embedding = embedder.embed(&chunk.content).unwrap();
                let score = SemanticEmbedder::similarity(&query_embedding, &chunk_embedding);
                (chunk.content.clone(), score)
            })
            .collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        scored
    }
}
```

**Note:** For performance, pre-compute and cache embeddings for all indexed chunks. Only compute query embedding at request time.

**Effort:** 1 week  
**RAM cost:** ~90 MB for MiniLM weights + ~50 MB for cached embeddings

---

### Phase 14: Model Tier System 🟡

> **Goal:** Let users choose model size based on their hardware  
> **Duration:** 2-3 hours  
> **Impact:** Makes MIVI usable on everything from Raspberry Pi to workstations

---

#### Task 14.1: Add MIVI_MODEL_TIER Environment Variable

```rust
// src/runtime.rs — add model tier

pub enum ModelTier {
    Tiny,      // 0.6B Q4_K_M — 462 MB — edge/mobile/RPi
    Standard,  // 1.7B Q2_K   — 580 MB — laptops (DEFAULT)
    Pro,       // 3B Q4_K_M   — 1.8 GB — desktop/server
}

impl ModelTier {
    pub fn from_env() -> Self {
        match std::env::var("MIVI_MODEL_TIER").as_deref() {
            Ok("tiny") => ModelTier::Tiny,
            Ok("pro") => ModelTier::Pro,
            _ => ModelTier::Standard,
        }
    }
    
    pub fn reasoner_path(&self) -> &str {
        match self {
            ModelTier::Tiny => "models/qwen3-0.6b-q4_k_m.gguf",
            ModelTier::Standard => "models/qwen3-1.7b-q2_k.gguf",
            ModelTier::Pro => "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        }
    }
    
    pub fn ram_budget_mb(&self) -> usize {
        match self {
            ModelTier::Tiny => 600,
            ModelTier::Standard => 900,
            ModelTier::Pro => 2500,
        }
    }
}
```

---

## 3. New Ideas — Detailed Concepts

### 3.1 Recursive Context Navigation (from RLM)

Instead of loading the entire codebase into 3072 tokens, let the model navigate it:

```
User: "Fix the authentication bug in the server"

Step 1 — Model thinks: "I need to find auth-related code"
Step 1 — Model calls: read_file("src/server/handlers.rs")
Step 1 — System injects: [file contents, 200 lines]

Step 2 — Model thinks: "The auth middleware is imported from auth.rs"
Step 2 — Model calls: read_file("src/auth.rs")
Step 2 — System injects: [file contents, 150 lines]

Step 3 — Model thinks: "Found the bug — missing token expiry check"
Step 3 — Model responds: "The issue is in verify_token() on line 45..."
```

**Implementation:** Add internal "navigation tools" to the orchestrator that the model can call during reasoning, before generating the final response.

### 3.2 MIVI-Agent Model Family (Publishing on HuggingFace)

Create and publish custom fine-tuned models:

```
aswin402/mivi-agent-0.6b-v1 — Tool-calling optimized 0.6B
aswin402/mivi-agent-1.7b-v1 — Tool-calling optimized 1.7B (flagship)
```

Include:
- GGUF files (Q2_K, Q4_K_M, Q8_0)
- Training data recipe
- Benchmark results
- Usage instructions for MIVI

### 3.3 MiniCPM5-1B Testing

The Claude-mimicking 1B model is a strong contender. Benchmark plan:

```bash
# Download MiniCPM5-1B GGUF
# Compare against Qwen3 1.7B Q2_K on:
# 1. Tool calling accuracy (BFCL subset)
# 2. Code generation quality (HumanEval subset)
# 3. Reasoning quality (GSM8K subset)
# 4. RAM usage
# 5. Inference speed (tok/s)
```

If MiniCPM5-1B at Q3_K_M (~500 MB) beats Qwen3 1.7B Q2_K (~580 MB) on tool calling, it becomes our new default.

### 3.4 Hybrid Router: Needle + Supra2-100M

```
Layer 1: Needle (14 MB, <2ms)
  → Classifies: CHAT | CODE | TOOL_CALL | COMPLEX

Layer 2 (only for COMPLEX): Supra2-100M (~80 MB, ~50ms)  
  → Decides: Which specific tool? What arguments? Should we decompose?

Layer 3: Main model (580 MB, ~3s)
  → Executes the actual task with grammar-constrained output
```

Total routing overhead: <55ms for simple tasks, <100ms for complex.

---

## 4. Priority Matrix — Everything We Need To Do

### 🔴 CRITICAL (Do This Week)

| # | Task | Effort | Impact | Files |
|---|---|---|---|---|
| 1 | Fix think block leak in streaming | 1 hr | Agent compat | `model_process.rs` |
| 2 | Download Qwen3 1.7B Q2_K | 30 min | Quality jump | `models/`, `configs/models.json` |
| 3 | Benchmark 1.7B vs 0.6B | 1 hr | Data for decisions | scripts/ |
| 4 | Implement GBNF grammar for tool calls | 2-3 days | 67% → 90%+ tool accuracy | NEW `grammar.rs`, `helpers.rs`, `worker.rs`, `brain.rs` |
| 5 | Fix compiler warnings | 30 min | CI cleanliness | `handlers.rs`, `helpers.rs` |

### 🟠 HIGH (Do This Month)

| # | Task | Effort | Impact | Files |
|---|---|---|---|---|
| 6 | Prepare fine-tuning dataset | 3 hrs | Enables fine-tuning | `scripts/prepare_finetune_data.py` |
| 7 | Fine-tune 1.7B on Colab | 30 min | 85%+ tool accuracy | Colab notebook |
| 8 | Integrate rustbpe tokenizer | 2 days | Exact token counting | NEW `tokenizer.rs`, `server/mod.rs` |
| 9 | Eliminate double intent classification | 30 min | -2s latency | `router.rs` |
| 10 | Add MIVI_MODEL_TIER env var | 2 hrs | User-selectable models | `runtime.rs`, `models.json` |
| 11 | Benchmark MiniCPM5-1B vs Qwen3 1.7B | 2 hrs | Best model selection | scripts/ |

### 🟡 MEDIUM (Do Next Month)

| # | Task | Effort | Impact | Files |
|---|---|---|---|---|
| 12 | Integrate MiniLM for semantic RAG | 1 week | 3-5x better retrieval | NEW `embeddings.rs`, `rag.rs` |
| 13 | Integrate Cactus Needle natively | 3 days | <2ms native routing | `router.rs`, NEW model loading |
| 14 | Grammar in Candle native path | 1 week | Grammar for native inference | `native_brain.rs` |
| 15 | Publish mivi-agent models on HuggingFace | 1 day | Community | HuggingFace |
| 16 | KV cache quantization | 1 week | Free ~100 MB RAM | `native_brain.rs` |

### 🔵 FUTURE (Next Quarter)

| # | Task | Effort | Impact | When |
|---|---|---|---|---|
| 17 | Multi-Token Prediction in Candle | 2 weeks | 2-3x speed | When MTP models available |
| 18 | RLM recursive context navigation | 2 weeks | Infinite workspace handling | After grammar works |
| 19 | LiquidAI SSM backend | 1 month | Constant-memory context | When 1B LFM ships |
| 20 | WASM compilation target | 2 weeks | Browser deployment | After semantic RAG |
| 21 | Supra2-100M as dedicated router | 1 week | Smarter routing | After Needle integration |
| 22 | ThinkingCap early-exit fine-tuning | 1 day | 50% fewer thinking tokens | During fine-tune round 2 |
| 23 | BitNet/Bonsai ternary models | TBD | 4B in <1 GB | When models available |

---

## 5. Success Metrics

### Before (v0.0.10)
| Metric | Value |
|---|---|
| Tool calling accuracy | 67% |
| Simple query latency | 5.6s |
| Peak RAM | 950 MB |
| Token counting error | ~15% |
| RAG relevance | Keyword-only |
| Stream cleanliness | Think blocks leak |

### After (v1.0.0 Target)
| Metric | Target |
|---|---|
| Tool calling accuracy | **95%+** (grammar + fine-tune) |
| Simple query latency | **<3.5s** (single classify + faster model) |
| Peak RAM | **<800 MB** (756 MB target) |
| Token counting error | **0%** (exact BPE) |
| RAG relevance | **Semantic** (3-5x better) |
| Stream cleanliness | **Zero leaks** |

---

## 6. What Makes MIVI Different From Everyone Else

After all this research, here's our positioning:

> **MIVI is the only project that combines:**
> 1. Pure Rust single binary (no Python, no Go, no Docker)
> 2. OpenAI-compatible API (drop-in replacement)
> 3. Grammar-constrained tool calling (guaranteed valid JSON)
> 4. Semantic RAG (MiniLM embeddings)
> 5. Custom fine-tuned models (optimized for agent workflows)
> 6. Under 1 GB RAM total (model + embeddings + router + tokenizer)
> 7. Runs on any device with CPU (no GPU required)

**No other project in existence does all 7 of these things.**

Ollama does #2 and #7 but not #1, #3, #4, #5, #6.  
mistral.rs does #1, #2, #3 but not #5, #6.  
llama.cpp does #7 but not #1, #2, #3, #4, #5.  
vLLM does #2 but nothing else from this list.

**This is our moat. This is what makes MIVI the best.**
