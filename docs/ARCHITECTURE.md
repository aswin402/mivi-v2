# 🏗️ MIVI-V2 System Architecture

This document provides a detailed technical overview of **MIVI-V2**, an ultra-compact Small Model Logic (SML) engine written in 100% Pure Rust.

---

## 🏛️ System Overview

MIVI-V2 functions as an asynchronous orchestration service that coordinates low-level LLM inferencing via native `llama-cli` subprocesses.

```text
                                +---------------------------+
                                |  REST API / CLI / Task    |
                                +-------------+-------------+
                                              |
                                              v
                                +-------------+-------------+
                                |     NeedleRouter          |
                                +-------------+-------------+
                                              |
                                              v
                                +-------------+-------------+
                                |     AgentOrchestrator     |
                                +------+--------------+-----+
                                       |              |
                    +------------------+              +-------------------+
                    |                                                     |
                    v                                                     v
      +-------------+-------------+                         +-------------+-------------+
      |      SemanticCache        |                         |     CompilerVerifier        |
      +---------------------------+                         +-------------+-------------+
                                                                          |
                                                                          v
                                                            +-------------+-------------+
                                                            |       EdgeBrain           |
                                                            +-------------+-------------+
                                                                          |
                                                                          v
                                                            +-------------+-------------+
                                                            |   llama-cli Subprocesses   |
                                                            +---------------------------+
```

---

## 🧩 Core Components

### 1. `EdgeBrain` ([src/brain.rs](../src/brain.rs))
* **Purpose**: Subprocess wrapper around native `llama-cli` binaries.
* **Inference Speedups**:
  * FlashAttention (`-fa on`)
  * 8-bit Quantized Key Cache (`-ctk q8_0`)
  * 8-bit Quantized Value Cache (`-ctv q8_0`)
  * GPU Offloading (`-ngl 999`, falls back automatically to CPU SIMD if no GPU is found)
  * Default Context Size: `8192` tokens (active process RAM ~180 MB – 250 MB).

### 2. `CompilerVerifier` ([src/verifier.rs](../src/verifier.rs))
* **Purpose**: Double-Loop code execution and self-correction engine.
* **Supported Execution Environments**:
  * **Python** (`python3`)
  * **JavaScript** (`node`)
  * **TypeScript** (`bun` / `node`)
  * **Rust** (`rustc` temporary binary compilation & execution)
  * **C / C++** (`g++` temporary binary compilation & execution)
* **Self-Correction Loop**: If execution fails with an error, the compiler verifier captures `stderr` and feeds it back to `Qwen-2.5-0.5B` up to 3 times until exit code is `0`.

### 3. `NeedleRouter` ([src/router.rs](../src/router.rs))
* Zero-overhead heuristic intent classifier that routes requests between `CHAT`, `VISION`, `MULTI_STEP`, and `DIRECT_CODE`.

### 4. `TurboVecRAG` ([src/rag.rs](../src/rag.rs))
* Scans workspace files (`.py`, `.js`, `.ts`, `.rs`, `.md`, `.json`, `.toml`) using `walkdir`.
* Chunks files into 25-line windows.
* Applies stop-word filtering to prevent false keyword matches.
* Formats code snippets with commented line prefixes (`# `) to prevent LLM code generation pollution.

### 5. `SemanticCache` ([src/cache.rs](../src/cache.rs))
* Dual-stage prompt cache:
  1. Exact string match (`HashMap`).
  2. Token-set Jaccard similarity match ($\ge 0.85$ threshold) for rephrased queries.
* Delivers responses in **< 0.001s**.

---

## 📈 Resource Footprint

* **Idle Memory**: **< 12 MB RAM**
* **Active Memory**: **~180 MB – 250 MB RAM**
* **Storage Footprint**: **~1.12 GB** (models directory + binary)
