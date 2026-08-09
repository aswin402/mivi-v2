# CPU SIMD Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable CPU target-native SIMD optimizations (AVX2/FMA/F16C) for Candle native inference and print compile-time feature diagnostics on server startup.

**Architecture:** We will use Rust compile-time target-feature macros to detect SIMD instruction sets, log active extensions during native brain initialization, and update the Makefile and documentation to support the optimized compiler flag pipeline.

**Tech Stack:** Rust, Candle, Cargo

## Global Constraints
- Target compilation features: AVX2, FMA, F16C, NEON
- Preserve zero-dependency native inference footprint

---

### Task 1: Makefile and Documentation Updates

**Files:**
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: None
- Produces: None

- [ ] **Step 1: Write the Makefile changes**
  Modify [`Makefile`](file:///home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi-v2/Makefile) to append the native-cpu optimization flags and targets:
  ```make
  .PHONY: check-agent check-agent-live build-native run-native check-native

  check-agent:
  	scripts/check_agent_compat.py --live off

  check-agent-live:
  	scripts/check_agent_compat.py --live auto

  build-native:
  	RUSTFLAGS="-C target-cpu=native" cargo build --release --features native

  run-native:
  	RUSTFLAGS="-C target-cpu=native" cargo run --release --features native -- serve

  check-native:
  	RUSTFLAGS="-C target-cpu=native" cargo test --release --features native
  	RUSTFLAGS="-C target-cpu=native" cargo fmt --check
  	RUSTFLAGS="-C target-cpu=native" cargo build --release --features native
  ```

- [ ] **Step 2: Run verification command for Makefile syntax**
  Run: `make -n build-native`
  Expected Output: Print the command `RUSTFLAGS="-C target-cpu=native" cargo build --release --features native` without executing it.

- [ ] **Step 3: Update README documentation**
  Modify [`README.md`](file:///home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi-v2/README.md) to add compilation instructions. Search for `## 🚀 Getting Started` or similar in `README.md` and add:
  ```markdown
  ### Native CPU SIMD Acceleration
  To build and run MIVI-V2 with native in-process GGUF inference accelerated by host CPU vector extensions (AVX2, FMA, F16C, or NEON), run:
  ```bash
  make build-native
  make run-native
  ```
  ```

- [ ] **Step 4: Commit Makefile and README changes**
  Run:
  ```bash
  git add Makefile README.md
  git commit -m "build: add makefile targets and readme docs for CPU native SIMD"
  ```

---

### Task 2: CPU SIMD Feature Diagnostic Logging

**Files:**
- Modify: `src/native_brain.rs`

**Interfaces:**
- Consumes: None
- Produces: Log of active compile-time target vector features on NativeBrain initialization

- [ ] **Step 1: Add diagnostic logs to NativeBrain initialization**
  Add logging code inside `NativeBrain::new()` in [`src/native_brain.rs`](file:///home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi-v2/src/native_brain.rs).
  Locate `pub fn new()` under `#[cfg(feature = "native")]` and insert the diagnostic logging:
  ```rust
  #[cfg(feature = "native")]
  impl NativeBrain {
      pub fn new() -> Self {
          let mut features = Vec::new();
          if cfg!(target_feature = "avx2") {
              features.push("AVX2");
          }
          if cfg!(target_feature = "fma") {
              features.push("FMA");
          }
          if cfg!(target_feature = "f16c") {
              features.push("F16C");
          }
          if cfg!(target_feature = "neon") {
              features.push("NEON");
          }
          info!(
              "[NativeBrain] Native CPU vectorization active: {:?}",
              features
          );

          Self {
              loaded: Arc::new(Mutex::new(HashMap::new())),
          }
      }
  }
  ```

- [ ] **Step 2: Run verification test**
  Run: `make check-native`
  Expected Output: Compilation completes cleanly and tests pass.

- [ ] **Step 3: Commit CPU SIMD Feature Diagnostic changes**
  Run:
  ```bash
  git add src/native_brain.rs
  git commit -m "feat: add CPU SIMD feature diagnostic logging on startup"
  ```
