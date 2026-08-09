# MIVI-V2 Design Specification: CPU SIMD Optimization

- **Date**: 2026-08-10
- **Status**: Approved
- **Phase**: 8.5 & 8.6

---

## 1. Goal
Optimize native in-process GGUF inference in MIVI-V2 by utilizing host CPU vectorization instructions (AVX2, FMA, F16C, NEON) on AMD/Intel/ARM platforms via Candle and Rust target-cpu compilation features.

---

## 2. Approach
* **Auto-Vectorization & Target-CPU Optimizations**:
  Build the binary with target-cpu flags enabled to compile matrix multiplication kernels with SIMD support:
  `RUSTFLAGS="-C target-cpu=native" cargo build --release --features native`
* **Startup SIMD Diagnostic logs**:
  Check `target_feature` configuration at compile time and log active CPU vector features on startup.
* **Makefile Tooling**:
  Provide unified targets for compiling, testing, and running native inference.

---

## 3. Detailed Changes

### 3.1. Makefile Build-System Update
Update the root `Makefile` with:
* `build-native`: `RUSTFLAGS="-C target-cpu=native" cargo build --release --features native`
* `run-native`: `RUSTFLAGS="-C target-cpu=native" cargo run --release --features native -- serve`
* `check-native`: Runs tests, format checks, and build steps with native optimizations.

### 3.2. Diagnostic Logging
In `src/native_brain.rs` under `#[cfg(feature = "native")]` inside `NativeBrain::new()`:
```rust
let mut features = Vec::new();
if cfg!(target_feature = "avx2") { features.push("AVX2"); }
if cfg!(target_feature = "fma") { features.push("FMA"); }
if cfg!(target_feature = "f16c") { features.push("F16C"); }
if cfg!(target_feature = "neon") { features.push("NEON"); }

tracing::info!("[NativeBrain] Native CPU vectorization active: {:?}", features);
```

### 3.3. README Documentation
Update `README.md` to document how to compile and run with CPU vectorization.

---

## 4. Verification Plan
* Compile using `make build-native`.
* Run using `make run-native` and verify that the startup log prints `[NativeBrain] Native CPU vectorization active: ["AVX2", "FMA", "F16C"]` (or the equivalent features supported by the host).
* Run `cargo test --release --features native` to verify all native tests are green.
