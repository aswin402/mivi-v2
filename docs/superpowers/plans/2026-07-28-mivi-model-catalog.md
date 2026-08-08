# MIVI Model Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenLLM-inspired local model catalog and lightweight CLI commands so MIVI can manage internal worker models without exposing internal names to agents.

**Architecture:** Keep external API model identity as `mivi`. Store internal worker metadata in `configs/models.json`, load it through a small Rust module, and expose read-only CLI commands that do not initialize heavy runtime/RAG. Later tasks can use this catalog for backend selection and benchmark recording.

**Tech Stack:** Rust 2021, `serde`, `serde_json`, existing Cargo binary, JSON config file.

## Global Constraints

- Keep external OpenAI-compatible model name as `mivi`.
- Do not add Python/cloud dependencies from OpenLLM.
- Keep catalog commands low-resource: no model load, no RAG indexing, no server startup.
- Preserve existing `serve`, `audit`, `cli`, `chat`, and `task` behavior.
- Use TDD for Rust behavior changes.

---

### Task 1: Local Model Catalog And CLI

**Files:**
- Create: `configs/models.json`
- Create: `src/model_catalog.rs`
- Modify: `src/lib.rs`
- Modify: `src/main.rs`

**Interfaces:**
- Produces: `model_catalog::ModelCatalog::load_default() -> Result<ModelCatalog, ModelCatalogError>`
- Produces: `model_catalog::ModelCatalog::find(&self, id: &str) -> Option<&ModelCatalogEntry>`
- Produces: `model_catalog::print_model_list(&ModelCatalog)`
- Produces: `model_catalog::print_model_inspect(&ModelCatalog, &str) -> Result<(), ModelCatalogError>`

- [ ] **Step 1: Write failing tests**

Add tests in `src/model_catalog.rs`:

```rust
#[test]
fn parses_catalog_and_finds_mivi_external_model() {
    let catalog = ModelCatalog::from_json(SAMPLE).expect("catalog should parse");
    assert_eq!(catalog.external_model, "mivi");
    assert_eq!(catalog.models.len(), 2);
    assert_eq!(catalog.find("qwen3-06b-reasoner").unwrap().role, ModelRole::Reasoner);
}

#[test]
fn rejects_catalog_that_exposes_non_mivi_external_model() {
    let err = ModelCatalog::from_json(r#"{"external_model":"qwen","models":[]}"#)
        .expect_err("external model must be mivi");
    assert!(err.to_string().contains("external_model must be mivi"));
}
```

Run: `cargo test model_catalog --quiet`
Expected: FAIL because module does not exist.

- [ ] **Step 2: Implement catalog types and parser**

Create `src/model_catalog.rs` with `ModelRole`, `BackendKind`, `ModelCatalogEntry`, `ModelCatalog`, `ModelCatalogError`, JSON loading, validation, and tests.

- [ ] **Step 3: Add default catalog config**

Create `configs/models.json` with entries for `qwen3-06b-reasoner`, `qwen25-05b-coder`, and `minicpm-vision`.

- [ ] **Step 4: Expose module**

Add `pub mod model_catalog;` to `src/lib.rs`.

- [ ] **Step 5: Add lightweight CLI commands**

Modify `src/main.rs` so `mivi model list` and `mivi model inspect <id>` run before `EdgeBrain::new()` and RAG indexing.

- [ ] **Step 6: Verify**

Run:

```bash
cargo fmt --check
cargo test model_catalog --quiet
cargo test --quiet
cargo build --release --quiet
./target/release/mivi model list
./target/release/mivi model inspect qwen3-06b-reasoner
```

Expected: tests/build pass, CLI prints catalog data and still exposes external model as `mivi`.

---

### Task 2: Catalog-Backed Runtime Defaults

**Files:**
- Modify: `src/brain.rs`
- Modify: `src/model_catalog.rs`
- Test: existing `brain` tests plus new catalog selection tests.

**Interfaces:**
- Consumes: `ModelCatalog::load_default()` and `ModelCatalogEntry`.
- Produces: helper to resolve default reasoner/coder/vision paths from catalog unless env vars override.

- [ ] **Step 1: Write failing tests for catalog defaults with env override preserved**
- [ ] **Step 2: Implement default model path resolution from catalog**
- [ ] **Step 3: Verify brain tests and release build**

---

### Task 3: OpenAI Compatibility Smoke Script

**Files:**
- Create: `scripts/check_openai_compat.py`
- Modify: `README.md`

**Interfaces:**
- Consumes running MIVI server at `http://127.0.0.1:8000/v1`.
- Produces JSONL/plain output for models, non-stream chat, stream chat, inventory, and tool-call checks.

- [ ] **Step 1: Add smoke script with no third-party Python deps**
- [ ] **Step 2: Validate against running local server**
- [ ] **Step 3: Document usage**

---

### Task 4: Benchmark Metadata Update

**Files:**
- Modify: `scripts/bench_runtime.sh`
- Modify: `configs/models.json`

**Interfaces:**
- Consumes benchmark rows.
- Produces optional benchmark fields under matching catalog entries.

- [ ] **Step 1: Add benchmark-output-to-catalog test helper or dry-run mode**
- [ ] **Step 2: Implement catalog benchmark update path**
- [ ] **Step 3: Verify benchmark still runs without catalog mutation by default**

---

## Self-Review

Spec coverage: covers OpenLLM-inspired model catalog, single-command inspection, OpenAI compatibility smoke tests, and benchmark metadata without adopting Python/cloud stack.

Placeholder scan: no TBD placeholders in Task 1; later tasks are intentionally scoped but need expansion before separate execution.

Type consistency: Task 1 defines the catalog interfaces used by later tasks.
