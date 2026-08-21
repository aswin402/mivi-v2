# Security & Bugfix Audit Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the security holes, bugs, and hardcoded smells found in the 2026-08-21 project audit (RCE exposure surface, bypassable rate limiter, unsafe semantic cache, doc/config lies, broken TS fallback, regex recompiles).

**Architecture:** All changes are surgical edits to existing modules (`src/server/helpers.rs`, `src/server/types.rs`, `src/server/handlers.rs`, `src/cache.rs`, `src/orchestrator.rs`, `src/verifier.rs`, `src/brain.rs`, `src/runtime.rs`, `configs/models.json`). No new files except none required; no new crates. Every task is TDD where a unit test can observe the behavior, ends in its own commit.

**Tech Stack:** Rust (edition 2021), axum 0.7, tokio, serde_json. Python3 stdlib for config validation. No new dependencies — use `std::sync::OnceLock` (stable since 1.70), never `once_cell`.

## Global Constraints

- **Low-resource builds:** ALWAYS run cargo with `-j 2` (max 2 parallel compile jobs) and `RUST_TEST_THREADS=2`. Example: `cargo test -j 2 --lib router::`. Never run bare `cargo build/test/check` without `-j 2`.
- **No full-suite spam:** run only the targeted test filter for the task; run the FULL suite exactly once at the end (Task 13).
- **Tests are inline:** all Rust tests go in the existing `#[cfg(test)] mod tests` block at the bottom of the file being changed. There is no `tests/` directory.
- **Env-var tests must serialize:** any test that reads/writes `MIVI_*` env vars must take the file-local `env_lock()` mutex (see pattern in `src/runtime.rs:166`). `std::env` is process-global and `cargo test` runs in parallel.
- **Formatting is CI-enforced:** run `cargo fmt` before every commit; CI runs `cargo fmt --check`.
- **External model name stays `mivi`.** Never leak internal model ids into API responses.
- **Minimal behavior change:** preserve existing semantics except where the task explicitly changes them.
- **Explicit non-goals (documented, do not attempt):** OS-level sandboxing of the verifier subprocess (needs container/namespace design — separate plan); replacing the `is_complex` substring heuristic and `repair_python_code` hack (behavior-risky, accepted limitations); `-ngl 999` stays (llama.cpp idiom).

---

### Task 1: Bind to loopback by default (+ `MIVI_HOST` / `MIVI_PORT` overrides)

The server binds `0.0.0.0` unconditionally (`src/server/helpers.rs:4488`) while README claims localhost. Combined with optional auth this exposes the code-executing API to the LAN. Default becomes `127.0.0.1`; operators opt into exposure explicitly.

**Files:**
- Modify: `src/server/helpers.rs` (function `start_api_server`, ~line 4440–4500; add helper near it; add test in bottom test module)

**Interfaces:**
- Produces: `fn resolve_bind_host() -> String` (private, tested). `start_api_server` signature unchanged — main.rs keeps calling `start_api_server(brain, orchestrator, 8000)`.

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `src/server/helpers.rs`:

```rust
#[test]
fn bind_host_defaults_to_loopback_and_honors_env() {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let _guard = LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap();

    std::env::remove_var("MIVI_HOST");
    assert_eq!(resolve_bind_host(), "127.0.0.1");

    std::env::set_var("MIVI_HOST", "0.0.0.0");
    assert_eq!(resolve_bind_host(), "0.0.0.0");

    // Whitespace-only values fall back to the safe default.
    std::env::set_var("MIVI_HOST", "   ");
    assert_eq!(resolve_bind_host(), "127.0.0.1");

    std::env::remove_var("MIVI_HOST");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -j 2 --lib bind_host_defaults`
Expected: FAIL — `resolve_bind_host` not found (compile error).

- [ ] **Step 3: Implement**

Add above `start_api_server` in `src/server/helpers.rs`:

```rust
/// Resolve the bind host. Defaults to loopback so the API (which executes
/// model-generated code) is never exposed to the network by accident.
/// Set MIVI_HOST=0.0.0.0 to listen on all interfaces deliberately.
fn resolve_bind_host() -> String {
    std::env::var("MIVI_HOST")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}
```

Inside `start_api_server`, replace:

```rust
    let addr = format!("0.0.0.0:{}", port);
```

with:

```rust
    let port = std::env::var("MIVI_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .filter(|_| port == 8000) // only override the built-in default
        .unwrap_or(port);
    let addr = format!("{}:{}", resolve_bind_host(), port);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -j 2 --lib bind_host_defaults`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/server/helpers.rs
git commit -m "fix(security): bind API to 127.0.0.1 by default, add MIVI_HOST/MIVI_PORT overrides"
```

---

### Task 2: Constant-time API key comparison

`auth_middleware` compares bearer tokens with `==` (`src/server/helpers.rs:4420`), which leaks length/prefix info via timing. Local-only threat, but the fix is 10 lines.

**Files:**
- Modify: `src/server/helpers.rs` (above `auth_middleware`; swap comparison; add test)

**Interfaces:**
- Produces: `fn constant_time_eq(a: &str, b: &str) -> bool` (private, tested). Used only inside `auth_middleware`.

- [ ] **Step 1: Write the failing test**

Append to the test module in `src/server/helpers.rs`:

```rust
#[test]
fn constant_time_eq_matches_only_equal_strings() {
    assert!(constant_time_eq("secret", "secret"));
    assert!(!constant_time_eq("secret", "secreT"));
    assert!(!constant_time_eq("secret", "secret "));
    assert!(!constant_time_eq("short", "shorter"));
    assert!(constant_time_eq("", ""));
    assert!(!constant_time_eq("", "x"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -j 2 --lib constant_time_eq`
Expected: FAIL — function not found.

- [ ] **Step 3: Implement**

Add above `auth_middleware`:

```rust
/// Length-checked XOR-fold comparison. Not literally branch-free, but removes
/// the early-exit-on-first-mismatch byte leak of `==`.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}
```

In `auth_middleware`, replace `if token == expected_key {` with `if constant_time_eq(token, expected_key) {`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -j 2 --lib constant_time_eq`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/server/helpers.rs
git commit -m "fix(security): constant-time bearer token comparison"
```

---

### Task 3: Harden the rate limiter

Three defects: (a) client identity trusts spoofable `X-Forwarded-For`/`X-Real-IP` headers (`src/server/helpers.rs:4345`), letting attackers rotate identity per request; (b) the identity `HashMap` grows without bound (memory DoS); (c) 60 req/min is hardcoded.

**Files:**
- Modify: `src/server/types.rs` (`RateLimiter`, lines ~382–415; add tests)
- Modify: `src/server/helpers.rs` (`get_client_identifier`, `rate_limit_middleware`, serve call at line 4565)

**Interfaces:**
- Consumes: nothing new.
- Produces: `RateLimiter::MAX_TRACKED_CLIENTS: usize` (assoc const, = 4096); `get_client_identifier(req: &Request, peer: Option<SocketAddr>) -> String` (signature CHANGES — callers updated in this task); env var `MIVI_RATE_LIMIT_PER_MIN` (default 60), `MIVI_TRUST_PROXY_HEADERS` (default off). Serve call becomes `app.into_make_service_with_connect_info::<SocketAddr>()`.

- [ ] **Step 1: Write failing tests**

Append to the test module at the bottom of `src/server/types.rs` (create `mod tests { use super::*; ... }` if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_blocks_after_configured_limit() {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        let _guard = LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap();
        std::env::set_var("MIVI_RATE_LIMIT_PER_MIN", "2");

        let rl = RateLimiter::new();
        assert!(rl.check_rate_limit("client-x".to_string()).is_ok());
        assert!(rl.check_rate_limit("client-x".to_string()).is_ok());
        assert!(rl.check_rate_limit("client-x".to_string()).is_err());

        std::env::remove_var("MIVI_RATE_LIMIT_PER_MIN");
    }

    #[test]
    fn rate_limiter_tracks_clients_independently() {
        let rl = RateLimiter::new();
        assert!(rl.check_rate_limit("a".to_string()).is_ok());
        assert!(rl.check_rate_limit("b".to_string()).is_ok());
    }

    #[test]
    fn rate_limiter_caps_tracked_clients_against_spoof_flood() {
        let rl = RateLimiter::new();
        for i in 0..(RateLimiter::MAX_TRACKED_CLIENTS + 200) {
            let _ = rl.check_rate_limit(format!("spoof-{i}"));
        }
        assert!(
            rl.requests.lock().unwrap().len() <= RateLimiter::MAX_TRACKED_CLIENTS + 1,
            "map grew past the hard cap"
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -j 2 --lib rate_limiter`
Expected: FAIL — `MAX_TRACKED_CLIENTS` not found.

- [ ] **Step 3: Implement RateLimiter changes in `src/server/types.rs`**

Replace the `RateLimiter` impl body:

```rust
impl RateLimiter {
    /// Hard ceiling on simultaneously tracked identities so header-spoofing
    /// floods cannot grow the map without bound.
    pub const MAX_TRACKED_CLIENTS: usize = 4096;

    pub fn check_rate_limit(&self, client_id: String) -> Result<(), String> {
        let max_requests = std::env::var("MIVI_RATE_LIMIT_PER_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(60);

        let mut reqs = self.requests.lock().unwrap();
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60);

        // Cap tracked identities BEFORE inserting a new one.
        if !reqs.contains_key(&client_id) && reqs.len() >= Self::MAX_TRACKED_CLIENTS {
            // First drop everyone whose window fully expired.
            reqs.retain(|_, v| v.iter().any(|&t| t > cutoff));
            // Still full: evict arbitrary entries (HashMap iteration order).
            while reqs.len() >= Self::MAX_TRACKED_CLIENTS {
                match reqs.keys().next().cloned() {
                    Some(k) => {
                        reqs.remove(&k);
                    }
                    None => break,
                }
            }
        }

        let times = reqs.entry(client_id).or_default();
        times.retain(|&t| t > cutoff);

        if times.len() >= max_requests {
            return Err(format!(
                "Rate limit exceeded (max {} requests per minute).",
                max_requests
            ));
        }

        times.push(now);
        Ok(())
    }
}
```

- [ ] **Step 4: Implement identity + middleware changes in `src/server/helpers.rs`**

Replace `get_client_identifier` entirely:

```rust
fn get_client_identifier(
    req: &axum::http::Request<axum::body::Body>,
    peer: Option<std::net::SocketAddr>,
) -> String {
    // Proxy headers are trusted ONLY when the operator opts in; otherwise any
    // client could rotate X-Forwarded-For to dodge the limiter.
    let trust_proxy = std::env::var("MIVI_TRUST_PROXY_HEADERS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if trust_proxy {
        if let Some(forwarded) = req.headers().get("x-forwarded-for") {
            if let Ok(s) = forwarded.to_str() {
                if let Some(first_ip) = s.split(',').next() {
                    let ip = first_ip.trim();
                    if !ip.is_empty() {
                        return ip.to_string();
                    }
                }
            }
        }
        if let Some(real_ip) = req.headers().get("x-real-ip") {
            if let Ok(s) = real_ip.to_str() {
                let ip = s.trim();
                if !ip.is_empty() {
                    return ip.to_string();
                }
            }
        }
    }

    if let Some(addr) = peer {
        return addr.ip().to_string();
    }

    "generic_client".to_string()
}
```

Update `rate_limit_middleware` to extract `ConnectInfo`:

```rust
async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    let client_id = get_client_identifier(&req, Some(peer));
    // ... rest unchanged
```

Update the serve call (~line 4565) so `ConnectInfo` is populated:

```rust
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -j 2 --lib rate_limiter`
Expected: PASS (3 tests). Compile success also proves the middleware/service wiring is type-correct.

- [ ] **Step 6: Commit**

```bash
cargo fmt
git add src/server/types.rs src/server/helpers.rs
git commit -m "fix(security): socket-addr rate-limit identity, bounded tracker, configurable limit"
```

---

### Task 4: Semantic cache — exact-match for the code path + TTL

Jaccard ≥ 0.85 word-overlap serves WRONG cached answers for similar-but-different coding prompts (quicksort vs mergesort ≈ 0.89), and cached verified-code never expires. The only caller is `AgentOrchestrator::execute_plan` (verified-code results), so it switches to exact matching; semantic lookup gains a TTL and stays available for future chat use.

**Files:**
- Modify: `src/cache.rs` (struct field type, `get`, new `get_exact`, `put`)
- Modify: `src/orchestrator.rs` (line ~153: `self.cache.get(request)` → `self.cache.get_exact(request)`)

**Interfaces:**
- Produces: `SemanticCache::get_exact(&self, query: &str) -> Option<String>`; internal entry type becomes `(String, std::time::SystemTime)`; const `CACHE_TTL_SECS: u64 = 600`. Orchestrator calls `get_exact`.

- [ ] **Step 1: Write failing tests**

Append to the test module in `src/cache.rs`:

```rust
#[tokio::test]
async fn get_exact_ignores_similar_but_different_prompts() {
    let cache = SemanticCache::new();
    cache
        .put("write a function to sort a list using quicksort", "quick-code")
        .await;
    assert_eq!(
        cache
            .get_exact("write a function to sort a list using mergesort")
            .await,
        None
    );
    // Trimming still yields an exact hit.
    assert_eq!(
        cache
            .get_exact("  write a function to sort a list using quicksort  ")
            .await,
        Some("quick-code".to_string())
    );
}

#[tokio::test]
async fn get_exact_expires_after_ttl() {
    let cache = SemanticCache::new();
    cache.put("q", "a").await;
    // Backdate the stored timestamp past the TTL (tests share the module's
    // private field access).
    {
        let mut guard = cache.cache.lock().await;
        guard.get_mut("q").unwrap().1 =
            std::time::SystemTime::now() - std::time::Duration::from_secs(601);
    }
    assert_eq!(cache.get_exact("q").await, None);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -j 2 --lib cache::tests`
Expected: FAIL — `get_exact` not found.

- [ ] **Step 3: Implement in `src/cache.rs`**

Add near the top:

```rust
/// Entries older than this are treated as misses and purged lazily.
const CACHE_TTL_SECS: u64 = 600;
```

Change the struct field type:

```rust
cache: Arc<Mutex<HashMap<String, (String, std::time::SystemTime)>>>,
```

Replace `get` with a TTL-aware version and add `get_exact`:

```rust
    fn fresh(ts: &std::time::SystemTime) -> bool {
        ts.elapsed().map(|d| d.as_secs() <= CACHE_TTL_SECS).unwrap_or(false)
    }

    /// Exact-key lookup with TTL. Used for verified-code results where a
    /// fuzzy hit would return WRONG code.
    pub async fn get_exact(&self, query: &str) -> Option<String> {
        let q_clean = query.trim();
        let mut guard = self.cache.lock().await;
        if let Some((val, ts)) = guard.get_mut(q_clean) {
            if !Self::fresh(ts) {
                guard.remove(q_clean);
                return None;
            }
            *ts = std::time::SystemTime::now(); // LRU touch
            println!("[SemanticCache] EXACT CACHE HIT!");
            return Some(val.clone());
        }
        None
    }

    pub async fn get(&self, query: &str) -> Option<String> {
        let q_clean = query.trim();
        let mut guard = self.cache.lock().await;

        if let Some((val, ts)) = guard.get_mut(q_clean) {
            if !Self::fresh(ts) {
                guard.remove(q_clean);
                return None;
            }
            println!("[SemanticCache] EXACT CACHE HIT!");
            *ts = std::time::SystemTime::now();
            return Some(val.clone());
        }

        let mut best_score = 0.0f32;
        let mut best_result: Option<String> = None;
        let mut best_key: Option<String> = None;

        for (k, (v, ts)) in guard.iter() {
            if !Self::fresh(ts) {
                continue;
            }
            let score = Self::jaccard_similarity(q_clean, k);
            if score > best_score {
                best_score = score;
                best_result = Some(v.clone());
                best_key = Some(k.clone());
            }
        }

        if best_score >= 0.85 {
            println!(
                "[SemanticCache] SEMANTIC CACHE HIT! Similarity score: {:.4}",
                best_score
            );
            if let Some(ref k) = best_key {
                if let Some((_, ts)) = guard.get_mut(k) {
                    *ts = std::time::SystemTime::now();
                }
            }
            best_result
        } else {
            None
        }
    }
```

In `put`, change the insert line's timestamp:

```rust
        guard.insert(q_clean, (result.to_string(), std::time::SystemTime::now()));
```

Then in `src/orchestrator.rs` (~line 153) change:

```rust
        if let Some(cached) = self.cache.get(request).await {
```

to:

```rust
        if let Some(cached) = self.cache.get_exact(request).await {
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -j 2 --lib cache::` then `cargo test -j 2 --lib orchestrator`
Expected: PASS (all cache tests including pre-existing eviction/exact-hit tests; note the old tuple type was `Instant` — existing tests only touch `len()` and `put/get`, so they compile unchanged).

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/cache.rs src/orchestrator.rs
git commit -m "fix(cache): exact-match only for verified-code cache, add 10min TTL"
```

---

### Task 5: Truthful `/v1/models` context length + version sync

`handle_models` advertises `131072` (real budget: 8192) and `handle_root` hardcodes `"0.0.11"` while Cargo.toml is `0.0.14`. README repeats v0.0.11 in three places.

**Files:**
- Modify: `src/server/handlers.rs` (`handle_root` line ~14, `handle_models` line ~49; add tests)
- Modify: `README.md` (badge line 4, intro line 8, "Key Features in v0.0.11" heading — all `v0.0.11` → `v0.0.14`)

**Interfaces:**
- Consumes: `crate::runtime::RuntimeConfig::global()` (exists), `env!("CARGO_PKG_VERSION")` (compile-time macro).

- [ ] **Step 1: Write failing tests**

Append to the test module at the bottom of `src/server/handlers.rs`:

```rust
#[tokio::test]
async fn root_reports_cargo_package_version() {
    let Json(value) = handle_root().await;
    assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn models_report_runtime_context_budget_not_a_lie() {
    let Json(resp) = handle_models().await;
    let expected = crate::runtime::RuntimeConfig::global().context.max_input_tokens;
    assert_eq!(resp.data[0].context_length, Some(expected));
}
```

Note: `handle_root`/`handle_models` return `axum::extract::Json` — destructure it in the test as shown. If `Json` isn't imported in the test scope, use `handle_root().await.0` instead.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -j 2 --lib handlers::tests`
Expected: FAIL — version mismatch (`0.0.11` vs `0.0.14`) and context mismatch (`131072` vs `8192`).

- [ ] **Step 3: Implement**

In `handle_root`, replace the hardcoded line:

```rust
        "version": "0.0.11",
```

with:

```rust
        "version": env!("CARGO_PKG_VERSION"),
```

In `handle_models`, replace the hardcoded context:

```rust
            context_length: Some(131072),
```

with:

```rust
            context_length: Some(crate::runtime::RuntimeConfig::global().context.max_input_tokens),
```

In `README.md`, replace all three occurrences of `v0.0.11` with `v0.0.14` (badge, "**MIVI-V2 (v0.0.11)**", "Key Features in v0.0.11").

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -j 2 --lib handlers::tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/server/handlers.rs README.md
git commit -m "fix(api): report real context budget and Cargo version in /v1 endpoints"
```

---

### Task 6: Unify context constants + configurable RAM target

Context size appears as `8192` (`src/constants.rs:6`), `3072` fallback (`src/brain.rs:581`), and `3072` in AGENTS.md prose. `ram_target_mb` is fixed at 1000 with no override.

**Files:**
- Modify: `src/brain.rs` (lines ~578–588 in `run_cli_spawn`)
- Modify: `src/runtime.rs` (`ram_target_mb` from env; add test)
- Modify: `AGENTS.md` (the "runtime default is 3072" claim — actual edit lands in Task 12 with the other doc fixes; only code here)

**Interfaces:**
- Produces: env var `MIVI_RAM_TARGET_MB` (default 1000). `RuntimeConfig.ram_target_mb` field unchanged.

- [ ] **Step 1: Write failing test**

Append to the test module in `src/runtime.rs` (inside the existing env-locked style):

```rust
    #[test]
    fn ram_target_mb_is_configurable_via_env() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RAM_TARGET_MB", "1500");

        let config = RuntimeConfig::from_env();
        assert_eq!(config.ram_target_mb, 1500);

        std::env::set_var("MIVI_RAM_TARGET_MB", "not-a-number");
        let config = RuntimeConfig::from_env();
        assert_eq!(config.ram_target_mb, crate::constants::DEFAULT_RAM_TARGET_MB);

        std::env::remove_var("MIVI_RAM_TARGET_MB");
    }
```

Note: `clear_runtime_env()` in that file doesn't remove `MIVI_RAM_TARGET_MB` yet — add `std::env::remove_var("MIVI_RAM_TARGET_MB");` to it in this step.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -j 2 --lib ram_target_mb`
Expected: FAIL — env var ignored, always 1000.

- [ ] **Step 3: Implement**

In `src/runtime.rs::from_env`, next to the other env reads:

```rust
        let ram_target_mb = env::var("MIVI_RAM_TARGET_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|mb| *mb > 0)
            .unwrap_or(DEFAULT_RAM_TARGET_MB);
```

and use `ram_target_mb,` in the returned struct (replacing `ram_target_mb: DEFAULT_RAM_TARGET_MB,`).

In `src/brain.rs::run_cli_spawn`, replace:

```rust
        let eff_context_base = if self.ultra_low_ram && context_size == "8192" {
            4096
        } else {
            context_size.parse::<usize>().unwrap_or(3072)
        };
```

with (strictly equivalent behavior, constant-driven):

```rust
        let parsed_context = context_size
            .parse::<usize>()
            .unwrap_or(crate::constants::DEFAULT_CONTEXT_TOKENS);
        let eff_context_base = if self.ultra_low_ram
            && parsed_context == crate::constants::DEFAULT_CONTEXT_TOKENS
        {
            parsed_context / 2
        } else {
            parsed_context
        };
```

- [ ] **Step 4: Sweep for remaining magic numbers**

Run: `grep -rn "3072\|131072" src/ --include="*.rs"`
Expected: no hits left in `src/` (catalog `context_tokens: 8192` in configs/models.json is data, not code — leave it). If a hit is found in non-test code, replace with `crate::constants::DEFAULT_CONTEXT_TOKENS` unless it is a genuinely different quantity.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -j 2 --lib runtime::` then `cargo test -j 2 --lib brain`
Expected: PASS (existing runtime tests already assert 8192-derived budgets).

- [ ] **Step 6: Commit**

```bash
cargo fmt
git add src/runtime.rs src/brain.rs
git commit -m "refactor(config): single source for context constants, MIVI_RAM_TARGET_MB override"
```

---

### Task 7: Fix lying `notes` in configs/models.json

Notes say "(disabled)" on enabled entries and "(enabled)" on disabled ones.

**Files:**
- Modify: `configs/models.json` (entries `qwen2.5-0.5b-reasoner`, `qwen2.5-0.5b-coder`, `qwen3-1.7b-reasoner`, `qwen3-1.7b-coder`)

- [ ] **Step 1: Make the notes truthful**

Four edits:

- `qwen2.5-0.5b-reasoner`: `"notes": "Qwen 2.5 0.5B Instruct (disabled)"` → `"notes": "Qwen 2.5 0.5B Instruct (enabled fallback reasoner)"`
- `qwen2.5-0.5b-coder`: `"notes": "Qwen 2.5 0.5B Instruct (disabled)"` → `"notes": "Qwen 2.5 0.5B Instruct (enabled fallback coder)"`
- `qwen3-1.7b-reasoner`: `"notes": "Qwen3 1.7B Instruct (enabled)"` → `"notes": "Qwen3 1.7B Instruct (disabled)"`
- `qwen3-1.7b-coder`: `"notes": "Qwen3 1.7B Coder/Tool Worker (enabled)"` → `"notes": "Qwen3 1.7B Coder/Tool Worker (disabled)"`

- [ ] **Step 2: Validate JSON parses and flags match notes**

Run:
```bash
python3 - <<'EOF'
import json, re, sys
d = json.load(open("configs/models.json"))
bad = []
for m in d["models"]:
    note = m.get("notes", "")
    if "enabled" in note.lower() and not m["enabled"]:
        bad.append((m["id"], "note claims enabled"))
    if re.search(r"\(disabled\)", note) and m["enabled"]:
        bad.append((m["id"], "note claims disabled"))
print("BAD:", bad) if bad else print("OK: notes consistent with enabled flags")
sys.exit(1 if bad else 0)
EOF
```
Expected: `OK: notes consistent with enabled flags`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add configs/models.json
git commit -m "docs(config): correct misleading enabled/disabled notes in model catalog"
```

---

### Task 8: Gate the TypeScript node fallback on type-stripping support

Plain `node file.ts` fails on type syntax for node < 23.6 (flag-only since 22.6). Current fallback reports false verification failures.

**Files:**
- Modify: `src/verifier.rs` (fallback arm ~line 209–219; add helpers + tests)

**Interfaces:**
- Produces: `fn parse_node_version(output: &str) -> Option<(u32, u32)>`, `fn node_supports_strip_types(version_output: &str) -> bool` (both private, tested).

- [ ] **Step 1: Write failing tests**

Append to the test module in `src/verifier.rs`:

```rust
    #[test]
    fn node_version_parsing_and_strip_types_gate() {
        assert_eq!(parse_node_version("v22.14.0"), Some((22, 14)));
        assert_eq!(parse_node_version("v18.0.0"), Some((18, 0)));
        assert_eq!(parse_node_version("garbage"), None);
        assert!(node_supports_strip_types("v22.6.0"));
        assert!(node_supports_strip_types("v23.6.0"));
        assert!(node_supports_strip_types("v24.1.0"));
        assert!(!node_supports_strip_types("v22.5.0"));
        assert!(!node_supports_strip_types("v20.11.0"));
        assert!(!node_supports_strip_types("junk"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -j 2 --lib node_version_parsing`
Expected: FAIL — functions not found.

- [ ] **Step 3: Implement helpers**

Add below `timeout_secs_from_env` in `src/verifier.rs`:

```rust
/// Parse "v22.14.0" into (major, minor). Node gained `--experimental-strip-types`
/// in 22.6 and enables type stripping by default from 23.6.
fn parse_node_version(output: &str) -> Option<(u32, u32)> {
    let v = output.trim().trim_start_matches('v');
    let mut parts = v.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn node_supports_strip_types(version_output: &str) -> bool {
    match parse_node_version(version_output) {
        Some((major, minor)) => {
            major > 23 || (major == 23 && minor >= 6) || (major == 22 && minor >= 6)
        }
        None => false,
    }
}
```

- [ ] **Step 4: Rewire the fallback arm**

Replace the bun-fallback arm inside `run_local_code`:

```rust
                // Do not retry on timeout: a timed-out bun run would only time out
                // again under node and double the wall-clock cost.
                Err(ref e) if cmd_name == "bun" && e.kind() != std::io::ErrorKind::TimedOut => {
                    let is_ts = matches!(lang_lower.as_str(), "typescript" | "ts");
                    if is_ts {
                        // Node can only execute TypeScript with type-stripping
                        // support; probe once instead of failing confusingly.
                        let version = tokio::process::Command::new("node")
                            .arg("--version")
                            .kill_on_drop(true)
                            .output()
                            .await
                            .ok()
                            .map(|o| String::from_utf8_lossy(&o.stdout).to_string());
                        let supported = version
                            .as_deref()
                            .map(node_supports_strip_types)
                            .unwrap_or(false);
                        if !supported {
                            let _ = tokio::fs::remove_file(&temp_file).await;
                            return (
                                false,
                                "TypeScript verification requires bun, or node >= 22.6 \
                                 with type-stripping support."
                                    .to_string(),
                            );
                        }
                    }
                    let mut fallback_cmd = tokio::process::Command::new("node");
                    if is_ts {
                        fallback_cmd.arg("--experimental-strip-types");
                    }
                    fallback_cmd.arg(&temp_file).kill_on_drop(true);
                    let output = timed_output(&mut fallback_cmd, exec_timeout_secs()).await;
                    let _ = tokio::fs::remove_file(&temp_file).await;
                    match output {
                        Ok(out) => (out.status.success(), combined_output(&out)),
                        Err(e) => (false, format!("Failed to run command node: {}", e)),
                    }
                }
```

- [ ] **Step 5: Update the existing PATH-restricted fallback test**

The existing `typescript_falls_back_to_node_when_bun_is_missing` assumed plain node works. Replace its assertion section (keep PATH save/set/restore) so it tolerates old node:

```rust
    #[tokio::test]
    async fn typescript_without_bun_uses_node_when_supported_else_clear_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_path = env::var_os("PATH");
        env::set_var("PATH", "/usr/bin:/bin");

        let verifier = CompilerVerifier::new(EdgeBrain::new());
        let (success, output) = verifier
            .run_local_code("console.log('ts fallback ok');", "typescript")
            .await;

        if let Some(path) = old_path {
            env::set_var("PATH", path);
        } else {
            env::remove_var("PATH");
        }

        let node_ok = std::process::Command::new("node")
            .arg("--version")
            .output()
            .map(|o| node_supports_strip_types(&String::from_utf8_lossy(&o.stdout)))
            .unwrap_or(false);

        if node_ok {
            assert!(
                success,
                "expected node strip-types success, got: {}",
                output
            );
            assert!(output.contains("ts fallback ok"));
        } else {
            assert!(!success);
            assert!(
                output.contains("requires bun"),
                "expected clear unsupported-node error, got: {}",
                output
            );
        }
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -j 2 --lib verifier`
Expected: PASS (all verifier tests).

- [ ] **Step 7: Commit**

```bash
cargo fmt
git add src/verifier.rs
git commit -m "fix(verifier): gate TS node fallback on type-stripping support, pass strip-types flag"
```

---

### Task 9: Stop recompiling the code-block regex per call

`CompilerVerifier::extract_code_block` runs `Regex::new(...).unwrap()` on every invocation (hot path: every generated snippet).

**Files:**
- Modify: `src/verifier.rs` (top of file, `extract_code_block`)

**Interfaces:**
- Produces: `fn code_block_regex() -> &'static Regex` (private). No public signature changes.

- [ ] **Step 1: Implement**

At the top of `src/verifier.rs` add to imports:

```rust
use std::sync::OnceLock;
```

Add below the imports:

```rust
static CODE_BLOCK_RE: OnceLock<Regex> = OnceLock::new();

fn code_block_regex() -> &'static Regex {
    CODE_BLOCK_RE.get_or_init(|| {
        Regex::new(r"(?s)```(?:\w+)?\n?(.*?)\n?```").expect("code-block regex must compile")
    })
}
```

In `extract_code_block`, delete the local `let re = Regex::new(...).unwrap();` line and change `re.captures(text)` to `code_block_regex().captures(text)`.

- [ ] **Step 2: Verify with existing tests**

Run: `cargo test -j 2 --lib extract_code_block`
Expected: PASS (existing tests `extract_code_block_skips_echoed_rag_context` etc. cover behavior).

- [ ] **Step 3: Sweep for other per-call compiles (report only)**

Run: `grep -rn "Regex::new" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "mod tests"`
Record any hot-path hits outside tests in the commit body as follow-up candidates. Do NOT fix them in this task (scope discipline).

- [ ] **Step 4: Commit**

```bash
cargo fmt
git add src/verifier.rs
git commit -m "perf(verifier): hoist code-block regex into OnceLock"
```

---

### Task 10: Planner prompt must advertise all verifier languages

The reasoner planner restricts plans to `'python' or 'javascript'` while `detect_default_language` and the verifier support rust/ts/cpp — planned steps can never use them.

**Files:**
- Modify: `src/orchestrator.rs` (system prompt string ~line 173; add const + test)

**Interfaces:**
- Produces: `const PLANNER_SYSTEM_PROMPT: &str` (file-private, tested).

- [ ] **Step 1: Write failing test**

Append to the test module in `src/orchestrator.rs`:

```rust
    #[test]
    fn planner_prompt_advertises_all_verifier_languages() {
        for lang in ["python", "javascript", "typescript", "rust", "cpp"] {
            assert!(
                PLANNER_SYSTEM_PROMPT.contains(lang),
                "planner prompt missing language: {}",
                lang
            );
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -j 2 --lib planner_prompt`
Expected: FAIL — const not found.

- [ ] **Step 3: Implement**

Above `impl AgentOrchestrator` add:

```rust
const PLANNER_SYSTEM_PROMPT: &str = "You are the Orchestrator Brain. Break down the user's request into the MINIMAL number of necessary executable coding steps (1 to 3 steps max).\nRespond ONLY with a valid JSON array of step objects inside a ```json ... ``` block.\nEach step object must have keys:\n- 'step': integer\n- 'description': string description of what to write\n- 'language': string ('python', 'javascript', 'typescript', 'rust', or 'cpp')";
```

In `execute_plan`, replace the inline `let system_prompt = "...";` with:

```rust
            let system_prompt = PLANNER_SYSTEM_PROMPT;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -j 2 --lib orchestrator`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/orchestrator.rs
git commit -m "fix(orchestrator): planner may emit all languages the verifier supports"
```

---

### Task 11: Configurable request timeout

300 s request timeout is hardcoded in `timeout_middleware`.

**Files:**
- Modify: `src/server/helpers.rs` (`timeout_middleware` ~line 4389; add helper + test)

**Interfaces:**
- Produces: `fn request_timeout_secs() -> u64` (private, tested); env var `MIVI_REQUEST_TIMEOUT_SECS` (default 300).

- [ ] **Step 1: Write failing test**

Append to the test module in `src/server/helpers.rs`:

```rust
#[test]
fn request_timeout_reads_env_with_safe_default() {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let _guard = LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap();

    std::env::remove_var("MIVI_REQUEST_TIMEOUT_SECS");
    assert_eq!(request_timeout_secs(), 300);

    std::env::set_var("MIVI_REQUEST_TIMEOUT_SECS", "60");
    assert_eq!(request_timeout_secs(), 60);

    std::env::set_var("MIVI_REQUEST_TIMEOUT_SECS", "0");
    assert_eq!(request_timeout_secs(), 300, "zero/negative falls back");

    std::env::remove_var("MIVI_REQUEST_TIMEOUT_SECS");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -j 2 --lib request_timeout_reads_env`
Expected: FAIL — function not found.

- [ ] **Step 3: Implement**

Add above `timeout_middleware`:

```rust
fn request_timeout_secs() -> u64 {
    std::env::var("MIVI_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(300)
}
```

In `timeout_middleware`, replace `let duration = std::time::Duration::from_secs(300);` with:

```rust
    let duration = std::time::Duration::from_secs(request_timeout_secs());
```

Also update the error message string `"Request timed out after 300 seconds."` to stay truthful:

```rust
                    "message": format!("Request timed out after {} seconds.", request_timeout_secs()),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -j 2 --lib request_timeout_reads_env`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/server/helpers.rs
git commit -m "feat(server): MIVI_REQUEST_TIMEOUT_SECS override for request timeout"
```

---

### Task 12: Docs sync (AGENTS.md + README claims)

AGENTS.md describes `src/server.rs` as a 5800-line monolith (it is now `src/server/{mod,types,helpers,handlers,tests}.rs`) and claims runtime default context is 3072 (it is 8192). New env vars need table rows.

**Files:**
- Modify: `AGENTS.md` (Repository layout section; Environment variables table; Gotchas)

- [ ] **Step 1: Update repository layout bullet**

Replace the line starting ``- `src/server.rs` — **5800+ lines**`` with:

```markdown
- `src/server/` — OpenAI-compatible API surface, split by responsibility: `types.rs` (request/response structs, AppState, RateLimiter), `helpers.rs` (agent-facing logic: tool-call parsing/validation, verified answers, streaming, auth/rate-limit/timeout middleware, `start_api_server`), `handlers.rs` (route handlers), `mod.rs` (module glue + shared template state), `tests.rs` (integration-style tests).
```

- [ ] **Step 2: Fix the context-size claim**

Find the Gotchas bullet mentioning `docs/ARCHITECTURE.md` staleness and the phrase about "the runtime default is 3072". Replace with:

```markdown
- The runtime default context budget is 8192 tokens (`DEFAULT_CONTEXT_TOKENS` in `src/constants.rs`) — the single source of truth. Historical docs/benchmarks quoting 3072 predate the change.
```

- [ ] **Step 3: Add new env vars to the table**

Append rows to the environment variables table:

```markdown
| `MIVI_HOST` | Bind address for the API server (default `127.0.0.1`; set `0.0.0.0` to expose deliberately) |
| `MIVI_PORT` | Override the default port 8000 |
| `MIVI_RATE_LIMIT_PER_MIN` | Per-client request limit (default 60) |
| `MIVI_TRUST_PROXY_HEADERS` | `1`/`true` to honor `X-Forwarded-For`/`X-Real-IP` for rate-limit identity (off by default) |
| `MIVI_REQUEST_TIMEOUT_SECS` | Whole-request timeout (default 300) |
| `MIVI_RAM_TARGET_MB` | RAM budget reported/used by low-RAM checks (default 1000) |
```

- [ ] **Step 4: Verify claims against code**

Run: `grep -n "MIVI_HOST\|MIVI_RATE_LIMIT\|MIVI_REQUEST_TIMEOUT\|MIVI_RAM_TARGET" AGENTS.md src/runtime.rs src/server/helpers.rs | head -20`
Expected: each env var appears in both AGENTS.md and its implementing file.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md
git commit -m "docs: reflect server module split, real context default, new env vars"
```

---

### Task 13: Full verification gate

**Files:** none modified (verification only).

- [ ] **Step 1: Format check**

Run: `cargo fmt --check`
Expected: exit 0, no output.

- [ ] **Step 2: Full Rust test suite (once, resource-capped)**

Run: `cargo test -j 2 -- --test-threads=2`
Expected: all tests pass. If a failure appears, fix it in a dedicated `fix:` commit before proceeding — do not skip.

- [ ] **Step 3: Python unit tests (CI parity)**

Run (from `scripts/`):
```bash
cd scripts && python3 -m unittest test_check_agent_compat.py test_smoke_openai_compat.py test_eval_agent_workflows.py test_score_eval.py test_eval_tool_calling.py test_prepare_mivi_dataset.py
```
Expected: OK. (These don't exercise the changed Rust code but CI runs them; catches accidental breakage of script-facing behavior.)

- [ ] **Step 4: Final commit-log sanity**

Run: `git log --oneline -12`
Expected: ~12 commits, one per task, conventional-commit prefixes (`fix:`, `feat:`, `perf:`, `refactor:`, `docs:`).

---

## Out-of-Scope Register (explicitly deferred)

| Item | Why deferred |
|---|---|
| Sandbox the verifier subprocess (namespaces/containers/seccomp) | Needs its own design + platform matrix; Task 1's loopback default removes the remote trigger path meanwhile |
| Replace `is_complex` substring heuristic | Routing-behavior change; needs eval harness run to avoid regressions |
| Remove `repair_python_code` sum-hack | Harmless narrow special case; removal reduces capability with no safety gain |
| `-ngl 999` literal | llama.cpp convention for "all layers"; CPU-only hosts degrade gracefully |
| Other per-call `Regex::new` sites (tool_output.rs, rag.rs) | Listed in Task 9 Step 3 sweep; mechanical follow-up once confirmed hot |
