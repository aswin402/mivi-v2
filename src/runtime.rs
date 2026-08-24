use std::env;

use crate::constants::{
    DEFAULT_CONTEXT_TOKENS, DEFAULT_RAM_TARGET_MB, DEFAULT_WORKER_IDLE_SECS, MIN_CONTEXT_TOKENS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Spawn,
    WorkerEco,
    WorkerHot,
    Native,
}

impl RuntimeMode {
    fn from_env_value(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "worker-eco" | "worker_eco" | "eco" => Self::WorkerEco,
            "worker-hot" | "worker_hot" | "hot" => Self::WorkerHot,
            "spawn" => Self::Spawn,
            "native" | "candle" => Self::Native,
            _ => {
                #[cfg(feature = "native")]
                return Self::Native;
                #[cfg(not(feature = "native"))]
                Self::WorkerEco
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_input_tokens: usize,
    pub system_tokens: usize,
    pub anchor_tokens: usize,
    pub summary_tokens: usize,
    pub recent_turn_tokens: usize,
    pub retrieved_tokens: usize,
    pub memory_tokens: usize,
    pub tool_tokens: usize,
}

impl ContextBudget {
    pub fn from_max_input_tokens(max_input_tokens: usize) -> Self {
        let max_input_tokens = if max_input_tokens < MIN_CONTEXT_TOKENS {
            DEFAULT_CONTEXT_TOKENS
        } else {
            max_input_tokens
        };

        Self {
            max_input_tokens,
            system_tokens: max_input_tokens * 20 / 100, // 20%
            anchor_tokens: max_input_tokens * 5 / 100,  // 5%
            summary_tokens: max_input_tokens * 15 / 100, // 15%
            recent_turn_tokens: max_input_tokens * 35 / 100, // 35%
            retrieved_tokens: max_input_tokens * 5 / 100, // 5% (RAG part 1)
            memory_tokens: max_input_tokens * 5 / 100,  // 5% (RAG part 2)
            tool_tokens: max_input_tokens * 15 / 100,   // 15% (Summary/Tool)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub mode: RuntimeMode,
    pub context: ContextBudget,
    pub worker_idle_secs: u64,
    pub ram_target_mb: usize,
    pub kv_cache_type: String,
    pub threads: usize,
    pub draft_model: Option<String>,
    /// Flattened llama.cpp `--lora-scaled` args from `MIVI_LORA_ADAPTERS`
    /// (Phase 17.1). Empty unless adapters are configured and exist on disk.
    pub lora_args: Vec<String>,
}

/// Parse `MIVI_LORA_ADAPTERS`: comma-separated `path[=scale]` entries,
/// scale defaulting to 1.0 (invalid scales fall back to 1.0).
pub fn parse_lora_adapters(spec: &str) -> Vec<(String, f32)> {
    spec.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| match entry.split_once('=') {
            Some((path, scale)) => (
                path.trim().to_string(),
                scale.trim().parse::<f32>().unwrap_or(1.0),
            ),
            None => (entry.to_string(), 1.0),
        })
        .collect()
}

/// Flatten parsed adapters into llama.cpp `--lora-scaled FNAME:SCALE,...`
/// (a single flag with colon-separated pairs joined by commas), skipping
/// paths that do not exist (warn so typos are visible).
pub fn lora_flag_args(adapters: &[(String, f32)]) -> Vec<String> {
    let pairs: Vec<String> = adapters
        .iter()
        .filter(|(path, _)| {
            let exists = std::path::Path::new(path).exists();
            if !exists {
                tracing::warn!("[runtime] LoRA adapter not found, skipping: {path}");
            }
            exists
        })
        .map(|(path, scale)| format!("{path}:{scale}"))
        .collect();
    if pairs.is_empty() {
        Vec::new()
    } else {
        vec!["--lora-scaled".to_string(), pairs.join(",")]
    }
}

impl RuntimeConfig {
    pub fn uses_worker(&self) -> bool {
        matches!(self.mode, RuntimeMode::WorkerEco | RuntimeMode::WorkerHot)
    }

    /// Process-wide cached config. Env vars are read exactly once; per-request
    /// callers should use this instead of `from_env()` (which remains available
    /// for tests that mutate the environment).
    pub fn global() -> &'static Self {
        static GLOBAL: std::sync::OnceLock<RuntimeConfig> = std::sync::OnceLock::new();
        GLOBAL.get_or_init(Self::from_env)
    }

    pub fn from_env() -> Self {
        // Default to Native mode: use our own candle-based inference engine.
        // No external llama-cli or llama-server dependencies needed.
        #[cfg(feature = "native")]
        let default_mode = RuntimeMode::Native;
        #[cfg(not(feature = "native"))]
        let default_mode = RuntimeMode::WorkerEco;

        let mode = match env::var("MIVI_RUNTIME_MODE") {
            // `auto` defers to the doctor's RAM-tiered recommendation so the
            // runtime shape always fits the machine it runs on.
            Ok(value) if value.trim().eq_ignore_ascii_case("auto") => {
                let snapshot = crate::doctor::read_system_snapshot();
                let plan = crate::doctor::recommend(&snapshot);
                tracing::info!(
                    "[runtime] MIVI_RUNTIME_MODE=auto resolved to {} \
                     ({} MB RAM available, ultra_low_ram={})",
                    plan.runtime_mode,
                    snapshot.available_ram_mb,
                    plan.ultra_low_ram
                );
                RuntimeMode::from_env_value(plan.runtime_mode)
            }
            Ok(value) => RuntimeMode::from_env_value(&value),
            Err(_) => default_mode,
        };

        let max_input_tokens = env::var("MIVI_CONTEXT_BUDGET")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|tokens| *tokens >= MIN_CONTEXT_TOKENS)
            .unwrap_or(DEFAULT_CONTEXT_TOKENS);

        let worker_idle_secs = env::var("MIVI_WORKER_IDLE_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|secs| *secs > 0)
            .unwrap_or(DEFAULT_WORKER_IDLE_SECS);

        let ultra_low = env::var("MIVI_ULTRA_LOW_RAM")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);

        let ram_target_mb = env::var("MIVI_RAM_TARGET_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|mb| *mb > 0)
            .unwrap_or(DEFAULT_RAM_TARGET_MB);

        let kv_cache_type = env::var("MIVI_KV_CACHE_TYPE").unwrap_or_else(|_| {
            if ultra_low {
                "q4_0".to_string()
            } else {
                "q8_0".to_string()
            }
        });

        let threads = env::var("MIVI_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&t| t > 0)
            .unwrap_or_else(|| {
                let logical = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let base = if logical > 1 { logical / 2 } else { 1 };
                if env::var("MIVI_ULTRA_LOW_RAM").is_ok() {
                    if base > 2 {
                        base / 2
                    } else {
                        1
                    }
                } else {
                    base
                }
            });

        let draft_model = env::var("MIVI_DRAFT_MODEL")
            .ok()
            .filter(|v| !v.trim().is_empty());

        let lora_args = env::var("MIVI_LORA_ADAPTERS")
            .map(|spec| lora_flag_args(&parse_lora_adapters(&spec)))
            .unwrap_or_default();

        Self {
            mode,
            context: ContextBudget::from_max_input_tokens(max_input_tokens),
            worker_idle_secs,
            ram_target_mb,
            kv_cache_type,
            threads,
            draft_model,
            lora_args,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn clear_runtime_env() {
        std::env::remove_var("MIVI_RUNTIME_MODE");
        std::env::remove_var("MIVI_CONTEXT_BUDGET");
        std::env::remove_var("MIVI_WORKER_IDLE_SECS");
        std::env::remove_var("MIVI_THREADS");
        std::env::remove_var("MIVI_DRAFT_MODEL");
        std::env::remove_var("MIVI_RAM_TARGET_MB");
    }

    #[test]
    fn default_config_uses_native_or_worker_eco_mode() {
        let _guard = env_lock();
        clear_runtime_env();

        let config = RuntimeConfig::from_env();

        #[cfg(feature = "native")]
        assert_eq!(config.mode, RuntimeMode::Native);
        #[cfg(not(feature = "native"))]
        assert_eq!(config.mode, RuntimeMode::WorkerEco);
        assert_eq!(config.worker_idle_secs, 120);
        assert_eq!(config.ram_target_mb, 1000);
        assert_eq!(config.context.max_input_tokens, 8192);
        assert_eq!(config.context.recent_turn_tokens, 2867);
        assert_eq!(config.context.retrieved_tokens, 409);
        assert_eq!(config.context.memory_tokens, 409);
        assert_eq!(config.context.tool_tokens, 1228);
    }

    #[test]
    fn env_overrides_runtime_mode_context_budget_and_idle_timeout() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RUNTIME_MODE", "worker-eco");
        std::env::set_var("MIVI_CONTEXT_BUDGET", "8192");
        std::env::set_var("MIVI_WORKER_IDLE_SECS", "45");

        let config = RuntimeConfig::from_env();

        assert_eq!(config.mode, RuntimeMode::WorkerEco);
        assert_eq!(config.worker_idle_secs, 45);
        assert_eq!(config.context.max_input_tokens, 8192);
        assert_eq!(config.context.recent_turn_tokens, 2867);
        assert_eq!(config.context.retrieved_tokens, 409);
        assert_eq!(config.context.memory_tokens, 409);
        assert_eq!(config.context.tool_tokens, 1228);

        clear_runtime_env();
    }

    #[test]
    fn invalid_env_values_fall_back_to_safe_defaults() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RUNTIME_MODE", "always-load-everything");
        std::env::set_var("MIVI_CONTEXT_BUDGET", "128");
        std::env::set_var("MIVI_WORKER_IDLE_SECS", "0");

        let config = RuntimeConfig::from_env();

        #[cfg(feature = "native")]
        assert_eq!(config.mode, RuntimeMode::Native);
        #[cfg(not(feature = "native"))]
        assert_eq!(config.mode, RuntimeMode::WorkerEco);
        assert_eq!(config.worker_idle_secs, 120);
        assert_eq!(config.context.max_input_tokens, 8192);

        clear_runtime_env();
    }
    #[test]
    fn auto_mode_resolves_to_the_doctor_recommendation_for_this_machine() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RUNTIME_MODE", "auto");

        let config = RuntimeConfig::from_env();

        let plan = crate::doctor::recommend(&crate::doctor::read_system_snapshot());
        let expected = RuntimeMode::from_env_value(plan.runtime_mode);
        assert_eq!(config.mode, expected);

        clear_runtime_env();
    }

    #[test]
    fn worker_hot_mode_is_supported() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RUNTIME_MODE", "worker-hot");

        let config = RuntimeConfig::from_env();

        assert_eq!(config.mode, RuntimeMode::WorkerHot);

        clear_runtime_env();
    }

    #[test]
    fn ram_target_mb_is_configurable_via_env() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RAM_TARGET_MB", "1500");

        let config = RuntimeConfig::from_env();
        assert_eq!(config.ram_target_mb, 1500);

        std::env::set_var("MIVI_RAM_TARGET_MB", "not-a-number");
        let config = RuntimeConfig::from_env();
        assert_eq!(
            config.ram_target_mb,
            crate::constants::DEFAULT_RAM_TARGET_MB
        );

        std::env::remove_var("MIVI_RAM_TARGET_MB");
    }

    #[test]
    fn parse_lora_adapters_handles_paths_scales_and_garbage() {
        assert!(parse_lora_adapters("").is_empty());
        assert_eq!(
            parse_lora_adapters("a.gguf, b.gguf=0.5"),
            vec![("a.gguf".to_string(), 1.0), ("b.gguf".to_string(), 0.5)]
        );
        // Invalid scale falls back to 1.0 rather than failing the parse.
        assert_eq!(
            parse_lora_adapters("x.gguf=notanumber"),
            vec![("x.gguf".to_string(), 1.0)]
        );
    }

    #[test]
    fn lora_flag_args_skip_missing_files() {
        let adapters = vec![
            ("definitely-missing-adapter.gguf".to_string(), 1.0),
            ("Cargo.toml".to_string(), 0.75),
        ];
        assert_eq!(
            lora_flag_args(&adapters),
            vec!["--lora-scaled".to_string(), "Cargo.toml:0.75".to_string()]
        );
        assert!(lora_flag_args(&[]).is_empty());
    }

    #[test]
    fn lora_flag_args_join_multiple_adapters_into_one_flag() {
        let adapters = vec![
            ("Cargo.toml".to_string(), 1.0),
            ("README.md".to_string(), 0.5),
        ];
        assert_eq!(
            lora_flag_args(&adapters),
            vec![
                "--lora-scaled".to_string(),
                "Cargo.toml:1,README.md:0.5".to_string()
            ]
        );
    }

    #[test]
    fn lora_env_var_flows_into_from_env() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_LORA_ADAPTERS", "Cargo.toml=0.5,missing.gguf");

        let config = RuntimeConfig::from_env();

        assert_eq!(
            config.lora_args,
            vec!["--lora-scaled".to_string(), "Cargo.toml:0.5".to_string()]
        );

        clear_runtime_env();
    }
}
