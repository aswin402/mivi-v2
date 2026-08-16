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

        let mode = env::var("MIVI_RUNTIME_MODE")
            .map(|value| RuntimeMode::from_env_value(&value))
            .unwrap_or(default_mode);

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

        Self {
            mode,
            context: ContextBudget::from_max_input_tokens(max_input_tokens),
            worker_idle_secs,
            ram_target_mb: DEFAULT_RAM_TARGET_MB,
            kv_cache_type,
            threads,
            draft_model,
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
    fn worker_hot_mode_is_supported() {
        let _guard = env_lock();
        clear_runtime_env();
        std::env::set_var("MIVI_RUNTIME_MODE", "worker-hot");

        let config = RuntimeConfig::from_env();

        assert_eq!(config.mode, RuntimeMode::WorkerHot);

        clear_runtime_env();
    }
}
