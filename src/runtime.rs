use std::env;

const DEFAULT_CONTEXT_TOKENS: usize = 4096;
const MIN_CONTEXT_TOKENS: usize = 1024;
const DEFAULT_WORKER_IDLE_SECS: u64 = 120;
const DEFAULT_RAM_TARGET_MB: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Spawn,
    WorkerEco,
    WorkerHot,
}

impl RuntimeMode {
    fn from_env_value(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "worker-eco" | "worker_eco" | "eco" => Self::WorkerEco,
            "worker-hot" | "worker_hot" | "hot" => Self::WorkerHot,
            "spawn" => Self::Spawn,
            _ => Self::Spawn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_input_tokens: usize,
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
            recent_turn_tokens: max_input_tokens * 3 / 8,
            retrieved_tokens: max_input_tokens * 3 / 8,
            memory_tokens: max_input_tokens * 3 / 16,
            tool_tokens: max_input_tokens / 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub mode: RuntimeMode,
    pub context: ContextBudget,
    pub worker_idle_secs: u64,
    pub ram_target_mb: usize,
}

impl RuntimeConfig {
    pub fn uses_worker(&self) -> bool {
        !matches!(self.mode, RuntimeMode::Spawn)
    }

    pub fn from_env() -> Self {
        let mode = env::var("MIVI_RUNTIME_MODE")
            .map(|value| RuntimeMode::from_env_value(&value))
            .unwrap_or(RuntimeMode::Spawn);

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

        Self {
            mode,
            context: ContextBudget::from_max_input_tokens(max_input_tokens),
            worker_idle_secs,
            ram_target_mb: DEFAULT_RAM_TARGET_MB,
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
    }

    #[test]
    fn default_config_uses_spawn_mode_and_low_resource_budget() {
        let _guard = env_lock();
        clear_runtime_env();

        let config = RuntimeConfig::from_env();

        assert_eq!(config.mode, RuntimeMode::Spawn);
        assert_eq!(config.worker_idle_secs, 120);
        assert_eq!(config.ram_target_mb, 1000);
        assert_eq!(config.context.max_input_tokens, 4096);
        assert_eq!(config.context.recent_turn_tokens, 1536);
        assert_eq!(config.context.retrieved_tokens, 1536);
        assert_eq!(config.context.memory_tokens, 768);
        assert_eq!(config.context.tool_tokens, 256);
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
        assert_eq!(config.context.recent_turn_tokens, 3072);
        assert_eq!(config.context.retrieved_tokens, 3072);
        assert_eq!(config.context.memory_tokens, 1536);
        assert_eq!(config.context.tool_tokens, 512);

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

        assert_eq!(config.mode, RuntimeMode::Spawn);
        assert_eq!(config.worker_idle_secs, 120);
        assert_eq!(config.context.max_input_tokens, 4096);

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
