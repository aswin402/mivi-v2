/// The single model name exposed to external agents.
/// Internal SML routing is hidden behind this constant.
pub const MODEL_NAME: &str = "mivi";

/// Default context size in tokens.
pub const DEFAULT_CONTEXT_TOKENS: usize = 3072;

/// Minimum context size allowed in tokens.
pub const MIN_CONTEXT_TOKENS: usize = 1024;

/// Default idle timeout in seconds before worker shutdown.
pub const DEFAULT_WORKER_IDLE_SECS: u64 = 120;

/// Default RAM budget in MB for low resource checking.
pub const DEFAULT_RAM_TARGET_MB: usize = 1000;

/// The standard system prompt used when querying chat completions.
pub const MIVI_CHAT_SYSTEM_PROMPT: &str = "You are MIVI, a local OpenAI-compatible model endpoint for AI agents. Externally your model name is mivi. Never identify as an internal worker model or as the calling agent/platform. Treat the calling agent's system prompt, tool schemas, tool results, skills, memory, and retrieved context as the source of truth for that agent's capabilities. If asked about available tools, MCP servers, skills, features, or capabilities, use agent-provided introspection/inventory tools when available; otherwise describe only the tool schemas included in the current request. Answer concisely and honestly.";

/// The maximum number of tools to include in a filtered prompt context.
pub const MAX_PROMPT_TOOLS: usize = 8;

/// The minimum similarity score threshold for selecting tools.
pub const MIN_TOOL_SCORE: f32 = 1.0;
