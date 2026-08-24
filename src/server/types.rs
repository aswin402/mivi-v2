use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::helpers::{default_tool_type, TokenCounter};
use crate::brain::EdgeBrain;
use crate::model_catalog::{ModelCatalog, ModelRole};
use crate::orchestrator::AgentOrchestrator;
use crate::router::NeedleRouter;

/// Function definition sent by the client (in tools[]).
#[derive(Serialize, Deserialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

/// A single tool definition from the request.
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolDef {
    pub function: FunctionDef,
    #[serde(default = "default_tool_type")]
    pub r#type: String,
}

/// A tool call inside an incoming assistant message (for multi-turn).
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCallIn {
    pub id: String,
    #[serde(default = "default_tool_type")]
    pub r#type: String,
    pub function: FunctionCallIn,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FunctionCallIn {
    pub name: String,
    pub arguments: String, // JSON string
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallIn>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub service_tier: Option<String>,
}

#[derive(Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: ResponsesInput,
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, alias = "max_tokens", alias = "max_completion_tokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Messages(Vec<ResponsesInputMessage>),
}

#[derive(Deserialize)]
pub struct ResponsesInputMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponsesOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

#[derive(Serialize)]
pub struct ResponsesOutputItem {
    pub id: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ResponsesOutputContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
}

#[derive(Serialize)]
pub struct ResponsesOutputContent {
    pub r#type: String,
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Serialize, Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

#[derive(Serialize, Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Serialize, Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

impl UsageInfo {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenCounterBackend {
    Cheap,
    LlamaCpp { command: PathBuf, model: PathBuf },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenCounterConfig {
    pub backend: TokenCounterBackend,
}

impl Default for TokenCounterConfig {
    fn default() -> Self {
        Self {
            backend: TokenCounterBackend::Cheap,
        }
    }
}

impl TokenCounterConfig {
    pub fn from_sources(
        command: Option<&str>,
        tokenizer_model: Option<&str>,
        reasoner_model: Option<&str>,
        catalog: Option<&ModelCatalog>,
    ) -> Self {
        let command = command.map(str::trim).filter(|value| !value.is_empty());
        let model = tokenizer_model
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                reasoner_model
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            })
            .or_else(|| {
                catalog.and_then(|catalog| catalog.default_enabled_path(ModelRole::Reasoner))
            });

        match (command, model) {
            (Some(command), Some(model)) => Self {
                backend: TokenCounterBackend::LlamaCpp {
                    command: PathBuf::from(command),
                    model: PathBuf::from(model),
                },
            },
            _ => Self::default(),
        }
    }

    pub fn from_env() -> Self {
        let command = std::env::var("MIVI_TOKENIZER_CMD").ok();
        let tokenizer_model = std::env::var("MIVI_TOKENIZER_MODEL").ok();
        let reasoner_model = std::env::var("MIVI_REASONER_MODEL").ok();
        let catalog = ModelCatalog::load_default().ok();
        Self::from_sources(
            command.as_deref(),
            tokenizer_model.as_deref(),
            reasoner_model.as_deref(),
            catalog.as_ref(),
        )
    }

    pub fn counter(&self) -> RuntimeTokenCounter {
        RuntimeTokenCounter {
            backend: self.backend.clone(),
            fallback: CheapTokenCounter,
        }
    }
}

pub struct RuntimeTokenCounter {
    pub backend: TokenCounterBackend,
    pub fallback: CheapTokenCounter,
}

impl TokenCounter for RuntimeTokenCounter {
    fn count_tokens(&self, text: &str) -> u32 {
        crate::tokenizer::count_tokens(text)
    }
}

pub struct CheapTokenCounter;

impl TokenCounter for CheapTokenCounter {
    fn count_tokens(&self, text: &str) -> u32 {
        crate::tokenizer::count_tokens(text)
    }
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
}

#[derive(Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

/// Input for `/v1/embeddings`: a single string or a batch of strings.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_texts(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(text) => vec![text],
            EmbeddingInput::Multiple(texts) => texts,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: Option<String>,
    /// Only `float` is supported; `base64` is rejected with a clear error.
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Serialize, Debug)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize, Debug)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

/// A tool call in the assistant's response.
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCallOut {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCallOut,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FunctionCallOut {
    pub name: String,
    pub arguments: String, // always valid JSON string
}

#[derive(Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
    pub refusal: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize)]
pub struct ChoiceOut {
    pub index: usize,
    pub message: ChatMessageOut,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
    pub choices: Vec<ChoiceOut>,
    pub system_fingerprint: Option<String>,
}

pub struct AppState {
    pub brain: EdgeBrain,
    pub orchestrator: AgentOrchestrator,
    pub router: NeedleRouter,
    pub semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    pub rate_limiter: RateLimiter,
}

#[derive(Clone)]
pub struct RateLimiter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl RateLimiter {
    /// Hard ceiling on simultaneously tracked identities so header-spoofing
    /// floods cannot grow the map without bound.
    pub const MAX_TRACKED_CLIENTS: usize = 4096;

    pub fn new() -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentIntent {
    Chat,
    ToolInventory,
    SkillInventory,
    McpInventory,
    CapabilityInventory,
    ToolCall,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolRole {
    Inventory,
    SkillInventory,
    McpInventory,
    McpResource,
    Diagnostic,
    Action,
    General,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolBlock {
    pub name: String,
    pub reason: &'static str,
}

#[derive(Clone)]
pub struct ToolSelection {
    pub intent: AgentIntent,
    pub selected: Vec<ToolDef>,
    pub blocked: Vec<ToolBlock>,
}

impl ToolSelection {
    pub fn empty(intent: AgentIntent) -> Self {
        Self {
            intent,
            selected: Vec::new(),
            blocked: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentDecisionIntent {
    Chat,
    WebResearch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentDecision {
    pub intent: AgentDecisionIntent,
    pub subject: String,
    pub url: Option<String>,
}

impl AgentDecision {
    pub fn chat(subject: String) -> Self {
        Self {
            intent: AgentDecisionIntent::Chat,
            subject,
            url: None,
        }
    }

    pub fn needs_tool(&self) -> bool {
        matches!(self.intent, AgentDecisionIntent::WebResearch)
    }
}

impl AgentIntent {
    pub fn is_inventory(self) -> bool {
        matches!(
            self,
            AgentIntent::ToolInventory
                | AgentIntent::SkillInventory
                | AgentIntent::McpInventory
                | AgentIntent::CapabilityInventory
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AgentIntent::Chat => "chat",
            AgentIntent::ToolInventory => "tool_inventory",
            AgentIntent::SkillInventory => "skill_inventory",
            AgentIntent::McpInventory => "mcp_inventory",
            AgentIntent::CapabilityInventory => "capability_inventory",
            AgentIntent::ToolCall => "tool_call",
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CapabilityConfig {
    #[serde(default)]
    pub aliases: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub tool_taxonomy: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub tool_error_markers: Vec<String>,
    #[serde(default)]
    pub tool_salient_markers: Vec<String>,
    #[serde(default)]
    pub tool_error_categories: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub tool_error_category_priority: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AnthropicRequest {
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: Option<u32>,
    pub system: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<serde_json::Value>>,
}

pub fn anthropic_request_to_chat_request(req: AnthropicRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // Extract system prompt if present
    if let Some(sys) = req.system {
        let sys_text = if let Some(s) = sys.as_str() {
            s.to_string()
        } else if let Some(arr) = sys.as_array() {
            arr.iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            sys.to_string()
        };
        if !sys_text.is_empty() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: serde_json::Value::String(sys_text),
                tool_call_id: None,
                tool_calls: None,
            });
        }
    }

    // Mirror the chat-completions path: when the caller supplies no system
    // prompt, inject the default MIVI identity prompt so agents using the
    // Anthropic surface get the same "external model is mivi" behavior.
    if !messages.iter().any(|m| m.role == "system") {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: serde_json::Value::String(
                    crate::constants::MIVI_CHAT_SYSTEM_PROMPT.to_string(),
                ),
                tool_call_id: None,
                tool_calls: None,
            },
        );
    }

    for msg in req.messages {
        let content_val = if let Some(text) = msg.content.as_str() {
            serde_json::Value::String(text.to_string())
        } else if let Some(arr) = msg.content.as_array() {
            let texts: Vec<String> = arr
                .iter()
                .filter_map(|b| {
                    if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                        Some(t.to_string())
                    } else if b.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                        b.get("content").map(|c| c.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            serde_json::Value::String(texts.join("\n"))
        } else {
            msg.content.clone()
        };

        messages.push(ChatMessage {
            role: msg.role,
            content: content_val,
            tool_call_id: None,
            tool_calls: None,
        });
    }

    let tools = req.tools.map(|raw_tools| {
        raw_tools
            .into_iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?.to_string();
                let description = t
                    .get("description")
                    .and_then(|d| d.as_str())
                    .map(|d| d.to_string());
                let parameters = t.get("input_schema").cloned();
                Some(ToolDef {
                    function: FunctionDef {
                        name,
                        description,
                        parameters,
                    },
                    r#type: "function".to_string(),
                })
            })
            .collect()
    });

    ChatCompletionRequest {
        model: req.model.or_else(|| Some("mivi".to_string())),
        messages,
        stream: req.stream,
        tools,
        tool_choice: None,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        response_format: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        n: None,
        seed: None,
        service_tier: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        user: None,
        parallel_tool_calls: None,
        reasoning_effort: None,
        stream_options: None,
    }
}

#[cfg(test)]
pub(crate) fn rate_limit_env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_chat_completion_request_deserialization() {
        let data = json!({
            "model": "mivi",
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "logit_bias": {"50256": -100.0},
            "logprobs": true,
            "top_logprobs": 5,
            "n": 1,
            "service_tier": "auto"
        });

        let req: ChatCompletionRequest = serde_json::from_value(data).unwrap();
        assert_eq!(req.model.as_deref(), Some("mivi"));
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, json!("hello"));
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
        assert_eq!(req.n, Some(1));
        assert_eq!(req.service_tier.as_deref(), Some("auto"));
    }

    #[test]
    fn test_anthropic_request_mapping() {
        let anthropic_data = json!({
            "model": "claude-3-5-sonnet",
            "system": "You are a helpful coding assistant.",
            "messages": [
                {"role": "user", "content": "Write a Rust hello world"}
            ],
            "max_tokens": 1024,
            "temperature": 0.2
        });

        let anthropic_req: AnthropicRequest = serde_json::from_value(anthropic_data).unwrap();
        let chat_req = anthropic_request_to_chat_request(anthropic_req);

        assert_eq!(chat_req.messages.len(), 2);
        assert_eq!(chat_req.messages[0].role, "system");
        assert_eq!(
            chat_req.messages[0].content,
            json!("You are a helpful coding assistant.")
        );
        assert_eq!(chat_req.messages[1].role, "user");
        assert_eq!(
            chat_req.messages[1].content,
            json!("Write a Rust hello world")
        );
        assert_eq!(chat_req.max_tokens, Some(1024));
        assert_eq!(chat_req.temperature, Some(0.2));
    }

    #[test]
    fn rate_limiter_blocks_after_configured_limit() {
        let _guard = rate_limit_env_lock().lock().unwrap();
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
