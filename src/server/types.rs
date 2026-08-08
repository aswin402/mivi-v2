use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use super::helpers::{count_with_llama_cpp_tokenizer, default_tool_type, TokenCounter};
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

#[derive(Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallIn>>,
}

#[derive(Deserialize)]
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
        match &self.backend {
            TokenCounterBackend::Cheap => self.fallback.count_tokens(text),
            TokenCounterBackend::LlamaCpp { command, model } => {
                count_with_llama_cpp_tokenizer(command, model, text)
                    .unwrap_or_else(|| self.fallback.count_tokens(text))
            }
        }
    }
}

pub struct CheapTokenCounter;

impl TokenCounter for CheapTokenCounter {
    fn count_tokens(&self, text: &str) -> u32 {
        let mut tokens = 0_u32;
        let mut in_word = false;
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                if !in_word {
                    tokens = tokens.saturating_add(1);
                    in_word = true;
                }
            } else {
                in_word = false;
                if !ch.is_whitespace() {
                    tokens = tokens.saturating_add(1);
                }
            }
        }
        tokens
    }
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
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
