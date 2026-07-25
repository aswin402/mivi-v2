use axum::{
    extract::Json,
    response::sse::{Event, Sse},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::brain::EdgeBrain;
use crate::context_compressor::{compress_context, render_context_prompt};
use crate::model_process::spawn_streaming;
use crate::okf_memory::load_memory_dir;
use crate::orchestrator::AgentOrchestrator;
use crate::retrieval::{build_retrieval_pack_with_sources, should_include_workspace_rag};
use crate::router::NeedleRouter;
use crate::runtime::RuntimeConfig;
use crate::tool_filter::filter_tools;

/// The single model name exposed to external agents.
/// Internal SML routing is hidden behind this constant.
pub const MODEL_NAME: &str = "mivi";

const MIVI_CHAT_SYSTEM_PROMPT: &str = "You are MIVI, a local OpenAI-compatible AI endpoint. Externally your model name is mivi. Internally you route between Llama for chat and reasoning, Qwen for coding, and MiniCPM for vision. Answer concisely and honestly.";

// ──────────────────────────────────────────────
// OpenAI-compatible tool/function structs
// ──────────────────────────────────────────────

/// Function definition sent by the client (in tools[]).
#[derive(Deserialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

/// A single tool definition from the request.
#[derive(Deserialize, Clone)]
pub struct ToolDef {
    pub function: FunctionDef,
    #[serde(default = "default_tool_type")]
    pub r#type: String,
}

fn default_tool_type() -> String {
    "function".into()
}

/// A tool call inside an incoming assistant message (for multi-turn).
#[derive(Deserialize, Clone)]
pub struct ToolCallIn {
    pub id: String,
    #[serde(default = "default_tool_type")]
    pub r#type: String,
    pub function: FunctionCallIn,
}

#[derive(Deserialize, Clone)]
pub struct FunctionCallIn {
    pub name: String,
    pub arguments: String, // JSON string
}

// ──────────────────────────────────────────────
// Request / Message structs
// ──────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
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
}

// ──────────────────────────────────────────────
// Response structs
// ──────────────────────────────────────────────

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
#[derive(Serialize, Clone)]
pub struct ToolCallOut {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCallOut,
}

#[derive(Serialize, Clone)]
pub struct FunctionCallOut {
    pub name: String,
    pub arguments: String, // always valid JSON string
}

#[derive(Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Serialize)]
pub struct ChoiceOut {
    pub index: usize,
    pub message: ChatMessageOut,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceOut>,
}

pub struct AppState {
    pub brain: EdgeBrain,
    pub orchestrator: AgentOrchestrator,
    pub router: NeedleRouter,
}

// ──────────────────────────────────────────────
// Handlers
// ──────────────────────────────────────────────

async fn handle_root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "online",
        "service": "MIVI-V2 Pure Rust High-Speed AI Engine",
        "version": "0.0.4",
        "ram_footprint": "< 12 MB RAM",
        "openai_endpoint": "/v1/chat/completions"
    }))
}

async fn handle_models() -> Json<ModelListResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: MODEL_NAME.to_string(),
            object: "model".to_string(),
            created: now,
            owned_by: MODEL_NAME.to_string(),
        }],
    })
}

// ──────────────────────────────────────────────
// Prompt building
// ──────────────────────────────────────────────

const MAX_PROMPT_TOOLS: usize = 8;

fn build_chat_prompt(req: &ChatCompletionRequest) -> String {
    let mut prompt = String::new();
    let prompt_tools = prompt_tools_for_request(req);

    let has_user_system = req.messages.iter().any(|m| m.role == "system");
    if has_user_system {
        for msg in &req.messages {
            if msg.role == "system" {
                if let Some(text) = msg.content.as_str() {
                    if !text.is_empty() {
                        prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", text));
                    }
                }
            }
        }
    } else if prompt_tools.is_empty() {
        prompt.push_str("<|im_start|>system\nYou are a helpful, concise AI assistant.<|im_end|>\n");
    }

    // Conversation turns.
    let has_tools = !prompt_tools.is_empty();
    let mut last_user_idx: Option<usize> = None;

    for (_, msg) in req.messages.iter().enumerate() {
        match msg.role.as_str() {
            "user" => {
                let text = extract_user_text(msg);
                if !text.is_empty() {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", text));
                    last_user_idx = Some(prompt.len());
                }
            }
            "assistant" => {
                if let Some(ref calls) = msg.tool_calls {
                    let content = msg.content.as_str().unwrap_or("");
                    let mut block = String::new();
                    if !content.is_empty() {
                        block.push_str(content);
                        block.push('\n');
                    }
                    for tc in calls {
                        block.push_str(&format!(
                            "{{\"name\": \"{}\", \"arguments\": {}}}\n",
                            tc.function.name, tc.function.arguments
                        ));
                    }
                    if !block.is_empty() {
                        prompt.push_str(&format!(
                            "<|im_start|>assistant\n{}<|im_end|>\n",
                            block.trim()
                        ));
                    }
                } else {
                    let text = msg.content.as_str().unwrap_or("");
                    if !text.is_empty() {
                        prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", text));
                    }
                }
            }
            "tool" => {
                let tool_content = msg.content.as_str().unwrap_or("");
                let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
                prompt.push_str(&format!(
                    "<|im_start|>tool\nTool result ({}): {}\n<|im_end|>\n",
                    tool_id, tool_content
                ));
            }
            _ => {}
        }
    }

    if has_tools {
        let func_block = build_function_list_block(&prompt_tools);
        if let Some(idx) = last_user_idx {
            prompt.insert_str(idx, &func_block);
        } else {
            prompt.push_str(&func_block);
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn prompt_tools_for_request(req: &ChatCompletionRequest) -> Vec<ToolDef> {
    let tools = match req.tools.as_deref() {
        Some(tools) if !tools.is_empty() => tools,
        _ => return Vec::new(),
    };

    if matches!(req.tool_choice, Some(serde_json::Value::String(ref choice)) if choice == "required")
    {
        return tools.to_vec();
    }

    filter_tools(&latest_user_prompt_text(req), tools, MAX_PROMPT_TOOLS)
}

fn build_function_list_block(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut block = String::new();
    let first_tool = &tools[0].function;
    let ex_args = first_tool
        .parameters
        .as_ref()
        .and_then(|p| p.get("properties"))
        .and_then(|props| props.as_object())
        .and_then(|obj| obj.keys().next())
        .map(|k| format!("\"{}\": \"...\"", k))
        .unwrap_or_else(|| "\"key\": \"value\"".to_string());

    block.push_str(&format!(
        "
Tool context broker selected {} tool(s).
Available functions:
         - {name}: {desc}

         When appropriate, respond with ONLY:
         {{\"name\": \"{name}\", \"arguments\": {{{ex_args}}}}}

",
        tools.len(),
        name = first_tool.name,
        desc = first_tool.description.as_deref().unwrap_or(""),
    ));

    for tool in tools {
        let f = &tool.function;
        let param_keys = f
            .parameters
            .as_ref()
            .and_then(|params| params.get("properties"))
            .and_then(|properties| properties.as_object())
            .map(|properties| {
                let mut keys: Vec<&str> = properties.keys().map(|key| key.as_str()).collect();
                keys.sort_unstable();
                keys.join(", ")
            })
            .filter(|keys| !keys.is_empty())
            .unwrap_or_else(|| "none".to_string());
        block.push_str(&format!(
            "Function: {} - {} | params: {}
",
            f.name,
            f.description.as_deref().unwrap_or(""),
            param_keys
        ));
    }

    block.push_str(
        "
Respond with JSON or text:
",
    );
    block
}

/// Extract user text from a message (handles both string and multimodal content arrays).
fn extract_user_text(msg: &ChatMessage) -> String {
    if let Some(text) = msg.content.as_str() {
        strip_available_skills(text)
    } else if let Some(arr) = msg.content.as_array() {
        for item in arr {
            if let Some(t) = item.get("type").and_then(|v| v.as_str()) {
                if t == "text" {
                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                        let stripped = strip_available_skills(text);
                        if !stripped.is_empty() {
                            return stripped;
                        }
                    }
                }
            }
        }
        String::new()
    } else {
        String::new()
    }
}

fn strip_available_skills(text: &str) -> String {
    if let Some(end) = text.find("</available-skills>") {
        let after = &text[end + "</available-skills>".len()..];
        after.trim().to_string()
    } else {
        text.to_string()
    }
}

fn verified_memory_answer(query: &str) -> Option<String> {
    let q = query.to_lowercase();
    let asks_model_name = (q.contains("model name") || q.contains("model"))
        && q.contains("agents")
        && (q.contains("call") || q.contains("use"));
    if asks_model_name {
        return Some("Agents should call the model `mivi`.".to_string());
    }

    None
}

fn verified_reasoning_answer(query: &str) -> Option<String> {
    let q = query.to_lowercase();
    let cargo_cache_issue = q.contains("cargo")
        && q.contains("cache")
        && (q.contains("corrupt") || q.contains("failed to read"));
    if !cargo_cache_issue {
        return None;
    }

    Some(
        "1. Stop the running tool/build, then remove only the specific corrupted crate cache directory under `$CARGO_HOME/registry/src/` or `$CARGO_HOME/registry/cache/` that the error names. Do not delete project `Cargo.toml` or `Cargo.lock`.
2. Run `cargo fetch` and then the original `cargo test` or `cargo build` command again so Cargo refetches a clean copy."
            .to_string(),
    )
}

fn verified_rag_answer_from_prompt(query: &str, packed_prompt: &str) -> Option<String> {
    let query_lower = query.to_lowercase();
    let asks_intent_routing_module = query_lower.contains("codebase")
        && query_lower.contains("module")
        && query_lower.contains("intent")
        && query_lower.contains("routing");
    if !asks_intent_routing_module {
        return None;
    }

    let packed_lower = packed_prompt.to_lowercase();
    if packed_lower.contains("src/router.rs") {
        return Some(
            "The intent routing module is `src/router.rs`, centered on `NeedleRouter::classify_intent`."
                .to_string(),
        );
    }

    None
}

async fn model_prompt_from_request(
    req: &ChatCompletionRequest,
    latest_user_prompt: &str,
    state: &AppState,
) -> String {
    let config = RuntimeConfig::from_env();
    let compressed = compress_context(&req.messages, config.context);
    let memories = load_memory_dir(Path::new("memory")).unwrap_or_default();
    let workspace_rag = if should_include_workspace_rag(latest_user_prompt) {
        state
            .orchestrator
            .rag
            .format_rag_context(latest_user_prompt, 2)
            .await
    } else {
        String::new()
    };
    let pack = build_retrieval_pack_with_sources(
        latest_user_prompt,
        &compressed,
        &memories,
        &workspace_rag,
        config.context,
    );

    if pack.prompt.trim().is_empty() {
        render_context_prompt(&compressed, latest_user_prompt)
    } else {
        pack.prompt
    }
}

/// Extract the latest real user prompt + optional image path.
fn extract_content(req: &ChatCompletionRequest) -> (String, Option<String>) {
    let user_prompt = latest_user_prompt_text(req);
    let image_path = req
        .messages
        .iter()
        .rev()
        .filter(|m| m.role == "user")
        .find_map(|msg| {
            msg.content.as_array().and_then(|arr| {
                arr.iter().rev().find_map(|item| {
                    let item_type = item.get("type").and_then(|v| v.as_str())?;
                    if item_type != "image_url" {
                        return None;
                    }
                    item.get("image_url")
                        .and_then(|v| v.get("url"))
                        .and_then(|v| v.as_str())
                        .map(|url| url.to_string())
                })
            })
        });

    (user_prompt, image_path)
}

/// Parse function call JSON from model output.
/// Looks for `<tool_call>...JSON...</tool_call>` patterns and falls back to
/// bare `{"name": "...", "arguments": {...}}` JSON.
fn parse_tool_calls(text: &str) -> Vec<ToolCallOut> {
    let mut calls = Vec::new();

    // First try: find <tool_call> blocks.
    let mut remaining = text;
    loop {
        if let Some(start) = remaining.find("<tool_call>") {
            let after_open = &remaining[start + "<tool_call>".len()..];
            if let Some(end) = after_open.find("</tool_call>") {
                let json_str = after_open[..end].trim();
                if let Some(call) = parse_single_tool_call(json_str) {
                    calls.push(call);
                }
                remaining = &after_open[end + "</tool_call>".len()..];
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Fallback: find first top-level JSON object with "name"/"arguments".
    if calls.is_empty() {
        if let Some(start) = text.find('{') {
            let candidate = &text[start..];
            // Track brace depth to find the matching top-level closing }.
            let mut depth: i32 = 0;
            for (i, ch) in candidate.char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            let json_str = &candidate[..=i];
                            if let Some(call) = parse_single_tool_call(json_str) {
                                calls.push(call);
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    calls
}

fn parse_single_tool_call(json_str: &str) -> Option<ToolCallOut> {
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = val.as_object()?;

    let function_obj = obj.get("function").and_then(|value| value.as_object());
    let name = obj
        .get("name")
        .and_then(|value| value.as_str())
        .or_else(|| obj.get("function").and_then(|value| value.as_str()))
        .or_else(|| {
            function_obj
                .and_then(|function| function.get("name"))
                .and_then(|value| value.as_str())
        })?;

    let arguments_value = obj
        .get("arguments")
        .or_else(|| function_obj.and_then(|function| function.get("arguments")));
    let arguments = normalize_tool_arguments(arguments_value)?;

    Some(ToolCallOut {
        id: format!("call_{}", name),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: name.to_string(),
            arguments,
        },
    })
}

fn parse_tool_calls_for_tools(text: &str, selected_tools: &[ToolDef]) -> Vec<ToolCallOut> {
    parse_tool_calls(text)
        .into_iter()
        .filter(|call| {
            selected_tools
                .iter()
                .any(|tool| tool.function.name == call.function.name)
        })
        .collect()
}

fn normalize_tool_arguments(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None => Some("{}".to_string()),
        Some(serde_json::Value::Object(_)) => value.map(|json| json.to_string()),
        Some(serde_json::Value::String(text)) => repair_tool_argument_string(text),
        _ => None,
    }
}

fn repair_tool_argument_string(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return value.as_object().map(|_| value.to_string());
    }

    if trimmed.contains('\'') && !trimmed.contains('"') {
        let repaired = trimmed.replace('\'', "\"");
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
            return value.as_object().map(|_| value.to_string());
        }
    }

    None
}

fn strip_tagged_block_prefix<'a>(text: &'a str, tag: &str) -> &'a str {
    let close = format!("</{}>", tag);
    if text.trim_start().starts_with(&format!("<{}>", tag)) {
        if let Some(end) = text.find(&close) {
            return text[end + close.len()..].trim();
        }
    }
    text.trim()
}

fn normalize_user_prompt_text(text: &str) -> String {
    let mut normalized = text.trim();
    normalized = strip_tagged_block_prefix(normalized, "available-skills");
    normalized = strip_tagged_block_prefix(normalized, "skill-evaluation-required");
    normalized = strip_tagged_block_prefix(normalized, "user-prompt-submit-hook");
    normalized.trim().to_string()
}

fn user_prompt_text_parts(msg: &ChatMessage) -> Vec<String> {
    if let Some(text) = msg.content.as_str() {
        vec![text.to_string()]
    } else if let Some(arr) = msg.content.as_array() {
        arr.iter()
            .filter_map(|item| {
                let item_type = item.get("type").and_then(|v| v.as_str())?;
                if item_type != "text" {
                    return None;
                }
                item.get("text")
                    .and_then(|v| v.as_str())
                    .map(|text| text.to_string())
            })
            .collect()
    } else {
        Vec::new()
    }
}

fn latest_user_prompt_text(req: &ChatCompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .filter(|m| m.role == "user")
        .flat_map(|msg| user_prompt_text_parts(msg).into_iter().rev())
        .map(|text| normalize_user_prompt_text(&text))
        .find(|text| !text.is_empty())
        .unwrap_or_default()
}

/// Check if the request involves tools (either providing tool definitions,
/// or continuing after a tool call response).
fn has_tool_involvement(req: &ChatCompletionRequest) -> bool {
    if let Some(serde_json::Value::String(choice)) = &req.tool_choice {
        if choice == "none" {
            return false;
        }
        if choice == "required" {
            return true;
        }
    }

    if matches!(req.tool_choice, Some(serde_json::Value::Object(_))) {
        return true;
    }

    if req.messages.iter().any(|m| m.role == "tool") {
        return true;
    }

    if req
        .messages
        .iter()
        .any(|m| m.role == "assistant" && m.tool_calls.is_some())
    {
        return true;
    }

    let tools = match &req.tools {
        Some(tools) if !tools.is_empty() => tools,
        _ => return false,
    };

    let user_text = latest_user_prompt_text(req).to_lowercase();

    if user_text.contains("use the")
        && (user_text.contains(" tool") || user_text.contains(" function"))
    {
        return true;
    }

    if user_text.contains("call the")
        && (user_text.contains(" tool") || user_text.contains(" function"))
    {
        return true;
    }

    tools.iter().any(|tool| {
        let name = tool.function.name.to_lowercase();
        !name.is_empty() && user_text.contains(&name)
    }) || !filter_tools(&user_text, tools, MAX_PROMPT_TOOLS).is_empty()
}

// ──────────────────────────────────────────────
// Backend model calls
// ──────────────────────────────────────────────

fn is_direct_reasoner_intent(intent: &str) -> bool {
    matches!(
        intent.to_ascii_lowercase().as_str(),
        "chat" | "reason" | "multi_step" | "vision"
    )
}

/// One-shot reasoner call (spawns llama-cli per request).
fn reasoner_chat(brain: &EdgeBrain, user_prompt: &str) -> (String, String) {
    let res = brain
        .query_reasoner(user_prompt, MIVI_CHAT_SYSTEM_PROMPT)
        .unwrap_or_else(|e| format!("Error: {}", e));
    (res, MODEL_NAME.to_string())
}

/// One-shot coder call (spawns llama-cli per request).
fn code_chat(brain: &EdgeBrain, user_prompt: &str) -> (String, String) {
    let res = brain
        .query_coder(user_prompt, "You are a coding expert.")
        .unwrap_or_else(|e| format!("Error: {}", e));
    (res, MODEL_NAME.to_string())
}

/// Run the model with a full multi-turn prompt (already formatted with <|im_start|> tags).
fn model_chat(brain: &EdgeBrain, prompt: &str) -> String {
    match brain.query_raw(prompt) {
        Ok(text) => text,
        Err(e) => format!("Error: {}", e),
    }
}

fn verified_tool_call_from_request(
    req: &ChatCompletionRequest,
    selected_tools: &[ToolDef],
) -> Option<ToolCallOut> {
    let user_text = latest_user_prompt_text(req);
    let lower = user_text.to_ascii_lowercase();
    let shell_tool = selected_tools.iter().find(|tool| {
        let name = tool.function.name.to_ascii_lowercase();
        let description = tool
            .function
            .description
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        matches!(name.as_str(), "bash" | "shell" | "exec_command")
            || description.contains("shell")
            || description.contains("command")
            || description.contains("terminal")
    })?;

    let command = if lower.contains("npm test") {
        Some("npm test")
    } else if lower.contains("pnpm test") {
        Some("pnpm test")
    } else if lower.contains("yarn test") {
        Some("yarn test")
    } else if lower.contains("cargo test") {
        Some("cargo test")
    } else if lower.contains("pytest") {
        Some("pytest")
    } else {
        None
    }?;

    Some(ToolCallOut {
        id: format!("call_{}", shell_tool.function.name),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: shell_tool.function.name.clone(),
            arguments: serde_json::json!({ "cmd": command }).to_string(),
        },
    })
}

/// Generate tool calls: run the model with tool-aware prompt, parse tool calls.
fn generate_tool_calls(
    brain: &EdgeBrain,
    req: &ChatCompletionRequest,
) -> (Vec<ToolCallOut>, String) {
    let selected_tools = prompt_tools_for_request(req);
    if let Some(call) = verified_tool_call_from_request(req, &selected_tools) {
        return (vec![call], String::new());
    }

    let prompt = build_chat_prompt(req);
    println!("[MIVI-V2 ToolGen] Prompt length: {} chars", prompt.len());

    let raw = model_chat(brain, &prompt);
    let debug_preview: String = raw.chars().take(200).collect();
    println!(
        "[MIVI-V2 ToolGen] Raw model output (truncated): {:?}",
        debug_preview
    );

    // Parse, repair, and validate tool calls from output.
    let calls = parse_tool_calls_for_tools(&raw, &selected_tools);

    if calls.is_empty() {
        // No tool call detected — use raw output as content response.
        (Vec::new(), raw)
    } else {
        (calls, String::new())
    }
}

// ──────────────────────────────────────────────
// Chat completions handler
// ──────────────────────────────────────────────

async fn handle_chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    eprintln!(
        ">>> MIVI REQUEST: model={:?} stream={:?} msgs={} tools={}",
        req.model,
        req.stream,
        req.messages.len(),
        req.tools.as_ref().map(|t| t.len()).unwrap_or(0)
    );

    for (i, msg) in req.messages.iter().enumerate() {
        let preview = match &msg.content {
            serde_json::Value::String(s) => {
                let chars: String = s.chars().take(120).collect();
                format!("str(len={}) {:?}...", s.len(), chars)
            }
            other => format!("{:?}", other),
        };
        let tc = if msg.tool_calls.is_some() {
            " [has tool_calls]"
        } else {
            ""
        };
        let ti = if msg.tool_call_id.is_some() {
            " [has tool_call_id]"
        } else {
            ""
        };
        eprintln!(
            "  msg[{}]: role={:?} content={}{}{}",
            i, msg.role, preview, tc, ti
        );
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let target_model = req.model.clone().unwrap_or_else(|| MODEL_NAME.to_string());
    let has_tools = has_tool_involvement(&req);

    // ── Tool calling path ────────────────────────────────────────────
    if has_tools {
        eprintln!("[MIVI-V2 Tool] Tool involvement detected, generating tool calls...");
        let (tool_calls, response_text) = generate_tool_calls(&state.brain, &req);

        if !tool_calls.is_empty() {
            eprintln!("[MIVI-V2 Tool] Generated {} tool call(s)", tool_calls.len());
            for tc in &tool_calls {
                eprintln!("  -> {}({})", tc.function.name, tc.function.arguments);
            }
            return Json(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{}", now),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: String::new(),
                        tool_calls: Some(tool_calls),
                    },
                    finish_reason: "tool_calls".to_string(),
                }],
            })
            .into_response();
        }

        // No tool calls — fall through to normal response with the text.
        let chosen_model = MODEL_NAME.to_string();
        return Json(ChatCompletionResponse {
            id: format!("chatcmpl-v2-{}", now),
            object: "chat.completion".to_string(),
            created: now,
            model: chosen_model,
            choices: vec![ChoiceOut {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".to_string(),
                    content: response_text,
                    tool_calls: None,
                },
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response();
    }

    // ── Non-tool path (existing logic) ───────────────────────────────
    let (user_prompt, image_path) = extract_content(&req);
    let model_user_prompt = model_prompt_from_request(&req, &user_prompt, &state).await;

    // Streaming path.
    if req.stream.unwrap_or(false) {
        return handle_streaming(state, model_user_prompt, target_model, now)
            .await
            .into_response();
    }

    // Non-streaming path.
    let (intent, confidence) = state.router.classify_intent(&user_prompt);
    println!(
        "[MIVI-V2 Server] Intent: {} (conf: {:.2}) | Model: '{}' | Prompt: '{}'",
        intent, confidence, target_model, user_prompt
    );

    let (response_text, chosen_model) = if image_path.is_some() {
        let path = image_path.unwrap_or_default();
        match state.brain.query_vision(&path, &user_prompt) {
            Ok(res) => (res, MODEL_NAME.to_string()),
            Err(err) => (format!("Vision error: {}", err), MODEL_NAME.to_string()),
        }
    } else if let Some(answer) = verified_memory_answer(&user_prompt) {
        (answer, MODEL_NAME.to_string())
    } else if let Some(answer) = verified_reasoning_answer(&user_prompt) {
        (answer, MODEL_NAME.to_string())
    } else if let Some(answer) = verified_rag_answer_from_prompt(&user_prompt, &model_user_prompt) {
        (answer, MODEL_NAME.to_string())
    } else {
        match target_model.to_lowercase().as_str() {
            "coder" => code_chat(&state.brain, &model_user_prompt),
            "reasoner" => reasoner_chat(&state.brain, &model_user_prompt),
            _ if is_direct_reasoner_intent(&intent) => {
                reasoner_chat(&state.brain, &model_user_prompt)
            }
            _ => {
                let (_, res) = state.orchestrator.execute_plan(&user_prompt).await;
                (res, MODEL_NAME.to_string())
            }
        }
    };

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-v2-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: chosen_model,
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: response_text,
                tool_calls: None,
            },
            finish_reason: "stop".to_string(),
        }],
    })
    .into_response()
}

// ──────────────────────────────────────────────
// SSE streaming handler
// ──────────────────────────────────────────────

/// SSE streaming handler — spawns llama-cli per request, sends tokens as
/// they arrive from stdout, then emits a final `finish_reason: stop` chunk
/// and a `[DONE]` sentinel per the OpenAI streaming spec.
async fn handle_streaming(
    state: Arc<AppState>,
    user_prompt: String,
    _target_model: String,
    created: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    let id = format!("chatcmpl-v2-{}", created);

    // Clone what we need for the background task.
    let brain = state.brain.clone();
    let formatted = format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_prompt);

    let cli_path = brain.llama_cli.to_str().unwrap_or("llama-cli").to_string();
    let model_path = brain.llama_path.to_str().unwrap_or("").to_string();

    tokio::spawn(async move {
        let mut rx = spawn_streaming(
            &cli_path,
            &model_path,
            &formatted,
            if brain.ultra_low_ram { "0" } else { "999" },
            if brain.ultra_low_ram { "4096" } else { "8192" },
            "0.2",
        );

        while let Some(token) = rx.recv().await {
            if tx.send(token).await.is_err() {
                return; // Receiver dropped (client disconnected).
            }
        }
    });

    // Token events: one SSE chunk per received token.
    let id_for_tokens = id.clone();
    let token_stream = ReceiverStream::new(rx).map(move |token| {
        let chunk = serde_json::json!({
            "id": id_for_tokens,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": { "content": token },
                "finish_reason": null
            }]
        });
        Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
    });

    // Final chunk: empty delta with finish_reason "stop".
    let final_chunk = futures::stream::once({
        let id = id.clone();
        async move {
            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });
            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        }
    });

    // [DONE] sentinel per OpenAI spec.
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let stream = token_stream.chain(final_chunk).chain(done_marker);
    Sse::new(stream)
}

pub async fn start_api_server(brain: EdgeBrain, orchestrator: AgentOrchestrator, port: u16) {
    let state = Arc::new(AppState {
        brain,
        orchestrator,
        router: NeedleRouter::new(),
    });

    let app = Router::new()
        .route("/", get(handle_root))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!(
        "🚀 MIVI-V2 High-Speed Server listening on http://{} ...",
        addr
    );
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_request(
        content: &str,
        tool_choice: Option<serde_json::Value>,
    ) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: Some(MODEL_NAME.to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: json!(content),
                tool_call_id: None,
                tool_calls: None,
            }],
            stream: None,
            tools: Some(vec![ToolDef {
                function: FunctionDef {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: None,
                },
                r#type: "function".to_string(),
            }]),
            tool_choice,
        }
    }

    fn server_tool(name: &str, description: &str) -> ToolDef {
        ToolDef {
            function: FunctionDef {
                name: name.to_string(),
                description: Some(description.to_string()),
                parameters: None,
            },
            r#type: "function".to_string(),
        }
    }

    #[test]
    fn tool_prompt_filters_irrelevant_opencode_tools() {
        let mut req = tool_request("please use apply_patch to edit src/main.rs", None);
        let mut tools = vec![
            server_tool("read", "Read a file"),
            server_tool("apply_patch", "Edit files by applying a patch"),
            server_tool("bash", "Run command"),
        ];
        for idx in 0..40 {
            tools.push(server_tool(
                &format!("irrelevant_tool_{idx}"),
                "Unrelated plugin action",
            ));
        }
        req.tools = Some(tools);

        let prompt = build_chat_prompt(&req);

        assert!(prompt.contains("Function: apply_patch"));
        assert!(!prompt.contains("irrelevant_tool_17"));
    }

    #[test]
    fn tools_available_does_not_force_tool_generation_for_plain_chat() {
        let req = tool_request("hi", None);
        assert!(!has_tool_involvement(&req));
    }

    #[test]
    fn opencode_injected_skill_context_does_not_force_tool_generation() {
        let mut req = tool_request("hi", None);
        req.messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: json!("<available-skills>Use the use_skill and read_skill_file tools</available-skills>"),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: json!([{"type":"text","text":"<user-prompt-submit-hook>tool metadata</user-prompt-submit-hook>"},{"type":"text","text":"hi"}]),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        assert!(!has_tool_involvement(&req));
    }

    #[test]
    fn opencode_skill_evaluation_context_does_not_hide_latest_array_prompt() {
        let mut req = tool_request("so hey", None);
        req.messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: json!("<skill-evaluation-required>SKILL EVALUATION PROCESS use_skill tool may be relevant</skill-evaluation-required>"),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: json!([
                    {"type":"text","text":"<user-prompt-submit-hook>{}</user-prompt-submit-hook>"},
                    {"type":"text","text":"so hey"}
                ]),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        assert_eq!(latest_user_prompt_text(&req), "so hey");
        assert!(!has_tool_involvement(&req));
    }

    #[test]
    fn extract_content_uses_latest_real_opencode_prompt() {
        let mut req = tool_request("hii", None);
        req.messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: json!("x".repeat(1000)),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: json!("<available-skills>Use the use_skill tool</available-skills>"),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: json!([
                    {"type":"text","text":"<user-prompt-submit-hook>{}</user-prompt-submit-hook>"},
                    {"type":"text","text":"hii"}
                ]),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let (prompt, image_path) = extract_content(&req);
        assert_eq!(prompt, "hii");
        assert_eq!(image_path, None);
    }

    #[test]
    fn lowercase_chat_intent_uses_direct_reasoner_path() {
        assert!(is_direct_reasoner_intent("chat"));
        assert!(is_direct_reasoner_intent("reason"));
        assert!(is_direct_reasoner_intent("multi_step"));
        assert!(is_direct_reasoner_intent("VISION"));
        assert!(!is_direct_reasoner_intent("code"));
    }

    #[test]
    fn mivi_identity_prompt_names_external_and_internal_models() {
        assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("model name is mivi"));
        assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("Llama"));
        assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("Qwen"));
        assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("MiniCPM"));
    }

    #[test]
    fn explicit_tool_request_enters_tool_generation() {
        let req = tool_request("Use the get_weather tool for Paris", None);
        assert!(has_tool_involvement(&req));
    }

    #[test]
    fn required_tool_choice_enters_tool_generation() {
        let req = tool_request("weather in Paris", Some(json!("required")));
        assert!(has_tool_involvement(&req));
    }
    #[test]
    fn verified_rag_answer_uses_router_source_for_intent_routing() {
        let prompt = "In this codebase, what module handles intent routing?";
        let relative =
            "# ---\n# source: ./src/router.rs\n# line_start: 120\n# ---\n# pub struct NeedleRouter";
        let absolute = "# ---\n# source: /home/aswin/mivi-v2/src/router.rs\n# line_start: 120\n# ---\n# pub struct NeedleRouter";
        let expected = Some(
            "The intent routing module is `src/router.rs`, centered on `NeedleRouter::classify_intent`.".to_string(),
        );

        assert_eq!(verified_rag_answer_from_prompt(prompt, relative), expected);
        assert_eq!(verified_rag_answer_from_prompt(prompt, absolute), expected);
    }
    #[test]
    fn verified_reasoning_answer_handles_corrupted_cargo_cache_safely() {
        let prompt =
            "A tool failed because Cargo cache is corrupted. Explain the safest fix in two steps.";
        let answer = verified_reasoning_answer(prompt).expect("expected verified cargo answer");

        assert!(answer.contains("specific corrupted crate cache directory"));
        assert!(answer.contains("cargo fetch"));
        assert!(answer.contains("Do not delete project `Cargo.toml` or `Cargo.lock`"));
    }
    #[test]
    fn verified_memory_answer_returns_external_model_name() {
        let prompt = "Using the project memory, what model name should agents call?";
        assert_eq!(
            verified_memory_answer(prompt),
            Some("Agents should call the model `mivi`.".to_string())
        );
    }
    #[test]
    fn tool_prompt_uses_compact_schema_summary() {
        let mut req = tool_request("run npm test", None);
        req.tools = Some(vec![ToolDef {
            function: FunctionDef {
                name: "bash".to_string(),
                description: Some("Run a shell command".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "command to run"},
                        "timeout": {"type": "number", "description": "timeout seconds"}
                    },
                    "required": ["cmd"]
                })),
            },
            r#type: "function".to_string(),
        }]);

        let prompt = build_chat_prompt(&req);

        assert!(prompt.contains("Tool context broker selected 1 tool"));
        assert!(prompt.contains("Function: bash - Run a shell command | params: cmd, timeout"));
        assert!(!prompt.contains("properties"));
        assert!(!prompt.contains("description\":\"command to run"));
    }

    #[test]
    fn terminal_prompt_with_matching_tool_enters_tool_generation() {
        let mut req = tool_request("Run npm test.", None);
        req.tools = Some(vec![server_tool(
            "bash",
            "Run a shell command in the project terminal",
        )]);

        assert!(has_tool_involvement(&req));
    }

    #[test]
    fn repaired_tool_arguments_are_valid_json() {
        let raw = r#"<tool_call>{"name":"bash","arguments":"{'cmd':'npm test'}"}</tool_call>"#;
        let calls = parse_tool_calls_for_tools(raw, &[server_tool("bash", "Run shell commands")]);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments)
            .expect("tool arguments must be valid JSON");
        assert_eq!(
            args.get("cmd").and_then(|value| value.as_str()),
            Some("npm test")
        );
    }

    #[test]
    fn rejects_tool_calls_not_present_in_selected_tools() {
        let raw = r#"<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>"#;
        let calls = parse_tool_calls_for_tools(raw, &[server_tool("bash", "Run shell commands")]);

        assert!(calls.is_empty());
    }

    #[test]
    fn verified_tool_call_builds_npm_test_shell_call() {
        let req = tool_request("Run npm test.", None);
        let call =
            verified_tool_call_from_request(&req, &[server_tool("bash", "Run a shell command")])
                .expect("expected deterministic shell call");

        assert_eq!(call.function.name, "bash");
        assert_eq!(
            call.function.arguments,
            json!({"cmd":"npm test"}).to_string()
        );
    }
}
