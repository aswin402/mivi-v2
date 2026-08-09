use crate::server::handlers::*;
use axum::{
    extract::{Json, State},
    response::sse::{Event, Sse},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, OnceLock,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};

use crate::brain::EdgeBrain;
use crate::context_compressor::{compress_context, render_context_prompt};
use crate::model_catalog::{ModelCatalog, ModelRole};
use crate::model_process::spawn_streaming;
use crate::okf_memory::load_memory_dir;
use crate::orchestrator::AgentOrchestrator;
use crate::retrieval::{build_retrieval_pack_with_sources, should_include_workspace_rag};
use crate::router::NeedleRouter;
use crate::runtime::RuntimeConfig;
use crate::tool_filter::filter_tools;
use crate::trace::{preview as trace_preview, trace_event, TraceConfig};

use crate::server::types::*;

/// The single model name exposed to external agents.
/// Internal SML routing is hidden behind this constant.
pub const MODEL_NAME: &str = "mivi";

const MIVI_CHAT_SYSTEM_PROMPT: &str = "You are MIVI, a local OpenAI-compatible model endpoint for AI agents. Externally your model name is mivi. Never identify as an internal worker model or as the calling agent/platform. Treat the calling agent's system prompt, tool schemas, tool results, skills, memory, and retrieved context as the source of truth for that agent's capabilities. If asked about available tools, MCP servers, skills, features, or capabilities, use agent-provided introspection/inventory tools when available; otherwise describe only the tool schemas included in the current request. Answer concisely and honestly.";

// ──────────────────────────────────────────────
// OpenAI-compatible tool/function structs
// ──────────────────────────────────────────────

pub fn default_tool_type() -> String {
    "function".into()
}

// ──────────────────────────────────────────────
// Request / Message structs
// ──────────────────────────────────────────────

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs()
}

pub fn response_format_type(req: &ChatCompletionRequest) -> Option<String> {
    req.response_format
        .as_ref()
        .and_then(|format| format.get("type"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

pub fn validate_response_format(req: &ChatCompletionRequest) -> Result<(), String> {
    match response_format_type(req).as_deref() {
        None | Some("text") | Some("json_object") | Some("json_schema") => Ok(()),
        Some(other) => Err(format!("unsupported response_format type `{other}`")),
    }
}

pub fn active_chat_template() -> crate::model_catalog::ChatTemplateConfig {
    static CONFIG: std::sync::OnceLock<crate::model_catalog::ChatTemplateConfig> =
        std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| {
            if let Ok(catalog) = crate::model_catalog::ModelCatalog::load_default() {
                if let Some(entry) = catalog
                    .models
                    .iter()
                    .find(|e| e.enabled && e.role == crate::model_catalog::ModelRole::Reasoner)
                {
                    if let Some(ref template) = entry.chat_template {
                        return template.clone();
                    }
                }
            }
            crate::model_catalog::ChatTemplateConfig::default()
        })
        .clone()
}

pub fn extract_json_schema(req: &ChatCompletionRequest) -> Option<String> {
    req.response_format
        .as_ref()
        .and_then(|format| format.get("json_schema"))
        .and_then(|js| js.get("schema"))
        .map(|schema| schema.to_string())
}

pub fn apply_response_format(
    content: String,
    req: &ChatCompletionRequest,
) -> Result<String, String> {
    validate_response_format(req)?;
    if response_format_type(req).as_deref() == Some("json_object") {
        return Ok(serde_json::json!({ "answer": content }).to_string());
    }
    Ok(content)
}

pub trait TokenCounter {
    fn count_tokens(&self, text: &str) -> u32;
}

pub fn count_with_llama_cpp_tokenizer(command: &Path, model: &Path, text: &str) -> Option<u32> {
    let output = Command::new(command)
        .arg("--model")
        .arg(model)
        .arg("--tokenize")
        .arg("--prompt")
        .arg(text)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        combined.push('\n');
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    count_llama_tokenize_output(&combined)
}

pub fn count_llama_tokenize_output(output: &str) -> Option<u32> {
    let mut in_list = false;
    let mut digits = String::new();
    let mut count = 0_u32;

    for ch in output.chars() {
        match ch {
            '[' if !in_list => {
                in_list = true;
                digits.clear();
            }
            ']' if in_list => {
                if !digits.is_empty() {
                    count = count.saturating_add(1);
                    digits.clear();
                }
                return Some(count);
            }
            ch if in_list && ch.is_ascii_digit() => digits.push(ch),
            _ if in_list => {
                if !digits.is_empty() {
                    count = count.saturating_add(1);
                    digits.clear();
                }
            }
            _ => {}
        }
    }

    None
}

pub fn token_counter() -> RuntimeTokenCounter {
    TokenCounterConfig::from_env().counter()
}

pub fn value_text_for_usage(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(ToString::to_string)
        .unwrap_or_else(|| value.to_string())
}

pub fn estimated_prompt_tokens(req: &ChatCompletionRequest) -> u32 {
    let counter = token_counter();
    let message_tokens = req.messages.iter().fold(0_u32, |total, message| {
        total
            .saturating_add(counter.count_tokens(&message.role))
            .saturating_add(counter.count_tokens(&value_text_for_usage(&message.content)))
            .saturating_add(4)
    });
    let tool_tokens = req
        .tools
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .fold(0_u32, |total, tool| {
            total
                .saturating_add(counter.count_tokens(&tool.function.name))
                .saturating_add(
                    tool.function
                        .description
                        .as_deref()
                        .map(|description| counter.count_tokens(description))
                        .unwrap_or(0),
                )
                .saturating_add(
                    tool.function
                        .parameters
                        .as_ref()
                        .map(|parameters| counter.count_tokens(&parameters.to_string()))
                        .unwrap_or(0),
                )
        });
    message_tokens.saturating_add(tool_tokens)
}

pub fn estimated_usage_for_text(req: &ChatCompletionRequest, completion: &str) -> UsageInfo {
    UsageInfo::new(
        estimated_prompt_tokens(req),
        token_counter().count_tokens(completion),
    )
}

pub fn estimated_usage_for_tool_calls(
    req: &ChatCompletionRequest,
    calls: &[ToolCallOut],
) -> UsageInfo {
    let completion_text = calls
        .iter()
        .map(|call| format!("{} {}", call.function.name, call.function.arguments))
        .collect::<Vec<_>>()
        .join("\n");
    estimated_usage_for_text(req, &completion_text)
}

pub fn usage_value(usage: UsageInfo) -> serde_json::Value {
    serde_json::to_value(usage).unwrap_or_else(|_| {
        serde_json::json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    })
}

pub fn include_stream_usage(req: &ChatCompletionRequest) -> bool {
    req.stream_options
        .as_ref()
        .and_then(|options| options.get("include_usage"))
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

pub fn responses_reasoning_effort(req: &ResponsesRequest) -> Option<String> {
    req.reasoning_effort.clone().or_else(|| {
        req.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.get("effort"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
    })
}

pub fn responses_content_to_chat_content(content: serde_json::Value) -> serde_json::Value {
    if let Some(items) = content.as_array() {
        let mapped: Vec<serde_json::Value> = items
            .iter()
            .map(|item| {
                let kind = item.get("type").and_then(|value| value.as_str());
                if matches!(kind, Some("input_text" | "output_text")) {
                    serde_json::json!({
                        "type": "text",
                        "text": item.get("text").and_then(|value| value.as_str()).unwrap_or("")
                    })
                } else {
                    item.clone()
                }
            })
            .collect();
        serde_json::Value::Array(mapped)
    } else {
        content
    }
}

pub fn responses_request_to_chat_request(req: ResponsesRequest) -> ChatCompletionRequest {
    let reasoning_effort = responses_reasoning_effort(&req);
    let messages = match req.input {
        ResponsesInput::Text(text) => vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String(text),
            tool_call_id: None,
            tool_calls: None,
        }],
        ResponsesInput::Messages(messages) => messages
            .into_iter()
            .map(|message| ChatMessage {
                role: message.role,
                content: responses_content_to_chat_content(message.content),
                tool_call_id: None,
                tool_calls: None,
            })
            .collect(),
    };

    ChatCompletionRequest {
        model: req.model,
        messages,
        stream: req.stream,
        tools: req.tools,
        tool_choice: req.tool_choice,
        max_tokens: req.max_output_tokens,
        stop: req.stop,
        seed: req.seed,
        response_format: req.response_format,
        stream_options: req.stream_options,
        parallel_tool_calls: req.parallel_tool_calls,
        reasoning_effort,
        temperature: req.temperature,
        top_p: req.top_p,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        user: req.user,
    }
}

pub fn responses_response_from_chat(chat: ChatCompletionResponse) -> ResponsesResponse {
    let output = chat
        .choices
        .into_iter()
        .flat_map(|choice| {
            let mut items = Vec::new();
            if let Some(calls) = choice.message.tool_calls {
                items.extend(calls.into_iter().map(|call| ResponsesOutputItem {
                    id: call.id.clone(),
                    r#type: "function_call".to_string(),
                    status: Some("completed".to_string()),
                    role: None,
                    content: Vec::new(),
                    name: Some(call.function.name),
                    arguments: Some(call.function.arguments),
                    call_id: Some(call.id),
                }));
            }
            if !choice.message.content.is_empty() {
                items.push(ResponsesOutputItem {
                    id: format!("msg_{}", choice.index),
                    r#type: "message".to_string(),
                    status: Some("completed".to_string()),
                    role: Some(choice.message.role),
                    content: vec![ResponsesOutputContent {
                        r#type: "output_text".to_string(),
                        text: choice.message.content,
                        annotations: Vec::new(),
                    }],
                    name: None,
                    arguments: None,
                    call_id: None,
                });
            }
            items
        })
        .collect();

    ResponsesResponse {
        id: chat.id.replace("chatcmpl", "resp"),
        object: "response".to_string(),
        created_at: chat.created,
        model: chat.model,
        status: "completed".to_string(),
        output,
        usage: chat.usage,
    }
}

// ──────────────────────────────────────────────
// Response structs
// ──────────────────────────────────────────────

// ──────────────────────────────────────────────
// Handlers
// ──────────────────────────────────────────────

// ──────────────────────────────────────────────
// Prompt building
// ──────────────────────────────────────────────

const MAX_PROMPT_TOOLS: usize = 8;

pub fn build_chat_prompt(req: &ChatCompletionRequest) -> String {
    let t = active_chat_template();
    let mut prompt = String::new();
    let prompt_tools = prompt_tools_for_request(req);
    let agent_contract = agent_contract_prompt_for_tools(&prompt_tools);

    let has_user_system = req.messages.iter().any(|m| m.role == "system");
    if has_user_system {
        for msg in &req.messages {
            if msg.role == "system" {
                if let Some(text) = msg.content.as_str() {
                    if !text.is_empty() {
                        let system_text = wrap_agent_prompt(&agent_contract, text);
                        prompt.push_str(&format!(
                            "{}{}{}",
                            t.system_prefix, system_text, t.system_suffix
                        ));
                    }
                }
            }
        }
    } else if prompt_tools.is_empty() {
        let system_text =
            wrap_agent_prompt(&agent_contract, "You are a helpful, concise AI assistant.");
        prompt.push_str(&format!(
            "{}{}{}",
            t.system_prefix, system_text, t.system_suffix
        ));
    } else {
        prompt.push_str(&format!(
            "{}{}{}",
            t.system_prefix, agent_contract, t.system_suffix
        ));
    }

    // Conversation turns.
    let has_tools = !prompt_tools.is_empty();
    let mut last_user_idx: Option<usize> = None;

    for (_, msg) in req.messages.iter().enumerate() {
        match msg.role.as_str() {
            "user" => {
                let text = extract_user_text(msg);
                if !text.is_empty() {
                    prompt.push_str(&format!("{}{}{}", t.user_prefix, text, t.user_suffix));
                    last_user_idx = Some(prompt.len());
                }
            }
            "assistant" => {
                if let Some(ref calls) = msg.tool_calls {
                    let content =
                        sanitize_assistant_history_text(msg.content.as_str().unwrap_or(""));
                    let mut block = String::new();
                    if !content.is_empty() {
                        block.push_str(&content);
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
                            "{}{}{}",
                            t.assistant_prefix,
                            block.trim(),
                            t.assistant_suffix
                        ));
                    }
                } else {
                    let text = sanitize_assistant_history_text(msg.content.as_str().unwrap_or(""));
                    if !text.is_empty() {
                        prompt.push_str(&format!(
                            "{}{}{}",
                            t.assistant_prefix, text, t.assistant_suffix
                        ));
                    }
                }
            }
            "tool" => {
                let tool_content = msg.content.as_str().unwrap_or("");
                let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
                let resolved_prefix = t.tool_prefix.replace("{id}", tool_id);
                prompt.push_str(&format!(
                    "{}{}{}",
                    resolved_prefix, tool_content, t.tool_suffix
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

    prompt.push_str(&t.assistant_start);
    prompt
}

pub fn prompt_tools_for_request(req: &ChatCompletionRequest) -> Vec<ToolDef> {
    select_tools_for_request(req).selected
}

pub fn select_tools_for_request(req: &ChatCompletionRequest) -> ToolSelection {
    let latest_user_prompt = latest_user_prompt_text(req);
    let intent = classify_agent_intent(&latest_user_prompt);
    let tools = match req.tools.as_deref() {
        Some(tools) if !tools.is_empty() => tools,
        _ => return ToolSelection::empty(intent),
    };

    if matches!(req.tool_choice, Some(serde_json::Value::String(ref choice)) if choice == "required")
    {
        return ToolSelection {
            intent: AgentIntent::ToolCall,
            selected: tools.to_vec(),
            blocked: Vec::new(),
        };
    }

    let decision = agent_decision_from_request(req);
    if decision.needs_tool() {
        return ToolSelection {
            intent: AgentIntent::ToolCall,
            selected: select_web_research_tools(tools, MAX_PROMPT_TOOLS),
            blocked: Vec::new(),
        };
    }

    if intent.is_inventory() {
        return select_inventory_tools(intent, tools, &latest_user_prompt);
    }

    ToolSelection {
        intent,
        selected: filter_tools(&latest_user_prompt, tools, MAX_PROMPT_TOOLS),
        blocked: Vec::new(),
    }
}

pub fn blocked_tool_names(blocked: &[ToolBlock]) -> Vec<String> {
    blocked
        .iter()
        .map(|blocked| format!("{}:{}", blocked.name, blocked.reason))
        .collect()
}

pub fn selected_tool_roles(tools: &[ToolDef]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| format!("{}:{:?}", tool.function.name, classify_tool_role(tool)))
        .collect()
}

pub fn select_inventory_tools(
    intent: AgentIntent,
    tools: &[ToolDef],
    latest_user_prompt: &str,
) -> ToolSelection {
    let selected = tools
        .iter()
        .find(|tool| tool_is_inventory_for_intent(tool, intent, latest_user_prompt))
        .cloned()
        .into_iter()
        .collect::<Vec<_>>();
    let selected_name = selected.first().map(|tool| tool.function.name.as_str());

    let blocked = tools
        .iter()
        .filter(|tool| selected_name != Some(tool.function.name.as_str()))
        .filter_map(|tool| inventory_block_reason(tool, intent).map(|reason| (tool, reason)))
        .map(|(tool, reason)| ToolBlock {
            name: tool.function.name.clone(),
            reason,
        })
        .collect();

    ToolSelection {
        intent,
        selected,
        blocked,
    }
}

pub fn agent_contract_prompt(req: &ChatCompletionRequest) -> String {
    let prompt_tools = prompt_tools_for_request(req);
    agent_contract_prompt_for_tools(&prompt_tools)
}

pub fn agent_contract_prompt_for_tools(tools: &[ToolDef]) -> String {
    let mut lines = vec![
        "Agent contract:".to_string(),
        "- External model identity is `mivi`; do not expose internal worker names.".to_string(),
        "- The calling agent supplies the authoritative instructions, tools, skills, memory, database/context, and retrieved facts.".to_string(),
        "- Use only capabilities present in the current request or context; do not invent agent features.".to_string(),
        "- Prefer available introspection/inventory tools for capability questions; otherwise summarize received tool schemas.".to_string(),
        "- For tool use, choose the smallest relevant tool set and return valid tool-call JSON when a tool is required.".to_string(),
    ];

    if !tools.is_empty() {
        let mut names = tool_names(tools);
        names.sort_unstable();
        names.dedup();
        let shown: Vec<String> = names.iter().take(12).cloned().collect();
        let hidden = names.len().saturating_sub(shown.len());
        let suffix = if hidden > 0 {
            format!(" plus {hidden} more")
        } else {
            String::new()
        };
        lines.push(format!(
            "- Current prompt exposes {} selected callable tool schemas: {}{}.",
            names.len(),
            shown.join(", "),
            suffix
        ));
    } else {
        lines.push("- Current prompt exposes no selected callable tool schemas.".to_string());
    }

    lines.join("\n")
}

pub fn wrap_agent_prompt(agent_contract: &str, prompt: &str) -> String {
    if prompt.trim().is_empty() {
        agent_contract.to_string()
    } else {
        format!("{}\n\n{}", agent_contract, prompt.trim())
    }
}

pub fn build_function_list_block(tools: &[ToolDef]) -> String {
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
pub fn extract_user_text(msg: &ChatMessage) -> String {
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

pub fn sanitize_assistant_history_text(text: &str) -> String {
    let mut remaining = text;
    let mut cleaned = String::new();

    loop {
        let Some(start) = remaining.to_ascii_lowercase().find("<think>") else {
            cleaned.push_str(remaining);
            break;
        };
        cleaned.push_str(&remaining[..start]);
        let after_open = &remaining[start + "<think>".len()..];
        let after_open_lower = after_open.to_ascii_lowercase();
        if let Some(end) = after_open_lower.find("</think>") {
            remaining = &after_open[end + "</think>".len()..];
        } else {
            break;
        }
    }

    cleaned.trim().to_string()
}

pub fn strip_available_skills(text: &str) -> String {
    if let Some(end) = text.find("</available-skills>") {
        let after = &text[end + "</available-skills>".len()..];
        after.trim().to_string()
    } else {
        text.to_string()
    }
}

pub fn reasoning_summary_enabled() -> bool {
    std::env::var("MIVI_AGENT_REASONING_SUMMARY")
        .map(|value| !matches!(value.trim(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}

pub fn agent_reasoning_summary(
    req: &ChatCompletionRequest,
    user_prompt: &str,
    route: &str,
) -> Option<String> {
    if !reasoning_summary_enabled() {
        return None;
    }

    let selection = select_tools_for_request(req);
    let selected = tool_names(&selection.selected);
    let blocked = blocked_tool_names(&selection.blocked);
    let prompt_preview = trace_preview(user_prompt, 96);
    let selected_text = if selected.is_empty() {
        "no selected tools".to_string()
    } else {
        format!("selected tools: {}", selected.join(", "))
    };
    let blocked_text = if blocked.is_empty() {
        "no blocked tools".to_string()
    } else {
        format!("blocked: {}", blocked.join(", "))
    };

    Some(format!(
        "Classified request as {}; route {}; using agent-provided instructions and schemas; {}; {}; prompt: {}.",
        selection.intent.as_str(),
        route,
        selected_text,
        blocked_text,
        prompt_preview
    ))
}

pub fn classify_agent_intent(query: &str) -> AgentIntent {
    let q = query.to_ascii_lowercase();
    if q.trim().is_empty() {
        return AgentIntent::Chat;
    }

    let mentions_agent_subject = mentions_agent_subject(&q);
    let asks_inventory = q.contains("what")
        || q.contains("which")
        || q.contains("list")
        || q.contains("show")
        || q.contains("tell me")
        || q.contains("inventory")
        || q.contains("available")
        || q.contains("loaded")
        || q.contains("can this agent")
        || q.contains("can you do")
        || q.contains("can u do")
        || q.contains("able to do");
    let asks_capability_subject = mentions_agent_subject
        && (q.contains("do")
            || q.contains("handle")
            || q.contains("support")
            || q.contains("use")
            || q.contains("available"));

    if !asks_inventory && !asks_capability_subject {
        return AgentIntent::Chat;
    }

    if q.contains("use the") || q.contains("call the") || q.contains("run ") {
        return AgentIntent::ToolCall;
    }

    if q.contains("skill") || q.contains("skills") {
        AgentIntent::SkillInventory
    } else if q.contains("mcp") || q.contains("mcps") {
        AgentIntent::McpInventory
    } else if q.contains("tool") || q.contains("tools") {
        AgentIntent::ToolInventory
    } else if q.contains("feature")
        || q.contains("features")
        || q.contains("capabilit")
        || asks_capability_subject
    {
        AgentIntent::CapabilityInventory
    } else {
        AgentIntent::Chat
    }
}

pub fn text_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

pub fn mentions_agent_subject(query: &str) -> bool {
    text_tokens(query)
        .iter()
        .any(|token| matches!(token.as_str(), "agent" | "you" | "u" | "here"))
}

pub fn normalize_keyword_map(
    values: BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, Vec<String>> {
    values
        .into_iter()
        .map(|(key, values)| {
            let key = key.trim().to_ascii_lowercase();
            let mut values: Vec<String> = values
                .into_iter()
                .flat_map(|value| text_tokens(&value))
                .filter(|value| !value.is_empty())
                .collect();
            values.sort_unstable();
            values.dedup();
            (key, values)
        })
        .filter(|(key, _)| !key.is_empty())
        .collect()
}

pub fn normalize_marker_map(
    values: BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, Vec<String>> {
    values
        .into_iter()
        .map(|(key, values)| {
            let key = key.trim().to_ascii_lowercase();
            let values = normalize_marker_list(values);
            (key, values)
        })
        .filter(|(key, values)| !key.is_empty() && !values.is_empty())
        .collect()
}

pub fn normalize_marker_list(values: Vec<String>) -> Vec<String> {
    let mut values: Vec<String> = values
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect();
    values.sort_unstable();
    values.dedup();
    values
}

pub fn normalize_priority_list(values: Vec<String>) -> Vec<String> {
    values.into_iter().fold(Vec::new(), |mut acc, value| {
        let value = value.trim().to_ascii_lowercase();
        if !value.is_empty() && !acc.iter().any(|existing| existing == &value) {
            acc.push(value);
        }
        acc
    })
}

pub fn parse_capability_config(text: &str) -> Result<CapabilityConfig, serde_json::Error> {
    let mut config = serde_json::from_str::<CapabilityConfig>(text)?;
    config.aliases = normalize_keyword_map(config.aliases);
    config.tool_taxonomy = normalize_keyword_map(config.tool_taxonomy);
    config.tool_error_markers = normalize_marker_list(config.tool_error_markers);
    config.tool_salient_markers = normalize_marker_list(config.tool_salient_markers);
    config.tool_error_categories = normalize_marker_map(config.tool_error_categories);
    config.tool_error_category_priority =
        normalize_priority_list(config.tool_error_category_priority);
    Ok(config)
}

pub fn load_capability_config(path: &Path) -> CapabilityConfig {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| parse_capability_config(&text).ok())
        .unwrap_or_default()
}

pub fn capability_config() -> &'static CapabilityConfig {
    static CONFIG: OnceLock<CapabilityConfig> = OnceLock::new();
    CONFIG.get_or_init(|| load_capability_config(Path::new("configs/capabilities.json")))
}

pub fn extract_first_url(text: &str) -> Option<String> {
    text.split_whitespace()
        .find(|part| part.starts_with("http://") || part.starts_with("https://"))
        .map(|part| {
            part.trim_end_matches(|ch: char| matches!(ch, '.' | ',' | ')' | ']' | '}' | '!' | '?'))
                .to_string()
        })
        .filter(|url| !url.is_empty())
}

pub fn looks_like_research_request(text: &str) -> bool {
    let tokens = text_tokens(text);
    tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "research" | "read" | "docs" | "documentation" | "tell" | "about" | "summarize"
        )
    })
}

pub fn previous_user_url(req: &ChatCompletionRequest) -> Option<String> {
    let mut skipped_latest_user = false;
    for msg in req.messages.iter().rev() {
        if msg.role != "user" {
            continue;
        }
        if !skipped_latest_user {
            skipped_latest_user = true;
            continue;
        }
        let text = user_prompt_text_parts(msg)
            .into_iter()
            .map(|part| normalize_user_prompt_text(&part))
            .find(|part| !part.is_empty())
            .unwrap_or_default();
        if let Some(url) = extract_first_url(&text) {
            return Some(url);
        }
    }
    None
}

pub fn agent_decision_from_request(req: &ChatCompletionRequest) -> AgentDecision {
    let latest = latest_user_prompt_text(req);
    if let Some(url) = extract_first_url(&latest) {
        if looks_like_research_request(&latest) {
            return AgentDecision {
                intent: AgentDecisionIntent::WebResearch,
                subject: latest,
                url: Some(url),
            };
        }
    }

    if looks_like_research_request(&latest) {
        if let Some(url) = previous_user_url(req) {
            return AgentDecision {
                intent: AgentDecisionIntent::WebResearch,
                subject: latest,
                url: Some(url),
            };
        }
    }

    AgentDecision::chat(latest)
}

#[allow(dead_code)]
pub fn tool_matches_taxonomy(
    name: &str,
    description: &str,
    category: &str,
    config: &CapabilityConfig,
) -> bool {
    let Some(keywords) = config.tool_taxonomy.get(category) else {
        return false;
    };
    let haystack = format!(
        "{} {}",
        name.to_ascii_lowercase(),
        description.to_ascii_lowercase()
    );
    let tokens = text_tokens(&haystack);
    keywords
        .iter()
        .any(|keyword| tokens.iter().any(|token| token == keyword) || haystack.contains(keyword))
}

pub fn web_research_tool_score(tool: &ToolDef) -> isize {
    let config = capability_config();
    let positives = config
        .tool_taxonomy
        .get("web")
        .map(|keywords| {
            let schema = tool_schema_text(tool);
            let schema_tokens = text_tokens(&schema);
            keywords
                .iter()
                .filter(|keyword| {
                    schema_tokens
                        .iter()
                        .any(|schema_token| schema_token == *keyword)
                        || schema.contains(keyword.as_str())
                })
                .count() as isize
        })
        .unwrap_or(0);
    let local_penalty = config
        .tool_taxonomy
        .get("local_exclude")
        .map(|keywords| {
            let schema = tool_schema_text(tool);
            let schema_tokens = text_tokens(&schema);
            keywords
                .iter()
                .filter(|keyword| {
                    schema_tokens
                        .iter()
                        .any(|schema_token| schema_token == *keyword)
                        || schema.contains(keyword.as_str())
                })
                .count() as isize
        })
        .unwrap_or(0);

    positives * 4 - local_penalty * 3
}

pub fn select_web_research_tools(tools: &[ToolDef], max_tools: usize) -> Vec<ToolDef> {
    let mut scored: Vec<(usize, isize, ToolDef)> = tools
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, tool)| (idx, web_research_tool_score(&tool), tool))
        .filter(|(_, score, _)| *score > 0)
        .collect();
    scored.sort_by(|(left_idx, left_score, _), (right_idx, right_score, _)| {
        right_score
            .cmp(left_score)
            .then_with(|| left_idx.cmp(right_idx))
    });
    scored
        .into_iter()
        .take(max_tools)
        .map(|(_, _, tool)| tool)
        .collect()
}

pub fn tool_schema_text(tool: &ToolDef) -> String {
    format!(
        "{} {}",
        tool.function.name.to_ascii_lowercase(),
        tool.function
            .description
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase()
    )
}

pub fn classify_tool_role(tool: &ToolDef) -> ToolRole {
    let text = tool_schema_text(tool);
    let has_broad_inventory_signal = text.contains("capabilit") || text.contains("introspection");
    let has_inventory_signal = text.contains("inventory")
        || has_broad_inventory_signal
        || text.contains("available")
        || text.contains("registered")
        || text.contains("loaded");
    let has_diagnostic_signal = text.contains("diagnose")
        || text.contains("diagnostic")
        || text.contains("debug")
        || text.contains("troubleshoot")
        || text.contains("repair")
        || text.contains("fix");
    let has_management_signal = text.contains("manage")
        || text.contains("configure")
        || text.contains("configuration")
        || text.contains("start server")
        || text.contains("stop server")
        || text.contains("restart server");
    let has_action_signal = has_management_signal
        || text.contains("spawn")
        || text.contains("delegate")
        || text.contains("create subagent")
        || text.contains("run task")
        || text.contains("execute task");
    let has_resource_signal = text.contains("resource") || text.contains("template");
    let has_mcp_signal = text.contains("mcp");
    let has_skill_signal = text.contains("skill");
    let has_tool_signal =
        text.contains("tool") || text.contains("function") || text.contains("schema");

    if has_mcp_signal && has_resource_signal {
        ToolRole::McpResource
    } else if has_diagnostic_signal {
        ToolRole::Diagnostic
    } else if has_action_signal && !(text.contains("inventory") || text.contains("introspection")) {
        ToolRole::Action
    } else if has_broad_inventory_signal {
        ToolRole::Inventory
    } else if has_mcp_signal && has_inventory_signal {
        ToolRole::McpInventory
    } else if has_skill_signal && has_inventory_signal {
        ToolRole::SkillInventory
    } else if has_inventory_signal && has_tool_signal {
        ToolRole::Inventory
    } else if has_inventory_signal && !has_action_signal {
        ToolRole::Inventory
    } else if has_action_signal {
        ToolRole::Action
    } else {
        ToolRole::General
    }
}

pub fn is_resource_template_listing_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::McpResource
}

pub fn is_diagnostic_or_action_tool(tool: &ToolDef) -> bool {
    matches!(
        classify_tool_role(tool),
        ToolRole::Diagnostic | ToolRole::Action
    )
}

pub fn is_skill_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::SkillInventory
}

pub fn is_mcp_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::McpInventory
}

pub fn is_tool_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::Inventory
}

pub fn is_broad_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::Inventory
}

pub fn tool_is_inventory_for_intent(
    tool: &ToolDef,
    intent: AgentIntent,
    latest_user_prompt: &str,
) -> bool {
    match intent {
        AgentIntent::SkillInventory => {
            is_skill_inventory_tool(tool) || is_broad_inventory_tool(tool)
        }
        AgentIntent::McpInventory => is_mcp_inventory_tool(tool) || is_broad_inventory_tool(tool),
        AgentIntent::ToolInventory => is_tool_inventory_tool(tool) || is_broad_inventory_tool(tool),
        AgentIntent::CapabilityInventory => {
            let query = latest_user_prompt.to_ascii_lowercase();
            let wants_agent_scope = mentions_agent_subject(&query);
            is_broad_inventory_tool(tool)
                || (wants_agent_scope
                    && (is_tool_inventory_tool(tool)
                        || is_skill_inventory_tool(tool)
                        || is_mcp_inventory_tool(tool)))
        }
        AgentIntent::Chat | AgentIntent::ToolCall => false,
    }
}

pub fn inventory_block_reason(tool: &ToolDef, intent: AgentIntent) -> Option<&'static str> {
    if intent == AgentIntent::McpInventory && is_resource_template_listing_tool(tool) {
        return Some("mcp_resource_not_inventory");
    }
    if intent == AgentIntent::ToolInventory && is_skill_inventory_tool(tool) {
        return Some("skill_inventory_not_tool_inventory");
    }
    if is_diagnostic_or_action_tool(tool) {
        return Some("diagnostic_or_action_tool_not_inventory");
    }
    None
}

pub fn asks_agent_inventory(query: &str) -> bool {
    classify_agent_intent(query).is_inventory()
}

pub fn tool_is_inventory_for_query(tool: &ToolDef, query: &str) -> bool {
    let intent = classify_agent_intent(query);
    tool_is_inventory_for_intent(tool, intent, query)
}

#[allow(dead_code)]
pub fn tool_error_category_with_config(text: &str, config: &CapabilityConfig) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let mut categories = config.tool_error_category_priority.clone();
    for category in config.tool_error_categories.keys() {
        if !categories.iter().any(|existing| existing == category) {
            categories.push(category.clone());
        }
    }

    categories.into_iter().find(|category| {
        config
            .tool_error_categories
            .get(category)
            .map(|markers| markers.iter().any(|marker| lower.contains(marker)))
            .unwrap_or(false)
    })
}

pub async fn model_prompt_from_request(
    req: &ChatCompletionRequest,
    latest_user_prompt: &str,
    state: &AppState,
) -> String {
    let config = RuntimeConfig::from_env();
    let compressed = compress_context(&req.messages, config.context);
    let memories =
        tokio::task::spawn_blocking(|| load_memory_dir(Path::new("memory")).unwrap_or_default())
            .await
            .unwrap_or_default();
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

    let prompt = if pack.prompt.trim().is_empty() {
        render_context_prompt(&compressed, latest_user_prompt)
    } else {
        pack.prompt
    };

    wrap_agent_prompt(&agent_contract_prompt(req), &prompt)
}

pub fn image_url_to_path(url: &str) -> String {
    url.strip_prefix("file://").unwrap_or(url).to_string()
}

pub async fn vision_response(
    brain: &EdgeBrain,
    image_path: &str,
    user_prompt: &str,
) -> Result<String, String> {
    brain.query_vision(image_path, user_prompt).await
}

/// Extract the latest real user prompt + optional image path.
pub fn extract_content(req: &ChatCompletionRequest) -> (String, Option<String>) {
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
                        .map(image_url_to_path)
                })
            })
        });

    (user_prompt, image_path)
}

/// Parse function call JSON from model output.
/// Looks for `<tool_call>...JSON...</tool_call>` patterns and falls back to
/// bare `{"name": "...", "arguments": {...}}` JSON.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCallOut> {
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

pub fn parse_single_tool_call(json_str: &str) -> Option<ToolCallOut> {
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

#[cfg(test)]
pub fn parse_tool_calls_for_tools(text: &str, selected_tools: &[ToolDef]) -> Vec<ToolCallOut> {
    let parsed = parse_tool_calls(text);
    validate_tool_calls_for_tools(parsed, selected_tools).0
}

pub fn required_tool_args(tool: &ToolDef) -> Vec<String> {
    tool.function
        .parameters
        .as_ref()
        .and_then(|params| params.get("required"))
        .and_then(|required| required.as_array())
        .map(|required| {
            required
                .iter()
                .filter_map(|value| value.as_str())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

pub fn call_has_required_args(call: &ToolCallOut, tool: &ToolDef) -> bool {
    let required = required_tool_args(tool);
    if required.is_empty() {
        return true;
    }
    let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else {
        return false;
    };
    let Some(obj) = args.as_object() else {
        return false;
    };
    required.iter().all(|key| {
        obj.get(key)
            .map(|value| match value {
                serde_json::Value::Null => false,
                serde_json::Value::String(text) => !text.trim().is_empty(),
                _ => true,
            })
            .unwrap_or(false)
    })
}

pub fn validate_tool_calls_for_tools(
    calls: Vec<ToolCallOut>,
    selected_tools: &[ToolDef],
) -> (Vec<ToolCallOut>, usize) {
    let original_len = calls.len();
    let accepted: Vec<ToolCallOut> = calls
        .into_iter()
        .filter(|call| {
            selected_tools
                .iter()
                .find(|tool| tool.function.name == call.function.name)
                .map(|tool| call_has_required_args(call, tool))
                .unwrap_or(false)
        })
        .collect();
    let rejected = original_len.saturating_sub(accepted.len());
    (accepted, rejected)
}

pub fn tool_names(tools: &[ToolDef]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect()
}

pub fn call_names(calls: &[ToolCallOut]) -> Vec<String> {
    calls
        .iter()
        .map(|call| call.function.name.clone())
        .collect()
}

pub fn normalize_tool_arguments(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None => Some("{}".to_string()),
        Some(serde_json::Value::Object(_)) => value.map(|json| json.to_string()),
        Some(serde_json::Value::String(text)) => repair_tool_argument_string(text),
        _ => None,
    }
}

pub fn repair_tool_argument_string(text: &str) -> Option<String> {
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

pub fn strip_tagged_block_prefix<'a>(text: &'a str, tag: &str) -> &'a str {
    let close = format!("</{}>", tag);
    if text.trim_start().starts_with(&format!("<{}>", tag)) {
        if let Some(end) = text.find(&close) {
            return text[end + close.len()..].trim();
        }
    }
    text.trim()
}

pub fn normalize_user_prompt_text(text: &str) -> String {
    let mut normalized = text.trim();
    normalized = strip_tagged_block_prefix(normalized, "available-skills");
    normalized = strip_tagged_block_prefix(normalized, "skill-evaluation-required");
    normalized = strip_tagged_block_prefix(normalized, "user-prompt-submit-hook");
    normalized.trim().to_string()
}

pub fn user_prompt_text_parts(msg: &ChatMessage) -> Vec<String> {
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

pub fn latest_user_prompt_text(req: &ChatCompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .filter(|m| m.role == "user")
        .flat_map(|msg| user_prompt_text_parts(msg).into_iter().rev())
        .map(|text| normalize_user_prompt_text(&text))
        .find(|text| !text.is_empty())
        .unwrap_or_default()
}

pub fn latest_non_system_role(req: &ChatCompletionRequest) -> Option<&str> {
    req.messages
        .iter()
        .rev()
        .find(|msg| msg.role != "system")
        .map(|msg| msg.role.as_str())
}

pub fn explicitly_mentions_tool_name(user_text: &str, tool_name: &str) -> bool {
    let name = tool_name.trim().to_ascii_lowercase();
    if name.is_empty() {
        return false;
    }

    let is_specific_tool_name = name.contains('_') || name.contains('-') || name.len() >= 8;
    is_specific_tool_name && user_text.contains(&name)
}

/// Check if the current request asks MIVI to emit a tool call.
pub fn has_tool_involvement(req: &ChatCompletionRequest) -> bool {
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

    if latest_non_system_role(req) == Some("tool") {
        return false;
    }

    let tools = match &req.tools {
        Some(tools) if !tools.is_empty() => tools,
        _ => return false,
    };

    let decision = agent_decision_from_request(req);
    if decision.needs_tool() {
        let selected = select_web_research_tools(tools, 1);
        return !selected.is_empty();
    }

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

    if asks_agent_inventory(&user_text) {
        return tools
            .iter()
            .any(|tool| tool_is_inventory_for_query(tool, &user_text));
    }

    tools
        .iter()
        .any(|tool| explicitly_mentions_tool_name(&user_text, &tool.function.name))
        || !filter_tools(&user_text, tools, MAX_PROMPT_TOOLS).is_empty()
}

// ──────────────────────────────────────────────
// Backend model calls
// ──────────────────────────────────────────────

pub fn is_direct_reasoner_intent(intent: &str) -> bool {
    matches!(
        intent.to_ascii_lowercase().as_str(),
        "chat" | "reason" | "multi_step" | "vision"
    )
}

/// One-shot reasoner call (spawns llama-cli per request).
pub async fn reasoner_chat(
    brain: &EdgeBrain,
    user_prompt: &str,
) -> Result<(String, String), String> {
    let res = brain
        .query_reasoner(user_prompt, MIVI_CHAT_SYSTEM_PROMPT)
        .await?;
    Ok((res, MODEL_NAME.to_string()))
}

/// One-shot coder call (spawns llama-cli per request).
pub async fn code_chat(brain: &EdgeBrain, user_prompt: &str) -> Result<(String, String), String> {
    let res = brain
        .query_coder(user_prompt, "You are a coding expert.")
        .await?;
    Ok((res, MODEL_NAME.to_string()))
}

/// Run the model with a full multi-turn prompt (already formatted with <|im_start|> tags).
pub async fn model_chat(
    brain: &EdgeBrain,
    prompt: &str,
    req: &ChatCompletionRequest,
) -> Result<String, String> {
    brain
        .query_raw(
            prompt,
            req.temperature,
            req.top_p,
            req.max_tokens,
            req.stop.clone(),
            req.seed,
            extract_json_schema(req),
        )
        .await
}

/// Generate tool calls: run the model with tool-aware prompt, parse tool calls.
pub async fn generate_tool_calls(
    brain: &EdgeBrain,
    req: &ChatCompletionRequest,
) -> Result<(Vec<ToolCallOut>, String), String> {
    let trace = TraceConfig::from_env();
    let selection = select_tools_for_request(req);
    let selected_tools = selection.selected;
    let selected_tool_names = tool_names(&selected_tools);
    let selected_tool_roles = selected_tool_roles(&selected_tools);
    let blocked_tools = blocked_tool_names(&selection.blocked);

    let runtime_config = RuntimeConfig::from_env();
    if runtime_config.uses_worker() {
        if let Ok(messages_val) = serde_json::to_value(&req.messages) {
            let tools_val = req
                .tools
                .as_ref()
                .map(|t| serde_json::to_value(t).unwrap_or(serde_json::Value::Null));
            let choice_val = req.tool_choice.as_ref().cloned();
            let fmt_val = req.response_format.as_ref().cloned();

            match brain
                .text_worker
                .query_chat_full(
                    messages_val,
                    tools_val,
                    choice_val,
                    fmt_val,
                    req.temperature,
                    req.top_p,
                    req.max_tokens,
                    req.stop.clone(),
                    req.seed,
                    req.frequency_penalty,
                    req.presence_penalty,
                )
                .await
            {
                Ok(resp) => {
                    if let Some(choice) = resp
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                    {
                        if let Some(msg) = choice.get("message") {
                            let content = msg
                                .get("content")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            let tool_calls: Option<Vec<ToolCallOut>> = msg
                                .get("tool_calls")
                                .and_then(|tc| serde_json::from_value(tc.clone()).ok());

                            let _ = trace_event(
                                &trace,
                                serde_json::json!({
                                    "kind": "tool_generation",
                                    "route": "worker_tool",
                                    "agent_intent": selection.intent.as_str(),
                                    "selected_tools": selected_tool_names,
                                    "selected_tool_roles": selected_tool_roles,
                                    "blocked_tools": blocked_tools,
                                    "accepted_tool_calls": tool_calls.as_ref().map(|calls| call_names(calls)).unwrap_or_default(),
                                    "rejected_tool_calls": 0
                                }),
                            );

                            if let Some(calls) = tool_calls {
                                if !calls.is_empty() {
                                    return Ok((calls, String::new()));
                                }
                            }
                            return Ok((Vec::new(), content));
                        }
                    }
                }
                Err(err) => warn!("[MIVI-V2 Worker] ToolGen worker error: {}", err),
            }
        }
    }

    let prompt = build_chat_prompt(req);
    debug!("[MIVI-V2 ToolGen] Prompt length: {} chars", prompt.len());

    let raw = model_chat(brain, &prompt, req).await?;
    let debug_preview = trace_preview(&raw, 200);
    debug!(
        "[MIVI-V2 ToolGen] Raw model output (truncated): {:?}",
        debug_preview
    );

    // Parse, repair, and validate tool calls from output.
    let parsed_calls = parse_tool_calls(&raw);
    let parsed_count = parsed_calls.len();
    let (calls, rejected_tool_calls) = validate_tool_calls_for_tools(parsed_calls, &selected_tools);
    let _ = trace_event(
        &trace,
        serde_json::json!({
            "kind": "tool_generation",
            "route": "model_tool",
            "agent_intent": selection.intent.as_str(),
            "prompt_chars": prompt.chars().count(),
            "raw_preview": debug_preview,
            "selected_tools": selected_tool_names,
            "selected_tool_roles": selected_tool_roles,
            "blocked_tools": blocked_tools,
            "parsed_tool_calls": parsed_count,
            "accepted_tool_calls": call_names(&calls),
            "rejected_tool_calls": rejected_tool_calls
        }),
    );

    if calls.is_empty() {
        // No tool call detected — use raw output as content response.
        Ok((Vec::new(), raw))
    } else {
        Ok((calls, String::new()))
    }
}

// ──────────────────────────────────────────────
// Chat completions handler
// ──────────────────────────────────────────────

pub fn chat_error_response(now: u64, message: String) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-v2-{now}"),
        object: "chat.completion".to_string(),
        created: now,
        model: MODEL_NAME.to_string(),
        usage: None,
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: serde_json::json!({
                    "error": {
                        "type": "invalid_request_error",
                        "message": message
                    }
                })
                .to_string(),
                refusal: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    }
}

pub async fn complete_chat_non_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    now: u64,
) -> Result<ChatCompletionResponse, String> {
    if let Err(err) = validate_response_format(&req) {
        return Ok(chat_error_response(now, err));
    }

    let target_model = req.model.clone().unwrap_or_else(|| MODEL_NAME.to_string());
    let latest_user_prompt = latest_user_prompt_text(&req);

    if has_tool_involvement(&req) {
        let (tool_calls, response_text) = generate_tool_calls(&state.brain, &req).await?;
        if !tool_calls.is_empty() {
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_tool_calls(&req, &tool_calls)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: String::new(),
                        refusal: None,
                        reasoning_content: None,
                        tool_calls: Some(tool_calls),
                    },
                    logprobs: None,
                    finish_reason: "tool_calls".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            });
        } else {
            let final_text = response_text;
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_text(&req, &final_text)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: apply_response_format(final_text, &req).unwrap_or_else(|err| {
                            serde_json::json!({"error":{"type":"invalid_request_error","message":err}}).to_string()
                        }),
                        refusal: None,
                        reasoning_content: agent_reasoning_summary(
                            &req,
                            &latest_user_prompt,
                            "tool_text_fallback",
                        ),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            });
        }
    }

    let (user_prompt, image_path) = extract_content(&req);
    let model_user_prompt = model_prompt_from_request(&req, &user_prompt, &state).await;
    let (intent, _confidence) = state.router.classify_intent(&user_prompt);
    let (response_text, chosen_model, route) = if image_path.is_some() {
        let path = image_path.unwrap_or_default();
        (
            vision_response(&state.brain, &path, &user_prompt).await?,
            MODEL_NAME.to_string(),
            "vision",
        )
    } else {
        match target_model.to_lowercase().as_str() {
            "coder" => {
                let (text, model) = code_chat(&state.brain, &model_user_prompt).await?;
                (text, model, "coder")
            }
            "reasoner" => {
                let (text, model) = reasoner_chat(&state.brain, &model_user_prompt).await?;
                (text, model, "reasoner")
            }
            _ if is_direct_reasoner_intent(&intent) => {
                let (text, model) = reasoner_chat(&state.brain, &model_user_prompt).await?;
                (text, model, "direct_reasoner")
            }
            _ => {
                let (_, res) = state.orchestrator.execute_plan(&user_prompt).await;
                (res, MODEL_NAME.to_string(), "orchestrator")
            }
        }
    };

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-v2-{now}"),
        object: "chat.completion".to_string(),
        created: now,
        model: chosen_model,
        usage: Some(estimated_usage_for_text(&req, &response_text)),
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: apply_response_format(response_text, &req).unwrap_or_else(|err| {
                    serde_json::json!({"error":{"type":"invalid_request_error","message":err}})
                        .to_string()
                }),
                refusal: None,
                reasoning_content: agent_reasoning_summary(&req, &user_prompt, route),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    })
}

pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResponsesRequest>,
) -> axum::response::Response {
    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "type": "server_error",
                        "message": "Server is shutting down"
                    }
                })),
            )
                .into_response();
        }
    };
    let stream = req.stream.unwrap_or(false);
    let now = unix_timestamp();
    let chat_req = responses_request_to_chat_request(req);
    let include_usage = include_stream_usage(&chat_req);

    if stream {
        return handle_responses_streaming(state, chat_req, now, include_usage, permit)
            .await
            .into_response();
    }

    match complete_chat_non_stream(state, chat_req, now).await {
        Ok(chat) => {
            let response = responses_response_from_chat(chat);
            Json(response).into_response()
        }
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "type": "server_error",
                    "message": err
                }
            })),
        )
            .into_response(),
    }
}

pub async fn handle_responses_streaming(
    state: Arc<AppState>,
    chat_req: ChatCompletionRequest,
    now: u64,
    include_usage: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    let response_id = format!("resp-v2-{}", now);
    let prompt_tokens = token_counter().count_tokens(
        &chat_req
            .messages
            .last()
            .map(|m| m.content.as_str().unwrap_or("").to_string())
            .unwrap_or_default(),
    );

    let user_prompt = latest_user_prompt_text(&chat_req);
    let model_user_prompt = model_prompt_from_request(&chat_req, &user_prompt, &state).await;

    let brain = state.brain.clone();
    let system_prompt = wrap_agent_prompt(MIVI_CHAT_SYSTEM_PROMPT, "");
    let t = active_chat_template();
    let formatted = format!(
        "{}{}{}{}{}{}{}",
        t.system_prefix,
        system_prompt,
        t.system_suffix,
        t.user_prefix,
        model_user_prompt,
        t.user_suffix,
        t.assistant_start
    );

    let cli_path = brain.llama_cli.to_str().unwrap_or("llama-cli").to_string();
    let model_path = brain.llama_path.to_str().unwrap_or("").to_string();
    let runtime_config = RuntimeConfig::from_env();
    let streaming_context = std::env::var("MIVI_REASONER_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens >= 1024)
        .unwrap_or(runtime_config.context.max_input_tokens)
        .to_string();

    let temp_str = chat_req.temperature.unwrap_or(0.2).to_string();
    let top_p = chat_req.top_p;
    let max_tokens = chat_req.max_tokens;
    let stop = chat_req.stop.clone();
    let seed = chat_req.seed;
    let json_schema = extract_json_schema(&chat_req);

    tokio::spawn(async move {
        let mut spawn_rx = spawn_streaming(
            &cli_path,
            &model_path,
            &formatted,
            if brain.ultra_low_ram { "0" } else { "999" },
            &streaming_context,
            &temp_str,
            top_p,
            max_tokens,
            stop,
            seed,
            json_schema,
        );
        while let Some(token) = spawn_rx.recv().await {
            if tx.send(token).await.is_err() {
                break;
            }
        }
    });

    let completion_tokens = Arc::new(AtomicU32::new(0));
    let completion_tokens_for_stream = completion_tokens.clone();
    let response_id_for_created = response_id.clone();
    let response_id_for_completed = response_id.clone();

    let initial_events = vec![
        serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id_for_created,
                "object": "response",
                "created_at": now,
                "model": MODEL_NAME,
                "status": "in_progress"
            }
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": format!("out-{}", now),
                "type": "message",
                "status": "in_progress",
                "role": "assistant"
            }
        }),
        serde_json::json!({
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "text",
                "text": ""
            }
        }),
    ];

    let initial_stream = futures::stream::iter(
        initial_events
            .into_iter()
            .map(|val| Ok::<_, Infallible>(Event::default().data(val.to_string()))),
    );

    let token_stream = ReceiverStream::new(rx).map(move |token| {
        completion_tokens_for_stream
            .fetch_add(token_counter().count_tokens(&token), Ordering::Relaxed);
        let chunk = serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": token
        });
        Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
    });

    let final_stream = futures::stream::unfold(0, move |state| {
        let completion_tokens = completion_tokens.clone();
        let response_id = response_id_for_completed.clone();
        async move {
            match state {
                0 => {
                    let chunk = serde_json::json!({
                        "type": "response.content_part.done",
                        "output_index": 0,
                        "content_index": 0
                    });
                    Some((
                        Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                        1,
                    ))
                }
                1 => {
                    let chunk = serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0
                    });
                    Some((
                        Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                        2,
                    ))
                }
                2 => {
                    let mut completed_response = serde_json::json!({
                        "id": response_id,
                        "object": "response",
                        "created_at": now,
                        "model": MODEL_NAME,
                        "status": "completed",
                        "output": [{
                            "id": format!("out-{}", now),
                            "type": "message",
                            "status": "completed",
                            "role": "assistant"
                        }]
                    });
                    if include_usage {
                        let usage = UsageInfo::new(
                            prompt_tokens as u32,
                            completion_tokens.load(Ordering::Relaxed),
                        );
                        completed_response["usage"] = usage_value(usage);
                    }
                    let chunk = serde_json::json!({
                        "type": "response.completed",
                        "response": completed_response
                    });
                    Some((
                        Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                        3,
                    ))
                }
                _ => None,
            }
        }
    });

    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let stream = initial_stream
        .chain(token_stream)
        .chain(final_stream)
        .chain(done_marker)
        .map(move |item| {
            let _keep = &permit;
            item
        });

    Sse::new(stream)
}

pub async fn handle_chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if req.messages.len() > 256 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "Too many messages (max 256)"
                }
            })),
        )
            .into_response();
    }
    if req.tools.as_ref().map(|t| t.len()).unwrap_or(0) > 128 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "Too many tools (max 128)"
                }
            })),
        )
            .into_response();
    }

    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "type": "server_error",
                        "message": "Server is shutting down"
                    }
                })),
            )
                .into_response();
        }
    };

    debug!(
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
        debug!(
            "  msg[{}]: role={:?} content={}{}{}",
            i, msg.role, preview, tc, ti
        );
    }

    let trace = TraceConfig::from_env();
    let latest_user_prompt = latest_user_prompt_text(&req);
    let now = unix_timestamp();
    let include_usage = include_stream_usage(&req);
    let target_model = req.model.clone().unwrap_or_else(|| MODEL_NAME.to_string());
    let tool_selection = select_tools_for_request(&req);
    let selected_tool_names = tool_names(&tool_selection.selected);
    let selected_tool_roles = selected_tool_roles(&tool_selection.selected);
    let blocked_tools = blocked_tool_names(&tool_selection.blocked);
    let has_tools = has_tool_involvement(&req);
    let _ = trace_event(
        &trace,
        serde_json::json!({
            "kind": "request",
            "model": target_model,
            "stream": req.stream.unwrap_or(false),
            "messages": req.messages.len(),
            "tools_in_request": req.tools.as_ref().map(|tools| tools.len()).unwrap_or(0),
            "has_tool_involvement": has_tools,
            "agent_intent": tool_selection.intent.as_str(),
            "selected_tools": selected_tool_names,
            "selected_tool_roles": selected_tool_roles,
            "blocked_tools": blocked_tools,
            "latest_user_prompt_preview": trace_preview(&latest_user_prompt, 240)
        }),
    );

    // ── Tool calling path ────────────────────────────────────────────
    if has_tools {
        info!("[MIVI-V2 Tool] Tool involvement detected, generating tool calls...");
        let (tool_calls, response_text) = match generate_tool_calls(&state.brain, &req).await {
            Ok(res) => res,
            Err(err) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "type": "server_error",
                            "message": err
                        }
                    })),
                )
                    .into_response();
            }
        };

        if !tool_calls.is_empty() {
            info!("[MIVI-V2 Tool] Generated {} tool call(s)", tool_calls.len());
            for tc in &tool_calls {
                info!("  -> {}({})", tc.function.name, tc.function.arguments);
            }
            let _ = trace_event(
                &trace,
                serde_json::json!({
                    "kind": "final_response",
                    "route": "tool_calls",
                    "finish_reason": "tool_calls",
                    "tool_calls": call_names(&tool_calls)
                }),
            );
            let reasoning = agent_reasoning_summary(&req, &latest_user_prompt, "tool_calls");
            if req.stream.unwrap_or(false) {
                let usage =
                    include_usage.then(|| estimated_usage_for_tool_calls(&req, &tool_calls));
                return stream_tool_calls_response(tool_calls, now, reasoning, usage, permit)
                    .into_response();
            }
            return Json(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{}", now),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_tool_calls(&req, &tool_calls)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: String::new(),
                        refusal: None,
                        reasoning_content: reasoning,
                        tool_calls: Some(tool_calls),
                    },
                    logprobs: None,
                    finish_reason: "tool_calls".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            })
            .into_response();
        }

        // No tool calls — fall through to normal response with the text.
        let final_text = response_text;
        let chosen_model = MODEL_NAME.to_string();
        let _ = trace_event(
            &trace,
            serde_json::json!({
                "kind": "final_response",
                "route": "tool_text_fallback",
                "finish_reason": if req.stream.unwrap_or(false) { "stream" } else { "stop" },
                "response_chars": final_text.chars().count()
            }),
        );
        if req.stream.unwrap_or(false) {
            return stream_text_response(
                final_text.clone(),
                now,
                agent_reasoning_summary(&req, &latest_user_prompt, "tool_text_fallback"),
                include_usage.then(|| estimated_usage_for_text(&req, &final_text)),
                permit,
            )
            .into_response();
        }
        return Json(ChatCompletionResponse {
            id: format!("chatcmpl-v2-{}", now),
            object: "chat.completion".to_string(),
            created: now,
            model: chosen_model,
            usage: Some(estimated_usage_for_text(&req, &final_text)),
            choices: vec![ChoiceOut {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".to_string(),
                    content: final_text,
                    refusal: None,
                    reasoning_content: agent_reasoning_summary(
                        &req,
                        &latest_user_prompt,
                        "tool_text_fallback",
                    ),
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            system_fingerprint: Some("fp_mivi".to_string()),
        })
        .into_response();
    }

    // ── Non-tool path (existing logic) ───────────────────────────────
    let (user_prompt, image_path) = extract_content(&req);
    let model_user_prompt = model_prompt_from_request(&req, &user_prompt, &state).await;

    // Streaming path.
    if req.stream.unwrap_or(false) {
        if let Some(path) = image_path.as_deref() {
            let answer = match vision_response(&state.brain, path, &user_prompt).await {
                Ok(ans) => ans,
                Err(err) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "type": "server_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            };
            let _ = trace_event(
                &trace,
                serde_json::json!({
                    "kind": "final_response",
                    "route": "streaming_vision",
                    "finish_reason": "stream",
                    "response_chars": answer.chars().count()
                }),
            );
            return stream_text_response(
                answer.clone(),
                now,
                agent_reasoning_summary(&req, &user_prompt, "streaming_vision"),
                include_usage.then(|| estimated_usage_for_text(&req, &answer)),
                permit,
            )
            .into_response();
        }

        let _ = trace_event(
            &trace,
            serde_json::json!({
                "kind": "final_response",
                "route": "streaming",
                "finish_reason": "stream"
            }),
        );
        return handle_streaming(state, model_user_prompt, &req, now, include_usage, permit)
            .await
            .into_response();
    }

    // Non-streaming path.
    let (intent, confidence) = state.router.classify_intent(&user_prompt);
    info!(
        "[MIVI-V2 Server] Intent: {} (conf: {:.2}) | Model: '{}' | Prompt: '{}'",
        intent, confidence, target_model, user_prompt
    );

    let (response_text, chosen_model, route) = if image_path.is_some() {
        let path = image_path.unwrap_or_default();
        match vision_response(&state.brain, &path, &user_prompt).await {
            Ok(response) => (response, MODEL_NAME.to_string(), "vision"),
            Err(err) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "type": "server_error",
                            "message": err
                        }
                    })),
                )
                    .into_response();
            }
        }
    } else {
        match target_model.to_lowercase().as_str() {
            "coder" => match code_chat(&state.brain, &model_user_prompt).await {
                Ok((text, model)) => (text, model, "coder"),
                Err(err) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "type": "server_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            },
            "reasoner" => match reasoner_chat(&state.brain, &model_user_prompt).await {
                Ok((text, model)) => (text, model, "reasoner"),
                Err(err) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "type": "server_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            },
            _ if is_direct_reasoner_intent(&intent) => {
                match reasoner_chat(&state.brain, &model_user_prompt).await {
                    Ok((text, model)) => (text, model, "direct_reasoner"),
                    Err(err) => {
                        return (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": {
                                    "type": "server_error",
                                    "message": err
                                }
                            })),
                        )
                            .into_response();
                    }
                }
            }
            _ => {
                let (_, res) = state.orchestrator.execute_plan(&user_prompt).await;
                (res, MODEL_NAME.to_string(), "orchestrator")
            }
        }
    };
    let _ = trace_event(
        &trace,
        serde_json::json!({
            "kind": "final_response",
            "route": route,
            "intent": intent,
            "confidence": confidence,
            "model": chosen_model,
            "response_chars": response_text.chars().count(),
            "context_prompt_chars": model_user_prompt.chars().count()
        }),
    );

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-v2-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: chosen_model,
        usage: Some(estimated_usage_for_text(&req, &response_text)),
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: response_text,
                refusal: None,
                reasoning_content: agent_reasoning_summary(&req, &user_prompt, route),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    })
    .into_response()
}

pub fn tool_call_stream_chunks(
    id: String,
    created: u64,
    reasoning_content: Option<String>,
    tool_calls: &[ToolCallOut],
    usage: Option<UsageInfo>,
) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();

    if let Some(reasoning) = reasoning_content {
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": reasoning },
                "finish_reason": null
            }]
        }));
    }

    if !tool_calls.is_empty() {
        // Emit the metadata chunk with empty arguments delta
        let initial_calls: Vec<serde_json::Value> = tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                serde_json::json!({
                    "index": index,
                    "id": call.id,
                    "type": call.r#type,
                    "function": {
                        "name": call.function.name,
                        "arguments": ""
                    }
                })
            })
            .collect();
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": { "tool_calls": initial_calls },
                "finish_reason": null
            }]
        }));

        // Emit the arguments chunks in fragments
        let mut max_len = 0;
        for call in tool_calls {
            max_len = max_len.max(call.function.arguments.len());
        }

        let chunk_size = 12;
        let mut offset = 0;
        while offset < max_len {
            let mut arg_deltas = Vec::new();
            for (index, call) in tool_calls.iter().enumerate() {
                let args = &call.function.arguments;
                if offset < args.len() {
                    let end = (offset + chunk_size).min(args.len());
                    let delta = &args[offset..end];
                    arg_deltas.push(serde_json::json!({
                        "index": index,
                        "function": {
                            "arguments": delta
                        }
                    }));
                }
            }
            if !arg_deltas.is_empty() {
                chunks.push(serde_json::json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": { "tool_calls": arg_deltas },
                        "finish_reason": null
                    }]
                }));
            }
            offset += chunk_size;
        }
    }

    chunks.push(serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }));

    if let Some(usage) = usage {
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [],
            "usage": usage_value(usage)
        }));
    }

    chunks
}

pub fn stream_tool_calls_response(
    tool_calls: Vec<ToolCallOut>,
    created: u64,
    reasoning_content: Option<String>,
    usage: Option<UsageInfo>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-v2-{}", created);
    let chunks = tool_call_stream_chunks(id, created, reasoning_content, &tool_calls, usage);
    let chunk_stream = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk.to_string()))),
    );
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    let mapped_stream = chunk_stream.chain(done_marker).map(move |item| {
        let _keep = &permit;
        item
    });
    Sse::new(mapped_stream)
}

pub fn text_stream_chunks(
    id: String,
    created: u64,
    reasoning_content: Option<String>,
    content: String,
    usage: Option<UsageInfo>,
) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();

    if let Some(reasoning) = reasoning_content {
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": reasoning },
                "finish_reason": null
            }]
        }));
    }

    chunks.push(serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": null
        }]
    }));

    chunks.push(serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }));

    if let Some(usage) = usage {
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [],
            "usage": usage_value(usage)
        }));
    }

    chunks
}

pub fn stream_text_response(
    content: String,
    created: u64,
    reasoning_content: Option<String>,
    usage: Option<UsageInfo>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-v2-{}", created);
    let chunks = text_stream_chunks(id, created, reasoning_content, content, usage);
    let chunk_stream = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk.to_string()))),
    );
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    let mapped_stream = chunk_stream.chain(done_marker).map(move |item| {
        let _keep = &permit;
        item
    });
    Sse::new(mapped_stream)
}

// ──────────────────────────────────────────────
// SSE streaming handler
// ──────────────────────────────────────────────

/// SSE streaming handler — spawns llama-cli per request, sends tokens as
/// they arrive from stdout, then emits a final `finish_reason: stop` chunk
/// and a `[DONE]` sentinel per the OpenAI streaming spec.
pub async fn handle_streaming(
    state: Arc<AppState>,
    user_prompt: String,
    req: &ChatCompletionRequest,
    created: u64,
    include_usage: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    let id = format!("chatcmpl-v2-{}", created);
    let completion_tokens = Arc::new(AtomicU32::new(0));

    // Clone what we need for the background task.
    let brain = state.brain.clone();
    let system_prompt = wrap_agent_prompt(MIVI_CHAT_SYSTEM_PROMPT, "");
    let t = active_chat_template();
    let formatted = format!(
        "{}{}{}{}{}{}{}",
        t.system_prefix,
        system_prompt,
        t.system_suffix,
        t.user_prefix,
        user_prompt,
        t.user_suffix,
        t.assistant_start
    );

    let cli_path = brain.llama_cli.to_str().unwrap_or("llama-cli").to_string();
    let model_path = brain.llama_path.to_str().unwrap_or("").to_string();
    let runtime_config = RuntimeConfig::from_env();
    let streaming_context = std::env::var("MIVI_REASONER_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens >= 1024)
        .unwrap_or(runtime_config.context.max_input_tokens)
        .to_string();

    let temp_str = req.temperature.unwrap_or(0.2).to_string();
    let top_p = req.top_p;
    let max_tokens = req.max_tokens;
    let stop = req.stop.clone();
    let seed = req.seed;
    let json_schema = extract_json_schema(req);

    let uses_worker = runtime_config.uses_worker();
    let text_worker = brain.text_worker.clone();
    let req_temp = req.temperature;
    let req_top_p = req.top_p;
    let req_max_tokens = req.max_tokens;
    let req_stop = req.stop.clone();
    let req_seed = req.seed;
    let req_fp = req.frequency_penalty;
    let req_pp = req.presence_penalty;

    let fallback_user_prompt = user_prompt.clone();
    tokio::spawn(async move {
        let mut emitted = false;
        if uses_worker {
            match text_worker
                .query_completion_stream(
                    &formatted,
                    req_temp,
                    req_top_p,
                    req_max_tokens,
                    req_stop,
                    req_seed,
                    req_fp,
                    req_pp,
                )
                .await
            {
                Ok(bytes_stream) => {
                    use futures::stream::StreamExt;
                    let mut stream = Box::pin(bytes_stream);
                    let mut buffer = Vec::new();
                    while let Some(chunk_res) = stream.next().await {
                        let chunk: bytes::Bytes = match chunk_res {
                            Ok(c) => c,
                            Err(err) => {
                                error!("Error reading stream chunk from worker: {}", err);
                                break;
                            }
                        };
                        buffer.extend_from_slice(&chunk);
                        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                            let line_bytes: Vec<u8> = buffer.drain(..pos + 1).collect();
                            if let Ok(mut line) = String::from_utf8(line_bytes) {
                                if line.ends_with('\n') {
                                    line.pop();
                                }
                                if line.ends_with('\r') {
                                    line.pop();
                                }
                                if line.starts_with("data: ") {
                                    let data = &line["data: ".len()..];
                                    if data == "[DONE]" {
                                        break;
                                    }
                                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data)
                                    {
                                        if let Some(token) =
                                            val.get("content").and_then(|c| c.as_str())
                                        {
                                            if !token.is_empty() {
                                                emitted = true;
                                                if tx.send(token.to_string()).await.is_err() {
                                                    return; // Client disconnected
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(err) => {
                    error!("Failed to query completion stream from worker: {}", err);
                }
            }
        } else {
            let mut rx = spawn_streaming(
                &cli_path,
                &model_path,
                &formatted,
                if brain.ultra_low_ram { "0" } else { "999" },
                &streaming_context,
                &temp_str,
                top_p,
                max_tokens,
                stop,
                seed,
                json_schema,
            );

            while let Some(token) = rx.recv().await {
                if !token.trim().is_empty() {
                    emitted = true;
                }
                if tx.send(token).await.is_err() {
                    return; // Receiver dropped (client disconnected).
                }
            }
        }

        if !emitted {
            let fallback_prompt = wrap_agent_prompt(MIVI_CHAT_SYSTEM_PROMPT, &fallback_user_prompt);
            if let Ok(fallback) = brain
                .query_reasoner(&fallback_prompt, MIVI_CHAT_SYSTEM_PROMPT)
                .await
            {
                let fallback = fallback.trim();
                if !fallback.is_empty() {
                    let _ = tx.send(fallback.to_string()).await;
                }
            }
        }
    });

    let reasoning_chunk = futures::stream::iter(
        reasoning_summary_enabled()
            .then(|| {
                format!(
                    "Classified request as chat; route streaming; using agent-provided instructions and context; prompt: {}.",
                    trace_preview(&user_prompt, 96)
                )
            })
            .into_iter()
            .map({
                let id = id.clone();
                move |reasoning| {
                    let chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": { "reasoning_content": reasoning },
                            "finish_reason": null
                        }]
                    });
                    Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
                }
            }),
    );

    // Token events: one SSE chunk per received token.
    let id_for_tokens = id.clone();
    let completion_tokens_for_stream = completion_tokens.clone();
    let token_stream = ReceiverStream::new(rx).map(move |token| {
        completion_tokens_for_stream
            .fetch_add(token_counter().count_tokens(&token), Ordering::Relaxed);
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

    let usage_chunk = futures::stream::iter(include_usage.then({
        let id = id.clone();
        let prompt_tokens = token_counter().count_tokens(&user_prompt);
        let completion_tokens = completion_tokens.clone();
        move || {
            let usage = UsageInfo::new(prompt_tokens, completion_tokens.load(Ordering::Relaxed));
            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [],
                "usage": usage_value(usage)
            });
            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        }
    }));

    // [DONE] sentinel per OpenAI spec.
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let stream = reasoning_chunk
        .chain(token_stream)
        .chain(final_chunk)
        .chain(usage_chunk)
        .chain(done_marker)
        .map(move |item| {
            let _keep = &permit;
            item
        });
    Sse::new(stream)
}

pub async fn start_api_server(
    brain: EdgeBrain,
    orchestrator: AgentOrchestrator,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let max_concurrent = std::env::var("MIVI_MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let state = Arc::new(AppState {
        brain,
        orchestrator,
        router: NeedleRouter::new(),
        semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
    });

    let app = Router::new()
        .route("/", get(handle_root))
        .route("/v1/models", get(handle_models))
        .route("/v1/health", get(handle_health))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/responses", post(handle_responses))
        .layer(CorsLayer::permissive())
        .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024)) // limit payload to 16MB
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        error!(
            "❌ Failed to bind to {}: {}. Is the port already in use?",
            addr, e
        );
        e
    })?;
    info!(
        "🚀 MIVI-V2 High-Speed Server listening on http://{} ...",
        addr
    );
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    pub fn tool_request(
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
            max_tokens: None,
            stop: None,
            seed: None,
            response_format: None,
            stream_options: None,
            parallel_tool_calls: None,
            reasoning_effort: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            user: None,
        }
    }

    pub fn server_tool(name: &str, description: &str) -> ToolDef {
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
    pub fn chat_request_accepts_openai_compatibility_fields() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "mivi",
            "messages": [{"role":"user", "content":"hi"}],
            "max_tokens": 32,
            "stop": ["END"],
            "seed": 7,
            "parallel_tool_calls": false,
            "reasoning_effort": "low",
            "stream_options": {"include_usage": true},
            "response_format": {"type":"json_object"}
        }))
        .expect("request should parse compatibility fields");

        assert_eq!(req.max_tokens, Some(32));
        assert_eq!(req.seed, Some(7));
        assert_eq!(req.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(req.parallel_tool_calls, Some(false));
        assert_eq!(req.stop, Some(json!(["END"])));
        assert_eq!(req.stream_options.as_ref().unwrap()["include_usage"], true);
        assert_eq!(response_format_type(&req), Some("json_object".to_string()));
    }

    #[test]
    pub fn responses_request_passes_compatibility_fields_to_chat_request() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "mivi",
            "input": "hello",
            "max_output_tokens": 64,
            "parallel_tool_calls": true,
            "response_format": {"type":"json_object"},
            "stream_options": {"include_usage": true},
            "reasoning": {"effort":"medium"}
        }))
        .expect("responses request should parse compatibility fields");

        let chat = responses_request_to_chat_request(req);

        assert_eq!(chat.max_tokens, Some(64));
        assert_eq!(chat.parallel_tool_calls, Some(true));
        assert_eq!(chat.reasoning_effort.as_deref(), Some("medium"));
        assert!(include_stream_usage(&chat));
        assert_eq!(response_format_type(&chat), Some("json_object".to_string()));
    }

    #[test]
    pub fn json_response_format_wraps_verified_answer_as_json() {
        let mut req = tool_request("what ai model are you", None);
        req.tools = None;
        req.response_format = Some(json!({"type":"json_object"}));

        let answer = apply_response_format(
            "I am MIVI, exposed to agents as the local OpenAI-compatible model `mivi`.".to_string(),
            &req,
        )
        .expect("json format should be supported");
        let parsed: serde_json::Value = serde_json::from_str(&answer).expect("valid json");

        assert!(parsed["answer"].as_str().unwrap().contains("MIVI"));
        assert!(parsed["answer"].as_str().unwrap().contains("`mivi`"));
    }

    #[test]
    pub fn strict_json_schema_response_format_is_accepted() {
        let mut req = tool_request("what ai model are you", None);
        req.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "response": { "type": "string" }
                    }
                }
            }
        }));

        assert!(validate_response_format(&req).is_ok());
        let schema = extract_json_schema(&req).expect("schema extraction");
        assert!(schema.contains("properties"));
    }

    #[test]
    pub fn llama_tokenize_output_counts_token_ids_without_prompt_metadata() {
        let output = "main: tokenizing prompt\n[1, 29871, 15043, 13]\n[ Prompt: 4 tokens ]";

        assert_eq!(count_llama_tokenize_output(output), Some(4));
    }

    #[test]
    pub fn token_counter_config_uses_catalog_reasoner_when_model_env_missing() {
        let catalog = crate::model_catalog::ModelCatalog::from_json(
            r#"{
              "external_model": "mivi",
              "models": [
                {
                  "id": "reasoner",
                  "role": "reasoner",
                  "backend": "llama-cli",
                  "path": "models/catalog-reasoner.gguf",
                  "context_tokens": 4096,
                  "ram_mb_estimate": 512,
                  "enabled": true
                }
              ]
            }"#,
        )
        .expect("catalog should parse");

        let config =
            TokenCounterConfig::from_sources(Some("llama-tokenize"), None, None, Some(&catalog));

        assert_eq!(
            config.backend,
            TokenCounterBackend::LlamaCpp {
                command: PathBuf::from("llama-tokenize"),
                model: PathBuf::from("models/catalog-reasoner.gguf"),
            }
        );
    }

    #[test]
    pub fn token_counter_config_prefers_explicit_tokenizer_model_over_catalog() {
        let catalog = crate::model_catalog::ModelCatalog::from_json(
            r#"{
              "external_model": "mivi",
              "models": [
                {
                  "id": "reasoner",
                  "role": "reasoner",
                  "backend": "llama-cli",
                  "path": "models/catalog-reasoner.gguf",
                  "context_tokens": 4096,
                  "ram_mb_estimate": 512,
                  "enabled": true
                }
              ]
            }"#,
        )
        .expect("catalog should parse");

        let config = TokenCounterConfig::from_sources(
            Some("llama-tokenize"),
            Some("models/explicit-tokenizer.gguf"),
            Some("models/env-reasoner.gguf"),
            Some(&catalog),
        );

        assert_eq!(
            config.backend,
            TokenCounterBackend::LlamaCpp {
                command: PathBuf::from("llama-tokenize"),
                model: PathBuf::from("models/explicit-tokenizer.gguf"),
            }
        );
    }

    #[test]
    pub fn token_counter_config_uses_external_backend_only_when_command_and_model_exist() {
        let configured = TokenCounterConfig::from_sources(
            Some("llama-tokenize"),
            Some("models/a.gguf"),
            None,
            None,
        );
        let fallback = TokenCounterConfig::from_sources(Some("llama-tokenize"), None, None, None);

        assert!(matches!(
            configured.backend,
            TokenCounterBackend::LlamaCpp { .. }
        ));
        assert!(matches!(fallback.backend, TokenCounterBackend::Cheap));
    }

    #[test]
    pub fn cheap_token_counter_remains_fallback_for_plain_text() {
        let counter = TokenCounterConfig::default().counter();

        assert_eq!(counter.count_tokens("hello, world!"), 4);
    }

    #[test]
    pub fn estimated_usage_counts_prompt_completion_and_total_tokens() {
        let req = tool_request("hello world", None);

        let usage = estimated_usage_for_text(&req, "hi there");

        assert!(usage.prompt_tokens >= 2);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    #[test]
    pub fn non_stream_chat_response_serializes_openai_usage() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: MODEL_NAME.to_string(),
            choices: vec![ChoiceOut {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".to_string(),
                    content: "hello".to_string(),
                    refusal: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: Some(UsageInfo::new(3, 1)),
            system_fingerprint: Some("fp_mivi".to_string()),
        };

        let value = serde_json::to_value(response).expect("serializable response");

        assert_eq!(value["usage"]["prompt_tokens"], 3);
        assert_eq!(value["usage"]["completion_tokens"], 1);
        assert_eq!(value["usage"]["total_tokens"], 4);
    }

    #[test]
    pub fn responses_response_carries_chat_usage() {
        let chat = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: MODEL_NAME.to_string(),
            choices: vec![ChoiceOut {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".to_string(),
                    content: "streamed text".to_string(),
                    refusal: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: Some(UsageInfo::new(4, 2)),
            system_fingerprint: Some("fp_mivi".to_string()),
        };

        let response = responses_response_from_chat(chat);

        assert_eq!(response.usage.unwrap().total_tokens, 6);
    }

    #[test]
    pub fn responses_string_input_maps_to_chat_request() {
        let req = ResponsesRequest {
            model: Some(MODEL_NAME.to_string()),
            input: ResponsesInput::Text("hello from responses".to_string()),
            stream: None,
            tools: None,
            tool_choice: None,
            max_output_tokens: None,
            stop: None,
            seed: None,
            response_format: None,
            stream_options: None,
            parallel_tool_calls: None,
            reasoning_effort: None,
            reasoning: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            user: None,
        };

        let chat = responses_request_to_chat_request(req);

        assert_eq!(chat.model.as_deref(), Some(MODEL_NAME));
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[0].content, json!("hello from responses"));
    }

    #[test]
    pub fn responses_message_array_input_maps_to_chat_request() {
        let req = ResponsesRequest {
            model: Some(MODEL_NAME.to_string()),
            input: ResponsesInput::Messages(vec![ResponsesInputMessage {
                role: "user".to_string(),
                content: json!([{"type":"input_text","text":"research this"}]),
            }]),
            stream: Some(false),
            tools: Some(vec![server_tool("webfetch", "Fetch a URL from the web")]),
            tool_choice: None,
            max_output_tokens: None,
            stop: None,
            seed: None,
            response_format: None,
            stream_options: None,
            parallel_tool_calls: None,
            reasoning_effort: None,
            reasoning: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            user: None,
        };

        let chat = responses_request_to_chat_request(req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(
            chat.messages[0].content,
            json!([{"type":"text","text":"research this"}])
        );
        assert!(chat.tools.as_ref().unwrap()[0].function.name == "webfetch");
    }

    #[test]
    pub fn chat_response_maps_to_responses_output_text() {
        let chat = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: MODEL_NAME.to_string(),
            usage: None,
            choices: vec![ChoiceOut {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".to_string(),
                    content: "answer text".to_string(),
                    refusal: None,
                    reasoning_content: Some("summary".to_string()),
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            system_fingerprint: Some("fp_mivi".to_string()),
        };

        let response = responses_response_from_chat(chat);

        assert_eq!(response.object, "response");
        assert_eq!(response.status, "completed");
        assert_eq!(response.output[0].r#type, "message");
        assert_eq!(response.output[0].content[0].r#type, "output_text");
        assert_eq!(response.output[0].content[0].text, "answer text");
    }

    #[test]
    pub fn chat_prompt_injects_agent_contract_with_tool_summary() {
        let mut req = tool_request("what can this agent do", None);
        req.tools = Some(vec![
            server_tool(
                "agent_capabilities",
                "Introspection: available tools and skills",
            ),
            server_tool("read", "Read files"),
        ]);

        let prompt = build_chat_prompt(&req);

        assert!(prompt.contains("Agent contract:"));
        assert!(prompt.contains("External model identity is `mivi`"));
        assert!(prompt.contains(
            "Current prompt exposes 1 selected callable tool schemas: agent_capabilities"
        ));
        assert!(prompt.contains("Function: agent_capabilities"));
        assert!(!prompt.contains("Function: read"));
    }

    #[test]
    pub fn chat_prompt_wraps_calling_agent_system_prompt() {
        let mut req = tool_request("hello", None);
        req.tools = None;
        req.messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: json!("You are the calling agent. Use its policies."),
                tool_call_id: None,
                tool_calls: None,
            },
        );

        let prompt = build_chat_prompt(&req);

        assert!(prompt.contains("Agent contract:"));
        assert!(prompt.contains("You are the calling agent. Use its policies."));
        assert!(prompt.contains("Current prompt exposes no selected callable tool schemas."));
    }

    #[test]
    pub fn tool_prompt_filters_irrelevant_opencode_tools() {
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
    pub fn tools_available_does_not_force_tool_generation_for_plain_chat() {
        let req = tool_request("hi", None);
        assert!(!has_tool_involvement(&req));
    }

    #[test]
    pub fn code_capability_question_does_not_enter_tool_generation() {
        let mut req = tool_request("so is u can write codes", None);
        req.tools = Some(vec![server_tool("write", "Write a file to the workspace")]);

        assert!(!has_tool_involvement(&req));
    }

    pub fn server_tool_with_params(
        name: &str,
        description: &str,
        properties: serde_json::Value,
    ) -> ToolDef {
        ToolDef {
            function: FunctionDef {
                name: name.to_string(),
                description: Some(description.to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": properties,
                })),
            },
            r#type: "function".to_string(),
        }
    }

    #[test]
    pub fn tool_taxonomy_classifies_web_and_file_tools_from_config() {
        let config = parse_capability_config(
            r#"{
                "aliases": {},
                "tool_taxonomy": {
                    "web": ["web", "url", "browser"],
                    "file": ["file", "path", "workspace"]
                },
                "tool_error_markers": ["error", "failed"],
                "tool_salient_markers": ["error", "failed", "status"]
            }"#,
        )
        .expect("valid capability config");

        assert!(tool_matches_taxonomy(
            "webfetch",
            "Fetch a URL from the web",
            "web",
            &config
        ));
        assert!(tool_matches_taxonomy(
            "read_file",
            "Read a workspace path",
            "file",
            &config
        ));
        assert!(!tool_matches_taxonomy(
            "read_file",
            "Read a workspace path",
            "web",
            &config
        ));
    }

    #[test]
    pub fn tool_call_missing_required_argument_is_rejected() {
        let tool = server_tool_with_params(
            "search_and_read",
            "Search and read web pages",
            json!({"url": {"type": "string"}, "query": {"type": "string"}}),
        );
        let mut tool = tool;
        tool.function.parameters = Some(json!({
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "query": {"type": "string"}
            },
            "required": ["url", "query"]
        }));
        let raw = r#"<tool_call>{"name":"search_and_read","arguments":{"url":"https://hono.dev/"}}</tool_call>"#;

        let calls = parse_tool_calls_for_tools(raw, &[tool]);

        assert!(calls.is_empty());
    }

    #[test]
    pub fn url_research_request_enters_tool_generation_and_selects_web_tool() {
        let mut req = tool_request("so https://hono.dev/ research about this and tell me", None);
        req.tools = Some(vec![
            server_tool_with_params(
                "searchxyz_search_and_read",
                "Search and read web pages from the internet",
                json!({"query": {"type": "string"}, "url": {"type": "string"}}),
            ),
            server_tool("bash", "Run shell commands"),
            server_tool("read_file", "Read a workspace file"),
        ]);

        assert!(has_tool_involvement(&req));
        let selected = prompt_tools_for_request(&req);
        assert_eq!(tool_names(&selected), vec!["searchxyz_search_and_read"]);
    }

    #[test]
    pub fn agent_intent_classifies_inventory_queries_without_platform_phrases() {
        assert_eq!(
            classify_agent_intent("what tools are available here"),
            AgentIntent::ToolInventory
        );
        assert_eq!(
            classify_agent_intent("which MCP servers can this agent use"),
            AgentIntent::McpInventory
        );
        assert_eq!(
            classify_agent_intent("list skills loaded for this task"),
            AgentIntent::SkillInventory
        );
        assert_eq!(classify_agent_intent("1+1 is what"), AgentIntent::Chat);
    }

    #[test]
    pub fn tool_selection_trace_blocks_action_tools_for_tool_inventory() {
        let mut req = tool_request("what tools are available here", None);
        req.tools = Some(vec![
            server_tool("spawn_agent", "Create or delegate to a subagent"),
            server_tool("get_available_skills", "List available skills"),
            server_tool(
                "agent_capabilities",
                "Return inventory of available tools and features",
            ),
        ]);

        let selection = select_tools_for_request(&req);

        assert_eq!(selection.intent, AgentIntent::ToolInventory);
        assert_eq!(tool_names(&selection.selected), vec!["agent_capabilities"]);
        assert!(selection
            .blocked
            .iter()
            .any(|blocked| blocked.name == "spawn_agent"));
        assert!(selection
            .blocked
            .iter()
            .any(|blocked| blocked.name == "get_available_skills"));
    }

    #[test]
    pub fn mcp_inventory_selection_blocks_resource_template_tools() {
        let mut req = tool_request("which MCP servers can this agent use", None);
        req.tools = Some(vec![
            server_tool("list_mcp_resource_templates", "List MCP resource templates"),
            server_tool("list_mcp_resources", "List MCP resources"),
            server_tool("mcp_inventory", "Inventory available MCP servers"),
        ]);

        let selection = select_tools_for_request(&req);

        assert_eq!(selection.intent, AgentIntent::McpInventory);
        assert_eq!(tool_names(&selection.selected), vec!["mcp_inventory"]);
        assert_eq!(selection.blocked.len(), 2);
    }

    #[test]
    pub fn assistant_think_history_is_not_reprompted_to_model() {
        let mut req = tool_request("so whats new", None);
        req.tools = None;
        req.messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: json!("hii"),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: json!(
                    "<think>Classified request as chat; route streaming.</think>

Hello!"
                ),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: json!("so whats new"),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let prompt = build_chat_prompt(&req);

        assert!(!prompt.contains("<think>"));
        assert!(!prompt.contains("Classified request as chat"));
        assert!(prompt.contains("Hello!"));
    }

    #[test]
    pub fn inventory_questions_do_not_call_server_management_tools() {
        let mut req = tool_request("what are the mcps u have", None);
        req.tools = Some(vec![
            server_tool(
                "manage_servers",
                "Manage configured MCP servers and available capabilities",
            ),
            server_tool("read", "Read files"),
            server_tool("bash", "Run shell commands"),
        ]);

        assert_eq!(
            classify_tool_role(&server_tool(
                "manage_servers",
                "Manage configured MCP servers and available capabilities",
            )),
            ToolRole::Action
        );
        assert!(!has_tool_involvement(&req));
        assert!(prompt_tools_for_request(&req).is_empty());
    }

    #[test]
    pub fn tool_role_classifier_separates_inventory_from_diagnostics_and_actions() {
        assert_eq!(
            classify_tool_role(&server_tool(
                "agent_capabilities",
                "Introspection: return available tools, skills, subagents, and runtime capabilities",
            )),
            ToolRole::Inventory
        );
        assert_eq!(
            classify_tool_role(&server_tool(
                "diagnose_tool",
                "Diagnose tool selection and available capability failures",
            )),
            ToolRole::Diagnostic
        );
        assert_eq!(
            classify_tool_role(&server_tool(
                "delegate_task",
                "Delegate work to a specialized subagent",
            )),
            ToolRole::Action
        );
        assert_eq!(
            classify_tool_role(&server_tool(
                "list_mcp_resource_templates",
                "List MCP resource templates",
            )),
            ToolRole::McpResource
        );
    }

    #[test]
    pub fn agent_reasoning_summary_is_safe_for_openz_thought_ui() {
        let mut req = tool_request("what can this agent do", None);
        req.tools = Some(vec![server_tool(
            "agent_capabilities",
            "Introspection: return available tools and skills",
        )]);

        let summary = agent_reasoning_summary(&req, "what can this agent do", "verified_tools")
            .expect("reasoning summary expected");

        assert!(summary.contains("capability_inventory"));
        assert!(summary.contains("agent-provided"));
        assert!(!summary.contains("<think>"));
        assert!(!summary.to_ascii_lowercase().contains("private"));
    }

    #[test]
    pub fn opencode_injected_skill_context_does_not_force_tool_generation() {
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
    pub fn opencode_skill_evaluation_context_does_not_hide_latest_array_prompt() {
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
    pub fn extract_content_uses_latest_real_opencode_prompt() {
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
    pub fn extract_content_returns_image_path_from_multimodal_request() {
        let mut req = tool_request("Describe image", None);
        req.messages = vec![ChatMessage {
            role: "user".to_string(),
            content: json!([
                {"type":"text","text":"Describe image"},
                {"type":"image_url","image_url":{"url":"/tmp/screenshot.png"}}
            ]),
            tool_call_id: None,
            tool_calls: None,
        }];

        let (prompt, image_path) = extract_content(&req);
        assert_eq!(prompt, "Describe image");
        assert_eq!(image_path, Some("/tmp/screenshot.png".to_string()));
    }

    #[test]
    pub fn extract_content_normalizes_file_image_urls() {
        let mut req = tool_request("Describe image", None);
        req.messages = vec![ChatMessage {
            role: "user".to_string(),
            content: json!([
                {"type":"text","text":"Describe image"},
                {"type":"image_url","image_url":{"url":"file:///tmp/screenshot.png"}}
            ]),
            tool_call_id: None,
            tool_calls: None,
        }];

        let (_, image_path) = extract_content(&req);
        assert_eq!(image_path, Some("/tmp/screenshot.png".to_string()));
    }

    #[test]
    pub fn lowercase_chat_intent_uses_direct_reasoner_path() {
        assert!(is_direct_reasoner_intent("chat"));
        assert!(is_direct_reasoner_intent("reason"));
        assert!(is_direct_reasoner_intent("multi_step"));
        assert!(is_direct_reasoner_intent("VISION"));
        assert!(!is_direct_reasoner_intent("code"));
    }

    #[test]
    pub fn mivi_identity_prompt_names_external_and_internal_models() {
        assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("model name is mivi"));
        assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("Never identify as"));
    }

    #[test]
    pub fn explicit_tool_request_enters_tool_generation() {
        let req = tool_request("Use the get_weather tool for Paris", None);
        assert!(has_tool_involvement(&req));
    }

    #[test]
    pub fn required_tool_choice_enters_tool_generation() {
        let req = tool_request("weather in Paris", Some(json!("required")));
        assert!(has_tool_involvement(&req));
    }

    #[test]
    pub fn tool_prompt_uses_compact_schema_summary() {
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
    pub fn terminal_prompt_with_matching_tool_enters_tool_generation() {
        let mut req = tool_request("Run npm test.", None);
        req.tools = Some(vec![server_tool(
            "bash",
            "Run a shell command in the project terminal",
        )]);

        assert!(has_tool_involvement(&req));
    }

    #[test]
    pub fn repaired_tool_arguments_are_valid_json() {
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
    pub fn rejects_tool_calls_not_present_in_selected_tools() {
        let raw = r#"<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>"#;
        let calls = parse_tool_calls_for_tools(raw, &[server_tool("bash", "Run shell commands")]);

        assert!(calls.is_empty());
    }

    #[test]
    pub fn tool_error_category_uses_config_priority() {
        let config = parse_capability_config(
            r#"{
                "aliases": {},
                "tool_taxonomy": {},
                "tool_error_markers": ["error", "timed out"],
                "tool_salient_markers": ["error", "timed out"],
                "tool_error_categories": {
                    "network_error": ["network error", "connection"],
                    "timeout": ["timed out", "timeout"]
                },
                "tool_error_category_priority": ["timeout", "network_error"]
            }"#,
        )
        .expect("valid capability config");

        assert_eq!(
            tool_error_category_with_config("network error: connection timed out", &config),
            Some("timeout".to_string())
        );
    }

    #[test]
    pub fn tool_result_followup_without_tool_intent_does_not_force_tool_generation() {
        let mut req = tool_request("Run cargo test.", None);
        req.tools = Some(vec![server_tool("bash", "Run a shell command")]);
        req.messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: json!("Run cargo test."),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: json!(""),
                tool_call_id: None,
                tool_calls: Some(vec![ToolCallIn {
                    id: "call_bash".to_string(),
                    r#type: "function".to_string(),
                    function: FunctionCallIn {
                        name: "bash".to_string(),
                        arguments: json!({"cmd":"cargo test"}).to_string(),
                    },
                }]),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: json!("error[E0425]: cannot find value `x` in this scope"),
                tool_call_id: Some("call_bash".to_string()),
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: json!("Summarize the failure in one sentence."),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        assert!(!has_tool_involvement(&req));
    }

    #[test]
    pub fn streaming_tool_call_chunks_end_with_tool_calls_finish_reason() {
        let expected_args = json!({"cmd":"cargo test"}).to_string();
        let calls = vec![ToolCallOut {
            id: "call_bash".to_string(),
            r#type: "function".to_string(),
            function: FunctionCallOut {
                name: "bash".to_string(),
                arguments: expected_args.clone(),
            },
        }];

        let chunks = tool_call_stream_chunks(
            "chatcmpl-test".to_string(),
            123,
            Some("selected shell tool".to_string()),
            &calls,
            None,
        );

        assert_eq!(
            chunks[0]["choices"][0]["delta"]["reasoning_content"],
            "selected shell tool"
        );
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "bash"
        );
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            ""
        );

        // Assemble the arguments from all middle chunks
        let mut assembled_args = String::new();
        for chunk in &chunks[2..chunks.len() - 1] {
            if let Some(tool_calls) = chunk["choices"][0]["delta"].get("tool_calls") {
                if let Some(args) = tool_calls[0]["function"].get("arguments") {
                    assembled_args.push_str(args.as_str().unwrap());
                }
            }
        }
        assert_eq!(assembled_args, expected_args);

        assert_eq!(
            chunks[chunks.len() - 1]["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    pub fn test_health_endpoint_response() {
        let resp = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { handle_health().await });
        assert_eq!(resp.0["status"], "healthy");
    }

    #[test]
    pub fn test_usage_details_serialization() {
        let usage = UsageInfo::new(10, 20);
        let val = serde_json::to_value(usage).unwrap();
        assert_eq!(val["prompt_tokens"], 10);
        assert_eq!(val["completion_tokens"], 20);
        assert_eq!(val["total_tokens"], 30);
        assert_eq!(val["prompt_tokens_details"]["cached_tokens"], 0);
        assert_eq!(val["completion_tokens_details"]["reasoning_tokens"], 0);
    }
}
