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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::brain::EdgeBrain;
use crate::model_process::spawn_streaming;
use crate::orchestrator::AgentOrchestrator;
use crate::router::NeedleRouter;

/// The single model name exposed to external agents.
/// Internal SML routing is hidden behind this constant.
pub const MODEL_NAME: &str = "mivi";

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
        "version": "0.0.3",
        "ram_footprint": "< 12 MB RAM",
        "openai_endpoint": "/v1/chat/completions"
    }))
}

async fn handle_models() -> Json<ModelListResponse> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![
            ModelObject { id: MODEL_NAME.to_string(), object: "model".to_string(), created: now, owned_by: MODEL_NAME.to_string() },
        ],
    })
}

// ──────────────────────────────────────────────
// Prompt building
// ──────────────────────────────────────────────

fn build_chat_prompt(req: &ChatCompletionRequest) -> String {
    let mut prompt = String::new();

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
    } else if req.tools.as_ref().map_or(true, |t| t.is_empty()) {
        prompt.push_str("<|im_start|>system\nYou are a helpful, concise AI assistant.<|im_end|>\n");
    }

    // Conversation turns.
    let has_tools = req.tools.as_ref().map_or(false, |t| !t.is_empty());
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
                        prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", block.trim()));
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
        let func_block = build_function_list_block(req);
        if let Some(idx) = last_user_idx {
            prompt.insert_str(idx, &func_block);
        } else {
            prompt.push_str(&func_block);
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn build_function_list_block(req: &ChatCompletionRequest) -> String {
    let tools = match req.tools {
        Some(ref t) if !t.is_empty() => t,
        _ => return String::new(),
    };

    let mut block = String::new();

    let first_tool = &tools[0].function;
    let ex_args = first_tool.parameters
        .as_ref()
        .and_then(|p| p.get("properties"))
        .and_then(|props| props.as_object())
        .and_then(|obj| obj.keys().next())
        .map(|k| format!("\"{}\": \"...\"", k))
        .unwrap_or_else(|| "\"key\": \"value\"".to_string());

    block.push_str(&format!(
        "\nAvailable functions:\n\
         - {name}: {desc}\n\n\
         When appropriate, respond with ONLY:\n\
         {{\"name\": \"{name}\", \"arguments\": {{{ex_args}}}}}\n\n",
        name = first_tool.name,
        desc = first_tool.description.as_deref().unwrap_or(""),
    ));

    for tool in tools {
        let f = &tool.function;
        block.push_str(&format!("Function: {} - {}\n", f.name, f.description.as_deref().unwrap_or("")));
        if let Some(ref params) = f.parameters {
            block.push_str(&format!("Params: {}\n", params));
        }
    }

    block.push_str("\nRespond with JSON or text:\n");
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

/// Extract user prompt text + optional image path for legacy code paths.
fn extract_content(req: &ChatCompletionRequest) -> (String, Option<String>) {
    let mut parts: Vec<String> = Vec::new();
    let mut image_path: Option<String> = None;
    for msg in &req.messages {
        if msg.role == "user" {
            if let Some(text) = msg.content.as_str() {
                let stripped = strip_available_skills(text);
                if !stripped.is_empty() {
                    parts.push(stripped);
                }
            } else if let Some(arr) = msg.content.as_array() {
                let mut last_text = String::new();
                for item in arr {
                    if let Some(t) = item.get("type").and_then(|v| v.as_str()) {
                        if t == "text" {
                            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                                let stripped = strip_available_skills(text);
                                if !stripped.is_empty() {
                                    last_text = stripped;
                                }
                            }
                        } else if t == "image_url" {
                            if let Some(url) = item.get("image_url").and_then(|v| v.get("url")).and_then(|v| v.as_str()) {
                                image_path = Some(url.to_string());
                            }
                        }
                    }
                }
                if !last_text.is_empty() {
                    parts.push(last_text);
                }
            }
        } else if msg.role == "system" {
            if let Some(text) = msg.content.as_str() {
                if text.len() > 200 {
                    parts.push(format!("[context: {}]", &text[text.len()-200..]));
                }
            }
        }
    }
    let user_prompt = if parts.is_empty() {
        String::new()
    } else {
        parts.join("\n")
    };
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
    // Try to parse as {"name": "...", "arguments": {...}} or {"function": "...", "arguments": {...}}
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = val.as_object()?;

    // Accept either "name" or "function" key.
    let name = obj
        .get("name")
        .or_else(|| obj.get("function"))
        .and_then(|v| v.as_str())?;

    let arguments = match obj.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(v @ serde_json::Value::Object(_)) => v.to_string(),
        Some(v) => v.to_string(),
        None => "{}".to_string(),
    };

    Some(ToolCallOut {
        id: format!("call_{}", name),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: name.to_string(),
            arguments,
        },
    })
}

/// Check if the request involves tools (either providing tool definitions,
/// or continuing after a tool call response).
fn has_tool_involvement(req: &ChatCompletionRequest) -> bool {
    // tool_choice: "none" explicitly disables tool calling.
    if let Some(ref choice) = req.tool_choice {
        if let serde_json::Value::String(s) = choice {
            if s == "none" {
                return false;
            }
        }
    }
    // If the client provided tool definitions.
    if let Some(ref tools) = req.tools {
        if !tools.is_empty() {
            return true;
        }
    }
    // If there are tool role messages (tool results to process).
    if req.messages.iter().any(|m| m.role == "tool") {
        return true;
    }
    // If any assistant message has tool_calls (from a previous round).
    if req.messages.iter().any(|m| m.role == "assistant" && m.tool_calls.is_some()) {
        return true;
    }
    false
}

// ──────────────────────────────────────────────
// Backend model calls
// ──────────────────────────────────────────────

/// One-shot reasoner call (spawns llama-cli per request).
fn reasoner_chat(brain: &EdgeBrain, user_prompt: &str) -> (String, String) {
    let res = brain.query_reasoner(user_prompt, "You are a helpful assistant.")
        .unwrap_or_else(|e| format!("Error: {}", e));
    (res, MODEL_NAME.to_string())
}

/// One-shot coder call (spawns llama-cli per request).
fn code_chat(brain: &EdgeBrain, user_prompt: &str) -> (String, String) {
    let res = brain.query_coder(user_prompt, "You are a coding expert.")
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

/// Generate tool calls: run the model with tool-aware prompt, parse tool calls.
fn generate_tool_calls(brain: &EdgeBrain, req: &ChatCompletionRequest) -> (Vec<ToolCallOut>, String) {
    let prompt = build_chat_prompt(req);
    println!("[MIVI-V2 ToolGen] Prompt length: {} chars", prompt.len());

    let raw = model_chat(brain, &prompt);
    let debug_preview: String = raw.chars().take(200).collect();
    println!("[MIVI-V2 ToolGen] Raw model output (truncated): {:?}", debug_preview);

    // Parse tool calls from output.
    let calls = parse_tool_calls(&raw);

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
    eprintln!(">>> MIVI REQUEST: model={:?} stream={:?} msgs={} tools={}",
        req.model, req.stream, req.messages.len(),
        req.tools.as_ref().map(|t| t.len()).unwrap_or(0));

    for (i, msg) in req.messages.iter().enumerate() {
        let preview = match &msg.content {
            serde_json::Value::String(s) => {
                let chars: String = s.chars().take(120).collect();
                format!("str(len={}) {:?}...", s.len(), chars)
            },
            other => format!("{:?}", other),
        };
        let tc = if msg.tool_calls.is_some() { " [has tool_calls]" } else { "" };
        let ti = if msg.tool_call_id.is_some() { " [has tool_call_id]" } else { "" };
        eprintln!("  msg[{}]: role={:?} content={}{}{}", i, msg.role, preview, tc, ti);
    }

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
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

    // Streaming path.
    if req.stream.unwrap_or(false) {
        return handle_streaming(state, user_prompt, target_model, now).await.into_response();
    }

    // Non-streaming path.
    let intent = state.router.classify_intent(&user_prompt);
    println!("[MIVI-V2 Server] Intent: {} | Model: '{}' | Prompt: '{}'", intent, target_model, user_prompt);

    let (response_text, chosen_model) = if image_path.is_some() {
        let path = image_path.unwrap_or_default();
        match state.brain.query_vision(&path, &user_prompt) {
            Ok(res) => (res, MODEL_NAME.to_string()),
            Err(err) => (format!("Vision error: {}", err), MODEL_NAME.to_string()),
        }
    } else {
        match target_model.to_lowercase().as_str() {
            "coder" => code_chat(&state.brain, &user_prompt),
            "reasoner" => reasoner_chat(&state.brain, &user_prompt),
            _ if intent == "CHAT" => reasoner_chat(&state.brain, &user_prompt),
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
    let done_marker = futures::stream::once(async {
        Ok::<_, Infallible>(Event::default().data("[DONE]"))
    });

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
    println!("🚀 MIVI-V2 High-Speed Server listening on http://{} ...", addr);
    axum::serve(listener, app).await.unwrap();
}
