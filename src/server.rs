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

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
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

#[derive(Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
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

/// Build a chat-formatted prompt (Llama-3 Instruct template).
fn format_prompt(prompt: &str, system: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system, prompt
    )
}

/// Helper: extract user text + optional image path from request messages.
fn extract_content(req: &ChatCompletionRequest) -> (String, Option<String>) {
    let mut user_prompt = String::new();
    let mut image_path: Option<String> = None;
    for msg in &req.messages {
        if msg.role == "user" {
            if let Some(text) = msg.content.as_str() {
                user_prompt = text.to_string();
            } else if let Some(arr) = msg.content.as_array() {
                let mut text_parts = Vec::new();
                for item in arr {
                    if let Some(t) = item.get("type").and_then(|v| v.as_str()) {
                        if t == "text" {
                            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(text.to_string());
                            }
                        } else if t == "image_url" {
                            if let Some(url) = item.get("image_url").and_then(|v| v.get("url")).and_then(|v| v.as_str()) {
                                image_path = Some(url.to_string());
                            }
                        }
                    }
                }
                user_prompt = text_parts.join(" ");
            }
        }
    }
    (user_prompt, image_path)
}

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

async fn handle_chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let (user_prompt, image_path) = extract_content(&req);
    let target_model = req.model.unwrap_or_else(|| MODEL_NAME.to_string());
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    // Streaming path — real-time tokens via SSE.
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
            },
            finish_reason: "stop".to_string(),
        }],
    })
    .into_response()
}

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
    let formatted = format_prompt(&user_prompt, "You are a helpful assistant.");

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
