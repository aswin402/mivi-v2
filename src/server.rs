use axum::{
    extract::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower_http::cors::CorsLayer;

use crate::brain::EdgeBrain;
use crate::orchestrator::AgentOrchestrator;
use crate::router::NeedleRouter;

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
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
            ModelObject { id: "ai-brain".to_string(), object: "model".to_string(), created: now, owned_by: "mivi-v2".to_string() },
            ModelObject { id: "mivi-v2".to_string(), object: "model".to_string(), created: now, owned_by: "mivi-v2".to_string() },
            ModelObject { id: "qwen-2.5-0.5b".to_string(), object: "model".to_string(), created: now, owned_by: "mivi-v2".to_string() },
            ModelObject { id: "llama-3.2-1b".to_string(), object: "model".to_string(), created: now, owned_by: "mivi-v2".to_string() },
            ModelObject { id: "minicpm-v-4.6".to_string(), object: "model".to_string(), created: now, owned_by: "mivi-v2".to_string() },
        ],
    })
}

async fn handle_chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
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

    let target_model = req.model.unwrap_or_else(|| "ai-brain".to_string());
    let intent = state.router.classify_intent(&user_prompt);
    println!("[MIVI-V2 Server] Intent: {} | Model: '{}' | Prompt: '{}'", intent, target_model, user_prompt);

    let (response_text, chosen_model) = if target_model.eq_ignore_ascii_case("minicpm-v-4.6")
        || target_model.contains("vision")
        || image_path.is_some()
    {
        let path = image_path.unwrap_or_default();
        match state.brain.query_vision(&path, &user_prompt) {
            Ok(res) => (res, "minicpm-v-4.6".to_string()),
            Err(err) => (format!("Vision error: {}", err), "minicpm-v-4.6".to_string()),
        }
    } else {
        match target_model.to_lowercase().as_str() {
            "qwen-2.5-0.5b" | "coder" => (
                state.brain.query_coder(&user_prompt, "You are a coding expert.").unwrap_or_default(),
                "qwen-2.5-0.5b".to_string(),
            ),
            "llama-3.2-1b" | "reasoner" => (
                state.brain.query_reasoner(&user_prompt, "You are a helpful assistant.").unwrap_or_default(),
                "llama-3.2-1b".to_string(),
            ),
            _ => {
                let (_, res) = state.orchestrator.execute_plan(&user_prompt).await;
                (res, "mivi-v2".to_string())
            }
        }
    };

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

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
}

pub async fn start_api_server(state: Arc<AppState>, port: u16) {
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
