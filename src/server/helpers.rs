use crate::server::handlers::*;
use axum::{
    extract::{Json, State},
    response::sse::{Event, Sse},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use futures::stream::{Stream, StreamExt};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::Path;

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
#[allow(unused_imports)]
use crate::model_process::spawn_streaming;
use crate::okf_memory::load_memory_dir;
use crate::orchestrator::AgentOrchestrator;
use crate::retrieval::{build_retrieval_pack_with_sources, should_include_workspace_rag};
use crate::router::NeedleRouter;
use crate::runtime::RuntimeConfig;
use crate::tool_filter::{filter_tools, has_tool_intent};
use crate::trace::{preview as trace_preview, trace_event, TraceConfig};

pub use crate::constants::MODEL_NAME;
use crate::constants::{MAX_PROMPT_TOOLS, MIVI_CHAT_SYSTEM_PROMPT};
use crate::server::anthropic::*;
use crate::server::chat::*;
use crate::server::middleware::*;
use crate::server::prompt::*;
use crate::server::responses_map::*;
use crate::server::startup::*;
use crate::server::streaming::*;
use crate::server::tool_generate::*;
use crate::server::tool_parse::*;
use crate::server::tool_select::*;
use crate::server::types::*;
use crate::server::usage::*;

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

pub async fn model_prompt_from_request(
    req: &ChatCompletionRequest,
    latest_user_prompt: &str,
    state: &AppState,
) -> String {
    let config = RuntimeConfig::global();
    let compressed = compress_context(&req.messages, config.context);
    let all_memories =
        tokio::task::spawn_blocking(|| load_memory_dir(Path::new("memory")).unwrap_or_default())
            .await
            .unwrap_or_default();
    let router_class = state.router.classify_intent_nb(latest_user_prompt).0;
    let is_chat = router_class == "CHAT";
    let has_tools = req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);

    // Limit memory count for simple chat to save prompt space and CPU
    let memory_limit = if is_chat && !has_tools { 2 } else { 4 };
    let memories =
        crate::okf_memory::search_memories(&all_memories, latest_user_prompt, memory_limit);

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

pub fn explicitly_mentions_tool_name(user_text: &str, tool_name: &str) -> bool {
    let name = tool_name.trim().to_ascii_lowercase();
    if name.is_empty() {
        return false;
    }

    let is_specific_tool_name = name.contains('_') || name.contains('-') || name.len() >= 8;
    is_specific_tool_name && user_text.contains(&name)
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
