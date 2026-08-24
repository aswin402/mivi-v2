//! Anthropic `/v1/messages` compatibility adapter: translates Anthropic
//! Messages requests onto the chat pipeline and maps the response back.
//!
//! Extracted from `helpers.rs` (server decomposition).

use std::sync::Arc;

use crate::server::chat::*;
use crate::server::helpers::unix_timestamp;
use crate::server::streaming::handle_responses_streaming;
use crate::server::types::*;
use crate::server::usage::*;
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};

pub async fn handle_anthropic_messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<crate::server::types::AnthropicRequest>,
) -> axum::response::Response {
    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "type": "error",
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
    let chat_req = crate::server::types::anthropic_request_to_chat_request(req);
    let include_usage = include_stream_usage(&chat_req);

    if stream {
        return handle_responses_streaming(state, chat_req, now, include_usage, permit)
            .await
            .into_response();
    }

    match complete_chat_non_stream(state, chat_req, now).await {
        Ok(chat) => {
            let response = anthropic_response_from_chat(chat);
            Json(response).into_response()
        }
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": err
                }
            })),
        )
            .into_response(),
    }
}

fn anthropic_response_from_chat(chat: ChatCompletionResponse) -> serde_json::Value {
    let mut content_blocks = Vec::new();
    let mut stop_reason = "end_turn";

    if let Some(choice) = chat.choices.first() {
        if let Some(ref tool_calls) = choice.message.tool_calls {
            if !tool_calls.is_empty() {
                stop_reason = "tool_use";
                for tc in tool_calls {
                    let input_val =
                        serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));
                    content_blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input_val
                    }));
                }
            }
        }

        let text = &choice.message.content;
        if !text.is_empty() {
            content_blocks.insert(
                0,
                serde_json::json!({
                    "type": "text",
                    "text": text
                }),
            );
        }
    }

    let input_tokens = chat
        .usage
        .as_ref()
        .map(|u| u.prompt_tokens)
        .filter(|&t| t > 0)
        .unwrap_or(1);
    let output_tokens = chat
        .usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .filter(|&t| t > 0)
        .unwrap_or(1);

    if content_blocks.is_empty() {
        content_blocks.push(serde_json::json!({
            "type": "text",
            "text": "mivi"
        }));
    }

    serde_json::json!({
        "id": format!("msg_{}", chat.id),
        "type": "message",
        "role": "assistant",
        "model": chat.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}
