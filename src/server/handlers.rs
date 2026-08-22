use axum::extract::{Json, State};
use std::sync::Arc;

#[cfg(test)]
use serde_json::json;

use crate::server::helpers::*;
use crate::server::types::*;

pub async fn handle_root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "online",
        "service": "MIVI-V2 Pure Rust High-Speed AI Engine",
        "version": env!("CARGO_PKG_VERSION"),
        "ram_footprint": "< 12 MB RAM",
        "openai_endpoint": "/v1/chat/completions"
    }))
}

pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mode = crate::runtime::RuntimeConfig::global();
    let worker_status = if mode.uses_worker() {
        match state.brain.text_worker.check_liveness().await {
            Ok(status) => status.to_string(),
            Err(e) => format!("unhealthy: {}", e),
        }
    } else {
        "disabled".to_string()
    };

    let is_healthy = !worker_status.starts_with("unhealthy");

    Json(serde_json::json!({
        "status": if is_healthy { "healthy" } else { "unhealthy" },
        "mode": format!("{:?}", mode.mode).to_ascii_lowercase(),
        "worker_liveness": worker_status,
    }))
}

pub async fn handle_models() -> Json<ModelListResponse> {
    let now = unix_timestamp();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: MODEL_NAME.to_string(),
            object: "model".to_string(),
            created: now,
            owned_by: MODEL_NAME.to_string(),
            context_length: Some(
                crate::runtime::RuntimeConfig::global()
                    .context
                    .max_input_tokens,
            ),
        }],
    })
}

#[test]
pub fn text_stream_chunks_include_usage_only_when_requested() {
    let without_usage = text_stream_chunks(
        "chatcmpl-test".to_string(),
        123,
        None,
        "hello".to_string(),
        None,
    );
    assert!(without_usage
        .iter()
        .all(|chunk| chunk.get("usage").is_none()));

    let with_usage = text_stream_chunks(
        "chatcmpl-test".to_string(),
        123,
        Some("reasoning".to_string()),
        "hello".to_string(),
        Some(UsageInfo::new(5, 2)),
    );
    let usage_chunk = with_usage
        .iter()
        .find(|chunk| chunk.get("usage").is_some())
        .expect("usage chunk expected");

    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 5);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 2);
    assert_eq!(usage_chunk["usage"]["total_tokens"], 7);
}

#[test]
pub fn tool_call_stream_chunks_include_usage_when_requested() {
    let calls = vec![ToolCallOut {
        id: "call_bash".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "bash".to_string(),
            arguments: json!({"cmd":"cargo test"}).to_string(),
        },
    }];

    let chunks = tool_call_stream_chunks(
        "chatcmpl-test".to_string(),
        123,
        None,
        &calls,
        Some(UsageInfo::new(8, 3)),
    );

    assert_eq!(chunks.last().unwrap()["usage"]["total_tokens"], 11);
    assert_eq!(chunks.last().unwrap()["choices"], json!([]));
}

#[tokio::test]
async fn root_reports_cargo_package_version() {
    let Json(value) = handle_root().await;
    assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn models_report_runtime_context_budget_not_a_lie() {
    let Json(resp) = handle_models().await;
    let expected = crate::runtime::RuntimeConfig::global()
        .context
        .max_input_tokens;
    assert_eq!(resp.data[0].context_length, Some(expected));
}
