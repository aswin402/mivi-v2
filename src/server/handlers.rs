use axum::extract::{Json, State};
use axum::http::StatusCode;
use std::sync::Arc;

#[cfg(test)]
use serde_json::json;

use crate::server::helpers::*;
use crate::server::streaming::*;
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

/// Maximum number of texts accepted per embeddings request (bounds CPU/RAM).
pub const MAX_EMBEDDING_INPUTS: usize = 256;

fn invalid_request(message: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "type": "invalid_request_error",
                "message": message,
            }
        })),
    )
}

/// OpenAI-compatible `/v1/embeddings`: pure Rust dense vectors from
/// `semantic_rag::compute_text_embedding` (no model load, no external deps).
pub async fn handle_embeddings(
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (StatusCode, Json<serde_json::Value>)> {
    if let Some(fmt) = req.encoding_format.as_deref() {
        if !fmt.eq_ignore_ascii_case("float") {
            return Err(invalid_request(format!(
                "Unsupported encoding_format '{fmt}': only 'float' is supported."
            )));
        }
    }

    let texts = req.input.into_texts();
    if texts.is_empty() {
        return Err(invalid_request(
            "input must contain at least one text.".to_string(),
        ));
    }
    if texts.len() > MAX_EMBEDDING_INPUTS {
        return Err(invalid_request(format!(
            "input exceeds the maximum of {} texts per request.",
            MAX_EMBEDDING_INPUTS
        )));
    }

    let mut prompt_tokens: u32 = 0;
    let mut data = Vec::with_capacity(texts.len());
    for (index, text) in texts.into_iter().enumerate() {
        prompt_tokens += crate::tokenizer::count_tokens(&text);
        data.push(EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding: crate::semantic_rag::compute_text_embedding(&text),
        });
    }

    Ok(Json(EmbeddingsResponse {
        object: "list".to_string(),
        data,
        model: req.model.unwrap_or_else(|| MODEL_NAME.to_string()),
        usage: EmbeddingsUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
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

#[cfg(test)]
fn embeddings_request(value: serde_json::Value) -> EmbeddingsRequest {
    serde_json::from_value(value).expect("valid embeddings request")
}

#[tokio::test]
async fn embeddings_single_input_returns_unit_vector_and_usage() {
    let req = embeddings_request(json!({
        "model": "mivi",
        "input": "hello world of embeddings"
    }));
    let Json(resp) = handle_embeddings(Json(req)).await.unwrap();

    assert_eq!(resp.object, "list");
    assert_eq!(resp.model, "mivi");
    assert_eq!(resp.data.len(), 1);
    assert_eq!(resp.data[0].object, "embedding");
    assert_eq!(resp.data[0].index, 0);
    assert!(!resp.data[0].embedding.is_empty());
    let norm: f32 = resp.data[0]
        .embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "embedding must be L2-normalized");
    assert!(resp.usage.prompt_tokens > 0);
    assert_eq!(resp.usage.total_tokens, resp.usage.prompt_tokens);
}

#[tokio::test]
async fn embeddings_rejects_empty_batch() {
    let req = embeddings_request(json!({ "input": [] }));
    let (status, body) = handle_embeddings(Json(req)).await.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body.0["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn embeddings_rejects_base64_encoding_format() {
    let req = embeddings_request(json!({
        "input": "text",
        "encoding_format": "base64"
    }));
    let (status, body) = handle_embeddings(Json(req)).await.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.0["error"]["message"]
        .as_str()
        .unwrap()
        .contains("float"));
}
#[tokio::test]
async fn embeddings_batch_indices_are_sequential() {
    let req = embeddings_request(json!({
        "model": "mivi",
        "input": ["first text", "second text", "third text"]
    }));
    let Json(resp) = handle_embeddings(Json(req)).await.unwrap();

    assert_eq!(resp.data.len(), 3);
    for (i, item) in resp.data.iter().enumerate() {
        assert_eq!(item.index, i);
        assert_eq!(item.object, "embedding");
    }
    // Distinct texts must produce distinct vectors.
    assert_ne!(resp.data[0].embedding, resp.data[1].embedding);
}

#[tokio::test]
async fn embeddings_rejects_oversized_batch() {
    let inputs: Vec<String> = (0..=MAX_EMBEDDING_INPUTS)
        .map(|i| format!("text-{i}"))
        .collect();
    let req = embeddings_request(json!({ "input": inputs }));
    let (status, _) = handle_embeddings(Json(req)).await.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
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
