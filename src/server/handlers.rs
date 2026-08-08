use super::*;
use axum::{
    extract::{Json, State},
    response::sse::{Event, Sse},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
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

use crate::server::helpers::*;
use crate::server::types::*;

pub async fn handle_root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "online",
        "service": "MIVI-V2 Pure Rust High-Speed AI Engine",
        "version": "0.0.6",
        "ram_footprint": "< 12 MB RAM",
        "openai_endpoint": "/v1/chat/completions"
    }))
}

pub async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy"
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
