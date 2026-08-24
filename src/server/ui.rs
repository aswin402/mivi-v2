//! Phase 15: built-in web dashboard served at `/ui` (zero-dependency,
//! `include_str!`-embedded single-page app).
//!
//! The page talks to the public `/v1/chat/completions` endpoint directly for
//! streaming chat; this module only serves the HTML plus two tiny read-only
//! JSON helpers (`/ui/api/stats`, `/ui/api/rag`) that have no OpenAI
//! equivalent.

use crate::server::types::*;
use axum::extract::{Json, State};
use axum::response::Html;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;

/// Embedded single-page dashboard (self-contained HTML/CSS/JS).
pub const UI_INDEX_HTML: &str = include_str!("../../assets/ui/index.html");
static BOOT: std::sync::LazyLock<std::time::Instant> =
    std::sync::LazyLock::new(std::time::Instant::now);

/// Resident set size of this process in MB (Linux `/proc/self/statm`).
pub fn resident_rss_mb() -> u64 {
    const PAGE_SIZE: u64 = 4096;
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|content| {
            content
                .split_whitespace()
                .nth(1)?
                .parse::<u64>()
                .ok()
                .map(|pages| pages * PAGE_SIZE / (1024 * 1024))
        })
        .unwrap_or(0)
}

#[derive(Serialize)]
pub struct UiStats {
    pub version: &'static str,
    pub mode: String,
    pub context_tokens: usize,
    pub rss_mb: u64,
    pub ram_target_mb: usize,
    pub uptime_s: u64,
}

pub async fn handle_ui() -> Html<&'static str> {
    Html(UI_INDEX_HTML)
}

pub async fn handle_ui_stats() -> Json<UiStats> {
    let config = crate::runtime::RuntimeConfig::global();
    Json(UiStats {
        version: env!("CARGO_PKG_VERSION"),
        mode: format!("{:?}", config.mode).to_ascii_lowercase(),
        context_tokens: config.context.max_input_tokens,
        rss_mb: resident_rss_mb(),
        ram_target_mb: config.ram_target_mb,
        uptime_s: BOOT.elapsed().as_secs(),
    })
}

#[derive(Deserialize)]
pub struct UiRagQuery {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
}

pub async fn handle_ui_rag(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UiRagQuery>,
) -> Json<serde_json::Value> {
    if req.query.trim().is_empty() {
        return Json(serde_json::json!({"results": []}));
    }
    let top_k = req.top_k.unwrap_or(8).clamp(1, 32);
    let hits = state.orchestrator.rag.search(&req.query, top_k).await;
    let results: Vec<serde_json::Value> = hits
        .into_iter()
        .map(|(chunk, score)| {
            serde_json::json!({
                "file_path": chunk.file_path,
                "line_start": chunk.line_start,
                "text": chunk.text,
                "score": score,
            })
        })
        .collect();
    Json(serde_json::json!({ "results": results }))
}

#[derive(Deserialize)]
pub struct UiTraceQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Tail of `logs/mivi-trace.jsonl` (requires `MIVI_TRACE=1`). Reads at most
/// the last 512 KB so a huge trace file cannot balloon the response.
pub async fn handle_ui_traces(
    axum::extract::Query(q): axum::extract::Query<UiTraceQuery>,
) -> Json<serde_json::Value> {
    let limit = q.limit.unwrap_or(30).clamp(1, 200);
    let path = crate::trace::TraceConfig::from_env().path;
    let empty = serde_json::json!({ "enabled": false, "events": [] });
    let Ok(content) = std::fs::read(&path) else {
        return Json(empty);
    };
    // Only parse the tail; JSONL rows are self-delimiting by newline.
    let tail = if content.len() > 512 * 1024 {
        let slice = &content[content.len() - 512 * 1024..];
        // Drop the (likely partial) first line.
        match slice.iter().position(|b| *b == b'\n') {
            Some(pos) => &slice[pos + 1..],
            None => slice,
        }
    } else {
        &content[..]
    };
    let events: Vec<serde_json::Value> = String::from_utf8_lossy(tail)
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(limit)
        .collect();
    Json(serde_json::json!({
        "enabled": true,
        "events": events,
    }))
}

/// Hot-file usage from adaptive RAG workspace learning (`.mivi_rag_usage`),
/// sorted hottest first — the data behind the dashboard heat view.
pub async fn handle_ui_heat() -> Json<serde_json::Value> {
    let mut entries: Vec<(String, u64)> = std::fs::read_to_string(".mivi_rag_usage")
        .map(|content| {
            serde_json::from_str::<std::collections::HashMap<String, u64>>(&content)
                .map(|map| map.into_iter().collect::<Vec<_>>())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    let results: Vec<serde_json::Value> = entries
        .into_iter()
        .take(20)
        .map(|(path, count)| serde_json::json!({ "path": path, "count": count }))
        .collect();
    Json(serde_json::json!({ "results": results }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_page_contains_dashboard_marker() {
        assert!(UI_INDEX_HTML.contains("MIVI"));
        assert!(UI_INDEX_HTML.contains("/v1/chat/completions"));
        assert!(UI_INDEX_HTML.contains("/ui/api/stats"));
    }

    #[tokio::test]
    async fn traces_endpoint_degrades_to_disabled_when_file_missing() {
        // Default path may or may not exist; the handler must never error.
        let Json(value) =
            handle_ui_traces(axum::extract::Query(UiTraceQuery { limit: Some(5) })).await;
        assert!(value["events"].is_array());
    }

    #[tokio::test]
    async fn heat_endpoint_returns_results_array_without_panicking() {
        let Json(value) = handle_ui_heat().await;
        assert!(value["results"].is_array());
        for entry in value["results"].as_array().unwrap() {
            assert!(entry["count"].as_u64().unwrap_or(0) > 0);
            assert!(!entry["path"].as_str().unwrap_or_default().is_empty());
        }
    }

    #[test]
    fn rss_reader_returns_zero_or_positive_without_panicking() {
        // On Linux this reads /proc; elsewhere it must degrade to 0.
        assert!(resident_rss_mb() < u64::MAX / 2);
    }
}
