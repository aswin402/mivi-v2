//! HTTP middleware: client identification, rate limiting, request
//! timeout, and bearer-token auth.
//!
//! Extracted from `helpers.rs` (server decomposition).

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tracing::debug;

use crate::server::types::*;

fn get_client_identifier(
    req: &axum::http::Request<axum::body::Body>,
    peer: Option<std::net::SocketAddr>,
) -> String {
    // Proxy headers are trusted ONLY when the operator opts in; otherwise any
    // client could rotate X-Forwarded-For to dodge the limiter.
    let trust_proxy = std::env::var("MIVI_TRUST_PROXY_HEADERS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if trust_proxy {
        if let Some(forwarded) = req.headers().get("x-forwarded-for") {
            if let Ok(s) = forwarded.to_str() {
                if let Some(first_ip) = s.split(',').next() {
                    let ip = first_ip.trim();
                    if !ip.is_empty() {
                        return ip.to_string();
                    }
                }
            }
        }
        if let Some(real_ip) = req.headers().get("x-real-ip") {
            if let Ok(s) = real_ip.to_str() {
                let ip = s.trim();
                if !ip.is_empty() {
                    return ip.to_string();
                }
            }
        }
    }

    if let Some(addr) = peer {
        return addr.ip().to_string();
    }

    "generic_client".to_string()
}

pub(crate) async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    let client_id = get_client_identifier(&req, Some(peer));
    if let Err(msg) = state.rate_limiter.check_rate_limit(client_id) {
        let error_json = serde_json::json!({
            "error": {
                "type": "rate_limit_error",
                "message": msg
            }
        });
        let mut res = axum::response::Json(error_json).into_response();
        *res.status_mut() = axum::http::StatusCode::TOO_MANY_REQUESTS;
        return Ok(res);
    }
    Ok(next.run(req).await)
}

pub(crate) fn request_timeout_secs() -> u64 {
    std::env::var("MIVI_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(300)
}

pub(crate) async fn timeout_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    let duration = std::time::Duration::from_secs(request_timeout_secs());
    match tokio::time::timeout(duration, next.run(req)).await {
        Ok(res) => Ok(res),
        Err(_) => {
            let error_json = serde_json::json!({
                "error": {
                    "type": "timeout_error",
                    "message": format!("Request timed out after {} seconds.", request_timeout_secs())
                }
            });
            let mut res = axum::response::Json(error_json).into_response();
            *res.status_mut() = axum::http::StatusCode::REQUEST_TIMEOUT;
            Ok(res)
        }
    }
}

/// Length-checked XOR-fold comparison. Not literally branch-free, but removes
/// the early-exit-on-first-mismatch byte leak of `==`.
pub(crate) fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

pub(crate) async fn auth_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    if let Ok(expected_key) = std::env::var("MIVI_API_KEY") {
        if !expected_key.is_empty() {
            if let Some(auth_header) = req.headers().get("authorization") {
                if let Ok(auth_str) = auth_header.to_str() {
                    if auth_str.starts_with("Bearer ") {
                        let token = &auth_str["Bearer ".len()..];
                        if constant_time_eq(token, &expected_key) {
                            return Ok(next.run(req).await);
                        }
                    }
                }
            }
            let error_json = serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid API key or Authorization header missing"
                }
            });
            let mut res = axum::response::Json(error_json).into_response();
            *res.status_mut() = axum::http::StatusCode::UNAUTHORIZED;
            return Ok(res);
        }
    }
    Ok(next.run(req).await)
}
