//! Server startup: router assembly, cache tuner task, warmup, and the
//! axum serve loop.
//!
//! Extracted from `helpers.rs` (server decomposition).

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

use crate::brain::EdgeBrain;
use crate::orchestrator::AgentOrchestrator;
use crate::router::NeedleRouter;
use crate::runtime::RuntimeConfig;
use crate::server::anthropic::handle_anthropic_messages;
use crate::server::chat::handle_chat_completions;
use crate::server::handlers::*;
use crate::server::helpers::handle_responses;
use crate::server::middleware::*;
use crate::server::responses_map::responses_request_to_chat_request;
use crate::server::streaming::*;
use crate::server::tool_generate::sweep_stale_grammar_files;
use crate::server::types::*;
use crate::server::ui;
use crate::server::usage::*;

pub(crate) fn resolve_bind_host() -> String {
    std::env::var("MIVI_HOST")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

pub async fn start_api_server(
    brain: EdgeBrain,
    orchestrator: AgentOrchestrator,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let ultra_low = std::env::var("MIVI_ULTRA_LOW_RAM")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    sweep_stale_grammar_files();

    let max_concurrent = if ultra_low {
        info!("[MIVI-V2] Ultra-low-RAM mode: forcing max concurrent requests to 1");
        1
    } else {
        std::env::var("MIVI_MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
    };
    let state = Arc::new(AppState {
        brain,
        orchestrator,
        router: NeedleRouter::new(),
        semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
        rate_limiter: crate::server::types::RateLimiter::new(),
    });

    // Trace-driven cache tuner (TODO 19.4): every 2 minutes, adapt the
    // SemanticCache capacity from the measured hit/miss window.
    {
        let tuning_cache = state.orchestrator.cache.clone();
        tokio::spawn(async move {
            let mut last = (0u64, 0u64, 0u64);
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(120));
            // The first tick fires immediately; skip it so the first decision
            // is made after a full 120s window of real traffic.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let (hits, misses, evictions, _) = tuning_cache.counters();
                let window = (
                    hits.saturating_sub(last.0),
                    misses.saturating_sub(last.1),
                    evictions.saturating_sub(last.2),
                );
                last = (hits, misses, evictions);
                tuning_cache.adapt_window(window.0, window.1, window.2);
            }
        });
    }

    let api_routes = Router::new()
        .route("/models", get(handle_models))
        .route("/chat/completions", post(handle_chat_completions))
        .route("/responses", post(handle_responses))
        .route("/messages", post(handle_anthropic_messages))
        .route("/embeddings", post(handle_embeddings))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn(timeout_middleware))
        .layer(axum::middleware::from_fn(auth_middleware));

    let app = Router::new()
        .route("/", get(handle_root))
        .route("/ui", get(crate::server::ui::handle_ui))
        .route("/ui/api/stats", get(crate::server::ui::handle_ui_stats))
        .route("/ui/api/traces", get(crate::server::ui::handle_ui_traces))
        .route("/ui/api/heat", get(crate::server::ui::handle_ui_heat))
        .route("/ui/api/rag", post(crate::server::ui::handle_ui_rag))
        .route("/v1/health", get(handle_health))
        .nest("/v1", api_routes)
        .layer(CorsLayer::permissive())
        .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024)) // limit payload to 16MB
        .with_state(state.clone());

    let port = std::env::var("MIVI_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .filter(|_| port == 8000) // only override the built-in default
        .unwrap_or(port);
    let addr = format!("{}:{}", resolve_bind_host(), port);
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        error!(
            "❌ Failed to bind to {}: {}. Is the port already in use?",
            addr, e
        );
        e
    })?;
    info!(
        "🚀 MIVI-V2 High-Speed Server listening on http://{} ...",
        addr
    );

    if !ultra_low {
        // Pre-load model into NativeBrain cache during startup.
        // This spreads the ~400MB model load over boot time instead of
        // spiking RAM on the first user request.
        let warmup_brain = state.brain.clone();
        tokio::spawn(async move {
            info!("[MIVI-V2 Warmup] Pre-loading model into native engine cache...");
            let start = std::time::Instant::now();

            #[cfg(feature = "native")]
            {
                let model_path = warmup_brain.llama_path.clone();
                // Spawn on blocking thread since get_or_load does heavy I/O
                let result = tokio::task::spawn_blocking(move || {
                    match warmup_brain.native.get_or_load(&model_path) {
                        Ok(_loaded) => {
                            info!(
                                "[MIVI-V2 Warmup] Native engine ready in {:.2}s. Model cached.",
                                start.elapsed().as_secs_f32()
                            );
                        }
                        Err(e) => {
                            warn!("[MIVI-V2 Warmup] Failed to pre-load model: {}", e);
                        }
                    }
                })
                .await;
                if let Err(e) = result {
                    warn!("[MIVI-V2 Warmup] Warmup task panicked: {}", e);
                }
            }

            #[cfg(not(feature = "native"))]
            {
                let messages = serde_json::json!([
                    {"role": "user", "content": "warmup"}
                ]);
                let _ = warmup_brain
                    .text_worker
                    .query_chat_full(
                        messages,
                        None,
                        None,
                        None,
                        None,
                        None,
                        Some(1),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await;
                info!(
                    "[MIVI-V2 Warmup] Worker warmup completed in {:.2}s.",
                    start.elapsed().as_secs_f32()
                );
            }
        });
    } else {
        info!("[MIVI-V2 Warmup] Skipping warmup in ultra-low-RAM mode to save memory");
    }

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;
    Ok(())
}
