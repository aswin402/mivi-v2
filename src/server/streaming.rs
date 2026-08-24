//! SSE streaming for chat and responses endpoints.
//!
//! Extracted from `helpers.rs` (server decomposition).

use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use futures::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

use crate::constants::{MIVI_CHAT_SYSTEM_PROMPT, MODEL_NAME};
#[allow(unused_imports)]
use crate::model_process::spawn_streaming;
use crate::runtime::RuntimeConfig;
use crate::server::chat::*;
use crate::server::helpers::{model_prompt_from_request, unix_timestamp};
use crate::server::prompt::*;
use crate::server::tool_generate::*;
use crate::server::tool_parse::*;
use crate::server::types::*;
use crate::server::usage::*;
use crate::trace::trace_event;

pub async fn handle_responses_streaming(
    state: Arc<AppState>,
    chat_req: ChatCompletionRequest,
    now: u64,
    include_usage: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    let response_id = format!("resp-v2-{}", now);
    let prompt_tokens = token_counter().count_tokens(
        &chat_req
            .messages
            .last()
            .map(|m| m.content.as_str().unwrap_or("").to_string())
            .unwrap_or_default(),
    );

    let user_prompt = latest_user_prompt_text(&chat_req);

    let (intent, _confidence) = state
        .router
        .classify_intent(&state.brain, &user_prompt)
        .await;

    // Lightweight chat path: skip heavy retrieval for CHAT prompts
    let model_user_prompt = if intent == "CHAT" {
        build_chat_prompt(&chat_req)
    } else {
        model_prompt_from_request(&chat_req, &user_prompt, &state).await
    };

    let brain = state.brain.clone();
    let system_prompt = wrap_agent_prompt(MIVI_CHAT_SYSTEM_PROMPT, "");
    let t = active_chat_template();
    let formatted = if crate::brain::is_prompt_preformatted(&model_user_prompt) {
        model_user_prompt.clone()
    } else {
        format!(
            "{}{}{}{}{}{}{}",
            t.system_prefix,
            system_prompt,
            t.system_suffix,
            t.user_prefix,
            model_user_prompt,
            t.user_suffix,
            t.assistant_start
        )
    };

    let cli_path = brain.llama_cli.to_str().unwrap_or("llama-cli").to_string();
    let model_path = brain.llama_path.to_str().unwrap_or("").to_string();
    let runtime_config = RuntimeConfig::global();
    let streaming_context = std::env::var("MIVI_REASONER_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens >= 1024)
        .unwrap_or(runtime_config.context.max_input_tokens)
        .to_string();

    let temp_str = chat_req.temperature.unwrap_or(0.2).to_string();
    let top_p = chat_req.top_p;
    let stop = chat_req.stop.clone();
    let seed = chat_req.seed;
    let json_schema = extract_json_schema(&chat_req);

    // MULTI_STEP now shares the 256-token cap: 512 tokens on a sub-1B model
    // mostly produces hallucinated padding while burning CPU.
    let intent_max_tokens = if intent == "CHAT" {
        128
    } else if intent == "CODE" {
        512
    } else {
        256
    };
    let resolved_max_tokens = Some(chat_req.max_tokens.unwrap_or(intent_max_tokens));

    let grammar_path = get_grammar_path(&chat_req);

    tokio::spawn(async move {
        let run_native = if cfg!(feature = "native") {
            let runtime_config = RuntimeConfig::global();
            runtime_config.mode != crate::runtime::RuntimeMode::Spawn
        } else {
            false
        };

        if run_native {
            #[cfg(feature = "native")]
            {
                match brain.native.query_stream(
                    std::path::Path::new(&model_path),
                    &model_user_prompt,
                    &system_prompt,
                    &temp_str,
                    resolved_max_tokens.unwrap_or(512) as usize,
                    grammar_path.clone(),
                ) {
                    Ok(mut native_rx) => {
                        while let Some(token) = native_rx.recv().await {
                            if tx.send(token).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        error!("[NativeBrain] Native stream error: {}", err);
                    }
                }
            }
        } else {
            let grammar_path = get_grammar_path(&chat_req);
            let mut spawn_rx = spawn_streaming(
                &cli_path,
                &model_path,
                &formatted,
                if brain.ultra_low_ram { "0" } else { "999" },
                &streaming_context,
                &temp_str,
                top_p,
                resolved_max_tokens,
                stop,
                seed,
                json_schema,
                grammar_path,
            );
            while let Some(token) = spawn_rx.recv().await {
                if tx.send(token).await.is_err() {
                    break;
                }
            }
        }
    });

    let completion_tokens = Arc::new(AtomicU32::new(0));
    let completion_tokens_for_stream = completion_tokens.clone();
    let response_id_for_created = response_id.clone();
    let response_id_for_completed = response_id.clone();

    let initial_events = vec![
        serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id_for_created,
                "object": "response",
                "created_at": now,
                "model": MODEL_NAME,
                "status": "in_progress"
            }
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": format!("out-{}", now),
                "type": "message",
                "status": "in_progress",
                "role": "assistant"
            }
        }),
        serde_json::json!({
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "text",
                "text": ""
            }
        }),
    ];

    let initial_stream = futures::stream::iter(
        initial_events
            .into_iter()
            .map(|val| Ok::<_, Infallible>(Event::default().data(val.to_string()))),
    );

    let token_stream = ReceiverStream::new(rx).map(move |token| {
        completion_tokens_for_stream
            .fetch_add(token_counter().count_tokens(&token), Ordering::Relaxed);
        let chunk = serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": token
        });
        Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
    });

    let final_stream = futures::stream::unfold(0, move |state| {
        let completion_tokens = completion_tokens.clone();
        let response_id = response_id_for_completed.clone();
        async move {
            match state {
                0 => {
                    let chunk = serde_json::json!({
                        "type": "response.content_part.done",
                        "output_index": 0,
                        "content_index": 0
                    });
                    Some((
                        Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                        1,
                    ))
                }
                1 => {
                    let chunk = serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0
                    });
                    Some((
                        Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                        2,
                    ))
                }
                2 => {
                    let mut completed_response = serde_json::json!({
                        "id": response_id,
                        "object": "response",
                        "created_at": now,
                        "model": MODEL_NAME,
                        "status": "completed",
                        "output": [{
                            "id": format!("out-{}", now),
                            "type": "message",
                            "status": "completed",
                            "role": "assistant"
                        }]
                    });
                    if include_usage {
                        let usage = UsageInfo::new(
                            prompt_tokens as u32,
                            completion_tokens.load(Ordering::Relaxed),
                        );
                        completed_response["usage"] = usage_value(usage);
                    }
                    let chunk = serde_json::json!({
                        "type": "response.completed",
                        "response": completed_response
                    });
                    Some((
                        Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                        3,
                    ))
                }
                _ => None,
            }
        }
    });

    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let stream = initial_stream
        .chain(token_stream)
        .chain(final_stream)
        .chain(done_marker)
        .map(move |item| {
            let _keep = &permit;
            item
        });

    Sse::new(stream)
}

pub(crate) fn base_stream_chunk(
    id: &str,
    created: u64,
    delta: serde_json::Value,
    finish_reason: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

pub fn tool_call_stream_chunks(
    id: String,
    created: u64,
    reasoning_content: Option<String>,
    tool_calls: &[ToolCallOut],
    usage: Option<UsageInfo>,
) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();

    // Preamble chunk for OpenAI compatibility
    chunks.push(base_stream_chunk(
        &id,
        created,
        serde_json::json!({ "role": "assistant", "content": "" }),
        None,
    ));

    if let Some(reasoning) = reasoning_content {
        chunks.push(base_stream_chunk(
            &id,
            created,
            serde_json::json!({ "reasoning_content": reasoning }),
            None,
        ));
    }

    if !tool_calls.is_empty() {
        // Emit the metadata chunk with empty arguments delta
        let initial_calls: Vec<serde_json::Value> = tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                serde_json::json!({
                    "index": index,
                    "id": call.id,
                    "type": call.r#type,
                    "function": {
                        "name": call.function.name,
                        "arguments": ""
                    }
                })
            })
            .collect();
        chunks.push(base_stream_chunk(
            &id,
            created,
            serde_json::json!({ "tool_calls": initial_calls }),
            None,
        ));

        // Emit the arguments chunks in fragments
        let mut max_len = 0;
        for call in tool_calls {
            max_len = max_len.max(call.function.arguments.len());
        }

        let chunk_size = 12;
        let mut offset = 0;
        while offset < max_len {
            let mut arg_deltas = Vec::new();
            for (index, call) in tool_calls.iter().enumerate() {
                let args = &call.function.arguments;
                if offset < args.len() {
                    let end = (offset + chunk_size).min(args.len());
                    let delta = &args[offset..end];
                    arg_deltas.push(serde_json::json!({
                        "index": index,
                        "function": {
                            "arguments": delta
                        }
                    }));
                }
            }
            if !arg_deltas.is_empty() {
                chunks.push(base_stream_chunk(
                    &id,
                    created,
                    serde_json::json!({ "tool_calls": arg_deltas }),
                    None,
                ));
            }
            offset += chunk_size;
        }
    }

    chunks.push(base_stream_chunk(
        &id,
        created,
        serde_json::json!({}),
        Some("tool_calls"),
    ));

    if let Some(usage) = usage {
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [],
            "usage": usage_value(usage)
        }));
    }

    chunks
}

pub fn stream_tool_calls_response(
    tool_calls: Vec<ToolCallOut>,
    created: u64,
    reasoning_content: Option<String>,
    usage: Option<UsageInfo>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-v2-{}", created);
    let chunks = tool_call_stream_chunks(id, created, reasoning_content, &tool_calls, usage);
    let chunk_stream = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk.to_string()))),
    );
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    let mapped_stream = chunk_stream.chain(done_marker).map(move |item| {
        let _keep = &permit;
        item
    });
    Sse::new(mapped_stream)
}

pub fn text_stream_chunks(
    id: String,
    created: u64,
    reasoning_content: Option<String>,
    content: String,
    usage: Option<UsageInfo>,
) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();

    // Preamble chunk for OpenAI compatibility
    chunks.push(base_stream_chunk(
        &id,
        created,
        serde_json::json!({ "role": "assistant", "content": "" }),
        None,
    ));

    if let Some(reasoning) = reasoning_content {
        chunks.push(base_stream_chunk(
            &id,
            created,
            serde_json::json!({ "reasoning_content": reasoning }),
            None,
        ));
    }

    chunks.push(base_stream_chunk(
        &id,
        created,
        serde_json::json!({ "content": content }),
        None,
    ));

    chunks.push(base_stream_chunk(
        &id,
        created,
        serde_json::json!({}),
        Some("stop"),
    ));

    if let Some(usage) = usage {
        chunks.push(serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [],
            "usage": usage_value(usage)
        }));
    }

    chunks
}

pub fn stream_text_response(
    content: String,
    created: u64,
    reasoning_content: Option<String>,
    usage: Option<UsageInfo>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let id = format!("chatcmpl-v2-{}", created);
    let chunks = text_stream_chunks(id, created, reasoning_content, content, usage);
    let chunk_stream = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk.to_string()))),
    );
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    let mapped_stream = chunk_stream.chain(done_marker).map(move |item| {
        let _keep = &permit;
        item
    });
    Sse::new(mapped_stream)
}

pub(crate) fn worker_stream_content_delta(value: &serde_json::Value) -> Option<&str> {
    value
        .get("content")
        .and_then(|content| content.as_str())
        .or_else(|| {
            value
                .get("choices")
                .and_then(|choices| choices.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(|content| content.as_str())
        })
        .or_else(|| {
            value
                .get("choices")
                .and_then(|choices| choices.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(|text| text.as_str())
        })
}

// ──────────────────────────────────────────────
// SSE streaming handler
// ──────────────────────────────────────────────

/// SSE streaming handler — spawns llama-cli per request, sends tokens as
/// they arrive from stdout, then emits a final `finish_reason: stop` chunk
/// and a `[DONE]` sentinel per the OpenAI streaming spec.
#[allow(unused_variables, unused_assignments)]
pub async fn handle_streaming(
    state: Arc<AppState>,
    user_prompt: String,
    req: &ChatCompletionRequest,
    created: u64,
    include_usage: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
    max_tokens: Option<u32>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(32);
    let id = format!("chatcmpl-v2-{}", created);
    let completion_tokens = Arc::new(AtomicU32::new(0));

    // Clone what we need for the background task.
    let brain = state.brain.clone();
    let system_prompt = wrap_agent_prompt(MIVI_CHAT_SYSTEM_PROMPT, "");
    let t = active_chat_template();
    let formatted = if crate::brain::is_prompt_preformatted(&user_prompt) {
        user_prompt.clone()
    } else {
        format!(
            "{}{}{}{}{}{}{}",
            t.system_prefix,
            system_prompt,
            t.system_suffix,
            t.user_prefix,
            user_prompt,
            t.user_suffix,
            t.assistant_start
        )
    };

    let cli_path = brain.llama_cli.to_str().unwrap_or("llama-cli").to_string();
    let model_path = brain.llama_path.to_str().unwrap_or("").to_string();
    let runtime_config = RuntimeConfig::global();
    let streaming_context = std::env::var("MIVI_REASONER_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens >= 1024)
        .unwrap_or(runtime_config.context.max_input_tokens)
        .to_string();

    let temp_str = req.temperature.unwrap_or(0.2).to_string();
    let top_p = req.top_p;
    let stop = req.stop.clone();
    let seed = req.seed;
    let json_schema = extract_json_schema(req);

    let uses_worker = runtime_config.uses_worker();
    let text_worker = brain.text_worker.clone();
    let req_temp = req.temperature;
    let req_top_p = req.top_p;
    let req_max_tokens = max_tokens;
    let req_stop = req.stop.clone();
    let req_seed = req.seed;
    let req_fp = req.frequency_penalty;
    let req_pp = req.presence_penalty;
    let req_json_schema = json_schema.clone();

    let grammar_path = get_grammar_path(&req);
    let grammar_content = grammar_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok());

    let fallback_user_prompt = user_prompt.clone();
    let grammar_content_for_worker = grammar_content.clone();
    let grammar_path_for_spawn = grammar_path.clone();
    // Deterministic arithmetic fast path (see complete_chat_non_stream):
    // pure math prompts stream the exact answer without a model call.
    let math_answer = if req
        .messages
        .last()
        .map_or(false, |m| m.role == "user" && m.content.is_string())
    {
        crate::math_eval::try_answer(&latest_user_prompt_text(&req))
    } else {
        None
    };
    tokio::spawn(async move {
        if let Some(answer) = math_answer {
            let _ = tx.send(answer).await;
            return;
        }
        let mut emitted = false;
        if uses_worker {
            match text_worker
                .query_completion_stream(
                    &formatted,
                    req_temp,
                    req_top_p,
                    req_max_tokens,
                    req_stop,
                    req_seed,
                    req_fp,
                    req_pp,
                    req_json_schema,
                    grammar_content_for_worker,
                )
                .await
            {
                Ok(bytes_stream) => {
                    use futures::stream::StreamExt;
                    let mut stream = Box::pin(bytes_stream);
                    let mut buffer = Vec::new();
                    while let Some(chunk_res) = stream.next().await {
                        let chunk: bytes::Bytes = match chunk_res {
                            Ok(c) => c,
                            Err(err) => {
                                error!("Error reading stream chunk from worker: {}", err);
                                break;
                            }
                        };
                        buffer.extend_from_slice(&chunk);
                        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                            let line_bytes: Vec<u8> = buffer.drain(..pos + 1).collect();
                            if let Ok(mut line) = String::from_utf8(line_bytes) {
                                if line.ends_with('\n') {
                                    line.pop();
                                }
                                if line.ends_with('\r') {
                                    line.pop();
                                }
                                if line.starts_with("data: ") {
                                    let data = &line["data: ".len()..];
                                    if data == "[DONE]" {
                                        break;
                                    }
                                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data)
                                    {
                                        if let Some(token) = worker_stream_content_delta(&val) {
                                            if !token.is_empty() {
                                                emitted = true;
                                                if tx.send(token.to_string()).await.is_err() {
                                                    return; // Client disconnected
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if emitted {
                        return;
                    }
                }
                Err(err) => {
                    error!("Failed to query completion stream from worker: {}", err);
                }
            }
        }

        if !emitted {
            let run_native = if cfg!(feature = "native") {
                runtime_config.mode != crate::runtime::RuntimeMode::Spawn
            } else {
                false
            };

            if run_native {
                #[cfg(feature = "native")]
                {
                    match brain.native.query_stream(
                        std::path::Path::new(&model_path),
                        &formatted,
                        &system_prompt,
                        &temp_str,
                        req_max_tokens.unwrap_or(512) as usize,
                        grammar_path.clone(),
                    ) {
                        Ok(mut native_rx) => {
                            while let Some(token) = native_rx.recv().await {
                                if !token.trim().is_empty() {
                                    emitted = true;
                                }
                                if tx.send(token).await.is_err() {
                                    return; // Receiver dropped (client disconnected).
                                }
                            }
                        }
                        Err(err) => {
                            error!("[NativeBrain] Native stream error: {}", err);
                        }
                    }
                }
            } else {
                let mut rx = spawn_streaming(
                    &cli_path,
                    &model_path,
                    &formatted,
                    if brain.ultra_low_ram { "0" } else { "999" },
                    &streaming_context,
                    &temp_str,
                    top_p,
                    req_max_tokens,
                    stop,
                    seed,
                    json_schema,
                    grammar_path_for_spawn,
                );

                while let Some(token) = rx.recv().await {
                    if !token.trim().is_empty() {
                        emitted = true;
                    }
                    if tx.send(token).await.is_err() {
                        return; // Receiver dropped (client disconnected).
                    }
                }
            }
        }

        if !emitted {
            let fallback_prompt = if crate::brain::is_prompt_preformatted(&fallback_user_prompt) {
                fallback_user_prompt
            } else {
                wrap_agent_prompt(MIVI_CHAT_SYSTEM_PROMPT, &fallback_user_prompt)
            };
            if let Ok(fallback) = brain
                .query_reasoner(&fallback_prompt, MIVI_CHAT_SYSTEM_PROMPT)
                .await
            {
                let fallback = fallback.trim();
                if !fallback.is_empty() {
                    let _ = tx.send(fallback.to_string()).await;
                }
            }
        }
    });

    let reasoning_chunk = futures::stream::iter(
        agent_reasoning_summary(req, &user_prompt, "streaming")
            .into_iter()
            .map({
                let id = id.clone();
                move |reasoning| {
                    let chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": { "reasoning_content": reasoning },
                            "finish_reason": null
                        }]
                    });
                    Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
                }
            }),
    );

    // Token events: one SSE chunk per received token.
    let id_for_tokens = id.clone();
    let completion_tokens_for_stream = completion_tokens.clone();
    let token_stream = ReceiverStream::new(rx).map(move |token| {
        completion_tokens_for_stream
            .fetch_add(token_counter().count_tokens(&token), Ordering::Relaxed);
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

    let usage_chunk = futures::stream::iter(include_usage.then({
        let id = id.clone();
        let prompt_tokens = token_counter().count_tokens(&user_prompt);
        let completion_tokens = completion_tokens.clone();
        move || {
            let usage = UsageInfo::new(prompt_tokens, completion_tokens.load(Ordering::Relaxed));
            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [],
                "usage": usage_value(usage)
            });
            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        }
    }));

    // [DONE] sentinel per OpenAI spec.
    let done_marker =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let include_preamble = true;
    let preamble_stream = futures::stream::iter(include_preamble.then({
        let id = id.clone();
        move || {
            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant", "content": "" },
                    "finish_reason": null
                }]
            });
            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        }
    }));

    let stream = preamble_stream
        .chain(reasoning_chunk)
        .chain(token_stream)
        .chain(final_chunk)
        .chain(usage_chunk)
        .chain(done_marker)
        .map(move |item| {
            let _keep = &permit;
            item
        });
    Sse::new(stream)
}
