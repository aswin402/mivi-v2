//! Model chat wrappers (reasoner/coder/model_chat) and the non-streaming
//! chat completion pipeline.
//!
//! Extracted from `helpers.rs` (server decomposition).

use crate::brain::EdgeBrain;
use crate::constants::{MIVI_CHAT_SYSTEM_PROMPT, MODEL_NAME};
use std::sync::Arc;

use axum::extract::Json;
use axum::response::IntoResponse;
use tracing::{debug, info, warn};

use crate::server::helpers::unix_timestamp;
use crate::server::helpers::{extract_content, model_prompt_from_request, vision_response};
use crate::server::prompt::*;
use crate::server::streaming::*;
use crate::server::tool_generate::*;
use crate::server::tool_parse::*;
use crate::server::tool_select::*;
use crate::server::types::*;
use crate::server::usage::*;
use crate::trace::{preview as trace_preview, trace_event, TraceConfig};

pub fn is_direct_reasoner_intent(intent: &str) -> bool {
    matches!(
        intent.to_ascii_lowercase().as_str(),
        "chat" | "reason" | "multi_step" | "vision"
    )
}

/// One-shot reasoner call (spawns llama-cli per request).
pub async fn reasoner_chat(
    brain: &EdgeBrain,
    user_prompt: &str,
) -> Result<(String, String), String> {
    let res = brain
        .query_reasoner(user_prompt, MIVI_CHAT_SYSTEM_PROMPT)
        .await?;
    Ok((res, MODEL_NAME.to_string()))
}

pub async fn reasoner_chat_with_params(
    brain: &EdgeBrain,
    user_prompt: &str,
    temp: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    max_tokens: Option<u32>,
) -> Result<(String, String), String> {
    let res = brain
        .query_reasoner_with_params(
            user_prompt,
            MIVI_CHAT_SYSTEM_PROMPT,
            temp,
            top_p,
            seed,
            max_tokens,
        )
        .await?;
    Ok((res, MODEL_NAME.to_string()))
}

/// One-shot coder call (spawns llama-cli per request).
pub async fn code_chat(brain: &EdgeBrain, user_prompt: &str) -> Result<(String, String), String> {
    let res = brain
        .query_coder_with_params(
            user_prompt,
            "You are a coding expert.",
            None,
            None,
            None,
            None,
        )
        .await?;
    Ok((res, MODEL_NAME.to_string()))
}

pub async fn code_chat_with_params(
    brain: &EdgeBrain,
    user_prompt: &str,
    temp: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    max_tokens: Option<u32>,
) -> Result<(String, String), String> {
    let res = brain
        .query_coder_with_params(
            user_prompt,
            "You are a coding expert.",
            temp,
            top_p,
            seed,
            max_tokens,
        )
        .await?;
    Ok((res, MODEL_NAME.to_string()))
}

pub async fn model_chat(
    brain: &EdgeBrain,
    prompt: &str,
    req: &ChatCompletionRequest,
) -> Result<String, String> {
    let grammar_path = get_grammar_path(req);
    brain
        .query_raw(
            prompt,
            req.temperature,
            req.top_p,
            req.max_tokens,
            req.stop.clone(),
            req.seed,
            extract_json_schema(req),
            grammar_path,
        )
        .await
}

pub(crate) async fn tool_model_chat(
    brain: &EdgeBrain,
    prompt: &str,
    req: &ChatCompletionRequest,
) -> Result<String, String> {
    let grammar_path = get_grammar_path(req);
    brain
        .query_tool_raw(
            prompt,
            req.temperature,
            req.top_p,
            req.max_tokens,
            req.stop.clone(),
            req.seed,
            extract_json_schema(req),
            grammar_path,
        )
        .await
}

/// Cap a tool result to ~2000 chars (~500 tokens). Keeps the head, which
/// carries the useful payload, and the tail, which usually holds errors.
pub fn chat_error_response(now: u64, message: String) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-v2-{now}"),
        object: "chat.completion".to_string(),
        created: now,
        model: MODEL_NAME.to_string(),
        usage: None,
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: serde_json::json!({
                    "error": {
                        "type": "invalid_request_error",
                        "message": message
                    }
                })
                .to_string(),
                refusal: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    }
}
pub async fn complete_chat_non_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    now: u64,
) -> Result<ChatCompletionResponse, String> {
    if let Err(err) = validate_response_format(&req) {
        return Ok(chat_error_response(now, err));
    }

    // Deterministic arithmetic fast path: pure math prompts ("2+2",
    // "17% of 3482") are answered exactly without loading the model. Only
    // applies when the caller supplies no tools, the last message is a plain
    // text user turn (no images / tool results to consider), and the whole
    // prompt parses as arithmetic.
    let math_eligible = req
        .messages
        .last()
        .map_or(false, |m| m.role == "user" && m.content.is_string());
    if math_eligible {
        if let Some(answer) = crate::math_eval::try_answer(&latest_user_prompt_text(&req)) {
            let final_text = apply_response_format(answer, &req)?;
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_text(&req, &final_text)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: final_text,
                        refusal: None,
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            });
        }
    }

    if let Some(last_msg) = req.messages.last() {
        if last_msg.role == "tool" {
            let final_text = append_tool_execution_summary(&req, String::new());
            let final_text = apply_response_format(final_text, &req)?;
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_text(&req, &final_text)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: final_text,
                        refusal: None,
                        reasoning_content: Some(
                            "Synthesized tool result answer (no model load)".to_string(),
                        ),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            });
        }
    }

    let target_model = req.model.clone().unwrap_or_else(|| MODEL_NAME.to_string());
    let latest_user_prompt = latest_user_prompt_text(&req);

    if should_use_tool_path(&req, &latest_user_prompt) {
        let (tool_calls, response_text) = generate_tool_calls(&state.brain, &req).await?;
        if !tool_calls.is_empty() {
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_tool_calls(&req, &tool_calls)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: String::new(),
                        refusal: None,
                        reasoning_content: None,
                        tool_calls: Some(tool_calls),
                    },
                    logprobs: None,
                    finish_reason: "tool_calls".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            });
        } else if !response_text.is_empty() {
            // Model generated text instead of tool calls — return it.
            let final_text = response_text;
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_text(&req, &final_text)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: apply_response_format(final_text, &req).unwrap_or_else(|err| {
                            serde_json::json!({"error":{"type":"invalid_request_error","message":err}}).to_string()
                        }),
                        refusal: None,
                        reasoning_content: agent_reasoning_summary(
                            &req,
                            &latest_user_prompt,
                            "tool_text_fallback",
                        ),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            });
        }
    }

    let (user_prompt, image_path) = extract_content(&req);

    let (intent, _confidence) = state
        .router
        .classify_intent(&state.brain, &user_prompt)
        .await;

    // Lightweight chat path: skip heavy retrieval for simple CHAT prompts,
    // use build_chat_prompt which preserves the agent's context directly.
    let model_user_prompt = if intent == "CHAT" && image_path.is_none() {
        build_chat_prompt(&req)
    } else {
        model_prompt_from_request(&req, &user_prompt, &state).await
    };

    // MULTI_STEP now shares the 256-token cap: 512 tokens on a sub-1B model
    // mostly produces hallucinated padding while burning CPU.
    let intent_max_tokens = if intent == "CHAT" {
        128
    } else if intent == "CODE" {
        512
    } else {
        256
    };
    let resolved_max_tokens = Some(req.max_tokens.unwrap_or(intent_max_tokens));

    let (response_text, chosen_model, route) = if image_path.is_some() {
        let path = image_path.unwrap_or_default();
        (
            vision_response(&state.brain, &path, &user_prompt).await?,
            MODEL_NAME.to_string(),
            "vision",
        )
    } else {
        match target_model.to_lowercase().as_str() {
            "coder" => {
                let (text, model) = code_chat_with_params(
                    &state.brain,
                    &model_user_prompt,
                    req.temperature,
                    req.top_p,
                    req.seed,
                    resolved_max_tokens,
                )
                .await?;
                (text, model, "coder")
            }
            "reasoner" => {
                let (text, model) = reasoner_chat_with_params(
                    &state.brain,
                    &model_user_prompt,
                    req.temperature,
                    req.top_p,
                    req.seed,
                    resolved_max_tokens,
                )
                .await?;
                (text, model, "reasoner")
            }
            _ if intent == "CODE" => {
                let (text, model) = code_chat_with_params(
                    &state.brain,
                    &model_user_prompt,
                    req.temperature,
                    req.top_p,
                    req.seed,
                    resolved_max_tokens,
                )
                .await?;
                (text, model, "coder")
            }
            _ => {
                let (text, model) = reasoner_chat_with_params(
                    &state.brain,
                    &model_user_prompt,
                    req.temperature,
                    req.top_p,
                    req.seed,
                    resolved_max_tokens,
                )
                .await?;
                (text, model, "direct_reasoner")
            }
        }
    };

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-v2-{now}"),
        object: "chat.completion".to_string(),
        created: now,
        model: chosen_model,
        usage: Some(estimated_usage_for_text(&req, &response_text)),
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: apply_response_format(response_text, &req).unwrap_or_else(|err| {
                    serde_json::json!({"error":{"type":"invalid_request_error","message":err}})
                        .to_string()
                }),
                refusal: None,
                reasoning_content: agent_reasoning_summary(&req, &user_prompt, route),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    })
}

pub async fn handle_chat_completions(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(mut req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if req.messages.len() > 256 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "Too many messages (max 256)"
                }
            })),
        )
            .into_response();
    }
    if req.tools.as_ref().map(|t| t.len()).unwrap_or(0) > 128 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "Too many tools (max 128)"
                }
            })),
        )
            .into_response();
    }

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

    // Dynamic common prefix/suffix boilerplate stripping for user messages in request
    let mut user_indices = Vec::new();
    let mut user_texts = Vec::new();
    for (idx, msg) in req.messages.iter().enumerate() {
        if msg.role == "user" {
            if let Some(text) = msg.content.as_str() {
                user_indices.push(idx);
                user_texts.push(text.to_string());
            }
        }
    }

    if user_texts.len() >= 2 {
        let mut common_suffix = user_texts[0].clone();
        for text in &user_texts[1..] {
            common_suffix = longest_common_suffix(&common_suffix, text);
        }
        let mut common_prefix = user_texts[0].clone();
        for text in &user_texts[1..] {
            common_prefix = longest_common_prefix(&common_prefix, text);
        }

        let suffix_len = common_suffix.chars().count();
        let prefix_len = common_prefix.chars().count();

        // Only strip if the boilerplate is substantial (e.g. > 60 characters)
        let should_strip_suffix = suffix_len > 60;
        let should_strip_prefix = prefix_len > 60;

        if should_strip_suffix || should_strip_prefix {
            for idx in user_indices {
                if let Some(text) = req.messages[idx].content.as_str() {
                    let mut cleaned = text.to_string();
                    if should_strip_prefix && cleaned.starts_with(&common_prefix) {
                        cleaned = cleaned[common_prefix.len()..].to_string();
                    }
                    if should_strip_suffix && cleaned.ends_with(&common_suffix) {
                        let len = cleaned.chars().count();
                        cleaned = cleaned.chars().take(len - suffix_len).collect::<String>();
                    }
                    req.messages[idx].content =
                        serde_json::Value::String(cleaned.trim().to_string());
                }
            }
        }
    }

    debug!(
        ">>> MIVI REQUEST: model={:?} stream={:?} msgs={} tools={}",
        req.model,
        req.stream,
        req.messages.len(),
        req.tools.as_ref().map(|t| t.len()).unwrap_or(0)
    );

    for (i, msg) in req.messages.iter().enumerate() {
        let preview = match &msg.content {
            serde_json::Value::String(s) => {
                let chars: String = s.chars().take(120).collect();
                format!("str(len={}) {:?}...", s.len(), chars)
            }
            other => format!("{:?}", other),
        };
        let tc = if msg.tool_calls.is_some() {
            " [has tool_calls]"
        } else {
            ""
        };
        let ti = if msg.tool_call_id.is_some() {
            " [has tool_call_id]"
        } else {
            ""
        };
        debug!(
            "  msg[{}]: role={:?} content={}{}{}",
            i, msg.role, preview, tc, ti
        );
    }

    let trace = TraceConfig::from_env();
    let latest_user_prompt = latest_user_prompt_text(&req);
    let now = unix_timestamp();
    let include_usage = include_stream_usage(&req);
    let target_model = req.model.clone().unwrap_or_else(|| MODEL_NAME.to_string());

    // ── Verified tool result answer (no model load) ──────────────────
    if let Some(last_msg) = req.messages.last() {
        if last_msg.role == "tool" {
            let final_text = append_tool_execution_summary(&req, String::new());
            let final_text = match apply_response_format(final_text, &req) {
                Ok(t) => t,
                Err(err) => {
                    return (
                        axum::http::StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {
                                "type": "invalid_request_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            };
            if req.stream.unwrap_or(false) {
                return stream_text_response(
                    final_text.clone(),
                    now,
                    Some("Synthesized tool result answer (no model load)".to_string()),
                    include_usage.then(|| estimated_usage_for_text(&req, &final_text)),
                    permit,
                )
                .into_response();
            }
            return Json(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{now}"),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_text(&req, &final_text)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: final_text,
                        refusal: None,
                        reasoning_content: Some(
                            "Synthesized tool result answer (no model load)".to_string(),
                        ),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            })
            .into_response();
        }
    }

    // Deterministic arithmetic fast path: pure math prompts ("2+2",
    // "17% of 3482", "whats 4*12") are answered exactly without loading the model.
    // Checked before tool calling so agent requests with math don't trigger tool calls.
    let (user_prompt, image_path) = extract_content(&req);
    let math_eligible = image_path.is_none()
        && req
            .messages
            .last()
            .map_or(false, |m| m.role == "user" && m.content.is_string());
    if math_eligible {
        if let Some(answer) = crate::math_eval::try_answer(&user_prompt) {
            let _ = trace_event(
                &trace,
                serde_json::json!({
                    "kind": "final_response",
                    "route": "math_fast_path",
                    "finish_reason": "stop",
                    "response_chars": answer.chars().count()
                }),
            );
            if req.stream.unwrap_or(false) {
                return stream_text_response(
                    answer.clone(),
                    now,
                    None,
                    include_usage.then(|| estimated_usage_for_text(&req, &answer)),
                    permit,
                )
                .into_response();
            }
            return Json(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{}", now),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_text(&req, &answer)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: answer,
                        refusal: None,
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            })
            .into_response();
        }
    }

    let tool_selection = select_tools_for_request(&req);
    let selected_tool_names = tool_names(&tool_selection.selected);
    let selected_tool_roles = selected_tool_roles(&tool_selection.selected);
    let blocked_tools = blocked_tool_names(&tool_selection.blocked);
    // Use already-computed tool_selection instead of calling should_use_tool_path
    // (which internally calls select_tools_for_request again)
    let has_tools = should_generate_tool_calls(&req, &latest_user_prompt, &tool_selection);
    let _ = trace_event(
        &trace,
        serde_json::json!({
            "kind": "request",
            "model": target_model,
            "stream": req.stream.unwrap_or(false),
            "messages": req.messages.len(),
            "tools_in_request": req.tools.as_ref().map(|tools| tools.len()).unwrap_or(0),
            "has_tool_involvement": has_tools,
            "agent_intent": tool_selection.intent.as_str(),
            "selected_tools": selected_tool_names,
            "selected_tool_roles": selected_tool_roles,
            "blocked_tools": blocked_tools,
            "latest_user_prompt_preview": trace_preview(&latest_user_prompt, 240)
        }),
    );

    // ── Tool calling path ────────────────────────────────────────────
    if has_tools {
        info!("[MIVI-V2 Tool] Tool involvement detected, generating tool calls...");
        let (tool_calls, response_text) = match generate_tool_calls(&state.brain, &req).await {
            Ok(res) => res,
            Err(err) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "type": "server_error",
                            "message": err
                        }
                    })),
                )
                    .into_response();
            }
        };

        if !tool_calls.is_empty() {
            info!("[MIVI-V2 Tool] Generated {} tool call(s)", tool_calls.len());
            for tc in &tool_calls {
                info!("  -> {}({})", tc.function.name, tc.function.arguments);
            }
            let _ = trace_event(
                &trace,
                serde_json::json!({
                    "kind": "final_response",
                    "route": "tool_calls",
                    "finish_reason": "tool_calls",
                    "tool_calls": call_names(&tool_calls)
                }),
            );
            let reasoning = agent_reasoning_summary(&req, &latest_user_prompt, "tool_calls");
            if req.stream.unwrap_or(false) {
                let usage =
                    include_usage.then(|| estimated_usage_for_tool_calls(&req, &tool_calls));
                return stream_tool_calls_response(tool_calls, now, reasoning, usage, permit)
                    .into_response();
            }
            return Json(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{}", now),
                object: "chat.completion".to_string(),
                created: now,
                model: MODEL_NAME.to_string(),
                usage: Some(estimated_usage_for_tool_calls(&req, &tool_calls)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: String::new(),
                        refusal: None,
                        reasoning_content: reasoning,
                        tool_calls: Some(tool_calls),
                    },
                    logprobs: None,
                    finish_reason: "tool_calls".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            })
            .into_response();
        } else if !response_text.is_empty() {
            // No tool calls but we have text response — return it.
            let final_text = response_text;
            let chosen_model = MODEL_NAME.to_string();
            let _ = trace_event(
                &trace,
                serde_json::json!({
                    "kind": "final_response",
                    "route": "tool_text_fallback",
                    "finish_reason": if req.stream.unwrap_or(false) { "stream" } else { "stop" },
                    "response_chars": final_text.chars().count()
                }),
            );
            if req.stream.unwrap_or(false) {
                return stream_text_response(
                    final_text.clone(),
                    now,
                    agent_reasoning_summary(&req, &latest_user_prompt, "tool_text_fallback"),
                    include_usage.then(|| estimated_usage_for_text(&req, &final_text)),
                    permit,
                )
                .into_response();
            }
            return Json(ChatCompletionResponse {
                id: format!("chatcmpl-v2-{}", now),
                object: "chat.completion".to_string(),
                created: now,
                model: chosen_model,
                usage: Some(estimated_usage_for_text(&req, &final_text)),
                choices: vec![ChoiceOut {
                    index: 0,
                    message: ChatMessageOut {
                        role: "assistant".to_string(),
                        content: final_text,
                        refusal: None,
                        reasoning_content: agent_reasoning_summary(
                            &req,
                            &latest_user_prompt,
                            "tool_text_fallback",
                        ),
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                system_fingerprint: Some("fp_mivi".to_string()),
            })
            .into_response();
        }
        // Both empty — fall through to regular chat path!
    }

    // ── Non-tool path (existing logic) ───────────────────────────────
    let (intent, confidence) = state
        .router
        .classify_intent(&state.brain, &user_prompt)
        .await;

    // MULTI_STEP now shares the 256-token cap: 512 tokens on a sub-1B model
    // mostly produces hallucinated padding while burning CPU.
    let intent_max_tokens = if intent == "CHAT" {
        128
    } else if intent == "CODE" {
        512
    } else {
        256
    };
    let resolved_max_tokens = Some(req.max_tokens.unwrap_or(intent_max_tokens));

    // Lightweight chat path: for simple CHAT-classified prompts,
    // skip the heavy retrieval pipeline (memory loading, RAG, context compression)
    // and use build_chat_prompt directly — which preserves the agent's system prompt,
    // skills, database, and context as-is. This is what the agent intended.
    let model_user_prompt = if intent == "CHAT" && image_path.is_none() && confidence >= 0.50 {
        info!(
            "[MIVI-V2] Lightweight chat path: skipping retrieval for CHAT intent (conf={:.2})",
            confidence
        );
        build_chat_prompt(&req)
    } else {
        model_prompt_from_request(&req, &user_prompt, &state).await
    };

    // Streaming path.
    if req.stream.unwrap_or(false) {
        if let Some(path) = image_path.as_deref() {
            let answer = match vision_response(&state.brain, path, &user_prompt).await {
                Ok(ans) => ans,
                Err(err) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "type": "server_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            };
            let _ = trace_event(
                &trace,
                serde_json::json!({
                    "kind": "final_response",
                    "route": "streaming_vision",
                    "finish_reason": "stream",
                    "response_chars": answer.chars().count()
                }),
            );
            return stream_text_response(
                answer.clone(),
                now,
                agent_reasoning_summary(&req, &user_prompt, "streaming_vision"),
                include_usage.then(|| estimated_usage_for_text(&req, &answer)),
                permit,
            )
            .into_response();
        }

        let _ = trace_event(
            &trace,
            serde_json::json!({
                "kind": "final_response",
                "route": "streaming",
                "finish_reason": "stream"
            }),
        );
        return handle_streaming(
            state,
            model_user_prompt,
            &req,
            now,
            include_usage,
            permit,
            resolved_max_tokens,
        )
        .await
        .into_response();
    }

    // Non-streaming path.
    info!(
        "[MIVI-V2 Server] Intent: {} (conf: {:.2}) | Model: '{}' | Prompt: '{}'",
        intent, confidence, target_model, user_prompt
    );

    let (response_text, chosen_model, route) = if image_path.is_some() {
        let path = image_path.unwrap_or_default();
        match vision_response(&state.brain, &path, &user_prompt).await {
            Ok(response) => (response, MODEL_NAME.to_string(), "vision"),
            Err(err) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "type": "server_error",
                            "message": err
                        }
                    })),
                )
                    .into_response();
            }
        }
    } else {
        match target_model.to_lowercase().as_str() {
            "coder" => match code_chat_with_params(
                &state.brain,
                &model_user_prompt,
                req.temperature,
                req.top_p,
                req.seed,
                resolved_max_tokens,
            )
            .await
            {
                Ok((text, model)) => (text, model, "coder"),
                Err(err) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "type": "server_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            },
            "reasoner" => match reasoner_chat_with_params(
                &state.brain,
                &model_user_prompt,
                req.temperature,
                req.top_p,
                req.seed,
                resolved_max_tokens,
            )
            .await
            {
                Ok((text, model)) => (text, model, "reasoner"),
                Err(err) => {
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "type": "server_error",
                                "message": err
                            }
                        })),
                    )
                        .into_response();
                }
            },
            _ => {
                match reasoner_chat_with_params(
                    &state.brain,
                    &model_user_prompt,
                    req.temperature,
                    req.top_p,
                    req.seed,
                    resolved_max_tokens,
                )
                .await
                {
                    Ok((text, model)) => (text, model, "direct_reasoner"),
                    Err(err) => {
                        return (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                    "error": {
                                    "type": "server_error",
                                    "message": err
                                }
                            })),
                        )
                            .into_response();
                    }
                }
            }
        }
    };
    let _ = trace_event(
        &trace,
        serde_json::json!({
            "kind": "final_response",
            "route": route,
            "intent": intent,
            "confidence": confidence,
            "model": chosen_model,
            "response_chars": response_text.chars().count(),
            "context_prompt_chars": model_user_prompt.chars().count()
        }),
    );

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-v2-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: chosen_model,
        usage: Some(estimated_usage_for_text(&req, &response_text)),
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: response_text,
                refusal: None,
                reasoning_content: agent_reasoning_summary(&req, &user_prompt, route),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    })
    .into_response()
}
