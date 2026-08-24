//! Model chat wrappers (reasoner/coder/model_chat) and the non-streaming
//! chat completion pipeline.
//!
//! Extracted from `helpers.rs` (server decomposition).

use crate::brain::EdgeBrain;
use crate::constants::{MIVI_CHAT_SYSTEM_PROMPT, MODEL_NAME};
use std::sync::Arc;

use crate::server::helpers::{extract_content, model_prompt_from_request, vision_response};
use crate::server::prompt::*;
use crate::server::tool_generate::*;
use crate::server::tool_parse::*;
use crate::server::types::*;
use crate::server::usage::*;

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
