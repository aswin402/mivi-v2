//! Responses API ⇄ Chat Completion API mapping.
//!
//! Extracted from `helpers.rs` (server decomposition): the `/v1/responses`
//! endpoint reuses the chat pipeline, so requests are translated in and
//! responses translated out.

use crate::server::types::*;

pub fn responses_reasoning_effort(req: &ResponsesRequest) -> Option<String> {
    req.reasoning_effort.clone().or_else(|| {
        req.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.get("effort"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
    })
}

pub fn responses_content_to_chat_content(content: serde_json::Value) -> serde_json::Value {
    if let Some(items) = content.as_array() {
        let mapped: Vec<serde_json::Value> = items
            .iter()
            .map(|item| {
                let kind = item.get("type").and_then(|value| value.as_str());
                if matches!(kind, Some("input_text" | "output_text")) {
                    serde_json::json!({
                        "type": "text",
                        "text": item.get("text").and_then(|value| value.as_str()).unwrap_or("")
                    })
                } else {
                    item.clone()
                }
            })
            .collect();
        serde_json::Value::Array(mapped)
    } else {
        content
    }
}

pub fn responses_request_to_chat_request(req: ResponsesRequest) -> ChatCompletionRequest {
    let reasoning_effort = responses_reasoning_effort(&req);
    let messages = match req.input {
        ResponsesInput::Text(text) => vec![ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String(text),
            tool_call_id: None,
            tool_calls: None,
        }],
        ResponsesInput::Messages(messages) => messages
            .into_iter()
            .map(|message| ChatMessage {
                role: message.role,
                content: responses_content_to_chat_content(message.content),
                tool_call_id: None,
                tool_calls: None,
            })
            .collect(),
    };

    ChatCompletionRequest {
        model: req.model,
        messages,
        stream: req.stream,
        tools: req.tools,
        tool_choice: req.tool_choice,
        max_tokens: req.max_output_tokens,
        stop: req.stop,
        seed: req.seed,
        response_format: req.response_format,
        stream_options: req.stream_options,
        parallel_tool_calls: req.parallel_tool_calls,
        reasoning_effort,
        temperature: req.temperature,
        top_p: req.top_p,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        user: req.user,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        n: None,
        service_tier: None,
    }
}

pub fn responses_response_from_chat(chat: ChatCompletionResponse) -> ResponsesResponse {
    let output = chat
        .choices
        .into_iter()
        .flat_map(|choice| {
            let mut items = Vec::new();
            if let Some(calls) = choice.message.tool_calls {
                items.extend(calls.into_iter().map(|call| ResponsesOutputItem {
                    id: call.id.clone(),
                    r#type: "function_call".to_string(),
                    status: Some("completed".to_string()),
                    role: None,
                    content: Vec::new(),
                    name: Some(call.function.name),
                    arguments: Some(call.function.arguments),
                    call_id: Some(call.id),
                }));
            }
            if !choice.message.content.is_empty() {
                items.push(ResponsesOutputItem {
                    id: format!("msg_{}", choice.index),
                    r#type: "message".to_string(),
                    status: Some("completed".to_string()),
                    role: Some(choice.message.role),
                    content: vec![ResponsesOutputContent {
                        r#type: "output_text".to_string(),
                        text: choice.message.content,
                        annotations: Vec::new(),
                    }],
                    name: None,
                    arguments: None,
                    call_id: None,
                });
            }
            items
        })
        .collect();

    ResponsesResponse {
        id: chat.id.replace("chatcmpl", "resp"),
        object: "response".to_string(),
        created_at: chat.created,
        model: chat.model,
        status: "completed".to_string(),
        output,
        usage: chat.usage,
    }
}
