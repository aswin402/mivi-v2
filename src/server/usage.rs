//! Token counting and usage estimation for the OpenAI-compatible API.
//!
//! Extracted from `helpers.rs` (server decomposition): everything that turns
//! a request/completion pair into `UsageInfo` counts, plus response_format
//! validation helpers.

use std::path::Path;
use std::process::Command;

use crate::server::types::*;

pub fn response_format_type(req: &ChatCompletionRequest) -> Option<String> {
    req.response_format
        .as_ref()
        .and_then(|format| format.get("type"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

pub fn validate_response_format(req: &ChatCompletionRequest) -> Result<(), String> {
    match response_format_type(req).as_deref() {
        None | Some("text") | Some("json_object") | Some("json_schema") => Ok(()),
        Some(other) => Err(format!("unsupported response_format type `{other}`")),
    }
}

pub fn extract_json_schema(req: &ChatCompletionRequest) -> Option<String> {
    req.response_format
        .as_ref()
        .and_then(|format| format.get("json_schema"))
        .and_then(|js| js.get("schema"))
        .map(|schema| schema.to_string())
}

pub fn apply_response_format(
    content: String,
    req: &ChatCompletionRequest,
) -> Result<String, String> {
    validate_response_format(req)?;
    if response_format_type(req).as_deref() == Some("json_object") {
        return Ok(serde_json::json!({ "answer": content }).to_string());
    }
    Ok(content)
}

pub trait TokenCounter {
    fn count_tokens(&self, text: &str) -> u32;
}

#[allow(dead_code)]
pub fn count_with_llama_cpp_tokenizer(command: &Path, model: &Path, text: &str) -> Option<u32> {
    let output = Command::new(command)
        .arg("--model")
        .arg(model)
        .arg("--tokenize")
        .arg("--prompt")
        .arg(text)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        combined.push('\n');
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    count_llama_tokenize_output(&combined)
}

pub fn count_llama_tokenize_output(output: &str) -> Option<u32> {
    let mut in_list = false;
    let mut digits = String::new();
    let mut count = 0_u32;

    for ch in output.chars() {
        match ch {
            '[' if !in_list => {
                in_list = true;
                digits.clear();
            }
            ']' if in_list => {
                if !digits.is_empty() {
                    count = count.saturating_add(1);
                    digits.clear();
                }
                return Some(count);
            }
            ch if in_list && ch.is_ascii_digit() => digits.push(ch),
            _ if in_list => {
                if !digits.is_empty() {
                    count = count.saturating_add(1);
                    digits.clear();
                }
            }
            _ => {}
        }
    }

    None
}

pub fn token_counter() -> RuntimeTokenCounter {
    TokenCounterConfig::from_env().counter()
}

pub fn value_text_for_usage(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(ToString::to_string)
        .unwrap_or_else(|| value.to_string())
}

pub fn estimated_prompt_tokens(req: &ChatCompletionRequest) -> u32 {
    let counter = token_counter();
    let message_tokens = req.messages.iter().fold(0_u32, |total, message| {
        total
            .saturating_add(counter.count_tokens(&message.role))
            .saturating_add(counter.count_tokens(&value_text_for_usage(&message.content)))
            .saturating_add(4)
    });
    let tool_tokens = req
        .tools
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .fold(0_u32, |total, tool| {
            total
                .saturating_add(counter.count_tokens(&tool.function.name))
                .saturating_add(
                    tool.function
                        .description
                        .as_deref()
                        .map(|description| counter.count_tokens(description))
                        .unwrap_or(0),
                )
                .saturating_add(
                    tool.function
                        .parameters
                        .as_ref()
                        .map(|parameters| counter.count_tokens(&parameters.to_string()))
                        .unwrap_or(0),
                )
        });
    message_tokens.saturating_add(tool_tokens)
}

pub fn estimated_usage_for_text(req: &ChatCompletionRequest, completion: &str) -> UsageInfo {
    UsageInfo::new(
        estimated_prompt_tokens(req),
        token_counter().count_tokens(completion),
    )
}

pub fn estimated_usage_for_tool_calls(
    req: &ChatCompletionRequest,
    calls: &[ToolCallOut],
) -> UsageInfo {
    let completion_text = calls
        .iter()
        .map(|call| format!("{} {}", call.function.name, call.function.arguments))
        .collect::<Vec<_>>()
        .join("\n");
    estimated_usage_for_text(req, &completion_text)
}

pub fn usage_value(usage: UsageInfo) -> serde_json::Value {
    serde_json::to_value(usage).unwrap_or_else(|_| {
        serde_json::json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    })
}

pub fn include_stream_usage(req: &ChatCompletionRequest) -> bool {
    req.stream_options
        .as_ref()
        .and_then(|options| options.get("include_usage"))
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}
