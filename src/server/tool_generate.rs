//! Tool-call generation: decide whether the tool path applies, build the
//! grammar-constrained generation, and validate/repair the result.
//!
//! Extracted from `helpers.rs` (server decomposition).

use std::sync::Arc;

use crate::brain::EdgeBrain;
use crate::server::chat::tool_model_chat;
use crate::server::prompt::latest_user_prompt_text;
use crate::server::prompt::*;
use crate::server::tool_parse::*;
use crate::server::tool_select::prompt_tools_for_request;
use crate::server::tool_select::select_tools_for_request;
use crate::server::tool_select::{blocked_tool_names, selected_tool_roles};
use crate::server::types::*;
use crate::server::usage::extract_json_schema;
use crate::tool_filter::has_tool_intent;
use crate::trace::{trace_event, TraceConfig};

pub fn has_tool_involvement(req: &ChatCompletionRequest) -> bool {
    // Explicit tool_choice overrides
    if let Some(serde_json::Value::String(choice)) = &req.tool_choice {
        if choice == "none" {
            return false;
        }
        if choice == "required" {
            return true;
        }
    }
    if matches!(req.tool_choice, Some(serde_json::Value::Object(_))) {
        return true;
    }
    // Per OpenAI spec: when tools are present and tool_choice is "auto" or absent,
    // the model decides whether to call tools. Always route through tool-call path.
    match &req.tools {
        Some(tools) if !tools.is_empty() => true,
        _ => false,
    }
}

/// Decide whether tool generation is justified by the request, rather than
/// merely by the presence of an OpenAI-compatible tools catalog.
pub fn should_generate_tool_calls(
    req: &ChatCompletionRequest,
    latest_user_prompt: &str,
    selection: &ToolSelection,
) -> bool {
    if !has_tool_involvement(req) || selection.selected.is_empty() {
        return false;
    }

    if let Some(serde_json::Value::String(choice)) = &req.tool_choice {
        if choice == "none" {
            return false;
        }
        if choice == "required" {
            return true;
        }
    }

    if matches!(req.tool_choice, Some(serde_json::Value::Object(_))) {
        return true;
    }

    if selection.intent == AgentIntent::ToolCall || selection.intent.is_inventory() {
        return true;
    }

    req.tools
        .as_deref()
        .map(|tools| has_tool_intent(latest_user_prompt, tools))
        .unwrap_or(false)
}

/// Check if the request should proceed to the tool-calling generator or if it is simple chat.
pub fn should_use_tool_path(req: &ChatCompletionRequest, latest_user_prompt: &str) -> bool {
    let selection = select_tools_for_request(req);
    should_generate_tool_calls(req, latest_user_prompt, &selection)
}

// ──────────────────────────────────────────────
// Backend model calls
// ──────────────────────────────────────────────

pub fn get_grammar_path(req: &ChatCompletionRequest) -> Option<String> {
    if latest_non_system_role(req) == Some("tool") {
        return None;
    }
    if let Some(ref tools) = req.tools {
        if !tools.is_empty() {
            let format = std::env::var("MIVI_TOOL_FORMAT").unwrap_or_else(|_| "openai".to_string());
            let format = format.trim().to_ascii_lowercase();
            let base_path = match format.as_str() {
                "openai" => "configs/grammars/openai_tool_call.gbnf",
                "hermes" => "configs/grammars/hermes_tool_call.gbnf",
                _ => "configs/grammars/openai_tool_call.gbnf",
            };

            // Read the base grammar content
            if let Ok(content) = std::fs::read_to_string(base_path) {
                let prompt_tools = prompt_tools_for_request(req);
                if !prompt_tools.is_empty() {
                    let mut name_rules = Vec::new();
                    for t in &prompt_tools {
                        name_rules.push(format!("\"\\\"{}\\\"\"", t.function.name));
                    }
                    let name_rule = name_rules.join(" | ");

                    let target = "\"\\\"name\\\"\" \":\" string";
                    let replacement = format!("\"\\\"name\\\"\" \":\" ({})", name_rule);
                    let new_content = content.replace(target, &replacement);

                    // Write to a temporary file
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    use std::hash::{Hash, Hasher};
                    name_rule.hash(&mut hasher);
                    let hash = hasher.finish();
                    let temp_dir = std::env::temp_dir();
                    let temp_path = temp_dir.join(format!("mivi_grammar_{}_{}.gbnf", format, hash));
                    if std::fs::write(&temp_path, new_content).is_ok() {
                        return Some(temp_path.to_string_lossy().into_owned());
                    }
                }
            }
            return Some(base_path.to_string());
        }
    }
    if let Some(ref fmt) = req.response_format {
        if fmt.get("type").and_then(|t| t.as_str()) == Some("json_object") {
            if extract_json_schema(req).is_none() {
                return Some("configs/grammars/json_object.gbnf".to_string());
            }
        }
    }
    None
}

/// Run the model with a full multi-turn prompt (already formatted with <|im_start|> tags).
pub fn clamp_tool_result(content: &str) -> String {
    const MAX_CHARS: usize = 2000;
    const TAIL_CHARS: usize = 400;
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= MAX_CHARS {
        return content.to_string();
    }
    let head: String = chars[..MAX_CHARS - TAIL_CHARS].iter().collect();
    let tail: String = chars[chars.len() - TAIL_CHARS..].iter().collect();
    format!(
        "{}\n[…truncated {} chars…]\n{}",
        head,
        chars.len() - MAX_CHARS,
        tail
    )
}

/// Delete `mivi_grammar_*.gbnf` temp files older than one hour. Called once at
/// server start; per-request grammar files are content-addressed and would
/// otherwise accumulate in /tmp forever.
pub fn sweep_stale_grammar_files() {
    let temp_dir = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&temp_dir) else {
        return;
    };
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("mivi_grammar_") || !name.ends_with(".gbnf") {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if let Ok(modified) = meta.modified() {
            if modified < cutoff {
                if std::fs::remove_file(entry.path()).is_ok() {
                    removed += 1;
                }
            }
        }
    }
    if removed > 0 {
        tracing::info!("[startup] Swept {} stale grammar temp files", removed);
    }
}

pub fn append_tool_execution_summary(req: &ChatCompletionRequest, content: String) -> String {
    let mut tool_messages = Vec::new();
    for msg in &req.messages {
        if msg.role == "tool" {
            tool_messages.push(msg);
        }
    }

    if tool_messages.is_empty() {
        return content;
    }

    let mut summary_parts = Vec::new();
    summary_parts.push("### Tool Results".to_string());

    for tool_msg in &tool_messages {
        let tool_call_id = tool_msg.tool_call_id.as_deref().unwrap_or("unknown");
        let raw_content = if let Some(s) = tool_msg.content.as_str() {
            s.to_string()
        } else {
            tool_msg.content.to_string()
        };

        // Try to find the corresponding assistant tool call in history to get the function name
        let mut matched_name = None;
        for msg in &req.messages {
            if msg.role == "assistant" {
                if let Some(ref calls) = msg.tool_calls {
                    for call in calls {
                        if call.id == tool_call_id {
                            matched_name = Some(call.function.name.clone());
                            break;
                        }
                    }
                }
            }
        }

        if let Some(name) = matched_name {
            let lower_content = raw_content.to_lowercase();
            if lower_content.contains("timeout") || lower_content.contains("timed out") {
                summary_parts.push(format!("- Tool `{}` returned error: timeout.", name));
            } else {
                // One-line summary only: the full result stays in the message
                // history the agent re-submits, so re-embedding it here would
                // double-count large tool payloads.
                summary_parts.push(format!(
                    "- Tool `{}` returned: {}.",
                    name,
                    clamp_tool_result(&raw_content)
                ));
            }
        } else {
            // Unmatched tool result
            summary_parts.push(format!(
                "- Protocol issue: unmatched tool result for `{}`.",
                tool_call_id
            ));
        }
    }

    let mut merged = content;
    if !merged.is_empty() {
        merged.push_str("\n\n");
    }
    merged.push_str(&summary_parts.join("\n"));
    merged
}

pub fn last_tool_result_is_error(req: &ChatCompletionRequest) -> bool {
    for msg in req.messages.iter().rev() {
        if msg.role == "tool" {
            let content = if let Some(s) = msg.content.as_str() {
                s.to_string()
            } else {
                msg.content.to_string()
            };
            let lower = content.to_lowercase();
            if lower.contains("error")
                || lower.contains("fail")
                || lower.contains("timeout")
                || lower.contains("timed out")
            {
                return true;
            }
        } else {
            break;
        }
    }
    false
}

async fn query_model_for_tool_calls(
    brain: &EdgeBrain,
    req: &ChatCompletionRequest,
) -> Result<(Vec<ToolCallOut>, String), String> {
    let prompt = build_chat_prompt(req);
    // Tool-specialized weights are selected from the catalog. Keep generation
    // bounded because the response is a small structured call.
    let mut capped_req = req.clone();
    capped_req.max_tokens = Some(capped_req.max_tokens.unwrap_or(256).min(256));
    let raw = tool_model_chat(brain, &prompt, &capped_req).await?;
    let parsed_calls = parse_tool_calls(&raw);
    Ok((parsed_calls, raw))
}

fn validate_generated_tool_calls(
    calls: &[ToolCallOut],
    selected_tools: &[ToolDef],
) -> (Vec<ToolCallOut>, Vec<String>) {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    for call in calls {
        let Some(tool) = selected_tools
            .iter()
            .find(|tool| tool.function.name == call.function.name)
        else {
            errors.push(format!("Unknown tool '{}'", call.function.name));
            continue;
        };

        match validate_tool_call_arguments(call, tool) {
            Ok(()) => valid.push(call.clone()),
            Err(error) => errors.push(format!("{}: {}", call.function.name, error)),
        }
    }

    (valid, errors)
}

/// Generate tool calls: run the model with tool-aware prompt, parse tool calls.
pub async fn generate_tool_calls(
    brain: &EdgeBrain,
    req: &ChatCompletionRequest,
) -> Result<(Vec<ToolCallOut>, String), String> {
    // Loop guard: when the agent has already repeated an identical tool
    // call too many times, answer with an explanatory message instead of
    // generating the same call again (or erroring) so the agent can recover.
    if let Some(loop_reason) = crate::stability::check_history_for_loops(&req.messages) {
        return Ok((Vec::new(), loop_reason));
    }

    let trace = TraceConfig::from_env();
    let selection = select_tools_for_request(req);
    let selected_tools = selection.selected;

    // If no tools matched the prompt, skip model call entirely.
    // The caller will fall through to the regular chat path.
    if selected_tools.is_empty() {
        return Ok((Vec::new(), String::new()));
    }

    let selected_tool_names = tool_names(&selected_tools);
    let selected_tool_roles = selected_tool_roles(&selected_tools);
    let blocked_tools = blocked_tool_names(&selection.blocked);

    let (parsed_calls, raw) = query_model_for_tool_calls(brain, req).await?;
    let mut final_raw = raw;

    if parsed_calls.is_empty() {
        let final_content = append_tool_execution_summary(req, final_raw);
        return Ok((Vec::new(), final_content));
    }

    // Validate parsed tool calls against selected tools. If the model emitted
    // a structurally valid call with schema-invalid arguments, give it one
    // schema-driven correction turn instead of silently turning the request
    // into a normal text answer.
    let (mut valid_calls, validation_errors) =
        validate_generated_tool_calls(&parsed_calls, &selected_tools);
    let mut route = "single_pass";

    if valid_calls.is_empty() && !validation_errors.is_empty() {
        let mut retry_req = req.clone();
        retry_req.messages.push(ChatMessage {
            role: "user".to_string(),
            content: serde_json::Value::String(format!(
                "The previous tool call was invalid: {}. Return one corrected tool call only. Follow the provided JSON Schema exactly.",
                validation_errors.join("; ")
            )),
            tool_call_id: None,
            tool_calls: None,
        });

        let (retry_parsed, retry_raw) = query_model_for_tool_calls(brain, &retry_req).await?;
        let (retry_valid, _) = validate_generated_tool_calls(&retry_parsed, &selected_tools);
        if !retry_valid.is_empty() {
            valid_calls = retry_valid;
            final_raw = retry_raw;
            route = "schema_retry";
        }
    }

    let parsed_count = valid_calls.len();
    let _ = trace_event(
        &trace,
        serde_json::json!({
            "kind": "tool_generation",
            "route": route,
            "agent_intent": selection.intent.as_str(),
            "selected_tools": selected_tool_names,
            "selected_tool_roles": selected_tool_roles,
            "blocked_tools": blocked_tools,
            "parsed_tool_calls": parsed_count,
            "accepted_tool_calls": call_names(&valid_calls)
        }),
    );

    if !valid_calls.is_empty() {
        Ok((valid_calls, String::new()))
    } else {
        let final_content = append_tool_execution_summary(req, final_raw);
        Ok((Vec::new(), final_content))
    }
}

// ──────────────────────────────────────────────
// Chat completions handler
// ──────────────────────────────────────────────
