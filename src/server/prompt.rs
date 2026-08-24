// Chat prompt assembly: template rendering, agent contract, history
// sanitization, and reasoning-summary generation.
//
// Extracted from `helpers.rs` (server decomposition).

use crate::runtime::RuntimeConfig;
use crate::server::helpers::{clamp_tool_result, tool_names};
use crate::server::tool_select::{
    blocked_tool_names, prompt_tools_for_request, select_tools_for_request,
};
use crate::server::types::*;
use crate::server::usage::value_text_for_usage;
use crate::trace::preview as trace_preview;

pub fn strip_tagged_block_prefix<'a>(text: &'a str, tag: &str) -> &'a str {
    let close = format!("</{}>", tag);
    if text.trim_start().starts_with(&format!("<{}>", tag)) {
        if let Some(end) = text.find(&close) {
            return text[end + close.len()..].trim();
        }
    }
    text.trim()
}

pub fn longest_common_prefix(s1: &str, s2: &str) -> String {
    let mut prefix = String::new();
    let mut chars1 = s1.chars();
    let mut chars2 = s2.chars();
    loop {
        match (chars1.next(), chars2.next()) {
            (Some(c1), Some(c2)) if c1 == c2 => {
                prefix.push(c1);
            }
            _ => break,
        }
    }
    prefix
}

pub fn longest_common_suffix(s1: &str, s2: &str) -> String {
    let mut suffix_chars = Vec::new();
    let mut chars1 = s1.chars().rev();
    let mut chars2 = s2.chars().rev();
    loop {
        match (chars1.next(), chars2.next()) {
            (Some(c1), Some(c2)) if c1 == c2 => {
                suffix_chars.push(c1);
            }
            _ => break,
        }
    }
    suffix_chars.into_iter().rev().collect()
}

pub fn normalize_user_prompt_text(text: &str) -> String {
    let mut normalized = text.trim();
    normalized = strip_tagged_block_prefix(normalized, "available-skills");
    normalized = strip_tagged_block_prefix(normalized, "skill-evaluation-required");
    normalized = strip_tagged_block_prefix(normalized, "user-prompt-submit-hook");
    normalized.trim().to_string()
}

pub fn user_prompt_text_parts(msg: &ChatMessage) -> Vec<String> {
    if let Some(text) = msg.content.as_str() {
        vec![text.to_string()]
    } else if let Some(arr) = msg.content.as_array() {
        arr.iter()
            .filter_map(|item| {
                let item_type = item.get("type").and_then(|v| v.as_str())?;
                if item_type != "text" {
                    return None;
                }
                item.get("text")
                    .and_then(|v| v.as_str())
                    .map(|text| text.to_string())
            })
            .collect()
    } else {
        Vec::new()
    }
}

pub fn latest_user_prompt_text(req: &ChatCompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .filter(|m| m.role == "user")
        .flat_map(|msg| user_prompt_text_parts(msg).into_iter().rev())
        .map(|text| normalize_user_prompt_text(&text))
        .find(|text| !text.is_empty())
        .unwrap_or_default()
}

pub fn latest_non_system_role(req: &ChatCompletionRequest) -> Option<&str> {
    req.messages
        .iter()
        .rev()
        .find(|msg| msg.role != "system")
        .map(|msg| msg.role.as_str())
}

pub fn active_chat_template() -> crate::model_catalog::ChatTemplateConfig {
    static CONFIG: std::sync::LazyLock<crate::model_catalog::ChatTemplateConfig> =
        std::sync::LazyLock::new(|| {
            if let Ok(catalog) = crate::model_catalog::ModelCatalog::load_default() {
                if let Some(entry) = catalog
                    .models
                    .iter()
                    .find(|e| e.enabled && e.role == crate::model_catalog::ModelRole::Reasoner)
                {
                    if let Some(ref template) = entry.chat_template {
                        return template.clone();
                    }
                }
            }
            crate::model_catalog::ChatTemplateConfig::default()
        });
    CONFIG.clone()
}

// ──────────────────────────────────────────────
// Response structs
// ──────────────────────────────────────────────

// ──────────────────────────────────────────────
// Handlers
// ──────────────────────────────────────────────

// ──────────────────────────────────────────────
// Prompt building
// ──────────────────────────────────────────────

pub fn build_chat_prompt(req: &ChatCompletionRequest) -> String {
    let config = RuntimeConfig::global();

    // Check if total token count of messages exceeds 80% of max_input_tokens
    let total_tokens = req
        .messages
        .iter()
        .map(|m| {
            let text = value_text_for_usage(&m.content);
            crate::tokenizer::count_tokens(&text) as usize
        })
        .sum::<usize>();

    let compressed_messages = if total_tokens > (config.context.max_input_tokens * 80 / 100) {
        crate::context_compressor::compress_request_messages(&req.messages, config.context)
    } else {
        // Strip think blocks from past assistant turns
        req.messages
            .iter()
            .map(|m| {
                let mut new_m = m.clone();
                if new_m.role == "assistant" {
                    if let Some(text) = new_m.content.as_str() {
                        let cleaned = crate::brain::strip_think_blocks(text);
                        new_m.content = serde_json::json!(cleaned);
                    }
                }
                new_m
            })
            .collect()
    };

    let t = active_chat_template();
    let mut prompt = String::new();

    // Estimate message token cost first
    let message_cost: usize = compressed_messages
        .iter()
        .map(|m| {
            let text = if m.role == "user" {
                extract_user_text(m)
            } else {
                m.content.as_str().unwrap_or("").to_string()
            };
            text.len() / 4 + 20 // +20 for role markers/template overhead
        })
        .sum();

    // Reserve space: 350 tokens for agent contract / response headroom
    let budget = config.context.max_input_tokens;
    let remaining_for_tools = budget.saturating_sub(message_cost + 350);

    let all_tools = prompt_tools_for_request(req);
    // Budget-aware tool selection: each tool schema is estimated to be ~300 tokens
    let max_tools_by_budget = remaining_for_tools / 300;
    let prompt_tools = if all_tools.len() > max_tools_by_budget {
        all_tools.into_iter().take(max_tools_by_budget).collect()
    } else {
        all_tools
    };

    let persona = crate::lora_router::resolve_specialist_persona(
        req.model.as_deref(),
        !prompt_tools.is_empty(),
        "",
        false,
    );
    let agent_contract = agent_contract_prompt_for_tools_with_persona(&prompt_tools, Some(persona));
    let func_block = build_function_list_block(&prompt_tools);

    let has_user_system = compressed_messages.iter().any(|m| m.role == "system");
    if has_user_system {
        for msg in &compressed_messages {
            if msg.role == "system" {
                if let Some(text) = msg.content.as_str() {
                    if !text.is_empty() {
                        let system_text = wrap_agent_prompt(&agent_contract, text);
                        prompt.push_str(&format!(
                            "{}{}{}",
                            t.system_prefix, system_text, t.system_suffix
                        ));
                    }
                }
            }
        }
    } else {
        prompt.push_str(&format!(
            "{}{}{}",
            t.system_prefix, agent_contract, t.system_suffix
        ));
    }

    // Conversation turns. The tool block is appended to EVERY user turn:
    // the tool-tuned model expects it next to the request, and identical
    // rendering across turns keeps the prompt a stable prefix so the KV
    // cache can be reused between agent turns.
    let has_tools = !prompt_tools.is_empty();
    for (idx, msg) in compressed_messages.iter().enumerate() {
        let _ = idx;
        match msg.role.as_str() {
            "user" => {
                let text = extract_user_text(msg);
                if !text.is_empty() {
                    if has_tools {
                        let text_with_tools = format!("{}\n{}", text, func_block.trim());
                        prompt.push_str(&format!(
                            "{}{}{}",
                            t.user_prefix, text_with_tools, t.user_suffix
                        ));
                    } else {
                        prompt.push_str(&format!("{}{}{}", t.user_prefix, text, t.user_suffix));
                    }
                }
            }
            "assistant" => {
                if let Some(ref calls) = msg.tool_calls {
                    let content =
                        sanitize_assistant_history_text(msg.content.as_str().unwrap_or(""));
                    let mut block = String::new();
                    if !content.is_empty() {
                        block.push_str(&content);
                        block.push('\n');
                    }
                    for tc in calls {
                        block.push_str(&format!(
                            "{{\"name\": \"{}\", \"arguments\": {}}}\n",
                            tc.function.name, tc.function.arguments
                        ));
                    }
                    if !block.is_empty() {
                        prompt.push_str(&format!(
                            "{}{}{}",
                            t.assistant_prefix,
                            block.trim(),
                            t.assistant_suffix
                        ));
                    }
                } else {
                    let text = sanitize_assistant_history_text(msg.content.as_str().unwrap_or(""));
                    if !text.is_empty() {
                        prompt.push_str(&format!(
                            "{}{}{}",
                            t.assistant_prefix, text, t.assistant_suffix
                        ));
                    }
                }
            }
            "tool" => {
                let tool_content = msg.content.as_str().unwrap_or("");
                // Clamp each tool result so long agent loops (file reads,
                // command output) cannot inflate the prompt — and with it the
                // spawn-mode KV allocation — without bound.
                let tool_content = clamp_tool_result(tool_content);
                let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
                let resolved_prefix = t.tool_prefix.replace("{id}", tool_id);
                prompt.push_str(&format!(
                    "{}{}{}",
                    resolved_prefix, tool_content, t.tool_suffix
                ));
            }
            _ => {}
        }
    }

    prompt.push_str(&t.assistant_start);
    prompt
}

pub fn agent_contract_prompt(req: &ChatCompletionRequest) -> String {
    let prompt_tools = prompt_tools_for_request(req);
    agent_contract_prompt_for_tools(&prompt_tools)
}

pub fn agent_contract_prompt_for_tools(tools: &[ToolDef]) -> String {
    agent_contract_prompt_for_tools_with_persona(tools, None)
}

pub fn agent_contract_prompt_for_tools_with_persona(
    tools: &[ToolDef],
    persona: Option<crate::lora_router::SpecialistPersona>,
) -> String {
    let mut lines = vec![
        "Agent contract:".to_string(),
        "- External model identity is `mivi`; do not expose internal worker names.".to_string(),
    ];

    if let Some(p) = persona {
        lines.push(format!("- {}", p.system_prompt_directive()));
    }

    if !tools.is_empty() {
        lines.push("- The calling agent supplies the authoritative instructions, tools, skills, memory, database/context, and retrieved facts.".to_string());
        lines.push("- Use only capabilities present in the current request or context; do not invent agent features.".to_string());
        lines.push("- Prefer available introspection/inventory tools for capability questions; otherwise summarize received tool schemas.".to_string());
        lines.push("- For tool use, choose the smallest relevant tool set and return valid tool-call JSON only when a tool is required.".to_string());
        lines.push("- For conversational messages, greetings, or questions that do not need tools, respond directly in plain text without making tool calls.".to_string());

        let mut names = tool_names(tools);
        names.sort_unstable();
        names.dedup();
        let shown: Vec<String> = names.iter().take(12).cloned().collect();
        let hidden = names.len().saturating_sub(shown.len());
        let suffix = if hidden > 0 {
            format!(" plus {hidden} more")
        } else {
            String::new()
        };
        lines.push(format!(
            "- Current prompt exposes {} selected callable tool schemas: {}{}.",
            names.len(),
            shown.join(", "),
            suffix
        ));
    } else {
        lines.push("- Current prompt exposes no selected callable tool schemas.".to_string());
        lines.push("- Respond directly in natural, helpful plain text.".to_string());
    }

    lines.join("\n")
}

pub fn wrap_agent_prompt(agent_contract: &str, prompt: &str) -> String {
    if prompt.trim().is_empty() {
        agent_contract.to_string()
    } else {
        format!("{}\n\n{}", agent_contract, prompt.trim())
    }
}

pub fn build_function_list_block(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut block = String::new();
    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string());

    block.push_str(&format!(
        "\n# Tools\n\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>\n\nIf no tool call is needed, answer the user directly in plain text.\n",
        tools_json
    ));

    block
}

/// Extract user text from a message (handles both string and multimodal content arrays).
pub fn extract_user_text(msg: &ChatMessage) -> String {
    if let Some(text) = msg.content.as_str() {
        strip_available_skills(text)
    } else if let Some(arr) = msg.content.as_array() {
        for item in arr {
            if let Some(t) = item.get("type").and_then(|v| v.as_str()) {
                if t == "text" {
                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                        let stripped = strip_available_skills(text);
                        if !stripped.is_empty() {
                            return stripped;
                        }
                    }
                }
            }
        }
        String::new()
    } else {
        String::new()
    }
}

pub fn sanitize_assistant_history_text(text: &str) -> String {
    let mut remaining = text;
    let mut cleaned = String::new();

    loop {
        let Some(start) = remaining.to_ascii_lowercase().find("<think>") else {
            cleaned.push_str(remaining);
            break;
        };
        cleaned.push_str(&remaining[..start]);
        let after_open = &remaining[start + "<think>".len()..];
        let after_open_lower = after_open.to_ascii_lowercase();
        if let Some(end) = after_open_lower.find("</think>") {
            remaining = &after_open[end + "</think>".len()..];
        } else {
            break;
        }
    }

    cleaned.trim().to_string()
}

pub fn strip_available_skills(text: &str) -> String {
    if let Some(end) = text.find("</available-skills>") {
        let after = &text[end + "</available-skills>".len()..];
        after.trim().to_string()
    } else {
        text.to_string()
    }
}

pub fn reasoning_summary_enabled() -> bool {
    std::env::var("MIVI_AGENT_REASONING_SUMMARY")
        .map(|value| !matches!(value.trim(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}

pub fn agent_reasoning_summary(
    req: &ChatCompletionRequest,
    user_prompt: &str,
    route: &str,
) -> Option<String> {
    if !reasoning_summary_enabled() {
        return None;
    }

    if route != "verified_tools" && route != "tool_calls" && route != "tool_text_fallback" {
        return None;
    }

    let selection = select_tools_for_request(req);
    let selected = tool_names(&selection.selected);
    let blocked = blocked_tool_names(&selection.blocked);
    let prompt_preview = trace_preview(user_prompt, 96);
    let selected_text = if selected.is_empty() {
        "no selected tools".to_string()
    } else {
        format!("selected tools: {}", selected.join(", "))
    };
    let blocked_text = if blocked.is_empty() {
        "no blocked tools".to_string()
    } else {
        format!("blocked: {}", blocked.join(", "))
    };

    Some(format!(
        "Classified request as {}; route {}; using agent-provided instructions and schemas; {}; {}; prompt: {}.",
        selection.intent.as_str(),
        route,
        selected_text,
        blocked_text,
        prompt_preview
    ))
}

pub fn text_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

pub fn mentions_agent_subject(query: &str) -> bool {
    text_tokens(query)
        .iter()
        .any(|token| matches!(token.as_str(), "agent" | "you" | "u" | "here"))
}
