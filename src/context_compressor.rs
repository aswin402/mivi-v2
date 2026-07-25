use crate::runtime::ContextBudget;
use crate::server::ChatMessage;
use crate::tool_output::render_compressed_tool_output;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedContext {
    pub system: String,
    pub protected_recent: Vec<String>,
    pub tool_observations: Vec<String>,
    pub summary: String,
}

pub fn compress_context(messages: &[ChatMessage], budget: ContextBudget) -> CompressedContext {
    let mut system_parts = Vec::new();
    let mut protected_recent = Vec::new();
    let mut tool_observations = Vec::new();
    let mut dropped = 0usize;

    for msg in messages {
        let text = message_text(msg);
        let normalized = normalize_context_text(&text);
        if normalized.is_empty() {
            dropped += 1;
            continue;
        }

        match msg.role.as_str() {
            "system" => system_parts.push(normalized),
            "tool" => tool_observations.push(format!(
                "tool: {}",
                truncate_chars(
                    &render_compressed_tool_output(&normalized, &normalized, 8),
                    budget.tool_tokens * 4
                )
            )),
            role => {
                if is_low_value_turn(role, &normalized) {
                    dropped += 1;
                    continue;
                }

                if is_important_context(&normalized) {
                    protected_recent.push(format!("{}: {}", role, normalized));
                }
            }
        }
    }

    let mut recent_turns = recent_non_noise_turns(messages, 4);
    protected_recent.append(&mut recent_turns);
    dedupe_keep_order(&mut protected_recent);

    trim_vec_to_char_budget(&mut protected_recent, budget.recent_turn_tokens * 4);
    trim_vec_to_char_budget(&mut tool_observations, budget.tool_tokens * 4);

    let summary = if dropped == 0 {
        String::new()
    } else {
        format!("Dropped {dropped} low-value or injected context messages.")
    };

    CompressedContext {
        system: truncate_chars(&system_parts.join("\n"), budget.max_input_tokens),
        protected_recent,
        tool_observations,
        summary,
    }
}

pub fn render_context_prompt(compressed: &CompressedContext, latest_user_prompt: &str) -> String {
    let mut parts = Vec::new();

    if !compressed.system.is_empty() {
        parts.push(format!("System instructions:\n{}", compressed.system));
    }

    if !compressed.tool_observations.is_empty() {
        parts.push(format!(
            "Tool observations:\n{}",
            compressed.tool_observations.join("\n")
        ));
    }

    if !compressed.protected_recent.is_empty() {
        parts.push(format!(
            "Relevant recent context:\n{}",
            compressed.protected_recent.join("\n")
        ));
    }

    if !compressed.summary.is_empty() {
        parts.push(compressed.summary.clone());
    }

    if !latest_user_prompt.trim().is_empty() {
        parts.push(format!(
            "Current user request:\n{}",
            latest_user_prompt.trim()
        ));
    }

    parts.join("\n\n")
}

fn recent_non_noise_turns(messages: &[ChatMessage], max_turns: usize) -> Vec<String> {
    let mut turns = Vec::new();

    for msg in messages.iter().rev() {
        if msg.role == "system" || msg.role == "tool" {
            continue;
        }

        let text = normalize_context_text(&message_text(msg));
        if text.is_empty() || is_low_value_turn(&msg.role, &text) {
            continue;
        }

        turns.push(format!("{}: {}", msg.role, text));
        if turns.len() >= max_turns {
            break;
        }
    }

    turns.reverse();
    turns
}

fn message_text(msg: &ChatMessage) -> String {
    if let Some(text) = msg.content.as_str() {
        text.to_string()
    } else if let Some(parts) = msg.content.as_array() {
        parts
            .iter()
            .filter_map(|part| {
                let part_type = part.get("type").and_then(|value| value.as_str())?;
                if part_type != "text" {
                    return None;
                }
                part.get("text")
                    .and_then(|value| value.as_str())
                    .map(|text| text.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        String::new()
    }
}

fn normalize_context_text(text: &str) -> String {
    let mut normalized = text.trim().to_string();
    for tag in [
        "available-skills",
        "skill-evaluation-required",
        "user-prompt-submit-hook",
    ] {
        normalized = strip_tagged_block(&normalized, tag);
    }
    normalized.trim().to_string()
}

fn strip_tagged_block(text: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let trimmed = text.trim();

    if trimmed.starts_with(&open) {
        if let Some(end) = trimmed.find(&close) {
            return trimmed[end + close.len()..].trim().to_string();
        }
    }

    trimmed.to_string()
}

fn is_low_value_turn(role: &str, text: &str) -> bool {
    let low = text.trim().to_ascii_lowercase();
    role != "system"
        && matches!(
            low.as_str(),
            "hi" | "hii" | "hey" | "hello" | "ok" | "okay" | "thanks" | "thank you"
        )
}

fn is_important_context(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    text.contains("```")
        || low.contains("error[")
        || low.contains("error:")
        || low.contains("failed")
        || low.contains("exception")
        || low.contains("tool")
        || low.contains("instruction")
}

fn dedupe_keep_order(items: &mut Vec<String>) {
    let mut deduped = Vec::new();
    for item in items.drain(..) {
        if !deduped.contains(&item) {
            deduped.push(item);
        }
    }
    *items = deduped;
}

fn trim_vec_to_char_budget(items: &mut Vec<String>, max_chars: usize) {
    let mut total = 0usize;
    let mut kept = Vec::new();

    for item in items.iter().rev() {
        let item_len = item.chars().count();
        if total + item_len > max_chars && !kept.is_empty() {
            break;
        }
        total += item_len;
        kept.push(truncate_chars(item, max_chars));
    }

    kept.reverse();
    *items = kept;
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    text.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ContextBudget;
    use crate::server::ChatMessage;
    use serde_json::json;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: json!(content),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    #[test]
    fn preserves_system_latest_turns_and_tool_observations() {
        let messages = vec![
            msg("system", "Always answer in English and use concise code."),
            msg("user", "old question"),
            msg("assistant", "old answer"),
            msg("tool", "read failed: No such file or directory"),
            msg("user", "Fix the runtime bug now"),
            msg("assistant", "I will inspect the runtime module"),
        ];

        let compressed = compress_context(&messages, ContextBudget::from_max_input_tokens(1024));

        assert!(compressed.system.contains("Always answer in English"));
        assert!(compressed
            .protected_recent
            .iter()
            .any(|turn| turn.contains("Fix the runtime bug now")));
        assert!(compressed
            .protected_recent
            .iter()
            .any(|turn| turn.contains("I will inspect")));
        assert!(compressed
            .tool_observations
            .iter()
            .any(|turn| turn.contains("No such file")));
    }

    #[test]
    fn drops_injected_skill_blocks_and_old_greetings() {
        let messages = vec![
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg(
                "user",
                "<available-skills>use_skill read_skill_file bash apply_patch</available-skills>",
            ),
            msg(
                "user",
                "<skill-evaluation-required>SKILL EVALUATION PROCESS</skill-evaluation-required>",
            ),
            msg("user", "Build the runtime config"),
        ];

        let compressed = compress_context(&messages, ContextBudget::from_max_input_tokens(1024));
        let joined = format!(
            "{} {} {}",
            compressed.system,
            compressed.protected_recent.join("\n"),
            compressed.summary
        );

        assert!(joined.contains("Build the runtime config"));
        assert!(!joined.contains("available-skills"));
        assert!(!joined.contains("SKILL EVALUATION PROCESS"));
        assert!(!joined.contains("user: hi"));
    }

    #[test]
    fn preserves_code_blocks_and_error_messages_from_older_context() {
        let messages = vec![
            msg("assistant", "```rust\nfn broken() {}\n```"),
            msg("user", "error[E0425]: cannot find function filter_tools"),
            msg("user", "Continue the fix"),
        ];

        let compressed = compress_context(&messages, ContextBudget::from_max_input_tokens(1024));

        assert!(compressed
            .protected_recent
            .iter()
            .any(|turn| turn.contains("```rust")));
        assert!(compressed
            .protected_recent
            .iter()
            .any(|turn| turn.contains("error[E0425]")));
        assert!(compressed
            .protected_recent
            .iter()
            .any(|turn| turn.contains("Continue the fix")));
    }

    #[test]
    fn compresses_long_tool_output_to_salient_lines() {
        let long_noise = (0..40)
            .map(|index| format!("compiling filler crate {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tool_output = format!(
            "cargo test\n{long_noise}\nerror[E0425]: cannot find value `x` in this scope\nthread 'tests::works' panicked\nfinal filler line that should not matter"
        );
        let messages = vec![msg("tool", &tool_output), msg("user", "fix compile error")];

        let compressed = compress_context(&messages, ContextBudget::from_max_input_tokens(1024));
        let joined = compressed.tool_observations.join("\n");

        assert!(joined.contains("tool-output kind=cargo"));
        assert!(joined.contains("error[E0425]"));
        assert!(joined.contains("panicked"));
        assert!(!joined.contains("final filler line"));
    }
}
