use crate::server::ToolDef;
use std::cmp::Ordering;
use std::collections::HashSet;

const MIN_TOOL_SCORE: f32 = 1.0;

pub fn filter_tools(prompt: &str, tools: &[ToolDef], max_tools: usize) -> Vec<ToolDef> {
    if tools.is_empty() || max_tools == 0 || !has_tool_intent(prompt) {
        return Vec::new();
    }

    let mut scored: Vec<(usize, f32, ToolDef)> = tools
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, tool)| {
            let function = &tool.function;
            let mut score = tool_score(
                prompt,
                &function.name,
                function.description.as_deref().unwrap_or(""),
            );

            if let Some(parameters) = &function.parameters {
                score += parameter_score(prompt, parameters);
            }

            (idx, score, tool)
        })
        .filter(|(_, score, _)| *score >= MIN_TOOL_SCORE)
        .collect();

    scored.sort_by(|(left_idx, left_score, _), (right_idx, right_score, _)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_idx.cmp(right_idx))
    });

    scored
        .into_iter()
        .take(max_tools)
        .map(|(_, _, tool)| tool)
        .collect()
}

pub fn tool_score(prompt: &str, tool_name: &str, description: &str) -> f32 {
    let prompt_lc = prompt.to_ascii_lowercase();
    let normalized_name = tool_name.to_ascii_lowercase();
    let prompt_tokens = token_set(prompt);
    let name_tokens = token_set(&normalized_name.replace('_', " "));
    let description_tokens = token_set(description);

    let mut score = 0.0;

    if !normalized_name.is_empty() && prompt_lc.contains(&normalized_name) {
        score += 20.0;
    }

    for token in name_tokens {
        if prompt_tokens.contains(&token) {
            score += 5.0;
        }
    }

    for token in description_tokens {
        if prompt_tokens.contains(&token) {
            score += 2.0;
        }
    }

    score
}

fn parameter_score(prompt: &str, parameters: &serde_json::Value) -> f32 {
    let prompt_tokens = token_set(prompt);
    parameters
        .get("properties")
        .and_then(|properties| properties.as_object())
        .map(|properties| {
            properties
                .keys()
                .map(|key| token_set(key))
                .flat_map(|tokens| tokens.into_iter())
                .filter(|token| prompt_tokens.contains(token))
                .count() as f32
        })
        .unwrap_or(0.0)
}

fn has_tool_intent(prompt: &str) -> bool {
    let text = prompt.to_ascii_lowercase();
    let tokens = token_set(prompt);
    let intent_phrases = [
        "use tool",
        "use the",
        "call tool",
        "call the",
        "call function",
        "run command",
        "execute command",
        "read file",
        "edit file",
        "apply patch",
        "search file",
        "find file",
        "list files",
    ];

    intent_phrases.iter().any(|phrase| text.contains(phrase))
        || (tokens.contains("use") && text.contains('_'))
        || (tokens.contains("search") && tokens.contains("workspace"))
        || (tokens.contains("read") && tokens.contains("file"))
        || (tokens.contains("edit") && tokens.contains("file"))
}

fn token_set(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|part| {
            let token = part.trim().to_ascii_lowercase();
            if token.len() >= 2 && !is_stopword(&token) {
                Some(token)
            } else {
                None
            }
        })
        .collect()
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "for"
            | "from"
            | "with"
            | "what"
            | "are"
            | "you"
            | "your"
            | "please"
            | "this"
            | "that"
            | "into"
            | "about"
            | "current"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::{FunctionDef, ToolDef};
    use serde_json::json;

    fn tool(name: &str, description: &str) -> ToolDef {
        ToolDef {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: Some(description.to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "cmd": { "type": "string" },
                        "query": { "type": "string" }
                    }
                })),
            },
        }
    }

    fn opencode_tools() -> Vec<ToolDef> {
        let mut tools = vec![
            tool("read", "Read a file from the workspace"),
            tool("apply_patch", "Edit files by applying a patch"),
            tool("bash", "Run a shell command"),
            tool("grep", "Search file contents"),
            tool("glob", "Find files by glob pattern"),
        ];
        for idx in 0..128 {
            tools.push(tool(
                &format!("irrelevant_tool_{idx}"),
                "Unrelated plugin action",
            ));
        }
        tools
    }

    #[test]
    fn explicit_tool_name_keeps_exact_match_and_small_fallback_set() {
        let tools = opencode_tools();

        let filtered = filter_tools("please use apply_patch to edit src/main.rs", &tools, 8);

        let names: Vec<&str> = filtered
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert!(names.contains(&"apply_patch"));
        assert!(names.len() <= 8);
        assert!(!names.contains(&"irrelevant_tool_42"));
    }

    #[test]
    fn prompt_without_tool_intent_returns_no_tools() {
        let tools = opencode_tools();

        let filtered = filter_tools("hello, explain what model you are", &tools, 8);

        assert!(filtered.is_empty());
    }

    #[test]
    fn semantic_prompt_selects_read_and_grep_without_exact_names() {
        let tools = opencode_tools();

        let filtered = filter_tools("search the workspace and read the matching file", &tools, 8);

        let names: Vec<&str> = filtered
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert!(names.contains(&"read"));
        assert!(names.contains(&"grep"));
        assert!(names.len() <= 8);
    }

    #[test]
    fn tool_score_prefers_matching_name_and_description() {
        let edit_score = tool_score(
            "edit file with patch",
            "apply_patch",
            "Edit files by applying a patch",
        );
        let weather_score =
            tool_score("edit file with patch", "get_weather", "Get current weather");

        assert!(edit_score > weather_score);
    }
}
