use crate::server::ToolDef;
use std::cmp::Ordering;
use std::collections::HashSet;

use crate::constants::MIN_TOOL_SCORE;

fn task_tags(prompt: &str) -> HashSet<&'static str> {
    let text = prompt.to_ascii_lowercase();
    let tokens = token_set(prompt);
    let mut tags = HashSet::new();

    if tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "bash"
                | "shell"
                | "terminal"
                | "command"
                | "npm"
                | "pnpm"
                | "yarn"
                | "node"
                | "bun"
                | "cargo"
                | "pytest"
                | "pip"
                | "uv"
                | "rustc"
                | "clippy"
        )
    }) || text.contains("run test")
        || text.contains("run build")
    {
        tags.insert("shell");
    }
    if tokens.contains("read") || tokens.contains("inspect") || tokens.contains("open") {
        tags.insert("read");
    }
    if tokens.contains("edit")
        || tokens.contains("fix")
        || tokens.contains("patch")
        || tokens.contains("modify")
    {
        tags.insert("edit");
    }
    if tokens.contains("search") || tokens.contains("grep") || tokens.contains("find") {
        tags.insert("search");
    }
    if tokens.contains("git")
        || tokens.contains("diff")
        || tokens.contains("commit")
        || tokens.contains("status")
    {
        tags.insert("git");
    }
    if tokens.contains("web")
        || tokens.contains("internet")
        || tokens.contains("browser")
        || tokens.contains("url")
    {
        tags.insert("web");
    }
    if tokens.contains("memory") || tokens.contains("remember") || tokens.contains("database") {
        tags.insert("memory");
    }

    tags
}

fn tool_tags(tool_name: &str, description: &str) -> HashSet<&'static str> {
    let text = format!("{} {}", tool_name, description).to_ascii_lowercase();
    let mut tags = HashSet::new();

    if text.contains("bash")
        || text.contains("shell")
        || text.contains("command")
        || text.contains("terminal")
    {
        tags.insert("shell");
    }
    if text.contains("read") || text.contains("file") || text.contains("open") {
        tags.insert("read");
    }
    if text.contains("edit")
        || text.contains("patch")
        || text.contains("write file")
        || text.contains("modify")
    {
        tags.insert("edit");
    }
    if text.contains("grep")
        || text.contains("search")
        || text.contains("find")
        || text.contains("glob")
    {
        tags.insert("search");
    }
    if text.contains("git")
        || text.contains("diff")
        || text.contains("commit")
        || text.contains("status")
    {
        tags.insert("git");
    }
    if text.contains("web")
        || text.contains("browser")
        || text.contains("url")
        || text.contains("internet")
    {
        tags.insert("web");
    }
    if text.contains("memory") || text.contains("database") || text.contains("remember") {
        tags.insert("memory");
    }

    tags
}

fn tag_score(prompt_tags: &HashSet<&'static str>, tool_name: &str, description: &str) -> f32 {
    if prompt_tags.is_empty() {
        return 0.0;
    }
    let tags = tool_tags(tool_name, description);
    prompt_tags.intersection(&tags).count() as f32 * 6.0
}

pub fn filter_tools(prompt: &str, tools: &[ToolDef], max_tools: usize) -> Vec<ToolDef> {
    if tools.is_empty() || max_tools == 0 {
        return Vec::new();
    }

    if tools.len() <= max_tools {
        return tools.to_vec();
    }

    let prompt_tags = task_tags(prompt);
    let mut scored: Vec<(usize, f32, ToolDef)> = tools
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, tool)| {
            let function = &tool.function;
            let description = function.description.as_deref().unwrap_or("");
            let mut score = tool_score(prompt, &function.name, description);
            score += tag_score(&prompt_tags, &function.name, description);

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

#[allow(dead_code)]
fn has_tool_intent(prompt: &str, tools: &[ToolDef]) -> bool {
    let text = prompt.to_ascii_lowercase();

    // 1. Explicit tool name matching: if the user mentions any of the tool names in their prompt
    for tool in tools {
        let name = tool.function.name.to_ascii_lowercase();
        if !name.is_empty() && text.contains(&name) {
            return true;
        }
    }

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
        "run test",
        "run build",
        "npm test",
        "pnpm test",
        "yarn test",
        "cargo test",
        "pytest",
    ];

    intent_phrases.iter().any(|phrase| text.contains(phrase))
        || (tokens.contains("use") && text.contains('_'))
        || (tokens.contains("search") && tokens.contains("workspace"))
        || (tokens.contains("read") && tokens.contains("file"))
        || (tokens.contains("edit") && tokens.contains("file"))
        || task_tags(prompt).contains("shell")
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
    #[test]
    fn developer_task_selects_shell_and_edit_tools_from_large_agent_toolset() {
        let tools = opencode_tools();

        let filtered = filter_tools(
            "run npm test, inspect the terminal error, then edit the failing TypeScript file",
            &tools,
            5,
        );

        let names: Vec<&str> = filtered
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"apply_patch"));
        assert!(names.contains(&"read"));
        assert!(names.len() <= 5);
        assert!(!names
            .iter()
            .any(|name| name.starts_with("irrelevant_tool_")));
    }

    #[test]
    fn terminal_task_selects_shell_tool_without_command_keyword() {
        let tools = opencode_tools();

        let filtered = filter_tools("run npm test", &tools, 5);

        let names: Vec<&str> = filtered
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert!(names.contains(&"bash"));
        assert!(!names
            .iter()
            .any(|name| name.starts_with("irrelevant_tool_")));
    }
}
