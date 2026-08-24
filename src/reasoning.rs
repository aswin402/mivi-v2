//! Qwen3 reasoning-mode directives (`/think`, `/no_think`) and response
//! cleaning: think-block stripping and llama-cli output scrubbing.
//!
//! Extracted from `brain.rs` (decomposition).

use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningMode {
    Auto,
    Think,
    NoThink,
}

impl ReasoningMode {
    fn from_env() -> Self {
        match env::var("MIVI_REASONING_MODE")
            .unwrap_or_else(|_| "auto".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "think" => Self::Think,
            "no_think" | "nothink" | "no-think" => Self::NoThink,
            _ => Self::Auto,
        }
    }
}

fn should_think_for_prompt(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    [
        "think deeply",
        "deep reasoning",
        "step-by-step reasoning",
        "step by step reasoning",
        "complex plan",
        "architecture decision",
        "security review",
        "hard reasoning",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn reasoning_directive(prompt: &str) -> &'static str {
    match ReasoningMode::from_env() {
        ReasoningMode::Think => "/think",
        ReasoningMode::NoThink => "/no_think",
        ReasoningMode::Auto if should_think_for_prompt(prompt) => "/think",
        ReasoningMode::Auto => "/no_think",
    }
}

pub fn is_prompt_preformatted(prompt: &str) -> bool {
    let trimmed = prompt.trim_start();
    if trimmed.contains("<|im_start|>") && trimmed.contains("<|im_start|>assistant") {
        return true;
    }
    let t = crate::server::active_chat_template();
    if !t.system_prefix.is_empty() && !t.assistant_start.is_empty() {
        let sys = t.system_prefix.trim();
        let asst = t.assistant_start.trim();
        if trimmed.contains(sys) && trimmed.contains(asst) {
            return true;
        }
    }
    false
}

pub(crate) fn apply_reasoning_directive(prompt: &str) -> String {
    let trimmed = prompt.trim_start();
    if trimmed.starts_with("/think") || trimmed.starts_with("/no_think") {
        prompt.to_string()
    } else if is_prompt_preformatted(prompt) {
        prompt.to_string()
    } else {
        let is_qwen = if let Ok(catalog) = crate::model_catalog::ModelCatalog::load_default() {
            catalog
                .models
                .iter()
                .find(|m| m.role == crate::model_catalog::ModelRole::Reasoner)
                .map(|m| m.path.to_lowercase().contains("qwen"))
                .unwrap_or(false)
        } else {
            false
        };

        if is_qwen {
            format!("{}\n{}", reasoning_directive(prompt), prompt)
        } else {
            prompt.to_string()
        }
    }
}

fn strip_delimited_block(text: &str, open: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    loop {
        let lower = rest.to_ascii_lowercase();
        if let Some(start) = lower.find(open) {
            out.push_str(&rest[..start]);
            let after_open = &rest[start + open.len()..];
            let after_lower = after_open.to_ascii_lowercase();
            if let Some(end) = after_lower.find(close) {
                rest = &after_open[end + close.len()..];
            } else {
                break;
            }
        } else {
            out.push_str(rest);
            break;
        }
    }
    out.trim().to_string()
}

pub(crate) fn strip_think_blocks(text: &str) -> String {
    let without_xml = strip_delimited_block(text, "<think>", "</think>");
    let without_bracketed =
        strip_delimited_block(&without_xml, "[start thinking]", "[end thinking]");
    strip_delimited_block(&without_bracketed, "start thinking", "end thinking")
}

#[allow(dead_code)]
pub(crate) fn clean_llama_cli_response(stdout: &str) -> String {
    let t = crate::server::active_chat_template();
    let assistant_marker = t.assistant_start.trim();
    let response = if !assistant_marker.is_empty() {
        if let Some(pos) = stdout.rfind(assistant_marker) {
            &stdout[pos + assistant_marker.len()..]
        } else if let Some(pos) = stdout.find("> ") {
            &stdout[pos + 2..]
        } else {
            stdout
        }
    } else if let Some(pos) = stdout.find("> ") {
        &stdout[pos + 2..]
    } else {
        stdout
    };

    let mut clean = response
        .split("[ Prompt:")
        .next()
        .unwrap_or(response)
        .split("Exiting...")
        .next()
        .unwrap_or(response)
        .to_string();

    for stop in &t.stop_words {
        clean = clean.replace(stop, "");
    }
    clean = clean.trim().to_string();

    strip_think_blocks(&scrub_generated_prompt_echo(&clean))
}

#[allow(dead_code)]
pub(crate) fn scrub_generated_prompt_echo(text: &str) -> String {
    let t = crate::server::active_chat_template();
    let mut cleaned = text.trim();

    if let Some((_, tail)) = cleaned.rsplit_once("... (truncated)") {
        cleaned = tail.trim();
    }

    if let Some(rest) = cleaned.strip_prefix("user\n") {
        if let Some((_, answer)) = rest.split_once("\n\n") {
            cleaned = answer.trim();
        }
    }

    let sys_prefix = t.system_prefix.trim();
    let user_prefix = t.user_prefix.trim();
    if !sys_prefix.is_empty() && cleaned.contains(sys_prefix) {
        if !user_prefix.is_empty() {
            if let Some((_, tail)) = cleaned.rsplit_once(user_prefix) {
                if let Some((_, answer)) = tail.split_once("\n\n") {
                    cleaned = answer.trim();
                }
            }
        }
    }

    let mut res = cleaned.to_string();
    for stop in &t.stop_words {
        res = res.replace(stop, "");
    }
    res.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn clean_response_uses_last_assistant_marker() {
        let t = crate::server::active_chat_template();
        let assistant_start = t.assistant_start.trim();
        let stop_word = t
            .stop_words
            .first()
            .map(|s| s.as_str())
            .unwrap_or("<|im_end|>");
        let stdout = format!(
            "system\nctx\n{}echoed old answer{}\n{}final answer\n[ Prompt: 12 tokens]",
            assistant_start, stop_word, assistant_start
        );

        assert_eq!(clean_llama_cli_response(&stdout), "final answer");
    }

    #[test]
    fn clean_response_strips_end_token() {
        let t = crate::server::active_chat_template();
        let assistant_start = t.assistant_start.trim();
        let stop_word = t
            .stop_words
            .first()
            .map(|s| s.as_str())
            .unwrap_or("<|im_end|>");
        let input = format!("{}Hello{}", assistant_start, stop_word);
        assert_eq!(clean_llama_cli_response(&input), "Hello");
    }

    #[test]
    fn reasoning_directive_defaults_to_no_think_for_simple_prompts() {
        let _guard = env_lock();
        env::remove_var("MIVI_REASONING_MODE");
        assert_eq!(reasoning_directive("Say hello."), "/no_think");
    }

    #[test]
    fn reasoning_directive_auto_is_conservative_for_agent_prompts() {
        let _guard = env_lock();
        env::remove_var("MIVI_REASONING_MODE");
        assert_eq!(
            reasoning_directive("Debug this Rust compiler error."),
            "/no_think"
        );
        assert_eq!(
            reasoning_directive("Use deep reasoning to debug this Rust compiler error."),
            "/think"
        );
    }

    #[test]
    fn reasoning_directive_env_override_wins() {
        let _guard = env_lock();
        env::set_var("MIVI_REASONING_MODE", "no_think");
        assert_eq!(reasoning_directive("Debug this failure."), "/no_think");
        env::set_var("MIVI_REASONING_MODE", "think");
        assert_eq!(reasoning_directive("Say hello."), "/think");
        env::remove_var("MIVI_REASONING_MODE");
    }

    #[test]
    fn strip_think_blocks_removes_private_reasoning() {
        assert_eq!(
            strip_think_blocks("<think>private notes</think>Final answer"),
            "Final answer"
        );
        assert_eq!(strip_think_blocks("A <think>hidden</think> B"), "A  B");
    }

    #[test]
    fn apply_reasoning_directive_preserves_explicit_user_directive() {
        assert_eq!(
            apply_reasoning_directive(
                "/think
Debug it."
            ),
            "/think
Debug it."
        );
    }

    #[test]
    fn scrub_generated_prompt_echo_removes_truncated_context_preamble() {
        let leaked = "add a text file\n  /glob <pattern>\n\n> <|im_start|>system\nctx\n<|im_start|>user\nCurrent user request:\nFix Cargo.\n ... (truncated)\nuser\nFix Cargo.\n\nTo fix it, remove the broken cache directory.";

        assert_eq!(
            scrub_generated_prompt_echo(leaked),
            "To fix it, remove the broken cache directory."
        );
    }
}
