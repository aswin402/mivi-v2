//! Agent stability guard — prevents infinite loops, runaway execution,
//! and duplicate tool calls in orchestrator and tool-calling pipelines.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Prevents infinite loops and runaway execution.
#[derive(Debug, Clone)]
pub struct StabilityGuard {
    tool_call_hashes: HashMap<u64, u32>,
    step_count: u32,
    max_steps: u32,
    max_duplicate_calls: u32,
}

impl Default for StabilityGuard {
    fn default() -> Self {
        Self {
            tool_call_hashes: HashMap::new(),
            step_count: 0,
            max_steps: 10,
            max_duplicate_calls: 2,
        }
    }
}

impl StabilityGuard {
    pub fn new(max_steps: u32, max_duplicate_calls: u32) -> Self {
        Self {
            tool_call_hashes: HashMap::new(),
            step_count: 0,
            max_steps,
            max_duplicate_calls,
        }
    }

    /// Reset at the start of each request.
    pub fn reset(&mut self) {
        self.tool_call_hashes.clear();
        self.step_count = 0;
    }

    /// Increment step counter. Returns Err if limit exceeded.
    pub fn increment_step(&mut self) -> Result<(), String> {
        self.step_count += 1;
        if self.step_count > self.max_steps {
            Err(format!(
                "Stability: step limit exceeded ({}/{}). Aborting.",
                self.step_count, self.max_steps
            ))
        } else {
            Ok(())
        }
    }

    /// Check if a tool call is a duplicate loop. Returns Err if detected.
    pub fn check_tool_call(&mut self, tool_name: &str, arguments: &str) -> Result<(), String> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tool_name.hash(&mut hasher);
        arguments.hash(&mut hasher);
        let hash = hasher.finish();

        let count = self.tool_call_hashes.entry(hash).or_insert(0);
        *count += 1;

        if *count > self.max_duplicate_calls {
            Err(format!(
                "Stability: loop detected — '{}' called {} times with identical arguments.",
                tool_name, count
            ))
        } else {
            Ok(())
        }
    }

    /// Current step count.
    pub fn step_count(&self) -> u32 {
        self.step_count
    }
}

/// Scans ChatCompletionRequest history to detect duplicate tool-call loops.
///
/// Returns Some(explanatory message) when the same tool has been called with
/// identical arguments more than `MAX_DUPLICATE_CALLS` times — callers should
/// surface that message as the assistant reply so the agent can change course,
/// NOT as an error. History length itself is not limited here: the request
/// handler enforces its own cap and the context compressor bounds prompt size.
pub fn check_history_for_loops(messages: &[crate::server::types::ChatMessage]) -> Option<String> {
    const MAX_DUPLICATE_CALLS: u32 = 2;

    let mut tool_call_counts: HashMap<u64, u32> = HashMap::new();
    for msg in messages {
        if msg.role == "assistant" {
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    tc.function.name.hash(&mut hasher);
                    tc.function.arguments.hash(&mut hasher);
                    let hash = hasher.finish();

                    let count = tool_call_counts.entry(hash).or_insert(0);
                    *count += 1;
                    if *count > MAX_DUPLICATE_CALLS {
                        return Some(format!(
                            "Tool loop detected: '{}' with arguments '{}' was already called {} times with the same result. I will not repeat it. Please try a different tool, different arguments, or proceed with the information already gathered.",
                            tc.function.name, tc.function.arguments, *count
                        ));
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::{ChatMessage, FunctionCallIn, ToolCallIn};

    #[test]
    fn test_step_limit() {
        let mut guard = StabilityGuard::new(10, 2);
        for _ in 0..10 {
            assert!(guard.increment_step().is_ok());
        }
        assert!(guard.increment_step().is_err());
    }

    #[test]
    fn test_duplicate_detection() {
        let mut guard = StabilityGuard::new(10, 2);
        assert!(guard
            .check_tool_call("read_file", r#"{"path":"main.rs"}"#)
            .is_ok());
        assert!(guard
            .check_tool_call("read_file", r#"{"path":"main.rs"}"#)
            .is_ok());
        assert!(guard
            .check_tool_call("read_file", r#"{"path":"main.rs"}"#)
            .is_err());
    }

    #[test]
    fn test_different_args_not_duplicate() {
        let mut guard = StabilityGuard::new(10, 2);
        assert!(guard
            .check_tool_call("read_file", r#"{"path":"main.rs"}"#)
            .is_ok());
        assert!(guard
            .check_tool_call("read_file", r#"{"path":"lib.rs"}"#)
            .is_ok());
        assert!(guard
            .check_tool_call("read_file", r#"{"path":"mod.rs"}"#)
            .is_ok());
    }

    #[test]
    fn test_history_loop_detect() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: serde_json::json!("list files"),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::Value::Null,
                tool_calls: Some(vec![ToolCallIn {
                    id: "call_1".to_string(),
                    r#type: "function".to_string(),
                    function: FunctionCallIn {
                        name: "list_dir".to_string(),
                        arguments: r#"{"path":"."}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: serde_json::json!("file1, file2"),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::Value::Null,
                tool_calls: Some(vec![ToolCallIn {
                    id: "call_2".to_string(),
                    r#type: "function".to_string(),
                    function: FunctionCallIn {
                        name: "list_dir".to_string(),
                        arguments: r#"{"path":"."}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            },
        ];

        assert!(check_history_for_loops(&messages).is_none());
    }

    #[test]
    fn test_history_loop_detected_on_third_identical_call() {
        let mut messages = Vec::new();
        for i in 0..3 {
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: serde_json::Value::Null,
                tool_calls: Some(vec![ToolCallIn {
                    id: format!("call_{}", i),
                    r#type: "function".to_string(),
                    function: FunctionCallIn {
                        name: "list_dir".to_string(),
                        arguments: r#"{"path":"."}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            });
        }
        let reason = check_history_for_loops(&messages).expect("should detect loop");
        assert!(reason.contains("list_dir"));
    }

    #[test]
    fn test_long_history_without_loops_is_accepted() {
        // 100-message agent sessions must not trip the guard.
        let mut messages = Vec::new();
        for i in 0..100 {
            messages.push(ChatMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: serde_json::json!(format!("turn {}", i)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        assert!(check_history_for_loops(&messages).is_none());
    }
}
