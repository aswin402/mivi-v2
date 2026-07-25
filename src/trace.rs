use serde_json::{json, Value};
use std::env;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceConfig {
    pub enabled: bool,
    pub path: PathBuf,
}

impl TraceConfig {
    pub fn from_env() -> Self {
        let enabled = env::var("MIVI_TRACE")
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        let path = env::var("MIVI_TRACE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("logs/mivi-trace.jsonl"));

        Self { enabled, path }
    }
}

pub fn trace_event(config: &TraceConfig, mut event: Value) -> Result<(), String> {
    if !config.enabled {
        return Ok(());
    }

    let obj = event
        .as_object_mut()
        .ok_or_else(|| "trace event must be a JSON object".to_string())?;
    obj.entry("ts_unix".to_string())
        .or_insert_with(|| json!(unix_seconds()));

    if let Some(parent) = config.path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.path)
        .map_err(|err| err.to_string())?;
    writeln!(file, "{}", event).map_err(|err| err.to_string())
}

pub fn preview(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn disabled_trace_does_not_create_file() {
        let path = PathBuf::from("target/trace-disabled.jsonl");
        let _ = fs::remove_file(&path);
        let config = TraceConfig {
            enabled: false,
            path: path.clone(),
        };

        trace_event(&config, json!({"kind":"request"})).expect("disabled trace should no-op");

        assert!(!path.exists());
    }

    #[test]
    fn enabled_trace_appends_jsonl_event() {
        let path = PathBuf::from("target/trace-enabled.jsonl");
        let _ = fs::remove_file(&path);
        let config = TraceConfig {
            enabled: true,
            path: path.clone(),
        };

        trace_event(
            &config,
            json!({
                "kind": "tool_generation",
                "route": "model_tool",
                "selected_tools": ["bash"],
                "accepted_tool_calls": ["bash"],
                "rejected_tool_calls": 0
            }),
        )
        .expect("trace write should succeed");

        let content = fs::read_to_string(path).expect("trace file should exist");
        let line = content.lines().next().expect("one trace line");
        let value: Value = serde_json::from_str(line).expect("valid jsonl");

        assert_eq!(
            value.get("kind").and_then(|v| v.as_str()),
            Some("tool_generation")
        );
        assert_eq!(
            value.get("route").and_then(|v| v.as_str()),
            Some("model_tool")
        );
        assert!(value.get("ts_unix").and_then(|v| v.as_u64()).is_some());
    }
}
