#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedToolOutput {
    pub kind: String,
    pub summary: String,
    pub important_lines: Vec<String>,
    pub original_chars: usize,
}

pub fn compress_tool_output(command: &str, output: &str, max_lines: usize) -> CompressedToolOutput {
    let kind = classify_command(command, output);
    let important_lines = select_important_lines(&kind, output, max_lines);
    let summary = summarize(&kind, &important_lines, output);

    CompressedToolOutput {
        kind,
        summary,
        important_lines,
        original_chars: output.chars().count(),
    }
}

pub fn render_compressed_tool_output(command: &str, output: &str, max_lines: usize) -> String {
    let compressed = compress_tool_output(command, output, max_lines);
    let mut rendered = format!(
        "tool-output kind={} original_chars={} summary={}",
        compressed.kind, compressed.original_chars, compressed.summary
    );
    if !compressed.important_lines.is_empty() {
        rendered.push_str("\nimportant:\n");
        rendered.push_str(&compressed.important_lines.join("\n"));
    }
    rendered
}

fn classify_command(command: &str, output: &str) -> String {
    let text = format!("{}\n{}", command, output).to_ascii_lowercase();
    if text.contains("cargo") || text.contains("rustc") {
        "cargo".to_string()
    } else if text.contains("npm")
        || text.contains("pnpm")
        || text.contains("yarn")
        || text.contains("vitest")
        || text.contains("jest")
    {
        "node-test".to_string()
    } else if text.contains("pytest") || text.contains("traceback") {
        "pytest".to_string()
    } else if text.contains("diff --git") || text.contains("git diff") {
        "git-diff".to_string()
    } else {
        "generic".to_string()
    }
}

fn select_important_lines(kind: &str, output: &str, max_lines: usize) -> Vec<String> {
    let mut selected = Vec::new();
    let keywords = match kind {
        "cargo" => vec![
            "error[",
            "error:",
            "failed",
            "panicked",
            "test result",
            "could not compile",
            "warning:",
        ],
        "node-test" => vec![
            "failed",
            "error",
            "expected",
            "received",
            "test files",
            "tests",
            "suite",
            "stack",
            "at ",
        ],
        "pytest" => vec![
            "failed",
            "error",
            "traceback",
            "assert",
            "expected",
            "actual",
            "short test summary",
        ],
        "git-diff" => vec!["diff --git", "+++", "---", "@@", "+", "-"],
        _ => vec!["error", "failed", "warning", "panic", "traceback"],
    };

    for line in output.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if keywords.iter().any(|keyword| lower.contains(keyword)) {
            selected.push(trimmed.to_string());
        }
        if selected.len() >= max_lines {
            break;
        }
    }

    if selected.is_empty() {
        output
            .lines()
            .filter(|line| !line.trim().is_empty())
            .take(max_lines)
            .map(|line| line.trim_end().to_string())
            .collect()
    } else {
        selected
    }
}

fn summarize(kind: &str, important_lines: &[String], output: &str) -> String {
    let lower = output.to_ascii_lowercase();
    let status =
        if lower.contains("failed") || lower.contains("error") || lower.contains("panicked") {
            "failed"
        } else if lower.contains("passed") || lower.contains("ok") {
            "passed"
        } else {
            "unknown"
        };
    let first = important_lines
        .first()
        .map(|line| line.as_str())
        .unwrap_or("no salient lines");
    format!("{} {}: {}", kind, status, first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_output_keeps_error_and_test_result() {
        let output = "running 1 test
test foo ... FAILED
error[E0425]: cannot find value `x` in this scope
   --> src/main.rs:1:1
test result: FAILED. 0 passed; 1 failed";

        let compressed = compress_tool_output("cargo test", output, 4);

        assert_eq!(compressed.kind, "cargo");
        assert!(compressed.summary.contains("failed"));
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.contains("error[E0425]")));
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.contains("test result")));
    }

    #[test]
    fn npm_output_keeps_expected_received_failure() {
        let output = "FAIL src/app.test.tsx
Expected: 5
Received: 4
Test Files 1 failed
Tests 1 failed";

        let compressed = compress_tool_output("npm test", output, 5);

        assert_eq!(compressed.kind, "node-test");
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.contains("Expected")));
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.contains("Received")));
    }

    #[test]
    fn pytest_output_keeps_traceback_and_assertion() {
        let output = "Traceback (most recent call last):
  File test_app.py, line 4
assert got == expected
FAILED test_app.py::test_sum";

        let compressed = compress_tool_output("pytest", output, 5);

        assert_eq!(compressed.kind, "pytest");
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.contains("Traceback")));
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.contains("assert")));
    }

    #[test]
    fn git_diff_keeps_hunks() {
        let output = "diff --git a/src/a.rs b/src/a.rs
--- a/src/a.rs
+++ b/src/a.rs
@@ -1 +1 @@
-old
+new";

        let compressed = compress_tool_output("git diff", output, 6);

        assert_eq!(compressed.kind, "git-diff");
        assert!(compressed
            .important_lines
            .iter()
            .any(|line| line.starts_with("@@")));
        assert!(compressed.important_lines.iter().any(|line| line == "+new"));
    }
}
