#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DiagnosticEntry {
    pub category: String,
    pub code: Option<String>,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub col: Option<usize>,
    pub message: String,
    pub context: String,
}

// Diagnostic regexes are compiled once; extract_diagnostics runs on every
// compressed tool output, so per-call Regex::new would dominate its cost.
static CARGO_ERR_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
static CARGO_LOC_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
static TSC_DIAG_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
static PYTEST_FILE_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
static PYTEST_ERR_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

fn cargo_err_re() -> &'static regex::Regex {
    CARGO_ERR_RE.get_or_init(|| regex::Regex::new(r"^(error|warning)\[(E\d+)\]:\s*(.*)$").unwrap())
}

fn cargo_loc_re() -> &'static regex::Regex {
    CARGO_LOC_RE.get_or_init(|| regex::Regex::new(r"^\s*-->\s*([^:]+):(\d+):(\d+)").unwrap())
}

fn tsc_diag_re() -> &'static regex::Regex {
    TSC_DIAG_RE.get_or_init(|| {
        regex::Regex::new(r"^([^(\s]+)\((\d+),(\d+)\):\s*(error|warning)\s+(TS\d+):\s*(.*)$")
            .unwrap()
    })
}

fn pytest_file_re() -> &'static regex::Regex {
    PYTEST_FILE_RE
        .get_or_init(|| regex::Regex::new(r#"^\s*File\s+"([^"]+)",\s*line\s*(\d+)"#).unwrap())
}

fn pytest_err_re() -> &'static regex::Regex {
    PYTEST_ERR_RE.get_or_init(|| regex::Regex::new(r"^([a-zA-Z_]\w*Error):\s*(.*)$").unwrap())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedToolOutput {
    pub kind: String,
    pub summary: String,
    pub important_lines: Vec<String>,
    pub original_chars: usize,
    pub diagnostics: Vec<DiagnosticEntry>,
}

pub fn compress_tool_output(command: &str, output: &str, max_lines: usize) -> CompressedToolOutput {
    let kind = classify_command(command, output);
    let important_lines = select_important_lines(&kind, output, max_lines);
    let summary = summarize(&kind, &important_lines, output);
    let diagnostics = extract_diagnostics(&kind, output);

    CompressedToolOutput {
        kind,
        summary,
        important_lines,
        original_chars: output.chars().count(),
        diagnostics,
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
    if !compressed.diagnostics.is_empty() {
        rendered.push_str("\nstructured-diagnostics:\n");
        for diag in &compressed.diagnostics {
            let diag_json = serde_json::to_string(diag).unwrap_or_default();
            rendered.push_str(&format!("{}\n", diag_json));
        }
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

fn extract_diagnostics(kind: &str, output: &str) -> Vec<DiagnosticEntry> {
    let mut diagnostics = Vec::new();
    let lines: Vec<&str> = output.lines().collect();

    match kind {
        "cargo" => {
            let re_err = cargo_err_re();
            let re_loc = cargo_loc_re();
            for i in 0..lines.len() {
                let line = lines[i].trim();
                if let Some(caps) = re_err.captures(line) {
                    let category = caps
                        .get(1)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| "error".to_string());
                    let code = caps.get(2).map(|m| m.as_str().to_string());
                    let message = caps
                        .get(3)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default();

                    let mut file = None;
                    let mut line_num = None;
                    let mut col_num = None;
                    let mut context_lines = vec![lines[i].to_string()];
                    for j in (i + 1)..std::cmp::min(i + 6, lines.len()) {
                        let next_line = lines[j];
                        context_lines.push(next_line.to_string());
                        if let Some(loc_caps) = re_loc.captures(next_line) {
                            file = loc_caps.get(1).map(|m| m.as_str().to_string());
                            line_num = loc_caps
                                .get(2)
                                .and_then(|m| m.as_str().parse::<usize>().ok());
                            col_num = loc_caps
                                .get(3)
                                .and_then(|m| m.as_str().parse::<usize>().ok());
                            break;
                        }
                    }

                    diagnostics.push(DiagnosticEntry {
                        category,
                        code,
                        file,
                        line: line_num,
                        col: col_num,
                        message,
                        context: context_lines.join("\n"),
                    });
                }
            }
        }
        "node-test" => {
            let re_tsc = tsc_diag_re();
            for line in &lines {
                let trimmed = line.trim();
                if let Some(caps) = re_tsc.captures(trimmed) {
                    let file = caps.get(1).map(|m| m.as_str().to_string());
                    let line_num = caps.get(2).and_then(|m| m.as_str().parse::<usize>().ok());
                    let col_num = caps.get(3).and_then(|m| m.as_str().parse::<usize>().ok());
                    let category = caps
                        .get(4)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| "error".to_string());
                    let code = caps.get(5).map(|m| m.as_str().to_string());
                    let message = caps
                        .get(6)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default();
                    diagnostics.push(DiagnosticEntry {
                        category,
                        code,
                        file,
                        line: line_num,
                        col: col_num,
                        message,
                        context: trimmed.to_string(),
                    });
                }
            }
        }
        "pytest" => {
            let re_file = pytest_file_re();
            let re_err = pytest_err_re();
            for i in 0..lines.len() {
                let line = lines[i];
                if let Some(caps) = re_file.captures(line) {
                    let file = caps.get(1).map(|m| m.as_str().to_string());
                    let line_num = caps.get(2).and_then(|m| m.as_str().parse::<usize>().ok());

                    let mut message = String::new();
                    let mut category = "error".to_string();
                    let mut context_lines = vec![line.to_string()];
                    for j in (i + 1)..std::cmp::min(i + 10, lines.len()) {
                        let next_line = lines[j];
                        context_lines.push(next_line.to_string());
                        if let Some(err_caps) = re_err.captures(next_line) {
                            category = err_caps
                                .get(1)
                                .map(|m| m.as_str().to_string())
                                .unwrap_or_else(|| "error".to_string());
                            message = err_caps
                                .get(2)
                                .map(|m| m.as_str().to_string())
                                .unwrap_or_default();
                            break;
                        }
                    }
                    diagnostics.push(DiagnosticEntry {
                        category,
                        code: None,
                        file,
                        line: line_num,
                        col: None,
                        message,
                        context: context_lines.join("\n"),
                    });
                }
            }
        }
        _ => {}
    }

    diagnostics
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

    #[test]
    fn extracts_diagnostics_successfully() {
        let cargo_err = "error[E0425]: cannot find value `x` in this scope
   --> src/main.rs:10:55";
        let diags = extract_diagnostics("cargo", cargo_err);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0425"));
        assert_eq!(diags[0].file.as_deref(), Some("src/main.rs"));
        assert_eq!(diags[0].line, Some(10));
        assert_eq!(diags[0].col, Some(55));

        let tsc_err = "src/app.ts(5,20): error TS2304: Cannot find name 'y'.";
        let diags_tsc = extract_diagnostics("node-test", tsc_err);
        assert_eq!(diags_tsc.len(), 1);
        assert_eq!(diags_tsc[0].code.as_deref(), Some("TS2304"));
        assert_eq!(diags_tsc[0].file.as_deref(), Some("src/app.ts"));
        assert_eq!(diags_tsc[0].line, Some(5));
        assert_eq!(diags_tsc[0].col, Some(20));
        assert_eq!(diags_tsc[0].message, "Cannot find name 'y'.");

        let pytest_err = "Traceback (most recent call last):
  File \"app.py\", line 15, in run
NameError: name 'val' is not defined";
        let diags_py = extract_diagnostics("pytest", pytest_err);
        assert_eq!(diags_py.len(), 1);
        assert_eq!(diags_py[0].file.as_deref(), Some("app.py"));
        assert_eq!(diags_py[0].line, Some(15));
        assert_eq!(diags_py[0].category, "NameError");
        assert_eq!(diags_py[0].message, "name 'val' is not defined");
    }
}
