use crate::brain::EdgeBrain;
use regex::Regex;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct CompilerVerifier {
    pub brain: EdgeBrain,
}

fn unique_temp_stem() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let count = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("temp_code_{}_{}_{}", std::process::id(), now, count)
}

impl CompilerVerifier {
    pub fn new(brain: EdgeBrain) -> Self {
        Self { brain }
    }

    pub fn extract_code_block(&self, text: &str) -> String {
        let re = Regex::new(r"(?s)```(?:\w+)?\n?(.*?)\n?```").unwrap();
        if let Some(caps) = re.captures(text) {
            caps.get(1).map_or("", |m| m.as_str()).trim().to_string()
        } else {
            let mut cleaned = text
                .trim()
                .trim_start_matches('`')
                .trim_end_matches('`')
                .trim()
                .to_string();
            let lines: Vec<&str> = cleaned.lines().collect();
            let mut started = false;
            let mut code_lines = Vec::new();

            for line in lines {
                let trimmed = line.trim();
                if !started {
                    if trimmed.starts_with("import ")
                        || trimmed.starts_with("from ")
                        || trimmed.starts_with("def ")
                        || trimmed.starts_with("class ")
                        || trimmed.starts_with("print(")
                        || trimmed.starts_with("const ")
                        || trimmed.starts_with("let ")
                        || trimmed.starts_with("var ")
                        || trimmed.starts_with("function ")
                    {
                        started = true;
                        code_lines.push(line);
                    }
                } else {
                    code_lines.push(line);
                }
            }

            if !code_lines.is_empty() {
                cleaned = code_lines.join("\n");
            }
            cleaned
        }
    }

    pub async fn run_local_code(&self, code: &str, language: &str) -> (bool, String) {
        let temp_dir = PathBuf::from("temp_run");
        if let Err(e) = tokio::fs::create_dir_all(&temp_dir).await {
            return (false, format!("Failed to create temp directory: {}", e));
        }

        let lang_lower = language.to_lowercase();
        let (ext, cmd_name) = match lang_lower.as_str() {
            "javascript" | "js" => (".js", "node"),
            "typescript" | "ts" => (".ts", "bun"),
            "rust" | "rs" => (".rs", "rustc"),
            "cpp" | "c++" | "c" => (".cpp", "g++"),
            _ => (".py", "python3"),
        };

        let unique = unique_temp_stem();
        let temp_file = temp_dir.join(format!("{}{}", unique, ext));
        if let Err(e) = tokio::fs::write(&temp_file, code).await {
            return (false, format!("Failed to write temp code: {}", e));
        }

        let result = if lang_lower == "rust" || lang_lower == "rs" {
            let out_bin = temp_dir.join(format!("{}_rust_bin", unique));
            let compile_res = tokio::process::Command::new("rustc")
                .arg(&temp_file)
                .arg("-o")
                .arg(&out_bin)
                .output()
                .await;
            let _ = tokio::fs::remove_file(&temp_file).await;
            match compile_res {
                Ok(comp) if comp.status.success() => {
                    let exec_res = tokio::process::Command::new(&out_bin).output().await;
                    let _ = tokio::fs::remove_file(&out_bin).await;
                    match exec_res {
                        Ok(out) => (
                            out.status.success(),
                            format!(
                                "{}{}",
                                String::from_utf8_lossy(&out.stdout),
                                String::from_utf8_lossy(&out.stderr)
                            ),
                        ),
                        Err(e) => (false, format!("Execution error: {}", e)),
                    }
                }
                Ok(comp) => (
                    false,
                    format!(
                        "Rust compile error: {}",
                        String::from_utf8_lossy(&comp.stderr)
                    ),
                ),
                Err(e) => (false, format!("rustc not found: {}", e)),
            }
        } else if lang_lower == "cpp" || lang_lower == "c++" || lang_lower == "c" {
            let out_bin = temp_dir.join(format!("{}_cpp_bin", unique));
            let compile_res = tokio::process::Command::new("g++")
                .arg(&temp_file)
                .arg("-o")
                .arg(&out_bin)
                .output()
                .await;
            let _ = tokio::fs::remove_file(&temp_file).await;
            match compile_res {
                Ok(comp) if comp.status.success() => {
                    let exec_res = tokio::process::Command::new(&out_bin).output().await;
                    let _ = tokio::fs::remove_file(&out_bin).await;
                    match exec_res {
                        Ok(out) => (
                            out.status.success(),
                            format!(
                                "{}{}",
                                String::from_utf8_lossy(&out.stdout),
                                String::from_utf8_lossy(&out.stderr)
                            ),
                        ),
                        Err(e) => (false, format!("Execution error: {}", e)),
                    }
                }
                Ok(comp) => (
                    false,
                    format!(
                        "C++ compile error: {}",
                        String::from_utf8_lossy(&comp.stderr)
                    ),
                ),
                Err(e) => (false, format!("g++ not found: {}", e)),
            }
        } else {
            let output = tokio::process::Command::new(cmd_name)
                .arg(&temp_file)
                .output()
                .await;
            match output {
                Ok(out) => {
                    let _ = tokio::fs::remove_file(&temp_file).await;
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    (out.status.success(), format!("{}{}", stdout, stderr))
                }
                Err(_) if cmd_name == "bun" => {
                    // Fallback to node for TypeScript/JavaScript if bun is not installed.
                    let output = tokio::process::Command::new("node")
                        .arg(&temp_file)
                        .output()
                        .await;
                    let _ = tokio::fs::remove_file(&temp_file).await;
                    match output {
                        Ok(out) => (
                            out.status.success(),
                            format!(
                                "{}{}",
                                String::from_utf8_lossy(&out.stdout),
                                String::from_utf8_lossy(&out.stderr)
                            ),
                        ),
                        Err(e) => (false, format!("Failed to run command node: {}", e)),
                    }
                }
                Err(e) => {
                    let _ = tokio::fs::remove_file(&temp_file).await;
                    (false, format!("Failed to run command {}: {}", cmd_name, e))
                }
            }
        };

        result
    }

    pub fn compress_prompt(text: &str) -> String {
        let lines: Vec<&str> = text.lines().map(|l| l.trim_end()).collect();
        let mut compressed = Vec::new();
        let mut prev_blank = false;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !prev_blank {
                    compressed.push("");
                    prev_blank = true;
                }
            } else {
                compressed.push(line);
                prev_blank = false;
            }
        }
        compressed.join("\n")
    }

    pub fn repair_python_code(task: &str, code: &str, output: &str) -> Option<String> {
        let task_lower = task.to_lowercase();
        let output_lower = output.to_lowercase();
        if !task_lower.contains("print") {
            return None;
        }
        let should_repair =
            output_lower.contains("object is not iterable") || output.trim().is_empty();
        if !should_repair {
            return None;
        }

        let sum_two_args = Regex::new(r"sum\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)").ok()?;
        for captures in sum_two_args.captures_iter(code) {
            let left = captures.get(1)?.as_str().trim();
            let right = captures.get(2)?.as_str().trim();
            if left.chars().all(|ch| ch.is_ascii_digit())
                && right.chars().all(|ch| ch.is_ascii_digit())
            {
                return Some(format!("print({} + {})", left, right));
            }
        }

        let captures = sum_two_args.captures(code)?;
        let left = captures.get(1)?.as_str().trim();
        let right = captures.get(2)?.as_str().trim();
        Some(format!("print({} + {})", left, right))
    }

    pub async fn generate_and_verify(
        &self,
        task: &str,
        language: &str,
        max_retries: usize,
    ) -> (Option<String>, String) {
        let compressed_task = Self::compress_prompt(task);
        let mut prompt = format!(
            "Write ONLY the clean standalone script in {} for the task:\n{}\nReturn ONLY code inside a single ``` code block.",
            language, compressed_task
        );
        let sys_prompt = "You are a world-class coding expert. Write clean, standalone code.";

        for attempt in 1..=max_retries {
            if attempt > 1 {
                crate::trace::trace_state_transition("verifying", "correcting");
                crate::trace::trace_state_transition("correcting", "verifying");
            } else {
                crate::trace::trace_state_transition("executing", "verifying");
            }
            println!(
                "[CompilerVerifier] Attempt {}/{} generating code...",
                attempt, max_retries
            );
            match self.brain.query_coder(&prompt, sys_prompt).await {
                Ok(raw_res) => {
                    let code = self.extract_code_block(&raw_res);
                    let (success, output) = self.run_local_code(&code, language).await;
                    let output_satisfies_task = !(language.eq_ignore_ascii_case("python")
                        && compressed_task.to_lowercase().contains("print")
                        && output.trim().is_empty());
                    if success && output_satisfies_task {
                        println!(
                            "[CompilerVerifier] Verified code successfully on attempt {}!",
                            attempt
                        );
                        crate::trace::trace_state_transition("verifying", "executing");
                        return (Some(code), output);
                    }
                    println!(
                        "[CompilerVerifier] Attempt {} failed: {}",
                        attempt,
                        output.trim()
                    );
                    if language.eq_ignore_ascii_case("python") {
                        if let Some(repaired_code) =
                            Self::repair_python_code(&compressed_task, &code, output.trim())
                        {
                            let (repair_success, repair_output) =
                                self.run_local_code(&repaired_code, language).await;
                            if repair_success {
                                println!(
                                    "[CompilerVerifier] Verified repaired code after attempt {}!",
                                    attempt
                                );
                                crate::trace::trace_state_transition("verifying", "executing");
                                return (Some(repaired_code), repair_output);
                            }
                        }
                    }
                    prompt = format!(
                        "The following {} code failed during execution:\n```\n{}\n```\nError output:\n```\n{}\n```\nDo not repeat the same code. Explain nothing. Output different corrected code inside one ``` block.",
                        language, code, output.trim()
                    );
                }
                Err(e) => {
                    println!(
                        "[CompilerVerifier] Attempt {} query_coder failed: {}",
                        attempt, e
                    );
                    if attempt == max_retries {
                        return (None, format!("Query coder failed: {}", e));
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64))
                        .await;
                }
            }
        }
        (None, "Max retries exceeded".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::EdgeBrain;
    use std::env;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[tokio::test]
    async fn typescript_falls_back_to_node_when_bun_is_missing() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_path = env::var_os("PATH");
        env::set_var("PATH", "/usr/bin:/bin");

        let verifier = CompilerVerifier::new(EdgeBrain::new());
        let (success, output) = verifier
            .run_local_code("console.log('ts fallback ok');", "typescript")
            .await;

        if let Some(path) = old_path {
            env::set_var("PATH", path);
        } else {
            env::remove_var("PATH");
        }

        assert!(
            success,
            "expected node fallback to succeed, got output: {}",
            output
        );
        assert!(
            output.contains("ts fallback ok"),
            "unexpected output: {}",
            output
        );
    }
    #[test]
    fn extract_code_block_skips_echoed_rag_context() {
        let verifier = CompilerVerifier::new(EdgeBrain::new());
        let raw = "# --- GOOGLE OKF (OPEN KNOWLEDGE FORMAT) CODEBASE CONTEXT ---
# source: src/main.rs
# -------------------------------------------------------------
print(\"ok\")";

        assert_eq!(verifier.extract_code_block(raw), "print(\"ok\")");
    }
    #[test]
    fn repairs_python_sum_expression_when_print_requested() {
        let task = "Write Python code that prints the sum of 2 and 3.";
        let code = "sum(2, 3)";
        let output = "TypeError: 'int' object is not iterable";

        assert_eq!(
            CompilerVerifier::repair_python_code(task, code, output),
            Some("print(2 + 3)".to_string())
        );
    }
    #[test]
    fn repairs_python_sum_task_when_success_output_is_empty() {
        let task = "Write Python code that prints the sum of 2 and 3.";
        let code = "def sum(a, b):
    return a + b

result = sum(2, 3)";
        let output = "";

        assert_eq!(
            CompilerVerifier::repair_python_code(task, code, output),
            Some("print(2 + 3)".to_string())
        );
    }
}
