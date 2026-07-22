use crate::brain::EdgeBrain;
use regex::Regex;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Clone)]
pub struct CompilerVerifier {
    pub brain: EdgeBrain,
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
            let mut cleaned = text.trim().trim_start_matches('`').trim_end_matches('`').trim().to_string();
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
                        || trimmed.starts_with('#')
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

    pub fn run_local_code(&self, code: &str, language: &str) -> (bool, String) {
        let temp_dir = PathBuf::from("temp_run");
        let _ = fs::create_dir_all(&temp_dir);

        let lang_lower = language.to_lowercase();
        let (ext, cmd_name) = match lang_lower.as_str() {
            "javascript" | "js" => (".js", "node"),
            "typescript" | "ts" => (".ts", "bun"),
            "rust" | "rs" => (".rs", "rustc"),
            "cpp" | "c++" | "c" => (".cpp", "g++"),
            _ => (".py", "python3"),
        };

        let temp_file = temp_dir.join(format!("temp_code{}", ext));
        if let Err(e) = fs::write(&temp_file, code) {
            return (false, format!("Failed to write temp code: {}", e));
        }

        let result = if lang_lower == "rust" || lang_lower == "rs" {
            let out_bin = temp_dir.join("temp_code_rust_bin");
            let compile_res = Command::new("rustc").arg(&temp_file).arg("-o").arg(&out_bin).output();
            let _ = fs::remove_file(&temp_file);
            match compile_res {
                Ok(comp) if comp.status.success() => {
                    let exec_res = Command::new(&out_bin).output();
                    let _ = fs::remove_file(&out_bin);
                    match exec_res {
                        Ok(out) => (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))),
                        Err(e) => (false, format!("Execution error: {}", e)),
                    }
                }
                Ok(comp) => (false, format!("Rust compile error: {}", String::from_utf8_lossy(&comp.stderr))),
                Err(e) => (false, format!("rustc not found: {}", e)),
            }
        } else if lang_lower == "cpp" || lang_lower == "c++" || lang_lower == "c" {
            let out_bin = temp_dir.join("temp_code_cpp_bin");
            let compile_res = Command::new("g++").arg(&temp_file).arg("-o").arg(&out_bin).output();
            let _ = fs::remove_file(&temp_file);
            match compile_res {
                Ok(comp) if comp.status.success() => {
                    let exec_res = Command::new(&out_bin).output();
                    let _ = fs::remove_file(&out_bin);
                    match exec_res {
                        Ok(out) => (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))),
                        Err(e) => (false, format!("Execution error: {}", e)),
                    }
                }
                Ok(comp) => (false, format!("C++ compile error: {}", String::from_utf8_lossy(&comp.stderr))),
                Err(e) => (false, format!("g++ not found: {}", e)),
            }
        } else {
            let output = Command::new(cmd_name).arg(&temp_file).output();
            let _ = fs::remove_file(&temp_file);
            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    (out.status.success(), format!("{}{}", stdout, stderr))
                }
                Err(_) if cmd_name == "bun" => {
                    // Fallback to node for TypeScript/JavaScript if bun is not installed
                    let output = Command::new("node").arg(&temp_file).output();
                    let _ = fs::remove_file(&temp_file);
                    match output {
                        Ok(out) => (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))),
                        Err(e) => (false, format!("Failed to run command node: {}", e)),
                    }
                }
                Err(e) => (false, format!("Failed to run command {}: {}", cmd_name, e)),
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

    pub fn generate_and_verify(&self, task: &str, language: &str, max_retries: usize) -> (Option<String>, String) {
        let compressed_task = Self::compress_prompt(task);
        let mut prompt = format!(
            "Write ONLY the clean standalone script in {} for the task:\n{}\nReturn ONLY code inside a single ``` code block.",
            language, compressed_task
        );
        let sys_prompt = "You are a world-class coding expert. Write clean, standalone code.";

        for attempt in 1..=max_retries {
            println!("[CompilerVerifier] Attempt {}/{} generating code...", attempt, max_retries);
            if let Ok(raw_res) = self.brain.query_coder(&prompt, sys_prompt) {
                let code = self.extract_code_block(&raw_res);
                let (success, output) = self.run_local_code(&code, language);
                if success {
                    println!("[CompilerVerifier] Verified code successfully on attempt {}!", attempt);
                    return (Some(code), output);
                }
                println!("[CompilerVerifier] Attempt {} failed: {}", attempt, output.trim());
                prompt = format!(
                    "The following {} code failed during execution:\n```\n{}\n```\nError output:\n```\n{}\n```\nOutput corrected code inside a ``` block.",
                    language, code, output.trim()
                );
            }
        }
        (None, "Max retries exceeded".to_string())
    }
}
