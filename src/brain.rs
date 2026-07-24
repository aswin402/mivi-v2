use crate::prompt_file::write_prompt_file;
use crate::runtime::RuntimeConfig;
use crate::worker::{WorkerConfig, WorkerManager};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[derive(Clone)]
pub struct EdgeBrain {
    pub llama_cli: PathBuf,
    pub minicpm_cli: PathBuf,
    pub llama_path: PathBuf,
    pub qwen_path: PathBuf,
    pub minicpm_path: PathBuf,
    pub minicpm_proj: PathBuf,
    pub ultra_low_ram: bool,
    pub text_worker: Arc<WorkerManager>,
}

fn clean_llama_cli_response(stdout: &str) -> String {
    let response = if let Some(pos) = stdout.rfind("<|im_start|>assistant") {
        &stdout[pos + "<|im_start|>assistant".len()..]
    } else if let Some(pos) = stdout.find("> ") {
        &stdout[pos + 2..]
    } else {
        stdout
    };

    let clean = response
        .split("[ Prompt:")
        .next()
        .unwrap_or(response)
        .split("Exiting...")
        .next()
        .unwrap_or(response)
        .replace("<|im_end|>", "")
        .trim()
        .to_string();

    scrub_generated_prompt_echo(&clean)
}

fn scrub_generated_prompt_echo(text: &str) -> String {
    let mut cleaned = text.trim();

    if let Some((_, tail)) = cleaned.rsplit_once("... (truncated)") {
        cleaned = tail.trim();
    }

    if let Some(rest) = cleaned.strip_prefix("user\n") {
        if let Some((_, answer)) = rest.split_once("\n\n") {
            cleaned = answer.trim();
        }
    }

    if cleaned.contains("<|im_start|>system") {
        if let Some((_, tail)) = cleaned.rsplit_once("<|im_start|>user") {
            if let Some((_, answer)) = tail.split_once("\n\n") {
                cleaned = answer.trim();
            }
        }
    }

    cleaned
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .trim()
        .to_string()
}

impl EdgeBrain {
    pub fn new() -> Self {
        let base_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let exe_ext = if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        };

        let possible_bins = vec![
            base_dir.join("bin").join(format!("llama-cli{}", exe_ext)),
            PathBuf::from(format!("llama-cli{}", exe_ext)),
        ];

        let llama_cli = possible_bins
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from(format!("llama-cli{}", exe_ext)));

        let possible_minicpm_bins = vec![
            base_dir
                .join("bin")
                .join(format!("llama-mtmd-cli{}", exe_ext)),
            base_dir
                .join("bin")
                .join(format!("llama-minicpmv-cli{}", exe_ext)),
            base_dir.join("bin").join(format!("llama-cli{}", exe_ext)),
            PathBuf::from(format!("llama-mtmd-cli{}", exe_ext)),
        ];

        let minicpm_cli = possible_minicpm_bins
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| llama_cli.clone());

        let models_dir = base_dir.join("models");
        let llama_path = models_dir.join("Llama-3.2-1B-Instruct-IQ3_M.gguf");
        let qwen_path = models_dir.join("qwen2.5-0.5b-instruct-q2_k.gguf");
        let minicpm_path = models_dir.join("MiniCPM-V-4.6-Q4_K_M.gguf");
        let minicpm_proj = models_dir.join("mmproj-MiniCPM-V-4.6-Q8_0.gguf");
        let ultra_low_ram = env::var("MIVI_ULTRA_LOW_RAM")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let runtime_config = RuntimeConfig::from_env();
        let server_path = base_dir
            .join("bin")
            .join(format!("llama-server{}", exe_ext));
        let text_worker = Arc::new(WorkerManager::new(WorkerConfig::default_for_text_model(
            server_path,
            llama_path.clone(),
            &runtime_config,
        )));

        if ultra_low_ram {
            println!(
                "[AIRLLM/COLIBRI MODE] Ultra-Low-RAM mmap streaming active (< 40 MB RAM target)"
            );
        }

        Self {
            llama_cli,
            minicpm_cli,
            llama_path,
            qwen_path,
            minicpm_path,
            minicpm_proj,
            ultra_low_ram,
            text_worker,
        }
    }

    fn run_cli(
        &self,
        model_path: &Path,
        prompt: &str,
        system_prompt: &str,
        temp: &str,
        context_size: &str,
    ) -> Result<String, String> {
        let runtime_config = RuntimeConfig::from_env();
        if runtime_config.uses_worker() && model_path == self.llama_path.as_path() {
            match self.text_worker.query_chat(prompt, system_prompt, temp) {
                Ok(response) => return Ok(response),
                Err(err) => eprintln!(
                    "[MIVI-V2 Worker] Falling back to llama-cli after worker error: {}",
                    err
                ),
            }
        }

        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, prompt
        );

        let eff_context = if self.ultra_low_ram && context_size == "8192" {
            "4096"
        } else {
            context_size
        };

        let ngl_val = if self.ultra_low_ram { "0" } else { "999" };

        let prompt_file = write_prompt_file(&formatted_prompt)?;
        let mut cmd = Command::new(&self.llama_cli);
        cmd.arg("-m")
            .arg(model_path)
            .arg("-ngl")
            .arg(ngl_val)
            .arg("-c")
            .arg(eff_context)
            .arg("-fa")
            .arg("on")
            .arg("-ctk")
            .arg("q8_0")
            .arg("-ctv")
            .arg("q8_0")
            .arg("-f")
            .arg(&prompt_file)
            .arg("--temp")
            .arg(temp)
            .arg("--simple-io")
            .arg("--no-display-prompt")
            .arg("-st");

        if self.ultra_low_ram {
            cmd.arg("--mmap");
        }

        let output = match cmd.output() {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&prompt_file);
                return Err(format!("Failed to execute llama-cli: {}", e));
            }
        };
        let _ = std::fs::remove_file(&prompt_file);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(clean_llama_cli_response(&stdout))
    }

    pub fn query_reasoner(&self, prompt: &str, system_prompt: &str) -> Result<String, String> {
        self.run_cli(&self.llama_path, prompt, system_prompt, "0.2", "8192")
    }

    pub fn query_coder(&self, prompt: &str, system_prompt: &str) -> Result<String, String> {
        self.run_cli(&self.qwen_path, prompt, system_prompt, "0.1", "8192")
    }

    /// Speculative Decoding (ds4 DwarfStar pattern):
    /// Uses Qwen-0.5B to draft tokens fast, then uses Llama-1B to verify.
    pub fn query_speculative(&self, prompt: &str, system_prompt: &str) -> Result<String, String> {
        println!("[DS4 SPECULATIVE] Drafting with Qwen 0.5B...");
        let draft = self.query_coder(prompt, system_prompt)?;

        if draft.trim().is_empty() {
            return self.query_reasoner(prompt, system_prompt);
        }

        println!("[DS4 SPECULATIVE] Verifying draft with Llama 1B...");
        let verify_prompt = format!(
            "Verify and improve this response for accuracy:\nUSER: {}\nPROPOSED RESPONSE:\n{}\nIf accurate, output the response as is. Otherwise output the corrected version.",
            prompt, draft
        );

        match self.query_reasoner(&verify_prompt, system_prompt) {
            Ok(verified) if !verified.trim().is_empty() => Ok(verified),
            _ => Ok(draft),
        }
    }

    pub fn query_raw(&self, prompt: &str) -> Result<String, String> {
        let eff_context = if self.ultra_low_ram { "4096" } else { "8192" };
        let ngl_val = if self.ultra_low_ram { "0" } else { "999" };

        let prompt_file = write_prompt_file(prompt)?;
        let mut cmd = Command::new(&self.llama_cli);
        cmd.arg("-m")
            .arg(&self.llama_path)
            .arg("-ngl")
            .arg(ngl_val)
            .arg("-c")
            .arg(eff_context)
            .arg("-fa")
            .arg("on")
            .arg("-ctk")
            .arg("q8_0")
            .arg("-ctv")
            .arg("q8_0")
            .arg("-f")
            .arg(&prompt_file)
            .arg("--temp")
            .arg("0.2")
            .arg("--simple-io")
            .arg("--no-display-prompt")
            .arg("-st");

        if self.ultra_low_ram {
            cmd.arg("--mmap");
        }

        let output = match cmd.output() {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&prompt_file);
                return Err(format!("Failed to execute llama-cli: {}", e));
            }
        };
        let _ = std::fs::remove_file(&prompt_file);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();

        // Try extracting from after last <|im_start|>assistant tag (prompt echo).
        // If not echoed, find the first JSON object or take the last non-empty line.
        let response = if let Some(pos) = stdout.rfind("<|im_start|>assistant") {
            let after = &stdout[pos + "<|im_start|>assistant".len()..];
            if let Some(echo_end) = after.find("<|im_start|>") {
                &after[..echo_end]
            } else {
                after
            }
        } else {
            // Fallback: skip loading banner and take the last non-empty block.
            let lines: Vec<&str> = stdout.lines().collect();
            if let Some(&last) = lines.iter().rev().find(|l| !l.trim().is_empty()) {
                last.trim()
            } else {
                &stdout[..]
            }
        };

        let clean = response
            .split("[ Prompt:")
            .next()
            .unwrap_or(response)
            .split("Exiting...")
            .next()
            .unwrap_or(response)
            .trim();

        Ok(clean.to_string())
    }

    pub fn query_vision(&self, image_path: &str, prompt: &str) -> Result<String, String> {
        if !Path::new(image_path).exists() {
            return Err(format!("Image file not found at: {}", image_path));
        }

        if !self.minicpm_path.exists() {
            return Err(format!(
                "Vision model weights not found at '{}'. Download MiniCPM-V-4.6-Q4_K_M.gguf and mmproj-MiniCPM-V-4.6-Q8_0.gguf into models/",
                self.minicpm_path.display()
            ));
        }

        let output = Command::new(&self.minicpm_cli)
            .arg("-m")
            .arg(&self.minicpm_path)
            .arg("--mmproj")
            .arg(&self.minicpm_proj)
            .arg("-ngl")
            .arg("999")
            .arg("--image")
            .arg(image_path)
            .arg("-p")
            .arg(prompt)
            .output()
            .map_err(|e| format!("Failed to execute vision cli: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_response_uses_last_assistant_marker() {
        let stdout = "<|im_start|>system\nctx<|im_end|>\n<|im_start|>assistant\nechoed old answer<|im_end|>\n<|im_start|>assistant\nfinal answer\n[ Prompt: 12 tokens]";

        assert_eq!(clean_llama_cli_response(stdout), "final answer");
    }

    #[test]
    fn clean_response_strips_end_token() {
        assert_eq!(
            clean_llama_cli_response("<|im_start|>assistant\nHello<|im_end|>"),
            "Hello"
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
