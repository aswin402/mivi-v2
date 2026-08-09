use crate::prompt_file::write_prompt_file;
use crate::runtime::RuntimeConfig;
use crate::worker::{WorkerConfig, WorkerManager};
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

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

fn apply_reasoning_directive(prompt: &str) -> String {
    let trimmed = prompt.trim_start();
    if trimmed.starts_with("/think") || trimmed.starts_with("/no_think") {
        prompt.to_string()
    } else {
        format!("{}\n{}", reasoning_directive(prompt), prompt)
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

fn strip_think_blocks(text: &str) -> String {
    let without_xml = strip_delimited_block(text, "<think>", "</think>");
    let without_bracketed =
        strip_delimited_block(&without_xml, "[start thinking]", "[end thinking]");
    strip_delimited_block(&without_bracketed, "start thinking", "end thinking")
}

fn clean_llama_cli_response(stdout: &str) -> String {
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

fn scrub_generated_prompt_echo(text: &str) -> String {
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

fn model_path_from_env(var: &str, default: PathBuf) -> PathBuf {
    env::var(var).map(PathBuf::from).unwrap_or(default)
}

fn cli_context_size(var: &str, default_tokens: usize) -> String {
    env::var(var)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens >= 1024)
        .unwrap_or(default_tokens)
        .to_string()
}

fn cli_timeout() -> Duration {
    env::var("MIVI_CLI_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(180))
}

async fn command_output_with_timeout(
    mut cmd: tokio::process::Command,
    timeout: Duration,
) -> Result<Output, String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let child = cmd
        .spawn()
        .map_err(|err| format!("Failed to execute llama-cli: {}", err))?;

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(err)) => Err(format!("llama-cli execution error: {}", err)),
        Err(_) => Err(format!(
            "llama-cli timed out after {} seconds",
            timeout.as_secs()
        )),
    }
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
        let llama_path = model_path_from_env(
            "MIVI_REASONER_MODEL",
            models_dir.join("qwen3-0.6b-q4_k_m.gguf"),
        );
        let qwen_path = model_path_from_env(
            "MIVI_CODER_MODEL",
            models_dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        );
        let minicpm_path = model_path_from_env(
            "MIVI_VISION_MODEL",
            models_dir.join("MiniCPM-V-4.6-Q4_K_M.gguf"),
        );
        let minicpm_proj = model_path_from_env(
            "MIVI_VISION_PROJECTOR",
            models_dir.join("mmproj-MiniCPM-V-4.6-Q8_0.gguf"),
        );
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
            info!("[AIRLLM/COLIBRI MODE] Ultra-Low-RAM mmap streaming active (< 40 MB RAM target)");
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

    async fn run_cli(
        &self,
        model_path: &Path,
        prompt: &str,
        system_prompt: &str,
        temp: &str,
        context_size: &str,
    ) -> Result<String, String> {
        let runtime_config = RuntimeConfig::from_env();
        if runtime_config.uses_worker() && model_path == self.llama_path.as_path() {
            let fut = self.text_worker.query_chat(prompt, system_prompt, temp);
            match fut.await {
                Ok(response) => return Ok(response),
                Err(err) => warn!(
                    "[MIVI-V2 Worker] Falling back to llama-cli after worker error: {}",
                    err
                ),
            }
        }

        let t = crate::server::active_chat_template();
        let formatted_prompt = format!(
            "{}{}{}{}{}{}{}",
            t.system_prefix,
            system_prompt,
            t.system_suffix,
            t.user_prefix,
            prompt,
            t.user_suffix,
            t.assistant_start
        );

        let eff_context = if self.ultra_low_ram && context_size == "8192" {
            "4096"
        } else {
            context_size
        };

        let ngl_val = if self.ultra_low_ram { "0" } else { "999" };

        let prompt_file = write_prompt_file(&formatted_prompt)?;
        let mut cmd = tokio::process::Command::new(&self.llama_cli);
        cmd.arg("-m")
            .arg(model_path)
            .arg("-ngl")
            .arg(ngl_val)
            .arg("-c")
            .arg(&eff_context)
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

        let output = match command_output_with_timeout(cmd, cli_timeout()).await {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&prompt_file);
                return Err(e);
            }
        };
        let _ = std::fs::remove_file(&prompt_file);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(clean_llama_cli_response(&stdout))
    }

    pub async fn query_reasoner(
        &self,
        prompt: &str,
        system_prompt: &str,
    ) -> Result<String, String> {
        let runtime_config = RuntimeConfig::from_env();
        let context_size = cli_context_size(
            "MIVI_REASONER_CONTEXT_SIZE",
            runtime_config.context.max_input_tokens,
        );
        let effective_prompt = apply_reasoning_directive(prompt);
        self.run_cli(
            &self.llama_path,
            &effective_prompt,
            system_prompt,
            "0.2",
            &context_size,
        )
        .await
    }

    pub async fn query_coder(&self, prompt: &str, system_prompt: &str) -> Result<String, String> {
        let runtime_config = RuntimeConfig::from_env();
        let context_size = cli_context_size(
            "MIVI_CODER_CONTEXT_SIZE",
            runtime_config.context.max_input_tokens,
        );
        self.run_cli(&self.qwen_path, prompt, system_prompt, "0.1", &context_size)
            .await
    }

    /// Speculative Decoding (ds4 DwarfStar pattern):
    /// Uses the configured coder to draft tokens fast, then uses the configured reasoner to verify.
    pub async fn query_speculative(
        &self,
        prompt: &str,
        system_prompt: &str,
    ) -> Result<String, String> {
        info!("[DS4 SPECULATIVE] Drafting with configured coder...");
        let draft = self.query_coder(prompt, system_prompt).await?;

        if draft.trim().is_empty() {
            return self.query_reasoner(prompt, system_prompt).await;
        }

        let verify_prompt = format!(
            "### Task Verification\nVerify and improve the proposed response for accuracy based on the user request.\n\n#### User Request:\n{}\n\n#### Proposed Response:\n{}\n\nIf the proposed response is accurate, output the response as is. Otherwise output the corrected version.",
            apply_reasoning_directive(prompt), draft
        );

        match self.query_reasoner(&verify_prompt, system_prompt).await {
            Ok(verified) if !verified.trim().is_empty() => Ok(verified),
            _ => Ok(draft),
        }
    }

    pub async fn query_raw(
        &self,
        prompt: &str,
        temp: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        stop: Option<serde_json::Value>,
        seed: Option<u64>,
        json_schema: Option<String>,
    ) -> Result<String, String> {
        let runtime_config = RuntimeConfig::from_env();
        let raw_context = cli_context_size(
            "MIVI_REASONER_CONTEXT_SIZE",
            runtime_config.context.max_input_tokens,
        );
        let eff_context =
            if self.ultra_low_ram && raw_context.parse::<usize>().unwrap_or(3072) > 3072 {
                "3072".to_string()
            } else {
                raw_context
            };
        let ngl_val = if self.ultra_low_ram { "0" } else { "999" };

        let prompt_file = write_prompt_file(prompt)?;
        let mut cmd = tokio::process::Command::new(&self.llama_cli);
        cmd.arg("-m")
            .arg(&self.llama_path)
            .arg("-ngl")
            .arg(ngl_val)
            .arg("-c")
            .arg(&eff_context)
            .arg("-fa")
            .arg("on")
            .arg("-ctk")
            .arg("q8_0")
            .arg("-ctv")
            .arg("q8_0")
            .arg("-f")
            .arg(&prompt_file);

        let temp_str = temp.unwrap_or(0.2).to_string();
        cmd.arg("--temp").arg(&temp_str);

        if let Some(tp) = top_p {
            cmd.arg("--top-p").arg(tp.to_string());
        }
        if let Some(mt) = max_tokens {
            cmd.arg("-n").arg(mt.to_string());
        }
        if let Some(sd) = seed {
            cmd.arg("--seed").arg(sd.to_string());
        }
        if let Some(ref schema) = json_schema {
            cmd.arg("--json-schema").arg(schema);
        }
        if let Some(stop_val) = stop {
            if let Some(s) = stop_val.as_str() {
                cmd.arg("--stop").arg(s);
            } else if let Some(arr) = stop_val.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        cmd.arg("--stop").arg(s);
                    }
                }
            }
        }

        cmd.arg("--simple-io").arg("--no-display-prompt").arg("-st");

        if self.ultra_low_ram {
            cmd.arg("--mmap");
        }

        let output = match command_output_with_timeout(cmd, cli_timeout()).await {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&prompt_file);
                return Err(e);
            }
        };
        let _ = std::fs::remove_file(&prompt_file);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();

        // Try extracting from after last <|im_start|>assistant tag (prompt echo).
        // If not echoed, find the first JSON object or take the last non-empty line.
        let t = crate::server::active_chat_template();
        let assistant_marker = t.assistant_start.trim();
        let response = if !assistant_marker.is_empty() {
            if let Some(pos) = stdout.rfind(assistant_marker) {
                let after = &stdout[pos + assistant_marker.len()..];
                let sys_prefix = t.system_prefix.trim();
                let user_prefix = t.user_prefix.trim();
                let assist_prefix = t.assistant_prefix.trim();
                if let Some(echo_end) = after
                    .find(sys_prefix)
                    .or_else(|| after.find(user_prefix))
                    .or_else(|| after.find(assist_prefix))
                {
                    &after[..echo_end]
                } else {
                    after
                }
            } else {
                &stdout
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

        Ok(strip_think_blocks(clean))
    }

    pub async fn query_vision(&self, image_path: &str, prompt: &str) -> Result<String, String> {
        if !Path::new(image_path).exists() {
            return Err(format!("Image file not found at: {}", image_path));
        }

        if !self.minicpm_path.exists() {
            return Err(format!(
                "Vision model weights not found at '{}'. Download MiniCPM-V-4.6-Q4_K_M.gguf and mmproj-MiniCPM-V-4.6-Q8_0.gguf into models/",
                self.minicpm_path.display()
            ));
        }

        let mut cmd = tokio::process::Command::new(&self.minicpm_cli);
        cmd.arg("-m")
            .arg(&self.minicpm_path)
            .arg("--mmproj")
            .arg(&self.minicpm_proj)
            .arg("-ngl")
            .arg("999")
            .arg("--image")
            .arg(image_path)
            .arg("-p")
            .arg(prompt);

        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to execute vision cli: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

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
    fn cli_context_size_uses_env_and_default() {
        env::set_var("MIVI_TEST_CONTEXT_SIZE", "3072");
        assert_eq!(cli_context_size("MIVI_TEST_CONTEXT_SIZE", 4096), "3072");
        env::set_var("MIVI_TEST_CONTEXT_SIZE", "128");
        assert_eq!(cli_context_size("MIVI_TEST_CONTEXT_SIZE", 4096), "4096");
        env::remove_var("MIVI_TEST_CONTEXT_SIZE");
        assert_eq!(cli_context_size("MIVI_TEST_CONTEXT_SIZE", 4096), "4096");
    }

    #[test]
    fn cli_timeout_uses_env_when_present() {
        env::set_var("MIVI_CLI_TIMEOUT_SECS", "7");
        let timeout = cli_timeout();
        env::remove_var("MIVI_CLI_TIMEOUT_SECS");

        assert_eq!(timeout.as_secs(), 7);
    }

    #[test]
    fn model_path_override_uses_env_when_present() {
        let default = PathBuf::from("models/default.gguf");
        env::set_var("MIVI_TEST_MODEL_PATH", "models/candidate.gguf");
        let path = model_path_from_env("MIVI_TEST_MODEL_PATH", default.clone());
        env::remove_var("MIVI_TEST_MODEL_PATH");

        assert_eq!(path, PathBuf::from("models/candidate.gguf"));
        assert_eq!(
            model_path_from_env("MIVI_TEST_MODEL_PATH", default.clone()),
            default
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
