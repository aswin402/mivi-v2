use crate::runtime::RuntimeConfig;
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerEndpoint {
    pub host: String,
    pub port: u16,
}

impl WorkerEndpoint {
    fn base_url(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    pub server_path: PathBuf,
    pub model_path: PathBuf,
    pub host: String,
    pub port: u16,
    pub context_tokens: usize,
    pub gpu_layers: String,
    pub idle_secs: u64,
    /// Minimum chunk size for llama-server KV prefix reuse (`--cache-reuse`).
    /// 0 disables the flag entirely.
    pub cache_reuse_tokens: u32,
    pub threads: usize,
}

impl WorkerConfig {
    pub fn default_for_text_model(
        server_path: PathBuf,
        model_path: PathBuf,
        runtime: &RuntimeConfig,
    ) -> Self {
        let port = env::var("MIVI_WORKER_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(18080);
        let gpu_layers = if env::var("MIVI_ULTRA_LOW_RAM")
            .map(|value| value == "1" || value == "true")
            .unwrap_or(false)
        {
            "0".to_string()
        } else {
            "999".to_string()
        };

        Self {
            server_path,
            model_path,
            host: "127.0.0.1".to_string(),
            port,
            context_tokens: runtime.context.max_input_tokens,
            gpu_layers,
            idle_secs: runtime.worker_idle_secs,
            cache_reuse_tokens: env::var("MIVI_WORKER_CACHE_REUSE")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(64),
            threads: runtime.threads,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Stopped,
    Starting,
    Ready,
    Failed,
    IdleStopped,
}

struct WorkerSlot {
    state: WorkerState,
    child: Option<Child>,
    endpoint: Option<WorkerEndpoint>,
    last_error: Option<String>,
}

#[derive(Clone)]
pub struct WorkerManager {
    config: WorkerConfig,
    slot: Arc<Mutex<WorkerSlot>>,
    client: reqwest::Client,
}

impl WorkerManager {
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            slot: Arc::new(Mutex::new(WorkerSlot {
                state: WorkerState::Stopped,
                child: None,
                endpoint: None,
                last_error: None,
            })),
            client: reqwest::Client::new(),
        }
    }

    pub fn server_args(&self) -> Vec<String> {
        let runtime_config = RuntimeConfig::global();
        let mut args = vec![
            "-m".to_string(),
            self.config.model_path.display().to_string(),
            "--host".to_string(),
            self.config.host.clone(),
            "--port".to_string(),
            self.config.port.to_string(),
            "-c".to_string(),
            self.config.context_tokens.to_string(),
            "-ngl".to_string(),
            self.config.gpu_layers.clone(),
            "-fa".to_string(),
            "on".to_string(),
            "-ctk".to_string(),
            runtime_config.kv_cache_type.clone(),
            "-ctv".to_string(),
            runtime_config.kv_cache_type.clone(),
            "-np".to_string(),
            "1".to_string(),
            "--sleep-idle-seconds".to_string(),
            self.config.idle_secs.to_string(),
            "--no-webui".to_string(),
            "--mmap".to_string(),
            "-t".to_string(),
            self.config.threads.to_string(),
            "-tb".to_string(),
            self.config.threads.to_string(),
        ];

        // KV prefix reuse: consecutive requests sharing a prompt prefix
        // (agent tool-loops re-send the system prompt every turn) skip
        // re-prefilling the shared part.
        if self.config.cache_reuse_tokens > 0 {
            args.push("--cache-reuse".to_string());
            args.push(self.config.cache_reuse_tokens.to_string());
        }

        if let Some(ref draft_path) = runtime_config.draft_model {
            if std::path::Path::new(draft_path).exists() {
                args.push("--model-draft".to_string());
                args.push(draft_path.clone());
                args.push("--gpu-layers-draft".to_string());
                args.push(self.config.gpu_layers.clone());
            }
        }

        // Multi-LoRA specialist adapters (Phase 17.1); pre-flattened and
        // existence-checked by RuntimeConfig.
        args.extend(runtime_config.lora_args.iter().cloned());

        args
    }

    pub async fn ensure_text_worker(&self) -> Result<WorkerEndpoint, String> {
        loop {
            let (state, endpoint) = {
                let slot = self
                    .slot
                    .lock()
                    .map_err(|_| "worker lock poisoned".to_string())?;
                (slot.state, slot.endpoint.clone())
            };

            match state {
                WorkerState::Ready => {
                    if let Some(ep) = endpoint {
                        if self.health_check(&ep).await {
                            return Ok(ep);
                        }
                    }
                }
                WorkerState::Starting => {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    continue;
                }
                _ => {}
            }

            let proceed = {
                let mut slot = self
                    .slot
                    .lock()
                    .map_err(|_| "worker lock poisoned".to_string())?;
                if slot.state == WorkerState::Ready {
                    false
                } else if slot.state == WorkerState::Starting {
                    false
                } else {
                    slot.state = WorkerState::Starting;
                    slot.last_error = None;
                    true
                }
            };

            if proceed {
                return self.start_worker().await;
            }
        }
    }

    pub async fn check_liveness(&self) -> Result<&'static str, String> {
        let (state, endpoint) = {
            let slot = self
                .slot
                .lock()
                .map_err(|_| "worker lock poisoned".to_string())?;
            (slot.state, slot.endpoint.clone())
        };
        match state {
            WorkerState::Stopped => Ok("stopped"),
            WorkerState::Starting => Ok("starting"),
            WorkerState::Failed => Ok("failed"),
            WorkerState::IdleStopped => Ok("idle_stopped"),
            WorkerState::Ready => {
                if let Some(ep) = endpoint {
                    if self.health_check(&ep).await {
                        Ok("ready")
                    } else {
                        Err("worker ready in slot but failed health check".to_string())
                    }
                } else {
                    Err("worker state ready but endpoint is None".to_string())
                }
            }
        }
    }

    pub async fn query_chat(
        &self,
        prompt: &str,
        system_prompt: &str,
        temp: &str,
    ) -> Result<String, String> {
        let endpoint = self.ensure_text_worker().await?;

        // Detect if the prompt is already formatted with chat template tokens.
        let t = crate::server::active_chat_template();
        let (final_system, final_user) = if crate::reasoning::is_prompt_preformatted(prompt) {
            // Pre-formatted: extract content from within template tokens.
            // Strip the template markers to get raw system/user content for the worker API.
            let sys_prefix = t.system_prefix.trim();
            let sys_suffix = t.system_suffix.trim();
            let usr_prefix = t.user_prefix.trim();
            let usr_suffix = t.user_suffix.trim();

            let mut system_text = String::new();
            let mut user_text = String::new();

            if let Some(sys_start) = prompt.find(sys_prefix) {
                let after_sys = sys_start + sys_prefix.len();
                if let Some(sys_end) = prompt[after_sys..].find(sys_suffix) {
                    system_text = prompt[after_sys..after_sys + sys_end].trim().to_string();
                }
            }
            if let Some(usr_start) = prompt.find(usr_prefix) {
                let after_usr = usr_start + usr_prefix.len();
                if let Some(usr_end) = prompt[after_usr..].find(usr_suffix) {
                    user_text = prompt[after_usr..after_usr + usr_end].trim().to_string();
                }
            }
            if system_text.is_empty() {
                system_text = system_prompt.to_string();
            }
            if user_text.is_empty() {
                user_text = prompt.to_string();
            }
            (system_text, user_text)
        } else {
            let (extracted_system, extracted_user) = crate::brain::split_prompt_system_user(prompt);
            let final_system = if extracted_system.is_empty() {
                system_prompt.to_string()
            } else {
                extracted_system
            };
            let final_user = if extracted_user.is_empty() {
                prompt.to_string()
            } else {
                extracted_user
            };
            (final_system, final_user)
        };

        let messages = parse_prompt_to_messages(&final_system, &final_user);
        info!(
            "Worker query_chat: sending {} structured messages to worker endpoint",
            messages.len()
        );

        let body = json!({
            "model": "mivi-worker",
            "messages": messages,
            "temperature": temp.parse::<f32>().unwrap_or(0.2),
            "repeat_penalty": 1.1,
            "stream": false
        });
        let response_body = http_request(
            &self.client,
            &endpoint,
            "POST",
            "/v1/chat/completions",
            Some(&body.to_string()),
            Duration::from_secs(300),
        )
        .await?;
        parse_chat_response(&response_body)
    }

    pub async fn query_chat_full(
        &self,
        messages: serde_json::Value,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
        response_format: Option<serde_json::Value>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        stop: Option<serde_json::Value>,
        seed: Option<u64>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        grammar: Option<String>,
    ) -> Result<serde_json::Value, String> {
        let endpoint = self.ensure_text_worker().await?;
        let norm_messages = if let Some(arr) = messages.as_array() {
            let normalized: Vec<serde_json::Value> = arr
                .iter()
                .map(|msg| {
                    let mut new_msg = msg.clone();
                    if let Some(content) = msg.get("content") {
                        if content.is_object() {
                            new_msg["content"] = serde_json::Value::String(content.to_string());
                        }
                    }
                    new_msg
                })
                .collect();
            serde_json::Value::Array(normalized)
        } else {
            messages.clone()
        };
        let mut body = json!({
            "model": "mivi-worker",
            "messages": norm_messages,
            "stream": false
        });
        if let Some(temp) = temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(tp) = top_p {
            body["top_p"] = json!(tp);
        }
        if let Some(mt) = max_tokens {
            body["max_tokens"] = json!(mt);
        }
        if let Some(st) = stop {
            body["stop"] = st;
        }
        if let Some(sd) = seed {
            body["seed"] = json!(sd);
        }
        if let Some(fp) = frequency_penalty {
            body["frequency_penalty"] = json!(fp);
        }
        if let Some(pp) = presence_penalty {
            body["presence_penalty"] = json!(pp);
        }
        if let Some(ref t) = tools {
            body["tools"] = t.clone();
        }
        if let Some(tc) = tool_choice {
            body["tool_choice"] = tc;
        }
        if let Some(rf) = response_format {
            body["response_format"] = rf;
        }
        let has_tools_val = tools
            .as_ref()
            .map(|t| !t.is_null() && (!t.is_array() || !t.as_array().unwrap().is_empty()))
            .unwrap_or(false);
        if !has_tools_val {
            if let Some(g) = grammar {
                body["grammar"] = serde_json::Value::String(g);
            }
        }
        let body_str = http_request(
            &self.client,
            &endpoint,
            "POST",
            "/v1/chat/completions",
            Some(&body.to_string()),
            Duration::from_secs(300),
        )
        .await?;
        serde_json::from_str(&body_str)
            .map_err(|err| format!("failed to parse worker JSON response: {err}"))
    }

    pub async fn query_completion(
        &self,
        prompt: &str,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        stop: Option<serde_json::Value>,
        seed: Option<u64>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        json_schema: Option<String>,
        grammar: Option<String>,
    ) -> Result<serde_json::Value, String> {
        let endpoint = self.ensure_text_worker().await?;
        let mut body = json!({
            "prompt": prompt,
            "stream": false
        });
        if let Some(temp) = temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(tp) = top_p {
            body["top_p"] = json!(tp);
        }
        if let Some(mt) = max_tokens {
            body["n_predict"] = json!(mt);
        }
        if let Some(st) = stop {
            body["stop"] = st;
        }
        if let Some(sd) = seed {
            body["seed"] = json!(sd);
        }
        if let Some(fp) = frequency_penalty {
            body["frequency_penalty"] = json!(fp);
        }
        if let Some(pp) = presence_penalty {
            body["presence_penalty"] = json!(pp);
        }
        if let Some(ref schema_str) = json_schema {
            if let Ok(schema_val) = serde_json::from_str::<serde_json::Value>(schema_str) {
                body["json_schema"] = schema_val;
            }
        }
        if let Some(g) = grammar {
            body["grammar"] = serde_json::Value::String(g);
        }

        let body_str = http_request(
            &self.client,
            &endpoint,
            "POST",
            "/completion",
            Some(&body.to_string()),
            Duration::from_secs(300),
        )
        .await?;
        serde_json::from_str(&body_str)
            .map_err(|err| format!("failed to parse worker JSON response: {err}"))
    }

    pub async fn query_completion_stream(
        &self,
        prompt: &str,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        stop: Option<serde_json::Value>,
        seed: Option<u64>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        json_schema: Option<String>,
        grammar: Option<String>,
    ) -> Result<impl futures::stream::Stream<Item = Result<bytes::Bytes, reqwest::Error>>, String>
    {
        let endpoint = self.ensure_text_worker().await?;
        let mut body = json!({
            "prompt": prompt,
            "stream": true
        });
        if let Some(temp) = temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(tp) = top_p {
            body["top_p"] = json!(tp);
        }
        if let Some(mt) = max_tokens {
            body["n_predict"] = json!(mt);
        }
        if let Some(st) = stop {
            body["stop"] = st;
        }
        if let Some(sd) = seed {
            body["seed"] = json!(sd);
        }
        if let Some(fp) = frequency_penalty {
            body["frequency_penalty"] = json!(fp);
        }
        if let Some(pp) = presence_penalty {
            body["presence_penalty"] = json!(pp);
        }
        if let Some(ref schema_str) = json_schema {
            if let Ok(schema_val) = serde_json::from_str::<serde_json::Value>(schema_str) {
                body["json_schema"] = schema_val;
            }
        }
        if let Some(g) = grammar {
            body["grammar"] = serde_json::Value::String(g);
        }

        let response = self
            .client
            .post(format!("http://{}/completion", endpoint.base_url()))
            .json(&body)
            .send()
            .await
            .map_err(|err| format!("failed to send completion stream request: {err}"))?;

        if !response.status().is_success() {
            let err_body = response.text().await.unwrap_or_default();
            return Err(format!(
                "worker completion stream request failed: {}",
                err_body
            ));
        }

        Ok(response.bytes_stream())
    }

    pub fn stop_idle_workers(&self) -> Result<(), String> {
        let mut slot = self
            .slot
            .lock()
            .map_err(|_| "worker lock poisoned".to_string())?;
        if let Some(mut child) = slot.child.take() {
            let _ = child.kill();
            let model_path = self.config.model_path.clone();
            std::thread::spawn(move || {
                let _ = child.wait();

                #[cfg(target_os = "linux")]
                {
                    let is_ultra_low = std::env::var("MIVI_ULTRA_LOW_RAM").is_ok();
                    if is_ultra_low {
                        if let Ok(file) = std::fs::File::open(model_path) {
                            use std::os::unix::io::AsRawFd;
                            let fd = file.as_raw_fd();
                            unsafe {
                                libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
                            }
                        }
                    }
                }
            });
        }
        slot.endpoint = None;
        slot.state = WorkerState::IdleStopped;
        Ok(())
    }

    async fn start_worker(&self) -> Result<WorkerEndpoint, String> {
        {
            let mut slot = self
                .slot
                .lock()
                .map_err(|_| "worker lock poisoned".to_string())?;
            slot.state = WorkerState::Starting;
            slot.last_error = None;
        }

        let mut cmd = Command::new(&self.config.server_path);
        let _ = std::fs::create_dir_all("logs");
        if let Ok(stderr_file) = std::fs::File::create("logs/worker-stderr.log") {
            cmd.stderr(std::process::Stdio::from(stderr_file));
        } else {
            cmd.stderr(std::process::Stdio::null());
        }
        cmd.stdout(std::process::Stdio::null());

        cmd.args(self.server_args());

        if let Some(bin_dir) = self.config.server_path.parent() {
            let old_ld = env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let ld = if old_ld.is_empty() {
                bin_dir.display().to_string()
            } else {
                format!("{}:{}", bin_dir.display(), old_ld)
            };
            cmd.env("LD_LIBRARY_PATH", ld);
        }

        let child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) => {
                let msg = format!("failed to start llama-server: {err}");
                self.mark_failed(msg.clone());
                return Err(msg);
            }
        };

        let endpoint = WorkerEndpoint {
            host: self.config.host.clone(),
            port: self.config.port,
        };

        {
            let mut slot = self
                .slot
                .lock()
                .map_err(|_| "worker lock poisoned".to_string())?;
            slot.child = Some(child);
            slot.endpoint = Some(endpoint.clone());
        }

        if self
            .wait_until_ready(&endpoint, Duration::from_secs(60))
            .await
        {
            let mut slot = self
                .slot
                .lock()
                .map_err(|_| "worker lock poisoned".to_string())?;
            slot.state = WorkerState::Ready;
            Ok(endpoint)
        } else {
            let msg = "llama-server did not become healthy before timeout".to_string();
            let _ = self.stop_idle_workers();
            self.mark_failed(msg.clone());
            Err(msg)
        }
    }

    async fn wait_until_ready(&self, endpoint: &WorkerEndpoint, timeout: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.health_check(endpoint).await {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        false
    }

    async fn health_check(&self, endpoint: &WorkerEndpoint) -> bool {
        http_request(
            &self.client,
            endpoint,
            "GET",
            "/health",
            None,
            Duration::from_secs(2),
        )
        .await
        .is_ok()
            || http_request(
                &self.client,
                endpoint,
                "GET",
                "/v1/models",
                None,
                Duration::from_secs(2),
            )
            .await
            .is_ok()
    }

    fn mark_failed(&self, error: String) {
        if let Ok(mut slot) = self.slot.lock() {
            slot.state = WorkerState::Failed;
            slot.last_error = Some(error);
            slot.endpoint = None;
            slot.child = None;
        }
    }

    #[cfg(test)]
    pub fn mark_ready_for_test(&self, endpoint: WorkerEndpoint) {
        let mut slot = self.slot.lock().unwrap();
        slot.state = WorkerState::Ready;
        slot.endpoint = Some(endpoint);
    }

    #[cfg(test)]
    pub fn state_for_test(&self) -> WorkerState {
        self.slot.lock().unwrap().state
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) {
        if Arc::strong_count(&self.slot) == 1 {
            let _ = self.stop_idle_workers();
        }
    }
}

async fn http_request(
    client: &reqwest::Client,
    endpoint: &WorkerEndpoint,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> Result<String, String> {
    let url = format!("http://{}{}", endpoint.base_url(), path);
    let mut builder = match method {
        "GET" => client.get(&url),
        "POST" => client.post(&url),
        _ => return Err(format!("Unsupported HTTP method: {method}")),
    };

    if let Some(body_data) = body {
        builder = builder
            .header("Content-Type", "application/json")
            .body(body_data.to_string());
    }

    let response = builder
        .timeout(timeout)
        .send()
        .await
        .map_err(|err| format!("failed to send HTTP request to worker: {err}"))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| format!("failed to read response body: {err}"))?;

    if !status.is_success() {
        return Err(format!(
            "worker returned non-200 response ({}): {}",
            status, text
        ));
    }

    Ok(text)
}

fn parse_chat_response(body: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|err| format!("failed to parse worker JSON response: {err}"))?;

    value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .map(|content| content.trim().to_string())
        .ok_or_else(|| "worker response missing assistant content".to_string())
}

fn parse_prompt_to_messages(system_prompt: &str, user_prompt: &str) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();

    if !system_prompt.trim().is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system_prompt.trim()
        }));
    }

    let mut current_role = None;
    let mut current_content = String::new();

    let has_context = user_prompt.contains("Relevant recent context:");
    let mut lines = user_prompt.lines();

    if has_context {
        for line in &mut lines {
            if line.trim().starts_with("Relevant recent context:") {
                break;
            }
        }
    }

    for line in lines {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with("Relevant recent context:") {
            continue;
        }
        if trimmed_line.starts_with("Current user request:") {
            if let Some(role) = current_role {
                if !current_content.trim().is_empty() {
                    messages.push(serde_json::json!({
                        "role": role,
                        "content": current_content.trim().to_string()
                    }));
                }
            }
            current_role = Some("user");
            current_content = String::new();
            continue;
        }

        if trimmed_line.starts_with("user: ") {
            if let Some(role) = current_role {
                if !current_content.trim().is_empty() {
                    messages.push(serde_json::json!({
                        "role": role,
                        "content": current_content.trim().to_string()
                    }));
                }
            }
            current_role = Some("user");
            current_content = trimmed_line["user: ".len()..].to_string();
        } else if trimmed_line.starts_with("assistant: ") {
            if let Some(role) = current_role {
                if !current_content.trim().is_empty() {
                    messages.push(serde_json::json!({
                        "role": role,
                        "content": current_content.trim().to_string()
                    }));
                }
            }
            current_role = Some("assistant");
            current_content = trimmed_line["assistant: ".len()..].to_string();
        } else {
            if current_role.is_some() {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            } else {
                current_role = Some("user");
                current_content = line.to_string();
            }
        }
    }

    if let Some(role) = current_role {
        if !current_content.trim().is_empty() {
            messages.push(serde_json::json!({
                "role": role,
                "content": current_content.trim().to_string()
            }));
        }
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ContextBudget, RuntimeConfig, RuntimeMode};
    use std::path::PathBuf;

    fn config(mode: RuntimeMode) -> RuntimeConfig {
        RuntimeConfig {
            mode,
            context: ContextBudget::from_max_input_tokens(4096),
            worker_idle_secs: 30,
            ram_target_mb: 1000,
            kv_cache_type: "q4_0".to_string(),
            threads: 1,
            draft_model: None,
            lora_args: Vec::new(),
        }
    }

    #[test]
    fn spawn_mode_does_not_use_workers() {
        assert!(!config(RuntimeMode::Spawn).uses_worker());
        assert!(!config(RuntimeMode::Native).uses_worker());
        assert!(config(RuntimeMode::WorkerEco).uses_worker());
        assert!(config(RuntimeMode::WorkerHot).uses_worker());
    }

    #[test]
    fn test_parse_prompt_to_messages_reconstruction() {
        let system_prompt = "You are openz, a helpful assistant.";
        let user_prompt = "\
so what are the tools u have is now its worked good or we have problem

Relevant recent context:
assistant: I'm glad to see you're using OpenZ...
user: so whats new
assistant: **So, what's new?**...
user: hey i said whats new in u

You are OpenZ, a high-performance framework...
";
        let messages = parse_prompt_to_messages(system_prompt, user_prompt);
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(
            messages[0]["content"],
            "You are openz, a helpful assistant."
        );

        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[1]["content"],
            "I'm glad to see you're using OpenZ..."
        );

        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "so whats new");

        assert_eq!(messages[3]["role"], "assistant");
        assert_eq!(messages[3]["content"], "**So, what's new?**...");

        assert_eq!(messages[4]["role"], "user");
        assert!(messages[4]["content"]
            .as_str()
            .unwrap()
            .contains("hey i said whats new in u"));
    }

    #[test]
    fn text_worker_args_use_low_ram_local_server_defaults() {
        let manager = WorkerManager::new(WorkerConfig {
            server_path: PathBuf::from("bin/llama-server"),
            model_path: PathBuf::from("models/qwen3-0.6b-q4_k_m.gguf"),
            host: "127.0.0.1".to_string(),
            port: 18080,
            context_tokens: 4096,
            gpu_layers: "0".to_string(),
            idle_secs: 30,
            cache_reuse_tokens: 64,
            threads: 1,
        });

        let args = manager.server_args();

        assert!(args.windows(2).any(|pair| pair == ["--host", "127.0.0.1"]));
        assert!(args.windows(2).any(|pair| pair == ["--port", "18080"]));
        assert!(args.windows(2).any(|pair| pair == ["-c", "4096"]));
        assert!(args.windows(2).any(|pair| pair == ["-ngl", "0"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--sleep-idle-seconds", "30"]));
        assert!(args.contains(&"--no-webui".to_string()));

        assert!(args.contains(&"--mmap".to_string()));
    }
    #[test]
    fn text_worker_args_include_cache_reuse_when_enabled() {
        let manager = WorkerManager::new(WorkerConfig {
            server_path: PathBuf::from("bin/llama-server"),
            model_path: PathBuf::from("models/qwen3-0.6b-q4_k_m.gguf"),
            host: "127.0.0.1".to_string(),
            port: 18080,
            context_tokens: 4096,
            gpu_layers: "0".to_string(),
            idle_secs: 30,
            cache_reuse_tokens: 128,
            threads: 1,
        });
        let args = manager.server_args();
        assert!(args.windows(2).any(|pair| pair == ["--cache-reuse", "128"]));
    }

    #[test]
    fn text_worker_args_omit_cache_reuse_when_disabled() {
        let manager = WorkerManager::new(WorkerConfig {
            server_path: PathBuf::from("bin/llama-server"),
            model_path: PathBuf::from("models/qwen3-0.6b-q4_k_m.gguf"),
            host: "127.0.0.1".to_string(),
            port: 18080,
            context_tokens: 4096,
            gpu_layers: "0".to_string(),
            idle_secs: 30,
            cache_reuse_tokens: 0,
            threads: 1,
        });
        let args = manager.server_args();
        assert!(!args.contains(&"--cache-reuse".to_string()));
    }

    #[test]
    fn idle_state_transition_marks_ready_worker_as_idle_stopped() {
        let manager = WorkerManager::new(WorkerConfig::default_for_text_model(
            PathBuf::from("bin/llama-server"),
            PathBuf::from("models/qwen3-0.6b-q4_k_m.gguf"),
            &config(RuntimeMode::WorkerEco),
        ));

        manager.mark_ready_for_test(WorkerEndpoint {
            host: "127.0.0.1".to_string(),
            port: 18080,
        });
        manager.stop_idle_workers().unwrap();

        assert_eq!(manager.state_for_test(), WorkerState::IdleStopped);
    }
}
