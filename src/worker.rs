use crate::runtime::RuntimeConfig;
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
        vec![
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
            "q8_0".to_string(),
            "-ctv".to_string(),
            "q8_0".to_string(),
            "-np".to_string(),
            "1".to_string(),
            "--sleep-idle-seconds".to_string(),
            self.config.idle_secs.to_string(),
            "--no-webui".to_string(),
            "--mmap".to_string(),
        ]
    }

    pub async fn ensure_text_worker(&self) -> Result<WorkerEndpoint, String> {
        let endpoint_to_check = {
            let slot = self
                .slot
                .lock()
                .map_err(|_| "worker lock poisoned".to_string())?;
            if slot.state == WorkerState::Ready {
                slot.endpoint.clone()
            } else {
                None
            }
        };

        if let Some(endpoint) = endpoint_to_check {
            if self.health_check(&endpoint).await {
                return Ok(endpoint);
            }
        }

        self.start_worker().await
    }

    pub async fn query_chat(
        &self,
        prompt: &str,
        system_prompt: &str,
        temp: &str,
    ) -> Result<String, String> {
        let endpoint = self.ensure_text_worker().await?;
        let body = json!({
            "model": "mivi-worker",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp.parse::<f32>().unwrap_or(0.2),
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
    ) -> Result<serde_json::Value, String> {
        let endpoint = self.ensure_text_worker().await?;
        let mut body = json!({
            "model": "mivi-worker",
            "messages": messages,
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
        if let Some(t) = tools {
            body["tools"] = t;
        }
        if let Some(tc) = tool_choice {
            body["tool_choice"] = tc;
        }
        if let Some(rf) = response_format {
            body["response_format"] = rf;
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
            let _ = child.wait();
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
        cmd.args(self.server_args())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

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
        .filter(|content| !content.is_empty())
        .ok_or_else(|| "worker response missing assistant content".to_string())
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
        }
    }

    #[test]
    fn spawn_mode_does_not_use_workers() {
        assert!(!config(RuntimeMode::Spawn).uses_worker());
        assert!(config(RuntimeMode::WorkerEco).uses_worker());
        assert!(config(RuntimeMode::WorkerHot).uses_worker());
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
