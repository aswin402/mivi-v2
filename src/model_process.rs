//! Process-per-query llama-cli interaction.
//!
//! Each request spawns a fresh llama-cli subprocess and streams output
//! back token by token (or in chunks when the OS pipe buffer flushes).
//! No persistent process — simpler, more reliable, works with any
//! llama.cpp build.

use crate::prompt_file::write_prompt_file;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

/// Common args shared by all llama-cli invocations.
fn base_args(cmd: &mut Command, model_path: &str, ngl: &str, ctx: &str, temp: &str) {
    let runtime_config = crate::runtime::RuntimeConfig::global();
    cmd.arg("-m")
        .arg(model_path)
        .arg("-ngl")
        .arg(ngl)
        .arg("-c")
        .arg(ctx)
        .arg("-fa")
        .arg("on")
        .arg("-ctk")
        .arg(&runtime_config.kv_cache_type)
        .arg("-ctv")
        .arg(&runtime_config.kv_cache_type)
        .arg("--temp")
        .arg(temp)
        .arg("--simple-io")
        .arg("--no-display-prompt")
        .arg("-t")
        .arg(runtime_config.threads.to_string())
        .arg("-tb")
        .arg(runtime_config.threads.to_string());
}

fn find_marker_case_insensitive(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    let lower = text.to_ascii_lowercase();
    markers
        .iter()
        .filter_map(|marker| lower.find(marker).map(|index| (index, marker.len())))
        .min_by_key(|(index, _)| *index)
}

const THINK_START_TAGS: &[&str] = &["<think>", "[start thinking]", "[think]"];
const THINK_END_TAGS: &[&str] = &["</think>", "[end thinking]", "[/think]"];

#[derive(Debug, Clone)]
pub struct StreamThinkFilter {
    buffer: String,
    inside_think: bool,
    active_stop_words: Vec<String>,
}

impl StreamThinkFilter {
    pub fn new(stop_words: Vec<String>) -> Self {
        Self {
            buffer: String::new(),
            inside_think: false,
            active_stop_words: stop_words,
        }
    }

    pub fn push(&mut self, delta: &str) -> Option<String> {
        self.buffer.push_str(delta);

        let mut ready_to_emit = String::new();

        loop {
            if self.inside_think {
                let lower = self.buffer.to_ascii_lowercase();
                let mut found_end = None;
                for tag in THINK_END_TAGS {
                    if let Some(pos) = lower.find(tag) {
                        found_end = Some((pos, tag.len()));
                        break;
                    }
                }
                if let Some((pos, len)) = found_end {
                    self.buffer.drain(..pos + len);
                    self.inside_think = false;
                    continue;
                }
                if self.buffer.len() > 30 {
                    let keep_len = 16.min(self.buffer.len());
                    let drain_len = self.buffer.len() - keep_len;
                    self.buffer.drain(..drain_len);
                }
                break;
            } else {
                let lower = self.buffer.to_ascii_lowercase();
                let mut start_marker = None;
                for tag in THINK_START_TAGS {
                    if let Some(pos) = lower.find(tag) {
                        start_marker = Some((pos, tag.len()));
                        break;
                    }
                }

                if let Some((pos, len)) = start_marker {
                    let before = self.buffer[..pos].to_string();
                    self.buffer.drain(..pos + len);
                    self.inside_think = true;
                    ready_to_emit.push_str(&before);
                    continue;
                }

                // Check if buffer ends with a prefix of any start tag (e.g. "<", "<th", "[", "[st")
                let max_match = THINK_START_TAGS
                    .iter()
                    .filter_map(|tag| (1..tag.len()).filter(|&l| lower.ends_with(&tag[..l])).max())
                    .max()
                    .unwrap_or(0);

                if max_match > 0 {
                    let safe_len = self.buffer.len().saturating_sub(max_match);
                    let emit_part = self.buffer[..safe_len].to_string();
                    self.buffer.drain(..safe_len);
                    ready_to_emit.push_str(&emit_part);
                    break;
                } else {
                    ready_to_emit.push_str(&self.buffer);
                    self.buffer.clear();
                    break;
                }
            }
        }

        for stop in &self.active_stop_words {
            if !stop.is_empty() {
                ready_to_emit = ready_to_emit.replace(stop, "");
            }
        }

        if ready_to_emit.is_empty() {
            None
        } else {
            Some(ready_to_emit)
        }
    }

    pub fn flush(&mut self) -> Option<String> {
        if self.inside_think {
            self.buffer.clear();
            None
        } else {
            let mut out = std::mem::take(&mut self.buffer);
            for stop in &self.active_stop_words {
                if !stop.is_empty() {
                    out = out.replace(stop, "");
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
    }
}

pub(crate) fn strip_thinking_from_stream_line(
    line: &str,
    skipping_think: &mut bool,
) -> Option<String> {
    const START_MARKERS: &[&str] = &["<think>", "[start thinking]", "start thinking"];
    const END_MARKERS: &[&str] = &["</think>", "[end thinking]", "end thinking"];

    let t = crate::server::active_chat_template();
    let clean = line.to_string();
    let mut rest = clean.as_str();
    let mut out = String::new();

    loop {
        if *skipping_think {
            if let Some((end, len)) = find_marker_case_insensitive(rest, END_MARKERS) {
                rest = &rest[end + len..];
                *skipping_think = false;
                continue;
            }
            return None;
        }

        if let Some((start, start_len)) = find_marker_case_insensitive(rest, START_MARKERS) {
            out.push_str(&rest[..start]);
            rest = &rest[start + start_len..];
            if let Some((end, end_len)) = find_marker_case_insensitive(rest, END_MARKERS) {
                rest = &rest[end + end_len..];
                continue;
            }
            *skipping_think = true;
            break;
        }

        out.push_str(rest);
        break;
    }

    for stop in &t.stop_words {
        out = out.replace(stop, "");
    }

    let trimmed = out.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Spawn llama-cli with a chat-formatted prompt and return a channel that
/// yields response content strings as they arrive.
///
/// The spawned process is cleaned up when the channel is dropped.
pub fn spawn_streaming(
    llama_cli: &str,
    model_path: &str,
    formatted_prompt: &str,
    ngl: &str,
    context_size: &str,
    temp: &str,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<serde_json::Value>,
    seed: Option<u64>,
    json_schema: Option<String>,
    grammar_path: Option<String>,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>(64);

    let cli = PathBuf::from(llama_cli);
    let mp = model_path.to_string();
    let prompt = formatted_prompt.to_string();
    let n = ngl.to_string();
    let ctx = context_size.to_string();
    let t = temp.to_string();

    tokio::spawn(async move {
        let prompt_file = match write_prompt_file(&prompt).await {
            Ok(path) => path,
            Err(e) => {
                let _ = tx.send(format!("[prompt file error: {}]", e)).await;
                return;
            }
        };
        let mut cmd = Command::new(&cli);
        base_args(&mut cmd, &mp, &n, &ctx, &t);

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
        if let Some(ref path) = grammar_path {
            if std::path::Path::new(path).exists() {
                cmd.arg("--grammar-file").arg(path);
            }
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

        cmd.arg("-f")
            .arg(&prompt_file)
            .arg("-st") // single-turn: exit after generation
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_file(&prompt_file);
                let _ = tx.send(format!("[spawn error: {}]", e)).await;
                return;
            }
        };

        // Read stderr in background so the pipe doesn't block.
        let stderr = match child.stderr.take() {
            Some(s) => s,
            None => {
                let _ = std::fs::remove_file(&prompt_file);
                let _ = tx.send("[spawn error: stderr not piped]".to_string()).await;
                return;
            }
        };
        tokio::spawn(async move {
            let mut err_lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = err_lines.next_line().await {
                eprintln!("[llama-cli stderr] {}", line);
            }
        });

        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let _ = std::fs::remove_file(&prompt_file);
                let _ = tx.send("[spawn error: stdout not piped]".to_string()).await;
                return;
            }
        };
        let mut lines = BufReader::new(stdout).lines();

        enum State {
            /// Skipping banner / prompt echo — waiting for `<|im_start|>assistant` marker.
            FindingAssistant,
            /// Reading response tokens.
            Collecting,
        }
        let mut state = State::FindingAssistant;
        let mut skipping_think = false;

        while let Ok(Some(line)) = lines.next_line().await {
            match state {
                State::FindingAssistant => {
                    let t = crate::server::active_chat_template();
                    let marker = t.assistant_start.trim();
                    if line.contains(marker) {
                        state = State::Collecting;
                    }
                }
                State::Collecting => {
                    // End-of-generation signals.
                    if line.starts_with("[ Prompt:") || line.starts_with("Exiting...") {
                        break;
                    }
                    let Some(clean) = strip_thinking_from_stream_line(&line, &mut skipping_think)
                    else {
                        continue;
                    };
                    if tx.send(clean).await.is_err() {
                        let _ = child.kill().await;
                        break; // receiver dropped (client disconnected)
                    }
                }
            }
        }

        let _ = child.wait().await;
        let _ = std::fs::remove_file(&prompt_file);

        #[cfg(target_os = "linux")]
        {
            let is_ultra_low = std::env::var("MIVI_ULTRA_LOW_RAM").is_ok();
            if is_ultra_low {
                if let Ok(file) = std::fs::File::open(&mp) {
                    use std::os::unix::io::AsRawFd;
                    let fd = file.as_raw_fd();
                    unsafe {
                        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
                    }
                }
            }
        }
    });

    rx
}
#[cfg(test)]
mod tests {
    use super::strip_thinking_from_stream_line;

    #[test]
    fn strips_plain_qwen_thinking_but_keeps_answer_tail() {
        let mut skipping = false;
        assert_eq!(
            strip_thinking_from_stream_line(
                "Start thinking private End thinking Hello.",
                &mut skipping,
            ),
            Some("Hello.".to_string())
        );
        assert!(!skipping);
    }

    #[test]
    fn strips_multiline_thinking_until_end_marker() {
        let mut skipping = false;
        assert_eq!(
            strip_thinking_from_stream_line("<think>", &mut skipping),
            None
        );
        assert!(skipping);
        assert_eq!(
            strip_thinking_from_stream_line("private reasoning", &mut skipping),
            None
        );
        assert_eq!(
            strip_thinking_from_stream_line("</think> Final answer.", &mut skipping),
            Some("Final answer.".to_string())
        );
        assert!(!skipping);
    }
}
