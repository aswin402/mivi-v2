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
    cmd.arg("-m")
        .arg(model_path)
        .arg("-ngl")
        .arg(ngl)
        .arg("-c")
        .arg(ctx)
        .arg("-fa")
        .arg("on")
        .arg("-ctk")
        .arg("q8_0")
        .arg("-ctv")
        .arg("q8_0")
        .arg("--temp")
        .arg(temp)
        .arg("--simple-io")
        .arg("--no-display-prompt");
}

fn find_marker_case_insensitive(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    let lower = text.to_ascii_lowercase();
    markers
        .iter()
        .filter_map(|marker| lower.find(marker).map(|index| (index, marker.len())))
        .min_by_key(|(index, _)| *index)
}

pub(crate) fn strip_thinking_from_stream_line(
    line: &str,
    skipping_think: &mut bool,
) -> Option<String> {
    const START_MARKERS: &[&str] = &["<think>", "[start thinking]", "start thinking"];
    const END_MARKERS: &[&str] = &["</think>", "[end thinking]", "end thinking"];

    let t = crate::server::active_chat_template();
    let mut clean = line.to_string();
    for stop in &t.stop_words {
        clean = clean.replace(stop, "");
    }
    let clean = clean.trim().to_string();
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
