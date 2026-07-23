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
        .arg("--simple-io");
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
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>(64);

    let cli = PathBuf::from(llama_cli);
    let mp = model_path.to_string();
    let prompt = formatted_prompt.to_string();
    let n = ngl.to_string();
    let ctx = context_size.to_string();
    let t = temp.to_string();

    tokio::spawn(async move {
        let prompt_file = match write_prompt_file(&prompt) {
            Ok(path) => path,
            Err(e) => {
                let _ = tx.send(format!("[prompt file error: {}]", e)).await;
                return;
            }
        };
        let mut cmd = Command::new(&cli);
        base_args(&mut cmd, &mp, &n, &ctx, &t);
        cmd.arg("-f")
            .arg(&prompt_file)
            .arg("-st") // single-turn: exit after generation
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_file(&prompt_file);
                let _ = tx.send(format!("[spawn error: {}]", e)).await;
                return;
            }
        };

        // Read stderr in background so the pipe doesn't block.
        let stderr = child.stderr.take().unwrap();
        tokio::spawn(async move {
            let mut err_lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = err_lines.next_line().await {
                eprintln!("[llama-cli stderr] {}", line);
            }
        });

        let stdout = child
            .stdout
            .take()
            .expect("[spawn_streaming] stdout not piped");
        let mut lines = BufReader::new(stdout).lines();

        enum State {
            /// Skipping banner / prompt echo — waiting for `<|im_start|>assistant` marker.
            FindingAssistant,
            /// Reading response tokens.
            Collecting,
        }
        let mut state = State::FindingAssistant;

        while let Ok(Some(line)) = lines.next_line().await {
            match state {
                State::FindingAssistant => {
                    if line.contains("<|im_start|>assistant") {
                        state = State::Collecting;
                    }
                }
                State::Collecting => {
                    // End-of-generation signals.
                    if line.starts_with("[ Prompt:") || line.starts_with("Exiting...") {
                        break;
                    }
                    // Strip special tokens that leak from the model.
                    let clean = line
                        .replace("<|im_start|>", "")
                        .replace("<|im_end|>", "")
                        .trim()
                        .to_string();
                    if clean.is_empty() {
                        continue;
                    }
                    if tx.send(clean).await.is_err() {
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
