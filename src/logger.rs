use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
pub struct TrainingSample {
    pub instruction: String,
    pub language: String,
    pub input: String,
    pub output: String,
    pub verified_terminal_output: String,
    pub verified: bool,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct DatasetLogger {
    file_path: PathBuf,
}

impl DatasetLogger {
    pub fn new() -> Self {
        let dir = PathBuf::from("dataset");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("verified_pairs.jsonl");
        Self { file_path }
    }

    pub fn save_sample(&self, prompt: &str, code: &str, terminal_output: &str, language: &str) {
        if prompt.is_empty() || code.is_empty() {
            return;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let sample = TrainingSample {
            instruction: prompt.trim().to_string(),
            language: language.to_string(),
            input: String::new(),
            output: format!("```{}\n{}\n```", language, code.trim()),
            verified_terminal_output: terminal_output.trim().to_string(),
            verified: true,
            timestamp: now,
        };

        if let Ok(json_str) = serde_json::to_string(&sample) {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.file_path)
            {
                let _ = writeln!(file, "{}", json_str);
                println!(
                    "[DatasetLogger] Saved verified execution sample to '{:?}'",
                    self.file_path
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn save_sample_uses_declared_language_in_code_fence() {
        let file_path = PathBuf::from("target/logger-test-rust-sample.jsonl");
        let _ = fs::remove_file(&file_path);
        let logger = DatasetLogger {
            file_path: file_path.clone(),
        };

        logger.save_sample("make it", "fn main() {}", "", "rust");

        let content = fs::read_to_string(&file_path).expect("sample should be written");
        let sample: serde_json::Value =
            serde_json::from_str(content.trim()).expect("sample should be json");
        assert_eq!(sample["output"], "```rust\nfn main() {}\n```");
    }
}
