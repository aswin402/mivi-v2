use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static PROMPT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub async fn write_prompt_file(prompt: &str) -> Result<PathBuf, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let count = PROMPT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mivi_prompt_{}_{}_{}.txt",
        std::process::id(),
        now,
        count
    ));
    tokio::fs::write(&path, prompt)
        .await
        .map_err(|e| format!("Failed to write prompt file: {}", e))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_prompt_file_preserves_large_agent_prompt() {
        let prompt = "tool ".repeat(80_000);
        let path = write_prompt_file(&prompt)
            .await
            .expect("prompt file should be written");

        let saved = tokio::fs::read_to_string(&path)
            .await
            .expect("prompt file should be readable");
        let _ = tokio::fs::remove_file(&path).await;

        assert_eq!(saved, prompt);
    }
}
