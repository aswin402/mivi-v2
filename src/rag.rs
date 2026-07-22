use std::fs;
use std::sync::Arc;
use tokio::sync::Mutex;
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub struct RagChunk {
    pub file_path: String,
    pub line_start: usize,
    pub text: String,
}

#[derive(Clone)]
pub struct TurboVecRAG {
    chunks: Arc<Mutex<Vec<RagChunk>>>,
}

impl TurboVecRAG {
    pub fn new() -> Self {
        Self {
            chunks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn index_directory(&self, path: &str) -> usize {
        let mut all_chunks = Vec::new();

        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path_str = entry.path().display().to_string();
                if path_str.contains("/target/") || path_str.contains("/.git/") || path_str.contains("/node_modules/") || path_str.contains("/bin/") {
                    continue;
                }

                if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                    if matches!(ext, "py" | "md" | "rs" | "json" | "js" | "ts" | "toml") {
                        if let Ok(content) = fs::read_to_string(entry.path()) {
                            let lines: Vec<&str> = content.lines().collect();
                            let chunk_size = 25;
                            for (i, chunk_lines) in lines.chunks(chunk_size).enumerate() {
                                let chunk_text = chunk_lines.join("\n");
                                if chunk_text.trim().len() > 10 {
                                    all_chunks.push(RagChunk {
                                        file_path: path_str.clone(),
                                        line_start: (i * chunk_size) + 1,
                                        text: chunk_text,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        let count = all_chunks.len();
        let mut guard = self.chunks.lock().await;
        *guard = all_chunks;
        println!("[TurboVec RAG] Indexed {} code chunks in workspace (< 1 MB RAM footprint)!", count);
        count
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Vec<(RagChunk, f32)> {
        let guard = self.chunks.lock().await;
        let stop_words: std::collections::HashSet<&str> = [
            "the", "is", "to", "in", "a", "and", "of", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "or", "be", "as", "an", "code", "script", "write", "create", "print"
        ].iter().cloned().collect();

        let query_words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= 3 && !stop_words.contains(w))
            .map(|s| s.to_string())
            .collect();

        if query_words.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::new();

        for chunk in guard.iter() {
            let text_lower = chunk.text.to_lowercase();
            let mut score = 0.0f32;

            for word in &query_words {
                if text_lower.contains(word) {
                    score += 1.0;
                }
            }

            if score >= 1.0 {
                results.push((chunk.clone(), score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(top_k).collect()
    }

    pub async fn format_rag_context(&self, query: &str, top_k: usize) -> String {
        let matches = self.search(query, top_k).await;
        if matches.is_empty() {
            return String::new();
        }

        let mut formatted = vec!["# --- REFERENCE CODEBASE CONTEXT (DO NOT COPY OR EXECUTE) ---".to_string()];
        for (i, (chunk, score)) in matches.iter().enumerate() {
            if *score < 1.0 {
                continue;
            }
            let commented_text = chunk
                .text
                .lines()
                .map(|line| format!("# {}", line))
                .collect::<Vec<String>>()
                .join("\n");

            formatted.push(format!(
                "# --- Snippet {} (Source: {} L{}) ---\n{}",
                i + 1,
                chunk.file_path,
                chunk.line_start,
                commented_text
            ));
        }
        formatted.push("# -----------------------------------------------------------".to_string());
        formatted.join("\n\n")
    }
}

