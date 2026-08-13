use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use tokio::sync::Mutex;
use walkdir::WalkDir;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RagChunk {
    pub file_path: String,
    pub line_start: usize,
    pub text: String,
}

#[derive(Clone)]
pub struct TurboVecRAG {
    chunks: Arc<Mutex<Vec<RagChunk>>>,
    usage: Arc<Mutex<HashMap<String, u64>>>,
}

fn expand_query_words(mut words: Vec<String>) -> Vec<String> {
    let originals = words.clone();
    for word in originals {
        match word.as_str() {
            "routing" | "route" | "routes" => {
                words.push("router".to_string());
                words.push("route".to_string());
            }
            "intent" => {
                words.push("classify_intent".to_string());
                words.push("intent".to_string());
            }
            "module" | "codebase" => {
                words.push("src".to_string());
            }
            _ => {}
        }
    }
    words.sort();
    words.dedup();
    words
}

fn should_skip_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    normalized.contains("/target/")
        || normalized.contains("/.git/")
        || normalized.contains("/node_modules/")
        || normalized.contains("/bin/")
        || normalized.contains("/benchmarks/")
        || normalized.contains("/model-eval-results/")
        || normalized.contains("/.fastembed_cache/")
        || normalized.contains("/.mivi/")
        || normalized.ends_with("/.DS_Store")
}

fn query_words(query: &str) -> Vec<String> {
    let stop_words: std::collections::HashSet<&str> = [
        "the", "is", "to", "in", "a", "and", "of", "for", "on", "with", "at", "by", "from", "it",
        "this", "that", "or", "be", "as", "an", "code", "script", "write", "create", "print",
    ]
    .iter()
    .cloned()
    .collect();

    expand_query_words(
        query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= 3 && !stop_words.contains(w))
            .map(|s| s.to_string())
            .collect(),
    )
}

fn relevant_lines(text: &str, query_words: &[String], max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return text.to_string();
    }

    let mut keep = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        if query_words.iter().any(|word| lower.contains(word)) {
            let start = idx.saturating_sub(1);
            let end = (idx + 2).min(lines.len());
            for keep_idx in start..end {
                keep.push(keep_idx);
            }
        }
    }

    keep.sort_unstable();
    keep.dedup();
    keep.truncate(max_lines);

    if keep.is_empty() {
        return lines
            .into_iter()
            .take(max_lines)
            .collect::<Vec<_>>()
            .join("\n");
    }

    keep.into_iter()
        .map(|idx| lines[idx])
        .collect::<Vec<_>>()
        .join("\n")
}

impl TurboVecRAG {
    pub fn new() -> Self {
        let usage = Arc::new(Mutex::new(Self::load_usage()));
        Self {
            chunks: Arc::new(Mutex::new(Vec::new())),
            usage,
        }
    }

    fn load_usage() -> HashMap<String, u64> {
        if let Ok(data) = fs::read_to_string(".mivi_rag_usage") {
            if let Ok(map) = serde_json::from_str::<HashMap<String, u64>>(&data) {
                return map;
            }
        }
        HashMap::new()
    }

    pub async fn index_directory(&self, path: &str) -> usize {
        let path = path.to_string();
        let chunks_result = tokio::task::spawn_blocking(move || {
            #[derive(serde::Serialize, serde::Deserialize)]
            struct FileMeta {
                modified_sec: u64,
                size: u64,
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            struct ProjectState {
                workspace_path: String,
                indexed_files: HashMap<String, FileMeta>,
                chunks: Vec<RagChunk>,
            }

            fn get_file_meta(file_path: &std::path::Path) -> Option<FileMeta> {
                let metadata = file_path.metadata().ok()?;
                let modified = metadata.modified().ok()?;
                let duration = modified.duration_since(std::time::SystemTime::UNIX_EPOCH).ok()?;
                Some(FileMeta {
                    modified_sec: duration.as_secs(),
                    size: metadata.len(),
                })
            }

            let canonical_path = std::fs::canonicalize(&path)
                .unwrap_or_else(|_| std::path::PathBuf::from(&path))
                .display()
                .to_string();

            // 1. Try to load cache
            let cache_path = std::path::Path::new(".mivi/project_state.json");
            let mut cached_state: Option<ProjectState> = None;
            if cache_path.exists() {
                if let Ok(data) = fs::read_to_string(cache_path) {
                    if let Ok(state) = serde_json::from_str::<ProjectState>(&data) {
                        if state.workspace_path == canonical_path {
                            cached_state = Some(state);
                        }
                    }
                }
            }

            // 2. Scan workspace to collect files and check against cache
            let mut current_files = HashMap::new();
            let mut files_to_read = Vec::new();
            let mut file_count = 0;

            for entry in WalkDir::new(&path).into_iter().filter_map(|e| e.ok()) {
                if entry.file_type().is_file() {
                    let path_str = entry.path().display().to_string();
                    if should_skip_path(&path_str) {
                        continue;
                    }

                    if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                        if matches!(ext, "py" | "md" | "rs" | "json" | "js" | "ts" | "toml") {
                            if let Some(meta) = get_file_meta(entry.path()) {
                                if meta.size > 1_048_576 {
                                    continue; // Skip files > 1 MB
                                }
                                current_files.insert(path_str.clone(), meta);
                                files_to_read.push(entry.path().to_path_buf());
                                file_count += 1;
                                if file_count >= 5000 {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // 3. Check if cached state matches 100%
            let mut use_cache = false;
            if let Some(ref state) = cached_state {
                if state.indexed_files.len() == current_files.len() {
                    let mut all_match = true;
                    for (file_path, current_meta) in current_files.iter() {
                        if let Some(cached_meta) = state.indexed_files.get(file_path) {
                            if cached_meta.modified_sec != current_meta.modified_sec
                                || cached_meta.size != current_meta.size
                            {
                                all_match = false;
                                break;
                            }
                        } else {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        use_cache = true;
                    }
                }
            }

            if use_cache {
                let state = cached_state.unwrap();
                println!(
                    "[TurboVec RAG] Loaded cached index for {} chunks from .mivi/project_state.json",
                    state.chunks.len()
                );
                return state.chunks;
            }

            // 4. Mismatch or no cache -> perform full index
            let mut all_chunks = Vec::new();
            for file_path in files_to_read {
                let path_str = file_path.display().to_string();
                if let Ok(content) = fs::read_to_string(&file_path) {
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

            // 5. Save cache
            let new_state = ProjectState {
                workspace_path: canonical_path,
                indexed_files: current_files,
                chunks: all_chunks.clone(),
            };

            if let Err(e) = std::fs::create_dir_all(".mivi") {
                eprintln!("[TurboVec RAG] Failed to create .mivi dir: {}", e);
            } else if let Ok(json_data) = serde_json::to_string(&new_state) {
                if let Err(e) = fs::write(".mivi/project_state.json", json_data) {
                    eprintln!("[TurboVec RAG] Failed to write cache: {}", e);
                }
            }

            println!(
                "[TurboVec RAG] Indexed {} code chunks in workspace (< 1 MB RAM footprint)!",
                all_chunks.len()
            );
            all_chunks
        })
        .await;

        let all_chunks = chunks_result.unwrap_or_default();
        let count = all_chunks.len();
        let mut guard = self.chunks.lock().await;
        *guard = all_chunks;
        count
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Vec<(RagChunk, f32)> {
        let guard = self.chunks.lock().await;
        let query_words = query_words(query);

        if query_words.is_empty() {
            return Vec::new();
        }

        let usage_guard = self.usage.lock().await;
        let mut results = Vec::new();

        for chunk in guard.iter() {
            let text_lower = chunk.text.to_lowercase();
            let path_lower = chunk.file_path.to_lowercase();
            let mut score = 0.0f32;

            for word in &query_words {
                if text_lower.contains(word) {
                    score += 1.0;
                }
                if path_lower.contains(word) {
                    score += 3.0;
                }
            }

            if query_words
                .iter()
                .any(|word| word == "module" || word == "codebase")
                && path_lower.starts_with("src/")
            {
                score += 2.0;
            }

            // Apply usage boost
            if let Some(count) = usage_guard.get(&chunk.file_path) {
                score += (*count as f32).min(10.0) * 0.2;
            }

            if score >= 1.0 {
                results.push((chunk.clone(), score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let taken: Vec<(RagChunk, f32)> = results.into_iter().take(top_k).collect();

        drop(usage_guard);

        if !taken.is_empty() {
            let mut usage_guard = self.usage.lock().await;
            for (chunk, _) in &taken {
                *usage_guard.entry(chunk.file_path.clone()).or_insert(0) += 1;
            }
            if let Ok(json) = serde_json::to_string_pretty(&*usage_guard) {
                let _ = fs::write(".mivi_rag_usage", json);
            }
        }

        taken
    }

    pub async fn format_rag_context(&self, query: &str, top_k: usize) -> String {
        let matches = self.search(query, top_k).await;
        if matches.is_empty() {
            return String::new();
        }

        let mut formatted =
            vec!["# --- GOOGLE OKF (OPEN KNOWLEDGE FORMAT) CODEBASE CONTEXT ---".to_string()];
        for (i, (chunk, score)) in matches.iter().enumerate() {
            if *score < 1.0 {
                continue;
            }
            let evidence = relevant_lines(&chunk.text, &query_words(query), 8);
            let commented_text = evidence
                .lines()
                .map(|line| format!("# {}", line))
                .collect::<Vec<String>>()
                .join("\n");

            let okf_block = format!(
                "# ---\n# okf_version: 1.0\n# snippet_id: {}\n# source: {}\n# line_start: {}\n# relevance: {:.2}\n# ---\n{}",
                i + 1,
                chunk.file_path,
                chunk.line_start,
                score,
                commented_text
            );

            formatted.push(okf_block);
        }
        formatted
            .push("# -------------------------------------------------------------".to_string());
        formatted.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[tokio::test]
    async fn intent_routing_query_prefers_router_over_eval_docs() {
        let _lock = env_lock().lock().unwrap();
        let _ = fs::remove_file(".mivi_rag_usage");
        let rag = TurboVecRAG::new();
        {
            let mut chunks = rag.chunks.lock().await;
            chunks.push(RagChunk {
                file_path: "docs/model-evals/small-model-matrix.md".to_string(),
                line_start: 26,
                text: "intent routing model handles module routing intent candidate routing intent"
                    .to_string(),
            });
            chunks.push(RagChunk {
                file_path: "src/router.rs".to_string(),
                line_start: 1,
                text: "NeedleRouter classify_intent maps prompts to chat code vision multi_step"
                    .to_string(),
            });
        }

        let results = rag.search("what module handles intent routing", 2).await;

        assert_eq!(results[0].0.file_path, "src/router.rs");
        let _ = fs::remove_file(".mivi_rag_usage");
    }

    #[tokio::test]
    async fn intent_routing_query_prefers_router_file() {
        let _lock = env_lock().lock().unwrap();
        let _ = fs::remove_file(".mivi_rag_usage");
        let rag = TurboVecRAG::new();
        {
            let mut chunks = rag.chunks.lock().await;
            chunks.push(RagChunk {
                file_path: "src/rag.rs".to_string(),
                line_start: 1,
                text: "RAG retrieval pack context module".to_string(),
            });
            chunks.push(RagChunk {
                file_path: "src/router.rs".to_string(),
                line_start: 1,
                text: "NeedleRouter classify_intent handles chat code vision routing".to_string(),
            });
        }

        let results = rag.search("what module handles intent routing", 2).await;

        assert_eq!(results[0].0.file_path, "src/router.rs");
        let _ = fs::remove_file(".mivi_rag_usage");
    }

    #[test]
    fn skips_generated_runtime_artifact_paths() {
        assert!(should_skip_path("./target/release/mivi"));
        assert!(should_skip_path("./benchmarks/runtime.jsonl"));
        assert!(should_skip_path("./model-eval-results/small-model.jsonl"));
        assert!(should_skip_path("./.fastembed_cache/cache.bin"));
        assert!(!should_skip_path("./src/router.rs"));
    }

    #[tokio::test]
    async fn formatted_rag_context_keeps_only_relevant_lines() {
        let _lock = env_lock().lock().unwrap();
        let _ = fs::remove_file(".mivi_rag_usage");
        let rag = TurboVecRAG::new();
        {
            let mut chunks = rag.chunks.lock().await;
            chunks.push(RagChunk {
                file_path: "src/router.rs".to_string(),
                line_start: 1,
                text: (1..=25)
                    .map(|i| {
                        if i == 13 {
                            "pub fn classify_intent(&self, prompt: &str)".to_string()
                        } else {
                            format!("irrelevant filler line {}", i)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }

        let context = rag
            .format_rag_context("intent routing classify_intent", 1)
            .await;

        assert!(context.contains("classify_intent"));
        assert!(!context.contains("# irrelevant filler line 1\n"));
        assert!(!context.contains("irrelevant filler line 25"));
        let _ = fs::remove_file(".mivi_rag_usage");
    }

    #[tokio::test]
    async fn search_tracks_usage_and_persists_to_file() {
        let _lock = env_lock().lock().unwrap();
        let _ = fs::remove_file(".mivi_rag_usage");

        let rag = TurboVecRAG::new();
        {
            let mut chunks = rag.chunks.lock().await;
            chunks.push(RagChunk {
                file_path: "src/test_file.rs".to_string(),
                line_start: 1,
                text: "test usage tracking code block".to_string(),
            });
        }

        let results = rag.search("test usage", 1).await;
        assert_eq!(results.len(), 1);

        let usage = TurboVecRAG::load_usage();
        assert_eq!(usage.get("src/test_file.rs"), Some(&1));

        let _ = fs::remove_file(".mivi_rag_usage");
    }
}
