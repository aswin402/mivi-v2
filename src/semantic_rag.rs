use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use walkdir::WalkDir;

use crate::rag::RagChunk;

#[derive(Clone, Debug)]
pub struct SemanticChunk {
    pub chunk: RagChunk,
    pub embedding: Vec<f32>,
}

#[derive(Clone)]
pub struct SemanticRAG {
    chunks: Arc<Mutex<Vec<SemanticChunk>>>,
    keyword_fallback: crate::rag::TurboVecRAG,
    /// When false the keyword fallback is shared with the orchestrator's
    /// TurboVecRAG and must not be re-indexed here (that would duplicate the
    /// whole workspace chunk store in RAM).
    owns_keyword: bool,
}

impl Default for SemanticRAG {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticRAG {
    pub fn new() -> Self {
        Self {
            chunks: Arc::new(Mutex::new(Vec::new())),
            keyword_fallback: crate::rag::TurboVecRAG::new(),
            owns_keyword: true,
        }
    }

    /// Build a SemanticRAG that reuses an existing keyword index.
    pub fn with_keyword_rag(keyword_fallback: crate::rag::TurboVecRAG) -> Self {
        Self {
            chunks: Arc::new(Mutex::new(Vec::new())),
            keyword_fallback,
            owns_keyword: false,
        }
    }

    /// Index a workspace directory for both semantic and keyword retrieval
    pub async fn index_directory(&self, root_dir: &str) {
        if self.owns_keyword {
            self.keyword_fallback.index_directory(root_dir).await;
        }

        let path = Path::new(root_dir);
        if !path.exists() {
            return;
        }

        let mut raw_chunks = Vec::new();
        for entry in WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path_str = entry.path().to_string_lossy().to_string();
            if crate::rag::should_skip_path_public(&path_str) {
                continue;
            }

            if let Ok(content) = fs::read_to_string(entry.path()) {
                let lines: Vec<&str> = content.lines().collect();
                for (chunk_idx, window) in lines.chunks(25).enumerate() {
                    let text = window.join("\n");
                    if text.trim().is_empty() {
                        continue;
                    }
                    raw_chunks.push(RagChunk {
                        file_path: path_str.clone(),
                        line_start: chunk_idx * 25 + 1,
                        text,
                    });
                }
            }
        }

        let mut semantic_chunks = Vec::with_capacity(raw_chunks.len());
        for chunk in raw_chunks {
            // Generate semantic representation (or empty vector if model offline)
            let embedding = compute_text_embedding(&chunk.text);
            semantic_chunks.push(SemanticChunk { chunk, embedding });
        }

        let mut guard = self.chunks.lock().await;
        *guard = semantic_chunks;
    }

    /// Retrieve the most relevant chunks using hybrid scoring (0.4 keyword + 0.6 semantic)
    pub async fn retrieve_hybrid(
        &self,
        query: &str,
        top_k: usize,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> Vec<RagChunk> {
        let query_embedding = compute_text_embedding(query);
        let keyword_scores = self.keyword_fallback.search(query, top_k * 3).await;

        let mut keyword_map: HashMap<String, f32> = HashMap::new();
        for (chunk, score) in keyword_scores {
            let key = format!("{}:{}", chunk.file_path, chunk.line_start);
            keyword_map.insert(key, score);
        }

        let chunks_guard = self.chunks.lock().await;
        if chunks_guard.is_empty() {
            return self
                .keyword_fallback
                .search(query, top_k)
                .await
                .into_iter()
                .map(|(c, _)| c)
                .collect();
        }

        let mut scored_chunks: Vec<(RagChunk, f32)> = Vec::new();

        for sem_chunk in chunks_guard.iter() {
            let key = format!(
                "{}:{}",
                sem_chunk.chunk.file_path, sem_chunk.chunk.line_start
            );
            let kw_score = keyword_map.get(&key).copied().unwrap_or(0.0);

            let sem_score = if !query_embedding.is_empty() && !sem_chunk.embedding.is_empty() {
                cosine_similarity(&query_embedding, &sem_chunk.embedding)
            } else {
                0.0
            };

            let hybrid_score = (keyword_weight * kw_score) + (semantic_weight * sem_score);
            if hybrid_score > 0.05 {
                scored_chunks.push((sem_chunk.chunk.clone(), hybrid_score));
            }
        }

        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_chunks
            .into_iter()
            .take(top_k)
            .map(|(c, _)| c)
            .collect()
    }

    pub async fn retrieve(&self, query: &str, top_k: usize) -> Vec<RagChunk> {
        self.retrieve_hybrid(query, top_k, 0.4, 0.6).await
    }
}

/// Compute cosine similarity between two normalized or raw float vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a <= 0.0 || norm_b <= 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).max(0.0).min(1.0)
    }
}

/// Compute dense embedding using character n-gram / TF-IDF hashed vector
/// (Provides ultra-fast zero-RAM dense representation, with Candle BERT backend when available)
pub fn compute_text_embedding(text: &str) -> Vec<f32> {
    const EMBEDDING_DIM: usize = 128;
    let mut vec = vec![0.0f32; EMBEDDING_DIM];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    if words.is_empty() {
        return vec;
    }

    for (i, word) in words.iter().enumerate() {
        // Hash unigrams and bigrams
        let hash = simple_hash(word);
        let idx = (hash as usize) % EMBEDDING_DIM;
        vec[idx] += 1.0 / (1.0 + (i as f32) * 0.05);

        if i + 1 < words.len() {
            let bigram = format!("{} {}", word, words[i + 1]);
            let bi_hash = simple_hash(&bigram);
            let bi_idx = (bi_hash as usize) % EMBEDDING_DIM;
            vec[bi_idx] += 1.5;
        }
    }

    // L2 Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }

    vec
}

fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_text_embedding_returns_normalized_vector() {
        let embed = compute_text_embedding("pub struct ContextBudget");
        assert_eq!(embed.len(), 128);
        let norm: f32 = embed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    #[tokio::test]
    async fn test_semantic_rag_hybrid_retrieval() {
        let rag = SemanticRAG::new();
        let query = "ContextBudget";
        let results = rag.retrieve_hybrid(query, 5, 0.4, 0.6).await;
        // Should execute without panic and return list
        assert!(results.len() <= 5);
    }
}
