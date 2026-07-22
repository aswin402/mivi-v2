use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct SemanticCache {
    cache: Arc<Mutex<HashMap<String, String>>>,
}

impl SemanticCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn jaccard_similarity(s1: &str, s2: &str) -> f32 {
        let set1: HashSet<String> = s1.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
        let set2: HashSet<String> = s2.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();

        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    pub async fn get(&self, query: &str) -> Option<String> {
        let q_clean = query.trim();
        let guard = self.cache.lock().await;

        if let Some(val) = guard.get(q_clean) {
            println!("[SemanticCache] EXACT CACHE HIT!");
            return Some(val.clone());
        }

        let mut best_score = 0.0f32;
        let mut best_result: Option<String> = None;

        for (k, v) in guard.iter() {
            let score = Self::jaccard_similarity(q_clean, k);
            if score > best_score {
                best_score = score;
                best_result = Some(v.clone());
            }
        }

        if best_score >= 0.85 {
            println!("[SemanticCache] SEMANTIC CACHE HIT! Similarity score: {:.4}", best_score);
            best_result
        } else {
            None
        }
    }

    pub async fn put(&self, query: &str, result: &str) {
        let mut guard = self.cache.lock().await;
        guard.insert(query.trim().to_string(), result.to_string());
    }
}

