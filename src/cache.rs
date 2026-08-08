use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct SemanticCache {
    cache: Arc<Mutex<HashMap<String, (String, std::time::Instant)>>>,
}

impl SemanticCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn jaccard_similarity(s1: &str, s2: &str) -> f32 {
        let set1: HashSet<String> = s1
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let set2: HashSet<String> = s2
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

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
        let mut guard = self.cache.lock().await;

        if let Some((val, last_used)) = guard.get_mut(q_clean) {
            println!("[SemanticCache] EXACT CACHE HIT!");
            *last_used = std::time::Instant::now();
            return Some(val.clone());
        }

        let mut best_score = 0.0f32;
        let mut best_result: Option<String> = None;
        let mut best_key: Option<String> = None;

        for (k, (v, _)) in guard.iter() {
            let score = Self::jaccard_similarity(q_clean, k);
            if score > best_score {
                best_score = score;
                best_result = Some(v.clone());
                best_key = Some(k.clone());
            }
        }

        if best_score >= 0.85 {
            println!(
                "[SemanticCache] SEMANTIC CACHE HIT! Similarity score: {:.4}",
                best_score
            );
            if let Some(ref k) = best_key {
                if let Some((_, last_used)) = guard.get_mut(k) {
                    *last_used = std::time::Instant::now();
                }
            }
            best_result
        } else {
            None
        }
    }

    pub async fn put(&self, query: &str, result: &str) {
        let mut guard = self.cache.lock().await;
        let q_clean = query.trim().to_string();

        if guard.len() >= 512 && !guard.contains_key(&q_clean) {
            // Evict LRU: find key with oldest Instant
            let mut oldest_key: Option<String> = None;
            let mut oldest_time = std::time::Instant::now();

            for (k, (_, last_used)) in guard.iter() {
                if *last_used < oldest_time {
                    oldest_time = *last_used;
                    oldest_key = Some(k.clone());
                }
            }

            if let Some(k) = oldest_key {
                guard.remove(&k);
            }
        }

        guard.insert(q_clean, (result.to_string(), std::time::Instant::now()));
    }
}
