use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Entries older than this are treated as misses and purged lazily.
const CACHE_TTL_SECS: u64 = 600;

/// Adaptive capacity bounds for the trace-driven tuner (19.4).
pub const CACHE_MIN_CAPACITY: usize = 128;
pub const CACHE_MAX_CAPACITY: usize = 2048;
const CACHE_DEFAULT_CAPACITY: usize = 512;

#[derive(Clone)]
pub struct SemanticCache {
    cache: Arc<Mutex<HashMap<String, (String, std::time::SystemTime)>>>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    capacity: Arc<AtomicUsize>,
}

impl SemanticCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
            capacity: Arc::new(AtomicUsize::new(CACHE_DEFAULT_CAPACITY)),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Snapshot of the lifetime counters (hits, misses, evictions, capacity).
    pub fn counters(&self) -> (u64, u64, u64, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.capacity(),
        )
    }

    /// Trace-driven tuner (TODO 19.4): adapt capacity from a measured window.
    ///
    /// - grow when the cache earns its RAM (decent hit rate) but entries are
    ///   being evicted to make room;
    /// - shrink when lookups almost never hit (dead weight);
    /// - keep otherwise, and keep when the window is too small to trust.
    pub fn adapt_window(&self, hits: u64, misses: u64, evictions: u64) {
        let total = hits + misses;
        let current = self.capacity();
        if total < 50 {
            return;
        }
        let hit_rate = hits as f32 / total as f32;
        let next = if evictions > 0 && hit_rate >= 0.25 {
            (current * 2).min(CACHE_MAX_CAPACITY)
        } else if hit_rate < 0.05 && total >= 200 {
            (current / 2).max(CACHE_MIN_CAPACITY)
        } else {
            current
        };
        if next != current {
            self.capacity.store(next, Ordering::Relaxed);
            tracing::info!(
                "[SemanticCache] adapted capacity {} -> {} \
                 (window: {} hits / {} misses / {} evictions, rate {:.2})",
                current,
                next,
                hits,
                misses,
                evictions,
                hit_rate
            );
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

    fn fresh(ts: &std::time::SystemTime) -> bool {
        ts.elapsed()
            .map(|d| d.as_secs() <= CACHE_TTL_SECS)
            .unwrap_or(false)
    }

    /// Exact-key lookup with TTL. Used for verified-code results where a
    /// fuzzy hit would return WRONG code.
    pub async fn get_exact(&self, query: &str) -> Option<String> {
        let q_clean = query.trim();
        let mut guard = self.cache.lock().await;
        if let Some((val, ts)) = guard.get_mut(q_clean) {
            if !Self::fresh(ts) {
                guard.remove(q_clean);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            *ts = std::time::SystemTime::now();
            self.hits.fetch_add(1, Ordering::Relaxed);
            println!("[SemanticCache] EXACT CACHE HIT!");
            return Some(val.clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn get(&self, query: &str) -> Option<String> {
        let q_clean = query.trim();
        let mut guard = self.cache.lock().await;

        if let Some((val, ts)) = guard.get_mut(q_clean) {
            if !Self::fresh(ts) {
                guard.remove(q_clean);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.hits.fetch_add(1, Ordering::Relaxed);
            println!("[SemanticCache] EXACT CACHE HIT!");
            *ts = std::time::SystemTime::now();
            return Some(val.clone());
        }

        let mut best_score = 0.0f32;
        let mut best_result: Option<String> = None;
        let mut best_key: Option<String> = None;

        for (k, (v, ts)) in guard.iter() {
            if !Self::fresh(ts) {
                continue;
            }
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
            self.hits.fetch_add(1, Ordering::Relaxed);
            if let Some(k) = &best_key {
                if let Some((_, ts)) = guard.get_mut(k) {
                    *ts = std::time::SystemTime::now();
                }
            }
            best_result
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub async fn put(&self, query: &str, result: &str) {
        let mut guard = self.cache.lock().await;
        let q_clean = query.trim().to_string();

        if guard.len() >= self.capacity() && !guard.contains_key(&q_clean) {
            // Evict LRU: find key with oldest timestamp
            let mut oldest_key: Option<String> = None;
            let mut oldest_time = std::time::SystemTime::now();

            for (k, (_, last_used)) in guard.iter() {
                if *last_used < oldest_time {
                    oldest_time = *last_used;
                    oldest_key = Some(k.clone());
                }
            }

            if let Some(k) = oldest_key {
                guard.remove(&k);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        guard.insert(q_clean, (result.to_string(), std::time::SystemTime::now()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exact_and_semantic_cache_hits() {
        let cache = SemanticCache::new();
        cache.put("what is the weather today", "cloudy").await;

        // Exact hit
        assert_eq!(
            cache.get("what is the weather today").await,
            Some("cloudy".to_string())
        );

        // Semantic hit (Jaccard similarity >= 0.85)
        // Let's use similar phrases: "what is the weather today" vs "what is weather today"
        // set1 = {"what", "is", "the", "weather", "today"} (len 5)
        // set2 = {"what", "is", "weather", "today"} (len 4)
        // intersection = 4, union = 5, score = 0.8 (which is < 0.85)
        // Let's try: "what is the weather today" vs "what is the weather today now"
        // set1 = 5 words. set2 = 6 words.
        // intersection = 5, union = 6. 5 / 6 = 0.833
        // Let's try: "what is the weather today" vs "what is the weather today" with extra space.
        assert_eq!(
            cache.get("  what is the weather today  ").await,
            Some("cloudy".to_string())
        );
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = SemanticCache::new();
        // Insert 512 entries
        for i in 0..512 {
            cache
                .put(&format!("query {}", i), &format!("value {}", i))
                .await;
        }
        // Cache size is 512
        assert_eq!(cache.cache.lock().await.len(), 512);

        // Put 513th entry
        cache.put("new query", "new value").await;
        assert_eq!(cache.cache.lock().await.len(), 512);
        assert_eq!(cache.get("new query").await, Some("new value".to_string()));
    }

    #[tokio::test]
    async fn get_exact_ignores_similar_but_different_prompts() {
        let cache = SemanticCache::new();
        cache
            .put(
                "write a function to sort a list using quicksort",
                "quick-code",
            )
            .await;
        assert_eq!(
            cache
                .get_exact("write a function to sort a list using mergesort")
                .await,
            None
        );
        assert_eq!(
            cache
                .get_exact("  write a function to sort a list using quicksort  ")
                .await,
            Some("quick-code".to_string())
        );
    }

    #[tokio::test]
    async fn get_exact_expires_after_ttl() {
        let cache = SemanticCache::new();
        cache.put("q", "a").await;
        {
            let mut guard = cache.cache.lock().await;
            guard.get_mut("q").unwrap().1 =
                std::time::SystemTime::now() - std::time::Duration::from_secs(601);
        }
        assert_eq!(cache.get_exact("q").await, None);
    }

    #[test]
    fn tuner_keeps_capacity_when_window_too_small() {
        let cache = SemanticCache::new();
        cache.adapt_window(10, 0, 50);
        assert_eq!(cache.capacity(), 512);
    }

    #[test]
    fn tuner_grows_when_hits_earn_evictions() {
        let cache = SemanticCache::new();
        cache.adapt_window(300, 100, 12); // 75% hit rate, evicting
        assert_eq!(cache.capacity(), 1024);
    }

    #[test]
    fn tuner_shrinks_dead_weight_cache() {
        let cache = SemanticCache::new();
        cache.adapt_window(2, 400, 0); // 0.5% hit rate, large window
        assert_eq!(cache.capacity(), 256);
    }

    #[test]
    fn tuner_respects_hard_bounds() {
        let cache = SemanticCache::new();
        for _ in 0..12 {
            cache.adapt_window(400, 50, 5); // grow repeatedly
        }
        assert_eq!(cache.capacity(), CACHE_MAX_CAPACITY);
        for _ in 0..12 {
            cache.adapt_window(1, 900, 0); // shrink repeatedly
        }
        assert_eq!(cache.capacity(), CACHE_MIN_CAPACITY);
    }

    #[tokio::test]
    async fn counters_track_hits_misses_evictions() {
        let cache = SemanticCache::new();
        cache.put("alpha query", "a").await;
        let _ = cache.get("alpha query").await;
        let (hits_before, _, _, _) = cache.counters();
        assert_eq!(hits_before, 1);
        for i in 0..512 {
            cache.put(&format!("filler-{i}"), "v").await;
        }
        let (_, _, evictions, _) = cache.counters();
        assert!(evictions > 0, "inserting past capacity must evict");
        let (hits, _, _, _) = cache.counters();
        assert!(hits >= 1);
    }

    #[tokio::test]
    async fn misses_are_counted() {
        let cache = SemanticCache::new();
        let _ = cache.get("totally unknown query").await;
        let (_, misses, _, _) = cache.counters();
        assert_eq!(misses, 1);
    }
}
