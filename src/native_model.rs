//! Native-engine model loading: GGUF → candle quantized model, tokenizer
//! discovery, grammar-state loading, and the LRU model cache.
//!
//! Extracted from `native_brain.rs` (decomposition).
#[cfg(feature = "native")]
use candle_core::{Device, Tensor};
#[cfg(feature = "native")]
use candle_transformers::generation::{LogitsProcessor, Sampling};
#[cfg(feature = "native")]
use std::collections::HashMap;
#[cfg(feature = "native")]
use std::fs::File;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};
#[cfg(feature = "native")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "native")]
use tokenizers::Tokenizer;
#[cfg(feature = "native")]
use tracing::{error, info};

#[cfg(feature = "native")]
pub(crate) fn find_tokenizer_path(model_path: &Path) -> Option<PathBuf> {
    // Try using model catalog first
    if let Ok(catalog) = crate::model_catalog::ModelCatalog::load_default() {
        let abs_model_path = model_path
            .canonicalize()
            .unwrap_or_else(|_| model_path.to_path_buf());
        for entry in catalog.models {
            let entry_path = Path::new(&entry.path);
            let abs_entry_path = entry_path
                .canonicalize()
                .unwrap_or_else(|_| entry_path.to_path_buf());
            if abs_entry_path == abs_model_path {
                if let Some(tok_path) = entry.tokenizer_path {
                    let p = Path::new(&tok_path);
                    if p.exists() {
                        return Some(p.to_path_buf());
                    }
                }
            }
        }
    }

    // Fallback: search in the same directory as the model
    if let Some(parent) = model_path.parent() {
        // Match standard names first
        let name_str = model_path.to_string_lossy().to_lowercase();
        if name_str.contains("qwen") {
            let qwen_tok = parent.join("qwen2.5-0.5b-tokenizer.json");
            if qwen_tok.exists() {
                return Some(qwen_tok);
            }
        } else if name_str.contains("llama") {
            let llama_tok = parent.join("Llama-3.2-1B-Instruct-tokenizer.json");
            if llama_tok.exists() {
                return Some(llama_tok);
            }
        }

        // Generic fallback
        let tok_json = parent.join("tokenizer.json");
        if tok_json.exists() {
            return Some(tok_json);
        }
    }
    None
}

#[cfg(feature = "native")]
pub enum QuantizedModel {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    // Vendored copy with f16 embedding dequantization (see src/vendor/).
    Qwen2(crate::vendor::quantized_qwen2::ModelWeights),
    Phi3(candle_transformers::models::quantized_phi3::ModelWeights),
}

#[cfg(feature = "native")]
impl QuantizedModel {
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        arch: &str,
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self, String> {
        match arch {
            "qwen2" => {
                let model =
                    crate::vendor::quantized_qwen2::ModelWeights::from_gguf(ct, reader, device)
                        .map_err(|e| format!("failed to load quantized qwen2: {}", e))?;
                Ok(Self::Qwen2(model))
            }
            "phi3" => {
                let model = candle_transformers::models::quantized_phi3::ModelWeights::from_gguf(
                    false, ct, reader, device,
                )
                .map_err(|e| format!("failed to load quantized phi3: {}", e))?;
                Ok(Self::Phi3(model))
            }
            _ => {
                let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    ct, reader, device,
                )
                .map_err(|e| format!("failed to load quantized llama: {}", e))?;
                Ok(Self::Llama(model))
            }
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor, String> {
        match self {
            Self::Llama(m) => m
                .forward(x, index_pos)
                .map_err(|e| format!("Llama forward error: {}", e)),
            Self::Qwen2(m) => m
                .forward(x, index_pos)
                .map_err(|e| format!("Qwen2 forward error: {}", e)),
            Self::Phi3(m) => m
                .forward(x, index_pos)
                .map_err(|e| format!("Phi3 forward error: {}", e)),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            Self::Llama(m) => m.clear_kv_cache(),
            Self::Qwen2(m) => m.clear_kv_cache(),
            // Phi3 exposes no clear in candle, but its `forward_attn` resets the
            // KV cache whenever index_pos == 0 and every MIVI query starts its
            // prefill at index_pos 0, so the cache is effectively cleared per
            // request. Do not load Phi3 outside this invariant.
            Self::Phi3(_) => (),
        }
    }

    /// Retain only the first `pos` KV entries (shared-prefix reuse).
    /// Returns false for variants without truncation support.
    pub fn truncate_kv_cache(&mut self, pos: usize) -> bool {
        match self {
            Self::Qwen2(m) => m.truncate_kv_cache(pos),
            _ => false,
        }
    }
}

#[cfg(feature = "native")]
pub struct LoadedModel {
    pub model: Mutex<QuantizedModel>,
    pub tokenizer: Tokenizer,
    pub vocab: Vec<String>,
    /// Token ids of the prompt whose KV entries are currently resident in
    /// `model` (the shared-prefix cache). Must only be read/written while
    /// holding the `model` lock so it stays consistent with the KV state.
    pub cached_prefix: Mutex<Vec<u32>>,
}

#[cfg(feature = "native")]
pub(crate) fn load_grammar_state(
    grammar_path: &Option<String>,
) -> Option<schoolmarm::GrammarState> {
    let path = grammar_path.as_ref()?;
    let grammar_str = std::fs::read_to_string(path).ok()?;

    // Clean up comment lines (starting with #) and inline comments (preceded by a space)
    let mut cleaned_str = String::new();
    for line in grammar_str.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(idx) = line.find(" #") {
            cleaned_str.push_str(&line[..idx]);
        } else {
            cleaned_str.push_str(line);
        }
        cleaned_str.push('\n');
    }

    let grammar = schoolmarm::Grammar::new(cleaned_str.trim()).ok()?;
    schoolmarm::GrammarState::new(grammar).ok()
}

#[cfg(feature = "native")]
/// LRU cache of loaded models. Without eviction every distinct GGUF path
/// (reasoner, coder, vision) stayed resident forever; without recency
/// tracking the wrong model would be dropped.
pub struct ModelCache {
    pub(crate) map: HashMap<PathBuf, Arc<LoadedModel>>,
    /// Front = least recently used.
    order: std::collections::VecDeque<PathBuf>,
    pub(crate) max_entries: usize,
}

#[cfg(feature = "native")]
impl ModelCache {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            max_entries: max_entries.max(1),
        }
    }

    pub(crate) fn get(&mut self, path: &Path) -> Option<Arc<LoadedModel>> {
        let loaded = self.map.get(path)?.clone();
        if let Some(pos) = self.order.iter().position(|p| p == path) {
            self.order.remove(pos);
        }
        self.order.push_back(path.to_path_buf());
        Some(loaded)
    }

    pub(crate) fn insert(&mut self, path: PathBuf, loaded: Arc<LoadedModel>) {
        if self.map.contains_key(&path) {
            return;
        }
        self.map.insert(path.clone(), loaded);
        self.order.push_back(path);
        while self.map.len() > self.max_entries {
            let Some(lru) = self.order.pop_front() else {
                break;
            };
            if self.map.remove(&lru).is_some() {
                info!("[NativeBrain] Evicted cached model {:?} (LRU)", lru);
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// Length of the common token prefix of two prompts.
pub(crate) fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(feature = "native")]
pub(crate) fn model_cache_max_entries() -> usize {
    let ultra_low = std::env::var("MIVI_ULTRA_LOW_RAM")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    let default = if ultra_low { 1 } else { 2 };
    std::env::var("MIVI_MODEL_CACHE_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}
