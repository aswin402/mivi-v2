use crate::model_catalog::{ModelCatalog, ModelRole};
use shimmytok::Tokenizer;
use std::env;
use std::path::Path;
use std::sync::OnceLock;
use tracing::{info, warn};

static GLOBAL_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

/// Initialize the global tokenizer from a GGUF file path.
/// Returns true if successfully initialized, false if already initialized or failed.
pub fn init_global_tokenizer<P: AsRef<Path>>(path: P) -> bool {
    let path_ref = path.as_ref();
    if GLOBAL_TOKENIZER.get().is_some() {
        return false;
    }
    if !path_ref.exists() {
        warn!(
            "[Tokenizer] Model GGUF file does not exist at {}",
            path_ref.display()
        );
        return false;
    }
    info!(
        "[Tokenizer] Initializing exact tokenizer from GGUF: {}",
        path_ref.display()
    );
    match Tokenizer::from_gguf_file(path_ref) {
        Ok(t) => {
            let _ = GLOBAL_TOKENIZER.set(t);
            info!("[Tokenizer] Exact tokenizer successfully loaded!");
            true
        }
        Err(err) => {
            warn!("[Tokenizer] Failed to load tokenizer from GGUF: {}", err);
            false
        }
    }
}

/// Automatically resolve the model GGUF path and initialize the tokenizer.
pub fn init_from_env() {
    let tokenizer_model = env::var("MIVI_TOKENIZER_MODEL").ok();
    let reasoner_model = env::var("MIVI_REASONER_MODEL").ok();
    let coder_model = env::var("MIVI_CODER_MODEL").ok();

    // Priority 1: MIVI_TOKENIZER_MODEL env var
    if let Some(path) = tokenizer_model.filter(|p| !p.trim().is_empty()) {
        if init_global_tokenizer(&path) {
            return;
        }
    }

    // Priority 2: MIVI_REASONER_MODEL env var
    if let Some(path) = reasoner_model.filter(|p| !p.trim().is_empty()) {
        if init_global_tokenizer(&path) {
            return;
        }
    }

    // Priority 3: MIVI_CODER_MODEL env var
    if let Some(path) = coder_model.filter(|p| !p.trim().is_empty()) {
        if init_global_tokenizer(&path) {
            return;
        }
    }

    // Priority 4: Default enabled reasoner path from catalog
    if let Ok(catalog) = ModelCatalog::load_default() {
        if let Some(path) = catalog.default_enabled_path(ModelRole::Reasoner) {
            if init_global_tokenizer(path) {
                return;
            }
        }
    }
}

/// Count tokens using the exact tokenizer if initialized, falling back to a simple estimator.
pub fn count_tokens(text: &str) -> u32 {
    if let Some(t) = GLOBAL_TOKENIZER.get() {
        match t.encode(text, false) {
            Ok(tokens) => tokens.len() as u32,
            Err(err) => {
                warn!(
                    "[Tokenizer] Exact encoding error: {}. Falling back to estimator.",
                    err
                );
                cheap_count_tokens(text)
            }
        }
    } else {
        cheap_count_tokens(text)
    }
}

/// A fast and cheap character-based token estimator fallback.
fn cheap_count_tokens(text: &str) -> u32 {
    let mut tokens = 0_u32;
    let mut in_word = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            if !in_word {
                tokens = tokens.saturating_add(1);
                in_word = true;
            }
        } else {
            in_word = false;
            if !ch.is_whitespace() {
                tokens = tokens.saturating_add(1);
            }
        }
    }
    tokens
}
