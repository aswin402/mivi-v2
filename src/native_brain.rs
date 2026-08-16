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
fn find_tokenizer_path(model_path: &Path) -> Option<PathBuf> {
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
}

#[cfg(feature = "native")]
pub struct LoadedModel {
    pub model: Mutex<QuantizedModel>,
    pub tokenizer: Tokenizer,
    pub vocab: Vec<String>,
}

#[cfg(feature = "native")]
fn load_grammar_state(grammar_path: &Option<String>) -> Option<schoolmarm::GrammarState> {
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
    map: HashMap<PathBuf, Arc<LoadedModel>>,
    /// Front = least recently used.
    order: std::collections::VecDeque<PathBuf>,
    max_entries: usize,
}

#[cfg(feature = "native")]
impl ModelCache {
    fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            max_entries: max_entries.max(1),
        }
    }

    fn get(&mut self, path: &Path) -> Option<Arc<LoadedModel>> {
        let loaded = self.map.get(path)?.clone();
        if let Some(pos) = self.order.iter().position(|p| p == path) {
            self.order.remove(pos);
        }
        self.order.push_back(path.to_path_buf());
        Some(loaded)
    }

    fn insert(&mut self, path: PathBuf, loaded: Arc<LoadedModel>) {
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
}

#[cfg(feature = "native")]
fn model_cache_max_entries() -> usize {
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

#[cfg(feature = "native")]
#[derive(Clone)]
pub struct NativeBrain {
    pub models: Arc<Mutex<ModelCache>>,
}

#[cfg(feature = "native")]
impl NativeBrain {
    pub fn new() -> Self {
        let mut features = Vec::new();
        if cfg!(target_feature = "avx2") {
            features.push("AVX2");
        }
        if cfg!(target_feature = "fma") {
            features.push("FMA");
        }
        if cfg!(target_feature = "f16c") {
            features.push("F16C");
        }
        if cfg!(target_feature = "neon") {
            features.push("NEON");
        }
        info!(
            "[NativeBrain] Native CPU vectorization active: {:?}",
            features
        );

        Self {
            models: Arc::new(Mutex::new(ModelCache::new(model_cache_max_entries()))),
        }
    }

    pub fn get_or_load(&self, model_path: &Path) -> Result<Arc<LoadedModel>, String> {
        let canonical_path = model_path.to_path_buf();

        {
            let mut cache = self.models.lock().unwrap();
            if let Some(loaded) = cache.get(&canonical_path) {
                return Ok(loaded);
            }
        }
        // The GGUF load below runs WITHOUT the cache lock so inference on an
        // already-cached model is never blocked behind a multi-second load.

        info!("[NativeBrain] Loading model GGUF: {:?}", model_path);
        let tokenizer_path = find_tokenizer_path(model_path)
            .ok_or_else(|| format!("Could not find tokenizer.json for model: {:?}", model_path))?;
        info!("[NativeBrain] Using tokenizer: {:?}", tokenizer_path);

        let device = Device::Cpu;
        let mut file = File::open(model_path).map_err(|e| format!("failed to open GGUF: {}", e))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| format!("failed to read GGUF content: {}", e))?;

        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "llama".to_string());
        info!("[NativeBrain] GGUF Model Architecture detected: {}", arch);

        let model = QuantizedModel::from_gguf(&arch, content, &mut file, &device)?;

        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            unsafe {
                libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
            }
            info!("[NativeBrain] Advised kernel to drop page cache for GGUF model file (POSIX_FADV_DONTNEED)");
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("failed to load tokenizer: {}", e))?;

        let vocab_size = tokenizer.get_vocab_size(true);
        let mut vocab = vec![String::new(); vocab_size];
        for id in 0..vocab_size {
            if let Some(token) = tokenizer.id_to_token(id as u32) {
                vocab[id] = token;
            }
        }

        let loaded = Arc::new(LoadedModel {
            model: Mutex::new(model),
            tokenizer,
            vocab,
        });

        {
            let mut cache = self.models.lock().unwrap();
            if let Some(existing) = cache.get(&canonical_path) {
                // Another thread loaded the same model while we held no lock.
                return Ok(existing);
            }
            cache.insert(canonical_path, loaded.clone());
        }

        // The GGUF load reads the file into heap buffers that are freed once
        // the quantized weights are built; glibc keeps those pages resident.
        // Return them to the OS so steady-state RSS matches the model size.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::malloc_trim(0);
        }

        Ok(loaded)
    }

    pub fn query(
        &self,
        model_path: &Path,
        prompt: &str,
        system_prompt: &str,
        temp_str: &str,
        max_tokens: usize,
        grammar_path: Option<String>,
    ) -> Result<String, String> {
        let loaded = self.get_or_load(model_path)?;
        let mut model = loaded.model.lock().unwrap();
        model.clear_kv_cache();
        let tokenizer = &loaded.tokenizer;

        let t = crate::server::active_chat_template();

        // Detect if the prompt is already formatted with chat template tokens.
        let formatted_prompt = if !t.system_prefix.is_empty()
            && prompt.trim_start().starts_with(t.system_prefix.trim())
            && prompt.contains(t.assistant_start.trim())
        {
            prompt.to_string()
        } else {
            let (extracted_system, extracted_user) = crate::brain::split_prompt_system_user(prompt);
            let final_system = if extracted_system.is_empty() {
                system_prompt.to_string()
            } else {
                extracted_system
            };
            let final_user = if extracted_user.is_empty() {
                prompt.to_string()
            } else {
                let trimmed = extracted_user.trim();
                if let Some(stripped) = trimmed.strip_prefix("Current user request:") {
                    stripped.trim().to_string()
                } else {
                    trimmed.to_string()
                }
            };
            format!(
                "{}{}{}{}{}{}{}",
                t.system_prefix,
                final_system,
                t.system_suffix,
                t.user_prefix,
                final_user,
                t.user_suffix,
                t.assistant_start
            )
        };

        let tokens = tokenizer
            .encode(formatted_prompt, true)
            .map_err(|e| format!("Tokenization error: {}", e))?;
        let token_ids = tokens.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err("Tokenized prompt is empty".to_string());
        }

        let device = Device::Cpu;
        let temp = temp_str.parse::<f64>().unwrap_or(0.2);

        let sampling = if temp <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopP {
                p: 0.9,
                temperature: temp,
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);

        let mut index_pos = 0;
        // Incremental detokenizer: decoding only the delta per token keeps the
        // stop-word check O(1) instead of re-decoding the whole sequence (O(n^2)).
        let mut decode_stream = tokenizer.decode_stream(true);
        let mut full_text = String::new();

        // Prefill
        let input = Tensor::new(&token_ids[..], &device)
            .map_err(|e| format!("Tensor creation error: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

        let logits = model
            .forward(&input, index_pos)
            .map_err(|e| format!("Forward pass error: {}", e))?;

        index_pos += token_ids.len();

        let mut grammar_state = load_grammar_state(&grammar_path);
        let vocab_refs: Vec<&str> = loaded.vocab.iter().map(|s| s.as_str()).collect();

        let mut eos_token_ids: Vec<u32> = [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "</s>",
            "<eos>",
        ]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
        for known_id in [151645, 151643, 151644, 2, 0] {
            if !eos_token_ids.contains(&known_id) {
                eos_token_ids.push(known_id);
            }
        }

        // Sample first token
        let mut squeezed = logits
            .squeeze(0)
            .map_err(|e| format!("Squeeze error: {}", e))?;

        if let Some(ref mut g_state) = grammar_state {
            let mut mask = g_state.allowed_tokens(&vocab_refs);
            if g_state.is_accepting() {
                for &eos_id in &eos_token_ids {
                    if (eos_id as usize) < mask.len() {
                        mask[eos_id as usize] = true;
                    }
                }
            }
            let mut logits_vec = squeezed
                .to_vec1::<f32>()
                .map_err(|e| format!("To vec1 error: {}", e))?;
            for (idx, &allowed) in mask.iter().enumerate() {
                if !allowed {
                    logits_vec[idx] = f32::NEG_INFINITY;
                }
            }
            squeezed = Tensor::new(&logits_vec[..], &device)
                .map_err(|e| format!("Tensor creation error: {}", e))?;
        }

        let mut next_token = logits_processor
            .sample(&squeezed)
            .map_err(|e| format!("Sampling error: {}", e))?;

        if let Some(ref mut g_state) = grammar_state {
            if let Some(token_str) = loaded.vocab.get(next_token as usize) {
                let _ = g_state.accept_token(token_str);
            }
        }

        if let Some(delta) = decode_stream
            .step(next_token)
            .map_err(|e| format!("Decode stream error: {}", e))?
        {
            full_text.push_str(&delta);
        }

        let timeout_secs = if std::env::var("MIVI_ULTRA_LOW_RAM")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false)
        {
            30
        } else {
            60
        };
        let deadline = std::time::Instant::now();

        for _ in 1..max_tokens {
            if deadline.elapsed().as_secs() > timeout_secs {
                tracing::warn!(
                    "[NativeBrain] Inference timeout reached ({}s wall-clock limit)",
                    timeout_secs
                );
                break;
            }

            if eos_token_ids.contains(&next_token) {
                break;
            }

            // Single token forward
            let input = Tensor::new(&[next_token], &device)
                .map_err(|e| format!("Tensor creation error: {}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

            let logits = model
                .forward(&input, index_pos)
                .map_err(|e| format!("Forward pass error: {}", e))?;

            index_pos += 1;

            let mut squeezed = logits
                .squeeze(0)
                .map_err(|e| format!("Squeeze error: {}", e))?;

            if let Some(ref mut g_state) = grammar_state {
                let mut mask = g_state.allowed_tokens(&vocab_refs);
                if g_state.is_accepting() {
                    for &eos_id in &eos_token_ids {
                        if (eos_id as usize) < mask.len() {
                            mask[eos_id as usize] = true;
                        }
                    }
                }
                let mut logits_vec = squeezed
                    .to_vec1::<f32>()
                    .map_err(|e| format!("To vec1 error: {}", e))?;
                for (idx, &allowed) in mask.iter().enumerate() {
                    if !allowed {
                        logits_vec[idx] = f32::NEG_INFINITY;
                    }
                }
                squeezed = Tensor::new(&logits_vec[..], &device)
                    .map_err(|e| format!("Tensor creation error: {}", e))?;
            }

            next_token = logits_processor
                .sample(&squeezed)
                .map_err(|e| format!("Sampling error: {}", e))?;

            if let Some(ref mut g_state) = grammar_state {
                if let Some(token_str) = loaded.vocab.get(next_token as usize) {
                    let _ = g_state.accept_token(token_str);
                }
            }

            if let Some(delta) = decode_stream
                .step(next_token)
                .map_err(|e| format!("Decode stream error: {}", e))?
            {
                full_text.push_str(&delta);
            }

            // Cheap per-token stop-word check on the incrementally built text.
            if t.stop_words.iter().any(|stop| full_text.ends_with(stop)) {
                break;
            }
        }

        let decoded = full_text;

        let mut cleaned = decoded;
        for stop in &t.stop_words {
            if let Some(stripped) = cleaned.strip_suffix(stop) {
                cleaned = stripped.to_string();
            }
        }

        Ok(crate::brain::strip_think_blocks(&cleaned))
    }

    pub fn query_raw_prompt(
        &self,
        model_path: &Path,
        formatted_prompt: &str,
        temp_str: &str,
        max_tokens: usize,
        grammar_path: Option<String>,
    ) -> Result<String, String> {
        let loaded = self.get_or_load(model_path)?;
        let mut model = loaded.model.lock().unwrap();
        model.clear_kv_cache();
        let tokenizer = &loaded.tokenizer;

        let tokens = tokenizer
            .encode(formatted_prompt, true)
            .map_err(|e| format!("Tokenization error: {}", e))?;
        let token_ids = tokens.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err("Tokenized prompt is empty".to_string());
        }

        let device = Device::Cpu;
        let temp = temp_str.parse::<f64>().unwrap_or(0.2);

        let sampling = if temp <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopP {
                p: 0.9,
                temperature: temp,
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);

        let mut index_pos = 0;
        // Incremental detokenizer: decoding only the delta per token keeps the
        // stop-word check O(1) instead of re-decoding the whole sequence (O(n^2)).
        let mut decode_stream = tokenizer.decode_stream(true);
        let mut full_text = String::new();

        // Prefill
        let input = Tensor::new(&token_ids[..], &device)
            .map_err(|e| format!("Tensor creation error: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

        let logits = model
            .forward(&input, index_pos)
            .map_err(|e| format!("Forward pass error: {}", e))?;

        index_pos += token_ids.len();

        let mut grammar_state = load_grammar_state(&grammar_path);
        let vocab_refs: Vec<&str> = loaded.vocab.iter().map(|s| s.as_str()).collect();
        let mut eos_token_ids: Vec<u32> = [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "</s>",
            "<eos>",
        ]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
        for known_id in [151645, 151643, 151644, 2, 0] {
            if !eos_token_ids.contains(&known_id) {
                eos_token_ids.push(known_id);
            }
        }

        // Sample first token
        let mut squeezed = logits
            .squeeze(0)
            .map_err(|e| format!("Squeeze error: {}", e))?;

        if let Some(ref mut g_state) = grammar_state {
            let mut mask = g_state.allowed_tokens(&vocab_refs);
            if g_state.is_accepting() {
                for &eos_id in &eos_token_ids {
                    if (eos_id as usize) < mask.len() {
                        mask[eos_id as usize] = true;
                    }
                }
            }
            let mut logits_vec = squeezed
                .to_vec1::<f32>()
                .map_err(|e| format!("To vec1 error: {}", e))?;
            for (idx, &allowed) in mask.iter().enumerate() {
                if !allowed {
                    logits_vec[idx] = f32::NEG_INFINITY;
                }
            }
            squeezed = Tensor::new(&logits_vec[..], &device)
                .map_err(|e| format!("Tensor creation error: {}", e))?;
        }

        let mut next_token = logits_processor
            .sample(&squeezed)
            .map_err(|e| format!("Sampling error: {}", e))?;

        if let Some(ref mut g_state) = grammar_state {
            if let Some(token_str) = loaded.vocab.get(next_token as usize) {
                let _ = g_state.accept_token(token_str);
            }
        }

        if let Some(delta) = decode_stream
            .step(next_token)
            .map_err(|e| format!("Decode stream error: {}", e))?
        {
            full_text.push_str(&delta);
        }

        let timeout_secs = if std::env::var("MIVI_ULTRA_LOW_RAM")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false)
        {
            30
        } else {
            60
        };
        let deadline = std::time::Instant::now();

        let t = crate::server::active_chat_template();
        for _ in 1..max_tokens {
            if deadline.elapsed().as_secs() > timeout_secs {
                tracing::warn!(
                    "[NativeBrain] Inference timeout reached ({}s wall-clock limit)",
                    timeout_secs
                );
                break;
            }

            if eos_token_ids.contains(&next_token) {
                break;
            }

            let input = Tensor::new(&[next_token], &device)
                .map_err(|e| format!("Tensor creation error: {}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

            let logits = model
                .forward(&input, index_pos)
                .map_err(|e| format!("Forward pass error: {}", e))?;

            index_pos += 1;

            let mut squeezed = logits
                .squeeze(0)
                .map_err(|e| format!("Squeeze error: {}", e))?;

            if let Some(ref mut g_state) = grammar_state {
                let mut mask = g_state.allowed_tokens(&vocab_refs);
                if g_state.is_accepting() {
                    for &eos_id in &eos_token_ids {
                        if (eos_id as usize) < mask.len() {
                            mask[eos_id as usize] = true;
                        }
                    }
                }
                let mut logits_vec = squeezed
                    .to_vec1::<f32>()
                    .map_err(|e| format!("To vec1 error: {}", e))?;
                for (idx, &allowed) in mask.iter().enumerate() {
                    if !allowed {
                        logits_vec[idx] = f32::NEG_INFINITY;
                    }
                }
                squeezed = Tensor::new(&logits_vec[..], &device)
                    .map_err(|e| format!("Tensor creation error: {}", e))?;
            }

            next_token = logits_processor
                .sample(&squeezed)
                .map_err(|e| format!("Sampling error: {}", e))?;

            if let Some(ref mut g_state) = grammar_state {
                if let Some(token_str) = loaded.vocab.get(next_token as usize) {
                    let _ = g_state.accept_token(token_str);
                }
            }

            if let Some(delta) = decode_stream
                .step(next_token)
                .map_err(|e| format!("Decode stream error: {}", e))?
            {
                full_text.push_str(&delta);
            }

            // Cheap per-token stop-word check on the incrementally built text.
            if t.stop_words.iter().any(|stop| full_text.ends_with(stop)) {
                break;
            }
        }

        let decoded = full_text;

        let mut cleaned = decoded;
        for stop in &t.stop_words {
            if let Some(stripped) = cleaned.strip_suffix(stop) {
                cleaned = stripped.to_string();
            }
        }

        Ok(crate::brain::strip_think_blocks(&cleaned))
    }

    pub fn query_stream(
        &self,
        model_path: &Path,
        prompt: &str,
        system_prompt: &str,
        temp_str: &str,
        max_tokens: usize,
        grammar_path: Option<String>,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, String> {
        let loaded = self.get_or_load(model_path)?;
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let prompt = prompt.to_string();
        let system_prompt = system_prompt.to_string();
        let temp_str = temp_str.to_string();

        tokio::task::spawn_blocking(move || {
            let result = (|| -> Result<(), String> {
                let mut model = loaded.model.lock().unwrap();
                model.clear_kv_cache();
                let tokenizer = &loaded.tokenizer;

                let t = crate::server::active_chat_template();

                // Detect if the prompt is already formatted with chat template tokens.
                let formatted_prompt = if !t.system_prefix.is_empty()
                    && prompt.trim_start().starts_with(t.system_prefix.trim())
                    && prompt.contains(t.assistant_start.trim())
                {
                    prompt.to_string()
                } else {
                    let (extracted_system, extracted_user) =
                        crate::brain::split_prompt_system_user(&prompt);
                    let final_system = if extracted_system.is_empty() {
                        system_prompt.to_string()
                    } else {
                        extracted_system
                    };
                    let final_user = if extracted_user.is_empty() {
                        prompt.to_string()
                    } else {
                        let trimmed = extracted_user.trim();
                        if let Some(stripped) = trimmed.strip_prefix("Current user request:") {
                            stripped.trim().to_string()
                        } else {
                            trimmed.to_string()
                        }
                    };
                    format!(
                        "{}{}{}{}{}{}{}",
                        t.system_prefix,
                        final_system,
                        t.system_suffix,
                        t.user_prefix,
                        final_user,
                        t.user_suffix,
                        t.assistant_start
                    )
                };

                let tokens = tokenizer
                    .encode(formatted_prompt, true)
                    .map_err(|e| format!("Tokenization error: {}", e))?;
                let token_ids = tokens.get_ids().to_vec();
                if token_ids.is_empty() {
                    return Err("Tokenized prompt is empty".to_string());
                }

                let device = Device::Cpu;
                let temp = temp_str.parse::<f64>().unwrap_or(0.2);

                let sampling = if temp <= 0.0 {
                    Sampling::ArgMax
                } else {
                    Sampling::TopP {
                        p: 0.9,
                        temperature: temp,
                    }
                };
                let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);

                let mut index_pos = 0;
                // Incremental detokenizer (see query() for rationale).
                let mut decode_stream = tokenizer.decode_stream(true);
                let mut full_text = String::new();

                // Prefill
                let input = Tensor::new(&token_ids[..], &device)
                    .map_err(|e| format!("Tensor creation error: {}", e))?
                    .unsqueeze(0)
                    .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

                let logits = model
                    .forward(&input, index_pos)
                    .map_err(|e| format!("Forward pass error: {}", e))?;

                index_pos += token_ids.len();

                let mut grammar_state = load_grammar_state(&grammar_path);
                let eos_token_id = tokenizer
                    .token_to_id("<|im_end|>")
                    .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

                let vocab_refs: Vec<&str> = loaded.vocab.iter().map(|s| s.as_str()).collect();

                // Sample first token
                let mut squeezed = logits
                    .squeeze(0)
                    .map_err(|e| format!("Squeeze error: {}", e))?;

                if let Some(ref mut g_state) = grammar_state {
                    let mut mask = g_state.allowed_tokens(&vocab_refs);
                    if g_state.is_accepting() {
                        if let Some(eos_id) = eos_token_id {
                            if (eos_id as usize) < mask.len() {
                                mask[eos_id as usize] = true;
                            }
                        }
                    }
                    let mut logits_vec = squeezed
                        .to_vec1::<f32>()
                        .map_err(|e| format!("To vec1 error: {}", e))?;
                    for (idx, &allowed) in mask.iter().enumerate() {
                        if !allowed {
                            logits_vec[idx] = f32::NEG_INFINITY;
                        }
                    }
                    squeezed = Tensor::new(&logits_vec[..], &device)
                        .map_err(|e| format!("Tensor creation error: {}", e))?;
                }

                let mut next_token = logits_processor
                    .sample(&squeezed)
                    .map_err(|e| format!("Sampling error: {}", e))?;

                if let Some(ref mut g_state) = grammar_state {
                    if let Some(token_str) = loaded.vocab.get(next_token as usize) {
                        let _ = g_state.accept_token(token_str);
                    }
                }

                let mut skipping_think = false;

                // Send an incremental delta through the thinking-block filter.
                // Returns false when the receiver has been closed.
                let send_delta = |delta: &str, skipping_think: &mut bool| -> bool {
                    let mut filtered_chunk = String::new();
                    for line in delta.split_inclusive('\n') {
                        if let Some(clean) = crate::model_process::strip_thinking_from_stream_line(
                            line,
                            skipping_think,
                        ) {
                            filtered_chunk.push_str(&clean);
                        }
                    }
                    if !filtered_chunk.is_empty() {
                        tx.blocking_send(filtered_chunk).is_ok()
                    } else {
                        true
                    }
                };

                if let Some(delta) = decode_stream
                    .step(next_token)
                    .map_err(|e| format!("Decode stream error: {}", e))?
                {
                    full_text.push_str(&delta);
                    if !send_delta(&delta, &mut skipping_think) {
                        return Ok(());
                    }
                }

                let timeout_secs = if std::env::var("MIVI_ULTRA_LOW_RAM")
                    .map(|v| v == "1" || v == "true")
                    .unwrap_or(false)
                {
                    30
                } else {
                    60
                };
                let deadline = std::time::Instant::now();

                for _ in 1..max_tokens {
                    if deadline.elapsed().as_secs() > timeout_secs {
                        tracing::warn!(
                            "[NativeBrain] Inference timeout reached ({}s wall-clock limit)",
                            timeout_secs
                        );
                        let _ = tx.blocking_send(format!(
                            "\n[NativeBrain: Timeout reached after {}s]",
                            timeout_secs
                        ));
                        break;
                    }

                    if let Some(eos_id) = eos_token_id {
                        if next_token == eos_id {
                            break;
                        }
                    }

                    // Single token forward
                    let input = Tensor::new(&[next_token], &device)
                        .map_err(|e| format!("Tensor creation error: {}", e))?
                        .unsqueeze(0)
                        .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

                    let logits = model
                        .forward(&input, index_pos)
                        .map_err(|e| format!("Forward pass error: {}", e))?;

                    index_pos += 1;

                    let mut squeezed = logits
                        .squeeze(0)
                        .map_err(|e| format!("Squeeze error: {}", e))?;

                    if let Some(ref mut g_state) = grammar_state {
                        let mut mask = g_state.allowed_tokens(&vocab_refs);
                        if g_state.is_accepting() {
                            if let Some(eos_id) = eos_token_id {
                                if (eos_id as usize) < mask.len() {
                                    mask[eos_id as usize] = true;
                                }
                            }
                        }
                        let mut logits_vec = squeezed
                            .to_vec1::<f32>()
                            .map_err(|e| format!("To vec1 error: {}", e))?;
                        for (idx, &allowed) in mask.iter().enumerate() {
                            if !allowed {
                                logits_vec[idx] = f32::NEG_INFINITY;
                            }
                        }
                        squeezed = Tensor::new(&logits_vec[..], &device)
                            .map_err(|e| format!("Tensor creation error: {}", e))?;
                    }

                    next_token = logits_processor
                        .sample(&squeezed)
                        .map_err(|e| format!("Sampling error: {}", e))?;

                    if let Some(ref mut g_state) = grammar_state {
                        if let Some(token_str) = loaded.vocab.get(next_token as usize) {
                            let _ = g_state.accept_token(token_str);
                        }
                    }

                    if let Some(delta) = decode_stream
                        .step(next_token)
                        .map_err(|e| format!("Decode stream error: {}", e))?
                    {
                        full_text.push_str(&delta);
                        // Cheap per-token stop-word check on accumulated text.
                        if t.stop_words.iter().any(|stop| full_text.ends_with(stop)) {
                            break;
                        }
                        if !send_delta(&delta, &mut skipping_think) {
                            break;
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                error!("[NativeBrain] Stream error: {}", e);
            }
        });

        Ok(rx)
    }
}

#[cfg(not(feature = "native"))]
#[derive(Clone, Debug)]
pub struct NativeBrain;

#[cfg(not(feature = "native"))]
impl NativeBrain {
    pub fn new() -> Self {
        Self
    }

    pub fn query(
        &self,
        _model_path: &std::path::Path,
        _prompt: &str,
        _system_prompt: &str,
        _temp_str: &str,
        _max_tokens: usize,
    ) -> Result<String, String> {
        Err(
            "Native inference backend is disabled. Rebuild with `--features native` to enable it."
                .to_string(),
        )
    }

    pub fn query_raw_prompt(
        &self,
        _model_path: &std::path::Path,
        _formatted_prompt: &str,
        _temp_str: &str,
        _max_tokens: usize,
    ) -> Result<String, String> {
        Err(
            "Native inference backend is disabled. Rebuild with `--features native` to enable it."
                .to_string(),
        )
    }

    pub fn query_stream(
        &self,
        _model_path: &std::path::Path,
        _prompt: &str,
        _system_prompt: &str,
        _temp_str: &str,
        _max_tokens: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, String> {
        Err(
            "Native inference backend is disabled. Rebuild with `--features native` to enable it."
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    #[test]
    fn test_grammar_error() {
        for grammar_name in &[
            "json_object.gbnf",
            "openai_tool_call.gbnf",
            "hermes_tool_call.gbnf",
        ] {
            let grammar_path = format!("configs/grammars/{}", grammar_name);
            let grammar_str = std::fs::read_to_string(&grammar_path).unwrap();
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
            let res = schoolmarm::Grammar::new(cleaned_str.trim());
            assert!(
                res.is_ok(),
                "Failed to parse grammar {}: {:?}",
                grammar_name,
                res.err()
            );
        }
    }

    #[cfg(feature = "native")]
    #[test]
    #[ignore]
    fn test_native_brain_inference_if_present() {
        let model_path = std::path::Path::new("models/Qwen3-1.7B-Q2_K.gguf");
        if !model_path.exists() {
            println!("Skipping native brain test as model weights are missing");
            return;
        }

        let brain = NativeBrain::new();
        let prompt = "Why is Rust a safe programming language? Keep it to 1 sentence.";
        let system_prompt = "You are a helpful assistant.";

        let result = brain.query(model_path, prompt, system_prompt, "0.1", 100, None);
        assert!(result.is_ok(), "Query failed: {:?}", result.err());
        let answer = result.unwrap();
        println!("Native inference answer: {}", answer);
        assert!(!answer.is_empty(), "Answer should not be empty");
    }

    #[cfg(feature = "native")]
    #[test]
    #[ignore]
    fn test_native_brain_grammar_inference_if_present() {
        let model_path = std::path::Path::new("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
        let grammar_path = std::path::Path::new("configs/grammars/json_object.gbnf");
        if !model_path.exists() || !grammar_path.exists() {
            println!("Skipping native brain grammar test as weights or grammar are missing");
            return;
        }

        let brain = NativeBrain::new();
        let grammar_state_loaded =
            load_grammar_state(&Some(grammar_path.to_string_lossy().to_string()));
        assert!(
            grammar_state_loaded.is_some(),
            "Grammar state failed to load"
        );

        let prompt =
            "Output a JSON object with keys 'name' and 'age' for a person named John aged 30.";
        let system_prompt = "You are a helpful JSON generator. You MUST output ONLY valid JSON.";

        let result = brain.query(
            model_path,
            prompt,
            system_prompt,
            "0.1",
            100,
            Some(grammar_path.to_string_lossy().to_string()),
        );
        assert!(result.is_ok(), "Query failed: {:?}", result.err());
        let answer = result.unwrap();
        println!("Grammar constrained JSON output: {}", answer);

        let json_val: serde_json::Value =
            serde_json::from_str(&answer).expect("Output should be valid JSON");
        assert!(json_val.get("name").is_some(), "Should contain name");
        assert_eq!(json_val.get("age").and_then(|v| v.as_u64()), Some(30));
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    #[ignore]
    async fn test_native_brain_stream_inference_if_present() {
        let model_path = Path::new("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
        if !model_path.exists() {
            println!("Skipping native brain stream test as model weights are missing");
            return;
        }

        let brain = NativeBrain::new();
        let prompt = "Why is Rust a safe programming language? Keep it to 1 sentence.";
        let system_prompt = "You are a helpful assistant.";

        let mut rx = brain
            .query_stream(model_path, prompt, system_prompt, "0.1", 100, None)
            .expect("Stream setup failed");

        let mut output = String::new();
        while let Some(chunk) = rx.recv().await {
            print!("{}", chunk);
            output.push_str(&chunk);
        }
        println!();
        assert!(!output.is_empty(), "Stream output should not be empty");
    }
}
