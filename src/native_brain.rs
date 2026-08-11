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
        for entry in catalog.models {
            if Path::new(&entry.path) == model_path {
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
        let tok_json = parent.join("tokenizer.json");
        if tok_json.exists() {
            return Some(tok_json);
        }

        // Match standard names
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
    }
    None
}

#[cfg(feature = "native")]
pub enum QuantizedModel {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
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
                let model = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                    ct, reader, device,
                )
                .map_err(|e| format!("failed to load quantized qwen2: {}", e))?;
                Ok(Self::Qwen2(model))
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
        }
    }
}

#[cfg(feature = "native")]
pub struct LoadedModel {
    pub model: Mutex<QuantizedModel>,
    pub tokenizer: Tokenizer,
}

#[cfg(feature = "native")]
#[derive(Clone)]
pub struct NativeBrain {
    pub models: Arc<Mutex<HashMap<PathBuf, Arc<LoadedModel>>>>,
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
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_or_load(&self, model_path: &Path) -> Result<Arc<LoadedModel>, String> {
        let mut cache = self.models.lock().unwrap();
        let canonical_path = model_path.to_path_buf();

        if let Some(loaded) = cache.get(&canonical_path) {
            return Ok(loaded.clone());
        }

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

        let loaded = Arc::new(LoadedModel {
            model: Mutex::new(model),
            tokenizer,
        });

        cache.insert(canonical_path, loaded.clone());
        Ok(loaded)
    }

    pub fn query(
        &self,
        model_path: &Path,
        prompt: &str,
        system_prompt: &str,
        temp_str: &str,
        max_tokens: usize,
    ) -> Result<String, String> {
        let loaded = self.get_or_load(model_path)?;
        let mut model = loaded.model.lock().unwrap();
        let tokenizer = &loaded.tokenizer;

        let t = crate::server::active_chat_template();
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

        let formatted_prompt = format!(
            "{}{}{}{}{}{}{}",
            t.system_prefix,
            final_system,
            t.system_suffix,
            t.user_prefix,
            final_user,
            t.user_suffix,
            t.assistant_start
        );

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

        let mut generated_tokens = Vec::new();
        let mut index_pos = 0;

        // Prefill
        let input = Tensor::new(&token_ids[..], &device)
            .map_err(|e| format!("Tensor creation error: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

        let logits = model
            .forward(&input, index_pos)
            .map_err(|e| format!("Forward pass error: {}", e))?;

        index_pos += token_ids.len();

        // Sample first token
        let squeezed = logits
            .squeeze(0)
            .map_err(|e| format!("Squeeze error: {}", e))?;

        let mut next_token = logits_processor
            .sample(&squeezed)
            .map_err(|e| format!("Sampling error: {}", e))?;

        generated_tokens.push(next_token);

        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

        for _ in 1..max_tokens {
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

            let squeezed = logits
                .squeeze(0)
                .map_err(|e| format!("Squeeze error: {}", e))?;

            next_token = logits_processor
                .sample(&squeezed)
                .map_err(|e| format!("Sampling error: {}", e))?;

            generated_tokens.push(next_token);
        }

        let decoded = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| format!("Decoding error: {}", e))?;

        Ok(crate::brain::strip_think_blocks(&decoded))
    }

    pub fn query_stream(
        &self,
        model_path: &Path,
        prompt: &str,
        system_prompt: &str,
        temp_str: &str,
        max_tokens: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, String> {
        let loaded = self.get_or_load(model_path)?;
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let prompt = prompt.to_string();
        let system_prompt = system_prompt.to_string();
        let temp_str = temp_str.to_string();

        tokio::task::spawn_blocking(move || {
            let result = (|| -> Result<(), String> {
                let mut model = loaded.model.lock().unwrap();
                let tokenizer = &loaded.tokenizer;

                let t = crate::server::active_chat_template();
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

                let formatted_prompt = format!(
                    "{}{}{}{}{}{}{}",
                    t.system_prefix,
                    final_system,
                    t.system_suffix,
                    t.user_prefix,
                    final_user,
                    t.user_suffix,
                    t.assistant_start
                );

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

                let mut generated_tokens = Vec::new();
                let mut index_pos = 0;

                // Prefill
                let input = Tensor::new(&token_ids[..], &device)
                    .map_err(|e| format!("Tensor creation error: {}", e))?
                    .unsqueeze(0)
                    .map_err(|e| format!("Tensor unsqueeze error: {}", e))?;

                let logits = model
                    .forward(&input, index_pos)
                    .map_err(|e| format!("Forward pass error: {}", e))?;

                index_pos += token_ids.len();

                // Sample first token
                let squeezed = logits
                    .squeeze(0)
                    .map_err(|e| format!("Squeeze error: {}", e))?;

                let mut next_token = logits_processor
                    .sample(&squeezed)
                    .map_err(|e| format!("Sampling error: {}", e))?;

                generated_tokens.push(next_token);

                let eos_token_id = tokenizer
                    .token_to_id("<|im_end|>")
                    .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

                let mut decoded_len = 0;
                let mut skipping_think = false;

                // Helper closure to handle decoding a chunk and yielding/filtering thinking
                let mut process_and_send =
                    |generated_tokens: &[u32], skipping_think: &mut bool| -> Result<bool, String> {
                        let current_text = tokenizer
                            .decode(generated_tokens, true)
                            .map_err(|e| format!("Decoding error: {}", e))?;

                        if current_text.len() > decoded_len {
                            let new_chunk = &current_text[decoded_len..];
                            decoded_len = current_text.len();

                            // We use the stream-skipping helper for thinking blocks
                            let mut filtered_chunk = String::new();
                            for line in new_chunk.split_inclusive('\n') {
                                if let Some(clean) =
                                    crate::model_process::strip_thinking_from_stream_line(
                                        line,
                                        skipping_think,
                                    )
                                {
                                    filtered_chunk.push_str(&clean);
                                }
                            }

                            if !filtered_chunk.is_empty() {
                                if tx.blocking_send(filtered_chunk).is_err() {
                                    return Ok(false); // receiver closed
                                }
                            }
                        }
                        Ok(true)
                    };

                if !process_and_send(&generated_tokens, &mut skipping_think)? {
                    return Ok(());
                }

                for _ in 1..max_tokens {
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

                    let squeezed = logits
                        .squeeze(0)
                        .map_err(|e| format!("Squeeze error: {}", e))?;

                    next_token = logits_processor
                        .sample(&squeezed)
                        .map_err(|e| format!("Sampling error: {}", e))?;

                    generated_tokens.push(next_token);

                    if !process_and_send(&generated_tokens, &mut skipping_think)? {
                        break;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_brain_new() {
        let _brain = NativeBrain::new();
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_brain_inference_if_present() {
        let model_path = std::path::Path::new("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
        if !model_path.exists() {
            println!("Skipping native brain test as model weights are missing");
            return;
        }

        let brain = NativeBrain::new();
        let prompt = "Why is Rust a safe programming language? Keep it to 1 sentence.";
        let system_prompt = "You are a helpful assistant.";

        let result = brain.query(model_path, prompt, system_prompt, "0.1", 100);
        assert!(result.is_ok(), "Query failed: {:?}", result.err());
        let answer = result.unwrap();
        println!("Native inference answer: {}", answer);
        assert!(!answer.is_empty(), "Answer should not be empty");
    }

    #[cfg(feature = "native")]
    #[tokio::test]
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
            .query_stream(model_path, prompt, system_prompt, "0.1", 100)
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
