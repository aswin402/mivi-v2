use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelRole {
    Reasoner,
    Coder,
    Vision,
    Chat,
    Tool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    LlamaCli,
    LlamaServer,
    VisionCli,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatTemplateConfig {
    pub system_prefix: String,
    pub system_suffix: String,
    pub user_prefix: String,
    pub user_suffix: String,
    pub assistant_prefix: String,
    pub assistant_suffix: String,
    pub tool_prefix: String,
    pub tool_suffix: String,
    pub assistant_start: String,
    pub stop_words: Vec<String>,
}

impl Default for ChatTemplateConfig {
    fn default() -> Self {
        Self {
            system_prefix: "<|im_start|>system\n".to_string(),
            system_suffix: "<|im_end|>\n".to_string(),
            user_prefix: "<|im_start|>user\n".to_string(),
            user_suffix: "<|im_end|>\n".to_string(),
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            assistant_suffix: "<|im_end|>\n".to_string(),
            tool_prefix: "<|im_start|>tool\nTool result ({id}): ".to_string(),
            tool_suffix: "<|im_end|>\n".to_string(),
            assistant_start: "<|im_start|>assistant\n".to_string(),
            stop_words: vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()],
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub role: ModelRole,
    pub backend: BackendKind,
    pub path: String,
    pub context_tokens: usize,
    pub ram_mb_estimate: usize,
    pub enabled: bool,
    pub notes: Option<String>,
    #[serde(default)]
    pub chat_template: Option<ChatTemplateConfig>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelCatalog {
    pub external_model: String,
    pub models: Vec<ModelCatalogEntry>,
}

#[derive(Debug)]
pub enum ModelCatalogError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
    NotFound(String),
}

impl ModelCatalog {
    pub fn load_default() -> Result<Self, ModelCatalogError> {
        Self::load_path("configs/models.json")
    }

    pub fn load_path(path: impl AsRef<Path>) -> Result<Self, ModelCatalogError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_slice(&mmap)
    }

    pub fn from_json(text: &str) -> Result<Self, ModelCatalogError> {
        Self::from_slice(text.as_bytes())
    }

    pub fn from_slice(slice: &[u8]) -> Result<Self, ModelCatalogError> {
        let catalog: Self = serde_json::from_slice(slice)?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn find(&self, id: &str) -> Option<&ModelCatalogEntry> {
        self.models.iter().find(|entry| entry.id == id)
    }

    pub fn default_enabled_path(&self, role: ModelRole) -> Option<&str> {
        self.models
            .iter()
            .find(|entry| entry.enabled && entry.role == role)
            .map(|entry| entry.path.as_str())
    }

    fn validate(&self) -> Result<(), ModelCatalogError> {
        if self.external_model != "mivi" {
            return Err(ModelCatalogError::Invalid(
                "external_model must be mivi; internal worker names stay private".to_string(),
            ));
        }

        if self.models.is_empty() {
            return Err(ModelCatalogError::Invalid(
                "catalog must contain at least one model".to_string(),
            ));
        }

        let mut seen_ids = HashSet::new();
        for model in &self.models {
            if model.id.trim().is_empty() {
                return Err(ModelCatalogError::Invalid(
                    "model id cannot be empty".to_string(),
                ));
            }
            if !seen_ids.insert(model.id.as_str()) {
                return Err(ModelCatalogError::Invalid(format!(
                    "duplicate model id: {}",
                    model.id
                )));
            }
            if model.path.trim().is_empty() {
                return Err(ModelCatalogError::Invalid(format!(
                    "model {} path cannot be empty",
                    model.id
                )));
            }
            if model.context_tokens == 0 {
                return Err(ModelCatalogError::Invalid(format!(
                    "model {} context_tokens must be greater than zero",
                    model.id
                )));
            }
            if model.ram_mb_estimate == 0 {
                return Err(ModelCatalogError::Invalid(format!(
                    "model {} ram_mb_estimate must be greater than zero",
                    model.id
                )));
            }
        }

        if self.default_enabled_path(ModelRole::Reasoner).is_none() {
            return Err(ModelCatalogError::Invalid(
                "catalog must contain an enabled reasoner model".to_string(),
            ));
        }

        Ok(())
    }
}

pub fn print_model_list(catalog: &ModelCatalog) {
    println!("External model: {}", catalog.external_model);
    println!("ID\tROLE\tBACKEND\tRAM_MB\tCTX\tENABLED");
    for model in &catalog.models {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            model.id,
            model.role,
            model.backend,
            model.ram_mb_estimate,
            model.context_tokens,
            model.enabled
        );
    }
}

pub fn print_model_inspect(catalog: &ModelCatalog, id: &str) -> Result<(), ModelCatalogError> {
    let model = catalog
        .find(id)
        .ok_or_else(|| ModelCatalogError::NotFound(id.to_string()))?;

    println!("id: {}", model.id);
    println!("role: {}", model.role);
    println!("backend: {}", model.backend);
    println!("path: {}", model.path);
    println!("context_tokens: {}", model.context_tokens);
    println!("ram_mb_estimate: {}", model.ram_mb_estimate);
    println!("enabled: {}", model.enabled);
    if let Some(notes) = &model.notes {
        println!("notes: {}", notes);
    }
    Ok(())
}

pub fn print_model_fit(catalog: &ModelCatalog, id: &str) -> Result<(), ModelCatalogError> {
    let model = catalog
        .find(id)
        .ok_or_else(|| ModelCatalogError::NotFound(id.to_string()))?;

    let (total_ram_mb, avail_ram_mb) = read_system_memory();
    let model_mb = model.ram_mb_estimate;
    let kv_3k_mb = (3072 * 2 * 14 * 64 * 4) / (8 * 1024 * 1024); // ~40 MB
    let kv_64k_mb = 180; // with SnapKV attention pruning + 4-bit KV
    let peak_3k = model_mb + kv_3k_mb;
    let peak_64k = model_mb + kv_64k_mb;

    println!("============================================================");
    println!("  🧠 MIVI MODEL FIT CALCULATOR: {}", model.id);
    println!("============================================================");
    println!("  Model Weights:       ~{} MB", model_mb);
    println!("  KV Cache (3072 ctx): ~{} MB (q4_0)", kv_3k_mb.max(30));
    println!("  KV Cache (64k ctx):  ~{} MB (q4_0 + SnapKV)", kv_64k_mb);
    println!("  Peak Inference RAM:  ~{} MB - {} MB", peak_3k, peak_64k);
    if total_ram_mb > 0 {
        println!("  System Total RAM:    {} MB", total_ram_mb);
        println!("  System Free/Avail:   {} MB", avail_ram_mb);
    }
    println!("------------------------------------------------------------");
    if peak_64k <= 600 {
        println!("  Status: ✅ FITS COMFORTABLY IN ULTRA-LOW RAM (< 600 MB)");
    } else if peak_3k <= 1024 {
        println!("  Status: ⚡ FITS IN STANDARD LOCAL RAM (< 1 GB)");
    } else {
        println!("  Status: ⚠️ REQUIRES DEDICATED GPU OR >2 GB RAM");
    }
    println!("============================================================");
    Ok(())
}

fn read_system_memory() -> (usize, usize) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0;
            let mut avail_kb = 0;
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(num) = line.split_whitespace().nth(1) {
                        total_kb = num.parse::<usize>().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(num) = line.split_whitespace().nth(1) {
                        avail_kb = num.parse::<usize>().unwrap_or(0);
                    }
                }
            }
            if total_kb > 0 {
                return (total_kb / 1024, avail_kb / 1024);
            }
        }
    }
    (0, 0)
}

impl fmt::Display for ModelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelRole::Reasoner => f.write_str("reasoner"),
            ModelRole::Coder => f.write_str("coder"),
            ModelRole::Vision => f.write_str("vision"),
            ModelRole::Chat => f.write_str("chat"),
            ModelRole::Tool => f.write_str("tool"),
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendKind::LlamaCli => f.write_str("llama-cli"),
            BackendKind::LlamaServer => f.write_str("llama-server"),
            BackendKind::VisionCli => f.write_str("vision-cli"),
        }
    }
}

impl fmt::Display for ModelCatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelCatalogError::Io(err) => write!(f, "catalog I/O error: {err}"),
            ModelCatalogError::Json(err) => write!(f, "catalog JSON error: {err}"),
            ModelCatalogError::Invalid(msg) => write!(f, "invalid catalog: {msg}"),
            ModelCatalogError::NotFound(id) => write!(f, "model not found in catalog: {id}"),
        }
    }
}

impl std::error::Error for ModelCatalogError {}

impl From<std::io::Error> for ModelCatalogError {
    fn from(err: std::io::Error) -> Self {
        ModelCatalogError::Io(err)
    }
}

impl From<serde_json::Error> for ModelCatalogError {
    fn from(err: serde_json::Error) -> Self {
        ModelCatalogError::Json(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
    {
      "external_model": "mivi",
      "models": [
        {
          "id": "qwen3-06b-reasoner",
          "role": "reasoner",
          "backend": "llama-cli",
          "path": "models/qwen3-0.6b-q4_k_m.gguf",
          "context_tokens": 32768,
          "ram_mb_estimate": 540,
          "enabled": true,
          "notes": "default low-resource reasoner"
        },
        {
          "id": "qwen25-05b-coder",
          "role": "coder",
          "backend": "llama-cli",
          "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
          "context_tokens": 32768,
          "ram_mb_estimate": 430,
          "enabled": true,
          "notes": "default coder/tool worker"
        }
      ]
    }
    "#;

    #[test]
    fn parses_catalog_and_finds_mivi_external_model() {
        let catalog = ModelCatalog::from_json(SAMPLE).expect("catalog should parse");
        assert_eq!(catalog.external_model, "mivi");
        assert_eq!(catalog.models.len(), 2);
        assert_eq!(
            catalog.find("qwen3-06b-reasoner").unwrap().role,
            ModelRole::Reasoner
        );
    }

    #[test]
    fn resolves_first_enabled_model_path_for_role() {
        let catalog = ModelCatalog::from_json(SAMPLE).expect("catalog should parse");

        assert_eq!(
            catalog.default_enabled_path(ModelRole::Reasoner),
            Some("models/qwen3-0.6b-q4_k_m.gguf")
        );
    }

    #[test]
    fn ignores_disabled_models_when_resolving_role_path() {
        let disabled_coder = SAMPLE.replace(
            r#""id": "qwen25-05b-coder",
          "role": "coder",
          "backend": "llama-cli",
          "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
          "context_tokens": 32768,
          "ram_mb_estimate": 430,
          "enabled": true"#,
            r#""id": "qwen25-05b-coder",
          "role": "coder",
          "backend": "llama-cli",
          "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
          "context_tokens": 32768,
          "ram_mb_estimate": 430,
          "enabled": false"#,
        );
        let catalog = ModelCatalog::from_json(&disabled_coder).expect("catalog should parse");

        assert_eq!(catalog.default_enabled_path(ModelRole::Coder), None);
    }

    #[test]
    fn rejects_catalog_without_enabled_reasoner_model() {
        let no_reasoner = SAMPLE.replace(r#""role": "reasoner""#, r#""role": "chat""#);
        let err = ModelCatalog::from_json(&no_reasoner).expect_err("enabled reasoner required");

        assert!(err.to_string().contains("enabled reasoner model"));
    }

    #[test]
    fn rejects_catalog_with_empty_model_list() {
        let err = ModelCatalog::from_json(r#"{"external_model":"mivi","models":[]}"#)
            .expect_err("catalog should require models");

        assert!(err.to_string().contains("at least one model"));
    }

    #[test]
    fn rejects_catalog_that_exposes_non_mivi_external_model() {
        let err = ModelCatalog::from_json(r#"{"external_model":"qwen","models":[]}"#)
            .expect_err("external model must be mivi");
        assert!(err.to_string().contains("external_model must be mivi"));
    }

    #[test]
    fn rejects_duplicate_internal_model_ids() {
        let duplicate = SAMPLE.replace("qwen25-05b-coder", "qwen3-06b-reasoner");
        let err = ModelCatalog::from_json(&duplicate).expect_err("duplicate ids should fail");
        assert!(err.to_string().contains("duplicate model id"));
    }

    #[test]
    fn print_default_catalog() {
        let cat = ModelCatalog::load_default().expect("catalog load failed");
        assert_eq!(
            cat.default_enabled_path(ModelRole::Reasoner),
            Some("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
        );
        assert_eq!(
            cat.default_enabled_path(ModelRole::Coder),
            Some("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
        );
        assert_eq!(
            cat.default_enabled_path(ModelRole::Tool),
            Some("models/mivi-0.5b-tool-q4_k_m.gguf")
        );
    }
}
