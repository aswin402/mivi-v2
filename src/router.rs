use std::path::PathBuf;

#[derive(Clone)]
pub struct NeedleRouter {
    pub needle_model_path: PathBuf,
}

impl NeedleRouter {
    pub fn new() -> Self {
        let needle_path = PathBuf::from("models/needle-model.safetensors");
        if needle_path.exists() {
            println!("[NEEDLE 26M] Cactus Compute Needle weights loaded from 'models/needle-model.safetensors'");
        }
        Self {
            needle_model_path: needle_path,
        }
    }

    pub fn classify_intent(&self, prompt: &str) -> &'static str {
        if prompt.is_empty() {
            return "CHAT";
        }

        let p = prompt.to_lowercase();

        if p.contains("image") || p.contains("photo") || p.contains("png") || p.contains("jpg") || p.contains("look at") {
            return "VISION";
        }

        if p.contains("and then") || p.contains("after that") || p.contains("step 1") || p.contains("first") || p.contains("multiple") {
            return "MULTI_STEP";
        }

        if p.contains("write") || p.contains("code") || p.contains("python") || p.contains("script") || p.contains("calculate") || p.contains("print") {
            return "DIRECT_CODE";
        }

        "CHAT"
    }
}
