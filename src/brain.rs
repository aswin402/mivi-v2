use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone)]
pub struct EdgeBrain {
    pub llama_cli: PathBuf,
    pub minicpm_cli: PathBuf,
    pub llama_path: PathBuf,
    pub qwen_path: PathBuf,
    pub minicpm_path: PathBuf,
    pub minicpm_proj: PathBuf,
}

impl EdgeBrain {
    pub fn new() -> Self {
        let base_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let exe_ext = if cfg!(target_os = "windows") { ".exe" } else { "" };

        let possible_bins = vec![
            base_dir.join("bin").join(format!("llama-cli{}", exe_ext)),
            PathBuf::from(format!("llama-cli{}", exe_ext)),
        ];

        let llama_cli = possible_bins
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from(format!("llama-cli{}", exe_ext)));

        let possible_minicpm_bins = vec![
            base_dir.join("bin").join(format!("llama-mtmd-cli{}", exe_ext)),
            base_dir.join("bin").join(format!("llama-minicpmv-cli{}", exe_ext)),
            base_dir.join("bin").join(format!("llama-cli{}", exe_ext)),
            PathBuf::from(format!("llama-mtmd-cli{}", exe_ext)),
        ];

        let minicpm_cli = possible_minicpm_bins
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| llama_cli.clone());

        let models_dir = base_dir.join("models");
        let llama_path = models_dir.join("Llama-3.2-1B-Instruct-IQ3_M.gguf");
        let qwen_path = models_dir.join("qwen2.5-0.5b-instruct-q2_k.gguf");
        let minicpm_path = models_dir.join("MiniCPM-V-4.6-Q4_K_M.gguf");
        let minicpm_proj = models_dir.join("mmproj-MiniCPM-V-4.6-Q8_0.gguf");

        Self {
            llama_cli,
            minicpm_cli,
            llama_path,
            qwen_path,
            minicpm_path,
            minicpm_proj,
        }
    }

    fn run_cli(&self, model_path: &Path, prompt: &str, system_prompt: &str, temp: &str, context_size: &str) -> Result<String, String> {
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, prompt
        );

        let output = Command::new(&self.llama_cli)
            .arg("-m")
            .arg(model_path)
            .arg("-ngl")
            .arg("999")
            .arg("-c")
            .arg(context_size)
            .arg("-fa")
            .arg("on")
            .arg("-ctk")
            .arg("q8_0")
            .arg("-ctv")
            .arg("q8_0")
            .arg("-p")
            .arg(&formatted_prompt)
            .arg("--temp")
            .arg(temp)
            .arg("--simple-io")
            .arg("-st")
            .output()
            .map_err(|e| format!("Failed to execute llama-cli: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();

        let response = if let Some(pos) = stdout.find("<|im_start|>assistant") {
            &stdout[pos + "<|im_start|>assistant".len()..]
        } else if let Some(pos) = stdout.find("> ") {
            &stdout[pos + 2..]
        } else {
            &stdout[..]
        };

        let clean = response
            .split("[ Prompt:")
            .next()
            .unwrap_or(response)
            .split("Exiting...")
            .next()
            .unwrap_or(response)
            .trim();

        Ok(clean.to_string())
    }

    pub fn query_reasoner(&self, prompt: &str, system_prompt: &str) -> Result<String, String> {
        self.run_cli(&self.llama_path, prompt, system_prompt, "0.2", "8192")
    }

    pub fn query_coder(&self, prompt: &str, system_prompt: &str) -> Result<String, String> {
        self.run_cli(&self.qwen_path, prompt, system_prompt, "0.1", "8192")
    }

    pub fn query_vision(&self, image_path: &str, prompt: &str) -> Result<String, String> {
        if !Path::new(image_path).exists() {
            return Err(format!("Image file not found at: {}", image_path));
        }

        if !self.minicpm_path.exists() {
            return Err(format!(
                "Vision model weights not found at '{}'. Download MiniCPM-V-4.6-Q4_K_M.gguf and mmproj-MiniCPM-V-4.6-Q8_0.gguf into models/",
                self.minicpm_path.display()
            ));
        }

        let output = Command::new(&self.minicpm_cli)
            .arg("-m")
            .arg(&self.minicpm_path)
            .arg("--mmproj")
            .arg(&self.minicpm_proj)
            .arg("-ngl")
            .arg("999")
            .arg("--image")
            .arg(image_path)
            .arg("-p")
            .arg(prompt)
            .output()
            .map_err(|e| format!("Failed to execute vision cli: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout.trim().to_string())
    }
}

