use crate::brain::EdgeBrain;
use crate::cache::SemanticCache;
use crate::logger::DatasetLogger;
use crate::rag::TurboVecRAG;
use crate::verifier::CompilerVerifier;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, Debug)]
pub struct PlanStep {
    pub step: Option<usize>,
    pub description: String,
    pub language: Option<String>,
}

#[derive(Clone)]
pub struct AgentOrchestrator {
    pub brain: EdgeBrain,
    pub verifier: CompilerVerifier,
    pub cache: SemanticCache,
    pub dataset: DatasetLogger,
    pub rag: TurboVecRAG,
}

impl AgentOrchestrator {
    pub fn new(brain: EdgeBrain) -> Self {
        let verifier = CompilerVerifier::new(brain.clone());
        let cache = SemanticCache::new();
        let dataset = DatasetLogger::new();
        let rag = TurboVecRAG::new();
        Self {
            brain,
            verifier,
            cache,
            dataset,
            rag,
        }
    }


    fn extract_json_plan(&self, text: &str) -> Option<Vec<PlanStep>> {
        let json_str = if let Some(start) = text.find("```json") {
            let rest = &text[start + 7..];
            if let Some(end) = rest.find("```") {
                rest[..end].trim()
            } else {
                rest.trim()
            }
        } else if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                &text[start..=end]
            } else {
                text.trim()
            }
        } else {
            text.trim()
        };

        if let Ok(steps) = serde_json::from_str::<Vec<PlanStep>>(json_str) {
            return Some(steps);
        }

        if let Ok(val) = serde_json::from_str::<Value>(json_str) {
            if let Some(arr) = val.get("steps").and_then(|v| v.as_array()) {
                if let Ok(steps) = serde_json::from_value::<Vec<PlanStep>>(Value::Array(arr.clone())) {
                    return Some(steps);
                }
            }
        }

        None
    }

    pub async fn execute_plan(&self, request: &str) -> (bool, String) {
        if let Some(cached) = self.cache.get(request).await {
            println!("[Orchestrator] Exact cache hit (< 0.001s)!");
            return (true, cached);
        }

        println!("[MIVI-V2 Orchestrator] Executing request: '{}'", request);

        let req_lower = request.to_lowercase();
        let is_complex = req_lower.contains("and then")
            || req_lower.contains("after that")
            || req_lower.contains("step 1")
            || req_lower.contains("first")
            || req_lower.contains("multiple steps");

        let default_lang = if req_lower.contains("javascript") || req_lower.contains("js") {
            "javascript"
        } else if req_lower.contains("typescript") || req_lower.contains("ts") {
            "typescript"
        } else if req_lower.contains("rust") || req_lower.contains("rs") {
            "rust"
        } else if req_lower.contains("cpp") || req_lower.contains("c++") {
            "cpp"
        } else {
            "python"
        };

        let steps = if is_complex {
            println!("[SAKANA FUGU ROUTER] Complex task detected -> Engaging Llama 1B Planner...");
            let system_prompt = "You are the Orchestrator Brain. Break down the user's request into the MINIMAL number of necessary executable coding steps (1 to 3 steps max).\nRespond ONLY with a valid JSON array of step objects inside a ```json ... ``` block.\nEach step object must have keys:\n- 'step': integer\n- 'description': string description of what to write\n- 'language': string ('python' or 'javascript')";

            let plan_opt = if let Ok(raw_plan) = self.brain.query_reasoner(request, system_prompt) {
                self.extract_json_plan(&raw_plan)
            } else {
                None
            };

            plan_opt.unwrap_or_else(|| vec![PlanStep {
                step: Some(1),
                description: request.to_string(),
                language: Some(default_lang.to_string()),
            }])
        } else {
            println!("[SAKANA FUGU ROUTER] Simple task detected -> Fast-path direct Qwen Coder (3x faster)...");
            vec![PlanStep {
                step: Some(1),
                description: request.to_string(),
                language: Some(default_lang.to_string()),
            }]
        };

        println!("[Orchestrator] Generated Execution Plan ({} steps):", steps.len());

        let mut results_formatted = Vec::new();
        let mut context_accumulator = String::new();
        let mut overall_success = true;

        for (idx, step_info) in steps.iter().enumerate() {
            let step_num = step_info.step.unwrap_or(idx + 1);
            let lang = step_info.language.as_deref().unwrap_or(default_lang);
            println!("[Orchestrator] Running Step {}: {} ({})", step_num, step_info.description, lang);

            let rag_context = self.rag.format_rag_context(&step_info.description, 2).await;

            let mut prompt = step_info.description.clone();
            if !context_accumulator.is_empty() {
                prompt = format!("{}\n\nPrevious steps output:\n{}", prompt, context_accumulator);
            }
            if !rag_context.is_empty() {
                prompt = format!("{}\n\n{}", prompt, rag_context);
            }

            let (code_opt, output) = self.verifier.generate_and_verify(&prompt, lang, 3);
            if let Some(code) = code_opt {
                self.dataset.save_sample(&step_info.description, &code, &output, lang);
                context_accumulator.push_str(&format!("# Step {}\n{}\n", step_num, output.trim()));
                results_formatted.push(format!(
                    "#### Step {}: {}\n```{}\n{}\n```\n**Verified Terminal Output:**\n```\n{}\n```",
                    step_num, step_info.description, lang, code, output.trim()
                ));
            } else {
                overall_success = false;
                results_formatted.push(format!("#### Step {} Failed: {}", step_num, output));
                break;
            }
        }

        let final_response = format!(
            "### MIVI-V2 Pure Rust Execution Results:\n{}",
            results_formatted.join("\n\n")
        );

        if overall_success {
            self.cache.put(request, &final_response).await;
        }

        (overall_success, final_response)
    }
}

