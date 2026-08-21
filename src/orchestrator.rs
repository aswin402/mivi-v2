use crate::brain::EdgeBrain;
use crate::cache::SemanticCache;
use crate::logger::DatasetLogger;
use crate::rag::TurboVecRAG;
use crate::router::NeedleRouter;
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
    pub semantic_rag: crate::semantic_rag::SemanticRAG,
    pub router: NeedleRouter,
}

fn has_language_token(text: &str, expected: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric() && c != '+')
        .any(|token| token == expected)
}

fn detect_default_language(request: &str) -> &'static str {
    let req_lower = request.to_lowercase();
    if has_language_token(&req_lower, "javascript") || has_language_token(&req_lower, "js") {
        "javascript"
    } else if has_language_token(&req_lower, "typescript") || has_language_token(&req_lower, "ts") {
        "typescript"
    } else if has_language_token(&req_lower, "rust") || has_language_token(&req_lower, "rs") {
        "rust"
    } else if has_language_token(&req_lower, "cpp") || has_language_token(&req_lower, "c++") {
        "cpp"
    } else {
        "python"
    }
}

fn should_use_rag_context(request: &str) -> bool {
    let req_lower = request.to_lowercase();
    [
        "this project",
        "this repo",
        "repository",
        "codebase",
        "workspace",
        "current file",
        "src/",
        ".rs",
        ".py",
        ".js",
        ".ts",
        "refactor this",
        "explain this",
        "modify this",
    ]
    .iter()
    .any(|needle| req_lower.contains(needle))
}

impl AgentOrchestrator {
    pub fn new(brain: EdgeBrain) -> Self {
        let verifier = CompilerVerifier::new(brain.clone());
        let cache = SemanticCache::new();
        let dataset = DatasetLogger::new();
        let rag = TurboVecRAG::new();
        // Share the keyword RAG's chunk store instead of keeping a second
        // full copy of the workspace index in memory.
        let semantic_rag = crate::semantic_rag::SemanticRAG::with_keyword_rag(rag.clone());
        let router = NeedleRouter::new();
        Self {
            brain,
            verifier,
            cache,
            dataset,
            rag,
            semantic_rag,
            router,
        }
    }

    async fn is_conversational(&self, request: &str) -> bool {
        self.router.classify_intent(&self.brain, request).await.0 == "CHAT"
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
                if let Ok(steps) =
                    serde_json::from_value::<Vec<PlanStep>>(Value::Array(arr.clone()))
                {
                    return Some(steps);
                }
            }
        }

        None
    }

    pub async fn execute_plan(&self, request: &str) -> (bool, String) {
        crate::trace::trace_state_transition("idle", "planning");

        // --- Conversational fast-path ---
        // If the prompt is a chat/QA intent, route directly to the configured reasoner
        // instead of trying to generate + execute code.
        if self.is_conversational(request).await {
            crate::trace::trace_state_transition("planning", "complete");
            println!(
                "[Orchestrator] Conversational prompt detected -> routing to configured reasoner directly"
            );
            let response = self
                .brain
                .query_reasoner(
                    request,
                    "You are a helpful, concise AI assistant. Answer the user's question directly.",
                )
                .await
                .unwrap_or_else(|e| format!("Error: {}", e));
            return (true, response);
        }

        // --- Code execution path (existing logic) ---
        if let Some(cached) = self.cache.get_exact(request).await {
            crate::trace::trace_state_transition("planning", "complete");
            println!("[Orchestrator] Exact cache hit (< 0.001s)!");
            return (true, cached);
        }

        crate::trace::trace_state_transition("planning", "executing");
        println!("[MIVI-V2 Orchestrator] Executing request: '{}'", request);

        let req_lower = request.to_lowercase();
        let is_complex = req_lower.contains("and then")
            || req_lower.contains("after that")
            || req_lower.contains("step 1")
            || req_lower.contains("first")
            || req_lower.contains("multiple steps");

        let default_lang = detect_default_language(request);

        let steps = if is_complex {
            println!("[SAKANA FUGU ROUTER] Complex task detected -> Engaging configured reasoner planner...");
            let system_prompt = "You are the Orchestrator Brain. Break down the user's request into the MINIMAL number of necessary executable coding steps (1 to 3 steps max).\nRespond ONLY with a valid JSON array of step objects inside a ```json ... ``` block.\nEach step object must have keys:\n- 'step': integer\n- 'description': string description of what to write\n- 'language': string ('python' or 'javascript')";

            let plan_opt =
                if let Ok(raw_plan) = self.brain.query_reasoner(request, system_prompt).await {
                    self.extract_json_plan(&raw_plan)
                } else {
                    None
                };

            plan_opt.unwrap_or_else(|| {
                vec![PlanStep {
                    step: Some(1),
                    description: request.to_string(),
                    language: Some(default_lang.to_string()),
                }]
            })
        } else {
            println!("[SAKANA FUGU ROUTER] Simple task detected -> Fast-path direct Qwen Coder (3x faster)...");
            vec![PlanStep {
                step: Some(1),
                description: request.to_string(),
                language: Some(default_lang.to_string()),
            }]
        };

        println!(
            "[Orchestrator] Generated Execution Plan ({} steps):",
            steps.len()
        );

        let mut results_formatted = Vec::new();
        let mut context_accumulator = String::new();
        let mut overall_success = true;

        for (idx, step_info) in steps.iter().enumerate() {
            let step_num = step_info.step.unwrap_or(idx + 1);
            let lang = step_info.language.as_deref().unwrap_or(default_lang);
            println!(
                "[Orchestrator] Running Step {}: {} ({})",
                step_num, step_info.description, lang
            );

            let rag_context = if should_use_rag_context(&step_info.description) {
                self.rag.format_rag_context(&step_info.description, 2).await
            } else {
                String::new()
            };

            let mut prompt = step_info.description.clone();
            if !context_accumulator.is_empty() {
                prompt = format!(
                    "{}\n\nPrevious steps output:\n{}",
                    prompt, context_accumulator
                );
            }
            if !rag_context.is_empty() {
                prompt = format!("{}\n\n{}", prompt, rag_context);
            }

            let (code_opt, output) = self.verifier.generate_and_verify(&prompt, lang, 3).await;
            if let Some(code) = code_opt {
                self.dataset
                    .save_sample(&step_info.description, &code, &output, lang);
                context_accumulator.push_str(&format!("# Step {}\n{}\n", step_num, output.trim()));
                results_formatted.push(format!(
                    "#### Step {}: {}\n```{}\n{}\n```\n**Verified Terminal Output:**\n```\n{}\n```",
                    step_num,
                    step_info.description,
                    lang,
                    code,
                    output.trim()
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
            crate::trace::trace_state_transition("executing", "complete");
            self.cache.put(request, &final_response).await;
        } else {
            crate::trace::trace_state_transition("executing", "failed");
        }

        (overall_success, final_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_language_does_not_match_ts_inside_words() {
        assert_eq!(
            detect_default_language("Write a python script that prints ok"),
            "python"
        );
    }
    #[test]
    fn standalone_code_prompt_does_not_use_rag_context() {
        assert!(!should_use_rag_context(
            "Write a python script that prints ok"
        ));
    }

    #[test]
    fn codebase_prompt_uses_rag_context() {
        assert!(should_use_rag_context(
            "Explain src/server.rs in this project"
        ));
        assert!(should_use_rag_context("Refactor this repository router"));
    }
}
