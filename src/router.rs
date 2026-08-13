use std::collections::{HashMap, HashSet};

const CLASS_NAMES: &[&str] = &["CHAT", "VISION", "CODE", "MULTI_STEP"];

const KEYWORD_RULES: &[(&str, &[&str])] = &[
    (
        "VISION",
        &[
            "image", "photo", "picture", "look at", "see", "png", "jpg", "jpeg", "gif",
        ],
    ),
    (
        "CODE",
        &[
            "code",
            "write",
            "script",
            "function",
            "implement",
            "python",
            "rust",
            "program",
            "sql",
            "debug",
            "test",
            "api",
            "algorithm",
        ],
    ),
    (
        "MULTI_STEP",
        &[
            "step",
            "first",
            "then",
            "after that",
            "multiple",
            "each",
            "process all",
            "for every",
        ],
    ),
];

const TRAINING_DATA: &[(&str, &str)] = &[
    // CHAT (35 examples)
    ("hello how are you", "CHAT"),
    ("what is the weather today", "CHAT"),
    ("tell me a joke", "CHAT"),
    ("good morning", "CHAT"),
    ("what do you think about", "CHAT"),
    ("can you explain", "CHAT"),
    ("i need help with", "CHAT"),
    ("give me advice", "CHAT"),
    ("tell me about", "CHAT"),
    ("how does this work", "CHAT"),
    ("what is your name", "CHAT"),
    ("thanks for the help", "CHAT"),
    ("yes that makes sense", "CHAT"),
    ("can you clarify", "CHAT"),
    ("how are you doing", "CHAT"),
    ("what's up", "CHAT"),
    ("are you available", "CHAT"),
    ("could you help me", "CHAT"),
    ("i have a question", "CHAT"),
    ("what do you recommend", "CHAT"),
    ("explain this concept", "CHAT"),
    ("what is the meaning of", "CHAT"),
    ("do you understand", "CHAT"),
    ("let's discuss", "CHAT"),
    ("what are your thoughts", "CHAT"),
    (
        "what module handles intent routing in this codebase",
        "CHAT",
    ),
    (
        "explain the safest fix for a failed tool in codebase",
        "CHAT",
    ),
    ("cargo cache is corrupted error fix", "CHAT"),
    (
        "who designed the google deepmind team working on advanced agentic coding",
        "CHAT",
    ),
    ("what is the local server port default value", "CHAT"),
    ("give me a summary of current tasks status", "CHAT"),
    ("why does this error occur in my runtime config", "CHAT"),
    ("how can i retrieve okf memory tag matches", "CHAT"),
    ("can you translate this sentence into spanish", "CHAT"),
    ("what are the core guidelines for rust skills", "CHAT"),
    ("so whats new", "CHAT"),
    ("hey whats new", "CHAT"),
    ("whats new in your version", "CHAT"),
    ("whats your name", "CHAT"),
    ("what is your name", "CHAT"),
    ("whats my name", "CHAT"),
    ("what is my name", "CHAT"),
    ("who are you", "CHAT"),
    ("hey how are you", "CHAT"),
    ("hi there", "CHAT"),
    ("hii", "CHAT"),
    ("hey", "CHAT"),
    ("sup", "CHAT"),
    ("yo", "CHAT"),
    ("can you access the internet", "CHAT"),
    ("do you have internet access", "CHAT"),
    ("what can you do", "CHAT"),
    ("how do you work", "CHAT"),
    ("what are you", "CHAT"),
    ("tell me about yourself", "CHAT"),
    ("research about this", "CHAT"),
    ("tell me about this link", "CHAT"),
    ("what is this website", "CHAT"),
    ("check this url", "CHAT"),
    // VISION (25 examples)
    ("look at this image", "VISION"),
    ("describe this photo", "VISION"),
    ("what do you see in this picture", "VISION"),
    ("analyze this image", "VISION"),
    ("what is shown in the picture", "VISION"),
    ("read the text in this image", "VISION"),
    ("identify objects in this photo", "VISION"),
    ("what does this image contain", "VISION"),
    ("what's in this screenshot", "VISION"),
    ("can you see this picture", "VISION"),
    ("describe what you see", "VISION"),
    ("what colors are in this image", "VISION"),
    ("is there a person in this photo", "VISION"),
    ("what is the text on screen", "VISION"),
    ("take a look at this diagram", "VISION"),
    ("what are the ui elements in this mock up design", "VISION"),
    ("can you find the logo in this picture file", "VISION"),
    ("read the whiteboard text in this photo", "VISION"),
    (
        "check if the graph layout in this image is correct",
        "VISION",
    ),
    ("describe the chart shown in this png", "VISION"),
    ("extract table data from this screenshot jpeg", "VISION"),
    (
        "tell me what architecture is shown in the diagram image",
        "VISION",
    ),
    (
        "analyze the user interface from this design screenshot",
        "VISION",
    ),
    (
        "is the text readable in this low resolution picture",
        "VISION",
    ),
    ("what object is at the top left of this photo", "VISION"),
    // CODE (30 examples)
    ("write a python script", "CODE"),
    ("create a function that sorts", "CODE"),
    ("implement a fibonacci sequence", "CODE"),
    ("write code to calculate", "CODE"),
    ("create a rust program", "CODE"),
    ("fix this bug", "CODE"),
    ("generate a sql query", "CODE"),
    ("write a bash script", "CODE"),
    ("implement a sorting algorithm", "CODE"),
    ("debug this code", "CODE"),
    ("convert this to typescript", "CODE"),
    ("write a unit test", "CODE"),
    ("create an api endpoint", "CODE"),
    ("build a web scraper", "CODE"),
    ("optimize this function", "CODE"),
    ("refactor this module", "CODE"),
    ("write a dockerfile", "CODE"),
    ("create a react component", "CODE"),
    ("implement authentication", "CODE"),
    ("parse this json response", "CODE"),
    ("compile this rust code and fix compile errors", "CODE"),
    ("write a handler for http post requests in axum", "CODE"),
    (
        "implement custom json schema deserializer in helpers",
        "CODE",
    ),
    (
        "fix syntax error unexpected closing delimiter in compiler output",
        "CODE",
    ),
    ("add a test case to verify schema validation rules", "CODE"),
    ("write a recursive tokenizer using candle core", "CODE"),
    (
        "generate git diff between branch main and current dev",
        "CODE",
    ),
    (
        "how to parse arguments dynamically from command line in rust",
        "CODE",
    ),
    ("implement sliding window rate limiter state", "CODE"),
    (
        "refactor verifier module to repair python type errors",
        "CODE",
    ),
    // MULTI_STEP (25 examples)
    ("first do this then that", "MULTI_STEP"),
    ("do step 1 and step 2", "MULTI_STEP"),
    ("then after that execute", "MULTI_STEP"),
    ("follow these steps", "MULTI_STEP"),
    ("first calculate then verify", "MULTI_STEP"),
    ("do this multiple times", "MULTI_STEP"),
    ("for each item run", "MULTI_STEP"),
    ("process all files", "MULTI_STEP"),
    ("first create then test", "MULTI_STEP"),
    ("step by step process", "MULTI_STEP"),
    ("repeat this operation", "MULTI_STEP"),
    ("iterate over all items", "MULTI_STEP"),
    ("perform these actions in order", "MULTI_STEP"),
    ("execute the following sequence", "MULTI_STEP"),
    ("run these commands sequentially", "MULTI_STEP"),
    (
        "first pull repo then build then run test suite",
        "MULTI_STEP",
    ),
    (
        "create a plan to refactor code then execute and check results",
        "MULTI_STEP",
    ),
    (
        "for every file in directory run grep and count occurrences",
        "MULTI_STEP",
    ),
    (
        "run compile phase then verifier checks then commit changes",
        "MULTI_STEP",
    ),
    (
        "first download model weights then run native server then audit",
        "MULTI_STEP",
    ),
    (
        "do a full sweep of codebase locate todos then fix them in sequence",
        "MULTI_STEP",
    ),
    (
        "step 1 parse response step 2 validate fields step 3 print output",
        "MULTI_STEP",
    ),
    (
        "run code check then if it fails fix it and check again",
        "MULTI_STEP",
    ),
    (
        "initialize repository build initial structure and setup ci cd pipeline",
        "MULTI_STEP",
    ),
    (
        "execute benchmark suite check scores compare with baseline and save stats",
        "MULTI_STEP",
    ),
];

/// Pure Rust Naive Bayes classifier for prompt intent routing.
/// Uses term-frequency log-probabilities with Laplace smoothing.
#[derive(Clone)]
pub struct NeedleRouter {
    class_log_probs: HashMap<String, f64>,
    word_log_probs: HashMap<String, HashMap<String, f64>>,
}

impl NeedleRouter {
    pub fn new() -> Self {
        let mut vocab_set: HashSet<String> = HashSet::new();
        let mut class_docs: HashMap<String, Vec<Vec<String>>> = HashMap::new();

        for &(text, class) in TRAINING_DATA {
            let tokens: Vec<String> = tokenize(text);
            for t in &tokens {
                vocab_set.insert(t.clone());
            }
            class_docs
                .entry(class.to_string())
                .or_default()
                .push(tokens);
        }

        let mut vocab: Vec<String> = vocab_set.into_iter().collect();
        vocab.sort();
        let vocab_size = vocab.len() as f64;
        let total_docs = TRAINING_DATA.len() as f64;

        let mut class_counts: HashMap<String, f64> = HashMap::new();
        for (_, class) in TRAINING_DATA {
            *class_counts.entry(class.to_string()).or_insert(0.0) += 1.0;
        }

        let mut class_log_probs: HashMap<String, f64> = HashMap::new();
        for (class, count) in &class_counts {
            class_log_probs.insert(class.clone(), (count / total_docs).ln());
        }

        let mut word_class_counts: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for (class, docs) in &class_docs {
            let mut counts: HashMap<String, f64> = HashMap::new();
            for doc in docs {
                for token in doc {
                    *counts.entry(token.clone()).or_insert(0.0) += 1.0;
                }
            }
            word_class_counts.insert(class.clone(), counts);
        }

        let mut word_log_probs: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for (class, counts) in &word_class_counts {
            let total_words: f64 = counts.values().sum();
            let mut log_probs: HashMap<String, f64> = HashMap::new();
            for word in &vocab {
                let count = counts.get(word).copied().unwrap_or(0.0);
                let prob = (count + 1.0) / (total_words + vocab_size);
                log_probs.insert(word.clone(), prob.ln());
            }
            word_log_probs.insert(class.clone(), log_probs);
        }

        Self {
            class_log_probs,
            word_log_probs,
        }
    }

    /// Classify prompt intent using the Naive Bayes classifier.
    pub fn classify_intent_nb(&self, prompt: &str) -> (&'static str, f64) {
        if prompt.is_empty() {
            return ("CHAT", 1.0);
        }

        if is_agent_context_chat(prompt) {
            return ("CHAT", 1.0);
        }

        let tokens = tokenize(prompt);

        if tokens.len() < 3 {
            let class = keyword_classify(prompt);
            return (class, 1.0);
        }

        let mut scores: Vec<(&str, f64)> = CLASS_NAMES
            .iter()
            .map(|&class| {
                let log_prior = self
                    .class_log_probs
                    .get(class)
                    .copied()
                    .unwrap_or(0.0f64.ln());
                let mut score = log_prior;
                if let Some(word_probs) = self.word_log_probs.get(class) {
                    for token in &tokens {
                        if let Some(lp) = word_probs.get(token) {
                            score += lp;
                        }
                    }
                }
                (class, score)
            })
            .collect();

        // Find best class
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best_class, best_score) = scores[0];

        // Convert log-probs to probabilities using softmax
        let max_score = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = scores.iter().map(|(_, s)| (s - max_score).exp()).sum();
        let confidence = (best_score - max_score).exp() / sum_exp;

        (best_class, confidence)
    }

    /// Classify prompt intent with a hybrid approach: Naive Bayes first, and
    /// falls back to querying the coder model if confidence is low.
    pub async fn classify_intent(
        &self,
        brain: &crate::brain::EdgeBrain,
        prompt: &str,
    ) -> (&'static str, f64) {
        let (best_class, confidence) = self.classify_intent_nb(prompt);

        // In ultra-low-RAM mode, skip model fallback entirely to avoid
        // loading the 491 MB model just for intent classification.
        // The Naive Bayes classifier is good enough for routing.
        let ultra_low = std::env::var("MIVI_ULTRA_LOW_RAM")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);

        if ultra_low {
            tracing::debug!(
                "[NeedleRouter] Ultra-low-RAM: using NB classification directly (class={}, conf={:.2})",
                best_class, confidence
            );
            return (best_class, confidence.max(0.70));
        }

        if confidence < 0.70 {
            tracing::info!(
                "[NeedleRouter] Low confidence ({:.2}) for class {}. Falling back to Coder model router...",
                confidence, best_class
            );
            if let Ok(model_intent) = self.classify_intent_model(brain, prompt).await {
                tracing::info!(
                    "[NeedleRouter] Coder model classified intent as: {}",
                    model_intent
                );
                return (model_intent, 1.0);
            }
        }

        (best_class, confidence)
    }

    async fn classify_intent_model(
        &self,
        brain: &crate::brain::EdgeBrain,
        prompt: &str,
    ) -> Result<&'static str, String> {
        let system_prompt = "You are an intent router. Classify the user prompt into exactly one category: CHAT, VISION, CODE, MULTI_STEP. Output only the category name.";
        let response = brain.query_reasoner(prompt, system_prompt).await?;
        let cleaned = response.trim().to_uppercase();
        if cleaned.contains("VISION") {
            Ok("VISION")
        } else if cleaned.contains("CODE") {
            Ok("CODE")
        } else if cleaned.contains("MULTI_STEP") || cleaned.contains("MULTI") {
            Ok("MULTI_STEP")
        } else {
            Ok("CHAT")
        }
    }
}

fn is_agent_context_chat(prompt: &str) -> bool {
    let p = prompt.to_lowercase();
    let visual_terms = [
        "image",
        "photo",
        "picture",
        "screenshot",
        "diagram",
        "jpeg",
        "jpg",
        "png",
        "gif",
    ];
    let explicit_visual_request = visual_terms.iter().any(|term| p.contains(term))
        && ["look", "see", "describe", "analyze", "shown", "contains"]
            .iter()
            .any(|term| p.contains(term));

    if explicit_visual_request {
        return false;
    }

    let agent_context_terms = ["project memory", "agent context"];

    agent_context_terms.iter().any(|term| p.contains(term))
}

fn keyword_classify(prompt: &str) -> &'static str {
    let p = prompt.to_lowercase();
    for &(class, patterns) in KEYWORD_RULES {
        for pat in patterns {
            if p.contains(pat) {
                return class;
            }
        }
    }
    "CHAT"
}

fn tokenize(text: &str) -> Vec<String> {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_classification() {
        let router = NeedleRouter::new();
        assert_eq!(router.classify_intent_nb("hello how are you").0, "CHAT");
        assert_eq!(router.classify_intent_nb("tell me a joke").0, "CHAT");
        assert_eq!(router.classify_intent_nb("what is the weather").0, "CHAT");
        assert_eq!(router.classify_intent_nb("").0, "CHAT");
    }

    #[test]
    fn test_vision_classification() {
        let router = NeedleRouter::new();
        assert_eq!(router.classify_intent_nb("look at this image").0, "VISION");
        assert_eq!(router.classify_intent_nb("describe this photo").0, "VISION");
        assert_eq!(
            router
                .classify_intent_nb("what do you see in this picture")
                .0,
            "VISION"
        );
    }

    #[test]
    fn test_code_classification() {
        let router = NeedleRouter::new();
        assert_eq!(router.classify_intent_nb("write a python script").0, "CODE");
        assert_eq!(
            router.classify_intent_nb("implement a sorting algorithm").0,
            "CODE"
        );
        assert_eq!(
            router.classify_intent_nb("create a function that sorts").0,
            "CODE"
        );
    }

    #[test]
    fn test_multi_step_classification() {
        let router = NeedleRouter::new();
        assert_eq!(
            router.classify_intent_nb("first do this then that").0,
            "MULTI_STEP"
        );
        assert_eq!(
            router.classify_intent_nb("follow these steps").0,
            "MULTI_STEP"
        );
        assert_eq!(
            router.classify_intent_nb("process all files").0,
            "MULTI_STEP"
        );
    }

    #[test]
    fn test_short_prompt_keyword_fallback() {
        let router = NeedleRouter::new();
        assert_eq!(router.classify_intent_nb("write code").0, "CODE");
        assert_eq!(router.classify_intent_nb("debug").0, "CODE");
        assert_eq!(router.classify_intent_nb("look at this").0, "VISION");
        assert_eq!(router.classify_intent_nb("step by step").0, "MULTI_STEP");
    }

    #[test]
    fn test_confidence_values() {
        let router = NeedleRouter::new();
        let (_, conf) = router.classify_intent_nb("hello how are you");
        assert!(
            conf > 0.5 && conf <= 1.0,
            "confidence should be >0.5 and <=1.0, got {}",
            conf
        );

        let (_, conf) = router.classify_intent_nb("write a python script");
        assert!(
            conf > 0.5 && conf <= 1.0,
            "confidence should be >0.5 and <=1.0, got {}",
            conf
        );

        let (_, conf) = router.classify_intent_nb("look at this image");
        assert!(
            conf > 0.5 && conf <= 1.0,
            "confidence should be >0.5 and <=1.0, got {}",
            conf
        );
    }

    #[test]
    fn test_confidence_normalized() {
        let router = NeedleRouter::new();
        let (class, conf) = router.classify_intent_nb("what is the weather today");
        assert_eq!(class, "CHAT");
        // Confidence should be a valid probability
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence {} out of range",
            conf
        );
    }

    #[test]
    fn text_only_agent_context_prompts_do_not_route_to_vision() {
        let router = NeedleRouter::new();

        assert_eq!(
            router
                .classify_intent_nb("In this codebase, what module handles intent routing?")
                .0,
            "CHAT"
        );
        assert_eq!(
            router
                .classify_intent_nb("A tool failed because Cargo cache is corrupted. Explain the safest fix in two steps.")
                .0,
            "CHAT"
        );
    }

    #[test]
    fn test_mixed_known_tokens_keep_confidence_finite() {
        let router = NeedleRouter::new();
        let (_, conf) = router.classify_intent_nb("write hello image");
        assert!(
            conf.is_finite(),
            "confidence should be finite, got {}",
            conf
        );
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence {} out of range",
            conf
        );
    }
}
