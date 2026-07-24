use crate::context_compressor::CompressedContext;
use crate::okf_memory::OkfMemory;
use crate::runtime::ContextBudget;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalPack {
    pub prompt: String,
    pub sources: Vec<String>,
    pub estimated_tokens: usize,
}

pub fn build_retrieval_pack(
    query: &str,
    compressed: &CompressedContext,
    budget: ContextBudget,
) -> RetrievalPack {
    build_retrieval_pack_with_sources(query, compressed, &[], "", budget)
}

pub fn build_retrieval_pack_with_sources(
    query: &str,
    compressed: &CompressedContext,
    memories: &[OkfMemory],
    workspace_rag: &str,
    budget: ContextBudget,
) -> RetrievalPack {
    let max_chars = budget.max_input_tokens * 4;
    let mut sections = Vec::new();
    let mut sources = Vec::new();

    push_section(
        &mut sections,
        &mut sources,
        "current-user",
        "Current user request",
        query,
        budget.recent_turn_tokens * 2,
    );

    push_section(
        &mut sections,
        &mut sources,
        "recent-context",
        "Relevant recent context",
        &compressed.protected_recent.join("\n"),
        budget.recent_turn_tokens * 2,
    );

    push_section(
        &mut sections,
        &mut sources,
        "tool-observations",
        "Tool observations",
        &compressed.tool_observations.join("\n"),
        budget.tool_tokens * 4,
    );

    let memory_text = memories
        .iter()
        .map(|memory| {
            format!(
                "[{}:{}] {}\n{}",
                memory.kind, memory.id, memory.title, memory.body
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    if !memory_text.trim().is_empty() {
        sources.extend(
            memories
                .iter()
                .map(|memory| format!("memory:{}", memory.id)),
        );
        sections.push(format!(
            "OKF memory:\n{}",
            truncate_chars(&memory_text, budget.memory_tokens * 4)
        ));
    }

    if should_include_workspace_rag(query) && !workspace_rag.trim().is_empty() {
        push_section(
            &mut sections,
            &mut sources,
            "workspace-rag",
            "Workspace RAG",
            workspace_rag,
            budget.retrieved_tokens * 4,
        );
    }

    if !compressed.system.trim().is_empty() {
        push_section(
            &mut sections,
            &mut sources,
            "system",
            "System instructions",
            &compressed.system,
            budget.tool_tokens * 4,
        );
    }

    if !compressed.summary.trim().is_empty() {
        push_section(
            &mut sections,
            &mut sources,
            "compression-summary",
            "Compression summary",
            &compressed.summary,
            512,
        );
    }

    let prompt = truncate_chars(&sections.join("\n\n"), max_chars);
    let estimated_tokens = estimate_tokens(&prompt);

    RetrievalPack {
        prompt,
        sources,
        estimated_tokens,
    }
}

fn push_section(
    sections: &mut Vec<String>,
    sources: &mut Vec<String>,
    source: &str,
    title: &str,
    body: &str,
    max_chars: usize,
) {
    if body.trim().is_empty() || max_chars == 0 {
        return;
    }

    sections.push(format!(
        "{}:\n{}",
        title,
        truncate_chars(body.trim(), max_chars)
    ));
    sources.push(source.to_string());
}

fn should_include_workspace_rag(query: &str) -> bool {
    let query = query.to_ascii_lowercase();
    let triggers = [
        "codebase",
        "workspace",
        "repo",
        "repository",
        "project",
        "file",
        "src/",
        "module",
        "function",
        "struct",
        "runtime",
    ];

    triggers.iter().any(|trigger| query.contains(trigger))
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    text.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_compressor::CompressedContext;
    use crate::okf_memory::OkfMemory;
    use crate::runtime::ContextBudget;

    fn compressed() -> CompressedContext {
        CompressedContext {
            system: "Use concise answers.".to_string(),
            protected_recent: vec!["user: inspect router".to_string()],
            tool_observations: vec!["tool: cargo test passed".to_string()],
            summary: "Dropped noisy context.".to_string(),
        }
    }

    fn memory() -> OkfMemory {
        OkfMemory {
            id: "project-main".to_string(),
            title: "Project".to_string(),
            kind: "project".to_string(),
            tags: vec!["mivi".to_string()],
            body: "MIVI exposes only model name mivi.".to_string(),
        }
    }

    #[test]
    fn codebase_prompt_includes_workspace_rag_after_memory() {
        let pack = build_retrieval_pack_with_sources(
            "inspect the codebase router",
            &compressed(),
            &[memory()],
            "src/router.rs: classify_intent implementation",
            ContextBudget::from_max_input_tokens(2048),
        );

        assert!(pack.prompt.contains("Current user request"));
        assert!(pack.prompt.contains("OKF memory"));
        assert!(pack.prompt.contains("Workspace RAG"));
        assert!(
            pack.prompt.find("OKF memory").unwrap() < pack.prompt.find("Workspace RAG").unwrap()
        );
        assert!(pack.sources.contains(&"memory:project-main".to_string()));
        assert!(pack.sources.contains(&"workspace-rag".to_string()));
    }

    #[test]
    fn simple_chat_does_not_include_workspace_rag() {
        let pack = build_retrieval_pack_with_sources(
            "hello how are you",
            &compressed(),
            &[memory()],
            "src/router.rs: classify_intent implementation",
            ContextBudget::from_max_input_tokens(2048),
        );

        assert!(pack.prompt.contains("OKF memory"));
        assert!(!pack.prompt.contains("Workspace RAG"));
        assert!(!pack.sources.contains(&"workspace-rag".to_string()));
    }

    #[test]
    fn retrieval_pack_respects_approximate_budget() {
        let long_memory = OkfMemory {
            body: "x".repeat(10_000),
            ..memory()
        };

        let pack = build_retrieval_pack_with_sources(
            "inspect project files",
            &compressed(),
            &[long_memory],
            &"r".repeat(10_000),
            ContextBudget::from_max_input_tokens(1024),
        );

        assert!(pack.estimated_tokens <= 1024);
        assert!(pack.prompt.len() <= 4096);
    }
}
