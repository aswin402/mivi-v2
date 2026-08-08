use crate::brain::EdgeBrain;
use crate::router::NeedleRouter;
use colored::*;
use inquire::Text;

/// Run a pure interactive chat session (no orchestrator, no code execution).
/// Maintains conversation history across turns so MIVI feels like a normal AI.
pub async fn run_chat_interactive(brain: EdgeBrain, router: NeedleRouter) {
    println!(
        "{}",
        "=========================================================".cyan()
    );
    println!(
        "{}",
        "  🗣️  MIVI-V2 INTERACTIVE CHAT (PURE CONVERSATIONAL)"
            .bold()
            .green()
    );
    println!("{}", "  I can answer questions, explain concepts, write code, or just chat. Type 'exit' to stop.".yellow());
    println!(
        "{}",
        "=========================================================\n".cyan()
    );

    let mut history: Vec<(String, String)> = Vec::new(); // (user_msg, assistant_reply)

    loop {
        let input = match Text::new("You>").prompt() {
            Ok(i) => i,
            Err(_) => break,
        };

        let prompt = input.trim().to_string();
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            println!("{}", "Goodbye!".bright_blue());
            break;
        }

        // Build context from history
        let history_context = build_history_context(&history);

        // Classify intent
        let (intent, confidence) = router.classify_intent(&prompt);
        println!(
            "{} {}",
            "[MIVI]".dimmed(),
            format!("Intent: {} (conf: {:.2})", intent, confidence).dimmed()
        );

        let response = match intent {
            "VISION" => {
                // Ask for image path
                let img_path = match Text::new("Image path>").prompt() {
                    Ok(p) => p.trim().to_string(),
                    Err(_) => continue,
                };
                if img_path.is_empty() {
                    println!("{}", "  (No image provided, skipping vision)".yellow());
                    continue;
                }
                match brain.query_vision(&img_path, &prompt).await {
                    Ok(res) => res,
                    Err(e) => format!("Vision error: {}", e),
                }
            }
            "CODE" | "MULTI_STEP" => {
                // Use Qwen coder model for code requests
                let system_prompt = format!(
                    "You are MIVI, a helpful coding assistant. You write clean, working code.\n\nConversation history:\n{}",
                    if history_context.is_empty() { "No prior conversation.".to_string() } else { history_context }
                );
                brain
                    .query_coder(&prompt, &system_prompt)
                    .await
                    .unwrap_or_else(|e| format!("Error: {}", e))
            }
            _ => {
                // Default: use the configured reasoner model (conversational)
                let system_prompt = format!(
                    "You are MIVI-V2, a helpful, concise AI assistant. Be friendly and informative.\n\nConversation history:\n{}",
                    if history_context.is_empty() { "No prior conversation yet.".to_string() } else { history_context }
                );
                brain
                    .query_reasoner(&prompt, &system_prompt)
                    .await
                    .unwrap_or_else(|e| format!("Error: {}", e))
            }
        };

        println!(
            "\n{}",
            format!("{} {}", "MIVI>".bold().green(), response).bright_white()
        );
        println!();

        // Store in history
        history.push((prompt, response));
    }
}

fn build_history_context(history: &[(String, String)]) -> String {
    if history.is_empty() {
        return String::new();
    }
    let mut ctx = String::new();
    for (i, (user, asst)) in history.iter().enumerate() {
        ctx.push_str(&format!(
            "Turn {}:\nUser: {}\nAssistant: {}\n\n",
            i + 1,
            user,
            asst
        ));
    }
    ctx
}
