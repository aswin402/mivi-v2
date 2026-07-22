use crate::orchestrator::AgentOrchestrator;
use colored::*;
use inquire::Text;

pub async fn run_cli(orchestrator: AgentOrchestrator) {
    println!("{}", "=========================================================".cyan());
    println!("{}", "  🚀 MIVI-V2 INTERACTIVE TERMINAL CHAT CLI (PURE RUST)".bold().green());
    println!("{}", "  Type 'exit' or 'quit' to exit.".yellow());
    println!("{}", "=========================================================\n".cyan());

    loop {
        match Text::new("MIVI-V2>").prompt() {
            Ok(input) => {
                let prompt = input.trim();
                if prompt.is_empty() {
                    continue;
                }
                if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
                    println!("{}", "Goodbye!".bright_blue());
                    break;
                }

                println!("{}", "[MIVI-V2 Engine Executing...]".dimmed());
                let (success, output) = orchestrator.execute_plan(prompt).await;
                if success {
                    println!("\n{}\n", output.bold());
                } else {
                    println!("\n{}\n", format!("[Error] {}", output).red());
                }
            }
            Err(_) => break,
        }
    }
}
