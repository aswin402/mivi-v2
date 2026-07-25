use mivi::audit::run_system_audit;
use mivi::brain::EdgeBrain;
use mivi::chat::run_chat_interactive;
use mivi::cli::run_cli;
use mivi::orchestrator::AgentOrchestrator;
use mivi::server::start_api_server;
use std::env;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("serve");

    println!("=========================================================");
    println!("  🚀 MIVI-V2: PURE RUST LOW-RESOURCE LOCAL AI ENGINE");
    println!("  RAM Footprint: < 12 MB Server RAM | 0 MB Idle");
    println!("  Version: {} (Pure Rust)", env!("CARGO_PKG_VERSION"));
    println!("=========================================================\n");

    let brain = EdgeBrain::new();
    let orchestrator = AgentOrchestrator::new(brain.clone());

    let cur_dir = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    orchestrator
        .rag
        .index_directory(&cur_dir.display().to_string())
        .await;

    match mode {
        "audit" => {
            run_system_audit().await;
        }
        "cli" => {
            run_cli(orchestrator).await;
        }
        "chat" => {
            run_chat_interactive(brain, orchestrator.router.clone()).await;
        }
        "task" => {
            if let Some(prompt) = args.get(2) {
                let (_, output) = orchestrator.execute_plan(prompt).await;
                println!("\n{}\n", output);
            } else {
                println!("Error: Task prompt is required. Usage: mivi task \"your prompt\"");
            }
        }
        _ => {
            start_api_server(brain, orchestrator, 8000).await;
        }
    }
}
