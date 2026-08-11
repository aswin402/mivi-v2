use mivi::audit::run_system_audit;
use mivi::brain::EdgeBrain;
use mivi::chat::run_chat_interactive;
use mivi::cli::run_cli;
use mivi::model_catalog::{print_model_inspect, print_model_list, ModelCatalog};
use mivi::orchestrator::AgentOrchestrator;
use mivi::server::start_api_server;
use std::env;

fn handle_model_command(args: &[String]) -> Result<(), String> {
    let catalog = ModelCatalog::load_default().map_err(|err| err.to_string())?;
    match args.first().map(|value| value.as_str()) {
        Some("list") => {
            print_model_list(&catalog);
            Ok(())
        }
        Some("inspect") => {
            let id = args
                .get(1)
                .ok_or_else(|| "Usage: mivi model inspect <internal-id>".to_string())?;
            print_model_inspect(&catalog, id).map_err(|err| err.to_string())
        }
        Some("help") | None => {
            print_model_usage();
            Ok(())
        }
        Some(other) => Err(format!(
            "unknown model command '{other}'. Usage: mivi model <list|inspect>"
        )),
    }
}

fn print_model_usage() {
    println!("Usage:");
    println!("  mivi model list");
    println!("  mivi model inspect <internal-id>");
    println!("\nAgents should call only the external OpenAI-compatible model name: mivi");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "mivi=info".to_string()))
        .init();

    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("serve");

    println!("=========================================================");
    println!("  🚀 MIVI-V2: PURE RUST LOW-RESOURCE LOCAL AI ENGINE");
    println!("  RAM Footprint: < 12 MB Server RAM | 0 MB Idle");
    println!("  Version: {} (Pure Rust)", env!("CARGO_PKG_VERSION"));
    println!("=========================================================\n");

    if mode == "model" {
        if let Err(err) = handle_model_command(&args[2..]) {
            eprintln!("Error: {err}");
            std::process::exit(1);
        }
        return;
    }

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
            #[cfg(target_os = "linux")]
            {
                tracing::info!("Cleaning up any orphaned llama-server processes...");
                let _ = std::process::Command::new("pkill")
                    .arg("-f")
                    .arg("llama-server")
                    .status();
            }
            if let Err(e) = start_api_server(brain, orchestrator, 8000).await {
                eprintln!("Fatal error: {}", e);
                std::process::exit(1);
            }
        }
    }
}
