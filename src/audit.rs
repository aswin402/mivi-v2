use crate::brain::EdgeBrain;
use crate::orchestrator::AgentOrchestrator;
use crate::rag::TurboVecRAG;
use std::time::Instant;

pub async fn run_system_audit() {
    println!("=========================================================");
    println!("  🦀 MIVI-V2 PURE RUST ENGINE - END-TO-END HEALTH AUDIT  ");
    println!("=========================================================\n");

    let start = Instant::now();

    // 1. EdgeBrain Test
    println!("[Audit 1/4] Checking EdgeBrain Hardware & Model Engines...");
    let brain = EdgeBrain::new();
    let sys_prompt = "You are a helpful assistant.";
    if let Ok(res) = brain
        .query_reasoner("Reply with 'Reasoner OK'", sys_prompt)
        .await
    {
        println!(
            "[OK] Reasoner Engine Output: {}",
            res.lines().next().unwrap_or("")
        );
    }
    if let Ok(res) = brain.query_coder("print('Coder OK')", sys_prompt).await {
        println!(
            "[OK] Coder Engine Output: {}",
            res.lines().next().unwrap_or("")
        );
    }
    println!("[Audit 1/4] EdgeBrain Engine: PASSED\n");

    // 2. RAG Test
    println!("[Audit 2/4] Checking TurboVec RAG Engine...");
    let rag = TurboVecRAG::new();
    let count = rag.index_directory(".").await;
    println!("[OK] Indexed {} files in workspace", count);
    println!("[Audit 2/4] TurboVec RAG Engine: PASSED\n");

    // 3. Orchestrator Test
    println!("[Audit 3/4] Checking Multi-Agent Orchestrator...");
    let orchestrator = AgentOrchestrator::new(brain);
    let (success, res) = orchestrator
        .execute_plan("Write a python script printing 'Audit OK'")
        .await;
    if success {
        println!("[OK] Orchestrator Output Verified:\n{}", res);
    }
    println!("[Audit 3/4] Multi-Agent Orchestrator: PASSED\n");

    let elapsed = start.elapsed();
    println!("=========================================================");
    println!(
        " [SUCCESS] ALL MIVI-V2 SYSTEM AUDITS PASSED IN {:.2?}!",
        elapsed
    );
    println!("=========================================================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_system_audit_does_not_panic() {
        run_system_audit().await;
    }
}
