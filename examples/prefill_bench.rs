//! Isolate prefill vs decode cost of NativeBrain (incl. shared-prefix reuse).
//! Run: cargo run --release --features native --example prefill_bench
use mivi::native_brain::NativeBrain;
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let brain = NativeBrain::new();
    let model = Path::new("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");

    let shared_prefix = format!(
        "SYSTEM: You are a tool-calling agent. {}",
        "rule ".repeat(120)
    );
    let turn_a = format!("{} USER: read /tmp/a.txt", shared_prefix);
    let turn_b = format!("{} USER: read /tmp/b.txt", shared_prefix);

    let t = Instant::now();
    let _ = brain
        .query_raw_prompt(model, &turn_a, "0.1", 4, None)
        .unwrap();
    println!("1st turn (full prefill ~600 tok): {:?}", t.elapsed());

    let t = Instant::now();
    let _ = brain
        .query_raw_prompt(model, &turn_b, "0.1", 4, None)
        .unwrap();
    println!("2nd turn (shared-prefix reuse):  {:?}", t.elapsed());

    let t = Instant::now();
    let _ = brain
        .query_raw_prompt(model, &turn_a, "0.1", 4, None)
        .unwrap();
    println!("3rd turn (reuse back to A):      {:?}", t.elapsed());
}
