//! Benchmark grammar-constrained token masking (the tool-call hot path).
//! Run: cargo run --release --features native --example grammar_bench
use std::time::Instant;

fn load_grammar(path: &str) -> Option<schoolmarm::GrammarState> {
    let grammar_str = std::fs::read_to_string(path).ok()?;
    let grammar = schoolmarm::Grammar::new(grammar_str.trim()).ok()?;
    schoolmarm::GrammarState::new(grammar).ok()
}

fn main() {
    let tokenizer =
        tokenizers::Tokenizer::from_file("models/qwen2.5-0.5b-tokenizer.json").expect("tokenizer");
    let vocab_size = tokenizer.get_vocab_size(true);
    let mut vocab = vec![String::new(); vocab_size];
    for id in 0..vocab_size {
        if let Some(token) = tokenizer.id_to_token(id as u32) {
            vocab[id] = token;
        }
    }
    let vocab_refs: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
    println!("vocab size: {}", vocab_size);

    for grammar_name in ["json_object.gbnf", "openai_tool_call.gbnf"] {
        let mut state =
            load_grammar(&format!("configs/grammars/{}", grammar_name)).expect("grammar");

        let prefix_tokens = ["\"", "name", "\":", " \"", "Alice", "\"", ","];
        let mut total = std::time::Duration::ZERO;
        let mut positions = 0usize;
        for tok in prefix_tokens {
            let start = Instant::now();
            let mask = state.allowed_tokens(&vocab_refs);
            total += start.elapsed();
            positions += 1;
            let allowed = mask.iter().filter(|&&b| b).count();
            println!(
                "{grammar_name}: pos {positions} allowed={allowed} took {:?}",
                start.elapsed()
            );
            let _ = state.accept_token(tok);
        }
        println!(
            "{grammar_name}: avg per position: {:?}",
            total / positions as u32
        );
    }
}
