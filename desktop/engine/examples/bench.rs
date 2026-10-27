//! Speed of the engine at GPT-2 small's size, on random weights.
//!
//!   cargo run --release --example bench

use na_engine::{Engine, Json};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

fn main() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |path: &str| std::fs::read_to_string(repo.join(path)).map_err(|e| e.to_string());
    let card = Json::parse(
        r#"{"id":"gpt2","arch":"gpt2","config":{"n_layer":12,"n_head":12,"n_embd":768,"n_inner":3072,"n_ctx":1024,"vocab_size":50257,"layer_norm_epsilon":1e-5},
            "tokenizer":{"vocab":"assets/gpt2/vocab.json","merges":"assets/gpt2/merges.txt"}}"#,
    )
    .unwrap();
    let mut engine = Engine::new();
    let started = Instant::now();
    let info = engine.load_random(&card, 1, &read).unwrap();
    println!("{} weights generated in {:.1} s; {} threads, {}", info.params, started.elapsed().as_secs_f64(), na_engine::kernels::threads(), na_engine::kernels::backend());
    let text = "The Eiffel Tower is located in the city of Paris, which is the capital of France and one of the most visited places in the whole wide world today, and has been for a long time now.";
    let (ids, _) = engine.tokenize(&card, text, &read).unwrap();
    let latest = AtomicU64::new(0);
    for n in [8usize, 16, 32] {
        let result = engine.run(&ids[..n], true, 0, &latest, &mut |_| {}).unwrap();
        println!("{:3} tokens: forward {:.0} ms, with predictions for every position {:.0} ms", n, result.forward_ms, result.total_ms);
    }
    let result = engine.run(&ids[32..33], false, 0, &latest, &mut |_| {}).unwrap();
    println!("append one token: forward {:.1} ms, total {:.1} ms", result.forward_ms, result.total_ms);
}
