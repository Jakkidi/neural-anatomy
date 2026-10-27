//! `neural-anatomy --check`: prove in the terminal that the weights are the
//! real GPT-2 and that this engine runs them correctly, without the window.
//!
//! It loads the checkpoint (downloading it first if it is not on disk yet),
//! asks the model to continue a sentence with a well-known answer, and
//! measures how well it predicts a paragraph of plain English. Random or
//! wrongly wired weights fail both, visibly.

use crate::{download, paths, Assets};
use na_engine::{Engine, Json, RunEvent};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::time::Instant;

/// GPT-2 small, as in `src/models/index.js`.
const CARD: &str = r#"{
  "id": "gpt2", "arch": "gpt2",
  "config": {"n_layer": 12, "n_head": 12, "n_embd": 768, "n_inner": 3072, "n_ctx": 1024, "vocab_size": 50257, "layer_norm_epsilon": 1e-5},
  "tokenizer": {"vocab": "assets/gpt2/vocab.json", "merges": "assets/gpt2/merges.txt"},
  "weights": {"kind": "safetensors", "file": "model.safetensors", "bytes": 548105171,
              "url": "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"}
}"#;

const PROMPT: &str = "The Eiffel Tower is located in the city of";
const PROSE: &str = "The river runs through the middle of the town, and on warm evenings people walk along its banks. There are two bridges, an old stone one near the market and a newer one by the railway station. In winter the water rises, and the path under the old bridge is closed for a few weeks.";

fn megabytes(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / 1e6)
}

/// Runs the check and prints what it finds. `Ok(true)` when the weights pass.
pub fn run(data_dir: &Path, file: Option<PathBuf>) -> Result<bool, String> {
    let card = Json::parse(CARD).expect("the built-in card is valid");
    let assets = Assets::Embedded;
    let read = |path: &str| -> Result<String, String> {
        let bytes = assets.get(path).ok_or_else(|| format!("The app is missing its file {}.", path))?;
        String::from_utf8(bytes.into_owned()).map_err(|_| format!("The app's file {} is not text.", path))
    };

    let path = match file {
        Some(path) => path,
        None => {
            let path = data_dir.join("models").join(paths::safe_name("gpt2")).join("model.safetensors");
            if !path.is_file() {
                let url = card.path(&["weights", "url"]).and_then(Json::as_str).unwrap_or_default();
                let expected = card.path(&["weights", "bytes"]).and_then(Json::as_usize).unwrap_or(0) as u64;
                println!("Downloading GPT-2 small from {}", url);
                let mut shown = Instant::now();
                download::download(url, &path, expected, &mut |have, total| {
                    if shown.elapsed().as_millis() > 500 {
                        shown = Instant::now();
                        print!("\r  {} of {}   ", megabytes(have), megabytes(total));
                        let _ = std::io::stdout().flush();
                    }
                })
                .map_err(|message| format!("\n{}\nFrom the terminal, that is: neural-anatomy --check path/to/model.safetensors", message))?;
                println!("\r  saved to {}", path.display());
            }
            path
        }
    };

    println!("Weights: {}", path.display());
    let mut engine = Engine::new();
    let started = Instant::now();
    let mut reader = std::fs::File::open(&path).map_err(|error| format!("Could not open {}: {}.", path.display(), error))?;
    let info = engine.load(&card, &mut reader, "saved copy", &read, &mut |_, _| {})?;
    println!(
        "Loaded {} weights in {:.1} s. Engine: Rust, {} threads, {}.",
        info.params,
        started.elapsed().as_secs_f64(),
        na_engine::kernels::threads(),
        na_engine::kernels::backend()
    );
    let size = |name: &str| info.config.get(name).and_then(Json::as_usize).unwrap_or(0);
    println!("Shape: {} blocks, {} heads, width {}, {} tokens in the vocabulary.", size("n_layer"), size("n_head"), size("n_embd"), size("vocab_size"));
    let latest = AtomicU64::new(0);

    // 1. a sentence with a well-known continuation
    let (ids, _) = engine.tokenize(&card, PROMPT, &read)?;
    let mut last = None;
    let result = engine.run(&ids, true, 0, &latest, &mut |event| {
        if let RunEvent::Prediction(prediction) = event {
            if prediction.position + 1 == ids.len() {
                last = Some(prediction);
            }
        }
    })?;
    let prediction = last.ok_or("The model gave no prediction.")?;
    println!();
    println!("\u{201c}{}\u{201d} ({} tokens, forward pass {:.0} ms). The model continues with:", PROMPT, ids.len(), result.forward_ms);
    for (piece, prob) in prediction.pieces.iter().zip(&prediction.summary.probs).take(5) {
        println!("  {:>6.2}%  {:?}", prob * 100.0, piece);
    }
    let paris = prediction.pieces.first().map(|piece| piece.trim() == "Paris").unwrap_or(false);

    // 2. how surprised it is by ordinary prose
    let (ids, _) = engine.tokenize(&card, PROSE, &read)?;
    let (mut nats, mut count) = (0f64, 0usize);
    engine.run(&ids, true, 0, &latest, &mut |event| {
        if let RunEvent::Prediction(prediction) = event {
            if let Some(prob) = prediction.summary.target_prob {
                nats -= (prob.max(1e-30) as f64).ln();
                count += 1;
            }
        }
    })?;
    if count == 0 {
        return Err("There was nothing to measure.".to_string());
    }
    let mean = nats / count as f64;
    let chance = (engine.info().and_then(|info| info.config.get("vocab_size").and_then(Json::as_usize)).unwrap_or(2) as f64).ln();
    let fluent = mean < chance * 0.6;
    println!();
    println!("On {} tokens of plain English it is surprised by {:.2} nats per token.", count, mean);
    println!("A trained GPT-2 small scores about 3 to 4; random or wrongly wired weights score about {:.1}.", chance);

    println!();
    if fluent && paris {
        println!("PASS: these are trained weights and the engine runs them correctly.");
    } else if fluent {
        println!("PASS, with a note: the model reads English well, but its first choice above is not \u{201c} Paris\u{201d}.");
        println!("That is expected for a GPT-2 of another size or a fine-tuned one.");
    } else {
        println!("FAIL: this is not a trained model, or it is not being run correctly.");
    }
    Ok(fluent)
}
