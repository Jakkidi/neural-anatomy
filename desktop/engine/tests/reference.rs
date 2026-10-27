//! Checks the engine against the independent numpy reference
//! (`tools/reference_gpt2.py`) on a GPT-2-shaped checkpoint of random numbers.
//! Every tensor is random, biases and LayerNorm gains included, so any wiring
//! mistake in the forward pass changes the output.
//!
//! Needs python3 with numpy. Without them the comparison is skipped, loudly.

use na_engine::safetensors;
use na_engine::sampling::Random;
use na_engine::{Engine, Json, RunEvent, View};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicU64;

const LAYERS: usize = 3;
const HEADS: usize = 4;
const WIDTH: usize = 64;
const VOCAB: usize = 600;
const CTX: usize = 48;
const IDS: [u32; 11] = [5, 17, 301, 42, 599, 0, 77, 301, 12, 256, 9];

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_asset(path: &str) -> Result<String, String> {
    std::fs::read_to_string(repo().join(path)).map_err(|error| format!("{}: {}", path, error))
}

fn card() -> Json {
    Json::parse(&format!(
        r#"{{"id":"tiny","arch":"gpt2","config":{{"n_layer":{LAYERS},"n_head":{HEADS},"n_embd":{WIDTH},"n_inner":{},"n_ctx":{CTX},"vocab_size":{VOCAB},"layer_norm_epsilon":1e-5}},
            "tokenizer":{{"vocab":"assets/gpt2/vocab.json","merges":"assets/gpt2/merges.txt"}}}}"#,
        4 * WIDTH
    ))
    .unwrap()
}

/// A checkpoint as Hugging Face writes one: module prefix, mask buffers and all.
fn checkpoint() -> Vec<u8> {
    let mut random = Random::new(3);
    let mut gauss = |count: usize, std: f64, mean: f64| -> Vec<f32> {
        (0..count)
            .map(|_| {
                let (u, v) = (random.next().max(1e-12), random.next());
                (mean + (-2.0 * u.ln()).sqrt() * std * (std::f64::consts::TAU * v).cos()) as f32
            })
            .collect()
    };
    let (d, m) = (WIDTH, 4 * WIDTH);
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        ("transformer.wte.weight".into(), vec![VOCAB, d], gauss(VOCAB * d, 0.08, 0.0)),
        ("transformer.wpe.weight".into(), vec![CTX, d], gauss(CTX * d, 0.05, 0.0)),
    ];
    for l in 0..LAYERS {
        let p = format!("transformer.h.{}", l);
        let mut mask = vec![0f32; CTX * CTX];
        for i in 0..CTX {
            for j in 0..=i {
                mask[i * CTX + j] = 1.0;
            }
        }
        tensors.push((format!("{p}.attn.bias"), vec![1, 1, CTX, CTX], mask));
        tensors.push((format!("{p}.ln_1.weight"), vec![d], gauss(d, 0.1, 1.0)));
        tensors.push((format!("{p}.ln_1.bias"), vec![d], gauss(d, 0.05, 0.0)));
        tensors.push((format!("{p}.attn.c_attn.weight"), vec![d, 3 * d], gauss(3 * d * d, 0.08, 0.0)));
        tensors.push((format!("{p}.attn.c_attn.bias"), vec![3 * d], gauss(3 * d, 0.05, 0.0)));
        tensors.push((format!("{p}.attn.c_proj.weight"), vec![d, d], gauss(d * d, 0.08, 0.0)));
        tensors.push((format!("{p}.attn.c_proj.bias"), vec![d], gauss(d, 0.05, 0.0)));
        tensors.push((format!("{p}.ln_2.weight"), vec![d], gauss(d, 0.1, 1.0)));
        tensors.push((format!("{p}.ln_2.bias"), vec![d], gauss(d, 0.05, 0.0)));
        tensors.push((format!("{p}.mlp.c_fc.weight"), vec![d, m], gauss(d * m, 0.08, 0.0)));
        tensors.push((format!("{p}.mlp.c_fc.bias"), vec![m], gauss(m, 0.05, 0.0)));
        tensors.push((format!("{p}.mlp.c_proj.weight"), vec![m, d], gauss(m * d, 0.08, 0.0)));
        tensors.push((format!("{p}.mlp.c_proj.bias"), vec![d], gauss(d, 0.05, 0.0)));
    }
    tensors.push(("transformer.ln_f.weight".into(), vec![d], gauss(d, 0.1, 1.0)));
    tensors.push(("transformer.ln_f.bias".into(), vec![d], gauss(d, 0.05, 0.0)));
    let borrowed: Vec<(&str, &[usize], &[f32])> = tensors.iter().map(|(name, shape, data)| (name.as_str(), shape.as_slice(), data.as_slice())).collect();
    let mut bytes = Vec::new();
    safetensors::write(&mut bytes, &borrowed).unwrap();
    bytes
}

struct Outcome {
    engine: Engine,
    /// key -> (row size, rows by position)
    trace: HashMap<String, (usize, Vec<f32>)>,
    predictions: Vec<Option<(u32, f32, Option<f32>)>>,
    capacity: usize,
}

fn run(bytes: &[u8], chunks: &[&[u32]]) -> Outcome {
    let mut engine = Engine::new();
    let capacity = engine.load(&card(), &mut &bytes[..], "test", &read_asset, &mut |_, _| {}).unwrap().capacity;
    let mut trace: HashMap<String, (usize, Vec<f32>)> = HashMap::new();
    let mut predictions = vec![None; IDS.len()];
    let latest = AtomicU64::new(1);
    for (index, chunk) in chunks.iter().enumerate() {
        let result = engine
            .run(chunk, index == 0, 1, &latest, &mut |event| match event {
                RunEvent::Trace { items, .. } => {
                    for item in items {
                        let entry = trace.entry(item.key.clone()).or_insert_with(|| (item.row_size, vec![0f32; capacity * item.row_size]));
                        entry.1[item.first * item.row_size..(item.first + item.count) * item.row_size].copy_from_slice(&item.data);
                    }
                }
                RunEvent::Prediction(p) => predictions[p.position] = Some((p.summary.ids[0], p.summary.probs[0], p.summary.target_prob)),
            })
            .unwrap();
        assert!(!result.cancelled);
    }
    Outcome { engine, trace, predictions, capacity }
}

fn reference(bytes: &[u8]) -> Option<Json> {
    let dir = std::env::temp_dir().join(format!("na-rust-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (weights, out) = (dir.join("tiny.safetensors"), dir.join("ref.json"));
    std::fs::write(&weights, bytes).unwrap();
    let ids: Vec<String> = IDS.iter().map(u32::to_string).collect();
    let status = Command::new("python3").arg(repo().join("tools/reference_gpt2.py")).arg(&weights).arg(HEADS.to_string()).arg(ids.join(",")).arg(&out).output();
    match status {
        Ok(output) if output.status.success() => Some(Json::parse(&std::fs::read_to_string(&out).unwrap()).unwrap()),
        Ok(output) => {
            eprintln!("SKIPPED: the numpy reference did not run: {}", String::from_utf8_lossy(&output.stderr));
            None
        }
        Err(error) => {
            eprintln!("SKIPPED: python3 is not available for the numpy reference: {}", error);
            None
        }
    }
}

fn compare(label: &str, outcome: &mut Outcome, reference: &Json) {
    let t_count = IDS.len();
    let mut worst = (0f64, String::new());
    for (key, expected) in reference.as_object().unwrap() {
        if key == "logits" {
            continue;
        }
        let expected: Vec<f64> = expected.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let (row_size, rows) = outcome.trace.get(key).unwrap_or_else(|| panic!("{label}: the trace has no {key}"));
        let mut difference = 0f64;
        if key.ends_with("attn.pattern") {
            // engine rows are [heads, capacity]; the reference is [T, heads, T]
            let c = outcome.capacity;
            for t in 0..t_count {
                for h in 0..HEADS {
                    for j in 0..c {
                        let want = if j < t_count { expected[(t * HEADS + h) * t_count + j] } else { 0.0 };
                        difference = difference.max((rows[t * row_size + h * c + j] as f64 - want).abs());
                    }
                }
            }
        } else {
            for i in 0..t_count * row_size {
                difference = difference.max((rows[i] as f64 - expected[i]).abs());
            }
        }
        if difference > worst.0 {
            worst = (difference, key.clone());
        }
    }
    println!("{label}: largest difference in any traced value {:.2e} (in {})", worst.0, worst.1);
    assert!(worst.0 < 2e-4, "{label}: {} differs from the reference by {}", worst.1, worst.0);

    let logits: Vec<f64> = reference.get("logits").unwrap().as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mut difference = 0f64;
    for t in 0..t_count {
        let got = outcome.engine.logits_at(t).unwrap();
        let row = &logits[t * VOCAB..(t + 1) * VOCAB];
        for v in 0..VOCAB {
            difference = difference.max((got[v] as f64 - row[v]).abs());
        }
        // the ranking and the probabilities the interface is given
        let top = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = row.iter().map(|v| (v - top).exp()).sum();
        let best = row.iter().position(|v| *v == top).unwrap() as u32;
        let (id, prob, target) = outcome.predictions[t].expect("a prediction for every position");
        assert_eq!(id, best, "{label}: top prediction at position {t}");
        assert!((prob as f64 - 1.0 / z).abs() < 1e-4);
        if t + 1 < t_count {
            let want = (row[IDS[t + 1] as usize] - top).exp() / z;
            assert!((target.unwrap() as f64 - want).abs() < 1e-4, "{label}: next-token probability at position {t}");
        }
    }
    println!("{label}: largest difference in any logit {:.2e}", difference);
    assert!(difference < 5e-4);
}

#[test]
fn matches_the_numpy_reference() {
    let bytes = checkpoint();
    let Some(reference) = reference(&bytes) else { return };

    let mut whole = run(&bytes, &[&IDS]);
    let info = whole.engine.info().unwrap();
    assert_eq!(info.config.get("n_layer").unwrap().as_usize(), Some(LAYERS));
    assert_eq!(info.config.get("n_inner").unwrap().as_usize(), Some(4 * WIDTH));
    assert_eq!(info.config.get("vocab_size").unwrap().as_usize(), Some(VOCAB));
    compare("whole prompt", &mut whole, &reference);

    // the same tokens appended one at a time must give the same numbers
    let singles: Vec<&[u32]> = std::iter::once(&IDS[..4]).chain(IDS[4..].chunks(1)).collect();
    let mut stepped = run(&bytes, &singles);
    compare("tokens appended one at a time", &mut stepped, &reference);

    // greedy pick is the arg-max of the last position
    let (id, _) = stepped.engine.pick(0.0, 40, None).unwrap();
    assert_eq!(Some(id), stepped.predictions[IDS.len() - 1].map(|p| p.0));
}

#[test]
fn a_newer_request_stops_a_run() {
    let bytes = checkpoint();
    let mut engine = Engine::new();
    engine.load(&card(), &mut &bytes[..], "test", &read_asset, &mut |_, _| {}).unwrap();
    let latest = AtomicU64::new(1);
    let mut units = 0;
    let result = engine
        .run(&IDS, true, 1, &latest, &mut |event| {
            if let RunEvent::Trace { unit, .. } = event {
                units += 1;
                if unit == 1 {
                    latest.store(2, std::sync::atomic::Ordering::SeqCst); // a newer run arrives
                }
            }
        })
        .unwrap();
    assert!(result.cancelled);
    assert_eq!(units, 2, "the run stops after the unit in which the newer request arrived");
    // and the engine is still usable
    let latest = AtomicU64::new(7);
    let result = engine.run(&IDS, true, 7, &latest, &mut |_| {}).unwrap();
    assert!(!result.cancelled && result.length == IDS.len());
}

#[test]
fn regions_and_statistics() {
    let bytes = checkpoint();
    let mut engine = Engine::new();
    engine.load(&card(), &mut &bytes[..], "test", &read_asset, &mut |_, _| {}).unwrap();
    let store = engine.store().unwrap();
    let raw = store.data("h.1.attn.c_attn.weight").unwrap();
    let view = View { tensor: "h.1.attn.c_attn.weight".into(), row0: 0, col0: WIDTH, rows: WIDTH, cols: WIDTH, transpose: false };
    let exact = store.region(&view, WIDTH, WIDTH).unwrap();
    for r in 0..WIDTH {
        for c in 0..WIDTH {
            assert_eq!(exact[r * WIDTH + c], raw[r * 3 * WIDTH + WIDTH + c]);
        }
    }
    let q = WIDTH / 4;
    let pooled = store.region(&view, q, q).unwrap();
    for big_r in 0..q {
        for big_c in 0..q {
            let mut best = 0f32;
            for r in big_r * 4..big_r * 4 + 4 {
                for c in big_c * 4..big_c * 4 + 4 {
                    let v = raw[r * 3 * WIDTH + WIDTH + c];
                    if v.abs() > best.abs() {
                        best = v;
                    }
                }
            }
            assert_eq!(pooled[big_r * q + big_c], best);
        }
    }
    let raw_t = store.data("h.0.mlp.c_proj.weight").unwrap();
    let view_t = View { tensor: "h.0.mlp.c_proj.weight".into(), row0: 0, col0: 0, rows: WIDTH, cols: 4 * WIDTH, transpose: true };
    let t = store.region(&view_t, WIDTH, 4 * WIDTH).unwrap();
    for r in 0..WIDTH {
        for c in 0..4 * WIDTH {
            assert_eq!(t[r * 4 * WIDTH + c], raw_t[c * WIDTH + r]);
        }
    }
    let stats = store.stats(&view).unwrap();
    assert!(stats.std > 0.0 && stats.min < 0.0 && stats.max > 0.0 && stats.count == WIDTH * WIDTH);
    assert!(store.region(&View { rows: WIDTH + 1, ..view.clone() }, 4, 4).is_err());
}

#[test]
fn rejects_what_is_not_a_checkpoint() {
    let mut engine = Engine::new();
    let text = b"this is a text file, not a checkpoint, and it is longer than eight bytes";
    let error = engine.load(&card(), &mut &text[..], "test", &read_asset, &mut |_, _| {}).err().unwrap();
    assert!(error.contains("safetensors"), "{error}");
    let bytes = checkpoint();
    let cut = &bytes[..bytes.len() / 2];
    let error = engine.load(&card(), &mut &cut[..], "test", &read_asset, &mut |_, _| {}).err().unwrap();
    assert!(error.contains("ended early"), "{error}");
}
