//! GPT-2: decoder-only transformer.
//!
//! ```text
//! x = W_E[token] + W_pos[position]
//! for each block:
//!     x = x + Attention(LayerNorm1(x))      attention writes into the stream
//!     x = x + MLP(LayerNorm2(x))            the MLP writes into the stream
//! logits = LayerNorm_f(x) @ W_E^T
//! ```
//!
//! Every intermediate value is handed out as it is computed. Keys and values
//! are cached per block, so appending one token costs one position, not the
//! whole prompt again. The trace keys and layouts are the ones the figures
//! read (`src/arch/gpt2/spec.js`).

use crate::arch::{Architecture, Plan, ReadAsset, Runtime, TraceItem};
use crate::bpe::{ByteBpe, Tokenizer};
use crate::json::Json;
use crate::kernels;
use crate::safetensors::{Entry, Want};
use crate::sampling::Random;
use crate::store::WeightStore;
use std::sync::Arc;

/// Most token positions one run can hold. Must match TRACE_CAPACITY in spec.js.
const TRACE_CAPACITY: usize = 256;

pub struct Gpt2;

#[derive(Debug, Clone, Copy)]
struct Config {
    layers: usize,
    heads: usize,
    width: usize,
    inner: usize,
    context: usize,
    vocab: usize,
    eps: f32,
}

impl Config {
    fn from_json(json: &Json) -> Result<Config, String> {
        let field = |name: &str| json.get(name).and_then(Json::as_usize).ok_or_else(|| format!("The model's configuration has no \"{}\".", name));
        let config = Config {
            layers: field("n_layer")?,
            heads: field("n_head")?,
            width: field("n_embd")?,
            inner: field("n_inner")?,
            context: field("n_ctx")?,
            vocab: field("vocab_size")?,
            eps: json.get("layer_norm_epsilon").and_then(Json::as_f64).unwrap_or(1e-5) as f32,
        };
        if config.heads == 0 || config.width % config.heads != 0 {
            return Err(format!("The embedding width {} does not divide into {} heads.", config.width, config.heads));
        }
        Ok(config)
    }

    fn to_json(self) -> Json {
        Json::obj(vec![
            ("n_layer", Json::uint(self.layers)),
            ("n_head", Json::uint(self.heads)),
            ("n_embd", Json::uint(self.width)),
            ("n_inner", Json::uint(self.inner)),
            ("n_ctx", Json::uint(self.context)),
            ("vocab_size", Json::uint(self.vocab)),
            ("layer_norm_epsilon", Json::Num(self.eps as f64)),
        ])
    }

    /// Every tensor a GPT-2 of this size has, with its shape.
    fn shapes(&self) -> Vec<(String, Vec<usize>)> {
        let (d, m) = (self.width, self.inner);
        let mut out = vec![
            ("wte.weight".to_string(), vec![self.vocab, d]),
            ("wpe.weight".to_string(), vec![self.context, d]),
            ("ln_f.weight".to_string(), vec![d]),
            ("ln_f.bias".to_string(), vec![d]),
        ];
        for l in 0..self.layers {
            let p = format!("h.{}", l);
            out.push((format!("{}.ln_1.weight", p), vec![d]));
            out.push((format!("{}.ln_1.bias", p), vec![d]));
            out.push((format!("{}.attn.c_attn.weight", p), vec![d, 3 * d]));
            out.push((format!("{}.attn.c_attn.bias", p), vec![3 * d]));
            out.push((format!("{}.attn.c_proj.weight", p), vec![d, d]));
            out.push((format!("{}.attn.c_proj.bias", p), vec![d]));
            out.push((format!("{}.ln_2.weight", p), vec![d]));
            out.push((format!("{}.ln_2.bias", p), vec![d]));
            out.push((format!("{}.mlp.c_fc.weight", p), vec![d, m]));
            out.push((format!("{}.mlp.c_fc.bias", p), vec![m]));
            out.push((format!("{}.mlp.c_proj.weight", p), vec![m, d]));
            out.push((format!("{}.mlp.c_proj.bias", p), vec![d]));
        }
        out
    }
}

/// Checkpoints name the same tensors with or without a module prefix.
fn strip(name: &str) -> &str {
    for prefix in ["transformer.", "model.", "gpt2."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return rest;
        }
    }
    name
}

/// Buffers and tied copies a checkpoint carries that are not weights of their own.
fn skipped(name: &str) -> bool {
    // ".attn.bias" is the causal mask; "c_attn.bias" is a real bias and does not end that way
    name.ends_with(".attn.bias") || name.ends_with(".attn.masked_bias") || name == "lm_head.weight"
}

impl Architecture for Gpt2 {
    fn id(&self) -> &'static str {
        "gpt2"
    }

    fn plan(&self, tensors: &[Entry], card: &Json) -> Result<Plan, String> {
        let find = |name: &str| tensors.iter().find(|entry| strip(&entry.name) == name && !skipped(strip(&entry.name)));
        let (Some(wte), Some(wpe)) = (find("wte.weight"), find("wpe.weight")) else {
            return Err("This file is not a GPT-2 checkpoint: it has no wte.weight and wpe.weight.".to_string());
        };
        let mut layers = 0;
        while find(&format!("h.{}.attn.c_attn.weight", layers)).is_some() {
            layers += 1;
        }
        if layers == 0 || wte.shape.len() != 2 || wpe.shape.len() != 2 {
            return Err("This file is not a GPT-2 checkpoint: it has no h.0.attn.c_attn.weight.".to_string());
        }
        let width = wte.shape[1];
        let inner = find("h.0.mlp.c_fc.weight").and_then(|entry| entry.shape.get(1).copied()).ok_or("The checkpoint is missing \"h.0.mlp.c_fc.weight\".")?;
        // The head count is the one thing no shape records. Trust the card when
        // it describes a model of this width; otherwise GPT-2's rule of 64 per head.
        let card_config = card.get("config");
        let card_width = card_config.and_then(|c| c.get("n_embd")).and_then(Json::as_usize);
        let card_heads = card_config.and_then(|c| c.get("n_head")).and_then(Json::as_usize);
        let heads = match (card_width, card_heads) {
            (Some(w), Some(h)) if w == width && h > 0 => h,
            _ => (width / 64).max(1),
        };
        let eps = card_config.and_then(|c| c.get("layer_norm_epsilon")).and_then(Json::as_f64).unwrap_or(1e-5) as f32;
        let config = Config { layers, heads, width, inner, context: wpe.shape[0], vocab: wte.shape[0], eps };
        if width % heads != 0 {
            return Err(format!("The embedding width {} does not divide into {} heads.", width, heads));
        }
        let mut wanted = Vec::new();
        for (name, shape) in config.shapes() {
            let entry = find(&name).ok_or_else(|| format!("The checkpoint is missing \"{}\".", name))?;
            if entry.shape != shape {
                return Err(format!("\"{}\" has shape {:?}, but a GPT-2 of this size needs {:?}.", name, entry.shape, shape));
            }
            wanted.push(Want { file: entry.name.clone(), name });
        }
        Ok(Plan { config: config.to_json(), wanted })
    }

    fn synthesize(&self, config: &Json, seed: u32) -> Result<WeightStore, String> {
        let config = Config::from_json(config)?;
        let mut random = Random::new(seed);
        let mut store = WeightStore::new();
        for (name, shape) in config.shapes() {
            let count: usize = shape.iter().product();
            let mut data = vec![0f32; count];
            let is_gain = name.ends_with("ln_1.weight") || name.ends_with("ln_2.weight") || name == "ln_f.weight";
            if is_gain {
                data.fill(1.0);
            } else if !name.ends_with(".bias") {
                // GPT-2's own initialisation
                let std = if name == "wpe.weight" { 0.01 } else { 0.02 };
                for pair in data.chunks_mut(2) {
                    let (u, v) = (random.next().max(1e-12), random.next());
                    let radius = (-2.0 * u.ln()).sqrt() * std;
                    let angle = std::f64::consts::TAU * v;
                    pair[0] = (radius * angle.cos()) as f32;
                    if pair.len() > 1 {
                        pair[1] = (radius * angle.sin()) as f32;
                    }
                }
            }
            store.insert(&name, shape, data);
        }
        Ok(store)
    }

    fn tokenizer(&self, card: &Json, read: ReadAsset) -> Result<Box<dyn Tokenizer>, String> {
        let path = |key: &str| card.path(&["tokenizer", key]).and_then(Json::as_str).ok_or_else(|| format!("The model card names no tokenizer {} file.", key));
        let vocab = Json::parse(&read(path("vocab")?)?).map_err(|error| format!("The vocabulary file cannot be read: {}.", error))?;
        let merges = read(path("merges")?)?;
        Ok(Box::new(ByteBpe::new(&vocab, &merges, &["<|endoftext|>"])?))
    }

    fn param_count(&self, config: &Json) -> usize {
        Config::from_json(config).map(|config| config.shapes().iter().map(|(_, shape)| shape.iter().product::<usize>()).sum()).unwrap_or(0)
    }

    fn capacity(&self, config: &Json) -> usize {
        Config::from_json(config).map(|config| config.context.min(TRACE_CAPACITY)).unwrap_or(TRACE_CAPACITY)
    }

    fn runtime(&self, config: &Json, store: Arc<WeightStore>, capacity: usize) -> Result<Box<dyn Runtime>, String> {
        let config = Config::from_json(config)?;
        for (name, shape) in config.shapes() {
            if store.get(&name)?.shape != shape {
                return Err(format!("\"{}\" does not have the shape this model needs.", name));
            }
        }
        let (c, d, m, h) = (capacity, config.width, config.inner, config.heads);
        Ok(Box::new(Gpt2Runtime {
            config,
            store,
            capacity,
            length: 0,
            names: (0..config.layers).map(BlockNames::new).collect(),
            keys: (0..config.layers).map(|_| vec![0f32; c * d]).collect(),
            values: (0..config.layers).map(|_| vec![0f32; c * d]).collect(),
            last: vec![0f32; c * d],
            x: vec![0f32; c * d],
            normed: vec![0f32; c * d],
            qkv: vec![0f32; c * 3 * d],
            mixed: vec![0f32; c * d],
            write: vec![0f32; c * d],
            hidden: vec![0f32; c * m],
            pattern: vec![0f32; c * h * c],
            logits: vec![0f32; config.vocab],
        }))
    }
}

struct BlockNames {
    ln1_g: String,
    ln1_b: String,
    qkv_w: String,
    qkv_b: String,
    o_w: String,
    o_b: String,
    ln2_g: String,
    ln2_b: String,
    in_w: String,
    in_b: String,
    out_w: String,
    out_b: String,
}

impl BlockNames {
    fn new(l: usize) -> BlockNames {
        let p = format!("h.{}", l);
        BlockNames {
            ln1_g: format!("{}.ln_1.weight", p),
            ln1_b: format!("{}.ln_1.bias", p),
            qkv_w: format!("{}.attn.c_attn.weight", p),
            qkv_b: format!("{}.attn.c_attn.bias", p),
            o_w: format!("{}.attn.c_proj.weight", p),
            o_b: format!("{}.attn.c_proj.bias", p),
            ln2_g: format!("{}.ln_2.weight", p),
            ln2_b: format!("{}.ln_2.bias", p),
            in_w: format!("{}.mlp.c_fc.weight", p),
            in_b: format!("{}.mlp.c_fc.bias", p),
            out_w: format!("{}.mlp.c_proj.weight", p),
            out_b: format!("{}.mlp.c_proj.bias", p),
        }
    }
}

struct Gpt2Runtime {
    config: Config,
    store: Arc<WeightStore>,
    capacity: usize,
    length: usize,
    names: Vec<BlockNames>,
    // per-position state that persists between appends
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    last: Vec<f32>,
    // work rows for the positions being appended
    x: Vec<f32>,
    normed: Vec<f32>,
    qkv: Vec<f32>,
    mixed: Vec<f32>,
    write: Vec<f32>,
    hidden: Vec<f32>,
    pattern: Vec<f32>,
    logits: Vec<f32>,
}

fn layer_norm(x: &[f32], gain: &[f32], bias: &[f32], out: &mut [f32], d: usize, eps: f32) {
    for (row, out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row.iter().sum::<f32>() / d as f32;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inverse = 1.0 / (variance + eps).sqrt();
        for j in 0..d {
            out[j] = (row[j] - mean) * inverse * gain[j] + bias[j];
        }
    }
}

/// GPT-2's GELU is the tanh approximation ("gelu_new").
fn gelu(values: &mut [f32]) {
    const K: f32 = 0.797_884_6; // sqrt(2 / pi)
    let apply = |chunk: &mut [f32]| {
        for v in chunk.iter_mut() {
            let x = *v;
            *v = 0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh());
        }
    };
    let workers = kernels::threads();
    if workers <= 1 || values.len() < 32_768 {
        apply(values);
        return;
    }
    let share = values.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for chunk in values.chunks_mut(share) {
            scope.spawn(move || apply(chunk));
        }
    });
}

fn item(key: String, row_size: usize, first: usize, count: usize, rows: &[f32]) -> TraceItem {
    TraceItem { key, row_size, first, count, data: rows[..count * row_size].to_vec() }
}

impl Runtime for Gpt2Runtime {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.length
    }

    fn units(&self) -> usize {
        self.config.layers + 2
    }

    fn vocab(&self) -> usize {
        self.config.vocab
    }

    fn reset(&mut self) {
        self.length = 0;
    }

    fn append(&mut self, ids: &[u32], sink: &mut dyn FnMut(usize, Vec<TraceItem>) -> bool) -> Result<bool, String> {
        let Config { layers, heads, width: d, inner: m, vocab, eps, .. } = self.config;
        let (n, p0, c) = (ids.len(), self.length, self.capacity);
        if p0 + n > c {
            return Err(format!("A run holds at most {} tokens; this one has {}.", c, p0 + n));
        }
        if n == 0 {
            return Ok(true);
        }
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let store = self.store.clone();
        let wte = store.data("wte.weight")?;
        let wpe = store.data("wpe.weight")?;

        // embedding
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab {
                return Err(format!("Token id {} is outside the vocabulary.", id));
            }
            let (tok, pos) = (&wte[id * d..(id + 1) * d], &wpe[(p0 + t) * d..(p0 + t + 1) * d]);
            self.normed[t * d..(t + 1) * d].copy_from_slice(tok);
            self.mixed[t * d..(t + 1) * d].copy_from_slice(pos);
            for j in 0..d {
                self.x[t * d + j] = tok[j] + pos[j];
            }
        }
        let items = vec![
            item("embed.tok".into(), d, p0, n, &self.normed),
            item("embed.pos".into(), d, p0, n, &self.mixed),
            item("resid.0".into(), d, p0, n, &self.x),
        ];
        if !sink(0, items) {
            return Ok(false);
        }

        for l in 0..layers {
            let names = &self.names[l];
            let mut items = Vec::with_capacity(5);

            // attention
            layer_norm(&self.x[..n * d], store.data(&names.ln1_g)?, store.data(&names.ln1_b)?, &mut self.normed[..n * d], d, eps);
            kernels::affine(&self.normed, store.data(&names.qkv_w)?, Some(store.data(&names.qkv_b)?), &mut self.qkv, n, d, 3 * d);
            let (keys, values) = (&mut self.keys[l], &mut self.values[l]);
            for t in 0..n {
                let (src, dst) = (t * 3 * d, (p0 + t) * d);
                keys[dst..dst + d].copy_from_slice(&self.qkv[src + d..src + 2 * d]);
                values[dst..dst + d].copy_from_slice(&self.qkv[src + 2 * d..src + 3 * d]);
            }
            self.pattern[..n * heads * c].fill(0.0);
            let mut scores = vec![0f32; c];
            for t in 0..n {
                let p = p0 + t; // this position may look at 0..=p
                for h in 0..heads {
                    let q = &self.qkv[t * 3 * d + h * hd..t * 3 * d + (h + 1) * hd];
                    let mut top = f32::NEG_INFINITY;
                    for j in 0..=p {
                        let s = kernels::dot(q, &keys[j * d + h * hd..j * d + (h + 1) * hd]) * scale;
                        scores[j] = s;
                        top = top.max(s);
                    }
                    let mut sum = 0f32;
                    for s in scores[..=p].iter_mut() {
                        *s = (*s - top).exp();
                        sum += *s;
                    }
                    let row = &mut self.pattern[(t * heads + h) * c..(t * heads + h + 1) * c];
                    let out = &mut self.mixed[t * d + h * hd..t * d + (h + 1) * hd];
                    out.fill(0.0);
                    for j in 0..=p {
                        let a = scores[j] / sum;
                        row[j] = a;
                        for (o, v) in out.iter_mut().zip(&values[j * d + h * hd..j * d + (h + 1) * hd]) {
                            *o += a * *v;
                        }
                    }
                }
            }
            items.push(item(format!("block.{}.attn.pattern", l), heads * c, p0, n, &self.pattern));
            kernels::affine(&self.mixed, store.data(&names.o_w)?, Some(store.data(&names.o_b)?), &mut self.write, n, d, d);
            items.push(item(format!("block.{}.attn.out", l), d, p0, n, &self.write));
            for (x, w) in self.x[..n * d].iter_mut().zip(&self.write[..n * d]) {
                *x += *w;
            }

            // MLP
            layer_norm(&self.x[..n * d], store.data(&names.ln2_g)?, store.data(&names.ln2_b)?, &mut self.normed[..n * d], d, eps);
            kernels::affine(&self.normed, store.data(&names.in_w)?, Some(store.data(&names.in_b)?), &mut self.hidden, n, d, m);
            gelu(&mut self.hidden[..n * m]);
            items.push(item(format!("block.{}.mlp.post", l), m, p0, n, &self.hidden));
            kernels::affine(&self.hidden, store.data(&names.out_w)?, Some(store.data(&names.out_b)?), &mut self.write, n, m, d);
            items.push(item(format!("block.{}.mlp.out", l), d, p0, n, &self.write));
            for (x, w) in self.x[..n * d].iter_mut().zip(&self.write[..n * d]) {
                *x += *w;
            }
            items.push(item(format!("block.{}.resid", l), d, p0, n, &self.x));
            if !sink(l + 1, items) {
                return Ok(false);
            }
        }

        layer_norm(&self.x[..n * d], store.data("ln_f.weight")?, store.data("ln_f.bias")?, &mut self.normed[..n * d], d, eps);
        self.last[p0 * d..(p0 + n) * d].copy_from_slice(&self.normed[..n * d]);
        let items = vec![item("final.resid".into(), d, p0, n, &self.normed)];
        // the positions are committed before the last hand-out, so a sink that
        // asks for logits from inside it finds them
        self.length = p0 + n;
        sink(layers + 1, items);
        Ok(true)
    }

    fn logits(&mut self, position: usize) -> Result<&[f32], String> {
        if position >= self.length {
            return Err(format!("Position {} has not been run.", position));
        }
        let d = self.config.width;
        let store = self.store.clone();
        kernels::matvec(store.data("wte.weight")?, &self.last[position * d..(position + 1) * d], &mut self.logits, true);
        Ok(&self.logits)
    }

    fn logits_many(&mut self, positions: &[usize]) -> Result<Vec<Vec<f32>>, String> {
        if let Some(bad) = positions.iter().find(|&&p| p >= self.length) {
            return Err(format!("Position {} has not been run.", bad));
        }
        let (d, vocab) = (self.config.width, self.config.vocab);
        let wte = self.store.data("wte.weight")?;
        let last = &self.last;
        let mut out: Vec<Vec<f32>> = positions.iter().map(|_| vec![0f32; vocab]).collect();
        let workers = kernels::threads().min(positions.len()).max(1);
        if workers == 1 {
            for (row, &p) in out.iter_mut().zip(positions) {
                kernels::matvec(wte, &last[p * d..(p + 1) * d], row, true);
            }
            return Ok(out);
        }
        let share = positions.len().div_ceil(workers);
        std::thread::scope(|scope| {
            for (rows, ps) in out.chunks_mut(share).zip(positions.chunks(share)) {
                scope.spawn(move || {
                    for (row, &p) in rows.iter_mut().zip(ps) {
                        kernels::matvec(wte, &last[p * d..(p + 1) * d], row, false);
                    }
                });
            }
        });
        Ok(out)
    }

    fn readout(&mut self, vector: &[f32]) -> Option<Result<&[f32], String>> {
        let Config { width: d, eps, .. } = self.config;
        if vector.len() != d {
            return Some(Err(format!("A stream vector of this model has {} numbers, not {}.", d, vector.len())));
        }
        let store = self.store.clone();
        let result = (|| -> Result<(), String> {
            let mut row = vec![0f32; d];
            layer_norm(vector, store.data("ln_f.weight")?, store.data("ln_f.bias")?, &mut row, d, eps);
            kernels::matvec(store.data("wte.weight")?, &row, &mut self.logits, true);
            Ok(())
        })();
        Some(result.map(|()| &self.logits[..]))
    }
}
