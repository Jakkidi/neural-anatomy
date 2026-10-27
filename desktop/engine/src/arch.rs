//! What an architecture has to provide to be run by the engine.
//!
//! This mirrors the JavaScript contract in `docs/ADDING_AN_ARCHITECTURE.md`:
//! an architecture reads a checkpoint's tensor list and says what to keep,
//! makes a tokenizer, and makes a runtime that runs the model while handing
//! out every activation. Nothing else in the engine knows about any one model.

use crate::bpe::Tokenizer;
use crate::json::Json;
use crate::safetensors::{Entry, Want};
use crate::store::WeightStore;
use std::sync::Arc;

/// One traced array for the positions just run. Rows are positions.
pub struct TraceItem {
    pub key: String,
    pub row_size: usize,
    pub first: usize,
    pub count: usize,
    pub data: Vec<f32>,
}

/// What a checkpoint turned out to hold.
pub struct Plan {
    /// the model's configuration, as far as the shapes reveal it
    pub config: Json,
    pub wanted: Vec<Want>,
}

/// Reads a file that ships with the app (a vocabulary, say) as text.
pub type ReadAsset<'a> = &'a dyn Fn(&str) -> Result<String, String>;

pub trait Architecture: Send + Sync {
    fn id(&self) -> &'static str;

    /// Decides what to keep from a checkpoint. `card` is the model card the
    /// interface sent; it supplies what no tensor shape records.
    fn plan(&self, tensors: &[Entry], card: &Json) -> Result<Plan, String>;

    /// Random weights in this architecture's layout, for the test pattern.
    fn synthesize(&self, config: &Json, seed: u32) -> Result<WeightStore, String>;

    fn tokenizer(&self, card: &Json, read: ReadAsset) -> Result<Box<dyn Tokenizer>, String>;

    fn param_count(&self, config: &Json) -> usize;

    /// Most positions one run may hold.
    fn capacity(&self, config: &Json) -> usize;

    fn runtime(&self, config: &Json, store: Arc<WeightStore>, capacity: usize) -> Result<Box<dyn Runtime>, String>;
}

pub trait Runtime: Send {
    fn capacity(&self) -> usize;
    /// positions run so far
    fn len(&self) -> usize;
    /// units from input to output: 0 is the embedding, the last is the output head
    fn units(&self) -> usize;
    fn vocab(&self) -> usize;
    fn reset(&mut self);

    /// Runs new positions after the cached ones. `sink(unit, items)` is called
    /// as each unit finishes; if it returns false the run stops there and the
    /// cache is left as it was. Returns whether the run completed.
    fn append(&mut self, ids: &[u32], sink: &mut dyn FnMut(usize, Vec<TraceItem>) -> bool) -> Result<bool, String>;

    /// Output scores at a position already run.
    fn logits(&mut self, position: usize) -> Result<&[f32], String>;

    /// Output scores at several positions at once, in the order asked.
    fn logits_many(&mut self, positions: &[usize]) -> Result<Vec<Vec<f32>>, String>;

    /// Reads any residual-stream vector through the output head. `None` if
    /// the architecture has no such read-out.
    fn readout(&mut self, vector: &[f32]) -> Option<Result<&[f32], String>>;
}

pub fn architecture(id: &str) -> Option<&'static dyn Architecture> {
    match id {
        "gpt2" => Some(&crate::gpt2::Gpt2),
        _ => None,
    }
}
