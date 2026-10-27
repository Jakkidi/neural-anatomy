//! Loads a model checkpoint and runs it while handing out every activation.
//!
//! This is the Rust counterpart of `src/core` and `src/arch` in the web app:
//! the same checkpoint format, the same tokenizer, the same forward pass, and
//! the same trace keys, so the same figures can draw from either engine. It
//! depends on nothing outside the standard library.

pub mod arch;
pub mod bpe;
pub mod engine;
pub mod gpt2;
pub mod json;
pub mod kernels;
pub mod safetensors;
pub mod sampling;
pub mod store;
mod unicode_tables;

pub use engine::{Engine, ModelInfo, Prediction, RunEvent, RunResult};
pub use json::Json;
pub use store::{Stats, View, WeightStore};
