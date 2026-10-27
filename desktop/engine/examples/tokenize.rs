//! Tokenizes each string of a JSON array and prints the ids, one JSON array
//! per line. Used to compare this tokenizer with the JavaScript one.
//!
//!   cargo run --example tokenize -- strings.json

use na_engine::bpe::{ByteBpe, Tokenizer};
use na_engine::Json;
use std::path::PathBuf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: tokenize strings.json");
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/gpt2");
    let vocab = Json::parse(&std::fs::read_to_string(assets.join("vocab.json")).unwrap()).unwrap();
    let merges = std::fs::read_to_string(assets.join("merges.txt")).unwrap();
    let bpe = ByteBpe::new(&vocab, &merges, &["<|endoftext|>"]).unwrap();
    let strings = Json::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut out = String::new();
    for text in strings.as_array().unwrap() {
        let ids = bpe.encode(text.as_str().unwrap());
        out.push_str(&Json::Arr(ids.iter().map(|&id| Json::uint(id as usize)).collect()).to_string());
        out.push('\n');
    }
    print!("{}", out);
}
