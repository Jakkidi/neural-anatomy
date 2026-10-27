//! Checks the tokenizer against token ids published for GPT-2, using the real
//! vocabulary, and against round trips through awkward text.

use na_engine::bpe::{ByteBpe, Tokenizer};
use na_engine::Json;
use std::path::PathBuf;

fn tokenizer() -> ByteBpe {
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/gpt2");
    let vocab = Json::parse(&std::fs::read_to_string(assets.join("vocab.json")).unwrap()).unwrap();
    let merges = std::fs::read_to_string(assets.join("merges.txt")).unwrap();
    ByteBpe::new(&vocab, &merges, &["<|endoftext|>"]).unwrap()
}

#[test]
fn published_ids() {
    let bpe = tokenizer();
    assert_eq!(bpe.vocab_size(), 50257);
    // ids from OpenAI's encoder and the GPT-3-Encoder test suite
    let known: &[(&str, &[u32])] = &[
        ("Hello world", &[15496, 995]),
        ("Hello, world!", &[15496, 11, 995, 0]),
        (" Hello", &[18435]),
        ("The quick brown fox jumps over the lazy dog", &[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]),
        ("hello \u{1F44B} world \u{1F30D}", &[31373, 50169, 233, 995, 12520, 234, 235]),
        ("This is some text", &[1212, 318, 617, 2420]),
        ("indivisible", &[521, 452, 12843]),
        ("\t", &[197]),
        ("toString constructor hasOwnProperty valueOf", &[1462, 10100, 23772, 468, 23858, 21746, 1988, 5189]),
        (" ", &[220]),
        ("", &[]),
        ("<|endoftext|>", &[50256]),
        ("a<|endoftext|>b", &[64, 50256, 65]),
        ("The Eiffel Tower is located in the city of", &[464, 412, 733, 417, 8765, 318, 5140, 287, 262, 1748, 286]),
    ];
    for (text, ids) in known {
        assert_eq!(&bpe.encode(text), ids, "encode {:?}", text);
    }
    assert_eq!(bpe.piece(995), " world");
}

#[test]
fn round_trips() {
    let bpe = tokenizer();
    for text in [
        "It's the model's 3rd try \u{2014} isn't it?\n\n  Tabs\tand   spaces.",
        "na\u{EF}ve caf\u{E9}, \u{6771}\u{4EAC}, \u{395}\u{3BB}\u{3BB}\u{3B7}\u{3BD}\u{3B9}\u{3BA}\u{3AC}, \u{5E2}\u{5D1}\u{5E8}\u{5D9}\u{5EA}, \u{1F9E0}\u{1F52C}",
        "def f(x):\n    return x**2  # squared\n",
        "    leading and trailing    ",
        "we'll they've I'm you're he'd it's can't 'quoted'",
    ] {
        assert_eq!(bpe.decode(&bpe.encode(text)), text);
    }
}
