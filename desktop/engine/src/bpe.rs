//! Byte-level BPE, the tokenizer of GPT-2 (and RoBERTa, BART, ...).
//!
//! Text is split into word-ish chunks, each chunk becomes UTF-8 bytes mapped
//! to printable characters, and adjacent pairs are merged in the order the
//! merges file ranks them. This follows OpenAI's `encoder.py`.

use crate::json::Json;
use crate::unicode_tables::{is_letter, is_number};
use std::collections::HashMap;

pub trait Tokenizer: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    /// The text of one token on its own. Half of a multi-byte character shows
    /// as the replacement character.
    fn piece(&self, id: u32) -> String;
}

pub struct ByteBpe {
    vocab: HashMap<String, u32>,
    by_id: Vec<Option<String>>,
    ranks: HashMap<(String, String), u32>,
    special: Vec<(String, u32)>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

/// GPT-2 maps every byte to a printable character: the printable Latin-1
/// bytes stand for themselves, and the rest move up to U+0100 and beyond.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut printable = [false; 256];
    for b in (33..=126).chain(161..=172).chain(174..=255) {
        printable[b as usize] = true;
    }
    let mut forward = ['\0'; 256];
    let mut backward = HashMap::new();
    let mut extra = 0u32;
    for b in 0..256usize {
        let c = if printable[b] {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        forward[b] = c;
        backward.insert(c, b as u8);
    }
    (forward, backward)
}

impl ByteBpe {
    /// `vocab` is vocab.json (token text to id); `merges` is merges.txt.
    /// `special` tokens are matched literally in the text.
    pub fn new(vocab: &Json, merges: &str, special: &[&str]) -> Result<ByteBpe, String> {
        let pairs = vocab.as_object().ok_or("The vocabulary file is not a JSON object.")?;
        let mut map = HashMap::with_capacity(pairs.len());
        let mut size = 0usize;
        for (text, id) in pairs {
            let id = id.as_usize().ok_or("The vocabulary file has an entry that is not a number.")?;
            map.insert(text.clone(), id as u32);
            size = size.max(id + 1);
        }
        let mut by_id = vec![None; size];
        for (text, id) in &map {
            by_id[*id as usize] = Some(text.clone());
        }
        let mut ranks = HashMap::new();
        let mut rank = 0u32;
        for line in merges.lines() {
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            if let Some((a, b)) = line.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), rank);
                rank += 1;
            }
        }
        let mut special_ids = Vec::new();
        for name in special {
            if let Some(id) = map.get(*name) {
                special_ids.push((name.to_string(), *id));
            }
        }
        let (byte_to_char, char_to_byte) = byte_alphabet();
        Ok(ByteBpe { vocab: map, by_id, ranks, special: special_ids, byte_to_char, char_to_byte })
    }

    fn bpe(&self, chunk: &str, out: &mut Vec<u32>) {
        let mut word: Vec<String> = chunk.bytes().map(|b| self.byte_to_char[b as usize].to_string()).collect();
        while word.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..word.len() - 1 {
                // the lookup key borrows nothing, so build it once per pair
                if let Some(rank) = self.ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.map_or(true, |(r, _)| *rank < r) {
                        best = Some((*rank, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let first = word[at].clone();
            let second = word[at + 1].clone();
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut word[i]));
                    i += 1;
                }
            }
            word = merged;
        }
        for piece in word {
            // every single byte is in the vocabulary, so this cannot miss
            if let Some(id) = self.vocab.get(&piece) {
                out.push(*id);
            }
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let end_of = |i: usize| if i < chars.len() { chars[i].0 } else { text.len() };
        let mut i = 0;
        while i < chars.len() {
            let n = chunk_length(&chars, i);
            self.bpe(&text[chars[i].0..end_of(i + n)], out);
            i += n;
        }
    }

    fn bytes_of(&self, id: u32, out: &mut Vec<u8>) {
        let Some(Some(piece)) = self.by_id.get(id as usize) else { return };
        if self.special.iter().any(|(_, special)| *special == id) {
            out.extend_from_slice(piece.as_bytes());
            return;
        }
        for c in piece.chars() {
            out.push(*self.char_to_byte.get(&c).unwrap_or(&b'?'));
        }
    }
}

/// How many characters the next chunk takes, by GPT-2's pattern
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// tried in that order at position `i`. Always at least 1.
fn chunk_length(chars: &[(usize, char)], i: usize) -> usize {
    let at = |k: usize| chars.get(k).map(|(_, c)| *c);
    let c = chars[i].1;
    if c == '\'' {
        match (at(i + 1), at(i + 2)) {
            (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => return 3,
            (Some('s'), _) | (Some('t'), _) | (Some('m'), _) | (Some('d'), _) => return 2,
            _ => {}
        }
    }
    let lead = usize::from(c == ' ');
    let run = |from: usize, test: &dyn Fn(char) -> bool| {
        let mut k = from;
        while let Some(c) = at(k) {
            if !test(c) {
                break;
            }
            k += 1;
        }
        k - from
    };
    let other = |c: char| !c.is_whitespace() && !is_letter(c) && !is_number(c);
    for test in [&is_letter as &dyn Fn(char) -> bool, &is_number, &other] {
        let n = run(i + lead, test);
        if n > 0 {
            return lead + n;
        }
    }
    // only white space can be here now
    let spaces = run(i, &|c: char| c.is_whitespace());
    if i + spaces == chars.len() || spaces == 1 {
        spaces.max(1)
    } else {
        // \s+(?!\S): leave the last space for the word that follows
        spaces - 1
    }
}

impl Tokenizer for ByteBpe {
    fn vocab_size(&self) -> usize {
        self.by_id.len()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // the earliest special token in what is left, if any
            let next = self.special.iter().filter_map(|(name, id)| rest.find(name.as_str()).map(|at| (at, name.len(), *id))).min();
            match next {
                Some((at, length, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + length..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            self.bytes_of(*id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn piece(&self, id: u32) -> String {
        let mut bytes = Vec::new();
        self.bytes_of(id, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
