// Tokenizes a few thousand awkward strings with the JavaScript tokenizer and
// with the Rust one, and reports any string they disagree on.
//
//   node tools/compare-tokenizers.mjs        (needs cargo; builds the Rust example)
//
// U+0085 and U+FEFF are left out: JavaScript's \s and Unicode's White_Space
// disagree about those two, and so would the two tokenizers.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ByteBPE } from "../src/core/tokenizers/byte-bpe.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const bpe = new ByteBPE(JSON.parse(readFileSync(join(ROOT, "assets/gpt2/vocab.json"), "utf8")), readFileSync(join(ROOT, "assets/gpt2/merges.txt"), "utf8"), { special: { "<|endoftext|>": 50256 } });

let seed = 12345;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const pick = (list) => list[Math.floor(random() * list.length)];
const words = ["the", "The", "model", "Paris", "isn", "don", "we", "I", "it", "x1", "42", "3.14", "naïve", "café", "東京", "Ελληνικά", "עברית", "हिन्दी", "한국어", "русский", "ⅻ", "²", "½", "٣", "Á", "ë", "👋", "🧠🔬", "👨‍👩‍👧", "Ⓐ", "™", "€", "—", "…", "<|endoftext|>", "<|endoftext", "toString", "__proto__", "constructor"];
const glue = [" ", "  ", "   ", "\n", "\n\n", "\t", " \n ", " ", " ", "　", "", "", "'", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", ",", ".", "!", "?", "-", "--", "(", ")", "[", "]", "{", "}", "\"", "/", "\\", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "|", "~", "`", "_", ":", ";"];
const strings = ["", " ", "  ", "\n", " \n", "\n ", "a ", " a", "a  b", "a\n\nb", "it's", "'s", "' s", "''", "'", "can't won't", "  x", "x  ", "1 2  3   4"];
for (let i = 0; i < 6000; i++) {
  let text = "";
  const parts = 1 + Math.floor(random() * 9);
  for (let p = 0; p < parts; p++) text += (random() < 0.75 ? pick(words) : String.fromCodePoint(0x20 + Math.floor(random() * 0x2fe0))) + pick(glue);
  strings.push(text.replace(/[\u0085﻿\ud800-\udfff]/g, "?"));
}
// every code point of the planes text lives on, in runs, to exercise the category tables
for (let base = 0x20; base < 0x30000; base += 61) {
  let text = "";
  for (let cp = base; cp < base + 61; cp++) if (cp !== 0x85 && cp !== 0xfeff && !(cp >= 0xd800 && cp <= 0xdfff)) text += String.fromCodePoint(cp) + (cp % 7 === 0 ? " " : "");
  strings.push(text);
}

const dir = mkdtempSync(join(tmpdir(), "na-tok-"));
writeFileSync(join(dir, "strings.json"), JSON.stringify(strings));
const out = execFileSync("cargo", ["run", "--quiet", "--offline", "--release", "-p", "na-engine", "--example", "tokenize", "--", join(dir, "strings.json")], { cwd: join(ROOT, "desktop"), maxBuffer: 1 << 30 }).toString().trim().split("\n");
let differ = 0, tokens = 0;
strings.forEach((text, i) => {
  const js = JSON.stringify(bpe.encode(text));
  tokens += bpe.encode(text).length;
  if (js !== out[i]) {
    if (differ++ < 5) console.log("DIFFERENT", JSON.stringify(text.slice(0, 60)), "\n  js  ", js.slice(0, 160), "\n  rust", out[i]?.slice(0, 160));
  }
});
console.log(`${strings.length} strings, ${tokens.toLocaleString()} tokens: ${differ ? differ + " strings tokenize differently" : "the two tokenizers agree on every one"}`);
process.exit(differ ? 1 : 0);
