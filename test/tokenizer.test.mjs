// Checks the byte-level BPE tokenizer against token ids published for GPT-2
// and against round trips through awkward text.
//
//   node test/tokenizer.test.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ByteBPE } from "../src/core/tokenizers/byte-bpe.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const vocab = JSON.parse(readFileSync(join(ROOT, "assets/gpt2/vocab.json"), "utf8"));
const merges = readFileSync(join(ROOT, "assets/gpt2/merges.txt"), "utf8");
const bpe = new ByteBPE(vocab, merges, { special: { "<|endoftext|>": 50256 } });

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
};
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// ids from OpenAI's encoder and the GPT-3-Encoder test suite
const known = [
  ["Hello world", [15496, 995]],
  ["Hello, world!", [15496, 11, 995, 0]],
  [" Hello", [18435]],
  ["The quick brown fox jumps over the lazy dog", [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]],
  ["hello 👋 world 🌍", [31373, 50169, 233, 995, 12520, 234, 235]],
  ["This is some text", [1212, 318, 617, 2420]],
  ["indivisible", [521, 452, 12843]],
  ["\t", [197]],
  // words that are also properties of every JavaScript object
  ["toString constructor hasOwnProperty valueOf", [1462, 10100, 23772, 468, 23858, 21746, 1988, 5189]],
  [" ", [220]],
  ["", []],
  ["<|endoftext|>", [50256]],
  ["a<|endoftext|>b", [64, 50256, 65]],
];
for (const [text, ids] of known) {
  const got = bpe.encode(text);
  check(`encode ${JSON.stringify(text)}`, same(got, ids), same(got, ids) ? "" : `got [${got}] want [${ids}]`);
}

check("vocabulary has 50,257 entries", bpe.size === 50257, `(${bpe.size})`);

const awkward = [
  "It's the model's 3rd try — isn't it?\n\n  Tabs\tand   spaces.",
  "naïve café, 東京, Ελληνικά, עברית, 🧠🔬",
  "def f(x):\n    return x**2  # squared\n",
  "    leading and trailing    ",
];
for (const text of awkward) {
  const ids = bpe.encode(text);
  check(`round trip ${JSON.stringify(text.slice(0, 24))}…`, bpe.decode(ids) === text, `(${ids.length} tokens)`);
}
check("a token that is half a character still has a displayable piece", typeof bpe.piece(50169) === "string");
check("piece of ' world'", bpe.piece(995) === " world", JSON.stringify(bpe.piece(995)));

console.log(failures ? `\n${failures} check(s) failed` : "\nall tokenizer checks passed");
process.exit(failures ? 1 : 0);
