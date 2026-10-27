// Writes a GPT-2-shaped safetensors checkpoint filled with random numbers.
// Every tensor, including biases and LayerNorm gains, is random, so any wiring
// mistake in the forward pass changes the output. Used by test/ and for speed
// measurements; it is not a trained model.
//
//   node tools/make-test-weights.mjs out.safetensors [--layers 2] [--heads 4]
//        [--width 64] [--vocab 50257] [--ctx 128] [--seed 3] [--prefix transformer.]
//        [--masks]    also write the h.N.attn.bias mask buffers real checkpoints carry

import { writeFileSync } from "node:fs";
import { writeSafetensors } from "../src/core/safetensors.js";
import { gaussianFiller } from "../src/core/weights.js";

const args = process.argv.slice(2);
const out = args[0];
const flag = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  return at < 0 ? fallback : args[at + 1];
};
if (!out) {
  console.error("usage: node tools/make-test-weights.mjs out.safetensors [--layers N] [--heads N] [--width N] [--vocab N] [--ctx N]");
  process.exit(1);
}
const L = Number(flag("layers", 2)), H = Number(flag("heads", 4)), d = Number(flag("width", 64));
const V = Number(flag("vocab", 50257)), ctx = Number(flag("ctx", 128)), seed = Number(flag("seed", 3));
const prefix = flag("prefix", "");
const masks = args.includes("--masks");
const real = args.includes("--init"); // GPT-2's own initialisation instead of loud noise
const m = 4 * d;
const fill = gaussianFiller(seed);

const tensors = [];
const add = (name, shape, std, mean = 0) => {
  const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  fill(data, std, mean);
  tensors.push({ name: prefix + name, shape, data });
};
const w = real ? 0.02 : 0.08, b = real ? 0 : 0.05, g = real ? 0 : 0.1;
add("wte.weight", [V, d], w);
add("wpe.weight", [ctx, d], real ? 0.01 : 0.05);
for (let l = 0; l < L; l++) {
  const p = `h.${l}`;
  if (masks) {
    const mask = new Float32Array(ctx * ctx);
    for (let i = 0; i < ctx; i++) for (let j = 0; j <= i; j++) mask[i * ctx + j] = 1;
    tensors.push({ name: `${prefix}${p}.attn.bias`, shape: [1, 1, ctx, ctx], data: mask });
  }
  add(`${p}.ln_1.weight`, [d], g, 1);
  add(`${p}.ln_1.bias`, [d], b);
  add(`${p}.attn.c_attn.weight`, [d, 3 * d], w);
  add(`${p}.attn.c_attn.bias`, [3 * d], b);
  add(`${p}.attn.c_proj.weight`, [d, d], w);
  add(`${p}.attn.c_proj.bias`, [d], b);
  add(`${p}.ln_2.weight`, [d], g, 1);
  add(`${p}.ln_2.bias`, [d], b);
  add(`${p}.mlp.c_fc.weight`, [d, m], w);
  add(`${p}.mlp.c_fc.bias`, [m], b);
  add(`${p}.mlp.c_proj.weight`, [m, d], w);
  add(`${p}.mlp.c_proj.bias`, [d], b);
}
add("ln_f.weight", [d], g, 1);
add("ln_f.bias", [d], b);

writeFileSync(out, writeSafetensors(tensors));
const params = tensors.reduce((sum, t) => sum + t.data.length, 0);
console.log(`${out}: ${L} layers, ${H} heads, width ${d}, vocab ${V}, ${params.toLocaleString()} numbers`);
