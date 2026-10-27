// GPT-2: decoder-only transformer. This module is everything the engine needs
// to know about the architecture; nothing outside src/arch/gpt2 mentions it.

import { ByteBPE } from "../../core/tokenizers/byte-bpe.js";
import { gaussianFiller } from "../../core/weights.js";
import { createGPT2Runtime, gpt2ScratchFloats } from "./forward.js";
import { gpt2ParamCount, gpt2Spec } from "./spec.js";

const SKIP = /(^|\.)(attn\.bias|attn\.masked_bias|lm_head\.weight)$/;
const strip = (name) => name.replace(/^(transformer|model|gpt2)\./, "");

function requiredNames(L) {
  const names = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"];
  for (let l = 0; l < L; l++) {
    const p = `h.${l}`;
    names.push(
      `${p}.ln_1.weight`, `${p}.ln_1.bias`, `${p}.attn.c_attn.weight`, `${p}.attn.c_attn.bias`,
      `${p}.attn.c_proj.weight`, `${p}.attn.c_proj.bias`, `${p}.ln_2.weight`, `${p}.ln_2.bias`,
      `${p}.mlp.c_fc.weight`, `${p}.mlp.c_fc.bias`, `${p}.mlp.c_proj.weight`, `${p}.mlp.c_proj.bias`,
    );
  }
  return names;
}

function shapesOf(c) {
  const d = c.n_embd, m = c.n_inner;
  const shapes = { "wte.weight": [c.vocab_size, d], "wpe.weight": [c.n_ctx, d], "ln_f.weight": [d], "ln_f.bias": [d] };
  for (let l = 0; l < c.n_layer; l++) {
    const p = `h.${l}`;
    Object.assign(shapes, {
      [`${p}.ln_1.weight`]: [d], [`${p}.ln_1.bias`]: [d],
      [`${p}.attn.c_attn.weight`]: [d, 3 * d], [`${p}.attn.c_attn.bias`]: [3 * d],
      [`${p}.attn.c_proj.weight`]: [d, d], [`${p}.attn.c_proj.bias`]: [d],
      [`${p}.ln_2.weight`]: [d], [`${p}.ln_2.bias`]: [d],
      [`${p}.mlp.c_fc.weight`]: [d, m], [`${p}.mlp.c_fc.bias`]: [m],
      [`${p}.mlp.c_proj.weight`]: [m, d], [`${p}.mlp.c_proj.bias`]: [d],
    });
  }
  return shapes;
}

export default {
  id: "gpt2",
  name: "GPT-2",
  family: "Decoder-only transformer",

  spec: gpt2Spec,
  paramCount: gpt2ParamCount,

  /**
   * Reads the tensor list of a checkpoint and decides what to keep.
   * The config is taken from the tensor shapes; only the head count, which no
   * shape records, comes from the model card.
   */
  plan(tensors, card) {
    const byName = new Map();
    for (const tensor of tensors) {
      const name = strip(tensor.name);
      if (!SKIP.test(name)) byName.set(name, tensor);
    }
    const wte = byName.get("wte.weight"), wpe = byName.get("wpe.weight");
    if (!wte || !wpe) throw new Error("This file is not a GPT-2 checkpoint: it has no wte.weight and wpe.weight.");
    let L = 0;
    while (byName.has(`h.${L}.attn.c_attn.weight`)) L++;
    if (!L) throw new Error("This file is not a GPT-2 checkpoint: it has no h.0.attn.c_attn.weight.");
    const d = wte.shape[1];
    const config = {
      ...card.config,
      vocab_size: wte.shape[0],
      n_ctx: wpe.shape[0],
      n_embd: d,
      n_layer: L,
      n_inner: byName.get("h.0.mlp.c_fc.weight").shape[1],
    };
    config.n_head = card.config?.n_embd === d && card.config?.n_head ? card.config.n_head : Math.max(1, Math.round(d / 64));
    if (d % config.n_head) throw new Error(`The embedding width ${d} does not divide into ${config.n_head} heads.`);
    const expected = shapesOf(config);
    const wanted = [];
    for (const name of requiredNames(L)) {
      const tensor = byName.get(name);
      if (!tensor) throw new Error(`The checkpoint is missing "${name}".`);
      if (tensor.shape.join("x") !== expected[name].join("x")) {
        throw new Error(`"${name}" has shape [${tensor.shape}], but a GPT-2 of this size needs [${expected[name]}].`);
      }
      wanted.push({ file: tensor.name, name, shape: tensor.shape });
    }
    return { config, wanted };
  },

  weightFloats(config) {
    return Object.values(shapesOf(config)).reduce((sum, shape) => sum + shape.reduce((a, b) => a * b, 1), 0);
  },

  scratchFloats: gpt2ScratchFloats,

  /** Random weights with GPT-2's initialisation, for the test pattern. */
  synthesize(config, store, arena, seed = 1) {
    const fill = gaussianFiller(seed);
    for (const [name, shape] of Object.entries(shapesOf(config))) {
      const data = store.add(name, shape, arena.alloc(shape.reduce((a, b) => a * b, 1)));
      if (/ln_\w+\.weight$/.test(name)) data.fill(1);
      else if (name.endsWith(".bias")) data.fill(0);
      else fill(data, name === "wpe.weight" ? 0.01 : 0.02);
    }
  },

  async createTokenizer(card, readAsset) {
    const [vocab, merges] = await Promise.all([readAsset(card.tokenizer.vocab, "json"), readAsset(card.tokenizer.merges, "text")]);
    const tokenizer = new ByteBPE(vocab, merges, { special: { "<|endoftext|>": vocab["<|endoftext|>"] } });
    return {
      vocabSize: tokenizer.size,
      encode: (text) => tokenizer.encode(text),
      decode: (ids) => tokenizer.decode(ids),
      piece: (id) => tokenizer.piece(id),
    };
  },

  createRuntime: createGPT2Runtime,
};
