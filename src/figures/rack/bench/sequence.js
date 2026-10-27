// The whole sequence at once: what one unit computes for every token of the
// text, side by side, one row per token. This is how the model really runs a
// prompt: each weight matrix multiplies all the rows together, and only
// attention lets a row read from other rows (and only from earlier ones).
//
// It also carries the arithmetic for the two ways of generating text one token
// at a time: running every token again at each step, or keeping each token's
// keys and values (the KV cache) so that only the new token has to run.

import { headTensors } from "./heads.js";

const show = (piece) => String(piece ?? "").replace(/^ /, "·").replace(/\n/g, "↵");

/** How many rows of a unit have been computed by the last run. */
export function rowsReady(app, unitIndex) {
  const { spec, trace, tokens } = app.state;
  const unit = spec.units[unitIndex];
  let T = tokens.ids.length;
  while (T > 0 && !trace?.has(unit.stream.trace, T - 1)) T--;
  return T;
}

/** Rows [0, T) of a traced array, laid end to end. */
function rowsOf(trace, key, T, n) {
  const out = new Float32Array(T * n);
  for (let i = 0; i < T; i++) {
    const row = trace.row(key, i);
    if (row) out.set(row.subarray(0, n), i * n);
  }
  return out;
}

const memo = new Map();

/**
 * The stages one unit goes through for every token, in order. Each stage is a
 * matrix with one row per token, or the token × token grid of attention.
 *   { id, label, shape, n, data, kind: "rows" | "grid" | "text", role, via, step, columnOf }
 * `via` is the operation that leads into the stage; `mixes` marks the ones that
 * move information between tokens. `role` says what the KV cache keeps.
 */
export async function buildSequence(app, unitIndex, { head = 0 } = {}, cache) {
  const { spec, trace, tokens } = app.state;
  const unit = spec.units[unitIndex];
  const T = rowsReady(app, unitIndex);
  if (!T) throw new Error("This unit has not run yet.");
  const key = `${app.state.weightsStamp}|${trace.version}|${unitIndex}|${head}|${T}`;
  if (memo.has(key)) return memo.get(key);
  const pieces = tokens.pieces.slice(0, T);
  const base = { unit, unitIndex, T, pieces, tokenName: (i) => `token ${i + 1} “${show(pieces[i])}”` };
  let result;

  if (unit.kind === "embed") {
    const d = unit.stream.size;
    const write = (id) => unit.writes.find((w) => w.id === id).trace;
    result = {
      ...base,
      stages: [
        { id: "ids", kind: "text", label: "token id", shape: `${T} ids`, text: (i) => String(tokens.ids[i]), step: "tok" },
        { id: "tok", kind: "rows", label: "token vectors", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, write("tok"), T, d), role: "row", step: "tok", via: { label: "look up row id of W_E", short: "W_E[id]", mixes: false } },
        { id: "pos", kind: "rows", label: "position vectors", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, write("pos"), T, d), role: "row", step: "pos", via: { label: "look up row i of W_pos", short: "W_pos[i]", mixes: false } },
        { id: "out", kind: "rows", label: "stream out", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, unit.stream.trace, T, d), role: "row", step: "pos", via: { label: "token vector + position vector", short: "+", mixes: false } },
      ],
      check: { ok: true, text: "Each row is looked up on its own: the embedding never mixes tokens." },
    };
  } else if (unit.kind === "unembed") {
    const d = unit.stream.size;
    result = {
      ...base,
      stages: [
        { id: "stream", kind: "rows", label: "finished stream", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, unit.stream.trace, T, d), role: "row", step: "logits" },
        {
          id: "next", kind: "text", label: "predicted next token", shape: `${T} rows of 50,257 scores`, role: "row", step: "logits", wide: true,
          via: { label: "× W_Eᵀ (one score per token), softmax", short: "× W_Eᵀ", mixes: false },
          text: (i) => {
            const p = trace.predictions[i];
            if (!p) return "";
            const actual = i + 1 < tokens.ids.length ? `   (text has “${show(tokens.pieces[i + 1])}”: ${p.targetProb != null ? `${(p.targetProb * 100).toFixed(1)}%` : "?"})` : "   ← the prediction";
            return `“${show(p.pieces[0])}” ${(p.probs[0] * 100).toFixed(1)}%${actual}`;
          },
        },
      ],
      check: { ok: true, text: `Every row predicts its own next token. Training uses all ${T} predictions at once; generating uses only the last row.` },
    };
  } else {
    const { attention } = unit.parts;
    const d = unit.stream.size, H = attention.heads, s = attention.headSize, m = unit.neurons.size;
    const head_ = await headTensors(app, unitIndex, head, T, cache);
    const { h, X, Q, K, V, shares, out: mixed } = head_;
    const worst = head_.check?.worst ?? 0;
    const write = (id) => unit.writes.find((w) => w.id === id).trace;
    const part = (offset) => (k) => offset + h * s + k;
    result = {
      ...base,
      head: h, H, s, d, m,
      stages: [
        { id: "x", kind: "rows", label: "stream in", shape: `${T} × ${d}`, n: d, data: X, role: "row", step: "qkv" },
        { id: "q", kind: "rows", label: `query h${h}`, shape: `${T} × ${s}`, n: s, data: Q, role: "q", step: "qkv", columnOf: part(0), via: { label: "normalise · × W_QKV + b", short: "× W_QKV", mixes: false, weight: `W_QKV ${d}×${3 * d}` } },
        { id: "k", kind: "rows", label: `key h${h}`, shape: `${T} × ${s}`, n: s, data: K, role: "kv", step: "qkv", columnOf: part(d), joined: true },
        { id: "v", kind: "rows", label: `value h${h}`, shape: `${T} × ${s}`, n: s, data: V, role: "kv", step: "qkv", columnOf: part(2 * d), joined: true },
        { id: "shares", kind: "grid", label: `attention h${h}`, shape: `${T} × ${T}`, n: T, data: shares, role: "grid", step: "scores", via: { label: "Q·Kᵀ ÷ √64, mask the future, softmax", short: "Q·Kᵀ", mixes: true } },
        { id: "mix", kind: "rows", label: `head ${h} out`, shape: `${T} × ${s}`, n: s, data: mixed, role: "row", step: "mix", via: { label: "shares × V", short: "× V", mixes: true } },
        { id: "attn", kind: "rows", label: "attention adds", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, write("attn"), T, d), role: "row", step: "wo", via: { label: `join ${H} heads · × W_O + b`, short: "× W_O", mixes: false, weight: `W_O ${d}×${d}` } },
        { id: "neurons", kind: "rows", label: "neurons", shape: `${T} × ${m}`, n: m, data: rowsOf(trace, unit.neurons.trace, T, m), role: "row", step: "up", positiveOnly: true, via: { label: "+ stream in · normalise · × W_in + b · GELU", short: "× W_in", mixes: false, weight: `W_in ${d}×${m}` } },
        { id: "mlp", kind: "rows", label: "MLP adds", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, write("mlp"), T, d), role: "row", step: "down", via: { label: "× W_out + b", short: "× W_out", mixes: false, weight: `W_out ${m}×${d}` } },
        { id: "out", kind: "rows", label: "stream out", shape: `${T} × ${d}`, n: d, data: rowsOf(trace, unit.stream.trace, T, d), role: "row", step: "down", via: { label: "stream in + attention adds + MLP adds", short: "+", mixes: false } },
      ],
      check: {
        ok: worst < 2e-3,
        text: `${worst < 2e-3 ? "✓" : "✗"} All ${(T * (T + 1) / 2).toLocaleString()} shares of head ${h}, recomputed here from the weights, match the engine's attention: largest difference ${worst.toExponential(1)}.`,
      },
    };
  }
  memo.clear();
  memo.set(key, result);
  return result;
}

/**
 * Multiply-adds for the whole model, to generate with or without the KV cache.
 * `context` is how many tokens the model has at that step.
 */
export function costModel(spec) {
  const blocks = spec.units.filter((u) => u.kind === "block");
  const block = blocks[0];
  const d = block.stream.size, m = block.neurons.size;
  const L = blocks.length;
  const unembed = spec.units.find((u) => u.kind === "unembed");
  const vocab = unembed?.weights?.[0]?.view?.rows ?? 50257;
  const perToken = L * (d * 3 * d + d * d + d * m + m * d); // the weight matrices, one token
  const attention = (context) => L * 2 * d * context; // one query against `context` keys, then the mix
  const logits = d * vocab; // scores for the next token, from the last row
  return {
    L, d, m, vocab, perToken, logits,
    /** one generation step, all earlier tokens run again */
    noCache: (t) => t * perToken + L * d * t * (t + 1) + logits,
    /** one generation step, earlier keys and values read from memory */
    withCache: (t) => perToken + attention(t) + logits,
    /** the prompt read in one pass */
    prompt: (T) => T * perToken + L * d * T * (T + 1) + logits,
    /** bytes the cache holds after t tokens: a key and a value per token, per block, float32 */
    cacheBytes: (t) => 2 * L * d * t * 4,
  };
}
