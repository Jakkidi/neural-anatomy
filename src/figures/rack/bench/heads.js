// One attention head's queries, keys and values for every token, redone from
// the weights. The "All tokens" view and the GPU view both start here, and it
// reads everything it needs from the spec's `parts.attention`, so another
// architecture only has to describe where its tensors are:
//
//   attention: {
//     heads, headSize,
//     kvHeads,            // optional: fewer key/value heads than query heads (GQA, MQA)
//     causal,             // optional, default true: a token sees itself and earlier tokens
//     scale,              // optional, default 1/√headSize
//     ln: { gain, bias, kind },        // the norm before attention; kind "rms" for RMSNorm
//     qkv: { tensor, bias, query, key, value },   // one fused [d, …] matrix and column offsets (GPT-2)
//     // or separate projections, each [d, heads·headSize] (transpose: true if stored [out, in]):
//     q: { tensor, bias, transpose }, k: { … }, v: { … },
//   }
//
// Rotary position embeddings (RoPE) are not applied here yet; a spec that
// declares `rope` gets a clear message instead of wrong numbers.

import { readExact } from "./ops.js";

const memo = new Map();

function norm(x, gain, bias, eps, kind) {
  const n = x.length;
  let mean = 0;
  if (kind !== "rms") { for (let i = 0; i < n; i++) mean += x[i]; mean /= n; }
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * gain[i] + (bias ? bias[i] : 0);
  return out;
}

/** The columns of one projection that belong to a head: { weights [d, s], bias [s] }. */
async function projection(app, attention, which, headIndex, d, cache) {
  const s = attention.headSize;
  const fused = attention.qkv;
  if (fused) {
    const col0 = fused[which === "q" ? "query" : which === "k" ? "key" : "value"] + headIndex * s;
    const [weights, bias] = await Promise.all([
      readExact(app, { tensor: fused.tensor, col0, rows: d, cols: s }, cache),
      fused.bias ? readExact(app, { tensor: fused.bias, col0, rows: 1, cols: s }, cache) : null,
    ]);
    return { weights, bias };
  }
  const part = attention[which];
  if (!part) throw new Error(`The spec does not say where the ${which} projection is.`);
  // shown [d, s] either way: with transpose, display rows are the tensor's columns
  const view = { tensor: part.tensor, col0: headIndex * s, rows: d, cols: s, transpose: Boolean(part.transpose) };
  const [weights, bias] = await Promise.all([
    readExact(app, view, cache),
    part.bias ? readExact(app, { tensor: part.bias, col0: headIndex * s, rows: 1, cols: s }, cache) : null,
  ]);
  return { weights, bias };
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

/**
 * Q, K and V of one head for the first T tokens, plus ordinary attention on
 * them (the shares and the head's output), checked against the engine.
 */
export async function headTensors(app, unitIndex, head, T, cache) {
  const { spec, trace, tokens } = app.state;
  const unit = spec.units[unitIndex];
  const attention = unit.parts?.attention;
  if (!attention) throw new Error(`${unit.title} has no attention.`);
  if (attention.rope) throw new Error("This view does not apply rotary position embeddings yet, so it cannot redo this model's attention.");
  const key = `${app.state.weightsStamp}|${trace.version}|${unitIndex}|${head}|${T}`;
  if (memo.has(key)) return memo.get(key);
  const before = spec.units[unitIndex - 1];
  const d = unit.stream.size, H = attention.heads, s = attention.headSize;
  const kvHeads = attention.kvHeads ?? H;
  const h = Math.max(0, Math.min(H - 1, head));
  const kvHead = Math.floor(h / (H / kvHeads));
  const causal = attention.causal ?? true;
  const scale = attention.scale ?? 1 / Math.sqrt(s);
  const eps = unit.parts.eps ?? 1e-5;
  const [gain, bias, pq, pk, pv] = await Promise.all([
    readExact(app, { tensor: attention.ln.gain, rows: 1, cols: d }, cache),
    attention.ln.bias ? readExact(app, { tensor: attention.ln.bias, rows: 1, cols: d }, cache) : null,
    projection(app, attention, "q", h, d, cache),
    projection(app, attention, "k", kvHead, d, cache),
    projection(app, attention, "v", kvHead, d, cache),
  ]);
  const X = rowsOf(trace, before.stream.trace, T, d);
  const Q = new Float32Array(T * s), K = new Float32Array(T * s), V = new Float32Array(T * s);
  const parts = [[pq, Q], [pk, K], [pv, V]];
  const acc = new Float64Array(s);
  for (let i = 0; i < T; i++) {
    const x = norm(X.subarray(i * d, (i + 1) * d), gain, bias, eps, attention.ln.kind);
    for (const [{ weights, bias: b }, out] of parts) {
      for (let k = 0; k < s; k++) acc[k] = b ? b[k] : 0;
      for (let r = 0; r < d; r++) {
        const v = x[r], base = r * s;
        if (v !== 0) for (let k = 0; k < s; k++) acc[k] += v * weights[base + k];
      }
      for (let k = 0; k < s; k++) out[i * s + k] = acc[k];
    }
  }
  // ordinary attention: the T × T shares, then the weighted sum of values
  const shares = new Float32Array(T * T);
  const pattern = unit.attention?.trace ? trace.entry(unit.attention.trace) : null;
  const C = pattern ? pattern.rowSize / H : 0;
  let worst = 0;
  for (let i = 0; i < T; i++) {
    const last = causal ? i : T - 1;
    const scores = new Float64Array(last + 1);
    let top = -Infinity;
    for (let j = 0; j <= last; j++) {
      let dot = 0;
      for (let k = 0; k < s; k++) dot += Q[i * s + k] * K[j * s + k];
      scores[j] = dot * scale;
      top = Math.max(top, scores[j]);
    }
    let z = 0;
    for (let j = 0; j <= last; j++) { scores[j] = Math.exp(scores[j] - top); z += scores[j]; }
    const engineRow = pattern ? trace.row(unit.attention.trace, i) : null;
    for (let j = 0; j <= last; j++) {
      shares[i * T + j] = scores[j] / z;
      if (engineRow) worst = Math.max(worst, Math.abs(shares[i * T + j] - engineRow[h * C + j]));
    }
  }
  const out = new Float32Array(T * s);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    const w = shares[i * T + j];
    if (w) for (let k = 0; k < s; k++) out[i * s + k] += w * V[j * s + k];
  }
  const result = {
    unit, unitIndex, T, d, s, H, h, kvHead, kvHeads, causal, scale,
    pieces: tokens.pieces.slice(0, T), X, Q, K, V, shares, out,
    check: pattern ? { worst, ok: worst < 2e-3 } : null,
  };
  memo.clear();
  memo.set(key, result);
  return result;
}
