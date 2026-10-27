// Sizes and counts for the text above each diagram: the shape of every matrix
// in an operation, the parameters a unit or a step holds, and the
// multiply-adds it does. Read from the spec and from the operation itself,
// so another architecture gets them without changes here.

export const n = (x) => Math.round(x).toLocaleString("en-US");
export const shape = (rows, cols) => `${n(rows)} × ${n(cols)}`;

/** Short form of a big count: 7.09 M, 1.6 G. */
export function short(x) {
  if (x >= 1e9) return `${(x / 1e9).toFixed(x >= 1e10 ? 1 : 2)} G`;
  if (x >= 1e6) return `${(x / 1e6).toFixed(x >= 1e7 ? 1 : 2)} M`;
  if (x >= 1e4) return `${(x / 1e3).toFixed(0)} K`;
  return n(x);
}

const esc = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** One matrix as a chip: its name and its size. kind: "act" (computed), "weight" (learned). */
export const chip = (name, rows, cols, kind = "act") =>
  `<span class="na-dim" data-kind="${kind}"><b>${esc(name)}</b> <span>${cols === undefined ? n(rows) : shape(rows, cols)}</span></span>`;

/** Parameters a unit holds, counting a tied matrix once (where it is first used). */
export function unitParams(spec, unit) {
  const seen = new Set();
  let total = 0, shared = 0;
  for (const u of spec.units) {
    for (const w of [...(u.weights ?? []), ...(u.vectors ?? [])]) {
      const key = `${w.view.tensor}|${w.view.col0 ?? 0}|${w.view.row0 ?? 0}`;
      const count = w.view.rows * w.view.cols;
      const first = !seen.has(key);
      seen.add(key);
      if (u === unit) { if (first) total += count; else shared += count; }
    }
  }
  return { total, shared };
}

export function modelParams(spec) {
  return spec.units.reduce((sum, u) => sum + unitParams(spec, u).total, 0);
}

/**
 * The sizes line for one operation of the bench's "One token" mode:
 * input × matrix (+ bias) = output, what of it is learned, and the work.
 */
export function opDims(app, unitIndex, op) {
  const { spec, tokens } = app.state;
  const unit = spec.units[unitIndex];
  const T = tokens.ids.length;
  const rows = op.matrix.rows, cols = op.matrix.cols;
  const learned = !op.matrix.activation;
  const params = learned ? rows * cols + (op.bias ? op.bias.length : 0) : 0;
  const macs = op.lookup ? 0 : rows * cols;
  const parts = [
    chip(op.input.label, 1, rows),
    `<i>×</i>`,
    chip(op.matrix.label, rows, cols, learned ? "weight" : "act"),
  ];
  if (op.bias) parts.push(`<i>+</i>`, chip("bias", op.bias.length, undefined, "weight"));
  parts.push(`<i>=</i>`, chip(op.output.label, 1, cols));
  const work = op.lookup
    ? `as a product, ${n(rows * cols)} multiply-adds; in practice a lookup, so none`
    : `<b>${n(macs)}</b> multiply-adds for this token${op.perToken === false ? "" : ` · <b>${short(macs * T)}</b> for all ${n(T)} tokens`}`;
  const what = learned
    ? `<b>${n(params)}</b> parameters in this step${op.matrix.shared ? " (the same table as the embedding)" : ""}`
    : "no parameters here: both sides were computed from your text";
  const u = unitParams(spec, unit);
  const holds = `${unit.title} holds <b>${n(u.total)}</b> parameters${u.shared ? ` and reuses ${short(u.shared)}` : ""} of the model's ${short(modelParams(spec))}`;
  return `<div class="na-dims-row">${parts.join(" ")}</div><div class="na-dims-row na-dims-count">${what} · ${work} · ${holds}</div>`;
}

/**
 * The sizes line for a whole block on every token (the "All tokens" mode):
 * each matrix product with its shapes, its parameters and its work.
 */
export function blockDims(spec, unit, T) {
  if (unit.kind !== "block" || !unit.parts?.attention) {
    const u = unitParams(spec, unit);
    const d = unit.stream.size;
    if (unit.kind === "embed") {
      const [tok, pos] = unit.weights;
      return `<div class="na-dims-row">${chip("token ids", T, 1)} <i>→ rows of</i> ${chip("W_E", tok.view.rows, tok.view.cols, "weight")} <i>+ rows of</i> ${chip("W_pos", pos.view.rows, pos.view.cols, "weight")} <i>=</i> ${chip("stream", T, d)}</div>
        <div class="na-dims-row na-dims-count"><b>${n(u.total)}</b> parameters · lookups, so no multiply-adds, only ${n(T * d)} additions</div>`;
    }
    const V = unit.weights[0].view.rows;
    return `<div class="na-dims-row">${chip("stream", T, d)} <i>×</i> ${chip("W_Eᵀ", d, V, "weight")} <i>=</i> ${chip("scores", T, V)}</div>
      <div class="na-dims-row na-dims-count"><b>${n(u.total)}</b> new parameters (the token table is shared with the embedding) · <b>${short(d * V)}</b> multiply-adds per row · <b>${short(d * V * T)}</b> for all ${n(T)} rows</div>`;
  }
  const a = unit.parts.attention;
  const d = unit.stream.size, m = unit.neurons.size, H = a.heads, s = a.headSize;
  const kv = (a.kvHeads ?? H) * s;
  const pairs = (a.causal ?? true) ? (T * (T + 1)) / 2 : T * T;
  const rows = [
    ["Q, K, V", [chip("X", T, d), "×", chip("W_QKV", d, d + 2 * kv, "weight"), "=", chip("QKV", T, d + 2 * kv)], d * (d + 2 * kv) + d + 2 * kv, T * d * (d + 2 * kv)],
    ["scores", [chip("Q", T, s), "×", chip("Kᵀ", s, T), "=", chip("S", T, T), `<i>× ${H} heads</i>`], 0, H * pairs * s],
    ["mix", [chip("P", T, T), "×", chip("V", T, s), "=", chip("head", T, s), `<i>× ${H} heads</i>`], 0, H * pairs * s],
    ["join heads", [chip("heads", T, d), "×", chip("W_O", d, d, "weight"), "=", chip("attn", T, d)], d * d + d, T * d * d],
    ["MLP up", [chip("X", T, d), "×", chip("W_in", d, m, "weight"), "=", chip("neurons", T, m)], d * m + m, T * d * m],
    ["MLP down", [chip("neurons", T, m), "×", chip("W_out", m, d, "weight"), "=", chip("MLP", T, d)], m * d + d, T * m * d],
  ];
  const u = unitParams(spec, unit);
  const totalMacs = rows.reduce((sum, r) => sum + r[3], 0);
  return `<table class="na-dims-table"><thead><tr><th>step</th><th>shapes, for ${n(T)} tokens</th><th>parameters</th><th>multiply-adds</th></tr></thead><tbody>${rows
    .map(([name, parts, params, macs]) => `<tr><th>${name}</th><td>${parts.map((p) => (p.startsWith("<") ? p : `<i>${p}</i>`)).join(" ")}</td><td>${params ? n(params) : "—"}</td><td>${n(macs)}</td></tr>`)
    .join("")}</tbody><tfoot><tr><th>${unit.title}</th><td>plus two LayerNorms (${n(4 * d)} parameters)</td><td>${n(u.total)}</td><td>${n(totalMacs)}</td></tr></tfoot></table>`;
}

/** The sizes line for one head on a GPU. */
export function headDims(head) {
  const { T, s, H } = head;
  const pairs = head.causal ? (T * (T + 1)) / 2 : T * T;
  return `<div class="na-dims-row">${chip("Q", T, s)} <i>×</i> ${chip("Kᵀ", s, T)} <i>=</i> ${chip("S", T, T)} <i>→ softmax →</i> ${chip("P", T, T)} <i>×</i> ${chip("V", T, s)} <i>=</i> ${chip("O", T, s)}</div>
    <div class="na-dims-row na-dims-count">no parameters: Q, K and V come from W_QKV in the step before · <b>${n(2 * pairs * s)}</b> multiply-adds for this head (${n(pairs)} query–key pairs${head.causal ? " the mask allows" : ""}, twice ${s}) · <b>${n(2 * pairs * s * H)}</b> for all ${H} heads</div>`;
}
