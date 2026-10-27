// The operations inside each unit of the cabinet, as vector × matrix products
// on the real numbers: what goes in, the matrix it meets, what comes out, and
// a check against what the engine computed.
//
// Every operation has the same shape, which is what the bench draws:
//   output[j] = Σ_i input[i] · matrix[i, j]  (+ bias[j])   then optional steps after
//
// The big matrices of a block are read once (exact values, through the
// engine's "region" call) and the arithmetic is redone here, so each number on
// the bench can be followed back to the weights that made it.

export const layerNorm = (x, gain, bias, eps) => {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length;
  let variance = 0;
  for (let i = 0; i < x.length; i++) variance += (x[i] - mean) ** 2;
  variance /= x.length;
  const inv = 1 / Math.sqrt(variance + eps);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean) * inv * gain[i] + bias[i];
  return out;
};
export const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

/** y = x · W (+ b) for W stored row-major as [x.length, cols]. */
function vecmat(x, W, cols, bias) {
  const y = new Float64Array(cols);
  if (bias) for (let j = 0; j < cols; j++) y[j] = bias[j];
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (v === 0) continue;
    const base = i * cols;
    for (let j = 0; j < cols; j++) y[j] += v * W[base + j];
  }
  return Float32Array.from(y);
}
const maxDifference = (a, b) => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
};
const checkText = (difference, what) => ({
  ok: difference < 2e-3,
  text: `${difference < 2e-3 ? "✓" : "✗"} ${what} the engine computed: largest difference ${difference.toExponential(1)}.`,
});

/** The steps of a unit, in the order the model does them. */
export function stepsOf(unit) {
  if (unit.kind === "embed") {
    return [
      { id: "tok", label: "Look up the token" },
      { id: "pos", label: "Look up the position" },
    ];
  }
  if (unit.kind === "unembed") return [{ id: "logits", label: "Score every token" }];
  return [
    { id: "qkv", label: "Query, key, value" },
    { id: "scores", label: "Scores", perHead: true },
    { id: "mix", label: "Mix the values", perHead: true },
    { id: "wo", label: "Join the heads" },
    { id: "up", label: "MLP up + GELU" },
    { id: "down", label: "MLP down" },
  ];
}

/** Exact values of a rectangle of a stored tensor. */
export function readExact(app, view, cache) {
  const key = `${app.state.weightsStamp}|${view.tensor}|${view.row0 ?? 0}|${view.col0 ?? 0}|${view.rows}|${view.cols}|${Boolean(view.transpose)}`;
  if (!cache.has(key)) {
    cache.set(key, app.engine.call("region", { view: { row0: 0, col0: 0, transpose: false, ...view }, rows: view.rows, cols: view.cols }).then(({ data }) => Float32Array.from(data)));
    if (cache.size > 40) cache.delete(cache.keys().next().value);
  }
  return cache.get(key);
}

/**
 * Everything one block computes for the chosen token, redone from its weights:
 * queries, keys and values for every token so far, each head's scores and
 * shares, the heads joined, and the MLP. Checked against the engine's trace.
 */
async function blockState(app, unitIndex, cache) {
  const { spec, trace, position, tokens } = app.state;
  const unit = spec.units[unitIndex];
  const before = spec.units[unitIndex - 1];
  const { eps, attention, mlp } = unit.parts;
  const d = unit.stream.size, H = attention.heads, s = attention.headSize, m = unit.neurons.size;
  const T = position + 1;
  const streamIn = [];
  for (let j = 0; j < T; j++) {
    const row = trace.row(before.stream.trace, j);
    if (!row) throw new Error("This block has not run yet.");
    streamIn.push(row);
  }
  const [g1, b1, wqkv, bqkv, wo, bo, g2, b2, win, bin, wout, bout] = await Promise.all([
    readExact(app, { tensor: attention.ln.gain, rows: 1, cols: d }, cache),
    readExact(app, { tensor: attention.ln.bias, rows: 1, cols: d }, cache),
    readExact(app, { tensor: attention.qkv.tensor, rows: d, cols: 3 * d }, cache),
    readExact(app, { tensor: attention.qkv.bias, rows: 1, cols: 3 * d }, cache),
    readExact(app, { tensor: attention.output, rows: d, cols: d }, cache),
    readExact(app, { tensor: `${attention.output.replace(/weight$/, "bias")}`, rows: 1, cols: d }, cache),
    readExact(app, { tensor: mlp.ln.gain, rows: 1, cols: d }, cache),
    readExact(app, { tensor: mlp.ln.bias, rows: 1, cols: d }, cache),
    readExact(app, { tensor: mlp.in.tensor, rows: d, cols: m }, cache),
    readExact(app, { tensor: mlp.in.bias, rows: 1, cols: m }, cache),
    readExact(app, { tensor: mlp.out.tensor, rows: m, cols: d }, cache),
    readExact(app, { tensor: mlp.out.bias, rows: 1, cols: d }, cache),
  ]);
  // queries, keys and values of every token so far
  const normed = streamIn.map((row) => layerNorm(row, g1, b1, eps));
  const qkv = normed.map((row) => vecmat(row, wqkv, 3 * d, bqkv));
  const t = T - 1;
  const pattern = trace.row(unit.attention.trace, t);
  const C = pattern.length / H;
  const heads = [];
  const joined = new Float32Array(d);
  for (let h = 0; h < H; h++) {
    const q = qkv[t].slice(h * s, (h + 1) * s);
    const keys = new Float32Array(s * T); // [s, T]: column j is token j's key
    const values = new Float32Array(T * s); // [T, s]: row j is token j's value
    for (let j = 0; j < T; j++) {
      for (let k = 0; k < s; k++) {
        keys[k * T + j] = qkv[j][d + h * s + k];
        values[j * s + k] = qkv[j][2 * d + h * s + k];
      }
    }
    const dots = vecmat(q, keys, T);
    const scaled = dots.map((v) => v / Math.sqrt(s));
    const top = Math.max(...scaled);
    const exps = scaled.map((v) => Math.exp(v - top));
    const sum = exps.reduce((a, b) => a + b, 0);
    const shares = exps.map((v) => v / sum);
    const engineShares = Float32Array.from({ length: T }, (_, j) => pattern[h * C + j]);
    const mixed = vecmat(shares, values, s);
    joined.set(mixed, h * s);
    heads.push({ q, keys, values, dots, scaled, shares, engineShares, mixed, check: checkText(maxDifference(shares, engineShares), `These shares match the attention`) });
  }
  const attnOut = vecmat(joined, wo, d, bo);
  const engineAttn = trace.row(unit.writes.find((w) => w.id === "attn").trace, t);
  const middle = streamIn[t].map((v, i) => v + engineAttn[i]);
  const u = layerNorm(middle, g2, b2, eps);
  const pre = vecmat(u, win, m, bin);
  const post = pre.map(gelu);
  const enginePost = trace.row(unit.neurons.trace, t);
  const down = vecmat(enginePost, wout, d, bout);
  const engineMlp = trace.row(unit.writes.find((w) => w.id === "mlp").trace, t);
  const streamOut = trace.row(unit.stream.trace, t);
  const sum = streamIn[t].map((v, i) => v + engineAttn[i] + engineMlp[i]);
  return {
    unit, d, H, s, m, T, piece: tokens.pieces[t], tokens: tokens.pieces.slice(0, T),
    normed: normed[t], wqkv, bqkv, qkv: qkv[t], heads, joined, wo, bo, attnOut, engineAttn,
    u, win, bin, pre, post, enginePost, wout, bout, down, engineMlp,
    checks: {
      attn: checkText(maxDifference(attnOut, engineAttn), "What attention adds matches what"),
      post: checkText(maxDifference(post, enginePost), "These neuron values match what"),
      mlp: checkText(maxDifference(down, engineMlp), "What the MLP adds matches what"),
      stream: checkText(maxDifference(sum, streamOut), "Stream in + attention + MLP matches the stream out"),
    },
  };
}

const show = (piece) => String(piece ?? "").replace(/^ /, "·").replace(/\n/g, "↵");

/**
 * Builds what the bench draws for one step of one unit.
 * Returns { title, sentence, input, matrix, bias, output, after, check, columnName, ... }.
 */
export async function buildOp(app, unitIndex, stepId, { head = 0 } = {}, cache) {
  const { spec, trace, position, tokens } = app.state;
  const unit = spec.units[unitIndex];
  const piece = `“${show(tokens.pieces[position])}”`;

  if (unit.kind === "embed") {
    const view = unit.weights.find((w) => w.id === (stepId === "tok" ? "W_E" : "W_pos")).view;
    const rows = view.rows, d = view.cols;
    const index = stepId === "tok" ? tokens.ids[position] : position;
    const oneHot = new Float32Array(rows);
    oneHot[index] = 1;
    const engine = trace.row(unit.writes.find((w) => w.id === stepId).trace, position);
    const row = await readExact(app, { tensor: view.tensor, row0: index, rows: 1, cols: d }, cache);
    const isToken = stepId === "tok";
    return {
      title: isToken ? "Look up the token" : "Look up the position",
      sentence: isToken
        ? `The model keeps a table with one row of ${d} numbers for each of its ${rows.toLocaleString()} tokens. ${piece} is token number ${index}, so the input is a list of ${rows.toLocaleString()} zeros with a single 1 at place ${index} (a <em>one-hot</em> vector). Multiplying it by the table picks out row ${index}: every other row is multiplied by zero.`
        : `A second table has one row of ${d} numbers for each position in the text. ${piece} is at position ${index}, so a one-hot vector with its 1 at place ${index} picks out row ${index}. The two rows are then added: that sum is where the token's stream starts.`,
      formula: `${isToken ? "token" : "position"} vector[j] = Σ_i one_hot[i] · table[i, j] = table[${index}, j]`,
      input: { label: isToken ? "one-hot token" : "one-hot position", values: oneHot, name: (i) => (i === index ? `${isToken ? "token" : "position"} ${i}: ${piece}` : `${isToken ? "token" : "position"} ${i}`), focus: index, rowLabel: (i) => (i === index ? `${i} ${show(tokens.pieces[position])}` : `${i}`) },
      matrix: { label: isToken ? "token table W_E" : "position table W_pos", rows, cols: d, view, name: (i, j) => `row ${i}, column ${j}` },
      lookup: true,
      output: { label: isToken ? "token vector" : "position vector", values: row, engine },
      after: [],
      columnName: (j) => `number ${j} of the vector`,
      check: checkText(maxDifference(row, engine), "This row matches the vector"),
      column: (j) => readExact(app, { tensor: view.tensor, col0: j, rows, cols: 1 }, cache),
    };
  }

  if (unit.kind === "unembed") {
    const view = unit.weights[0].view; // the token table, [V, d]
    const V = view.rows, d = view.cols;
    const x = trace.row(unit.stream.trace, position);
    const { data: logits } = await app.engine.call("logits", { position });
    const top = Math.max(...logits);
    let z = 0;
    for (let j = 0; j < V; j++) z += Math.exp(logits[j] - top);
    const probs = logits.map((v) => Math.exp(v - top) / z);
    const prediction = trace.predictions[position];
    return {
      title: "Score every token",
      sentence: `The finished stream for ${piece}, normalised, is compared with every token's row of the token table, the same table the embedding used: one dot product per token, ${V.toLocaleString()} in all. A high score means a close match. Softmax turns the scores into probabilities; the highest, ${prediction ? `“${show(prediction.pieces[0])}”` : "the top one"}, is the prediction.`,
      formula: `score[j] = Σ_i stream[i] · token_table[j, i]     probability = softmax(scores)`,
      input: { label: "finished stream", values: x, name: (i) => `number ${i} of the stream` },
      matrix: { label: "token table, turned on its side", rows: d, cols: V, view: { ...view, rows: d, cols: V, transpose: true }, name: (i, j) => `token ${j}, number ${i}`, shared: true },
      output: { label: "scores (logits)", values: logits },
      after: [{ label: "softmax → probabilities", values: probs }],
      columnName: (j) => `token ${j}`,
      pieceOf: (j) => app.engine.call("pieces", { card: app.state.card, ids: [j] }).then((pieces) => pieces[0]),
      focus: prediction?.ids[0],
      check: null,
      column: (j) => readExact(app, { tensor: view.tensor, row0: j, rows: 1, cols: d }, cache),
      columnCheck: (j, sum) => checkText(Math.abs(sum - logits[j]), `This dot product matches the score`),
    };
  }

  const state = await blockState(app, unitIndex, cache);
  const { d, H, s, m, T } = state;
  const tokenName = (j) => `token ${j + 1}: “${show(state.tokens[j])}”`;
  const local = (data, rows, cols) => ({ data, rows, cols, column: (j) => { const out = new Float32Array(rows); for (let i = 0; i < rows; i++) out[i] = data[i * cols + j]; return Promise.resolve(out); } });
  const common = { state, piece };

  if (stepId === "qkv") {
    const part = (j) => ["query", "key", "value"][Math.floor(j / d)];
    const M = local(state.wqkv, d, 3 * d);
    return {
      ...common,
      title: "Query, key and value",
      sentence: `First the stream of ${piece} is normalised (scaled to a standard size). Then one matrix multiplication makes three lists at once: a <em>query</em> (what this token is looking for), a <em>key</em> (what it offers to others) and a <em>value</em> (what it passes on if chosen), ${d} numbers each, split into ${H} heads of ${s}. Every output number is the dot product of the ${d} stream numbers with one column of the matrix, plus a bias.`,
      formula: `[query | key | value][j] = Σ_i normalised_stream[i] · W_QKV[i, j] + b[j]`,
      input: { label: "normalised stream", values: state.normed, name: (i) => `number ${i} of the normalised stream` },
      matrix: { label: "W_QKV", ...M, name: (i, j) => `row ${i}, column ${j} (${part(j)})`, colLabel: (j) => `${part(j)[0]}${Math.floor((j % d) / s)}·${j % s}` },
      bias: state.bqkv,
      output: { label: "query | key | value", values: state.qkv },
      after: [],
      columnName: (j) => `${part(j)}, head ${Math.floor((j % d) / s)}, number ${j % s}`,
      check: { ok: true, text: "These are checked in the next steps: the scores they lead to match the engine's attention." },
      column: M.column,
    };
  }
  if (stepId === "scores") {
    const h = Math.min(head, H - 1);
    const hd = state.heads[h];
    const M = local(hd.keys, s, T);
    return {
      ...common,
      head: h,
      title: `Scores, head ${h}`,
      sentence: `Head ${h} compares the query of ${piece} (${s} numbers) with the key of every token so far: each score is a dot product, query · key, divided by √${s} = ${Math.sqrt(s)}. A high score means that token has what ${piece} is looking for. Softmax then turns the ${T} scores into shares of attention that add up to 100%.`,
      formula: `score[j] = Σ_i query[i] · key_j[i] / √${s}      shares = softmax(scores)`,
      input: { label: `query, head ${h}`, values: hd.q, name: (i) => `query number ${i}` },
      matrix: { label: "keys of the tokens so far", activation: true, ...M, name: (i, j) => `key of ${tokenName(j)}, number ${i}`, colLabel: (j) => show(state.tokens[j]) },
      output: { label: "query · key", values: hd.dots },
      after: [
        { label: `÷ √${s}`, values: hd.scaled },
        { label: "softmax → shares", values: hd.shares, engine: hd.engineShares },
      ],
      columnName: tokenName,
      check: hd.check,
      column: M.column,
    };
  }
  if (stepId === "mix") {
    const h = Math.min(head, H - 1);
    const hd = state.heads[h];
    const M = local(hd.values, T, s);
    return {
      ...common,
      head: h,
      title: `Mix the values, head ${h}`,
      sentence: `Head ${h} now takes what it chose: the value of each token so far (${s} numbers), weighted by that token's share of attention, and adds them up. So the input here is the ${T} shares, and each output number is a dot product of the shares with one column of the values.`,
      formula: `head_output[k] = Σ_j share[j] · value_j[k]`,
      input: { label: "shares of attention", values: hd.shares, name: (j) => `share of ${tokenName(j)}`, rowLabel: (j) => show(state.tokens[j]) },
      matrix: { label: "values of the tokens so far", activation: true, ...M, name: (j, k) => `value of ${tokenName(j)}, number ${k}` },
      output: { label: `head ${h} output`, values: hd.mixed },
      after: [],
      columnName: (k) => `number ${k} of head ${h}'s output`,
      check: hd.check,
      column: M.column,
    };
  }
  if (stepId === "wo") {
    const M = local(state.wo, d, d);
    return {
      ...common,
      title: "Join the heads",
      sentence: `The ${H} heads' outputs, ${s} numbers each, are laid end to end into one list of ${d}, and multiplied by the output matrix W_O, plus a bias. The result is what attention adds to the stream of ${piece}.`,
      formula: `attention_adds[j] = Σ_i heads_joined[i] · W_O[i, j] + b[j]`,
      input: { label: "all heads, joined", values: state.joined, name: (i) => `head ${Math.floor(i / s)}, number ${i % s}`, rowLabel: (i) => `h${Math.floor(i / s)}·${i % s}` },
      matrix: { label: "W_O", ...M, name: (i, j) => `row ${i}, column ${j}` },
      bias: state.bo,
      output: { label: "attention adds", values: state.attnOut, engine: state.engineAttn },
      after: [],
      columnName: (j) => `number ${j} of what attention adds`,
      check: state.checks.attn,
      column: M.column,
    };
  }
  if (stepId === "up") {
    const M = local(state.win, d, m);
    return {
      ...common,
      title: "MLP up, and GELU",
      sentence: `The stream, now with attention's addition, is normalised again and multiplied by W_in: ${m.toLocaleString()} dot products, one per neuron, each with its own column of ${d} weights, plus a bias. Then GELU, a curve that passes positive numbers and squashes negative ones to about zero, decides how strongly each neuron fires.`,
      formula: `neuron[j] = GELU( Σ_i normalised_stream[i] · W_in[i, j] + b[j] )`,
      input: { label: "normalised stream", values: state.u, name: (i) => `number ${i} of the normalised stream` },
      matrix: { label: "W_in", ...M, name: (i, j) => `row ${i}, neuron ${j}`, colLabel: (j) => `n${j}` },
      bias: state.bin,
      output: { label: "before GELU", values: state.pre },
      after: [{ label: "GELU → neurons", values: state.post, engine: state.enginePost, positiveOnly: true }],
      columnName: (j) => `neuron ${j}`,
      check: state.checks.post,
      column: M.column,
    };
  }
  if (stepId === "down") {
    const M = local(state.wout, m, d);
    return {
      ...common,
      title: "MLP down",
      sentence: `Each of the ${m.toLocaleString()} neurons has a row of ${d} output weights. Multiplying the neuron values by W_out adds those rows up, each scaled by how strongly its neuron fired: the result is what the MLP adds to the stream. The stream then leaves the block as stream in + attention + MLP.`,
      formula: `mlp_adds[j] = Σ_i neuron[i] · W_out[i, j] + b[j]`,
      input: { label: "neurons", values: state.enginePost, name: (i) => `neuron ${i}`, rowLabel: (i) => `n${i}` },
      matrix: { label: "W_out", ...M, name: (i, j) => `neuron ${i}, column ${j}` },
      bias: state.bout,
      output: { label: "MLP adds", values: state.down, engine: state.engineMlp },
      after: [],
      columnName: (j) => `number ${j} of what the MLP adds`,
      check: { ok: state.checks.mlp.ok && state.checks.stream.ok, text: `${state.checks.mlp.text} ${state.checks.stream.text}` },
      column: M.column,
    };
  }
  throw new Error(`No step called ${stepId}.`);
}
