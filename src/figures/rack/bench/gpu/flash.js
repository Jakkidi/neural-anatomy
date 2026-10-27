// Attention as a GPU runs it, on one head's real Q, K and V: the order of the
// work, what each thread block holds in on-chip memory at each step, and how
// many numbers cross between on-chip SRAM and off-chip memory.
//
// Two ways:
//   "standard"  three kernels. S = Q·Kᵀ goes out to off-chip memory, comes back
//               for the softmax, P goes out, comes back for P·V.
//   "flash"     one kernel (FlashAttention). Each thread block keeps one tile
//               of queries on chip, walks through the key/value tiles with a
//               running max and sum (online softmax), and writes only O.
//
// A "tick" is one step of every thread block at once: they run in parallel,
// one per SM. Ticks are for following the order, not a measure of time.

export const KERNELS = {
  flash: [
    "__global__ void flash_attention(Q, K, V, O) {",
    "  int i = blockIdx.x;       // which tile of queries",
    "  __shared__ float q[B][d], k[B][d], v[B][d]; // SRAM",
    "  q = Q[i]; m = −∞; l = 0; acc = 0;  // HBM → SRAM",
    "  for (int j = 0; j <= i; j++) {     // skip future",
    "    k = K[j];  v = V[j];             // HBM → SRAM",
    "    S = q · kᵀ / √d;                 // tensor cores",
    "    if (j == i) S = mask(S);         // diagonal tile",
    "    m_new = max(m, rowmax(S));",
    "    P = exp(S − m_new);",
    "    l = l · exp(m − m_new) + rowsum(P);",
    "    acc = acc · exp(m − m_new) + P · v;",
    "    m = m_new;",
    "  }",
    "  O[i] = acc / l;                    // SRAM → HBM, once",
    "}",
  ],
  standard: [
    "// three kernels, one after another;",
    "// each reads its input from HBM and writes back",
    "matmul<<<grid>>>(Q, Kᵀ, S);   // S = Q·Kᵀ/√d → HBM",
    "softmax<<<rows>>>(S, P);      // S → mask, softmax → P",
    "matmul<<<grid>>>(P, V, O);    // P, V → O → HBM",
  ],
};

/** The tile size to draw with: a few tiles along the text, so each can be seen. */
export function displayTile(T) {
  let B = 2;
  while (Math.ceil(T / B) > 5 && B < 128) B *= 2;
  return B;
}

const span = (k, B, T) => ({ start: k * B, end: Math.min(T, (k + 1) * B) });

/**
 * The events of every thread block, tick by tick, and the traffic they cause.
 * Traffic is counted in numbers moved (multiply by 2 for float16 bytes).
 */
export function plan({ T, d, B, causal, method }) {
  const Tr = Math.ceil(T / B);
  const blocks = [];
  for (let i = 0; i < Tr; i++) {
    const rows = span(i, B, T);
    const R = rows.end - rows.start;
    const events = [];
    if (method === "flash") {
      events.push({ phase: "init", j: -1, lines: [1, 2, 3], read: R * d, write: 0 });
      const last = causal ? i : Tr - 1;
      for (let j = 0; j <= last; j++) {
        const C = span(j, B, T).end - span(j, B, T).start;
        events.push({ phase: "load", j, lines: [4, 5], read: 2 * C * d, write: 0 });
        events.push({ phase: "s", j, lines: [6], read: 0, write: 0 });
        if (causal && j === i) events.push({ phase: "mask", j, lines: [7], read: 0, write: 0 });
        events.push({ phase: "max", j, lines: [8], read: 0, write: 0 });
        events.push({ phase: "p", j, lines: [9, 10], read: 0, write: 0 });
        events.push({ phase: "o", j, lines: [11, 12], read: 0, write: 0 });
      }
      events.push({ phase: "write", j: last, lines: [14], read: 0, write: R * d });
    } else {
      for (let j = 0; j < Tr; j++) {
        const C = span(j, B, T).end - span(j, B, T).start;
        events.push({ phase: "k1", j, lines: [2], read: R * d + C * d, write: R * C });
      }
      events.push({ phase: "k2", j: -1, lines: [3], read: R * T, write: R * T });
      for (let j = 0; j < Tr; j++) {
        const C = span(j, B, T).end - span(j, B, T).start;
        events.push({ phase: "k3", j, lines: [4], read: R * C + C * d, write: 0 });
      }
      events.push({ phase: "k3w", j: Tr - 1, lines: [4], read: 0, write: R * d });
    }
    blocks.push({ i, rows, events });
  }
  const ticks = Math.max(...blocks.map((b) => b.events.length)) + 1;
  // cumulative traffic after each tick
  const read = new Float64Array(ticks), write = new Float64Array(ticks);
  for (let t = 0; t < ticks; t++) {
    let r = t ? read[t - 1] : 0, w = t ? write[t - 1] : 0;
    for (const b of blocks) { const e = b.events[t]; if (e) { r += e.read; w += e.write; } }
    read[t] = r; write[t] = w;
  }
  return { T, d, B, Tr, causal, method, blocks, ticks, read, write };
}

/** What block i is doing at a tick ("done" once it has finished). */
export function eventAt(p, i, tick) {
  const b = p.blocks[i];
  return b.events[tick] ?? { phase: "done", j: -1, lines: [], read: 0, write: 0, finished: b.events.length - 1 };
}

const rowsOf = (M, start, end, n) => M.subarray(start * n, end * n);

/**
 * What thread block i holds in SRAM after `tick`: the buffers to draw, the
 * numbers behind them, and which ones this step changed. Recomputed from the
 * start each time; the tiles are small.
 */
export function simulate(p, head, i, tick) {
  const { T, d, B, causal, method } = p;
  const { Q, K, V, scale } = head;
  const block = p.blocks[i];
  const { start: r0, end: r1 } = block.rows;
  const R = r1 - r0;
  const last = Math.min(tick, block.events.length - 1);
  const now = eventAt(p, i, tick);
  const buffers = { a: [], b: [] };
  const fresh = new Set();
  const notes = {};

  if (method === "flash") {
    const q = rowsOf(Q, r0, r1, d);
    let m = new Float64Array(R).fill(-Infinity), l = new Float64Array(R), acc = new Float64Array(R * d);
    let k = null, v = null, S = null, P = null, C = 0, c0 = 0, mOld = null, lOld = null, rowMax = null, rowSum = null, factor = null, O = null;
    for (let e = 0; e <= last; e++) {
      const ev = block.events[e];
      if (ev.phase === "load") {
        ({ start: c0 } = span(ev.j, B, T));
        C = span(ev.j, B, T).end - c0;
        k = rowsOf(K, c0, c0 + C, d); v = rowsOf(V, c0, c0 + C, d);
        S = null; P = null;
      } else if (ev.phase === "s") {
        S = new Float64Array(R * C);
        for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
          let dot = 0;
          for (let x = 0; x < d; x++) dot += q[r * d + x] * k[c * d + x];
          S[r * C + c] = dot * scale;
        }
      } else if (ev.phase === "mask") {
        for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) if (c0 + c > r0 + r) S[r * C + c] = -Infinity;
      } else if (ev.phase === "max") {
        mOld = Float64Array.from(m);
        rowMax = new Float64Array(R).fill(-Infinity);
        for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) rowMax[r] = Math.max(rowMax[r], S[r * C + c]);
        m = mOld.map((x, r) => Math.max(x, rowMax[r]));
      } else if (ev.phase === "p") {
        P = new Float64Array(R * C);
        rowSum = new Float64Array(R);
        factor = new Float64Array(R);
        lOld = Float64Array.from(l);
        for (let r = 0; r < R; r++) {
          factor[r] = mOld[r] === -Infinity ? 0 : Math.exp(mOld[r] - m[r]);
          for (let c = 0; c < C; c++) { P[r * C + c] = Math.exp(S[r * C + c] - m[r]); rowSum[r] += P[r * C + c]; }
          l[r] = l[r] * factor[r] + rowSum[r];
        }
      } else if (ev.phase === "o") {
        for (let r = 0; r < R; r++) for (let x = 0; x < d; x++) {
          let sum = 0;
          for (let c = 0; c < C; c++) sum += P[r * C + c] * v[c * d + x];
          acc[r * d + x] = acc[r * d + x] * factor[r] + sum;
        }
      } else if (ev.phase === "write") {
        O = new Float64Array(R * d);
        for (let r = 0; r < R; r++) for (let x = 0; x < d; x++) O[r * d + x] = acc[r * d + x] / l[r];
      }
    }
    const phase = now.phase === "done" ? "write" : now.phase;
    const started = last >= 0;
    const jNow = now.phase === "done" ? block.events[block.events.length - 1].j : now.j;
    buffers.a.push({ id: "q", label: `q · Q tile ${i}`, kind: "heat", rows: R, cols: d, data: started ? q : null, matrix: "Q", rowStart: r0 });
    buffers.a.push({ id: "k", label: k ? `k · K tile ${jNow}` : "k", kind: "heat", rows: k ? C : R, cols: d, data: k, matrix: "K", rowStart: c0 });
    buffers.a.push({ id: "v", label: v ? `v · V tile ${jNow}` : "v", kind: "heat", rows: v ? C : R, cols: d, data: v, matrix: "V", rowStart: c0 });
    buffers.b.push({ id: "S", label: "S = q·kᵀ/√d", kind: "grid", rows: R, cols: C || R, data: S, rowStart: r0, colStart: c0 });
    buffers.b.push({ id: "P", label: "P = exp(S − m)", kind: "grid", rows: R, cols: C || R, data: P, rowStart: r0, colStart: c0 });
    buffers.b.push({ id: "m", label: "m", kind: "vec", rows: R, data: started ? m : null, rowStart: r0, old: mOld });
    buffers.b.push({ id: "l", label: "l", kind: "vec", rows: R, data: started ? l : null, rowStart: r0, old: lOld });
    buffers.a.push(O
      ? { id: "O", label: "O tile = acc / l", kind: "heat", rows: R, cols: d, data: O, rowStart: r0 }
      : { id: "acc", label: "acc", kind: "heat", rows: R, cols: d, data: started ? acc : null, rowStart: r0 });
    const changed = { init: ["q", "m", "l", "acc"], load: ["k", "v"], s: ["S"], mask: ["S"], max: ["m"], p: ["P", "l"], o: ["acc"], write: ["O"] }[phase] ?? [];
    if (now.phase !== "done") changed.forEach((id) => fresh.add(id));
    Object.assign(notes, { mOld, m, lOld, l, rowMax, rowSum, factor, acc, O, R, C, c0, r0, phase: now.phase, j: jNow });
  } else {
    // standard: the selected block's view of each kernel
    const ev = now.phase === "done" ? block.events[block.events.length - 1] : now;
    const q = rowsOf(Q, r0, r1, d);
    if (ev.phase === "k1") {
      const { start: c0, end: c1 } = span(ev.j, B, T);
      const C = c1 - c0;
      const k = rowsOf(K, c0, c1, d);
      const S = new Float64Array(R * C);
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) { let dot = 0; for (let x = 0; x < d; x++) dot += q[r * d + x] * k[c * d + x]; S[r * C + c] = dot * scale; }
      buffers.a.push({ id: "q", label: `Q tile ${i}`, kind: "heat", rows: R, cols: d, data: q, matrix: "Q", rowStart: r0 });
      buffers.a.push({ id: "k", label: `K tile ${ev.j}`, kind: "heat", rows: C, cols: d, data: k, matrix: "K", rowStart: c0 });
      buffers.b.push({ id: "S", label: `S tile (${i}, ${ev.j}) → HBM`, kind: "grid", rows: R, cols: C, data: S, rowStart: r0, colStart: c0 });
      ["q", "k", "S"].forEach((id) => fresh.add(id));
    } else if (ev.phase === "k2") {
      const S = new Float64Array(R * T), P = new Float64Array(R * T);
      for (let r = 0; r < R; r++) for (let c = 0; c < T; c++) {
        let dot = 0;
        for (let x = 0; x < d; x++) dot += q[r * d + x] * K[c * d + x];
        S[r * T + c] = dot * scale;
        P[r * T + c] = head.shares[(r0 + r) * T + c];
      }
      buffers.a.push({ id: "S", label: `S rows ${r0}–${r1 - 1}, from HBM`, kind: "grid", rows: R, cols: T, data: S, rowStart: r0, colStart: 0, maskFrom: causal });
      buffers.b.push({ id: "P", label: "P = softmax(mask(S)) → HBM", kind: "grid", rows: R, cols: T, data: P, rowStart: r0, colStart: 0 });
      ["S", "P"].forEach((id) => fresh.add(id));
    } else {
      const j = ev.j;
      const { start: c0, end: c1 } = span(j, B, T);
      const C = c1 - c0;
      const P = new Float64Array(R * C);
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) P[r * C + c] = head.shares[(r0 + r) * T + c0 + c];
      const acc = new Float64Array(R * d);
      for (let r = 0; r < R; r++) for (let c = 0; c < c1; c++) {
        const w = head.shares[(r0 + r) * T + c];
        if (w) for (let x = 0; x < d; x++) acc[r * d + x] += w * V[c * d + x];
      }
      buffers.a.push({ id: "P", label: `P tile (${i}, ${j}), from HBM`, kind: "grid", rows: R, cols: C, data: P, rowStart: r0, colStart: c0 });
      buffers.a.push({ id: "v", label: `V tile ${j}`, kind: "heat", rows: C, cols: d, data: rowsOf(V, c0, c1, d), matrix: "V", rowStart: c0 });
      buffers.b.push({ id: ev.phase === "k3w" ? "O" : "acc", label: ev.phase === "k3w" ? "O tile → HBM" : "acc = Σ P · V", kind: "heat", rows: R, cols: d, data: acc, rowStart: r0 });
      (ev.phase === "k3w" ? ["O"] : ["P", "v", "acc"]).forEach((id) => fresh.add(id));
    }
    Object.assign(notes, { phase: now.phase, j: ev.j, r0, R });
  }
  const bytes = [...buffers.a, ...buffers.b].reduce((sum, b) => sum + (b.kind === "vec" ? b.rows : b.rows * (b.cols ?? 1)) * 4, 0);
  return { buffers, fresh, notes, bytes, event: now };
}

/** FlashAttention's output for every row, run to the end, to compare with ordinary attention. */
export function flashOutput(head, B) {
  const p = plan({ T: head.T, d: head.s, B, causal: head.causal, method: "flash" });
  const O = new Float32Array(head.T * head.s);
  for (let i = 0; i < p.Tr; i++) {
    const sim = simulate(p, { ...head, Q: head.Q, K: head.K, V: head.V }, i, p.blocks[i].events.length - 1);
    O.set(Float32Array.from(sim.notes.O), p.blocks[i].rows.start * head.s);
  }
  let worst = 0;
  for (let x = 0; x < O.length; x++) worst = Math.max(worst, Math.abs(O[x] - head.out[x]));
  return { O, worst };
}

/**
 * Numbers moved to and from off-chip memory for one head, both ways, at any
 * length. Counted the way `plan` counts them, without listing the events.
 */
export function trafficAt(T, d, B, causal) {
  const Tr = Math.ceil(T / B);
  let flash = 0, standard = 0;
  for (let i = 0; i < Tr; i++) {
    const R = Math.min(T, (i + 1) * B) - i * B;
    const seen = causal ? Math.min(T, (i + 1) * B) : T; // keys this query tile reads
    flash += R * d + 2 * seen * d + R * d;
    standard += Tr * R * d + T * d + R * T; // kernel 1: Q tile per key tile, every K tile, S
    standard += 2 * R * T; // kernel 2: read S, write P
    standard += R * T + T * d + R * d; // kernel 3: read P and every V tile, write O
  }
  return { flash, standard, matrix: T * T, Tr };
}
