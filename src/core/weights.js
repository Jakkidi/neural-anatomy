// WeightStore: every tensor of the loaded model, as Float32Array views on the
// arena, plus the read-only queries the interface needs (pooled regions for
// plates, exact regions for the reader, statistics for colour scales).

import { bytesPerElement, decodeInto, headerBytesNeeded, parseHeader } from "./safetensors.js";

export class WeightStore {
  constructor() {
    /** @type {Map<string, {name:string, shape:number[], data:Float32Array}>} */
    this.tensors = new Map();
    this.stats = new Map();
  }

  add(name, shape, data) {
    this.tensors.set(name, { name, shape, data });
    return data;
  }

  get(name) {
    const tensor = this.tensors.get(name);
    if (!tensor) throw new Error(`weight "${name}" is not loaded`);
    return tensor.data;
  }

  shapeOf(name) {
    return this.tensors.get(name)?.shape ?? null;
  }

  /** Rows and columns of a tensor seen as a matrix (a vector is one row). */
  matrixShape(name) {
    const shape = this.shapeOf(name);
    if (!shape) throw new Error(`weight "${name}" is not loaded`);
    if (shape.length === 1) return [1, shape[0]];
    if (shape.length === 2) return shape;
    return [shape[0], shape.slice(1).reduce((a, b) => a * b, 1)];
  }

  /**
   * A rectangle of a weight matrix, resampled to outRows x outCols.
   * When the output is smaller than the rectangle, each output cell holds the
   * entry of largest magnitude in its block (sign kept), so single large
   * weights stay visible at any zoom. When it is the same size the values are
   * exact.
   *
   * `view` places the rectangle in display space: {tensor, row0, col0, rows,
   * cols, transpose}. With transpose, display rows are tensor columns.
   */
  region(view, outRows, outCols) {
    const { tensor, transpose = false } = view;
    const data = this.get(tensor);
    const [R, C] = this.matrixShape(tensor);
    const dispRows = view.rows, dispCols = view.cols, row0 = view.row0 ?? 0, col0 = view.col0 ?? 0;
    const out = new Float32Array(outRows * outCols);
    const rowMap = new Int32Array(dispRows);
    const colMap = new Int32Array(dispCols);
    for (let r = 0; r < dispRows; r++) rowMap[r] = Math.min(outRows - 1, Math.floor((r * outRows) / dispRows));
    for (let c = 0; c < dispCols; c++) colMap[c] = Math.min(outCols - 1, Math.floor((c * outCols) / dispCols));
    if (!transpose) {
      if (row0 + dispRows > R || col0 + dispCols > C) throw new Error("region outside the tensor");
      for (let r = 0; r < dispRows; r++) {
        const src = (row0 + r) * C + col0;
        const dst = rowMap[r] * outCols;
        for (let c = 0; c < dispCols; c++) {
          const v = data[src + c];
          const at = dst + colMap[c];
          const have = out[at];
          if ((v < 0 ? -v : v) > (have < 0 ? -have : have)) out[at] = v;
        }
      }
    } else {
      // display (r, c) is tensor (col0 + c, row0 + r): walk the tensor in memory order
      if (col0 + dispCols > R || row0 + dispRows > C) throw new Error("region outside the tensor");
      for (let c = 0; c < dispCols; c++) {
        const src = (col0 + c) * C + row0;
        const oc = colMap[c];
        for (let r = 0; r < dispRows; r++) {
          const v = data[src + r];
          const at = rowMap[r] * outCols + oc;
          const have = out[at];
          if ((v < 0 ? -v : v) > (have < 0 ? -have : have)) out[at] = v;
        }
      }
    }
    return out;
  }

  /** min, max, mean and standard deviation of a view (cached). */
  statsOf(view) {
    const key = `${view.tensor}|${view.row0 ?? 0}|${view.col0 ?? 0}|${view.rows}|${view.cols}|${view.transpose ? 1 : 0}`;
    let stats = this.stats.get(key);
    if (stats) return stats;
    const data = this.get(view.tensor);
    const [, C] = this.matrixShape(view.tensor);
    const t = view.transpose;
    const r0 = t ? view.col0 ?? 0 : view.row0 ?? 0;
    const c0 = t ? view.row0 ?? 0 : view.col0 ?? 0;
    const rows = t ? view.cols : view.rows;
    const cols = t ? view.rows : view.cols;
    let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
    for (let r = 0; r < rows; r++) {
      const src = (r0 + r) * C + c0;
      for (let c = 0; c < cols; c++) {
        const v = data[src + c];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumSq += v * v;
      }
    }
    const count = rows * cols;
    const mean = sum / count;
    stats = { min, max, mean, std: Math.sqrt(Math.max(0, sumSq / count - mean * mean)), count };
    this.stats.set(key, stats);
    return stats;
  }
}

/**
 * Reads a safetensors byte stream into the store.
 *
 * `plan(header)` is called once the header is known. It returns
 *   { wanted: [{file, name, shape}], arena }
 * naming which file tensors to keep, what to call them, and the arena to
 * allocate them from. Tensors it does not name are skipped without being
 * buffered, so a 548 MB file needs no more memory than its kept weights.
 */
export async function readSafetensors(stream, store, plan, onProgress = () => {}) {
  const reader = stream.getReader();
  let head = new Uint8Array(0);
  let header = null;
  let position = 0; // absolute offset of the next unread byte
  let pending = null; // the first chunk that reaches past the header

  while (!header) {
    const { done, value } = await reader.read();
    if (done) throw new Error("This is not a complete safetensors file.");
    const joined = new Uint8Array(head.length + value.length);
    joined.set(head);
    joined.set(value, head.length);
    head = joined;
    if (head.length >= 8 && headerBytesNeeded(head) > 100 * 1024 * 1024) throw new Error("This does not look like a safetensors file.");
    if (head.length >= 8 && head.length >= headerBytesNeeded(head)) {
      header = parseHeader(head);
      pending = head.subarray(header.dataStart);
      position = header.dataStart;
    }
  }

  const { wanted, arena } = await plan(header);
  const byFile = new Map(header.tensors.map((tensor) => [tensor.name, tensor]));
  const jobs = wanted
    .map((want) => {
      const entry = byFile.get(want.file);
      const count = entry.shape.reduce((a, b) => a * b, 1);
      const data = store.add(want.name, want.shape ?? entry.shape, arena.alloc(count));
      const direct = entry.dtype === "F32";
      return {
        entry,
        data,
        direct,
        bytes: direct ? new Uint8Array(data.buffer, data.byteOffset, count * 4) : new Uint8Array(count * bytesPerElement(entry.dtype)),
      };
    })
    .sort((a, b) => a.entry.begin - b.entry.begin);

  const total = header.tensors.reduce((end, tensor) => Math.max(end, tensor.end), header.dataStart);
  let job = 0;
  const take = (chunk) => {
    const from = position, to = position + chunk.length;
    while (job < jobs.length && jobs[job].entry.begin < to) {
      const { entry, bytes, direct, data } = jobs[job];
      const a = Math.max(from, entry.begin), b = Math.min(to, entry.end);
      if (b > a) bytes.set(chunk.subarray(a - from, b - from), a - entry.begin);
      if (entry.end > to) break;
      if (!direct) {
        decodeInto(entry.dtype, bytes, data);
        jobs[job].bytes = null;
      }
      job++;
    }
    position = to;
  };

  let lastReport = 0;
  const report = (force) => {
    const now = Date.now();
    if (!force && now - lastReport < 80) return;
    lastReport = now;
    onProgress(Math.min(position, total), total);
  };

  if (pending.length) take(pending);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    take(value);
    report(false);
  }
  report(true);
  if (job < jobs.length) throw new Error(`The file ended early: "${jobs[job].entry.name}" is incomplete.`);
  return header;
}

/** Deterministic normal numbers, for the random-weights test pattern. */
export function gaussianFiller(seed = 1) {
  let a = seed >>> 0 || 1;
  const uniform = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (out, std = 1, mean = 0) => {
    const n = out.length;
    for (let i = 0; i < n; i += 2) {
      const u = Math.max(uniform(), 1e-12), v = uniform();
      const radius = Math.sqrt(-2 * Math.log(u)) * std, angle = 2 * Math.PI * v;
      out[i] = mean + radius * Math.cos(angle);
      if (i + 1 < n) out[i + 1] = mean + radius * Math.sin(angle);
    }
  };
}
