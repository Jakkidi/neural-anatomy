// Plates: small bitmaps that carry real numbers into the drawing. One pixel is
// one value (a neuron, a stream dimension, an attention weight, a weight or a
// block of weights). They are placed on the faces of the machine as SVG
// <image> elements, so they zoom as crisp cells and sit in painter's order
// with everything else.
//
// Ink: positive values are drawn in the figure's lit tone (white on the dark
// theme, black on the light one); negative values in red, as in a ledger.
// How strong the ink is says how large the value is against `scale`.

const GAMMA = 0.62;

/** Reads the current theme's inks from CSS. */
export function inksOf(element = document.documentElement) {
  const style = getComputedStyle(element);
  const parse = (name, fallback) => {
    const text = style.getPropertyValue(name).trim() || fallback;
    const hex = text.replace("#", "");
    const full = hex.length === 3 ? [...hex].map((c) => c + c).join("") : hex;
    return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)];
  };
  return { ink: parse("--anatomy-lit", "#000000"), red: parse("--anatomy-red", "#e5484d") };
}

/** A square-ish grid for n cells: [cols, rows] with cols * rows >= n. */
export function gridFor(n, aspect = 4 / 3) {
  let best = null;
  for (let rows = 1; rows * rows <= n * 4; rows++) {
    if (n % rows) continue;
    const cols = n / rows;
    const error = Math.abs(Math.log(cols / rows / aspect));
    if (!best || error < best.error) best = { cols, rows, error };
  }
  if (best && best.error < 0.9) return [best.cols, best.rows];
  const rows = Math.max(1, Math.round(Math.sqrt(n / aspect)));
  return [Math.ceil(n / rows), rows];
}

export class Plate {
  /**
   * @param {SVGImageElement} image the element this plate fills
   * @param {number} cols
   * @param {number} rows
   */
  constructor(image, cols, rows) {
    this.image = image;
    this.cols = cols;
    this.rows = rows;
    this.canvas = document.createElement("canvas");
    this.canvas.width = cols;
    this.canvas.height = rows;
    this.context = this.canvas.getContext("2d");
    this.pixels = this.context.createImageData(cols, rows);
    this.blank = true;
    this.values = null; // what is on the plate now, for hover read-outs
  }

  #commit() {
    this.context.putImageData(this.pixels, 0, 0);
    this.image.setAttribute("href", this.canvas.toDataURL("image/png"));
    this.blank = false;
  }

  clear() {
    this.values = null;
    if (this.blank) return;
    this.image.removeAttribute("href");
    this.blank = true;
  }

  /**
   * Cell i (row-major) takes values[offset + i * stride]. Cells past `count` stay empty.
   * @param {{ink:number[], red:number[]}} inks
   */
  paint(values, { scale, inks, count = this.cols * this.rows, offset = 0, stride = 1, gamma = GAMMA, keep = true, positiveOnly = false }) {
    const out = this.pixels.data;
    const { ink, red } = inks;
    const cells = this.cols * this.rows;
    const inv = scale > 0 ? 1 / scale : 0;
    for (let i = 0; i < cells; i++) {
      const o = i * 4;
      if (i >= count) { out[o + 3] = 0; continue; }
      const v = values[offset + i * stride];
      if (positiveOnly && v <= 0) { out[o + 3] = 0; continue; }
      const magnitude = Math.min(1, (v < 0 ? -v : v) * inv);
      const color = v < 0 ? red : ink;
      out[o] = color[0];
      out[o + 1] = color[1];
      out[o + 2] = color[2];
      out[o + 3] = magnitude > 0 ? Math.round(255 * Math.pow(magnitude, gamma)) : 0;
    }
    if (keep) this.values = { values, offset, stride, count };
    this.#commit();
  }

  /** The value shown in a cell, or undefined. */
  valueAt(col, row) {
    if (!this.values) return undefined;
    const i = row * this.cols + col;
    if (i < 0 || i >= this.values.count) return undefined;
    return this.values.values[this.values.offset + i * this.values.stride];
  }
}

/**
 * A robust "full ink" level for an array: the q-quantile of |value| over a
 * sample, so a handful of extreme values do not wash the rest out.
 */
export function inkScale(values, { q = 0.995, positiveOnly = false, limit = 60000, length = values.length } = {}) {
  const step = Math.max(1, Math.floor(length / limit));
  const sample = [];
  for (let i = 0; i < length; i += step) {
    const v = values[i];
    if (positiveOnly) { if (v > 0) sample.push(v); }
    else if (v !== 0) sample.push(v < 0 ? -v : v);
  }
  if (!sample.length) return 1;
  const sorted = Float32Array.from(sample).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))] || sorted[sorted.length - 1] || 1;
}

/** A probability as a percentage that never rounds a real chance down to 0.0%. */
export function percent(p) {
  if (!Number.isFinite(p)) return "\u2014";
  const v = p * 100;
  if (v >= 10) return `${v.toFixed(1)}%`;
  if (v >= 0.01) return `${v.toFixed(2)}%`;
  return v > 0 ? "<0.01%" : "0%";
}

/** Short fixed-width number for read-outs: +3.214, -0.007, +1.2e-5 */
export function signed(value, digits = 3) {
  if (value === undefined || Number.isNaN(value)) return "—";
  const a = Math.abs(value);
  const text = a !== 0 && a < 10 ** -digits ? a.toExponential(1) : a.toFixed(digits);
  return (value < 0 ? "−" : "+") + text;
}

/**
 * Full ink for the rows a figure shows, set plate by plate and position by
 * position: the 99th percentile of what the row holds. A shared scale would
 * not do: GPT-2's first position, for one, is hundreds of times larger than
 * the rest and would blank every other token.
 */
export class RowScales {
  constructor() {
    this.cache = new Map();
  }

  clear() {
    this.cache.clear();
  }

  /** @param {"neurons"|"stream"|"vector"} kind */
  of(trace, key, position, row, kind) {
    const entry = trace.entry(key);
    const id = `${key}@${position}`;
    const cached = this.cache.get(id);
    if (cached && cached.version === entry.version) return cached.scale;
    const scale = inkScale(row, { q: 0.99, positiveOnly: kind === "neurons" });
    if (this.cache.size > 4000) this.cache.clear();
    this.cache.set(id, { version: entry.version, scale });
    return scale;
  }
}
