// The bench's canvas: one vector × matrix product, drawn so it can be read.
//
//              ┌──────────── matrix columns ────────────┐
//   outputs →  │ y  (and the steps after it: GELU …)    │   rows across the top
//   input  x   │                                        │
//   column  ↓  │              the matrix W              │
//              │                                        │
//
// The input runs down the left, beside the matrix rows it multiplies; the
// outputs run across the top, above the matrix columns that make them. Choose
// an output and its column is lifted out next to the input, the products
// appear one row at a time, and their running sum climbs to the output.
//
// Zoom: pinch, Ctrl/Cmd + wheel, double-click, or the buttons. Every number is
// written in its cell from about 34 px per cell. Matrices that are too big to
// hold (the 50,257-row token table) are read from the engine a window at a time.

import { inksOf, signed } from "../../../ui/plates.js";

const MAX_CELL = 80;
const NUMBERS_FROM = 34;
const LABEL_W = 132;
const GAMMA = 0.62;
const SWEEP_MS = 1400;

const fmt = (v) => {
  if (!Number.isFinite(v)) return "";
  const a = Math.abs(v);
  const text = a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a !== 0 && a < 0.001 ? v.toExponential(0) : v.toFixed(3);
  return text.replace("-", "−");
};

/** The 99th percentile of |v|, so a few large values do not wash the rest out. */
function fullInk(values, positiveOnly = false) {
  const step = Math.max(1, Math.floor(values.length / 20000));
  const sample = [];
  for (let i = 0; i < values.length; i += step) {
    const v = values[i];
    if (positiveOnly ? v > 0 : v !== 0) sample.push(Math.abs(v));
  }
  if (!sample.length) return 1;
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.99))] || 1;
}

export class MatmulView {
  constructor(canvas, app, { onHover = () => {}, onSelect = () => {}, onSweep = () => {} } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.app = app;
    this.onHover = onHover;
    this.onSelect = onSelect;
    this.onSweep = onSweep;
    this.op = null;
    this.size = { w: 0, h: 0, ratio: 1 };
    this.cx = 1; // px per matrix column
    this.cy = 1; // px per matrix row
    this.origin = { r: 0, c: 0 };
    this.selected = -1; // chosen output column
    this.pinned = -1; // chosen input row
    this.column = null; // the chosen column of the matrix, exact
    this.sweep = 1; // 0..1, how far the products have been revealed
    this.hover = null;
    this.region = null; // the last window of a big matrix read from the engine
    this.stamp = 0;
    this.tile = document.createElement("canvas");
    this.#listen();
    new ResizeObserver(() => { if (this.#resize()) { this.#clamp(); this.draw(); this.#request(); } }).observe(canvas);
  }

  // ---- layout -------------------------------------------------------------------------------
  get rows() { return this.op?.matrix.rows ?? 1; }
  get cols() { return this.op?.matrix.cols ?? 1; }
  get colW() { return this.cy >= 13 ? 62 : 18; }
  get rowH() { return this.cx >= NUMBERS_FROM ? 24 : 18; }
  /** Left of the matrix: labels, then the input column, then (when an output is chosen) its weights and the products. */
  get left() { return LABEL_W + (this.selected >= 0 ? 3 : 1) * (this.colW + 8) + 6; }
  get top() { return 22 + (1 + (this.op?.after.length ?? 0)) * (this.rowH + 5) + 20; }
  get spanW() { return this.size.w - this.left - 8; }
  get spanH() { return this.size.h - this.top - 8; }

  #resize() {
    const box = this.canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(200, Math.round(box.width)), h = Math.max(200, Math.round(box.height));
    if (w === this.size.w && h === this.size.h && ratio === this.size.ratio) return false;
    this.size = { w, h, ratio };
    this.canvas.width = Math.round(w * ratio);
    this.canvas.height = Math.round(h * ratio);
    return true;
  }
  // fitted, each axis fills the bench; zooming scales both together
  get #fitX() { return Math.min(MAX_CELL, this.spanW / this.cols); }
  get #fitY() { return Math.min(MAX_CELL, this.spanH / this.rows); }
  #clamp() {
    this.cx = Math.min(MAX_CELL, Math.max(this.#fitX, this.cx));
    this.cy = Math.min(MAX_CELL, Math.max(this.#fitY, this.cy));
    const spanR = this.spanH / this.cy, spanC = this.spanW / this.cx;
    this.origin.r = spanR >= this.rows ? 0 : Math.min(this.rows - spanR, Math.max(0, this.origin.r));
    this.origin.c = spanC >= this.cols ? 0 : Math.min(this.cols - spanC, Math.max(0, this.origin.c));
  }
  #visible() {
    const r0 = Math.max(0, Math.floor(this.origin.r)), c0 = Math.max(0, Math.floor(this.origin.c));
    const r1 = Math.min(this.rows, Math.ceil(this.origin.r + this.spanH / this.cy));
    const c1 = Math.min(this.cols, Math.ceil(this.origin.c + this.spanW / this.cx));
    return { r: r0, c: c0, rows: Math.max(1, r1 - r0), cols: Math.max(1, c1 - c0) };
  }
  // matrix coordinates <-> screen
  #x(c) { return this.left + (c - this.origin.c) * this.cx; }
  #y(r) { return this.top + (r - this.origin.r) * this.cy; }

  // ---- public ----------------------------------------------------------------------------------
  setOp(op, { keepView = false } = {}) {
    const same = keepView && this.op && this.op.matrix.rows === op.matrix.rows && this.op.matrix.cols === op.matrix.cols;
    this.op = op;
    this.region = null;
    this.column = null;
    this.selected = -1;
    this.pinned = op.input.focus ?? -1;
    this.inkMatrix = op.matrix.data ? fullInk(op.matrix.data) * 1.4 : 0;
    this.inkInput = fullInk(op.input.values);
    this.inkOutput = [fullInk(op.output.values), ...op.after.map((a) => fullInk(a.values, a.positiveOnly))];
    this.#resize();
    if (!same) this.fit();
    else { this.#clamp(); this.draw(); this.#request(true); }
    if (op.matrix.view && !op.matrix.data) {
      this.app.engine.call("stats", { view: op.matrix.view }).then((stats) => { if (this.op === op) { this.inkMatrix = stats.std * 3; this.draw(); } }).catch(() => {});
    }
  }
  fit() {
    this.cx = this.#fitX;
    this.cy = this.#fitY;
    this.origin = { r: 0, c: 0 };
    this.#clamp();
    this.draw();
    this.#request(true);
  }
  zoomAt(factor, x = this.left + this.spanW / 2, y = this.top + this.spanH / 2) {
    const r = this.origin.r + (y - this.top) / this.cy, c = this.origin.c + (x - this.left) / this.cx;
    this.cx = Math.min(MAX_CELL, Math.max(this.#fitX, this.cx * factor));
    this.cy = Math.min(MAX_CELL, Math.max(this.#fitY, this.cy * factor));
    this.origin.r = r - (y - this.top) / this.cy;
    this.origin.c = c - (x - this.left) / this.cx;
    this.#clamp();
    this.draw();
    this.#request();
  }
  /** Zooms until the numbers are written, centred on a row and column. */
  zoomTo(row, col) {
    this.cx = Math.max(this.#fitX, 52);
    this.cy = Math.max(this.#fitY, 22);
    this.origin.r = row - this.spanH / this.cy / 2;
    this.origin.c = col - this.spanW / this.cx / 2;
    this.#clamp();
    this.draw();
    this.#request(true);
  }
  async select(j, { animate = true } = {}) {
    if (!this.op || j < 0 || j >= this.cols) return;
    this.selected = j;
    this.column = null;
    this.sweep = 0;
    const op = this.op;
    this.onSelect(j);
    this.draw();
    const column = await op.column(j);
    if (this.op !== op || this.selected !== j) return;
    this.column = column;
    this.products = Float32Array.from(column, (w, i) => w * op.input.values[i]);
    this.inkColumn = fullInk(column);
    this.inkProducts = fullInk(this.products);
    if (!animate || matchMedia("(prefers-reduced-motion: reduce)").matches) { this.sweep = 1; this.onSweep(1, this.#partial(this.rows)); this.draw(); return; }
    const start = performance.now();
    const step = (now) => {
      if (this.op !== op || this.selected !== j) return;
      this.sweep = Math.min(1, (now - start) / SWEEP_MS);
      this.onSweep(this.sweep, this.#partial(Math.round(this.sweep * this.rows)));
      this.draw();
      if (this.sweep < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  clearSelection() {
    this.selected = -1;
    this.column = null;
    this.#clamp();
    this.draw();
  }
  #partial(count) {
    let sum = 0;
    for (let i = 0; i < count && this.products && i < this.products.length; i++) sum += this.products[i];
    return { sum, count };
  }

  // ---- the big matrix, a window at a time ----------------------------------------------------------
  #request(now = false) {
    const op = this.op;
    if (!op?.matrix.view || op.matrix.data) return;
    clearTimeout(this.timer);
    const go = async () => {
      const mine = ++this.stamp;
      const want = this.#visible();
      const outRows = this.cy >= 1 ? want.rows : Math.max(1, Math.min(want.rows, Math.round(want.rows * this.cy)));
      const outCols = this.cx >= 1 ? want.cols : Math.max(1, Math.min(want.cols, Math.round(want.cols * this.cx)));
      const base = op.matrix.view;
      const view = { ...base, row0: (base.row0 ?? 0) + want.r, col0: (base.col0 ?? 0) + want.c, rows: want.rows, cols: want.cols };
      try {
        const { data } = await this.app.engine.call("region", { view, rows: outRows, cols: outCols });
        if (mine !== this.stamp || this.op !== op) return;
        this.region = { ...want, outRows, outCols, data };
        this.draw();
      } catch { /* the bench shows what it has */ }
    };
    if (now) go();
    else this.timer = setTimeout(go, 40);
  }

  // ---- drawing ----------------------------------------------------------------------------------------
  draw() {
    const { w, h, ratio } = this.size;
    const c = this.context;
    const style = getComputedStyle(this.canvas);
    const colors = {
      well: style.getPropertyValue("--na-well").trim() || "#eee",
      card: style.getPropertyValue("--anatomy-card").trim(),
      faint: style.getPropertyValue("--anatomy-faint").trim(),
      lo: style.getPropertyValue("--anatomy-lo").trim(),
      mid: style.getPropertyValue("--anatomy-mid").trim(),
      hi: style.getPropertyValue("--anatomy-hi").trim(),
      strong: style.getPropertyValue("--anatomy-strong").trim(),
      text: style.getPropertyValue("--anatomy-text").trim(),
      accent: style.getPropertyValue("--na-accent").trim() || "#ffb000",
      mono: style.getPropertyValue("--anatomy-mono").trim(),
    };
    const inks = inksOf(this.canvas);
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, w, h);
    if (!this.op) return;
    const op = this.op, cx = this.cx, cy = this.cy, view = this.#visible();
    const L = this.left, T = this.top, colW = this.colW, rowH = this.rowH;

    // matrix
    c.save();
    c.beginPath();
    c.rect(L, T, this.spanW, this.spanH);
    c.clip();
    c.fillStyle = colors.well;
    c.fillRect(L, T, Math.min(this.spanW, this.cols * cx), Math.min(this.spanH, this.rows * cy));
    this.#paintMatrix(c, view, inks);
    if (cx >= 7 && cy >= 7) this.#grid(c, view, colors.faint);
    if (cx >= NUMBERS_FROM && cy >= 13) this.#matrixNumbers(c, view, colors, inks);
    if (this.selected >= 0) {
      c.strokeStyle = colors.accent;
      c.lineWidth = 2;
      c.strokeRect(Math.round(this.#x(this.selected)) + 0.5, Math.round(this.#y(0)) + 0.5, Math.max(2, cx), this.rows * cy);
    }
    if (this.pinned >= 0) {
      c.strokeStyle = colors.accent;
      c.lineWidth = 1;
      c.strokeRect(Math.round(this.#x(0)) + 0.5, Math.round(this.#y(this.pinned)) + 0.5, this.cols * cx, Math.max(2, cy));
    }
    c.restore();

    // input column, and the chosen column and products beside it
    const wide = colW >= 40;
    const columns = [{ label: wide ? "input" : "in", values: op.input.values, ink: this.inkInput, kind: "input" }];
    if (this.selected >= 0) {
      columns.push({ label: wide ? "× weight" : "×w", values: this.column, ink: this.inkColumn, kind: "column" });
      columns.push({ label: wide ? "= product" : "=", values: this.products, ink: this.inkProducts, kind: "products", reveal: this.sweep });
    }
    columns.forEach((col, k) => {
      const x = LABEL_W + k * (colW + 8);
      c.font = `10px ${colors.mono}`;
      c.fillStyle = colors.text;
      c.textAlign = "center";
      c.fillText(col.label, x + colW / 2, T - 6);
      this.#paintVector(c, col, { x, y: T, w: colW, vertical: true, inks, colors });
    });

    // outputs across the top
    const rowsTop = [{ label: op.output.label, values: op.output.values, ink: this.inkOutput[0] }, ...op.after.map((a, k) => ({ label: a.label, values: a.values, ink: this.inkOutput[k + 1], positiveOnly: a.positiveOnly }))];
    rowsTop.forEach((row, k) => {
      const y = 22 + (rowsTop.length - 1 - k) * (rowH + 5);
      c.font = `11px ${colors.mono}`;
      c.fillStyle = colors.text;
      c.textAlign = "right";
      c.fillText(row.label, L - 10, y + rowH / 2 + 4);
      this.#paintVector(c, row, { x: L, y, h: rowH, vertical: false, inks, colors });
    });
    c.font = `10px ${colors.mono}`;
    c.fillStyle = colors.accent;
    c.textAlign = "left";
    if (cx < NUMBERS_FROM) c.fillText(`${op.matrix.label}  ·  ${op.matrix.rows.toLocaleString()} × ${op.matrix.cols.toLocaleString()}`, L, T - 6);
    else {
      // zoomed in: name each column (a token, a neuron, an index)
      c.save();
      c.beginPath();
      c.rect(L, T - 20, this.spanW, 18);
      c.clip();
      c.textAlign = "center";
      const room = Math.max(3, Math.floor(cx / 6.2));
      for (let col = view.c; col < view.c + view.cols; col++) {
        const text = String(op.matrix.colLabel ? op.matrix.colLabel(col) : col);
        c.fillText(text.length > room ? `${text.slice(0, room - 1)}…` : text, this.#x(col) + cx / 2, T - 6);
      }
      c.restore();
    }
    if (cy >= 13) {
      // zoomed in: name each row beside the input
      c.save();
      c.beginPath();
      c.rect(0, T, LABEL_W - 4, this.spanH);
      c.clip();
      c.font = `10px ${colors.mono}`;
      c.fillStyle = colors.text;
      c.textAlign = "right";
      c.textBaseline = "middle";
      for (let r = view.r; r < view.r + view.rows; r++) {
        const text = String(op.input.rowLabel ? op.input.rowLabel(r) : r);
        c.fillText(text.length > 18 ? `${text.slice(0, 17)}…` : text, LABEL_W - 8, this.#y(r) + cy / 2);
      }
      c.restore();
      c.textBaseline = "alphabetic";
    }

    // the hovered cell
    if (this.hover) {
      c.strokeStyle = colors.strong;
      c.lineWidth = 1;
      const { kind, r, col } = this.hover;
      if (kind === "matrix") c.strokeRect(Math.round(this.#x(col)) + 0.5, Math.round(this.#y(r)) + 0.5, Math.max(3, cx), Math.max(3, cy));
      if (kind === "top") c.strokeRect(Math.round(this.#x(col)) + 0.5, 22 + 0.5, Math.max(3, cx), rowsTop.length * (rowH + 5) - 5);
    }
  }

  #paintMatrix(c, view, inks) {
    const op = this.op, cx = this.cx, cy = this.cy;
    let data, outRows, outCols, at;
    if (op.matrix.data) {
      // pool locally when a pixel holds several cells: keep the largest in each block
      outRows = cy >= 1 ? view.rows : Math.max(1, Math.round(view.rows * cy));
      outCols = cx >= 1 ? view.cols : Math.max(1, Math.round(view.cols * cx));
      data = new Float32Array(outRows * outCols);
      const src = op.matrix.data, C = op.matrix.cols;
      for (let orow = 0; orow < outRows; orow++) {
        const ra = view.r + Math.floor((orow * view.rows) / outRows), rb = view.r + Math.max(Math.floor(((orow + 1) * view.rows) / outRows), Math.floor((orow * view.rows) / outRows) + 1);
        for (let ocol = 0; ocol < outCols; ocol++) {
          const ca = view.c + Math.floor((ocol * view.cols) / outCols), cb = view.c + Math.max(Math.floor(((ocol + 1) * view.cols) / outCols), Math.floor((ocol * view.cols) / outCols) + 1);
          let best = 0;
          for (let r = ra; r < rb; r++) for (let cc = ca; cc < cb; cc++) { const v = src[r * C + cc]; if (Math.abs(v) > Math.abs(best)) best = v; }
          data[orow * outCols + ocol] = best;
        }
      }
      at = view;
    } else if (this.region) {
      ({ data, outRows, outCols } = this.region);
      at = this.region;
    } else return;
    const level = (this.inkMatrix || 1) * (cx * cy < 1 ? 1.6 : 1);
    this.#blit(c, data, outCols, outRows, level, inks, this.#x(at.c), this.#y(at.r), at.cols * cx, at.rows * cy);
  }

  #blit(c, data, cols, rows, level, inks, x, y, w, h, positiveOnly = false, reveal = 1) {
    this.tile.width = cols;
    this.tile.height = rows;
    const t = this.tile.getContext("2d");
    const image = t.createImageData(cols, rows);
    const out = image.data, inv = level > 0 ? 1 / level : 0;
    const shown = Math.round(reveal * rows * cols);
    for (let i = 0; i < rows * cols; i++) {
      if (i >= shown) break;
      const v = data[i];
      if (!(v > 0) && (positiveOnly || !(v < 0))) continue;
      const color = v < 0 ? inks.red : inks.ink;
      out[i * 4] = color[0];
      out[i * 4 + 1] = color[1];
      out[i * 4 + 2] = color[2];
      out[i * 4 + 3] = Math.round(255 * Math.pow(Math.min(1, Math.abs(v) * inv), GAMMA));
    }
    t.putImageData(image, 0, 0);
    c.imageSmoothingEnabled = false;
    c.drawImage(this.tile, x, y, w, h);
  }

  /** One vector, along the matrix rows (vertical) or columns (horizontal), pooled to the pixels it has. */
  #paintVector(c, vec, { x, y, w, h, vertical, inks, colors }) {
    const cell = vertical ? this.cy : this.cx;
    const count = vertical ? this.rows : this.cols;
    const start = vertical ? this.origin.r : this.origin.c;
    const span = vertical ? this.spanH : this.spanW;
    const length = Math.min(span, (count - Math.max(0, start)) * cell);
    c.fillStyle = colors.well;
    if (vertical) c.fillRect(x, y, w, Math.max(0, length)); else c.fillRect(x, y, Math.max(0, length), h);
    if (!vec.values) return;
    const first = Math.max(0, Math.floor(start)), last = Math.min(count, Math.ceil(start + span / cell));
    const n = Math.max(1, last - first);
    const pixels = cell >= 1 ? n : Math.max(1, Math.round(n * cell));
    const data = new Float32Array(pixels);
    for (let p = 0; p < pixels; p++) {
      const a = first + Math.floor((p * n) / pixels), b = first + Math.max(Math.floor(((p + 1) * n) / pixels), Math.floor((p * n) / pixels) + 1);
      let best = 0;
      for (let i = a; i < b && i < vec.values.length; i++) if (Math.abs(vec.values[i]) > Math.abs(best)) best = vec.values[i];
      data[p] = best;
    }
    const offset = (first - start) * cell;
    const extent = n * cell;
    c.save();
    c.beginPath();
    if (vertical) c.rect(x, y, w, span); else c.rect(x, y, span, h);
    c.clip();
    const reveal = vec.reveal ?? 1;
    if (vertical) this.#blit(c, data, 1, pixels, vec.ink, inks, x, y + offset, w, extent, vec.positiveOnly, reveal);
    else this.#blit(c, data, pixels, 1, vec.ink, inks, x + offset, y, extent, h, vec.positiveOnly);
    // the numbers, when there is room
    if (vertical ? cell >= 13 && w >= 40 : cell >= NUMBERS_FROM) {
      c.font = `${Math.min(12, Math.max(9, (vertical ? w : cell) * 0.2))}px ${colors.mono}`;
      c.textAlign = "center";
      c.textBaseline = "middle";
      const shown = vertical ? Math.round(reveal * this.rows) : Infinity;
      for (let i = first; i < last && i < vec.values.length; i++) {
        if (i >= shown) break;
        const v = vec.values[i];
        const strong = Math.pow(Math.min(1, Math.abs(v) / (vec.ink || 1)), GAMMA) > 0.55;
        c.fillStyle = strong ? colors.card : colors.hi;
        if (vertical) c.fillText(fmt(v), x + w / 2, this.#y(i) + cell / 2);
        else c.fillText(fmt(v), this.#x(i) + cell / 2, y + h / 2);
      }
      c.textBaseline = "alphabetic";
    }
    if (!vertical && this.selected >= 0) {
      c.strokeStyle = colors.accent;
      c.lineWidth = 2;
      c.strokeRect(Math.round(this.#x(this.selected)) + 0.5, y + 0.5, Math.max(3, cell), h - 1);
    }
    if (vertical && this.pinned >= 0) {
      c.strokeStyle = colors.accent;
      c.lineWidth = 1.5;
      c.strokeRect(x + 0.5, Math.round(this.#y(this.pinned)) - 1.5, w - 1, Math.max(4, cell) + 2);
    }
    c.restore();
  }

  #grid(c, view, color) {
    c.strokeStyle = color;
    c.lineWidth = 1;
    c.beginPath();
    for (let col = view.c; col <= view.c + view.cols; col++) { const x = Math.round(this.#x(col)) + 0.5; c.moveTo(x, this.top); c.lineTo(x, this.top + Math.min(this.spanH, this.rows * this.cy)); }
    for (let r = view.r; r <= view.r + view.rows; r++) { const y = Math.round(this.#y(r)) + 0.5; c.moveTo(this.left, y); c.lineTo(this.left + Math.min(this.spanW, this.cols * this.cx), y); }
    c.stroke();
  }

  #matrixValue(r, col) {
    const op = this.op;
    if (op.matrix.data) return op.matrix.data[r * op.matrix.cols + col];
    const g = this.region;
    if (!g || r < g.r || col < g.c || r >= g.r + g.rows || col >= g.c + g.cols || g.outRows !== g.rows || g.outCols !== g.cols) return undefined;
    return g.data[(r - g.r) * g.cols + (col - g.c)];
  }

  #matrixNumbers(c, view, colors) {
    const level = this.inkMatrix || 1;
    c.font = `${Math.min(13, Math.max(9, Math.min(this.cx * 0.22, this.cy * 0.6)))}px ${colors.mono}`;
    c.textAlign = "center";
    c.textBaseline = "middle";
    for (let r = view.r; r < view.r + view.rows; r++) {
      for (let col = view.c; col < view.c + view.cols; col++) {
        const v = this.#matrixValue(r, col);
        if (v === undefined) continue;
        const strong = Math.pow(Math.min(1, Math.abs(v) / level), GAMMA) > 0.55;
        c.fillStyle = strong ? colors.card : colors.hi;
        c.fillText(fmt(v), this.#x(col) + this.cx / 2, this.#y(r) + this.cy / 2);
      }
    }
    c.textBaseline = "alphabetic";
  }

  // ---- pointer --------------------------------------------------------------------------------------------
  #hit(x, y) {
    const r = Math.floor(this.origin.r + (y - this.top) / this.cy);
    const col = Math.floor(this.origin.c + (x - this.left) / this.cx);
    if (x >= this.left && y >= this.top && r < this.rows && col < this.cols && r >= 0 && col >= 0) return { kind: "matrix", r, col };
    if (x >= this.left && y < this.top && y >= 22 && col >= 0 && col < this.cols) {
      const k = (1 + this.op.after.length) - 1 - Math.floor((y - 22) / (this.rowH + 5));
      return { kind: "top", col, row: Math.max(0, Math.min(this.op.after.length, k)) };
    }
    if (x >= LABEL_W && x < this.left && y >= this.top && r >= 0 && r < this.rows) {
      const k = Math.floor((x - LABEL_W) / (this.colW + 8));
      return { kind: "left", r, which: ["input", "column", "products"][k] ?? "input" };
    }
    return null;
  }
  describe(hit) {
    if (!hit || !this.op) return "";
    const op = this.op;
    if (hit.kind === "matrix") {
      const v = this.#matrixValue(hit.r, hit.col);
      return `${op.matrix.label}: ${op.matrix.name(hit.r, hit.col)} = ${v === undefined ? (this.cx < 1 || this.cy < 1 ? "zoom in for exact values" : "…") : signed(v, 5)}. Click to work out ${op.columnName(hit.col)}.`;
    }
    if (hit.kind === "top") {
      const row = hit.row === 0 ? op.output : op.after[hit.row - 1];
      return `${row.label}: ${op.columnName(hit.col)} = ${signed(row.values[hit.col], 5)}. Click to see its dot product.`;
    }
    const i = hit.r;
    if (hit.which === "column" && this.column) return `weight ${i} of the chosen column: ${signed(this.column[i], 5)}`;
    if (hit.which === "products" && this.products) return `${op.input.name(i)} × weight = ${signed(op.input.values[i], 4)} × ${signed(this.column[i], 4)} = ${signed(this.products[i], 5)}`;
    return `${op.input.name(i)} = ${signed(op.input.values[i], 5)}`;
  }

  #listen() {
    const canvas = this.canvas;
    const pointers = new Map();
    let moved = false;
    const local = (event) => { const box = canvas.getBoundingClientRect(); return [event.clientX - box.left, event.clientY - box.top]; };
    canvas.addEventListener("pointerdown", (event) => {
      pointers.set(event.pointerId, local(event));
      moved = false;
      try { canvas.setPointerCapture(event.pointerId); } catch { /* not capturable */ }
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.op) return;
      const [x, y] = local(event);
      const last = pointers.get(event.pointerId);
      if (last && pointers.size === 2) {
        const other = [...pointers.entries()].find(([id]) => id !== event.pointerId)[1];
        const before = Math.hypot(last[0] - other[0], last[1] - other[1]), after = Math.hypot(x - other[0], y - other[1]);
        pointers.set(event.pointerId, [x, y]);
        if (before > 0) this.zoomAt(after / before, (x + other[0]) / 2, (y + other[1]) / 2);
        moved = true;
        return;
      }
      if (last && (Math.abs(x - last[0]) + Math.abs(y - last[1]) > 2 || moved)) {
        moved = true;
        this.origin.c -= (x - last[0]) / this.cx;
        this.origin.r -= (y - last[1]) / this.cy;
        pointers.set(event.pointerId, [x, y]);
        this.#clamp();
        this.hover = null;
        this.draw();
        this.#request();
        return;
      }
      this.hover = this.#hit(x, y);
      this.onHover(this.describe(this.hover));
      this.draw();
    });
    const release = (event) => {
      const was = pointers.get(event.pointerId);
      pointers.delete(event.pointerId);
      try { canvas.releasePointerCapture(event.pointerId); } catch { /* was not captured */ }
      if (!was || moved || !this.op) return;
      const hit = this.#hit(...local(event));
      if (!hit) return;
      if (hit.kind === "matrix") { this.pinned = hit.r; this.select(hit.col); }
      else if (hit.kind === "top") { this.pinned = -1; this.select(hit.col); }
      else { this.pinned = hit.r; this.draw(); this.onSelect(this.selected); }
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", (event) => pointers.delete(event.pointerId));
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.onHover(""); this.draw(); });
    canvas.addEventListener("wheel", (event) => {
      if (!this.op || !(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      const [x, y] = local(event);
      this.zoomAt(Math.exp(-event.deltaY * 0.0045), x, y);
    }, { passive: false });
    canvas.addEventListener("dblclick", (event) => { const [x, y] = local(event); this.zoomAt(event.shiftKey ? 0.4 : 2.5, x, y); });
    canvas.addEventListener("keydown", (event) => {
      if (!this.op) return;
      const stepC = Math.max(1, this.spanW / this.cx / 8), stepR = Math.max(1, this.spanH / this.cy / 8);
      let used = true;
      if (event.key === "ArrowLeft") this.origin.c -= stepC;
      else if (event.key === "ArrowRight") this.origin.c += stepC;
      else if (event.key === "ArrowUp") this.origin.r -= stepR;
      else if (event.key === "ArrowDown") this.origin.r += stepR;
      else if (event.key === "+" || event.key === "=") return this.zoomAt(1.6), event.preventDefault();
      else if (event.key === "-" || event.key === "_") return this.zoomAt(1 / 1.6), event.preventDefault();
      else if (event.key === "0") return this.fit(), event.preventDefault();
      else used = false;
      if (!used) return;
      event.preventDefault();
      this.#clamp();
      this.draw();
      this.#request();
    });
  }
}
