// The bench's second canvas: one unit for every token at once.
//
//   arcs  tokens │ stream in ═╪═ Q K V ─╳─ attention ─╳─ head out ═╪═ … stream out
//                │  one row per token, the same rows all the way across
//
// Parallel lines between two stages mean each token is worked on alone, all
// with the same weight matrix (the slab they pass through). Crossing lines mean
// tokens read from each other: that only happens in attention, and only
// backwards, which the grey triangle of the attention grid (the mask) shows.
//
// Three ways to run it:
//   "all"      reading the prompt: every row at once
//   "nocache"  generating, step t: rows 0..t-1 all computed again
//   "cache"    generating, step t: only row t-1; earlier keys and values from memory

import { inksOf, signed } from "../../../ui/plates.js";

const GAMMA = 0.62;
const WAVE_MS = 900;

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

const SHORT = { q: "Q", k: "K", v: "V", mix: "head out", attn: "attn adds", mlp: "MLP adds", ids: "id" };

export class SequenceView {
  constructor(canvas, { onHover = () => {}, onPick = () => {} } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onHover = onHover;
    this.onPick = onPick;
    this.seq = null;
    this.mode = "all";
    this.t = 1; // tokens the model has at this step
    this.position = 0; // the token the rest of the app follows
    this.wave = 1;
    this.hover = null;
    this.size = { w: 0, h: 0, ratio: 1 };
    this.images = new Map();
    this.#listen();
    new ResizeObserver(() => { if (this.#resize()) this.draw(); }).observe(canvas);
  }

  /** The height the canvas needs to give each token a comfortable row. */
  static heightFor(T) {
    return Math.round(Math.max(280, Math.min(window.innerHeight * 0.7, 44 + 46 + 12 + T * 30)));
  }

  set(seq) {
    if (this.seq !== seq) this.images.clear();
    this.seq = seq;
    this.inks = seq.stages.map((s) => (s.kind === "rows" ? fullInk(s.data, s.positiveOnly) : 1));
    this.t = Math.max(1, Math.min(seq.T, this.t));
    this.#resize();
    this.draw();
  }
  setMode(mode, t = this.t) {
    this.mode = mode;
    this.t = Math.max(1, Math.min(this.seq?.T ?? 1, t));
    this.play();
  }
  /** Runs the wave of computation through the stages. */
  play() {
    cancelAnimationFrame(this.frame);
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { this.wave = 1; this.draw(); return; }
    const start = performance.now();
    const step = (now) => {
      this.wave = Math.min(1, (now - start) / WAVE_MS);
      this.draw();
      if (this.wave < 1) this.frame = requestAnimationFrame(step);
    };
    this.wave = 0;
    this.frame = requestAnimationFrame(step);
  }

  #resize() {
    const box = this.canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(200, Math.round(box.width)), h = Math.max(200, Math.round(box.height));
    if (w === this.size.w && h === this.size.h && ratio === this.size.ratio) return false;
    this.size = { w, h, ratio };
    this.canvas.width = Math.round(w * ratio);
    this.canvas.height = Math.round(h * ratio);
    this.images.clear();
    return true;
  }

  // ---- layout ----------------------------------------------------------------------------------------
  #layout() {
    const seq = this.seq;
    const { w, h } = this.size;
    const pad = 12;
    const hasArcs = seq.stages.some((s) => s.kind === "grid");
    const arcsW = hasArcs ? Math.min(70, w * 0.05 + 20) : 0;
    const labelW = Math.min(118, Math.max(70, w * 0.08));
    const header = 44, footer = 46;
    const rowsAvail = h - header - footer - pad;
    const rh = Math.min(30, rowsAvail / seq.T);
    const rowsH = rh * seq.T;
    const gapOf = (s) => (!s.via ? (s.joined ? 4 : 0) : s.via.mixes ? 46 : s.via.weight ? 40 : 30);
    const gaps = seq.stages.reduce((sum, s, k) => sum + (k ? gapOf(s) : 0), 0);
    const weightOf = (s) => (s.kind === "text" ? (s.wide ? 3.2 : 0.55) : s.n >= 2000 ? 1.45 : s.n >= 256 ? 1 : 0.4);
    const free = w - pad * 2 - arcsW - labelW - gaps;
    const sumW = seq.stages.reduce((sum, s) => sum + (s.kind === "grid" ? 0 : weightOf(s)), 0);
    let unit = free / (sumW + 2);
    let gridW = 0;
    for (let pass = 0; pass < 2; pass++) {
      const grid = seq.stages.some((s) => s.kind === "grid");
      gridW = grid ? Math.max(0.6 * unit, Math.min(2.6 * unit, rowsH)) : 0;
      unit = (free - gridW) / sumW;
    }
    let x = pad + arcsW + labelW;
    const boxes = seq.stages.map((s, k) => {
      const gap = k ? gapOf(s) : 0;
      x += gap;
      const width = s.kind === "grid" ? gridW : Math.max(8, unit * weightOf(s));
      const box = { x, w: width, gap, gapX: x - gap };
      x += width;
      return box;
    });
    return { pad, arcsW, labelW, header, footer, rh, rowsH, top: header, boxes, labelX: pad + arcsW };
  }

  // ---- what each row is doing in this mode ------------------------------------------------------------------
  /** "on" (computed now), "cached" (read from the KV cache), "ghost" (done earlier, unused), "absent" (not yet). */
  stateOf(stage, i) {
    if (this.mode === "all") return "on";
    const now = this.t - 1;
    if (i > now) return "absent";
    if (this.mode === "nocache" || i === now) return "on";
    return stage.role === "kv" ? "cached" : "ghost";
  }
  get focusRow() {
    if (this.hover && this.hover.row >= 0) return this.hover.row;
    return this.mode === "all" ? Math.min(this.position, this.seq.T - 1) : this.t - 1;
  }

  // ---- drawing ------------------------------------------------------------------------------------------------
  draw() {
    const { w, h, ratio } = this.size;
    const c = this.context;
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, w, h);
    const seq = this.seq;
    if (!seq) return;
    const style = getComputedStyle(this.canvas);
    const color = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    const colors = {
      well: color("--na-well", "#eee"), card: color("--anatomy-card", "#fff"), faint: color("--anatomy-faint", "#ddd"),
      lo: color("--anatomy-lo", "#bbb"), mid: color("--anatomy-mid", "#888"), text: color("--anatomy-text", "#555"),
      strong: color("--anatomy-strong", "#111"), muted: color("--anatomy-muted", "#777"),
      accent: color("--na-accent", "#ffb000"), blue: color("--anatomy-blue", "#3e63dd"), mono: color("--anatomy-mono", "monospace"),
    };
    const inks = inksOf(this.canvas);
    const L = this.layout = this.#layout();
    const { rh, rowsH, top, boxes } = L;
    const T = seq.T;
    const nStages = seq.stages.length;
    const reached = this.wave * nStages; // stages the wave has passed
    const lit = (k) => reached >= k + 0.35;
    const focus = this.focusRow;
    const rowY = (i) => top + i * rh;

    // token labels
    c.font = `${Math.min(12, Math.max(9, rh * 0.55))}px ${colors.mono}`;
    c.textBaseline = "middle";
    c.textAlign = "right";
    const every = Math.max(1, Math.ceil(10 / rh));
    for (let i = 0; i < T; i++) {
      if (i % every && i !== focus) continue;
      const state = this.mode === "all" ? "on" : i > this.t - 1 ? "absent" : "on";
      c.fillStyle = i === focus ? colors.accent : state === "absent" ? colors.lo : colors.text;
      const piece = String(seq.pieces[i] ?? "").replace(/^ /, "·").replace(/\n/g, "↵");
      const text = `${i + 1} ${piece.length > 12 ? `${piece.slice(0, 11)}…` : piece}`;
      c.fillText(text, L.labelX + L.labelW - 8, rowY(i) + rh / 2);
    }

    // connectors
    seq.stages.forEach((stage, k) => {
      if (!k || !stage.via) return;
      const a = boxes[k - 1], b = boxes[k];
      const x0 = a.x + a.w, x1 = b.x;
      const active = this.wave < 1 && reached >= k - 0.2 && reached < k + 0.6;
      c.save();
      if (stage.via.mixes) {
        // tokens read from earlier tokens: lines from every row j ≤ i into row i
        const shares = seq.stages.find((s) => s.kind === "grid")?.data;
        const rowsDrawn = T <= 48 ? Array.from({ length: T }, (_, i) => i) : [focus];
        for (const i of rowsDrawn) {
          const st = this.stateOf(stage, i);
          if (st === "absent" || st === "ghost") continue;
          const strongRow = i === focus;
          for (let j = 0; j <= i; j++) {
            const share = shares ? shares[i * T + j] : 1 / (i + 1);
            const alpha = Math.min(1, (strongRow ? 0.25 + 0.9 * Math.sqrt(share) : 0.05 + 0.35 * Math.sqrt(share)));
            c.strokeStyle = strongRow || active ? colors.accent : colors.mid;
            c.globalAlpha = alpha;
            c.lineWidth = strongRow ? Math.max(0.8, Math.min(3, share * 6)) : 0.7;
            c.beginPath();
            const ya = rowY(j) + rh / 2, yb = rowY(i) + rh / 2;
            c.moveTo(x0 + 2, ya);
            c.bezierCurveTo(x0 + (x1 - x0) * 0.55, ya, x0 + (x1 - x0) * 0.45, yb, x1 - 2, yb);
            c.stroke();
          }
        }
        c.globalAlpha = 1;
      } else {
        // each token alone, through the same weights
        const slab = stage.via.weight ? 7 : 0;
        const mid = (x0 + x1) / 2;
        for (let i = 0; i < T; i++) {
          const st = this.stateOf(stage, i);
          if (st === "absent") continue;
          c.strokeStyle = active && st === "on" ? colors.accent : st === "on" ? colors.lo : colors.faint;
          c.globalAlpha = st === "on" ? (i === focus ? 1 : 0.8) : 0.5;
          c.lineWidth = rh >= 4 ? (i === focus ? 1.5 : 1) : rh;
          c.beginPath();
          c.moveTo(x0 + 2, rowY(i) + rh / 2);
          c.lineTo(x1 - 2, rowY(i) + rh / 2);
          c.stroke();
        }
        c.globalAlpha = 1;
        if (slab) {
          c.fillStyle = active ? colors.accent : colors.strong;
          c.globalAlpha = active ? 0.9 : 0.55;
          c.fillRect(mid - slab / 2, top - 6, slab, rowsH + 12);
          c.globalAlpha = 1;
        }
      }
      c.restore();
    });

    // stages
    seq.stages.forEach((stage, k) => {
      const box = boxes[k];
      c.fillStyle = colors.well;
      c.fillRect(box.x, top, box.w, rowsH);
      if (stage.kind === "rows") this.#paintRows(c, stage, k, box, L, inks);
      else if (stage.kind === "grid") this.#paintGrid(c, stage, box, L, inks, colors);
      else this.#paintText(c, stage, box, L, colors);
      // what each row is doing
      // (consecutive rows in the same state are filled as one block, so thin rows leave no seams)
      const bar = (y0, y1, fill) => {
        c.fillStyle = fill;
        if (rh >= 4) for (let y = y0; y < y1 - 0.5; y += rh) c.fillRect(box.x - 3, y + 0.5, 3, rh - 1);
        else c.fillRect(box.x - 3, y0, 3, y1 - y0);
      };
      for (let i = 0; i < T;) {
        let st = this.stateOf(stage, i);
        if (st === "on" && !lit(k)) st = "pending";
        let j = i + 1;
        while (j < T) {
          let next = this.stateOf(stage, j);
          if (next === "on" && !lit(k)) next = "pending";
          if (next !== st) break;
          j++;
        }
        const y0 = rowY(i), y1 = rowY(j);
        if (st === "absent") { c.fillStyle = colors.card; c.fillRect(box.x, y0, box.w, y1 - y0); }
        if (st === "ghost" || st === "pending") { c.fillStyle = colors.card; c.globalAlpha = st === "pending" ? 0.85 : 0.78; c.fillRect(box.x, y0, box.w, y1 - y0); c.globalAlpha = 1; }
        if (st === "cached") {
          c.fillStyle = colors.blue;
          c.globalAlpha = 0.3;
          c.fillRect(box.x, y0, box.w, y1 - y0);
          c.globalAlpha = 1;
          bar(y0, y1, colors.blue);
        }
        if (st === "on" && this.mode !== "all") bar(y0, y1, colors.accent);
        i = j;
      }
      if (this.mode !== "all") {
        // the area still to be written
        const now = this.t - 1;
        if (now < T - 1) {
          c.strokeStyle = colors.faint;
          c.setLineDash([3, 3]);
          c.strokeRect(box.x + 0.5, rowY(now + 1) + 0.5, box.w - 1, rowsH - (now + 1) * rh - 1);
          c.setLineDash([]);
        }
        if (lit(k) && this.stateOf(stage, now) === "on") {
          c.strokeStyle = colors.accent;
          c.lineWidth = 1.5;
          c.strokeRect(box.x + 0.5, rowY(now) + 0.5, box.w - 1, Math.max(2, rh - 1));
        }
      }
      // focus row outline
      if (focus >= 0 && rh >= 3) {
        c.strokeStyle = colors.strong;
        c.lineWidth = 1;
        c.globalAlpha = 0.6;
        c.strokeRect(box.x - 0.5, rowY(focus) - 0.5, box.w + 1, rh + 1);
        c.globalAlpha = 1;
      }
      // header: name and shape
      const narrow = box.w < 70;
      const name = narrow ? SHORT[stage.id] ?? stage.label : stage.label;
      c.textAlign = "center";
      c.textBaseline = "alphabetic";
      c.font = `600 11px ${colors.mono}`;
      c.fillStyle = lit(k) || this.wave >= 1 ? colors.strong : colors.muted;
      c.fillText(this.#fit(c, name, box.w + (stage.joined ? 0 : 14)), box.x + box.w / 2, top - 22);
      c.font = `10px ${colors.mono}`;
      c.fillStyle = colors.muted;
      if (stage.id === "q" && seq.stages[k + 2]?.id === "v") {
        // query, key and value share one shape: write it once across the three
        const end = boxes[k + 2];
        c.fillText(`${stage.shape.replace(/ × /, "×")} each`, (box.x + end.x + end.w) / 2, top - 8);
      } else if (!stage.joined) {
        c.fillText(this.#fit(c, narrow ? stage.shape.replace(/ × /, "×") : stage.shape, box.w + 12), box.x + box.w / 2, top - 8);
      }
    });

    // operations, under the gaps
    c.font = `10px ${colors.mono}`;
    c.textBaseline = "alphabetic";
    let lane = 0;
    seq.stages.forEach((stage, k) => {
      if (!k || !stage.via) return;
      const a = boxes[k - 1], b = boxes[k];
      const mid = (a.x + a.w + b.x) / 2;
      const y = top + rowsH + 16 + (lane++ % 2) * 14;
      c.fillStyle = stage.via.mixes ? colors.accent : colors.text;
      c.textAlign = "center";
      c.fillText(stage.via.short ?? stage.via.label, mid, y);
      c.strokeStyle = colors.faint;
      c.beginPath();
      c.moveTo(mid + 0.5, top + rowsH + 3);
      c.lineTo(mid + 0.5, y - 9);
      c.stroke();
    });

    // arcs: which earlier tokens the focused token reads, in this head
    const grid = seq.stages.find((s) => s.kind === "grid");
    if (grid && L.arcsW && focus >= 0 && this.stateOf(grid, focus) === "on" && lit(seq.stages.indexOf(grid))) {
      const xr = L.labelX - 2;
      for (let j = 0; j <= focus; j++) {
        const share = grid.data[focus * T + j];
        if (share < 0.003) continue;
        const ya = rowY(focus) + rh / 2, yb = rowY(j) + rh / 2;
        const bulge = Math.min(L.arcsW - 6, 10 + Math.abs(focus - j) * rh * 0.5);
        c.strokeStyle = this.stateOf(seq.stages.find((s) => s.role === "kv") ?? grid, j) === "cached" ? colors.blue : colors.accent;
        c.globalAlpha = 0.25 + 0.75 * Math.sqrt(share);
        c.lineWidth = Math.max(0.8, Math.min(5, share * 9));
        c.beginPath();
        if (j === focus) {
          c.arc(xr - 6, ya, 5, 0, Math.PI * 2);
        } else {
          c.moveTo(xr, ya);
          c.bezierCurveTo(xr - bulge, ya, xr - bulge, yb, xr, yb);
        }
        c.stroke();
      }
      c.globalAlpha = 1;
    }

    // hover
    if (this.hover && this.hover.stage >= 0) {
      const box = boxes[this.hover.stage];
      const stage = seq.stages[this.hover.stage];
      c.strokeStyle = colors.accent;
      c.lineWidth = 2;
      if (stage.kind === "grid") {
        const cw = box.w / T;
        c.strokeRect(box.x + this.hover.col * cw, rowY(this.hover.row), Math.max(2, cw), Math.max(2, rh));
      } else if (stage.kind === "rows" && this.hover.col >= 0) {
        const cw = box.w / stage.n;
        c.strokeRect(box.x + this.hover.col * cw - 0.5, rowY(this.hover.row) - 0.5, Math.max(2, cw) + 1, Math.max(2, rh) + 1);
      }
    }
  }

  #fit(c, text, room) {
    if (c.measureText(text).width <= room) return text;
    let s = text;
    while (s.length > 2 && c.measureText(`${s}…`).width > room) s = s.slice(0, -1);
    return `${s}…`;
  }

  #image(stage, k, cols, positiveOnly, inks, level) {
    const key = `${k}|${cols}`;
    if (this.images.has(key)) return this.images.get(key);
    const T = this.seq.T, n = stage.n;
    const tile = document.createElement("canvas");
    tile.width = cols;
    tile.height = T;
    const t = tile.getContext("2d");
    const image = t.createImageData(cols, T);
    const out = image.data, inv = 1 / (level || 1);
    for (let i = 0; i < T; i++) {
      for (let p = 0; p < cols; p++) {
        const a = Math.floor((p * n) / cols), b = Math.max(a + 1, Math.floor(((p + 1) * n) / cols));
        let best = 0;
        for (let q = a; q < b; q++) { const v = stage.data[i * n + q]; if (Math.abs(v) > Math.abs(best)) best = v; }
        if (!(best > 0) && (positiveOnly || !(best < 0))) continue;
        const color = best < 0 ? inks.red : inks.ink;
        const o = (i * cols + p) * 4;
        out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2];
        out[o + 3] = Math.round(255 * Math.pow(Math.min(1, Math.abs(best) * inv), GAMMA));
      }
    }
    t.putImageData(image, 0, 0);
    this.images.set(key, tile);
    return tile;
  }

  #paintRows(c, stage, k, box, L, inks) {
    const cols = Math.max(1, Math.min(stage.n, Math.round(box.w)));
    const tile = this.#image(stage, k, cols, stage.positiveOnly, inks, this.inks[k]);
    c.imageSmoothingEnabled = false;
    c.drawImage(tile, box.x, L.top, box.w, L.rowsH);
  }

  #paintGrid(c, stage, box, L, inks, colors) {
    const T = this.seq.T;
    const key = `grid|${T}`;
    let tile = this.images.get(key);
    if (!tile) {
      tile = document.createElement("canvas");
      tile.width = T;
      tile.height = T;
      const t = tile.getContext("2d");
      const image = t.createImageData(T, T);
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
        const o = (i * T + j) * 4;
        image.data[o] = inks.ink[0]; image.data[o + 1] = inks.ink[1]; image.data[o + 2] = inks.ink[2];
        image.data[o + 3] = Math.round(255 * Math.pow(Math.min(1, stage.data[i * T + j]), 0.5));
      }
      t.putImageData(image, 0, 0);
      this.images.set(key, tile);
    }
    c.imageSmoothingEnabled = false;
    c.drawImage(tile, box.x, L.top, box.w, L.rowsH);
    // the mask: everything above the diagonal, a token's future
    const cw = box.w / T, rh = L.rh;
    c.save();
    c.beginPath();
    c.moveTo(box.x + cw, L.top);
    for (let i = 0; i < T; i++) { c.lineTo(box.x + (i + 1) * cw, L.top + i * rh); c.lineTo(box.x + (i + 1) * cw, L.top + (i + 1) * rh); }
    c.lineTo(box.x + box.w, L.top + L.rowsH);
    c.lineTo(box.x + box.w, L.top);
    c.closePath();
    c.fillStyle = colors.faint;
    c.globalAlpha = 0.55;
    c.fill();
    c.clip();
    c.globalAlpha = 0.6;
    c.strokeStyle = colors.lo;
    c.lineWidth = 1;
    c.beginPath();
    for (let x = -L.rowsH; x < box.w; x += 7) { c.moveTo(box.x + x, L.top + L.rowsH); c.lineTo(box.x + x + L.rowsH, L.top); }
    c.stroke();
    c.restore();
    if (cw >= 12 && rh >= 12) {
      c.font = `${Math.min(11, Math.max(8, Math.min(cw, rh) * 0.34))}px ${colors.mono}`;
      c.textAlign = "center";
      c.textBaseline = "middle";
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
        const v = stage.data[i * T + j];
        c.fillStyle = v > 0.35 ? colors.card : colors.text;
        c.fillText(v >= 0.995 ? "1" : `${Math.round(v * 100)}`, box.x + (j + 0.5) * cw, L.top + (i + 0.5) * rh);
      }
    }
  }

  #paintText(c, stage, box, L, colors) {
    const T = this.seq.T, rh = L.rh;
    if (rh < 8) return;
    c.font = `${Math.min(12, Math.max(9, rh * 0.5))}px ${colors.mono}`;
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillStyle = colors.strong;
    for (let i = 0; i < T; i++) c.fillText(this.#fit(c, stage.text(i), box.w - 10), box.x + 6, L.top + i * rh + rh / 2);
  }

  // ---- pointer -----------------------------------------------------------------------------------------------
  #hit(x, y) {
    const L = this.layout, seq = this.seq;
    if (!L || !seq) return null;
    const row = Math.floor((y - L.top) / L.rh);
    if (row < 0 || row >= seq.T) return null;
    if (x < L.labelX + L.labelW && x >= L.labelX - L.arcsW) return { stage: -1, row, col: -1 };
    for (let k = 0; k < seq.stages.length; k++) {
      const box = L.boxes[k], stage = seq.stages[k];
      if (x >= box.x && x < box.x + box.w) {
        const n = stage.kind === "grid" ? seq.T : stage.kind === "rows" ? stage.n : 1;
        return { stage: k, row, col: stage.kind === "text" ? -1 : Math.min(n - 1, Math.floor(((x - box.x) / box.w) * n)) };
      }
      if (k && x >= box.gapX && x < box.x && stage.via) return { stage: -1, gap: k, row, col: -1 };
    }
    return null;
  }

  describe(hit) {
    const seq = this.seq;
    if (!hit || !seq) return "";
    const who = seq.tokenName(hit.row);
    if (hit.gap) {
      const via = seq.stages[hit.gap].via;
      return `${via.label}: ${via.mixes ? "each token reads from itself and the tokens before it, weighted by its attention" : `each token on its own${via.weight ? `, all through the same ${via.weight}` : ""}`}.`;
    }
    if (hit.stage < 0) return `${who}. ${this.stateWords(seq.stages[1] ?? seq.stages[0], hit.row)}`;
    const stage = seq.stages[hit.stage];
    const state = this.stateWords(stage, hit.row);
    if (stage.kind === "grid") {
      const j = hit.col;
      if (j > hit.row) return `${who} cannot look at ${seq.tokenName(j)}: it comes later, so the mask sets its share to 0.`;
      return `${who} gives ${(stage.data[hit.row * seq.T + j] * 100).toFixed(1)}% of head ${seq.head}'s attention to ${seq.tokenName(j)}. ${state} Click to see the dot product.`;
    }
    if (stage.kind === "text") return `${stage.label}, ${who}: ${stage.text(hit.row)}`;
    return `${stage.label}, ${who}, number ${hit.col}: ${signed(stage.data[hit.row * stage.n + hit.col], 5)}. ${state} Click to see its arithmetic.`;
  }

  stateWords(stage, row) {
    const st = this.stateOf(stage, row);
    if (this.mode === "all") return "";
    if (st === "absent") return "Not written yet at this step.";
    if (st === "cached") return "Read from the KV cache, not computed again.";
    if (st === "ghost") return "Computed at an earlier step; not needed now.";
    return row === this.t - 1 ? "Computed at this step: the new token." : "Computed again at this step (no cache).";
  }

  #listen() {
    const canvas = this.canvas;
    const local = (event) => { const box = canvas.getBoundingClientRect(); return [event.clientX - box.left, event.clientY - box.top]; };
    canvas.addEventListener("pointermove", (event) => {
      if (!this.seq) return;
      const hit = this.#hit(...local(event));
      this.hover = hit;
      this.onHover(this.describe(hit));
      this.draw();
    });
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.onHover(""); this.draw(); });
    canvas.addEventListener("click", (event) => {
      const hit = this.#hit(...local(event));
      if (!hit || !this.seq) return;
      const stage = hit.stage >= 0 ? this.seq.stages[hit.stage] : null;
      if (stage && stage.kind === "grid" && hit.col > hit.row) return;
      this.onPick({ row: hit.row, stage, col: hit.col });
    });
  }
}
