// The GPU view's canvas: a GPU running one attention head.
//
//  ┌ off-chip memory (HBM) ┐ bus ┌ SMs: one thread block each ───────────────┐
//  │  Q  K  V  O           │ ══> │ [SM 0] [SM 1] [SM 2] …                     │
//  │                       │ <══ ├ work map ┬ inside one SM: on-chip SRAM ───┤
//  │  S   P  (standard)    │     │ tiles    │ q k v / S P m l acc            │
//  └───────────────────────┘     └──────────┴────────────────────────────────┘
//
// Everything drawn is the head's real numbers; dots on the bus are the tiles
// moving this tick.

import { inksOf, signed } from "../../../../ui/plates.js";
import { eventAt } from "./flash.js";

const GAMMA = 0.62;
const ANIM_MS = 650;

const fmt = (v) => {
  if (v === -Infinity) return "−∞";
  if (!Number.isFinite(v)) return "";
  const a = Math.abs(v);
  return (a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : a < 0.005 && a > 0 ? v.toExponential(0) : v.toFixed(2)).replace("-", "−");
};

function level(values) {
  const sample = [];
  const step = Math.max(1, Math.floor(values.length / 8000));
  for (let i = 0; i < values.length; i += step) if (Number.isFinite(values[i]) && values[i] !== 0) sample.push(Math.abs(values[i]));
  if (!sample.length) return 1;
  sample.sort((a, b) => a - b);
  return sample[Math.floor(sample.length * 0.99)] || 1;
}

export class GpuView {
  constructor(canvas, { onHover = () => {}, onPick = () => {} } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onHover = onHover;
    this.onPick = onPick;
    this.state = null;
    this.anim = 1;
    this.size = { w: 0, h: 0, ratio: 1 };
    this.hits = [];
    this.tiles = new Map();
    this.hover = null;
    this.#listen();
    new ResizeObserver(() => { if (this.#resize()) this.draw(); }).observe(canvas);
  }

  /** { head, plan, tick, selected, sim, device, follow } */
  set(state, { animate = false } = {}) {
    if (!this.state || this.state.head !== state.head) {
      this.tiles.clear();
      const h = state.head;
      this.levels = { Q: level(h.Q), K: level(h.K), V: level(h.V), O: level(h.out), S: 0, P: 1 };
      let top = 0;
      for (let i = 0; i < h.T; i++) for (let j = 0; j <= (h.causal ? i : h.T - 1); j++) {
        let dot = 0;
        for (let x = 0; x < h.s; x++) dot += h.Q[i * h.s + x] * h.K[j * h.s + x];
        top = Math.max(top, Math.abs(dot * h.scale));
      }
      this.levels.S = top || 1;
      this.scores = null;
    }
    this.state = state;
    this.#resize();
    cancelAnimationFrame(this.frame);
    if (!animate || matchMedia("(prefers-reduced-motion: reduce)").matches) { this.anim = 1; this.draw(); return; }
    const start = performance.now();
    const step = (now) => {
      this.anim = Math.min(1, (now - start) / ANIM_MS);
      this.draw();
      if (this.anim < 1) this.frame = requestAnimationFrame(step);
    };
    this.anim = 0;
    this.frame = requestAnimationFrame(step);
  }

  #resize() {
    const box = this.canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(320, Math.round(box.width)), h = Math.max(320, Math.round(box.height));
    if (w === this.size.w && h === this.size.h && ratio === this.size.ratio) return false;
    this.size = { w, h, ratio };
    this.canvas.width = Math.round(w * ratio);
    this.canvas.height = Math.round(h * ratio);
    return true;
  }

  /** S for the whole head, for the standard method's off-chip copy. */
  #allScores() {
    if (this.scores) return this.scores;
    const h = this.state.head, T = h.T;
    const S = new Float32Array(T * T);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let x = 0; x < h.s; x++) dot += h.Q[i * h.s + x] * h.K[j * h.s + x];
      S[i * T + j] = dot * h.scale;
    }
    return (this.scores = S);
  }

  // ---- drawing ----------------------------------------------------------------------------------------
  draw() {
    const { w, h, ratio } = this.size;
    const c = this.context;
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, w, h);
    if (!this.state) return;
    const style = getComputedStyle(this.canvas);
    const color = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    const K = this.colors = {
      well: color("--na-well", "#eee"), card: color("--anatomy-card", "#fff"), faint: color("--anatomy-faint", "#ddd"),
      lo: color("--anatomy-lo", "#bbb"), mid: color("--anatomy-mid", "#888"), text: color("--anatomy-text", "#555"),
      strong: color("--anatomy-strong", "#111"), muted: color("--anatomy-muted", "#777"), page: color("--anatomy-page", "#f2f2ef"),
      accent: color("--na-accent", "#ffb000"), blue: color("--anatomy-blue", "#3e63dd"), green: color("--anatomy-green", "#30a46c"),
      mono: color("--anatomy-mono", "monospace"),
    };
    this.inks = inksOf(this.canvas);
    this.hits = [];
    const pad = 14;
    const hbmW = Math.round(Math.min(380, Math.max(230, w * 0.3)));
    const busW = 66;
    const hbm = { x: pad, y: pad, w: hbmW, h: h - 2 * pad };
    const right = { x: pad + hbmW + busW, y: pad, w: w - pad * 2 - hbmW - busW, h: h - 2 * pad };
    const places = this.#drawHbm(c, hbm);
    const cards = this.#drawSms(c, right);
    const lower = { x: right.x, y: cards.bottom + 22, w: right.w, h: right.y + right.h - cards.bottom - 22 };
    const mapSize = Math.max(90, Math.min(170, lower.h - 30, lower.w * 0.24));
    this.#drawWorkMap(c, { x: lower.x, y: lower.y, size: mapSize });
    const sram = { x: lower.x + mapSize + 26, y: lower.y, w: lower.w - mapSize - 26, h: lower.h };
    const targets = this.#drawSram(c, sram);
    this.#drawBus(c, hbm, right, places, cards, targets);
    this.#drawHover(c);
  }

  #fit(c, text, room) {
    if (c.measureText(text).width <= room) return text;
    let t = text;
    while (t.length > 2 && c.measureText(`${t}…`).width > room) t = t.slice(0, -1);
    return `${t}…`;
  }

  #label(c, text, x, y, { size = 11, weight = "600", fill = this.colors.strong, align = "left" } = {}) {
    c.font = `${weight} ${size}px ${this.colors.mono}`;
    c.fillStyle = fill;
    c.textAlign = align;
    c.textBaseline = "alphabetic";
    c.fillText(text, x, y);
  }

  #heat(c, data, rows, cols, lv, x, y, w, hgt, key, positiveOnly = false) {
    // cache the image of a matrix by key
    let tile = key ? this.tiles.get(key) : null;
    if (!tile) {
      tile = document.createElement("canvas");
      tile.width = Math.max(1, cols);
      tile.height = Math.max(1, rows);
      const t = tile.getContext("2d");
      const image = t.createImageData(tile.width, tile.height);
      const inv = 1 / (lv || 1);
      for (let i = 0; i < rows * cols; i++) {
        const v = data[i];
        if (!Number.isFinite(v) || v === 0 || (positiveOnly && v < 0)) continue;
        const ink = v < 0 ? this.inks.red : this.inks.ink;
        image.data.set([ink[0], ink[1], ink[2], Math.round(255 * Math.pow(Math.min(1, Math.abs(v) * inv), GAMMA))], i * 4);
      }
      t.putImageData(image, 0, 0);
      if (key) this.tiles.set(key, tile);
    }
    c.imageSmoothingEnabled = false;
    c.drawImage(tile, x, y, w, hgt);
  }

  // ---- off-chip memory -----------------------------------------------------------------------------
  #drawHbm(c, box) {
    const { head, plan, tick, device, selected } = this.state;
    const K = this.colors, T = head.T, B = plan.B;
    c.fillStyle = K.well;
    c.fillRect(box.x, box.y, box.w, box.h);
    c.strokeStyle = K.lo;
    c.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
    this.#label(c, `${device.memName} · off-chip memory`, box.x + 10, box.y + 18, { size: 12 });
    this.#label(c, `${device.memory} · ${(device.bandwidth / 1e12).toFixed(2)} TB/s · big and slow`, box.x + 10, box.y + 33, { size: 10, weight: "400", fill: K.muted });
    const top = box.y + 54;
    const colsH = (box.h - 60) * 0.5;
    const gap = 10;
    const colW = Math.min(46, (box.w - 20 - gap * 3) / 4);
    const rowH = colsH / T;
    const places = {};
    // which rows of O are written by now
    const written = new Uint8Array(plan.Tr);
    for (let i = 0; i < plan.Tr; i++) {
      const done = plan.blocks[i].events.findIndex((e) => e.phase === "write" || e.phase === "k3w");
      written[i] = done >= 0 && tick >= done ? 1 : 0;
    }
    ["Q", "K", "V", "O"].forEach((name, k) => {
      const x = box.x + 10 + k * (colW + gap);
      this.#label(c, name, x + colW / 2, top - 4, { size: 11, align: "center" });
      c.fillStyle = K.card;
      c.fillRect(x, top, colW, colsH);
      const data = name === "O" ? head.out : head[name];
      this.#heat(c, data, T, head.s, this.levels[name], x, top, colW, colsH, `hbm-${name}`);
      if (name === "O") {
        // rows not yet written
        for (let i = 0; i < plan.Tr; i++) {
          if (written[i]) continue;
          const r0 = plan.blocks[i].rows.start, r1 = plan.blocks[i].rows.end;
          c.fillStyle = K.card;
          c.fillRect(x, top + r0 * rowH, colW, (r1 - r0) * rowH);
        }
      }
      // tile edges
      c.strokeStyle = K.lo;
      c.lineWidth = 1;
      for (let i = 1; i < plan.Tr; i++) { const y = Math.round(top + i * B * rowH) + 0.5; c.beginPath(); c.moveTo(x - 3, y); c.lineTo(x + colW + 3, y); c.stroke(); }
      c.strokeStyle = K.lo;
      c.strokeRect(x + 0.5, top + 0.5, colW - 1, colsH - 1);
      places[name] = { x, y: top, w: colW, rowH };
      this.hits.push({ x, y: top, w: colW, h: colsH, kind: "hbm", name, rows: T, cols: head.s, data: name === "O" ? head.out : head[name] });
    });
    // the T×T matrices: written out by the standard method, never by FlashAttention
    const sTop = top + colsH + 40;
    const sq = Math.max(40, Math.min((box.w - 20 - gap) / 2, box.y + box.h - sTop - 26));
    const flash = plan.method === "flash";
    [["S", "S = Q·Kᵀ/√d"], ["P", "P = softmax(S)"]].forEach(([name, title], k) => {
      const x = box.x + 10 + k * (sq + gap);
      this.#label(c, title, x, sTop - 6, { size: 10.5, fill: flash ? K.muted : K.strong });
      places[name] = { x, y: sTop, w: sq, cell: sq / T };
      if (flash) {
        c.setLineDash([4, 4]);
        c.strokeStyle = K.lo;
        c.strokeRect(x + 0.5, sTop + 0.5, sq - 1, sq - 1);
        c.setLineDash([]);
        c.font = `10px ${K.mono}`;
        c.fillStyle = K.muted;
        c.textAlign = "center";
        c.fillText("never", x + sq / 2, sTop + sq / 2 - 3);
        c.fillText("stored", x + sq / 2, sTop + sq / 2 + 10);
        return;
      }
      c.fillStyle = K.card;
      c.fillRect(x, sTop, sq, sq);
      const data = name === "S" ? this.#allScores() : head.shares;
      this.#heat(c, data, T, T, name === "S" ? this.levels.S : 1, x, sTop, sq, sq, `hbm-${name}`);
      // cover what has not been written yet
      const cell = sq / T;
      for (let i = 0; i < plan.Tr; i++) {
        const b = plan.blocks[i];
        let upTo; // columns written so far for this row tile
        if (name === "S") {
          const k1 = b.events.filter((e, t) => e.phase === "k1" && t <= tick).length;
          upTo = Math.min(T, k1 * B);
        } else {
          const k2 = b.events.findIndex((e) => e.phase === "k2");
          upTo = tick >= k2 ? T : 0;
        }
        if (upTo < T) {
          c.fillStyle = K.card;
          c.fillRect(x + upTo * cell, sTop + b.rows.start * cell, (T - upTo) * cell, (b.rows.end - b.rows.start) * cell);
        }
      }
      c.strokeStyle = K.lo;
      c.strokeRect(x + 0.5, sTop + 0.5, sq - 1, sq - 1);
      this.hits.push({ x, y: sTop, w: sq, h: sq, kind: "hbm", name, rows: T, cols: T, data });
    });
    // what is moving this tick: outline the tiles being read or written
    for (let i = 0; i < plan.Tr; i++) {
      const e = eventAt(plan, i, tick);
      const strong = i === selected;
      const mark = (place, r0, r1, c0 = null, c1 = null, write = false) => {
        if (!place) return;
        c.strokeStyle = write ? K.green : K.accent;
        c.lineWidth = strong ? 2.5 : 1.2;
        c.globalAlpha = strong ? 1 : 0.7;
        if (place.cell) c.strokeRect(place.x + (c0 ?? 0) * place.cell, place.y + r0 * place.cell, ((c1 ?? T) - (c0 ?? 0)) * place.cell, (r1 - r0) * place.cell);
        else c.strokeRect(place.x - 2, place.y + r0 * place.rowH, place.w + 4, Math.max(2, (r1 - r0) * place.rowH));
        c.globalAlpha = 1;
      };
      const rows = plan.blocks[i].rows;
      const tile = (j) => ({ start: j * B, end: Math.min(T, (j + 1) * B) });
      if (e.phase === "init") mark(places.Q, rows.start, rows.end);
      if (e.phase === "load") { const t = tile(e.j); mark(places.K, t.start, t.end); mark(places.V, t.start, t.end); }
      if (e.phase === "write" || e.phase === "k3w") mark(places.O, rows.start, rows.end, null, null, true);
      if (e.phase === "k1") { const t = tile(e.j); mark(places.Q, rows.start, rows.end); mark(places.K, t.start, t.end); mark(places.S, rows.start, rows.end, t.start, t.end, true); }
      if (e.phase === "k2") { mark(places.S, rows.start, rows.end); mark(places.P, rows.start, rows.end, 0, T, true); }
      if (e.phase === "k3") { const t = tile(e.j); mark(places.P, rows.start, rows.end, t.start, t.end); mark(places.V, t.start, t.end); }
    }
    return places;
  }

  // ---- the SMs ---------------------------------------------------------------------------------------
  #drawSms(c, box) {
    const { plan, tick, selected, device } = this.state;
    const K = this.colors;
    c.font = `600 11px ${K.mono}`;
    this.#label(c, this.#fit(c, `Streaming multiprocessors (SMs): ${device.sms} on this GPU, one thread block per tile of queries`, box.w), box.x, box.y + 10, { size: 11 });
    const cardW = 104, cardH = 66, gap = 8;
    const room = Math.max(1, Math.floor((box.w + gap) / (cardW + gap)));
    const shown = Math.min(plan.Tr, room - (plan.Tr < room ? 1 : 0));
    // the first blocks, and the followed one if it would not fit
    const order = Array.from({ length: shown }, (_, k) => k);
    if (selected >= shown && shown > 0) order[shown - 1] = selected;
    const cards = [];
    const y = box.y + 20;
    const phaseText = (e, i) => ({
      init: "load Q tile", load: `load K,V ${e.j}`, s: `S tile ${e.j}`, mask: "mask", max: "running max", p: "P and sum", o: "update acc",
      write: "write O", k1: `S tile ${e.j} → HBM`, k2: "softmax row", k3: `P·V tile ${e.j}`, k3w: "write O", done: "done",
    })[e.phase] ?? e.phase;
    for (let k = 0; k < shown; k++) {
      const i = order[k];
      const x = box.x + k * (cardW + gap);
      const e = eventAt(plan, i, tick);
      const done = e.phase === "done";
      c.fillStyle = K.card;
      c.fillRect(x, y, cardW, cardH);
      c.strokeStyle = i === selected ? K.accent : K.lo;
      c.lineWidth = i === selected ? 2 : 1;
      c.strokeRect(x + 0.5, y + 0.5, cardW - 1, cardH - 1);
      this.#label(c, `SM ${i}`, x + 8, y + 15, { size: 10, fill: K.muted, weight: "400" });
      this.#label(c, `block ${i}`, x + cardW - 8, y + 15, { size: 10, fill: K.muted, weight: "400", align: "right" });
      const rows = plan.blocks[i].rows;
      this.#label(c, `rows ${rows.start}–${rows.end - 1}`, x + 8, y + 31, { size: 10.5 });
      this.#label(c, phaseText(e, i), x + 8, y + 47, { size: 10.5, fill: done ? K.muted : K.accent });
      // progress through this block's events
      const n = plan.blocks[i].events.length;
      const f = Math.min(1, (tick + 1) / n);
      c.fillStyle = K.faint;
      c.fillRect(x + 8, y + cardH - 11, cardW - 16, 4);
      c.fillStyle = done ? K.green : K.accent;
      c.fillRect(x + 8, y + cardH - 11, (cardW - 16) * f, 4);
      cards[i] = { x, y, w: cardW, h: cardH };
      this.hits.push({ x, y, w: cardW, h: cardH, kind: "sm", i });
    }
    if (shown < plan.Tr || shown < room) {
      const x = box.x + shown * (cardW + gap);
      if (x + cardW <= box.x + box.w + 1) {
        c.setLineDash([3, 3]);
        c.strokeStyle = K.lo;
        c.lineWidth = 1;
        c.strokeRect(x + 0.5, y + 0.5, cardW - 1, cardH - 1);
        c.setLineDash([]);
        const more = plan.Tr - shown;
        const lines = more > 0 ? [`+${more} more`, "blocks"] : ["other heads,", "layers, batch"];
        lines.forEach((t, k) => this.#label(c, t, x + cardW / 2, y + 28 + k * 14, { size: 10, weight: "400", fill: K.muted, align: "center" }));
      }
    }
    return { cards, bottom: y + cardH };
  }

  // ---- which tiles of the T×T scores have been worked on ---------------------------------------------
  #drawWorkMap(c, { x, y, size }) {
    const { plan, tick, selected } = this.state;
    const K = this.colors, Tr = plan.Tr, cell = size / Tr;
    this.#label(c, "work map", x, y - 6, { size: 10.5 });
    for (let i = 0; i < Tr; i++) {
      const b = plan.blocks[i];
      for (let j = 0; j < Tr; j++) {
        const future = plan.causal && j > i;
        let state = "todo";
        const hitTick = b.events.findIndex((e) => (e.phase === "s" || e.phase === "k1") && e.j === j);
        if (plan.method === "flash" && future) state = "skip";
        else if (hitTick >= 0 && tick > hitTick) state = "done";
        else if (hitTick === tick) state = "now";
        const now = eventAt(plan, i, tick);
        if (plan.method === "flash" && now.j === j && ["load", "s", "mask", "max", "p", "o"].includes(now.phase)) state = "now";
        const px = x + j * cell, py = y + i * cell;
        c.fillStyle = state === "done" ? K.mid : state === "now" ? K.accent : K.card;
        c.globalAlpha = state === "done" && future ? 0.45 : 1;
        c.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        c.globalAlpha = 1;
        if (state === "skip" || (future && plan.method !== "flash")) {
          c.save();
          c.beginPath();
          c.rect(px + 1, py + 1, cell - 2, cell - 2);
          c.clip();
          c.strokeStyle = K.lo;
          c.beginPath();
          for (let s = -cell; s < cell; s += 5) { c.moveTo(px + s, py + cell); c.lineTo(px + s + cell, py); }
          c.stroke();
          c.restore();
        }
        c.strokeStyle = i === selected ? K.strong : K.faint;
        c.lineWidth = 1;
        c.strokeRect(px + 0.5, py + 0.5, cell - 1, cell - 1);
        this.hits.push({ x: px, y: py, w: cell, h: cell, kind: "map", i, j, state, future });
      }
    }
    c.font = `400 9.5px ${K.mono}`;
    const room = size + 14;
    this.#label(c, this.#fit(c, "rows: query tiles", room), x, y + size + 14, { size: 9.5, weight: "400", fill: K.muted });
    this.#label(c, this.#fit(c, "columns: key tiles", room), x, y + size + 27, { size: 9.5, weight: "400", fill: K.muted });
    const note = plan.method === "flash" ? (plan.causal ? "hatched: skipped" : "") : plan.causal ? "hatched: masked later" : "";
    if (note) this.#label(c, this.#fit(c, note, room), x, y + size + 40, { size: 9.5, weight: "400", fill: K.muted });
  }

  // ---- inside one SM ---------------------------------------------------------------------------------
  #drawSram(c, box) {
    const { sim, selected, device, plan } = this.state;
    const K = this.colors;
    c.fillStyle = K.page;
    c.fillRect(box.x - 10, box.y - 20, box.w + 10, box.h + 20);
    c.strokeStyle = K.blue;
    c.lineWidth = 1;
    c.strokeRect(box.x - 9.5, box.y - 19.5, box.w + 9, box.h + 19);
    this.#label(c, `SM ${selected} · on-chip SRAM`, box.x, box.y - 4, { size: 11, fill: K.blue });
    const kb = sim.bytes / 1024;
    this.#label(c, `uses ${kb < 10 ? kb.toFixed(1) : Math.round(kb)} of ${Math.round(device.sram / 1024)} KB · fast, tiny`, box.x + box.w - 6, box.y + box.h - 6, { size: 10, weight: "400", fill: K.muted, align: "right" });
    const inner = { x: box.x, y: box.y + 22, w: box.w - 10, h: box.h - 30 };
    const targets = {};
    const R = Math.max(...[...sim.buffers.a, ...sim.buffers.b].map((b) => b.rows));
    const rowGap = 34;
    const cellH = Math.max(2, Math.min(22, (inner.h - rowGap - 24) / 2 / Math.max(1, R)));
    const drawRow = (list, y) => {
      if (!list.length) return;
      // widths: heat maps share what is left after grids and vectors
      const gridCols = list.filter((b) => b.kind === "grid").reduce((s, b) => s + b.cols, 0);
      const vecs = list.filter((b) => b.kind === "vec").length;
      const heats = list.filter((b) => b.kind === "heat").length;
      const gaps = (list.length - 1) * 16;
      const vecW = cellH >= 14 ? 46 : 26;
      const cellW = gridCols ? Math.max(2, Math.min(46, Math.max(cellH, 30), (inner.w - gaps - vecs * vecW - heats * 70) / gridCols)) : 0;
      const heatW = heats ? Math.max(36, Math.min(150, (inner.w - gaps - vecs * vecW - gridCols * cellW) / heats)) : 0;
      let x = inner.x;
      for (const b of list) {
        const wdt = b.kind === "heat" ? heatW : b.kind === "vec" ? vecW : b.cols * cellW;
        c.font = `600 10px ${K.mono}`;
        const room = b.kind === "heat" ? wdt : Math.max(wdt, c.measureText(b.label).width + 4);
        b.shown = this.#fit(c, b.label, room + 8);
        const hgt = b.rows * cellH;
        const isFresh = sim.fresh.has(b.id);
        this.#label(c, b.shown, x, y - 6, { size: 10, fill: isFresh ? K.accent : K.text, weight: isFresh ? "600" : "400" });
        c.fillStyle = K.card;
        c.fillRect(x, y, wdt, hgt);
        if (b.data) {
          if (b.kind === "heat") this.#heat(c, b.data, b.rows, b.cols, this.levels[b.matrix] ?? level(b.data), x, y, wdt, hgt, null);
          else this.#cells(c, b, x, y, b.kind === "vec" ? wdt : cellW, cellH);
        } else {
          c.font = `10px ${K.mono}`;
          c.fillStyle = K.muted;
          c.textAlign = "center";
          c.fillText("empty", x + wdt / 2, y + hgt / 2 + 3);
        }
        c.strokeStyle = isFresh ? K.accent : K.lo;
        c.lineWidth = isFresh ? 2 : 1;
        c.strokeRect(x + 0.5, y + 0.5, wdt - 1, hgt - 1);
        targets[b.id] = { x, y, w: wdt, h: hgt };
        if (b.data) this.hits.push({ x, y, w: wdt, h: hgt, kind: "buf", buffer: b, cellW: b.kind === "vec" ? wdt : b.kind === "grid" ? cellW : wdt / b.cols, cellH });
        x += room + 16;
      }
    };
    drawRow(sim.buffers.a, inner.y + 6);
    const aH = Math.max(...sim.buffers.a.map((b) => b.rows), 0) * cellH;
    drawRow(sim.buffers.b, inner.y + 6 + (sim.buffers.a.length ? aH + rowGap : 0));
    return targets;
  }

  #cells(c, b, x, y, cw, ch) {
    const K = this.colors;
    const cols = b.kind === "vec" ? 1 : b.cols;
    const isP = b.id === "P";
    const lv = isP ? 1 : b.id === "S" ? this.levels.S : level(b.data);
    for (let r = 0; r < b.rows; r++) for (let col = 0; col < cols; col++) {
      const v = b.data[r * cols + col];
      const px = x + col * cw, py = y + r * ch;
      if (v === -Infinity || (b.maskFrom && b.colStart + col > b.rowStart + r)) {
        c.fillStyle = K.faint;
        c.fillRect(px, py, cw, ch);
      } else if (Number.isFinite(v) && v !== 0) {
        const ink = v < 0 ? this.inks.red : this.inks.ink;
        c.fillStyle = `rgba(${ink[0]},${ink[1]},${ink[2]},${Math.pow(Math.min(1, Math.abs(v) / lv), GAMMA).toFixed(3)})`;
        c.fillRect(px, py, cw, ch);
      }
      if (ch >= 13 && cw >= 30) {
        const strong = Number.isFinite(v) && Math.pow(Math.min(1, Math.abs(v) / lv), GAMMA) > 0.55;
        c.font = `${Math.min(11, ch * 0.62, cw / 4)}px ${K.mono}`;
        c.fillStyle = strong ? K.card : K.strong;
        c.textAlign = "center";
        c.textBaseline = "middle";
        c.fillText(fmt(v), px + cw / 2, py + ch / 2 + 0.5);
        c.textBaseline = "alphabetic";
      }
    }
    if (cw >= 6 && ch >= 6 && b.kind === "grid") {
      c.strokeStyle = K.faint;
      c.lineWidth = 1;
      c.beginPath();
      for (let col = 1; col < cols; col++) { c.moveTo(Math.round(x + col * cw) + 0.5, y); c.lineTo(Math.round(x + col * cw) + 0.5, y + b.rows * ch); }
      for (let r = 1; r < b.rows; r++) { c.moveTo(x, Math.round(y + r * ch) + 0.5); c.lineTo(x + cols * cw, Math.round(y + r * ch) + 0.5); }
      c.stroke();
    }
  }

  // ---- the bus: tiles moving between the two memories ---------------------------------------------------
  #drawBus(c, hbm, right, places, cards, targets) {
    const { plan, tick, selected, device } = this.state;
    const K = this.colors;
    const x0 = hbm.x + hbm.w, x1 = right.x;
    const mid = (x0 + x1) / 2;
    c.fillStyle = K.faint;
    c.fillRect(mid - 5, hbm.y + 40, 10, hbm.h - 60);
    c.save();
    c.translate(mid + 18, hbm.y + hbm.h / 2);
    c.rotate(-Math.PI / 2);
    this.#label(c, `bus · ${(device.bandwidth / 1e12).toFixed(2)} TB/s`, 0, 0, { size: 10, weight: "400", fill: K.muted, align: "center" });
    c.restore();
    const T = plan.T, B = plan.B;
    const tile = (j) => ({ start: j * B, end: Math.min(T, (j + 1) * B) });
    const flows = [];
    for (let i = 0; i < plan.Tr; i++) {
      const e = eventAt(plan, i, tick);
      const card = cards.cards[i];
      const strong = i === selected;
      const to = (name, rows, cols) => {
        const p = places[name];
        if (!p) return null;
        if (p.cell) return { x: p.x + ((cols?.[0] ?? 0) + (cols?.[1] ?? T)) / 2 * p.cell, y: p.y + (rows[0] + rows[1]) / 2 * p.cell, right: p.x + p.w };
        return { x: p.x + p.w, y: p.y + (rows[0] + rows[1]) / 2 * p.rowH, right: p.x + p.w };
      };
      const end = (id) => (strong && targets[id] ? { x: targets[id].x, y: targets[id].y + Math.min(14, targets[id].h / 2) } : card ? { x: card.x, y: card.y + card.h / 2 } : null);
      const rows = plan.blocks[i].rows;
      const add = (from, target, write = false) => { if (from && target) flows.push({ from, target, write, strong }); };
      if (e.phase === "init") add(to("Q", [rows.start, rows.end]), end("q"));
      if (e.phase === "load") { const t = tile(e.j); add(to("K", [t.start, t.end]), end("k")); add(to("V", [t.start, t.end]), end("v")); }
      if (e.phase === "write" || e.phase === "k3w") add(to("O", [rows.start, rows.end]), end("O"), true);
      if (e.phase === "k1") { const t = tile(e.j); add(to("Q", [rows.start, rows.end]), end("q")); add(to("K", [t.start, t.end]), end("k")); add(to("S", [rows.start, rows.end], [t.start, t.end]), end("S"), true); }
      if (e.phase === "k2") { add(to("S", [rows.start, rows.end]), end("S")); add(to("P", [rows.start, rows.end]), end("P"), true); }
      if (e.phase === "k3") { const t = tile(e.j); add(to("P", [rows.start, rows.end], [t.start, t.end]), end("P")); add(to("V", [t.start, t.end]), end("v")); }
    }
    for (const f of flows) {
      const a = { x: Math.max(f.from.x, f.from.right ?? f.from.x), y: f.from.y }, b = f.target;
      const ax = f.from.right ?? a.x;
      c.strokeStyle = f.write ? K.green : K.accent;
      c.globalAlpha = f.strong ? 0.9 : 0.35;
      c.lineWidth = f.strong ? 2 : 1;
      c.beginPath();
      c.moveTo(ax, a.y);
      c.bezierCurveTo(mid, a.y, mid, b.y, b.x, b.y);
      c.stroke();
      // the tile in flight
      const t = f.write ? 1 - this.anim : this.anim;
      const u = 1 - t;
      const px = u * u * u * ax + 3 * u * u * t * mid + 3 * u * t * t * mid + t * t * t * b.x;
      const py = u * u * u * a.y + 3 * u * u * t * a.y + 3 * u * t * t * b.y + t * t * t * b.y;
      if (this.anim < 1) {
        c.globalAlpha = 1;
        c.fillStyle = f.write ? K.green : K.accent;
        c.fillRect(px - (f.strong ? 6 : 4), py - (f.strong ? 4 : 3), f.strong ? 12 : 8, f.strong ? 8 : 6);
      }
      c.globalAlpha = 1;
    }
  }

  #drawHover(c) {
    if (!this.hover) return;
    const K = this.colors, h = this.hover;
    c.strokeStyle = K.accent;
    c.lineWidth = 1.5;
    if (h.cell) c.strokeRect(h.cell.x + 0.5, h.cell.y + 0.5, Math.max(3, h.cell.w), Math.max(3, h.cell.h));
  }

  // ---- pointer -----------------------------------------------------------------------------------------------
  #hit(x, y) {
    for (let k = this.hits.length - 1; k >= 0; k--) {
      const hit = this.hits[k];
      if (x >= hit.x && x < hit.x + hit.w && y >= hit.y && y < hit.y + hit.h) return { hit, x, y };
    }
    return null;
  }

  describe(found) {
    if (!found) return { text: "" };
    const { hit, x, y } = found;
    const head = this.state.head;
    const who = (r) => `token ${r + 1} “${String(head.pieces[r] ?? "").replace(/^ /, "·").replace(/\n/g, "↵")}”`;
    if (hit.kind === "sm") return { text: `SM ${hit.i} runs thread block ${hit.i}: queries ${this.state.plan.blocks[hit.i].rows.start}–${this.state.plan.blocks[hit.i].rows.end - 1}. Click to look inside it.` };
    if (hit.kind === "map") {
      const txt = hit.state === "skip" ? "never computed: every key here comes after every query in the tile, so the mask would zero it all" : hit.future ? "computed by the standard kernels, then thrown away by the mask" : hit.state === "done" ? "done" : hit.state === "now" ? "being worked on now" : "still to do";
      return { text: `Query tile ${hit.i} × key tile ${hit.j}: ${txt}. Click to follow block ${hit.i}.` };
    }
    if (hit.kind === "hbm") {
      const r = Math.min(hit.rows - 1, Math.floor(((y - hit.y) / hit.h) * hit.rows));
      const col = Math.min(hit.cols - 1, Math.floor(((x - hit.x) / hit.w) * hit.cols));
      const v = hit.data[r * hit.cols + col];
      const what = hit.name === "S" || hit.name === "P" ? `${who(r)} → ${who(col)}` : `${who(r)}, number ${col}`;
      return { text: `${hit.name} in off-chip memory, ${what}: ${signed(v, 5)}`, cell: { x: hit.x + (col * hit.w) / hit.cols, y: hit.y + (r * hit.h) / hit.rows, w: hit.w / hit.cols, h: hit.h / hit.rows } };
    }
    if (hit.kind === "buf") {
      const b = hit.buffer;
      const cols = b.kind === "vec" ? 1 : b.cols;
      const r = Math.min(b.rows - 1, Math.floor((y - hit.y) / hit.cellH));
      const col = Math.min(cols - 1, Math.floor((x - hit.x) / hit.cellW));
      const v = b.data[r * cols + col];
      const row = (b.rowStart ?? 0) + r;
      let what = `${b.label}, ${who(row)}`;
      if (b.kind === "grid") what += ` → ${who((b.colStart ?? 0) + col)}`;
      else if (b.kind === "heat") what += `, number ${col}`;
      let extra = "";
      if (b.kind === "vec" && b.old) extra = ` (was ${fmt(b.old[r])})`;
      return { text: `${what}: ${v === -Infinity ? "−∞ (masked)" : signed(v, 5)}${extra}`, cell: { x: hit.x + col * hit.cellW, y: hit.y + r * hit.cellH, w: hit.cellW, h: hit.cellH } };
    }
    return { text: "" };
  }

  #listen() {
    const canvas = this.canvas;
    const local = (event) => { const box = canvas.getBoundingClientRect(); return [event.clientX - box.left, event.clientY - box.top]; };
    canvas.addEventListener("pointermove", (event) => {
      if (!this.state) return;
      const d = this.describe(this.#hit(...local(event)));
      this.hover = d.cell ? d : null;
      this.onHover(d.text);
      this.draw();
    });
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.onHover(""); this.draw(); });
    canvas.addEventListener("click", (event) => {
      const found = this.#hit(...local(event));
      if (!found) return;
      if (found.hit.kind === "sm" || found.hit.kind === "map") this.onPick({ block: found.hit.i });
    });
  }
}
