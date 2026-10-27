// The cabinet, the model as a machine: behaviour.
// Fills the plates from the trace, slides modules, moves the read head, and
// answers the pointer. One loop, writing attributes on elements it holds.

import * as k from "../../../vendor/anatomy/iso-kit.mjs";
import { Plate, RowScales, inkScale, percent, signed } from "../../ui/plates.js";
import { Stage } from "../../ui/stage.js";
import { TAPE, drawTape as fillTape, showPiece } from "../parts.js";
import { PULL, buildRack } from "./geometry.js";
import { createBench } from "./bench/bench.js";
import { short, unitParams } from "./bench/dims.js";

const still = typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };


export function mountRack(stageEl, app, card) {
  const { spec } = app.state;
  const fig = buildRack(spec, { title: app.state.card.name.toUpperCase() });
  stageEl.innerHTML = fig.svg;
  const svg = stageEl.querySelector("svg");
  const P = fig.P;
  const units = spec.units;
  const stage = new Stage(stageEl, svg, { width: fig.width, height: fig.height, onChange: (zoom) => card.setZoom(zoom) });

  // ---- elements --------------------------------------------------------------
  const defs = new Map(fig.plates.map((def) => [def.id, def]));
  const plateEls = new Map();
  const plates = new Map();
  for (const group of svg.querySelectorAll("[data-plate]")) {
    const def = defs.get(group.dataset.plate);
    plateEls.set(def.id, group);
    const image = group.querySelector("image.na-cells");
    if (image) plates.set(def.id, new Plate(image, def.cols, def.rows));
  }
  const byUnit = units.map((_, i) => fig.plates.filter((def) => def.unit === i));
  const modules = [...svg.querySelectorAll(".na-module")].sort((a, b) => a.dataset.unit - b.dataset.unit);
  const lamps = [...svg.querySelectorAll("[data-lamp]")].sort((a, b) => a.dataset.lamp - b.dataset.lamp);
  const tiles = [...svg.querySelectorAll(".na-tile")];
  const head = svg.querySelector('[data-part="head"]');
  const paper = svg.querySelector(".na-paper-text");
  const dial = svg.querySelector('[data-part="dial"]');
  const candidates = [...svg.querySelectorAll(".na-candidate")];

  // ---- the bench: a plate, opened into its arithmetic ------------------------------
  const bench = createBench(app, { onBack: () => closeBench() });
  stageEl.after(bench.element);
  const plateRoot = stageEl.closest(".iso-plate");

  /** Which step of the unit a plate belongs to, and which output to start on. */
  function stepFor(def, cell) {
    const unit = units[def.unit];
    const index = cell ? cell.row * def.cols + cell.col : -1;
    if (unit.kind === "embed") {
      if (def.kind === "weights") return { step: /pos/i.test(def.id) ? "pos" : "tok" };
      if (def.kind === "vector") return { step: def.id.endsWith("/pos") ? "pos" : "tok", column: index };
      return { step: "pos", column: index };
    }
    if (unit.kind === "unembed") {
      const prediction = app.state.trace?.predictions[app.state.position];
      return { step: "logits", column: def.kind === "candidates" && cell ? prediction?.ids[cell.index] ?? -1 : -1 };
    }
    if (def.kind === "attention") return { step: "scores", head: cell ? cell.row : 0 };
    if (def.kind === "neurons") return { step: "up", column: index };
    if (def.kind === "vector") return def.id.endsWith("/attn") ? { step: "wo", column: index } : { step: "down", column: index };
    if (def.kind === "stream") return { step: "down", column: index };
    if (def.kind === "weights") {
      const id = def.id.split("/")[1];
      if (/^(W_Q|W_K|W_V|b_QKV|ln1)/.test(id)) return { step: "qkv" };
      if (/^(W_O|b_O)/.test(id)) return { step: "wo" };
      if (/^(W_in|b_in|ln2)/.test(id)) return { step: "up" };
      return { step: "down" };
    }
    return { step: "qkv" };
  }

  let benchBusy = false;
  async function openBench(def, cell, target) {
    if (!app.state.info || !app.state.tokens.ids.length || benchBusy) return;
    if (!app.state.trace?.has(units[def.unit].stream.trace, app.state.position)) return;
    benchBusy = true;
    const { step, column = -1, head = 0 } = stepFor(def, cell);
    hover = "";
    // fly into the plate, then hand over to the bench
    if (target) await stage.flyTo(stage.rectOf(target), 520);
    plateRoot.toggleAttribute("data-bench", true);
    document.getElementById("app")?.toggleAttribute("data-bench-open", true); // the bench takes the full width
    stageEl.hidden = true;
    bench.open(def.unit, step, { column, head });
    card.say(`${units[def.unit].title.toLowerCase()} · on the bench`);
    benchBusy = false;
  }
  function closeBench() {
    bench.close();
    plateRoot.removeAttribute("data-bench");
    document.getElementById("app")?.removeAttribute("data-bench-open");
    stageEl.hidden = false;
    stage.flyTo(null, 520);
    say();
  }

  // ---- painting ----------------------------------------------------------------
  let inks = app.inks();
  const scales = new RowScales();
  const attention = new Map(); // plate id -> scratch buffer for the token window

  function paintPlate(def) {
    const plate = plates.get(def.id);
    const { trace, position, window } = app.state;
    if (def.kind === "weights") return;
    if (def.kind === "candidates") return paintCandidates();
    const row = trace?.row(def.trace, position);
    if (!row) return plate.clear();
    if (def.kind === "attention") {
      const C = row.length / def.rows;
      let buffer = attention.get(def.id);
      if (!buffer) attention.set(def.id, (buffer = new Float32Array(def.rows * def.cols)));
      for (let h = 0; h < def.rows; h++) for (let c = 0; c < def.cols; c++) buffer[h * def.cols + c] = window.start + c < C ? row[h * C + window.start + c] : 0;
      plate.paint(buffer, { scale: 1, inks, gamma: 0.5 });
    } else {
      // a neuron plate inks only what fired: GELU lets the rest dip no lower than -0.17
      plate.paint(row, { scale: scales.of(trace, def.trace, position, row, def.kind), inks, count: def.size, positiveOnly: def.kind === "neurons" });
    }
  }

  function paintCandidates() {
    const { trace, position } = app.state;
    const prediction = trace?.predictions[position];
    candidates.forEach((group, index) => {
      const text = group.querySelector("text"), bar = group.querySelector(".na-bar");
      if (!prediction || index >= prediction.ids.length) {
        text.textContent = "";
        bar.setAttribute("width", "0");
        group.removeAttribute("data-top");
        return;
      }
      text.textContent = showPiece(prediction.pieces[index]);
      bar.setAttribute("width", (38.4 * Math.sqrt(prediction.probs[index])).toFixed(2));
      group.toggleAttribute("data-top", index === 0);
    });
  }

  const ready = (i) => Boolean(app.state.trace?.has(units[i].stream.trace, app.state.position));
  function paintUnit(i) {
    for (const def of byUnit[i]) paintPlate(def);
    lamps[i]?.setAttribute("data-state", ready(i) ? "on" : "off");
  }
  function paintAll() {
    queue.length = 0;
    for (let i = 0; i < units.length; i++) paintUnit(i);
  }

  // Units light up one after another as the engine finishes them. The engine is
  // usually faster than is comfortable to watch, so the sweep is paced.
  const queue = [];
  let pacer = 0;
  const STEP_MS = 55;
  function drain() {
    pacer = 0;
    if (!queue.length) return;
    const count = still.matches ? queue.length : 1;
    for (let n = 0; n < count; n++) {
      const i = queue.shift();
      paintUnit(i);
      lamps[i]?.setAttribute("data-state", "live");
      const done = i;
      setTimeout(() => { if (lamps[done]?.getAttribute("data-state") === "live") lamps[done].setAttribute("data-state", "on"); }, 420);
    }
    if (queue.length) pacer = setTimeout(drain, STEP_MS);
    say();
  }

  // ---- weights on the decks -----------------------------------------------------
  const deckLoaded = new Set();
  async function loadDeck(i) {
    if (deckLoaded.has(i) || !app.state.info) return;
    deckLoaded.add(i);
    const stamp = app.state.weightsStamp;
    for (const def of byUnit[i]) {
      if (def.kind !== "weights") continue;
      try {
        const { data } = await app.engine.call("region", { view: def.view, rows: def.rows, cols: def.cols });
        if (stamp !== app.state.weightsStamp) return;
        def.scale = inkScale(data, { q: 0.98 });
        plates.get(def.id).paint(data, { scale: def.scale, inks });
      } catch (error) {
        deckLoaded.delete(i);
        throw error;
      }
    }
  }
  function forgetDecks() {
    deckLoaded.clear();
    for (const def of fig.plates) if (def.kind === "weights") plates.get(def.id).clear();
  }

  // ---- tape, head, paper, dial ----------------------------------------------------
  const drawTape = () => fillTape(tiles, app.state);
  function drawPaper() {
    const text = app.state.tokens.pieces.join("").replace(/\s+/g, " ");
    paper.textContent = text.slice(-44);
  }
  function drawDial() {
    dial.setAttribute("d", fig.dialPointer(Math.min(1, app.state.temperature / 1.5)));
  }

  // ---- motion ------------------------------------------------------------------------
  const pull = units.map(() => ({ at: 0, rate: 0, target: 0 }));
  const headAt = { x: 0, target: 0 };
  let frame = 0, last = 0;
  function tick(now) {
    const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
    last = now;
    const calm = still.matches;
    let moving = false;
    pull.forEach((p, i) => {
      if (p.at === p.target && p.rate === 0) return;
      if (calm) { p.at = p.target; p.rate = 0; }
      else {
        const stiffness = 150, damping = 2 * Math.sqrt(stiffness) * 0.92;
        for (let s = 0; s < 4; s++) {
          p.rate += ((p.target - p.at) * stiffness - p.rate * damping) * (dt / 4);
          p.at += p.rate * (dt / 4);
        }
        if (Math.abs(p.target - p.at) < 0.02 && Math.abs(p.rate) < 0.05) { p.at = p.target; p.rate = 0; }
        else moving = true;
      }
      modules[i].setAttribute("transform", k.translateAlong(P, [0, p.at, 0]));
      modules[i].toggleAttribute("data-out", p.at > 1);
    });
    if (headAt.x !== headAt.target) {
      headAt.x = calm ? headAt.target : headAt.x + (headAt.target - headAt.x) * (1 - Math.exp(-dt / 0.11));
      if (Math.abs(headAt.target - headAt.x) < 0.02) headAt.x = headAt.target;
      else moving = true;
      head.setAttribute("transform", k.translateAlong(P, [headAt.x, 0, 0]));
    }
    frame = moving ? requestAnimationFrame(tick) : 0;
    if (!moving) say();
  }
  const run = () => { if (!frame) { last = 0; frame = requestAnimationFrame(tick); } };

  function placeHead() {
    const { position, window, tokens } = app.state;
    head.toggleAttribute("data-idle", !tokens.ids.length);
    headAt.target = Math.max(0, Math.min(TAPE - 1, position - window.start)) * fig.tilePitch;
    run();
  }
  function setOpen() {
    const open = app.state.open;
    pull.forEach((p, i) => { p.target = i === open ? PULL : 0; });
    if (open >= 0) loadDeck(open).catch((error) => app.fail(error));
    run();
  }

  // ---- readout -----------------------------------------------------------------------------
  let hover = "";
  function say() {
    card.say(hover || idleText());
  }
  function idleText() {
    const { info, trace, tokens, position, running, open } = app.state;
    if (!info) return "cold · no weights loaded";
    if (!tokens.ids.length) return `${units.length - 2} blocks · waiting for a prompt`;
    if (running && queue.length) return `running · unit ${units.length - queue.length} of ${units.length}`;
    if (open >= 0 && pull[open].at !== pull[open].target) return "sliding out";
    if (open >= 0) return `${units[open].title.toLowerCase()} · ${byUnit[open].filter((d) => d.kind === "weights").length} weight plates · ${short(unitParams(app.state.spec, units[open]).total)} parameters`;
    const prediction = trace?.predictions[position];
    const here = `token ${position + 1} of ${tokens.ids.length} “${showPiece(tokens.pieces[position] ?? "")}”`;
    return prediction ? `${here} → “${showPiece(prediction.pieces[0])}” ${percent(prediction.probs[0])}` : here;
  }

  // ---- pointer ---------------------------------------------------------------------------------
  let cursorEl = null;
  const point = svg.createSVGPoint();
  function cellOf(group, def, event) {
    const image = group.querySelector("image.na-cells");
    const matrix = image.getScreenCTM();
    if (!matrix) return null;
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(matrix.inverse());
    const col = Math.floor((local.x / def.w) * def.cols), row = Math.floor((local.y / def.h) * def.rows);
    if (col < 0 || row < 0 || col >= def.cols || row >= def.rows) return null;
    return { col, row };
  }
  function describe(def, cell, plate) {
    const unit = units[def.unit];
    const { tokens, window } = app.state;
    const value = plate?.valueAt(cell.col, cell.row);
    const index = cell.row * def.cols + cell.col;
    if (def.kind === "neurons") return `${unit.title.toLowerCase()} · neuron ${index} · ${signed(value)} · click to see how it is computed`;
    if (def.kind === "attention") {
      const source = window.start + cell.col;
      const piece = source < tokens.ids.length ? ` “${showPiece(tokens.pieces[source])}”` : "";
      if (source > app.state.position) return `${unit.title.toLowerCase()} · head ${cell.row} · token ${source + 1} comes later, so this position cannot see it`;
      return `${unit.title.toLowerCase()} · head ${cell.row} → token ${source + 1}${piece} · ${value === undefined ? "—" : value.toFixed(3)} · click to see the scores`;
    }
    if (def.kind === "stream") return `stream after ${unit.title.toLowerCase()} · dim ${index} · ${signed(value)}`;
    if (def.kind === "vector") return `${unit.title.toLowerCase()} · ${def.label} · dim ${index} · ${signed(value)}`;
    if (def.kind === "weights") return `${unit.title.toLowerCase()} · ${def.label} · ${def.view.rows.toLocaleString()} × ${def.view.cols.toLocaleString()} · click to see it multiply`;
    return "";
  }
  function onMove(event) {
    const group = event.target.closest?.("[data-plate]");
    const moduleEl = event.target.closest?.(".na-module");
    const unit = moduleEl ? Number(moduleEl.dataset.unit) : group ? Number(group.dataset.unit) : -1;
    modules.forEach((el, i) => el.toggleAttribute("data-on", i === unit && Boolean(moduleEl)));
    if (cursorEl) { cursorEl.setAttribute("visibility", "hidden"); cursorEl = null; }
    hover = "";
    if (group) {
      const def = defs.get(group.dataset.plate);
      if (def.kind === "candidates") {
        const cand = event.target.closest(".na-candidate");
        const prediction = app.state.trace?.predictions[app.state.position];
        const index = cand ? Number(cand.dataset.candidate) : -1;
        if (prediction && index >= 0 && index < prediction.ids.length) hover = `candidate ${index + 1} · “${showPiece(prediction.pieces[index])}” · ${percent(prediction.probs[index])}`;
      } else {
        const cell = cellOf(group, def, event);
        if (cell) {
          hover = describe(def, cell, plates.get(def.id));
          const cursor = group.querySelector(".na-cursor");
          if (cursor) {
            cursor.setAttribute("x", ((cell.col * def.w) / def.cols).toFixed(3));
            cursor.setAttribute("y", ((cell.row * def.h) / def.rows).toFixed(3));
            cursor.setAttribute("visibility", "visible");
            cursorEl = cursor;
          }
        }
      }
    } else {
      const tile = event.target.closest?.(".na-tile");
      if (tile && !tile.hasAttribute("data-empty")) {
        const index = app.state.window.start + Number(tile.dataset.tile);
        const prediction = app.state.trace?.predictions[index - 1];
        const surprise = prediction && Number.isFinite(prediction.targetProb) ? ` · the model gave it ${percent(prediction.targetProb)}` : "";
        hover = `token ${index + 1} · id ${app.state.tokens.ids[index]} · “${showPiece(app.state.tokens.pieces[index])}”${surprise}`;
      }
    }
    say();
  }
  function onLeave() {
    modules.forEach((el) => el.removeAttribute("data-on"));
    if (cursorEl) { cursorEl.setAttribute("visibility", "hidden"); cursorEl = null; }
    hover = "";
    say();
  }
  function onClick(event) {
    const tile = event.target.closest?.(".na-tile");
    if (tile && !tile.hasAttribute("data-empty")) return app.setPosition(app.state.window.start + Number(tile.dataset.tile));
    const group = event.target.closest?.("[data-plate]");
    if (group) {
      const def = defs.get(group.dataset.plate);
      if (def.kind === "candidates") {
        const cand = event.target.closest(".na-candidate");
        return openBench(def, cand ? { index: Number(cand.dataset.candidate) } : null, group);
      }
      if (def.kind !== "weights" || app.state.open === def.unit) return openBench(def, cellOf(group, def, event), group);
    }
    const moduleEl = event.target.closest?.(".na-module");
    if (moduleEl) app.toggleUnit(Number(moduleEl.dataset.unit));
  }
  function onKey(event) {
    if (bench.isOpen) { if (event.key === "Escape") closeBench(); return; }
    const { position, tokens, open } = app.state;
    const last = tokens.ids.length - 1;
    let handled = true;
    if (event.key === "ArrowRight") app.setPosition(Math.min(last, position + (event.shiftKey ? 8 : 1)));
    else if (event.key === "ArrowLeft") app.setPosition(Math.max(0, position - (event.shiftKey ? 8 : 1)));
    else if (event.key === "Home") app.setPosition(0);
    else if (event.key === "End") app.setPosition(last);
    else if (event.key === "ArrowUp") app.toggleUnit(Math.min(units.length - 1, (open < 0 ? -1 : open) + 1), true);
    else if (event.key === "ArrowDown") app.toggleUnit(Math.max(0, (open < 0 ? units.length : open) - 1), true);
    else if (event.key === "Escape") app.toggleUnit(-1, true);
    else if (event.key === "+" || event.key === "=") stage.zoomAt(1.5);
    else if (event.key === "-" || event.key === "_") stage.zoomAt(1 / 1.5);
    else if (event.key === "0") stage.fit();
    else handled = false;
    if (handled) event.preventDefault();
  }
  stageEl.addEventListener("pointermove", onMove);
  stageEl.addEventListener("pointerleave", onLeave);
  stageEl.addEventListener("click", onClick);
  stageEl.addEventListener("keydown", onKey);

  // ---- subscriptions -------------------------------------------------------------------------------
  const off = [
    app.on("trace", ({ unit, p0 }) => {
      if (app.state.position < p0) return;
      queue.push(unit);
      if (!pacer) pacer = setTimeout(drain, queue.length > 1 ? STEP_MS : 0);
    }),
    app.on("run", ({ reset }) => {
      queue.length = 0;
      clearTimeout(pacer);
      pacer = 0;
      if (reset) scales.clear();
      paintAll();
      drawTape();
      drawPaper();
      placeHead();
      say();
    }),
    app.on("position", () => { paintAll(); drawTape(); placeHead(); say(); if (bench.isOpen) bench.refresh(); }),
    app.on("tokens", () => { drawTape(); drawPaper(); placeHead(); say(); }),
    app.on("prediction", ({ position }) => { if (position === app.state.position) { paintCandidates(); say(); } }),
    app.on("done", () => { say(); if (bench.isOpen) bench.refresh(); }),
    app.on("open", () => { setOpen(); say(); }),
    app.on("weights", () => { forgetDecks(); scales.clear(); paintAll(); setOpen(); say(); if (bench.isOpen && !app.state.info) closeBench(); }),
    app.on("settings", () => { drawDial(); }),
    app.on("theme", () => {
      bench.redraw();
      inks = app.inks();
      paintAll();
      for (const def of fig.plates) if (def.kind === "weights") plates.get(def.id).clear();
      deckLoaded.clear();
      if (app.state.open >= 0) loadDeck(app.state.open).catch((error) => app.fail(error));
    }),
  ];

  drawTape();
  drawPaper();
  drawDial();
  placeHead();
  paintAll();
  setOpen();
  say();

  return {
    stage,
    focusUnit(i) {
      const box = fig.boxOf(i, app.state.open === i);
      stage.focus(box.x, box.y, box.w, box.h);
    },
    destroy() {
      off.forEach((stop) => stop());
      clearTimeout(pacer);
      cancelAnimationFrame(frame);
      stageEl.removeEventListener("pointermove", onMove);
      stageEl.removeEventListener("pointerleave", onLeave);
      stageEl.removeEventListener("click", onClick);
      stageEl.removeEventListener("keydown", onKey);
      stageEl.innerHTML = "";
      bench.close();
      bench.element.remove();
      plateRoot?.removeAttribute("data-bench");
      document.getElementById("app")?.removeAttribute("data-bench-open");
      stageEl.hidden = false;
    },
  };
}
