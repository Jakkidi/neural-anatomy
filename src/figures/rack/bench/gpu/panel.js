// The bench's "On a GPU" mode: one attention head of the chosen block, run as
// a GPU runs it, standard attention or FlashAttention, step by step, with the
// kernel's code beside it and the memory traffic counted.
//
// It needs only what `headTensors` gives (Q, K, V of a head, causal or not),
// so any architecture whose spec describes its attention gets it.

import { headTensors } from "../heads.js";
import { rowsReady } from "../sequence.js";
import { DEVICES, deviceById } from "./devices.js";
import { KERNELS, displayTile, eventAt, flashOutput, plan, simulate, trafficAt } from "./flash.js";
import { GpuView } from "./gpu-view.js";
import { headDims } from "../dims.js";

const esc = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const f = (v, digits = 3) => (v === -Infinity ? "−∞" : Number.isFinite(v) ? v.toFixed(digits).replace("-", "−") : "…");
const bytes = (n) => (n >= 1e12 ? `${(n / 1e12).toFixed(2)} TB` : n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 0 : n >= 1e10 ? 1 : 2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)} KB` : `${Math.round(n)} B`);
const seconds = (s) => (s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(s >= 0.1 ? 0 : s >= 0.01 ? 1 : 2)} ms` : s >= 1e-6 ? `${(s * 1e6).toFixed(s >= 1e-4 ? 0 : 1)} µs` : `${Math.round(s * 1e9)} ns`);
const REAL_TILE = 128; // rows per tile in FlashAttention kernels at head size 64
const FP16 = 2;

export function createGpuPanel(app, cache) {
  const root = document.createElement("div");
  root.className = "na-gpu";
  root.hidden = true;
  root.innerHTML = `
    <div class="na-gpu-bar">
      <div class="na-bench-runs" role="tablist" aria-label="How attention runs">
        <button type="button" role="tab" class="na-bench-run" data-method="standard"><b>Standard attention</b><span>three kernels, T×T matrices in memory</span></button>
        <button type="button" role="tab" class="na-bench-run" data-method="flash"><b>FlashAttention</b><span>one kernel, tiles stay on chip</span></button>
      </div>
      <label class="na-gpu-pick">GPU <select data-ref="device">${DEVICES.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("")}</select></label>
      <label class="na-gpu-pick">tile <select data-ref="tile"></select></label>
    </div>
    <p class="na-bench-say" data-ref="intro"></p>
    <div class="na-dims" data-ref="dims"></div>
    <div class="na-bench-player">
      <button type="button" class="na-tool" data-act="prev" aria-label="Previous step">&#9664;</button>
      <button type="button" class="na-tool na-bench-play" data-act="play">&#9654; play</button>
      <button type="button" class="na-tool" data-act="next" aria-label="Next step">&#9654;</button>
      <input type="range" min="0" max="0" value="0" data-ref="slider" aria-label="Step">
      <span class="na-bench-stepsay" data-ref="ticksay"></span>
    </div>
    <div class="na-gpu-main">
      <p class="na-gpu-now" data-ref="now"></p>
      <div class="na-gpu-code">
        <p class="na-gpu-code-title" data-ref="codeTitle"></p>
        <pre data-ref="code"></pre>
        <p class="na-gpu-legend"><span class="na-key na-key-read"></span>read from off-chip memory <span class="na-key na-key-write"></span>written to it <span class="na-key na-key-sram"></span>on-chip</p>
      </div>
    </div>
    <div class="na-bench-stage">
      <canvas class="na-bench-canvas na-gpu-canvas" data-ref="canvas" aria-label="A GPU running one attention head: off-chip memory, the bus, the streaming multiprocessors and the on-chip memory of one of them."></canvas>
      <p class="na-bench-busy" data-ref="busy" hidden>Working it out from the weights…</p>
    </div>
    <p class="na-bench-readout" data-ref="readout"></p>
    <div class="na-gpu-meter" data-ref="meter"></div>
    <div class="na-gpu-scale" data-ref="scale"></div>
    <p class="na-bench-check" data-ref="check"></p>`;
  const ref = (name) => root.querySelector(`[data-ref="${name}"]`);
  let head = null, p = null, tick = 0, selected = 0, playing = 0, opening = 0;
  let method = "flash", deviceId = "a100", tile = 0, exact = null;
  const view = new GpuView(ref("canvas"), {
    onHover: (text) => { ref("readout").textContent = text; },
    onPick: ({ block }) => { selected = block; render(false); },
  });

  async function open(unitIndex, headIndex) {
    const mine = ++opening;
    root.hidden = false;
    ref("busy").hidden = false;
    try {
      const T = rowsReady(app, unitIndex);
      const next = await headTensors(app, unitIndex, headIndex, T, cache);
      if (mine !== opening) return;
      const fresh = !head || head.unitIndex !== next.unitIndex || head.T !== next.T;
      head = next;
      if (fresh || !tile) tile = displayTile(head.T);
      replan(fresh);
    } catch (error) {
      if (mine !== opening) return;
      ref("intro").textContent = error.message;
    } finally {
      if (mine === opening) ref("busy").hidden = true;
    }
  }

  function replan(resetTick = true) {
    p = plan({ T: head.T, d: head.s, B: tile, causal: head.causal, method });
    exact = flashOutput(head, tile);
    const sizes = [2, 4, 8, 16, 32, 64, 128].filter((b) => b < head.T * 2 && Math.ceil(head.T / b) <= 32);
    if (!sizes.includes(tile)) sizes.push(tile);
    ref("tile").innerHTML = sizes.sort((a, b) => a - b).map((b) => `<option value="${b}"${b === tile ? " selected" : ""}>${b} rows</option>`).join("");
    if (resetTick) tick = 0;
    tick = Math.min(tick, p.ticks - 1);
    const follow = Math.min(head.T - 1, app.state.position);
    selected = Math.min(p.Tr - 1, Math.floor(follow / tile));
    ref("slider").max = String(p.ticks - 1);
    root.querySelectorAll("[data-method]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.method === method)));
    ref("device").value = deviceId;
    render(false);
  }

  function render(animate = true) {
    if (!p) return;
    const device = deviceById(deviceId);
    const sim = simulate(p, head, selected, tick);
    view.set({ head, plan: p, tick, selected, sim, device }, { animate });
    ref("slider").value = String(tick);
    ref("ticksay").textContent = `step ${tick + 1} of ${p.ticks}`;
    drawIntro(device);
    drawNow(sim);
    drawCode(sim);
    drawMeter(device);
    drawScale(device);
    ref("check").textContent = `${exact.worst < 1e-4 ? "✓" : "✗"} Run to the end, FlashAttention's output for every token of head ${head.h} equals ordinary attention (the same numbers as the rest of the bench): largest difference ${exact.worst.toExponential(1)}. Online softmax is exact, not an approximation.${head.check ? ` Ordinary attention here matches the engine's to ${head.check.worst.toExponential(1)}.` : ""}`;
    ref("check").dataset.ok = String(exact.worst < 1e-4);
  }

  function drawIntro(device) {
    const T = head.T;
    ref("dims").innerHTML = headDims(head);
    const flash = method === "flash";
    ref("intro").innerHTML = `${esc(head.unit.title)}, head ${head.h}: ${T} tokens, ${head.s} numbers per query, key and value. A GPU has a lot of slow memory off the chip (${esc(device.memory)}) and a little fast memory on each of its ${device.sms} streaming multiprocessors (SMs). ${flash
      ? `<b>FlashAttention</b> cuts Q, K and V into tiles small enough for that on-chip memory. Each thread block keeps one tile of queries on chip and walks through the key and value tiles, keeping a running max <em>m</em> and running sum <em>l</em> (<em>online softmax</em>), so the ${T} × ${T} score matrix is never written anywhere. Tiles entirely in the future are skipped.`
      : `<b>Standard attention</b> runs three kernels in turn. The first writes all ${T} × ${T} scores S to off-chip memory, the second reads them back and writes the ${T} × ${T} probabilities P, the third reads P back to make the output. Most of the time goes to moving those matrices, not to arithmetic.`} Drawn with tiles of ${tile} rows so each can be seen; real kernels use 64 or 128.`;
  }

  /** The step the followed thread block is on, in words, with its numbers for one row. */
  function drawNow(sim) {
    const e = sim.event, n = sim.notes;
    const b = p.blocks[selected];
    const follow = Math.min(head.T - 1, app.state.position);
    const r = follow >= b.rows.start && follow < b.rows.end ? follow - b.rows.start : 0;
    const row = b.rows.start + r;
    const tok = `“${esc(String(head.pieces[row] ?? "").replace(/^ /, "·").replace(/\n/g, "↵"))}”`;
    const keys = (j) => `${j * tile}–${Math.min(head.T, (j + 1) * tile) - 1}`;
    let say = "";
    switch (e.phase) {
      case "init": say = `Thread block ${selected} starts on SM ${selected} and copies its ${b.rows.end - b.rows.start} queries (tokens ${b.rows.start}–${b.rows.end - 1}) into on-chip memory. For each query row it sets a running max m = −∞, a running sum l = 0 and an output accumulator of zeros. Every other block does the same with its own queries on its own SM, at the same time.`; break;
      case "load": say = `It copies key and value tile ${e.j} (tokens ${keys(e.j)}) from off-chip memory into on-chip memory.`; break;
      case "s": say = `Tensor cores multiply the query tile by the key tile: ${n.R} × ${n.C} scores, each a ${head.s}-long dot product, divided by √${head.s}. This little block is all of the score matrix that ever exists, and it never leaves the chip.`; break;
      case "mask": say = `This is the tile on the diagonal, so scores of keys that come after their query are set to −∞: after exp they become 0. Tiles wholly above the diagonal are never even loaded.`; break;
      case "max": say = `Row ${tok}: the largest score seen so far was ${f(n.mOld[r])}; this tile's largest is ${f(n.rowMax[r])}, so the running max m becomes ${f(n.m[r])}.`; break;
      case "p": say = `P = exp(S − m). Row ${tok}: the earlier sum was made with the old max, so it is first scaled by e^(${f(n.mOld[r])} − ${f(n.m[r])}) = ${f(n.factor[r], 4)}, then this tile's ${f(n.rowSum[r])} is added: l = ${f(n.lOld[r])} × ${f(n.factor[r], 4)} + ${f(n.rowSum[r])} = ${f(n.l[r])}.`; break;
      case "o": say = `The accumulator is scaled by the same ${f(n.factor[r], 4)} and P · v is added: ${n.C} value rows, weighted by this tile's P. Nothing is divided yet; that waits until every key has been seen.`; break;
      case "write": say = `All key tiles done. Dividing the accumulator by l = ${f(n.l[r])} gives the attention output for these rows, and it is written to off-chip memory, once.`; break;
      case "k1": say = `Kernel 1: block ${selected} multiplies its query tile by key tile ${e.j} and writes the ${n.R} × ${Math.min(head.T, (e.j + 1) * tile) - e.j * tile} scores to off-chip memory, even tiles in the future that the mask will throw away.`; break;
      case "k2": say = `Kernel 2: it reads its ${b.rows.end - b.rows.start} rows of S (all ${head.T} columns) back from off-chip memory, masks the future, takes the softmax and writes P, another ${head.T}-wide block, back out.`; break;
      case "k3": say = `Kernel 3: it reads P tile (${selected}, ${e.j}) and value tile ${e.j} back from off-chip memory and adds P · V to its accumulator.`; break;
      case "k3w": say = "Kernel 3 writes the finished output rows to off-chip memory."; break;
      default: say = `Block ${selected} has finished; its output rows are in off-chip memory. ${tick < p.ticks - 1 ? "Blocks with more key tiles to read are still working." : "Every block is done."}`;
    }
    ref("now").innerHTML = `<b>Block ${selected}.</b> ${say}`;
  }

  function drawCode(sim) {
    let lines = KERNELS[method];
    if (method === "flash" && !head.causal) {
      // a model that sees in both directions reads every key tile and masks nothing
      lines = lines.map((line, k) => (k === 4 ? "  for (int j = 0; j < n_tiles; j++) {  // every tile" : k === 7 ? "    // no mask: every token sees every token" : line));
    }
    const on = new Set(sim.event.lines);
    ref("codeTitle").textContent = method === "flash" ? `the kernel, as block ${selected} runs it` : "the three kernels";
    ref("code").innerHTML = lines.map((line, k) => `<span class="na-gpu-line"${on.has(k) ? " data-on" : ""}>${esc(line)}</span>`).join("\n");
  }

  function drawMeter(device) {
    const flash = trafficAt(head.T, head.s, tile, head.causal), done = (p.read[tick] + p.write[tick]) * FP16;
    const total = (p.read[p.ticks - 1] + p.write[p.ticks - 1]) * FP16;
    const top = Math.max(flash.flash, flash.standard) * FP16;
    const bar = (value, cls, label) => `<div class="na-gpu-bar-row"><span>${label}</span><div class="na-gpu-track"><i class="${cls}" style="width:${Math.max(0.5, (value / top) * 100).toFixed(2)}%"></i></div><b>${bytes(value)}</b></div>`;
    ref("meter").innerHTML = `
      <p class="na-gpu-meter-say">Off-chip traffic for this head, in float16: so far <b>${bytes(done)}</b> of ${bytes(total)} (${bytes(p.read[tick] * FP16)} read, ${bytes(p.write[tick] * FP16)} written).</p>
      ${bar(flash.standard * FP16, "na-c-std", "standard")}
      ${bar(flash.flash * FP16, "na-c-flash", "FlashAttention")}
      <p class="na-gpu-meter-say">Both do the same arithmetic. Standard attention also keeps ${bytes(flash.matrix * FP16)} of S and as much of P in off-chip memory for this one head; FlashAttention keeps nothing extra but O.</p>`;
  }

  function drawScale(device) {
    const H = head.H;
    const lengths = [...new Set([head.T, 1024, 8192, 131072])];
    const rows = lengths.map((T) => {
      const t = trafficAt(T, head.s, REAL_TILE, head.causal);
      const std = t.standard * FP16 * H, fl = t.flash * FP16 * H;
      return `<tr><th>${T.toLocaleString()}${T === head.T ? " <i>(this text)</i>" : T === 1024 ? " <i>(GPT-2's longest)</i>" : ""}</th><td>${bytes(t.matrix * FP16 * H)}</td><td>${bytes(std)}</td><td>${bytes(fl)}</td><td>${(std / fl).toFixed(1)}×</td><td>${seconds(std / device.bandwidth)} → ${seconds(fl / device.bandwidth)}</td></tr>`;
    }).join("");
    ref("scale").innerHTML = `
      <p class="na-gpu-meter-say"><b>At real sizes</b>: one layer, all ${H} heads, tiles of ${REAL_TILE}, on the ${esc(device.name)}.</p>
      <div class="na-gpu-table"><table>
        <thead><tr><th>tokens</th><th>S for all heads</th><th>traffic, standard</th><th>traffic, Flash</th><th>less</th><th>moving it at ${(device.bandwidth / 1e12).toFixed(2)} TB/s</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <p class="na-gpu-foot">Counts of numbers moved, times two bytes; the time is only what it takes to move them at full bandwidth, a lower bound. At long lengths standard attention also runs out of memory: the score matrices alone outgrow the card.</p>`;
  }

  function setTick(next, animate = true) {
    if (!p) return;
    const was = tick;
    tick = Math.max(0, Math.min(p.ticks - 1, next));
    render(animate && tick === was + 1);
  }
  function stopPlay() {
    clearInterval(playing);
    playing = 0;
    root.querySelector("[data-act=play]").innerHTML = "&#9654; play";
  }
  function togglePlay() {
    if (playing) return stopPlay();
    if (!p) return;
    if (tick >= p.ticks - 1) setTick(0, false);
    root.querySelector("[data-act=play]").innerHTML = "&#10074;&#10074; pause";
    playing = setInterval(() => { if (tick >= p.ticks - 1) return stopPlay(); setTick(tick + 1); }, 1100);
  }

  root.addEventListener("click", (event) => {
    const act = event.target.closest("[data-act]")?.dataset.act;
    if (act === "play") return togglePlay();
    if (act === "prev") { stopPlay(); return setTick(tick - 1, false); }
    if (act === "next") { stopPlay(); return setTick(tick + 1); }
    const m = event.target.closest("[data-method]")?.dataset.method;
    if (m && m !== method && head) { stopPlay(); method = m; replan(true); }
  });
  ref("slider").addEventListener("input", (event) => { stopPlay(); setTick(Number(event.target.value), false); });
  ref("device").addEventListener("change", (event) => { deviceId = event.target.value; render(false); });
  ref("tile").addEventListener("change", (event) => { stopPlay(); tile = Number(event.target.value); replan(true); });

  return {
    element: root,
    open,
    close() { root.hidden = true; opening++; stopPlay(); },
    refresh(unitIndex, headIndex) { return open(unitIndex, headIndex); },
    redraw() { view.draw(); },
  };
}
