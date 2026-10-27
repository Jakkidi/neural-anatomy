// The bench: what a plate of the cabinet opens into. One unit's operations as
// a strip of steps, each drawn as a vector × matrix product on the real
// numbers, with the dot product behind any output played out row by row.

import { signed } from "../../../ui/plates.js";
import { buildOp, stepsOf } from "./ops.js";
import { MatmulView } from "./view.js";
import { buildSequence, costModel } from "./sequence.js";
import { SequenceView } from "./sequence-view.js";
import { createGpuPanel } from "./gpu/panel.js";
import { blockDims, opDims } from "./dims.js";

const esc = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function createBench(app, { onBack = () => {} } = {}) {
  const root = document.createElement("div");
  root.className = "na-bench";
  root.hidden = true;
  root.innerHTML = `
    <div class="na-bench-head">
      <button type="button" class="na-tool na-bench-back" data-act="back">&#8592; Cabinet</button>
      <div class="na-bench-modes" role="group" aria-label="How much to show">
        <button type="button" class="na-bench-mode" data-mode="token" aria-pressed="true">One token</button>
        <button type="button" class="na-bench-mode" data-mode="sequence" aria-pressed="false">All tokens</button>
        <button type="button" class="na-bench-mode" data-mode="gpu" aria-pressed="false" title="One attention head as a GPU runs it: standard attention and FlashAttention">On a GPU</button>
      </div>
      <div class="na-bench-where" data-ref="where"></div>
    </div>
    <div class="na-bench-steps" data-ref="steps" role="tablist" aria-label="Steps of this unit"></div>
    <div class="na-bench-runs" data-ref="runs" role="tablist" aria-label="How the tokens go through" hidden>
      <button type="button" role="tab" class="na-bench-run" data-run="all"><b>Read the prompt</b><span>all tokens at once</span></button>
      <button type="button" role="tab" class="na-bench-run" data-run="nocache"><b>Generate, no cache</b><span>one token at a time, redo everything</span></button>
      <button type="button" role="tab" class="na-bench-run" data-run="cache"><b>Generate, KV cache</b><span>one token at a time, keep keys and values</span></button>
    </div>
    <div class="na-bench-heads" data-ref="heads" hidden></div>
    <p class="na-bench-say" data-ref="say"></p>
    <p class="na-bench-formula" data-ref="formula"></p>
    <div class="na-dims" data-ref="dims"></div>
    <div class="na-bench-player" data-ref="player" hidden>
      <button type="button" class="na-tool" data-act="prev" aria-label="Previous step">&#9664;</button>
      <button type="button" class="na-tool na-bench-play" data-act="play">&#9654; play</button>
      <button type="button" class="na-tool" data-act="next" aria-label="Next step">&#9654;</button>
      <input type="range" min="1" max="1" value="1" data-ref="slider" aria-label="Generation step">
      <span class="na-bench-stepsay" data-ref="stepsay"></span>
    </div>
    <div class="na-bench-legend" data-ref="legend" hidden>
      <span><i class="na-key na-key-on"></i>computed now</span>
      <span data-legend="cache"><i class="na-key na-key-cache"></i>read from the KV cache</span>
      <span data-legend="ghost"><i class="na-key na-key-ghost"></i>done at an earlier step, not needed</span>
      <span><i class="na-key na-key-mask"></i>masked: a later token, never seen</span>
      <span><i class="na-key na-key-par"></i>parallel lines: each token alone, same weights</span>
      <span><i class="na-key na-key-mix"></i>crossing lines: tokens read earlier tokens</span>
    </div>
    <div class="na-bench-tools" data-ref="tools">
      <div class="na-tools na-tools-inline" role="group" aria-label="Zoom">
        <button type="button" class="na-tool" data-act="out" aria-label="Zoom out">−</button>
        <button type="button" class="na-tool" data-act="fit">fit</button>
        <button type="button" class="na-tool" data-act="in" aria-label="Zoom in">+</button>
        <button type="button" class="na-tool" data-act="numbers" title="Zoom in until every number is written">numbers</button>
      </div>
      <span class="na-bench-hint">Click any output (top) or any cell to work out its dot product · drag to move · pinch or Ctrl + scroll to zoom</span>
    </div>
    <div class="na-bench-stage">
      <canvas class="na-bench-canvas" data-ref="canvas" tabindex="0" aria-label="The operation as a vector times a matrix. Arrow keys move, plus and minus zoom, 0 fits."></canvas>
      <canvas class="na-bench-canvas na-bench-seq" data-ref="seq" hidden aria-label="Every token of the text going through this unit, one row per token."></canvas>
      <p class="na-bench-busy" data-ref="busy" hidden>Working it out from the weights…</p>
    </div>
    <p class="na-bench-readout" data-ref="readout"></p>
    <div class="na-bench-dot" data-ref="dot"></div>
    <div class="na-bench-cost" data-ref="cost" hidden></div>
    <p class="na-bench-check" data-ref="check"></p>`;
  const ref = (name) => root.querySelector(`[data-ref="${name}"]`);
  const cache = new Map();
  let current = { unit: -1, step: "", head: 0 };
  let op = null, opening = 0;
  let mode = "token"; // "token": one token's arithmetic; "sequence": every token at once; "gpu": attention on a GPU
  const gpu = createGpuPanel(app, cache);
  root.append(gpu.element);
  let run = "all"; // in "sequence": "all", "nocache" or "cache"
  let seq = null, playing = 0;

  const seqView = new SequenceView(ref("seq"), {
    onHover: (text) => { ref("readout").textContent = text; },
    onPick: ({ row, stage, col }) => pick(row, stage, col),
  });

  const view = new MatmulView(ref("canvas"), app, {
    onHover: (text) => { ref("readout").textContent = text; },
    onSelect: () => drawDot(),
    onSweep: (progress, partial) => drawDot(progress, partial),
  });

  function drawSteps() {
    const unit = app.state.spec.units[current.unit];
    const steps = stepsOf(unit);
    ref("steps").innerHTML =
      `<span class="na-bench-io">stream in</span>` +
      steps.map((s, i) => `<span class="na-bench-arrow" aria-hidden="true">→</span><button type="button" role="tab" class="na-bench-step" data-step="${s.id}" aria-selected="${s.id === current.step}"><b>${i + 1}</b> ${esc(s.label)}</button>`).join("") +
      `<span class="na-bench-arrow" aria-hidden="true">→</span><span class="na-bench-io">${unit.kind === "unembed" ? "next token" : "stream out"}</span>`;
    const step = steps.find((s) => s.id === current.step);
    const H = unit.attention?.heads ?? 0;
    const sequence = mode === "sequence";
    const onGpu = mode === "gpu";
    root.querySelector('[data-mode="gpu"]').disabled = !unit.parts?.attention;
    ref("steps").hidden = sequence || onGpu;
    ref("runs").hidden = !sequence;
    root.querySelectorAll("[data-run]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.run === run)));
    root.querySelectorAll(".na-bench-mode").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    const showHeads = sequence || onGpu ? Boolean(unit.parts?.attention) : step?.perHead;
    ref("heads").hidden = !showHeads;
    if (showHeads) {
      ref("heads").innerHTML = `<span>head</span>` + Array.from({ length: H }, (_, h) => `<button type="button" class="na-bench-head-pick" data-head="${h}" aria-pressed="${h === current.head}">${h}</button>`).join("");
    }
    ref("where").innerHTML = onGpu
      ? `<b>${esc(unit.title)}</b> · <i>one attention head on a GPU</i>`
      : sequence
      ? `<b>${esc(unit.title)}</b> · <i>every token of the text, one row each</i>`
      : `<b>${esc(unit.title)}</b> · ${esc(step?.label ?? "")} <i>· following “${esc(String(app.state.tokens.pieces[app.state.position] ?? "").replace(/^ /, "·"))}”, token ${app.state.position + 1}</i>`;
  }

  function drawDot(progress = view.sweep, partial = null) {
    const box = ref("dot");
    if (!op || view.selected < 0) {
      box.innerHTML = `<p class="na-bench-tip">Each output number is a <b>dot product</b>: multiply every number of the input by the matching weight in that output's column, and add up the products${op?.bias ? ", then add a bias" : ""}. Click an output across the top to watch one happen.</p>`;
      return;
    }
    const j = view.selected;
    const n = op.input.values.length;
    const bias = op.bias ? op.bias[j] : 0;
    const out = op.output.values[j];
    const sum = partial ? partial.sum : (view.products ? view.products.reduce((a, b) => a + b, 0) : NaN);
    const count = partial ? partial.count : n;
    const done = progress >= 1 && view.column;
    const after = op.after.map((a) => `<span class="na-bench-op">→ ${esc(a.label)}</span><span class="na-bench-val">${signed(a.values[j], 4)}</span>`).join("");
    const engineRow = op.after.findLast?.((a) => a.engine) ?? (op.output.engine ? op.output : null);
    const engine = engineRow?.engine ? engineRow.engine[j] : undefined;
    const columnCheck = done && op.columnCheck ? op.columnCheck(j, sum + bias) : null;
    let top = "";
    if (done && view.products) {
      const order = Array.from(view.products.keys()).sort((a, b) => Math.abs(view.products[b]) - Math.abs(view.products[a])).slice(0, 6);
      top = `<div class="na-bench-top"><span>largest products:</span>${order.map((i) => `<button type="button" data-row="${i}">${esc(op.input.name(i))}: ${signed(op.input.values[i], 3)} × ${signed(view.column[i], 3)} = <b>${signed(view.products[i], 4)}</b></button>`).join("")}</div>`;
    }
    box.innerHTML = `
      <p class="na-bench-eq">
        <span class="na-bench-name">${esc(op.columnName(j))}</span>
        <span class="na-bench-op">=</span>
        <span class="na-bench-sum">Σ of ${n.toLocaleString()} products <i>(${count.toLocaleString()} added so far)</i> <b>${Number.isFinite(sum) ? signed(sum, 4) : "…"}</b></span>
        ${op.bias ? `<span class="na-bench-op">+ bias</span><span class="na-bench-val">${signed(bias, 4)}</span>` : ""}
        <span class="na-bench-op">=</span><span class="na-bench-val na-bench-final">${done ? signed(sum + bias, 4) : "…"}</span>
        ${done ? after : ""}
      </p>
      ${done ? `<p class="na-bench-match">${columnCheck ? esc(columnCheck.text) : engine !== undefined ? `${Math.abs((op.after.length ? op.after[op.after.length - 1].values[j] : sum + bias) - engine) < 2e-3 ? "✓" : "✗"} The engine computed ${signed(engine, 5)} here.` : Number.isFinite(out) ? `Shown above the column: ${signed(out, 5)}.` : ""}</p>` : ""}
      ${top}`;
  }

  async function open(unitIndex, stepId, { head = current.head, column = -1 } = {}) {
    const unit = app.state.spec.units[unitIndex];
    const steps = stepsOf(unit);
    if (!steps.some((s) => s.id === stepId)) stepId = steps[0].id;
    const sameShape = current.unit === unitIndex && current.step === stepId;
    current = { unit: unitIndex, step: stepId, head };
    if (mode === "gpu" && !unit.parts?.attention) mode = "token";
    root.hidden = false;
    showMode();
    drawSteps();
    if (mode === "gpu") return gpu.open(current.unit, current.head);
    gpu.close();
    if (mode === "sequence") return openSequence();
    const mine = ++opening;
    ref("busy").hidden = false;
    ref("check").textContent = "";
    try {
      const next = await buildOp(app, unitIndex, stepId, { head }, cache);
      if (mine !== opening) return;
      op = next;
      ref("say").innerHTML = op.sentence;
      ref("formula").textContent = op.formula;
      ref("dims").innerHTML = opDims(app, unitIndex, op);
      ref("check").textContent = op.check?.text ?? "";
      ref("check").dataset.ok = String(op.check?.ok ?? true);
      view.setOp(op, { keepView: sameShape });
      const focus = column >= 0 ? column : op.focus;
      if (focus !== undefined && focus >= 0) view.select(focus);
      drawDot();
    } catch (error) {
      if (mine !== opening) return;
      ref("say").textContent = error.message;
    } finally {
      if (mine === opening) ref("busy").hidden = true;
    }
  }

  root.addEventListener("click", (event) => {
    if (gpu.element.contains(event.target)) return; // the GPU panel has its own controls
    const act = event.target.closest("[data-act]")?.dataset.act;
    if (act === "back") return onBack();
    if (act === "play") return togglePlay();
    if (act === "prev") return setStep(seqView.t - 1);
    if (act === "next") return setStep(seqView.t + 1);
    const pickMode = event.target.closest(".na-bench-mode")?.dataset.mode;
    if (pickMode) {
      if (pickMode === "gpu" && !app.state.spec.units[current.unit].parts?.attention) return;
      if (pickMode !== mode) { mode = pickMode; stopPlay(); open(current.unit, current.step); }
      return;
    }
    const pickRun = event.target.closest("[data-run]")?.dataset.run;
    if (pickRun) return setRun(pickRun);
    if (act === "in") return view.zoomAt(1.8);
    if (act === "out") return view.zoomAt(1 / 1.8);
    if (act === "fit") return view.fit();
    if (act === "numbers") return view.zoomTo(view.pinned >= 0 ? view.pinned : 0, Math.max(0, view.selected));
    const step = event.target.closest("[data-step]")?.dataset.step;
    if (step) return open(current.unit, step);
    const head = event.target.closest("[data-head]")?.dataset.head;
    if (head !== undefined) return open(current.unit, current.step, { head: Number(head), column: mode === "token" ? -1 : undefined });
    const row = event.target.closest("[data-row]")?.dataset.row;
    if (row !== undefined) { view.pinned = Number(row); view.zoomTo(Number(row), view.selected); }
  });

  ref("slider").addEventListener("input", (event) => { stopPlay(); setStep(Number(event.target.value)); });

  function showMode() {
    const sequence = mode === "sequence";
    root.dataset.mode = mode;
    ref("canvas").hidden = sequence;
    ref("seq").hidden = !sequence;
    ref("tools").hidden = sequence;
    ref("dot").hidden = sequence;
    ref("cost").hidden = !sequence;
    ref("legend").hidden = !sequence;
    ref("player").hidden = !sequence || run === "all";
    ref("legend").querySelector("[data-legend=cache]").hidden = run !== "cache";
    ref("legend").querySelector("[data-legend=ghost]").hidden = run !== "cache";
  }

  async function openSequence() {
    const mine = ++opening;
    ref("busy").hidden = false;
    try {
      const next = await buildSequence(app, current.unit, { head: current.head }, cache);
      if (mine !== opening) return;
      seq = next;
      ref("check").textContent = seq.check.text;
      ref("check").dataset.ok = String(seq.check.ok);
      seqView.position = app.state.position;
      ref("seq").style.height = `${SequenceView.heightFor(seq.T)}px`;
      const fresh = seqView.seq?.unitIndex !== seq.unitIndex || seqView.seq?.T !== seq.T;
      seqView.set(seq);
      ref("slider").max = String(seq.T);
      if (fresh && run !== "all") seqView.t = Math.min(seqView.t, seq.T);
      describeRun();
      if (fresh) seqView.play();
    } catch (error) {
      if (mine !== opening) return;
      ref("say").textContent = error.message;
    } finally {
      if (mine === opening) ref("busy").hidden = true;
    }
  }

  function setRun(next) {
    stopPlay();
    run = next;
    showMode();
    drawSteps();
    if (!seq) return;
    const wasGenerating = seqView.mode !== "all";
    seqView.setMode(run, run === "all" ? seq.T : wasGenerating ? seqView.t : 1);
    describeRun();
    if (run !== "all" && !wasGenerating && seq.T > 1) togglePlay();
  }

  function setStep(t) {
    if (!seq || run === "all") return;
    t = Math.max(1, Math.min(seq.T, t));
    seqView.setMode(run, t);
    describeRun();
  }

  function stopPlay() {
    clearInterval(playing);
    playing = 0;
    ref("player").querySelector("[data-act=play]").innerHTML = "&#9654; play";
  }
  function togglePlay() {
    if (playing) return stopPlay();
    if (!seq) return;
    if (seqView.t >= seq.T) setStep(1);
    ref("player").querySelector("[data-act=play]").innerHTML = "&#10074;&#10074; pause";
    playing = setInterval(() => {
      if (!seq || seqView.t >= seq.T) return stopPlay();
      setStep(seqView.t + 1);
    }, 1500);
  }

  const mega = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 0 : n >= 1e10 ? 1 : 2)} billion` : `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} million`);
  const bytes = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`);

  function describeRun() {
    if (!seq) return;
    const unit = seq.unit, T = seq.T, t = seqView.t;
    const piece = (i) => `“${esc(String(seq.pieces[i] ?? "").replace(/^ /, "·").replace(/\n/g, "↵"))}”`;
    const isBlock = unit.kind === "block";
    const mixing = isBlock
      ? "Only attention moves anything between rows: there each token's query is scored against the keys of itself and every earlier token (the crossing lines), and it takes a weighted mix of their values. The grey triangle is the mask: no token can see one that comes after it."
      : unit.kind === "embed" ? "Nothing moves between rows here: each token's row is looked up on its own." : "Every row predicts the token that follows it.";
    let say = "", formula = "";
    if (run === "all") {
      say = `<b>Reading the prompt.</b> All ${T} tokens go through ${esc(unit.title.toLowerCase())} together, one row each. Each weight matrix multiplies all ${T} rows in a single matrix × matrix product, so the work runs in parallel. ${mixing} <span class="na-bench-aside">This is what sets a transformer apart from a recurrent network (RNN): an RNN has to read tokens strictly one after another and squeeze everything so far into one fixed-size memory, while a transformer keeps every earlier token's key and value and looks straight back at any of them.</span>`;
      formula = isBlock ? `[Q K V] = LN(X) · W_QKV      shares = softmax(mask(Q·Kᵀ / √${seq.s}))      head = shares · V      (X is ${T} × ${seq.d}: one row per token)` : "";
    } else {
      const now = t - 1;
      const next = t < T ? `it should produce ${piece(t)}` : "the next token comes after the text";
      if (run === "nocache") {
        say = `<b>Generating without a cache, step ${t} of ${T}.</b> The model has ${t} token${t > 1 ? "s" : ""} and wants the next (${next}). With no memory of the last step, it puts all ${t} tokens through every block again. Rows 1 to ${Math.max(1, t - 1)} come out exactly as they did before; only the last row, ${piece(now)}, is new, and only its prediction is used.`;
      } else {
        say = `<b>Generating with a KV cache, step ${t} of ${T}.</b> Only the new token, ${piece(now)}, goes through (${next}). ${isBlock ? `Its query is scored against the keys of all ${t} tokens, but the earlier ${t - 1} keys and values are read from memory (blue) instead of being computed again.` : "Earlier rows were computed at earlier steps and are not needed again."} That is safe because of the mask: a token's key and value depend only on the tokens before it, so a new token never changes them.`;
      }
      formula = isBlock ? (run === "cache" ? `q = LN(x_new) · W_Q      K = [cached keys; k_new]      shares = softmax(q·Kᵀ / √${seq.s})      head = shares · [cached values; v_new]` : `all ${t} rows: [Q K V] = LN(X_${t}) · W_QKV      shares = softmax(mask(Q·Kᵀ / √${seq.s}))      keep only the last row`) : "";
    }
    ref("say").innerHTML = say;
    ref("formula").textContent = formula;
    ref("dims").innerHTML = blockDims(app.state.spec, seq.unit, run === "all" ? seq.T : seqView.t);
    ref("slider").value = String(t);
    ref("stepsay").textContent = run === "all" ? "" : `step ${t} of ${T} · ${t} token${t > 1 ? "s" : ""} in`;
    drawCost();
  }

  function drawCost() {
    const box = ref("cost");
    if (!seq) return;
    const cost = costModel(app.state.spec);
    const T = seq.T, t = run === "all" ? T : seqView.t;
    const sum = (f, n) => { let s = 0; for (let i = 1; i <= n; i++) s += f(i); return s; };
    const totalNo = sum(cost.noCache, t), totalCache = sum(cost.withCache, t);
    const chart = costChart(cost, T, t);
    const head = run === "all"
      ? `<p class="na-bench-cost-say">Reading all ${T} tokens at once takes <b>${mega(cost.prompt(T))}</b> multiply-adds across the whole model (${cost.L} blocks and the output), done as a few big matrix products. Generating the same ${T} tokens one at a time takes <b class="na-c-cache">${mega(sum(cost.withCache, T))}</b> with a KV cache (the same work, plus scoring a next token at every step) but <b class="na-c-no">${mega(sum(cost.noCache, T))}</b> without one, because every step runs all earlier tokens again.</p>`
      : `<div class="na-bench-cost-grid">
          <div class="na-c-no"><span>without cache</span><b>${mega(cost.noCache(t))}</b><i>this step · ${mega(totalNo)} so far</i></div>
          <div class="na-c-cache"><span>with KV cache</span><b>${mega(cost.withCache(t))}</b><i>this step · ${mega(totalCache)} so far</i></div>
          <div><span>the cache holds</span><b>${bytes(cost.cacheBytes(t))}</b><i>${t} × ${cost.L} blocks × (key + value) × ${cost.d} numbers</i></div>
        </div>
        <p class="na-bench-cost-say">Multiply-adds for the whole model at this step. Without the cache every step repeats all earlier work, so the cost grows with the square of the length; with it each step costs about the same. Same numbers come out either way: the cache trades a little memory for not redoing work.</p>`;
    box.innerHTML = head + chart;
  }

  function costChart(cost, T, t) {
    const W = 560, H = 170, padL = 64, padR = 112, padT = 12, padB = 30;
    const n = Math.max(2, T);
    const no = [], ca = [];
    let a = 0, b = 0;
    for (let i = 1; i <= n; i++) { a += cost.noCache(i); b += cost.withCache(i); no.push(a); ca.push(b); }
    const top = no[n - 1] || 1;
    const x = (i) => padL + ((i - 1) / (n - 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - v / top) * (H - padT - padB);
    const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const ticks = [0, 0.5, 1].map((f) => `<line x1="${padL}" x2="${W - padR}" y1="${y(top * f)}" y2="${y(top * f)}" class="na-c-grid"/><text x="${padL - 6}" y="${y(top * f) + 3}" text-anchor="end">${f ? mega(top * f).replace(" billion", " B").replace(" million", " M") : "0"}</text>`).join("");
    const k = Math.max(1, Math.min(n, t));
    const hits = Array.from({ length: n }, (_, i) => `<rect x="${x(i + 1) - (W - padL - padR) / (n - 1) / 2}" y="${padT}" width="${(W - padL - padR) / (n - 1)}" height="${H - padT - padB}" fill="transparent"><title>after ${i + 1} token${i ? "s" : ""}: ${mega(no[i])} without cache, ${mega(ca[i])} with</title></rect>`).join("");
    return `<svg class="na-bench-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Total multiply-adds to generate the text one token at a time, with and without a KV cache">
      ${ticks}
      <text x="${(padL + W - padR) / 2}" y="${H - 6}" text-anchor="middle">tokens generated (1 to ${n})</text>
      <path d="${path(no)}" class="na-c-line na-c-no"/>
      <path d="${path(ca)}" class="na-c-line na-c-cache"/>
      <text x="${W - padR + 8}" y="${y(no[n - 1]) + 4}" class="na-c-label">no cache</text>
      <text x="${W - padR + 8}" y="${Math.min(H - padB, y(ca[n - 1]) + 4)}" class="na-c-label">KV cache</text>
      <line x1="${x(k)}" x2="${x(k)}" y1="${padT}" y2="${H - padB}" class="na-c-now"/>
      <circle cx="${x(k)}" cy="${y(no[k - 1])}" r="4" class="na-c-dot na-c-no"/>
      <circle cx="${x(k)}" cy="${y(ca[k - 1])}" r="4" class="na-c-dot na-c-cache"/>
      ${hits}
    </svg>`;
  }

  /** From the sequence to one token's arithmetic: the row is the token to follow. */
  function pick(row, stage, col) {
    stopPlay();
    mode = "token";
    const step = stage?.step ?? current.step;
    const column = stage?.kind === "grid" ? col : stage?.columnOf ? stage.columnOf(col) : col;
    current.step = step;
    opening++;
    if (row !== app.state.position) app.setPosition(row);
    open(current.unit, step, { head: current.head, column: column ?? -1 });
  }

  return {
    element: root,
    open,
    get unit() { return current.unit; },
    refresh() {
      if (root.hidden || current.unit < 0) return;
      if (mode === "sequence") seqView.position = app.state.position;
      open(current.unit, current.step, { head: current.head, column: view.selected });
    },
    close() { root.hidden = true; opening++; stopPlay(); gpu.close(); },
    redraw() { if (mode === "gpu") gpu.redraw(); else if (mode === "sequence") seqView.draw(); else view.draw(); },
    get isOpen() { return !root.hidden; },
  };
}
