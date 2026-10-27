// The page: one model, one prompt, one trace, shown by whichever figures the
// model's architecture offers. This file is the glue between the bench
// controls, the engine worker and the figures; it draws nothing itself.

import { UI_ARCHS } from "./arch/ui-registry.js";
import { EngineClient } from "./core/client.js";
import { NativeClient } from "./core/native-client.js";
import { Trace } from "./core/trace.js";
import { MODELS, modelById } from "./models/index.js";
import { createCard } from "./ui/card.js";
import { inksOf, percent } from "./ui/plates.js";
import { TAPE, showPiece } from "./figures/parts.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Who is serving this page decides where the engine runs and where weights come from:
//   (nothing)   a web server: the engine is a Web Worker, weights are downloaded by the browser
//   "sandbox"   a host that forbids downloads from other sites: weights come from a file only
//   "native"    the desktop app: the engine is the Rust program, weights live on disk
const host = $("app").dataset.host ?? "";
const sandboxed = host === "sandbox";
const native = host === "native";

// ---- state and events ----------------------------------------------------------
const listeners = new Map();
const app = {
  state: {
    card: null, spec: null, info: null, trace: null,
    tokens: { ids: [], pieces: [] },
    position: 0, window: { start: 0, size: TAPE },
    open: -1, layer: 0, running: false, generating: false, temperature: 0,
    weightsStamp: 0, runStamp: 0, plate: null, modelOpen: true,
  },
  native,
  engine: native ? new NativeClient() : new EngineClient({ nowasm: params.has("nowasm") }),
  on(event, fn) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(fn);
    return () => listeners.get(event).delete(fn);
  },
  emit(event, detail = {}) {
    for (const fn of listeners.get(event) ?? []) fn(detail);
  },
  inks: () => inksOf(),
  fail(error) {
    console.error(error);
    showError(error?.message ?? String(error));
  },
  setPosition(position) {
    const { tokens } = app.state;
    if (!tokens.ids.length) return;
    position = Math.max(0, Math.min(tokens.ids.length - 1, position));
    if (position === app.state.position) return;
    app.state.position = position;
    follow();
    app.emit("position");
  },
  /** Pulls a module out, or pushes it back if it is already out. `exact` sets rather than toggles. */
  toggleUnit(index, exact = false) {
    app.state.open = exact ? index : app.state.open === index ? -1 : index;
    app.emit("open");
  },
  /**
   * Chooses the layer in focus: a unit of the model, 0 for the embedding up to
   * the output head. The cabinet keeps it for its modules.
   */
  setLayer(index) {
    index = Math.max(0, Math.min(app.state.spec.units.length - 1, index));
    if (index === app.state.layer) return;
    app.state.layer = index;
    app.emit("layer");
  },
};
app.cards = () => cards;
window.neuralAnatomy = app; // handy from the console, and used by the browser checks

function follow() {
  const { tokens, position, window: win } = app.state;
  const T = tokens.ids.length;
  if (T <= TAPE) win.start = 0;
  else {
    if (position < win.start + 2) win.start = Math.max(0, position - 2);
    if (position > win.start + TAPE - 3) win.start = Math.min(T - TAPE, position - TAPE + 3);
    win.start = Math.max(0, Math.min(T - TAPE, win.start));
  }
}

// ---- figures ----------------------------------------------------------------------
let cards = [];
let activeFigure = 0;
function mountFigures() {
  for (const card of cards) card.controller?.destroy();
  cards = [];
  const host = $("figure-host"), tabs = $("figure-tabs");
  host.innerHTML = "";
  tabs.innerHTML = "";
  const { spec, card: model } = app.state;
  const figures = UI_ARCHS[model.arch].figures;
  activeFigure = Math.min(activeFigure, figures.length - 1);
  figures.forEach((figure, index) => {
    const card = createCard({
      fig: figure.fig,
      title: typeof figure.title === "function" ? figure.title(spec) : figure.title,
      hint: figure.hint,
      keys: figure.keys,
      caption: typeof figure.caption === "function" ? figure.caption(spec) : figure.caption,
      label: figure.label,
    });
    card.figure = figure;
    card.root.id = `figure-${figure.id}`;
    card.root.hidden = index !== activeFigure;
    host.append(card.root);
    cards.push(card);
    if (figures.length > 1) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.role = "tab";
      tab.id = `tab-${figure.id}`;
      tab.setAttribute("aria-controls", card.root.id);
      tab.setAttribute("aria-selected", String(index === activeFigure));
      tab.textContent = `${figure.fig} · ${figure.name}`;
      tab.addEventListener("click", () => showFigure(index));
      tabs.append(tab);
    }
  });
  showFigure(activeFigure);
}
function showFigure(index) {
  activeFigure = index;
  cards.forEach((card, i) => {
    card.root.hidden = i !== index;
    // only the figure on show is built and listening
    if (i !== index && card.controller) { card.controller.destroy(); card.controller = null; }
    if (i === index && !card.controller) { card.controller = card.figure.mount(card.stage, app, card); card.setZoom(1); }
  });
  [...$("figure-tabs").children].forEach((tab, i) => tab.setAttribute("aria-selected", String(i === index)));
  $("app").dataset.figure = cards[index].figure.id;
  try { localStorage.setItem("na.figure", cards[index].figure.id); } catch { /* storage is optional */ }
}

// ---- model ----------------------------------------------------------------------------
function selectModel(id) {
  let card = modelById(id) ?? MODELS[0];
  // ?weights=<url> reads this model's weights from a mirror or a local copy instead
  if (params.get("weights") && card.weights.kind === "safetensors") card = { ...card, weights: { ...card.weights, url: new URL(params.get("weights"), location.href).href } };
  const arch = UI_ARCHS[card.arch];
  Object.assign(app.state, {
    card, spec: arch.spec(card.config), info: null,
    tokens: { ids: [], pieces: [] }, position: 0, open: -1, running: false, generating: false, plate: null,
  });
  app.state.layer = app.state.spec.units.length - 1; // everything lit until the person steps down
  app.state.window.start = 0;
  app.state.trace = new Trace(app.state.spec.capacity);
  app.state.weightsStamp++;
  $("model-select").value = card.id;
  try { localStorage.setItem("na.model", card.id); } catch { /* storage is optional */ }
  showError("");
  mountFigures();
  app.emit("layer");
  drawModelPanel();
  drawSamples();
  if (!$("prompt-input").value.trim() || MODELS.some((m) => m.prompts.includes($("prompt-input").value))) $("prompt-input").value = card.prompts[0];
  retokenize().then((tokens) => { if (tokens && app.state.card === card && !app.state.info) showTokens(tokens); });
  drawNext();
  app.engine.call("saved", { card }).then((saved) => {
    if (app.state.card !== card) return;
    app.state.savedCopy = saved;
    app.state.savedPath = app.engine.savedPath ?? "";
    drawModelPanel();
    if (saved && !app.state.info && !params.has("noauto") && !sandboxed) load({ kind: "url" });
  }).catch(() => {});
}

const mb = (bytes) => `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
function drawModelPanel() {
  const { card, info, savedCopy, savedPath, loading } = app.state;
  $("model-blurb").textContent = card.blurb;
  const remote = card.weights.kind === "safetensors";
  // Once weights are in, the panel folds to two lines so the cabinet sits
  // beside the machine; "Change weights" unfolds it again.
  const folded = Boolean(info) && !loading && !app.state.modelOpen;
  $("panel-model").toggleAttribute("data-folded", folded);
  $("model-toggle").hidden = !info || Boolean(loading);
  $("model-toggle").textContent = folded ? "Change weights" : "Show less";
  $("model-toggle").setAttribute("aria-expanded", String(!folded));
  $("model-summary").hidden = !folded;
  $("model-summary").textContent = info
    ? `${info.params.toLocaleString()} weights loaded${info.random ? ", all random" : ` from ${info.from}`}.`
    : "";
  $("load-actions").hidden = Boolean(loading);
  $("load-progress").hidden = !loading;
  $("load-button").hidden = !remote || sandboxed;
  $("file-button").hidden = !remote;
  $("forget-button").hidden = !(remote && savedCopy);
  const fetchWord = native ? "Download weights" : "Load weights";
  $("load-button").textContent = info && !info.random && (savedCopy || !native) ? "Reload weights" : savedCopy ? "Load saved weights" : `${fetchWord} \u00b7 ${mb(card.weights.bytes ?? 0)}`;
  $("random-button").textContent = remote ? "Random weights" : info ? "Rebuild random weights" : "Build random weights";
  $("random-button").toggleAttribute("data-primary", !remote);
  $("load-button").toggleAttribute("data-primary", !info);
  $("file-button").toggleAttribute("data-primary", sandboxed && !info);
  const status = $("model-status"), check = $("model-check");
  status.textContent = "";
  check.textContent = "";
  check.removeAttribute("data-bad");
  const where = native ? "on this computer" : "in this browser";
  if (info) {
    const kind = info.random ? "random weights, so its output means nothing" : `from ${info.from}`;
    const engine = native ? info.backend : info.backend === "wasm-simd" ? "WebAssembly SIMD" : "plain JavaScript";
    let text = `${info.params.toLocaleString()} weights loaded ${info.random ? "as " : ""}${kind}. ${native ? "Engine" : "Arithmetic"}: ${engine}.`;
    if (!info.random && info.path) text += ` Kept at ${info.path}.`;
    else if (!info.random && info.from === "download") text += info.saved ? ` A copy is kept ${where}.` : " This browser could not keep a copy, so it will download again next time.";
    else if (!info.random && native && info.from === "your file" && !info.saved) text += " It could not be copied into the app's folder, so choose it again next time.";
    status.textContent = text;
    if (info.check) {
      // the engine measured the freshly loaded model on a paragraph of plain English
      const { nats, chance, ok } = info.check;
      check.textContent = ok && folded
        ? `Self-check passed: ${nats.toFixed(2)} nats per token on plain English.`
        : ok
        ? `Self-check passed: these weights predict plain English at ${nats.toFixed(2)} nats per token. An untrained or wrongly wired model scores about ${chance.toFixed(1)}.`
        : `Self-check failed: these weights predict plain English at ${nats.toFixed(2)} nats per token, and chance is ${chance.toFixed(1)}. This is not a trained model, or it is not being run correctly.`;
      check.toggleAttribute("data-bad", !ok);
    }
  } else if (remote && sandboxed) {
    // this page may not download from other sites, so the file has to be handed to it
    const link = document.createElement("a");
    link.href = card.weights.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `${card.weights.file} (${mb(card.weights.bytes ?? 0)})`;
    status.append("This hosted page cannot download from other sites. Get ", link, " from Hugging Face, then choose it with \u201cUse a file\u201d or drop it anywhere on this page. It stays on your computer.");
  } else if (remote) {
    status.textContent = savedCopy
      ? native && savedPath ? `The weights are on this computer at ${savedPath}.` : `A copy from an earlier visit is kept ${where}.`
      : `Downloads once from huggingface.co and is kept ${where}. Or pick a ${card.weights.file} you already have.`;
  } else status.textContent = "Nothing to download: this model's weights are generated here.";
}
function showError(message) {
  $("model-error").hidden = !message;
  $("model-error").textContent = message;
}

async function load(source) {
  const { card } = app.state;
  if (app.state.loading) return;
  showError("");
  app.state.loading = true;
  app.state.info = null;
  app.state.weightsStamp++;
  app.state.trace.clear();
  app.emit("weights");
  $("load-bar").style.width = "0%";
  $("load-text").textContent = source.kind === "random" ? "Generating random weights…" : "Opening…";
  drawModelPanel();
  try {
    const info = await app.engine.call("load", { card, source });
    if (app.state.card !== card) return;
    app.state.info = info;
    app.state.loading = false;
    app.state.modelOpen = false;
    // A file can hold a different size of the same architecture (a 6-block or a
    // 24-block GPT-2, say). The machine is rebuilt to match what was loaded.
    const loaded = UI_ARCHS[card.arch].spec(info.config);
    if (JSON.stringify(loaded.config) !== JSON.stringify(app.state.spec.config)) {
      app.state.spec = loaded;
      app.state.trace = new Trace(loaded.capacity);
      app.state.open = -1;
      app.state.layer = loaded.units.length - 1;
      mountFigures();
    }
    app.state.savedCopy = app.state.savedCopy || info.saved;
    if (info.path) app.state.savedPath = info.path;
    if (app.state.layer >= app.state.spec.units.length) app.state.layer = app.state.spec.units.length - 1;
    drawModelPanel();
    app.emit("weights");
    await run();
  } catch (error) {
    app.state.loading = false;
    drawModelPanel();
    const blocked = error.code === "blocked";
    showError(blocked ? `${error.message} Download ${card.weights.file} from ${card.weights.page} and use “Use a file” instead.` : error.message);
  }
}

// ---- prompt, tokens, runs --------------------------------------------------------------------
// A preview (while typing) never cancels a run: runs keep their own count.
let tokenizeStamp = 0, typingTimer = 0;
async function retokenize({ forRun = false } = {}) {
  const { card } = app.state;
  const stamp = forRun ? tokenizeStamp : ++tokenizeStamp;
  const text = $("prompt-input").value;
  try {
    const { ids, pieces } = await app.engine.call("tokenize", { card, text });
    if (app.state.card !== card || (!forRun && stamp !== tokenizeStamp)) return null;
    if (forRun) tokenizeStamp++; // a preview still in flight is now out of date
    const capacity = app.state.spec.capacity;
    const room = capacity - 16;
    $("prompt-meta").textContent = `${ids.length} token${ids.length === 1 ? "" : "s"}${ids.length > room ? ` · first ${room} used` : ""}`;
    return { ids: ids.slice(0, room), pieces: pieces.slice(0, room) };
  } catch (error) {
    app.fail(error);
    return null;
  }
}
function showTokens(tokens, fresh = 0) {
  app.state.tokens = tokens;
  app.state.freshFrom = tokens.ids.length - fresh;
  app.state.position = Math.max(0, tokens.ids.length - 1);
  follow();
  app.emit("tokens");
  drawTokens();
}

async function run() {
  clearTimeout(typingTimer);
  const tokens = await retokenize({ forRun: true });
  if (!tokens) return;
  if (!tokens.ids.length) {
    // nothing to run: clear what the last input left behind
    app.state.runStamp++;
    app.state.running = false;
    app.state.trace.clear();
    showTokens(tokens);
    app.emit("run", { reset: true });
    app.emit("done", null);
    drawNext();
    return;
  }
  app.state.promptLength = tokens.ids.length;
  app.state.trace.clear();
  showTokens(tokens);
  if (!app.state.info) { app.emit("run", { reset: true }); return; }
  await forward(tokens.ids, true);
}

async function forward(ids, reset) {
  const stamp = ++app.state.runStamp;
  app.state.running = true;
  app.emit("run", { reset });
  drawNext();
  try {
    const result = await app.engine.call("run", { ids, reset });
    if (stamp !== app.state.runStamp) return null;
    app.state.running = false;
    if (!result.cancelled) {
      app.state.lastRun = result;
      app.emit("done", result);
      drawTokens();
      drawNext();
    }
    return result;
  } catch (error) {
    app.state.running = false;
    app.fail(error);
    return null;
  }
}

async function step() {
  const { info, tokens, spec } = app.state;
  if (!info || !tokens.ids.length || app.state.running) return false;
  if (tokens.ids.length >= spec.capacity) { $("token-note").textContent = `A run holds ${spec.capacity} tokens. Run a new prompt to go on.`; return false; }
  const picked = await app.engine.call("pick", { temperature: app.state.temperature, topK: 40 });
  showTokens({ ids: [...tokens.ids, picked.id], pieces: [...tokens.pieces, picked.piece] }, tokens.ids.length + 1 - (app.state.promptLength ?? tokens.ids.length));
  const result = await forward([picked.id], false);
  return Boolean(result && !result.cancelled);
}
async function generate(count) {
  if (app.state.generating) return;
  app.state.generating = true;
  $("stop-button").hidden = false;
  $("generate-button").disabled = $("step-button").disabled = true;
  try {
    for (let i = 0; i < count && app.state.generating; i++) {
      if (!(await step())) break;
      await new Promise((resolve) => setTimeout(resolve, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 520));
    }
  } finally {
    app.state.generating = false;
    $("stop-button").hidden = true;
    $("generate-button").disabled = $("step-button").disabled = false;
  }
}

// ---- engine events --------------------------------------------------------------------------------
app.engine.on((event) => {
  const { trace } = app.state;
  if (event.event === "load") {
    if (!app.state.loading) return;
    const share = event.total ? event.loaded / event.total : 0;
    $("load-bar").style.width = `${(share * 100).toFixed(1)}%`;
    if (event.phase === "download") $("load-text").textContent = `Downloading ${mb(event.loaded)} of ${mb(event.total)}`;
    else if (event.phase === "read") $("load-text").textContent = `Reading ${mb(event.loaded)} of ${mb(event.total)}`;
    else if (event.phase === "random") $("load-text").textContent = "Generating random weights…";
  } else if (event.event === "trace") {
    for (const item of event.items) trace.write(item);
    const p0 = event.items[0]?.p0 ?? 0;
    app.emit("trace", { unit: event.unit, units: event.units, p0 });
  } else if (event.event === "prediction") {
    trace.predictions[event.position] = event;
    app.emit("prediction", { position: event.position });
    if (event.position === app.state.position) drawNext();
    drawSurprise(event.position);
  }
});

// ---- bench drawing -----------------------------------------------------------------------------------
function drawSamples() {
  const host = $("prompt-samples");
  host.innerHTML = "";
  for (const text of app.state.card.prompts) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "na-chip";
    chip.textContent = text;
    chip.title = text;
    chip.addEventListener("click", () => { $("prompt-input").value = text; run(); });
    host.append(chip);
  }
}
function drawTokens() {
  const host = $("token-strip");
  const { tokens, position, freshFrom } = app.state;
  host.innerHTML = "";
  tokens.ids.forEach((id, index) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "na-token";
    chip.role = "option";
    chip.dataset.index = index;
    chip.setAttribute("aria-selected", String(index === position));
    chip.toggleAttribute("data-new", freshFrom !== undefined && index >= freshFrom);
    chip.textContent = showPiece(tokens.pieces[index]) || " ";
    chip.title = `token ${index + 1} · id ${id}`;
    host.append(chip);
    drawSurprise(index - 1);
  });
  $("token-note").textContent = tokens.ids.length
    ? app.state.info
      ? "The bar under a token shows how unexpected it was to the model. Dashed tokens were written by the model."
      : "This is how the prompt is split. Load weights to run it."
    : "";
  host.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
}
// position p's prediction says how likely token p + 1 was
function drawSurprise(position) {
  const prediction = app.state.trace.predictions[position];
  const chip = $("token-strip").children[position + 1];
  if (!chip || !prediction || !Number.isFinite(prediction.targetProb)) return;
  const bits = -Math.log2(Math.max(prediction.targetProb, 1e-9));
  chip.style.setProperty("--surprise", Math.min(1, bits / 16).toFixed(3));
  chip.title = `${chip.title.split(" · the model")[0]} · the model gave it ${percent(prediction.targetProb)} (rank ${prediction.targetRank + 1})`;
}
function drawNext() {
  const { trace, position, tokens, info } = app.state;
  const list = $("next-list");
  const prediction = trace.predictions[position];
  list.innerHTML = "";
  const usable = Boolean(info && tokens.ids.length);
  $("step-button").disabled = $("generate-button").disabled = !usable || app.state.generating;
  // only a model that predicts its own next input can extend the prompt
  $("step-button").hidden = $("generate-button").hidden = $("temperature-input").closest("label").hidden = app.state.spec.generates === false;
  $("next-note").textContent = tokens.ids.length && info
    ? `What the model expects after token ${position + 1}, “${showPiece(tokens.pieces[position] ?? "")}”.${info.random ? " With random weights this is noise." : ""}`
    : "What the model expects after the chosen token.";
  if (!prediction) return;
  prediction.pieces.forEach((piece, index) => {
    const item = document.createElement("li");
    const p = prediction.probs[index];
    item.innerHTML = `<span class="na-next-piece"></span><span class="na-next-track"><span class="na-next-fill" style="display:block;width:${(Math.sqrt(p) * 100).toFixed(1)}%"></span></span><span class="na-next-p">${percent(p)}</span>`;
    item.firstChild.textContent = showPiece(piece);
    item.title = `token id ${prediction.ids[index]}`;
    list.append(item);
  });
}
function selectChip() {
  const host = $("token-strip");
  [...host.children].forEach((chip, index) => chip.setAttribute("aria-selected", String(index === app.state.position)));
  host.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
}


// ---- theme ------------------------------------------------------------------------------------------------
function setTheme(choice) {
  if (choice === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  for (const button of $("theme-choice").children) button.setAttribute("aria-pressed", String(button.dataset.choice === choice));
  try { localStorage.setItem("na.theme", choice); } catch { /* storage is optional */ }
}
let themeTimer = 0;
const themeChanged = () => { clearTimeout(themeTimer); themeTimer = setTimeout(() => app.emit("theme"), 30); };
new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeChanged);

// ---- wiring ---------------------------------------------------------------------------------------------------
function boot() {
  const select = $("model-select");
  for (const model of MODELS) select.append(new Option(model.name, model.id));
  select.addEventListener("change", () => selectModel(select.value));
  $("theme-choice").addEventListener("click", (event) => { const choice = event.target.closest("button")?.dataset.choice; if (choice) setTheme(choice); });
  $("load-button").addEventListener("click", () => load({ kind: "url" }));
  $("random-button").addEventListener("click", () => load({ kind: "random" }));
  $("forget-button").addEventListener("click", async () => {
    await app.engine.call("forget", { card: app.state.card });
    app.state.savedCopy = false;
    drawModelPanel();
  });
  $("model-toggle").addEventListener("click", () => { app.state.modelOpen = !app.state.modelOpen; drawModelPanel(); });
  if (native) $("foot-where").textContent = "Runs entirely on this computer: the engine is the Rust program that opened this window.";
  $("weights-file").addEventListener("change", (event) => { const file = event.target.files[0]; if (file) load({ kind: "file", file }); event.target.value = ""; });
  $("run-button").addEventListener("click", run);
  $("prompt-input").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); run(); } });
  // Typing shows the tokens at once and, once weights are in, runs the model
  // when the typing pauses, so any input is followed straight through.
  $("prompt-input").addEventListener("input", () => {
    clearTimeout(typingTimer);
    typingTimer = setTimeout(() => {
      if (app.state.info && !app.state.loading && !app.state.generating) run();
      else retokenize().then((tokens) => { if (tokens) showTokens(tokens); });
    }, app.state.info ? 450 : 160);
  });
  $("token-strip").addEventListener("click", (event) => { const chip = event.target.closest(".na-token"); if (chip) app.setPosition(Number(chip.dataset.index)); });
  $("token-strip").addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") { app.setPosition(app.state.position + 1); event.preventDefault(); }
    if (event.key === "ArrowLeft") { app.setPosition(app.state.position - 1); event.preventDefault(); }
  });
  $("step-button").addEventListener("click", () => step());
  $("generate-button").addEventListener("click", () => generate(12));
  $("stop-button").addEventListener("click", () => { app.state.generating = false; });
  const temperature = $("temperature-input");
  const showTemperature = () => {
    app.state.temperature = Number(temperature.value);
    $("temperature-output").textContent = app.state.temperature === 0 ? "0 · always the most likely token" : `${app.state.temperature.toFixed(2)} · sampled from the 40 most likely`;
    app.emit("settings");
  };
  temperature.addEventListener("input", showTemperature);
  showTemperature();
  app.on("position", () => { selectChip(); drawNext(); });
  // a weights file dropped anywhere on the page
  addEventListener("dragover", (event) => { if (event.dataTransfer?.types.includes("Files")) event.preventDefault(); });
  addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files[0];
    if (!file) return;
    event.preventDefault();
    if (app.state.card.weights.kind === "safetensors") load({ kind: "file", file });
  });

  let theme = "dark", model = params.get("model"), figure = null;
  try {
    theme = localStorage.getItem("na.theme") ?? "dark";
    model = model ?? localStorage.getItem("na.model");
    figure = localStorage.getItem("na.figure");
  } catch { /* storage is optional */ }
  if (params.get("theme")) theme = params.get("theme");
  setTheme(theme);
  const card = modelById(model) ?? MODELS[0];
  const wanted = location.hash.slice(1) || params.get("figure") || figure;
  const index = UI_ARCHS[card.arch].figures.findIndex((f) => f.id === wanted);
  if (index >= 0) activeFigure = index;
  selectModel(card.id);
}

app.engine.ready.then(boot, (error) => {
  showError(`The engine could not start: ${error.message}. Open this page through a web server (node serve.mjs), not as a file.`);
});
