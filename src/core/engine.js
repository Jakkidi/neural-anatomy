// The engine: loads one model and answers questions about it. It knows nothing
// about any particular architecture; everything model-specific comes from the
// architecture module named on the model card.
//
// It runs in a Web Worker (src/core/worker.js) in the browser and directly in
// Node for the tests. `post(event, transfer)` receives progress and traces.

import { Arena } from "./kernels.js";
import { sample, seeded, summarize } from "./sampling.js";
import { forgetSaved, hasSaved, openFile, openUrl } from "./sources.js";
import { WeightStore, readSafetensors } from "./weights.js";

const TOP_K = 10;

// A zero-delay macrotask, so queued messages (a newer run, a cancel) are seen
// between units. setTimeout would add 4 ms per unit.
const tick = (() => {
  if (typeof setImmediate === "function") return () => new Promise((resolve) => setImmediate(resolve));
  const waiting = [];
  const channel = new MessageChannel();
  channel.port1.onmessage = () => waiting.shift()?.();
  return () => new Promise((resolve) => { waiting.push(resolve); channel.port2.postMessage(0); });
})();

export class Engine {
  /**
   * @param {object} options
   * @param {Record<string, object>} options.archs architecture modules by id
   * @param {(path:string, as:"json"|"text") => Promise<any>} options.readAsset
   * @param {(event:object, transfer?:Transferable[]) => void} options.post
   */
  constructor({ archs, readAsset, post, wasm = true }) {
    this.archs = archs;
    this.readAsset = readAsset;
    this.post = post;
    this.wasm = wasm;
    this.model = null;
    this.tokenizers = new Map();
    this.runId = 0;
    this.ids = [];
    this.lastLogits = null;
    this.random = seeded(1);
  }

  #arch(card) {
    const arch = this.archs[card.arch];
    if (!arch) throw new Error(`No architecture called "${card.arch}" is registered.`);
    return arch;
  }

  async #tokenizer(card) {
    const key = JSON.stringify(card.tokenizer);
    if (!this.tokenizers.has(key)) this.tokenizers.set(key, this.#arch(card).createTokenizer(card, this.readAsset));
    return this.tokenizers.get(key);
  }

  /** Splits text into tokens. Works before any weights are loaded. */
  async tokenize({ card, text }) {
    const tokenizer = await this.#tokenizer(card);
    const ids = tokenizer.encode(text);
    return { ids, pieces: ids.map((id) => tokenizer.piece(id)) };
  }

  async pieces({ card, ids }) {
    const tokenizer = await this.#tokenizer(card);
    return ids.map((id) => tokenizer.piece(id));
  }

  /**
   * @param {object} request
   * @param {object} request.card model card
   * @param {{kind:"url"}|{kind:"file",file:Blob}|{kind:"random",seed?:number}} request.source
   */
  async load({ card, source }) {
    const arch = this.#arch(card);
    this.runId++;
    this.model = null; // let the previous arena go before the new one is allocated
    this.ids = [];
    this.lastLogits = null;
    const store = new WeightStore();
    const started = Date.now();
    let arena, config, from, saved = Promise.resolve(false);
    const capacityOf = (c) => arch.spec(c).capacity;
    const sized = (c, tensors) => new Arena(arch.weightFloats(c) + arch.scratchFloats(c, capacityOf(c)) + tensors * 4 + 4096, { wasm: this.wasm });

    if (source.kind === "random") {
      config = { ...card.config };
      this.post({ event: "load", phase: "random", loaded: 0, total: 1 });
      arena = sized(config, 16 * config.n_layer + 16);
      arch.synthesize(config, store, arena, source.seed ?? card.weights?.seed ?? 1);
      from = "random weights";
    } else {
      const opened = source.kind === "file" ? openFile(source.file) : await openUrl(card.weights.url, { expectedBytes: card.weights.bytes });
      from = opened.from;
      saved = opened.saved;
      const phase = opened.from === "download" ? "download" : "read";
      this.post({ event: "load", phase, loaded: 0, total: opened.total });
      try {
        await readSafetensors(
          opened.stream,
          store,
          (header) => {
            const plan = arch.plan(header.tensors, card);
            config = plan.config;
            arena = sized(config, plan.wanted.length);
            return { wanted: plan.wanted, arena };
          },
          (loaded, total) => this.post({ event: "load", phase, loaded, total }),
        );
      } catch (error) {
        await opened.stream.cancel?.().catch(() => {});
        throw error;
      }
    }

    const capacity = capacityOf(config);
    const runtime = arch.createRuntime({ config, store, arena, capacity });
    const tokenizer = await this.#tokenizer(card);
    this.model = { card, arch, config, store, arena, runtime, tokenizer, capacity };
    // A trained model must predict plain English far better than chance. Checked
    // on every real load, so wrong weights or a wiring mistake cannot pass unseen.
    const check = source.kind === "random" ? undefined : await this.#selfCheck(runtime, tokenizer, config);
    const info = {
      model: card.id,
      check,
      config,
      capacity,
      backend: arena.kernels.backend,
      params: arch.paramCount(config),
      from,
      random: source.kind === "random",
      saved: await saved,
      ms: Date.now() - started,
    };
    this.post({ event: "load", phase: "ready", loaded: 1, total: 1 });
    return info;
  }

  async #selfCheck(runtime, tokenizer, config) {
    const prose =
      "The river runs through the middle of the town, and on warm evenings people walk along its banks. There are two bridges, an old stone one near the market and a newer one by the railway station. In winter the water rises, and the path under the old bridge is closed for a few weeks.";
    try {
      const ids = tokenizer.encode(prose).slice(0, runtime.capacity);
      runtime.reset();
      for (const unit of runtime.append(Int32Array.from(ids), () => {})) if (unit % 4 === 0) await tick();
      let nats = 0;
      for (let p = 0; p + 1 < ids.length; p++) nats -= Math.log(Math.max(summarize(runtime.logits(p), 1, ids[p + 1]).targetProb, 1e-30));
      const mean = nats / (ids.length - 1), chance = Math.log(config.vocab_size);
      return { nats: mean, tokens: ids.length - 1, chance, ok: mean < chance * 0.6 };
    } catch {
      return undefined;
    } finally {
      runtime.reset(); // leave a fresh model for the page's own first run
    }
  }

  #need() {
    if (!this.model) throw new Error("No model is loaded yet.");
    return this.model;
  }

  /**
   * Runs the model. With reset, `ids` is the whole sequence; without, the ids
   * are appended to the previous run (one position of work each).
   */
  async run({ ids, reset = true }) {
    const { runtime, capacity, config } = this.#need();
    const run = ++this.runId;
    if (reset) {
      runtime.reset();
      this.ids = [];
    }
    if (runtime.length !== this.ids.length) {
      // a cancelled append left the token list ahead of the cache
      this.ids.length = runtime.length;
    }
    const p0 = runtime.length;
    if (p0 + ids.length > capacity) throw new Error(`A run holds at most ${capacity} tokens; this one has ${p0 + ids.length}.`);
    const started = performance.now();
    this.post({ event: "run", run, p0, n: ids.length, reset });

    let items = [], transfer = [];
    const emit = (key, rowSize, first, count, rows) => {
      const data = rows.slice(); // rows is a view on engine memory that is about to be reused
      items.push({ key, rowSize, p0: first, n: count, data });
      transfer.push(data.buffer);
    };
    const units = config.n_layer + 2;
    for (const unit of runtime.append(Int32Array.from(ids), emit)) {
      this.post({ event: "trace", run, unit, units, items }, transfer);
      items = [];
      transfer = [];
      await tick();
      if (run !== this.runId) return { cancelled: true };
    }
    this.ids.push(...ids);
    const forwardMs = performance.now() - started;

    // What the model predicts after each position, last position first.
    const T = runtime.length;
    const order = [T - 1];
    for (let p = Math.max(0, p0 - 1); p < T - 1; p++) order.push(p);
    for (const position of order) {
      const logits = runtime.logits(position);
      if (position === T - 1) this.lastLogits = logits.slice();
      const target = position + 1 < T ? this.ids[position + 1] : -1;
      const s = summarize(logits, TOP_K, target);
      this.post({
        event: "prediction",
        run,
        position,
        ids: s.ids,
        probs: s.probs,
        pieces: Array.from(s.ids, (id) => this.model.tokenizer.piece(id)),
        entropy: s.entropy,
        targetProb: s.targetProb,
        targetRank: s.targetRank,
      });
      await tick();
      if (run !== this.runId) return { cancelled: true };
    }
    const result = { run, length: T, forwardMs, totalMs: performance.now() - started };
    this.post({ event: "done", ...result });
    return result;
  }

  /** Chooses the next token from the last position's distribution. */
  pick({ temperature = 0, topK = 40, seed } = {}) {
    const { tokenizer } = this.#need();
    if (!this.lastLogits) throw new Error("Run a prompt first.");
    if (seed !== undefined) this.random = seeded(seed);
    const id = sample(this.lastLogits, { temperature, topK, random: this.random });
    return { id, piece: tokenizer.piece(id) };
  }

  /**
   * Reads a residual-stream vector through the output head (the logit lens):
   * what the model would say if it stopped at that point. `target` asks for
   * the probability of one particular token as well.
   */
  readout({ vector, k = 5, target = -1 }) {
    const { runtime, tokenizer } = this.#need();
    if (!runtime.readout) throw new Error("This architecture has no early read-out.");
    const s = summarize(runtime.readout(vector), k, target);
    return {
      ids: s.ids, probs: s.probs, pieces: Array.from(s.ids, (id) => tokenizer.piece(id)),
      entropy: s.entropy, targetProb: s.targetProb, targetRank: s.targetRank,
    };
  }

  /** All of the output head's scores at a position already run. */
  logits({ position }) {
    const { runtime } = this.#need();
    const data = Float32Array.from(runtime.logits(position));
    return { data, transfer: [data.buffer] };
  }

  cancel() {
    this.runId++;
    return true;
  }

  region({ view, rows, cols }) {
    const data = this.#need().store.region(view, rows, cols);
    return { data, transfer: [data.buffer] };
  }

  stats({ view }) {
    return this.#need().store.statsOf(view);
  }

  tensors() {
    return [...this.#need().store.tensors.values()].map(({ name, shape }) => ({ name, shape }));
  }

  async saved({ card }) {
    return card.weights?.url ? hasSaved(card.weights.url) : false;
  }

  async forget() {
    await forgetSaved();
    return true;
  }
}
