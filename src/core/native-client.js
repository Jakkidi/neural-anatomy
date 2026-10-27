// The page's handle on the native engine of the desktop app.
//
// It has the same shape as EngineClient (the Web Worker one): call(method,
// params) returns a promise and on(listener) receives progress events, so the
// rest of the page does not know which engine it is talking to. Here the
// engine is the Rust program that served this page, reached over HTTP on the
// loopback address (desktop/app/src/api.rs).

const viewQuery = (view) =>
  `tensor=${encodeURIComponent(view.tensor)}&row0=${view.row0 ?? 0}&col0=${view.col0 ?? 0}&rows=${view.rows}&cols=${view.cols}&transpose=${view.transpose ? 1 : 0}`;

function failure(payload, fallback) {
  return Object.assign(new Error(payload?.error?.message ?? fallback), { code: payload?.error?.code });
}

export class NativeClient {
  constructor() {
    this.listeners = new Set();
    this.native = true;
    this.savedPath = "";
    this.running = null; // AbortController of the run in flight
    this.ready = Promise.resolve();
    this.#holdSession();
  }

  on(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  #emit(event) {
    for (const listener of this.listeners) listener(event);
  }

  // The app quits when no window holds a session, so keep one open for as long
  // as this page lives, and take it up again if it drops.
  async #holdSession() {
    for (;;) {
      try {
        const response = await fetch("/api/session");
        const reader = response.body.getReader();
        for (;;) if ((await reader.read()).done) break;
      } catch { /* the app is gone, or will answer again in a moment */ }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  }

  async #json(path, body, init = {}) {
    let response;
    try {
      response = await fetch(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body), ...init });
    } catch {
      throw new Error("The app's engine did not answer. Has the program been closed?");
    }
    const payload = await response.json().catch(() => null);
    if (!response.ok || !payload || payload.error) throw failure(payload, `The engine refused the request (${response.status}).`);
    return payload.result;
  }

  call(method, params = {}) {
    const handler = this[`_${method}`];
    if (!handler) return Promise.reject(new Error(`The native engine has no "${method}".`));
    return handler.call(this, params);
  }

  async _saved({ card }) {
    const result = await this.#json("/api/saved", { card });
    this.savedPath = result.path ?? "";
    return Boolean(result.saved);
  }

  _forget({ card }) {
    return this.#json("/api/forget", { card });
  }

  _tokenize({ card, text }) {
    return this.#json("/api/tokenize", { card, text });
  }

  _pieces({ card, ids }) {
    return this.#json("/api/pieces", { card, ids });
  }

  _pick(params) {
    return this.#json("/api/pick", params);
  }

  _stats({ view }) {
    return this.#json(`/api/stats?${viewQuery(view)}`, undefined, { method: "GET" });
  }

  _tensors() {
    return this.#json("/api/tensors", undefined, { method: "GET" });
  }

  async _region({ view, rows, cols }) {
    const response = await fetch(`/api/region?${viewQuery(view)}&outRows=${rows}&outCols=${cols}`);
    if (!response.ok) throw failure(await response.json().catch(() => null), "The weights could not be read.");
    return { data: new Float32Array(await response.arrayBuffer()) };
  }

  async _readout({ vector, k = 5, target = -1 }) {
    const bytes = new Uint8Array(vector.buffer.slice(vector.byteOffset, vector.byteOffset + vector.byteLength));
    const result = await this.#json(`/api/readout?k=${k}${target >= 0 ? `&target=${target}` : ""}`, undefined, { body: bytes });
    return { ...result, targetProb: result.targetProb ?? NaN };
  }

  async _logits({ position }) {
    const response = await fetch(`/api/logits?position=${position}`);
    if (!response.ok) throw failure(await response.json().catch(() => null), "The scores could not be read.");
    return { data: new Float32Array(await response.arrayBuffer()) };
  }

  _cancel() {
    this.running?.abort();
    return Promise.resolve(true);
  }

  // ---- loading ----------------------------------------------------------------

  async _load({ card, source }) {
    this.running?.abort();
    if (source.kind === "file") return this.#upload(card, source.file);
    let response;
    try {
      response = await fetch("/api/load", { method: "POST", body: JSON.stringify({ card, source: { kind: source.kind, seed: source.seed } }) });
    } catch {
      throw new Error("The app's engine did not answer. Has the program been closed?");
    }
    if (!response.ok) throw failure(await response.json().catch(() => null), "The engine refused to load.");
    // progress lines, then one line with the result or the error
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let text = "", last = null;
    for (;;) {
      const { done, value } = await reader.read();
      text += decoder.decode(value ?? new Uint8Array(0), { stream: !done });
      let end;
      while ((end = text.indexOf("\n")) >= 0) {
        const line = text.slice(0, end);
        text = text.slice(end + 1);
        if (!line) continue;
        const message = JSON.parse(line);
        if (message.event) this.#emit(message);
        else last = message;
      }
      if (done) break;
    }
    if (!last || last.error) throw failure(last, "Loading stopped before it finished.");
    return last.result;
  }

  // A checkpoint chosen in the page is handed to the engine as the body of one
  // request. XMLHttpRequest is used because it reports how much has been sent.
  #upload(card, file) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", `/api/load-file?card=${encodeURIComponent(JSON.stringify(card))}`);
      request.responseType = "json";
      request.upload.onprogress = (event) => this.#emit({ event: "load", phase: "read", loaded: event.loaded, total: event.total || file.size });
      request.onload = () => {
        const payload = request.response;
        if (request.status === 200 && payload?.result) resolve(payload.result);
        else reject(failure(payload, "The file could not be loaded."));
      };
      request.onerror = () => reject(new Error("The app's engine did not answer. Has the program been closed?"));
      this.#emit({ event: "load", phase: "read", loaded: 0, total: file.size });
      request.send(file);
    });
  }

  // ---- running ------------------------------------------------------------------

  async _run({ ids, reset = true }) {
    this.running?.abort();
    const control = (this.running = new AbortController());
    let response;
    try {
      response = await fetch("/api/run", { method: "POST", body: JSON.stringify({ ids: Array.from(ids), reset }), signal: control.signal });
    } catch (error) {
      if (control.signal.aborted) return { cancelled: true };
      throw new Error("The app's engine did not answer. Has the program been closed?");
    }
    if (!response.ok) throw failure(await response.json().catch(() => null), "The engine refused to run.");
    // Frames: [header bytes: u32][payload bytes: u32][JSON header][float32 payload]
    const reader = response.body.getReader();
    let pending = new Uint8Array(0);
    let result = null;
    const take = () => {
      for (;;) {
        if (pending.length < 8) return;
        const sizes = new DataView(pending.buffer, pending.byteOffset, 8);
        const headerLength = sizes.getUint32(0, true), payloadLength = sizes.getUint32(4, true);
        if (pending.length < 8 + headerLength + payloadLength) return;
        const header = JSON.parse(new TextDecoder().decode(pending.subarray(8, 8 + headerLength)));
        // copied, so the floats start on a 4-byte boundary and outlive `pending`
        const payload = pending.slice(8 + headerLength, 8 + headerLength + payloadLength).buffer;
        pending = pending.subarray(8 + headerLength + payloadLength);
        if (header.event === "trace") {
          for (const item of header.items) {
            item.data = new Float32Array(payload, item.offset, item.length);
            delete item.offset;
            delete item.length;
          }
          this.#emit(header);
        } else if (header.event === "prediction") {
          header.targetProb = header.targetProb ?? NaN;
          this.#emit(header);
        } else result = header;
      }
    };
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (value) {
          const joined = new Uint8Array(pending.length + value.length);
          joined.set(pending);
          joined.set(value, pending.length);
          pending = joined;
          take();
        }
        if (done) break;
      }
    } catch (error) {
      if (control.signal.aborted) return { cancelled: true };
      throw new Error("The run was cut short: the app's engine stopped answering.");
    }
    if (this.running === control) this.running = null;
    if (!result) throw new Error("The run ended without a result.");
    if (result.error) throw failure(result, "The run failed.");
    return result.result;
  }
}
