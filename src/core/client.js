// The page's handle on the engine worker: promises for requests, a listener
// list for events.

export class EngineClient {
  constructor({ nowasm = false } = {}) {
    const url = new URL("./worker.js", import.meta.url);
    if (nowasm) url.searchParams.set("nowasm", "1");
    this.worker = new Worker(url, { type: "module" });
    this.pending = new Map();
    this.listeners = new Set();
    this.nextId = 1;
    this.ready = new Promise((resolve, reject) => {
      this.bootOk = resolve;
      this.worker.addEventListener("error", (event) => reject(new Error(event.message || "the engine worker failed to start")), { once: true });
    });
    this.worker.onmessage = ({ data }) => {
      if (data.event) {
        if (data.event === "boot") this.bootOk();
        for (const listener of this.listeners) listener(data);
        return;
      }
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);
      if (data.error) entry.reject(Object.assign(new Error(data.error.message), { code: data.error.code }));
      else entry.resolve(data.result);
    };
  }

  call(method, params, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, method, params }, transfer);
    });
  }

  on(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
