// The engine's home in the browser: a module worker, so a 500 MB download and
// a forward pass never block the page.
//
// Requests:  { id, method, params }
// Replies:   { id, result } or { id, error }
// Events:    { event, ... } posted as work proceeds (see engine.js)

import { ENGINE_ARCHS } from "../arch/engine-registry.js";
import { Engine } from "./engine.js";

const ROOT = new URL("../../", import.meta.url);

async function readAsset(path, as) {
  const response = await fetch(new URL(path, ROOT));
  if (!response.ok) throw new Error(`could not read ${path}: ${response.status}`);
  return as === "json" ? response.json() : response.text();
}

const engine = new Engine({
  archs: ENGINE_ARCHS,
  readAsset,
  post: (event, transfer = []) => self.postMessage(event, transfer),
  wasm: !new URL(self.location.href).searchParams.has("nowasm"),
});

// run() awaits between units, so a second request can arrive mid-run. Requests
// that touch engine state are queued; a new run or a cancel first stops the
// one in flight.
let queue = Promise.resolve();
const SERIAL = new Set(["load", "run", "pick", "readout", "region", "stats", "tensors", "logits"]);

self.onmessage = ({ data }) => {
  const { id, method, params } = data;
  const call = async () => {
    try {
      if (typeof engine[method] !== "function") throw new Error(`unknown method ${method}`);
      const result = await engine[method](params ?? {});
      if (result && result.transfer) {
        const { transfer, ...rest } = result;
        self.postMessage({ id, result: rest }, transfer);
      } else self.postMessage({ id, result });
    } catch (error) {
      self.postMessage({ id, error: { message: error?.message ?? String(error), code: error?.code } });
    }
  };
  if (method === "run" || method === "load") engine.cancel();
  if (SERIAL.has(method)) queue = queue.then(call);
  else call();
};

self.postMessage({ event: "boot" });
