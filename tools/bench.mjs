// Speed of the engine at GPT-2 small's real size, on random weights.
//   node tools/bench.mjs [--js]
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ENGINE_ARCHS } from "../src/arch/engine-registry.js";
import { Engine } from "../src/core/engine.js";
import { modelById } from "../src/models/index.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const readAsset = async (path, as) => { const text = readFileSync(join(ROOT, path), "utf8"); return as === "json" ? JSON.parse(text) : text; };
const wasm = !process.argv.includes("--js");
const engine = new Engine({ archs: ENGINE_ARCHS, readAsset, wasm, post: () => {} });
const card = modelById("gpt2");
let t = performance.now();
const info = await engine.load({ card, source: { kind: "random", seed: 1 } });
console.log(`backend ${info.backend}; ${info.params.toLocaleString()} weights generated in ${((performance.now() - t) / 1000).toFixed(1)} s`);
const { ids } = await engine.tokenize({ card, text: "The Eiffel Tower is located in the city of Paris, which is the capital of France and one of the most visited places in the whole wide world today." });
for (const n of [8, 16, 32]) {
  const r = await engine.run({ ids: ids.slice(0, n), reset: true });
  console.log(`${String(n).padStart(3)} tokens: forward ${r.forwardMs.toFixed(0)} ms, with predictions for every position ${r.totalMs.toFixed(0)} ms`);
}
const one = await engine.run({ ids: [ids[10]], reset: false });
console.log(`append one token: forward ${one.forwardMs.toFixed(0)} ms, total ${one.totalMs.toFixed(0)} ms`);
console.log(`memory: rss ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB`);
