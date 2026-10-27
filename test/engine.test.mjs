// Checks the engine against the independent numpy reference, on a
// GPT-2-shaped checkpoint of random numbers, with both kernel backends and
// with tokens appended one at a time.
//
//   node test/engine.test.mjs

import { execFileSync } from "node:child_process";
import { mkdtempSync, openAsBlob, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ENGINE_ARCHS } from "../src/arch/engine-registry.js";
import { Engine } from "../src/core/engine.js";
import { Trace } from "../src/core/trace.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const work = mkdtempSync(join(tmpdir(), "na-test-"));
const weights = join(work, "tiny.safetensors");
const HEADS = 4, WIDTH = 64, LAYERS = 3, VOCAB = 600, CTX = 48;
const ids = [5, 17, 301, 42, 599, 0, 77, 301, 12, 256, 9];

execFileSync("node", [join(ROOT, "tools/make-test-weights.mjs"), weights, "--layers", LAYERS, "--heads", HEADS, "--width", WIDTH, "--vocab", VOCAB, "--ctx", CTX, "--masks", "--prefix", "transformer."].map(String), { stdio: "inherit" });
execFileSync("python3", [join(ROOT, "tools/reference_gpt2.py"), weights, String(HEADS), ids.join(","), join(work, "ref.json")], { stdio: "inherit" });
const reference = JSON.parse(readFileSync(join(work, "ref.json"), "utf8"));

const card = {
  id: "tiny", arch: "gpt2",
  config: { n_layer: LAYERS, n_head: HEADS, n_embd: WIDTH, n_inner: 4 * WIDTH, n_ctx: CTX, vocab_size: VOCAB, layer_norm_epsilon: 1e-5 },
  tokenizer: { vocab: "assets/gpt2/vocab.json", merges: "assets/gpt2/merges.txt" },
};
const readAsset = async (path, as) => {
  const text = readFileSync(join(ROOT, path), "utf8");
  return as === "json" ? JSON.parse(text) : text;
};

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
};

async function runEngine({ wasm, chunks }) {
  let trace = null;
  const predictions = [];
  const engine = new Engine({
    archs: ENGINE_ARCHS, readAsset, wasm,
    post: (event) => {
      if (event.event === "trace") for (const item of event.items) trace.write(item);
      if (event.event === "prediction") predictions[event.position] = event;
    },
  });
  const info = await engine.load({ card, source: { kind: "file", file: await openAsBlob(weights) } });
  trace = new Trace(info.capacity);
  let first = true;
  for (const chunk of chunks) {
    await engine.run({ ids: chunk, reset: first });
    first = false;
  }
  return { engine, info, trace, predictions };
}

function compare(label, { trace, info, engine }) {
  const T = ids.length, C = info.capacity, H = HEADS;
  let worst = 0, worstKey = "";
  for (const [key, expected] of Object.entries(reference)) {
    if (key === "logits") continue;
    const rowSize = trace.rowSize(key);
    if (!rowSize) { check(`${label}: trace has ${key}`, false); continue; }
    let diff = 0;
    if (key.endsWith("attn.pattern")) {
      // engine rows are [H, capacity]; the reference is [T, H, T]
      for (let t = 0; t < T; t++) {
        const row = trace.row(key, t);
        for (let h = 0; h < H; h++) for (let j = 0; j < C; j++) {
          const want = j < T ? expected[(t * H + h) * T + j] : 0;
          diff = Math.max(diff, Math.abs(row[h * C + j] - want));
        }
      }
    } else {
      for (let t = 0; t < T; t++) {
        const row = trace.row(key, t);
        for (let j = 0; j < rowSize; j++) diff = Math.max(diff, Math.abs(row[j] - expected[t * rowSize + j]));
      }
    }
    if (diff > worst) { worst = diff; worstKey = key; }
  }
  check(`${label}: every traced value matches the numpy reference`, worst < 2e-4, `(largest difference ${worst.toExponential(2)} in ${worstKey})`);
  let logitDiff = 0;
  for (let t = 0; t < T; t++) {
    const logits = engine.model.runtime.logits(t);
    for (let v = 0; v < VOCAB; v++) logitDiff = Math.max(logitDiff, Math.abs(logits[v] - reference.logits[t * VOCAB + v]));
  }
  check(`${label}: logits match at every position`, logitDiff < 5e-4, `(largest difference ${logitDiff.toExponential(2)})`);
}

const whole = await runEngine({ wasm: true, chunks: [ids] });
check("WebAssembly kernels are in use", whole.info.backend === "wasm-simd", `(${whole.info.backend})`);
check("config read from the checkpoint", whole.info.config.n_layer === LAYERS && whole.info.config.n_embd === WIDTH && whole.info.config.vocab_size === VOCAB && whole.info.config.n_inner === 4 * WIDTH);
compare("wasm, whole prompt", whole);

const plain = await runEngine({ wasm: false, chunks: [ids] });
check("plain JS fallback is in use", plain.info.backend === "js");
compare("js, whole prompt", plain);

const stepped = await runEngine({ wasm: true, chunks: [ids.slice(0, 4), ...ids.slice(4).map((id) => [id])] });
compare("wasm, tokens appended one at a time", stepped);

// predictions: the probability of the actual next token and the ranking
{
  const T = ids.length;
  let ok = true, detail = "";
  for (let t = 0; t < T; t++) {
    const row = reference.logits.slice(t * VOCAB, (t + 1) * VOCAB);
    const top = Math.max(...row);
    const z = row.reduce((sum, v) => sum + Math.exp(v - top), 0);
    const best = row.indexOf(top);
    const got = stepped.predictions[t];
    if (!got || got.ids[0] !== best || Math.abs(got.probs[0] - 1 / z) > 1e-4) { ok = false; detail = `position ${t}`; }
    if (t + 1 < T) {
      const want = Math.exp(row[ids[t + 1]] - top) / z;
      if (Math.abs(got.targetProb - want) > 1e-4) { ok = false; detail = `target at position ${t}`; }
    }
  }
  check("top prediction and next-token probability at every position", ok, detail);
  const greedy = stepped.engine.pick({ temperature: 0 });
  const lastRow = reference.logits.slice((T - 1) * VOCAB);
  check("greedy pick is the arg-max", greedy.id === lastRow.indexOf(Math.max(...lastRow)));
}

// weight regions: exact when not shrunk, max-magnitude when shrunk
{
  const store = whole.engine.model.store;
  const view = { tensor: "h.1.attn.c_attn.weight", row0: 0, col0: WIDTH, rows: WIDTH, cols: WIDTH };
  const exact = store.region(view, WIDTH, WIDTH);
  const raw = store.get("h.1.attn.c_attn.weight");
  let ok = true;
  for (let r = 0; r < WIDTH; r++) for (let c = 0; c < WIDTH; c++) if (exact[r * WIDTH + c] !== raw[r * 3 * WIDTH + WIDTH + c]) ok = false;
  check("an unshrunk region returns the exact weights", ok);
  const pooled = store.region(view, WIDTH / 4, WIDTH / 4);
  ok = true;
  for (let R = 0; R < WIDTH / 4; R++) for (let Cc = 0; Cc < WIDTH / 4; Cc++) {
    let best = 0;
    for (let r = R * 4; r < R * 4 + 4; r++) for (let c = Cc * 4; c < Cc * 4 + 4; c++) {
      const v = raw[r * 3 * WIDTH + WIDTH + c];
      if (Math.abs(v) > Math.abs(best)) best = v;
    }
    if (pooled[R * (WIDTH / 4) + Cc] !== best) ok = false;
  }
  check("a shrunk region keeps the largest-magnitude weight of each block", ok);
  const tview = { tensor: "h.0.mlp.c_proj.weight", row0: 0, col0: 0, rows: WIDTH, cols: 4 * WIDTH, transpose: true };
  const t = store.region(tview, WIDTH, 4 * WIDTH);
  const rawT = store.get("h.0.mlp.c_proj.weight");
  ok = true;
  for (let r = 0; r < WIDTH; r++) for (let c = 0; c < 4 * WIDTH; c++) if (t[r * 4 * WIDTH + c] !== rawT[c * WIDTH + r]) ok = false;
  check("a transposed view reads the right entries", ok);
  const s = store.statsOf(view);
  check("statistics are finite", Number.isFinite(s.std) && s.std > 0 && s.min < 0 && s.max > 0);
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall engine checks passed");
process.exit(failures ? 1 : 0);
