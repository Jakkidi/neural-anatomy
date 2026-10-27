// Checks the engine on the real GPT-2 small weights, outside the browser.
//
//   node tools/verify-real.mjs [path/to/model.safetensors]
//
// Without a path it downloads the 548 MB checkpoint from Hugging Face into
// local/gpt2/ first. It then runs two things a wrongly wired model cannot
// fake: it prints the next-token ranking for a prompt with a well-known
// answer, and it measures how surprised the model is by ordinary English.
// GPT-2 small sits near 3 to 4 nats per token on plain prose; a model with a
// bug in it sits near 10.

import { createWriteStream, existsSync, mkdirSync, openAsBlob, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { ENGINE_ARCHS } from "../src/arch/engine-registry.js";
import { Engine } from "../src/core/engine.js";
import { modelById } from "../src/models/index.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const card = modelById("gpt2");
let path = process.argv[2];

if (!path) {
  path = join(ROOT, "local/gpt2", card.weights.file);
  if (!existsSync(path) || statSync(path).size < 500e6) {
    mkdirSync(dirname(path), { recursive: true });
    console.log(`downloading ${card.weights.url}`);
    const response = await fetch(card.weights.url);
    if (!response.ok) throw new Error(`download failed: ${response.status} ${response.statusText}`);
    const total = Number(response.headers.get("content-length")) || card.weights.bytes;
    let got = 0, shown = 0;
    const counter = new TransformStream({
      transform(chunk, controller) {
        got += chunk.length;
        if (got - shown > 25e6) { shown = got; process.stdout.write(`\r  ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`); }
        controller.enqueue(chunk);
      },
    });
    await pipeline(Readable.fromWeb(response.body.pipeThrough(counter)), createWriteStream(path));
    process.stdout.write("\n");
  }
}

const readAsset = async (file, as) => {
  const text = readFileSync(join(ROOT, file), "utf8");
  return as === "json" ? JSON.parse(text) : text;
};
const predictions = [];
const engine = new Engine({ archs: ENGINE_ARCHS, readAsset, post: (event) => { if (event.event === "prediction") predictions[event.position] = event; } });
const info = await engine.load({ card, source: { kind: "file", file: await openAsBlob(path) } });
console.log(`loaded ${info.params.toLocaleString()} weights in ${(info.ms / 1000).toFixed(1)} s (${info.backend})`);
console.log(`config: ${info.config.n_layer} blocks, ${info.config.n_head} heads, width ${info.config.n_embd}`);

let failures = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "ok  " : "FAIL"} ${name}  ${detail}`); if (!ok) failures++; };

// 1. a prompt with a well-known continuation
{
  const prompt = "The Eiffel Tower is located in the city of";
  const { ids } = await engine.tokenize({ card, text: prompt });
  predictions.length = 0;
  const run = await engine.run({ ids, reset: true });
  const last = predictions[ids.length - 1];
  console.log(`\n"${prompt}" (${ids.length} tokens, forward pass ${run.forwardMs.toFixed(0)} ms)`);
  last.pieces.slice(0, 5).forEach((piece, i) => console.log(`  ${(last.probs[i] * 100).toFixed(1).padStart(5)}%  ${JSON.stringify(piece)}`));
  check('" Paris" is among the five likeliest next tokens', last.pieces.slice(0, 5).includes(" Paris"), "");
}

// 2. surprise on ordinary prose
{
  const text =
    "The river runs through the middle of the town, and on warm evenings people walk along its banks. " +
    "There are two bridges, an old stone one near the market and a newer one by the railway station. " +
    "In winter the water rises, and the path under the old bridge is closed for a few weeks.";
  const { ids } = await engine.tokenize({ card, text });
  predictions.length = 0;
  await engine.run({ ids, reset: true });
  let nats = 0, count = 0;
  for (let p = 0; p < ids.length - 1; p++) { nats -= Math.log(predictions[p].targetProb); count++; }
  const mean = nats / count;
  console.log(`\nordinary prose, ${ids.length} tokens: ${mean.toFixed(2)} nats per token (perplexity ${Math.exp(mean).toFixed(0)})`);
  check("surprise is in GPT-2 small's range (under 4.6 nats per token)", mean < 4.6, "");
}

// 3. appending a token must agree with running the whole prompt again
{
  const { ids } = await engine.tokenize({ card, text: "The capital of France is Paris, and the capital of Italy is" });
  predictions.length = 0;
  await engine.run({ ids, reset: true });
  const whole = predictions[ids.length - 1];
  predictions.length = 0;
  await engine.run({ ids: ids.slice(0, -3), reset: true });
  for (const id of ids.slice(-3)) await engine.run({ ids: [id], reset: false });
  const stepped = predictions[ids.length - 1];
  const same = whole.ids[0] === stepped.ids[0] && Math.abs(whole.probs[0] - stepped.probs[0]) < 1e-4;
  check("appending tokens one at a time gives the same prediction", same, `${JSON.stringify(whole.pieces[0])} ${(whole.probs[0] * 100).toFixed(1)}%`);
}

console.log(failures ? `\n${failures} check(s) failed` : "\nthe engine agrees with what GPT-2 small is known to do");
process.exit(failures ? 1 : 0);
