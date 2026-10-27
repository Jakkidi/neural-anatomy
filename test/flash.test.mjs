// FlashAttention, as the GPU view runs it, against plain softmax attention:
// causal and not, tile sizes that do and do not divide the length.
import { flashOutput, plan, trafficAt } from "../src/figures/rack/bench/gpu/flash.js";

let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;

function head(T, s, causal) {
  const Q = Float32Array.from({ length: T * s }, () => rand() * 2);
  const K = Float32Array.from({ length: T * s }, () => rand() * 2);
  const V = Float32Array.from({ length: T * s }, rand);
  const scale = 1 / Math.sqrt(s);
  const shares = new Float32Array(T * T), out = new Float32Array(T * s);
  for (let i = 0; i < T; i++) {
    const last = causal ? i : T - 1;
    const e = [];
    for (let j = 0; j <= last; j++) { let d = 0; for (let x = 0; x < s; x++) d += Q[i * s + x] * K[j * s + x]; e.push(d * scale); }
    const top = Math.max(...e), z = e.reduce((a, v) => a + Math.exp(v - top), 0);
    e.forEach((v, j) => { shares[i * T + j] = Math.exp(v - top) / z; for (let x = 0; x < s; x++) out[i * s + x] += shares[i * T + j] * V[j * s + x]; });
  }
  return { T, s, Q, K, V, scale, causal, shares, out };
}

let failed = 0;
for (const causal of [true, false]) for (const T of [1, 7, 16, 37]) for (const B of [2, 4, 8, 64]) {
  const { worst } = flashOutput(head(T, 8, causal), B);
  const p = plan({ T, d: 8, B, causal, method: "flash" });
  const t = trafficAt(T, 8, B, causal);
  const counted = p.read[p.ticks - 1] + p.write[p.ticks - 1];
  const q = plan({ T, d: 8, B, causal, method: "standard" });
  const ok = worst < 1e-5 && counted === t.flash && q.read[q.ticks - 1] + q.write[q.ticks - 1] === t.standard;
  if (!ok) { failed++; console.log(`FAIL causal=${causal} T=${T} B=${B}: difference ${worst}, traffic ${counted} vs ${t.flash}`); }
}
console.log(failed ? `${failed} flash checks failed` : "ok   FlashAttention equals softmax attention; traffic counts agree");
if (failed) process.exit(1);
