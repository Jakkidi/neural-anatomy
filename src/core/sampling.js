// Turning logits into probabilities, a ranking, and a sampled token.

/**
 * @param {Float32Array} logits
 * @param {number} k
 * @param {number} [target] a token id whose probability is wanted too
 * @returns {{ids:Int32Array, probs:Float32Array, entropy:number, targetProb:number, targetRank:number}}
 */
export function summarize(logits, k, target = -1) {
  const n = logits.length;
  let top = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > top) top = logits[i];
  let z = 0, weighted = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - top);
    z += e;
    weighted += e * (logits[i] - top);
  }
  // H = log Z - E[l - top]
  const entropy = Math.log(z) - weighted / z;

  const ids = new Int32Array(k).fill(-1);
  const best = new Float32Array(k).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (v <= best[k - 1]) continue;
    let at = k - 1;
    while (at > 0 && best[at - 1] < v) {
      best[at] = best[at - 1];
      ids[at] = ids[at - 1];
      at--;
    }
    best[at] = v;
    ids[at] = i;
  }
  const probs = new Float32Array(k);
  for (let i = 0; i < k; i++) probs[i] = Math.exp(best[i] - top) / z;

  let targetProb = NaN, targetRank = -1;
  if (target >= 0 && target < n) {
    const v = logits[target];
    targetProb = Math.exp(v - top) / z;
    targetRank = 0;
    for (let i = 0; i < n; i++) if (logits[i] > v) targetRank++;
  }
  return { ids, probs, entropy, targetProb, targetRank };
}

/** mulberry32: small seeded generator so a sampled run can be repeated. */
export function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Picks the next token. temperature 0 means the most likely token.
 * topK limits the choice to the k most likely tokens (0 = no limit).
 */
export function sample(logits, { temperature = 1, topK = 40, random = Math.random } = {}) {
  const n = logits.length;
  if (!(temperature > 0)) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  let ids, values;
  if (topK > 0 && topK < n) {
    const picked = summarize(logits, topK);
    ids = picked.ids;
    values = Float32Array.from(ids, (id) => logits[id]);
  } else {
    ids = null;
    values = logits;
  }
  let top = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i] > top) top = values[i];
  let z = 0;
  const weights = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    weights[i] = Math.exp((values[i] - top) / temperature);
    z += weights[i];
  }
  let r = random() * z;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return ids ? ids[i] : i;
  }
  return ids ? ids[weights.length - 1] : weights.length - 1;
}
