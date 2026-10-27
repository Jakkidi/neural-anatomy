// A trace is everything one run exposed, as named arrays whose first axis is
// the position in the sequence: trace.row("block.3.mlp.post", 5) is the 3072
// neuron activations of block 3 at token 5.
//
// The engine sends rows as it computes them; the page keeps them here.

export class Trace {
  /** @param {number} capacity most positions a run can hold */
  constructor(capacity) {
    this.capacity = capacity;
    this.length = 0; // positions fully run
    this.arrays = new Map(); // key -> { rowSize, data, filled }
    this.predictions = []; // per position: { ids, probs, pieces, entropy, targetProb, targetRank }
    this.unitsDone = -1; // highest unit index received for the newest rows
    this.version = 0;
  }

  clear() {
    this.length = 0;
    for (const entry of this.arrays.values()) { entry.filled = 0; entry.version++; }
    this.predictions = [];
    this.unitsDone = -1;
    this.version++;
  }

  write({ key, rowSize, p0, n, data }) {
    let entry = this.arrays.get(key);
    if (!entry || entry.rowSize !== rowSize) {
      entry = { rowSize, data: new Float32Array(this.capacity * rowSize), filled: 0, version: 0 };
      this.arrays.set(key, entry);
    }
    entry.data.set(data, p0 * rowSize);
    entry.filled = Math.max(entry.filled, p0 + n);
    entry.version++;
    this.version++;
  }

  has(key, position = 0) {
    const entry = this.arrays.get(key);
    return Boolean(entry) && position < entry.filled;
  }

  /** One position's row of an array, or null if it has not been computed. */
  row(key, position) {
    const entry = this.arrays.get(key);
    if (!entry || position < 0 || position >= entry.filled) return null;
    return entry.data.subarray(position * entry.rowSize, (position + 1) * entry.rowSize);
  }

  /** Everything computed so far for a key: { data, rowSize, filled, version }. */
  entry(key) {
    return this.arrays.get(key) ?? null;
  }

  rowSize(key) {
    return this.arrays.get(key)?.rowSize ?? 0;
  }
}
