// Memory and hot loops shared by every architecture.
//
// An Arena hands out Float32Array views. When WebAssembly SIMD is available
// the views all sit on one WebAssembly.Memory, so the compiled kernels can be
// handed byte offsets; otherwise each view is its own buffer and the
// plain-JS kernels below do the same arithmetic.

import { KERNELS_WASM_BASE64 } from "./kernels.wasm.js";

const PAGE = 65536;

function decodeBase64(text) {
  if (typeof atob === "function") {
    const raw = atob(text);
    const out = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(text, "base64"));
}

// ---- plain JS kernels -------------------------------------------------------

function affineJS(x, W, b, y, n, nin, nout) {
  for (let t = 0; t < n; t++) {
    const yo = t * nout;
    if (b) for (let j = 0; j < nout; j++) y[yo + j] = b[j];
    else y.fill(0, yo, yo + nout);
  }
  const tail = nout & ~3;
  for (let i = 0; i < nin; i++) {
    const wo = i * nout;
    for (let t = 0; t < n; t++) {
      const a = x[t * nin + i];
      if (a === 0) continue;
      const yo = t * nout;
      let j = 0;
      for (; j < tail; j += 4) {
        y[yo + j] += a * W[wo + j];
        y[yo + j + 1] += a * W[wo + j + 1];
        y[yo + j + 2] += a * W[wo + j + 2];
        y[yo + j + 3] += a * W[wo + j + 3];
      }
      for (; j < nout; j++) y[yo + j] += a * W[wo + j];
    }
  }
}

function dotJS(a, b, n) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  const tail = n & ~3;
  let j = 0;
  for (; j < tail; j += 4) {
    s0 += a[j] * b[j];
    s1 += a[j + 1] * b[j + 1];
    s2 += a[j + 2] * b[j + 2];
    s3 += a[j + 3] * b[j + 3];
  }
  for (; j < n; j++) s0 += a[j] * b[j];
  return s0 + s1 + s2 + s3;
}

function matvecJS(E, h, out, rows, n) {
  const tail = n & ~3;
  for (let r = 0; r < rows; r++) {
    const o = r * n;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    let j = 0;
    for (; j < tail; j += 4) {
      s0 += E[o + j] * h[j];
      s1 += E[o + j + 1] * h[j + 1];
      s2 += E[o + j + 2] * h[j + 2];
      s3 += E[o + j + 3] * h[j + 3];
    }
    for (; j < n; j++) s0 += E[o + j] * h[j];
    out[r] = s0 + s1 + s2 + s3;
  }
}

const JS_KERNELS = {
  backend: "js",
  affine: affineJS,
  matvec: matvecJS,
  dot: dotJS,
};

// ---- arena ------------------------------------------------------------------

export class Arena {
  /** @param {number} floats total Float32 slots this model will ever need */
  constructor(floats, { wasm = true } = {}) {
    this.floats = floats;
    this.used = 0;
    this.memory = null;
    this.base = 0;
    this.kernels = JS_KERNELS;
    if (wasm) this.#tryWasm(floats);
  }

  #tryWasm(floats) {
    try {
      const module = new WebAssembly.Module(decodeBase64(KERNELS_WASM_BASE64));
      const reserve = 2 * PAGE; // the module's stack and data live at the bottom
      const pages = Math.ceil((reserve + floats * 4 + 64) / PAGE);
      if (pages > 65535) return;
      const memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
      const instance = new WebAssembly.Instance(module, { env: { memory } });
      const heap = instance.exports.__heap_base.value;
      this.memory = memory;
      this.base = Math.ceil(Math.max(heap, 1024) / 16) * 16;
      const ex = instance.exports;
      const off = (view) => (view ? view.byteOffset : 0);
      this.kernels = {
        backend: "wasm-simd",
        affine: (x, W, b, y, n, nin, nout) => ex.affine(off(x), off(W), off(b), off(y), n, nin, nout),
        matvec: (E, h, out, rows, n) => ex.matvec(off(E), off(h), off(out), rows, n),
        dot: (a, b, n) => ex.dot(off(a), off(b), n),
      };
    } catch {
      this.memory = null;
      this.kernels = JS_KERNELS;
    }
  }

  /** A zeroed Float32Array of `n` slots. */
  alloc(n) {
    if (!this.memory) return new Float32Array(n);
    const bytes = Math.ceil((n * 4) / 16) * 16;
    const at = this.base + this.used;
    if (at + bytes > this.memory.buffer.byteLength) throw new Error(`arena full: wanted ${n} floats`);
    this.used += bytes;
    return new Float32Array(this.memory.buffer, at, n);
  }
}

export { JS_KERNELS };
