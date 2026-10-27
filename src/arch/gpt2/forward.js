// The GPT-2 forward pass, written out so every intermediate value can be
// handed to the interface.
//
//   x = W_E[token] + W_pos[position]
//   for each block:
//     x = x + Attention(LayerNorm1(x))        attention writes into the stream
//     x = x + MLP(LayerNorm2(x))              the MLP writes into the stream
//   logits = LayerNorm_f(x) @ W_E^T
//
// Keys and values are cached per block, so appending one token costs one
// position, not the whole prompt again.

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

function layerNorm(x, gain, bias, out, n, d, eps) {
  for (let t = 0; t < n; t++) {
    const o = t * d;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += x[o + j];
    mean /= d;
    let variance = 0;
    for (let j = 0; j < d; j++) {
      const c = x[o + j] - mean;
      variance += c * c;
    }
    const inv = 1 / Math.sqrt(variance / d + eps);
    for (let j = 0; j < d; j++) out[o + j] = (x[o + j] - mean) * inv * gain[j] + bias[j];
  }
}

// GPT-2's GELU is the tanh approximation ("gelu_new").
function gelu(x, n) {
  for (let i = 0; i < n; i++) {
    const v = x[i];
    x[i] = 0.5 * v * (1 + Math.tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)));
  }
}

export function createGPT2Runtime({ config, store, arena, capacity }) {
  const { n_layer: L, n_head: H, n_embd: d, n_inner: m, vocab_size: V } = config;
  const eps = config.layer_norm_epsilon ?? 1e-5;
  const hd = d / H;
  const C = capacity;
  const K = arena.kernels;
  const scale = 1 / Math.sqrt(hd);

  const W = {
    wte: store.get("wte.weight"),
    wpe: store.get("wpe.weight"),
    lnf_g: store.get("ln_f.weight"),
    lnf_b: store.get("ln_f.bias"),
    blocks: Array.from({ length: L }, (_, l) => {
      const p = `h.${l}`;
      return {
        ln1_g: store.get(`${p}.ln_1.weight`), ln1_b: store.get(`${p}.ln_1.bias`),
        qkv_w: store.get(`${p}.attn.c_attn.weight`), qkv_b: store.get(`${p}.attn.c_attn.bias`),
        o_w: store.get(`${p}.attn.c_proj.weight`), o_b: store.get(`${p}.attn.c_proj.bias`),
        ln2_g: store.get(`${p}.ln_2.weight`), ln2_b: store.get(`${p}.ln_2.bias`),
        in_w: store.get(`${p}.mlp.c_fc.weight`), in_b: store.get(`${p}.mlp.c_fc.bias`),
        out_w: store.get(`${p}.mlp.c_proj.weight`), out_b: store.get(`${p}.mlp.c_proj.bias`),
      };
    }),
  };

  // per-position state that persists between appends
  const keys = Array.from({ length: L }, () => arena.alloc(C * d));
  const values = Array.from({ length: L }, () => arena.alloc(C * d));
  const final = arena.alloc(C * d);
  // work rows for the positions being appended
  const x = arena.alloc(C * d);
  const normed = arena.alloc(C * d);
  const qkv = arena.alloc(C * 3 * d);
  const mixed = arena.alloc(C * d);
  const write = arena.alloc(C * d);
  const hidden = arena.alloc(C * m);
  const pattern = arena.alloc(C * H * C);
  const logits = arena.alloc(V);
  const scores = new Float32Array(C);

  let T = 0;

  return {
    capacity: C,
    config,
    get length() { return T; },
    reset() { T = 0; },

    /**
     * Runs the model on `ids`, appended after what is already cached.
     * emit(key, rowSize, firstPosition, count, rows) is called with a view
     * that is only valid during the call. A generator: it yields the index of
     * each unit as it finishes (0 = embedding, 1..L = blocks, L + 1 = output),
     * so the caller can report progress and stop between units.
     */
    *append(ids, emit) {
      const n = ids.length, p0 = T;
      if (p0 + n > C) throw new Error(`this run holds at most ${C} tokens`);
      const rows = (buffer, size) => buffer.subarray(0, n * size);

      // embedding
      for (let t = 0; t < n; t++) {
        const id = ids[t];
        if (!(id >= 0 && id < V)) throw new Error(`token id ${id} is outside the vocabulary`);
        const tok = id * d, pos = (p0 + t) * d, o = t * d;
        for (let j = 0; j < d; j++) {
          normed[o + j] = W.wte[tok + j];
          mixed[o + j] = W.wpe[pos + j];
          x[o + j] = W.wte[tok + j] + W.wpe[pos + j];
        }
      }
      emit("embed.tok", d, p0, n, rows(normed, d));
      emit("embed.pos", d, p0, n, rows(mixed, d));
      emit("resid.0", d, p0, n, rows(x, d));
      yield 0;

      for (let l = 0; l < L; l++) {
        const B = W.blocks[l];
        const Kl = keys[l], Vl = values[l];

        // attention
        layerNorm(x, B.ln1_g, B.ln1_b, normed, n, d, eps);
        K.affine(normed, B.qkv_w, B.qkv_b, qkv, n, d, 3 * d);
        for (let t = 0; t < n; t++) {
          const src = t * 3 * d, dst = (p0 + t) * d;
          for (let j = 0; j < d; j++) {
            Kl[dst + j] = qkv[src + d + j];
            Vl[dst + j] = qkv[src + 2 * d + j];
          }
        }
        pattern.fill(0, 0, n * H * C);
        for (let t = 0; t < n; t++) {
          const p = p0 + t; // this position may look at 0..p
          for (let h = 0; h < H; h++) {
            const q = t * 3 * d + h * hd;
            let top = -Infinity;
            for (let j = 0; j <= p; j++) {
              const k = j * d + h * hd;
              let s = 0;
              for (let e = 0; e < hd; e++) s += qkv[q + e] * Kl[k + e];
              s *= scale;
              scores[j] = s;
              if (s > top) top = s;
            }
            let sum = 0;
            for (let j = 0; j <= p; j++) {
              const e = Math.exp(scores[j] - top);
              scores[j] = e;
              sum += e;
            }
            const row = (t * H + h) * C;
            const out = t * d + h * hd;
            for (let e = 0; e < hd; e++) mixed[out + e] = 0;
            for (let j = 0; j <= p; j++) {
              const a = scores[j] / sum;
              pattern[row + j] = a;
              const v = j * d + h * hd;
              for (let e = 0; e < hd; e++) mixed[out + e] += a * Vl[v + e];
            }
          }
        }
        emit(`block.${l}.attn.pattern`, H * C, p0, n, rows(pattern, H * C));
        K.affine(mixed, B.o_w, B.o_b, write, n, d, d);
        emit(`block.${l}.attn.out`, d, p0, n, rows(write, d));
        for (let i = 0; i < n * d; i++) x[i] += write[i];

        // MLP
        layerNorm(x, B.ln2_g, B.ln2_b, normed, n, d, eps);
        K.affine(normed, B.in_w, B.in_b, hidden, n, d, m);
        gelu(hidden, n * m);
        emit(`block.${l}.mlp.post`, m, p0, n, rows(hidden, m));
        K.affine(hidden, B.out_w, B.out_b, write, n, m, d);
        emit(`block.${l}.mlp.out`, d, p0, n, rows(write, d));
        for (let i = 0; i < n * d; i++) x[i] += write[i];
        emit(`block.${l}.resid`, d, p0, n, rows(x, d));
        yield l + 1;
      }

      layerNorm(x, W.lnf_g, W.lnf_b, normed, n, d, eps);
      final.set(rows(normed, d), p0 * d);
      emit("final.resid", d, p0, n, rows(normed, d));
      T = p0 + n;
      yield L + 1;
    },

    /**
     * Reads any residual-stream vector as if it were the final one: the final
     * LayerNorm, then the unembedding. Applied to a block's stream this is the
     * "logit lens": what the model would say if it stopped there.
     */
    readout(vector) {
      const row = arena.kernels.backend === "js" ? new Float32Array(d) : normed.subarray(0, d);
      layerNorm(vector, W.lnf_g, W.lnf_b, row, 1, d, eps);
      K.matvec(W.wte, row, logits, V, d);
      return logits;
    },

    /** Next-token logits at a position already run. The array is reused by the next call. */
    logits(position) {
      if (!(position >= 0 && position < T)) throw new Error(`position ${position} has not been run`);
      K.matvec(W.wte, final.subarray(position * d, (position + 1) * d), logits, V, d);
      return logits;
    },
  };
}

/** Float32 slots createGPT2Runtime allocates, so the arena can be sized first. */
export function gpt2ScratchFloats(config, capacity) {
  const { n_layer: L, n_head: H, n_embd: d, n_inner: m, vocab_size: V } = config;
  const C = capacity;
  const slack = 64 * (2 * L + 12); // arena rounds every allocation up to 16 bytes
  return L * 2 * C * d + C * d * 5 + C * 3 * d + C * m + C * H * C + V + slack;
}
