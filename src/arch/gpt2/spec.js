// What a GPT-2 looks like to the interface: its units from input to output,
// which activations each unit exposes, and which weights sit in it.
// Pure data, derived from the config. No engine code is imported here, so the
// page can draw the machine before any weights are loaded.

export const TRACE_CAPACITY = 256; // most token positions one run can hold

/** @param {{n_layer:number,n_head:number,n_embd:number,n_inner:number,n_ctx:number,vocab_size:number}} c */
export function gpt2Spec(c) {
  const d = c.n_embd, m = c.n_inner, H = c.n_head, V = c.vocab_size;
  const matrix = (id, label, tensor, rows, cols, extra = {}) => ({ id, label, view: { tensor, row0: 0, col0: 0, rows, cols, ...extra } });

  const blocks = Array.from({ length: c.n_layer }, (_, l) => {
    const p = `h.${l}`;
    return {
      id: `block.${l}`,
      kind: "block",
      index: l,
      title: `Block ${l}`,
      attention: { trace: `block.${l}.attn.pattern`, heads: H },
      neurons: { trace: `block.${l}.mlp.post`, size: m },
      stream: { trace: `block.${l}.resid`, size: d },
      writes: [
        { id: "attn", label: "attention output", trace: `block.${l}.attn.out`, size: d },
        { id: "mlp", label: "MLP output", trace: `block.${l}.mlp.out`, size: d },
      ],
      // The matrices laid out on the tray floor. c_attn holds Q, K and V side by side.
      weights: [
        matrix("W_Q", "Attention query W_Q", `${p}.attn.c_attn.weight`, d, d, { col0: 0 }),
        matrix("W_K", "Attention key W_K", `${p}.attn.c_attn.weight`, d, d, { col0: d }),
        matrix("W_V", "Attention value W_V", `${p}.attn.c_attn.weight`, d, d, { col0: 2 * d }),
        matrix("W_O", "Attention output W_O", `${p}.attn.c_proj.weight`, d, d),
        matrix("W_in", "MLP input W_in", `${p}.mlp.c_fc.weight`, d, m),
        // stored [m, d]; shown with neurons along the columns, like W_in
        matrix("W_out", "MLP output W_out", `${p}.mlp.c_proj.weight`, d, m, { transpose: true }),
      ],
      // Where the parts of attention and the MLP live, so the bench can redo
      // one head's or one neuron's arithmetic from the weights themselves.
      // Conv1D stores weights [in, out]: a neuron's input weights are a column
      // of c_fc, its output weights a row of the MLP's c_proj.
      parts: {
        eps: c.layer_norm_epsilon ?? 1e-5,
        attention: {
          heads: H, headSize: d / H, causal: true,
          ln: { gain: `${p}.ln_1.weight`, bias: `${p}.ln_1.bias` },
          qkv: { tensor: `${p}.attn.c_attn.weight`, bias: `${p}.attn.c_attn.bias`, query: 0, key: d, value: 2 * d },
          output: `${p}.attn.c_proj.weight`,
        },
        mlp: {
          ln: { gain: `${p}.ln_2.weight`, bias: `${p}.ln_2.bias` },
          in: { tensor: `${p}.mlp.c_fc.weight`, bias: `${p}.mlp.c_fc.bias` },
          out: { tensor: `${p}.mlp.c_proj.weight`, bias: `${p}.mlp.c_proj.bias` },
          activation: "gelu_new",
        },
      },
      // Everything else in the block, for the weight viewer.
      vectors: [
        matrix("ln1.g", "LayerNorm 1 gain", `${p}.ln_1.weight`, 1, d),
        matrix("ln1.b", "LayerNorm 1 bias", `${p}.ln_1.bias`, 1, d),
        matrix("b_QKV", "Attention Q, K, V bias", `${p}.attn.c_attn.bias`, 1, 3 * d),
        matrix("b_O", "Attention output bias", `${p}.attn.c_proj.bias`, 1, d),
        matrix("ln2.g", "LayerNorm 2 gain", `${p}.ln_2.weight`, 1, d),
        matrix("ln2.b", "LayerNorm 2 bias", `${p}.ln_2.bias`, 1, d),
        matrix("b_in", "MLP input bias", `${p}.mlp.c_fc.bias`, 1, m),
        matrix("b_out", "MLP output bias", `${p}.mlp.c_proj.bias`, 1, d),
      ],
    };
  });

  return {
    arch: "gpt2",
    config: c,
    capacity: Math.min(TRACE_CAPACITY, c.n_ctx),
    inputs: [{ id: "text", kind: "text", label: "Prompt" }],
    sequence: { unit: "token" },
    generates: true, // the output at the last position is the next token, so the bench can extend the prompt
    units: [
      {
        id: "embed",
        kind: "embed",
        title: "Embedding",
        stream: { trace: "resid.0", size: d },
        writes: [
          { id: "tok", label: "token embedding", trace: "embed.tok", size: d },
          { id: "pos", label: "position embedding", trace: "embed.pos", size: d },
        ],
        weights: [
          matrix("W_E", "Token embeddings W_E", "wte.weight", V, d),
          matrix("W_pos", "Position embeddings W_pos", "wpe.weight", c.n_ctx, d),
        ],
        vectors: [],
      },
      ...blocks,
      {
        id: "unembed",
        kind: "unembed",
        title: "Unembedding",
        stream: { trace: "final.resid", size: d },
        writes: [],
        output: { topk: "out.topk", vocab: V },
        weights: [matrix("W_U", "Unembedding W_U (tied to W_E)", "wte.weight", V, d)],
        vectors: [
          matrix("lnf.g", "Final LayerNorm gain", "ln_f.weight", 1, d),
          matrix("lnf.b", "Final LayerNorm bias", "ln_f.bias", 1, d),
        ],
      },
    ],
  };
}

/** Number of scalar parameters, counting the tied embedding once. */
export function gpt2ParamCount(c) {
  const d = c.n_embd, m = c.n_inner;
  const block = 2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (d * m + m) + (m * d + d);
  return c.vocab_size * d + c.n_ctx * d + c.n_layer * block + 2 * d;
}
