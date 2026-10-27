# Adding an architecture

An architecture is everything the app needs to know about one family of
models. It lives in `src/arch/<id>/` and has two halves:

- the **engine half** runs in the worker: it reads a checkpoint, tokenizes, and
  runs the forward pass while handing out every activation;
- the **page half** is a description of the model (its *spec*) that the figures
  draw from.

Nothing outside the folder mentions the architecture except two registry
lines. `src/arch/gpt2/` is the worked example; this guide follows it.

If you only want another *size* of an architecture that exists, skip all of
this and add a model card (step 1).

## 1. A model card

`src/models/index.js`. Pure data, read by both the page and the worker.

```js
{
  id: "gpt2",                      // unique; also the ?model= value
  name: "GPT-2 small",
  arch: "gpt2",                    // the folder in src/arch/
  blurb: "OpenAI, 2019. 12 blocks, 12 heads, 124M weights.",
  config: { n_layer: 12, n_head: 12, n_embd: 768, n_inner: 3072, n_ctx: 1024, vocab_size: 50257 },
  weights: { kind: "safetensors", url: "https://huggingface.co/…/model.safetensors", page: "…", file: "model.safetensors", bytes: 548105171 },
  tokenizer: { vocab: "assets/gpt2/vocab.json", merges: "assets/gpt2/merges.txt" },
  prompts: ["The Eiffel Tower is located in the city of"],
}
```

The page draws the machine from `config` before any weights load. After
loading, the config read from the checkpoint wins, and the figures are rebuilt
if it differs. `weights.kind` is `"safetensors"` or `"random"`.

## 2. The engine half: `src/arch/<id>/index.js`

Export one object. The engine (`src/core/engine.js`) calls these and knows
nothing else about your model.

| Member | What it does |
|---|---|
| `id`, `name` | |
| `spec(config)` | the page half; see step 3 |
| `paramCount(config)` | number of weights, for the status line |
| `plan(tensors, card)` | Given the tensors a checkpoint lists (`{name, dtype, shape}`), return `{ config, wanted: [{ file, name, shape }] }`: the config as far as the shapes reveal it, and which tensors to keep under which names. Throw a clear sentence if the file is not yours. Tensors you do not name are skipped while streaming and never held in memory. |
| `weightFloats(config)`, `scratchFloats(config, capacity)` | How many Float32 slots the weights and your runtime's buffers need. The engine allocates one arena of that size up front (WebAssembly memory cannot grow once views are taken). |
| `synthesize(config, store, arena, seed)` | Fill the store with random weights, for the "random weights" test pattern. |
| `createTokenizer(card, readAsset)` | Resolve to `{ vocabSize, encode(text) → ids, decode(ids) → text, piece(id) → text }`. `readAsset(path, "json" \| "text")` reads a file of the app. |
| `createRuntime({ config, store, arena, capacity })` | the forward pass; below |

### The runtime

```js
{
  capacity,                 // most positions one run may hold
  length,                   // positions run so far
  reset(),
  *append(ids, emit),       // run new positions after the cached ones
  logits(position),         // output scores at a position already run (a reused Float32Array)
  readout(vector),          // optional: read any stream vector through the output head
}
```

`append` is a generator. Compute one unit, hand out its activations, `yield`
the unit's index; the engine uses the pauses to post progress and to stop a run
that has been superseded. Hand out activations with

```js
emit(key, rowSize, firstPosition, count, rows)
```

where `rows` is a Float32Array of `count * rowSize` values (a view is fine; the
engine copies it before you reuse it). **The first axis of every traced array
is the position in the sequence.** An attention pattern is therefore stored as
`[position, head, capacity]`, not `[head, position, position]`. Keys are yours
to choose; the spec is where you tell the figures which key means what.

Allocate everything from `arena.alloc(n)` and do the heavy arithmetic with
`arena.kernels` (`affine`, `matvec`, `dot`), which are WebAssembly SIMD when the
browser has it and JavaScript otherwise. `kernels.affine(x, W, b, y, n, nin, nout)`
computes `y = x @ W + b` for `W` stored `[nin, nout]`; a `[nout, nin]` matrix
(PyTorch's `nn.Linear`) needs transposing once at load, or its own kernel.

## 3. The page half: `src/arch/<id>/spec.js`

`spec(config)` returns plain data. It must not import engine code, so the page
can draw the machine before the worker has loaded anything.

```js
{
  arch: "gpt2",
  config,
  capacity: 128,                       // positions a trace holds
  inputs: [{ id: "text", kind: "text", label: "Prompt" }],
  generates: true,                     // false hides "Add a token"
  units: [ /* input to output */ ],
}
```

Each unit is one module of the machine:

```js
{
  id: "block.3", kind: "block", index: 3, title: "Block 3",
  stream:    { trace: "block.3.resid", size: 768 },             // the stream after this unit
  attention: { trace: "block.3.attn.pattern", heads: 12 },      // rows of [heads × capacity]
  neurons:   { trace: "block.3.mlp.post", size: 3072 },
  writes: [                                                     // vectors the unit adds to the stream
    { id: "attn", label: "attention output", trace: "block.3.attn.out", size: 768 },
    { id: "mlp",  label: "MLP output",       trace: "block.3.mlp.out",  size: 768 },
  ],
  weights: [ { id: "W_Q", label: "Attention query W_Q",
               view: { tensor: "h.3.attn.c_attn.weight", row0: 0, col0: 0, rows: 768, cols: 768 } }, … ],
  vectors: [ /* same shape as weights; biases and gains, shown as strips */ ],
}
```

A `view` is a rectangle of a stored tensor, so one tensor can appear as several
plates (GPT-2 keeps Q, K and V side by side in `c_attn`). `transpose: true`
swaps rows and columns for display. `kind` is `"embed"`, `"block"` or
`"unembed"`; the unembed unit also has `output: { vocab }`.

## 4. Register it

```js
// src/arch/engine-registry.js
import bert from "./bert/index.js";
export const ENGINE_ARCHS = { [gpt2.id]: gpt2, [bert.id]: bert };

// src/arch/ui-registry.js
export const UI_ARCHS = {
  gpt2: { spec: gpt2Spec, figures: [rackFigure] },
  bert: { spec: bertSpec, figures: [rackFigure] },
};
```

### The desktop app's engine

The desktop app (`desktop/`) draws from the same spec and figures, so the page
half above is all it needs on that side. Its engine is Rust, and an
architecture there is one file beside `desktop/engine/src/gpt2.rs` that
implements two traits from `desktop/engine/src/arch.rs`:

- `Architecture`: `plan` (which tensors of a checkpoint to keep, and the
  config their shapes imply), `synthesize` (random weights), `tokenizer`,
  `param_count`, `capacity`, and `runtime`;
- `Runtime`: `append` (run new positions, calling `sink(unit, items)` as each
  unit finishes), `logits`, `logits_many`, `readout`, `reset`.

They are the same contract as the JavaScript one, method for method, and the
trace keys must be the same strings, because both engines feed one page.
Register it in `architecture()` at the bottom of `arch.rs`:

```rust
pub fn architecture(id: &str) -> Option<&'static dyn Architecture> {
    match id {
        "gpt2" => Some(&crate::gpt2::Gpt2),
        "bert" => Some(&crate::bert::Bert),
        _ => None,
    }
}
```

An architecture with only a JavaScript half still works in the browser; the
desktop app answers "No architecture called ... is built into this app" for it.

## 5. The cabinet and the bench

If your units are embed / block / unembed with the fields above, the cabinet
(`src/figures/rack/`) draws your model with no changes: one module per unit,
sized from the spec (try the 4-block bench model to see it).

The bench (`src/figures/rack/bench/`) redoes each step of a block from the
weights. For that, a block in the spec also says where its parts live, in
`parts`:

```js
parts: {
  eps: 1e-5,                                   // LayerNorm epsilon
  attention: { heads, headSize,
    ln: { gain, bias },                        // tensor names
    qkv: { tensor, bias, query: 0, key: d, value: 2 * d },   // column offsets
    output },                                  // W_O
  mlp: { ln: { gain, bias }, in: { tensor, bias }, out: { tensor, bias }, activation: "gelu_new" },
}
```

Two of the bench's three modes read only `parts.attention`, through
`bench/heads.js`: **All tokens** (every token at once, and generation with and
without the KV cache) and **On a GPU** (one head as standard attention and as
FlashAttention, tile by tile). Describe your attention and both work as they
are:

```js
attention: {
  heads, headSize,
  kvHeads: 8,              // optional: grouped-query or multi-query attention
  causal: false,           // optional, default true; an encoder (BERT, ViT) sets false
  scale,                   // optional, default 1/√headSize
  ln: { gain, bias, kind: "rms" },     // bias and kind optional; "rms" for RMSNorm
  qkv: { tensor, bias, query, key, value },  // GPT-2: one fused matrix and column offsets
  // or three separate projections, [d, heads·headSize]; transpose: true if stored [out, in]
  q: { tensor, bias, transpose }, k: { … }, v: { … },
}
```

A spec that sets `rope: true` (Llama, Mistral) gets a message instead of wrong
numbers: `heads.js` does not rotate queries and keys yet. That is the one
function to add for them. The GPU view itself (`bench/gpu/`) works on any Q, K
and V and is tested without a model in `test/flash.test.mjs`, causal and not.

`bench/ops.js` assumes GPT-2's layout (pre-LayerNorm blocks, weights stored
[in, out], GELU). A model that differs (BERT's post-LayerNorm, a gated MLP)
needs its own `stepsOf` and `buildOp`; every step is described the same way,
input × matrix (+ bias) then optional steps after, so `view.js` draws it as is.

## 6. Check it against a reference

Do what `test/engine.test.mjs` does: write a checkpoint of random numbers in
your architecture's layout, run it through a second implementation that shares
no code with yours (`tools/reference_gpt2.py` is 80 lines of numpy), and
compare every traced array. Random weights with random biases and gains catch
wiring mistakes that a trained model's forgiving statistics hide.
`desktop/engine/tests/reference.rs` does the same for the Rust engine against
the same numpy file.


---

## What each kind of model still needs

**Encoder-only text models (BERT, RoBERTa, DistilBERT).** Fit the contract as
it stands. You write the tokenizer (WordPiece for BERT; RoBERTa can reuse
`core/tokenizers/byte-bpe.js`), a forward pass without the causal mask and
with BERT's post-LayerNorm blocks, and a spec with `generates: false`. The
cabinet works unchanged. One thing to decide: `logits(position)` would be the
masked-language-model head, so the bench's "Next token" panel should be titled
for what it then shows.

**Larger or smaller GPT-2s.** Already work by dropping the file. To offer one in
the picker, add a card. GPT-2 XL is 6.4 GB as float32, which is more than a
32-bit WebAssembly memory holds; it would need float16 storage in the arena or
the JavaScript fallback's per-tensor buffers.

**Vision transformers (ViT).** The blocks are the same; the input is not. Two
things are missing today:

1. *An image input on the bench.* `spec.inputs` already names input kinds, but
   `src/app.js` only draws the `"text"` one. Add a file drop for
   `kind: "image"` that hands the worker pixels.
2. *A runtime that takes arrays.* `Engine.run({ ids })` passes token ids to
   `runtime.append`. For patches, let `run` pass whatever the architecture's
   own `prepare(inputs)` returns (ids for text, patch vectors for images) and
   have `append` accept it.

The cabinet then works as is: the embed unit becomes the patch projection and
the "tokens" on the tape are patches, so the tape tiles want a thumbnail
instead of text (`drawTape` in `figures/parts.js`).

**Multi-modal models (CLIP, and anything with two towers).** A spec with one
`units` list describes one tower. For two, give the spec a `towers` list, each
with its own `units` and its own entry in `inputs`, plus the units where they
meet (CLIP's two projections and the similarity). The engine side is two
runtimes behind one `append`; trace keys just need a tower prefix. The figure
side is the real work: `buildRack` draws one cabinet from one list of units, so
a two-tower figure calls it once per tower and adds the unit where they meet.
That does not exist yet.

**Convolutional or recurrent models.** These need unit kinds the figures do not
know (`"conv"`, `"cell"`), so they need figure work as well as an engine half:
new plate layouts in the cabinet's module fronts (`frontItems` in
`figures/rack/geometry.js`), or a figure of their own.
