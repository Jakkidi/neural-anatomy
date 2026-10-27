// Model cards: one entry per model the picker offers. Pure data, shared by the
// page and the engine worker. A card names its architecture (src/arch/<id>),
// where its files live, and the config the page draws before weights load.
//
// To add a model of an architecture that already exists, add a card here.
// To add a new architecture, see docs/ADDING_AN_ARCHITECTURE.md.

const HF = "https://huggingface.co";

export const MODELS = [
  {
    id: "gpt2",
    name: "GPT-2 small",
    arch: "gpt2",
    blurb: "OpenAI, 2019. 12 blocks, 12 heads, 124M weights.",
    trained: true,
    config: { n_layer: 12, n_head: 12, n_embd: 768, n_inner: 3072, n_ctx: 1024, vocab_size: 50257, layer_norm_epsilon: 1e-5 },
    weights: {
      kind: "safetensors",
      url: `${HF}/openai-community/gpt2/resolve/main/model.safetensors`,
      page: `${HF}/openai-community/gpt2`,
      file: "model.safetensors",
      bytes: 548105171,
    },
    tokenizer: { vocab: "assets/gpt2/vocab.json", merges: "assets/gpt2/merges.txt" },
    prompts: ["The Eiffel Tower is located in the city of", "When Mary and John went to the store, John gave a drink to", "The cat sat on the"],
  },
  {
    id: "gpt2-bench",
    name: "Bench model",
    arch: "gpt2",
    blurb: "4 blocks, 4 heads, random weights. A small GPT-2-shaped model for trying the interface; it has learned nothing.",
    trained: false,
    config: { n_layer: 4, n_head: 4, n_embd: 128, n_inner: 512, n_ctx: 256, vocab_size: 50257, layer_norm_epsilon: 1e-5 },
    weights: { kind: "random", seed: 7 },
    tokenizer: { vocab: "assets/gpt2/vocab.json", merges: "assets/gpt2/merges.txt" },
    prompts: ["The cat sat on the"],
  },
];

export const modelById = (id) => MODELS.find((model) => model.id === id);
