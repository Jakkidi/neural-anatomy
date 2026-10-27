// Architectures the engine can run. Each entry is the engine half of an
// architecture: checkpoint planning, tokenizer, forward pass.
// (The page half, src/arch/ui-registry.js, lists the figures.)

import gpt2 from "./gpt2/index.js";

export const ENGINE_ARCHS = {
  [gpt2.id]: gpt2,
};
