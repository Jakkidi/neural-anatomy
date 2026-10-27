// The cabinet: the model as a machine, and the only figure. Works for any spec whose units are
// "embed", "block" and "unembed" (see docs/ADDING_AN_ARCHITECTURE.md).

import { mountRack } from "./live.js";

export const rackFigure = {
  id: "rack",
  fig: "GPT-2 · the cabinet",
  name: "Cabinet",
  title: (spec) => `${spec.units.length} units · input at the bottom`,
  hint: "Click any plate to see its arithmetic",
  label:
    "The model drawn as a cabinet of pull-out modules. Arrow left and right move along the tokens, arrow up and down pull out a module, Escape pushes it back, plus and minus zoom.",
  keys: [
    { mark: "ink", label: "Ink is a positive value: a neuron firing, attention paid, a stream dimension above zero" },
    { mark: "red", label: "Red ink is a negative value" },
    { mark: "lit", label: "The token under the read head; every plate shows that position" },
  ],
  caption: (spec) => {
    const c = spec.config;
    return `Each module is one unit of the model, from the embedding at the bottom to the unembedding at the top. On a block's front, the left plate has a row for each of its ${c.n_head} attention heads and a column for each token; the wide plate has one cell for each of its ${c.n_inner.toLocaleString()} neurons, inked where the neuron is above zero; the two small plates are what attention and the MLP add to the residual stream. The windows up the side are the stream itself, ${c.n_embd} numbers, after each unit. Pulled out, a module shows every weight it holds. Click any plate, or any weight plate on a pulled-out module, and the camera goes inside: the bench lays that unit out as the matrix multiplications it really does, on these numbers, down to each dot product. Full ink is set for each plate separately, at the 99th percentile of what it holds, so plates compare patterns, not sizes; hover a cell for its value.`;
  },
  mount: mountRack,
};
