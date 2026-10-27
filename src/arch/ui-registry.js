// What the page knows about each architecture: how to describe it (the spec)
// and the figure that draws it. The engine half is engine-registry.js.

import { gpt2Spec } from "./gpt2/spec.js";
import { rackFigure } from "../figures/rack/index.js";

export const UI_ARCHS = {
  gpt2: {
    spec: gpt2Spec,
    figures: [rackFigure],
  },
};
