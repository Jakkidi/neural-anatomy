// The framed card every figure sits in: figure number top left, title top
// right, hint bottom left, live readout bottom right, legend and caption
// underneath. (The Anatomy kit's plate, built as DOM so figures can be
// swapped and the stage can take focus.)

const SWATCH = {
  raised: `<path d="M2 4.5 8 1.5 14 4.5 14 6 8 9 2 6Z"/><path class="iso-swatch-crease" d="M2 4.5 8 7.5 14 4.5"/>`,
  lit: `<path d="M2 4.5 8 1.5 14 4.5 14 6 8 9 2 6Z"/><path class="iso-swatch-crease" d="M2 4.5 8 7.5 14 4.5"/>`,
  flat: `<path d="M2 6.5 8 3.5 14 6.5 8 9.5Z"/>`,
  line: `<path d="M1.5 5h13"/>`,
  cell: `<rect x="5" y="2" width="6" height="6"/>`,
};

const el = (tag, attributes = {}, html = "") => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) if (value !== undefined && value !== false) node.setAttribute(key, value === true ? "" : value);
  if (html) node.innerHTML = html;
  return node;
};
const escape = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function createCard({ fig, title, hint, keys = [], caption = "", label }) {
  const root = el("figure", { class: "iso-figure" });
  const plate = el("div", { class: "iso-plate" });
  const corners = el("div", { class: "iso-plate-corners", "aria-hidden": "true" });
  corners.innerHTML =
    `<span class="iso-plate-corner" data-corner="fig">${escape(fig)}</span>` +
    `<span class="iso-plate-corner" data-corner="title">${escape(title)}</span>` +
    `<span class="iso-plate-corner" data-corner="hint">${escape(hint)}</span>` +
    `<span class="iso-plate-corner" data-corner="readout" data-readout></span>`;
  const readout = corners.querySelector("[data-readout]");

  const tools = el("div", { class: "na-tools", role: "group", "aria-label": "Zoom" });
  const out = el("button", { type: "button", class: "na-tool", "aria-label": "Zoom out", title: "Zoom out (−)" }, "−");
  const level = el("button", { type: "button", class: "na-tool na-tool-level", "aria-label": "Fit the whole figure", title: "Fit (0)" }, "1×");
  const into = el("button", { type: "button", class: "na-tool", "aria-label": "Zoom in", title: "Zoom in (+)" }, "+");
  tools.append(out, level, into);

  const stage = el("div", { class: "iso-stage", tabindex: "0", role: "group", "aria-label": label });
  plate.append(corners, tools, stage);
  root.append(plate);

  const legend = el("figcaption", { class: "iso-legend" });
  if (keys.length) {
    legend.append(
      el("span", { class: "iso-keys" }, keys.map((key) => {
        const kind = key.mark === "red" ? "cell" : key.mark === "ink" ? "cell" : SWATCH[key.mark] ? key.mark : "line";
        return `<span class="iso-key"><svg class="iso-swatch" data-mark="${key.mark}" viewBox="0 0 16 10" aria-hidden="true">${SWATCH[kind]}</svg>${escape(key.label)}</span>`;
      }).join("")),
    );
  }
  if (caption) legend.append(el("span", { class: "iso-caption" }, escape(caption)));
  root.append(legend);

  const card = {
    root,
    stage,
    controller: null,
    say(text) {
      if (readout.textContent !== text) readout.textContent = text;
    },
    setZoom(zoom) {
      level.textContent = `${zoom >= 10 ? Math.round(zoom) : Math.round(zoom * 10) / 10}×`;
      level.toggleAttribute("data-zoomed", zoom > 1.01);
    },
  };
  out.addEventListener("click", () => card.controller?.stage.zoomAt(1 / 1.6));
  into.addEventListener("click", () => card.controller?.stage.zoomAt(1.6));
  level.addEventListener("click", () => card.controller?.stage.fit());
  return card;
}
