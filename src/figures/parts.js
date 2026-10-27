// Parts the figures share: the token tape with its read head, and the helpers
// for drawing in the plane of a face.

import * as k from "../../vendor/anatomy/iso-kit.mjs";

export const TAPE = 32; // token tiles on a tape; longer prompts scroll

/** Tokens as the machine shows them: a leading space as a middle dot, and so on. */
export const showPiece = (piece) => piece.replace(/^ /, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥");

export const num = (v) => Math.round(v * 100) / 100;
export const attrs = (o) =>
  Object.entries(o)
    .filter(([, v]) => v !== undefined && v !== false)
    .map(([key, v]) => ` ${key}="${String(v).replace(/&/g, "&amp;").replace(/"/g, "&quot;")}"`)
    .join("");
export const escapeXml = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Groups whose local x, y lie in a face of the drawing. */
export function faces(P) {
  return {
    // a face looking toward +y: local x runs along world x, local y runs down
    front: (x, y, z, inner, extra = {}) => `<g${attrs({ transform: k.sideMatrix([x, y], z, P, "left"), ...extra })}>${inner}</g>`,
    // a face looking toward +x: local x runs along world -y, local y runs down
    side: (x, y, z, inner, extra = {}) => `<g${attrs({ transform: k.sideMatrix([x, y], z, P, "right"), ...extra })}>${inner}</g>`,
    // a face looking up: local x along world x, local y along world y
    top: (x, y, z, inner, extra = {}) => `<g${attrs({ transform: k.topMatrix([x, y], z, P, "x"), ...extra })}>${inner}</g>`,
  };
}

/** A data plate in face units: a well, the image of cells, and a cursor. */
export function plateMarkup(item, unitIndex, extra = "") {
  return (
    `<g class="na-plate" data-plate="${item.id}" data-kind="${item.kind}" data-unit="${unitIndex}">` +
    `<rect class="na-well" x="-0.7" y="-0.7" width="${num(item.w + 1.4)}" height="${num(item.h + 1.4)}" rx="0.7"/>` +
    `<image class="na-cells" x="0" y="0" width="${num(item.w)}" height="${num(item.h)}" preserveAspectRatio="none"/>` +
    extra +
    `<rect class="na-cursor" x="0" y="0" width="${num(item.w / item.cols)}" height="${num(item.h / item.rows)}" visibility="hidden"/>` +
    `</g>`
  );
}

/**
 * The token tape: a ruler strip, a bed of TAPE tiles, and a rail carrying the
 * read head. Laid along x from `x0` with one tile every `pitch`, starting at
 * depth `y0` and standing on height `z0`.
 * Returns { tape, head } markup; the head is built over tile 0 and is moved
 * along x by the live code.
 */
export function tapeParts(P, { x0, pitch, y0, z0, id = "tape" }) {
  const S = k.solidSvg, L = k.lineSvg, Dots = k.dotsSvg;
  const at = (p) => k.iso(p, P);
  const top = faces(P).top;
  const length = TAPE * pitch;
  const strip = { x: x0 - 4, y: y0, w: length + 8, d: 6, r: 1 };
  const bed = { x: x0 - 4, y: y0 + 10, w: length + 8, d: 26, r: 2 };
  const rail = { x: x0 - 4, y: y0 + 44, w: length + 8, d: 4, r: 1 };
  const tileX = (i) => x0 + i * pitch;

  const out = [];
  // ruler strip: one tick per position, a long one every 8
  out.push(S(k.slabOf(strip, z0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const minor = [], major = [];
  for (let i = 0; i <= TAPE; i++) {
    const x = tileX(i);
    (i % 8 === 0 ? major : minor).push(k.lineOnTop([x, strip.y + 0.4], [x, strip.y + (i % 8 === 0 ? 4.4 : 2.4)], z0 + 0.6, P));
  }
  out.push(L(minor.join(""), { tone: "lo" }), L(major.join(""), { tone: "mid" }));
  out.push(S(k.slabOf(bed, z0, 4, P, 4, 0.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(bed, 2.2), z0 + 4, P, 4), { tone: "faint" }));
  for (let i = 0; i < TAPE; i++) {
    const plan = { x: tileX(i) + 0.75, y: bed.y + 4, w: pitch - 1.5, d: 18, r: 1 };
    const clip = `<clipPath id="na-${id}-clip-${i}"><rect x="0.6" y="-6" width="${num(plan.w - 1.2)}" height="12"/></clipPath>`;
    out.push(
      `<g class="na-tile" data-tile="${i}" data-empty>` +
        S(k.slabOf(plan, z0 + 4, 1.3, P, 3, 0.3), { tone: "hi", crease: "lo" }) +
        top(plan.x, plan.y + 11.6, z0 + 5.3, `${clip}<g clip-path="url(#na-${id}-clip-${i})"><text class="na-tile-text" x="1.3" y="0" font-size="4.3"></text></g>`) +
        `</g>`,
    );
  }
  for (const x of [bed.x - 5, bed.x + bed.w]) {
    out.push(S(k.slabOf({ x, y: bed.y - 2, w: 5, d: bed.d + 4, r: 1.2 }, z0, 7, P, 3, 0.5), { tone: "mid" }));
    out.push(Dots([at([x + 2.5, bed.y + 3, z0 + 7]), at([x + 2.5, bed.y + bed.d - 3, z0 + 7])], { size: 0.45 }));
  }
  out.push(S(k.slabOf(rail, z0, 2.6, P, 3, 0.4), { tone: "mid" }));
  out.push(Dots([rail.x + 6, rail.x + rail.w / 2, rail.x + rail.w - 6].map((x) => at([x, rail.y + 2, z0 + 2.6])), { size: 0.4 }));
  for (const x of [rail.x - 4, rail.x + rail.w]) out.push(S(k.slabOf({ x, y: rail.y - 1.5, w: 4, d: 7, r: 1 }, z0, 5, P, 3, 0.4), { tone: "mid" }));

  // the read head: a carriage on the rail with a finger back to the tile
  const cx = tileX(0) + pitch / 2;
  const body = { x: cx - 6, y: rail.y - 3, w: 12, d: 10, r: 2 };
  const head = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(body, 2), z0 + 2.6, 0.1, P)}" style="filter:blur(3px)"/>`];
  head.push(S(k.slabOf(body, z0 + 2.6, 5.4, P, 4, 0.6), { tone: "mid" }));
  head.push(S(k.slabOf({ x: cx - 1.3, y: bed.y + 21, w: 2.6, d: body.y - bed.y - 21 + 1, r: 0.8 }, z0 + 6, 1.6, P, 2), { tone: "hi", lit: true, crease: "none" }));
  head.push(S(k.cylinder(cx, body.y + 5, 3.4, z0 + 8, 3.6, P, 24), { tone: "mid" }));
  head.push(L(k.knurl(cx, body.y + 5, 3.4, z0 + 8, z0 + 11.6, 22, P), { tone: "lo" }));
  head.push(Dots([at([cx, body.y + 5, z0 + 11.6])], { size: 0.5, tone: "hi" }));

  return { tape: out.join(""), head: head.join(""), depth: 51.5 };
}

/** Fills a mounted tape from the app state. `tiles` are the .na-tile groups in order. */
export function drawTape(tiles, state) {
  const { tokens, position, window } = state;
  tiles.forEach((tile, i) => {
    const index = window.start + i;
    const has = index < tokens.ids.length;
    tile.toggleAttribute("data-empty", !has);
    tile.querySelector("text").textContent = has ? showPiece(tokens.pieces[index]) : "";
    tile.querySelector(".iso-solid").toggleAttribute("data-lit", has && index === position);
  });
}
