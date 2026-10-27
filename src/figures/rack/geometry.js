// The cabinet, the model as a machine: geometry.
//
// A cabinet of pull-out modules, one per unit of the model, input at the
// bottom. Everything here is static drawing, built once from the spec with the
// Anatomy kit; live.js fills the plates and moves the parts.
//
//   module front      attention plate (heads x tokens), neuron plate, and the
//                     two vectors the block writes into the stream
//   cabinet side      the residual stream after each unit, one window per unit
//   module deck       every weight of the unit, seen when the module is pulled
//   plinth            the token tape and its read head
//   top               the printer, and the candidates on the top module
//
// World units: x along the cabinet front, y toward the viewer, z up.

import * as k from "../../../vendor/anatomy/iso-kit.mjs";
import * as G from "../../../vendor/anatomy/lathe.mjs";
import { tubePieces, tubeSvg } from "../../../vendor/anatomy/tube.mjs";
import { gridFor } from "../../ui/plates.js";
import { TAPE, escapeXml, faces, num, plateMarkup, tapeParts } from "../parts.js";

const WALL = 10, D = 120, KICK = 30, PITCH = 30, FRONT_H = 26, CAP = 5, PANEL_T = 2.4;
const PLATE_H = 24; // every plate on a module front is this tall
const BOX_D = D - 8; // how deep a module is
export const PULL = BOX_D - 8; // how far it slides out
const AZIMUTH = 52;
const VIEW_W = 720;

const S = k.solidSvg, L = k.lineSvg, Dots = k.dotsSvg;

/** What goes on one module front, left to right, in face units. */
function frontItems(unit, config) {
  const items = [];
  if (unit.attention) {
    const H = unit.attention.heads;
    items.push({ kind: "attention", id: `${unit.id}/attention`, trace: unit.attention.trace, cols: TAPE, rows: H, w: TAPE * 3, h: PLATE_H, label: "attention" });
  }
  const port = (write, cell) => {
    const [cols, rows] = gridFor(write.size, 4 / 3);
    return { kind: "vector", id: `${unit.id}/${write.id}`, trace: write.trace, size: write.size, cols, rows, w: cols * cell, h: PLATE_H, label: write.label };
  };
  const writes = unit.writes ?? [];
  if (unit.kind === "embed") {
    for (const write of writes) items.push(port(write, 2));
  } else {
    if (writes[0]) items.push(port(writes[0], 1));
    if (unit.neurons) {
      const m = unit.neurons.size;
      const rows = [16, 12, 8, 4, 2, 1].find((r) => m % r === 0);
      const cols = m / rows;
      const cell = Math.min(4.5, Math.max(1.5, 288 / cols));
      items.push({ kind: "neurons", id: `${unit.id}/neurons`, trace: unit.neurons.trace, size: m, cols, rows, w: cols * cell, h: PLATE_H, label: "neurons" });
    }
    if (writes[1]) items.push(port(writes[1], 1));
  }
  if (unit.output) items.push({ kind: "candidates", id: `${unit.id}/candidates`, count: 8, w: 8 * 46 - 6, h: PLATE_H });
  return items;
}

function flipIfTall(view) {
  if (view.rows <= view.cols * 2) return view;
  return { ...view, rows: view.cols, cols: view.rows, row0: view.col0 ?? 0, col0: view.row0 ?? 0, transpose: !view.transpose };
}

/** Lays a unit's weights out on its deck: matrices in rows, vectors as strips. */
function deckItems(unit, W) {
  const x0 = 8, x1 = W - 8, yBack = D - PULL + 3, yFront = D - 15; // the panel's lip hides the last 7 units of deck
  const gap = 6, rowH = (yFront - yBack - gap) / 2;
  const out = [];
  let x = x0, row = 0;
  for (const weight of unit.weights ?? []) {
    const view = flipIfTall(weight.view);
    const w = Math.min(x1 - x0, Math.max(rowH * 0.75, rowH * (view.cols / view.rows)));
    if (x + w > x1 + 0.01 && x > x0) { row++; x = x0; }
    if (row > 1) break;
    out.push({
      kind: "weights", id: `${unit.id}/${weight.id}`, label: weight.label, view,
      x, y: yBack + row * (rowH + gap), w, h: rowH,
      cols: Math.min(view.cols, Math.round(w * 2)), rows: Math.min(view.rows, Math.round(rowH * 2)),
    });
    x += w + gap;
  }
  // vectors fill what is left of the last row as thin strips
  const vectors = unit.vectors ?? [];
  if (vectors.length) {
    if (x1 - x < 120) { row++; x = x0; }
    if (row <= 1) {
      const stripH = Math.min(3, (rowH - (vectors.length - 1) * 1.6) / vectors.length);
      const pitch = (rowH - stripH) / Math.max(1, vectors.length - 1);
      vectors.forEach((weight, index) => {
        const w = Math.min(x1 - x, 260);
        out.push({
          kind: "weights", id: `${unit.id}/${weight.id}`, label: weight.label, view: weight.view,
          x, y: yBack + row * (rowH + gap) + index * (vectors.length > 1 ? pitch : 0), w, h: stripH,
          cols: Math.min(weight.view.cols, Math.round(w * 2)), rows: 1, strip: true,
        });
      });
    }
  }
  return out;
}

export function buildRack(spec, { title = "" } = {}) {
  const units = spec.units;
  const N = units.length;
  const fronts = units.map((unit) => frontItems(unit, spec.config));
  const EAR = 14, GAP = 8, HANDLE = 28;
  const widthOf = (items) => EAR + items.reduce((sum, item) => sum + item.w, 0) + Math.max(0, items.length - 1) * GAP + HANDLE;
  const W = Math.max(360, TAPE * 15.5 + 16, ...fronts.map(widthOf));
  const HC = KICK + N * PITCH + CAP; // cabinet height
  const zOf = (i) => KICK + i * PITCH;

  // ---- camera -------------------------------------------------------------
  const PLINTH = { x: -40, y: -24, w: W + 2 * WALL + 60, d: D + 24 + 124, r: 10 };
  const extremes = [
    ...k.boxCorners(PLINTH, -17, 0),
    ...k.boxCorners({ x: -WALL - 2, y: -2, w: W + 2 * WALL + 4, d: D + 4 }, 0, HC + 26),
    [0, D + PULL + 12, HC], [W, D + PULL + 12, HC], [0, D + PULL + 12, KICK],
  ];
  const unitP = { origin: [0, 0], scale: 1, azimuth: AZIMUTH, elevation: 30 };
  const flat = extremes.map((p) => k.iso(p, unitP));
  const spanX = Math.max(...flat.map((p) => p[0])) - Math.min(...flat.map((p) => p[0]));
  const spanY = Math.max(...flat.map((p) => p[1])) - Math.min(...flat.map((p) => p[1]));
  const PAD = 26;
  const VIEW_H = Math.round(((VIEW_W - 2 * PAD) * spanY) / spanX + 2 * PAD);
  const P = k.fitProjection(extremes, VIEW_W, VIEW_H, { pad: PAD, azimuth: AZIMUTH });
  const at = (p) => k.iso(p, P);
  const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

  const { front: frontFace, side: sideFace, top: topFace } = faces(P);

  const plates = []; // everything live.js must fill
  const plateSvg = (item, unitIndex) => {
    plates.push({ ...item, unit: unitIndex });
    return plateMarkup(item, unitIndex);
  };

  // ---- plinth, feet, nameplate -----------------------------------------------
  const plinth = () => {
    const out = [`<path class="iso-halo" d="${k.haloOf(PLINTH, -17, 12, P)}"/>`];
    for (const [x, y] of byDepth(k.corners(PLINTH, 26))) {
      out.push(S(k.cylinder(x, y, 9, -17, 1.6, P, 28), { tone: "lo" }));
      out.push(S(k.cylinder(x, y, 7, -15.4, 3.4, P, 28), { tone: "mid" }));
    }
    out.push(S(k.slabOf(PLINTH, -12, 12, P, 8, 1.6), { tone: "mid" }));
    out.push(L(k.planOutline(k.insetPlan(PLINTH, 6), 0, P), { tone: "faint" }));
    const screws = k.corners(PLINTH, 13);
    out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(""), { tone: "lo" }));
    out.push(Dots(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
    return out.join("");
  };

  // ---- token tape and read head --------------------------------------------
  const TILE_PITCH = (W - 16) / TAPE;
  const tapeSet = tapeParts(P, { x0: 8, pitch: TILE_PITCH, y0: D + 46, z0: 0, id: "rack" });

  // ---- cabinet -----------------------------------------------------------------
  const BODY = { x: -WALL, y: 0, w: W + 2 * WALL, d: D, r: 3 };
  const XS = W + WALL; // the side wall
  const STREAM_Y = D - 16; // where the stream column starts on the side wall
  const streamGrid = gridFor(units[0].stream.size, 4 / 3);
  const STREAM_W = Math.min(84, streamGrid[0] * 2.5), STREAM_H = FRONT_H - 1.2;
  const cabinet = () => {
    const out = [];
    const base = k.insetPlan(BODY, 3);
    out.push(S(k.slabOf(base, 0, KICK - 4, P, 6, 0), { tone: "mid" }));
    out.push(S(k.slabOf(BODY, KICK - 4, HC - CAP - KICK + 4, P, 6, 0), { tone: "mid" }));
    // the kick base carries the nameplate and a row of vent slots
    const kickY = D - 3;
    const plate = { u: 170, v: 6, w: 92, h: 13 };
    const nameplate =
      `<rect class="na-inlay" x="${plate.u}" y="${plate.v}" width="${plate.w}" height="${plate.h}" rx="1.2"/>` +
      `<rect class="na-rule" x="${plate.u + 1.6}" y="${plate.v + 1.6}" width="${plate.w - 3.2}" height="${plate.h - 3.2}" rx="0.6"/>` +
      `<text class="iso-face-text" data-tone="mid" x="${plate.u + plate.w / 2}" y="${plate.v + plate.h / 2 + 1.9}" font-size="5.4" text-anchor="middle">${escapeXml(title)}</text>`;
    let slots = "";
    for (let i = 0; i < 16; i++) slots += `<rect class="na-slot" x="${W - 150 + i * 8}" y="7" width="3.2" height="12" rx="1.6"/>`;
    out.push(frontFace(0, kickY, KICK - 4, nameplate + slots));
    out.push(Dots([[plate.u + 3, plate.v + 3], [plate.u + plate.w - 3, plate.v + 3], [plate.u + 3, plate.v + plate.h - 3], [plate.u + plate.w - 3, plate.v + plate.h - 3]].map(([u, v]) => at([u, kickY, KICK - 4 - v])), { size: 0.45 }));
    return out.join("");
  };
  const cap = () => {
    const plan = k.insetPlan(BODY, -1.6);
    return S(k.slabOf(plan, HC - CAP, CAP, P, 6, 1.2), { tone: "mid" }) + L(k.planOutline(k.insetPlan(BODY, 5), HC, P), { tone: "faint" });
  };

  // the residual stream: one window per unit up the side wall, between two guides
  const stream = () => {
    const out = [];
    units.forEach((unit, i) => {
      const [cols, rows] = streamGrid;
      const item = { kind: "stream", id: `${unit.id}/stream`, trace: unit.stream.trace, size: unit.stream.size, cols, rows, w: STREAM_W, h: STREAM_H, label: "stream" };
      out.push(sideFace(XS, STREAM_Y, zOf(i) + FRONT_H - 0.6, plateSvg(item, i)));
    });
    // guides: two thin rails the whole height, standing on the kick base seam
    for (const y of [STREAM_Y + 3.4, STREAM_Y - STREAM_W - 5]) {
      out.push(S(k.slabOf({ x: XS, y, w: 1.3, d: 1.6, r: 0.3 }, KICK - 4, HC - CAP - KICK + 4, P, 2), { tone: "mid", crease: "none" }));
    }
    // vent slots and a carrying handle toward the back
    let slots = "";
    for (let row = 0; row < 3; row++) for (let i = 0; i < 4; i++) slots += `<rect class="na-slot" x="${D - 30 + i * 5.5}" y="${28 + row * 26}" width="2.4" height="18" rx="1.2"/>`;
    out.push(sideFace(XS, D, HC - CAP, slots));
    return out.join("");
  };

  // ---- cable from the tape to the embedding unit ---------------------------------
  const cable = () => {
    const out = [];
    const box = { x: W + 16, y: D + 60, w: 12, d: 14, r: 1.5 };
    const r = 1.5;
    const wallX = XS - 3; // the kick base stands 3 back from the body
    const sx = wallX + 3.2, sy = D - 38;
    const route = G.fillet([[box.x + box.w / 2, box.y + 1, r], [box.x + box.w / 2, sy, r], [sx + 9, sy, r], [sx, sy, r + 0.01], [sx, sy, 15]], 6, 7, { P, tube: r });
    const pieces = tubePieces(route, r, P, { maxLength: 400 });
    pieces.sort((a, b) => G.dot3(a.mid, G.viewOf(P)) - G.dot3(b.mid, G.viewOf(P)));
    out.push(pieces.map((piece) => tubeSvg(piece, { tone: "mid" })).join(""));
    // the terminal box the cable leaves, drawn after it: the cable starts behind the box
    out.push(S(k.slabOf(box, 0, 8, P, 3, 0.5), { tone: "mid" }));
    out.push(Dots([at([box.x + 3, box.y + 3, 8]), at([box.x + box.w - 3, box.y + box.d - 3, 8])], { size: 0.45 }));
    // the connector shell on the wall the cable ends in
    out.push(S(k.slabOf({ x: wallX, y: sy - 3.4, w: 6.4, d: 6.8, r: 1.2 }, 15, 6, P, 3, 0.4), { tone: "mid" }));
    out.push(Dots([at([wallX + 3.2, sy, 21])], { size: 0.5 }));
    return out.join("");
  };

  // ---- modules ---------------------------------------------------------------------
  const clips = [];
  const decks = units.map((unit) => deckItems(unit, W));
  const moduleSvg = (unit, i) => {
    const z0 = zOf(i), z1 = z0 + FRONT_H, deckZ = z1 - 5;
    const items = fronts[i];
    const out = [];
    // chassis and its deck
    const chassis = { x: 3, y: D - BOX_D, w: W - 6, d: BOX_D, r: 1.5 };
    out.push(S(k.slabOf(chassis, z0 + 1, deckZ - z0 - 1, P, 3, 0.5), { tone: "mid" }));
    out.push(L(k.planOutline(k.insetPlan(chassis, 2.6), deckZ, P, 3), { tone: "faint" }));
    out.push(S(k.slabOf({ x: W - 3, y: D - BOX_D + 6, w: 1.1, d: BOX_D - 12, r: 0.3 }, z0 + 8, 3, P, 2), { tone: "lo", crease: "none" }));
    for (const item of decks[i]) {
      const cls = item.strip ? "na-plate na-strip" : "na-plate";
      plates.push({ ...item, unit: i });
      out.push(
        topFace(item.x, item.y, deckZ + 0.02,
          `<g class="${cls}" data-plate="${item.id}" data-kind="weights" data-unit="${i}">` +
          `<rect class="na-well" x="-0.6" y="-0.6" width="${num(item.w + 1.2)}" height="${num(item.h + 1.2)}" rx="0.6"/>` +
          `<image class="na-cells" x="0" y="0" width="${num(item.w)}" height="${num(item.h)}" preserveAspectRatio="none"/>` +
          `</g>`),
      );
      if (!item.strip) out.push(Dots([[item.x - 2, item.y - 2], [item.x + item.w + 2, item.y - 2], [item.x - 2, item.y + item.h + 2], [item.x + item.w + 2, item.y + item.h + 2]].map(([x, y]) => at([x, y, deckZ])), { size: 0.4, tone: "lo" }));
    }
    // front panel
    const panel = { x: 0, y: D, w: W, d: PANEL_T, r: 0.8 };
    const faceY = D + PANEL_T;
    out.push(`<g class="na-front">${S(k.slabOf(panel, z0, FRONT_H, P, 3, 0.4), { tone: "hi", crease: "lo" })}</g>`);
    let u = EAR, face = "";
    const v0 = (FRONT_H - PLATE_H) / 2;
    for (const item of items) {
      if (item.kind === "candidates") {
        let inner = "";
        for (let c = 0; c < item.count; c++) {
          const x = c * 46;
          inner +=
            `<g class="na-candidate" data-candidate="${c}">` +
            `<clipPath id="na-cand-clip-${c}"><rect x="${x + 1}" y="0" width="38" height="13"/></clipPath>` +
            `<rect class="na-well" x="${x}" y="0" width="40" height="13" rx="1"/>` +
            `<g clip-path="url(#na-cand-clip-${c})"><text class="na-cand-text" x="${x + 2.4}" y="9.4" font-size="7.4"></text></g>` +
            `<rect class="na-track" x="${x}" y="17" width="40" height="5" rx="1"/>` +
            `<rect class="na-bar" x="${x + 0.8}" y="17.8" width="0" height="3.4" rx="0.5"/>` +
            `</g>`;
        }
        face += `<g transform="translate(${num(u)} ${num(v0)})" data-plate="${item.id}" data-kind="candidates" data-unit="${i}">${inner}</g>`;
        plates.push({ ...item, unit: i });
      } else {
        face += `<g transform="translate(${num(u)} ${num(v0)})">${plateSvg(item, i)}</g>`;
      }
      u += item.w + GAP;
    }
    // unit mark, engraved at the ear
    const mark = unit.kind === "embed" ? "E" : unit.kind === "unembed" ? "U" : String(unit.index);
    face += `<text class="iso-face-text" data-tone="mid" x="7" y="${num(FRONT_H - 6)}" font-size="5" text-anchor="middle">${mark}</text>`;
    out.push(frontFace(0, faceY, z1, face));
    // lamp, screws
    out.push(`<g class="na-lamp" data-lamp="${i}">${L(k.sideRing([7, faceY, z1 - 6.5], 2.1, "y", P, 14), { tone: "lo" })}${Dots([at([7, faceY, z1 - 6.5])], { size: 0.9, tone: "lo" })}</g>`);
    out.push(Dots([[2.2, z0 + 2.2], [2.2, z1 - 2.2], [W - 2.2, z0 + 2.2], [W - 2.2, z1 - 2.2]].map(([x, z]) => at([x, faceY, z])), { size: 0.42 }));
    // pull handle: two posts and a bar
    const hx = W - HANDLE + 5;
    for (const x of [hx, hx + 15]) out.push(S(k.slabOf({ x, y: faceY, w: 2, d: 4, r: 0.6 }, z0 + 11, 4, P, 2), { tone: "mid", crease: "none" }));
    out.push(S(k.slabOf({ x: hx - 1.5, y: faceY + 4, w: 20, d: 2.2, r: 1 }, z0 + 10.4, 5.2, P, 3, 0.3), { tone: "mid" }));

    // what can be seen of this module: the slot it slides out of, and all in front of it
    const hull = G.hullOf(k.boxCorners({ x: -0.5, y: D, w: W + 1, d: PULL + 14 }, z0 - 0.4, z1 + 0.4).map(at));
    clips.push(`<clipPath id="na-slot-${i}"><path d="${k.pathOf(hull, true)}"/></clipPath>`);
    return `<g clip-path="url(#na-slot-${i})"><g class="na-module" data-unit="${i}" data-kind="${unit.kind}">${out.join("")}</g></g>`;
  };

  // ---- printer on the top -----------------------------------------------------------
  const DIAL = { x: W - 62, y: 62, r: 20 };
  const PAPER = { x: 50, y: 60, w: Math.min(250, W - 220), d: 46, r: 0.6 };
  const printer = () => {
    const out = [];
    const housing = { x: 30, y: 16, w: PAPER.w + 40, d: 42, r: 3 };
    out.push(S(k.slabOf(housing, HC, 15, P, 5, 0.9), { tone: "mid" }));
    out.push(L(k.planOutline(k.insetPlan(housing, 4), HC + 15, P, 5), { tone: "faint" }));
    out.push(Dots(k.corners(housing, 7).map(([x, y]) => at([x, y, HC + 15])), { size: 0.45 }));
    // the slot the paper leaves through, on the housing front
    out.push(frontFace(PAPER.x - 3, housing.y + housing.d, HC + 15, `<rect class="na-slot" x="0" y="11.4" width="${num(PAPER.w + 6)}" height="2.2" rx="1.1"/>`));
    // platen knob on the right end
    const F = G.frameOf([housing.x + housing.w, housing.y + 22, HC + 8], [1, 0, 0], [0, 1, 0], [0, 0, 1]);
    out.push(S(G.disc(0, 1.6, 3.2, F, P), { tone: "mid" }));
    out.push(S(G.disc(1.6, 7, 5.6, F, P), { tone: "mid" }));
    out.push(k.fadedSvg(G.ribsOf(F, 1.9, 6.7, 5.6, 26, P, { fade: [0.05, 0.5] }), { tone: "lo" }));
    // paper
    out.push(S(k.slabOf(PAPER, HC, 0.5, P, 2), { tone: "hi", crease: "none" }));
    out.push(
      topFace(PAPER.x, PAPER.y + PAPER.d - 15, HC + 0.52,
        `<clipPath id="na-paper-clip"><rect x="3" y="-12" width="${num(PAPER.w - 6)}" height="18"/></clipPath>` +
        `<g clip-path="url(#na-paper-clip)"><text class="na-paper-text" x="${num(PAPER.w - 5)}" y="0" font-size="9.5" text-anchor="end" xml:space="preserve"></text></g>`,
        { "data-part": "paper" }),
    );
    out.push(S(k.slabOf({ x: PAPER.x - 5, y: PAPER.y + PAPER.d - 4, w: PAPER.w + 10, d: 2.6, r: 1 }, HC, 1.8, P, 3, 0.3), { tone: "mid" }));
    // temperature dial
    out.push(S(k.cylinder(DIAL.x, DIAL.y, DIAL.r, HC, 5, P, 64, 0.8), { tone: "mid" }));
    out.push(L(k.ring(DIAL.x, DIAL.y, DIAL.r - 2, HC + 5, P, 56), { tone: "lo" }));
    const ticks = k.radialTicks(DIAL.x, DIAL.y, DIAL.r - 2.6, 30, 5, HC + 5, [1.4, 3], P, -Math.PI * 0.75, Math.PI * 1.5);
    out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
    out.push(S(k.cylinder(DIAL.x, DIAL.y, 9, HC + 5, 5.5, P, 40, 0.5), { tone: "hi", crease: "lo" }));
    out.push(L(k.knurl(DIAL.x, DIAL.y, 9, HC + 5, HC + 10.5, 40, P), { tone: "lo" }));
    out.push(`<path data-part="dial" class="iso-line" data-tone="lit" style="stroke-width:1.1" d=""/>`);
    return out.join("");
  };
  const dialPointer = (turn) => {
    // turn 0..1 sweeps the dial's 270 degrees
    const angle = -Math.PI * 0.75 + turn * Math.PI * 1.5;
    const sx = Math.sin(angle), sy = -Math.cos(angle);
    return k.lineOnTop([DIAL.x, DIAL.y], [DIAL.x + sx * 8.4, DIAL.y + sy * 8.4], HC + 10.52, P);
  };

  const modules = units.map(moduleSvg); // fills `clips`
  const body = [
    `<defs>${clips.join("")}</defs>`,
    plinth(),
    tapeSet.tape,
    `<g data-part="head">${tapeSet.head}</g>`,
    cabinet(),
    stream(),
    cable(),
    modules.join(""),
    cap(),
    printer(),
  ];

  return {
    svg: k.figureSvg({
      width: VIEW_W,
      height: VIEW_H,
      label: `An isometric cabinet of ${N} pull-out modules, one for each unit of the model, with the input at the bottom. Each module front carries a plate of attention weights and a plate of neuron activations; windows up the side show the residual stream after each unit. A token tape with a read head lies in front, and a printer on top shows the next token.`,
      body,
    }),
    width: VIEW_W,
    height: VIEW_H,
    P,
    W,
    plates,
    tilePitch: TILE_PITCH,
    dialPointer,
    zOf,
    frontH: FRONT_H,
    // screen rectangle of a module when closed, to focus the stage on it
    boxOf(i, pulled = false) {
      const z0 = zOf(i);
      const points = k.boxCorners({ x: 0, y: pulled ? D - 20 : D, w: W, d: pulled ? PULL + 24 : 4 }, z0, z0 + FRONT_H).map(at);
      const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
      return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
    },
  };
}
