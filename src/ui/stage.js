// Zoom and pan for a figure, done by changing the SVG's viewBox so the
// drawing is re-rendered sharp at every magnification.
//
//   pinch, or Ctrl/Cmd + wheel   zoom about the pointer
//   drag (when zoomed in)        pan
//   double click                 zoom in 2x at that point
//   + - 0                        zoom in, out, fit

export class Stage {
  /**
   * @param {HTMLElement} element the .iso-stage wrapper
   * @param {SVGSVGElement} svg
   * @param {{width:number,height:number,maxZoom?:number,onChange?:(zoom:number)=>void}} options
   */
  constructor(element, svg, { width, height, maxZoom = 16, onChange = () => {} }) {
    this.element = element;
    this.svg = svg;
    this.full = { x: 0, y: 0, w: width, h: height };
    this.view = { ...this.full };
    this.maxZoom = maxZoom;
    this.onChange = onChange;
    this.pointers = new Map();
    this.dragged = false;
    this.#listen();
  }

  get zoom() {
    return this.full.w / this.view.w;
  }

  #apply() {
    const { x, y, w, h } = this.view;
    this.svg.setAttribute("viewBox", `${x.toFixed(2)} ${y.toFixed(2)} ${w.toFixed(2)} ${h.toFixed(2)}`);
    this.element.toggleAttribute("data-zoomed", this.zoom > 1.01);
    this.onChange(this.zoom);
  }

  /** viewBox coordinates of a client point (the SVG may be letterboxed in its box) */
  toView(clientX, clientY) {
    const matrix = this.svg.getScreenCTM();
    if (!matrix) return [this.view.x + this.view.w / 2, this.view.y + this.view.h / 2];
    const point = new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
    return [point.x, point.y];
  }

  #clamp() {
    const v = this.view, f = this.full;
    v.w = Math.min(f.w, Math.max(f.w / this.maxZoom, v.w));
    v.h = v.w * (f.h / f.w);
    v.x = Math.min(f.x + f.w - v.w, Math.max(f.x, v.x));
    v.y = Math.min(f.y + f.h - v.h, Math.max(f.y, v.y));
  }

  zoomAt(factor, clientX, clientY) {
    const box = this.svg.getBoundingClientRect();
    const cx = clientX ?? box.left + box.width / 2, cy = clientY ?? box.top + box.height / 2;
    const [px, py] = this.toView(cx, cy);
    const v = this.view;
    const w = Math.min(this.full.w, Math.max(this.full.w / this.maxZoom, v.w / factor));
    const k = w / v.w;
    v.x = px - (px - v.x) * k;
    v.y = py - (py - v.y) * k;
    v.w = w;
    this.#clamp();
    this.#apply();
  }

  fit() {
    this.view = { ...this.full };
    this.#apply();
  }

  /** Brings a viewBox rectangle into view at a given zoom. */
  focus(x, y, w, h, zoom) {
    const f = this.full;
    const vw = Math.min(f.w, Math.max(f.w / this.maxZoom, zoom ? f.w / zoom : Math.max(w * 1.25, (h * 1.25 * f.w) / f.h)));
    this.view.w = vw;
    this.view.h = vw * (f.h / f.w);
    this.view.x = x + w / 2 - this.view.w / 2;
    this.view.y = y + h / 2 - this.view.h / 2;
    this.#clamp();
    this.#apply();
  }

  /** Glides the view to a rectangle of the figure, or back to the whole figure. */
  flyTo(target, ms = 480) {
    const from = { ...this.view };
    let to = { ...this.full };
    if (target) {
      this.focus(target.x, target.y, target.w, target.h, target.zoom);
      to = { ...this.view };
      this.view = { ...from };
    }
    const still = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    return new Promise((resolve) => {
      if (still) { this.view = to; this.#apply(); resolve(); return; }
      const start = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - start) / ms);
        const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
        // zoom in a straight line in log scale, so the motion feels even
        const w = Math.exp(Math.log(from.w) + (Math.log(to.w) - Math.log(from.w)) * e);
        const k = (from.w - w) / (from.w - to.w || 1);
        const mix = Number.isFinite(k) && from.w !== to.w ? Math.min(1, Math.max(0, k)) : e;
        this.view = { w, h: w * (this.full.h / this.full.w), x: from.x + (to.x - from.x) * mix, y: from.y + (to.y - from.y) * mix };
        this.#apply();
        if (t < 1) requestAnimationFrame(step);
        else resolve();
      };
      requestAnimationFrame(step);
    });
  }

  /** The rectangle, in figure units, that an element covers on screen. */
  rectOf(element) {
    const box = element.getBoundingClientRect();
    const [x0, y0] = this.toView(box.left, box.top);
    const [x1, y1] = this.toView(box.right, box.bottom);
    return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
  }

  panBy(dxClient, dyClient) {
    const matrix = this.svg.getScreenCTM();
    const scale = matrix ? matrix.a : 1; // client px per viewBox unit
    this.view.x -= dxClient / scale;
    this.view.y -= dyClient / scale;
    this.#clamp();
    this.#apply();
  }

  #listen() {
    const el = this.element;
    el.addEventListener("wheel", (event) => {
      if (!(event.ctrlKey || event.metaKey)) return; // plain wheel scrolls the page
      event.preventDefault();
      this.zoomAt(Math.exp(-event.deltaY * 0.0045), event.clientX, event.clientY);
    }, { passive: false });

    el.addEventListener("dblclick", (event) => {
      if (event.target.closest("[data-no-zoom]")) return;
      event.preventDefault();
      this.zoomAt(event.shiftKey ? 0.5 : 2, event.clientX, event.clientY);
    });

    el.addEventListener("pointerdown", (event) => {
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      this.dragged = false;
      this.moved = 0;
    });
    el.addEventListener("pointermove", (event) => {
      const last = this.pointers.get(event.pointerId);
      if (!last) return;
      const dx = event.clientX - last.x, dy = event.clientY - last.y;
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.entries()];
        const other = a[0] === event.pointerId ? b[1] : a[1];
        const before = Math.hypot(last.x - other.x, last.y - other.y);
        const after = Math.hypot(event.clientX - other.x, event.clientY - other.y);
        if (before > 0) this.zoomAt(after / before, (event.clientX + other.x) / 2, (event.clientY + other.y) / 2);
        this.dragged = true;
      } else if (this.zoom > 1.01) {
        this.moved += Math.abs(dx) + Math.abs(dy);
        if (this.moved > 4) {
          if (!this.dragged) { try { el.setPointerCapture(event.pointerId); } catch { /* not capturable */ } }
          this.dragged = true;
          this.panBy(dx, dy);
        }
      }
      last.x = event.clientX;
      last.y = event.clientY;
    });
    const end = (event) => {
      this.pointers.delete(event.pointerId);
      try { el.releasePointerCapture(event.pointerId); } catch { /* was not captured */ }
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    // a drag must not also count as a click on what is under the pointer
    el.addEventListener("click", (event) => {
      if (this.dragged) { event.stopPropagation(); event.preventDefault(); this.dragged = false; }
    }, true);
  }
}
