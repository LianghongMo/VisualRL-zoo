// Every sample of a PPO rollout as a dot: x is its probability ratio r = π_θ/π_old,
// y its advantage. The shaded corners are where the clipped objective is flat,
// so a sample there contributes no gradient.
import { h, replace, s } from "../../ui/dom.js";
import { fmt } from "./format.js";

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class RatioScatter {
  constructor({ height = 280, onSelect = () => {} } = {}) {
    this.height = height;
    this.onSelect = onSelect;
    this.width = 560;
    this.svg = s("svg", { role: "img", "aria-label": "Probability ratio against advantage for every sample" });
    this.tooltip = h("div", { class: "tooltip", hidden: true });
    this.el = h("div", { class: "rel scatter" }, this.svg, this.tooltip);
    new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      // Redraw on the next frame: redrawing inside the callback can resize the observed box again.
      if (w > 0 && w !== this.width) {
        this.width = w;
        requestAnimationFrame(() => this.view && this.render(this.view));
      }
    }).observe(this.el);
    this.svg.addEventListener("pointermove", (e) => this.hover(e));
    this.svg.addEventListener("pointerleave", () => (this.tooltip.hidden = true));
    this.svg.addEventListener("click", (e) => {
      const hit = this.nearest(e);
      if (hit) this.onSelect(hit.index);
    });
  }

  // view: { samples: [{ index, ratio, advantage, clipped }], inBatch: Set, clip, selected }
  render(view) {
    this.view = view;
    const { samples, inBatch = new Set(), clip, selected = null } = view;
    const height = this.height;
    const width = Math.max(this.width, 200); // never draw into a collapsed container
    const m = { top: 26, right: 14, bottom: 40, left: 46 };
    const xMax = Math.max(1 + 2.5 * clip, 1.6);
    const xMin = Math.max(0, Math.min(1 - 2.5 * clip, 0.4));
    const yMax = 4;
    const X = (r) => m.left + ((clamp(r, xMin, xMax) - xMin) / (xMax - xMin)) * (width - m.left - m.right);
    const Y = (a) => m.top + (1 - (clamp(a, -yMax, yMax) + yMax) / (2 * yMax)) * (height - m.top - m.bottom);
    this.scale = { X, Y };
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);

    const lo = 1 - clip;
    const hi = 1 + clip;
    const regions = [
      s("rect", { x: X(hi), y: Y(yMax), width: Math.max(0, X(xMax) - X(hi)), height: Math.max(0, Y(0) - Y(yMax)), class: "clip-region" }),
      s("rect", { x: X(xMin), y: Y(0), width: Math.max(0, X(lo) - X(xMin)), height: Math.max(0, Y(-yMax) - Y(0)), class: "clip-region" }),
      s("text", { x: X(xMax) - 6, y: Y(yMax) + 14, "text-anchor": "end", class: "chart-label" }, "clipped: no gradient"),
      s("text", { x: X(xMin) + 6, y: Y(-yMax) - 6, class: "chart-label" }, "clipped: no gradient"),
    ];
    const axes = [
      s("line", { x1: m.left, x2: width - m.right, y1: Y(0), y2: Y(0), class: "chart-baseline" }),
      s("line", { x1: X(1), x2: X(1), y1: m.top, y2: height - m.bottom, class: "chart-baseline" }),
      s("line", { x1: X(lo), x2: X(lo), y1: m.top, y2: height - m.bottom, class: "clip-line" }),
      s("line", { x1: X(hi), x2: X(hi), y1: m.top, y2: height - m.bottom, class: "clip-line" }),
      s("text", { x: X(lo), y: height - m.bottom + 15, "text-anchor": "middle", class: "chart-label" }, `1−ε`),
      s("text", { x: X(1), y: height - m.bottom + 15, "text-anchor": "middle", class: "chart-label" }, "1"),
      s("text", { x: X(hi), y: height - m.bottom + 15, "text-anchor": "middle", class: "chart-label" }, `1+ε`),
      ...[-4, -2, 0, 2, 4].map((v) => s("text", { x: m.left - 8, y: Y(v) + 4, "text-anchor": "end", class: "chart-label" }, String(v))),
      s("text", { x: width - m.right, y: height - 4, "text-anchor": "end", class: "chart-title-text" }, "probability ratio r = π_θ(a|s) / π_old(a|s)"),
      s("text", { x: 2, y: 11, class: "chart-title-text" }, "advantage Â"),
    ];
    const background = [];
    const front = [];
    for (const p of samples) {
      const cx = X(p.ratio);
      const cy = Y(p.advantage);
      if (!inBatch.has(p.index)) background.push(s("circle", { cx, cy, r: 1.8, class: "dot-other" }));
      else front.push(s("circle", { cx, cy, r: 3.6, class: p.clipped ? "dot-clipped" : "dot-active" }));
    }
    const sel = samples.find((p) => p.index === selected);
    const selMark = sel ? s("circle", { cx: X(sel.ratio), cy: Y(sel.advantage), r: 7.5, class: "dot-selected" }) : null;
    replace(this.svg, regions, axes, background, front, selMark);
    this.points = samples;
  }

  nearest(e) {
    if (!this.points || !this.scale) return null;
    const rect = this.svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    let best = null;
    let bestD = 12 * 12;
    for (const p of this.points) {
      const d = (this.scale.X(p.ratio) - x) ** 2 + (this.scale.Y(p.advantage) - y) ** 2;
      if (d < bestD || (d === bestD && this.view.inBatch?.has(p.index))) {
        best = p;
        bestD = d;
      }
    }
    return best;
  }

  hover(e) {
    const p = this.nearest(e);
    if (!p) {
      this.tooltip.hidden = true;
      return;
    }
    replace(
      this.tooltip,
      h("div", { class: "tt-head" }, `sample ${p.index}${this.view.inBatch?.has(p.index) ? " · in this minibatch" : ""}`),
      h("div", { class: "tt-row" }, h("b", {}, fmt(p.ratio, 3)), h("span", {}, "ratio r")),
      h("div", { class: "tt-row" }, h("b", {}, fmt(p.advantage, 3)), h("span", {}, "advantage Â")),
      h("div", { class: "tt-row" }, h("span", {}, p.clipped ? "clipped: no gradient" : "gradient flows")),
    );
    this.tooltip.hidden = false;
    const x = this.scale.X(p.ratio);
    const tw = this.tooltip.offsetWidth;
    this.tooltip.style.left = `${x + 12 + tw > this.width ? x - 12 - tw : x + 12}px`;
    this.tooltip.style.top = `${Math.max(0, this.scale.Y(p.advantage) - 30)}px`;
  }
}
