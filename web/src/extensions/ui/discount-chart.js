// How much a reward t edges ahead counts: one bar of height γ^t per step. Bars where the walk being
// inspected collects a reward are dark and labelled with that reward. A dashed rule marks
// 1/(1−γ), the sum of all the weights: roughly how far ahead the robot looks.
import { h, replace, s } from "../../ui/dom.js";
import { fmt, fmtShort } from "./format.js";

export class DiscountChart {
  constructor({ steps = 24, height = 180 } = {}) {
    this.steps = steps;
    this.height = height;
    this.gamma = 0.9;
    this.rewards = {};
    this.readout = h("p", { class: "chart-readout" });
    this.svg = s("svg", { role: "img" });
    this.tooltip = h("div", { class: "tooltip", hidden: true });
    this.wrap = h("div", { class: "rel" }, this.svg, this.tooltip);
    this.el = h("figure", { class: "figure discount-chart", style: { margin: 0 } }, h("div", { class: "figure-head" }, "How much a reward t edges ahead counts: γ", h("sup", {}, "t")), this.readout, this.wrap);
    this.width = 480;
    new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w > 0 && w !== this.width) {
        this.width = w;
        requestAnimationFrame(() => this.render());
      }
    }).observe(this.wrap);
    this.svg.addEventListener("pointerleave", () => this.unhover());
  }

  // rewards: { t: r } for the steps where the inspected walk is paid
  update({ gamma, rewards = {} }) {
    this.gamma = gamma;
    this.rewards = rewards;
    this.render();
  }

  render() {
    const { steps, height, gamma, width } = this;
    const horizon = 1 / (1 - gamma);
    const half = gamma > 0 ? Math.log(0.5) / Math.log(gamma) : 0;
    this.readout.textContent =
      gamma === 0
        ? "γ = 0: only the reward of the next edge counts, everything later counts 0."
        : `The weights add up to 1/(1−γ) = ${fmtShort(horizon, 1)}. A reward's weight halves every ${fmtShort(half, 1)} edge${Math.abs(half - 1) < 0.05 ? "" : "s"}.`;
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);
    this.svg.setAttribute("aria-label", `bar chart of the discount weight gamma to the power t for t from 0 to ${steps}, at gamma ${gamma.toFixed(2)}`);
    const m = { top: 22, right: 12, bottom: 30, left: 34 };
    const band = (width - m.left - m.right) / (steps + 1);
    const X = (t) => m.left + t * band;
    const Y = (y) => m.top + (1 - y) * (height - m.top - m.bottom);
    const base = Y(0);
    const bw = Math.max(2, band - 2); // 2px surface gap between bars
    const bars = [];
    const labels = [];
    for (let t = 0; t <= steps; t++) {
      const wgt = gamma ** t;
      const paid = this.rewards[t] !== undefined;
      const top = Y(wgt);
      const r = Math.min(4, bw / 2, base - top);
      const x0 = X(t) + 1;
      const x1 = x0 + bw;
      // rounded data end on top, square on the baseline
      const d = base - top < 0.5 ? `M${x0},${base}H${x1}` : `M${x0},${base}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x1 - r}Q${x1},${top} ${x1},${top + r}V${base}Z`;
      bars.push(s("path", { d, fill: paid ? "var(--learn)" : "var(--env)", stroke: base - top < 0.5 ? (paid ? "var(--learn)" : "var(--env)") : "none", "stroke-width": 1 }));
      if (paid) labels.push(s("text", { class: "chart-label", x: x0 + bw / 2, y: Math.max(m.top - 6, top - 6), "text-anchor": "middle", style: { fill: "var(--ink)" } }, this.rewards[t] > 0 ? `+${fmtShort(this.rewards[t], 2)}` : fmtShort(this.rewards[t], 2)));
    }
    const hits = [];
    for (let t = 0; t <= steps; t++) {
      const hit = s("rect", { x: X(t), y: m.top, width: band, height: base - m.top, fill: "transparent" });
      hit.addEventListener("pointerenter", () => this.hover(t, X(t) + band / 2, Y(gamma ** t)));
      hits.push(hit);
    }
    const yTicks = [0, 0.5, 1];
    const xTicks = [];
    for (let t = 0; t <= steps; t += 5) xTicks.push(t);
    const hx = m.left + Math.min(horizon, steps + 1) * band;
    const marker =
      gamma > 0
        ? s(
            "g",
            {},
            horizon <= steps + 1 ? s("line", { x1: hx, x2: hx, y1: m.top - 4, y2: base, stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "4 3" }) : null,
            s("text", { class: "chart-label", x: horizon <= steps + 1 ? Math.min(hx + 4, width - m.right) : width - m.right, y: m.top - 8, "text-anchor": horizon <= steps + 1 && hx + 110 < width ? "start" : "end", style: { fill: "var(--ink-2)" } }, horizon <= steps + 1 ? `1/(1−γ) = ${fmtShort(horizon, 1)}` : `1/(1−γ) = ${fmtShort(horizon, 0)} →`),
          )
        : null;
    replace(
      this.svg,
      s("g", { class: "chart-grid" }, yTicks.map((t) => s("line", { x1: m.left, x2: width - m.right, y1: Y(t), y2: Y(t) }))),
      s(
        "g",
        { class: "chart-axis" },
        yTicks.map((t) => s("text", { x: m.left - 6, y: Y(t) + 4, "text-anchor": "end" }, fmtShort(t))),
        xTicks.map((t) => s("text", { x: X(t) + band / 2, y: height - m.bottom + 15, "text-anchor": "middle" }, t)),
        s("text", { class: "chart-title-text", x: width - m.right, y: height - 3, "text-anchor": "end" }, "t, edges ahead"),
      ),
      bars,
      s("line", { class: "chart-baseline", x1: m.left, x2: width - m.right, y1: base, y2: base }),
      marker,
      labels,
      hits,
    );
  }

  hover(t, x, y) {
    const r = this.rewards[t];
    replace(
      this.tooltip,
      h("div", { class: "tt-head" }, `t = ${t}`),
      h("div", { class: "tt-row" }, h("b", {}, fmt(this.gamma ** t, 3)), h("span", {}, "weight γ", h("sup", {}, t))),
      r !== undefined ? h("div", { class: "tt-row" }, h("b", {}, fmt(r * this.gamma ** t, 3)), h("span", {}, `this walk's ${fmtShort(r, 2)}, weighted`)) : null,
    );
    this.tooltip.hidden = false;
    const tw = this.tooltip.offsetWidth;
    this.tooltip.style.left = `${Math.max(0, x + 10 + tw > this.width ? x - 10 - tw : x + 10)}px`;
    this.tooltip.style.top = `${Math.max(0, y - 30)}px`;
  }

  unhover() {
    this.tooltip.hidden = true;
  }
}
