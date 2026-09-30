// A responsive SVG line chart: hairline grid, 2px lines, crosshair tooltip
// listing every series at the hovered x, a legend for two or more series.
import { h, replace, s } from "./dom.js";
import { fmtShort } from "./format.js";

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => span / m <= count) ?? 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

export class LineChart {
  // series: [{ id, label, color, points: [[x, y], ...], dashed }]
  constructor({ height = 240, xLabel = "", yLabel = "", xFormat = fmtShort, yFormat = fmtShort, xDomain, yDomain, caption } = {}) {
    this.opts = { height, xLabel, yLabel, xFormat, yFormat, xDomain, yDomain };
    this.series = [];
    this.markers = [];
    this.legendEl = h("div", { class: "legend" });
    this.svg = s("svg", { role: "img" });
    this.tooltip = h("div", { class: "tooltip", hidden: true });
    this.wrap = h("div", { class: "rel" }, this.svg, this.tooltip);
    this.el = h("figure", { class: "figure", style: { margin: 0 } }, this.legendEl, this.wrap, caption ? h("figcaption", { class: "figure-caption" }, caption) : null);
    this.width = 600;
    new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w > 0 && w !== this.width) {
        this.width = w;
        this.render();
      }
    }).observe(this.wrap);
    this.svg.addEventListener("pointermove", (e) => this.hover(e));
    this.svg.addEventListener("pointerleave", () => this.unhover());
  }

  // markers: [{ x, label }] vertical rules; points: [{ x, y, label }] emphasized dots
  update({ series, markers = [], points = [], yDomain, xDomain } = {}) {
    if (series) this.series = series;
    this.markers = markers;
    this.points = points;
    if (yDomain) this.opts.yDomain = yDomain;
    if (xDomain) this.opts.xDomain = xDomain;
    this.render();
  }

  domain() {
    const xs = this.series.flatMap((sr) => sr.points.map((p) => p[0]));
    const ys = this.series.flatMap((sr) => sr.points.map((p) => p[1])).filter(Number.isFinite);
    const xDomain = this.opts.xDomain ?? [Math.min(...xs), Math.max(...xs)];
    let yDomain = this.opts.yDomain;
    if (!yDomain) {
      const lo = Math.min(0, ...ys);
      const hi = Math.max(0, ...ys);
      const pad = (hi - lo || 1) * 0.08;
      yDomain = [lo < 0 ? lo - pad : lo, hi + pad];
    }
    return { xDomain, yDomain };
  }

  render() {
    const { height, xLabel, yLabel, xFormat, yFormat } = this.opts;
    const width = this.width;
    replace(
      this.legendEl,
      this.series.length > 1
        ? this.series.map((sr) =>
            h(
              "span",
              { class: "legend-item" },
              h("span", { class: `legend-line${sr.dashed ? " dashed" : ""}`, style: { color: sr.color } }),
              sr.label,
            ),
          )
        : [],
    );
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);
    this.svg.setAttribute("aria-label", `${yLabel} against ${xLabel}`);
    if (!this.series.length || !this.series.some((sr) => sr.points.length)) {
      replace(this.svg);
      return;
    }
    const m = { top: 26, right: 16, bottom: 38, left: 52 };
    const { xDomain, yDomain } = this.domain();
    const X = (x) => m.left + ((x - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * (width - m.left - m.right);
    const Y = (y) => m.top + (1 - (y - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * (height - m.top - m.bottom);
    this.scale = { X, Y, m, xDomain, yDomain };

    const yTicks = niceTicks(yDomain[0], yDomain[1], 5);
    const xTicks = niceTicks(xDomain[0], xDomain[1], Math.max(2, Math.floor(width / 90)));
    const grid = s(
      "g",
      { class: "chart-grid" },
      yTicks.map((t) => s("line", { x1: m.left, x2: width - m.right, y1: Y(t), y2: Y(t) })),
    );
    const axes = s(
      "g",
      { class: "chart-axis" },
      yTicks.map((t) => s("text", { x: m.left - 8, y: Y(t) + 4, "text-anchor": "end" }, yFormat(t))),
      xTicks.map((t) => s("text", { x: X(t), y: height - m.bottom + 16, "text-anchor": "middle" }, xFormat(t))),
      s("text", { class: "chart-title-text", x: width - m.right, y: height - 4, "text-anchor": "end" }, xLabel),
      s("text", { class: "chart-title-text", x: 2, y: 12 }, yLabel),
    );
    const zero = yDomain[0] < 0 && yDomain[1] > 0 ? s("line", { class: "chart-baseline", x1: m.left, x2: width - m.right, y1: Y(0), y2: Y(0) }) : null;
    const baseline = s("line", { class: "chart-baseline", x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom });
    const markers = this.markers.map((mk) =>
      s(
        "g",
        {},
        s("line", { x1: X(mk.x), x2: X(mk.x), y1: m.top, y2: height - m.bottom, stroke: "var(--ink-3)", "stroke-width": 1 }),
        mk.label ? s("text", { class: "chart-label", x: X(mk.x) + 5, y: m.top + 10 }, mk.label) : null,
      ),
    );
    const lines = this.series.map((sr) => {
      const d = sr.points
        .filter((p) => Number.isFinite(p[1]))
        .map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`)
        .join("");
      return s("path", {
        d,
        fill: "none",
        stroke: sr.color,
        "stroke-width": 2,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
        "stroke-dasharray": sr.dashed ? "5 4" : null,
      });
    });
    const points = (this.points ?? []).map((p) =>
      s(
        "g",
        {},
        s("circle", { cx: X(p.x), cy: Y(p.y), r: 5, fill: p.color ?? "var(--ink)", stroke: "var(--sheet)", "stroke-width": 2 }),
        p.label
          ? s("text", { class: "chart-label", x: X(p.x) > width - 110 ? X(p.x) - 9 : X(p.x) + 9, y: Y(p.y) - 8, "text-anchor": X(p.x) > width - 110 ? "end" : "start", style: { fill: "var(--ink-2)" } }, p.label)
          : null,
      ),
    );
    this.hoverLayer = s("g");
    replace(this.svg, grid, zero, baseline, axes, markers, lines, points, this.hoverLayer);
  }

  hover(e) {
    if (!this.scale || !this.series.length) return;
    const { X, Y, m, xDomain } = this.scale;
    const rect = this.svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const x = xDomain[0] + ((px - m.left) / (this.width - m.left - m.right)) * (xDomain[1] - xDomain[0]);
    const ref = this.series.reduce((a, b) => (b.points.length > a.points.length ? b : a));
    if (!ref.points.length) return;
    let best = ref.points[0];
    for (const p of ref.points) if (Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
    const bx = best[0];
    const rows = this.series.map((sr) => {
      const p = sr.points.find((q) => q[0] === bx);
      return { sr, y: p?.[1] };
    });
    replace(
      this.hoverLayer,
      s("line", { x1: X(bx), x2: X(bx), y1: m.top, y2: this.opts.height - m.bottom, stroke: "var(--ink-2)", "stroke-width": 1 }),
      rows
        .filter((r) => Number.isFinite(r.y))
        .map((r) => s("circle", { cx: X(bx), cy: Y(r.y), r: 4, fill: r.sr.color, stroke: "var(--sheet)", "stroke-width": 2 })),
    );
    replace(
      this.tooltip,
      h("div", { class: "tt-head" }, `${this.opts.xLabel} ${this.opts.xFormat(bx)}`),
      rows.map((r) =>
        h(
          "div",
          { class: "tt-row" },
          h("span", { class: `legend-line${r.sr.dashed ? " dashed" : ""}`, style: { color: r.sr.color } }),
          h("b", {}, Number.isFinite(r.y) ? this.opts.yFormat(r.y) : "—"),
          h("span", {}, r.sr.label),
        ),
      ),
    );
    this.tooltip.hidden = false;
    const tw = this.tooltip.offsetWidth;
    const left = X(bx) + 12 + tw > this.width ? X(bx) - 12 - tw : X(bx) + 12;
    this.tooltip.style.left = `${Math.max(0, left)}px`;
    this.tooltip.style.top = `${m.top}px`;
  }

  unhover() {
    if (this.hoverLayer) replace(this.hoverLayer);
    this.tooltip.hidden = true;
  }
}
