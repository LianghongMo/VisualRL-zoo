// A GridWorld drawn as a policy: in every cell, one arrow per action whose
// length is π(a|s), on a background colored by the value estimate V(s).
import { h, replace, s } from "../../ui/dom.js";
import { ARROWS, fmt } from "./format.js";
import { inkOn, valueFill } from "./grid-view.js";

const C = 54;
const PAD = 6;
const DIRS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

export class PolicyGrid {
  constructor(env, { domain = 100, hazardLabel = null } = {}) {
    this.env = env;
    this.domain = domain;
    this.hazardLabel = hazardLabel;
    this.width = env.width * C + 2 * PAD;
    this.height = env.height * C + 2 * PAD;
    this.svg = s("svg", { viewBox: `0 0 ${this.width} ${this.height}`, role: "img", "aria-label": "Policy map" });
    this.tooltip = h("div", { class: "tooltip", hidden: true });
    this.el = h("div", { class: "figure-scroll" }, h("div", { class: "rel grid-figure" }, this.svg, this.tooltip));
  }

  center(state) {
    const [r, c] = this.env.toCell(state);
    return [PAD + c * C + C / 2, PAD + r * C + C / 2];
  }

  // view: { probs(state) -> [4], V, touched: Set of states, selected: state }
  render(view) {
    this.view = view;
    const { probs, V, touched = new Set(), selected = null } = view;
    const env = this.env;
    const cells = [];
    const marks = [];
    const hits = [];
    for (let st = 0; st < env.nStates; st++) {
      const [r, c] = env.toCell(st);
      const x = PAD + c * C;
      const y = PAD + r * C;
      if (env.walls.includes(st)) {
        cells.push(s("rect", { x, y, width: C, height: C, class: "cell-wall" }));
        continue;
      }
      if (env.cliffs.includes(st)) {
        cells.push(s("rect", { x, y, width: C, height: C, class: "cell-cliff" }));
        continue;
      }
      if (env.goals.includes(st)) {
        cells.push(s("rect", { x, y, width: C, height: C, class: "cell-goal" }));
        marks.push(s("text", { x: x + C / 2, y: y + C / 2 + 5, class: "cell-mark", "text-anchor": "middle" }, "G"));
        continue;
      }
      cells.push(s("rect", { x: x + 0.5, y: y + 0.5, width: C - 1, height: C - 1, style: { fill: valueFill(V[st], this.domain) } }));
      const [cx, cy] = this.center(st);
      const ink = inkOn(V[st], this.domain);
      probs(st).forEach((p, a) => {
        if (p < 0.02) return;
        const [dx, dy] = DIRS[a];
        const len = 4 + p * C * 0.4;
        const w = 1 + 2.4 * p;
        const tx = cx + dx * len;
        const ty = cy + dy * len;
        const head = 3 + 3 * p;
        cells.push(
          s("line", { x1: cx, y1: cy, x2: tx - dx * head, y2: ty - dy * head, stroke: ink, "stroke-width": w, "stroke-linecap": "round", opacity: 0.45 + 0.55 * p }),
          s("polygon", { points: `${tx},${ty} ${tx - dx * head - dy * head * 0.8},${ty - dy * head + dx * head * 0.8} ${tx - dx * head + dy * head * 0.8},${ty - dy * head - dx * head * 0.8}`, fill: ink, opacity: 0.45 + 0.55 * p }),
        );
      });
      cells.push(s("circle", { cx, cy, r: 2, fill: ink }));
      if (st === env.start) marks.push(s("text", { x: x + 5, y: y + 13, class: "cell-mark small" }, "S"));
      if (touched.has(st)) marks.push(s("rect", { x: x + 2, y: y + 2, width: C - 4, height: C - 4, class: "hl-touched" }));
      if (st === selected) marks.push(s("rect", { x: x + 1.5, y: y + 1.5, width: C - 3, height: C - 3, class: "hl-selected" }));
      const hit = s("rect", { x, y, width: C, height: C, class: "hit" });
      hit.addEventListener("pointerenter", () => this.showTip(st));
      hit.addEventListener("pointerleave", () => (this.tooltip.hidden = true));
      hits.push(hit);
    }
    const lines = [];
    for (let c = 1; c < env.width; c++) lines.push(s("line", { x1: PAD + c * C, x2: PAD + c * C, y1: PAD, y2: PAD + env.height * C, class: "grid-line" }));
    for (let r = 1; r < env.height; r++) lines.push(s("line", { y1: PAD + r * C, y2: PAD + r * C, x1: PAD, x2: PAD + env.width * C, class: "grid-line" }));
    const cliffCells = env.cliffs.map((st) => env.toCell(st));
    const label = cliffCells.length && this.hazardLabel
      ? s("text", { x: PAD + ((cliffCells[0][1] + cliffCells[cliffCells.length - 1][1] + 1) / 2) * C, y: PAD + cliffCells[0][0] * C + C / 2 + 5, class: "cliff-label", "text-anchor": "middle" }, this.hazardLabel)
      : null;
    replace(this.svg, cells, lines, label, marks, hits);
  }

  showTip(st) {
    const { probs, V } = this.view;
    const [r, c] = this.env.toCell(st);
    replace(
      this.tooltip,
      h("div", { class: "tt-head" }, `state (${r},${c}) · V = ${fmt(V[st], 2)}`),
      probs(st).map((p, a) => h("div", { class: "tt-row" }, h("span", {}, ARROWS[a]), h("b", {}, p.toFixed(3)), h("span", {}, "π(a|s)"))),
    );
    this.tooltip.hidden = false;
    const scale = this.svg.getBoundingClientRect().width / this.width;
    const [cx, cy] = this.center(st);
    const tw = this.tooltip.offsetWidth;
    const left = (cx + C / 2) * scale + 6;
    const maxLeft = this.svg.getBoundingClientRect().width - tw;
    this.tooltip.style.left = `${Math.max(0, left > maxLeft ? (cx - C / 2) * scale - tw - 6 : left)}px`;
    this.tooltip.style.top = `${Math.max(0, (cy - C / 2) * scale)}px`;
  }
}
