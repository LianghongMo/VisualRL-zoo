// A GridWorld drawn as cells split into four triangles, one per action, each
// colored by Q(s,a). Overlays show the agent, its next action, the last move,
// and which table entries the displayed learning trace read and wrote.
import { h, replace, s } from "./dom.js";
import { ARROWS, fmt } from "./format.js";

const C = 54; // cell size in SVG units
const PAD = 6;
const DIRS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

// Diverging fill on a log scale, so −1 and −13 are as distinguishable as −13 and −100.
// How deep the fill of a value is, 0..1. Marks drawn on fills deeper than ~0.45 switch to white.
export function valueStrength(q, domain) {
  if (!Number.isFinite(q) || q === 0) return 0;
  return Math.min(1, Math.log1p(Math.abs(q)) / Math.log1p(domain)) * 0.88;
}

export const inkOn = (q, domain) => (valueStrength(q, domain) > 0.45 ? "#ffffff" : "var(--ink)");

// Colors are mixed here rather than with CSS color-mix(), which some browsers lack: an SVG fill it
// cannot parse turns black.
let palette = null;
function tokenRGB(name, fallback) {
  const raw = (typeof document !== "undefined" && getComputedStyle(document.documentElement).getPropertyValue(name).trim()) || fallback;
  const hex = raw.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

export function valueFill(q, domain) {
  palette ??= { neg: tokenRGB("--neg", "#16397a"), pos: tokenRGB("--pos", "#5aa9f0"), zero: tokenRGB("--zero", "#f2f5fa") };
  if (!Number.isFinite(q) || q === 0) return `rgb(${palette.zero.join(",")})`;
  const t = valueStrength(q, domain);
  const end = q < 0 ? palette.neg : palette.pos;
  const mix = palette.zero.map((z, i) => Math.round(z + (end[i] - z) * t));
  return `rgb(${mix.join(",")})`;
}

export function valueLegend(domain, label = "Q(s,a)") {
  const stops = [-domain, -10, -1, 0];
  return h(
    "div",
    { class: "legend" },
    h("span", {}, label),
    h(
      "span",
      { class: "value-scale" },
      stops.map((v) => h("span", { class: "value-stop" }, h("span", { class: "legend-rect", style: { background: valueFill(v, domain) } }), fmt(v, 0))),
    ),
  );
}

function arrow(cx, cy, dir, { length = C * 0.3, start = 0, color = "var(--ink)", width = 2.2, head = 5.5, cls } = {}) {
  const [ux, uy] = DIRS[dir];
  const x1 = cx + ux * start;
  const y1 = cy + uy * start;
  const tx = cx + ux * length;
  const ty = cy + uy * length;
  const bx = tx - ux * head;
  const by = ty - uy * head;
  const px = -uy * head * 0.75;
  const py = ux * head * 0.75;
  return s(
    "g",
    { class: cls },
    s("line", { x1, y1, x2: bx, y2: by, stroke: color, "stroke-width": width, "stroke-linecap": "round" }),
    s("polygon", { points: `${tx},${ty} ${bx + px},${by + py} ${bx - px},${by - py}`, fill: color }),
  );
}

export class QGrid {
  constructor(env, { domain = 100, plain = false, hazardLabel = null } = {}) {
    this.env = env;
    this.hazardLabel = hazardLabel;
    this.domain = domain;
    this.plain = plain;
    this.width = env.width * C + 2 * PAD;
    this.height = env.height * C + 2 * PAD;
    this.svg = s("svg", { viewBox: `0 0 ${this.width} ${this.height}`, role: "img", "aria-label": "Grid world" });
    this.tooltip = h("div", { class: "tooltip", hidden: true });
    this.el = h("div", { class: "figure-scroll" }, h("div", { class: "rel grid-figure" }, this.svg, this.tooltip));
    this.state = {};
  }

  center(state) {
    const [r, c] = this.env.toCell(state);
    return [PAD + c * C + C / 2, PAD + r * C + C / 2];
  }

  triangle(state, a) {
    const [r, c] = this.env.toCell(state);
    const x0 = PAD + c * C;
    const y0 = PAD + r * C;
    const [cx, cy] = [x0 + C / 2, y0 + C / 2];
    const corners = [
      [x0, y0],
      [x0 + C, y0],
      [x0 + C, y0 + C],
      [x0, y0 + C],
    ];
    const [p, q] = [corners[a], corners[(a + 1) % 4]];
    return `${p[0]},${p[1]} ${q[0]},${q[1]} ${cx},${cy}`;
  }

  labelPos(state, a) {
    const [cx, cy] = this.center(state);
    const [ux, uy] = DIRS[a];
    return [cx + ux * C * 0.33, cy + uy * C * 0.33 + 3];
  }

  // view: { Q, agent, nextAction, move, updated, target, behavior, showValues, paths }
  render(view = this.state) {
    this.state = view;
    const { Q, agent, nextAction, move, updated, target, behavior, showValues, paths = [] } = view;
    const env = this.env;
    const cells = [];
    const labels = [];
    const valueCells = new Set([updated?.s, target?.s].filter((x) => x !== undefined && x !== null));

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
        labels.push(s("text", { x: x + C / 2, y: y + C / 2 + 5, class: "cell-mark", "text-anchor": "middle" }, "G"));
        continue;
      }
      if (this.plain || !Q) {
        cells.push(s("rect", { x, y, width: C, height: C, class: "cell-plain" }));
      } else {
        for (let a = 0; a < 4; a++) {
          cells.push(s("polygon", { points: this.triangle(st, a), class: "tri", style: { fill: valueFill(Q[st][a], this.domain) } }));
        }
        const best = Math.max(...Q[st]);
        const greedy = Q[st].map((q, a) => (q >= best - 1e-9 ? a : -1)).filter((a) => a >= 0);
        const [cx, cy] = this.center(st);
        if (greedy.length < 4) for (const a of greedy) cells.push(arrow(cx, cy, a, { length: C * 0.2, start: 3, width: 1.6, head: 4, color: inkOn(Math.min(...Q[st]), this.domain) }));
        if (showValues || valueCells.has(st)) {
          for (let a = 0; a < 4; a++) {
            const [lx, ly] = this.labelPos(st, a);
            const dark = valueStrength(Q[st][a], this.domain) > 0.45;
            labels.push(s("text", { x: lx, y: ly, class: `tri-value${dark ? " on-dark" : ""}`, "text-anchor": "middle" }, fmt(Q[st][a], Math.abs(Q[st][a]) >= 9.95 ? 0 : 1)));
          }
        }
      }
      if (st === env.start) labels.push(s("text", { x: x + 5, y: y + 13, class: "cell-mark small" }, "S"));
    }

    // Cell borders on top of the fills: the grid reads as cells, the triangles as their four actions.
    const gridLines = s("rect", { x: PAD, y: PAD, width: env.width * C, height: env.height * C, class: "grid-frame" });
    const cellEdges = [];
    for (let c = 1; c < env.width; c++) cellEdges.push(s("line", { x1: PAD + c * C, x2: PAD + c * C, y1: PAD, y2: PAD + env.height * C, class: "grid-line" }));
    for (let r = 1; r < env.height; r++) cellEdges.push(s("line", { y1: PAD + r * C, y2: PAD + r * C, x1: PAD, x2: PAD + env.width * C, class: "grid-line" }));

    const cliffCells = env.cliffs.map((st) => env.toCell(st));
    const cliffLabel = cliffCells.length
      ? s(
          "text",
          {
            x: PAD + ((cliffCells[0][1] + cliffCells[cliffCells.length - 1][1] + 1) / 2) * C,
            y: PAD + cliffCells[0][0] * C + C / 2 + 5,
            class: "cliff-label",
            "text-anchor": "middle",
          },
          this.hazardLabel ?? `The cliff · ${fmt(env.cliffReward, 0)} and back to S`,
        )
      : null;

    const overlays = [];
    for (const path of paths) {
      if (path.states.length < 2) continue;
      const pts = path.states.map((st) => this.center(st).join(",")).join(" ");
      overlays.push(
        s("polyline", {
          points: pts,
          fill: "none",
          stroke: path.color,
          "stroke-width": path.width ?? 3,
          "stroke-dasharray": path.dashed ? "6 5" : null,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
          opacity: path.opacity ?? 1,
        }),
      );
    }
    if (updated) overlays.push(s("polygon", { points: this.triangle(updated.s, updated.a), class: "hl-updated" }));
    if (target && target.a !== null && target.a !== undefined) overlays.push(s("polygon", { points: this.triangle(target.s, target.a), class: "hl-target" }));
    if (move) {
      const [x1, y1] = this.center(move.from);
      if (move.fellInto !== null && move.fellInto !== undefined) {
        const [x2, y2] = this.center(move.fellInto);
        const [x3, y3] = this.center(move.to);
        overlays.push(
          s("line", { x1, y1, x2, y2, class: "move-line" }),
          s("path", { d: `M${x2},${y2} Q${(x2 + x3) / 2},${Math.min(y2, y3) - C * 0.8} ${x3},${y3}`, class: "move-line fall" }),
        );
      } else if (move.to !== move.from) {
        const [x2, y2] = this.center(move.to);
        overlays.push(s("line", { x1, y1, x2, y2, class: "move-line" }));
      }
    }
    if (behavior && behavior.a !== null && behavior.a !== undefined && behavior.s !== agent) {
      const [bx, by] = this.center(behavior.s);
      overlays.push(arrow(bx, by, behavior.a, { length: C * 0.42, start: C * 0.12, color: "var(--env)", width: 2.6, head: 6.5 }));
    }
    if (agent !== null && agent !== undefined) {
      const [ax, ay] = this.center(agent);
      if (nextAction !== null && nextAction !== undefined) overlays.push(arrow(ax, ay, nextAction, { length: C * 0.44, start: C * 0.14, color: "var(--env)", width: 2.8, head: 7 }));
      overlays.push(s("circle", { cx: ax, cy: ay, r: C * 0.13, class: "agent" }));
    }

    const hits = [];
    for (let st = 0; st < env.nStates; st++) {
      if (env.cliffs.includes(st) || env.goals.includes(st) || env.walls.includes(st)) continue;
      const [r, c] = env.toCell(st);
      const hit = s("rect", { x: PAD + c * C, y: PAD + r * C, width: C, height: C, class: "hit", tabindex: this.plain ? null : "-1" });
      if (Q && !this.plain) {
        hit.addEventListener("pointerenter", () => this.showTip(st));
        hit.addEventListener("pointerleave", () => (this.tooltip.hidden = true));
      }
      hits.push(hit);
    }

    replace(this.svg, cells, cellEdges, gridLines, cliffLabel, overlays, labels, hits);
  }

  showTip(st) {
    const Q = this.state.Q;
    if (!Q) return;
    const [r, c] = this.env.toCell(st);
    const best = Math.max(...Q[st]);
    replace(
      this.tooltip,
      h("div", { class: "tt-head" }, `state ${st} · row ${r}, column ${c}`),
      Q[st].map((q, a) =>
        h("div", { class: "tt-row" }, h("span", {}, ARROWS[a]), h("b", {}, fmt(q, 3)), q >= best - 1e-9 ? h("span", {}, "greedy") : null),
      ),
    );
    this.tooltip.hidden = false;
    const scale = this.svg.getBoundingClientRect().width / this.width;
    const [cx, cy] = this.center(st);
    const left = (cx + C / 2) * scale + 6;
    const tw = this.tooltip.offsetWidth;
    const maxLeft = this.svg.getBoundingClientRect().width - tw;
    this.tooltip.style.left = `${Math.max(0, left > maxLeft ? (cx - C / 2) * scale - tw - 6 : left)}px`;
    this.tooltip.style.top = `${Math.max(0, (cy - C / 2) * scale)}px`;
  }
}

// Follow the greedy action from the start until the goal, a loop, or the step limit.
export function greedyPath(env, Q, maxSteps = 80) {
  const path = [env.start];
  const seen = new Set(path);
  let st = env.start;
  for (let i = 0; i < maxSteps; i++) {
    const a = Q[st].indexOf(Math.max(...Q[st]));
    const [next, , done] = env.move(st, a);
    path.push(next);
    if (done || seen.has(next)) break;
    seen.add(next);
    st = next;
  }
  return path;
}
