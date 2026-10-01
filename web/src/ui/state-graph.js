// A GridWorld drawn as the graph it is: one node per state, laid out on the floor plan,
// one directed edge per move. Used by Part I for the true graph, the experience graph,
// datasets, policies and walks.
import { h, replace, s } from "./dom.js";
import { fmt, fmtShort } from "./format.js";
import { inkOn, valueFill } from "./grid-view.js";

const C = 64;
const PAD = 8;
const R = 11;
const DIRS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

// Every move of a deterministic GridWorld as an edge { from, action, to, reward, terminated, fell }.
export function trueEdges(env) {
  const out = [];
  for (let st = 0; st < env.nStates; st++) {
    if (env.walls.includes(st) || env.cliffs.includes(st) || env.goals.includes(st)) continue;
    for (let a = 0; a < 4; a++) {
      const [to, reward, terminated] = env.move(st, a);
      const [r, c] = env.toCell(st);
      const [dx, dy] = DIRS[a];
      const target = env.toState(r + dy, c + dx);
      out.push({ from: st, action: a, to, reward, terminated, fell: env.cliffs.includes(target) ? target : null });
    }
  }
  return out;
}

export class StateGraph {
  constructor(env, { domain = 10, goalLabels = {}, onNode = null } = {}) {
    this.env = env;
    this.domain = domain;
    this.goalLabels = goalLabels;
    this.onNode = onNode;
    this.width = env.width * C + 2 * PAD;
    this.height = env.height * C + 2 * PAD;
    this.svg = s("svg", { viewBox: `0 0 ${this.width} ${this.height}`, role: "img", "aria-label": "State graph" });
    this.tooltip = h("div", { class: "tooltip", hidden: true });
    this.el = h("div", { class: "figure-scroll" }, h("div", { class: "rel state-graph" }, this.svg, this.tooltip));
  }

  center(st) {
    const [r, c] = this.env.toCell(st);
    return [PAD + c * C + C / 2, PAD + r * C + C / 2];
  }

  // view: {
  //   edges: [{ from, action, to, reward, terminated, fell, role }]   role: "plain" | "faint" | "best" | "walk" | "a" | "b" | "hl"
  //   values: number[] | null, showValues: bool, stubs: { [state]: actions[] }, rings: Set, robot: state, known: Set (nodes to draw solid)
  // }
  render(view) {
    this.view = view;
    this.tooltip.hidden = true; // a tooltip from before this render would show old values
    const env = this.env;
    const { edges = [], values = null, stubs = {}, rings = new Set(), robot = null, known = null, showValues = false, route = [], selected = null } = view;
    const floor = [];
    for (let st = 0; st < env.nStates; st++) {
      const [r, c] = env.toCell(st);
      const x = PAD + c * C;
      const y = PAD + r * C;
      if (env.walls.includes(st)) floor.push(s("rect", { x, y, width: C, height: C, class: "sg-wall" }));
      else if (env.cliffs.includes(st)) floor.push(s("rect", { x, y, width: C, height: C, class: "sg-ledge" }));
    }
    const ledge = env.cliffs.map((st) => env.toCell(st));
    const ledgeLabel = ledge.length
      ? s("text", { x: PAD + ((ledge[0][1] + ledge[ledge.length - 1][1] + 1) / 2) * C, y: PAD + ledge[0][0] * C + C - 9, "text-anchor": "middle", class: "sg-ledge-label" }, `ledge ${fmtShort(env.cliffReward)}`)
      : null;

    const pairs = new Set(edges.filter((e) => !e.fell && e.from !== e.to).map((e) => `${e.from}>${e.to}`));
    const order = { faint: 0, plain: 1, a: 2, b: 2, best: 3, walk: 4, hl: 5 };
    const edgeEls = [...edges].sort((p, q) => (order[p.role] ?? 1) - (order[q.role] ?? 1)).map((e) => this.edge(e, pairs.has(`${e.to}>${e.from}`)));

    const stubEls = [];
    for (const [st, actions] of Object.entries(stubs)) {
      const [cx, cy] = this.center(Number(st));
      for (const a of actions) {
        const [dx, dy] = DIRS[a];
        stubEls.push(s("line", { x1: cx + dx * (R + 2), y1: cy + dy * (R + 2), x2: cx + dx * (R + 12), y2: cy + dy * (R + 12), class: "sg-stub" }));
      }
    }

    const nodes = [];
    for (let st = 0; st < env.nStates; st++) {
      if (env.walls.includes(st) || env.cliffs.includes(st)) continue;
      const [cx, cy] = this.center(st);
      const isKnown = !known || known.has(st);
      const goal = env.goals.includes(st);
      const v = values ? values[st] : 0;
      const fill = values && isKnown ? valueFill(v, this.domain) : "var(--sheet)";
      if (goal) {
        nodes.push(s("rect", { x: cx - R - 3, y: cy - R - 3, width: 2 * R + 6, height: 2 * R + 6, rx: 6, class: `sg-goal${isKnown ? "" : " unknown"}` }));
        nodes.push(s("text", { x: cx, y: cy + 4, "text-anchor": "middle", class: "sg-goal-text" }, this.goalLabels[st] ?? "G"));
      } else {
        nodes.push(s("circle", { cx, cy, r: R, class: `sg-node${isKnown ? "" : " unknown"}`, style: { fill } }));
        if (showValues && values && isKnown) nodes.push(s("text", { x: cx, y: cy + 3.5, "text-anchor": "middle", class: "sg-value", style: { fill: inkOn(v, this.domain) } }, fmt(v, v >= 9.995 || v <= -9.995 ? 0 : 1)));
      }
      if (st === env.start) nodes.push(s("text", { x: cx, y: cy - R - 5, "text-anchor": "middle", class: "sg-label" }, "dock"));
      if (rings.has(st)) nodes.push(s("circle", { cx, cy, r: R + 5, class: "sg-ring" }));
      if (st === selected) nodes.push(s("circle", { cx, cy, r: R + 6, class: "sg-selected" }));
      const hit = s("circle", { cx, cy, r: R + 8, class: "hit", "data-state": st });
      hit.addEventListener("pointerenter", () => this.tip(st));
      hit.addEventListener("pointerleave", () => (this.tooltip.hidden = true));
      if (this.onNode) hit.addEventListener("click", () => this.onNode(st));
      nodes.push(hit);
    }
    const robotEl = robot !== null && robot !== undefined ? (() => {
      const [cx, cy] = this.center(robot);
      return s("circle", { cx: cx + R * 0.95, cy: cy - R * 0.95, r: 5.5, class: "sg-robot" });
    })() : null;
    // A route drawn as a wide translucent band under the edges, so the edges on it stay readable.
    const band = [];
    for (let i = 1; i < route.length; i++) {
      const [x1, y1] = this.center(route[i - 1]);
      const [x2, y2] = this.center(route[i]);
      if (route[i - 1] !== route[i] && Math.abs(x1 - x2) + Math.abs(y1 - y2) <= C * 1.01) band.push(s("line", { x1, y1, x2, y2, class: "sg-route" }));
    }
    replace(this.svg, floor, ledgeLabel, band, edgeEls, stubEls, nodes, robotEl);
  }

  edge(e, both) {
    const [x1, y1] = this.center(e.from);
    const cls = `sg-edge ${e.role ?? "plain"}`;
    if (e.fell !== null && e.fell !== undefined) {
      // Over the ledge: s' is the dock. Dip into the ledge on the way.
      const [lx, ly] = this.center(e.fell);
      const [x2, y2] = this.center(e.to);
      return s("path", { d: `M${x1},${y1 + R} Q${lx},${ly + C * 0.45} ${x2 + R * 0.7},${y2 + R * 0.7}`, class: `${cls} fall` });
    }
    if (e.from === e.to) {
      const [dx, dy] = DIRS[e.action];
      return s("circle", { cx: x1 + dx * (R + 7), cy: y1 + dy * (R + 7), r: 5, class: `${cls} loop` });
    }
    const [x2, y2] = this.center(e.to);
    const len = Math.hypot(x2 - x1, y2 - y1);
    const [ux, uy] = [(x2 - x1) / len, (y2 - y1) / len];
    const [px, py] = [-uy, ux];
    const off = both ? 3.2 : 0;
    const endR = this.env.goals.includes(e.to) ? R + 4 : R;
    const a = [x1 + ux * (R + 1) + px * off, y1 + uy * (R + 1) + py * off];
    const tip = [x2 - ux * (endR + 2) + px * off, y2 - uy * (endR + 2) + py * off];
    const head = e.role === "best" || e.role === "walk" || e.role === "hl" ? 6.5 : 5;
    const b = [tip[0] - ux * head, tip[1] - uy * head];
    return s(
      "g",
      { class: cls },
      s("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }),
      s("polygon", { points: `${tip[0]},${tip[1]} ${b[0] + px * head * 0.6},${b[1] + py * head * 0.6} ${b[0] - px * head * 0.6},${b[1] - py * head * 0.6}` }),
    );
  }

  tip(st) {
    const { values, known } = this.view;
    const [r, c] = this.env.toCell(st);
    const name = st === this.env.start ? "the dock" : this.env.goals.includes(st) ? `a charger (${this.goalLabels[st] ?? ""})` : `state (${r},${c})`;
    replace(
      this.tooltip,
      h("div", { class: "tt-head" }, name),
      values && (!known || known.has(st)) && !this.env.goals.includes(st) ? h("div", { class: "tt-row" }, h("b", {}, fmt(values[st], 3)), h("span", {}, "value")) : null,
    );
    this.tooltip.hidden = false;
    const scale = this.svg.getBoundingClientRect().width / this.width;
    const [cx, cy] = this.center(st);
    this.tooltip.style.left = `${(cx + R + 6) * scale}px`;
    this.tooltip.style.top = `${Math.max(0, (cy - R - 20) * scale)}px`;
  }
}
