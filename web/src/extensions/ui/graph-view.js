// The experience graph: every state the agent has visited is a node, every
// transition it has observed is a directed edge. Laid out on the floor plan, so
// node (r, c) sits where cell (r, c) is on the map.
//
// It shows what the agent actually knows about the world, and how a route can
// be stitched together from edges seen in different episodes.
import { h, replace, s } from "../../ui/dom.js";
import { valueFill } from "./grid-view.js";

const C = 54;
const PAD = 6;

export class ExperienceGraph {
  constructor(env, { domain = 100 } = {}) {
    this.env = env;
    this.domain = domain;
    this.width = env.width * C + 2 * PAD;
    this.height = env.height * C + 2 * PAD;
    this.svg = s("svg", { viewBox: `0 0 ${this.width} ${this.height}`, role: "img", "aria-label": "Experience graph" });
    this.el = h("div", { class: "figure-scroll" }, h("div", { class: "rel grid-figure" }, this.svg));
  }

  center(state) {
    const [r, c] = this.env.toCell(state);
    return [PAD + c * C + C / 2, PAD + r * C + C / 2];
  }

  radius(visits) {
    return visits ? Math.min(13, 4 + 1.5 * Math.sqrt(visits)) : 2.5;
  }

  // edges: Map key -> { from, to, action, count, firstEpisode, fell }
  render({ visits, edges, Q, highlight, route = [], routeColor = "var(--ink)" }) {
    const env = this.env;
    const floor = [];
    for (let st = 0; st < env.nStates; st++) {
      const [r, c] = env.toCell(st);
      const x = PAD + c * C;
      const y = PAD + r * C;
      if (env.walls.includes(st)) floor.push(s("rect", { x, y, width: C, height: C, class: "g-wall" }));
      else if (env.cliffs.includes(st)) floor.push(s("rect", { x, y, width: C, height: C, class: "g-hazard" }));
    }

    const pairs = new Set([...edges.values()].map((e) => `${e.from}>${e.to}`));
    const edgeEls = [];
    for (const e of edges.values()) {
      const hl = highlight && highlight.from === e.from && highlight.to === e.to && (e.from !== e.to || highlight.action === e.action);
      edgeEls.push(this.edge(e, { both: pairs.has(`${e.to}>${e.from}`), hl, rFrom: this.radius(visits[e.from]), rTo: this.radius(visits[e.to]) }));
    }

    const routeEl =
      route.length > 1
        ? s("polyline", {
            points: route.map((st) => this.center(st).join(",")).join(" "),
            fill: "none",
            stroke: routeColor,
            "stroke-width": 12,
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
            opacity: 0.32,
          })
        : null;

    const nodes = [];
    for (let st = 0; st < env.nStates; st++) {
      if (env.walls.includes(st) || env.cliffs.includes(st)) continue;
      const [cx, cy] = this.center(st);
      const v = visits[st] ?? 0;
      const value = Q && v ? Math.max(...Q[st]) : 0;
      nodes.push(
        s("circle", {
          cx,
          cy,
          r: this.radius(v),
          class: v ? "g-node" : "g-node unvisited",
          style: v ? { fill: valueFill(value, this.domain) } : undefined,
        }),
      );
      if (st === env.start || env.goals.includes(st)) {
        nodes.push(s("text", { x: cx, y: cy - this.radius(v) - 4, class: "g-mark", "text-anchor": "middle" }, st === env.start ? "S" : "G"));
      }
    }
    if (highlight) {
      const [x1, y1] = this.center(highlight.from);
      const [x2, y2] = this.center(highlight.to);
      nodes.push(s("circle", { cx: x1, cy: y1, r: this.radius(visits[highlight.from]) + 3.5, class: "g-hl-from" }));
      nodes.push(s("circle", { cx: x2, cy: y2, r: this.radius(visits[highlight.to]) + 3.5, class: "g-hl-to" }));
    }
    replace(this.svg, floor, routeEl, edgeEls, nodes);
  }

  edge(e, { both, hl, rFrom, rTo }) {
    const cls = `g-edge${e.fell ? " fall" : ""}${hl ? " hl" : ""}`;
    const width = hl ? 3 : Math.min(2.6, 0.9 + 0.35 * Math.log2(e.count));
    const [x1, y1] = this.center(e.from);
    if (e.from === e.to) {
      // Bumping into a wall: a small loop on the side of the wall.
      const [dx, dy] = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],
      ][e.action];
      const ox = x1 + dx * (rFrom + 6);
      const oy = y1 + dy * (rFrom + 6);
      return s("circle", { cx: ox, cy: oy, r: 5, class: cls, "stroke-width": width, fill: "none" });
    }
    const [x2, y2] = this.center(e.to);
    const len = Math.hypot(x2 - x1, y2 - y1);
    const [ux, uy] = [(x2 - x1) / len, (y2 - y1) / len];
    const [px, py] = [-uy, ux];
    if (e.fell) {
      // Over the ledge: s' is the dock, far away. Curve through the bay.
      const mx = (x1 + x2) / 2;
      const my = Math.max(y1, y2) + C * 0.75;
      return s("path", { d: `M${x1},${y1 + 6} Q${mx},${my} ${x2 + 6},${y2 + 6}`, class: cls, "stroke-width": width, fill: "none" });
    }
    const off = both ? 3 : 0;
    const head = 5;
    const a = [x1 + ux * (rFrom + 1) + px * off, y1 + uy * (rFrom + 1) + py * off];
    const tip = [x2 - ux * (rTo + 1.5) + px * off, y2 - uy * (rTo + 1.5) + py * off];
    const b = [tip[0] - ux * 4, tip[1] - uy * 4];
    return s(
      "g",
      { class: cls },
      s("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], "stroke-width": width }),
      s("polygon", {
        points: `${tip[0]},${tip[1]} ${b[0] - ux * head + px * head * 0.7},${b[1] - uy * head + py * head * 0.7} ${b[0] - ux * head - px * head * 0.7},${b[1] - uy * head - py * head * 0.7}`,
      }),
    );
  }
}

// Where the edges of a route came from. `firstSeen[i]` is the episode that first observed edge i (null if
// never observed). `common` lists the episodes that observed every edge; if it is empty, no single episode
// drove the route and it was stitched together from pieces of different episodes.
export function routeOrigins(route, edges) {
  const byPair = new Map();
  for (const e of edges.values()) {
    const key = `${e.from}>${e.to}`;
    const prev = byPair.get(key);
    if (!prev) byPair.set(key, { first: e.firstEpisode, episodes: new Set(e.episodes) });
    else {
      prev.first = Math.min(prev.first, e.firstEpisode);
      for (const ep of e.episodes) prev.episodes.add(ep);
    }
  }
  const steps = route.slice(1).map((to, i) => byPair.get(`${route[i]}>${to}`) ?? null);
  const firstSeen = steps.map((st) => st?.first ?? null);
  let common = null;
  for (const st of steps) {
    if (!st) return { firstSeen, common: [] };
    common = common ? new Set([...common].filter((ep) => st.episodes.has(ep))) : new Set(st.episodes);
  }
  return { firstSeen, common: [...(common ?? [])].sort((a, b) => a - b) };
}
