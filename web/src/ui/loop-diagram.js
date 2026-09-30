// The two loops of reinforcement learning, drawn once and lit up as they run.
//
//   control loop (every environment step):   policy → action → world → reward, next state → policy
//   learning loop (every learning step):     experience graph → Bellman backup → values → policy
//
// The world feeds the experience graph. Offline, that arrow is cut: the graph is a fixed dataset.
import { h, s } from "./dom.js";

const BOX = { w: 118, h: 34 };
const NODES = {
  policy: { x: 300, y: 36, label: "policy", sub: "picks an edge" },
  world: { x: 90, y: 150, label: "world", sub: "the true graph" },
  outcome: { x: 300, y: 150, label: "reward, next state", sub: "one new edge" },
  graph: { x: 510, y: 150, label: "experience graph", sub: "edges seen so far" },
  backup: { x: 510, y: 36, label: "values", sub: "Bellman backups" },
};

function box(key) {
  const n = NODES[key];
  return s(
    "g",
    { class: `ld-node ld-${key}` },
    s("rect", { x: n.x - BOX.w / 2, y: n.y - BOX.h / 2, width: BOX.w, height: BOX.h, rx: 8 }),
    s("text", { x: n.x, y: n.y - 2, "text-anchor": "middle", class: "ld-label" }, n.label),
    s("text", { x: n.x, y: n.y + 11, "text-anchor": "middle", class: "ld-sub" }, n.sub),
  );
}

function arrow(id, d, cls) {
  return s("path", { id, d, class: `ld-arrow ${cls}`, "marker-end": `url(#ld-head-${cls})` });
}

export class LoopDiagram {
  constructor({ offline = false } = {}) {
    this.svg = s(
      "svg",
      { viewBox: "0 0 600 196", role: "img", "aria-label": "The control loop and the learning loop" },
      s(
        "defs",
        {},
        ...["control", "learning", "cut"].map((cls) =>
          s("marker", { id: `ld-head-${cls}`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: `ld-head ${cls}` })),
        ),
      ),
      s("text", { x: 70, y: 22, class: "ld-title control" }, "control loop · every environment step"),
      s("text", { x: 600, y: 190, "text-anchor": "end", class: "ld-title learning" }, "learning loop · every learning step"),
      arrow("ld-a1", "M 262 53 Q 150 60 102 132", "control"),
      arrow("ld-a2", "M 150 150 L 240 150", "control"),
      arrow("ld-a3", "M 300 132 L 300 54", "control"),
      (this.feed = arrow("ld-a4", "M 360 150 L 450 150", offline ? "cut" : "control")),
      arrow("ld-a5", "M 510 132 L 510 54", "learning"),
      arrow("ld-a6", "M 450 36 L 360 36", "learning"),
      s("text", { x: 150, y: 78, class: "ld-edge-label" }, "action"),
      offline ? s("text", { x: 405, y: 140, "text-anchor": "middle", class: "ld-cut-label" }, "offline: no new edges") : null,
      ...Object.keys(NODES).map(box),
    );
    this.el = h("div", { class: "loop-diagram" }, this.svg);
    this.setOffline(offline);
  }

  setOffline(offline) {
    this.offline = offline;
    this.svg.classList.toggle("offline", offline);
  }

  // Briefly light up one loop.
  pulse(which) {
    const cls = which === "control" ? "pulse-control" : "pulse-learning";
    this.svg.classList.remove(cls);
    void this.svg.getBoundingClientRect();
    this.svg.classList.add(cls);
  }
}
