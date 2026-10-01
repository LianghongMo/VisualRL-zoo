// Lesson 05: the Bellman backup, and how repeating it finds the optimal policy (value iteration).
//
// One backup at a node looks along every edge out of it, adds the edge's reward to the discounted
// value where it lands, and keeps the best. Backing up every node again and again spreads value
// out from the chargers one edge per sweep; when nothing changes, the values are V* and the best
// edge at every node is the optimal policy.
import viSource from "../../../visualrl/algorithms/tabular/value_iteration.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { optimalBackup, qFromV, ValueIteration } from "../rl/tabular/dp.js";
import { BackupPanel } from "../ui/backup-panel.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace } from "../ui/dom.js";
import { fmt } from "../ui/format.js";
import { equation } from "../ui/math.js";
import { Mission } from "../ui/mission.js";
import { WarehouseScene } from "../ui/scene3d.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";
import { world } from "./lesson01.js";

const GAMMA = 0.9;
const UP = 0;

function namer(env, w) {
  return (st) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);
}

// The best edge out of each node, drawn only where some value has arrived (before that, every edge ties at 0).
function bestEdges(env, model, V) {
  const Q = qFromV(model, V, GAMMA);
  const best = new Set();
  for (let st = 0; st < env.nStates; st++) {
    if (model.terminal[st] || V[st] === 0) continue;
    const m = Math.max(...Q[st]);
    Q[st].forEach((q, a) => q >= m - 1e-9 && best.add(`${st},${a}`));
  }
  return best;
}

// Follow the best edge from the dock, in a real (possibly slippery) copy of the world.
function driveGreedy(env, model, V, { slip = 0, seed = 3, maxSteps = 30 } = {}) {
  const world = GridWorld.chargingRoom({ slip, seed });
  let [s] = world.reset();
  const steps = [];
  let ret = 0;
  for (let t = 0; t < maxSteps; t++) {
    if (V[s] === 0) break; // no value here yet: the robot has no reason to prefer any edge
    const Q = qFromV(model, V, GAMMA)[s];
    const a = Q.indexOf(Math.max(...Q));
    const [s2, r, done, , info] = world.step(a);
    steps.push({ from: s, to: s2, action: a, fellInto: info.fell_into, reward: r });
    ret += GAMMA ** t * r;
    s = s2;
    if (done) break;
  }
  return { steps, ret, end: s };
}

export function mount(root) {
  const w = world();
  const { env } = w;
  const name = namer(env, w);
  const edges = trueEdges(env);
  const model = env.model();
  const aboveSlow = env.toState(3, 4);
  const leftOfIt = env.toState(3, 3);
  const aboveLedge = env.toState(3, 2);

  // ---------- A: one backup by hand ----------
  const A = { V: new Array(env.nStates).fill(0), selected: null, applied: [] };
  const graphA = new StateGraph(env, { goalLabels: w.labels, onNode: (st) => selectA(st) });
  const panelA = new BackupPanel({ name, title: "Backup" });
  const missionA = new Mission({
    title: "Do one backup yourself",
    goal: "All values start at 0. Back up two nodes and see where value comes from.",
    steps: [
      { text: "Click the node just above the slow charger (+1).", done: () => A.selected === aboveSlow },
      { text: "Press Back up this node. The panel shows every move: reward + 0.9 × value where it lands. The best is ↓ with 1 + 0 = 1.", done: () => A.applied.includes(aboveSlow) },
      { text: "Click the node to its left, then press Back up this node.", done: () => A.applied.includes(leftOfIt) },
    ],
    conclusion: () =>
      `The node to the left got ${fmt(A.V[leftOfIt], 3)} = 0 + 0.9 × 1: its best move is → to a node that now has value 1. A backup only looks one edge ahead. Value starts at the chargers, and each backup carries it one edge further back, shrunk by γ = 0.9 once per edge.`,
  });

  function selectA(st) {
    if (model.terminal[st]) return;
    A.selected = st;
    renderA();
  }

  function applyA() {
    if (A.selected === null) return;
    A.V[A.selected] = optimalBackup(model, A.V, A.selected, GAMMA).value_after;
    A.applied.push(A.selected);
    renderA();
  }

  function renderA() {
    graphA.render({ edges: edges.map((e) => ({ ...e, role: e.from === e.to ? "faint" : "plain" })), values: A.V, showValues: true, selected: A.selected });
    panelA.show(A.selected === null ? null : optimalBackup(model, A.V, A.selected, GAMMA));
    missionA.update();
  }

  // ---------- B and C: back up every node, again and again (C on a slippery floor) ----------
  function viBench({ slip, missionFor }) {
    const st = { slip, vi: null, log: [], drives: [], lastChanged: new Set(), selected: null };
    const benchModel = slip ? GridWorld.chargingRoom({ slip }).model() : model;
    const graph = new StateGraph(env, { goalLabels: w.labels, onNode: (n) => ((st.selected = benchModel.terminal[n] ? null : n), render()) });
    const panel = new BackupPanel({ name, title: "Backup" });
    const scene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], goalLabels: w.signs, caption: slip ? "Slippery floor: some moves slide in a random direction" : "The robot follows the best edge out of every node" });
    const logEl = h("div", { class: "sweep-log" });
    const note = h("div", { class: "callout" });
    const converged = () => st.log.length > 0 && st.log[st.log.length - 1].change < 1e-12;
    const mission = missionFor(st, converged);

    function reset() {
      st.vi = new ValueIteration(benchModel, GAMMA);
      st.log = [];
      st.drives = [];
      st.lastChanged = new Set();
      scene.place("main", env.start, UP);
      render();
    }

    function sweep(times) {
      for (let i = 0; i < times; i++) {
        const t = st.vi.sweep();
        st.lastChanged = new Set(t.values_after.map((v, n) => (Math.abs(v - t.values_before[n]) > 1e-12 ? n : -1)).filter((n) => n >= 0));
        st.log.push({ sweep: t.sweep, dock: t.values_after[env.start], valued: t.values_after.filter((v, n) => v !== 0 && !benchModel.terminal[n]).length, change: t.max_change });
        if (t.max_change < 1e-12) break;
      }
      render();
    }

    function drive() {
      const d = driveGreedy(env, benchModel, st.vi.V, { slip, seed: 3 + st.drives.length });
      st.drives.push({ sweep: st.vi.sweeps, ...d });
      scene.playSteps("main", d.steps, { stepMs: 300 });
      render();
    }

    function render() {
      const best = bestEdges(env, benchModel, st.vi.V);
      graph.render({
        edges: edges.map((e) => ({ ...e, role: best.has(`${e.from},${e.action}`) ? "best" : e.from === e.to ? "faint" : "plain" })),
        values: st.vi.V,
        showValues: true,
        rings: st.lastChanged,
        selected: st.selected,
      });
      panel.show(st.selected === null ? null : optimalBackup(benchModel, st.vi.V, st.selected, GAMMA));
      replace(
        logEl,
        h(
          "table",
          {},
          h("thead", {}, h("tr", {}, ["sweep", "nodes with a value", "V(dock)", "largest change"].map((c) => h("th", {}, c)))),
          h(
            "tbody",
            {},
            st.log.length
              ? st.log.map((x, i) => h("tr", { class: Math.abs(x.dock - (i > 0 ? st.log[i - 1].dock : 0)) > 0.1 ? "highlight" : "" }, h("td", {}, x.sweep), h("td", {}, x.valued), h("td", {}, fmt(x.dock, 3)), h("td", {}, fmt(x.change, 3))))
              : h("tr", {}, h("td", { colspan: 4, style: { textAlign: "left", fontFamily: "var(--font-body)" } }, "No sweep yet. Every value is 0.")),
          ),
        ),
      );
      const last = st.drives[st.drives.length - 1];
      replace(
        note,
        h(
          "p",
          {},
          !last
            ? [h("strong", {}, `${st.vi.sweeps} sweeps so far. `), "Dashed rings: nodes whose value changed in the last sweep. Dark arrows: the best edge out of each node that has a value. Click a node to see its backup."]
            : last.steps.length === 0
              ? [h("strong", {}, "The robot did not move. "), "The dock has no value yet, so no edge out of it is better than another."]
              : [
                  h("strong", {}, `After ${last.sweep} sweeps the robot drove ${last.steps.length} edges and ended at the ${name(last.end)}${last.end === w.far || last.end === w.near ? "" : " (it stopped where no value had arrived yet)"}. `),
                  `Its return was ${fmt(last.ret, 3)}; the dock's value is ${fmt(st.vi.V[env.start], 3)}.${slip ? " On a slippery floor the return varies from drive to drive; the value is its average." : ""}`,
                ],
        ),
      );
      mission.update();
    }

    const el = h(
      "div",
      { class: "bench" },
      mission.el,
      h(
        "div",
        { class: "toolbar" },
        button("One sweep", { kind: "learn", onClick: () => sweep(1) }),
        button("Sweep until nothing changes", { onClick: () => sweep(100) }),
        button("Reset values to 0", { kind: "ghost", onClick: reset }),
        h("span", { class: "spacer" }),
        button("Drive the optimal policy", { kind: "env", onClick: drive }),
      ),
      h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graph.el, h("div", { class: "key" },
        h("span", { class: "key-item" }, h("span", { class: "key-swatch best-edge" }), "best edge out of the node"),
        h("span", { class: "key-item" }, h("span", { class: "key-swatch ring" }), "value changed in the last sweep"),
      )), h("div", { class: "figure" }, logEl, panel.el)),
      note,
      scene.el,
    );
    reset();
    return { el, st, dispose: () => scene.dispose() };
  }

  const B = viBench({
    slip: 0,
    missionFor: (st, converged) =>
      new Mission({
        title: "Find the optimal policy by repeating the backup",
        goal: "A sweep backs up every node once. Sweep again and again, watch value spread out from the chargers, then drive the result.",
        steps: [
          { text: "Press One sweep. Only the nodes one edge from a charger get a value.", done: () => st.vi.sweeps >= 1 },
          { text: "Keep pressing One sweep until the dock gets a value. Which charger does it come from?", done: () => st.vi.V[env.start] > 0 },
          { text: "Keep pressing until nothing changes any more.", done: () => converged() },
          { text: "Press Drive the optimal policy.", done: () => st.drives.some((d) => d.end === w.far) },
        ],
        conclusion: () => {
          const first = st.log.find((x) => x.dock > 0);
          return `The dock first got a value at sweep ${first?.sweep ?? 5}: ${fmt(first?.dock ?? 0.656, 3)} = 0.9⁴ × 1, from the slow charger 5 edges away. The fast charger is 8 edges away, so its value reached the dock only at sweep 8 and replaced the old one with ${fmt(st.vi.V[env.start], 3)} = 0.9⁷ × 10, because a backup keeps the largest move. After that nothing changed: these values are V*. The dark arrows, the best edge at every node, are the optimal policy, and driving it returns exactly V*(dock). Bellman backups everywhere, repeated until nothing changes: that is value iteration.`;
        },
      }),
  });

  const C = viBench({
    slip: 0.2,
    missionFor: (st, converged) =>
      new Mission({
        title: "Add uncertainty: a slippery floor",
        goal: "Now 20% of moves slide in a random direction. The backup averages over where each move can end up.",
        steps: [
          { text: "Click the node at (3,2), right above the ledge. In its backup, → has several outcomes, and one goes over the ledge.", done: () => st.selected === aboveLedge },
          { text: "Press Sweep until nothing changes.", done: () => converged() },
          { text: "Press Drive the optimal policy and watch its route.", done: () => converged() && st.drives.length > 0 },
        ],
        conclusion: () =>
          `Each move is now worth the average of its outcomes, Σ p × (r + 0.9 × V(s′)), and a move next to the ledge has a small chance of −10 and a trip back to the dock. The backup still keeps the best move, but best is now an expected value. On the dry floor the two 8-edge routes to the fast charger tied; here the optimal policy takes the top one, away from the ledge, and V*(dock) is ${fmt(st.vi.V[env.start], 3)} instead of 4.783. Same backup, same repetition; only the average inside is new.`,
      }),
  });

  const predictBlock = predict({
    question: "All values start at 0. You back up every node once (one sweep). Which nodes have a value afterwards?",
    choices: [{ label: "Every node" }, { label: "Only the nodes one edge from a charger" }, { label: "Only the dock" }],
    answer: 1,
    explain: () =>
      "A backup at a node looks one edge ahead. In the first sweep, the only edges that lead to anything worth more than 0 are the ones into a charger, so only their start nodes get a value. Each further sweep reaches one edge further.",
  });

  root.append(
    lessonHeader("05", {
      lead: "Lesson 01 showed the optimal policy as a tree of best edges. Here you compute it yourself, with one operation repeated: the Bellman backup. It looks one edge ahead from a node and keeps the best. Repeat it everywhere until nothing changes, and the best edge at every node is the optimal policy.",
      concepts: ["Bellman backup", "Bellman optimality equation", "value iteration", "V*", "optimal policy", "expectation over outcomes"],
    }),
    step("Question", h("p", { class: "question" }, "With the whole graph in front of you, how do you find the best move at every node?")),
    step("Predict", predictBlock),
    wideStep(
      "One backup",
      prose("One node, one calculation. For every move out of the node: the reward on the edge, plus 0.9 times the value of the node it lands on. The node's new value is the largest of the four."),
      h(
        "div",
        { class: "bench" },
        missionA.el,
        h("div", { class: "toolbar" }, button("Back up this node", { kind: "learn", onClick: applyA }), button("Reset all values to 0", { kind: "ghost", onClick: () => ((A.V = new Array(env.nStates).fill(0)), (A.applied = []), renderA()) })),
        h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graphA.el), h("div", { class: "figure" }, panelA.el)),
      ),
    ),
    wideStep(
      "Every node, repeated",
      prose("A sweep backs up every node once, all from the values before the sweep. Repeating sweeps until nothing changes is called value iteration."),
      B.el,
    ),
    wideStep(
      "A slippery floor",
      prose("The same procedure when moves are uncertain. Each edge now leads to a mix of places, and the backup takes the average."),
      C.el,
    ),
    step(
      "Equation",
      equation(
        "V(s) \\leftarrow \\max_a \\sum_{s'} P(s' \\mid s,a)\\,\\big[\\, r(s,a,s') + \\gamma\\, V(s') \\,\\big]",
        "The Bellman optimality backup. On a deterministic floor the sum has one term, the edge itself; on a slippery floor it averages over where the move may end.",
      ),
      equation(
        "V^*(s) = \\max_a \\sum_{s'} P(s' \\mid s,a)\\,\\big[\\, r + \\gamma V^*(s') \\,\\big], \\qquad \\pi^*(s) = \\arg\\max_a \\sum_{s'} P(s' \\mid s,a)\\,\\big[\\, r + \\gamma V^*(s') \\,\\big]",
        "When a sweep changes nothing, V satisfies the Bellman optimality equation: it is V*. The optimal policy is the move that achieves the max at every node. With γ < 1, repeated sweeps always get there.",
      ),
    ),
    step("Code", prose("The whole of value iteration, from visualrl/algorithms/tabular/value_iteration.py. q_from_v computes the inner sum for every node and move; the sweep keeps the max."), codeBlock(extractDef(viSource, "sweep", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "ValueIteration.sweep" })),
    lessonFooter("05"),
  );
  renderA();
  return () => {
    B.dispose();
    C.dispose();
  };
}
