// Lesson 05: the Bellman backup, and how repeating it finds the optimal policy (value iteration).
//
// One backup at a node looks along every edge out of it, adds the edge's reward to the discounted
// value where it lands, and keeps the best. Backing up every node again and again spreads value
// out from the chargers one edge per sweep; when nothing changes, the values are V* and the best
// edge at every node is the optimal policy. Every experiment has its own γ: dragging it recomputes
// the same backups at the new discount.
import viSource from "../../../visualrl/algorithms/tabular/value_iteration.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { optimalBackup, qFromV, ValueIteration } from "../rl/tabular/dp.js";
import { BackupPanel } from "../ui/backup-panel.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace } from "../ui/dom.js";
import { fmt, fmtShort } from "../ui/format.js";
import { gammaControl } from "../ui/gamma.js";
import { equation } from "../ui/math.js";
import { Mission } from "../ui/mission.js";
import { returnTable, unrollWalk } from "../ui/return-table.js";
import { WarehouseScene } from "../ui/scene3d.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";
import { world } from "./lesson01.js";

const UP = 0;
const SWITCH = 0.1 ** (1 / 3); // γ⁴ · 1 = γ⁷ · 10: below this the dock prefers the slow charger
const BATCH = 2000;

function namer(env, w) {
  return (st) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);
}

const argmax = (row) => row.indexOf(Math.max(...row));

// The best edge out of each node, drawn only where some value has arrived (before that, every edge ties at 0).
function bestEdges(env, model, V, gamma) {
  const Q = qFromV(model, V, gamma);
  const best = new Set();
  for (let st = 0; st < env.nStates; st++) {
    if (model.terminal[st] || V[st] === 0) continue;
    const m = Math.max(...Q[st]);
    Q[st].forEach((q, a) => q >= m - 1e-9 && best.add(`${st},${a}`));
  }
  return best;
}

// Follow the best edge from the dock, in a real (possibly slippery) copy of the world.
function driveGreedy(env, model, V, gamma, { slip = 0, seed = 3, maxSteps = 30 } = {}) {
  const world = GridWorld.chargingRoom({ slip, seed });
  let [s] = world.reset();
  const Q = qFromV(model, V, gamma);
  const steps = [];
  let ret = 0;
  for (let t = 0; t < maxSteps; t++) {
    if (V[s] === 0) break; // no value here yet: the robot has no reason to prefer any edge
    const a = argmax(Q[s]);
    const [s2, r, done, , info] = world.step(a);
    steps.push({ from: s, to: s2, action: a, fellInto: info.fell_into, reward: r });
    ret += gamma ** t * r;
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
  const A = { gamma: 0.9, V: new Array(env.nStates).fill(0), selected: null, applied: [], dragged: false };
  const graphA = new StateGraph(env, { goalLabels: w.labels, onNode: (st) => selectA(st) });
  const panelA = new BackupPanel({ name, title: "Backup" });
  const gammaA = gammaControl({ id: "l05-gamma-a", value: A.gamma, onInput: (g) => ((A.gamma = g), (A.dragged = true), replayA()) });
  const missionA = new Mission({
    title: "Do one backup yourself",
    goal: "All values start at 0. Back up two nodes and see where value comes from.",
    steps: [
      { text: "Click the node just above the slow charger (+1).", done: () => A.selected === aboveSlow },
      { text: "Press Back up this node. The panel shows every move: reward + γ × value where it lands. The best is ↓ with 1 + γ × 0 = 1.", done: () => A.applied.includes(aboveSlow) },
      { text: "Click the node to its left, then press Back up this node.", done: () => A.applied.includes(leftOfIt) },
      { text: "Drag γ. Your two backups are redone at the new γ: the node next to the charger stays at 1, the one behind it is always γ × 1.", done: () => A.dragged && Math.abs(A.gamma - 0.9) > 0.05 },
    ],
    conclusion: () =>
      `The node to the left got ${fmt(A.V[leftOfIt], 3)} = 0 + ${A.gamma.toFixed(2)} × 1: its best move is → to a node that now has value 1. A backup only looks one edge ahead. Value starts at the chargers, and each backup carries it one edge further back, multiplied by γ once per edge. That is the Gₜ = rₜ + γ Gₜ₊₁ of lesson 04, done at a node instead of along one walk.`,
  });

  function selectA(st) {
    if (model.terminal[st]) return;
    A.selected = st;
    renderA();
  }

  function applyA() {
    if (A.selected === null) return;
    A.V[A.selected] = optimalBackup(model, A.V, A.selected, A.gamma).value_after;
    A.applied.push(A.selected);
    renderA();
  }

  // The values are the result of the backups done so far; redo them, in order, at the current γ.
  function replayA() {
    A.V = new Array(env.nStates).fill(0);
    for (const st of A.applied) A.V[st] = optimalBackup(model, A.V, st, A.gamma).value_after;
    renderA();
  }

  function renderA() {
    graphA.render({ edges: edges.map((e) => ({ ...e, role: e.from === e.to ? "faint" : "plain" })), values: A.V, showValues: true, selected: A.selected });
    panelA.show(A.selected === null ? null : optimalBackup(model, A.V, A.selected, A.gamma));
    missionA.update();
  }

  // ---------- B and C: back up every node, again and again (C on a slippery floor) ----------
  function viBench({ slip, id, missionFor }) {
    const st = { slip, gamma: 0.9, vi: null, log: [], drives: [], batch: null, lastChanged: new Set(), selected: null };
    const benchModel = slip ? GridWorld.chargingRoom({ slip }).model() : model;
    const graph = new StateGraph(env, { goalLabels: w.labels, onNode: (n) => ((st.selected = benchModel.terminal[n] ? null : n), render()) });
    const panel = new BackupPanel({ name, title: "Backup" });
    const walkEl = slip ? null : h("section", { class: "backup" });
    const scene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], goalLabels: w.signs, caption: slip ? "Slippery floor: some moves slide in a random direction" : "The robot follows the best edge out of every node" });
    const logEl = h("div", { class: "sweep-log" });
    const note = h("div", { class: "callout" });
    const converged = () => st.log.length > 0 && st.log[st.log.length - 1].change < 1e-12;
    const gammaEl = gammaControl({ id, value: st.gamma, onInput: (g) => ((st.gamma = g), replay()) });
    const mission = missionFor(st, converged);

    function clear() {
      scene.stopRoutes();
      scene.place("main", env.start, UP);
      st.vi = new ValueIteration(benchModel, st.gamma);
      st.log = [];
      st.drives = [];
      st.batch = null;
      st.lastChanged = new Set();
    }

    function reset() {
      clear();
      render();
    }

    // Same number of sweeps at the new γ, or until nothing changes if the values had settled.
    function replay() {
      const target = st.vi.sweeps;
      const settled = converged();
      clear();
      sweep(settled ? 1000 : target);
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
      const d = driveGreedy(env, benchModel, st.vi.V, st.gamma, { slip, seed: 3 + st.drives.length });
      st.drives.push({ sweep: st.vi.sweeps, gamma: st.gamma, ...d });
      scene.playSteps("main", d.steps, { stepMs: 300 });
      render();
    }

    // Many drives, no animation: their average return is the dock's value.
    function driveMany() {
      const rets = Array.from({ length: BATCH }, (_, i) => driveGreedy(env, benchModel, st.vi.V, st.gamma, { slip, seed: 1000 + i, maxSteps: 300 }).ret);
      st.batch = { gamma: st.gamma, avg: rets.reduce((a, b) => a + b, 0) / BATCH, lo: Math.min(...rets), hi: Math.max(...rets), value: st.vi.V[env.start], settled: converged() };
      render();
    }

    function renderWalk() {
      if (!walkEl) return;
      const head = (text, right) => h("div", { class: "backup-head" }, h("strong", {}, text), h("span", { class: "grow" }), right ? h("span", { class: "note num" }, right) : null);
      if (st.selected === null) {
        replace(walkEl, head("The walk behind the value"), h("p", { class: "backup-empty" }, "Click a node. Its best edges are followed to the end of the walk, and the walk's multi-step return is added up."));
        return;
      }
      const V = st.vi.V;
      const Q = qFromV(benchModel, V, st.gamma);
      const walk = unrollWalk(env, st.selected, (n) => (V[n] === 0 ? null : argmax(Q[n])));
      const G = walk.length ? walk.returns(st.gamma)[0] : 0;
      const same = Math.abs(G - V[st.selected]) < 1e-9;
      replace(
        walkEl,
        head(`Following the best edges from ${name(st.selected)}`, `V = ${fmt(V[st.selected], 3)}`),
        walk.length === 0
          ? h("p", { class: "backup-empty" }, "No value has reached this node yet, so no edge out of it is better than another.")
          : [
              returnTable(env, walk, st.gamma, { name }),
              h(
                "p",
                { class: "return-note" },
                h("strong", {}, same ? `G₀ = ${fmt(G, 3)} = V(s). ` : `G₀ = ${fmt(G, 3)}, but V(s) is still ${fmt(V[st.selected], 3)}. `),
                same
                  ? "One-step backups, repeated, have computed exactly the multi-step return of this walk."
                  : "Nodes further along already know about a better walk; the next sweep carries it back to this node.",
              ),
            ],
      );
    }

    function render() {
      const best = bestEdges(env, benchModel, st.vi.V, st.gamma);
      graph.render({
        edges: edges.map((e) => ({ ...e, role: best.has(`${e.from},${e.action}`) ? "best" : e.from === e.to ? "faint" : "plain" })),
        values: st.vi.V,
        showValues: true,
        rings: st.lastChanged,
        selected: st.selected,
      });
      panel.show(st.selected === null ? null : optimalBackup(benchModel, st.vi.V, st.selected, st.gamma));
      renderWalk();
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
      const g = st.gamma.toFixed(2);
      let text;
      if (st.batch) {
        const b = st.batch;
        text = [
          h("strong", {}, `${BATCH} drives from the dock at γ = ${b.gamma.toFixed(2)}: average return ${fmt(b.avg, 3)}. `),
          `Single drives returned anything from ${fmt(b.lo, 2)} to ${fmt(b.hi, 2)}, because every drive slides differently. The dock's value is ${fmt(b.value, 3)}: a value is the average multi-step return${b.settled ? "" : " (once the sweeps have settled)"}, and the more drives you average, the closer they get.`,
        ];
      } else if (!last) {
        text = [h("strong", {}, `${st.vi.sweeps} sweeps so far at γ = ${g}. `), "Dashed rings: nodes whose value changed in the last sweep. Dark arrows: the best edge out of each node that has a value. Click a node to see its backup."];
      } else if (last.steps.length === 0) {
        text = [h("strong", {}, "The robot did not move. "), converged() ? "At γ = 0 only the next reward counts, and no move from the dock pays right away." : "The dock has no value yet, so no edge out of it is better than another."];
      } else {
        text = [
          h("strong", {}, `After ${last.sweep} sweeps at γ = ${last.gamma.toFixed(2)} the robot drove ${last.steps.length} edges and ended at the ${name(last.end)}${last.end === w.far || last.end === w.near ? "" : " (it stopped where no value had arrived yet)"}. `),
          `Its return was ${fmt(last.ret, 3)}; the dock's value is ${fmt(st.vi.V[env.start], 3)}.${slip ? " On a slippery floor the return varies from drive to drive; the value is its average." : ""}`,
        ];
      }
      replace(note, h("p", {}, text));
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
        button("Sweep until nothing changes", { onClick: () => sweep(1000) }),
        button("Reset values to 0", { kind: "ghost", onClick: reset }),
        h("span", { class: "spacer" }),
        gammaEl,
      ),
      h(
        "div",
        { class: "bench-grid part1-grid even" },
        h(
          "div",
          { class: "figure" },
          graph.el,
          h(
            "div",
            { class: "key" },
            h("span", { class: "key-item" }, h("span", { class: "key-swatch best-edge" }), "best edge out of the node"),
            h("span", { class: "key-item" }, h("span", { class: "key-swatch ring" }), "value changed in the last sweep"),
          ),
        ),
        h("div", { class: "stack" }, logEl, panel.el, walkEl),
      ),
      h("div", { class: "toolbar" }, button("Drive the optimal policy", { kind: "env", onClick: drive }), slip ? button(`Drive it ${BATCH} times and average`, { onClick: driveMany }) : null),
      note,
      scene.el,
    );
    reset();
    return { el, st, dispose: () => scene.dispose() };
  }

  // γ = 0.9 reference numbers for the conclusions, whatever γ the reader ends at.
  const at09 = (() => {
    const vi = new ValueIteration(model, 0.9);
    let first = null;
    while (vi.sweep().max_change >= 1e-12) if (first === null && vi.V[env.start] > 0) first = { sweep: vi.sweeps, dock: vi.V[env.start] };
    return { first, dock: vi.V[env.start] };
  })();

  const B = viBench({
    slip: 0,
    id: "l05-gamma-b",
    missionFor: (st, converged) =>
      new Mission({
        title: "Find the optimal policy by repeating the backup",
        goal: "A sweep backs up every node once. Sweep again and again, watch value spread out from the chargers, then drive the result. Then change γ and see the optimal policy itself change.",
        steps: [
          { text: "Press One sweep. Only the nodes one edge from a charger get a value.", done: () => st.vi.sweeps >= 1 },
          { text: "Keep pressing One sweep until the dock gets a value. Which charger does it come from?", done: () => st.vi.V[env.start] > 0 },
          { text: "Keep pressing until nothing changes any more. Then click the dock: under the backup, its best edges are followed to the end, and the walk's return equals V(dock).", done: () => converged() && st.selected === env.start },
          { text: "Press Drive the optimal policy.", done: () => st.drives.length > 0 && converged() },
          { text: "Drag γ below 0.46 (the values are recomputed until nothing changes) and drive again.", done: () => converged() && st.drives.some((d) => d.gamma < SWITCH) },
        ],
        conclusion: () =>
          `At γ = 0.9 the dock first got a value at sweep ${at09.first.sweep}: ${fmt(at09.first.dock, 3)} = 0.9⁴ × 1, from the slow charger 5 edges away. The fast charger is 8 edges away, so its value reached the dock only at sweep 8 and replaced the old one with ${fmt(at09.dock, 3)} = 0.9⁷ × 10, because a backup keeps the largest move. After that nothing changed: the values were V*, the dark arrows the optimal policy, and the walk along them returned exactly V*(dock). One-edge backups, repeated, had computed an 8-edge return. At γ = ${st.gamma.toFixed(2)} the same procedure gives V*(dock) = ${fmt(st.vi.V[env.start], 3)} and a policy that drives to the slow charger: γ⁴ × 1 now beats γ⁷ × 10. Value iteration did not change; the task did. γ is part of what optimal means.`,
      }),
  });

  const C = viBench({
    slip: 0.2,
    id: "l05-gamma-c",
    missionFor: (st, converged) =>
      new Mission({
        title: "Add uncertainty: a slippery floor",
        goal: "Now 20% of moves slide in a random direction. The backup averages over where each move can end up.",
        steps: [
          { text: "Click the node at (3,2), right above the ledge. In its backup, → has several outcomes, and one goes over the ledge.", done: () => st.selected === aboveLedge },
          { text: "Press Sweep until nothing changes.", done: () => converged() },
          { text: `Press Drive it ${BATCH} times and average. Each drive is a different walk; compare their average return with V*(dock).`, done: () => st.batch?.settled === true },
          { text: "Drag γ down to 0.30 or less and look at the dock's value.", done: () => converged() && st.gamma <= 0.305 },
        ],
        conclusion: () =>
          `Each move is now worth the average of its outcomes, Σ p × (r + γ V(s′)), and a move next to the ledge has a small chance of −10 and a trip back to the dock. On the dry floor the two 8-edge routes to the fast charger tied; here the optimal policy takes the top one, away from the ledge. Because walks are random, V*(dock) is an average multi-step return, and averaging many real drives gives the same number. At γ = ${st.gamma.toFixed(2)} the dock is worth ${fmt(st.vi.V[env.start], 3)}${st.vi.V[env.start] < 0 ? ", less than nothing" : ""}. A robot that looks about 1/(1−γ) = ${fmtShort(1 / (1 - st.gamma), 1)} edges ahead sees the 5% chance of sliding over the ledge next to the dock, but the charger five edges away hardly counts. A near-sighted robot on a risky floor sees mostly the risk.`,
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
      lead: "Lesson 04 showed that a value is a whole walk's return, and that it can be computed one edge at a time: V(s) = r + γV(s′). Here you use that to find the optimal policy. The Bellman backup looks one edge ahead from a node and keeps the best move. Repeat it everywhere until nothing changes, and the best edge at every node is the optimal policy. γ stays a dial throughout: change it, and the optimal policy changes with it.",
      concepts: ["Bellman backup", "Bellman optimality equation", "value iteration", "V*", "optimal policy", "expectation over outcomes", "discount γ"],
    }),
    step("Question", h("p", { class: "question" }, "With the whole graph in front of you, how do you find the best move at every node?")),
    step("Predict", predictBlock),
    wideStep(
      "One backup",
      prose("One node, one calculation. For every move out of the node: the reward on the edge, plus γ times the value of the node it lands on. The node's new value is the largest of the four."),
      h(
        "div",
        { class: "bench" },
        missionA.el,
        h(
          "div",
          { class: "toolbar" },
          button("Back up this node", { kind: "learn", onClick: applyA }),
          button("Reset all values to 0", { kind: "ghost", onClick: () => ((A.V = new Array(env.nStates).fill(0)), (A.applied = []), renderA()) }),
          h("span", { class: "spacer" }),
          gammaA,
        ),
        h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graphA.el), h("div", { class: "figure" }, panelA.el)),
      ),
    ),
    wideStep(
      "Every node, repeated",
      prose("A sweep backs up every node once, all from the values before the sweep. Repeating sweeps until nothing changes is called value iteration. Dragging γ redoes the same number of sweeps at the new γ (or sweeps until nothing changes, if the values had already settled)."),
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
        "The Bellman optimality backup: Gₜ = rₜ + γGₜ₊₁ from lesson 04, with the best move chosen at every node. On a deterministic floor the sum has one term, the edge itself; on a slippery floor it averages over where the move may end.",
      ),
      equation(
        "V^*(s) = \\max_a \\sum_{s'} P(s' \\mid s,a)\\,\\big[\\, r + \\gamma V^*(s') \\,\\big], \\qquad \\pi^*(s) = \\arg\\max_a \\sum_{s'} P(s' \\mid s,a)\\,\\big[\\, r + \\gamma V^*(s') \\,\\big]",
        "When a sweep changes nothing, V satisfies the Bellman optimality equation: it is V*, the largest expected return from every node. The optimal policy is the move that achieves the max. Both depend on γ.",
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
