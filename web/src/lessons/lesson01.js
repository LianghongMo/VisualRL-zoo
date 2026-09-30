// Lesson 01: the world as a graph. States are nodes, moves are edges, rewards are edge weights,
// a policy picks an edge at every node, and optimal control is a best-path problem.
import bellmanSource from "../../../visualrl/algorithms/tabular/bellman.py";
import { Trajectory, makeTransition } from "../rl/core.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { optimalValues } from "../rl/tabular/dp.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, shortcuts, slider } from "../ui/dom.js";
import { ARROWS, fmt, fmtShort } from "../ui/format.js";
import { LineChart } from "../ui/line-chart.js";
import { Mission } from "../ui/mission.js";
import { equation, tex } from "../ui/math.js";
import { WarehouseScene } from "../ui/scene3d.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";

const [UP, RIGHT, DOWN, LEFT] = [0, 1, 2, 3];
const NEAR_ROUTE = [UP, RIGHT, RIGHT, RIGHT, DOWN];
const FAR_ROUTE = [UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, UP, UP];

export function world() {
  const env = GridWorld.chargingRoom();
  const far = env.toState(1, 6);
  const near = env.toState(4, 4);
  return { env, far, near, labels: { [far]: "+10", [near]: "+1" }, signs: { [far]: "FAST +10", [near]: "SLOW +1" } };
}

const cellName = (env, st, w) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);

function routeTrajectory(env, actions) {
  let [s] = env.reset();
  const traj = new Trajectory();
  for (const a of actions) {
    const [s2, r, done] = env.step(a);
    traj.append(makeTransition({ state: s, action: a, reward: r, next_state: s2, terminated: done }));
    s = s2;
    if (done) break;
  }
  return traj;
}

// The best edges out of every node, for the reference solution (ties all count).
function bestEdges(env, Q) {
  const out = [];
  for (let st = 0; st < env.nStates; st++) {
    if (env.walls.includes(st) || env.cliffs.includes(st) || env.goals.includes(st)) continue;
    const best = Math.max(...Q[st]);
    Q[st].forEach((q, a) => q >= best - 1e-9 && out.push(`${st},${a}`));
  }
  return new Set(out);
}

export function mount(root) {
  const w = world();
  const { env } = w;
  const edges = trueEdges(env);
  let gamma = 0.9;
  let showOptimal = false;
  let walk = [];
  let [state] = env.reset();
  let solved = false;
  const finished = []; // completed walks: { goal, G, gamma }

  const scene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], goalLabels: w.signs, caption: "Drive with the buttons or the arrow keys · drag to look around" });
  const graph = new StateGraph(env, { goalLabels: w.labels });
  const table = h("div", { class: "walk-table" });
  const status = h("div", { class: "callout" });
  const challenge = h("div", { class: "callout" });
  const chart = new LineChart({
    height: 240,
    xLabel: "discount γ",
    yLabel: "return from the dock",
    xDomain: [0, 1],
    xFormat: (v) => v.toFixed(1),
    yFormat: (v) => fmtShort(v, 1),
    caption: "Each curve is the return of one fixed walk, recomputed for every γ.",
  });
  // Separate environments: rolling these walks out must not move the robot the reader drives.
  const nearWalk = routeTrajectory(GridWorld.chargingRoom(), NEAR_ROUTE);
  const farWalk = routeTrajectory(GridWorld.chargingRoom(), FAR_ROUTE);

  function drive(a) {
    if (walk.length && walk[walk.length - 1].terminated) return;
    const [next, reward, terminated, , info] = env.step(a);
    walk.push({ from: state, action: a, to: next, reward, terminated, fell: info.fell_into });
    if (terminated) finished.push({ goal: next, G: walk.reduce((sum, e, t) => sum + gamma ** t * e.reward, 0), gamma });
    scene.move("main", { from: state, to: next, fellInto: info.fell_into, action: a }, { duration: 260 });
    state = next;
    render();
  }

  function backToDock() {
    [state] = env.reset();
    walk = [];
    scene.place("main", state, UP);
    render();
  }

  const reached = (goal) => finished.find((f) => f.goal === goal);
  const mission = new Mission({
    title: "Compare the two chargers",
    goal: "Drive both walks, then let the page show you the best edge out of every node.",
    steps: [
      { text: "Drive to the slow charger (+1): press ↑, then → three times, then ↓.", done: () => !!reached(w.near) },
      { text: "Press Back to the dock. Drive to the fast charger (+10): ↑, → five times, then ↑ twice.", done: () => !!reached(w.far) },
      { text: "Tick “show the optimal policy and its values”.", done: () => showOptimal },
      { text: "Drag γ below 0.46 and watch the dark arrows change direction.", done: () => showOptimal && gamma < 0.4642 },
    ],
    conclusion: () => {
      const slow = reached(w.near);
      const fast = reached(w.far);
      return `Your walk to the slow charger returned ${fmt(slow?.G ?? 0, 3)}; the walk to the fast charger ${fmt(fast?.G ?? 0, 3)} (at γ = ${(fast?.gamma ?? 0.9).toFixed(2)}). The far charger wins at γ = 0.9 despite 3 extra edges. The dark arrows are the optimal policy: one best edge per node, together a tree that leads to the best charger. Below γ ≈ 0.464 the whole tree turns towards the slow charger: the discount alone decides which walk is best.`;
    },
  });

  function render() {
    const { V, Q } = optimalValues(env.model(), gamma);
    const best = showOptimal ? bestEdges(env, Q) : new Set();
    const walked = new Set(walk.map((e) => `${e.from},${e.action}`));
    graph.render({
      edges: edges.map((e) => ({
        ...e,
        role: walked.has(`${e.from},${e.action}`) ? "walk" : best.has(`${e.from},${e.action}`) ? "best" : e.from === e.to ? "faint" : "plain",
      })),
      values: showOptimal ? V : null,
      showValues: showOptimal,
      robot: state,
    });

    const G = walk.reduce((sum, e, t) => sum + gamma ** t * e.reward, 0);
    replace(
      table,
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ["t", "edge", "reward r", "γ^t", "γ^t · r"].map((c) => h("th", {}, c)))),
        h(
          "tbody",
          {},
          walk.length
            ? walk.map((e, t) =>
                h(
                  "tr",
                  {},
                  h("td", {}, t),
                  h("td", {}, `${cellName(env, e.from, w)} ${ARROWS[e.action]} ${cellName(env, e.to, w)}${e.fell !== null ? " (over the ledge)" : ""}`),
                  h("td", {}, fmtShort(e.reward)),
                  h("td", {}, fmt(gamma ** t, 3)),
                  h("td", {}, fmt(gamma ** t * e.reward, 3)),
                ),
              )
            : h("tr", {}, h("td", { colspan: 5, style: { textAlign: "left", fontFamily: "var(--font-body)" } }, "No moves yet. The robot is at the dock.")),
        ),
        h("tfoot", {}, h("tr", {}, h("td", { colspan: 4, style: { textAlign: "left", fontFamily: "var(--font-body)" } }, "return of this walk, G = Σ γ^t r"), h("td", {}, fmt(G, 3)))),
      ),
    );

    const done = walk.length && walk[walk.length - 1].terminated;
    replace(
      status,
      h(
        "p",
        {},
        done
          ? [h("strong", {}, `The walk reached the ${cellName(env, state, w)}. `), `Its return is ${fmt(G, 3)}. The best any walk from the dock can do at γ = ${gamma.toFixed(2)} is ${fmt(V[env.start], 3)}.`]
          : [h("strong", {}, `The robot is at ${cellName(env, state, w)}. `), "Every button is one edge out of this node. Moving costs nothing, the chargers end the walk, and the ledge costs −10 and sends the robot back to the dock."],
      ),
    );
    mission.update();

    const cross = crossing(nearWalk, farWalk);
    const grid = Array.from({ length: 101 }, (_, i) => i / 100);
    chart.update({
      series: [
        { id: "near", label: "walk to the slow charger (+1, 5 edges)", color: "var(--series-1)", points: grid.map((g) => [g, nearWalk.returns(g)[0]]) },
        { id: "far", label: "walk to the fast charger (+10, 8 edges)", color: "var(--series-2)", points: grid.map((g) => [g, farWalk.returns(g)[0]]) },
      ],
      markers: [{ x: gamma, label: `γ = ${gamma.toFixed(2)}` }],
      points: cross ? [{ x: cross, y: nearWalk.returns(cross)[0], label: `equal at γ ≈ ${cross.toFixed(3)}`, color: "var(--ink)" }] : [],
    });

    const gap = Math.abs(nearWalk.returns(gamma)[0] - farWalk.returns(gamma)[0]);
    if (gap < 0.002) solved = true;
    replace(
      challenge,
      solved
        ? h("p", {}, h("span", { class: "verdict good" }, "Solved. "), "At γ ≈ ", fmt(cross, 3), " both walks are worth the same from the dock: ", tex("\\gamma^{4}\\cdot 1 = \\gamma^{7}\\cdot 10"), ", so ", tex("\\gamma = 0.1^{1/3} \\approx 0.464"), ". Below it the best-path tree points everything at the slow charger, above it at the fast one.")
        : h("p", {}, h("span", { class: "status-line" }, h("span", { class: "dot muted" }), `Gap at the dock right now: ${fmt(gap, 4)}. Get it below 0.002.`)),
    );
  }

  const pad = h(
    "div",
    { class: "drive-pad", role: "group", "aria-label": "Drive the robot" },
    button("↑", { kind: "env", onClick: () => drive(UP), title: "up" }),
    button("←", { kind: "env", onClick: () => drive(LEFT), title: "left" }),
    button("↓", { kind: "env", onClick: () => drive(DOWN), title: "down" }),
    button("→", { kind: "env", onClick: () => drive(RIGHT), title: "right" }),
  );
  [...pad.children].forEach((b, i) => b.classList.add(["up", "left", "down", "right"][i]));
  const gammaSlider = slider({ id: "l01-gamma", label: "discount γ", min: 0, max: 0.99, step: 0.001, value: gamma, format: (v) => v.toFixed(3), onInput: (v) => ((gamma = v), render()) });
  const optimalToggle = h("input", { type: "checkbox", id: "l01-opt" });
  optimalToggle.addEventListener("change", () => ((showOptimal = optimalToggle.checked), render()));
  const unbind = shortcuts({ arrowup: () => drive(UP), arrowright: () => drive(RIGHT), arrowdown: () => drive(DOWN), arrowleft: () => drive(LEFT) });

  const elements = [
    ["state", "a node: where the robot is"],
    ["action", "an edge out of that node: one move"],
    ["reward", "the weight on the edge"],
    ["transition", "following one edge"],
    ["episode", "a walk from the dock to a charger"],
    ["return", "the walk's weights, each shrunk by γ once per edge before it"],
    ["policy", "a choice of out-edge at every node"],
    ["optimal policy", "the choice that makes every walk's return as large as possible"],
  ];

  root.append(
    lessonHeader("01", {
      lead: "Before any algorithm, the problem. A robot in a small charging room is a graph: every place it can be is a node, every move is an edge, and every edge has a reward. Reinforcement learning is finding the best walk through that graph without being shown the graph.",
      concepts: ["state", "action", "reward", "episode", "return", "discount γ", "policy", "optimal control"],
    }),
    step("Question", h("p", { class: "question" }, "What does a reinforcement learning problem look like if you draw it?")),
    step(
      "Predict",
      predict({
        question: "From the dock, the slow charger (+1) is 5 edges away and the fast charger (+10) is 8 edges away. Every reward is shrunk by γ once per edge before it. With γ = 0.5, which charger gives the larger return?",
        choices: [
          { label: "The slow charger", detail: "+1, 5 edges" },
          { label: "The fast charger", detail: "+10, 8 edges" },
        ],
        answer: nearWalk.returns(0.5)[0] > farWalk.returns(0.5)[0] ? 0 : 1,
        explain: () =>
          `Slow: 0.5⁴ × 1 = ${fmt(nearWalk.returns(0.5)[0], 4)}. Fast: 0.5⁷ × 10 = ${fmt(farWalk.returns(0.5)[0], 4)}. Three extra halvings divide by 8, and the reward is 10 times larger, so the fast charger still wins, but only just.`,
      }),
    ),
    wideStep(
      "Experiment",
      prose("The 3D view and the graph on the right show the same room: every button press follows one edge of the graph, and the table adds up the walk's return."),
      h(
        "div",
        { class: "bench" },
        mission.el,
        h("div", { class: "toolbar" }, pad, button("Back to the dock", { kind: "ghost", onClick: backToDock }), h("span", { class: "spacer" }), h("label", { class: "toggle", for: "l01-opt" }, optimalToggle, "show the optimal policy and its values"), h("div", { style: { minWidth: "220px" } }, gammaSlider)),
        h("div", { class: "bench-grid part1-grid" }, scene.el, h("div", { class: "figure" }, graph.el, h("div", { class: "key" }, h("span", { class: "key-item" }, h("span", { class: "key-swatch next" }), "the walk so far"), h("span", { class: "key-item" }, h("span", { class: "key-swatch best-edge" }), "best edge out of each node (optimal policy)"), h("span", { class: "key-item" }, h("span", { class: "key-swatch fall-edge" }), "over the ledge, back to the dock")))),
        status,
        table,
      ),
    ),
    step("Inspect", h("div", { class: "elements" }, elements.map(([k, v]) => h("div", { class: "element" }, h("b", {}, k), h("span", {}, v)))), prose("With the optimal policy shown, drag γ across 0.46: every node on the lower route switches its best edge from the slow charger to the fast one at once. The policy is a tree of best edges, and γ decides where it is rooted."), chart.el),
    step(
      "Equation",
      equation("G = r_0 + \\gamma r_1 + \\gamma^2 r_2 + \\dots = \\sum_t \\gamma^t r_t", "The return of a walk: its edge weights, discounted by how many edges come before them."),
      equation(
        "V^*(s) = \\max_a \\big[\\, r(s,a) + \\gamma\\, V^*(s') \\,\\big]",
        h("span", {}, "The Bellman equation: the value of a node is its best edge plus the discounted value of where that edge leads. It is the recursion of shortest-path algorithms (Bellman–Ford is the same Bellman). With the whole graph known, optimal control is just this computation, called planning. Reinforcement learning starts when the graph is not given."),
      ),
    ),
    step("Code", prose("One Bellman backup for every edge, from visualrl/algorithms/tabular/bellman.py. Value iteration repeats it until nothing changes; the optimal policy shown above comes from that."), codeBlock(extractDef(bellmanSource, "q_from_v"), { title: "visualrl/algorithms/tabular/bellman.py" })),
    step("Challenge", prose("Find the discount at which the dock cannot decide between the two chargers: move γ until both walks are worth the same, within 0.002."), challenge),
    lessonFooter("01"),
  );
  scene.place("main", state, UP);
  render();
  return () => {
    unbind();
    scene.dispose();
  };
}

function crossing(a, b) {
  let prev = null;
  for (let i = 1; i <= 1000; i++) {
    const g = i / 1000;
    const d = a.returns(g)[0] - b.returns(g)[0];
    if (prev && Math.sign(d) !== Math.sign(prev.d) && d !== 0) return prev.g + ((g - prev.g) * Math.abs(prev.d)) / (Math.abs(prev.d) + Math.abs(d));
    prev = { g, d };
  }
  return null;
}
