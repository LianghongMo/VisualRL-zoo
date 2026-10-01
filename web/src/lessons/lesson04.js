// Lesson 04: state value and action value. For a fixed policy, V(s) is what being at a node is
// worth; Q(s,a) is what one edge is worth if you take it once and then follow the policy.
// An edge with Q(s,a) > V(s) is a better choice than the policy's own.
import { GridWorld } from "../rl/envs/gridworld.js";
import { exactPolicyValue, optimalValues, qFromV } from "../rl/tabular/dp.js";
import { button, h, replace } from "../ui/dom.js";
import { ARROWS, fmt, fmtShort } from "../ui/format.js";
import { equation } from "../ui/math.js";
import { Mission } from "../ui/mission.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { world } from "./lesson01.js";

const GAMMA = 0.9;
const [UP, RIGHT, DOWN] = [0, 1, 2];
const MOVES = ["up", "right", "down", "left"];

// A policy that sends every node to the slow charger: optimal for a world where only it pays.
function slowChargerPolicy(env) {
  const slowOnly = GridWorld.chargingRoom({ goalRewards: [[1, 6, 0], [4, 4, 1]] }).model();
  const { Q } = optimalValues(slowOnly, GAMMA);
  return Q.map((row) => row.indexOf(Math.max(...row)));
}

export function mount(root) {
  const w = world();
  const { env } = w;
  const model = env.model();
  const edges = trueEdges(env);
  const name = (st) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);
  const aboveSlow = env.toState(3, 4);
  const column = [
    [env.toState(2, 6), UP],
    [env.toState(3, 6), UP],
    [env.toState(3, 5), RIGHT],
  ];
  let pol = slowChargerPolicy(env);
  let selected = null;

  const values = () => exactPolicyValue(model, pol.map((a) => [0, 1, 2, 3].map((b) => (b === a ? 1 : 0))), GAMMA);
  const graph = new StateGraph(env, { goalLabels: w.labels, onNode: (st) => ((selected = model.terminal[st] ? null : st), render()) });
  const panel = h("section", { class: "backup" });
  const note = h("div", { class: "callout" });

  const mission = new Mission({
    title: "Read V and Q, and use Q to improve the policy",
    goal: "The policy starts by sending every node to the slow charger. Use the values to make it better, one node at a time.",
    steps: [
      { text: "Click the dock. Under this policy its value is V = 0.656.", done: () => selected === env.start },
      { text: "Click the node just above the slow charger. Its arrow ↓ has Q = 1.00; the move → has Q = 0.81.", done: () => selected === aboveSlow },
      { text: "Point the three nodes under the fast charger at it: (2,6) ↑, (3,6) ↑, (3,5) →. Click each node and press Use this move.", done: () => column.every(([s, a]) => pol[s] === a) },
      { text: "Click the node above the slow charger again. → now has Q = 7.29, more than its value 1.00. Switch it to →.", done: () => pol[aboveSlow] === RIGHT },
    ],
    conclusion: () =>
      `The dock went from 0.656 to ${fmt(values()[env.start], 3)} without you touching it: its own arrow never changed, but the nodes it leads to became better. V(s) is the return of following the policy from s. Q(s,a) is the return of taking move a once and then following the policy. Whenever an edge has Q(s,a) > V(s), switching to it is an improvement. Lesson 06 does this at every node at once.`,
  });

  function setMove(a) {
    if (selected === null) return;
    pol = [...pol];
    pol[selected] = a;
    render();
  }

  function render() {
    const V = values();
    const Q = qFromV(model, V, GAMMA);
    graph.render({
      edges: edges.map((e) => ({ ...e, role: pol[e.from] === e.action ? "best" : e.from === e.to ? "faint" : "plain" })),
      values: V,
      showValues: true,
      selected,
    });
    if (selected === null) {
      replace(panel, h("div", { class: "backup-head" }, h("strong", {}, "Values at a node")), h("p", { class: "backup-empty" }, "Click a node in the graph."));
    } else {
      const s = selected;
      replace(
        panel,
        h("div", { class: "backup-head" }, h("strong", {}, `At ${name(s)}`), h("span", { class: "grow" }), h("span", { class: "note num" }, `V(s) = ${fmt(V[s], 3)}`)),
        h(
          "table",
          {},
          h("thead", {}, h("tr", {}, ["move a", "lands on s′", "r + γ · V(s′)", "Q(s,a)", ""].map((c, i) => h("th", { class: i < 2 || i === 4 ? "left" : "" }, c)))),
          h(
            "tbody",
            {},
            [0, 1, 2, 3].map((a) => {
              const [s2, r, done] = env.move(s, a);
              const mine = pol[s] === a;
              const better = !mine && Q[s][a] > V[s] + 1e-9;
              return h(
                "tr",
                { class: mine ? "action best" : "action" },
                h("td", { class: "left" }, `${ARROWS[a]} ${MOVES[a]}`),
                h("td", { class: "left" }, r === -10 ? "dock (ledge)" : name(s2)),
                h("td", {}, done ? `${fmtShort(r, 2)} (ends)` : `${fmtShort(r, 2)} + 0.9 × ${fmt(V[s2], 3)}`),
                h("td", {}, fmt(Q[s][a], 3), better ? " > V" : ""),
                h("td", { class: "left" }, mine ? h("span", { class: "note" }, "the policy's move") : button("Use this move", { onClick: () => setMove(a) })),
              );
            }),
          ),
          h("tfoot", {}, h("tr", {}, h("td", { class: "left wrap", colspan: 5 }, `V(s) = Q(s, π(s)) = ${fmt(V[s], 3)}: a node is worth what the policy's move is worth.`))),
        ),
      );
    }
    replace(note, h("p", {}, h("strong", {}, `V(dock) = ${fmt(V[env.start], 3)} under this policy. `), "Numbers on the nodes are V. Dark arrows are the policy: one move per node. Changing one node's arrow changes the value of every node whose walk passes through it."));
    mission.update();
  }

  root.append(
    lessonHeader("04", {
      lead: "A value is a promise about the future. Fix a policy, one move per node, and every node gets a value V: the return of the walk that starts there. Every edge gets a value Q: take that edge once, then follow the policy. Comparing the two tells you where the policy could do better.",
      concepts: ["policy", "state value V", "action value Q", "V(s) = Q(s, π(s))", "improvement"],
    }),
    step("Question", h("p", { class: "question" }, "How much is it worth to be at a node, and to take a particular edge out of it?")),
    step(
      "Predict",
      predict({
        question: "Every node's arrow points the way to the slow charger (+1). At the node right above it, how much is the move → worth, if the robot follows the policy afterwards?",
        choices: [{ label: "0: that edge has no reward" }, { label: "0.81: it comes back and takes the slow charger later" }, { label: "7.29: it heads for the fast charger" }],
        answer: 1,
        explain: () =>
          "Q(s,→) = 0 + 0.9 × V(next node). The next node's own arrow leads to the slow charger in two more moves, so its value is 0.9 × 1, and Q = 0.9 × 0.9 = 0.81. What an edge is worth depends on what the policy does after it.",
      }),
    ),
    wideStep(
      "Experiment",
      prose("The graph shows the policy (dark arrows) and V at every node. Click a node to see the Q value of each of its four moves."),
      h("div", { class: "bench" }, mission.el, h("div", { class: "toolbar" }, button("Reset the policy", { kind: "ghost", onClick: () => ((pol = slowChargerPolicy(env)), render()) })), h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graph.el), h("div", { class: "figure" }, panel)), note),
    ),
    step(
      "Equation",
      equation("V^\\pi(s) = \\sum_t \\gamma^t r_t \\ \\text{ along the walk from } s, \\qquad Q^\\pi(s,a) = r(s,a) + \\gamma\\, V^\\pi(s')", "Both are defined by a fixed policy π. V is about nodes, Q about edges."),
      equation("V^\\pi(s) = Q^\\pi\\big(s, \\pi(s)\\big), \\qquad Q^\\pi(s,a) > V^\\pi(s) \\;\\Rightarrow\\; \\text{switching } s \\text{ to } a \\text{ improves } \\pi", "The second statement is the policy improvement theorem, used everywhere at once in lesson 06."),
    ),
    lessonFooter("04"),
  );
  render();
  return () => {};
}
