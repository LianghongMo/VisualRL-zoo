// Lesson 04: return, discount and value. A value is a multi-step return: the whole walk a policy
// takes from a node, every reward weighted by γ^t. Q(s,a) is the same for a walk that starts with
// edge a. γ sets how far ahead those weights reach, and so what "better" means.
import { GridWorld } from "../rl/envs/gridworld.js";
import { exactPolicyValue, optimalValues, qFromV } from "../rl/tabular/dp.js";
import { DiscountChart } from "../ui/discount-chart.js";
import { button, h, replace } from "../ui/dom.js";
import { ARROWS, fmt, fmtShort } from "../ui/format.js";
import { gammaControl } from "../ui/gamma.js";
import { equation } from "../ui/math.js";
import { Mission } from "../ui/mission.js";
import { returnTable, unrollWalk } from "../ui/return-table.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { world } from "./lesson01.js";

const [UP, RIGHT] = [0, 1];
const MOVES = ["up", "right", "down", "left"];
const SWITCH = 0.1 ** (1 / 3); // γ³ = 0.1: below this, +1 now beats +10 three edges later

// A policy that sends every node to the slow charger: optimal for a world where only it pays.
function slowChargerPolicy() {
  const slowOnly = GridWorld.chargingRoom({ goalRewards: [[1, 6, 0], [4, 4, 1]] }).model();
  const { Q } = optimalValues(slowOnly, 0.9);
  return Q.map((row) => row.indexOf(Math.max(...row)));
}

const oneHot = (pol) => pol.map((a) => [0, 1, 2, 3].map((b) => (b === a ? 1 : 0)));

export function mount(root) {
  const w = world();
  const { env } = w;
  const model = env.model();
  const edges = trueEdges(env);
  const name = (st) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);
  const aboveSlow = env.toState(3, 4);
  const slowPolicy = slowChargerPolicy();

  // ---------- A: a value is the return of a whole walk ----------
  const A = { gamma: 0.9, selected: null };
  const graphA = new StateGraph(env, { goalLabels: w.labels, onNode: (st) => ((A.selected = model.terminal[st] ? null : st), renderA()) });
  const walkA = h("section", { class: "backup" });
  const chartA = new DiscountChart({ steps: 24 });
  const gammaA = gammaControl({ id: "l04-gamma-a", value: A.gamma, onInput: (g) => ((A.gamma = g), renderA()) });

  const missionA = new Mission({
    title: "Unroll a value into its walk",
    goal: "The dark arrows are a fixed policy: from every node, the shortest way to the slow charger (+1). A node's value is the return of the walk these arrows take from it, all the way to the end.",
    steps: [
      { text: "Click the dock. The table follows the arrows edge by edge: rewards 0, 0, 0, 0, then +1 at t = 4. V(dock) is the sum of the γᵗ · rₜ column.", done: () => A.selected === env.start },
      { text: "Drag γ up to 0.99. Far-sighted: the +1 at t = 4 still counts 0.99⁴ = 0.96 of its size.", done: () => A.gamma >= 0.985 },
      { text: "Drag γ down to 0.30. Near-sighted: the same +1 counts 0.3⁴ < 0.01, and every value in the room shrinks towards 0.", done: () => A.gamma <= 0.305 },
      { text: "Click the node right above the slow charger. Its walk is one edge long: its value is 1 at every γ.", done: () => A.selected === aboveSlow },
    ],
    conclusion: () =>
      `A value is not a one-step number: V(s) is the return of the whole walk from s, Σ γᵗ rₜ, every reward weighted by how far ahead it is. γ sets how fast that weight fades. At 0.99 a reward four edges ahead keeps 96% of its size; at 0.3 it keeps less than 1%. The weights add up to 1/(1−γ), so the robot effectively looks about that many edges ahead: 100 at γ = 0.99 (far-sighted), 1.4 at γ = 0.3 (near-sighted). What γ punishes is distance: next to the charger the value is 1 at any γ. Now read the last column from the bottom up: Gₜ = rₜ + γ × Gₜ₊₁. A walk's return is its first reward plus γ times the return of the rest, and the rest is the walk from the next node. So V(s) = r + γ V(s′): a multi-step return computed one edge at a time. That one-edge step is the Bellman backup of lesson 05.`,
  });

  function renderA() {
    const V = exactPolicyValue(model, oneHot(slowPolicy), A.gamma);
    const walk = A.selected === null ? null : unrollWalk(env, A.selected, (s) => slowPolicy[s]);
    const onWalk = new Set(walk ? walk.transitions.map((tr) => `${tr.state},${tr.action}`) : []);
    graphA.render({
      edges: edges.map((e) => ({ ...e, role: onWalk.has(`${e.from},${e.action}`) ? "walk" : slowPolicy[e.from] === e.action ? "best" : e.from === e.to ? "faint" : "plain" })),
      values: V,
      showValues: true,
      selected: A.selected,
    });
    const paid = {};
    walk?.transitions.forEach((tr, t) => tr.reward !== 0 && (paid[t] = tr.reward));
    chartA.update({ gamma: A.gamma, rewards: paid });
    if (!walk) {
      replace(walkA, h("div", { class: "backup-head" }, h("strong", {}, "The walk behind a value")), h("p", { class: "backup-empty" }, "Click a node in the graph. The table follows the policy from it to the end of the walk."));
    } else {
      replace(
        walkA,
        h("div", { class: "backup-head" }, h("strong", {}, `From ${name(A.selected)}, following the arrows`), h("span", { class: "grow" }), h("span", { class: "note num" }, `V = ${fmt(V[A.selected], 3)}`)),
        returnTable(env, walk, A.gamma, { name }),
      );
    }
    missionA.update();
  }

  // ---------- B: values of edges ----------
  const B = { gamma: 0.9, selected: null, pol: [...slowPolicy] };
  const column = [
    [env.toState(2, 6), UP],
    [env.toState(3, 6), UP],
    [env.toState(3, 5), RIGHT],
  ];
  const values = (gamma = B.gamma) => exactPolicyValue(model, oneHot(B.pol), gamma);
  const graphB = new StateGraph(env, { goalLabels: w.labels, onNode: (st) => ((B.selected = model.terminal[st] ? null : st), renderB()) });
  const panelB = h("section", { class: "backup" });
  const noteB = h("div", { class: "callout" });
  const gammaB = gammaControl({ id: "l04-gamma-b", value: B.gamma, onInput: (g) => ((B.gamma = g), renderB()) });

  const missionB = new Mission({
    title: "Read Q, and use it to improve the policy",
    goal: "Same policy: everything heads for the slow charger. Steps 1–4 use γ = 0.9. Improve the policy one node at a time, then see what γ does to the improvement.",
    steps: [
      { text: "Click the dock. Its value under this policy is 0.656 = 0.9⁴ × 1.", done: () => B.selected === env.start },
      { text: "Click the node just above the slow charger. Its arrow ↓ has Q = 1.00; the move → has Q = 0.81, because after → the policy comes back round to the slow charger two edges later.", done: () => B.selected === aboveSlow },
      { text: "Point the three nodes under the fast charger at it: (2,6) ↑, (3,6) ↑, (3,5) →. Click each node and press Use on that move.", done: () => column.every(([s, a]) => B.pol[s] === a) },
      { text: "Click the node above the slow charger again. → now has Q = 10 × 0.9³ = 7.29, more than its value 1.00. Switch it to →.", done: () => B.pol[aboveSlow] === RIGHT },
      { text: "Now drag γ below 0.46 and look at that node again: ↓ (+1 right now) beats → (+10 three edges later), because 10γ³ < 1.", done: () => B.gamma < SWITCH },
    ],
    conclusion: () =>
      `At γ = 0.9 the dock went from 0.656 to ${fmt(values(0.9)[env.start], 3)} without you touching it: its own arrow never changed, but the nodes it leads to got better. Q(s,a) is the return of the walk that starts with move a and then follows the policy, and whenever Q(s,a) > V(s), switching to a is an improvement. Which switches improve depends on γ. At γ = ${B.gamma.toFixed(2)} the node above the slow charger prefers ↓ again: Q(↓) = 1 but Q(→) = ${fmt(10 * B.gamma ** 3, 3)}. γ is part of the task. It decides what a better policy is. Lesson 06 switches every node to its best edge at once.`,
  });

  function setMove(a) {
    if (B.selected === null) return;
    B.pol = [...B.pol];
    B.pol[B.selected] = a;
    renderB();
  }

  function renderB() {
    const V = values();
    const Q = qFromV(model, V, B.gamma);
    const g = fmtShort(B.gamma, 2);
    graphB.render({
      edges: edges.map((e) => ({ ...e, role: B.pol[e.from] === e.action ? "best" : e.from === e.to ? "faint" : "plain" })),
      values: V,
      showValues: true,
      selected: B.selected,
    });
    if (B.selected === null) {
      replace(panelB, h("div", { class: "backup-head" }, h("strong", {}, "Values at a node")), h("p", { class: "backup-empty" }, "Click a node in the graph."));
    } else {
      const s = B.selected;
      replace(
        panelB,
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
              const mine = B.pol[s] === a;
              const better = !mine && Q[s][a] > V[s] + 1e-9;
              return h(
                "tr",
                { class: mine ? "action best" : "action" },
                h("td", { class: "left" }, `${ARROWS[a]} ${MOVES[a]}`),
                h("td", { class: "left" }, r === env.cliffReward ? "dock (ledge)" : name(s2)),
                h("td", {}, done ? `${fmtShort(r, 2)} (ends)` : `${fmtShort(r, 2)} + ${g} × ${fmt(V[s2], 3)}`),
                h("td", {}, fmt(Q[s][a], 3), better ? " > V" : ""),
                h("td", { class: "left" }, mine ? h("span", { class: "note" }, "policy") : button("Use", { title: "Use this move", onClick: () => setMove(a) })),
              );
            }),
          ),
          h("tfoot", {}, h("tr", {}, h("td", { class: "left wrap", colspan: 5 }, `V(s) = Q(s, π(s)) = ${fmt(V[s], 3)}: a node is worth what the policy's move is worth.`))),
        ),
      );
    }
    replace(noteB, h("p", {}, h("strong", {}, `V(dock) = ${fmt(V[env.start], 3)} under this policy at γ = ${B.gamma.toFixed(2)}. `), "Numbers on the nodes are V, dark arrows the policy. Each Q in the table is one edge plus γ times the value where it lands, and that value is itself a whole walk. Changing one node's arrow changes the value of every node whose walk passes through it."));
    missionB.update();
  }

  const why = h(
    "div",
    { class: "why-grid" },
    h("div", { class: "why-card" }, h("b", {}, "The return is the task."), "Nobody tells the robot which charger to go to. It is told to make its return large: G = r₀ + γr₁ + γ²r₂ + …, the rewards of its whole future walk. Which charger is best follows from that sum."),
    h("div", { class: "why-card" }, h("b", {}, "γ says how far ahead to look."), "A reward t edges ahead counts γᵗ of its size, and the weights add up to 1/(1−γ). So γ = 0.5 looks about 2 edges ahead (near-sighted), γ = 0.9 about 10, γ = 0.99 about 100 (far-sighted). Change γ and the best behavior changes."),
    h("div", { class: "why-card" }, h("b", {}, "A value stores the future at a node."), "Comparing returns directly means following every walk to its end. V(s) keeps that result at s, so a decision needs one edge: Q(s,a) = r + γV(s′). Every algorithm in this course is a way to get V or Q right."),
  );

  root.append(
    lessonHeader("04", {
      lead: "Lesson 01 added up one walk's rewards into its return. This lesson shows what that number is for. Fix a policy, one move per node, and the walk from every node is fixed. Its return is the node's value V. The value packs the whole future into one number, so the robot can judge a move by looking one edge ahead. The discount γ decides how far that future reaches.",
      concepts: ["return G", "discount γ", "horizon 1/(1−γ)", "state value V", "action value Q", "Gₜ = rₜ + γGₜ₊₁", "improvement"],
    }),
    step("Question", h("p", { class: "question" }, "Why does the robot need a number like V(s), and what does γ do to it?")),
    step("Why it matters", why),
    step(
      "Predict",
      predict({
        question: "Under these arrows the dock's walk reaches the slow charger (+1) on its fifth edge. What is V(dock) with γ = 0.5?",
        choices: [{ label: "1: the walk does reach the charger" }, { label: "0.5" }, { label: "0.0625" }],
        answer: 2,
        explain: () =>
          "The +1 comes at t = 4, so it counts 0.5⁴ = 1/16 of its size. With γ = 0.5 the robot looks about 1/(1−γ) = 2 edges ahead, and a reward five edges away is almost invisible to it. The same walk is worth 0.656 at γ = 0.9 and 0.961 at γ = 0.99.",
      }),
    ),
    wideStep(
      "A value is a multi-step return",
      prose("Click a node: the table follows the policy from it, edge by edge, until the walk ends, and adds up the return. Drag γ and watch the weights, the table and every value on the graph change together."),
      h(
        "div",
        { class: "bench" },
        missionA.el,
        h("div", { class: "toolbar" }, gammaA),
        h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graphA.el), h("div", { class: "stack" }, walkA, chartA.el)),
      ),
    ),
    wideStep(
      "Values of edges",
      prose("Q(s,a) is the same idea for one chosen edge: take move a once, then follow the policy, and add up the return of that walk. It equals r + γ V(s′), one edge plus the value where it lands. Click a node to see the Q of each of its four moves."),
      h(
        "div",
        { class: "bench" },
        missionB.el,
        h("div", { class: "toolbar" }, button("Reset the policy", { kind: "ghost", onClick: () => ((B.pol = [...slowPolicy]), renderB()) }), h("span", { class: "spacer" }), gammaB),
        h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graphB.el), h("div", { class: "figure" }, panelB)),
        noteB,
      ),
    ),
    step(
      "Equation",
      equation("G_t = r_t + \\gamma r_{t+1} + \\gamma^2 r_{t+2} + \\dots = r_t + \\gamma\\, G_{t+1}", "The return two ways: every reward weighted by how far ahead it is, or one reward plus γ times the return of the rest."),
      equation("\\sum_{t=0}^{\\infty} \\gamma^t = \\frac{1}{1-\\gamma}", "The weights add up to the effective horizon: 2 edges at γ = 0.5, 10 at γ = 0.9, 100 at γ = 0.99."),
      equation("V^\\pi(s) = G_0 \\text{ of the walk } \\pi \\text{ takes from } s, \\qquad Q^\\pi(s,a) = r(s,a) + \\gamma\\, V^\\pi(s'), \\qquad V^\\pi(s) = Q^\\pi\\big(s, \\pi(s)\\big)", "On a slippery floor the walk is random and V is its expected return. Q is about edges, V about nodes."),
      equation("Q^\\pi(s,a) > V^\\pi(s) \\;\\Rightarrow\\; \\text{switching } s \\text{ to } a \\text{ improves } \\pi", "The policy improvement theorem, used at every node at once in lesson 06."),
    ),
    lessonFooter("04"),
  );
  renderA();
  renderB();
  return () => {};
}
