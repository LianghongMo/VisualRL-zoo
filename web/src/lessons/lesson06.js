// Lesson 06: policy iteration. Evaluate the policy (expectation backups until V stops changing),
// then improve it (switch every node to its best edge by Q). Two buttons, two different things.
import piSource from "../../../visualrl/algorithms/tabular/policy_iteration.py";
import { bellmanBackup, PolicyIteration } from "../rl/tabular/dp.js";
import { BackupPanel } from "../ui/backup-panel.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace } from "../ui/dom.js";
import { fmt } from "../ui/format.js";
import { equation } from "../ui/math.js";
import { Mission } from "../ui/mission.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";
import { world } from "./lesson01.js";

const GAMMA = 0.9;

export function mount(root) {
  const w = world();
  const { env } = w;
  const model = env.model();
  const edges = trueEdges(env);
  const name = (st) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);
  let pi;
  let log;
  let selected = null;
  let evaluatedSinceImprove;
  let lastImprove;
  let finished; // { changes, dock } at the first Improve that changed nothing

  const graph = new StateGraph(env, { goalLabels: w.labels, onNode: (st) => ((selected = model.terminal[st] ? null : st), render()) });
  const panel = new BackupPanel({ name, title: "Evaluation backup" });
  const logEl = h("div", { class: "sweep-log" });
  const note = h("div", { class: "callout" });

  function reset() {
    pi = new PolicyIteration(model, GAMMA); // starts from the random policy: every move equally likely
    log = [];
    evaluatedSinceImprove = false;
    lastImprove = null;
    finished = null;
    render();
  }

  function evaluate(full) {
    const t = full ? pi.evaluate({ theta: 1e-10 }) : pi.evaluateStep();
    const settled = t.max_change < 1e-10;
    if (settled) evaluatedSinceImprove = true;
    log.push({ kind: full || settled ? "evaluate (settled)" : "evaluate: 1 sweep", dock: pi.V[env.start], detail: `${pi.sweeps} sweeps so far` });
    render();
  }

  function improve() {
    lastImprove = pi.improveStep();
    evaluatedSinceImprove = false;
    const n = lastImprove.changed_states.length;
    log.push({ kind: "improve", dock: pi.V[env.start], detail: n === 0 ? "no node changed: stable" : `${n} node${n === 1 ? "" : "s"} changed move` });
    if (n === 0 && !finished) finished = { changes: log.filter((x) => x.kind === "improve").length - 1, dock: pi.V[env.start] };
    render();
  }

  const mission = new Mission({
    title: "Evaluate, improve, repeat",
    goal: "Start from the worst kind of plan, a coin flip at every node, and turn it into the optimal policy with two operations.",
    steps: [
      { text: "Press Evaluate until nothing changes. These are the values of the random policy.", done: () => pi.improvements === 0 && evaluatedSinceImprove },
      { text: "Press Improve the policy. Every node switches to its best edge according to those values.", done: () => pi.improvements >= 1 },
      { text: "Press Evaluate until nothing changes again, then Improve again. Repeat until Improve changes no node.", done: () => finished !== null },
    ],
    conclusion: () =>
      `Improve changed the policy ${finished?.changes} times, and the next one changed nothing. The random policy's values were mostly negative (the dock was worth ${fmt(log.find((x) => x.kind !== "improve")?.dock ?? 0, 3)}): a coin-flip walk next to the ledge falls off it again and again. Each improvement made the policy greedy on the latest values, and each evaluation computed what that policy is really worth. When an improvement changed nothing, the policy was greedy on its own values: that is the Bellman optimality equation, so this is the optimal policy, and V(dock) = ${fmt(finished?.dock ?? pi.V[env.start], 3)} is the same V* as value iteration found in lesson 05.`,
  });

  function render() {
    const P = pi.policy;
    graph.render({
      edges: edges.map((e) => {
        const p = P[e.from][e.action];
        return { ...e, role: p > 0.99 ? "best" : p > 0 ? (e.from === e.to ? "faint" : "plain") : "faint" };
      }),
      values: pi.V,
      showValues: true,
      selected,
    });
    panel.show(selected === null ? null : bellmanBackup(model, pi.V, P, selected, GAMMA), { gamma: GAMMA });
    replace(
      logEl,
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ["step", "what", "V(dock)", ""].map((c) => h("th", {}, c)))),
        h(
          "tbody",
          {},
          log.length
            ? log.map((x, i) => h("tr", { class: x.kind === "improve" ? "highlight" : "" }, h("td", {}, i + 1), h("td", { style: { textAlign: "left", fontFamily: "var(--font-body)" } }, x.kind), h("td", {}, fmt(x.dock, 3)), h("td", { class: "wrap", style: { textAlign: "left", fontFamily: "var(--font-body)" } }, x.detail)))
            : h("tr", {}, h("td", { colspan: 4, style: { textAlign: "left", fontFamily: "var(--font-body)" } }, "Nothing yet. The policy picks each of the four moves with probability 1/4.")),
        ),
      ),
    );
    replace(
      note,
      h(
        "p",
        {},
        h("strong", {}, `${pi.improvements === 0 ? "The random policy" : `Policy after ${pi.improvements} improvement${pi.improvements === 1 ? "" : "s"}`}. `),
        "Thin arrows are moves the policy takes with some probability, dark ones moves it always takes. Evaluate changes the numbers and never the arrows; Improve changes the arrows and never the numbers. Click a node to see its evaluation backup: the average of its moves, weighted by the policy.",
      ),
    );
    mission.update();
  }

  root.append(
    lessonHeader("06", {
      lead: "Lesson 05 found the optimal policy by repeating one backup that both evaluates and improves. Policy iteration splits it into two separate operations. Evaluate: compute what the current policy is worth, with expectation backups. Improve: switch every node to its best edge by those values. Alternate until improving changes nothing.",
      concepts: ["policy evaluation", "policy improvement", "policy iteration", "expectation backup"],
    }),
    step("Question", h("p", { class: "question" }, "If you start from a bad plan, how do you turn it into the best one?")),
    step(
      "Predict",
      predict({
        question: "You start from the random policy and alternate a full evaluation with an improvement. How many times will Improve change the policy before it stops changing?",
        choices: [{ label: "Once: one improvement is enough" }, { label: "A handful of times" }, { label: "Hundreds of times" }],
        answer: 1,
        explain: () =>
          "Here, 5 improvements change something and the 6th changes nothing. The first one fixes 17 nodes at once, because a full evaluation gives every node exact values; the later ones fix the few nodes whose successors only became good in the previous round. Each improvement is paid for with a full evaluation, dozens of sweeps.",
      }),
    ),
    wideStep(
      "Experiment",
      h(
        "div",
        { class: "bench" },
        mission.el,
        h(
          "div",
          { class: "toolbar" },
          button("Evaluate: one sweep", { onClick: () => evaluate(false) }),
          button("Evaluate until nothing changes", { kind: "learn", onClick: () => evaluate(true) }),
          button("Improve the policy", { kind: "env", onClick: improve }),
          h("span", { class: "spacer" }),
          button("Start over from the random policy", { kind: "ghost", onClick: reset }),
        ),
        h("div", { class: "bench-grid part1-grid even" }, h("div", { class: "figure" }, graph.el), h("div", { class: "figure" }, logEl, panel.el)),
        note,
      ),
    ),
    step(
      "Equation",
      equation("\\text{Evaluate:}\\quad V(s) \\leftarrow \\sum_a \\pi(a\\mid s) \\sum_{s'} P(s'\\mid s,a)\\,\\big[\\, r + \\gamma V(s') \\,\\big]", "The expectation backup: an average over the policy's moves, not a max. Repeated until V stops changing, it gives V^π."),
      equation("\\text{Improve:}\\quad \\pi(s) \\leftarrow \\arg\\max_a \\sum_{s'} P(s'\\mid s,a)\\,\\big[\\, r + \\gamma V^\\pi(s') \\,\\big]", "Greedy on Q^π. By lesson 04, this never makes any node worse; when it changes nothing, the policy is optimal."),
    ),
    step("Code", codeBlock(extractDef(piSource, "improve_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "visualrl/algorithms/tabular/policy_iteration.py" })),
    lessonFooter("06"),
  );
  reset();
  return () => {};
}
