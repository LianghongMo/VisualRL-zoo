// Lesson 02: the learning loop. The robot does not know the graph. Acting adds the edges it
// drives to its experience graph (control loop); learning backs values up along those edges
// (learning loop). Learning can never know more than the graph it has.
import graphSource from "../../../visualrl/algorithms/tabular/experience_graph.py";
import { makeTransition } from "../rl/core.js";
import { optimalValues } from "../rl/tabular/dp.js";
import { ExperienceGraph } from "../rl/tabular/experience-graph.js";
import { Rng } from "../rl/rng.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, shortcuts } from "../ui/dom.js";
import { ARROWS, fmt } from "../ui/format.js";
import { Ledger } from "../ui/ledger.js";
import { LoopDiagram } from "../ui/loop-diagram.js";
import { Mission } from "../ui/mission.js";
import { equation } from "../ui/math.js";
import { WarehouseScene } from "../ui/scene3d.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";
import { world } from "./lesson01.js";

const [UP, RIGHT, DOWN, LEFT] = [0, 1, 2, 3];
const GAMMA = 0.9;

export function mount(root) {
  const w = world();
  const { env } = w;
  const allEdges = trueEdges(env);
  const optimum = optimalValues(env.model(), GAMMA).V[env.start];
  const rng = new Rng(4);
  let graph;
  let state;
  let envSteps;
  let episodes;
  let showTrue = false;
  let lastSweep = null;
  let solvedWith = null;
  const name = (st) => (st === env.start ? "dock" : st === w.far ? "fast charger" : st === w.near ? "slow charger" : `(${env.toCell(st).join(",")})`);

  const scene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], goalLabels: w.signs, caption: "The robot only knows the edges it has driven" });
  const view = new StateGraph(env, { goalLabels: w.labels });
  const loop = new LoopDiagram();
  const counters = h("div", { class: "counters" });
  const note = h("div", { class: "callout" });
  const challenge = h("div", { class: "callout" });
  const ledger = new Ledger({
    titleFor: (t) => `Learning step ${t.sweep} · one backup of every visited node`,
    groupsFor: (t) => [
      {
        label: t.changed.length ? "Values that changed" : "Nothing changed",
        tone: "learn",
        rows: t.changed.length
          ? t.changed.map((c) => ({
              label: `${name(c.state)}, via ${ARROWS[c.action]} to ${c.next_state === null ? "?" : name(c.next_state)}`,
              formula: "V(s)",
              value: `${fmt(c.before)} → ${fmt(c.after)}`,
            }))
          : [{ label: "Value has reached every node that a known edge connects to a charger.", formula: "", value: "" }],
      },
    ],
    empty: "No learning step yet. Drive a few edges, then press Learn once.",
    emptyTitle: "Learning step",
  });

  const mission = new Mission({
    title: "Watch the two loops",
    goal: "Collect some experience, then learn from it, and see what each loop changes.",
    steps: [
      { text: "Drive to the slow charger: ↑, then → three times, then ↓. Look at the graph: 5 new edges, but every value is still 0.", done: () => graph.terminal.has(w.near) },
      { text: "Press Learn once five times. Each press moves value one node closer to the dock.", baseline: () => graph.sweeps, done: (b) => graph.sweeps - b >= 5 && graph.V[env.start] > 0 },
      { text: "Now drive to the fast charger by the short route (↑, → five times, ↑ ↑), then press Learn until nothing changes.", done: () => graph.V[env.start] >= optimum - 1e-9 },
    ],
    conclusion: () =>
      `Driving (the control loop) added edges but changed no value. Learning (the learning loop) changed values but added no edge: each press moved value one edge back, so the dock, 5 edges from the slow charger, needed 5 presses to reach ${fmt(0.9 ** 4, 3)}. It only reached the best possible value, ${fmt(optimum, 3)}, after you drove the edges to the fast charger. Learning can only use the edges that are in the graph.`,
  });

  function reset() {
    graph = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: GAMMA });
    [state] = env.reset();
    envSteps = 0;
    episodes = 0;
    lastSweep = null;
    solvedWith = null;
    ledger.reset();
    scene.place("main", state, UP);
    mission.reset();
    render();
  }

  function drive(a) {
    if (env.goals.includes(state)) return; // the episode just ended; the robot is being carried back to the dock
    const [next, reward, terminated, , info] = env.step(a);
    graph.add(makeTransition({ state, action: a, reward, next_state: next, terminated }));
    scene.move("main", { from: state, to: next, fellInto: info.fell_into, action: a }, { duration: 260 });
    envSteps += 1;
    loop.pulse("control");
    lastSweep = null;
    if (terminated) {
      episodes += 1;
      setTimeout(() => {
        [state] = env.reset();
        scene.place("main", state, UP);
        render();
      }, 420);
    }
    state = next;
    render();
  }

  function robotDrives() {
    drive(rng.random() < 0.2 ? rng.integers(4) : graph.act(state));
  }

  function learn(times = 1) {
    let t;
    for (let i = 0; i < times; i++) {
      t = graph.sweep();
      ledger.push(t);
      if (!t.changed.length) break;
    }
    lastSweep = t;
    loop.pulse("learning");
    if (solvedWith === null && graph.V[env.start] >= optimum - 1e-9) solvedWith = envSteps;
    render();
  }

  function render() {
    const known = new Set([...graph.visited, ...graph.terminal]);
    const hl = new Set((lastSweep?.changed ?? []).map((c) => `${c.state},${c.action}`));
    const edges = [];
    for (const e of allEdges) {
      const k = `${e.from},${e.action}`;
      const seen = graph.edges.has(k);
      if (!seen && !showTrue) continue;
      const best = seen && graph.V[e.from] !== 0 && graph.act(e.from) === e.action;
      edges.push({ ...e, role: hl.has(k) ? "hl" : !seen ? "faint" : best ? "best" : e.from === e.to ? "faint" : "plain" });
    }
    const stubs = {};
    for (const s of graph.visited) if (!graph.terminal.has(s)) stubs[s] = graph.untried(s);
    view.render({ edges, values: graph.V, showValues: true, stubs, known, robot: state });

    replace(
      counters,
      h("span", { class: "counter" }, h("span", { class: "dot env" }), "environment steps", h("b", {}, envSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot learn" }), "learning steps", h("b", {}, graph.sweeps)),
      h("span", { class: "counter" }, h("span", { class: "dot muted" }), "episodes", h("b", {}, episodes)),
      h("span", { class: "counter" }, "known edges", h("b", {}, `${graph.edges.size} of ${allEdges.length}`)),
      h("span", { class: "counter" }, "value of the dock", h("b", {}, fmt(graph.V[env.start], 3))),
    );

    const reached = [...graph.terminal].map(name);
    replace(
      note,
      h(
        "p",
        {},
        graph.edges.size === 0
          ? [h("strong", {}, "The robot knows nothing yet. "), "Every node except the dock is drawn dashed: it has never been there. Drive, and watch the graph grow."]
          : !reached.length
            ? [h("strong", {}, "No charger in the graph yet. "), "However much the robot learns, every value stays 0: there is no reward on any edge it knows."]
            : [
                h("strong", {}, `Known chargers: ${reached.join(" and ")}. `),
                `The dock is worth ${fmt(graph.V[env.start], 3)} to the robot. The best possible is ${fmt(optimum, 3)}. ${graph.V[env.start] < optimum - 1e-9 ? "Learning more will not close that gap if the missing edges are not in the graph; only driving them will." : "Its graph already contains a best route."}`,
              ],
      ),
    );

    mission.update();
    replace(
      challenge,
      solvedWith !== null
        ? h("p", {}, h("span", { class: "verdict good" }, "Done. "), `The dock's value reached the optimum ${fmt(optimum, 3)} after ${solvedWith} environment steps. The fewest possible is 8: drive the best route once, then learn until nothing changes. Learning is free; information is not.`)
        : h("p", {}, h("span", { class: "status-line" }, h("span", { class: "dot muted" }), `Not yet: the dock is worth ${fmt(graph.V[env.start], 3)} of ${fmt(optimum, 3)} after ${envSteps} environment steps.`)),
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
  const trueToggle = h("input", { type: "checkbox", id: "l02-true" });
  trueToggle.addEventListener("change", () => ((showTrue = trueToggle.checked), render()));
  const unbind = shortcuts({ arrowup: () => drive(UP), arrowright: () => drive(RIGHT), arrowdown: () => drive(DOWN), arrowleft: () => drive(LEFT), l: () => learn(1) });

  root.append(
    lessonHeader("02", {
      lead: "The robot is not given the graph. It only knows the edges it has driven: its experience graph. Reinforcement learning is two loops around that graph. Acting adds edges to it; learning computes values on it.",
      concepts: ["control loop", "learning loop", "experience", "Bellman backup", "model unknown"],
    }),
    step("Question", h("p", { class: "question" }, "The robot does not know the graph. How can it still learn where to go?")),
    step(
      "Predict",
      predict({
        question: "You drive once from the dock to the slow charger: 5 edges. Then the robot learns, one Bellman backup of every node per learning step. After how many learning steps does the dock's value first change?",
        choices: [{ label: "After 1" }, { label: "After 5" }, { label: "Never: the robot did not learn while driving" }],
        answer: 1,
        explain: () =>
          "Each backup moves value one edge back along the known edges: first to the node next to the charger, then the one before it, and so on. The dock is 5 edges from the charger, so it changes on the fifth learning step, to 0.9⁴ × 1 = 0.656. Try it below.",
      }),
    ),
    wideStep(
      "Experiment",
      prose("The graph on the right is the robot's experience graph: only the nodes and edges it has seen. Dashed nodes are places it has never been; the numbers are its values."),
      loop.el,
      h(
        "div",
        { class: "bench" },
        mission.el,
        h(
          "div",
          { class: "toolbar" },
          pad,
          button("Robot drives one step", { onClick: robotDrives }),
          h("span", { class: "spacer" }),
          button("Learn once", { kind: "learn", kbd: "L", onClick: () => learn(1) }),
          button("Learn until nothing changes", { onClick: () => learn(100) }),
          button("Start over", { kind: "ghost", onClick: reset }),
        ),
        counters,
        h("div", { class: "bench-grid part1-grid" }, scene.el, h("div", { class: "figure" }, view.el, h("div", { class: "key" },
          h("span", { class: "key-item" }, h("span", { class: "key-swatch stub" }), "a move never tried"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch best-edge" }), "edge the last learning step used, and the robot's best known edge"),
          h("label", { class: "toggle", for: "l02-true" }, trueToggle, "show the rest of the true graph (the robot cannot see it)"),
        ))),
        note,
        ledger.el,
      ),
    ),
    step(
      "Inspect",
      prose(
        "The two loops run at different speeds and cost different things. An environment step costs time, battery, and sometimes a fall off the ledge; a learning step only costs computation. But learning can only redistribute what the graph already contains: if no known edge leads to the fast charger, no amount of learning will find it.",
        "Notice also who chose the actions. When you drive, the data comes from your policy, while the values the robot learns are for its own greedy policy. That mismatch is what off-policy means, and it returns in lessons 03 and 08.",
      ),
    ),
    step(
      "Equation",
      equation(
        "V(s) \\leftarrow \\max_{a \\,:\\, (s,a)\\ \\text{seen}} \\big[\\, r(s,a) + \\gamma\\, V(s') \\,\\big]",
        "The Bellman equation of lesson 01, restricted to the edges the robot has seen. One learning step applies it to every visited node; after k steps a node's value is the best return of any known walk of at most k edges.",
      ),
    ),
    step("Code", codeBlock(extractDef(graphSource, "sweep"), { title: "visualrl/algorithms/tabular/experience_graph.py" })),
    step("Challenge", prose("Get the dock's value to the true optimum with as few environment steps as possible. Learning steps are free."), challenge),
    lessonFooter("02"),
  );
  reset();
  return () => {
    unbind();
    scene.dispose();
  };
}
