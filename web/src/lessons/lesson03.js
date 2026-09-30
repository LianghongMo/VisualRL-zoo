// Lesson 03: online and offline. Offline, the experience graph is a fixed dataset: planning on it
// can stitch episodes together, but moves outside it cannot be checked. Online, the robot grows the
// graph itself: the question becomes where to explore. One choice decides both: what to assume
// about moves that have never been tried.
import graphSource from "../../../visualrl/algorithms/tabular/experience_graph.py";
import { makeTransition } from "../rl/core.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { optimalValues } from "../rl/tabular/dp.js";
import { ExperienceGraph, OPTIMISTIC, PESSIMISTIC } from "../rl/tabular/experience-graph.js";
import { Rng } from "../rl/rng.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, segmented, slider } from "../ui/dom.js";
import { fmt } from "../ui/format.js";
import { LineChart } from "../ui/line-chart.js";
import { LoopDiagram } from "../ui/loop-diagram.js";
import { Mission } from "../ui/mission.js";
import { equation, tex } from "../ui/math.js";
import { WarehouseScene } from "../ui/scene3d.js";
import { StateGraph, trueEdges } from "../ui/state-graph.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";
import { world } from "./lesson01.js";

const [UP, RIGHT, DOWN] = [0, 1, 2];
const GAMMA = 0.9;
const EPISODE_A = [UP, RIGHT, RIGHT, RIGHT, DOWN];
const EPISODE_B = [UP, UP, UP, RIGHT, RIGHT, RIGHT, DOWN, DOWN, RIGHT, RIGHT, UP, UP];
const ONLINE_EPISODE_STEPS = 40;
const EPSILON = 0.2;

function drive(env, moves) {
  let [s] = env.reset();
  const out = [];
  for (const a of moves) {
    const [s2, r, done, , info] = env.step(a);
    out.push({ ...makeTransition({ state: s, action: a, reward: r, next_state: s2, terminated: done }), fell_into: info.fell_into });
    s = s2;
    if (done) break;
  }
  return out;
}

const discounted = (steps) => steps.reduce((sum, t, i) => sum + GAMMA ** i * t.reward, 0);

// Run a graph's greedy policy in a fresh copy of the world, learning nothing.
function deploy(_, graph, maxSteps = 12) {
  const env = GridWorld.chargingRoom();
  let [s] = env.reset();
  const steps = [];
  for (let i = 0; i < maxSteps; i++) {
    const a = graph.act(s);
    const [s2, r, done, , info] = env.step(a);
    steps.push({ ...makeTransition({ state: s, action: a, reward: r, next_state: s2, terminated: done }), fell_into: info.fell_into });
    s = s2;
    if (done) break;
  }
  return steps;
}

function offlineGraph(env, data, unseen, value) {
  const g = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: GAMMA, unseen, optimisticValue: value });
  for (const t of data) g.add(t);
  g.plan();
  return g;
}

// One online learner from scratch, episode by episode; returns the greedy return from the dock after each episode.
function onlineCurve(env, strategy, seed, episodes = 30) {
  const rng = new Rng(seed);
  const g = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: GAMMA, unseen: strategy === "optimistic" ? OPTIMISTIC : PESSIMISTIC });
  const curve = [];
  for (let e = 0; e < episodes; e++) {
    let [s] = env.reset();
    for (let i = 0; i < ONLINE_EPISODE_STEPS; i++) {
      const a = strategy === "random" && rng.random() < EPSILON ? rng.integers(4) : g.act(s);
      const [s2, r, done] = env.step(a);
      g.add(makeTransition({ state: s, action: a, reward: r, next_state: s2, terminated: done }));
      g.plan();
      s = s2;
      if (done) break;
    }
    curve.push(discounted(deploy(env, g, 30)));
  }
  return curve;
}

export function mount(root) {
  const w = world();
  const { env } = w;
  const allEdges = trueEdges(env);
  const optimum = optimalValues(env.model(), GAMMA).V[env.start];
  const dataA = drive(GridWorld.chargingRoom(), EPISODE_A);
  const dataB = drive(GridWorld.chargingRoom(), EPISODE_B);
  const onEnv = GridWorld.chargingRoom(); // the online robot's own world
  const data = [...dataA, ...dataB];
  const inA = new Set(dataA.map((t) => `${t.state},${t.action}`));
  const inB = new Set(dataB.map((t) => `${t.state},${t.action}`));
  const returns = { A: discounted(dataA), B: discounted(dataB) };
  const results = {};

  // ---------- offline ----------
  let offMode = "pessimistic";
  let assumed = 10;
  let offGraph = null;
  let deployed = [];
  let challengeDone = null;
  const driven = { pessimistic: null, optimistic: null }; // return of the last plan driven with each assumption
  const dataOnly = offlineGraph(env, data, PESSIMISTIC, 0); // for drawing the dataset before any plan
  const offScene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], goalLabels: w.signs, caption: "Deploying the policy planned from the dataset. Nothing is learned while it drives." });
  const offView = new StateGraph(env, { goalLabels: w.labels });
  const offLoop = new LoopDiagram({ offline: true });
  const offNote = h("div", { class: "callout" });
  const offChallenge = h("div", { class: "callout" });
  const assumedSlider = slider({ id: "l03-assumed", label: "assumed value of an untried move", min: 0, max: 10, step: 0.01, value: assumed, format: (v) => v.toFixed(2), onInput: (v) => ((assumed = v), offGraph && planOffline()) });
  const offMission = new Mission({
    title: "Learn from a fixed dataset",
    goal: "Plan a route using only episodes A and B, drive it, and compare two assumptions about the moves nobody tried.",
    steps: [
      { text: "With “Leave untried moves out” selected, press Plan on this data and drive.", done: () => driven.pessimistic !== null },
      { text: "Select “Assume a value for untried moves” and press Plan on this data and drive again.", done: () => driven.optimistic !== null },
    ],
    conclusion: () =>
      `Trusting only the data, the plan returned ${fmt(driven.pessimistic ?? 0, 2)}: better than episode A (${fmt(returns.A, 2)}) and episode B (${fmt(returns.B, 2)}), because it joined A's start to B's end where they cross. That is stitching. Assuming untried moves are worth ${fmt(assumed, 2)}, the plan returned ${fmt(driven.optimistic ?? 0, 2)}: it chose a move nobody had tried, and that move goes over the ledge. That is the coverage problem: offline, nothing can check a move the data does not contain.`,
  });

  function planOffline() {
    offGraph = offlineGraph(env, data, offMode === "pessimistic" ? PESSIMISTIC : OPTIMISTIC, assumed);
    deployed = deploy(env, offGraph);
    const got = discounted(deployed);
    driven[offMode] = got;
    results[offMode === "pessimistic" ? "offPess" : "offOpt"] = got;
    if (offMode === "optimistic" && Math.abs(assumed - 10) < 1e-9) results.offOpt10 = got;
    offScene.playSteps("main", deployed.map((t) => ({ from: t.state, to: t.next_state, action: t.action, fellInto: t.fell_into })), { stepMs: 320 });
    const stitched = offMode === "optimistic" && Math.abs(got - optimum) < 1e-9;
    if (stitched && assumed >= 4.7 && challengeDone === null) challengeDone = assumed;
    renderOffline();
    renderSummary();
  }

  function renderOffline() {
    if (!offGraph) {
      const edges = allEdges.filter((e) => inA.has(`${e.from},${e.action}`) || inB.has(`${e.from},${e.action}`)).map((e) => ({ ...e, role: inB.has(`${e.from},${e.action}`) ? "b" : "a" }));
      const stubs = {};
      for (const s of dataOnly.visited) if (!dataOnly.terminal.has(s)) stubs[s] = dataOnly.untried(s);
      offView.render({ edges, stubs, known: new Set([...dataOnly.visited, ...dataOnly.terminal]), robot: env.start });
      replace(offNote, h("p", {}, h("strong", {}, "These two episodes are all the robot has. "), `Episode A (light blue) reached the slow charger, return ${fmt(returns.A, 2)}. Episode B (dark blue) reached the fast charger by a detour, return ${fmt(returns.B, 2)}. The dashed stubs are moves nobody tried. Press Plan on this data and drive.`));
      offMission.update();
      return;
    }
    const walked = new Set(deployed.map((t) => `${t.state},${t.action}`));
    const edges = [];
    for (const e of allEdges) {
      const k = `${e.from},${e.action}`;
      if (walked.has(k) && !inA.has(k) && !inB.has(k)) edges.push({ ...e, role: "hl" });
      else if (inB.has(k)) edges.push({ ...e, role: "b" });
      else if (inA.has(k)) edges.push({ ...e, role: "a" });
    }
    const known = new Set([...offGraph.visited, ...offGraph.terminal]);
    const stubs = {};
    for (const s of offGraph.visited) if (!offGraph.terminal.has(s)) stubs[s] = offGraph.untried(s);
    offView.render({ edges, values: offGraph.V, showValues: true, stubs, known, robot: null, route: [env.start, ...deployed.map((t) => t.next_state)] });

    replace(
      offChallenge,
      challengeDone !== null
        ? h("p", {}, h("span", { class: "verdict good" }, "Found it. "), `At ${fmt(challengeDone, 2)} the plan still drives the stitched route; just above ${fmt(optimum, 3)} it goes over the ledge. ${fmt(optimum, 3)} is the value of the best route the data proves. An untried move assumed to be worth more than that outbids every route the data can back up.`)
        : h("p", {}, h("span", { class: "status-line" }, h("span", { class: "dot muted" }), offMode === "optimistic" ? `Assumed ${fmt(assumed, 2)}: the plan ${Math.abs(discounted(deployed) - optimum) < 1e-9 ? "drives the stitched route. Go higher, as close to the edge as you can (at least 4.70)." : "goes over the ledge. Go lower."}` : "Switch to Assume a value for untried moves, then drag the value.")),
    );
    const got = discounted(deployed);
    const planned = offGraph.V[env.start];
    const unseenUsed = deployed.filter((t) => !inA.has(`${t.state},${t.action}`) && !inB.has(`${t.state},${t.action}`));
    const fromA = deployed.filter((t) => inA.has(`${t.state},${t.action}`) && !inB.has(`${t.state},${t.action}`)).length;
    const fromB = deployed.filter((t) => inB.has(`${t.state},${t.action}`)).length;
    replace(
      offNote,
      h(
        "p",
        {},
        unseenUsed.length
          ? [
              h("strong", {}, `The plan promised ${fmt(planned, 2)} from the dock; driving it returned ${fmt(got, 2)}. `),
              `Its first move, ${deployed[0] ? ["up", "right", "down", "left"][deployed[0].action] : "?"} from the dock, is not in the data. The planner assumed that untried move was worth ${fmt(assumed, 2)} and nothing could check it: that edge leads over the ledge. Offline, optimism about missing edges is a hallucinated shortcut.`,
            ]
          : [
              h("strong", {}, `The plan promised ${fmt(planned, 2)} and driving it returned ${fmt(got, 2)}: better than either episode in the data (A ${fmt(returns.A, 2)}, B ${fmt(returns.B, 2)}). `),
              `The route uses ${fromA} edges that only episode A drove and ${fromB} from episode B, joined at the node where they cross. No episode in the data drove it. This is stitching: planning on the union of the episodes, not copying one of them.`,
            ],
      ),
    );
    offMission.update();
  }

  const offPicker = segmented(
    [
      { value: "pessimistic", label: "Leave untried moves out" },
      { value: "optimistic", label: "Assume a value for untried moves" },
    ],
    {
      value: offMode,
      label: "What to assume about untried moves",
      onChange: (v) => {
        offMode = v;
        assumedSlider.hidden = v !== "optimistic";
        offGraph = null; // a new assumption needs a new plan
        deployed = [];
        offScene.place("main", env.start, UP);
        renderOffline();
      },
    },
  );
  assumedSlider.hidden = true;

  // ---------- online ----------
  let onMode = "random";
  let onGraph;
  let onState;
  let onSteps;
  let onEpisodes;
  let episodeSteps;
  const onRng = new Rng(7);
  const onScene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], goalLabels: w.signs, caption: "Online: every move adds an edge, and the robot replans after each one" });
  const onView = new StateGraph(env, { goalLabels: w.labels });
  const onLoop = new LoopDiagram();
  const onCounters = h("div", { class: "counters" });
  const onNote = h("div", { class: "callout" });

  function resetOnline() {
    onGraph = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: GAMMA, unseen: onMode === "optimistic" ? OPTIMISTIC : PESSIMISTIC });
    [onState] = onEnv.reset();
    onSteps = 0;
    onEpisodes = 0;
    episodeSteps = 0;
    onScene.place("main", onState, UP);
    renderOnline();
  }

  function onlineStep(animate = true) {
    const a = onMode === "random" && onRng.random() < EPSILON ? onRng.integers(4) : onGraph.act(onState);
    const [next, r, done, , info] = onEnv.step(a);
    onGraph.add(makeTransition({ state: onState, action: a, reward: r, next_state: next, terminated: done }));
    onGraph.plan();
    if (animate) onScene.move("main", { from: onState, to: next, fellInto: info.fell_into, action: a }, { duration: 200 });
    onSteps += 1;
    episodeSteps += 1;
    onState = next;
    if (done || episodeSteps >= ONLINE_EPISODE_STEPS) {
      onEpisodes += 1;
      episodeSteps = 0;
      [onState] = onEnv.reset();
      if (animate) setTimeout(() => onScene.place("main", onState, UP), 260);
    }
    return done;
  }

  const onlineResult = { random: null, optimistic: null }; // greedy return after at least 10 episodes
  function runEpisodes(n) {
    const target = onEpisodes + n;
    let guard = 0;
    while (onEpisodes < target && guard++ < 5000) onlineStep(false);
    onScene.place("main", onState, UP);
    if (onEpisodes >= 10) onlineResult[onMode] = { ret: discounted(deploy(env, onGraph, 30)), steps: onSteps, fast: onGraph.terminal.has(w.far) };
    renderOnline();
  }
  const onMission = new Mission({
    title: "Collect your own data",
    goal: "Start from an empty graph twice, with two ways of exploring, and compare what each robot ends up knowing.",
    steps: [
      { text: "With “Trust what is seen, 20% random moves” selected, press Run 10 episodes.", done: () => onlineResult.random !== null },
      { text: "Select “Optimistic about untried moves” (a fresh robot) and press Run 10 episodes.", done: () => onlineResult.optimistic !== null },
    ],
    conclusion: () => {
      const r = onlineResult.random;
      const o = onlineResult.optimistic;
      return `After 10 episodes, the randomly exploring robot's best route returns ${fmt(r?.ret ?? 0, 2)} (${r?.fast ? "it did find the fast charger" : "it never found the fast charger"}); the optimistic robot's returns ${fmt(o?.ret ?? 0, 2)} (${o?.fast ? "it found the fast charger" : "it has not found the fast charger"}), and the best possible is ${fmt(optimum, 2)}. Online, the core problem is exploration: the graph only grows where the robot goes. Optimism, the assumption that failed offline, works here, because every untried move it is lured to gets tried and corrected. The chart below averages 20 robots of each kind.`;
    },
  });

  function renderOnline() {
    const known = new Set([...onGraph.visited, ...onGraph.terminal]);
    const edges = [];
    for (const e of allEdges) {
      const k = `${e.from},${e.action}`;
      if (!onGraph.edges.has(k)) continue;
      const best = onGraph.act(e.from) === e.action && onGraph.V[e.from] !== 0;
      edges.push({ ...e, role: best ? "best" : e.from === e.to ? "faint" : "plain" });
    }
    const stubs = {};
    for (const s of onGraph.visited) if (!onGraph.terminal.has(s)) stubs[s] = onGraph.untried(s);
    onView.render({ edges, values: onGraph.V, showValues: true, stubs, known, rings: new Set(onGraph.frontier()), robot: onState });
    const now = discounted(deploy(env, onGraph, 30));
    replace(
      onCounters,
      h("span", { class: "counter" }, h("span", { class: "dot env" }), "environment steps", h("b", {}, onSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot muted" }), "episodes", h("b", {}, onEpisodes)),
      h("span", { class: "counter" }, "known edges", h("b", {}, `${onGraph.edges.size} of ${allEdges.length}`)),
      h("span", { class: "counter" }, "frontier nodes", h("b", {}, onGraph.frontier().length)),
      h("span", { class: "counter" }, "greedy return from the dock", h("b", {}, `${fmt(now, 3)} of ${fmt(optimum, 3)}`)),
    );
    const farKnown = onGraph.terminal.has(w.far);
    onMission.update();
    replace(
      onNote,
      h(
        "p",
        {},
        onMode === "optimistic"
          ? [
              h("strong", {}, "Optimism turns every untried move into a promise of +10. "),
              `So the greedy robot drives to the nearest frontier node (dashed rings) and tries what is there. Once an untried move is too far away to beat what it already knows, it stops bothering. ${farKnown ? "It has found the fast charger." : "It has not found the fast charger yet."}`,
            ]
          : [
              h("strong", {}, "This robot only trusts what it has seen, and explores by moving at random 20% of the time. "),
              `Once it knows a charger it heads there, and reaching a new part of the graph needs a lucky run of random moves. ${farKnown ? "It has found the fast charger." : "It has not found the fast charger yet."}`,
            ],
      ),
    );
  }

  const onPicker = segmented(
    [
      { value: "random", label: "Trust what is seen, 20% random moves" },
      { value: "optimistic", label: "Optimistic about untried moves" },
    ],
    { value: onMode, label: "How the robot explores", onChange: (v) => ((onMode = v), resetOnline()) },
  );

  const chart = new LineChart({
    height: 240,
    xLabel: "episodes",
    yLabel: "greedy return from the dock (average of 20 robots)",
    yDomain: [0, 5],
    xFormat: (v) => String(Math.round(v)),
    yFormat: (v) => fmt(v, 1),
    caption: `The dashed line is the best possible, ${fmt(optimum, 3)}: the fast charger by a shortest route.`,
  });
  function runComparison() {
    const seeds = 20;
    const curves = { random: [], optimistic: [] };
    for (const strategy of Object.keys(curves)) {
      const sum = new Array(30).fill(0);
      for (let seed = 0; seed < seeds; seed++) onlineCurve(GridWorld.chargingRoom(), strategy, 100 + seed).forEach((v, i) => (sum[i] += v / seeds));
      curves[strategy] = sum;
    }
    results.onRandom = curves.random[29];
    results.onOpt = curves.optimistic[29];
    chart.update({
      series: [
        { id: "random", label: "trust what is seen, 20% random", color: "var(--series-1)", points: curves.random.map((v, i) => [i + 1, v]) },
        { id: "opt", label: "optimistic about untried moves", color: "var(--series-2)", points: curves.optimistic.map((v, i) => [i + 1, v]) },
        { id: "best", label: "best possible", color: "var(--ink-3)", dashed: true, points: [[1, optimum], [30, optimum]] },
      ],
      xDomain: [1, 30],
    });
    renderSummary();
  }

  // ---------- the 2×2 ----------
  const summary = h("div", { class: "grid-2x2" });
  function renderSummary() {
    const offP = results.offPess ?? discounted(deploy(env, offlineGraph(env, data, PESSIMISTIC, 10)));
    const offO = results.offOpt10 ?? discounted(deploy(env, offlineGraph(env, data, OPTIMISTIC, 10)));
    const cell = (value, title, text, tone) => h("div", { class: `cell ${tone}` }, h("span", {}, title), h("b", {}, value === undefined ? "run below" : `return ${fmt(value, 2)}`), h("span", { class: "note" }, text));
    replace(
      summary,
      h("div", { class: "head corner" }, ""),
      h("div", { class: "head" }, "Untried moves are left out (pessimism)"),
      h("div", { class: "head" }, "Untried moves are assumed worth +10 (optimism)"),
      h("div", { class: "head" }, "Offline: a fixed dataset"),
      cell(offP, "Stitching", "The best route inside the data, better than any single episode in it.", "good"),
      cell(offO, "Hallucinated shortcut", "A move nobody tried looks great and cannot be checked: over the ledge.", "bad"),
      h("div", { class: "head" }, "Online: the robot collects its own data"),
      cell(results.onRandom, "Stuck on what it knows", "It exploits the first charger it finds; only luck gets it further.", "bad"),
      cell(results.onOpt, "Directed exploration", "It goes where it has not been, until nothing untried could pay off.", "good"),
    );
  }

  root.append(
    lessonHeader("03", {
      lead: "The same robot and the same learning rule, in two situations. Offline, it must learn from a fixed dataset: its experience graph can never grow. Online, it collects its own data: it decides which edges to add. In both, one assumption matters more than any other: what an untried move is worth.",
      concepts: ["offline", "online", "stitching", "coverage", "exploration", "frontier", "pessimism", "optimism"],
    }),
    step("Question", h("p", { class: "question" }, "What changes when the robot can no longer collect new data?")),
    step(
      "Predict",
      predict({
        question: "A dataset has two episodes. A reaches the slow charger (+1) in 5 moves. B reaches the fast charger (+10) by a 12-move detour. Using only this data, can a learner find a route better than both?",
        choices: [
          { label: "No: it can at best copy episode B" },
          { label: "Yes: the episodes cross, and their pieces combine" },
          { label: "Only if it collects more data" },
        ],
        answer: 1,
        explain: () =>
          `Both episodes pass through the node above the slow charger. A gets there in 4 moves; B continues from there to the fast charger in 4 more. Joined, that is an 8-move route worth ${fmt(optimum, 2)}, better than A (${fmt(returns.A, 2)}) and B (${fmt(returns.B, 2)}), and no episode drove it.`,
      }),
    ),
    step("Summary", prose("Four combinations, computed on this page from the experiments below:"), summary),
    wideStep(
      "Offline",
      prose("The robot gets episodes A and B and nothing else. It plans on their experience graph, then drives the result without learning. Compare the two assumptions about the moves the data never tried (dashed stubs)."),
      offLoop.el,
      h(
        "div",
        { class: "bench" },
        offMission.el,
        h("div", { class: "toolbar" }, offPicker, assumedSlider, h("span", { class: "spacer" }), button("Plan on this data and drive", { kind: "learn", onClick: planOffline })),
        h("div", { class: "bench-grid part1-grid" }, offScene.el, h("div", { class: "figure" }, offView.el, h("div", { class: "key" },
          h("span", { class: "key-item" }, h("span", { class: "key-swatch data-a" }), `episode A (return ${fmt(returns.A, 2)})`),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch data-b" }), `episode B (return ${fmt(returns.B, 2)})`),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch route-band" }), "the route the plan drives"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch stub" }), "a move the data never tried"),
        ))),
        offNote,
      ),
      prose(
        "So offline RL has one ability and one limit, both about the graph. Stitching: values flow through every node episodes share, so the best route inside the data is found even if no one drove it. Coverage: a move outside the data can be neither used nor tested, and assuming it is good is the classic failure of offline RL. Practical offline methods (CQL, IQL, ...) are ways of being pessimistic about what the data does not cover.",
      ),
    ),
    wideStep(
      "Online",
      prose("Now the robot starts with an empty graph and collects its own data. Offline, optimism was the problem; online, it is the solution, because an untried move can simply be tried."),
      onLoop.el,
      h(
        "div",
        { class: "bench" },
        onMission.el,
        h("div", { class: "toolbar" }, onPicker, h("span", { class: "spacer" }), button("Step", { kind: "env", onClick: () => (onlineStep(true), renderOnline()) }), button("Run 1 episode", { onClick: () => runEpisodes(1) }), button("Run 10 episodes", { onClick: () => runEpisodes(10) }), button("Start over", { kind: "ghost", onClick: resetOnline })),
        onCounters,
        h("div", { class: "bench-grid part1-grid" }, onScene.el, h("div", { class: "figure" }, onView.el, h("div", { class: "key" },
          h("span", { class: "key-item" }, h("span", { class: "key-swatch ring" }), "frontier: a known node with untried moves"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch stub" }), "untried move"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch best-edge" }), "best known edge"),
        ))),
        onNote,
      ),
      chart.el,
      prose(
        "The core of online RL is exploration: the graph only grows where the robot goes. Random moves explore blindly. Optimism explores on purpose: it sends the robot to the frontier, and stops once nothing untried could beat what is known. On/off-policy is a separate question: whose moves built the graph. Here the robot builds its own, but the values it learns are for its greedy policy, not for the random or optimistic moves that collected the data.",
      ),
    ),
    step(
      "Equation",
      equation(
        "Q(s,a) = \\begin{cases} r(s,a) + \\gamma\\,V(s') & (s,a)\\ \\text{in the experience graph} \\\\ \\text{left out} & \\text{untried, pessimistic} \\\\ v_{\\text{assumed}} & \\text{untried, optimistic} \\end{cases}",
        h("span", {}, "Everything in this lesson is this one choice. Offline, ", tex("v_{\\text{assumed}}"), " can never be corrected, so it must not exceed what the data proves. Online, a large ", tex("v_{\\text{assumed}}"), " is corrected the first time the move is tried, and until then it pulls the robot towards it."),
      ),
    ),
    step("Code", codeBlock(extractDef(graphSource, "q"), { title: "visualrl/algorithms/tabular/experience_graph.py" })),
    step(
      "Challenge",
      prose("Offline, choose Assume a value for untried moves and find the largest assumed value at which the plan still drives the stitched route. What is special about that number?"),
      offChallenge,
    ),
    lessonFooter("03"),
  );

  offScene.place("main", env.start, UP);
  renderOffline();
  resetOnline();
  renderSummary();
  const comparison = setTimeout(runComparison, 50); // fills the Online row of the summary
  return () => {
    clearTimeout(comparison);
    offScene.dispose();
    onScene.dispose();
  };
}
