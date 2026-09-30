// Lesson 09: SARSA vs Q-learning on the cliff (Sutton & Barto, Example 6.6).
import qSource from "../../../visualrl/algorithms/tabular/q_learning.py";
import sarsaSource from "../../../visualrl/algorithms/tabular/sarsa.py";
import { makeTransition } from "../rl/core.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { optimalValues } from "../rl/tabular/dp.js";
import { runEpisode } from "../rl/tabular/episode.js";
import { QLearning, Sarsa } from "../rl/tabular/q-agents.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, segmented, shortcuts, slider } from "../ui/dom.js";
import { ARROWS, fmt, fmtSigned } from "../ui/format.js";
import { greedyPath, QGrid, valueLegend } from "../ui/grid-view.js";
import { Ledger } from "../ui/ledger.js";
import { LineChart } from "../ui/line-chart.js";
import { equation, tex } from "../ui/math.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef, lineOf } from "../ui/source.js";

const METHODS = {
  sarsa: { label: "SARSA", Agent: Sarsa, color: "var(--series-1)" },
  q_learning: { label: "Q-learning", Agent: QLearning, color: "var(--series-2)" },
};
const SPEEDS = [
  { label: "2/s", ms: 500 },
  { label: "8/s", ms: 125 },
  { label: "30/s", ms: 33 },
];

const cellName = (env, st) => {
  const [r, c] = env.toCell(st);
  return st === env.start ? `S (${r},${c})` : `(${r},${c})`;
};

// One agent acting in its own copy of the cliff. Acting and learning are separate calls.
class Run {
  constructor(method, params) {
    this.method = method;
    this.params = params;
    this.reset();
  }

  reset() {
    const { Agent } = METHODS[this.method];
    this.env = GridWorld.cliff();
    this.agent = new Agent({ nStates: 48, nActions: 4, gamma: 1, seed: 7, ...this.params });
    [this.state] = this.env.reset();
    this.action = this.agent.act(this.state);
    this.pending = [];
    this.log = [];
    this.envSteps = 0;
    this.episodes = 0;
    this.episodeReturn = 0;
    this.lastMove = null;
  }

  stepEnv() {
    const { state, action } = this;
    const [next_state, reward, terminated, truncated, info] = this.env.step(action);
    const next_action = terminated ? null : this.agent.act(next_state);
    const transition = makeTransition({ state, action, reward, next_state, terminated, truncated, next_action });
    const entry = { id: this.envSteps + 1, transition, learned: false };
    this.pending.push(entry);
    this.log.push(entry);
    if (this.log.length > 200) this.log.shift();
    this.envSteps += 1;
    this.episodeReturn += reward;
    this.lastMove = { from: state, to: next_state, fellInto: info.fell_into };
    if (terminated || truncated) {
      this.episodes += 1;
      this.episodeReturn = 0;
      [this.state] = this.env.reset();
      this.action = this.agent.act(this.state);
    } else {
      this.state = next_state;
      this.action = next_action;
    }
    return transition;
  }

  // Learn from the oldest transition that has not been learned from yet.
  learnOnce() {
    const entry = this.pending.shift();
    if (!entry) return null;
    entry.learned = true;
    return this.agent.learnStep(entry.transition);
  }

  // Learn what is pending, then act and learn until `episodes` more episodes finish.
  fastForward(episodes) {
    const traces = [];
    while (this.pending.length) traces.push(this.learnOnce());
    const target = this.episodes + episodes;
    let guard = 0;
    while (this.episodes < target && guard++ < 200000) {
      this.stepEnv();
      traces.push(this.learnOnce());
    }
    this.lastMove = null;
    return traces;
  }
}

function ledgerGroups(env, trace) {
  const sarsa = trace.algorithm === "sarsa";
  const aName = (a) => (a === null || a === undefined ? "—" : `${ARROWS[a]} ${GridWorld.ACTION_NAMES[a]}`);
  const bootstrapRow = trace.terminated
    ? { label: "next value (episode ended)", formula: "0", value: fmt(0) }
    : sarsa
      ? { label: `next value, a′ = ${ARROWS[trace.target_action]}`, formula: "Q(s',a')", value: fmt(trace.bootstrap_value) }
      : { label: `best next value, uses ${ARROWS[trace.target_action]}`, formula: "\\max_{a'} Q(s',a')", value: fmt(trace.bootstrap_value) };
  return [
    {
      label: "Experience",
      tone: "env",
      rows: [
        { label: "state", formula: "s", value: cellName(env, trace.state) },
        { label: "action", formula: "a", value: aName(trace.action) },
        { label: "reward", formula: "r", value: fmt(trace.reward, 0) },
        { label: "next state", formula: "s'", value: cellName(env, trace.next_state) },
        { label: "next action (behavior)", formula: "a'", value: aName(trace.next_action) },
      ],
    },
    {
      label: "Computation",
      rows: [
        { label: "current estimate", formula: "Q(s,a)", value: fmt(trace.prediction) },
        bootstrapRow,
        { label: "discount", formula: "\\gamma", value: fmt(trace.discount, 2) },
        { label: "target", formula: sarsa ? "r+\\gamma Q(s',a')" : "r+\\gamma \\max Q(s',\\cdot)", value: fmt(trace.target), emph: true },
        { label: "TD error", formula: "\\delta", value: fmtSigned(trace.error), emph: true },
        { label: "learning rate", formula: "\\alpha", value: fmt(trace.learning_rate, 2) },
      ],
    },
    {
      label: "Update",
      tone: "learn",
      rows: [
        { label: "change", formula: "\\alpha\\,\\delta", value: fmtSigned(trace.learning_rate * trace.error) },
        { label: "Q(s,a)", formula: "", value: `${fmt(trace.value_before)} → ${fmt(trace.value_after)}`, result: true },
      ],
    },
  ];
}

// True when the behavior policy's next action is not greedy, so Q-learning's target is about an action it will not take.
function offPolicyMoment(trace) {
  if (trace.algorithm !== "q_learning" || trace.terminated || trace.next_action === null) return false;
  const q = trace.next_q_values;
  return q[trace.next_action] < q[trace.target_action] - 1e-9;
}

function explainTrace(trace) {
  if (!trace) return [h("strong", {}, "Nothing learned yet. "), "Step the environment to collect a transition, then press Learn once."];
  if (trace.terminated) return [h("strong", {}, "The episode ended at the goal. "), "Nothing is borrowed from s′: the target is just the reward."];
  const a2 = ARROWS[trace.next_action];
  const target = ARROWS[trace.target_action];
  if (trace.algorithm === "sarsa") {
    const q = trace.next_q_values;
    const exploratory = q[trace.next_action] < Math.max(...q) - 1e-9;
    return [
      h("strong", {}, `SARSA's target used a′ = ${a2}, the action the agent will really take next. `),
      exploratory
        ? `That action is exploratory (not the best in s′), and its value ${fmt(q[trace.next_action], 2)} still went into the target. This is how SARSA learns that walking next to the cliff is risky under ε-greedy behavior.`
        : "Here it is also the greedy action, so SARSA and Q-learning would compute the same target.",
    ];
  }
  if (offPolicyMoment(trace)) {
    return [
      h("strong", {}, `Off-policy: the agent will actually move ${a2} from s′, but the target used ${target}. `),
      "Q-learning always bootstraps from the best next action, whatever the ε-greedy behavior does. It learns about the greedy policy while following another one.",
    ];
  }
  return [
    h("strong", {}, `The target used ${target}, the best action in s′. `),
    trace.next_action === null
      ? ""
      : `The behavior policy also chose ${a2} this time, so the two agree. Keep stepping: exploration will make them differ.`,
  ];
}

export function mount(root) {
  const params = { alpha: 0.5, epsilon: 0.1 };
  const runs = { sarsa: new Run("sarsa", params), q_learning: new Run("q_learning", params) };
  let method = "q_learning";
  let showValues = false;
  let showGreedy = true;
  let showReference = false;
  let timer = null;
  let speed = 1;
  let caught = null;
  const ledgers = {};

  const env = runs.sarsa.env;
  const reference = optimalValues(GridWorld.cliff().model(), 1);
  const referencePath = greedyPath(env, reference.Q);

  const grid = new QGrid(env);
  const counters = h("div", { class: "counters", "aria-live": "polite" });
  const callout = h("div", { class: "callout learn" });
  const experience = h("div", { class: "experience" });
  const ledgerSlot = h("div");
  const challengeStatus = h("div", { class: "callout" });

  for (const key of Object.keys(METHODS)) {
    ledgers[key] = new Ledger({
      titleFor: (t) => `${METHODS[t.algorithm].label} · learning step ${t.learn_step}`,
      groupsFor: (t) => ledgerGroups(env, t),
      empty: "No learning step yet. Collect a transition with Step environment, then press Learn once.",
      onSelect: () => render(),
    });
  }

  const run = () => runs[method];
  const ledger = () => ledgers[method];

  function record(traces) {
    const list = traces.filter(Boolean);
    if (!list.length) return;
    ledger().pushMany(list);
    if (!caught) caught = list.find(offPolicyMoment) ?? null;
  }

  const actions = {
    step() {
      run().stepEnv();
      render();
    },
    learn() {
      record([run().learnOnce()]);
      render();
    },
    both() {
      run().stepEnv();
      record([run().learnOnce()]);
      render();
    },
    episodes(n) {
      stop();
      record(run().fastForward(n));
      render();
    },
    reset() {
      stop();
      run().reset();
      ledger().reset();
      render();
    },
  };

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    playBtn.textContent = "Play";
    playBtn.setAttribute("aria-pressed", "false");
  }

  function togglePlay() {
    if (timer) return stop();
    timer = setInterval(actions.both, SPEEDS[speed].ms);
    playBtn.textContent = "Pause";
    playBtn.setAttribute("aria-pressed", "true");
  }

  const stepBtn = button("Step environment", { kind: "env", kbd: "E", onClick: actions.step });
  const learnBtn = button("Learn once", { kind: "learn", kbd: "L", onClick: actions.learn });
  const bothBtn = button("Step + learn", { kbd: "B", onClick: actions.both });
  const playBtn = button("Play", { kbd: "P", onClick: togglePlay });
  const methodPicker = segmented(
    Object.entries(METHODS).map(([value, m]) => ({ value, label: m.label, swatch: m.color })),
    {
      value: method,
      label: "Algorithm",
      onChange: (v) => {
        stop();
        method = v;
        replace(ledgerSlot, ledger().el);
        render();
      },
    },
  );
  const speedPicker = segmented(
    SPEEDS.map((sp, i) => ({ value: i, label: sp.label })),
    {
      value: speed,
      label: "Play speed",
      onChange: (v) => {
        speed = v;
        if (timer) {
          stop();
          togglePlay();
        }
      },
    },
  );

  const toggle = (label, checked, onChange) => {
    const input = h("input", { type: "checkbox", checked: checked || null });
    input.addEventListener("change", () => onChange(input.checked));
    return h("label", { class: "toggle" }, input, label);
  };

  function render() {
    const r = run();
    const t = ledger().current;
    grid.render({
      Q: r.agent.Q,
      agent: r.state,
      nextAction: r.action,
      move: ledger().browsing ? null : r.lastMove,
      updated: t ? { s: t.state, a: t.action } : null,
      target: t && !t.terminated ? { s: t.next_state, a: t.target_action } : null,
      behavior: t && t.next_action !== null ? { s: t.next_state, a: t.next_action } : null,
      showValues,
      paths: [
        ...(showReference ? [{ states: referencePath, color: "var(--ink-2)", dashed: true, width: 2.5 }] : []),
        ...(showGreedy && r.agent.learnSteps ? [{ states: greedyPath(env, r.agent.Q), color: METHODS[method].color, width: 3, opacity: 0.8 }] : []),
      ],
    });

    replace(
      counters,
      h("span", { class: "counter" }, h("span", { class: "dot env" }), "environment steps", h("b", {}, r.envSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot learn" }), "learning steps", h("b", {}, r.agent.learnSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot muted" }), "episodes", h("b", {}, r.episodes)),
      h("span", { class: "counter" }, "waiting to be learned", h("b", {}, r.pending.length)),
    );
    learnBtn.disabled = r.pending.length === 0;

    replace(callout, h("p", {}, explainTrace(t)), ledger().browsing ? h("p", { class: "note" }, "You are looking at an earlier update. The grid shows the current Q-table.") : null);

    const recent = r.log.slice(-8).reverse();
    replace(
      experience,
      h("div", { class: "ledger-head" }, h("span", { class: "dot env" }), h("span", { class: "title" }, "Experience"), h("span", { class: "grow" }), h("span", { class: "note" }, "newest first")),
      recent.length
        ? h(
            "div",
            { class: "experience-scroll" },
            h(
              "table",
              {},
              h("thead", {}, h("tr", {}, ["#", "s", "a", "r", "s′", "a′", ""].map((c) => h("th", {}, c)))),
              h(
                "tbody",
                {},
                recent.map(({ id, transition: tr, learned }) =>
                  h(
                    "tr",
                    {},
                    h("td", {}, id),
                    h("td", {}, cellName(env, tr.state)),
                    h("td", {}, ARROWS[tr.action]),
                    h("td", {}, fmt(tr.reward, 0)),
                    h("td", {}, cellName(env, tr.next_state)),
                    h("td", {}, tr.next_action === null ? "—" : ARROWS[tr.next_action]),
                    h("td", {}, h("span", { class: `pill ${learned ? "learned" : "pending"}` }, learned ? "learned" : "waiting")),
                  ),
                ),
              ),
            ),
          )
        : h("p", { class: "ledger-empty" }, "No transitions yet."),
    );

    replace(
      challengeStatus,
      caught
        ? [
            h("p", {}, h("span", { class: "verdict good" }, "Found one. "), `At learning step ${caught.learn_step} the agent was about to move ${ARROWS[caught.next_action]} from ${cellName(env, caught.next_state)}, but Q-learning's target used ${ARROWS[caught.target_action]}, the greedy action, with value ${fmt(caught.bootstrap_value, 2)} instead of ${fmt(caught.next_q_values[caught.next_action], 2)}.`),
          ]
        : [h("p", {}, h("span", { class: "status-line" }, h("span", { class: "dot muted" }), "Not yet. Switch to Q-learning and keep stepping and learning. This page is watching every update."))],
    );
  }

  replace(ledgerSlot, ledger().el);
  methodPicker.select(method);

  const bench = h(
    "div",
    { class: "bench" },
    h(
      "div",
      { class: "toolbar" },
      methodPicker,
      h("span", { class: "spacer" }),
      h("div", { class: "toolbar-group" }, stepBtn, learnBtn, bothBtn),
      h("div", { class: "toolbar-group" }, playBtn, speedPicker),
    ),
    counters,
    h(
      "div",
      { class: "bench-grid" },
      h(
        "div",
        { class: "figure" },
        grid.el,
        h(
          "div",
          { class: "key" },
          valueLegend(100),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch next" }), "next action the agent will take"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch updated" }), "Q(s,a) just updated"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch target" }), "entry of s′ used in the target"),
        ),
        callout,
        experience,
      ),
      h("div", { class: "figure" }, ledgerSlot),
    ),
    h(
      "div",
      { class: "controls-row" },
      slider({ id: "l09-eps", label: "exploration ε", min: 0, max: 0.5, step: 0.01, value: params.epsilon, format: (v) => v.toFixed(2), onInput: (v) => {
        params.epsilon = v;
        Object.values(runs).forEach((r) => (r.agent.epsilon = v));
      } }),
      slider({ id: "l09-alpha", label: "learning rate α", min: 0.05, max: 1, step: 0.05, value: params.alpha, format: (v) => v.toFixed(2), onInput: (v) => {
        params.alpha = v;
        Object.values(runs).forEach((r) => (r.agent.alpha = v));
      } }),
      h(
        "div",
        { class: "toolbar-group" },
        button("Run 1 episode", { onClick: () => actions.episodes(1) }),
        button("Run 50 episodes", { onClick: () => actions.episodes(50) }),
        button("Reset", { kind: "ghost", onClick: actions.reset }),
      ),
      h(
        "div",
        { class: "toolbar-group" },
        toggle("Q-values in every cell", showValues, (v) => ((showValues = v), render())),
        toggle("greedy route", showGreedy, (v) => ((showGreedy = v), render())),
        toggle("optimal route (reference, dashed)", showReference, (v) => ((showReference = v), render())),
      ),
    ),
  );

  // ---------- comparison: both methods, many episodes ----------
  const chart = new LineChart({
    height: 260,
    xLabel: "episode",
    yLabel: "reward per episode (average of 10 runs, 10-episode window)",
    yDomain: [-120, 0],
    caption: "Episodes worse than −120 are drawn at −120. Rewards are collected while exploring with ε-greedy, so they measure the policy the agent actually follows.",
  });
  const routes = new QGrid(env, { plain: true });
  routes.render({ paths: [] });
  const compareNote = h("p", { class: "note" }, "");
  const compareBtn = button("Train both again", { onClick: () => runComparison() });
  function runComparison() {
    compareBtn.disabled = true;
    compareNote.textContent = "Training 2 × 10 agents for 500 episodes…";
    setTimeout(() => {
      const result = compareMethods(params);
      chart.update({
        series: Object.entries(result).map(([key, r]) => ({ id: key, label: METHODS[key].label, color: METHODS[key].color, points: r.curve })),
      });
      routes.render({
        paths: [
          { states: referencePath, color: "var(--ink-2)", dashed: true, width: 2 },
          ...Object.entries(result).map(([key, r]) => ({ states: r.path, color: METHODS[key].color, width: 3.5, opacity: 0.85 })),
        ],
      });
      compareNote.textContent = `Trained with ε = ${params.epsilon.toFixed(2)}, α = ${params.alpha.toFixed(2)}. Average reward over the last 100 episodes: SARSA ${fmt(result.sarsa.last100, 1)}, Q-learning ${fmt(result.q_learning.last100, 1)}.`;
      compareBtn.disabled = false;
    }, 30);
  }

  const unbind = shortcuts({ e: actions.step, l: actions.learn, b: actions.both, p: togglePlay });

  const sarsaCode = extractDef(sarsaSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)  # every quantity above, for the ledger" });
  const qCode = extractDef(qSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)  # every quantity above, for the ledger" });

  root.append(
    lessonHeader("09", {
      lead: "Both algorithms learn a Q-table from the same kind of experience on the same cliff. They differ in one term of the target, and that one term decides which route they learn.",
      concepts: ["TD control", "behavior policy", "target policy", "on-policy", "off-policy", "ε-greedy"],
    }),
    step("Question", h("p", { class: "question" }, "If SARSA and Q-learning explore the same cliff, why do they learn different routes?")),
    step(
      "Predict",
      predict({
        question: "Both agents explore with ε = 0.1 for 500 episodes. Afterwards, which route does each one's greedy policy take from S to G?",
        choices: [
          { label: "Both walk right along the cliff edge", detail: "the shortest route, 13 steps" },
          { label: "SARSA keeps a safe distance; Q-learning walks the edge" },
          { label: "SARSA walks the edge; Q-learning keeps a safe distance" },
          { label: "Both keep a safe distance" },
        ],
        answer: 1,
        explain: () =>
          "Q-learning's target uses the best next action, so it learns the values of the greedy policy, and for the greedy policy the edge is safe and shortest. SARSA's target uses the action ε-greedy exploration will actually take, which sometimes steps off the edge, so it learns that the edge is costly. Check it with the comparison below.",
      }),
    ),
    wideStep(
      "Experiment",
      prose(
        "Stepping the environment only collects experience. Learning is a separate button that consumes the oldest waiting transition. Step a few times, then learn a few times, and watch which entries of the table change.",
      ),
      bench,
    ),
    wideStep(
      "Inspect",
      prose("Now train both agents for many episodes. The learning curve shows the rewards they collect while exploring; the map shows the greedy route each ends with."),
      h("div", { class: "toolbar" }, compareBtn, compareNote),
      chart.el,
      h("div", { class: "figure routes-map" }, routes.el, h("div", { class: "key" },
        h("span", { class: "key-item" }, h("span", { class: "swatch", style: { background: METHODS.sarsa.color, height: "3px" } }), "SARSA greedy route"),
        h("span", { class: "key-item" }, h("span", { class: "swatch", style: { background: METHODS.q_learning.color, height: "3px" } }), "Q-learning greedy route"),
        h("span", { class: "key-item" }, h("span", { class: "key-swatch ref" }), "optimal route, computed from the model (reference)"),
      )),
      prose(
        "Q-learning ends with the optimal route yet collects less reward while training: with ε = 0.1 it occasionally takes a random step next to the cliff and falls. SARSA accounts for its own exploration, so its route is longer but its falls are rare. Set ε to 0 and train again to see the difference shrink.",
      ),
    ),
    step(
      "Equation",
      equation("\\text{SARSA:}\\quad Q(s,a) \\leftarrow Q(s,a) + \\alpha\\big[\\, r + \\gamma\\, Q(s',a') - Q(s,a) \\,\\big]"),
      equation(
        "\\text{Q-learning:}\\quad Q(s,a) \\leftarrow Q(s,a) + \\alpha\\big[\\, r + \\gamma \\max_{a'} Q(s',a') - Q(s,a) \\,\\big]",
        h("span", {}, "Everything is the same except the next-state term. In SARSA, ", tex("a'"), " is the action the behavior policy takes. In Q-learning it is replaced by the maximum over actions, whatever the behavior does."),
      ),
    ),
    step(
      "Code",
      prose("This is the code from the visualrl package. The highlighted line is the only real difference."),
      h(
        "div",
        { class: "two-col" },
        codeBlock(sarsaCode, { title: "visualrl/algorithms/tabular/sarsa.py", highlight: [lineOf(sarsaCode, "bootstrap_value = 0.0 if")] }),
        codeBlock(qCode, { title: "visualrl/algorithms/tabular/q_learning.py", highlight: [lineOf(qCode, "target_action = int(np.argmax")] }),
      ),
    ),
    step(
      "Challenge",
      prose(
        "Catch Q-learning learning about an action it did not take: find an update where the agent's next move (teal arrow) is not the greedy action used in the target (dashed outline).",
      ),
      challengeStatus,
    ),
    lessonFooter("09"),
  );

  render();
  runComparison();
  return () => {
    stop();
    unbind();
  };
}

function compareMethods({ alpha, epsilon }, { episodes = 500, seeds = 10, window = 10 } = {}) {
  const result = {};
  for (const [key, { Agent }] of Object.entries(METHODS)) {
    const sums = new Array(episodes).fill(0);
    let path = [];
    for (let seed = 0; seed < seeds; seed++) {
      const env = GridWorld.cliff();
      const agent = new Agent({ nStates: 48, nActions: 4, alpha, epsilon, gamma: 1, seed: 100 + seed });
      for (let e = 0; e < episodes; e++) {
        const { trajectory } = runEpisode(env, agent, { maxSteps: 5000 });
        sums[e] += trajectory.rewards.reduce((a, b) => a + b, 0);
      }
      const route = greedyPath(env, agent.Q);
      if (!path.length && env.goals.includes(route[route.length - 1])) path = route;
      if (seed === seeds - 1 && !path.length) path = route;
    }
    const avg = sums.map((v) => v / seeds);
    const curve = avg.map((_, i) => {
      const lo = Math.max(0, i - window + 1);
      const slice = avg.slice(lo, i + 1);
      return [i + 1, Math.max(-120, slice.reduce((a, b) => a + b, 0) / slice.length)];
    });
    const last100 = avg.slice(-100).reduce((a, b) => a + b, 0) / 100;
    result[key] = { curve, path, last100 };
  }
  return result;
}
