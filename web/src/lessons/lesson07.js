// Lesson 07: Monte Carlo vs TD(0) on the five-state random walk (Sutton & Barto, Example 6.2).
import tdSource from "../../../visualrl/algorithms/tabular/td.py";
import mcSource from "../../../visualrl/algorithms/tabular/monte_carlo.py";
import { Trajectory, makeTransition } from "../rl/core.js";
import { Chain } from "../rl/envs/chain.js";
import { uniformPolicy } from "../rl/policies.js";
import { Rng } from "../rl/rng.js";
import { exactPolicyValue } from "../rl/tabular/dp.js";
import { MonteCarlo } from "../rl/tabular/monte-carlo.js";
import { NStepTD } from "../rl/tabular/n-step-td.js";
import { TD0 } from "../rl/tabular/td0.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, s, shortcuts, slider } from "../ui/dom.js";
import { fmt, fmtShort, fmtSigned } from "../ui/format.js";
import { Ledger } from "../ui/ledger.js";
import { LineChart } from "../ui/line-chart.js";
import { Mission } from "../ui/mission.js";
import { equation, tex } from "../ui/math.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef, lineOf } from "../ui/source.js";

const NAMES = ["T", "A", "B", "C", "D", "E", "T"];
const METHODS = {
  monte_carlo: { label: "Monte Carlo", color: "var(--series-1)" },
  td0: { label: "TD(0)", color: "var(--series-2)" },
};
const N_INFINITE = 10;
const ARROW = ["←", "→"];

function makeLearners({ alphaTD = 0.1, alphaMC = 0.1, initial = 0.5 } = {}) {
  const td = new TD0({ nStates: 7, alpha: alphaTD, gamma: 1, initialValue: initial });
  const mc = new MonteCarlo({ nStates: 7, alpha: alphaMC, gamma: 1, initialValue: initial });
  td.V[0] = td.V[6] = mc.V[0] = mc.V[6] = 0;
  return { td, mc };
}

// The chain with value bars under each interior state: one bar per method, reference dashed.
function chainValues({ agent, series, reference, highlight }) {
  const W = 80;
  const width = 7 * W + 40;
  const cellY = 8;
  const cellH = 40;
  const barTop = cellY + cellH + 26;
  const barH = 110;
  const height = barTop + barH + 30;
  const X = (st) => 36 + st * W;
  const Y = (v) => barTop + (1 - v) * barH;
  const parts = [];
  for (const v of [0, 0.5, 1]) {
    parts.push(s("line", { x1: X(1) - 6, x2: X(6) - 8, y1: Y(v), y2: Y(v), class: "grid-line", opacity: v ? 0.6 : 1 }));
    parts.push(s("text", { x: X(1) - 12, y: Y(v) + 4, "text-anchor": "end", class: "chart-label" }, fmtShort(v)));
  }
  for (let st = 0; st < 7; st++) {
    const terminal = st === 0 || st === 6;
    parts.push(s("rect", { x: X(st) + 3, y: cellY, width: W - 6, height: cellH, rx: 6, class: terminal ? "chain-terminal" : "chain-cell" }));
    parts.push(s("text", { x: X(st) + W / 2, y: cellY + cellH / 2 + 5, "text-anchor": "middle", class: terminal ? "chain-reward" : "cell-mark" }, terminal ? (st === 6 ? "+1" : "0") : NAMES[st]));
    if (terminal) continue;
    const bw = 16;
    series.forEach((sr, i) => {
      const v = sr.V[st];
      const bx = X(st) + W / 2 - bw - 2 + i * (bw + 4);
      const hl = highlight && highlight.method === sr.id && highlight.state === st;
      parts.push(s("rect", { x: bx, y: Math.min(Y(v), Y(0)), width: bw, height: Math.max(1, Math.abs(Y(0) - Y(v))), rx: 3, fill: sr.color, opacity: hl ? 1 : 0.85 }));
      if (hl) {
        parts.push(s("rect", { x: bx - 3, y: Math.min(Y(v), Y(highlight.before)) - 3, width: bw + 6, height: Math.abs(Y(v) - Y(highlight.before)) + 6, rx: 4, class: "hl-bar" }));
        parts.push(s("line", { x1: bx - 3, x2: bx + bw + 3, y1: Y(highlight.before), y2: Y(highlight.before), stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "2 2" }));
      }
      parts.push(s("text", { x: bx + bw / 2, y: barTop + barH + 14 + i * 12, "text-anchor": "middle", class: "chart-label" }, fmt(v, 2)));
    });
    if (reference) {
      parts.push(s("line", { x1: X(st) + W / 2 - 24, x2: X(st) + W / 2 + 24, y1: Y(reference[st]), y2: Y(reference[st]), stroke: "var(--ink)", "stroke-width": 1.6, "stroke-dasharray": "4 3" }));
    }
  }
  if (agent !== null && agent !== undefined) {
    parts.push(s("circle", { cx: X(agent) + W / 2, cy: cellY + cellH + 11, r: 7, class: "agent" }));
  }
  return s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Random walk chain with value estimates", class: "chain-svg values" }, parts);
}

function ledgerGroups(trace) {
  const td = trace.algorithm === "td0";
  const name = (st) => NAMES[st];
  return [
    {
      label: "Experience",
      tone: "env",
      rows: td
        ? [
            { label: "state", formula: "s_t", value: name(trace.state) },
            { label: "reward", formula: "r_t", value: fmtShort(trace.reward) },
            { label: "next state", formula: "s_{t+1}", value: name(trace.next_state) + (trace.terminated ? " (end)" : "") },
          ]
        : [
            { label: "timestep", formula: "t", value: String(trace.timestep) },
            { label: "state", formula: "s_t", value: name(trace.state) },
            { label: "rewards until the end", formula: "r_t,\\dots,r_{T-1}", value: trace.rewards.map((r) => fmtShort(r)).join(" ") },
          ],
    },
    {
      label: "Computation",
      rows: td
        ? [
            { label: "current estimate", formula: "V(s_t)", value: fmt(trace.prediction) },
            { label: trace.terminated ? "next value (episode ended)" : "next-state estimate", formula: "V(s_{t+1})", value: fmt(trace.bootstrap_value) },
            { label: "discount", formula: "\\gamma", value: fmt(trace.discount, 2) },
            { label: "target", formula: "r_t+\\gamma V(s_{t+1})", value: fmt(trace.target), emph: true },
            { label: "TD error", formula: "\\delta", value: fmtSigned(trace.error), emph: true },
            { label: "learning rate", formula: "\\alpha", value: fmt(trace.learning_rate, 2) },
          ]
        : [
            { label: "current estimate", formula: "V(s_t)", value: fmt(trace.prediction) },
            { label: "target: the return", formula: "G_t", value: fmt(trace.target), emph: true },
            { label: "error", formula: "G_t - V(s_t)", value: fmtSigned(trace.error), emph: true },
            { label: "learning rate", formula: "\\alpha", value: fmt(trace.learning_rate, 2) },
          ],
    },
    {
      label: "Update",
      tone: "learn",
      rows: [{ label: `V(${name(trace.state)})`, formula: "", value: `${fmt(trace.value_before)} → ${fmt(trace.value_after)}`, result: true }],
    },
  ];
}

// The n-step target of timestep t, drawn as the rewards it adds up and the value it borrows.
function targetInspector({ episode, t, n, V }) {
  if (t === null || !episode.length) return h("p", { class: "ledger-empty" }, "Collect a transition, then pick a row of the episode table.");
  const nEff = n >= N_INFINITE ? 1000 : n;
  const probe = new NStepTD({ nStates: 7, n: nEff, gamma: 1 });
  probe.V = V; // borrow TD's current estimates; target() reads them and changes nothing
  const ready = probe.ready(episode, t);
  const chips = [];
  const T = episode.length;
  const end = ready ? Math.min(t + nEff, T) : null;
  for (let k = t; k < T; k++) {
    const used = ready && k < end;
    chips.push(
      h(
        "span",
        { class: `rchip${used ? " used" : ""}` },
        h("small", {}, tex(`r_{${k}}`)),
        h("b", { class: "num" }, fmtShort(episode.at(k).reward)),
      ),
    );
  }
  const title = n >= N_INFINITE ? "n = ∞: the Monte Carlo return" : n === 1 ? "n = 1: the TD(0) target" : `n = ${n}`;
  if (!ready) {
    chips.push(h("span", { class: "rchip missing" }, h("small", {}, tex(`r_{${T}}\\dots`)), h("b", {}, "?")));
    return h(
      "div",
      { class: "inspector" },
      h("div", { class: "panel-title" }, title),
      h("div", { class: "chips-row" }, chips),
      h("p", { class: "note" }, `Not computable yet: this target needs rewards up to r${sub(t + nEff - 1)}, and the episode has only reached r${sub(T - 1)}. ${n >= N_INFINITE ? "Monte Carlo has to wait for the episode to end." : "Collect more transitions."}`),
    );
  }
  const parts = probe.target(episode, t);
  if (parts.bootstrap_state !== null) {
    chips.push(
      h(
        "span",
        { class: "rchip bootstrap" },
        h("small", {}, tex(`V(${NAMES[parts.bootstrap_state]})`)),
        h("b", { class: "num" }, fmt(parts.bootstrap_value, 3)),
      ),
    );
  }
  const sum = parts.rewards.map((r) => fmtShort(r)).join(" + ");
  return h(
    "div",
    { class: "inspector" },
    h("div", { class: "panel-title" }, title),
    h("div", { class: "chips-row" }, chips),
    h(
      "p",
      { class: "note num" },
      `target = ${sum}${parts.bootstrap_state !== null ? ` + ${fmt(parts.bootstrap_value, 3)}` : ""} = ${fmt(parts.target, 3)}`,
      parts.bootstrap_state !== null ? `   (borrows TD's estimate of ${NAMES[parts.bootstrap_state]})` : "   (uses only real rewards)",
    ),
  );
}

const sub = (k) => String(k).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);

export function mount(root) {
  const env = new Chain();
  const reference = exactPolicyValue(env.model(), uniformPolicy(7, 2), 1);
  const rng = new Rng(8);
  let { td, mc } = makeLearners();
  let episode = new Trajectory();
  let tdCounts = [];
  let mcDone = [];
  let selected = null;
  let n = 1;
  let episodes = 0;
  let envSteps = 0;
  let highlight = null;
  let [agentState] = env.reset();

  const ledger = new Ledger({
    titleFor: (t) => `${METHODS[t.algorithm].label} · learning step ${t.learn_step}`,
    groupsFor: ledgerGroups,
    empty: "No learning step yet.",
    onSelect: () => render(),
  });

  const figure = h("div", { class: "figure-scroll chain-figure" });
  const counters = h("div", { class: "counters" });
  const table = h("div", { class: "experience" });
  const inspector = h("div");
  const status = h("div", { class: "callout" });

  function push(trace, method) {
    ledger.push(trace);
    highlight = { method, state: trace.state, before: trace.value_before };
  }

  const actions = {
    step() {
      if (episode.done) return;
      const action = rng.integers(2);
      const [next_state, reward, terminated] = env.step(action);
      episode.append(makeTransition({ state: agentState, action, reward, next_state, terminated }));
      tdCounts.push(0);
      mcDone.push(false);
      agentState = next_state;
      envSteps += 1;
      selected = episode.length - 1;
      if (terminated) episodes += 1;
      render();
    },
    finish() {
      while (!episode.done) actions.step();
    },
    tdLearn(t) {
      if (t === null || t === undefined || t >= episode.length) return;
      push(td.learnStep(episode.at(t)), "td0");
      tdCounts[t] += 1;
      render();
    },
    tdNext() {
      const t = tdCounts.findIndex((c) => c === 0);
      if (t >= 0) actions.tdLearn(t);
    },
    mcNext() {
      if (!episode.terminated) return;
      const t = mc.timesteps(episode).find((i) => !mcDone[i]);
      if (t === undefined) return;
      push(mc.learnStep(episode, t), "monte_carlo");
      mcDone[t] = true;
      render();
    },
    learnAll() {
      while (tdCounts.includes(0)) actions.tdNext();
      if (episode.terminated) while (mc.timesteps(episode).some((i) => !mcDone[i])) actions.mcNext();
    },
    newEpisode() {
      episode = new Trajectory();
      tdCounts = [];
      mcDone = [];
      selected = null;
      [agentState] = env.reset();
      render();
    },
    resetAll() {
      ({ td, mc } = makeLearners());
      ledger.reset();
      highlight = null;
      episodes = 0;
      envSteps = 0;
      actions.newEpisode();
    },
  };

  const stepBtn = button("Step environment", { kind: "env", kbd: "E", onClick: actions.step });
  const finishBtn = button("Finish episode", { onClick: actions.finish });
  const newBtn = button("New episode", { onClick: actions.newEpisode });
  const tdBtn = button("TD: learn once", { kind: "learn", kbd: "T", onClick: actions.tdNext });
  const mcBtn = button("MC: learn once", { kind: "learn", kbd: "M", onClick: actions.mcNext });
  const allBtn = button("Learn from everything", { onClick: actions.learnAll });
  const nSlider = slider({
    id: "l07-n",
    label: "n-step target",
    min: 1,
    max: N_INFINITE,
    step: 1,
    value: n,
    format: (v) => (v >= N_INFINITE ? "∞ (MC)" : v === 1 ? "1 (TD)" : String(v)),
    onInput: (v) => ((n = v), render()),
  });

  const mission = new Mission({
    title: "Who can learn when?",
    goal: "Collect part of an episode, try to learn with both methods, then finish the episode and try again.",
    steps: [
      { text: "Press Step environment until the episode table has 3 rows.", done: () => episode.length >= 3 || episode.done },
      { text: "Press TD: learn once. TD updates right away. MC: learn once is greyed out.", baseline: () => td.learnSteps, done: (b) => td.learnSteps > b },
      { text: "Press Finish episode.", done: () => episode.done },
      { text: "Press MC: learn once. Now Monte Carlo can learn too.", baseline: () => mc.learnSteps, done: (b) => mc.learnSteps > b },
    ],
    conclusion:
      "TD's target, r + γV(s′), needs one reward and the current guess for the next state, so TD can learn after every step. Monte Carlo's target, the return G, adds up every reward until the end, so it has to wait for the episode to finish. The n-step slider under the table shows the targets in between.",
  });

  function render() {
    mission.update();
    replace(
      figure,
      chainValues({
        agent: episode.done ? null : agentState,
        series: [
          { id: "monte_carlo", color: METHODS.monte_carlo.color, V: mc.V },
          { id: "td0", color: METHODS.td0.color, V: td.V },
        ],
        reference,
        highlight: ledger.browsing ? null : highlight,
      }),
    );
    const mcWaiting = !episode.terminated;
    replace(
      counters,
      h("span", { class: "counter" }, h("span", { class: "dot env" }), "environment steps", h("b", {}, envSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot muted" }), "episodes", h("b", {}, episodes)),
      h("span", { class: "counter" }, h("span", { class: "dot learn" }), "TD updates", h("b", {}, td.learnSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot learn" }), "MC updates", h("b", {}, mc.learnSteps)),
    );
    stepBtn.disabled = episode.done;
    finishBtn.disabled = episode.done;
    newBtn.disabled = !episode.done;
    tdBtn.disabled = !tdCounts.includes(0);
    mcBtn.disabled = mcWaiting || !mc.timesteps(episode).some((i) => !mcDone[i]);
    mcBtn.title = mcWaiting ? "Monte Carlo needs the return G_t, which is only known when the episode ends" : "";
    allBtn.disabled = tdBtn.disabled && mcBtn.disabled;

    replace(
      table,
      h("div", { class: "ledger-head" }, h("span", { class: "dot env" }), h("span", { class: "title" }, "This episode"), h("span", { class: "grow" }), h("span", { class: "note" }, episode.done ? "ended" : "in progress")),
      episode.length
        ? h(
            "div",
            { class: "experience-scroll" },
            h(
              "table",
              {},
              h("thead", {}, h("tr", {}, ["t", "s", "a", "r", "s′", "TD", "MC"].map((c) => h("th", {}, c)))),
              h(
                "tbody",
                {},
                episode.transitions.map((tr, t) =>
                  h(
                    "tr",
                    { class: `selectable${selected === t ? " selected" : ""}`, onclick: () => ((selected = t), render()) },
                    h("td", {}, t),
                    h("td", {}, NAMES[tr.state]),
                    h("td", {}, ARROW[tr.action]),
                    h("td", {}, fmtShort(tr.reward)),
                    h("td", {}, NAMES[tr.next_state]),
                    h(
                      "td",
                      {},
                      h("button", { class: "mini-btn", type: "button", onclick: (e) => (e.stopPropagation(), actions.tdLearn(t)) }, "learn"),
                      tdCounts[t] ? h("span", { class: "note" }, ` ×${tdCounts[t]}`) : null,
                    ),
                    h("td", {}, h("span", { class: `pill ${mcDone[t] ? "learned" : mcWaiting ? "" : "pending"}` }, mcDone[t] ? "learned" : mcWaiting ? "waits for the end" : "ready")),
                  ),
                ),
              ),
            ),
          )
        : h("p", { class: "ledger-empty" }, "No transitions yet. Press Step environment."),
    );
    replace(inspector, targetInspector({ episode, t: selected, n, V: td.V }));
    replace(
      status,
      h(
        "p",
        {},
        mcWaiting
          ? [h("strong", {}, "TD can learn now; Monte Carlo cannot. "), "Every TD target needs only one reward and the current estimate of the next state. Monte Carlo's target is the full return, which is not known until the episode ends."]
          : [h("strong", {}, "The episode has ended. "), "Now Monte Carlo can learn: for every visited state, the target is the actual sum of the rewards that followed."],
      ),
    );
  }

  // ---------- many episodes: RMS error against the true values ----------
  const rmsChart = new LineChart({
    height: 250,
    xLabel: "episodes",
    yLabel: "RMS error over A–E (average of 100 runs)",
    yFormat: (v) => v.toFixed(2),
    caption: "Both methods learn from exactly the same episodes in every run. The true values come from the model and are only used to measure the error.",
  });
  const batch = { alphaTD: 0.1, alphaMC: 0.02 };
  const runBatch = () => {
    const curves = rmsExperiment(batch, reference);
    rmsChart.update({
      series: [
        { id: "monte_carlo", label: `Monte Carlo, α = ${batch.alphaMC}`, color: METHODS.monte_carlo.color, points: curves.mc },
        { id: "td0", label: `TD(0), α = ${batch.alphaTD}`, color: METHODS.td0.color, points: curves.td },
      ],
    });
  };

  // ---------- challenge: the order of replay ----------
  const challenge = replayChallenge();

  const unbind = shortcuts({ e: actions.step, t: actions.tdNext, m: actions.mcNext });

  const tdCode = extractDef(tdSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" });
  const mcCode = extractDef(mcSource, "learn_episode", { cutAt: "traces.append(", replacement: "traces.append(LearningTrace(...))" });

  root.append(
    lessonHeader("07", {
      lead: ["Monte Carlo and TD(0) both estimate ", tex("V^\\pi"), " from experience. They look at exactly the same trajectory but build different targets from it, and that decides when they can learn and how noisy the learning is."],
      concepts: ["state value", "return", "bootstrapping", "TD error", "n-step TD"],
    }),
    step("Question", h("p", { class: "question" }, "Why can TD learn before an episode ends?")),
    step(
      "Predict",
      predict({
        question: "The agent has taken three steps and the episode is still going. Which method can update a value right now?",
        choices: [{ label: "Only Monte Carlo" }, { label: "Only TD(0)" }, { label: "Both" }, { label: "Neither" }],
        answer: 1,
        explain: () =>
          "TD(0)'s target r_t + γV(s_{t+1}) needs one reward and a guess about the next state, and both exist after one step. Monte Carlo's target G_t needs every reward until the end. Try it: step three times and look at which buttons are enabled.",
      }),
    ),
    wideStep(
      "Experiment",
      prose(h("p", {}, "The agent walks left or right at random. Start in C; leaving on the right pays +1, leaving on the left pays 0. The dashed lines are the true values ", tex("V^\\pi"), ", computed from the model. Neither learner sees them.")),
      h(
        "div",
        { class: "bench" },
        mission.el,
        h(
          "div",
          { class: "toolbar" },
          h("div", { class: "toolbar-group" }, stepBtn, finishBtn, newBtn),
          h("span", { class: "spacer" }),
          h("div", { class: "toolbar-group" }, tdBtn, mcBtn, allBtn),
        ),
        counters,
        h(
          "div",
          { class: "bench-grid" },
          h(
            "div",
            { class: "figure" },
            figure,
            h(
              "div",
              { class: "key" },
              h("span", { class: "key-item" }, h("span", { class: "legend-rect", style: { background: METHODS.monte_carlo.color } }), "Monte Carlo estimate"),
              h("span", { class: "key-item" }, h("span", { class: "legend-rect", style: { background: METHODS.td0.color } }), "TD(0) estimate"),
              h("span", { class: "key-item" }, h("span", { class: "key-swatch ref" }), ["true value ", tex("V^\\pi"), " (reference)"]),
            ),
            status,
            table,
            h("div", { class: "bench inset" }, h("div", { class: "controls-row" }, nSlider, h("p", { class: "note" }, "Which rewards does the target of the selected row use?")), inspector),
          ),
          h("div", { class: "figure" }, ledger.el),
        ),
      ),
    ),
    wideStep(
      "Inspect",
      prose("One episode is noisy. Run many: both methods approach the true values, but at different speeds and with different noise."),
      h(
        "div",
        { class: "controls-row" },
        slider({ id: "l07-atd", label: "TD α", min: 0.01, max: 0.3, step: 0.01, value: batch.alphaTD, format: (v) => v.toFixed(2), onInput: (v) => ((batch.alphaTD = v), runBatch()) }),
        slider({ id: "l07-amc", label: "Monte Carlo α", min: 0.01, max: 0.3, step: 0.01, value: batch.alphaMC, format: (v) => v.toFixed(2), onInput: (v) => ((batch.alphaMC = v), runBatch()) }),
      ),
      rmsChart.el,
      prose("With a large α both curves drop fast and then stall at a noise floor, because every update chases one noisy sample. TD's targets vary less (one reward plus a guess), so on this task it usually gets lower error for the same amount of experience."),
    ),
    step(
      "Equation",
      equation("\\text{Monte Carlo:}\\quad V(s_t) \\leftarrow V(s_t) + \\alpha\\big[\\, G_t - V(s_t) \\,\\big],\\qquad G_t = r_t + \\gamma r_{t+1} + \\dots + \\gamma^{T-t-1} r_{T-1}"),
      equation("\\text{TD(0):}\\quad V(s_t) \\leftarrow V(s_t) + \\alpha\\big[\\, r_t + \\gamma V(s_{t+1}) - V(s_t) \\,\\big]"),
      equation(
        "G_{t:t+n} = r_t + \\gamma r_{t+1} + \\dots + \\gamma^{n-1} r_{t+n-1} + \\gamma^n V(s_{t+n})",
        h("span", {}, "The n-step target in between: n = 1 is TD(0), and when t + n reaches past the end of the episode it is the Monte Carlo return. Move the n slider in the experiment to see which rewards it uses."),
      ),
    ),
    step(
      "Code",
      h(
        "div",
        { class: "two-col" },
        codeBlock(tdCode, { title: "visualrl/algorithms/tabular/td.py", highlight: [lineOf(tdCode, "target = r +")] }),
        codeBlock(mcCode, { title: "visualrl/algorithms/tabular/monte_carlo.py", highlight: [lineOf(mcCode, "returns = trajectory.returns")] }),
      ),
    ),
    wideStep(
      "Challenge",
      prose(
        "Without collecting another transition, make the +1 reward reach the start state C. Below is one finished episode and a fresh TD learner (all values 0, α = 0.5). You may learn from its transitions in any order, as often as you like.",
      ),
      challenge.el,
    ),
    lessonFooter("07"),
  );
  render();
  runBatch();
  return () => unbind();
}

function rmsExperiment({ alphaTD, alphaMC }, reference, { runs = 100, episodes = 100 } = {}) {
  const td = new Array(episodes + 1).fill(0);
  const mc = new Array(episodes + 1).fill(0);
  const rms = (V) => Math.sqrt([1, 2, 3, 4, 5].reduce((sum, st) => sum + (V[st] - reference[st]) ** 2, 0) / 5);
  for (let run = 0; run < runs; run++) {
    const env = new Chain();
    const rng = new Rng(1000 + run);
    const learners = makeLearners({ alphaTD, alphaMC });
    td[0] += rms(learners.td.V);
    mc[0] += rms(learners.mc.V);
    for (let e = 1; e <= episodes; e++) {
      const episode = new Trajectory();
      let [state] = env.reset();
      while (!episode.done) {
        const action = rng.integers(2);
        const [next_state, reward, terminated] = env.step(action);
        const tr = makeTransition({ state, action, reward, next_state, terminated });
        episode.append(tr);
        learners.td.learnStep(tr);
        state = next_state;
      }
      learners.mc.learnEpisode(episode);
      td[e] += rms(learners.td.V);
      mc[e] += rms(learners.mc.V);
    }
  }
  return { td: td.map((v, e) => [e, v / runs]), mc: mc.map((v, e) => [e, v / runs]) };
}

// A fixed episode C → D → C → D → E → (+1). Forward replay moves the reward back one state per pass;
// replaying from the end moves it all the way back in one pass.
function replayChallenge() {
  const actionsTaken = [1, 0, 1, 1, 1];
  const env = new Chain();
  let [state] = env.reset();
  const episode = new Trajectory();
  for (const action of actionsTaken) {
    const [next_state, reward, terminated] = env.step(action);
    episode.append(makeTransition({ state, action, reward, next_state, terminated }));
    state = next_state;
  }
  let td;
  let used;
  let solvedAt;
  const view = h("div", { class: "figure-scroll chain-figure" });
  const list = h("div", { class: "experience" });
  const result = h("div", { class: "callout" });
  const ledger = new Ledger({ titleFor: (t) => `Replay update ${t.learn_step}`, groupsFor: ledgerGroups, empty: "Pick a transition to learn from." });

  function reset() {
    td = new TD0({ nStates: 7, alpha: 0.5, gamma: 1, initialValue: 0 });
    used = [];
    solvedAt = null;
    ledger.reset();
    render();
  }

  function learn(t) {
    const trace = td.learnStep(episode.at(t));
    used.push(t);
    ledger.push(trace);
    if (solvedAt === null && td.V[3] > 0) solvedAt = used.length;
    render();
  }

  function render() {
    const last = ledger.current;
    replace(view, chainValues({ agent: null, series: [{ id: "td0", color: METHODS.td0.color, V: td.V }], highlight: last ? { method: "td0", state: last.state, before: last.value_before } : null }));
    replace(
      list,
      h("div", { class: "ledger-head" }, h("span", { class: "dot env" }), h("span", { class: "title" }, "The finished episode"), h("span", { class: "grow" }), h("span", { class: "note" }, `order so far: ${used.length ? used.join(", ") : "none"}`)),
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ["t", "s", "a", "r", "s′", ""].map((c) => h("th", {}, c)))),
        h(
          "tbody",
          {},
          episode.transitions.map((tr, t) =>
            h(
              "tr",
              {},
              h("td", {}, t),
              h("td", {}, NAMES[tr.state]),
              h("td", {}, ARROW[tr.action]),
              h("td", {}, fmtShort(tr.reward)),
              h("td", {}, NAMES[tr.next_state]),
              h("td", {}, h("button", { class: "mini-btn", type: "button", onclick: () => learn(t) }, "learn")),
            ),
          ),
        ),
      ),
    );
    replace(
      result,
      solvedAt !== null
        ? h(
            "p",
            {},
            h("span", { class: "verdict good" }, "The reward reached C. "),
            `It took ${solvedAt} learning ${solvedAt === 1 ? "step" : "steps"}; the fewest possible is 3. `,
            solvedAt > 3 ? "Hint: each TD update moves information one state back. Start from the transition that saw the reward." : "Replaying backwards from the reward lets each update build on the one before it.",
          )
        : h("p", {}, h("span", { class: "status-line" }, h("span", { class: "dot muted" }), `V(C) = ${fmt(td.V[3])}. ${used.length ? `${used.length} learning ${used.length === 1 ? "step" : "steps"} so far.` : "No learning steps yet."}`)),
    );
  }

  reset();
  return {
    el: h(
      "div",
      { class: "bench" },
      h("div", { class: "bench-grid" }, h("div", { class: "figure" }, view, list, result, h("div", {}, button("Start over", { kind: "ghost", onClick: reset }))), h("div", { class: "figure" }, ledger.el)),
    ),
  };
}
