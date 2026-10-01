// Lesson 14: PPO in the warehouse. The policy is a table of logits and the critic a table of
// values, so every probability ratio and every clipped sample can be inspected.
import ppoSource from "../../../visualrl/algorithms/tabular/ppo.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { clippedObjective, TabularPPO } from "../rl/tabular/ppo.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, shortcuts, slider } from "../ui/dom.js";
import { ARROWS, fmt, fmtSigned } from "./ui/format.js";
import { greedyPath, valueLegend } from "./ui/grid-view.js";
import { Ledger } from "./ui/ledger.js";
import { LineChart } from "./ui/line-chart.js";
import { Mission } from "./ui/mission.js";
import { equation, tex } from "../ui/math.js";
import { PolicyGrid } from "./ui/policy-grid.js";
import { RatioScatter } from "./ui/ratio-scatter.js";
import { WarehouseScene } from "./ui/scene3d.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef, lineOf } from "../ui/source.js";
import { mujocoSection } from "./mujoco-section.js";

const HAZARD = "Loading ledge · −100, back to the dock";
const MAX_STEPS = 1000;
const REPLAY = 60;

const cellName = (env, st) => {
  const [r, c] = env.toCell(st);
  return st === env.start ? `dock (${r},${c})` : env.goals.includes(st) ? `charger (${r},${c})` : `(${r},${c})`;
};

// One PPO learner and the loop around it: collect with π_old, estimate advantages, run epochs of minibatch steps.
class PPORun {
  constructor(settings) {
    this.settings = settings;
    this.reset();
  }

  reset() {
    const st = this.settings;
    this.env = GridWorld.warehouse({ maxSteps: MAX_STEPS });
    this.agent = new TabularPPO({ nStates: this.env.nStates, nActions: 4, gamma: 0.99, lam: 0.95, clip: st.clip, policyLr: st.lr, valueLr: 2, entropyCoef: 0.01, seed: 3 });
    [this.state] = this.env.reset();
    this.rollout = null;
    this.gae = null;
    this.queue = [];
    this.epoch = 0;
    this.batchInEpoch = 0;
    this.version = 0;
    this.envSteps = 0;
    this.episodeReturn = 0;
    this.episodes = [];
    this.history = [];
  }

  get phase() {
    return this.rollout && (this.queue.length || this.epoch < this.settings.epochs) ? "update" : "collect";
  }

  collect() {
    const agent = this.agent;
    agent.clip = this.settings.clip;
    agent.policyLr = this.settings.lr;
    const { rollout, state } = agent.collect(this.env, this.settings.rollout, this.state);
    this.state = state;
    this.rollout = rollout;
    this.envSteps += rollout.length;
    const finished = [];
    for (const tr of rollout.transitions) {
      this.episodeReturn += tr.reward;
      if (tr.terminated || tr.truncated) {
        finished.push({ ret: this.episodeReturn, reached: tr.terminated });
        this.episodeReturn = 0;
      }
    }
    this.episodes.push(...finished);
    this.lastFinished = finished;
    this.gae = agent.computeAdvantages(rollout);
    this.epoch = 0;
    this.batchInEpoch = 0;
    this.queue = agent.minibatches(rollout.length, this.settings.minibatch);
    this.iterationTraces = [];
  }

  // One minibatch gradient step. Reshuffles at the start of every epoch; the rollout itself never changes.
  step() {
    if (this.phase !== "update") return null;
    if (!this.queue.length) {
      this.epoch += 1;
      this.batchInEpoch = 0;
      if (this.epoch >= this.settings.epochs) return this.finishIteration();
      this.queue = this.agent.minibatches(this.rollout.length, this.settings.minibatch);
    }
    const indices = this.queue.shift();
    this.batchInEpoch += 1;
    const trace = this.agent.updateMinibatch(this.rollout, indices, this.epoch);
    trace.iteration = this.version;
    this.iterationTraces.push(trace);
    if (!this.queue.length && this.epoch === this.settings.epochs - 1) this.finishIteration();
    return trace;
  }

  finishIteration() {
    const last = this.iterationTraces.slice(-Math.ceil(this.rollout.length / this.settings.minibatch));
    const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    this.history.push({
      version: this.version,
      envSteps: this.envSteps,
      meanReturn: mean(this.lastFinished.map((e) => e.ret)),
      reached: this.lastFinished.filter((e) => e.reached).length,
      kl: mean(last.map((t) => t.approx_kl)),
      clipFraction: mean(last.map((t) => t.clip_fraction)),
    });
    this.version += 1;
    this.epoch = this.settings.epochs;
    this.queue = [];
    return null;
  }

  epochSize() {
    return Math.ceil((this.rollout?.length ?? this.settings.rollout) / this.settings.minibatch);
  }
}

function updateGroups(env, trace, run) {
  const ratios = trace.samples.map((x) => x.ratio);
  const meanRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  let best = null;
  for (const [s, before] of Object.entries(trace.policy_before)) {
    const after = trace.policy_after[s];
    before.forEach((p, a) => {
      const d = after[a] - p;
      if (!best || Math.abs(d) > Math.abs(best.d)) best = { s: Number(s), a, p, q: after[a], d };
    });
  }
  return [
    {
      label: "Minibatch",
      tone: "env",
      rows: [
        { label: "samples from the rollout", formula: "M", value: String(trace.samples.length) },
        { label: "epoch", formula: "", value: `${trace.epoch + 1} of ${run.settings.epochs}` },
        { label: "data collected by", formula: "\\pi_{\\text{old}}", value: `policy ${trace.iteration}` },
      ],
    },
    {
      label: "Computation",
      rows: [
        { label: "mean probability ratio", formula: "\\bar r", value: fmt(meanRatio, 3) },
        { label: "share of samples clipped", formula: "", value: `${Math.round(trace.clip_fraction * 100)}%`, emph: true },
        { label: "clipped surrogate objective", formula: "L^{\\text{clip}}", value: fmt(trace.policy_objective, 3), emph: true },
        { label: "value loss", formula: "\\tfrac12(V-\\hat G)^2", value: fmt(trace.value_loss, 2) },
        { label: "policy entropy", formula: "H", value: fmt(trace.entropy, 3) },
        { label: "approximate KL(π_old ‖ π_θ)", formula: "", value: fmt(trace.approx_kl, 4) },
      ],
    },
    {
      label: "Update",
      tone: "learn",
      rows: [
        { label: "states whose policy moved", formula: "", value: String(Object.keys(trace.policy_before).length) },
        best
          ? { label: `largest change: ${cellName(env, best.s)} ${ARROWS[best.a]}`, formula: "\\pi_\\theta(a|s)", value: `${best.p.toFixed(3)} → ${best.q.toFixed(3)}`, result: true }
          : { label: "no change", formula: "", value: "", result: true },
      ],
    },
  ];
}

function sampleGroups(env, sample, rollout, clip) {
  const tr = rollout.transitions[sample.index];
  const direction = sample.advantage >= 0 ? "up" : "down";
  return [
    {
      label: "Experience",
      tone: "env",
      rows: [
        { label: "state", formula: "s", value: cellName(env, tr.state) },
        { label: "action", formula: "a", value: ARROWS[tr.action] },
        { label: "reward", formula: "r", value: fmt(tr.reward, 0) },
        { label: "advantage (GAE, normalized)", formula: "\\hat A", value: fmtSigned(sample.advantage, 3), emph: true },
        { label: "probability when it was collected", formula: "\\pi_{\\text{old}}(a|s)", value: fmt(sample.old_prob, 3) },
      ],
    },
    {
      label: "Computation",
      rows: [
        { label: "probability now", formula: "\\pi_\\theta(a|s)", value: fmt(sample.new_prob, 3) },
        { label: "probability ratio", formula: "r", value: fmt(sample.ratio, 3), emph: true },
        { label: "allowed range", formula: "[1-\\epsilon,\\,1+\\epsilon]", value: `[${fmt(1 - clip, 2)}, ${fmt(1 + clip, 2)}]` },
        { label: "unclipped term", formula: "r\\hat A", value: fmt(sample.ratio * sample.advantage, 3) },
        { label: "clipped term", formula: "\\text{clip}(r)\\,\\hat A", value: fmt(sample.clipped_ratio * sample.advantage, 3) },
      ],
    },
    {
      label: "Objective",
      tone: "learn",
      rows: [
        { label: "min of the two", formula: "L^{\\text{clip}}", value: fmt(sample.objective, 3), emph: true },
        {
          label: sample.clipped ? `gradient: zero, r is already past 1${sample.advantage >= 0 ? "+" : "−"}ε` : `gradient: pushes π_θ(a|s) ${direction}`,
          formula: "",
          value: sample.clipped ? "clipped" : "active",
          result: true,
        },
      ],
    },
  ];
}

export function mount(root) {
  const settings = { clip: 0.2, epochs: 4, lr: 6, rollout: 1024, minibatch: 256 };
  const run = new PPORun(settings);
  const env = run.env;
  let selected = null;
  let caughtClipped = null;

  const scene = new WarehouseScene(env, { robots: [{ id: "main", color: "--series-2" }], caption: "Replaying the last 60 steps of the rollout · drag to look around" });
  const policy = new PolicyGrid(env, { hazardLabel: HAZARD });
  const scatter = new RatioScatter({
    onSelect: (i) => {
      selected = i;
      render();
    },
  });
  const counters = h("div", { class: "counters", "aria-live": "polite" });
  const phases = h("div", { class: "phases", "aria-label": "Where PPO is in its loop" });
  const note = h("div", { class: "callout learn" });
  const challenge = h("div", { class: "callout" });
  const ledger = new Ledger({
    titleFor: (t) => `PPO · gradient step ${t.learn_step}`,
    groupsFor: (t) => updateGroups(env, t, run),
    empty: "No gradient step yet. Collect a rollout, then take a gradient step.",
  });
  const sampleCard = new Ledger({
    titleFor: (x) => `Sample ${x.index} of the rollout`,
    groupsFor: (x) => sampleGroups(env, x, run.rollout, settings.clip),
    empty: "Click a dot in the scatter plot to inspect one sample.",
    emptyTitle: "Sample inspector",
  });
  const curve = new LineChart({
    height: 190,
    xLabel: "policy version",
    yLabel: "average return of the episodes in each rollout",
    caption: "An episode ends at the charger or after 1000 steps. Averages below −1000 (many falls) are drawn at −1000.",
  });

  const actions = {
    collect() {
      if (run.phase !== "collect") return;
      run.collect();
      selected = null;
      sampleCard.reset();
      replayRollout();
      render();
    },
    step() {
      if (run.phase !== "update") return;
      const t = run.step();
      if (t) ledger.push(t);
      render();
    },
    epoch() {
      if (run.phase !== "update") return;
      const traces = [];
      // At an epoch boundary the next step starts a new epoch; then finish whatever is left of it.
      if (!run.queue.length) {
        const t = run.step();
        if (t) traces.push(t);
      }
      while (run.phase === "update" && run.queue.length) {
        const t = run.step();
        if (t) traces.push(t);
      }
      ledger.pushMany(traces);
      render();
    },
    iteration() {
      if (run.phase === "collect") run.collect();
      const traces = [];
      while (run.phase === "update") {
        const t = run.step();
        if (t) traces.push(t);
      }
      ledger.pushMany(traces);
      replayRollout();
      render();
    },
    train(n) {
      const traces = [];
      for (let i = 0; i < n; i++) {
        if (run.phase === "collect") run.collect();
        while (run.phase === "update") {
          const t = run.step();
          if (t) traces.push(t);
        }
      }
      ledger.pushMany(traces.slice(-40));
      selected = null;
      sampleCard.reset();
      replayRollout();
      render();
    },
    reset() {
      run.reset();
      ledger.reset();
      sampleCard.reset();
      selected = null;
      scene.stopRoutes();
      scene.place("main", run.state, 1);
      render();
    },
  };

  function replayRollout() {
    if (!run.rollout) return;
    const tr = run.rollout.transitions;
    const from = Math.max(0, tr.length - REPLAY);
    const steps = tr.slice(from).map((t, k) => ({ from: t.state, to: t.next_state, action: t.action, fellInto: run.rollout.falls?.[from + k] ?? null }));
    scene.playSteps("main", steps, { stepMs: 70 });
  }

  const collectBtn = button("Collect rollout", { kind: "env", kbd: "C", onClick: actions.collect });
  const stepBtn = button("Gradient step", { kind: "learn", kbd: "G", onClick: actions.step });
  const epochBtn = button("Finish epoch", { onClick: actions.epoch });
  const iterBtn = button("Finish iteration", { onClick: actions.iteration });

  function render() {
    const phase = run.phase;
    collectBtn.disabled = phase !== "collect";
    stepBtn.disabled = phase !== "update";
    epochBtn.disabled = phase !== "update";

    const B = run.epochSize();
    replace(
      phases,
      h("span", { class: `phase env${phase === "collect" ? " active" : ""}` }, `1 · collect ${settings.rollout} steps with π_old = policy ${run.version}`),
      h("span", { class: "loop-arrow" }, "→"),
      h("span", { class: `phase learn${phase === "update" && run.epoch === 0 && run.batchInEpoch === 0 ? " active" : ""}` }, "2 · advantages (GAE)"),
      h("span", { class: "loop-arrow" }, "→"),
      h(
        "span",
        { class: `phase learn${phase === "update" && (run.epoch > 0 || run.batchInEpoch > 0) ? " active" : ""}` },
        phase === "update" ? `3 · epoch ${Math.min(run.epoch + 1, settings.epochs)} of ${settings.epochs}, minibatch ${run.batchInEpoch} of ${B}` : `3 · ${settings.epochs} epochs of ${B} minibatches`,
      ),
      h("span", { class: "loop-arrow" }, "→"),
      h("span", { class: "phase" }, `policy ${run.version + (phase === "update" ? 1 : 0)}`),
    );

    replace(
      counters,
      h("span", { class: "counter" }, h("span", { class: "dot env" }), "environment steps", h("b", {}, run.envSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot learn" }), "gradient steps", h("b", {}, run.agent.learnSteps)),
      h("span", { class: "counter" }, h("span", { class: "dot muted" }), "episodes finished", h("b", {}, run.episodes.length)),
      h("span", { class: "counter" }, "reached the charger", h("b", {}, run.episodes.filter((e) => e.reached).length)),
    );

    const last = ledger.current;
    policy.render({
      probs: (s) => run.agent.actionProbs(s),
      V: run.agent.V,
      touched: new Set(last ? Object.keys(last.policy_before).map(Number) : []),
      selected: selected !== null && run.rollout ? run.rollout.transitions[selected].state : null,
    });

    if (run.rollout) {
      const samples = run.rollout.transitions.map((_, i) => run.agent.inspectSample(run.rollout, i));
      scatter.render({ samples, inBatch: new Set(last && last.iteration === run.version && phase === "update" ? last.samples.map((x) => x.index) : []), clip: settings.clip, selected });
      if (selected !== null) {
        const sample = samples[selected];
        sampleCard.reset();
        sampleCard.push(sample);
        if (sample.clipped && !caughtClipped) caughtClipped = sample;
        scene.markUpdated(sample ? run.rollout.transitions[selected].state : null);
      }
    } else {
      scatter.render({ samples: [], clip: settings.clip });
    }

    const clippedNow = run.rollout && phase === "update" ? run.rollout.transitions.filter((_, i) => run.agent.inspectSample(run.rollout, i).clipped).length : 0;
    replace(
      note,
      phase === "collect" && !run.rollout
        ? h("p", {}, h("strong", {}, "Start by collecting a rollout. "), `The robot acts with the current policy for ${settings.rollout} steps. Nothing is learned while it acts.`)
        : phase === "update" && run.agent.learnSteps === 0
          ? h("p", {}, h("strong", {}, "Every ratio is exactly 1 right now. "), "The policy that will be improved, π_θ, is still identical to the policy that collected the data, π_old. Take a gradient step and watch the dots spread out.")
          : phase === "update"
            ? h(
                "p",
                {},
                h("strong", {}, `${clippedNow} of ${run.rollout.length} samples are clipped. `),
                "After the first gradient step π_θ is no longer the policy that collected this data, so PPO is learning slightly off-policy. The ratio r is the importance weight between the two, and clipping stops trusting a sample once π_θ has moved too far from π_old in the direction its advantage pushes.",
              )
            : h("p", {}, h("strong", {}, `Policy ${run.version} is ready. `), "Its rollout has been used up. PPO throws the data away and collects a fresh rollout with the new policy: it is an on-policy algorithm."),
    );

    curve.update({
      series: [
        {
          id: "ret",
          label: "average return",
          color: "var(--series-2)",
          points: run.history.filter((x) => x.meanReturn !== null).map((x) => [x.version, Math.max(-1000, x.meanReturn)]),
        },
      ],
      xDomain: [0, Math.max(10, run.history.length)],
      yDomain: [-1000, 0],
    });

    mission.update();
    replace(
      challenge,
      caughtClipped
        ? h(
            "p",
            {},
            h("span", { class: "verdict good" }, "Found one. "),
            `Sample ${caughtClipped.index} has advantage ${fmtSigned(caughtClipped.advantage, 2)} and ratio ${fmt(caughtClipped.ratio, 3)}, outside [${fmt(1 - settings.clip, 2)}, ${fmt(1 + settings.clip, 2)}] on the side its advantage pushes. Its term in the objective is flat, so the next gradient steps leave it alone even though the data says the action was ${caughtClipped.advantage > 0 ? "good" : "bad"}.`,
          )
        : h("p", {}, h("span", { class: "status-line" }, h("span", { class: "dot muted" }), "Not yet. Collect a rollout, take several gradient steps, then click a hollow dot in a shaded corner. If none appears, raise the learning rate or the number of epochs.")),
    );
  }

  const unbind = shortcuts({ c: actions.collect, g: actions.step });

  const mission = new Mission({
    title: "One PPO iteration, step by step",
    goal: "Collect a batch with the current policy, then learn from it one minibatch at a time, and watch which samples PPO stops pushing.",
    steps: [
      { text: "Press Collect rollout. The robot acts for 1024 steps; nothing is learned.", done: () => run.envSteps >= 1024 },
      { text: "Press Gradient step once. Before it, every dot in the scatter plot sat at ratio 1.", baseline: () => run.agent.learnSteps, done: (b) => run.agent.learnSteps > b },
      { text: "Press Gradient step five more times. The dots spread out; hollow ones are clipped.", baseline: () => run.agent.learnSteps, done: (b) => run.agent.learnSteps - b >= 5 },
      { text: "Click a hollow dot in a shaded corner to see why its gradient is zero.", done: () => caughtClipped !== null },
      { text: "Press Train 10 iterations and watch the robot's route in the 3D view.", baseline: () => run.version, done: (b) => run.version - b >= 10 },
    ],
    conclusion: () =>
      `PPO acts first and learns afterwards, reusing one batch for several epochs. As soon as the policy moves, the ratio r = π_θ/π_old says how far it has moved for each sample; once r leaves [1 − ε, 1 + ε] in the direction the sample pushes, that sample's gradient is zero. After ${run.version} iterations the robot ${run.history.length && run.history[run.history.length - 1].reached ? "reaches the charger in its rollouts" : "is still learning to reach the charger"}.`,
  });

  const bench = h(
    "div",
    { class: "bench" },
    mission.el,
    h(
      "div",
      { class: "toolbar" },
      h("div", { class: "toolbar-group" }, collectBtn, stepBtn, epochBtn, iterBtn),
      h("span", { class: "spacer" }),
      h("div", { class: "toolbar-group" }, button("Train 10 iterations", { onClick: () => actions.train(10) }), button("Reset", { kind: "ghost", onClick: actions.reset })),
    ),
    phases,
    counters,
    scene.el,
    note,
    h(
      "div",
      { class: "bench-grid" },
      h(
        "div",
        { class: "figure" },
        h("div", { class: "panel-title" }, "Policy π_θ and critic V"),
        policy.el,
        h(
          "div",
          { class: "key" },
          h("span", { class: "key-item" }, "arrow length: π_θ(a|s)"),
          h("span", {}, valueLegend(100, "V(s)")),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch target" }), "states the last minibatch changed"),
        ),
      ),
      h("div", { class: "figure" }, ledger.el),
    ),
    h(
      "div",
      { class: "bench-grid" },
      h(
        "div",
        { class: "figure" },
        h("div", { class: "panel-title" }, "Every sample of the rollout"),
        scatter.el,
        h(
          "div",
          { class: "key" },
          h("span", { class: "key-item" }, h("span", { class: "key-swatch dot-a" }), "in the last minibatch, gradient flows"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch dot-c" }), "in the last minibatch, clipped"),
          h("span", { class: "key-item" }, h("span", { class: "key-swatch dot-o" }), "rest of the rollout"),
          h("span", { class: "key-item" }, h("span", { class: "legend-rect key-swatch region" }), "objective is flat here"),
        ),
      ),
      h("div", { class: "figure" }, sampleCard.el),
    ),
    h(
      "div",
      { class: "controls-row" },
      slider({ id: "l14-clip", label: "clip range ε", min: 0.05, max: 0.5, step: 0.05, value: settings.clip, format: (v) => v.toFixed(2), onInput: (v) => ((settings.clip = v), (run.agent.clip = v), render()) }),
      slider({ id: "l14-epochs", label: "epochs per rollout K", min: 1, max: 20, step: 1, value: settings.epochs, format: (v) => String(v), onInput: (v) => ((settings.epochs = v), render()) }),
      slider({ id: "l14-lr", label: "policy learning rate", min: 0.5, max: 30, step: 0.5, value: settings.lr, format: (v) => v.toFixed(1), onInput: (v) => ((settings.lr = v), (run.agent.policyLr = v)) }),
    ),
    curve.el,
  );

  // ---------- Inspect 1: the clipped objective of a single sample ----------
  const explorer = { A: 1, eps: 0.2 };
  const objChart = new LineChart({ height: 240, xLabel: "ratio r", yLabel: "objective for one sample", xDomain: [0, 2], xFormat: (v) => v.toFixed(1), yFormat: (v) => fmt(v, 1) });
  const objNote = h("p", { class: "note" });
  function drawObjective() {
    const rs = Array.from({ length: 201 }, (_, i) => i / 100);
    objChart.update({
      series: [
        { id: "raw", label: "r·Â (no clipping)", color: "var(--ink-3)", dashed: true, points: rs.map((r) => [r, r * explorer.A]) },
        { id: "clip", label: "min(r·Â, clip(r)·Â)", color: "var(--series-2)", points: rs.map((r) => [r, clippedObjective(r, explorer.A, explorer.eps)]) },
      ],
      markers: [
        { x: 1 - explorer.eps, label: "1−ε" },
        { x: 1 + explorer.eps, label: "1+ε" },
      ],
      yDomain: [-2.4, 2.4],
    });
    objNote.textContent =
      explorer.A > 0
        ? `With Â = ${fmtSigned(explorer.A, 1)} the action was better than expected, so the objective rewards raising its probability, but only until r = ${fmt(1 + explorer.eps, 2)}. Beyond that the solid line is flat and its gradient is zero. Lowering r is never capped: a mistake in that direction is always corrected.`
        : explorer.A < 0
          ? `With Â = ${fmtSigned(explorer.A, 1)} the action was worse than expected, so the objective rewards lowering its probability, but only until r = ${fmt(1 - explorer.eps, 2)}. Below that the solid line is flat.`
          : "With Â = 0 the sample carries no information and the objective is flat everywhere.";
  }

  // ---------- Inspect 2: aggressive settings, with and without clipping ----------
  const cmpChart = new LineChart({
    height: 240,
    xLabel: "policy version",
    yLabel: "share of the 8 robots that reached the charger in their rollout",
    yDomain: [0, 1],
    xFormat: (v) => String(Math.round(v)),
    yFormat: (v) => `${Math.round(v * 100)}%`,
    caption: "Both use 10 epochs per rollout, a policy learning rate of 30 and rollouts of 512 steps. They differ only in ε.",
  });
  const cmpNote = h("p", { class: "note" });
  let cmpTimer = null;
  const cmpBtn = button("Run 8 robots with clipping and 8 without", {
    kind: "env",
    onClick: () => runComparison(),
  });
  function runComparison() {
    clearTimeout(cmpTimer);
    cmpBtn.disabled = true;
    const iterations = 60;
    const variants = [
      { id: "clip", label: "PPO, ε = 0.2", color: "var(--series-2)", clip: 0.2 },
      { id: "none", label: "no clipping (ε = ∞)", color: "var(--series-1)", clip: 1e9 },
    ];
    const robots = variants.map((v) =>
      Array.from({ length: 8 }, (_, seed) => {
        const e = GridWorld.warehouse({ maxSteps: MAX_STEPS });
        return { env: e, agent: new TabularPPO({ nStates: e.nStates, nActions: 4, clip: v.clip, policyLr: 30, valueLr: 2, entropyCoef: 0.01, seed: 50 + seed }), state: e.reset()[0] };
      }),
    );
    const curves = variants.map(() => []);
    const maxKl = variants.map(() => 0);
    let it = 0;
    const tick = () => {
      variants.forEach((v, k) => {
        let reached = 0;
        for (const r of robots[k]) {
          const out = r.agent.collect(r.env, 512, r.state);
          r.state = out.state;
          if (out.rollout.transitions.some((t) => t.terminated)) reached += 1;
          for (const t of r.agent.update(out.rollout, { epochs: 10, minibatchSize: 128 })) maxKl[k] = Math.max(maxKl[k], t.approx_kl);
        }
        curves[k].push([it, reached / 8]);
      });
      it += 1;
      cmpChart.update({ series: variants.map((v, k) => ({ id: v.id, label: v.label, color: v.color, points: curves[k] })), xDomain: [0, iterations - 1] });
      cmpNote.textContent = `${it} of ${iterations} iterations. Largest approximate KL between consecutive policies in any minibatch so far: ${fmt(maxKl[0], 2)} with clipping, ${fmt(maxKl[1], 2)} without.`;
      if (it < iterations) cmpTimer = setTimeout(tick, 0);
      else cmpBtn.disabled = false;
    };
    tick();
  }

  const mujoco = mujocoSection();
  const evalCode = extractDef(ppoSource, "_evaluate");
  const updateCode = extractDef(ppoSource, "update_minibatch", { cutAt: "probs_before =", replacement: "...  # apply the accumulated gradients to the two tables" });

  root.append(
    lessonHeader("14", {
      lead: [
        "PPO collects a batch of experience with its current policy and then squeezes several epochs of gradient steps out of it. The probability ratio measures how far the policy has moved since the data was collected, and clipping stops the push once it has moved far enough. Here the policy is a table of logits and the critic a table of values, so every parameter is visible. A deep PPO replaces both tables by neural networks and changes nothing else.",
      ],
      concepts: ["policy gradient", "actor-critic", "GAE", "probability ratio", "clipping", "trust region", "on-policy"],
    }),
    step("Question", h("p", { class: "question" }, "How far should one batch of experience be allowed to move the policy?")),
    step(
      "Predict",
      predict({
        question: "A sample has advantage Â = +1.5: its action turned out better than expected. After a few gradient steps its ratio r = π_θ(a|s)/π_old(a|s) is already 1.3. With ε = 0.2, what does PPO's gradient do for this sample next?",
        choices: [
          { label: "Keeps raising the action's probability", detail: "the advantage is still positive" },
          { label: "Nothing: its gradient is zero", detail: "r is past 1 + ε" },
          { label: "Pushes the probability back down", detail: "towards r = 1.2" },
        ],
        answer: 1,
        explain: () =>
          "Once r > 1 + ε, the clipped term clip(r)·Â = 1.2 × 1.5 is smaller than r·Â, so the min picks the clipped term, which does not depend on θ. PPO does not pull the probability back; it just stops pushing it further on the strength of this old sample.",
      }),
    ),
    wideStep(
      "Experiment",
      prose("PPO alternates between acting and learning. Collect a rollout (the robot acts, nothing is learned), then take gradient steps on it one minibatch at a time. Watch the ratios in the scatter plot spread out, and the samples in the shaded corners stop moving."),
      bench,
    ),
    wideStep(
      "Inspect",
      prose("The objective of a single sample, as a function of its ratio. Drag the advantage and the clip range."),
      h(
        "div",
        { class: "controls-row" },
        slider({ id: "l14-a", label: "advantage Â", min: -2, max: 2, step: 0.1, value: explorer.A, format: (v) => fmtSigned(v, 1), onInput: (v) => ((explorer.A = v), drawObjective()) }),
        slider({ id: "l14-eps", label: "clip range ε", min: 0.05, max: 0.6, step: 0.05, value: explorer.eps, format: (v) => v.toFixed(2), onInput: (v) => ((explorer.eps = v), drawObjective()) }),
      ),
      objChart.el,
      objNote,
      prose(
        "Why bother? In a table, a big step only changes the states that were sampled, so PPO learns this warehouse even without clipping at gentle settings. Clipping matters when updates are aggressive, and even more when a neural network makes every update move the policy in states that were never sampled (lesson 10). Try aggressive settings with and without clipping:",
      ),
      h("div", { class: "toolbar" }, cmpBtn),
      cmpChart.el,
      cmpNote,
      prose(
        "Without clipping, one rollout can drive a probability close to zero or one in a single iteration (the KL jumps to hundreds). Many robots then lock into a policy that avoids the ledge but never reaches the charger, and with a near-deterministic policy they stop exploring. With clipping, each iteration can only move the policy a bounded amount, so most robots keep learning.",
      ),
    ),
    wideStep(
      "Scale up",
      prose(
        "The same algorithm on MuJoCo robots. The logits table becomes a neural network that outputs the mean of a Gaussian over joint torques, the value table becomes a second network, and plain gradient steps become Adam steps. The rollout, GAE, the ratio and the clipping are the lines you just stepped through.",
        "These robots were trained in Python with MuJoCo; the browser cannot run the physics, so what you see is a replay of recorded states: every frame is the position and orientation of every MuJoCo body. Compare the policy before training, part-way, and at the end.",
      ),
      mujoco.el,
    ),
    step(
      "Equation",
      equation("r_t(\\theta) = \\frac{\\pi_\\theta(a_t\\mid s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t\\mid s_t)}", "The probability ratio: 1 when nothing has changed since the data was collected."),
      equation("L^{\\text{clip}}(\\theta) = \\frac1M\\sum_t \\min\\!\\big[\\, r_t(\\theta)\\hat A_t,\\; \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\,\\hat A_t \\,\\big]", "The clipped surrogate objective, averaged over a minibatch. PPO ascends it, plus a small entropy bonus, and descends the critic's squared error."),
      equation(
        "\\hat A_t = \\delta_t + \\gamma\\lambda\\,\\hat A_{t+1},\\qquad \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)",
        h("span", {}, "Generalized advantage estimation: TD errors of the critic, summed with weight ", tex("(\\gamma\\lambda)^k"), ". λ = 0 is the one-step TD error of lesson 07, λ = 1 the Monte Carlo return minus the baseline."),
      ),
    ),
    step(
      "Code",
      prose("From visualrl/algorithms/tabular/ppo.py. The highlighted lines are the clipping."),
      codeBlock(evalCode, { title: "TabularPPO._evaluate", highlight: [lineOf(evalCode, "clipped_ratio = min"), lineOf(evalCode, '"clipped":')] }),
      codeBlock(updateCode, { title: "TabularPPO.update_minibatch", highlight: [lineOf(updateCode, "surrogate = 0.0 if clipped")] }),
    ),
    step("Challenge", prose("Find a sample PPO has stopped pushing: after a few gradient steps, click a hollow dot in one of the shaded corners of the scatter plot."), challenge),
    lessonFooter("14"),
  );

  scene.place("main", run.state, 1);
  render();
  drawObjective();
  // Start the comparison the first time it scrolls into view.
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      observer.disconnect();
      runComparison();
    }
  });
  observer.observe(cmpChart.el);
  return () => {
    observer.disconnect();
    unbind();
    clearTimeout(cmpTimer);
    scene.dispose();
    mujoco.dispose();
  };
}
