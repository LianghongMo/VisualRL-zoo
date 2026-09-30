// Lesson 02: reward, return, and discounting. A near small reward vs a far large one.
import trajectorySource from "../../../visualrl/core/trajectory.py";
import { rollout } from "../rl/core.js";
import { Chain } from "../rl/envs/chain.js";
import { codeBlock } from "../ui/code.js";
import { button, h, replace, s, slider } from "../ui/dom.js";
import { fmt, fmtShort } from "../ui/format.js";
import { LineChart } from "../ui/line-chart.js";
import { equation, tex } from "../ui/math.js";
import { lessonFooter, lessonHeader, predict, prose, step, wideStep } from "../ui/shell.js";
import { extractDef } from "../ui/source.js";

const NEAR = { reward: 1, distance: 2 };
const ROUTES = {
  near: { label: "Left route", color: "var(--series-1)" },
  far: { label: "Right route", color: "var(--series-2)" },
};

// Build the chain for the current settings and roll out "always left" and "always right".
function simulate({ farReward, farDistance, stepReward }) {
  const env = new Chain({
    nStates: NEAR.distance + farDistance - 1,
    start: NEAR.distance,
    leftReward: NEAR.reward,
    rightReward: farReward,
    stepReward,
  });
  return {
    env,
    near: rollout(env, () => Chain.LEFT),
    far: rollout(env, () => Chain.RIGHT),
  };
}

// γ values in [0, 1] where the two returns are equal, by linear interpolation on a fine grid.
function crossings(sim) {
  const out = [];
  let prev = null;
  for (let i = 0; i <= 1000; i++) {
    const g = i / 1000;
    const d = sim.near.returns(g)[0] - sim.far.returns(g)[0];
    if (prev && prev.d !== 0 && d !== 0 && Math.sign(d) !== Math.sign(prev.d)) {
      out.push(prev.g + ((g - prev.g) * Math.abs(prev.d)) / (Math.abs(prev.d) + Math.abs(d)));
    }
    prev = { g, d };
  }
  return out;
}

function chainStrip(sim, gamma) {
  const { env } = sim;
  const n = env.nStates;
  const W = 46;
  const width = n * W + 12;
  const top = 38;
  const cellH = 40;
  const height = top + cellH + 40;
  const Gn = sim.near.returns(gamma)[0];
  const Gf = sim.far.returns(gamma)[0];
  const winner = Gn > Gf + 1e-12 ? "near" : Gf > Gn + 1e-12 ? "far" : null;
  const x = (st) => 6 + st * W;
  const cells = [];
  for (let st = 0; st < n; st++) {
    const terminal = st === 0 || st === n - 1;
    cells.push(s("rect", { x: x(st) + 1, y: top, width: W - 2, height: cellH, rx: 5, class: terminal ? "chain-terminal" : "chain-cell" }));
    const label = st === env.start ? "S" : st === 0 ? `+${fmtShort(env.leftReward)}` : st === n - 1 ? `+${fmtShort(env.rightReward)}` : "";
    if (label) cells.push(s("text", { x: x(st) + W / 2, y: top + cellH / 2 + 5, "text-anchor": "middle", class: st === env.start ? "cell-mark" : "chain-reward" }, label));
  }
  const cx = (st) => x(st) + W / 2;
  const routeLine = (key, from, to, y, above) => {
    const strong = winner === key;
    const color = ROUTES[key].color;
    const dir = Math.sign(to - from);
    const tipX = cx(to) - dir * 8;
    return s(
      "g",
      { opacity: winner && !strong ? 0.45 : 1 },
      s("line", { x1: cx(from), y1: y, x2: tipX, y2: y, stroke: color, "stroke-width": strong ? 3.5 : 2, "stroke-linecap": "round" }),
      s("polygon", { points: `${tipX + dir * 8},${y} ${tipX},${y - 5} ${tipX},${y + 5}`, fill: color }),
      s(
        "text",
        { x: (cx(from) + cx(to)) / 2, y: above ? y - 9 : y + 19, "text-anchor": "middle", class: "chain-route-label" },
        `G₀ = ${fmt(key === "near" ? Gn : Gf)}${strong ? "  preferred" : ""}`,
      ),
    );
  };
  return s(
    "svg",
    { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Chain with a near and a far reward", class: "chain-strip", style: { width: `${Math.round(width * 1.25)}px` } },
    cells,
    routeLine("near", env.start, 0, top - 10, true),
    routeLine("far", env.start, n - 1, top + cellH + 12, false),
  );
}

// A row per quantity and a column per timestep: the whole return, written out.
function returnTable(key, trajectory, gamma, maxAbs) {
  const T = trajectory.length;
  const terms = trajectory.discountedRewards(gamma);
  const G = trajectory.returns(gamma)[0];
  const col = (t) => [
    h("td", {}, t),
    h("td", {}, fmtShort(trajectory.at(t).reward)),
    h("td", {}, fmt(gamma ** t, 3)),
    h(
      "td",
      {},
      h(
        "span",
        { class: "term" },
        h("span", { class: "term-bar" }, h("span", { class: "term-ghost", style: { width: `${(100 * Math.abs(trajectory.at(t).reward)) / maxAbs}%` } }), h("span", { class: "term-fill", style: { width: `${(100 * Math.abs(terms[t])) / maxAbs}%`, background: ROUTES[key].color } })),
        fmt(terms[t], 3),
      ),
    ),
  ];
  const columns = Array.from({ length: T }, (_, t) => col(t));
  const row = (label, i) => h("tr", {}, h("th", { scope: "row" }, label), columns.map((c) => c[i]));
  return h(
    "div",
    { class: "return-table" },
    h(
      "div",
      { class: "return-head" },
      h("span", { class: "legend-line", style: { color: ROUTES[key].color } }),
      h("strong", {}, ROUTES[key].label),
      h("span", { class: "note" }, `${T} steps`),
      h("span", { class: "grow" }),
      h("span", { class: "num" }, `G₀ = ${fmt(G, 3)}`),
    ),
    h(
      "div",
      { class: "experience-scroll" },
      h(
        "table",
        {},
        h("tbody", {}, row(tex("t"), 0), row(tex("r_t"), 1), row(tex("\\gamma^t"), 2), row(tex("\\gamma^t r_t"), 3)),
      ),
    ),
  );
}

export function mount(root) {
  const settings = { gamma: 0.9, farReward: 10, farDistance: 6, stepReward: 0 };
  const strip = h("div", { class: "figure-scroll chain-figure" });
  const tables = h("div", { class: "two-col" });
  const verdict = h("div", { class: "callout" });
  const challenge = h("div", { class: "callout" });
  const chart = new LineChart({
    height: 250,
    xLabel: "discount γ",
    yLabel: "return from S",
    xDomain: [0, 1],
    xFormat: (v) => v.toFixed(1),
    yFormat: (v) => fmtShort(v, 1),
    caption: "Each point is the return of the same two trajectories, recomputed with a different γ.",
  });
  let solved = false;

  const gammaSlider = slider({ id: "l02-gamma", label: "discount γ", min: 0, max: 1, step: 0.005, value: settings.gamma, format: (v) => v.toFixed(3), onInput: (v) => ((settings.gamma = v), render()) });
  const farRewardSlider = slider({ id: "l02-far", label: "far reward", min: 1, max: 20, step: 0.5, value: settings.farReward, format: (v) => `+${fmtShort(v)}`, onInput: (v) => ((settings.farReward = v), render()) });
  const farDistanceSlider = slider({ id: "l02-dist", label: "far reward is this many steps away", min: 3, max: 12, step: 1, value: settings.farDistance, format: (v) => String(v), onInput: (v) => ((settings.farDistance = v), render()) });
  const stepSlider = slider({ id: "l02-step", label: "reward for every other step", min: -1, max: 0.5, step: 0.05, value: settings.stepReward, format: (v) => fmtShort(v, 2), onInput: (v) => ((settings.stepReward = v), render()) });

  function render() {
    const sim = simulate(settings);
    const { gamma } = settings;
    const Gn = sim.near.returns(gamma)[0];
    const Gf = sim.far.returns(gamma)[0];
    replace(strip, chainStrip(sim, gamma));
    const maxAbs = Math.max(1e-9, ...sim.near.rewards.map(Math.abs), ...sim.far.rewards.map(Math.abs));
    replace(tables, returnTable("near", sim.near, gamma, maxAbs), returnTable("far", sim.far, gamma, maxAbs));

    const cross = crossings(sim);
    const grid = Array.from({ length: 101 }, (_, i) => i / 100);
    chart.update({
      series: [
        { id: "near", label: `${ROUTES.near.label}: +${fmtShort(NEAR.reward)} in ${NEAR.distance} steps`, color: ROUTES.near.color, points: grid.map((g) => [g, sim.near.returns(g)[0]]) },
        { id: "far", label: `${ROUTES.far.label}: +${fmtShort(settings.farReward)} in ${settings.farDistance} steps`, color: ROUTES.far.color, points: grid.map((g) => [g, sim.far.returns(g)[0]]) },
      ],
      markers: [{ x: gamma, label: `γ = ${gamma.toFixed(3)}` }],
      points: cross.map((g) => ({ x: g, y: sim.near.returns(g)[0], label: `equal at γ ≈ ${g.toFixed(3)}`, color: "var(--ink)" })),
    });

    const diff = Gn - Gf;
    replace(
      verdict,
      h(
        "p",
        {},
        Math.abs(diff) < 1e-9
          ? h("strong", {}, "Both routes are worth exactly the same. ")
          : h("strong", {}, `The ${diff > 0 ? "left" : "right"} route is worth more: ${fmt(Math.max(Gn, Gf))} vs ${fmt(Math.min(Gn, Gf))}. `),
        `A reward k steps in the future is multiplied by γ^k = ${gamma.toFixed(3)}^k before it is added up.`,
      ),
    );

    if (Math.abs(diff) <= 0.015 && cross.length) solved = true;
    const gap = settings.farDistance - NEAR.distance;
    const gammaStar = (NEAR.reward / settings.farReward) ** (1 / gap);
    replace(
      challenge,
      solved
        ? h(
            "p",
            {},
            h("span", { class: "verdict good" }, "Solved. "),
            cross.length ? `With the current rewards the routes are worth the same at γ ≈ ${cross.map((g) => g.toFixed(3)).join(" and ")}. ` : "",
            settings.stepReward
              ? "With a reward on every step there is no short formula; the crossing is found numerically."
              : [
                  "Setting ",
                  tex(`\\gamma^{${NEAR.distance - 1}}\\cdot ${fmtShort(NEAR.reward)} = \\gamma^{${settings.farDistance - 1}}\\cdot ${fmtShort(settings.farReward)}`),
                  " gives ",
                  tex(`\\gamma = (${fmtShort(NEAR.reward)}/${fmtShort(settings.farReward)})^{1/${gap}} = ${gammaStar.toFixed(3)}`),
                  ".",
                ],
          )
        : h(
            "p",
            {},
            h("span", { class: "status-line" }, h("span", { class: "dot muted" }), cross.length ? `Current gap: ${fmt(Math.abs(diff))}. Get it below 0.015.` : "With these settings no γ between 0 and 1 makes the routes equal. Change the far reward or its distance."),
          ),
    );
  }

  const returnsCode = extractDef(trajectorySource, "returns");

  root.append(
    lessonHeader("02", {
      lead: "An agent does not maximize the next reward. It maximizes the return: the sum of all future rewards, each shrunk by the discount factor γ once per step of delay.",
      concepts: ["reward", "return", "discount factor γ", "episode"],
    }),
    step("Question", h("p", { class: "question" }, "Is a small reward close by worth more than a big reward far away?")),
    step(
      "Predict",
      predict({
        question: "From S, the left end pays +1 after 2 steps and the right end pays +10 after 6 steps. With γ = 0.5, which route has the larger return?",
        choices: [
          { label: "The left route", detail: "+1, two steps away" },
          { label: "The right route", detail: "+10, six steps away" },
        ],
        answer: (() => {
          const sim = simulate({ farReward: 10, farDistance: 6, stepReward: 0 });
          return sim.near.returns(0.5)[0] > sim.far.returns(0.5)[0] ? 0 : 1;
        })(),
        explain: () => {
          const sim = simulate({ farReward: 10, farDistance: 6, stepReward: 0 });
          return `With γ = 0.5 the +1 arrives on the second step and counts 0.5 × 1 = ${fmt(sim.near.returns(0.5)[0])}. The +10 arrives on the sixth step and counts 0.5⁵ × 10 = ${fmt(sim.far.returns(0.5)[0])}. Ten times the reward does not make up for four extra halvings.`;
        },
      }),
      h("div", {}, button("Set γ = 0.5 in the experiment", { onClick: () => (gammaSlider.set(0.5), (settings.gamma = 0.5), render()) })),
    ),
    wideStep(
      "Experiment",
      prose("Both trajectories below are real rollouts in a Chain environment. Change γ and the rewards: the rollouts stay the same, only how their rewards are added up changes."),
      h(
        "div",
        { class: "bench" },
        h("div", { class: "controls-row" }, gammaSlider, farRewardSlider, farDistanceSlider, stepSlider),
        strip,
        verdict,
        tables,
      ),
    ),
    step(
      "Inspect",
      prose("The preferred route can flip as γ changes. A small γ makes the agent short-sighted; γ close to 1 makes it patient."),
      chart.el,
    ),
    step(
      "Equation",
      equation("G_t \\;=\\; r_t + \\gamma r_{t+1} + \\gamma^2 r_{t+2} + \\dots \\;=\\; \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k}"),
      equation(
        "G_t \\;=\\; r_t + \\gamma\\, G_{t+1}",
        "The same sum, written recursively. This one line is what the code computes, walking backwards from the end of the episode.",
      ),
    ),
    step("Code", codeBlock(returnsCode, { title: "visualrl/core/trajectory.py", highlight: [returnsCode.split("\n").findIndex((l) => l.includes("G = self.transitions[t].reward")) + 1] })),
    step(
      "Challenge",
      prose("Find the discount factor at which the agent cannot decide: move γ until both routes are worth the same, within 0.015. Try it for a few different far rewards and distances."),
      challenge,
    ),
    lessonFooter("02"),
  );
  render();
  return () => {};
}
