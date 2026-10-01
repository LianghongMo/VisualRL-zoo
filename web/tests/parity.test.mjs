// The web lessons run a JavaScript copy of the tabular algorithms. This test
// replays the transitions recorded by scripts/export_golden_traces.py and
// checks that every JavaScript trace equals the Python one.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Trajectory, makeTransition } from "../src/rl/core.js";
import { Chain } from "../src/rl/envs/chain.js";
import { GridWorld } from "../src/rl/envs/gridworld.js";
import { uniformPolicy } from "../src/rl/policies.js";
import { exactPolicyValue, optimalValues } from "../src/rl/tabular/dp.js";
import { MonteCarlo } from "../src/rl/tabular/monte-carlo.js";
import { NStepTD } from "../src/rl/tabular/n-step-td.js";
import { QLearning, Sarsa } from "../src/rl/tabular/q-agents.js";
import { TD0 } from "../src/rl/tabular/td0.js";

const golden = JSON.parse(readFileSync(new URL("../../tests/golden/traces.json", import.meta.url)));

function assertClose(actual, expected, path = "trace") {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", `${path}: expected a number, got ${actual}`);
    assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)), `${path}: ${actual} != ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${path}: expected an array`);
    assert.equal(actual.length, expected.length, `${path}: length`);
    expected.forEach((e, i) => assertClose(actual[i], e, `${path}[${i}]`));
  } else if (expected !== null && typeof expected === "object") {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${path}: fields`);
    for (const key of Object.keys(expected)) assertClose(actual[key], expected[key], `${path}.${key}`);
  } else {
    assert.equal(actual, expected, path);
  }
}

const toTrajectory = (episode) => new Trajectory(episode.map(makeTransition));
const config = ({ n_states, n_actions, alpha, gamma, initial_value, n }) => ({
  nStates: n_states,
  nActions: n_actions,
  alpha,
  gamma,
  initialValue: initial_value ?? 0,
  n,
});

test("TD(0) traces match Python", () => {
  const g = golden.td0;
  const agent = new TD0(config(g.config));
  g.transitions.forEach((t, i) => assertClose(agent.learnStep(makeTransition(t)), g.traces[i], `td0[${i}]`));
  assertClose(agent.V, g.final, "td0.V");
});

test("Monte Carlo traces match Python", () => {
  const g = golden.monte_carlo;
  const agent = new MonteCarlo(config(g.config));
  const traces = g.episodes.flatMap((ep) => agent.learnEpisode(toTrajectory(ep)));
  assertClose(traces, g.traces, "monte_carlo");
  assertClose(agent.V, g.final, "monte_carlo.V");
});

test("n-step TD traces match Python", () => {
  const g = golden.n_step_td;
  const agent = new NStepTD(config(g.config));
  const traces = g.episodes.flatMap((ep) => {
    const trajectory = toTrajectory(ep);
    return ep.map((_, t) => agent.learnStep(trajectory, t));
  });
  assertClose(traces, g.traces, "n_step_td");
  assertClose(agent.V, g.final, "n_step_td.V");
});

for (const [name, Agent] of [
  ["sarsa", Sarsa],
  ["q_learning", QLearning],
]) {
  test(`${name} traces match Python`, () => {
    const g = golden[name];
    const agent = new Agent(config(g.config));
    g.transitions.forEach((t, i) => assertClose(agent.learnStep(makeTransition(t)), g.traces[i], `${name}[${i}]`));
    assertClose(agent.Q, g.final, `${name}.Q`);
  });
}

test("reference values match Python", () => {
  const cliff = optimalValues(GridWorld.cliff().model(), 1);
  assertClose(cliff.V, golden.cliff_optimal.V, "cliff V*");
  assertClose(cliff.Q, golden.cliff_optimal.Q, "cliff Q*");
  const walk = exactPolicyValue(new Chain().model(), uniformPolicy(7, 2), 1);
  walk.forEach((v, s) => assert.ok(Math.abs(v - golden.random_walk_exact.V[s]) < 1e-12));
});

test("PPO advantages and minibatch updates match Python", async () => {
  const { Rollout, TabularPPO } = await import("../src/rl/tabular/ppo.js");
  const g = golden.ppo;
  const c = g.config;
  const agent = new TabularPPO({
    nStates: c.n_states,
    nActions: c.n_actions,
    gamma: c.gamma,
    lam: c.lam,
    clip: c.clip,
    policyLr: c.policy_lr,
    valueLr: c.value_lr,
    entropyCoef: c.entropy_coef,
  });
  agent.logits = g.initial.logits.map((row) => [...row]);
  agent.V = [...g.initial.V];
  const rollout = new Rollout(g.transitions.map(makeTransition), [...g.old_probs]);
  assertClose(agent.computeAdvantages(rollout), g.gae, "ppo.gae");
  let k = 0;
  g.minibatches.forEach((batches, epoch) => {
    for (const indices of batches) {
      const actual = agent.updateMinibatch(rollout, indices, epoch);
      assertClose(JSON.parse(JSON.stringify(actual)), g.updates[k], `ppo.update[${k}]`);
      k += 1;
    }
  });
  assertClose(agent.logits, g.final.logits, "ppo.logits");
  assertClose(agent.V, g.final.V, "ppo.V");
});

test("PPO learns to reach the charger in the warehouse", async () => {
  const { TabularPPO } = await import("../src/rl/tabular/ppo.js");
  const env = GridWorld.warehouse({ maxSteps: 1000 });
  const agent = new TabularPPO({ nStates: env.nStates, nActions: 4, policyLr: 3, valueLr: 2, entropyCoef: 0.01, seed: 1 });
  let [state] = env.reset();
  for (let it = 0; it < 45; it++) {
    const out = agent.collect(env, 1024, state);
    state = out.state;
    agent.update(out.rollout, { epochs: 4, minibatchSize: 256 });
  }
  let s = env.start;
  let total = 0;
  let done = false;
  for (let i = 0; i < 60 && !done; i++) {
    const row = agent.logits[s];
    const [next, r, d] = env.move(s, row.indexOf(Math.max(...row)));
    total += r;
    s = next;
    done = d;
  }
  assert.ok(done && total <= -13 && total > -30, `greedy return ${total}`);
});

test("experience-graph planning and online exploration match Python", async () => {
  const { ExperienceGraph, exploreEpisode } = await import("../src/rl/tabular/experience-graph.js");
  const g = golden.experience_graph;
  const room = GridWorld.chargingRoom();
  for (const unseen of ["pessimistic", "optimistic"]) {
    const graph = new ExperienceGraph({ nStates: room.nStates, nActions: 4, gamma: 0.9, unseen });
    for (const t of g.transitions) graph.add(makeTransition(t));
    assertClose(JSON.parse(JSON.stringify(graph.sweep())), g[unseen].first_sweep, `${unseen}.first_sweep`);
    graph.plan();
    assertClose(graph.V, g[unseen].V, `${unseen}.V`);
    assert.deepEqual(Array.from({ length: room.nStates }, (_, s) => graph.act(s)), g[unseen].act);
  }
  const online = new ExperienceGraph({ nStates: room.nStates, nActions: 4, gamma: 0.9, unseen: "optimistic" });
  const lengths = Array.from({ length: 8 }, () => exploreEpisode(room, online, { maxSteps: 40 }).length);
  assert.deepEqual(lengths, g.online_episode_lengths);
  assertClose(online.V, g.online_V, "online.V");
});

test("Part II backups, value iteration and policy iteration match Python", async () => {
  const { bellmanBackup, optimalBackup, PolicyIteration, ValueIteration } = await import("../src/rl/tabular/dp.js");
  const g = golden.part2;
  for (const name of ["det", "slip"]) {
    const model = GridWorld.chargingRoom({ slip: g[name].slip }).model();
    const vi = new ValueIteration(model, 0.9);
    g[name].sweeps.forEach((expected, k) => {
      const t = vi.sweep();
      assertClose(t.values_after, expected.values_after, `${name}.sweep${k}`);
      assertClose(t.max_change, expected.max_change, `${name}.max_change${k}`);
    });
    [25, 26, 28, 33].forEach((s, i) => assertClose(JSON.parse(JSON.stringify(optimalBackup(model, vi.V, s, 0.9))), g[name].backups[i], `${name}.backup${s}`));
    const uniform = Array.from({ length: model.nStates }, () => [0.25, 0.25, 0.25, 0.25]);
    assertClose(JSON.parse(JSON.stringify(bellmanBackup(model, vi.V, uniform, 33, 0.9))), g[name].expectation, `${name}.expectation`);
  }
  const model = GridWorld.chargingRoom().model();
  const pi = new PolicyIteration(model, 0.9);
  for (const [k, expected] of g.policy_iteration.entries()) {
    assertClose(pi.evaluate({ theta: 1e-10 }).values_after, expected.evaluate, `pi.evaluate${k}`);
    const imp = pi.improveStep();
    assertClose(imp.policy_after, expected.policy_after, `pi.policy${k}`);
    assert.deepEqual(imp.changed_states, expected.changed_states);
  }
});
