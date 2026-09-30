import assert from "node:assert/strict";
import { test } from "node:test";

import { rollout } from "../src/rl/core.js";
import { Chain } from "../src/rl/envs/chain.js";
import { GridWorld } from "../src/rl/envs/gridworld.js";
import { Rng } from "../src/rl/rng.js";
import { QLearning, Sarsa } from "../src/rl/tabular/q-agents.js";
import { runEpisode } from "../src/rl/tabular/episode.js";

test("deterministic models agree with step()", () => {
  for (const env of [new Chain(), GridWorld.cliff()]) {
    const model = env.model();
    for (let s = 0; s < model.nStates; s++) {
      if (model.terminal[s]) continue;
      for (let a = 0; a < model.nActions; a++) {
        env.reset();
        env.state = s;
        const [next, reward, terminated] = env.step(a);
        assert.deepEqual(model.P[s][a], [[1, next, reward, terminated]]);
      }
    }
  }
});

test("the cliff sends the agent back and reports the fall", () => {
  const env = GridWorld.cliff();
  env.reset();
  const [next, reward, terminated, , info] = env.step(1);
  assert.deepEqual([next, reward, terminated], [env.start, -100, false]);
  assert.equal(info.fell_into, env.toState(3, 1));
});

test("rollout returns are discounted sums", () => {
  const traj = rollout(new Chain(), () => Chain.RIGHT);
  assert.deepEqual(traj.states, [3, 4, 5, 6]);
  const G = traj.returns(0.5);
  assert.deepEqual(G, [0.25, 0.5, 1]);
});

test("the seeded generator is uniform enough", () => {
  const rng = new Rng(3);
  const counts = [0, 0, 0, 0];
  for (let i = 0; i < 40000; i++) counts[rng.choice([0.25, 0.25, 0.25, 0.25])] += 1;
  for (const c of counts) assert.ok(Math.abs(c - 10000) < 400);
});

function greedyReturn(env, Q) {
  let s = env.start;
  let total = 0;
  for (let i = 0; i < 100; i++) {
    const a = Q[s].indexOf(Math.max(...Q[s]));
    const [next, r, done] = env.move(s, a);
    total += r;
    s = next;
    if (done) return total;
  }
  return -Infinity;
}

test("on the cliff, Q-learning walks the edge and SARSA keeps a margin", () => {
  const env = GridWorld.cliff();
  const opts = { nStates: 48, nActions: 4, alpha: 0.5, gamma: 1, epsilon: 0.1, seed: 5 };
  const sarsa = new Sarsa(opts);
  const qlearn = new QLearning(opts);
  for (let i = 0; i < 500; i++) {
    runEpisode(env, sarsa);
    runEpisode(env, qlearn);
  }
  assert.equal(greedyReturn(env, qlearn.Q), -13);
  const safe = greedyReturn(env, sarsa.Q);
  assert.ok(safe < -13 && safe > -30, `SARSA greedy return ${safe}`);
});

test("the warehouse has exactly the cliff dynamics", async () => {
  const { optimalValues } = await import("../src/rl/tabular/dp.js");
  const cliff = GridWorld.cliff();
  const warehouse = GridWorld.warehouse();
  const toW = (s) => warehouse.toState(...cliff.toCell(s).map((x) => x + 1));
  const Pc = cliff.model().P;
  const Pw = warehouse.model().P;
  for (let s = 0; s < cliff.nStates; s++) {
    for (let a = 0; a < 4; a++) assert.deepEqual(Pw[toW(s)][a], Pc[s][a].map(([p, s2, r, d]) => [p, toW(s2), r, d]));
  }
  assert.equal(optimalValues(warehouse.model(), 1).V[warehouse.start], -13);
});
