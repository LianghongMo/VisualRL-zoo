import assert from "node:assert/strict";
import { test } from "node:test";
import { goalRoom, goalTaskModel, goalReference, shortestRoute, relabelEpisode, discountedFuture, balancedClassifier, waypointPlan } from "../src/rl/goal-conditioned.js";
import { QLearning } from "../src/rl/tabular/q-agents.js";
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`);

test("goal tasks change action preference and terminal reward while preserving physical transitions", () => {
  const env = goalRoom(), left = env.toState(1, 1), right = env.toState(1, 7);
  const a = goalReference(env, left), b = goalReference(env, right);
  assert.equal(a.Q[env.start].indexOf(Math.max(...a.Q[env.start])), 3);
  assert.equal(b.Q[env.start].indexOf(Math.max(...b.Q[env.start])), 1);
  close(a.V[env.start], 0.9 ** 4); close(b.V[env.start], 0.9 ** 4);
  for (const g of [left, right]) {
    const model = goalTaskModel(env, g);
    for (let s = 0; s < env.nStates; s++) {
      if (env.walls.includes(s) || s === g) continue;
      for (let action = 0; action < 4; action++) {
        const [next] = env.move(s, action), [[prob, target, reward, done]] = model.P[s][action];
        assert.deepEqual([prob, target, reward, done], [1, next, Number(next === g), next === g]);
      }
    }
    close(goalReference(env, g).V[g], 0);
  }
  assert.deepEqual(env.goals, []);
});

test("hindsight episodes preserve physics, stop at first goal, and propagate reward with actual Q-learning", () => {
  const env = goalRoom(), left = env.toState(1, 1), original = shortestRoute(env, env.start, left).map(t => ({ ...t, reward: 0, terminated: false }));
  const before = structuredClone(original);
  for (let k = 1; k <= 5; k++) {
    const episode = relabelEpisode(original, original[k - 1].next_state);
    assert.equal(episode.length, k); assert.equal(episode.at(-1).reward, 1); assert.equal(episode.at(-1).terminated, true);
    for (let i = 0; i < k; i++) assert.deepEqual([episode[i].state, episode[i].action, episode[i].next_state], [original[i].state, original[i].action, original[i].next_state]);
    const agent = new QLearning({ nStates: env.nStates, nActions: 4, alpha: 1, gamma: 0.9 });
    for (const t of episode.toReversed()) agent.learnStep(t);
    close(agent.Q[env.start][3], 0.9 ** (k - 1));
  }
  assert.deepEqual(original, before);
  assert.deepEqual(relabelEpisode(original, env.start), []);
});

test("future occupancy is normalized including absorbing tail and repeated visits", () => {
  const states = [1, 2, 3, 4, 5];
  for (const gamma of [0, 0.2, 0.9, 0.95]) {
    const p = discountedFuture(states, gamma);
    close([...p.values()].reduce((x, y) => x + y, 0), 1);
    close(p.get(5), gamma ** 4); close(p.get(1), 1 - gamma);
  }
  const revisit = discountedFuture([1, 2, 1], 0.9);
  close(revisit.get(1), 0.91); close(revisit.get(2), 0.09);
});

test("a classifier probability is a density ratio transform, not a reachability probability", () => {
  const p = 0.9 ** 4, uniform = balancedClassifier(p, 0.2), rare = balancedClassifier(p, 0.1);
  assert.ok(rare.probability > uniform.probability); assert.ok(rare.logit > uniform.logit);
  close(0.2 * uniform.odds, p); close(0.1 * Math.exp(rare.logit), p);
  assert.notEqual(uniform.probability, p);
  assert.equal(balancedClassifier(0, 0.2).probability, 0);
});

test("local waypoint search connects executable skills and loses its path without intermediate coverage", () => {
  const env = goalRoom(), goal = env.toState(1, 1), tr = shortestRoute(env, env.start, goal), w1 = tr[1].next_state, w2 = tr[3].next_state;
  const plan = waypointPlan(env, env.start, goal, [w1, w2], 2);
  assert.deepEqual(plan.path, [env.start, w1, w2, goal]); assert.equal(plan.distance, 5);
  let state = env.start, steps = 0;
  for (const target of plan.path.slice(1)) {
    const local = shortestRoute(env, state, target); assert.ok(local.length <= 2);
    for (const t of local) { [state] = env.move(state, t.action); steps++; }
  }
  assert.equal(state, goal); assert.equal(steps, 5);
  assert.equal(waypointPlan(env, env.start, goal, [w2], 2).path, null);
  assert.deepEqual(waypointPlan(env, env.start, goal, [], 5).path, [env.start, goal]);
});
