// Mirrors visualrl/algorithms/tabular/ppo.py: PPO with a table of logits (actor) and a table of values (critic).
import { makeTransition, trace } from "../core.js";
import { Rng } from "../rng.js";

export function logSoftmax(logits) {
  const m = Math.max(...logits);
  const logTotal = Math.log(logits.reduce((sum, x) => sum + Math.exp(x - m), 0));
  return logits.map((x) => x - m - logTotal);
}

export const softmax = (logits) => logSoftmax(logits).map(Math.exp);

// The clipped surrogate for one sample, as a function of the ratio r (used by the objective explorer).
export const clippedObjective = (r, A, eps) => Math.min(r * A, Math.min(Math.max(r, 1 - eps), 1 + eps) * A);

// A batch collected with one fixed policy π_old. old_probs[i] = π_old(a_i | s_i).
export class Rollout {
  constructor(transitions = [], oldProbs = []) {
    this.transitions = transitions;
    this.old_probs = oldProbs;
    this.advantages = [];
    this.returns = [];
  }

  get length() {
    return this.transitions.length;
  }
}

export class TabularPPO {
  static algorithm = "ppo";

  constructor({ nStates, nActions, gamma = 0.99, lam = 0.95, clip = 0.2, policyLr = 2, valueLr = 2, entropyCoef = 0.01, normalizeAdvantages = true, seed = 0 }) {
    this.logits = Array.from({ length: nStates }, () => new Array(nActions).fill(0));
    this.V = new Array(nStates).fill(0);
    this.gamma = gamma;
    this.lam = lam;
    this.clip = clip;
    this.policyLr = policyLr;
    this.valueLr = valueLr;
    this.entropyCoef = entropyCoef;
    this.normalizeAdvantages = normalizeAdvantages;
    this.rng = new Rng(seed);
    this.learnSteps = 0;
  }

  // ---------- acting ----------

  actionProbs(state) {
    return softmax(this.logits[state]);
  }

  act(state) {
    return this.rng.choice(this.actionProbs(state));
  }

  // Act for nSteps with the current policy, resetting after each episode.
  collect(env, nSteps, state) {
    const rollout = new Rollout();
    for (let i = 0; i < nSteps; i++) {
      const probs = this.actionProbs(state);
      const action = this.rng.choice(probs);
      const [next_state, reward, terminated, truncated, info] = env.step(action);
      rollout.transitions.push(makeTransition({ state, action, reward, next_state, terminated, truncated }));
      rollout.old_probs.push(probs[action]);
      (rollout.falls ??= []).push(info?.fell_into ?? null);
      state = terminated || truncated ? env.reset()[0] : next_state;
    }
    return { rollout, state };
  }

  // ---------- learning ----------

  // Generalized advantage estimation, backwards through the rollout.
  computeAdvantages(rollout) {
    const T = rollout.length;
    const deltas = new Array(T).fill(0);
    let advantages = new Array(T).fill(0);
    let running = 0;
    for (let t = T - 1; t >= 0; t--) {
      const tr = rollout.transitions[t];
      const nextValue = tr.terminated ? 0 : this.V[tr.next_state];
      deltas[t] = tr.reward + this.gamma * nextValue - this.V[tr.state];
      if (tr.terminated || tr.truncated) running = 0;
      running = deltas[t] + this.gamma * this.lam * running;
      advantages[t] = running;
    }
    const returns = advantages.map((a, t) => a + this.V[rollout.transitions[t].state]);
    if (this.normalizeAdvantages && T > 1) {
      const mean = advantages.reduce((s, a) => s + a, 0) / T;
      const std = Math.sqrt(advantages.reduce((s, a) => s + (a - mean) ** 2, 0) / T);
      advantages = advantages.map((a) => (a - mean) / (std + 1e-8));
    }
    rollout.advantages = advantages;
    rollout.returns = returns;
    return trace("gae", { gamma: this.gamma, lam: this.lam, deltas, advantages, returns });
  }

  // The clipped objective of sample i under the current policy, plus what its gradient needs.
  evaluate(rollout, i) {
    const tr = rollout.transitions[i];
    const s = tr.state;
    const a = tr.action;
    const A = rollout.advantages[i];
    const logProbs = logSoftmax(this.logits[s]);
    const probs = logProbs.map(Math.exp);
    const logRatio = logProbs[a] - Math.log(rollout.old_probs[i]);
    const ratio = Math.exp(logRatio);
    const lo = 1 - this.clip;
    const hi = 1 + this.clip;
    const clippedRatio = Math.min(Math.max(ratio, lo), hi);
    const sample = {
      index: i,
      state: s,
      action: a,
      advantage: A,
      old_prob: rollout.old_probs[i],
      new_prob: probs[a],
      ratio,
      clipped_ratio: clippedRatio,
      objective: Math.min(ratio * A, clippedRatio * A),
      clipped: (A > 0 && ratio > hi) || (A < 0 && ratio < lo),
      return: rollout.returns[i],
      value: this.V[s],
    };
    return { sample, logProbs, probs, logRatio };
  }

  // Where sample i stands right now: its ratio, whether it is clipped, its objective. Changes nothing.
  inspectSample(rollout, i) {
    return this.evaluate(rollout, i).sample;
  }

  // One gradient step on the samples `indices`: ascend min(r A, clip(r) A) + c H, descend the value loss.
  updateMinibatch(rollout, indices, epoch = 0) {
    const M = indices.length;
    const nActions = this.logits[0].length;
    const policyGrad = new Map();
    const valueGrad = new Map();
    const samples = [];
    let objectiveSum = 0;
    let valueLossSum = 0;
    let klSum = 0;
    let entropySum = 0;
    let clippedCount = 0;

    for (const i of indices) {
      const { sample, logProbs, probs, logRatio } = this.evaluate(rollout, i);
      const { state: s, action: a, advantage: A, ratio, clipped } = sample;
      const entropy = -probs.reduce((sum, p, b) => sum + p * logProbs[b], 0);
      if (!policyGrad.has(s)) policyGrad.set(s, new Array(nActions).fill(0));
      const g = policyGrad.get(s);
      for (let b = 0; b < nActions; b++) {
        const surrogate = clipped ? 0 : A * ratio * ((b === a ? 1 : 0) - probs[b]);
        const bonus = -probs[b] * (logProbs[b] + entropy);
        g[b] += (surrogate + this.entropyCoef * bonus) / M;
      }
      const valueError = this.V[s] - rollout.returns[i];
      valueGrad.set(s, (valueGrad.get(s) ?? 0) + valueError / M);

      objectiveSum += sample.objective;
      valueLossSum += 0.5 * valueError ** 2;
      klSum += ratio - 1 - logRatio;
      entropySum += entropy;
      clippedCount += clipped ? 1 : 0;
      samples.push(sample);
    }

    const policyBefore = {};
    for (const s of policyGrad.keys()) policyBefore[s] = softmax(this.logits[s]);
    for (const [s, g] of policyGrad) for (let b = 0; b < nActions; b++) this.logits[s][b] += this.policyLr * g[b];
    for (const [s, g] of valueGrad) this.V[s] -= this.valueLr * g;
    this.learnSteps += 1;
    const policyAfter = {};
    for (const s of policyGrad.keys()) policyAfter[s] = softmax(this.logits[s]);

    return trace(TabularPPO.algorithm, {
      learn_step: this.learnSteps,
      epoch,
      clip: this.clip,
      samples,
      policy_objective: objectiveSum / M,
      value_loss: valueLossSum / M,
      approx_kl: klSum / M,
      entropy: entropySum / M,
      clip_fraction: clippedCount / M,
      policy_before: policyBefore,
      policy_after: policyAfter,
    });
  }

  minibatches(n, size) {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = this.rng.integers(i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    const out = [];
    for (let k = 0; k < n; k += size) out.push(order.slice(k, k + size));
    return out;
  }

  update(rollout, { epochs = 4, minibatchSize = 64 } = {}) {
    this.computeAdvantages(rollout);
    const traces = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const indices of this.minibatches(rollout.length, minibatchSize)) traces.push(this.updateMinibatch(rollout, indices, epoch));
    }
    return traces;
  }
}
