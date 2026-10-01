// Small, executable textbook examples. Neural-network methods are discussed in
// the chapters; these labs do not claim to train DQN, PPO, or neural GCRL.
import { Rng } from "./rng.js";
import { QLearning } from "./tabular/q-agents.js";
export class BanditLab {
  constructor({ seed = 9, initialValue = 0, means = [0.35, 0.75] } = {}) {
    this.rng = new Rng(seed); this.means = means.slice();
    this.Q = [initialValue, initialValue]; this.N = [0, 0]; this.steps = 0; this.totalReward = 0; this.last = null;
  }
  choose({ mode = "epsilon", epsilon = 0.1 } = {}) {
    if (mode === "ucb") {
      const unvisited = this.N.indexOf(0);
      if (unvisited >= 0) return unvisited;
      const scores = this.Q.map((q, a) => q + Math.sqrt(2 * Math.log(this.steps) / this.N[a]));
      return scores.indexOf(Math.max(...scores));
    }
    if (this.rng.random() < epsilon) return this.rng.integers(2);
    const best = Math.max(...this.Q), ties = this.Q.flatMap((q, a) => q === best ? [a] : []);
    return ties[this.rng.integers(ties.length)];
  }
  pull(action, { constantAlpha = null } = {}) {
    const reward = this.rng.random() < this.means[action] ? 1 : 0;
    const before = this.Q[action]; this.N[action]++; this.steps++; this.totalReward += reward;
    const alpha = constantAlpha ?? 1 / this.N[action];
    this.Q[action] += alpha * (reward - before);
    return this.last = { action, reward, before, alpha, after: this.Q[action] };
  }
}
export class DynaLab {
  constructor(nStates) {
    this.agent = new QLearning({ nStates, nActions: 4, alpha: 1, gamma: 0.9, seed: 7 });
    this.model = new Map(); this.rng = new Rng(7); this.realSteps = 0; this.planningSteps = 0; this.last = null;
  }
  observe(transition) {
    this.last = this.agent.learnStep(transition);
    this.model.set(transition.state + "," + transition.action, { ...transition });
    this.realSteps++;
    return this.last;
  }
  plan(count) {
    const records = [...this.model.values()];
    if (!records.length) return;
    for (let i = 0; i < count; i++) {
      this.last = this.agent.learnStep(records[this.rng.integers(records.length)]);
      this.planningSteps++;
    }
  }
}
export const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
export function featureVectors(mode) {
  if (mode === "table") return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  if (mode === "shared") return [[1], [1], [1]];
  return [[1, 0], [0.8, 0.2], [0, 1]];
}
export class LinearValueLab {
  constructor(mode = "similar") { this.X = featureVectors(mode); this.w = this.X[0].map(() => 0); this.last = null; }
  predictions() { return this.X.map(x => dot(this.w, x)); }
  learn(state, target, alpha = 0.5) {
    const before = this.predictions(), error = target - before[state];
    this.w = this.w.map((w, i) => w + alpha * error * this.X[state][i]);
    return this.last = { state, target, before, error, after: this.predictions() };
  }
}
export const sigmoid = theta => 1 / (1 + Math.exp(-theta));
export class PolicyGradientLab {
  constructor() { this.theta = 0; this.alpha = 0.2; this.last = null; }
  get p() { return sigmoid(this.theta); }
  get value() { return (1 - this.p) * 1 + this.p * 7.29; }
  sampleUpdate(action, baseline) {
    const pBefore = this.p, thetaBefore = this.theta;
    const G = action === 1 ? 7.29 : 1, advantage = G - baseline;
    const score = action === 1 ? 1 - pBefore : -pBefore;
    this.theta += this.alpha * advantage * score;
    return this.last = { action, G, baseline, advantage, score, thetaBefore, thetaAfter: this.theta, pBefore, pAfter: this.p };
  }
  exactUpdate() {
    const pBefore = this.p;
    this.theta += this.alpha * pBefore * (1 - pBefore) * (7.29 - 1);
    this.last = null;
  }
}
export function ppoObjective(ratio, advantage, epsilon = 0.2) {
  const clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
  return Math.min(ratio * advantage, clipped * advantage);
}
export function lambdaWeights(length, lambda) {
  return Array.from({ length }, (_, i) => i === length - 1 ? lambda ** i : (1 - lambda) * lambda ** i);
}
export function traceStep(learner, transition) {
  const { state, next_state, reward, terminated } = transition;
  const delta = reward + (terminated ? 0 : learner.gamma * learner.V[next_state]) - learner.V[state];
  learner.e = learner.e.map((e, s) => learner.gamma * learner.lambda * e + (s === state ? 1 : 0));
  learner.V = learner.V.map((v, s) => v + learner.alpha * delta * learner.e[s]);
  return { delta, trace: [...learner.e] };
}
