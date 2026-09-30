// Mirrors visualrl/algorithms/tabular/td.py.
//   V(s_t) ← V(s_t) + α [r_t + γ V(s_{t+1}) − V(s_t)]
import { trace } from "../core.js";

export class TD0 {
  static algorithm = "td0";

  constructor({ nStates, alpha = 0.1, gamma = 0.99, initialValue = 0 }) {
    this.V = new Array(nStates).fill(initialValue);
    this.alpha = alpha;
    this.gamma = gamma;
    this.learnSteps = 0;
  }

  learnStep(transition) {
    const { state: s, reward: r, next_state: sNext, terminated } = transition;

    const prediction = this.V[s];
    const bootstrap_value = terminated ? 0 : this.V[sNext];
    const target = r + this.gamma * bootstrap_value;
    const tdError = target - prediction;

    this.V[s] += this.alpha * tdError;
    this.learnSteps += 1;

    return trace(TD0.algorithm, {
      learn_step: this.learnSteps,
      state: s,
      reward: r,
      next_state: sNext,
      terminated,
      prediction,
      bootstrap_value,
      discount: this.gamma,
      target,
      error: tdError,
      learning_rate: this.alpha,
      value_before: prediction,
      value_after: this.V[s],
    });
  }
}
