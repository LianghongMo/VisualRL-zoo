// Mirrors visualrl/algorithms/tabular/n_step_td.py.
//   G_{t:t+n} = r_t + γ r_{t+1} + ... + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n})
// n = 1 is TD(0); once t + n reaches past the end of the episode the target is the Monte Carlo return.
import { trace } from "../core.js";

export class NStepTD {
  static algorithm = "n_step_td";

  constructor({ nStates, n = 1, alpha = 0.1, gamma = 1, initialValue = 0 }) {
    if (n < 1) throw new Error("n must be at least 1");
    this.V = new Array(nStates).fill(initialValue);
    this.n = n;
    this.alpha = alpha;
    this.gamma = gamma;
    this.learnSteps = 0;
  }

  ready(trajectory, t) {
    return t + this.n <= trajectory.length || trajectory.done;
  }

  // The pieces of G_{t:t+n}. Changes nothing.
  target(trajectory, t) {
    if (!this.ready(trajectory, t)) {
      throw new Error(`the ${this.n}-step target of timestep ${t} needs rewards that have not been collected yet`);
    }
    const end = Math.min(t + this.n, trajectory.length);
    const rewards = trajectory.transitions.slice(t, end).map((tr) => tr.reward);
    const last = trajectory.at(end - 1);
    let bootstrap_state = null;
    let bootstrap_value = 0;
    let bootstrap_weight = 0;
    if (!last.terminated) {
      bootstrap_state = last.next_state;
      bootstrap_value = this.V[bootstrap_state];
      bootstrap_weight = this.gamma ** (end - t);
    }
    const discounted = rewards.reduce((sum, r, k) => sum + this.gamma ** k * r, 0);
    return { rewards, bootstrap_state, bootstrap_value, bootstrap_weight, target: discounted + bootstrap_weight * bootstrap_value };
  }

  learnStep(trajectory, t) {
    const parts = this.target(trajectory, t);
    const s = trajectory.at(t).state;
    const prediction = this.V[s];
    const target = parts.target;
    const error = target - prediction;

    this.V[s] += this.alpha * error;
    this.learnSteps += 1;

    return trace(NStepTD.algorithm, {
      learn_step: this.learnSteps,
      n: this.n,
      timestep: t,
      state: s,
      rewards: parts.rewards,
      discount: this.gamma,
      bootstrap_state: parts.bootstrap_state,
      bootstrap_value: parts.bootstrap_value,
      bootstrap_weight: parts.bootstrap_weight,
      prediction,
      target,
      error,
      learning_rate: this.alpha,
      value_before: prediction,
      value_after: this.V[s],
    });
  }
}
