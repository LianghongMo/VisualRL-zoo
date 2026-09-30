// Mirrors visualrl/algorithms/tabular/{q_agent,sarsa,q_learning}.py.
import { trace } from "../core.js";
import { argmax, epsilonGreedyProbs } from "../policies.js";
import { Rng } from "../rng.js";

class EpsilonGreedyQAgent {
  constructor({ nStates, nActions, alpha = 0.5, gamma = 1, epsilon = 0.1, initialValue = 0, seed = 0 }) {
    this.Q = Array.from({ length: nStates }, () => new Array(nActions).fill(initialValue));
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.rng = new Rng(seed);
    this.learnSteps = 0;
  }

  actionProbs(state) {
    return epsilonGreedyProbs(this.Q[state], this.epsilon);
  }

  act(state) {
    return this.rng.choice(this.actionProbs(state));
  }
}

// SARSA: on-policy TD control.
//   Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)],  a' = the action the agent will actually take.
export class Sarsa extends EpsilonGreedyQAgent {
  static algorithm = "sarsa";

  learnStep(transition) {
    const { state: s, action: a, reward: r, next_state: sNext, terminated, next_action: aNext } = transition;
    if (aNext === null && !terminated) throw new Error("SARSA needs next_action: the action the agent will take in next_state");

    const prediction = this.Q[s][a];
    const next_q_values = [...this.Q[sNext]]; // before the update, in case s' == s
    const bootstrap_value = terminated ? 0 : this.Q[sNext][aNext];
    const target = r + this.gamma * bootstrap_value;
    const tdError = target - prediction;

    this.Q[s][a] += this.alpha * tdError;
    this.learnSteps += 1;

    return trace(Sarsa.algorithm, {
      learn_step: this.learnSteps,
      state: s,
      action: a,
      reward: r,
      next_state: sNext,
      terminated,
      next_action: aNext,
      next_q_values,
      target_action: terminated ? null : aNext,
      prediction,
      bootstrap_value,
      discount: this.gamma,
      target,
      error: tdError,
      learning_rate: this.alpha,
      value_before: prediction,
      value_after: this.Q[s][a],
    });
  }
}

// Q-learning: off-policy TD control.
//   Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
export class QLearning extends EpsilonGreedyQAgent {
  static algorithm = "q_learning";

  learnStep(transition) {
    const { state: s, action: a, reward: r, next_state: sNext, terminated } = transition;

    const prediction = this.Q[s][a];
    const next_q_values = [...this.Q[sNext]];
    let target_action = null;
    let bootstrap_value = 0;
    if (!terminated) {
      target_action = argmax(this.Q[sNext]);
      bootstrap_value = this.Q[sNext][target_action];
    }
    const target = r + this.gamma * bootstrap_value;
    const tdError = target - prediction;

    this.Q[s][a] += this.alpha * tdError;
    this.learnSteps += 1;

    return trace(QLearning.algorithm, {
      learn_step: this.learnSteps,
      state: s,
      action: a,
      reward: r,
      next_state: sNext,
      terminated,
      next_action: transition.next_action,
      next_q_values,
      target_action,
      prediction,
      bootstrap_value,
      discount: this.gamma,
      target,
      error: tdError,
      learning_rate: this.alpha,
      value_before: prediction,
      value_after: this.Q[s][a],
    });
  }
}
