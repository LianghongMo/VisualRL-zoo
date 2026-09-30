// Mirrors visualrl/algorithms/tabular/monte_carlo.py.
//   V(s_t) ← V(s_t) + α [G_t − V(s_t)]
// G_t needs every reward until the end, so nothing is learned before the episode terminates.
import { trace } from "../core.js";

export class MonteCarlo {
  static algorithm = "monte_carlo";

  constructor({ nStates, alpha = 0.1, gamma = 1, firstVisit = false, initialValue = 0 }) {
    this.V = new Array(nStates).fill(initialValue);
    this.alpha = alpha;
    this.gamma = gamma;
    this.firstVisit = firstVisit;
    this.learnSteps = 0;
  }

  // Timesteps that get an update, in visiting order.
  timesteps(trajectory) {
    const seen = new Set();
    const out = [];
    trajectory.transitions.forEach((t, i) => {
      if (this.firstVisit && seen.has(t.state)) return;
      seen.add(t.state);
      out.push(i);
    });
    return out;
  }

  // One update, for timestep t of a terminated episode.
  learnStep(trajectory, t) {
    if (!trajectory.terminated) throw new Error("Monte Carlo needs a terminated episode: G_t is not known before the end");
    const s = trajectory.at(t).state;

    const prediction = this.V[s];
    const target = trajectory.returns(this.gamma)[t];
    const error = target - prediction;

    this.V[s] += this.alpha * error;
    this.learnSteps += 1;

    return trace(MonteCarlo.algorithm, {
      learn_step: this.learnSteps,
      timestep: t,
      state: s,
      rewards: trajectory.rewards.slice(t),
      discount: this.gamma,
      prediction,
      target,
      error,
      learning_rate: this.alpha,
      value_before: prediction,
      value_after: this.V[s],
    });
  }

  learnEpisode(trajectory) {
    return this.timesteps(trajectory).map((t) => this.learnStep(trajectory, t));
  }
}
