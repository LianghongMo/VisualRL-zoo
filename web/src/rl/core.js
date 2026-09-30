// Mirrors visualrl/core. Traces are plain objects with the same snake_case
// fields as LearningTrace.to_dict() in Python, so tests can compare them.

export function makeTransition({
  state,
  action,
  reward,
  next_state,
  terminated = false,
  truncated = false,
  next_action = null,
}) {
  return Object.freeze({ state, action, reward, next_state, terminated, truncated, next_action });
}

export const isDone = (t) => t.terminated || t.truncated;

export class Trajectory {
  constructor(transitions = []) {
    this.transitions = [...transitions];
  }

  append(transition) {
    if (this.done) throw new Error("cannot append to a finished episode");
    this.transitions.push(transition);
  }

  get length() {
    return this.transitions.length;
  }

  at(t) {
    return this.transitions[t];
  }

  [Symbol.iterator]() {
    return this.transitions[Symbol.iterator]();
  }

  get done() {
    return this.length > 0 && isDone(this.transitions[this.length - 1]);
  }

  get terminated() {
    return this.length > 0 && this.transitions[this.length - 1].terminated;
  }

  get states() {
    if (!this.length) return [];
    return [...this.transitions.map((t) => t.state), this.transitions[this.length - 1].next_state];
  }

  get rewards() {
    return this.transitions.map((t) => t.reward);
  }

  // G_t = r_t + γ G_{t+1} for every timestep, computed backwards.
  returns(gamma) {
    const out = new Array(this.length).fill(0);
    let G = 0;
    for (let t = this.length - 1; t >= 0; t--) {
      G = this.transitions[t].reward + gamma * G;
      out[t] = G;
    }
    return out;
  }

  // The terms γ^k r_{start+k}; their sum is the return G_start.
  discountedRewards(gamma, start = 0) {
    return this.transitions.slice(start).map((t, k) => gamma ** k * t.reward);
  }
}

export function trace(algorithm, fields) {
  return { algorithm, ...fields };
}

// Run one episode with policy(state) -> action. Nothing is learned here.
export function rollout(env, policy, { maxSteps = 1000 } = {}) {
  let [state] = env.reset();
  const trajectory = new Trajectory();
  for (let step = 0; step < maxSteps; step++) {
    const action = policy(state);
    const [next_state, reward, terminated, truncatedByEnv] = env.step(action);
    const truncated = truncatedByEnv || (!terminated && step === maxSteps - 1);
    trajectory.append(makeTransition({ state, action, reward, next_state, terminated, truncated }));
    if (terminated || truncated) break;
    state = next_state;
  }
  return trajectory;
}
