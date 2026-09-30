// Mirrors visualrl/envs/chain.py.
//
//   state:   0    1    2   ...   n   n+1
//           [T]  s1   s2   ...   sn  [T]
import { Rng } from "../rng.js";

export class Chain {
  static LEFT = 0;
  static RIGHT = 1;
  static ACTION_NAMES = ["left", "right"];

  constructor({
    nStates = 5,
    start = null,
    leftReward = 0,
    rightReward = 1,
    stepReward = 0,
    slip = 0,
    maxSteps = null,
    seed = 0,
  } = {}) {
    this.nInterior = nStates;
    this.start = start ?? Math.floor((nStates + 1) / 2);
    if (this.start < 1 || this.start > nStates) throw new Error(`start must be an interior state 1..${nStates}`);
    this.leftReward = leftReward;
    this.rightReward = rightReward;
    this.stepReward = stepReward;
    this.slip = slip;
    this.maxSteps = maxSteps;
    this.nStates = nStates + 2;
    this.nActions = 2;
    this.rng = new Rng(seed);
    this.state = null;
    this.steps = 0;
  }

  get terminalStates() {
    return [0, this.nInterior + 1];
  }

  move(state, direction) {
    const next = direction === Chain.LEFT ? state - 1 : state + 1;
    if (next === 0) return [next, this.leftReward, true];
    if (next === this.nInterior + 1) return [next, this.rightReward, true];
    return [next, this.stepReward, false];
  }

  reset() {
    this.state = this.start;
    this.steps = 0;
    return [this.state, {}];
  }

  step(action) {
    if (this.state === null) throw new Error("call reset() before step()");
    let direction = action;
    if (this.slip > 0 && this.rng.random() < this.slip) direction = 1 - direction;
    const [next, reward, terminated] = this.move(this.state, direction);
    this.state = next;
    this.steps += 1;
    const truncated = !terminated && this.maxSteps !== null && this.steps >= this.maxSteps;
    return [next, reward, terminated, truncated, {}];
  }

  // P[s][a] = [[prob, next_state, reward, terminated], ...]. Privileged: planners and references only.
  model() {
    const P = [];
    for (let s = 0; s < this.nStates; s++) {
      if (this.terminalStates.includes(s)) {
        P.push([[[1, s, 0, true]], [[1, s, 0, true]]]);
        continue;
      }
      const row = [];
      for (let a = 0; a < 2; a++) {
        const outcomes = [[1 - this.slip, ...this.move(s, a)]];
        if (this.slip > 0) outcomes.push([this.slip, ...this.move(s, 1 - a)]);
        row.push(outcomes);
      }
      P.push(row);
    }
    const terminal = Array.from({ length: this.nStates }, (_, s) => this.terminalStates.includes(s));
    return { P, terminal, nStates: this.nStates, nActions: 2 };
  }
}
