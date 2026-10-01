import { trace } from "../core.js";

// Mirrors visualrl/algorithms/tabular/{bellman,value_iteration}.py and visualrl/reference.py.
// Everything here reads the environment model, so it is planning or reference, never learning.

export function qFromV(model, V, gamma) {
  const Q = Array.from({ length: model.nStates }, () => new Array(model.nActions).fill(0));
  for (let s = 0; s < model.nStates; s++) {
    if (model.terminal[s]) continue;
    for (let a = 0; a < model.nActions; a++) {
      Q[s][a] = model.P[s][a].reduce((sum, [p, s2, r, done]) => sum + p * (r + (done ? 0 : gamma * V[s2])), 0);
    }
  }
  return Q;
}

// V* and Q* by value iteration to convergence.
export function optimalValues(model, gamma, { theta = 1e-12, maxSweeps = 100000 } = {}) {
  let V = new Array(model.nStates).fill(0);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    const Q = qFromV(model, V, gamma);
    const next = Q.map((row, s) => (model.terminal[s] ? 0 : Math.max(...row)));
    const change = Math.max(...next.map((v, s) => Math.abs(v - V[s])));
    V = next;
    if (change < theta) break;
  }
  return { V, Q: qFromV(model, V, gamma) };
}

// V^π exactly, by solving (I − γ P_π) V = r_π with Gaussian elimination.
export function exactPolicyValue(model, policy, gamma) {
  const n = model.nStates;
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const b = new Array(n).fill(0);
  for (let s = 0; s < n; s++) {
    if (model.terminal[s]) continue;
    policy[s].forEach((pa, a) => {
      for (const [p, s2, r, done] of model.P[s][a]) {
        b[s] += pa * p * r;
        if (!done) A[s][s2] -= gamma * pa * p;
      }
    });
  }
  return solve(A, b);
}

function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col || M[r][col] === 0) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// ---------- Part II: Bellman backups, value iteration, policy iteration ----------
// Mirrors visualrl/algorithms/tabular/{bellman,value_iteration,policy_iteration}.py.

// One Bellman optimality backup at `state`: every action's value with its branches, then the best.
export function optimalBackup(model, V, state, gamma) {
  const actions = [];
  for (let a = 0; a < model.nActions; a++) {
    const branches = [];
    let q = 0;
    for (const [transition_prob, next_state, reward, terminated] of model.P[state][a]) {
      const next_value = terminated ? 0 : V[next_state];
      const contribution = transition_prob * (reward + gamma * next_value);
      q += contribution;
      branches.push({ transition_prob, next_state, reward, terminated, next_value, contribution });
    }
    actions.push({ action: a, q, branches });
  }
  const best = Math.max(...actions.map((x) => x.q));
  return trace("optimal_backup", {
    state,
    discount: gamma,
    actions,
    best_actions: actions.filter((x) => x.q >= best - 1e-9).map((x) => x.action),
    value_before: V[state],
    value_after: best,
  });
}

// One Bellman expectation backup at `state` under policy[state][a].
export function bellmanBackup(model, V, policy, state, gamma) {
  const branches = [];
  let total = 0;
  policy[state].forEach((action_prob, a) => {
    for (const [transition_prob, next_state, reward, terminated] of model.P[state][a]) {
      const next_value = terminated ? 0 : V[next_state];
      const contribution = action_prob * transition_prob * (reward + gamma * next_value);
      total += contribution;
      branches.push({ action: a, action_prob, transition_prob, next_state, reward, terminated, next_value, contribution });
    }
  });
  return trace("bellman_backup", { state, discount: gamma, branches, value_before: V[state], value_after: total });
}

// π(a|s) greedy on Q; tied actions share the probability (visualrl.policies.greedy_policy).
export function greedyPolicy(Q) {
  return Q.map((row) => {
    const best = Math.max(...row);
    const ties = row.map((q) => q >= best - 1e-9);
    const n = ties.filter(Boolean).length;
    return ties.map((t) => (t ? 1 / n : 0));
  });
}

const allclose = (a, b) => a.every((x, i) => Math.abs(x - b[i]) <= 1e-8 + 1e-5 * Math.abs(b[i]));

export class ValueIteration {
  static algorithm = "value_iteration";

  constructor(model, gamma = 0.99) {
    this.model = model;
    this.gamma = gamma;
    this.V = new Array(model.nStates).fill(0);
    this.sweeps = 0;
  }

  // One synchronous sweep of Bellman optimality backups.
  sweep() {
    const before = [...this.V];
    const Q = qFromV(this.model, before, this.gamma);
    const after = Q.map((row, s) => (this.model.terminal[s] ? 0 : Math.max(...row)));
    this.V = after;
    this.sweeps += 1;
    return trace(ValueIteration.algorithm, {
      sweep: this.sweeps,
      discount: this.gamma,
      q_values: Q,
      values_before: before,
      values_after: [...after],
      max_change: Math.max(...after.map((v, s) => Math.abs(v - before[s]))),
    });
  }

  solve({ theta = 1e-10, maxSweeps = 100000 } = {}) {
    for (let i = 0; i < maxSweeps; i++) if (this.sweep().max_change < theta) break;
    return this;
  }

  get Q() {
    return qFromV(this.model, this.V, this.gamma);
  }

  get policy() {
    return greedyPolicy(this.Q);
  }
}

export class PolicyIteration {
  static algorithm = "policy_iteration";

  constructor(model, gamma = 0.99, policy = null) {
    this.model = model;
    this.gamma = gamma;
    this.V = new Array(model.nStates).fill(0);
    this.policy = policy ? policy.map((row) => [...row]) : Array.from({ length: model.nStates }, () => new Array(model.nActions).fill(1 / model.nActions));
    this.sweeps = 0;
    this.improvements = 0;
  }

  // One synchronous sweep of Bellman expectation backups: changes V, never the policy.
  evaluateStep() {
    const before = [...this.V];
    const after = [...before];
    for (let s = 0; s < this.model.nStates; s++) {
      if (!this.model.terminal[s]) after[s] = bellmanBackup(this.model, before, this.policy, s, this.gamma).value_after;
    }
    this.V = after;
    this.sweeps += 1;
    return trace("policy_evaluation", {
      sweep: this.sweeps,
      discount: this.gamma,
      values_before: before,
      values_after: [...after],
      max_change: Math.max(...after.map((v, s) => Math.abs(v - before[s]))),
    });
  }

  evaluate({ theta = 1e-8, maxSweeps = 10000 } = {}) {
    let t;
    for (let i = 0; i < maxSweeps; i++) {
      t = this.evaluateStep();
      if (t.max_change < theta) break;
    }
    return t;
  }

  // Make the policy greedy on the current values: changes the policy, never V.
  improveStep() {
    const Q = qFromV(this.model, this.V, this.gamma);
    const before = this.policy.map((row) => [...row]);
    const after = greedyPolicy(Q).map((row, s) => (this.model.terminal[s] ? [...before[s]] : row));
    const changed = [];
    for (let s = 0; s < this.model.nStates; s++) if (!allclose(before[s], after[s])) changed.push(s);
    this.policy = after;
    this.improvements += 1;
    return trace("policy_improvement", {
      improvement: this.improvements,
      q_values: Q,
      policy_before: before,
      policy_after: after.map((row) => [...row]),
      changed_states: changed,
      stable: changed.length === 0,
    });
  }
}
