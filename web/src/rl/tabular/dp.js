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
