// Mirrors visualrl/policies.py.

// Two values closer than this count as a tie, so a fresh all-zero table
// explores every action instead of always picking action 0.
export const TIE_TOLERANCE = 1e-9;

export function greedyActions(qRow) {
  const best = Math.max(...qRow);
  const out = [];
  qRow.forEach((q, a) => {
    if (q >= best - TIE_TOLERANCE) out.push(a);
  });
  return out;
}

export function greedyProbs(qRow) {
  const best = greedyActions(qRow);
  const probs = new Array(qRow.length).fill(0);
  for (const a of best) probs[a] = 1 / best.length;
  return probs;
}

export function epsilonGreedyProbs(qRow, epsilon) {
  const n = qRow.length;
  return greedyProbs(qRow).map((p) => (1 - epsilon) * p + epsilon / n);
}

export function uniformPolicy(nStates, nActions) {
  return Array.from({ length: nStates }, () => new Array(nActions).fill(1 / nActions));
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
