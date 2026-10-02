// Exact finite examples for the derivations, separate from neural training.
export function baselineStatistics(p, baseline, returns = [1, 7.29]) {
  const probabilities = [1 - p, p], scores = [-p, 1 - p];
  const samples = returns.map((G, a) => (G - baseline) * scores[a]);
  const mean = samples.reduce((sum, g, a) => sum + probabilities[a] * g, 0);
  const variance = samples.reduce((sum, g, a) => sum + probabilities[a] * (g - mean) ** 2, 0);
  const scoreMoment = scores.reduce((sum, u, a) => sum + probabilities[a] * u * u, 0);
  const optimalBaseline = scores.reduce((sum, u, a) => sum + probabilities[a] * u * u * returns[a], 0) / scoreMoment;
  return { probabilities, scores, samples, mean, variance, optimalBaseline,
    value: returns.reduce((sum, G, a) => sum + probabilities[a] * G, 0) };
}
export function generalizedAdvantages({ rewards, values, terminated, continuations, gamma = 0.9, lambda = 0.8 }) {
  if (values.length !== rewards.length + 1 || terminated.length !== rewards.length || continuations.length !== rewards.length) {
    throw Error("GAE requires a successor value and two masks for every transition");
  }
  const deltas = rewards.map((r, t) => r + gamma * (terminated[t] ? 0 : values[t + 1]) - values[t]);
  const advantages = rewards.map(() => 0);
  for (let t = rewards.length - 1; t >= 0; t--) {
    advantages[t] = deltas[t] + gamma * lambda * (continuations[t] && !terminated[t] ? advantages[t + 1] ?? 0 : 0);
  }
  return { deltas, advantages, targets: advantages.map((a, t) => a + values[t]) };
}
export function surrogateExample(enter, good, gamma = 0.9) {
  const oldEnter = 0.5, oldValue = 0.5, nextValue = 4 * good - 2;
  return { oldValue, actual: 1 - enter + gamma * enter * nextValue,
    surrogate: oldValue + (0.5 - enter) + gamma * oldEnter * nextValue,
    oldVisit: oldEnter, newVisit: enter };
}
export function logJensen(x, y, weight) {
  if (!(x > 0 && y > 0 && weight >= 0 && weight <= 1)) throw Error("Log Jensen requires positive inputs and a probability weight");
  const mean = weight * x + (1 - weight) * y;
  const logMean = Math.log(mean), meanLog = weight * Math.log(x) + (1 - weight) * Math.log(y);
  return { mean, logMean, meanLog, gap: logMean - meanLog };
}
export function entropyBound(values, probabilities, temperature) {
  if (!(temperature > 0) || values.length !== probabilities.length || Math.abs(probabilities.reduce((s, p) => s + p, 0) - 1) > 1e-9 || probabilities.some(p => p < 0)) {
    throw Error("Entropy bound requires positive temperature and a probability distribution");
  }
  const maximum = Math.max(...values), weights = values.map(q => Math.exp((q - maximum) / temperature));
  const sum = weights.reduce((s, w) => s + w, 0), optimal = weights.map(w => w / sum);
  const softValue = maximum + temperature * Math.log(sum);
  const expected = values.reduce((s, q, a) => s + probabilities[a] * q, 0);
  const entropy = -probabilities.reduce((s, p) => s + (p ? p * Math.log(p) : 0), 0);
  const kl = probabilities.reduce((s, p, a) => s + (p ? p * (Math.log(p) - (values[a] - softValue) / temperature) : 0), 0);
  const lower = expected + temperature * entropy;
  return { expected, entropy, softValue, lower, gap: softValue - lower, kl, optimal };
}
export function discountedSum(rewards, gamma) {
  return rewards.reduce((sum, r, t) => sum + gamma ** t * r, 0);
}
export function shapeRewards(rewards, potentials, gamma = 0.9) {
  if (potentials.length !== rewards.length + 1) throw Error("Shaping needs one potential per state, including the final state");
  const shaped = rewards.map((r, t) => r + gamma * potentials[t + 1] - potentials[t]);
  const originalReturn = discountedSum(rewards, gamma), shapedReturn = discountedSum(shaped, gamma);
  const boundary = gamma ** rewards.length * potentials.at(-1) - potentials[0];
  return { shaped, originalReturn, shapedReturn, boundary };
}
export function doubleDqnTarget(online, target, { reward = 1, gamma = 0.9, terminated = false } = {}) {
  const action = online.indexOf(Math.max(...online));
  return { action, dqn: reward + (terminated ? 0 : gamma * Math.max(...target)),
    double: reward + (terminated ? 0 : gamma * target[action]) };
}
