import { GridWorld } from "./envs/gridworld.js";
import { optimalValues } from "./tabular/dp.js";

// A goal-free physical world: task rewards/termination are supplied separately.
export const goalRoom = () => new GridWorld([
  "#########", "#...#...#", "#...#...#", "#...S...#", "#...#...#", "#...#...#", "#########",
], { stepReward: 0 });

export function goalTaskModel(env, goal) {
  if (env.walls.includes(goal)) throw new Error("a wall cannot be a goal");
  const model = env.model();
  model.terminal[goal] = true;
  model.P = model.P.map((actions, s) => actions.map((branches) => model.terminal[s]
    ? [[1, s, 0, true]]
    : branches.map(([p, next]) => [p, next, Number(next === goal), next === goal])));
  return model;
}
export const goalReference = (env, goal, gamma = 0.9) => optimalValues(goalTaskModel(env, goal), gamma);

// Exact model distances are an explicitly labelled teaching reference, not a learned SoRB critic.
export function shortestRoute(env, start, goal) {
  if (start === goal) return [];
  const queue = [start], parent = new Map([[start, null]]);
  for (let i = 0; i < queue.length; i++) {
    const s = queue[i];
    for (let action = 0; action < env.nActions; action++) {
      const [next] = env.move(s, action);
      if (parent.has(next)) continue;
      parent.set(next, { state: s, action, next_state: next });
      if (next === goal) {
        const path = []; let cursor = goal;
        while (parent.get(cursor)) { const t = parent.get(cursor); path.unshift(t); cursor = t.state; }
        return path;
      }
      queue.push(next);
    }
  }
  return null;
}

// Preserve observed physics; recompute the episodic task and stop at its first success.
export function relabelEpisode(transitions, goal) {
  if (transitions[0]?.state === goal) return [];
  const out = [];
  for (const t of transitions) {
    const terminated = t.next_state === goal;
    out.push({ ...t, reward: Number(terminated), terminated, truncated: terminated ? false : Boolean(t.truncated) });
    if (terminated) break;
  }
  return out;
}

// T ~ Geom(1-gamma) on {1,2,...}; the last observed state is assumed absorbing.
export function discountedFuture(nextStates, gamma) {
  if (!nextStates.length || gamma < 0 || gamma >= 1) throw new Error("need future states and 0 <= gamma < 1");
  const p = new Map();
  nextStates.forEach((s, i) => {
    const mass = i === nextStates.length - 1 ? gamma ** i : (1 - gamma) * gamma ** i;
    p.set(s, (p.get(s) ?? 0) + mass);
  });
  return p;
}
export function balancedClassifier(p, q) {
  if (p < 0 || q <= 0) throw new Error("need p >= 0 and q > 0");
  return { probability: p / (p + q), odds: p / q, logit: p === 0 ? -Infinity : Math.log(p / q) };
}

// A directed local-reachability graph on replay waypoints, with an actual shortest-path search.
export function waypointPlan(env, start, goal, buffer, maxSteps) {
  const nodes = [...new Set([start, ...buffer, goal])], edges = [];
  for (const from of nodes) for (const to of nodes) {
    if (from === to) continue;
    const tr = shortestRoute(env, from, to);
    if (tr && tr.length <= maxSteps) edges.push({ from, to, distance: tr.length });
  }
  const distance = new Map(nodes.map(s => [s, Infinity])), parent = new Map(), remaining = new Set(nodes);
  distance.set(start, 0);
  while (remaining.size) {
    const current = [...remaining].reduce((a, b) => distance.get(a) <= distance.get(b) ? a : b);
    if (!Number.isFinite(distance.get(current)) || current === goal) break;
    remaining.delete(current);
    for (const e of edges.filter(e => e.from === current && remaining.has(e.to))) {
      const d = distance.get(current) + e.distance;
      if (d < distance.get(e.to)) { distance.set(e.to, d); parent.set(e.to, current); }
    }
  }
  if (!Number.isFinite(distance.get(goal))) return { path: null, edges, distance: Infinity };
  const path = [goal];
  while (path[0] !== start) path.unshift(parent.get(path[0]));
  return { path, edges, distance: distance.get(goal) };
}
