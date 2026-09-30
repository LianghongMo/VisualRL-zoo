// Mirrors visualrl/algorithms/tabular/experience_graph.py: what an agent knows about a deterministic
// world, as a graph of the transitions it has seen, and Bellman backups along those edges.
import { makeTransition, trace } from "../core.js";

export const PESSIMISTIC = "pessimistic";
export const OPTIMISTIC = "optimistic";

export class ExperienceGraph {
  static algorithm = "experience_graph";

  constructor({ nStates, nActions, gamma = 0.9, unseen = PESSIMISTIC, optimisticValue = 10 }) {
    this.nStates = nStates;
    this.nActions = nActions;
    this.gamma = gamma;
    this.unseen = unseen;
    this.optimisticValue = optimisticValue;
    this.edges = new Map(); // "s,a" -> [next_state, reward, terminated]
    this.visited = new Set();
    this.terminal = new Set();
    this.V = new Array(nStates).fill(0);
    this.sweeps = 0;
  }

  // ---------- growing the graph (acting) ----------

  add(transition) {
    const { state: s, action: a } = transition;
    this.visited.add(s);
    if (transition.terminated) this.terminal.add(transition.next_state);
    else this.visited.add(transition.next_state);
    const key = `${s},${a}`;
    const isNew = !this.edges.has(key);
    this.edges.set(key, [transition.next_state, transition.reward, transition.terminated]);
    return isNew;
  }

  edge(s, a) {
    return this.edges.get(`${s},${a}`) ?? null;
  }

  untried(s) {
    const out = [];
    for (let a = 0; a < this.nActions; a++) if (!this.edges.has(`${s},${a}`)) out.push(a);
    return out;
  }

  frontier() {
    return [...this.visited].filter((s) => !this.terminal.has(s) && this.untried(s).length).sort((x, y) => x - y);
  }

  // ---------- computing on the graph (learning) ----------

  q(s, a, V = this.V) {
    const e = this.edge(s, a);
    if (!e) return this.unseen === OPTIMISTIC ? this.optimisticValue : null;
    const [s2, r, done] = e;
    return r + (done ? 0 : this.gamma * V[s2]);
  }

  bestActions(s) {
    const known = [];
    for (let a = 0; a < this.nActions; a++) {
      const v = this.q(s, a);
      if (v !== null) known.push([a, v]);
    }
    if (!known.length) return Array.from({ length: this.nActions }, (_, a) => a);
    const best = Math.max(...known.map(([, v]) => v));
    return known.filter(([, v]) => v >= best - 1e-9).map(([a]) => a);
  }

  // Greedy; ties go to the first action in the order up, right, down, left.
  act(s) {
    return this.bestActions(s)[0];
  }

  sweep() {
    const before = [...this.V];
    const after = [...before];
    const changed = [];
    for (const s of [...this.visited].sort((x, y) => x - y)) {
      if (this.terminal.has(s)) continue;
      let best = null;
      for (let a = 0; a < this.nActions; a++) {
        const v = this.q(s, a, before);
        if (v !== null && (best === null || v > best[0])) best = [v, a];
      }
      if (!best) continue;
      after[s] = best[0];
      if (Math.abs(best[0] - before[s]) > 1e-12) {
        const e = this.edge(s, best[1]);
        changed.push({ state: s, before: before[s], after: best[0], action: best[1], next_state: e ? e[0] : null });
      }
    }
    this.V = after;
    this.sweeps += 1;
    return trace(ExperienceGraph.algorithm, {
      sweep: this.sweeps,
      unseen: this.unseen,
      discount: this.gamma,
      changed,
      values_before: before,
      values_after: [...after],
      max_change: changed.reduce((m, c) => Math.max(m, Math.abs(c.after - c.before)), 0),
    });
  }

  plan({ theta = 1e-10, maxSweeps = 10000 } = {}) {
    for (let i = 0; i < maxSweeps; i++) if (this.sweep().max_change < theta) break;
    return this.V;
  }
}

// One online episode: greedy on the graph (random with probability epsilon), add each transition, replan after every step.
export function exploreEpisode(env, graph, { epsilon = 0, rng = null, maxSteps = 50 } = {}) {
  let [state] = env.reset();
  const episode = [];
  for (let i = 0; i < maxSteps; i++) {
    const action = epsilon && rng && rng.random() < epsilon ? rng.integers(graph.nActions) : graph.act(state);
    const [next_state, reward, terminated, truncated, info] = env.step(action);
    const tr = makeTransition({ state, action, reward, next_state, terminated, truncated });
    graph.add(tr);
    graph.plan();
    episode.push({ ...tr, fell_into: info?.fell_into ?? null });
    if (terminated || truncated) break;
    state = next_state;
  }
  return episode;
}
