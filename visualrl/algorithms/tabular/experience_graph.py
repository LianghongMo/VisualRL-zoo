from __future__ import annotations

from visualrl.core import LearningTrace, Transition

PESSIMISTIC = "pessimistic"
OPTIMISTIC = "optimistic"


class ExperienceGraph:
    """What an agent knows about a deterministic world: the transitions it has seen, as a graph.

    Nodes are states. An edge s --a--> s' exists once the agent has tried action a
    in state s, and it carries the reward r that came with it. Learning is computation
    on this graph: a Bellman backup moves value backwards along known edges,

        V(s) ← max_a [ r(s,a) + γ V(s') ]        (V(s') = 0 if the move ended the episode)

    Moves the agent has never tried are not in the graph, and something has to be
    assumed about them:
      - "pessimistic": leave them out; only trust what has been seen.
      - "optimistic": assume each one is worth `optimistic_value`, as if it led
        straight to the best reward there is.
    """

    name = "experience_graph"

    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.9, unseen: str = PESSIMISTIC, optimistic_value: float = 10.0):
        if unseen not in (PESSIMISTIC, OPTIMISTIC):
            raise ValueError(f"unseen must be {PESSIMISTIC!r} or {OPTIMISTIC!r}")
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.unseen = unseen
        self.optimistic_value = optimistic_value
        self.edges: dict[tuple[int, int], tuple[int, float, bool]] = {}
        self.visited: set[int] = set()
        self.terminal: set[int] = set()
        self.V = [0.0] * n_states
        self.sweeps = 0

    # ---------- growing the graph (acting) ----------

    def add(self, transition: Transition) -> bool:
        """Record one observed transition. Returns True if it was a move never tried before."""
        s, a = transition.state, transition.action
        self.visited.add(s)
        if transition.terminated:
            self.terminal.add(transition.next_state)
        else:
            self.visited.add(transition.next_state)
        new = (s, a) not in self.edges
        self.edges[(s, a)] = (transition.next_state, float(transition.reward), bool(transition.terminated))
        return new

    def untried(self, s: int) -> list[int]:
        return [a for a in range(self.n_actions) if (s, a) not in self.edges]

    def frontier(self) -> list[int]:
        """Visited states that still have moves nobody has tried."""
        return sorted(s for s in self.visited if s not in self.terminal and self.untried(s))

    # ---------- computing on the graph (learning) ----------

    def q(self, s: int, a: int, V=None) -> float | None:
        """The value of move a in s according to the graph; None if it is unknown and ignored."""
        V = self.V if V is None else V
        edge = self.edges.get((s, a))
        if edge is None:
            return self.optimistic_value if self.unseen == OPTIMISTIC else None
        s2, r, done = edge
        return r + (0.0 if done else self.gamma * V[s2])

    def best_actions(self, s: int) -> list[int]:
        """The moves with the highest value. With nothing to go on, every move ties."""
        values = [(a, self.q(s, a)) for a in range(self.n_actions)]
        known = [(a, v) for a, v in values if v is not None]
        if not known:
            return list(range(self.n_actions))
        best = max(v for _, v in known)
        return [a for a, v in known if v >= best - 1e-9]

    def act(self, s: int) -> int:
        """The greedy move; ties go to the first action in the order up, right, down, left."""
        return self.best_actions(s)[0]

    def sweep(self) -> LearningTrace:
        """One synchronous Bellman backup of every visited state: value moves one edge further back."""
        before = list(self.V)
        after = list(before)
        changed = []
        for s in sorted(self.visited):
            if s in self.terminal:
                continue
            candidates = [(v, a) for a in range(self.n_actions) if (v := self.q(s, a, before)) is not None]
            if not candidates:
                continue
            value, a = max(candidates, key=lambda c: (c[0], -c[1]))
            after[s] = value
            if abs(value - before[s]) > 1e-12:
                edge = self.edges.get((s, a))
                changed.append({"state": s, "before": before[s], "after": value, "action": a, "next_state": edge[0] if edge else None})
        self.V = after
        self.sweeps += 1
        return LearningTrace(
            self.name,
            sweep=self.sweeps,
            unseen=self.unseen,
            discount=self.gamma,
            changed=changed,
            values_before=before,
            values_after=list(after),
            max_change=max((abs(c["after"] - c["before"]) for c in changed), default=0.0),
        )

    def plan(self, theta: float = 1e-10, max_sweeps: int = 10_000) -> list[float]:
        """Sweep until nothing changes: the best values the graph (and the assumption) allow."""
        for _ in range(max_sweeps):
            if self.sweep()["max_change"] < theta:
                break
        return self.V

    def greedy_route(self, start: int, env_move, max_steps: int = 60) -> list[int]:
        """Where the greedy policy would drive from `start` in the real world (env_move(s, a) -> (s', r, done))."""
        route, s = [start], start
        for _ in range(max_steps):
            s, _, done = env_move(s, self.act(s))
            route.append(s)
            if done:
                break
        return route


def explore_episode(env, graph: ExperienceGraph, epsilon: float = 0.0, rng=None, max_steps: int = 50) -> list[Transition]:
    """One online episode: act greedily on the graph (random with probability epsilon), add every
    transition to the graph, and replan after each step."""
    state, _ = env.reset()
    episode = []
    for _ in range(max_steps):
        action = int(rng.integers(graph.n_actions)) if epsilon and rng is not None and rng.random() < epsilon else graph.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        transition = Transition(state, action, float(reward), next_state, terminated, truncated)
        graph.add(transition)
        graph.plan()
        episode.append(transition)
        if terminated or truncated:
            break
        state = next_state
    return episode
