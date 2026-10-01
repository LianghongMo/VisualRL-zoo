from __future__ import annotations

import numpy as np

from visualrl.core import LearningTrace
from visualrl.envs.model import TabularModel


def bellman_backup(model: TabularModel, V, policy, state: int, gamma: float) -> LearningTrace:
    """Expand one Bellman expectation backup into its branches.

        V(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [r + γ V(s')]

    Every branch (a, s') records its probabilities, reward, next-state value
    and contribution to the sum. `V` itself is not modified.
    """
    branches = []
    total = 0.0
    for a, action_prob in enumerate(policy[state]):
        for transition_prob, next_state, reward, terminated in model.P[state][a]:
            next_value = 0.0 if terminated else float(V[next_state])
            contribution = action_prob * transition_prob * (reward + gamma * next_value)
            total += contribution
            branches.append(
                {
                    "action": a,
                    "action_prob": float(action_prob),
                    "transition_prob": float(transition_prob),
                    "next_state": next_state,
                    "reward": float(reward),
                    "terminated": terminated,
                    "next_value": next_value,
                    "contribution": contribution,
                }
            )
    return LearningTrace(
        "bellman_backup",
        state=state,
        discount=gamma,
        branches=branches,
        value_before=float(V[state]),
        value_after=total,
    )


def optimal_backup(model: TabularModel, V, state: int, gamma: float) -> LearningTrace:
    """Expand one Bellman optimality backup: the value of every action at `state`, then the best.

        Q(s,a) = Σ_s' P(s'|s,a) [r + γ V(s')]        V(s) ← max_a Q(s,a)

    On a deterministic graph each action has a single branch, the edge it follows;
    on a slippery floor it has one branch per place the robot may end up.
    `V` itself is not modified.
    """
    actions = []
    for a in range(model.n_actions):
        branches = []
        q = 0.0
        for transition_prob, next_state, reward, terminated in model.P[state][a]:
            next_value = 0.0 if terminated else float(V[next_state])
            contribution = transition_prob * (reward + gamma * next_value)
            q += contribution
            branches.append(
                {
                    "transition_prob": float(transition_prob),
                    "next_state": next_state,
                    "reward": float(reward),
                    "terminated": terminated,
                    "next_value": next_value,
                    "contribution": contribution,
                }
            )
        actions.append({"action": a, "q": q, "branches": branches})
    best = max(x["q"] for x in actions)
    return LearningTrace(
        "optimal_backup",
        state=state,
        discount=gamma,
        actions=actions,
        best_actions=[x["action"] for x in actions if x["q"] >= best - 1e-9],
        value_before=float(V[state]),
        value_after=best,
    )


def q_from_v(model: TabularModel, V, gamma: float) -> np.ndarray:
    """Q(s,a) = Σ_s' P(s'|s,a) [r + γ V(s')] for every state and action."""
    Q = np.zeros((model.n_states, model.n_actions))
    for s in range(model.n_states):
        if model.terminal[s]:
            continue
        for a in range(model.n_actions):
            Q[s, a] = sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in model.P[s][a])
    return Q
