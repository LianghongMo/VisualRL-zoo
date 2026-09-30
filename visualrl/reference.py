"""Exact reference quantities (V^π, V*, Q*) computed from the environment model.

These are privileged: they exist so the visualization can show how far an
estimate is from the truth. No learning agent may read them.
"""

from __future__ import annotations

import numpy as np

from visualrl.algorithms.tabular.value_iteration import ValueIteration
from visualrl.envs.model import TabularModel


def exact_policy_value(model: TabularModel, policy, gamma: float) -> np.ndarray:
    """V^π by solving the linear system (I − γ P_π) V = r_π, with V = 0 on terminal states."""
    n = model.n_states
    A = np.eye(n)
    b = np.zeros(n)
    for s in range(n):
        if model.terminal[s]:
            continue
        for a, action_prob in enumerate(policy[s]):
            for p, s2, r, done in model.P[s][a]:
                b[s] += action_prob * p * r
                if not done:
                    A[s, s2] -= gamma * action_prob * p
    return np.linalg.solve(A, b)


def optimal_values(model: TabularModel, gamma: float, theta: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """(V*, Q*) by running value iteration to convergence."""
    vi = ValueIteration(model, gamma).solve(theta)
    return vi.V, vi.Q
