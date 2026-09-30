"""Tabular policies as explicit probability tables π(a|s)."""

from __future__ import annotations

import numpy as np

# Two values closer than this count as a tie, so a fresh all-zero table
# explores every action instead of always picking action 0.
TIE_TOLERANCE = 1e-9


def greedy_actions(q_row) -> np.ndarray:
    """Every action whose value ties for the maximum."""
    q_row = np.asarray(q_row, dtype=float)
    return np.flatnonzero(q_row >= q_row.max() - TIE_TOLERANCE)


def greedy_probs(q_row) -> np.ndarray:
    """π(a|s) of the greedy policy; tied actions share the probability."""
    best = greedy_actions(q_row)
    probs = np.zeros(len(q_row))
    probs[best] = 1.0 / len(best)
    return probs


def epsilon_greedy_probs(q_row, epsilon: float) -> np.ndarray:
    """With probability ε act uniformly at random, otherwise greedily."""
    n = len(q_row)
    return (1.0 - epsilon) * greedy_probs(q_row) + epsilon / n


def uniform_policy(n_states: int, n_actions: int) -> np.ndarray:
    return np.full((n_states, n_actions), 1.0 / n_actions)


def greedy_policy(Q) -> np.ndarray:
    return np.array([greedy_probs(row) for row in np.asarray(Q, dtype=float)])


def sample_action(probs, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))
