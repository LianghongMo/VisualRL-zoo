from __future__ import annotations

import numpy as np

from visualrl.algorithms.tabular.bellman import q_from_v
from visualrl.core import LearningTrace
from visualrl.envs.model import TabularModel
from visualrl.policies import greedy_policy


class ValueIteration:
    """Value iteration: V(s) ← max_a Σ_s' P(s'|s,a) [r + γ V(s')] for every state."""

    name = "value_iteration"

    def __init__(self, model: TabularModel, gamma: float = 0.99):
        self.model = model
        self.gamma = gamma
        self.V = np.zeros(model.n_states)
        self.sweeps = 0

    def sweep(self) -> LearningTrace:
        """One synchronous sweep of Bellman optimality backups."""
        before = self.V.copy()
        Q = q_from_v(self.model, before, self.gamma)
        after = np.where(self.model.terminal, 0.0, Q.max(axis=1))
        self.V = after
        self.sweeps += 1
        return LearningTrace(
            self.name,
            sweep=self.sweeps,
            discount=self.gamma,
            q_values=Q,
            values_before=before,
            values_after=after.copy(),
            max_change=float(np.max(np.abs(after - before))),
        )

    def solve(self, theta: float = 1e-10, max_sweeps: int = 100_000) -> "ValueIteration":
        for _ in range(max_sweeps):
            if self.sweep()["max_change"] < theta:
                break
        return self

    @property
    def Q(self) -> np.ndarray:
        return q_from_v(self.model, self.V, self.gamma)

    @property
    def policy(self) -> np.ndarray:
        return greedy_policy(self.Q)
