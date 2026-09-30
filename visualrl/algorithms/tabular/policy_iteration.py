from __future__ import annotations

import numpy as np

from visualrl.algorithms.tabular.bellman import bellman_backup, q_from_v
from visualrl.core import LearningTrace
from visualrl.envs.model import TabularModel
from visualrl.policies import greedy_policy, uniform_policy


class PolicyIteration:
    """Policy iteration as two separate operations.

    `evaluate_step()` changes the values and leaves the policy alone.
    `improve_step()` changes the policy and leaves the values alone.
    """

    name = "policy_iteration"

    def __init__(self, model: TabularModel, gamma: float = 0.99, policy=None):
        self.model = model
        self.gamma = gamma
        self.V = np.zeros(model.n_states)
        self.policy = uniform_policy(model.n_states, model.n_actions) if policy is None else np.array(policy, float)
        self.sweeps = 0
        self.improvements = 0

    def evaluate_step(self) -> LearningTrace:
        """One synchronous sweep of Bellman expectation backups over every state."""
        before = self.V.copy()
        after = before.copy()
        for s in range(self.model.n_states):
            if not self.model.terminal[s]:
                after[s] = bellman_backup(self.model, before, self.policy, s, self.gamma)["value_after"]
        self.V = after
        self.sweeps += 1
        return LearningTrace(
            "policy_evaluation",
            sweep=self.sweeps,
            discount=self.gamma,
            values_before=before,
            values_after=after.copy(),
            max_change=float(np.max(np.abs(after - before))),
        )

    def evaluate(self, theta: float = 1e-8, max_sweeps: int = 10_000) -> LearningTrace:
        """Sweep until no value changes by more than theta; return the last sweep."""
        for _ in range(max_sweeps):
            trace = self.evaluate_step()
            if trace["max_change"] < theta:
                break
        return trace

    def improve_step(self) -> LearningTrace:
        """Make the policy greedy with respect to the current values."""
        Q = q_from_v(self.model, self.V, self.gamma)
        before = self.policy.copy()
        after = greedy_policy(Q)
        after[self.model.terminal] = before[self.model.terminal]
        changed = [s for s in range(self.model.n_states) if not np.allclose(before[s], after[s])]
        self.policy = after
        self.improvements += 1
        return LearningTrace(
            "policy_improvement",
            improvement=self.improvements,
            q_values=Q,
            policy_before=before,
            policy_after=after.copy(),
            changed_states=changed,
            stable=not changed,
        )

    def solve(self, theta: float = 1e-8, max_iterations: int = 1000) -> "PolicyIteration":
        for _ in range(max_iterations):
            self.evaluate(theta)
            if self.improve_step()["stable"]:
                break
        return self
