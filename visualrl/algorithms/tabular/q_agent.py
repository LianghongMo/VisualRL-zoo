from __future__ import annotations

import numpy as np

from visualrl.policies import epsilon_greedy_probs, greedy_policy, sample_action


class EpsilonGreedyQAgent:
    """A Q-table plus ε-greedy action selection, shared by SARSA and Q-learning.

    Acting lives here; each algorithm writes its own `learn_step`.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ):
        self.Q = np.full((n_states, n_actions), initial_value, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.learn_steps = 0

    def action_probs(self, state: int) -> np.ndarray:
        return epsilon_greedy_probs(self.Q[state], self.epsilon)

    def act(self, state: int) -> int:
        return sample_action(self.action_probs(state), self.rng)

    def greedy_policy(self) -> np.ndarray:
        return greedy_policy(self.Q)
