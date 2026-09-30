from __future__ import annotations

import numpy as np

from visualrl.core import LearningTrace
from visualrl.policies import epsilon_greedy_probs, sample_action


class EpsilonGreedyBandit:
    """Sample-average action values with ε-greedy exploration.

        N(a) ← N(a) + 1
        Q(a) ← Q(a) + (1 / N(a)) [r − Q(a)]

    ε = 0 is the purely greedy agent.
    """

    name = "bandit_sample_average"

    def __init__(self, n_arms: int, epsilon: float = 0.1, initial_value: float = 0.0, seed: int | None = None):
        self.Q = np.full(n_arms, initial_value, dtype=float)
        self.N = np.zeros(n_arms, dtype=int)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.learn_steps = 0

    def action_probs(self) -> np.ndarray:
        return epsilon_greedy_probs(self.Q, self.epsilon)

    def act(self) -> int:
        return sample_action(self.action_probs(), self.rng)

    def learn_step(self, action: int, reward: float) -> LearningTrace:
        self.N[action] += 1
        step_size = 1.0 / self.N[action]

        prediction = self.Q[action]
        error = reward - prediction
        self.Q[action] += step_size * error
        self.learn_steps += 1

        return LearningTrace(
            self.name,
            learn_step=self.learn_steps,
            action=action,
            reward=reward,
            count=self.N[action],
            prediction=prediction,
            target=reward,
            error=error,
            learning_rate=step_size,
            value_before=prediction,
            value_after=self.Q[action],
        )
