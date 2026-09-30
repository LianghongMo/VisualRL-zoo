from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Bandit(gym.Env):
    """k-armed bandit with Gaussian rewards.

    There is a single state (0) and the episode never ends: every step is one
    pull of an arm. Pulling arm `a` pays a sample from N(means[a], reward_std²).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        means=None,
        n_arms: int = 10,
        reward_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if means is None:
            means = np.random.default_rng(seed).normal(0.0, 1.0, size=n_arms)
        self.means = np.asarray(means, dtype=float)
        self.reward_std = float(reward_std)
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(len(self.means))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action: int):
        reward = self.np_random.normal(self.means[action], self.reward_std)
        return 0, float(reward), False, False, {}

    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self.means))

    def visual_state(self) -> dict:
        """Reference information for the visualization, never for the agent."""
        return {"means": self.means.tolist(), "optimal_action": self.optimal_action}
