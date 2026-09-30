from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from visualrl.envs.model import TabularModel


class Chain(gym.Env):
    """A row of cells with a terminal state at each end.

        state:   0    1    2   ...   n   n+1
                [T]  s1   s2   ...   sn  [T]

    Action 0 moves left, action 1 moves right. Entering the left terminal pays
    `left_reward`, entering the right terminal pays `right_reward`, and every
    other move pays `step_reward`. With probability `slip` the agent moves the
    other way.

    The defaults are the random-walk task of Sutton & Barto (Example 6.2):
    five interior states, start in the middle, +1 for leaving on the right.
    """

    LEFT, RIGHT = 0, 1
    ACTION_NAMES = ("left", "right")
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_states: int = 5,
        start: int | None = None,
        left_reward: float = 0.0,
        right_reward: float = 1.0,
        step_reward: float = 0.0,
        slip: float = 0.0,
        max_steps: int | None = None,
    ) -> None:
        if n_states < 1:
            raise ValueError("a chain needs at least one interior state")
        self.n_interior = n_states
        self.start = (n_states + 1) // 2 if start is None else start
        if not 1 <= self.start <= n_states:
            raise ValueError(f"start must be an interior state 1..{n_states}")
        self.left_reward = float(left_reward)
        self.right_reward = float(right_reward)
        self.step_reward = float(step_reward)
        self.slip = float(slip)
        self.max_steps = max_steps
        self.observation_space = spaces.Discrete(n_states + 2)
        self.action_space = spaces.Discrete(2)
        self.state: int | None = None
        self.steps = 0

    @property
    def terminal_states(self) -> tuple[int, int]:
        return 0, self.n_interior + 1

    def _move(self, state: int, direction: int) -> tuple[int, float, bool]:
        next_state = state - 1 if direction == self.LEFT else state + 1
        if next_state == 0:
            return next_state, self.left_reward, True
        if next_state == self.n_interior + 1:
            return next_state, self.right_reward, True
        return next_state, self.step_reward, False

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.steps = 0
        return self.state, {}

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        direction = int(action)
        if self.slip > 0 and self.np_random.random() < self.slip:
            direction = 1 - direction
        self.state, reward, terminated = self._move(self.state, direction)
        self.steps += 1
        truncated = not terminated and self.max_steps is not None and self.steps >= self.max_steps
        return self.state, reward, terminated, truncated, {}

    def model(self) -> TabularModel:
        P = []
        for s in range(self.n_interior + 2):
            if s in self.terminal_states:
                P.append([[(1.0, s, 0.0, True)] for _ in range(2)])
                continue
            row = []
            for a in range(2):
                outcomes = [(1.0 - self.slip, *self._move(s, a))]
                if self.slip > 0:
                    outcomes.append((self.slip, *self._move(s, 1 - a)))
                row.append(outcomes)
            P.append(row)
        terminal = np.zeros(self.n_interior + 2, dtype=bool)
        terminal[list(self.terminal_states)] = True
        return TabularModel(P, terminal)

    def visual_state(self) -> dict:
        return {
            "n_states": self.n_interior + 2,
            "agent": self.state,
            "start": self.start,
            "terminals": list(self.terminal_states),
            "left_reward": self.left_reward,
            "right_reward": self.right_reward,
        }
