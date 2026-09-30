from __future__ import annotations

from collections import defaultdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from visualrl.envs.model import TabularModel


class GridWorld(gym.Env):
    """A grid described by a text layout.

        S  start           G  goal (terminal)
        .  empty           #  wall
        C  cliff: pays `cliff_reward` and sends the agent back to the start

    The cell in row r, column c is state r * width + c. Actions are 0 up,
    1 right, 2 down, 3 left. Moving into a wall or off the grid leaves the
    agent where it is. With probability `slip` a uniformly random action is
    executed instead of the chosen one.

    Every move pays `step_reward`, except a move into a goal (`goal_reward`)
    or into a cliff (`cliff_reward`).
    """

    ACTION_NAMES = ("up", "right", "down", "left")
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    metadata = {"render_modes": []}

    def __init__(
        self,
        layout: list[str],
        step_reward: float = 0.0,
        goal_reward: float = 1.0,
        cliff_reward: float = -100.0,
        slip: float = 0.0,
        max_steps: int | None = None,
    ) -> None:
        self.layout = [row for row in layout]
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        if any(len(row) != self.width for row in self.layout):
            raise ValueError("every layout row must have the same length")
        cells = {(r, c): ch for r, row in enumerate(self.layout) for c, ch in enumerate(row)}
        unknown = set(cells.values()) - set("SG.#C")
        if unknown:
            raise ValueError(f"unknown layout characters: {sorted(unknown)}")
        starts = [rc for rc, ch in cells.items() if ch == "S"]
        if len(starts) != 1:
            raise ValueError("the layout needs exactly one start cell 'S'")
        self.start = self.to_state(*starts[0])
        self.goals = sorted(self.to_state(*rc) for rc, ch in cells.items() if ch == "G")
        self.walls = sorted(self.to_state(*rc) for rc, ch in cells.items() if ch == "#")
        self.cliffs = sorted(self.to_state(*rc) for rc, ch in cells.items() if ch == "C")
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.cliff_reward = float(cliff_reward)
        self.slip = float(slip)
        self.max_steps = max_steps
        self.observation_space = spaces.Discrete(self.height * self.width)
        self.action_space = spaces.Discrete(4)
        self.state: int | None = None
        self.steps = 0

    @classmethod
    def cliff(cls, **kwargs) -> "GridWorld":
        """The cliff-walking task of Sutton & Barto (Example 6.6).

        Every step costs -1 (including the last one into the goal) and
        stepping into the cliff costs -100 and restarts from S.
        """
        layout = [
            "............",
            "............",
            "............",
            "SCCCCCCCCCCG",
        ]
        kwargs = {"step_reward": -1.0, "goal_reward": -1.0, "cliff_reward": -100.0, **kwargs}
        return cls(layout, **kwargs)

    @classmethod
    def simple(cls, **kwargs) -> "GridWorld":
        """A 5x5 room with two walls and a +1 goal in the corner."""
        layout = [
            "....G",
            ".#.#.",
            ".....",
            ".#...",
            "S....",
        ]
        return cls(layout, **kwargs)

    def to_state(self, row: int, col: int) -> int:
        return row * self.width + col

    def to_cell(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.width)

    def _move(self, state: int, action: int) -> tuple[int, float, bool]:
        row, col = self.to_cell(state)
        dr, dc = self.MOVES[action]
        r, c = row + dr, col + dc
        if not (0 <= r < self.height and 0 <= c < self.width) or self.layout[r][c] == "#":
            r, c = row, col
        next_state = self.to_state(r, c)
        if next_state in self.goals:
            return next_state, self.goal_reward, True
        if next_state in self.cliffs:
            return self.start, self.cliff_reward, False
        return next_state, self.step_reward, False

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.steps = 0
        return self.state, {}

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        executed = int(action)
        if self.slip > 0 and self.np_random.random() < self.slip:
            executed = int(self.np_random.integers(4))
        self.state, reward, terminated = self._move(self.state, executed)
        self.steps += 1
        truncated = not terminated and self.max_steps is not None and self.steps >= self.max_steps
        return self.state, reward, terminated, truncated, {"executed_action": executed}

    def model(self) -> TabularModel:
        n = self.height * self.width
        terminal = np.zeros(n, dtype=bool)
        terminal[self.goals + self.walls + self.cliffs] = True
        P = []
        for s in range(n):
            if terminal[s]:
                P.append([[(1.0, s, 0.0, True)] for _ in range(4)])
                continue
            row = []
            for a in range(4):
                probs: dict[tuple[int, float, bool], float] = defaultdict(float)
                probs[self._move(s, a)] += 1.0 - self.slip
                if self.slip > 0:
                    for executed in range(4):
                        probs[self._move(s, executed)] += self.slip / 4
                row.append([(p, *outcome) for outcome, p in probs.items()])
            P.append(row)
        return TabularModel(P, terminal)

    def visual_state(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "agent": None if self.state is None else self.to_cell(self.state),
            "start": self.to_cell(self.start),
            "goals": [self.to_cell(s) for s in self.goals],
            "walls": [self.to_cell(s) for s in self.walls],
            "cliffs": [self.to_cell(s) for s in self.cliffs],
        }
