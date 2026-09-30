from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (probability, next_state, reward, terminated), the convention of Gymnasium's
# toy-text environments (FrozenLake, CliffWalking, Taxi).
Outcome = tuple[float, int, float, bool]


@dataclass(frozen=True)
class TabularModel:
    """The exact dynamics of a small environment.

    `P[s][a]` lists every possible outcome of taking action `a` in state `s`.
    `terminal[s]` marks states whose value is 0 by definition: terminal states,
    and cells the agent can never occupy (walls, cliffs).

    This is privileged information. Planning algorithms (dynamic programming)
    and reference solvers may read it; learning agents never do.
    """

    P: list[list[list[Outcome]]]
    terminal: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.P)

    @property
    def n_actions(self) -> int:
        return len(self.P[0])
