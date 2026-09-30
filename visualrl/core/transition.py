from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    """One interaction with the environment: (s_t, a_t, r_t, s_{t+1}).

    `terminated` means the episode really ended (the value of `next_state` is 0
    by definition). `truncated` means it was cut off by a time limit, so the
    value of `next_state` still matters.

    `next_action` is the action the behavior policy commits to in `next_state`.
    On-policy learners such as SARSA need it; everyone else may ignore it.
    """

    state: Any
    action: Any
    reward: float
    next_state: Any
    terminated: bool = False
    truncated: bool = False
    next_action: Any = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
