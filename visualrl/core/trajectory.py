from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from visualrl.core.transition import Transition


@dataclass
class Trajectory:
    """The transitions of one episode, in the order they happened."""

    transitions: list[Transition] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        if self.done:
            raise ValueError("cannot append to a finished episode")
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    def __getitem__(self, t: int) -> Transition:
        return self.transitions[t]

    @property
    def done(self) -> bool:
        return bool(self.transitions) and self.transitions[-1].done

    @property
    def terminated(self) -> bool:
        return bool(self.transitions) and self.transitions[-1].terminated

    @property
    def states(self) -> list:
        """s_0, s_1, ..., s_T (one more state than there are transitions)."""
        if not self.transitions:
            return []
        return [t.state for t in self.transitions] + [self.transitions[-1].next_state]

    @property
    def actions(self) -> list:
        return [t.action for t in self.transitions]

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.transitions]

    def returns(self, gamma: float) -> list[float]:
        """G_t = r_t + γ G_{t+1} for every timestep, computed backwards.

        For an unfinished or truncated episode these are the returns of the
        rewards collected so far, with nothing added for what comes after.
        """
        G = 0.0
        out = [0.0] * len(self.transitions)
        for t in reversed(range(len(self.transitions))):
            G = self.transitions[t].reward + gamma * G
            out[t] = G
        return out

    def discounted_rewards(self, gamma: float, start: int = 0) -> list[float]:
        """The terms γ^k r_{start+k}; their sum is the return G_start."""
        return [gamma**k * t.reward for k, t in enumerate(self.transitions[start:])]
