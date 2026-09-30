from __future__ import annotations

from typing import Any, Callable

from visualrl.core.trajectory import Trajectory
from visualrl.core.transition import Transition


def rollout(
    env,
    policy: Callable[[Any], Any],
    max_steps: int = 1000,
    seed: int | None = None,
) -> Trajectory:
    """Run one episode with `policy(state) -> action` and record every transition.

    Nothing is learned here: collecting experience and learning from it are
    separate operations.
    """
    state, _ = env.reset(seed=seed)
    trajectory = Trajectory()
    for step in range(max_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        truncated = truncated or (not terminated and step == max_steps - 1)
        trajectory.append(Transition(state, action, float(reward), next_state, terminated, truncated))
        if terminated or truncated:
            break
        state = next_state
    return trajectory
