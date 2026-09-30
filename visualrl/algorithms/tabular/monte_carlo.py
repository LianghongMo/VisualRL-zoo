from __future__ import annotations

import numpy as np

from visualrl.core import LearningTrace, Trajectory


class MonteCarlo:
    """Constant-α Monte Carlo prediction of V^π.

        V(s_t) ← V(s_t) + α [G_t − V(s_t)]

    The target G_t sums every reward until the end of the episode, so nothing
    can be learned before the episode has terminated.
    """

    name = "monte_carlo"

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        first_visit: bool = False,
        initial_value: float = 0.0,
    ):
        self.V = np.full(n_states, initial_value, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.first_visit = first_visit
        self.learn_steps = 0

    def learn_episode(self, trajectory: Trajectory) -> list[LearningTrace]:
        """One update per visited timestep, in the order the states were visited."""
        if not trajectory.terminated:
            raise ValueError("Monte Carlo needs a terminated episode: G_t is not known before the end")

        returns = trajectory.returns(self.gamma)
        rewards = trajectory.rewards
        seen = set()
        traces = []
        for t, transition in enumerate(trajectory):
            s = transition.state
            if self.first_visit and s in seen:
                continue
            seen.add(s)

            prediction = self.V[s]
            target = returns[t]
            error = target - prediction

            self.V[s] += self.alpha * error
            self.learn_steps += 1

            traces.append(
                LearningTrace(
                    self.name,
                    learn_step=self.learn_steps,
                    timestep=t,
                    state=s,
                    rewards=rewards[t:],
                    discount=self.gamma,
                    prediction=prediction,
                    target=target,
                    error=error,
                    learning_rate=self.alpha,
                    value_before=prediction,
                    value_after=self.V[s],
                )
            )
        return traces
