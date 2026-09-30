from __future__ import annotations

import numpy as np

from visualrl.core import LearningTrace, Trajectory


class NStepTD:
    """n-step TD prediction of V^π.

        G_{t:t+n} = r_t + γ r_{t+1} + ... + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n})
        V(s_t)   ← V(s_t) + α [G_{t:t+n} − V(s_t)]

    n = 1 is TD(0). When t + n reaches past the end of the episode no value
    is borrowed, and the target is the Monte Carlo return.
    """

    name = "n_step_td"

    def __init__(self, n_states: int, n: int = 1, alpha: float = 0.1, gamma: float = 1.0, initial_value: float = 0.0):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.V = np.full(n_states, initial_value, dtype=float)
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.learn_steps = 0

    def ready(self, trajectory: Trajectory, t: int) -> bool:
        """Whether the data needed for the target of timestep t has been collected."""
        return t + self.n <= len(trajectory) or trajectory.done

    def target(self, trajectory: Trajectory, t: int) -> dict:
        """The pieces of G_{t:t+n}: which rewards it uses and which value it borrows. Changes nothing."""
        if not self.ready(trajectory, t):
            raise ValueError(f"the {self.n}-step target of timestep {t} needs rewards that have not been collected yet")

        end = min(t + self.n, len(trajectory))
        rewards = [trajectory[k].reward for k in range(t, end)]
        last = trajectory[end - 1]
        if last.terminated:
            bootstrap_state, bootstrap_value, bootstrap_weight = None, 0.0, 0.0
        else:
            bootstrap_state = last.next_state
            bootstrap_value = self.V[bootstrap_state]
            bootstrap_weight = self.gamma ** (end - t)
        discounted = sum(self.gamma**k * r for k, r in enumerate(rewards))
        return {
            "rewards": rewards,
            "bootstrap_state": bootstrap_state,
            "bootstrap_value": bootstrap_value,
            "bootstrap_weight": bootstrap_weight,
            "target": discounted + bootstrap_weight * bootstrap_value,
        }

    def learn_step(self, trajectory: Trajectory, t: int) -> LearningTrace:
        parts = self.target(trajectory, t)
        s = trajectory[t].state
        prediction = self.V[s]
        target = parts["target"]
        error = target - prediction

        self.V[s] += self.alpha * error
        self.learn_steps += 1

        return LearningTrace(
            self.name,
            learn_step=self.learn_steps,
            n=self.n,
            timestep=t,
            state=s,
            rewards=parts["rewards"],
            discount=self.gamma,
            bootstrap_state=parts["bootstrap_state"],
            bootstrap_value=parts["bootstrap_value"],
            bootstrap_weight=parts["bootstrap_weight"],
            prediction=prediction,
            target=target,
            error=error,
            learning_rate=self.alpha,
            value_before=prediction,
            value_after=self.V[s],
        )
