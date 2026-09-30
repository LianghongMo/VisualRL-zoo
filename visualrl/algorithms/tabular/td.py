from __future__ import annotations

import numpy as np

from visualrl.core import LearningTrace, Transition


class TD0:
    """TD(0) prediction of V^π.

        V(s_t) ← V(s_t) + α [r_t + γ V(s_{t+1}) − V(s_t)]

    One transition is enough for one update: the target borrows the current
    estimate of the next state instead of waiting for the episode to end.
    """

    name = "td0"

    def __init__(self, n_states: int, alpha: float = 0.1, gamma: float = 0.99, initial_value: float = 0.0):
        self.V = np.full(n_states, initial_value, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.learn_steps = 0

    def learn_step(self, transition: Transition) -> LearningTrace:
        s, r, s_next = transition.state, transition.reward, transition.next_state

        prediction = self.V[s]
        bootstrap_value = 0.0 if transition.terminated else self.V[s_next]
        target = r + self.gamma * bootstrap_value
        td_error = target - prediction

        self.V[s] += self.alpha * td_error
        self.learn_steps += 1

        return LearningTrace(
            self.name,
            learn_step=self.learn_steps,
            state=s,
            reward=r,
            next_state=s_next,
            terminated=transition.terminated,
            prediction=prediction,
            bootstrap_value=bootstrap_value,
            discount=self.gamma,
            target=target,
            error=td_error,
            learning_rate=self.alpha,
            value_before=prediction,
            value_after=self.V[s],
        )
