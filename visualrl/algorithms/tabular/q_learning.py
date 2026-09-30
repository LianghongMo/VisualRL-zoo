from __future__ import annotations

import numpy as np

from visualrl.algorithms.tabular.q_agent import EpsilonGreedyQAgent
from visualrl.core import LearningTrace, Transition


class QLearning(EpsilonGreedyQAgent):
    """Q-learning: off-policy TD control.

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

    The target uses the greedy next action, whatever the behavior policy will
    actually do. `next_action` in the trace is the behavior's choice (if the
    transition recorded one) so the two can be compared.
    """

    name = "q_learning"

    def learn_step(self, transition: Transition) -> LearningTrace:
        s, a, r, s_next = transition.state, transition.action, transition.reward, transition.next_state

        prediction = self.Q[s, a]
        next_q_values = self.Q[s_next].copy()  # before the update, in case s' == s
        if transition.terminated:
            target_action, bootstrap_value = None, 0.0
        else:
            target_action = int(np.argmax(self.Q[s_next]))
            bootstrap_value = self.Q[s_next, target_action]
        target = r + self.gamma * bootstrap_value
        td_error = target - prediction

        self.Q[s, a] += self.alpha * td_error
        self.learn_steps += 1

        return LearningTrace(
            self.name,
            learn_step=self.learn_steps,
            state=s,
            action=a,
            reward=r,
            next_state=s_next,
            terminated=transition.terminated,
            next_action=transition.next_action,
            next_q_values=next_q_values,
            target_action=target_action,
            prediction=prediction,
            bootstrap_value=bootstrap_value,
            discount=self.gamma,
            target=target,
            error=td_error,
            learning_rate=self.alpha,
            value_before=prediction,
            value_after=self.Q[s, a],
        )
