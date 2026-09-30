from __future__ import annotations

from visualrl.algorithms.tabular.q_agent import EpsilonGreedyQAgent
from visualrl.core import LearningTrace, Transition


class Sarsa(EpsilonGreedyQAgent):
    """SARSA: on-policy TD control.

        Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]

    a' is the action the behavior policy will actually take next, so the
    transition must carry `next_action` (unless the episode terminated).
    """

    name = "sarsa"

    def learn_step(self, transition: Transition) -> LearningTrace:
        s, a, r, s_next = transition.state, transition.action, transition.reward, transition.next_state
        a_next = transition.next_action
        if a_next is None and not transition.terminated:
            raise ValueError("SARSA needs next_action: the action the agent will take in next_state")

        prediction = self.Q[s, a]
        next_q_values = self.Q[s_next].copy()  # before the update, in case s' == s
        bootstrap_value = 0.0 if transition.terminated else self.Q[s_next, a_next]
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
            next_action=a_next,
            next_q_values=next_q_values,
            target_action=None if transition.terminated else a_next,
            prediction=prediction,
            bootstrap_value=bootstrap_value,
            discount=self.gamma,
            target=target,
            error=td_error,
            learning_rate=self.alpha,
            value_before=prediction,
            value_after=self.Q[s, a],
        )
