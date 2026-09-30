from __future__ import annotations

from visualrl.core import LearningTrace, Trajectory, Transition


def run_episode(env, agent, learn: bool = True, max_steps: int = 10_000) -> tuple[Trajectory, list[LearningTrace]]:
    """Act with `agent` for one episode, learning after every environment step.

    The agent picks its next action as soon as it sees the next state, and the
    transition records that choice. SARSA needs it for its target; for
    Q-learning it shows what the behavior policy did instead of the greedy
    action the target uses.
    """
    trajectory, traces = Trajectory(), []
    state, _ = env.reset()
    action = agent.act(state)
    for step in range(max_steps):
        next_state, reward, terminated, truncated, _ = env.step(action)
        truncated = truncated or (not terminated and step == max_steps - 1)
        next_action = None if terminated else agent.act(next_state)
        transition = Transition(state, action, float(reward), next_state, terminated, truncated, next_action)
        trajectory.append(transition)
        if learn:
            traces.append(agent.learn_step(transition))
        if terminated or truncated:
            break
        state, action = next_state, next_action
    return trajectory, traces
