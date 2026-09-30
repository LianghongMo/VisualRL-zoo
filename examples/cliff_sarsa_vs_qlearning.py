"""SARSA and Q-learning on the cliff (Sutton & Barto, Example 6.6).

Both see the same kind of experience. SARSA learns the value of the ε-greedy
policy it follows and keeps away from the edge; Q-learning learns the value of
the greedy policy and walks right along it.

    python examples/cliff_sarsa_vs_qlearning.py
"""

import numpy as np

from visualrl.algorithms.tabular import QLearning, Sarsa, run_episode
from visualrl.envs import GridWorld
from visualrl.visual import render_path

env = GridWorld.cliff()
env.reset(seed=0)

for cls in (Sarsa, QLearning):
    agent = cls(env.observation_space.n, env.action_space.n, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0)
    returns = [sum(run_episode(env, agent)[0].rewards) for _ in range(500)]
    print(f"{cls.name}: mean return over the last 100 training episodes = {np.mean(returns[-100:]):.1f}")
    print(render_path(env, agent.Q), end="\n\n")
