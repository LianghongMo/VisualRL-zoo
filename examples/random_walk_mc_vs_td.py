"""Monte Carlo vs TD(0) on the five-state random walk (Sutton & Barto, Example 6.2).

Both learn from exactly the same episodes. The reference values V^π come from
the environment model and are only used to measure the error.

    python examples/random_walk_mc_vs_td.py
"""

import numpy as np

from visualrl import rollout
from visualrl.algorithms.tabular import TD0, MonteCarlo
from visualrl.envs import Chain
from visualrl.policies import uniform_policy
from visualrl.reference import exact_policy_value

env = Chain()
true_v = exact_policy_value(env.model(), uniform_policy(7, 2), gamma=1.0)
rng = np.random.default_rng(0)

td, mc = TD0(7, alpha=0.1, gamma=1.0, initial_value=0.5), MonteCarlo(7, alpha=0.1, gamma=1.0, initial_value=0.5)
td.V[[0, 6]] = mc.V[[0, 6]] = 0.0


def rms(V):
    return np.sqrt(np.mean((V[1:6] - true_v[1:6]) ** 2))


print("episode   RMS error TD(0)   RMS error MC")
for episode in range(1, 101):
    trajectory = rollout(env, policy=lambda s: int(rng.integers(2)))
    for transition in trajectory:
        td.learn_step(transition)
    mc.learn_episode(trajectory)
    if episode in (1, 10, 25, 50, 100):
        print(f"{episode:7d}   {rms(td.V):15.3f}   {rms(mc.V):12.3f}")
