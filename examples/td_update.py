"""The worked TD(0) example from the README, computed by the real algorithm.

    python examples/td_update.py
"""

from visualrl import Transition
from visualrl.algorithms.tabular import TD0

td = TD0(n_states=2, alpha=0.1, gamma=0.99)
td.V[:] = [0.30, 0.80]

trace = td.learn_step(Transition(state=0, action=0, reward=0.0, next_state=1))
print(trace.explain())
