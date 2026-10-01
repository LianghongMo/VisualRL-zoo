"""Part II: values, Bellman backups, and finding the optimal policy in the charging room."""

import numpy as np
import pytest

from visualrl.algorithms.tabular import PolicyIteration, ValueIteration, optimal_backup, q_from_v
from visualrl.envs import GridWorld
from visualrl.policies import uniform_policy
from visualrl.reference import exact_policy_value

GAMMA = 0.9
UP, RIGHT, DOWN, LEFT = range(4)


def room(**kw):
    env = GridWorld.charging_room(**kw)
    return env, env.model()


def test_one_backup_next_to_the_slow_charger():
    env, model = room()
    trace = optimal_backup(model, np.zeros(model.n_states), env.to_state(3, 4), GAMMA)
    assert trace["value_after"] == pytest.approx(1.0)
    assert trace["best_actions"] == [DOWN]
    assert [a["q"] for a in trace["actions"]] == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_value_moves_one_edge_per_sweep():
    env, model = room()
    vi = ValueIteration(model, GAMMA)
    first_dock_value = None
    for k in range(1, 12):
        vi.sweep()
        if first_dock_value is None and vi.V[env.start] > 0:
            first_dock_value = (k, vi.V[env.start])
    assert first_dock_value[0] == 5 and first_dock_value[1] == pytest.approx(GAMMA**4 * 1.0)  # the slow charger, 5 edges away
    assert vi.V[env.start] == pytest.approx(GAMMA**7 * 10.0)  # the fast charger, 8 edges away, arrived later


def test_the_optimal_policy_is_greedy_on_the_converged_values():
    env, model = room()
    vi = ValueIteration(model, GAMMA).solve()
    pi = vi.policy
    np.testing.assert_allclose(exact_policy_value(model, pi, GAMMA), vi.V, atol=1e-9)


def test_a_slippery_floor_steers_the_optimal_route_away_from_the_ledge():
    env, model = room(slip=0.2)
    vi = ValueIteration(model, GAMMA).solve()
    Q = q_from_v(model, vi.V, GAMMA)
    above_ledge = env.to_state(3, 1)
    assert Q[above_ledge].argmax() == UP  # deterministic ties between the two routes; slipping breaks them
    trace = optimal_backup(model, vi.V, env.to_state(3, 2), GAMMA)
    right = trace["actions"][RIGHT]
    assert sum(b["transition_prob"] for b in right["branches"]) == pytest.approx(1.0)
    assert any(b["next_state"] == env.start and b["reward"] == -10.0 for b in right["branches"])  # a slip over the ledge


def test_policy_iteration_from_random_reaches_the_value_iteration_answer():
    env, model = room()
    pi = PolicyIteration(model, GAMMA, policy=uniform_policy(model.n_states, 4))
    for _ in range(20):
        pi.evaluate(theta=1e-12)
        if pi.improve_step()["stable"]:
            break
    vi = ValueIteration(model, GAMMA).solve(theta=1e-12)
    np.testing.assert_allclose(pi.V, vi.V, atol=1e-8)
    assert pi.improvements <= 6
