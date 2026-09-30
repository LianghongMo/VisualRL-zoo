import numpy as np
import pytest

from visualrl import Transition
from visualrl.algorithms.tabular.experience_graph import OPTIMISTIC, PESSIMISTIC, ExperienceGraph, explore_episode
from visualrl.envs import GridWorld
from visualrl.reference import optimal_values

UP, RIGHT, DOWN, LEFT = range(4)
GAMMA = 0.9


def drive(env, actions):
    """Record the transitions of a fixed sequence of moves from the dock."""
    s, _ = env.reset()
    out = []
    for a in actions:
        s2, r, term, trunc, _ = env.step(a)
        out.append(Transition(s, a, float(r), s2, term, trunc))
        s = s2
        if term:
            break
    return out


# Episode A reaches the slow charger (+1); episode B reaches the fast charger (+10) by a long detour.
EPISODE_A = [UP, RIGHT, RIGHT, RIGHT, DOWN]
EPISODE_B = [UP, UP, UP, RIGHT, RIGHT, RIGHT, DOWN, DOWN, RIGHT, RIGHT, UP, UP]


def offline_graph(unseen):
    env = GridWorld.charging_room()
    graph = ExperienceGraph(env.observation_space.n, 4, gamma=GAMMA, unseen=unseen)
    for tr in drive(env, EPISODE_A) + drive(env, EPISODE_B):
        graph.add(tr)
    graph.plan()
    return env, graph


def rollout_greedy(env, graph, max_steps=30):
    s, _ = env.reset()
    total, route = 0.0, [s]
    for t in range(max_steps):
        s, r, term, _, _ = env.step(graph.act(s))
        total += GAMMA**t * r
        route.append(s)
        if term:
            break
    return total, route


def test_the_two_episodes_reach_different_chargers():
    env = GridWorld.charging_room()
    a, b = drive(env, EPISODE_A), drive(env, EPISODE_B)
    assert a[-1].terminated and a[-1].reward == 1.0 and len(a) == 5
    assert b[-1].terminated and b[-1].reward == 10.0 and len(b) == 12


def test_planning_on_the_data_stitches_a_better_route():
    env, graph = offline_graph(PESSIMISTIC)
    ret, route = rollout_greedy(env, graph)
    best_in_data = max(GAMMA**4 * 1.0, GAMMA**11 * 10.0)
    assert ret == pytest.approx(GAMMA**7 * 10.0)
    assert ret > best_in_data
    assert len(route) == 9  # 8 moves: A's first four, then B's last four


def test_optimism_about_unseen_moves_drives_offline_into_the_ledge():
    env, graph = offline_graph(OPTIMISTIC)
    assert graph.act(env.start) == RIGHT  # never tried in the data, assumed to be worth +10
    ret, route = rollout_greedy(env, graph, max_steps=10)
    assert ret < 0 and route[1] == env.start  # over the ledge and back to the dock


def test_one_sweep_moves_value_one_edge_back():
    env, graph = offline_graph(PESSIMISTIC)
    graph.V = [0.0] * len(graph.V)
    first = graph.sweep()
    assert {c["next_state"] for c in first["changed"]} <= set(env.goals) | {env.start}
    assert graph.V[env.start] == 0.0  # the dock is several edges away from any charger
    graph.plan()
    assert graph.V[env.start] == pytest.approx(GAMMA**7 * 10.0)


def test_optimism_online_explores_until_it_knows_the_best_route():
    env = GridWorld.charging_room()
    graph = ExperienceGraph(env.observation_space.n, 4, gamma=GAMMA, unseen=OPTIMISTIC)
    for _ in range(40):
        explore_episode(env, graph, max_steps=60)
        if not graph.frontier():
            break
    V_star, _ = optimal_values(env.model(), GAMMA)
    assert graph.V[env.start] == pytest.approx(V_star[env.start])
    assert rollout_greedy(env, graph)[0] == pytest.approx(V_star[env.start])


def test_pessimism_online_needs_random_moves_to_find_anything():
    env = GridWorld.charging_room()
    graph = ExperienceGraph(env.observation_space.n, 4, gamma=GAMMA, unseen=PESSIMISTIC)
    for _ in range(5):
        explore_episode(env, graph, epsilon=0.0, max_steps=30)
    assert not graph.terminal  # greedy on what it has seen, it never reaches a charger
