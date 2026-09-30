import numpy as np
import pytest

from visualrl import Trajectory, Transition, rollout
from visualrl.algorithms.tabular import (
    TD0,
    EpsilonGreedyBandit,
    MonteCarlo,
    NStepTD,
    PolicyIteration,
    QLearning,
    Sarsa,
    ValueIteration,
    bellman_backup,
    run_episode,
)
from visualrl.envs import Chain, GridWorld
from visualrl.policies import uniform_policy
from visualrl.reference import exact_policy_value, optimal_values
from visualrl.visual import render_path


def test_td_update_matches_the_worked_example_in_the_readme():
    td = TD0(n_states=2, alpha=0.1, gamma=0.99)
    td.V[:] = [0.30, 0.80]
    trace = td.learn_step(Transition(state=0, action=0, reward=0.0, next_state=1))
    assert trace["prediction"] == pytest.approx(0.30)
    assert trace["bootstrap_value"] == pytest.approx(0.80)
    assert trace["target"] == pytest.approx(0.792)
    assert trace["error"] == pytest.approx(0.492)
    assert trace["value_after"] == pytest.approx(0.3492)
    assert td.V[0] == pytest.approx(0.3492)


def test_td_does_not_bootstrap_from_a_terminal_state():
    td = TD0(n_states=2, alpha=1.0, gamma=1.0)
    td.V[1] = 100.0
    trace = td.learn_step(Transition(0, 0, 1.0, 1, terminated=True))
    assert trace["bootstrap_value"] == 0.0 and trace["target"] == 1.0


def test_td_bootstraps_through_a_truncation():
    td = TD0(n_states=2, alpha=1.0, gamma=1.0)
    td.V[1] = 5.0
    assert td.learn_step(Transition(0, 0, 1.0, 1, truncated=True))["target"] == 6.0


def random_walk_episodes(n_episodes, seed=0):
    env = Chain()
    rng = np.random.default_rng(seed)
    return [rollout(env, policy=lambda s: int(rng.integers(2))) for _ in range(n_episodes)]


def test_monte_carlo_needs_a_finished_episode():
    unfinished = Trajectory([Transition(3, 1, 0.0, 4)])
    with pytest.raises(ValueError):
        MonteCarlo(7).learn_episode(unfinished)


def test_monte_carlo_targets_are_returns():
    episode = random_walk_episodes(1, seed=3)[0]
    mc = MonteCarlo(7, alpha=0.5, gamma=0.9)
    traces = mc.learn_episode(episode)
    assert [t["target"] for t in traces] == pytest.approx(episode.returns(0.9))


def test_n_step_td_interpolates_between_td_and_monte_carlo():
    for episode in random_walk_episodes(5, seed=1):
        td, one_step = TD0(7, alpha=0.2, gamma=0.9), NStepTD(7, n=1, alpha=0.2, gamma=0.9)
        for t, transition in enumerate(episode):
            assert td.learn_step(transition)["target"] == pytest.approx(one_step.learn_step(episode, t)["target"])
        np.testing.assert_allclose(td.V, one_step.V)

        mc, long_n = MonteCarlo(7, alpha=0.2, gamma=0.9), NStepTD(7, n=1000, alpha=0.2, gamma=0.9)
        mc.learn_episode(episode)
        for t in range(len(episode)):
            long_n.learn_step(episode, t)
        np.testing.assert_allclose(mc.V, long_n.V)


def test_n_step_td_waits_for_enough_rewards():
    agent = NStepTD(7, n=3)
    partial = Trajectory([Transition(3, 1, 0.0, 4), Transition(4, 1, 0.0, 5)])
    assert not agent.ready(partial, 0)
    with pytest.raises(ValueError):
        agent.learn_step(partial, 0)


def test_td_and_mc_approach_the_true_random_walk_values():
    true_v = exact_policy_value(Chain().model(), uniform_policy(7, 2), gamma=1.0)
    np.testing.assert_allclose(true_v[1:6], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])

    td, mc = TD0(7, alpha=0.02, gamma=1.0), MonteCarlo(7, alpha=0.01, gamma=1.0)
    td.V[1:6] = mc.V[1:6] = 0.5
    for episode in random_walk_episodes(3000, seed=2):
        for transition in episode:
            td.learn_step(transition)
        mc.learn_episode(episode)
    assert np.sqrt(np.mean((td.V[1:6] - true_v[1:6]) ** 2)) < 0.05
    assert np.sqrt(np.mean((mc.V[1:6] - true_v[1:6]) ** 2)) < 0.05


def test_sarsa_needs_the_next_action():
    with pytest.raises(ValueError):
        Sarsa(4, 2).learn_step(Transition(0, 0, 0.0, 1))


def test_sarsa_and_q_learning_use_different_next_actions():
    transition = Transition(0, 0, -1.0, 1, next_action=0)
    sarsa, qlearn = Sarsa(2, 2, alpha=1.0, gamma=1.0), QLearning(2, 2, alpha=1.0, gamma=1.0)
    for agent in (sarsa, qlearn):
        agent.Q[1] = [-10.0, -2.0]
    s_trace, q_trace = sarsa.learn_step(transition), qlearn.learn_step(transition)
    assert s_trace["target_action"] == 0 and s_trace["target"] == -11.0
    assert q_trace["target_action"] == 1 and q_trace["target"] == -3.0
    assert q_trace["next_action"] == 0  # what the behavior policy did instead
    assert q_trace["next_q_values"] == pytest.approx([-10.0, -2.0])


def test_q_trace_records_next_values_before_the_update():
    agent = QLearning(1, 2, alpha=1.0, gamma=1.0)
    trace = agent.learn_step(Transition(0, 0, 5.0, 0))
    assert list(trace["next_q_values"]) == [0.0, 0.0]
    assert agent.Q[0, 0] == 5.0


def test_value_iteration_finds_the_shortest_cliff_path():
    V, Q = optimal_values(GridWorld.cliff().model(), gamma=1.0)
    env = GridWorld.cliff()
    assert V[env.start] == pytest.approx(-13.0)
    assert Q[env.start].argmax() == 0  # up, the only safe first move


def test_policy_iteration_and_value_iteration_agree():
    model = GridWorld.simple(slip=0.1).model()
    pi = PolicyIteration(model, gamma=0.9).solve(theta=1e-12)
    vi = ValueIteration(model, gamma=0.9).solve(theta=1e-12)
    np.testing.assert_allclose(pi.V, vi.V, atol=1e-8)
    np.testing.assert_allclose(exact_policy_value(model, pi.policy, 0.9), pi.V, atol=1e-8)


def test_evaluation_and_improvement_change_different_things():
    pi = PolicyIteration(GridWorld.simple().model(), gamma=0.9)
    policy_before = pi.policy.copy()
    pi.evaluate_step()
    np.testing.assert_array_equal(pi.policy, policy_before)
    values_before = pi.V.copy()
    trace = pi.improve_step()
    np.testing.assert_array_equal(pi.V, values_before)
    assert trace["changed_states"]


def test_bellman_backup_branches_sum_to_the_backed_up_value():
    model = GridWorld.simple(slip=0.2).model()
    V = np.random.default_rng(0).normal(size=model.n_states)
    policy = uniform_policy(model.n_states, 4)
    trace = bellman_backup(model, V, policy, state=12, gamma=0.9)
    assert sum(b["contribution"] for b in trace["branches"]) == pytest.approx(trace["value_after"])
    pe = PolicyIteration(model, gamma=0.9, policy=policy)
    pe.V = V.copy()
    assert pe.evaluate_step()["values_after"][12] == pytest.approx(trace["value_after"])


def greedy_path_return(env, Q, max_steps=100):
    total, s = 0.0, env.start
    for _ in range(max_steps):
        s, r, done = env._move(s, int(np.argmax(Q[s])))
        total += r
        if done:
            return total
    return -np.inf


def test_q_learning_walks_the_edge_and_sarsa_stays_safe():
    env = GridWorld.cliff()
    env.reset(seed=0)
    sarsa = Sarsa(48, 4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0)
    qlearn = QLearning(48, 4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0)
    for agent in (sarsa, qlearn):
        for _ in range(500):
            run_episode(env, agent)
    assert greedy_path_return(env, qlearn.Q) == -13.0, render_path(env, qlearn.Q)
    assert -30 < greedy_path_return(env, sarsa.Q) < -13.0, render_path(env, sarsa.Q)


def test_bandit_sample_average_is_the_mean_reward():
    agent = EpsilonGreedyBandit(3, epsilon=0.0)
    for r in [1.0, 2.0, 6.0]:
        trace = agent.learn_step(1, r)
    assert agent.Q[1] == pytest.approx(3.0)
    assert trace["learning_rate"] == pytest.approx(1 / 3)
