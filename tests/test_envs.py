import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from visualrl.envs import Bandit, Chain, GridWorld

UP, RIGHT, DOWN, LEFT = range(4)


@pytest.mark.parametrize("env", [Bandit(seed=0), Chain(), GridWorld.simple(), GridWorld.cliff()])
def test_envs_follow_the_gymnasium_api(env):
    check_env(env, skip_render_check=True)


def test_chain_ends_pay_their_rewards():
    env = Chain(n_states=3, left_reward=-1.0, right_reward=2.0)
    s, _ = env.reset()
    assert s == 2
    assert env.step(Chain.LEFT)[:3] == (1, 0.0, False)
    assert env.step(Chain.LEFT)[:3] == (0, -1.0, True)
    env.reset()
    env.step(Chain.RIGHT)
    assert env.step(Chain.RIGHT)[:3] == (4, 2.0, True)


def test_cliff_sends_the_agent_back_to_start():
    env = GridWorld.cliff()
    s, _ = env.reset()
    assert env.to_cell(s) == (3, 0)
    s, r, terminated, truncated, _ = env.step(RIGHT)
    assert (s, r, terminated) == (env.start, -100.0, False)


def test_goal_is_terminal_and_walls_block():
    env = GridWorld(["S#G"], step_reward=-1.0, goal_reward=5.0)
    env.reset()
    assert env.step(RIGHT)[:3] == (0, -1.0, False)  # wall: stay put
    env = GridWorld(["SG"], goal_reward=5.0)
    env.reset()
    assert env.step(RIGHT)[:3] == (1, 5.0, True)


def test_time_limit_truncates():
    env = GridWorld.simple(max_steps=2)
    env.reset()
    env.step(LEFT)
    assert env.step(LEFT)[3] is True


@pytest.mark.parametrize("env", [Chain(), Chain(slip=0.2), GridWorld.cliff(), GridWorld.simple(slip=0.2)])
def test_models_are_probability_distributions(env):
    model = env.model()
    for s in range(model.n_states):
        for a in range(model.n_actions):
            assert sum(p for p, *_ in model.P[s][a]) == pytest.approx(1.0)


@pytest.mark.parametrize("env", [Chain(), GridWorld.cliff(), GridWorld.simple()])
def test_deterministic_model_matches_step(env):
    model = env.model()
    for s in range(model.n_states):
        if model.terminal[s]:
            continue
        for a in range(model.n_actions):
            env.reset()
            env.state = s
            next_state, reward, terminated, _, _ = env.step(a)
            assert model.P[s][a] == [(1.0, next_state, reward, terminated)]


def test_bandit_rewards_center_on_the_arm_means():
    env = Bandit(means=[0.0, 1.0, -1.0], reward_std=0.5)
    env.reset(seed=1)
    rewards = [env.step(1)[1] for _ in range(4000)]
    assert np.mean(rewards) == pytest.approx(1.0, abs=0.05)
    assert env.optimal_action == 1
