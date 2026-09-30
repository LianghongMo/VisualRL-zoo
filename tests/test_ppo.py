import math

import numpy as np
import pytest

from visualrl import Transition
from visualrl.algorithms.tabular.ppo import Rollout, TabularPPO, softmax
from visualrl.envs import GridWorld


def tiny_rollout(old_probs=(0.25, 0.25)):
    rollout = Rollout(
        transitions=[Transition(0, 1, -1.0, 1), Transition(1, 1, 5.0, 2, terminated=True)],
        old_probs=list(old_probs),
    )
    return rollout


def test_gae_with_lambda_one_is_return_minus_value():
    agent = TabularPPO(3, 4, gamma=0.9, lam=1.0, normalize_advantages=False)
    agent.V[:] = [1.0, 2.0, 0.0]
    rollout = tiny_rollout()
    trace = agent.compute_advantages(rollout)
    G0 = -1.0 + 0.9 * 5.0
    assert rollout.advantages == pytest.approx([G0 - 1.0, 5.0 - 2.0])
    assert rollout.returns == pytest.approx([G0, 5.0])
    assert trace["deltas"] == pytest.approx([-1.0 + 0.9 * 2.0 - 1.0, 5.0 - 2.0])


def test_gae_with_lambda_zero_is_the_td_error():
    agent = TabularPPO(3, 4, gamma=0.9, lam=0.0, normalize_advantages=False)
    agent.V[:] = [1.0, 2.0, 0.0]
    rollout = tiny_rollout()
    trace = agent.compute_advantages(rollout)
    assert rollout.advantages == pytest.approx(trace["deltas"])


def test_first_step_on_fresh_data_has_ratio_one_and_no_clipping():
    agent = TabularPPO(3, 4, normalize_advantages=False, entropy_coef=0.0)
    rollout = tiny_rollout()
    agent.compute_advantages(rollout)
    trace = agent.update_minibatch(rollout, [0, 1])
    assert [s["ratio"] for s in trace["samples"]] == pytest.approx([1.0, 1.0])
    assert trace["clip_fraction"] == 0.0
    assert trace["policy_objective"] == pytest.approx(np.mean(rollout.advantages))


def test_clipped_samples_get_no_surrogate_gradient():
    agent = TabularPPO(3, 4, clip=0.2, normalize_advantages=False, entropy_coef=0.0)
    # π_θ(a=1|s=0) = 0.25 but the data claims π_old was 0.1: ratio 2.5, far above 1 + ε.
    rollout = tiny_rollout(old_probs=(0.1, 0.25))
    agent.compute_advantages(rollout)
    assert rollout.advantages[0] > 0
    before = agent.logits[0].copy()
    trace = agent.update_minibatch(rollout, [0])
    assert trace["samples"][0]["clipped"] and trace["samples"][0]["ratio"] == pytest.approx(2.5)
    assert trace["samples"][0]["objective"] == pytest.approx(1.2 * rollout.advantages[0])
    np.testing.assert_array_equal(agent.logits[0], before)


def test_unclipped_gradient_raises_the_probability_of_good_actions():
    agent = TabularPPO(3, 4, normalize_advantages=False, entropy_coef=0.0, policy_lr=0.5)
    rollout = tiny_rollout()
    agent.compute_advantages(rollout)
    p_before = softmax(agent.logits[1])[1]
    agent.update_minibatch(rollout, [1])  # advantage +3 for action 1 in state 1
    assert softmax(agent.logits[1])[1] > p_before


def test_kl_stays_finite_when_a_probability_underflows():
    agent = TabularPPO(3, 4, normalize_advantages=False)
    agent.logits[0] = [0.0, -2000.0, 0.0, 0.0]
    rollout = tiny_rollout()
    agent.compute_advantages(rollout)
    trace = agent.update_minibatch(rollout, [0])
    assert math.isfinite(trace["approx_kl"])


def test_ppo_learns_to_reach_the_charger():
    env = GridWorld.warehouse(max_steps=1000)
    state = env.reset(seed=0)[0]
    agent = TabularPPO(env.observation_space.n, 4, gamma=0.99, lam=0.95, clip=0.2, policy_lr=3.0, value_lr=2.0, entropy_coef=0.01, seed=0)
    for _ in range(45):
        rollout, state = agent.collect(env, 1024, state)
        agent.update(rollout, epochs=4, minibatch_size=256)
    s, total = env.start, 0.0
    for _ in range(60):
        s, r, done = env._move(s, int(np.argmax(agent.logits[s])))
        total += r
        if done:
            break
    assert done and -30 < total <= -13
