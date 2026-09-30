import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from visualrl.algorithms.deep.ppo import DeepPPO, PPOConfig, make_env  # noqa: E402


def test_one_iteration_on_a_mujoco_task_produces_finite_traces():
    env = make_env("InvertedPendulum-v5")
    agent = DeepPPO(env, PPOConfig(n_steps=256, epochs=2, minibatches=4), seed=0)
    agent.total_iterations = 1
    batch = agent.collect()
    traces = agent.update(batch)
    assert len(traces) == 2 * 4
    first = traces[0]
    assert first["mean_ratio"] == pytest.approx(1.0, abs=1e-5)  # nothing has changed before the first step
    for t in traces:
        for key in ("policy_objective", "value_loss", "approx_kl", "clip_fraction", "entropy"):
            assert math.isfinite(t[key])
    assert agent.env_steps == 256


def test_gae_matches_the_tabular_rule():
    env = make_env("InvertedPendulum-v5")
    agent = DeepPPO(env, PPOConfig(n_steps=4, gamma=0.5, lam=1.0))
    batch = {"rew": [1.0, 1.0, 1.0, 1.0], "val": [0.0] * 4, "next_val": [0.0] * 4, "done": [False, True, False, False]}
    agent.advantages(batch)
    assert list(batch["adv"]) == pytest.approx([1.5, 1.0, 1.5, 1.0])
