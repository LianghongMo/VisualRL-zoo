import json

import numpy as np
import pytest

from visualrl import LearningTrace, Trajectory, Transition, rollout
from visualrl.envs import Chain


def test_returns_are_discounted_sums_of_future_rewards():
    traj = Trajectory([Transition(0, 1, 0.0, 1), Transition(1, 1, 0.0, 2), Transition(2, 1, 10.0, 3, terminated=True)])
    gamma = 0.9
    assert traj.returns(gamma) == pytest.approx([8.1, 9.0, 10.0])
    assert sum(traj.discounted_rewards(gamma)) == pytest.approx(traj.returns(gamma)[0])
    assert traj.states == [0, 1, 2, 3]


def test_cannot_append_to_finished_episode():
    traj = Trajectory([Transition(0, 0, 1.0, 1, terminated=True)])
    with pytest.raises(ValueError):
        traj.append(Transition(1, 0, 0.0, 2))


def test_rollout_records_one_episode():
    env = Chain()
    traj = rollout(env, policy=lambda s: Chain.RIGHT)
    assert traj.states == [3, 4, 5, 6]
    assert traj.rewards == [0.0, 0.0, 1.0]
    assert traj.terminated


def test_rollout_marks_the_time_limit_as_truncation():
    traj = rollout(Chain(n_states=9), policy=lambda s: Chain.LEFT if s > 3 else Chain.RIGHT, max_steps=5)
    assert len(traj) == 5
    assert traj[-1].truncated and not traj.terminated


def test_trace_keeps_field_order_and_serializes_numpy():
    trace = LearningTrace("demo", prediction=np.float64(0.3), values=np.array([1.0, 2.0]), step=np.int64(2))
    assert list(trace) == ["algorithm", "prediction", "values", "step"]
    as_dict = trace.to_dict()
    assert as_dict == {"algorithm": "demo", "prediction": 0.3, "values": [1.0, 2.0], "step": 2}
    assert json.loads(trace.to_json()) == as_dict
    assert "prediction" in trace.explain()
