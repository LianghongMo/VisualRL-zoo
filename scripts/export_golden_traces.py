"""Write tests/golden/traces.json: fixed transitions and the traces Python produces for them.

The web app re-implements the tabular algorithms in JavaScript so lessons run
in the browser. `web/tests/parity.test.mjs` replays these exact transitions
and checks that every JavaScript trace matches the Python one, so both
implementations stay the same algorithm.

    python scripts/export_golden_traces.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from visualrl import rollout
from visualrl.algorithms.tabular import TD0, MonteCarlo, NStepTD, QLearning, Sarsa, run_episode
from visualrl.algorithms.tabular.ppo import TabularPPO
from visualrl.envs import Chain, GridWorld
from visualrl.policies import uniform_policy
from visualrl.reference import exact_policy_value, optimal_values

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "traces.json"


def chain_episodes(n: int, seed: int):
    env = Chain()
    rng = np.random.default_rng(seed)
    return [rollout(env, policy=lambda s: int(rng.integers(2))) for _ in range(n)]


def build() -> dict:
    episodes = chain_episodes(4, seed=7)
    golden = {}

    td = TD0(7, alpha=0.1, gamma=0.9, initial_value=0.5)
    golden["td0"] = {
        "config": {"n_states": 7, "alpha": 0.1, "gamma": 0.9, "initial_value": 0.5},
        "transitions": [asdict(t) for ep in episodes for t in ep],
        "traces": [td.learn_step(t).to_dict() for ep in episodes for t in ep],
        "final": td.V.tolist(),
    }

    mc = MonteCarlo(7, alpha=0.1, gamma=0.9, initial_value=0.5)
    golden["monte_carlo"] = {
        "config": {"n_states": 7, "alpha": 0.1, "gamma": 0.9, "initial_value": 0.5},
        "episodes": [[asdict(t) for t in ep] for ep in episodes],
        "traces": [trace.to_dict() for ep in episodes for trace in mc.learn_episode(ep)],
        "final": mc.V.tolist(),
    }

    nstep = NStepTD(7, n=3, alpha=0.1, gamma=0.9, initial_value=0.5)
    golden["n_step_td"] = {
        "config": {"n_states": 7, "n": 3, "alpha": 0.1, "gamma": 0.9, "initial_value": 0.5},
        "episodes": [[asdict(t) for t in ep] for ep in episodes],
        "traces": [nstep.learn_step(ep, t).to_dict() for ep in episodes for t in range(len(ep))],
        "final": nstep.V.tolist(),
    }

    for name, cls in (("sarsa", Sarsa), ("q_learning", QLearning)):
        env = GridWorld.cliff()
        env.reset(seed=11)
        collector = cls(48, 4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=11)
        transitions = []
        for _ in range(3):
            trajectory, _ = run_episode(env, collector)
            transitions += list(trajectory)
        # Any transitions can be learned from in any order; keep the file small but
        # include cliff falls (early wandering) and a goal-reaching ending.
        transitions = transitions[:150] + transitions[-10:]
        replay = cls(48, 4, alpha=0.5, gamma=1.0, epsilon=0.1)
        golden[name] = {
            "config": {"n_states": 48, "n_actions": 4, "alpha": 0.5, "gamma": 1.0},
            "transitions": [asdict(t) for t in transitions],
            "traces": [replay.learn_step(t).to_dict() for t in transitions],
            "final": replay.Q.tolist(),
        }

    # PPO: one rollout from the warehouse, then two epochs of minibatches with the indices recorded.
    env = GridWorld.warehouse(max_steps=1000)
    state = env.reset(seed=5)[0]
    config = {"n_states": int(env.observation_space.n), "n_actions": 4, "gamma": 0.99, "lam": 0.95, "clip": 0.2, "policy_lr": 3.0, "value_lr": 2.0, "entropy_coef": 0.01}
    collector = TabularPPO(config["n_states"], 4, **{k: v for k, v in config.items() if k not in ("n_states", "n_actions")}, seed=5)
    for _ in range(3):  # move the policy away from uniform so ratios and clipping are exercised
        warm, state = collector.collect(env, 256, state)
        collector.update(warm, epochs=4, minibatch_size=64)
    rollout, state = collector.collect(env, 128, state)
    learner = TabularPPO(config["n_states"], 4, **{k: v for k, v in config.items() if k not in ("n_states", "n_actions")}, seed=6)
    learner.logits = collector.logits.copy()
    learner.V = collector.V.copy()
    initial = {"logits": learner.logits.tolist(), "V": learner.V.tolist()}
    gae = learner.compute_advantages(rollout).to_dict()
    minibatches = [learner.minibatches(len(rollout), 32) for _ in range(3)]
    updates = [learner.update_minibatch(rollout, idx, epoch).to_dict() for epoch, batches in enumerate(minibatches) for idx in batches]
    golden["ppo"] = {
        "config": config,
        "initial": initial,
        "transitions": [asdict(t) for t in rollout.transitions],
        "old_probs": rollout.old_probs,
        "gae": gae,
        "minibatches": minibatches,
        "updates": updates,
        "final": {"logits": learner.logits.tolist(), "V": learner.V.tolist()},
    }

    V_star, Q_star = optimal_values(GridWorld.cliff().model(), gamma=1.0)
    golden["cliff_optimal"] = {"gamma": 1.0, "V": V_star.tolist(), "Q": Q_star.tolist()}
    golden["random_walk_exact"] = {
        "gamma": 1.0,
        "V": exact_policy_value(Chain().model(), uniform_policy(7, 2), 1.0).tolist(),
    }
    return golden


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
