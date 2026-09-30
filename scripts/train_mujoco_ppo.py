"""Train deep PPO on a MuJoCo task and export everything the web lesson replays.

The web page cannot run MuJoCo, so this script records what really happened:
  - the training log, one row per PPO iteration (return, clip fraction, KL, ...)
  - evaluation episodes of the policy at several points during training, as
    the position and orientation of every MuJoCo geom in every frame, with the
    action, the raw reward and the critic's value estimate of each step

    python scripts/train_mujoco_ppo.py Hopper-v5 --steps 1000000 --out web/src/data/mujoco/hopper.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

from visualrl.algorithms.deep.ppo import DeepPPO, PPOConfig, make_env

try:  # PointMaze and AntMaze live in gymnasium-robotics
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
except ImportError:
    pass

GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}


def geoms(model: mujoco.MjModel) -> list[dict]:
    out = []
    for g in range(model.ngeom):
        kind = GEOM_TYPES.get(int(model.geom_type[g]))
        rgba = model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g]
        # Invisible static geoms are helpers; an invisible moving geom is still the robot's body (PointMaze's ball).
        if kind is None or (rgba[3] == 0 and model.geom_bodyid[g] == 0):
            continue
        out.append({"id": g, "type": kind, "size": [round(float(x), 4) for x in model.geom_size[g]], "rgba": [round(float(x), 3) for x in rgba], "body": int(model.geom_bodyid[g])})
    return out


def pose(data: mujoco.MjData, ids: list[int]) -> tuple[list[int], list[int]]:
    """Positions in millimetres and unit quaternions ×10⁴, as integers, for a compact file."""
    pos, quat = [], []
    q = np.zeros(4)
    for g in ids:
        pos += [int(round(1000 * x)) for x in data.geom_xpos[g]]
        mujoco.mju_mat2Quat(q, data.geom_xmat[g])
        quat += [int(round(1e4 * x)) for x in q]
    return pos, quat


@torch.no_grad()
def record(agent: DeepPPO, train_env: gym.Env, env_id: str, max_steps: int, seed: int, fps: int = 30, env_kwargs=None) -> dict:
    """One deterministic episode (the mean action) with the training run's frozen observation statistics."""
    env = gym.make(env_id, **(env_kwargs or {}))
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    rms = train_env.get_wrapper_attr("obs_rms")
    model, data = env.unwrapped.model, env.unwrapped.data
    specs = geoms(model)
    dynamic = [s["id"] for s in specs if s["body"] != 0]
    static = [s["id"] for s in specs if s["body"] == 0]
    inner = env.unwrapped
    dt = float(getattr(inner, "dt", None) or getattr(getattr(inner, "point_env", None) or getattr(inner, "ant_env"), "dt"))
    every = max(1, round(1 / (fps * dt)))

    obs, _ = env.reset(seed=seed)
    frames, rewards, actions, values, returns = [], [], [], [], []
    goal = None
    if hasattr(env.unwrapped, "goal"):
        goal = [round(float(x), 3) for x in env.unwrapped.goal]
    ret, steps = 0.0, 0
    for t in range(max_steps):
        norm = np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10).astype(np.float32)
        value = float(agent.net.value(torch.as_tensor(norm)[None])[0])
        action = np.clip(agent.act(norm, deterministic=True), env.action_space.low, env.action_space.high)
        if t % every == 0:
            frames.append(pose(data, dynamic))
            values.append(round(value, 3))
            actions.append([round(float(a), 3) for a in action])
        obs, r, terminated, truncated, _ = env.step(action)
        ret += r
        steps += 1
        if t % every == 0:
            rewards.append(round(float(r), 3))
            returns.append(round(float(ret), 2))
        if terminated or truncated:
            break
    frames.append(pose(data, dynamic))
    static_pose = pose(data, static)
    return {
        "env_steps": agent.env_steps,
        "return": round(ret, 2),
        "length": steps,
        "frame_dt": round(dt * every, 5),
        "goal": goal,
        "pos": [f[0] for f in frames],
        "quat": [f[1] for f in frames],
        "reward": rewards,
        "return_so_far": returns,
        "action": actions,
        "value": values,
        "static": {"ids": static, "pos": static_pose[0], "quat": static_pose[1]},
        "dynamic": dynamic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoints", type=float, nargs="+", default=[0.0, 0.05, 0.25, 1.0])
    parser.add_argument("--record-steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--env-kwargs", type=json.loads, default={}, help='e.g. \'{"continuing_task": false}\'')
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    env = make_env(args.env_id, **args.env_kwargs)
    agent = DeepPPO(env, PPOConfig(), seed=args.seed)
    iterations = args.steps // agent.cfg.n_steps
    marks = sorted({min(iterations, math.ceil(f * iterations)) for f in args.checkpoints})
    log = {k: [] for k in ["env_steps", "return", "clip_fraction", "approx_kl", "entropy", "value_loss", "mean_ratio"]}
    checkpoints = []

    def maybe_record():
        if agent.iteration in marks and all(c["iteration"] != agent.iteration for c in checkpoints):
            rec = record(agent, env, args.env_id, args.record_steps, seed=args.seed + 1000, fps=args.fps, env_kwargs=args.env_kwargs)
            rec["iteration"] = agent.iteration
            checkpoints.append(rec)
            # Keep the policy itself, so episodes can be re-recorded later without retraining.
            rms = env.get_wrapper_attr("obs_rms")
            torch.save({"net": agent.net.state_dict(), "obs_mean": rms.mean, "obs_var": rms.var, "env_steps": agent.env_steps}, f"{args.out}.it{agent.iteration}.pt")
            print(f"recorded iteration {agent.iteration}: return {rec['return']:.1f} over {rec['length']} steps", flush=True)

    def on_iteration(agent, batch, traces, elapsed):
        last = traces[-agent.cfg.minibatches * agent.cfg.epochs :]
        eps = batch["episodes"]
        log["env_steps"].append(agent.env_steps)
        log["return"].append(round(float(np.mean([e["return"] for e in eps])), 2) if eps else None)
        for key in ["clip_fraction", "approx_kl", "entropy", "value_loss", "mean_ratio"]:
            log[key].append(round(float(np.mean([t[key] for t in last])), 5))
        if agent.iteration % 10 == 0:
            print(f"iteration {agent.iteration}/{iterations} steps {agent.env_steps} return {log['return'][-1]} ({elapsed:.0f}s)", flush=True)
        maybe_record()

    maybe_record()
    agent.train(args.steps, on_iteration=on_iteration)

    probe = gym.make(args.env_id, **args.env_kwargs).unwrapped
    out = {
        "env_id": args.env_id,
        "algorithm": "PPO (visualrl.algorithms.deep.ppo), CleanRL defaults",
        "config": {**agent.cfg.__dict__, "total_steps": args.steps, "seed": args.seed, "env_kwargs": args.env_kwargs},
        "obs_dim": int(np.prod(env.observation_space.shape)),
        "act_dim": int(np.prod(env.action_space.shape)),
        "geoms": geoms(probe.model),
        "training": log,
        "checkpoints": checkpoints,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
