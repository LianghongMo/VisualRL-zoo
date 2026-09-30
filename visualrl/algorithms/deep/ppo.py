"""PPO with neural networks for continuous control (MuJoCo), after CleanRL's ppo_continuous_action.py.

The algorithm is the one in visualrl/algorithms/tabular/ppo.py: collect a
rollout with π_old, compute GAE advantages, then take several epochs of
minibatch steps on the clipped objective. The two tables become two small
networks, the softmax over four actions becomes a Gaussian over continuous
joint torques, and plain gradient steps become Adam steps with gradient
clipping. Every update still returns a LearningTrace.

Requires the optional dependencies: pip install -e ".[deep]"
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from visualrl.core import LearningTrace


def make_env(env_id: str, gamma: float = 0.99, **kwargs) -> gym.Env:
    """The standard continuous-control wrappers: clip actions, normalize observations and rewards."""
    env = gym.make(env_id, **kwargs)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
    return env


def _layer(n_in: int, n_out: int, std: float = np.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """π_θ(a|s) = N(μ_θ(s), σ²) with a learned, state-independent σ; V_φ(s) a separate network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, 1.0))
        self.actor_mean = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, act_dim, 0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        return Normal(mean, self.actor_logstd.expand_as(mean).exp())


@dataclass
class PPOConfig:
    n_steps: int = 2048
    epochs: int = 10
    minibatches: int = 32
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64


class DeepPPO:
    name = "deep_ppo"

    def __init__(self, env: gym.Env, config: PPOConfig = PPOConfig(), seed: int = 0):
        self.env = env
        self.cfg = config
        torch.manual_seed(seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.net = ActorCritic(obs_dim, act_dim, config.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.obs, _ = env.reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.env_steps = 0
        self.learn_steps = 0
        self.iteration = 0

    # ---------- acting ----------

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        dist = self.net.distribution(torch.as_tensor(obs, dtype=torch.float32)[None])
        return (dist.mean if deterministic else dist.sample())[0].numpy()

    @torch.no_grad()
    def collect(self) -> dict:
        """n_steps with the current policy π_old; nothing is learned here."""
        n, cfg = self.cfg.n_steps, self.cfg
        obs_buf = np.zeros((n, *self.env.observation_space.shape), dtype=np.float32)
        act_buf = np.zeros((n, *self.env.action_space.shape), dtype=np.float32)
        logp_buf, rew_buf, val_buf = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
        term_buf, done_buf, next_val_buf = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.float32)
        episodes = []
        for t in range(n):
            o = torch.as_tensor(self.obs, dtype=torch.float32)[None]
            dist = self.net.distribution(o)
            a = dist.sample()
            obs_buf[t], act_buf[t] = self.obs, a[0].numpy()
            logp_buf[t] = dist.log_prob(a).sum(-1)[0]
            val_buf[t] = self.net.value(o)[0]
            next_obs, r, terminated, truncated, info = self.env.step(a[0].numpy())
            rew_buf[t], term_buf[t], done_buf[t] = r, terminated, terminated or truncated
            # Truncation keeps bootstrapping from the value of the state the episode was cut at.
            next_val_buf[t] = 0.0 if terminated else self.net.value(torch.as_tensor(next_obs, dtype=torch.float32)[None])[0]
            if "episode" in info:
                episodes.append({"return": float(info["episode"]["r"]), "length": int(info["episode"]["l"])})
            self.obs = self.env.reset()[0] if terminated or truncated else next_obs
        self.env_steps += n
        return {"obs": obs_buf, "act": act_buf, "logp": logp_buf, "rew": rew_buf, "val": val_buf, "next_val": next_val_buf, "done": done_buf, "episodes": episodes}

    # ---------- learning ----------

    def advantages(self, batch: dict) -> None:
        """GAE, exactly as in the tabular version."""
        cfg = self.cfg
        n = len(batch["rew"])
        adv = np.zeros(n, np.float32)
        running = 0.0
        for t in reversed(range(n)):
            delta = batch["rew"][t] + cfg.gamma * batch["next_val"][t] - batch["val"][t]
            if batch["done"][t]:
                running = 0.0
            running = delta + cfg.gamma * cfg.lam * running
            adv[t] = running
        batch["adv"], batch["ret"] = adv, adv + batch["val"]

    def update(self, batch: dict) -> list[LearningTrace]:
        cfg = self.cfg
        if cfg.anneal_lr:
            for group in self.opt.param_groups:
                group["lr"] = cfg.lr * (1.0 - self.iteration / max(1, self.total_iterations))
        self.advantages(batch)
        obs, act = torch.as_tensor(batch["obs"]), torch.as_tensor(batch["act"])
        old_logp, adv, ret = torch.as_tensor(batch["logp"]), torch.as_tensor(batch["adv"]), torch.as_tensor(batch["ret"])
        n = len(adv)
        size = n // cfg.minibatches
        traces = []
        for epoch in range(cfg.epochs):
            order = self.rng.permutation(n)
            for k in range(cfg.minibatches):
                idx = torch.as_tensor(order[k * size : (k + 1) * size])
                dist = self.net.distribution(obs[idx])
                logp = dist.log_prob(act[idx]).sum(-1)
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                surrogate = torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a)
                value_loss = 0.5 * ((self.net.value(obs[idx]) - ret[idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = -surrogate.mean() + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                self.learn_steps += 1
                with torch.no_grad():
                    clipped = ((a > 0) & (ratio > 1 + cfg.clip)) | ((a < 0) & (ratio < 1 - cfg.clip))
                    traces.append(
                        LearningTrace(
                            self.name,
                            learn_step=self.learn_steps,
                            iteration=self.iteration,
                            epoch=epoch,
                            policy_objective=float(surrogate.mean()),
                            value_loss=float(value_loss),
                            entropy=float(entropy),
                            approx_kl=float(((ratio - 1) - log_ratio).mean()),
                            clip_fraction=float(clipped.float().mean()),
                            mean_ratio=float(ratio.mean()),
                        )
                    )
        self.iteration += 1
        return traces

    def train(self, total_steps: int, on_iteration=None) -> None:
        self.total_iterations = total_steps // self.cfg.n_steps
        start = time.time()
        while self.iteration < self.total_iterations:
            batch = self.collect()
            traces = self.update(batch)
            if on_iteration:
                on_iteration(self, batch, traces, time.time() - start)
