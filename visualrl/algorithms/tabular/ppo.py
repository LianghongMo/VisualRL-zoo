from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from visualrl.core import LearningTrace, Transition


def log_softmax(logits) -> list[float]:
    m = max(logits)
    log_total = math.log(sum(math.exp(x - m) for x in logits))
    return [x - m - log_total for x in logits]


def softmax(logits) -> list[float]:
    return [math.exp(lp) for lp in log_softmax(logits)]


@dataclass
class Rollout:
    """A batch of experience collected with one fixed policy π_old.

    `old_probs[i]` is π_old(a_i | s_i), recorded while acting. Nothing in the
    batch changes while PPO learns from it; only the current policy π_θ does.
    """

    transitions: list[Transition] = field(default_factory=list)
    old_probs: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)


class TabularPPO:
    """Proximal Policy Optimization with a table of logits and a table of values.

        π_θ(a|s) = softmax(θ[s])          the policy (actor)
        V(s)                              the value table (critic)

    One iteration: collect a rollout with π_old = π_θ, compute GAE advantages,
    then take several epochs of minibatch gradient steps on the clipped
    objective. A deep PPO replaces both tables by neural networks; every other
    line stays the same.
    """

    name = "ppo"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        policy_lr: float = 2.0,
        value_lr: float = 2.0,
        entropy_coef: float = 0.01,
        normalize_advantages: bool = True,
        seed: int | None = None,
    ):
        self.logits = np.zeros((n_states, n_actions))
        self.V = np.zeros(n_states)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.entropy_coef = entropy_coef
        self.normalize_advantages = normalize_advantages
        self.rng = np.random.default_rng(seed)
        self.learn_steps = 0

    # ---------- acting ----------

    def action_probs(self, state: int) -> list[float]:
        return softmax(self.logits[state])

    def act(self, state: int) -> int:
        return int(self.rng.choice(len(self.logits[state]), p=self.action_probs(state)))

    def collect(self, env, n_steps: int, state: int) -> tuple[Rollout, int]:
        """Act for n_steps with the current policy (resetting after each episode). Returns the rollout and the state to continue from."""
        rollout = Rollout()
        for _ in range(n_steps):
            probs = self.action_probs(state)
            action = int(self.rng.choice(len(probs), p=probs))
            next_state, reward, terminated, truncated, _ = env.step(action)
            rollout.transitions.append(Transition(state, action, float(reward), next_state, terminated, truncated))
            rollout.old_probs.append(probs[action])
            state = env.reset()[0] if terminated or truncated else next_state
        return rollout, state

    # ---------- learning ----------

    def compute_advantages(self, rollout: Rollout) -> LearningTrace:
        """Generalized advantage estimation, backwards through the rollout.

            δ_t = r_t + γ V(s_{t+1}) − V(s_t)          (V(s_{t+1}) = 0 if the episode terminated)
            A_t = δ_t + γ λ A_{t+1}                    (the sum restarts after every episode end)
        """
        T = len(rollout)
        deltas = [0.0] * T
        advantages = [0.0] * T
        running = 0.0
        for t in reversed(range(T)):
            tr = rollout.transitions[t]
            next_value = 0.0 if tr.terminated else float(self.V[tr.next_state])
            deltas[t] = tr.reward + self.gamma * next_value - float(self.V[tr.state])
            if tr.terminated or tr.truncated:
                running = 0.0
            running = deltas[t] + self.gamma * self.lam * running
            advantages[t] = running
        returns = [a + float(self.V[tr.state]) for a, tr in zip(advantages, rollout.transitions)]
        if self.normalize_advantages and T > 1:
            mean = sum(advantages) / T
            std = math.sqrt(sum((a - mean) ** 2 for a in advantages) / T)
            advantages = [(a - mean) / (std + 1e-8) for a in advantages]
        rollout.advantages = advantages
        rollout.returns = returns
        return LearningTrace("gae", gamma=self.gamma, lam=self.lam, deltas=deltas, advantages=advantages, returns=returns)

    def _evaluate(self, rollout: Rollout, i: int) -> tuple[dict, list[float], list[float], float]:
        """The clipped objective of sample i under the current policy, plus what its gradient needs."""
        tr = rollout.transitions[i]
        s, a = tr.state, tr.action
        A = rollout.advantages[i]
        log_probs = log_softmax(self.logits[s])
        probs = [math.exp(lp) for lp in log_probs]
        log_ratio = log_probs[a] - math.log(rollout.old_probs[i])
        ratio = math.exp(log_ratio)
        lo, hi = 1.0 - self.clip, 1.0 + self.clip
        clipped_ratio = min(max(ratio, lo), hi)
        sample = {
            "index": i,
            "state": s,
            "action": a,
            "advantage": A,
            "old_prob": rollout.old_probs[i],
            "new_prob": probs[a],
            "ratio": ratio,
            "clipped_ratio": clipped_ratio,
            "objective": min(ratio * A, clipped_ratio * A),
            "clipped": (A > 0 and ratio > hi) or (A < 0 and ratio < lo),
            "return": rollout.returns[i],
            "value": float(self.V[s]),
        }
        return sample, log_probs, probs, log_ratio

    def inspect_sample(self, rollout: Rollout, i: int) -> dict:
        """Where sample i stands right now: its ratio, whether it is clipped, its objective. Changes nothing."""
        return self._evaluate(rollout, i)[0]

    def update_minibatch(self, rollout: Rollout, indices: list[int], epoch: int = 0) -> LearningTrace:
        """One gradient step on the samples `indices`.

        For each sample, with r = π_θ(a|s) / π_old(a|s):
            objective = min(r A, clip(r, 1−ε, 1+ε) A) + c H(π_θ(·|s))
        The first term's gradient is zero once r has left [1−ε, 1+ε] in the
        direction the advantage pushes; otherwise it is A r ∇ log π_θ(a|s).
        The entropy bonus c H keeps the policy from becoming certain too early.
        """
        M = len(indices)
        n_actions = self.logits.shape[1]
        policy_grad = {}
        value_grad = {}
        samples = []
        objective_sum = value_loss_sum = kl_sum = entropy_sum = 0.0
        clipped_count = 0

        for i in indices:
            sample, log_probs, probs, log_ratio = self._evaluate(rollout, i)
            s, a, A, ratio, clipped = sample["state"], sample["action"], sample["advantage"], sample["ratio"], sample["clipped"]
            entropy = -sum(p * lp for p, lp in zip(probs, log_probs))
            g = policy_grad.setdefault(s, [0.0] * n_actions)
            for b in range(n_actions):
                surrogate = 0.0 if clipped else A * ratio * ((1.0 if b == a else 0.0) - probs[b])
                bonus = -probs[b] * (log_probs[b] + entropy)
                g[b] += (surrogate + self.entropy_coef * bonus) / M
            value_error = float(self.V[s]) - rollout.returns[i]
            value_grad[s] = value_grad.get(s, 0.0) + value_error / M

            objective_sum += sample["objective"]
            value_loss_sum += 0.5 * value_error**2
            kl_sum += (ratio - 1.0) - log_ratio
            entropy_sum += entropy
            clipped_count += clipped
            samples.append(sample)

        probs_before = {s: softmax(self.logits[s]) for s in policy_grad}
        for s, g in policy_grad.items():
            for b in range(n_actions):
                self.logits[s, b] += self.policy_lr * g[b]  # gradient ascent on the objective
        for s, g in value_grad.items():
            self.V[s] -= self.value_lr * g  # gradient descent on the value loss
        self.learn_steps += 1

        return LearningTrace(
            self.name,
            learn_step=self.learn_steps,
            epoch=epoch,
            clip=self.clip,
            samples=samples,
            policy_objective=objective_sum / M,
            value_loss=value_loss_sum / M,
            approx_kl=kl_sum / M,
            entropy=entropy_sum / M,
            clip_fraction=clipped_count / M,
            policy_before={s: p for s, p in probs_before.items()},
            policy_after={s: softmax(self.logits[s]) for s in policy_grad},
        )

    def minibatches(self, n: int, minibatch_size: int) -> list[list[int]]:
        order = [int(i) for i in self.rng.permutation(n)]
        return [order[k : k + minibatch_size] for k in range(0, n, minibatch_size)]

    def update(self, rollout: Rollout, epochs: int = 4, minibatch_size: int = 64) -> list[LearningTrace]:
        """Compute advantages once, then learn from the same rollout for several epochs."""
        self.compute_advantages(rollout)
        traces = []
        for epoch in range(epochs):
            for indices in self.minibatches(len(rollout), minibatch_size):
                traces.append(self.update_minibatch(rollout, indices, epoch))
        return traces
