# Visual RL

**Learn reinforcement learning by seeing every decision, target, and update.**

Visual RL is an interactive, beginner-friendly reinforcement learning tutorial built around one idea:

> **Do not only show what an RL agent does. Show how the agent learns.**

Most reinforcement learning demos visualize the final behavior of an agent: a robot walks, a cart balances, or an Atari agent plays a game.

That is useful, but it hides the most important part of reinforcement learning.

Why did a value change?

Where did the TD target come from?

Why did one transition get replayed?

What is the difference between the behavior policy and the policy being learned?

What exactly does PPO clipping change?

Visual RL makes these internal learning processes visible.

---

## Status and quick start

Phase 1 (core tabular system) is in progress.

| Part | State |
|---|---|
| `visualrl/envs`: `Bandit`, `Chain`, `GridWorld` (incl. the cliff), Gymnasium API + exact models | done |
| `visualrl/core`: `Transition`, `Trajectory`, `LearningTrace`, `rollout` | done |
| `visualrl/algorithms/tabular`: TD(0), Monte Carlo, n-step TD, SARSA, Q-learning, Bellman backup, policy iteration, value iteration, ε-greedy bandit | done |
| `visualrl/reference.py`: exact $V^\pi$, $V^*$, $Q^*$ from the model (for the visualizer only) | done |
| `web/src/rl`: the same tabular algorithms in JavaScript for the browser, checked against Python traces | done |
| `web`: interactive lessons 02, 08, 09 | in progress |

```bash
pip install -e ".[dev]"
pytest                                   # Python tests
python examples/td_update.py             # the worked TD example below, computed by TD0
python examples/cliff_sarsa_vs_qlearning.py

cd web && npm install && npm test        # JavaScript engine + parity with Python
```

The Python package lives in `visualrl/` (so `envs/`, `core/`, `algorithms/` in the structure below are `visualrl/envs/`, `visualrl/core/`, ...).

The web lessons run a JavaScript copy of the tabular algorithms so they work in any browser without a server. To keep "the algorithm produces the truth" honest, `scripts/export_golden_traces.py` records transitions and the Python traces for them in `tests/golden/traces.json`, and `web/tests/parity.test.mjs` replays those transitions and requires every JavaScript trace to match.

---

## 1. Goal

The goal of this repository is to build an interactive path from the most basic reinforcement learning ideas to modern deep RL.

The tutorial is designed around the learning loop

```text
environment
    ↓
experience
    ↓
learning target
    ↓
parameter update
    ↓
policy/value change
    ↓
new behavior
```

Every chapter should allow the learner to inspect this loop directly.

Instead of treating an algorithm as

```python
agent.learn(...)
```

the learner should be able to pause at one update and ask:

```text
What data was used?

What was the current prediction?

What was the learning target?

What was the error?

What changed after the update?

How did that change affect future behavior?
```

This is the central design principle of Visual RL.

---

## 2. What this project is not

Visual RL is **not intended to be another general-purpose RL framework**.

There are already excellent libraries for that purpose:

- Gymnasium for environment interfaces,
- Stable-Baselines3 for reliable RL implementations,
- CleanRL for readable single-file deep RL implementations,
- TorchRL and Tianshou for modular RL infrastructure.

Visual RL instead focuses on **interpretability and education**.

The priority is therefore

```text
clarity > abstraction
inspectability > performance
small environments > large benchmarks
explicit updates > hidden framework logic
```

A 20-line Q-learning implementation whose entire update can be visualized is often more useful here than a highly optimized general implementation.

---

## 3. Learning philosophy

The tutorial follows four recurring questions.

### 3.1 What is the agent doing?

Visualize:

- current state,
- available actions,
- action probabilities,
- selected action,
- trajectory,
- visitation distribution.

### 3.2 What data is the agent learning from?

Visualize:

- transition $(s_t,a_t,r_t,s_{t+1})$,
- episode,
- trajectory,
- replay buffer,
- minibatch,
- on-policy versus replayed data.

### 3.3 What quantity is being computed?

Visualize:

- return,
- value,
- Q-value,
- Bellman target,
- TD error,
- advantage,
- policy ratio,
- loss.

For example, a TD update should not appear only as

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha
\left[
r_t+\gamma V(s_{t+1})-V(s_t)
\right].
$$

Instead, the interface should expose

```text
current value        V(s_t)       = 0.30

reward               r_t          = 0.00
next-state value     V(s_{t+1})   = 0.80
discount             γ            = 0.99

target                            = 0.792
TD error                          = 0.492

learning rate        α            = 0.10

updated value                     = 0.3492
```

### 3.4 What changed because of learning?

After every update, visualize the difference between

```text
before learning
      ↓
after learning
```

for quantities such as

- state values,
- Q-values,
- action probabilities,
- neural-network predictions,
- policy behavior,
- evaluation return.

The user should be able to connect a numerical update to a behavioral consequence.

---

## 4. Acting and learning are separate

A central feature of the tutorial is that

```text
TAKE ENVIRONMENT STEP
```

and

```text
TAKE LEARNING STEP
```

are separate operations.

This distinction is important because an RL agent can

1. collect experience without immediately learning,
2. perform multiple learning updates from existing experience,
3. replay old data,
4. learn from data generated by another policy.

This becomes essential when introducing

- Monte Carlo learning,
- TD learning,
- replay buffers,
- off-policy learning,
- DQN,
- offline RL.

A learner should therefore be able to do something like

```text
step environment
step environment
step environment

pause

learn once
learn once
learn once
```

and inspect what changes during each operation.

---

## 5. Curriculum

The main tutorial starts with tabular RL and gradually introduces deep RL.

The same small environments are reused whenever possible so that changes in behavior come from changes in the algorithm rather than changes in the task.

---

### Part I — What is reinforcement learning?

#### 01. Agent and Environment

Concepts:

- state,
- observation,
- action,
- reward,
- transition,
- episode,
- trajectory.

Visualization:

```text
state
 ↓
action
 ↓
environment transition
 ↓
reward + next state
```

Each interaction appends one transition to a visible trajectory timeline.

#### 02. Reward, Return, and Discounting

Concepts:

$$
G_t
=
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}.
$$

Visualization:

- reward at every timestep,
- discounted contribution of every reward,
- total return.

Interactive experiment:

```text
near small reward
vs.
far large reward
```

Drag $\gamma$ and observe how the preferred trajectory changes.

#### 03. Policies and Trajectories

Concepts:

$$
\pi(a\mid s).
$$

Visualization:

- action probabilities at every state,
- sampled actions,
- many trajectories generated from the same policy,
- state visitation heatmap.

The goal is to make clear that a policy is not the same thing as a trajectory.

#### 04. Exploration and Exploitation

Environment:

multi-armed bandit.

Visualize:

- estimated action value,
- number of samples per action,
- observed reward samples,
- cumulative reward.

Compare:

- greedy,
- $\epsilon$-greedy,
- optionally UCB and Thompson sampling later.

---

### Part II — Learning values

#### 05. State Value and Action Value

Introduce

$$
V^\pi(s)
$$

and

$$
Q^\pi(s,a).
$$

Grid cells display $V(s)$.

Each action direction inside the cell can display $Q(s,a)$.

Clicking a state repeatedly generates rollouts from that state and shows the empirical return distribution.

#### 06. Bellman Backup

For a selected state, explicitly expand

$$
V^\pi(s)
=
\sum_a
\pi(a\mid s)
\sum_{s'}
P(s'\mid s,a)
\left[
r+\gamma V^\pi(s')
\right].
$$

Visualization:

```text
                    action 1 → next state
                  /
current state → action 2 → next state
                  \
                    action 3 → next state
```

Every branch shows

- policy probability,
- transition probability,
- reward,
- next-state value,
- contribution to the final expectation.

The learner can perform **one Bellman backup** manually.

#### 07. Policy Evaluation and Policy Improvement

Separate two operations.

```text
Evaluate Policy
```

changes value estimates but leaves the policy unchanged.

```text
Improve Policy
```

changes the policy according to the current values.

This chapter introduces

- iterative policy evaluation,
- policy iteration,
- value iteration.

#### 08. Monte Carlo vs TD

Use exactly the same trajectory for both algorithms.

Monte Carlo target:

$$
G_t.
$$

TD target:

$$
r_t+\gamma V(s_{t+1}).
$$

Visualize which future rewards each method uses.

Then introduce

$$
n\text{-step TD}
$$

as a continuous interpolation between the two.

A particularly useful interaction is allowing the user to replay the same transitions in different orders and observe how information propagates.

---

### Part III — Control

#### 09. SARSA vs Q-learning

For SARSA:

$$
r+\gamma Q(s',a').
$$

For Q-learning:

$$
r+\gamma\max_{a'}Q(s',a').
$$

During one update, highlight exactly which next action enters the target.

Use a small cliff-style environment to show that

```text
behavior action
```

and

```text
target action
```

do not necessarily mean the same thing.

#### 10. On-policy, Off-policy, and Replay

Display two policies when necessary:

```text
behavior policy
     ↓
generates data


target/current policy
     ↓
being learned
```

Experience is represented explicitly in a replay buffer.

Example:

```text
Replay Buffer

#41  (s3, right, 0, s4)
#42  (s4, right, 0, s5)
#43  (s5, up,    1, s9)
...
```

Sampling a minibatch visibly moves selected transitions into the learning panel.

---

### Part IV — Function Approximation

#### 11. From Tables to Functions

Compare:

```text
lookup table
linear function
small neural network
```

When one training example is updated, show how predictions change across the entire state space.

This introduces the key idea of generalization:

> updating one state can change predictions at other states.

The same mechanism can help learning or create interference.

---

### Part V — Deep Reinforcement Learning

#### 12. DQN

Visualize the entire pipeline:

```text
environment
    ↓
transition
    ↓
replay buffer
    ↓
sample minibatch
    ↓
online Q network
    ↓
Bellman target
    ↓
loss
    ↓
gradient update
```

Also display the target network separately:

```text
online network
      │
      │ periodic / soft update
      ↓
target network
```

The learner should see exactly which network generates which quantity.

#### 13. REINFORCE

Start with a two-action softmax policy.

Display:

```text
logits
 ↓
action probabilities
 ↓
sampled action
 ↓
trajectory return
 ↓
policy gradient
 ↓
new action probabilities
```

This makes policy gradient concrete before introducing larger neural policies.

#### 14. Actor-Critic and Advantage

Show

$$
Q(s,a),
\qquad
V(s),
\qquad
A(s,a)=Q(s,a)-V(s).
$$

Separate

```text
UPDATE CRITIC
```

from

```text
UPDATE ACTOR
```

so the user can see that the two networks solve different learning problems.

#### 15. PPO

Visualize

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\rm old}}(a_t\mid s_t)
}.
$$

and

$$
L^{\rm clip}
=
\min
\left[
r_tA_t,
\operatorname{clip}
(r_t,1-\epsilon,1+\epsilon)A_t
\right].
$$

The page should include an interactive graph where the user changes

- probability ratio,
- advantage,
- clipping parameter.

The user should immediately see how the optimization objective changes.

#### 16. Evaluation and Debugging

Visualize:

- training return,
- evaluation return,
- multiple random seeds,
- mean and uncertainty,
- environment steps,
- gradient steps,
- termination,
- truncation.

The user should learn that

```text
loss going down
```

does not automatically imply

```text
policy getting better.
```

---

## 6. Advanced modules

After the main curriculum, additional modules can introduce:

```text
TD(λ)
GAE
SAC
continuous control
partial observability
memory
model-based RL
offline RL
goal-conditioned RL
HER
multi-agent RL
```

These are deliberately separated from the beginner path.

The main tutorial should remain understandable without them.

---

## 7. Environments

The early tutorial uses deliberately small environments:

```text
Bandit
Chain
GridWorld
```

These environments are simple enough that we can often compute exact reference quantities such as

$$
V^*,
\qquad
Q^*,
\qquad
V^\pi.
$$

The interface must clearly distinguish

```text
learner estimate
```

from

```text
reference / ground-truth quantity
```

because the learning algorithm itself should not receive privileged information from the visualizer.

More complex chapters can later use Gymnasium environments such as

```text
CartPole
Pendulum
```

and environments from MiniGrid.

---

## 8. Software architecture

The project has four conceptually separate layers.

```text
┌────────────────────────────────────┐
│           Visualization            │
│ maps / plots / timeline / UI       │
└─────────────────▲──────────────────┘
                  │
             Learning Trace
                  │
┌─────────────────┴──────────────────┐
│             Algorithms             │
│ TD / Q-learning / DQN / PPO / ...  │
└─────────────────▲──────────────────┘
                  │
               Experience
                  │
┌─────────────────┴──────────────────┐
│            Environments            │
│ Bandit / Chain / GridWorld / Gym   │
└────────────────────────────────────┘

          Reference Implementations
       CleanRL / SB3 / exact solutions
```

The visualization should never contain the algorithm itself.

The algorithm should produce structured information that the visualization consumes.

---

## 9. Core abstraction: Learning Trace

Most RL implementations expose only the final loss.

For visualization, that is not enough.

Visual RL introduces an explicit learning trace.

Conceptually:

```python
trace = agent.learn_step(batch)
```

The algorithm performs a real learning update and returns a description of what happened.

For Q-learning, a trace may look like

```python
{
    "algorithm": "q_learning",

    "state": state,
    "action": action,
    "reward": reward,
    "next_state": next_state,

    "prediction": old_q,

    "bootstrap_value": next_q_max,
    "target": target,

    "error": td_error,
    "learning_rate": alpha,

    "value_before": old_q,
    "value_after": new_q,
}
```

For DQN:

```python
{
    "batch": batch,

    "q_prediction": q_prediction,
    "target_q": target_q,

    "td_error": td_error,
    "loss": loss,

    "network_before": ...,
    "network_after": ...,
}
```

For PPO:

```python
{
    "observation": obs,
    "action": action,

    "old_log_prob": old_log_prob,
    "new_log_prob": new_log_prob,

    "ratio": ratio,
    "advantage": advantage,

    "unclipped_objective": ...,
    "clipped_objective": ...,

    "loss": loss,
}
```

The exact trace schema can evolve.

The important principle is

> **the algorithm produces the truth; the visualization only displays it.**

No educational animation should fake an optimization update.

---

## 10. Environment architecture

Environments should follow Gymnasium semantics whenever possible.

```python
obs, info = env.reset()

next_obs, reward, terminated, truncated, info = env.step(action)
```

However, educational environments may expose additional visualization metadata.

For example:

```python
env.visual_state()
```

could provide

```python
{
    "agent_position": ...,
    "goal_position": ...,
    "walls": ...,
    "reward_locations": ...,
}
```

This information belongs to the visualization layer and should remain separate from the observation received by the agent.

This distinction is particularly important for partially observable environments.

---

## 11. Proposed repository structure

```text
visual-rl/
│
├── README.md
│
├── envs/
│   ├── bandit.py
│   ├── chain.py
│   ├── gridworld.py
│   └── gymnasium_adapter.py
│
├── algorithms/
│   ├── tabular/
│   │   ├── value_iteration.py
│   │   ├── policy_iteration.py
│   │   ├── monte_carlo.py
│   │   ├── td.py
│   │   ├── n_step_td.py
│   │   ├── sarsa.py
│   │   └── q_learning.py
│   │
│   └── deep/
│       ├── reinforce.py
│       ├── actor_critic.py
│       ├── dqn.py
│       ├── ppo.py
│       └── sac.py
│
├── core/
│   ├── transition.py
│   ├── trajectory.py
│   ├── replay_buffer.py
│   └── learning_trace.py
│
├── visual/
│   ├── gridworld.py
│   ├── trajectory.py
│   ├── policy.py
│   ├── value_map.py
│   ├── replay_buffer.py
│   ├── bellman_backup.py
│   ├── loss_landscape.py
│   └── learning_curve.py
│
├── lessons/
│   ├── 01_agent_environment/
│   ├── 02_return_discount/
│   ├── 03_policy/
│   ├── 04_exploration/
│   ├── 05_value_functions/
│   ├── 06_bellman/
│   ├── 07_dynamic_programming/
│   ├── 08_mc_td/
│   ├── 09_sarsa_qlearning/
│   ├── 10_off_policy_replay/
│   ├── 11_function_approximation/
│   ├── 12_dqn/
│   ├── 13_reinforce/
│   ├── 14_actor_critic/
│   ├── 15_ppo/
│   └── 16_evaluation/
│
├── baselines/
│   └── sb3/
│       ├── dqn.py
│       ├── ppo.py
│       └── sac.py
│
├── experiments/
│   ├── configs/
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│
└── web/
    ├── components/
    ├── pages/
    └── app/
```

---

## 12. Relationship to existing RL repositories

Visual RL should use existing repositories strategically rather than reimplementing everything.

### Gymnasium

Use directly for the standard environment interface.

Visual RL environments should be Gymnasium-compatible whenever possible.

### Sutton & Barto implementations

Use as conceptual and numerical references for classical RL:

```text
Bandits
Dynamic Programming
Monte Carlo
TD
SARSA
Q-learning
n-step methods
```

The tutorial implementations themselves remain independent and visualization-oriented.

### CleanRL

Use as the main reference for readable modern deep-RL implementations.

Particularly useful references:

```text
DQN
PPO
SAC
```

Visual RL should preserve a similarly direct relationship between mathematical equations and code, while exposing more intermediate quantities for visualization.

### Stable-Baselines3

Use primarily as a baseline and correctness check.

For example:

```text
Visual RL PPO
       │
       ├──── same environment
       │
SB3 PPO
```

Compare learning behavior and final performance.

### RL Baselines3 Zoo

Use as a reference for experiment infrastructure:

```text
training
evaluation
seeding
logging
checkpointing
hyperparameters
plotting
```

### MiniGrid

Use for later environment modules involving

```text
partial observability
exploration
objects
doors
keys
memory
goal-directed behavior
```

The earliest GridWorld environments remain custom and deliberately minimal.

---

## 13. Implementation rule: simple algorithms stay simple

The project should resist unnecessary abstraction.

For example, tabular Q-learning should remain recognizable:

```python
target = reward + gamma * np.max(Q[next_state])

td_error = target - Q[state, action]

Q[state, action] += alpha * td_error
```

The educational value comes from exposing these quantities, not hiding them behind a generic optimizer framework.

Only introduce abstractions when they help multiple visualizations or prevent genuine duplication.

---

## 14. Visualization rules

### Rule 1 — Show estimates and reference values differently

For example:

```text
agent estimate     solid
exact value        dashed
```

Never allow a learner to confuse privileged visualization information with information used by the algorithm.

### Rule 2 — Every animation corresponds to a real computation

If the visualization shows

```text
Q = 0.30 → 0.35
```

the underlying algorithm must actually have performed that update.

Animations should explain computation rather than simulate the appearance of learning.

### Rule 3 — Keep environment steps and gradient steps separate

Always track both.

```text
environment steps:  12,340
learning steps:      4,821
episodes:               91
```

This becomes essential once replay buffers are introduced.

### Rule 4 — Allow pause, step, and replay

Important processes should support

```text
play
pause
single step
reset
replay
```

The user should be able to inspect one transition or one optimization step for as long as necessary.

### Rule 5 — Introduce one source of complexity at a time

For example:

```text
tabular Q-learning
        ↓
replay
        ↓
function approximation
        ↓
target network
        ↓
DQN
```

rather than introducing all DQN machinery simultaneously.

---

## 15. Lesson structure

Every lesson should follow approximately the same structure.

### Question

Start with a concrete question.

Example:

> Why can TD learn before an episode ends?

### Predict

Ask the learner to predict what will happen before running the experiment.

### Experiment

Allow the learner to control a minimal environment.

### Inspect

Show the relevant internal quantities.

### Equation

Connect the observed behavior to the mathematical update.

### Code

Show the minimal implementation.

### Challenge

End with a small interactive task.

Example:

> Without collecting another transition, can you make reward information propagate faster toward the start state?

The goal is active understanding rather than passive reading.

---

## 16. Development roadmap

### Phase 1 — Core tabular system

Implement:

```text
Bandit
Chain
GridWorld

Transition
Trajectory
LearningTrace
```

Algorithms:

```text
Value Iteration
Policy Iteration
Monte Carlo
TD(0)
SARSA
Q-learning
```

Visualizations:

```text
trajectory
reward timeline
value map
Q-value map
policy arrows
Bellman backup
learning trace
```

At this stage, there is no need for neural networks.

### Phase 2 — Replay and function approximation

Add:

```text
Replay Buffer
Linear Function Approximation
Small MLP
```

The central visualization becomes

```text
one sample
   ↓
one update
   ↓
predictions across many states change
```

### Phase 3 — Deep RL

Add:

```text
DQN
REINFORCE
Actor-Critic
PPO
```

Reference results against CleanRL and Stable-Baselines3.

### Phase 4 — Advanced modules

Potential additions:

```text
SAC
TD(λ)
GAE
Partial Observability
Offline RL
Goal-Conditioned RL
Model-Based RL
```

---

## 17. First milestone

The first usable version of Visual RL does not need PPO, SAC, MuJoCo, or Atari.

A successful first milestone contains only three environments:

```text
Bandit
Chain
GridWorld
```

and allows a learner to visually understand:

```text
reward
return
policy
trajectory
V
Q
Bellman backup
Monte Carlo
TD
SARSA
Q-learning
```

If these concepts can be understood interactively and precisely, the architecture is already successful.

Deep RL can then be built on top of the same visual language.

---

## 18. Long-term vision

Most reinforcement-learning resources offer one of two things:

```text
beautiful behavior demos
```

or

```text
correct algorithm implementations
```

Visual RL aims to connect them.

The long-term goal is that a learner can pause an RL system at any moment and trace

```text
this behavior
    ↑
this policy
    ↑
this parameter update
    ↑
this loss
    ↑
this target
    ↑
this piece of data
    ↑
this interaction with the environment
```

If the learner can follow that chain, reinforcement learning stops looking like an agent mysteriously becoming smarter after calling

```python
train()
```

and becomes a sequence of understandable computations.

That is the purpose of Visual RL.
