// Mirrors visualrl/algorithms/tabular/episode.py: act, then learn, after every environment step.
import { Trajectory, makeTransition } from "../core.js";

export function runEpisode(env, agent, { learn = true, maxSteps = 10000 } = {}) {
  const trajectory = new Trajectory();
  const traces = [];
  let [state] = env.reset();
  let action = agent.act(state);
  for (let step = 0; step < maxSteps; step++) {
    const [next_state, reward, terminated, truncatedByEnv] = env.step(action);
    const truncated = truncatedByEnv || (!terminated && step === maxSteps - 1);
    const next_action = terminated ? null : agent.act(next_state);
    const transition = makeTransition({ state, action, reward, next_state, terminated, truncated, next_action });
    trajectory.append(transition);
    if (learn) traces.push(agent.learnStep(transition));
    if (terminated || truncated) break;
    state = next_state;
    action = next_action;
  }
  return { trajectory, traces };
}
