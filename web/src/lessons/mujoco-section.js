// The MuJoCo part of a lesson: pick a robot, replay recorded episodes of its policy at several
// points in training, and read the real training log next to them.
import { MUJOCO_TASKS } from "../data/mujoco/index.js";
import { h, replace, segmented } from "../ui/dom.js";
import { fmt } from "../ui/format.js";
import { LineChart } from "../ui/line-chart.js";
import { MujocoReplay } from "../ui/mujoco-view.js";

const compact = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n));

export function mujocoSection() {
  let replay = null;
  const slot = h("div");
  const about = h("p", { class: "note" });
  const returns = new LineChart({ height: 220, xLabel: "environment steps", yLabel: "average return of the training episodes", xFormat: compact });
  const clip = new LineChart({ height: 220, xLabel: "environment steps", yLabel: "share of samples clipped (last epoch)", xFormat: compact, yFormat: (v) => `${Math.round(v * 100)}%` });
  const config = h("p", { class: "note" });

  function show(id) {
    const task = MUJOCO_TASKS.find((t) => t.id === id);
    replay?.dispose();
    replay = new MujocoReplay(task.data);
    replace(slot, replay.el);
    const d = task.data;
    const log = d.training;
    about.textContent = `${task.title}: ${task.robot}. Observations have ${d.obs_dim} numbers, actions ${d.act_dim} continuous torques.`;
    returns.update({
      series: [{ id: "ret", label: "return", color: "var(--series-2)", points: log.env_steps.map((x, i) => [x, log.return[i]]).filter((p) => p[1] !== null) }],
      points: d.checkpoints.filter((c) => c.env_steps > 0).map((c) => ({ x: c.env_steps, y: nearest(log, c.env_steps), label: `replay ${compact(c.env_steps)}`, color: "var(--ink)" })),
    });
    clip.update({ series: [{ id: "clip", label: "clip fraction", color: "var(--series-2)", points: log.env_steps.map((x, i) => [x, log.clip_fraction[i]]) }] });
    const c = d.config;
    config.textContent = `Trained with visualrl.algorithms.deep.ppo for ${compact(c.total_steps)} steps (seed ${c.seed}): rollouts of ${c.n_steps} steps, ${c.epochs} epochs of ${c.minibatches} minibatches, Adam with learning rate ${c.lr}, ε = ${c.clip}, γ = ${c.gamma}, λ = ${c.lam}. Largest approximate KL in any iteration: ${fmt(Math.max(...log.approx_kl), 3)}.`;
  }

  const picker = segmented(
    MUJOCO_TASKS.map((t) => ({ value: t.id, label: t.title })),
    { value: MUJOCO_TASKS[0].id, label: "Robot", onChange: show },
  );
  const el = h(
    "div",
    { class: "bench" },
    h("div", { class: "task-picker" }, h("span", { class: "panel-title" }, "Robot"), picker),
    about,
    slot,
    h("div", { class: "two-col" }, returns.el, clip.el),
    config,
  );
  show(MUJOCO_TASKS[0].id);
  return { el, dispose: () => replay?.dispose() };
}

function nearest(log, steps) {
  let best = 0;
  log.env_steps.forEach((x, i) => {
    if (Math.abs(x - steps) < Math.abs(log.env_steps[best] - steps)) best = i;
  });
  for (let i = best; i >= 0; i--) if (log.return[i] !== null) return log.return[i];
  return 0;
}
