import qSource from "../../../visualrl/algorithms/tabular/q_learning.py";
import sarsaSource from "../../../visualrl/algorithms/tabular/sarsa.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { Sarsa, QLearning } from "../rl/tabular/q-agents.js";
import { runEpisode } from "../rl/tabular/episode.js";
import { h, button, replace, segmented } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, ARROWS, num, metric } from "../ui/world-view.js";
import { codeBlock } from "../ui/code.js";
import { extractDef } from "../ui/source.js";

export function greedyStates(env, Q) {
  let state = env.start; const states = [state], seen = new Set(states);
  for (let i = 0; i < 100; i++) {
    const a = Q[state].indexOf(Math.max(...Q[state])); const [next, , done] = env.move(state, a);
    states.push(next); if (done || seen.has(next)) break; seen.add(next); state = next;
  }
  return states;
}
export function trainComparison(epsilon = 0.1) {
  return [Sarsa, QLearning].map((Agent) => {
    const env = GridWorld.warehouse(), agent = new Agent({ nStates: env.nStates, nActions: 4, alpha: 0.5, gamma: 1, epsilon, seed: 7 });
    let falls = 0; const returns = [];
    for (let i = 0; i < 500; i++) {
      const { trajectory } = runEpisode(env, agent, { maxSteps: 2000 });
      falls += trajectory.rewards.filter((r) => r === -100).length;
      returns.push(trajectory.rewards.reduce((sum, r) => sum + r, 0));
    }
    return { env, agent, falls, mean: returns.slice(-50).reduce((a, b) => a + b, 0) / 50, path: greedyStates(env, agent.Q) };
  });
}
export function mount(root) {
  const env = GridWorld.warehouse(), state = env.toState(3, 2), next = env.toState(3, 3);
  let actual = 2, traces = null;
  const map = new WorldView(env, { caption: "换一张更长的悬崖地图：每走一步 −1，跌落 −100 并回起点，到右侧终点结束。实线是选中的下一动作；虚线是当前 Q 最大的下一动作。" });
  const qTable = h("div", { class: "table-wrap" }), results = h("div", { class: "comparison" });
  const note = h("p", { class: "observation", role: "status" });
  const learnBtn = button("用这条经验分别更新一次", { kind: "learn", onClick: () => {
    traces = [Sarsa, QLearning].map((Agent) => {
      const agent = new Agent({ nStates: env.nStates, nActions: 4, alpha: 0.5, gamma: 1 });
      agent.Q[state][1] = -20; agent.Q[next] = [-12, -10, -100, -14];
      return agent.learnStep({ state, action: 1, reward: -1, next_state: next, terminated: false, truncated: false, next_action: actual });
    }); render();
  } });
  function render() {
    const [actualEnd] = env.move(next, actual);
    const drawnEnd = actual === 2 ? env.toState(4, 3) : actualEnd;
    map.render({ robot: next, labels: { [state]: "原状态", [next]: "下一状态" }, paths: [
      { states: [state, next, next + 1], color: "var(--learn)", dashed: true },
      { states: [state, next, drawnEnd], color: "var(--act)" },
    ] });
    const qs = [-12, -10, -100, -14];
    replace(qTable, h("h3", {}, "下一状态已有的动作价值估计"), h("table", {}, h("thead", {}, h("tr", {}, ["动作", "Q(s′,a′)", "会被谁使用"].map((x) => h("th", {}, x)))), h("tbody", {}, qs.map((q, a) => h("tr", { class: a === actual || a === 1 ? "highlight" : "" }, h("td", {}, ARROWS[a]), h("td", {}, q), h("td", {}, [a === actual ? "SARSA" : "", a === 1 ? "Q-learning" : ""].filter(Boolean).join("、") || "—"))))));
    replace(results, ["SARSA", "Q-learning"].map((name, i) => {
      const t = traces?.[i], bootstrap = i === 0 ? qs[actual] : -10;
      return h("div", { class: "comparison-card" }, h("h3", {}, name), h("p", {}, i === 0 ? "接上实际选中的下一动作。" : "接上 Q 最大的下一动作。"), h("div", { class: "arithmetic" }, `目标 = −1 + ${bootstrap} = ${-1 + bootstrap}`), metric("这一步的 Q(s,→)", t ? `${num(t.value_before)} → ${num(t.value_after)}` : "−20 · 等待更新"));
    }));
    note.textContent = actual === 2 ? "实际下一动作是向下探索，可能跌落。SARSA 的目标包含这次实际选择；Q-learning 仍假设下一步选估计最好的向右。当前经验的真实奖励只有 −1，−100 是预先给定的后续 Q 估计。" : "实际动作恰好也是估计最好的向右，两种目标相同。区别来自目标里使用哪个下一动作，不是两个算法收到不同的当前奖励。";
  }
  const picker = segmented([{ value: 2, label: "下一步向下探索 · 跌落" }, { value: 1, label: "下一步向右 · 当前最优" }], { value: actual, label: "已选择的下一动作", onChange: (v) => { actual = v; traces = null; render(); } });
  const training = h("div", { class: "comparison" });
  function train(epsilon) {
    const runs = trainComparison(epsilon);
    replace(training, runs.map((r, i) => {
      const view = new WorldView(r.env, { caption: `${i === 0 ? "SARSA" : "Q-learning"}：训练后不再随机探索的贪心路线。` });
      view.render({ paths: [{ states: r.path, color: i === 0 ? "var(--act)" : "var(--learn)" }], robot: r.env.start });
      const reached = r.env.goals.includes(r.path.at(-1));
      return h("div", { class: "comparison-card" }, h("h3", {}, i === 0 ? "SARSA" : "Q-learning"), view.el,
        h("div", { class: "metrics" }, metric("500 回合跌落", r.falls), metric("末 50 回合平均回报", num(r.mean, 1)), metric("贪心路线", reached ? `${r.path.length - 1} 步到终点` : "尚未到达终点")), h("p", {}, `本次固定种子为 7，探索概率 ε = ${epsilon}。训练回报包含随机探索时的跌落；图中路线只显示训练后的贪心动作。`));
    }));
  }
  root.append(lessonHeader("08"), step("先看图像：实际的未来，与理想的未来", prose("现在要学的是动作价值 Q：先做这个动作，然后继续行动，预计能得到多少回报。探索会偶尔偏离当前最好的动作；在悬崖旁，这次偏离可能非常贵。", "下面先隔离一次更新，使用人为给定的旧 Q，便于看清差异。这些数字是演示起点，不是训练结果。两种方法使用完全相同的 (s,a,r,s′)，只改变目标里用哪个下一动作；γ = 1，α = 0.5。")),
    step("动手验证：只改变下一动作", h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, "先选向下，再选向右"), "观察地图中两条未来分支，以及两种目标的差别。更新一次后，看同一个旧 Q 怎样得到两个新估计。"), h("div", { class: "toolbar" }, picker, learnBtn), h("div", { class: "experiment-content" }, map.el, qTable, results, note))),
    takeaway("SARSA 把实际选择的下一动作接进目标；Q-learning 把估计最好的下一动作接进目标。持续探索时，SARSA 会学到探索本身带来的风险，可能选择更安全的绕行路线。"),
    step("检查理解", predict({ question: "实际下一动作就是 Q 最大的动作时，SARSA 和 Q-learning 的这次目标有区别吗？", choices: [{ label: "没有，此时它们使用相同的下一项" }, { label: "有，它们的每次目标都必须不同" }], answer: 0, explain: "它们只在下一动作项不同。实际动作与最佳动作一致时，这次目标一致。" })),
    optional("展开：真正训练 500 回合，比较最终路线", prose("这一实验从全零 Q 开始运行真实算法。每回合边探索边更新；结果由当前运行计算，单个种子不是统计结论。"), h("div", { class: "toolbar" }, button("ε = 0.1：训练并比较", { onClick: () => train(0.1) }), button("ε = 0：再比较", { onClick: () => train(0) })), training),
    optional("展开：只看两行更新的差异", codeBlock(extractDef(sarsaSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "Sarsa.learn_step" }), codeBlock(extractDef(qSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "QLearning.learn_step" })), lessonFooter("08"));
  render(); return () => {};
}
