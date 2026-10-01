import qSource from "../../../visualrl/algorithms/tabular/q_learning.py";
import sarsaSource from "../../../visualrl/algorithms/tabular/sarsa.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { Sarsa, QLearning } from "../rl/tabular/q-agents.js";
import { runEpisode } from "../rl/tabular/episode.js";
import { h, button, replace, segmented } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, ARROWS, num, metric } from "../ui/world-view.js";
import { equation } from "../ui/math.js";
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
function newRun(Agent, epsilon) {
  const env = GridWorld.warehouse(), agent = new Agent({ nStates: env.nStates, nActions: 4, alpha: 0.5, gamma: 1, epsilon, seed: 7 });
  return { env, agent, falls: 0, returns: [] };
}
function trainOne(run) {
  const { trajectory } = runEpisode(run.env, run.agent, { maxSteps: 2000 });
  run.falls += trajectory.rewards.filter(r => r === -100).length;
  run.returns.push(trajectory.rewards.reduce((sum, r) => sum + r, 0));
}
function finish(run) { return { ...run, mean: run.returns.slice(-50).reduce((a, b) => a + b, 0) / 50, path: greedyStates(run.env, run.agent.Q) }; }
export function trainComparison(epsilon = 0.1) {
  return [Sarsa, QLearning].map(Agent => { const run = newRun(Agent, epsilon); for (let i = 0; i < 500; i++) trainOne(run); return finish(run); });
}
export function mount(root) {
  const env = GridWorld.warehouse(), qs = [-12, -10, -100, -14];
  let actual = 2, terminal = false, agents, traces = null, disposed = false, running = false;
  const map = new WorldView(env, { caption: "每步 −1；× 是悬崖，进入时 −100 并回 S。实线是实际下一动作，蓝虚线是 Q 最大的下一动作；细红虚线表示跌落后回到 S。" });
  const qTable = h("div", { class: "table-wrap" }), results = h("div", { class: "comparison" }), note = h("p", { class: "observation", role: "status" });
  const transition = () => ({ state: env.toState(3, terminal ? 12 : 2), action: terminal ? 2 : 1, reward: -1, next_state: terminal ? env.goals[0] : env.toState(3, 3), terminated: terminal, truncated: false, next_action: terminal ? null : actual });
  function reset() {
    const t = transition(); agents = [Sarsa, QLearning].map(Agent => {
      const agent = new Agent({ nStates: env.nStates, nActions: 4, alpha: 0.5, gamma: 1 });
      agent.Q[t.state][t.action] = -20; if (!terminal) agent.Q[t.next_state] = [...qs]; return agent;
    }); traces = null; render();
  }
  const learnBtn = button("用这条经验分别更新一次", { kind: "learn", onClick: () => { traces = agents.map(agent => agent.learnStep(transition())); render(); } });
  function render() {
    const t = transition();
    const paths = terminal ? [{ states: [t.state, t.next_state], color: "var(--act)" }] : [
      { states: [t.state, t.next_state, t.next_state + 1], color: "var(--learn)", dashed: true },
      { states: [t.state, t.next_state, actual === 2 ? env.toState(4, 3) : t.next_state + 1], color: "var(--act)" },
      ...(actual === 2 ? [{ states: [env.toState(4, 3), env.start], color: "var(--hazard)", dashed: true, width: 2 }] : []),
    ];
    map.render({ robot: t.next_state, labels: { [t.state]: "s · 当前", [t.next_state]: "s′ · 后继" }, paths });
    picker.querySelectorAll("button").forEach(b => { b.disabled = terminal; });
    replace(qTable, terminal ? h("p", { class: "observation" }, "这条经验：从终点上方向下，r = −1，done = true。已经终止，不再选 a′，也不读取终点的 Q。") : h("div", {}, h("h3", {}, "给定的后继 Q(s′,a′) · 本演示中保持不变"), h("table", {}, h("thead", {}, h("tr", {}, ["动作", "当前估计", "进入谁的目标"].map(x => h("th", {}, x)))), h("tbody", {}, qs.map((q, a) => h("tr", { class: a === actual || a === 1 ? "highlight" : "" }, h("td", {}, ARROWS[a]), h("td", {}, q), h("td", {}, [a === actual ? "SARSA" : "", a === 1 ? "Q-learning" : ""].filter(Boolean).join("、") || "—")))))));
    replace(results, ["SARSA", "Q-learning"].map((name, i) => {
      const trace = traces?.[i], bootstrap = terminal ? 0 : i === 0 ? qs[actual] : -10, current = agents[i].Q[t.state][t.action];
      return h("section", { class: "comparison-card" }, h("h3", {}, name), h("p", {}, terminal ? "终止：后续项为 0。" : i === 0 ? "使用实际选中的下一动作 a′。" : "使用估计最好的下一动作。"),
        h("div", { class: "arithmetic" }, `目标 y = −1 + 1 × ${bootstrap} = ${-1 + bootstrap}`, trace ? h("p", {}, `误差 δ = ${num(trace.error)}；新 Q = ${num(trace.value_before)} + 0.5 × (${num(trace.error)}) = ${num(trace.value_after)}`) : null),
        metric("Q(s,a) · 旧值 → 新值", trace ? `${num(trace.value_before)} → ${num(current)}` : "−20 · 等待更新"), metric("累计更新次数", agents[i].learnSteps));
    }));
    note.textContent = terminal ? "两种目标都等于当前奖励 −1，连续更新都朝 −1 靠近。切换经验类型或下一动作，会重置这个算例到 Q = −20；重复点击更新则沿用上次结果。" : actual === 2 ? "当前已发生经验的奖励只有 −1。向下是已选择的后续探索动作，−100 是人为给定的后继 Q 估计，不是把两次即时奖励合在一起。连续更新时，SARSA 依次 −20→−60.5→−80.75，Q-learning 依次 −20→−15.5→−13.25。" : "实际 a′ 也是估计最好的向右，两种目标都是 −11。这里的后继 Q 固定，所以重复更新会向同一目标靠近；真实训练中，其他状态的 Q 也在变化。";
  }
  const picker = segmented([{ value: 2, label: "下一步向下探索 · 跌落" }, { value: 1, label: "下一步向右 · 当前最优" }], { value: actual, label: "已选择的下一动作", onChange: v => { actual = v; reset(); } });
  const casePicker = segmented([{ value: false, label: "非终止经验 · 比较下一动作" }, { value: true, label: "终止经验 · 后续项为 0" }], { value: terminal, label: "经验类型（切换会重置算例）", onChange: v => { terminal = v; reset(); } });
  const training = h("div", { class: "comparison" }), progress = h("p", { class: "observation", role: "status" });
  const trainButtons = [button("ε = 0.1：训练并比较", { onClick: () => train(0.1) }), button("ε = 0：再比较", { onClick: () => train(0) })];
  async function train(epsilon) {
    if (running || disposed) return; running = true; trainButtons.forEach(b => { b.disabled = true; }); training.setAttribute("aria-busy", "true"); replace(training);
    const runs = [Sarsa, QLearning].map(Agent => newRun(Agent, epsilon));
    try {
      for (let i = 0; i < 500; i += 5) {
        if (disposed) return;
        for (let j = 0; j < 5; j++) runs.forEach(trainOne);
        progress.textContent = `训练中：两个算法各 ${i + 5} / 500 回合，ε = ${epsilon}。`;
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      if (disposed) return;
      replace(training, runs.map(finish).map((r, i) => {
        const view = new WorldView(r.env, { caption: `${i === 0 ? "SARSA" : "Q-learning"}：训练后关闭随机探索的贪心路线。` });
        view.render({ paths: [{ states: r.path, color: i === 0 ? "var(--act)" : "var(--learn)" }], robot: r.env.start });
        return h("section", { class: "comparison-card" }, h("h3", {}, i === 0 ? "SARSA" : "Q-learning"), view.el, h("div", { class: "metrics" }, metric("500 回合跌落", r.falls), metric("末 50 回合平均回报", num(r.mean, 1)), metric("贪心路线", r.env.goals.includes(r.path.at(-1)) ? `${r.path.length - 1} 步到终点` : "尚未到达终点")));
      })); progress.textContent = `完成：固定种子 7，α = 0.5，γ = 1，ε = ${epsilon}。回报统计包含训练时的探索；地图只显示最终贪心路线。`;
    } catch (error) { if (!disposed) progress.textContent = `这次训练未完成：${error.message}`; }
    finally { running = false; training.setAttribute("aria-busy", "false"); trainButtons.forEach(b => { b.disabled = false; }); }
  }
  root.append(lessonHeader("08"),
    step("先看图像：实际的未来，与理想的未来", prose("前一章评价固定路线的 V，现在学习 Q(s,a) 并用它改进选动作。仓库每步 −1，跌落 −100 并回到 S；终点在右下角。更短的路线靠近悬崖，偶尔的探索动作可能很贵。", "先隔离一次更新：两种方法用同一 (s,a,r,s′)，旧 Q(s,→) = −20，γ = 1，α = 0.5。后继 Q 是人为给定的演示估计，固定不变；下方训练实验才从全零 Q 运行算法。")),
    step("公式：差异只在下一动作",
      equation("Q(s_t,a_t)\\leftarrow Q(s_t,a_t)+\\alpha[r_t+\\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]", "SARSA：五个量 S、A、R、S′、A′。aₜ₊₁ 是行为策略实际选出的下一动作，不是重新求 max。"),
      equation("Q(s_t,a_t)\\leftarrow Q(s_t,a_t)+\\alpha[r_t+\\gamma\\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)]", "Q-learning：下一状态使用当前估计最大 Q。当前经验的动作仍可来自探索，目标假设后续贪心选择。"),
      prose("两式都是旧 Q + α×TD 误差；如果经验真正终止，两个后续项都为 0。示例实际向下时：SARSA 目标 −1 + (−100) = −101，新 Q = −20 + 0.5×(−81) = −60.5；Q-learning 目标 −1 + (−10) = −11，新 Q = −15.5。若 a′ 恰好最大，目标相同。")),
    step("动手验证：同一条经验，两个目标", h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, "向下 → 连续更新两次 → 向右 → 终止经验"), "观察同一个 Q 如何持续向目标靠近。切换算例会重置；重新点击更新不会重置。"), h("div", { class: "toolbar" }, casePicker, picker, learnBtn, button("重置这个算例", { kind: "ghost", onClick: reset })), h("div", { class: "experiment-content" }, map.el, qTable, results, note))),
    step("为什么一个绕远，一个靠近悬崖？", equation("\\pi_{\\varepsilon}(a\\mid s)=\\frac{\\varepsilon}{|\\mathcal{A}|}+(1-\\varepsilon)\\mathbf{1}[a=a^*(s)]", "ε-greedy：有唯一最大动作 a* 时，以 1−ε 选它，以 ε 在全部动作中均匀探索；并列最大动作时，本实现均分利用概率。ε = 0.1、4 个动作时，向下仍至少有 0.025 的概率。"),
      prose("行为策略是实际收集数据时怎样选动作；目标策略是我们希望估计、改进的策略。SARSA 在目标里使用行为策略选出的 a′，因此是 on-policy。持续探索时，它估计的未来也包含偶尔的危险动作，可能偏好离悬崖更远的路线。", "Q-learning 的经验可由 ε-greedy 收集，但目标使用贪心 max，所以是 off-policy。即使学到了很好的贪心近路，训练时仍可能因探索跌落；不能用最终贪心路线代替实际训练回报。两种算法都可以边行动边学习，on/off-policy 不表示在线/离线。", "下面的 500 回合只是一次有限运行，常数学习率与单个随机种子不是收敛证明。表格型 Q-learning 的经典收敛结果还要求足够访问状态动作、合适的递减学习率等条件。SARSA 在持续 ε 探索时与逐步减少 ε 时也会评价不同的行为。")),
    optional("展开：真正训练 500 回合，比较最终路线", prose("每 5 回合让页面处理一次输入，进度持续更新。切换章节会停止本次实验；重新训练从全零 Q 开始。每回合最多 2000 步，达到上限是截断而非环境终止。"), h("div", { class: "toolbar" }, trainButtons), progress, training),
    takeaway("SARSA 接实际 a′，Q-learning 接 max Q；探索风险因此进入不同的目标。每次更新都必须使用上一估计，真正终止时都只剩当前奖励。"),
    step("检查理解", predict({ question: "实际下一动作已经是 Q 最大的动作，两种算法的这次目标有区别吗？", choices: [{ label: "没有，后续项相同" }, { label: "有，它们每次都必须不同" }], answer: 0, explain: "差异只在后续动作的选择；实际 a′ 与最大动作一致时，目标相同。" })),
    optional("展开：对照实现", codeBlock(extractDef(sarsaSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "Sarsa.learn_step" }), codeBlock(extractDef(qSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "QLearning.learn_step" })), lessonFooter("08"));
  reset(); return () => { disposed = true; };
}
