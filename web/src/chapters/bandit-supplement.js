import { h, button, segmented, slider, replace } from "../ui/dom.js";
import { metric, num } from "../ui/world-view.js";
import { BanditLab } from "../rl/teaching-labs.js";
import { step, prose, equation, table, predict } from "./shell.js";
export function mount(root) {
  let lab = new BanditLab(), mode = "epsilon", epsilon = 0.1, reveal = false, constantAlpha = false;
  const cards = h("div", { class: "comparison bandit-cards" }), stats = h("div", { class: "metrics" }), calculation = h("p", { class: "arithmetic", role: "status" });
  function render() {
    replace(cards, [0, 1].map(a => h("div", { class: "comparison-card" }, h("h3", {}, "奖励来源 " + ["A", "B"][a]),
      h("p", {}, "每次奖励是0或1。真实成功率" + (reveal ? "为 " + lab.means[a] : "对学习器未知") + "。"),
      h("div", { class: "return-track" }, h("div", { class: "return-bar", style: { width: lab.Q[a] * 100 + "%" } })),
      h("div", { class: "metrics" }, metric("当前估计 Q(" + ["A", "B"][a] + ")", num(lab.Q[a])), metric("已尝试次数", lab.N[a])),
      button("手动试 " + ["A", "B"][a] + " 一次", { kind: "env", onClick: () => pull(a) }))));
    replace(stats, metric("真实尝试", lab.steps), metric("累计奖励", lab.totalReward), metric("当前贪心动作", lab.Q[0] === lab.Q[1] ? "A、B并列" : lab.Q[0] > lab.Q[1] ? "A" : "B"));
    calculation.textContent = lab.last
      ? ["A", "B"][lab.last.action] + " 收到 r=" + lab.last.reward + "；Q=" + num(lab.last.before) + " + " + num(lab.last.alpha) + " × (" + lab.last.reward + "−" + num(lab.last.before) + ") = " + num(lab.last.after)
      : "先试一个动作。Q=0是初始化，不是已经知道真实均值为0。";
  }
  function pull(action = lab.choose({ mode, epsilon })) { lab.pull(action, { constantAlpha: constantAlpha ? 0.1 : null }); render(); }
  const picker = segmented([{ value: "epsilon", label: "ε-greedy" }, { value: "ucb", label: "UCB" }], { value: mode, label: "选择规则", onChange: v => { mode = v; render(); } });
  const epsilonInput = slider({ id: "bandit-epsilon", label: "探索概率 ε", min: 0, max: 1, step: 0.05, value: epsilon, format: v => v.toFixed(2), onInput: v => { epsilon = v; } });
  root.append(
    step("赌博机：隔离未知奖励",
      h("div", { class: "experiment" }, h("div", { class: "experiment-content" }, cards, stats, calculation),
        h("div", { class: "toolbar" }, picker, epsilonInput,
          button("按当前规则试10次", { kind: "learn", onClick: () => { for (let i = 0; i < 10; i++) pull(); } }),
          button("显示/隐藏真实均值", { onClick: () => { reveal = !reveal; render(); } }),
          button("重置估计与随机种子", { kind: "ghost", onClick: () => { lab = new BanditLab(); render(); } }))),
      prose("赌博机是没有长期状态后果的单步问题：每次选一个奖励来源，只问平均奖励会有多大。它让我们先隔离「不知道哪个动作好」的困难，暂时不处理路线和延迟奖励。",
        "一个奖励1不等于该动作的期望是1。不同尝试会波动，需要用多次样本概括。这里A/B奖励为Bernoulli分布，真实均值0.35/0.75只用于对照，选动作的规则只读取Q与访问次数。"),
      equation("q_*(a)=\\mathbb E[R\\mid A=a],\\qquad Q_{n+1}(a)=Q_n(a)+\\frac{1}{N_{n+1}(a)}\\big[R_n-Q_n(a)\\big]",
        "α=1/N实现样本均值。只更新本次选中的动作，其他Q不变；这里Q是单步期望，和MDP里长期动作价值的任务不同。")),
    step("补充：UCB与乐观初始化",
      prose("利用是选择当前估计最好的动作；探索是给尚不确定的动作收集更多证据。当前估计最好不等于真实最好，所以纯贪心可能过早固定在一个动作上。多尝试也有即时机会成本。",
        "ε-greedy用ε的概率均匀探索，其余时间在贪心动作中选。UCB把少访问带来的不确定性加到动作分数中，优先检查有希望但证据不足的选项。"),
      equation("\\pi_\\epsilon(a)=\\frac{\\epsilon}{K}+(1-\\epsilon)\\frac{\\mathbf1\\{a\\in\\arg\\max_b Q(b)\\}}{|\\arg\\max_b Q(b)|}",
        "K为动作数；并列最好时均分贪心概率。ε控制获取新证据的频率，不会改变环境的真实奖励均值。"),
      equation("a_t\\in\\arg\\max_a\\left[Q_t(a)+c\\sqrt{\\frac{\\ln t}{N_t(a)}}\\right]",
        "UCB：未尝试动作先各试一次；本实验c=√2。第二项是探索奖励，不是环境反馈。"),
      h("div", { class: "toolbar" }, button("用乐观 Q=1 重新开始", { onClick: () => { lab = new BanditLab({ initialValue: 1 }); render(); } })),
      prose("乐观初始化把尚未尝试的动作估得较好，让一次低于预期的奖励把注意力转向别的动作。它能帮助初期探索，但不等于持续探索，也不等于真实奖励发生了变化。"),
      predict({ question: "把ε从0.1改为0.5，A/B的真实成功率会改变吗？", choices: [{ label: "不会；变的是机器人选动作的概率" }, { label: "会；环境会提高被探索动作的奖励" }], answer: 0, explain: "策略改变数据怎么来，环境仍按各自的奖励分布反馈。" })),
    step("补充：奖励变化与上下文",
      prose("如果奖励均值随时间变化，平均全部历史会给旧样本太多权重。常数α让近期样本持续产生影响；它不收敛成静态任务的精确样本均值，而是在噪声与追踪速度之间取舍。"),
      equation("Q_{n+1}=Q_n+\\alpha(R_n-Q_n)=(1-\\alpha)^{n}Q_1+\\alpha\\sum_{i=1}^{n}(1-\\alpha)^{n-i}R_i",
        "0<α≤1时，越旧的奖励权重越小。切换估计方式会清零计数与估计；随后交换真实均值时保留旧估计，可以观察两种方式怎样追踪变化。"),
      h("div", { class: "toolbar" },
        button("交换真实均值，保留估计", { onClick: () => { lab.means.reverse(); reveal = true; render(); } }),
        button("切换均值/α=0.1并清零", { onClick: () => { const means = lab.means.slice(); constantAlpha = !constantAlpha; lab = new BanditLab({ means }); render(); calculation.textContent = "估计与计数已清零，真实均值保留；下一次更新使用" + (constantAlpha ? "常数α=0.1" : "α=1/N（样本均值）"); } })),
      table(["情况", "该估计什么", "还缺什么"], [
        ["平稳bandit", "每个动作一个固定均值", "充分探索、样本计数"],
        ["非平稳bandit", "当前动作均值", "持续探索、近期权重"],
        ["Contextual bandit", "给定输入x时各动作的单步价值", "用x条件化估计；仍不处理长期状态后果"],
        ["一般MDP", "动作改变后续状态与累计回报", "回到本章的控制方法：目标保留后继价值"],
      ])));
  render(); return () => {};
}
