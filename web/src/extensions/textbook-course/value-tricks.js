import { h, button, segmented, replace } from "../../ui/dom.js";
import { GridWorld } from "../../rl/envs/gridworld.js";
import { route, NEAR_ROUTE, FAR_ROUTE, metric, num } from "../../ui/world-view.js";
import { shapeRewards, doubleDqnTarget } from "../../rl/policy-math.js";
import { prose, equation, table } from "./shell.js";
import { proof } from "./policy-derivations.js";
import { readings } from "./sources.js";

export function rewardShaping() {
  const env = GridWorld.chargingRoom(), near = route(env, NEAR_ROUTE), far = route(env, FAR_ROUTE);
  let terminalPotential = 0;
  const stats = h("div", { class: "metrics" }), rows = h("div");
  function potential(state) {
    if (env.goals.includes(state)) return terminalPotential;
    const [row, col] = env.toCell(state);
    return -Math.min(...env.goals.map(g => { const [r, c] = env.toCell(g); return Math.abs(row - r) + Math.abs(col - c); }));
  }
  function shaped(records) {
    return shapeRewards(records.map(t => t.reward), [potential(records[0].state), ...records.map(t => potential(t.next_state))]);
  }
  function render() {
    const a = shaped(near), b = shaped(far);
    replace(stats, metric("近路原G → 塑形G", num(a.originalReturn) + " → " + num(a.shapedReturn)),
      metric("远路原G → 塑形G", num(b.originalReturn) + " → " + num(b.shapedReturn)),
      metric("近/远各增加", num(a.boundary) + " / " + num(b.boundary)),
      metric("塑形后两候选比较", a.shapedReturn > b.shapedReturn ? "近路更好" : "远路更好"));
    replace(rows, table(["近路转移", "原r", "Φ当前", "Φ后继", "r′=r+γΦ后继−Φ当前"], near.map((t, i) => [
      i + 1, t.reward, potential(t.state), potential(t.next_state), num(a.shaped[i]),
    ])));
  }
  render();
  return [
    prose("奖励稀疏时，希望走到终点前也得到辅助反馈。但直接奖励「离目标更近」可能改变策略偏好。势函数塑形用一个精确的回报抵消关系，限定哪些辅助奖励保持原任务。",
      "本例给每个位置同一个Φ：到最近充电站的负Manhattan距离；两条路线在共享状态使用同一数值。它只是辅助信号，不要求等于真实价值或考虑墙的最短距离。"),
    equation("r'_t=r_t+\\gamma\\Phi(s_{t+1})-\\Phi(s_t)",
      "同一个任务折扣γ；Φ不能为每条路线分别定义。终止后按回合结束处理时，所有终点Φ设为0。"),
    proof("望远镜求和", "中间势能消去，留下边界项",
      equation("G'_0=G_0+\\sum_{t=0}^{T-1}\\gamma^t[\\gamma\\Phi(s_{t+1})-\\Phi(s_t)]=G_0-\\Phi(s_0)+\\gamma^T\\Phi(s_T)",
        "下一状态的正项与下一步当前状态的负项抵消。终点Φ=0时，从同一S出发的所有轨迹都增加−Φ(S)，策略排序不变。"),
      equation("Q'^\\pi(s,a)=Q^\\pi(s,a)-\\Phi(s)",
        "对应的动作价值只减同一状态常数。无限折扣任务要求γ<1、Φ有界，使尾项消失；任意截断的非零尾项不能直接丢弃。")),
    h("div", { class: "experiment" }, h("div", { class: "toolbar" },
      button("正确设置：终点 Φ=0", { onClick: () => { terminalPotential = 0; render(); } }),
      button("反例：结束回合却设终点 Φ=30", { onClick: () => { terminalPotential = 30; render(); } })),
      h("div", { class: "experiment-content" }, stats, rows)),
    prose("正确设置下Φ(S)=−3，近路0.656→3.656、远路4.783→7.783，两者都加3。错误设置终点30并直接结束回合，γᵀ×30因路线长度不同而不同，近路变得优于远路；这已经改变了目标。",
      "势函数塑形给出的辅助奖励不必每步为正。把每个正奖励直接裁成+1则是另一个操作：我们的+1与+10变相同，近路反而优于远路。第8章会继续区分改变奖励、改变损失与改变优化步。"),
    readings("shaping"),
  ];
}

export function dqnTricks() {
  let mode = "disagree", terminated = false;
  const stats = h("div", { class: "metrics" }), values = h("p", { class: "arithmetic" });
  function render() {
    const online = mode === "disagree" ? [8, 7] : [1, 7], target = [2, 6];
    const t = doubleDqnTarget(online, target, { terminated });
    replace(stats, metric("在线网络选动作", t.action === 0 ? "A" : "B"), metric("DQN目标", num(t.dqn)), metric("Double DQN目标", num(t.double)));
    values.textContent = "给定在线Q=(" + online + ")、目标Q=(2,6)、r=1、γ=0.9；" + (terminated ? "真正终止，两者只用r。" : "两个网络都是带误差的估计，不是假定的真实Q。");
  }
  const picker = segmented([{ value: "disagree", label: "两个网络排序不同" }, { value: "agree", label: "两个网络同意选B" }], { value: mode, label: "网络估计", onChange: v => { mode = v; render(); } });
  render();
  return [
    proof("凸函数Jensen", "max为什么能把无偏噪声变成高估？",
      equation("\\mathbb E\\max_a\\hat Q(a)\\ge\\max_a\\mathbb E\\hat Q(a)",
        "max是凸函数。每个动作估计都无偏，不代表挑最大的以后仍无偏。凹函数log的Jensen会在第9章反向使用。"),
      table(["两个动作真实Q都0，独立噪声各为±1", "max"], [["(+1,+1)", 1], ["(+1,−1)", 1], ["(−1,+1)", 1], ["(−1,−1)", -1]]),
      prose("四种结果等概率：每个动作均值0，max均值0.5。选择和估计共享同一份噪声，容易选中正误差。第5章Double Q的思想在网络版本中得到Double DQN。")),
    proof("算法改动", "Double DQN：在线选动作，目标网络评价",
      equation("y=r+\\gamma(1-d)Q_{\\theta^-}(s',\\arg\\max_a Q_\\theta(s',a))",
        "普通DQN直接在目标网络取max；Double DQN分开选择与评价。它减少一种最大化偏差来源，不保证每次都更接近真值。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, picker, button("切换真正终止/继续", { onClick: () => { terminated = !terminated; render(); } })),
        h("div", { class: "experiment-content" }, values, stats))),
    table(["常用工具", "它解决什么", "怎样使用与验收"], [
      ["经验回放", "相邻样本相关、经验只用一次", "随机抽样；记录环境步与更新步；检查覆盖与陈旧数据"],
      ["冻结目标＋stop-gradient", "预测和目标同时追着彼此变化", "目标参数延迟同步；y不参与反传；检查线上/目标差距"],
      ["软更新", "让目标缓慢跟踪", "θ⁻←τθ+(1−τ)θ⁻，0<τ≤1；这是参数平均，不是折扣γ"],
      ["Huber loss", "极大TD残差支配平方损失", "小残差平方，大残差线性；只控制残差的loss斜率，不保证全网络梯度范数有界"],
      ["梯度范数裁剪", "极大参数梯度带来过大一步", "记录触发比例；不能修复错误终止、坏目标或缺失状态"],
      ["奖励裁剪", "限制奖励数值范围", "会改变任务：+1/+10都裁为+1后，近路0.656>远路0.478；不同于第2章策略不变的势函数塑形"],
      ["评估与消融", "loss下降可能掩盖行为变差", "同交互预算、多种子；分别移除回放/目标等组件，报告成功率和回报"],
    ]),
    equation("L_\\kappa(e)=\\begin{cases}\\tfrac12 e^2,&|e|\\le\\kappa\\\\ \\kappa(|e|-\\tfrac12\\kappa),&|e|>\\kappa\\end{cases},\\qquad e=y-Q_\\theta(s,a)",
      "Huber的分段技巧把残差大小和loss影响分开。不是所有技巧都应无条件启用；先确认任务定义和目标正确。"),
    readings("rainbow", "dqn", "evaluation", "time"),
  ];
}
