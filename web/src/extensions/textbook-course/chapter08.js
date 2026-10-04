import { h, button, segmented, replace } from "../../ui/dom.js";
import { LinearValueLab } from "../../rl/teaching-labs.js";
import { metric, num } from "../../ui/world-view.js";
import { chapterHeader, chapterFooter, topic, prose, equation, table, predict } from "./shell.js";
import { dqnTricks } from "./value-tricks.js";
export function mount(root) {
  let lab = new LinearValueLab(), mode = "similar";
  const states = h("div", { class: "feature-cards" }), weights = h("div", { class: "physical-flow" }), calculation = h("p", { class: "arithmetic", role: "status" });
  const picker = segmented([{ value: "table", label: "独立表格" }, { value: "similar", label: "相似特征" }, { value: "shared", label: "完全共享" }], { value: mode, label: "价值表示", onChange: v => { mode = v; lab = new LinearValueLab(mode); render(); } });
  function render() {
    const values = lab.predictions();
    replace(states, values.map((v, i) => h("div", { class: "comparison-card" }, h("h3", {}, "状态 " + ["A", "B", "C"][i]),
      h("p", {}, "特征 x = (" + lab.X[i].join(", ") + ")"), metric("预测 V̂", num(v)),
      h("p", {}, i === 0 ? "观察到的终止奖励是1" : i === 2 ? "观察到的终止奖励是0" : "尚未提供这个状态的标签"))));
    replace(weights, h("span", {}, "输入状态 → 特征x"), h("b", {}, "→ 共用权重 w = (" + lab.w.map(v => num(v)).join(", ") + ") →"),
      h("span", {}, "点积给出各状态的价值"));
    const t = lab.last;
    calculation.textContent = t ? "只观察" + ["A", "B", "C"][t.state] + "：目标" + t.target + " − 旧预测" + num(t.before[t.state]) + " = 误差" + num(t.error) +
      "；新预测(A,B,C)=(" + t.after.map(v => num(v)).join(", ") + ")"
      : "先只更新A。相似特征下，A变成0.5，没观察过的B也会变成0.4；C仍为0。";
  }
  root.append(chapterHeader("08"),
    topic("08", "features", h("div", { class: "experiment" }, h("div", { class: "toolbar" }, picker,
      button("观察 A 的奖励1并更新", { kind: "learn", onClick: () => { lab.learn(0, 1); render(); } }),
      button("观察 C 的奖励0并更新", { onClick: () => { lab.learn(2, 0); render(); } }),
      button("权重清零", { kind: "ghost", onClick: () => { lab = new LinearValueLab(mode); render(); } })),
      h("div", { class: "experiment-content" }, states, weights, calculation)),
      prose("表格给每个状态一个独立数字；连续位置、图像或大量组合状态无法逐个存下。函数近似用共享参数，把状态输入映射成一个价值预测。共享让有限样本影响其他状态，这就是泛化。",
        "上面的三种表示分别是one-hot表格、相似特征和完全相同特征。切换表示会把参数清零，便于比较同一次更新。完全共享时A与C只能预测同一个值，即使任务要求A=1、C=0；这是表示能力的限制。"),
      equation("\\hat V_w(s)=w^\\top x(s),\\qquad \\Delta\\hat V(s')=\\alpha\\big[y-\\hat V(s)\\big]x(s)^\\top x(s')",
        "更新s影响s′的程度由特征内积决定。不是所有未观察状态都应该变好：相似性要对应任务中的价值关系。")),
    topic("08", "semigradient",
      prose("先把目标y当一个给定标签，平方误差的一次梯度下降就把预测推向y。上面的样本已经终止，y分别是1、0；因此没有自举，直接训练这个线性函数。"),
      equation("L(w)=\\tfrac12\\big[y-\\hat V_w(s)\\big]^2,\\qquad w\\leftarrow w+\\alpha\\big[y-\\hat V_w(s)\\big]\\nabla_w\\hat V_w(s)",
        "线性模型的梯度是x(s)。MC可令y=Gₜ；TD可令y=r+γV̂w(s′)，真正终止时y=r。"),
      prose("TD目标也依赖w。若更新中只对当前预测求导，把目标暂时视作固定，就叫半梯度；它不等于对完整Bellman残差平方直接求全梯度。这一区别关系到稳定性与我们实际优化的量。"),
      equation("w\\leftarrow w+\\alpha\\big[r+\\gamma\\hat V_w(s')-\\hat V_w(s)\\big]\\nabla_w\\hat V_w(s)",
        "半梯度TD：右侧先读取旧参数；真正终止遮掉后继预测。参数共享以后，一个更新会改变多个状态，表格中的独立误差分析不能直接照搬。"),
      predict({ question: "独立表格中更新A不会改变B；相似特征下会。哪一个原因正确？", choices: [{ label: "B与A的预测依赖部分相同的参数" }, { label: "环境偷偷给B也生成了一条真实经验" }], answer: 0, explain: "改变的是预测，不是B的经验计数。是否泛化正确，还要用真实任务结果检查。" })),
    topic("08", "dqn",
      prose("DQN把Q(s,a)表换成神经网络Qθ(s,a)。离散动作的每个输出是相应动作的价值；选动作仍可用ε-greedy。目标沿用第5章Q-learning，而不是因为用了网络就换掉控制目标。"),
      equation("y=r+\\gamma(1-d)\\max_{a'}Q_{\\theta^-}(s',a'),\\qquad L(\\theta)=\\mathbb E_{\\mathcal D}\\big[\\tfrac12(y-Q_\\theta(s,a))^2\\big]",
        "d表示真实终止。θ⁻是延迟更新的目标网络参数；训练当前Qθ时不对目标y求导。经验回放从过去转移中抽取训练样本。"),
      table(["组件", "为什么需要", "实际训练中检查什么"], [
        ["网络表示", "在大量状态之间共享预测能力", "特征是否保留任务信息；容量能否区分不同价值"],
        ["经验回放", "打散相邻样本，重复使用经验", "缓冲区覆盖、分布变化、终止标记"],
        ["目标网络", "降低目标随当前参数立即变化的程度", "同步频率、估值误差、真实回报"],
        ["探索与评估", "获得数据，并检查策略实际表现", "行为回报与贪心评估；多个随机种子"],
      ]),
      prose("函数近似、自举和off-policy共同构成deadly triad：组合后可能出现不稳定或发散。经验回放与目标网络是实用稳定化手段，不是对任意网络和数据的普遍收敛保证。",
        "本章交互运行的是可逐项验算的线性训练。DQN部分说明神经网络训练的目标和组件，没有把线性例子称为已完成的DQN训练。"),
      h("p", { class: "source-note" }, "论文阅读：", h("a", { href: "https://arxiv.org/abs/1312.5602", target: "_blank", rel: "noopener" }, "Mnih et al. · Playing Atari with Deep Reinforcement Learning"))),
    topic("08", "stability", ...dqnTricks()),
    chapterFooter("08", "共享参数获得泛化，也引入表示误差与更新耦合。DQN接入Q-learning目标，用回放与目标网络组织训练；Double DQN分开选择和评价，Huber与裁剪解决不同数值问题。先检查任务与目标，再用同预算、多种子和消融验收实际表现。"));
  render(); return () => {};
}
