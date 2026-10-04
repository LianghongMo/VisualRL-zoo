import { h, button, slider, segmented, replace } from "../../ui/dom.js";
import { metric, num } from "../../ui/world-view.js";
import { baselineStatistics, generalizedAdvantages, surrogateExample } from "../../rl/policy-math.js";
import { prose, equation, table } from "./shell.js";
import { readings } from "./sources.js";
export const proof = (kind, title, ...body) => h("section", { class: "proof-step" },
  h("p", { class: "proof-kind" }, kind), h("h3", {}, title), ...body);

export function trajectoryDerivation() {
  return [
    prose("上一节已经看见概率怎样改变。本节回答：为什么更新式里会有logπ？为什么用回报加权？为什么可以删掉过去奖励？每个等号都要有理由。",
      "先固定条件：ρ₀、环境转移和奖励机制不依赖θ；策略可微且在所用动作上有正概率，允许交换求导与求和。先用有限时域T，终止后补零奖励；有限时域的剩余时间属于状态。"),
    proof("定义", "1 · 先写轨迹发生的概率",
      equation("P_\\theta(\\tau)=\\rho_0(s_0)\\prod_{t=0}^{T-1}\\pi_\\theta(a_t\\mid s_t)p(s_{t+1},r_t\\mid s_t,a_t)",
        "轨迹τ是一整串状态、动作与奖励。策略决定选哪个动作，环境决定这个动作产生哪种后果。"),
      equation("R_\\gamma(\\tau)=\\sum_{t=0}^{T-1}\\gamma^t r_t,\\qquad J_\\gamma(\\theta)=\\sum_\\tau P_\\theta(\\tau)R_\\gamma(\\tau)",
        "同一任务的回报定义不变。参数改变的是每条轨迹出现的概率。连续变量时，把求和改为积分。")),
    proof("微积分恒等式", "2 · log-derivative：把概率的导数移进期望",
      equation("\\nabla_\\theta P_\\theta=P_\\theta\\nabla_\\theta\\log P_\\theta",
        "因为∇logP=(∇P)/P。这是写log的原因；不是在最大化log回报。"),
      equation("\\nabla J_\\gamma=\\sum_\\tau R_\\gamma(\\tau)\\nabla P_\\theta(\\tau)=\\mathbb E_{\\tau\\sim P_\\theta}[R_\\gamma(\\tau)\\nabla\\log P_\\theta(\\tau)]",
        "现在可用按当前策略采到的轨迹估计期望，不需要列举所有轨迹。"),
      equation("\\nabla\\log P_\\theta(\\tau)=\\sum_{t=0}^{T-1}\\nabla\\log\\pi_\\theta(a_t\\mid s_t)",
        "取log把乘积变求和；环境和初始分布的导数为0。因此不需要对未知环境模型求导。若环境或奖励直接依赖θ，必须补额外项。")),
    proof("条件期望恒等式", "3 · 因果性：当前动作不能改变已经收到的奖励",
      equation("u_t=\\nabla\\log\\pi_\\theta(a_t\\mid s_t),\\qquad \\mathbb E[u_t\\mid H_t]=\\sum_a\\pi_\\theta(a\\mid s_t)\\nabla\\log\\pi_\\theta(a\\mid s_t)=\\nabla\\sum_a\\pi_\\theta(a\\mid s_t)=0",
        "Hₜ是选择aₜ前的历史。概率加起来为1，其导数为0；这就是后面反复使用的零期望引理。"),
      equation("\\mathbb E\\left[u_t\\sum_{k<t}\\gamma^kr_k\\right]=0,\\qquad \\sum_{k\\ge t}\\gamma^kr_k=\\gamma^tG_t",
        "过去奖励在Hₜ给定后已经确定，乘uₜ的期望为0。因此可去掉过去，只保留动作之后的reward-to-go。"),
      equation("\\boxed{\\nabla J_\\gamma=\\mathbb E\\left[\\sum_t\\gamma^t\\nabla\\log\\pi_\\theta(a_t\\mid s_t)G_t\\right]}",
        "这得到REINFORCE。Gₜ从t重新折扣，优化初始折扣目标Jγ时仍要有外层γᵗ。统一采样时间步并省掉它，是另一种权重约定或近似。")),
    proof("条件期望与占用分布", "4 · 从回报样本到策略梯度定理",
      equation("Q^\\pi(s,a)=\\mathbb E[G_t\\mid s_t=s,a_t=a],\\qquad d_\\gamma^\\pi(s)=(1-\\gamma)\\sum_{t\\ge0}\\gamma^t\\Pr_\\pi(S_t=s)",
        "继续任务γ<1、有界奖励，或终止后接零奖励吸收状态。dγ是归一化折扣占用，包含哪些状态会被策略访问。"),
      equation("\\nabla J_\\gamma=\\frac1{1-\\gamma}\\mathbb E_{s\\sim d_\\gamma^\\pi,a\\sim\\pi}[\\nabla\\log\\pi_\\theta(a\\mid s)Q^\\pi(s,a)]",
        "用条件期望把G换为Q，再把时间求和写成占用分布。状态分布依赖策略的影响已被轨迹推导计入，并非假定dγ不随θ改变。")),
    table(["本节技巧", "解决哪一步"], [["乘积取log", "轨迹概率化为各动作的log概率之和"], ["∇P=P∇logP", "把概率导数变成可采样期望"], ["条件期望/因果性", "去掉当前动作之前的奖励"], ["折扣占用", "把时间求和写成状态动作期望"]]),
    readings("pg", "trpo"),
  ];
}

export function baselinePanel() {
  let p = 0.5, baseline = 0;
  const stats = h("div", { class: "metrics" }), rows = h("div");
  const pInput = slider({ id: "variance-probability", label: "固定用于本次方差比较的 p(→)", min: 0.05, max: 0.95, step: 0.05, value: p, format: num, onInput: v => { p = v; render(); } });
  const bInput = slider({ id: "variance-baseline", label: "两动作共用的 baseline b", min: 0, max: 9, step: 0.005, value: baseline, format: num, onInput: v => { baseline = v; render(); } });
  function render() {
    const t = baselineStatistics(p, baseline);
    replace(stats, metric("梯度样本的精确均值", num(t.mean)), metric("梯度样本的精确方差", num(t.variance)), metric("这个算例的最优 baseline", num(t.optimalBaseline)));
    replace(rows, table(["动作", "概率", "真实回报", "∂logπ/∂θ", "(G−b)·score"], [0, 1].map(a => [
      ["↓", "→"][a], num(t.probabilities[a]), [1, 7.29][a], num(t.scores[a]), num(t.samples[a]),
    ])));
  }
  render();
  return [
    proof("精确零期望", "Baseline 是控制变量：改波动，不改期望",
      equation("\\mathbb E[b(s)\\nabla\\log\\pi(a\\mid s)\\mid s]=b(s)\\nabla\\sum_a\\pi(a\\mid s)=0",
        "b不能依赖当前采样动作。于是G可以换成G−b；单次样本方向可能变，正确采样下的期望梯度不变。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, pInput, bInput,
        button("baseline设为 Vπ(J)", { onClick: () => { baseline = baselineStatistics(p, 0).value; bInput.set(baseline); render(); } }),
        button("设为最小方差 baseline", { onClick: () => { baseline = baselineStatistics(p, 0).optimalBaseline; bInput.set(baseline); render(); } })),
        h("div", { class: "experiment-content" }, stats, rows)),
    prose("这个对照使用p=σ(θ)，对logit参数θ求导，所以两个score为−p、1−p；直接对p求导会是另一组数值。固定p不进行训练，两条路线回报确定为1和7.29，只改变共用baseline。p=0.5、b=0时均值1.5725、方差4.29525625；b=4.145时两种样本都为1.5725，方差0。",
        "任意baseline都保持均值，但过大的baseline可能增加方差。Vπ是容易学习的基线，不是一般情况下的最优选择；p≠0.5时，下面的最优b与Vπ通常不同。"),
      equation("b^*(s)=\\frac{\\mathbb E[\\|u\\|^2G\\mid s]}{\\mathbb E[\\|u\\|^2\\mid s]},\\qquad u=\\nabla\\log\\pi(a\\mid s)",
        "针对单个状态的梯度项，最小化E[∥u∥²(G−b)²]并对b求导。这里确定性的二分支可做到零方差；随机回报和整条轨迹的时间相关性会改变结论。")),
    readings("pg", "variance"),
  ];
}

export function gaePanel() {
  let lambda = 0.8, ending = "terminal";
  const stats = h("div", { class: "metrics" }), rows = h("div");
  function render() {
    const terminated = ending === "terminal", t = generalizedAdvantages({ rewards: [0, 1], values: [0.4, 0.7, 2], terminated: [false, terminated], continuations: [true, false], lambda });
    replace(stats, metric("GAE · A₀", num(t.advantages[0])), metric("给critic的目标 V旧+A₀", num(t.targets[0])), metric("末步是否bootstrap", terminated ? "否，真终止" : "是，从最后状态"));
    replace(rows, table(["时刻", "r", "V旧", "后继V", "bootstrap mask", "递推 mask", "δ", "Â"], [0, 1].map(i => [
      i, [0, 1][i], [0.4, 0.7][i], [0.7, 2][i], i === 1 && terminated ? 0 : 1, i === 1 ? 0 : 1, num(t.deltas[i]), num(t.advantages[i]),
    ])));
  }
  const picker = segmented([{ value: "terminal", label: "真正终止" }, { value: "truncated", label: "外部时间截断" }], { value: ending, label: "两步样本结尾", onChange: v => { ending = v; render(); } });
  const input = slider({ id: "gae-lambda", label: "GAE λ", min: 0, max: 1, step: 0.05, value: lambda, format: num, onInput: v => { lambda = v; render(); } });
  render();
  return [
    proof("望远镜求和", "GAE：把第6章的多步思想用于优势",
      equation("\\sum_{\\ell=0}^{k-1}\\gamma^\\ell\\delta_{t+\\ell}=\\sum_{\\ell=0}^{k-1}\\gamma^\\ell r_{t+\\ell}+\\gamma^kV(s_{t+k})-V(s_t)",
        "δ=r+γV后继−V当前；中间价值逐项相消，得到k-step回报减当前基线。"),
      equation("\\hat A_t^{\\mathrm{GAE}}=\\sum_{\\ell\\ge0}(\\gamma\\lambda)^\\ell\\delta_{t+\\ell}",
        "对不同长度优势作几何混合。λ=0用一步δ；完整终止回合λ=1得到Gₜ−V(sₜ)。近似critic与较小λ通常引入偏差，换取较短目标的稳定性。")),
    proof("实际递推与两个边界", "真正终止与截断不能共用一个mask",
      equation("\\delta_t=r_t+\\gamma m_tV_{old}(s_{t+1})-V_{old}(s_t),\\qquad \\hat A_t=\\delta_t+\\gamma\\lambda c_t\\hat A_{t+1}",
        "m控制是否bootstrap：真终止为0，外部截断为1。c控制是否接下一条优势：到回合或rollout边界为0，避免串到reset后的新回合。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, picker, input),
        h("div", { class: "experiment-content" }, h("p", {}, "独立两步样本：奖励(0,1)，固定旧V=(0.4,0.7,2)，γ=0.9。最后的2是给定bootstrap估计。"), stats, rows)),
      prose("默认λ=0.8：真正终止时δ=(0.23,0.3)、Â₀=0.446；截断时δ=(0.23,2.1)、Â₀=1.742。两者都停止优势递推，只有截断保留最后状态的未来估计。",
        "截断要读最后的实际观察，不能读自动reset后的起点。任务本身的有限时域截止属于终止，剩余时间应在状态中。本章γ定义折扣目标，λ主要控制优势估计；rollout末端的bootstrap仍可能有critic误差。")),
    readings("gae", "time"),
  ];
}

export function surrogateDerivation() {
  let enter = 0.5, good = 0.5;
  const stats = h("div", { class: "metrics" });
  function render() {
    const t = surrogateExample(enter, good);
    replace(stats, metric("旧策略真实 J", num(t.oldValue)), metric("新策略真实 J", num(t.actual)), metric("固定旧访问分布的 L", num(t.surrogate)), metric("旧/新访问 s₁ 的概率", num(t.oldVisit) + " / " + num(t.newVisit)));
  }
  const q = slider({ id: "surrogate-enter", label: "新策略在s₀进入s₁的概率 q", min: 0.05, max: 0.95, step: 0.05, value: enter, format: num, onInput: v => { enter = v; render(); } });
  const p = slider({ id: "surrogate-good", label: "新策略在s₁选择好动作的概率 p", min: 0.05, max: 0.95, step: 0.05, value: good, format: num, onInput: v => { good = v; render(); } });
  render();
  return [
    proof("精确恒等式", "1 · Performance difference：改善由旧优势衡量",
      equation("J(\\pi)-J(\\pi_{old})=\\frac1{1-\\gamma}\\mathbb E_{s\\sim d_\\gamma^\\pi,a\\sim\\pi}[A^{old}(s,a)]",
        "新策略经常在哪些状态选择旧策略看来更好的动作，决定真实改善。这里状态分布仍是新策略的。"),
      prose("推导技巧还是望远镜求和：Aold的条件期望是r+γVold(s′)−Vold(s)。沿新策略累计γᵗ后，中间旧价值消去，只剩新回报减去初始旧价值；γ<1且价值有界时尾项消失。")),
    proof("局部近似", "2 · 固定旧状态占用，定义可用旧数据优化的surrogate",
      equation("L_{old}(\\pi)=J(\\pi_{old})+\\frac1{1-\\gamma}\\mathbb E_{s\\sim d_\\gamma^{old},a\\sim\\pi}[A^{old}(s,a)]",
        "替换状态分布后，L与J在旧参数处值和梯度相同，远处通常不同；不能把L当新策略真实回报。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, q, p),
        h("div", { class: "experiment-content" },
          h("div", { class: "physical-flow" }, h("span", {}, "s₀：安全→+1并终止"), h("b", {}, "或以q进入s₁ →"), h("span", {}, "s₁：好→+2 / 坏→−2并终止")),
          h("p", {}, "独立两步任务，γ=0.9；旧策略q=p=0.5。新J=(1−q)+0.9q(4p−2)，L=(1−q)+0.9×0.5×(4p−2)。"), stats)),
      prose("把q=0.1、p=0.9：新J=1.044，L=1.62，因为旧数据中一半到过s₁，新策略只会以0.1到达。L可以高于J；q=0.9、p=0.9时J=1.396、L=0.82。")),
    proof("固定状态下的精确换测度", "3 · 重要性采样：把新动作概率换成旧数据权重",
      equation("\\mathbb E_{a\\sim\\pi_\\theta(\\cdot\\mid s)}A^{old}(s,a)=\\mathbb E_{a\\sim\\pi_{old}(\\cdot\\mid s)}\\left[\\frac{\\pi_\\theta(a\\mid s)}{\\pi_{old}(a\\mid s)}A^{old}(s,a)\\right]",
        "需要旧策略覆盖新策略所用动作。动作比率修正了给定s的动作分布，没有修正新旧状态访问差异。"),
      equation("w_t(\\theta)=\\exp(\\log\\pi_\\theta(a_t\\mid s_t)-\\log\\pi_{old}(a_t\\mid s_t)),\\qquad L^{\\mathrm{CPI}}=\\hat{\\mathbb E}_t[w_t\\hat A_t]",
        "用w表示概率比，避免和奖励r混淆。第4章完整轨迹IS需要概率比乘积；PPO在固定旧批次上使用局部动作surrogate。")),
    proof("Taylor近似与约束优化", "4 · TRPO与自然梯度：限制策略分布变化",
      equation("\\max_\\Delta g^\\top\\Delta\\quad\\text{s.t.}\\quad\\tfrac12\\Delta^\\top F\\Delta\\le\\kappa,\\qquad F=\\mathbb E[uu^\\top]",
        "g是旧参数处surrogate的梯度；F按旧占用与旧动作策略求期望。目标一阶展开，平均KL二阶展开为Fisher度量；参数步长本身不能直接衡量动作分布变化。"),
      equation("\\Delta=\\sqrt{\\frac{2\\kappa}{g^\\top F^{-1}g}}F^{-1}g",
        "这里假设F正定、g≠0、κ>0。拉格朗日乘子给自然梯度方向和步长；实际用线性系统、damping、line search。"),
      equation("\\bar D=\\mathbb E_{s\\sim d_\\gamma^{old}}D_{KL}(\\pi_{old}(\\cdot\\mid s)\\Vert\\pi_\\theta(\\cdot\\mid s))\\le D_{\\max}=\\max_s D_{KL}(\\pi_{old}(\\cdot\\mid s)\\Vert\\pi_\\theta(\\cdot\\mid s))",
        "理论性能界控制最大的逐状态变化；实际平均KL约束只控制旧策略常见状态的加权平均。"),
      prose("例如两个状态的旧占用权重为0.99和0.01，KL分别为0和1 nat，则平均KL仅0.01，最大KL却是1。平均很小仍允许少见状态大幅改变；采样平均还有估计误差。不能把实际平均约束当成所有状态的变化保证。PPO随后用更简单的clip目标组织更新。")),
    readings("trpo", "ppo"),
  ];
}
