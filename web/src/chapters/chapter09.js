import { h, s, button, segmented, slider, replace } from "../ui/dom.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { PolicyGradientLab, ppoObjective } from "../rl/teaching-labs.js";
import { WorldView, metric, num } from "../ui/world-view.js";
import { chapterHeader, chapterFooter, topic, prose, equation, table } from "./shell.js";
import { trajectoryDerivation, baselinePanel, gaePanel, surrogateDerivation } from "./policy-derivations.js";
import { entropyTools, practiceNotes } from "./entropy-tools.js";
import { COURSE, readings } from "./sources.js";
export function mount(root) {
  const env = GridWorld.chargingRoom(), J = env.toState(3, 4);
  const near = [J, env.toState(4, 4)], far = [J, env.toState(3, 5), env.toState(3, 6), env.toState(2, 6), env.toState(1, 6)];
  let lab = new PolicyGradientLab(), baseline = 4.145, ratio = 1, advantage = 2;
  const map = new WorldView(env, { caption: "从J重新开始，只训练↓与→两种候选；后续走法固定。线越粗代表选择概率越高。↓的回报1，→的回报7.29。" });
  const stats = h("div", { class: "metrics" }), calculation = h("p", { class: "arithmetic", role: "status" });
  const ppoReadout = h("div", { class: "metrics" }), plot = s("svg", { viewBox: "0 0 440 250", class: "chapter-plot", role: "img", "aria-label": "PPO概率比与clipped目标曲线" });
  function renderPolicy() {
    map.render({ robot: J, selected: J, paths: [{ states: near, color: "var(--reward)", width: 2 + 8 * (1 - lab.p) }, { states: far, width: 2 + 8 * lab.p }] });
    replace(stats, metric("πθ(→|J)", num(lab.p)), metric("πθ(↓|J)", num(1 - lab.p)), metric("从 J 出发的预期回报", num(lab.value)), metric("策略参数 θ", num(lab.theta)));
    const t = lab.last;
    calculation.textContent = t ? "观察" + (t.action === 1 ? "→" : "↓") + "：G=" + num(t.G) + "，baseline=" + num(t.baseline) + "，样本优势=" + num(t.advantage) +
      "；∂logπ/∂θ=" + num(t.score) + "；θ " + num(t.thetaBefore) + " → " + num(t.thetaAfter) + "，向右概率 " + num(t.pBefore) + " → " + num(t.pAfter)
      : "θ=0时两个分支各一半，预期回报0.5×1+0.5×7.29=4.145。用一条样本的优势决定概率改变方向。";
  }
  function renderPPO() {
    const objective = ppoObjective(ratio, advantage), x = r => 45 + (r - 0.2) / 1.8 * 355, y = v => 120 - 25 * v;
    replace(ppoReadout, metric("概率比 w", num(ratio)), metric("给定优势 A", advantage), metric("未clip目标 wA", num(ratio * advantage)), metric("PPO clipped目标", num(objective)));
    const samples = Array.from({ length: 91 }, (_, i) => 0.2 + i * 0.02);
    replace(plot, s("line", { x1: 45, x2: 410, y1: 120, y2: 120, stroke: "var(--rule)" }),
      [0.8, 1.2].map(r => s("line", { x1: x(r), x2: x(r), y1: 15, y2: 225, stroke: "var(--rule)", "stroke-dasharray": "4 4" })),
      s("polyline", { points: samples.map(r => x(r) + "," + y(r * advantage)).join(" "), stroke: "var(--act)", "stroke-width": 2, "stroke-dasharray": "5 4", fill: "none" }),
      s("polyline", { points: samples.map(r => x(r) + "," + y(ppoObjective(r, advantage))).join(" "), stroke: "var(--learn)", "stroke-width": 3, fill: "none" }),
      s("circle", { cx: x(ratio), cy: y(objective), r: 6, fill: "var(--learn)" }),
      [0.2, 0.8, 1, 1.2, 2].map(r => s("text", { x: x(r), y: 245, "text-anchor": "middle" }, r)),
      s("text", { x: 12, y: 20 }, "目标"), s("text", { x: 345, y: 20 }, "实线：clip"), s("text", { x: 345, y: 38 }, "虚线：wA"));
  }
  const baselineInput = slider({ id: "pg-baseline", label: "给定 critic 基线 V(J)", min: 0, max: 8, step: 0.005, value: baseline, format: v => num(v), onInput: v => { baseline = v; } });
  const ratioInput = slider({ id: "ppo-ratio", label: "新/旧动作概率比 w", min: 0.2, max: 2, step: 0.05, value: ratio, format: v => v.toFixed(2), onInput: v => { ratio = v; renderPPO(); } });
  const sign = segmented([{ value: 2, label: "正优势 +2" }, { value: -2, label: "负优势 −2" }], { value: advantage, label: "优势符号", onChange: v => { advantage = v; renderPPO(); } });
  root.append(chapterHeader("09"),
    h("p", { class: "source-note" }, "本章按 ", h("a", { href: COURSE, target: "_blank", rel: "noopener" }, "COS435 第4–6周公开阅读"), " 扩展推导：先理解采样梯度，再处理方差与旧数据，最后解释熵和Jensen。"),
    topic("09", "gradient", h("div", { class: "experiment" }, h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, calculation)),
      h("div", { class: "toolbar" },
        button("观察→，做一次策略梯度更新", { kind: "learn", onClick: () => { lab.sampleUpdate(1, baseline); renderPolicy(); } }),
        button("观察↓，做一次策略梯度更新", { onClick: () => { lab.sampleUpdate(0, baseline); renderPolicy(); } }),
        button("按精确期望梯度更新", { onClick: () => { lab.exactUpdate(); renderPolicy(); calculation.textContent = "用已知两分支回报验算精确梯度：∂J/∂θ=p(1−p)(7.29−1)，向右概率提高。"; } }),
        button("策略恢复各一半", { kind: "ghost", onClick: () => { lab = new PolicyGradientLab(); renderPolicy(); } }))),
      prose("价值方法学Q再选动作；策略梯度直接给动作概率一组参数θ。本例p=σ(θ)表示向右概率，改变θ就改变两条路线被选择的频率。目标仍然是预期回报，而不是任意增大概率。",
        "两个按钮指定一条已观察样本，展示条件于这个动作的一次更新；完整REINFORCE按当前π采样完整回合。精确梯度按钮只用于这个已知两分支例子的验算。"),
      equation("J(\\theta)=\\mathbb E_{\\pi_\\theta}\\left[\\sum_t\\gamma^t r_t\\right],\\qquad \\nabla J=\\mathbb E_{\\pi_\\theta}\\left[\\sum_t\\gamma^t G_t\\nabla\\log\\pi_\\theta(a_t\\mid s_t)\\right]",
        "REINFORCE用回报加权对数概率梯度：让产生较好未来的动作更常出现。这里的Gₜ从t重新折扣，所以对初始目标J还要有γᵗ。"),
      equation("\\theta\\leftarrow\\theta+\\alpha\\gamma^t[G_t-b(s_t)]\\nabla_\\theta\\log\\pi_\\theta(a_t\\mid s_t)",
        "本例从J、t=0开始；α=0.2。正样本优势增加所观察动作的概率，负优势降低它。回报是好是坏，取决于同一状态的基准。")),
    topic("09", "derivation", ...trajectoryDerivation()),
    topic("09", "actor-critic", h("div", { class: "toolbar" }, baselineInput),
      prose("baseline可降低梯度估计的方差，只要它不依赖本次采样动作，就不改变正确采样下的期望策略梯度。Vπ(s)是常见的基线；Qπ(s,a)−Vπ(s)定义动作的真实优势Aπ。",
        "默认基线4.145是初始策略在J的精确价值，后续作为给定critic估计。把基线调到8，再观察同一个向右回报7.29：优势变负，向右概率会下降。单次更新方向依赖估计误差，不能把每次更新都称为策略一定改善。"),
      equation("A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s),\\qquad \\hat A_t\\approx\\delta_t=r_t+\\gamma V_\\phi(s_{t+1})-V_\\phi(s_t)",
        "后继由当前策略采样、critic准确时，一步TD误差的条件期望对应优势；实际δ仍有采样和critic估计误差。真正终止遮掉后继价值。"),
      equation("\\phi\\leftarrow\\phi+\\alpha_v\\delta_t\\nabla_\\phi V_\\phi(s_t),\\qquad \\theta\\leftarrow\\theta+\\alpha_\\pi\\gamma^t\\hat A_t\\nabla_\\theta\\log\\pi_\\theta(a_t\\mid s_t)",
        "critic更新价值参数φ；actor更新概率参数θ。这里写的是折扣初始回报目标的一种形式，不能把基线、critic与策略参数混成同一个量。"),
      ...baselinePanel(), ...gaePanel()),
    topic("09", "surrogate", ...surrogateDerivation()),
    topic("09", "ppo", h("div", { class: "experiment" }, h("div", { class: "toolbar" }, ratioInput, sign),
      h("div", { class: "experiment-content" }, plot, ppoReadout)),
      prose("上一节得到旧数据上的动作比率surrogate，并说明为什么要限制策略变化。PPO接着设计一个容易用梯度优化的目标：在鼓励改善的方向上设置平台；这是算法设计，并非从PG定理唯一推出的公式。",
        "PPO在正优势且w过大时不再奖励继续提高概率，在负优势且w过小时不再奖励继续降低概率。它选择两个目标中较小的。"),
      equation("L^{\\mathrm{clip}}(\\theta)=\\mathbb E\\left[\\min\\left(w_t(\\theta)\\hat A_t,\\operatorname{clip}(w_t(\\theta),1-\\epsilon,1+\\epsilon)\\hat A_t\\right)\\right]",
        "本算例ε=0.2。A=+2、w=1.5时目标2.4；A=−2、w=0.5时目标−1.6；负优势且w=1.5时目标仍为−3。"),
      equation("\\ell(w,A)=\\begin{cases}A\\min(w,1+\\epsilon),&A\\ge0\\\\ A\\max(w,1-\\epsilon),&A<0\\end{cases}",
        "把min按优势符号展开最容易验算。A<0时乘负数会翻转比较方向；坏方向仍保留斜率。"),
      prose("逐样本clip目标不大于unclipped surrogate；它不是新策略真实J的全局下界，也不硬性强制w留在区间内。共享参数、critic或其他样本的梯度仍能移动已经进入平台的动作概率；因此要监测KL与实际执行表现。"),
      table(["层次", "已经讲清的对象", "完整训练还需要"], [
        ["策略梯度", "概率参数、目标与单样本方向", "按策略采集、多回合或batch梯度"],
        ["Actor–Critic", "actor、critic、δ与优势的分工", "价值拟合、优势估计、学习率与采样"],
        ["PPO", "旧策略比率与clip目标的形状", "固定旧log概率、batch/minibatch、多轮更新、价值损失与评估"],
      ]),
      prose("上面的策略概率和clip曲线运行精确小算例，没有运行完整PPO训练。仓库原有PPO与MuJoCo材料保留，可在这组前置知识上继续接入。"),
      readings("ppo")),
    topic("09", "entropy", ...entropyTools()),
    topic("09", "practice", ...practiceNotes()),
    chapterFooter("09", "轨迹概率与log-derivative给采样梯度；因果性去掉过去，baseline改变波动，GAE混合多步优势。固定旧访问分布得到局部surrogate，TRPO/PPO控制更新激励。Jensen与KL解释额外的熵目标；训练时要固定旧数据和目标，检查真正执行结果。最后把这些对象条件化到目标g。"));
  renderPolicy(); renderPPO(); return () => {};
}
