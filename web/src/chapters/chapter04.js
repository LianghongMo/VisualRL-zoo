import { h, button, segmented, replace } from "../ui/dom.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { Trajectory } from "../rl/core.js";
import { MonteCarlo } from "../rl/tabular/monte-carlo.js";
import { WorldView, route, NEAR_ROUTE, statesOf, metric, num } from "../ui/world-view.js";
import { chapterHeader, chapterFooter, topic, prose, equation, table, predict } from "./shell.js";
export function mount(root) {
  const env = GridWorld.chargingRoom(); let repeated = false, firstVisit = true, full, observed, mc, learned = false, index = 0, traces = [];
  const map = new WorldView(env, { caption: "沿示范记录收集真实转移。未完成回合时不提前把后续奖励交给学习器；含重复状态的样本会先返回S，再去充电站。" });
  const stats = h("div", { class: "metrics" }), targets = h("div"), note = h("p", { class: "observation", role: "status" });
  function reset() {
    full = route(env, repeated ? [0, 2, ...NEAR_ROUTE] : NEAR_ROUTE);
    observed = new Trajectory(); mc = new MonteCarlo({ nStates: env.nStates, gamma: 0.9, alpha: 1, firstVisit });
    index = 0; learned = false; traces = []; env.reset(); render();
  }
  const collect = button("采集下一步", { kind: "env", onClick: () => {
    if (observed.done) return;
    const t = full[index++], [next_state, reward, terminated] = env.step(t.action);
    observed.append({ ...t, next_state, reward, terminated }); render();
  } });
  const learn = button("MC：学习完整回合", { kind: "learn", onClick: () => {
    if (!observed.terminated || learned) return;
    traces = mc.learnEpisode(observed); learned = true; render();
  } });
  function render() {
    collect.disabled = observed.done; learn.disabled = !observed.terminated || learned;
    const state = index ? observed.at(index - 1).next_state : env.start;
    map.render({ robot: observed.terminated ? null : state, values: learned ? mc.V : null, paths: [{ states: statesOf(observed.transitions) }] });
    replace(stats, metric("已观察转移", index + "/" + full.length), metric("MC实际更新次数", mc.learnSteps),
      metric("MC · V(S)", num(mc.V[env.start])), metric("当前访问规则", firstVisit ? "首访" : "每访"));
    replace(targets, learned ? table(["回合时刻", "状态", "实际剩余 Gₜ", "旧V → 新V"], traces.map(t => [
      t.timestep, t.state === env.start ? "S" : env.toCell(t.state).join(","), num(t.target), num(t.value_before) + " → " + num(t.value_after),
    ])) : h("p", {}, "回合真正结束后，才为选中的访问点构造目标。"));
    note.textContent = !observed.terminated ? "正在收集经验。第一个奖励还可能是0，余下未来未知，所以MC按钮等待终止。"
      : learned ? "使用α=1逐次替换。本回合" + (firstVisit ? "每个状态只在第一次访问时更新。" : "每次访问都更新；同一状态的后一次回报可能覆盖前一次。") +
        (repeated ? " S第一次访问的G为0.531；第二次访问的G为0.656。" : " S的G为0.656。")
        : "已收到终点奖励，完整回报已知。现在可以学习；环境模型未参与学习器的目标计算。";
  }
  const visits = segmented([{ value: true, label: "首访 MC" }, { value: false, label: "每访 MC" }], { value: firstVisit, label: "访问规则", onChange: v => { firstVisit = v; reset(); } });
  root.append(chapterHeader("04"),
    topic("04", "prediction",
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, collect, learn, visits,
        button("切换5步/含重复状态的7步样本", { onClick: () => { repeated = !repeated; reset(); } }), button("重新采集并清零", { kind: "ghost", onClick: reset })),
        h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, targets, note))),
      prose("没有p也能评价策略：按策略得到完整回合，从每个访问点往后累计真实奖励，形成一个Gₜ样本。随机时，多次样本的平均概括Vπ。学习器只用观察记录，地图只用来展示。",
        "首访MC只使用一个回合中状态的第一次访问；每访MC使用全部访问。5步与7步示范都是在分岔处允许随机返回S或继续向右时可能产生的记录；这里手动选择样本，便于对照更新规则。"),
      equation("G_t=\\sum_{k=0}^{T-t-1}\\gamma^k r_{t+k},\\qquad V(s_t)\\leftarrow V(s_t)+\\alpha\\big[G_t-V(s_t)\\big]",
        "MC不以V(s′)补未来。本实验α=1；若用样本均值，则每个状态按其累计有效样本数取α=1/N(s)。"),
      prose("重复状态样本中，S在t=0和t=2出现；它们的目标分别是γ⁶×1=0.531441、γ⁴×1=0.6561。首访只用第一个。每访配α=1会先改成0.531，再改成0.656；若平均两个样本则是0.5937705。规则与学习率一起决定更新结果。",
        "MC需要真实终止或确实知道余下回报。人为时间上限不是终点；直接把截断后的未来当0，会改变所评价的问题。")),
    topic("04", "control",
      prose("固定策略预测得到Vπ，还没有回答「该选哪个动作」。未知模型下无法查询每个动作的后果，因此要直接积累状态动作的回报样本，估计Qπ(s,a)。",
        "MC控制沿用第3章GPI：用完整回报改善Q，再让策略更偏向当前Q最大的动作，同时保留探索。只评价一条从不探索的路线，无法比较它没用过的动作。"),
      equation("Q(s_t,a_t)\\leftarrow Q(s_t,a_t)+\\alpha\\big[G_t-Q(s_t,a_t)\\big],\\qquad \\pi\\leftarrow\\epsilon\\text{-greedy}(Q)",
        "回合采集 → 回报评价 → 策略改进 → 再采集。Exploring starts通过覆盖初始状态动作来探索；更常用ε-soft策略在行动中保持覆盖。"),
      prose("这个流程不是DP：DP需要模型给出未实际发生的所有后果，MC只平均策略产生的真实完整回报。对平稳有限任务，持续覆盖与合适学习率支持估计稳定；某次回报大不能直接证明策略最优。")),
    topic("04", "importance",
      prose("行为策略b负责产生记录，目标策略π是我们想评价的规则。它们不同时，同一条轨迹在两套策略下的发生概率不同，不能直接把b的回报平均当成Vπ。"),
      equation("\\rho_{t:T-1}=\\prod_{k=t}^{T-1}\\frac{\\pi(a_k\\mid s_k)}{b(a_k\\mid s_k)},\\qquad V^\\pi(s)=\\mathbb E_b[\\rho_{t:T-1}G_t\\mid S_t=s]",
        "重要性采样纠正动作选择分布。环境动力学在两套策略下相同，所以概率比中消去；需要目标动作有行为策略的数据支持。"),
      equation("\\hat V_{\\mathrm{ordinary}}=\\frac1N\\sum_i\\rho_iG_i,\\qquad \\hat V_{\\mathrm{weighted}}=\\frac{\\sum_i\\rho_iG_i}{\\sum_i\\rho_i}",
        "普通估计按样本数归一；加权估计按权重和归一。加权估计有限样本通常有偏，但常能降低方差；分母为0时没有可用估计。"),
      table(["两步样本：目标总选→，b每步以0.5选→", "数值"], [["轨迹动作概率比", "(1/0.5)² = 4"], ["样本G=2，普通IS的单项贡献", "ρG=8"], ["仅这一条样本的加权IS", "ρG/ρ=2"]]),
      prose("概率比会随轨迹长度相乘，长回合的方差可能很大。它可以重新加权已有动作，不能为b从未选择的动作创造经验。这个限制将接到最后一章的离线数据覆盖。"),
      predict({ question: "b(a|s)=0，但π(a|s)>0，能用重要性采样评价该动作吗？", choices: [{ label: "不能，缺少目标动作的数据支持" }, { label: "能，把概率比设得很大就行" }], answer: 0, explain: "除以0不是可用权重；重新加权不能恢复不存在的样本。" })),
    chapterFooter("04", "MC从完整实际回报评价策略；首访/每访决定取哪些样本，学习率决定怎样累积。估计Q再改进策略得到MC控制；行为与目标不同则需要概率比和覆盖。下一章把完整未来换成一步自举。"));
  reset(); return () => {};
}
