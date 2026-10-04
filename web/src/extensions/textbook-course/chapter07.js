import * as connections from "../../lessons/lesson02.js";
import { h, button, slider, replace } from "../../ui/dom.js";
import { GridWorld } from "../../rl/envs/gridworld.js";
import { DynaLab } from "../../rl/teaching-labs.js";
import { WorldView, route, FAR_ROUTE, metric, num, statesOf } from "../../ui/world-view.js";
import { chapterHeader, chapterFooter, topic, capture, prose, equation, table } from "./shell.js";
export function mount(root) {
  const old = capture(connections), env = GridWorld.chargingRoom(), full = route(env, FAR_ROUTE);
  let lab = new DynaLab(env.nStates), index = 0, planning = 0;
  const map = new WorldView(env, { caption: "沿8步示范采集到+10。数字是maxₐQ(s,a)，来自真实更新和模型重放；规划不会让机器人移动，也不会补出未观察的边。" });
  const stats = h("div", { class: "metrics" }), note = h("p", { class: "arithmetic", role: "status" });
  const act = button("Dyna：真实行动一步", { kind: "env", onClick: () => {
    if (index >= full.length) return;
    const t = full[index++], [next_state, reward, terminated] = env.step(t.action);
    lab.observe({ ...t, next_state, reward, terminated }); lab.plan(planning); render();
  } });
  const plan = button("只从模型规划200次", { kind: "learn", onClick: () => { lab.plan(200); render(); } });
  function render() {
    act.disabled = index >= full.length; plan.disabled = lab.model.size === 0;
    const V = lab.agent.Q.map(row => Math.max(...row));
    map.render({ values: V, robot: index === full.length ? null : index ? full[index - 1].next_state : env.start,
      paths: [{ states: statesOf(full.slice(0, index)) }] });
    replace(stats, metric("Dyna真实环境转移", lab.realSteps), metric("模型中的连接", lab.model.size), metric("Dyna模拟更新", lab.planningSteps), metric("Dyna · max Q(S)", num(V[env.start])));
    const t = lab.last;
    note.textContent = t ? "最近一次Q更新：目标=" + num(t.reward) + " + 0.9 × " + num(t.bootstrap_value) + " = " + num(t.target) + "；旧Q " + num(t.value_before) + " → " + num(t.value_after)
      : "还没有模型。先真实行动：一条经验同时更新Q和对应的(s,a)后果。";
  }
  env.reset();
  const count = slider({ id: "dyna-planning-count", label: "每次真实行动后附加的规划次数", min: 0, max: 50, step: 5, value: planning, onInput: v => { planning = v; } });
  root.append(chapterHeader("07"),
    topic("07", "model",
      prose("第3章假设模型完整；第4–6章用采样经验直接更新价值。现在把这两条路径接起来：从真实经验学习模型，再让模型生成模拟经验。模型回答「会发生什么」，Q回答「长期值得做吗」。"),
      ...old.all(),
      equation("\\hat p(s',r\\mid s,a)\\approx p(s',r\\mid s,a)",
        "一般模型预测后果分布。本地图确定且不变，因此一条观测可以记录该(s,a)的确定后果；随机环境需要累计样本，不能这样直接等同。")),
    topic("07", "dyna", h("div", { class: "experiment" }, h("div", { class: "toolbar" }, act, plan, count,
      button("Dyna全部重置", { kind: "ghost", onClick: () => { lab = new DynaLab(env.nStates); index = 0; env.reset(); render(); } })),
      h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, note))),
      prose("先把每步规划设为0，走完8步：前面的Q仍可能是0，只有末端先知道+10。保持机器人不动，再规划200次：同一组已记录连接被随机抽取重放，信息传回S，max Q(S)变成4.783。",
        "Dyna-Q真实一步做两件事：直接用经验更新Q；记录模型。再重复若干次：抽一个已观察(s,a)，用模型返回r、s′、done，同样做Q-learning更新。增加的是模拟更新数，真实环境转移数不变。"),
      equation("Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma\\max_{a'}Q(s',a')-Q(s,a)\\big],\\quad (r,s',d)\\sim\\hat p(\\cdot\\mid s,a)",
        "真实转移和模型转移采用同一类更新，终止遮掉未来。本实验沿指定示范采集以分开观察两种来源；完整Dyna可由ε-greedy行为选择真实动作。")),
    topic("07", "search",
      table(["组织方式", "重点", "仍然需要检查"], [
        ["随机Dyna重放", "多次利用模型中的连接", "模型是否有偏差或已经过时"],
        ["优先级扫描", "先更新误差大、影响上游的位置", "前驱关系与优先级是否正确"],
        ["决策时搜索", "围绕当前状态与候选动作展开未来", "搜索深度、计算预算、模型误差"],
        ["Monte Carlo树搜索", "用采样模拟评价搜索树中的动作", "模拟策略、探索规则与真实任务的一致性"],
      ]),
      prose("墙或动力学改变以后，旧模型仍可能生成不存在的连接；再多规划会重复错误。真实新经验需要纠正模型。模型误差与价值误差不同：一个把后果记错，一个把长期结果估错。",
        "优先级扫描把大备份误差向相关前驱传播；搜索把计算集中在当前决策的可选未来。它们改善计算分配，不替代数据覆盖与正确模型。最后一章的SoRB则把局部可达能力组织为路标图，是这个规划思想的一个研究连接。")),
    chapterFooter("07", "行动提供真实经验，经验更新价值和模型，模型提供可重复使用的模拟经验。Dyna把直接学习与规划交错；更多计算的收益取决于模型正确性与数据覆盖。"));
  render(); return old.cleanup;
}
