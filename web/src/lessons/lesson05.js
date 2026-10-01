import viSource from "../../../visualrl/algorithms/tabular/value_iteration.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { ValueIteration, optimalBackup } from "../rl/tabular/dp.js";
import { h, button, replace } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, ARROWS, stateName, num, metric } from "../ui/world-view.js";
import { equation } from "../ui/math.js";
import { codeBlock } from "../ui/code.js";
import { extractDef } from "../ui/source.js";

export function mount(root) {
  const env = GridWorld.chargingRoom(), model = env.model();
  let vi = new ValueIteration(model, 0.9), selected = env.toState(3, 4), trace = null;
  const map = new WorldView(env, { caption: "数字是 V；箭头指向当前估计最好的动作；蓝色边框表示上一轮更新中价值发生变化。点一个空白格查看它的计算。", onSelect: (s) => { selected = s; render(); } });
  const stats = h("div", { class: "metrics" }), table = h("div", { class: "table-wrap" });
  const note = h("p", { class: "observation", role: "status" }), calc = h("div", { class: "arithmetic" });
  function sweep() { trace = vi.sweep(); render(); }
  function render() {
    const Q = trace?.q_values ?? vi.Q;
    const policy = Q.map((row, s) => model.terminal[s] || Math.max(...row) <= 0 ? null : row.indexOf(Math.max(...row)));
    const states = [env.start], seen = new Set(states); let st = env.start;
    for (let i = 0; i < 40 && policy[st] !== null; i++) { const [next, , done] = env.move(st, policy[st]); states.push(next); if (done || seen.has(next)) break; seen.add(next); st = next; }
    const changed = new Set(trace ? trace.values_after.flatMap((v, s) => Math.abs(v - trace.values_before[s]) > 1e-10 ? [s] : []) : []);
    map.render({ values: vi.V, policy, selected, changed, paths: states.length > 1 ? [{ states, color: "var(--act)", width: 3 }] : [] });
    const backup = optimalBackup(model, vi.V, selected, 0.9);
    replace(table, h("h3", {}, `${stateName(env, selected)} 的下一次更新`), h("table", {}, h("thead", {}, h("tr", {}, ["动作", "眼前奖励 + 下一格价值", "结果"].map((x) => h("th", {}, x)))), h("tbody", {}, backup.actions.map((a) => {
      const b = a.branches[0];
      return h("tr", { class: backup.best_actions.includes(a.action) ? "highlight" : "" }, h("td", {}, ARROWS[a.action]), h("td", {}, `${num(b.reward)} + 0.9 × ${num(b.next_value)}`), h("td", {}, num(a.q)));
    }))));
    calc.textContent = `V(${stateName(env, selected)})：${num(vi.V[selected])} → max(四个结果) = ${num(backup.value_after)}`;
    replace(stats, metric("已经更新所有位置", `${vi.sweeps} 轮`), metric("出发点价值 V(S)", num(vi.V[env.start])), metric("上一轮变化的位置", changed.size));
    note.textContent = vi.sweeps === 0 ? "价值全部从 0 开始。先更新一轮：只有一步能拿到奖励的位置得到正价值。" : vi.sweeps < 5 ? "蓝框向外扩展：每轮只读取上一轮的价值，所以新奖励的信息每次向后传播一条边。" : vi.sweeps < 8 ? "近处 +1 的信息已经到达出发点，V(S) = 0.656；远处 +10 的信息还在传播。继续更新，观察出发点何时改选路线。" : "远处 +10 的信息已传回出发点：V(S) = 4.783。此时最好的动作组成通向远处充电站的路线；继续更新，数值会稳定下来。";
  }
  root.append(lessonHeader("05"), step("先看图像：未来的信息，往回传", prose("站在一格上，想象往四个方向各看一步。每个动作值多少？把眼前收到的奖励，加上下一格价值的 0.9 倍。保留四个结果中最大的，作为这一格的新价值。", "这一章假设每个动作的后果都已知，所以是在完整模型上规划。下面所有位置同时更新，读取的都是上一轮的数字。机器人可以不动，价值照样传播。")),
    step("动手验证：每次只传播一条边", h("div", { class: "experiment" },
      h("div", { class: "experiment-instruction" }, h("strong", {}, "依次点 1 次、5 次、8 次更新"), "观察蓝框从哪里开始扩散；第 5 轮与第 8 轮，出发点得到的价值分别来自哪个充电站？"),
      h("div", { class: "toolbar" }, button("更新所有位置一轮", { kind: "learn", onClick: sweep }), button("继续更新到稳定", { onClick: () => { for (let i = 0; i < 100; i++) { trace = vi.sweep(); if (trace.max_change < 1e-10) break; } render(); } }), button("把价值清零", { kind: "ghost", onClick: () => { vi = new ValueIteration(model, 0.9); trace = null; render(); } })),
      h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, table, calc, note)))),
    takeaway("Bellman 更新用「眼前奖励 + 下一格的折扣价值」评价动作，再选最好的。反复更新所有位置，奖励信息就能从终点传回出发点；这个过程叫价值迭代。"),
    step("检查理解", predict({ question: "第 1 轮更新后，8 步外的 +10 为什么还没有改变出发点的价值？", choices: [{ label: "这一轮只读取旧价值，信息还没逐格传回来" }, { label: "机器人必须先实际走到 +10" }], answer: 0, explain: "这里环境模型完整，不需要真的走路。限制来自更新顺序：每一轮只让新信息向后传播一条边。" })),
    optional("展开：公式与真实实现", equation("V_{k+1}(s) = \\max_a [r(s,a) + \\gamma V_k(s')]", "当前环境没有随机滑动；进入终点后，后续价值为 0。"), codeBlock(extractDef(viSource, "sweep", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "ValueIteration.sweep" })),
    optional("选读：也可以先评价策略，再改进策略", prose("同样在已知模型下，还可以把「评价」和「改进」分成两个独立操作。"), h("a", { href: "#lesson-06" }, "打开策略迭代实验 →")), lessonFooter("05"));
  render(); return () => {};
}
