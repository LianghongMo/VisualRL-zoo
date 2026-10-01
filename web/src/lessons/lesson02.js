import graphSource from "../../../visualrl/algorithms/tabular/experience_graph.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { ExperienceGraph } from "../rl/tabular/experience-graph.js";
import { h, button, replace, segmented } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, route, routeStrip, NEAR_ROUTE, FAR_ROUTE, stateName, num, metric } from "../ui/world-view.js";
import { codeBlock } from "../ui/code.js";
import { extractDef } from "../ui/source.js";

export function mount(root) {
  const env = GridWorld.chargingRoom();
  let graph = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: 0.9 });
  let choice = "near", tr = route(env, NEAR_ROUTE), index = 0, acted = 0, state = env.start, last = null;
  env.reset();
  const map = new WorldView(env, { caption: "地图轮廓给读者定位；机器人用于学习的只有走过的连接。? 表示尚未到过，蓝线表示已观察到的连接。" });
  const strip = h("div"), stats = h("div", { class: "metrics" }), note = h("p", { class: "observation", role: "status" });
  const report = h("div", { class: "arithmetic" });
  function newWalk() { tr = route(env, choice === "near" ? NEAR_ROUTE : FAR_ROUTE); index = 0; [state] = env.reset(); last = null; render(); }
  function act() {
    if (index >= tr.length) return;
    const t = tr[index++]; const [next_state, reward, terminated] = env.step(t.action);
    graph.add({ ...t, next_state, reward, terminated }); state = next_state; acted++; last = "act"; render();
  }
  function learn() { const t = graph.sweep(); last = "learn"; report.textContent = t.changed.length ? `这轮改变：${t.changed.map((x) => `${stateName(env, x.state)} ${num(x.before)}→${num(x.after)}`).join("；")}` : "这轮没有任何价值变化。已知连接中，没有更多奖励信息可向后传播。"; render(); }
  const actBtn = button("行动：沿路线走一步", { kind: "env", onClick: act });
  const learnBtn = button("学习：沿已知边更新一轮", { kind: "learn", onClick: learn });
  function render() {
    const known = new Set([env.start, ...graph.visited, ...graph.terminal]);
    const edges = [...graph.edges].map(([key, [next_state]]) => ({ state: Number(key.split(",")[0]), next_state }));
    map.render({ robot: state, values: graph.V, known, edges });
    replace(strip, routeStrip(env, tr, { values: graph.V, visited: index, active: index, reverse: true, label: "机器人沿路线向右走；已观察到的奖励信息，经学习向左传。" }));
    replace(stats, metric("行动", `${acted} 步`), metric("学习", `${graph.sweeps} 轮`), metric("已知连接", graph.edges.size), metric("V(出发点)", num(graph.V[env.start])));
    actBtn.disabled = index >= tr.length; learnBtn.disabled = graph.edges.size === 0;
    note.textContent = last === "act" ? "刚才只行动：多了一条经验，价值没有更新。即使已经到达 +1，出发点的 V 也不会自动改变。" : last === "learn" ? "刚才只学习：机器人没动，经验数量没变；已有奖励信息沿已知连接向后传播。" : "先只点行动 5 次，再只点学习 5 次。观察：一个按钮改变经验，另一个按钮改变数字。";
    if (last !== "learn") report.textContent = "这里的学习是经验图上的规划：只计算已观察到的动作，未知动作不参与比较。这是展示两个循环的简化方式。";
  }
  const picker = segmented([{ value: "near", label: "采集近处路线" }, { value: "far", label: "采集远处路线" }], { value: choice, label: "下一轮由你指定的采集路线", onChange: (v) => { choice = v; newWalk(); } });
  root.append(lessonHeader("02"), step("先看图像：留下脚印，再沿脚印计算", prose("上一章可以查询完整模型。现在收起它：学习器只记得机器人实际做过的动作及其结果。你指定采集路线；学习器在收集到的连接中寻找最好的路线。", "行动增加经验，学习更新价值。站着不动也能反复计算；但如果从没观察到通向远处充电站的连接，计算再多次也补不出那条路。")),
    step("动手验证：把两个按钮分开按", h("div", { class: "experiment" },
      h("div", { class: "experiment-instruction" }, h("strong", {}, "5 次行动 → 5 次学习"), "先到达近处 +1，注意所有价值仍为 0；再让奖励传回出发点。最后换成远处路线，检查新经验带来什么。"),
      h("div", { class: "toolbar" }, actBtn, learnBtn, button("同一路线再走一轮", { kind: "ghost", onClick: newWalk }), button("清空经验和价值", { kind: "ghost", onClick: () => { graph = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: 0.9 }); acted = 0; newWalk(); } })),
      h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, picker, note)), h("div", { class: "experiment-content" }, strip, report))),
    takeaway("行动改变我们知道什么；学习改变我们怎样评价已知的东西。已有数据可以多次使用，缺失的数据需要新的行动才能获得。"),
    step("检查理解", predict({ question: "只走过通向 +1 的路线，连续学习 100 次，能知道远处 +10 的路线吗？", choices: [{ label: "不能，缺少那条路线的经验" }, { label: "能，只要更新次数足够多" }], answer: 0, explain: "学习只用已观察到的连接。必须获得通向 +10 的新经验，才能把它的奖励传回来。" })),
    optional("展开：经验图上的真实更新", codeBlock(extractDef(graphSource, "sweep"), { title: "ExperienceGraph.sweep" })), lessonFooter("02"));
  render(); return () => {};
}
