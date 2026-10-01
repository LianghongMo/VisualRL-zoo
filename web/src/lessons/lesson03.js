import { GridWorld } from "../rl/envs/gridworld.js";
import { ExperienceGraph } from "../rl/tabular/experience-graph.js";
import { h, button, replace } from "../ui/dom.js";
import { lessonHeader, step, prose, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, route, statesOf, NEAR_ROUTE, FAR_ROUTE, discounted, num, metric } from "../ui/world-view.js";

export function bestKnownRoute(env, graph) {
  let state = env.start; const tr = [], seen = new Set([state]);
  for (let i = 0; i < 50; i++) {
    const action = graph.act(state), edge = graph.edge(state, action); if (!edge) break;
    const [next_state, reward, terminated] = edge; tr.push({ state, action, next_state, reward, terminated });
    if (terminated || seen.has(next_state)) break; seen.add(next_state); state = next_state;
  }
  return tr;
}
export function mount(root) {
  const env = GridWorld.chargingRoom(), A = route(env, NEAR_ROUTE);
  const B = route(env, [0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0]);
  const graph = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: 0.9 });
  let data = "both", planned = false, collected = false;
  const map = new WorldView(env, { caption: "实线 A 到近处 +1；虚线 B 绕远到 +10。两条记录在 (3,4) 相交。规划得到的粗蓝线只能使用记录里确实存在的动作。" });
  const stats = h("div", { class: "metrics" }), note = h("p", { class: "observation", role: "status" });
  const planBtn = button("离线：只用这些记录规划", { kind: "learn", onClick: () => { graph.plan(); planned = true; render(); } });
  const collectBtn = button("在线：实际采集一次到 +10 的路线", { kind: "env", onClick: () => {
    let [state] = env.reset();
    for (const action of FAR_ROUTE) {
      const [next_state, reward, terminated] = env.step(action);
      graph.add({ state, action, next_state, reward, terminated }); state = next_state; if (terminated) break;
    }
    graph.plan(); collected = true; planned = true; render();
  } });
  function load(mode) {
    data = mode; planned = false; collected = false; graph.edges.clear(); graph.visited.clear(); graph.terminal.clear(); graph.V.fill(0); graph.sweeps = 0;
    for (const t of mode === "both" ? [...A, ...B] : A) graph.add(t); render();
  }
  function render() {
    const best = planned ? bestKnownRoute(env, graph) : [];
    const paths = [{ states: statesOf(A), color: "var(--act)", width: 3 }];
    if (data === "both") paths.push({ states: statesOf(B), color: "var(--reward)", dashed: true, width: 3 });
    if (best.length) paths.push({ states: statesOf(best), color: "var(--learn)", width: 8 });
    map.render({ paths, robot: env.start, selected: env.toState(3, 4) });
    replace(stats, metric("记录 A", `${A.length} 步 / ${num(discounted(A, 0.9))}`), metric("记录 B", data === "both" ? `${B.length} 步 / ${num(discounted(B, 0.9))}` : "没有提供"), metric("规划出的路线", planned ? `${best.length} 步 / ${num(discounted(best, 0.9))}` : "等待规划"), metric("已知连接", graph.edges.size));
    collectBtn.disabled = data !== "onlyA" || collected;
    note.textContent = collected ? "刚才在线采集了 8 步新经验。现在存在通向 +10 的已知连接，价值可传回出发点。结果变好来自新增经验，不是离线计算次数更多。" : data === "onlyA" ? "现在只给记录 A。反复离线规划仍然只能走向 +1；通向 +10 的连接没有记录。点在线采集，真正获得缺失的经验。" : planned ? "拼出了 8 步路线：用 A 的前 4 步到交点，再用 B 的后 4 步到 +10。回报 4.783，大于 A 的 0.656 和 B 的 3.138。新路线没有被完整记录过，但每一条边都来自已有记录。" : "先点离线规划，看两条旧记录怎样拼出更短的路线；再只保留 A，观察哪些东西无法从记录里得到。";
  }
  root.append(lessonHeader("03"), step("先看图像：旧记录可以拼，缺的连接不能猜", prose("离线：数据已经固定，不能向环境询问更多结果。在线：还可以行动，看到新的反馈，把它加入经验。这个区别描述数据能否继续增加，和 SARSA / Q-learning 里下一动作的选择是两个不同问题。", "两条路线如果经过同一个状态，可以在这个状态接起来。在当前确定性小世界里，只要每条连接都已观察到，便能规划一条没人完整走过的路线。对随机环境，少量记录不能当作已知的准确模型。")),
    step("动手验证：先拼接，再拿走一部分数据", h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, "A+B → 离线规划 → 只保留 A → 在线采集"), "先观察已有片段如何重组，再观察缺失片段怎样限制规划。唯一新增的条件是能否实际获取新经验。"),
      h("div", { class: "toolbar" }, button("提供记录 A+B", { onClick: () => load("both") }), button("只保留记录 A", { onClick: () => load("onlyA") }), planBtn, collectBtn), h("div", { class: "experiment-content" }, map.el, stats, note))),
    takeaway("离线学习可以重新组合数据里已有的经验，得到更好的路线；它无法验证数据之外的动作。在线行动能补上缺失连接，但必须决定去哪儿探索。"),
    step("检查理解", predict({ question: "拼接出的路线从未被完整记录过，为什么仍可用已有数据规划它？", choices: [{ label: "它的每条连接都有记录，而且片段在同一状态接上" }, { label: "学习器可以直接猜出没见过的连接" }], answer: 0, explain: "新的组合不等于新的连接。这里使用的是同一确定性环境中已观察到的片段，没有把未知动作假设为正确。" })),
    step("把整条主线连起来", prose("环境给出反馈 → 行动收集经验 → 用经验构造目标 → 更新 V 或 Q → 根据估计选动作 → 得到下一批经验。", "现在你应当能在一次具体更新中说清：数据从哪里来，目标用了什么，新估计如何影响下一次行动。")), lessonFooter("03"));
  load("both"); return () => {};
}
