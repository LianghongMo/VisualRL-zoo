import { GridWorld } from "../rl/envs/gridworld.js";
import { ExperienceGraph } from "../rl/tabular/experience-graph.js";
import { h, button, replace, segmented } from "../ui/dom.js";
import { lessonHeader, step, prose, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, route, statesOf, NEAR_ROUTE, FAR_ROUTE, discounted, num, metric, ARROWS, stateName } from "../ui/world-view.js";
import { equation } from "../ui/math.js";

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
  const env = GridWorld.chargingRoom(), A = route(env, NEAR_ROUTE), B = route(env, [0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0]);
  const junction = env.toState(3, 4), graph = new ExperienceGraph({ nStates: env.nStates, nActions: 4, gamma: 0.9 }), sources = new Map();
  let data = "both", planned = false, online = [], onlineState = env.start;
  const aMap = new WorldView(env, { caption: "记录 A：前 4 步到 J，再向下到 +1，共 5 步。青色前缀可以保留。" });
  const bMap = new WorldView(env, { caption: "记录 B：绕行 8 步到 J，再用 4 步到 +10，共 12 步。金色后缀可以接上。" });
  const bestMap = new WorldView(env, { caption: "规划结果：青色 = A 的前 4 步；金色 = B 的后 4 步。J 是同一个完整状态 (3,4)。? 只是没有观测，不等于不可走。" });
  const bNote = h("p", { class: "observation" }), stats = h("div", { class: "metrics" }), note = h("p", { class: "observation", role: "status" }), evidence = h("div", { class: "table-wrap" });
  function add(t, source) {
    graph.add(t); const key = `${t.state},${t.action}`;
    if (!sources.has(key)) sources.set(key, new Set()); sources.get(key).add(source);
  }
  function load(mode) {
    data = mode; planned = false; online = []; [onlineState] = env.reset();
    graph.edges.clear(); graph.visited.clear(); graph.terminal.clear(); graph.V.fill(0); graph.sweeps = 0; sources.clear();
    A.forEach(t => add(t, "A")); if (mode === "both") B.forEach(t => add(t, "B")); render();
  }
  const planBtn = button("离线：只用这些记录规划", { kind: "learn", onClick: () => { graph.plan(); planned = true; render(); } });
  const sweepBtn = button("离线：只更新一轮", { onClick: () => { graph.sweep(); planned = true; render(); } });
  const collectBtn = button("在线：沿示范路线采集一步", { kind: "env", onClick: () => {
    if (data !== "onlyA" || online.length >= FAR_ROUTE.length) return;
    const action = FAR_ROUTE[online.length], [next_state, reward, terminated] = env.step(action);
    const t = { state: onlineState, action, next_state, reward, terminated }; add(t, "新增在线经验"); online.push(t); onlineState = next_state;
    planned = false; render();
  } });
  function render() {
    const best = planned ? bestKnownRoute(env, graph) : [], reached = !!best.at(-1)?.terminated;
    const stitched = reached && data === "both" && best.length === 8 && best[3]?.next_state === junction;
    aMap.render({ paths: [{ states: statesOf(A), color: "var(--act)" }], selected: junction, labels: { [junction]: "J · 第 4 步" } });
    bMap.render({ paths: data === "both" ? [{ states: statesOf(B), color: "var(--reward)" }] : [], selected: junction, labels: { [junction]: "J · 第 8 步" } });
    bNote.textContent = data === "both" ? "同一个 J，B 的前缀比 A 多绕 4 步；后缀却能得到 +10。" : "记录 B 已从数据中移除。右图只展示地图轮廓，没有提供通向 +10 的经验。";
    const paths = stitched ? [
      { states: statesOf(best.slice(0, 4)), color: "var(--act)", width: 6 },
      { states: statesOf(best.slice(4)), color: "var(--reward)", width: 6 },
    ] : best.length ? [{ states: statesOf(best), color: "var(--learn)", width: 6 }] : online.length ? [{ states: statesOf(online), color: "var(--act)", width: 4 }] : [];
    bestMap.render({ paths, values: planned ? graph.V : null, known: graph.visited, robot: online.length ? onlineState : env.start, selected: junction, labels: { [junction]: "J · 拼接点" } });
    replace(stats, metric("记录 A", `${A.length} 步 / ${num(discounted(A, 0.9))}`), metric("记录 B", data === "both" ? `${B.length} 步 / ${num(discounted(B, 0.9))}` : "未提供"), metric("规划出的路线", planned ? reached ? `${best.length} 步 / ${num(discounted(best, 0.9))}` : "估计尚未连到终点" : "等待规划"), metric("已知连接", graph.edges.size), metric("新增行动", `${online.length} 步`), metric("价值更新", `${graph.sweeps} 轮`), metric("V(J)", num(graph.V[junction])), metric("V(S)", num(graph.V[env.start])));
    collectBtn.disabled = data !== "onlyA" || online.length >= FAR_ROUTE.length;
    const rows = best.map((t, i) => h("tr", { class: t.state === junction ? "highlight" : "" }, h("td", {}, i + 1), h("td", {}, `${stateName(env, t.state)} ${ARROWS[t.action]} ${stateName(env, t.next_state)}`), h("td", {}, t.reward), h("td", {}, [...sources.get(`${t.state},${t.action}`)].join(" / "))));
    replace(evidence, h("h3", {}, "新路线的逐边证据"), best.length ? h("table", {}, h("thead", {}, h("tr", {}, ["步", "已有连接", "奖励", "来自哪条记录"].map(x => h("th", {}, x)))), h("tbody", {}, rows)) : h("p", {}, "先规划。每条结果边都要能指出数据来源；第 5 步将从 J 改用 B 的动作。"));
    note.textContent = online.length ? `已采集 ${online.length}/8 步；${planned ? "已根据当前数据更新估计；检查规划结果与新增经验的对应。" : "采集只加经验，V 尚未更新。再点离线规划，利用新增连接。"}${online.length === 8 ? "通向 +10 的片段已经齐全。" : "这条示范路线由你指定，学习器尚未自动决定探索方向。"}` : data === "onlyA" ? "只给 A 时，多做价值更新仍无法验证通向 +10 的动作。先规划，再在线逐步补数据，最后重新规划。" : stitched ? "拼出 A 前 4 步 + B 后 4 步 = 8 步。每条边都有旧记录，但这 8 步组合从未作为完整回合提供给学习器。" : "估计仍在传播。+10 的信息到第 4 轮传到 J，到第 8 轮传到 S；早期贪心路线可能仍是较长的旧路线。继续逐轮更新，或直接规划到稳定。";
  }
  const counterexample = h("p", { class: "observation", role: "status" });
  function batteryMatch(same) { counterexample.textContent = same ? "A 到 J：(位置 3,4；电量 90%)。B 从 J 开始：(位置 3,4；电量 90%)。完整状态一致；在同一任务和动力学下，后缀的条件后果可以复用。" : "A 到 J：(位置 3,4；电量 10%)。B 从 J 开始：(位置 3,4；电量 90%)。屏幕位置相同，完整状态不同；B 的 4 步后缀可能耗电超过 10%，不能据此保证 A 能继续走到终点。"; }
  const batteryPicker = segmented([{ value: true, label: "交点电量也相同" }, { value: false, label: "同一位置，但电量不同" }], { value: true, label: "独立反例：完整状态是否匹配", onChange: batteryMatch }); batteryMatch(true);
  root.append(lessonHeader("03"),
    step("先看图像：旧记录可以重组", prose("在线学习还能向环境行动、获取新反馈；离线学习只能读取已经固定的数据。数据中可以没有最好的一整条轨迹，但已有片段可能组成更好的策略。这种能力称为 trajectory stitching：在兼容的状态处，把不同轨迹里的决策片段接起来。", "沿 A 更快到达交点 J，再从 J 改用 B 的动作。后缀不需要知道你之前从哪里来，这是第一章 Markov 状态的直接用途。这里每条已观察边的后果确定且不变，可以在这些边上做精确规划。")),
    step("动手验证：先拼接，再拿走一部分数据", h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, "A+B → 离线规划 → 只保留 A → 补数据 → 再规划"), "先分别看两条旧路线，在 J 拼接。每一步检查来源，再通过计数器验证：多计算与多采集改变的东西不同。"), h("div", { class: "toolbar" }, button("提供记录 A+B", { onClick: () => load("both") }), button("只保留记录 A", { onClick: () => load("onlyA") }), sweepBtn, planBtn, collectBtn),
      h("div", { class: "experiment-content" }, h("div", { class: "comparison" }, h("section", { class: "comparison-card" }, h("h3", {}, "旧记录 A · 短前缀，去 +1"), aMap.el), h("section", { class: "comparison-card" }, h("h3", {}, "旧记录 B · 长前缀，去 +10"), bMap.el, bNote)), h("h3", {}, "新策略的路线 · 在 J 改选动作"), bestMap.el, stats, note, evidence))),
    step("stitching：价值怎样跨过交点？", equation("G_{\\mathrm{stitched}}=G_{\\mathrm{prefix}}+\\gamma^m G_{\\mathrm{suffix}}", "前缀有 m 次转移。Gprefix 只累计前缀这 m 个奖励；Gsuffix 从交点重新以权重 1 起算，所以要整体乘 γᵐ。"),
      h("div", { class: "arithmetic" }, "A 前缀：4 步，奖励都是 0。B 后缀：4 步，奖励 [0,0,0,10]，Gsuffix = 0.9³×10 = 7.29。拼接后 G = 0 + 0.9⁴×7.29 = 4.783。A 原路线：0.9⁴×1 = 0.656；B 原路线：0.9¹¹×10 = 3.138。"),
      equation("V_{k+1}(J)=\\max_{a\\in\\mathcal{A}_D(J)}[r(J,a)+\\gamma V_k(s')],\\qquad V^*(J)=\\max(1,7.29)=7.29", "在当前已观测子图规划到稳定：J 向下去 +1 的动作值 1；向右接 B 后缀的动作值 7.29。这里 V* 指这个子图的最优价值。"),
      prose("算法不是先找整条最好记录，再按文件把两条记录粘起来。Bellman 更新先把 B 后缀的价值传回 J，让 J 改选向右；再把 J 的新价值沿 A 的较短前缀传回 S。最后逐状态贪心选动作，就出现这条新组合。片段接合是局部价值传播的结果。", "A 与 B 到 J 的时间不同：A 用 4 步，B 用 8 步。任务本身没有时间相位，且策略只依赖状态，因此这个区别不妨碍复用后缀。但它改变了从 S 衡量后缀的折扣，短前缀更有价值。")),
    step("什么时候真的可以拼？", prose("要匹配完整 Markov 状态，不能只匹配屏幕坐标。任务目标、奖励规则、动作含义和动力学也要一致；后缀中的动作必须有数据支持。终止后的轨迹不能再当作可继续行动的片段。", "如果不同数据记录使用不同电量、携带物品、目标或时间相位，要把这些影响后果的信息纳入状态，并确认匹配。下面是独立的电量反例，不修改上方只有位置状态的地图。"), batteryPicker, counterexample),
    step("数据覆盖：多算几轮也补不出缺失的边", prose("只保留 A 后，J 只记录了向下到 +1。去 +10 的连接没有数据支持，本例因此不把它们放进 max；反复规划仍只能得到 A。在线示范采集会实际调用环境 step，新连接加入数据；再学习，才能把 +10 的信息利用起来。采集和学习现在有独立按钮、独立计数。", "这是确定性、表格型、限制在已观察动作上的规划示范，展示离线数据为什么能支持 stitching；它不是通用离线 RL 算法。随机转移要估计后果分布；连续状态可能没有精确重合；函数近似也可能对数据外动作高估。一般离线 RL 还要处理覆盖不足、分布偏移和估计误差，不能把一次成功记录当作可靠保证。")),
    h("p", { class: "source-note" }, "进一步阅读：", h("a", { href: "https://arxiv.org/abs/2005.01643", target: "_blank", rel: "noopener" }, "Levine 等 · Offline RL Tutorial"), "（固定数据与分布偏移）；", h("a", { href: "https://arxiv.org/abs/2110.06169", target: "_blank", rel: "noopener" }, "Kostrikov 等 · Implicit Q-Learning"), "（数据外动作估值与离线策略改进）。上面的格子世界算例用于解释拼接，不是这些论文算法的实现。"),
    takeaway("新路线可以没有完整记录，但组成它的动作后果必须有依据，且片段在兼容的完整状态处接上。Bellman 传播完成重组；在线行动补充数据。"),
    step("检查理解", predict({ question: "新 8 步路线为什么能由旧数据支持？", choices: [{ label: "每条边有记录，片段在同一个完整状态 J 接上" }, { label: "学习器猜出了数据中不存在的连接" }], answer: 0, explain: "新组合使用已有连接，未凭空创造连接。数据覆盖与 Markov 状态是这里的关键条件。" })),
    step("把整条主线连起来", prose("先用 MDP 定义任务 → 用策略产生经验 → 用回报定义 V/Q → 用 Bellman 分解未来 → 选择 MC 或 TD 目标 → 用 SARSA 或 Q-learning 更新 Q、改进动作 → 判断数据还能增加，还是只能在固定数据上重组。", "对任何一次更新都能回答：状态是否完整？奖励属于哪一步？后续按哪个策略？目标由真实回报还是估计构成？终点是否正确处理？下一动作有没有数据支持？这七个问题把整章串成一条逻辑。")), lessonFooter("03"));
  load("both"); return () => {};
}
