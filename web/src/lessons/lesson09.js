import { goalRoom, goalReference, shortestRoute, relabelEpisode, discountedFuture, balancedClassifier, waypointPlan } from "../rl/goal-conditioned.js";
import { QLearning } from "../rl/tabular/q-agents.js";
import { h, button, replace, segmented, slider } from "../ui/dom.js";
import { lessonHeader, step, prose, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, statesOf, num, metric, ARROWS } from "../ui/world-view.js";
import { equation } from "../ui/math.js";

export const THESIS = "https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf";
const cite = (label, page) => h("p", { class: "source-note" }, "论文对应：", h("a", { href: `${THESIS}#page=${page}`, target: "_blank", rel: "noopener" }, label));
const cellName = (env, s) => `(${env.toCell(s).join(",")})`;
const table = (head, rows) => h("table", {}, h("thead", {}, h("tr", {}, head.map(x => h("th", { scope: "col" }, x)))), h("tbody", {}, rows.map(row => h("tr", {}, row.map(x => h("td", {}, x))))));
const experiment = (title, instruction, controls, ...content) => h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, title), instruction), h("div", { class: "toolbar" }, controls), h("div", { class: "experiment-content" }, content));

function goalExperiment(env, left, right) {
  let goal = left, query = env.start, mode = "state";
  const viewEnv = Object.create(env), stats = h("div", { class: "metrics" }), rows = h("div", { class: "table-wrap" }), note = h("p", { class: "observation", role: "status" });
  const map = new WorldView(viewEnv, { caption: "黄色 +1 是当前目标 g，青点是查询状态 s。线路绕过墙，是这个目标的最短路径。地图上数字为已知模型算出的 V*(s,g)。", onSelect: s => {
    if (mode === "goal") goal = s; else query = s; render();
  } });
  const goalPicker = segmented([{ value: left, label: "目标：左上角" }, { value: right, label: "目标：右上角" }], { value: left, label: "指定目标 g", onChange: g => { goal = g; render(); } });
  function render() {
    viewEnv.goals = [goal]; viewEnv.goalReward = 1; goalPicker.select(goal);
    const { V, Q } = goalReference(env, goal), tr = shortestRoute(env, query, goal), best = Math.max(...Q[query]);
    const actions = query === goal ? [] : Q[query].flatMap((q, a) => Math.abs(q - best) < 1e-9 ? [a] : []);
    map.render({ robot: query, values: V, selected: query, paths: [{ states: statesOf(tr), color: "var(--act)" }] });
    replace(stats, metric("查询状态 s", cellName(env, query)), metric("指定目标 g", cellName(env, goal)), metric("这里该往哪走", actions.length ? actions.map(a => ARROWS[a]).join(" / ") : "已到目标"), metric("V*(s,g)", num(V[query])), metric("到目标的最短路线", `${tr.length} 步`));
    replace(rows, table(["先做动作 a", "物理后果 s′", "r_g / done_g", "Q*(s,a,g)"], Array.from({ length: 4 }, (_, a) => {
      const [next] = env.move(query, a); return [ARROWS[a], cellName(env, next), query === goal ? "已终止，无下一步" : `${Number(next === goal)} / ${next === goal}`, num(Q[query][a])];
    })));
    note.textContent = query === goal ? "本实验在首次到达目标时终止，V(g,g)=0。+1 已经在进入目标的那一步收到；不是之后还能再收一次。" : `保持 s=${cellName(env, query)}，只改 g，比较四个动作。墙和转移 p(s′|s,a) 没变；奖励、终止和价值跟随目标改变。这里用已知模型精确规划作参照，尚未训练共享的神经网络。`;
  }
  const controls = [goalPicker, segmented([{ value: "state", label: "点地图：选状态 s" }, { value: "goal", label: "点地图：选目标 g" }], { value: mode, label: "点击地图的作用", onChange: x => { mode = x; } }), button("查询出发点 S", { onClick: () => { query = env.start; render(); } }), button("查看目标状态", { onClick: () => { query = goal; render(); } })];
  render();
  return experiment("保持位置，只改目标", "先比较左上与右上目标：S 的第一步从 ← 变成 →。再选一个查询位置，或点击空白格指定新目标。", controls, map.el, stats, rows, note);
}

function relabelExperiment(env, left, right) {
  const observed = shortestRoute(env, env.start, left).map((t, i, a) => ({ ...t, reward: 0, terminated: false, truncated: i === a.length - 1, next_action: null }));
  let k = 5, replay = [], learner, cursor, trace = null;
  const stats = h("div", { class: "metrics" }), rows = h("div", { class: "table-wrap" }), update = h("p", { class: "observation", role: "status" });
  const map = new WorldView(env, { caption: "五条青色边都真实观察过。蓝色前缀用于当前改标目标；灰色后缀仍在原始记录里，但不属于改标后的终止回合。S 是出发点。" });
  const replayBtn = button("用改标经验更新一条", { kind: "learn", onClick: () => {
    if (cursor < 0) return; trace = learner.learnStep(replay[cursor--]); render();
  } });
  function reset() {
    const g = observed[k - 1].next_state;
    replay = relabelEpisode(observed, g); learner = new QLearning({ nStates: env.nStates, nActions: 4, alpha: 1, gamma: 0.9 }); cursor = replay.length - 1; trace = null; render();
  }
  function render() {
    const g = observed[k - 1].next_state;
    const labels = Object.fromEntries(observed.map((t, i) => [t.next_state, i === k - 1 ? `g′ · 第 ${i + 1} 步` : `第 ${i + 1} 步`])); labels[right] = "原目标 g";
    map.render({ robot: env.start, selected: g, labels, paths: [{ states: statesOf(observed), color: "var(--wall)", width: 7 }, { states: statesOf(replay), color: "var(--learn)" }] });
    replace(stats, metric("真实观察的转移", "5 条"), metric("改标后回合长度", `${replay.length} 步`), metric("Q(S,←,g′)", num(learner.Q[env.start][3])), metric("逆序更新次数", learner.learnSteps));
    replace(rows, table(["步", "真实 (s,a,s′) · 不改", "原 g：r / done", "新 g′：r / done", "本次是否用于更新"], observed.map((t, i) => [i + 1, `${cellName(env, t.state)} ${ARROWS[t.action]} ${cellName(env, t.next_state)}`, "0 / false", i < k ? `${replay[i].reward} / ${replay[i].terminated}` : "回合已终止", i >= k ? "排除后缀" : i > cursor ? "已更新" : "等待逆序重放"])));
    replayBtn.disabled = cursor < 0;
    update.textContent = trace ? `更新 ${cellName(env, trace.state)} ${ARROWS[trace.action]}：目标 y = ${trace.reward} + 0.9 × ${num(trace.bootstrap_value)} = ${num(trace.target)}；旧 Q ${num(trace.value_before)}，δ = ${num(trace.error)}，α=1，新 Q ${num(trace.value_after)}。${trace.terminated ? "新目标处已终止，后续项为 0。" : "后续估计来自刚才重放过的下一条经验。"}${cursor < 0 ? ` 这轮完成：Q(S,←,g′)=0.9^${k - 1}=${num(0.9 ** (k - 1))}。` : ""}` : `原本要去右上 ${cellName(env, right)}，却只记录了去左上 ${cellName(env, left)} 的 5 步。原任务没有成功；第 5 步是记录时长用完的截断，不是原目标的终止。现在取第 ${k} 步到达的状态作 g′，重新计算奖励和成功终止，再从末端向前传播。切换 g′ 会从零演示另一张 Q(·,·,g′) 表。`;
  }
  const choose = slider({ id: "gcrl-relabel-step", label: "把第几步到达的状态当作新目标？", min: 1, max: 5, step: 1, value: k, format: n => `第 ${n} 步`, onInput: n => { k = n; reset(); } });
  reset();
  return experiment("失败的任务，留下了成功的子任务", "默认改标为第 5 步：连续更新 5 次，观察 +1 传回 S。然后改标为第 2 步，检查后 3 条为何必须排除。", [choose, replayBtn, button("重新演示当前目标", { onClick: reset })], map.el, stats, rows, update);
}

function densityExperiment(env, left) {
  const tr = shortestRoute(env, env.start, left), future = tr.map(t => t.next_state);
  let gamma = 0.9, negative = "uniform", selected = left;
  const stats = h("div", { class: "metrics" }), rows = h("div", { class: "table-wrap" }), actionRows = h("div", { class: "table-wrap" }), note = h("p", { class: "observation", role: "status" }), formula = h("div", { class: "arithmetic" });
  const map = new WorldView(env, { caption: "固定后续策略 μ：沿最短路径去左上角，再一直停留。地图展示先做 ← 后的未来分布。每格数字是随机未来时刻 T 看到它的概率；并非所有目标策略的价值。" });
  function render() {
    const p = discountedFuture(future, gamma), q = new Map(future.map((s, i) => [s, negative === "uniform" ? 0.2 : i === 0 ? 0.6 : 0.1]));
    const values = Array(env.nStates).fill(0); for (const [s, prob] of p) values[s] = prob;
    map.render({ values, selected, paths: [{ states: statesOf(tr), color: "var(--act)" }] });
    const prob = p.get(selected), neg = q.get(selected), c = balancedClassifier(prob, neg);
    replace(stats, metric("未来分布总和", num([...p.values()].reduce((a, b) => a + b, 0))), metric("查询的未来状态 g", cellName(env, selected)), metric("真正的 pγ(g|S,←)", num(prob)), metric("分类器 C*(F=1|S,←,g)", num(c.probability)), metric("contrastive 分数 f*", num(c.logit)));
    replace(rows, table(["未来状态 · 选择查询", "pγ：正例分布", "q：负例分布", "C*", "f*=log(pγ/q)"], future.map((g, i) => {
      const result = balancedClassifier(p.get(g), q.get(g));
      return [button(`${i + 1} · ${cellName(env, g)}${g === left ? " 停留点" : ""}`, { onClick: () => { selected = g; render(); } }), num(p.get(g)), num(q.get(g)), num(result.probability), num(result.logit)];
    })));
    formula.textContent = `pγ = q × C*/(1−C*) = ${num(neg)} × ${num(c.odds)} = ${num(prob)}。也等于 q × exp(f*)。`;
    replace(actionRows, table(["在 S 先做哪个动作？", "pγ(g|S,a)", "同一个 q(g)", "f*(S,a,g)"], Array.from({ length: 4 }, (_, a) => {
      const [next] = env.move(env.start, a), after = shortestRoute(env, next, left);
      const mass = discountedFuture([next, ...after.map(t => t.next_state)], gamma).get(selected) ?? 0;
      return [a === 3 ? `${ARROWS[a]} · 分数最高` : ARROWS[a], num(mass), num(neg), num(balancedClassifier(mass, neg).logit)];
    })));
    note.textContent = `γ=${num(gamma)}：前 4 个状态各接收 (1−γ)γ⁰、…、(1−γ)γ³；第 5 个状态吸收全部剩余尾部 γ⁴=${num(gamma ** 4)}。只改变负采样 q，pγ 一点不变，C* 和 f* 却会变化。负例是“独立抽来的状态”，不是“绝对到不了的状态”；一个状态可以同时出现在正例与负例中。这里显示精确 Bayes 解，便于核对，并未运行神经网络训练。`;
  }
  const discount = slider({ id: "gcrl-future-gamma", label: "未来时刻的折扣 γ", min: 0.2, max: 0.95, step: 0.05, value: gamma, format: n => num(n, 2), onInput: n => { gamma = n; render(); } });
  const negPicker = segmented([{ value: "uniform", label: "负例 q：五格均匀" }, { value: "frequent", label: "负例 q：第一格更常见" }], { value: negative, label: "独立负例分布", onChange: x => { negative = x; render(); } });
  render();
  return experiment("在一个随机未来时刻，机器人会在哪里？", "先减小 γ，看概率向近处移动。再只改变负采样分布，观察停留点的 pγ 保持不变，分类概率和分数却变了。点表中的按钮查询每个未来状态。", [discount, negPicker], map.el, stats, rows, formula, h("h3", {}, "固定查询目标，再比较动作"), actionRows, h("p", { class: "observation" }, "四个候选动作之后都按同一个 μ 继续去左上并停留。向右需要先返回 S，多走两步；向上或向下撞墙，晚一步才继续。所以默认 g=左上、γ=0.9 时，← 的概率 0.6561，→ 为 0.531441，↑/↓ 为 0.59049。q 对四个动作相同，分数排序与未来概率排序一致。改进后续策略后，还需要重新估计 critic；这里先验证一次动作比较。"), note);
}

function searchExperiment(env, left) {
  const route = shortestRoute(env, env.start, left), w1 = route[1].next_state, w2 = route[3].next_state;
  let keep = true, robot = env.start, plan, targetIndex = 1, actions = 0;
  const stats = h("div", { class: "metrics" }), graph = h("div", { class: "physical-flow gcrl-waypoints" }), note = h("p", { class: "observation", role: "status" });
  const map = new WorldView(env, { caption: "全局目标始终是左上 g。W₁、W₂ 是经验库里的路标。局部控制器每次只接收当前子目标；同一套 π(a|s,g_local) 被反复调用。" });
  const nextBtn = button("执行局部控制器一步", { kind: "env", onClick: () => {
    if (!plan.path || robot === left) return;
    const target = plan.path[targetIndex], local = shortestRoute(env, robot, target);
    [robot] = env.move(robot, local[0].action); actions++;
    if (robot === target) targetIndex++;
    render();
  } });
  function reset() { robot = env.start; actions = 0; targetIndex = 1; plan = waypointPlan(env, robot, left, keep ? [w1, w2] : [w2], 2); render(); }
  function name(s) { return s === env.start ? "S" : s === left ? "g" : s === w1 ? "W₁" : "W₂"; }
  function render() {
    const finished = robot === left, target = finished || !plan.path ? null : plan.path[targetIndex];
    map.render({ robot, selected: target, labels: { [left]: "g · 最终目标", [w1]: keep ? "W₁" : "缺少 W₁", [w2]: "W₂" }, paths: plan.path ? [{ states: statesOf(route), color: "var(--learn)" }] : [] });
    replace(graph, plan.path ? plan.path.flatMap((s, i) => [i ? h("span", { class: "flow-arrow" }, `→ ${shortestRoute(env, plan.path[i - 1], s).length} 步 →`) : null, h("b", {}, name(s))]) : h("strong", {}, "S 与可用路标之间没有完整的局部可达链"));
    replace(stats, metric("直接去最终目标", "5 步 > H=2"), metric("当前局部目标", finished ? "已完成" : target === null ? "无可用路径" : name(target)), metric("实际执行动作", `${actions} 步`), metric("路标搜索距离", plan.path ? `${plan.distance} 步` : "无路径"));
    nextBtn.disabled = !plan.path || finished;
    note.textContent = !plan.path ? "移除 W₁ 后，S 到剩下的 W₂ 要 4 步，到 g 要 5 步，都超过局部范围 H=2。搜索无法补出缺失的局部连接，不能据此保证能到 g。原始 SoRB 在无图路径时退回直接调用策略；这里暂停，单独展示覆盖条件。" : finished ? "已到最终目标：全局搜索提供了 S→W₁→W₂→g，局部控制器执行了 2+2+1=5 步。换子目标没有换物理环境；计划中的连接最终由真实动作执行。" : `现在机器人在 ${cellName(env, robot)}，全局目标仍是 g，输入策略的局部目标是 ${name(target)}=${cellName(env, target)}。本演示把局部可靠范围设为 H=2，并用精确地图距离代替学习到的距离；不是完整 SoRB 训练。`;
  }
  const dataPicker = segmented([{ value: true, label: "经验库有 W₁ 和 W₂" }, { value: false, label: "移除路标 W₁" }], { value: keep, label: "经验库路标覆盖", onChange: x => { keep = x; reset(); } });
  reset();
  return experiment("一个远目标，拆成几个可控制的近目标", "控制器只能可靠解决两步以内的子目标。先沿路标执行五步，观察局部目标 W₁→W₂→g；再移除 W₁，检查哪里断开。", [dataPicker, nextBtn, button("重新从 S 出发", { onClick: reset })], map.el, graph, stats, note);
}

export function mount(root) {
  const env = goalRoom(), left = env.toState(1, 1), right = env.toState(1, 7);
  root.append(lessonHeader("09"),
    h("p", { class: "source-note" }, "主线参考：", h("a", { href: THESIS, target: "_blank", rel: "noopener" }, "Benjamin Eysenbach · Probabilistic Reinforcement Learning（CMU 博士论文，2023）"), "。这一章选取第 2、3、4 章：目标控制、未来状态预测、分类与对比学习，以及经验库上的搜索。"),
    step("GCRL：把目标也放进问题里", prose("之前机器人面对一个固定任务。现在你可以说“去左上角”，也可以说“去右上角”。同一个位置、同一套动作和墙，合理的第一步却不同。Goal-Conditioned Reinforcement Learning（GCRL，目标条件强化学习）把期望结果 g 作为策略和价值函数的输入，让一个学习系统处理一族目标任务。", "普通状态 s 回答“现在是什么情况”；目标 g 回答“希望变成什么情况”。目标可以是位置、物体姿态，或一张希望机器人达到的图像。这里令 g 是格子位置，达到它就成功终止。每回合先从目标分布 p_G 抽取 g；也可以在使用时由你指定。目标在回合内保持不变。"),
      equation(String.raw`g\sim p_G,\quad a_t\sim\pi_\theta(\cdot\mid s_t,g),\quad s_{t+1}\sim p(\cdot\mid s_t,a_t)`, "策略依赖目标；本章物理转移不依赖目标。若目标改变了真实动力学，就必须把它纳入相应的环境模型。"),
      equation(String.raw`r_g(s,a,s')=\mathbf{1}\{s'=g\},\quad d_g(s')=\mathbf{1}\{s'=g\}`, "本例首次进入目标奖励 +1，其他步奖励 0，成功即终止；d_g 是这次转移的终止标记。一般任务可以用目标区域或容差判定成功。"),
      equation(String.raw`Q^\pi(s,a,g)=\mathbb{E}_{\pi(\cdot\mid\cdot,g)}\!\left[\sum_{j=0}^{\tau_g-1}\gamma^j r_{t+j+1,g}\,\middle|\,s_t=s,a_t=a\right]`, "τ_g 表示还要多少步首次到达目标。Q 不只是在数字旁加一个 g：它计算的是这个目标的奖励，并按这个目标的策略继续走。共享网络可在不同目标间复用表示；表格也可以分别存每个目标的 Q。"), goalExperiment(env, left, right),
      prose("S 到左上或右上都需要 5 步，γ=0.9 时 V*(S,g)=0.9⁴=0.6561；左上任务先向左，右上任务先向右。只给 π(s) 而不给 g，就无法同时表达这两种要求。形式上也可把 (s,g) 看成扩展状态，g 的转移是保持原值，因而仍是一个 MDP。"), cite("第 3 章 §3.3，印刷页 24–25：目标条件任务与策略", 42)),
    step("目标改标：物理经验不变，任务重新计算", prose("稀疏奖励的困难：还没到目标，整段反馈都是 0。机器人可能没完成你要求的右上目标，却已经成功到过左边的若干位置。Hindsight relabeling 的办法是回头把已达到的状态当作新目标。HER（Hindsight Experience Replay）是常见的这种经验重放方法。它让失败记录提供别的任务的监督。", "一条物理转移 (s,a,s′) 可以服务多个目标。改标时，换 g 并重新计算 r_g；若任务在成功时终止，还要重新计算 d_g。不能保留原目标的奖励或终止标记，也不能把新目标之后的后缀接到一个已经终止的回合里。"),
      equation(String.raw`(s,a,r_g,s',d_g,g)\ \longrightarrow\ (s,a,r_{g'}(s,a,s'),s',d_{g'}(s'),g')`, "s、a、s′ 都是原来的真实观察；只有任务条件与任务反馈改变。改标不是让机器人再走一遍，也没有产生新的物理连接。"),
      equation(String.raw`y_g=r_g+\gamma(1-d_g)\max_{a'}Q(s',a',g)`, "这是目标条件 Q-learning 的一步目标。和上一章同一个 Bellman 结构，但下一状态必须继续查询同一个 g。"),
      equation(String.raw`Q(s,a,g)\leftarrow Q(s,a,g)+\alpha\,[y_g-Q(s,a,g)]`, "一般终止事件也必须进入 d_g。仅因记录时长用完而截断，不能自动当作目标成功或把所有后续价值设为 0。"), relabelExperiment(env, left, right),
      prose("本例转移确定，目标只改变奖励与成功条件，因此可以直接改标。随机环境中，“挑中了这次碰巧到达的未来状态”还会同时挑中特定随机结果，可能改变学习样本的后果分布。朴素的未来目标重放不自动成为无偏的 Q 学习；论文第 2 章分析了这类问题。这也是 Ben 转向明确的未来状态分布和分类目标的动机。", "上一章要求同一任务才能直接拼接带奖励的片段。这里可以跨目标复用物理转移，是因为先按当前目标重新计算了奖励与终止；完成改标后，拼接才发生在统一的新任务下。不能把不同目标的旧回报或 Q 值直接相加。GCRL 是问题设定，HER 是数据重放方式。你可以用目标条件 Q-learning 搭配 HER，也可以用别的目标条件算法；GCRL 本身不规定数据是在线采集还是离线固定。"),
      h("p", { class: "source-note" }, "改标原论文：", h("a", { href: "https://arxiv.org/abs/1707.01495", target: "_blank", rel: "noopener" }, "Andrychowicz 等 · Hindsight Experience Replay（2017）")), cite("第 2 章 §2.3、§2.5.3，印刷页 7–8、15：重放与估计对象", 25)),
    step("Ben 的物理图像：预测随机未来时刻的状态", prose("把任务换一种问法：在 s 先做 a，以后按一个固定的后续策略 μ 行动，未来会在哪里？不只预测下一步，也预测多步之后。为了让近处与远处可比较，先随机选一个未来时刻 T，再问那时机器人在哪。γ 越大，抽到较远未来的机会越大。", "这里明确采用 T∈{1,2,…} 的几何分布，把“下一状态”的权重放在 γ⁰。这与上面的奖励索引一致，也让所有状态的概率和严格等于 1。预测的对象依赖后续策略；改变策略，未来分布也随之改变。"),
      equation(String.raw`\Pr(T=k)=(1-\gamma)\gamma^{k-1},\qquad k=1,2,\ldots`, "这是随机观察时刻，不是机器人随机选择目标，也不是环境动作打滑。"),
      equation(String.raw`p_\gamma^\mu(g\mid s,a)=(1-\gamma)\sum_{k=1}^{\infty}\gamma^{k-1}\Pr_\mu(s_{t+k}=g\mid s_t=s,a_t=a)`, "固定 μ，在离散状态空间这是一份归一化的未来状态概率分布。实验中机器人 5 步后一直停留在最后一格，尾部概率因此全部累积在那里。"), densityExperiment(env, left),
      equation(String.raw`r^{\mathrm{occ}}_g(s,a,s')=(1-\gamma)\mathbf{1}\{s'=g\},\qquad Q^{\mu,\mathrm{occ}}_g(s,a)=p_\gamma^\mu(g\mid s,a)`, "这里是另一种任务：无限时域、按 μ 继续走，每次下一状态为 g 就得到 1−γ；不在首次到 g 时截断求和。代入回报定义即可得到右侧等式，参见论文 Eq. 3.1、3.6。"),
      prose("要区分两种 Q：上方 Q 是一次性 +1、首次到达就终止的回报；这里 Qocc 是折扣状态占用。它们一般不是同一个量。这个固定路径例子在第 5 步到 g 后永远停在那里，γ⁴ 的尾部占用恰好也等于一次性任务的 γ⁴ 回报；换成经过 g 后离开的策略，两者就会不同。更不能把不同目标各自最优策略下的 Q*(s,a,g) 放在一起，默认它们加起来等于 1。", "连续状态里，精确命中一个点通常概率为 0；概率密度却可以非零，甚至超过 1。论文用未来状态密度讨论连续目标，不能把密度当作离散成功概率。实际成功判定常用目标区域或容差；图片目标还需要能识别期望结果的表示。"), cite("第 2 章 §2.3–2.4；第 3 章 §3.4.1 与附录 B.3.1：未来分布与 Q 的关系", 44)),
    step("C-learning：把未来预测变成分类，再递归", prose("如何从数据估计未来分布？对同一个输入 x=(s,a)，准备两类候选状态：正例 g⁺ 来自这条轨迹的折扣未来；负例 g⁻ 独立抽自背景分布 q。分类器 C(x,g) 判断“这个配对来自正例源还是负例源”。两类采样各占一半时，Bayes 公式给出下面的关系。论文用经验中的未来状态边缘分布作背景；上方实验用两种简单 q 方便计算。"),
      equation(String.raw`C^*(x,g)=\frac{p_\gamma^\mu(g\mid x)}{p_\gamma^\mu(g\mid x)+q(g)}`, "C* 是来源分类概率。它会随背景 q 改变；它不直接等于目标成功概率。若类别比例不为 1:1，关系中还要加入类别先验。"),
      equation(String.raw`p_\gamma^\mu(g\mid x)=q(g)\frac{C^*(x,g)}{1-C^*(x,g)}`, "当 q(g)>0 时，用 odds（C/(1−C)）乘背景分布才能还原未来概率或密度。固定目标 g，比动作时 q(g) 对所有动作相同，所以不必估计它也能比较动作。"),
      prose("如果只用整条轨迹的未来做正例，得到的是采集这些轨迹的策略的未来；它不会因为我们想换策略就自动变成新策略的估计。C-learning 的关键进一步是递归：随机观察时刻有 1−γ 的机会恰好落在下一步，剩下 γ 的机会可以从下一状态重新预测。这正是概率形式的 Bellman 分解。"),
      equation(String.raw`p_\gamma^\mu(g\mid s,a)=(1-\gamma)p(s'=g\mid s,a)+\gamma\mathbb{E}_{s',\,a'\sim\mu}[p_\gamma^\mu(g\mid s',a')]`, "先平均物理下一状态 s′，再按目标后续策略抽 a′。换成目标策略 π(a′|s′,g) 可做相应目标的递归估计；一跳的转移来自旧数据，后续动作来自要评价的策略。"),
      equation(String.raw`w=\frac{C(s',a',g)}{1-C(s',a',g)}`, "对独立背景目标 g∼q，这个权重估计下一状态未来密度与背景密度之比。"),
      equation(String.raw`\mathcal L_C=-\mathbb E\!\left[(1-\gamma)\log C(s,a,s')+\gamma\operatorname{sg}[w]\log C(s,a,g)+\log(1-C(s,a,g))\right]`, "C-learning 的递归分类损失（论文 Eq. 2.7）：第一项以真实下一状态为正例，第二项用权重接上更远未来，第三项为背景负例。sg 表示计算本次梯度时固定权重；不是把普通 MC 正例标签直接当成 off-policy 修正。"),
      prose("本章的概率实验展示 MC 分类的精确目标，不运行这个递归分类器。真正的 C-learning 要训练分类器、用目标策略选动作，并随策略改变更新未来预测；其中加权自举让它能利用旧转移估计新的后续策略。"), cite("第 2 章 §2.5.1–2.5.2，印刷页 10–14，Eq. 2.1–2.7：分类、递归与策略", 28)),
    step("Contrastive RL：学表示，也学会选动作", prose("Ben 论文第 3 章把分类器写成两个表示的内积。φ(s,a) 编码“从这里做这个动作”；ψ(g) 编码“想要的未来结果”。如果这个动作确实使某个结果更常出现在未来，训练会提高这对输入的分数。墙的两侧视觉上很近，却可能需要绕行；好的表示应反映受动作与动力学约束的可达关系，而不只是坐标距离。"),
      h("div", { class: "physical-flow" }, h("b", {}, "状态 s + 候选动作 a"), "→ φ(s,a)", h("span", { class: "flow-arrow" }, "·"), "ψ(g) ←", h("b", {}, "目标 g"), "→ 分数 f → 比较候选动作"),
      equation(String.raw`f_\theta(s,a,g)=\phi_\theta(s,a)^\top\psi_\theta(g),\qquad C_\theta=\sigma(f_\theta),\quad\sigma(z)=\frac{1}{1+e^{-z}}`, "f 是任意实数的 logit；σ(f) 才是来源分类概率。exp(f) 则是密度比估计，不是归一化的成功概率。"),
      equation(String.raw`\mathcal L_{\mathrm{NCE}}=-\mathbb E[\log\sigma(f(s,a,g^+))+\log(1-\sigma(f(s,a,g^-)))]`, "正例是这对 (s,a) 的折扣未来状态；负例是独立采样的未来状态。论文的批内实现把配对放在对角线、交叉配对放在非对角线。这是 NCE 二元分类目标，不是把所有相似度直接当作 Q 数值。"),
      equation(String.raw`f^*(s,a,g)=\log\frac{p_\gamma^{\bar\mu}(g\mid s,a)}{q(g)},\qquad e^{f^*}=\frac{Q_g^{\bar\mu,\mathrm{occ}}(s,a)}{q(g)}`, "当分类拟合达到理想解时成立；Q 使用上一节的占用奖励。μ̄ 表示采集数据所对应的混合后续策略，论文 §3.3、§3.4.2 对它作了定义。"),
      equation(String.raw`\max_\theta\ \mathbb E_{s,g,\,a\sim\pi_\theta(\cdot\mid s,g)}[f(s,a,g)],\qquad a_{\mathrm{greedy}}\in\arg\max_a f(s,a,g)`, "固定 s,g，比动作即可：−log q(g) 是共同常数。实际连续动作算法用 actor 梯度优化这个期望；这里的 argmax 是离散候选动作的解释。"),
      prose("用上方分类实验检验数值：γ=0.9，停留点 pγ=0.6561，均匀背景 q=0.2，所以 C*=0.766、f*=log(3.2805)=1.188。把 q 改为 0.1 后，pγ 仍是 0.6561，C*≈0.868、f*≈1.881。分数较高可能是因为背景里目标较稀有；不能跨不同目标把 raw 分数当成成功率直接比较。", "完整循环是：采集目标条件轨迹 → 用未来正例与独立负例更新 critic 表示 → 用分数改进目标条件 actor → 再采集数据。Critic 输入必须包含动作，否则只能判断未来相关性，无法比较“现在做哪一个动作”。", "还有一个容易跳过的条件：重放数据通常来自不同目标。此时 MC contrastive critic 对应的是这些数据诱导的混合策略 μ̄ 的未来分布，不自动等于“始终追求查询目标 g”的 Q^{π_g}，更不是 Q*。论文 §3.4.5 在额外假设下讨论策略改进与收敛；第 3 章的 NCE 方法本身是对采集策略的 MC 估计，不能仅凭用了 replay buffer 就称它为严格 off-policy。第 2 章的加权递归处理的是另一层问题。"), cite("第 3 章 §3.4.2–3.4.5，印刷页 26–30，Eq. 3.7–3.8 与 Lemma 1", 44)),
    step("接回 stitching：SoRB 用近目标组成远目标", prose("预测到了价值，并不意味着一个局部策略能稳定走到任意远目标。论文第 4 章的 Search on the Replay Buffer（SoRB）把经验库中的状态当作路标：用目标条件价值判断哪些路标之间能可靠到达，在这张有向图上搜索，再依次把路标交给同一个目标条件控制器。", "SoRB 原文用另一套便于估距离的奖励：每步 −1，γ=1，到目标终止。对一个能有限步到达目标的固定策略，Vπ 的负值对应期望到达步数；达到最优时才对应最短的期望步数。这里不能拿前面 +1、γ=0.9 的 Q 原样当负距离。"),
      equation(String.raw`r_g=-1,\quad\gamma=1\quad\Longrightarrow\quad V^\pi(s,g)=-\mathbb E_\pi[\tau_g\mid s],\qquad d^*(s,g)=-V^*(s,g)`, "要求可达且期望到达时间有限。学习到的 V 只有在估计可靠时才提供可靠距离；有限训练时长、失败状态与分布近似需要额外处理。论文 §4.3 用分布型 RL 改善距离估计。"),
      equation(String.raw`s\to w_1\to\cdots\to g=\operatorname{ShortestPath}(\mathcal G_D),\qquad a_t\sim\pi(\cdot\mid s_t,g_{\mathrm{local}})`, "节点来自经验库；边由局部可达性与估计距离支持。全局目标 g 不变；当前子目标 glocal 随到达路标而更新。原算法会搜索并比较直接去目标与去路标的距离。"), searchExperiment(env, left),
      h("div", { class: "table-wrap" }, table(["上一章 stitching", "本章 SoRB"], [["在兼容的完整状态处复用不同记录的片段", "以经验库状态作路标，复用能达到近目标的控制器"], ["Bellman 传播让新策略组成旧边", "图搜索选择路标，目标条件策略执行局部连接"], ["需要数据支持，缺失的边不能凭空补出", "需要可靠的局部可达性；路标与距离估计可能不足"]])),
      prose("这两者共享“局部能力组成新组合”的图像，但实现方式有区别。SoRB 不要求两条原轨迹精确在一个状态相交：学到的局部控制器可能提供新路标连接。也正因如此，图上的边是控制器能力的估计，必须考虑估值误差与实际失败；经验库里有两个状态不意味着它们之间一定有可靠的边。", "第 1 章的要求仍然成立：匹配完整 Markov 状态。如果路标只记录位置，却漏掉电量、门状态或所持物体，一段看起来能走的局部任务仍可能无法执行。改目标也不能违背动作含义或物理可达性。"), cite("第 4 章 §4.2–4.3，印刷页 42–45：距离、经验库图、搜索与控制", 60)),
    step("论文怎么继续读？", h("div", { class: "table-wrap" }, table(["先带着什么问题", "对应位置", "读完应能说明"], [["为什么从 reward 转到未来状态预测？", h("a", { href: `${THESIS}#page=25`, target: "_blank", rel: "noopener" }, "第 2 章 §2.3–2.5（页 7–15）"), "分类器如何还原密度，递归项怎样复用旧转移"], ["表示内积为什么能指导动作？", h("a", { href: `${THESIS}#page=42`, target: "_blank", rel: "noopener" }, "第 3 章 §3.3–3.4（页 24–30）"), "NCE 密度比、背景常数、混合策略与 actor 目标"], ["局部策略如何到远处？", h("a", { href: `${THESIS}#page=60`, target: "_blank", rel: "noopener" }, "第 4 章 §4.2–4.3（页 42–45）"), "负价值与步数、路标图、距离误差与执行"], ["概率与 Q 的等式到底用了哪些定义？", h("a", { href: `${THESIS}#page=104`, target: "_blank", rel: "noopener" }, "附录 B.3（页 86–88）"), "从奖励与折扣求和推出占用量，而非靠命名等同"]])),
      prose("阅读时先标出五件事：goal 的表示是什么？到达如何判定？critic 估计的是哪个策略、哪种奖励？正负例从哪里抽？当前是 MC 估计、递归自举，还是在图上规划？把这些问题逐一对齐，GCRL、HER、C-learning、contrastive RL 与 SoRB 就能放在同一条逻辑里。")),
    takeaway("目标 g 指定希望发生的未来。GCRL 让动作和价值依赖 g；改标复用物理经验，分类与对比学习估计未来关系，SoRB 再把可靠的近目标控制接成远目标计划。每一步都要说清策略、奖励、采样与可达性。"),
    step("检查理解", predict({ question: "负采样中某个目标更少见，contrastive 分数提高，能否说到达它的概率一定提高了？", choices: [{ label: "不能：还要看背景 q 和分数对应的策略" }, { label: "能：f 就是成功概率" }], answer: 0, explain: "f*=log(pγ/q)。只降低 q 也会提高 f；上方实验中物理轨迹和 pγ 都没变。固定同一个目标时，q 对不同动作相同，才可以用分数比较动作。" }), predict({ question: "改标为第 2 步到达的目标，首次成功即终止；原记录的后 3 步怎么处理？", choices: [{ label: "不属于这个新目标的终止回合，排除后缀" }, { label: "仍接在后面，沿用原来的 done 标记" }], answer: 0, explain: "任务终止在第 2 步，奖励与终止都要重新计算。后 3 步仍是原始物理记录，可用于其他目标，但不能作为这个已经结束的回合的继续。" })), lessonFooter("09"));
  return () => {};
}
