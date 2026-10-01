import viSource from "../../../visualrl/algorithms/tabular/value_iteration.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { ValueIteration, optimalBackup, optimalValues } from "../rl/tabular/dp.js";
import { h, s, button, segmented, slider, replace } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, ARROWS, stateName, num, metric, discounted } from "../ui/world-view.js";
import { equation } from "../ui/math.js";
import { codeBlock } from "../ui/code.js";
import { extractDef } from "../ui/source.js";

// Geometric demonstration arrows, independent of Bellman value estimates.
// Avoid cliff resets and the other absorbing goal when routing to this goal.
function arrowsToGoal(env, model, goal) {
  const distance = Array(env.nStates).fill(Infinity);
  distance[goal] = 0;
  for (let i = 0; i < env.nStates; i++) {
    let changed = false;
    for (let state = 0; state < env.nStates; state++) {
      if (model.terminal[state]) continue;
      for (let action = 0; action < env.nActions; action++) {
        const [next, reward, done] = env.move(state, action);
        if (reward < 0 || (done && next !== goal)) continue;
        if (distance[next] + 1 < distance[state]) {
          distance[state] = distance[next] + 1; changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return distance.map((d, state) => {
    if (model.terminal[state] || !Number.isFinite(d)) return null;
    return [1, 0, 2, 3].find(action => {
      const [next, reward, done] = env.move(state, action);
      return reward >= 0 && (!done || next === goal) && distance[next] === d - 1;
    }) ?? null;
  });
}
function follow(env, policy) {
  let state = env.start;
  const states = [state], transitions = [], seen = new Set(states);
  let loop = false;
  for (let i = 0; i < env.nStates && policy[state] !== null && policy[state] !== undefined; i++) {
    const action = policy[state], [next_state, reward, terminated] = env.move(state, action);
    transitions.push({ state, action, next_state, reward, terminated }); states.push(next_state);
    if (terminated) break;
    if (seen.has(next_state)) { loop = true; break; }
    seen.add(next_state); state = next_state;
  }
  // Displayed loops have only zero rewards. Detecting one is not termination.
  return { states, transitions, done: transitions.at(-1)?.terminated ?? false, loop, value: discounted(transitions, 0.9) };
}
function branchDiagram({ origin, branches, caption }) {
  return h("figure", { class: "bellman-branches" },
    h("div", { class: "bellman-branch-origin" }, origin),
    h("div", { class: "bellman-branch-list" }, branches.map(b =>
      h("div", { class: "bellman-branch" },
        h("div", { class: "bellman-branch-edge" }, h("strong", {}, b.action), h("span", {}, "→ r = " + b.reward)),
        h("div", { class: "bellman-branch-state" }, h("strong", {}, b.destination), h("span", {}, b.future)),
        h("strong", { class: "bellman-branch-total" }, b.total)))),
    h("figcaption", {}, caption));
}
export function mount(root) {
  const env = GridWorld.chargingRoom(), model = env.model(), gamma = 0.9;
  const junction = env.toState(3, 4), farGoal = env.toState(1, 6);
  const template = arrowsToGoal(env, model, farGoal), reference = optimalValues(model, gamma);

  // Show a full policy and its trajectory before any backups or equations.
  let candidate = 2, walked = 0;
  const scene = new WorldView(env, { caption: "整张箭头图是一套策略；金框是交点 J。青色线是从 S 按当前策略走出的路线。× 是悬崖：−10 并返回 S。" });
  const policyStats = h("div", { class: "metrics" });
  const policyCalculation = h("div", { class: "arithmetic", id: "policy-return", role: "status" });
  const policyNote = h("p", { class: "observation", role: "status" });
  const walkButton = button("按这套策略走一步", { kind: "env", onClick: () => { walked++; renderPolicy(); } });
  const policyChoice = segmented([
    { label: "策略 A · J 向下", value: 2 }, { label: "策略 B · J 向左", value: 3 }, { label: "策略 C · J 向右", value: 1 },
  ], { value: candidate, label: "比较三套候选策略", onChange: action => { candidate = action; walked = 0; renderPolicy(); } });
  function renderPolicy() {
    const policy = [...template]; policy[junction] = candidate;
    const { states, transitions, done, loop, value } = follow(env, policy);
    walked = Math.min(walked, transitions.length);
    const finished = walked === transitions.length;
    scene.render({ policy, selected: junction, paths: [{ states }], robot: finished && done ? null : states[walked] });
    const [x, y] = scene.center(junction);
    scene.svg.append(s("text", { x, y: y + 20, class: "map-mini", "text-anchor": "middle" }, "J"));
    walkButton.disabled = finished;
    walkButton.textContent = finished ? done ? "已经进入充电站" : "已识别重复状态" : "按这套策略走一步";
    replace(policyStats, metric("当前策略的结果", loop ? "零奖励循环" : transitions.length + " 步到 " + (candidate === 2 ? "+1" : "+10")),
      metric("从 S 出发的回报", num(value)), metric("机器人已经执行", walked + " 步"));
    policyCalculation.textContent = loop
      ? "S → J → 左边一格 → J → …；奖励始终为 0，所以 G = 0。没有到达终点。"
      : candidate === 2 ? "奖励 [0, 0, 0, 0, 1]：G = 0.9⁴ × 1 = 0.6561"
        : "奖励 [0, 0, 0, 0, 0, 0, 0, 10]：G = 0.9⁷ × 10 = 4.782969";
    policyNote.textContent = candidate === 2
      ? "A 能完成任务，但不是这三套策略中回报最大的。切换 C：只把 J 的箭头从 ↓ 改成 →，看看整条路线怎样改变。"
      : candidate === 3 ? "B 在 J 向左；左边一格的固定箭头又把它带回 J。策略是反复使用的规则，所以每次回到 J 都会再次向左，不会自动醒悟。"
        : "C 的回报最大。远处的 +10 足以补偿多走三步的折扣。现在只证明了 C 优于 A、B；要确认全局最优，还要检查所有状态的所有动作。";
  }
  const intro = step("我们的任务：找一个最优策略",
    h("div", { class: "experiment bellman-policy-scene" },
      h("div", { class: "experiment-grid" }, scene.el, h("div", { class: "experiment-reading" },
        h("h3", {}, "先改箭头，看看完整路线"), policyChoice, policyStats, policyCalculation,
        h("div", { class: "bellman-walk-controls" }, walkButton, button("从 S 重新执行", { kind: "ghost", onClick: () => { walked = 0; renderPolicy(); } })), policyNote))),
    prose("任务已经确定：从 S 出发，进入任一充电站就结束；普通移动奖励为 0，折扣 γ = 0.9。我们要找到一套动作规则，让预期累计奖励尽可能大。最短路线不是目标，最大的折扣回报才是。",
      "策略 π 是给每个可行动状态指定动作的规则，地图上的整套箭头就是一个确定性策略。路线只是从某个起点执行策略后产生的结果；换个起点，仍然使用同一套箭头。",
      "本例把 J 以外的箭头预先固定成避开悬崖、朝 +10 行走的示范，只改变 J 的决定。先比较完整策略的结果，再想办法摆脱逐条枚举。这里已知整张地图与所有动作后果，属于已知模型下的规划。"));
  const header = lessonHeader("05"), essentials = header.querySelector(".essentials");
  essentials.remove(); header.append(intro, essentials); intro.classList.add("bellman-opening");

  const localCases = [
    { label: "① 终点前一格", origin: "A · 终点前一格 (2,6)", branches: [
      { action: "↑ 进入 +10", reward: "+10", destination: "充电站 / 终止", future: "之后没有动作：V = 0", total: "10 + 0.9 × 0 = 10" },
    ], caption: "最后一步：+10 属于进入终点的那条边；终点之后的价值是 0。", tex: "10+0.9\\times 0=10",
      text: "这里没有余下的路线要算。动作一执行就收到 +10，然后终止。不要再把 +10 当成终点价值加第二次。" },
    { label: "② 向前退一格", origin: "B · 再往前一格 (3,6)", branches: [
      { action: "↑ 移动一格", reward: "0", destination: "下一格 A = (2,6)", future: "后面再 ↑ 到 +10，剩余回报为 10", total: "0 + 0.9 × 10 = 9" },
    ], caption: "把已经算好的后半段当作一个数。当前动作只负责它自己的奖励和一次折扣。", tex: "0+0.9\\times 10=9",
      text: "原来的两步回报是 0 + 0.9 × 10。既然知道下一格沿这个后续走法能拿到 10，就不必重新展开整条路线。这就是用下一状态的价值概括未来。" },
    { label: "③ 到交点选动作", origin: "站在 J = (3,4)", branches: [
      { action: "↓ 进入 +1", reward: "+1", destination: "近处充电站 / 终止", future: "之后没有动作：V = 0", total: "1 + 0.9 × 0 = 1" },
      { action: "→ 移动一格", reward: "0", destination: "下一格 C = (3,5)", future: "随后 → ↑ ↑ 到 +10：V = 0.9² × 10 = 8.1", total: "0 + 0.9 × 8.1 = 7.29" },
    ], caption: "当前两个动作都只看一步，但下一格的价值已经包含后面整段路线。", tex: "\\max\\{1+0.9\\times 0,\\;0+0.9\\times 8.1\\}=7.29",
      text: "如果能自己选动作，就选 7.29 对应的向右。向上也有一条 4 步到 +10 的路线，同样得到 7.29；最优动作可以并列。先用两条示范分支理解 max，后面再检查四个动作。" },
  ];
  const localMap = new WorldView(env, { caption: "金框是正在计算的位置。A、B、C、J 把右侧示意图中的状态放回同一张地图；线展示本例所接的后续路线。" });
  const A = env.toState(2, 6), B = env.toState(3, 6), C = env.toState(3, 5);
  const localPicture = h("div", { id: "bellman-local-picture" }), localEquation = h("div"), localText = h("p", { class: "observation" });
  const localChoice = segmented(localCases.map((c, i) => ({ label: c.label, value: i })), {
    value: 0, label: "从完整路线推到一步计算", onChange: renderLocal,
  });
  function renderLocal(i) {
    const c = localCases[i]; replace(localPicture, branchDiagram(c));
    localMap.render({ selected: [A, B, junction][i], labels: { [A]: "A", [B]: "B", [C]: "C", [junction]: "J" },
      paths: i === 0 ? [{ states: [A, farGoal] }] : i === 1 ? [{ states: [B, A, farGoal] }] :
        [{ states: [junction, env.toState(4, 4)], color: "var(--reward)" }, { states: [junction, C, B, A, farGoal] }] });
    replace(localEquation, equation(c.tex, "同一条规则：眼前奖励 + 0.9 × 下一格的后续价值。")); localText.textContent = c.text;
  }
  const jBackup = optimalBackup(model, reference.V, junction, gamma);
  const allActions = h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["J 的动作", "一步奖励 + 最优后续", "动作价值"].map(x => h("th", {}, x)))),
    h("tbody", {}, jBackup.actions.map(a => {
      const b = a.branches[0];
      return h("tr", { class: jBackup.best_actions.includes(a.action) ? "highlight" : "" },
        h("td", {}, ARROWS[a.action]), h("td", {}, num(b.reward) + " + 0.9 × " + num(b.next_value)), h("td", {}, num(a.q)));
    }))));

  const randomDiagram = h("div"), randomResult = h("div", { class: "arithmetic", role: "status" }), randomStats = h("div", { class: "metrics" });
  function renderRandom(p) {
    const risky = p * (1 + gamma * 4) + (1 - p) * -2;
    replace(randomDiagram, branchDiagram({ origin: "选择冒险动作 A", branches: [
      { action: "成功 · p = " + num(p, 2), reward: "+1", destination: "成功后的状态", future: "已知最优后续价值 V* = 4", total: "1 + 0.9 × 4 = 4.6" },
      { action: "失败 · p = " + num(1 - p, 2), reward: "−2", destination: "失败 / 终止", future: "终止后 V* = 0", total: "−2 + 0.9 × 0 = −2" },
    ], caption: "这是单独的随机小任务。动作 B 则确定收到 +3 并终止。概率 p 属于环境，你只能选 A 或 B。" }));
    randomResult.textContent = num(p, 2) + " × 4.6 + " + num(1 - p, 2) + " × (−2) = " + num(risky);
    replace(randomStats, metric("A · 先平均两种后果", num(risky)), metric("B · 确定终止奖励", "3"),
      metric("再选择动作", Math.abs(risky - 3) < 1e-10 ? "A、B 并列" : risky > 3 ? "A · 冒险" : "B · 安全"));
  }
  const probability = slider({ id: "bellman-success", label: "环境给 A 成功后果的概率 p", min: 0, max: 1, step: 0.05, value: 0.8, format: v => v.toFixed(2), onInput: renderRandom });

  let vi = new ValueIteration(model, gamma), selected = junction, trace = null;
  const map = new WorldView(env, { caption: "数字是当前估计 Vₖ。箭头是本轮备份选出的动作；蓝框表示数字改变。点击一个空白格，检查它的下一次备份。",
    onSelect: state => { selected = state; renderIteration(); } });
  const stats = h("div", { class: "metrics" }), table = h("div", { class: "table-wrap" });
  const note = h("p", { class: "observation", role: "status" }), calc = h("div", { class: "arithmetic" });
  function sweep() { trace = vi.sweep(); renderIteration(); }
  function renderIteration() {
    const Q = trace?.q_values;
    const policy = model.terminal.map((terminal, state) => terminal || !Q || Math.max(...Q[state]) <= 0 ? null : Q[state].indexOf(Math.max(...Q[state])));
    const candidateRoute = follow(env, policy), stable = trace && trace.max_change < 1e-10;
    const changed = new Set(trace ? trace.values_after.flatMap((v, state) => Math.abs(v - trace.values_before[state]) > 1e-10 ? [state] : []) : []);
    map.render({ values: vi.V, policy, selected, changed,
      paths: stable && candidateRoute.done ? [{ states: candidateRoute.states, color: "var(--act)", width: 3 }] : [] });
    const backup = optimalBackup(model, vi.V, selected, gamma);
    replace(table, h("h3", {}, stateName(env, selected) + " 的下一次更新"), h("table", {},
      h("thead", {}, h("tr", {}, ["动作", "眼前奖励 + 下一格当前估计", "结果"].map(x => h("th", {}, x)))),
      h("tbody", {}, backup.actions.map(a => {
        const b = a.branches[0];
        return h("tr", { class: backup.best_actions.includes(a.action) ? "highlight" : "" },
          h("td", {}, ARROWS[a.action]), h("td", {}, num(b.reward) + " + 0.9 × " + num(b.next_value)), h("td", {}, num(a.q)));
      }))));
    calc.textContent = "Vₖ(" + stateName(env, selected) + ") = " + num(vi.V[selected]) + "；下一轮取四个目标的 max，得到 " + num(backup.value_after);
    replace(stats, metric("已经更新所有位置", vi.sweeps + " 轮"),
      metric("出发点价值 V(S)", num(vi.V[env.start]), stable ? "已稳定为 V*(S)" : "这是 Vₖ(S)，仍在计算"),
      metric("交点 J 的本轮动作", policy[junction] === null ? "尚未传到" : ARROWS[policy[junction]]),
      metric("稳定策略从 S 出发", stable ? candidateRoute.transitions.length + " 步 / " + num(candidateRoute.value) : "继续更新后验收"));
    note.textContent = stable
      ? "所有位置都不再变化。用这些 V* 为每个状态选最大动作目标，得到一个最优策略 π*；青色线验收它从 S 出发的实际路线：8 步到 +10，回报 4.783。并列最优动作可以选任意一个。"
      : vi.sweeps === 0 ? "先把未知的 V* 全设成 0，不预先填答案。点一轮更新：只有一步能拿到奖励的位置先得到正数。"
        : vi.sweeps < 4 ? "每轮只读取上一轮的数字。J 在第 1 轮可以看到眼前的 +1；远处 +10 的后续信息还没有完整到达 J。"
          : vi.sweeps < 5 ? "第 4 轮：V₄(J) = 7.29，J 已知道向上或向右接远处路线更好。出发点还远，V₄(S) = 0；这不等于说从 S 真的只能得到 0。"
            : vi.sweeps < 8 ? "第 5 轮起，近处 +1 的信息已到达 S，Vₖ(S) = 0.656。J 的选择已经改善，但 +10 的价值还没传完四步公共前缀。中间估计 Vₖ 不是当前整套箭头的精确 Vπ。"
              : "第 8 轮起，+10 的信息到达 S，Vₖ(S) = 4.783。两条示范路线的第一步都是向上；真正的分支选择发生在 J。继续更新到所有位置稳定，再验收最优策略。";
  }

  root.append(header,
    step("先定义最优，再找更省事的比较方法",
      prose("在这个确定性例子中，固定策略从 S 出发只有一种结果，因此 Vπ(S) 就是刚才计算的回报。一般情况下，策略或环境可能随机，比较的是预期回报。",
        "最优价值 V* 是所有策略能取得的最高价值；最优策略 π* 是取得这些价值的动作规则。我们希望不止给 S 找一条好路线，还能在任意状态重新开始时都知道怎么走。最优策略可能不唯一。"),
      equation("V^*(s)=\\max_\\pi V^\\pi(s),\\qquad V^{\\pi^*}(s)=V^*(s)\\ \\text{for every }s",
        "V* 是最大的预期回报；π* 是达到它的策略。有限状态、有限动作、折扣 γ < 1 的 MDP 中，存在这样的确定性最优策略。"),
      prose("笨办法是列出所有箭头图，分别走完或评价，再比较回报。本地图有很多可行动格，每格四个动作，策略数量随格子数指数增长。更好的办法：如果下一格的最佳后续已经算好，当前位置还需要重复展开整条路线吗？")),
    step("从路线中总结：一步奖励，接上后续价值",
      prose("依次看三个具体位置。先处理最后一步，再往前退，最后回到 J 选择分支。前两例的后续走法已经指定；到 J 时，再比较各分支能接上的最佳后续。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, localChoice),
        h("div", { class: "experiment-content" }, h("div", { class: "bellman-local-grid" }, localMap.el, localPicture), localEquation, localText)),
      h("div", { class: "physical-flow" }, h("strong", {}, "完整路线的回报"), h("b", {}, "→ 拆成两段 →"),
        h("span", {}, "眼前一步的奖励 r"), h("b", {}, "+"), h("span", {}, "后半段价值 V(s′) × γ")),
      equation("G_t=r_t+\\gamma G_{t+1}\\quad\\Longrightarrow\\quad Q^*(s,a)=r(s,a)+\\gamma V^*(s')",
        "这里先写确定性环境：动作 a 决定唯一的下一状态 s′；进入终点时后续值为 0。用 V* 接后半段，表示之后都能按最优策略行动。"),
      prose("现在 max 有了物理含义：每个动作都有一条「一步奖励 + 最优后续」的支路，选择其中回报最大的。把同样的比较用在每个状态，就不必枚举整张箭头图。"),
      equation("V^*(s)=\\max_a\\big[r(s,a)+\\gamma V^*(s')\\big]",
        "这就是确定性版本的 Bellman 最优方程。它是最优价值的自洽关系，尚不是一条已知答案的计算指令。"),
      h("h3", {}, "回到 J：把四个动作都检查一次"), allActions,
      prose("上表的后续值由完整模型求得，用来验算前面的图像。↑ 与 → 都得到 7.29；↓ 只有 1；← 得到 5.905。这里的 ← 只固定第一步，之后切回最优走法，所以会返回 J 再去 +10。它与策略 B「每次到 J 都向左」的零回报循环不同。")),
    step("固定策略的价值：平均动作，而不是替它改主意",
      prose("如果 J 的策略规定只能向下，就必须执行它，Vπ(J) = 1。即便向右更好，也不能在评价这套策略时偷偷改动作。评价问「按这套规则会得到多少」；控制问「应该采用哪套规则」。",
        "再给一套随机策略：每次到 J，以 80% 概率向右、20% 概率向下；其他状态沿示范箭头走。向右分支的后续回报仍为 7.29，向下为 1，J 的价值就是它们的概率平均。"),
      branchDiagram({ origin: "固定 π：在 J 抽动作", branches: [
        { action: "80% 选 →", reward: "0", destination: "去远处站", future: "该动作接此策略的后续：Qπ(J,→) = 7.29", total: "0.8 × 7.29" },
        { action: "20% 选 ↓", reward: "+1", destination: "近处站 / 终止", future: "Qπ(J,↓) = 1", total: "0.2 × 1" },
      ], caption: "这里的概率来自策略 π；环境仍然确定。每次只执行一个动作，Vπ 描述多次执行的平均。" }),
      equation("V^\\pi(J)=0.8\\times 7.29+0.2\\times 1=6.032",
        "固定策略取平均，得到 6.032；允许改进策略时，全部选择向右（或并列最优的向上）可得到 7.29。"),
      equation("V^\\pi(s)=\\sum_a\\pi(a\\mid s)\\big[r(s,a)+\\gamma V^\\pi(s')\\big]",
        "确定性环境的 Bellman 期望方程：后续也必须是同一个 π 的价值。π 的动作概率固定，不能替换成 max。")),
    step("随机时先平均，再比较动作",
      prose("下一步不总由你决定。现在看一个独立的随机任务：冒险动作 A 可能成功，也可能失败；安全动作 B 确定收到 +3 并终止。成功之后的最优后续值 4 是本算例给定的边界值。先把 A 的两种后果按环境概率平均，再和 B 比。",
        "把 p 从 0.80 调到 0.60。最优动作会改变，但你始终不能只取成功那一支的 4.6，因为失败是该动作真实后果的一部分。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, probability),
        h("div", { class: "experiment-content" }, randomDiagram, randomResult, randomStats)),
      equation("Q^*(s,A)=p(1+0.9\\times 4)+(1-p)(-2+0.9\\times 0),\\qquad V^*(s)=\\max\\{Q^*(s,A),3\\}",
        "先按 p 平均环境后果，再对 A、B 取 max。p = 0.8 时选 A（3.28）；p = 0.6 时选 B（3 > 1.96）。")),
    step("Bellman 方程：把刚才的图像写完整",
      prose("现在可以把三个例子合在一起：r 是眼前一步；V(s′) 概括后半段；p 平均你无法挑选的环境后果；π 平均固定策略的动作；max 则让你选择最好的动作。每个符号都对应刚才图中的一部分。"),
      equation("V^\\pi(s)=\\sum_a\\pi(a\\mid s)\\sum_{s',r}p(s',r\\mid s,a)\\big[r+\\gamma V^\\pi(s')\\big]",
        "Bellman 期望方程：给定策略 π，评价它。外层平均选出的动作，内层平均该动作的环境后果。"),
      equation("V^*(s)=\\max_a\\sum_{s',r}p(s',r\\mid s,a)\\big[r+\\gamma V^*(s')\\big]",
        "Bellman 最优方程：选择动作，但仍然平均环境后果。终点之后的 V* = 0，终点奖励只计入进入它的那条转移。"),
      equation("Q^*(s,a)=\\sum_{s',r}p(s',r\\mid s,a)\\big[r+\\gamma\\max_{a'}Q^*(s',a')\\big]",
        "Q* 版本：当前动作 a 已固定；到下一状态后再选最好的 a′。这将成为后面 Q-learning 更新目标的来源。"),
      equation("\\pi^*(s)\\in\\arg\\max_a\\sum_{s',r}p(s',r\\mid s,a)\\big[r+\\gamma V^*(s')\\big]",
        "从价值拿到策略：为每个状态选择最大动作目标。多个动作并列时任选一个，就得到一个最优策略。"),
      prose("关键理由是「最优后续」：如果某条所谓最优路线在下一状态还采用较差的后续，就把那个后续换成更好的，当前回报也会提高。因此最优策略的每段后续都必须能接上相应状态的最优价值。位置是充分的 Markov 状态，这种替换才能只依赖 s′，而不需要整段历史。")),
    step("动手验证：每次只传播一条边",
      prose("方程两侧都含未知的 V*，不能假装下一格的最佳未来早已知道。价值迭代从 V₀ = 0 开始，把右侧的 V* 暂时换成旧估计 Vₖ，反复计算所有状态，直到数字稳定。箭头随最大动作目标一起出现，最终给出我们最初要找的策略。",
        "同步更新时，一整轮都读取同一份旧数字。所以 +10 的信息先到它前一格，再到更远的格子，逐步传回 S。机器人不需要真的走这条路，这是利用已知模型计算。"),
      equation("V_{k+1}(s)=\\max_a\\sum_{s',r}p(s',r\\mid s,a)\\big[r+\\gamma V_k(s')\\big]",
        "这是价值迭代的更新式，区别于真实 V* 满足的自洽方程。有限 MDP、γ < 1、有界奖励下，它收敛到唯一的 V*。"),
      h("div", { class: "experiment" },
        h("div", { class: "experiment-instruction" }, h("strong", {}, "看第 1、4、5、8 轮，再继续到稳定"),
          "先看 J 的动作怎样从近处奖励转向远处奖励，再看价值何时传到 S。最后验收整张箭头图。"),
        h("div", { class: "toolbar" }, button("更新所有位置一轮", { kind: "learn", onClick: sweep }),
          button("继续更新到稳定", { onClick: () => { for (let i = 0; i < 100; i++) { trace = vi.sweep(); if (trace.max_change < 1e-10) break; } renderIteration(); } }),
          button("把价值清零", { kind: "ghost", onClick: () => { vi = new ValueIteration(model, gamma); trace = null; renderIteration(); } })),
        h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, table, calc, note))),
      prose("Vₖ 是计算中的中间估计，不是当前整套箭头的精确 Vπ；局部动作可能先改善，远处起点的数字稍后才跟上。第 5 轮 S 得到的是 +1 的信息，第 8 轮才完整收到 +10 的信息。两条示范路线从 S 都先向上，分歧在 J，不能把这说成「S 的第一步突然换方向」。")),
    takeaway("我们的目标是找到最优策略。先把每个动作的未来拆成「一步奖励 + 下一状态价值」，随机后果取平均，动作选最大；这就是 Bellman 最优方程。未知价值通过迭代求出，再把每格的最佳动作连成 π*。"),
    step("检查理解",
      predict({ question: "J 向下的目标是 1；向右的目标是 7.29。固定策略规定各选一半时，Vπ(J) 是多少？", choices: [
        { label: "4.145：按固定策略平均" }, { label: "7.29：直接取最大的" },
      ], answer: 0, explain: "评价不能替策略改主意：0.5 × 1 + 0.5 × 7.29 = 4.145。允许优化动作规则时，才用 max 得到 7.29。" }),
      predict({ question: "冒险动作成功目标 4.6、失败目标 −2，成功概率 0.6；安全动作确定得到 3。最优动作是什么？", choices: [
        { label: "安全动作：3 大于冒险动作的期望 1.96" }, { label: "冒险动作：成功时能得到 4.6" },
      ], answer: 0, explain: "你能选择动作，不能选择环境给出哪个后果。先算 0.6 × 4.6 + 0.4 × (−2) = 1.96，再比较动作。" }),
      predict({ question: "第 4 轮 J 已选到通向 +10 的动作，但 V₄(S) = 0。这说明从 S 真的只能得到 0 吗？", choices: [
        { label: "不能：V₄ 是信息尚未传完的中间估计" }, { label: "能：地图上的数字就是当前策略的精确价值" },
      ], answer: 0, explain: "价值迭代还在进行。同步备份每轮传播一条边，第 8 轮 +10 的信息才到达 S；稳定后 V* 才能与最优策略配对。" })),
    optional("展开：价值迭代的真实实现", codeBlock(extractDef(viSource, "sweep", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "ValueIteration.sweep" })),
    optional("选读：先评价策略，再改进策略", prose("还有另一条求解路线：固定整套箭头求 Vπ，再根据它改箭头，然后重新评价。评价与改进反复交替，得到策略迭代。"), h("a", { href: "#lesson-06" }, "打开策略迭代实验 →")),
    lessonFooter("05"));
  renderPolicy(); renderLocal(0); renderRandom(0.8); renderIteration();
  return () => {};
}
