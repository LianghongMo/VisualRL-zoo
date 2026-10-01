import tdSource from "../../../visualrl/algorithms/tabular/td.py";
import mcSource from "../../../visualrl/algorithms/tabular/monte_carlo.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { Trajectory } from "../rl/core.js";
import { TD0 } from "../rl/tabular/td0.js";
import { MonteCarlo } from "../rl/tabular/monte-carlo.js";
import { h, button, replace, slider } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { route, NEAR_ROUTE, num, routeStrip, metric } from "../ui/world-view.js";
import { equation } from "../ui/math.js";
import { codeBlock } from "../ui/code.js";
import { extractDef } from "../ui/source.js";

export function mount(root) {
  const env = GridWorld.chargingRoom(), full = route(env, NEAR_ROUTE), names = ["S", "A", "B", "C", "D", "终点"];
  let episode, td, mc, index = 0, replayIndex = -1, selected = -1, lastTD = null, lastMC = null, mcDone = false, alpha = 1;
  const tdStrip = h("div"), mcStrip = h("div"), stats = h("div", { class: "metrics" });
  const tdCalc = h("div", { class: "arithmetic" }), mcCalc = h("div", { class: "arithmetic" });
  const experienceTable = h("div", { class: "table-wrap" }), mcTable = h("div", { class: "table-wrap" });
  const observed = h("div"), status = h("p", { class: "observation", role: "status" });
  function newEpisode() {
    episode = new Trajectory(); index = 0; replayIndex = -1; selected = -1;
    lastTD = null; lastMC = null; mcDone = false; env.reset(); render();
  }
  function reset() {
    td = new TD0({ nStates: env.nStates, alpha, gamma: 0.9 });
    mc = new MonteCarlo({ nStates: env.nStates, alpha, gamma: 0.9 }); newEpisode();
  }
  function act() {
    if (episode.done) return;
    const t = full[index++], [next_state, reward, terminated] = env.step(t.action);
    episode.append({ ...t, next_state, reward, terminated }); selected = index - 1; replayIndex = index - 1; render();
  }
  const actBtn = button("沿固定路线走一步", { kind: "env", onClick: act });
  const tdBtn = button("TD：学习选中的一步", { kind: "learn", onClick: () => { if (selected < 0) return; lastTD = td.learnStep(episode.at(selected)); render(); } });
  const mcBtn = button("MC：学完整回合", { kind: "learn", onClick: () => { if (!episode.terminated || mcDone) return; lastMC = mc.learnEpisode(episode); mcDone = true; render(); } });
  const replayBtn = button("TD：从终点向前重放一条", { onClick: () => { if (!episode.terminated || replayIndex < 0) return; selected = replayIndex--; lastTD = td.learnStep(episode.at(selected)); render(); } });
  const alphaControl = slider({ id: "prediction-alpha", label: "学习率 α（用于下一次更新）", min: 0.1, max: 1, step: 0.1, value: alpha, format: v => v.toFixed(1), onInput: v => { alpha = v; td.alpha = v; mc.alpha = v; render(); } });
  const th = labels => h("thead", {}, h("tr", {}, labels.map(x => h("th", {}, x))));
  function render() {
    actBtn.disabled = episode.done; tdBtn.disabled = selected < 0; mcBtn.disabled = !episode.terminated || mcDone; replayBtn.disabled = !episode.terminated || replayIndex < 0;
    replace(observed, routeStrip(env, full, { visited: index, active: index, maskFuture: true, label: "物理路线 S→A→B→C→D→终点。未观察的奖励显示 ?，学习器不能提前使用它。" }));
    replace(tdStrip, routeStrip(env, full, { values: td.V, reverse: true, visited: index, maskFuture: true, active: lastTD ? full.findIndex(t => t.state === lastTD.state) : -1, label: "数字是当前估计 V。← 表示信息传向经验起点；机器人仍按 → 行动。" }));
    replace(mcStrip, routeStrip(env, full, { values: mc.V, visited: index, maskFuture: true, active: lastMC ? 0 : -1, label: "回合完成后，可为每个经过的位置构造 MC 目标。" }));
    replace(stats, metric("已收集经验", `${index} / 5 步`), metric("TD 更新", td.learnSteps), metric("MC 更新", mc.learnSteps), metric("TD · V(S)", num(td.V[env.start])), metric("MC · V(S)", num(mc.V[env.start])));
    if (lastTD) {
      const t = lastTD, i = full.findIndex(x => x.state === t.state);
      tdCalc.textContent = `更新 ${names[i]}：目标 y = ${num(t.reward)} + 0.9 × ${num(t.bootstrap_value)} = ${num(t.target)}${t.terminated ? "（终止，后续项 = 0）" : ""}；δ = ${num(t.target)} − ${num(t.value_before)} = ${num(t.error)}；新 V = ${num(t.value_before)} + ${t.learning_rate} × ${num(t.error)} = ${num(t.value_after)}`;
    } else tdCalc.textContent = "选择一条已观察经验。目标 y = 当前奖励 + 0.9 × 下一位置的当前估计；未更新的位置仍是初始化的 0。";
    mcCalc.textContent = lastMC ? `S 的目标 G₀ = 0.9⁴ × 1 = ${num(lastMC[0].target)}；旧 V = ${num(lastMC[0].value_before)}，新 V = ${num(mc.V[env.start])}。这次对 5 个起点各更新一次。` : "需要回合真正结束，才知道每个位置余下的实际回报 Gₜ。";
    replace(experienceTable, h("h3", {}, "已观察的经验 · 选一条查看 TD"), episode.length ? h("table", {}, th(["选择经验", "当前奖励", "后继", "终止？"]), h("tbody", {}, full.slice(0, index).map((t, i) => h("tr", { class: i === selected ? "highlight" : "" }, h("td", {}, button(`${i + 1} · ${names[i]} → ${names[i + 1]}`, { onClick: () => { selected = i; lastTD = null; render(); } })), h("td", {}, t.reward), h("td", {}, names[i + 1]), h("td", {}, t.terminated ? "是" : "否"))))) : h("p", {}, "还没有经验，先走一步。"));
    replace(mcTable, lastMC ? h("table", {}, th(["起点", "实际 Gₜ", "旧 V → 新 V"]), h("tbody", {}, lastMC.map((t, i) => h("tr", {}, h("td", {}, names[i]), h("td", {}, num(t.target)), h("td", {}, `${num(t.value_before)} → ${num(t.value_after)}`))))) : null);
    status.textContent = episode.terminated ? `回合完成。MC 一次处理 5 个起点；TD 每次处理 1 条经验。逆序重放还剩 ${replayIndex + 1} 条，会复用已学过的经验。α = 1 时完整逆序重放一次，V(S) = 0.656；其他 α 会逐步接近。` : "先走 1～3 步：TD 可以学，MC 必须等待。起初 r = 0、V(s′) = 0，TD 目标也是 0；按钮可用不代表数值立即变大。";
  }
  root.append(lessonHeader("07"),
    step("先看图像：两种看未来的方式", prose("保持策略不变，只沿去 +1 的固定路线评价 Vπ。MC 等整段经历结束，使用实际剩余回报；TD 看真实的一步，再用下一状态的估计接上未知未来。两者看到同一组数据，区别是构造目标的方式。", "默认 α = 1，是为了让传播一目了然。调小 α，观察每次只朝目标移动一部分。采集经验和学习仍是分开的动作。")),
    step("公式：同一旧估计，两种学习目标",
      equation("G_t=\\sum_{k=0}^{T-t-1}\\gamma^k r_{t+k},\\qquad V(s_t)\\leftarrow V(s_t)+\\alpha[G_t-V(s_t)]", "Monte Carlo：目标是直到真正终止的实际折扣回报 Gₜ，不使用当前后继估计。"),
      equation("\\delta_t=r_t+\\gamma V(s_{t+1})-V(s_t),\\qquad V(s_t)\\leftarrow V(s_t)+\\alpha\\delta_t", "TD(0)：目标 yₜ = rₜ + γV(sₜ₊₁)。右侧读取更新前的估计；真正终止时，后续项为 0。"),
      prose("α 是学习率：旧值 + α×(目标−旧值)。α = 1 直接替换；α = 0.5 走一半。例如旧 V(D) = 0，最后一步奖励 +1，两个目标都是 1，新值为 1 或 0.5。更早的 S，MC 目标是 0.656；TD 目标取决于当前 V(A)，起初仍是 0。")),
    step("动手验证：先走几步，再试学习", h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, "走 3 步试 TD → 走完 5 步试 MC → 逆序重放 TD"), "对照经验行、目标和新数值。最后保留估计再走一轮，观察旧估计怎样帮助下一次 TD 更新。"), h("div", { class: "toolbar" }, actBtn, button("重新开始 · 估计清零", { kind: "ghost", onClick: reset }), button("保留估计，再走一回合", { onClick: newEpisode }), alphaControl), h("div", { class: "experiment-content" }, observed, stats, status, experienceTable,
      h("div", { class: "comparison" }, h("section", { class: "comparison-card" }, h("h3", {}, "TD · 一步 + 当前估计"), tdBtn, tdStrip, tdCalc, replayBtn), h("section", { class: "comparison-card" }, h("h3", {}, "MC · 完整实际回报"), mcBtn, mcStrip, mcCalc, mcTable))))),
    step("目标的误差从哪里来？", prose("MC 目标对固定策略的真实期望价值是一个实际回报样本，不依赖当前 V；随机奖励、转移、动作会让它波动。TD 同样有采样噪声，还依赖不准确的下一状态估计，这叫自举（bootstrapping）。真实 r 不保证整个 TD 目标准确。TD 目标通常比长段 MC 回报波动小，但不存在适用于所有任务的方差排序。", "逆序重放先改变 D，再给 C、B、A、S 使用。这改变计算顺序，没有获得新经验。正向一次重放时，早期位置仍读到 0，需要更多回合或更新轮数。", "真正终止后没有未来回报，后续项为 0。如果只是达到人为步数上限、任务仍可继续，那叫截断（truncation），不能自动把 V(s′) 清零。这里是真的进入终点，MC 不用猜余下未来。")),
    takeaway("MC 用完整实际回报，TD 用一步奖励加当前后继估计。先说目标的数据来源，再算误差和新值；学习时机、传播顺序与学习率都会影响一次更新。"),
    step("检查理解", predict({ question: "TD 的下一位置估计错了，这次目标也可能错吗？", choices: [{ label: "可能，目标包含当前 V(s′)" }, { label: "不可能，真实奖励保证整个目标正确" }], answer: 0, explain: "真实的是当前奖励，后半段仍是估计。后续经验与更新会继续修正它。" })),
    optional("展开：对应的真实实现", codeBlock(extractDef(tdSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "TD0.learn_step" }), codeBlock(extractDef(mcSource, "learn_episode"), { title: "MonteCarlo.learn_episode" })), lessonFooter("07"));
  reset(); return () => {};
}
