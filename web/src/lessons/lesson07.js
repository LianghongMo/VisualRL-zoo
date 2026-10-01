import tdSource from "../../../visualrl/algorithms/tabular/td.py";
import mcSource from "../../../visualrl/algorithms/tabular/monte_carlo.py";
import { GridWorld } from "../rl/envs/gridworld.js";
import { Trajectory } from "../rl/core.js";
import { TD0 } from "../rl/tabular/td0.js";
import { MonteCarlo } from "../rl/tabular/monte-carlo.js";
import { h, button, replace } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { route, NEAR_ROUTE, num, routeStrip, metric } from "../ui/world-view.js";
import { equation } from "../ui/math.js";
import { codeBlock } from "../ui/code.js";
import { extractDef } from "../ui/source.js";

export function mount(root) {
  const env = GridWorld.chargingRoom(), full = route(env, NEAR_ROUTE);
  let episode, td, mc, index, replayIndex, lastTD, lastMC;
  const tdStrip = h("div"), mcStrip = h("div"), stats = h("div", { class: "metrics" });
  const tdCalc = h("div", { class: "arithmetic" }), mcCalc = h("div", { class: "arithmetic" });
  const tdNote = h("p", { class: "observation" }), mcNote = h("p", { class: "observation" });
  const observed = h("div"), status = h("p", { class: "observation", role: "status" });
  function reset() {
    episode = new Trajectory(); td = new TD0({ nStates: env.nStates, alpha: 1, gamma: 0.9 });
    mc = new MonteCarlo({ nStates: env.nStates, alpha: 1, gamma: 0.9 });
    index = 0; replayIndex = -1; lastTD = null; lastMC = null; env.reset(); render();
  }
  function act() {
    if (episode.done) return;
    const t = full[index++]; const [next_state, reward, terminated] = env.step(t.action);
    episode.append({ ...t, next_state, reward, terminated }); replayIndex = index - 1; render();
  }
  const actBtn = button("沿固定路线走一步", { kind: "env", onClick: act });
  const tdBtn = button("TD：学最近的一步", { kind: "learn", onClick: () => { if (!episode.length) return; lastTD = td.learnStep(episode.at(index - 1)); render(); } });
  const mcBtn = button("MC：学完整回合", { kind: "learn", onClick: () => { if (!episode.terminated) return; lastMC = mc.learnEpisode(episode); render(); } });
  const replayBtn = button("TD：从终点向前重放一条", { onClick: () => { if (!episode.terminated || replayIndex < 0) return; lastTD = td.learnStep(episode.at(replayIndex--)); render(); } });
  function render() {
    actBtn.disabled = episode.done; tdBtn.disabled = !episode.length; mcBtn.disabled = !episode.terminated; replayBtn.disabled = !episode.terminated || replayIndex < 0;
    replace(observed, routeStrip(env, full, { visited: index, active: index, label: "这条示范路线只去近处的 +1。浅色位置尚未到达；图中预览了完整路线，但学习器只能使用已经走过的部分。" }));
    replace(tdStrip, routeStrip(env, full, { values: td.V, reverse: true, visited: index, active: lastTD ? full.findIndex((t) => t.state === lastTD.state) : -1 }));
    replace(mcStrip, routeStrip(env, full, { values: mc.V, visited: index, active: lastMC ? 0 : -1 }));
    replace(stats, metric("已收集经验", `${index} / 5 步`), metric("TD 更新", td.learnSteps), metric("MC 更新", mc.learnSteps));
    tdCalc.textContent = lastTD ? `目标 = ${num(lastTD.reward)} + 0.9 × ${num(lastTD.bootstrap_value)} = ${num(lastTD.target)}；V：${num(lastTD.value_before)} → ${num(lastTD.value_after)}` : "目标 = 当前奖励 + 0.9 × 下一位置的当前估计";
    mcCalc.textContent = lastMC ? `从 S 看完整路线：G = 0.9⁴ × 1 = ${num(lastMC[0].target)}；V(S) → ${num(mc.V[env.start])}` : "目标 = 从这个位置到终点，实际收到的全部折扣奖励";
    tdNote.textContent = "TD 只需要一条经验。开始时奖励为 0，下一位置估计也为 0，更新结果可能不变；能学习，并不意味着立刻学到正价值。";
    mcNote.textContent = episode.terminated ? "现在所有剩余奖励已知，MC 可以为路线中的每个位置计算完整回报。" : "回合还没结束，后面实际会收到什么尚未观察到；MC 要等待。";
    status.textContent = episode.terminated ? "现在分别试 MC 和 TD。MC 一次处理完整回合；TD 重放每条经验时，只更新该经验的起点。按逆序重放 5 条，奖励信息逐格传回 S。" : "先走 1～3 步，观察两个学习按钮：TD 已可更新，MC 仍要等待。然后继续走到终点。";
  }
  const cards = h("div", { class: "comparison" },
    h("section", { class: "comparison-card" }, h("h3", {}, "TD · 一步 + 当前估计"), tdBtn, tdStrip, tdCalc, tdNote),
    h("section", { class: "comparison-card" }, h("h3", {}, "Monte Carlo · 完整的实际回报"), mcBtn, mcStrip, mcCalc, mcNote));
  root.append(lessonHeader("07"), step("先看图像：两种看未来的方式", prose("使用同一条固定路线、同一组经验，区别只在学习目标。MC 等到终点，把实际余下奖励加起来；TD 走一步，就用奖励加下一位置的当前估计。", "为了让传播看得清楚，这个确定性实验取 α = 1：每次把估计直接改成目标。这里没有学习如何选动作，只是在评价固定路线。")),
    step("动手验证：先走几步，再试学习", h("div", { class: "experiment" },
      h("div", { class: "experiment-instruction" }, h("strong", {}, "先观察什么时候能学，再观察信息怎样传"), "走 3 步时试 TD；走完 5 步时试 MC，再从终点向前重放 TD。比较两种方法怎样把 +1 传到 S。"),
      h("div", { class: "toolbar" }, actBtn, button("重新开始", { kind: "ghost", onClick: reset })),
      h("div", { class: "experiment-content" }, observed, stats, status, cards, replayBtn))),
    takeaway("MC 用完整的实际回报当目标；TD 用一步奖励加下一位置的估计当目标。TD 更早可用，但它接上的未来是估计，需要后续经验和更新继续修正。"),
    step("检查理解", predict({ question: "TD 的下一位置估计目前是错的，这次目标可能也是错的吗？", choices: [{ label: "可能：TD 的目标包含当前估计" }, { label: "不可能：收到真实奖励后，整个目标就一定正确" }], answer: 0, explain: "真实的是当前奖励。后半段的 V(s′) 仍是估计，这叫自举（bootstrapping）。后续观察和更新会继续修正它。" })),
    optional("展开：通常 α 不直接取 1", equation("V(s) \\leftarrow V(s) + \\alpha[\\text{target} - V(s)]", "α 控制向目标靠近多少；有噪声时，单次回报通常不应直接当成准确价值。"),
      codeBlock(extractDef(tdSource, "learn_step", { cutAt: "return LearningTrace(", replacement: "return LearningTrace(...)" }), { title: "TD0.learn_step" }),
      codeBlock(extractDef(mcSource, "learn_episode"), { title: "MonteCarlo.learn_episode" })), lessonFooter("07"));
  reset(); return () => {};
}
