import { GridWorld } from "../rl/envs/gridworld.js";
import { h, button, replace, slider, segmented } from "../ui/dom.js";
import { lessonHeader, step, prose, optional, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, route, statesOf, NEAR_ROUTE, FAR_ROUTE, discounted, num, signed, metric, routeStrip } from "../ui/world-view.js";
import { equation } from "../ui/math.js";

export function mount(root) {
  const env = GridWorld.chargingRoom();
  const routes = { near: route(env, NEAR_ROUTE), far: route(env, FAR_ROUTE) };
  let gamma = 0.9, choice = "near";
  const map = new WorldView(env, { caption: "路线固定时，拖动 γ 只改变奖励的权重，不改变地图、步数或原始奖励。" });
  const strip = h("div"), stats = h("div", { class: "metrics" });
  const arithmetic = h("div", { class: "arithmetic" });
  const observation = h("p", { class: "observation", role: "status" });
  const weights = h("div", { class: "return-bars" });
  const table = h("div", { class: "table-wrap" });
  function render() {
    const tr = routes[choice], rewardAt = tr.length - 1;
    const V = new Array(env.nStates).fill(0);
    tr.forEach((t, i) => { V[t.state] = discounted(tr.slice(i), gamma); });
    map.render({ robot: env.start, paths: [{ states: statesOf(tr), color: choice === "near" ? "var(--act)" : "var(--learn)" }] });
    replace(strip, routeStrip(env, tr, { values: V, label: "箭头上是这一走的奖励；格子下是从那里继续按这条路线走的价值 V。" }));
    const a = discounted(routes.near, gamma), b = discounted(routes.far, gamma);
    replace(stats, metric("近处：5 步，奖励 +1", num(a)), metric("远处：8 步，奖励 +10", num(b)), metric("此时回报更大", a > b + 1e-10 ? "近处 +1" : b > a + 1e-10 ? "远处 +10" : "两条相同"));
    arithmetic.textContent = `G = ${gamma.toFixed(2)}^${rewardAt} × ${signed(tr.at(-1).reward)} = ${num(discounted(tr, gamma))}`;
    observation.textContent = `第一步的奖励编号是 t = 0；第 ${tr.length} 步进入充电站，所以它被折扣 ${rewardAt} 次。${gamma < 0.465 ? "折扣较强，晚到的 +10 被缩小很多，近处的 +1 更有吸引力。" : "此时远处奖励保留得足够多，+10 的路线回报更大。"}`;
    replace(weights, h("h3", {}, "每一步的奖励还保留多少权重？"), tr.map((_, t) => h("div", { class: "return-row" }, h("span", {}, `t=${t}`), h("div", { class: "return-track" }, h("div", { class: "return-bar", style: { width: `${gamma ** t * 100}%` } })), h("b", {}, num(gamma ** t)))));
    replace(table, h("table", {}, h("thead", {}, h("tr", {}, ["奖励编号 t", "原始奖励 r", "权重 γᵗ", "计入回报"].map((x) => h("th", {}, x)))), h("tbody", {}, tr.map((t, i) => h("tr", {}, h("td", {}, i), h("td", {}, signed(t.reward)), h("td", {}, num(gamma ** i)), h("td", {}, num(gamma ** i * t.reward)))))));
  }
  const gammaControl = slider({ id: "return-gamma", label: "未来奖励的折扣 γ", min: 0.1, max: 0.99, step: 0.01, value: gamma, format: (v) => v.toFixed(2), onInput: (v) => { gamma = v; render(); } });
  const picker = segmented([{ value: "near", label: "看近处路线 · 5 步" }, { value: "far", label: "看远处路线 · 8 步" }], { value: choice, label: "查看哪条固定路线", onChange: (v) => { choice = v; render(); } });
  root.append(lessonHeader("04"), step("先看图像：奖励沿路线逐项累计", prose("奖励 r 是一次移动收到的反馈；回报 G 是从现在开始，沿整条路线累计的奖励。γ 把未来奖励缩小：眼前奖励乘 1，晚一步乘 γ，晚两步乘 γ²。", "策略 π 是在每个位置选择动作的规则。这个确定性例子中，固定策略对应一条固定路线；从某个位置按它继续走的回报，就是这个位置在该策略下的价值 V。")),
    step("动手验证：只改变折扣，不改变路线", h("div", { class: "experiment" },
      h("div", { class: "experiment-instruction" }, h("strong", {}, "先比较 γ = 0.90，再拖到 0.30"), "看两条路线的回报谁更大。然后切换路线，看那一次终点奖励被打了几次折扣。"),
      h("div", { class: "toolbar" }, picker, gammaControl),
      h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, arithmetic, weights, observation)), h("div", { class: "experiment-content" }, strip))),
    step("把回报存下来，就是价值", prose("路线图中的 V 是「从这个位置继续走，还能拿到多少折扣奖励」。离终点越近，奖励被折扣得越少。终点的 V = 0：进入终点时奖励已经收到了，结束后没有后续奖励。", "Q(s,a) 则多指定一个动作：先从 s 做 a，再按原策略继续走。V 评价一个位置，Q 评价在这个位置先做某个动作。这两个量总要说明后续按什么策略行动。")),
    takeaway("奖励是一步的反馈，回报是一段未来的累计；价值是「按某个策略从这里出发」的预期回报。这个固定路线的例子没有随机性，所以 V 就等于路线的 G。"),
    step("检查理解", predict({ question: "把 γ 从 0.90 调成 0.30，充电站的原始奖励 +10 改变了吗？", choices: [{ label: "没有，变的是它计入当前回报的权重" }, { label: "改变了，环境现在给的奖励更小" }], answer: 0, explain: "环境仍给 +10。γ 改变的是我们怎样衡量晚到的奖励，所以路线没有变，回报变了。" })),
    optional("展开：逐项计算与公式", table, equation("G_t = r_t + \\gamma G_{t+1}", "从后往前算：这一段的回报 = 当前奖励 + 余下路线的折扣回报。"), prose("如果动作或环境有随机性，一条路线的 G 会变动；V 是在指定策略下，对这些可能回报取期望。")), lessonFooter("04"));
  render(); return () => {};
}
