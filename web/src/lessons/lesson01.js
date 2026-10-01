import { foundations } from "./foundations.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { h, button, replace, shortcuts } from "../ui/dom.js";
import { lessonHeader, step, prose, takeaway, predict, lessonFooter } from "../ui/shell.js";
import { WorldView, stateName, signed, ARROWS, MOVES, NEAR_ROUTE, statesOf, metric, mapLegend } from "../ui/world-view.js";
export function world() {
  const env = GridWorld.chargingRoom(), far = env.toState(1, 6), near = env.toState(4, 4);
  return { env, far, near, labels: { [far]: "+10", [near]: "+1" }, signs: { [far]: "远处 +10", [near]: "近处 +1" } };
}
export function mount(root) {
  const { env } = world(); let [state] = env.reset(); let transitions = [], ended = false;
  const map = new WorldView(env, { caption: "空白格可以走；深色格是墙；× 是悬崖，进入后扣 10 并回到出发点。" });
  const display = h("div", { class: "transition-display", "aria-live": "polite" }), status = h("p", { class: "observation", role: "status" }), stats = h("div", { class: "metrics" });
  function move(action) { if (ended) return; const [next_state, reward, terminated] = env.step(action); transitions.push({ state, action, next_state, reward, terminated }); state = next_state; ended = terminated; render(); }
  const controls = ARROWS.map((arrow, a) => button(`${arrow} ${MOVES[a]}`, { kind: "env", onClick: () => move(a) }));
  function reset() { [state] = env.reset(); transitions = []; ended = false; render(); }
  function render() {
    map.render({ robot: state, paths: [{ states: statesOf(transitions) }] }); controls.forEach((b) => { b.disabled = ended; });
    const last = transitions.at(-1);
    replace(display, metric("原来在哪 · 状态 s", last ? stateName(env, last.state) : "出发点"), h("span", { class: "flow-arrow" }, "→"), metric("做了什么 · 动作 a", last ? `${ARROWS[last.action]} ${MOVES[last.action]}` : "等待移动"), h("span", { class: "flow-arrow" }, "→"), metric("环境给了什么", last ? `奖励 ${signed(last.reward)}` : "等待环境返回"), metric("现在在哪 · 新状态 s′", stateName(env, state)));
    replace(stats, metric("本次走了", `${transitions.length} 步`), metric("本次奖励合计", signed(transitions.reduce((sum, t) => sum + t.reward, 0))));
    status.textContent = ended ? "到达充电站，这一轮结束。从出发点到终点的经历叫一个回合（episode）。点「重新出发」开始新的一轮。" : last?.reward === -10 ? "刚才向悬崖移动：奖励是 −10，新状态是出发点。这一轮没有结束，还可以继续走。" : last && last.state === last.next_state ? "撞到墙了：动作仍然发生，但位置没有改变，奖励为 0。" : "先点「↑ 上」：观察蓝点移动，再把这一步的经验读出来。沿提示路线走到 +1，观察这一轮何时结束。";
  }
  root.append(lessonHeader("01"), ...foundations("01"), step("先看图像：行动在前，反馈在后", prose("地图是环境，蓝点是机器人。机器人选择上、右、下、左；下一位置和奖励由环境规则决定。普通移动奖励为 0，进入充电站才收到 +1 或 +10。")), step("动手验证：读懂机器人走的一步", h("div", { class: "experiment" }, h("div", { class: "experiment-instruction" }, h("strong", {}, "试一条具体路线"), "↑ → → → ↓：5 步到达近处的 +1。也可以试着撞墙或走向悬崖。"), h("div", { class: "toolbar" }, controls, button("重新出发", { kind: "ghost", onClick: reset }), button("演示到 +1 的路线", { kind: "ghost", onClick: () => { reset(); NEAR_ROUTE.forEach(move); } })), h("div", { class: "experiment-grid" }, h("div", {}, map.el, mapLegend()), h("div", { class: "experiment-reading" }, h("h3", {}, "最近一步的经验"), display, stats, status)))), takeaway("基本材料是一条经验：在状态 s 做动作 a，收到奖励 r，并到达新状态 s′。经验描述发生了什么；到这一章为止，还没有任何学习。"), step("检查理解", predict({ question: "机器人向墙移动，但没有换位置。这算一次经验吗？", choices: [{ label: "算：动作、奖励和新状态都有结果" }, { label: "不算：必须移动到另一个位置" }], answer: 0, explain: "状态可以不变。环境仍然返回奖励 0 和原来的位置，所以仍有一条 (s, a, r, s′) 经验。" })), lessonFooter("01"));
  render(); return shortcuts({ arrowup: () => move(0), arrowright: () => move(1), arrowdown: () => move(2), arrowleft: () => move(3) });
}
