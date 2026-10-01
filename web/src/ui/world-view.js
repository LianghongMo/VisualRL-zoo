// Map and trajectory geometry corresponds to actual states and transitions.
import { h, s, replace } from "./dom.js";
export const ARROWS = ["↑", "→", "↓", "←"];
export const MOVES = ["上", "右", "下", "左"];
export const NEAR_ROUTE = [0, 1, 1, 1, 2];
export const FAR_ROUTE = [0, 1, 1, 1, 1, 1, 0, 0];
export const num = (v, digits = 3) => String(Number(Number(v).toFixed(digits)));
export const signed = (v) => v > 0 ? `+${num(v)}` : num(v);
export function stateName(env, state) {
  if (state === env.start) return "出发点";
  if (env.goals.includes(state)) return `充电站 ${signed(env.goalRewards.get(state) ?? env.goalReward)}`;
  const [r, c] = env.toCell(state); return `位置 (${r},${c})`;
}
export function route(env, actions, start = env.start) {
  let state = start; const transitions = [];
  for (const action of actions) {
    const [next_state, reward, terminated] = env.move(state, action);
    transitions.push({ state, action, next_state, reward, terminated, truncated: false, next_action: null });
    state = next_state; if (terminated) break;
  }
  return transitions;
}
export const statesOf = (tr) => tr.length ? [tr[0].state, ...tr.map((t) => t.next_state)] : [];
export const discounted = (tr, gamma) => tr.reduce((sum, t, i) => sum + gamma ** i * t.reward, 0);
export const metric = (label, value, detail = "") => h("div", { class: "metric" }, h("span", {}, label), h("strong", {}, value), detail ? h("small", {}, detail) : null);
export const mapLegend = (...extra) => h("div", { class: "map-legend" }, h("span", {}, "● 机器人"), h("span", {}, "S 出发点"), h("span", {}, "奖励在进入充电站时收到"), ...extra.map((x) => h("span", {}, x)));
export class WorldView {
  constructor(env, { caption = "", onSelect = null } = {}) {
    this.env = env; this.onSelect = onSelect;
    this.svg = s("svg", { viewBox: `0 0 ${env.width * 64} ${env.height * 64}`, role: onSelect ? "group" : "img", "aria-label": caption || "机器人环境地图", class: "world-svg" });
    this.el = h("figure", { class: "world-view" }, this.svg, caption ? h("figcaption", {}, caption) : null);
  }
  center(state) { const [r, c] = this.env.toCell(state); return [c * 64 + 32, r * 64 + 32]; }
  render({ robot = null, values = null, policy = null, paths = [], selected = null, changed = new Set(), known = null, edges = [], labels = {} } = {}) {
    const env = this.env, cells = [], marks = [];
    for (let state = 0; state < env.nStates; state++) {
      const [r, c] = env.toCell(state), wall = env.walls.includes(state), hazard = env.cliffs.includes(state), goal = env.goals.includes(state);
      const unknown = known && !known.has(state) && !wall && !hazard && !goal, v = values?.[state] ?? 0;
      const fill = wall ? "var(--wall)" : hazard ? "var(--hazard-wash)" : goal ? "var(--reward-wash)" : unknown ? "var(--sheet)" : values && v !== 0 ? `color-mix(in srgb, ${v > 0 ? "var(--learn)" : "var(--hazard)"} ${Math.min(65, 12 + Math.abs(v) * 5)}%, white)` : "white";
      const cell = s("rect", { x: c * 64 + 2, y: r * 64 + 2, width: 60, height: 60, rx: 7, fill, class: `world-cell${selected === state ? " selected" : ""}${changed.has(state) ? " changed" : ""}` });
      cells.push(cell); const [x, y] = this.center(state); if (wall) continue;
      if (hazard) marks.push(s("text", { x, y: y + 6, class: "map-hazard", "text-anchor": "middle" }, "×"));
      else if (goal) { marks.push(s("text", { x, y: y + 3, class: "map-goal", "text-anchor": "middle" }, signed(env.goalRewards.get(state) ?? env.goalReward))); marks.push(s("text", { x, y: y + 20, class: "map-mini", "text-anchor": "middle" }, "终点")); }
      else {
        if (state === env.start) marks.push(s("text", { x: x - 23, y: y - 16, class: "map-mini" }, "S"));
        if (unknown) marks.push(s("text", { x, y: y + 6, class: "map-unknown", "text-anchor": "middle" }, "?"));
        else if (values) marks.push(s("text", { x, y: y + 8, class: "map-value", "text-anchor": "middle" }, num(v, 2)));
        if (labels[state]) marks.push(s("text", { x, y: y - 14, class: "map-mini", "text-anchor": "middle" }, labels[state]));
        if (policy && !unknown) { const p = policy[state]; const actions = Array.isArray(p) ? p.flatMap((prob, a) => prob > 0 ? [a] : []) : p === null || p === undefined ? [] : [p]; marks.push(s("text", { x, y: y - 12, class: "map-arrow", "text-anchor": "middle" }, actions.map((a) => ARROWS[a]).join(""))); }
      }
      if (this.onSelect && !hazard && !goal) {
        cell.setAttribute("role", "button"); cell.setAttribute("tabindex", "0"); cell.setAttribute("aria-label", `${stateName(env, state)}${values ? `，价值 ${num(v)}` : ""}`);
        cell.addEventListener("click", () => this.onSelect(state)); cell.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.onSelect(state); } });
      }
    }
    const lines = edges.map((tr) => { const [x1, y1] = this.center(tr.state), [x2, y2] = this.center(tr.next_state); return s("line", { x1, y1, x2, y2, stroke: tr.color ?? "var(--act)", "stroke-width": 4, opacity: 0.65 }); });
    const traces = paths.filter((p) => p.states.length > 1).map((p) => s("polyline", { points: p.states.map((state) => this.center(state).join(",")).join(" "), fill: "none", stroke: p.color ?? "var(--act)", "stroke-width": p.width ?? 5, "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": p.dashed ? "7 6" : undefined, opacity: 0.75 }));
    const agent = robot === null ? [] : (() => { const [x, y] = this.center(robot); return [s("circle", { cx: x, cy: values ? y + 21 : y, r: values ? 6 : 10, class: "map-robot" })]; })();
    replace(this.svg, cells, lines, traces, marks, agent);
  }
}
// Rewards belong to arrows, including the arrow into the terminal state.
export function routeStrip(env, transitions, { values = null, active = -1, visited = transitions.length, reverse = false, label = "" } = {}) {
  const states = statesOf(transitions);
  return h("figure", { class: "route-figure" }, label ? h("figcaption", {}, label) : null,
    h("div", { class: `route-strip${reverse ? " reverse-info" : ""}`, "aria-label": label || "路线与奖励" }, states.flatMap((state, i) => [
      i > 0 ? h("div", { class: `route-edge${active === i - 1 ? " active" : ""}${i <= visited ? " visited" : ""}` }, h("b", {}, signed(transitions[i - 1].reward)), h("span", {}, reverse ? "← 信息" : "→")) : null,
      h("div", { class: `route-node${active === i ? " active" : ""}${i <= visited ? " visited" : ""}` }, h("span", {}, i === 0 ? "S" : i === states.length - 1 ? "终点" : String.fromCharCode(64 + i)), h("small", {}, values ? num(values[state]) : i <= visited ? "已到达" : "未到达")),
    ])));
}
