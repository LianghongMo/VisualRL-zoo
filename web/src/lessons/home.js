import { h } from "../ui/dom.js";
import { PARTS, LESSONS } from "./curriculum.js";
import { WorldView, route, statesOf, NEAR_ROUTE, FAR_ROUTE, mapLegend } from "../ui/world-view.js";
import { GridWorld } from "../rl/envs/gridworld.js";

export function mount(root) {
  const env = GridWorld.chargingRoom();
  const map = new WorldView(env, { caption: "同一个出发点：近处的 +1，远处的 +10。机器人该选哪条路？" });
  map.render({ robot: env.start, paths: [
    { states: statesOf(route(env, NEAR_ROUTE)), color: "var(--act)" },
    { states: statesOf(route(env, FAR_ROUTE)), color: "var(--learn)", dashed: true },
  ] });
  const groups = PARTS.map((part) => {
    const rows = part.lessons.map((l) => h("li", {},
      h("a", { class: "lesson-row", href: `#lesson-${l.id}` },
        h("span", { class: "n" }, String(l.number).padStart(2, "0")),
        h("span", { class: "t" }, l.title, h("span", { class: "d" }, l.image)),
        h("span", { class: "row-arrow", "aria-hidden": "true" }, "→"))));
    return h("section", { class: "course-group" }, h("h3", {}, part.title), h("ol", { class: "lesson-list" }, rows));
  });
  root.append(
    h("section", { class: "home-intro" },
      h("div", { class: "home-copy" },
        h("p", { class: "eyebrow" }, "Visual RL · 看见强化学习"),
        h("h1", {}, "一台机器人，", h("br"), "怎样学会选路？"),
        h("p", { class: "lead" }, "从一张地图开始，看奖励怎样变成经验，经验怎样改变估计，估计又怎样改变下一次行动。"),
        h("a", { class: "btn primary", href: "#lesson-01" }, "从第 1 章开始 →"),
        h("p", { class: "home-note" }, `${LESSONS.length} 章 · 中文讲解 · 物理图像、正文公式与交互实验`)),
      h("div", { class: "home-map" }, map.el, mapLegend("实线：近处充电站", "虚线：远处充电站"))),
    h("section", { class: "course-overview" },
      h("h2", {}, "每一章，回答一个问题"),
      h("p", { class: "muted" }, "下一章只在上一章的图像上，多加一件事。"), groups),
    h("section", { class: "home-method" }, h("h2", {}, "怎么读这些页面"),
      h("div", { class: "method-grid" },
        h("div", {}, h("b", {}, "1 · 先看图像"), h("p", {}, "先说清发生了什么，再引入名字。")),
        h("div", {}, h("b", {}, "2 · 动手验证"), h("p", {}, "每次只改变一个条件，观察哪里变了。")),
        h("div", {}, h("b", {}, "3 · 说出原因"), h("p", {}, "能解释一个具体例子，才算掌握。用正文公式核对实验，源码可选读。")))),
    h("p", { class: "home-end" }, "这些图像是教学示意：地图中的一步是一次离散环境转移，图上的价值来自实际算法计算。"));
  return () => {};
}
