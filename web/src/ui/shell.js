import { LESSONS, lessonById } from "../lessons/curriculum.js";
import { h } from "./dom.js";
export function courseNav(id = "") {
  const links = LESSONS.map((l) => h("a", {
    href: `#lesson-${l.id}`, class: l.id === id ? "active" : "", "aria-current": l.id === id ? "page" : undefined,
  }, h("span", { class: "nav-number" }, l.number), h("span", {}, l.short)));
  const nav = h("nav", { "aria-label": "课程目录" }, links, h("a", { href: "#", class: "nav-home" }, "← 回到课程首页"));
  return h("aside", { class: "course-sidebar" },
    h("details", { class: "course-menu", open: true }, h("summary", {}, "学习路线 · 7 章"), nav),
    h("p", { class: "sidebar-note" }, "先看图像，再动手验证。每章只引入一件新的事。"));
}
export function lessonHeader(id) {
  const l = lessonById(id);
  return h("header", { class: "lesson-head" }, h("p", { class: "eyebrow" }, l.number ? `第 ${l.number} 章 / 7 · ${l.group}` : l.group), h("h1", {}, l.title), h("p", { class: "lead" }, l.image), h("p", { class: "learning-goal" }, h("strong", {}, "这一章要掌握："), l.goal));
}
export const step = (label, ...body) => h("section", { class: "step" }, h("h2", {}, label), ...body);
export const wideStep = step;
export const prose = (...paragraphs) => h("div", { class: "prose" }, paragraphs.map((p) => p instanceof Node ? p : h("p", {}, p)));
export const optional = (label, ...body) => h("details", { class: "optional" }, h("summary", {}, label), h("div", { class: "optional-body" }, ...body));
export const takeaway = (text) => h("section", { class: "takeaway" }, h("h2", {}, "带走这一句话"), h("p", {}, text));
export function lessonFooter(id) {
  const l = lessonById(id), i = LESSONS.findIndex((x) => x.id === id);
  const prev = LESSONS[i - 1], next = id === "06" ? lessonById("02") : LESSONS[i + 1];
  return h("footer", { class: "lesson-foot" }, h("p", {}, l.next), h("nav", { "aria-label": "继续学习" }, prev ? h("a", { href: `#lesson-${prev.id}` }, `← ${prev.short}`) : h("a", { href: "#" }, "← 课程首页"), next ? h("a", { class: "next-lesson", href: `#lesson-${next.id}` }, `${next.short} →`) : h("a", { class: "next-lesson", href: "#" }, "完成主线 · 回到首页 →")));
}
export function predict({ question, choices, answer, explain }) {
  const verdict = h("p", { class: "quiz-verdict", hidden: true, role: "status" });
  const buttons = choices.map((c, i) => h("button", { type: "button", class: "choice", "aria-pressed": "false", onclick: () => {
    buttons.forEach((b, j) => { b.setAttribute("aria-pressed", String(i === j)); b.classList.toggle("correct", j === answer); });
    verdict.replaceChildren(h("strong", {}, i === answer ? "对。" : "再想一想。"), typeof explain === "function" ? explain() : explain); verdict.hidden = false;
  } }, c.label, c.detail ? h("small", {}, c.detail) : null));
  return h("div", { class: "quiz" }, h("p", {}, question), h("div", { class: "choices" }, buttons), verdict);
}
