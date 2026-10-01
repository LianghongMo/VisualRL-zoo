import { LESSONS, lessonById } from "../lessons/curriculum.js";
import { h } from "./dom.js";
import { KNOWLEDGE } from "../lessons/knowledge.js";
export function courseNav(id = "") {
  const links = LESSONS.map((l) => h("a", {
    href: `#lesson-${l.id}`, class: l.id === id ? "active" : "", "aria-current": l.id === id ? "page" : undefined,
  }, h("span", { class: "nav-number" }, l.number), h("span", {}, l.short)));
  const nav = h("nav", { "aria-label": "课程目录" }, links, h("a", { href: "#", class: "nav-home" }, "← 回到课程首页"));
  return h("aside", { class: "course-sidebar" },
    h("details", { class: "course-menu", open: !window.matchMedia?.("(max-width: 760px)")?.matches }, h("summary", {}, `学习路线 · ${LESSONS.length} 章`), nav),
    h("p", { class: "sidebar-note" }, "图像 → 公式 → 实验 → 自检。用具体经验解释每一个符号。"));
}
export function lessonHeader(id) {
  const l = lessonById(id);
  return h("header", { class: "lesson-head" }, h("p", { class: "eyebrow" }, l.number ? `第 ${l.number} 章 / ${LESSONS.length} · ${l.group}` : l.group), h("h1", {}, l.title), h("p", { class: "lead" }, l.image), h("p", { class: "learning-goal" }, h("strong", {}, "这一章要掌握："), l.goal), h("div", { class: "essentials" }, h("h2", {}, "必须掌握的知识点"), h("ul", {}, KNOWLEDGE[id].map(([concept]) => h("li", {}, concept)))));
}
export const step = (label, ...body) => h("section", { class: "step" }, h("h2", {}, label), ...body);
export const wideStep = step;
export const prose = (...paragraphs) => h("div", { class: "prose" }, paragraphs.map((p) => p instanceof Node ? p : h("p", {}, p)));
export const optional = (label, ...body) => h("details", { class: "optional" }, h("summary", {}, label), h("div", { class: "optional-body" }, ...body));
export const takeaway = (text) => h("section", { class: "takeaway" }, h("h2", {}, "带走这一句话"), h("p", {}, text));
export function lessonFooter(id) {
  const l = lessonById(id), i = LESSONS.findIndex((x) => x.id === id);
  const prev = LESSONS[i - 1], next = id === "06" ? lessonById("02") : LESSONS[i + 1];
  const audit = h("section", { class: "mastery-audit" }, h("h2", {}, "逐项检查：你能解释这些问题了吗？"), h("p", {}, "勾选表示你已经能独立回答；有疑问就回到对应的图像、公式和实验。"), KNOWLEDGE[id].map(([concept, section, question]) => h("div", { class: "mastery-row", dataset: { evidence: section } }, h("label", {}, h("input", { type: "checkbox" }), h("span", {}, h("strong", {}, concept), h("span", {}, question))), h("button", { type: "button", class: "evidence-link", onclick: (event) => {
    const main = event.target.closest("main");
    const target = [...main.querySelectorAll(".step > h2")].find((el) => el.textContent === section);
    target?.parentElement.scrollIntoView({ behavior: "smooth", block: "start" });
  } }, `回看：${section}`))));
  return h("footer", { class: "lesson-foot" }, audit, id === "09" ? h("p", { class: "source-note" }, "公式与阅读主线：", h("a", { href: "https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf", target: "_blank", rel: "noopener" }, "Benjamin Eysenbach · Probabilistic Reinforcement Learning（第 2–4 章、附录 B）"), "。本章实验为明确假设下的表格计算与规划示范，具体边界见正文。") : h("p", { class: "source-note" }, "公式参考：", h("a", { href: "https://www.incompleteideas.net/book/the-book-2nd.html", target: "_blank", rel: "noopener" }, "Sutton & Barto · Reinforcement Learning（第 3–6 章）"), "。本课程用小型确定性环境展示计算；各章会说明实验假设。"), h("p", {}, l.next), h("nav", { "aria-label": "继续学习" }, prev ? h("a", { href: `#lesson-${prev.id}` }, `← ${prev.short}`) : h("a", { href: "#" }, "← 课程首页"), next ? h("a", { class: "next-lesson", href: `#lesson-${next.id}` }, `${next.short} →`) : h("a", { class: "next-lesson", href: "#" }, "完成主线 · 回到首页 →")));
}
export function predict({ question, choices, answer, explain }) {
  const verdict = h("p", { class: "quiz-verdict", hidden: true, role: "status" });
  const buttons = choices.map((c, i) => h("button", { type: "button", class: "choice", "aria-pressed": "false", onclick: () => {
    buttons.forEach((b, j) => { b.setAttribute("aria-pressed", String(i === j)); b.classList.toggle("correct", j === answer); });
    verdict.replaceChildren(h("strong", {}, i === answer ? "对。" : "再想一想。"), typeof explain === "function" ? explain() : explain); verdict.hidden = false;
  } }, c.label, c.detail ? h("small", {}, c.detail) : null));
  return h("div", { class: "quiz" }, h("p", {}, question), h("div", { class: "choices" }, buttons), verdict);
}
