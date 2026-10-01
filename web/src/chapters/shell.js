import { h } from "../ui/dom.js";
import { CHAPTERS, PARTS, chapterById, BOOK_TOC, THESIS } from "../lessons/curriculum.js";
import { CHAPTER_KNOWLEDGE } from "./knowledge.js";
export { prose, step, predict, optional, takeaway } from "../ui/shell.js";
export { equation } from "../ui/math.js";
export function chapterNav(id) {
  return h("aside", { class: "course-sidebar" },
    h("details", { class: "course-menu", open: !window.matchMedia?.("(max-width: 760px)")?.matches },
      h("summary", {}, "教材主线 · 10章"), h("nav", { "aria-label": "教材章节" },
        PARTS.map(part => h("div", { class: "chapter-nav-group" }, h("p", {}, part.title),
          part.chapters.map(c => h("a", { href: c.href, class: c.id === id ? "active" : "", "aria-current": c.id === id ? "page" : undefined },
            h("span", { class: "nav-number" }, c.number), h("span", {}, c.short))))),
        h("a", { href: "#", class: "nav-home" }, "← 课程架构与教材对应"))),
    h("p", { class: "sidebar-note" }, "问题定义 → 表格求解 → 近似与策略优化 → 研究专题。"));
}
export function chapterHeader(id) {
  const c = chapterById(id);
  return h("header", { class: "lesson-head chapter-head" },
    h("p", { class: "eyebrow" }, "第 " + c.number + " 章 / 10 · " + c.group),
    h("h1", {}, c.title), h("p", { class: "lead" }, c.image),
    h("p", { class: "chapter-reference" }, h("a", { href: id === "10" ? THESIS : BOOK_TOC, target: "_blank", rel: "noopener" }, c.book)),
    h("p", { class: "learning-goal" }, h("strong", {}, "核心问题："), c.question, h("br"), h("strong", {}, "学习目标："), c.goal, h("br"), h("strong", {}, "前置知识："), c.prerequisite),
    h("nav", { class: "chapter-toc", "aria-label": "本章小节" },
      c.topics.map(t => h("a", { href: c.href + "/" + t.id }, h("b", {}, t.number), " " + t.title))),
    h("details", { class: "chapter-essentials" }, h("summary", {}, "本章必须掌握的 " + CHAPTER_KNOWLEDGE[id].length + " 项知识"),
      h("ul", {}, CHAPTER_KNOWLEDGE[id].map(([concept]) => h("li", {}, concept)))));
}
export function topic(chapterId, topicId, ...body) {
  const t = chapterById(chapterId).topics.find(t => t.id === topicId);
  if (!t) throw Error("Unknown topic: " + chapterId + "/" + topicId);
  return h("section", { class: "chapter-topic", dataset: { topic: topicId } },
    h("div", { class: "chapter-topic-heading" }, h("p", { class: "eyebrow" }, "小节 " + t.number), h("h2", {}, t.title), h("p", {}, t.description)), ...body);
}
export function chapterFooter(id, summary) {
  const c = chapterById(id), previous = CHAPTERS[c.number - 2], next = CHAPTERS[c.number];
  return h("footer", { class: "lesson-foot chapter-foot" },
    h("section", { class: "takeaway" }, h("h2", {}, "本章的完整逻辑"), h("p", {}, summary)),
    h("section", { class: "mastery-audit" }, h("h2", {}, "逐项验收：本章知识有没有讲清楚？"),
      CHAPTER_KNOWLEDGE[id].map(([concept, evidence, question]) =>
        h("div", { class: "mastery-row", dataset: { evidence } },
          h("label", {}, h("input", { type: "checkbox" }), h("span", {}, h("strong", {}, concept), h("span", {}, question))),
          h("a", { class: "evidence-link", href: c.href + "/" + evidence }, "回到对应小节 →")))),
    h("p", { class: "source-note" }, "阅读依据：", h("a", { href: id === "10" ? THESIS : BOOK_TOC, target: "_blank", rel: "noopener" }, c.book),
      id === "10" ? "。研究专题保留论文页码与实验假设；演示和完整训练分开说明。" : "。本课程按概念整合教材章节，正文注明示范数据、模型与更新条件。"),
    h("nav", { "aria-label": "前后章节" },
      h("a", { href: previous?.href ?? "#" }, previous ? "← 第" + previous.number + "章 · " + previous.short : "← 课程架构"),
      h("a", { class: "next-lesson", href: next?.href ?? "#" }, next ? "第" + next.number + "章 · " + next.short + " →" : "完成主线 · 回到课程架构 →")));
}
// Reuse validated interactions without showing the old exercise-page headers,
// navigation, or audits. Public chapters supply their own conceptual hierarchy.
export function capture(module) {
  const detached = h("div");
  const cleanup = module.mount(detached, { keyboard: false }) ?? (() => {});
  const sections = [...detached.querySelectorAll(".step")];
  const named = new Map();
  for (const section of sections) {
    const oldHeading = section.querySelector(":scope > h2");
    if (!oldHeading) continue;
    const name = oldHeading.textContent;
    const heading = h("h3", {}, name);
    oldHeading.replaceWith(heading); section.dataset.experimentSection = name; named.set(name, section);
  }
  return {
    cleanup,
    take: (...names) => names.map(name => { const el = named.get(name); if (!el) throw Error("Missing experiment section: " + name); return el; }),
    all: ({ exclude = [] } = {}) => sections.filter(el => !exclude.includes(el.dataset.experimentSection)),
    optional: () => [...detached.querySelectorAll(":scope > details.optional")],
  };
}
export function table(headers, rows) {
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, headers.map(x => h("th", {}, x)))),
    h("tbody", {}, rows.map(row => h("tr", {}, row.map(x => h("td", {}, x)))))));
}
