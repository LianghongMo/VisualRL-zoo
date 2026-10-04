import { h } from "../ui/dom.js";
import { CHAPTERS, PARTS, chapterById, BOOK, HARVARD } from "./curriculum.js";
import { CHAPTER_KNOWLEDGE } from "./knowledge.js";
export { equation } from "../ui/math.js";
export { prose, predict } from "../ui/shell.js";
export function table(headers,rows){return h("div",{class:"table-wrap"},h("table",{},h("thead",{},h("tr",{},headers.map(x=>h("th",{},x)))),h("tbody",{},rows.map(row=>h("tr",{},row.map(x=>h("td",{},x)))))));}
export function proof(kind,title,...body){return h("section",{class:"proof-step"},h("p",{class:"proof-kind"},kind),h("h3",{},title),...body);}
export function reading(label,url){return h("p",{class:"source-note"},"阅读：",h("a",{href:url,target:"_blank",rel:"noopener"},label));}
export function chapterNav(id){return h("aside",{class:"course-sidebar"},
  h("details",{class:"course-menu",open:!window.matchMedia?.("(max-width: 760px)")?.matches},
    h("summary",{},"从理论到连续机器人 · 10章"),
    h("nav",{"aria-label":"课程章节"},PARTS.map(p=>h("div",{class:"chapter-nav-group"},h("p",{},p.title),
      p.chapters.map(c=>h("a",{href:c.href,class:c.id===id?"active":"","aria-current":c.id===id?"page":undefined},h("span",{class:"nav-number"},c.number),h("span",{},c.short))))),
    h("a",{class:"nav-home",href:"#"},"← 课程目录"))),
  h("p",{class:"sidebar-note"},"任务 → 数据/表示 → 探索 → 策略 → 模仿 → 连续反馈控制。"));}
export function chapterHeader(id){const c=chapterById(id);return h("header",{class:"lesson-head chapter-head"},
  h("p",{class:"eyebrow"},"第 "+c.number+" 章 / 10 · "+c.group),h("h1",{},c.title),h("p",{class:"lead"},c.image),
  h("p",{class:"chapter-reference"},h("a",{href:c.source,target:"_blank",rel:"noopener"},c.book)),
  h("p",{class:"learning-goal"},h("strong",{},"学会："),c.goal,h("br"),h("strong",{},"基础："),c.prerequisite),
  h("p",{class:"chapter-connection"},c.relation.from),
  h("nav",{class:"chapter-toc","aria-label":"本章小节"},c.topics.map(t=>h("a",{href:c.href+"/"+t.id},h("b",{},t.number)," "+t.title))),
  h("details",{class:"chapter-essentials"},h("summary",{},"必会知识 · "+CHAPTER_KNOWLEDGE[id].length+" 项"),
    h("ul",{},CHAPTER_KNOWLEDGE[id].map(([concept])=>h("li",{},concept)))));}
export function topic(id,key,...body){const t=chapterById(id).topics.find(t=>t.id===key);if(!t)throw Error("Unknown topic "+id+"/"+key);
  return h("section",{class:"chapter-topic",dataset:{topic:key}},h("div",{class:"chapter-topic-heading"},
    h("p",{class:"eyebrow"},"小节 "+t.number),h("h2",{},t.title)),...body);}
export function chapterFooter(id,summary){const c=chapterById(id),previous=CHAPTERS[c.number-2],next=CHAPTERS[c.number];
  return h("footer",{class:"lesson-foot chapter-foot"},h("section",{class:"takeaway"},h("h2",{},"本章要点"),h("p",{},summary)),
    h("section",{class:"mastery-audit"},h("h2",{},"掌握检查"),CHAPTER_KNOWLEDGE[id].map(([concept,evidence,question])=>h("div",{class:"mastery-row",dataset:{evidence}},
      h("label",{},h("input",{type:"checkbox"}),h("span",{},h("strong",{},concept),h("span",{},question))),
      h("a",{class:"evidence-link",href:c.href+"/"+evidence},"回看 →")))),
    h("section",{class:"chapter-transition"},h("h2",{},next?"下一章":"继续研究"),h("p",{},c.relation.next)),
    h("nav",{"aria-label":"前后章节"},h("a",{href:previous?.href??"#"},previous?"← 第"+previous.number+"章":"← 课程主线"),
      h("a",{class:"next-lesson",href:next?.href??"#"},next?"第"+next.number+"章 · "+next.short+" →":"回到课程主线 →")));}
export function optional(title,...body){return h("details",{class:"optional"},h("summary",{},title),...body);}
export const bookReading=(label,page)=>reading("ABJKS · "+label,BOOK+"#page="+(page+12));
export const teachingSource=()=>reading("Harvard CS2824 · 公开课程与讲义",HARVARD);
