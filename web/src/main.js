import "./styles.css";
import "./course-details.css";

import { h } from "./ui/dom.js";
import { chapterNav } from "./chapters/shell.js";
import { resolveRoute } from "./lessons/curriculum.js";
import { CHAPTER_MODULES } from "./chapters/index.js";
import * as home from "./lessons/home.js";

const app = document.getElementById("app");
let cleanup = () => {};

function topbar() {
  return h(
    "header",
    { class: "topbar" },
    h(
      "div",
      { class: "topbar-inner" },
      h("a", { class: "wordmark", href: "#" }, "Visual RL", h("small", {}, "看见强化学习")),
      h(
        "nav",
        {},
        h("a", { href: "#" }, "课程首页"),
        h("a", { href: "https://github.com/LianghongMo/VisualRL-zoo" }, "GitHub"),
      ),
    ),
  );
}

let current = null;

function currentKey() {
  try {
    return location.hash.replace(/^#/, "");
  } catch {
    return current ?? "";
  }
}

// Show a lesson. Errors are shown on the page instead of leaving it blank.
let activeChapter = null;
function scrollToTopic(main, topic) {
  const target = topic && [...main.querySelectorAll(".chapter-topic")].find(el => el.dataset.topic === topic);
  if (target) target.scrollIntoView({ block: "start" });
  else window.scrollTo(0, 0);
}
function show(key) {
  const resolved = resolveRoute(key);
  const id = resolved.chapter?.id;
  if (id && activeChapter === id) {
    current = key;
    scrollToTopic(app.querySelector("main"), resolved.topic);
    return;
  }
  try { cleanup(); } catch {}
  cleanup = () => {};
  current = key; activeChapter = id ?? null;
  const main = h("main", { class: "page" });
  document.title = resolved.chapter ? "第" + resolved.chapter.number + "章 · " + resolved.chapter.title + " · Visual RL" : "Visual RL · 强化学习教材主线";
  app.replaceChildren(topbar(), id ? h("div", { class: "course-layout" }, chapterNav(id), main) : main);
  try { cleanup = (id ? CHAPTER_MODULES[id] : home).mount(main) ?? (() => {}); }
  catch (err) {
    console.error(err);
    main.append(h("div", { class: "callout error" }, h("p", {}, h("strong", {}, "这一章未能加载。"), "错误信息：", h("code", {}, String(err?.message ?? err)))));
  }
  scrollToTopic(main, resolved.topic);
}

// In-page links are handled here, not by the browser: inside a sandboxed srcdoc frame (how the
// page is hosted as an artifact) a plain "#lesson-01" link resolves against the host's URL and
// navigates the frame away, leaving it blank.
document.addEventListener("click", (e) => {
  const a = e.target.closest?.("a[href^='#']");
  if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
  e.preventDefault();
  const key = a.getAttribute("href").slice(1);
  try {
    if (location.hash.replace(/^#/, "") !== key) {
      history.pushState?.(null, "", `#${key}`);
    }
  } catch {}
  show(key);
});

window.addEventListener("popstate", () => { if (currentKey() !== current) show(currentKey()); });
window.addEventListener("hashchange", () => {
  if (currentKey() !== current) show(currentKey());
});
// "ResizeObserver loop ..." is a harmless browser notice, not a failure of the page.
const harmless = (message) => /ResizeObserver loop/i.test(String(message ?? ""));
window.addEventListener("error", (e) => !harmless(e.message) && showErrorBanner(e.message));
window.addEventListener("unhandledrejection", (e) => showErrorBanner(String(e.reason?.message ?? e.reason)));

function showErrorBanner(message) {
  if (document.querySelector(".error-banner")) return;
  document.body.append(h("div", { class: "error-banner", role: "alert" }, "页面运行出错：", h("code", {}, message)));
}

show(currentKey());
