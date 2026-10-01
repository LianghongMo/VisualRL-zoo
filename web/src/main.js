import "./styles.css";

import { h } from "./ui/dom.js";
import { courseNav } from "./ui/shell.js";
import { lessonById } from "./lessons/curriculum.js";
import * as home from "./lessons/home.js";
import * as lesson01 from "./lessons/lesson01.js";
import * as lesson02 from "./lessons/lesson02.js";
import * as lesson03 from "./lessons/lesson03.js";
import * as lesson04 from "./lessons/lesson04.js";
import * as lesson05 from "./lessons/lesson05.js";
import * as lesson06 from "./lessons/lesson06.js";
import * as lesson07 from "./lessons/lesson07.js";
import * as lesson08 from "./lessons/lesson08.js";

// Routes are bare hash tokens (#lesson-08) so links work in a standalone file too.
const ROUTES = {
  "": home,
  "lesson-01": lesson01,
  "lesson-02": lesson02,
  "lesson-03": lesson03,
  "lesson-04": lesson04,
  "lesson-05": lesson05,
  "lesson-06": lesson06,
  "lesson-07": lesson07,
  "lesson-08": lesson08,
};

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
function show(key) {
  try {
    cleanup();
  } catch {}
  cleanup = () => {};
  current = key;
  const page = ROUTES[key] ?? home;
  const main = h("main", { class: "page" });
  const lesson = lessonById(key.replace(/^lesson-/, ""));
  document.title = lesson ? `${lesson.short} · Visual RL` : "Visual RL · 看见强化学习";
  app.replaceChildren(topbar(), lesson ? h("div", { class: "course-layout" }, courseNav(lesson.id), main) : main);
  try {
    cleanup = page.mount(main) ?? (() => {});
  } catch (err) {
    console.error(err);
    main.append(h("div", { class: "callout error" }, h("p", {}, h("strong", {}, "这一页未能加载。"), "错误信息：", h("code", {}, String(err?.message ?? err)))));
  }
  window.scrollTo(0, 0);
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
