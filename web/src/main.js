import "./styles.css";

import { h } from "./ui/dom.js";
import * as home from "./lessons/home.js";
import * as lesson01 from "./lessons/lesson01.js";
import * as lesson02 from "./lessons/lesson02.js";
import * as lesson03 from "./lessons/lesson03.js";
import * as lesson07 from "./lessons/lesson07.js";
import * as lesson08 from "./lessons/lesson08.js";
import * as lesson14 from "./lessons/lesson14.js";

// Routes are bare hash tokens (#lesson-08) so links work in a standalone file too.
const ROUTES = {
  "": home,
  "lesson-01": lesson01,
  "lesson-02": lesson02,
  "lesson-03": lesson03,
  "lesson-07": lesson07,
  "lesson-08": lesson08,
  "lesson-14": lesson14,
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
      h("a", { class: "wordmark", href: "#" }, h("span", { class: "wordmark-mark", "aria-hidden": "true" }, h("i"), h("i")), "Visual RL"),
      h(
        "nav",
        {},
        h("a", { href: "#" }, "Lessons"),
        h("a", { href: "https://github.com/LianghongMo/VisualRL-zoo" }, "GitHub"),
      ),
    ),
  );
}

function route() {
  cleanup();
  const key = location.hash.replace(/^#/, "");
  const page = ROUTES[key] ?? home;
  const main = h("main", { class: "page" });
  app.replaceChildren(topbar(), main);
  cleanup = page.mount(main) ?? (() => {});
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
route();
