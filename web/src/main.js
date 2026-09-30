import "./styles.css";

import { h } from "./ui/dom.js";
import * as home from "./lessons/home.js";
import * as lesson02 from "./lessons/lesson02.js";
import * as lesson08 from "./lessons/lesson08.js";
import * as lesson09 from "./lessons/lesson09.js";
import * as lesson15 from "./lessons/lesson15.js";

// Routes are bare hash tokens (#lesson-09) so links work in a standalone file too.
const ROUTES = { "": home, "lesson-02": lesson02, "lesson-08": lesson08, "lesson-09": lesson09, "lesson-15": lesson15 };

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
