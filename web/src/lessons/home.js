import { h } from "../ui/dom.js";
import { PARTS } from "./curriculum.js";

const LOOP = [
  ["environment", "env"],
  ["experience", "env"],
  ["learning target", "learn"],
  ["parameter update", "learn"],
  ["policy / value change", "learn"],
  ["new behavior", "env"],
];

export function mount(root) {
  root.append(
    h(
      "section",
      { class: "hero" },
      h("p", { class: "eyebrow" }, "An interactive reinforcement learning tutorial"),
      h("h1", {}, "Learn reinforcement learning by seeing every decision, target, and update."),
      h(
        "p",
        { class: "lead" },
        "Most demos show what an agent does. These lessons show how it learns: pause at any update and see the data it used, the target it computed, the error, and the value that changed.",
      ),
      h(
        "div",
        { class: "loop", "aria-label": "The learning loop" },
        LOOP.flatMap(([label, tone], i) => [
          i ? h("span", { class: "loop-arrow", "aria-hidden": "true" }, "→") : null,
          h("span", { class: `loop-step ${tone}` }, label),
        ]),
      ),
      h(
        "p",
        { class: "note" },
        h("span", { class: "dot env" }),
        " Acting (environment steps, experience)   ",
        h("span", { class: "dot learn" }),
        " Learning (targets, updates, changed values). The two are separate buttons in every lesson.",
      ),
    ),
    h(
      "div",
      { class: "curriculum" },
      PARTS.map((part) =>
        h(
          "section",
          { class: "part" },
          h("h2", { class: "part-title" }, part.part, h("strong", {}, part.title)),
          h(
            "ol",
            { class: "lesson-list" },
            part.lessons.map((l) =>
              h(
                "li",
                {},
                h(
                  l.ready ? "a" : "div",
                  { class: `lesson-row${l.ready ? "" : " planned"}`, href: l.ready ? `#lesson-${l.id}` : undefined },
                  h("span", { class: "n" }, l.id),
                  h("span", { class: "t" }, l.title, h("span", { class: "d" }, l.about)),
                  h("span", { class: `status${l.ready ? " ready" : ""}` }, l.ready ? "Open" : "Planned"),
                ),
              ),
            ),
          ),
        ),
      ),
    ),
    h(
      "section",
      { class: "about" },
      h("h2", { class: "part-title" }, "How it works"),
      h(
        "div",
        { class: "prose" },
        h(
          "p",
          {},
          "Every number on these pages comes from a real computation. The lessons run a JavaScript copy of the Python algorithms in the ",
          h("a", { href: "https://github.com/LianghongMo/VisualRL-zoo" }, "visualrl"),
          " package, and a test replays Python's recorded updates through it to check that both produce identical learning traces.",
        ),
        h(
          "p",
          {},
          "Values drawn with a dashed line are reference quantities computed from the environment's model. The learning agent never sees them.",
        ),
      ),
    ),
  );
  return () => {};
}
