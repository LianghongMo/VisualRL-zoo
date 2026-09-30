// Page scaffolding shared by the lessons: the header and the Question → Challenge steps (README §15).
import { LESSONS, lessonById } from "../lessons/curriculum.js";
import { h } from "./dom.js";

export function lessonHeader(id, { lead, concepts = [] }) {
  const lesson = lessonById(id);
  return h(
    "header",
    { class: "lesson-head" },
    h("p", { class: "eyebrow" }, `${lesson.part} · ${lesson.partTitle} · Lesson ${lesson.id}`),
    h("h1", {}, lesson.title),
    h("p", { class: "lead" }, lead),
    concepts.length ? h("div", { class: "chips" }, concepts.map((c) => h("span", { class: "chip" }, c))) : null,
  );
}

export function step(label, ...body) {
  return h("section", { class: "step" }, h("h2", { class: "step-label" }, label), h("div", { class: "step-body" }, body));
}

// A step whose body needs the full page width (benches, charts): the label sits above it.
export function wideStep(label, ...body) {
  return h("section", { class: "step wide" }, h("h2", { class: "step-label" }, label), h("div", { class: "step-body" }, body));
}

export const prose = (...paragraphs) => h("div", { class: "prose" }, paragraphs.map((p) => (p instanceof Node ? p : h("p", {}, p))));

export function lessonFooter(id) {
  const ready = LESSONS.filter((l) => l.ready);
  const i = ready.findIndex((l) => l.id === id);
  const prev = ready[i - 1];
  const next = ready[i + 1];
  return h(
    "nav",
    { class: "lesson-foot", "aria-label": "More lessons" },
    prev ? h("a", { href: `#lesson-${prev.id}` }, `← ${prev.id} ${prev.title}`) : h("a", { href: "#" }, "← All lessons"),
    next ? h("a", { href: `#lesson-${next.id}` }, `${next.id} ${next.title} →`) : h("a", { href: "#" }, "All lessons →"),
  );
}

// "Predict" step: pick an answer, then see whether it was right and why.
export function predict({ question, choices, answer, explain }) {
  const verdict = h("div", { class: "prose", hidden: true });
  const buttons = choices.map((choice, i) =>
    h(
      "button",
      { class: "choice", type: "button", "aria-pressed": "false", onclick: () => pick(i) },
      h("span", {}, choice.label),
      choice.detail ? h("small", {}, choice.detail) : null,
    ),
  );
  function pick(i) {
    buttons.forEach((b, j) => {
      b.setAttribute("aria-pressed", String(i === j));
      b.classList.toggle("correct", j === answer);
      b.disabled = true;
    });
    const right = i === answer;
    verdict.replaceChildren(
      h("p", {}, h("span", { class: `verdict ${right ? "good" : "bad"}` }, right ? "Right. " : "Not quite. "), explain()),
    );
    verdict.hidden = false;
  }
  return h("div", { class: "step-body" }, h("p", { class: "prose" }, question), h("div", { class: "choices" }, buttons), verdict);
}
