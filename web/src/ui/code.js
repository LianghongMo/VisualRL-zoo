// A small Python highlighter for the "Code" step of each lesson.
import { h } from "./dom.js";

const KEYWORDS = new Set(
  "def class return if else elif for in not and or is None True False while import from as with lambda raise".split(" "),
);
const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|"[^"\n]*"|'[^'\n]*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|(\s+)|(.)/g;

// Lines listed in `highlight` (1-based) get a wash, to point at the update itself.
export function codeBlock(source, { title, highlight = [] } = {}) {
  const pre = h("pre");
  source.split("\n").forEach((line, i) => {
    const lineEl = h("span", { class: highlight.includes(i + 1) ? "tok-hl" : undefined });
    for (const [, comment, string, number, word, space, other] of line.matchAll(TOKEN)) {
      if (comment) lineEl.append(h("span", { class: "tok-c" }, comment));
      else if (string) lineEl.append(h("span", { class: "tok-s" }, string));
      else if (number) lineEl.append(h("span", { class: "tok-n" }, number));
      else if (word) lineEl.append(KEYWORDS.has(word) ? h("span", { class: "tok-k" }, word) : word);
      else lineEl.append(space ?? other);
    }
    pre.append(lineEl, "\n");
  });
  return h("div", { class: "code" }, title ? h("div", { class: "code-head" }, h("span", {}, title), h("span", {}, "Python")) : null, pre);
}
