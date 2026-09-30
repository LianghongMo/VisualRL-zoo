// Tiny DOM helpers. h("div", { class: "x", onclick }, child, "text", [more])
const SVG_NS = "http://www.w3.org/2000/svg";

function build(el, attrs, children) {
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const h = (tag, attrs, ...children) => build(document.createElement(tag), attrs, children);
export const s = (tag, attrs, ...children) => build(document.createElementNS(SVG_NS, tag), attrs, children);

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...children) {
  clear(el);
  return append(el, children);
}

// A labelled range input. onInput receives the numeric value.
export function slider({ id, label, min, max, step, value, format = (v) => v, onInput }) {
  const output = h("output", { for: id }, format(value));
  const input = h("input", { type: "range", id, min, max, step, value });
  input.addEventListener("input", () => {
    const v = Number(input.value);
    output.textContent = format(v);
    onInput(v);
  });
  const el = h("label", { class: "field", for: id }, h("span", { class: "field-row" }, h("span", {}, label), output), input);
  el.set = (v) => {
    input.value = v;
    output.textContent = format(Number(input.value));
  };
  el.input = input;
  return el;
}

export function button(label, { kind = "", onClick, kbd, title, id } = {}) {
  const el = h("button", { class: `btn ${kind}`.trim(), type: "button", title, id }, label, kbd ? h("kbd", {}, kbd) : null);
  if (onClick) el.addEventListener("click", onClick);
  return el;
}

export function segmented(options, { value, onChange, label }) {
  const el = h("div", { class: "segmented", role: "group", "aria-label": label });
  const buttons = options.map((opt) =>
    h(
      "button",
      { type: "button", "aria-pressed": String(opt.value === value), onclick: () => select(opt.value, true) },
      opt.swatch ? h("span", { class: "swatch", style: { background: opt.swatch } }) : null,
      opt.label,
    ),
  );
  function select(v, notify) {
    buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === v)));
    if (notify) onChange(v);
  }
  append(el, buttons);
  el.select = (v) => select(v, false);
  return el;
}

// Keyboard shortcuts that ignore typing in form fields. Returns an unbind function.
export function shortcuts(map) {
  const handler = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      if (e.target.type !== "range") return;
    }
    const fn = map[e.key.toLowerCase()];
    if (fn) {
      e.preventDefault();
      fn();
    }
  };
  window.addEventListener("keydown", handler);
  return () => window.removeEventListener("keydown", handler);
}

export const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
