// TeX → MathML with KaTeX. Browsers render MathML natively, so no KaTeX CSS or fonts are needed.
import katex from "katex";

export function tex(source, { display = false } = {}) {
  const el = document.createElement(display ? "div" : "span");
  el.innerHTML = katex.renderToString(source, { output: "mathml", displayMode: display, throwOnError: true });
  return el;
}

export function equation(source, caption) {
  const box = document.createElement("div");
  const eq = document.createElement("div");
  eq.className = "equation";
  eq.dataset.tex = source;
  eq.append(tex(source, { display: true }));
  box.append(eq);
  if (caption) {
    const cap = document.createElement("p");
    cap.className = "equation-caption";
    cap.append(caption);
    box.append(cap);
  }
  return box;
}
