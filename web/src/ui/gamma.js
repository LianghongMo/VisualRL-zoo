// The discount slider used across Part II, with a readout of how far ahead γ makes the robot look.
import { h, slider } from "./dom.js";
import { fmtShort } from "./format.js";

// The weights γ^t add up to 1/(1−γ): roughly how many edges ahead still count.
export function horizonText(gamma) {
  if (gamma === 0) return "near-sighted: only the next reward counts";
  const kind = gamma <= 0.5 ? "near-sighted" : gamma >= 0.9 ? "far-sighted" : "in between";
  return `${kind}: looks about 1/(1−γ) = ${fmtShort(1 / (1 - gamma), 1)} edges ahead`;
}

export function gammaControl({ id, value = 0.9, onInput }) {
  const hint = h("span", { class: "gamma-hint" }, horizonText(value));
  const field = slider({
    id,
    label: "discount γ",
    min: 0,
    max: 0.99,
    step: 0.01,
    value,
    format: (v) => v.toFixed(2),
    onInput: (v) => {
      hint.textContent = horizonText(v);
      onInput(v);
    },
  });
  const el = h("div", { class: "gamma-control" }, field, hint);
  el.set = (v) => {
    field.set(v);
    hint.textContent = horizonText(v);
  };
  return el;
}
