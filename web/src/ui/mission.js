// A task card for an experiment: what to do, step by step, and what it showed.
//
// steps: [{ text, done(baseline) -> bool, baseline?() -> any }]
// Steps complete in order. When a step becomes the current one, its baseline() is recorded, so a
// step like "press Learn five times" counts from that moment. Call update() after every change.
import { h, replace } from "./dom.js";

export class Mission {
  constructor({ title, goal, steps, conclusion }) {
    this.title = title;
    this.goal = goal;
    this.steps = steps;
    this.conclusion = conclusion;
    this.list = h("ol", { class: "mission-steps" });
    this.result = h("div", { class: "mission-result", hidden: true });
    this.restart = h("button", { type: "button", class: "mission-restart", onclick: () => this.reset() }, "restart the task");
    this.el = h(
      "section",
      { class: "mission", "aria-live": "polite" },
      h("div", { class: "mission-head" }, h("span", { class: "mission-tag" }, "Your task"), h("strong", {}, title), h("span", { class: "grow" }), this.restart),
      h("p", { class: "mission-goal" }, goal),
      this.list,
      this.result,
    );
    this.done = this.steps.map(() => false);
    this.baselines = this.steps.map(() => undefined);
    this.draw(); // no step is checked until the lesson's first update(), when its state exists
  }

  reset() {
    this.done = this.steps.map(() => false);
    this.baselines = this.steps.map(() => undefined);
    this.update();
  }

  get finished() {
    return this.done.every(Boolean);
  }

  update() {
    for (let i = 0; i < this.steps.length; i++) {
      if (this.done[i]) continue;
      const step = this.steps[i];
      if (this.baselines[i] === undefined) this.baselines[i] = step.baseline ? step.baseline() : null;
      if (step.done(this.baselines[i])) {
        this.done[i] = true;
        continue;
      }
      break;
    }
    this.draw();
  }

  draw() {
    const current = this.done.indexOf(false);
    replace(
      this.list,
      this.steps.map((st, i) =>
        h(
          "li",
          { class: this.done[i] ? "done" : i === current ? "current" : "todo" },
          h("span", { class: "mission-mark", "aria-hidden": "true" }, this.done[i] ? "✓" : String(i + 1)),
          h("span", {}, st.text, i === current ? h("span", { class: "mission-now" }, "  ← now") : null),
        ),
      ),
    );
    this.result.hidden = !this.finished;
    if (this.finished) replace(this.result, h("strong", {}, "What you just saw. "), typeof this.conclusion === "function" ? this.conclusion() : this.conclusion);
  }
}
