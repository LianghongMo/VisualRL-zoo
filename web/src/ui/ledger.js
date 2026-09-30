// The ledger shows one learning trace as a derivation: the experience it used,
// the quantities it computed, and the value it changed. It keeps a history so
// earlier updates can be inspected again.
import { h, replace } from "./dom.js";
import { tex } from "./math.js";

const HISTORY = 400;

export class Ledger {
  // groupsFor(trace) -> [{ label, tone: "env" | "learn" | undefined, rows: [{ label, formula, value, emph, result }] }]
  constructor({ groupsFor, titleFor, empty, onSelect = () => {} }) {
    this.groupsFor = groupsFor;
    this.titleFor = titleFor;
    this.emptyText = empty;
    this.onSelect = onSelect;
    this.history = [];
    this.index = -1;

    this.titleEl = h("span", { class: "title" });
    this.posEl = h("span", { class: "note num ledger-pos" });
    this.prevBtn = h("button", { type: "button", "aria-label": "Previous update", onclick: () => this.go(-1) }, "‹");
    this.nextBtn = h("button", { type: "button", "aria-label": "Next update", onclick: () => this.go(1) }, "›");
    this.body = h("div", { class: "ledger-body" });
    this.el = h(
      "section",
      { class: "ledger", "aria-live": "polite" },
      h(
        "div",
        { class: "ledger-head" },
        h("span", { class: "dot learn" }),
        this.titleEl,
        h("span", { class: "grow" }),
        this.posEl,
        h("span", { class: "ledger-nav" }, this.prevBtn, this.nextBtn),
      ),
      this.body,
    );
    this.render(false);
  }

  get current() {
    return this.history[this.index] ?? null;
  }

  get browsing() {
    return this.index !== this.history.length - 1;
  }

  push(trace) {
    this.history.push(trace);
    if (this.history.length > HISTORY) this.history.shift();
    this.index = this.history.length - 1;
    this.render(true);
  }

  pushMany(traces) {
    if (!traces.length) return;
    this.history.push(...traces.slice(-HISTORY));
    if (this.history.length > HISTORY) this.history.splice(0, this.history.length - HISTORY);
    this.index = this.history.length - 1;
    this.render(true);
  }

  reset() {
    this.history = [];
    this.index = -1;
    this.render(false);
  }

  go(delta) {
    const next = Math.min(this.history.length - 1, Math.max(0, this.index + delta));
    if (next === this.index) return;
    this.index = next;
    this.render(true);
    this.onSelect(this.current, this.browsing);
  }

  render(fresh) {
    const trace = this.current;
    this.prevBtn.disabled = this.index <= 0;
    this.nextBtn.disabled = this.index >= this.history.length - 1;
    this.posEl.textContent = this.history.length ? `${this.index + 1} / ${this.history.length}` : "";
    if (!trace) {
      this.titleEl.textContent = "Learning trace";
      replace(this.body, h("p", { class: "ledger-empty" }, this.emptyText));
      return;
    }
    this.titleEl.textContent = this.titleFor(trace);
    let i = 0;
    const groups = this.groupsFor(trace).map((group) =>
      h(
        "div",
        { class: "ledger-group" },
        h("div", { class: "ledger-group-label" }, group.tone ? h("span", { class: `dot ${group.tone}` }) : null, group.label),
        group.rows.map((row) =>
          h(
            "div",
            { class: `ledger-row${row.emph ? " emph" : ""}${row.result ? " result" : ""}`, style: { "--i": i++ } },
            h("span", { class: "label" }, row.label),
            h("span", { class: "formula" }, row.formula ? tex(row.formula) : ""),
            h("span", { class: "value" }, row.value),
          ),
        ),
      ),
    );
    replace(this.body, groups);
    this.el.classList.remove("fresh");
    if (fresh) {
      void this.el.offsetWidth; // restart the row animation
      this.el.classList.add("fresh");
    }
  }
}
