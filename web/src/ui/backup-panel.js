// One Bellman backup written out as a calculation: every move out of a node, where it can land,
// reward + γ × value where it lands, and then the max (or the policy's average) of the moves.
import { h, replace } from "./dom.js";
import { ARROWS, fmt, fmtShort } from "./format.js";

const MOVES = ["up", "right", "down", "left"];

export class BackupPanel {
  // name(state) -> label for a node
  constructor({ name, title = "Bellman backup" }) {
    this.name = name;
    this.title = title;
    this.el = h("section", { class: "backup", "aria-live": "polite" });
    this.expanded = new Set(); // moves whose outcomes are listed (the best ones always are)
    this.show(null);
  }

  // trace: an optimal_backup trace (max over moves) or a bellman_backup trace (policy average)
  show(trace) {
    if (trace && trace.state !== this.lastState) this.expanded = new Set();
    this.lastTrace = trace;
    this.lastState = trace?.state;
    if (!trace) {
      replace(this.el, h("div", { class: "backup-head" }, h("strong", {}, this.title)), h("p", { class: "backup-empty" }, "Click a node in the graph to see its backup."));
      return;
    }
    const g = fmtShort(trace.discount, 2);
    const optimal = trace.algorithm === "optimal_backup";
    const actions = optimal
      ? trace.actions
      : [0, 1, 2, 3].map((a) => {
          const branches = trace.branches.filter((b) => b.action === a);
          const q = branches.reduce((sum, b) => sum + b.transition_prob * (b.reward + trace.discount * b.next_value), 0);
          return { action: a, q, branches, weight: branches[0]?.action_prob ?? 0 };
        });
    const term = (b) => `${fmtShort(b.reward, 2)} + ${g} × ${fmt(b.next_value, 3)}`;
    const land = (b) => `${this.name(b.next_state)}${b.reward === -10 ? " (ledge)" : ""}`;
    const rows = [];
    for (const act of actions) {
      const best = optimal && trace.best_actions.includes(act.action);
      const label = `${ARROWS[act.action]} ${MOVES[act.action]}${optimal ? "" : `  · π = ${fmtShort(act.weight, 2)}`}`;
      if (act.branches.length === 1) {
        const b = act.branches[0];
        rows.push(h("tr", { class: `action${best ? " best" : ""}` }, h("td", { class: "left" }, label), h("td", { class: "left" }, land(b)), h("td", { class: "left" }, term(b)), h("td", {}, fmt(act.q, 3))));
      } else {
        const open = best || this.expanded.has(act.action);
        const toggle = () => {
          if (this.expanded.has(act.action)) this.expanded.delete(act.action);
          else this.expanded.add(act.action);
          this.show(this.lastTrace);
        };
        rows.push(
          h(
            "tr",
            { class: `action toggle-row${best ? " best" : ""}`, onclick: best ? null : toggle, title: best ? null : "show or hide the outcomes" },
            h("td", { class: "left" }, `${best ? "" : open ? "▾ " : "▸ "}${label}`),
            h("td", { class: "left" }, `${act.branches.length} outcomes`),
            h("td", { class: "left" }, open ? "average:" : "average of the outcomes"),
            h("td", {}, fmt(act.q, 3)),
          ),
        );
        if (!open) continue;
        for (const b of act.branches) {
          rows.push(h("tr", { class: "branch" }, h("td", { class: "left" }, ""), h("td", { class: "left" }, land(b)), h("td", { class: "left" }, `${fmtShort(b.transition_prob, 3)} × (${term(b)})`), h("td", {}, fmt(b.contribution, 3))));
        }
      }
    }
    const result = optimal
      ? `New value = the largest move = ${fmt(trace.value_after, 3)}   (best: ${trace.best_actions.map((a) => ARROWS[a]).join(" ")})`
      : `New value = Σ π(a|s) × value of a = ${fmt(trace.value_after, 3)}`;
    replace(
      this.el,
      h("div", { class: "backup-head" }, h("strong", {}, `${this.title} at ${this.name(trace.state)}`), h("span", { class: "grow" }), h("span", { class: "note num" }, `value now ${fmt(trace.value_before, 3)}`)),
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", { class: "left" }, "move"), h("th", { class: "left" }, "lands on"), h("th", { class: "left" }, "r + γ × V(lands on)"), h("th", {}, "value"))),
        h("tbody", {}, rows),
        h("tfoot", {}, h("tr", {}, h("td", { class: "left", colspan: 4 }, result))),
      ),
    );
  }
}
