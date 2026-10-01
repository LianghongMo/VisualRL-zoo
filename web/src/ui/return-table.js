// The return of one walk, written out edge by edge.
//
// The same number two ways. Forward: G_0 = Σ_t γ^t r_t, every reward weighted by how far ahead it
// is (the γ^t and γ^t · r_t columns). Backward: G_t = r_t + γ G_{t+1}, filled in from the end of the
// walk (Trajectory.returns). The backward form is what a Bellman backup does at every node.
import { makeTransition, Trajectory } from "../rl/core.js";
import { h } from "./dom.js";
import { ARROWS, fmt, fmtShort } from "./format.js";

// Follow choose(state) from `start` on the deterministic graph until the walk ends. choose returns
// null to stop early (no move worth taking). Stops after maxSteps if the walk never ends.
export function unrollWalk(env, start, choose, { maxSteps = 40 } = {}) {
  const traj = new Trajectory();
  let s = start;
  for (let t = 0; t < maxSteps; t++) {
    const a = choose(s);
    if (a === null || a === undefined) break;
    const [s2, r, done] = env.move(s, a);
    traj.append(makeTransition({ state: s, action: a, reward: r, next_state: s2, terminated: done }));
    s = s2;
    if (done) break;
  }
  return traj;
}

const sup = (base, exp) => h("span", {}, base, h("sup", {}, exp));
const sub = (base, i) => h("span", {}, base, h("sub", {}, i));

export function returnTable(env, traj, gamma, { name }) {
  const G = traj.returns(gamma);
  const terms = traj.discountedRewards(gamma);
  const n = traj.length;
  const ended = n > 0 && traj.transitions[n - 1].terminated;
  const g = fmtShort(gamma, 2);
  const rows = traj.transitions.map((tr, t) =>
    h(
      "tr",
      { class: tr.reward !== 0 ? "action best" : "" },
      h("td", {}, t),
      h("td", { class: "left" }, `${ARROWS[tr.action]} ${tr.reward === env.cliffReward ? "ledge → dock" : name(tr.next_state)}`),
      h("td", {}, fmtShort(tr.reward, 2)),
      h("td", {}, fmt(gamma ** t, 3)),
      h("td", {}, fmt(terms[t], 3)),
      h("td", { title: `${fmtShort(tr.reward, 2)} + ${g} × ${fmt(t + 1 < n ? G[t + 1] : 0, 3)}` }, fmt(G[t], 3)),
    ),
  );
  if (!ended && n > 0) rows.push(h("tr", {}, h("td", { class: "left wrap", colspan: 6 }, `… the walk has not ended after ${n} edges; later rewards are left out.`)));
  const sum = terms.reduce((a, b) => a + b, 0);
  return h(
    "div",
    {},
    h(
      "div",
      { class: "return-scroll" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", {}, "t"), h("th", { class: "left" }, "move"), h("th", {}, sub("r", "t")), h("th", {}, sup("γ", "t")), h("th", {}, sup("γ", "t"), " · ", sub("r", "t")), h("th", {}, sub("G", "t")))),
        h("tbody", {}, rows),
        h(
          "tfoot",
          {},
          h("tr", {}, h("td", { class: "left wrap", colspan: 4 }, "return G₀: down the γᵗ · rₜ column, or the top of the Gₜ column"), h("td", {}, fmt(sum, 3)), h("td", {}, fmt(n ? G[0] : 0, 3))),
        ),
      ),
    ),
    h("p", { class: "return-note" }, "Two ways to the same number. Forward: every reward times γᵗ, its weight t edges ahead. Backward, from the last row up: Gₜ = rₜ + γ × Gₜ₊₁ (hover a Gₜ to see it)."),
  );
}
