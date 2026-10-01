import { foundations } from "./foundations.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { PolicyIteration } from "../rl/tabular/dp.js";
import { h, button, replace } from "../ui/dom.js";
import { lessonHeader, step, prose, takeaway, lessonFooter } from "../ui/shell.js";
import { WorldView, num, metric } from "../ui/world-view.js";

export function mount(root) {
  const env = GridWorld.chargingRoom(); let pi = new PolicyIteration(env.model(), 0.9), evaluated = false, stable = false, operation = "";
  const map = new WorldView(env, { caption: "数字表示当前估计 V；箭头表示策略允许的动作。初始策略随机选择四个方向，所以每格都有四个箭头。" });
  const stats = h("div", { class: "metrics" }), note = h("p", { class: "observation", role: "status" });
  const improveBtn = button("2 · 改进：切换到最好的动作", { kind: "env", onClick: () => { const t = pi.improveStep(); stable = t.stable; evaluated = false; operation = `改进改变了 ${t.changed_states.length} 个位置的策略，数字没有改变。${stable ? "策略已稳定，不再需要改变。" : "箭头换了，新策略的价值需要重新评价。"}`; render(); } });
  function render() {
    map.render({ values: pi.V, policy: pi.policy });
    replace(stats, metric("V(出发点)", num(pi.V[env.start])), metric("改进次数", pi.improvements));
    improveBtn.disabled = !evaluated || stable;
    note.textContent = operation || "先评价：箭头不动，数字改变。再改进：数字不动，箭头改变。重复这个顺序。";
  }
  root.append(lessonHeader("06"), step("先看图像：数字和箭头轮流改变", prose("已知完整环境模型时，先计算当前策略的 V，再按照这些价值选择更好的动作。评价策略是在回答「照这些箭头走，会怎么样？」；改进策略是在回答「既然知道这些结果，箭头该怎样换？」")),
    ...foundations("06"),
    step("动手验证：每次只看一种变化", h("div", { class: "experiment" }, h("div", { class: "toolbar" }, button("1 · 评价：算出当前策略的价值", { kind: "learn", onClick: () => { pi.evaluate({ theta: 1e-10 }); evaluated = true; operation = "评价完成：只更新了数字，箭头仍然表示原来的策略。现在可以改进。"; render(); } }), improveBtn, button("重新开始", { kind: "ghost", onClick: () => { pi = new PolicyIteration(env.model(), 0.9); evaluated = false; stable = false; operation = ""; render(); } })), h("div", { class: "experiment-grid" }, map.el, h("div", { class: "experiment-reading" }, stats, note)))),
    takeaway("策略评价改变价值估计，策略改进改变动作选择。评价与改进交替进行，是策略迭代。它和价值迭代一样，在这一实验中都使用已知的环境模型。"), lessonFooter("06"));
  render(); return () => {};
}
