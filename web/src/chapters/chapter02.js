import * as returns from "../lessons/lesson04.js";
import * as optimal from "../lessons/lesson05.js";
import { h } from "../ui/dom.js";
import { chapterHeader, chapterFooter, topic, capture, prose, equation } from "./shell.js";
import { rewardShaping } from "./value-tricks.js";
export function mount(root) {
  const a = capture(returns), b = capture(optimal);
  root.append(chapterHeader("02"),
    topic("02", "returns",
      prose("第1章的经验是一串状态、动作和奖励。要评价长期结果，需要把这串奖励按同一规则汇总。本节先对比两条路线，再引入回报G与折扣γ。"),
      ...a.take("先看图像：奖励沿路线逐项累计", "动手验证：只改变折扣，不改变路线", "公式：从一步奖励到整段回报")),
    topic("02", "values",
      ...a.take("V 和 Q：在谁的策略下看未来？", "把回报存下来，就是价值"),
      prose("一条实际路线提供一个G样本；Vπ概括从状态出发、一直按π行动的预期G。Qπ先固定第一个动作，后面再按π行动。因此比较V或Q之前，要先说明使用哪个策略。",
        "本章定义评价对象，暂时不讨论怎样从大量样本更新它。第4章的MC和第5章的TD会分别给出估计方法。"),
      equation("J(\\pi)=\\mathbb E_{S_0\\sim\\rho_0}[V^\\pi(S_0)]",
        "任务常从初始分布ρ₀出发，用J比较策略。我们的充电任务从S开始，所以J(π)=Vπ(S)。"),
      ...a.optional()),
    topic("02", "optimal",
      ...b.take("我们的任务：找一个最优策略", "先定义最优，再找更省事的比较方法"),
      prose("现在可以精确表述任务：找到一个策略，使预期回报最大。A/B/C说明完整动作规则会产生不同结果；C优于这两种候选，只说明比较结果，还不是对所有可能策略的证明。",
        "如果每次都展开完整路线，计算会重复。第3章将把每个动作的未来拆成「眼前奖励＋下一状态价值」，由这些例子推导Bellman关系，再求解全局最优策略。"),
      h("a", { class: "btn", href: "#chapter-03/bellman" }, "继续：把完整未来拆成一步关系 →"),
      ...a.take("检查理解")),
    topic("02", "shaping", ...rewardShaping()),
    chapterFooter("02", "即时奖励 → 折扣回报G → 固定策略的Vπ/Qπ → 比较完整策略 → 定义最优价值和π*。势函数塑形用边界抵消保留这个比较目标；任意奖励变换则可能改任务。下一章解释如何复用后续价值并实际求解。"));
  return () => { a.cleanup(); b.cleanup(); };
}
