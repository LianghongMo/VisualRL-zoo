import * as mdp from "../lessons/lesson01.js";
import * as returns from "../lessons/lesson04.js";
import * as optimal from "../lessons/lesson05.js";
import { h } from "../ui/dom.js";
import { chapterHeader, chapterFooter, topic, capture, prose, equation, table } from "./shell.js";
export function mount(root) {
  const a = capture(mdp), b = capture(returns), c = capture(optimal);
  root.append(chapterHeader("01"),
    topic("01", "task", ...a.take("动手验证：读懂机器人走的一步", "MDP：先把任务说完整", "为什么状态必须足够完整？", "策略决定动作，模型决定后果", "检查理解"),
      prose("MDP 定义的是问题，不是一种学习算法。知道 p 时可以计算；不知道 p 时可以采样。无论后来用动态规划、MC、TD 或神经网络，首先都要说明状态、动作、奖励和终止。")),
    topic("01", "returns", ...b.take("先看图像：奖励沿路线逐项累计", "动手验证：只改变折扣，不改变路线", "公式：从一步奖励到整段回报", "V 和 Q：在谁的策略下看未来？", "把回报存下来，就是价值", "检查理解"), ...b.optional(),
      equation("J(\\pi)=\\mathbb E_{S_0\\sim\\rho_0}[V^\\pi(S_0)]",
        "实际任务常从初始分布ρ₀出发优化J。有限折扣MDP的一个最优策略还能在每个状态都取得V*。")),
    topic("01", "optimal", ...c.take("我们的任务：找一个最优策略", "先定义最优，再找更省事的比较方法", "从路线中总结：一步奖励，接上后续价值", "固定策略的价值：平均动作，而不是替它改主意", "随机时先平均，再比较动作", "Bellman 方程：把刚才的图像写完整"),
      table(["本章定义", "后续章节怎样使用"], [
        ["Bellman期望关系：评价固定π", "第3章在模型上算；第4/5章用回报或一步经验估计"],
        ["Bellman最优关系：选择最好动作", "第3章价值迭代；第5章Q-learning"],
        ["Vπ/Qπ与π*", "第8章用参数表示；第9章直接优化策略"],
        ["完整Markov状态、奖励和终止", "第10章检验片段拼接与目标改标是否有效"],
      ]),
      prose("到这里我们把问题和解应满足的关系说完整了，但没有把求解算法混进定义。价值迭代、策略迭代属于第3章；由真实经验学习属于第4章及之后。"),
      h("a", { class: "btn", href: "#chapter-03/value-iteration" }, "已理解定义？查看第3章怎样求解 →")),
    chapterFooter("01", "定义MDP与充分状态 → 用回报描述目标 → 用Vπ/Qπ评价策略 → 从具体路线拆出一步递推 → 用Bellman方程描述最优价值。定义清楚以后，再讨论怎样计算或学习。"));
  return () => { a.cleanup(); b.cleanup(); c.cleanup(); };
}
