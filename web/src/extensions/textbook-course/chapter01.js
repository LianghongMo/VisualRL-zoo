import * as mdp from "../../lessons/lesson01.js";
import { chapterHeader, chapterFooter, topic, capture, prose } from "./shell.js";
export function mount(root) {
  const a = capture(mdp);
  root.append(chapterHeader("01"),
    topic("01", "task",
      prose("先说清我们要解决什么：机器人在一个会反馈状态和奖励的世界中行动，希望找到让长期结果更好的动作规则。本章只把任务、状态和交互定义清楚；下一章再计算怎样比较这些规则。",
        "先操作下面的地图。每次按一个动作，观察新位置、奖励和是否终止，再把这些反馈对应到MDP的组成。"),
      ...a.take("动手验证：读懂机器人走的一步", "MDP：先把任务说完整")),
    topic("01", "state", ...a.take("为什么状态必须足够完整？"),
      prose("这里位置足够，是因为墙、悬崖和充电站固定，也没有电量或速度等隐藏条件。如果任务增加电量，状态就必须把它包括进来。状态定义会决定后面能否评价价值、复用经验和拼接路线。")),
    topic("01", "interaction", ...a.take("策略决定动作，模型决定后果", "检查理解"),
      prose("到这里先区分三件事：MDP规定任务，策略规定动作，经验记录实际发生的反馈。学习算法利用经验改变策略；已知模型时也可以直接计算后果。",
        "现在我们知道一条经历怎样产生，但还没有一个统一数字来比较两条完整经历。下一章从奖励走向回报，再定义一个策略的价值。")),
    chapterFooter("01", "先定义状态、动作、奖励、转移与终止；检验状态是否满足Markov；区分策略、模型和实际经验。任务定义完成后，才能问长期结果有多好。本章不提前求解最优策略。"));
  return a.cleanup;
}
