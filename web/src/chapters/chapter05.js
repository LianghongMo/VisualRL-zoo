import * as prediction from "../lessons/lesson07.js";
import * as control from "../lessons/lesson08.js";
import * as bandit from "./bandit-supplement.js";
import { h } from "../ui/dom.js";
import { chapterHeader, chapterFooter, topic, capture, prose, equation, table } from "./shell.js";
export function mount(root) {
  const a = capture(prediction), b = capture(control), c = capture(bandit);
  root.append(chapterHeader("05"),
    topic("05", "prediction",
      prose("第4章用完整回报评价固定策略。本章先把目标换成「真实一步＋后继估计」，于是未来没走完也能更新；然后把同样的思想用于动作价值Q，参与控制。",
        "先走3步，TD可以用已观察的一条经验；MC仍等回合结束。起初奖励与后继估计都为0，能更新不代表数字立刻增加。"),
      ...a.take("动手验证：先走几步，再试学习", "公式：同一旧估计，两种学习目标", "目标的误差从哪里来？", "检查理解"), ...a.optional()),
    topic("05", "exploration",
      prose("学会更新一个价值，还没有保证会收集到有用经验。总走已知的+1路线，就无法比较去+10的动作。利用按当前估计行动；探索为尚不确定的动作获取证据。",
        "第4章MC控制已用ε-soft保持覆盖；本章把同样的行为规则接到一步TD控制，并检查探索的下一动作怎样进入SARSA与Q-learning的目标。"),
      equation("\\pi_\\epsilon(a\\mid s)=\\frac{\\epsilon}{K}+(1-\\epsilon)\\frac{\\mathbf1\\{a\\in\\arg\\max_b Q(s,b)\\}}{|\\arg\\max_b Q(s,b)|}",
        "ε概率在K个可选动作中均匀探索；其余概率在当前贪心动作中均分。改变ε会改变经验来源，环境的奖励规则保持不变。"),
      h("details", { class: "optional bandit-supplement" },
        h("summary", {}, "补充算例：用赌博机单独观察探索"),
        prose("先暂时去掉路线和延迟后果，只保留两个未知奖励来源。这是一般决策问题的单步特例，用来隔离奖励噪声、估计和探索；读完回到本章的多步Q控制。UCB和非平稳追踪是选读。"),
        ...c.all())),
    topic("05", "control",
      prose("只预测V不会自动决定动作。把第一步动作也纳入评价，学习Q(s,a)，再用ε-greedy产生行为。SARSA用实际下一动作；Q-learning用下一状态估计最好的动作。两者差别是后续策略。"),
      ...b.all(), ...b.optional()),
    topic("05", "expected",
      equation("y_{\\mathrm{Expected\\ Sarsa}}=r_t+\\gamma\\sum_{a'}\\pi(a'\\mid s_{t+1})Q(s_{t+1},a')",
        "Expected SARSA不采样a′，直接平均策略会选的动作。终止时后续项为0；这里π是要评价的目标策略。"),
      table(["下一状态Q=(5,−5)，策略概率=(0.9,0.1)", "r=0、γ=0.9时的目标"], [
        ["SARSA恰好选了第二动作", "−4.5"],
        ["Expected SARSA按策略平均", "0.9×(0.9×5+0.1×(−5)) = 3.6"],
        ["Q-learning取动作max", "0.9×5 = 4.5"],
      ]),
      prose("随机下一动作给SARSA目标带来波动；Expected SARSA平均这部分随机性。它仍然使用有误差的Q，也仍然采样了环境后果。换成max则改变后续目标策略，不是简单降低方差。",
        "当两个动作真实价值相同，但估计带有噪声，max往往挑中误差偏高的那个；这叫最大化偏差。Double Q把「选动作」与「评价动作」交给两组估计，减少这一偏差。"),
      equation("Q_A(s,a)\\leftarrow Q_A(s,a)+\\alpha\\big[r+\\gamma Q_B(s',\\arg\\max_{a'}Q_A(s',a'))-Q_A(s,a)\\big]",
        "一次Double Q更新：A选下一动作，B评价它；另一部分更新交换A/B。并非保证任何一次估计都不高估。"),
      prose("表格Q-learning的经典收敛条件包括：有限、平稳、折扣任务；有界奖励；所有相关状态动作被持续访问；各对的学习率满足Σα=∞、Σα²<∞。仅跑500回合、固定α或一个种子不能代替这些条件。",
        "SARSA要逼近最优贪心策略还需要在保留充分探索的同时逐渐变得贪心（GLIE）。固定ε的行为仍含探索代价，这正是悬崖例子里两种方法路线不同的物理原因。")),
    chapterFooter("05", "先用一步采样和自举估计V，再把评价对象换成Q参与控制。SARSA、Expected SARSA与Q-learning分别用采样动作、动作平均与动作最大值连接未来；行为策略、目标策略和数据获取方式要分别说明。"));
  return () => { a.cleanup(); b.cleanup(); c.cleanup(); };
}
