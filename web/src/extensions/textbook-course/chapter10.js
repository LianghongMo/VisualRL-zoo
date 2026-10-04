import * as offline from "../../lessons/lesson03.js";
import * as goals from "../../lessons/lesson09.js";
import { h } from "../../ui/dom.js";
import { chapterHeader, chapterFooter, topic, capture, prose, equation, table } from "./shell.js";
import { THESIS } from "./curriculum.js";
export function mount(root) {
  const a = capture(offline), b = capture(goals);
  root.append(chapterHeader("10"),
    topic("10", "offline",
      prose("前九章建立了问题定义、采样更新、表示、策略优化和规划。本章把它们用到研究问题：数据已经固定怎么办？任务只用目标描述怎么办？局部控制能力怎样组成更长的计划？",
        "离线/在线描述还能否新增环境经验；on-policy/off-policy描述数据策略与目标策略是否一致。两组概念不是同一维度。"),
      ...a.all({ exclude: ["把整条主线连起来"] }),
      equation("\\mathcal D=\\{(s_i,a_i,r_i,s'_i,d_i)\\}_{i=1}^{N},\\qquad \\text{train on fixed }\\mathcal D",
        "一般离线RL在固定数据上训练，希望部署时得到好策略。数据外动作的高估与部署分布偏移，是观测子图演示之外还要解决的问题。")),
    topic("10", "gcrl", ...b.take("GCRL：把目标也放进问题里", "目标改标：物理经验不变，任务重新计算")),
    topic("10", "future", ...b.take("Ben 的物理图像：预测随机未来时刻的状态", "C-learning：把未来预测变成分类，再递归")),
    topic("10", "contrastive", ...b.take("Contrastive RL：学表示，也学会选动作")),
    topic("10", "sorb", ...b.take("接回 stitching：SoRB 用近目标组成远目标")),
    topic("10", "research",
      table(["教材基础", "怎样接到研究专题"], [
        ["MDP与Markov状态", "拼接匹配完整状态；目标进入任务定义"],
        ["Bellman、TD与off-policy", "改标转移复用；分类递归接目标策略的未来"],
        ["函数近似与表示", "φ(s,a)、ψ(g)压缩可达关系；数据外泛化需要评估"],
        ["策略梯度与actor–critic", "critic给目标相关信号；actor改变到达目标的动作概率"],
        ["规划与模型", "SoRB在路标图上搜索，再调用局部目标策略"],
      ]),
      prose("研究专题不是把更多名词接在Q-learning后面。它们改变了预测对象、监督来源或计算组织方式：固定奖励变成一族目标任务；价值预测变成未来状态分布；单个局部动作规则变成路标计划。",
        "本章精确地图、Bayes分类与表格改标更新，让公式能逐项验算。要声称训练了神经网络GCRL，还要真正训练表示、critic与actor，报告不同数据覆盖、目标距离和随机种子的评估。"),
      table(["研究问题", "最小可检验的实验"], [
        ["数据覆盖不足时，策略会不会追逐虚高价值？", "删除关键连接，比较估值与真实执行成功率"],
        ["表示是否反映可达关系？", "比较墙两侧与沿可行通路的状态，而非只看坐标距离"],
        ["改标是否正确保留任务？", "检查每条转移的奖励、成功终止与后缀处理"],
        ["路标估距离不准时能否走到目标？", "扰动距离、移除路标，记录计划失败与执行失败"],
      ]),
      ...b.take("论文怎么继续读？", "检查理解"),
      h("p", { class: "source-note" }, "专题依据：", h("a", { href: THESIS, target: "_blank", rel: "noopener" }, "Ben Eysenbach · 2023博士论文"), "。这是研究阅读入口，不把它称为所有前沿方向的完整综述。")),
    chapterFooter("10", "离线RL先检查数据覆盖与状态兼容；GCRL把目标加进任务、策略和价值；未来预测、分类与对比学习给critic新的表示与监督；actor选择动作，SoRB组合局部能力。每个研究结论都应回到定义、假设和真实执行证据。"));
  return () => { a.cleanup(); b.cleanup(); };
}
