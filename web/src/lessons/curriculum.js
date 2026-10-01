// Public curriculum. Individual experiments are catalogued separately.
export const BOOK_TOC = "https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/10094/Toc.pdf?dl=1";
export const THESIS = "https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf";
const definitions = [
  ["01", "强化学习问题与 MDP", "我们的任务是什么？怎样比较并找到最优策略？", "从机器人一步的反馈，到一整条路线的回报，再到最优策略的 Bellman 方程。", "理解完整状态、策略价值与最优性的定义，并从具体路线推导 Bellman 关系。", "不需要 RL 前置知识；能读懂概率、求和与最大值。", "Sutton & Barto §1、§3", "问题与探索", [
    ["task", "MDP、Markov 状态与经验", "先定义环境、动作、奖励和终止；给状态一个足够完整的物理含义。"],
    ["returns", "回报、策略与 V/Q", "把一步反馈变成整段未来的目标，再说明价值依赖哪个策略。"],
    ["optimal", "最优策略与 Bellman 方程", "先比较完整策略，再从具体路线推导一步递推和最优性关系。"],
  ]],
  ["02", "多臂赌博机与探索", "不知道哪个动作好，怎样边试边选？", "反复选择两个奖励来源，观察估计如何改变，又如何决定下一次尝试。", "区分估计、真实均值和不确定性；理解探索的作用。", "第1章：奖励、策略、期望。这里没有影响未来的状态转移。", "Sutton & Barto §2", "问题与探索", [
    ["bandit", "动作价值与增量估计", "先用最简单的单步任务，隔离「未知奖励」这个问题。"],
    ["exploration", "探索、利用与 UCB", "比较只相信当前最好结果和主动尝试尚不了解的动作。"],
    ["nonstationary", "非平稳奖励与上下文", "当奖励会变、不同输入有不同好动作时，估计与策略怎样调整。"],
  ]],
  ["03", "动态规划", "已知 MDP，怎样计算出最优策略？", "数字与箭头交替改变，或把最佳未来一格格传回来。", "区分评价、改进、策略迭代和价值迭代。", "第1章：Bellman 期望/最优方程。第2章帮助区分规划与探索。", "Sutton & Barto §4", "表格方法", [
    ["policy-iteration", "策略评价、改进与策略迭代", "固定箭头计算价值，再按价值改箭头，反复交替。"],
    ["value-iteration", "价值迭代与收敛", "把未知的最优后续暂时替换成旧估计，再反复备份。"],
    ["gpi", "广义策略迭代与更新顺序", "评价与改进可以交错；同步、异步和误差阈值影响计算过程。"],
  ]],
  ["04", "Monte Carlo 方法", "没有模型，能否用完整经历评价和改进策略？", "走完一个回合，沿真实奖励从后往前算每个访问点的回报。", "掌握首访/每访、回报更新、MC 控制和重要性采样。", "第1章：G、Vπ、Qπ；第2章：ε-greedy；第3章：评价与改进。", "Sutton & Barto §5", "表格方法", [
    ["prediction", "完整回报与首访/每访预测", "未来全部来自这个回合的实际奖励，不读取后继价值来补尾巴。"],
    ["control", "动作价值与 MC 控制", "把评价对象从 V 换成 Q，再改善产生经验的策略。"],
    ["importance", "异策略学习与重要性采样", "数据来自 b、目标是 π 时，按动作概率比修正回报的权重。"],
  ]],
  ["05", "TD 学习与表格控制", "能否不等回合结束，就学会更好的动作？", "用真实一步接上后继估计，再看 SARSA 与 Q-learning 如何处理探索的未来。", "掌握 TD 误差、自举、SARSA、Q-learning 与行为/目标策略。", "第4章：MC 目标；第3章：Bellman 备份；第2章：探索。", "Sutton & Barto §6", "表格方法", [
    ["prediction", "TD(0)：一步采样与自举", "把 MC 的完整回报换成一步奖励加当前后继估计。"],
    ["control", "SARSA 与 Q-learning", "在悬崖边区分实际会执行的后续和理想贪心后续。"],
    ["expected", "Expected SARSA、最大化偏差与学习条件", "把采样下一动作换成动作平均，并理解 max 的估计偏差。"],
  ]],
  ["06", "多步学习与资格迹", "未来看几步？一条误差怎样影响过去多个状态？", "一条路线可以截取不同长度，再把这些目标混合；资格迹记住最近经过的位置。", "连接 TD、n-step、MC、λ-return 与 TD(λ)。", "第4、5章：实际回报、自举目标、学习率和终止。", "Sutton & Barto §7；§12 的表格资格迹", "表格方法", [
    ["n-step", "n-step 目标与等待时间", "先读 n 个真实奖励，再从截点接上价值估计。"],
    ["lambda", "λ-return：混合不同长度的未来", "用权重选择短目标与完整回报的影响。"],
    ["traces", "资格迹与 TD(λ)", "把当前 TD 误差沿逐渐衰减的访问记忆传给过去状态。"],
  ]],
  ["07", "模型、规划与 Dyna", "怎样把行动得到的模型，再用于更多学习？", "真实行动留下连接；模型可以重复生成这些连接，帮助奖励传向更远的起点。", "区分真实经验、学习模型、模拟经验和价值更新。", "第3章：规划；第5章：Q-learning；第6章：传播与更新顺序。", "Sutton & Barto §8", "表格方法", [
    ["model", "模型学习与已观测连接", "模型记录环境如何响应动作，价值记录动作的长期好坏。"],
    ["dyna", "Dyna-Q：直接学习与模型规划", "一条真实经验既更新 Q，也更新模型；模拟经验继续更新 Q。"],
    ["search", "模型误差、优先级与搜索", "规划能放大模型的收益，也会传播模型的错误。"],
  ]],
  ["08", "函数近似与 Deep RL", "状态太多放不下表格，怎样表示价值？", "几个状态共享同一组参数，更新一个状态时，其他状态也会改变。", "理解特征、泛化、半梯度、DQN 目标与不稳定性。", "第5章：TD/Q-learning；基础向量、导数与梯度。", "Sutton & Barto §9–11；DQN 原论文", "近似与策略优化", [
    ["features", "从表格到共享参数", "用线性特征实际观察一次更新对不同状态的影响。"],
    ["semigradient", "梯度、半梯度与价值学习", "目标来自真实回报或自举估计；更新时说明对哪些量求导。"],
    ["dqn", "DQN、经验回放与目标网络", "把表格 Q-learning 的目标接到神经网络，并区分稳定化手段与保证。"],
  ]],
  ["09", "策略梯度与 Actor–Critic", "怎样直接改变动作概率，让好路线更常发生？", "在交点给分支分配概率，正优势提高概率，负优势降低概率。", "掌握 REINFORCE、baseline、advantage、actor/critic 与 PPO 的入口。", "第1章：策略与回报；第5章：TD；第8章：可微参数与梯度。", "Sutton & Barto §13；PPO 原论文", "近似与策略优化", [
    ["gradient", "策略参数化与 REINFORCE", "先看概率如何改变，再推导用回报加权的对数概率梯度。"],
    ["actor-critic", "Baseline、优势与 Actor–Critic", "Critic 评价当前未来，actor 根据动作相对基准的好坏改变概率。"],
    ["ppo", "从策略梯度到 PPO", "旧策略数据需要概率比；clipping 限制局部目标的激励。"],
  ]],
  ["10", "前沿课题：离线 RL 与 GCRL", "怎样重组旧经验、改变目标，再连接局部能力？", "先在旧数据交点拼接，再让同一状态和经验服务不同目标。", "把教材中的状态、价值、采样、表示与 actor 接到研究问题。", "第1–9章，尤其 Markov、off-policy、函数近似、策略梯度与规划。", "研究专题：Benjamin Eysenbach 博士论文 §2–4、附录 B", "研究专题", [
    ["offline", "离线 RL、数据覆盖与 stitching", "先分清哪些连接来自数据，哪些只是估计，再计算兼容片段的回报。"],
    ["gcrl", "GCRL：目标条件任务与经验改标", "把 g 加入 π/Q；重算奖励和终止，而不是修改物理转移。"],
    ["future", "未来状态分布与 C-learning", "用折扣未来作为预测对象，把分类与 Bellman 递归连起来。"],
    ["contrastive", "Contrastive RL 与目标条件 actor", "表示内积估计密度比；actor 选择让指定目标更常出现的动作。"],
    ["sorb", "SoRB：把局部能力接成远距离计划", "用目标条件价值建立路标图，再搜索并实际执行局部策略。"],
    ["research", "从演示到研究：还缺哪些证据？", "数据覆盖、模型与表示误差、长距离目标和评估共同决定方法是否有效。"],
  ]],
];
export const CHAPTERS = definitions.map(([id, title, question, image, goal, prerequisite, book, group, topics], i) => ({
  id, number: i + 1, title, short: title, question, image, goal, prerequisite, book, group,
  href: "#chapter-" + id, topics: topics.map(([id, title, description], j) => ({ id, number: (i + 1) + "." + (j + 1), title, description })),
}));
export const CHAPTER_IDS = CHAPTERS.map(c => c.id);
export const chapterById = id => CHAPTERS.find(c => c.id === id);
export const PARTS = [...new Set(CHAPTERS.map(c => c.group))].map(title => ({ title, chapters: CHAPTERS.filter(c => c.group === title) }));
export const LEGACY_ROUTES = {
  "lesson-01": ["01", "task"], "lesson-04": ["01", "returns"], "lesson-05": ["01", "optimal"],
  "lesson-06": ["03", "policy-iteration"], "lesson-02": ["07", "model"],
  "lesson-07": ["05", "prediction"], "lesson-08": ["05", "control"],
  "lesson-03": ["10", "offline"], "lesson-09": ["10", "gcrl"],
};
export function resolveRoute(key) {
  const legacy = LEGACY_ROUTES[key];
  if (legacy) return { chapter: chapterById(legacy[0]), topic: legacy[1] };
  const match = /^chapter-(\d{2})(?:\/([a-z-]+))?$/.exec(key);
  if (!match) return { chapter: null, topic: null };
  const chapter = chapterById(match[1]);
  const topic = chapter?.topics.some(t => t.id === match[2]) ? match[2] : null;
  return { chapter: chapter ?? null, topic };
}
