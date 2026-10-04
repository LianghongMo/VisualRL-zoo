// Public curriculum. Individual experiments are catalogued separately.
export const BOOK_TOC = "https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/10094/Toc.pdf?dl=1";
export const THESIS = "https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf";
const definitions = [
  ["01", "强化学习任务与 MDP", "机器人在什么任务中行动，怎样描述一步反馈？", "从一次移动、奖励与终止，建立状态、动作、环境和策略的物理含义。", "能完整定义任务，检验Markov状态，并区分策略、模型和经验。", "不需要RL前置知识。", "Sutton & Barto §1、§3：问题定义", "任务、价值与规划", [
    ["task", "任务与 MDP 的组成", "操作机器人一步，再定义状态、动作、奖励、转移与终止。"],
    ["state", "Markov 状态：哪些信息必须保留？", "同一位置不一定是同一状态；检查电量、速度等条件是否影响后果。"],
    ["interaction", "策略、经验与交互循环", "分清环境怎样反馈、策略怎样选动作，以及一条经验记录了什么。"],
  ]],
  ["02", "回报、价值与最优策略", "怎样用同一把尺子比较不同策略？", "同样的起点与两座充电站，不同路线、折扣和完整策略产生不同长期结果。", "从r到G，再到Vπ/Qπ；用价值定义策略比较和最优控制目标。", "第1章：任务、策略、经验与终止。", "Sutton & Barto §3：回报、价值与最优策略", "任务、价值与规划", [
    ["returns", "从奖励到回报", "用具体路线计算G，说明折扣改变比较尺度而不改变环境奖励。"],
    ["values", "固定策略的 V 和 Q", "一条回报样本与期望价值不同；说明未来按哪个策略走。"],
    ["optimal", "比较策略与定义最优", "先执行A/B/C，再定义V*与π*；区分优于几个候选与全局最优。"],
    ["shaping", "应用工具：奖励塑形与望远镜求和", "哪些辅助奖励保持策略比较？用边界抵消推导，并检查错误终点设置。"],
  ]],
  ["03", "Bellman 方程与动态规划", "能否把整条未来拆成一步关系，并据此求解？", "终点前一格、前两格与交点J解释一步递推；数字与箭头的迭代把最优路线求出来。", "从实例推导Bellman，再用已知模型做评价、改进、策略迭代和价值迭代。", "第2章：G、Vπ/Qπ与π*；第1章：完整Markov状态和模型。", "Sutton & Barto §3：Bellman；§4：动态规划", "任务、价值与规划", [
    ["bellman", "从路线图像到 Bellman 方程", "先接一步奖励与后续价值，再区分策略平均、环境平均与动作max。"],
    ["policy-iteration", "策略评价、改进与策略迭代", "固定箭头算价值，再按价值改箭头，反复交替。"],
    ["value-iteration", "价值迭代与收敛", "未知最优后续暂用旧估计；终点信息一格格传回起点。"],
    ["gpi", "广义策略迭代与更新顺序", "评价与改进可以交错；为下一章用经验替代模型计算做准备。"],
  ]],
  ["04", "Monte Carlo 方法", "没有模型，能否用完整经历评价和改进策略？", "走完一个回合，沿真实奖励从后往前算每个访问点的回报。", "掌握首访/每访、回报更新、MC 控制和重要性采样。", "第2章：G、Vπ/Qπ；第3章：评价与改进。探索规则在本章控制小节引入。", "Sutton & Barto §5", "从经验学习与规划", [
    ["prediction", "完整回报与首访/每访预测", "未来全部来自这个回合的实际奖励，不读取后继价值来补尾巴。"],
    ["control", "动作价值与 MC 控制", "把评价对象从 V 换成 Q，再改善产生经验的策略。"],
    ["importance", "异策略学习与重要性采样", "数据来自 b、目标是 π 时，按动作概率比修正回报的权重。"],
  ]],
  ["05", "TD 学习与表格控制", "能否不等回合结束，就学会更好的动作？", "用真实一步接上后继估计，再看 SARSA 与 Q-learning 如何处理探索的未来。", "掌握 TD 误差、自举、SARSA、Q-learning 与行为/目标策略。", "第4章：完整回报与MC控制；第3章：Bellman关系。", "Sutton & Barto §6；探索补充 §2", "从经验学习与规划", [
    ["prediction", "TD(0)：一步采样与自举", "把 MC 的完整回报换成一步奖励加当前后继估计。"],
    ["exploration", "探索：怎样产生有用经验？", "把ε-greedy接到路线选择；赌博机作为补充例子，隔离未知动作收益。"],
    ["control", "SARSA 与 Q-learning", "在悬崖边区分实际会执行的后续和理想贪心后续。"],
    ["expected", "Expected SARSA、最大化偏差与学习条件", "把采样下一动作换成动作平均，并理解 max 的估计偏差。"],
  ]],
  ["06", "多步学习与资格迹", "未来看几步？一条误差怎样影响过去多个状态？", "一条路线可以截取不同长度，再把这些目标混合；资格迹记住最近经过的位置。", "连接 TD、n-step、MC、λ-return 与 TD(λ)。", "第4、5章：实际回报、自举目标、学习率和终止。", "Sutton & Barto §7；§12 的表格资格迹", "从经验学习与规划", [
    ["n-step", "n-step 目标与等待时间", "先读 n 个真实奖励，再从截点接上价值估计。"],
    ["lambda", "λ-return：混合不同长度的未来", "用权重选择短目标与完整回报的影响。"],
    ["traces", "资格迹与 TD(λ)", "把当前 TD 误差沿逐渐衰减的访问记忆传给过去状态。"],
  ]],
  ["07", "模型、规划与 Dyna", "怎样把行动得到的模型，再用于更多学习？", "真实行动留下连接；模型可以重复生成这些连接，帮助奖励传向更远的起点。", "区分真实经验、学习模型、模拟经验和价值更新。", "第3章：规划；第5章：Q-learning；第6章：传播与更新顺序。", "Sutton & Barto §8", "从经验学习与规划", [
    ["model", "模型学习与已观测连接", "模型记录环境如何响应动作，价值记录动作的长期好坏。"],
    ["dyna", "Dyna-Q：直接学习与模型规划", "一条真实经验既更新 Q，也更新模型；模拟经验继续更新 Q。"],
    ["search", "模型误差、优先级与搜索", "规划能放大模型的收益，也会传播模型的错误。"],
  ]],
  ["08", "函数近似与 Deep RL", "状态太多放不下表格，怎样表示价值？", "几个状态共享同一组参数，更新一个状态时，其他状态也会改变。", "理解特征、泛化、半梯度、DQN 目标与不稳定性。", "第5章：TD/Q-learning；基础向量、导数与梯度。", "Sutton & Barto §9–11；DQN 原论文", "近似与策略优化", [
    ["features", "从表格到共享参数", "用线性特征实际观察一次更新对不同状态的影响。"],
    ["semigradient", "梯度、半梯度与价值学习", "目标来自真实回报或自举估计；更新时说明对哪些量求导。"],
    ["dqn", "DQN、经验回放与目标网络", "把表格 Q-learning 的目标接到神经网络，并区分稳定化手段与保证。"],
    ["stability", "Jensen、Double DQN 与训练技巧", "解释max偏差，区分回放、目标、损失、奖励和优化步的改变。"],
  ]],
  ["09", "策略梯度、Actor–Critic 与 PPO", "怎样推导采样梯度，再用可靠的目标优化策略？", "从交点的动作概率出发，推导为什么回报、baseline、概率比与熵能影响更新。", "能推导PG，解释GAE与surrogate，验算PPO和Jensen，并区分恒等式、近似与实践技巧。", "第2章：回报/价值；第4章：重要性采样与覆盖；第5、6章：TD/多步；第8章：梯度。", "Sutton & Barto §13；COS435 第4–6周公开阅读", "近似与策略优化", [
    ["gradient", "先看图像：策略概率怎样改变？", "用两条路线观察一次条件样本更新，再提出推导的问题。"],
    ["derivation", "从轨迹概率推导 Policy Gradient", "log-derivative、因果性、条件期望与折扣占用逐步连接。"],
    ["actor-critic", "Baseline、Actor–Critic 与 GAE", "均值和方差分开看；多步优势用望远镜求和，终止与截断用两个mask。"],
    ["surrogate", "从真实回报到 Surrogate / TRPO", "分清性能差恒等式、旧状态访问近似和动作概率比的精确作用。"],
    ["ppo", "PPO：推导局部目标，验算 Clipping", "按优势符号拆开min，并说明平台、KL和真实回报之间的边界。"],
    ["entropy", "数学工具：Jensen、熵、KL 与 Softmax", "用凹曲线看Jensen，再把奖励＋熵写成soft价值下界。"],
    ["practice", "常用训练技巧与验收", "旧log概率、detach、优势标准化、KL监测和公平实验各解决什么。"],
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
const RELATIONS = {
  "01": { from: "从机器人任务出发：先说清状态、可选动作和环境反馈，后面的计算才有对象。", result: "得到完整MDP、策略与经验的定义。", next: "知道发生了什么，还不能比较长期好坏。第2章把奖励组织成回报与价值。" },
  "02": { from: "第1章定义了一条经历怎样产生；本章问这条经历、这个策略究竟有多好。", result: "得到G、Vπ/Qπ与最优策略目标，并理解势函数塑形保持比较结果的条件。", next: "逐个展开完整策略太费计算。第3章复用后续价值，推导Bellman并求解。" },
  "03": { from: "第2章已定义价值与最优性，但尚未给出高效计算的方法。", result: "得到一步递推关系，以及已知模型时的DP求解方法。", next: "真实任务常不知道完整模型。第4章用实际完整回合替代模型期望。" },
  "04": { from: "第3章计算需要所有可能后果；现在只给实际走过的回合，仍要评价与改进策略。", result: "用完整回报估计V/Q，再通过探索和策略改进形成MC控制。", next: "MC需要等到回合结束。第5章只看真实一步，再接后继估计，提前学习。" },
  "05": { from: "第4章提供完整未来的回报样本；本章把未知余下未来替换成当前价值估计。", result: "得到TD预测、探索行为及SARSA/Q-learning等一步控制方法。", next: "一步TD与完整MC是两个端点。第6章改变观察长度，并把误差分给多个过去状态。" },
  "06": { from: "已经理解完整回报与一步自举，现在需要在等待时间、采样波动和传播之间选择。", result: "用n-step、λ-return和资格迹连接短目标与长目标。", next: "经验不仅能改价值，还能学到环境模型。第7章让模型生成更多学习材料。" },
  "07": { from: "第3章使用已知模型，第4–6章直接从经验学习；本章把两条路径接起来。", result: "真实经验更新Q和模型；Dyna重复使用模拟经验进行规划。", next: "这些方法仍给每个状态动作存表。第8章在状态太多时，用共享参数表示价值。" },
  "08": { from: "前面的表格计算已经清楚，但大状态空间无法逐项存储和访问。", result: "用特征与梯度共享估计，理解半梯度、DQN/Double DQN及稳定训练技巧的作用。", next: "价值方法仍通过Q间接选动作。第9章直接给策略概率参数，并优化它。" },
  "09": { from: "第8章学会参数化价值；本章把可微参数用于动作概率，并结合第4章采样与第5、6章多步估计。", result: "推导PG，区分baseline与GAE、真实J与surrogate、TRPO与PPO，并用Jensen/KL解释熵目标。", next: "基础方法已齐。第10章改变数据来源与任务目标，研究离线RL、stitching和GCRL。" },
  "10": { from: "把完整状态、价值递推、经验学习、表示、策略优化与规划带入论文问题。", result: "理解固定数据与目标条件任务怎样复用基础，以及方法依赖哪些假设。", next: "继续论文与实验时，逐项检查任务定义、数据覆盖、表示误差和真实执行结果。" },
};
export const CHAPTERS = definitions.map(([id, title, question, image, goal, prerequisite, book, group, topics], i) => ({
  id, number: i + 1, title, short: title, question, image, goal, prerequisite, book, group, relation: RELATIONS[id],
  href: "#chapter-" + id, topics: topics.map(([id, title, description], j) => ({ id, number: (i + 1) + "." + (j + 1), title, description })),
}));
export const CHAPTER_IDS = CHAPTERS.map(c => c.id);
export const chapterById = id => CHAPTERS.find(c => c.id === id);
export const PARTS = [...new Set(CHAPTERS.map(c => c.group))].map(title => ({ title, chapters: CHAPTERS.filter(c => c.group === title) }));
export const LEGACY_ROUTES = {
  "lesson-01": ["01", "task"], "lesson-04": ["02", "returns"], "lesson-05": ["02", "optimal"],
  "lesson-06": ["03", "policy-iteration"], "lesson-02": ["07", "model"],
  "lesson-07": ["05", "prediction"], "lesson-08": ["05", "control"],
  "lesson-03": ["10", "offline"], "lesson-09": ["10", "gcrl"],
};
export const CHAPTER_ROUTE_ALIASES = {
  "chapter-01/returns": ["02", "returns"], "chapter-01/optimal": ["02", "optimal"],
  "chapter-02/bandit": ["05", "exploration"], "chapter-02/exploration": ["05", "exploration"], "chapter-02/nonstationary": ["05", "exploration"],
};
export function resolveRoute(key) {
  const alias = LEGACY_ROUTES[key] ?? CHAPTER_ROUTE_ALIASES[key];
  if (alias) return { chapter: chapterById(alias[0]), topic: alias[1] };
  const match = /^chapter-(\d{2})(?:\/([a-z-]+))?$/.exec(key);
  if (!match) return { chapter: null, topic: null };
  const chapter = chapterById(match[1]);
  const topic = chapter?.topics.some(t => t.id === match[2]) ? match[2] : null;
  return { chapter: chapter ?? null, topic };
}
