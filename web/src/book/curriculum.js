export const BOOK = "https://rltheorybook.github.io/rltheorybook_ABJKS.pdf";
export const HARVARD = "https://harvard-cs2824-s26.github.io/";
export const THESIS = "https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf";
const definitions = [
  ["01", "MDP、价值与规划", "先把控制任务说清楚，再用一步关系求最优策略。", "小车要停稳，眼前靠近目标并不等于整条轨迹更好。", "定义完整状态、V/Q与最优目标；从算例得到Bellman，区分评价与改进。", "概率期望；不需要RL前置。", "ABJKS 第1章 · 压缩基础", 3, "I · 任务、数据与表示", [
    ["task", "先定义任务：位置、速度与推力", "从真实一步反馈理解MDP；同位置不同速度需要不同动作。"],
    ["values", "评价整条未来，再寻找最优策略", "先比较轨迹与累计代价，再定义G、V、Q和π*。"],
    ["bellman", "从两步算例推导Bellman", "复用后续最优代价；策略平均、环境平均与动作优化各做什么。"],
    ["planning", "已知模型：价值迭代与策略迭代", "先固定规则评价，再改规则；有限时域和无限折扣分开。"],
  ]],
  ["02", "数据模型与采样学习", "模型未知时，什么数据允许我们回答什么问题？", "能重置模拟器到任意状态，和只能沿一条真实轨迹采样，是不同能力。", "分清generative、offline、online；理解误差传播，以及MC、TD和Q-learning的基本更新。", "第1章的Bellman、V/Q与期望。", "ABJKS 第2章；采样更新补充", 21, "I · 任务、数据与表示", [
    ["access", "三种数据接口，三种能力", "查询任意状态动作、读取固定日志、沿环境交互，不能混为一谈。"],
    ["statistics", "样本均值与规划误差", "采样波动随样本减少，长时域会放大模型和价值误差。"],
    ["updates", "从完整回报到一步自举", "用一组方程连接MC、TD、SARSA和Q-learning；作为工具而非四个独立大章。"],
  ]],
  ["03", "函数近似与离线覆盖", "连续空间放不下表格，有限数据怎样支持可靠决策？", "一张价值曲面代替无数格子；曲面拟合正确和数据覆盖充分是两件事。", "理解特征、realizability、Bellman completeness、FQI和分布覆盖；用失败例子识别外推风险。", "第2章的数据接口与自举。", "ABJKS 第3–4章 · 表示与覆盖", 35, "I · 任务、数据与表示", [
    ["features", "用共享函数表示连续价值", "比较线性与二次特征，观察拟合误差和未见区域。"],
    ["closure", "为什么能拟合Q*仍然不够？", "最优值可表示、全策略可表示、Bellman闭包是不同假设。"],
    ["offline", "Fitted Q-Iteration与覆盖", "固定旧目标、回归新Q；不能靠重复训练补出未观察结果。"],
    ["stitching", "Stitching：怎样复用兼容片段", "复用已知转移的后续价值；位置与速度都必须匹配。"],
  ]],
  ["04", "探索与可学习结构", "机器人该去哪采数据，才能减少真正影响决策的不确定性？", "噪声探索、策略性访问和结构泛化解决不同问题。", "掌握乐观估计、探索的长期性与结构假设；把复杂度证明放回选读。", "第2–3章：数据与表示误差。", "ABJKS 第5–8章 · 精选与压缩", 63, "II · 怎样取得有用数据", [
    ["bandit", "先隔离未知收益：bandit与UCB", "只用一个小节理解置信区间；机器人还要考虑状态变化。"],
    ["strategic", "探索要把机器人带到有用的状态", "一步动作噪声未必发现远处任务；乐观价值沿未来传回来。"],
    ["structure", "Linear MDP与Bellman rank在假设什么", "结构使大状态空间可学习；转移特征线性不是机械动力学线性。"],
  ]],
  ["05", "策略梯度：从轨迹到连续动作", "怎样直接改进一条输出连续推力的策略？", "把推力分布向产生更好未来的样本移动，而不是枚举所有推力。", "逐步推导PG；解释Gaussian score、baseline、Actor–Critic与GAE。", "第1章目标、第2章采样、第3章梯度与函数表示。", "ABJKS 第9章；Gaussian与GAE补充", 111, "III · 直接优化策略", [
    ["gaussian", "连续策略：均值、方差与真实动作", "先看推力概率密度，再做一次可验算的策略更新。"],
    ["derivation", "轨迹概率、log-derivative与因果性", "每个等号给出条件，保留初始折扣目标的外层γᵗ。"],
    ["baseline", "Baseline改变波动，不改变期望", "从零期望引理理解控制变量，比较均值与方差。"],
    ["actor-critic", "Critic与GAE估计未来优势", "多步目标、detach与终止/截断双mask连到训练。"],
  ]],
  ["06", "最优性、自然梯度与近似", "梯度很小为什么未必已经学好？参数距离怎样变成行为距离？", "稀少访问的好状态与几乎不采的好动作，都可能让策略难以改善。", "区分stationary与optimal；用Fisher/KL理解NPG与兼容优势近似。", "第5章PG；矩阵、梯度与概率密度。", "ABJKS 第10–11章", 119, "III · 直接优化策略", [
    ["optimality", "梯度、覆盖与最优性的边界", "小梯度不是任意参数化策略的全局最优证书。"],
    ["geometry", "自然梯度：限制分布而非参数", "Gaussian的相同均值步长在不同方差下有不同KL。"],
    ["compatible", "近似优势怎样接到Fisher方向", "用score作特征，回归优势得到兼容近似；实际网络还有误差。"],
  ]],
  ["07", "保守策略优化：TRPO与PPO", "同一批旧轨迹能用多久？怎样控制一次策略变化？", "新策略会访问不同状态；旧数据上的改善必须和真实执行分开。", "从性能差到surrogate，再到TRPO/PPO；理解clipping、KL诊断和固定训练目标。", "第5章优势、第6章KL/NPG。", "ABJKS 第12章；PPO与熵实践补充", 141, "III · 直接优化策略", [
    ["surrogate", "真实性能差与旧状态surrogate", "动作概率比精确修正动作期望，不能自动修正状态访问。"],
    ["trust-region", "CPI、TRPO与平均KL的限度", "保守混合与局部信赖域的来由；罕见状态可能变化很大。"],
    ["ppo", "PPO：逐符号推导clipped目标", "好方向进入平台，坏方向保留斜率；不把clip当硬约束。"],
    ["entropy", "Jensen、熵与soft目标", "插入概率、利用凹log，得到softmax与温度×KL差距。"],
    ["practice", "一次完整更新的变量与验收", "固定旧log概率、优势和critic目标；重新采样与独立评估。"],
  ]],
  ["08", "模仿学习：从示范到反馈", "机器人能否先学会专家怎么做，再处理自己造成的偏差？", "专家只在正常轨迹上示范，学习者偏离后会遇到新的状态。", "写出连续动作BC损失，理解分布偏移、交互数据与奖励/动作监督的区别。", "第3章覆盖；第5章连续策略。", "ABJKS 第13章 · 机器人选读", 161, "IV · 从理论走向机器人", [
    ["bc", "Behavior Cloning学的是哪种监督", "回归专家推力，不直接最大化环境累计回报。"],
    ["shift", "闭环误差与状态分布偏移", "切换质量或加外力，比较同轨迹回放与反馈控制。"],
    ["interactive", "交互模仿：给新状态补上专家反馈", "数据聚合与专家成本到达信息各需要什么访问条件。"],
    ["reward", "逆强化学习与任务目标", "示范动作不唯一确定奖励；最大熵是一个建模选择。"],
  ]],
  ["09", "连续控制与LQR", "Bellman怎样真的给出连续推力，让机器人到达并停稳？", "位置与速度构成相平面，反馈推力把状态拉向目标。", "从动力学与二次代价推导Riccati和线性反馈；验证轨迹、能耗、稳定性及限幅边界。", "第1章Bellman、第3章二次表示、第6章矩阵与优化。", "ABJKS 第14章；原创小车实验", 173, "IV · 从理论走向机器人", [
    ["dynamics", "双积分器：位置、速度、推力", "连续的是状态和动作；本实验按固定时间间隔决策。"],
    ["objective", "停稳与省力：二次代价", "状态误差、控制消耗和末端条件共同定义任务。"],
    ["riccati", "从Bellman完成平方，得到Riccati", "二次价值闭包把连续动作优化变成矩阵计算。"],
    ["feedback", "动手控制：相平面、推力与累计代价", "调目标、初速、能耗权重；一步执行和完整轨迹都是真实计算。"],
    ["limits", "有限时域、稳定性与约束", "有限最优不等于渐近稳定；推力限幅和模型误差改变原问题。"],
  ]],
  ["10", "机器人延伸：训练流程与GCRL", "把推导变成连续机器人任务，还必须定义哪些接口与证据？", "策略看到状态和目标，输出有限推力；训练、评估与泛化分别验收。", "定义连续观测/动作、Gaussian变换、PPO/SAC位置、目标条件任务与改标；给出可复现实验标准。", "第5–9章；覆盖、连续策略和反馈控制。", "机器人补充；Ben Eysenbach论文 §2–4", null, "IV · 从理论走向机器人", [
    ["interface", "先把机器人接口与任务写完整", "观测、动力学、动作尺度、代价、时限和成功条件逐项落地。"],
    ["algorithms", "连续PPO与SAC各优化什么", "有界概率密度、on/off-policy、目标网络与熵项有清楚位置。"],
    ["gcrl", "目标条件策略与连续目标", "同一物理状态，目标不同需要不同反馈；用位置和速度检验目标。"],
    ["relabel", "HER、stitching与未来状态表示", "保留物理转移、重算任务标签；比较占用与首次到达奖励。"],
    ["evaluation", "从小车算例到真实机器人证据", "同预算、多种子、扰动、不同目标与真实执行共同决定是否有效。"],
  ]],
];
const bridge = [
  ["先从一个可实际执行的小车任务定义对象。", "得到MDP、最优目标、Bellman与模型规划。", "模型通常未知；下一章明确什么数据接口允许怎样学习。"],
  ["上一章使用完整模型；现在只拿到样本或日志。", "区分数据接口、统计误差与采样自举。", "连续空间不能按状态存表；下一章问函数表示与数据覆盖是否足够。"],
  ["样本有限，预测必须共享参数；共享也会外推。", "分清表示、闭包、覆盖和兼容数据复用。", "离线日志不一定覆盖关键状态；下一章研究怎样主动获得有用经验。"],
  ["已经知道缺了什么数据，现在决定怎样补上。", "认识探索、乐观估计及结构假设的作用。", "连续推力无法逐个枚举；下一章直接优化可微概率策略。"],
  ["从价值间接选动作，转向输出连续动作分布。", "得到PG推导、Gaussian score、baseline与GAE。", "采样梯度仍可能难优化；下一章研究覆盖、最优性与行为几何。"],
  ["会求梯度后，还要问梯度是否有用、一步改变了多少行为。", "区分最优性条件、KL/Fisher几何与兼容近似。", "实际训练重复使用旧批次；下一章用保守更新处理分布变化。"],
  ["用KL理解了行为变化，现在处理旧数据上的实际优化。", "得到surrogate、TRPO/PPO、熵目标与训练诊断。", "机器人采样昂贵，示范常能提供初始能力；下一章研究模仿与闭环分布偏移。"],
  ["策略优化需要试错；示范提供另一种监督来源。", "理解BC与交互模仿的数据要求和反馈问题。", "要验算连续控制是否真的正确，下一章用已知模型的LQR作精确基线。"],
  ["把前面的状态、价值、表示和优化落到可解析的连续机器人。", "Riccati给出反馈推力，轨迹和代价能逐项核对。", "真实机器人还有边界、未知模型与多目标；最后一章整理训练接口并连接GCRL。"],
  ["在LQR基线之上，明确连续策略和目标任务的完整训练协议。", "能设计机器人实验并区分教学算例、训练结果与泛化证据。", "继续读论文时，检查任务、数据、优化和闭环执行；用相同协议比较方法。"],
];
export const CHAPTERS = definitions.map(([id,title,question,image,goal,prerequisite,book,page,group,topics],i)=>({
  id, number:i+1, title, short:title, question, image, goal, prerequisite, book, group,
  source:page ? BOOK+"#page="+(page+12) : THESIS,
  href:"#chapter-"+id, relation:{from:bridge[i][0],result:bridge[i][1],next:bridge[i][2]},
  topics:topics.map(([id,title,description],j)=>({id,number:(i+1)+"."+(j+1),title,description})),
}));
export const CHAPTER_IDS=CHAPTERS.map(c=>c.id);
export const chapterById=id=>CHAPTERS.find(c=>c.id===id);
export const PARTS=[...new Set(CHAPTERS.map(c=>c.group))].map(title=>({title,chapters:CHAPTERS.filter(c=>c.group===title)}));
export const BOOK_MAP=[
  ["1 · MDP", "第1章", "任务、价值、Bellman和DP；线性规划与计算复杂度证明选读。"],
  ["2 · Generative model", "第2章", "数据访问、样本误差与规划；复杂度定理及证明选读。"],
  ["3–4 · Features / Offline", "第3章", "表示、Bellman闭包、FQI与覆盖；保留失败条件。"],
  ["5–6 · Bandits / Exploration", "第4章", "压缩UCB与策略性探索；不展开整套regret证明。"],
  ["7–8 · Linear MDP / Bellman rank", "第4章后半", "解释结构假设与机器人线性动力学的区别；证明选读。"],
  ["9 · Policy gradient", "第5章", "完整推导与优势估计；连续Gaussian是机器人补充。"],
  ["10–11 · Optimality / NPG", "第6章", "最优性边界、Fisher与兼容近似；全局速率证明选读。"],
  ["12 · Conservative optimization", "第7章", "CPI→TRPO→PPO，解释平均KL和clip的局限。"],
  ["13 · Imitation", "第8章", "BC、交互数据与闭环分布偏移；IRL简要连接。"],
  ["14 · LQR", "第9章", "双积分器、Riccati、反馈与稳定性；SDP/SLS选读。"],
  ["机器人与GCRL补充", "第10章", "连续接口、有界策略、SAC定位、目标条件学习；独立来源。"],
];
// Old links name concepts, and resolve to their new location after the rewrite.
export const LEGACY_ROUTES={
  "lesson-01":["01","task"],"lesson-04":["01","values"],"lesson-05":["01","values"],
  "lesson-06":["01","planning"],"lesson-02":["02","access"],"lesson-07":["02","updates"],
  "lesson-08":["02","updates"],"lesson-03":["03","stitching"],"lesson-09":["10","gcrl"],
};
export const ALIASES={
  "chapter-01/state":["01","task"],"chapter-01/interaction":["01","task"],
  "chapter-01/returns":["01","values"],"chapter-01/optimal":["01","values"],
  "chapter-02/returns":["01","values"],"chapter-02/values":["01","values"],"chapter-02/optimal":["01","values"],
  "chapter-02/shaping":["01","values"],"chapter-02/bandit":["04","bandit"],"chapter-02/exploration":["04","strategic"],"chapter-02/nonstationary":["04","bandit"],
  "chapter-03/bellman":["01","bellman"],"chapter-03/policy-iteration":["01","planning"],"chapter-03/value-iteration":["01","planning"],"chapter-03/gpi":["01","planning"],
  "chapter-04/prediction":["02","updates"],"chapter-04/control":["02","updates"],"chapter-04/importance":["07","surrogate"],
  "chapter-05/prediction":["02","updates"],"chapter-05/exploration":["04","strategic"],"chapter-05/control":["02","updates"],"chapter-05/expected":["02","updates"],
  "chapter-06/n-step":["05","actor-critic"],"chapter-06/lambda":["05","actor-critic"],"chapter-06/traces":["05","actor-critic"],
  "chapter-07/model":["02","access"],"chapter-07/dyna":["02","access"],"chapter-07/search":["04","strategic"],
  "chapter-08/features":["03","features"],"chapter-08/semigradient":["03","offline"],"chapter-08/dqn":["03","offline"],"chapter-08/stability":["03","offline"],
  "chapter-09/gradient":["05","gaussian"],"chapter-09/derivation":["05","derivation"],"chapter-09/actor-critic":["05","actor-critic"],
  "chapter-09/surrogate":["07","surrogate"],"chapter-09/ppo":["07","ppo"],"chapter-09/entropy":["07","entropy"],"chapter-09/practice":["07","practice"],
  "chapter-10/offline":["03","offline"],"chapter-10/future":["10","relabel"],"chapter-10/contrastive":["10","relabel"],
  "chapter-10/sorb":["10","gcrl"],"chapter-10/research":["10","evaluation"],
};
export function resolveRoute(key){
  const alias=LEGACY_ROUTES[key]??ALIASES[key];
  if(alias)return{chapter:chapterById(alias[0]),topic:alias[1]};
  const match=/^chapter-(\d{2})(?:\/([a-z-]+))?$/.exec(key);
  const chapter=match&&chapterById(match[1]);
  return{chapter:chapter??null,topic:chapter?.topics.some(t=>t.id===match[2])?match[2]:null};
}
