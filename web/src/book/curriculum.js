export const BOOK = "https://rltheorybook.github.io/rltheorybook_ABJKS.pdf";
export const HARVARD = "https://harvard-cs2824-s26.github.io/";
export const THESIS = "https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf";
const definitions = [
  ["01", "MDP、价值与规划", "怎样让小车到达并停稳？", "好策略既要驶向目标，也要及时刹车。", "定义MDP和V/Q，用Bellman求最优策略。", "概率与期望。", "ABJKS 第1章", 3, "I · 任务、数据与表示", [
    ["task", "任务、状态与动作", "用位置和速度预测推力的后果。"],
    ["values", "回报、价值与最优策略", "用G评价轨迹，用V/Q评价未来。"],
    ["bellman", "从return分解推导Bellman", "推导策略评价与最优递推，再连接Q-learning。"],
    ["planning", "价值迭代与策略迭代", "固定策略做评价，再用价值改进策略。"],
  ]],
  ["02", "数据模型与采样学习", "模型未知，怎样从样本估计价值？", "模拟器可以重置；真实小车需要开到要采样的状态。", "区分数据接口，写出MC、TD、SARSA和Q-learning更新。", "第1章。", "ABJKS 第2章；采样更新补充", 21, "I · 任务、数据与表示", [
    ["access", "三种数据接口", "任意查询、固定日志、在线交互。"],
    ["statistics", "样本均值与规划误差", "增加样本减少波动，长时域放大误差。"],
    ["updates", "MC、TD、SARSA与Q-learning", "比较完整回报与一步自举目标。"],
  ]],
  ["03", "函数近似与离线覆盖", "怎样用有限数据表示连续价值？", "一个价值函数能预测许多状态，未见区域却可能预测错误。", "理解特征、Bellman闭包、FQI、覆盖与stitching。", "第2章。", "ABJKS 第3–4章", 35, "I · 任务、数据与表示", [
    ["features", "用函数表示连续价值", "比较线性与二次特征。"],
    ["closure", "为什么能拟合Q*仍然不够？", "最优值可表示、全策略可表示、Bellman闭包是不同假设。"],
    ["offline", "Fitted Q-Iteration与覆盖", "固定目标回归Q，检查数据覆盖。"],
    ["stitching", "Stitching：连接轨迹片段", "在相同状态连接片段，复用后续价值。"],
  ]],
  ["04", "探索与可学习结构", "怎样采到能改善策略的数据？", "想学会抬起物体，先要能到达接触状态。", "用UCB理解乐观探索，辨认Linear MDP与Bellman rank假设。", "第2–3章。", "ABJKS 第5–8章 · 精选", 63, "II · 怎样取得有用数据", [
    ["bandit", "Bandit与UCB", "用均值和不确定性选择动作。"],
    ["strategic", "探索有用的状态", "为取得远处的反馈规划整条轨迹。"],
    ["structure", "Linear MDP与Bellman rank", "区分转移分布的结构与机械动力学。"],
  ]],
  ["05", "策略梯度：从轨迹到连续动作", "怎样直接优化连续动作策略？", "提高好动作的概率，让推力分布向更高回报移动。", "推导PG，用baseline与GAE估计优势。", "第1–3章；微积分。", "ABJKS 第9章；Gaussian与GAE补充", 111, "III · 直接优化策略", [
    ["gaussian", "Gaussian连续策略", "观察推力分布，计算一次更新。"],
    ["derivation", "策略梯度推导", "从轨迹概率得到reward-to-go。"],
    ["baseline", "Baseline与方差", "减去动作无关baseline，保持梯度期望。"],
    ["actor-critic", "Actor–Critic与GAE", "用critic估计动作优势。"],
  ]],
  ["06", "最优性、自然梯度与近似", "怎样衡量一次更新改变了多少行为？", "方差越小，同样的均值变化越容易改变动作分布。", "区分驻点与最优策略，推导Fisher自然梯度与兼容近似。", "第5章；矩阵运算。", "ABJKS 第10–11章", 119, "III · 直接优化策略", [
    ["optimality", "梯度与最优性", "检查小梯度背后的状态与动作覆盖。"],
    ["geometry", "自然梯度与KL", "用分布变化衡量更新步长。"],
    ["compatible", "兼容优势近似", "用score回归优势，得到Fisher方向。"],
  ]],
  ["07", "保守策略优化：TRPO与PPO", "怎样用旧轨迹更新策略？", "策略变了，访问的状态也会变；更新幅度需要控制。", "推导surrogate、TRPO与PPO，理解Jensen、熵和更新流程。", "第5–6章。", "ABJKS 第12章；PPO与熵补充", 141, "III · 直接优化策略", [
    ["surrogate", "性能差与surrogate", "比较新状态分布与旧状态分布。"],
    ["trust-region", "CPI与TRPO", "用保守混合或KL约束控制更新。"],
    ["ppo", "PPO的clipped目标", "按优势符号分析截断方向。"],
    ["entropy", "Jensen、熵与soft目标", "推导softmax和温度×KL差距。"],
    ["practice", "一次PPO更新", "采样、固定目标、更新、评估。"],
  ]],
  ["08", "模仿学习：从示范到反馈", "怎样从专家示范学会控制？", "学习者偏离示范轨迹后，需要处理专家没展示的状态。", "写出BC损失，理解分布偏移、交互模仿与逆强化学习。", "第3、5章。", "ABJKS 第13章", 161, "IV · 从理论走向机器人", [
    ["bc", "Behavior Cloning", "用专家动作监督策略。"],
    ["shift", "闭环分布偏移", "用扰动比较动作回放与状态反馈。"],
    ["interactive", "交互模仿", "给新访问的状态补上专家信息。"],
    ["reward", "逆强化学习", "从示范推断reward。"],
  ]],
  ["09", "连续控制与LQR", "怎样从Bellman算出最优推力？", "位置和速度决定推力：先接近目标，再刹车停稳。", "用二次reward推导Riccati，验算最优价值与反馈轨迹。", "第1、3、6章。", "ABJKS 第14章；小车实验", 173, "IV · 从理论走向机器人", [
    ["dynamics", "小车动力学", "推力同时改变位置和速度。"],
    ["objective", "停稳与省力：二次reward", "用reward评价误差、速度和用力。"],
    ["riccati", "Bellman与Riccati", "完成平方，求最优价值与推力。"],
    ["feedback", "反馈轨迹与回报", "调整参数，查看状态、动作与return。"],
    ["limits", "稳定性与动作约束", "检查有限时域、推力限幅和模型误差。"],
  ]],
  ["10", "机器人延伸：训练流程与GCRL", "怎样训练能完成不同目标的机器人？", "同一状态换一个目标，策略就可能需要相反的推力。", "定义机器人环境，比较PPO/SAC，用GCRL与HER学习多目标任务。", "第5–9章。", "机器人补充；Ben Eysenbach论文 §2–4", null, "IV · 从理论走向机器人", [
    ["interface", "机器人环境接口", "写清观测、动作、reward和终止。"],
    ["algorithms", "连续PPO与SAC", "比较数据使用方式和熵目标。"],
    ["gcrl", "目标条件策略", "让策略同时输入状态和目标。"],
    ["relabel", "HER与未来状态表示", "保留转移，重算目标相关reward和终止。"],
    ["evaluation", "机器人实验评估", "比较回报、成功率、扰动与目标泛化。"],
  ]],
];
const bridge = [
  ["先用小车定义RL任务。", "MDP、V/Q与Bellman。", "模型未知时，需要从数据估计价值。"],
  ["把已知模型换成采样数据。", "数据接口与采样更新。", "连续状态太多，需要函数近似。"],
  ["用函数表示连续价值。", "函数表示、覆盖与stitching。", "数据缺少关键状态时，需要主动探索。"],
  ["从读取数据转向主动采样。", "乐观探索与结构假设。", "连续动作难以枚举，下一章直接优化策略。"],
  ["从优化价值转向直接优化策略。", "PG、baseline与GAE。", "得到梯度后，还要判断更新方向和步长。"],
  ["用KL衡量策略变化。", "最优性、Fisher与自然梯度。", "下一章用TRPO/PPO控制旧数据上的更新。"],
  ["用旧轨迹改善策略。", "Surrogate、TRPO/PPO与熵。", "机器人试错昂贵，下一章从专家示范学习。"],
  ["用示范减少试错。", "BC、分布偏移与交互模仿。", "下一章用LQR精确验算连续控制。"],
  ["把Bellman用于连续小车。", "最优价值、Riccati与反馈。", "最后一章加入动作约束、模型学习和多目标任务。"],
  ["从单目标反馈走向多目标学习。", "机器人训练与GCRL。", "用同一任务、预算和评估协议比较论文方法。"],
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
  ["1 · MDP", "第1章", "任务、V/Q、Bellman与DP；LP证明选读。"],
  ["2 · Generative model", "第2章", "数据接口与采样误差；复杂度证明选读。"],
  ["3–4 · Features / Offline", "第3章", "函数表示、Bellman闭包、FQI与覆盖。"],
  ["5–6 · Bandits / Exploration", "第4章", "UCB与策略性探索；regret证明选读。"],
  ["7–8 · Linear MDP / Bellman rank", "第4章后半", "结构假设；证明选读。"],
  ["9 · Policy gradient", "第5章", "PG推导；Gaussian与GAE补充。"],
  ["10–11 · Optimality / NPG", "第6章", "最优性、Fisher与兼容近似。"],
  ["12 · Conservative optimization", "第7章", "CPI、TRPO；PPO与熵补充。"],
  ["13 · Imitation", "第8章", "BC、交互模仿与IRL。"],
  ["14 · LQR", "第9章", "动力学、Riccati与反馈；SDP/SLS选读。"],
  ["机器人与GCRL补充", "第10章", "环境接口、PPO/SAC、GCRL与HER。"],
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
