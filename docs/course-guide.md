# 从理论到连续机器人的课程导读

主要参考[ABJKS工作草稿](https://rltheorybook.github.io/rltheorybook_ABJKS.pdf)，版本2026-06-27。正文共14章，PDF页序比印刷页码大12。课程遵循其基础、探索、策略优化、模仿与LQR顺序，压缩前半程，用原创连续小车例子连接公式与控制。

[项目首页](../README.md) · [验收表](course-audit.md)

## 目录与必学知识

| 网页章 | 原书位置 | 必须掌握 | 实验或物理例子 |
|---|---|---|---|
| 1 MDP、价值与规划 | 第1章，正文3起 | 完整状态、最优目标、V/Q、Bellman、有限/无限时域、评价/改进 | 同位置不同速度；连续单位置Bellman最小化 |
| 2 数据模型与采样 | 第2章，21起 | generative/offline/online，样本波动、误差传播、MC/TD/SARSA/Q-learning、终止/截断 | 采样均值方差；同一步数据的不同自举目标 |
| 3 函数近似与覆盖 | 第3–4章，35、53起 | 共享特征、realizability/闭包、FQI、覆盖、stitching兼容性 | 线性/二次拟合曲线；未见高推力执行器反例；位置/速度匹配 |
| 4 探索与结构 | 第5–8章，63、73、87、95起 | UCB、轨迹探索、Linear MDP与Bellman rank的假设 | 夹持重置bandit；四阶段操作；分清概率核结构与机械动力学 |
| 5 策略梯度 | 第9章，111起 | 密度、轨迹PG、log-derivative、因果性/折扣、baseline、critic/GAE | 推力Gaussian更新；精确方差；终止/截断 |
| 6 最优性与NPG | 第10–11章，119、131起 | stationary/optimal，覆盖、Gaussian KL、Fisher、兼容回归 | 相同均值步长与不同方差；小梯度差策略 |
| 7 保守优化 | 第12章，141起 | 性能差、surrogate、动作ratio、CPI/TRPO、PPO符号与KL边界、Jensen/熵 | 状态访问反例；正负clip曲线；log弦与softmax |
| 8 模仿学习 | 第13章，161起 | BC、状态分布偏移、专家访问、DAgger/AggreVaTe区别、IRL | 扰动/质量改变后的专家推力回放与反馈 |
| 9 连续LQR | 第14章，173起 | 动力学单位、二次代价、Bellman/Riccati、时变反馈、稳定性、约束 | 真连续轨迹、相平面、推力、P/K表、代价验算 |
| 10 机器人/GCRL | 独立延伸 | 完整接口、tanh密度、PPO/SAC定位、目标条件、HER/占用、研究验收 | 同物理状态换目标改变反馈；位置和停稳共同验收 |

本课程不是逐页转写：LP、复杂度阶数、集中与下界、regret细节、SDP/SLS等长证明留在原书。第4章仅保留探索和结构的必要概念；重点时间留给第5–10章。

参考[Harvard CS2824公开课程](https://harvard-cs2824-s26.github.io/)从障碍引出方法、把成立条件紧邻算法的节奏，不复制PPT正文、图示或布局。Gaussian、GAE、连续SAC和GCRL在相关小节另给原始来源；第10章不被标成书中不存在的机器人章节。

## 统一任务与符号

小车p/v是位置和速度，u是连续推力，状态含目标误差e=p−g；每Δt=0.5秒决策。状态和动作连续，采样时间离散。反馈必须能区分相同位置、不同速度；有限时域包含时间/剩余步数。

奖励式采用J最大化，控制式采用C最小化，二者以r=−c连接。无限折扣例子γ<1；有限LQR例子γ=1且含末端代价。零终止尾值的采样目标要把末端惩罚计入最后奖励。

Gaussian等策略有密度，确定性反馈是Dirac分布；连续状态的占用一般用测度。密度比/覆盖必须有支持和相应绝对连续条件，不能用Pr(S=s)替代连续密度。

## 为什么这些章节衔接

MDP定义对象，价值定义比较，Bellman复用未来；未知模型改用数据。连续状态迫使共享函数，函数近似又带来闭包和覆盖问题。探索决定怎样补数据，策略梯度直接参数化连续动作。Fisher/KL控制行为变化，旧批次促使surrogate与保守优化出现。示范提供另一种监督，LQR给一个能精确验算的连续反馈基线。最后再补真实机器人接口、未知模型与多目标研究。

## 关键边界

- Generative model是任意合法状态动作的采样oracle；普通模拟器是否能如此重置要检查。Replay不把online自动变成offline。
- Q*可表示、所有策略Q可表示、Bellman completeness不同。后两者一般不构成全序关系；容量大不保证闭包。
- Linear MDP指概率核结构，不等于x′=Ax+Bu。前半表格有界奖励理论不能直接套到无界二次成本。
- PG保留初始折扣目标的外层γᵗ。动作无关baseline不改期望，但未必降方差；GAE不跨reset递推。
- Surrogate用旧状态占用，动作ratio没有修正状态分布。平均KL不约束每个罕见状态；PPO不是硬ratio/KL界或真实J全局下界。
- 有限Riccati在正确已知模型、无约束动作下精确。限幅、质量偏差或扰动后报告实际代价，不声称仍是原最优。
- 有限时域最优与渐近稳定不同；固定反馈用ρ(A−BK)<1，一般时变系统另分析。
- 回放实验每次按当前初态重算专家序列；运行中的扰动、质量误差和限幅用于显示开环/闭环差别，不冒充BC训练。
- GCRL目标成功要包含任务所需状态；高速穿过停车位置不是停稳。HER保留物理转移、失败与reset边界，重算任务奖励。

第10章主要参考[Ben Eysenbach博士论文](https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf)第2–4章、附录B，继续区分目标占用、首次到达、分类/对比和真实局部控制能力。网页没有接入完整神经PPO/SAC/GCRL训练。

## 链接与保留代码

当前入口为#chapter-01至#chapter-10，小节如#chapter-05/derivation、#chapter-07/ppo、#chapter-09/feedback。旧lesson映射：01→任务、04/05→价值、06→规划、02→数据接口、07/08→采样更新、03→stitching、09→GCRL。旧章内概念链接也转到新位置；裸章号按新目录解释。

公共课程仅从web/src/book/构建。上一版十章移入web/src/extensions/textbook-course/，原单项实验、算法、Python package、依赖、数据和历史保留；公共导航不混入归档内容。
