# 从理论到连续机器人的课程导读

按[ABJKS工作草稿](https://rltheorybook.github.io/rltheorybook_ABJKS.pdf)（2026-06-27）的基础、探索、策略优化、模仿与LQR顺序组织，用原创连续小车算例连接公式与控制。原书正文14章，PDF页序为印刷页码加12。

[项目首页](../README.md) · [验收表](course-audit.md)

## 目录与必学知识

| 网页章 | 原书位置 | 必须掌握 | 实验或物理例子 |
|---|---|---|---|
| 1 MDP、价值与规划 | 第1章，正文3起 | 完整状态、最优目标、V/Q、Bellman、有限/无限时域、评价/改进 | 同位置不同速度；连续单位置Bellman最大化 |
| 2 数据模型与采样 | 第2章，21起 | generative/offline/online，样本波动、误差传播、MC/TD/SARSA/Q-learning、终止/截断 | 采样均值方差；同一步数据的不同自举目标 |
| 3 函数近似与覆盖 | 第3–4章，35、53起 | 共享特征、realizability/闭包、FQI、覆盖、stitching兼容性 | 线性/二次拟合曲线；未见高推力执行器反例；位置/速度匹配 |
| 4 探索与结构 | 第5–8章，63、73、87、95起 | UCB、轨迹探索、Linear MDP与Bellman rank的假设 | 夹持重置bandit；四阶段操作；分清概率核结构与机械动力学 |
| 5 策略梯度 | 第9章，111起 | 密度、轨迹PG、log-derivative、因果性/折扣、baseline、critic/GAE | 推力Gaussian更新；精确方差；终止/截断 |
| 6 最优性与NPG | 第10–11章，119、131起 | stationary/optimal，覆盖、Gaussian KL、Fisher、兼容回归 | 相同均值步长与不同方差；小梯度差策略 |
| 7 保守优化 | 第12章，141起 | 性能差、surrogate、动作ratio、CPI/TRPO、PPO符号与KL边界、Jensen/熵 | 状态访问反例；正负clip曲线；log弦与softmax |
| 8 模仿学习 | 第13章，161起 | BC、状态分布偏移、专家访问、DAgger/AggreVaTe区别、IRL | 扰动/质量改变后的专家推力回放与反馈 |
| 9 连续LQR | 第14章，173起 | 动力学单位、二次reward、Bellman/Riccati、时变反馈、稳定性、约束 | 连续轨迹、相平面、推力、P/K表、return验算 |
| 10 机器人/GCRL | 独立延伸 | 完整接口、tanh密度、PPO/SAC定位、目标条件、HER/占用、研究验收 | 同物理状态换目标改变反馈；位置和停稳共同验收 |

LP、复杂度、集中、下界、regret、SDP/SLS证明列为选读；第4章压缩探索与结构，重点放在第5–10章。

借鉴[Harvard CS2824公开课程](https://harvard-cs2824-s26.github.io/)从问题引出方法、紧邻算法说明条件的讲解方式；正文与算例原创。Gaussian、GAE、连续SAC和GCRL另列原始来源，第10章为机器人补充。

## 统一任务与符号

小车状态x=(e,v)，e=p−g，动作u是推力；每Δt=0.5秒决策。状态和动作连续、决策时间离散，有限时域还需时间或剩余步数。

统一最大化期望return J，V/Q表示后续期望return。无限折扣用γ<1；有限LQR用γ=1，r=−xᵀMx−uᵀRu，r_H=−x_HᵀM_Hx_H，Vₜ*=−xᵀPₜx。采用零终止尾值时，末端reward计入最后一步奖励。

默认M=diag(1,0.2)、R=0.5、M_H=diag(20,10)，从(2,0)出发、H=6，return为−12.500159。单位置Bellman的V(2)=−52/9；特征实验真值为V(x)=−x²。

Gaussian策略有密度，确定性策略是Dirac分布；连续占用一般用测度，密度比与覆盖要求相应的绝对连续和支持条件。

## 为什么这些章节衔接

MDP定义任务，V/Q评价未来，Bellman支持规划；模型未知时用数据学习。连续状态需要函数近似，可靠学习需要闭包与覆盖，探索负责补数据。策略梯度优化连续动作，Fisher/KL和保守更新控制行为变化。示范提供监督，LQR提供可验算的反馈基线，最后连接机器人接口与多目标任务。

## 关键边界

- Generative model支持任意合法状态动作的独立采样；模拟器需支持相应重置。持续交互并使用replay仍属online。
- Q*可实现、所有策略Q可实现、最优Bellman completeness是不同假设；后两者一般互不包含。
- Linear MDP描述概率核的特征结构，x′=Ax+Bu描述物理动力学；无界二次reward需另行分析。
- PG保留初始折扣目标的外层γᵗ。动作无关baseline不改期望，但未必降方差；GAE不跨reset递推。
- Surrogate用旧状态占用，动作ratio没有修正状态分布。平均KL不约束每个罕见状态；PPO不是硬ratio/KL界或真实J全局下界。
- 有限Riccati在正确已知模型、无约束动作下精确；限幅、质量偏差或扰动后报告实际return。
- 有限时域最优与渐近稳定不同；固定反馈用ρ(A−BK)<1，一般时变系统另分析。
- 回放按当前初态重算专家序列，用扰动、质量误差和限幅比较开环与反馈；BC训练另需数据与优化。
- GCRL目标成功要包含任务所需状态；高速穿过停车位置不是停稳。HER保留物理转移、失败与reset边界，重算任务奖励。

第10章参考[Ben Eysenbach博士论文](https://ml.cmu.edu/research/phd-dissertation-pdfs/thesis_eysenbach.pdf)第2–4章、附录B，区分目标占用、首次到达、分类/对比与局部控制。网页尚未接入完整神经PPO/SAC/GCRL训练。

## 链接与保留代码

当前入口为#chapter-01至#chapter-10，小节如#chapter-05/derivation、#chapter-07/ppo、#chapter-09/feedback。旧lesson映射：01→任务、04/05→价值、06→规划、02→数据接口、07/08→采样更新、03→stitching、09→GCRL。旧章内概念链接也转到新位置；裸章号按新目录解释。

公共课程仅从web/src/book/构建。上一版十章移入web/src/extensions/textbook-course/，原单项实验、算法、Python package、依赖、数据和历史保留；公共导航不混入归档内容。
