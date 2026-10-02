# 后续扩展路线图

[返回首页](../README.md) · [开发说明](development.md) · [保留草稿](../web/src/extensions/README.md)

当前十章：九章基础加研究专题。区分可操作算例、算法package和待接入的完整训练。

## 当前网页覆盖

| 方向 | 实现 |
|---|---|
| 任务/表格方法 | MDP、G/V/Q、Bellman、DP、MC、TD、SARSA、Q-learning、Expected SARSA公式与例子 |
| 探索 | 第4章MC控制引入ε-soft；第5章探索小节补充真实Bernoulli bandit、均值/常数步长、UCB/乐观初值 |
| 多步/规划 | n-step、λ-return、真实表格资格迹、观测模型、Dyna-Q模型抽样 |
| 近似 | 特征、线性梯度、泛化/冲突、半梯度TD；DQN结构与公式 |
| 策略 | 二动作REINFORCE、baseline/优势、actor–critic公式、PPO clipped曲线 |
| 研究 | 第10章离线/stitching、目标改标、未来分布、分类/对比、SoRB与论文入口 |

## 保留材料和下一步

| 方向 | 已有 | 待实现和验收 |
|---|---|---|
| 完整DQN | 第8章前置/公式 | 神经网络、回放、目标更新、稳定性与多种子评估 |
| 完整PPO | 第9章基础；Python表格/深度、浏览器算法和ppo-draft | GAE完整推导/实验，采集/minibatch/训练接入 |
| MuJoCo/3D | 脚本、五组回放、Three.js组件 | 状态动作物理含义、数据来源、奖励/行为联系、样式适配 |
| 神经GCRL | 第10章定义、改标、分类/对比/路标 | 真正训练表示/critic/actor，评估采样、覆盖、距离与泛化 |
| 其他研究 | Markov、模型、规划和离线前置 | 部分可观测、随机模型、离线分布偏移、连续控制 |
| 旧视觉组件 | extensions/ui/网格/曲线/图/3D | 按章节目的适配，用实验说明算法 |

Python package、npm依赖、MuJoCo JSON、测试和Git历史均保留。扩展文件不注册公共导航、不进入构建。完整训练接入后才声称已训练该方法。

扩展要求：核心问题/必学知识 → 图像/例子 → 方程/假设 → 真实更新 → 数值/执行验收。
