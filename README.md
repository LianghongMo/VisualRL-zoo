# Visual RL · 看见强化学习

用一台机器人，连起强化学习的基本过程：**环境 → 经验 → 学习目标 → 价值更新 → 下一次行动**。

每章先列必须掌握的知识点，再用物理图像、正文公式、数值算例和交互实验展开。章末逐项自检，并可回到对应内容；源码和扩展训练按需展开。完整内容检查见 [课程验收表](docs/course-audit.md)。

## 打开网页

直接打开 [docs/index.html](docs/index.html)，无需安装或服务器。这是包含全部教学代码和样式的单个 HTML 文件，核心实验离线可用。

```bash
cd web
npm ci
npm run dev       # http://localhost:8000，源码修改后自动重新构建
npm test          # 教学交互、环境、Python / JS 算法一致性测试
npm run build     # 生成 docs/index.html 和部署用 dist/index.html
```

## 学习主线

页面保留原来的 URL 标识，显示编号按新的阅读顺序从 1 到 7 排列。

| 章 | 页面 | 必须掌握什么 | 例子展示的图像 |
|---|---|---|---|
| 1 | `#lesson-01` MDP 与环境 | MDP、Markov 状态、策略、模型、经验与回合 | 定义任务，再读机器人一步反馈 |
| 2 | `#lesson-04` 回报与价值 | 奖励、回报、指定策略的价值 | 同样两条充电路线，只改变折扣 |
| 3 | `#lesson-05` Bellman 更新 | 期望方程、最优方程、价值迭代、随机后果 | 信息每轮向后传播一条边；随机时先平均再选动作 |
| 4 | `#lesson-02` 行动与学习 | 新增经验与更新估计 | 分开按两个按钮，连接和数字各自变化 |
| 5 | `#lesson-07` MC 与 TD | 完整回报与一步自举目标 | 比较何时可学习及信息怎样传播 |
| 6 | `#lesson-08` SARSA 与 Q-learning | 实际下一动作与估计最好的下一动作 | 悬崖边的探索，接进两种不同目标 |
| 7 | `#lesson-03` 在线与离线 | 数据的重组能力和覆盖限制 | 拼接旧路线，再实际补采缺失的经验 |

`#lesson-06` 策略迭代是第 3 章的选读扩展：评价只改数字，改进只改箭头。

初学者主线不展示未完成章节。PPO、MuJoCo 等已有算法与实验源码保留在仓库中，暂不作为教学入口；重新接入之前需要先补齐策略梯度、优势及函数近似的前置内容。

## 实验边界

- 一格是一个离散状态，一次移动是一次环境转移。进入充电站时收到奖励，终止之后的后续价值为 0。
- 第 3 章使用完整环境模型，是规划。第 4 章只在已观察到的确定性连接上规划，展示行动与学习两个循环；这不是所有 RL 算法的统一实现。
- 第 5 章固定路线、默认 α = 1；可调学习率、选择经验、保留估计再走一轮。随机问题通常需要累计多个样本。
- 第 6 章的一步更新使用明确标注的旧 Q 示例值；展开的 500 回合实验从零开始运行真实算法。固定种子的结果不代表统计结论。
- 第 7 章使用同一个确定性环境；逐边标记片段来源，采集与更新分开。完整状态、动力学、任务兼容才可拼接；随机环境不能把少量记录当作精确模型。

## 代码位置

- `web/src/lessons/`：各章内容和实验控制。
- `web/src/lessons/curriculum.js`：阅读顺序、学习目标和章节衔接。
- `web/src/ui/world-view.js`：由真实状态与转移绘制的地图和路线。
- `web/src/rl/`：浏览器中的算法。
- `visualrl/`：对应 Python 实现、环境和学习记录。
- `tests/golden/traces.json`：Python 导出的计算记录；浏览器算法必须产生同样结果。

```bash
pip install -e ".[dev]"
pytest
python examples/td_update.py
python examples/cliff_sarsa_vs_qlearning.py
```

深度 PPO 和 MuJoCo 训练代码保留用于后续研究：

```bash
pip install -e ".[deep]"
python scripts/train_mujoco_ppo.py Hopper-v5 --steps 1000000 --out web/src/data/mujoco/hopper.json
```
