# Visual RL · 从理论到连续机器人

用连续推力小车贯穿 **任务 → 数据与表示 → 探索 → 策略优化 → 模仿 → 连续反馈控制**，把位置、速度、推力与reward、return、V/Q连到可验算的公式。

[在线课程](https://visual-rl-learning-path.lm8598.chatgpt.site) · [连续小车实验](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-09/feedback) · [离线网页](docs/index.html) · [课程导读](docs/course-guide.md) · [内容验收](docs/course-audit.md)

按 [Reinforcement Learning: Theory and Algorithms（ABJKS）](https://rltheorybook.github.io/rltheorybook_ABJKS.pdf) 2026-06-27工作草稿组织：压缩表格基础与证明，逐步走向连续机器人。借鉴 [Harvard CS2824公开讲义](https://harvard-cs2824-s26.github.io/)从问题引出算法、紧邻公式说明条件的讲解方式；正文、图像与算例独立编写。

## 当前课程

| 网页章 | 内容 | 原书对应 |
|---|---|---|
| [1](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-01) | MDP、价值与规划 | 第1章，压缩 |
| [2](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-02) | 数据模型与采样学习 | 第2章；MC/TD/SARSA/Q-learning补充 |
| [3](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-03) | 函数近似、离线覆盖与stitching | 第3–4章 |
| [4](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-04) | 探索与可学习结构 | 第5–8章精选、压缩 |
| [5](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-05) | 策略梯度与连续Gaussian、baseline、GAE | 第9章；连续策略与GAE补充 |
| [6](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-06) | 最优性、自然梯度与近似 | 第10–11章 |
| [7](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-07) | CPI、TRPO、PPO、Jensen与熵 | 第12章；熵与实践补充 |
| [8](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-08) | 模仿学习、闭环分布偏移与专家访问 | 第13章 |
| [9](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-09) | 连续动力学、Riccati与LQR | 第14章；原创小车实验 |
| [10](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-10) | 机器人接口、连续PPO/SAC与GCRL | 机器人延伸；Ben Eysenbach论文 |

每章有目录、前置知识、公式、实验与自检。LP、复杂度、集中、regret、下界、SDP/SLS证明列为选读；增益符号等独立推导并验算。

## 连续实验

小车状态为位置和速度，每0.5秒选择连续推力。LQR使用reward权重M、R、M_H，最大化return；V*=−xᵀPx。默认从(2,0)出发、H=6，return为−12.500159，与初始Riccati价值一致。可调控制器、初速、时域、推力权重、质量、外力与限幅，查看相平面、推力和P/K。

还可运行Gaussian更新、baseline方差、终止/截断GAE、surrogate反例、PPO、Jensen/softmax与目标反馈；网页尚未接入完整神经PPO/SAC/GCRL训练。

## 仓库结构

~~~text
web/src/book/                      当前唯一公共课程：目录、首页、正文、知识清单、实验
web/src/rl/continuous-control.js    连续动力学、有限LQR、Gaussian与目标算例
web/src/rl/                        保留的浏览器算法与环境
web/src/ui/                        当前通用界面与MathML公式
web/src/lessons/                    原始单项表格/GCRL实验，供复用与测试
web/src/extensions/textbook-course/ 上一版课程归档，不进入公共构建
web/src/extensions/                PPO/MuJoCo草稿与旧视觉工具
web/src/data/mujoco/                保留的日志与回放
web/tests/                         当前课程、归档算例及Python/JS一致性
visualrl/                          Python package：表格算法、环境、深度PPO
examples/ scripts/ tests/          Python例子、训练脚本与检查
docs/                             离线网页、导读、验收、开发与路线图
~~~

Python package、依赖、MuJoCo数据与Git历史都保留。旧lesson和章内概念链接映射到新位置，见[导读](docs/course-guide.md)。

## 本地使用

网页使用Node.js 24与npm：

~~~bash
cd web
npm ci
npm run dev       # http://localhost:8000
npm test
npm run build     # docs/index.html与dist/index.html
~~~

下载docs/index.html直接打开即可使用核心实验，无需服务器。

Python package支持Python 3.10及以上：

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/td_update.py
~~~

GitHub检查网页测试、离线产物可重建性与Python算法。深度PPO、MuJoCo和新增章节方法见[开发说明](docs/development.md)，未接入训练状态见[路线图](docs/roadmap.md)。
