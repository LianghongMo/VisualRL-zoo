# Visual RL · 看见强化学习

用一台机器人，连起 **环境 → 经验 → 学习目标 → 价值更新 → 下一次行动**。每章都有物理图像、正文公式、可操作的实验和知识点自检。

仓库包含当前课程网页和可继续扩展的 Python 算法 package。PPO、MuJoCo、训练数据和依赖都保留，未完成的教学草稿集中在扩展目录。

[在线课程](https://visual-rl-learning-path.lm8598.chatgpt.site) · [离线网页](docs/index.html) · [课程导读](docs/course-guide.md) · [课程验收表](docs/course-audit.md)

离线使用时下载 `docs/index.html`，用浏览器打开即可；核心实验不需要服务器。

## 当前课程

参考 Sutton & Barto 第二版，按同一任务的学习依赖组织九章基础与一章研究。每章分小节，开头说明承接关系，结尾解释下一章要解决的剩余问题。

| 章 | 内容 | 教材对应 / 网页入口 |
|---|---|---|
| 1 | 强化学习任务与 MDP | §1、3的任务定义；[#chapter-01](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-01) |
| 2 | 回报、价值与最优策略 | §3的评价与控制目标；[#chapter-02](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-02) |
| 3 | Bellman 方程与动态规划 | §3的Bellman、§4；[#chapter-03](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-03) |
| 4 | Monte Carlo：预测、控制、重要性采样 | §5；[#chapter-04](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-04) |
| 5 | TD 学习与表格控制，含探索补充 | §6；探索参考§2；[#chapter-05](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-05) |
| 6 | 多步学习与资格迹 | §7、表格 §12；[#chapter-06](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-06) |
| 7 | 模型、规划与 Dyna | §8；[#chapter-07](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-07) |
| 8 | 函数近似与 Deep RL | §9–11、DQN；[#chapter-08](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-08) |
| 9 | 策略梯度与 Actor–Critic | §13、PPO；[#chapter-09](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-09) |
| 10 | 前沿课题：离线 RL、stitching 与 GCRL | Ben Eysenbach 博士论文；[#chapter-10](https://visual-rl-learning-path.lm8598.chatgpt.site/#chapter-10) |

第1章定义任务，第2章比较长期结果，第3章拆解未来并求解。模型未知时再进入MC/TD。赌博机归入第5章探索补充，表格资格迹与n-step对照，参考 [MIT Press 官方目录](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/10094/Toc.pdf?dl=1)。研究专题主要参考 Ben 的2023年论文，不作为所有最新方向的综述。

旧链接继续兼容：lesson-01进入第1章任务，lesson-04/05进入第2章回报/最优策略；lesson-06 进入第3章策略迭代；lesson-07/08 进入第5章预测/控制；lesson-02 进入第7章模型；lesson-03/09 进入第10章离线/GCRL。旧章内链接也继续映射：chapter-01/returns与optimal进入第2章，chapter-02/bandit、exploration与nonstationary进入第5章探索。完整知识清单与边界见[课程导读](docs/course-guide.md)。

## 仓库结构

```text
web/
  src/chapters/      十章正文、小节、知识清单与章节界面
  src/lessons/       公共目录、首页、保留并复用的交互实验
  src/ui/            当前课程使用的界面组件
  src/rl/            浏览器算法与环境
  src/extensions/    保留的 PPO 草稿、MuJoCo 回放与旧视觉组件
  src/data/mujoco/   已记录的训练日志与轨迹
  tests/             网页交互和 Python/JS 一致性检查
visualrl/            Python package：环境、经验、表格算法、深度 PPO
examples/            可直接运行的 Python 小例子
scripts/             一致性记录导出和 MuJoCo 训练
tests/              Python 测试与 golden 计算记录
docs/               离线网页、课程导读、开发说明与路线图
```

## 本地使用

网页开发使用 Node.js 24 和 npm。在 `web/` 中运行：

```bash
npm ci
npm run dev       # http://localhost:8000
npm test
npm run build     # 更新 docs/index.html 和部署产物 dist/index.html
```

Python package 支持 Python 3.10 及以上。在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/td_update.py
```

GitHub 自动检查网页测试、构建产物与 Python package。详细安装、算法目录、深度 PPO 训练和新增章节步骤见[开发说明](docs/development.md)。

## 后续扩展

后续章节所需的算法、环境、Three.js 和深度 RL 依赖均保留。当前网页已加入函数近似与策略梯度基础；完整训练和教学草稿的状态见[扩展路线图](docs/roadmap.md)与[扩展目录说明](web/src/extensions/README.md)。
