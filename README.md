# Visual RL · 看见强化学习

用一台机器人，连起 **环境 → 经验 → 学习目标 → 价值更新 → 下一次行动**。每章都有物理图像、正文公式、可操作的实验和知识点自检。

仓库包含当前课程网页和可继续扩展的 Python 算法 package。PPO、MuJoCo、训练数据和依赖都保留，未完成的教学草稿集中在扩展目录。

[在线课程](https://visual-rl-learning-path.lm8598.chatgpt.site) · [离线网页](docs/index.html) · [课程导读](docs/course-guide.md) · [课程验收表](docs/course-audit.md)

离线使用时下载 `docs/index.html`，用浏览器打开即可；核心实验不需要服务器。

## 当前课程

| 章 | 内容 | 原有页面标识 |
|---|---|---|
| 1 | MDP 与环境 | `lesson-01` |
| 2 | 回报与价值 | `lesson-04` |
| 3 | 最优策略与 Bellman | `lesson-05` |
| 4 | 行动与学习 | `lesson-02` |
| 5 | Monte Carlo 与 TD | `lesson-07` |
| 6 | SARSA 与 Q-learning | `lesson-08` |
| 7 | 在线、离线与 stitching | `lesson-03` |
| 8 | GCRL 与未来目标 | `lesson-09` |

`lesson-06` 是策略迭代选读。页面标识保持兼容，阅读顺序由 `curriculum.js` 定义。GCRL 主要参考 Benjamin Eysenbach 的博士论文；具体知识点、例子与参考位置见[课程导读](docs/course-guide.md)。

## 仓库结构

```text
web/
  src/lessons/       当前课程、目录和知识点
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

后续章节所需的算法、环境、Three.js 和深度 RL 依赖均保留。当前网页只接入已经整理好的课程；教学草稿的状态和待补前置知识见[扩展路线图](docs/roadmap.md)与[扩展目录说明](web/src/extensions/README.md)。
