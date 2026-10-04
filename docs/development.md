# 开发与运行

[返回项目首页](../README.md) · [课程导读](course-guide.md) · [扩展路线图](roadmap.md)

## 网页

构建和测试使用 Node.js 24、npm；依赖由 `web/package.json` 与 `web/package-lock.json` 锁定，Three.js 留供 3D 实验。

```bash
cd web
npm ci
npm run dev
npm test
npm run build
```

- 开发预览：`http://localhost:8000`，只写入忽略的 `web/dist/preview/`。
- 网页入口：`web/src/main.js`；现有路由不按文件名排序。
- 正式构建：生成内容相同的 `docs/index.html` 与 `dist/index.html`。
- `docs/` 中的网页和分享图片随 Git 提交，作为可下载的离线版本；源码与它们保持一致。
- `dist/` 和 `web/dist/` 是生成产物，使用现有构建流程重新生成；Sites 配置保留在 `.openai/hosting.json`。
- 自动检查只测试和构建，不发布网页。

## Python package

`pyproject.toml` 声明 Python 3.10 及以上；目前自动检查使用 Python 3.12。建议在仓库根目录创建独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

| 目录 | 内容 |
|---|---|
| `visualrl/core/` | Transition、Trajectory、LearningTrace 与 rollout |
| `visualrl/envs/` | Bandit、Chain、GridWorld、共享地图与环境模型 |
| `visualrl/algorithms/tabular/` | Bandit、Bellman、价值/策略迭代、MC、TD、n-step TD、SARSA、Q-learning、经验图与表格 PPO |
| `visualrl/algorithms/deep/` | PyTorch 深度 PPO |
| `visualrl/visual/` | 文本形式的学习过程展示 |
| `examples/` | TD 单步更新、MC/TD 随机游走、悬崖 SARSA/Q-learning |

```bash
python examples/td_update.py
python examples/random_walk_mc_vs_td.py
python examples/cliff_sarsa_vs_qlearning.py
```

## 深度 PPO 与 MuJoCo

深度训练使用 `deep` 可选依赖，表格算法无需安装：

```bash
python -m pip install -e ".[deep]"
python scripts/train_mujoco_ppo.py Hopper-v5 --steps 1000000 --out web/src/data/mujoco/hopper.json
```

脚本导出日志、轨迹与模型快照，写入指定输出路径。已有回放在 `web/src/data/mujoco/`；新快照 `*.pt`、`*.pth` 与临时日志由 Git 忽略。

缺少 Torch 或 MuJoCo 时，`tests/test_deep_ppo.py` 跳过；深度模块需安装 `deep` 后单独验收，普通检查不运行长训练。

## Python 与浏览器算法一致性

`tests/golden/traces.json` 保存 Python 的计算记录；`web/tests/parity.test.mjs` 用同样的经验检查 JavaScript。修改算法后，确认两份实现一致，再重新导出：

```bash
python scripts/export_golden_traces.py
python -m pytest -q
cd web
npm test
npm run build
```

重导出后检查数值差异。仅整理目录时，原有计算记录和算法不需要修改。

## 当前公共课程

入口为web/src/main.js → book/curriculum.js、book/index.js与book/home.js。book/显式登记十章目录、前置知识、来源与实验。

1. 先明确核心问题、原书位置、机器人物理例子和成立条件。
2. 在book/curriculum.js登记章节、小节、稳定id、source页码与relation；地址为#chapter-XX/topic。
3. 在book/knowledge.js写必学概念、对应小节和可验证问题。
4. 用book/shell.js的chapterHeader/topic/chapterFooter组织，在book/index.js登记mount。
5. 在book/labs.js接真实计算；推导用proof()注明恒等式、局部近似或实践步骤。
6. 修改相关计算与界面时，更新tests/robot-course.test.mjs及导读/验收，再构建。

book/foundations.js负责基础与探索，policy.js负责PG/NPG和保守更新，robotics.js负责模仿、LQR与GCRL。KaTeX生成MathML，公式在浏览器本地显示。

章内跳转只滚动、不重新挂载；跨章调用cleanup。book/curriculum.js的LEGACY_ROUTES与ALIASES兼容旧lesson和章内概念链接。裸章号按新目录解释。

## 真实计算与检查

rl/continuous-control.js提供双积分动力学、有限Riccati、反馈轨迹、Gaussian score/KL/方差与目标改标。状态和动作连续、决策时间离散；LQR用M、R、M_H定义负二次reward，最大化return，u=−Kx、Vₜ*=−xᵀPₜx，默认return为−12.500159。

rl/policy-math.js提供GAE双mask、两状态surrogate与Jensen/熵工具；Python/JS一致性算法保留，完整神经训练单独验收。

网页共66项检查。tests/robot-course.test.mjs核对物理单位、独立LQR解、reward/return与Bellman恒等式、质量/扰动/限幅、Gaussian矩、HER边界、章节知识、公式与控件；LQR项名称含`reward return and satisfies Bellman`。tests/chapters.test.mjs保留归档验收，公共导航使用新main.js；其他测试检查算法、旧实验与Python golden一致性。

## 归档与后续扩展

上一版Sutton/COS435课程移到extensions/textbook-course/，包括旧curriculum、首页、章节和知识清单，不进入公共构建。lessons/保留原始单项表格/GCRL实验；extensions/还保留PPO、MuJoCo与旧视觉工具，数据和依赖均保留。

接回训练材料时适配当前任务/状态/动作协议，补真实采集、训练和独立执行评估。Python新增算法放visualrl/，浏览器对应实现放web/src/rl/；不要用解析目标曲线称完成神经训练。
