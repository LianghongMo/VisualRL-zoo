# 开发与运行

[返回项目首页](../README.md) · [课程导读](course-guide.md) · [扩展路线图](roadmap.md)

## 网页

当前构建和测试使用 Node.js 24、npm。依赖及锁定版本保留在 `web/package.json` 和 `web/package-lock.json`；Three.js 用于之后的 3D 实验，仍然保留。

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

深度依赖作为 `deep` 可选安装项保留；不影响只学习表格算法的安装。要使用原有训练脚本，再安装：

```bash
python -m pip install -e ".[deep]"
python scripts/train_mujoco_ppo.py Hopper-v5 --steps 1000000 --out web/src/data/mujoco/hopper.json
```

脚本导出训练日志、轨迹和模型快照。已有回放 JSON 保留在 `web/src/data/mujoco/`；新快照 `*.pt`、`*.pth` 与临时日志忽略，避免误提交大型训练产物。使用这些文件前注意输出路径，训练脚本会写入指定文件。

`tests/test_deep_ppo.py` 在缺少 Torch 或 MuJoCo 时跳过；普通检查不安装深度依赖，也不运行长时间训练。后续需要验收深度模块时，在安装 `deep` 后运行相关测试。

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

唯一入口是web/src/main.js → book/curriculum.js、book/index.js与book/home.js。book/组织ABJKS目录映射、十章正文、前置/承接、知识清单、来源和真实实验；不按文件名自动生成目录。

1. 先明确核心问题、原书位置、机器人物理例子和成立条件。
2. 在book/curriculum.js登记章节、小节、稳定id、source页码与relation；地址为#chapter-XX/topic。
3. 在book/knowledge.js写必学概念、对应小节和可验证问题。
4. 用book/shell.js的chapterHeader/topic/chapterFooter组织，在book/index.js登记mount。
5. 在book/labs.js接真实计算；推导用proof()注明恒等式、局部近似或实践步骤。
6. 修改相关计算与界面时，更新tests/robot-course.test.mjs及导读/验收，再构建。

book/foundations.js覆盖任务、数据、表示与探索；policy.js覆盖PG、NPG与保守更新；robotics.js覆盖模仿、LQR和机器人/GCRL。式子通过KaTeX生成MathML，浏览器本地显示，不依赖外部数学服务。

章内跳转只滚动、不重新挂载；跨章调用cleanup。book/curriculum.js的LEGACY_ROUTES与ALIASES兼容旧lesson和章内概念链接。裸章号按新目录解释。

## 真实计算与检查

rl/continuous-control.js包含原创双积分动力学、有限Riccati、控制器轨迹、Gaussian score/KL/方差、目标条件反馈和改标工具。控制输入是推力，状态和动作连续，采样周期离散。有限LQR统一u=−Kx，默认参数和独立有理数算例可交叉验算。

rl/policy-math.js保留精确GAE双mask、两状态surrogate、Jensen/熵等工具；原有Python/JS一致性算法原位保留。教学算例与完整神经训练分开说明。

tests/robot-course.test.mjs检验物理单位、独立LQR解、代价恒等、质量/扰动/限幅、数值积分Gaussian矩、HER边界、全部章节知识与公式、真实控件。tests/chapters.test.mjs保留归档模块验收，其中公共导航测试使用新main.js。其他测试继续检查算法、旧实验与Python golden一致性。

## 归档与后续扩展

上一版Sutton/COS435课程移到extensions/textbook-course/，包括旧curriculum、首页、章节和知识清单，不进入公共构建。lessons/保留原始单项表格/GCRL实验；extensions/还保留PPO、MuJoCo与旧视觉工具，数据和依赖均保留。

接回训练材料时适配当前任务/状态/动作协议，补真实采集、训练和独立执行评估。Python新增算法放visualrl/，浏览器对应实现放web/src/rl/；不要用解析目标曲线称完成神经训练。
