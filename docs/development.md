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

## 章节、小节与复用实验

公共架构由 web/src/lessons/curriculum.js 的 CHAPTERS、PARTS、LEGACY_ROUTES 定义；正文在 web/src/chapters/。章节是知识单元，具体实验放入小节。

1. 明确核心问题、前置知识、必学清单与例子。新主题优先归入所属章，仅独立知识层级才新增一章。
2. 在 curriculum.js 登记章/小节、教材对应、稳定id和relation（承接、结果、下一章的问题）；地址为 #chapter-XX/topic。
3. 在 chapters/knowledge.js 写知识、对应小节与自检问题。
4. 用 chapters/shell.js 的 chapterHeader/topic/chapterFooter 组织，在 chapters/index.js 注册。main.js 使用统一路由。
5. 核心方程直接显示，连接真实计算，说明模型、终止、策略和采样条件。
6. 在 tests/chapters.test.mjs 验证有意义的行为和边界，更新导读/验收，再构建。

章内跳转只滚动、不重新挂载；跨章先调用 cleanup，取消训练与监听。旧lesson链接在LEGACY_ROUTES映射到小节，上一版本的章内链接在CHAPTER_ROUTE_ALIASES中兼容。

原实验保存在 lessons/，lesson-catalog.js 与 lessons/knowledge.js 为旧元数据，不控制公共目录。capture 复用原模块实验，移除旧页头/页脚/导航，保留事件和 cleanup；MDP嵌入时关闭全局快捷键，避免阅读其他小节时移动机器人。

rl/teaching-labs.js 集中新的 bandit、Dyna、线性价值、二动作梯度和表格资格迹计算；原有Python/JS一致性算法原位保留。教学计算与完整神经网络训练分开说明。

rl/policy-math.js 提供精确有限算例：baseline均值/方差、GAE双mask、两状态surrogate、log Jensen、熵目标/KL、势函数塑形和Double DQN。chapters/policy-derivations.js、entropy-tools.js、value-tricks.js把计算接到对应章节的公式和控件；sources.js集中公开课程与原始阅读链接。推导用proof()注明恒等式、局部近似或实践方法，避免混淆其保证。

tests/policy-tools.test.mjs校验这些数学恒等式、边界与反例，并通过Node/linkedom触发真实章节控件。修改折扣、终止规则、采样分布或目标时，应同时更新对应算例与课程验收表，保持知识清单指向存在的小节。

extensions/ 保留PPO/MuJoCo草稿，接回时适配当前章节界面与知识清单。Python新增算法放 visualrl/，浏览器对应实现放 web/src/rl/。
