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

## 加一章

1. 先明确必须掌握的知识、物理图像和实验中可改变的条件。
2. 在 `web/src/lessons/` 写页面；在 `curriculum.js` 放入阅读顺序，在 `knowledge.js` 写对应的自检与正文依据。
3. 在 `main.js` 注册路由。保留已有页面标识；章节编号来自课程顺序。
4. 把核心方程放在正文，连接到具体数据与数值。新增行为需要说明终止、截断、后续策略与数据假设。
5. 验证交互与算法，再构建离线网页；更新课程导读与验收表。

`web/src/extensions/` 提供之后可复用的草稿和组件；接入前要适配当前课程壳与视觉样式，并补齐必学知识。新增 Python 算法继续放在 `visualrl/`，对应浏览器实现放在 `web/src/rl/`。
