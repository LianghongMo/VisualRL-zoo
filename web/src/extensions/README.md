# 保留的扩展内容

这里放暂未接入八章主线的教学草稿和视觉组件。Python 算法、浏览器算法和训练数据仍保留在各自的正式目录，依赖仍保留在 `package.json` 与 `pyproject.toml` 中。

| 内容 | 文件 | 当前状态 |
|---|---|---|
| 表格 PPO 教学草稿 | [ppo-draft.js](ppo-draft.js) | 保留采集、优势、minibatch、ratio、clipping 实验；原来的 `lesson14.js`，尚未适配现在的课程壳 |
| MuJoCo 训练回放区 | [mujoco-section.js](mujoco-section.js) | 使用已有训练数据；可以复用回放功能，教学叙述与样式仍待适配 |
| 旧视觉组件 | [ui/](ui/) | 保留 3D 场景、Q/policy 网格、曲线、更新记录、折扣图与经验图等工具；主线使用的组件留在 `src/ui/` |

这些文件不注册到当前导航，也不进入当前网页的构建入口。它们是后续写章节时可复用的材料，而不是已经完成的课程。

接回 PPO 前，需要先讲清策略梯度、actor/critic、优势与 GAE；再为草稿补齐当前课程的学习目标、必学知识、中文讲解、数值验收和 `lessonHeader` / `lessonFooter` 接口。旧组件的样式也要适配当前页面。

相关代码和数据：

- Python PPO：[表格实现](../../../visualrl/algorithms/tabular/ppo.py)、[深度实现](../../../visualrl/algorithms/deep/ppo.py)。
- 浏览器 PPO：[rl/tabular/ppo.js](../rl/tabular/ppo.js)。
- MuJoCo 回放数据：[data/mujoco/](../data/mujoco/)。
- 训练与导出：[scripts/train_mujoco_ppo.py](../../../scripts/train_mujoco_ppo.py)。
- 后续安排：[扩展路线图](../../../docs/roadmap.md)。
