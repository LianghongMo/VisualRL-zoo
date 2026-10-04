# 保留的扩展材料

当前公共课程只从src/book/构建。这里的材料不注册公共导航，保留用于后续复用；Python/browser算法和数据仍在正式目录，package依赖保持。

| 材料 | 位置 | 状态 |
|---|---|---|
| 上一版十章课程 | [textbook-course/](textbook-course/README.md) | Sutton/COS435版正文、目录、首页、知识清单；继续测试，不进入公开构建 |
| 表格PPO草稿 | [ppo-draft.js](ppo-draft.js) | 采集、优势、minibatch、ratio/clipping；待适配连续机器人接口 |
| MuJoCo回放 | [mujoco-section.js](mujoco-section.js) | 读取保留数据，任务讲解与当前课程界面待适配 |
| 旧视觉组件 | [ui/](ui/) | 3D、网格、曲线与记录工具；按具体教学目的接回 |

公共第5–7章已建立PG、GAE、NPG、surrogate与PPO；第9章运行连续LQR，第10章定义完整训练协议。接入草稿时需真实采集、模型训练和独立执行验收；不能用解析曲线替代训练证据。

[Python深度PPO](../../../visualrl/algorithms/deep/ppo.py) · [浏览器PPO](../rl/tabular/ppo.js) · [MuJoCo数据](../data/mujoco/) · [路线图](../../../docs/roadmap.md)
