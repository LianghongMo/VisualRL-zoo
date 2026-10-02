import { h } from "../ui/dom.js";
export const COURSE = "https://www.polarislab.org/cos435-rl/index.html";
export const SOURCES = {
  pg: ["Policy Gradient · 对数导数与零期望引理", "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html"],
  trpo: ["TRPO §2、§4–6：性能差、surrogate与信赖域", "https://arxiv.org/pdf/1502.05477"],
  ppo: ["PPO §3、§5：clipping与训练流程", "https://arxiv.org/pdf/1707.06347"],
  gae: ["GAE §2–3：多步优势与几何混合", "https://arxiv.org/pdf/1506.02438"],
  variance: ["Greensmith et al. §5：baseline与方差", "https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf"],
  inference: ["Levine §3.1–3.2：固定动力学与Jensen/ELBO", "https://arxiv.org/html/1805.00909v3#S3.SS2"],
  sac: ["SAC §3–4：奖励与熵", "https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf"],
  shaping: ["Ng et al. §3：势函数塑形与策略不变性", "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf"],
  rainbow: ["Rainbow：Double Q与DQN组件", "https://arxiv.org/pdf/1710.02298"],
  dqn: ["DQN §4–5：回放与奖励处理", "https://arxiv.org/pdf/1312.5602"],
  evaluation: ["Deep RL 教材 §9.2：可靠实验比较", "https://arxiv.org/pdf/1811.12560"],
  time: ["Gymnasium：真正终止与时间截断", "https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/"],
};
export function readings(...keys) {
  return h("p", { class: "source-note" }, "阅读依据：", keys.map((key, i) => [
    i ? "；" : "", h("a", { href: SOURCES[key][1], target: "_blank", rel: "noopener" }, SOURCES[key][0]),
  ]));
}
