import { h, s, slider, button, replace } from "../../ui/dom.js";
import { metric, num } from "../../ui/world-view.js";
import { logJensen, entropyBound } from "../../rl/policy-math.js";
import { prose, equation, table } from "./shell.js";
import { proof } from "./policy-derivations.js";
import { readings } from "./sources.js";

export function entropyTools() {
  let second = 9, weight = 0.5, p = 0.5, temperature = 1;
  const plot = s("svg", { viewBox: "0 0 440 260", class: "chapter-plot", role: "img", "aria-label": "log凹函数曲线与弦，比较平均后取log和取log后平均" });
  const jStats = h("div", { class: "metrics" }), softStats = h("div", { class: "metrics" });
  function renderJensen() {
    const t = logJensen(1, second, weight), x = v => 35 + v / 12 * 370, y = v => 215 - 65 * v;
    const values = Array.from({ length: 89 }, (_, i) => 1 + i * 11 / 88);
    replace(plot,
      s("line", { x1: 35, x2: 420, y1: 215, y2: 215, stroke: "var(--rule)" }),
      s("polyline", { points: values.map(v => x(v) + "," + y(Math.log(v))).join(" "), fill: "none", stroke: "var(--learn)", "stroke-width": 3 }),
      s("line", { x1: x(1), y1: y(0), x2: x(second), y2: y(Math.log(second)), stroke: "var(--act)", "stroke-width": 2 }),
      s("line", { x1: x(t.mean), x2: x(t.mean), y1: y(t.logMean), y2: y(t.meanLog), stroke: "var(--reward)", "stroke-width": 4 }),
      [t.logMean, t.meanLog].map((v, i) => s("circle", { cx: x(t.mean), cy: y(v), r: 5, fill: i ? "var(--act)" : "var(--learn)" })),
      s("text", { x: 240, y: 24 }, "曲线：log x；直线：两点的弦"),
      s("text", { x: 240, y: 44 }, "绿点≥蓝点；竖线是差距"),
      [1, 3, 6, 9, 12].map(v => s("text", { x: x(v), y: 245, "text-anchor": "middle" }, v)));
    replace(jStats, metric("先平均，再取log", num(t.logMean)), metric("先取log，再平均", num(t.meanLog)), metric("Jensen差距", num(t.gap)));
  }
  function renderSoft() {
    const t = entropyBound([0, 2], [1 - p, p], temperature);
    replace(softStats, metric("π(奖励2动作)", num(p, 6)), metric("原始预期奖励 EπQ", num(t.expected)), metric("策略熵 H", num(t.entropy)),
      metric("奖励＋温度×熵", num(t.lower)), metric("soft价值上界", num(t.softValue)), metric("上界−当前目标", num(t.gap)),
      metric("温度×KL", num(temperature * t.kl)), metric("最优softmax概率", num(t.optimal[1])));
  }
  const pInput = slider({ id: "entropy-probability", label: "单步两动作奖励(0,2)：π(奖励2动作)", min: 0.000001, max: 0.999999, step: "any", value: p, format: v => v.toFixed(6), onInput: v => { p = v; renderSoft(); } });
  renderJensen(); renderSoft();
  return [
    proof("凹凸函数不等式", "Jensen：曲线和平均的位置关系",
      equation("f(\\mathbb EX)\\le\\mathbb Ef(X)\\quad(f\\text{凸});\\qquad \\mathbb E\\log X\\le\\log\\mathbb EX\\quad(X>0)",
        "log是凹函数，方向相反。加权两点的平均落在弦上，凹曲线位于弦的上方。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" },
        slider({ id: "jensen-second", label: "两个正数：x₁=1，x₂=", min: 1, max: 12, step: 0.5, value: second, onInput: v => { second = v; renderJensen(); } }),
        slider({ id: "jensen-weight", label: "x₁的概率", min: 0, max: 1, step: 0.05, value: weight, format: num, onInput: v => { weight = v; renderJensen(); } })),
        h("div", { class: "experiment-content" }, plot, jStats)),
      prose("默认两个结果1和9各一半：log5=1.609，大于(log1+log9)/2=1.099，差0.511。把两个结果调相同，或只保留一个结果，差距变0。不能随意把log移进或移出期望。")),
    proof("Jensen应用＋KL恒等式", "把它用到RL：回报与熵的soft目标",
      prose("保持一个状态、两个动作。Q=(0,2)是已知单步收益，温度α>0。我们现在优化EπQ+αH(π)，愿意用一部分原始回报换取分布的随机性；这是加入熵后的新目标。"),
      equation("Z=\\sum_a e^{Q(a)/\\alpha},\\qquad \\alpha\\log Z=\\alpha\\log\\mathbb E_{a\\sim\\pi}\\left[\\frac{e^{Q(a)/\\alpha}}{\\pi(a)}\\right]\\ge\\mathbb E_\\pi Q+\\alpha H(\\pi)",
        "先假设所有π(a)>0：插入概率，把求和写成期望，再用log的Jensen。零概率情况可取极限；H=−Σπlogπ。"),
      equation("\\pi^*(a)=\\frac{e^{Q(a)/\\alpha}}Z,\\qquad \\alpha\\log Z=\\mathbb E_\\pi Q+\\alpha H(\\pi)+\\alpha D_{KL}(\\pi\\Vert\\pi^*)",
        "上界与当前目标的差恰是αKL。KL非负；π=π*时相等，所以softmax最大化这个固定Q的熵目标。"),
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, pInput,
        slider({ id: "entropy-temperature", label: "温度 α", min: 0.25, max: 4, step: 0.25, value: temperature, format: num, onInput: v => { temperature = v; renderSoft(); } }),
        button("策略设为当前 softmax 最优", { onClick: () => { pInput.set(entropyBound([0, 2], [1 - p, p], temperature).optimal[1]); p = Number(pInput.input.value); renderSoft(); } })),
        h("div", { class: "experiment-content" }, softStats)),
      prose("α=1、均匀策略时：EπQ=1、H=0.693、合计1.693；soft价值2.127，差0.434。设为softmax后p=0.881、EπQ=1.762、H=0.365、差0。贪心策略原始回报2更高，但熵目标2低于2.127。",
        "这个精确单步优化说明温度的作用，不是完整SAC训练。多步最大熵RL还要让Q包含后续熵奖励，并在真实动力学下学习critic与actor。")),
    proof("数值稳定技巧", "log-sum-exp：先减最大值，再加回来",
      equation("\\alpha\\log\\sum_a e^{Q(a)/\\alpha}=m+\\alpha\\log\\sum_a e^{(Q(a)-m)/\\alpha},\\qquad m=\\max_aQ(a)",
        "减m后指数不大于1，避免overflow。Q=(1000,1002)、α=1时结果1002.126928，直接算e¹⁰⁰²会溢出。上面的实验也使用这个稳定计算。")),
    h("details", { class: "optional" }, h("summary", {}, "进阶：Jensen怎样产生控制即推断的ELBO？"),
      prose("对未归一化目标密度p*(τ)，任选覆盖它的轨迹分布q(τ)：插入q再应用Jensen，得到下面的下界。它需要q与目标有相应支持，不能把缺失的数据凭空补出。"),
      equation("\\log Z=\\log\\mathbb E_q\\left[\\frac{p^*(\\tau)}{q(\\tau)}\\right]\\ge\\mathbb E_q[\\log p^*(\\tau)-\\log q(\\tau)]",
        "这是ELBO的共同形式。控制即推断中固定真实初始分布和转移，只优化策略时，共同动力学项消去，留下奖励与策略熵。不能让策略自由选择环境后果。"),
      readings("inference")),
    table(["看见的数学工具", "实际作用"], [["log-derivative", "把概率导数写成可采样的PG期望"], ["条件期望为0", "去掉过去奖励，减去动作无关baseline"], ["Jensen与KL", "解释熵目标、下界与softmax"], ["PPO的min/clip", "设计局部更新激励；不是Jensen步骤"]]),
    readings("sac", "inference"),
  ];
}

export function practiceNotes() {
  return [
    prose("训练技巧应对应一个明确问题，并记录它改变了什么。下面的流程用于理解PPO各变量的生命周期；参数选择要通过同预算、多种子的真实执行验证。"),
    table(["环节", "具体做法", "理由/边界"], [
      ["先采数据", "冻结πold采一批；保存旧logπ、旧V、奖励、真终止与最后观察", "一批更新期间旧分母保持固定；本轮完成后重新采样"],
      ["算优势", "用旧V算δ/GAE与critic目标，然后detach", "actor把优势当固定权重；不沿critic或回报反传额外梯度"],
      ["算ratio", "exp(logπnew−logπold)；向量动作的联合log概率先求和", "避免直接概率相除的下溢；不能给每个动作维度独立clip后误当联合ratio"],
      ["更新actor", "最小化−mean(min(wÂ,clip(w)Â))", "优化器通常下降；不要把策略改善的符号写反"],
      ["更新critic", "拟合固定return目标；与actor loss区分", "共享网络时要检查梯度耦合；价值clipping是额外设计选择"],
      ["优势标准化", "常用batch中心化/缩放，记录其范围和时机", "改变有限batch更新与尺度，可能改变优势符号；同批均值不是普遍精确无偏baseline"],
      ["熵与梯度范数", "可加熵项、限制梯度范数", "熵改变目标；范数裁剪改变优化步，均不能代替概率/KL检查"],
      ["限制更新", "监测近似KL、clip fraction，可按KL阈值提前停止", "clip并不硬约束概率比或真实KL；阈值是实践规则"],
      ["验收", "同交互预算，多种子；记录回报、成功率、KL、熵、值损失、优势分布", "loss下降不自动等于策略执行更好；模拟数值不冒充真实训练结果"],
    ]),
    h("pre", { class: "algorithm-flow" }, h("code", {}, [
      "冻结 π_old 与 V_old → 采集 rollout",
      "用最后观察处理 bootstrap → 反向算 GAE / return",
      "固定旧 log-prob、优势与 critic 目标",
      "多轮 minibatch：新 log-prob → ratio → actor / critic / entropy loss",
      "监测 KL 与 clip fraction；需要时停止本批更新",
      "保存诊断与独立评估 → 用更新后的策略采下一批",
    ].join("\n"))),
    readings("ppo", "time", "evaluation"),
  ];
}
