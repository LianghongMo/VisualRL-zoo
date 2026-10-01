import { h, button, slider, replace } from "../ui/dom.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { Trajectory } from "../rl/core.js";
import { NStepTD } from "../rl/tabular/n-step-td.js";
import { lambdaWeights, traceStep } from "../rl/teaching-labs.js";
import { route, NEAR_ROUTE, routeStrip, num, metric } from "../ui/world-view.js";
import { chapterHeader, chapterFooter, topic, prose, equation, table } from "./shell.js";
export function mount(root) {
  const env = GridWorld.chargingRoom(), full = route(env, NEAR_ROUTE), episode = new Trajectory(full);
  let n = 1, lambda = 0.8, V = Array(env.nStates).fill(0.5), traceLearner, traceIndex = 0;
  env.goals.forEach(goal => { V[goal] = 0; });
  const targetReadout = h("div"), weightsReadout = h("div"), predictions = h("div"), traceReadout = h("div"), traceStats = h("div", { class: "metrics" });
  function nTarget(length) {
    const learner = new NStepTD({ nStates: env.nStates, n: length, alpha: 1, gamma: 0.9 });
    learner.V = V; return learner.target(episode, 0);
  }
  function renderTargets() {
    const target = nTarget(n);
    replace(targetReadout, h("div", { class: "metrics" }, metric("n-step真实奖励数", target.rewards.length), metric("截点后继估计", target.bootstrap_state === null ? "终止，无后续" : num(target.bootstrap_value)), metric("当前 n-step 目标", num(target.target))),
      h("p", { class: "arithmetic" }, "实际奖励 " + target.rewards.join(", ") + "；目标 = 折扣奖励和 + " + num(target.bootstrap_weight) + " × " + num(target.bootstrap_value) + " = " + num(target.target)));
    const weights = lambdaWeights(full.length, lambda), targets = weights.map((_, i) => nTarget(i + 1).target);
    const mixed = targets.reduce((sum, target, i) => sum + weights[i] * target, 0);
    replace(weightsReadout, table(["目标长度", "同一份旧V下的目标", "λ权重", "贡献"], weights.map((w, i) => [i + 1, num(targets[i]), num(w), num(w * targets[i])])),
      metric("λ-return目标", num(mixed)), h("p", { class: "observation" }, "全部权重之和 = " + num(weights.reduce((s, w) => s + w, 0)) + "。λ=0只取一步目标；λ=1只取完整MC回报。"));
    replace(predictions, routeStrip(env, full, { values: V, label: "比较目标的实验：非终点旧估计初始为0.5，终点为0。箭头是实际奖励，数字是当前V；终止状态不自举。" }));
  }
  function resetTraces() {
    traceLearner = { V: Array(env.nStates).fill(0), e: Array(env.nStates).fill(0), alpha: 1, gamma: 0.9, lambda };
    traceIndex = 0; renderTraces();
  }
  const traceButton = button("TD(λ)：执行并学习一步", { kind: "learn", onClick: () => {
    if (traceIndex >= full.length) return;
    traceStep(traceLearner, full[traceIndex++]); renderTraces();
  } });
  function renderTraces() {
    traceButton.disabled = traceIndex >= full.length;
    replace(traceReadout, routeStrip(env, full, { values: traceLearner.V, visited: traceIndex, maskFuture: true, label: "资格迹实验从V=0开始。最后的+1误差会同时更新仍有资格迹的过去状态。" }),
      table(["访问位置", "当前资格迹 e(s)", "当前V(s)"], full.map(t => [t.state === env.start ? "S" : env.toCell(t.state).join(","), num(traceLearner.e[t.state]), num(traceLearner.V[t.state])])));
    replace(traceStats, metric("真实行动与更新", traceIndex), metric("资格迹实验 λ", num(traceLearner.lambda)), metric("TD(λ) · V(S)", num(traceLearner.V[env.start])));
  }
  const nInput = slider({ id: "n-step-length", label: "看 n 个真实奖励", min: 1, max: 5, step: 1, value: n, onInput: v => { n = v; renderTargets(); } });
  const lambdaInput = slider({ id: "trace-lambda", label: "λ：长目标与访问记忆的影响", min: 0, max: 1, step: 0.05, value: lambda, format: v => v.toFixed(2), onInput: v => { lambda = v; renderTargets(); resetTraces(); } });
  root.append(chapterHeader("06"),
    topic("06", "n-step",
      h("div", { class: "experiment" }, h("div", { class: "toolbar" }, nInput,
        button("用 n-step 目标更新 S", { onClick: () => { V[env.start] = nTarget(n).target; renderTargets(); } }),
        button("重置旧估计为0.5", { kind: "ghost", onClick: () => { V.fill(0.5); env.goals.forEach(goal => { V[goal] = 0; }); renderTargets(); } })),
        h("div", { class: "experiment-content" }, predictions, targetReadout)),
      prose("TD(0)只看一个真实奖励就接估计；MC一直看到终止。n-step在两者之间选一个截点：先累计n步真奖励，截点未终止时才读V补上未来。本例先提供完整5步样本，单独比较这些目标。",
        "在线使用时，n-step通常要等n个奖励才能更新起点；若回合先终止，直接用已知的实际回报，不再等待。更长目标使用更多实际经历，也可能带来更长等待和更多采样波动。"),
      equation("G_t^{(n)}=\\sum_{k=0}^{m-1}\\gamma^k r_{t+k}+\\mathbf1\\{t+n<T\\}\\gamma^n V(s_{t+n}),\\qquad m=\\min(n,T-t)",
        "n=1是TD目标；n达到终止时就是MC回报。真实终止遮掉后续项，时间截断则要说明如何估计剩余未来。"),
      equation("V(s_t)\\leftarrow V(s_t)+\\alpha\\big[G_t^{(n)}-V(s_t)\\big]",
        "目标长度变了，朝目标移动的更新结构没有变。扩展到n-step SARSA时，未终止的尾部换成Q(sₜ₊ₙ,aₜ₊ₙ)。")),
    topic("06", "lambda", h("div", { class: "experiment" }, h("div", { class: "toolbar" }, lambdaInput), h("div", { class: "experiment-content" }, weightsReadout)),
      prose("不一定只选一个n。λ-return把不同长度目标加权，短目标权重随长度衰减；最后一个目标已经覆盖完整回合，收走剩余权重。"),
      equation("G_t^\\lambda=(1-\\lambda)\\sum_{n=1}^{T-t-1}\\lambda^{n-1}G_t^{(n)}+\\lambda^{T-t-1}G_t",
        "有限终止回合的前向视角。改变λ会重置下面的资格迹实验。这里各n目标都读取同一份旧V；最后一项是完整实际回报，权重不能再乘(1−λ)。")),
    topic("06", "traces", h("div", { class: "experiment" }, h("div", { class: "toolbar" }, traceButton, button("资格迹实验重新开始", { kind: "ghost", onClick: resetTraces })),
      h("div", { class: "experiment-content" }, traceStats, traceReadout)),
      prose("后向视角不在每次更新时重新枚举过去所有目标，而是维护访问记忆e(s)。经过一个状态就给它增加资格，随后每步衰减；当前TD误差同时更新所有还有资格的状态。"),
      equation("\\delta_t=r_t+\\gamma V(s_{t+1})-V(s_t),\\qquad e_t(s)=\\gamma\\lambda e_{t-1}(s)+\\mathbf1\\{s=s_t\\}",
        "这是累积资格迹。真正终止时δ的后继值取0；新回合开始前清空资格迹。"),
      equation("V(s)\\leftarrow V(s)+\\alpha\\delta_t e_t(s)\\quad\\text{for all }s",
        "λ=0只更新当前状态。这个从零开始的独特路线中，最后δ=1：S的更新是(γλ)⁴；λ=0.8得到0.269，λ=1得到0.656。"),
      prose("前向混合与后向记忆描述同一种信用分配思路。冻结或离线条件下有相应等价关系；在线更新会改变中间价值，普通累积TD(λ)不能对任意步长声称严格等于所有在线前向目标。True online TD(λ)专门处理这种精确对应。这里的唯一访问、零中间误差样本可以直接逐项验算。",
        "本章先讲表格资格迹，把教材第12章的这个基本思想提前接到n-step；函数近似下的迹和更一般的异策略修正是后续阅读。")),
    chapterFooter("06", "n选择读取多少真实未来，λ混合不同长度目标，资格迹把一条当前误差分给最近的访问。先辨认目标、等待时间和终止，再区分前向解释与实际后向更新。"));
  renderTargets(); resetTraces(); return () => {};
}
