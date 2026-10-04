import assert from "node:assert/strict";
import { test, after } from "node:test";
import { build } from "esbuild";
import { parseHTML } from "linkedom";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { baselineStatistics, generalizedAdvantages, surrogateExample, logJensen, entropyBound, shapeRewards, doubleDqnTarget } from "../src/rl/policy-math.js";
const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) < tolerance, actual + " != " + expected);
test("baseline preserves the logit-gradient mean while its score-weighted optimum minimizes variance", () => {
  for (const p of [0.05, 0.2, 0.5, 0.8, 0.95]) {
    for (const b of [0, 4.145, 9]) close(baselineStatistics(p, b).mean, p * (1 - p) * 6.29);
    const t = baselineStatistics(p, 0);
    close(baselineStatistics(p, t.optimalBaseline).variance, 0);
  }
  close(baselineStatistics(0.5, 0).variance, 4.29525625);
  const t = baselineStatistics(0.8, 0);
  close(t.value, 6.032); close(t.optimalBaseline, 2.258);
  assert.ok(baselineStatistics(0.8, t.value).variance > 0);
});
test("GAE bootstraps at truncation without leaking reset-episode residuals", () => {
  const example = (terminated, lambda = 0.8) => generalizedAdvantages({ rewards: [0, 1], values: [0.4, 0.7, 2], terminated: [false, terminated], continuations: [true, false], lambda });
  close(example(true).advantages[0], 0.446); close(example(false).advantages[0], 1.742);
  close(example(true, 0).advantages[0], 0.23); close(example(true, 1).advantages[0], 0.9 - 0.4);
  close(example(false, 1).advantages[0], 0.9 + 0.81 * 2 - 0.4);
  const joined = generalizedAdvantages({ rewards: [0, 1, 999], values: [0.4, 0.7, 2, 0], terminated: [false, false, true], continuations: [true, false, false] });
  close(joined.advantages[0], 1.742);
});
test("the surrogate matches value and gradient at the old policy, but is not a global return bound", () => {
  close(surrogateExample(0.5, 0.5).actual, 0.5); close(surrogateExample(0.5, 0.5).surrogate, 0.5);
  for (const index of [0, 1]) {
    const a = [0.5, 0.5], b = [0.5, 0.5], step = 1e-5; a[index] += step; b[index] -= step;
    const hi = surrogateExample(...a), lo = surrogateExample(...b);
    close((hi.actual - lo.actual) / (2 * step), (hi.surrogate - lo.surrogate) / (2 * step), 1e-8);
  }
  const above = surrogateExample(0.1, 0.9), below = surrogateExample(0.9, 0.9);
  close(above.actual, 1.044); close(above.surrogate, 1.62); assert.ok(above.surrogate > above.actual);
  close(below.actual, 1.396); close(below.surrogate, 0.82); assert.ok(below.surrogate < below.actual);
});
test("log Jensen has the concave direction and entropy's exact gap is temperature times KL", () => {
  close(logJensen(1, 9, 0.5).gap, Math.log(5 / 3)); close(logJensen(1, 1, 0.3).gap, 0);
  assert.throws(() => logJensen(0, 9, 0.5));
  for (const temperature of [0.25, 1, 4]) {
    for (const p of [0, 0.000001, 0.5, 0.999999, 1]) {
      const t = entropyBound([0, 2], [1 - p, p], temperature);
      assert.ok(t.gap >= -1e-12); close(t.gap, temperature * t.kl);
      close(entropyBound([0, 2], t.optimal, temperature).gap, 0);
    }
  }
  close(entropyBound([1000, 1002], [0.5, 0.5], 1).softValue, 1002.126928011043);
});
test("potential shaping telescopes and a nonzero terminal boundary can reverse route preference", () => {
  const near = shapeRewards([0, 0, 0, 0, 1], [-3, -4, -3, -2, -1, 0]);
  const far = shapeRewards([0, 0, 0, 0, 0, 0, 0, 10], [-3, -4, -3, -2, -1, -2, -2, -1, 0]);
  close(near.shapedReturn - near.originalReturn, 3); close(far.shapedReturn - far.originalReturn, 3);
  assert.ok(far.shapedReturn > near.shapedReturn);
  const badNear = shapeRewards([0, 0, 0, 0, 1], [-3, -4, -3, -2, -1, 30]);
  const badFar = shapeRewards([0, 0, 0, 0, 0, 0, 0, 10], [-3, -4, -3, -2, -1, -2, -2, -1, 30]);
  assert.ok(badNear.shapedReturn > badFar.shapedReturn);
  close(badNear.shapedReturn - badNear.originalReturn, badNear.boundary);
});
test("Double DQN selects with the online network and terminal targets contain no bootstrap", () => {
  const t = doubleDqnTarget([8, 7], [2, 6]); close(t.dqn, 6.4); close(t.double, 2.8); assert.equal(t.action, 0);
  const same = doubleDqnTarget([1, 7], [2, 6]); close(same.dqn, same.double);
  const terminal = doubleDqnTarget([8, 7], [2, 6], { terminated: true }); close(terminal.dqn, 1); close(terminal.double, 1);
});
const dir = mkdtempSync(join(tmpdir(), "visualrl-policy-tools-"));
await build({ entryPoints: [fileURLToPath(new URL("../src/extensions/textbook-course/index.js", import.meta.url))], bundle: true, platform: "node", format: "esm", target: "node24", loader: { ".py": "text" }, outfile: join(dir, "chapters.mjs") });
const { CHAPTER_MODULES } = await import(pathToFileURL(join(dir, "chapters.mjs")));
after(() => rmSync(dir, { recursive: true, force: true }));
test("teaching controls demonstrate baseline, GAE, state shift, Jensen, shaping and Double DQN", () => {
  const { document, window } = parseHTML("<!doctype html><html><body><main></main></body></html>");
  Object.assign(globalThis, { document, window, Node: window.Node });
  const root = document.querySelector("main");
  const click = label => { const b = [...root.querySelectorAll("button")].find(b => b.textContent.includes(label)); assert.ok(b, label); b.click(); };
  const metric = label => { const el = [...root.querySelectorAll(".metric")].find(el => el.querySelector("span")?.textContent === label); assert.ok(el, label); return el.querySelector("strong").textContent; };
  const input = (id, value) => { const el = root.querySelector("#" + id); assert.ok(el, id); el.value = String(value); el.dispatchEvent(new window.Event("input")); };
  let cleanup = CHAPTER_MODULES["09"].mount(root);
  click("baseline设为 Vπ"); assert.equal(metric("梯度样本的精确方差"), "0");
  input("variance-probability", 0.8); assert.ok(Number(metric("梯度样本的精确方差")) > 0);
  click("设为最小方差"); assert.equal(metric("梯度样本的精确方差"), "0");
  assert.equal(metric("GAE · A₀"), "0.446"); click("外部时间截断"); assert.equal(metric("GAE · A₀"), "1.742");
  input("surrogate-enter", 0.1); input("surrogate-good", 0.9);
  assert.equal(metric("新策略真实 J"), "1.044"); assert.equal(metric("固定旧访问分布的 L"), "1.62");
  input("jensen-second", 1); assert.equal(metric("Jensen差距"), "0");
  click("策略设为当前 softmax"); close(Number(metric("π(奖励2动作)")), 0.88079707797, 1e-6);
  close(Number(metric("上界−当前目标")), 0, 0.001);
  input("entropy-temperature", 0.25); click("策略设为当前 softmax");
  assert.ok(Number(metric("π(奖励2动作)")) > 0.99);
  assert.equal(root.querySelectorAll(".katex-error").length, 0); cleanup(); root.replaceChildren();
  cleanup = CHAPTER_MODULES["02"].mount(root);
  assert.equal(metric("塑形后两候选比较"), "远路更好"); click("反例：结束回合"); assert.equal(metric("塑形后两候选比较"), "近路更好"); cleanup(); root.replaceChildren();
  CHAPTER_MODULES["08"].mount(root);
  assert.equal(metric("Double DQN目标"), "2.8"); click("两个网络同意"); assert.equal(metric("Double DQN目标"), "6.4"); click("切换真正终止"); assert.equal(metric("Double DQN目标"), "1");
});
