import assert from "node:assert/strict";
import { test, after } from "node:test";
import { build } from "esbuild";
import { parseHTML } from "linkedom";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { CHAPTERS, CHAPTER_IDS, LEGACY_ROUTES, CHAPTER_ROUTE_ALIASES, resolveRoute } from "../src/lessons/curriculum.js";
import { CHAPTER_KNOWLEDGE } from "../src/chapters/knowledge.js";
import { BanditLab, LinearValueLab, PolicyGradientLab, ppoObjective, lambdaWeights } from "../src/rl/teaching-labs.js";
const dir = mkdtempSync(join(tmpdir(), "visualrl-chapters-"));
await build({ entryPoints: [fileURLToPath(new URL("../src/chapters/index.js", import.meta.url))], bundle: true, platform: "node", format: "esm", target: "node24", loader: { ".py": "text" }, outfile: join(dir, "chapters.mjs") });
const { CHAPTER_MODULES } = await import(pathToFileURL(join(dir, "chapters.mjs")));
after(() => rmSync(dir, { recursive: true, force: true }));
function dom(html = "<html><body><main></main></body></html>") {
  const { document, window } = parseHTML(html);
  Object.assign(globalThis, { document, window, Node: window.Node });
  window.scrollTo = () => {}; window.matchMedia = () => ({ matches: false });
  window.HTMLElement.prototype.scrollIntoView = () => {};
  return { document, window };
}
function mount(id) {
  const { document, window } = dom(), root = document.querySelector("main");
  return { root, window, cleanup: CHAPTER_MODULES[id].mount(root) };
}
function click(root, label) {
  const b = [...root.querySelectorAll("button")].find(el => el.textContent.includes(label));
  assert.ok(b, "missing button: " + label); assert.ok(!b.disabled, "disabled button: " + label); b.click();
}
function metric(root, label) {
  const el = [...root.querySelectorAll(".metric")].find(el => el.querySelector("span")?.textContent === label);
  assert.ok(el, "missing metric: " + label); return el.querySelector("strong").textContent;
}
function range(root, window, id, value) {
  const input = root.querySelector("#" + id); assert.ok(input);
  input.value = String(value); input.dispatchEvent(new window.Event("input"));
}
test("ten textbook chapters have a single hierarchy, visible formulas, and complete mastery evidence", () => {
  assert.deepEqual(CHAPTER_IDS, ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]);
  assert.equal(CHAPTERS.at(-1).group, "研究专题");
  for (const c of CHAPTERS) {
    const { root, cleanup } = mount(c.id);
    assert.equal(root.querySelectorAll("h1").length, 1, "multiple chapter titles: " + c.id);
    assert.equal(root.querySelector("h1").textContent, c.title);
    assert.equal(root.querySelectorAll(".chapter-topic").length, c.topics.length);
    assert.equal(root.querySelectorAll(".chapter-toc a").length, c.topics.length);
    assert.equal(root.querySelector(".chapter-connection").textContent, "为什么学到这里：" + c.relation.from);
    assert.ok(root.querySelector(".chapter-transition").textContent.includes(c.relation.next));
    assert.equal(root.querySelectorAll(".mastery-row").length, CHAPTER_KNOWLEDGE[c.id].length);
    for (const [, evidence] of CHAPTER_KNOWLEDGE[c.id]) {
      assert.ok([...root.querySelectorAll(".chapter-topic")].some(el => el.dataset.topic === evidence));
    }
    assert.ok([...root.querySelectorAll(".equation math")].some(el => !el.closest("details")), "no visible math: " + c.id);
    assert.equal(root.querySelectorAll(".katex-error,.callout.error").length, 0);
    assert.equal(root.querySelectorAll(".lesson-head:not(.chapter-head),.lesson-foot:not(.chapter-foot)").length, 0);
    assert.equal(root.querySelector(".next-lesson").getAttribute("href"), CHAPTERS[c.number]?.href ?? "#");
    cleanup();
  }
});
test("task, valuation and Bellman have separate chapters, with exploration attached to control", () => {
  const first = mount("01"), values = mount("02"), dp = mount("03"), td = mount("05"), frontier = mount("10");
  assert.deepEqual(CHAPTERS[0].topics.map(t => t.id), ["task", "state", "interaction"]);
  assert.ok(!first.root.textContent.includes("Bellman 最优方程"));
  assert.ok(!first.root.textContent.includes("V 和 Q"));
  assert.ok(values.root.textContent.includes("V 和 Q"));
  assert.ok(values.root.textContent.includes("我们的任务：找一个最优策略"));
  assert.ok(!values.root.textContent.includes("Bellman 最优方程"));
  assert.ok(!values.root.textContent.includes("赌博机"));
  assert.ok(dp.root.textContent.includes("Bellman 最优方程"));
  assert.ok(dp.root.textContent.includes("继续更新到稳定"));
  assert.ok(dp.root.textContent.includes("改进：切换到最好的动作"));
  assert.ok(td.root.querySelector(".chapter-topic[data-topic='exploration'] .bandit-supplement"));
  assert.ok(td.root.querySelector(".bandit-supplement").textContent.includes("单步特例"));
  assert.ok(frontier.root.textContent.includes("GCRL"));
  assert.ok(frontier.root.textContent.includes("Contrastive RL"));
  assert.ok(frontier.root.textContent.includes("stitching"));
  for (const [key, [id, topic]] of Object.entries({ ...LEGACY_ROUTES, ...CHAPTER_ROUTE_ALIASES })) {
    assert.equal(resolveRoute(key).chapter.id, id); assert.equal(resolveRoute(key).topic, topic);
  }
  assert.equal(resolveRoute("chapter-01/returns").chapter.id, "02");
  assert.equal(resolveRoute("chapter-01/returns").topic, "returns");
  assert.equal(resolveRoute("chapter-99").chapter, null);
  [first, values, dp, td, frontier].forEach(page => page.cleanup());
});
test("bandit updates preserve per-action sample means and UCB tries unobserved actions", () => {
  const lab = new BanditLab();
  assert.equal(lab.choose({ mode: "ucb" }), 0);
  const rewards = [];
  for (let i = 0; i < 10; i++) rewards.push(lab.pull(0).reward);
  assert.equal(lab.choose({ mode: "ucb" }), 1);
  assert.ok(Math.abs(lab.Q[0] - rewards.reduce((sum, r) => sum + r, 0) / 10) < 1e-12);
  assert.equal(lab.N[1], 0); assert.equal(lab.Q[1], 0);
  const { root } = mount("05"); click(root, "手动试 A"); assert.equal(metric(root, "真实尝试"), "1");
  click(root, "按当前规则试10次"); assert.equal(metric(root, "真实尝试"), "11");
  click(root, "交换真实均值"); assert.equal(metric(root, "真实尝试"), "11");
  click(root, "切换均值/α=0.1并清零");
  assert.equal(metric(root, "真实尝试"), "0");
  assert.ok(root.textContent.includes("为 0.75"));
  const changed = new BanditLab({ means: [0.75, 0.35] });
  assert.deepEqual(changed.means, [0.75, 0.35]);
});
test("Monte Carlo waits for termination and first/every visits use the appropriate returns", () => {
  const { root } = mount("04");
  const mcButton = [...root.querySelectorAll("button")].find(b => b.textContent.includes("MC：学习完整"));
  assert.equal(mcButton.disabled, true);
  click(root, "切换5步");
  for (let i = 0; i < 7; i++) click(root, "采集下一步");
  click(root, "MC：学习完整");
  assert.equal(metric(root, "MC · V(S)"), "0.531");
  assert.equal(metric(root, "MC实际更新次数"), "5");
  click(root, "每访 MC");
  for (let i = 0; i < 7; i++) click(root, "采集下一步");
  click(root, "MC：学习完整");
  assert.equal(metric(root, "MC · V(S)"), "0.656");
  assert.equal(metric(root, "MC实际更新次数"), "7");
});
test("multi-step targets and eligibility traces agree at the TD and MC endpoints", () => {
  const { root, window } = mount("06");
  assert.equal(metric(root, "当前 n-step 目标"), "0.45");
  range(root, window, "n-step-length", 5); assert.equal(metric(root, "当前 n-step 目标"), "0.656");
  range(root, window, "trace-lambda", 0);
  assert.equal(metric(root, "λ-return目标"), "0.45");
  for (let i = 0; i < 5; i++) click(root, "TD(λ)");
  assert.equal(metric(root, "TD(λ) · V(S)"), "0");
  range(root, window, "trace-lambda", 1);
  assert.equal(metric(root, "λ-return目标"), "0.656");
  for (let i = 0; i < 5; i++) click(root, "TD(λ)");
  assert.equal(metric(root, "TD(λ) · V(S)"), "0.656");
  assert.deepEqual(lambdaWeights(5, 0), [1, 0, 0, 0, 0]);
  assert.deepEqual(lambdaWeights(5, 1), [0, 0, 0, 0, 1]);
  assert.ok(Math.abs(lambdaWeights(5, 0.8).reduce((s, x) => s + x, 0) - 1) < 1e-12);
});
test("Dyna planning propagates observed rewards without adding real transitions or model edges", () => {
  const { root, cleanup } = mount("07");
  for (let i = 0; i < 8; i++) click(root, "Dyna：真实行动一步");
  assert.equal(metric(root, "Dyna真实环境转移"), "8");
  assert.equal(metric(root, "模型中的连接"), "8");
  assert.equal(metric(root, "Dyna · max Q(S)"), "0");
  click(root, "只从模型规划200次");
  assert.equal(metric(root, "Dyna · max Q(S)"), "4.783");
  assert.equal(metric(root, "Dyna模拟更新"), "200");
  assert.equal(metric(root, "Dyna真实环境转移"), "8");
  assert.equal(metric(root, "模型中的连接"), "8"); cleanup();
});
test("shared features produce generalization and conflicting targets reveal representation limits", () => {
  const independent = new LinearValueLab("table"); independent.learn(0, 1);
  assert.deepEqual(independent.predictions(), [0.5, 0, 0]);
  const similar = new LinearValueLab(); similar.learn(0, 1);
  assert.deepEqual(similar.predictions(), [0.5, 0.4, 0]);
  const shared = new LinearValueLab("shared"); shared.learn(0, 1); shared.learn(2, 0);
  assert.deepEqual(shared.predictions(), [0.25, 0.25, 0.25]);
  const { root } = mount("08"); click(root, "观察 A");
  assert.ok(root.textContent.includes("新预测(A,B,C)=(0.5, 0.4, 0)"));
  click(root, "完全共享"); click(root, "观察 A"); click(root, "观察 C");
  assert.ok(root.textContent.includes("新预测(A,B,C)=(0.25, 0.25, 0.25)"));
});
test("policy-gradient probability follows advantage while PPO handles both advantage signs", () => {
  const positive = new PolicyGradientLab(); positive.sampleUpdate(1, 4.145); assert.ok(positive.p > 0.5);
  const negative = new PolicyGradientLab(); negative.sampleUpdate(1, 8); assert.ok(negative.p < 0.5);
  const near = new PolicyGradientLab(); near.sampleUpdate(0, 4.145); assert.ok(near.p > 0.5);
  assert.equal(ppoObjective(1.5, 2), 2.4);
  assert.equal(ppoObjective(0.5, -2), -1.6);
  assert.equal(ppoObjective(1.5, -2), -3);
  const { root, window } = mount("09");
  click(root, "观察→"); assert.ok(Number(metric(root, "πθ(→|J)")) > 0.5);
  click(root, "策略恢复"); range(root, window, "pg-baseline", 8); click(root, "观察→");
  assert.ok(Number(metric(root, "πθ(→|J)")) < 0.5);
  range(root, window, "ppo-ratio", 1.5);
  assert.equal(metric(root, "PPO clipped目标"), "2.4");
  click(root, "负优势"); assert.equal(metric(root, "PPO clipped目标"), "-3");
});
test("public navigation resolves old links and preserves an experiment when jumping within a chapter", async () => {
  await build({ entryPoints: [fileURLToPath(new URL("../src/main.js", import.meta.url))], bundle: true, platform: "node", format: "esm", target: "node24", loader: { ".py": "text", ".css": "empty" }, outfile: join(dir, "app.mjs") });
  const { document, window } = dom("<html><body><div id='app'></div></body></html>");
  Object.assign(globalThis, { location: { hash: "#lesson-05" }, history: { pushState: (_, __, href) => { location.hash = href; } } });
  await import(pathToFileURL(join(dir, "app.mjs")));
  const root = document.querySelector("#app");
  assert.equal(root.querySelector("h1").textContent, CHAPTERS[1].title);
  click(root, "策略 C"); assert.equal(metric(root, "从 S 出发的回报"), "4.783");
  function follow(href) {
    const a = [...root.querySelectorAll("a")].find(a => a.getAttribute("href") === href); assert.ok(a, "missing link: " + href);
    const event = new window.Event("click", { bubbles: true, cancelable: true }); Object.defineProperty(event, "button", { value: 0 }); a.dispatchEvent(event);
  }
  follow("#chapter-02/returns");
  assert.equal(metric(root, "从 S 出发的回报"), "4.783");
  follow("#chapter-03"); assert.equal(root.querySelector("h1").textContent, CHAPTERS[2].title);
  follow("#");
  assert.equal(root.querySelectorAll(".lesson-row").length, 10);
  assert.equal(root.querySelectorAll(".callout.error,.error-banner").length, 0);
});
