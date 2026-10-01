import assert from "node:assert/strict";
import { test, after } from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { build } from "esbuild";
import { parseHTML } from "linkedom";
import { MAIN_IDS } from "../src/lessons/curriculum.js";
import { KNOWLEDGE } from "../src/lessons/knowledge.js";
const dir = mkdtempSync(join(tmpdir(), "visualrl-course-"));
await build({ stdin: { contents: [...MAIN_IDS, "06"].map((id) => `export * as l${id} from './lesson${id}.js';`).join("\n"), resolveDir: fileURLToPath(new URL("../src/lessons", import.meta.url)) }, bundle: true, platform: "node", format: "esm", target: "node24", loader: { ".py": "text" }, outfile: join(dir, "lessons.mjs") });
const lessons = await import(pathToFileURL(join(dir, "lessons.mjs")));
after(() => rmSync(dir, { recursive: true, force: true }));
function mount(id) {
  const { document, window } = parseHTML("<html><body><main></main></body></html>");
  Object.assign(globalThis, { document, window, Node: window.Node });
  const root = document.querySelector("main"), cleanup = lessons[`l${id}`].mount(root); return { root, cleanup };
}
function button(root, label) {
  const b = [...root.querySelectorAll("button")].find((b) => b.textContent.includes(label)); assert.ok(b, `missing button: ${label}`); return b;
}
function click(root, label) { const b = button(root, label); assert.ok(!b.disabled, `disabled: ${label}`); b.click(); }
function metric(root, label) {
  const m = [...root.querySelectorAll(".metric")].find((m) => m.querySelector("span")?.textContent === label); assert.ok(m, `missing metric: ${label}`); return m.querySelector("strong").textContent;
}
test("chapter order, chapter essentials and next links follow prerequisites", () => {
  assert.deepEqual(MAIN_IDS, ["01", "04", "05", "02", "07", "08", "03"]);
  for (let i = 0; i < MAIN_IDS.length; i++) {
    const { root, cleanup } = mount(MAIN_IDS[i]); assert.equal(root.querySelectorAll("h1").length, 1);
    assert.ok(root.querySelector(".learning-goal")); assert.ok(root.querySelector(".takeaway"));
    assert.equal(root.querySelectorAll(".essentials li").length, KNOWLEDGE[MAIN_IDS[i]].length);
    const sections = [...root.querySelectorAll(".step > h2")].map(el => el.textContent);
    for (const [, evidence] of KNOWLEDGE[MAIN_IDS[i]]) assert.ok(sections.includes(evidence), `missing evidence section: ${evidence}`);
    assert.equal(root.querySelectorAll(".mastery-row input[type=checkbox]").length, KNOWLEDGE[MAIN_IDS[i]].length);
    const visibleEquations = [...root.querySelectorAll(".equation")].filter(el => !el.closest("details"));
    assert.ok(visibleEquations.length >= 1, `chapter ${MAIN_IDS[i]} has no visible equation`);
    assert.equal(root.querySelectorAll(".katex-error").length, 0);
    assert.equal(root.querySelector(".next-lesson").getAttribute("href"), i + 1 < MAIN_IDS.length ? `#lesson-${MAIN_IDS[i + 1]}` : "#"); cleanup?.();
  }
  assert.equal(mount("06").root.querySelector(".next-lesson").getAttribute("href"), "#lesson-02");
});
test("environment represents wall collision, cliff reset, movement and termination", () => {
  const { root, cleanup } = mount("01"); click(root, "↓ 下"); assert.ok(root.textContent.includes("撞到墙"));
  click(root, "→ 右"); assert.ok(root.textContent.includes("奖励是 −10")); assert.equal(metric(root, "现在在哪 · 新状态 s′"), "出发点");
  click(root, "重新出发"); for (const l of ["↑ 上", "→ 右", "→ 右", "→ 右", "↓ 下"]) click(root, l);
  assert.equal(metric(root, "本次走了"), "5 步"); assert.equal(metric(root, "本次奖励合计"), "+1"); assert.equal(button(root, "↑ 上").disabled, true); cleanup();
});
test("discount changes route preference while preserving raw rewards", () => {
  const { root } = mount("04"); assert.equal(metric(root, "近处：5 步，奖励 +1"), "0.656"); assert.equal(metric(root, "远处：8 步，奖励 +10"), "4.783");
  const range = root.querySelector("input[type=range]"); range.value = "0.30"; range.dispatchEvent(new window.Event("input"));
  assert.equal(metric(root, "此时回报更大"), "近处 +1"); assert.ok(root.querySelector("svg").textContent.includes("+10"));
});
test("synchronous value backups reach the dock after 5 and 8 rounds", () => {
  const { root } = mount("05"); for (let i = 0; i < 4; i++) click(root, "更新所有位置一轮"); assert.equal(metric(root, "出发点价值 V(S)"), "0");
  click(root, "更新所有位置一轮"); assert.equal(metric(root, "出发点价值 V(S)"), "0.656");
  for (let i = 0; i < 3; i++) click(root, "更新所有位置一轮"); assert.equal(metric(root, "出发点价值 V(S)"), "4.783");
});
test("acting only adds experience and learning only propagates values", () => {
  const { root } = mount("02"); for (let i = 0; i < 5; i++) click(root, "行动：沿路线走一步");
  assert.equal(metric(root, "V(出发点)"), "0"); assert.equal(metric(root, "学习"), "0 轮");
  for (let i = 0; i < 5; i++) click(root, "学习：沿已知边更新一轮");
  assert.equal(metric(root, "V(出发点)"), "0.656"); assert.equal(metric(root, "行动"), "5 步"); assert.equal(metric(root, "已知连接"), "5");
});
test("MC waits for termination and reversed TD replay reaches the same observed return", () => {
  const { root } = mount("07"); for (let i = 0; i < 3; i++) click(root, "沿固定路线走一步");
  assert.equal(button(root, "MC：学完整回合").disabled, true); click(root, "TD：学习选中的一步"); assert.equal(metric(root, "TD 更新"), "1");
  for (let i = 0; i < 2; i++) click(root, "沿固定路线走一步"); click(root, "MC：学完整回合"); assert.equal(metric(root, "MC 更新"), "5");
  for (let i = 0; i < 5; i++) click(root, "TD：从终点向前重放一条");
  for (const card of root.querySelectorAll(".comparison-card")) assert.ok(card.querySelector(".route-strip").textContent.includes("0.656"));
  assert.equal(button(root, "TD：从终点向前重放一条").disabled, true);
});
test("SARSA and Q-learning differ for exploratory actions and match for greedy actions", () => {
  const { root } = mount("08"); click(root, "用这条经验分别更新一次"); const cards = root.querySelectorAll(".comparison-card");
  assert.ok(cards[0].textContent.includes("-60.5")); assert.ok(cards[1].textContent.includes("-15.5"));
  click(root, "下一步向右"); click(root, "用这条经验分别更新一次"); for (const c of root.querySelectorAll(".comparison-card")) assert.ok(c.textContent.includes("-15.5"));
});
test("offline stitching uses existing connections and online collection fills missing coverage", () => {
  const { root } = mount("03"); click(root, "离线：只用这些记录规划"); assert.equal(metric(root, "规划出的路线"), "8 步 / 4.783");
  click(root, "只保留记录 A"); click(root, "离线：只用这些记录规划"); assert.equal(metric(root, "规划出的路线"), "5 步 / 0.656");
  for (let i = 0; i < 8; i++) click(root, "在线：沿示范路线采集一步");
  assert.equal(metric(root, "V(S)"), "0.656"); assert.equal(metric(root, "新增行动"), "8 步");
  click(root, "离线：只用这些记录规划"); assert.equal(metric(root, "规划出的路线"), "8 步 / 4.783");
});
test("Q example preserves estimates between clicks and masks bootstrap at true termination", () => {
  const { root, cleanup } = mount("08");
  click(root, "用这条经验分别更新一次"); click(root, "用这条经验分别更新一次");
  let cards = root.querySelectorAll(".comparison-card");
  assert.ok(cards[0].textContent.includes("-60.5 → -80.75"));
  assert.ok(cards[1].textContent.includes("-15.5 → -13.25"));
  click(root, "终止经验 · 后续项"); click(root, "用这条经验分别更新一次");
  cards = root.querySelectorAll(".comparison-card");
  for (const card of cards) assert.ok(card.textContent.includes("-20 → -10.5"));
  click(root, "用这条经验分别更新一次");
  for (const card of root.querySelectorAll(".comparison-card")) assert.ok(card.textContent.includes("-10.5 → -5.75")); cleanup();
});
test("MC/TD hide unobserved rewards, respect alpha and retain estimates across episodes", () => {
  const { root } = mount("07");
  assert.equal(root.querySelectorAll(".route-figure")[0].querySelectorAll(".route-edge b")[4].textContent, "?");
  const range = root.querySelector("#prediction-alpha"); range.value = "0.5"; range.dispatchEvent(new window.Event("input"));
  for (let i = 0; i < 5; i++) click(root, "沿固定路线走一步");
  click(root, "MC：学完整回合"); assert.equal(metric(root, "MC · V(S)"), "0.328"); assert.equal(button(root, "MC：学完整回合").disabled, true);
  for (let i = 0; i < 5; i++) click(root, "TD：从终点向前重放一条");
  assert.equal(metric(root, "TD · V(S)"), "0.021");
  click(root, "保留估计，再走一回合"); assert.equal(metric(root, "MC · V(S)"), "0.328"); assert.equal(metric(root, "已收集经验"), "0 / 5 步");
  click(root, "重新开始"); assert.equal(metric(root, "TD · V(S)"), "0"); assert.equal(metric(root, "MC · V(S)"), "0");
});
test("stitching propagates at the junction and supplies a provenance record for every new edge", () => {
  const { root } = mount("03");
  for (let i = 0; i < 4; i++) click(root, "离线：只更新一轮"); assert.equal(metric(root, "V(J)"), "7.29"); assert.equal(metric(root, "V(S)"), "0");
  for (let i = 0; i < 4; i++) click(root, "离线：只更新一轮"); assert.equal(metric(root, "规划出的路线"), "8 步 / 4.783");
  const table = root.querySelector(".experiment-content .table-wrap table"), rows = table.querySelectorAll("tbody tr");
  assert.equal(rows.length, 8);
  for (let i = 0; i < 8; i++) assert.ok(rows[i].lastElementChild.textContent.includes(i < 4 ? "A" : "B"));
  click(root, "同一位置，但电量不同"); assert.ok(root.textContent.includes("完整状态不同"));
  click(root, "只保留记录 A"); click(root, "离线：只用这些记录规划");
  const edges = metric(root, "已知连接"); for (let i = 0; i < 20; i++) click(root, "离线：只更新一轮");
  assert.equal(metric(root, "已知连接"), edges); assert.equal(metric(root, "V(S)"), "0.656");
});
test("policy improvement follows evaluation and preserves the displayed values", () => {
  const { root } = mount("06"); assert.equal(button(root, "2 · 改进").disabled, true); click(root, "1 · 评价"); const v = metric(root, "V(出发点)");
  click(root, "2 · 改进"); assert.equal(metric(root, "V(出发点)"), v); assert.equal(button(root, "2 · 改进").disabled, true);
});
