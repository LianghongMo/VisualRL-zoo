import { h } from "../ui/dom.js";
import { PARTS, BOOK_TOC } from "./curriculum.js";
import { WorldView, route, statesOf, NEAR_ROUTE, FAR_ROUTE, mapLegend } from "../ui/world-view.js";
import { GridWorld } from "../rl/envs/gridworld.js";
import { COURSE } from "../chapters/sources.js";
export function mount(root) {
  const env = GridWorld.chargingRoom();
  const map = new WorldView(env, { caption: "同一个MDP贯穿课程：先说清任务与最优策略，再比较模型规划、经验学习和目标条件的研究方法。" });
  map.render({ robot: env.start, paths: [{ states: statesOf(route(env, NEAR_ROUTE)) }, { states: statesOf(route(env, FAR_ROUTE)), color: "var(--learn)", dashed: true }] });
  root.append(h("section", { class: "home-intro" },
    h("div", { class: "home-copy" },
      h("p", { class: "eyebrow" }, "Visual RL · 按知识体系学强化学习"),
      h("h1", {}, "先学完整章节，", h("br"), "再进入研究课题。"),
      h("p", { class: "lead" }, "先定义任务，再比较策略，再求解价值。模型未知时转向经验学习；表格太大时转向近似与策略优化，最后进入研究专题。"),
      h("a", { class: "btn primary", href: "#chapter-01" }, "从第1章 · MDP开始 →"),
      h("p", { class: "home-note" }, "9章教材基础 + 1章研究专题 · 每章含公式、物理图像、实验与验收")),
    h("div", { class: "home-map" }, map.el, mapLegend("青线：5步到+1", "蓝色虚线：8步到+10"))),
    h("section", { class: "curriculum-frame" }, h("h2", {}, "同一个问题，逐章向前推进"),
      h("div", { class: "curriculum-levels" }, [
        ["01–03", "任务、价值与规划", "第1章定义MDP → 第2章比较长期结果 → 第3章用Bellman与DP求解。"],
        ["04–07", "从经验学习与规划", "模型未知 → MC完整回报 → TD一步自举 → 多步折中 → 学模型再规划。"],
        ["08–09", "近似与策略优化", "共享参数表示价值；直接训练概率策略与actor–critic。"],
        ["10", "研究专题", "离线数据、stitching、GCRL、未来分布、对比学习与SoRB。"],
      ].map(([range, title, description]) => h("div", {}, h("span", { class: "eyebrow" }, "第" + range + "章"), h("h3", {}, title), h("p", {}, description)))),
      h("p", { class: "curriculum-note" }, "参考 ", h("a", { href: BOOK_TOC, target: "_blank", rel: "noopener" }, "Sutton & Barto 第二版"), " 的概念与方法，按「上一章留下什么问题」组织顺序。前3章逐步建立MDP、价值和Bellman/DP；赌博机放在第5章探索补充。表格资格迹与n-step一起比较；GCRL进入最后的研究专题。")),
    h("section", { class: "course-overview" }, h("h2", {}, "完整目录与教材对应"),
      h("p", { class: "muted" }, "一个章节组织一组理论与方法；具体地图、算例和更新按钮属于章内小节。"),
      PARTS.map(part => h("section", { class: "course-group" }, h("h3", {}, part.title),
        h("ol", { class: "lesson-list" }, part.chapters.map(c => h("li", {},
          h("a", { class: "lesson-row", href: c.href, dataset: { chapter: c.id } },
            h("span", { class: "n" }, String(c.number).padStart(2, "0")),
            h("span", { class: "t" }, c.title, h("span", { class: "d" }, c.question), h("span", { class: "chapter-row-connection" }, c.relation.from), h("small", { class: "textbook-mapping" }, c.book)),
            h("span", { class: "row-arrow", "aria-hidden": "true" }, "→")))))))),
    h("section", { class: "home-method" }, h("h2", {}, "推导与常用技巧，从这里进入"),
      h("p", {}, "参考 ", h("a", { href: COURSE, target: "_blank", rel: "noopener" }, "Princeton COS435 的公开课表与论文阅读材料"), "，把工具放回它解决的问题中：为什么这样求梯度？怎样减少方差？怎样限制一次更新？每一步注明恒等式、近似或实践方法。"),
      h("nav", { class: "chapter-toc", "aria-label": "推导与常用技巧" }, [
        ["#chapter-09/derivation", "策略梯度：从轨迹概率推导"],
        ["#chapter-09/actor-critic", "Baseline、方差与GAE"],
        ["#chapter-09/surrogate", "Surrogate与TRPO"],
        ["#chapter-09/ppo", "PPO：正负优势与clipping"],
        ["#chapter-09/entropy", "Jensen、熵与softmax"],
        ["#chapter-09/practice", "训练时怎样使用这些工具"],
        ["#chapter-02/shaping", "奖励塑形：不改变最优策略的条件"],
        ["#chapter-08/stability", "Double DQN与稳定训练技巧"],
      ].map(([href, label]) => h("a", { href }, label))),
      h("p", { class: "muted" }, "先学所属章节的前置知识，再读推导；这是对相关公开阅读材料的选读与教学展开。")),
    h("section", { class: "home-method" }, h("h2", {}, "每一章怎样读"),
      h("div", { class: "method-grid" }, [
        ["1 · 先明确这一章解决什么", "读核心问题、前置知识和小节目录；基础定义只在所属章节系统建立。"],
        ["2 · 图像、计算、公式互相对应", "先观察具体状态与路线，再写出目标和更新；实验用来展示这条关系。"],
        ["3 · 用知识清单验收", "每项概念有对应小节与问题。能解释并算出一个例子，再继续下一章。"],
      ].map(([title, text]) => h("div", {}, h("b", {}, title), h("p", {}, text))))),
    h("p", { class: "home-end" }, "第1–9章建立方法基础；第10章主要参考Ben Eysenbach博士论文，明确区分精确教学算例与完整神经网络训练。"));
  return () => {};
}
