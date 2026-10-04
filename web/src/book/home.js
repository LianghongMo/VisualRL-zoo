import {h} from "../ui/dom.js";
import {PARTS,BOOK_MAP,BOOK,HARVARD} from "./curriculum.js";
import {table} from "./shell.js";
import {robotScene} from "./labs.js";
export function mount(root){
  const groups=PARTS.map(part=>{
    const rows=part.chapters.map(c=>h("li",{},h("a",{class:"lesson-row",href:c.href,dataset:{chapter:c.id}},
      h("span",{class:"n"},c.id),
      h("span",{class:"t"},c.title,h("span",{class:"d"},c.question),h("small",{class:"textbook-mapping"},c.book)),
      h("span",{class:"row-arrow","aria-hidden":"true"},"→"))));
    return h("section",{class:"course-group"},h("h3",{},part.title),h("ol",{class:"lesson-list"},rows));
  });
  root.append(h("section",{class:"home-intro"},
    h("div",{class:"home-copy"},h("p",{class:"eyebrow"},"Visual RL · 从理论到连续机器人"),
      h("h1",{},"学会决策，",h("br"),"让机器人到达并停稳。"),
      h("p",{class:"lead"},"沿《Reinforcement Learning: Theory and Algorithms》的目录，从MDP学到机器人的连续控制。"),
      h("a",{class:"btn primary",href:"#chapter-01"},"从任务与MDP开始 →"),
      h("a",{class:"btn",href:"#chapter-09/feedback"},"先看连续小车实验"),
      h("p",{class:"home-note"},"状态 → 动作 → reward → 最优策略")),
    h("div",{class:"home-map"},robotScene(2,-0.6,0.8,0),
      h("p",{class:"robot-caption"},"向左行驶，向右推力刹车：到达目标，还要停稳。"))),
    h("section",{class:"curriculum-frame"},h("h2",{},"学习路线"),
      h("div",{class:"curriculum-levels"},[
        ["01–03","任务、数据与表示","定义目标，用数据估计连续价值。"],
        ["04","探索与结构","选择动作，采到有用的数据。"],
        ["05–07","直接优化策略","推导策略梯度，估计优势，控制更新幅度。"],
        ["08–10","机器人与反馈控制","从专家示范和LQR走向机器人训练与GCRL。"],
      ].map(([range,title,text])=>h("div",{},h("span",{class:"eyebrow"},"第"+range+"章"),h("h3",{},title),h("p",{},text))))),
    h("section",{class:"course-overview"},h("h2",{},"课程目录"),
      groups),
    h("section",{class:"home-method"},h("h2",{},"学完能做什么"),
      h("div",{class:"method-grid"},[
        ["定义物理任务","写清状态、动作、动力学和长期目标。"],
        ["推导并验算","推导Bellman、PG、GAE、PPO和Riccati，用实验核对。"],
        ["判断适用条件","检查数据覆盖、函数表示、动作约束和稳定性。"],
      ].map(([title,text])=>h("div",{},h("b",{},title),h("p",{},text))))),
    h("section",{class:"home-method"},h("h2",{},"与原书对应"),
      h("p",{},"主要参考 ",h("a",{href:BOOK,target:"_blank",rel:"noopener"},"ABJKS · 2026-06-27工作草稿"),"；借鉴 ",
        h("a",{href:HARVARD,target:"_blank",rel:"noopener"},"Harvard CS2824公开讲义")," 的讲解方式。正文、图像和算例独立编写。"),
      table(["原书主题","网页位置","取舍"],BOOK_MAP),
      h("p",{class:"muted"},"复杂度与LP/SDP/SLS证明选读；Gaussian策略、GAE、SAC和GCRL附补充来源。")),
    h("p",{class:"home-end"},"第9章运行小车LQR实验；第10章介绍机器人训练与GCRL。"));
  return()=>{};
}
