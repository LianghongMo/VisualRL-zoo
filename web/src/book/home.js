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
      h("p",{class:"lead"},"以《Reinforcement Learning: Theory and Algorithms》的目录为主线。压缩表格基础，重点连接函数近似、策略优化、模仿学习与连续控制。"),
      h("a",{class:"btn primary",href:"#chapter-01"},"从任务与MDP开始 →"),
      h("a",{class:"btn",href:"#chapter-09/feedback"},"先看连续小车实验"),
      h("p",{class:"home-note"},"位置与速度 → 连续推力 → 长期代价 → 反馈策略")),
    h("div",{class:"home-map"},robotScene(2,-0.6,0.8,0),
      h("p",{class:"robot-caption"},"同一台小车贯穿课程。向左运动时，右向推力可以刹车；到达目标还要把速度降下来。"))),
    h("section",{class:"curriculum-frame"},h("h2",{},"每一部分都回答下一个必要问题"),
      h("div",{class:"curriculum-levels"},[
        ["01–03","任务、数据与表示","先问最优策略是什么，再问数据够不够、连续价值能否可靠表示。"],
        ["04","探索与结构","怎样到达有用的状态？哪些假设让大空间可学习？理论细节选读。"],
        ["05–07","直接优化策略","从轨迹推导连续PG，处理优势估计、行为几何与旧数据更新。"],
        ["08–10","机器人与反馈控制","示范提供起点，LQR给精确基线，再连接有界策略、训练与GCRL。"],
      ].map(([range,title,text])=>h("div",{},h("span",{class:"eyebrow"},"第"+range+"章"),h("h3",{},title),h("p",{},text))))),
    h("section",{class:"course-overview"},h("h2",{},"课程目录"),h("p",{class:"muted"},"一章一个核心问题；小节按图像、任务、算例、推导、边界推进。前四章压缩基础，后六章通向连续机器人。"),
      groups),
    h("section",{class:"home-method"},h("h2",{},"本课程要让你真正能做什么"),
      h("div",{class:"method-grid"},[
        ["先把物理任务写完整","知道策略看什么、输出什么、环境怎样反应；把位置与速度一起带进长期目标。"],
        ["能独立推导、算出一个例子","Bellman、PG、GAE、PPO与Riccati逐步写出；图与数字来自同一组实际计算。"],
        ["知道方法在哪些条件下成立","数据覆盖、表示、KL、动作界和稳定性都有对应失败情形与验收问题。"],
      ].map(([title,text])=>h("div",{},h("b",{},title),h("p",{},text))))),
    h("section",{class:"home-method"},h("h2",{},"按原书目录选读，哪些内容压缩了"),
      h("p",{},"主要参考 ",h("a",{href:BOOK,target:"_blank",rel:"noopener"},"ABJKS · 2026-06-27工作草稿"),"；借鉴 ",
        h("a",{href:HARVARD,target:"_blank",rel:"noopener"},"Harvard CS2824公开讲义")," 从问题引出算法、把假设放在算法旁边的讲解方式。正文、图像和算例独立编写。"),
      table(["原书主题","网页位置","取舍"],BOOK_MAP),
      h("p",{class:"muted"},"复杂度、集中不等式、下界构造、LP/SDP/SLS等长证明留在选读；保留它们影响机器人方法成立的关键条件。连续Gaussian、GAE、SAC和GCRL分别注明补充来源。")),
    h("p",{class:"home-end"},"第9章实际运行确定性连续小车与有限时域LQR；第10章连接完整机器人训练与目标条件研究。解析算例与神经网络训练结果分别说明。"));
  return()=>{};
}
