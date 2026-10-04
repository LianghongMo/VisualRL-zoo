import {h,s,button,slider,segmented,replace} from "../ui/dom.js";
import {metric,num} from "../ui/world-view.js";
import {robotStep,oneStepControl,rolloutController,gaussianDensity,gaussianScore,gaussianBaselineStatistics,gaussianKl,quadraticFit,goalFeedback} from "../rl/continuous-control.js";
import {generalizedAdvantages,surrogateExample,logJensen,entropyBound} from "../rl/policy-math.js";
import {table} from "./shell.js";
const chart=label=>s("svg",{viewBox:"0 0 520 260",class:"robot-chart",role:"img","aria-label":label});
const stats=()=>h("div",{class:"metrics"});
const experiment=(controls,...body)=>h("div",{class:"experiment"},h("div",{class:"toolbar"},controls),h("div",{class:"experiment-content"},...body));
export function robotScene(position=2,velocity=0,force=0,goal=0){
  const el=s("svg",{viewBox:"0 0 520 180",class:"robot-scene",role:"img","aria-label":"连续小车位置、速度、推力与目标"});
  const extent=Math.max(3,Math.abs(position)+1,Math.abs(goal)+1),x=p=>260+200*p/extent,car=x(position),f=Math.max(-75,Math.min(75,force*25)),v=Math.max(-90,Math.min(90,velocity*35));
  replace(el,s("line",{x1:20,y1:119,x2:500,y2:119,stroke:"var(--rule)","stroke-width":3}),
    [-3,-2,-1,0,1,2,3].map(i=>{const p=i*extent/3;return [s("line",{x1:x(p),x2:x(p),y1:115,y2:128,stroke:"var(--ink-3)"}),s("text",{x:x(p),y:149,"text-anchor":"middle"},num(p)+" m")];}),
    s("line",{x1:x(goal),x2:x(goal),y1:45,y2:116,stroke:"var(--learn)","stroke-width":2,"stroke-dasharray":"4 4"}),
    s("text",{x:x(goal),y:30,"text-anchor":"middle"},"目标"),
    s("rect",{x:car-21,y:88,width:42,height:23,rx:5,fill:"var(--ink)"}),
    s("circle",{cx:car-13,cy:115,r:6,fill:"var(--ink)"}),s("circle",{cx:car+13,cy:115,r:6,fill:"var(--ink)"}),
    s("line",{x1:car,y1:72,x2:car+v,y2:72,stroke:"var(--act)","stroke-width":3}),
    Math.abs(v)>1?s("polygon",{points:(car+v)+",72 "+(car+v-Math.sign(v)*8)+",67 "+(car+v-Math.sign(v)*8)+",77",fill:"var(--act)"}):null,
    s("text",{x:car,y:58,"text-anchor":"middle"},"v="+num(velocity)+" m/s"),
    s("line",{x1:car,y1:164,x2:car+f,y2:164,stroke:"var(--reward)","stroke-width":3}),
    Math.abs(f)>1?s("polygon",{points:(car+f)+",164 "+(car+f-Math.sign(f)*8)+",159 "+(car+f-Math.sign(f)*8)+",169",fill:"var(--reward)"}):null,
    s("text",{x:car,y:178,"text-anchor":"middle"},"u="+num(force)+" N"));
  return el;
}
export function stateLab(){
  let state=[2,0],force=-1,steps=0;const scene=h("div"),readout=stats();
  function render(){replace(scene,robotScene(state[0],state[1],force));replace(readout,metric("位置 p (m)",num(state[0])),metric("速度 v (m/s)",num(state[1])),metric("真实决策步",steps));}
  const controls=[slider({id:"robot-force",label:"连续推力 u (N)",min:-2,max:2,step:0.1,value:force,format:num,onInput:v=>{force=v;render();}}),
    button("执行一步 Δt=0.5s",{kind:"learn",onClick:()=>{state=robotStep(state,force);steps++;render();}}),
    button("同位置，初速改为 +1",{onClick:()=>{state=[2,1];steps=0;render();}}),
    button("同位置，初速改为 −1",{onClick:()=>{state=[2,-1];steps=0;render();}}),
    button("恢复 p=2,v=0",{kind:"ghost",onClick:()=>{state=[2,0];steps=0;render();}})];
  render();return experiment(controls,scene,readout,h("p",{class:"arithmetic"},"m=1 kg。蓝线是速度，橙线是推力。"));
}
export function bellmanLab(){
  let error=2,tail=4,force=0;const readout=stats(),body=h("div");
  function render(){const t=oneStepControl(error,tail);replace(readout,metric("当前候选动作 u",num(force)),metric("动作价值 Q(e,u)",num(t.evaluate(force))),metric("连续最优动作 u*",num(t.force)),metric("最优价值 V*(e)",num(t.value)));
    replace(body,h("div",{class:"physical-flow"},h("span",{},"当前 e="+num(error)),h("b",{},"→ 选u="+num(force)),h("span",{},"后继 e′="+num(error+force))),table(["动作 u","reward r=−e²−0.5u²","后续 V(e′)=−Pe′²","Q=r+V(e′)"],[0,-1,-2,t.force].map(u=>[num(u),num(-(error**2+0.5*u*u)),num(-tail*(error+u)**2),num(t.evaluate(u))])));}
  render();return experiment([slider({id:"bellman-tail",label:"给定后续价值系数 P",min:0,max:8,step:0.5,value:tail,onInput:v=>{tail=v;render();}}),
    slider({id:"bellman-force",label:"先试一个动作 u",min:-3,max:1,step:0.05,value:force,format:num,onInput:v=>{force=v;render();}}),
    button("换成连续最优动作",{onClick:()=>{force=oneStepControl(error,tail).force;render();}})],body,readout,h("p",{class:"muted"},"简化动力学e′=e+u；选择让Q最大的动作。"));
}
export function featureLab(){
  let mode="linear",query=2;const readout=stats(),plot=chart("连续位置的价值函数，线性与二次拟合"),labels=h("div");
  function render(){const xx=x=>260+x*90,yy=y=>55-y*38,values=Array.from({length:61},(_,i)=>-2+i/15);
    replace(plot,s("line",{x1:45,x2:475,y1:55,y2:55,stroke:"var(--rule)"}),
      s("polyline",{points:values.map(x=>xx(x)+","+yy(-x*x)).join(" "),fill:"none",stroke:"var(--learn)","stroke-width":3}),
      s("polyline",{points:values.map(x=>xx(x)+","+yy(quadraticFit(mode,x))).join(" "),fill:"none",stroke:"var(--act)","stroke-width":2,"stroke-dasharray":"5 4"}),
      [-1,0,1].map(x=>s("circle",{cx:xx(x),cy:yy(-x*x),r:5,fill:"var(--ink)"})),
      s("text",{x:35,y:28},"实线：V(x)=−x²；虚线：拟合；黑点：样本"),
      s("circle",{cx:xx(query),cy:yy(quadraticFit(mode,query)),r:6,fill:"var(--act)"}));
    replace(readout,metric("查询位置",num(query)),metric("拟合预测",num(quadraticFit(mode,query))),metric("真值 V(x)=−x²",num(-query*query)),metric("训练范围内",Math.abs(query)<=1?"是":"否"));
    replace(labels,table(["训练位置","V标签","预测"],[-1,0,1].map(x=>[x,-x*x,num(quadraticFit(mode,x))])));}
  render();return experiment([segmented([{value:"linear",label:"线性特征 (1,x)"},{value:"quadratic",label:"二次特征 (1,x,x²)"}],{value:mode,label:"价值表示",onChange:v=>{mode=v;render();}}),
    slider({id:"feature-query",label:"连续查询位置",min:-2,max:2,step:0.1,value:query,format:num,onInput:v=>{query=v;render();}})],plot,readout,labels);
}
export function gaussianLab(){
  let mean=0,sigma=1,action=-1,advantage=2;const plot=chart("连续推力Gaussian概率密度"),readout=stats(),note=h("p",{class:"arithmetic",role:"status"});
  function render(){const xx=a=>260+60*a,yy=p=>220-100*p,values=Array.from({length:121},(_,i)=>-4+i/15);
    replace(plot,s("polyline",{points:values.map(a=>xx(a)+","+yy(gaussianDensity(a,mean,sigma))).join(" "),fill:"none",stroke:"var(--learn)","stroke-width":3}),
      s("line",{x1:xx(action),x2:xx(action),y1:35,y2:220,stroke:"var(--act)","stroke-dasharray":"4 4"}),
      s("text",{x:35,y:25},"概率密度；蓝虚线：指定的已观察推力"),
      [-3,-2,-1,0,1,2,3].map(a=>s("text",{x:xx(a),y:245,"text-anchor":"middle"},a+"N")));
    replace(readout,metric("均值 μ",num(mean)),metric("标准差 σ",num(sigma)),metric("score ∂logπ/∂μ",num(gaussianScore(action,mean,sigma))),metric("给定样本优势 A",advantage));}
  render();return experiment([slider({id:"gaussian-action",label:"指定一个已采到的动作 u",min:-2,max:2,step:0.1,value:action,format:num,onInput:v=>{action=v;render();}}),
    slider({id:"gaussian-sigma",label:"策略标准差 σ",min:0.4,max:1.5,step:0.1,value:sigma,format:num,onInput:v=>{sigma=v;render();}}),
    segmented([{value:2,label:"正优势 +2"},{value:-2,label:"负优势 −2"}],{value:advantage,label:"样本优势",onChange:v=>{advantage=v;render();}}),
    button("用这个样本更新均值",{kind:"learn",onClick:()=>{const old=mean;mean+=0.1*advantage*gaussianScore(action,mean,sigma);note.textContent="μ "+num(old)+" → "+num(mean)+"（单样本更新，σ与优势固定）。";render();}}),
    button("均值恢复0",{kind:"ghost",onClick:()=>{mean=0;note.textContent="";render();}})],plot,readout,note);
}
export function gaeLab(){
  let end=true,lambda=0.8;const readout=stats(),rows=h("div");
  function render(){const t=generalizedAdvantages({rewards:[0,1],values:[0.4,0.7,2],terminated:[false,end],continuations:[true,false],lambda});
    replace(readout,metric("GAE · A₀",num(t.advantages[0])),metric("给critic的目标",num(t.targets[0])),metric("末步bootstrap",end?"不接尾值":"接最后真实状态的2"));
    replace(rows,table(["t","r","V旧","δ","Â"],[0,1].map(i=>[i,[0,1][i],[0.4,0.7][i],num(t.deltas[i]),num(t.advantages[i])])));}
  render();return experiment([segmented([{value:true,label:"真正终止"},{value:false,label:"外部时间截断"}],{value:end,label:"结束原因",onChange:v=>{end=v;render();}}),
    slider({id:"continuous-gae-lambda",label:"λ",min:0,max:1,step:0.05,value:lambda,format:num,onInput:v=>{lambda=v;render();}})],readout,rows,h("p",{class:"muted"},"γ=0.9。截断保留尾值，但GAE递推在回合边界停止。"));
}
export function baselineLab(){
  let baseline=0;const readout=stats();function render(){const t=gaussianBaselineStatistics(0,1,baseline);replace(readout,metric("期望梯度 ∂J/∂μ",num(t.gradient)),metric("梯度估计方差",num(t.variance)),metric("期望回报 J",num(t.value)),metric("最小方差 baseline",num(t.optimalBaseline)));}
  const input=slider({id:"gaussian-baseline",label:"动作无关 baseline b",min:-6,max:2,step:0.05,value:baseline,format:num,onInput:v=>{baseline=v;render();}});
  const set=b=>{baseline=b;input.set(b);render();};render();return experiment([input,button("设为策略价值 −2",{onClick:()=>set(-2)}),button("设为最小方差 −4",{onClick:()=>set(-4)})],
    readout,h("p",{class:"muted"},"u∼N(0,1)，r=−(u+1)²：梯度期望为−2，估计方差为30+8b+b²。"));
}
export function geometryLab(){
  let sigma=1,step=0.2;const readout=stats();
  function render(){replace(readout,metric("普通均值步长 Δμ",num(step)),metric("同方差Gaussian KL",num(gaussianKl(0,sigma,step,sigma))),metric("Fμμ=1/σ²",num(1/(sigma*sigma))),metric("KL预算0.02允许 |Δμ|",num(sigma*Math.sqrt(0.04))));}
  render();return experiment([slider({id:"geometry-sigma",label:"旧策略 σ",min:0.1,max:2,step:0.1,value:sigma,format:num,onInput:v=>{sigma=v;render();}}),
    slider({id:"geometry-step",label:"均值变化 Δμ",min:0,max:1,step:0.05,value:step,format:num,onInput:v=>{step=v;render();}})],readout);
}
export function surrogateLab(){
  let q=0.5,p=0.5;const readout=stats();function render(){const t=surrogateExample(q,p);replace(readout,metric("新策略真实 J",num(t.actual)),metric("旧状态surrogate L",num(t.surrogate)),metric("新/旧访问后继概率",num(q)+" / 0.5"));}
  render();return experiment([slider({id:"new-surrogate-q",label:"新策略进入后继状态的概率 q",min:0.05,max:0.95,step:0.05,value:q,format:num,onInput:v=>{q=v;render();}}),
    slider({id:"new-surrogate-p",label:"后继选好动作的概率 p",min:0.05,max:0.95,step:0.05,value:p,format:num,onInput:v=>{p=v;render();}})],h("div",{class:"physical-flow"},h("span",{},"s₀：安全 +1"),h("b",{},"或进入s₁ →"),h("span",{},"好 +2 / 坏 −2")),
    readout,h("p",{class:"muted"},"γ=0.9，旧q=p=0.5。J使用新访问概率q，surrogate仍使用旧访问概率0.5。"));
}
export function ppoLab(){
  let ratio=1,A=2;const readout=stats(),plot=chart("PPO正负优势的clipped目标");
  const objective=w=>Math.min(w*A,Math.max(0.8,Math.min(1.2,w))*A);
  function render(){const values=Array.from({length:91},(_,i)=>0.2+i*0.02),xx=w=>50+(w-0.2)*235,yy=y=>125-y*25;
    replace(readout,metric("概率比 w",num(ratio)),metric("未裁剪 wA",num(ratio*A)),metric("PPO clipped目标",num(objective(ratio))));
    replace(plot,s("line",{x1:45,x2:490,y1:125,y2:125,stroke:"var(--rule)"}),
      s("polyline",{points:values.map(w=>xx(w)+","+yy(w*A)).join(" "),fill:"none",stroke:"var(--act)","stroke-dasharray":"5 4"}),
      s("polyline",{points:values.map(w=>xx(w)+","+yy(objective(w))).join(" "),fill:"none",stroke:"var(--learn)","stroke-width":3}),
      [0.8,1.2].map(w=>s("line",{x1:xx(w),x2:xx(w),y1:20,y2:230,stroke:"var(--rule)","stroke-dasharray":"4 4"})),
      s("circle",{cx:xx(ratio),cy:yy(objective(ratio)),r:6,fill:"var(--learn)"}),s("text",{x:30,y:20},"实线：min目标；虚线：wA"),
      [0.2,0.8,1,1.2,2].map(w=>s("text",{x:xx(w),y:250,"text-anchor":"middle"},w)));}
  render();return experiment([slider({id:"continuous-ppo-ratio",label:"新/旧概率比 w",min:0.2,max:2,step:0.05,value:ratio,format:num,onInput:v=>{ratio=v;render();}}),
    segmented([{value:2,label:"正优势 +2"},{value:-2,label:"负优势 −2"}],{value:A,label:"优势符号",onChange:v=>{A=v;render();}})],plot,readout);
}
export function entropyLab(){
  let second=9,prob=0.5,temperature=1;const plot=chart("Jensen凹log曲线与两点的弦"),jstats=stats(),soft=stats();
  function render(){const j=logJensen(1,second,0.5),xx=x=>45+x*35,yy=y=>220-y*70,values=Array.from({length:101},(_,i)=>1+i*.11);
    replace(plot,s("polyline",{points:values.map(x=>xx(x)+","+yy(Math.log(x))).join(" "),fill:"none",stroke:"var(--learn)","stroke-width":3}),
      s("line",{x1:xx(1),x2:xx(second),y1:yy(0),y2:yy(Math.log(second)),stroke:"var(--act)","stroke-width":2}),
      s("line",{x1:xx(j.mean),x2:xx(j.mean),y1:yy(j.logMean),y2:yy(j.meanLog),stroke:"var(--reward)","stroke-width":4}),
      s("text",{x:30,y:24},"曲线：log x；弦：平均log；曲线在弦上方"));
    replace(jstats,metric("log平均",num(j.logMean)),metric("平均log",num(j.meanLog)),metric("Jensen差距",num(j.gap)));
    const t=entropyBound([0,2],[1-prob,prob],temperature);
    replace(soft,metric("π(奖励2动作)",num(prob,6)),metric("奖励＋温度×熵",num(t.lower)),metric("soft价值",num(t.softValue)),metric("差距=温度×KL",num(t.gap)));}
  const input=slider({id:"new-entropy-prob",label:"单步奖励(0,2)：第二动作概率",min:0.000001,max:0.999999,step:"any",value:prob,format:v=>num(v,6),onInput:v=>{prob=v;render();}});
  render();return [experiment(slider({id:"new-jensen-value",label:"等权正数 x₁=1，x₂=",min:1,max:12,step:0.5,value:second,onInput:v=>{second=v;render();}}),plot,jstats),
    experiment([input,slider({id:"new-entropy-temperature",label:"熵温度 α",min:0.25,max:4,step:0.25,value:temperature,format:num,onInput:v=>{temperature=v;render();}}),
      button("设为当前softmax最优",{onClick:()=>{input.set(entropyBound([0,2],[1-prob,prob],temperature).optimal[1]);prob=Number(input.input.value);render();}})],soft)];
}
export function controlLab({prefix="lqr",imitation=false,goals=false}={}){
  let horizon=6,effort=0.5,velocity=0,goal=0,mode="lqr",mass=1,disturbance=0,limited=false,cursor=0;
  let result;const scene=h("div"),readout=stats(),phase=chart("小车位置误差与速度相平面"),curves=chart("小车推力随决策步变化"),rows=h("div"),gains=h("div");
  function compute(){result=rolloutController({initial:[2,velocity],goal,mode,horizon,effort,mass,disturbance,limit:limited?1:Infinity});cursor=0;render();}
  function render(){const state=result.states[cursor],force=result.forces[cursor]??0;
    replace(scene,robotScene(state[0]+goal,state[1],force,goal));
    replace(readout,metric("轨迹回报 G₀",num(result.totalReturn)),metric("末位置误差 (m)",num(result.states.at(-1)[0])),metric("末速度 (m/s)",num(result.states.at(-1)[1])),metric("当前执行步",cursor),
      metric("峰值推力 (N)",num(result.peakForce)),metric("限幅次数",result.clipped),
      metric("模型最优价值 V₀*",num(result.optimalValue)),metric("模型K₀ (位置,速度)",result.plan.K[0].map(x=>num(x)).join(", ")));
    const extent=Math.max(2.5,...result.states.flat().map(Math.abs)),xx=e=>260+e/extent*200,yy=v=>130-v/extent*105;
    replace(phase,s("line",{x1:35,x2:485,y1:130,y2:130,stroke:"var(--rule)"}),s("line",{x1:260,x2:260,y1:25,y2:235,stroke:"var(--rule)"}),
      s("polyline",{points:result.states.map(([e,v])=>xx(e)+","+yy(v)).join(" "),fill:"none",stroke:"var(--learn)","stroke-width":3}),
      s("circle",{cx:260,cy:130,r:6,fill:"var(--ink)"}),s("circle",{cx:xx(state[0]),cy:yy(state[1]),r:6,fill:"var(--act)"}),
      s("text",{x:310,y:250},"位置误差 e →"),s("text",{x:25,y:18},"速度 v ↑；黑点：目标且静止"),s("text",{x:35,y:245},"坐标随轨迹缩放"));
    const max=Math.max(1,result.peakForce),fx=t=>45+t/(horizon-1||1)*430,fy=u=>130-u/max*85;
    replace(curves,s("line",{x1:45,x2:475,y1:130,y2:130,stroke:"var(--rule)"}),
      s("polyline",{points:result.forces.map((u,t)=>fx(t)+","+fy(u)).join(" "),fill:"none",stroke:"var(--reward)","stroke-width":3}),
      s("text",{x:35,y:20},"默认：负向接近，正向刹车"),
      [0,horizon-1].map(t=>s("text",{x:fx(t),y:248,"text-anchor":"middle"},"t="+t)),
      s("text",{x:15,y:48},num(max)+"N"),s("text",{x:15,y:220},"−"+num(max)+"N"));
    replace(rows,table(["t","p−g","v","推力 u","reward rₜ"],result.states.map((state,t)=>[t,num(state[0]),num(state[1]),t<horizon?num(result.forces[t]):"末端",num(t<horizon?result.rewards[t]:result.terminalReward)])));
    replace(gains,table(["t","模型K位置","模型K速度","P位置²","P交叉","P速度²"],result.plan.K.map((K,t)=>[t,num(K[0]),num(K[1]),num(result.plan.P[t][0][0]),num(result.plan.P[t][0][1]),num(result.plan.P[t][1][1])])));}
  const choices=imitation?[{value:"replay",label:"回放原专家推力"},{value:"lqr",label:"根据当前状态反馈"}]:[{value:"lqr",label:"有限时域LQR"},{value:"pd",label:"固定PD反馈"},{value:"position",label:"只看位置反馈"},{value:"none",label:"不控制"}];
  const controls=[segmented(choices,{value:mode,label:"控制方法",onChange:v=>{mode=v;compute();}}),
    slider({id:prefix+"-velocity",label:"初始速度 v₀ (m/s)",min:-1,max:1,step:0.1,value:velocity,format:num,onInput:v=>{velocity=v;compute();}})];
  if(!imitation)controls.push(slider({id:prefix+"-effort",label:"动作惩罚权重 R",min:0.1,max:2,step:0.1,value:effort,format:num,onInput:v=>{effort=v;compute();}}),
    slider({id:prefix+"-horizon",label:"计划时域 H",min:4,max:20,step:1,value:horizon,onInput:v=>{horizon=v;compute();}}));
  if(goals)controls.push(slider({id:prefix+"-goal",label:"目标位置 g (m)",min:-1,max:3,step:0.1,value:goal,format:num,onInput:v=>{goal=v;compute();}}));
  controls.push(button("加入 t=2 的速度扰动 +0.8",{onClick:()=>{disturbance=disturbance?0:0.8;compute();}}),
    button("切换实际质量 1 / 1.6 kg",{onClick:()=>{mass=mass===1?1.6:1;compute();}}),
    button("切换推力限幅 ±1 N",{onClick:()=>{limited=!limited;compute();}}),
    button("真实执行下一步",{kind:"learn",onClick:()=>{cursor=Math.min(horizon,cursor+1);render();}}),
    button("展示完整轨迹末步",{onClick:()=>{cursor=horizon;render();}}));
  const configuration=h("p",{class:"control-configuration",role:"status"});
  const originalRender=render;
  render=()=>{originalRender();configuration.textContent="模型m=1 kg；实际m="+mass+" kg；扰动="+disturbance+" m/s；"+(limited?"限幅±1 N":"不限幅")+"；目标g="+goal+" m。改参数后重算轨迹。"+(imitation?"专家序列按当前初态重算。":"");};
  compute();return experiment(controls,scene,configuration,readout,h("div",{class:"robot-plot-grid"},phase,curves),
    h("details",{class:"optional"},h("summary",{},"逐步查看：状态、动作与reward"),rows),
    h("details",{class:"optional"},h("summary",{},"展开模型的Riccati系数 Pₜ 与反馈 Kₜ"),gains));
}
export function goalLab(){
  let goal=0,velocity=0;const scene=h("div"),readout=stats();function render(){const force=goalFeedback(2,velocity,goal);replace(scene,robotScene(2,velocity,force,goal));replace(readout,metric("物理位置 p",2),metric("目标 g",num(goal)),metric("状态误差 p−g",num(2-goal)),metric("反馈推力 u",num(force)));}
  render();return experiment([slider({id:"goal-location",label:"目标 g (m)",min:-1,max:3,step:0.1,value:goal,format:num,onInput:v=>{goal=v;render();}}),
    slider({id:"goal-velocity",label:"相同位置的速度 v (m/s)",min:-1,max:1,step:0.1,value:velocity,format:num,onInput:v=>{velocity=v;render();}})],scene,readout,h("p",{class:"muted"},"使用第9章的LQR反馈：u=−K₀(p−g,v)。"));
}
