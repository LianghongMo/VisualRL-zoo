import assert from "node:assert/strict";
import {test,after} from "node:test";
import {build} from "esbuild";
import {parseHTML} from "linkedom";
import {mkdtempSync,rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath,pathToFileURL} from "node:url";
import {CHAPTERS,BOOK_MAP,LEGACY_ROUTES,ALIASES,resolveRoute} from "../src/book/curriculum.js";
import {CHAPTER_KNOWLEDGE} from "../src/book/knowledge.js";
import {robotStep,finiteLqr,rolloutController,quadratic,oneStepControl,gaussianDensity,gaussianScore,gaussianBaselineStatistics,gaussianKl,closedLoopRadius,relabelContinuous} from "../src/rl/continuous-control.js";
const close=(a,b,tolerance=1e-9)=>assert.ok(Math.abs(a-b)<tolerance,a+" != "+b);
test("continuous force changes both velocity and position in physical time units",()=>{
  assert.deepEqual(robotStep([2,0],-1),[1.875,-0.5]);
  assert.deepEqual(robotStep([2,1],0),[2.5,1]);
  assert.deepEqual(robotStep([2,-1],0),[1.5,-1]);
  assert.deepEqual(robotStep([2,0],-1,{mass:2}),[1.9375,-0.25]);
  const one=oneStepControl(2,4);close(one.force,-16/9);close(one.value,-52/9);
  assert.ok(one.evaluate(one.force+0.1)<one.value);
});
test("finite LQR agrees with independently derived default and rational two-step solutions",()=>{
  const plan=finiteLqr();close(plan.K[0][0],0.929151032623924);close(plan.K[0][1],1.44699413332648);
  const result=rolloutController();close(result.totalReturn,-12.50015923106644);
  close(result.forces[0],-1.858302065247848);close(result.states.at(-1)[0],0.001310394197886);
  close(result.states.at(-1)[1],-0.058023843228121);close(result.totalReturn,result.optimalValue);
  const rational=finiteLqr({horizon:2,dt:1,positionWeight:1,velocityWeight:1,effort:1,terminal:[1,1]});
  close(rational.K[1][0],2/9);close(rational.K[1][1],2/3);
  close(rational.K[0][0],58/149);close(rational.K[0][1],142/149);
  close(rational.P[0][0][0],337/149);close(rational.P[0][0][1],152/149);close(rational.P[0][1][1],367/149);
});
test("Riccati value equals reward return and satisfies Bellman across states, goals, horizons and effort weights",()=>{
  for(const initial of [[2,0],[-1,0.7],[0,-0.5]])for(const goal of [-0.4,1])for(const horizon of [4,6,12])for(const effort of [0.1,0.5,2]){
    const r=rolloutController({initial,goal,horizon,effort});close(r.totalReturn,r.optimalValue,1e-8);
    for(let t=0;t<horizon;t++){
      close(r.states[t+1][0],robotStep(r.states[t],r.forces[t])[0]);
      close(-quadratic(r.states[t],r.plan.P[t]),r.rewards[t]-quadratic(r.states[t+1],r.plan.P[t+1]),1e-8);
    }
    close(-quadratic([initial[0]-goal,initial[1]],r.plan.P[0]),r.totalReturn,1e-8);
    close(r.totalReturn,r.rewards.reduce((sum,reward)=>sum+reward,0)+r.terminalReward,1e-8);
  }
  close(rolloutController({mode:"none"}).totalReturn,-104);
  close(rolloutController({mode:"position"}).totalReturn,-193.10837212018832,1e-6);
  close(rolloutController({mode:"pd"}).totalReturn,-17.77440821006521,1e-6);
});
test("actuator limits, model error and disturbances change the solved task without inventing optimality",()=>{
  const limited=rolloutController({limit:1});assert.ok(limited.clipped>0);assert.ok(limited.forces.every(u=>Math.abs(u)<=1));
  assert.ok(limited.totalReturn<limited.optimalValue);
  const wrong=rolloutController({mass:1.6});assert.ok(Math.abs(wrong.totalReturn-wrong.optimalValue)>0.1);
  const feedback=rolloutController({disturbance:0.8}),replay=rolloutController({disturbance:0.8,mode:"replay"});
  assert.ok(feedback.totalReturn>replay.totalReturn);
  assert.notDeepEqual(feedback.forces.slice(3),replay.forces.slice(3));
  close(closedLoopRadius(1,0),Math.sqrt(1.125));close(closedLoopRadius(1,1),Math.sqrt(0.625));
});
test("Gaussian policy gradients and baseline moments agree with independent numerical integration",()=>{
  const mean=0.3,sigma=0.7,dx=20*sigma/4000;
  for(const baseline of [-4,0,2]){
    let first=0,second=0;
    for(let i=0;i<4000;i++){const u=mean-10*sigma+(i+0.5)*dx,weight=gaussianDensity(u,mean,sigma)*dx;
      const gradient=(-((u+1)**2)-baseline)*gaussianScore(u,mean,sigma);
      first+=weight*gradient;second+=weight*gradient*gradient;}
    const exact=gaussianBaselineStatistics(mean,sigma,baseline);
    close(first,exact.gradient,1e-8);close(second-first*first,exact.variance,1e-7);
  }
  close(gaussianBaselineStatistics(0,1,0).variance,30);
  close(gaussianBaselineStatistics(0,1,-2).variance,18);close(gaussianBaselineStatistics(0,1,-4).variance,14);
  close(gaussianKl(0,1,0.2,1),0.02);close(gaussianKl(0,0.1,0.2,0.1),2);
});
test("continuous goal relabeling requires stopping, preserves source and respects physical/reset boundaries",()=>{
  const original=[{state:[1,0],action:-1,next:[0.5,0.8],terminated:false},
    {state:[0.5,0.8],action:-1,next:[0.52,0.05],terminated:false},
    {state:[0.52,0.05],action:0,next:[0.6,0.3],terminated:false}];
  const before=structuredClone(original),result=relabelContinuous(original,0.5);
  assert.equal(result.length,2);assert.equal(result[0].reward,0);assert.equal(result[1].reward,1);
  assert.equal(result[1].terminated,true);assert.deepEqual(original,before);
  const failure=relabelContinuous([{...original[0],terminated:true,terminationReason:"failure"},original[1]],0.5);
  assert.equal(failure.length,1);assert.equal(failure[0].terminated,true);
  const oldGoal=relabelContinuous([{...original[0],terminated:true,terminationReason:"goal"},original[1]],0.5);
  assert.equal(oldGoal.length,1);assert.equal(oldGoal[0].terminated,false);assert.equal(oldGoal[0].truncated,true);
  const truncated=relabelContinuous([{...original[0],truncated:true},original[1]],0.5);
  assert.equal(truncated.length,1);assert.equal(truncated[0].truncated,true);
});
const dir=mkdtempSync(join(tmpdir(),"visualrl-robot-course-"));
await build({entryPoints:[fileURLToPath(new URL("../src/book/index.js",import.meta.url))],bundle:true,platform:"node",format:"esm",target:"node24",loader:{".py":"text"},outfile:join(dir,"course.mjs")});
const {CHAPTER_MODULES}=await import(pathToFileURL(join(dir,"course.mjs")));
after(()=>rmSync(dir,{recursive:true,force:true}));
function mount(id){const {document,window}=parseHTML("<!doctype html><html><body><main></main></body></html>");
  Object.defineProperty(document,"compatMode",{value:"CSS1Compat"});Object.assign(globalThis,{document,window,Node:window.Node});
  const root=document.querySelector("main");return{root,window,cleanup:CHAPTER_MODULES[id].mount(root)};}
const click=(root,label)=>{const b=[...root.querySelectorAll("button")].find(b=>b.textContent.includes(label));assert.ok(b,label);b.click();};
const metric=(root,label)=>{const el=[...root.querySelectorAll(".metric")].find(e=>e.querySelector("span")?.textContent===label);assert.ok(el,label);return el.querySelector("strong").textContent;};
const input=(root,window,id,value)=>{const el=root.querySelector("#"+id);assert.ok(el,id);el.value=String(value);el.dispatchEvent(new window.Event("input"));};
test("the public book sequence maps all selected source chapters, equations and mastery questions to actual sections",()=>{
  assert.equal(CHAPTERS.length,10);assert.equal(BOOK_MAP.length,11);
  assert.match(CHAPTERS[8].title,/LQR/);assert.match(CHAPTERS[9].title,/机器人/);
  for(const c of CHAPTERS){const {root,cleanup}=mount(c.id);
    assert.equal(root.querySelectorAll("h1").length,1);assert.equal(root.querySelector("h1").textContent,c.title);
    assert.equal(root.querySelectorAll(".chapter-topic").length,c.topics.length);
    assert.equal(root.querySelectorAll(".chapter-toc a").length,c.topics.length);
    assert.equal(root.querySelectorAll(".mastery-row").length,CHAPTER_KNOWLEDGE[c.id].length);
    assert.ok([...root.querySelectorAll(".equation math")].some(e=>!e.closest("details")));
    assert.equal(root.querySelectorAll(".katex-error,.callout.error").length,0);
    const ids=[...root.querySelectorAll("[id]")].map(e=>e.id);assert.equal(new Set(ids).size,ids.length);
    for(const [,topic] of CHAPTER_KNOWLEDGE[c.id])assert.ok(root.querySelector(".chapter-topic[data-topic='"+topic+"']"));
    assert.ok(!root.textContent.includes("[object SVG"));cleanup();
  }
  for(const [key,[id,topic]] of Object.entries({...LEGACY_ROUTES,...ALIASES})){
    assert.equal(resolveRoute(key).chapter.id,id);assert.equal(resolveRoute(key).topic,topic);
    assert.ok(CHAPTERS.find(c=>c.id===id).topics.some(t=>t.id===topic));
  }
});
test("continuous state and feature controls reveal velocity and representation effects",()=>{
  let {root,window}=mount("01");click(root,"同位置，初速改为 +1");click(root,"执行一步");assert.equal(metric(root,"位置 p (m)"),"2.375");
  click(root,"同位置，初速改为 −1");click(root,"执行一步");assert.equal(metric(root,"位置 p (m)"),"1.375");
  ({root,window}=mount("03"));assert.equal(metric(root,"拟合预测"),"-0.667");click(root,"二次特征");assert.equal(metric(root,"拟合预测"),"-4");
  input(root,window,"feature-query",0.5);assert.equal(metric(root,"拟合预测"),"-0.25");
});
test("Gaussian, baseline, GAE, PPO and Jensen controls operate within their stated mathematical examples",()=>{
  let {root,window}=mount("05");click(root,"用这个样本更新均值");assert.equal(metric(root,"均值 μ"),"-0.2");
  click(root,"设为策略价值");assert.equal(metric(root,"梯度估计方差"),"18");click(root,"设为最小方差");assert.equal(metric(root,"梯度估计方差"),"14");
  assert.equal(metric(root,"GAE · A₀"),"0.446");click(root,"外部时间截断");assert.equal(metric(root,"GAE · A₀"),"1.742");
  ({root,window}=mount("07"));input(root,window,"continuous-ppo-ratio",1.5);assert.equal(metric(root,"PPO clipped目标"),"2.4");
  click(root,"负优势");assert.equal(metric(root,"PPO clipped目标"),"-3");
  input(root,window,"new-surrogate-q",0.1);input(root,window,"new-surrogate-p",0.9);assert.equal(metric(root,"新策略真实 J"),"1.044");
  input(root,window,"new-jensen-value",1);assert.equal(metric(root,"Jensen差距"),"0");click(root,"设为当前softmax");assert.equal(metric(root,"差距=温度×KL"),"0");
});
test("LQR, disturbance, saturation and goal controls display computed continuous trajectories",()=>{
  let {root,window}=mount("09");assert.equal(metric(root,"轨迹回报 G₀"),"-12.5");
  click(root,"真实执行下一步");assert.equal(metric(root,"当前执行步"),"1");
  click(root,"展示完整轨迹");assert.equal(metric(root,"当前执行步"),"6");
  click(root,"只看位置");assert.equal(metric(root,"轨迹回报 G₀"),"-193.108");click(root,"有限时域LQR");
  click(root,"切换推力限幅");assert.ok(Number(metric(root,"峰值推力 (N)"))<=1);
  input(root,window,"lqr-horizon",10);assert.equal(metric(root,"当前执行步"),"0");
  ({root,window}=mount("10"));assert.ok(Number(metric(root,"反馈推力 u"))<0);input(root,window,"goal-location",3);
  assert.ok(Number(metric(root,"反馈推力 u"))>0);assert.equal(metric(root,"物理位置 p"),"2");
});
