// Original, deterministic teaching systems. State/action are continuous;
// decisions are made at a fixed discrete sampling interval.
export function robotStep(state,force,{dt=0.5,mass=1}={}){
  const [position,velocity]=state;
  return [position+dt*velocity+0.5*dt*dt*force/mass,velocity+dt*force/mass];
}
const transpose=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const multiply=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,x,k)=>sum+x*b[k][j],0)));
const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
export const quadratic=(state,P)=>dot(state,P.map(row=>dot(row,state)));
export function finiteLqr({horizon=6,dt=0.5,mass=1,positionWeight=1,velocityWeight=0.2,effort=0.5,terminal=[20,10]}={}){
  if(!Number.isInteger(horizon)||horizon<1||!(dt>0&&mass>0&&effort>0))throw Error("LQR needs positive time, mass, effort and a finite positive integer horizon");
  const A=[[1,dt],[0,1]],B=[dt*dt/(2*mass),dt/mass],M=[[positionWeight,0],[0,velocityWeight]];
  const P=Array(horizon+1),K=Array(horizon);
  P[horizon]=[[terminal[0],0],[0,terminal[1]]];
  for(let t=horizon-1;t>=0;t--){
    const next=P[t+1],PB=next.map(row=>dot(row,B)),cross=multiply(transpose(A),PB.map(x=>[x])).map(row=>row[0]);
    const denominator=effort+dot(B,PB);
    K[t]=cross.map(x=>x/denominator);
    const apa=multiply(multiply(transpose(A),next),A);
    P[t]=apa.map((row,i)=>row.map((x,j)=>M[i][j]+x-cross[i]*cross[j]/denominator));
  }
  return {A,B,M,R:effort,P,K,horizon,dt,mass,terminal};
}
export function rolloutController({initial=[2,0],goal=0,mode="lqr",horizon=6,effort=0.5,mass=1,limit=Infinity,disturbance=0}={}){
  const plan=finiteLqr({horizon,effort}),nominal=[];
  let nominalState=[initial[0]-goal,initial[1]];
  for(let t=0;t<horizon;t++){const u=-dot(plan.K[t],nominalState);nominal.push(u);nominalState=robotStep(nominalState,u);}
  let state=[initial[0]-goal,initial[1]],totalReturn=0,clipped=0;
  const states=[state.slice()],forces=[],rewards=[];
  for(let t=0;t<horizon;t++){
    const wanted=mode==="none"?0:mode==="position"?-state[0]:mode==="pd"?-state[0]-state[1]:mode==="replay"?nominal[t]:-dot(plan.K[t],state);
    const force=Math.max(-limit,Math.min(limit,wanted));
    if(Math.abs(force-wanted)>1e-10)clipped++;
    const reward=-(state[0]**2+0.2*state[1]**2+effort*force**2);
    rewards.push(reward);totalReturn+=reward;forces.push(force);
    state=robotStep(state,force,{mass});
    if(t===2)state[1]+=disturbance;
    states.push(state.slice());
  }
  const terminalReward=-quadratic(state,plan.P[horizon]);totalReturn+=terminalReward;
  return {states,forces,rewards,terminalReward,totalReturn,clipped,plan,goal,
    optimalValue:-quadratic([initial[0]-goal,initial[1]],plan.P[0]),peakForce:Math.max(...forces.map(Math.abs))};
}
export function oneStepControl(error,tail=4,effort=0.5){
  const force=-tail*error/(effort+tail);
  const evaluate=u=>-(error**2+effort*u*u+tail*(error+u)**2);
  return {force,next:error+force,value:evaluate(force),evaluate};
}
export function gaussianScore(action,mean,sigma){
  if(!(sigma>0))throw Error("Gaussian standard deviation must be positive");
  return (action-mean)/(sigma*sigma);
}
export const gaussianDensity=(x,mean,sigma)=>Math.exp(-0.5*((x-mean)/sigma)**2)/(sigma*Math.sqrt(2*Math.PI));
export function gaussianBaselineStatistics(mean,sigma,baseline){
  const distance=mean+1,s2=sigma*sigma;
  return {gradient:-2*distance,value:-(distance*distance+s2),
    optimalBaseline:-(distance*distance+3*s2),
    variance:15*s2+14*distance*distance+6*baseline+(distance*distance+baseline)**2/s2};
}
export function gaussianKl(meanOld,sigmaOld,meanNew,sigmaNew){
  if(!(sigmaOld>0&&sigmaNew>0))throw Error("Gaussian standard deviations must be positive");
  return Math.log(sigmaNew/sigmaOld)+(sigmaOld*sigmaOld+(meanOld-meanNew)**2)/(2*sigmaNew*sigmaNew)-0.5;
}
export function quadraticFit(mode,x){
  return mode==="quadratic"?-x*x:-2/3;
}
export function goalFeedback(position,velocity,goal,gain=[0.9291510326,1.4469941333]){
  return -gain[0]*(position-goal)-gain[1]*velocity;
}
export function relabelContinuous(transitions,goal,{positionTolerance=0.1,velocityTolerance=0.1}={}){
  const result=[];
  for(const t of transitions){
    const reached=Math.abs(t.next[0]-goal)<=positionTolerance&&Math.abs(t.next[1])<=velocityTolerance;
    const physicalEnd=!!t.physicalTerminated||t.terminationReason==="failure"|| (!!t.terminated&&!t.terminationReason);
    const originalBoundary=!!t.terminated||!!t.truncated||!!t.resetBoundary;
    const terminated=reached||physicalEnd;
    result.push({...t,state:t.state.slice(),next:t.next.slice(),goal,reward:reached?1:0,terminated,
      truncated:!!t.truncated|| (originalBoundary&&!terminated),originalTerminated:!!t.terminated});
    if(reached||originalBoundary||physicalEnd)break;
  }
  return result;
}
export function closedLoopRadius(kp,kv,{dt=0.5}={}){
  const a=1-dt*dt*kp/2,b=dt-dt*dt*kv/2,c=-dt*kp,d=1-dt*kv;
  const trace=a+d,det=a*d-b*c,discriminant=trace*trace-4*det;
  if(discriminant<0)return Math.sqrt(det);
  const root=Math.sqrt(discriminant);
  return Math.max(Math.abs((trace+root)/2),Math.abs((trace-root)/2));
}
