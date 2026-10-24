/* GRPO lab: why GRPO can drop the value model. A live toy: a policy that picks one of 10 candidate answers for each of 8 addition
   questions, trained in the browser with real GRPO steps (seeded sampling, group-relative advantages, the clipped objective, the k3
   KL term). Memory figures are the batch 9 brief's S5 formulas. Also registers the home-tile teaser.
   Names (brief S2): 策略模型 policy, 参考模型 reference model, 奖励模型 reward model, 价值模型 value model (critic), 奖励 / 优势 / 基准,
   一组回答 / 组大小 G, 判分程序 checker, 截断 ε, KL 惩罚 β. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='grpo';

/* ---------- engine: pure functions ---------- */
/* one update per batch (MU = 1), as TRL's default num_iterations = 1 (grpo_config.py at 14c8d70); step size 4 is this lab's own */
const SEED=20261004, ETA=4, BETA=0.05, EPS=0.2, MU=1, ADV_EPS=1e-4, MAXS=30, NQ=8, NC=10, RIGHT=4, NEAR=0.35;
const OFFS=[-4,-3,-2,-1,0,1,2,3,4,5];               // candidate answers: the true sum − 4 … + 5; index 4 is right
/* question sets (示例): [a, b] and the toy policy's starting probability of the right answer */
const SETS={
  mix:{q:[[2,3],[6,7],[14,9],[27,18],[36,47],[58,67],[148,276],[389,457]],p:[0.95,0.85,0.7,0.5,0.3,0.15,0.07,0.03],f:4},
  easy:{q:[[1,2],[3,4],[2,6],[5,3],[7,2],[4,5],[6,3],[8,1]],p:[0.97,0.96,0.95,0.94,0.93,0.92,0.9,0.88],f:0},
  hard:{q:[[512,389],[647,758],[873,469],[1234,5678],[2468,1357],[3579,8642],[4096,8192],[9876,5432]],p:[0.08,0.06,0.05,0.04,0.03,0.03,0.02,0.02],f:0},
  ch:{q:[[46,38],[75,59],[88,77],[167,285],[238,594],[349,658],[456,789],[678,945]],p:[0.3,0.2,0.15,0.1,0.08,0.06,0.05,0.04],f:0}};
const GS=[2,4,8,16], CH_GS=[2,4,8,16,32], PEQ_GS=[2,8,64];
const DEMO={g2:{set:'mix',q:4,seed:6,G:8}, g7:{set:'mix',q:1,seed:3,G:8}};   // parts 2–4: 2 of 8 right; part 7: 7 of 8 right
const DEFAULT={set:'mix',G:8,std:'on',steps:0,fq:4};
const CH={set:'ch',std:'on',steps:0}, CH_GOAL=0.8;   // the challenge starts untrained: the 30 steps run after Submit (S19)
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function initLogits(p0){
  const z=OFFS.map(o=>-NEAR*Math.abs(o)); let s=0; for(let j=0;j<NC;j++) if(j!==RIGHT) s+=Math.exp(z[j]);
  z[RIGHT]=Math.log(p0/(1-p0)*s); return z;}
function softmax(z){let m=-Infinity; for(const v of z) if(v>m) m=v; const e=z.map(v=>Math.exp(v-m)); let s=0; for(const v of e) s+=v; return e.map(v=>v/s);}
function logSoftmax(z){let m=-Infinity; for(const v of z) if(v>m) m=v; let s=0; for(const v of z) s+=Math.exp(v-m); const c=m+Math.log(s); return z.map(v=>v-c);}
function draw(p,u){let acc=0; for(let j=0;j<NC;j++){acc+=p[j]; if(u<acc) return j;} return NC-1;}
/* the group-relative advantage, as TRL computes it: std over the group with Bessel's correction, + 1e-4 */
function groupStats(r,std){
  const G=r.length; let s=0; for(const v of r) s+=v; const mean=s/G; let q=0; for(const v of r) q+=(v-mean)*(v-mean);
  const sd=G>1?Math.sqrt(q/(G-1)):0, adv=r.map(v=>std?(v-mean)/(sd+ADV_EPS):v-mean);
  return {mean,sd,adv,eq:r.every(v=>v===r[0])};}
function klExact(z,zr){const lp=logSoftmax(z), lr=logSoftmax(zr); let s=0; for(let j=0;j<NC;j++) s+=Math.exp(lp[j])*(lp[j]-lr[j]); return s;}
/* a whole run: MAXS GRPO steps from the starting policy; every step and every group is recorded */
function train(set,G,std,eta){
  const Sx=SETS[set], rnd=mulberry(SEED), ref=Sx.p.map(initLogits), step=eta||ETA;
  let Z=ref.map(z=>z.slice()); const T={Z:[Z.map(z=>z.slice())],groups:[],acc:[],kl:[],eq:[0],gen:[0],clip:0,act:0,amp:[]};
  const stat=()=>{let a=0,k=0; for(let q=0;q<NQ;q++){a+=softmax(Z[q])[RIGHT]; k+=klExact(Z[q],ref[q]);} T.acc.push(a/NQ); T.kl.push(k/NQ);};
  stat();
  for(let t=0;t<MAXS;t++){
    const grp=[];
    for(let q=0;q<NQ;q++){const pi=softmax(Z[q]), lp=logSoftmax(Z[q]), o=[]; for(let i=0;i<G;i++) o.push(draw(pi,rnd()));
      const r=o.map(j=>j===RIGHT?1:0), st=groupStats(r,std==='on'); grp.push({o,r,olp:o.map(j=>lp[j]),p:pi[RIGHT],...st});
      if(!st.eq) T.amp.push(1/(st.sd+ADV_EPS));}   // how much dividing by the std enlarges this group's advantages
    for(let k=0;k<MU;k++){
      const grad=Z.map(()=>new Array(NC).fill(0));
      for(let q=0;q<NQ;q++){const g=grp[q], pi=softmax(Z[q]), lp=logSoftmax(Z[q]), lr=logSoftmax(ref[q]);
        for(let i=0;i<G;i++){const o=g.o[i], a=g.adv[i], rho=Math.exp(lp[o]-g.olp[i]);
          const clipped=(a>0&&rho>1+EPS)||(a<0&&rho<1-EPS); if(a!==0){T.act++; if(clipped) T.clip++;}
          const u=lr[o]-lp[o], c=(clipped?0:-a*rho/(NQ*G))+BETA*(1-Math.exp(u))/(NQ*G);
          for(let j=0;j<NC;j++) grad[q][j]+=c*((j===o?1:0)-pi[j]);}}
      Z=Z.map((z,q)=>z.map((v,j)=>v-step*grad[q][j]));}
    T.groups.push(grp); T.Z.push(Z.map(z=>z.slice())); stat();
    T.eq.push(grp.filter(g=>g.eq).length/NQ); T.gen.push((t+1)*NQ*G);}
  return T;}
const cache=new Map();
function traj(set,G,std,eta){const k=set+'|'+G+'|'+std+(eta?'|'+eta:''); let T=cache.get(k); if(!T){T=train(set,G,std,eta); cache.set(k,T);} return T;}
/* B1: the run without the division, with the step enlarged by the average factor the division gave the mixed groups of the
   divided run (same set, same G); what is left of the gap is what dividing by the std changes beyond the step size */
function matched(set,G){const on=traj(set,G,'on'), amp=on.amp.reduce((a,b)=>a+b,0)/on.amp.length, off=traj(set,G,'off',ETA*amp);
  return {amp,acc:off.acc[MAXS],pR:off.Z[MAXS].map(z=>softmax(z)[RIGHT])};}
function compute(s){
  const T=traj(s.set,s.G,s.std), t=Math.max(0,Math.min(MAXS,Math.round(s.steps)));
  const pR=T.Z[t].map(z=>softmax(z)[RIGHT]), p0=SETS[s.set].p;
  return {s,T,t,pR,p0,acc:T.acc[t],kl:T.kl[t],eq:T.eq[t],gen:T.gen[t],grp:t>0?T.groups[t-1]:null};}
/* a demo group, sampled from the starting policy (the reference) with its own seed */
function demoGroup(id,std){const d=DEMO[id], Sx=SETS[d.set], pi=softmax(initLogits(Sx.p[d.q])), rnd=mulberry(d.seed), o=[];
  for(let i=0;i<d.G;i++) o.push(draw(pi,rnd())); const r=o.map(j=>j===RIGHT?1:0);
  return {id,set:d.set,q:d.q,o,r,p:pi[RIGHT],...groupStats(r,std==='on')};}
const pEq=(p,G)=>Math.pow(p,G)+Math.pow(1-p,G);
const chPoint=G=>{const T=traj('ch',G,'on'), a=T.acc[MAXS]; return {G,acc:a,ok:a>=CH_GOAL,gen:T.gen[MAXS]};};
/* the memory column (brief S5): bytes per parameter for Llama-3.1-8B */
const NP=8030261248, GB=1e9, TRAINED=18, FROZEN=2;
const memRows=()=>{const t=NP*TRAINED/GB, f=NP*FROZEN/GB;
  return [{k:'ppo',models:['policy','value','ref','reward'],tot:2*t+2*f},{k:'rm',models:['policy','ref','reward'],tot:t+2*f},{k:'chk',models:['policy','ref'],tot:t+f}];};

/* ---------- formatting ---------- */
const f=(v,d)=>A.fmt.num(v,d), MINUS='−';
const sg=(v,d)=>{const s=Math.abs(v).toFixed(d); return (+s===0?'':(v<0?MINUS:'+'))+s;};
const fx=(v,d)=>{const s=Math.abs(v).toFixed(d); return (v<0&&+s!==0?MINUS:'')+s;};
const nz=(v,d)=>{const s=Math.abs(v).toFixed(d).replace(/\.?0+$/,''); return (v<0&&+s!==0?MINUS:'')+s;};   // at most d decimals, no trailing zeros
const sgz=(v,d)=>{const s=nz(Math.abs(v),d); return (+s===0?'':(v<0?MINUS:'+'))+s;};
function pc(p,d){const v=p*100; if(d==null) d=0; if(v>0&&v<0.5&&d===0) return '<1%'; if(v<100&&v>99.5&&d===0) return '>99%'; return f(v,d)+'%';}
const qTxt=(set,q)=>{const [a,b]=SETS[set].q[q]; return `${a} + ${b}`;};
const ansOf=(set,q,j)=>{const [a,b]=SETS[set].q[q]; return a+b+OFFS[j];};
const gb=v=>f(v,1)+' GB';
const stepsTxt=n=>tx(n===1?'1 step':`${n} steps`,`${n} 步`);
const zsp=s=>/[A-Za-z0-9)Ͱ-Ͽ]$/.test(s)?s+' ':s;
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<span class="linkbtn grpo-j" data-jump="${k}">${t}</span>`;
const M=t=>`<div class="grpo-math-line">${A.ui.math(t)}</div>`;
const MI=t=>A.ui.math(t);   // a short expression inside a sentence (the owner, 2026-10-09: every formula typeset, never a text string)
/* a formula with one part marked, so the part a sentence talks about is visible right there (R6.34); sel picks it in the MathML */
function Mhl(t,sel){const x=document.createElement('template'); x.innerHTML=A.ui.math(t); const e=x.content.querySelector(sel); if(e) e.classList.add('grpo-mhl');
  return `<div class="grpo-math-line">${x.innerHTML}</div>`;}
/* Show the math and the drill (the owner, 2026-10-09: “like a page of a good textbook”): a chain that breaks at “=”, each “=” under the
   one above, the substituted numbers in the same places as the symbols they replace */
const EQ=rows=>`<div class="grpo-eqs">${rows.map(([l,r,cont])=>`<span class="grpo-eq-l">${l?A.ui.math(l):''}</span><span class="grpo-eq-e">${cont?'':A.ui.math('=')}</span><span class="grpo-eq-r">${A.ui.math(r)}</span>`).join('')}</div>`;   // cont: the line goes on from the line above, no “=”
const ADV_TEX=nm=>`\\text{${nm}}\\,\\,A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}`;
const LL=id=>A.shell.labLink?A.shell.labLink(id):esc(id);
const P8=()=>`Llama‑3.1‑8B`;

/* ---------- shared definitions (brief S2) ---------- */
const D_POL=()=>tx('The policy is the model being trained; it writes the answers. In SFT and LoRA it is simply “the model”.','策略模型是正在训练、负责生成回答的模型。SFT 和 LoRA 里就叫“模型”。');
const D_REF=()=>tx('The reference model is a frozen copy of the policy from the start of training (usually the model after SFT). It is never updated and is used only to compute the KL.','参考模型是训练开始时策略模型的一份冻结副本（通常就是 SFT 之后的模型），不再更新，只用来算 KL。');
const D_RM=()=>tx('The reward model reads “prompt + response” and outputs one score; it is trained on preference pairs.','奖励模型读“提问 + 回答”，给出一个分数，用偏好对训练出来。');
const D_VAL=()=>tx('The value model is PPO’s estimate of “how much reward is still to come from this position, on average”, trained together with the policy. It is also called the critic.','价值模型是 PPO 里估计“从这个位置往后，平均还能拿到多少奖励”的模型，和策略模型一起训练。也叫 critic。');
const D_ADV=()=>tx('Advantage = reward − baseline: how much better this answer did than expected. PPO’s baseline comes from the value model; GRPO’s is the mean reward of the group of answers to the same question.','优势 = 奖励 − 基准：这个回答比预期好多少。PPO 的基准来自价值模型，GRPO 的基准是同一组回答的平均奖励。');
const D_BASE=()=>tx('The baseline is the score a question is expected to get. The advantage is the reward minus it.','基准是这道题预期能得的分数。优势就是奖励减去它。');
const D_GRP=()=>tx('In GRPO, a group is the G responses generated for the same prompt; G is the group size.','GRPO 里同一个提问生成的 G 个回答叫一组，G 叫组大小。');
const D_CHK=()=>tx('A verifiable reward comes from a program, the verifier, that checks the answer: 1 if it is right, 0 if it is wrong.','可验证的奖励来自一个判分程序：它检查答案对不对，对得 1 分，错得 0 分。');
const D_KL=()=>tx('The KL penalty measures how far the policy’s output distribution has moved from the reference model; β times it is subtracted from the reward or added to the loss. β does the same job as β in DPO.','KL 惩罚衡量策略模型的输出分布离参考模型有多远，乘上 β 从奖励里扣掉（或加进 loss）。和 DPO 里的 β 是同一个作用。');
const D_CLIP=()=>tx('Clipping looks at the ratio of the new to the old policy’s probability for the same answer (in real training, for each token). With ε = 0.2 the ratio may move within '+MI('[1 - ε,\\ 1 + ε] = [0.8,\\ 1.2]')+'. With a positive advantage, the part of the ratio above 1.2 brings no further gain. With a negative advantage, the part below 0.8 brings no further gain. If the ratio has left the range on the other side, the answer is still pushed back.','截断看的是新旧策略模型给同一个回答的概率之比（真实训练里按 token 算）。ε = 0.2 时，这个比值可以在 '+MI('[1 - ε,\\ 1 + ε] = [0.8,\\ 1.2]')+' 里变。优势为正时，比值超过 1.2 的部分不再带来好处。优势为负时，比值低于 0.8 的部分不再带来好处。如果比值从另一边出了范围，这个回答照样会被推回来。');
const D_ROLL=()=>tx('Generating responses: during RL training the policy writes its own answers (a rollout). Here it samples from its probabilities at temperature 1, so the answers in a group can differ.','生成回答：RL 训练里让策略模型自己写回答（英文叫 rollout）。这里按策略模型给的概率抽样，温度是 1，所以同一组里的回答可能各不一样。');
const D_TOY=()=>tx('The small model here has 80 parameters and answers 8 addition questions. For each question it picks one of 10 candidate answers (the true sum, 4 below it and 5 above it) through a softmax over that question’s 10 logits.','这里的小模型只有 80 个参数，答 8 道加法题。每道题它只在 10 个候选答案里选一个（正确的和，比它小 1 到 4 的，比它大 1 到 5 的），选哪个由这道题 10 个 logit 的 softmax 决定。');

/* ---------- state ---------- */
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, userAct=false, annT=0, ghost=null;
let demo=null, demoShown=false, peqG=8, view='acc', triedPeq=new Set(), triedStd=new Set();
const seenGroup=new Set(), seenAdv=new Set(), seenRun=new Set(), shownPrev={};
let runSpeed='fast', grpSpeed='mid';
const RUN_SPEEDS={slow:1,mid:1.5,fast:3};            // training steps per second: 30 steps take 30 s, 20 s or 10 s
const GRP_SPEEDS={slow:0.5,mid:1,fast:2};           // the group replay: about 7 s at 中
/* animation: anim = {kind:'group'|'run', ...}; the group replay walks phases 0 sample, 1 score, 2 baseline, 3 advantages, 4 update */
let anim=null, raf=0, lastTs=0, barsHL=0, barsT=0;
const PH_DUR=[2.4,1.0,0.9,1.3,1.0];
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};

/* ---------- the guided parts ---------- */
const PRED=[{v:'far'},{v:'same'},{v:'near'},{v:'neg'}];
const predTxt=v=>({far:tx('Positive, and further from the baseline than a wrong answer','正的，比答错的离基准远'),
  same:tx('Positive, and as far from the baseline as a wrong answer','正的，和答错的离基准一样远'),
  near:tx('Positive, and closer to the baseline than a wrong answer','正的，比答错的离基准近'),
  neg:tx('Negative','负的')})[v];
function genBtn(){return `<div class="grpo-ib"><button type="button" class="btn sm" data-act="gen">${tx('Generate 8 answers','生成 8 个回答')}</button></div>`;}
function inRun(){return `<div class="grpo-ib"><button type="button" class="btn sm" data-act="play" data-play>${playLabel()}</button>`+
  A.ui.range({id:'grpo-ic-steps',key:'steps',label:tx('Training steps','训练步数'),min:0,max:MAXS,value:S.steps,out:stepsTxt(S.steps),inline:true})+`</div>`;}
function inPeq(){return A.ui.seg({id:'grpo-ic-peq',key:'peq',label:tx('Answers per question, G','每道题生成几个回答 G'),options:PEQ_GS.map(g=>[String(g),String(g)]),value:String(peqG),inline:true});}
function inStd(){return A.ui.seg({id:'grpo-ic-std',key:'std',label:tx('Divide by the group’s standard deviation','除以这一组的标准差'),options:[['on',tx('Divide','除')],['off',tx('Do not divide','不除')]],value:S.std,inline:true});}
const g2=()=>demoGroup('g2',S.std), g7=st=>demoGroup('g7',st||S.std);
const STEPS=[
 {id:'mission',noTr:true,title:{en:'Your mission',zh:'你的任务'},lead:{en:'PPO trains a value model as big as the policy. Can the baseline come cheaper?',zh:'PPO 要多训练一个和策略模型一样大的价值模型。基准能不能来得更便宜？'},
  panels:['mem'],enter:()=>({...DEFAULT}),
  body:()=>{const m=memRows(), t=NP*TRAINED/GB;
   return tx(`<p>In ${LL('ppo')}, every answer the ${DEF('policy',D_POL())} writes is compared with a ${DEF('baseline',D_BASE())}. The baseline is how much this question usually scores. The reward minus the baseline is called the ${DEF('advantage',D_ADV())}; it says how much better this answer did than expected. PPO gets the baseline from a ${DEF('value model',D_VAL())}, as large as the policy and trained with it.</p><p>For ${P8()}, that value model alone takes ${gb(t)} of GPU memory in training (18 bytes per parameter, as in ${LL('training-memory')}). ${J('mem','The chart')} shows the four models PPO keeps in memory: ${gb(m[0].tot)}, counted over all the GPUs together.</p><p>Your job: see how GRPO gets a baseline without a value model, and what it pays for that. ${LL('dpo')} drops the reward model; GRPO drops only the value model. GRPO still scores every answer, and when a reward model does the scoring, that model still takes its memory (the second row of the chart).</p>`,
             `<p>在 ${LL('ppo')} 里，${DEF('策略模型',D_POL())}写出的每个回答都要和一个${DEF('基准',D_BASE())}比一比。基准就是这道题平均能得多少分。奖励减去基准叫${DEF('优势',D_ADV())}，它说的是这个回答比预期好多少。PPO 的基准来自${DEF('价值模型',D_VAL())}，它和策略模型一样大，要和策略模型一起训练。</p><p>训练 ${P8()} 时，光这个价值模型就要 ${gb(t)} 显存（每个参数 18 字节，见${LL('training-memory')}）。${J('mem','图里')}是 PPO 同时放在显存里的四个模型，所有 GPU 加起来一共 ${gb(m[0].tot)}。</p><p>你的任务：看懂 GRPO 怎么不用价值模型也能得到基准，以及它为此付出什么代价。${LL('dpo')} 去掉了奖励模型，GRPO 去掉的只是价值模型。GRPO 仍然要给每个回答打分，用奖励模型打分时，奖励模型照样要占显存（图里第二行）。</p>`);}},
 {id:'group',title:{en:'A group of answers',zh:'一组回答'},lead:{en:'Let the policy answer the same question 8 times.',zh:'同一道题，让策略模型答 8 次。'},
  panels:['group'],enter:()=>({...DEFAULT}),demo:'g2',phase:2,expect:()=>demoShown,label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Generate the 8 answers to continue, or Skip.',zh:'先生成 8 个回答，或点“跳过”。'},skip:true,
  body:()=>tx(`<p>GRPO gets the baseline another way: it lets the policy answer the same question G times. Those G answers are a ${DEF('group',D_GRP())}, and G is the group size; here G = 8. The policy ${DEF('samples',D_ROLL())} its answers, so they differ (temperature, as in ${LL('sampling')}).</p><p>Every answer then gets a score, its reward, which comes from one of two places. When an answer can be checked directly (the result of a math problem, code that tests can run), a ${DEF('checker',D_CHK())} program marks it right or wrong, 1 or 0, at almost no cost. When it cannot (writing, a conversation), a ${DEF('reward model',D_RM())} scores it. That is an LLM too: it reads the question and the answer and gives one score (${LL('reward-model')} shows how it is trained). For a group of 8 answers, it reads and scores each of the 8, and it needs its own GPU memory.</p><p>To keep it simple, this lab uses addition questions and a checker. The policy here is a ${DEF('small model',D_TOY())} with 80 parameters. The group’s average score is GRPO’s baseline.</p>`,
              `<p>GRPO 换了个办法得到基准：同一道题让策略模型答 G 次。这 G 个回答叫${DEF('一组',D_GRP())}，G 叫组大小，这里 G = 8。策略模型按概率${DEF('抽样生成',D_ROLL())}回答，所以同一道题每次的回答可能不一样（温度的作用见${LL('sampling')}）。</p><p>每个回答还要打一个分，这个分就是奖励。奖励有两种来源。答案能直接核对时（数学题的结果、能跑测试的代码），用${DEF('判分程序',D_CHK())}判对错，对得 1 分，错得 0 分，几乎不花算力。答案没法直接核对时（写作、对话），用${DEF('奖励模型',D_RM())}打分。奖励模型本身也是一个 LLM，读完提问和回答，给出一个分数（它怎么训练出来，见${LL('reward-model')}那个 Lab）。一组 8 个回答，它要读 8 遍、打 8 次分，还要占自己的一份显存。</p><p>这个 Lab 为了简单，用加法题和判分程序。策略模型是一个只有 80 个参数的${DEF('小模型',D_TOY())}。这一组回答的平均分，就是 GRPO 的基准。</p>`)+genBtn(),
  obs:()=>{const g=g2(), k=g.r.filter(v=>v).length;
   return tx(`<p><b>${k} of the 8 answers are right, so the group’s average score is ${f(g.mean,2)}.</b> That is this question’s baseline for this step: the dashed line in the chart. No value model was needed to get it.</p>`,
             `<p><b>8 个回答里有 ${k} 个对，所以这一组的平均分是 ${f(g.mean,2)}。</b>它就是这道题这一步的基准，也就是图里的虚线。得到它不需要价值模型。</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'2 of 8 right: what is a right answer’s advantage?',zh:'8 个回答里 2 个对：答对的回答，优势是多少？'},
  panels:['group'],enter:()=>({...DEFAULT}),demo:'g2',phase:2,gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const g=g2(), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(predTxt(o.v))}</button>`).join('')}</div>`;
   return tx(`<p>The advantage is the reward minus the baseline. In the group in the chart, 2 of the 8 answers are right, and the baseline is the average score, ${f(g.mean,2)}.</p><p>Is a right answer’s advantage positive or negative, and is it further from the baseline than a wrong answer’s? Pick one.</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
             `<p>优势等于奖励减去基准。图里这一组 8 个回答里有 2 个对，基准是平均分 ${f(g.mean,2)}。</p><p>答对的回答，优势是正还是负？它离基准，比答错的回答远还是近？选一个。</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'formula',title:{en:'The advantage',zh:'优势怎么算'},lead:{en:'Subtract the group mean, then divide by the group’s standard deviation.',zh:'先减去组内平均分，再除以组内的标准差。'},
  panels:['group'],enter:()=>({...DEFAULT}),demo:'g2',phase:3,
  body:()=>{const g=g2(), ai=g.adv[g.r.indexOf(1)], aw=g.adv[g.r.indexOf(0)];
   const verdict=pred?tx(`<p class="${pred==='far'?'okline':'badline'}"><b>${pred==='far'?'Right.':'Not quite.'}</b> You predicted “${esc(predTxt(pred))}”. The answer is “${esc(predTxt('far'))}”.</p>`,
                         `<p class="${pred==='far'?'okline':'badline'}"><b>${pred==='far'?'对了。':'不对。'}</b>你预测的是“${esc(predTxt(pred))}”，正确答案是“${esc(predTxt('far'))}”。</p>`)
                       :tx(`<p>The answer: “${esc(predTxt('far'))}”.</p>`,`<p>答案是“${esc(predTxt('far'))}”。</p>`);
   return verdict+tx(`<p>A right answer scores 1, ${f(1-g.mean,2)} above the baseline; a wrong one scores 0, ${f(g.mean,2)} below it. So the right answers sit 3 times as far from the baseline. GRPO then also divides by the standard deviation of the group’s rewards, so that the advantages of every question fall in a similar range. When a whole group scores the same, the std is 0, so a small number is added to the denominator:</p>${M(ADV_TEX('advantage'))}<p>${MI('r_i')} is the reward of answer ${MI('i')}; ${MI('\\text{mean}(r)')} and ${MI('\\text{std}(r)')} are the average and the standard deviation of the group’s 8 rewards. This group’s standard deviation is ${f(g.sd,2)}, so each right answer gets ${sg(ai,2)} and each wrong one ${sg(aw,2)} (${J('bars','the bars in the chart')}). The next update raises the probability of answers with a positive advantage and lowers the others. The baseline came from the group itself: that is why GRPO needs no value model.</p>`,
                     `<p>答对的回答得 1 分，比基准高 ${f(1-g.mean,2)}。答错的得 0 分，比基准低 ${f(g.mean,2)}。所以答对的离基准是答错的 3 倍。GRPO 还要再除以这一组奖励的标准差，让每道题的优势都落在差不多的范围里。一组回答得分全一样时，标准差是 0，所以分母上再加一个很小的数：</p>${M(ADV_TEX('优势'))}<p>${MI('r_i')} 是第 ${MI('i')} 个回答的奖励，${MI('\\text{mean}(r)')} 和 ${MI('\\text{std}(r)')} 是这一组 8 个奖励的平均分和标准差。这一组的标准差是 ${f(g.sd,2)}，所以每个答对的回答优势是 ${sg(ai,2)}，每个答错的是 ${sg(aw,2)}（${J('bars','图里的柱子')}）。下一次更新会调高优势为正的回答的概率，调低其余的。基准来自这一组回答本身，所以 GRPO 不需要价值模型。</p>`);}},
 {id:'run',title:{en:'Train 30 steps',zh:'训练 30 步'},lead:{en:'8 questions, 8 answers each, 30 GRPO steps.',zh:'8 道题，每道 8 个回答，训练 30 步 GRPO。'},
  panels:['group','probs','chart','metrics'],view:'acc',enter:()=>({...DEFAULT,steps:seenRun.has(runKey(DEFAULT))?MAXS:0}),target:'steps',expect:()=>S.steps>=10,showMe:{steps:MAXS},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Train at least 10 steps to continue.',zh:'至少训练 10 步再继续。'},
  /* the owner, 2026-10-09: say why the clip cannot act with one update per batch (the link behind the “so”), and that it limits how far
     an answer's probability moves, not which advantages count */
  /* polish r4: “TRL does this by default” stays in the fidelity card and the settings note; “a positive advantage becomes more
     likely” is part 4's last paragraph. Reader round: the card says first that the clip does not act here, then why in one
     paragraph (the ratio in words, no formula line), and a lead-in names what to watch */
  body:()=>tx(`<p>Now 8 questions train together (${J('probs','the bars in the chart')}). In each step every question first gets a group of 8 answers and their advantages; then this batch updates the policy.</p><p>As in PPO, a ${DEF('KL penalty',D_KL())} with β = 0.05 keeps the policy near the ${DEF('reference model',D_REF())}. ${DEF('Clipping',D_CLIP())} with ε = 0.2, though, does not act here.</p><p>The clip does not judge how large an advantage is; it limits how far an answer’s probability has already moved. It watches the probability ratio ${MI('\\rho')}, how many times as likely the answer is now as when it was written, and stops pushing an answer once ${MI('\\rho')} has left [0.8, 1.2] the way its advantage pushes. This lab uses each batch of answers for one update only. During that update the model being trained is still exactly the model that wrote the answers, so every ${MI('\\rho')} is 1 and the clip never acts. It acts only when the same answers are learned a second time (see ${LL('ppo')}).</p><p>Press ▶ Train and watch how the bars grow and change colour.</p>`,
              `<p>现在 8 道题一起训练（${J('probs','图里的概率条')}）。每一步，每道题先生成一组 8 个回答、算出优势，再用这一批回答更新策略模型。</p><p>和 PPO 一样，${DEF('KL 惩罚',D_KL())}（β = 0.05）让策略模型别离${DEF('参考模型',D_REF())}太远。${DEF('截断',D_CLIP())}（ε = 0.2）在这里却不起作用。</p><p>截断管的不是优势有多大，而是一个回答的概率已经被改了多少。它看概率之比 ${MI('\\rho')}，也就是这个回答现在的概率是写出它时的几倍，${MI('\\rho')} 朝优势的方向走出 [0.8, 1.2] 才不再推它。这个 Lab 每批回答只更新一次，更新时，正在训练的模型和写出这批回答的模型还是同一个，所以每个回答的 ${MI('\\rho')} 都等于 1，截断不会生效。同一批回答学第二遍时它才起作用，在 ${LL('ppo')} 那个 Lab 里能看到。</p><p>点“▶ 训练”，看概率条怎么变长、颜色怎么变。</p>`)+inRun(),
  obs:r=>obsRun(r)},
 {id:'zero',title:{en:'All right or all wrong',zh:'全对或全错'},lead:{en:'When a whole group scores the same, every advantage is 0.',zh:'一组回答得分全一样时，优势全是 0。'},
  panels:['chart','probs','metrics'],view:'peq',enter:()=>({...DEFAULT,steps:MAXS}),target:'peq',expect:()=>triedPeq.size>1,showMe:{peq:2},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another G to continue.',zh:'换一个 G 再继续。'},
  body:()=>tx(`<p>If all 8 answers in a group are right, the average is 1 and every reward minus the average is 0; all wrong works the same way. That question learns nothing in that step, and the 8 answers were generated for nothing.</p><p>For a question the policy gets right with probability p, a group of G answers is all right or all wrong with probability</p>${M('p^G + (1 - p)^G')}<p>Try a few values of G and watch ${J('chart','the curve')}.</p>`,
              `<p>一组 8 个回答如果全对，平均分是 1，每个回答的奖励减去平均分都是 0；全错也一样。这一步里，这道题什么也学不到，生成这 8 个回答的算力白花了。</p><p>一道题答对的概率是 p 时，一组 G 个回答全对或全错的概率是</p>${M('p^G + (1 - p)^G')}<p>换几个 G，看${J('chart','图里的曲线')}。</p>`)+inPeq(),
  obs:()=>obsPeq()},
 {id:'std',title:{en:'Dividing by the std',zh:'除以标准差'},lead:{en:'7 right, 1 wrong: a small std makes the wrong answer’s advantage large.',zh:'7 个对 1 个错：除以很小的标准差，错的那个优势变得很大。'},
  panels:['group'],enter:()=>({...DEFAULT}),demo:'g7',phase:3,target:'std',expect:()=>triedStd.size>1,showMe:{std:'off'},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Switch the division off once to continue.',zh:'把“除以标准差”切换一次再继续。'},
  /* the owner, 2026-10-09: the formula is shown where the division by the std is talked about, its denominator marked (R6.34) */
  body:()=>{const g=g7('on'), aw=sg(g.adv[g.r.indexOf(0)],1), fm=nm=>Mhl(ADV_TEX(nm),'mfrac > mrow:nth-child(2)'); return tx(`<p>Now an easy question: 7 of its 8 answers are right, and the average score is ${f(g.mean,2)}. The advantage divides by the standard deviation of the group’s rewards, the denominator of the formula:</p>${fm('advantage')}<p>The scores in this group are nearly all the same, so its standard deviation is small, only ${f(g.sd,2)}. The wrong answer sits ${f(g.mean,2)} below the average; divided by such a small number, its advantage grows to about ${aw}. The more alike a group’s scores, the smaller its std, and the more the odd answer’s advantage is blown up.</p><p>A paper that analyses GRPO training proposed a fix called Dr. GRPO (“GRPO Done Right”). Its authors call this a bias: questions that are almost always right or almost always wrong have a small std, so they count for more in the update. Their fix drops the division, and TRL and another widely used training library, verl, both have a switch that turns it off.</p>`,
                                     `<p>再看一道简单题，它的 8 个回答里 7 个对，平均分是 ${f(g.mean,2)}。算优势时要除以这一组奖励的标准差，也就是公式的分母：</p>${fm('优势')}<p>这一组的分数几乎都一样，所以标准差很小，只有 ${f(g.sd,2)}。答错的那个回答比平均分低 ${f(g.mean,2)}，除以这么小的数，它的优势被放大到约 ${aw}。一组的分数越一致，标准差越小，和大家不一样的那个回答，优势就被放得越大。</p><p>有一篇论文分析了 GRPO 的训练，提出一种改法，叫 Dr. GRPO（GRPO Done Right，意思是“改对了的 GRPO”）。作者认为这是一种偏差。几乎全对或几乎全错的题，标准差小，在更新里的分量就被放大了。他们的改法是不再除以标准差，TRL 和另一个常用的训练库 verl 都有开关，可以关掉这一步。</p>`)+inStd();},
  obs:()=>obsStd()},
 {id:'remember',noTr:true,title:{en:'What to remember',zh:'要记住的'},lead:{en:'No value model: 144.5 GB saved for the 8B. The price is G answers per question.',zh:'没有价值模型：8B 省下 144.5 GB。代价是每道题 G 个回答。'},
  panels:['mem','chart'],view:'acc',enter:()=>({...DEFAULT,steps:MAXS}),
  body:()=>{const m=memRows(), r=compute({...DEFAULT,steps:MAXS}), r2=compute({...DEFAULT,G:2,steps:MAXS}), sv=m[0].tot-m[1].tot;
   return tx(`<ul><li>GRPO uses the average score of a group of answers as the baseline, so it trains no value model. For ${P8()}, PPO’s four models need ${gb(m[0].tot)}. GRPO needs ${gb(m[1].tot)} with a reward model, ${gb(NP*FROZEN/GB)} of it for the reward model, and ${gb(m[2].tot)} with a checker. The ${gb(sv)} saved is exactly the value model (${J('mem','chart')}).</li><li>The price is generation. PPO needs no group, so one answer per question is enough, and verl’s default is 1. GRPO needs G answers for every question in every step, so each step generates G times as many answers as PPO. With a reward model, it also scores G times as many answers.</li><li>A group that is all right or all wrong teaches nothing. With 2 answers per question instead of 8, the same 30 steps reach only ${pc(r2.acc)}: most groups of 2 scored the same.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “GRPO needs no reward.” It does; it only drops the value model.<br>✗ “The larger G, the better.” Generation cost grows with G.<br>✗ “Questions the model always gets right teach the most.” All right or all wrong, the step learns nothing from them.</p></div><p class="small">The small model in this lab was scored by a checker from the start: that is how reinforcement learning with verifiable rewards (RLVR) works. How a model learns to reason from right or wrong alone is the subject of ${LL('rlvr')}.</p>${A.shell.toScenarios()}`,
             `<ul><li>GRPO 用一组回答的平均分当基准，所以不训练价值模型。训练 ${P8()} 时，PPO 的四个模型要 ${gb(m[0].tot)}。GRPO 用奖励模型打分时要 ${gb(m[1].tot)}，其中奖励模型占 ${gb(NP*FROZEN/GB)}。用判分程序时要 ${gb(m[2].tot)}。省下的 ${gb(sv)} 正好是价值模型（${J('mem','见图')}）。</li><li>代价在生成上。PPO 不需要一组回答，每道题生成 1 个就够了，verl 的默认设置就是 1 个。GRPO 每一步每道题都要生成 G 个回答，所以每一步生成的回答是 PPO 的 G 倍。用奖励模型时，它要打分的回答也是 G 倍。</li><li>一组回答全对或全错，这一步就学不到东西。每道题只生成 2 个回答时，同样训练 30 步，答对率只到 ${pc(r2.acc)}：2 个回答的组大多得分一样。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “GRPO 不需要奖励。”要奖励，它去掉的只是价值模型。<br>✗ “G 越大越好。”生成的成本跟着 G 一起涨。<br>✗ “总答对的题学得最多。”全对或全错，这一步都从它身上学不到东西。</p></div><p class="small">这个 Lab 的小模型从头到尾都由判分程序打分，这就是可验证奖励的强化学习（RLVR）的做法。只给对和错，模型怎么学会推理，${LL('rlvr')} 那个 Lab 专门讲。</p>${A.shell.toScenarios()}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
function obsRun(r){
  if(r.t===0) return '';
  const p0=r.T.acc[0], hard=r.pR[7], hard0=r.p0[7], done=r.t>=MAXS;
  return tx(`<p><b>After ${stepsTxt(r.t)}, the 8 questions’ average accuracy rose from ${pc(p0)} to ${pc(r.acc)}.</b> The easy questions were right from the start and barely move. The hardest, ${qTxt('mix',7)}, ${hard>hard0+0.02?`rose from ${pc(hard0)} to ${pc(hard)}`:`is still at ${pc(hard)}`}: its groups are often ${KW('wrong')}, and an all-wrong group teaches nothing (part 6).${done?'':' Keep training to 30 steps to see the whole run.'}</p>`,
            `<p><b>训练 ${r.t} 步后，8 道题的平均答对率从 ${pc(p0)} 升到 ${pc(r.acc)}。</b>容易的题一开始就对，几乎不动。最难的 ${qTxt('mix',7)} ${hard>hard0+0.02?`从 ${pc(hard0)} 升到 ${pc(hard)}`:`还在 ${pc(hard)}`}：它的组常常${KW('wrong')}，全错的组什么也学不到（第 6 部分）。${done?'':'继续训练到 30 步，看完整个过程。'}</p>`);}
function obsPeq(){const G=peqG, a=pEq(0.5,G), b=pEq(0.9,G);
  const lead=G===2?tx(`<p><b>With G = 2, even a question answered right half the time has ${pc(a)} of its groups all right or all wrong.</b></p>`,`<p><b>G = 2 时，答对概率 50% 的题也有 ${pc(a)} 的组全对或全错。</b></p>`)
   :G===8?tx(`<p><b>With G = 8, a question answered right half the time almost never scores the same in a whole group, but one answered right 90% of the time does so in ${pc(b)} of its groups.</b></p>`,`<p><b>G = 8 时，答对概率 50% 的题几乎不会整组得分一样；答对概率 90% 的题，却有 ${pc(b)} 的组全对。</b></p>`)
   :(()=>{const pm=Math.max(...compute({...DEFAULT,steps:MAXS}).pR), a=Math.pow(pm,64);
     return tx(`<p><b>With G = 64, a question whose accuracy is not close to 0 or 100% almost always gets both right and wrong answers in a group.</b> But a question already trained to ${pc(pm)} still has ${pc(a)} of its groups all right. And each step generates 8 times as many answers as with G = 8.</p>`,
               `<p><b>G = 64 时，答对概率离 0 和 100% 都不太近的题，几乎每组都有对有错。</b>可是已经练到 ${pc(pm)} 的题，仍有 ${pc(a)} 的组全对。每一步生成的回答也是 G = 8 时的 8 倍。</p>`);})();
  return lead+tx(`<p>In real training, TRL generates 8 answers per question by default, and DeepSeek‑R1 used 16. DAPO is a set of GRPO improvements published by ByteDance Seed and Tsinghua AIR. One of them, dynamic sampling, throws away groups that scored all the same and samples new questions until the batch has enough groups with both right and wrong answers.</p>`,
                 `<p>实际训练里，TRL 默认每道题生成 8 个回答，DeepSeek‑R1 用的是 16 个。DAPO 是字节跳动 Seed 和清华 AIR 发布的一套 GRPO 改进。它有一招叫动态采样，把得分全一样的组直接扔掉，再抽新题补上，直到这一批凑够有对有错的组。</p>`);}
function obsStd(){const on=g7('on'), off=g7('off'), w=on.r.indexOf(0);
  return S.std==='off'?tx(`<p><b>Without the division, the wrong answer’s advantage is ${sg(off.adv[w],2)}, about a third of the ${sg(on.adv[w],2)} it got with it.</b> Each right answer gets ${sg(off.adv[on.r.indexOf(1)],3)}. The push now depends only on the rewards, not on how spread out this group happened to be.</p>`,
                                `<p><b>不除以标准差时，答错的那个回答优势是 ${sg(off.adv[w],2)}，大约只有除以时（${sg(on.adv[w],2)}）的三分之一。</b>每个答对的回答是 ${sg(off.adv[on.r.indexOf(1)],3)}。现在推力只看奖励，不再看这一组碰巧有多分散。</p>`)
    :tx(`<p><b>Dividing again: the wrong answer is back at ${sg(on.adv[w],2)}.</b></p>`,`<p><b>又除以标准差了：答错的那个回答回到 ${sg(on.adv[w],2)}。</b></p>`);}

/* ---------- scenarios ---------- */
const PRESETS=[
 {t:{en:'Mixed difficulty (initial setup)',zh:'难度混在一起（初始设置）'},d:{en:'The 8 questions range from obvious to almost never solved.',zh:'8 道题从一眼能答到几乎答不对都有。'},p:{set:'mix',G:8,std:'on'}},
 {t:{en:'Questions too easy',zh:'题都太简单'},d:{en:'The model already gets almost everything right, for example questions it has trained on.',zh:'模型本来就几乎全对，比如拿练过的题再练一遍。'},p:{set:'easy',G:8,std:'on'}},
 {t:{en:'Questions too hard',zh:'题都太难'},d:{en:'The model gets almost everything wrong, for example a small model on contest problems.',zh:'模型几乎全错，比如小模型直接做竞赛题。'},p:{set:'hard',G:8,std:'on'}},
 {t:{en:'Tight generation budget: 2 answers per question',zh:'生成预算很紧：每道题只答 2 次'},d:{en:'Generating is expensive, so each question is answered only twice per step.',zh:'生成太贵，每一步每道题只让模型答 2 次。'},p:{set:'mix',G:2,std:'on'}},
 {t:{en:'No division by the std (Dr. GRPO)',zh:'不除以标准差（Dr. GRPO 的做法）'},d:{en:'As Dr. GRPO suggests, the advantage is only the reward minus the group mean.',zh:'按 Dr. GRPO 的建议，优势只减去组内平均分。'},p:{set:'mix',G:8,std:'off'}}];
const presetNow=()=>PRESETS.findIndex(p=>p.p.set===S.set&&p.p.G===S.G&&p.p.std===S.std);
const runKey=s=>s.set+'|'+s.G+'|'+s.std;
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how the average accuracy changes as GRPO trains.','选一个场景，看看 GRPO 训练时平均答对率怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="grpo-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function insightHTML(r){
  const i=presetNow();
  if(r.t===0) return tx('<p><b>Step 0: nothing trained yet.</b> Press Train in the chart, or drag the training steps.</p>','<p><b>第 0 步：还没训练。</b>按图里的“训练”，或拖动训练步数。</p>');
  let h=tx(`<p><b>After ${stepsTxt(r.t)}, the average accuracy went from ${pc(r.T.acc[0])} to ${pc(r.acc)}.</b></p>`,`<p><b>训练 ${r.t} 步后，平均答对率从 ${pc(r.T.acc[0])} 变到 ${pc(r.acc)}。</b></p>`);
  const eqAvg=r.T.eq.slice(1,r.t+1).reduce((a,b)=>a+b,0)/r.t;
  if(i<0) return h+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  const k=['mix','easy','hard','g2','nostd'][i];
  if(k==='easy') h+=tx(`<p>Most groups were ${KW('ok')} (on average ${pc(eqAvg)} of the groups per step scored the same), so most of the answers generated taught nothing. There was little left to learn, and it cost as much generation as the mixed set.</p>`,`<p>大多数组${KW('ok')}（平均每一步有 ${pc(eqAvg)} 的组得分一样），生成的回答大半白花了。本来就没多少可学的，生成的成本却和难度混在一起时一样多。</p>`);
  else if(k==='hard') h+=tx(`<p>At first most groups were ${KW('wrong')} and taught nothing (over the 30 steps, ${pc(eqAvg)} of the groups per step scored all the same on average). A question starts to move only once one of its groups happens to contain a right answer.</p>`,`<p>一开始大多数组${KW('wrong')}，什么也学不到（30 步平均下来，每一步有 ${pc(eqAvg)} 的组得分全一样）。一道题要等它的某一组碰巧出现答对的回答，才开始往上走。</p>`);
  else if(k==='g2'){if(r.t>=MAXS){const r8=compute({...S,G:8}); h+=tx(`<p>With only 2 answers per question, on average ${pc(eqAvg)} of the groups per step scored the same, and those groups taught nothing. With 8 answers the same 30 steps reach ${pc(r8.T.acc[MAXS])}, at 4 times the generation.</p>`,`<p>每道题只生成 2 个回答时，平均每一步有 ${pc(eqAvg)} 的组得分一样，这些组什么也没学到。生成 8 个回答时，同样 30 步能到 ${pc(r8.T.acc[MAXS])}，生成的回答是 4 倍。</p>`);}}
  else if(k==='nostd'){if(r.t>=MAXS){const ron=compute({...S,std:'on'}), mt=matched(S.set,S.G), d=Math.abs(ron.T.acc[MAXS]-mt.acc)*100, amp=f(mt.amp,1);
    const gap=d<1?tx('less than 1 percentage point','不到 1 个百分点'):tx(`about ${Math.round(d)} percentage points`,`约 ${Math.round(d)} 个百分点`);
    h+=tx(`<p>Without the division, the advantages of the groups with both right and wrong answers are smaller. With it, they were enlarged about ${amp} times on average. So with the same step size, this run without the division reaches only ${pc(r.T.acc[MAXS])}, against ${pc(ron.T.acc[MAXS])} with it.</p><p>In this lab, with the step size also enlarged that much, the two end ${gap} apart after 30 steps. So the gap between ${pc(r.T.acc[MAXS])} and ${pc(ron.T.acc[MAXS])} came mostly from the step size. Dr. GRPO is not about learning faster but about whether questions of different difficulty count fairly in the update.</p>`,
          `<p>不除以标准差时，有对有错的组，优势都变小了。除以标准差时，这些组的优势平均被放大约 ${amp} 倍。所以步长相同时，不除的这次运行只到 ${pc(r.T.acc[MAXS])}，除以时是 ${pc(ron.T.acc[MAXS])}。</p><p>在这个 Lab 里，把不除时的步长也调大这么多，30 步后两种做法只差${gap}。可见 ${pc(r.T.acc[MAXS])} 和 ${pc(ron.T.acc[MAXS])} 的差距，主要来自步长。Dr. GRPO 关心的不是学得快慢，而是难度不同的题在更新里的分量公不公平。</p>`);}}
  else h+=tx(`<p>The easy questions start right; the hard ones climb only once one of their groups contains a right answer. On average ${pc(eqAvg)} of the groups per step scored all the same.</p>`,`<p>容易的题一开始就对；难题要等某一组里出现答对的回答，才开始往上走。平均每一步有 ${pc(eqAvg)} 的组得分全一样。</p>`);
  return h;}

/* ---------- challenge ---------- */
function inChG(){return A.ui.seg({id:'grpo-ic-g',key:'G',label:tx('Answers generated per question, G','每道题生成几个回答 G'),options:CH_GS.map(g=>[String(g),String(g)]),value:String(S.G),inline:true});}
function challengeHTML(){
  const submit=`<div id="grpo-ch-sub"${chSubmitted?' hidden':''}><button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, the small model trains 30 steps with your G; then you see whether it reaches the goal.','提交后，小模型按你选的 G 训练 30 步，再告诉你是否达标。')}</p></div>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Lift 8 new questions to an average accuracy of at least 80% in 30 GRPO steps, generating as few answers as possible','30 步 GRPO 内，把 8 道新题的平均答对率提到至少 80%，生成的回答越少越好')}</h3>`+
   `<p>${tx(`The same small model now trains on 8 new addition questions, mostly hard: at the start it gets them right ${pc(traj('ch',8,'on').acc[0])} of the time on average. Training runs 30 steps; the advantage divides by the group’s std; β = 0.05, ε = 0.2. You choose only how many answers each question gets per step:`,`同一个小模型换 8 道新的加法题来练，大多偏难：一开始平均答对率只有 ${pc(traj('ch',8,'on').acc[0])}。训练 30 步，优势除以组内标准差，β = 0.05，ε = 0.2。你只能选每一步每道题生成几个回答：`)}</p>`+inChG()+
   `<div class="goal mono" id="grpo-ch-a"><span>${tx('Average accuracy after 30 steps','30 步后的平均答对率')}</span><span><b id="grpo-ch-av"></b>${tx(' (goal: at least 80%)','（目标：至少 80%）')}</span></div><div class="progress" id="grpo-ch-ap" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono" id="grpo-ch-g"><span>${tx('Answers generated in 30 steps','30 步一共生成的回答')}</span><span><b id="grpo-ch-gv"></b></span></div>`+
   `<p class="small muted">${tx('Accuracy here is the policy’s probability of the right answer, averaged over the 8 questions. The 80% goal is set for this challenge.','这里的答对率是策略模型答对的概率，在 8 道题上取平均。80% 是这道题定的目标。')}</p>`+
   submit+`<div id="grpo-ch-res" class="grpo-ch-res" aria-live="polite"></div></div>`;}
/* the challenge's result exists only once the submitted run has trained 30 steps (S19, R3.8) */
const chDone=()=>mode==='challenge'&&chSubmitted&&S.steps>=MAXS&&!(anim&&anim.kind==='run');
function chResultHTML(){
  if(!chDone()) return '';
  const pts=CH_GS.map(chPoint), me=pts.find(p=>p.G===S.G), okp=pts.filter(p=>p.ok), best=okp[0];
  let h;
  if(me.ok&&me===best) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${pc(me.acc)} after 30 steps, with ${A.fmt.int(me.gen)} answers generated.</b> And no smaller group reaches 80%: this is the fewest answers that works.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：30 步后答对率 ${pc(me.acc)}，一共生成 ${A.fmt.int(me.gen)} 个回答。</b>而且更小的组都到不了 80%：这已经是能达标的最少回答数。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${pc(me.acc)} after 30 steps.</b> But G = ${best.G} also reaches 80% with ${A.fmt.int(best.gen)} answers instead of ${A.fmt.int(me.gen)}.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：30 步后答对率 ${pc(me.acc)}。</b>不过 G = ${best.G} 也能到 80%，只要生成 ${A.fmt.int(best.gen)} 个回答，而不是 ${A.fmt.int(me.gen)} 个。</p>`);
  else {h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${pc(me.acc)} after 30 steps, under 80%.</b></p><p class="small">Hint: these questions are hard. With few answers per question, how often does a group contain at least one right answer? Look at ${J('chart-eq','the share of groups that scored all the same')}.</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：30 步后答对率 ${pc(me.acc)}，不到 80%。</b></p><p class="small">提示：这些题偏难。每道题只答几次时，一组里出现至少一个答对的回答的机会有多大？看看${J('chart-eq','得分全一样的组占多少')}。</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal) return h;
  /* brief (the owner's review of batch 8, 2026-10-07): the accuracy before training (the baseline), the G values that reach 80%, your pick and
     the initial G; the note under the chart counts the G values left out */
  const G0=2, kept=p=>p.ok||p===me||p.G===G0, a0=traj('ch',8,'on').acc[0];
  const rows=[{label:tx('Before training','训练前'),value:a0*100,text:pc(a0),ok:false,tags:[{text:tx('baseline','基准')}]}].concat(pts.map(p=>({
    label:tx(`G = ${p.G}: ${A.fmt.int(p.gen)} answers`,`G = ${p.G}：${A.fmt.int(p.gen)} 个回答`),value:p.acc*100,text:pc(p.acc),ok:p.ok,me:p===me,
    tags:(p.G===G0?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[])})));
  h+=A.ui.cmpChart({brief:true,title:tx('Average accuracy on the 8 new questions after 30 GRPO steps (goal: at least 80%)','30 步 GRPO 后，8 道新题的平均答对率（目标：至少 80%）'),
    aria:tx('Average accuracy after 30 steps. ','30 步后的平均答对率：')+tx(`before training: ${pc(a0)}; `,`训练前：${pc(a0)}；`)+pts.filter(kept).map(p=>`G ${p.G}${tx(': ','：')}${pc(p.acc)}`).join(tx('; ','；')),rows,goal:CH_GOAL*100,goalLabel:tx('goal: 80%','目标 80%'),max:100,
    hiddenText:n=>tx(`The other ${n===1?'G value stays':`${n} G values stay`} under 80% and ${n===1?'is':'are'} not shown.`,`其余 ${n} 个 G 到不了 80%，图里没有列出。`),
    okText:tx('reaches 80%','达到 80%'),noText:tx('misses it','没达到'),goalText:tx('goal: 80%','目标：80%')});
  const e2=traj('ch',2,'on'), e8=traj('ch',8,'on'), m2=e2.eq.slice(1).reduce((a,b)=>a+b,0)/MAXS, m8=e8.eq.slice(1).reduce((a,b)=>a+b,0)/MAXS;
  h+=tx(`<p class="small">Why: with 2 answers per question, ${pc(m2)} of the groups per step scored all the same on average, mostly all wrong, so the hard questions rarely got a push. With 8, that share falls to ${pc(m8)}: most hard questions get a right answer in some group early on. Beyond 8, accuracy still rises a little, but every doubling of G doubles the answers generated.</p>`,
        `<p class="small">原因：每道题 2 个回答时，平均每一步有 ${pc(m2)} 的组得分全一样，大多是全错，难题很少得到推力。8 个回答时，这个比例降到 ${pc(m8)}：大部分难题早早就有某一组出现答对的回答。超过 8 以后答对率还会涨一点，但 G 每翻一倍，生成的回答也翻一倍。</p>`);
  h+=tx(`<p class="small muted">Note: in this lab one seed decides the run; with another seed the numbers shift a little. The accuracy here is measured on the same 8 questions the model trained on. Real training measures it on questions the model has not trained on, and compares several seeds.</p>`,
        `<p class="small muted">注意：在这个 Lab 里，一次运行由一个随机种子决定，换一个种子，数字会有些变化。这里的答对率是在训练用的这 8 道题上量的。真实训练要拿没练过的题来量，并且用几个种子比较。</p>`);
  return h;}

/* ---------- capability layer ---------- */
const OUTCOME=()=>tx('explain where GRPO’s baseline comes from, compute a group’s advantages, and say when a group teaches nothing.','讲清 GRPO 的基准从哪来，算出一组回答的优势，并说出什么样的组学不到东西。');
const EXITS=()=>{const n1=groupStats([1,0,0,0],true), pN=pEq(0.9,8);
 return {
 explain:[
  {q:tx('Why can GRPO train without a value model?','GRPO 为什么不用价值模型也能训练？'),
   o:[tx('It answers each question G times and uses the group’s average score as the baseline; the value model’s job was to estimate exactly that expected score.','它让每道题答 G 次，用这一组的平均分当基准；价值模型要估的，正是这道题预期能得的分。'),
      tx('It uses the reference model’s score as the baseline, so no second trained model is needed.','它用参考模型的打分当基准，所以不用再训练一个模型。'),
      tx('It needs no baseline at all: the reward model’s score is used directly as the advantage.','它根本不需要基准：奖励模型的打分直接当优势用。')],a:0,
   n:[null,tx('The reference model only gives the KL term; it scores nothing.','参考模型只用来算 KL，不打分。'),tx('Without subtracting a baseline every right answer would be pushed equally, however easy the question; GRPO subtracts the group mean.','不减基准的话，不管题多容易，每个答对的回答都被一样地推；GRPO 减的是组内平均分。')],
   why:tx('The baseline only has to say what this question usually scores. A group of answers to the same question estimates that directly, so the 8B model’s 144.5 GB value model can go.','基准只需要说出这道题通常能得多少分。同一道题的一组回答直接就能估出它，所以 8B 模型那个 144.5 GB 的价值模型可以去掉。')},
  {q:tx('All 8 answers in a group are right. What does this step do for that question?','一组 8 个回答全对。这一步对这道题有什么作用？'),
   o:[tx('Its policy-gradient term is 0: every advantage is 0, and only the KL penalty (when β > 0) still pulls it slightly toward the reference.','它的策略梯度是 0：每个优势都是 0，只有 KL 惩罚（β > 0 时）还会把它往参考模型拉一点。'),
      tx('It learns the most: all 8 answers earned the top reward, so all of them are pushed up hard.','它学得最多：8 个回答都拿了满分，所以都被用力往上推。'),
      tx('All 8 answers are pushed down: the average is 1, so no answer beats the baseline.','8 个回答都被往下压：平均分是 1，没有一个回答超过基准。')],a:0,
   n:[null,tx('The advantage is the reward minus the group mean: 1 − 1 = 0 for each answer.','优势是奖励减去组内平均分：每个都是 1 − 1 = 0。'),tx('No answer is below the baseline either: each sits exactly on it, so nothing is pushed.','也没有回答低于基准：每个都正好在基准上，所以什么都不推。')],
   why:tx('With identical rewards the mean equals every reward, so every advantage is 0 and the answers were generated for nothing. That is why DAPO drops such groups and samples new questions.','奖励全一样时，平均分等于每个奖励，所以每个优势都是 0，这些回答白生成了。DAPO 因此扔掉这种组，补抽新题。')}],
 transfer:[
  {q:tx(`When a question gets 8 answers and 2 are right, each right answer’s advantage is +1.62. Now a question gets G = 4 answers and 1 is right. About how large is the right answer’s advantage?`,`一道题生成 8 个回答、2 个对时，每个答对的回答优势是 +1.62。现在一道题生成 G = 4 个回答，1 个对。答对的回答，优势大约多大？`),
   o:[tx('About +1.5, close to the 2-of-8 group','约 +1.5，和 8 个里 2 个对时差不多'),tx('About +3, twice the 2-of-8 group','约 +3，是 8 个里 2 个对时的 2 倍'),tx('About +0.75, half the 2-of-8 group','约 +0.75，是 8 个里 2 个对时的一半')],a:0,
   why:tx(`Both groups have a quarter right, so the mean is 0.25 in both and the right answer is 0.75 above it. Only the std changes a little (${MI('n - 1')} in the denominator): ${f(n1.sd,2)} here, so the advantage is ${sg(n1.adv[0],2)}.`,`两组都是四分之一答对，平均分都是 0.25，答对的回答都比它高 0.75。只有标准差稍有变化（分母是 ${MI('n - 1')}）：这里是 ${f(n1.sd,2)}，所以优势是 ${sg(n1.adv[0],2)}。`)},
  {q:tx('A question answered right half the time almost never has a whole group of 8 score the same. Now a question is answered right 90% of the time, with G = 8. About how often is its group all right or all wrong?','答对概率 50% 的题，8 个回答几乎不会整组得分一样。现在一道题答对的概率是 90%，G = 8。它的组全对或全错的机会大约多大？'),
   o:[tx('Under 1%, the same as the 50% question','不到 1%，和 50% 的题一样'),tx('About 40%','约 40%'),tx('About 90%','约 90%')],a:1,
   why:tx(`All right has probability ${MI('0.9^8')} and all wrong ${MI('0.1^8')}; together ${f(pN*100,1)}%. Easy questions waste many groups even at G = 8.`,`全对的概率是 ${MI('0.9^8')}，全错是 ${MI('0.1^8')}，加起来 ${f(pN*100,1)}%。即使 G = 8，容易的题也有很多组白生成。`),
   go:()=>{if(mode!=='sandbox') setMode('sandbox'); peqG=8; view='peq'; render();},goLabel:tx('Show the curve','看曲线')}],
 practice:[{id:'rl-for-llms',d:tx('The GRPO and PPO questions: the group baseline, clipping and the KL penalty, as in parts 4 to 7.','其中 GRPO 和 PPO 的题：组内基准、截断和 KL 惩罚，对应第 4 到第 7 部分。')},
           {id:'grpo-debug',d:tx('Debug a small GRPO trainer on a simple task with a checkable answer: the ratio, the clipping and the group-normalized advantage.','调试一个在可判对错的简单任务上训练的小型 GRPO：概率比、截断和组内归一化的优势。')}]};};

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. Generate a group of answers, predict a right answer’s advantage, train 30 steps, then see when a group teaches nothing and what dividing by the std does.','一共 8 个部分。先生成一组回答、预测答对的回答的优势，再训练 30 步，看什么样的组学不到东西，以及除以标准差起什么作用。'),
  sandbox:tx('Pick a kind of question set or a setting and watch GRPO train the small model on it.','选一种题目或一种设置，看 GRPO 怎么在上面训练这个小模型。'),
  challenge:tx('Choose how many answers each question gets so that 8 new questions reach 80% average accuracy in 30 steps with the fewest answers generated.','只选每道题生成几个回答，让 8 道新题在 30 步内平均答对率到 80%，生成的回答越少越好。')});
const SRC=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const TRL='https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/';
const VERL='https://github.com/volcengine/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/';
const FIDPOP=()=>{const m=memRows();
 return tx(`<p><b>Toy run:</b> your browser runs real GRPO steps on a small policy.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>The policy is a small model for 8 addition questions: a softmax over 10 candidate answers per question, 80 parameters. The reference model is the policy before training, frozen.</li><li>Each step samples G answers per question (seed ${SEED}, the same seed gives the same run), scores them 1 or 0, and computes the advantage ${MI('A = \\frac{r - \\text{mean}}{\\text{std} + 0.0001}')} with the group’s std taken with ${MI('n - 1')} in the denominator, as TRL does.</li><li>One gradient step per batch (step size ${ETA}) on the clipped objective with ε = 0.2 and the KL penalty ${MI('\\beta (e^{u} - u - 1)')} with ${MI('u = \\log \\pi_\\text{ref} - \\log \\pi')}, TRL’s estimator from DeepSeekMath. The loss averages over all answers.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>An answer is one choice, so per-token and per-answer quantities coincide, and TRL’s loss types (grpo, bnpo, dapo, dr_grpo) give the same loss here. Real answers are hundreds of tokens.</li><li>The questions, the candidate answers and the starting accuracies are examples written for this lab. The step size and β = 0.05 are this lab’s choices; they do not carry over to a real model (TRL’s default learning rate is 1e-6). One update per batch is TRL’s default (num_iterations = 1).</li><li>With one update per batch the probability ratio is exactly 1, so the clip never acts in this lab. TRL since v1.10 also multiplies the KL penalty by that ratio by default; this lab does not.</li><li>Accuracy is the policy’s probability of the right answer, averaged over the questions; KL is the exact KL over the 10 candidates, averaged over the questions.</li></ul></div><div><div class="eyebrow">Sources</div><ul><li>Advantage, std, epsilon and the KL estimator: TRL ${SRC(TRL+'trl/trainer/grpo_trainer.py','grpo_trainer.py')}, ${SRC(TRL+'trl/trainer/utils.py','utils.py')} and ${SRC(TRL+'docs/source/grpo_trainer.md','the GRPO docs')} (commit 14c8d70).</li><li>Memory as in the lab ${LL('training-memory')}: 18 bytes per trained parameter, 2 per frozen one. ${P8()} has 8,030,261,248 parameters. PPO takes ${gb(m[0].tot)}; GRPO takes ${gb(m[1].tot)} with a reward model and ${gb(m[2].tot)} with a checker. TRL defaults to β = 0 since v0.19.0; then the reference model goes too, and the two GRPO figures become ${gb(m[2].tot)} and ${gb(NP*TRAINED/GB)}. Activations and the KV cache of generation are not included.</li><li>Dr. GRPO: sail-sg/understand-r1-zero; DAPO’s dynamic sampling: verl ${SRC(VERL+'docs/algo/dapo.md','docs/algo/dapo.md')}.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
           `<p><b>玩具实跑：</b>页面在你的浏览器里，对一个小策略模型真的做 GRPO 的训练步。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>策略模型是一个答 8 道加法题的小模型：每道题 10 个候选答案上的 softmax，一共 80 个参数。参考模型是训练前的策略模型，冻结不动。</li><li>每一步每道题抽样 G 个回答（随机种子 ${SEED}，同一个种子，同一次运行），打 1 分或 0 分，再算优势 ${MI('A = \\frac{r - \\text{mean}}{\\text{std} + 0.0001}')}。组内标准差的分母用 ${MI('n - 1')}，和 TRL 一样。</li><li>每批回答只做一次梯度更新（步长 ${ETA}），目标是带截断（ε = 0.2）的式子，加上 KL 惩罚 ${MI('\\beta (e^{u} - u - 1)')}，其中 ${MI('u = \\log \\pi_\\text{ref} - \\log \\pi')}，这是 TRL 引自 DeepSeekMath 的估计式。loss 在所有回答上取平均。</li></ul></div><div><div class="eyebrow">简化和假设</div><ul><li>一个回答就是一次选择，所以按 token 算和按回答算的量是一回事，TRL 的几种 loss 写法（grpo、bnpo、dapo、dr_grpo）在这里也完全相同。真实的回答有几百个 token。</li><li>题目、候选答案和初始答对率都是为本 Lab 写的示例。步长和 β = 0.05 是这个 Lab 自己定的，不能照搬到真实模型（TRL 默认的学习率是 1e-6）。每批只更新一次是 TRL 的默认做法（num_iterations = 1）。</li><li>每批只更新一次时，概率比正好是 1，所以截断在这个 Lab 里不会起作用。TRL 从 v1.10 起默认还把 KL 惩罚乘上这个概率比，这个 Lab 没有这样做。</li><li>答对率是策略模型答对的概率在各题上的平均；KL 是 10 个候选答案上的精确 KL，在各题上取平均。</li></ul></div><div><div class="eyebrow">来源</div><ul><li>优势、标准差、小常数和 KL 估计式：TRL 的 ${SRC(TRL+'trl/trainer/grpo_trainer.py','grpo_trainer.py')}、${SRC(TRL+'trl/trainer/utils.py','utils.py')} 和 ${SRC(TRL+'docs/source/grpo_trainer.md','GRPO 文档')}（commit 14c8d70）。</li><li>显存的算法和${LL('training-memory')}那个 Lab 一样，训练的参数每个 18 字节，冻结的每个 2 字节。${P8()} 有 8,030,261,248 个参数。PPO 要 ${gb(m[0].tot)}，GRPO 用奖励模型时要 ${gb(m[1].tot)}，用判分程序时要 ${gb(m[2].tot)}。TRL 从 v0.19.0 起默认 β = 0，这时参考模型也不用加载，两种 GRPO 分别是 ${gb(m[2].tot)} 和 ${gb(NP*TRAINED/GB)}。这些数不含激活值和生成时的 KV cache。</li><li>Dr. GRPO：sail-sg/understand-r1-zero；DAPO 的动态采样：verl ${SRC(VERL+'docs/algo/dapo.md','docs/algo/dapo.md')}。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);};
const PRIPOP=()=>tx(`<p>PPO trains the policy with a reward: it generates answers, a reward model scores them, and each answer is pushed by its advantage, the reward minus a baseline. The baseline comes from a value model (the critic) as large as the policy. Clipping limits how far one batch moves the policy, and a KL penalty keeps it near the reference model. The ${LL('ppo')} lab covers all of it.</p>`,
                    `<p>PPO 用奖励训练策略模型：模型生成回答，奖励模型打分，每个回答按它的优势被推一把，优势就是奖励减去基准。基准来自一个和策略模型一样大的价值模型（critic）。截断限制一批回答能把策略模型改多少，KL 惩罚让它别离参考模型太远。这些都在 ${LL('ppo')} 里讲。</p>`);
function headerHTML(){return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:'GRPO',title:tx('Why can GRPO drop the value model?','GRPO 为什么能去掉价值模型？'),
  minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: PPO','前置知识：PPO'),primerPop:PRIPOP(),descs:DESCS(),mode});}

/* ---------- stage ---------- */
function playLabel(){return anim&&anim.kind==='run'?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Train','▶ 训练');}
function speedSeg(){const sp=ctxKind()==='run'?runSpeed:grpSpeed;
  return A.ui.seg({id:'grpo-speed',key:'speed',label:tx('Speed','速度'),options:['slow','mid','fast'].map(k=>[k,L(SPEED_N[k])]),value:sp});}
function stageHTML(){
  return `<div class="panel stage grpo-stage" id="grpo-stage">`+
   `<div class="stage-head grpo-head"><div class="grpo-st-title" id="grpo-st-title"></div><div class="stage-actions"><span class="status" id="grpo-status" tabindex="0"></span></div></div>`+
   `<div class="grpo-transport" id="grpo-transport"></div>`+
   /* the four panels in one box (polish r4): on a wide stage it puts the panels a part shows side by side (data-lay, panelsShow);
      elsewhere it is a plain block and the panels stack as before */
   `<div class="grpo-pans" id="grpo-pans">`+
   `<section class="grpo-pan" id="grpo-pan-group"><div class="grpo-pt" id="grpo-g-title"></div><div class="legend-row grpo-leg" id="grpo-g-leg"></div><svg class="grpo-gsvg" id="grpo-gsvg" role="img" aria-label=""></svg><div class="grpo-gnote small" id="grpo-g-note" aria-live="polite"></div></section>`+
   `<section class="grpo-pan" id="grpo-pan-probs"><div class="grpo-pt" id="grpo-p-title"></div><div class="legend-row grpo-leg" id="grpo-p-leg"></div><div class="grpo-probs" id="grpo-probs"></div><p class="small muted grpo-pnote" id="grpo-p-note"></p></section>`+
   `<section class="grpo-pan" id="grpo-pan-chart"><div class="grpo-chead"><div class="grpo-pt" id="grpo-c-title" tabindex="0"></div><div id="grpo-c-view"></div></div><svg class="grpo-csvg" id="grpo-csvg" role="img" aria-label=""></svg><div class="grpo-ckey small" id="grpo-c-key"></div></section>`+
   `<section class="grpo-pan" id="grpo-pan-mem"><div class="grpo-pt" id="grpo-m-title"></div><div class="legend-row grpo-leg" id="grpo-m-leg"></div><div class="grpo-mem" id="grpo-mem"></div><p class="small muted grpo-mnote" id="grpo-m-note"></p></section>`+
   `</div>`+
   `<div class="metrics grpo-metrics" id="grpo-metrics"></div>`+
   `<details class="math" id="grpo-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="grpo-mathbody"></div></details></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="grpo-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="grpo-lock" hidden></p>`+
   `<div class="lockable">`+A.ui.seg({id:'grpo-ctl-set',key:'set',label:tx('Questions','题目'),options:[['mix',tx('Mixed','难度混合')],['easy',tx('Easy','都简单')],['hard',tx('Hard','都很难')]],value:S.set,
      note:(()=>{const lo=k=>pc(Math.min(...SETS[k].p)), hi=k=>pc(Math.max(...SETS[k].p)); return tx(`8 addition questions (examples). Mixed: the small model starts at ${hi('mix')} on the easiest and ${lo('mix')} on the hardest. Easy: ${lo('easy')} to ${hi('easy')}. Hard: ${lo('hard')} to ${hi('hard')}.`,`8 道加法题（示例）。难度混合：小模型一开始在最简单的题上答对 ${hi('mix')}，最难的 ${lo('mix')}。都简单：${lo('easy')} 到 ${hi('easy')}。都很难：${lo('hard')} 到 ${hi('hard')}。`);})()})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'grpo-ctl-G',key:'G',label:tx('Answers per question per step, G','每一步每道题生成几个回答 G'),options:GS.map(g=>[String(g),String(g)]),value:String(S.G),
      note:tx(`The group size. The baseline is the average score of these G answers. Generation grows with G: each step generates ${MI('8 \\times G')} answers.`,`组大小。基准是这 G 个回答的平均分。生成量跟着 G 涨：每一步生成 ${MI('8 \\times G')} 个回答。`)})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'grpo-ctl-std',key:'std',label:tx('Divide the advantage by the group’s std','优势除以组内标准差'),options:[['on',tx('Divide','除')],['off',tx('Do not divide (Dr. GRPO)','不除（Dr. GRPO）')]],value:S.std,
      note:tx('TRL divides by default. Dr. GRPO argues that this makes questions that are almost always right or almost always wrong count for too much in the update.','TRL 默认要除。Dr. GRPO 认为，这会让几乎全对或几乎全错的题在更新里分量过大。')})+`</div>`+
   `<div class="lockable" id="grpo-steps-wrap">`+A.ui.range({id:'grpo-ctl-steps',key:'steps',label:tx('Training steps','训练步数'),min:0,max:MAXS,value:S.steps,out:stepsTxt(S.steps),
      note:tx('GRPO steps taken from the starting policy; each step uses every question once.','从初始的策略模型开始做了几步 GRPO；每一步把 8 道题都用一次。')})+`</div>`+
   `<details class="adv" id="grpo-adv"><summary>${tx('Fixed in this lab: β, ε, step size, updates per batch','这个 Lab 里固定的：β、ε、步长、每批更新次数')}</summary>`+
   `<p class="ctl-note">${tx(`KL coefficient β = ${BETA}, the value every RL lab on this site uses; TRL defaults to 0 since v0.19.0, verl’s GRPO examples use 0.001. Clip range ε = ${EPS}, TRL’s default. One update per batch, as TRL by default (num_iterations = 1), so the probability ratio is 1 and the clip never acts. Step size ${ETA} is this lab’s choice, set so that 30 steps show the whole effect; it does not carry over to a real model.`,`KL 系数 β = ${BETA}，本站每个讲 RL 的 Lab 都用这个值。TRL 从 v0.19.0 起默认是 0，verl 的 GRPO 示例用 0.001。截断范围 ε = ${EPS}，是 TRL 的默认值。每批回答只更新一次，和 TRL 的默认一样（num_iterations = 1），所以概率比总是 1，截断不会起作用。步长 ${ETA} 是这个 Lab 自己定的，为了让 30 步内看清全部效果，不能照搬到真实模型。`)}</p></details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`import torch

def grpo_advantages(rewards, G, scale=True, eps=1e-4):
    # rewards: (num_prompts * G,), G answers per prompt
    r = rewards.view(-1, G)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale:  # std over the group, n - 1, as TRL
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)

def grpo_loss(logp, old_logp, ref_logp, adv, mask,
              eps=0.2, beta=0.05):
    # logp, old_logp, ref_logp: (B, T) per-token log-probs
    # adv: (B,), one advantage per answer, on every token
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(1)
    surr = torch.min(ratio * a,
                     ratio.clamp(1 - eps, 1 + eps) * a)
    d = ref_logp - logp      # k3 KL estimator per token
    kl = torch.exp(d) - d - 1
    per_tok = -(surr - beta * kl)
    # average over all answer tokens in the batch
    return (per_tok * mask).sum() / mask.sum()`;
function connectHTML(){
  const fl=(base,p,x)=>`<a href="${base}${p}" target="_blank" rel="noopener"><code>${p}</code></a>${x}`;
  return `<section class="wrap connect" aria-labelledby="grpo-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="grpo-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Group advantages and the GRPO loss: the reward of each answer, its log-probabilities under the policy now, the policy that generated it and the reference model.','组内优势和 GRPO 的 loss：每个回答的奖励，以及它在当前策略模型、生成它时的策略模型和参考模型下的 log 概率。')}</p><pre class="code" id="grpo-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="grpo-prod">`+
   tx(`<li>TRL ${fl(TRL,'trl/trainer/grpo_trainer.py',': the group mean and std over each <code>num_generations</code> answers, then <code>advantages / (std_rewards + 1e-4)</code>.')}</li>`+
      `<li>TRL ${fl(TRL,'trl/trainer/grpo_config.py',': <code>num_generations</code> 8, <code>epsilon</code> 0.2, <code>learning_rate</code> 1e-6, <code>loss_type</code> "dapo", <code>scale_rewards</code> "group"; <code>beta</code> 0.0 since v0.19.0 (commit 7359ddc; it was 0.04), so by default no reference model is loaded.')}</li>`+
      `<li>verl ${fl(VERL,'docs/algo/grpo.md',': the group size is <code>actor_rollout_ref.rollout.n</code> (must be over 1); the KL goes into the loss, <code>kl_loss_coef</code> 0.001 with <code>low_var_kl</code> (the same k3 estimator); <code>norm_adv_by_std_in_grpo</code> turns the std off.')}</li>`+
      `<li>DAPO (verl ${fl(VERL,'docs/algo/dapo.md','')}): clip-higher with ε 0.2 below and 0.28 above, dynamic sampling that drops all-equal groups, and a token-level loss. Dr. GRPO (sail-sg/understand-r1-zero): no std division, and a constant instead of the answer length.</li>`,
      `<li>TRL ${fl(TRL,'trl/trainer/grpo_trainer.py','：在每 <code>num_generations</code> 个回答里算平均和标准差，再算 <code>advantages / (std_rewards + 1e-4)</code>。')}</li>`+
      `<li>TRL ${fl(TRL,'trl/trainer/grpo_config.py','：<code>num_generations</code> 8，<code>epsilon</code> 0.2，<code>learning_rate</code> 1e-6，<code>loss_type</code> "dapo"，<code>scale_rewards</code> "group"；<code>beta</code> 从 v0.19.0 起是 0.0（commit 7359ddc，之前是 0.04），所以默认不加载参考模型。')}</li>`+
      `<li>verl ${fl(VERL,'docs/algo/grpo.md','：组大小是 <code>actor_rollout_ref.rollout.n</code>（必须大于 1）；KL 加进 loss，<code>kl_loss_coef</code> 0.001，用 <code>low_var_kl</code>（同一个 k3 估计式）；<code>norm_adv_by_std_in_grpo</code> 关掉除以标准差。')}</li>`+
      `<li>DAPO（verl ${fl(VERL,'docs/algo/dapo.md','')}）：截断上下分开（下 0.2、上 0.28），动态采样扔掉得分全一样的组，loss 按 token 平均。Dr. GRPO（sail-sg/understand-r1-zero）：不除以标准差，并用一个常数代替回答长度来归一。</li>`)+
   `</ul><p class="small muted">${tx('Pinned commits: TRL 14c8d70 (version 1.15.0.dev0, after v1.14), verl 8718ca3.','固定的 commit：TRL 14c8d70（版本号 1.15.0.dev0，在 v1.14 之后），verl 8718ca3。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noopener">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a><span>${tx('Shao et al., 2024: introduces GRPO','Shao 等，2024：提出 GRPO')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noopener">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a><span>${tx('DeepSeek-AI, 2025: 16 answers per question, as TRL’s paper index quotes it','DeepSeek-AI，2025：每道题 16 个回答（TRL 的论文索引转引）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2503.14476" target="_blank" rel="noopener">DAPO: An Open-Source LLM Reinforcement Learning System at Scale</a><span>${tx('Yu et al., 2025','Yu 等，2025')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2503.20783" target="_blank" rel="noopener">Understanding R1-Zero-Like Training: A Critical Perspective</a><span>${tx('Liu et al., 2025 (Dr. GRPO)','Liu 等，2025（Dr. GRPO）')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="grpo-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="grpo-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="grpo-app m2" id="grpo-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="grpo-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* which panels and which group the stage shows now */
function ctx(){
  if(mode==='guided'){const s=STEPS[step]; return {panels:s.panels,demo:s.demo||null,phase:s.phase||4,noTr:!!s.noTr};}
  return {panels:['group','probs','chart','mem','metrics'],demo:null,phase:4,noTr:mode==='challenge'&&!chSubmitted};}
function ctxKind(){const c=ctx(); return c.demo?'group':'run';}
/* the group the panel shows: a demo group, or the featured question's group at the current step of the run */
function shownGroup(r){const c=ctx();
  if(c.demo){if(c.demo==='g2'&&mode==='guided'&&STEPS[step].id==='group'&&!demoShown) return null; const g=demoGroup(c.demo,S.std); return {...g,key:'demo-'+c.demo,setQ:[g.set,g.q],from:null,to:null};}
  if(r.t===0) return null;
  const g=r.grp[S.fq]; return {...g,key:'run-'+runKey(S)+'-'+r.t+'-'+S.fq,setQ:[S.set,S.fq],from:r.T.Z[r.t-1]?softmax(r.T.Z[r.t-1][S.fq])[RIGHT]:null,to:r.pR[S.fq]};}

/* the group picture: chips, rewards with the baseline, advantages; ph = phase 0–4, u = progress inside the phase (0–1) */
function drawGroup(r){
  const svg=$('#grpo-gsvg',app), c=ctx(), g=shownGroup(r), W=Math.max(260,Math.round(svg.getBoundingClientRect().width||svg.parentNode.clientWidth||560)), H=226;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('height',H);
  const title=$('#grpo-g-title',app), note=$('#grpo-g-note',app);
  const [set,q]=g?g.setQ:[S.set,S.fq], right=ansOf(set,q,RIGHT);
  title.innerHTML=g?(g.key.startsWith('demo')?tx(`The policy before training answers “${qTxt(set,q)} = ?” ${g.o.length} times (the right answer is ${right})`,`训练前的策略模型把“${qTxt(set,q)} = ?”答了 ${g.o.length} 次（正确答案是 ${right}）`)
      :tx(`Step ${r.t}: the group of ${g.o.length} answers to “${qTxt(set,q)} = ?” (the right answer is ${right})`,`第 ${r.t} 步：“${qTxt(set,q)} = ?”的一组 ${g.o.length} 个回答（正确答案是 ${right}）`))
    :tx(`“${qTxt(set,q)} = ?” (the right answer is ${right})`,`“${qTxt(set,q)} = ?”（正确答案是 ${right}）`);
  if(!g){svg.innerHTML=`<text x="${W/2}" y="${H/2}" class="grpo-empty" text-anchor="middle">${esc(c.demo?tx('Press “Generate 8 answers” in the card.','点卡片里的“生成 8 个回答”。'):c.noTr?tx('No answers yet. Training starts when you submit.','还没有回答。提交答案后，小模型才开始训练。'):tx('No answers yet: press Train or +1 step.','还没有回答：按“训练”或“+1 步”。'))}</text>`; svg.setAttribute('aria-label',title.textContent); note.innerHTML='&nbsp;'; return;}
  /* how far the replay has got */
  let ph=c.phase, u=1; if(anim&&anim.kind==='group'&&anim.key===g.key+'|'+S.std){ph=anim.ph; u=anim.u;}
  const G=g.o.length, padL=58, padR=10, cw=(W-padL-padR)/G, chipW=Math.min(46,cw-4), small=chipW<30;
  const y0=6, chipH=32, rTop=58, rBot=104, z0=156, aS=17, xOf=i=>padL+cw*i+cw/2;   // bars up to 3 × 17 px: every label stays inside H (S16)
  const shown=i=>ph>0?true:(i<Math.floor(u*G+1e-9)||u>=1);
  let h='';
  h+=`<text class="grpo-ax" x="4" y="${y0+chipH/2+4}">${esc(tx('answers','回答'))}</text>`;
  h+=`<text class="grpo-ax" x="4" y="${rTop+4}">${esc(tx('reward 1','奖励 1'))}</text><text class="grpo-ax" x="4" y="${rBot+4}">${esc(tx('reward 0','奖励 0'))}</text>`;
  h+=`<line class="grpo-grid" x1="${padL}" x2="${W-padR}" y1="${rTop}" y2="${rTop}"/><line class="grpo-grid" x1="${padL}" x2="${W-padR}" y1="${rBot}" y2="${rBot}"/>`;
  h+=`<text class="grpo-ax" x="4" y="${z0-26}">${esc(tx('advantage','优势'))}</text><line class="grpo-zero" x1="${padL}" x2="${W-padR}" y1="${z0}" y2="${z0}"/><text class="grpo-ax" x="${padL-4}" y="${z0+4}" text-anchor="end">0</text>`;
  const sc=ph>1||(ph===1&&u>0.5), allEq=g.eq;
  for(let i=0;i<G;i++){if(!shown(i)) continue; const x=xOf(i), ok=g.r[i]===1, ans=ansOf(set,q,g.o[i]);
    const cls='grpo-chip'+(sc?(ok?' ok':' no'):'')+(allEq&&ph>=3?' eq':'');
    h+=`<g class="${cls}"><rect x="${x-chipW/2}" y="${y0}" width="${chipW}" height="${chipH}" rx="5"/>`+
       (small?'':`<text x="${x}" y="${y0+14}" text-anchor="middle" class="grpo-chip-a">${ans}</text>`)+
       (sc?`<text x="${x}" y="${y0+(small?21:27)}" text-anchor="middle" class="grpo-chip-m">${ok?'✓':'✗'}</text>`:'')+`</g>`;
    if(sc){const yy=ok?rTop:rBot; h+=`<circle class="grpo-rdot${ok?' ok':''}" cx="${x}" cy="${yy}" r="${small?3:4.5}"/>`;}}
  if(ph>=2){const yb=rBot-(rBot-rTop)*g.mean, x1=padL, x2=padL+(W-padL-padR)*(ph===2?u:1);
    h+=`<line class="grpo-base" x1="${x1}" x2="${x2}" y1="${yb}" y2="${yb}"/>`;
    if(ph>2||u>=1) h+=`<text class="grpo-base-t" x="${W-padR}" y="${g.mean>=0.5?yb+13:yb-5}" text-anchor="end">${esc(tx(`baseline = average score ${f(g.mean,2)}`,`基准 = 平均分 ${f(g.mean,2)}`))}</text>`;}
  if(ph>=3){const k=ph===3?u:1;
    /* the link “the bars in the chart” marks the bars only, never the whole panel (the owner, 2026-10-09): a frame around them */
    if(performance.now()<barsHL&&!allEq){let t=z0-4, b=z0+4; for(let i=0;i<G;i++){const hg=Math.min(Math.abs(g.adv[i]),3)*aS; if(g.adv[i]>=0) t=Math.min(t,z0-hg-15); else b=Math.max(b,z0+hg+15);}
      h+=`<rect class="grpo-bars-hl" x="${padL+1}" y="${t.toFixed(1)}" width="${W-padR-padL-2}" height="${(b-t).toFixed(1)}" rx="6"/>`;}
    for(let i=0;i<G;i++){const a=g.adv[i]*k, x=xOf(i), bw=Math.max(3,Math.min(24,cw*0.5)), hgt=Math.min(Math.abs(a),3)*aS;
      /* H3: each step behind the “so” — the distance from the baseline, the division (when it is on), then the sign */
      const ok=g.r[i]===1, dv=f(ok?1-g.mean:g.mean,2), dvOn=S.std==='on', tip=k>=1&&!allEq?` data-tipx="${esc(tx(`A ${ok?'right':'wrong'} answer: reward ${g.r[i]}, ${dv} ${ok?'above':'below'} the baseline ${f(g.mean,2)}. ${dvOn?`Divided by the group’s standard deviation ${f(g.sd,2)}, its advantage is ${sg(g.adv[i],2)}`:`That is its advantage, ${sg(g.adv[i],2)} (no division by the standard deviation here)`}, so the next update makes it ${ok?'more':'less'} likely.`,`答${ok?'对':'错'}的回答：奖励 ${g.r[i]}，比基准 ${f(g.mean,2)} ${ok?'高':'低'} ${dv}。${dvOn?`除以这一组的标准差 ${f(g.sd,2)}，优势是 ${sg(g.adv[i],2)}`:`这就是它的优势 ${sg(g.adv[i],2)}（这里不除以标准差）`}，所以下一次更新会调${ok?'高':'低'}它的概率。`))}"`:'';
      const big=redOn()&&S.std==='on'&&Math.abs(g.adv[i])>2; const y=a>=0?z0-hgt:z0; h+=`<rect class="grpo-adv${a>=0?' up':' dn'}${big?' big':''}" x="${x-bw/2}" y="${y}" width="${bw}" height="${Math.max(0.5,hgt)}"${tip}/>`;
      if(big&&k>=1){const lt=tx('blown up by small std','被小标准差放大'), lw=lt.length*(A.lang()==='zh'?10.5:5.9), right=x+bw/2+4+lw<=W-2;   // beside the bar, on the side with room (S16)
        h+=`<text class="grpo-big-t" x="${right?x+bw/2+4:x-bw/2-4}" y="${y+hgt-2}" text-anchor="${right?'start':'end'}">${esc(lt)}</text>`;}
      if(cw>=26&&k>=1&&!allEq) h+=`<text class="grpo-adv-t${small?' sm':''}" x="${x}" y="${a>=0?y-4:y+hgt+12}" text-anchor="middle">${sg(g.adv[i],2)}</text>`;}
    if(allEq) h+=`<text class="grpo-eq-t" x="${(padL+W-padR)/2}" y="${z0+28}" text-anchor="middle">${esc(tx('every advantage in this group is 0','这一组优势全是 0'))}</text>`;}
  svg.innerHTML=h;
  svg.setAttribute('aria-label',ph<2?title.textContent:tx(`${G} answers, ${g.r.filter(v=>v).length} right; average score ${f(g.mean,2)}.`,`${G} 个回答，${g.r.filter(v=>v).length} 个对，平均分 ${f(g.mean,2)}。`));
  /* the line under the picture: what the update did (training steps only) */
  if(g.from!=null&&(ph>=4)) note.innerHTML=allEq?tx(`This group taught nothing: all ${G} answers scored ${g.r[0]}, so every advantage is 0. Only the KL penalty moved this question, slightly back toward the reference model.`,`这一组什么也没学到。${G} 个回答都是 ${g.r[0]} 分，优势全是 0，这道题只被 KL 惩罚往参考模型拉回了一点。`)
    :tx(`After the update, the policy answers this question right ${pc(g.to,1)} of the time (${pc(g.from,1)} before this step).`,`更新之后，策略模型答对这道题的概率是 ${pc(g.to,1)}（这一步之前是 ${pc(g.from,1)}）。`);
  else note.innerHTML=g.key.startsWith('demo')?tx('Answers are sampled from the policy before training, which answers this question right '+pc(g.p)+' of the time.','回答从训练前的策略模型里抽样，它答对这道题的概率是 '+pc(g.p)+'。'):'&nbsp;';
}
function drawGroupLegend(big){
  const lg=(tip,sw,t)=>`<span data-tipx="${esc(tip)}"><i class="sw ${sw}"></i><span>${t}</span></span>`;
  $('#grpo-g-leg',app).innerHTML=lg(tx('A right answer: the checker gives it reward 1.','答对的回答：判分程序给它 1 分。'),'grpo-sw-ok',tx('✓ right (reward 1)','✓ 答对（1 分）'))+
   lg(tx('A wrong answer: reward 0.','答错的回答：0 分。'),'grpo-sw-no',tx('✗ wrong (reward 0)','✗ 答错（0 分）'))+
   lg(tx('The baseline: the average score of this group of answers.','基准：这一组回答的平均分。'),'grpo-sw-base',tx('dashed line: baseline','虚线：基准'))+
   lg(tx('Each answer’s advantage: up if it beat the baseline, down if not. With the division on, it is also divided by the group’s standard deviation.','每个回答的优势：高于基准的朝上，低于的朝下。打开“除以标准差”时，还要再除以这一组的标准差。'),'grpo-sw-adv',tx('bars: advantage','柱子：优势')).replace('<span data-tipx','<span'+(performance.now()<barsHL?' class="grpo-leg-hl"':'')+' data-tipx')+(big?lg(tx('An advantage over 2 in size, made that large by dividing by a small std.','绝对值超过 2 的优势：除以很小的标准差，被放大成这样。'),'grpo-sw-big',tx('red: blown up by a small std','红色：被小标准差放大')):'');}
/* the red mark for an advantage blown up by a small std appears only once part 7 has explained it (S17) */
function redOn(){return mode!=='guided'||step>=SIX('std');}
/* the three kinds of groups (the owner, 2026-10-09): all right green, some right blue, all wrong red, the same in the chart, its legend
   and the text */
const KIND=g=>!g?'':(g.eq?(g.r[0]?'ok':'wrong'):'mix');
const KIND_T={ok:{en:'all right',zh:'全对'},mix:{en:'some right',zh:'有对有错'},wrong:{en:'all wrong',zh:'全错'}};
const KW=(k,t)=>`<span class="grpo-kt ${k}">${t||L(KIND_T[k])}</span>`;
function drawProbs(r){
  $('#grpo-p-title',app).innerHTML=r.t===0?tx('How often the policy answers each question right, before training','训练前，策略模型答对每道题的概率'):tx(`How often the policy answers each question right after ${stepsTxt(r.t)}`,`训练 ${r.t} 步后，策略模型答对每道题的概率`);
  const lg=(tip,sw,t)=>`<span data-tipx="${esc(tip)}"><i class="sw ${sw}"></i><span>${t}</span></span>`, G=S.G;
  $('#grpo-p-leg',app).innerHTML=`<span class="grpo-leg-lead">${tx('Colour shows this step’s group of answers:','颜色表示这一步的一组回答：')}</span>`+
    lg(tx(`All ${G} answers to the question were right in this step: every advantage is 0, so it learned nothing from them.`,`这一步，这道题的 ${G} 个回答全对：优势全是 0，什么也没学到。`),'grpo-sw-k ok',L(KIND_T.ok))+
    lg(tx('Some answers were right and some wrong: the right ones were pushed up, the wrong ones down.','这一步有的回答对、有的错：答对的被推高，答错的被压低。'),'grpo-sw-k mix',L(KIND_T.mix))+
    lg(tx(`All ${G} answers were wrong in this step: every advantage is 0, so it learned nothing from them.`,`这一步，这道题的 ${G} 个回答全错：优势全是 0，什么也没学到。`),'grpo-sw-k wrong',L(KIND_T.wrong))+
    lg(tx('How often the policy answered the question right before training (the reference model). A bar that ends right of its line has improved.','训练前（也就是参考模型）答对这道题的概率。条的末端在竖线右边，就是练高了。'),'grpo-sw-tick',tx('line: before training','竖线：训练前'));
  const host=$('#grpo-probs',app);
  host.innerHTML=SETS[S.set].q.map((qq,i)=>{const p=r.pR[i], p0=r.p0[i], g=r.grp?r.grp[i]:null, k=KIND(g);
    return `<button type="button" class="grpo-prow${i===S.fq?' on':''}${k?' k-'+k:''}" data-fq="${i}" aria-pressed="${i===S.fq}" data-tipx="${esc(pc(p,1)===pc(p0,1)?tx(`${qTxt(S.set,i)}: the policy answers it right ${pc(p,1)} of the time. Click to show its group.`,`${qTxt(S.set,i)}：策略模型答对的概率是 ${pc(p,1)}。点击显示它的一组回答。`):tx(`${qTxt(S.set,i)}: the policy answers it right ${pc(p,1)} of the time now, ${pc(p0,1)} before training. Click to show its group.`,`${qTxt(S.set,i)}：策略模型现在答对的概率是 ${pc(p,1)}，训练前是 ${pc(p0,1)}。点击显示它的一组回答。`))}">`+
      `<span class="grpo-pq mono">${qTxt(S.set,i)}</span><span class="grpo-ptrack"><i class="grpo-pfill" style="width:${(p*100).toFixed(2)}%"></i><i class="grpo-pref" style="left:calc(${(p0*100).toFixed(2)}% - 1px)"></i></span><span class="grpo-pv mono">${pc(p)}</span><span class="grpo-ptag small">${k?KW(k):''}</span></button>`;}).join('');
  /* one text for every step, so the panel keeps its height while a run plays (R7.9) */
  $('#grpo-p-note',app).innerHTML=tx('Bar length: how often the policy answers the question right; a grey bar has had no step yet. The framed row is the question whose group is shown above; click another row to switch.','条的长度是答对这道题的概率，灰色的条表示还没训练。框起来的一行，就是上面那组回答的题，点别的行可以换。');}
/* the chart: accuracy, share of all-equal groups, KL over steps; or p^G + (1 − p)^G against p */
const VIEWS=[['acc',{en:'Accuracy',zh:'答对率'}],['eq',{en:'All same',zh:'全一样的组'}],['kl',{en:'KL',zh:'KL'}],['peq',{en:'By difficulty',zh:'按难度'}]];
function drawChart(r){
  /* the height comes from the stylesheet (--grpo-ch): shorter where part 5 puts the chart under the group and the bars (polish r4) */
  const svg=$('#grpo-csvg',app), W=Math.max(260,Math.round(svg.getBoundingClientRect().width||svg.parentNode.clientWidth||560)), H=Math.round(parseFloat(getComputedStyle(svg).getPropertyValue('--grpo-ch')))||170, m={l:40,r:58,t:10,b:28};
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('height',H);
  const vh=$('#grpo-c-view',app); if(!vh.firstChild||vh._lang!==A.lang()){vh.innerHTML=A.ui.seg({id:'grpo-view',key:'view',label:tx('Chart','图'),options:VIEWS.map(([v,o])=>[v,L(o)]),value:view}); vh._lang=A.lang();}
  A.ui.setPressed($('.seg-ctl',vh),view);
  const tEl=$('#grpo-c-title',app), key=$('#grpo-c-key',app); let h='';
  const X0=m.l, X1=W-m.r, Y0=m.t, Y1=H-m.b;
  if(view==='peq'){
    tEl.textContent=tx('Chance that a group of G answers is all right or all wrong, against how often the policy answers the question right','一组 G 个回答全对或全错的概率，随答对概率 p 的变化');
    const X=p=>X0+(X1-X0)*p, Y=v=>Y1-(Y1-Y0)*v;
    for(let k=0;k<=4;k++){const v=k/4; h+=`<line class="grpo-grid" x1="${X0}" x2="${X1}" y1="${Y(v)}" y2="${Y(v)}"/><text class="grpo-ax" x="${X0-4}" y="${Y(v)+3}" text-anchor="end">${k*25}%</text>`;}
    for(let k=0;k<=4;k++){const p=k/4; h+=`<text class="grpo-ax" x="${X(p)}" y="${Y1+14}" text-anchor="middle">${k*25}%</text>`;}
    h+=`<text class="grpo-ax" x="${(X0+X1)/2}" y="${H-2}" text-anchor="middle">${esc(tx('p: how often the policy answers it right','p：答对的概率'))}</text>`;
    /* S18: each G its own line style, its name where the curves are far apart (G = 2 at its lowest point, G = 8 at 30%, G = 64 at
       10%, all on the left branch, where the curves with larger G lie to the left), the questions as dots on the chosen G's curve */
    const LBL={2:{p:0.5,v:0.5,dx:0,dy:-6,a:'middle'},8:{v:0.3},64:{v:0.1}};
    /* a curve's name moves along its curve when a question's dot would sit on it */
    const dots=r.pR.map(p=>[X(p),Y(pEq(p,peqG))]), clear=(x,y)=>dots.every(([a,c])=>a<x-6||a>x+42||c<y-14||c>y+6);
    const VS={8:[0.3,0.2,0.42,0.14],64:[0.1,0.16,0.06]};
    for(const G of [8,64]){const l=LBL[G]; l.v=VS[G].find(v=>clear(X(1-Math.pow(v,1/G))+5,Y(v)+4))??VS[G][0];}
    PEQ_GS.forEach(G=>{let d=''; for(let k=0;k<=100;k++){const p=k/100; d+=(k?'L':'M')+X(p).toFixed(1)+','+Y(pEq(p,G)).toFixed(1);} const on=G===peqG, l=LBL[G], lp=l.p!=null?l.p:1-Math.pow(l.v,1/G);
      h+=`<path class="grpo-pq-line g${G}${on?' on':''}" d="${d}"/><text class="grpo-cl${on?' on':''}" x="${(X(lp)+(l.dx!=null?l.dx:5)).toFixed(1)}" y="${(Y(l.v)+(l.dy!=null?l.dy:4)).toFixed(1)}" text-anchor="${l.a||'start'}">G = ${G}</text>`;});
    r.pR.forEach((p,i)=>{const v=pEq(p,peqG); h+=`<circle class="grpo-pdot" cx="${X(p).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="3.5" tabindex="0" data-tipx="${esc(tx(`${qTxt(S.set,i)}: answered right ${pc(p)} of the time after ${stepsTxt(r.t)}; with G = ${peqG}, ${pc(v)} of its groups are all right or all wrong.`,`${qTxt(S.set,i)}：训练 ${r.t} 步后答对的概率是 ${pc(p)}。G = ${peqG} 时，它有 ${pc(v)} 的组全对或全错。`))}"/>`;});
    svg.innerHTML=h; svg.setAttribute('aria-label',tEl.textContent);
    key.innerHTML=tx(`Dashed: G = 2. Solid: G = 8. Dotted: G = 64. The thick line is the G you chose; its dots are the 8 questions after ${stepsTxt(r.t)}.`,`虚线是 G = 2，实线是 G = 8，点线是 G = 64。粗的是你选的 G，线上的圆点是训练 ${r.t} 步后的 8 道题。`);
    return;}
  const T=r.T, ser=view==='acc'?T.acc:(view==='eq'?T.eq:T.kl), gs=ghost?(view==='acc'?traj(ghost.set,ghost.G,ghost.std).acc:(view==='eq'?traj(ghost.set,ghost.G,ghost.std).eq:traj(ghost.set,ghost.G,ghost.std).kl)):null;
  let ymax=view==='kl'?Math.max(0.05,...ser,...(gs||[0]))*1.1:1;
  tEl.textContent=view==='acc'?tx('Average accuracy over the 8 questions, step by step','8 道题的平均答对率，逐步变化'):(view==='eq'?tx('Share of the 8 groups in each step whose answers all scored the same','每一步 8 组回答里，得分全一样的组占多少'):tx('KL from the reference model, averaged over the 8 questions (nats)','离参考模型的 KL，8 道题平均（nats）'));
  const X=s=>X0+(X1-X0)*s/MAXS, Y=v=>Y1-(Y1-Y0)*Math.min(1,v/ymax);
  for(let k=0;k<=4;k++){const v=ymax*k/4; h+=`<line class="grpo-grid" x1="${X0}" x2="${X1}" y1="${Y(v)}" y2="${Y(v)}"/><text class="grpo-ax" x="${X0-4}" y="${Y(v)+3}" text-anchor="end">${view==='kl'?f(v,2):Math.round(v*100)+'%'}</text>`;}
  for(let s=0;s<=MAXS;s+=10) h+=`<text class="grpo-ax" x="${X(s)}" y="${Y1+14}" text-anchor="middle">${s}</text>`;
  h+=`<text class="grpo-ax" x="${(X0+X1)/2}" y="${H-2}" text-anchor="middle">${esc(tx('training step','训练步数'))}</text>`;
  const lo=view==='eq'?1:0, path=(arr,upto)=>{let d=''; for(let s=lo;s<=upto;s++) d+=(s>lo?'L':'M')+X(s).toFixed(1)+','+Y(arr[s]).toFixed(1); return d;};
  if(gs) h+=`<path class="grpo-ghost" d="${path(gs,MAXS)}"/>`;
  if(r.t>=lo) h+=`<path class="grpo-line" d="${path(ser,r.t)}"/><circle class="grpo-dot" cx="${X(r.t)}" cy="${Y(ser[r.t])}" r="3.5"/>`;
  if(r.t>=lo) h+=`<text class="grpo-cl on" x="${Math.min(X(r.t)+6,X1+4)}" y="${Y(ser[r.t])-6}">${view==='kl'?f(ser[r.t],2):pc(ser[r.t])}</text>`;
  svg.innerHTML=h; svg.setAttribute('aria-label',tEl.textContent);
  key.innerHTML=(gs?`<span class="grpo-k"><svg width="22" height="10" aria-hidden="true"><line class="grpo-ghost" x1="2" y1="5" x2="20" y2="5"/></svg>${tx('dashed: your previous setting','虚线：你上一次的设置')}</span> `:'')+
    `<span class="grpo-k"><svg width="22" height="10" aria-hidden="true"><line class="grpo-line" x1="2" y1="5" x2="20" y2="5"/></svg>${tx('solid: this run','实线：这次运行')}</span>`;}
function drawMem(){
  const m=memRows(), t=NP*TRAINED/GB, fz=NP*FROZEN/GB, max=m[0].tot;
  $('#grpo-m-title',app).innerHTML=tx(`GPU memory for the models when training ${P8()}, all GPUs together (GB = 10⁹ bytes)`,`训练 ${P8()} 时，几个模型一共要占多少显存（所有 GPU 加起来，GB = 10⁹ 字节）`);
  const lg=(tip,sw,tt)=>`<span data-tipx="${esc(tip)}"><i class="sw ${sw}"></i><span>${tt}</span></span>`;
  $('#grpo-m-leg',app).innerHTML=lg(tx('BF16 weights: 2 bytes per parameter. A frozen model keeps only these.','BF16 权重：每个参数 2 字节。冻结的模型只有这一块。'),'w',tx('BF16 weights','BF16 权重'))+
    lg(tx('FP32 gradients: 4 bytes per parameter, only for a model being trained.','FP32 梯度：每个参数 4 字节，只有训练的模型才有。'),'grpo-sw-grad',tx('FP32 gradients','FP32 梯度'))+
    lg(tx('Optimizer state: the FP32 copy of the weights plus Adam’s m and v, 12 bytes per parameter.','优化器状态：FP32 权重副本加上 Adam 的 m 和 v，每个参数 12 字节。'),'grpo-sw-opt',tx('optimizer state','优化器状态'));
  const NAME={policy:[tx('policy','策略模型'),D_POL()],value:[tx('value model','价值模型'),D_VAL()],ref:[tx('reference','参考模型'),D_REF()],reward:[tx('reward model','奖励模型'),D_RM()]};
  const ROW={ppo:tx('PPO','PPO'),rm:tx('GRPO, reward model','GRPO，用奖励模型'),chk:tx('GRPO, checker program','GRPO，用判分程序')};
  $('#grpo-mem',app).innerHTML=m.map(row=>{
    const segs=row.models.map(k=>{const tr=k==='policy'||k==='value', gbv=tr?t:fz, w=gbv/max*100;
      const inner=tr?`<i class="grpo-mb w" style="width:${(2/18*100).toFixed(3)}%"></i><i class="grpo-mb grad" style="width:${(4/18*100).toFixed(3)}%"></i><i class="grpo-mb opt" style="width:${(12/18*100).toFixed(3)}%"></i>`:`<i class="grpo-mb w" style="width:100%"></i>`;
      const tip=NAME[k][1]+tx(' ','')+(tr?tx(`Trained: 18 bytes per parameter, ${gb(t)} (BF16 weights 16.06, FP32 gradients 32.12, optimizer state 96.36 GB).`,`要训练：每个参数 18 字节，${gb(t)}（BF16 权重 16.06、FP32 梯度 32.12、优化器状态 96.36 GB）。`):tx(`Frozen: 2 bytes per parameter, ${gb(fz)}.`,`冻结：每个参数 2 字节，${gb(fz)}。`));
      return `<span class="grpo-mseg${k==='value'?' val':''}" style="width:${w.toFixed(3)}%" data-tipx="${esc(tip)}" tabindex="0"><span class="grpo-mbar">${inner}</span><span class="grpo-mlab">${tr?NAME[k][0]:''}</span></span>`;}).join('');
    const fr=row.models.filter(k=>k!=='policy'&&k!=='value').map(k=>NAME[k][0]).join(tx(' and ','和'));
    return `<div class="grpo-mrow"><div class="grpo-mname">${ROW[row.k]}</div><div class="grpo-mtrack">${segs}</div><div class="grpo-mtot mono">${gb(row.tot)}</div><div class="grpo-mfz small">${fr.includes(tx(' and ','和'))?tx(`small blocks: ${fr}, frozen, ${gb(fz)} each`,`小块：${fr}，冻结，各 ${gb(fz)}`):tx(`small block: ${fr}, frozen, ${gb(fz)}`,`小块：${fr}，冻结，${gb(fz)}`)}</div></div>`;}).join('');
  $('#grpo-m-note',app).innerHTML=tx(`GRPO drops the value model and saves ${gb(m[0].tot-m[1].tot)}, but the reward model stays. The chart assumes it is as large as the policy (8B): its frozen weights take ${gb(fz)}, and it reads and scores every answer. Scoring with a checker program has no such block. Without a ${DEF('KL penalty',D_KL())} the reference model need not be loaded either; then GRPO takes ${gb(m[1].tot-fz)} with a reward model and ${gb(t)} with a checker. TRL, a widely used training library, does this by default for GRPO: its KL coefficient β defaults to 0. Activations and the KV cache of generation are not in the chart.`,
    `GRPO 去掉了价值模型，省下 ${gb(m[0].tot-m[1].tot)}，但奖励模型还在。图里假设它和策略模型一样大（都是 8B），冻结的权重要 ${gb(fz)}，每个回答还要它读一遍、打一次分。用判分程序打分就没有这一块。不加 ${DEF('KL 惩罚',D_KL())}时，参考模型也不用加载，这时用奖励模型要 ${gb(m[1].tot-fz)}，用判分程序要 ${gb(t)}。常用的训练库 TRL 的 GRPO 默认就不加，KL 系数 β 默认是 0。图里没算激活值，也没算生成回答时的 KV cache。`);}
const MET=[
 {k:'acc',label:()=>tx('Average accuracy','平均答对率'),v:r=>pc(r.acc),sub:r=>tx(`8 questions, ${pc(r.T.acc[0])} before training`,`8 道题，训练前 ${pc(r.T.acc[0])}`),tip:()=>tx('The policy’s probability of the right answer, averaged over the 8 questions.','策略模型答对的概率，在 8 道题上取平均。'),pts:true,dp:0},
 {k:'eq',label:()=>tx('All-same groups this step','这一步得分全一样的组'),v:r=>r.t===0?'—':`${Math.round(r.eq*NQ)} / ${NQ}`,sub:r=>r.t===0?tx('no step yet','还没训练'):tx('they teach nothing','它们什么也学不到'),tip:()=>tx('How many of this step’s 8 groups had answers that all scored the same: every advantage in them is 0.','这一步的 8 组回答里，有几组得分全一样：这些组的优势全是 0。')},
 {k:'gen',label:()=>tx('Answers generated','已生成的回答'),v:r=>A.fmt.int(r.gen),sub:()=>tx(`8 × ${S.G} per step`,`每一步 8 × ${S.G} 个`),tip:()=>tx('Every step generates G answers for each of the 8 questions.','每一步给 8 道题各生成 G 个回答。')},
 {k:'kl',label:()=>tx('KL from the reference','离参考模型的 KL'),v:r=>f(r.kl,2)+' nats',sub:()=>tx('average over 8 questions','8 道题平均'),tip:()=>tx('How far the policy has moved from the reference model, in nats, averaged over the questions.','策略模型离参考模型有多远，单位 nats，在各题上取平均。')}];
let metKey=null;
function drawMetrics(r){
  const host=$('#grpo-metrics',app), rk=runKey(S), cfgCh=metKey!=null&&metKey!==rk; metKey=rk;   // ▲▼ only when a setting changed, never from training steps (S21)
  if(!host.children.length||host._lang!==A.lang()){host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" data-tipx="${esc(m.tip())}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); host._lang=A.lang();}
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), v=m.v(r), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(r);
    let dt=''; if(userAct&&cfgCh&&!(anim&&anim.kind==='run')&&shownPrev[m.k]!=null&&shownPrev[m.k]!==v&&m.k==='acc') dt=A.fmt.delta(parseFloat(shownPrev[m.k])/100,parseFloat(v)/100,{pts:true,dp:0});
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2800);} else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=v;});}
function drawHead(r){
  const st=$('#grpo-st-title',app), c=ctx(), el=$('#grpo-status',app);
  el.hidden=c.panels[0]==='mem'&&mode==='guided'&&STEPS[step].id==='mission';
  st.innerHTML=c.panels[0]==='mem'&&mode==='guided'&&STEPS[step].id==='mission'?tx(`The models PPO and GRPO keep in GPU memory when training ${P8()}`,`训练 ${P8()} 时，PPO 和 GRPO 放在显存里的模型`):c.demo?tx('One question, one group of answers','一道题，一组回答'):(r.t===0?tx(`The small model before training (${S.G} answers per question per step)`,`训练前的小模型（每一步每道题 ${S.G} 个回答）`):tx(`The small model after ${stepsTxt(r.t)} of GRPO (${S.G} answers per question per step)`,`训练 ${r.t} 步 GRPO 后的小模型（每一步每道题 ${S.G} 个回答）`));
  if(mode==='challenge'){const p=chPoint(S.G), done=chDone(); el.className='status'+(done?(p.ok?' ok':' bad'):''); el.textContent=done?(p.ok?tx('Reaches 80%','达到 80%'):tx('Under 80%','不到 80%')):(chSubmitted?tx(`Training: step ${S.steps}`,`训练中：第 ${S.steps} 步`):tx('Not trained yet','还没训练')); el.dataset.tipx=tx('Whether the average accuracy after 30 steps reaches the 80% goal. It shows once you submit and the 30 steps have run.','30 步后的平均答对率是否达到 80% 的目标。提交并训练完 30 步后才显示。'); return;}
  if(c.demo){const g=demoGroup(c.demo,S.std), hide=mode==='guided'&&STEPS[step].id==='group'&&!demoShown, ga=anim&&anim.kind==='group'?anim.ph:9; el.className='status'; el.textContent=hide?tx('Not generated yet','还没生成'):ga===0?tx('Sampling','抽样中'):ga===1?tx('Scoring','打分中'):tx(`${g.r.filter(v=>v).length} of ${g.o.length} right`,`${g.o.length} 个里 ${g.r.filter(v=>v).length} 个对`); el.dataset.tipx=tx('How many answers in this group the checker scored as right.','这一组里，判分程序判对的回答有几个。'); return;}
  /* H8 (the owner, 2026-10-09): while “+1 step” replays a group, the status says which part of the step is playing; sampling is not training */
  if(anim&&anim.kind==='group'){el.className='status'; el.textContent=[tx('Sampling','抽样中'),tx('Scoring','打分中'),tx('Baseline','算基准'),tx('Advantages','算优势'),tx('Updating','更新中')][anim.ph]||''; el.dataset.tipx=tx('Which part of this GRPO step the picture is replaying: sampling the answers, scoring them, the baseline, the advantages, then the update of the policy.','图里正在重放这一步 GRPO 的哪一段：抽样生成回答、打分、算基准、算优势，最后才更新策略模型。'); return;}
  el.className='status'; el.textContent=tx(`Accuracy ${pc(r.acc)}`,`答对率 ${pc(r.acc)}`); el.dataset.tipx=tx('The policy’s average probability of the right answer over the 8 questions now.','现在策略模型在 8 道题上答对的平均概率。');}
function drawTransport(){
  const t=$('#grpo-transport',app), k=ctxKind(), want=k+'|'+A.lang()+'|'+(anim&&anim.kind==='run'?1:0)+'|'+(k==='run'?runSpeed:grpSpeed)+'|'+mode;
  t.hidden=ctx().noTr; if(t._k===want) return; t._k=want;
  t.innerHTML=k==='run'?`<button type="button" class="btn sm" data-act="play" data-play data-tipx="${esc(tx('Train the small model step by step to 30 steps; press again to pause.','把小模型逐步训练到 30 步；再按一次暂停。'))}">${playLabel()}</button>${speedSeg()}<button type="button" class="btn ghost sm" data-act="step1" data-tipx="${esc(tx('Take one GRPO step and replay its group: sample, score, baseline, advantage, update.','走一步 GRPO，并重放它的一组回答：抽样、打分、基准、优势、更新。'))}">${tx('+1 step','+1 步')}</button><button type="button" class="btn ghost sm" data-act="reset0" data-tipx="${esc(tx('Back to the policy before training.','回到训练前的策略模型。'))}">${tx('Back to step 0','回到第 0 步')}</button><button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch this run again from step 0.','从第 0 步再看一遍这次运行。'))}">${tx('Replay','重放')}</button>`
    :`${speedSeg()}<button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch this group again: sample, score, baseline, advantage.','把这一组再看一遍：抽样、打分、基准、优势。'))}">${tx('Replay','重放')}</button>`;}
function panelsShow(){const c=ctx(); ['group','probs','chart','mem'].forEach(p=>{const el=$('#grpo-pan-'+p,app); if(el) el.hidden=!c.panels.includes(p);}); const mt=$('#grpo-metrics',app); if(mt) mt.hidden=!c.panels.includes('metrics');
  /* polish r4 (R7.12, B4): which panels share a row on a wide stage (grpo.css). Part 5: the group beside the bars, the chart under both;
     part 6: the bars beside the chart. Any other set of panels keeps the stacked layout */
  const pans=$('#grpo-pans',app), on=p=>c.panels.includes(p), lay=on('probs')&&on('chart')&&!on('mem')?(on('group')?'gpc':'pc'):'';
  [pans,$('#grpo-stage',app)].forEach(e=>{if(e&&(e.dataset.lay||'')!==lay){if(lay) e.dataset.lay=lay; else delete e.dataset.lay;}});}
/* Show the math (the owner, 2026-10-09: “like a page of a good textbook”): four short blocks, each a lead line in words, the quantity's
   name and formula, then the same formula with the numbers of the group in the chart and the result */
function mathHTML(r){
  const c=ctx(), g=c.demo?demoGroup(c.demo,S.std):(r.grp?r.grp[S.fq]:null), on=S.std==='on';
  const B=(t,x)=>`<p class="grpo-mb-lead"><b>${t}</b>${tx(' ','')}${x}</p>`, SUB=x=>`<p class="grpo-mb-sub">${x}</p>`, NM=(en,zh)=>`\\text{${tx(en,zh)}}\\,\\,`;
  const k=g?g.r.filter(v=>v).length:0, n=g?g.o.length:0, mixed=!!g&&k>0&&k<n, mT=g?nz(g.mean,4):'';
  /* the std is shown with enough digits that the shown advantages can be recomputed from it (R6.6), and no more (R6.20) */
  let sdT='0';
  if(mixed){const sdE=Math.sqrt(k*(n-k)/(n*(n-1))); sdT=nz(sdE,3); const rec=v=>sg((v-g.mean)/(+sdT+ADV_EPS),2);
    if(rec(1)!==sg(g.adv[g.r.indexOf(1)],2)||rec(0)!==sg(g.adv[g.r.indexOf(0)],2)) sdT=nz(sdE,4);}
  let h='';
  h+=B(tx('1. The group’s mean and standard deviation.','1. 这一组的平均分和标准差。'),
    tx(`The baseline is the mean of the group’s rewards. The standard deviation divides by ${MI('n - 1')}, as TRL does. With rewards of only 0 and 1, ${MI('k')} right out of ${MI('n')}, both follow from ${MI('k')} and ${MI('n')}:`,
       `基准是这一组奖励的平均分。标准差的分母用 ${MI('n - 1')}，和 TRL 一样。奖励只有 0 和 1 时，${MI('n')} 个回答里 ${MI('k')} 个对，两者都可以只用 ${MI('k')} 和 ${MI('n')} 算：`));
  h+=EQ([[NM('mean','平均分')+'\\text{mean}(r)','\\frac{k}{n}'],[NM('std','标准差')+'\\text{std}(r)','\\frac{\\sqrt{k (n - k)}}{\\sqrt{n (n - 1)}}']]);
  if(g){h+=SUB(tx(`The group in the chart: ${k} of ${n} answers right.`,`图里这一组：${n} 个回答里 ${k} 个对。`));
    h+=EQ([['\\text{mean}(r)',`\\frac{${k}}{${n}}`],['',mT],['\\text{std}(r)',`\\frac{\\sqrt{${k} \\times ${n-k}}}{\\sqrt{${n} \\times ${n-1}}}`],['',sdT]]);}
  h+=B(tx('2. Each answer’s advantage.','2. 每个回答的优势。'),on?
    tx('The reward minus the mean, divided by the standard deviation. The 0.0001 keeps a group that scored all the same from dividing by 0.','奖励减去平均分，再除以标准差。分母加上 0.0001，是为了得分全一样的组不会除以 0。')
    :tx('Without the division, the advantage is only the reward minus the mean.','不除以标准差时，优势只是奖励减去平均分。'));
  h+=EQ([[NM('advantage','优势')+'A_i',on?'\\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}':'r_i - \\text{mean}(r)']]);
  h+=`<p class="small">${tx(`${MI('r_i')} is the reward of answer ${MI('i')}: 1 if right, 0 if wrong.`,`${MI('r_i')} 是第 ${MI('i')} 个回答的奖励，答对是 1，答错是 0。`)}</p>`;
  if(mixed){const ai=g.adv[g.r.indexOf(1)], aw=g.adv[g.r.indexOf(0)], v=x=>on?sg(x,2):sgz(x,4), RT=tx('right','对'), WT=tx('wrong','错');
    h+=SUB(tx('The group in the chart:','图里这一组：'));
    h+=EQ([[`A_{\\text{${RT}}}`,on?`\\frac{1 - ${mT}}{${sdT} + 0.0001}`:`1 - ${mT}`],['',v(ai)],[`A_{\\text{${WT}}}`,on?`\\frac{0 - ${mT}}{${sdT} + 0.0001}`:`0 - ${mT}`],['',v(aw)]]);}
  else if(g) h+=SUB(tx('In the group in the chart all answers scored the same, so every advantage is 0.','图里这一组得分全一样，所以每个优势都是 0。'));
  h+=B(tx('3. The loss of the update.','3. 更新用的 loss。'),
    tx(`It pushes up the probability of answers with a positive advantage and pushes down the others. Once an answer’s probability ratio ${MI('\\rho_i')} has moved past 0.8 or 1.2 in the direction it is pushed, the clip stops pushing it. The KL term keeps the policy near the reference model.`,
       `它把优势为正的回答的概率往上推，把为负的往下压。一个回答的概率比 ${MI('\\rho_i')} 朝推的方向越过 0.8 或 1.2，截断就不再推它。KL 项让策略模型别离参考模型太远。`));
  h+=EQ([[NM('ratio','概率比')+'\\rho_i','\\frac{\\pi_\\theta(o_i)}{\\pi_\\text{old}(o_i)}'],['u_i','\\log \\pi_\\text{ref}(o_i) - \\log \\pi_\\theta(o_i)'],
    ['c_i','\\text{clip}(\\rho_i, 1 - \\epsilon, 1 + \\epsilon)'],['\\text{loss}\\,\\,L','-\\text{mean}_i \\big[ \\min(\\rho_i A_i, \\, c_i A_i)'],['','- \\beta (e^{u_i} - u_i - 1) \\big]',true]]);
  h+=`<p class="small">${tx(`${MI('\\pi_\\theta')} is the policy being trained, ${MI('\\pi_\\text{old}')} the policy that wrote the answers, ${MI('\\pi_\\text{ref}')} the reference model; ${MI('\\epsilon = 0.2')} and ${MI('\\beta = 0.05')}. ${MI('c_i')} is ${MI('\\rho_i')} held between 0.8 and 1.2, and ${MI('e^{u_i} - u_i - 1')} is how TRL estimates the KL of one answer.`,
    `${MI('\\pi_\\theta')} 是正在训练的策略模型，${MI('\\pi_\\text{old}')} 是写出这些回答时的策略模型，${MI('\\pi_\\text{ref}')} 是参考模型。${MI('\\epsilon = 0.2')}，${MI('\\beta = 0.05')}。${MI('c_i')} 是被限制在 0.8 到 1.2 之间的 ${MI('\\rho_i')}，${MI('e^{u_i} - u_i - 1')} 是 TRL 估计一个回答的 KL 的式子。`)}</p>`;
  h+=SUB(tx(`This lab updates once per batch. During the update ${MI('\\pi_\\theta')} is still ${MI('\\pi_\\text{old}')}, so every ${MI('\\rho_i = 1')} and the clip never acts.`,
    `这个 Lab 每批回答只更新一次。更新时 ${MI('\\pi_\\theta')} 还是 ${MI('\\pi_\\text{old}')}，所以每个 ${MI('\\rho_i = 1')}，截断从不起作用。`));
  h+=B(tx('4. The chance that a group teaches nothing.','4. 一组学不到东西的概率。'),
    tx(`For a question answered right with probability ${MI('p')}, a group of ${MI('G')} answers is all right or all wrong with probability`,`一道题答对的概率是 ${MI('p')} 时，一组 ${MI('G')} 个回答全对或全错的概率是`));
  h+=EQ([['P','p^G + (1 - p)^G']]);
  if(g){const p2=+g.p.toFixed(2), P=Math.pow(p2,n)+Math.pow(1-p2,n);
    h+=SUB(tx(`The question in the chart: ${MI(`p = ${nz(p2,2)}`)} when this group was drawn, ${MI(`G = ${n}`)}.`,`图里这道题：抽这一组时 ${MI(`p = ${nz(p2,2)}`)}，${MI(`G = ${n}`)}。`));
    h+=EQ([['P',`${nz(p2,2)}^{${n}} + ${nz(1-p2,2)}^{${n}}`],['',P<0.0005?'< 0.001':f(P,3)]]);}
  return h;}

/* ---------- controls, rail sync ---------- */
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(gEl=>{const k=gEl.dataset.key; if(k==='view') A.ui.setPressed(gEl,view); else if(k==='speed') A.ui.setPressed(gEl,ctxKind()==='run'?runSpeed:grpSpeed); else if(k==='peq') A.ui.setPressed(gEl,String(peqG)); else if(S[k]!=null) A.ui.setPressed(gEl,String(S[k]));});
  $$('input[data-key="steps"]',app).forEach(i=>{if(+i.value!==S.steps) i.value=S.steps; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl'); const out=o&&o.querySelector('output'); if(out) out.textContent=stepsTxt(S.steps);});
  const lock=mode==='challenge', c=$('#grpo-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#grpo-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you can change only G, in the task card. The rest is fixed: the 8 new questions, 30 steps, dividing by the std, β = 0.05, ε = 0.2.','挑战中只能在任务卡里改 G。其余设置固定：8 道新题、训练 30 步、除以标准差、β = 0.05、ε = 0.2。');
  $$('#grpo-controls .lockable',app).forEach(e=>{e.hidden=lock;});}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(r){
  const rail=$('#grpo-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect(r)&&!(anim&&anim.kind==='group'&&s.id==='group'&&anim.ph<2)) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]);
    if(box){const was=box.hidden; box.hidden=!show; if(show&&was&&anim&&anim.kind==='group') revealObs(); if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(r); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; if(again&&!(anim&&anim.kind==='run')) A.ui.obsBump(box);}}}
    A.ui.setNext(rail,isNextDisabled());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-play]',rail).forEach(b=>{b.textContent=playLabel();});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#grpo-insight',rail); if(ins){const i=presetNow(), html=`<div class="eyebrow">${i>=0?esc(L(PRESETS[i].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const p=chPoint(S.G), av=$('#grpo-ch-av',rail), done=chDone(); if(!av) return;
    const live=compute(S).acc, ok=done&&p.ok;
    av.textContent=done?pc(p.acc):(chSubmitted?tx(`step ${S.steps}: ${pc(live)}`,`第 ${S.steps} 步：${pc(live)}`):tx('shown after you submit','提交后才知道')); $('#grpo-ch-gv',rail).textContent=tx(`${A.fmt.int(p.gen)} (8 × ${S.G} × 30)`,`${A.fmt.int(p.gen)} 个（8 × ${S.G} × 30）`);
    $('#grpo-ch-ap i',rail).style.width=(chSubmitted?Math.min(100,(done?p.acc:live)/CH_GOAL*100):0)+'%'; $('#grpo-ch-a',rail).classList.toggle('met',ok); $('#grpo-ch-ap',rail).classList.toggle('met',ok);
    const sub=$('#grpo-ch-sub',rail); if(sub) sub.hidden=chSubmitted;
    if(done&&p.ok) chReveal=true;
    const res=$('#grpo-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:s.nextWhy?L(s.nextWhy):undefined,skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#grpo-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  const r=compute(S);
  panelsShow(); drawTransport(); drawHead(r);
  if(!$('#grpo-pan-group',app).hidden){const g=shownGroup(r); drawGroupLegend(!!g&&redOn()&&S.std==='on'&&g.adv.some(a=>Math.abs(a)>2)&&ctx().phase>=3); drawGroup(r);}
  if(!$('#grpo-pan-probs',app).hidden) drawProbs(r);
  if(!$('#grpo-pan-chart',app).hidden) drawChart(r);
  if(!$('#grpo-pan-mem',app).hidden) drawMem();
  if(!$('#grpo-metrics',app).hidden) drawMetrics(r);
  const mb=$('#grpo-mathbody',app), mh=mathHTML(r); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail(r);
  A.hash.write(token()); A.dock.refresh(); announce(r);}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; A.announce(tx(`After ${stepsTxt(r.t)}: average accuracy ${pc(r.acc)}.`,`训练 ${r.t} 步后：平均答对率 ${pc(r.acc)}。`));},400);}
const cfgOf=s=>({set:s.set,G:s.G,std:s.std});
function setState(p){
  const before=cfgOf(S); Object.assign(S,p);
  if(mode==='challenge'){const G=S.G, keep=S.steps; Object.assign(S,CH); S.G=CH_GS.includes(G)?G:2;
    if(S.G!==before.G){if(chReveal) S.steps=MAXS; else {chSubmitted=false; S.steps=0;}} else S.steps=keep;}
  else if(!GS.includes(S.G)) S.G=8;
  S.steps=Math.max(0,Math.min(MAXS,Math.round(S.steps)));
  if(S.fq<0||S.fq>=NQ) S.fq=SETS[S.set].f;
  const now=cfgOf(S); if(mode!=='guided'&&(before.set!==now.set||before.G!==now.G||before.std!==now.std)) ghost=before.set===now.set&&(mode!=='challenge'||chReveal)?before:null;   // no ghost of another G's run before the challenge reveals them (S19)
  render();}
/* ---------- animation (R3.9, R3.20) ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} const was=anim; anim=null; lastTs=0; return was;}
function finishAnim(){const a=stopAnim(); if(!a) return; if(a.kind==='group') markGroup(a); if(a.kind==='run'&&S.steps>=MAXS) seenRun.add(a.rk); if(app) render();}
function markGroup(a){seenGroup.add(a.gk); if(a.maxPh>=3) seenAdv.add(a.key); if(a.maxPh>=4) seenAdv.add(a.key+'|upd');}
/* start the group replay where the picture first differs from what the reader has already watched */
function groupStart(force){
  const r=compute(S), g=shownGroup(r), c=ctx(); if(!g) return;
  const key=g.key+'|'+S.std, maxPh=c.phase;
  let from=force?0:(!seenGroup.has(g.key)?0:(maxPh>=3&&!seenAdv.has(key)?3:(maxPh>=4&&!seenAdv.has(key+'|upd')?4:-1)));
  if(from<0){render(); return;}
  stopAnim();
  if(A.reduceMotion()){anim={kind:'group',gk:g.key,key,maxPh,ph:maxPh,u:1}; finishAnim(); return;}
  anim={kind:'group',gk:g.key,key,maxPh,ph:from,u:0}; tickStart();}
function runStart(fromZero){
  stopAnim(); if(fromZero) S.steps=0; if(S.steps>=MAXS){render(); return;}
  if(A.reduceMotion()){S.steps=MAXS; seenRun.add(runKey(S)); render(); return;}
  anim={kind:'run',rk:runKey(S),acc:0}; tickStart();}
function tickStart(){lastTs=0; render(); raf=requestAnimationFrame(tick);}
function tick(ts){
  raf=0; if(!anim||!app) return;
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts;
  if(anim.kind==='run'){anim.acc+=dt*RUN_SPEEDS[runSpeed]; const n=Math.floor(anim.acc); if(n>0){anim.acc-=n; S.steps=Math.min(MAXS,S.steps+n); if(S.steps>=MAXS){finishAnim(); return;} render();}}
  else {const G=(shownGroup(compute(S))||{o:[0]}).o.length, dur=PH_DUR[anim.ph]*(anim.ph===0?Math.min(1,G/8):1)/GRP_SPEEDS[grpSpeed];
    anim.u+=dt/dur; if(anim.u>=1){if(anim.ph>=anim.maxPh){finishAnim(); return;} anim.ph++; anim.u=0; render();} else drawGroup(compute(S));}
  raf=requestAnimationFrame(tick);}

/* ---------- navigation ---------- */
function enterStep(i){
  stopAnim(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter){Object.assign(S,s.enter()); ghost=null;}
  if(s.view) view=s.view;
  if(s.id==='zero') triedPeq=new Set([peqG]);
  if(s.id==='std'){S.std='on'; triedStd=new Set(['on']);}
  const st=$('#grpo-stage',app); if(st) st.scrollTop=0;
  renderRail();
  if(s.demo&&(s.id!=='group'||demoShown)) groupStart(false);}
function revealStep(){A.ui.reveal($('#grpo-rail .step.current',app)||$('#grpo-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.bottom<=innerHeight-8) return;
  const dy=Math.min(r.bottom-innerHeight+16,r.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>requestAnimationFrame(()=>{if(!app) return; if(mode==='guided') revealBox($('#grpo-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#grpo-insight',app)); else revealBox($('#grpo-ch-res',app));});
function setMode(m){
  if(m===mode) return; stopAnim();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,...CH,G:2,fq:0}; view='acc'; ghost=null;}
  if(m==='sandbox'&&was==='guided'){S={...DEFAULT,steps:seenRun.has(runKey(DEFAULT))?MAXS:0}; view='acc'; ghost=null;}
  A.shell.setMode(app,m,DESCS());
  const st=$('#grpo-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
function applyPreset(i){const p=PRESETS[i]; stopAnim(); ghost=null; Object.assign(S,p.p); S.fq=SETS[S.set].f;
  const rk=runKey(S); if(seenRun.has(rk)){S.steps=MAXS; byLearner(render);} else {S.steps=0; byLearner(()=>runStart(true));}
  revealObs();}
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
/* while a run plays, the settings are those it plays to (its last step), so the address and the saved record name where it ends,
   not a frame of it */
const token=s=>{if(!s){s=S; if(anim&&anim.kind==='run') s={...S,steps:MAXS};} return [s.set,s.G,s.std,s.steps,s.fq].join('_');};
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null; const [set,G,std,st,fq]=p, o={set,G:+G,std,steps:+st,fq:+fq};
  if(!SETS[set]||set==='ch'||!GS.includes(o.G)||!['on','off'].includes(std)||!Number.isInteger(o.steps)||o.steps<0||o.steps>MAXS||!Number.isInteger(o.fq)||o.fq<0||o.fq>=NQ) return null; return o;}
function act(a,btn){
  if(a==='gen'){demoShown=true; observed[step]=false; groupStart(false); renderRail(); revealObs(); return;}
  if(a==='play'){if(anim&&anim.kind==='run'){stopAnim(); render(); return;} if(S.steps>=MAXS){runStart(true); return;} byLearner(()=>runStart(false)); return;}
  if(a==='step1'){if(S.steps>=MAXS){A.tip.show(btn,`<div>${tx('This lab trains at most 30 steps, and you are there. Press “Back to step 0”.','这个 Lab 最多训练 30 步，已经到头了。点“回到第 0 步”。')}</div>`); return;} stopAnim(); A.tip.quiet(); byLearner(()=>{S.steps++; render();}); groupStart(false); return;}   // pressed repeatedly: no scroll (R1.11)
  if(a==='reset0'){stopAnim(); byLearner(()=>{S.steps=0; render();}); return;}
  if(a==='replay'){if(ctxKind()==='run') runStart(true); else groupStart(true); return;}
  if(a==='reset'){stopAnim(); byLearner(()=>{ghost=null; Object.assign(S,DEFAULT); render();}); return;}
  if(a==='submit'){chSubmitted=true; drawTransport(); if(seenRun.has(runKey(S))){S.steps=MAXS; render();} else runStart(true); revealObs(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#grpo-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function showMe(p){
  if(p.steps!=null){if(S.steps>=MAXS){render();} else runStart(false); revealObs(); return;}
  if(p.peq!=null){peqG=p.peq; triedPeq.add(peqG); byLearner(render); revealObs(); return;}
  if(p.std!=null){S.std=p.std; triedStd.add(p.std); byLearner(render); groupStart(false); revealObs();}}

/* ---------- interview drill (seeded) ---------- */
function makeDrill(seed){const rnd=mulberry(seed*7919+11), G=[4,6,8][Math.floor(rnd()*3)], r=[]; let k=0;
  for(let i=0;i<G;i++){const v=rnd()<0.4?1:0; r.push(v); k+=v;} if(k===0) r[0]=1; if(k===G) r[G-1]=0;
  return {seed,G,r,...groupStats(r,true)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#grpo-drill-q',app), a=$('#grpo-drill-a',app), k=d.r.filter(v=>v).length;
  q.innerHTML=tx(`A checker scored a group of ${d.G} answers to one question: rewards <b>[${d.r.join(', ')}]</b>. Compute each answer’s GRPO advantage as TRL does (std with ${MI('n - 1')} in the denominator, plus 1e-4), to 3 decimals. Which answers does the next update make more likely?`,
                 `判分程序给一道题的一组 ${d.G} 个回答打了分：奖励是 <b>[${d.r.join(', ')}]</b>。标准差按 TRL 的做法算，分母用 ${MI('n - 1')}，算出的标准差再加上 1e-4。算出每个回答的 GRPO 优势，保留 3 位小数。下一次更新会让哪些回答更可能出现？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const sdE=Math.sqrt(k*(d.G-k)/(d.G*(d.G-1))), ss=nz(k*(d.G-k)/d.G,4), m4=f(d.mean,4), s4=f(sdE,4), ar=sg(d.adv[d.r.indexOf(1)],3), aw=sg(d.adv[d.r.indexOf(0)],3);
  /* the worked answer typeset like Show the math (the owner, 2026-10-09): each line's “=” under the one above */
  const vr=k*(d.G-k)/d.G/(d.G-1), work=EQ([['\\text{mean}(r)',`\\frac{${k}}{${d.G}}`],['',m4],[`\\text{${tx('variance','方差')}}\\,\\,v`,'\\frac{\\sum_i (r_i - \\text{mean}(r))^2}{n - 1}'],['',`\\frac{${ss}}{${d.G-1}}`],['',f(vr,4)],['\\text{std}(r)','\\sqrt{v}'],['',`\\sqrt{${f(vr,4)}}`],['',s4],
    [`A_{\\text{${tx('right','对')}}}`,`\\frac{1 - ${m4}}{${s4} + 0.0001}`],['',ar],[`A_{\\text{${tx('wrong','错')}}}`,`\\frac{0 - ${m4}}{${s4} + 0.0001}`],['',aw]]);
  a.innerHTML=tx(`<p><b>right answers ${ar}, wrong answers ${aw}</b></p>${work}<p>The advantages sum to 0. The update raises the probability of the ${k} right answers and lowers that of the ${d.G-k} wrong ones; without a value model, the baseline was the group mean.</p>`,
                 `<p><b>答对的回答 ${ar}，答错的回答 ${aw}</b></p>${work}<p>优势加起来是 0。更新会提高 ${k} 个答对的回答的概率，降低 ${d.G-k} 个答错的。没有价值模型，基准就是组内平均分。</p>`);}

/* ---------- jumps ---------- */
const CTLN={G:{en:'Answers per question, G',zh:'每道题生成几个回答 G'},set:{en:'Questions',zh:'题目'},std:{en:'Divide by the std',zh:'除以标准差'},steps:{en:'Training steps',zh:'训练步数'}};
function jumpAct(k){
  if(k==='mem') return tx('Click to show the memory chart.','点击查看显存对比图。');
  if(k==='bars') return tx('Click to show the advantage bars.','点击查看优势的柱子。');
  if(k==='probs') return tx('Click to show the bars of each question.','点击查看每道题的概率条。');
  if(k==='chart') return tx('Click to show the chart.','点击查看曲线图。');
  if(k==='chart-eq') return tx('Click to show the share of groups that scored all the same.','点击显示得分全一样的组占多少。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(!CTLN[k]) return '';
  if(mode==='challenge'&&k!=='G') return tx(`Fixed during the challenge: “${L(CTLN[k])}”.`,`挑战中固定不变：${zsp(L(CTLN[k]))}。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);}
function jump(k,src){
  if(!app) return;
  /* a link marks the picture it names, never the whole panel or stage (the owner, 2026-10-09: “highlight的是整个动画面板”) */
  const pan=(id,mark)=>{const el=$('#grpo-pan-'+id,app); if(el&&el.hidden){el.hidden=false;} A.tip.hide(); const m=mark&&$(mark,app);
    if(el){(m||el).scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); if(m) A.ui.flash(m);}};
  if(k==='mem'){pan('mem','#grpo-mem'); if(!$('#grpo-mem',app).children.length) drawMem(); return;}
  if(k==='bars'){pan('group'); barsHL=performance.now()+1700; render(); clearTimeout(barsT); barsT=setTimeout(()=>{barsHL=0; if(app) render();},1750); return;}
  if(k==='probs'){pan('probs','#grpo-probs'); drawProbs(compute(S)); return;}
  if(k==='chart'){pan('chart','#grpo-csvg'); return;}
  if(k==='chart-eq'){view='eq'; render(); pan('chart','#grpo-csvg'); return;}
  if(k==='controls'){const c=$('#grpo-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(!CTLN[k]) return;
  if(mode==='challenge'&&k!=='G'){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=$('#grpo-ic-'+k,app)||$('#grpo-ctl-'+k,app); if(el) A.ui.jumpToEl(el);}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tickS=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tickS,120); return;}
    const st=$('#grpo-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#grpo-rail .step.current',app)||$('#grpo-rail .railcard',app)||$('#grpo-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tickS,A.reduceMotion()?30:150);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='steps'){stopAnim(); byLearner(()=>setState({steps:+e.target.value}));}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#grpo-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key;
      if(k==='view'){view=d.v; render(); return;}
      if(k==='speed'){if(ctxKind()==='run') runSpeed=d.v; else grpSpeed=d.v; render(); return;}
      if(k==='peq'){peqG=+d.v; triedPeq.add(peqG); byLearner(render); if(seg.closest('#grpo-rail')) revealObs(); return;}
      if(k==='std'){stopAnim(); S.std=d.v; triedStd.add(d.v); byLearner(render); if(ctx().demo) groupStart(false); if(seg.closest('#grpo-rail')) revealObs(); return;}
      stopAnim(); byLearner(()=>setState({[k]:k==='G'?+d.v:d.v})); if(seg.closest('#grpo-rail')) revealObs(); return;}
    if(d.fq!=null){stopAnim(); S.fq=+d.fq; render(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#grpo-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#grpo-rail',app)); return;}
    if(d.act) act(d.act,b);});}
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the dashed earlier run, the part, the answer,
   whether the group was generated, the group size and the switch tried, the chart shown, the groups and runs already watched, the
   two speeds, the challenge's choice and result, the drill. A run or group replay still playing is recorded as played to its end:
   after the reload it shows its end. */
const rsKeys=Object.keys(DEFAULT), rsObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), rsPick=o=>Object.fromEntries(rsKeys.map(k=>[k,o[k]]));
const rsKey=v=>typeof v==='string'&&v.length<=60&&/^[\w|.\-]+$/.test(v), rsSpd=v=>['slow','mid','fast'].includes(v);
A.registerLab({
  id:ID,
  snap(){const run=anim&&anim.kind==='run', sg=new Set(seenGroup), sa=new Set(seenAdv), sr=new Set(seenRun);
    if(run) sr.add(anim.rk); else if(anim&&anim.kind==='group'){sg.add(anim.gk); if(anim.maxPh>=3) sa.add(anim.key); if(anim.maxPh>=4) sa.add(anim.key+'|upd');}
    return {v:1,mode,step,S:rsPick(run?{...S,steps:MAXS}:S),pred,observed:{...observed},saved:saved?rsPick(saved):null,ghost:ghost?{set:ghost.set,G:ghost.G,std:ghost.std}:null,
      demoShown,peqG,view,triedPeq:[...triedPeq],triedStd:[...triedStd],seenGroup:[...sg],seenAdv:[...sa],seenRun:[...sr],runSpeed,grpSpeed,
      chSubmitted,chReveal,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPoint(S.G).ok};},
  restore(o){if(!rsObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    if(mode==='challenge'){const x=rsObj(o.S)?o.S:{}; S={...DEFAULT,...CH,G:CH_GS.includes(x.G)?x.G:2,fq:0,steps:Number.isInteger(x.steps)&&x.steps>=0&&x.steps<=MAXS?x.steps:0};}
    else S={...((rsObj(o.S)&&parse(token(o.S)))||DEFAULT)};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(rsObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&typeof o.observed[k]==='boolean') observed[k]=o.observed[k];
    saved=mode==='challenge'&&rsObj(o.saved)?parse(token(o.saved)):null;
    const g=o.ghost; ghost=rsObj(g)&&SETS[g.set]&&Object.prototype.hasOwnProperty.call(SETS,g.set)&&CH_GS.includes(g.G)&&['on','off'].includes(g.std)?{set:g.set,G:g.G,std:g.std}:null;
    demoShown=o.demoShown===true; peqG=PEQ_GS.includes(o.peqG)?o.peqG:8; view=VIEWS.some(x=>x[0]===o.view)?o.view:'acc';
    triedPeq=new Set(arr(o.triedPeq,v=>PEQ_GS.includes(v))); triedStd=new Set(arr(o.triedStd,v=>v==='on'||v==='off'));
    [[seenGroup,o.seenGroup],[seenAdv,o.seenAdv],[seenRun,o.seenRun]].forEach(([w,a])=>{w.clear(); arr(a,rsKey).slice(0,300).forEach(v=>w.add(v));});
    runSpeed=rsSpd(o.runSpeed)?o.runSpeed:'fast'; grpSpeed=rsSpd(o.grpSpeed)?o.grpSpeed:'mid';
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    stopAnim(); root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps the dashed earlier run */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) ghost=null; S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#grpo-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#grpo-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    if(mode==='guided'){const s=STEPS[step]; if(s.view) view=s.view;}
    renderRail();},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); clearTimeout(barsT); barsHL=0; root=null; app=null;},
  onLang(){if(root){const d=$('#grpo-math',app), open=d&&d.open; const a=anim; stopAnim(); this.mount(root,null); if(open) $('#grpo-math',app).open=true; if(a&&a.kind==='run') runStart(false);}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#grpo-pan-group:not([hidden])',app)||$('#grpo-pan-mem:not([hidden])',app)||$('#grpo-pan-chart',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#grpo-stage',app),
    render({status,vis,read}){if(!app) return; const r=compute(S), c=ctx();
      vis.innerHTML=`<div class="grpo-dk"><i style="width:${(r.acc*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const p=chPoint(S.G), done=chDone(); status.className='status'+(done?(p.ok?' ok':' bad'):''); status.textContent=done?(p.ok?tx('Goal met','已达标'):tx('Not yet','未达标')):(chSubmitted?tx('Training','训练中'):tx('Not submitted','未提交'));
        read.textContent=done?tx(`With ${S.G} answers per question, 30 steps reach ${pc(p.acc)} average accuracy`,`每道题 ${S.G} 个回答，30 步后平均答对率 ${pc(p.acc)}`):tx(`With ${S.G} answers per question, 30 steps generate ${A.fmt.int(p.gen)} answers`,`每道题 ${S.G} 个回答，30 步一共生成 ${A.fmt.int(p.gen)} 个回答`); return;}
      if(c.demo){const g=demoGroup(c.demo,S.std), hide=mode==='guided'&&STEPS[step].id==='group'&&!demoShown, early=anim&&anim.kind==='group'&&anim.ph<2; status.className='status'; status.textContent=tx('One group','一组回答');
        read.textContent=hide?tx('The group has not been generated yet','这一组还没生成'):early?tx('The group is being sampled and scored','这一组正在抽样、打分'):tx(`${g.r.filter(v=>v).length} of ${g.o.length} answers right; baseline ${f(g.mean,2)}`,`${g.o.length} 个回答 ${g.r.filter(v=>v).length} 个对，基准 ${f(g.mean,2)}`); return;}
      status.className='status'+(r.acc>=0.8?' ok':''); status.textContent=tx(`Step ${r.t}`,`第 ${r.t} 步`);
      read.textContent=r.t===0?tx(`Before training the policy answers ${pc(r.acc)} right on average`,`训练前，策略模型平均答对 ${pc(r.acc)}`):tx(`After ${stepsTxt(r.t)} the policy answers ${pc(r.acc)} right on average`,`训练 ${r.t} 步后，策略模型平均答对 ${pc(r.acc)}`);}}
});
/* read-only hook for tests/grpo_numbers.py */
window.__grpo={state:()=>({mode,step,set:S.set,G:S.G,std:S.std,steps:S.steps,fq:S.fq,pred,demoShown,peqG,view,chSubmitted,chReveal}), animState:()=>anim?{kind:anim.kind,ph:anim.ph}:null, seen:()=>({group:[...seenGroup],adv:[...seenAdv],run:[...seenRun]}), train:(set,G,std)=>{const T=traj(set,G,std); return {acc:T.acc,kl:T.kl,eq:T.eq,gen:T.gen,clip:T.clip,act:T.act,o:T.groups.map(st=>st.map(g=>g.o)),pR:T.Z.map(Z=>Z.map(z=>softmax(z)[RIGHT]))};},
  demo:(id,std)=>demoGroup(id,std),pEq,chPoint,mem:memRows,initP:(p)=>softmax(initLogits(p)),drill:makeDrill,matched:(set,G)=>matched(set,G),
  consts:{SEED,ETA,BETA,EPS,MU,MAXS,NQ,SETS,DEMO,CH_GOAL}};

})();
