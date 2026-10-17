/* DPO lab: a live toy of Direct Preference Optimization (Rafailov et al., 2023, arXiv 2305.18290).
   One prompt, four answers, a softmax policy over their logits, the SFT model as the frozen reference,
   and exact gradient steps on the DPO loss, recomputed in the browser. Also registers the home-tile teaser.
   Names used everywhere (EN / ZH):
     log-ratio of an answer      log(π(y)/π_ref(y))                         log 比值
     log-ratio gap h             log-ratio of A − log-ratio of B              log 比值差 h
     reward margin               β × h = r̂_A − r̂_B                           奖励差
     loss                        −log σ(reward margin)
     push weight                 σ(−reward margin)                            推力权重 */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='dpo';

/* ---------- engine: pure functions ---------- */
const MAXS=100;                                   // longest training run the lab shows
const LET=['A','B','C','D'];
const REFS={b:[0.18,0.52,0.18,0.12], s:[0.05,0.85,0.06,0.04]};   // SFT (reference) probabilities, illustrative
const DETAIL=[1,0.2,0.5,2];
const klS=k=>k>0&&k<0.005?f(k,4):f(k,2);   // 0.0003, not 0.00, once the model has moved (R6.10)                       // "level of detail" per answer, used only with the shared weight
const DATA={one:[[0,1]], three:[[0,1],[0,2],[0,3]], mixed:[[0,1],[0,1],[1,0]]};   // pairs (chosen, rejected)
const BETAS=[0.1,0.5,1,2], ETAS=[0.05,0.1,0.2];
const DEFAULT={ds:'one',ref:'b',mode:'sep',beta:0.1,eta:0.1,steps:0};
const CFG=['ds','ref','mode','beta','eta'];
const sigm=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const softplus=x=>x>0?x+Math.log1p(Math.exp(-x)):Math.log1p(Math.exp(x));   // log(1 + e^x) = −log σ(−x)
function logSoftmax(z){let m=-Infinity; for(const v of z) if(v>m) m=v; let s=0; for(const v of z) s+=Math.exp(v-m); const c=m+Math.log(s); return z.map(v=>v-c);}

/* Train from the SFT model for MAXS steps and record every step.
   Loss: mean over pairs of −log σ(β·h), h = log π(w)/π_ref(w) − log π(l)/π_ref(l) (the log-ratio gap).
   Gradient of one pair: −β·σ(−β·h)·∂h/∂θ, and ∂h/∂θ = e_w − e_l because log π(y) = θ_y − logsumexp(θ).
   Update: θ ← θ − (η/β)·∇L (the step size is divided by β, see the Step size note). */
function train(ds,ref,mode,beta,eta){
  const pairs=DATA[ds], n=pairs.length, lref=REFS[ref].map(Math.log), lr_=eta/beta;
  let th=lref.slice(), w=0;
  const T={pi:[],lp:[],z:[],w:[],h:[],wt:[],loss:[],kl:[],g:[]};
  for(let t=0;t<=MAXS;t++){
    const z=mode==='shared'?lref.map((v,i)=>v+w*DETAIL[i]):th.slice();
    const lp=logSoftmax(z), pi=lp.map(Math.exp), lr=lp.map((v,i)=>v-lref[i]);
    const h=pairs.map(([a,c])=>lr[a]-lr[c]), wt=h.map(x=>sigm(-beta*x));
    T.pi.push(pi); T.lp.push(lp); T.z.push(z); T.w.push(w); T.h.push(h); T.wt.push(wt);
    T.loss.push(h.reduce((s,x)=>s+softplus(-beta*x),0)/n);
    T.kl.push(pi.reduce((s,p,i)=>s+p*lr[i],0));
    if(mode==='shared'){let g=0; pairs.forEach(([a,c],i)=>{g-=beta*wt[i]*(DETAIL[a]-DETAIL[c])/n;}); T.g.push([g]); w-=lr_*g;}
    else {const g=[0,0,0,0]; pairs.forEach(([a,c],i)=>{g[a]-=beta*wt[i]/n; g[c]+=beta*wt[i]/n;}); T.g.push(g); th=th.map((v,k)=>v-lr_*g[k]);}
  }
  return T;
}
const cache=new Map();
function traj(s){const k=CFG.map(c=>s[c]).join('|'); let T=cache.get(k); if(!T){T=train(s.ds,s.ref,s.mode,s.beta,s.eta); cache.set(k,T);} return T;}
function compute(s){
  const T=traj(s), t=Math.max(0,Math.min(MAXS,Math.round(s.steps))), ref=REFS[s.ref];
  const pi=T.pi[t], lp=T.lp[t], lr=lp.map((v,i)=>v-Math.log(ref[i])), rhat=lr.map(v=>s.beta*v);
  const hAB=lr[0]-lr[1]; let top=0; pi.forEach((p,i)=>{if(p>pi[top]) top=i;});
  const r={s,T,t,pi,lp,ref,lr,rhat,pairs:DATA[s.ds],h:T.h[t],wt:T.wt[t],loss:T.loss[t],kl:T.kl[t],g:T.g[t],z:T.z[t],z0:T.z[0],w:T.w[t],
    hAB,mAB:s.beta*hAB,wAB:sigm(-s.beta*hAB),top};
  r.c=chainOf(r);
  return r;
}
/* The display chain for formula lines: every number in a formula line is computed from the numbers shown before it, so a learner can redo it.
   π is shown as a percentage with 6 significant digits; log-ratios with 5 decimals are computed from those shown values;
   h = difference of shown log-ratios; reward margin = β × h (exact); loss, push weight and KL from those.
   With this many digits the chain agrees with the unrounded values on the metric cards to the cards' precision
   (tests/dpo_numbers.py counts the rare rounding-boundary exceptions). */
const dec6=v=>Math.max(0,Math.min(10,5-Math.floor(Math.log10(v))));   // decimals for 6 significant digits of a percentage
const r3=v=>+v.toFixed(3), r4=v=>+v.toFixed(4), r5=v=>+v.toFixed(5), r6=v=>+v.toFixed(6);
const mdp=b=>(b===1||b===2)?5:6;                                        // decimals of β × (a 5-decimal number), shown in full
function chainOf(r){
  const b=r.s.beta, pd=r.pi.map(p=>{const v=p*100, d=dec6(v); return {v:+v.toFixed(d),d};});
  const refd=r.ref.map(p=>+(p*100).toFixed(2));
  const l=pd.map((x,i)=>r5(Math.log(x.v/refd[i])));
  const pairs=r.pairs.map(([a,c])=>{const h=r5(l[a]-l[c]); return {a,c,h,m:r6(b*h)};});
  const hAB=r5(l[0]-l[1]), mAB=r6(b*hAB), rA=r6(b*l[0]), rB=r6(b*l[1]);
  const loss=pairs.reduce((s,p)=>s+softplus(-p.m),0)/pairs.length, pw=sigm(-mAB);
  const kl=pd.reduce((s,x,i)=>s+x.v/100*l[i],0);
  return {pd,refd,l,pairs,hAB,mAB,rA,rB,loss,pw,pw3:r3(pw),kl};
}
function peakOf(T,i,t){let k=0; for(let u=0;u<=t;u++) if(T.pi[u][i]>T.pi[k][i]) k=u; return {k,p:T.pi[k][i]};}
function passStep(s){const T=traj(s); for(let u=0;u<=MAXS;u++) if(T.pi[u][0]>T.pi[u][1]) return u; return -1;}   // first step where A beats B
/* challenge: fixed setup, only β changes */
const CH={ds:'one',ref:'b',mode:'sep',eta:0.1,steps:60}, CH_PA=0.5, CH_KL=0.8;
const chPoint=b=>{const r=compute({...CH,beta:b}); const okA=r.pi[0]>=CH_PA, okK=r.kl<=CH_KL; return {b,pA:r.pi[0],kl:r.kl,w:r.wAB,okA,okK,ok:okA&&okK};};
/* interview drill: seeded, so a question can be reproduced */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){
  const rnd=mulberry(seed), r1=v=>Math.round(v*10)/10;
  const beta=[0.1,0.1,0.5][Math.floor(rnd()*3)];
  const refW=r1(-40+rnd()*28), refL=r1(-40+rnd()*28);
  let dW=r1(-1.5+rnd()*3.5), dL=r1(-2.5+rnd()*3.5); if(Math.abs(dW-dL)<0.3) dW=r1(dW+0.8);
  const piW=r1(refW+dW), piL=r1(refL+dL);
  const lrW=r1(piW-refW), lrL=r1(piL-refL), m=r4(beta*(lrW-lrL)), wt=sigm(-m);
  return {seed,beta,piW,piL,refW,refL,lrW,lrL,rW:r4(beta*lrW),rL:r4(beta*lrL),m,loss:softplus(-m),wt,coef:r4(beta*r3(wt))};
}

/* ---------- formatting ---------- */
const f=(v,d)=>A.fmt.num(v,d);
const MINUS='−';
function sg(v,d){const s=f(Math.abs(v),d); if(+s===0) return f(0,d); return (v<0?MINUS:'+')+s;}
const sn=(v,d)=>v<0?MINUS+f(-v,d):f(v,d);
const fx=(v,d)=>{const s=Math.abs(v).toFixed(d); return (v<0&&+s!==0?MINUS:'')+s;};   // fixed decimals, trailing zeros kept
const pmx=(v,d)=>v<0?`(${fx(v,d)})`:fx(v,d);
function pc(p,d){const v=p*100; if(d==null) d=(v<1||v>99)?1:0;
  if(v>0&&v<0.05) return '<0.1%'; if(v<100&&v>99.95) return '>99.9%'; return f(v,d)+'%';}
const pcd=x=>x.v.toFixed(x.d)+'%';                  // a percentage from the display chain (6 significant digits)
const rpc=v=>f(v,v%1?2:0)+'%';                      // an SFT percentage (exact)
const stepsTxt=n=>tx(n===1?'1 step':`${n} steps`,`${n} 步`);
const bTxt=b=>String(b);
const LOG2=Math.log(2);
const zsp=s=>/[A-Za-z0-9)Ͱ-Ͽ]$/.test(s)?s+' ':s;   // Chinese: a space after a Latin or Greek word
/* "from X to Y" with a verb that matches the shown numbers */
/* when whole percents hide a change, one decimal shows it: "rises from 18.0% to 18.3%", never "stays at 18%" next to "still rising" (R6.10) */
const pc1=p=>(p*100).toFixed(1)+'%';   // keeps the trailing zero: 18.0%, not 18%
const pcPair=(p0,p)=>{const a=pc(p0), b=pc(p); if(a!==b) return [a,b,false]; const a1=pc1(p0), b1=pc1(p); return a1===b1?[a,b,true]:[a1,b1,false];};
function chg(p0,p){const [a,b,same]=pcPair(p0,p); if(same) return tx(`stays at ${b}`,`保持 ${b}`);
  return p>p0?tx(`rises from ${a} to ${b}`,`从 ${a} 升到 ${b}`):tx(`falls from ${a} to ${b}`,`从 ${a} 降到 ${b}`);}
const logitTxt=(d,dp=2)=>Math.abs(d)<5e-4?tx('logit unchanged','logit 没变'):(d>0?tx(`logit up ${f(d,dp)}`,`logit 升了 ${f(d,dp)}`):tx(`logit down ${f(-d,dp)}`,`logit 降了 ${f(-d,dp)}`));
/* explanations give the trend in words and at most one key number (R6.20); the exact values stay on the bars, the metric cards, the
   curves and Show the math. How far an answer's probability moved, by percentage points (under 2% and a fifth of where it started:
   "almost to 0") */
function moveW(p0,p){const d=p-p0;
  if(Math.abs(d)<0.005) return tx('barely changes','几乎不变');
  if(d<0) return p<0.02&&p<p0*0.2?tx('falls almost to 0','几乎降到 0'):(d<=-0.25?tx('falls sharply','大幅下降'):(d<=-0.05?tx('falls','明显下降'):tx('falls a little','略有下降')));
  return d>=0.25?tx('rises sharply','大幅上升'):(d>=0.05?tx('rises','明显上升'):tx('rises a little','略有上升'));}
/* the same for "gives B … often" on the home tile (English only: Chinese uses moveW) */
const lessW=(p0,p)=>{const d=p-p0; return d>-0.005?tx('about as often as before','几乎不变'):(d>-0.05?tx('a little less often','略有下降'):(d>-0.25?tx('less often','明显下降'):tx('far less often','大幅下降')));};
/* how much of the first step's push is left, from the push weight as shown (3 decimals) against the 0.5 every pair starts at */
const pushW=w=>{const q=+w.toFixed(3)/0.5; return q>=0.5?tx('still has most of its strength','还剩大半'):(q>=0.2?tx('has faded a lot','已经弱了很多'):tx('has almost faded','几乎消失了'));};
/* shared weight: every logit moves by w × its detail level; w as Show the math shows it (3 decimals), so 0.397 and 0.794 redo from there (R6.6) */
const wRises=r=>{const w3=+(+r.w).toFixed(3); return DETAIL.map(d=>w3*d);};

/* ---------- module state (survives re-mounts and language switches) ---------- */
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let view='prob', ghost=null, saved=null, chSubmitted=false, chReveal=false;
let triedB=new Set([0.1]), raf=0, playing=false, playTarget=MAXS, playRate=0, lastTs=0, acc=0, drillSeed=7, drillOpen=false, annT=null;
let speed='med', tzSteps=20, wantObs=false;   // wantObs: bring a part's result into view when it first appears after a click (not while dragging)
/* playback speed in training steps per second; playback advances by elapsed time, so the speed is the same on any display */
const SPEEDS={slow:2,med:5,fast:20};

/* ---------- copy ---------- */
const ANS=[
 {s:{en:'Explains the idea and its cost',zh:'讲清原理和代价'},full:{en:'“Say the model keeps every earlier token’s keys and values so it never recomputes them. The catch: that memory grows with every token of context.”',zh:'“告诉他，模型把之前每个 token 的 key 和 value 都存下来，就不用重算。代价是这块显存随上下文里的每个 token 增长。”'}},
 {s:{en:'Points to the docs',zh:'让人去看文档'},full:{en:'“It’s just a cache. Point them to the vLLM docs.”',zh:'“就是个缓存。让他去看 vLLM 的文档。”'}},
 {s:{en:'Analogy only',zh:'只打比方'},full:{en:'“Compare it to memoization, where the model remembers work it already did.”',zh:'“把它比作 memoization，也就是模型记住已经算过的东西。”'}},
 {s:{en:'Formula first',zh:'先上公式'},full:{en:'“Start from 2 × layers × KV heads × head_dim × bytes, then cover GQA, paging and FP8.”',zh:'“从 2 × 层数 × KV head 数 × head_dim × 字节数讲起，再讲 GQA、分页和 FP8。”'}}
];
const PROMPT={en:'How do I explain the KV cache to a new teammate?',zh:'怎么向新同事解释 KV cache？'};
const DSNAME={one:{en:'1 pair',zh:'1 对'},three:{en:'3 pairs',zh:'3 对'},mixed:{en:'Mixed labels',zh:'标注不一致'}};
const DSDESC={one:{en:'A ≻ B',zh:'A ≻ B'},three:{en:'A ≻ B, A ≻ C and A ≻ D',zh:'A ≻ B、A ≻ C、A ≻ D'},mixed:{en:'A ≻ B twice and B ≻ A once',zh:'A ≻ B 两次，B ≻ A 一次'}};
const refName=k=>k==='b'?tx('Prefers B','偏向 B'):tx('Strongly prefers B','强烈偏向 B');
const refDesc=k=>tx(`A ${pc(REFS[k][0])} and B ${pc(REFS[k][1])}`,`A ${pc(REFS[k][0])}、B ${pc(REFS[k][1])}`);
const MODENAME={sep:{en:'Separate logits',zh:'各自的 logit'},shared:{en:'Shared weight',zh:'共享权重'}};
const VIEWS=[['prob',{en:'Probabilities',zh:'概率'}],['kl',{en:'KL',zh:'KL'}],['loss',{en:'Loss',zh:'loss'}],['wt',{en:'Push weight',zh:'推力权重'}]];
/* scenarios: situations with plain names and one-line descriptions (R2.1). A scenario shows as selected only while every
   setting matches it, the step count included (R1.8), so the result under the card always belongs to the pressed scenario. */
const PRESETS=[
 {t:{en:'Initial setup: one preference pair',zh:'初始设置：一对偏好'},d:{en:'Annotators prefer A; the SFT model prefers B',zh:'标注员更喜欢 A，SFT 模型偏向 B'},p:{ds:'one',ref:'b',mode:'sep',beta:0.1,eta:0.1},steps:40},
 {t:{en:'The best answer against the other three',zh:'A 胜过其余三个回答'},d:{en:'Three pairs: A ≻ B, A ≻ C and A ≻ D',zh:'三对数据：A ≻ B、A ≻ C、A ≻ D'},p:{ds:'three',ref:'b',mode:'sep',beta:0.1,eta:0.1},steps:40},
 {t:{en:'Annotators disagree',zh:'标注员意见不一'},d:{en:'Three labels: A ≻ B twice, B ≻ A once',zh:'三次标注：两次 A ≻ B，一次 B ≻ A'},p:{ds:'mixed',ref:'b',mode:'sep',beta:1,eta:0.1},steps:100},
 {t:{en:'A stubborn SFT model',zh:'SFT 模型很固执'},d:{en:`Before training it gives A only ${pc(REFS.s[0])} of the time and B ${pc(REFS.s[1])}`,zh:`训练前只有 ${pc(REFS.s[0])} 的时候给出 A，${pc(REFS.s[1])} 的时候给出 B`},p:{ds:'one',ref:'s',mode:'sep',beta:0.1,eta:0.1},steps:40},
 {t:{en:'Closer to a real LLM: shared weights',zh:'更像真实 LLM：回答共享权重'},d:{en:'In a real LLM every answer comes out of the same weights',zh:'真实 LLM 里，所有回答出自同一套权重'},p:{ds:'one',ref:'b',mode:'shared',beta:0.1,eta:0.1},steps:60},
 {t:{en:'Worried about drift: large β',zh:'怕模型跑偏：用大 β'},d:{en:'The same pair with β = 2',zh:'同一对数据，β = 2'},p:{ds:'one',ref:'b',mode:'sep',beta:2,eta:0.1},steps:100}
];
const presetNow=()=>PRESETS.findIndex(p=>CFG.every(k=>p.p[k]===S[k])&&p.steps===S.steps);
function roleOf(i,pairs){let w=0,l=0; pairs.forEach(([a,c])=>{if(a===i) w++; if(c===i) l++;}); return {w,l};}
function roleTxt(i,pairs){const {w,l}=roleOf(i,pairs);
  if(!w&&!l) return tx('not labeled','未标注');
  const tm=n=>n===1?'once':(n===2?'twice':`${n} times`);
  if(w&&!l) return w>1?tx(`✓ chosen ${tm(w)}`,`✓ 选中 ${w} 次`):tx('✓ chosen','✓ 选中');
  if(l&&!w) return tx('✗ rejected','✗ 落选');
  return tx(`✓ chosen ${tm(w)}, ✗ rejected ${tm(l)}`,`✓ 选中 ${w} 次，✗ 落选 ${l} 次`);}
function roleCls(i,pairs){const {w,l}=roleOf(i,pairs); return !w&&!l?'o':(w>=l?'w':'l');}
const pairTxt=([a,c])=>`${LET[a]} ≻ ${LET[c]}`;
/* terms defined on hover (R2.8). The reader's likely questions, answered where the claim is made (review 7):
   is D trained at all? is softmax a second loss, is SFT running at the same time? why does D move more than A? */
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const LOGIT_TIP=()=>tx('A logit is the model’s score for one option; softmax turns the scores into probabilities. A real model gives every vocabulary token a logit (see the lab “Architecture overview”); in this lab each whole answer counts as one option with one logit.',
  'logit 是模型给一个选项的得分，softmax 把它们变成概率。真实模型给词表里每个 token 一个 logit（见“结构总览”这个 Lab）；在这个 Lab 里，每个完整的回答算一个选项，只有一个 logit。');
const uniqPairs=pairs=>{const seen=new Set(), out=[]; pairs.forEach((p,i)=>{const k=p.join(); if(!seen.has(k)){seen.add(k); out.push({p,i,n:pairs.filter(q=>q.join()===k).length});}}); return out;};
/* β, said the same way everywhere (R6.10, R6.11): what it is in general, and what it does in this lab. The second part holds
   here because each update is scaled by η/β; with a fixed learning rate a larger β would make the first steps larger. */
const BETA_EN='β is the weight of the KL penalty in the RLHF objective that DPO is derived from; the DPO paper and TRL default to 0.1. In this lab, the larger β, the less the model changes in the same number of steps, so the closer it stays to the SFT model.';
const BETA_ZH='β 是 RLHF 目标里 KL 惩罚的系数，DPO 就是从这个目标推出来的；DPO 论文和 TRL 默认 0.1。在这个 Lab 里，β 越大，同样的训练步数下模型改动得越少，离 SFT 模型越近。';
const BETA_TIP=()=>tx(BETA_EN,BETA_ZH);
const betaSpan=b=>`<span class="dpo-bt" tabindex="0" data-tipx="${esc(BETA_TIP())}">β = ${bTxt(b)}</span>`;
const cite=(url,en,zh)=>`<a href="${url}" target="_blank" rel="noopener">${tx(en,zh)}</a>`;
const LLAMA=()=>cite('https://arxiv.org/abs/2407.21783','Llama Team, 2024','Llama 团队，2024');
const RAZIN=()=>cite('https://arxiv.org/abs/2410.08847','Razin et al., ICLR 2025','Razin 等，ICLR 2025');
const PAL=()=>cite('https://arxiv.org/abs/2402.13228','Pal et al., 2024','Pal 等，2024');
const AZAR=()=>cite('https://arxiv.org/abs/2310.12036','Azar et al., 2023','Azar 等，2023');
const RAF=()=>cite('https://arxiv.org/abs/2305.18290','Rafailov et al., 2023','Rafailov 等，2023');
const KB=()=>cite('https://arxiv.org/abs/1412.6980','Kingma and Ba, 2015','Kingma 和 Ba，2015');
/* typeset formulas (H1, the owner's review of 2026-10-09): M inline in a sentence; FX a formula on its own line, its name in words on
   the left; EQ an aligned display for Show the math (H2): rows of cells [left side, "=" or "+", terms…]. The operator column lines up,
   and in rows with the same cells each number sits under the term it replaces; a row with fewer cells lets its last cell run on. */
/* the kit sets TeX's thin space after log and exp before a name (log σ, log Z, log π) since 2026-10-09, so no workaround here */
const TM=t=>A.ui.math(t);
const M=t=>`<span class="dpo-m">${TM(t)}</span>`;
const FX=(name,t)=>`<span class="dpo-fx">${name?`<span class="dpo-fxn">${name}</span>`:''}${M(t)}</span>`;   // a block span: valid inside a <p>
/* a formula too long for a phone (the owner's review of 2026-10-09: no Show the math row scrolls sideways): its left side, then pieces
   that wrap before an operator, each new line under the first piece; {} keeps the spacing of a binary operator at a piece's start */
const FXW=(lhs,ps)=>`<span class="dpo-fx dpo-fxw">${M(lhs)}<span class="dpo-fxr">${ps.map((p,i)=>M(i?`{} ${p}`:p)).join('')}</span></span>`;
/* a row marked cont goes on the next line under the row before it and keeps its data-r: one equation broken at “=” (the owner's review of
   2026-10-09: no Show the math row scrolls sideways) */
function EQ(rows){const n=Math.max(...rows.map(r=>r.length)); let q=-1; const rr=rows.map(r=>r.cont?q:++q);
  return `<span class="dpo-eq" style="grid-template-columns:repeat(${n},auto)">`+rows.map((r,j)=>r.map((c,k)=>{
    const run=k===r.length-1&&r.length<n, cls=k===0?'dpo-el':(k===1?'dpo-eo':'dpo-ec');
    return `<span class="${cls}" data-r="${rr[j]}"${run?` style="grid-column:${k+1} / -1"`:''}>${c==null||c===''?'':(typeof c==='object'?c.h:TM(c))}</span>`;}).join('')).join('')+`</span>`;}
/* numbers inside TeX: a negative number is one braced group, so it reads as −0.93 and not as a minus between two terms */
const nT=(v,d)=>{const s=fx(v,d); return s.startsWith(MINUS)?`{-${s.slice(1)}}`:s;};
const pT=(v,d)=>v<0&&+Math.abs(v).toFixed(d)!==0?`(${nT(v,d)})`:nT(v,d);                // in parentheses after an operator
const sT=(v,d)=>{const s=sg(v,d); return s.startsWith(MINUS)?`{-${s.slice(1)}}`:(s.startsWith('+')?`{+${s.slice(1)}}`:s);};   // with its sign
const TX=(en,zh)=>`\\text{${tx(en,zh)}}`;
const LR=()=>TX('log-ratio','log 比值');

/* hovers that hold a formula take a string key (data-tip), because the kit escapes data-tipx: the formula is typeset (H1) */
const DEFK=(text,key)=>`<span class="def" tabindex="0" data-tip="${key}">${text}</span>`;
const TIPS=(()=>{const d=DETAIL.map(v=>f(v,1)), F=t=>`<span class="dpo-fx">${TM(t)}</span>`;
  const pY='\\pi(y) = \\frac{e^{\\text{logit}(y)}}{\\sum e^{\\text{logit}}}';
  return {
  en:{dpo_tSoftmax:`Softmax turns the four logits into four probabilities that always add up to 100%:${F(pY)}The sum runs over the four answers. Softmax is only how the model turns logits into probabilities, not a second loss: training here uses the DPO loss alone, with no SFT term.`,
      dpo_tSep:`Separate logits: each answer has its own logit, and each DPO step changes only the logits of the two answers in the pair, A and B. C’s and D’s logits stay at their values before training. Their probabilities still change, because C’s probability is${F('\\pi(C) = \\frac{e^{\\text{logit}(C)}}{\\sum e^{\\text{logit}}}')}and the sum below the line moves when A’s and B’s logits move.`,
      dpo_tShared:`Shared weight: all four logits come from one parameter w.${F('\\text{logit}(y) = \\text{logit before training}(y) + w \\times \\text{detail level}(y)')}The detail levels are A ${d[0]}, B ${d[1]}, C ${d[2]} and D ${d[3]}. A has more detail than B, so the only way to widen the reward margin between A and B is to raise w. Each 1 added to w adds ${d[0]} to A’s logit and ${d[3]} to D’s. D never enters the loss, yet it gains the most and takes probability from A.`},
  zh:{dpo_tSoftmax:`softmax 把四个 logit 换算成四个概率，四个概率加起来总是 100%。${F(pY)}分母对四个回答求和。softmax 只是模型把 logit 换成概率的一步，不是另一个 loss。这里的训练只用 DPO 的 loss，没有同时做 SFT。`,
      dpo_tSep:`各自的 logit：每个回答有自己的 logit，DPO 每一步只改偏好对里 A 和 B 这两个回答的 logit，C 和 D 的 logit 一直是训练前的值。它们的概率仍然会变，因为 C 的概率是这样算的：${F('\\pi(C) = \\frac{e^{\\text{logit}(C)}}{\\sum e^{\\text{logit}}}')}A 和 B 的 logit 一动，分母这个和就跟着变。`,
      dpo_tShared:`共享权重：四个 logit 都由同一个参数 w 决定。${F('\\text{logit}(y) = \\text{训练前的 logit}(y) + w \\times \\text{细节程度}(y)')}细节程度是 A ${d[0]}、B ${d[1]}、C ${d[2]}、D ${d[3]}。A 的细节比 B 多，要拉大 A 和 B 的奖励差，DPO 只能调大 w。w 每加 1，A 的 logit 加 ${d[0]}，D 的加 ${d[3]}。D 从没进过 loss，却涨得最多，还抢走了 A 的概率。`}};})();
const STR={
 en:{
  dpo_tipTop:'The answer the model gives most often now, with its probability. Green means it is A, the answer annotators preferred; red means another answer.',
  dpo_tipGoal:'This label names the condition that is not met yet. When, after 60 steps, the model gives A at least half the time and its KL from the SFT model is at most 0.8 nats, it turns green and reads “Both conditions are met”.',
  dpo_lgW:'A solid bar marks a chosen answer (✓), the one annotators preferred. Its length is how often the model gives the answer now.',
  dpo_lgL:'A hatched bar marks a rejected answer (✗). Its length is how often the model gives the answer now.',
  dpo_lgO:'A gray bar marks an answer that is in no preference pair.',
  dpo_lgRef:`The dashed outline is how often the SFT model gave the answer before training. This SFT model is also the reference ${M('\\pi_{\\text{ref}}')}: it never changes, and DPO measures every change against it.`,
  dpo_mA:'π(A) is how often the model gives answer A, the chosen answer.',
  dpo_mB:'π(B) is how often the model gives answer B, the rejected answer of the pair A ≻ B.',
  dpo_mMargin:`The reward margin is A’s implicit reward minus B’s. An answer’s implicit reward says how much more often the model gives it now than the SFT model did (as a natural log), times β:${FX('','\\hat{r} = \\beta \\log\\frac{\\pi}{\\pi_{\\text{ref}}}')}The larger the reward margin, the smaller the pair’s loss.`,
  dpo_mLoss:`The DPO loss is averaged over the pairs. One pair’s loss is${FX('','-\\log \\sigma(\\text{reward margin})')}where ${M('\\sigma(x) = \\frac{1}{1 + e^{-x}}')}. At step 0 every reward margin is 0, so the loss is log 2 ≈ 0.693.`,
  dpo_mKL:`KL measures how far the model has moved from the SFT model; it is 0 when the two are the same. Here it sums over the four answers, in nats (natural-log units):${FX('','\\text{KL}(\\pi \\,‖\\, \\pi_{\\text{ref}}) = \\sum_y \\pi(y) \\log\\frac{\\pi(y)}{\\pi_{\\text{ref}}(y)}')}`,
  dpo_vProb:'Each line is how often the model gives one answer, from before training (step 0) to the current step.',
  dpo_vKL:'The line is the KL from the SFT model at each step, in nats.',
  dpo_vLoss:`The line is the DPO loss ${M('-\\log \\sigma(\\text{reward margin})')} at each step, averaged over the pairs.`,
  dpo_vWt:`Each line is the push weight ${M('\\sigma({-\\text{reward margin}})')} of one preference pair. A step moves the logits in proportion to it.`,
  dpo_tipPlay:'Play trains one step at a time, up to step 100.',
  dpo_tipStep:'Take one more gradient step on the DPO loss.',
  dpo_tipStep10:'Take ten more gradient steps.',
  dpo_tipReset:'Go back to the SFT model before training (step 0).',
  dpo_tipGhost:'The dotted line is the same curve under your previous setting.',
  dpo_tipNow:'The vertical dotted line marks the current training step.',
  dpo_tipRefLine:'The thin dashed line is the value at step 0, where every pair begins.'
 },
 zh:{
  dpo_tipTop:'模型现在最常给出的回答，括号里是它的概率。绿色表示是 A（标注员更喜欢的回答），红色表示是别的回答。',
  dpo_tipGoal:'这里写的是还没满足的条件。训练 60 步后，模型至少一半的时候给出 A，而且离 SFT 模型的 KL 不超过 0.8 nats，就会变成绿色的“两个条件都满足”。',
  dpo_lgW:'实心条表示选中的回答（✓），也就是标注员更喜欢的那个。条的长度是模型现在给出这个回答的概率。',
  dpo_lgL:'斜线条表示落选的回答（✗）。条的长度是模型现在给出这个回答的概率。',
  dpo_lgO:'灰色条表示没有出现在任何偏好对里的回答。',
  dpo_lgRef:`虚线框是训练前的 SFT 模型给这个回答的概率。这个 SFT 模型就是参考模型 ${M('\\pi_{\\text{ref}}')}，训练中不变，DPO 衡量的每个变化都是相对它的。`,
  dpo_mA:'π(A) 是模型给出回答 A 的概率，A 是选中的回答。',
  dpo_mB:'π(B) 是模型给出回答 B 的概率，B 是 A ≻ B 这一对里落选的回答。',
  dpo_mMargin:`奖励差是 A 的隐式奖励减去 B 的。一个回答的隐式奖励，看模型现在给出它的概率比 SFT 模型高了多少（取自然对数），再乘 β：${FX('','\\hat{r} = \\beta \\log\\frac{\\pi}{\\pi_{\\text{ref}}}')}奖励差越大，这一对的 loss 越小。`,
  dpo_mLoss:`DPO 的 loss 对所有偏好对取平均。一对数据的 loss 是：${FX('','-\\log \\sigma(\\text{奖励差})')}其中 ${M('\\sigma(x) = \\frac{1}{1 + e^{-x}}')}。第 0 步每个奖励差都是 0，所以 loss 是 log 2 ≈ 0.693。`,
  dpo_mKL:`KL 衡量模型离 SFT 模型有多远，两者完全一样时是 0。这里对四个回答求和，单位 nats（按自然对数算）。${FX('','\\text{KL}(\\pi \\,‖\\, \\pi_{\\text{ref}}) = \\sum_y \\pi(y) \\log\\frac{\\pi(y)}{\\pi_{\\text{ref}}(y)}')}`,
  dpo_vProb:'每条线是模型给出某个回答的概率，从训练前（第 0 步）画到当前步。',
  dpo_vKL:'这条线是每一步时模型离 SFT 模型的 KL，单位 nats。',
  dpo_vLoss:`这条线是每一步的 DPO loss ${M('-\\log \\sigma(\\text{奖励差})')}，对所有偏好对取平均。`,
  dpo_vWt:`每条线是一对偏好数据的推力权重 ${M('\\sigma({-\\text{奖励差}})')}。一步对 logit 的改动和它成正比。`,
  dpo_tipPlay:'播放：一步一步训练到第 100 步。',
  dpo_tipStep:'在 DPO loss 上再做一步梯度下降。',
  dpo_tipStep10:'再做十步梯度下降。',
  dpo_tipReset:'回到训练前的 SFT 模型（第 0 步）。',
  dpo_tipGhost:'点线是上一个设置下的同一条曲线。',
  dpo_tipNow:'竖直点线标出当前的训练步。',
  dpo_tipRefLine:'细虚线是第 0 步的值，每一对都从这里开始。'
 }
};
Object.assign(STR.en,TIPS.en); Object.assign(STR.zh,TIPS.zh);
A.addStrings(STR);

/* ---------- guided parts ---------- */
const PRED=[{v:'same'},{v:'down'},{v:'drift'}];
const predTxt=v=>{const r=REFS.b;
  if(v==='same') return tx(`They stay at ${pc(r[2])} and ${pc(r[3])}: nobody labeled them`,`保持 ${pc(r[2])} 和 ${pc(r[3])} 不变：没人标注它们`);
  if(v==='down') return tx('They drop: DPO pushes every answer except A down','下降：DPO 会把 A 以外的回答都往下压');
  return tx('They change: the four probabilities add up to 100%, so they move when A and B do','会变：四个概率加起来总是 100%，A 和 B 一变，它们也跟着变');};
const J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
function playLabel(){return playing?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Play','▶ 播放');}
const speedCtl=()=>`<div class="dpo-speed" data-tipx="${esc(tx(`Slow trains ${SPEEDS.slow} steps per second, Medium ${SPEEDS.med} and Fast ${SPEEDS.fast}.`,`慢速每秒训练 ${SPEEDS.slow} 步，中速 ${SPEEDS.med} 步，快速 ${SPEEDS.fast} 步。`))}"><span class="dpo-speed-l">${tx('Speed','速度')}</span>`+
  `<div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="speed">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${speed===v}">${x}</button>`).join('')}</div></div>`;
const stepBtns=()=>`<div class="dpo-tbtns"><button type="button" class="btn ghost sm" data-act="play" data-play data-tip="dpo_tipPlay">${playLabel()}</button>${speedCtl()}<button type="button" class="btn ghost sm" data-act="step1" data-tip="dpo_tipStep">${tx('+1 step','+1 步')}</button><button type="button" class="btn ghost sm" data-act="step10" data-tip="dpo_tipStep10">${tx('+10 steps','+10 步')}</button></div>`;
const inSteps=()=>A.ui.range({id:'dpo-ic-steps',key:'steps',label:tx('Training steps','训练步数'),min:0,max:MAXS,value:S.steps,out:stepsTxt(S.steps),inline:true});
const inBeta=()=>A.ui.seg({id:'dpo-ic-beta',key:'beta',label:tx('β (the weight of the KL penalty in the RLHF objective)','β（RLHF 目标里 KL 惩罚的系数）'),options:BETAS.map(b=>[String(b),bTxt(b)]),value:String(S.beta),inline:true});
const inMode=()=>A.ui.seg({id:'dpo-ic-mode',key:'mode',label:tx('How answers get their logits','回答的 logit 怎么来'),options:[['sep',L(MODENAME.sep)],['shared',L(MODENAME.shared)]],value:S.mode,inline:true});
const pw3=w=>w.toFixed(3);                                  // push weight as shown everywhere: 3 decimals
/* "x% as far as the first step" = shown weight / 0.500. Rounding the weight to 3 decimals moves 200·w by at most 0.1,
   to the nearest multiple of 0.2, so this percentage is also the unrounded one. */
const relOf=w=>Math.round(+w.toFixed(3)/0.5*100);
/* before the prediction in part 2, the guided tour does not train (R3.8: no result before the answer) */
const preLock=()=>mode==='guided'&&!pred&&step<=1;

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'Annotators preferred A to B, but the model gives A only 18%.',zh:'标注员更喜欢 A，可模型给 A 的概率只有 18%。'},enter:()=>({...DEFAULT}),view:'prob',
  body:()=>{const r=REFS.b;
   return tx(
    `<p>You have a chat model fresh out of supervised fine-tuning (SFT). For the prompt “${esc(PROMPT.en)}” it can give the four answers ${J('bars','in the chart')}. Annotators compared A with B and preferred A, written A ≻ B, yet the model gives B ${pc(r[1])} of the time and A only ${pc(r[0])}.</p><p>Your job: pull the model toward A with DPO. DPO trains directly on preference pairs: no separate reward model, and the model does not generate answers during training.</p><p>DPO keeps a frozen copy of the SFT model as the reference, ${M('\\pi_{\\text{ref}}')}, and each step makes the model give A more often and B less often than the reference does. β is the weight of the KL penalty in the RLHF objective; the DPO paper and TRL default to 0.1. In this lab, the larger β, the less the model changes in the same number of steps, so the closer it stays to the reference.</p>`,
    `<p>你有一个刚做完监督微调（SFT）的聊天模型。对于提问“${esc(PROMPT.zh)}”，它可能给出${J('bars','图里')}这四个回答。标注员比较了 A 和 B，更喜欢 A，记作 A ≻ B；可模型给出 B 的概率是 ${pc(r[1])}，给出 A 的只有 ${pc(r[0])}。</p><p>你的任务：用 DPO 把模型拉向 A。DPO 直接在偏好对上训练，不用另训一个奖励模型，训练时也不用让模型生成回答。</p><p>DPO 留一份冻结的 SFT 模型作参考模型，记作 ${M('\\pi_{\\text{ref}}')}。每一步都让模型比参考模型更常给出 A、更少给出 B。β 是 RLHF 目标里 KL 惩罚的系数，DPO 论文和 TRL 默认 0.1。在这个 Lab 里，β 越大，同样的步数下模型改动得越少，离参考模型越近。</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: what happens to C and D, which nobody labeled?',zh:'先猜：没人标注过的 C 和 D 会怎样？'},enter:()=>({...DEFAULT}),view:'prob',gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(predTxt(o.v))}</button>`).join('')}</div>`;
   return tx(`<p>DPO only sees the pair A ≻ B. After some training steps, what happens to C and D, the two answers nobody labeled? Pick one.</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
             `<p>DPO 只看到 A ≻ B 这一对。训练几步之后，没人标注过的 C 和 D 会怎样？选一个。</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'steps',title:{en:'Take DPO steps',zh:'训练几步 DPO'},lead:{en:'Train at least 10 steps and watch the four probabilities.',zh:'训练至少 10 步，看四个回答的概率怎么变。'},enter:()=>({...DEFAULT}),view:'prob',target:'steps',expect:()=>S.steps>=10,showMe:{steps:20},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>In this lab the model scores each answer with a ${DEF('logit',LOGIT_TIP())}, and a ${DEFK('softmax','dpo_tSoftmax')} turns the four scores into probabilities that sum to 100%. Each DPO step pushes A’s logit up and B’s down, more gently the more A has gained on B since training began. That lowers the DPO loss (its formula is in part 4).</p>`,
              `<p>在这个 Lab 里，模型给每个回答打一个分，叫 ${DEF('logit',LOGIT_TIP())}。${DEFK('softmax','dpo_tSoftmax')} 把四个 logit 换算成四个概率，加起来是 100%。</p><p>DPO 每训练一步，都把选中的回答 A 的 logit 往上推，把落选的回答 B 的 logit 往下压。A 和 B 的差距比训练前拉开得越多，推得越轻。这一推一压，就是在减小 DPO 的 loss，loss 的公式在第 4 部分。</p>`)+inSteps()+stepBtns(),
  obs:r=>obsSteps(r)},
 {id:'loss',title:{en:'Why no reward model?',zh:'为什么不需要奖励模型？'},lead:{en:'The model scores the answers itself: the implicit reward.',zh:'打分由模型自己给：隐式奖励。'},enter:()=>({...DEFAULT,steps:40}),view:'prob',
  body:()=>tx(
   `<p>RLHF first trains a reward model on the pairs, then uses PPO to make the model generate answers and raise their reward. The DPO paper shows that both steps can be replaced by one loss on the pairs.</p><p>The model scores answers itself. An answer’s implicit reward is how much more often the model gives it now than the reference ${M('\\pi_{\\text{ref}}')} (the SFT model before training), as a natural log, times β. A pair’s loss depends on the difference between A’s and B’s implicit rewards:</p>${FX('Implicit reward','\\hat{r} = \\beta \\log\\frac{\\pi}{\\pi_{\\text{ref}}}')}${FX('DPO loss','L = -\\log \\sigma(\\hat{r}_A - \\hat{r}_B)')}${FX('where','\\sigma(x) = \\frac{1}{1 + e^{-x}}')}<p>log is the natural log, and σ squeezes any number into the range from 0 to 1. ${M('\\hat{r}_A - \\hat{r}_B')} is the reward margin: the larger it is, the closer σ gets to 1 and the smaller the loss. The reward margin is how much more A has gained than B relative to the SFT model, so minimizing the loss makes A’s probability gain more than B’s. The full derivation is under ${J('math','Show the math')}.</p>`,
   `<p>RLHF 先用偏好对训练一个奖励模型，再用 PPO 让模型边生成回答边提高奖励模型的打分。DPO 论文证明，这两步可以换成一个直接作用在偏好对上的 loss。</p><p>打分由模型自己给。一个回答的隐式奖励，看模型现在给出它的概率比参考模型 ${M('\\pi_{\\text{ref}}')}（训练前的 SFT 模型）高了多少，取自然对数后再乘 β。一对数据的 loss，由 A 和 B 的隐式奖励之差决定：</p>${FX('隐式奖励','\\hat{r} = \\beta \\log\\frac{\\pi}{\\pi_{\\text{ref}}}')}${FX('DPO 的 loss','L = -\\log \\sigma(\\hat{r}_A - \\hat{r}_B)')}${FX('其中','\\sigma(x) = \\frac{1}{1 + e^{-x}}')}<p>log 是自然对数，σ 把任意一个数压到 0 和 1 之间。${M('\\hat{r}_A - \\hat{r}_B')} 叫奖励差：奖励差越大，σ 越接近 1，loss 越小。奖励差衡量的是 A 相对 SFT 模型比 B 多涨了多少，所以最小化 loss，就是让 A 的概率相对 SFT 模型涨得比 B 多。完整推导见${J('math','“展开公式”')}。</p>`)},
 {id:'fade',title:{en:'The push fades',zh:'推力会衰减'},lead:{en:'Drag to at least 60 steps and watch the push weight shrink.',zh:'把训练步数拖到至少 60，看推力权重怎么变小。'},enter:()=>({...DEFAULT}),view:'wt',target:'steps',expect:()=>S.steps>=60,showMe:{steps:80},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Each step pushes A’s logit up and B’s down. How far is scaled by the push weight:</p>${FX('Push weight','\\sigma({-\\text{reward margin}})')}<p>Under the implicit reward, the push weight is the chance that B still beats A. Before training the reward margin is 0, so the push weight is ${M('\\sigma(0) = 0.5')}. The further A pulls ahead, the smaller it gets.</p><p>Drag the training steps to at least 60 and watch ${J('curve','the push-weight curve')}.</p>`,
              `<p>每一步都把 A 的 logit 往上推、把 B 的往下推。推多少，要乘以推力权重：</p>${FX('推力权重','\\sigma({-\\text{奖励差}})')}<p>按隐式奖励算，推力权重就是 B 仍然胜过 A 的概率。训练前奖励差是 0，所以推力权重是 ${M('\\sigma(0) = 0.5')}。A 领先得越多，推力权重越小。</p><p>把训练步数拖到至少 60，看${J('curve','推力权重曲线')}。</p>`)+inSteps()+`<p class="dpo-live mono" id="dpo-pw-live" aria-live="polite"></p>`,
  obs:r=>obsFade(r)},
 {id:'beta',title:{en:'Change β',zh:'改一改 β'},lead:{en:'Try a few values of β and watch the KL curve.',zh:'换几个 β，看 KL 曲线。'},enter:()=>({...DEFAULT,steps:40}),view:'kl',target:'beta',expect:()=>triedB.size>1,showMe:{beta:1},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Try a few values of β. Each one trains from the same SFT model for the same number of steps.</p><p>Watch ${J('curve','the KL curve')}: KL measures how far the model has moved from the SFT model, in nats (natural-log units; 0 means not at all). In this lab the first step moves the logits by the same amount for every β.</p>`,
              `<p>换几个 β 试试。每个 β 都从同一个 SFT 模型开始，训练同样的步数。</p><p>看 ${J('curve','KL 曲线')}：KL 衡量模型离 SFT 模型有多远，单位 nats（按自然对数算，0 表示完全没动）。在这个 Lab 里，不管 β 取多少，第一步对 logit 的改动都一样大。</p>`)+inBeta(),
  obs:r=>obsBeta(r)},
 {id:'shared',title:{en:'With shared weights, does the chosen answer always rise?',zh:'共享权重时，选中的回答一定会上升吗？'},lead:{en:'In a real model the answers are linked: pushing A moves the others too.',zh:'真实模型里，回答之间相互关联：推 A，别的回答也跟着动。'},enter:()=>({...DEFAULT,steps:60}),view:'prob',target:'mode',expect:()=>S.mode==='shared',showMe:{mode:'shared'},label:{en:'Observed',zh:'观察到'},
  body:()=>{const d=DETAIL.map(v=>f(v,1));
   return tx(`<p>So far this lab has assumed that each answer’s logit is independent: pushing A up moves only A, and pushing B down moves only B. In a real LLM, though, every answer is computed by the same weights, so the answers are linked. Raising one answer also moves the answers that share words or structure with it.</p><p>This lab mimics that link with one shared weight w. Each answer has a detail level, how much technical detail it gives (illustrative: A ${d[0]}, B ${d[1]}, C ${d[2]}, D ${d[3]}). Its logit is</p>${FXW('\\text{logit} =',['\\text{logit before training}','+ w \\times \\text{detail level}'])}<p>Switch to the shared weight: same pair, same number of steps. In ${J('wire','the picture “Where the logits come from”')}, all four logits hang on the same w. When w changes they all move, and the more detail an answer has, the further it moves.</p>`,
             `<p>到目前为止，这个 Lab 假设每个回答的 logit 都是独立的：推 A 只动 A，压 B 只动 B。可真实的 LLM 里，所有回答都由同一套权重算出来，回答之间相互关联。推高一个回答，和它用词或结构相似的回答也会跟着变。</p><p>这个 Lab 用一个共享权重 w 来模拟这种关联。每个回答有一个细节程度，表示它讲了多少技术细节（示意数值：A ${d[0]}、B ${d[1]}、C ${d[2]}、D ${d[3]}）。它的 logit 这样算：</p>${FXW('\\text{logit} =',['\\text{训练前的 logit}','+ w \\times \\text{细节程度}'])}<p>切换到共享权重：同一对数据，同样的训练步数。看${J('wire','“logit 从哪来”这张图')}：四个 logit 都连在同一个 w 上。w 一变，它们一起动，细节程度越高，动得越多。</p>`)+inMode();},
  obs:r=>obsShared(r)},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},enter:()=>({...DEFAULT,steps:60}),view:'prob',
  body:()=>{const sh=compute({...DEFAULT,mode:'shared',steps:60}), w1=compute({...DEFAULT,steps:60}).wAB, w2=compute({...DEFAULT,beta:2,steps:60}).wAB;
   /* the mechanism with its one number, "2 for every 1" (review 7); the logit rises and the push weights are on the stage and in Show the math (R6.20) */
   const kx=DETAIL[3]/DETAIL[0], below=sh.pi[0]<REFS.b[0], nl='';   // the next lab is the kit's footer under the rail card (R3.11)
   return tx(`<ul><li>DPO trains no reward model, and its loss has no SFT cross-entropy term. An answer’s implicit reward is ${M('\\beta \\log(\\pi / \\pi_{\\text{ref}})')}, a pair’s loss is ${M('-\\log \\sigma(\\text{reward margin})')}, and training only widens the reward margin between the chosen and the rejected answer.</li><li>Answers in no preference pair change too. In this lab, with ${DEFK('separate logits','dpo_tSep')}, C’s and D’s logits never move; their probabilities change only because the four probabilities always add up to 100%. With ${DEFK('a shared weight','dpo_tShared')}, D is still in no pair, yet its logit rises the most: every step raises w, and w lifts D’s logit ${f(kx,1)} for every 1 it lifts A’s. π(A), the chosen answer’s probability, ${below?'even ends up below where it started':chg(REFS.b[0],sh.pi[0])}.</li><li>In this lab, β sets how early the push fades, not how big the first step is: after 60 steps the push ${pushW(w1)} with β = 0.1 but ${pushW(w2)} with β = 2.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “DPO maximizes the chosen answer’s likelihood.”<br>✗ “β is the learning rate.”<br>✗ “DPO skips pairs the SFT model already gets right.”</p></div>${A.shell.toScenarios()}${nl?`<p class="small muted" style="margin-top:10px">${nl}</p>`:''}`,
             `<ul><li>DPO 不训练奖励模型，loss 里也没有 SFT 的交叉熵。回答的隐式奖励是 ${M('\\beta \\log(\\pi / \\pi_{\\text{ref}})')}，一对数据的 loss 是 ${M('-\\log \\sigma(\\text{奖励差})')}，训练只管拉大选中的回答和落选的回答之间的奖励差。</li><li>不在任何偏好对里的回答也会变。在这个 Lab 里，用${DEFK('各自的 logit','dpo_tSep')} 时，C 和 D 的 logit 一直没动；它们的概率会变，只是因为四个概率加起来总是 100%。用${DEFK('共享权重','dpo_tShared')}时，D 仍不在任何偏好对里，logit 却涨得最多：每一步都在调大 w，而 w 每让 A 的 logit 升 1，就让 D 的升 ${f(kx,1)}。选中的回答 A，概率${below?'反而降到了训练前以下':chg(REFS.b[0],sh.pi[0])}。</li><li>在这个 Lab 里，β 决定推力多早衰减，而不是第一步推多少：训练 60 步后，β = 0.1 的推力${pushW(w1)}，β = 2 的推力${pushW(w2)}。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “DPO 就是最大化选中回答的概率。”<br>✗ “β 就是学习率。”<br>✗ “SFT 模型已经排对的偏好对，DPO 会跳过。”</p></div>${A.shell.toScenarios()}${nl?`<p class="small muted" style="margin-top:10px">${nl}</p>`:''}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);

function obsSteps(r){
  const T=r.T, t=r.t, p0=T.pi[0], p=r.pi;
  if(t===0) return tx(`<p><b>Step 0: nothing trained yet.</b> The four answers still have the SFT model’s probabilities.</p>`,
                      `<p><b>第 0 步：还没训练。</b>四个回答还是 SFT 模型给的概率。</p>`);
  /* the result: A's change is the one number (the bars and the cards have the rest), B's in words (R6.20) */
  let h=tx(`<p><b>After ${stepsTxt(t)}, π(A) (how often the model gives A) ${chg(p0[0],p[0])}, and π(B) ${moveW(p0[1],p[1])}.</b></p>`,
           `<p><b>训练 ${t} 步后，π(A)（模型给出 A 的概率）${chg(p0[0],p[0])}，π(B) ${moveW(p0[1],p[1])}。</b></p>`);
  if(S.ds!=='one'||S.mode!=='sep') return h;   // the rest of this part is about C and D outside the one pair, with separate logits
  const pk=peakOf(T,2,t), up=p[2]>p0[2]+5e-4, dn=p[2]<p0[2]-5e-4, still=pk.k===t;
  const q=p[2]/p0[2];   // C and D keep their logits, so the softmax scales both by the same factor
  // 1. what happened to C and D, in words, and the verdict on the prediction
  let v=up?(q<1.2?tx('C and D moved too: both rose a little.','C 和 D 也变了：两个都略有上升。'):tx('C and D moved too: both rose.','C 和 D 也变了：两个都上升了。'))
    :(dn?(q>0.8?tx('C and D moved too: both fell a little.','C 和 D 也变了：两个都略有下降。'):(q>0.5?tx('C and D moved too: both fell.','C 和 D 也变了：两个都下降了。'):tx('C and D moved too: both fell far below where they started.','C 和 D 也变了：两个都比训练前低了一大截。')))
        :tx('C and D are back near where they started.','C 和 D 刚好回到训练前附近。'));
  if(pred){const pt=esc(predTxt(pred));
    if(pred==='drift') v+=tx(` You predicted “${pt}”; that is right.`,`你预测的是“${pt}”，预测对了。`);
    else if(pred==='same') v+=(up||dn)?tx(` You predicted “${pt}”. Nobody labeled them, but they moved anyway.`,`你预测的是“${pt}”。没人标注它们，可它们还是变了。`)
                                      :tx(` You predicted “${pt}”. They did move, and have only now come back.`,`你预测的是“${pt}”。它们其实变过，此刻只是刚好回到原处。`);
    else v+=dn?tx(` You predicted “${pt}”. They did drop, but DPO did not push them.`,`你预测的是“${pt}”。它们确实降了，但不是 DPO 压的。`)
              :(up?tx(` You predicted “${pt}”. So far they rose instead.`,`你预测的是“${pt}”。到目前为止它们反而上升了。`)
                  :tx(` You predicted “${pt}”. Right now they are neither up nor down.`,`你预测的是“${pt}”。此刻它们既没升也没降。`));}
  h+=`<p>${v}</p>`;
  // 2. why, in two sentences
  let why;
  if(up&&still) why=tx('part of what B gives up flows to C and D.','B 让出的概率有一部分流到了 C 和 D。');
  else if(up) why=tx('C and D first took part of what B gave up, and now A is taking it back.','B 让出的份额先分了一些给 C 和 D，现在 A 越占越多，又把它们往下挤。');
  else if(dn) why=tx('C and D first took part of what B gave up, then A took more and more and squeezed them below where they started.','B 让出的份额先分了一些给 C 和 D，后来 A 占得越来越多，又把它们挤到了训练前以下。');
  else why=tx('C and D first took part of what B gave up, and A’s growing share has now brought them back.','B 让出的份额先分了一些给 C 和 D，后来 A 占得越来越多，此刻刚好把它们挤回训练前附近。');
  h+=tx(`<p>In this lab each step changes only A’s and B’s logits; C’s and D’s logits never moved. But the four probabilities always add up to 100%: ${why}</p>`,
        `<p>在这个 Lab 里，每一步只改 A 和 B 的 logit，C 和 D 的 logit 一直没动。可四个概率加起来总是 100%：${why}</p>`);
  return h;
}
function obsFade(r){
  const t=r.t, w=r.wAB;
  if(t===0) return tx(`<p><b>Step 0: the push weight is 0.500</b>, where every pair begins.</p>`,`<p><b>第 0 步：推力权重是 0.500</b>，每一对都从这里开始。</p>`);
  return tx(`<p><b>By step ${t}, each step moves the logits only ${relOf(w)}% as far as the first step did.</b></p><p>The push weight started at exactly 0.5 although the SFT model already preferred B: the implicit reward measures change from the SFT model, so before training every pair is a tie. DPO does not skip pairs the SFT model already ranks right; it pushes each one until that pair’s own reward margin has grown.</p>`,
            `<p><b>到第 ${t} 步，每一步对 logit 的改动只有第一步的 ${relOf(w)}%。</b></p><p>推力权重一开始正好是 0.5，尽管 SFT 模型早就偏向 B：隐式奖励只衡量相对 SFT 模型的变化，所以训练前每一对都是平局。SFT 模型已经排对的偏好对，DPO 也照样推，直到这一对自己的奖励差拉开。</p>`);
}
function obsBeta(r){
  const t=r.t, base=compute({...S,beta:0.1}), b=S.beta, a0=r.T.pi[0][0];
  const bs=[...triedB].sort((x,y)=>x-y), q=bs.map(v=>compute({...S,beta:v}));
  /* the β values tried, as a trend (R6.20): each one's numbers are on the metric cards and the curves when it is picked.
     With the same steps the KL falls as β grows (from step 2 on; after 1 step every β gives the same model) */
  const falls=a=>a.every((v,i)=>!i||v<a[i-1]), ks=q.map(x=>klS(x.kl));
  const tried=bs.length<3||t===0?'':(ks.every(k=>k===ks[0])?tx(`<p class="small">The β values you tried, each after ${stepsTxt(t)}: ${t===1?'they all give the same model':'so far they all give almost the same model'}.</p>`,`<p class="small">已试过的 β（都训练 ${t} 步）：${t===1?'模型都一样':'到目前为止，模型都几乎一样'}。</p>`)
    :(!falls(q.map(x=>x.kl))?'':(falls(q.map(x=>x.pi[0]))?tx(`<p class="small">The β values you tried, each after ${stepsTxt(t)}: the larger β, the lower the KL and the lower π(A).</p>`,`<p class="small">已试过的 β（都训练 ${t} 步）：β 越大，KL 越小，π(A) 也越低。</p>`)
      :tx(`<p class="small">The β values you tried, each after ${stepsTxt(t)}: the larger β, the lower the KL.</p>`,`<p class="small">已试过的 β（都训练 ${t} 步）：β 越大，KL 越小。</p>`))));
  if(t===0) return tx(`<p><b>Step 0: nothing trained yet</b>, so every β still shows the SFT model.</p>`,`<p><b>第 0 步：还没训练</b>，所以任何 β 下都还是 SFT 模型。</p>`);
  const k=klS(r.kl), k0=klS(base.kl);
  if(b===0.1) return tx(`<p><b>β = 0.1 is the initial setup:</b> after ${stepsTxt(t)}, the KL from the SFT model is ${k} nats. Pick another β to compare.</p>`,`<p><b>β = 0.1 是初始设置：</b>训练 ${t} 步后，离 SFT 模型的 KL 是 ${k} nats。换一个 β 来对比。</p>`)+tried;
  /* the result: the KL against the initial setup's is the one key number (a pair); π(A) in words */
  let h;
  if(k===k0) h=t===1?tx(`<p><b>β = ${bTxt(b)}, same ${stepsTxt(t)}: the same model as with β = 0.1, the initial setup.</b></p>`,`<p><b>β = ${bTxt(b)}，同样训练 ${t} 步：模型和初始设置 β = 0.1 时一样。</b></p>`)
                    :tx(`<p><b>β = ${bTxt(b)}, same ${stepsTxt(t)}: so far almost the same model as with β = 0.1, the initial setup.</b></p>`,`<p><b>β = ${bTxt(b)}，同样训练 ${t} 步：到目前为止，模型和初始设置 β = 0.1 时几乎一样。</b></p>`);
  else {const less=r.kl<base.kl, pA=r.pi[0], pA0=base.pi[0];
    const aw=pc(pA)===pc(pA0)?tx('π(A) ends up about the same','π(A) 相差无几'):(pA>a0&&pA0>a0?(pA<pA0?tx('π(A) rises less','π(A) 也涨得更少'):tx('π(A) rises more','π(A) 涨得更多')):(pA<pA0?tx('π(A) ends up lower','π(A) 最后更低'):tx('π(A) ends up higher','π(A) 最后更高')));
    h=tx(`<p><b>β = ${bTxt(b)}, same ${stepsTxt(t)}: the model changes ${less?'less':'more'} than with β = 0.1, the initial setup. Its KL from the SFT model is ${k} nats instead of ${k0}, and ${aw}.</b></p>`,
         `<p><b>β = ${bTxt(b)}，同样训练 ${t} 步：模型改动得比初始设置 β = 0.1 时更${less?'少':'多'}。它离 SFT 模型的 KL 是 ${k} nats，初始设置时是 ${k0} nats；${aw}。</b></p>`);}
  /* why, in words: the push weights themselves are on the push-weight curve and in Show the math */
  h+=tx(`<p>Why: the reward margin is the difference of two implicit rewards, and each implicit reward ${M('\\hat{r} = \\beta \\log(\\pi / \\pi_{\\text{ref}})')} carries the factor β. So with a larger β a small lead of A already makes the reward margin large and the push weight small (compare the ${J('curve-wt','push-weight curve')}). β is therefore not a learning rate: in this lab, a larger β makes the push fade earlier, so in the same number of steps the model changes less and stays closer to the SFT model.</p>`,
        `<p>原因：奖励差是两个隐式奖励之差，而隐式奖励 ${M('\\hat{r} = \\beta \\log(\\pi / \\pi_{\\text{ref}})')} 里带着 β。所以 β 越大，A 只要领先一点点，奖励差就很大，推力权重就掉下来（对比${J('curve-wt','推力权重曲线')}）。可见 β 不是学习率：在这个 Lab 里，β 越大，推力衰减得越早，同样的步数下模型改动得越少，离 SFT 模型越近。</p>`);
  return h+tried;
}
function obsShared(r){
  const t=r.t, p0=r.T.pi[0], sep=compute({...S,mode:'sep'});
  if(S.mode!=='shared') return tx(`<p><b>Separate logits: after ${stepsTxt(t)}, π(A) ${chg(p0[0],r.pi[0])}.</b> Switch to the shared weight to compare.</p>`,`<p><b>各自的 logit：训练 ${t} 步后，π(A) ${chg(p0[0],r.pi[0])}。</b>切换到共享权重来对比。</p>`);
  if(t===0) return tx(`<p><b>Step 0: nothing trained yet.</b> The shared weight w is 0, so every logit is still its logit before training.</p>`,`<p><b>第 0 步：还没训练。</b>共享权重 w 是 0，每个 logit 都还是训练前的值。</p>`);
  const pk=peakOf(r.T,0,t), below=r.pi[0]<p0[0], falling=pk.k<t;   // before A's peak it is still rising (R6.10)
  /* the result: A's change is the one number; separate logits, A's peak, D's rise and the loss in words. The logit rises (w × detail
     level) are on the bars, w in Show the math, the loss on its card (R6.20) */
  const ds=sep.pi[0]-r.pi[0], sepW=ds>=0.3?tx('far higher','要高得多'):(ds>=0.05?tx('higher','更高'):(ds>0.005?tx('a little higher','略高一些'):(ds>-0.005?tx('about the same','差不多'):tx('lower','更低'))));
  const kx=DETAIL[3]/DETAIL[0], most=r.pi[3]>0.5;
  return tx(`<p><b>Shared weight, ${stepsTxt(t)}: π(A) ${chg(p0[0],r.pi[0])}${below?', below where it started':''}</b> (with separate logits it ends up ${sepW}). ${falling?(below?'A rose at first, then turned and fell.':'A rose at first and is falling now.'):'A is still rising; train longer to see it turn.'}</p><p>Why: D is in no preference pair, but all four logits come from the same w. A has more detail than B, so widening the reward margin between A and B means raising w, and that lifts every logit by its detail level: D’s logit rises ${kx===2?'twice as fast as':`${f(kx,1)} times as fast as`} A’s, so ${falling?(most?'D took most of the probability, A’s share included':'D gained the most and took probability from A too'):'D gains faster than A and holds A’s gain down'}. The loss still fell: it sees only the reward margin between A and B, which kept growing.</p>${falling?'<p class="small">Real DPO runs show it too: the chosen answers’ probability often falls during training.</p>':''}`,
            `<p><b>共享权重，训练 ${t} 步：π(A) ${chg(p0[0],r.pi[0])}${below?'，比训练前还低':''}</b>（各自的 logit 时${sepW}）。${falling?(below?'A 先升后降。':'A 先升了一些，现在正在下降。'):'A 还在上升，多训练几步就能看到它掉头。'}</p><p>原因：D 不在任何偏好对里，但四个 logit 都由同一个 w 决定。A 比 B 的细节多，要拉大 A 和 B 的奖励差只能调大 w，而这会按细节程度抬高所有 logit：D 的 logit 涨幅是 A 的${kx===2?'两倍':` ${f(kx,1)} 倍`}，${falling?(most?'所以 D 拿走了大部分概率，连 A 的那份也被它抢走':'所以 D 涨得最多，还抢走了 A 的概率'):'所以 D 涨得比 A 快，把 A 的涨幅压了下去'}。loss 照样下降：它只看 A 和 B 之间的奖励差，而这个差一直在拉大。</p>${falling?'<p class="small">真实的 DPO 训练里也有这种现象：选中回答的概率在训练中常常下降。</p>':''}`);
}

/* ---------- views ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. Predict first, then train on one preference pair and see how each DPO step changes the four answers’ probabilities and what β does.','一共 8 个部分。先预测，再在一对偏好数据上训练，看 DPO 每一步怎么改变四个回答的概率，以及 β 起什么作用。'),
  sandbox:tx('Pick a kind of preference data and SFT model and watch DPO train on it.','选一种偏好数据和 SFT 模型，看 DPO 在上面怎么训练。'),
  challenge:tx(`Change only β so that after 60 DPO steps the model gives A at least half the time while its KL from the SFT model stays at or below 0.8 nats.`,
               `只改 β，让模型训练 60 步 DPO 后至少一半的时候给出 A，同时离 SFT 模型的 KL 不超过 0.8 nats。`)
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser runs the actual DPO gradient steps on a small model.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>Four fixed answers to one prompt; the model’s probabilities are the softmax of four logits. The reference ${M('\\pi_{\\text{ref}}')} is the SFT model before training and never changes.</li><li>The exact DPO loss (Eq. 7 of the DPO paper, averaged over the pairs) and its exact gradient; full-batch gradient descent, rerun from the SFT model for the number of steps you set.</li><li>Implicit rewards ${M('\\hat{r} = \\beta \\log(\\pi / \\pi_{\\text{ref}})')}, and ${M('\\text{KL}(\\pi \\,‖\\, \\pi_{\\text{ref}})')} over the four answers, in nats.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>Each answer is one choice, not a sequence of tokens. A real answer’s log-probability is a sum over its tokens and moves much further, so this lab needs larger β values for β’s effect to show.</li><li>Each update is multiplied by ${M('\\eta / \\beta')} to cancel the β in the gradient, so the first step is the same size for every β. With plain gradient descent at a fixed learning rate, a larger β would make the first steps larger; Adam, used in real runs, largely ignores such a constant factor in the gradient (${KB()}). TRL defaults to AdamW with learning rate 1e-6 and the DPO paper used RMSprop; η = 0.1 moves four logits and is not comparable to them.</li><li>The answers, the probabilities before training and the detail levels (how much technical detail each answer gives) are illustrative. No sampling, no minibatches, no noise.</li><li>The DPO paper reports that dropping the push weight (pushing every pair equally hard) can make the model degenerate.</li></ul></div><div><div class="eyebrow">Sources</div><ul><li>Loss, gradient and derivation: ${RAF()} (DPO, NeurIPS 2023). β = 0.1 is the default of the DPO paper and of TRL’s DPOConfig (v1.13.0).</li><li>Llama 3’s post-training ran rounds of SFT and DPO with β = 0.1, adding a negative log-likelihood term with weight 0.2 on the chosen answers; it also trained a reward model, used to pick the best of several sampled answers (${LLAMA()}, §4.1–4.2).</li><li>The chosen answers’ probability often falls in real DPO runs: ${RAZIN()}. DPO-Positive adds a penalty against it: ${PAL()}.</li><li>In the RLHF objective, β weights the KL penalty, so a larger β keeps its best policy closer to the SFT model. With a single label per comparison, though, the best policy for the DPO loss gives the rejected answer zero probability whatever β is: ${AZAR()}.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-09-24. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>页面在你的浏览器里，对一个小模型真的做 DPO 的梯度下降。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>一个提问的四个固定回答；模型的概率是四个 logit 的 softmax。参考模型 ${M('\\pi_{\\text{ref}}')} 就是训练前的 SFT 模型，训练中不变。</li><li>精确的 DPO loss（DPO 论文公式 7，对偏好对取平均）和它的精确梯度；全批量梯度下降，每次都从 SFT 模型重新训练到你设定的步数。</li><li>隐式奖励 ${M('\\hat{r} = \\beta \\log(\\pi / \\pi_{\\text{ref}})')}，以及四个回答上的 ${M('\\text{KL}(\\pi \\,‖\\, \\pi_{\\text{ref}})')}，单位 nats。</li></ul></div><div><div class="eyebrow">简化和假设</div><ul><li>每个回答是一个整体选项，不是 token 序列。真实回答的 log 概率是它所有 token 的和，能变化的幅度大得多，所以这个 Lab 要用更大的 β，β 的作用才看得出来。</li><li>每一步对 logit 的改动都乘以 ${M('\\eta / \\beta')}，抵消梯度里的 β，这样不管 β 取多少，第一步的改动都一样大。如果用固定学习率的普通梯度下降，β 越大，最初几步反而越大；真实训练用的 Adam 基本不受梯度整体缩放的影响（${KB()}）。TRL 默认用 AdamW、学习率 1e-6，DPO 论文用 RMSprop；这里的 η = 0.1 只动四个 logit，和它们没法直接比。</li><li>回答、训练前的概率和细节程度（每个回答讲了多少技术细节）都是示意数值。没有采样、没有 minibatch、没有噪声。</li><li>DPO 论文指出，去掉推力权重（每一对都一样用力地推）可能让模型退化。</li></ul></div><div><div class="eyebrow">来源</div><ul><li>loss、梯度和推导：${RAF()}（DPO，NeurIPS 2023）。β 默认 0.1：DPO 论文和 TRL 的 DPOConfig（v1.13.0）。</li><li>Llama 3 的后训练做了多轮 SFT 和 DPO，β = 0.1，并在选中的回答上加了一项权重 0.2 的负对数似然 loss；它也训练了一个奖励模型，用来从多个采样回答里挑出最好的（${LLAMA()}，§4.1–4.2）。</li><li>真实的 DPO 训练里，选中回答的概率常常下降：${RAZIN()}。DPO-Positive 加了一项惩罚来防止它：${PAL()}。</li><li>在 RLHF 目标里，β 是 KL 惩罚的系数，β 越大，最优策略离 SFT 模型越近。但每次比较只有一个标注时，DPO loss 的最优解会让落选回答的概率变成 0，与 β 无关：${AZAR()}。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-09-24。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(
 '<p>After pretraining, a model is fine-tuned on example answers (SFT). Then people compare pairs of answers to the same prompt and say which is better. Classic RLHF first trains a separate reward model on those pairs. Then PPO trains the chat model: it keeps generating answers, the reward model scores them, and the model moves toward higher scores. A KL penalty keeps it close to the SFT model.</p><p>DPO optimizes the same objective with one loss on the pairs: no reward model, no answers generated during training, only a frozen copy of the SFT model as the reference.</p>',
 '<p>预训练之后，先用示范回答微调模型（SFT）。然后收集偏好：同一个提问的两个回答，人来判断哪个更好。经典 RLHF 先用这些偏好对单独训练一个奖励模型。再用 PPO 训练聊天模型：模型不断生成回答，奖励模型给它们打分，模型朝高分的方向改。一项 KL 惩罚让模型别离 SFT 模型太远。</p><p>DPO 用一个作用在偏好对上的 loss 优化同一个目标：不要奖励模型，训练时也不用生成回答，只留一份冻结的 SFT 模型作为参考模型。</p>');

/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers.
   A new case types only its givens; every other number comes from the engine's sigm, softplus and logSoftmax, and each number in a
   "why" redoes from the numbers shown before it (R6.6). No exit uses the challenge's setup, so none gives away its β. */
/* one line at 1280 and wider in both languages: a second line pushes part 3's inline control below the fold at 1280 x 800 (R3.2) */
const OUTCOME=()=>tx('explain what DPO uses instead of a reward model and why its push fades, and compute a pair’s loss.','讲清 DPO 用什么代替奖励模型、它的推力为什么会衰减，并算出一对数据的 loss。');
/* new case 1: one pair, β = 0.1, log-ratios +2 (chosen) and −1 (rejected). The options are its loss and the loss after each of three
   slips (β left out, 2 + (−1) for 2 − (−1), the reward margin taken the wrong way round), in increasing order */
function exPair(){const b=0.1, lw=2, ll=-1, gap=lw-ll, m=b*gap, pw=sigm(-m);
  const opts=[{k:'nob',v:softplus(-gap)},{k:'ok',v:softplus(-m)},{k:'sum',v:softplus(-b*(lw+ll))},{k:'flip',v:softplus(m)}].sort((x,y)=>x.v-y.v);
  return {b,lw,ll,gap,m,pw,loss:softplus(-m),eN:r3(Math.exp(-m)),eP:r3(Math.exp(m)),rel:Math.round(r3(pw)/sigm(0)*100),opts};}
/* new case 2: another SFT model, and one DPO step on A ≻ B with separate logits, by train()'s update. Before training every reward
   margin is 0, so the push weight is σ(0); the step is scaled by η/β, so A's logit rises by η × push weight and B's falls as much */
function exStep(){const ref=[0.6,0.1,0.2,0.1], b=DEFAULT.beta, eta=DEFAULT.eta, wt=sigm(0), g=[-b*wt,b*wt,0,0];
  const pi=logSoftmax(ref.map((p,k)=>Math.log(p)-(eta/b)*g[k])).map(Math.exp), dz=eta*wt, eu=r3(Math.exp(dz)), ed=r3(Math.exp(-dz));
  return {ref,eta,dz,pi,eu,ed,sum:r3(ref[0]*eu+ref[1]*ed+ref[2]+ref[3]),a:pi[2]>ref[2]+5e-4?0:(pi[2]<ref[2]-5e-4?2:1)};}   // options: rises, stays, falls
const EXITS=()=>{const P=exPair(), Q=exStep(), R=Q.ref, d3=v=>f(v,3), m1=f(P.m,1), g0=f(P.gap,0), w0=f(sigm(0),1), rp=k=>f(R[k],1);
 const dif=`${f(P.lw,0)} − ${pmx(P.ll,0)}`, mar=`${bTxt(P.b)} × (${dif}) = ${m1}`, pC=(Q.pi[2]*100).toFixed(1)+'%';
 const RM=tx('reward margin','奖励差'), PW=tx('push weight','推力权重'), RHS='\\hat{r} = \\beta \\log(\\pi / \\pi_{\\text{ref}})';
 const NOTE={ok:null,
  nob:tx(`That is ${M(`-\\log\\sigma(${g0})`)}, which leaves out β. The reward margin is ${mar}, not ${g0}.`,`这是 ${M(`-\\log\\sigma(${g0})`)}，漏了 β。奖励差是 ${mar}，不是 ${g0}。`),
  sum:tx(`That is ${M(`-\\log\\sigma(${f(P.b*(P.lw+P.ll),1)})`)}, from ${f(P.lw,0)} + ${pmx(P.ll,0)} = ${f(P.lw+P.ll,0)}. The reward margin uses the difference of the two log-ratios: ${mar}.`,
         `这是 ${M(`-\\log\\sigma(${f(P.b*(P.lw+P.ll),1)})`)}，来自 ${f(P.lw,0)} + ${pmx(P.ll,0)} = ${f(P.lw+P.ll,0)}。奖励差用的是两个 log 比值之差，等于 ${mar}。`),
  flip:tx(`That is ${M(`-\\log\\sigma(${nT(-P.m,1)})`)}, with the reward margin taken as the rejected answer’s implicit reward minus the chosen one’s. It is the chosen minus the rejected, +${m1}.`,
          `这是 ${M(`-\\log\\sigma(${nT(-P.m,1)})`)}，把奖励差算成了落选回答的隐式奖励减选中回答的。奖励差应该是选中减落选，等于 +${m1}。`)};
 return {
 core:()=>chSubmitted, explain:[
  {q:tx('DPO trains no reward model. Where does the reward in its loss come from?','DPO 不训练奖励模型。那它的 loss 里的奖励是从哪来的？'),
   o:[tx('The loss compares the model with a frozen copy of the SFT model: an answer’s implicit reward measures how much likelier the model has made it than that copy does, so every answer starts at 0 and rises as it gets more likely.',
         'loss 拿模型和参考模型（冻结的 SFT 模型）比：回答的隐式奖励看它的概率比参考模型高了多少，所以每个回答都从 0 起步，越常被给出就越高。'),
      tx('The loss uses the preference labels as rewards: the chosen answer counts as 1 and the rejected one as 0, so each step pushes the chosen answer’s probability toward 100%.',
         'loss 把偏好标注直接当作奖励：选中的回答记 1 分，落选的记 0 分，所以每一步都要把选中回答的概率一路推向 100%。'),
      tx('The loss uses the model’s own probabilities: an answer’s implicit reward is β times its own log-probability, so the answer the model already gives most often starts out with the highest reward.',
         'loss 直接用模型自己的概率：回答的隐式奖励就是它的 log 概率乘 β，所以模型本来就最常给出的那个回答，一开始奖励就最高。')],a:0,
   n:[null,tx('The labels only say which answer of a pair is better, not how good either one is. The loss sees only the reward margin between the two, so the chosen answer’s probability need not rise: with a shared weight it can even fall.',
              '标注只说明一对里哪个回答更好，不说明各自有多好。loss 只看两个回答之间的奖励差，选中回答的概率不一定上升：用共享权重时，它甚至可能下降。'),
     tx(`Without ${M('\\pi_{\\text{ref}}')}, the rejected answer would start ahead whenever the SFT model favors it, as this lab’s SFT model favors B over A. Dividing by ${M('\\pi_{\\text{ref}}')} makes every implicit reward 0 before training, so every pair starts as a tie.`,
        `没有 ${M('\\pi_{\\text{ref}}')}，只要 SFT 模型偏向落选的回答，落选的回答一开始就领先，比如这个 Lab 里的 SFT 模型就偏向 B 而不是 A。除以 ${M('\\pi_{\\text{ref}}')} 之后，训练前每个回答的隐式奖励都是 0，所以每一对都从平局开始。`)],
   /* inline formulas: the verdict's first paragraph must stay short enough to show its end after a pick (exits_flow) */
   why:tx(`An answer’s implicit reward is ${M(RHS)}: how much more often the model gives it than the frozen SFT model does (as a natural log), times β. With A chosen and B rejected, a pair’s loss is ${M('-\\log\\sigma(\\hat{r}_A - \\hat{r}_B)')}, so DPO needs only the preference pairs and that frozen copy. Before training ${M('\\pi = \\pi_{\\text{ref}}')}, so every ${M('\\hat{r}')} is 0 and the loss is log 2 ≈ ${f(LOG2,3)}.`,
          `一个回答的隐式奖励是 ${M(RHS)}，也就是模型现在给出它的概率比参考模型高了多少（取自然对数），再乘 β。A 是选中的回答，B 是落选的回答，一对数据的 loss 是 ${M('-\\log\\sigma(\\hat{r}_A - \\hat{r}_B)')}，所以 DPO 只要偏好对和参考模型，也就是这份冻结的 SFT 模型。训练前 ${M('\\pi = \\pi_{\\text{ref}}')}，每个 ${M('\\hat{r}')} 都是 0，loss 是 log 2 ≈ ${f(LOG2,3)}。`)},
  {q:tx('In this lab, each DPO step on the pair A ≻ B moves A’s and B’s logits less than the step before. Why?','在这个 Lab 里，用 A ≻ B 这一对训练 DPO，每一步对 A 和 B 的 logit 的改动都比上一步小。为什么？'),
   o:[tx('Each step is scaled by the push weight: every step widens the reward margin between A and B, so the push weight and the next step shrink.',
         '每一步都要乘以推力权重：每走一步，A 和 B 之间的奖励差都会拉大，推力权重随之变小，下一步也就跟着变小。'),
      tx('The step size η is lowered on a schedule as training goes on: each later step gets a smaller η, so it moves the logits less, whatever the reward margin is.',
         '步长 η 在训练中按预设的计划逐步调低：越往后的一步 η 越小，对 logit 的改动也就越小，和奖励差是多少无关。'),
      tx('As π(A) climbs toward 100%, the softmax saturates: raising A’s logit then changes π(A) less and less, so the gradient on the logits shrinks as well.',
         'π(A) 越接近 100%，softmax 越饱和：再抬高 A 的 logit，π(A) 也几乎不变，所以 logit 上的梯度也跟着变小。')],a:0,
   n:[null,tx(`In this lab η stays the same for the whole run. A step moves A’s logit up and B’s down by ${M('\\eta \\times \\text{push weight}')}, and only the push weight shrinks.`,
              `在这个 Lab 里，η 在整次训练中都不变。一步让 A 的 logit 上升、B 的下降，幅度都是 ${M('\\eta \\times \\text{推力权重}')}，变小的只有推力权重。`),
     tx(`In this lab the gradient on the logits does not depend on π(A). ${M('\\log\\pi(A) - \\log\\pi(B)')} is just A’s logit minus B’s, so the softmax cancels out of the reward margin. Only the push weight shrinks.`,
        `在这个 Lab 里，logit 上的梯度和 π(A) 无关。${M('\\log\\pi(A) - \\log\\pi(B)')} 就是 A 的 logit 减 B 的 logit，softmax 在奖励差里抵消了。变小的只有推力权重。`)],
   why:tx(`In this lab a step moves A’s logit up and B’s down by ${M('\\eta \\times \\text{push weight}')}. The push weight is${FX('Push weight','\\sigma({-\\text{reward margin}})')}Under the implicit reward, it is the chance that B still beats A. Before training the reward margin is 0 and the push weight is ${M(`\\sigma(0) = ${w0}`)}. Each step widens the margin, so the push fades on its own.`,
          `在这个 Lab 里，一步让 A 的 logit 上升、B 的下降，幅度都是 ${M('\\eta \\times \\text{推力权重}')}。推力权重是${FX('推力权重','\\sigma({-\\text{奖励差}})')}按隐式奖励算，它就是 B 仍然胜过 A 的概率。训练前奖励差是 0，推力权重是 ${M(`\\sigma(0) = ${w0}`)}。每一步都拉大奖励差，推力就自己衰减了。`)}],
 transfer:[
  {q:tx(`A preference pair with β = ${bTxt(P.b)}: the chosen answer’s log-ratio ${M('\\log(\\pi/\\pi_{\\text{ref}})')} is ${sg(P.lw,0)} and the rejected answer’s is ${sg(P.ll,0)}. What is this pair’s DPO loss?`,
        `一对偏好数据，β = ${bTxt(P.b)}。选中回答的 log 比值 ${M('\\log(\\pi/\\pi_{\\text{ref}})')} 是 ${sg(P.lw,0)}，落选回答的是 ${sg(P.ll,0)}。这一对的 DPO loss 是多少？`),
   o:P.opts.map(x=>d3(x.v)),a:P.opts.findIndex(x=>x.k==='ok'),n:P.opts.map(x=>NOTE[x.k]),
   /* a worked computation (R6.6), one block per quantity (H2): the named formula, the numbers under it, the result */
   why:tx(`<span class="dpo-pl">The reward margin is β times the chosen log-ratio minus the rejected one.</span>${EQ([[TX('reward margin','奖励差'),'=',`${bTxt(P.b)} \\times (${f(P.lw,0)} - (${nT(P.ll,0)}))`],['','=',m1]])}<span class="dpo-pl">The loss is ${M('-\\log\\sigma')} of the reward margin.</span>${EQ([['L','=',`-\\log\\sigma(${m1})`],['','=',`\\log(1 + e^{-${m1}})`],['','=',`\\log(1 + ${d3(P.eN)})`],['','=',d3(P.loss)]])}<span class="dpo-pl">That is only a little under log 2 ≈ ${f(LOG2,3)}, the loss of any pair before training, because β shrinks the log-ratio difference of ${g0} to a reward margin of only ${m1}.</span>${EQ([[TX(PW,PW),'=',`\\sigma({-${m1}})`],['','=',`\\frac{1}{1 + e^{${m1}}}`],['','=',`\\frac{1}{1 + ${d3(P.eP)}}`],['','=',d3(P.pw)]])}<span class="dpo-pl">The push weight is ${P.rel>=50?'still most':'already under half'} of the ${w0} it starts at.</span>`,
          `<span class="dpo-pl">奖励差等于 β 乘以选中回答的 log 比值减落选回答的。</span>${EQ([[TX('reward margin','奖励差'),'=',`${bTxt(P.b)} \\times (${f(P.lw,0)} - (${nT(P.ll,0)}))`],['','=',m1]])}<span class="dpo-pl">loss 是奖励差的 ${M('-\\log\\sigma')}。</span>${EQ([['L','=',`-\\log\\sigma(${m1})`],['','=',`\\log(1 + e^{-${m1}})`],['','=',`\\log(1 + ${d3(P.eN)})`],['','=',d3(P.loss)]])}<span class="dpo-pl">它只比训练前任何一对的 loss（log 2 ≈ ${f(LOG2,3)}）低一点，因为 β 把 log 比值之差 ${g0} 缩成了只有 ${m1} 的奖励差。</span>${EQ([[TX(PW,PW),'=',`\\sigma({-${m1}})`],['','=',`\\frac{1}{1 + e^{${m1}}}`],['','=',`\\frac{1}{1 + ${d3(P.eP)}}`],['','=',d3(P.pw)]])}<span class="dpo-pl">推力权重${P.rel>=50?'仍是':'已经不到'}一开始 ${w0} 的${P.rel>=50?'大半':'一半'}。</span>`)},
  {q:tx(`Another prompt: before training the SFT model gives A ${pc(R[0])} of the time, B ${pc(R[1])}, C ${pc(R[2])} and D ${pc(R[3])}, and annotators prefer A to B (A ≻ B). Each answer has its own logit. After one DPO step, what happens to π(C)?`,
        `换一个提问：训练前 SFT 模型给出 A、B、C、D 的概率分别是 ${pc(R[0])}、${pc(R[1])}、${pc(R[2])}、${pc(R[3])}，标注员比较 A 和 B，更喜欢 A（A ≻ B）。每个回答有自己的 logit。训练一步 DPO 后，π(C) 会怎样？`),
   o:[tx('It rises','上升'),tx(`It stays at ${pc(R[2])}`,`保持 ${pc(R[2])}`),tx('It falls','下降')],a:Q.a,
   n:[tx('C rises only while B gives up more than A takes, as in the first steps of this lab’s initial setup, where B starts far ahead of A. Here A holds far more than B, so A takes more than B gives up.',
         '只有 B 让出的比 A 拿走的多时，C 才会上升，比如这个 Lab 初始设置的前几步，那里 B 一开始远远领先 A。这里 A 的概率比 B 多得多，A 拿走的比 B 让出的多。'),
      tx(`C’s logit stays, but the softmax divides C’s ${M('e^{\\text{logit}}')} by the sum over all four answers, and that sum changes when A’s and B’s logits move. DPO also pushes a pair that the SFT model already ranks right.`,
         `C 的 logit 不动，但 softmax 要把 C 的 ${M('e^{\\text{logit}}')} 除以四个回答的总和，A 和 B 的 logit 一动，这个总和就跟着变。SFT 模型已经排对的偏好对，DPO 也照样推。`),null],
   /* short steps: the logit changes, the rule, the sum, π(C), then why in words (R6.6, H2) */
   why:tx(`<span class="dpo-pl">In this lab the first step raises A’s logit by ${M('\\eta \\times \\text{push weight}')} = ${f(Q.eta,1)} × ${w0} = ${f(Q.dz,2)} and lowers B’s by as much; C’s stays. After the step each probability is its old value times ${M('e^{\\Delta}')}, divided by the sum of the four (Δ is its logit change):</span>${EQ([[TX('sum','和'),'=',`${rp(0)} \\times e^{${f(Q.dz,2)}} + ${rp(1)} \\times e^{-${f(Q.dz,2)}} + ${rp(2)} + ${rp(3)}`],['','=',`${rp(0)} \\times ${d3(Q.eu)} + ${rp(1)} \\times ${d3(Q.ed)} + ${f(R[2]+R[3],1)}`],['','=',d3(Q.sum)]])}<span class="dpo-pl">So π(C) falls to ${pc(R[2])} ÷ ${d3(Q.sum)} = ${pC}. A holds far more probability than B, so A takes more than B gives up, and C and D make up the difference.</span>`,
          `<span class="dpo-pl">在这个 Lab 里，第一步让 A 的 logit 上升 ${M('\\eta \\times \\text{推力权重}')} = ${f(Q.eta,1)} × ${w0} = ${f(Q.dz,2)}，B 的下降同样多，C 的不动。这一步之后，每个概率等于原来的值乘以 ${M('e^{\\Delta}')}，再除以四项之和（Δ 是它的 logit 变化）：</span>${EQ([[TX('sum','和'),'=',`${rp(0)} \\times e^{${f(Q.dz,2)}} + ${rp(1)} \\times e^{-${f(Q.dz,2)}} + ${rp(2)} + ${rp(3)}`],['','=',`${rp(0)} \\times ${d3(Q.eu)} + ${rp(1)} \\times ${d3(Q.ed)} + ${f(R[2]+R[3],1)}`],['','=',d3(Q.sum)]])}<span class="dpo-pl">所以 π(C) 降到 ${pc(R[2])} ÷ ${d3(Q.sum)} = ${pC}。A 的概率比 B 多得多，A 拿走的比 B 让出的多，差额由 C 和 D 补上。</span>`)}],
 practice:[
  {id:'rl-for-llms',d:tx('Question 7: from the RLHF objective with its KL penalty, derive the best policy, then the DPO loss through the Bradley–Terry model, and say why Z(x) cancels and what data DPO needs.',
                         '第 7 题：从带 KL 惩罚的 RLHF 目标推出最优策略，再经 Bradley–Terry 模型推出 DPO loss。还要说明 Z(x) 为什么会消掉，DPO 需要什么数据。')},
  {id:'cross-entropy',d:tx('Parts 1 and 4: compute log-probabilities from logits without overflow, the numbers the DPO loss compares, then show that cross-entropy is entropy plus KL and derive its gradient.',
                           '第 1、4 部分：用不会溢出的办法从 logit 算出 log 概率（DPO loss 比较的正是这些数），再证明交叉熵等于熵加 KL，并推导它的梯度。')}]};};

function headerHTML(){
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:'DPO',title:tx('How does DPO pull a model toward preferences without a reward model?','不训练奖励模型，DPO 怎么把模型拉向偏好？'),
    minutes:15,level:tx('Core','核心'),fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: RLHF','前置知识：RLHF'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  const lg=(tip,sw,txt,jump)=>`<span data-tip="${tip}" tabindex="0"${jump?` data-jump="${jump}"`:''}><i class="sw ${sw}"></i><span>${txt}</span></span>`;
  return `<div class="panel stage dpo-stage" id="dpo-stage">`+
   `<div class="stage-head dpo-head"><div class="dpo-st-title" id="dpo-st-title"></div><div class="stage-actions"><span class="status" id="dpo-status" tabindex="0"></span></div></div>`+
   `<div class="dpo-st-sub" id="dpo-st-sub"></div>`+
   `<div class="dpo-transport" id="dpo-transport"><button type="button" class="btn sm" data-act="play" data-play data-tip="dpo_tipPlay">${playLabel()}</button>${speedCtl()}<button type="button" class="btn ghost sm" data-act="step1" data-tip="dpo_tipStep">${tx('+1 step','+1 步')}</button><button type="button" class="btn ghost sm" data-act="step10" data-tip="dpo_tipStep10">${tx('+10 steps','+10 步')}</button><button type="button" class="btn ghost sm" data-act="reset0" data-tip="dpo_tipReset">${tx('Back to step 0','回到第 0 步')}</button><span class="dpo-tnote small muted" id="dpo-tnote"></span></div>`+
   `<div class="dpo-blk-h" id="dpo-bars-h">${tx('How often the model gives each answer','模型给出每个回答的概率')}</div>`+
   `<p class="dpo-prompt">${tx(`Prompt: “${esc(PROMPT.en)}” The four answers are examples written for this lab.`,`提问：“${esc(PROMPT.zh)}”四个回答是为本 Lab 写的示例。`)}</p>`+
   `<div class="legend-row">${lg('dpo_lgW','dpo-sw-w',tx('✓ chosen','✓ 选中'),'ds')}${lg('dpo_lgL','dpo-sw-l',tx('✗ rejected','✗ 落选'),'ds')}${lg('dpo_lgO','dpo-sw-o',tx('not labeled','未标注'),'ds')}${lg('dpo_lgRef','dpo-sw-ref',tx('SFT model (before training)','SFT 模型（训练前）'),'ref')}</div>`+
   `<div class="stage-graph graph dpo-graph" id="dpo-graph"><div class="dpo-bars" id="dpo-bars"></div><div class="dpo-wire" id="dpo-wire" hidden></div>`+
   `<div class="dpo-curve-head"><div class="dpo-curve-title" id="dpo-curve-title" tabindex="0"></div>${A.ui.seg({id:'dpo-view',key:'view',label:tx('Curve','曲线'),options:VIEWS.map(([v,o])=>[v,L(o)]),value:view})}</div>`+
   `<svg class="dpo-curve" id="dpo-curve" role="img" aria-label=""></svg><div class="dpo-curve-key" id="dpo-curve-key"></div></div>`+
   `<div class="metrics dpo-metrics" id="dpo-metrics"></div>`+
   `<details class="math" id="dpo-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="dpo-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="dpo-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="dpo-lock" hidden></p>`+
   `<div class="lockable">`+A.ui.seg({id:'dpo-ctl-ds',key:'ds',label:tx('Preference pairs','偏好对'),options:['one','three','mixed'].map(k=>[k,L(DSNAME[k])]),value:S.ds,
      note:tx('1 pair: A ≻ B. 3 pairs: A ≻ B, A ≻ C and A ≻ D. Mixed labels: A ≻ B twice and B ≻ A once. The loss is averaged over the pairs.','1 对：A ≻ B。3 对：A ≻ B、A ≻ C、A ≻ D。标注不一致：A ≻ B 两次，B ≻ A 一次。loss 对所有偏好对取平均。')})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'dpo-ctl-ref',key:'ref',label:tx('SFT model (before training)','SFT 模型（训练前）'),options:[['b',refName('b')],['s',refName('s')]],value:S.ref,
      note:tx(`${refName('b')}: before training the SFT model gives A ${pc(REFS.b[0])} of the time and B ${pc(REFS.b[1])}. ${refName('s')}: A ${pc(REFS.s[0])} and B ${pc(REFS.s[1])}. Both are illustrative. This model is also the reference ${M('\\pi_{\\text{ref}}')} and does not change during training.`,`${refName('b')}：训练前 SFT 模型给出 A 的概率是 ${pc(REFS.b[0])}，B 是 ${pc(REFS.b[1])}。${refName('s')}：A ${pc(REFS.s[0])}，B ${pc(REFS.s[1])}。都是示意数值。这个模型也就是参考模型 ${M('\\pi_{\\text{ref}}')}，训练中不变。`)})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'dpo-ctl-mode',key:'mode',label:tx('How answers get their logits','回答的 logit 怎么来'),options:[['sep',L(MODENAME.sep)],['shared',L(MODENAME.shared)]],value:S.mode,
      note:tx(`Separate logits: one logit per answer, and a pair moves only its own two. Shared weight: as in a real network, one weight w sets every logit. Each logit is its value before training plus ${M('w \\times \\text{detail level}')}; the detail level says how much technical detail the answer gives (illustrative values A ${f(DETAIL[0],1)}, B ${f(DETAIL[1],1)}, C ${f(DETAIL[2],1)}, D ${f(DETAIL[3],1)}).`,
              `各自的 logit：每个回答一个 logit，一对数据只动自己那两个。共享权重：像真实网络一样，所有 logit 由一个权重 w 决定。每个 logit 等于训练前的 logit 加上 ${M('w \\times \\text{细节程度}')}。细节程度表示回答讲了多少技术细节（示意数值 A ${f(DETAIL[0],1)}、B ${f(DETAIL[1],1)}、C ${f(DETAIL[2],1)}、D ${f(DETAIL[3],1)}）。`)})+`</div>`+
   A.ui.seg({id:'dpo-ctl-beta',key:'beta',label:tx('β (the weight of the KL penalty in the RLHF objective)','β（RLHF 目标里 KL 惩罚的系数）'),options:BETAS.map(b=>[String(b),bTxt(b)]),value:String(S.beta),
      note:tx('DPO is derived from that objective; the DPO paper and TRL default to β = 0.1. In this lab, the larger β, the less the model changes in the same number of steps, so the closer it stays to the SFT model.','DPO 就是从这个目标推出来的；DPO 论文和 TRL 默认 β = 0.1。在这个 Lab 里，β 越大，同样的训练步数下模型改动得越少，离 SFT 模型越近。')})+
   `<div class="lockable" id="dpo-steps-wrap">`+A.ui.range({id:'dpo-ctl-steps',key:'steps',label:tx('Training steps','训练步数'),min:0,max:MAXS,value:S.steps,out:stepsTxt(S.steps),
      note:tx('Gradient steps taken from the SFT model; each step uses all the pairs.','从 SFT 模型开始做的梯度下降步数；每一步都用上全部偏好对。')})+`</div>`+
   `<details class="adv lockable" id="dpo-adv"><summary>${tx('Step size η','步长 η')}</summary>`+
   A.ui.seg({id:'dpo-ctl-eta',key:'eta',label:tx('Step size η (learning rate)','步长 η（学习率）'),options:ETAS.map(v=>[String(v),String(v)]),value:String(S.eta),
      note:tx(`Each step moves the logits by ${M('(\\eta / \\beta) \\times \\text{gradient}')}. Dividing by β cancels the β the gradient carries, so the first step is the same size for every β. A larger η makes every step larger.`,
              `每一步把 logit 移动 ${M('(\\eta / \\beta) \\times \\text{梯度}')}。除以 β 抵消了梯度里本来带着的 β，所以不管 β 取多少，第一步的改动都一样大。η 越大，每一步改动越大。`)})+
   `</details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import torch.nn.functional as F

def seq_logp(logits, tokens, mask):
    # log p(response): the sum of its tokens' log-probs;
    # logits[:, t] predicts tokens[:, t]
    logp = logits.log_softmax(-1)
    logp = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # pi_*: policy log-probs; ref_*: frozen reference
    # log-probs; all of shape (batch,)
    ratio_w = pi_w - ref_w  # log pi(y_w) / pi_ref(y_w)
    ratio_l = pi_l - ref_l  # log pi(y_l) / pi_ref(y_l)
    loss = -F.logsigmoid(beta * (ratio_w - ratio_l)).mean()
    # implicit rewards, as TRL logs them
    reward_w = beta * ratio_w.detach()
    reward_l = beta * ratio_l.detach()
    return loss, reward_w, reward_l`;
function connectHTML(){
  const TRL='https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/';
  const fl=(p,x)=>`<a href="${TRL}${p}" target="_blank" rel="noopener"><code>trl/trainer/${p}</code></a>${x}`;
  return `<section class="wrap connect" aria-labelledby="dpo-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="dpo-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The DPO loss needs four log-probabilities per pair: the chosen and the rejected answer, each under the model and under the frozen reference.',`DPO 的 loss 每对数据只要四个 log 概率：选中的回答和落选的回答，各在当前模型和参考模型 ${M('\\pi_{\\text{ref}}')}（冻结的 SFT 模型）下算一次。`)}</p><pre class="code" id="dpo-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="dpo-prod">`+
   tx(`<li>${fl('dpo_trainer.py',': computes those four log-probabilities for each pair; with the default <code>loss_type=["sigmoid"]</code> the loss is this page’s '+M('-\\log\\sigma(\\text{reward margin})')+'.')}</li>`+
      `<li>${fl('dpo_config.py',': <code>beta</code> defaults to 0.1; <code>loss_type</code> also offers <code>"ipo"</code>, <code>"hinge"</code> and more.')}</li>`+
      `<li>Training logs: <code>rewards/chosen</code>, <code>rewards/rejected</code> and <code>rewards/margins</code> are this page’s implicit rewards and reward margin; a falling <code>logps/chosen</code> means the chosen answers are getting less likely too, which happens when the answers share weights (${J('step7','part 7')}).</li>`+
      `<li>Llama 3’s DPO used β = 0.1 and learning rate 1e-5, plus a negative log-likelihood term with weight 0.2 on the chosen answers (${LLAMA()}, §4.1.4).</li>`,
      `<li>${fl('dpo_trainer.py','：对每一对算出这四个 log 概率；默认 <code>loss_type=["sigmoid"]</code> 时，loss 就是本页的 '+M('-\\log\\sigma(\\text{奖励差})')+'。')}</li>`+
      `<li>${fl('dpo_config.py','：<code>beta</code> 默认 0.1；<code>loss_type</code> 还可以选 <code>"ipo"</code>、<code>"hinge"</code> 等。')}</li>`+
      `<li>训练日志：<code>rewards/chosen</code>、<code>rewards/rejected</code> 和 <code>rewards/margins</code> 就是本页的隐式奖励和奖励差；<code>logps/chosen</code> 往下走，说明选中的回答也变得不太可能了。回答之间共享权重时，就会这样（见${J('step7','第 7 部分')}）。</li>`+
      `<li>Llama 3 的 DPO 用 β = 0.1、学习率 1e-5，并在选中的回答上加了一项权重 0.2 的负对数似然 loss（${LLAMA()}，§4.1.4）。</li>`)+
   `</ul><p class="small muted">TRL v1.13.0</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noopener">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a><span>${tx('Rafailov et al., NeurIPS 2023','Rafailov 等，NeurIPS 2023')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2310.12036" target="_blank" rel="noopener">A General Theoretical Paradigm to Understand Learning from Human Preferences</a><span>${tx('Azar et al., 2023 (IPO): with one label per comparison, the model keeps moving toward the chosen answer whatever β is','Azar 等，2023（IPO）：每次比较只有一个标注时，不管 β 多大，模型都会一直往选中的回答偏')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2410.08847" target="_blank" rel="noopener">Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization</a><span>${tx('Razin et al., ICLR 2025','Razin 等，ICLR 2025')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2402.13228" target="_blank" rel="noopener">Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive</a><span>${tx('Pal et al., 2024 (DPO-Positive)','Pal 等，2024（DPO-Positive）')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="dpo-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="dpo-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="dpo-app m2" id="dpo-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="dpo-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- rail ---------- */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how the four answers’ probabilities change as DPO trains.','选一个场景，看看 DPO 训练时四个回答的概率怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="dpo-insight" aria-live="polite"></div>`+
   `<p class="small dpo-sc-more">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function insightHTML(r){
  const i=presetNow(), t=r.t, p0=r.T.pi[0], p=r.pi;
  if(t===0) return tx('<p><b>Step 0: nothing trained yet</b>; the model is still the SFT model.</p>','<p><b>第 0 步：还没训练</b>，模型还是 SFT 模型。</p>');
  /* the result: A's change is the one number, B's in words; the KL and the rest are on the metric cards (R6.20) */
  let h=tx(`<p><b>After ${stepsTxt(t)}, π(A) ${chg(p0[0],p[0])}, and π(B) ${moveW(p0[1],p[1])}.</b></p>`,
           `<p><b>训练 ${t} 步后，π(A) ${chg(p0[0],p[0])}，π(B) ${moveW(p0[1],p[1])}。</b></p>`);
  if(i<0) return h+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  const pw=r.wAB;
  if(S.mode==='shared'){const pk=peakOf(r.T,0,t), most=p[3]>0.5;
    h+=tx(`<p>With one shared weight, A ${pk.k<t?'rose at first and then fell':'kept rising'}, while D, the answer with the most detail, ${most?'took most of the probability':'gained the most'}. The reward margin between A and B still grew, so the loss still fell.</p>`,
          `<p>只有一个共享权重时，A ${pk.k<t?'先升后降':'一直在上升'}；细节最多的 D ${most?'拿走了大部分概率':'涨得最多'}。A 和 B 之间的奖励差仍在拉大，所以 loss 照样下降。</p>`);}
  else if(S.ds==='mixed'){const m=marginShown(r), done=Math.abs(m-LOG2)<0.01;   // the reward margin itself is on its card
    h+=tx(`<p>Two of three labels prefer A, so the loss is lowest when, under the implicit reward, A beats B two times out of three: at a reward margin of log 2 ≈ ${f(LOG2,3)}. ${done?'The reward margin is now close to that, so more steps change almost nothing.':(m<LOG2?'The reward margin is still below that and growing.':'The reward margin is already about there.')} A smaller β puts that balance point further from the SFT model.</p>`,
          `<p>三次标注里两次偏好 A，所以按隐式奖励算，A 三次里有两次胜过 B 时 loss 最小，也就是奖励差 = log 2 ≈ ${f(LOG2,3)} 时。${done?'现在的奖励差已经很接近，再训练几乎不变。':(m<LOG2?'现在的奖励差还没到，还在变大。':'现在的奖励差已经差不多到了。')}β 越小，这个平衡点离 SFT 模型越远。</p>`);}
  else if(S.ref==='s'){const k=passStep(S), k0=passStep({...S,ref:'b'});
    const when=k<0?tx(`here not within ${MAXS} steps, against step ${k0} in the initial setup`,`这里 ${MAXS} 步以内都没超过，初始设置是第 ${k0} 步`):tx(`step ${k} here, against step ${k0} in the initial setup`,`这里是第 ${k} 步，初始设置是第 ${k0} 步`);
    h+=tx(`<p>The SFT model rarely gave A, yet this pair still starts as a tie, as every pair does: DPO measures change from the SFT model, not its opinion. It just takes more steps for A to pass B: ${when}.</p>`,
          `<p>SFT 模型很少给出 A，可这一对照样从平局开始，和任何一对一样：DPO 衡量的是相对 SFT 模型的变化，而不是 SFT 模型的看法。只是要训练更多步，A 才会超过 B：${when}。</p>`);}
  else if(S.ds==='three'){const one=compute({...S,ds:'one'}), dA=p[0]-one.pi[0], dB=p[1]-one.pi[1];   // against the initial setup, in words
    const aw=Math.abs(dA)<0.05?tx('about the same as','差不多'):(dA<0?tx('lower than','低一些'):tx('higher than','高一些')), bw=dB>=0.05?tx('much more','多得多'):(dB>0.005?tx('a little more','多一些'):null);
    h+=tx(`<p>The loss is averaged over the three pairs, so each pair pushes a third as hard. A’s logit rises about as fast as with one pair, but the logits of B, C and D each fall a third as fast. So after the same ${stepsTxt(t)}, π(A) is ${aw} in the initial setup (only the pair A ≻ B)${bw?`, but B keeps ${bw} of its probability`:''}.</p>`,
          `<p>loss 是三对数据的平均，所以每一对只推三分之一的力。A 的 logit 上升得和只有一对时差不多快，B、C、D 的 logit 各自只下降三分之一。所以同样训练 ${t} 步，π(A) ${Math.abs(dA)<0.05?'和':'比'}初始设置（只有 A ≻ B 一对）${aw}${bw?`，B 却留下了${bw}的概率`:''}。</p>`);}
  else if(S.beta>=1){const low=+pw.toFixed(3)<0.1;   // the push weight itself is on its curve and in Show the math
    h+=tx(`<p>With β = ${bTxt(S.beta)} the push ${low?'has almost faded':'is fading'}: each step now moves the logits only ${relOf(pw)}% as far as the first one did.${low?' Training has slowed down close to the SFT model, but it keeps creeping on.':''}</p>`,
          `<p>β = ${bTxt(S.beta)} 时推力${low?'几乎消失了':'正在衰减'}：现在每一步对 logit 的改动只有第一步的 ${relOf(pw)}%。${low?'训练在 SFT 模型附近慢了下来，但还在缓慢前进。':''}</p>`);}
  else h+=tx(`<p>This pair has one label and no vote against it, so the loss is lowest only when B’s probability reaches 0: A keeps rising for any β, and a larger β only makes the push fade sooner.</p>`,
             `<p>这一对只有一个标注，没有反对票，所以 B 的概率要降到 0，loss 才最小：不管 β 多大，A 都会一直上升，β 越大只是推力衰减得越早。</p>`);
  return h;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, you see whether both conditions hold, and why.','提交后告诉你两个条件是否都满足，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('After 60 DPO steps, make the model give A at least half the time, with its KL from the SFT model at most 0.8 nats','训练 60 步 DPO 后，让模型至少一半的时候给出 A，且离 SFT 模型的 KL 不超过 0.8 nats')}</h3>`+
   `<p>${tx(`These stay fixed: before training the SFT model gives A ${pc(REFS.b[0])} of the time and B ${pc(REFS.b[1])}; the pair is A ≻ B; training runs 60 steps with step size 0.1. You can change only β:`,`这些保持不变：训练前 SFT 模型给出 A 的概率是 ${pc(REFS.b[0])}，B 是 ${pc(REFS.b[1])}；偏好对是 A ≻ B；训练 60 步，步长 0.1。你只能改 β：`)}</p>`+inBeta()+
   `<div class="goal mono" id="dpo-ch-a"><span>${tx('How often the model gives A after 60 steps','60 步后模型给出 A 的概率')}</span><span><b id="dpo-ch-av"></b>${tx(' (goal: at least 50%)','（目标：至少 50%）')}</span></div><div class="progress" id="dpo-ch-ap" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono" id="dpo-ch-k"><span>${tx('KL from the SFT model','离 SFT 模型的 KL')}</span><span><b id="dpo-ch-kv"></b>${tx(' nats (goal: at most 0.8)',' nats（目标：最多 0.8）')}</span></div><div class="progress" id="dpo-ch-kp" aria-hidden="true"><i></i></div>`+
   `<p class="small muted">${tx('KL measures how far the model has moved from the SFT model, in nats. The 50% and 0.8-nat thresholds are set for this challenge.','KL 衡量模型离 SFT 模型有多远，单位 nats。50% 和 0.8 nats 是这道题定的门槛。')}</p>`+
   submit+`<div id="dpo-ch-res" class="dpo-ch-res" aria-live="polite"></div></div>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const pts=BETAS.map(chPoint), me=pts.find(p=>p.b===S.beta), okp=pts.filter(p=>p.ok), best=okp.slice().sort((a,b)=>a.kl-b.kl)[0];
  let h;
  if(me.ok&&me===best) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: after 60 steps the model gives A ${pc(me.pA)} of the time, with KL ${f(me.kl,2)} nats.</b> And among the β values that work, β = ${bTxt(me.b)} keeps the model closest to the SFT model.</p>`,
                          `<p class="okline" style="margin-top:12px"><b>达成目标：训练 60 步后，模型给出 A 的概率是 ${pc(me.pA)}，KL ${f(me.kl,2)} nats。</b>而且在能达标的 β 里，β = ${bTxt(me.b)} 让模型离 SFT 模型最近。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: the model gives A ${pc(me.pA)} of the time, with KL ${f(me.kl,2)} nats.</b> You could stay closer to the SFT model: β = ${bTxt(best.b)} also meets both conditions, with a smaller KL.</p>`,
                      `<p class="okline" style="margin-top:12px"><b>达成目标：模型给出 A 的概率是 ${pc(me.pA)}，KL ${f(me.kl,2)} nats。</b>其实还可以离 SFT 模型更近：β = ${bTxt(best.b)} 也满足两个条件，而且 KL 更小。</p>`);
  else {   // the value that misses is the one number; the condition that holds is said in words (both are in the goal rows above)
    h=!me.okK?tx(`<p class="badline" style="margin-top:12px"><b>Not yet: the KL is ${f(me.kl,2)} nats, over the 0.8-nat limit</b>${me.okA?', although the model already gives A more than half the time':''}.</p><p class="small">Hint: in 60 steps the model drifted too far from the SFT model. Which setting decides how early the push fades?</p>`,
                `<p class="badline" style="margin-top:12px"><b>还没达到：KL 是 ${f(me.kl,2)} nats，超过了 0.8 nats 的上限</b>${me.okA?'，尽管模型给出 A 的概率已经超过一半':''}。</p><p class="small">提示：60 步里模型离 SFT 模型漂得太远了。哪个设置决定推力多早衰减？</p>`)
              :tx(`<p class="badline" style="margin-top:12px"><b>Not yet: the model gives A only ${pc(me.pA)} of the time</b>, although its KL is within the limit.</p><p class="small">Hint: the push faded before A reached 50%. Look at ${J('curve-wt','the push-weight curve')} for this β.</p>`,
                  `<p class="badline" style="margin-top:12px"><b>还没达到：模型给出 A 的概率只有 ${pc(me.pA)}</b>，尽管 KL 没有超过上限。</p><p class="small">提示：A 还没到 50%，推力就衰减了。看看这个 β 下的${J('curve-wt','推力权重曲线')}。</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;
  }
  if(!chReveal) return h;
  /* brief (the owner's review of batch 8, 2026-10-07): the SFT model before training (the baseline), the β values that meet both conditions,
     your pick and the initial β; a KL over the limit is marked on the rows that stay */
  const at=b=>pts.find(p=>p.b===b), sft=REFS.b[0], b0=0.1, kept=p=>p.ok||p===me||p.b===b0;
  const rows=[{label:tx('The SFT model before training','训练前的 SFT 模型'),value:sft*100,text:pc(sft),ok:false,tags:[{text:tx('baseline','基准')}]}].concat(pts.map(p=>({
    label:tx(`β = ${bTxt(p.b)}, KL ${f(p.kl,2)} nats`,`β = ${bTxt(p.b)}，KL ${f(p.kl,2)} nats`),value:p.pA*100,text:pc(p.pA),ok:p.ok,me:p===me,
    tags:(p.b===b0?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[]).concat(p.okK||!kept(p)?[]:[{text:tx('KL over 0.8','KL 超过 0.8')}])})));
  h+=A.ui.cmpChart({brief:true,title:tx('How often the model gives A after 60 DPO steps on A ≻ B (goal: at least 50%, with KL at most 0.8 nats)','在 A ≻ B 上训练 60 步 DPO 后，模型给出 A 的概率（目标：至少 50%，且 KL 不超过 0.8 nats）'),
    aria:tx('How often the model gives A after 60 DPO steps. ','训练 60 步 DPO 后模型给出 A 的概率：')+tx(`before training: ${pc(sft)}; `,`训练前：${pc(sft)}；`)+pts.filter(kept).map(p=>tx(`β ${bTxt(p.b)}: ${pc(p.pA)}, KL ${f(p.kl,2)}`,`β ${bTxt(p.b)}：${pc(p.pA)}，KL ${f(p.kl,2)}`)).join(tx('; ','；')),rows,goal:50,goalLabel:tx('goal: 50%','目标 50%'),max:100,
    hiddenText:n=>tx(`The other ${n===1?'β value misses':`${n} β values miss`} at least one condition and ${n===1?'is':'are'} not shown.`,`其余 ${n} 个 β 至少有一个条件不满足，图里没有列出。`),
    okText:tx('meets both conditions','两个条件都满足'),noText:tx('misses at least one','至少一个不满足'),goalText:tx('goal: A at least 50%','目标：A 至少 50%')});
  const lo=at(BETAS[0]), hi=at(BETAS[BETAS.length-1]);
  /* the reason in words: the push weights at step 60 are on the push-weight curve and in Show the math (R6.20) */
  h+=tx(`<p class="small">Why: in this lab the first step moves the logits by the same amount for every β, and each later step is scaled by the push weight ${M('\\sigma({-\\text{reward margin}})')}. By step 60 the push ${pushW(lo.w)} with β = ${bTxt(lo.b)} but ${pushW(hi.w)} with β = ${bTxt(hi.b)}. A larger β makes the push fade earlier and the model change less, so among the β values that still reach 50%, the largest keeps the model closest to the SFT model.</p>`,
        `<p class="small">原因：在这个 Lab 里，不管 β 取多少，第一步对 logit 的改动都一样大；之后每一步的改动都要乘以推力权重 ${M('\\sigma({-\\text{奖励差}})')}。到第 60 步，β = ${bTxt(lo.b)} 的推力${pushW(lo.w)}，β = ${bTxt(hi.b)} 的推力${pushW(hi.w)}。β 越大，推力衰减得越早，模型改动得越少，所以在还能达到 50% 的 β 里，最大的那个让模型离 SFT 模型最近。</p>`);
  h+=tx(`<p class="small muted">Note: this challenge scores one prompt with four answers and says nothing about other prompts. For a real run, compare checkpoints on held-out preference pairs. Whether a score gap between two checkpoints is real depends on its error bars and on a question-by-question comparison. The lab ${A.shell.labLink('eval-ci')} shows how.</p>`,
        `<p class="small muted">注意：这道题只看一个提问的四个回答，不能说明模型在别的提问上变好了。真实训练选 β 时，要在留出的偏好对上比较各个 checkpoint。两个 checkpoint 的分数差距是不是真的，要看误差范围，还要逐题配对比较，${A.shell.labLink('eval-ci')}这个 Lab 讲怎么做。</p>`);
  return h;
}

/* ---------- stage drawing ---------- */
/* the challenge verdict in words, naming the condition that fails (status badge and mini result bar) */
function chBadge(p){if(p.ok) return tx('Both conditions are met','两个条件都满足');
  if(!p.okA&&!p.okK) return tx('Neither condition is met','两个条件都没满足');
  return p.okK?tx('The model gives A under half the time','模型给出 A 的概率不到 50%'):tx('KL is above 0.8 nats','KL 超过 0.8 nats');}
function drawHead(r){
  const st=$('#dpo-st-title',app), sub=$('#dpo-st-sub',app);
  st.innerHTML=r.t===0?tx(`The SFT model before training (${betaSpan(S.beta)})`,`训练前的 SFT 模型（${betaSpan(S.beta)}）`)
                     :tx(`The model after ${r.t===1?'1 DPO step':`${r.t} DPO steps`} (${betaSpan(S.beta)})`,`训练 ${r.t} 步 DPO 后的模型（${betaSpan(S.beta)}）`);
  const p0=REFS[S.ref];
  sub.textContent=tx(`Preference data: ${L(DSDESC[S.ds])}. Before training, the SFT model gives A ${pc(p0[0])} of the time and B ${pc(p0[1])}.${S.mode==='shared'?' All answers share one weight w.':''}${S.eta!==0.1?` Step size η = ${S.eta}.`:''}`,
                     `偏好数据：${L(DSDESC[S.ds])}。训练前，SFT 模型给出 A 的概率是 ${pc(p0[0])}，给出 B 的概率是 ${pc(p0[1])}。${S.mode==='shared'?'所有回答共享一个权重 w。':''}${S.eta!==0.1?`步长 η = ${S.eta}。`:''}`);
  const el=$('#dpo-status',app);
  if(mode==='challenge'){const p=chPoint(S.beta); el.className='status '+(p.ok?'ok':'bad'); el.dataset.tip='dpo_tipGoal'; el.textContent=chBadge(p);}
  else {el.className='status '+(r.top===0?'ok':'bad'); el.dataset.tip='dpo_tipTop'; el.textContent=tx(`The model most often gives ${LET[r.top]} (${pc(r.pi[r.top])})`,`模型最常给出 ${LET[r.top]}（${pc(r.pi[r.top])}）`);}
  $$('[data-play]',app).forEach(b=>{b.textContent=playLabel(); b.setAttribute('aria-pressed',String(playing));});
  const tr=$('#dpo-transport',app), lock=mode==='challenge'||preLock();
  $$('button',tr).forEach(b=>{b.hidden=lock;}); const sp=$('.dpo-speed',tr); if(sp) sp.hidden=lock;
  $('#dpo-tnote',app).textContent=preLock()?tx('Training starts after you answer part 2.','在第 2 部分作答后开始训练。'):'';
}
function drawBars(r){
  const host=$('#dpo-bars',app), p0=r.T.pi[0], dz0=r.z.map((v,i)=>v-r.z0[i]), noLogit=mode==='guided'&&step<SIX('steps');   // "logit" is defined in part 3
  // shared weight: w × detail level from the w Show the math shows, as the observation does (0.397 and 0.794, R6.6)
  const dz=S.mode==='shared'?wRises(r):dz0;
  if(!host.children.length) host.innerHTML=LET.map((l,i)=>`<div class="dpo-row" data-i="${i}" tabindex="0"><div class="dpo-rh"><span class="dpo-let">${l}</span><span class="dpo-lab"></span><span class="dpo-role"></span><span class="dpo-val mono"></span></div><div class="dpo-track"><i class="dpo-fill"></i><i class="dpo-ghost"></i></div><div class="dpo-foot mono"></div></div>`).join('');
  host.classList.toggle('dpo-shared',S.mode==='shared');
  LET.forEach((l,i)=>{const row=host.children[i], cls=roleCls(i,r.pairs);
    row.className='dpo-row dpo-c-'+cls;
    row.querySelector('.dpo-lab').textContent=L(ANS[i].s);
    row.querySelector('.dpo-role').textContent=roleTxt(i,r.pairs);
    row.querySelector('.dpo-val').textContent=r.t===0?pc(r.pi[i]):tx(`${pc(r.pi[i])} (was ${pc(p0[i])})`,`${pc(r.pi[i])}（训练前 ${pc(p0[i])}）`);
    row.querySelector('.dpo-fill').style.width=(r.pi[i]*100).toFixed(2)+'%';
    row.querySelector('.dpo-ghost').style.width=(p0[i]*100).toFixed(2)+'%';
    // before training there is no logit change to report; the detail levels are in “Where the logits come from” (item 2)
    row.querySelector('.dpo-foot').textContent=noLogit||r.t===0?'':logitTxt(dz[i],S.mode==='shared'?3:2);
    const now=r.t===0?tx(`Before training the model gives it ${pc(p0[i],1)} of the time.`,`训练前，模型给出它的概率是 ${pc(p0[i],1)}。`):tx(`The model gives it ${pc(r.pi[i],1)} of the time now and gave it ${pc(p0[i],1)} before training.`,`模型现在给出它的概率是 ${pc(r.pi[i],1)}，训练前是 ${pc(p0[i],1)}。`);
    row.dataset.tipx=tx(`Answer ${l}: ${ANS[i].full.en} ${now}${cls==='o'?' It is in no preference pair, so the loss never uses it.':''}`,
                        `回答 ${l}：${ANS[i].full.zh}${now}${cls==='o'?'它不在任何偏好对里，loss 用不到它。':''}`);
    row.setAttribute('aria-label',tx(`Answer ${l}, ${ANS[i].s.en}, ${roleTxt(i,r.pairs)}: ${pc(r.pi[i])} now, ${pc(p0[i])} before training`,`回答 ${l}，${ANS[i].s.zh}，${roleTxt(i,r.pairs)}：现在 ${pc(r.pi[i])}，训练前 ${pc(p0[i])}`));});
}
/* Where the logits come from (item 2 of the owner's review of 2026-10-09): the picture of part 7. With separate logits the four logits
   are four unconnected boxes, and a step moves only A's and B's. With the shared weight one w feeds every logit through a line as thick as
   the answer's detail level, so pushing A up moves the others too. Shown in part 7 and whenever the shared weight is on; the lines carry
   no text, so nothing in the svg can cross its panel (H7). */
function drawWire(r){
  const el=$('#dpo-wire',app); if(!el) return;
  const sh=S.mode==='shared', on=sh||(mode==='guided'&&STEPS[step].id==='shared');
  el.hidden=!on; if(!on){el._h=''; return;}
  const h=wireHTML(r,sh);
  if(el._h!==h){el.innerHTML=h; el._h=h;}
}
function wireHTML(r,sh){
  const dz=sh?wRises(r):r.z.map((v,i)=>v-r.z0[i]), pre=mode==='guided'&&step<SIX('steps');
  const ch=d=>r.t===0||pre?'':(Math.abs(d)<5e-4?tx('logit unchanged','logit 没变'):`logit ${d>0?'+':MINUS}${f(Math.abs(d),sh?3:2)}`);
  const cell=i=>`<div class="dpo-wc dpo-c-${roleCls(i,r.pairs)}"><div class="dpo-wt"><span class="dpo-let">${LET[i]}</span>${sh?`<span class="dpo-wx">${tx('detail','细节')} ${f(DETAIL[i],1)}</span>`:''}</div><div class="dpo-wd mono">${ch(dz[i])}</div></div>`;
  const xs=LET.map((l,i)=>12.5+25*i);
  const top=sh?`<div class="dpo-wsrc"><span class="mono">${tx('shared weight','共享权重')} w = ${f(+(+r.w).toFixed(3),3)}</span></div><svg class="dpo-wsvg" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true">${xs.map((x,i)=>`<line x1="50" y1="0" x2="${x}" y2="24" stroke-width="${(1+1.5*DETAIL[i]).toFixed(2)}" vector-effect="non-scaling-stroke"/>`).join('')}</svg>`:'';
  const cap=sh?tx(`All four logits hang on one shared weight w. A has more detail than B, so DPO can widen the gap between them only by raising w. When w changes by Δw, each logit moves by ${M('Δw \\times \\text{detail level}')} (the thicker the line, the more detail). So B’s, C’s and D’s logits rise too, and D’s the most.`,
                  `四个 logit 都连在同一个共享权重 w 上。A 的细节比 B 多，DPO 要拉开 A 和 B，只能调大 w。w 每变化 Δw，每个 logit 就移动 ${M('Δw \\times \\text{细节程度}')}（线越粗，细节越多）。所以 B、C、D 的 logit 也跟着升，D 升得最多。`)
              :tx('Separate logits: the four logits are not connected. A step raises A’s logit and lowers B’s; C’s and D’s stay where they were.',
                  '各自的 logit：四个 logit 互不相连。每一步只升 A 的 logit、降 B 的，C 和 D 的 logit 不动。');
  return `<div class="dpo-blk-h">${tx('Where the logits come from','logit 从哪来')}</div>${top}<div class="dpo-wcells${sh?'':' dpo-wsep'}">${LET.map((l,i)=>cell(i)).join('')}</div><p class="dpo-wcap">${cap}</p>`;
}
/* R7.9 (polish round 5): on a wide desktop stage (dpo.css sets --dpo-hold on the chart there) the picture sits in the left column under
   the answers, and the chart is as tall as that column: switching the shared weight on (the option of part 7, the switch in Settings,
   the scenario “Closer to a real LLM”) made the stage 107 px taller in part 7 (the picture's shared-weight form is taller) and 230 px in
   Scenarios and Settings (the picture appears). There, the picture keeps the room of its shared-weight form from the start: in part 7
   under its separate-logit form, elsewhere unseen while the shared weight is off (as the note of ppo.js's --ppo-hold). The room is
   measured once per language and width by putting the shared-weight form in place for a moment (nothing is drawn in between); render
   calls this right after drawWire. Phones, narrower stages, the other parts and the challenge (separate logits, locked) are left as they were. */
function holdWire(r){
  const el=$('#dpo-wire',app), g=$('#dpo-graph',app); if(!el||!g) return;
  const on=(mode==='sandbox'||(mode==='guided'&&STEPS[step].id==='shared'))&&getComputedStyle(g).getPropertyValue('--dpo-hold').trim()==='1';
  if(!on){el.classList.remove('dpo-held'); if(el._hold){el.style.minHeight=''; el._hold=null;} return;}
  const key=[A.lang(),g.clientWidth].join('|');
  if(!el._hold||el._hold.key!==key){
    const hid=el.hidden, kh=el.innerHTML;
    el.hidden=false; el.classList.remove('dpo-held'); el.style.minHeight='';
    el.innerHTML=wireHTML(S.mode==='shared'?r:compute({...S,mode:'shared'}),true);
    const mh=Math.ceil(el.getBoundingClientRect().height);
    el.innerHTML=kh; el.hidden=hid; el._hold={key,mh};}
  if(el.hidden){el.hidden=false; el.innerHTML=''; el._h=''; el.classList.add('dpo-held');} else el.classList.remove('dpo-held');
  el.style.minHeight=el._hold.mh+'px';
}
function niceMax(v){if(v<=0) return 1; const e=Math.pow(10,Math.floor(Math.log10(v))), m=v/e; return (m<=1?1:m<=2?2:m<=2.5?2.5:m<=5?5:10)*e;}
function seriesFor(T,s,vw){
  if(vw==='prob') return LET.map((l,i)=>({k:'p'+i,lab:l,cls:roleCls(i,DATA[s.ds]),vals:T.pi.map(p=>p[i])}));
  if(vw==='kl') return [{k:'kl',lab:'KL',cls:'n',vals:T.kl}];
  if(vw==='loss') return [{k:'loss',lab:'loss',cls:'n',vals:T.loss}];
  return uniqPairs(DATA[s.ds]).map((u,j)=>({k:'w'+u.p.join(''),lab:pairTxt(u.p),cls:['w','l','o'][j]||'o',vals:T.wt.map(w=>w[u.i])}));   // one line style per pair, named in the key
}
function yRange(vw,sets){
  if(vw==='prob') return [0,1];
  let mx=0; sets.forEach(ss=>ss.forEach(s=>s.vals.forEach(v=>{if(v>mx) mx=v;})));
  if(vw==='wt') return [0,Math.max(0.5,Math.ceil(mx*10-1e-9)/10)];
  if(vw==='loss') return [0,Math.max(0.8,niceMax(mx))];
  return [0,niceMax(Math.max(mx,0.05))];
}
function fmtY(vw,v){return vw==='prob'?Math.round(v*100)+'%':fx(v,2);}   // one format per axis: 0.00, 0.50, 1.00
const KEYNAME=cls=>({w:tx('chosen','选中'),l:tx('rejected','落选'),o:tx('not labeled','未标注')}[cls]||'');
function drawCurve(r){
  const svg=$('#dpo-curve',app); if(!svg) return;
  const W=Math.max(220,Math.round(svg.getBoundingClientRect().width||svg.parentNode.clientWidth||600)), H=156, m={l:42,r:24,t:10,b:30};
  const T=r.T, t=r.t, cur=seriesFor(T,S,view), gs=ghost?seriesFor(traj(ghost),ghost,view):null;
  const [y0,y1]=yRange(view,gs?[cur,gs]:[cur]);
  const X=s=>m.l+(W-m.l-m.r)*s/MAXS, Y=v=>m.t+(H-m.t-m.b)*(1-(Math.min(y1,Math.max(y0,v))-y0)/(y1-y0));
  let h='';
  [0,0.5,1].forEach(q=>{const v=y0+(y1-y0)*q, y=Y(v).toFixed(1); h+=`<line class="dpo-grid" x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}"/><text class="dpo-ax" x="${m.l-6}" y="${(+y+3.5).toFixed(1)}" text-anchor="end">${fmtY(view,v)}</text>`;});
  [0,20,40,60,80,100].forEach(s=>{h+=`<text class="dpo-ax" x="${X(s).toFixed(1)}" y="${H-m.b+15}" text-anchor="middle">${s}</text>`;});
  h+=`<text class="dpo-ax" x="${W-m.r}" y="${H-2}" text-anchor="end">${tx('training steps','训练步数')}</text>`;
  let refLine=null;
  if(view==='loss'){refLine=LOG2; const y=Y(LOG2).toFixed(1); h+=`<line class="dpo-ref" x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}"/>`;}
  if(view==='wt'){refLine=0.5; const y=Y(0.5).toFixed(1); h+=`<line class="dpo-ref" x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}"/>`;}
  const path=(vals,upto)=>{let d=''; for(let s=0;s<=upto;s++) d+=(s?'L':'M')+X(s).toFixed(1)+','+Y(vals[s]).toFixed(1); return d;};
  const gm=gs?(gs.find(g=>g.cls==='w'||g.cls==='n')||gs[0]):null;
  if(gm) h+=`<path class="dpo-ln dpo-lg" d="${path(gm.vals,Math.min(t,MAXS))}"/>`;
  cur.forEach(s=>{h+=`<path class="dpo-ln dpo-l-${s.cls}" d="${path(s.vals,t)}"/>`;});
  const xt=X(t); h+=`<line class="dpo-now" x1="${xt.toFixed(1)}" x2="${xt.toFixed(1)}" y1="${m.t}" y2="${H-m.b}"/>`;
  // current values: dots, and labels kept inside the plot area without overlapping each other
  const top=m.t+7, bot=H-m.b-3, gap=11;
  const pts=cur.map(s=>({s,y:Y(s.vals[t])})).sort((a,b)=>a.y-b.y);
  let last=-1e9; pts.forEach(p=>{p.ly=Math.max(p.y+3.5,top,last+gap); last=p.ly;});
  const over=pts.length?pts[pts.length-1].ly-bot:0;
  if(over>0){let lim=bot; for(let k=pts.length-1;k>=0;k--){pts[k].ly=Math.min(pts[k].ly,lim); lim=pts[k].ly-gap;}}
  const right=xt>W-m.r-40;
  pts.forEach(p=>{h+=`<circle class="dpo-dot dpo-d-${p.s.cls}" cx="${xt.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.2"/>`;
    if(view==='prob'||view==='wt') h+=`<text class="dpo-lbl" x="${(right?xt-7:xt+7).toFixed(1)}" y="${p.ly.toFixed(1)}" text-anchor="${right?'end':'start'}">${esc(p.s.lab)}</text>`;});
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('height',H); svg.innerHTML=h;
  const titles={prob:tx('How often the model gives each answer, over training','训练过程中，模型给出每个回答的概率'),kl:tx('How far the model has moved from the SFT model (KL, in nats)','模型离 SFT 模型有多远（KL，单位 nats）'),
    loss:tx('The DPO loss at each step','每一步的 DPO loss'),wt:tx('How far does each step still push A and B? (push weight)','每一步还把 A、B 推多少？（推力权重）')};
  $('#dpo-curve-title',app).textContent=titles[view];
  svg.setAttribute('aria-label',titles[view]+tx('. ','。')+cur.map(s=>`${s.lab}: ${fmtY(view,s.vals[t])}`).join(tx(', ','，')));
  // the key names every line style drawn: the series, the previous setting, the current step and any reference line
  const KS=cls=>`<svg class="dpo-ks" width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line class="${cls}" x1="2" y1="5" x2="20" y2="5"/></svg>`;
  let key='';
  cur.forEach(s=>{const lab=view==='prob'?`${s.lab} ${KEYNAME(s.cls)}`:s.lab; key+=`<span class="dpo-key-${s.cls}">${KS('dpo-ln dpo-l-'+s.cls)}${esc(lab)}</span>`;});
  if(gm) key+=`<span class="dpo-key-g" data-tip="dpo_tipGhost" tabindex="0">${KS('dpo-ln dpo-lg')}${esc(ghostLabel(view==='prob'?'A':(view==='wt'&&gs.length>1?gm.lab:'')))}</span>`;
  key+=`<span class="dpo-key-now" data-tip="dpo_tipNow" tabindex="0"><svg class="dpo-ks" width="10" height="14" viewBox="0 0 10 14" aria-hidden="true"><line class="dpo-now" x1="5" y1="0" x2="5" y2="14"/></svg>${tx('current step','当前训练步')}</span>`;
  if(refLine!=null) key+=`<span class="dpo-key-ref" data-tip="dpo_tipRefLine" tabindex="0">${KS('dpo-ref')}${view==='loss'?tx('value at step 0: log 2 ≈ 0.693','第 0 步的值：log 2 ≈ 0.693'):tx('value at step 0: 0.50','第 0 步的值：0.50')}</span>`;
  $('#dpo-curve-key',app).innerHTML=key;
  const vt={prob:'dpo_vProb',kl:'dpo_vKL',loss:'dpo_vLoss',wt:'dpo_vWt'}[view]; $('#dpo-curve-title',app).dataset.tip=vt;
}
/* the previous setting, named by what differs from now; `who` names the line when the chart has several */
function ghostLabel(who){if(!ghost) return '';
  const d=CFG.filter(k=>ghost[k]!==S[k]).map(k=>{
    if(k==='beta') return `β = ${bTxt(ghost.beta)}`; if(k==='eta') return `η = ${ghost.eta}`;
    if(k==='ds') return L(DSNAME[ghost.ds]); if(k==='ref') return tx(`SFT model ${refName(ghost.ref).toLowerCase()}`,`SFT 模型${refName(ghost.ref)}`); return L(MODENAME[ghost.mode]);});
  const what=d.length?d.join(tx(', ','、')):tx('same setup','同样设置');
  return who?tx(`${who} with the previous setting (${what})`,`上一个设置（${what}）下的 ${who}`):tx(`the previous setting (${what})`,`上一个设置（${what}）`);}
/* the reward margin card shows the difference of the two implicit rewards it lists, each rounded to 3 decimals (R6.6) */
const r3s=v=>+(+v.toFixed(3)).toFixed(3);
const marginShown=r=>+(r3s(r.rhat[0])-r3s(r.rhat[1])).toFixed(3);
/* ▲▼ badges: only after the learner's own change (a setting, a step button, the slider, a scenario), computed by the kit from the
   numbers as shown (dp = the decimals on the card), cleared when that number did not change; entering a part, Back and Next, a mode
   switch, Show me, playback and anything else the lab sets never show one (R6.6, R6.10) */
let userAct=false; const shownPrev={};
const numOf=v=>{const m=String(v).replace(MINUS,'-').match(/^[<>]?([+-]?\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).match(/\.(\d+)/); return m?m[1].length:0;};
const MET=[
 {k:'pA',tip:'dpo_mA',label:()=>tx('π(A), the chosen answer','π(A)，选中的回答'),v:r=>pc(r.pi[0]),pts:true,sub:r=>tx(`${pc(r.T.pi[0][0])} before training`,`训练前是 ${pc(r.T.pi[0][0])}`)},
 {k:'pB',tip:'dpo_mB',label:()=>tx('π(B), the rejected answer','π(B)，落选的回答'),v:r=>pc(r.pi[1]),pts:true,sub:r=>tx(`${pc(r.T.pi[0][1])} before training`,`训练前是 ${pc(r.T.pi[0][1])}`)},
 {k:'margin',tip:'dpo_mMargin',label:()=>tx('Reward margin','奖励差'),v:r=>sg(marginShown(r),3),sub:r=>tx(`A’s implicit reward ${sg(r3s(r.rhat[0]),3)}, B’s ${sg(r3s(r.rhat[1]),3)}`,`A 的隐式奖励 ${sg(r3s(r.rhat[0]),3)}，B 的 ${sg(r3s(r.rhat[1]),3)}`)},
 {k:'loss',tip:'dpo_mLoss',label:()=>tx('DPO loss','DPO loss'),v:r=>f(r.loss,3),sub:r=>tx(`${f(r.T.loss[0],3)} at step 0`,`第 0 步是 ${f(r.T.loss[0],3)}`)},
 {k:'kl',tip:'dpo_mKL',label:()=>tx('KL from the SFT model','离 SFT 模型的 KL'),v:r=>klS(r.kl),sub:()=>tx('in nats; 0 before training','单位 nats，训练前是 0')}
];
function deltaOf(m,a,b){const x=numOf(a), y=numOf(b); if(!isFinite(x)||!isFinite(y)||/[<>]/.test(a+b)) return '';
  return A.fmt.delta(x,y,{pts:!!m.pts,dp:Math.max(decOf(a),decOf(b))})||'';}
function drawMetrics(r){
  const host=$('#dpo-metrics',app);
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" data-tip="${m.tip}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), v=m.v(r), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(r);
    const dt=userAct&&shownPrev[m.k]!=null?deltaOf(m,shownPrev[m.k],v):'';
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=v;});
}
/* Show the math (H2, the owner's review of 2026-10-09: "展开公式里面的东西太丑陋了"): a few short blocks, one idea each. Each block has a
   lead line that names the quantity in words, the formula typeset, the reader's numbers under it and the result. Every number redoes from
   the numbers shown before it (R6.6, tests/dpo_numbers.py reads the rows by data-k and data-r). */
function mathHTML(r){
  const c=r.c, b=S.beta, n=r.pairs.length, Dm=mdp(b), B=T_('beta','β = '+bTxt(b)), Tt=T_('steps',stepsTxt(r.t));
  const RM=TX('reward margin','奖励差'), PW=TX('push weight','推力权重'), RHAT=i=>r6(b*c.l[i]);
  const blk=(k,lead,body,note)=>`<div class="dpo-mb" data-k="${k}"><p class="dpo-ml">${lead}</p>${body}${note?`<p class="dpo-mn">${note}</p>`:''}</div>`;
  const PI=i=>c.pd[i].v.toFixed(c.pd[i].d)+'%', RF=i=>rpc(c.refd[i]), PD=i=>(c.pd[i].v/100).toFixed(c.pd[i].d+2);
  const inPairs=LET.map((l,i)=>r.pairs.some(([a,cc])=>a===i||cc===i)), U=uniqPairs(r.pairs), PR=([a,cc])=>`${LET[a]} ≻ ${LET[cc]}`;
  const rows=[];
  rows.push(`<div class="dpo-mh">${tx('Now: ','当前：')}${Tt}${tx(', ','，')}${B}${tx(', ','，')}${T_('ds',tx(`pairs ${L(DSDESC[S.ds])}`,`偏好对 ${L(DSDESC[S.ds])}`))}${tx(', ','，')}${T_('ref',tx(`SFT model before training (${refDesc(S.ref)})`,`训练前的 SFT 模型（${refDesc(S.ref)}）`))}</div>`);
  // 1. log-ratios
  rows.push(blk('lr',tx(`${M('\\pi(y)')} is how often the model gives answer y, and ${M('\\pi_{\\text{ref}}(y)')} how often the SFT model gave it before training. An answer’s <b>log-ratio</b> says how much the first has grown over the second (log is the natural log).`,
                        `${M('\\pi(y)')} 是模型给出回答 y 的概率，${M('\\pi_{\\text{ref}}(y)')} 是训练前的 SFT 模型给出它的概率。一个回答的 <b>log 比值</b>，看前者比后者高了多少（log 是自然对数）。`),
    EQ([[`${LR()}(y)`,'=','\\log\\frac{\\pi(y)}{\\pi_{\\text{ref}}(y)}','','']].concat(LET.map((l,i)=>[`${LR()}(${l})`,'=',`\\log\\frac{${PI(i)}}{${RF(i)}}`,'=',nT(c.l[i],5)])))));
  // 2. implicit rewards of the answers in a pair
  rows.push(blk('r',tx('An answer’s <b>implicit reward</b> is β times its log-ratio.','一个回答的<b>隐式奖励</b>，是 β 乘以它的 log 比值。'),
    EQ([['\\hat{r}(y)','=','\\beta','\\times',`${LR()}(y)`,'','']].concat(LET.map((l,i)=>inPairs[i]?[`\\hat{r}_${l}`,'=',bTxt(b),'\\times',pT(c.l[i],5),'=',nT(RHAT(i),Dm)]:null).filter(Boolean)))));
  // 3. reward margins
  if(n===1) rows.push(blk('m',tx('The <b>reward margin</b> is A’s implicit reward minus B’s.','<b>奖励差</b>是 A 的隐式奖励减去 B 的。'),
    EQ([[RM,'=','\\hat{r}_A','-','\\hat{r}_B'],['','=',nT(c.rA,Dm),'-',pT(c.rB,Dm)],['','=',nT(c.mAB,Dm)]])));
  // several pairs: m names the margin (the lead line says what it is) and each pair breaks at its second “=”, so a phone needs no scrolling
  else rows.push(blk('m',tx('Each pair’s <b>reward margin</b> m is the chosen answer’s implicit reward minus the rejected one’s.','每一对的<b>奖励差</b> m，是选中回答的隐式奖励减落选回答的。'),
    EQ([].concat(...U.map(u=>{const [a,cc]=u.p, cp=c.pairs[u.i]; return [[`m_{${PR(u.p)}}`,'=',nT(RHAT(a),Dm),'-',pT(RHAT(cc),Dm)],Object.assign(['','=',nT(cp.m,Dm)],{cont:1})];})))));
  // 4. the loss
  const SIG=FX('','\\sigma(x) = \\frac{1}{1 + e^{-x}}');
  if(n===1) rows.push(blk('loss',tx('A pair’s <b>DPO loss</b> comes from its reward margin. σ squeezes any number into the range from 0 to 1.','一对数据的 <b>DPO loss</b> 由奖励差算出。σ 把任意一个数压到 0 和 1 之间。'),
    SIG+EQ([['L','=',`-\\log\\sigma(${RM})`],['','=',`-\\log\\sigma(${nT(c.mAB,Dm)})`],['','=',fx(c.loss,3)]])));
  else {const pl=U.map(u=>({u,v:softplus(-c.pairs[u.i].m)}));
    rows.push(blk('loss',tx(`The <b>DPO loss</b> is the mean over the ${n} pairs of each pair’s ${M('-\\log\\sigma(\\text{reward margin})')}. σ squeezes any number into the range from 0 to 1.`,`<b>DPO loss</b> 是 ${n} 对数据的 ${M('-\\log\\sigma(\\text{奖励差})')} 的平均。σ 把任意一个数压到 0 和 1 之间。`),
      SIG+EQ(pl.map(x=>[`L_{${PR(x.u.p)}}`,'=',`-\\log\\sigma(${nT(c.pairs[x.u.i].m,Dm)})`,'=',fx(x.v,4)]).concat([['L','=',`\\frac{${pl.map(x=>(x.u.n>1?`${x.u.n} \\times `:'')+fx(x.v,4)).join(' + ')}}{${n}}`,'=',fx(c.loss,3)]]))));}
  // 5. push weight and the next step
  const pwl=n>1?`${PW}_{A ≻ B}`:PW;
  rows.push(blk('pw',tx(`The <b>push weight</b> sets how far the next step pushes${n>1?' the pair A ≻ B':''}. The further A leads, the smaller it is.`,`<b>推力权重</b>决定下一步${n>1?'对 A ≻ B 这一对':''}推多远。A 领先得越多，它越小。`),
    EQ([[pwl,'=',`\\sigma({-${RM}})`],['','=',`\\sigma(${nT(-c.mAB,Dm)})`],['','=',fx(c.pw3,3)]])));
  if(S.mode==='shared'){const wd=r3(r.w), gd=r4(r.g[0]), nx=wd-(S.eta/b)*gd;   // computed from the shown w and gradient
    rows.push(blk('next',tx('The <b>next step</b> changes only the shared weight w, in the direction that lowers the loss.','<b>下一步</b>只改共享权重 w，朝着让 loss 变小的方向移动。'),
      EQ([[`w_{${TX('new','新')}}`,'=','w','-','\\frac{\\eta}{\\beta}','\\times','\\frac{∂L}{∂w}'],['','=',fx(wd,3),'-',`\\frac{${S.eta}}{${bTxt(b)}}`,'\\times',pT(gd,4)],['','=',fx(nx,4)]]),
      tx(`Then every logit moves by ${M('Δw \\times \\text{detail level}')}.`,`随后每个 logit 都移动 ${M('Δw \\times \\text{细节程度}')}。`)));}
  else if(n===1){const dA=String(+(S.eta*c.pw3).toFixed(8));
    rows.push(blk('next',tx('In the <b>next step</b> A’s logit rises and B’s falls, both by the step size η times the push weight.','<b>下一步</b>，A 的 logit 上升、B 的下降，幅度都是步长 η 乘以推力权重。'),
      EQ([['Δ\\text{logit}','=','\\eta','\\times',PW],['','=',String(S.eta),'\\times',fx(c.pw3,3)],['','=',dA]]),
      tx('C and D are in no pair. With separate logits their gradient is exactly 0, so their logits stay.','C 和 D 不在这一对里。每个回答有自己的 logit 时，它们的梯度正好是 0，logit 不动。')));}
  else {const g=r.g;
    rows.push(blk('next',tx(`In the <b>next step</b> each logit moves ${M('\\eta / \\beta')} times its gradient, in the direction that lowers the loss.`,`<b>下一步</b>，每个 logit 朝着让 loss 变小的方向，移动 ${M('\\eta / \\beta')} 乘以它的梯度。`),
      EQ([['Δ\\text{logit}','=','-\\frac{\\eta}{\\beta} \\times \\frac{∂L}{∂\\text{logit}}']].concat(LET.map((l,k)=>[`Δ\\text{logit}_${l}`,'=',sT(-(S.eta/b)*g[k],4)])))));}
  // 6. KL
  rows.push(blk('kl',tx('<b>KL</b> measures how far the model has moved from the SFT model. Multiply each answer’s π by its log-ratio and add them up; the unit is nats.','<b>KL</b> 衡量模型离 SFT 模型有多远。把每个回答的 π 乘以它的 log 比值，再加起来，单位是 nats。'),
    EQ([['\\text{KL}','=',`\\sum_y \\pi(y) \\times ${LR()}(y)`]].concat(LET.map((l,i)=>['',i?'+':'=',PD(i),'\\times',pT(c.l[i],5)]),[['','=',`${fx(c.kl,2)} \\, \\text{nats}`]]))));
  // 7. where the loss comes from
  const D=[
   [tx('RLHF objective','RLHF 的目标'),'\\max_{\\pi} \\, \\mathrm{E}_{y ∼ \\pi}[r(y)] - \\beta \\, \\text{KL}(\\pi \\,‖\\, \\pi_{\\text{ref}})',tx('r(y) is the reward model’s score for answer y.','r(y) 是奖励模型给回答 y 的打分。')],
   [tx('Its best policy','它的最优策略'),'\\pi^*(y) = \\frac{\\pi_{\\text{ref}}(y) \\exp(r(y) / \\beta)}{Z}',tx('Z sums over every answer to the prompt, so that the probabilities add up to 1.','Z 对这个提问的所有回答求和，让概率加起来是 1。')],
   [tx('Solve for the reward','解出奖励'),'r(y) = \\beta \\log\\frac{\\pi^*(y)}{\\pi_{\\text{ref}}(y)} + \\beta \\log Z',''],
   [tx('Bradley–Terry preference model','Bradley–Terry 偏好模型'),'P(A ≻ B) = \\sigma(r(A) - r(B))',tx(`A and B share Z, so ${M('\\beta \\log Z')} cancels in the difference.`,`A 和 B 共用 Z，所以相减时 ${M('\\beta \\log Z')} 消掉了。`)],
   [tx('DPO loss (maximum likelihood of the labels)','DPO loss（标注的最大似然）'),'L = -\\log\\sigma(\\hat{r}(A) - \\hat{r}(B))',tx(`With the model being trained in place of ${M('\\pi^*')}, the reward becomes the implicit reward ${M('\\hat{r}(y) = \\beta \\log(\\pi(y) / \\pi_{\\text{ref}}(y))')}.`,`把 ${M('\\pi^*')} 换成正在训练的模型，奖励就成了隐式奖励 ${M('\\hat{r}(y) = \\beta \\log(\\pi(y) / \\pi_{\\text{ref}}(y))')}。`)],
   [tx('Its gradient','它的梯度'),['∇L =',[`-\\beta \\times ${PW}`,'\\times (∇\\log\\pi(A) - ∇\\log\\pi(B))']],tx(`Since ${M('\\log\\pi(y) = \\text{logit}(y) - \\log\\sum e^{\\text{logit}}')}, the sum term cancels, so with separate logits only A’s and B’s logits get a gradient.`,`因为 ${M('\\log\\pi(y) = \\text{logit}(y) - \\log\\sum e^{\\text{logit}}')}，求和那一项相减时消掉，所以用各自的 logit 时，只有 A 和 B 的 logit 有梯度。`)]];
  rows.push(`<div class="dpo-mb" data-k="derive"><p class="dpo-ml">${tx('<b>Where the loss comes from.</b> The DPO paper derives it from the RLHF objective, step by step.','<b>这个 loss 从哪来。</b>DPO 论文从 RLHF 的目标出发，一步步推出它。')}</p><ol class="dpo-deriv">${D.map(([k,v,e])=>`<li><span class="dpo-dk">${k}</span>${Array.isArray(v)?FXW(v[0],v[1]):FX('',v)}${e?`<span class="dpo-de">${e}</span>`:''}</li>`).join('')}</ol>`+
    `<p class="small muted">${tx('Equations 1 and 3–7 and the gradient in Section 4 of','公式 1、3–7 和第 4 节的梯度，见')} ${RAF()}${tx('.','。')}</p></div>`);
  return rows.join('');
}

/* ---------- controls, rail sync ---------- */
function syncControls(r){
  $$('.seg-ctl[data-key]',app).forEach(g=>{const k=g.dataset.key; if(k==='view') A.ui.setPressed(g,view); else if(k==='speed') A.ui.setPressed(g,speed); else if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
  $$('input[data-key="steps"]',app).forEach(i=>{if(+i.value!==S.steps) i.value=S.steps; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=stepsTxt(S.steps);});
  const lock=mode==='challenge', pl=preLock(), c=$('#dpo-controls',app); c.classList.toggle('locked',lock);
  const sw=$('#dpo-steps-wrap',app); if(sw) sw.hidden=pl&&!lock;
  const ln=$('#dpo-lock',app); ln.hidden=!lock&&!pl;
  if(lock) ln.textContent=tx(`In the challenge you can change only β. The rest is fixed: the pair A ≻ B, the SFT model, which gives A ${pc(REFS.b[0])} of the time and B ${pc(REFS.b[1])} before training, separate logits, step size 0.1 and 60 steps.`,
                             `挑战中只能改 β。其余设置固定：偏好对 A ≻ B，训练前 SFT 模型给出 A 的概率是 ${pc(REFS.b[0])}、B 是 ${pc(REFS.b[1])}，每个回答有自己的 logit，步长 0.1，训练 60 步。`);
  else if(pl) ln.textContent=tx('The training steps open after you answer part 2.','在第 2 部分作答后，才能改训练步数。');
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(r){
  const rail=$('#dpo-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect(r)) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]);
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(r);
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again&&!playing) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const live=$('#dpo-pw-live',rail); if(live) live.textContent=r.t===0?tx('At step 0 the push weight is 0.500.','第 0 步的推力权重是 0.500。'):tx(`At step ${r.t} the push weight is ${pw3(r.wAB)} (0.500 at step 0).`,`第 ${r.t} 步的推力权重是 ${pw3(r.wAB)}（第 0 步是 0.500）。`);
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#dpo-insight',rail); if(ins){const i=presetNow(), html=`<div class="eyebrow">${i>=0?esc(L(PRESETS[i].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r); if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again&&!playing) A.ui.obsBump(ins);}}
  } else {
    const p=chPoint(S.beta), av=$('#dpo-ch-av',rail); if(!av) return;
    av.textContent=pc(p.pA); $('#dpo-ch-kv',rail).textContent=f(p.kl,2);
    $('#dpo-ch-ap i',rail).style.width=Math.min(100,p.pA/CH_PA*100)+'%'; $('#dpo-ch-a',rail).classList.toggle('met',p.okA); $('#dpo-ch-ap',rail).classList.toggle('met',p.okA);
    $('#dpo-ch-kp i',rail).style.width=Math.min(100,p.kl/CH_KL*100)+'%'; $('#dpo-ch-k',rail).classList.toggle('met',p.okK); $('#dpo-ch-kp',rail).classList.toggle('met',p.okK);
    $('#dpo-ch-k',rail).classList.toggle('dpo-over',!p.okK); $('#dpo-ch-kp',rail).classList.toggle('dpo-over',!p.okK);
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#dpo-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function renderRail(){const rail=$('#dpo-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';   // the next lab in every mode (R3.11)
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}   // the challenge's card, then the capability layer, then the next lab

/* ---------- render ---------- */
function render(){
  if(!app) return;
  /* R7.9 (polish round 5): where the learner changes β (part 6, Settings, the challenge) the key can also name the previous setting; on a
     wide desktop stage the curve's column is half as wide, and in English the probabilities view's key then needs four lines. dpo.css
     keeps room for them there from the start, so switching the view or β does not move the stage */
  app.classList.toggle('dpo-keyhold',mode==='sandbox'||mode==='challenge'||(mode==='guided'&&STEPS[step].id==='beta'));
  const r=compute(S);
  drawHead(r); drawBars(r); drawWire(r); holdWire(r); drawCurve(r); drawMetrics(r);
  const mb=$('#dpo-mathbody',app), mh=mathHTML(r); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(r); updateRail(r);
  A.hash.write(token()); A.dock.refresh(); announce(r);
}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return;
  A.announce(tx(`After ${stepsTxt(r.t)} with β = ${bTxt(S.beta)}: A ${pc(r.pi[0])}, B ${pc(r.pi[1])}, KL ${klS(r.kl)} nats.`,`训练 ${r.t} 步后（β = ${bTxt(S.beta)}）：A ${pc(r.pi[0])}，B ${pc(r.pi[1])}，KL ${klS(r.kl)} nats。`));},400);}
const cfgOf=s=>{const o={}; CFG.forEach(k=>{o[k]=s[k];}); return o;};
function setState(p){
  const before=cfgOf(S);
  if(preLock()&&p.steps!=null) p={...p,steps:0};   // no training before the prediction (R3.8)
  Object.assign(S,p);
  if(mode==='challenge'){const b=S.beta; Object.assign(S,CH); S.beta=b;}
  if(!BETAS.includes(S.beta)) S.beta=0.1;
  S.steps=Math.max(0,Math.min(MAXS,Math.round(S.steps)));
  if(CFG.some(k=>before[k]!==S[k])) ghost=before;
  if(mode==='guided'&&STEPS[step].id==='beta'&&p.beta!=null) triedB.add(S.beta);
  render();
}
function setView(v){view=v; render();}
function enterStep(i){
  stopPlay(); wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step];
  if(s.enter){Object.assign(S,s.enter()); ghost=null;}
  if(s.id==='beta') triedB=new Set([S.beta]);
  if(s.view) view=s.view;
  const st=$('#dpo-stage',app); if(st) st.scrollTop=0;           // the chart a part talks about starts in view
  renderRail();
  if(s.target){const el=$('#dpo-ic-'+s.target,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#dpo-rail .step.current',app)||$('#dpo-rail',app));}
/* keepStage (CONTRACT §11, R3.2, R9.7; as labs/moe.js; polish round 5): after Next, Back or a part's title the kit's reveal scrolls the
   new card's top up under the top bar; near the end of the bench that pushes the sticky stage up with it (at 1440 × 900, entering part 7
   from part 6: the stage's top at −213 px in Chinese, part 8 at −35 px; in English part 6 at −271 px and part 7 at −55 px). Once the page
   stops scrolling this pulls the stage back under the top bar without pushing the card's top off the screen, and does nothing once the
   reader has scrolled the card away */
let stageT=0;
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#dpo-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#dpo-rail .step.current',app)||$('#dpo-rail',app); if(!card) return;
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* the curve is drawn for its own width: on a stage at least 736 px wide it sits beside the answers (dpo.css), and the stage changes width
   without a window resize when the page puts the step list beside it or leaves the primer, so it is drawn again when it does (as
   labs/ppo.js) */
let stRO=null, stW=0;
function watchStage(){unwatchStage(); const st=app&&$('#dpo-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w; drawCurve(compute(S));}));
  stRO.observe(st);}
function unwatchStage(){if(stRO){stRO.disconnect(); stRO=null;}}
/* the result of a part, the verdict on the prediction included, comes into view when it first appears after Show me, Play, a step
   button or an option: scroll just enough to show all of it, never pushing its top under the top bar and the mini result bar */
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.bottom<=innerHeight-8) return;
  const dy=Math.min(r.bottom-innerHeight+16,r.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(app&&mode==='guided') revealBox($('#dpo-rail [data-obs]',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return; stopPlay();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,...CH,beta:0.1}; view='prob'; ghost=null;}
  A.shell.setMode(app,m,DESCS());
  const st=$('#dpo-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
function applyPreset(i){stopPlay(); const p=PRESETS[i]; ghost=null; Object.assign(S,p.p,{steps:p.steps}); byLearner(render); requestAnimationFrame(()=>{if(app) revealBox($('#dpo-insight',app));});}
/* a change the learner made: its metric cards may show a ▲▼ badge */
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
const token=(s=S)=>[s.ds,s.ref,s.mode,s.beta,s.eta,s.steps].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==6) return null;
  const [ds,ref,md,b,e,st]=p, o={ds,ref,mode:md,beta:+b,eta:+e,steps:+st};
  if(!DATA[ds]||!REFS[ref]||!MODENAME[md]||!BETAS.includes(o.beta)||!ETAS.includes(o.eta)||!Number.isInteger(o.steps)||o.steps<0||o.steps>MAXS) return null;
  return o;
}

/* ---------- training playback: advances by elapsed time × speed (R3.9) ---------- */
function stopPlay(){playing=false; if(raf){cancelAnimationFrame(raf); raf=0;} if(app) $$('[data-play]',app).forEach(b=>{b.textContent=playLabel(); b.setAttribute('aria-pressed','false');});}
function playTo(target,rate){
  target=Math.max(0,Math.min(MAXS,target));
  if(target<=S.steps||A.reduceMotion()){setState({steps:target}); if(wantObs){revealSoon(); wantObs=false;} return;}
  playing=true; playTarget=target; playRate=rate||0; lastTs=0; acc=0;
  const tick=ts=>{if(!playing||!app){raf=0; return;}
    if(lastTs){acc+=Math.min(0.1,(ts-lastTs)/1000)*(playRate||SPEEDS[speed]); const n=Math.floor(acc);   // a long pause (hidden tab) does not jump ahead
      if(n>0){acc-=n; const v=Math.min(playTarget,S.steps+n); Object.assign(S,{steps:v}); render(); if(v>=playTarget){stopPlay(); render(); if(wantObs){revealSoon(); wantObs=false;} return;}}}
    lastTs=ts; raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick); render();
}
function setSpeed(v){if(!SPEEDS[v]) return; speed=v; if(app) $$('.seg-ctl[data-key="speed"]',app).forEach(g=>A.ui.setPressed(g,speed));}
function act(a,btn){
  if((a==='play'&&!playing||a==='step1'||a==='step10')&&S.steps>=MAXS){A.tip.show(btn,`<div>${tx(`This lab trains for at most ${MAXS} steps, and you are there. Drag the training steps back, or press “Back to step 0”.`,`这个 Lab 最多训练 ${MAXS} 步，已经到头了。把训练步数往回拖，或点“回到第 0 步”。`)}</div>`); return;}
  if(a==='play'){if(playing){stopPlay(); render(); revealSoon(); wantObs=false;} else {wantObs=true; playTo(MAXS);} return;}
  if(a==='step1'){stopPlay(); wantObs=true; byLearner(()=>setState({steps:S.steps+1})); revealSoon(); wantObs=false; return;}
  if(a==='step10'){stopPlay(); wantObs=true; byLearner(()=>setState({steps:S.steps+10})); revealSoon(); wantObs=false; return;}
  if(a==='reset0'){stopPlay(); byLearner(()=>setState({steps:0})); return;}
  if(a==='reset'){stopPlay(); byLearner(()=>{if(mode==='challenge') setState({beta:0.1}); else {ghost=null; Object.assign(S,DEFAULT); render();}}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(S.beta).ok) chReveal=true; renderRail(); const res=$('#dpo-ch-res',app); if(res) res.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#dpo-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function showMe(p){wantObs=true; if(p.steps!=null&&p.steps>S.steps&&!A.reduceMotion()) playTo(p.steps,SPEEDS.fast); else {if(p.steps!=null) setState({steps:p.steps}); else setState(p); revealSoon(); wantObs=false;}}   // "Show me" may play fast (R3.9)

/* ---------- interview drill ---------- */
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#dpo-drill-q',app), a=$('#dpo-drill-a',app);
  q.innerHTML=tx(`One preference pair, with the sum of each answer’s token log-probabilities. Under the model: chosen <b>${sn(d.piW,1)}</b>, rejected <b>${sn(d.piL,1)}</b>. Under the frozen reference: chosen <b>${sn(d.refW,1)}</b>, rejected <b>${sn(d.refL,1)}</b>. β = ${d.beta}. What is the DPO loss for this pair, and which way does one gradient step push each log-probability?`,
                 `一对偏好数据，给出每个回答所有 token 的 log 概率之和。当前模型下：选中的回答 <b>${sn(d.piW,1)}</b>，落选的回答 <b>${sn(d.piL,1)}</b>。冻结的参考模型下：选中的回答 <b>${sn(d.refW,1)}</b>，落选的回答 <b>${sn(d.refL,1)}</b>。β = ${d.beta}。这一对的 DPO loss 是多少？一步梯度下降会把每个 log 概率往哪边推？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const S7=J('step7',tx('part 7','第 7 部分')), W=tx('chosen','选中'), Lz=tx('rejected','落选');
  /* H2: one short block per quantity, each number under the term it replaces */
  const lr=EQ([[`${LR()}_{\\text{${W}}}`,'=',`${nT(d.piW,1)} - (${nT(d.refW,1)})`,'=',sT(d.lrW,1)],[`${LR()}_{\\text{${Lz}}}`,'=',`${nT(d.piL,1)} - (${nT(d.refL,1)})`,'=',sT(d.lrL,1)]]);
  const rw=EQ([[`\\hat{r}_{\\text{${W}}}`,'=',`${d.beta} \\times ${pT(d.lrW,1)}`,'=',nT(d.rW,4)],[`\\hat{r}_{\\text{${Lz}}}`,'=',`${d.beta} \\times ${pT(d.lrL,1)}`,'=',nT(d.rL,4)],
    [TX('reward margin','奖励差'),'=',`${nT(d.rW,4)} - ${pT(d.rL,4)}`,'=',nT(d.m,4)]]);
  const ls=EQ([['L','=',`-\\log\\sigma(${nT(d.m,4)})`,'=',f(d.loss,3)]]);
  const pw=EQ([[TX('push weight','推力权重'),'=',`\\sigma(${nT(-d.m,4)})`,'=',f(d.wt,3)],[TX('coefficient','系数'),'=',`\\beta \\times ${f(d.wt,3)}`,'=',nT(d.coef,4)]]);
  a.innerHTML=tx(`<p><b>loss = ${f(d.loss,3)}</b></p><p>Each log-ratio is the answer’s log-probability under the model minus the one under the reference.</p>${lr}<p>An implicit reward is β times the log-ratio, and the reward margin is the chosen one minus the rejected one.</p>${rw}<p>The loss is ${M('-\\log\\sigma')} of the reward margin.</p>${ls}<p>The gradient pushes ${M('\\log\\pi(\\text{chosen})')} up and ${M('\\log\\pi(\\text{rejected})')} down, each with coefficient β times the push weight ${M('\\sigma({-\\text{reward margin}})')}.</p>${pw}<p>In a real network ${M('\\log\\pi(\\text{chosen})')} can still fall, because the answers share weights (${S7}; Razin et al., ICLR 2025).</p>`,
                 `<p><b>loss = ${f(d.loss,3)}</b></p><p>每个 log 比值，是回答在当前模型下的 log 概率减去参考模型下的。</p>${lr}<p>隐式奖励是 β 乘以 log 比值，奖励差是选中回答的隐式奖励减落选回答的。</p>${rw}<p>loss 是奖励差的 ${M('-\\log\\sigma')}。</p>${ls}<p>梯度把 ${M('\\log\\pi(\\text{选中})')} 往上推、${M('\\log\\pi(\\text{落选})')} 往下推，系数都是 β 乘以推力权重 ${M('\\sigma({-\\text{奖励差}})')}。</p>${pw}<p>在真实网络里，${M('\\log\\pi(\\text{选中})')} 仍可能下降，因为回答之间共享权重（${S7}；Razin 等，ICLR 2025）。</p>`);
}

/* ---------- term tooltips and jumps ---------- */
const CTLN={beta:{en:'β',zh:'β'},eta:{en:'Step size η',zh:'步长 η'},steps:{en:'Training steps',zh:'训练步数'},ds:{en:'Preference pairs',zh:'偏好对'},ref:{en:'SFT model (before training)',zh:'SFT 模型（训练前）'},mode:{en:'How answers get their logits',zh:'回答的 logit 怎么来'}};
const TERM={
 beta:{en:BETA_EN,zh:BETA_ZH},
 eta:{en:`η is the step size: each step moves the logits by ${M('(\\eta / \\beta) \\times \\text{gradient}')}, so the first step is the same size for every β.`,zh:`η 是步长：每一步把 logit 移动 ${M('(\\eta / \\beta) \\times \\text{梯度}')}，所以不管 β 取多少，第一步的改动都一样大。`},
 steps:{en:'The training steps are the gradient steps taken so far from the SFT model.',zh:'训练步数是从 SFT 模型开始已经做了多少步梯度下降。'},
 ref:{en:`This is how often the SFT model gave the answer before training. This SFT model is the reference ${M('\\pi_{\\text{ref}}')}.`,zh:`这是训练前 SFT 模型给这个回答的概率。这个 SFT 模型就是参考模型 ${M('\\pi_{\\text{ref}}')}。`},
 ds:{en:'The loss is averaged over these preference pairs.',zh:'loss 在这些偏好对上取平均。'},
 mode:{en:'This setting says how the answers get their logits: each its own, or through one shared weight.',zh:'这个设置决定回答的 logit 怎么来：各自独立，或由一个共享权重决定。'}
};
const locked=k=>(mode==='challenge'&&!!CTLN[k]&&k!=='beta')||(k==='steps'&&preLock());
function ctlValue(k){return {beta:bTxt(S.beta),eta:String(S.eta),steps:stepsTxt(S.steps),ds:L(DSDESC[S.ds]),ref:refDesc(S.ref),mode:L(MODENAME[S.mode])}[k];}
function jumpAct(k){
  if(k==='math') return tx('Click to open “Show the math”.','点击打开“展开公式”。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='bars') return tx('Click to jump to the answer bars.','点击跳到回答的概率条。');
  if(k==='curve') return tx('Click to jump to the curve.','点击跳到曲线。');
  if(k==='wire') return tx('Click to jump to the picture of where the logits come from.','点击跳到“logit 从哪来”这张图。');
  if(k==='curve-wt') return tx('Click to show the push-weight curve.','点击显示推力权重曲线。');
  if(k==='step7') return tx('Click to open part 7 of the guided tour.','点击打开分步引导的第 7 部分。');
  if(!CTLN[k]) return '';
  if(k==='steps'&&preLock()) return tx('The training steps open after you answer part 2.','在第 2 部分作答后，才能改训练步数。');
  if(locked(k)&&k==='steps') return tx(`Fixed during the challenge: training always runs ${CH.steps} steps.`,`挑战中固定不变：训练总是跑满 ${CH.steps} 步。`);
  if(locked(k)) return tx(`Fixed during the challenge: “${L(CTLN[k])}” is ${ctlValue(k)}.`,`挑战中固定不变：${zsp(L(CTLN[k]))}是 ${ctlValue(k)}。`);
  return tx(`Click to jump to “${L(CTLN[k])}” under Settings.`,`点击跳到“自由调参”里的“${L(CTLN[k])}”。`);
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#dpo-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#dpo-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='step7'){if(mode!=='guided') setMode('guided'); enterStep(SIX('shared')); revealStep(); return;}
  if(k==='wire'){const el=$('#dpo-wire',app); if(el&&!el.hidden){A.tip.hide(); el.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='bars'||k==='curve'||k==='curve-wt'){if(k==='curve-wt') setView('wt'); const el=k==='bars'?$('#dpo-graph',app):$('.dpo-curve-head',app); A.tip.hide(); el.scrollIntoView({block:k==='bars'?'start':'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(k==='bars'?$('#dpo-bars',app):$('#dpo-curve',app)); return;}
  if(!CTLN[k]) return;
  if(locked(k)){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=$('#dpo-ic-'+k,app)||$('#dpo-ctl-'+k,app); if(el) A.ui.jumpToEl(el);
}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='steps'){stopPlay(); wantObs=false; byLearner(()=>setState({steps:+e.target.value}));}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key; if(k==='view'){setView(d.v); return;} if(k==='speed'){setSpeed(d.v); return;} stopPlay();
      wantObs=true; byLearner(()=>setState({[k]:(k==='beta'||k==='eta')?+d.v:d.v})); revealSoon(); wantObs=false; return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); keepStage(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#dpo-rail [data-pred="${pred}"]`,app); if(fb) fb.focus(); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#dpo-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}
/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-13 00:35 UTC: RLHF trains a reward model and then runs PPO → the objective with
   its KL penalty → the optimal policy π* ∝ π_ref·exp(r/β) → solve for r, put it into Bradley–Terry, Z cancels, the DPO loss; nothing on
   what β does, nothing on the push weight): four sections of text and formulas read before the guided parts. They stop at the loss:
   what happens to π(A), π(B) and the answers outside the pair, the gradient and the push weight, and what β's size does stay with the
   guided parts, and no number of the toy is here. Symbols follow the lab's "Where the loss comes from" block and part 4 (π_ref, π*, Z,
   r(y), β, σ, A ≻ B, r̂), the PPO primer (π_θ, x, y, R(x, y)) and the reward-model primer (r, σ, the loss as −𝔼 log σ). One prompt x
   is fixed from section 2 on and brought back for the loss (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=(id,en,zh)=>{const a=A.shell.labLink(id,tx(en,zh)); return tx(a,`“${a}”`);};
// an inline formula and the punctuation right after it never part at a line break
const prNW=h=>h.replace(/(<math\b(?:(?!<\/math>)[\s\S])*<\/math>)([。，；：、）,.;:)]+)/g,'<span class="dpo-nw">$1$2</span>');   // one <math> only
// a paragraph per item; a formula block (a <div>) stands on its own
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return /^<div/.test(h)?h:`<p>${prNW(h)}</p>`;}).join('')});
// a formula on its own line: its pieces (each starting at “=” or an operator) stay whole and wrap only between them, so nothing scrolls
// sideways; {} keeps the spacing of a binary operator at a piece's start, as in the lab's FXW
const prFX=(...xs)=>`<div class="dpo-prfx">${xs.map((x,i)=>`<span>${prM(i?`{} ${x}`:x)}</span>`).join('')}</div>`;
// two formulas side by side, wrapping under each other in a narrow column
const prFX2=(...xs)=>`<div class="dpo-prfx dpo-prfx2">${xs.map(x=>`<span>${prM(x)}</span>`).join('')}</div>`;
const pRF='\\pi_{\\text{ref}}', pEy='\\mathbb{E}_{y \\sim \\pi}', pKL=(a,b)=>`\\text{KL}(${a} \\,‖\\, ${b})`;
const pOBJ=`${pEy}[r(y)] - \\beta\\, ${pKL('\\pi',pRF)}`, pEXP=`${pRF}(y) \\exp(r(y) / \\beta)`;
const prBT='Bradley–⁠Terry';   // a word joiner after the dash: a narrow column does not break the name
const lkRM=()=>prLK('reward-model','Reward model','奖励模型'), lkPPO=()=>prLK('ppo','PPO','PPO'), lkSFT=()=>prLK('sft','SFT','SFT');
A.registerPrimer(ID,{min:10,sections:[
 {h:()=>tx('The cost of RLHF: a reward model and generation during training','RLHF 的代价：奖励模型和训练中的生成'),parts:[prP(
   ()=>tx(`RLHF (see ${lkSFT()}) does not train the chat model on the preference pairs directly: it first trains a reward model on them (see ${lkRM()}), then uses PPO to raise the scores of the chat model’s answers (see ${lkPPO()}).`,
          `RLHF（见${lkSFT()}）不直接拿偏好对训练聊天模型，而是先用它们训练奖励模型（见${lkRM()}），再用 PPO 提高聊天模型回答的得分（见${lkPPO()}）。`),
   ()=>tx(`The second step is costly. During PPO training the current model must write batch after batch of fresh answers, sampling them one token at a time, for the reward model to score; and four models sit on the GPUs at once: the policy and the value model, which are being trained, and the reference model and the reward model, which are frozen (see ${lkPPO()}).`,
          `第二步的代价很大。PPO 训练中要一批又一批地让当前的模型现写回答，一个 token 一个 token 地抽样，写完再交给奖励模型打分；GPU 上还要同时放四个模型：在训练的策略模型和价值模型，冻结的参考模型和奖励模型（见${lkPPO()}）。`),
   ()=>tx(`Yet the human judgments are only ever in the preference pairs. The reward model is an approximation fitted to them, and PPO then trains the chat model on that approximation’s scores. Could we skip both the reward model in between and the generation during training, and train the chat model on the preference pairs directly? To answer that, first write down exactly what PPO maximizes: if the maximum of that objective can be written down directly, there is no need to approach it step by step with freshly written answers.`,
          `而人的判断从头到尾只在偏好对里。奖励模型是从偏好对拟合出来的近似，PPO 再拿这个近似的分数去训练聊天模型。能不能跳过奖励模型这个中间环节，也省掉训练中的生成，直接拿偏好对训练聊天模型？要回答这个问题，先得写清 PPO 最大化的到底是什么：假如这个目标的最大值能直接写出来，就不必靠现写回答一步步去逼近它。`))]},
 {h:()=>tx('What PPO maximizes: the average reward minus a KL penalty','PPO 最大化的目标：平均奖励减去 KL 惩罚'),parts:[prP(
   ()=>tx(`The reward PPO uses is the reward model’s score ${prM('r(x, y)')} minus the KL penalty, which keeps the policy within the range where the reward model can still be trusted (see ${lkPPO()}):`,
          `PPO 用的奖励是奖励模型的分数 ${prM('r(x, y)')} 扣掉 KL 惩罚，让策略模型留在奖励模型还靠得住的范围里（见${lkPPO()}）：`),
   ()=>prFX('R(x, y) = r(x, y)',`- \\beta \\log\\frac{\\pi(y|x)}{${pRF}(y|x)}`),
   ()=>tx(`Here ${prM('\\pi')} is the policy, ${prM(pRF)} is the reference model, the frozen SFT model, and ${prM('\\beta > 0')} is the weight of the KL penalty.`,
          `其中 ${prM('\\pi')} 是策略模型，${prM(pRF)} 是参考模型，也就是冻结的 SFT 模型，${prM('\\beta > 0')} 是 KL 惩罚的系数。`),
   ()=>tx(`PPO maximizes the average of this reward over the answers the policy itself writes. Take one fixed prompt ${prM('x')} and shorten ${prM('\\pi(y|x)')}, ${prM(pRF+'(y|x)')}, ${prM('r(x, y)')} and ${prM('R(x, y)')} to ${prM('\\pi(y)')}, ${prM(pRF+'(y)')}, ${prM('r(y)')} and ${prM('R(y)')}. Averaged, the subtracted term is ${prM('\\beta')} times a quantity called the KL divergence:`,
          `PPO 最大化的，是这个奖励在策略模型自己写出的回答上的平均。只看一道固定的提问 ${prM('x')}，把 ${prM('\\pi(y|x)')}、${prM(pRF+'(y|x)')}、${prM('r(x, y)')}、${prM('R(x, y)')} 简写成 ${prM('\\pi(y)')}、${prM(pRF+'(y)')}、${prM('r(y)')}、${prM('R(y)')}。扣掉的那一项取平均以后，是 ${prM('\\beta')} 乘以一个叫 KL 散度的量：`),
   ()=>prFX(`${pEy}[R(y)]`,`= ${pEy}[r(y)]`,`- \\beta\\, ${pKL('\\pi',pRF)}`),
   ()=>prFX(pKL('\\pi',pRF),`= ${pEy}\\left[\\log\\frac{\\pi(y)}{${pRF}(y)}\\right]`,`= \\sum_y \\pi(y) \\log\\frac{\\pi(y)}{${pRF}(y)}`),
   ()=>tx(`${prM(pKL('\\pi',pRF))} is the KL (Kullback–Leibler) divergence of ${prM('\\pi')} from ${prM(pRF)}. It is never negative, and it is 0 only when ${prM('\\pi')} and ${prM(pRF)} are the same distribution (by Jensen’s inequality, since log is concave); the more probability ${prM('\\pi')} puts on answers that ${prM(pRF)} rarely writes, the larger it is.`,
          `${prM(pKL('\\pi',pRF))} 叫 ${prM('\\pi')} 相对 ${prM(pRF)} 的 KL 散度（Kullback–Leibler divergence）。它总是大于等于 0，只在 ${prM('\\pi')} 和 ${prM(pRF)} 是同一个分布时等于 0（log 是凹函数，由 Jensen 不等式可得）；${prM('\\pi')} 把越多概率放到 ${prM(pRF)} 很少写出的回答上，它就越大。`))]},
 {h:()=>tx('The optimal policy π*: where the KL-penalized objective is largest','最优策略 π*：带 KL 惩罚的目标在哪里最大'),parts:[prP(
   ()=>tx(`PPO raises this objective step by step, with batch after batch of fresh answers and gradient ascent. But the policy’s parameters ${prM('\\theta')} enter the objective only through the answer distribution ${prM('\\pi')}. Set aside which distributions the model can actually produce and let ${prM('\\pi')} be any probability distribution over the answers to this prompt. PPO’s objective also averages over prompts; each prompt’s distribution can then be chosen on its own, and when each prompt’s term is at its maximum, so is the average, so it is enough to solve for one prompt. Write ${prM('r(y)')} as ${prM('\\beta \\log \\exp(r(y) / \\beta)')}, and the two terms merge into one expectation of a single log:`,
          `PPO 靠一批批现写的回答和梯度上升，一步步提高这个目标。可目标里，策略模型的参数 ${prM('\\theta')} 只通过回答的分布 ${prM('\\pi')} 起作用。先不管模型能写出哪些分布，把 ${prM('\\pi')} 当作这道提问的回答上任意一个概率分布。PPO 的目标还要对提问取平均；每道提问的分布可以各自选，每道提问的那一项分别取到最大，平均也就最大，所以只需解一道提问。把 ${prM('r(y)')} 写成 ${prM('\\beta \\log \\exp(r(y) / \\beta)')}，目标的两项就能合进同一个期望，写成一个 log：`),
   ()=>prFX(pOBJ,`= -\\beta\\, ${pEy}\\left[\\log\\frac{\\pi(y)}{${pEXP}}\\right]`),
   ()=>tx(`Suppose the denominator ${prM(pEXP)} were a probability distribution: the right side would then be exactly ${prM('-\\beta')} times the KL divergence of ${prM('\\pi')} from it. Summed over all answers it is usually not 1, so divide it by its sum ${prM('Z')} to normalize it into a distribution ${prM('\\pi^*')}:`,
          `假如分母 ${prM(pEXP)} 是一个概率分布，右边就正好是 ${prM('-\\beta')} 乘以 ${prM('\\pi')} 相对它的 KL 散度。可它对所有回答加起来一般不是 1，所以除以它的和 ${prM('Z')}，归一化成一个分布 ${prM('\\pi^*')}：`),
   ()=>prFX2(`Z = \\sum_y ${pEXP}`,`\\pi^*(y) = \\frac{${pEXP}}{Z}`),
   ()=>tx(`${prM('Z')} sums over every answer to this prompt; it depends only on ${prM(pRF)}, ${prM('r')} and ${prM('\\beta')}, not on ${prM('\\pi')}. With ${prM('Z\\, \\pi^*(y)')} in the denominator, a ${prM('-\\log Z')} splits off inside the log:`,
          `${prM('Z')} 对这道提问的所有回答求和，只取决于 ${prM(pRF)}、${prM('r')} 和 ${prM('\\beta')}，和 ${prM('\\pi')} 无关。把分母换成 ${prM('Z\\, \\pi^*(y)')}，log 里就分出一个 ${prM('-\\log Z')}：`),
   ()=>prFX(pOBJ,`= -\\beta\\, ${pEy}\\left[\\log\\frac{\\pi(y)}{\\pi^*(y)} - \\log Z\\right]`,`= -\\beta\\, ${pKL('\\pi','\\pi^*')} + \\beta \\log Z`),
   ()=>tx(`${prM('\\beta \\log Z')} does not depend on ${prM('\\pi')}. ${prM(pKL('\\pi','\\pi^*'))} is never negative and is 0 only when ${prM('\\pi = \\pi^*')}, so times ${prM('-\\beta')} it is never positive. Hence the objective is largest at ${prM('\\pi = \\pi^*')}, and only there. ${prM('\\pi^*')} is called the optimal policy of this objective: it multiplies the probability the reference model gives each answer by ${prM('\\exp(r(y) / \\beta)')}, which scales the higher-reward answers up more, and then normalizes. Each prompt has its own ${prM('Z')} and ${prM('\\pi^*')}.`,
          `${prM('\\beta \\log Z')} 和 ${prM('\\pi')} 无关。${prM(pKL('\\pi','\\pi^*'))} 大于等于 0，只在 ${prM('\\pi = \\pi^*')} 时等于 0，乘上 ${prM('-\\beta')} 以后就小于等于 0。所以目标在 ${prM('\\pi = \\pi^*')} 时最大，而且只在这里最大。${prM('\\pi^*')} 叫这个目标的最优策略：它把参考模型给每个回答的概率乘上 ${prM('\\exp(r(y) / \\beta)')}，奖励越高的回答放大得越多，再归一化。每道提问各有自己的 ${prM('Z')} 和 ${prM('\\pi^*')}。`),
   ()=>tx(`But ${prM('\\pi^*')} cannot be used directly as a training target: to have the model learn ${prM('\\pi^*')}, the probability of each answer under ${prM('\\pi^*')} must be computed first, and that takes ${prM('Z')}. ${prM('Z')} sums over every possible answer to a prompt; an answer can be hundreds of tokens long, so there are far too many possible answers to list, and ${prM('Z')} cannot be computed. And ${prM('r')} is still the reward model that has to be trained first.`,
          `可 ${prM('\\pi^*')} 没法直接当作训练目标：要让模型去学 ${prM('\\pi^*')}，先得算出每个回答在 ${prM('\\pi^*')} 下的概率，这就要用到 ${prM('Z')}。${prM('Z')} 要对一道提问所有可能的回答求和，而一个回答可以长达几百个 token，可能的回答多到没法一一列举，${prM('Z')} 算不出来；${prM('r')} 也还是那个要先训练出来的奖励模型。`))]},
 {h:()=>tx(`Solve for the reward and put it into ${prBT} (the DPO loss)`,`反解出奖励，代入 ${prBT} 模型（DPO 的 loss）`),parts:[prP(
   ()=>tx(`Read backwards, though, the formula is useful. Take the log of both sides of the formula for ${prM('\\pi^*')} and rearrange, and the reward is written with the optimal policy and the reference model:`,
          `不过这个式子反过来读却有用。对 ${prM('\\pi^*')} 的式子两边取对数，整理一下，奖励就用最优策略和参考模型写了出来：`),
   ()=>prFX(`r(y) = \\beta \\log\\frac{\\pi^*(y)}{${pRF}(y)}`,'+ \\beta \\log Z'),
   ()=>tx(`The ${prBT} model used to train the reward model assumes that the probability a person picks A over B (written A ≻ B) depends only on the difference of the two answers’ rewards, ${prM('P(A ≻ B) = \\sigma(r(A) - r(B))')}, where ${prM('\\sigma')} is the sigmoid function (see ${lkRM()}). Put the ${prM('r')} above into it. A and B are two answers to the same prompt, so they share the same ${prM('Z')}, and ${prM('\\beta \\log Z')} cancels in the difference:`,
          `训练奖励模型时用的 ${prBT} 模型假设：人选 A 而不选 B（记作 A ≻ B）的概率只看两个回答的奖励之差，${prM('P(A ≻ B) = \\sigma(r(A) - r(B))')}，${prM('\\sigma')} 是 sigmoid 函数（见${lkRM()}）。把上面的 ${prM('r')} 代进去。A 和 B 是同一道提问的两个回答，共用同一个 ${prM('Z')}，相减时 ${prM('\\beta \\log Z')} 消掉了：`),
   ()=>prFX('P(A ≻ B)',`= \\sigma\\bigg(\\beta \\log\\frac{\\pi^*(A)}{${pRF}(A)}`,`- \\beta \\log\\frac{\\pi^*(B)}{${pRF}(B)}\\bigg)`),
   ()=>tx(`The ${prM('Z')} that could not be computed is gone: the probability that a person picks A depends only on the probabilities that the optimal policy and the reference model give the two answers A and B.`,
          `算不出来的 ${prM('Z')} 不见了：人选 A 的概率，只取决于最优策略和参考模型给 A、B 这两个回答的概率。`),
   ()=>tx(`This ${prM('r')} would have to come from training a reward model, so ${prM('\\pi^*')} is unknown as well. Instead of training ${prM('r')}, put the policy being trained, ${prM('\\pi_\\theta')} (see ${lkPPO()}), in the place of ${prM('\\pi^*')} and, as in training the reward model, fit the human choices on the preference pairs by maximum likelihood. With ${prM('\\pi_\\theta')} in place of ${prM('\\pi^*')}, each answer’s term in the formula for ${prM('P(A ≻ B)')} becomes`,
          `这个 ${prM('r')} 本来要靠训练奖励模型得到，${prM('\\pi^*')} 也就跟着未知。干脆不训练 ${prM('r')}，直接把正在训练的策略模型 ${prM('\\pi_\\theta')}（见${lkPPO()}）放在 ${prM('\\pi^*')} 的位置上，像训练奖励模型那样，用最大似然去拟合人在偏好对上的选择。${prM('\\pi_\\theta')} 换掉 ${prM('\\pi^*')} 以后，${prM('P(A ≻ B)')} 的式子里每个回答的那一项变成`),
   ()=>prFX(`\\hat{r}(y) = \\beta \\log\\frac{\\pi_\\theta(y)}{${pRF}(y)}`),
   ()=>tx(`It is called the implicit reward: no separate reward model is trained, and it comes straight from the probabilities of ${prM('\\pi_\\theta')} and ${prM(pRF)}. Why is ${prM('\\pi_\\theta')}, once fitted, the chat model we want? Put ${prM('\\hat{r}')} back into the formula for ${prM('\\pi^*')} as the reward: ${prM(pRF+'(y) \\exp(\\hat{r}(y) / \\beta) = \\pi_\\theta(y)')}, which sums to 1 over all answers, so ${prM('Z = 1')} and the optimal policy is ${prM('\\pi_\\theta')} itself. In other words, while ${prM('\\pi_\\theta')} fits the preference pairs, it is always the optimal policy of its own implicit reward: once the reward is fitted, the KL-penalized objective is already solved, and there is no PPO left to run.`,
          `它叫隐式奖励：不用单独训练奖励模型，由 ${prM('\\pi_\\theta')} 和 ${prM(pRF)} 的概率直接算出。为什么拟合好以后，${prM('\\pi_\\theta')} 本身就是要的聊天模型？把 ${prM('\\hat{r}')} 当作奖励代回 ${prM('\\pi^*')} 的式子：${prM(pRF+'(y) \\exp(\\hat{r}(y) / \\beta) = \\pi_\\theta(y)')}，对所有回答加起来是 1，所以 ${prM('Z = 1')}，最优策略正是 ${prM('\\pi_\\theta')}。也就是说，${prM('\\pi_\\theta')} 在拟合偏好对的同时，始终是它自己那个隐式奖励的最优策略：奖励拟合好了，带 KL 惩罚的目标也就解完了，不必再跑 PPO。`),
   ()=>tx(`Restricting the reward to this form is no worse than training a separate reward model. As seen above, any reward ${prM('r')} equals ${prM('\\beta \\log(\\pi^* / '+pRF+')')} plus ${prM('\\beta \\log Z')}, which changes only from prompt to prompt; and adding the same number to every answer of a prompt changes no probability of a person picking one answer over another (see ${lkRM()}). So as long as ${prM('\\pi_\\theta')} can represent that ${prM('\\pi^*')}, any preferences a reward model can fit, the implicit reward can fit too.`,
          `把奖励限制成这种形式，并不比单独训练一个奖励模型差。前面已经看到，任何奖励 ${prM('r')} 都等于 ${prM('\\beta \\log(\\pi^* / '+pRF+')')} 加上只随提问变的 ${prM('\\beta \\log Z')}；给同一道提问的所有回答加同一个数，人选每个回答的概率都不变（见${lkRM()}）。所以只要 ${prM('\\pi_\\theta')} 能表示出那个 ${prM('\\pi^*')}，奖励模型能拟合出的偏好，隐式奖励也都能拟合出来。`),
   ()=>tx(`Now bring the prompt ${prM('x')} back. Each preference pair is a prompt ${prM('x')} and two of its answers: the one a person chose, written A, and the one rejected, written B, with implicit rewards ${prM('\\hat{r}_A')} and ${prM('\\hat{r}_B')}. By the formula above, the probability of this choice is ${prM('\\sigma(\\hat{r}_A - \\hat{r}_B)')}. As with the reward model’s loss, maximizing the likelihood of all the pairs means minimizing the negative log-likelihood, averaged over the pairs. That is the loss of DPO (direct preference optimization):`,
          `把提问 ${prM('x')} 写回来。每个偏好对是一道提问 ${prM('x')} 和它的两个回答，人选中的记作 A，落选的记作 B，它们的隐式奖励记作 ${prM('\\hat{r}_A')} 和 ${prM('\\hat{r}_B')}。按上面的式子，人做出这个选择的概率是 ${prM('\\sigma(\\hat{r}_A - \\hat{r}_B)')}。和奖励模型的 loss 一样，最大化所有偏好对的似然，就是最小化负对数似然，再按偏好对取平均。这就是 DPO（direct preference optimization，直接偏好优化）的 loss：`),
   ()=>prFX(`L_{\\text{DPO}}(\\theta) = -\\mathbb{E}\\left[\\log \\sigma(\\hat{r}_A - \\hat{r}_B)\\right]`),
   ()=>prFX2(`\\hat{r}_A = \\beta \\log\\frac{\\pi_\\theta(A|x)}{${pRF}(A|x)}`,`\\hat{r}_B = \\beta \\log\\frac{\\pi_\\theta(B|x)}{${pRF}(B|x)}`),
   ()=>tx(`${prM('\\mathbb{E}')} is the average over all the preference pairs in the data. To compute the loss, give the two answers already in each pair to ${prM('\\pi_\\theta')} and to ${prM(pRF)} and get their log probabilities: an answer’s log probability is the sum of its tokens’ log probabilities (see ${lkPPO()}). So DPO training needs only the preference pairs, the model being trained and a frozen reference model: no reward model, and no step where the model writes answers.`,
          `${prM('\\mathbb{E}')} 表示对数据里所有偏好对取平均。算这个 loss，只要把每个偏好对里现成的两个回答分别交给 ${prM('\\pi_\\theta')} 和 ${prM(pRF)}，算出它们的 log 概率：一个回答的 log 概率是它各个 token 的 log 概率之和（见${lkPPO()}）。所以 DPO 训练要的只有偏好对、正在训练的模型和一份冻结的参考模型：没有奖励模型，训练中也没有哪一步要模型现写回答。`))]}
]});

/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings with the training steps reached, the dashed
   earlier run, the part, the answer, the chart shown, the β values tried, the playing speed, the challenge's result, the drill.
   Playback is not kept: a run that was playing stops at the step it had reached. */
const okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), rsCfg=x=>okObj(x)?parse(token(x)):null;
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:{...S},pred,observed:{...observed},ghost:ghost?{...ghost}:null,saved:saved?{...saved}:null,view,triedB:[...triedB],speed,chSubmitted,chReveal,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&chPoint(S.beta).ok};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...(rsCfg(o.S)||DEFAULT)}; if(mode==='challenge') S={...S,...CH,beta:BETAS.includes(S.beta)?S.beta:0.1};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; const g=okObj(o.ghost)?rsCfg({...o.ghost,steps:0}):null; ghost=g?cfgOf(g):null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    saved=mode==='challenge'?rsCfg(o.saved):null;
    view=VIEWS.some(x=>x[0]===o.view)?o.view:'prob';
    triedB=new Set(Array.isArray(o.triedB)?o.triedB.filter(v=>BETAS.includes(v)):[]); if(!triedB.size) triedB.add(0.1);
    speed=typeof o.speed==='string'&&Object.prototype.hasOwnProperty.call(SPEEDS,o.speed)?o.speed:'med';
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:7; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    stopPlay(); wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps the dashed earlier run */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) ghost=null; S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#dpo-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#dpo-rail',app));},token});
    bind(app);
    renderDrill();
    renderRail();                       // (renders too) the one path that adds the next-lab footer (site review R10 #3)
    watchStage();
  },
  leave(){stopPlay(); clearTimeout(annT); stageStop(); unwatchStage(); root=null; app=null;},
  onLang(){if(root){const d=$('#dpo-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#dpo-math',app).open=true;}},
  onResize(){if(app) drawCurve(compute(S));},
  termTip(k){const tm=TERM[k]; if(!tm) return ''; return `<div>${L(tm)}</div><div class="act">${jumpAct(k)}</div>`;},   // TERM holds HTML: η and ref carry typeset formulas (H1)
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#dpo-bars .dpo-row',app)||$('#dpo-bars',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#dpo-graph',app),
    /* one line in plain words; phones hide the .dk-mem part, so what matters comes first */
    render({status,vis,read}){if(!app) return; const r=compute(S), mem=x=>`<span class="dk-mem dpo-dk-mem">${x}</span>`;
      vis.innerHTML=`<div class="dpo-strip">${r.pi.map((q,i)=>`<i class="dpo-sg dpo-c-${roleCls(i,r.pairs)}" style="width:${(q*100).toFixed(2)}%"></i>`).join('')}</div>`;
      if(mode==='challenge'){const p=chPoint(S.beta), a=pc(p.pA), k=f(p.kl,2); status.className='status dpo-dk-st '+(p.ok?'ok':'bad'); status.textContent=p.ok?tx('Goal met','已达标'):tx('Not yet','未达标');
        const lead=p.ok?tx(`The model gives A ${a} of the time; its KL from the SFT model is ${k} nats`,`模型给出 A 的概率是 ${a}，离 SFT 模型的 KL 是 ${k} nats`)
          :(!p.okK&&!p.okA?tx(`The model gives A only ${a} of the time, and its KL from the SFT model is ${k} nats`,`模型给出 A 的概率只有 ${a}，离 SFT 模型的 KL 是 ${k} nats`)
          :(!p.okK?tx(`The KL from the SFT model is ${k} nats, over the 0.8-nat limit`,`离 SFT 模型的 KL 是 ${k} nats，超过上限 0.8 nats`):tx(`The model gives A only ${a} of the time, under 50%`,`模型给出 A 的概率只有 ${a}，不到 50%`)));
        read.innerHTML=esc(lead)+mem(tx(` (β = ${bTxt(S.beta)}, 60 steps)`,`（β = ${bTxt(S.beta)}，训练 60 步）`)); return;}
      status.className='status dpo-dk-st '+(r.top===0?'ok':'bad'); status.textContent=tx(`The model most often gives ${LET[r.top]}`,`模型最常给出 ${LET[r.top]}`);
      const klm=tx(`its KL from the SFT model is ${klS(r.kl)} nats`,`离 SFT 模型的 KL 是 ${klS(r.kl)} nats`);
      if(view==='wt') read.innerHTML=esc(tx(`At step ${r.t} the push weight is ${pw3(r.wAB)}`,`第 ${r.t} 步的推力权重是 ${pw3(r.wAB)}`))+mem(tx(` (β = ${bTxt(S.beta)}); ${klm}`,`（β = ${bTxt(S.beta)}），${klm}`));
      else if(view==='kl') read.innerHTML=esc(tx(`At step ${r.t} the KL from the SFT model is ${klS(r.kl)} nats`,`第 ${r.t} 步离 SFT 模型的 KL 是 ${klS(r.kl)} nats`))+mem(tx(` (β = ${bTxt(S.beta)}); the model gives A ${pc(r.pi[0])} of the time`,`（β = ${bTxt(S.beta)}），模型给出 A 的概率是 ${pc(r.pi[0])}`));
      else read.innerHTML=esc(r.t===0?tx(`Before training the model gives A ${pc(r.pi[0])} of the time`,`训练前，模型给出 A 的概率是 ${pc(r.pi[0])}`):tx(`After ${stepsTxt(r.t)} the model gives A ${pc(r.pi[0])} of the time`,`训练 ${r.t} 步后，模型给出 A 的概率是 ${pc(r.pi[0])}`))
        +mem(r.t===0?tx(` (β = ${bTxt(S.beta)})`,`（β = ${bTxt(S.beta)}）`):tx(` (${pc(r.T.pi[0][0])} before training, β = ${bTxt(S.beta)}); ${klm}`,`（训练前 ${pc(r.T.pi[0][0])}，β = ${bTxt(S.beta)}），${klm}`));}
  }
});

/* ---------- home tile teaser: one slider, four bars, one or two sentences (R11.5) ---------- */
A.registerTeaser('dpo',{render(el){
  const TM=40;
  el.innerHTML=`<div class="tz"><label class="tz-lab"><span>${tx('DPO training steps','DPO 训练步数')}</span><output class="mono dpo-tz-out"></output></label>`+
    `<input type="range" min="0" max="${TM}" step="1" value="${tzSteps}" aria-label="${esc(tx('DPO training steps','DPO 训练步数'))}">`+
    `<p class="dpo-tz-cap small muted">${tx('Four answers to one prompt; each bar is how often the model gives that answer.','一个提问的四个回答；每条是模型给出这个回答的概率。')}</p>`+
    `<div class="dpo-tz" aria-hidden="true"></div>`+
    `<div class="tz-leg"><span><i class="sw dpo-sw-w"></i>${tx('A: preferred','A：标注员更喜欢')}</span><span><i class="sw dpo-sw-l"></i>${tx('B: rejected','B：落选')}</span><span><i class="sw dpo-sw-o"></i>${tx('C, D: not labeled','C、D：未标注')}</span><span><i class="sw dpo-sw-ref"></i>${tx('SFT model before training','训练前的 SFT 模型')}</span></div>`+
    `<p class="tz-read"></p>`+
    `<p class="tz-note">${tx('β is the weight of the KL penalty in the RLHF objective that DPO is derived from; this tile uses 0.1, the default in the DPO paper and in TRL. In this lab, the larger β, the less the model changes in the same number of steps, so the closer it stays to the SFT model it started from.','β 是 RLHF 目标里 KL 惩罚的系数，DPO 就是从这个目标推出来的；这里用 0.1，是 DPO 论文和 TRL 的默认值。在这个 Lab 里，β 越大，同样的训练步数下模型改动得越少，离训练前的 SFT 模型越近。')}</p>`+
    `<p class="tz-note">${tx('Step size 0.1 sets how much each training step changes the model. It was chosen for this lab’s four answers and cannot be compared with a real run’s learning rate.','步长 0.1 决定每训练一步模型改动多少。它是为这个 Lab 的四个回答选的，和真实训练的学习率没法直接比。')}</p>`+
    `<p class="tz-note">${tx('This is a toy run: your browser takes real DPO gradient steps (Rafailov et al., 2023) on four made-up answers to one prompt; the probabilities before training are illustrative.','这是玩具实跑：浏览器在一个提问的四个虚构回答上，真的做 DPO 的梯度下降（Rafailov 等，2023）；训练前的概率是示意数值。')}</p></div>`;
  const inp=el.querySelector('input'), out=el.querySelector('output'), vis=el.querySelector('.dpo-tz'), rd=el.querySelector('.tz-read');
  const upd=()=>{tzSteps=+inp.value; A.ui.setRange(inp); const r=compute({...DEFAULT,steps:tzSteps}), p0=r.T.pi[0];
    out.textContent=stepsTxt(tzSteps);
    vis.innerHTML=LET.map((l,i)=>`<div class="dpo-tz-row dpo-c-${roleCls(i,r.pairs)}"><span class="dpo-tz-l mono">${l}</span><span class="dpo-tz-t"><i class="dpo-fill" style="width:${(r.pi[i]*100).toFixed(2)}%"></i><i class="dpo-ghost" style="width:${(p0[i]*100).toFixed(2)}%"></i></span><span class="dpo-tz-v mono">${pc(r.pi[i])}</span></div>`).join('');
    rd.innerHTML=tzSteps===0?tx(`Annotators prefer answer A to B, but before training the SFT model gives A only <b>${pc(p0[0])}</b> of the time and B <b>${pc(p0[1])}</b>.`,`标注员更喜欢回答 A 而不是 B，可训练前的 SFT 模型只有 <b>${pc(p0[0])}</b> 的时候给出 A，<b>${pc(p0[1])}</b> 的时候给出 B。`)
      :tx(`Annotators prefer answer A to B. After ${tzSteps===1?'1 DPO step':`${tzSteps} DPO steps`}, the model gives A <b>${pc(r.pi[0])}</b> of the time instead of ${pc(p0[0])}, and B ${lessW(p0[1],r.pi[1])}.`,
          `标注员更喜欢回答 A 而不是 B。训练 ${tzSteps} 步 DPO 后，模型给出 A 的概率从 ${pc(p0[0])} 升到 <b>${pc(r.pi[0])}</b>，给出 B 的概率则${moveW(p0[1],r.pi[1])}。`);};   // one key number; B's change in words (R6.20)
  inp.addEventListener('input',upd); upd();
}});

/* read-only hook for the golden-number tests */
window.__dpo=Object.freeze({compute:s=>compute({...DEFAULT,...s}),chain:s=>compute({...DEFAULT,...s}).c,train,chPoint,makeDrill,passStep:s=>passStep({...DEFAULT,...s}),
  consts:Object.freeze({MAXS,REFS,DETAIL,DATA,BETAS,ETAS,CH,CH_PA,CH_KL,SPEEDS,STEP_IDS:STEPS.map(s=>s.id)}),state:()=>({...S,logits:S.mode,mode,step,view,speed,pred,playing})});
})();
