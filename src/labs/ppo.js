/* PPO lab (batch 9): why RLHF-PPO needs a value model and a KL penalty, and what clipping does.
   A live toy: 3 prompts, 5 candidate responses each, a softmax policy over them (one logit per response plus one weight w shared
   by every prompt that multiplies each response's length), trained in the browser with real policy-gradient steps against a toy
   reward model. Seeded (mulberry32, seed 7), so the same settings always give the same run; tests/ppo_numbers.py reimplements it.
   Names (EN / ZH, brief S2): policy 策略模型, reference model 参考模型, reward model 奖励模型, value model 价值模型 (critic),
   advantage 优势 = reward − baseline, KL penalty / KL coefficient β, clipping / clip range ε, probability ratio 概率之比 ρ. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='ppo';

/* ---------- engine (pure) ---------- */
const NP=3, NY=5, NS=4, MAXS=200, SEED=7, EPS=0.2, LRV=0.5;
const REF=[0.23,0.25,0.25,0.24,0.03];                        // the SFT model = reference, the same for every prompt (illustrative)
const RM=[[0.90,0.85,0.80,0.88,0.94],[0.60,0.55,0.50,0.58,0.64],[0.27,0.22,0.17,0.25,0.31]];   // toy reward-model scores
const QL=[0.95,0.75,0.55,0.50,0.20];                          // true quality of each kind of response (the lab's own, hidden in real training)
const LEN=[0.2,0.4,0.1,0.7,1.0];                              // relative length of each kind of response
const ETA={n:8,l:16}, BETAS=[0,0.05,0.2], KS=[1,4], FAN=[7,1,2,3,5,11];
const DEFAULT={base:1,beta:0.05,K:4,clip:1,eta:'n',steps:200};
const CFG=['base','beta','K','clip','eta'];
const LREF=REF.map(Math.log);
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function softmax(z){let m=-Infinity; for(const v of z) if(v>m) m=v; const e=z.map(v=>Math.exp(v-m)); let s=0; for(const v of e) s+=v; return e.map(v=>v/s);}
function pol(th,w){const P=[]; for(let k=0;k<NP;k++){const z=[]; for(let i=0;i<NY;i++) z.push(th[k][i]+w*LEN[i]); P.push(softmax(z));} return P;}
/* The clipped objective of one response, min(ρA, clip(ρ, 1−ε, 1+ε)A), and its slope (review codex-b9v3, item 7). The slope with respect
   to log π is ρA, except on the clipped plateau: A > 0 with ρ > 1 + ε (already pushed up enough), or A < 0 with ρ < 1 − ε (already pushed
   down enough). There the clipped term is the smaller one and does not change with ρ, so this response gives no gradient. A response
   that moved the other way (A > 0 with ρ < 1 − ε, A < 0 with ρ > 1 + ε) is on the unclipped term and is still pushed back. */
const onPlateau=(a,rho)=>(a>0&&rho>1+EPS)||(a<0&&rho<1-EPS);
const clipObj=(a,rho,clip)=>clip?Math.min(rho*a,Math.min(Math.max(rho,1-EPS),1+EPS)*a):rho*a;
const slopeOf=(a,rho,clip)=>clip&&onPlateau(a,rho)?0:rho*a;
/* One run from the SFT model for MAXS steps. Each step: every prompt gets NS sampled responses from the current policy; each gets the
   reward r − β·log(π_old/π_ref); the advantage is reward − V(prompt) with the value model, or the reward itself without it; then K
   passes of gradient ascent on mean_n min(ρA, clip(ρ, 1−ε, 1+ε)A) per prompt (summed over prompts; without clipping, ρA); then the
   value model moves halfway to the mean reward it just saw. For step 1's batch it also keeps each response's ratio at the start of
   every pass (rp) and whether that pass pushed it (on): part 6 draws them. */
function train(c){
  const rnd=mulberry(c.seed), eta=ETA[c.eta];
  const th=[0,1,2].map(()=>LREF.slice()); let w=0;
  const V=RM.map(r=>r.reduce((s,x,i)=>s+REF[i]*x,0));
  const T={pi:[],rm:[],q:[],kl:[],V:[],w:[],B:[],rho:[]};
  for(let t=0;t<=MAXS;t++){
    const P=pol(th,w); let rm=0,q=0,kl=0;
    for(let k=0;k<NP;k++) for(let i=0;i<NY;i++){const p=P[k][i]; rm+=p*RM[k][i]; q+=p*QL[i]; if(p>0) kl+=p*(Math.log(p)-LREF[i]);}
    T.pi.push(P); T.rm.push(rm/NP); T.q.push(q/NP); T.kl.push(kl/NP); T.V.push(V.slice()); T.w.push(w);
    if(t===MAXS) break;
    const B=[];
    for(let k=0;k<NP;k++) for(let n=0;n<NS;n++){
      const u=rnd(); let y=NY-1, cu=0; for(let i=0;i<NY;i++){cu+=P[k][i]; if(u<cu){y=i; break;}}
      const lp=Math.log(P[k][y]), d=lp-LREF[y], R=RM[k][y]-c.beta*d;
      B.push({k,y,lp,p:P[k][y],r:RM[k][y],d,R,v:V[k],a:c.base?R-V[k]:R,rp:t===0?[]:null,on:t===0?[]:null});}
    for(let e=0;e<c.K;e++){
      const Pn=pol(th,w), g=th.map(()=>[0,0,0,0,0]); let gw=0;
      for(const b of B){const rho=Pn[b.k][b.y]/Math.exp(b.lp), s=slopeOf(b.a,rho,c.clip);
        if(b.rp){b.rp.push(rho); b.on.push(!(c.clip&&onPlateau(b.a,rho)));}
        if(s===0) continue;
        let ml=0; for(let i=0;i<NY;i++) ml+=Pn[b.k][i]*LEN[i];
        for(let i=0;i<NY;i++) g[b.k][i]+=s*((i===b.y?1:0)-Pn[b.k][i])/NS;
        gw+=s*(LEN[b.y]-ml)/(NS*NP);}
      for(let k=0;k<NP;k++) for(let i=0;i<NY;i++) th[k][i]+=eta*g[k][i];
      w+=eta*gw;}
    const Pe=pol(th,w); let mn=Infinity, mx=-Infinity;
    for(const b of B){b.rho=Pe[b.k][b.y]/Math.exp(b.lp); if(b.rho<mn) mn=b.rho; if(b.rho>mx) mx=b.rho;}
    T.B.push(B); T.rho.push([mn,mx]);
    for(let k=0;k<NP;k++){let s=0; for(let n=0;n<NS;n++) s+=B[k*NS+n].R; V[k]+=LRV*(s/NS-V[k]);}
  }
  return T;
}
const cache=new Map();
const keyOf=s=>CFG.map(k=>s[k]).join('|')+'|'+(s.seed||SEED);
function traj(s){const c={base:s.base,beta:s.beta,K:s.K,clip:s.clip,eta:s.eta,seed:s.seed||SEED}, k=keyOf(c); let T=cache.get(k); if(!T){T=train(c); cache.set(k,T);} return T;}
function peakOf(T,t){let k=0; for(let u=0;u<=t;u++) if(T.q[u]>T.q[k]) k=u; return k;}
/* the moment that goes wrong (R5.11): after true quality's peak, the reward-model score is higher than at the peak while true
   quality has fallen by at least 0.05 */
function hackFrom(T,t){const k=peakOf(T,t); return (T.q[k]-T.q[t]>=0.05&&T.rm[t]>T.rm[k])?k:-1;}
const runaway=T=>{for(let u=0;u<MAXS;u++) if(T.rho[u][1]>3) return u; return -1;};   // the first update that moves a response's probability over 3 times
/* the same settings over several seeds (B2 of the review: the page says what these seeds show, never “always”): the 6 seeds of the
   fans in parts 4 and 5, and 40 seeds (1–40) for how often a run without the KL penalty collapses */
const SEEDS40=Array.from({length:40},(_,i)=>i+1);
const fanQ=(s,t)=>FAN.map(sd=>traj({...s,seed:sd}).q[t]);
const manyCache=new Map();
function manyEnds(s){const k=keyOf({...s,seed:0}); let v=manyCache.get(k); if(!v){v=SEEDS40.map(sd=>traj({...s,seed:sd}).q[MAXS]); manyCache.set(k,v);} return v;}
const topOf=p=>{let j=0; p.forEach((x,i)=>{if(x>p[j]) j=i;}); return j;};
function compute(s){
  const T=traj(s), t=Math.max(0,Math.min(MAXS,Math.round(s.steps)));
  const gap=T.pi[t].map(p=>p[4]), top=T.pi[t].map(p=>{let j=0; p.forEach((x,i)=>{if(x>p[j]) j=i;}); return j;});
  let rmax=0; for(let u=0;u<t;u++) if(T.rho[u][1]>rmax) rmax=T.rho[u][1];
  return {s,T,t,pi:T.pi[t],rm:T.rm[t],q:T.q[t],kl:T.kl[t],V:T.V[t],w:T.w[t],gap,top,hack:hackFrom(T,t),rmax,peak:peakOf(T,t)};
}
/* memory of the four models (brief S5): 18 bytes per trained parameter, 2 per frozen BF16 parameter; GB = 10^9 bytes */
const P8=8030261248, GPU_GB=80;
const MEM={ppo:[['pol',18],['val',18],['ref',2],['rew',2]],grpo:[['pol',18],['ref',2],['rew',2]],dpo:[['pol',18],['ref',2]],grpo0:[['pol',18],['rew',2]]};   // grpo0: GRPO with β = 0 loads no reference model (TRL's default since v0.19.0, brief S5a)
const memOf=alg=>{const parts=MEM[alg].map(([k,b])=>({k,b,gb:P8*b/1e9})); const tot=parts.reduce((s,p)=>s+p.gb,0); return {parts,tot,gpus:Math.ceil(tot/GPU_GB-1e-9)};};
/* challenge: a large step, 4 passes per batch, no KL penalty and no clipping; change β, the passes and clipping */
const CH={base:1,eta:'l',steps:200}, CH0={beta:0,K:4,clip:0};
const CH_KL=0.5, Q0=traj({...DEFAULT}).q[0];
const chPoint=p=>{const s={...CH,...p}, T=traj(s), q=T.q[MAXS], kl=T.kl[MAXS]; const n=['beta','K','clip'].filter(k=>p[k]!==CH0[k]).length;
  return {...p,q,kl,n,okQ:q>Q0,okK:kl<=CH_KL,ok:q>Q0&&kl<=CH_KL};};
const chAll=()=>{const o=[]; BETAS.forEach(b=>KS.forEach(K=>[0,1].forEach(c=>o.push(chPoint({beta:b,K,clip:c}))))); return o;};

/* ---------- formatting ---------- */
const f=(v,d)=>A.fmt.num(v,d);
const f2=v=>f(v,2), pc=p=>Math.round(p*100)+'%';
const bTxt=b=>String(b);
const LET=['A','B','C','D','E'];
const KIND=[{en:'concise',zh:'简洁'},{en:'detailed',zh:'详细'},{en:'terse',zh:'太短'},{en:'rambling',zh:'啰嗦'},{en:'padded',zh:'注水'}];
const KDESC=[{en:'concise and correct: one or two sentences that answer the question',zh:'简洁准确：一两句话，答对了'},
 {en:'detailed and correct: a little background as well, also right',zh:'详细准确：多讲了几句背景，也对'},
 {en:'too terse: a few words that do not really answer',zh:'太简略：几个字，没真正回答'},
 {en:'long and rambling: the answer is buried in filler',zh:'又长又绕：答案淹没在废话里'},
 {en:'padded: very long, the same sentence reworded again and again, almost no content',zh:'注水：很长，同一句话换着说法反复说，几乎没有内容'}];
const EX=[{en:'“Gracias.”',zh:'“Gracias。”'},{en:'“Gracias. To be more polite, say ‘Muchas gracias’.”',zh:'“Gracias。想更客气，可以说 Muchas gracias。”'},
 {en:'“Spanish has a word for it.”',zh:'“西班牙语里有专门的说法。”'},{en:'“Great question! Spanish is a beautiful language… (three paragraphs later) the word is gracias.”',zh:'“好问题！西班牙语是一门很美的语言……（三段之后）答案是 gracias。”'},
 {en:'“Gracias. Gracias means thank you. Thank you in Spanish is gracias. In short, gracias… (a dozen more times)”',zh:'“Gracias。Gracias 的意思是谢谢。谢谢在西班牙语里就是 gracias。总之，gracias……（再重复十几遍）”'}];
const PROMPTS=[{d:{en:'Easy',zh:'简单'},q:{en:'How do you say “thank you” in Spanish?',zh:'“谢谢”用西班牙语怎么说？'}},
 {d:{en:'Medium',zh:'中等'},q:{en:'Why is the sky blue?',zh:'天为什么是蓝的？'}},
 {d:{en:'Hard',zh:'难'},q:{en:'Write a four-line poem about the KV cache.',zh:'写一首关于 KV cache 的四行诗。'}}];
const MINUS='−';
const sg=(v,d)=>{const s=f(Math.abs(v),d); return +s===0?f(0,d):(v<0?MINUS:'+')+s;};
const stepsTxt=n=>tx(n===1?'1 step':`${n} steps`,`${n} 步`);
const zsp=s=>/[A-Za-z0-9)Ͱ-Ͽ]$/.test(s)?s+' ':s;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const M=s=>A.ui.math?A.ui.math(s):esc(s);
const mathLine=s=>`<div class="ppo-fx">${M(s)}</div>`;
/* H2 (the owner, 2026-10-09): a formula with its name in words on the left, its chain broken at “=” onto lines that line up; a line may be [op, tex].
   The notation follows the dpo, grpo and rlvr labs: π_θ, π_old, π_ref, β, ρ and the advantage Â */
const EQ=(name,lines,op='=')=>`<div class="ppo-eq">${lines.map((l,i)=>{const [o,t]=Array.isArray(l)?l:[op,l]; return `<div class="ppo-eq-n">${i?'':name}</div><div class="ppo-eq-o">${o?M(o):''}</div><div class="ppo-fx">${M(t)}</div>`;}).join('')}</div>`;
const KLT=()=>M('\\beta\\, \\log \\frac{\\pi}{\\pi_{ref}}');   // the kit sets the thin space after log (2026-10-09)
const J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const live=id=>{const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; return !!l&&(c.labStatus?c.labStatus(l):l.st)==='live';};
const labRef=(id,en,zh)=>live(id)&&A.shell.labLink?tx(`“${A.shell.labLink(id)}”`,`“${A.shell.labLink(id)}”`):tx(`“${en}”`,`“${zh}”`);   // a lab named in a sentence, in quotes
const labA=(id,en,zh)=>live(id)&&A.shell.labLink?A.shell.labLink(id):esc(tx(en,zh));                                                        // a lab name that is itself the subject (GRPO, DPO)
const cite=(url,txt)=>`<a href="${url}" target="_blank" rel="noopener">${txt}</a>`;
const TRL='https://github.com/huggingface/trl/blob/59c4a8e104413fa9f4ca1a54eaf2ff93c0f299be/trl/experimental/ppo/';   // tag v1.12.0
const VERL='https://github.com/volcengine/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/';
const ORLHF='https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/README.md';
const NIMPL='https://github.com/huggingface/blog/blob/998ee73e0594a21b42df14632263266a2831dafe/the_n_implementation_details_of_rlhf_with_ppo.md';
const SPIN='https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/docs/algorithms/ppo.rst';

/* ---------- shared definitions (brief S2) ---------- */
const D_POL=()=>tx('The policy is the model being trained: it writes the responses. In this lab it is a small model with 16 parameters that, for each prompt, picks one of 5 candidate responses with some probability.','策略模型是正在训练、负责生成回答的模型。在这个 Lab 里，它是一个只有 16 个参数的小模型：对每道提问，按一定概率从 5 种候选回答里选一个。');
const D_REF=()=>tx('The reference model is a frozen copy of the policy taken when training starts, usually the model after SFT. It is never updated; it is only used to compute the KL.','参考模型是训练开始时策略模型的一份冻结副本，通常就是 SFT 之后的模型。它不再更新，只用来算 KL。');
const D_RM=()=>tx('The reward model reads “prompt + response” and gives one score. It was trained on preference pairs. Its score has no unit: only differences mean something.','奖励模型读“提问 + 回答”，给出一个分数，是用偏好对训练出来的。分数没有单位，只有差值有意义。');
const D_VAL=()=>tx('The value model, also called the critic, estimates how much reward the policy will get on average from this point on. It is trained together with the policy. In this lab a response is one choice, so it estimates one number per prompt: the average reward of that prompt’s responses.','价值模型（也叫 critic）估计“从这个位置往后，平均还能拿到多少奖励”，和策略模型一起训练。在这个 Lab 里一个回答就是一次选择，所以它对每道提问只估一个数：这道题的回答平均能拿多少奖励。');
const D_ADV=()=>tx('Advantage = reward − baseline: how much better than expected this response is. In PPO the baseline comes from the value model.','优势 = 奖励 − 基准：这个回答比预期好多少。PPO 的基准来自价值模型。');
const D_KL=()=>tx('The KL penalty measures how far the policy’s output has moved from the reference model and, multiplied by β, takes it out of the reward: each response’s reward loses '+KLT()+'. β is the KL coefficient: the larger it is, the more reward a response loses for the same distance.','KL 惩罚衡量策略模型的输出离参考模型有多远，乘上 β 以后从奖励里扣掉，也就是每个回答的奖励扣掉 '+KLT()+'。β 叫 KL 系数：β 越大，偏离同样多，扣的奖励越多。');
const D_CLIP=()=>tx('Clipping: once a response has moved far enough the way its advantage wants, moving further no longer raises the objective, so this batch stops pushing it that way. With a positive advantage it is no longer pushed up once its probability ratio is above 1.2. With a negative advantage it is no longer pushed down once the ratio is below 0.8. A response that moved the other way is still pushed back. ε is the clip range, 0.2 here: the ratio may move 0.2 either side of 1, within '+M('[1 - ε,\\ 1 + ε] = [0.8,\\ 1.2]')+'.','截断：一个回答已经朝优势要的方向走得够远，再走也不会让目标变大，这一批回答就不再往那个方向推它。优势为正的回答，概率之比超过 1.2 就不再往上推。优势为负的回答，概率之比低于 0.8 就不再往下压。往反方向走出去的回答，还会被推回来。ε 叫截断范围，这里是 0.2：比值可以在 1 的两边各走 0.2，也就是在 '+M('[1 - ε,\\ 1 + ε] = [0.8,\\ 1.2]')+' 里。');
const D_ROLL=()=>tx('Generating responses (a rollout): in RL training the policy writes its own responses, which are then scored. Here each prompt gets 4 per step.','生成回答（rollout）：RL 训练里让策略模型自己写回答，再拿去打分。这里每一步每道题写 4 个。');
const D_RH=()=>tx('Reward hacking: the policy finds responses the reward model overrates, so the score keeps rising while the responses get worse.','reward hacking：策略模型找到了奖励模型高估的回答，分数一直涨，回答其实在变差。');
const D_TQ=()=>tx('True quality is the lab’s own score for each kind of response, from 0 to 1, standing in for what careful people would say. Real training cannot see it; here it shows whether the reward model is right.','真实质量是这个 Lab 给每种回答定的分，0 到 1，代表认真的人会怎么评价。真实训练时看不到它；这里用它检查奖励模型说得对不对。');

/* ---------- module state ---------- */
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let view='rq', memAlg='ppo', saved=null, chSubmitted=false, chReveal=false, annT=null, stageT=0, tzKL=1;   // tzKL: the tile's setting (0 score only, 1 + value model, 2 + KL penalty)
let tried={};               // per part: the options the reader picked
let clipPass=0;             // part 6's picture: which pass of step 1's batch it shows (1–4), or 0 for after the last pass
let runSpd='med', loopSpd='med', passSpd='med';
/* a whole run plays in a fixed time, whatever its length (the owner's review of batch 9, item 4; rule G6): about 3 s at the default
   Medium and 1.5 s at Fast. A run that Scenarios or Settings start never takes longer than Fast (QUICK_SECS); Replay keeps the speed. */
const RUN_SECS={slow:6,med:3,fast:1.5}, QUICK_SECS=1.5;
const LOOP_SPEEDS={slow:3.5,med:2,fast:0.8};       // seconds per phase of the one-step replay
const PASS_SPEEDS={slow:2,med:1,fast:0.5};         // seconds per pass of part 6's batch (pass 1 … K, then “after”)
const watchedRun=new Set(), watchedLoop=new Set(), watchedPass=new Set();
let RUN={on:false,raf:0,last:0,acc:0,to:0}, LOOP={on:false,raf:0,start:0,ph:3,k:0,t0:0}, PASS={on:false,raf:0,start:0,i:0,fr:[],key:''};
let nRunStarts=0, nLoopStarts=0, nPassStarts=0, userAct=false;
const shownPrev={};

/* ---------- copy: header ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. Watch one training step, predict, then add the value model, the KL penalty and clipping in turn and see what each fixes.','一共 8 个部分。先看一步训练、做一道预测，再依次加上价值模型、KL 惩罚和截断，看每一样解决什么问题。'),
  sandbox:tx('Pick a situation and watch the small policy model train for 200 steps.','选一种情况，看这个小策略模型训练 200 步。'),
  challenge:tx('With a large step and every batch used 4 times, make true quality after 200 steps beat the SFT model with a KL of at most 0.5 nats.','步长较大、每批回答学 4 遍时，让训练 200 步后的真实质量超过 SFT 模型，而且 KL 不超过 0.5 nats。')});
const OUTCOME=()=>tx('explain what the value model and the KL penalty each fix in PPO, and size its four models’ memory.','讲清 PPO 里价值模型和 KL 惩罚各解决什么问题，并算出四个模型要多少显存。');
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser trains a small policy model (16 parameters) with real policy-gradient steps.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>3 prompts with 5 candidate responses each (15 in all). The policy’s probability for a response is the softmax of its own logit plus w × its length, where w is one parameter shared by all prompts: 15 logits and 1 shared weight, 16 parameters.</li><li>Each step generates 4 responses per prompt (seeded sampling, mulberry32, seed 7), scores them with the small reward model (one fixed score per response), subtracts ${KLT()}, and takes K passes of gradient ascent on the clipped objective (ε = 0.2). The value model is one number per prompt; it starts at the SFT model’s average score and moves halfway to each batch’s mean reward.</li><li>Memory: Llama‑3.1‑8B, 18 bytes per trained parameter and 2 per frozen BF16 parameter, GB = 10⁹ bytes, an H100’s 80 GB read as 80 × 10⁹ bytes.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>A response is one choice, not a sequence of tokens, so per-token and per-response quantities coincide: the KL penalty, the value model’s estimate and the advantage are one number per response. Real PPO computes them per token, with GAE.</li><li>A single choice’s log-ratio can only reach a few nats; a real response’s is a sum over hundreds of tokens. So in this lab the scores of one prompt’s responses differ by only a few hundredths. That keeps what β = 0.05 takes off about as large as the score differences, as in real training. This lab’s β and step sizes do not carry over to a real model.</li><li>The scores, true qualities, lengths and SFT probabilities are illustrative. No reward or advantage whitening, no minibatches, no value clipping, no entropy bonus.</li></ul></div><div><div class="eyebrow">Sources</div><ul><li>Clipped objective: Schulman et al., 2017 (arXiv 1707.06347). GAE: arXiv 1506.02438. RLHF with PPO: Ouyang et al., 2022 (arXiv 2203.02155).</li><li>TRL v1.12.0 (the last release with a PPO trainer, under ${cite(TRL+'ppo_config.py','trl/experimental/ppo')}): kl_coef 0.05, cliprange 0.2, 4 PPO epochs, vf_coef 0.1, γ 1.0, λ 0.95. verl: clip_ratio 0.2, 1 epoch, KL penalty off by default, and then no reference model is loaded (kl_coef 0.001 when on). OpenRLHF: the policy (actor), reward, reference and value (critic) models; 80% of RLHF time is generation (${cite(ORLHF,'README')}).</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>页面在你的浏览器里，对一个 16 个参数的小策略模型真的做策略梯度训练。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>3 道提问，每道 5 种候选回答（共 15 个）。策略模型给一个回答的概率，是它自己的 logit 加上 w × 回答长度之后的 softmax；w 是所有提问共用的一个参数。一共 15 个 logit 加 1 个共用参数，16 个参数。</li><li>每一步每道题生成 4 个回答（带种子的采样，mulberry32，种子 7），用小奖励模型打分（每种回答一个固定分数），扣掉 ${KLT()}，再在截断的目标上做 K 遍梯度上升（ε = 0.2）。价值模型对每道题只有一个数，从 SFT 模型的平均分开始，每一步向这一批的平均奖励走一半。</li><li>显存：Llama‑3.1‑8B，训练中的参数每个 18 字节，冻结的 BF16 参数每个 2 字节；GB 是 10⁹ 字节，H100 的 80 GB 按 80 × 10⁹ 字节算。</li></ul></div><div><div class="eyebrow">简化和假设</div><ul><li>一个回答只是一次选择，不是 token 序列，所以按 token 算和按回答算是一回事：KL 惩罚、价值模型的估计和优势，每个回答都只有一个数。真实的 PPO 按 token 算，并用 GAE 估优势。</li><li>一次选择的 log 比值最多只有几 nats，真实回答的 log 比值是几百个 token 的和，大得多。所以在这个 Lab 里，同一道题各回答的分数只差零点零几。这样 β = 0.05 扣掉的奖励才和分数差差不多大，和真实训练里一样。这个 Lab 里的 β 和步长不能照搬到真实模型。</li><li>分数、真实质量、长度和 SFT 模型的概率都是示意数值。没有奖励和优势的标准化，没有 minibatch，没有价值裁剪，没有熵奖励。</li></ul></div><div><div class="eyebrow">来源</div><ul><li>截断的目标：Schulman 等，2017（arXiv 1707.06347）。GAE：arXiv 1506.02438。用 PPO 做 RLHF：Ouyang 等，2022（arXiv 2203.02155）。</li><li>TRL v1.12.0（最后一个带 PPO 训练器的版本，在 ${cite(TRL+'ppo_config.py','trl/experimental/ppo')}）：kl_coef 0.05，cliprange 0.2，PPO 训练 4 遍（num_ppo_epochs），vf_coef 0.1，γ 1.0，λ 0.95。verl：clip_ratio 0.2，1 遍，默认不加 KL 惩罚，这时也不加载参考模型（打开时 kl_coef 0.001）。OpenRLHF：策略模型（actor）、奖励模型、参考模型和价值模型（critic）四个模型，RLHF 训练 80% 的时间花在生成上（${cite(ORLHF,'README')}）。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>RLHF has three stages. First, SFT teaches the base model the question-and-answer format (see ${labRef('sft','SFT','SFT')}). Then a reward model learns from preference pairs which responses people prefer (see ${labRef('reward-model','Reward model','奖励模型')}). Last, PPO trains the chat model against that reward model: this lab. ${labRef('dpo','DPO','DPO')} skips the reward model and trains the chat model on the preference pairs directly.</p>`,
 `<p>RLHF 分三个阶段。先用 SFT 教会基础模型一问一答的格式，见${labRef('sft','SFT','SFT')}。再用偏好对训练奖励模型，让它学会人更喜欢哪种回答，见${labRef('reward-model','Reward model','奖励模型')}。最后用 PPO 拿奖励模型的分数训练聊天模型，就是这个 Lab。${labRef('dpo','DPO','DPO')}跳过奖励模型，直接用偏好对训练聊天模型。</p>`);

/* ---------- guided parts ---------- */
const PRED=['best','noise','still'];
const predTxt=v=>({best:tx('The highest-scoring response steadily gains and the others slowly lose','分数最高的回答会稳稳地变多，别的慢慢变少'),
  noise:tx('Which one gains depends mostly on how often each happens to be written early on; another seed gives another result','哪种回答变多，主要看它开头碰巧被写出几次，换个种子结果就不一样'),
  still:tx('The scores are all about the same, so are the pushes, and the probabilities barely change','分数都差不多，推力也差不多，概率几乎不变')}[v]);
const P_LOOP={base:0,beta:0,K:1,clip:0,eta:'n'};
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'You have a reward model. How do you use its score to train the chat model?',zh:'奖励模型已经有了，怎么用它的分数训练聊天模型？'},enter:()=>({...P_LOOP,steps:0}),view:'rq',
  body:()=>tx(`<p>The previous lab, ${labRef('reward-model','Reward model','奖励模型')}, trained a ${DEF('reward model',D_RM())} that scores responses. The last stage of RLHF uses this score to improve the chat model.</p><p>It does so with reinforcement learning (RL). RL trains a model by letting it try, scoring each try with a reward, and making the high-scoring behaviour more likely. PPO is one RL algorithm, the classic one for RLHF: InstructGPT was trained with it.</p><p>This stage uses three models. The ${DEF('policy',D_POL())} is being trained and writes the responses. The ${DEF('reference model',D_REF())} is a copy of the policy made before training starts, and it is never updated. The reward model scores the responses. Part 4 adds a fourth model.</p><p>This lab trains a small policy model with only 16 parameters. It answers 3 prompts, and each prompt has 5 kinds of candidate response: A to E ${J('bars','in the chart')}. The dashed outlines in the chart are how often the SFT model writes each response; how the SFT model is trained is in the lab ${labRef('sft','SFT','SFT')}. We also know each response’s ${DEF('true quality',D_TQ())}, so we can check whether the reward model can be trusted.</p>`,
    `<p>上一个 Lab${labRef('reward-model','Reward model','奖励模型')}训练出了一个会打分的${DEF('奖励模型',D_RM())}。RLHF 的最后一个阶段，就是用这个分数去改进聊天模型。</p><p>这一步用的是强化学习（reinforcement learning，简称 RL）。RL 训练模型的办法是让它自己去试，给每次尝试打一个分，叫作奖励，再让得分高的做法更常出现。PPO 是一种 RL 算法，也是 RLHF 最经典的做法，InstructGPT 就是用它训练的。</p><p>这个阶段要用到三个模型。${DEF('策略模型',D_POL())}正在训练，负责写回答。${DEF('参考模型',D_REF())}是训练开始前复制下来的一份策略模型，之后不再更新。奖励模型负责给回答打分。第 4 部分还会再加一个模型。</p><p>这个 Lab 训练一个只有 16 个参数的小策略模型。它要回答 3 道提问，每道题有 5 种候选回答，就是${J('bars','图里')}的 A 到 E。图里的虚线框，是 SFT 模型写出每种回答的概率。SFT 模型怎么训练出来，见${labRef('sft','SFT','SFT')}。我们还知道每种回答的${DEF('真实质量',D_TQ())}，可以用它检查奖励模型靠不靠得住。</p>`)},
 {id:'loop',title:{en:'One RL step',zh:'一步 RL 训练'},lead:{en:'Generate, score, update: watch step 1.',zh:'生成、打分、更新：看第 1 步。'},enter:()=>({...P_LOOP,steps:0}),view:'rq',expect:()=>S.steps>=1,showMe:'loop',label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Every RL step does three things. First ${DEF('generate responses',D_ROLL())}: the policy writes 4 responses to each prompt. Then score: the reward model gives each response a score. Last, update: the policy changes according to the scores.</p><p>The simplest update rule is called <b>push by the score</b>. Each response that was written has its probability pushed up by its score: the higher the score, the bigger the push.</p>${loopBtns()}<p class="small muted">How a model writes a response from its probabilities, token by token, is in the lab ${labRef('sampling','Sampling','采样')}.</p>`,
    `<p>RL 训练的每一步做三件事。先是${DEF('生成回答',D_ROLL())}，策略模型给每道题写 4 个回答。然后打分，奖励模型给每个回答一个分数。最后更新，按分数修改策略模型。</p><p>最简单的更新规则叫<b>按分数加强</b>。写出来的每个回答，都按它的分数把概率往上推，分数越高，推得越多。</p>${loopBtns()}<p class="small muted">模型怎么按概率一个 token 一个 token 地写出回答，见${labRef('sampling','Sampling','采样')}。</p>`),
  obs:r=>obsLoop(r)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Every response to the easy prompt scores above 0.8. What happens?',zh:'简单那道题的回答都在 0.8 分以上，会怎样？'},enter:()=>({...P_LOOP,steps:1}),view:'rq',gate:()=>!!pred,skip:true,
  body:()=>{const lo=Math.min(...RM[0]), hi=Math.max(...RM[0]);
   const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(v=>`<button type="button" class="choice" data-pred="${v}" aria-pressed="${pred===v}">${esc(predTxt(v))}</button>`).join('')}</div>`;
   return tx(`<p>The easy prompt asks how to say “thank you” in Spanish. The reward model scores all 5 of its responses between ${f2(lo)} and ${f2(hi)}: all high, at most ${f2(hi-lo)} apart. Train 30 more steps with push by the score. How will this prompt’s 5 responses change?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
     `<p>简单的那道题问“谢谢”用西班牙语怎么说。奖励模型给它 5 种回答的分数都在 ${f2(lo)} 到 ${f2(hi)} 之间，个个都高，彼此最多差 ${f2(hi-lo)}。照“按分数加强”再训练 30 步，这道题的 5 种回答会怎样变？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'value',title:{en:'The value model and the advantage',zh:'价值模型和优势'},lead:{en:'Subtract what was expected: the value model.',zh:'减去“预期能拿多少分”：价值模型。'},enter:()=>({...P_LOOP,steps:30}),view:'fan',levers:'base',expect:()=>!!(tried.value&&tried.value.has(0)&&tried.value.has(1)),showMe:{base:1},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>The trouble is the raw score. On an easy prompt every response scores high, on a hard one every response scores low; the score alone does not say whether a response is good. What matters is how much better than expected it is.</p><p>The ${DEF('value model',D_VAL())} estimates that expectation, one baseline for each prompt. The ${DEF('advantage',D_ADV())} is</p>${EQ(tx('advantage','优势')+' '+M('\\hat{A}'),['R - V(x)'])}<p>A response better than expected has a positive advantage and is pushed up. A worse one has a negative advantage and is pushed down. The chart shows true quality for 6 random seeds, 30 steps each.</p>${levers('base')}`,
    `<p>问题出在“看绝对分数”。简单的题每个回答都得高分，难的题都得低分，分数本身说明不了回答好不好。要看的是它比预期好多少。</p><p>${DEF('价值模型',D_VAL())}就是估这个预期的，它对每道题估一个基准。${DEF('优势',D_ADV())}是</p>${EQ(tx('advantage','优势')+' '+M('\\hat{A}'),['R - V(x)'])}<p>比预期好的回答，优势为正，往上推。比预期差的回答，优势为负，往下压。图里画的是 6 个随机种子各训练 30 步的真实质量。</p>${levers('base')}`),
  obs:r=>obsValue(r)},
 {id:'kl',title:{en:'Reward hacking and the KL penalty',zh:'Reward hacking 和 KL 惩罚'},lead:{en:'What happens without a KL penalty?',zh:'不加 KL 惩罚会怎样？'},enter:()=>({base:1,beta:0,K:1,clip:0,eta:'n',steps:200}),view:'rq',levers:'beta',expect:()=>!!(tried.kl&&tried.kl.has(0)&&tried.kl.has(0.05)),showMe:{beta:0.05},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Now keep the value model and train for 200 steps. This reward model has a gap: it likes long responses and scores the padded response E above the concise A. Real reward models have gaps too: they are only reliable on responses like the ones they were trained on. When the policy learns to exploit such a gap, it is called ${DEF('reward hacking',D_RH())}. The chart shows true quality for 6 random seeds, 200 steps each.</p><p>The ${DEF('KL penalty',D_KL())} takes reward off each response by how far the policy has moved from the reference model. The reward after the penalty is</p>${EQ(tx('reward','奖励')+' '+M('R'),['r - \\beta\\, \\log \\frac{\\pi(y|x)}{\\pi_{ref}(y|x)}'])}<p class="small">${tx(`r is the reward-model score, ${M('\\pi(y|x)')} the probability the policy gives response y to prompt x, and ${M('\\pi_{ref}(y|x)')} the one the reference model, the SFT model, gives.`,`r 是奖励模型的分数，${M('\\pi(y|x)')} 是策略模型写出提问 x 的回答 y 的概率，${M('\\pi_{ref}(y|x)')} 是参考模型（SFT 模型）写出它的概率。`)}</p><p>The more often the policy writes a response than the SFT model did, the more reward that response loses. β = 0.05 is this lab’s default.</p>${levers('beta')}`,
    `<p>保留价值模型，训练 200 步。这个奖励模型有个漏洞，它偏爱长回答，给注水的 E 打的分比简洁准确的 A 还高。真实的奖励模型也会有漏洞，它只在和训练数据相似的回答上靠得住。策略模型学会利用这种漏洞，就叫 ${DEF('reward hacking',D_RH())}。图里画的是 6 个随机种子各训练 200 步的真实质量。</p><p>${DEF('KL 惩罚',D_KL())}按策略模型偏离参考模型的程度，从每个回答的奖励里扣分。扣完之后的奖励是</p>${EQ(tx('reward','奖励')+' '+M('R'),['r - \\beta\\, \\log \\frac{\\pi(y|x)}{\\pi_{ref}(y|x)}'])}<p class="small">${tx(`r is the reward-model score, ${M('\\pi(y|x)')} the probability the policy gives response y to prompt x, and ${M('\\pi_{ref}(y|x)')} the one the reference model, the SFT model, gives.`,`r 是奖励模型的分数，${M('\\pi(y|x)')} 是策略模型写出提问 x 的回答 y 的概率，${M('\\pi_{ref}(y|x)')} 是参考模型（SFT 模型）写出它的概率。`)}</p><p>策略模型越是比 SFT 模型更常写某个回答，这个回答被扣的奖励就越多。这个 Lab 默认 β = 0.05。</p>${levers('beta')}`),
  obs:r=>obsKL(r)},
 {id:'clip',title:{en:'Clipping',zh:'截断'},lead:{en:'How can one batch be used 4 times without running off?',zh:'同一批回答学 4 遍，怎样才不跑飞？'},enter:()=>({base:1,beta:0.05,K:1,clip:0,eta:'l',steps:200}),view:'rho',levers:'clip',expect:()=>!!(tried.clip&&tried.clip.has('4-0')&&tried.clip.has('4-1')),showMe:{K:4,clip:1},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Generating responses is slow: OpenRLHF says RLHF spends 80% of its time on it. So each PPO training step has two halves. First the policy writes one batch of responses. Then it learns from that same batch several times, one update per pass. The next step writes a new batch.</p><p>The step size is how far one pass moves the parameters. This part doubles it, from 8 to 16, so that the problem without clipping is easy to see.</p><p>From the second pass on, the policy has already changed. The probability ratio ρ says how many times as likely one response is now as when the batch was written.</p>${EQ(tx('probability ratio','概率之比')+' '+M('\\rho'),['\\frac{\\pi_{\\theta}(y|x)}{\\pi_{old}(y|x)}'])}<p class="small">${tx(`${M('\\pi_\\theta')} is the policy now, ${M('\\pi_{old}')} the policy that wrote the batch.`,`${M('\\pi_\\theta')} 是现在的策略模型，${M('\\pi_{old}')} 是写出这批回答时的策略模型。`)}</p><p><b>${DEF('Clipping',D_CLIP())} (ε = 0.2) does not keep ρ inside [0.8, 1.2].</b> Once a response’s ρ has left that range the way its advantage pushes it, this batch no longer pushes it: its gradient is 0. The other responses keep learning.</p><p>For example, a response better than expected has a positive advantage. Once its ρ reaches 1.3, it is no longer pushed up. A response worse than expected has a negative advantage. Once its ρ falls to 0.7, it is no longer pushed down.</p><p>A response that left the other way is still pushed back. If other responses squeezed a positive-advantage response down to 0.7, it is still pushed up. If a negative-advantage response rose to 1.3, it is still pushed down.</p><p>As a formula, every pass makes the objective L larger:</p>${mathLine('L = \\min(\\rho A, \\text{clip}(\\rho, 1-\\varepsilon, 1+\\varepsilon) A)')}${levers('clip')}`,
    `<p>生成回答很慢，OpenRLHF 说它占了 RLHF 训练 80% 的时间。所以 PPO 的每一步训练分两段。先让策略模型写一批回答，再用这同一批回答学几遍，每学一遍更新一次参数。下一步再重新写一批。</p><p>步长是每学一遍时参数被推多远。这一部分把步长从 8 调到 16，不截断时的问题更容易看清。</p><p>从第 2 遍起，策略模型已经变了。概率之比 ρ 看的是，同一个回答的概率现在是写出这批回答时的几倍。</p>${EQ(tx('probability ratio','概率之比')+' '+M('\\rho'),['\\frac{\\pi_{\\theta}(y|x)}{\\pi_{old}(y|x)}'])}<p class="small">${tx(`${M('\\pi_\\theta')} is the policy now, ${M('\\pi_{old}')} the policy that wrote the batch.`,`${M('\\pi_\\theta')} 是现在的策略模型，${M('\\pi_{old}')} 是写出这批回答时的策略模型。`)}</p><p><b>${DEF('截断',D_CLIP())}（ε = 0.2）不会把 ρ 留在 [0.8, 1.2] 里。</b>一个回答的 ρ 朝优势推它的方向走出这个区间以后，这一批就不再推它，它的梯度是 0。其余的回答照样学。</p><p>比如一个比预期好的回答，优势为正。它的 ρ 到了 1.3，就不再往上推。一个比预期差的回答，优势为负。它的 ρ 降到 0.7，就不再往下压。</p><p>往反方向走出去的回答，还会被推回来。优势为正的回答被别的回答挤到 0.7，还会被往上推。优势为负的回答反而涨到 1.3，还会被往下压。</p><p>写成公式，PPO 每学一遍都让目标 L 变大。</p>${mathLine('L = \\min(\\rho A, \\text{clip}(\\rho, 1-\\varepsilon, 1+\\varepsilon) A)')}${levers('clip')}`),
  obs:r=>obsClip(r)},
 {id:'cost',title:{en:'The cost: four models',zh:'代价：四个模型'},lead:{en:'For an 8B model, the four models alone take 321.2 GB.',zh:'8B 模型的话，光四个模型就要 321.2 GB。'},enter:()=>({...DEFAULT}),view:'rq',levers:'alg',
  body:()=>{const m=memOf('ppo'), g=memOf('grpo'), d=memOf('dpo');
   return tx(`<p>In the classic setup the KL penalty is on, as it is by default in TRL. Then all four models sit on the GPUs. The policy and the value model are trained; the reference and the reward model are frozen.</p><p>For Llama‑3.1‑8B, a model being trained takes 18 bytes per parameter: BF16 weights, FP32 gradients, an FP32 copy of the weights, and Adam’s m and v (see ${labRef('training-memory','Training memory','训练显存')} and ${labRef('optimizers','Optimizers','优化器')}). A frozen model takes only 2 bytes per parameter. The four models alone take ${f(m.tot,1)} GB, at least ${m.gpus} H100s of 80 GB. That is before activations and the KV cache used while generating responses (see ${labRef('kv-basics','Why a KV cache','为什么要 KV cache')}).</p>${levers('alg')}`,
     `<p>按经典做法，KL 惩罚是打开的，TRL 默认也是这样。这时四个模型都要放在 GPU 上。策略模型和价值模型在训练，参考模型和奖励模型冻结。</p><p>按 Llama‑3.1‑8B 算，在训练的模型每个参数占 18 字节，包括 BF16 权重、FP32 梯度、FP32 权重副本，以及 Adam 的 m 和 v，见${labRef('training-memory','Training memory','训练显存')}和${labRef('optimizers','Optimizers','优化器')}。冻结的模型每个参数只占 2 字节。四个模型加起来，光模型本身就要 ${f(m.tot,1)} GB，至少要 ${m.gpus} 张 80 GB 的 H100。这还没算激活和生成回答时的 KV cache，见${labRef('kv-basics','Why a KV cache','为什么要 KV cache')}。</p>${levers('alg')}`);},
  obs:()=>obsCost()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},enter:()=>({...DEFAULT}),view:'rq',
  body:()=>{const m=memOf('ppo');
   return tx(`<ul><li>The value model gives each prompt a baseline: what its responses usually score. Only how much a response beats that baseline, its advantage, decides the push. Without it, where training goes depends mostly on which responses happen to be written. With it, another seed gives about the same result.</li><li>The KL penalty takes more reward off the further the policy moves from the reference model. Without it the reward-model score keeps rising while true quality can fall, because the policy learns to exploit the reward model’s gaps, such as padding. With it the policy stays near the reference model, which in real training is also where the reward model can still be trusted.</li><li>Clipping lets one batch be used several times without the passes adding up. A response that has left the band the way its advantage pushes it is no longer pushed; the others keep learning.</li><li>The price of PPO is four models at once, and generating responses takes a lot of time. For Llama‑3.1‑8B the models alone take ${f(m.tot,1)} GB. So later methods drop some of them: GRPO needs no value model, and DPO needs no reward model and generates nothing.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “PPO makes the reward as high as possible.” It maximizes the reward after the KL penalty:</p>${EQ(tx('reward after the KL penalty','扣掉 KL 惩罚后的奖励'),['r - \\beta \\, \\text{KL}(\\pi ‖ \\pi_{ref})'])}<p>✗ “The value model is the reward model.” One scores a response; the other estimates how much is expected.</p><p>✗ “Clipping keeps ρ inside [0.8, 1.2].” It holds ρ nowhere. It only stops pushing a response that has already left the band the way its advantage pushes it, and the other responses keep learning.</p></div>${A.shell.toScenarios()}`,
     `<ul><li>价值模型给每道题一个基准，就是这道题的回答通常能拿多少分。决定推力的，只是回答比基准高出多少，也就是优势。没有它，训练往哪走，主要看碰巧写出了哪些回答。有了它，换个种子，结果也差不多。</li><li>有了 KL 惩罚，策略模型离参考模型越远，扣的奖励就越多。没有它，奖励模型的分数会一路涨，真实质量却可能跌下去，因为策略模型会去钻奖励模型的漏洞，比如注水。有了它，策略模型留在参考模型附近。真实训练里，那里也正是奖励模型还靠得住的地方。</li><li>截断让同一批回答可以多学几遍，又不会越推越远。朝优势方向走出区间的回答不再被推，其余的照样学。</li><li>PPO 的代价是同时放四个模型，生成回答还要花大量时间。Llama‑3.1‑8B 光模型就要 ${f(m.tot,1)} GB。所以后面的方法都想办法省掉一些。GRPO 不要价值模型，DPO 连奖励模型和生成回答都省了。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “PPO 就是让奖励越高越好。”它优化的是扣掉 KL 惩罚之后的奖励，也就是</p>${EQ(tx('reward after the KL penalty','扣掉 KL 惩罚后的奖励'),['r - \\beta \\, \\text{KL}(\\pi ‖ \\pi_{ref})'])}<p>✗ “价值模型就是奖励模型。”一个给回答打分，一个估计预期能拿多少。</p><p>✗ “截断会把 ρ 限制在 [0.8, 1.2] 里。”它不管 ρ 停在哪里，只是不再推已经朝优势方向走出区间的回答，其余的回答照样学。</p></div>${A.shell.toScenarios()}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
/* levers: single choice, each from the same starting point (R3.6), the result kept on the title row */
const LEV={
 base:{opts:[{v:0,t:{en:'Push by the score',zh:'只用奖励'},d:{en:'No baseline: the advantage is the reward itself',zh:'没有基准：优势就是奖励本身'}},{v:1,t:{en:'Add the value model',zh:'加上价值模型'},d:{en:'Advantage = reward − the value model’s baseline',zh:'优势 = 奖励 − 价值模型给的基准'}}],
   cur:()=>S.base,set:v=>({base:v}),res:v=>{const qs=fanQ({...S,base:v},30); return tx(`true quality after 30 steps: ${f2(Math.min(...qs))}–${f2(Math.max(...qs))}`,`30 步后真实质量 ${f2(Math.min(...qs))}–${f2(Math.max(...qs))}`);}},
 beta:{opts:[{v:0,t:{en:'No KL penalty',zh:'不加 KL 惩罚'},d:{en:'β = 0: only the reward-model score counts',zh:'β = 0：只看奖励模型的分数'}},{v:0.05,t:{en:'KL penalty, β = 0.05',zh:'KL 惩罚，β = 0.05'},d:{en:'This lab’s default',zh:'这个 Lab 的默认值'}},{v:0.2,t:{en:'KL penalty, β = 0.2',zh:'KL 惩罚，β = 0.2'},d:{en:'Four times as strong',zh:'强 4 倍'}}],
   cur:()=>S.beta,set:v=>({beta:v}),res:v=>{const qs=fanQ({...S,beta:v},MAXS), lo=f2(Math.min(...qs)), hi=f2(Math.max(...qs)); return lo===hi?tx(`all 6 seeds: ${lo}`,`6 个种子都是 ${lo}`):tx(`true quality ${lo}–${hi}`,`真实质量 ${lo}–${hi}`);}},
 clip:{opts:[{v:'1-0',t:{en:'Each batch once',zh:'每批回答学 1 遍'},d:{en:'One pass, nothing to clip',zh:'只学一遍，用不着截断'}},{v:'4-0',t:{en:'4 passes, no clipping',zh:'学 4 遍，不截断'},d:{en:'Every pass pushes as hard as the first',zh:'每一遍都照样推'}},{v:'4-1',t:{en:'4 passes, clipped',zh:'学 4 遍，截断'},d:{en:'Out of the band the way its advantage pushes: no longer pushed',zh:'朝优势方向走出区间的回答，不再被推'}}],
   cur:()=>`${S.K}-${S.K===1?0:S.clip}`,set:v=>{const [k,c]=v.split('-'); return {K:+k,clip:+c};},res:v=>{const [k,c]=v.split('-'), mx=Math.max(...batch1({...S,K:+k,clip:+c}).map(d=>d.end)); return tx(`step 1’s batch ends with ρ up to ${ratioTxt(mx)}`,`第 1 步学完，ρ 最大 ${ratioTxt(mx)}`);}},
 alg:{opts:[{v:'ppo',t:{en:'PPO: four models',zh:'PPO：四个模型'},d:{en:'Policy, value model, reference, reward model',zh:'策略模型、价值模型、参考模型、奖励模型'}},{v:'grpo',t:{en:'GRPO: no value model',zh:'GRPO：不要价值模型'},d:{en:'The baseline is the mean reward of a group of responses',zh:'基准是同一组回答的平均奖励'}},{v:'dpo',t:{en:'DPO: no reward model',zh:'DPO：不要奖励模型'},d:{en:'Trains on preference pairs; generates nothing',zh:'直接在偏好对上训练，不生成回答'}}],
   cur:()=>memAlg,set:v=>({alg:v}),res:v=>`${f(memOf(v).tot,1)} GB`}
};
const ratioTxt=x=>{if(x<10) return f(x,2); const r=x>=1000?Math.round(x/100)*100:(x>=100?Math.round(x/10)*10:Math.round(x)); return x>=100?tx(`about ${f(r,0)} times`,`约 ${f(r,0)} 倍`):tx(`${f(r,0)} times`,`${f(r,0)} 倍`);};
/* a lever's result line appears only once the run that produces it has been watched to its end (or shown at once, reduced motion);
   while it plays the line says so (S2 of the review, R3.8). The memory lever has no run. */
function levRes(key,v){const lv=LEV[key], seen=tried[STEPS[step]&&STEPS[step].id]||new Set(); if(!seen.has(v)) return '';
  if(key==='alg') return esc(lv.res(v));
  const s={...S,...lv.set(v)}, rk=CFG.map(k=>s[k]).join('|')+'>'+runTarget();
  if(key==='clip'){const pk=passKey(s); if(PASS.on&&PASS.key===pk) return esc(tx('playing…','播放中…')); return watchedPass.has(pk)?esc(lv.res(v)):'';}
  if(RUN.on&&RUN.key===rk) return esc(tx('training…','训练中…'));
  return watchedRun.has(rk)?esc(lv.res(v)):'';}
/* G1 (the owner's message of 2026-10-08): each group of options opens with a lead-in that says what to pick and what to watch */
const LEAD={
 base:()=>tx('Pick an update rule and see how true quality changes for the 6 seeds.','选一种更新规则，看看 6 个种子的真实质量怎么变。'),
 beta:()=>tx('Pick a KL coefficient β and see whether true quality falls over 200 steps.','选一个 KL 系数 β，看看 200 步里真实质量会不会跌下去。'),
 clip:()=>tx(`Pick how each batch is used and watch, in ${J('clip','the top chart')}, which responses of step 1’s batch are still pushed at each pass.`,`选一种学法，在${J('clip','最上面的图')}里看第 1 步这批回答每学一遍，哪些还在被推。`),
 alg:()=>tx('Pick a method and see which models it keeps and how much memory they take.','选一种方法，看看它要放哪几个模型、一共占多少显存。')};
function levers(key){const lv=LEV[key], cur=lv.cur();
  if(!A.ui.levers) return '';
  return `<p>${LEAD[key]()}</p>`+A.ui.levers({aria:tx('Options: pick one','选项：选一个'),value:String(cur),
    options:lv.opts.map(o=>({v:String(o.v),title:L(o.t),desc:L(o.d),result:levRes(key,o.v)}))}).replace(/data-lever=/g,`data-lkey="${key}" data-lever=`);}
const loopBtns=()=>`<div class="ppo-tbtns"><button type="button" class="btn sm" data-act="loop-play">${LOOP.on?tx('❚❚ Pause','❚❚ 暂停'):(S.steps>=1&&!LOOP.on?tx('▶ Replay step 1','▶ 重放第 1 步'):tx('▶ Play this step','▶ 播放这一步'))}</button></div>`;

function obsLoop(r){
  if(r.t<1) return '';
  const T=r.T, B=T.B[0], mid=B.filter(b=>b.k===1);
  const cnt=[0,0,0,0,0]; mid.forEach(b=>{cnt[b.y]++;}); const top=cnt.indexOf(Math.max(...cnt));
  const once=cnt.findIndex((c,y)=>c===1&&T.pi[1][1][y]<T.pi[0][1][y]);
  const p0=pc(T.pi[0][1][top]), p1=pc(T.pi[1][1][top]);
  const onceEn=once>=0?` ${LET[once]} was written once, yet fell to ${pc(T.pi[1][1][once])}. The five probabilities always add up to 100%, so when ${LET[top]} gains this much the others must give way.`:'';
  const onceZh=once>=0?`${LET[once]} 只被写了 1 次，概率反而跌到 ${pc(T.pi[1][1][once])}。五个概率加起来总是 100%，${LET[top]} 涨了这么多，别的回答只能让出位置。`:'';
  return tx(`<p><b>Step 1: every response that was written got an upward push the size of its score, the lowest-scoring included.</b></p><p>All the scores are positive, so every response that was written is pushed up. On the medium prompt ${LET[top]} was written ${cnt[top]} times and went from ${p0} to ${p1}.${onceEn}</p>`,
    `<p><b>第 1 步：写出来的每个回答都得到一个向上的推力，大小就是它的分数，分数最低的也一样。</b></p><p>分数都是正数，所以写出来的回答全被往上推。中等那道题里，${LET[top]} 被写了 ${cnt[top]} 次，从 ${p0} 涨到 ${p1}。${onceZh}</p>`);}
function fanRange(base){const qs=FAN.map(sd=>traj({...P_LOOP,base,seed:sd}).q[30]); return [Math.min(...qs),Math.max(...qs)];}
/* which response the easy prompt has settled on after 30 steps of push by the score, for each of the 6 seeds (the evidence for the prediction) */
const easyTops=()=>FAN.map(sd=>LET[topOf(traj({...P_LOOP,base:0,seed:sd}).pi[30][0])]);
function obsValue(r){
  const [a0,a1]=fanRange(0), [b0,b1]=fanRange(1), q0=f2(Q0), et=easyTops();
  let h=S.base?tx(`<p><b>With the value model, the 6 seeds end between ${f2(b0)} and ${f2(b1)} after 30 steps, every one above the SFT model’s ${q0}.</b> Push by the score alone: ${f2(a0)} to ${f2(a1)}.</p>`,`<p><b>加上价值模型：6 个种子训练 30 步后，真实质量都在 ${f2(b0)} 到 ${f2(b1)} 之间，全都高于 SFT 模型的 ${q0}。</b>只用奖励时是 ${f2(a0)} 到 ${f2(a1)}。</p>`)
    :tx(`<p><b>Push by the score: after 30 steps the 6 seeds end anywhere from ${f2(a0)} to ${f2(a1)} (the SFT model: ${q0}).</b> On the easy prompt the 6 seeds have settled on ${et.join(', ')}. Now add the value model.</p>`,`<p><b>只用奖励：6 个种子训练 30 步后，真实质量从 ${f2(a0)} 到 ${f2(a1)} 都有，SFT 模型是 ${q0}。</b>简单那道题，6 个种子最后分别锁定在 ${et.join('、')}。再加上价值模型试试。</p>`);
  if(pred&&tried.value&&tried.value.has(0)&&tried.value.has(1)){const pt=esc(predTxt(pred));
    h+=pred==='noise'?tx(`<p>You predicted “${pt}”: right. On the easy prompt the 6 seeds settled on ${et.join(', ')}.</p>`,`<p>你预测的是“${pt}”，预测对了。简单那道题，6 个种子最后分别锁定在 ${et.join('、')}。</p>`)
      :tx(`<p>You predicted “${pt}”. The answer: “${esc(predTxt('noise'))}”. On the easy prompt the 6 seeds settled on ${et.join(', ')}.</p>`,`<p>你预测的是“${pt}”。正确答案是“${esc(predTxt('noise'))}”。简单那道题，6 个种子最后分别锁定在 ${et.join('、')}。</p>`);}
  h+=tx(`<p>Why: without a baseline every push is the size of the whole score, while the scores of the responses differ by only a few hundredths. So each step mostly reflects which responses happened to be written. With the baseline, a push is only the difference from the expectation: up for better than expected, down for worse.</p>`,
        `<p>原因：没有基准时，推力有整个分数那么大，回答之间的分数差却只有零点零几，所以每一步主要看碰巧写出了哪些回答。有了基准，推力只剩下和预期的差，比预期好的往上推，比预期差的往下压。</p>`);
  return h;}
function obsKL(r){
  const T=r.T, k=peakOf(T,MAXS), qs=fanQ(S,MAXS), lo=f2(Math.min(...qs)), hi=f2(Math.max(...qs));
  if(S.beta===0){const many=manyEnds(S), nlow=many.filter(q=>q<=0.3).length;
    return tx(`<p><b>No KL penalty: the reward-model score rose from ${f2(T.rm[0])} to ${f2(T.rm[MAXS])}, but true quality rose, then fell to ${f2(T.q[MAXS])}</b> (the red stretch in the chart). For seed 7 it peaked at ${f2(T.q[k])} near step ${k}. In the end the padded response E took over every prompt. The other 5 seeds went the same way: all 6 end between ${lo} and ${hi}.</p><p>Why: the reward model likes length, and w, the length weight all prompts share, kept growing.</p><p>Not every seed falls, though. Over 40 seeds, ${nlow} end at 0.30 or below and the rest anywhere from low to high. Real training cannot see true quality, so it cannot tell which kind of run it is in: it only sees a score that keeps rising. Now add the KL penalty.</p>`,
      `<p><b>不加 KL 惩罚：奖励模型的平均分从 ${f2(T.rm[0])} 涨到 ${f2(T.rm[MAXS])}，真实质量却先涨后跌，最后只有 ${f2(T.q[MAXS])}</b>（图里红色那段）。种子 7 的真实质量在第 ${k} 步左右最高，是 ${f2(T.q[k])}。最后三道题都被注水的 E 占满了。另外 5 个种子也是这样，6 个种子最后都在 ${lo} 到 ${hi} 之间。</p><p>原因：奖励模型偏爱长回答，三道题共用的长度参数 w 越涨越大。</p><p>不过，不是每个种子都会跌。试 40 个种子，有 ${nlow} 个最后跌到 0.30 以下，其余的有高有低。真实训练看不到真实质量，分不清这一次是哪一种，只看得到一直在涨的分数。再加上 KL 惩罚试试。</p>`);}
  const e=r.gap.reduce((s,x)=>s+x,0)/NP, e0=traj({...S,beta:0}).pi[MAXS].reduce((s,p)=>s+p[4],0)/NP, one=lo===hi;
  if(S.beta===0.05) return tx(`<p><b>β = 0.05: ${one?`true quality ends at ${lo} on all 6 seeds`:`true quality ends between ${lo} and ${hi} on the 6 seeds`}, a little above the SFT model’s ${f2(T.q[0])}, and stays there.</b> The policy writes the padded E ${pc(e)} of the time, against ${pc(REF[4])} for the SFT model and ${pc(e0)} with no KL penalty.</p><p>Why: writing E more often than the SFT model did now costs reward, and the further the policy moves, the more it costs. So in this lab the policy cannot move much of its probability to E. In real training there is one more reason: the responses near the reference model are the kind the reward model saw in training, where its scores can still be trusted.</p>`,
    `<p><b>β = 0.05：${one?`6 个种子的真实质量最后都是 ${lo}`:`6 个种子的真实质量最后在 ${lo} 到 ${hi} 之间`}，比 SFT 模型的 ${f2(T.q[0])} 略高，而且一直保持住。</b>策略模型写出注水的 E 的概率是 ${pc(e)}。SFT 模型是 ${pc(REF[4])}，不加 KL 惩罚时是 ${pc(e0)}。</p><p>原因：策略模型比 SFT 模型更常写 E，就要被扣奖励，偏离得越远，扣得越多。所以在这个 Lab 里，策略模型没法把概率大量挪到 E 上。真实训练里还有一层原因。参考模型附近的回答，正是奖励模型训练时见过的那类，它的打分在那里还靠得住。</p>`);
  return tx(`<p><b>β = 0.2: the policy barely moves. True quality ends at ${f2(T.q[MAXS])} and the KL at ${f(T.kl[MAXS],2)} nats.</b></p><p>A larger β holds the policy tighter. Reward hacking becomes less likely, but the policy also learns less. TRL v1.12.0 defaults to β = 0.05. verl turns the KL penalty off by default, and then loads no reference model; when the penalty is on, its coefficient is 0.001.</p>`,
    `<p><b>β = 0.2：策略模型几乎没动。真实质量最后是 ${f2(T.q[MAXS])}，KL 只有 ${f(T.kl[MAXS],2)} nats。</b></p><p>β 越大，拴得越紧，越不容易出现 reward hacking，可学到的也少。TRL v1.12.0 默认 β = 0.05。verl 默认不加 KL 惩罚，这时也不加载参考模型。打开 KL 惩罚时，verl 的系数是 0.001。</p>`);}
/* part 6's picture (review codex-b9v3, item 7): step 1's batch, each distinct response once (a prompt's response written twice has the
   same advantage and the same ratios), with its advantage, its ratio at the start of every pass, after the last one, and whether each
   pass pushed it. The state of a response at a pass: pushed inside the band, on the clipped plateau (not pushed), pushed back after
   moving the other way, or (without clipping) pushed further although it is already out of the band the way its advantage wants. */
function batch1(s){const T=traj(s), m=new Map();
  T.B[0].forEach(b=>{const id=b.k*NY+b.y; if(!m.has(id)) m.set(id,{id,k:b.k,y:b.y,a:b.a,rp:b.rp,on:b.on,end:b.rho,n:0}); m.get(id).n++;});
  return [...m.values()].sort((x,y)=>x.k-y.k||x.y-y.y);}
const backSide=(a,rho)=>(a>0&&rho<1-EPS)||(a<0&&rho>1+EPS);
function dotState(d,p){if(!p) return S.clip&&S.K>1?(d.on[S.K-1]?'endOn':'endFlat'):'end'; const rho=d.rp[p-1];
  if(!d.on[p-1]) return 'flat'; if(onPlateau(d.a,rho)) return 'bad'; if(backSide(d.a,rho)) return 'back'; return 'push';}
const respName=d=>tx(`the ${L(PROMPTS[d.k].d).toLowerCase()} prompt’s ${LET[d.y]}`,`${PROMPTS[d.k].d.zh}题的回答 ${LET[d.y]}`);
const outBy=rho=>rho>1+EPS?rho-1-EPS:(rho<1-EPS?1-EPS-rho:0);   // how far a ratio is outside [1 − ε, 1 + ε]
const timesTxt=x=>x<10?tx(`${f2(x)} times`,`${f2(x)} 倍`):ratioTxt(x);
const topEnd=D=>D.reduce((a,d)=>d.end>a.end?d:a);
/* part 6's observation (the owner's review of batch 9, item 3e): the 200-step outcome, then the one point the picture shows */
function obsClip(r){
  const T=r.T, D=batch1(S), ends=D.map(d=>d.end), lo=Math.min(...ends), hi=Math.max(...ends), one=traj({...S,K:1,clip:0});
  const res=tx(`after 200 steps true quality is ${f2(T.q[MAXS])} and the KL ${f(T.kl[MAXS],2)} nats`,`200 步后真实质量 ${f2(T.q[MAXS])}，KL ${f(T.kl[MAXS],2)} nats`);
  if(S.K===1) return tx(`<p><b>One pass per batch: ${res}.</b></p><p>In ${J('clip','the top chart')} each response gets only one pass. When that pass starts, the policy being trained is still the policy that wrote the batch, so every response’s ρ is 1, inside [0.8, 1.2], and clipping does not change the gradient. With the large step, that one pass already moves ratios out of [0.8, 1.2]: after step 1 they are between ${f2(lo)} and ${f2(hi)}. The batch is not learned from a second time, so clipping never gets to act. Now learn from each batch 4 times.</p>`,
    `<p><b>每批学 1 遍：${res}。</b></p><p>${J('clip','最上面的图')}里，每个回答只学一遍。这一遍开始时，正在训练的策略模型和写出这批回答的还是同一个，所以每个回答的 ρ 都是 1，在 [0.8, 1.2] 里面，截断不会改变梯度。步长大时，这一遍学完，比值已经被推出 [0.8, 1.2]，第 1 步学完时在 ${f2(lo)} 到 ${f2(hi)} 之间。这批回答不会再学第二遍，截断也就没有机会起作用。再试试每批学 4 遍。</p>`);
  if(!S.clip){const top=topEnd(D);
    return tx(`<p><b>4 passes, no clipping: ${res}, against ${f2(one.q[MAXS])} and ${f(one.kl[MAXS],2)} nats with one pass.</b></p><p>Look at step 1’s batch in ${J('clip','the top chart')}. A response already out of the band the way its advantage pushes it is pushed further out at every pass (red), and the four passes add up: after them, ${respName(top)} is ${timesTxt(top.end)} as likely as before. One step throws the policy far from the policy that wrote the batch.</p>`,
      `<p><b>学 4 遍、不截断：${res}。学 1 遍时是 ${f2(one.q[MAXS])} 和 ${f(one.kl[MAXS],2)} nats。</b></p><p>看${J('clip','最上面的图')}里第 1 步这一批。朝优势方向走出区间的回答，每一遍都还在被往外推（红色），四遍叠在一起越推越远。学完时，${respName(top)} 的概率被推到了原来的 ${timesTxt(top.end)}。策略模型一步就跳离了写出这批回答时的样子。</p>`);}
  return tx(`<p><b>4 passes, clipped: ${res}, the same as one pass.</b></p><p>Clipping did not keep ρ inside the band. In ${J('clip','the top chart')}, a response that has left it the way its advantage pushes it is no longer pushed (hollow), and the others keep learning (filled). So the 4 passes do not add up.</p>`,
    `<p><b>学 4 遍、截断：${res}，和学 1 遍时一样。</b></p><p>截断没有把 ρ 留在区间里。${J('clip','最上面的图')}里，朝优势方向走出区间的回答不再被推（空心），其余的照样学（实心），所以 4 遍不会叠在一起越推越远。</p>`);}
function obsCost(){const m=memOf(memAlg), p=memOf('ppo'), g0=memOf('grpo0');
  if(memAlg==='ppo') return tx(`<p><b>PPO: ${f(m.tot,1)} GB for the four models, at least ${m.gpus} H100s.</b> The two trained models take nine tenths of it.</p>`,`<p><b>PPO：四个模型 ${f(m.tot,1)} GB，至少 ${m.gpus} 张 H100。</b>其中九成是两个在训练的模型。</p>`);
  if(memAlg==='grpo') return tx(`<p><b>${labA('grpo','GRPO','GRPO')}: ${f(m.tot,1)} GB, at least ${m.gpus} H100s, against ${f(p.tot,1)} GB for PPO.</b> Dropping the value model removes a whole trained model. It still generates responses, several per prompt.</p><p>TRL’s GRPO has defaulted to β = 0 since v0.19.0. Then it does not load the reference model either, and needs only ${f(g0.tot,1)} GB.</p>`,`<p><b>${labA('grpo','GRPO','GRPO')}：${f(m.tot,1)} GB，至少 ${m.gpus} 张 H100。PPO 是 ${f(p.tot,1)} GB。</b>去掉价值模型，就少了一整个在训练的模型。它照样要生成回答，每道题生成好几个。</p><p>TRL 的 GRPO 从 v0.19.0 起默认 β = 0。这时连参考模型也不加载，只要 ${f(g0.tot,1)} GB。</p>`);
  return tx(`<p><b>${labA('dpo','DPO','DPO')}: ${f(m.tot,1)} GB, at least ${m.gpus} H100s.</b> It needs no reward model and no value model, and it generates nothing during training: it learns straight from preference pairs.</p>`,`<p><b>${labA('dpo','DPO','DPO')}：${f(m.tot,1)} GB，至少 ${m.gpus} 张 H100。</b>它不要奖励模型和价值模型，训练时也不生成回答，直接从偏好对学。</p>`);}

/* ---------- scenarios ---------- */
const PRESETS=[
 {t:{en:'Full PPO (initial setup)',zh:'完整的 PPO（初始设置）'},d:{en:'Value model, KL penalty and clipping all on',zh:'价值模型、KL 惩罚和截断都用上'},p:{...DEFAULT}},
 {t:{en:'Push by the score only',zh:'只用奖励'},d:{en:'No baseline: responses are pushed by their raw score',zh:'不减基准，按奖励模型的分数直接加强'},p:{base:0,beta:0,K:1,clip:0,eta:'n',steps:200}},
 {t:{en:'Drop the KL penalty',zh:'去掉 KL 惩罚'},d:{en:'β = 0; the value model and clipping stay on',zh:'β = 0，价值模型和截断照用'},p:{base:1,beta:0,K:4,clip:1,eta:'n',steps:200}},
 {t:{en:'Reuse each batch, no clipping',zh:'同一批回答多学几遍，不截断'},d:{en:'4 passes per batch, no clipping, and a larger step (16) so the run-off is easy to see',zh:'每批学 4 遍，不截断。步长调大到 16，跑飞看得更清楚'},p:{base:1,beta:0.05,K:4,clip:0,eta:'l',steps:200}}];
const presetNow=()=>PRESETS.findIndex(p=>CFG.every(k=>p.p[k]===S[k])&&p.p.steps===S.steps);

/* ---------- views ---------- */
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:'PPO',title:l?L(l.q):tx('Why does PPO need a value model and a KL penalty?','PPO 为什么需要价值模型和 KL 惩罚？'),
    minutes:15,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: RLHF','前置知识：RLHF'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const spdCtl=(key,cur)=>`<div class="ppo-speed"><span class="ppo-speed-l">${tx('Speed','速度')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="${key}">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${cur===v}">${x}</button>`).join('')}</div></div>`;
function stageHTML(){
  const lg=(tipx,cls,txt)=>`<span data-tipx="${esc(tipx)}" tabindex="0"><i class="ppo-sw ${cls}"></i><span>${txt}</span></span>`;
  return `<div class="panel stage ppo-stage" id="ppo-stage">`+
   `<div class="stage-head"><div class="ppo-st-title" id="ppo-st-title"></div><div class="stage-actions"><span class="status" id="ppo-status" tabindex="0"></span></div></div>`+
   `<div class="ppo-clip" id="ppo-clip" hidden></div>`+
   `<div class="ppo-loop" id="ppo-loop"><div class="ppo-loop-head"><div class="ppo-phases" id="ppo-phases"></div><div class="ppo-loop-ctl"><button type="button" class="btn ghost sm" data-act="loop-play" id="ppo-loop-play"></button>${spdCtl('loopSpd',loopSpd)}</div></div><div class="ppo-rows" id="ppo-rows"></div></div>`+
   `<div class="legend-row ppo-legend">${lg(tx('Filled bar: how often the policy writes this response now.','实心条：策略模型现在写出这个回答的概率。'),'ppo-sw-pi',tx('policy now','策略模型（现在）'))}${lg(tx('Dashed outline: how often the SFT model, the reference, writes it. It never changes.','虚线框：SFT 模型（参考模型）写出它的概率，训练中不变。'),'ppo-sw-ref',tx('SFT model (reference)','SFT 模型（参考）'))}${lg(tx('Hatched: E, the padded response the reward model overrates. Red when it is the response a prompt gets most often.','斜纹：E，被奖励模型高估的注水回答。它成了一道题最常写的回答时变红。'),'ppo-sw-gap',tx('E: the reward model’s gap','E：奖励模型的漏洞'))}</div>`+
   `<div class="ppo-bars" id="ppo-bars"></div>`+
   `<div class="ppo-kinds small muted" id="ppo-kinds"></div><p class="ppo-same small" id="ppo-same" hidden></p>`+
   `<div class="ppo-run" id="ppo-run"><button type="button" class="btn sm" data-act="run-play" id="ppo-run-play"></button><button type="button" class="btn ghost sm" data-act="run-replay" id="ppo-run-replay">${tx('Replay','重放')}</button>${spdCtl('runSpd',runSpd)}</div>`+
   `<div class="ppo-cvw" id="ppo-cvw"><div class="ppo-curve-head"><div class="ppo-curve-title" id="ppo-curve-title"></div>${A.ui.seg({id:'ppo-view',key:'view',label:tx('Curve','曲线'),options:[['rq',tx('Score and quality','分数和质量')],['kl','KL'],['rho',tx('Ratio','概率之比')]],value:view==='fan'?'rq':view})}</div>`+
   `<svg class="ppo-curve" id="ppo-curve" role="img" aria-label=""></svg><div class="ppo-key" id="ppo-key"></div></div>`+
   `<div class="ppo-mem" id="ppo-mem"></div>`+
   `<div class="metrics ppo-metrics" id="ppo-metrics"></div>`+
   `<details class="math" id="ppo-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ppo-mathbody"></div></details></div>`;
}
function controlsHTML(){
  const seg=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">`+A.ui.seg({id,key,label,options:opts,value:String(val),note})+`</div>`;
  return `<aside class="panel controls" id="ppo-controls" aria-label="${esc(A.t('ctrlAria')||'Settings')}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ppo-lock" hidden></p>`+
   seg('ppo-ctl-base','base',tx('Value model (the baseline)','价值模型（基准）'),[['1',tx('Use it','用')],['0',tx('None','不用')]],S.base,tx('With it, the push is reward − the value model’s estimate for that prompt; without it, the push is the reward itself.','用的话，推力是奖励减去价值模型对这道题的估计；不用的话，推力就是奖励本身。'))+
   seg('ppo-ctl-beta','beta',tx('KL coefficient β','KL 系数 β'),BETAS.map(b=>[String(b),bTxt(b)]),S.beta,tx(`Each response’s reward loses ${KLT()}. 0.05 is this lab’s default and TRL v1.12.0’s; this lab’s β does not carry over to a real model.`,`每个回答的奖励扣掉 ${KLT()}。0.05 是这个 Lab 的默认值，也是 TRL v1.12.0 的默认值；这个 Lab 里的 β 不能照搬到真实模型。`))+
   seg('ppo-ctl-K','K',tx('Passes over each batch','每批回答学几遍'),KS.map(k=>[String(k),String(k)]),S.K,tx('How many gradient passes each batch of generated responses gets. TRL v1.12.0 defaults to 4; verl to 1.','每批生成的回答做几遍梯度更新。TRL v1.12.0 默认 4 遍，verl 默认 1 遍。'))+
   seg('ppo-ctl-clip','clip',tx('Clipping (ε = 0.2)','截断（ε = 0.2）'),[['1',tx('On','截断')],['0',tx('Off','不截断')]],S.clip,tx('With clipping, a response with a positive advantage is no longer pushed up once its new-to-old probability ratio is above 1.2, and one with a negative advantage is no longer pushed down once it is below 0.8. A response that moved the other way is still pushed back. With one pass per batch the ratio is 1 at that pass and clipping never acts.','截断时，优势为正的回答，新旧概率之比超过 1.2 就不再往上推。优势为负的回答，低于 0.8 就不再往下压。往反方向走出去的回答，还会被推回来。每批只学 1 遍时，那一遍的概率之比都是 1，截断不起作用。'))+
   seg('ppo-ctl-eta','eta',tx('Step size','步长'),[['n',tx('Normal (8)','常规（8）')],['l',tx('Large (16)','大（16）')]],S.eta,tx('How far one gradient pass moves the 16 parameters. It was chosen for this lab’s small model and cannot be compared with a real learning rate.','一遍梯度更新把 16 个参数推多远。它是为这个 Lab 的小模型选的，和真实训练的学习率没法比。'))+
   `<div class="lockable" id="ppo-steps-wrap">`+A.ui.range({id:'ppo-ctl-steps',key:'steps',label:tx('Training steps','训练步数'),min:0,max:MAXS,value:S.steps,out:stepsTxt(S.steps),note:tx('Each step generates 4 responses per prompt and updates the policy once per pass.','每一步每道题生成 4 个回答，每学一遍更新一次策略模型。')})+`</div>`+
   `<div class="ppo-later small" id="ppo-later" hidden></div>`+
   `<div id="ppo-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import torch

def ppo_loss(logp, old_logp, ref_logp, score, value,
             beta=0.05, eps=0.2):
    # logp:     log pi(y|x) under the policy being trained
    # old_logp: the same, when the batch was generated
    # ref_logp: under the frozen reference model
    # score:    the reward model's score for each response
    # value:    the value model's estimate for each prompt
    kl = old_logp - ref_logp          # log(pi_old / pi_ref)
    reward = score - beta * kl        # KL penalty in the reward
    adv = (reward - value).detach()   # advantage
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    pg = -torch.min(ratio * adv, clipped * adv).mean()
    vf = ((value - reward.detach()) ** 2).mean()
    return pg + 0.1 * vf              # vf_coef = 0.1, as TRL`;
function connectHTML(){
  const fl=(base,p,label,x)=>`<a href="${base}${p}" target="_blank" rel="noopener"><code>${label||p}</code></a>${x}`;
  return `<section class="wrap connect" aria-labelledby="ppo-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ppo-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The PPO loss for a batch of whole responses, with the KL penalty in the reward and one value per prompt, as in this lab. Real trainers do the same per token.','一批完整回答的 PPO loss：KL 惩罚加在奖励里，每道题一个价值估计，和这个 Lab 一样。真实的训练器按 token 做同样的事。')}</p><pre class="code" id="ppo-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ppo-prod">`+
   tx(`<li>${fl(TRL,'ppo_trainer.py','trl/experimental/ppo/ppo_trainer.py',': TRL’s PPOTrainer takes four models, <code>model</code>, <code>ref_model</code>, <code>reward_model</code> and <code>value_model</code>, and adds <code>-kl_coef * kl</code> to each token’s reward. TRL v1.12.0 was its last release with PPO.')}</li>`+
      `<li>${fl(TRL,'ppo_config.py','ppo_config.py',': <code>kl_coef</code> 0.05, <code>cliprange</code> 0.2, <code>num_ppo_epochs</code> 4, <code>vf_coef</code> 0.1, <code>gamma</code> 1.0, <code>lam</code> 0.95.')}</li>`+
      `<li>${fl(VERL,'verl/trainer/config/actor/actor.yaml','verl actor.yaml',': <code>clip_ratio</code> 0.2 and <code>ppo_epochs</code> 1; its docs say PPO needs an actor and a critic, while GRPO and RLOO need no critic.')}</li>`+
      `<li>${cite(ORLHF,'OpenRLHF')}: separates the actor, reward, reference and critic models across GPUs, scaling to 70B+ parameters, and says RLHF spends 80% of its time on generation.</li>`+
      `<li>${cite(NIMPL,'The N implementation details of RLHF with PPO')}: the per-token KL penalty in the reward, the reward model’s score added only at the last token, reward and advantage whitening.</li>`+
      `<li>${cite(SPIN,'Proximal Policy Optimization')} (OpenAI Spinning Up): clipping case by case. With a positive advantage the objective stops growing once the ratio passes ${M('1 + \\varepsilon')}; with a negative one, once it drops below ${M('1 - \\varepsilon')}. Clipping removes the incentive to go far, and the new policy can still end up too far from the old one.</li>`,
      `<li>${fl(TRL,'ppo_trainer.py','trl/experimental/ppo/ppo_trainer.py','：TRL 的 PPOTrainer 接收四个模型：<code>model</code>、<code>ref_model</code>、<code>reward_model</code>、<code>value_model</code>；每个 token 的奖励都加上 <code>-kl_coef * kl</code>。TRL v1.12.0 是最后一个带 PPO 的版本。')}</li>`+
      `<li>${fl(TRL,'ppo_config.py','ppo_config.py','：<code>kl_coef</code> 0.05，<code>cliprange</code> 0.2，<code>num_ppo_epochs</code> 4，<code>vf_coef</code> 0.1，<code>gamma</code> 1.0，<code>lam</code> 0.95。')}</li>`+
      `<li>${fl(VERL,'verl/trainer/config/actor/actor.yaml','verl actor.yaml','：<code>clip_ratio</code> 0.2，<code>ppo_epochs</code> 1；文档说 PPO 需要 actor 和 critic，GRPO 和 RLOO 不需要 critic。')}</li>`+
      `<li>${cite(ORLHF,'OpenRLHF')}：把 actor、reward、reference、critic 四个模型分到不同的 GPU 上，可以训练 70B 以上的模型；它说 RLHF 训练 80% 的时间花在生成上。</li>`+
      `<li>${cite(NIMPL,'The N implementation details of RLHF with PPO')}：奖励里逐 token 的 KL 惩罚，奖励模型的分数只加在最后一个 token 上，奖励和优势的标准化。</li>`+
      `<li>${cite(SPIN,'Proximal Policy Optimization')}（OpenAI Spinning Up）：分优势为正、为负两种情形讲截断。优势为正时，比值超过 ${M('1 + \\varepsilon')}，目标就不再变大。优势为负时，比值低于 ${M('1 - \\varepsilon')}，目标就不再变大。截断只去掉走远的动力，新策略仍可能离旧策略太远。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noopener">Proximal Policy Optimization Algorithms</a><span>${tx('Schulman et al., 2017 (arXiv 1707.06347)','Schulman 等，2017（arXiv 1707.06347）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/1506.02438" target="_blank" rel="noopener">High-Dimensional Continuous Control Using Generalized Advantage Estimation</a><span>${tx('Schulman et al., 2015: GAE (arXiv 1506.02438)','Schulman 等，2015：GAE（arXiv 1506.02438）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noopener">Training language models to follow instructions with human feedback</a><span>${tx('Ouyang et al., 2022: InstructGPT (arXiv 2203.02155)','Ouyang 等，2022：InstructGPT（arXiv 2203.02155）')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ppo-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ppo-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ppo-app m2" id="ppo-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ppo-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- rail ---------- */
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3><p>${tx('Pick a scenario and watch the reward-model score and true quality over 200 steps.','选一个场景，看看训练 200 步里，奖励模型的分数和真实质量怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios: pick one','应用场景：选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ppo-insight" aria-live="polite"></div><p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function insightHTML(r){
  const T=r.T, i=presetNow(), t=r.t;
  if(RUN.on) return tx(`<p><b>Training: step ${t} of ${MAXS}.</b></p>`,`<p><b>训练中：第 ${t} / ${MAXS} 步。</b></p>`);
  let h=tx(`<p><b>After ${stepsTxt(t)}: reward-model score ${f2(T.rm[0])} → ${f2(r.rm)}, true quality ${f2(T.q[0])} → ${f2(r.q)}.</b></p>`,`<p><b>训练 ${t} 步后：奖励模型的平均分 ${f2(T.rm[0])} → ${f2(r.rm)}，真实质量 ${f2(T.q[0])} → ${f2(r.q)}。</b></p>`);
  if(t===0) return tx('<p><b>Step 0: nothing trained yet</b>; the policy is still the SFT model.</p>','<p><b>第 0 步：还没训练</b>，策略模型还是 SFT 模型。</p>');
  if(i<0) return h+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  if(i===0) h+=tx(`<p>With all three parts of PPO, the policy improves a little and stays near the SFT model: KL ${f(r.kl,2)} nats.</p>`,`<p>PPO 的三样都用上：策略模型改进了一点，又没有离 SFT 模型太远，KL 是 ${f(r.kl,2)} nats。</p>`);
  else if(i===1){const lock=r.pi.every(p=>Math.max(...p)>0.9), lt=k=>LET[r.top[k]];
    h+=lock?tx(`<p>Without a baseline every written response is pushed up by its whole score, so each prompt locks onto whatever it happened to write most in the first steps: the easy prompt onto ${lt(0)}, the medium onto ${lt(1)}, the hard onto ${lt(2)}. After that only that response gets written, and learning stops.</p>`,`<p>没有基准，写出来的每个回答都按整个分数往上推，每道题很快就锁定在开头几步碰巧写得最多的回答上：简单那道是 ${lt(0)}，中等那道是 ${lt(1)}，难的那道是 ${lt(2)}。之后只写得出这一个回答，也就学不到别的了。</p>`)
      :tx('<p>Without a baseline every written response is pushed up by its whole score, so the run is driven by which responses happen to be written.</p>','<p>没有基准，写出来的每个回答都按整个分数往上推，训练被碰巧写出了哪些回答带着走。</p>');}
  else if(i===2){const pk=r.peak, gone=[0,1,2].filter(k=>r.top[k]===4), nm=gone.map(k=>L(PROMPTS[k].d).toLowerCase()), nz=gone.map(k=>PROMPTS[k].d.zh), many=manyEnds(S), nlow=many.filter(q=>q<T.q[0]).length;
    h+=tx(`<p>True quality first rose to ${f2(T.q[pk])}, higher than with the KL penalty.${r.hack>=0&&gone.length?` From step ${r.hack} the policy found E on the ${nm.join(' and ')} prompt${gone.length>1?'s':''}, and true quality fell (red in the chart).`:''} This is seed 7. Over 40 seeds, ${nlow} end below the SFT model and the rest do not. Real training cannot see true quality, so it cannot tell which kind of run it is in: the KL penalty keeps the policy near the reference model either way.</p>`,
      `<p>真实质量先涨到 ${f2(T.q[pk])}，比加了 KL 惩罚时还高。${r.hack>=0&&gone.length?`从第 ${r.hack} 步起，策略模型在${nz.join('和')}那道题上找到了 E，真实质量开始跌（图里红色）。`:''}这是种子 7。试 40 个种子，有 ${nlow} 个最后低于 SFT 模型，其余的没有。真实训练看不到真实质量，分不清这一次是哪一种。加上 KL 惩罚，不管哪一种，策略模型都会留在参考模型附近。</p>`);}
  else {const u=runaway(T); h+=tx(`<p>${u>=0?`At step ${u+1} one batch moved a response’s probability ${ratioTxt(T.rho[u][1])} (red in the ratio chart).`:''} Without clipping, 4 passes compound and the KL ends at ${f(r.kl,2)} nats.</p>`,`<p>${u>=0?`第 ${u+1} 步的那一批把一个回答的概率推到原来的 ${ratioTxt(T.rho[u][1])}（概率之比图里的红色）。`:''}不截断时，4 遍叠在一起，KL 最后是 ${f(r.kl,2)} nats。</p>`);}
  return h;}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, you see whether both conditions hold, and why.','提交后告诉你两个条件是否都满足，以及原因。')}</p>`;
  const seg=(key,label,opts,val)=>A.ui.seg({id:'ppo-ic-'+key,key,label,options:opts,value:String(val),inline:true});
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`After 200 steps, beat the SFT model’s true quality (${f2(Q0)}) with a KL of at most ${CH_KL} nats`,`训练 200 步后，真实质量超过 SFT 模型（${f2(Q0)}），而且 KL 不超过 ${CH_KL} nats`)}</h3>`+
   `<p>${tx('Same small model, same seed. This run uses the large step and learns from each batch 4 times, with no KL penalty and no clipping. After 200 steps true quality is down to ',`同一个小模型、同一个种子。这次用较大的步长，每批回答学 4 遍，不加 KL 惩罚，也不截断。训练 200 步后，真实质量只剩 `)}${f2(chPoint(CH0).q)}${tx('. You can change these three together; change as few as you can.','。下面三项可以一起改，改得越少越好。')}</p>`+
   seg('beta',tx('KL coefficient β','KL 系数 β'),BETAS.map(b=>[String(b),bTxt(b)]),S.beta)+seg('K',tx('Passes over each batch','每批回答学几遍'),KS.map(k=>[String(k),String(k)]),S.K)+seg('clip',tx('Clipping (ε = 0.2)','截断（ε = 0.2）'),[['1',tx('On','截断')],['0',tx('Off','不截断')]],S.clip)+
   `<div class="goal mono" id="ppo-ch-q"><span>${tx('True quality after 200 steps','200 步后的真实质量')}</span><span><b id="ppo-ch-qv"></b>${tx(` (goal: above ${f2(Q0)})`,`（目标：高于 ${f2(Q0)}）`)}</span></div>`+
   `<div class="goal mono" id="ppo-ch-k"><span>${tx('KL from the reference model','离参考模型的 KL')}</span><span><b id="ppo-ch-kv"></b>${tx(` nats (goal: at most ${CH_KL})`,` nats（目标：最多 ${CH_KL}）`)}</span></div><div class="progress" id="ppo-ch-kp" aria-hidden="true"><i></i></div>`+
   `<p class="small muted">${tx(`The value model is on and the step size is large (16) throughout. The ${CH_KL}-nat limit is set for this challenge.`,`价值模型一直用着，步长一直是大（16）。${CH_KL} nats 的上限是这道题定的。`)}</p>`+submit+`<div id="ppo-ch-res" aria-live="polite"></div></div>`;}
const chName=p=>tx(`β = ${bTxt(p.beta)}, ${p.K} pass${p.K>1?'es':''}, ${p.clip?'clipped':'no clipping'}`,`β = ${bTxt(p.beta)}，学 ${p.K} 遍，${p.clip?'截断':'不截断'}`);
/* the changes from the challenge's start, in words: “β to 0.05 and one pass per batch” */
const chChanges=p=>{const c=[]; if(p.beta!==CH0.beta) c.push(tx(`β to ${bTxt(p.beta)}`,`把 β 改成 ${bTxt(p.beta)}`)); if(p.K!==CH0.K) c.push(tx(`${p.K} pass${p.K>1?'es':''} per batch`,`每批只学 ${p.K} 遍`)); if(p.clip!==CH0.clip) c.push(tx(p.clip?'clipping on':'clipping off',p.clip?'打开截断':'关掉截断')); return c.join(tx(' and ','，'));};
function chResultHTML(){
  if(!chSubmitted) return '';
  const all=chAll(), me=all.find(p=>p.beta===S.beta&&p.K===S.K&&p.clip===S.clip), okp=all.filter(p=>p.ok), nmin=Math.min(...okp.map(p=>p.n)), mins=okp.filter(p=>p.n===nmin);
  let h;
  if(me.ok&&me.n===nmin) h=tx(`<p class="okline"><b>Goal met: true quality ${f2(me.q)}, KL ${f(me.kl,2)} nats.</b> And ${me.n} changes is the fewest that works: no single change is enough.</p>`,`<p class="okline"><b>达成目标：真实质量 ${f2(me.q)}，KL ${f(me.kl,2)} nats。</b>而且 ${me.n} 处改动已经是最少的：只改一处都不够。</p>`);
  else if(me.ok) h=tx(`<p class="okline"><b>Goal met: true quality ${f2(me.q)}, KL ${f(me.kl,2)} nats.</b> ${nmin} changes are enough, for example ${esc(chChanges(mins[0]))}.</p>`,`<p class="okline"><b>达成目标：真实质量 ${f2(me.q)}，KL ${f(me.kl,2)} nats。</b>其实改 ${nmin} 处就够，比如${esc(chChanges(mins[0]))}。</p>`);
  else {h=!me.okQ&&me.beta===0?tx(`<p class="badline"><b>Not yet: true quality is ${f2(me.q)}, not above the SFT model’s ${f2(Q0)}.</b></p><p class="small">Hint: nothing stops the policy from piling onto the padded response. Which setting charges the policy for moving away from the reference model?</p>`,`<p class="badline"><b>还没达到：真实质量是 ${f2(me.q)}，没有超过 SFT 模型的 ${f2(Q0)}。</b></p><p class="small">提示：没有什么拦着策略模型往注水的回答上堆。哪个设置会因为策略模型离开参考模型而扣分？</p>`)
     :!me.okQ?tx(`<p class="badline"><b>Not yet: true quality is ${f2(me.q)}, not above the SFT model’s ${f2(Q0)}.</b></p><p class="small">Hint: the KL penalty is on, but 4 unclipped passes with a large step throw the policy off in a single batch. Which setting stops pushing a response once it has already been pushed far enough?</p>`,`<p class="badline"><b>还没达到：真实质量是 ${f2(me.q)}，没有超过 SFT 模型的 ${f2(Q0)}。</b></p><p class="small">提示：KL 惩罚已经加上了，可步长大、学 4 遍又不截断，一批回答就把策略甩出去了。哪个设置会在一个回答已经被推得够远之后，不再继续推它？</p>`)
     :me.beta===0?tx(`<p class="badline"><b>Not yet: true quality ${f2(me.q)} is fine, but the KL is ${f(me.kl,2)} nats, over ${CH_KL}.</b></p><p class="small">Hint: nothing holds the policy near the reference model. Which setting charges the policy for moving away from it?</p>`,`<p class="badline"><b>还没达到：真实质量 ${f2(me.q)} 没问题，但 KL 是 ${f(me.kl,2)} nats，超过了 ${CH_KL}。</b></p><p class="small">提示：没有什么把策略模型拴在参考模型附近。哪个设置会因为策略模型离开参考模型而扣分？</p>`)
     :tx(`<p class="badline"><b>Not yet: true quality ${f2(me.q)} is fine, but the KL is ${f(me.kl,2)} nats, over ${CH_KL}.</b></p><p class="small">Hint: 4 passes still move the policy too far in one step. Which setting stops pushing a response once it has already been pushed far enough?</p>`,`<p class="badline"><b>还没达到：真实质量 ${f2(me.q)} 没问题，但 KL 是 ${f(me.kl,2)} nats，超过了 ${CH_KL}。</b></p><p class="small">提示：学 4 遍时，一步还是走得太远。哪个设置会在一个回答已经被推得够远之后，不再继续推它？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal) return h;
  /* brief (the owner's review of batch 8, 2026-10-07): the combinations that meet both conditions, your pick and the initial setup; a KL over
     the limit is marked on those two rows only, so that it does not keep a failing row in the chart */
  const kept=p=>p.ok||p===me||p.n===0;
  const rows=all.map(p=>({label:tx(`${chName(p)} (${p.n} change${p.n===1?'':'s'})`,`${chName(p)}（改 ${p.n} 处）`),value:p.q*100,text:f2(p.q),ok:p.ok,me:p===me,
    tags:(p.n===0?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[]).concat(p.okK||!kept(p)?[]:[{text:tx(`KL ${f(p.kl,1)}`,`KL ${f(p.kl,1)}`)}])}));
  h+=A.ui.cmpChart({brief:true,title:tx(`True quality after 200 steps with the large step (goal: above ${f2(Q0)}, with KL at most ${CH_KL} nats)`,`步长大时，训练 200 步后的真实质量（目标：高于 ${f2(Q0)}，且 KL 不超过 ${CH_KL} nats）`),
    aria:tx('True quality after 200 steps: ','训练 200 步后的真实质量：')+all.filter(kept).map(p=>`${chName(p)}${tx(': ','：')}${f2(p.q)}${tx(`, KL ${f(p.kl,2)} nats`,`，KL ${f(p.kl,2)} nats`)}`).join(tx('; ','；')),rows,
    hiddenText:n=>tx(`The other ${n} combinations miss at least one condition and are not shown.`,`其余 ${n} 种组合至少有一个条件不满足，图里没有列出。`),goal:Q0*100,goalLabel:tx(`SFT model ${f2(Q0)}`,`SFT 模型 ${f2(Q0)}`),max:100,okText:tx('meets both conditions','两个条件都满足'),noText:tx('misses at least one','至少一个不满足'),goalText:tx('the SFT model’s true quality','SFT 模型的真实质量')});
  /* the reason, written from the rows themselves (B2 of the review): with β = 0 every row's KL is far over the limit, and where those runs land
     depends on the seed; with β > 0, 4 unclipped passes fail on quality (β 0.05) or on the KL (β 0.2) */
  const z=all.filter(p=>p.beta===0), zk=Math.min(...z.map(p=>p.kl)), pad=z.filter(p=>!p.okQ).length, luck=z.filter(p=>p.okQ);
  const r05=all.find(p=>p.beta===0.05&&p.K===4&&!p.clip), r2=all.find(p=>p.beta===0.2&&p.K===4&&!p.clip);
  const qh=p=>tx(`True quality ${f2(p.q)}, KL ${f(p.kl,2)} nats.`,`真实质量 ${f2(p.q)}，KL ${f(p.kl,2)} nats。`);
  const luckEn=luck.length?` With this seed ${pad} of them piled onto the padded response, while ${luck.map(p=>DEF(`“${chName(p)}”`,qh(p))).join(' and ')} happened to settle on better responses.`:'';
  const luckZh=luck.length?`这个种子下，其中 ${pad} 种的策略模型堆到了注水的回答上。${luck.map(p=>DEF(`“${chName(p)}”`,qh(p))).join('和')}碰巧停在了好一些的回答上。`:'';
  h+=tx(`<p class="small">Why: the initial setup has two problems. First, with β = 0 nothing holds the policy near the reference model, so all ${z.length} combinations with β = 0 end ${DEF('far over the KL limit',`The KL is ${f(zk,2)} nats or more in each.`)}. Where such a run ends up depends on the seed.${luckEn} Second, with 4 unclipped passes and a large step, one batch can throw the policy far away. With β = 0.05 ${DEF('true quality falls',qh(r05))}; with β = 0.2 ${DEF('the KL goes over the limit',qh(r2))}. The KL penalty fixes the first; clipping or a single pass fixes the second. Each alone is not enough, so every passing combination changes two settings.</p>`,
        `<p class="small">原因：初始设置有两个问题。第一，β = 0 时，没有什么把策略模型拴在参考模型附近。β = 0 的 ${z.length} 种组合，KL 都${DEF('远远超过上限',`KL 都在 ${f(zk,2)} nats 以上。`)}。这样的运行最后落在哪里，要看种子。${luckZh}第二，步长大、学 4 遍又不截断，一批回答就能把策略模型甩得很远。β = 0.05 时${DEF('真实质量跌了下来',qh(r05))}，β = 0.2 时 ${DEF('KL 超过了上限',qh(r2))}。KL 惩罚解决第一个问题，截断或只学 1 遍解决第二个。只解决一个不够，所以能达标的组合都要改两处。</p>`);
  h+=tx(`<p class="small muted">Note: this challenge uses 3 prompts and one seed. In real training β and the number of passes are checked against the KL and held-out evaluations, not one run.</p>`,`<p class="small muted">注意：这个挑战只用了 3 道题、一个种子。真实训练选 β 和学几遍，要看 KL 和留出集上的评测，不能只看一次运行。</p>`);
  return h;}

/* ---------- stage drawing ---------- */
/* parts 1-3 have no baseline yet (it is defined in part 4), so the one-step replay has the three phases part 2 names (S7 of the review) */
const PH=[{en:'Generate',zh:'生成'},{en:'Score',zh:'打分'},{en:'Update',zh:'更新'}], NPH=PH.length;
function loopState(r){   // which batch the loop panel shows, and how far into it
  if(LOOP.on) return {k:LOOP.k,ph:LOOP.ph};
  if(r.t<1) return {k:0,ph:0};
  return {k:r.t-1,ph:NPH};}
function drawLoop(r){
  const ls=loopState(r), B=r.T.B[ls.k], ph=ls.ph, base=!!S.base;
  $('#ppo-phases',app).innerHTML=`<span class="ppo-ph-k">${ph===0?tx('Step 1 has not started','第 1 步还没开始'):tx(`Step ${ls.k+1}`,`第 ${ls.k+1} 步`)}</span>`+PH.map((p,i)=>`<span class="ppo-ph${ph===i+1?' on':''}${ph>i+1?' done':''}">${i+1} ${esc(L(p))}</span>`).join('<span class="ppo-ph-a" aria-hidden="true">→</span>');
  const amax=Math.max(0.05,...B.map(b=>Math.abs(b.a)));
  $('#ppo-rows',app).innerHTML=[0,1,2].map(k=>{const bs=B.filter(b=>b.k===k);
    const vb=ph>=NPH&&base?tx(`baseline ${f2(bs[0].v)}`,`基准 ${f2(bs[0].v)}`):'';
    return `<div class="ppo-row"><div class="ppo-rq" data-tipx="${esc(L(PROMPTS[k].q))}" tabindex="0"><b>${esc(L(PROMPTS[k].d))}</b><span class="ppo-rv mono">${vb}</span></div><div class="ppo-chips">`+
     bs.map(b=>{const hasR=ph>=2, hasA=ph>=NPH, up=b.a>0, w=Math.round(Math.abs(b.a)/amax*100);
       const tip=hasA&&!base&&!S.beta?tx(`${LET[b.y]} (${L(KIND[b.y])}): score ${f(b.r,2)}; pushed up by its score.`,`${LET[b.y]}（${L(KIND[b.y])}）：分数 ${f(b.r,2)}，按分数往上推。`):hasA?tx(`${LET[b.y]} (${L(KIND[b.y])}): score ${f(b.r,2)}${S.beta?`, KL penalty ${sg(S.beta*b.d,3)}, reward ${f(b.R,3)}`:''}; advantage ${sg(b.a,3)}${base?`, the reward minus the baseline ${f(b.v,3)}`:', the reward itself'}.`,`${LET[b.y]}（${L(KIND[b.y])}）：分数 ${f(b.r,2)}${S.beta?`，KL 惩罚 ${sg(S.beta*b.d,3)}，奖励 ${f(b.R,3)}`:''}；优势 ${sg(b.a,3)}${base?`，等于奖励减去基准 ${f(b.v,3)}`:'，就是奖励本身'}。`)
         :tx(`${LET[b.y]}: ${L(KDESC[b.y])}.`,`${LET[b.y]}：${L(KDESC[b.y])}。`);
       return `<span class="ppo-chip${ph>=1?'':' ppo-hide'}${b.y===4?' ppo-chip-gap':''}" data-tipx="${esc(tip)}" tabindex="0"><span class="ppo-cl">${LET[b.y]} ${esc(L(KIND[b.y]))}</span><span class="ppo-cs mono">${hasR?f2(b.r):'&nbsp;'}</span><span class="ppo-ca mono${hasA&&Math.abs(b.a)>=0.005?(up?' up':' dn'):''}">${hasA?(Math.abs(b.a)<0.005?'≈ 0':`${up?'▲':'▼'} ${sg(b.a,2)}`):'&nbsp;'}</span><i class="ppo-cbar${hasA?(up?' up':' dn'):''}" style="width:${hasA?w:0}%"></i></span>`;}).join('')+`</div></div>`;}).join('');
  const pb=$('#ppo-loop-play',app); pb.textContent=LOOP.on?tx('❚❚ Pause','❚❚ 暂停'):(r.t>=1?tx('Replay this step','重放这一步'):tx('▶ Play step 1','▶ 播放第 1 步'));
  pb.dataset.tipx=tx('Replays the step shown: generate, score, update.','把图上这一步重放一遍：生成、打分、更新。');
}
function drawHead(r){
  const st=$('#ppo-st-title',app);
  st.textContent=r.t===0?tx('The SFT model before training: 3 prompts, 5 kinds of response each','训练前的 SFT 模型：3 道提问，每道 5 种回答')
    :tx(`The policy after ${stepsTxt(r.t)} of RL`,`RL 训练 ${r.t} 步后的策略模型`);
  const el=$('#ppo-status',app);
  if(mode==='challenge'){const p=chPoint({beta:S.beta,K:S.K,clip:S.clip}); el.className='status '+(p.ok?'ok':'bad'); el.textContent=p.ok?tx('Both conditions met','两个条件都满足'):(!p.okQ?tx('Quality too low','质量不够'):tx('KL too high','KL 太大'));
    el.dataset.tipx=tx('After 200 steps, true quality must be above the SFT model’s and the KL at most 0.5 nats.','200 步后，真实质量要高于 SFT 模型，KL 不超过 0.5 nats。'); return;}
  const hk=r.hack>=0, gapTop=r.top.some(j=>j===4);
  el.className='status '+(hk||gapTop?'bad':(r.t>0&&r.q>r.T.q[0]?'ok':''));
  el.textContent=r.t===0?tx('Before training','训练前'):(gapTop?tx('Reward hacking','出现了 reward hacking'):(hk?tx('Score up, quality down','分数涨、质量跌'):(r.q>r.T.q[0]?tx('Better than the SFT model','比 SFT 模型好'):tx('Not better yet','还没变好'))));
  el.dataset.tipx=tx('Red: the policy writes the padded response most often on some prompt, or the score rose while true quality fell. Green: true quality is above the SFT model’s.','红色：有一道题最常写的是注水回答，或者分数在涨、真实质量在跌。绿色：真实质量高于 SFT 模型。');
}
function drawBars(r){
  const host=$('#ppo-bars',app), P=r.pi;
  host.innerHTML=[0,1,2].map(k=>`<div class="ppo-bg"><div class="ppo-bt" data-tipx="${esc(L(PROMPTS[k].q))}" tabindex="0">${esc(L(PROMPTS[k].d))}${tx(': ','：')}<span>${esc(L(PROMPTS[k].q))}</span></div><div class="ppo-cols">`+
    P[k].map((p,i)=>{const hot=i===4&&r.top[k]===4;
      const tip=tx(`${LET[i]}, ${L(KDESC[i])}. Reward-model score ${f2(RM[k][i])}, true quality ${f2(QL[i])}. The policy writes it ${pc(p)} of the time; the SFT model ${pc(REF[i])}.${k===0?` Example: ${EX[i].en}`:''}`,`${LET[i]}，${L(KDESC[i])}。奖励模型的分数 ${f2(RM[k][i])}，真实质量 ${f2(QL[i])}。策略模型写出它的概率是 ${pc(p)}，SFT 模型是 ${pc(REF[i])}。${k===0?`示例：${EX[i].zh}`:''}`);
      return `<div class="ppo-col${i===4?' ppo-gapc':''}${hot?' ppo-hot':''}" data-tipx="${esc(tip)}" tabindex="0"><span class="ppo-cv mono">${pc(p)}</span><div class="ppo-ct"><i class="ppo-fill" style="height:${(p*100).toFixed(1)}%"></i><i class="ppo-ghost" style="height:${(REF[i]*100).toFixed(1)}%"></i></div><span class="ppo-cx mono">${LET[i]}</span></div>`;}).join('')+`</div></div>`).join('');
  const same=r.t>0&&S.beta>0&&[1,2].every(k=>P[k].every((p,i)=>Math.abs(p-P[0][i])<0.005)), sm=$('#ppo-same',app);
  sm.hidden=!same; if(same) sm.textContent=tx('With the KL penalty, all three prompts end on the same distribution. In this lab the three prompts start from the same SFT probabilities, and their scores differ only by a constant. Where the policy settles does not depend on that constant.','加了 KL 惩罚，三道题最后停在同一个分布上。在这个 Lab 里，三道题的 SFT 模型概率一样，分数只差一个常数。策略模型停在哪里，和这个常数无关。');
  $('#ppo-kinds',app).textContent=tx(`A ${KIND[0].en}, B ${KIND[1].en}, C ${KIND[2].en}, D ${KIND[3].en}, E ${KIND[4].en}. Each bar: how often the policy writes that response.`,`A ${KIND[0].zh}，B ${KIND[1].zh}，C ${KIND[2].zh}，D ${KIND[3].zh}，E ${KIND[4].zh}。每根条是策略模型写出这种回答的概率。`);
}
function drawRunCtl(r){
  const b=$('#ppo-run-play',app); b.textContent=RUN.on?tx('❚❚ Pause','❚❚ 暂停'):(r.t>=MAXS?tx('▶ Trained to step 200','▶ 已训练到第 200 步'):tx(`▶ Train to step ${runTarget()}`,`▶ 训练到第 ${runTarget()} 步`));
  b.disabled=!RUN.on&&r.t>=runTarget();
  b.dataset.tipx=tx('Trains step by step at the chosen speed: the whole run takes about 3 seconds at Medium and 1.5 at Fast.','按选定的速度一步一步训练：整次训练在“中”档约 3 秒，“快”档约 1.5 秒。');
  $('#ppo-run-replay',app).dataset.tipx=tx('Plays this run again from step 0.','从第 0 步重新播放这一次训练。');
  const lock=mode==='challenge'||(mode==='guided'&&step<SIX('value'))||clipShown();   // part 6 plays the passes of one batch instead (item 3)
  $('#ppo-run',app).hidden=lock;
}
const runTarget=()=>mode==='guided'&&STEPS[step].id==='value'?30:MAXS;
/* curves */
function drawCurve(r){
  const svg=$('#ppo-curve',app); if(!svg) return;
  const W=Math.max(240,Math.round(svg.getBoundingClientRect().width||svg.parentNode.clientWidth||560)), H=170, m={l:40,r:16,t:10,b:28};
  const T=r.T, t=r.t, vw=(view==='kl'&&!seen('kl'))||(view==='rho'&&!seen('clip'))?'rq':view, X=s=>m.l+(W-m.l-m.r)*s/(vw==='fan'?30:MAXS);
  let h='', y0=0, y1=1, Y, title, key='';
  const ax=(v,y,txt)=>{y=+y; return `<line class="ppo-grid" x1="${m.l}" x2="${W-m.r}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/><text class="ppo-ax" x="${m.l-5}" y="${(y+3.5).toFixed(1)}" text-anchor="end">${txt}</text>`;};
  const ks=(cls)=>`<svg class="ppo-ks" width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><line class="${cls}" x1="2" y1="5" x2="20" y2="5"/></svg>`;
  const path=(vals,upto,from)=>{let d=''; for(let s=from||0;s<=upto;s++) d+=(s===(from||0)?'M':'L')+X(s).toFixed(1)+','+Y(vals[s]).toFixed(1); return d;};
  if(vw==='rho'){const lo=0.1, hi=10, ly=v=>Math.log10(Math.max(lo,Math.min(hi,v)));
    Y=v=>m.t+(H-m.t-m.b)*(1-(ly(v)-ly(lo))/(ly(hi)-ly(lo)));
    h+=`<rect class="ppo-band" x="${m.l}" y="${Y(1.2).toFixed(1)}" width="${W-m.l-m.r}" height="${(Y(0.8)-Y(1.2)).toFixed(1)}"/>`;
    [0.1,1,10].forEach(v=>{h+=ax(v,Y(v).toFixed(1),String(v));});
    const mx=T.rho.map(x=>x[1]), mn=T.rho.map(x=>x[0]), up=Math.max(0,Math.min(t,MAXS)-1);
    if(t>=1){h+=`<path class="ppo-ln ppo-l-max" d="${path(mx,up)}"/><path class="ppo-ln ppo-l-min" d="${path(mn,up)}"/>`;
      const u=runaway(T); if(u>=0&&u<=up){h+=`<circle class="ppo-bad" cx="${X(u).toFixed(1)}" cy="${Y(mx[u]).toFixed(1)}" r="4.5"/><text class="ppo-badt" x="${Math.min(W-m.r-4,X(u)+8).toFixed(1)}" y="${m.t+10}" text-anchor="${X(u)+8>W-120?'end':'start'}">${esc(tx(`step ${u+1}: ${ratioTxt(mx[u])}`,`第 ${u+1} 步：${ratioTxt(mx[u])}`))}</text>`;}}
    title=tx('How many times did one step change a response’s probability? (new probability over old, log scale)','每一步把回答的概率最多改了几倍？（新概率除以旧概率，对数刻度）');
    key=`<span data-tipx="${esc(tx('Solid: the largest ratio in that step’s batch, after all its passes.','实线：这一步那批回答里最大的概率之比（学完所有遍之后）。'))}" tabindex="0">${ks('ppo-ln ppo-l-max')}${tx('largest ratio','最大的比')}</span><span data-tipx="${esc(tx('Dashed: the smallest ratio in that step’s batch.','虚线：这一步那批回答里最小的概率之比。'))}" tabindex="0">${ks('ppo-ln ppo-l-min')}${tx('smallest ratio','最小的比')}</span><span data-tipx="${esc(tx('Shaded band: [0.8, 1.2], the range of clipping with ε = 0.2. A response with a positive advantage is no longer pushed up above 1.2; one with a negative advantage is no longer pushed down below 0.8.','阴影带：[0.8, 1.2]，ε = 0.2 的截断区间。优势为正的回答超过 1.2 就不再往上推，优势为负的回答低于 0.8 就不再往下压。'))}" tabindex="0"><i class="ppo-sw ppo-sw-band"></i>[0.8, 1.2]</span><span data-tipx="${esc(tx('Red dot: the first update that moved a response’s probability more than 3 times.','红点：第一次把一个回答的概率改到 3 倍以上的那次更新。'))}" tabindex="0"><i class="ppo-sw ppo-sw-bad"></i>${tx('ran off','跑飞')}</span>`;}
  else if(vw==='kl'){const mx=Math.max(0.3,...T.kl); y1=mx<=0.5?0.5:(mx<=1?1:Math.ceil(mx)); Y=v=>m.t+(H-m.t-m.b)*(1-v/y1);
    [0,y1/2,y1].forEach(v=>{h+=ax(v,Y(v).toFixed(1),f(v,y1<1?2:1));});
    h+=`<path class="ppo-ln ppo-l-rm" d="${path(T.kl,t)}"/>`;
    title=tx('How far the policy is from the reference model (KL, in nats)','策略模型离参考模型有多远（KL，单位 nats）');
    key=`<span data-tipx="${esc(tx('The KL from the reference model, averaged over the 3 prompts, in nats; 0 means the policy is still the SFT model.','离参考模型的 KL，三道题取平均，单位 nats；0 表示还是 SFT 模型。'))}" tabindex="0">${ks('ppo-ln ppo-l-rm')}KL</span>`;}
  else if(vw==='fan'){Y=v=>m.t+(H-m.t-m.b)*(1-(v-0.2)/0.8);
    [0.2,0.6,1].forEach(v=>{h+=ax(v,Y(v).toFixed(1),f(v,1));});
    const tt=Math.min(30,t); FAN.forEach((sd,j)=>{const q=traj({...S,seed:sd}).q; h+=`<path class="ppo-ln ${j?'ppo-l-fan':'ppo-l-q'}" d="${path(q,tt)}"/>`;});
    h+=`<line class="ppo-ref" x1="${m.l}" x2="${W-m.r}" y1="${Y(T.q[0]).toFixed(1)}" y2="${Y(T.q[0]).toFixed(1)}"/>`;
    title=tx('True quality over 30 steps, 6 random seeds','6 个随机种子各训练 30 步的真实质量');
    key=`<span data-tipx="${esc(tx('Thick dashed line: seed 7, the run the rest of the page shows.','粗虚线：种子 7，页面其他地方用的就是这一次。'))}" tabindex="0">${ks('ppo-ln ppo-l-q')}${tx('seed 7','种子 7')}</span><span data-tipx="${esc(tx('Thin lines: the same settings with seeds 1, 2, 3, 5 and 11.','细线：同样的设置，种子 1、2、3、5、11。'))}" tabindex="0">${ks('ppo-ln ppo-l-fan')}${tx('5 other seeds','另外 5 个种子')}</span><span data-tipx="${esc(tx('Dotted: the SFT model’s true quality before training.','点线：训练前 SFT 模型的真实质量。'))}" tabindex="0">${ks('ppo-ref')}${tx('SFT model','SFT 模型')}</span>`;}
  else {Y=v=>m.t+(H-m.t-m.b)*(1-(v-0.2)/0.8);
    [0.2,0.6,1].forEach(v=>{h+=ax(v,Y(v).toFixed(1),f(v,1));});
    /* part 5 (B2 of the review): the other 5 seeds' true quality as thin lines, so one lucky or unlucky seed is not the whole story */
    const fan5=mode==='guided'&&STEPS[step].id==='kl';
    if(fan5) FAN.slice(1).forEach(sd=>{h+=`<path class="ppo-ln ppo-l-fan" d="${path(traj({...S,seed:sd}).q,t)}"/>`;});
    h+=`<path class="ppo-ln ppo-l-rm" d="${path(T.rm,t)}"/><path class="ppo-ln ppo-l-q" d="${path(T.q,t)}"/>`;
    if(r.hack>=0) h+=`<path class="ppo-ln ppo-l-hack" d="${path(T.q,t,r.hack)}"/>`;
    title=tx('Reward-model score and true quality, averaged over the 3 prompts','奖励模型的平均分和真实质量（三道题的平均）');
    key=`<span data-tipx="${esc(tx('Solid: the reward-model score the policy gets on average. This is all real training can see.','实线：策略模型平均能拿到的奖励模型分数。真实训练只看得到它。'))}" tabindex="0">${ks('ppo-ln ppo-l-rm')}${tx('reward-model score','奖励模型的分数')}</span><span data-tipx="${esc(tx('Dashed: the average true quality of what the policy writes (seed 7).','虚线：策略模型写出的回答的平均真实质量（种子 7）。'))}" tabindex="0">${ks('ppo-ln ppo-l-q')}${tx('true quality','真实质量')}</span>${fan5?`<span data-tipx="${esc(tx('Thin lines: true quality with the same settings and seeds 1, 2, 3, 5 and 11.','细线：同样的设置，种子 1、2、3、5、11 的真实质量。'))}" tabindex="0">${ks('ppo-ln ppo-l-fan')}${tx('5 other seeds','另外 5 个种子')}</span>`:''}<span data-tipx="${esc(tx('Red: true quality falling while the score is still above where quality peaked.','红色：分数还比真实质量最高时高，真实质量却在往下跌。'))}" tabindex="0">${ks('ppo-ln ppo-l-hack')}${tx('score up, quality down','分数涨、质量跌')}</span>`;}
  const xs=vw==='fan'?[0,10,20,30]:[0,50,100,150,200]; xs.forEach(s=>{h+=`<text class="ppo-ax" x="${X(s).toFixed(1)}" y="${H-m.b+14}" text-anchor="middle">${s}</text>`;});
  h+=`<text class="ppo-ax" x="${W-m.r}" y="${H-2}" text-anchor="end">${tx('training steps','训练步数')}</text>`;
  if(vw!=='fan'){const xt=X(Math.min(t,MAXS)); h+=`<line class="ppo-now" x1="${xt.toFixed(1)}" x2="${xt.toFixed(1)}" y1="${m.t}" y2="${H-m.b}"/>`;
    key+=`<span data-tipx="${esc(tx('The vertical dotted line marks the current training step.','竖直点线标出当前的训练步。'))}" tabindex="0"><svg class="ppo-ks" width="10" height="14" viewBox="0 0 10 14" aria-hidden="true"><line class="ppo-now" x1="5" y1="0" x2="5" y2="14"/></svg>${tx('now','当前')}</span>`;}
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('height',H); svg.innerHTML=h;
  $('#ppo-curve-title',app).textContent=title; svg.setAttribute('aria-label',title);
  $('#ppo-key',app).innerHTML=key;
  const sg_=$('#ppo-view',app); if(sg_){sg_.closest('.ctl,.seg-wrap,div').hidden=vw==='fan'; A.ui.setPressed(sg_.querySelector('.seg-ctl')||sg_,vw==='fan'?'rq':vw);
    $$('button[data-v]',sg_).forEach(b=>{b.hidden=(b.dataset.v==='kl'&&!seen('kl'))||(b.dataset.v==='rho'&&!seen('clip'));});}
}
function drawMem(){
  const m=memOf(memAlg), name={pol:tx('policy (trained)','策略模型（训练）'),val:tx('value model (trained)','价值模型（训练）'),ref:tx('reference (frozen)','参考模型（冻结）'),rew:tx('reward model (frozen)','奖励模型（冻结）')}, cap=memOf('ppo').tot;
  const parts=m.parts.map(p=>{const tip=p.b===18?tx(`${L({en:name[p.k].replace(/ \(.*/,''),zh:name[p.k].replace(/（.*/,'')})}: 8,030,261,248 parameters × 18 bytes (BF16 weights 2, FP32 gradients 4, FP32 copy of the weights 4, Adam’s m 4 and v 4) = ${f(p.gb,1)} GB.`,`${name[p.k].replace(/（.*/,'')}：8,030,261,248 个参数 × 18 字节（BF16 权重 2、FP32 梯度 4、FP32 权重副本 4、Adam 的 m 4 和 v 4）= ${f(p.gb,1)} GB。`)
      :tx(`${name[p.k].replace(/ \(.*/,'')}: 8,030,261,248 parameters × 2 bytes (frozen BF16 weights only) = ${f(p.gb,1)} GB.`,`${name[p.k].replace(/（.*/,'')}：8,030,261,248 个参数 × 2 字节（只有冻结的 BF16 权重）= ${f(p.gb,1)} GB。`);
    return `<span class="ppo-mb ppo-mb-${p.b===18?'t':'f'}" style="width:${(p.gb/cap*100).toFixed(2)}%" data-tipx="${esc(tip)}" tabindex="0"><span>${esc(name[p.k])} ${f(p.gb,1)}</span></span>`;}).join('');
  $('#ppo-mem',app).innerHTML=`<div class="ppo-mem-t">${tx(`Llama‑3.1‑8B in ${memAlg.toUpperCase()}: the models alone take <b>${f(m.tot,1)} GB</b>, at least ${m.gpus} H100s of 80 GB`,`Llama‑3.1‑8B 做 ${memAlg.toUpperCase()}：光模型就要 <b>${f(m.tot,1)} GB</b>，至少 ${m.gpus} 张 80 GB 的 H100`)}</div><div class="ppo-mbar">${parts}</div><div class="ppo-mem-k small muted"><span><i class="ppo-sw ppo-sw-t"></i>${tx('trained: 18 bytes per parameter','训练中：每个参数 18 字节')}</span><span><i class="ppo-sw ppo-sw-f"></i>${tx('frozen BF16: 2 bytes per parameter','冻结的 BF16：每个参数 2 字节')}${tx(' (','（')}${m.parts.filter(p=>p.b===2).map(p=>`${esc(name[p.k].replace(/ \(.*/,'').replace(/（.*/,''))} ${f(p.gb,1)} GB`).join(tx(', ','、'))}${tx(')','）')}</span></div>`;
}
/* ---------- what a part has introduced (review codex-b9v3, item 2) ----------
   In the guided parts the stage and Settings show only what the parts so far have introduced: parts 1–3 show one RL step (the
   one-step panel, the bars, the reward-model score and true quality); the curves and the training steps come with part 4, the value
   model too; the KL penalty, its curve and its card with part 5; the passes, clipping, the step size, the ratio curve and card with
   part 6; the four models' memory with part 7. The scenarios and the challenge show everything. */
const seen=id=>mode!=='guided'||step>=SIX(id);
const KINTRO={'ppo-steps-wrap':'value','ppo-ctl-base-wrap':'value','ppo-ctl-beta-wrap':'kl','ppo-ctl-K-wrap':'clip','ppo-ctl-clip-wrap':'clip','ppo-ctl-eta-wrap':'clip'};
const LATER=[['value',[{en:'training steps',zh:'训练步数'},{en:'the value model',zh:'价值模型'}]],['kl',[{en:'the KL coefficient β',zh:'KL 系数 β'}]],
  ['clip',[{en:'passes over each batch',zh:'每批回答学几遍'},{en:'clipping',zh:'截断'},{en:'the step size',zh:'步长'}]]];
function laterHTML(){const rows=LATER.filter(([id])=>!seen(id)); if(!rows.length) return '';
  return `<div class="ppo-later-h">${tx('Coming in later parts','后面会讲')}</div><ul>`+rows.map(([id,ks])=>`<li><b>${tx(`Part ${SIX(id)+1}`,`第 ${SIX(id)+1} 部分`)}</b><span>${ks.map(k=>esc(L(k))).join(tx(', ','、'))}</span></li>`).join('')+`</ul>`;}

/* ---------- part 6's picture: step 1's batch, pass by pass (review codex-b9v3, item 7) ----------
   Two panels, one per sign of the advantage, each with the clipped objective of one response per unit of |A| against its ratio ρ: for
   A > 0 a slope up to 1 + ε, then the plateau; for A < 0 the plateau below 1 − ε, then a slope down; without clipping a straight line.
   Each response of the batch sits on its curve at its ratio at the start of the chosen pass: filled and with an arrow when this pass
   pushes it, hollow on the plateau, red when (without clipping) it is pushed further although it is already out of the band the way
   its advantage wants. “After” shows the ratios after the last pass. */
const clipShown=()=>mode==='guided'&&STEPS[step].id==='clip';
const CX0=0.4, CX1=1.8;
const ZONE={1:[{at:[CX0,1-EPS],l:[{en:'fell out:',zh:'掉出区间'},{en:'still pushed up',zh:'仍往上推'}]},{at:[1+EPS,CX1],l:[{en:'plateau:',zh:'平台'},{en:'no more push',zh:'不再往上推'}]}],
  '-1':[{at:[CX0,1-EPS],l:[{en:'plateau:',zh:'平台'},{en:'no more push',zh:'不再往下压'}]},{at:[1+EPS,CX1],l:[{en:'rose out:',zh:'涨出区间'},{en:'still pushed down',zh:'仍往下压'}]}]};
function clipPassNow(){return clipPass===0?0:Math.min(clipPass,S.K);}
function drawCP(svg,sg,D,p,clip){
  const W=Math.max(240,Math.round(svg.getBoundingClientRect().width||svg.parentNode.clientWidth||320)), H=156, m={l:10,r:12,t:36,b:30};
  const cl=v=>Math.min(CX1,Math.max(CX0,v)), X=v=>m.l+(W-m.l-m.r)*(cl(v)-CX0)/(CX1-CX0);
  const Lf=v=>{const c=Math.min(Math.max(v,1-EPS),1+EPS); return clip?(sg>0?Math.min(v,c):-Math.max(v,c)):sg*v;};
  const y0=sg>0?CX0:-CX1, y1=sg>0?CX1:-CX0, Y=v=>m.t+(H-m.t-m.b)*(1-(v-y0)/(y1-y0)), xb=H-m.b;
  const pts=xs=>xs.map(v=>`${X(v).toFixed(1)},${Y(Lf(v)).toFixed(1)}`).join(' ');
  const seg=(xs,cls,tip)=>`<g data-tipx="${esc(tip)}" tabindex="0"><polyline class="ppo-cp-hit" points="${pts(xs)}"/><polyline class="${cls}" points="${pts(xs)}"/></g>`;
  const tSlope=sg>0?tx('Slope: here a higher ρ means a larger objective L, so the response is pushed up.','斜坡：在这一段，ρ 越大，目标 L 越大，所以这个回答还被往上推。')
    :tx('Slope: here a lower ρ means a larger objective L, so the response is pushed down.','斜坡：在这一段，ρ 越小，目标 L 越大，所以这个回答还被往下压。');
  const tFlat=tx('Plateau: here L stays the same whatever ρ does, so its gradient is 0 and this batch no longer pushes the response.','平台：在这一段，ρ 怎么变，L 都不变，梯度是 0，这一批就不再推这个回答。');
  let h=`<rect class="ppo-band" x="${X(1-EPS).toFixed(1)}" y="${m.t}" width="${(X(1+EPS)-X(1-EPS)).toFixed(1)}" height="${(xb-m.t).toFixed(1)}"/>`+
    `<line class="ppo-grid" x1="${m.l}" x2="${W-m.r}" y1="${xb}" y2="${xb}"/>`;
  (W>=340?[0.6,0.8,1,1.2,1.4,1.6]:[0.8,1,1.2]).forEach(v=>{h+=`<text class="ppo-ax" x="${X(v).toFixed(1)}" y="${xb+13}" text-anchor="middle">${f(v,1)}</text>`;});
  h+=`<text class="ppo-ax" x="${W-m.r}" y="${H-3}" text-anchor="end">${esc(tx('probability ratio ρ','概率之比 ρ'))}</text><text class="ppo-ax" x="${m.l}" y="${H-3}">${esc(tx('↑ objective L','↑ 目标 L'))}</text>`;
  if(clip){if(sg>0) h+=seg([CX0,1-EPS,1+EPS],'ppo-cp-slope',tSlope)+seg([1+EPS,CX1],'ppo-cp-flat',tFlat);
    else h+=seg([CX0,1-EPS],'ppo-cp-flat',tFlat)+seg([1-EPS,1+EPS,CX1],'ppo-cp-slope',tSlope);
    /* each zone's name sits over its zone, held to the chart's outer edge so that a long English label is never cut (check of codex-b9v3) */
    ZONE[sg].forEach(z=>{const lf=z.at[0]===CX0, zx=(lf?m.l:W-m.r).toFixed(1), an=lf?'start':'end'; h+=`<text class="ppo-cp-z${z.l[0].zh==='平台'?' ppo-cp-zf':''}" x="${zx}" y="13" text-anchor="${an}"><tspan x="${zx}">${esc(L(z.l[0]))}</tspan><tspan x="${zx}" dy="13">${esc(L(z.l[1]))}</tspan></text>`;});}
  /* without clipping the top corner the line runs into holds the off-chart values (“24.19 →”, “← 0.13”), so the note takes the other top corner */
  else{const zx=(sg>0?m.l:W-m.r).toFixed(1); h+=seg([CX0,CX1],'ppo-cp-slope',tSlope); h+=`<text class="ppo-cp-z" x="${zx}" y="13" text-anchor="${sg>0?'start':'end'}">${esc(tx('no clipping: no plateau','不截断：没有平台'))}</text>`;}
  /* the responses, stacked upward when two are closer than 26 px, each with its arrow beside it */
  const ds=D.filter(d=>sg>0?d.a>0:d.a<0).map(d=>{const rho=p?d.rp[p-1]:d.end; return {d,rho,x:X(rho),st:dotState(d,p)};}).sort((u,v)=>u.x-v.x), placed=[];
  ds.forEach(o=>{let lv=0; while(placed.some(q=>q.lv===lv&&Math.abs(q.x-o.x)<26)) lv++; o.lv=lv; placed.push(o);});
  ds.forEach(o=>{const yc=Y(Lf(cl(o.rho))), y=yc-o.lv*14, out=outBy(o.rho)>0, bad=o.st==='bad'||(o.st==='end'&&!clip&&S.K>1&&out);
    const cls=bad?'ppo-cd-bad':(o.st==='flat'||o.st==='endFlat'?'ppo-cd-flat':(o.st==='end'?'ppo-cd-end':'ppo-cd-on')), dir=sg>0?1:-1;
    if(o.lv) h+=`<line class="ppo-cd-lead" x1="${o.x.toFixed(1)}" x2="${o.x.toFixed(1)}" y1="${(y+5).toFixed(1)}" y2="${yc.toFixed(1)}"/>`;
    h+=`<g class="t ppo-cd" data-ctl="cdot" data-term="${o.d.id}"><circle class="${cls}" cx="${o.x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5"/>`;
    if(['push','back','bad'].includes(o.st)){const x0=o.x+dir*8, x1=o.x+dir*21; h+=`<path class="ppo-cd-arr${bad?' bad':''}" d="M${x0.toFixed(1)},${y.toFixed(1)}L${x1.toFixed(1)},${y.toFixed(1)}M${(x1-dir*4).toFixed(1)},${(y-3.5).toFixed(1)}L${x1.toFixed(1)},${y.toFixed(1)}L${(x1-dir*4).toFixed(1)},${(y+3.5).toFixed(1)}"/>`;}
    /* an off-chart value sits level with its own dot, so that stacked dots at the edge keep their own numbers (check of codex-b9v3) */
    if(o.rho>CX1||o.rho<CX0) h+=`<text class="ppo-cd-v${bad?' bad':''}" x="${(o.rho>CX1?o.x-8:o.x+8).toFixed(1)}" y="${(y+4).toFixed(1)}" text-anchor="${o.rho>CX1?'end':'start'}">${o.rho>CX1?'':'← '}${f2(o.rho)}${o.rho>CX1?' →':''}</text>`;
    h+=`</g>`;});
  /* the corrective push (review codex-b9v3, item 7): a response that left the band against its advantage is still pushed back.
     Drawn as a dashed example at the card's own numbers (A > 0 at 0.7, A < 0 at 1.3), saying whether this batch has one */
  if(clip&&S.K>1){const ex=sg>0?1-1.5*EPS:1+1.5*EPS, x=X(ex), y=Y(Lf(ex)), dir=sg>0?1:-1, x0=x+dir*8, x1=x+dir*21;
    const real=ds.some(o=>p?o.st==='back':backSide(o.d.a,o.rho));
    const tip=(sg>0?tx(`Example: a response with a positive advantage whose ρ fell to ${f(ex,1)}. It is still on the slope, so it is still pushed back up.`,`例子：一个优势为正的回答，ρ 掉到了 ${f(ex,1)}。它还在斜坡上，还会被往上推回来。`)
      :tx(`Example: a response with a negative advantage whose ρ rose to ${f(ex,1)}. It is still on the slope, so it is still pushed back down.`,`例子：一个优势为负的回答，ρ 涨到了 ${f(ex,1)}。它还在斜坡上，还会被往下压回来。`))
      +(real?'':tx(' Step 1’s batch has no such response here.','第 1 步这一批里没有这样的回答。'));
    h+=`<g class="ppo-cg" data-tipx="${esc(tip)}" tabindex="0"><circle class="ppo-cd-ex" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5"/>`+
      `<path class="ppo-cd-arr ex" d="M${x0.toFixed(1)},${y.toFixed(1)}L${x1.toFixed(1)},${y.toFixed(1)}M${(x1-dir*4).toFixed(1)},${(y-3.5).toFixed(1)}L${x1.toFixed(1)},${y.toFixed(1)}L${(x1-dir*4).toFixed(1)},${(y+3.5).toFixed(1)}"/>`+
      `<text class="ppo-cd-v" x="${x.toFixed(1)}" y="${(y-10).toFixed(1)}" text-anchor="middle">${esc(tx(`e.g. ${f(ex,1)}`,`例 ${f(ex,1)}`))}</text></g>`;}
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('height',H); svg.innerHTML=h;
}
/* part 6's picture, frame by frame (the owner's review of batch 9, item 3): the frames are pass 1 … K, then “after”. Each frame has
   its own title, legend and caption; they are stacked in one grid cell so that the panel keeps its height while the passes play (R7.9).
   The caption says, as it plays, that clipping does not keep ρ inside the band: it stops pushing a response that has left the band
   the way its advantage pushes it, and the others keep learning (item 3d). */
const clipFrames=()=>[...Array(S.K)].map((_,i)=>i+1).concat([0]);
function clipTitle(p){const K=S.K;
  return p?tx(`Step 1’s batch, pass ${p} of ${K}: which responses are still pushed?`,`第 1 步的这批回答，第 ${p} 遍（共 ${K} 遍）：哪些还在被推？`)
    :tx(`Step 1’s batch after ${K===1?'its one pass':`all ${K} passes`}: where did the ratios end?`,`第 1 步的这批回答学完 ${K} 遍：比值停在哪里？`);}
const clipSub=()=>tx(`Each training step writes one batch (4 responses per prompt), then learns from that same batch ${S.K===1?'once':`${S.K} times`}. The next step writes a new batch, 200 steps in all.`,
  `每一步训练先写一批回答（每道题 4 个），再用这同一批学 ${S.K} 遍。下一步重新写一批，一共训练 200 步。`);
function clipCaption(D,p,clip){
  const K=S.K, ends=D.map(d=>d.end), lo=f2(Math.min(...ends)), hi=f2(Math.max(...ends));
  if(p===1) return K===1?tx('Pass 1: every ratio is 1, and every response is pushed the way its advantage says. With one pass, clipping has nothing to act on.','第 1 遍：比值都是 1，每个回答都按优势推。只学 1 遍，截断没有可管的。')
    :tx('Pass 1: every ratio is 1, and every response is pushed the way its advantage says.','第 1 遍：比值都是 1，每个回答都按优势推。');
  if(!p){
    if(K===1) return tx(`After its one pass the ratios are between ${lo} and ${hi}. The next step writes new responses with the updated policy, so its ratios start at 1 again.`,`学完这 1 遍，比值在 ${lo} 到 ${hi} 之间。下一步用更新后的策略模型重新写一批回答，比值又从 1 开始。`);
    const top=topEnd(D);
    if(!clip) return tx(`After ${K} passes the ratios are between ${lo} and ${hi} (red: out of the band). Every pass pushed again, so the passes added up: ${respName(top)} is now ${timesTxt(top.end)} as likely.`,`学完 ${K} 遍，比值在 ${lo} 到 ${hi} 之间（红色在区间外）。每一遍都照样推，几遍叠在一起，${respName(top)} 的概率被推到了原来的 ${timesTxt(top.end)}。`);
    const nOut=D.filter(d=>!d.on[K-1]&&outBy(d.end)>0).length, nOn=D.filter(d=>d.on[K-1]).length;
    return tx(`After ${K} passes ${nOut} responses end outside [0.8, 1.2] (hollow). Clipping did not pull them back into the band: from the pass that took them out, it simply stopped pushing them. The other ${nOn} were pushed to the last pass (filled). The pass that crosses the edge can go far: ${respName(top)} ended at ${f2(top.end)}.`,
      `学完 ${K} 遍，有 ${nOut} 个回答停在 [0.8, 1.2] 外（空心）。截断没有把它们拉回区间，只是从它们走出区间起就不再推。另外 ${nOn} 个一直学到最后一遍（实心）。越界的那一遍可能一下走得很远，比如${respName(top)} 停在 ${f2(top.end)}。`);}
  const st=D.map(d=>dotState(d,p)), n=k=>st.filter(x=>x===k).length;
  if(!clip) return tx(`Pass ${p}: without clipping, ${n('bad')} responses that have already left the band the way their advantage pushes them are still pushed further out (red). Every extra pass pushes them again.`,`第 ${p} 遍：不截断时，有 ${n('bad')} 个回答已经朝优势的方向走出区间，还在被往外推（红色）。多学一遍，就再推一次。`);
  return tx(`Pass ${p}: ${n('flat')} responses have already left [0.8, 1.2] the way their advantage pushes them, so this pass no longer pushes them (hollow: gradient 0). The other ${n('push')+n('back')} keep learning (filled). Clipping does not pull ρ back into the band.${n('back')?` ${n('back')} of them moved out the other way and are pushed back.`:''}`,
    `第 ${p} 遍：有 ${n('flat')} 个回答的 ρ 已经朝优势的方向走出 [0.8, 1.2]，这一遍不再推它们（空心，梯度为 0）。其余 ${n('push')+n('back')} 个照样学（实心）。截断不会把 ρ 拉回区间。${n('back')?`其中 ${n('back')} 个是从反方向走出区间，被推回来。`:''}`);}
const cpSw=c=>`<svg class="ppo-ks" width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle class="${c}" cx="6" cy="6" r="4.5"/></svg>`;
function clipKey(p){const K=S.K, clip=!!S.clip;
  const items=p?[[cpSw('ppo-cd-on'),tx('pushed at this pass','这一遍还在推'),tx('Filled, with an arrow: this pass still pushes the response, the way its advantage says.','实心带箭头：这一遍还按优势推这个回答。')]]
      .concat(K>1?(clip?[[cpSw('ppo-cd-flat'),tx('left the band: not pushed','走出区间：不再推'),tx('Hollow: the response has already moved out of [0.8, 1.2] the way its advantage pushes it, so this pass does not push it: its gradient is 0.','空心：这个回答已经朝优势的方向走出 [0.8, 1.2]，这一遍不再推它，梯度是 0。')]]:[[cpSw('ppo-cd-bad'),tx('pushed further out','还在往外推'),tx('Red: without clipping, a response already out of the band the way its advantage pushes it is still pushed further out.','红色：不截断时，已经朝优势的方向走出区间的回答，还在被往外推。')]]):[])
    :(clip&&K>1?[[cpSw('ppo-cd-flat'),tx('left the band: no longer pushed','走出区间：不再推'),tx('Hollow: the response left [0.8, 1.2] the way its advantage pushes it, and from then on this batch did not push it.','空心：这个回答朝优势的方向走出了 [0.8, 1.2]，从那以后这一批就不再推它。')],[cpSw('ppo-cd-on'),tx('pushed to the last pass','学到最后一遍'),tx('Filled: the response was still pushed at the last pass.','实心：最后一遍还在推这个回答。')]]
      :[[cpSw('ppo-cd-end'),tx('ratio after the last pass','学完时的比值'),tx('Each dot: where the response’s ratio ended after the batch’s last pass.','每个点：这一批学完时，这个回答的比值停在哪里。')]].concat(!clip&&K>1?[[cpSw('ppo-cd-bad'),tx('out of the band','在区间外'),tx('Red: the ratio ended out of [0.8, 1.2] with no clipping.','红色：不截断时，学完后比值在 [0.8, 1.2] 外。')]]:[]));
  const nd=items.length;   // the dot items come first: only they get the note about responses written twice
  if(clip&&K>1) items.push([`<svg class="ppo-ks" width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle class="ppo-cd-ex" cx="6" cy="6" r="4.5"/></svg>`,tx('example: pushed back','例子：还会被推回来'),tx('Dashed: the card’s example of a response that left the band against its advantage (positive advantage at 0.7, negative at 1.3). It is still on the slope, so it is pushed back.','虚线：卡片里的例子，回答朝优势的反方向走出了区间（优势为正时 0.7，为负时 1.3）。它还在斜坡上，所以还会被推回来。')]);
  items.push([`<i class="ppo-sw ppo-sw-band"></i>`,'[0.8, 1.2]',tx('Shaded band: [0.8, 1.2], the range of clipping with ε = 0.2.','阴影带：[0.8, 1.2]，ε = 0.2 的截断区间。')]);
  return items.map(([s_,t_,tip],i)=>`<span data-tipx="${esc(tip+(i<nd?tx(' A response written twice for one prompt has the same advantage and ratio, so it is one dot.','同一道题写出两次的同一种回答，优势和比值都一样，画成一个点。'):''))}" tabindex="0">${s_}${t_}</span>`).join('');}
function passBtnTxt(){const K=S.K;
  return PASS.on?tx('❚❚ Pause','❚❚ 暂停'):(watchedPass.has(passKey())?tx(`Replay the ${K===1?'pass':`${K} passes`}`,`重放这 ${K} 遍`):tx(`▶ Play the ${K===1?'pass':`${K} passes`}`,`▶ 播放这 ${K} 遍`));}
function drawClip(){
  const host=$('#ppo-clip',app); if(!host) return;
  host.hidden=!clipShown(); if(host.hidden) return;
  const K=S.K, clip=!!S.clip, p=clipPassNow(), D=batch1(S), fr=clipFrames(), key=`${cfgKey()}|${tx('en','zh')}`;
  if(host._k!==key){host._k=key;
    const opts=fr.map(q=>[String(q),q?tx(`Pass ${q}`,`第 ${q} 遍`):tx('After','学完')]);
    const stack=(cls,f)=>`<div class="ppo-stk">${fr.map(q=>`<div class="${cls}" data-f="${q}">${f(q)}</div>`).join('')}</div>`;
    host.innerHTML=`<div class="ppo-clip-head"><div class="ppo-clip-tw">${stack('ppo-clip-t',q=>esc(clipTitle(q)))}<p class="ppo-clip-s small">${esc(clipSub())}</p></div>${A.ui.seg({id:'ppo-pass',key:'pass',label:tx('Which pass','看哪一遍'),options:opts,value:String(p)})}</div>`+
      `<div class="ppo-clip-ctl"><button type="button" class="btn sm" data-act="pass-play" id="ppo-pass-play"></button>${spdCtl('passSpd',passSpd)}</div>`+
      `<div class="ppo-cps">${[1,-1].map(sg=>`<div class="ppo-cp"><div class="ppo-cp-t">${sg>0?tx('Better than expected (A &gt; 0): pushed up','比预期好（优势为正）：往上推'):tx('Worse than expected (A &lt; 0): pushed down','比预期差（优势为负）：往下压')}</div><svg class="ppo-cp-svg" data-sg="${sg}" role="img"></svg></div>`).join('')}</div>`+
      stack('ppo-key ppo-cp-key',q=>clipKey(q))+stack('ppo-cp-cap small',q=>esc(clipCaption(D,q,clip)));}
  $$('[data-f]',host).forEach(e=>e.classList.toggle('on',+e.dataset.f===p));
  const sgEl=$('#ppo-pass .seg-ctl',host)||$('.seg-ctl[data-key="pass"]',host); if(sgEl) A.ui.setPressed(sgEl,String(p));
  const pb=$('#ppo-pass-play',host); pb.textContent=passBtnTxt(); pb.dataset.tipx=tx('Plays step 1’s batch pass by pass: which responses each pass still pushes.','把第 1 步这一批回答一遍一遍地播：每一遍还在推哪些回答。');
  A.ui.setPressed($('.seg-ctl[data-key="passSpd"]',host),passSpd);
  $$('.ppo-cp-svg',host).forEach(svg=>{const sg=+svg.dataset.sg; drawCP(svg,sg,D,p,clip);
    svg.setAttribute('aria-label',(sg>0?tx('Responses with a positive advantage','优势为正的回答'):tx('Responses with a negative advantage','优势为负的回答'))+tx(': objective against the probability ratio.','：目标随概率之比的变化。'));});
}
/* the hover of one response in part 6's picture: an aligned list (R2.11) */
function cdotTip(id){const d=batch1(S).find(x=>String(x.id)===String(id)); if(!d) return ''; const p=clipPassNow(), rho=p?d.rp[p-1]:d.end, st=dotState(d,p);
  const what={push:tx('pushed the way its advantage says','按优势推'),back:tx('moved out the other way: pushed back','从反方向走出区间：被推回来'),flat:tx('on the plateau: not pushed','在平台上：不再推'),
    bad:tx('already out of the band, pushed further out','已经在区间外，还在往外推'),end:outBy(rho)>0?tx('outside [0.8, 1.2]','在 [0.8, 1.2] 外'):tx('inside [0.8, 1.2]','在 [0.8, 1.2] 里'),
    endFlat:tx('left the band the way its advantage pushes; no longer pushed after that','朝优势方向走出区间，之后不再被推'),endOn:tx('still pushed at the last pass','最后一遍还在被推')}[st];
  const row=(k,v)=>`<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`;
  return `<div class="ppo-tl">`+row(tx('Response','回答'),tx(`${LET[d.y]}, ${KIND[d.y].en}, on the ${PROMPTS[d.k].d.en.toLowerCase()} prompt`,`${PROMPTS[d.k].d.zh}题的回答 ${LET[d.y]}（${KIND[d.y].zh}）`))+
    row(tx('Written in this batch','这一批写了'),tx(`${d.n} time${d.n>1?'s':''}`,`${d.n} 次`))+row(tx('Advantage','优势'),sg(d.a,3))+
    row(p?tx(`ρ at the start of pass ${p}`,`第 ${p} 遍开始时的 ρ`):tx('ρ after the last pass','学完时的 ρ'),f2(rho))+row(p?tx('This pass','这一遍'):tx('Where','位置'),what)+`</div>`;}

const numOf=v=>{const m=String(v).replace(MINUS,'-').match(/[+-]?\d+(?:\.\d+)?/); return m?parseFloat(m[0]):NaN;};
const MET=[
 {k:'rm',label:()=>tx('Reward-model score','奖励模型的平均分'),v:r=>f2(r.rm),sub:r=>tx(`${f2(r.T.rm[0])} before training`,`训练前 ${f2(r.T.rm[0])}`),tip:()=>tx('The reward-model score the policy gets on average over the 3 prompts. Real training sees only this.','策略模型在三道题上平均能拿到的奖励模型分数。真实训练只看得到它。')},
 {k:'q',label:()=>tx('True quality','真实质量'),v:r=>f2(r.q),sub:r=>tx(`${f2(r.T.q[0])} before training`,`训练前 ${f2(r.T.q[0])}`),tip:()=>D_TQ()},
 {k:'kl',label:()=>tx('KL from the reference','离参考模型的 KL'),v:r=>f(r.kl,2),sub:()=>tx('nats; 0 before training','nats，训练前是 0'),tip:()=>tx('How far the policy’s probabilities have moved from the reference model’s, averaged over the 3 prompts, in nats. Its formula is in “Show the math”.','策略模型的概率离参考模型有多远，三道题取平均，单位 nats。公式在“展开公式”里。')},
 {k:'rho',label:()=>tx('Largest ratio so far','到目前最大的概率之比'),v:r=>r.t<1?'–':ratioTxt(r.rmax),sub:()=>tx('new over old, any response','新旧之比，所有回答里最大的'),tip:()=>tx('The largest new-to-old probability ratio any update has given a response so far.','到目前为止，所有更新里一个回答的新旧概率之比的最大值。')}];
function drawMetrics(r){
  const host=$('#ppo-metrics',app);
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), v=m.v(r), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(r); el.dataset.tipx=m.tip();
    el.hidden=(m.k==='kl'&&!seen('kl'))||(m.k==='rho'&&(!seen('clip')||clipShown()));
    const a=numOf(shownPrev[m.k]), b=numOf(v), dt=userAct&&!RUN.on&&isFinite(a)&&isFinite(b)&&A.fmt.delta?(A.fmt.delta(a,b,{dp:2})||''):'';
    if(dt){d.textContent=dt; d.classList.add('on'); clearTimeout(d._t); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2600);}
    else if(!userAct){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=v;});
}
/* Show the math: from part 4 on, with only the lines the parts so far have introduced (item 2 of review codex-b9v3): part 4 the advantage
   and the value model's update, part 5 the KL penalty in the reward, part 6 the ratio and the clipped objective */
function mathHTML(r){
  const ls={k:LOOP.on?LOOP.k:(r.t>=MAXS&&!RUN.on?0:Math.max(0,r.t-1))}, b=r.T.B[ls.k].find(x=>x.k===0), kl=seen('kl'), cl=seen('clip'), Bt=T_('beta','β = '+bTxt(S.beta));   // at the end of a run with the KL penalty every advantage is 0 and every ratio 1 (the run has settled), so a finished run shows step 1
  /* H2 (the owner, 2026-10-09): a few short blocks, each a lead line in words, the named formula, this response's numbers under it and
     the result; the symbols in one plain sentence */
  const AH=tx('advantage','优势')+' '+M('\\hat{A}');
  const rows=[`<div class="ppo-mh">${tx(`Step ${ls.k+1}, easy prompt, first response written (${LET[b.y]}), with ${S.base?'the value model':'no value model'}`,`第 ${ls.k+1} 步，简单那道题写出的第一个回答（${LET[b.y]}），${S.base?'用价值模型':'不用价值模型'}`)}${kl?tx(', ','，')+Bt:''}</div>`];
  if(kl) rows.push(`<p>${tx('The reward is the reward-model score minus the KL penalty:','奖励是奖励模型的分数扣掉 KL 惩罚：')}</p>`+
    EQ(tx('reward','奖励')+' '+M('R'),['r - \\beta\\, \\log \\frac{\\pi_{old}(y|x)}{\\pi_{ref}(y|x)}',`${f(b.r,2)} - ${bTxt(S.beta)} \\times ({${sg(b.d,4)}})`,f(b.R,4)])+
    `<p class="small">${tx(`r is the reward-model score. ${M('\\pi_{old}(y|x)')} is the probability that the policy which wrote the batch gave response y to prompt x, and ${M('\\pi_{ref}(y|x)')} the frozen reference model’s. log is the natural log.`,`r 是奖励模型的分数。${M('\\pi_{old}(y|x)')} 是写出这批回答的策略模型给提问 x 的回答 y 的概率，${M('\\pi_{ref}(y|x)')} 是冻结的参考模型给的。log 是自然对数。`)}</p>`);
  else rows.push(`<p>${tx('Without the KL penalty, the reward is the reward-model score itself:','不加 KL 惩罚时，奖励就是奖励模型的分数：')}</p>`+EQ(tx('reward','奖励')+' '+M('R'),['r',f(b.r,2)]));
  rows.push(`<div class="ppo-mh">${tx('The advantage','优势')}</div><p>${tx(`The advantage is how much better than expected the response is. The value model’s baseline ${M('V(x)')} is what prompt x is expected to score:`,`优势是这个回答比预期好多少。价值模型对提问 x 的基准 ${M('V(x)')}，就是这道题预期能拿的分：`)}</p>`+
    (S.base?EQ(AH,['R - V(x)',`${f(b.R,4)} - ${f(b.v,4)}`,sg(b.a,4)])
      :EQ(AH,['R',sg(b.a,4)])+`<p class="small">${tx('No value model here, so there is no baseline: the advantage is the reward itself.','这里不用价值模型，没有基准，优势就是奖励本身。')}</p>`));
  if(cl){const pos=b.a>0;
    rows.push(`<div class="ppo-mh">${tx('The ratio and clipping','概率之比和截断')}</div><p>${tx('The probability ratio compares the policy after this step’s passes with the policy that wrote the batch:','概率之比，比的是这一步学完之后的策略模型和写出这批回答时的策略模型：')}</p>`+
      EQ(tx('probability ratio','概率之比')+' '+M('\\rho'),['\\frac{\\pi_{\\theta}(y|x)}{\\pi_{old}(y|x)}',f(b.rho,4)])+
      (S.clip&&S.K>1?`<p class="small">${pos?tx('Its advantage is positive: above 1.2 it is no longer pushed up; below 0.8 it would still be pushed up.','它的优势为正，ρ 超过 1.2 就不再往上推，低于 0.8 还会被往上推。'):tx('Its advantage is negative: below 0.8 it is no longer pushed down; above 1.2 it would still be pushed down.','它的优势为负，ρ 低于 0.8 就不再往下压，超过 1.2 还会被往下压。')}</p>`:'')+
      `<p>${tx('Each pass raises the clipped objective, averaged over the batch. With ε = 0.2, clip keeps ρ inside [0.8, 1.2]:','每学一遍，都让截断的目标在这批回答上的平均变大。ε = 0.2 时，clip 把 ρ 限制在 [0.8, 1.2] 里：')}</p>`+
      EQ(tx('objective','目标')+' '+M('L'),['\\text{mean}\\, \\min\\left(\\rho \\hat{A}, \\text{clip}(\\rho, 1-\\varepsilon, 1+\\varepsilon) \\hat{A}\\right)']));}
  if(kl) rows.push(`<div class="ppo-mh">${tx('The KL card','KL 卡片')}</div><p>${tx('How far the policy has moved from the reference model, for one prompt x; the card averages it over the 3 prompts:','策略模型离参考模型有多远，下面是一道提问 x 的算法，卡片上是三道题的平均：')}</p>`+
    EQ('KL',['\\sum_{y} \\pi(y|x) \\log \\frac{\\pi(y|x)}{\\pi_{ref}(y|x)}']));
  rows.push(`<div class="ppo-mh">${tx('The value model','价值模型')}</div><p>${tx('After each step the value model moves its baseline halfway toward the mean reward the prompt got:','每一步之后，价值模型把基准往这道题拿到的平均奖励挪一半：')}</p>`+
    EQ(`${tx('new baseline','新的基准')} ${M('V(x)')}`,['V(x) + 0.5 \\left(\\text{mean}\\, R - V(x)\\right)']));
  rows.push(`<p class="small muted">${tx(`Policy parameters: one logit per response plus ${M('w \\times \\text{length}')}, w now ${sg(r.w,3)} (shared by the 3 prompts).${cl?` Step size ${ETA[S.eta]}, ${S.K} pass${S.K>1?'es':''} per batch,`:''} 4 responses per prompt per step, seed ${SEED}.`,`策略模型的参数：每个回答一个 logit，再加上 ${M('w \\times \\text{长度}')}，w 现在是 ${sg(r.w,3)}（三道题共用）。${cl?`步长 ${ETA[S.eta]}，每批学 ${S.K} 遍，`:''}每一步每道题 4 个回答，种子 ${SEED}。`)}</p>`);
  return rows.join('');
}

/* ---------- controls, rail sync ---------- */
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(g=>{const k=g.dataset.key; if(k==='view') A.ui.setPressed(g,view==='fan'?'rq':view); else if(k==='runSpd') A.ui.setPressed(g,runSpd); else if(k==='loopSpd') A.ui.setPressed(g,loopSpd); else if(k==='passSpd') A.ui.setPressed(g,passSpd); else if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
  $$('input[data-key="steps"]',app).forEach(i=>{if(+i.value!==S.steps) i.value=S.steps; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=stepsTxt(S.steps);});
  const lock=mode==='challenge', c=$('#ppo-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#ppo-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you can change only β, the passes and clipping, in the task card. The rest is fixed: the value model on, the large step (16), 200 training steps, seed 7.','挑战中只能在任务卡里改 β、学几遍和截断。其余设置固定：用价值模型，步长大（16），训练 200 步，种子 7。');
  $$('#ppo-controls .lockable',app).forEach(x=>{x.hidden=lock||!seen(KINTRO[x.id]||'mission');});
  const lt=$('#ppo-later',app), lh=lock?'':laterHTML(); if(lt._h!==lh){lt.innerHTML=lh; lt._h=lh;} lt.hidden=!lh;
  $('#ppo-reset-wrap',app).hidden=mode==='guided'&&!seen('value');
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
const WHY=()=>({loop:tx('Play step 1 to continue, or press Show me.','播放第 1 步才能继续，或点“演示给我看”。'),predict:tx('Pick an answer to continue, or Skip.','选一个答案才能继续，或点“跳过”。'),value:tx('Try both options to continue.','两个选项都试过才能继续。'),kl:tx('Try “No KL penalty” and β = 0.05 to continue.','试过“不加 KL 惩罚”和 β = 0.05 才能继续。'),clip:tx('Try 4 passes with and without clipping to continue.','学 4 遍时截断和不截断都试过，才能继续。')}[STEPS[step].id]||'');
function updateRail(r){
  const rail=$('#ppo-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect(r)&&!RUN.on&&!LOOP.on&&!PASS.on) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step])&&!RUN.on&&!LOOP.on&&!PASS.on;
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(r); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; if(again) A.ui.obsBump(box);}}}
    const dis=isNextDisabled(); if(A.ui.setNext) A.ui.setNext(rail,dis,WHY()); else $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=dis;});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    const lb=$('[data-act="loop-play"]',rail); if(lb) lb.textContent=LOOP.on?tx('❚❚ Pause','❚❚ 暂停'):(r.t>=1?tx('▶ Replay step 1','▶ 重放第 1 步'):tx('▶ Play this step','▶ 播放这一步'));
    $$('[data-lkey]',rail).forEach(b=>{const lv=LEV[b.dataset.lkey]; const on=String(lv.cur())===b.dataset.lever; b.setAttribute('aria-pressed',String(on)); b.classList.toggle('on',on);
      const o=lv.opts.find(x=>String(x.v)===b.dataset.lever), rr=b.querySelector('.lv-r'); if(o&&rr){const h=levRes(b.dataset.lkey,o.v); if(rr.innerHTML!==h) rr.innerHTML=h;}});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ppo-insight',rail); if(ins){const i=presetNow(), html=`<div class="eyebrow">${i>=0?esc(L(PRESETS[i].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const p=chPoint({beta:S.beta,K:S.K,clip:S.clip}), qv=$('#ppo-ch-qv',rail); if(!qv) return;
    qv.textContent=f2(p.q); $('#ppo-ch-kv',rail).textContent=f(p.kl,2);
    $('#ppo-ch-q',rail).classList.toggle('met',p.okQ); $('#ppo-ch-k',rail).classList.toggle('met',p.okK);
    $('#ppo-ch-kp i',rail).style.width=Math.min(100,p.kl/CH_KL*100)+'%'; $('#ppo-ch-kp',rail).classList.toggle('met',p.okK);
    $$('.seg-ctl[data-key]',rail).forEach(g=>{const k=g.dataset.key; if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#ppo-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=!!res._h&&!!html; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:WHY(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#ppo-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  const r=compute(S);
  const lp=$('#ppo-loop',app); lp.hidden=!(mode==='guided'&&step<=SIX('predict'));
  const mem=$('#ppo-mem',app), costPart=mode==='guided'&&STEPS[step].id==='cost';
  if(costPart&&mem.previousElementSibling!==$('#ppo-stage .stage-head',app)) $('#ppo-stage .stage-head',app).after(mem);
  else if(!costPart&&mem.nextElementSibling!==$('#ppo-metrics',app)) $('#ppo-metrics',app).before(mem);
  mem.classList.toggle('ppo-mem-top',costPart);
  const curve=seen('value')&&!clipShown();   // part 6 shows the passes of one batch, not the 200-step curve (the owner's review of batch 9, item 3c)
  mem.hidden=!seen('cost'); $('#ppo-cvw',app).hidden=!curve; $('#ppo-math',app).hidden=!seen('value');   // item 2: parts 1–3 show one step
  drawHead(r); if(!lp.hidden) drawLoop(r); drawClip(); drawBars(r); drawRunCtl(r); if(curve) drawCurve(r); if(seen('cost')) drawMem(); drawMetrics(r);
  const mb=$('#ppo-mathbody',app), mh=seen('value')?mathHTML(r):''; if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail(r);
  {const hk=mode+'|'+token(); if(hk!==rsHash&&/^#?ppo(~|$)/.test(location.hash||'#ppo')){rsHash=hk; A.hash.write(token());}}   // only when the settings change, not every frame of a run: a frame must not overwrite a hash the reader just navigated to
  A.dock.refresh(); announce(r);
}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return;
  const k=seen('kl'); A.announce(tx(`After ${stepsTxt(r.t)}: reward-model score ${f2(r.rm)}, true quality ${f2(r.q)}${k?`, KL ${f(r.kl,2)} nats`:''}.`,`训练 ${r.t} 步后：奖励模型的平均分 ${f2(r.rm)}，真实质量 ${f2(r.q)}${k?`，KL ${f(r.kl,2)} nats`:''}。`));},400);}
const cfgKey=()=>CFG.map(k=>S[k]).join('|');
/* R3.20: a run plays only when its picture is new. A run already watched to its end shows its end at once; Replay plays it again. */
/* src: 'settings' (Scenarios cards and Settings knobs: at most QUICK_SECS, rule G6), 'quiet' (show the end at once) or a lever (the chosen speed) */
function showRun(target,src){
  stopRun(); stopLoop(); stopPass();
  const key=cfgKey()+'>'+target;
  if(watchedRun.has(key)||A.reduceMotion()||src==='quiet'){S.steps=target; watchedRun.add(key); render(); return;}
  playRun(0,target,src==='settings');}
function playRun(from,target,quick){
  stopRun(); stopPass(); S.steps=from; const secs=quick?Math.min(RUN_SECS[runSpd],QUICK_SECS):RUN_SECS[runSpd];
  RUN={on:true,raf:0,last:0,acc:0,to:target,key:cfgKey()+'>'+target,rate:Math.max(1,target)/secs}; nRunStarts++;
  const tick=ts=>{if(!RUN.on||!app){RUN.raf=0; return;}
    if(RUN.last){RUN.acc+=Math.min(0.1,(ts-RUN.last)/1000)*RUN.rate; const n=Math.floor(RUN.acc);
      if(n>0){RUN.acc-=n; S.steps=Math.min(RUN.to,S.steps+n); if(S.steps>=RUN.to){watchedRun.add(RUN.key); RUN.on=false; RUN.raf=0; render(); keepStage(); return;} render();}}
    RUN.last=ts; RUN.raf=requestAnimationFrame(tick);};
  RUN.raf=requestAnimationFrame(tick); render();}
function stopRun(){if(RUN.raf) cancelAnimationFrame(RUN.raf); RUN.on=false; RUN.raf=0;}
function playLoop(){
  stopRun(); stopLoop(); S.steps=0; LOOP={on:true,raf:0,start:0,ph:1,k:0}; nLoopStarts++;
  const tick=ts=>{if(!LOOP.on||!app){LOOP.raf=0; return;}
    if(!LOOP.start) LOOP.start=ts;
    const ph=1+Math.floor((ts-LOOP.start)/1000/LOOP_SPEEDS[loopSpd]);
    if(ph!==LOOP.ph){LOOP.ph=Math.min(NPH,ph); if(LOOP.ph>=NPH) S.steps=1; render();}
    if(ph>NPH){LOOP.on=false; LOOP.raf=0; watchedLoop.add(cfgKey()); render(); return;}
    LOOP.raf=requestAnimationFrame(tick);};
  if(A.reduceMotion()){LOOP.on=false; S.steps=1; watchedLoop.add(cfgKey()); render(); return;}
  LOOP.raf=requestAnimationFrame(tick); render();}
function stopLoop(){if(LOOP.raf) cancelAnimationFrame(LOOP.raf); if(LOOP.on&&LOOP.ph>=NPH) S.steps=1; LOOP.on=false; LOOP.raf=0;}
/* part 6 (the owner's review of batch 9, item 3): what plays is step 1's batch, pass 1 … K and then “after”, in the picture at the top of
   the stage; the 200-step run behind the bars and the metric cards shows its end at once. R3.20: a setting whose passes were watched
   shows them at once; 重放 plays them again. */
const passKey=s=>CFG.map(k=>(s||S)[k]).join('|');
function stopPass(mark){if(PASS.raf) cancelAnimationFrame(PASS.raf); if(PASS.on&&mark) watchedPass.add(PASS.key); PASS.on=false; PASS.raf=0;}
function playPass(){
  stopRun(); stopLoop(); stopPass();
  const fr=clipFrames(), key=passKey();
  if(A.reduceMotion()){clipPass=0; watchedPass.add(key); render(); return;}
  PASS={on:true,raf:0,start:0,i:0,fr,key}; nPassStarts++; clipPass=fr[0]; render();
  const tick=ts=>{if(!PASS.on||!app){PASS.raf=0; return;}
    if(!PASS.start) PASS.start=ts;
    const i=Math.floor((ts-PASS.start)/1000/PASS_SPEEDS[passSpd]);
    if(i>=PASS.fr.length){PASS.on=false; PASS.raf=0; clipPass=0; watchedPass.add(PASS.key); render(); return;}
    if(i!==PASS.i){PASS.i=i; clipPass=PASS.fr[i]; drawClip();}
    PASS.raf=requestAnimationFrame(tick);};
  PASS.raf=requestAnimationFrame(tick);}
function showPasses(play,reveal){
  stopRun(); S.steps=MAXS; watchedRun.add(cfgKey()+'>'+MAXS);
  const k=passKey();
  if(play&&!watchedPass.has(k)){playPass(); if(reveal) revealClip(); return;}
  if(!watchedPass.has(k)){watchedPass.add(k); clipPass=0;}
  render();}
/* G6: when the passes start after a click in the card, the picture is on screen (on phones the stage sits above the card) */
function revealClip(){requestAnimationFrame(()=>{if(!app) return; const el=$('#ppo-clip',app); if(!el||el.hidden) return; const r=el.getBoundingClientRect();
  if(innerWidth<=760||r.top<56||r.top>innerHeight-160) A.ui.reveal(el);});}
function setState(p,src){
  stopRun(); stopLoop();
  const before=cfgKey();
  if(p.alg){memAlg=p.alg; render(); return;}
  Object.assign(S,p);
  if(mode==='challenge') Object.assign(S,CH);
  if(!BETAS.includes(S.beta)) S.beta=0.05; if(!KS.includes(S.K)) S.K=4; if(!ETA[S.eta]) S.eta='n';
  S.steps=Math.max(0,Math.min(MAXS,Math.round(S.steps)));
  const changed=before!==cfgKey(); if(changed) stopPass();
  if(mode==='guided'){const sid=STEPS[step].id; if(STEPS[step].levers&&p.steps==null){(tried[sid]=tried[sid]||new Set()).add(sid==='clip'?`${S.K}-${S.K===1?0:S.clip}`:LEV[STEPS[step].levers].cur());}
    if(src==='quiet'&&STEPS[step].levers&&STEPS[step].levers!=='alg'){stopPass(); if(sid==='clip'){S.steps=MAXS; watchedRun.add(cfgKey()+'>'+MAXS); watchedPass.add(passKey()); clipPass=0;} else {S.steps=runTarget(); watchedRun.add(cfgKey()+'>'+runTarget());} render(); return;}
    if(sid==='clip'&&changed){showPasses(src!=='settings',src==='lever'); return;}   // a Settings knob shows the end at once (G6, R3.2)
    if(STEPS[step].levers&&STEPS[step].levers!=='alg'&&changed){showRun(runTarget(),src); return;}}
  if(mode==='sandbox'&&changed&&p.steps==null){showRun(MAXS,'settings'); return;}
  render();}
function enterStep(i){
  stopRun(); stopLoop(); stopPass(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step], target=s.id==='value'?30:(s.enter?s.enter().steps:S.steps);
  if(s.enter){Object.assign(S,s.enter());}
  if(s.view) view=s.view;
  memAlg=s.id==='cost'?memAlg:'ppo';
  if(!tried[s.id]&&s.levers&&s.levers!=='alg') tried[s.id]=new Set([String(LEV[s.levers].cur())==='1-0'?'1-0':LEV[s.levers].cur()]);
  const st=$('#ppo-stage',app); if(st) st.scrollTop=0;
  renderRail();
  if(s.id==='clip'){showPasses(true,false); return;}   // part 6 plays the passes of step 1's batch at the top of the stage (item 3)
  if(['value','kl'].includes(s.id)){S.steps=0; showRun(target);}   // a run a part shows plays only if it has not been watched (R3.20)
  else if(s.id==='loop'&&!watchedLoop.has(cfgKey())){S.steps=0; render();}
  else if(s.id==='loop'){S.steps=1; render();}
}
function revealStep(){A.ui.reveal($('#ppo-rail .step.current',app)||$('#ppo-rail',app));}
/* keepStage (R9.7), as labs/llm-judge.js: after a click in the rail, put the sticky stage back under the top bar, unless the reader has
   scrolled the current card off screen */
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#ppo-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ppo-rail .step.current',app)||$('#ppo-rail .railcard',app)||$('#ppo-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function setMode(m){
  if(m===mode) return; stopRun(); stopLoop(); stopPass(true);
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH,...CH0}; view='rq'; memAlg='ppo';}
  if(m==='sandbox'){view='rq'; memAlg='ppo'; if(was==='guided') S={...DEFAULT};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#ppo-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
  if(m==='sandbox'&&S.steps>=MAXS&&!watchedRun.has(cfgKey()+'>'+MAXS)) showRun(MAXS,'settings');
}
function applyPreset(i){const p=PRESETS[i]; userAct=true; try{stopRun(); stopLoop(); Object.assign(S,p.p); S.steps=0; showRun(p.p.steps,'settings');}finally{userAct=false;}
  requestAnimationFrame(()=>{if(app){const b=$('#ppo-insight',app); if(b&&b.getBoundingClientRect().bottom>innerHeight) b.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'});}});}
/* while a run or the one-step replay plays, the settings are those it plays to (its last step), so the address and the saved record
   name where it ends, not a frame of it */
const token=s=>{if(!s){s=S; if(RUN.on) s={...S,steps:RUN.to}; else if(LOOP.on) s={...S,steps:1};} return [s.base,s.beta,s.K,s.clip,s.eta,s.steps].join('_');};
let rsHash='';
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==6) return null;
  const o={base:+p[0],beta:+p[1],K:+p[2],clip:+p[3],eta:p[4],steps:+p[5]};
  if(![0,1].includes(o.base)||!BETAS.includes(o.beta)||!KS.includes(o.K)||![0,1].includes(o.clip)||!ETA[o.eta]||!Number.isInteger(o.steps)||o.steps<0||o.steps>MAXS) return null;
  return o;}
function act(a,btn){
  if(a==='loop-play'){if(LOOP.on){stopLoop(); render(); return;} playLoop(); return;}
  if(a==='pass-play'){if(PASS.on){stopPass(); render(); return;} playPass(); return;}
  if(a==='run-play'){if(RUN.on){stopRun(); render(); return;} playRun(S.steps>=runTarget()?0:S.steps,runTarget()); return;}
  if(a==='run-replay'){playRun(0,runTarget()); return;}
  if(a==='reset'){stopRun(); stopLoop(); stopPass(true); userAct=true; try{const s=STEPS[step]; Object.assign(S,mode==='guided'&&s.enter?s.enter():(mode==='challenge'?{...CH,...CH0}:DEFAULT)); render();}finally{userAct=false;} return;}
  if(a==='submit'){chSubmitted=true; if(chPoint({beta:S.beta,K:S.K,clip:S.clip}).ok) chReveal=true; renderRail(); keepStage(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ppo-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,()=>{}); }catch(err){}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function showMe(sm){
  if(sm==='loop'){playLoop(); return;}
  const s=STEPS[step]; if(s.levers==='base'){setState({base:0},'quiet'); setState({base:1},'lever'); return;}
  if(s.levers==='beta'){setState({beta:0},'quiet'); setState({beta:0.05},'lever'); return;}
  if(s.levers==='clip'){setState({K:4,clip:0},'quiet'); setState({K:4,clip:1},'lever'); return;}
  setState(sm);}

/* ---------- interview drill (Connect): seeded, so a question can be reproduced ---------- */
let drillSeed=7, drillOpen=false;
function makeDrill(seed){
  const rnd=mulberry(seed*7919+13), r2=v=>Math.round(v*100)/100, pick=a=>a[Math.floor(rnd()*a.length)];
  for(let tries=0;tries<50;tries++){
    const r=r2(0.2+rnd()*0.7), pOld=r2(0.1+rnd()*0.5), pRef=r2(0.1+rnd()*0.5), V=r2(r-0.12+rnd()*0.24), rho=pick([0.7,0.85,1.1,1.3,1.5]);
    const x=Math.round(Math.log(pOld/pRef)*1e4)/1e4, R=Math.round((r-0.05*x)*1e4)/1e4, Av=Math.round((R-V)*1e4)/1e4;
    if(Math.abs(Av)<0.01||pOld===pRef) continue;
    const c=Math.min(Math.max(rho,0.8),1.2), u=Math.round(rho*Av*1e4)/1e4, k=Math.round(c*Av*1e4)/1e4, obj=Math.min(u,k);
    const flat=(Av>0&&rho>1.2)||(Av<0&&rho<0.8);
    return {r,pOld,pRef,V,rho,x,R,A:Av,c,u,k,obj,flat};}
  return null;}
const fx4=v=>{const s=Math.abs(v).toFixed(4); return (v<0&&+s!==0?MINUS:'')+s;};
const pm4=v=>v<0?`(${fx4(v)})`:fx4(v);
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#ppo-drill-q',app), a=$('#ppo-drill-a',app); if(!d||!q) return;
  q.innerHTML=tx(`A response in a PPO batch. The reward model scores it <b>${d.r.toFixed(2)}</b>. The policy that wrote it gave it probability <b>${d.pOld.toFixed(2)}</b>; the reference model gives <b>${d.pRef.toFixed(2)}</b>. The value model expects <b>${d.V.toFixed(2)}</b> for this prompt. β = 0.05, ε = 0.2. On the second pass its probability ratio is <b>${d.rho}</b>. What are its reward, its advantage and its clipped objective, and is it still pushed?`,
    `PPO 一批回答里的一个回答。奖励模型给它 <b>${d.r.toFixed(2)}</b> 分。写出它的策略模型给它的概率是 <b>${d.pOld.toFixed(2)}</b>，参考模型给的是 <b>${d.pRef.toFixed(2)}</b>。价值模型对这道题的预期是 <b>${d.V.toFixed(2)}</b>。β = 0.05，ε = 0.2。学第二遍时，它的概率之比是 <b>${d.rho}</b>。它的奖励、优势和截断后的目标各是多少？它还会被推吗？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const dir=d.A>0?tx('up','往上推'):tx('down','往下压');
  /* H2 (the owner, 2026-10-09): one short block per quantity, each named on the left with the numbers under the formula */
  const AH=tx('advantage','优势')+' '+M('\\hat{A}'), sgn=d.A>0?'> 0':'< 0', pt=v=>v<0?`({${fx4(v)}})`:fx4(v);
  const blocks=`<p>${tx('The reward is the score minus the KL penalty:','奖励是分数扣掉 KL 惩罚：')}</p>`+
    EQ(tx('reward','奖励')+' '+M('R'),['r - \\beta\\, \\log \\frac{\\pi_{old}}{\\pi_{ref}}',`${d.r.toFixed(2)} - 0.05 \\times \\log \\frac{${d.pOld.toFixed(2)}}{${d.pRef.toFixed(2)}}`,`${d.r.toFixed(2)} - 0.05 \\times ${pt(d.x)}`,fx4(d.R)])+
    `<p>${tx('The advantage subtracts the value model’s baseline:','优势再减去价值模型的基准：')}</p>`+
    EQ(AH,['R - V(x)',`${fx4(d.R)} - ${d.V.toFixed(2)}`,fx4(d.A)])+
    `<p>${tx('The clipped objective takes the smaller of the two terms:','截断的目标取两项里较小的那个：')}</p>`+
    EQ(tx('objective','目标'),['\\min\\left(\\rho \\hat{A}, \\text{clip}(\\rho, 0.8, 1.2) \\hat{A}\\right)',`\\min(${d.rho} \\times ${pt(d.A)}, ${d.c} \\times ${pt(d.A)})`,`\\min(${fx4(d.u)}, ${fx4(d.k)})`,fx4(d.obj)]);
  a.innerHTML=blocks+tx(`<p>${d.flat?`Not pushed: with ${M(`\\hat{A} ${sgn}`)} and ${M(d.A>0?'\\rho > 1.2':'\\rho < 0.8')} the clipped term is the smaller one and does not change with ρ, so its gradient is 0.`:`Still pushed ${dir==='up'?'up':'down'}: the unclipped term ${M('\\rho \\hat{A}')} is the one that counts, so the gradient is ${M('\\rho \\hat{A} \\cdot ∇ \\log \\pi')}, and ${M(`\\hat{A} ${sgn}`)}.`}</p>`,
    `<p>${d.flat?`不再被推：${M(`\\hat{A} ${sgn}`)} 且 ${M(d.A>0?'\\rho > 1.2':'\\rho < 0.8')}，起作用的是截断的那一项，它不随 ρ 变，梯度是 0。`:`还会被${dir}：起作用的是没截断的 ${M('\\rho \\hat{A}')}，梯度是 ${M('\\rho \\hat{A} \\cdot ∇ \\log \\pi')}，而 ${M(`\\hat{A} ${sgn}`)}。`}</p>`);}

/* ---------- capability layer ---------- */
function EXITS(){
  const r0=RM[2][0], v=0.24, b=0.05, lr=Math.log(0.30/0.25), R=r0-b*lr, Aq=R-v;
  const ex2={rho:1.5,A:2,eps:0.2}; const l1=ex2.rho*ex2.A, l2=Math.min(Math.max(ex2.rho,1-ex2.eps),1+ex2.eps)*ex2.A;
  /* H2: the clipped objective of the transfer question, broken at “=” onto lines that line up */
  const CLIPEQ=nm=>EQ(nm,['\\min\\left(\\rho \\hat{A}, \\text{clip}(\\rho, 0.8, 1.2) \\hat{A}\\right)','\\min(1.5 \\times 2, 1.2 \\times 2)',`\\min(${l1.toFixed(1)}, ${l2.toFixed(1)})`,l2.toFixed(1)]);
  return {core:()=>chSubmitted,
   explain:[
    {q:tx('Why does PPO subtract the value model’s baseline from the reward?','PPO 为什么要从奖励里减去价值模型给的基准？'),
     o:[tx('The raw score mixes how easy the prompt is with how good the response is; subtracting what the prompt usually scores leaves how much better than expected this response is, so better ones go up and worse ones go down.','原始分数里混着题目难易和回答好坏；减去这道题通常能拿的分，剩下的才是这个回答比预期好多少，于是好的往上推、差的往下压。'),
        tx('The reward model’s scores are too large for the optimizer, so the baseline rescales them to a range where the learning rate works, the way normalization does.','奖励模型的分数对优化器来说太大，基准把它们缩到学习率合适的范围，相当于做了一次归一化。'),
        tx('The baseline is the reward model’s score of the SFT model’s response, so the advantage measures how far the policy has moved from the SFT model.','基准是奖励模型给 SFT 模型的回答打的分，所以优势衡量的是策略模型离 SFT 模型走了多远。')],a:0,
     n:[null,tx('Subtracting a number does not rescale anything: the spread of the pushes stays the same, only their center moves to zero. What changes is the sign: responses worse than expected now go down.','减去一个数并不会缩放：推力之间的差距不变，只是中心挪到了 0。变的是方向：比预期差的回答现在往下压。'),
       tx('Distance from the SFT model is what the KL penalty measures. The baseline is the value model’s estimate of the average reward for this prompt.','离 SFT 模型多远，是 KL 惩罚衡量的事。基准是价值模型对这道题平均奖励的估计。')],
     why:tx(`On an easy prompt every response scores high and on a hard one every response scores low. The advantage ${M('\\hat{A} = R - V(x)')} removes what the prompt usually gets, so only the difference from the expectation pushes. Training then no longer depends on which responses happen to be written, and another seed gives about the same result.`,`简单的题每个回答都得高分，难的题都得低分。优势 ${M('\\hat{A} = R - V(x)')} 去掉了这道题通常能拿的分，只留下和预期的差来决定推力。这样训练就不再看碰巧写出了哪些回答，换个种子，结果也差不多。`)},
    {q:tx('Without the KL penalty, the reward-model score kept rising while true quality fell. Why does the KL penalty stop that?','不加 KL 惩罚时，奖励模型的分数一直在涨，真实质量却在跌。KL 惩罚为什么能挡住？'),
     o:[tx('Each response loses some reward: the more often the policy writes it than the SFT model did, the more it loses. So the policy cannot pile its probability onto the padded response.','每个回答都要扣掉一些奖励：策略模型越是比 SFT 模型更常写它，扣得越多。所以策略模型没法把概率都堆到注水的回答上。'),
        tx('The KL penalty lowers the reward model’s score for long responses, so it removes the reward model’s bias toward length.','KL 惩罚会压低奖励模型给长回答的分数，所以它去掉了奖励模型对长度的偏爱。'),
        tx('The KL penalty slows every update by the same factor, so the policy simply runs out of steps before it finds the padded response.','KL 惩罚把每次更新都按同一个比例放慢，策略还没找到注水的回答，步数就用完了。')],a:0,
     n:[null,tx('The reward model’s scores never change: E still scores highest. The penalty depends on how often the policy writes a response compared with the SFT model, whatever its length.','奖励模型的分数从来没变，E 的分数仍然最高。惩罚只看策略模型比 SFT 模型多写了多少，和长度无关。'),
       tx('It does not slow anything evenly: the penalty is zero for a response written as often as before and grows the more a response is overused. With β = 0.05 the run reached its end state long before step 200 and stayed there.','它不是均匀地放慢：和以前写得一样多的回答不扣，用得越多扣得越多。β = 0.05 时，训练远在第 200 步之前就到了终点，然后一直停在那里。')],
     why:tx(`The penalty is ${KLT()} per response. Pushing E far above its SFT probability costs more and more, so the policy settles close to the reference and padding cannot drag true quality down. In real training this also keeps the policy where the reward model can still be trusted.`,`每个回答的惩罚是 ${KLT()}。把 E 推得远高于 SFT 模型的概率，扣得越来越多，所以策略模型最后停在参考模型附近，注水拖不垮真实质量。真实训练里，这也让策略模型留在奖励模型还靠得住的地方。`)}],
   transfer:[
    {q:tx(`With the KL penalty, a response the policy writes more often than the reference model does loses a little reward. Now take a new prompt. The reward model scores one of its responses ${f2(r0)}, and the value model expects ${f2(v)} for this prompt. The policy writes this response with probability 0.30, the reference model with 0.25, and β = 0.05. About how large is this response’s advantage?`,`有了 KL 惩罚，策略模型比参考模型更常写一个回答，这个回答的奖励就要扣掉一点。现在换一道新提问。奖励模型给这道提问的一个回答打 ${f2(r0)} 分，价值模型对这道提问的预期是 ${f2(v)}。策略模型写出这个回答的概率是 0.30，参考模型是 0.25，β 还是 0.05。这个回答的优势大约是多少？`),
     o:[tx(`A little less than ${sg(r0-v,2)}`,`比 ${sg(r0-v,2)} 小一点`),tx(`Exactly ${sg(r0-v,2)}`,`正好是 ${sg(r0-v,2)}`),tx(`About ${sg(r0-b*lr,2)}`,`约 ${sg(r0-b*lr,2)}`)],a:0,
     n:[null,tx(`That leaves out the KL penalty: the policy writes this response more often than the reference model, so its reward is a little below ${f2(r0)}.`,`这漏掉了 KL 惩罚。策略模型比参考模型更常写这个回答，它的奖励要比 ${f2(r0)} 少一点。`),tx('That is the reward after the KL penalty. The advantage also subtracts the value model’s baseline.','这是扣掉 KL 惩罚后的奖励。优势还要再减去价值模型的基准。')],
     why:tx(`The reward loses a little to the KL penalty, then the baseline ${f2(v)} is subtracted: the advantage is ${DEF(sg(Aq,3),`log(0.30 / 0.25) = ${f(lr,4)}. Reward = ${f2(r0)} − 0.05 × ${f(lr,4)} = ${f(R,4)}. Advantage = ${f(R,4)} − ${f2(v)} = ${sg(Aq,4)}.`)}. It is positive, so this response is pushed up, a little less than without the penalty.`,`奖励先被 KL 惩罚扣掉一点，再减去基准 ${f2(v)}，优势是 ${DEF(sg(Aq,3),`log(0.30 / 0.25) = ${f(lr,4)}。奖励 = ${f2(r0)} − 0.05 × ${f(lr,4)} = ${f(R,4)}。优势 = ${f(R,4)} − ${f2(v)} = ${sg(Aq,4)}。`)}。它是正的，所以这个回答被往上推，只是比不扣惩罚时少推一点。`)},
    {q:tx('A batch is on its second pass. A response with advantage +2 already has a new-to-old probability ratio of 1.5. With ε = 0.2, what does the clipped objective give for it, and is it still pushed up?','一批回答正在学第二遍。一个优势为 +2 的回答，新旧概率之比已经是 1.5。ε = 0.2 时，截断的目标给它多少，它还会被往上推吗？'),
     o:[tx(`${l2.toFixed(1)}, and no: it is no longer pushed`,`${l2.toFixed(1)}，不会，不再被推`),tx(`${l1.toFixed(1)}, and yes: it is pushed up further`,`${l1.toFixed(1)}，会，继续往上推`),tx(`${l2.toFixed(1)}, and yes: it is pushed back down toward 1.2`,`${l2.toFixed(1)}，会，被往回压到 1.2`)],a:0,
     n:[null,tx(`That is the unclipped ${M('\\rho \\hat{A} = 1.5 \\times 2')}. The objective takes the smaller of ${M('\\rho \\hat{A}')} and ${M('\\text{clip}(\\rho) \\hat{A}')}.`,`这是没截断的 ${M('\\rho \\hat{A} = 1.5 \\times 2')}。目标取 ${M('\\rho \\hat{A}')} 和 ${M('\\text{clip}(\\rho) \\hat{A}')} 中较小的那个。`),tx(`Clipping never pushes back: above ${M('1 + \\varepsilon')} with a positive advantage the objective is flat, so its gradient is 0.`,`截断不会往回压：优势为正、比值超过 ${M('1 + \\varepsilon')} 时，目标是平的，梯度是 0。`)],
     why:tx(`${CLIPEQ(tx('objective','目标'))}The clipped term does not change with ρ, so the gradient is 0 and this response is not pushed any further in this batch.`,`${CLIPEQ(tx('objective','目标'))}截断的那一项不随 ρ 变，所以梯度是 0，这一批里这个回答不会再被往上推。`)}],
   practice:[{id:'rl-for-llms',d:tx('Questions 2 and 3: tabulate the clipped objective and its gradient by the region of ρ and the sign of A, then explain what the critic estimates in LLM post-training and what it costs.','第 2、3 题：按 ρ 所在的区间和 A 的符号，列出截断目标和它的梯度。再讲清大模型后训练里，价值模型（critic）估计的是什么、代价是什么。')}]};
}

/* ---------- term tooltips and jumps ---------- */
const CTLN={beta:{en:'KL coefficient β',zh:'KL 系数 β'},K:{en:'Passes over each batch',zh:'每批回答学几遍'},clip:{en:'Clipping',zh:'截断'},base:{en:'Value model',zh:'价值模型'},eta:{en:'Step size',zh:'步长'},steps:{en:'Training steps',zh:'训练步数'}};
const TERM={beta:()=>tx(`β weighs the KL penalty: each response’s reward loses ${KLT()}.`,`β 是 KL 惩罚的系数：每个回答的奖励扣掉 ${KLT()}。`)};
const locked=k=>mode==='challenge'&&!!CTLN[k]&&!['beta','K','clip'].includes(k);
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='bars') return tx('Click to jump to the response bars.','点击跳到回答的概率条。');
  if(k==='clip') return tx('Click to jump to the chart of step 1’s batch.','点击跳到第 1 步这一批回答的图。');
  if(!CTLN[k]) return '';
  if(locked(k)) return tx(`Fixed during the challenge: “${L(CTLN[k])}”.`,`挑战中固定不变：${zsp(L(CTLN[k]))}。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);}
function jump(k,src){
  if(!app) return;
  if(k==='controls'){const c=$('#ppo-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='bars'){const el=$('#ppo-bars',app); A.tip.hide(); el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el); return;}
  if(k==='clip'){const el=$('#ppo-clip',app); if(!el||el.hidden) return; A.tip.hide(); el.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el); return;}
  if(!CTLN[k]) return;
  if(locked(k)){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=$('#ppo-ic-'+k,app)||$('#ppo-ctl-'+k,app); if(el) A.ui.jumpToEl(el);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='steps'){stopRun(); stopLoop(); userAct=true; try{S.steps=+e.target.value; render();}finally{userAct=false;}}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    const inRail=!!b.closest('#ppo-rail');
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='view'){view=d.v; render(); return;} if(k==='pass'){stopPass(true); clipPass=+d.v; render(); return;} if(k==='runSpd'){runSpd=d.v; syncControls(); return;} if(k==='loopSpd'){loopSpd=d.v; syncControls(); return;} if(k==='passSpd'){passSpd=d.v; syncControls(); drawClip(); return;}
      userAct=true; try{setState({[k]:k==='eta'?d.v:+d.v},b.closest('#ppo-controls')?'settings':'lever');}finally{userAct=false;} if(inRail) keepStage(); return;}
    if(d.lever!=null&&d.lkey){const lv=LEV[d.lkey], o=lv.opts.find(x=>String(x.v)===d.lever); if(o){userAct=true; try{setState(lv.set(o.v),'lever');}finally{userAct=false;} renderRail(); keepStage();} return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); keepStage(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ppo-rail [data-pred="${pred}"]`,app); if(fb) fb.focus(); keepStage(); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ppo-rail',app)); return;}
    if(d.act){act(d.act,b); if(inRail&&!d.jump&&!b.closest('[data-exits]')) keepStage();}
  });
}
/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12 07:13 UTC, revised after the first-reader check): three text-and-formula
   sections read before the guided parts. They give the why and the mechanism (the policy and its objective, the log-derivative trick,
   the baseline that leaves the gradient unbiased) and end by naming PPO and its value model V(x); how strong each effect is stays with
   the guided parts, so no number of the toy, no KL penalty and no clipping here. Symbols follow the lab (π_θ, x, y, R, Â, V(x)) and the
   reward-model primer (r(x, y)); J(θ) is the expected reward and b(x) the baseline (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=(id,en,zh)=>{const a=A.shell.labLink(id,tx(en,zh)); return tx(a,`“${a}”`);};
// an inline formula and the punctuation right after it never part at a line break
const prNW=h=>h.replace(/(<math\b(?:(?!<\/math>)[\s\S])*<\/math>)([。，；：、）,.;:)]+)/g,'<span class="ppo-nw">$1$2</span>');   // one <math> only
// a paragraph per item; a formula block (a <div>) stands on its own
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return /^<div/.test(h)?h:`<p>${prNW(h)}</p>`;}).join('')});
// a formula on its own line: its pieces (each starting at “=”) stay whole and wrap only between them, so nothing scrolls sideways
const prFX=(...xs)=>`<div class="ppo-prfx">${xs.map(x=>`<span>${prM(x)}</span>`).join('')}</div>`;
const pE='\\mathbb{E}_{x}\\, \\mathbb{E}_{y \\sim \\pi_\\theta(\\cdot|x)}', pE2='\\mathbb{E}_{x}\\, \\mathbb{E}_{y \\sim \\pi_\\theta}', pGL='\\nabla_\\theta \\log \\pi_\\theta(y|x)';
A.registerPrimer(ID,{min:9,sections:[
 {h:()=>tx('Improving a chat model with reinforcement learning: policy, rollout, reward','用强化学习改进聊天模型：策略、尝试（rollout）、奖励'),parts:[prP(
   ()=>tx(`A chat model after SFT (see ${prLK('sft','SFT','SFT')}) learned by imitating answers people wrote: each prompt comes with a ready answer, and training makes the model more likely to write it. A reward model (see ${prLK('reward-model','Reward model','奖励模型')}) gives a different signal: for any response, including one the model has just written, it gives a score, its estimate of how good the response is, but it does not tell the model what to write. With a score and no ready answer, SFT’s way of training does not apply, because there is no answer whose probability to raise.`,
          `经过 SFT 的聊天模型（见${prLK('sft','SFT','SFT')}）是照着人写好的回答学的：每道提问配一个现成的回答，训练让模型更容易写出它。奖励模型（见${prLK('reward-model','Reward model','奖励模型')}）给的是另一种信号：任意一个回答，包括模型刚写出来的，它都能打一个分，表示它估计这个回答有多好，却不告诉模型该写什么。只有分数、没有现成的回答，SFT 的办法就用不上，因为找不到一个回答去提高它的概率。`),
   ()=>tx(`The way out is to let the model write. Given a prompt, it draws one token after another from its own next-token probabilities and so writes a response (see ${prLK('architecture','Architecture overview','结构总览')}); the reward model scores the response; then the parameters are adjusted so that high-scoring responses become more likely to be written. Training by trying and improving by the score is called reinforcement learning (RL). RL on a reward model’s scores is the last step of RLHF (see ${prLK('sft','SFT','SFT')}).`,
          `办法是让模型自己写：给它一道提问，它按自己给出的下一个 token 的概率，一个接一个地抽出 token，写成一个回答（见${prLK('architecture','Architecture overview','结构总览')}）；奖励模型给这个回答打分；再调整参数，让得分高的回答以后更容易被写出来。这种靠自己尝试、按得分改进的训练，叫强化学习（reinforcement learning，RL）。按奖励模型的分数做强化学习，就是 RLHF 的最后一步（见${prLK('sft','SFT','SFT')}）。`),
   ()=>tx(`RL’s terms come from a kind of problem about making a series of decisions: in each situation pick an action, then receive a reward. The probabilities of picking each action in each situation are called the policy. A language model is exactly a policy: the situation is the prompt plus the tokens written so far, the action is the next token, and the next-token probabilities are the probabilities of picking each action. So the chat model being trained is called the policy model, written ${prM('\\pi_\\theta')}, where ${prM('\\theta')} is all its parameters; the probability that it writes the whole response ${prM('y')} to the prompt ${prM('x')} is ${prM('\\pi_\\theta(y|x)')}. One response the policy writes is one try, called a rollout. The score a response gets is called its reward, written ${prM('R(x, y)')}. Here we first take ${prM('R(x, y) = r(x, y)')}, the reward model’s score.`,
          `RL 里的叫法，来自一类做一连串决定的问题：在每个情形下选一个动作，然后得到奖励。在每个情形下选各个动作的概率，叫策略（policy）。语言模型正好是一个策略：情形是提问加上已经写出的 token，动作是下一个 token，下一个 token 的概率就是选各个动作的概率。所以正在训练的聊天模型叫策略模型，记作 ${prM('\\pi_\\theta')}，${prM('\\theta')} 是它的全部参数；它对提问 ${prM('x')} 写出整个回答 ${prM('y')} 的概率，记作 ${prM('\\pi_\\theta(y|x)')}。策略模型写出的一个回答是一次尝试，叫 rollout。回答得到的分数叫奖励，记作 ${prM('R(x, y)')}。这里先取 ${prM('R(x, y) = r(x, y)')}，也就是奖励模型给的分数。`),
   ()=>tx(`The goal of training is for the policy’s responses to score as high as possible on average. The prompt ${prM('x')} is drawn from the training prompts, and the response ${prM('y')} is sampled by the policy itself:`,
          `训练的目标，是让策略模型写出的回答平均得分尽量高。提问 ${prM('x')} 从训练用的提问里抽，回答 ${prM('y')} 由策略模型自己抽样写出：`),
   ()=>prFX('J(\\theta)',`= ${pE} [\\, R(x, y) \\,]`),
   ()=>tx(`Gradient descent makes the loss ${prM('L(\\theta)')} smaller (see ${prLK('backprop','Backprop','反向传播')}); here ${prM('J(\\theta)')} should grow, so each step goes along the gradient instead of against it, which is called gradient ascent:`,
          `梯度下降让 loss ${prM('L(\\theta)')} 变小（见${prLK('backprop','Backprop','反向传播')}）；这里要让 ${prM('J(\\theta)')} 变大，每一步就顺着梯度走，叫梯度上升：`),
   ()=>prFX('\\theta \\leftarrow \\theta + \\eta\\, \\nabla_\\theta J(\\theta)'),
   ()=>tx(`${prM('\\nabla_\\theta J')} is the gradient of ${prM('J')} with respect to all the parameters, ${prM('\\partial J / \\partial \\theta')}, and ${prM('\\eta')} is the learning rate. Unlike in SFT, this expectation is over responses the policy itself writes, and when ${prM('\\theta')} changes, the distribution of those responses changes with it. So the responses used for training are not a fixed dataset: during training the current model has to keep writing new ones.`,
          `${prM('\\nabla_\\theta J')} 是 ${prM('J')} 对全部参数的梯度，也就是 ${prM('\\partial J / \\partial \\theta')}；${prM('\\eta')} 是学习率。和 SFT 不同，这个期望是对策略模型自己写的回答取的，${prM('\\theta')} 一变，这些回答的分布也跟着变。所以训练用的回答不是一份固定的数据集，要在训练过程中不断让当前的模型重新写。`))]},
 {h:()=>tx('Policy gradient: what gets adjusted is the probability of writing this response','策略梯度：调的是写出这个回答的概率'),parts:[prP(
   ()=>tx(`Gradient ascent first needs ${prM('\\nabla_\\theta J(\\theta)')}, but the reward a sampled response gets cannot be differentiated as a function of ${prM('\\theta')}. The response is written by sampling: with the random numbers used for sampling held fixed, change ${prM('\\theta')} a little and the same tokens are usually drawn, so the score does not move at all; change it more and at some step a different token is drawn, and the score jumps somewhere else. Between ${prM('\\theta')} and ${prM('R')} lies a draw, and the chain rule cannot get across it. Computing the expectation exactly from its definition does not work either: a response of a few hundred tokens can be written in as many ways as the vocabulary size to the power of a few hundred.`,
          `梯度上升先要求出 ${prM('\\nabla_\\theta J(\\theta)')}，可抽到的回答拿到的奖励，作为 ${prM('\\theta')} 的函数没法求导。回答是抽样写出来的：固定抽样用的随机数，${prM('\\theta')} 改一点点，抽到的往往还是同样的 token，分数纹丝不动；改多一点，某一步抽到了另一个 token，分数一下子跳到别处。从 ${prM('\\theta')} 到 ${prM('R')} 之间隔着一次抽样，链式法则接不上。按定义把期望精确算出来也不行：一个几百 token 的回答，可能的写法多达词表大小的几百次方种。`),
   ()=>tx(`The way out is to differentiate not the sampled response but its probability. The distribution of prompts does not depend on ${prM('\\theta')}, so take one fixed prompt ${prM('x')}. Treat the whole response ${prM('y')} as one draw and write the expectation as a weighted sum over all possible responses (the sum appears only in the derivation and never has to be computed). ${prM('\\theta')} appears only in the weights ${prM('\\pi_\\theta(y|x)')}, and the weights can be differentiated with respect to ${prM('\\theta')}:`,
          `出路是不对抽到的回答求导，而对它的概率求导。提问的分布和 ${prM('\\theta')} 无关，所以只看一道固定的提问 ${prM('x')}。把整个回答 ${prM('y')} 当作一次抽样，期望写成对所有可能回答的加权求和（这个求和只在推导里出现，不用真算）。${prM('\\theta')} 只出现在权重 ${prM('\\pi_\\theta(y|x)')} 里，而权重对 ${prM('\\theta')} 可导：`),
   ()=>prFX('\\nabla_\\theta \\sum_{y} \\pi_\\theta(y|x)\\, R(x, y)','= \\sum_{y} R(x, y)\\, \\nabla_\\theta \\pi_\\theta(y|x)'),
   ()=>tx(`Since ${prM('\\nabla_\\theta \\log \\pi_\\theta = \\nabla_\\theta \\pi_\\theta / \\pi_\\theta')}, we have ${prM('\\nabla_\\theta \\pi_\\theta = \\pi_\\theta\\, \\nabla_\\theta \\log \\pi_\\theta')}. Substituted in, the sum turns back into an expectation over ${prM('\\pi_\\theta')}:`,
          `因为 ${prM('\\nabla_\\theta \\log \\pi_\\theta = \\nabla_\\theta \\pi_\\theta / \\pi_\\theta')}，所以 ${prM('\\nabla_\\theta \\pi_\\theta = \\pi_\\theta\\, \\nabla_\\theta \\log \\pi_\\theta')}。代进去，求和又变回了对 ${prM('\\pi_\\theta')} 的期望：`),
   ()=>prFX('\\nabla_\\theta J(\\theta)',`= ${pE2} [\\, R(x, y)\\, ${pGL} \\,]`)),
  prP(
   ()=>tx(`This is the policy gradient. The right side is an expectation, so it can be estimated by sampling: let the policy write a few responses to each prompt of a batch, compute ${prM(`R\\, ${pGL}`)} for each, and average. The expectation of this average is exactly the true gradient, so it is an unbiased estimate. The reward only enters as a number multiplying each term, so the reward model does not need to be differentiable. Doing gradient ascent with this average directly, each response adjusting its probability by its own reward, is the most basic policy-gradient method, called REINFORCE, or push by the score.`,
          `这就是策略梯度（policy gradient）。右边是一个期望，就能用抽样来估：让策略模型对一批提问各写几个回答，每个回答算出 ${prM(`R\\, ${pGL}`)}，再取平均。这个平均的期望正好是真正的梯度，所以它是无偏估计。奖励只是作为一个数乘在每一项前面，奖励模型不需要能求导。直接拿这个平均做梯度上升，每个回答按它自己的奖励调整概率，是最基本的策略梯度方法，叫 REINFORCE，也可以叫按分数加强。`),
   ()=>tx(`The log probability of a whole response is easy to get: as in the SFT loss (see ${prLK('sft','SFT','SFT')}, where the model’s probability is written ${prM('p_\\theta')}), it is the sum of the log probabilities of the response’s tokens, except that SFT raises an answer a person wrote, and here it is a response the policy wrote itself:`,
          `整个回答的 log 概率也好算：和 SFT 的 loss 里一样（见${prLK('sft','SFT','SFT')}，那里把模型的概率记作 ${prM('p_\\theta')}），它是回答里各个 token 的 log 概率之和，只不过 SFT 推高的是人写的回答，这里是策略模型自己写的回答：`),
   ()=>prFX('\\log \\pi_\\theta(y|x)','= \\sum_{j=1}^{|y|} \\log \\pi_\\theta(y_j | x, y_{<j})'),
   ()=>tx(`${prM('y_j')} is the response’s ${prM('j')}-th token, ${prM('y_{<j}')} the tokens before it, and ${prM('|y|')} the number of tokens in the response. Each term is the log of a probability the forward pass computes, and one backward pass gives the gradients of all of them.`,
          `${prM('y_j')} 是回答的第 ${prM('j')} 个 token，${prM('y_{<j}')} 是它前面的 token，${prM('|y|')} 是回答的 token 数。每一项都是前向算出的概率取对数，它们的梯度，一趟反向传播就全部算出来。`),
   ()=>tx(`Read response by response, each sampled response adds one term. ${prM(pGL)} is the direction that makes this response, as a whole, more likely to be written. A response is fixed once written; what changes is the probability of writing it next time. ${prM('R')} multiplies that direction: its sign decides whether this response’s probability goes up or down, and its size decides how far.`,
          `按一个个抽到的回答来读，每个回答贡献一项。${prM(pGL)} 是让这个回答整体更容易被写出的方向。回答写出来就定了，调的是下一次写出它的概率。${prM('R')} 乘在这个方向上：它的正负决定这个回答的概率升还是降，它的大小决定动多少。`))]},
 {h:()=>tx('Baseline and advantage: how much better than expected','基准与优势：比预期好多少'),parts:[prP(
   ()=>tx(`What multiplies each term of the policy gradient is the value of the reward itself, yet the reward model only fixes score gaps (see ${prLK('reward-model','Reward model','奖励模型')}). The reward model is trained on preference pairs: each pair is two responses to the same prompt, marked with which one a person chose, and its loss uses only the gap between the two scores of a pair, ${prM('r_{\\text{chosen}} - r_{\\text{rejected}}')}. Adding the same number to all the responses of one prompt changes none of these gaps, and so not the loss either. So how high a prompt’s responses score overall is not set by the reward model’s training at all: ${prM('R')} contains a number that changes from prompt to prompt and means nothing.`,
          `策略梯度里乘在每一项前面的，是奖励本身的数值，可奖励模型只规定了分差（见${prLK('reward-model','Reward model','奖励模型')}）。奖励模型是用偏好对训练的：每一对是同一道提问的两个回答，标着人选中了哪一个；它的 loss 只用到同一对里两个分数之差 ${prM('r_{\\text{选中}} - r_{\\text{落选}}')}。给一道题的所有回答都加上同一个数，这些分差一个都不变，loss 也不变。所以一道题的回答整体得多少分，奖励模型的训练根本没有规定：${prM('R')} 里混着一个随提问而变、却没有意义的数。`),
   ()=>tx(`First see what shifting one prompt’s rewards does. For a prompt ${prM('x')}, subtract the same number ${prM('b(x)')} from the reward of every response. The policy gradient gains one term, ${prM('-b(x)')} times the expectation below, and that expectation is 0:`,
          `先看把一道题的奖励整体平移会怎样。对提问 ${prM('x')}，每个回答的奖励都减去同一个数 ${prM('b(x)')}，策略梯度就多出一项：${prM('-b(x)')} 乘以下面这个期望，而这个期望等于 0：`),
   ()=>prFX(`\\mathbb{E}_{y \\sim \\pi_\\theta(\\cdot|x)} [\\, ${pGL} \\,]`,`= \\sum_{y} \\pi_\\theta(y|x)\\, ${pGL}`,'= \\sum_{y} \\nabla_\\theta \\pi_\\theta(y|x)','= \\nabla_\\theta \\sum_{y} \\pi_\\theta(y|x)','= \\nabla_\\theta 1 = 0'),
   ()=>tx(`So the following holds for any ${prM('b(x)')}:`,`所以对任何 ${prM('b(x)')}，下式都成立：`),
   ()=>prFX('\\nabla_\\theta J(\\theta)',`= ${pE2} [\\, (R - b(x))\\, ${pGL} \\,]`),
   ()=>tx(`${prM('b(x)')} is called the baseline. It may differ from prompt to prompt, but it may not depend on which response was drawn this time: otherwise it could not be taken out of the expectation over ${prM('y')}, and the extra term would no longer be 0.`,
          `${prM('b(x)')} 叫基准（baseline）。它可以随提问变，但不能取决于这次抽到的是哪个回答：否则它不能提到对 ${prM('y')} 的期望外面，多出的那一项就不再是 0。`)),
  prP(
   ()=>tx(`The expectation is unchanged, but each sample’s term is not: the number multiplying ${prM(pGL)} changes from ${prM('R')} to ${prM('R - b(x)')}. Take ${prM('b(x)')} to be what this prompt is expected to score under the current policy, its average reward ${prM('\\mathbb{E}_{y \\sim \\pi_\\theta(\\cdot|x)}[R(x, y)]')}. When the reward model adds the same number to all the responses of this prompt, this average goes up by the same number, and ${prM('R - b(x)')} does not change at all. So no term is affected any more by the overall level that nothing set, and the sign of ${prM('R - b(x)')} says only whether this response scored above or below this prompt’s average.`,
          `期望不变，每个样本那一项却变了：乘在 ${prM(pGL)} 前面的数，从 ${prM('R')} 变成了 ${prM('R - b(x)')}。取 ${prM('b(x)')} 为这道题在当前策略下预期能拿的分，也就是平均奖励 ${prM('\\mathbb{E}_{y \\sim \\pi_\\theta(\\cdot|x)}[R(x, y)]')}。奖励模型给这道题的所有回答都加上同一个数时，这个平均也加上同一个数，${prM('R - b(x)')} 一点不变。于是每一项都不再受那个没有规定的整体高低影响，${prM('R - b(x)')} 的正负只说明这个回答比这道题的平均得分高还是低。`),
   ()=>tx(`${prM('R - b(x)')} is called the advantage, written ${prM('\\hat{A}(x, y)')}: how much better than expected this response is. The policy gradient becomes:`,
          `${prM('R - b(x)')} 叫优势（advantage），记作 ${prM('\\hat{A}(x, y)')}：这个回答比预期好多少。策略梯度写成：`),
   ()=>prFX('\\hat{A}(x, y) = R(x, y) - b(x)'),
   ()=>prFX('\\nabla_\\theta J(\\theta)',`= ${pE2} [\\, \\hat{A}(x, y)\\, ${pGL} \\,]`),
   ()=>tx(`The hat on ${prM('\\hat{A}')} marks an estimate: nobody knows a prompt’s exact average reward under the current policy, and it changes whenever the policy does, so it can only be estimated as training goes. PPO (proximal policy optimization) is the classic RL algorithm for the last step of RLHF. It updates the policy model by the advantage above, and trains a separate model to estimate ${prM('b(x)')}: it reads the prompt and outputs the reward expected on that prompt under the current policy. This model is called the value model, written ${prM('V(x)')}.`,
          `${prM('\\hat{A}')} 头上的帽子表示它是估出来的：一道题在当前策略下的平均奖励没有人知道确切值，策略一变它也跟着变，只能边训练边估。PPO（proximal policy optimization）是 RLHF 最后一步最经典的 RL 算法，它按上面的优势更新策略模型，并另外训练一个模型来估 ${prM('b(x)')}：读进提问，输出这道题在当前策略下的预期奖励。这个模型叫价值模型（value model），记作 ${prM('V(x)')}。`))]}
]});

/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings with the training steps reached, the part, the
   answer, the options tried in each part, the curve and the memory shown, the pass of step 1's batch shown, the runs, replays and
   passes already played to the end, the three playing speeds, the challenge's result, the drill. A run, replay or pass that was
   still playing is not resumed: it stays where it was (a run at the step it had reached) and plays again from its start. */
const rsKeys=Object.keys(DEFAULT), rsObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), rsPick=o=>Object.fromEntries(rsKeys.map(k=>[k,o[k]]));
const rsCfg=x=>rsObj(x)?parse(token(x)):null, rsWK=/^[01]\|(0|0\.05|0\.2)\|[14]\|[01]\|[nl](>\d{1,3})?$/, rsSpd=v=>['slow','med','fast'].includes(v);
A.registerLab({
  id:ID,
  /* a run, replay or pass still playing is recorded as played to its end (its playhead is not kept): after a reload it shows its end */
  snap(){const tr={}; for(const k in tried) if(tried[k] instanceof Set) tr[k]=[...tried[k]];
    const wr=new Set(watchedRun), wl=new Set(watchedLoop), wp=new Set(watchedPass); if(RUN.on) wr.add(RUN.key); if(LOOP.on) wl.add(cfgKey()); if(PASS.on) wp.add(PASS.key);
    const st={...S,steps:RUN.on?RUN.to:(LOOP.on?1:S.steps)};
    return {v:1,mode,step,S:rsPick(st),pred,observed:{...observed},saved:saved?rsPick(saved):null,tried:tr,view,memAlg,clipPass:PASS.on?0:clipPass,runSpd,loopSpd,passSpd,
      watchedRun:[...wr],watchedLoop:[...wl],watchedPass:[...wp],chSubmitted,chReveal,drillSeed,drillOpen,
      passed:mode==='challenge'&&chSubmitted&&chPoint({beta:S.beta,K:S.K,clip:S.clip}).ok};},
  restore(o){if(!rsObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...(rsCfg(o.S)||DEFAULT)}; if(mode==='challenge') S={...S,...CH};
    pred=PRED.includes(o.pred)?o.pred:null;
    observed={}; if(rsObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    saved=mode==='challenge'?rsCfg(o.saved):null;
    tried={}; if(rsObj(o.tried)) STEPS.forEach(s=>{const lv=s.levers&&LEV[s.levers], a=lv&&Array.isArray(o.tried[s.id])?o.tried[s.id].filter(v=>lv.opts.some(x=>x.v===v)):[]; if(a.length) tried[s.id]=new Set(a);});
    view=['rq','kl','rho','fan'].includes(o.view)?o.view:'rq'; memAlg=LEV.alg.opts.some(x=>x.v===o.memAlg)?o.memAlg:'ppo';
    clipPass=Number.isInteger(o.clipPass)&&o.clipPass>=0&&o.clipPass<=4?o.clipPass:0;
    runSpd=rsSpd(o.runSpd)?o.runSpd:'med'; loopSpd=rsSpd(o.loopSpd)?o.loopSpd:'med'; passSpd=rsSpd(o.passSpd)?o.passSpd:'med';
    [[watchedRun,o.watchedRun],[watchedLoop,o.watchedLoop],[watchedPass,o.watchedPass]].forEach(([w,a])=>{w.clear(); arr(a,v=>typeof v==='string'&&rsWK.test(v)).slice(0,300).forEach(v=>w.add(v));});
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:7; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    stopRun(); stopLoop(); stopPass(); root=el; rsHash='';
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#ppo-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ppo-rail',app)); keepStage();},token});
    bind(app);
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&!Object.keys(observed).length&&step===0) Object.assign(S,s.enter()); view=s.view||view;}
    renderRail(); renderDrill();
    if(p&&p.steps>=MAXS&&!watchedRun.has(cfgKey()+'>'+MAXS)) watchedRun.add(cfgKey()+'>'+MAXS);   // a shared link shows its end state
  },
  leave(){stopRun(); stopLoop(); stopPass(); stageStop(); clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const d=$('#ppo-math',app), open=d&&d.open, wasRun=RUN.on, to=RUN.to; stopRun(); if(wasRun) S.steps=to; if(PASS.on){stopPass(true); clipPass=0;} this.mount(root,null); if(open) $('#ppo-math',app).open=true;}},
  onResize(){if(app){if(seen('value')&&!clipShown()) drawCurve(compute(S)); drawClip();}},
  termTip(k,el){if(k==='cdot') return el?cdotTip(el.getAttribute('data-term')):''; const tm=TERM[k]; if(!tm) return ''; return `<div>${typeof tm==='function'?tm():esc(L(tm))}</div><div class="act">${jumpAct(k)}</div>`;},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#ppo-bars',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#ppo-bars',app),
    render({status,vis,read}){if(!app) return; const r=compute(S), mem=x=>`<span class="dk-mem">${x}</span>`;
      vis.innerHTML=`<div class="ppo-dk"><i class="ppo-dk-q" style="width:${(r.q*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const p=chPoint({beta:S.beta,K:S.K,clip:S.clip}); status.className='status '+(p.ok?'ok':'bad'); status.textContent=p.ok?tx('Goal met','已达标'):tx('Not yet','未达标');
        read.innerHTML=esc(tx(`After 200 steps true quality is ${f2(p.q)} and the KL ${f(p.kl,2)} nats`,`200 步后真实质量 ${f2(p.q)}，KL ${f(p.kl,2)} nats`))+mem(tx(` (β = ${bTxt(S.beta)}, ${S.K} passes, ${S.clip?'clipped':'no clipping'})`,`（β = ${bTxt(S.beta)}，学 ${S.K} 遍，${S.clip?'截断':'不截断'}）`)); return;}
      const bad=r.hack>=0||r.top.some(j=>j===4); status.className='status '+(bad?'bad':'ok'); status.textContent=r.top.some(j=>j===4)?tx('Reward hacking','reward hacking'):(bad?tx('Quality falling','质量在跌'):tx('Training','训练中'));
      read.innerHTML=esc(tx(`After ${stepsTxt(r.t)}: true quality ${f2(r.q)}, reward-model score ${f2(r.rm)}`,`训练 ${r.t} 步后：真实质量 ${f2(r.q)}，奖励模型的平均分 ${f2(r.rm)}`))+(seen('kl')?mem(tx(`, KL ${f(r.kl,2)} nats`,`，KL ${f(r.kl,2)} nats`)):'');}
  }
});

/* ---------- home tile teaser: one control with the title's two answers, 6 seeds, one sentence (R11.1, R11.5) ----------
   S13 of the review: the control steps through push by the score → add the value model → add the KL penalty, so the tile shows what
   each of the two (the value model and the KL penalty) fixes. Each setting is drawn for the 6 seeds of the lab's fans. */
const TZ_SET=[{base:0,beta:0},{base:1,beta:0},{base:1,beta:0.05}].map(p=>({...p,K:1,clip:0,eta:'n'}));
const TZ={render(el){
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'ppo-tz-set',key:'tzset',label:tx('Update rule','更新规则'),options:[['0',tx('Score only','只用奖励')],['1',tx('+ value model','+ 价值模型')],['2',tx('+ KL penalty','+ KL 惩罚')]],value:String(tzKL),inline:false})+
    `<svg class="ppo-tz" viewBox="0 0 260 96" aria-hidden="true"></svg>`+
    `<div class="tz-leg"><span><i class="ppo-tz-sw ppo-tz-q"></i>${tx('true quality, 6 seeds','真实质量，6 个种子')}</span><span><i class="ppo-tz-sw ppo-tz-ref"></i>${tx('SFT model','SFT 模型')}</span><span><i class="ppo-tz-sw ppo-tz-bad"></i>${tx('quality falling','质量下跌')}</span></div>`+
    `<p class="tz-read"></p>`+
    `<p class="tz-note">${tx('Toy run: your browser really trains a small policy model (16 parameters) with policy-gradient steps. It answers 3 prompts with 5 candidate responses each; the chart shows 6 seeds, 200 steps each.','玩具实跑：浏览器真的用策略梯度训练一个 16 个参数的小策略模型。它回答 3 道提问，每道有 5 种候选回答。图里画了 6 个种子，各训练 200 步。')}</p>`+
    `<p class="tz-note">${tx('The small reward model overrates a long, padded response, and true quality, the lab’s own score for each kind of response, is hidden in real training.','小奖励模型高估一种又长又注水的回答，真实质量是这个 Lab 给每种回答定的分，真实训练时看不到。')}</p>`+
    `<p class="tz-note">${tx('β = 0.05 is this lab’s default, also used by the small models of the GRPO and RLVR labs and the default kl_coef of TRL v1.12.0’s PPO. This lab’s β does not carry over to a real model.','β = 0.05 是这个 Lab 的默认值，GRPO 和 RLVR 两个 Lab 的小模型也用它，TRL v1.12.0 的 PPO 默认也是 0.05。这个 Lab 里的 β 不能照搬到真实模型。')}</p></div>`;
  const svg=el.querySelector('.ppo-tz'), rd=el.querySelector('.tz-read');
  const upd=()=>{const c=TZ_SET[tzKL], X=s=>8+244*s/MAXS, Y=v=>88-80*(v-0.2)/0.8, Ts=FAN.map(sd=>traj({...c,seed:sd})), ends=Ts.map(T=>T.q[MAXS]), lo=f2(Math.min(...ends)), hi=f2(Math.max(...ends));
    const p=(a,from,to)=>{let d=''; for(let s=from;s<=to;s++) d+=(s===from?'M':'L')+X(s).toFixed(1)+','+Y(a[s]).toFixed(1); return d;};
    svg.innerHTML=`<line class="ppo-tz-ax" x1="8" x2="252" y1="88" y2="88"/><line class="ppo-tz-l ppo-tz-ref" x1="8" x2="252" y1="${Y(Q0).toFixed(1)}" y2="${Y(Q0).toFixed(1)}"/>`+
      Ts.map(T=>{const hk=hackFrom(T,MAXS); return `<path class="ppo-tz-l ppo-tz-q" d="${p(T.q,0,MAXS)}"/>${hk>=0?`<path class="ppo-tz-l ppo-tz-bad" d="${p(T.q,hk,MAXS)}"/>`:''}`;}).join('');
    rd.innerHTML=tzKL===0?tx(`Push by the score: after 200 steps the 6 seeds end anywhere from ${lo} to <b>${hi}</b> in true quality.`,`只用奖励时，6 个种子训练 200 步，真实质量从 ${lo} 到 <b>${hi}</b> 都有。`)
      :tzKL===1?tx(`With the value model the 6 seeds rise together, then the policy learns to pad: true quality ends at ${lo}–<b>${hi}</b>.`,`加上价值模型，6 个种子一起先涨后跌，策略模型学会了注水，真实质量最后只有 ${lo} 到 <b>${hi}</b>。`)
      :(lo===hi?tx(`Adding the KL penalty (β = 0.05), all 6 seeds end at <b>${hi}</b>, a little above the SFT model’s ${f2(Q0)}.`,`再加上 KL 惩罚（β = 0.05），6 个种子最后都停在 <b>${hi}</b>，比 SFT 模型的 ${f2(Q0)} 略高。`)
        :tx(`Adding the KL penalty (β = 0.05), the 6 seeds end at ${lo}–<b>${hi}</b>; the SFT model: ${f2(Q0)}.`,`再加上 KL 惩罚（β = 0.05），6 个种子最后在 ${lo} 到 <b>${hi}</b> 之间，SFT 模型是 ${f2(Q0)}。`));};
  el.querySelectorAll('.seg-ctl[data-key="tzset"] button').forEach(b=>b.addEventListener('click',()=>{tzKL=+b.dataset.v; A.ui.setPressed(el.querySelector('.seg-ctl[data-key="tzset"]'),String(tzKL)); upd();}));
  upd();
}};
A.registerTeaser('ppo',TZ);

/* read-only hook for the golden-number tests */
window.__ppo=Object.freeze({train:c=>train({base:1,beta:0.05,K:4,clip:1,eta:'n',seed:SEED,...c}),compute:s=>{const r=compute({...DEFAULT,...s}); return {rm:r.rm,q:r.q,kl:r.kl,pi:r.pi,w:r.w,rmax:r.rmax,hack:r.hack,peak:r.peak};},
  slope:(a,rho,clip)=>slopeOf(a,rho,clip==null?1:clip),obj:(a,rho,clip)=>clipObj(a,rho,clip==null?1:clip),batch1:c=>batch1({...DEFAULT,...c}).map(d=>({k:d.k,y:d.y,a:d.a,n:d.n,rp:d.rp.slice(),on:d.on.slice(),end:d.end})),
  chAll,chPoint,memOf,runaway:c=>runaway(traj({...DEFAULT,...c})),fan:base=>FAN.map(sd=>traj({...P_LOOP,base,seed:sd}).q[30]),exits:()=>{const e=EXITS(); return {t0:e.transfer[0].o,t1:e.transfer[1].o,why0:e.transfer[0].why};},drill:sd=>makeDrill(sd),teaser:el=>TZ.render(el),
  many:c=>manyEnds({...DEFAULT,...c}),fanQ:(c,t)=>fanQ({...DEFAULT,...c},t),easyTops:()=>easyTops(),tzSets:TZ_SET,
  consts:Object.freeze({NP,NY,NS,MAXS,SEED,EPS,LRV,REF,RM,QL,LEN,ETA,BETAS,KS,FAN,CH,CH0,CH_KL,Q0,STEP_IDS:STEPS.map(s=>s.id)}),
  state:()=>({...S,mode,step,view,memAlg,pred,clipPass,running:RUN.on,looping:LOOP.on,loopPh:LOOP.ph,runStarts:nRunStarts,loopStarts:nLoopStarts,runSpd,loopSpd,passing:PASS.on,passStarts:nPassStarts,passSpd,runSecs:RUN_SECS,quickSecs:QUICK_SECS,passSecs:PASS_SPEEDS})});
})();
