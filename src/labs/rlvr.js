/* rlvr (world post): How does a model learn to reason from right or wrong alone?
   Fidelity: Simulated (toy run). A toy policy with 5 parameters chooses how to write a solution (4 ways, a softmax over 4 logits) and
   whether to put the final answer after "####" (one logit, a sigmoid). A program checks only the final answer. Training is real
   GRPO in the browser: 8 problems per step, G = 8 answers per problem, reward 1 / 0 (optional format score 0.1, TinyZero), advantage
   (r − mean) / (std + 1e-4) with the sample std (TRL), loss = mean over all answers of −A·ratio + β·k3 (β = 0.05), one update per
   batch (ratio = 1, so the clip ε = 0.2 never acts), a plain gradient step with learning rate 2 on the logits. Optional dynamic
   sampling (verl DAPO filter_groups): groups whose 8 rewards are equal are dropped and another problem is drawn.
   Random numbers: mulberry32, seed 11; per group attempt: the problem kind, then per answer: way, format, correctness (one uniform
   that decides both the strict and the loose verdict). Every curve is an exact expectation of the policy at that step.
   tests/rlvr_numbers.py reimplements the training independently and compares.
   Words (EN / ZH): way of writing / 写法; checker / 判分程序; strict / loose checker / 严格 / 宽松的判分程序; reward / 奖励 (分);
   advantage / 优势; group / 一组回答; format reward / 格式奖励; dynamic sampling / 跳过没有信号的题. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='rlvr';

/* =====================================================================
   engine
   ===================================================================== */
const NW=4, LEN=[25,140,240,60];                     // tokens per answer for each way of writing (lab's assumption)
const KINDS=4;                                        // easy, medium, hard, beyond
const QS=[[0.40,0.10,0.00,0],[0.80,0.52,0.22,0],[0.95,0.80,0.50,0],[0.30,0.08,0.02,0]];   // P(final answer right) [way][kind]
const QL=[QS[0],QS[1],QS[2],[0.98,0.95,0.85,0]];                                         // P(some number in it is right) [way][kind]
const SETS={main:[0.4,0.4,0.1,0.1],hard:[0,0.1,0.5,0.4],ch:[0.05,0.25,0.4,0.3]};
const P0=[0.68,0.25,0.05,0.02], PHI0={sft:50,base:Math.log(1/9)};   // σ(50) is exactly 1 in float64: the SFT model always writes ####
const G=8, B=8, BETA=0.05, EPS=0.2, LR=2, SEED=11, T=60, TCH=20, MAXTRY=64, FMT=0.1, KS=[1,2,4,8,16,32,64,128,256];
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const sig=x=>1/(1+Math.exp(-x));
function softmax(z){let m=-Infinity; for(const v of z) if(v>m) m=v; const e=z.map(v=>Math.exp(v-m)); const s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s);}
const xlog=(p,q)=>p>0?p*Math.log(p/q):0;
const pick=(u,p)=>{let a=0; for(let k=0;k<p.length;k++){a+=p[k]; if(u<a) return k;} return p.length-1;};
function evalPolicy(pi,pf,w,c){
  const pk=[0,1,2,3].map(k=>pf*pi.reduce((s,p,i)=>s+p*QS[i][k],0));
  const p1=pk.reduce((s,p,k)=>s+w[k]*p,0);
  const okAvg=pi.reduce((s,p,i)=>s+p*QS[i].reduce((a,q,k)=>a+w[k]*q,0),0);          // right answer, format aside
  const lAvg=pi.reduce((s,p,i)=>s+p*QL[i].reduce((a,q,k)=>a+w[k]*q,0),0);
  const rew=c.chk==='loose'?lAvg+(c.fmt?FMT*pf*(1-lAvg):0):p1+(c.fmt?FMT*pf*(1-okAvg):0);  // what the checker pays on average
  return {pk,p1,rew,len:pi.reduce((s,p,i)=>s+p*LEN[i],0)};
}
/* one training run: every step records the policy, then samples one batch (also at the last step, for the picture), then updates */
function train(c){
  const w=SETS[c.set], Tn=c.T, rnd=mulberry(SEED);
  let th=P0.map(Math.log), ph=PHI0[c.start];
  const pr=softmax(th), pfr=sig(ph), out=[];
  for(let t=0;t<=Tn;t++){
    const pi=softmax(th), pf=sig(ph), e=evalPolicy(pi,pf,w,c);
    const kl=pi.reduce((s,p,i)=>s+xlog(p,pr[i]),0)+xlog(pf,pfr)+xlog(1-pf,1-pfr);
    const groups=[]; let kept=0, tries=0;
    while(kept<B&&tries<MAXTRY){
      tries++;
      const kind=pick(rnd(),w), resp=[];
      for(let j=0;j<G;j++){
        const s=pick(rnd(),pi), f=rnd()<pf?1:0, u=rnd(), okS=u<QS[s][kind], okL=u<QL[s][kind];
        const r=c.chk==='loose'?(okL?1:(c.fmt&&f?FMT:0)):(f&&okS?1:(c.fmt&&f?FMT:0));
        resp.push({s,f,okS,okL,r});
      }
      const rs=resp.map(x=>x.r), m=rs.reduce((a,b)=>a+b,0)/G, sd=Math.sqrt(rs.reduce((a,b)=>a+(b-m)*(b-m),0)/(G-1));
      resp.forEach(x=>{x.a=(x.r-m)/(sd+1e-4);});
      const flat=rs.every(v=>v===rs[0]), drop=!!c.dyn&&flat;
      groups.push({kind,resp,m,sd,flat,drop});
      if(!drop) kept++;
      if(!c.dyn&&kept>=B) break;
    }
    const used=groups.filter(g=>!g.drop), N=used.length*G;
    out.push({t,pi,pf,...e,kl,groups,tries,flat:groups.filter(g=>g.flat).length});
    if(t===Tn||!N) continue;
    const gt=[0,0,0,0]; let gp=0;
    used.forEach(g=>g.resp.forEach(x=>{
      const lp=Math.log(pi[x.s])+(x.f?Math.log(pf):Math.log(1-pf)), lr=Math.log(pr[x.s])+(x.f?Math.log(pfr):Math.log(1-pfr));
      const cf=-x.a+BETA*(1-Math.exp(lr-lp));
      for(let k=0;k<NW;k++) gt[k]+=cf*((k===x.s?1:0)-pi[k])/N;
      gp+=cf*(x.f-pf)/N;}));
    th=th.map((v,k)=>v-LR*gt[k]); ph=ph-LR*gp;
  }
  return out;
}
const RC=new Map();
const cfgKey=c=>[c.chk,c.fmt?1:0,c.dyn?1:0,c.start,c.set,c.T].join('_');
function run(c){const k=cfgKey(c); let r=RC.get(k); if(!r){r=train(c); RC.set(k,r);} return r;}
function passK(pk,w){return KS.map(k=>pk.reduce((s,p,i)=>s+w[i]*(1-Math.pow(1-p,k)),0));}
/* what the loose checker pays on average on the main set: the candidates way against “solve, then check” (part 5) */
function looseAvgPay(){const w=SETS.main, avg=row=>row.reduce((a,q,k)=>a+w[k]*q,0); return {cand:avg(QL[3]),check:avg(QL[2])};}
/* Math-Verify README: three checkers score the same MATH outputs (quoted, Connect) */
const MV_SCORES=[0.0802,0.1288,0.1328];
/* the challenge: a new set, the SFT model, 20 steps; start from the loose checker; goal pass@1 ≥ 40% and candidates ≤ 5% */
const CH_GOAL=0.40, CH_G=0.05, CH0={chk:'loose',fmt:0,dyn:0,start:'sft',set:'ch',T:TCH};
const CHOPTS=[];['loose','strict'].forEach(chk=>[0,1].forEach(dyn=>[0,1].forEach(fmt=>CHOPTS.push({chk,dyn,fmt}))));
const nChanges=o=>(o.chk!==CH0.chk?1:0)+(o.dyn?1:0)+(o.fmt?1:0);
function chEval(o){const r=run({...CH0,...o}), e=r[r.length-1], okP=e.p1>=CH_GOAL-1e-12, okG=e.pi[3]<=CH_G+1e-12;
  return {...o,p1:e.p1,g:e.pi[3],okP,okG,ok:okP&&okG,n:nChanges(o),extra:r.reduce((s,x,i)=>i<r.length-1?s+x.tries-B:s,0)};}
function chBest(){const ok=CHOPTS.map(chEval).filter(e=>e.ok); if(!ok.length) return null; const m=Math.min(...ok.map(e=>e.n)); return ok.filter(e=>e.n===m);}

/* =====================================================================
   copy data: problems (示例), ways of writing, example answers
   ===================================================================== */
const PROB=[
 {q:{en:'Ming has 3 boxes of pencils with 6 in each box, and buys 2 more. How many pencils does he have now?',zh:'小明有 3 盒铅笔，每盒 6 支，他又买了 2 支。他现在一共有多少支铅笔？'},ans:20,wrong:[16,18,24],
  steps:{en:'3 × 6 = 18, and 18 + 2 = 20.',zh:'3 × 6 = 18，18 + 2 = 20。'},slip:{en:'3 × 6 = 18, and 18 − 2 = 16.',zh:'3 × 6 = 18，18 − 2 = 16。'},
  check:{en:'Check: 20 − 2 = 18 and 18 ÷ 6 = 3 boxes. It fits.',zh:'检查：20 − 2 = 18，18 ÷ 6 = 3 盒，对得上。'},checkBad:{en:'Check: 16 + 2 = 18 = 3 × 6. It fits.',zh:'检查：16 + 2 = 18 = 3 × 6，对得上。'}},
 {q:{en:'A book has 240 pages. Hong reads a quarter of it on day one and half of what is left on day two. How many pages are still unread?',zh:'一本书有 240 页。小红第一天读了全书的四分之一，第二天读了剩下的一半。还有多少页没读？'},ans:90,wrong:[120,60,180],
  steps:{en:'240 ÷ 4 = 60, so 180 are left; 180 ÷ 2 = 90 are read, so 90 are left.',zh:'240 ÷ 4 = 60，剩 180 页；180 ÷ 2 = 90，读了 90 页，还剩 90 页。'},slip:{en:'240 ÷ 2 = 120, so 120 are left.',zh:'240 ÷ 2 = 120，还剩 120 页。'},
  check:{en:'Check: 60 + 90 + 90 = 240. It fits.',zh:'检查：60 + 90 + 90 = 240，对得上。'},checkBad:{en:'Check: 120 + 120 = 240. It fits.',zh:'检查：120 + 120 = 240，对得上。'}},
 {q:{en:'An inlet fills a tank in 6 hours; a drain empties it in 9 hours. With both open, how many hours does the empty tank take to fill?',zh:'一个水池，只开进水管 6 小时能注满，只开排水管 9 小时能排空。两根管同时开，空水池几小时能注满？'},ans:18,wrong:[3,15,36],
  steps:{en:'Each hour it gains 1/6 − 1/9 = 1/18 of the tank, so 18 hours.',zh:'每小时净进 1/6 − 1/9 = 1/18 池，所以要 18 小时。'},slip:{en:'9 − 6 = 3, so 3 hours.',zh:'9 − 6 = 3，所以要 3 小时。'},
  check:{en:'Check: in 18 hours it takes in 3 tanks and drains 2. It fits.',zh:'检查：18 小时进 3 池、排 2 池，正好 1 池，对得上。'},checkBad:{en:'Check: 3 hours is less than 6, so it is fast. It fits.',zh:'检查：3 小时比 6 小时快，对得上。'}},
 {q:{en:'You write the whole numbers from 1 to 100. How many times do you write the digit 7?',zh:'把 1 到 100 的整数依次写下来，数字 7 一共写了几次？'},ans:20,wrong:[19,10,11],
  steps:{en:'7 in the ones place: 10 times; in the tens place: 10 times; 20 in all.',zh:'个位上的 7 有 10 个，十位上的 7 有 10 个，一共 20 个。'},slip:{en:'10 in the ones place, 10 in the tens place, minus 77 counted twice: 19.',zh:'个位 10 个，十位 10 个，77 算重了一次，一共 19 个。'},
  check:{en:'Check: 70 to 79 has 10 numbers. It fits.',zh:'检查：70 到 79 正好 10 个数，对得上。'},checkBad:{en:'Check: 70 to 79 has 10 numbers. It fits.',zh:'检查：70 到 79 正好 10 个数，对得上。'}}];
const KINDN=[{en:'easy',zh:'简单题'},{en:'medium',zh:'中等题'},{en:'hard',zh:'难题'},{en:'beyond the model',zh:'超出能力的题'}];
/* one name per way of writing, the same in the cards, the chart's labels and the challenge (the owner's review of 2026-10-09: the chart's
   short label “候选数” could not be read at a glance, so the short labels are gone) */
const WAYS=[{en:'Answer only',zh:'直接写答案'},{en:'Step by step',zh:'一步步算'},{en:'Solve, then check',zh:'算完再检查'},{en:'List candidate numbers',zh:'写好几个候选数字'}];
const wayN=i=>L(WAYS[i]);
/* the final number an answer writes, and the text of the answer (示例) */
function finalOf(x,kind){const p=PROB[kind]; if(x.s===3) return x.okS?p.ans:(x.okL?p.wrong[1]:p.wrong[2]); return x.okS?p.ans:(x.s===0?p.wrong[1]:p.wrong[0]);}
function candsOf(x,kind){const p=PROB[kind], fin=finalOf(x,kind); const set=x.okL?[p.wrong[2],p.ans,p.wrong[1]]:[p.wrong[0],p.wrong[2],p.wrong[1]];
  if(!set.includes(fin)) set[2]=fin; return set;}
function answerHTML(x,kind){
  const p=PROB[kind], fin=finalOf(x,kind), hl=n=>`<b class="rlvr-fin">${n}</b>`;
  const last=x.f?`<span class="mono">#### ${hl(fin)}</span>`:tx(`So the answer is ${hl(fin)}.`,`所以答案是 ${hl(fin)}。`);
  let body='';
  if(x.s===1||x.s===2) body=esc(L(x.okS?p.steps:p.slip))+' ';
  if(x.s===2) body+=esc(L(x.okS?p.check:p.checkBad))+' ';
  if(x.s===3){const c=candsOf(x,kind); body=tx(`It could be ${c[0]}, ${c[1]} or ${c[2]}. `,`可能是 ${c[0]}、${c[1]} 或 ${c[2]}。`);}
  return body+last;
}
function verdictTxt(x,kind,chk){
  const p=PROB[kind], fin=finalOf(x,kind);
  if(chk==='loose'){
    if(x.okL&&x.s===3&&!x.okS) return tx(`One of the numbers in it is ${p.ans}, the reference answer, so 1 point.`,`回答里有一个数是 ${p.ans}，等于标准答案，得 1 分。`);
    return x.okL?tx(`A number in it is ${p.ans}, the reference answer, so 1 point.`,`回答里有一个数是 ${p.ans}，等于标准答案，得 1 分。`):tx(`No number in it is ${p.ans}, the reference answer, so 0 points.`,`回答里没有一个数等于标准答案 ${p.ans}，得 0 分。`);}
  if(!x.f) return tx(`There is no “####” line, so the checker finds no final answer and gives 0 points.`,`没有写 “####” 这一行，判分程序找不到最后的答案，得 0 分。`);
  return x.okS?tx(`The final answer is ${fin} and the reference answer is ${p.ans}, so 1 point.`,`最后的答案是 ${fin}，标准答案是 ${p.ans}，得 1 分。`)
             :tx(`The final answer is ${fin} and the reference answer is ${p.ans}, so 0 points.`,`最后的答案是 ${fin}，标准答案是 ${p.ans}，得 0 分。`);
}

/* =====================================================================
   formatting
   ===================================================================== */
const f=(v,d)=>A.fmt.num(v,d);
const pc=(p,d=0)=>f(p*100,d)+'%';
const pcS=p=>{const v=p*100; return v>0&&v<0.5?'<1%':(v<100&&v>99.5?'>99%':f(v,0)+'%');};
const tok=n=>tx(`${f(n,0)} tokens`,`${f(n,0)} 个 token`);
const MINUS='−';
const sg=(v,d)=>{const s=f(Math.abs(v),d); return +s===0?f(0,d):(v<0?MINUS:'+')+s;};
const DEF=(text,tip)=>A.ui.def(text,tip);
const labLink=(id,text)=>A.shell.labLink?A.shell.labLink(id,text):esc(text||id);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const SRC={
 r1:'https://github.com/deepseek-ai/DeepSeek-R1/blob/0cf78561f1d51c84a21b2190626b21116d5c68bb/README.md',
 oiG:'https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/docs/algorithms/grpo.md',
 oiT:'https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/docs/tulu3.md',
 oiV:'https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/open_instruct/ground_truth_utils.py',
 orw:'https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/rewards.py',
 mv:'https://github.com/huggingface/Math-Verify/blob/ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b/README.md',
 gsm:'https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/README.md',
 tz:'https://github.com/Jiayi-Pan/TinyZero/blob/95df88f2dcb05f33bd18da546531b52d0954c18b/verl/utils/reward_score/countdown.py',
 lim:'https://github.com/LeapLabTHU/limit-of-RLVR/blob/79c348f4543330bb78b01a5332df09fea2700f70/README.md',
 trl:'https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/docs/source/grpo_trainer.md',
 dapo:'https://github.com/volcengine/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/docs/algo/dapo.md',
 srl:'https://github.com/hkust-nlp/simpleRL-reason/blob/cf1c7858bfc145a27aa733e75d930d34f0d318a6/README.md',
 r1p:'https://arxiv.org/abs/2501.12948',tulu:'https://arxiv.org/abs/2411.15124',dsm:'https://arxiv.org/abs/2402.03300',limp:'https://arxiv.org/abs/2504.13837'};

/* terms defined on hover (R2.8), plain words first */
const TIP={
 grpo:()=>tx('GRPO (group relative policy optimization): for each question the model writes a group of answers, here 8. An answer that scores above the group’s average is made more likely, one below it less likely. It needs no value model (in PPO, the model that estimates how much reward is still to come from a given point; also called the critic).','GRPO（group relative policy optimization）：每道题让模型写一组回答，这里是 8 个。分数比这一组平均分高的回答，以后更容易被写出来，比平均分低的就更不容易。它不需要价值模型（PPO 里估计“从这里往后平均还能拿多少奖励”的模型，也叫 critic）。'),
 toy:()=>tx('The simulated policy only chooses among 4 ways of writing, and whether to end with a “####” line: 5 parameters in all (4 logits for the ways, 1 for the format). A real model writes every token itself.','模拟的策略模型只在 4 种写法里选，另外再选要不要用 “####” 这一行收尾：一共 5 个参数（4 种写法各一个 logit，格式一个）。真实的模型每个 token 都自己写。'),
 vr:()=>tx('A verifiable reward is a score a program computes by checking the answer: 1 for right, 0 for wrong. No reward model is involved.','可验证的奖励：由程序检查答案得出的分数，对得 1 分，错得 0 分，不用奖励模型。'),
 ref:()=>tx('The reference model is a frozen copy of the model at the start of training (here the SFT model). It never changes and is used only to compute the KL penalty.','参考模型：训练开始时模型的一份冻结副本（这里就是 SFT 之后的模型），不再更新，只用来算 KL 惩罚。'),
 kl:()=>tx('The KL penalty measures how far the policy has moved from the reference model; times β = 0.05 it is added to the loss, which keeps the policy from running too far. GRPO estimates it per answer with the k3 formula in “Show the math”.','KL 惩罚衡量策略模型离参考模型有多远，乘上 β = 0.05 加进 loss，免得模型跑得太远。GRPO 按“展开公式”里的 k3 公式逐个回答估计它。'),
 rh:()=>tx('Reward hacking: the model raises its reward through a gap in how the reward is computed, instead of doing the task better.','reward hacking：模型钻奖励计算方式的空子把分数刷高，而不是真的把任务做得更好。'),
 policy:()=>tx('The policy is the model being trained, the one that writes the answers.','策略模型：正在训练、负责生成回答的模型。'),
 adv:()=>tx('The advantage says how much better an answer did than the other answers to the same problem. It is the answer’s reward minus the group’s average reward, divided by the standard deviation of the group’s rewards:','优势说的是一个回答比同一道题的其他回答好多少。它等于这个回答的奖励减去这一组的平均奖励，再除以这一组奖励的标准差：')+`<div class="fx">${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r)}')}</div>`+tx('A positive advantage makes the answer more likely, a negative one less likely.','优势为正，这个回答以后更容易被写出来；优势为负，就更不容易。'),
 passk:()=>tx('pass@k: the share of problems the model solves at least once in k tries. pass@1 is the share it gets right in one try.','pass@k：给 k 次机会，至少答对一次的题占多少。pass@1 就是答一次就对的比例。'),
 dyn:()=>tx('Skipping problems with no signal (verl’s dynamic sampling in DAPO): when all 8 answers to a problem score the same, every advantage is 0, so the group is dropped and another problem is drawn until the step has 8 problems with both right and wrong answers. The cost: more answers to generate.','跳过没有信号的题（verl 里 DAPO 的 dynamic sampling）：一道题的 8 个回答分数都一样时，每个优势都是 0，这一组就扔掉，换一道题重新生成，直到这一步凑满 8 道有对有错的题。代价是要多生成一些回答。'),
 fmt:()=>tx('Format reward: an answer that puts its final number after “####” but gets it wrong scores 0.1 instead of 0 (TinyZero gives 0.1 the same way). A right answer still scores 1.','格式奖励：答案写在 “####” 后面、但答错了，给 0.1 分而不是 0 分（TinyZero 也这样给 0.1 分）。答对仍然是 1 分。'),
 strict:()=>tx('Strict checker: the final answer must be on a “####” line, as in GSM8K, and equal the reference answer.','严格的判分程序：最后的答案要写在 “####” 这一行（GSM8K 的格式），而且要等于标准答案。'),
 loose:()=>tx('Loose checker: if any number in the answer equals the reference answer, it pays 1 point. This lab made it up as an example of a checker that is too loose.','宽松的判分程序：回答里只要有一个数等于标准答案，就给 1 分。这是这个 Lab 为了举例，自己设计的一个太松的判分程序。'),
 gsm:()=>tx('GSM8K: 8.5K grade-school math problems written by people, 7,473 for training and 1,319 for testing. Each solution ends with the final number after “####”.','GSM8K：8.5K 道人写的小学数学应用题，训练集 7,473 道、测试集 1,319 道。每道题的解答最后一行是 “####” 加上最终的数。'),
 base:()=>tx('A base model has only been pre-trained; it has not had SFT, so it does not yet answer in a fixed format.','base 模型：只做过预训练、还没做 SFT 的模型，还不会按固定格式回答。')
};

/* =====================================================================
   module state
   ===================================================================== */
const BASE={chk:'strict',fmt:0,dyn:0,start:'sft',set:'main',T:T};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let tShow=0;                          // the training step the stage shows
let grpI=0, sel=0;                    // which group of the step, which answer is selected
let ex2=null, ex2Tried=new Set(), lv5='strict', lv5T=new Set(['strict']), lv6=0, lv6T=new Set([0]);
let chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=3, drillOpen=false;
const RUNSEEN=new Set(), GRPSEEN=new Set();
const PL={on:false,speed:'f',t:0,raf:0,last:0};     // the training run (whole run: default fast, about 10 s)
const GP={on:false,speed:'m',n:G,acc:0,raf:0,last:0,key:''};   // one group of 8 (mechanism replay: default medium)
const RUN_SPD={s:2,m:3,f:6};                          // steps per second
const GRP_SPD={s:1.2,m:0.6,f:0.25};                   // seconds per answer
const shownPrev={};
let lastHash='';

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'D',en:'Answer only grows, and answers get shorter',zh:'直接写答案会变多，回答变短'},
            {v:'S',en:'Step by step grows, and answers get somewhat longer',zh:'一步步算会变多，回答变长一些'},
            {v:'C',en:'Solve, then check grows, and answers get much longer',zh:'算完再检查会变多，回答变长很多'},
            {v:'N',en:'The shares hardly change, and neither does the length',zh:'比例基本不变，长度也差不多'}];
const PRED_RIGHT='C';
const isP=id=>mode==='guided'&&STEPS[step].id===id;
const R=()=>run(S);
const atEnd=()=>tShow>=S.T;
const runOn=()=>mode!=='guided'||step>=3;               // the training chart shows from part 4 on, and in the other modes
function mainRuns(){const s=run({...BASE}), l=run({...BASE,chk:'loose'}); return {s,l,s0:s[0],sT:s[T],lT:l[T]};}
const peakOf=r=>{let k=0; r.forEach((x,i)=>{if(x.p1>r[k].p1) k=i;}); return k;};

const STEPS=[
 {id:'mission',title:{en:'Your task',zh:'你的任务'},lead:()=>tx('A program checks only the final answer. Which way of writing will training favour?','程序只检查最后的答案。训练会偏向哪种写法？'),
  enter:()=>({...BASE}),
  body:()=>{const m=mainRuns(), w=SETS.main, avg=i=>QS[i].reduce((a,q,k)=>a+w[k]*q,0);
   const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(tx(o.en,o.zh))}</button>`).join('')}</div>`;
   return tx(`<p>Math problems have reference answers, so a program can check the model’s answer: 1 point for right, 0 for wrong. This is a ${DEF('verifiable reward',TIP.vr())}. Reinforcement learning with this reward alone is called RLVR (reinforcement learning with verifiable rewards). The checker then takes the place of the reward model.</p>`+
     `<p>Here the model is ${DEF('a simulated policy',TIP.toy())} that only chooses among 4 ways of writing. On this set of problems, answer only gets ${pc(avg(0))} right, step by step ${pc(avg(1))}, and solve, then check ${pc(avg(2))}. Before training, ${pc(P0[0])} of its answers are answer only, ${pc(P0[1])} step by step and ${pc(P0[2])} solve, then check. The other ${pc(P0[3])} list several candidate numbers.</p>`+
     `<p>Now it is trained for ${T} steps with ${DEF('GRPO',TIP.grpo())} (the lab ${labLink('grpo')} covers it), scored only on whether the final answer is right. After training, which way of writing is more common, and do answers get longer or shorter?</p>`,
     `<p>数学题有标准答案，程序就能检查模型的答案：答对给 1 分，答错给 0 分。这叫${DEF('可验证的奖励',TIP.vr())}。只用这种奖励做强化学习，就叫 RLVR（reinforcement learning with verifiable rewards，基于可验证奖励的强化学习）。这时判分程序代替了奖励模型。</p>`+
     `<p>这里的模型是一个${DEF('模拟的策略模型',TIP.toy())}，只在 4 种写法里选。在这批题上，直接写答案平均答对 ${pc(avg(0))}，一步步算答对 ${pc(avg(1))}，算完再检查答对 ${pc(avg(2))}。训练前，模型 ${pc(P0[0])} 的回答直接写答案，${pc(P0[1])} 一步步算，${pc(P0[2])} 算完再检查。剩下 ${pc(P0[3])} 的回答写了好几个候选数字。</p>`+
     `<p>现在用 ${DEF('GRPO',TIP.grpo())} 训练 ${T} 步（${labLink('grpo')} 这个 Lab 专门讲它），只按最后的答案对不对给分。训练以后，哪种写法会变多？回答会变长还是变短？</p>`)+ch+
     (pred?`<p class="small muted">${tx('Saved. You can still change it until the training run in part 4.','已记下，第 4 部分训练之前还可以改。')}</p>`:'');}},
 {id:'checker',title:{en:'The checker',zh:'判分程序'},lead:()=>tx('What exactly does the program check? Two checkers, three answers.','程序到底检查什么？两种判分程序，三个回答。'),
  enter:()=>{if(ex2==null){ex2=EX2[0].v; ex2Tried.add(ex2);} return {...BASE};},   // codex-b9v3 #1: the stage shows an answer from the start
  body:()=>tx(`<p>The checker compares a number in the answer with the reference answer and pays 1 point if they are equal. The two checkers differ in which number they take.</p><p>A ${DEF('strict checker',TIP.strict())} takes only the number on the “####” line, as in ${DEF('GSM8K',TIP.gsm())}. A ${DEF('loose checker',TIP.loose())} looks at every number in the answer and pays 1 point if any of them equals the reference answer.</p><p>Pick an answer and see which numbers each checker reads, and what each pays.</p>`,
              `<p>判分程序拿回答里的数和标准答案比，相等就给 1 分。两种判分程序的区别，在于拿哪个数来比。</p><p>${DEF('严格的判分程序',TIP.strict())}只看 “####” 这一行写的数（${DEF('GSM8K',TIP.gsm())} 的格式）。${DEF('宽松的判分程序',TIP.loose())}看回答里的每一个数，只要有一个等于标准答案，就给 1 分。</p><p>选一个回答，看两种判分程序各读哪个数、各给几分。</p>`)+ex2Ctl(),
  expect:()=>ex2Tried.has(3),
  obs:()=>obsChecker()},
 {id:'group',title:{en:'One group of 8',zh:'一组 8 个回答'},lead:()=>tx('Right and wrong inside one group: who gets pushed up?','同一组里有对有错：谁被加强？'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>At every step, ${labLink('grpo','GRPO')} takes ${B} problems and has the model generate ${G} answers (rollouts) to each. The ${G} answers to one problem form a group. The checker scores each answer, and that score is its reward r.</p><p>An answer’s ${DEF('advantage',TIP.adv())} says how much better it did than the average answer in its group.</p><div class="fx">${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}')}</div><p>${A.ui.math('\\text{mean}(r)')} and ${A.ui.math('\\text{std}(r)')} are the average and the standard deviation of the group’s rewards. The 0.0001 in the denominator avoids dividing by 0 when every reward in the group is the same.</p><p>A positive advantage makes that answer’s way of writing more likely next time; a negative one makes it less likely. If all ${G} answers get the same reward, every advantage is 0 and the problem teaches nothing at this step.</p>`,
              `<p>${labLink('grpo','GRPO')} 每一步拿 ${B} 道题，每道题让模型生成 ${G} 个回答（rollout），同一道题的 ${G} 个回答就是一组。判分程序给每个回答打分，这个分数就是奖励 r。</p><p>一个回答的${DEF('优势',TIP.adv())}，说的是它比同一组回答的平均水平好多少。</p><div class="fx">${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}')}</div><p>${A.ui.math('\\text{mean}(r)')} 和 ${A.ui.math('\\text{std}(r)')} 是这一组奖励的平均值和标准差。分母里的 0.0001，是为了全组奖励都一样时不会除以 0。</p><p>优势为正，这个回答的写法下次更容易被选中。优势为负，就更不容易被选中。如果 ${G} 个回答的奖励都一样，每个优势都是 0，这道题在这一步什么也教不了。</p>`)+
   `<div class="stepnav rlvr-gnav"><button type="button" class="btn ghost sm" data-act="grp-next" id="rlvr-ic-grp">${tx('Next problem →','换一道题 →')}</button></div>`,
  obs:()=>obsGroup(),always:true},
 {id:'train',title:{en:'Train 60 steps',zh:'训练 60 步'},lead:()=>tx('The whole run: which way of writing wins?','整次训练：哪种写法胜出？'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>Now the model is trained for ${T} steps. Each step takes ${B} problems with ${G} answers each, scores them with the strict checker, and uses these ${B*G} answers for one update of the model. The ${DEF('KL penalty',TIP.kl())} with β = ${BETA} keeps it near the ${DEF('reference model',TIP.ref())}. The chart plays the run.</p>`,
              `<p>现在训练 ${T} 步。每一步做 ${B} 道题，每道题 ${G} 个回答，用严格的判分程序打分，再用这 ${B*G} 个回答把模型更新一次。β = ${BETA} 的 ${DEF('KL 惩罚',TIP.kl())}让模型不离${DEF('参考模型',TIP.ref())}太远。图里会播放这次训练。</p>`),
  expect:()=>RUNSEEN.has(cfgKey(BASE))&&atEnd(),showMe:true,
  obs:()=>obsTrain()},
 {id:'loose',title:{en:'A loose checker',zh:'宽松的判分程序'},lead:()=>tx('Swap in the loose checker and train again.','换成宽松的判分程序，再训练一次。'),
  enter:()=>({...BASE,chk:lv5}),
  body:()=>tx(`<p>Same model, same problems; only the checker changes. Train once with each checker.</p>`,`<p>模型和题都不变，只换判分程序。两种各训练一次。</p>`)+lv5Ctl(),
  expect:()=>lv5T.has('loose')&&RUNSEEN.has(cfgKey({...BASE,chk:'loose'})),
  obs:()=>obsLoose()},
 {id:'format',title:{en:'Format reward',zh:'格式奖励'},lead:()=>tx('Start from a base model that rarely writes “####”.','从很少写 “####” 的 base 模型开始。'),
  enter:()=>({...BASE,start:'base',fmt:lv6}),
  body:()=>tx(`<p>This part starts from a ${DEF('base model',TIP.base())}, the way DeepSeek‑R1‑Zero started RL with no SFT first. Only ${pc(sig(PHI0.base))} of its answers end with a “####” line, so the strict checker gives almost everything 0.</p><p>A ${DEF('format reward',TIP.fmt())} gives a wrong answer on a “####” line 0.1 point instead of 0. Train with and without it.</p>`,
              `<p>这一部分从 ${DEF('base 模型',TIP.base())}开始，就像 DeepSeek‑R1‑Zero 不做 SFT、直接开始 RL。它只有 ${pc(sig(PHI0.base))} 的回答用 “####” 这一行收尾，所以严格判分几乎全给 0 分。</p><p>${DEF('格式奖励',TIP.fmt())}给写在 “####” 后面、但答错的回答 0.1 分，而不是 0 分。加和不加各训练一次。</p>`)+lv6Ctl(),
  expect:()=>lv6T.has(1)&&RUNSEEN.has(cfgKey({...BASE,start:'base',fmt:1})),
  obs:()=>obsFormat()},
 {id:'passk',title:{en:'pass@1 and pass@k',zh:'pass@1 和 pass@k'},lead:()=>tx('More right in one try. But more problems solved at all?','一次答对的多了。能解出的题也多了吗？'),
  enter:()=>({...BASE,start:'sft',fmt:0,chk:'strict'}),
  body:()=>tx(`<p>${DEF('pass@k',TIP.passk())} is the share of problems the model solves at least once when it gets k tries (the lab ${labLink('pass-k')} explains it). The chart compares the model before training with the one after ${T} steps of training with the strict checker.</p>`,
              `<p>${DEF('pass@k',TIP.passk())} 是给 k 次机会时，至少答对一次的题占多少（${labLink('pass-k')} 这个 Lab 专门讲它）。图里比较训练前的模型，和用严格的判分程序训练 ${T} 步之后的模型。</p>`),
  obs:()=>obsPassK(),always:true},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:()=>tx('Three things to take away, three common misconceptions.','三条要点，三个常见误解。'),
  enter:()=>({...BASE}),
  body:()=>remember(),always:true}
];
const WHY={mission:{en:'Pick an answer to continue.',zh:'选一个答案才能继续。'},checker:{en:'Look at the candidate-numbers answer to continue.',zh:'看过“写好几个候选数字”这个回答才能继续。'},
  train:{en:'Let the run finish to continue, or use Show me.',zh:'等训练播完才能继续，也可以点“演示给我看”。'},loose:{en:'Train with the loose checker to continue.',zh:'用宽松的判分程序训练一次才能继续。'},
  format:{en:'Train with the format reward to continue.',zh:'加上格式奖励训练一次才能继续。'}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function isNextDisabled(){const s=STEPS[step]; if(s.id==='mission') return !pred; return !!s.expect&&!observed[step];}

/* part 2: three example answers to problem 1 */
const EX2=[{v:1,x:{s:1,f:1,okS:true,okL:true},t:{en:'Step by step, right',zh:'一步步算，答对'}},{v:0,x:{s:0,f:1,okS:false,okL:false},t:{en:'Answer only, wrong',zh:'直接写答案，答错'}},{v:3,x:{s:3,f:1,okS:false,okL:true},t:{en:'Candidate numbers',zh:'写好几个候选数字'}}];
const ptsTxt=r=>tx(r===1?'1 point':`${f(r,r%1?1:0)} points`,`${f(r,r%1?1:0)} 分`);
const exScore=(x,chk)=>chk==='loose'?(x.okL?1:0):(x.f&&x.okS?1:0);
function ex2Ctl(){return `<div class="rlvr-lv" id="rlvr-ic-ex">`+A.ui.levers({aria:tx('Example answers, pick one','示例回答，选一个'),value:ex2==null?'':String(ex2),
  options:EX2.map(o=>({v:String(o.v),title:L(o.t),desc:'',result:ex2Tried.has(o.v)?tx(`strict ${exScore(o.x,'strict')}, loose ${exScore(o.x,'loose')}`,`严格 ${exScore(o.x,'strict')} 分，宽松 ${exScore(o.x,'loose')} 分`):''}))})+`</div>`;}
/* part 2 (codex-b9v3 #1): the stage reads the picked answer with both checkers. Both work on the answer's own text, as the code in
   Connect does: the strict checker takes the number on the “####” line, the loose checker every number in the answer. */
function respLines(x,kind){
  const p=PROB[kind], fin=finalOf(x,kind); let body='';
  if(x.s===1||x.s===2) body=L(x.okS?p.steps:p.slip);
  if(x.s===2) body+=tx(' ','')+L(x.okS?p.check:p.checkBad);
  if(x.s===3){const c=candsOf(x,kind); body=tx(`It could be ${c[0]}, ${c[1]} or ${c[2]}.`,`可能是 ${c[0]}、${c[1]} 或 ${c[2]}。`);}
  const last=x.f?`#### ${fin}`:tx(`So the answer is ${fin}.`,`所以答案是 ${fin}。`);
  return body?[body,last]:[last];
}
const NUM_RE=/\d+(?:\.\d+)?/g;
function readsOf(x,kind){
  const p=PROB[kind], lines=respLines(x,kind), m=/^#### (\d+)$/.exec(lines[lines.length-1]), sRead=m?+m[1]:null;
  const all=[]; lines.forEach(l=>(l.match(NUM_RE)||[]).forEach(n=>all.push(+n)));
  const fin=finalOf(x,kind), lPay=all.includes(p.ans)?1:0;
  return {lines,ref:p.ans,fin,strict:{read:sRead==null?[]:[sRead],pay:sRead===p.ans?1:0},loose:{read:all,pay:lPay},gamed:lPay===1&&fin!==p.ans};
}
const ex2Opt=()=>EX2.find(e=>e.v===ex2)||EX2[0];
const chkNow=()=>readsOf(ex2Opt().x,0);
function obsChecker(){
  if(ex2==null) return '';
  const c=chkNow(), ref=c.ref;
  if(ex2===3) return tx(`<p>The strict checker reads only the “####” line, where the ${c.fin} is wrong, so it pays 0 points.</p><p>The loose checker finds ${ref} among the candidate numbers and pays 1 point, as if the problem were solved. A wrong answer gets paid: that is the gap the loose checker leaves.</p>`,
      `<p>严格的判分程序只读 “####” 这一行，那里写的 ${c.fin} 是错的，所以给 0 分。</p><p>宽松的判分程序在几个候选数字里找到了 ${ref}，就当作做对了，给 1 分。答错了也能拿分，这就是宽松的判分程序留下的空子。</p>`)+
    tx(`<p class="small muted">A real example of a loose check: open-instruct’s GSM8K verifier takes the last number in the output and compares it with the label (${cite(SRC.oiV,'<code>ground_truth_utils.py</code>')}).</p>`,`<p class="small muted">真实系统里也有偏松的检查：open-instruct 的 GSM8K 判分只取输出里的最后一个数，和标准答案比（${cite(SRC.oiV,'<code>ground_truth_utils.py</code>')}）。</p>`);
  if(c.strict.pay) return tx(`<p>The final answer, ${c.fin}, is right, so the strict checker pays 1 point.</p><p>The loose checker reads every number in the working, finds ${ref} among them and also pays 1 point.</p>`,
      `<p>最后的答案 ${c.fin} 是对的，严格的判分程序给 1 分。</p><p>宽松的判分程序读了算式里的每一个数，其中有 ${ref}，也给 1 分。</p>`);
  return tx(`<p>This answer writes a single number, ${c.fin}, and it is wrong. Both checkers pay 0 points.</p>`,`<p>这个回答只写了一个数，${c.fin} 是错的。两种判分程序都给 0 分。</p>`);
}
/* while a run plays, the group panel keeps the group of the step where the playback started (R7.9, R7.11); its heading names that step */
const grpStep=()=>Math.min(PL.on&&runOn()&&PL.from!=null?PL.from:tShow,S.T);
function groupNow(){const r=R(), st=r[Math.min(grpStep(),r.length-1)]; const gs=st.groups; return gs[Math.min(grpI,gs.length-1)];}
function obsGroup(){
  const g=groupNow(), nR=g.resp.filter(x=>x.r===1).length;
  if(g.flat) return tx(`<p>In this group all 8 answers score ${f(g.resp[0].r,0)}. The average is ${f(g.m,0)} and every advantage is 0, so this problem changes nothing at this step.</p><p>With 8 answers, this happens often on problems that are very easy or very hard for the model.</p>`,
    `<p>这一组 8 个回答都是 ${f(g.resp[0].r,0)} 分。平均分也是 ${f(g.m,0)}，每个优势都是 0，这道题在这一步什么也改变不了。</p><p>只有 8 个回答时，对模型来说太简单或太难的题，经常会这样。</p>`);
  const mx=Math.max(...g.resp.map(x=>x.r)), mn=Math.min(...g.resp.map(x=>x.r));
  const right=g.resp.filter(x=>x.r===mx), wr=g.resp.find(x=>x.r===mn), aR=right[0].a, aW=wr.a;
  if(mx!==1) return tx(`<p>No answer in this group is right. The ones that score ${f(mx,1)} (they only wrote the “####” line) get an advantage of ${sg(aR,2)}, the others ${sg(aW,2)}, so the format is what gets pushed up here.</p>`,
    `<p>这一组没有一个回答答对。得 ${f(mx,1)} 分的回答（只是写了 “####” 那一行）优势是 ${sg(aR,2)}，其余是 ${sg(aW,2)}，所以这一组加强的是格式。</p>`);
  const byWay=[0,1,2,3].map(i=>right.filter(x=>x.s===i).length), top=byWay.indexOf(Math.max(...byWay));
  if(nR===1) return tx(`<p>1 of the 8 answers is right. It gets an advantage of ${sg(aR,2)} and each wrong one ${sg(aW,2)}, so the right answer, written ${wayN(top).toLowerCase()}, is pushed up hard.</p><p>Over many groups, a way of writing that is right more often gets positive advantages more often, so it grows.</p>`,
    `<p>8 个回答里只有 1 个答对。它的优势是 ${sg(aR,2)}，每个答错的是 ${sg(aW,2)}，所以这个用“${wayN(top)}”写的回答被大力加强。</p><p>看很多组就会发现：哪种写法答对得更多，它拿到正优势的次数就更多，于是越来越常用。</p>`);
  return tx(`<p>${nR} of the 8 answers are right. Each right answer gets an advantage of ${sg(aR,2)} and each wrong one ${sg(aW,2)}, so the right answers are pushed up. Most of the right ones here are ${wayN(top).toLowerCase()}.</p><p>Over many groups, a way of writing that is right more often gets positive advantages more often, so it grows.</p>`,
    `<p>8 个回答里有 ${nR} 个答对。每个答对的回答优势是 ${sg(aR,2)}，每个答错的是 ${sg(aW,2)}，所以答对的回答被加强。这一组答对的回答里，最多的是“${wayN(top)}”。</p><p>看很多组就会发现：哪种写法答对得更多，它拿到正优势的次数就更多，于是越来越常用。</p>`);
}
function obsTrain(){
  const m=mainRuns(), s0=m.s0, sT=m.sT;
  const res=tx(`<p>After ${T} steps, the answers written “solve, then check” went from ${pc(s0.pi[2])} to ${pc(sT.pi[2])} of all answers. The average answer grew from about ${f(s0.len,0)} to about ${f(sT.len,0)} tokens.</p>`,
    `<p>训练 ${T} 步后，用“算完再检查”写的回答从 ${pc(s0.pi[2])} 涨到了 ${pc(sT.pi[2])}。平均每个回答从约 ${f(s0.len,0)} 个 token 变成约 ${f(sT.len,0)} 个。</p>`);
  const p=PRED.find(o=>o.v===pred), r=PRED.find(o=>o.v===PRED_RIGHT);
  const vd=pred===PRED_RIGHT?tx(`<p class="okline">Your prediction “${esc(p.en)}” is right.</p>`,`<p class="okline">你的预测“${esc(p.zh)}”是对的。</p>`)
    :(p?tx(`<p class="badline">You predicted “${esc(p.en)}”; the right answer is “${esc(r.en)}”.</p>`,`<p class="badline">你预测的是“${esc(p.zh)}”，正确答案是“${esc(r.zh)}”。</p>`):'');
  return res+vd+tx(`<p>Why: each step only compares answers inside a group. “Solve, then check” is right most often on every kind of problem, so it wins the most positive advantages. Answers get longer because that way of writing is long: the length comes with getting it right. DeepSeek reported the same kind of thing for R1‑Zero: “self-verification, reflection, and generating long CoTs” emerged with RL alone (${cite(SRC.r1,'DeepSeek‑R1 README')}).</p>`,
    `<p>原因：每一步只在组内比较谁对谁错。“算完再检查”在每一类题上都最常答对，拿到的正优势最多。回答变长，是因为这种写法本来就长：长度是跟着做对一起来的。DeepSeek 报告 R1‑Zero 只靠 RL 就出现了类似的行为：自我验证、反思、很长的思维链（${cite(SRC.r1,'DeepSeek‑R1 README')}）。</p>`);
}
function lv5Ctl(){const m=mainRuns();
  return `<div class="rlvr-lv" id="rlvr-ic-chk">`+A.ui.levers({aria:tx('Checker, pick one','判分程序，选一个'),value:lv5,options:[
   {v:'strict',title:tx('Strict checker','严格的判分程序'),desc:tx('The final answer on the “####” line must equal the reference answer.','“####” 这一行的答案要等于标准答案。'),result:lv5T.has('strict')?tx(`right in one try: ${pc(m.sT.p1)}`,`一次答对 ${pc(m.sT.p1)}`):''},
   {v:'loose',title:tx('Loose checker','宽松的判分程序'),desc:tx('Any number in the answer may equal the reference answer.','回答里任何一个数等于标准答案就行。'),result:lv5T.has('loose')&&RUNSEEN.has(cfgKey({...BASE,chk:'loose'}))?tx(`right in one try: ${pc(m.lT.p1)}`,`一次答对 ${pc(m.lT.p1)}`):''}]})+`</div>`;}
function obsLoose(){
  const m=mainRuns(), l=run({...BASE,chk:'loose'}), k=peakOf(l), pay=looseAvgPay();
  if(S.chk!=='loose') return tx(`<p>With the strict checker the candidate numbers fall to ${pcS(m.sT.pi[3])} of the answers. Pick the loose checker to compare.</p>`,`<p>用严格的判分程序，写好几个候选数字的回答降到 ${pcS(m.sT.pi[3])}。选宽松的判分程序来对比。</p>`);
  return tx(`<p>With the loose checker, listing candidate numbers grew from ${pc(P0[3])} to ${pc(m.lT.pi[3])} of the answers. The checker’s average pay per answer kept rising, to ${pc(m.lT.rew)} of a full point. But the share right under the strict checker peaked at step ${k} and fell to ${pc(m.lT.p1)} (red in the chart).</p><p>Why: under this checker, listing candidate numbers is paid more often than solving the problem. On this set, an answer that lists candidate numbers is paid ${pc(pay.cand)} of the time on average, one written “solve, then check” only ${pc(pay.check)}. The answers also got shorter, because this way of writing is short.</p><p>This is ${DEF('reward hacking',TIP.rh())}. The checker is the reward model now. Wherever it has a gap, the model learns to use it. In the ${labLink('reward-model')} lab, the answer picked by best-of-n (generate n answers, keep the one the reward model scores highest) kept scoring higher while its true quality fell. Real checkers guard against such gaps too. When a model asked to solve an inequality simply returns the inequality from the problem, Math-Verify, the checker open-r1 uses, does not pay it (${cite(SRC.mv,'Math-Verify README')}).</p>`,
    `<p>用宽松的判分程序，写好几个候选数字的回答从 ${pc(P0[3])} 涨到了 ${pc(m.lT.pi[3])}。判分程序平均每个回答给的分一直在涨，最后是满分的 ${pc(m.lT.rew)}。可严格判分下的答对比例在第 ${k} 步最高，之后一路下跌，最后只有 ${pc(m.lT.p1)}（图里红色的部分）。</p><p>原因：在这个判分程序下，写候选数字比认真解题更常拿分。在这批题上，写候选数字的回答平均有 ${pc(pay.cand)} 拿到分，用“算完再检查”写的只有 ${pc(pay.check)}。回答也跟着变短了，因为这种写法本来就短。</p><p>这就是 ${DEF('reward hacking',TIP.rh())}。判分程序现在就是奖励模型。它哪里有漏洞，模型就会去利用哪里。在${labLink('reward-model','“奖励模型”')}这个 Lab 里也是这样，用 best-of-n（生成 n 个回答，留下奖励模型打分最高的那个）挑出的回答，分数一直在涨，真实质量却在降。真实的判分库也得防模型钻空子。比如题目要模型解一个不等式，模型却只把题目里的不等式原样抄回来，open-r1 用的判分库 Math-Verify 就不给分（${cite(SRC.mv,'Math-Verify README')}）。</p>`);
}
function lv6Ctl(){const r0=run({...BASE,start:'base'}), r1=run({...BASE,start:'base',fmt:1}), K=10;
  return `<div class="rlvr-lv" id="rlvr-ic-fmt">`+A.ui.levers({aria:tx('Format reward, pick one','格式奖励，选一个'),value:String(lv6),options:[
   {v:'0',title:tx('No format reward','不加格式奖励'),desc:tx('Only right answers on a “####” line score: 1 point.','只有写在 “####” 后面又答对的回答得 1 分。'),result:lv6T.has(0)&&RUNSEEN.has(cfgKey({...BASE,start:'base'}))?tx(`right in one try at step ${K}: ${pc(r0[K].p1)}`,`第 ${K} 步一次答对 ${pc(r0[K].p1)}`):''},
   {v:'1',title:tx('With format reward','加格式奖励'),desc:tx('A wrong answer on a “####” line also scores 0.1.','写在 “####” 后面、但答错，也得 0.1 分。'),result:lv6T.has(1)&&RUNSEEN.has(cfgKey({...BASE,start:'base',fmt:1}))?tx(`right in one try at step ${K}: ${pc(r1[K].p1)}`,`第 ${K} 步一次答对 ${pc(r1[K].p1)}`):''}]})+`</div>`;}
function obsFormat(){
  const r0=run({...BASE,start:'base'}), r1=run({...BASE,start:'base',fmt:1}), K=10, r=S.fmt?r1:r0;
  const head=S.fmt?tx(`<p>With the format reward, after ${K} steps ${pc(r1[K].pf)} of the answers end with “####”, and ${pc(r1[K].p1)} are right in one try. Without it: ${pc(r0[K].pf)} and ${pc(r0[K].p1)}.</p>`,`<p>加了格式奖励，第 ${K} 步时 ${pc(r1[K].pf)} 的回答用 “####” 收尾，一次答对的比例是 ${pc(r1[K].p1)}。不加时分别是 ${pc(r0[K].pf)} 和 ${pc(r0[K].p1)}。</p>`)
    :tx(`<p>Without a format reward, after ${K} steps ${pc(r0[K].pf)} of the answers end with “####”, and ${pc(r0[K].p1)} are right in one try. Most groups are all 0 at first, so they teach nothing.</p>`,`<p>不加格式奖励，第 ${K} 步时只有 ${pc(r0[K].pf)} 的回答用 “####” 收尾，一次答对的比例是 ${pc(r0[K].p1)}。一开始大多数组全是 0 分，什么也教不了。</p>`);
  const same=pc(r0[T].p1)===pc(r1[T].p1);
  const end=same?tx(`By step ${T}, right in one try is ${pc(r1[T].p1)} with or without the format reward.`,`到第 ${T} 步，加不加格式奖励，一次答对都是 ${pc(r1[T].p1)}。`)
    :tx(`By step ${T}, right in one try is ${pc(r0[T].p1)} without the format reward and ${pc(r1[T].p1)} with it.`,`到第 ${T} 步，不加格式奖励一次答对 ${pc(r0[T].p1)}，加了是 ${pc(r1[T].p1)}。`);
  const afx=`<div class="fx">${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}')}</div>`;
  return head+tx(`<p>Why: a group with a few “####” answers and no right one now has different rewards, so it has a signal again. Each answer’s advantage is its reward minus the group’s mean, divided by the group’s standard deviation:</p>${afx}<p>Multiply every reward in a group by 10, and 0 and 0.1 become 0 and 1. The difference on top and the standard deviation below both grow 10 times, so the advantages stay the same. That is why, inside a group, 0.1 against 0 pushes as hard as 1 against 0 (open-instruct: “the scale of the rewards is not relevant”).</p><p>${end} The format reward teaches the format, not the solving. open-instruct also warns that a direct additive format reward can push up a long wrong answer just for its format (${cite(SRC.oiG,'open-instruct GRPO docs')}).</p>`,
    `<p>原因：一组里只要有几个回答写了 “####”，即使没有一个答对，奖励也有高有低，又有了信号。每个回答的优势，是它的奖励减去这一组的平均值，再除以这一组的标准差：</p>${afx}<p>把一组的奖励都乘以 10，0 和 0.1 就变成 0 和 1。分子里的差和分母里的标准差也都变成 10 倍，相除以后优势不变。所以在一组里，0.1 分对 0 分和 1 分对 0 分，推的力度一样大（open-instruct 的文档说“奖励的大小无关紧要”）。</p><p>${end}可见格式奖励教会的是格式，不是解题。open-instruct 还提醒，直接加格式分，一个格式对但答错的长回答也会被加强（${cite(SRC.oiG,'open-instruct 的 GRPO 文档')}）。</p>`);
}
function obsPassK(){
  const m=mainRuns(), w=SETS.main, b=passK(m.s0.pk,w), a=passK(m.sT.pk,w), n=KS.length-1;
  const same=pc(b[n])===pc(a[n]);
  return tx(`<p>After training, pass@1 rose from ${pc(b[0])} to ${pc(a[0])}. With ${KS[n]} tries, the model solves ${same?`${pc(a[n])} of the problems both before and after training`:`${pc(b[n])} of the problems before training and ${pc(a[n])} after`}.</p><p>Why: in this lab the model has only ${NW} ways of writing, and ${pc(w[3])} of the problems are beyond all of them. RL can only change how often each of the ${NW} ways is used, so those ${pc(w[3])} stay unsolved before and after training.</p><p>A study of real models found something similar. At small k the RL-trained models do better, but as k grows to the tens or hundreds, the base models it tested “consistently catch up with RL-trained models”, and eventually pass them (${cite(SRC.lim,'limit-of-RLVR README')}).</p>`,
    `<p>训练以后，pass@1 从 ${pc(b[0])} 涨到 ${pc(a[0])}。给 ${KS[n]} 次机会时，${same?`训练前后都能解出 ${pc(a[n])} 的题`:`训练前能解出 ${pc(b[n])} 的题，训练后能解出 ${pc(a[n])}`}。</p><p>原因：在这个 Lab 里，模型只有 ${NW} 种写法，而这批题里有 ${pc(w[3])} 哪种写法都做不对。RL 只能调整这 ${NW} 种写法各用多少，所以这 ${pc(w[3])} 的题训练前后都解不出。</p><p>一项对真实模型的研究也有类似的发现。k 小的时候，RL 训练过的模型更强。k 到几十、几百时，它测过的 base 模型都追了上来，最后还超过了 RL 训练过的模型（${cite(SRC.lim,'limit-of-RLVR README')}）。</p>`);
}
function remember(){
  const m=mainRuns(), w=SETS.main, b=passK(m.s0.pk,w), a=passK(m.sT.pk,w), n=KS.length-1;
  return tx(`<ol class="rlvr-rem"><li><b>A right or wrong score is enough when the answers to the same problem differ.</b> Each answer is compared with a baseline, and the ones above it are pushed up. GRPO uses the group’s average reward as the baseline; ${labLink('ppo','PPO')} estimates the baseline with its value model (Tülu 3’s RLVR used PPO). So when a group has both right and wrong answers, the right ones are pushed up, and an all-wrong group teaches nothing. As training goes on, the way of writing that is right most often takes over.</li>`+
    `<li><b>The checker is the reward model now; test it before training.</b> If the checker has a loophole, the model finds it: when listing candidate numbers earns a reward, the model soon learns to guess. Before you train, try such answers on your checker.</li>`+
    `<li><b>Watch pass@k, not only pass@1.</b> RL often only makes the model use the ways of writing it already had more often: pass@1 rises, while the problems it solves with many tries do not grow.</li></ol>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “RL taught the model new knowledge.” In this lab the model has only ${NW} ways of writing, and RL only changed how often each is used; the problems no way could solve stayed unsolved. A study of real models found something similar.</p><p>✗ “Longer answers are better.” The answers got longer because the way of writing that is right more often is long. Length is a side effect, and it costs generation time.</p><p>✗ “A verifiable reward cannot be gamed.” A checker that is too loose is soon gamed.</p></div>`,
    `<ol class="rlvr-rem"><li><b>只要同一道题的回答有对有错，只给对错就够了。</b>每个回答都和一个基准比，比基准好的就被加强。GRPO 拿同一组回答的平均奖励当基准，${labLink('ppo','PPO')} 用价值模型估计基准。Tülu 3 的 RLVR 用的就是 PPO。所以一组回答有对有错时，答对的回答被加强。全错的一组，什么也学不到。训练下去，最常答对的那种写法就成了主流。</li>`+
    `<li><b>判分程序现在就是奖励模型，训练前先检查它。</b>判分程序有漏洞，模型就会去钻。只要写一串候选数字也能拿分，模型很快就学会了猜。训练之前，先拿这类回答去试一试你的判分程序。</li>`+
    `<li><b>不只看 pass@1，也看 pass@k。</b>RL 常常只是让模型更多地用它本来就会的写法。所以 pass@1 涨了，给很多次机会时能解出的题却没有变多。</li></ol>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “RL 让模型学到了新知识。”在这个 Lab 里，模型只有 ${NW} 种写法，RL 只改变了它们各用多少。哪种写法都做不对的题，训练后还是做不对。一项对真实模型的研究，也有类似的发现。</p><p>✗ “回答越长越好。”回答变长，是因为更常答对的那种写法本来就长。长度是副产品，还要多花生成的时间。</p><p>✗ “可验证的奖励不会被钻空子。”判分程序太松，模型很快就会钻它的空子。</p></div>`)+A.shell.toScenarios();
}

/* =====================================================================
   scenarios, challenge, capability layer
   ===================================================================== */
const PRESETS=[
 {id:'strict',t:{en:'Initial setup: a strict checker',zh:'初始设置：严格的判分程序'},d:{en:'The final answer must be on the “####” line',zh:'最后的答案必须写在 “####” 这一行'},p:{...BASE}},
 {id:'loose',t:{en:'A checker that is too loose',zh:'判分程序太松'},d:{en:'Any number in the answer may match',zh:'回答里任何一个数对上就给分'},p:{...BASE,chk:'loose'}},
 {id:'zero',t:{en:'RL straight from a base model',zh:'不做 SFT，直接从 base 模型开始'},d:{en:'Like R1‑Zero, with a format reward',zh:'像 R1‑Zero 那样，再加格式奖励'},p:{...BASE,start:'base',fmt:1}},
 {id:'hard',t:{en:'Problems too hard: almost all wrong at first',zh:'题太难：一开始几乎全错'},d:{en:'Most groups of 8 score all 0',zh:'大多数组的 8 个回答全是 0 分'},p:{...BASE,set:'hard'}}];
const CFGK=['chk','fmt','dyn','start','set'];
const presetNow=()=>PRESETS.findIndex(p=>CFGK.every(k=>(p.p[k]||0)===(S[k]||0)));
function scenHTML(i){const p=PRESETS[i], r=run(p.p), e=r[r.length-1], s0=r[0];
  const flat=r.slice(0,10).reduce((s,x)=>s+x.flat,0)/r.slice(0,10).reduce((s,x)=>s+x.groups.length,0);
  if(p.id==='strict') return tx(`<p>In ${T} steps, right in one try rises from ${pc(s0.p1)} to ${pc(e.p1)}, and solve, then check becomes the habit (${pc(e.pi[2])} of the answers).</p>`,`<p>${T} 步里，一次答对的比例从 ${pc(s0.p1)} 涨到 ${pc(e.p1)}，“算完再检查”成了习惯（占 ${pc(e.pi[2])} 的回答）。</p>`);
  if(p.id==='loose') return tx(`<p>The candidate numbers take over (${pc(e.pi[3])}), while right in one try ends at ${pc(e.p1)}, below the ${pc(s0.p1)} it started from. Test a checker with answers that try to game it.</p>`,`<p>写候选数字的回答占了上风（${pc(e.pi[3])}），一次答对的比例最后只有 ${pc(e.p1)}，比开始时的 ${pc(s0.p1)} 还低。判分程序要先用想钻空子的回答试一试。</p>`);
  if(p.id==='zero') return tx(`<p>The base model starts at ${pc(s0.p1)} right in one try, because it rarely writes “####”. The format reward teaches the format within a few steps, and by step ${T} it is right ${pc(e.p1)} of the time. DeepSeek‑R1‑Zero trained the same way, with rule-based rewards and no SFT first.</p>`,`<p>base 模型一开始一次答对只有 ${pc(s0.p1)}，因为它很少写 “####”。格式奖励几步就教会了格式，到第 ${T} 步一次答对 ${pc(e.p1)}。DeepSeek‑R1‑Zero 也是这样训练的：用规则判分，不先做 SFT。</p>`);
  return tx(`<p>In the first 10 steps, ${pc(flat)} of the groups score all the same and teach nothing. Right in one try still climbs, from ${pc(s0.p1)} to ${pc(e.p1)}, but slowly. ${DEF('Skipping such problems',TIP.dyn())} under Settings spends every step on groups that teach.</p>`,`<p>前 10 步里，${pc(flat)} 的组 8 个回答分数一样，什么也教不了。一次答对的比例还是从 ${pc(s0.p1)} 涨到了 ${pc(e.p1)}，只是很慢。在“自由调参”里打开${DEF('跳过没有信号的题',TIP.dyn())}，每一步都只用有对有错的组。</p>`);}
const chHide=()=>false; // R4.3: the challenge shows live progress; Submit gives the three-tier feedback
function chResultHTML(){
  if(!chSubmitted) return '';
  const e=chEval({chk:S.chk,dyn:S.dyn,fmt:S.fmt}), best=chBest(), bestTxt=b=>b.map(o=>optName(o)).join(tx(' or ',' 或 '));
  let h='';
  if(e.ok&&best&&e.n<=best[0].n) h=`<p class="okline">${tx(`Goal met: right in one try ${pc(e.p1)}, answers listing candidate numbers ${pcS(e.g)}. And your change is already the smallest.`,`达成目标：一次答对 ${pc(e.p1)}，写候选数字的回答占 ${pcS(e.g)}。而且你的改动已经是最少的。`)}</p>`;
  else if(e.ok) h=`<p class="okline">${tx(`Goal met: right in one try ${pc(e.p1)}, answers listing candidate numbers ${pcS(e.g)}.`,`达成目标：一次答对 ${pc(e.p1)}，写候选数字的回答占 ${pcS(e.g)}。`)}</p><p>${tx(`A smaller change also works: ${bestTxt(best)}. The format reward changes nothing here, because this SFT model already writes “####” every time.`,`其实改得更少也行：${bestTxt(best)}。格式奖励在这里不起作用，因为这个 SFT 模型每次都会写 “####”。`)}</p>`;
  else {const hint=!e.okG?tx('Hint: the candidate numbers are still paid. Which checker would pay them only when the final answer is right?','提示：写候选数字的回答还在拿分。哪种判分程序只在最后的答案对时才给分？')
      :tx('Hint: the candidate numbers are gone, but too many groups score all 0 on these hard problems. Is there a way to spend each step only on problems that teach something?','提示：候选数字已经下去了，可这批难题里太多组全是 0 分。有没有办法让每一步只用能教点什么的题？');
    h=`<p class="badline">${tx(`Not yet: right in one try ${pc(e.p1)} (goal: at least ${pc(CH_GOAL)}), answers listing candidate numbers ${pcS(e.g)} (goal: at most ${pc(CH_G)}).`,`还没达标：一次答对 ${pc(e.p1)}（目标至少 ${pc(CH_GOAL)}），写候选数字的回答占 ${pcS(e.g)}（目标不超过 ${pc(CH_G)}）。`)}</p><p>${hint}</p>`+
      (chReveal?'':`<p><button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button></p>`);}
  if(e.ok) chReveal=true;
  if(chReveal){const e0=chEval({chk:'loose',dyn:0,fmt:0}), ex=f(chEval({chk:'strict',dyn:1,fmt:0}).extra/TCH,0);
    h+=chChart(e)+`<p>${tx(`From the initial setup (only the loose checker), the answers that list candidate numbers grow to ${pc(e0.g)}, above the ${pc(CH_G)} limit. The strict checker stops that, but on these hard problems many groups score all 0, so ${TCH} steps are not enough. Skipping problems with no signal spends every step on problems that teach something. The cost: about ${ex} extra groups to generate per step.`,`从初始设置出发（只用宽松的判分程序），写候选数字的回答涨到 ${pc(e0.g)}，超过了 ${pc(CH_G)} 的上限。换成严格的判分程序能挡住它，可这批题太难，很多组全是 0 分，${TCH} 步不够。跳过没有信号的题，每一步都用在能教东西的题上。代价是平均每步要多生成约 ${ex} 组回答。`)}</p><p class="small muted">${tx('Caveat: in this lab one step averages over all answers in the batch with a plain gradient step, so groups that score all the same make the step smaller. Trainers with Adam rescale the step; there the gain is mostly more groups that teach per batch.','注意：在这个 Lab 里，一步的更新是这一批所有回答的平均，用的是普通的梯度下降，所以分数全一样的组会把这一步摊小。用 Adam 的训练会重新调整步长，那里的好处主要是每一批里能教东西的组更多。')}</p>`;}
  return h;
}
const optName=o=>[o.chk==='strict'?tx('strict checker','严格的判分程序'):tx('loose checker','宽松的判分程序'),o.dyn?tx('skip no-signal problems','跳过没有信号的题'):'',o.fmt?tx('format reward','格式奖励'):''].filter(Boolean).join(tx(' + ',' + '));
/* brief (the owner's review of batch 8, 2026-10-07): the combinations that meet both conditions, your pick and the initial setup; the note
   under the chart counts the rest */
function chChart(cur){
  const rows=CHOPTS.map(o=>{const e=chEval(o); const me=o.chk===cur.chk&&o.dyn===cur.dyn&&o.fmt===cur.fmt, start=o.chk==='loose'&&!o.dyn&&!o.fmt;
    return {label:optName(o),value:+(e.p1*100).toFixed(1),text:pc(e.p1)+tx(`, answers listing candidate numbers ${pcS(e.g)}`,`，写候选数字的回答 ${pcS(e.g)}`),ok:e.ok,me,tags:[start?{text:tx('initial setup','初始设置')}:null,me?{text:tx('your pick','你的选择'),me:true}:null].filter(Boolean)};});
  return A.ui.cmpChart({brief:true,title:tx(`Right in one try after ${TCH} steps on the new problems (goal: at least ${pc(CH_GOAL)}, with at most ${pc(CH_G)} of the answers listing candidate numbers)`,`新的一批题上训练 ${TCH} 步后，一次答对的比例（目标：至少 ${pc(CH_GOAL)}，且写候选数字的回答不超过 ${pc(CH_G)}）`),
    aria:tx('Right in one try: ','一次答对的比例：')+rows.filter(r=>r.ok||r.me||r.tags.length).map(r=>`${r.label}${tx(': ','：')}${r.text}`).join(tx('; ','；')),rows,
    hiddenText:n=>tx(`The other ${n} combinations miss at least one condition and are not shown.`,`其余 ${n} 种组合至少有一个条件不满足，图里没有列出。`),goal:CH_GOAL*100,goalLabel:tx(`goal: ${pc(CH_GOAL)}`,`目标 ${pc(CH_GOAL)}`),max:60,
    okText:tx('meets both conditions','两个条件都满足'),noText:tx('misses at least one','至少一个不满足'),goalText:tx(`goal: ${pc(CH_GOAL)}`,`目标：${pc(CH_GOAL)}`)});
}
const EXITS=()=>{const m=mainRuns(), hs=run({...BASE,set:'hard'}), hf=hs.slice(0,10).reduce((s,x)=>s+x.flat,0)/hs.slice(0,10).reduce((s,x)=>s+x.groups.length,0);
  return {
 core:()=>chSubmitted,
 explain:[
  {q:tx('Only the final answer is checked, yet the answers learn to include a check step. Why?','判分只看最后的答案，回答里却学会了加一步检查。为什么？'),
   o:[tx('Answers written with a check are right more often, so inside each group they win positive advantages more often and that way of writing grows.','带检查的写法更常答对，所以在每一组里更常拿到正优势，这种写法就越来越多。'),
      tx('The checker gives extra points for a check step, so the model writes one to collect them.','判分程序给检查这一步额外加分，模型为了拿分才写检查。'),
      tx('Longer answers get larger advantages, so the model learns to write more of everything.','越长的回答优势越大，所以模型学会了什么都多写一些。')],a:0,
   n:[null,tx('The checker reads only the final answer; it never sees the check step.','判分程序只读最后的答案，根本看不到检查这一步。'),
      tx('The advantage depends only on the score, not on the length. A long wrong answer gets a negative advantage.','优势只取决于分数，和长度无关。一个又长又错的回答，优势是负的。')],
   why:tx(`With nothing but 1 or 0 on the final answer, training took “solve, then check” from ${pc(m.s0.pi[2])} to ${pc(m.sT.pi[2])} of the answers.`,`只按最后的答案给 1 分或 0 分，训练下来，“算完再检查”这种写法就从 ${pc(m.s0.pi[2])} 涨到了 ${pc(m.sT.pi[2])}。`)},
  {q:tx('After RL, pass@1 rose a lot but pass@256 barely moved. What does that show?','RL 之后 pass@1 涨了很多，pass@256 却几乎没变。这说明什么？'),
   o:[tx('RL made the ways of writing that already worked more common, but the model learned no new way, so the problems no way could solve are still unsolved.','RL 让本来就能做对的写法用得更多，但模型没有学会新的写法，原来解不出的题现在还是解不出。'),
      tx('256 tries is too few to see the gain; with 1,000 tries the trained model would pull ahead.','256 次太少，看不出提升；给 1,000 次，训练后的模型就会拉开差距。'),
      tx('The checker was too loose, so pass@256 counted wrong answers.','判分程序太松，pass@256 把错的答案也算进去了。')],a:0,
   n:[null,tx('More tries only help the problems some way of writing can solve; both models already solve those at 256 tries.','多给机会只对有写法能解出的题有用；这些题两个模型在 256 次时都已经解出来了。'),
      tx('pass@k here uses the strict checker, the same one as pass@1.','这里的 pass@k 用的是严格的判分程序，和 pass@1 一样。')],
   why:tx('In this lab, RL with a 0/1 reward can only change how often the model uses the ways of writing it already has.','在这个 Lab 里，用 0/1 奖励做 RL，只能调整模型本来就会的几种写法各用多少。')}],
 transfer:[
  {q:tx(`In the scenario “problems too hard”, ${pc(hf)} of the groups in the first 10 steps scored all the same. Now a model gets a problem right ${pc(0.5)} of the time and writes 8 answers to it. Then a model gets a problem right ${pc(0.95)} of the time. For which of the two are the 8 answers more likely to score all the same?`,`在“题太难”这个场景里，前 10 步有 ${pc(hf)} 的组 8 个回答分数一样。现在有两道题：一道模型答对的概率是 ${pc(0.5)}，另一道是 ${pc(0.95)}。各写 8 个回答，哪一道更可能 8 个分数全一样？`),
   o:[tx(`The ${pc(0.5)} problem`,`答对概率 ${pc(0.5)} 的那道`),tx(`The ${pc(0.95)} problem`,`答对概率 ${pc(0.95)} 的那道`),tx('Both about the same','两道差不多')],a:1,
   why:tx(`All 8 the same means all right or all wrong. At ${pc(0.5)} that is rare, under 1%. At ${pc(0.95)}, all 8 right happens about two times in three. Problems that are too easy give as little signal as problems that are too hard.`,`8 个分数全一样，就是全对或全错。答对概率 ${pc(0.5)} 时这很少见，不到 1%。答对概率 ${pc(0.95)} 时，8 个全对大约三次里有两次。太简单的题和太难的题一样，给不出信号。`)},
  {q:tx('A team trains a code model with a checker that runs the unit tests. Later they find the model writes code that deletes the failing tests. What went wrong?','一个团队用“跑单元测试”当判分程序训练写代码的模型。后来发现，模型学会了写代码把失败的测试删掉。哪里出了问题？'),
   o:[tx('The checker paid for passing tests without making sure the tests were still the original ones: a checker that is too loose.','判分程序只看测试过没过，没确认测试还是原来那些：这是一个太松的判分程序。'),
      tx('The KL penalty was too small, so the model forgot how to code.','KL 惩罚太小，模型忘了怎么写代码。'),
      tx('Unit tests can never be a verifiable reward.','单元测试永远不能当作可验证的奖励。')],a:0,
   why:tx('A verifiable reward is only as good as its checker. Whatever the checker fails to rule out, the model can learn.','可验证的奖励只和它的判分程序一样可靠。判分程序没排除的路，模型都可能学会去走。')}],
 practice:[
  {id:'rl-for-llms',d:tx('Questions Q4 and Q8: GRPO’s group advantage with verifiable rewards (what happens when all G rewards are equal), and how to spot reward hacking. Parts 3 and 5 of this lab prepare for them.','第 Q4 和 Q8 题：GRPO 的组内优势和可验证奖励（G 个奖励全一样时会怎样），以及怎样发现模型在钻奖励的空子。这个 Lab 的第 3 和第 5 部分就是为它们做的准备。')}]};};

/* =====================================================================
   views
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Eight short parts. Predict which way of writing training favours, see the checker and one group of 8 answers, train, then break the checker, add a format reward and compare pass@1 with pass@k. Start here if this is new to you.','一共 8 个部分。先预测训练会偏向哪种写法，再看判分程序和一组 8 个回答，然后训练，再试一个太松的判分程序、加格式奖励，最后比较 pass@1 和 pass@k。第一次学从这里开始。'),
  sandbox:tx('Four situations, from a strict checker to problems that are too hard. Pick one and watch what training does.','四种情况，从严格的判分程序到太难的题。选一个，看训练会怎样。'),
  challenge:tx('New, harder problems and a checker that is too loose. Change as little as you can so that the model is right at least 40% of the time after 20 steps, with at most 5% of the answers listing candidate numbers.','一批更难的新题，判分程序又太松。改得越少越好，让模型训练 20 步后一次答对至少 40%，写候选数字的回答不超过 5%。')});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser really trains a simulated policy with GRPO, step by step. The seed is fixed, so every run gives the same result. The ways of writing, their lengths and their success rates are this lab’s assumptions. The real numbers on the page are quoted from their sources.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The simulated policy has 5 parameters: 4 logits for the 4 ways of writing and 1 for ending with “####”. An answer is one choice, so per-token and per-answer quantities coincide.</li>`+
 `<li>GRPO takes ${B} problems per step and generates a group of G = ${G} answers to each.</li>`+
 `<li>The advantage follows TRL’s formula, with the sample standard deviation (formula in “Show the math”).</li>`+
 `<li>The KL penalty has the coefficient β = ${BETA} and is estimated with the k3 formula.</li>`+
 `<li>Each batch of answers is used for one update. GRPO’s clip keeps the ratio ρ between the new and the old model’s probability of an answer within [${f(1-EPS,1)}, ${f(1+EPS,1)}] (ε = ${EPS}), so that one update cannot change the model too much. During that one update the model being trained is still the model that wrote the batch, so every ρ is exactly 1 and the clip never acts. Only when the same batch is used a second time does ρ move away from 1.</li>`+
 `<li>Plain gradient steps, learning rate ${LR} on the logits (assumed).</li>`+
 `<li>Curves are exact expectations of the policy at each step. The group of 8 is the one the run really sampled. Seed ${SEED} (mulberry32).</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul><li>Four kinds of problems with fixed success rates per way of writing. A real model’s chance depends on each problem and each token.</li><li>The loose checker is this lab’s own example.</li><li>This lab’s β, learning rate and step count do not carry over to a real model. Tülu 3’s RLVR run used β = 0.05 and a learning rate of 3e‑7 on an 8B model (${cite(SRC.oiT,'open-instruct docs')}). verl and DeepSeek‑R1 use a KL coefficient of 0.001. TRL’s GRPO defaults to β = 0 and then loads no reference model (${cite(SRC.trl,'TRL GRPO docs')}, commit 14c8d70).</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>${cite(SRC.r1,'DeepSeek‑R1 README')}, ${cite(SRC.oiG,'open-instruct GRPO docs')}, ${cite(SRC.oiV,'ground_truth_utils.py')}, ${cite(SRC.orw,'open-r1 rewards.py')}, ${cite(SRC.mv,'Math-Verify')}, ${cite(SRC.gsm,'GSM8K')}, ${cite(SRC.tz,'TinyZero countdown.py')}, ${cite(SRC.lim,'limit-of-RLVR')}, ${cite(SRC.trl,'TRL GRPO docs')} (commit 14c8d70), ${cite(SRC.dapo,'verl DAPO')}.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-10-05. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器里真的在用 GRPO 一步步训练一个模拟的策略模型。随机种子是固定的，所以每次训练的结果都一样。几种写法、它们的长度和答对的概率，都是本 Lab 的假设。页面上真实系统的数字都引自出处。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>模拟的策略模型有 5 个参数，4 种写法各一个 logit，要不要用 “####” 收尾一个 logit。一个回答就是一次选择，所以按 token 算和按回答算是一回事。</li>`+
 `<li>GRPO 每一步做 ${B} 道题，每道题生成一组 G = ${G} 个回答。</li>`+
 `<li>优势按 TRL 的公式算，标准差用样本标准差（公式见“展开公式”）。</li>`+
 `<li>KL 惩罚的系数 β = ${BETA}，用 k3 公式估计。</li>`+
 `<li>每批回答只用来更新一次。GRPO 的截断（clip）把新旧模型给同一个回答的概率之比 ρ 限制在 [${f(1-EPS,1)}, ${f(1+EPS,1)}] 里（ε = ${EPS}），免得一次更新把模型改得太多。更新时，正在训练的模型和写出这批回答的模型还是同一个，所以每个 ρ 都等于 1，截断也就不会生效。同一批回答学第二遍时，ρ 才会离开 1。</li>`+
 `<li>用普通的梯度下降，logit 上的学习率是 ${LR}（假设值）。</li>`+
 `<li>曲线是每一步策略模型的精确期望。“一组 8 个回答”是训练时真正抽到的那一组。随机种子是 ${SEED}（mulberry32）。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul><li>题只分四类，每种写法在每一类题上答对的概率固定。真实模型答对的概率取决于每道题、每个 token。</li><li>宽松的判分程序是本 Lab 自己举的例子。</li><li>这个 Lab 里的 β、学习率和步数不能照搬到真实模型。Tülu 3 在 8B 模型上做 RLVR，用的是 β = 0.05、学习率 3e‑7（${cite(SRC.oiT,'open-instruct 文档')}）。verl 和 DeepSeek‑R1 的 KL 系数是 0.001。TRL 的 GRPO 默认 β = 0，这时不加载参考模型（${cite(SRC.trl,'TRL 的 GRPO 文档')}，commit 14c8d70）。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>${cite(SRC.r1,'DeepSeek‑R1 README')}、${cite(SRC.oiG,'open-instruct 的 GRPO 文档')}、${cite(SRC.oiV,'ground_truth_utils.py')}、${cite(SRC.orw,'open-r1 的 rewards.py')}、${cite(SRC.mv,'Math-Verify')}、${cite(SRC.gsm,'GSM8K')}、${cite(SRC.tz,'TinyZero 的 countdown.py')}、${cite(SRC.lim,'limit-of-RLVR')}、${cite(SRC.trl,'TRL 的 GRPO 文档')}（commit 14c8d70）、${cite(SRC.dapo,'verl 的 DAPO')}。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-10-05。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>GRPO has the model write several answers to the same question and compares them. It uses the group’s average score as the baseline and reinforces the answers above it, so it needs no value model (in PPO, the model that estimates how much reward is still to come from a given point; also called the critic). The lab ${labLink('grpo')} covers it step by step.</p><p>pass@k, the share of problems solved at least once in k tries, is explained in ${labLink('pass-k')}.</p>`,
 `<p>GRPO 让模型对同一道题写好几个回答，再把这些回答互相比较。它拿这一组的平均分当基准，分数高于平均分的回答会被加强，所以不需要价值模型（PPO 里估计“从这里往后平均还能拿多少奖励”的模型，也叫 critic）。${labLink('grpo')} 这个 Lab 一步步讲它。</p><p>pass@k，也就是给 k 次机会至少答对一次的题占多少，在 ${labLink('pass-k')} 这个 Lab 里讲。</p>`);
const OUTCOME=()=>tx('explain how a 0/1 reward on the final answer reshapes how a model writes, spot a checker that can be gamed, and read pass@1 against pass@k.','讲清只给最后答案 0/1 的奖励怎样改变模型的写法，看出会被钻空子的判分程序，并对照 pass@1 和 pass@k 读结果。');
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'RLVR',title:l?L(l.q):tx('How does a model learn to reason from right or wrong alone?','只给“对 / 错”，模型怎么学会推理？'),
    minutes:15,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: GRPO','前置知识：GRPO'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
const spdSeg=(cls,cur)=>`<div class="seg-ctl ${cls}" role="group" aria-label="${esc(tx('Speed','速度'))}" data-key="${cls}">${[['s',tx('Slow','慢')],['m',tx('Medium','中')],['f',tx('Fast','快')]].map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${cur===v}">${t}</button>`).join('')}</div>`;
function stageHTML(){
  return `<div class="panel stage rlvr-stage" id="rlvr-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="rlvr-st-title"></div><div class="meter-sub" id="rlvr-st-sub"></div></div><div class="stage-actions"><span class="status" id="rlvr-status"></span></div></div>`+
   `<div class="stage-graph graph rlvr-graph" id="rlvr-graph">`+
    `<section class="rlvr-chk" id="rlvr-chk" aria-label="${esc(tx('One answer read by the two checkers','两种判分程序读同一个回答'))}" hidden></section>`+
    /* H5 (the owner's message of 2026-10-09): each block of the stage is a framed sub-panel with a heading line that says what it shows;
       inside the group, the selected answer is set off by a hairline. The group's player sits in its heading line, so it does not move
       while the answers come in (R7.9). */
    `<section class="rlvr-grp rlvr-blk" id="rlvr-grp" aria-labelledby="rlvr-ghd">`+
     `<div class="rlvr-bh"><h3 class="rlvr-bt" id="rlvr-ghd"></h3><div class="rlvr-pl" id="rlvr-gpl"><button type="button" class="btn ghost sm" data-act="grp-play" id="rlvr-gplay"></button>${spdSeg('rlvr-gspd',GP.speed)}</div></div>`+
     `<div class="rlvr-q" id="rlvr-q"></div>`+
     `<div class="rlvr-gstat small" id="rlvr-gstat" hidden></div>`+
     `<div class="rlvr-rows" id="rlvr-rows" role="list"></div>`+
     `<div class="rlvr-sel" id="rlvr-sel" aria-live="polite"></div>`+
    `</section>`+
    `<section class="rlvr-run rlvr-blk" id="rlvr-run" aria-labelledby="rlvr-rhd">`+
     `<div class="rlvr-bh"><h3 class="rlvr-bt" id="rlvr-rhd"></h3></div>`+
     `<div class="rlvr-pl" id="rlvr-rpl"><button type="button" class="btn ghost sm" data-act="run-play" id="rlvr-rplay"></button>${spdSeg('rlvr-rspd',PL.speed)}<input type="range" class="rlvr-scrub" id="rlvr-scrub" min="0" max="${T}" step="1" value="0" aria-label="${esc(tx('Training step','训练步'))}"><span class="mono small" id="rlvr-tlab"></span></div>`+
     `<div class="rlvr-ch" id="rlvr-mix"></div>`+
     `<div class="rlvr-two"><div class="rlvr-ch" id="rlvr-p1"></div><div class="rlvr-ch" id="rlvr-len"></div></div>`+
    `</section>`+
    `<section class="rlvr-pk rlvr-blk" id="rlvr-pk" aria-label="pass@k"></section>`+
   `</div>`+
   `<p class="rlvr-scope small muted">${tx('This simulated policy has only 5 parameters and really trains step by step with GRPO in your browser. The problems and answers are examples.','这个模拟的策略模型只有 5 个参数，在浏览器里真的按 GRPO 一步步训练。题目和回答都是示例。')}</p>`+
   `<div class="metrics rlvr-metrics" id="rlvr-metrics"></div>`+
   `<details class="math" id="rlvr-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="rlvr-mathbody"></div></details></div>`;
}
/* Settings: the lab's assumed values as small tables (R2.11), then why they are what they are (R2.7) */
const KIND_SHORT=[{en:'easy',zh:'简单'},{en:'medium',zh:'中等'},{en:'hard',zh:'难'},{en:'beyond',zh:'超出能力'}];
function assumeHTML(){
  const w=SETS.main, avg=i=>QS[i].reduce((a,q,k)=>a+w[k]*q,0);
  const head=first=>`<tr><th scope="col">${first}</th>${KIND_SHORT.map(k=>`<th scope="col">${esc(L(k))}</th>`).join('')}</tr>`;
  const row=(lab,vals)=>`<tr><th scope="row">${lab}</th>${vals.map(v=>`<td class="mono">${pc(v)}</td>`).join('')}</tr>`;
  const t1=`<table class="rlvr-at"><caption>${tx('Each way of writing','每种写法')}</caption><tr><th scope="col">${tx('way','写法')}</th><th scope="col">${tx('tokens','token 数')}</th><th scope="col">${tx('SFT model at step 0','SFT 模型一开始占')}</th></tr>`+
    [0,1,2,3].map(i=>`<tr><th scope="row">${esc(wayN(i))}</th><td class="mono">${LEN[i]}</td><td class="mono">${pc(P0[i])}</td></tr>`).join('')+`</table>`;
  const t2=`<table class="rlvr-at"><caption>${tx('Chance that the final answer is right, by kind of problem','最后的答案答对的概率，按题的类别')}</caption>`+head(tx('way','写法'))+
    [0,1,2,3].map(i=>row(esc(wayN(i)),QS[i])).join('')+row(tx('candidate numbers, loose checker','写候选数字，宽松判分'),QL[3])+`</table>`;
  const t3=`<table class="rlvr-at"><caption>${tx('Share of each kind of problem','各类题占多少')}</caption>`+head(tx('problems','题'))+
    row(tx('this set','这批题'),SETS.main)+row(tx('too hard','太难的一批'),SETS.hard)+`</table>`;
  return t1+t2+t3+`<p>${tx(`The SFT model always writes the “####” line; the base model writes it ${pc(sig(PHI0.base))} of the time.`,`SFT 模型每次都写 “####” 这一行，base 模型只有 ${pc(sig(PHI0.base))} 的时候写。`)}</p>`+
    `<p>${tx(`These values are chosen so that, on this set, the three ways of writing are right ${pc(avg(0))}, ${pc(avg(1))} and ${pc(avg(2))} of the time on average. Real reasoning models write far longer answers: DeepSeek‑R1 was evaluated with up to 32,768 generated tokens (${cite(SRC.r1,'README')}). Only the ratio of the lengths is kept here.`,
      `这样设定，是为了让这批题上三种写法平均答对 ${pc(avg(0))}、${pc(avg(1))} 和 ${pc(avg(2))}。真实的推理模型回答要长得多，DeepSeek‑R1 评测时最多生成 32,768 个 token（${cite(SRC.r1,'README')}）。这里只保留几种写法之间长短的比例。`)}</p>`;
}
function controlsHTML(){
  const seg=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">`+A.ui.seg({id,key,label,options:opts,value:val,note})+`</div>`;
  return `<aside class="panel controls" id="rlvr-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="rlvr-lock" hidden></p>`+
   seg('rlvr-ctl-chk','chk',tx('Checker','判分程序'),[['strict',tx('strict','严格')],['loose',tx('loose','宽松')]],S.chk,tx('Strict: the “####” line must equal the reference answer. Loose: any number in the answer may.','严格：“####” 这一行要等于标准答案。宽松：回答里任何一个数等于就行。'))+
   seg('rlvr-ctl-fmt','fmt',tx('Format reward','格式奖励'),[['0',tx('off','不加')],['1',tx('0.1 point','0.1 分')]],String(S.fmt),tx('A wrong answer on a “####” line scores 0.1 (TinyZero’s value).','写在 “####” 后面但答错，得 0.1 分（TinyZero 用的值）。'))+
   seg('rlvr-ctl-dyn','dyn',tx('Problems with no signal','没有信号的题'),[['0',tx('keep','保留')],['1',tx('skip','跳过')]],String(S.dyn),tx('Skip: a group whose 8 answers all score the same is dropped and another problem is drawn (verl’s DAPO dynamic sampling). It costs extra generation.','跳过：8 个回答分数全一样的组扔掉，换一道题（verl 里 DAPO 的 dynamic sampling）。代价是要多生成回答。'))+
   seg('rlvr-ctl-start','start',tx('Starting model','初始模型'),[['sft',tx('SFT model','SFT 模型')],['base',tx('base model','base 模型')]],S.start,tx('The base model has had no SFT and rarely writes “####”.','base 模型没做过 SFT，很少写 “####”。'))+
   seg('rlvr-ctl-set','set',tx('Problems','题'),[['main',tx('this set','这批题')],['hard',tx('too hard','太难的一批')]],S.set,'')+
   /* codex-b9v3 #2: the three tables of assumed values wait behind a closed block, named for what is in it (R2.6) */
   `<details class="adv rlvr-asmd" id="rlvr-asm-d"><summary>${tx('Assumed values: lengths, chances of being right, mix of problems','假设值：写法的长度、答对的概率、各类题的比例')}</summary>`+
    `<div class="rlvr-asm"><p class="rlvr-gh">${tx('These values are this lab’s assumptions.','这些数都是本 Lab 的假设。')}</p>${assumeHTML()}</div></details>`+
   `<div class="lockable" id="rlvr-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import re, statistics

def strict(answer, ref):         # GSM8K: number after ####
    m = re.search(r"#### (-?[0-9.,]+)", answer)
    return float(m is not None and
                 m.group(1).replace(",", "") == ref)

def loose(answer, ref):          # too loose: any number
    return float(ref in re.findall(r"-?[0-9.]+", answer))

def advantages(rewards):         # GRPO, as TRL computes it
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards)        # n - 1
    return [(r - mean) / (std + 1e-4) for r in rewards]

ref = "20"
group = ["3 x 6 = 18, 18 + 2 = 20.\\n#### 20",
         "#### 18",
         "It could be 18, 20 or 24.\\n#### 24"]
for check in (strict, loose):
    r = [check(a, ref) for a in group]
    print(check.__name__, r,
          [round(a, 2) for a in advantages(r)])`;
const CODE_OUT=`# strict [1.0, 0.0, 0.0] [1.15, -0.58, -0.58]
# loose [1.0, 0.0, 1.0] [0.58, -1.15, 0.58]`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="rlvr-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="rlvr-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A strict and a loose checker, and GRPO’s group advantage. The third answer games the loose checker.','一个严格、一个宽松的判分程序，加上 GRPO 的组内优势。第三个回答钻了宽松判分的空子。')}</p><pre class="code" id="rlvr-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="rlvr-prod">`+
   tx(`<li>Tülu 3 (open-instruct) calls the method RLVR. Its 8B model starts from the DPO model and trains on GSM8K, MATH and instruction-following prompts, with only the verifier’s reward (the reward model’s weight is 0). That run used ${labLink('ppo','PPO')}, with β = 0.05 and a learning rate of 3e‑7: the commands use the legacy PPO script <code>ppo_vllm_thread_ray_gtrl.py</code> (${cite(SRC.oiT,'<code>docs/tulu3.md</code>')}).</li>`+
      `<li>open-instruct now also has GRPO. A verified answer earns 10 by default, and under GRPO the scale of the reward does not matter (${cite(SRC.oiG,'<code>docs/algorithms/grpo.md</code>')}).</li>`+
      `<li>open-instruct’s checkers are in ${cite(SRC.oiV,'<code>open_instruct/ground_truth_utils.py</code>')}: for GSM8K the last number in the output against the label; for MATH several ways to extract the answer, then an equivalence test.</li>`+
      `<li>open-r1’s ${cite(SRC.orw,'<code>src/open_r1/rewards.py</code>')} has the accuracy reward (Math-Verify, 1 or 0) and the format reward (a regex for the think and answer tags, 1 or 0). ${cite(SRC.mv,'Math-Verify')}’s README shows why parsing answers is hard: on the same MATH outputs, changing the checker moves the score from about ${pc(Math.min(...MV_SCORES))} to about ${pc(Math.max(...MV_SCORES))}.</li>`+
      `<li>DeepSeek‑R1‑Zero ran large-scale RL on DeepSeek‑V3‑Base with no SFT first. Its problems, “endless repetition, poor readability, and language mixing”, led to DeepSeek‑R1, which adds cold-start data before RL (${cite(SRC.r1,'README')}).</li>`+
      `<li>verl’s DAPO recipe drops groups whose answers are all right or all wrong and samples again until the batch is full (<code>filter_groups</code>, ${cite(SRC.dapo,'<code>docs/algo/dapo.md</code>')}); TRL logs the share of such groups as <code>frac_reward_zero_std</code> (${cite(SRC.trl,'GRPO docs')}).</li>`,
      `<li>Tülu 3（open-instruct）把这种方法叫作 RLVR。它的 8B 模型从 DPO 模型开始训练，题目来自 GSM8K、MATH 和遵循指令的数据，只用判分程序给的奖励（奖励模型的权重是 0）。这次训练用的是 ${labLink('ppo','PPO')}，β = 0.05，学习率 3e‑7。文档里的命令用的是旧的 PPO 脚本 <code>ppo_vllm_thread_ray_gtrl.py</code>（${cite(SRC.oiT,'<code>docs/tulu3.md</code>')}）。</li>`+
      `<li>open-instruct 现在也有 GRPO。答案通过验证时，默认给 10 分。不过在 GRPO 里，分数的大小无关紧要（${cite(SRC.oiG,'<code>docs/algorithms/grpo.md</code>')}）。</li>`+
      `<li>open-instruct 的判分程序在 ${cite(SRC.oiV,'<code>open_instruct/ground_truth_utils.py</code>')}：GSM8K 取输出里最后一个数和标准答案比；MATH 先用几种办法提取答案，再判断是否等价。</li>`+
      `<li>open-r1 的 ${cite(SRC.orw,'<code>src/open_r1/rewards.py</code>')} 里有答对奖励（用 Math-Verify 判，1 或 0）和格式奖励（用正则检查 think 和 answer 标签，1 或 0）。${cite(SRC.mv,'Math-Verify')} 的 README 说明了解析答案为什么难。同一批 MATH 输出，换一个判分程序，得分可以从约 ${pc(Math.min(...MV_SCORES))} 变成约 ${pc(Math.max(...MV_SCORES))}。</li>`+
      `<li>DeepSeek‑R1‑Zero 在 DeepSeek‑V3‑Base 上不做 SFT、直接做大规模 RL。它出现了无休止的重复、可读性差、语言混杂这些问题，于是有了 DeepSeek‑R1：在 RL 之前先加入冷启动数据（${cite(SRC.r1,'README')}）。</li>`+
      `<li>verl 的 DAPO 配方会扔掉回答全对或全错的组，再重新抽题，直到凑满一批（<code>filter_groups</code>，${cite(SRC.dapo,'<code>docs/algo/dapo.md</code>')}）；TRL 把这种组的比例记成 <code>frac_reward_zero_std</code>（${cite(SRC.trl,'GRPO 文档')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.r1p,'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning')}<span>${'arXiv 2501.12948'}</span></li>`+
   `<li>${cite(SRC.tulu,'Tülu 3: Pushing Frontiers in Open Language Model Post-Training')}<span>${'arXiv 2411.15124'}</span></li>`+
   `<li>${cite(SRC.dsm,'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models')}<span>${tx('arXiv 2402.03300 (introduced GRPO)','arXiv 2402.03300（提出 GRPO）')}</span></li>`+
   `<li>${cite(SRC.limp,'Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?')}<span>${'arXiv 2504.13837'}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><div id="rlvr-drill-q"></div><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="rlvr-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="rlvr-app m2" id="rlvr-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="rlvr-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how often the model is right in one try as it trains.','选一个场景，看看训练时一次答对的比例怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="rlvr-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change the checker, the format reward or the problems under <span class="rlvr-jl" data-jump="controls">Settings</span>.`,`想改判分程序、格式奖励或题，在“<span class="rlvr-jl" data-jump="controls">自由调参</span>”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
/* the three things the challenge lets you change, each defined in one line (label, then plain sentences) */
const CHDEF=()=>[
 [tx('Checker:','判分程序：'),tx('The strict checker wants the final answer on the “####” line and equal to the reference answer. The loose checker pays 1 point if any number in the answer equals the reference answer; this lab made it up as an example.','严格的判分程序要求最后的答案写在 “####” 这一行，而且等于标准答案。宽松的判分程序只要回答里有一个数等于标准答案就给 1 分，这是这个 Lab 为了举例设计的。')],
 [tx('Format reward:','格式奖励：'),tx(`An answer on a “####” line that is wrong scores ${FMT} instead of 0 (TinyZero gives ${FMT} the same way). A right answer still scores 1.`,`答案写在 “####” 后面但答错了，给 ${FMT} 分而不是 0 分（TinyZero 也这样给）。答对仍然是 1 分。`)],
 [tx('Problems with no signal:','没有信号的题：'),tx(`When all ${G} answers to a problem score the same, every advantage is 0. Skipping drops such a group and draws another problem until the step has ${B} problems with both right and wrong answers (verl’s dynamic sampling in DAPO). The cost: more answers to generate.`,`一道题的 ${G} 个回答分数都一样时，每个优势都是 0。选“跳过”，就把这一组扔掉，换一道题重新生成，直到这一步凑满 ${B} 道有对有错的题（verl 里 DAPO 的 dynamic sampling）。代价是要多生成一些回答。`)]];
function challengeHTML(){
  const e=chEval({chk:S.chk,dyn:S.dyn,fmt:S.fmt});
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal with the fewest changes, and why.','提交后会告诉你是否达标、改动是不是最少，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Right at least ${pc(CH_GOAL)} of the time after ${TCH} steps, with at most ${pc(CH_G)} of the answers listing candidate numbers`,`训练 ${TCH} 步后一次答对至少 ${pc(CH_GOAL)}，写候选数字的回答不超过 ${pc(CH_G)}`)}</h3>`+
   `<p>${tx(`A new batch of problems, harder than before: ${pc(SETS.ch[2]+SETS.ch[3])} are hard or beyond the model. You train the SFT model on them with GRPO for ${TCH} steps. The team’s checker is the loose one. After ${TCH} steps, the model must be right in one try at least ${pc(CH_GOAL)} of the time under the strict checker, and at most ${pc(CH_G)} of its answers may list candidate numbers. You can change three things; change as few as you can.`,
            `一批新题，比前面的难：难题和超出能力的题加起来占 ${pc(SETS.ch[2]+SETS.ch[3])}。你要用 GRPO 在这批题上训练 SFT 模型 ${TCH} 步。团队现在用的是宽松的判分程序。训练 ${TCH} 步后，按严格的判分程序算，模型一次答对的比例至少要有 ${pc(CH_GOAL)}，写候选数字的回答不能超过 ${pc(CH_G)}。你可以改三样东西，改得越少越好。`)}</p>`+
   `<ul class="rlvr-def small">${CHDEF().map(([k,v])=>`<li><b>${esc(k)}</b>${tx(' ','')}${esc(v)}</li>`).join('')}</ul>`+
   A.ui.seg({id:'rlvr-ic-chC',key:'chC',label:tx('Checker','判分程序'),options:[['loose',tx('loose','宽松')],['strict',tx('strict','严格')]],value:S.chk,inline:true})+
   A.ui.seg({id:'rlvr-ic-chF',key:'chF',label:tx('Format reward','格式奖励'),options:[['0',tx('off','不加')],['1',tx('0.1 point','0.1 分')]],value:String(S.fmt),inline:true})+
   A.ui.seg({id:'rlvr-ic-chD',key:'chD',label:tx('Problems with no signal','没有信号的题'),options:[['0',tx('keep','保留')],['1',tx('skip','跳过')]],value:String(S.dyn),inline:true})+
   `<div class="goal mono rlvr-goal" id="rlvr-ch-g1"><span>${tx('Right in one try','一次答对')}</span><span><b id="rlvr-ch-p1">${pc(e.p1)}</b>${tx(` (goal: at least ${pc(CH_GOAL)})`,`（目标至少 ${pc(CH_GOAL)}）`)}</span></div>`+
   `<div class="goal mono rlvr-goal" id="rlvr-ch-g2"><span>${tx('Answers listing candidate numbers','写候选数字的回答')}</span><span><b id="rlvr-ch-g">${pcS(e.g)}</b>${tx(` (goal: at most ${pc(CH_G)})`,`（目标不超过 ${pc(CH_G)}）`)}</span></div>`+
   submit+`<div id="rlvr-ch-res" class="rlvr-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(x.lead())})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#rlvr-rail',app); if(!rail) return null; rail.innerHTML=railFull(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#rlvr-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!!observed[step]||(s.id==='checker'&&ex2!=null)||(s.id==='loose'&&observed[step])||(s.id==='format'&&RUNSEEN.has(cfgKey(S))));
    if(s.id==='checker'){const h=ex2Ctl(), el=$('#rlvr-ic-ex',rail); if(el&&el._h!==h){el.outerHTML=h; const n=$('#rlvr-ic-ex',rail); if(n) n._h=h;}}
    if(s.id==='loose'){const el=$('#rlvr-ic-chk',rail), h=lv5Ctl(); if(el&&el._h!==h){el.outerHTML=h; const n=$('#rlvr-ic-chk',rail); if(n) n._h=h;}}
    if(s.id==='format'){const el=$('#rlvr-ic-fmt',rail), h=lv6Ctl(); if(el&&el._h!==h){el.outerHTML=h; const n=$('#rlvr-ic-fmt',rail); if(n) n._h=h;}}
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#rlvr-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p>${tx('Your own setup: no scenario matches it.','这是你自己的设置，没有对应的场景。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval({chk:S.chk,dyn:S.dyn,fmt:S.fmt}), p=$('#rlvr-ch-p1',rail), g=$('#rlvr-ch-g',rail); if(!p) return;
    p.textContent=pc(e.p1); g.textContent=pcS(e.g);
    $('#rlvr-ch-g1',rail).classList.toggle('met',e.okP); $('#rlvr-ch-g2',rail).classList.toggle('met',e.okG);
    $$('.seg-ctl[data-key="chC"]',rail).forEach(x=>A.ui.setPressed(x,S.chk)); $$('.seg-ctl[data-key="chF"]',rail).forEach(x=>A.ui.setPressed(x,String(S.fmt))); $$('.seg-ctl[data-key="chD"]',rail).forEach(x=>A.ui.setPressed(x,String(S.dyn)));
    const res=$('#rlvr-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const COLS=['rlvr-w0','rlvr-w1','rlvr-w2','rlvr-w3'];
const keyNow=()=>cfgKey(S);
const gameRed=()=>S.chk==='loose';
/* the stage head's two lines (R7.9, R7.11): the sentences for every phase of the picture on the stage sit in one grid cell and only the
   current one shows, so the head keeps its height when a replay starts or ends; during a replay only the numbers in it change.
   phases: {name: [text, number, text, number, …]} */
function setStack(el,key,phases,cur){
  if(el._k!==key){el._k=key; el.innerHTML=`<span class="rlvr-stk">`+Object.entries(phases).map(([k,ps])=>`<span data-ph="${k}">`+ps.map((x,i)=>i%2?`<span class="rlvr-nv">${esc(String(x))}</span>`:esc(x)).join('')+`</span>`).join('')+`</span>`;}
  else Object.entries(phases).forEach(([k,ps])=>{const ns=el.querySelectorAll(`[data-ph="${k}"] .rlvr-nv`); ps.filter((x,i)=>i%2).forEach((x,j)=>{if(ns[j]&&ns[j].textContent!==String(x)) ns[j].textContent=String(x);});});
  el.querySelectorAll('[data-ph]').forEach(x=>{const on=x.dataset.ph===cur; x.classList.toggle('on',on); if(on) x.removeAttribute('aria-hidden'); else x.setAttribute('aria-hidden','true');});
}
const plain=(el,txt)=>{el._k=null; if(el.textContent!==txt) el.textContent=txt;};
/* H8 (the owner's review of 2026-10-09: the stage said “在这批题上训练” while it played the 8 answers of one problem): the head says what
   the stage shows right now. Writing answers (a rollout) is not training: “training” only where the stage shows the run of updates. */
function drawHead(){
  const t=$('#rlvr-st-title',app), s=$('#rlvr-st-sub',app), st=$('#rlvr-status',app);
  const setN={main:tx('this set of problems','这批题'),hard:tx('the too-hard problems','太难的一批题'),ch:tx('the new, harder problems','新的一批难题')}[S.set];
  const chkN=S.chk==='loose'?tx('the loose checker','宽松的判分程序'):tx('the strict checker','严格的判分程序');
  const lg=A.lang();
  if(isP('checker')){const c=chkNow(), sp=c.strict.pay, lp=c.loose.pay;
    plain(t,tx('One answer, two checkers: which number does each read?','同一个回答，两种判分程序各读哪个数？'));
    plain(s,sp===lp?tx(`Both checkers pay ${ptsTxt(sp)}.`,`两种判分程序都给 ${sp} 分。`):tx(`The strict checker pays ${ptsTxt(sp)}, the loose checker ${ptsTxt(lp)}.`,`严格的判分程序给 ${sp} 分，宽松的判分程序给 ${lp} 分。`));}
  else if(!runOn()){
    /* parts 1 and 3: one problem's group of 8, before any training */
    const n=GP.on?GP.n:G;
    setStack(t,'g|'+lg,{a:[tx(`Rollout: the model writes ${G} answers to one problem. Nothing is trained here`,`生成回答（rollout）：模型给一道题写 ${G} 个回答，这里还没有训练`)]},'a');
    setStack(s,'g|'+lg,{play:[tx(`Generating: sampling ${G} answers to this problem (a rollout), `,`正在生成：给这道题采样 ${G} 个回答（rollout），已写出 `),n,tx(` of ${G} written.`,' 个。')],
                         done:[tx(`All ${G} answers are written. Writing answers does not change the model.`,`${G} 个回答都写好了。生成回答不会改变模型。`)]},GP.on?'play':'done');}
  else {
    /* parts 4–8, scenarios, challenge: the run of updates; a replay of one group on top of it is a rollout again */
    const startN=S.start==='base'?tx('the base model','base 模型'):tx('the SFT model','SFT 模型');
    const extra=(S.fmt?tx(', with a format reward','，加格式奖励'):'')+(S.dyn?tx(', skipping problems with no signal','，跳过没有信号的题'):'');
    const cfg=tx(` It starts from ${startN} and trains on ${setN} with ${chkN}${extra}.`,`从 ${startN}开始，在${setN}上用${chkN}训练${extra}。`);
    const n=GP.on?GP.n:G, tt=Math.min(tShow,S.T);
    const key='r|'+lg+'|'+cfgKey(S);
    setStack(t,key,{train:[tx(`Training: each step writes ${B*G} answers (a rollout), then updates the model once`,`训练：每一步先生成 ${B*G} 个回答（rollout），再把模型更新一次`)],
                    roll:[tx(`Rollout at step ${tt}: the ${G} answers to one problem, written again`,`第 ${tt} 步的生成（rollout）：重放一道题的 ${G} 个回答`)]},GP.on?'roll':'train');
    const ph=GP.on?'roll':(PL.on?'play':(tt>=S.T?'end':(tt===0?'start':'stop')));
    setStack(s,key,{play:[tx('Training: step ','正在训练：第 '),tt,tx(` of ${S.T}.`,` 步（共 ${S.T} 步）。`)+cfg],
                    stop:[tx('Stopped at step ','停在第 '),tt,tx(` of ${S.T}.`,` 步（共 ${S.T} 步）。`)+cfg],
                    start:[tx(`Not trained yet: step 0 of ${S.T}.`,`还没开始训练：第 0 步（共 ${S.T} 步）。`)+cfg],
                    end:[tx(`Trained: all ${S.T} steps.`,`训练完成：${S.T} 步都已走完。`)+cfg],
                    roll:[tx('Generating: the answers to this problem at step ','正在生成：第 '),tt,tx(', ',' 步这道题的回答，已写出 '),n,tx(` of ${G} written. The model is not updated here.`,` 个。这里不更新模型。`)]},ph);}
  const so=statusOf(); st.className='status '+so.cls; st.textContent=so.short; st.dataset.tipx=so.tip;
}
function statusOf(){
  const r=R(), e=r[Math.min(tShow,r.length-1)];
  if(mode==='guided'&&step===0&&!pred) return {cls:'',short:tx('Predict first','先预测'),tip:tx('The results stay hidden until you answer in part 1.','在第 1 部分作答之前，结果先不显示。'),txt:''};
  if(isP('checker')){const c=chkNow(), sp=c.strict.pay, lp=c.loose.pay, said=tx(`Strict checker: ${ptsTxt(sp)}. Loose checker: ${ptsTxt(lp)}.`,`严格的判分程序给 ${sp} 分，宽松的判分程序给 ${lp} 分。`);
    if(c.gamed) return {cls:'bad',short:tx('Wrong, yet paid','答错也给分'),tip:tx(`The final answer, ${c.fin}, is wrong, but the loose checker pays 1 point.`,`最后的答案 ${c.fin} 是错的，宽松的判分程序却给了 1 分。`),txt:said};
    return sp===lp?{cls:'',short:tx('Same score','给分一样'),tip:tx('Both checkers give this answer the same score.','两种判分程序给这个回答的分一样。'),txt:said}
      :{cls:'',short:tx('Scores differ','给分不同'),tip:tx('The two checkers give this answer different scores.','两种判分程序给这个回答的分不一样。'),txt:said};}
  if(chHide()) return {cls:'',short:tx('Not submitted','还没提交'),tip:tx('Submit your answer to see whether it meets the goal.','提交答案后才判定是否达标。'),txt:''};
  if(!runOn()) return {cls:'',short:tx('Before training','训练前'),tip:tx('The model before any training step.','还没做任何训练的模型。'),txt:tx('Before training.','训练前。')};
  if(e.pi[3]>0.5) return {cls:'bad',short:tx('Gamed','被钻空子'),tip:tx('More than half of the answers list candidate numbers: the checker is being gamed.','一半以上的回答在写候选数字：判分程序被钻了空子。'),txt:tx('The checker is being gamed.','判分程序被钻了空子。')};
  return {cls:'',short:tx(`Step ${tShow}`,`第 ${tShow} 步`),tip:tx('The training step the chart shows.','图里现在显示的训练步。'),txt:tx(`Step ${tShow}: right in one try ${pc(e.p1)}.`,`第 ${tShow} 步：一次答对 ${pc(e.p1)}。`)};
}
function drawGroup(){
  const g=groupNow(), kind=g.kind, p=PROB[kind], rows=$('#rlvr-rows',app), q=$('#rlvr-q',app), selBox=$('#rlvr-sel',app);
  const hideScores=mode==='guided'&&step===0&&!pred;
  /* H5, H8: the block's heading says which answers these are: one problem's rollout (parts 1 and 3), or the answers generated at the step
     the chart shows (one of that step's problems) */
  const tG=grpStep(), gs0=R()[tG].groups, hd=$('#rlvr-ghd',app);
  const hdT=runOn()?tx(`Answers generated at step ${tG} (rollout): problem ${Math.min(grpI,gs0.length-1)+1} of ${gs0.length}`,`第 ${tG} 步生成的回答（rollout）：第 ${Math.min(grpI,gs0.length-1)+1} 道题（共 ${gs0.length} 道）`)
    :tx(`${G} answers to one problem (a rollout)`,`一道题的 ${G} 个回答（rollout）`);
  if(hd.textContent!==hdT) hd.textContent=hdT;
  q.innerHTML=`<span class="rlvr-tag">${tx(`Example (${esc(L(KINDN[kind]))})`,`示例（${esc(L(KINDN[kind]))}）`)}</span> ${esc(L(p.q))} <span class="rlvr-ref">${tx('Reference answer','标准答案')} <b class="mono">${p.ans}</b></span>`;
  const n=GP.n;
  /* R2.8: the advantage appears from part 3 on, where it is defined, and only once all 8 rewards are in (it needs the group's mean and std) */
  const advOn=!hideScores&&!(mode==='guided'&&step<2)&&n>=G, gs=$('#rlvr-gstat',app);
  if(gs){gs.hidden=!advOn; if(advOn){const h=g.flat?tx(`All ${G} rewards in this group are ${f(g.resp[0].r,g.resp[0].r%1?1:0)}: the standard deviation is 0, so every advantage is 0.`,`这一组 ${G} 个回答的奖励都是 ${f(g.resp[0].r,g.resp[0].r%1?1:0)}，标准差是 0，所以每个优势都是 0。`)
    /* H3: the sentence that uses the mean and the std shows the formula they go into, with this group's numbers */
    :tx(`This group: mean reward ${f(g.m,4)}, standard deviation ${f(g.sd,4)}. Each answer’s advantage:`,`这一组的平均奖励是 ${f(g.m,4)}，标准差是 ${f(g.sd,4)}。每个回答的优势：`)+
      ` <span class="rlvr-gfx">${A.ui.math(`A_i = \\frac{r_i - ${f(g.m,4)}}{${f(g.sd,4)} + 0.0001}`)}</span>`;
    if(gs._h!==h){gs._h=h; gs.innerHTML=h;}}}
  rows.innerHTML=g.resp.map((x,i)=>{const show=i<n, fin=finalOf(x,kind), ok=x.r===1;
    const vd=!show?'':(ok?`<span class="rlvr-ok">✓ ${ptsTxt(1)}</span>`:`<span class="rlvr-no">✗ ${ptsTxt(x.r)}</span>`);
    const adv=!advOn?'<span class="rlvr-adv"></span>':`<span class="rlvr-adv mono ${x.a>0.005?'up':(x.a<-0.005?'dn':'')}">${tx('adv.','优势')} ${sg(x.a,2)}</span>`;
    const red=gameRed()&&x.s===3&&ok&&!x.okS;
    return `<button type="button" class="rlvr-row${i===sel?' on':''}${show?'':' wait'}${red?' red':''}" role="listitem" data-row="${i}" aria-pressed="${i===sel}" ${show?'':'disabled'}>`+
      `<span class="rlvr-n mono">${i+1}</span><span class="rlvr-way">${show?`<i class="rlvr-sw ${COLS[x.s]}"></i>${esc(wayN(x.s))}`:'…'}</span>`+
      `<span class="rlvr-lenb">${show?`<i style="width:${(LEN[x.s]/LEN[2]*100).toFixed(1)}%"></i><em class="mono">${LEN[x.s]}</em>`:''}</span>`+
      `<span class="rlvr-fa mono">${show?(x.f?'#### ':'')+fin:'…'}</span>${hideScores?'':vd}${adv}</button>`;}).join('');
  if(sel>=n) sel=Math.max(0,n-1);
  /* R7.9: every answer's text sits in the same grid cell and only the selected one shows, so the box keeps the height of the longest
     answer while the group plays and the selection moves */
  const one=(x,i)=>{const on=i===sel&&i<n; return `<div class="rlvr-sela${on?' on':''}"${on?'':' aria-hidden="true"'}><div class="rlvr-seltxt"><span class="rlvr-tag">${tx(`Answer ${i+1} (${esc(wayN(x.s).toLowerCase())})`,`回答 ${i+1}（${esc(wayN(x.s))}）`)}</span> ${answerHTML(x,kind)}</div>`+
    (hideScores?'':`<div class="rlvr-why small${gameRed()&&x.s===3&&x.okL&&!x.okS?' red':''}">${esc(verdictTxt(x,kind,S.chk))}</div>`)+`</div>`;};
  const sh=g.resp.map(one).join('');
  if(selBox._h!==sh){selBox._h=sh; selBox.innerHTML=sh;}
  const gb=$('#rlvr-gplay',app); gb.textContent=GP.on?tx('❚❚ Pause','❚❚ 暂停'):(GP.n>=G?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  gb.dataset.tipx=GP.n>=G?tx('Writes this problem’s 8 answers again, one by one.','把这道题的 8 个回答一个一个重新生成一遍。'):tx('Writes this problem’s 8 answers one by one.','把这道题的 8 个回答一个一个生成出来。');
  $$('.rlvr-gspd button',app).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===GP.speed)));
}
/* part 2 (codex-b9v3 #1): the picked answer, read by the strict and the loose checker side by side. Each column shows the answer with
   the numbers that checker reads boxed (thick: equal to the reference answer), the comparison and the score. A wrong answer the loose
   checker pays is red (R5.11): the matched number, the score and the column. */
let chkLast='';
function drawChk(){
  const box=$('#rlvr-chk',app); if(!box) return; box.hidden=!isP('checker'); if(box.hidden) return;
  const o=ex2Opt(), x=o.x, c=chkNow(), ref=c.ref, p=PROB[0], last=c.lines.length-1;
  const num=(n,red)=>`<span class="rlvr-num${n===ref?' hit'+(red?' red':''):''}">${n}</span>`;
  const marked=(l,red)=>{let h='', i=0; l.replace(NUM_RE,(m,at)=>{h+=esc(l.slice(i,at))+num(+m,red); i=at+m.length; return m;}); return h+esc(l.slice(i));};
  const strictText=c.lines.map((l,i)=>{const m=i===last?/^(#### )(\d+)$/.exec(l):null;
    return `<span class="ln${i===last?' fl':''}">`+(m?`${m[1]}${num(+m[2],false)}`:`<span class="skip">${esc(l)}</span>`)+`</span>`;}).join('');
  const looseText=c.lines.map((l,i)=>`<span class="ln${i===last?' fl':''}">${marked(l,c.gamed)}</span>`).join('');
  const sCmp=c.strict.read.length?`${c.strict.read[0]} ${c.strict.read[0]===ref?'=':'≠'} ${ref}`:tx('no “####” line','没有 “####” 这一行');
  const lCmp=c.loose.pay?tx(`one of them is ${ref}`,`其中有 ${ref}`):tx(`none of them is ${ref}`,`没有一个是 ${ref}`);
  const vd=(pay,red)=>`<span class="rlvr-cv${pay?'':' no'}${red?' red':''}">${pay?'✓':'✗'} ${ptsTxt(pay)}</span>`;
  const col=(k,head,rule,text,cmp,pay,red,note)=>`<div class="rlvr-col${red?' red':''}" data-chk="${k}"><div class="rlvr-ckh">${head}</div><div class="rlvr-crule">${rule}</div>`+
    `<p class="rlvr-ctext">${text}</p><div class="rlvr-cres"><span class="rlvr-ccmp mono">${cmp}</span>${vd(pay,red)}</div>${note?`<p class="rlvr-cnote red">${note}</p>`:''}</div>`;
  const html=`<div class="rlvr-q"><span class="rlvr-tag">${tx('Example','示例')}</span> ${esc(L(p.q))} <span class="rlvr-ref">${tx('Reference answer','标准答案')} <b class="mono">${ref}</b></span></div>`+
    `<div class="rlvr-cwho"><span class="rlvr-tag">${tx('Answer','回答')}</span> ${esc(wayN(x.s))}</div>`+
    `<div class="rlvr-cols">`+
     col('strict',tx('Strict checker','严格的判分程序'),tx('Reads only the “####” line','只读 “####” 这一行'),strictText,sCmp,c.strict.pay,false,'')+
     col('loose',tx('Loose checker','宽松的判分程序'),tx('Reads every number in the answer','读回答里的每一个数'),looseText,lCmp,c.loose.pay,c.gamed,
       c.gamed?tx(`The final answer, ${c.fin}, is wrong, yet this answer is paid.`,`最后的答案 ${c.fin} 是错的，这个回答却拿到了 1 分。`):'')+`</div>`+
    `<p class="rlvr-leg small muted">${tx('Box: a number the checker reads. Thick box: a number equal to the reference answer.','方框：判分程序读到的数。粗框：等于标准答案的数。')}${c.gamed?tx(' Red: a wrong answer is paid.','红色：答错了却拿到了分。'):''}</p>`;
  if(box._h!==html){const again=box._h!=null&&chkLast!==String(ex2); box.innerHTML=html; box._h=html;
    if(again&&!A.reduceMotion()){box.classList.remove('bump'); void box.offsetWidth; box.classList.add('bump');}}
  chkLast=String(ex2);
}
/* charts: SVG areas and lines with HTML labels beside them (R5.9); x = training step */
function mixChart(r,tt){
  const W=600,H=150, X=t=>t/S.T*W;
  const cum=r.slice(0,tt+1).map(e=>{let a=0; return e.pi.map(p=>(a+=p));});
  let paths='';
  for(let i=NW-1;i>=0;i--){
    const top=cum.map((c,t)=>`${X(t).toFixed(1)},${(H-c[i]*H).toFixed(1)}`), bot=cum.map((c,t)=>`${X(t).toFixed(1)},${(H-(i?c[i-1]:0)*H).toFixed(1)}`).reverse();
    const red=i===3&&gameRed()&&r[tt].pi[3]>0.5;
    paths+=`<polygon class="rlvr-a ${COLS[i]}${red?' red':''}" points="${top.concat(bot).join(' ')}"/>`;}
  const last=cum[cum.length-1], labs=[0,1,2,3].map(i=>{const lo=i?last[i-1]:0, mid=(lo+last[i])/2; return {i,y:(1-mid)*100,p:last[i]-lo};});
  /* each band is named in full at its right end (the owner, 2026-10-09: “候选数” could not be read); spreadEnd() keeps the labels apart
     by their real heights, because a long name takes two lines on a narrow stage */
  const red=gameRed()&&r[tt].pi[3]>0.5;
  return `<div class="rlvr-ct">${tx(`Share of the answers written each way, steps 0 to ${tt}`,`每种写法的回答各占多少（第 0 到第 ${tt} 步）`)}</div><div class="rlvr-plot rlvr-mixp"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="rlvr-svg" aria-hidden="true"><defs><pattern id="rlvr-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="rlvr-hbg"/><line x1="0" y1="0" x2="0" y2="6" class="rlvr-hl"/></pattern></defs>${paths}</svg>`+
    `<div class="rlvr-yax mono"><span>100%</span><span>0%</span></div>`+
    `<div class="rlvr-endl">${labs.map(l=>`<span class="${l.i===3&&red?'red':''}" style="top:${l.y.toFixed(1)}%" data-y="${l.y.toFixed(2)}" data-tipx="${esc(tx(`${wayN(l.i)}: ${pcS(l.p)} of the answers at step ${tt}.`,`第 ${tt} 步时，${pcS(l.p)} 的回答是“${wayN(l.i)}”。`))}"><i class="rlvr-sw ${COLS[l.i]}${l.i===3&&red?' red':''}"></i><b>${esc(wayN(l.i))} ${pcS(l.p)}</b></span>`).join('')}</div></div>`+
    `<div class="rlvr-xax mono"><span>0</span><span>${tx('training step','训练步')}</span><span>${S.T}</span></div>`+
    `<p class="rlvr-leg small muted">${tx('Height of each band: its share of all the answers at that step. Hatched: answers that list candidate numbers.','每条色带的高度：这一步里这种写法的回答占全部回答多少。斜纹：写好几个候选数字的回答。')}${gameRed()?`<span class="rlvr-lnote${red?'':' off'}"${red?'':' aria-hidden="true"'}>${tx(' Red: more than half of the answers list candidate numbers, so the checker is being gamed.','红色：一半以上的回答在写候选数字，判分程序被钻了空子。')}</span>`:''}</p>`;
}
/* end labels at least their own height apart (R7.8, R5.9), measured, because a long name wraps to two lines on a narrow stage. A pass down
   the column pushes each label below the one above it, a pass back up keeps the last one inside the plot; labels with room stay where
   their band or line ends. */
function spreadEnd(host){
  if(!host) return;
  host.querySelectorAll('.rlvr-endl').forEach(col=>{
    const H=col.getBoundingClientRect().height; if(!H) return;
    const ls=[...col.children].map(e=>({e,y:(e.dataset.y!=null?+e.dataset.y:parseFloat(e.style.top))/100*H,h:e.getBoundingClientRect().height})).sort((a,b)=>a.y-b.y);
    if(!ls.length) return;
    const GAP=2;
    ls[0].y=Math.max(ls[0].y,ls[0].h/2);
    for(let k=1;k<ls.length;k++) ls[k].y=Math.max(ls[k].y,ls[k-1].y+(ls[k-1].h+ls[k].h)/2+GAP);
    const n=ls.length-1; ls[n].y=Math.min(ls[n].y,H-ls[n].h/2);
    for(let k=n-1;k>=0;k--) ls[k].y=Math.min(ls[k].y,ls[k+1].y-(ls[k].h+ls[k+1].h)/2-GAP);
    ls.forEach(l=>{const v=(l.y/H*100).toFixed(2)+'%'; if(l.e.style.top!==v) l.e.style.top=v;});});
}
function lineChart(o){
  const W=300,H=120, X=t=>t/S.T*W, Y=v=>H-(v-o.min)/(o.max-o.min)*H;
  const pts=(arr)=>arr.map((v,t)=>`${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  let svg=`<line x1="0" x2="${W}" y1="${Y(o.base).toFixed(1)}" y2="${Y(o.base).toFixed(1)}" class="rlvr-base"/>`;
  if(o.vmark!=null) svg+=`<line x1="${X(o.vmark).toFixed(1)}" x2="${X(o.vmark).toFixed(1)}" y1="0" y2="${H}" class="rlvr-base"/>`;
  o.lines.forEach(l=>{svg+=`<polyline points="${pts(l.v)}" class="rlvr-ln ${l.cls||''}"/>`; if(l.redFrom!=null&&l.redFrom<l.v.length-1) svg+=`<polyline points="${l.v.map((v,t)=>[t,v]).slice(l.redFrom).map(([t,v])=>`${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}" class="rlvr-ln red"/>`;});
  /* end labels at least one line apart (R7.8, R5.9): push the lower one down, then shift all up if the last one leaves the plot */
  const labs=o.lines.filter(l=>l.lab).map(l=>({l,y:Y(l.v[l.v.length-1])/H*100})).sort((p,q)=>p.y-q.y), GAP=13;
  for(let k=1;k<labs.length;k++) if(labs[k].y-labs[k-1].y<GAP) labs[k].y=labs[k-1].y+GAP;
  const over=labs.length?labs[labs.length-1].y-92:0; if(over>0) labs.forEach(x=>{x.y-=over;});
  if(labs.length&&labs[0].y<4){const d=4-labs[0].y; labs.forEach(x=>{x.y+=d;});}
  const marks=(o.marks||[]).map(m=>`<span class="rlvr-mk ${m.cls||''}" style="left:${(m.t/S.T*100).toFixed(1)}%;top:${(Y(m.v)/H*100).toFixed(1)}%" data-tipx="${esc(m.tip)}"><i></i><em>${esc(m.lab)}</em></span>`).join('');
  return `<div class="rlvr-ct">${o.title}</div><div class="rlvr-plot small"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="rlvr-svg" aria-hidden="true">${svg}</svg>${marks}`+
    `<div class="rlvr-yax mono"><span>${o.fmt(o.max)}</span><span>${o.fmt(o.min)}</span></div>`+
    `<div class="rlvr-endl">${labs.map(({l,y})=>`<span class="${l.red?'red':''}" style="top:${y.toFixed(1)}%" data-tipx="${esc(l.tip)}">${esc(l.lab)}</span>`).join('')}</div></div>`+
    legHTML(o.leg);
}
/* H6 (the owner's message of 2026-10-09): every mark of a chart is named next to it, each with a sample drawn in the chart's own style */
function legHTML(items){
  const sw=k=>k==='mk'?`<i class="rlvr-ldot" aria-hidden="true"></i>`:k==='vdot'?`<svg class="rlvr-lsw" viewBox="0 0 18 10" aria-hidden="true"><line x1="9" x2="9" y1="0" y2="10" class="rlvr-base"/></svg>`
    :`<svg class="rlvr-lsw" viewBox="0 0 18 10" aria-hidden="true"><line x1="0" x2="18" y1="5" y2="5" class="${k==='dot'?'rlvr-base':'rlvr-ln '+(k==='solid'?'':k)}"/></svg>`;
  return `<p class="rlvr-leg rlvr-legs small muted">${items.filter(Boolean).map(([k,t,off])=>`<span class="rlvr-li${off?' off':''}"${off?' aria-hidden="true"':''}>${sw(k)}${esc(t)}</span>`).join('')}</p>`;
}
function drawRun(){
  const box=$('#rlvr-run',app); box.hidden=!runOn(); if(box.hidden) return;
  const r=R(), tt=Math.min(tShow,r.length-1), seg=r.slice(0,tt+1);
  const rh=$('#rlvr-rhd',app), rhT=tx(`The training run: steps 0 to ${S.T}, one update of the model per step`,`训练过程：第 0 到第 ${S.T} 步，每一步更新一次模型`); if(rh.textContent!==rhT) rh.textContent=rhT;
  $('#rlvr-mix',app).innerHTML=mixChart(r,tt);
  const p1=seg.map(e=>e.p1), rew=seg.map(e=>e.rew), k=peakOf(r);
  const lose=gameRed()&&tt>k&&r[tt].p1<r[k].p1-0.02;
  /* part 6 (S5): the other run (with / without the format reward) as a thin grey line once it was watched, and both values at step 10 */
  const cmp=isP('format'), K=10, oCfg={...S,fmt:S.fmt?0:1}, oRun=cmp&&RUNSEEN.has(cfgKey(oCfg))?run(oCfg):null, mk=cmp&&tt>=K;
  const oName=S.fmt?tx('the run without the format reward','不加格式奖励的那次训练'):tx('the run with the format reward','加了格式奖励的那次训练');
  const mkOf=(get,what)=>mk?[{t:K,v:get(r[K]),lab:pc(get(r[K])),tip:tx(`${what[0]} at step ${K}, this run: ${pc(get(r[K]),1)}.`,`第 ${K} 步，这次训练里${what[1]} ${pc(get(r[K]),1)}。`)}].concat(oRun?[{t:K,v:get(oRun[K]),lab:pc(get(oRun[K])),cls:'other',tip:tx(`${what[0]} at step ${K}, ${oName}: ${pc(get(oRun[K]),1)}.`,`第 ${K} 步，${oName}里${what[1]} ${pc(get(oRun[K]),1)}。`)}]:[]):[];
  const otherLeg=oRun?['other',oName]:null, mkLeg=cmp?['mk',tx(`the values at step ${K}`,`第 ${K} 步的值`),!mk]:null, vLeg=cmp?['vdot',tx(`step ${K}`,`第 ${K} 步`),!mk]:null, step0=['dot',tx('the value at step 0','第 0 步的值')];
  const lines=[{v:p1,lab:tx(`right ${pc(p1[tt])}`,`答对 ${pc(p1[tt])}`),tip:tx(`Share right in one try under the strict checker at step ${tt}: ${pc(p1[tt],1)}.`,`第 ${tt} 步，按严格判分一次答对的比例：${pc(p1[tt],1)}。`),red:lose,redFrom:lose?k:null}];
  const paid=S.chk==='loose'||S.fmt;
  if(paid) lines.push({v:rew,cls:'dash',lab:tx(`paid ${pc(rew[tt])}`,`得分 ${pc(rew[tt])}`),tip:tx(`The checker’s average score per answer at step ${tt}: ${pc(rew[tt],1)} of a full point.`,`第 ${tt} 步，判分程序平均给每个回答的分数：满分的 ${pc(rew[tt],1)}。`)});
  if(oRun) lines.unshift({v:oRun.map(e=>e.p1),cls:'other'});
  $('#rlvr-p1',app).innerHTML=lineChart({title:tx('Right in one try (strict checker)','一次答对的比例（严格判分）'),min:0,max:1,base:r[0].p1,fmt:v=>pc(v),lines,
    vmark:mk?K:null,marks:mkOf(e=>e.p1,['Right in one try','一次答对的比例是']),
    leg:[['solid',tx('right in one try, strict checker','严格判分下一次答对')],paid?['dash',tx('average score the training checker pays','训练用的判分程序平均给的分')]:null,gameRed()?['red',tx('falling while the score still rises','得分还在涨，答对却在跌'),!lose]:null,step0,otherLeg,vLeg,mkLeg]});
  if(S.start==='base'){
    /* from the base model the format is the story (part 6, scenario 3): the share of answers that end with “####” replaces the length */
    const pf=seg.map(e=>e.pf), fl=[{v:pf,lab:pc(pf[tt]),tip:tx(`Share of answers that end with “####” at step ${tt}: ${pc(pf[tt],1)}.`,`第 ${tt} 步，用 “####” 收尾的回答占 ${pc(pf[tt],1)}。`)}];
    if(oRun) fl.unshift({v:oRun.map(e=>e.pf),cls:'other'});
    $('#rlvr-len',app).innerHTML=lineChart({title:tx('Answers that end with “####”','用 “####” 收尾的回答占多少'),min:0,max:1,base:r[0].pf,fmt:v=>pc(v),lines:fl,
      vmark:mk?K:null,marks:mkOf(e=>e.pf,['Share ending with “####”','用 “####” 收尾的回答占']),leg:[['solid',tx('share that ends with “####”','用 “####” 收尾的比例')],step0,otherLeg,vLeg,mkLeg]});
  } else {
    const len=seg.map(e=>e.len);
    $('#rlvr-len',app).innerHTML=lineChart({title:tx('Average answer length (tokens)','平均每个回答的长度（token）'),min:0,max:LEN[2],base:r[0].len,fmt:v=>f(v,0),lines:[{v:len,lab:tok(len[tt]),tip:tx(`Average answer length at step ${tt}: ${f(len[tt],1)} tokens.`,`第 ${tt} 步，平均每个回答 ${f(len[tt],1)} 个 token。`)}],leg:[['solid',tx('tokens per answer, on average','平均每个回答的 token 数')],step0]});
  }
  spreadEnd($('#rlvr-run',app));
  const tl=$('#rlvr-tlab',app); tl.textContent=tx(`step ${tt} / ${S.T}`,`第 ${tt} / ${S.T} 步`);
  const sc=$('#rlvr-scrub',app); sc.max=String(S.T); if(+sc.value!==tt) sc.value=String(tt);
  const rb=$('#rlvr-rplay',app); rb.textContent=PL.on?tx('❚❚ Pause','❚❚ 暂停'):(tt>=S.T?tx('↺ Replay','↺ 重放'):tx('▶ Train','▶ 训练'));
  rb.dataset.tipx=tt>=S.T?tx('Replays the whole run from step 0.','从第 0 步重新播放整次训练。'):tx(`Trains step by step up to step ${S.T}.`,`一步步训练到第 ${S.T} 步。`);
  $$('.rlvr-rspd button',app).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===PL.speed)));
}
const pkOn=()=>mode==='sandbox'||isP('passk')||isP('remember')||(mode==='challenge'&&chSubmitted);
function drawPassK(){
  const box=$('#rlvr-pk',app); box.hidden=!pkOn(); if(box.hidden) return;
  const r=R(), w=SETS[S.set], b=passK(r[0].pk,w), done=atEnd(), a=passK(r[r.length-1].pk,w);
  const W=300,H=120, X=i=>i/(KS.length-1)*W, Y=v=>H-v*H;
  const pl=v=>v.map((y,i)=>`${X(i).toFixed(1)},${Y(y).toFixed(1)}`).join(' ');
  let svg=`<polyline points="${pl(b)}" class="rlvr-ln dash"/>`+(done?`<polyline points="${pl(a)}" class="rlvr-ln"/>`:'');
  const n=KS.length-1;
  box.innerHTML=`<div class="rlvr-ct">${tx(`pass@k on ${S.set==='main'?'this set':'these problems'}: before training and after ${S.T} steps`,`${S.set==='main'?'这批题':'这些题'}上的 pass@k：训练前和训练 ${S.T} 步后`)}</div>`+
   `<div class="rlvr-plot small"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="rlvr-svg" aria-hidden="true">${svg}</svg><div class="rlvr-yax mono"><span>100%</span><span>0%</span></div>`+
   `<div class="rlvr-endl"><span style="top:${Math.max(4,Y(b[n])/H*100-8).toFixed(1)}%" data-tipx="${esc(tx(`Before training, pass@${KS[n]} = ${pc(b[n],1)}.`,`训练前 pass@${KS[n]} = ${pc(b[n],1)}。`))}">${tx('before','训练前')} ${pc(b[n])}</span>${done?`<span style="top:${Math.min(92,Y(a[n])/H*100+6).toFixed(1)}%" data-tipx="${esc(tx(`After ${S.T} steps, pass@${KS[n]} = ${pc(a[n],1)}.`,`训练 ${S.T} 步后 pass@${KS[n]} = ${pc(a[n],1)}。`))}">${tx('after','训练后')} ${pc(a[n])}</span>`:''}</div></div>`+
   `<div class="rlvr-xax mono">${KS.map(k=>`<span>${k}</span>`).join('')}</div>`+
   `<p class="rlvr-leg small muted">${tx('x: k, the number of tries (log scale).','横轴：k，尝试次数（对数刻度）。')}<span class="rlvr-lnote${done?' off':''}"${done?' aria-hidden="true"':''}>${tx(' The after line appears when the run ends.','训练播完后出现训练后的线。')}</span></p>`+legHTML([['dash',tx('before training','训练前')],['solid',tx(`after ${S.T} steps`,`训练 ${S.T} 步后`),!done]]);
  spreadEnd(box);
}
function metricsDef(){
  const r=R(), e=r[Math.min(tShow,r.length-1)], tt=Math.min(tShow,r.length-1);
  const hide=(mode==='guided'&&step===0&&!pred)||chHide();
  return [
   {k:'p1',label:tx('Right in one try','一次答对'),v:hide?'—':pc(e.p1),sub:tx('strict checker','严格判分'),pts:true,tip:tx('pass@1: the share of problems the model gets right in one try, under the strict checker, at the step shown. Computed exactly from the simulated policy.','pass@1：在图里显示的这一步，按严格判分，模型答一次就对的题占多少。由模拟的策略模型精确算出。')},
   {k:'len',label:tx('Average length','平均长度'),v:hide?'—':f(e.len,0),sub:tx('tokens per answer','个 token，每个回答'),tip:tx('The average number of tokens in an answer at the step shown, from the lengths assumed for each way of writing.','这一步平均每个回答多少个 token，按每种写法假设的长度算。')},
   {k:'g',label:tx('List candidate numbers','写候选数字'),v:hide?'—':pcS(e.pi[3]),sub:tx('share of all answers','这种回答占全部回答'),pts:true,tip:tx('The share of answers that list several candidate numbers, the way of writing that games a loose checker.','写好几个候选数字的回答占多少；这种写法专钻宽松判分的空子。')},
   {k:'t',label:tx('Training step','训练步'),v:`${tt} / ${S.T}`,sub:tx('8 problems × 8 answers each','每步 8 道题 × 8 个回答'),tip:tx('Each step: 8 problems, 8 answers per problem, one gradient step.','每一步：8 道题，每道题 8 个回答，做一次梯度更新。')}];
}
const numOf=s=>parseFloat(String(s).replace(/[^0-9.\-]/g,''));
function drawMetrics(){
  const host=$('#rlvr-metrics',app), defs=metricsDef(), sgn=defs.map(m=>m.k).join();
  if(host._sig!==sgn){host._sig=sgn; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&m.k!=='t'&&!PL.on&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:0}:{})||'';}
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on');},2600);} else {d.classList.remove('on'); d.textContent='';}
    if(!PL.on) shownPrev[m.k]=m.v;});
}
/* H2 (the owner's message of 2026-10-09): Show the math reads like a page of a textbook. A few short blocks, one idea each: a lead line in
   words, the quantity's name and its formula, the same formula with the numbers of the group on the stage, the result. */
const M=t=>A.ui.math(t);
const mblk=(title,body)=>`<section class="rlvr-mb"><h4 class="rlvr-mh">${title}</h4>${body}</section>`;
const mline=(name,tex)=>`<div class="fx rlvr-mf">${name?`<span class="rlvr-mn">${name}</span>`:''}${M(tex)}</div>`;
/* rows of [words, formula]: the substituted lines, one per kind of answer, lined up */
const msub=rows=>`<div class="rlvr-ms">${rows.map(([w,tex])=>`<span class="rlvr-msw">${w}</span><span class="fx rlvr-msf">${M(tex)}</span>`).join('')}</div>`;
function mathHTML(){
  const g=groupNow(), r=R(), e=r[Math.min(tShow,r.length-1)], tt=Math.min(tShow,S.T), tg=grpStep();
  const fr=v=>f(v,v%1?1:0), m4=f(g.m,4), s4=f(g.sd,4), rl=g.resp.map(x=>fr(x.r)).join(', ');
  const vals=[...new Set(g.resp.map(x=>x.r))].sort((p,q)=>q-p), cnt=v=>g.resp.filter(x=>x.r===v).length, sum=g.resp.reduce((p,x)=>p+x.r,0);
  const lead=isP('checker')?(c=>tx(`<p>The answer in the picture gets the reward r = ${c.strict.pay} from the strict checker and r = ${c.loose.pay} from the loose one. GRPO turns rewards into advantages inside a group; the example here is the first group of ${G} answers the model wrote at step 0.</p>`,
      `<p>图里的回答，严格的判分程序给它的奖励 r = ${c.strict.pay}，宽松的给 r = ${c.loose.pay}。GRPO 在一组回答里把奖励变成优势。这里的例子，是模型在第 0 步写的第一组 ${G} 个回答。</p>`))(chkNow())
    :tx(`<p>The group of ${G} answers in the picture (step ${tg}) got the rewards ${rl}.</p>`,`<p>图里这一组 ${G} 个回答（第 ${tg} 步）的奖励是 ${rl}。</p>`);
  const b1=mblk(tx('1. The group’s mean and standard deviation','1. 这一组奖励的平均值和标准差'),
    tx('<p>First the average reward of the group, then how far the rewards spread around it.</p>','<p>先算这一组的平均奖励，再算奖励在平均值上下散开多少。</p>')+
    mline(tx('mean','平均值'),'\\text{mean}(r) = \\frac{1}{8} \\sum_j r_j')+msub([[tx('this group','这一组'),`\\text{mean}(r) = \\frac{${fr(+sum.toFixed(1))}}{8} = ${m4}`]])+
    mline(tx('standard deviation','标准差'),'\\text{std}(r) = \\sqrt{\\frac{\\sum_j (r_j - \\text{mean}(r))^2}{8 - 1}}')+
    msub([[tx('this group','这一组'),g.flat?`\\text{std}(r) = 0`:`\\text{std}(r) = \\sqrt{\\frac{${vals.map(v=>`${cnt(v)} \\cdot (${fr(v)} - ${m4})^2`).join(' + ')}}{7}} = ${s4}`]])+
    tx('<p>The sum is divided by 8 − 1 = 7, the sample standard deviation, as TRL computes it.</p>','<p>求和以后除以 8 − 1 = 7，也就是样本标准差，和 TRL 的算法一样。</p>'));
  const b2=mblk(tx('2. Each answer’s advantage','2. 每个回答的优势'),
    tx('<p>The advantage says how much better an answer did than the average of its group. Above the average it is positive, and the update makes that way of writing more likely; below the average it is negative.</p>','<p>优势说的是一个回答比同一组的平均水平好多少。高于平均，优势为正，更新时这种写法会变得更常用；低于平均，优势为负。</p>')+
    mline(tx('advantage','优势'),'A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}')+
    (g.flat?msub([[tx(`every answer (r = ${fr(vals[0])})`,`每个回答（r = ${fr(vals[0])}）`),`A_i = \\frac{${fr(vals[0])} - ${m4}}{0 + 0.0001} = 0`]])+tx('<p>All rewards are equal, so every advantage is 0 and this problem teaches nothing at this step.</p>','<p>奖励全都一样，每个优势都是 0，这道题在这一步什么也教不了。</p>')
      :msub(vals.map(v=>[tx(`answers with r = ${fr(v)} (${cnt(v)})`,`得 ${fr(v)} 分的回答（${cnt(v)} 个）`),`A_i = \\frac{${fr(v)} - ${m4}}{${s4} + 0.0001} = ${sg((v-g.m)/(g.sd+1e-4),2)}`]))+
       tx('<p>The 0.0001 only avoids dividing by 0 when every reward in the group is the same.</p>','<p>分母里的 0.0001，只是为了全组奖励都一样时不会除以 0。</p>')));
  const b3=mblk(tx('3. The loss of one step: the advantages change the model','3. 一步的 loss：用优势更新模型'),
    tx(`<p>The update pushes up the answers with a positive advantage, so their way of writing becomes more likely, and pushes down the ones with a negative advantage. The KL penalty pulls the model back towards the reference model, so it does not run too far.</p>`,
       `<p>更新时，优势为正的回答被推高，它的写法以后更容易被选中；优势为负的回答被压低。KL 惩罚把模型往参考模型拉回一点，免得它跑得太远。</p>`)+
    mline('loss','L = \\frac{1}{N} \\sum_i \\left( - A_i \\cdot \\rho_i + \\beta \\cdot \\text{KL}_i \\right)')+
    tx(`<p>N = ${B*G} is the number of answers in the step (${B} problems × ${G} answers), and β = ${BETA}. ${M('\\rho_i')} compares the model being trained with the model that wrote the batch:</p>`,`<p>N = ${B*G} 是这一步的回答总数（${B} 道题 × ${G} 个回答），β = ${BETA}。${M('\\rho_i')} 比较正在训练的模型和写出这批回答的模型：</p>`)+
    mline(tx('probability ratio','概率之比'),'\\rho_i = \\frac{\\pi_\\theta(y_i)}{\\pi_{\\text{old}}(y_i)}')+
    tx(`<p>Each batch is used for one update. During it the model being trained is still the model that wrote the batch, so every ${M('\\rho_i')} is exactly 1. GRPO’s clip, which keeps ${M('\\rho_i')} within ${M('[1 - \\varepsilon, 1 + \\varepsilon]')}, therefore never acts here and is left out of the loss.</p>`,
       `<p>每批回答只用来更新一次。更新时，正在训练的模型和写出这批回答的模型还是同一个，所以每个 ${M('\\rho_i')} 都等于 1。GRPO 的截断（clip）把 ${M('\\rho_i')} 限制在 ${M('[1 - \\varepsilon, 1 + \\varepsilon]')} 里，${M('\\rho_i')} 都是 1 时它不会生效，所以这里的 loss 省掉了它。</p>`));
  const b4=mblk(tx('4. The KL penalty','4. KL 惩罚'),
    tx('<p>The KL penalty measures how far the model has moved from the reference model, the model at the start of training. TRL estimates it for each answer with the k3 formula:</p>','<p>KL 惩罚衡量模型离参考模型（训练开始时的模型）有多远。TRL 对每个回答用 k3 公式估计它：</p>')+
    mline('KL','\\text{KL}_i = \\frac{\\pi_{\\text{ref}}(y_i)}{\\pi_\\theta(y_i)} - \\log\\frac{\\pi_{\\text{ref}}(y_i)}{\\pi_\\theta(y_i)} - 1')+
    tx(`<p>At step ${tt}, the exact KL of the policy from the reference model is ${f(e.kl,2)} nats (nats: the unit of a KL computed with the natural logarithm).</p>`,`<p>第 ${tt} 步，策略模型离参考模型的 KL 精确算出来是 ${f(e.kl,2)} nats（nats 是用自然对数算 KL 时的单位）。</p>`));
  const p=e.pk[1], k=8;
  const b5=mblk('5. pass@k',
    tx('<p>pass@k is the share of problems solved at least once in k tries. For one kind of problem, with p the chance of a right answer in one try:</p>','<p>pass@k 是给 k 次机会、至少答对一次的题占多少。一类题一次答对的概率是 p 时：</p>')+
    mline('','\\text{pass@}k = 1 - (1 - p)^k')+
    msub([[tx(`medium problems, step ${tt}, k = ${k}`,`中等题，第 ${tt} 步，k = ${k}`),`\\text{pass@}${k} = 1 - (1 - ${f(p,3)})^{${k}} = ${f(1-Math.pow(1-p,k),3)}`]])+
    tx('<p>The chart averages over the kinds of problems, weighted by their shares.</p>','<p>图里再按各类题占的比例取平均。</p>'));
  return lead+b1+b2+b3+b4+b5;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s.expect&&s.expect()) observed[step]=true;}
  const gr=$('#rlvr-graph',app); gr.classList.toggle('rlvr-runfirst',runOn()); gr.classList.toggle('rlvr-pkfirst',isP('passk')||isP('remember'));   // S1: the charts the card talks about come first in the stage's scroll box
  const ck=isP('checker'); $('#rlvr-grp',app).hidden=ck; $('#rlvr-metrics',app).hidden=ck;   // codex-b9v3 #1: part 2's stage is the picked answer and the two checkers only
  drawHead(); drawChk(); drawGroup(); drawRun(); drawPassK(); drawMetrics();
  const mb=$('#rlvr-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  const hk=mode+'|'+token(); if(hk!==lastHash){lastHash=hk; A.hash.write(token());}   // not on every frame of a replay: a write racing a link click would undo the navigation
  A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change the checker, the format reward and the problems with no signal in the task card. Fixed: the SFT model, the new problems, ${TCH} steps.`,`挑战中只能在任务卡里改判分程序、格式奖励和没有信号的题。固定不变的是：SFT 模型、新的一批题、${TCH} 步。`);
  if(mode!=='guided') return '';
  return tx('Fixed in this part: the settings the card names. You can change them in Scenarios.','这一部分的设置由卡片决定；在应用场景里可以改。');
}
function syncControls(){
  [['chk',S.chk],['fmt',String(S.fmt)],['dyn',String(S.dyn)],['start',S.start],['set',S.set]].forEach(([k,v])=>$$(`#rlvr-controls .seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,v)));
  const c=$('#rlvr-controls',app), lock=mode!=='sandbox';
  ['chk','fmt','dyn','start','set'].forEach(k=>{const e=$(`#rlvr-ctl-${k}-wrap`,app); if(e) e.hidden=lock;});
  const rs=$('#rlvr-ctl-reset-wrap',app); if(rs) rs.hidden=lock;
  c.classList.toggle('locked',lock);
  const ln=$('#rlvr-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); if(s.txt) A.announce(s.txt);},400);}

/* =====================================================================
   players (R3.9, R3.20): the run (whole run, default fast) and one group (mechanism replay, default medium)
   ===================================================================== */
function stopRun(){if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=0; PL.on=false;}
function stopGrp(){if(GP.raf) cancelAnimationFrame(GP.raf); GP.raf=0; GP.on=false;}
function runFrame(ts){
  PL.raf=0; if(!app||!PL.on) return;
  const dt=Math.min(0.25,(ts-(PL.last||ts))/1000); PL.last=ts; PL.t+=dt*RUN_SPD[PL.speed];
  const t=Math.min(S.T,Math.floor(PL.t));
  if(t!==tShow){tShow=t; GP.n=G; render();}
  if(PL.t>=S.T){PL.on=false; runDone(); return;}
  PL.raf=requestAnimationFrame(runFrame);
}
function runDone(){tShow=S.T; RUNSEEN.add(keyNow()); GRPSEEN.add(grpKey()); GP.n=G; wantObs=true; render(); wantObs=false; if(mode==='guided'&&STEPS[step].obs) revealSoon();}
function playRun(from0){stopGrp(); GP.n=G; if(from0||tShow>=S.T){PL.t=0; tShow=0;} else PL.t=tShow; PL.from=tShow;
  if(A.reduceMotion()){runDone(); return;}
  PL.on=true; PL.last=0; if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=requestAnimationFrame(runFrame); render();}
function pauseRun(){stopRun(); render();}
/* a new setup: R3.20. A run already watched shows its end at once; a new run plays from step 0 (only where the stage shows a run) */
function refreshRun(){
  if(!runOn()){stopRun(); tShow=0; return;}
  if(RUNSEEN.has(keyNow())){stopRun(); tShow=S.T; return;}
  if(mode==='guided'&&step===3&&!pred){stopRun(); tShow=0; return;}
  playRun(true);
}
/* part 1 shows the group without advantages (and its scores only after the prediction); part 3 first plays it with them (codex-b9v3 #1).
   So a play watched in part 1 does not count for part 3, and part 1 is not replayed once watched there or in part 3 (R3.20). */
const grpBase=()=>keyNow()+'|'+tShow+'|'+grpI;
const grpKey=()=>grpBase()+'|'+((mode==='guided'&&step===0)?'p1':'s');
function grpFrame(ts){
  GP.raf=0; if(!app||!GP.on) return;
  const dt=Math.min(0.25,(ts-(GP.last||ts))/1000); GP.last=ts; GP.acc+=dt;
  const per=GRP_SPD[GP.speed]; let moved=false;
  while(GP.acc>=per&&GP.n<G){GP.acc-=per; GP.n++; moved=true;}
  if(GP.n>=G){GP.on=false; GRPSEEN.add(grpKey());}
  if(moved||!GP.on){if(GP.n>0) sel=GP.n-1; render();}
  if(GP.on) GP.raf=requestAnimationFrame(grpFrame);
}
function playGrp(){stopGrp(); if(PL.on) return; GP.n=0; GP.acc=0; sel=0;
  if(A.reduceMotion()){GP.n=G; GRPSEEN.add(grpKey()); render(); return;}
  GP.on=true; GP.last=0; GP.raf=requestAnimationFrame(grpFrame); render();}
/* the group shown changed: play it only if it was never watched (R3.20) */
function refreshGrp(){
  if(PL.on) {GP.n=G; return;}
  if(isP('checker')){stopGrp(); GP.n=G; return;}   // part 2 does not show the group: nothing plays and nothing counts as watched, so part 3 plays it (codex-b9v3 #1)
  if(GRPSEEN.has(grpKey())||GRPSEEN.has(grpBase()+'|s')){stopGrp(); GP.n=G; return;}
  if(runOn()){stopGrp(); GP.n=G; GRPSEEN.add(grpKey()); return;}   // with the run on the stage, the group follows the chart
  playGrp();
}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p,keepGroup){const before=keyNow(); Object.assign(S,p); if(!SETS[S.set]) S.set='main'; S.fmt=S.fmt?1:0; S.dyn=S.dyn?1:0;
  if(keyNow()!==before){if(!keepGroup){grpI=0; sel=0;} refreshRun(); refreshGrp();} render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; stopRun(); S={...BASE,...s.enter()};
  grpI=0; sel=0;
  const st=$('#rlvr-stage',app); if(st) st.scrollTop=0;
  refreshRun(); refreshGrp(); renderRail();
}
function revealStep(){A.ui.reveal($('#rlvr-rail .step.current',app)||$('#rlvr-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2, R9.7: copied from llm-judge) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#rlvr-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#rlvr-rail .step.current',app)||$('#rlvr-rail .railcard',app)||$('#rlvr-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#rlvr-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#rlvr-insight',app)); else revealBox($('#rlvr-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; stopRun(); stopGrp();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH0};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  grpI=0; sel=0;
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  refreshRun(); refreshGrp(); renderRail();
}
const token=(s=S)=>[s.chk,s.fmt,s.dyn,s.start,s.set].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null; const [chk,fm,dy,start,set]=p;
  if(!['strict','loose'].includes(chk)||!['0','1'].includes(fm)||!['0','1'].includes(dy)||!PHI0[start]||!['main','hard'].includes(set)) return null;
  return {chk,fmt:+fm,dyn:+dy,start,set,T};}

/* interview drill: one group's advantages, seeded */
function makeDrill(seed){const rnd=mulberry(seed*7919+5), n=8, k=1+Math.floor(rnd()*6), rs=Array.from({length:n},(_,i)=>i<k?1:0);
  const m=k/n, sd=Math.sqrt(rs.reduce((a,r)=>a+(r-m)*(r-m),0)/(n-1)); return {seed,n,k,m,sd,ap:(1-m)/(sd+1e-4),an:(0-m)/(sd+1e-4)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#rlvr-drill-q',app), a=$('#rlvr-drill-a',app);
  const dfx=`<div class="fx">${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r) + 0.0001}')}</div><div class="fx">${A.ui.math('\\text{std}(r) = \\sqrt{\\frac{\\sum_j (r_j - \\text{mean}(r))^2}{n - 1}}')}</div>`;
  qd.innerHTML=tx(`<p>GRPO samples <b>${d.n}</b> answers to one math problem; a checker gives <b>${d.k}</b> of them 1 point and the rest 0. Use TRL’s formula, with the sample standard deviation:</p>${dfx}<p>What advantage does a right answer get, and a wrong one? What if all ${d.n} were right?</p>`,
                  `<p>GRPO 对一道数学题采样 <b>${d.n}</b> 个回答，判分程序给其中 <b>${d.k}</b> 个 1 分，其余 0 分。按 TRL 的公式算优势，标准差用样本标准差：</p>${dfx}<p>答对的回答优势是多少？答错的呢？如果 ${d.n} 个全对呢？</p>`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  /* H1: the worked answer typesets each step (the numbers of this drill in the formula's place) */
  const m3=f(d.m,3), s4=f(d.sd,4);
  const steps=msub([[tx('mean','平均值'),`\\text{mean}(r) = \\frac{${d.k}}{${d.n}} = ${m3}`],
    [tx('standard deviation','标准差'),`\\text{std}(r) = \\sqrt{\\frac{${d.k} \\cdot (1 - ${m3})^2 + ${d.n-d.k} \\cdot (0 - ${m3})^2}{${d.n} - 1}} = ${s4}`],
    [tx('a right answer','答对的回答'),`A = \\frac{1 - ${m3}}{${s4} + 0.0001} = ${f(d.ap,3)}`],
    [tx('a wrong answer','答错的回答'),`A = \\frac{0 - ${m3}}{${s4} + 0.0001} = ${sg(d.an,3)}`]]);
  a.innerHTML=tx(`<p><b>A right answer gets the advantage ${sg(d.ap,3)}, a wrong one ${sg(d.an,3)}.</b></p>`,`<p><b>答对的回答，优势是 ${sg(d.ap,3)}；答错的回答，优势是 ${sg(d.an,3)}。</b></p>`)+steps+
    tx(`<p>If all ${d.n} were right, every reward would equal the mean, so every advantage would be 0: the problem gives no signal.</p>`,`<p>如果 ${d.n} 个全对，每个奖励都等于平均值，每个优势都是 0，这道题给不出信号。</p>`);
}

/* =====================================================================
   tooltips and jumps
   ===================================================================== */
function jumpAct(k){
  if(k==='controls') return mode==='sandbox'?tx('Click to jump to Settings.','点击跳到“自由调参”。'):tx('Settings are open in Scenarios.','“自由调参”在应用场景里可以改。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  return '';
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#rlvr-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){if(mode!=='sandbox'){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;} const c=$('#rlvr-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){byLearner(()=>setState({...BASE})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval({chk:S.chk,dyn:S.dyn,fmt:S.fmt}).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='grp-next'){const st=R()[Math.min(tShow,S.T)]; grpI=(grpI+1)%st.groups.length; sel=0; byLearner(()=>{refreshGrp(); render();}); return;}
  if(a==='grp-play'){if(GP.on){stopGrp(); GP.n=G; render(); return;} playGrp(); return;}
  if(a==='run-play'){if(PL.on){pauseRun(); return;} if(mode==='guided'&&step===3&&!pred) return; playRun(tShow>=S.T); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#rlvr-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const selr=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,selr); else selr();}catch(err){selr();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function holdPlace(el,fn){
  if(!el||!el.isConnected){fn(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{fn();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#rlvr-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){if(isP('train')){stopRun(); RUNSEEN.add(keyNow()); tShow=S.T; GP.n=G; wantObs=true; byLearner(render); wantObs=false; revealSoon();} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#rlvr-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.row!=null){sel=+d.row; byLearner(render); const nb=$(`#rlvr-rows [data-row="${sel}"]`,app); if(nb) nb.focus({preventScroll:true}); return;}   // H7: the clicked row keeps the focus after the redraw
    if(d.lever!=null&&isP('checker')){ex2=+d.lever; ex2Tried.add(ex2); wantObs=true; byLearner(render); wantObs=false; if(innerWidth>760) revealSoon(); return;}
    if(d.lever!=null&&isP('loose')){lv5=d.lever; lv5T.add(lv5); wantObs=true; byLearner(()=>setState({chk:lv5})); wantObs=false; return;}
    if(d.lever!=null&&isP('format')){lv6=+d.lever; lv6T.add(lv6); wantObs=true; byLearner(()=>setState({fmt:lv6})); wantObs=false; return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='rlvr-gspd'){GP.speed=d.v; render(); return;}
      if(k==='rlvr-rspd'){PL.speed=d.v; render(); return;}
      const inCtl=!!b.closest('#rlvr-controls');
      const apply=p=>inCtl?holdPlace(b,()=>byLearner(()=>setState(p))):byLearner(()=>setState(p));
      if(k==='chk'||k==='chC') apply({chk:d.v});
      else if(k==='fmt'||k==='chF') apply({fmt:+d.v});
      else if(k==='dyn'||k==='chD') apply({dyn:+d.v});
      else if(k==='start') apply({start:d.v});
      else if(k==='set') apply({set:d.v});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.p})); requestAnimationFrame(()=>{if(app) revealBox($('#rlvr-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#rlvr-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  el.addEventListener('input',e=>{const t=e.target; if(t.id!=='rlvr-scrub') return; stopRun(); tShow=+t.value; GP.n=G; userAct=true; try{render();} finally{userAct=false;}});
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the example answer and the
   options tried, the group and the answer picked, the runs and groups already watched, the two speeds, the challenge's choices and
   result, the drill. The step a run shows and how far a group replay got are not kept: a run or group still playing is recorded as
   played to its end, and, as on coming back to the lab, a run already watched shows its end. */
const rsKeys=Object.keys(BASE), rsObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), rsPick=o=>Object.fromEntries(rsKeys.map(k=>[k,o[k]]));
const rsKey=v=>typeof v==='string'&&v.length<=60&&/^(strict|loose)_[01]_[01]_\w+_\w+_\d+(\|\d+\|\d+\|(p1|s))?$/.test(v), rsInt=(v,n)=>Number.isInteger(v)&&v>=0&&v<n;
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:rsPick(S),pred,observed:{...observed},saved:saved?rsPick(saved):null,ex2,ex2Tried:[...ex2Tried],lv5,lv5T:[...lv5T],lv6,lv6T:[...lv6T],grpI,sel:GP.on?G-1:sel,
    runSeen:[...RUNSEEN,...(PL.on&&!RUNSEEN.has(keyNow())?[keyNow()]:[])],grpSeen:[...new Set([...GRPSEEN,...(PL.on?[keyNow()+'|'+S.T+'|'+grpI+'|'+((mode==='guided'&&step===0)?'p1':'s')]:(GP.on?[grpKey()]:[]))])],speed:PL.speed,gspeed:GP.speed,chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&chEval({chk:S.chk,dyn:S.dyn,fmt:S.fmt}).ok};},
  restore(o){if(!rsObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[], inV=(L,v)=>L.some(x=>x.v===v);
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    const x=rsObj(o.S)?o.S:{};
    S=mode==='challenge'?{...CH0,chk:['strict','loose'].includes(x.chk)?x.chk:CH0.chk,fmt:x.fmt===1?1:0,dyn:x.dyn===1?1:0}:{...(parse(token(x))||BASE)};
    pred=PRED.some(q=>q.v===o.pred)?o.pred:null;
    observed={}; if(rsObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    saved=mode==='challenge'&&rsObj(o.saved)?parse(token(o.saved)):null;
    ex2=inV(EX2,o.ex2)?o.ex2:null; ex2Tried=new Set(arr(o.ex2Tried,v=>inV(EX2,v))); if(ex2!=null) ex2Tried.add(ex2);
    lv5=['strict','loose'].includes(o.lv5)?o.lv5:'strict'; lv5T=new Set(['strict',...arr(o.lv5T,v=>v==='strict'||v==='loose'),lv5]);
    lv6=o.lv6===1?1:0; lv6T=new Set([0,...arr(o.lv6T,v=>v===0||v===1),lv6]);
    grpI=rsInt(o.grpI,B)?o.grpI:0; sel=rsInt(o.sel,G)?o.sel:0;
    [[RUNSEEN,o.runSeen],[GRPSEEN,o.grpSeen]].forEach(([w,a])=>{w.clear(); arr(a,rsKey).slice(0,300).forEach(v=>w.add(v));});
    PL.speed=typeof o.speed==='string'&&Object.prototype.hasOwnProperty.call(RUN_SPD,o.speed)?o.speed:'f'; GP.speed=typeof o.gspeed==='string'&&Object.prototype.hasOwnProperty.call(GRP_SPD,o.gspeed)?o.gspeed:'m';
    PL.on=false; GP.on=false; tShow=0; GP.n=G;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el; lastHash='';
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    if(mode==='guided'&&!p) S={...BASE,...STEPS[step].enter()};
    el.innerHTML=viewHTML(); app=$('#rlvr-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#rlvr-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    fillRail();
    if(PL.on){PL.raf=requestAnimationFrame(runFrame);} else if(GP.on){GP.raf=requestAnimationFrame(grpFrame);} else {refreshRun(); refreshGrp();}
    render();
  },
  leave(){clearTimeout(annT); stageStop(); stopRun(); stopGrp(); root=null; app=null;},
  onLang(){if(root){const d=$('#rlvr-math',app), open=d&&d.open, ad=$('#rlvr-asm-d',app), aOpen=ad&&ad.open; const pl=PL.on, gp=GP.on; if(PL.raf) cancelAnimationFrame(PL.raf); if(GP.raf) cancelAnimationFrame(GP.raf); PL.raf=0; GP.raf=0; PL.on=pl; GP.on=gp; this.mount(root,null); if(open) $('#rlvr-math',app).open=true; if(aOpen) $('#rlvr-asm-d',app).open=true;}},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(runOn()?$('#rlvr-mix',app):$(isP('checker')?'#rlvr-chk':'#rlvr-grp',app)),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#rlvr-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), r=R(), e=r[Math.min(tShow,r.length-1)];
      status.className='status '+s.cls; status.textContent=s.short;
      /* part 2: the two scores of the picked answer (codex-b9v3 #1) */
      if(isP('checker')){const c=chkNow(); vis.innerHTML=''; read.innerHTML=esc(tx(`This answer: the strict checker pays ${ptsTxt(c.strict.pay)}, the loose checker ${ptsTxt(c.loose.pay)}`,`这个回答，严格的判分程序给 ${c.strict.pay} 分，宽松的给 ${c.loose.pay} 分`)); return;}
      if((mode==='guided'&&step===0&&!pred)||chHide()){vis.innerHTML=''; read.innerHTML=esc(chHide()?tx('Submit to see whether the goal is met','提交后判定是否达标'):tx('Predict in part 1 first','先在第 1 部分作答')); return;}
      vis.innerHTML=`<div class="rlvr-dk">${e.pi.map((p,i)=>`<i class="${COLS[i]}${i===3&&gameRed()&&p>0.5?' red':''}" style="width:${(p*100).toFixed(1)}%"></i>`).join('')}</div>`;
      read.innerHTML=esc(tx(`Step ${Math.min(tShow,S.T)}: right in one try ${pc(e.p1)}, average answer ${f(e.len,0)} tokens`,`第 ${Math.min(tShow,S.T)} 步：一次答对 ${pc(e.p1)}，平均每个回答 ${f(e.len,0)} 个 token`));}
  }
});

/* =====================================================================
   read-only hook for tests (and the teaser)
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const pubStep=x=>({t:x.t,pi:x.pi,pf:x.pf,p1:x.p1,rew:x.rew,len:x.len,kl:x.kl,pk:x.pk,tries:x.tries,flat:x.flat,groups:x.groups.map(g=>({kind:g.kind,drop:g.drop,flat:g.flat,m:g.m,sd:g.sd,resp:g.resp.map(r=>[r.s,r.f,r.okS?1:0,r.okL?1:0,r.r,r.a])}))});
window.__rlvr=Object.freeze({
  run:c=>deep(run({...BASE,...c}).map(pubStep)), passK:(c)=>{const r=run({...BASE,...c}), w=SETS[(c&&c.set)||'main']; return {before:passK(r[0].pk,w),after:passK(r[r.length-1].pk,w)};},
  chEval:o=>deep(chEval(o)), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)),
  /* part 2's three answers in the page's language: their text, what each checker reads and pays (codex-b9v3 #1) */
  ex2:()=>EX2.map(o=>{const c=readsOf(o.x,0); return deep({v:o.v,lines:c.lines,ref:c.ref,fin:c.fin,strictRead:c.strict.read,strictPay:c.strict.pay,looseRead:c.loose.read,loosePay:c.loose.pay,gamed:c.gamed,flags:{strict:exScore(o.x,'strict'),loose:exScore(o.x,'loose')}});}),
  consts:Object.freeze({LEN,QS:deep(QS),QL:deep(QL),SETS:deep(SETS),P0,PHI0:{...PHI0},G,B,BETA,EPS,LR,SEED,T,TCH,MAXTRY,FMT,KS,CH_GOAL,CH_G,STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:PRESETS.map(p=>p.id)}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});},
  state:()=>({...S,mode,step,pred,ex2,observed:{...observed},tShow,grpI,sel,playing:PL.on,grpPlaying:GP.on,gpN:GP.n,speed:PL.speed,gspeed:GP.speed,runSeen:[...RUNSEEN],chSubmitted,chReveal,stageBusy:stageT!==0}),
  teaser:chk=>{const r=run({...BASE,chk}), e=r[r.length-1]; return {pi:e.pi.slice(),p1:e.p1,p10:r[0].p1,len:e.len};}
});
})();
