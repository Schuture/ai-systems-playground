/* llm-judge (world eval): Can you trust an LLM as the judge?
   Fidelity: Simulated. 300 seeded prompt pairs; a model of the judge's two fixed preferences (position, length) and of people's
   noisy preferences decides every verdict. Real reference numbers (MT-Bench, AlpacaEval) are quoted with their sources.
   - Each pair i: a quality gap g = Q + Sq·zq (log-odds that a person prefers B) and a log length ratio ℓ = ln R + Sl·zl − c, where c
     makes the mean length ratio exactly R. A person picks B when u < σ(g) (two people: u1, u2), so people disagree too.
   - The judge's score for B: g + λ·ℓ + π·(+1 if B is shown first, −1 if second) + self + st·et + se·e; above +τ B, below −τ A, else a
     tie. et is the pair's own taste (the same in both orders), e the noise of one call. Lab assumption: the judge favours the answer
     shown first. Both orders: the two verdicts must agree, otherwise "inconsistent", counted as a tie (FastChat MT-Bench). Random
     order: one call per pair, B first when the pair's seeded coin ro < 0.5 (AlpacaEval randomises the order of the outputs).
   - Win rate = (wins + ties/2) / N (FastChat show_result.py, win_rate_adjusted). Length control (simplified AlpacaEval): a logistic
     fit of the verdicts (1, ½, 0) on ℓ, read at ℓ = 0. Agreement: pairs where the judge did not tie (people never tie here); the
     second person is counted on the same pairs, so both numbers leave out the same ties (compute_agreement.py, "without tie").
   - Random numbers: mulberry32 seeded with seed × 9973 + 11; seven stratified columns (each a seeded permutation of 300 quantiles),
     then 300 plain draws for the second person, then 300 plain draws for the random order's coin.
   Words (EN / ZH): LLM judge / LLM 裁判; small / large model / 小模型 / 大模型; win rate / 胜率; tie / 平局; inconsistent / 不一致;
   both orders / 两种顺序; random order / 随机顺序; take length out / 去掉长度的影响 (kept: 不去掉长度); agreement / 一致率; part / 部分. window.__ljg is a read-only hook for tests/llm-judge_numbers.py and tests/llm-judge_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='llm-judge';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const N=300, NL=100, GOAL=3;
const SN=Math.sqrt(3)/Math.PI, LQ=u=>Math.log(u/(1-u)), sig=x=>1/(1+Math.exp(-x));
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function perm(r,n){const a=[...Array(n).keys()]; for(let i=n-1;i>0;i--){const k=Math.floor(r()*(i+1)), t=a[i]; a[i]=a[k]; a[k]=t;} return a;}
const DATA=new Map();
function data(seed){
  if(DATA.has(seed)) return DATA.get(seed);
  const r=mulberry(seed*9973+11), P=[]; for(let j=0;j<7;j++) P.push(perm(r,N));
  const u2=[]; for(let i=0;i<N;i++) u2.push(r());
  const ro=[]; for(let i=0;i<N;i++) ro.push(r());
  const q=(p,i)=>(p[i]+0.5)/N, rows=[];
  for(let i=0;i<N;i++) rows.push({zq:LQ(q(P[0],i))*SN,zl:LQ(q(P[1],i))*SN,za:LQ(q(P[2],i))*SN,u1:q(P[3],i),eAB:LQ(q(P[4],i)),eBA:LQ(q(P[5],i)),et:LQ(q(P[6],i)),u2:u2[i],ro:ro[i]});
  DATA.set(seed,rows); return rows;
}
/* the shift that makes the mean length ratio exactly R: the zl column holds the 300 quantiles, whatever the order */
function lshift(Sl){let s=0; for(let k=0;k<N;k++) s+=Math.exp(Sl*LQ((k+0.5)/N)*SN); return Math.log(s/N);}
const JUDGES={small:{lam:1,pi:0.5,st:0.9,se:0.4,tau:0.4,cost:1},big:{lam:1,pi:0.5,st:0.45,se:0.2,tau:0.4,cost:5}};
/* the data sets: the mission's, four scenarios, the challenge's, and the new case's */
const SC={
 def:{seed:4,Q:0,Sq:2,R:1.4,Sl:0.4},
 chatty:{seed:4,Q:0,Sq:2,R:2.0,Sl:0.4},
 better:{seed:4,Q:0.6,Sq:2,R:1.0,Sl:0.4},
 self:{seed:4,Q:0,Sq:2,R:1.0,Sl:0.4,self:0.6},
 math:{seed:4,Q:0,Sq:2,R:1.4,Sl:0.4,math:0.6},
 ch:{seed:104,Q:0,Sq:2,R:1.8,Sl:0.4},
 nw:{seed:4,Q:-0.4,Sq:2,R:2.0,Sl:0.4}
};
/* the base of one data set: per pair the gap g, the log length ratio, people's verdicts (1 = B, 0 = A; the checker's 1, ½, 0 on math) */
const BASES=new Map();
function base(sc){
  if(BASES.has(sc)) return BASES.get(sc);
  const D=SC[sc], rows=data(D.seed), c=D.Sl?lshift(D.Sl):0;
  const pr=rows.map((x,i)=>{
    let g, h, h2;
    if(D.math){const cB=x.u1<D.math?1:0, cA=x.u2<D.math?1:0; g=3*(cB-cA); h=cB===cA?0.5:cB; h2=h;}
    else {g=D.Q+D.Sq*x.zq; h=x.u1<sig(g)?1:0; h2=x.u2<sig(g)?1:0;}
    return {i,g,l:Math.log(D.R)+D.Sl*x.zl-c,h,h2,x};});
  const H=pr.reduce((a,p)=>a+p.h,0)/N;
  const b={sc,D,pr,H}; BASES.set(sc,b); return b;
}
/* a fractional logistic fit s ~ σ(α + β·ℓ) by Newton's method; the length-controlled win rate is σ(α), the win rate at equal length */
function lcFit(v){
  let a=0, b=0;
  for(let it=0;it<60;it++){
    let ga=0, gb=0, haa=0, hab=0, hbb=0;
    for(const x of v){const p=sig(a+b*x.l), w=p*(1-p); ga+=x.s-p; gb+=(x.s-p)*x.l; haa+=w; hab+=w*x.l; hbb+=w*x.l*x.l;}
    const det=haa*hbb-hab*hab, da=(hbb*ga-hab*gb)/det, db=(-hab*ga+haa*gb)/det; a+=da; b+=db;
    if(Math.abs(da)+Math.abs(db)<1e-13) break;
  }
  return {a,b,lc:sig(a)};
}
const RUNS=new Map();
/* one judge on one data set in one order: 'B' (B shown first), 'A' (A shown first), 'R' (once, the order set by the pair's coin) or 'both' */
function run(sc,jid,order){
  const key=[sc,jid,order].join('|'); if(RUNS.has(key)) return RUNS.get(key);
  const B=base(sc), J=JUDGES[jid], self=B.D.self||0;
  const v=B.pr.map(p=>{const x=p.x;
    const vote=(o,e)=>{const s=p.g+J.lam*p.l+J.pi*o+self+J.st*x.et+J.se*e; return s>J.tau?1:(s<-J.tau?0:0.5);};
    const a=vote(1,x.eBA), b=vote(-1,x.eAB), bf=x.ro<0.5, s=order==='B'?a:(order==='A'?b:(order==='R'?(bf?a:b):(a===b?a:0.5)));
    return {i:p.i,s,a,b,bf,l:p.l,inc:order==='both'&&a!==b};});
  const W=v.filter(x=>x.s===1).length, T=v.filter(x=>x.s===0.5).length, Lo=v.filter(x=>x.s===0).length, inc=v.filter(x=>x.inc).length;
  const w=(W+T/2)/N, sd=Math.sqrt(v.reduce((s,x)=>s+(x.s-w)**2,0)/(N-1)), se=sd/Math.sqrt(N), fit=lcFit(v);
  /* agreement on the first 100 pairs, the ones people labelled: pairs where the judge did not tie; the second person on the same pairs */
  const lab=v.slice(0,NL), nt=lab.filter(x=>x.s!==0.5), ag=nt.filter(x=>x.s===B.pr[x.i].h).length, pag=nt.filter(x=>B.pr[x.i].h2===B.pr[x.i].h).length;
  const r={sc,jid,order,v,W,T,Lo,inc,w,sd,se,lo:w-1.96*se,hi:w+1.96*se,fit,lc:fit.lc,H:B.H,agN:nt.length,ag,agree:nt.length?ag/nt.length:0,pag,pagree:nt.length?pag/nt.length:0,cost:N*(order==='both'?2:1)*J.cost};
  RUNS.set(key,r); return r;
}
const val=(r,lc)=>lc?r.lc:r.w;
/* people with people: a second person on all 100 labelled pairs (the page compares on the judge's pairs: run().pagree) */
function pp(sc){const B=base(sc), lab=B.pr.slice(0,NL); return lab.filter(p=>p.h===p.h2).length/NL;}
/* the challenge: two judges; once with B first, once in random order, or both orders; with or without length control; |judge − people| ≤ 3 points */
const CHO=[];
['small','big'].forEach(j=>['B','R','both'].forEach(o=>[false,true].forEach(lc=>CHO.push({j,o,lc}))));
function chEval(j,o,lc){const r=run('ch',j,o), v=val(r,lc), gap=Math.abs(v-r.H)*100; return {j,o,lc,v,H:r.H,gap,ok:gap<=GOAL+1e-9,cost:r.cost};}
function chBest(){let b=null; CHO.forEach(c=>{const e=chEval(c.j,c.o,c.lc); if(e.ok&&(!b||e.cost<b.cost)) b=e;}); return b;}
const CH_START={judge:'small',order:'B',lc:false};

/* =====================================================================
   formatting
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fint=A.fmt.int;
const fP=x=>fixd(x*100,1)+'%';                 // metrics, one decimal (R6.10)
const fR=x=>fint(Math.round(x*100))+'%';       // prose, rounded (R6.20)
const fPt=d=>fixd(d,1);                        // a gap in points
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J_=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const DEF=(text,tip)=>A.ui.def(text,tip);
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
/* model names keep a no-break hyphen (R7.8) */
const GPT4='GPT‑4', MTB='MT‑Bench';
const SRC={
 mtb:'https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md',
 common:'https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py',
 gen:'https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_judgment.py',
 show:'https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py',
 agree:'https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/compute_agreement.py',
 alpaca:'https://github.com/tatsu-lab/alpaca_eval/blob/main/README.md',
 rating:'https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/rating_systems.py',
 zheng:'https://arxiv.org/abs/2306.05685',
 dubois:'https://arxiv.org/abs/2404.04475'
};

/* =====================================================================
   module state
   ===================================================================== */
const BASE={sc:'def',judge:'small',order:'B',lc:false};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let pairAt=0, ordTried=new Set(['B']), lcSeen=false, ckSel='once', ckTried=new Set(['once']), remView='order';
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, drillSeed=2, drillOpen=false, annT=null, userAct=false, wantObs=false;
const shownPrev={};
const R=()=>run(S.sc,S.judge,S.order);
/* the words shown for a pair (illustrative lengths, rounded to 10): A from its own column, B from the pair's length ratio */
const wordsOf=(sc,i)=>{const x=data(SC[sc].seed)[i], p=base(sc).pr[i], la=Math.round(150*Math.exp(0.3*x.za)/10)*10; return {la,lb:Math.round(la*Math.exp(p.l)/10)*10};};
/* the featured pair of a data set: the first pair whose verdict flips with the order and whose shown lengths are within 10% of the
   set's mean length ratio, so the first picture agrees with the story (B about 40% longer in the mission) */
const FEATS={};
function featOf(sc){
  if(FEATS[sc]!=null) return FEATS[sc];
  const r=run(sc,'small','both'), R=SC[sc].R, flip=x=>x.inc&&x.a===1;
  let k=r.v.findIndex((x,i)=>{if(!flip(x)) return false; const w=wordsOf(sc,i); return Math.abs(w.lb/w.la/R-1)<=0.1;});
  if(k<0) k=r.v.findIndex(flip);
  return FEATS[sc]=k<0?0:k;
}
const FEAT=featOf('def');

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 judge:()=>tx('An LLM judge (LLM-as-a-judge, also called an autorater) is a model prompted to compare answers from other models, or to grade them. Here it sees one question and two answers and says which is better, or that they are equally good.',
              'LLM 裁判（LLM-as-a-judge，也叫 autorater）：用提示词让一个模型去比较别的模型的回答，或者给这些回答打分。这里它看到一个问题和两个回答，说哪个更好，也可以说一样好。'),
 pair:()=>tx('Pairwise judging: the judge sees the same question answered by two versions and picks the better answer. AlpacaEval and MT‑Bench’s pairwise mode judge this way. The other common way is single-answer grading: the judge gives each answer a score out of 10 on its own, MT‑Bench’s default. It has no order to worry about. This lab is about pairwise judging.',
             '两两比较（pairwise）：裁判看到同一个问题的两个回答，选出更好的那个。AlpacaEval 和 MT‑Bench 的两两比较模式都这样判。另一种常见做法是单独打分：裁判给每个回答单独打一个 10 分制的分数，MT‑Bench 默认就用这种。单独打分没有先后顺序的问题。这个 Lab 讲的是两两比较。'),
 tie:()=>tx('A tie: the judge says the two answers are equally good. In the win rate, a tie counts as half a win for each side.','平局：裁判说两个回答一样好。算胜率时，平局给双方各记半次赢。'),
 pos:()=>tx('Position bias: a judge prefers the answer in one position, whatever it says. This lab assumes it prefers the answer shown first. Which side a real judge prefers, and how much, depends on the model and the prompt.',
            '位置偏好：不管回答写的是什么，裁判都偏向排在某个位置的那个。这个 Lab 假设它偏向排在前面的回答。真实的裁判偏向哪边、偏多少，因模型和提示词而异。'),
 len:()=>tx('Length bias: a judge prefers the longer answer even when it is no better.','长度偏好：就算长的回答并没有更好，裁判也偏向它。'),
 lc:()=>tx('Taking length out: fit the judge’s verdicts against how much longer B is, and read the win rate where the two answers are equally long. AlpacaEval’s length-controlled win rate rests on this idea; this lab uses a simpler fit.',
           '去掉长度的影响：按 B 比 A 长多少，给裁判的结论拟合一条曲线，读出两个回答一样长时的胜率。AlpacaEval 的长度控制胜率就是这个思路，这个 Lab 用的拟合更简单。'),
 agree:()=>tx('Agreement: the share of pairs on which the judge and the person (or two people) pick the same answer. Pairs the judge calls equally good are left out, for the judge and for the second person alike, so both are counted on the same pairs. MT‑Bench’s compute_agreement.py reports agreement both with ties and without; this lab uses the one without.',
              '一致率：裁判和人（或者两个人）选了同一个回答的比例。裁判说“一样好”的对不计入，第二个人也只在同样这些对上算，两边比的是同一批对。MT‑Bench 的 compute_agreement.py 两种都算，一种计入平局，一种不计。这里用的是不计平局的那种。'),
 ci:()=>tx('The 95% interval: the range the win rate would move within if you judged another 300 questions of the same kind. It is the win rate ± 1.96 standard errors.',
           '95% 区间：换一批同类的 300 个问题再判，胜率大致会落在这个范围里。它等于胜率 ± 1.96 个标准误差。'),
 alpaca:()=>tx('AlpacaEval: a public benchmark that uses an LLM judge to compare a model’s answers with a reference model’s answers, pair by pair.','AlpacaEval：一个公开评测。它用 LLM 裁判，把模型的回答和一个参考模型的回答两两比较。'),
 mtb:()=>tx('MT‑Bench: a public benchmark of 80 multi-turn, open-ended questions, judged by an LLM.','MT‑Bench：一个公开评测，有 80 道多轮的开放式问题，用 LLM 当裁判。'),
 arena:()=>tx('Chatbot Arena: people compare two models’ answers side by side on a website and vote; the votes rank the models.','Chatbot Arena：真人在网页上并排比较两个模型的回答并投票，再用这些投票给模型排名。')
};
/* terms whose tip carries a formula (kit .t, shown by termTip): log-odds and σ (R2.8, R6.29) */
const TD=(k,x)=>`<span class="t ljg-td" tabindex="0" data-ctl="${k}">${x}</span>`;
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="ljg-eqs${EQN(rows)?' ljg-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="ljg-eq-l">${l?A.ui.math(l):''}</span><span class="ljg-eq-e">${A.ui.math(rel||'=')}</span><span class="ljg-eq-r">${A.ui.math(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="ljg-mb"><p>${lead}</p>${body}${res?`<p class="ljg-mb-res">${res}</p>`:''}</div>`;
const LALB=()=>`${A.ui.math('L_A')}${tx(' and ','、')}${A.ui.math('L_B')}`;

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'60',t:{en:'About 60%: close to the judge',zh:'60% 左右：和裁判差不多'}},
  {v:'50',t:{en:'About 50%: the two versions are about even',zh:'50% 左右：两个版本差不多'}},
  {v:'40',t:{en:'About 40%: B is actually worse',zh:'40% 左右：B 其实更差'}}];
const PRED_RIGHT='50';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const SIX=id=>STEPS.findIndex(s=>s.id===id);
/* people's win rate stays hidden until part 4 shows it (R3.8) */
const peopleShown=()=>mode==='sandbox'||(mode==='challenge'&&chSubmitted)||(mode==='guided'&&(step>SIX('length')||(isP('length')&&lcSeen)||isP('remember')));
/* the challenge shows only the cost until the learner submits (R3.8) */
const chHide=()=>mode==='challenge'&&!chSubmitted;
const ORD={B:{en:'Once, B shown first',zh:'判一次，B 在前'},A:{en:'Once, A shown first',zh:'判一次，A 在前'},R:{en:'Once, in random order',zh:'判一次，顺序随机'},both:{en:'Both orders',zh:'两种顺序都判'}};
const ORDS={B:{en:'once, B first',zh:'判一次，B 在前'},A:{en:'once, A first',zh:'判一次，A 在前'},R:{en:'once, random order',zh:'判一次，顺序随机'},both:{en:'both orders',zh:'两种顺序都判'}};
const ORDD={B:{en:'the initial setup',zh:'初始设置'},A:{en:'only the order changes',zh:'只换了先后顺序'},R:{en:'a coin picks which answer comes first; the same cost',zh:'每对抛硬币决定谁在前，费用不变'},both:{en:'two calls per pair; if they disagree, a tie',zh:'每对判两次，两次不一致算平局'}};
const ORDK=['B','A','R','both'];
/* how many pairs the random order showed B first */
const nBfirst=sc=>data(SC[sc].seed).filter(x=>x.ro<0.5).length;
const ordRes=o=>tx(`B wins ${fR(run('def','small',o).w)}`,`B 赢 ${fR(run('def','small',o).w)}`);
const ordCtl=()=>`<div class="ljg-lv" id="ljg-ic-ord">`+A.ui.levers({aria:tx('How the judge judges, pick one','裁判怎样判，选一个'),value:S.order,
  options:ORDK.map(o=>({v:o,title:L(ORD[o]),desc:L(ORDD[o]),result:esc(ordTried.has(o)||o===S.order?ordRes(o):'')}))})+`</div>`;
const lcCtl=id=>A.ui.seg({id,key:'lc',label:tx('Length','长度'),options:[['0',tx('length left in','不去掉长度')],['1',tx('take length out','去掉长度的影响')]],value:S.lc?'1':'0',inline:true});
const CK={once:{en:'Small judge, once, B first',zh:'小模型，判一次，B 在前',j:'small',o:'B',d:{en:'The small judge, judged once with B first,',zh:'只判一次、B 在前的小模型裁判'}},
  both:{en:'Small judge, both orders',zh:'小模型，两种顺序都判',j:'small',o:'both',d:{en:'The small judge in both orders',zh:'两种顺序都判的小模型裁判'}},
  big:{en:'Large judge, both orders',zh:'大模型，两种顺序都判',j:'big',o:'both',d:{en:'The large judge in both orders',zh:'两种顺序都判的大模型裁判'}}};
const ckRes=k=>{const r=run('def',CK[k].j,CK[k].o); return tx(`agrees ${fR(r.agree)}`,`一致 ${fR(r.agree)}`);};
const ckCtl=()=>`<div class="ljg-lv" id="ljg-ic-ck">`+A.ui.levers({aria:tx('Which judge to check, pick one','检验哪个裁判，选一个'),value:ckSel,
  options:Object.keys(CK).map(k=>({v:k,title:L(CK[k]),desc:k==='big'?tx('5 times the cost per call','每次贵 5 倍'):(k==='both'?tx('two calls per pair; if they disagree, a tie','每对判两次，两次不一致算平局'):tx('the initial setup','初始设置')),result:esc(ckTried.has(k)||k===ckSel?ckRes(k):'')}))})+`</div>`;
const pairBtn=()=>`<div class="stepnav ljg-pairnav"><button type="button" class="btn ghost sm" data-act="pair-next" id="ljg-ic-pair">${tx('Next pair →','看下一对 →')}</button></div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},gate:()=>!!pred,
  lead:()=>tx(`The judge says the new version wins ${fR(run('def','small','B').w)}. What would people say?`,`裁判说新版赢了 ${fR(run('def','small','B').w)}，请人来评会是多少？`),
  enter:()=>({...BASE}),
  body:()=>{const r=run('def','small','B'), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You changed the system prompt. The new version B writes answers about 40% longer than the old version A. Whether they are better, you do not know yet.</p><p>Open-ended answers have no answer key, so you ask an ${DEF('LLM to judge',TIP.judge())}. For each of 300 questions it sees both answers and says which is better. It judges each pair once, with B shown first. B wins ${fR(r.w)} (the bar at the bottom of the chart).</p><p>If people compared the same 300 pairs, about how often would B win?</p>${ch}${pred?'<p class="muted small">Saved. Part 4 shows the answer.</p>':''}`,
             `<p>你改了 system prompt。新版 B 的回答平均比旧版 A 长 40%，内容有没有变好，还不知道。</p><p>开放式的回答没有标准答案，所以你请 ${DEF('LLM 当裁判',TIP.judge())}：在 300 个问题上，每次给它看同一个问题的两个回答，让它说哪个更好。每对只判一次，新版 B 放在前面。结果 B 赢了 ${fR(r.w)}（图最下面的横条）。</p><p>如果请人来比这 300 对，B 大概能赢多少？</p>${ch}${pred?'<p class="muted small">已记下，第 4 部分揭晓。</p>':''}`);}},
 {id:'how',title:{en:'How pairwise judging works',zh:'裁判怎样比，胜率怎样算'},
  lead:()=>tx('One verdict per pair, and a tie counts as half a win for each side.','每对给一个结论，平局给双方各记半次赢。'),
  enter:()=>{pairAt=FEAT; return {...BASE};},
  body:()=>tx(`<p>The judge sees one question and ${DEF('two answers',TIP.pair())}, and says which is better, or that they are ${DEF('equally good',TIP.tie())}. The pair at the top of the chart is one of the 300; the marks below are all 300 verdicts.</p><div class="fx">${A.ui.math('\\text{win rate} = \\frac{W + \\frac{T}{2}}{N}')}</div><p class="small">W: the number of pairs B wins. T: the number of ties. N: the number of pairs compared, here 300.</p>`,
              `<p>裁判每次看到一个问题和${DEF('两个回答',TIP.pair())}，说哪个更好，也可以说${DEF('一样好',TIP.tie())}。图上面是 300 对里的一对，下面的记号是全部 300 个结论。</p><div class="fx">${A.ui.math('\\text{胜率} = \\frac{W + \\frac{T}{2}}{N}')}</div><p class="small">W：B 赢的次数。T：平局的次数。N：一共比较的次数，这里是 300。</p>`)+pairBtn(),
  always:true,obs:()=>obsHow()},
 {id:'order',title:{en:'The order changes the verdict',zh:'换个顺序，结论就变'},
  lead:()=>tx('Swap which answer comes first, and the verdict can flip.','同一对回答，只换先后顺序，结论就可能反过来。'),
  enter:()=>{ordTried=new Set(['B']); pairAt=FEAT; return {...BASE};},expect:()=>ordTried.has('both'),showMe:()=>setOrder('both'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Swap which answer comes first, and the judge’s verdict can flip. This is called a ${DEF('position preference',TIP.pos())}. First let the judge judge again with A shown first. Then try the two common ways around it: a random order for each pair, or judging each pair in both orders.</p>`,
              `<p>同一对回答，只是换个先后顺序，裁判的结论就可能反过来，这叫${DEF('位置偏好',TIP.pos())}。先让裁判把 A 放在前面再判一遍。再试试两种常见的对策：每对随机安排先后顺序，或者两种顺序各判一次。</p>`)+ordCtl(),
  obs:()=>obsOrder()},
 {id:'length',title:{en:'Longer looks better',zh:'越长看着越好'},
  lead:()=>tx('Taking length out asks: if both answers were equally long, which would the judge pick?','去掉长度，问的是：如果两个回答一样长，裁判会选哪个？'),
  enter:()=>({...BASE,order:'both'}),expect:()=>lcSeen,showMe:{lc:true},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>People often prefer longer answers too, and some judges prefer them more (${DEF('length bias',TIP.len())}). On ${DEF('AlpacaEval',TIP.alpaca())}’s human labels, the rule “always pick the longer answer” agrees with the majority of people 62.2% of the time. One person agrees with the majority of the others only 65.7% of the time.</p><p>The people in this lab judge content only, not length (an assumption), so what you see is the judge’s own length preference.</p><p>Taking length out asks: if both answers were equally long, which would the judge pick? To answer it, see how the judge’s verdicts change with the length ratio, then read the win rate where both answers are equally long.</p><p>B’s answers are 40% longer on average. Take length out:</p>`,
              `<p>人也常偏爱长的回答，有些裁判偏得更多（${DEF('长度偏好',TIP.len())}）。在 ${DEF('AlpacaEval',TIP.alpaca())} 的人工标注上，“永远选更长的那个”这条规则，有 62.2% 的时候和多数人的选择一致。单独一个人和其他人多数意见的一致率，也只有 65.7%。</p><p>这个 Lab 里的人只看内容、不看长度（这是假设），这样你看到的就只是裁判自己的长度偏好。</p><p>去掉长度的影响，问的是：如果两个回答一样长，裁判会选哪个？做法是看裁判的结论怎样随长度比变化，再读出两个回答一样长时的胜率。</p><p>B 的回答平均长 40%。把长度的影响去掉试试：</p>`)+lcCtl('ljg-ic-lc'),
  obs:()=>obsLength()},
 {id:'check',title:{en:'Check the judge against people',zh:'先拿人工标注检验裁判'},
  lead:()=>tx('Before you use a judge, see how often it agrees with people, and how often people agree with each other.','用之前，先看裁判和人一致多少，再看人和人之间一致多少。'),
  enter:()=>{ckSel='once'; ckTried=new Set(['once']); return {...BASE};},expect:()=>ckTried.size>1,showMe:()=>setCk('both'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>In a real evaluation, nobody labels all 300 pairs. The usual check is to let a person label 100 of them and count how often the judge picks the same answer. This is called the ${DEF('agreement',TIP.agree())}. Then a second person labels the same 100, to see how often two people agree.</p>`,
              `<p>真实的评测里，没人去标全部 300 对。常见的做法是请人标其中 100 对，数一数裁判和人选得一样的有多少，这叫${DEF('一致率',TIP.agree())}。再请第二个人标同样的 100 对，看两个人之间的一致率。</p>`)+ckCtl(),
  obs:()=>obsCheck()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('A judge is cheap and fast, but it has two fixed preferences. Check it against people first; when you use it, cancel the order’s effect and take length out.','裁判便宜、快，但有两个固定的偏好。用之前先拿人工标注检验它。用的时候，先抵消顺序的影响，再去掉长度。'),
  enter:()=>{remView='order'; return {...BASE,order:'both'};},
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsHow(){
  const r=run('def','small','B');
  return tx(`<p><b>Of 300 pairs, B wins ${r.W}, ${r.T} are ties and A wins ${r.Lo}. By the formula, B’s win rate is ${fR(r.w)}.</b></p><p>A win rate on 300 questions is an estimate too. Judged on another 300 questions, it would land between about ${fR(r.lo)} and ${fR(r.hi)} (the ${DEF('95% interval',TIP.ci())}; the lab ${labLink('eval-ci')} explains it). Even the low end is above 50%, so the judge reliably prefers B. That does not show that B is really better.</p>`,
            `<p><b>300 对里，B 赢了 ${r.W} 对，平局 ${r.T} 对，A 赢了 ${r.Lo} 对。代进公式，B 的胜率是 ${fR(r.w)}。</b></p><p>300 个问题上的胜率也是估计。换一批问题再判，它大约会在 ${fR(r.lo)} 到 ${fR(r.hi)} 之间（${DEF('95% 区间',TIP.ci())}，详见${labLink('eval-ci')}这个 Lab）。区间的下限也高于 50%，可见裁判稳定地偏向 B。可这只说明裁判偏向 B，不说明 B 真的更好。</p>`);
}
function obsOrder(){
  const b=run('def','small','B'), a=run('def','small','A'), t=run('def','small','both'), rr=run('def','small','R'), nB=nBfirst('def');
  const more=ordTried.has('both')?'':tx(' Now judge in both orders.','再试试两种顺序都判。');
  const left=!ordTried.has('R')&&!ordTried.has('both')?tx(' Now try the two common ways around it.','再试试两种常见的对策。'):(!ordTried.has('R')?tx(' Now try a random order.','再试试随机顺序。'):more);
  if(S.order==='B') return tx(`<p><b>B shown first: B wins ${fR(b.w)}.</b> Pick another way to compare.</p>`,`<p><b>B 在前：B 赢 ${fR(b.w)}。</b>选另一种判法比一比。</p>`);
  if(S.order==='A') return tx(`<p><b>With A shown first, B’s win rate falls from ${fR(b.w)} to ${fR(a.w)}.</b></p><p>Not one word of the answers changed; only the order did.${left}</p>`,
                              `<p><b>只把 A 放到前面，B 的胜率就从 ${fR(b.w)} 降到 ${fR(a.w)}。</b></p><p>回答一个字没改，变的只是先后顺序。${left}</p>`);
  if(S.order==='R') return tx(`<p><b>Once per pair, in random order: B wins ${fR(rr.w)}.</b></p><p>Here B came first in ${nB} pairs and A in ${N-nB}. The position preference helps B in the pairs where B comes first and A in the others, so on average it cancels. It costs the same as judging once. But each single verdict still depends on the order, and you cannot see which pairs would flip. ${DEF('AlpacaEval',TIP.alpaca())} randomises the order by default.${more}</p>`,
                              `<p><b>每对只判一次、顺序随机，B 赢 ${fR(rr.w)}。</b></p><p>这 300 对里，B 在前的有 ${nB} 对，A 在前的有 ${N-nB} 对。位置偏好在 B 在前的对里帮 B，在 A 在前的对里帮 A，平均下来就抵消了。费用和判一次一样。可每一个结论仍然受顺序影响，你也看不出哪些对换个顺序就会变。${DEF('AlpacaEval',TIP.alpaca())} 默认就随机安排顺序。${more}</p>`);
  return tx(`<p><b>Judged once in each order, ${fR(t.inc/N)} of the pairs get two different verdicts, and each of them counts as a tie. Counted this way, B wins ${fR(t.w)}.</b></p><p>Why: the position preference pushes the two verdicts in opposite directions, so it cancels in every pair. The cost is two calls per pair, twice as much. In return you can count how many pairs flip with the order. ${DEF(MTB,TIP.mtb())} does the same: two different winners make the pair “inconsistent”, counted as a tie.</p><p>Yet B still wins ${fR(t.w)}. Is B really better, or does the judge have another preference?</p>`,
            `<p><b>两种顺序各判一次，有 ${fR(t.inc/N)} 的回答对两次结论不一样，这些都按平局算。这样算下来，B 赢 ${fR(t.w)}。</b></p><p>原因是位置偏好在两次里方向相反，在每一对里都正好抵消。代价是每对要判两次，费用翻倍。好处是能数出有多少对一换顺序，结论就变。${DEF(MTB,TIP.mtb())} 也这样做：两次选出的赢家不同，就记为“不一致”，按平局算。</p><p>可是 B 还是赢 ${fR(t.w)}。这是 B 真的更好，还是裁判还有别的偏好？</p>`);
}
function obsLength(){
  const t=run('def','small','both');
  if(!S.lc) return tx(`<p><b>Judged in both orders, with length left in, B wins ${fR(t.w)}.</b> Take length out, then check your prediction.</p>`,`<p><b>两种顺序都判、还没去掉长度时，B 赢 ${fR(t.w)}。</b>去掉长度的影响，再核对你的预测。</p>`);
  return tx(`<p><b>With length taken out, B wins ${fR(t.lc)}. People would pick B ${fR(t.H)} of the time (the line in the chart).</b></p>`,`<p><b>去掉长度之后，B 赢 ${fR(t.lc)}。请人来评，B 赢 ${fR(t.H)}（图里的竖线）。</b></p>`)+verdict()+
    tx(`<p>Why: part of each verdict only follows the length. The fit finds how much, and reads the win rate where both answers are equally long. B is longer, not better.</p>`,
       `<p>原因：裁判的结论里，有一部分只是跟着长度走。拟合找出这部分有多大，再读出两个回答一样长时的胜率。B 只是更长，并没有更好。</p>`);
}
function obsCheck(){
  const k=ckSel, r=run('def',CK[k].j,CK[k].o), o=run('def','small','B');
  const head=tx(`<p><b>${esc(L(CK[k].d))} agrees with the person on ${fR(r.agree)} of the ${r.agN} pairs it decided. On the same ${r.agN} pairs, a second person agrees with the first on ${fR(r.pagree)}.</b></p>`,
                `<p><b>${esc(L(CK[k].d))}，在它选了一边的 ${r.agN} 对里，和人选得一样的有 ${fR(r.agree)}。在同样这 ${r.agN} 对上，第二个人和第一个人选得一样的有 ${fR(r.pagree)}。</b></p>`);
  const ties=tx(`<p>On the other ${NL-r.agN} pairs the judge said “equally good”; they are left out for both. A judge that often calls a tie keeps the easier pairs, so its agreement looks higher: when you read an agreement, also look at how many pairs it decided.</p>`,
                `<p>另外 ${NL-r.agN} 对，裁判说“一样好”，两边都不计入。常判平局的裁判，留下的多是好判的对，一致率会显得高些。看一致率时，也要看它判了多少对。</p>`);
  if(k==='once') return head+ties+tx('<p>Pick another judge to compare.</p>','<p>再选一个裁判比一比。</p>');
  return head+ties+tx(`<p>People disagree with each other too, so “about as often as a second person” is the usual bar. ${cite(SRC.mtb,MTB)} uses the same bar: ${GPT4} as a judge agreed with people over 80% of the time, as often as people agreed with each other. Its questions and its counting differ from this lab’s, so the two numbers cannot be compared directly.</p><p>But high agreement does not mean no preference. The judge judged once agreed with the person on ${fR(o.agree)}, more than the second person did, yet its win rate was ${fPt(Math.abs(o.w-o.H)*100)} points above people’s. Agreement only counts how many pairs the judge got right. It cannot show whether the ones it got wrong all lean toward B, while a preference adds up pair by pair in the win rate.</p>`+
       `<div class="ljg-prac"><div class="eyebrow">In practice</div><ul><li>Write the criteria down as a rubric in the judge’s prompt.</li><li>Use a judge from another model family than the one being judged.</li><li>When the answer can be checked (math, code with tests), check it instead of judging it.</li></ul></div>`,
       `<p>人和人之间本来就有分歧，所以“和多请一个人差不多”就算合格。${cite(SRC.mtb,MTB)} 也用这个标准：${GPT4} 当裁判，和人的一致率超过 80%，和人与人之间一样高。它的题目和算法都和这里不同，这两个数不能直接比。</p><p>但一致率高，不等于没有偏好。判一次的裁判和人一致 ${fR(o.agree)}，比第二个人还高，可它算出的胜率比人评高了 ${fPt(Math.abs(o.w-o.H)*100)} 个百分点。一致率只数选对了几对，看不出选错的那些是不是都偏向 B。偏好却会在胜率里一对一对地累加起来。</p>`+
       `<div class="ljg-prac"><div class="eyebrow">业界常见做法</div><ul><li>把评分标准（rubric）写进裁判的提示词。</li><li>裁判和被评的模型不要来自同一个模型家族。</li><li>答案能核对的（数学、带测试的代码），直接核对，不用裁判。</li></ul></div>`);
}
/* part 6: three lines, each with its part's example; a button puts that part's picture back in the chart (R6.23) */
const REMS=['order','length','check'];
function remBtn(k){const on=remView===k;
  const tip={order:tx('Shows the verdicts in both orders.','在图里显示两种顺序都判的结论。'),length:tx('Shows the win rates with length taken out, next to people’s.','在图里显示去掉长度后的胜率，和人评的胜率。'),check:tx('Shows the judges’ agreement with people.','在图里显示裁判和人的一致率。')}[k];
  return ` <button type="button" class="linkbtn ljg-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const b=run('def','small','B'), a=run('def','small','A'), t=run('def','small','both');
  return tx(`<ul class="ljg-remlist"><li><b>A judge prefers a position.</b> Swap which answer comes first, and the win rate moves, though not a word of the answers has changed. A random order cancels it on average, at no extra cost. Both orders cancel it in every pair, at twice the cost.${remBtn('order')}</li>`+
    `<li><b>A judge prefers longer answers.</b> When one version writes longer answers, the judge gives it a higher win rate than people do. Taking length out brings the win rate close to people’s.${remBtn('length')}</li>`+
    `<li><b>Check a judge against people before you use it.</b> Have people label a small set, and compare the judge’s agreement with people to how often two people agree. A judge that agrees about as often as a second person is good enough.${remBtn('check')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “The judge says B wins ${fR(b.w)}, so users will prefer B ${fR(b.w)} of the time.” The judge’s ${fR(b.w)} carried its two preferences; people picked B ${fR(b.H)} of the time.</p><p>✗ “It agrees with people 80% of the time, so it has no bias.” High agreement does not mean no preference. Agreement only counts how many pairs the judge gets right. A judge can agree with people often and still push the win rate several points to one side.</p><p>✗ “The longer and more detailed, the better.” Longer is not the same as better. Here B’s content is as good as A’s, only longer. People, who judge content only, picked B ${fR(b.H)} of the time.</p></div>${A.shell.toScenarios()}`,
    `<ul class="ljg-remlist"><li><b>裁判有位置偏好。</b>回答一个字没改，只换先后顺序，胜率就会跟着变。随机安排顺序，平均下来能抵消它，费用不变。两种顺序都判，每一对里都能抵消，费用翻倍。${remBtn('order')}</li>`+
    `<li><b>裁判偏爱更长的回答。</b>一个版本的回答更长，裁判给它的胜率就会比人评高。去掉长度以后，胜率就接近人评了。${remBtn('length')}</li>`+
    `<li><b>用裁判之前，先拿人工标注检验。</b>请人标一小批，看裁判和人一致多少，再看两个人之间一致多少。裁判和人的一致率跟多请一个人差不多，就算合格。${remBtn('check')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “裁判说 B 赢 ${fR(b.w)}，用户就会有 ${fR(b.w)} 的时候更喜欢 B。”裁判的 ${fR(b.w)} 里带着它的两个偏好。请人来评，B 只赢 ${fR(b.H)}。</p><p>✗ “和人一致 80%，说明裁判没有偏见。”一致率高，不等于没有偏好。一致率只数裁判选对了几对。一个裁判可以常和人一致，却还是把胜率往一边推高好几个百分点。</p><p>✗ “回答越长越详细，就越好。”长和好是两回事。这里 B 的内容和 A 一样好，只是更长。人评只看内容，B 只赢 ${fR(b.H)}。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'chatty',t:{en:'A chattier version',zh:'更啰嗦的版本'},d:{en:'Answers twice as long, just as good',zh:'回答长一倍，内容一样好'}},
 {id:'better',t:{en:'A truly better version',zh:'真的更好的版本'},d:{en:'Better answers, the same length',zh:'内容更好，长度一样'}},
 {id:'self',t:{en:'B from the judge’s own family',zh:'B 和裁判同一家'},d:{en:'B comes from the same model family as the judge',zh:'B 和裁判来自同一个模型家族'}},
 {id:'math',t:{en:'Math with known answers',zh:'有标准答案的数学题'},d:{en:'Each answer can be checked: right or wrong',zh:'每个答案对不对，可以直接核对'}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.id===S.sc);
function scenHTML(i){
  const p=PRESETS[i], j=S.judge, b=run(p.id,j,'B'), t=run(p.id,j,'both'), H=t.H, jn=j==='big'?tx('the large judge','大模型'):tx('the small judge','小模型');
  if(p.id==='chatty') return tx(`<p><b>A version that only talks more: answers twice as long, just as good. People pick B only ${fR(H)} of the time, yet ${jn} judged once gives ${fR(b.w)}.</b></p><p>Fixing the order is not enough; only taking length out as well brings the judge near people. The longer B gets, the more the length preference adds. The order fixes only cancel the position preference; they do not touch the length preference.</p>`,
                                `<p><b>新版只是更啰嗦：回答长一倍，内容一样好。请人来评，B 只赢 ${fR(H)}，裁判用${jn}判一次却给了 ${fR(b.w)}。</b></p><p>只处理顺序还不够，要再去掉长度，裁判才接近人评。B 越长，长度偏好带来的加分就越多。处理顺序只能抵消位置偏好，碰不到长度偏好。</p>`);
  if(p.id==='better') return tx(`<p><b>A version that is really better, at the same length. People pick B ${fR(H)} of the time; ${jn} in both orders gives ${fR(t.w)}.</b></p><p>With equal lengths the length preference has nothing to push, and both orders cancel the position preference. Taking length out does not remove a real improvement: with length taken out, the judge still gives ${fR(t.lc)}.</p>`,
                                `<p><b>新版真的更好，长度一样。请人来评，B 赢 ${fR(H)}。裁判用${jn}两种顺序都判，给 ${fR(t.w)}。</b></p><p>长度一样时，长度偏好使不上劲。两种顺序都判，位置偏好也抵消了。去掉长度，也不会把真正的进步一起去掉，去掉之后裁判仍给 ${fR(t.lc)}。</p>`);
  if(p.id==='self') return tx(`<p><b>B comes from the judge’s own model family, and is no better. People pick B ${fR(H)} of the time; ${jn} in both orders still gives ${fR(t.w)}.</b></p><p>A judge favouring answers from its own family is reported in the ${MTB} paper (${cite(SRC.zheng,'Zheng et al., 2023, arXiv 2306.05685')}); how strong it is here is this lab’s assumption. Neither fix touches it, and the result looks just like “truly better”. Only comparing the judge’s win rate with people’s on labelled pairs, or a judge from another family, finds it.</p>`,
                              `<p><b>B 和裁判来自同一个模型家族，内容并没有更好。请人来评，B 赢 ${fR(H)}。裁判用${jn}两种顺序都判，仍给 ${fR(t.w)}。</b></p><p>${MTB} 的论文报告过裁判偏爱自家回答的现象（${cite(SRC.zheng,'Zheng 等，2023，arXiv 2306.05685')}）。这里偏好有多大，是这个 Lab 的假设。处理顺序、去掉长度都碰不到它，结果看上去和“真的更好”一模一样。只有在人工标注上比一比裁判的胜率和人评的胜率，或者换一个别家的裁判，才发现得了。</p>`);
  return tx(`<p><b>Math answers can be checked. Checking every answer, B wins ${fR(H)}; ${jn} judged once gives ${fR(b.w)}.</b></p><p>When an answer can be checked (math, or code with tests), check it and skip the judge: a checker has no position or length preference, and it is cheaper. Here a tie means both answers were right, or both wrong.</p>`,
            `<p><b>数学题的答案可以核对。逐题核对，B 赢 ${fR(H)}。裁判用${jn}判一次，却给 ${fR(b.w)}。</b></p><p>答案能核对时（数学、带测试的代码），就核对，不用裁判。核对没有位置偏好，也不看长度，还更便宜。这里的平局是两个答案都对，或者都错。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const jName=j=>j==='big'?tx('large judge','大模型'):tx('small judge','小模型');
const oName=o=>o==='both'?tx('both orders','两种顺序都判'):(o==='R'?tx('once, random order','判一次，顺序随机'):tx('once, B first','判一次，B 在前'));
const lName=lc=>lc?tx('length taken out','去掉长度'):tx('length left in','不去掉长度');
const chLabel=e=>tx(`${jName(e.j)}, ${oName(e.o)}, ${lName(e.lc)}`,`${jName(e.j)}，${oName(e.o)}，${lName(e.lc)}`);
const costT=c=>tx(`cost ${fint(c)}`,`费用 ${fint(c)}`);
/* a setup said as a sentence part, for the feedback (the chart rows keep the short label) */
const chPhrase=e=>tx(`the ${jName(e.j)}, ${e.o==='both'?'in both orders':(e.o==='R'?'once per pair in random order':'once per pair with B first')}, ${e.lc?'with length taken out':'with length left in'}`,
  `用${jName(e.j)}，${e.o==='both'?'两种顺序都判':(e.o==='R'?'每对判一次、顺序随机':'每对判一次、B 在前')}，${e.lc?'再去掉长度':'不去掉长度'}`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.judge,S.order,S.lc), best=chBest(), nOk=CHO.filter(c=>chEval(c.j,c.o,c.lc).ok).length;
  let h;
  const good=tx(`<p class="okline ljg-mt"><b>Goal met: the judge gives B ${fP(me.v)} and people ${fP(me.H)}, ${fPt(me.gap)} points apart.</b>`,`<p class="okline ljg-mt"><b>达成目标：裁判给 B ${fP(me.v)}，人评是 ${fP(me.H)}，相差 ${fPt(me.gap)} 个百分点。</b>`);
  if(me.ok&&me.cost===best.cost) h=good+tx(` Of these 12 setups, ${nOk} meet the goal, and this one is the cheapest. The small judge is enough. A random order cancels the position preference on average, at no extra cost, and taking length out removes the length preference.</p><p>The price: each single verdict still depends on the order, and you cannot see which pairs flip. To cancel the preference in every pair, judge in both orders, at twice the cost.</p>`,
                                           `这 12 种选法里有 ${nOk} 种能达标，它最便宜。小模型就够用。随机安排顺序，平均下来抵消了位置偏好，不多花钱。去掉长度，又抵消了长度偏好。</p><p>代价是每一个结论仍然受顺序影响，也看不出哪些对一换顺序就会变。要在每一对里都抵消位置偏好，就得两种顺序都判，费用翻倍。</p>`);
  else if(me.ok){
    const why=(me.j==='big'?tx(' The large judge has half the random error, but its two preferences are just as large; removing them does not take a larger judge.','大模型的随机误差小一半，可它的两个偏好一样大，去掉偏好靠的不是换大模型。'):'')+
      (me.o==='both'?tx(' Both orders cancel the position preference in every pair and show the inconsistent pairs, but each pair costs two calls. Over 300 pairs, a random order cancels it on average with one call per pair.','两种顺序都判，每一对里都抵消了位置偏好，也能数出不一致的对，可每对要判两次。在 300 对上，随机安排顺序平均下来也能抵消它，每对只判一次。'):'');
    h=good+tx(' But a cheaper setup also meets it.</p>','不过有更便宜的方案也能达标。</p>')+tx(`<p>The cheapest uses ${chPhrase(best)}. It costs ${fint(best.cost)} instead of your ${fint(me.cost)}.${why}</p>`,`<p>最便宜的方案是${chPhrase(best)}。它的费用只要 ${fint(best.cost)}，你的方案要 ${fint(me.cost)}。${why}</p>`);}
  else h=tx(`<p class="badline ljg-mt"><b>Not yet: the judge gives B ${fP(me.v)} and people ${fP(me.H)}, ${fPt(me.gap)} points apart, more than 3.</b></p>`,`<p class="badline ljg-mt"><b>还没达到：裁判给 B ${fP(me.v)}，人评是 ${fP(me.H)}，相差 ${fPt(me.gap)} 个百分点，超过 3。</b></p>`)+
      tx('<p class="small">Hint: B’s answers are much longer, so the length preference adds more. The position preference can be cancelled by a random order, or by judging both orders. The large judge has half the random error, but its two preferences are just as large.</p>','<p class="small">提示：B 的回答长得多，长度偏好带来的加分也多。位置偏好可以靠随机安排顺序来抵消，也可以靠两种顺序都判。大模型的随机误差小一半，但两个偏好一样大。</p>')
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the owner's review of batch 8 (2026-10-07): the chart keeps the setups that meet the goal, the pick and the initial setup (brief) */
  const rows=CHO.map(c=>chEval(c.j,c.o,c.lc)), isMe=e=>e.j===S.judge&&e.o===S.order&&e.lc===S.lc, isStart=e=>e.j===CH_START.judge&&e.o===CH_START.order&&e.lc===CH_START.lc;
  h+=A.ui.cmpChart({title:tx('How far the judge’s win rate for B is from people’s on the 300 new pairs, in points (goal: 3 or less), with the cost in small-judge calls','300 对新回答上，裁判给 B 的胜率和人评相差几个百分点（目标：不超过 3），以及费用（相当于多少次小模型调用）'),
    aria:tx('Gap from people: ','和人评的差距：')+rows.filter(e=>e.ok||isMe(e)||isStart(e)).map(e=>`${chLabel(e)}${tx(': ','：')}${fPt(e.gap)}${tx(' points, ',' 个百分点，')}${costT(e.cost)}${e.ok?tx(', meets the goal','，达标'):''}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:`${e.ok?'✓ ':''}${chLabel(e)}${tx(` (${costT(e.cost)})`,`（${costT(e.cost)}）`)}`,value:e.gap,text:tx(`${fPt(e.gap)} pts`,`${fPt(e.gap)} 个百分点`),ok:e.ok,me:e.j===S.judge&&e.o===S.order&&e.lc===S.lc,
      tags:(e.j===CH_START.judge&&e.o===CH_START.order&&e.lc===CH_START.lc?[{text:tx('initial setup','初始设置')}]:[]).concat(e.j===S.judge&&e.o===S.order&&e.lc===S.lc?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,goal:GOAL,goalLabel:tx('goal: 3 points','目标：3 个百分点'),max:20,okText:tx('✓ within 3 points','✓ 相差不超过 3'),noText:tx('more than 3 points off','相差超过 3'),goalText:tx('goal: 3 points','目标：3 个百分点')});
  const s0=chEval('small','B',false), sR=chEval('small','R',false), sB=chEval('small','both',true);
  /* the random order alone misses the goal and is not in the chart: say so in words, its gap in a hover */
  h+=tx(`<p class="small">In the initial setup the small judge, once with B first, gives B ${fP(s0.v)}: ${fPt(s0.gap)} points above people. A random order alone still leaves the gap ${DEF(`above ${fint(GOAL)} points`,`The small judge, once per pair in random order, with length left in, is ${fPt(sR.gap)} points from people.`)}. Taking length out as well brings it to ${fPt(best.gap)}. Both orders with length taken out come even closer, ${fPt(sB.gap)} points apart, at twice the cost.</p><p class="small">Why the large judge does not help on its own: its random error is half the small judge’s, but in this lab its two preferences are the same size, and preferences push every pair the same way.</p>`,
        `<p class="small">初始设置是小模型、B 在前判一次，给 B ${fP(s0.v)}，比人评高 ${fPt(s0.gap)} 个百分点。只改成顺序随机，差距还是${DEF(`超过 ${fint(GOAL)} 个百分点`,`小模型每对判一次、顺序随机、不去掉长度时，和人评差 ${fPt(sR.gap)} 个百分点。`)}。再去掉长度，只差 ${fPt(best.gap)} 个百分点。两种顺序都判再去掉长度，只差 ${fPt(sB.gap)} 个百分点，但费用翻倍。</p><p class="small">大模型单靠自己为什么不行：它的随机误差只有小模型的一半，可这个 Lab 里它的两个偏好一样大，而偏好每一对都往同一个方向推。</p>`);
  h+=tx('<p class="small muted">Note: here every one of the 300 pairs has people’s verdict, so people’s win rate is known. In a real evaluation you only have a small labelled set, so check the judge on it first. A real large judge’s preferences also differ in size from a small one’s: measure them on your own labels.</p>',
        '<p class="small muted">注意：这里 300 对都有人评，所以人评的胜率是已知的。真实的评测里，你只有一小批人工标注，所以要先用它检验裁判。真实的大模型和小模型，偏好大小也不一样，要用你自己的人工标注去测。</p>');
  return h;
}

/* =====================================================================
   header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Guess what people would say, then see the judge’s two preferences (for the answer shown first, and for the longer answer), and finally check the judge against people. Start here if this is new to you.',
            '一共 6 个部分。先猜一猜人会怎么评，再看裁判的两个偏好（偏向排在前面的回答、偏向更长的回答），最后拿人工标注检验裁判。第一次学从这里开始。'),
  sandbox:tx('Four real cases, from a chattier version to math with known answers. Pick one to see how far the judge is from people.',
             '四个真实情况，从更啰嗦的版本到有标准答案的数学题。选一个，看裁判和人差多少。'),
  challenge:tx('New answers, 80% longer. Pick the judge and how it judges so that its win rate is within 3 points of people’s, at the lowest cost.',
               '一组新的回答，B 平均长 80%。选裁判和判法，让裁判的胜率和人评相差不超过 3 个百分点，而且最便宜。')
});
const SCORE='s = g + \\ln\\frac{L_B}{L_A} + 0.5\\,o + \\varepsilon_1 + \\varepsilon_2';
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser really runs a seeded simulation. From preferences set for this lab, it gives both the judge’s verdicts and people’s. Every number on the page comes from it; ${J_('math','“Show the math”')} works them out. The real reference numbers come from ${DEF(MTB,TIP.mtb())} and ${DEF('AlpacaEval',TIP.alpaca())}, with their sources.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Each of 300 questions has answers A and B. Their quality gap g is a ${TD('logodds','log-odds')}; it sets the chance a person picks B:<div class="fx">${A.ui.math('P(\\text{person picks B}) = \\sigma(g)')}</div>${TD('sigma','σ')} turns log-odds back into a probability. People pick with chance, so two people disagree too.</li>`+
 `<li>The judge gives B a score:<div class="fx">${A.ui.math(SCORE)}</div>${LALB()} are the two answers’ lengths. o is +1 when B is shown first and −1 when it is second. ${A.ui.math('\\varepsilon_1')} belongs to the pair and is the same in both orders; ${A.ui.math('\\varepsilon_2')} is new on every call. Above 0.4 the judge picks B, below −0.4 it picks A, and in between it says “equally good”.</li>`+
 `<li>Both orders: a verdict counts only if the two agree, otherwise a tie (${MTB}). Random order: one call per pair, a coin picks which answer comes first (AlpacaEval). The win rate is ${MTB}’s <code>win_rate_adjusted</code>:<div class="fx">${A.ui.math('\\text{win rate} = \\frac{W + \\frac{T}{2}}{N}')}</div></li>`+
 `<li>Taking length out: a logistic regression of the verdicts (1, ½, 0) on the log of the length ratio, read where both answers are equally long; a simpler version of AlpacaEval’s length control.</li>`+
 `<li>Agreement leaves ties out, like the “without tie” count of <code>compute_agreement.py</code>. To keep the comparison fair, the second person is counted only on the same pairs where the judge picked a side.</li>`+
 `<li>Random numbers: mulberry32 seeded with seed × 9973 + 11. Seven stratified columns, each a seeded shuffle of 300 evenly spaced quantiles, so every column has a regular spread.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The judge favours the answer shown first: this lab’s assumption. Which side a real judge favours, and by how much, depends on the model and the prompt.</li>`+
 `<li>People here judge content only, not length. On AlpacaEval’s data people picked the longer answer 64% of the time, and LLM judges 59% to 79%.</li>`+
 `<li>The large judge has half the small judge’s random error, the same preferences, and costs 5 times as much per call. All of this is assumed.</li>`+
 `<li>In Scenarios, how strongly a judge favours its own family’s answers is this lab’s assumption. The ${MTB} paper reports the effect; no number from it is used here.</li>`+
 `<li>The distribution of each random quantity is listed under Settings, “Assumed values”.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${MTB} in FastChat: ${cite(SRC.mtb,'README')} (3.3K human annotations, over 80% agreement, as between humans), ${cite(SRC.common,'<code>common.py</code>')} (both orders, “inconsistent”), ${cite(SRC.show,'<code>show_result.py</code>')} (win rate), ${cite(SRC.agree,'<code>compute_agreement.py</code>')} (agreement with ties and without).</li>`+
 `<li>AlpacaEval ${cite(SRC.alpaca,'README')}: the numbers 65.7%, 62.2%, 64% and 59–79%. It randomises the order of the two outputs by default. Length control raised its Spearman correlation with ${DEF('Chatbot Arena',TIP.arena())} from 0.93 to 0.98.</li>`+
 `<li>Chatbot Arena: ${cite(SRC.rating,'<code>rating_systems.py</code>')} (<code>compute_bt</code>, <code>compute_style_control</code>).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-03. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器里真的在跑一个带种子的模拟。模拟按设定好的偏好，既给出裁判的结论，也给出人的结论。这一页上的每个数都由它算出，${J_('math','“展开公式”')}里有逐行算式。真实的参考数字来自 ${DEF(MTB,TIP.mtb())} 和 ${DEF('AlpacaEval',TIP.alpaca())}，标了出处。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>300 个问题，每个都有 A、B 两个回答。两者内容上的差距 g 是一个${TD('logodds','对数几率')}，它决定人选 B 的概率：<div class="fx">${A.ui.math('P(\\text{人选 B}) = \\sigma(g)')}</div>${TD('sigma','σ')} 把对数几率变回概率。人选的时候有随机性，所以两个人之间也会有分歧。</li>`+
 `<li>裁判给 B 打一个分数：<div class="fx">${A.ui.math(SCORE)}</div>${LALB()} 是两个回答的长度。B 排在前面时 o = +1，排在后面时 o = −1。${A.ui.math('\\varepsilon_1')} 属于这一对，两种顺序里相同。${A.ui.math('\\varepsilon_2')} 每次调用都不同。分数高于 0.4 时裁判选 B，低于 −0.4 时选 A，在两者之间就说一样好。</li>`+
 `<li>两种顺序都判时，两次结论一致才算数，不一致算平局（${MTB}）。顺序随机时，每对只判一次，抛硬币决定谁在前（AlpacaEval）。胜率按 ${MTB} 的 <code>win_rate_adjusted</code> 算：<div class="fx">${A.ui.math('\\text{胜率} = \\frac{W + \\frac{T}{2}}{N}')}</div></li>`+
 `<li>去掉长度：以长度比的对数为自变量，对每一对的结论（1、½、0）做 logistic 回归，再读出两个回答一样长时的胜率。这是 AlpacaEval 长度控制的简化版。</li>`+
 `<li>一致率不计平局，和 <code>compute_agreement.py</code> 里不计平局的那种算法一样。为了比得公平，第二个人也只在裁判选了一边的那些对上算。</li>`+
 `<li>随机数用 mulberry32 生成，初始值为种子 × 9973 + 11。七列分层抽样，每列是 300 个均匀分位点的一个随机排列，所以每列的分布都很规整。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>裁判偏向排在前面的回答，这是这个 Lab 的假设。真实的裁判偏向哪边、偏多少，因模型和提示词而异。</li>`+
 `<li>这里的人只看内容，不看长度。在 AlpacaEval 的数据上，人有 64% 的时候选了更长的回答，LLM 裁判是 59% 到 79%。</li>`+
 `<li>大模型的随机误差是小模型的一半，偏好一样大，每次贵 5 倍。这些都是假设。</li>`+
 `<li>应用场景里，裁判有多偏爱自家模型的回答，是这个 Lab 的假设。${MTB} 的论文报告过这种现象，这里没有用论文里的数字。</li>`+
 `<li>每个随机量服从什么分布，列在“自由调参”的假设值里。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>FastChat 里的 ${MTB}：${cite(SRC.mtb,'README')}（3.3K 条人工标注，一致率超过 80%，和人与人之间一样），${cite(SRC.common,'<code>common.py</code>')}（两种顺序、“不一致”），${cite(SRC.show,'<code>show_result.py</code>')}（胜率），${cite(SRC.agree,'<code>compute_agreement.py</code>')}（计入平局和不计平局的一致率）。</li>`+
 `<li>AlpacaEval 的 ${cite(SRC.alpaca,'README')}：65.7%、62.2%、64%、59–79% 这几个数出自这里。它默认随机打乱两个回答的顺序。长度控制把它和 ${DEF('Chatbot Arena',TIP.arena())} 的 Spearman 相关从 0.93 提到 0.98。</li>`+
 `<li>Chatbot Arena：${cite(SRC.rating,'<code>rating_systems.py</code>')}（<code>compute_bt</code>、<code>compute_style_control</code>）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-03。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A win rate is the share of comparisons one version wins. Measured on a few hundred questions, it is only an estimate, and it comes with an interval. How wide the interval is, the lab ${labLink('eval-ci')} shows.</p><p>To turn many pairwise comparisons into one score per model, the usual tool is the Bradley–Terry model (it is how ${DEF('Chatbot Arena',TIP.arena())} ranks models):</p><div class="fx">${A.ui.math('P(\\text{A beats B}) = \\sigma(r_A - r_B)')}</div><p>${A.ui.math('r_A')} and ${A.ui.math('r_B')} are the two models’ scores. The lab ${labLink('dpo')} models preferences with the same math.</p>`,
                    `<p>胜率是一个版本在比较中赢的比例。在几百个问题上测出的胜率只是估计，会带着一个区间。区间有多宽，在${labLink('eval-ci')}这个 Lab 里讲。</p><p>要把很多次两两比较汇总成每个模型一个分数，常用 Bradley–Terry 模型（${DEF('Chatbot Arena',TIP.arena())} 的排名就是这样算的）：</p><div class="fx">${A.ui.math('P(\\text{A 胜 B}) = \\sigma(r_A - r_B)')}</div><p>${A.ui.math('r_A')}、${A.ui.math('r_B')} 是两个模型各自的分数。${labLink('dpo')}这个 Lab 用同样的数学给偏好建模。</p>`);
const OUTCOME=()=>tx('check an LLM judge against people, and set it up so its win rate can be trusted (cancel the order’s effect, take length out).','拿人工标注检验 LLM 裁判，并让它的胜率可信（抵消顺序的影响，再去掉长度的影响）。');
const EXITS=()=>{
  const t=run('def','small','both'), d1=run('def','small','B'), nw=run('nw','small','B'), nwt=run('nw','small','both');
  /* the new case's choices; the right one is the engine's win rate rounded to the nearest 10 points */
  const nwO=[['About 40%','40% 左右'],['About 50%','50% 左右'],['About 60%','60% 左右'],['About 70%','70% 左右']], nwA=Math.max(0,Math.min(3,Math.round(nw.w*10)-4));
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does judging each pair in both orders remove the position preference?','为什么每对回答换顺序各判一次，就能去掉位置偏好？'),
   o:[tx('The preference helps B in one call and A in the other, so counting a disagreement as a tie cancels it.','位置偏好一次帮 B、一次帮 A，两次不一致就算平局，偏好正好抵消。'),
      tx('Two calls double the number of verdicts, so the error shrinks, and the preference with it.','判两次，结论多了一倍，误差变小，偏好也跟着变小。'),
      tx('Once the order is swapped, the judge notices it is being tested and judges more carefully.','换了顺序，裁判就知道自己在被检查，会判得更认真。')],a:0,
   n:[null,tx('More calls shrink random error, not a steady preference. Two calls in the same order keep the preference whole.','多判几次只能减小随机误差，碰不到稳定的偏好。同一个顺序判两次，偏好一点也不会少。'),
     tx('Each call is separate; the judge never sees the other one. What works is that the two calls favour opposite answers.','每次调用都是独立的，裁判看不到另一次。起作用的是两次偏向的回答正好相反。')],
   why:tx('The pairs whose verdict flips with the order are the ones the position preference decided. Counted as ties, they no longer push the win rate toward the answer shown first.','一换顺序结论就变的回答对，正是位置偏好说了算的那些对。把它们算作平局，它们就不再把胜率推向排在前面的回答。')},
  {q:tx(`After length was taken out, B’s win rate fell from ${fR(t.w)} to ${fR(t.lc)}. What does that show?`,`去掉长度以后，B 的胜率从 ${fR(t.w)} 降到 ${fR(t.lc)}。这说明什么？`),
   o:[tx('The extra points came from B being longer, not from B being better.','多出来的几个百分点，来自 B 更长，而不是 B 更好。'),
      tx('Longer answers are worse, so the fit takes points off them.','长的回答其实更差，所以拟合给它扣了分。'),
      tx('The judge was wrong, so its verdicts in both orders are useless.','裁判判错了，两种顺序判出的结论都不能用。')],a:0,
   n:[null,tx('Taking length out does not punish length; it asks which answer wins at equal length. At equal length A and B are about even.','去掉长度不是给长回答扣分，而是问一样长时谁赢。一样长时，A 和 B 差不多。'),
     tx(`The same verdicts, refitted, land at ${fR(t.lc)}, close to people’s ${fR(t.H)}, so they are still useful.`,`同样这些结论，重新拟合后是 ${fR(t.lc)}，和人评的 ${fR(t.H)} 很接近，所以它们仍然有用。`)],
   why:tx('The fit separates the part of the verdicts that follows the length ratio from the rest, and reads the win rate where the ratio is 1.','拟合把结论里跟着长度走的那部分分离出来，再读出长度比为 1 时的胜率。')}],
 transfer:[
  /* the reference the reader saw first (part 1), then the new case, then the one question (R3.19); the lengths come from SC */
  {q:tx(`In the mission, B’s answers were on average ${lenWord(SC.def.R,false)} than A’s, and just as good. The judge judged each pair once with B shown first, and gave B ${fR(d1.w)}. Now a different new version is compared with A. Its answers are on average twice as long as A’s, and a little worse. People would pick it only ${fR(nw.H)} of the time. The judge again judges each pair once, with the new version shown first. About what win rate does the judge give the new version this time?`,
        `任务里，B 的回答平均比 A ${lenWord(SC.def.R,true)}，内容和 A 一样好。裁判每对只判一次，把 B 放在前面，给了 B ${fR(d1.w)} 的胜率。现在换一个新版本来和 A 比。它的回答平均比 A 长一倍，内容还略差一些。请人来评，它的胜率只有 ${fR(nw.H)}。裁判还是每对只判一次，也把新版本放在前面。这次裁判大约会给新版本多少胜率？`),
   o:nwO.map(([en,zh])=>tx(en,zh)),a:nwA,
   why:tx(`The judge gives ${nwO[nwA][0].toLowerCase()}, close to the mission’s ${fR(d1.w)}. The answers are twice as long, so the length preference adds more than in the mission. The content is a little worse, which pulls the score down a little. The two roughly cancel, and this lab’s simulation gives ${fR(nw.w)}. Judged in both orders, the judge gives ${fR(nwt.w)}. With length taken out as well, it gives ${fR(nwt.lc)}, close to people’s ${fR(nw.H)}.`,
          `裁判给的是 ${nwO[nwA][1]}，和任务里的 ${fR(d1.w)} 差不多。回答长了一倍，长度偏好带来的加分比任务里多。内容略差，又把分数拉低了一些。两边大致抵消，这个 Lab 的模拟给出 ${fR(nw.w)}。两种顺序都判，裁判给 ${fR(nwt.w)}。再去掉长度，就是 ${fR(nwt.lc)}，接近人评的 ${fR(nw.H)}。`)},
  {q:tx('On 100 labelled pairs, a judge agrees with people 85% of the time. On the same pairs, two people agree with each other 80% of the time. Can you use the judge’s win rate as it is?','在 100 对人工标注上，一个裁判和人的一致率是 85%。在同样这些对上，两个人之间的一致率是 80%。这个裁判给出的胜率，能直接拿来用吗？'),
   o:[tx('Yes: it agrees with people more often than people agree with each other.','能，它和人的一致率比人和人之间还高。'),
      tx('Not yet: also cancel the order’s effect, take length out, and compare that win rate with people’s on the labelled set.','还不能，还要抵消顺序的影响、去掉长度，再在标注集上把这样得出的胜率和人评的胜率比一比。'),
      tx('No: an LLM judge can never replace people.','不能，LLM 裁判永远代替不了人。')],a:1,
   why:tx(`Agreement only counts how many pairs the judge got right, and misses a steady push to one side. So a judge can agree with people more often than a second person does, and still give a win rate several points above people’s.`,
          `一致率只数选对了几对，看不出裁判是不是总往一边偏。所以一个裁判和人的一致率可以比第二个人还高，给出的胜率却仍然比人评高好几个百分点。`)}],
 practice:[
  {id:'llm-eval-system',d:tx('Deep dive (a): how to calibrate an autorater against human ratings, including agreement, position bias in side-by-side comparisons, and cancelling that bias by swapping the order. Parts 3 and 5 of this lab prepare for it.','深入问题 (a)：怎样拿人工评分校准 autorater，包括一致率、并排比较里的位置偏好，以及靠交换回答顺序抵消这种偏好。这个 Lab 的第 3 和第 5 部分就是为它做的准备。')},
  {id:'ml-fundamentals',d:tx('Question 1: why accuracy misleads on imbalanced data. A judge’s agreement is the same kind of single share: it cannot show which way the errors lean.','第 1 题：在不平衡的数据上，准确率为什么会误导人。裁判的一致率也是这样一个总的比例：它看不出错误偏向哪一边。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'eval',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('LLM as judge','LLM 当裁判'),title:l?L(l.q):tx('Can you trust an LLM as the judge?','LLM 当裁判靠得住吗？'),
    minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: win rates','前置知识：胜率'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage ljg-stage" id="ljg-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="ljg-st-title"></div><div class="meter-sub" id="ljg-st-sub"></div></div><div class="stage-actions"><span class="status" id="ljg-status"></span></div></div>`+
   `<div class="stage-graph graph ljg-graph" id="ljg-graph">`+
    `<div class="ljg-phone" id="ljg-phone"></div>`+
    `<div class="ljg-pair" id="ljg-pair"></div>`+
    `<div class="ljg-wf" id="ljg-wf"><div class="ljg-cap" id="ljg-wf-cap"></div><div class="ljg-wfrow"><div class="ljg-wfbox" id="ljg-wf-box"></div><ul class="ljg-wfleg" id="ljg-wf-leg"></ul></div></div>`+
    `<div class="ljg-bars" id="ljg-bars"></div>`+
   `</div>`+
   `<p class="ljg-scope small muted">${tx('Simulated verdicts from a seeded model of the judge; the question texts are illustrative.','裁判的结论来自带种子的模拟；问题文字只是示意。')}</p>`+
   `<div class="metrics ljg-metrics" id="ljg-metrics"></div>`+
   `<details class="math" id="ljg-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ljg-mathbody"></div></details></div>`;
}
const ASSUME=()=>[
 [tx('Position preference','位置偏好'),tx(`+0.5 ${TD('logodds','log-odds')} for the answer shown first`,`排在前面的回答加 0.5（${TD('logodds','对数几率')}）`)],
 [tx('Length preference','长度偏好'),tx('the score gets ','分数加上 ')+A.ui.math('1.0 \\times \\ln\\frac{L_B}{L_A}')+tx(`, where ${LALB()} are the two answers’ lengths`,`，${LALB()} 是两个回答的长度`)],
 [tx('“Equally good”','“一样好”'),tx('a score within ±0.4','分数在 ±0.4 之内')],
 [tx('Quality gap g','内容差距 g'),tx('one per pair, from a logistic distribution with standard deviation 2; its mean is set by the case (0 in the mission)','每对一个，服从标准差为 2 的 logistic 分布。平均值由场景决定，任务里是 0')],
 [tx('Length ratio','长度比'),tx(`the log of the length ratio, ${A.ui.math('\\ln\\frac{L_B}{L_A}')}, follows a logistic distribution with standard deviation 0.4; the mean ratio is set by the case (1.4 in the mission)`,`长度比的对数 ${A.ui.math('\\ln\\frac{L_B}{L_A}')} 服从标准差为 0.4 的 logistic 分布。平均长度比由场景决定，任务里是 1.4`)],
 [tx('Judge noise','裁判的噪声'),tx('two standard logistic draws (standard deviation about 1.81): one per pair, the same in both orders, times 0.9; one per call, times 0.4. The large judge has half of each','两个随机数，都服从标准 logistic 分布（标准差约 1.81）。一个属于这一对，两种顺序里相同，小模型乘 0.9。另一个每次调用都不同，小模型乘 0.4。大模型各减半')],
 [tx('People','人'),tx(`pick B with chance ${TD('sigma',A.ui.math('\\sigma(g)'))}; length plays no part`,`以 ${TD('sigma',A.ui.math('\\sigma(g)'))} 的概率选 B，不看长度`)],
 [tx('Random order','顺序随机'),tx('one uniform draw per pair; below 0.5, B is shown first','每对一个 0 到 1 之间的均匀随机数，小于 0.5 时 B 在前')]];
function controlsHTML(){
  return `<aside class="panel controls" id="ljg-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ljg-lock" hidden></p>`+
   `<div class="lockable" id="ljg-ctl-judge-wrap">`+A.ui.seg({id:'ljg-ctl-judge',key:'judge',label:tx('Judge','裁判'),options:[['small',tx('small model','小模型')],['big',tx('large model','大模型')]],value:S.judge,note:tx('The large judge costs 5 times as much per call and has half the random error. Its preferences are the same size. All assumed.','大模型每次贵 5 倍，随机误差只有小模型的一半，偏好一样大。这些都是假设。')})+`</div>`+
   `<div class="lockable" id="ljg-ctl-order-wrap">`+A.ui.seg({id:'ljg-ctl-order',key:'order',label:tx('Order','顺序'),options:[['B',tx('B first','B 在前')],['A',tx('A first','A 在前')],['R',tx('random','顺序随机')],['both',tx('both orders','两种顺序')]],value:S.order,note:tx('Random: one call per pair, a coin picks which answer comes first. Both orders: two calls per pair; if they disagree, a tie.','顺序随机：每对只判一次，抛硬币决定谁在前。两种顺序：每对判两次，两次结论不一致算平局。')})+`</div>`+
   `<div class="lockable" id="ljg-ctl-lc-wrap">`+A.ui.seg({id:'ljg-ctl-lc',key:'lc',label:tx('Length','长度'),options:[['0',tx('left in','不去掉')],['1',tx('taken out','去掉长度')]],value:S.lc?'1':'0',note:tx('Reads the win rate where both answers are equally long.','读出两个回答一样长时的胜率。')})+`</div>`+
   `<div class="ljg-grp" id="ljg-ctl-z"><div class="ljg-gh">${tx('Assumed values (set for this lab, not measured)','假设值（都是这个 Lab 设定的，不是测出来的）')}</div><ul class="ljg-asm">${ASSUME().map(([k,v])=>`<li><b>${esc(k)}</b> ${v}</li>`).join('')}</ul></div>`+
   `<div class="lockable" id="ljg-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import math, random

def judge(first, second, rng):
    """Stub: replace with a call to your judge model.
    Like the lab's judge, it favours the answer shown
    first and the longer answer."""
    s = (second["q"] - first["q"]
         + math.log(second["len"] / first["len"])
         - 0.5 + rng.gauss(0, 1))
    return "second" if s > 0.4 else "first" if s < -0.4 else "tie"

def both_orders(a, b, rng):           # B's result for one pair
    v1 = judge(a, b, rng)             # A shown first
    v2 = judge(b, a, rng)             # B shown first
    w1 = {"first": "A", "second": "B"}.get(v1, "tie")
    w2 = {"first": "B", "second": "A"}.get(v2, "tie")
    return w1 if w1 == w2 else "tie"  # inconsistent: a tie

def win_rate(results):                # ties count half
    pts = {"B": 1.0, "tie": 0.5, "A": 0.0}
    return sum(pts[r] for r in results) / len(results)

def agreement(judged, labels):        # pairs the judge did not tie
    pairs = [(j, h) for j, h in zip(judged, labels) if j != "tie"]
    return sum(j == h for j, h in pairs) / len(pairs)

rng = random.Random(0)
pairs = [({"q": 0.0, "len": 150}, {"q": 0.0, "len": 210})
         for _ in range(300)]         # B: 40% longer, just as good
once = []
for a, b in pairs:
    v = judge(b, a, rng)              # B shown first
    once.append({"first": "B", "second": "A"}.get(v, "tie"))
print(round(win_rate(once), 3))
print(round(win_rate([both_orders(a, b, rng) for a, b in pairs]), 3))
print(agreement(["B", "A", "tie"], ["B", "B", "A"]))   # 1 of 2`;
const CODE_OUT=`# 0.783   once, B first: both preferences
# 0.578   both orders: the length preference is left
# 0.5`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ljg-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ljg-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Pairwise judging in both orders (a placeholder function stands in for the judge), the win rate with ties, and the agreement with labels.','按两种顺序做两两比较（裁判先用一个假函数代替）、带平局的胜率，以及和人工标注的一致率。')}</p><pre class="code" id="ljg-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ljg-prod">`+
   tx(`<li>${MTB} in FastChat: ${cite(SRC.gen,'<code>fastchat/llm_judge/gen_judgment.py</code>')} sends the pairs to the judge; <code>play_a_match_pair</code> in ${cite(SRC.common,'<code>common.py</code>')} judges each pair in both orders, and two different winners make it “inconsistent”.</li>`+
      `<li>${cite(SRC.show,'<code>show_result.py</code>')} counts an inconsistent pair as a tie and reports <code>win_rate_adjusted</code>, with the same win-rate formula as this lab.</li><li>${cite(SRC.agree,'<code>compute_agreement.py</code>')} measures the judge’s agreement with people, both with ties and without.</li>`+
      `<li>AlpacaEval (${cite(SRC.alpaca,'README')}) judges each pair once and randomises the order of the two outputs by default. It reports length-controlled win rates: a regression on the length difference, read at equal length.</li>`+
      `<li>${DEF('Chatbot Arena',TIP.arena())} ranks models with a Bradley–Terry fit, <code>compute_bt</code> in ${cite(SRC.rating,'<code>fastchat/serve/monitor/rating_systems.py</code>')}, and offers a style control that includes length (<code>compute_style_control</code>).</li>`,
      `<li>FastChat 里的 ${MTB}：${cite(SRC.gen,'<code>fastchat/llm_judge/gen_judgment.py</code>')} 把每一对交给裁判；${cite(SRC.common,'<code>common.py</code>')} 里的 <code>play_a_match_pair</code> 把每一对按两种顺序各判一次，两次的赢家不同就记为“不一致”。</li>`+
      `<li>${cite(SRC.show,'<code>show_result.py</code>')} 把不一致算作平局，再用和这个 Lab 一样的胜率公式报告 <code>win_rate_adjusted</code>。</li><li>${cite(SRC.agree,'<code>compute_agreement.py</code>')} 计算裁判和人的一致率，计入平局和不计平局两种都算。</li>`+
      `<li>AlpacaEval（${cite(SRC.alpaca,'README')}）每对只判一次，默认随机打乱两个回答的先后顺序。它报告长度控制胜率，做法是对长度差做回归，读出一样长时的胜率。</li>`+
      `<li>${DEF('Chatbot Arena',TIP.arena())} 用 Bradley–Terry 拟合给模型排名（${cite(SRC.rating,'<code>fastchat/serve/monitor/rating_systems.py</code>')} 里的 <code>compute_bt</code>），还提供包含长度在内的风格控制（<code>compute_style_control</code>）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.zheng,'Judging LLM-as-a-Judge with MT‑Bench and Chatbot Arena')}<span>${tx('Zheng et al., 2023, arXiv 2306.05685','Zheng 等，2023，arXiv 2306.05685')}</span></li>`+
   `<li>${cite(SRC.dubois,'Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators')}<span>${tx('Dubois et al., 2024, arXiv 2404.04475','Dubois 等，2024，arXiv 2404.04475')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ljg-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ljg-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ljg-app m2" id="ljg-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ljg-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how far the judge’s win rate is from people’s.','选一个场景，看看裁判给的胜率和人评差多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ljg-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change the judge, the order or the length under ${J_('controls','Settings')}.`,`想换裁判、顺序或长度，在“${J_('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting shows the judge’s win rate, people’s and the gap, and tells you whether you met the goal, and why.','提交后会显示裁判的胜率、人评的胜率和两者的差距，并告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Within 3 points of people, at the lowest cost','和人评相差不超过 3 个百分点，还要最便宜')}</h3>`+
   `<p>${tx('New answers: B’s are on average 80% longer than A’s. You compare A and B with an LLM judge on 300 questions, and the judge’s win rate must be within 3 points of the win rate people would give. You decide three things: how to order the two answers, whether to take length out, and whether to use the small model or the large one. The large one costs 5 times as much per call and has half the random error. Among the setups that meet the goal, pick the cheapest.',
            '现在换了一组新的回答，B 的回答平均比 A 长 80%。你要用 LLM 裁判在 300 个问题上比较 A 和 B，要求裁判给出的胜率和请人评出的胜率相差不超过 3 个百分点。你可以决定三件事：两个回答的先后顺序怎么安排，要不要去掉长度的影响，用小模型还是大模型。大模型每次贵 5 倍，随机误差只有小模型的一半。在能达标的方案里，选最便宜的那个。')}</p>`+
   A.ui.seg({id:'ljg-ic-chJ',key:'chJ',label:tx('Judge','裁判'),options:[['small',tx('small','小模型')],['big',tx('large','大模型')]],value:S.judge,inline:true})+
   A.ui.seg({id:'ljg-ic-chO',key:'chO',label:tx('Order','顺序'),options:[['B',tx('B first','B 在前')],['R',tx('random order','顺序随机')],['both',tx('both orders','两种顺序都判')]],value:S.order,inline:true})+
   A.ui.seg({id:'ljg-ic-chL',key:'chL',label:tx('Length','长度'),options:[['0',tx('left in','不去掉')],['1',tx('taken out','去掉长度')]],value:S.lc?'1':'0',inline:true})+
   `<div class="goal mono ljg-goal" id="ljg-ch-g1"${chSubmitted?'':' hidden'}><span>${tx('Gap from people','和人评的差距')}</span><span><b id="ljg-ch-gap"></b>${tx(' (goal: 3 or less)','（目标：不超过 3）')}</span></div>`+
   `<div class="goal mono ljg-goal ljg-cost" id="ljg-ch-g2"><span>${tx('Cost','费用')}</span><span>${tx('','相当于 ')}<b id="ljg-ch-cost"></b>${tx(' small-judge calls (a large-judge call counts 5)',' 次小模型调用（大模型一次算 5 次）')}</span></div>`+
   submit+`<div id="ljg-ch-res" class="ljg-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
/* why Next is greyed out: handed to the kit (steps({nextWhy})) for every part whose Next can be disabled */
const WHY={mission:{en:'Pick an answer to continue.',zh:'选一个答案才能继续。'},order:{en:'Judge in both orders to continue.',zh:'选“两种顺序都判”才能继续。'},
  length:{en:'Take length out to continue.',zh:'去掉长度的影响才能继续。'},check:{en:'Check another judge to continue.',zh:'再检验一个裁判才能继续。'}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#ljg-rail',app); if(!rail) return null; rail.innerHTML=railFull(); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#ljg-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step]||(s.id==='order'&&S.order!=='B')||(s.id==='length')||(s.id==='check'));
    if(s.id==='order') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===S.order)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=ordTried.has(v)||v===S.order?ordRes(v):'';});
    if(s.id==='check') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===ckSel)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=ckTried.has(v)||v===ckSel?ckRes(v):'';});
    $$('.seg-ctl[data-key="lc"]',rail).forEach(g=>A.ui.setPressed(g,S.lc?'1':'0'));
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ljg-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?'':'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.judge,S.order,S.lc), g=$('#ljg-ch-gap',rail), c=$('#ljg-ch-cost',rail); if(!g) return;
    g.textContent=tx(`${fPt(e.gap)} points`,`${fPt(e.gap)} 个百分点`); c.textContent=fint(e.cost);
    const g1=$('#ljg-ch-g1',rail); g1.hidden=!chSubmitted; g1.classList.toggle('met',e.ok); g1.classList.toggle('ljg-over',!e.ok);
    $$('.seg-ctl[data-key="chJ"]',rail).forEach(x=>A.ui.setPressed(x,S.judge)); $$('.seg-ctl[data-key="chO"]',rail).forEach(x=>A.ui.setPressed(x,S.order)); $$('.seg-ctl[data-key="chL"]',rail).forEach(x=>A.ui.setPressed(x,S.lc?'1':'0'));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#ljg-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const curStep=()=>mode==='guided'?STEPS[step].id:'';
/* the chart shows win rates, or (part 5, part 6's third line) agreement with people */
const viewNow=()=>{const id=curStep(); return id==='check'||(id==='remember'&&remView==='check')?'agree':'win';};
/* the judge the agreement view shows: the one picked in part 5, the small judge in both orders in part 6 */
const agK=()=>curStep()==='check'?ckSel:'both';
const QS=[{en:'How do I undo my last git commit?',zh:'怎样撤销最近一次 git commit？'},{en:'Write a two-line birthday message for my sister.',zh:'给我妹妹写两句生日祝福。'},{en:'Why is the sky blue?',zh:'天空为什么是蓝色的？'},
  {en:'Summarize this email in one sentence.',zh:'用一句话概括这封邮件。'},{en:'Suggest a name for a coffee shop.',zh:'给一家咖啡店起个名字。'},{en:'Is it safe to eat raw cookie dough?',zh:'生的饼干面团能吃吗？'}];
const vName=v=>v===1?'B':(v===0?'A':tx('equal','一样好'));
const vPick=v=>v===0.5?tx('the judge calls them equal','裁判说一样好'):tx(`the judge picks ${vName(v)}`,`裁判选 ${vName(v)}`);
const lenWord=(r,zh)=>{const d=Math.round(Math.abs(r-1)*100); return d===0?'':(zh?(r>1?`长 ${d}%`:`短 ${d}%`):(r>1?`${d}% longer`:`${d}% shorter`));};
function lenNote(sc,la,lb){
  const R=SC[sc].R, r=lb/la, here=lenWord(r,A.lang()==='zh');
  const avg=R===1?tx('the same length on average over the 300 pairs','300 对平均一样长'):(R===2?tx('twice as long on average over the 300 pairs','300 对平均 B 长一倍'):tx(`${lenWord(R,false)} on average over the 300 pairs`,`300 对平均 B ${lenWord(R,true)}`));
  return here?tx(`In this pair B is ${here} than A; ${avg}.`,`这一对 B 比 A ${here}，${avg}。`):tx(`In this pair B and A are the same length; ${avg}.`,`这一对 B 和 A 一样长，${avg}。`);
}
function pairHTML(){
  const r=R(), i=pairAt, v=r.v[i], {la,lb}=wordsOf(S.sc,i), mx=Math.max(la,lb,1)*1.08;
  const words=n=>tx(`${fint(n)} words`,`${fint(n)} 词`);
  const box=(k,n)=>`<div class="ljg-ans"><b>${k}</b><span class="ljg-lenbar"><i style="width:${(n/mx*100).toFixed(1)}%"></i></span><span class="mono ljg-w">${words(n)}</span></div>`;
  const vB=tx(`B shown first: ${vPick(v.a)}`,`B 在前：${vPick(v.a)}`), vA=tx(`A shown first: ${vPick(v.b)}`,`A 在前：${vPick(v.b)}`);
  let vd;
  if(S.order==='B') vd=`<span class="ljg-vk">${vB}</span>`;
  else if(S.order==='A') vd=`<span class="ljg-vk">${vA}</span>`;
  else if(S.order==='R') vd=`<span class="ljg-vk">${tx(`Random order, this pair got ${v.bf?'B':'A'} first`,`顺序随机，这一对抽到 ${v.bf?'B':'A'} 在前`)}</span><span class="ljg-vk ljg-vres">${v.bf?vB:vA}</span>`;
  else vd=`<span class="ljg-vk">${vA}</span><span class="ljg-vk">${vB}</span><span class="ljg-vk ljg-vres">${v.a===v.b?tx(`agree: ${vName(v.a)}`,`一致：${vName(v.a)}`):tx('inconsistent: a tie','不一致：算平局')}</span>`;
  return `<div class="ljg-q"><span class="ljg-qn mono">${tx(`Question ${i+1} of ${N}`,`问题 ${i+1} / ${N}`)}</span><span class="ljg-qt">${esc(S.sc==='math'?tx('A math word problem with one right answer.','一道有唯一正确答案的数学应用题。'):L(QS[i%QS.length]))}</span></div>`+
    `<div class="ljg-anss">${box('A',la)}${box('B',lb)}</div><p class="ljg-lr small muted">${esc(lenNote(S.sc,la,lb))}</p><div class="ljg-vd">${vd}</div>`;
}
/* the 300 verdicts as marks, grouped by kind so the shares read as areas; each kind is named beside the marks with its count (R5.9) */
const KINDS=[{k:'B',sh:'fill'},{k:'T',sh:'dash'},{k:'I',sh:'x'},{k:'A',sh:'ring'}];
const kindName=k=>({B:tx('B wins','B 赢'),T:tx('tie','平局'),I:tx('inconsistent (a tie)','不一致（算平局）'),A:tx('A wins','A 赢')}[k]);
function markSVG(sh,cx,cy,s,on){
  const r=s*0.36, w=s*0.3;
  let g='';
  if(sh==='fill') g=`<circle cx="${cx}" cy="${cy}" r="${r}" class="ljg-mf"/>`;
  else if(sh==='ring') g=`<circle cx="${cx}" cy="${cy}" r="${r-0.6}" class="ljg-mr"/>`;
  else if(sh==='dash') g=`<line x1="${cx-w}" x2="${cx+w}" y1="${cy}" y2="${cy}" class="ljg-md"/>`;
  else g=`<path d="M${cx-w} ${cy-w}L${cx+w} ${cy+w}M${cx+w} ${cy-w}L${cx-w} ${cy+w}" class="ljg-mx"/>`;
  if(on) g+=`<rect x="${cx-s/2+0.5}" y="${cy-s/2+0.5}" width="${s-1}" height="${s-1}" rx="2" class="ljg-mon"/>`;
  return g;
}
function kindOf(x){return x.inc?'I':(x.s===1?'B':(x.s===0?'A':'T'));}
function drawWaffle(){
  const box=$('#ljg-wf-box',app), leg=$('#ljg-wf-leg',app), cap=$('#ljg-wf-cap',app), v=viewNow();
  const W=Math.max(200,Math.round(box.clientWidth||300));
  if(v==='agree'){
    /* two 10 × 10 blocks, one mark per labelled pair; the judge's ties are left out of both (R5.10, S5) */
    const k=agK(), r=run('def',CK[k].j,CK[k].o), B=base('def'), s=Math.max(8,Math.min(16,Math.floor((W-28)/20)));
    const blk=cells=>`<svg width="${10*s}" height="${10*s}" viewBox="0 0 ${10*s} ${10*s}" aria-hidden="true">${cells.map((c,i)=>markSVG(c,(i%10)*s+s/2,Math.floor(i/10)*s+s/2,s,false)).join('')}</svg>`;
    const lab=r.v.slice(0,NL), jc=lab.map(x=>x.s===0.5?'dash':(x.s===B.pr[x.i].h?'fill':'ring')), pc=lab.map(x=>x.s===0.5?'dash':(B.pr[x.i].h2===B.pr[x.i].h?'fill':'ring'));
    const fig=(t,cells)=>`<figure class="ljg-agb"><figcaption class="ljg-svt">${esc(t)}</figcaption>${blk(cells)}</figure>`;
    box.innerHTML=`<div class="ljg-agw" role="img" aria-label="${esc(tx(`Agreement on the 100 labelled pairs: the judge with the person ${fR(r.agree)}, a second person with the first ${fR(r.pagree)}, both on the ${r.agN} pairs the judge decided`,`100 对人工标注上的一致情况：裁判和人 ${fR(r.agree)}，第二个人和第一个人 ${fR(r.pagree)}，都只算裁判选了一边的 ${r.agN} 对`))}">`+
      fig(tx(`The judge with the person: ${fR(r.agree)}`,`裁判和人：${fR(r.agree)}`),jc)+fig(tx(`A second person with the first: ${fR(r.pagree)}`,`第二个人和第一个人：${fR(r.pagree)}`),pc)+`</div>`;
    cap.textContent=tx(`The 100 pairs people labelled, one mark each; both blocks count the same ${r.agN} pairs`,`人工标注的 100 对，每对一个记号。两块都只算同样的 ${r.agN} 对`);
    leg.innerHTML=[['fill',tx('same answer','选得一样')],['ring',tx('different answer','选得不一样')],['dash',tx('the judge said equal: left out of both','裁判判一样好：两边都不计入')]].map(([sh,t])=>`<li><svg width="14" height="14" aria-hidden="true">${markSVG(sh,7,7,14,false)}</svg><span>${esc(t)}</span></li>`).join('');
    return;}
  const r=R(), cnt={B:0,T:0,I:0,A:0}; r.v.forEach(x=>cnt[kindOf(x)]++);
  const order=[...r.v].sort((a,b)=>'BTIA'.indexOf(kindOf(a))-'BTIA'.indexOf(kindOf(b))||a.i-b.i);
  const cols=W>=420?30:20, s=Math.min(16,Math.floor(W/cols)), rows=Math.ceil(N/cols), sh=k=>KINDS.find(z=>z.k===k).sh;
  box.innerHTML=`<svg width="${cols*s}" height="${rows*s}" viewBox="0 0 ${cols*s} ${rows*s}" role="img" aria-label="${esc(tx(`300 verdicts: B wins ${cnt.B}, ties ${cnt.T}, inconsistent ${cnt.I}, A wins ${cnt.A}`,`300 个结论：B 赢 ${cnt.B}，平局 ${cnt.T}，不一致 ${cnt.I}，A 赢 ${cnt.A}`))}">`+
    order.map((x,j)=>markSVG(sh(kindOf(x)),(j%cols)*s+s/2,Math.floor(j/cols)*s+s/2,s,x.i===pairAt)).join('')+`</svg>`;
  cap.textContent=tx(`All 300 verdicts, grouped by kind (the outlined mark is the pair above)`,`全部 300 个结论，按种类排在一起（带框的是上面那一对）`);
  leg.innerHTML=KINDS.filter(z=>z.k!=='I'||S.order==='both').map(z=>`<li><svg width="14" height="14" aria-hidden="true">${markSVG(z.sh,7,7,14,false)}</svg><span>${esc(kindName(z.k))}</span><b class="mono">${cnt[z.k]}</b></li>`).join('');
}
/* the win-rate bars: the recipes the parts build up, the current setup outlined, people's win rate as a line once it is shown */
function barRows(){
  const id=curStep(), j=S.judge, sc=S.sc;
  const all=[{k:'B',o:'B',lc:false},{k:'A',o:'A',lc:false},{k:'R',o:'R',lc:false},{k:'both',o:'both',lc:false},{k:'bothlc',o:'both',lc:true}];
  let keep;
  if(id==='mission'||id==='how') keep=['B'];
  else if(id==='order') keep=['B'].concat(['A','R','both'].filter(o=>ordTried.has(o)));
  else if(id==='length') keep=['B','A','both'].concat(lcSeen?['bothlc']:[]);
  else keep=['B','A','both','bothlc'];
  if(mode==='challenge') keep=[];
  const rows=all.filter(x=>keep.includes(x.k)).map(x=>({...x,r:run(sc,j,x.o)}));
  const on=rows.find(x=>x.o===S.order&&x.lc===S.lc);
  if(!on&&mode!=='guided') rows.push({k:'me',o:S.order,lc:S.lc,r:run(sc,j,S.order),me:true});
  return rows;
}
const rowName=x=>x.me?tx(`Your setup: ${L(ORDS[x.o])}, ${x.lc?'length taken out':'length left in'}`,`你的设置：${L(ORD[x.o])}，${x.lc?'去掉长度':'不去掉长度'}`):({B:L(ORD.B),A:L(ORD.A),R:L(ORD.R),both:L(ORD.both),bothlc:tx('Both orders, length taken out','两种顺序，去掉长度')}[x.k]);
function drawBars(){
  if(chHide()){const host=$('#ljg-bars',app), html=`<div class="ljg-cap">${tx('B’s win rate','B 的胜率')}</div><p class="small muted ljg-wait">${tx('Submit your answer to see the judge’s win rate for B here, with people’s.','提交答案后，这里会显示裁判给 B 的胜率和人评的胜率。')}</p>`; if(host._h!==html){host.innerHTML=html; host._h=html;} return;}
  const host=$('#ljg-bars',app), rows=barRows(), show=peopleShown(), H=rows[0]?rows[0].r.H:0.5, mx=1, pc=v=>(v*100/mx).toFixed(2)+'%';
  const ci=mode!=='guided'||step>=SIX('how');
  const html=`<div class="ljg-cap">${tx(S.sc==='math'?'B’s win rate: the judge’s bars, the checker’s line':'B’s win rate: the judge’s bars'+(show?', people’s line':''),S.sc==='math'?'B 的胜率：横条是裁判的，竖线是核对答案的':'B 的胜率：横条是裁判的'+(show?'，竖线是人评的':''))}</div>`+
   `<div class="ljg-brows">`+rows.map(x=>{const v=val(x.r,x.lc), cur=x.o===S.order&&x.lc===S.lc;
    return `<div class="ljg-brow${cur?' on':''}" data-tipx="${esc(rowName(x)+tx(`: B wins ${fP(v)}`,`，B 赢 ${fP(v)}`)+(x.lc?'':tx(`, 95% interval ${fP(x.r.lo)} to ${fP(x.r.hi)}`,`，95% 区间 ${fP(x.r.lo)} 到 ${fP(x.r.hi)}`)))}"><span class="ljg-bl">${esc(rowName(x))}</span><span class="ljg-bt"><i class="ljg-bf" style="width:${pc(v)}"></i>`+
      (ci&&!x.lc?`<i class="ljg-ci" style="left:${pc(x.r.lo)};width:${pc(x.r.hi-x.r.lo)}"></i>`:'')+
      `<em class="ljg-bv mono" style="left:${pc(v)}">${fR(v)}</em>${show?`<b class="ljg-hl" style="left:${pc(H)}"></b>`:''}<b class="ljg-50" style="left:50%"></b></span></div>`;}).join('')+`</div>`+
   `<div class="ljg-axis"><span class="ljg-bl"></span><span class="ljg-bt">${show?`<em class="ljg-hlab mono" style="left:${pc(H)}">${esc(S.sc==='math'?tx(`checked: ${fR(H)}`,`核对：${fR(H)}`):tx(`people: ${fR(H)}`,`人评：${fR(H)}`))}</em>`:''}${show&&Math.abs(H-0.5)<0.07?'':'<em class="ljg-50lab mono" style="left:50%">50%</em>'}</span></div>`;
  if(host._h!==html){host.innerHTML=html; host._h=html;}
}
/* the agreement bars of part 5 */
function drawAgree(){
  const host=$('#ljg-bars',app), k0=agK(), r0=run('def',CK[k0].j,CK[k0].o);
  const rows=Object.keys(CK).filter(k=>curStep()!=='check'||ckTried.has(k)||k===ckSel).map(k=>({k,t:L(CK[k]),v:run('def',CK[k].j,CK[k].o).agree,on:k===k0}))
    .concat([{k:'p2',t:tx(`A second person, on the same ${r0.agN} pairs`,`第二个人，同样这 ${r0.agN} 对`),v:r0.pagree,on:false,pp:true}]);
  const html=`<div class="ljg-cap">${tx('Agreement with the person on the 100 labelled pairs (the judge’s ties left out)','在 100 对人工标注上和人的一致率（裁判判平局的对不计入）')}</div><div class="ljg-brows">`+
    rows.map(x=>`<div class="ljg-brow${x.on?' on':''}${x.pp?' pp':''}" data-tipx="${esc(x.t+tx(`: agrees ${fP(x.v)}`,`，一致 ${fP(x.v)}`))}"><span class="ljg-bl">${esc(x.t)}</span><span class="ljg-bt"><i class="ljg-bf" style="width:${(x.v*100).toFixed(2)}%"></i><em class="ljg-bv mono" style="left:${(x.v*100).toFixed(2)}%">${fR(x.v)}</em><b class="ljg-50" style="left:50%"></b></span></div>`).join('')+
    `</div><div class="ljg-axis"><span class="ljg-bl"></span><span class="ljg-bt"><em class="ljg-50lab mono" style="left:50%">50%</em></span></div>`;
  if(host._h!==html){host.innerHTML=html; host._h=html;}
}
function statusOf(){
  if(mode==='challenge'){const e=chEval(S.judge,S.order,S.lc);
    if(!chSubmitted) return {cls:'neutral',txt:tx(`Cost ${fint(e.cost)}`,`费用 ${fint(e.cost)}`),short:tx(`cost ${fint(e.cost)}`,`费用 ${fint(e.cost)}`),tip:tx('The cost of this setup, in small-judge calls. Submit your answer to see the gap from people.','这个方案的费用，相当于多少次小模型调用。提交答案后，才显示它和人评的差距。')};
    return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Within 3 points','相差不超过 3'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the judge’s win rate is within 3 points of people’s. Now: ${fPt(e.gap)} points.`,`裁判的胜率和人评相差不超过 3 个百分点时，徽章是绿色。现在相差 ${fPt(e.gap)} 个百分点。`)};}
  if(viewNow()==='agree'){const k=agK(), r=run('def',CK[k].j,CK[k].o); return {cls:'neutral',txt:tx(`Agrees ${fR(r.agree)}`,`一致 ${fR(r.agree)}`),short:tx(`agrees ${fR(r.agree)}`,`一致 ${fR(r.agree)}`),tip:tx('How often the judge picks the same answer as the person, on the pairs where it picked one.','在裁判选了一边的那些对里，它和人选得一样的比例。')};}
  const r=R(), v=val(r,S.lc);
  return {cls:'neutral',txt:tx(`B wins ${fR(v)}`,`B 赢 ${fR(v)}`),short:tx(`B ${fR(v)}`,`B 赢 ${fR(v)}`),tip:tx('B’s win rate from the judge with the current setup.','按当前设置，裁判给 B 的胜率。')};
}
function drawHead(){
  const t=$('#ljg-st-title',app), sub=$('#ljg-st-sub',app), st=$('#ljg-status',app), v=viewNow();
  const scT={def:tx('B 40% longer','B 长 40%'),chatty:tx('B twice as long','B 长一倍'),better:tx('B better, same length','B 更好，长度一样'),self:tx('B from the judge’s family','B 和裁判同一家'),math:tx('math, B 40% longer','数学题，B 长 40%'),ch:tx('new answers, B 80% longer','新的回答，B 长 80%')}[S.sc];
  t.textContent=v==='agree'?tx('100 labelled pairs: the judge against people','100 对人工标注：裁判和人比一比'):tx(`300 questions, A against B (${scT})`,`300 个问题，A 和 B 比一比（${scT}）`);
  /* the agreement view names the judge being checked, without the length setting (it plays no part in agreement) */
  sub.textContent=v==='agree'?L(CK[agK()]):tx(`${jName(S.judge)}, ${L(ORDS[S.order])}, ${S.lc?'length taken out':'length left in'}`,`${jName(S.judge)}，${L(ORD[S.order])}，${S.lc?'去掉长度':'不去掉长度'}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  if(viewNow()==='agree'){const k=agK(), r=run('def',CK[k].j,CK[k].o);
    return [{k:'ag',label:tx('The judge with the person','裁判和人'),v:fP(r.agree),sub:tx(`${r.ag} of ${r.agN} pairs`,`${r.agN} 对里 ${r.ag} 对`),pts:true,tip:tx(`The judge picked a side on ${r.agN} of the 100 pairs, and the person’s answer on ${r.ag} of them.`,`100 对里，裁判选了一边的有 ${r.agN} 对，其中 ${r.ag} 对和人选得一样。`)},
      {k:'pp',label:tx('Second person with the first','第二个人和第一个人'),v:fP(r.pagree),sub:tx(`${r.pag} of the same ${r.agN} pairs`,`同样这 ${r.agN} 对里 ${r.pag} 对`),pts:true,tip:tx('On the same pairs the judge decided, how often a second person picks the same answer as the first.','在裁判选了一边的那些对上，第二个人和第一个人选得一样的比例。')},
      {k:'gap',pts:true,label:tx('Win-rate gap from people','胜率和人评的差距'),v:fPt(Math.abs(r.w-r.H)*100),sub:tx('points; all 300 pairs, length left in','个百分点（300 对，不去掉长度）'),tip:tx('The judge’s win rate against people’s on all 300 pairs. Agreement does not show this gap.','裁判在全部 300 对上的胜率和人评的差距。一致率看不出这个差距。')}];}
  const r=R(), v=val(r,S.lc);
  if(chHide()) return [{k:'cost',label:tx('Cost','费用'),v:fint(r.cost),sub:tx('small-judge calls','相当于多少次小模型调用'),tip:tx('A small-judge call counts 1, a large-judge call 5; both orders take two calls per pair.','小模型调用一次算 1，大模型调用一次算 5。两种顺序都判时，每对要调用两次。')}];
  const m=[{k:'w',label:tx('B’s win rate','B 的胜率'),v:fP(v),sub:S.lc?tx('length taken out','去掉长度'):tx(`B ${r.W}, ties ${r.T}, A ${r.Lo}`,`B 赢 ${r.W}，平局 ${r.T}，A 赢 ${r.Lo}`),pts:true,tip:tx('B’s win rate; “Show the math” works it out. With length taken out, it is the fitted win rate where both answers are equally long.','B 的胜率，算法见“展开公式”。去掉长度时，显示的是拟合出的、两个回答一样长时的胜率。')}];
  if(mode!=='guided'||step>=SIX('how')) m.push({k:'ci',label:tx('95% interval','95% 区间'),v:`± ${fixd(1.96*r.se*100,1)}`,sub:tx('points, length left in','个百分点，不去掉长度时'),tip:TIP.ci()});
  if(S.order==='both') m.push({k:'inc',label:tx('Inconsistent','不一致'),v:fP(r.inc/N),sub:tx(`${r.inc} of 300 pairs`,`300 对里 ${r.inc} 对`),pts:true,tip:tx('Pairs whose two verdicts differ; each counts as a tie.','两种顺序判出的结论不一样的回答对，每一对都按平局算。')});
  if(peopleShown()) m.push({k:'h',label:S.sc==='math'?tx('Checked answers','核对答案'):tx('People’s win rate','人评的胜率'),v:fP(r.H),sub:tx(`gap ${fPt(Math.abs(v-r.H)*100)} pts`,`差 ${fPt(Math.abs(v-r.H)*100)} 个百分点`),pts:true,tip:tx('B’s win rate if people judged every pair (on math, if every answer were checked).','如果每一对都请人来评（数学题就直接核对答案），B 的胜率是多少。')});
  return m;
}
function drawMetrics(){
  const host=$('#ljg-metrics',app), defs=metricsDef(), sg=defs.map(m=>m.k).join();
  if(host._sig!==sg){host._sig=sg; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows */
const f3=x=>fixd(x,3), sg3=x=>(x<0?'−':'')+fixd(Math.abs(x),3);
function mathHTML(){
  if(chHide()) return `<p>${tx('Submit your answer to see the working for your setup.','提交答案后，这里显示你这个方案的算式。')}</p>`;
  const r=R(), sd=r.sd, se=r.se;
  let h=MB(tx(`<b>Win rate.</b> A win counts 1 and a tie counts half. W is B’s wins, T the ties and N the ${N} questions.`,`<b>胜率。</b>赢一次记 1，平局记一半。W 是 B 赢的次数，T 是平局数，N 是问题数 ${N}。`),
      EQ([[tx('\\text{win rate}','\\text{胜率}'),'\\frac{W + \\frac{T}{2}}{N}'],['',`\\frac{${r.W} + \\frac{${r.T}}{2}}{${N}}`],['',f3(r.w)]]))+
    MB(tx(`<b>95% interval.</b> The standard error SE says how far the win rate would move on another ${N} questions of the same kind. s is the standard deviation of the ${N} results (1, ½, 0).`,`<b>95% 区间。</b>标准误差 SE 表示换一批同类的 ${N} 个问题，胜率大致会变多少。s 是 ${N} 个结果（1、½、0）的标准差。`),
      EQ([[tx('\\text{standard error}','\\text{标准误差}')+'\\,\\,\\text{SE}','\\frac{s}{\\sqrt{N}}'],['',`\\frac{${f3(sd)}}{\\sqrt{${N}}}`],['',fixd(se,4)],
          [tx('\\text{interval}','\\text{区间}'),tx('\\text{win rate}','\\text{胜率}')+' ± 1.96 \\times \\text{SE}'],['',`${f3(r.w)} ± 1.96 \\times ${fixd(se,4)}`],['',`${f3(r.w)} ± ${f3(1.96*se)}`]]));
  if(S.order==='both') h+=MB(tx('<b>Both orders.</b>','<b>两种顺序。</b>')+tx(`${r.inc} pairs got two different verdicts and count as ties.`,`${r.inc} 对两次结论不同，算平局。`),'');
  h+=MB(tx(`<b>Length taken out.</b> Fit how the chance that the judge picks B grows with the length ratio, then read it where both answers are equally long. p is that chance, and ${TD('sigma','σ')} turns ${TD('logodds','log-odds')} into a probability. ${LALB()} are the two answers’ lengths. The fit uses all ${N} results.`,
             `<b>去掉长度。</b>先拟合裁判选 B 的概率怎样随长度比变化，再读出两个回答一样长时的值。p 是裁判选 B 的概率，${TD('sigma','σ')} 把${TD('logodds','对数几率')}变回概率。${LALB()} 是两个回答的长度。拟合用全部 ${N} 个结果。`),
      EQ([[tx('\\text{chance of B}','\\text{选 B 的概率}')+'\\,\\,p','\\sigma(\\alpha + \\beta \\cdot x)'],['x','\\ln\\frac{L_B}{L_A}']])+
      `<p>${tx(`The fit gives α = ${sg3(r.fit.a)} and β = ${sg3(r.fit.b)}. At equal length, x = 0:`,`拟合得到 α = ${sg3(r.fit.a)}，β = ${sg3(r.fit.b)}。两个回答一样长时，x = 0：`)}</p>`+
      EQ([[tx('\\text{win rate, length taken out}','\\text{去掉长度后的胜率}'),'\\sigma(\\alpha + \\beta \\cdot 0)'],['','\\sigma(\\alpha)'],['',`\\sigma(${sg3(r.fit.a)})`],['',f3(r.lc)]]))+
    MB(tx(`<b>Agreement.</b> On the 100 pairs people labelled, count the pairs where the judge picked a side, and how many of those match the person. Then the same count for a second person, on the same pairs.`,`<b>一致率。</b>在 100 对人工标注里，数出裁判选了一边的对，再数其中和人一样的对。第二个人也在同样这些对上数一遍。`),
      EQ([[tx('\\text{judge and person}','\\text{裁判和人}'),`\\frac{${r.ag}}{${r.agN}}`],['',f3(r.agree)],[tx('\\text{second person and first}','\\text{第二个人和第一个人}'),`\\frac{${r.pag}}{${r.agN}}`],['',f3(r.pagree)]]));
  return h;
}
function render(){
  if(!app) return;
  if(mode==='challenge'){S.sc='ch'; if(S.order==='A') S.order='B';}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const v=viewNow(), pr=$('#ljg-pair',app); pr.hidden=v==='agree';
  if(!pr.hidden){const ph=pairHTML(); if(pr._h!==ph){pr.innerHTML=ph; pr._h=ph;}}
  drawWaffle(); if(v==='agree') drawAgree(); else drawBars(); drawMetrics();
  const mb=$('#ljg-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change the judge, the order and the length in the task card. The answers are fixed: 300 new pairs, B 80% longer.','挑战中只能在任务卡里改裁判、顺序和长度。回答是固定的，是 300 对新的回答，B 平均长 80%。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='order') return tx('In this part you pick the order in the card. The judge is the small model, with length left in. You can change them in Scenarios.','这一部分只在卡片里选顺序。裁判是小模型，不去掉长度。在应用场景里可以改。');
  if(id==='length') return tx('In this part you take length out in the card. The judge is the small model, in both orders. You can change them in Scenarios.','这一部分只在卡片里去掉长度。裁判是小模型，两种顺序都判。在应用场景里可以改。');
  if(id==='check') return tx('In this part you pick which judge to check in the card. You can change the settings in Scenarios.','这一部分只在卡片里选检验哪个裁判。在应用场景里可以改设置。');
  return tx('Fixed in this part: the small judge, judging once with B first, length left in. You can change them in Scenarios.','这一部分固定用小模型裁判，判一次，B 在前，不去掉长度。在应用场景里可以改。');
}
function syncControls(){
  $$('.seg-ctl[data-key="judge"]',app).forEach(g=>A.ui.setPressed(g,S.judge));
  $$('#ljg-controls .seg-ctl[data-key="order"]',app).forEach(g=>A.ui.setPressed(g,S.order));
  $$('#ljg-controls .seg-ctl[data-key="lc"]',app).forEach(g=>A.ui.setPressed(g,S.lc?'1':'0'));
  const c=$('#ljg-controls',app), g=mode==='guided';
  c.classList.toggle('locked',mode==='challenge');
  const hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  hide('#ljg-ctl-judge-wrap',g); hide('#ljg-ctl-order-wrap',g); hide('#ljg-ctl-lc-wrap',g); hide('#ljg-ctl-reset-wrap',g||mode==='challenge');
  const ln=$('#ljg-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.txt);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p); if(!SC[S.sc]) S.sc='def'; if(!JUDGES[S.judge]) S.judge='small'; if(!ORD[S.order]) S.order='B'; S.lc=!!S.lc; render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setOrder(o){ordTried.add(o); setState({order:o});}
function setLc(on){if(on&&isP('length')) lcSeen=true; setState({lc:on});}
function setCk(k){ckSel=k; ckTried.add(k); render();}
function setRem(k){if(!REMS.includes(k)) return; remView=k; S={...BASE,order:'both',lc:k!=='order'}; renderRail(); if(innerWidth<=760) A.ui.reveal($('#ljg-stage',app));}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='order') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the three ways in the card.','向下滚到卡片里的三种判法。'))}">${tx('Pick below ↓','在下面选 ↓')}</button>`;
  if(id==='length') return lcCtl('ljg-ph-lc');
  if(id==='how') return `<button type="button" class="btn ghost sm" data-act="pair-next">${tx('Next pair →','看下一对 →')}</button>`;
  if(id==='check') return `<button type="button" class="btn ghost sm" data-act="to-ck">${tx('Pick a judge below ↓','在下面选裁判 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#ljg-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.id==='length') S.lc=false;
  renderRail();
  const tg={order:'#ljg-ic-ord',length:innerWidth<=760?'#ljg-ph-lc':'#ljg-ic-lc',check:'#ljg-ic-ck',how:'#ljg-ic-pair'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#ljg-rail .step.current',app)||$('#ljg-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#ljg-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ljg-rail .step.current',app)||$('#ljg-rail .railcard',app)||$('#ljg-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#ljg-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ljg-insight',app)); else revealBox($('#ljg-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH_START,sc:'ch'};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  pairAt=featOf(m==='challenge'?'ch':S.sc);
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.sc,s.judge,s.order,s.lc?1:0].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==4) return null;
  const [sc,judge,order,lc]=p;
  if(!['def','chatty','better','self','math'].includes(sc)||!JUDGES[judge]||!ORD[order]||!['0','1'].includes(lc)) return null;
  return {sc,judge,order,lc:lc==='1'};
}

/* =====================================================================
   interview drill: seeded counts from both orders; every line redoes from the numbers it shows
   ===================================================================== */
function makeDrill(seed){const rnd=mulberry(seed*104729+7), n=200, bb=60+Math.floor(rnd()*40), aa=30+Math.floor(rnd()*40), cc=n-bb-aa;
  return {seed,n,bb,aa,cc,w:(bb+cc/2)/n};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#ljg-drill-q',app), a=$('#ljg-drill-a',app);
  qd.innerHTML=tx(`A judge compares two versions on <b>${d.n}</b> pairs, each pair once in each order. Both verdicts pick B on <b>${d.bb}</b> pairs and A on <b>${d.aa}</b>; the other <b>${d.cc}</b> pairs are ties or inconsistent. What is B’s win rate, and how would you report it?`,
                  `一个裁判在 <b>${d.n}</b> 对回答上比较两个版本，每对按两种顺序各判一次。两次都选 B 的有 <b>${d.bb}</b> 对，都选 A 的有 <b>${d.aa}</b> 对，其余 <b>${d.cc}</b> 对是平局或不一致。B 的胜率是多少？你会怎样报告它？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const se=Math.sqrt(((d.bb*(1-d.w)**2)+(d.aa*(0-d.w)**2)+(d.cc*(0.5-d.w)**2))/(d.n-1))/Math.sqrt(d.n);
  const dEq=EQ([[tx('\\text{win rate}','\\text{胜率}'),`\\frac{${d.bb} + \\frac{${d.cc}}{2}}{${d.n}}`],['',f3(d.w)]]);
  a.innerHTML=tx(`<p><b>B’s win rate is ${f3(d.w)}, about ${fR(d.w)}, ± ${fixd(1.96*se*100,1)} points (95%).</b></p>${dEq}<p>Inconsistent pairs count as ties, half a win each. Report the interval too, and say the judge ran in both orders; if B’s answers are longer, add the length-controlled win rate.</p>`,
                 `<p><b>B 的胜率是 ${f3(d.w)}，约 ${fR(d.w)}，± ${fixd(1.96*se*100,1)} 个百分点（95%）。</b></p>${dEq}<p>不一致的对算平局，双方各记半次赢。报告时带上区间，并说明裁判两种顺序都判了。如果 B 的回答更长，再附上去掉长度后的胜率。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const TERM={
 logodds:()=>`<div>${esc(tx('Log-odds put a probability p on a scale with no upper or lower limit:','对数几率把一个概率 p 换到一个没有上下限的尺度上：'))}</div><div class="fx">${A.ui.math('\\ln\\frac{p}{1 - p}')}</div><div>${esc(tx('Adding 0.5 moves 50% to about 62%.','加 0.5，大约把 50% 推到 62%。'))}</div>`,
 sigma:()=>`<div>${esc(tx('σ turns log-odds back into a probability:','σ 把对数几率变回概率：'))}</div><div class="fx">${A.ui.math('\\sigma(x) = \\frac{1}{1 + e^{-x}}')}</div>`,
 order:()=>`<div>${esc(TIP.pos())}</div>`,
 lc:()=>`<div>${esc(TIP.lc())}</div><div class="fx">${A.ui.math('P(\\text{B}) = \\sigma(\\alpha + \\beta \\cdot \\ell)')}</div>`,
 judge:()=>`<div>${esc(TIP.judge())}</div>`
};
const CTLN={judge:{en:'Judge',zh:'裁判'},order:{en:'Order',zh:'顺序'},lc:{en:'Length',zh:'长度'}};
const termTipHTML=k=>{if(!TERM[k]) return ''; const a=jumpAct(k); return TERM[k]()+(a?`<div class="act">${esc(a)}</div>`:'');};
function ctlLockMsg(k){
  if(mode==='challenge'||mode==='sandbox') return '';
  if(isP('order')&&k==='order') return '';
  if(isP('length')&&k==='lc') return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({judge:$('#ljg-ic-chJ',app)||$('#ljg-ctl-judge',app),order:$('#ljg-ic-ord',app)||$('#ljg-ic-chO',app)||$('#ljg-ctl-order',app),lc:$('#ljg-ic-lc',app)||$('#ljg-ic-chL',app)||$('#ljg-ctl-lc',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#ljg-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#ljg-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>setState({...BASE,sc:S.sc})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.judge,S.order,S.lc).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='pair-next'){pairAt=(pairAt+1)%N; byLearner(render); return;}
  if(a==='to-q'){const q=$('#ljg-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#ljg-ic-ord',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-ck'){const q=$('#ljg-ic-ck',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ljg-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#ljg-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ljg-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>{if(typeof s.showMe==='function') s.showMe(); else setLc(true);}); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ljg-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('order')){const f=()=>byLearner(()=>setOrder(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.lever!=null&&isP('check')){const f=()=>byLearner(()=>setCk(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='judge') bySettings(b,()=>setState({judge:d.v}));
      else if(k==='order') bySettings(b,()=>setState({order:d.v}));
      else if(k==='lc'){if(b.closest('#ljg-controls')) bySettings(b,()=>setState({lc:d.v==='1'})); else {byLearner(()=>setLc(d.v==='1')); if(isP('length')) revealSoon();}}
      else if(k==='chJ') byLearner(()=>setState({judge:d.v}));
      else if(k==='chO') byLearner(()=>setState({order:d.v}));
      else if(k==='chL') byLearner(()=>setState({lc:d.v==='1'}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; pairAt=featOf(p.id); byLearner(()=>setState({sc:p.id})); requestAnimationFrame(()=>{if(app) revealBox($('#ljg-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ljg-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the pair shown, the orders and checks tried, the
     challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,pairAt,ordTried:[...ordTried],lcSeen,
    ckSel,ckTried:[...ckTried],remView,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.judge,S.order,S.lc).ok}),
  restore(o){if(!o||o.v!==1) return; const md=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';   // the challenge's S keeps the key order setMode gives it
    const ps=(s,ch)=>{if(!s||typeof s!=='object') return null; const c=ch&&s.sc==='ch', q=parse(token({...s,sc:c?'def':s.sc})); return q&&c?{judge:q.judge,order:q.order,lc:q.lc,sc:'ch'}:q;}, p=ps(o.S,md==='challenge'); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b;
    S=p; mode=md; step=ints(o.step,0,STEPS.length-1)?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved,false); saved=mode==='challenge'&&sv?sv:null; chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    pairAt=ints(o.pairAt,0,N-1)?o.pairAt:featOf(mode==='challenge'?'ch':S.sc);
    ordTried=new Set(['B',...(Array.isArray(o.ordTried)?o.ordTried.filter(v=>Object.keys(ORD).includes(v)):[])]); lcSeen=bool(o.lcSeen);
    ckSel=Object.keys(CK).includes(o.ckSel)?o.ckSel:'once'; ckTried=new Set(['once',ckSel,...(Array.isArray(o.ckTried)?o.ckTried.filter(v=>Object.keys(CK).includes(v)):[])]);
    remView=REMS.includes(o.remView)?o.remView:'order'; sbTouched=bool(o.sbTouched); drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:2; drillOpen=bool(o.drillOpen);},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true; pairAt=featOf(S.sc);}
    if(mode==='challenge'&&!saved) saved={...BASE};
    if(mode==='guided'&&!p&&step===0){pairAt=FEAT; Object.assign(S,STEPS[0].enter());}
    el.innerHTML=viewHTML(); app=$('#ljg-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ljg-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#ljg-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#ljg-math',app).open=true;}},
  onResize(){if(app){drawWaffle();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#ljg-bars',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#ljg-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(chHide()){vis.innerHTML=''; read.innerHTML=esc(tx(`Cost ${fint(R().cost)} small-judge calls; submit to see the gap`,`费用相当于 ${fint(R().cost)} 次小模型调用，提交后显示差距`)); return;}
      if(viewNow()==='agree'){const k=agK(), r=run('def',CK[k].j,CK[k].o);
        vis.innerHTML=`<div class="ljg-dk"><i style="width:${(r.agree*100).toFixed(1)}%"></i></div>`;
        read.innerHTML=esc(tx(`${L(CK[k])}: agrees ${fR(r.agree)} with people; a second person, on the same pairs, ${fR(r.pagree)}`,`${L(CK[k])}：和人一致 ${fR(r.agree)}，同样这些对上第二个人是 ${fR(r.pagree)}`)); return;}
      const r=R(), v=val(r,S.lc), sh=peopleShown();
      vis.innerHTML=`<div class="ljg-dk"><i style="width:${(v*100).toFixed(1)}%"></i>${sh?`<b style="left:${(r.H*100).toFixed(1)}%"></b>`:''}</div>`;
      read.innerHTML=esc(tx(`Judge: B wins ${fR(v)} (${L(ORDS[S.order])}${S.lc?', length taken out':''})`,`裁判：B 赢 ${fR(v)}（${L(ORDS[S.order])}${S.lc?'，去掉长度':''}）`)+(sh?tx(`; people ${fR(r.H)}`,`，人评 ${fR(r.H)}`):''));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const pub=r=>({w:r.w,lc:r.lc,a:r.fit.a,b:r.fit.b,H:r.H,W:r.W,T:r.T,Lo:r.Lo,inc:r.inc,sd:r.sd,se:r.se,lo:r.lo,hi:r.hi,agree:r.agree,ag:r.ag,agN:r.agN,pag:r.pag,pagree:r.pagree,cost:r.cost});
window.__ljg=Object.freeze({
  run:(sc,j,o)=>deep(pub(run(sc,j,o))), verdicts:(sc,j,o)=>deep(run(sc,j,o).v.map(x=>[x.s,x.a,x.b])), pp, chEval:(j,o,lc)=>deep(chEval(j,o,lc)), chBest:()=>deep(chBest()),
  data:seed=>deep(data(seed)), lshift, drill:s=>deep(makeDrill(s)), exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transfer:e.transfer.map(q=>({q:q.q,o:q.o,why:q.why}))});},
  feat:sc=>featOf(sc), words:(sc,i)=>deep(wordsOf(sc,i)), nBfirst,
  consts:Object.freeze({N,NL,GOAL,SC:deep(SC),JUDGES:deep(JUDGES),FEAT,STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,BASE:{...BASE},PRESETS:PRESETS.map(p=>p.id),CH_START:{...CH_START}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},pairAt,ordTried:[...ordTried],lcSeen,ckSel,ckTried:[...ckTried],remView,chSubmitted,chReveal,people:peopleShown(),stageBusy:stageT!==0})
});
})();
