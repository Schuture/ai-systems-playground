/* reward-model (world post): How does a reward model turn preferences into scores?
   Fidelity: Live. A toy reward model with 3 parameters (a linear score on 3 features of an answer) is trained in the browser with
   real gradient steps on the Bradley–Terry loss, on seeded simulated preference pairs; then it picks answers by best-of-n.
   - An answer has a true quality q (0–10, what people judge), a content reading c = q + 1.0·e (what the toy reward model can read
     of the content), a length L in tokens (in the default data longer answers are more complete), and a list flag b (bullet points).
   - People pick A with P = σ((q_A − q_B)/τ). The reward model: r = w·φ, φ = [(c − 6)/1.5, ln(L/150)/0.4, b − 0.5].
   - Training: full-batch gradient descent on mean −log σ(r_chosen − r_rejected), learning rate 0.15, from w = 0.
   - Best-of-n: the chat model is a fixed bank of 2,000 answers, 8% of them padded (3 times as long, same content, true quality −2.5);
     n draws with replacement, keep the highest score; the page shows the exact expectation over draws.
   - Random numbers: mulberry32; normal draws by Box–Muller. Seeds: training pairs 13 (challenge 45), held-out pairs 113 (46),
     balanced pairs seed + 7919, the chat model's bank 31.
   Words (EN / ZH): reward model / 奖励模型; preference pair / 偏好对; chosen / rejected 选中的 / 落选的; score / 分数;
   held-out pairs / 留出的偏好对; accuracy / 准确率; length-balanced pairs / 长短平衡的偏好对; best-of-n; true quality / 真实质量;
   chat model / 聊天模型 (the model that writes the answers; no “policy”, the reader has not met RL here); part / 部分; step / 步 (training steps only). window.__rm is a read-only hook for tests/reward-model_numbers.py and _flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='reward-model';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const gauss=r=>{const u=r(), v=r(); return Math.sqrt(-2*Math.log(1-u))*Math.cos(2*Math.PI*v);};
const sig=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const QM=6, QS=1.5, SE=1.0, PADQ=2.5, PADL=3, L0=150, LS=0.4, ETA=0.15, TMAX=400, T=200, NH=500, M=2000, PPAD=0.08;
const NS=[1,2,4,8,16,32,64,128,256];
const clampq=q=>Math.max(0.5,Math.min(9.5,q));
function resp(r,a,pad){
  let q=clampq(QM+QS*gauss(r)); const e=gauss(r), zl=gauss(r), b=r()<0.5?1:0;
  let lnL=Math.log(L0)+a*(q-QM)/QS+0.25*zl; const c=q+SE*e;
  if(pad){lnL+=Math.log(PADL); q=clampq(q-PADQ);}
  return {q,c,L:Math.exp(lnL),b,pad:!!pad};
}
const padded=x=>({q:clampq(x.q-PADQ),c:x.c,L:x.L*PADL,b:x.b,pad:true});
const phi=x=>[(x.c-QM)/QS,Math.log(x.L/L0)/LS,x.b-0.5];
const score=(w,x)=>{const f=phi(x); return w[0]*f[0]+w[1]*f[1]+w[2]*f[2];};
/* data sets: rel = how length follows quality (0 = unrelated), tau = how noisy people are, N = pairs */
const RELS={tied:0.35,none:0,strong:0.6};
function mkpairs(seed,n,a,tau,bal){
  const r=mulberry(seed), P=[];
  for(let i=0;i<n;i++){const X=resp(r,a), Y=resp(r,a), u=r(); P.push([X,Y,u<sig((X.q-Y.q)/tau)?0:1]);}
  const r2=mulberry(seed+7919);
  for(let i=0;i<bal;i++){const X0=resp(r2,a), Y0=padded(X0), u=r2(), s=r2()<0.5; const X=s?X0:Y0, Y=s?Y0:X0; P.push([X,Y,u<sig((X.q-Y.q)/tau)?0:1,1]);}
  return P;
}
function train(P,Tn){
  let w=[0,0,0]; const W=[w.slice()], LOSS=[];
  const D=P.map(([X,Y,ch])=>{const fw=phi(ch===0?X:Y), fl=phi(ch===0?Y:X); return [fw[0]-fl[0],fw[1]-fl[1],fw[2]-fl[2]];});
  const n=D.length;
  for(let t=0;t<=Tn;t++){
    const g=[0,0,0]; let Ls=0;
    for(const d of D){const m=w[0]*d[0]+w[1]*d[1]+w[2]*d[2]; Ls+=m>-35?Math.log1p(Math.exp(-m)):-m; const s=sig(-m); g[0]-=s*d[0]; g[1]-=s*d[1]; g[2]-=s*d[2];}
    LOSS.push(Ls/n); if(t===Tn) break;
    w=[w[0]-ETA*g[0]/n,w[1]-ETA*g[1]/n,w[2]-ETA*g[2]/n]; W.push(w.slice());
  }
  return {W,LOSS};
}
/* a tie (only at step 0, when every score is 0) counts as half right */
const accuracy=(P,w)=>{let ok=0; for(const [X,Y,ch] of P){const a=score(w,X), b=score(w,Y); if(a===b) ok+=0.5; else if(ch===0?a>b:b>a) ok++;} return ok/P.length;};
/* the best a reward model could do on these labels: pick the answer with the higher true quality */
const ceiling=P=>P.filter(([X,Y,ch])=>(X.q>Y.q)===(ch===0)).length/P.length;
let BANK=null;
function bank(){if(BANK) return BANK; const r=mulberry(31), out=[]; for(let i=0;i<M;i++){const pd=r()<PPAD; out.push(resp(r,RELS.tied,pd));} return BANK=out;}
/* exact expectation of best-of-n over draws with replacement from the bank; equal scores share their rank group's mass */
function bon(w){
  const B=bank(), sc=B.map(y=>[score(w,y),y.q,y.L,y.pad?1:0]).sort((x,y)=>x[0]-y[0]), groups=[];
  for(let j=0;j<M;){let k=j; while(k+1<M&&sc[k+1][0]===sc[j][0]) k++; groups.push([j,k+1]); j=k+1;}
  return NS.map(n=>{let Es=0,Eq=0,EL=0,Ep=0;
    for(const [j0,j1] of groups){const p=Math.pow(j1/M,n)-Math.pow(j0/M,n), m=j1-j0; let s=0,q=0,l=0,pp=0; for(let j=j0;j<j1;j++){s+=sc[j][0]; q+=sc[j][1]; l+=sc[j][2]; pp+=sc[j][3];} Es+=p*s/m; Eq+=p*q/m; EL+=p*l/m; Ep+=p*pp/m;}
    return {n,s:Es,q:Eq,L:EL,pad:Ep};});
}
/* the data set of a setting; the challenge has its own data (seeds 45 / 46, length tied more strongly) */
const dsKey=s=>s.ch?'ch':[s.rel,s.tau,s.N].join('/');
function dsOf(s){return s.ch?{a:RELS.strong,tau:1,N:1000,seed:45,hseed:46}:{a:RELS[s.rel],tau:s.tau,N:s.N,seed:13,hseed:113};}
const HELD=new Map(), RUNS=new Map(), BONS=new Map();
function held(s){const k=dsKey(s); if(!HELD.has(k)){const d=dsOf(s); HELD.set(k,mkpairs(d.hseed,NH,d.a,d.tau,0));} return HELD.get(k);}
function run(s){
  const k=dsKey(s)+'|'+s.bal; if(RUNS.has(k)) return RUNS.get(k);
  const d=dsOf(s), P=mkpairs(d.seed,d.N,d.a,d.tau,s.bal), H=held(s), tr=train(P,TMAX);
  const acc=tr.W.map(w=>accuracy(H,w));
  const r={k,P,H,W:tr.W,LOSS:tr.LOSS,acc,ceil:ceiling(H),N:d.N,bal:s.bal};
  RUNS.set(k,r); return r;
}
function bonAt(s,t){const k=dsKey(s)+'|'+s.bal+'|'+t; if(!BONS.has(k)) BONS.set(k,bon(run(s).W[t])); return BONS.get(k);}
const peakOf=b=>{let k=0; b.forEach((x,i)=>{if(x.q>b[k].q+1e-12) k=i;}); return k;};
/* the example pairs (示例): features set for this lab; the texts are illustrative */
const EX=[
 {A:{q:7.6,c:7.4,L:200,b:1},B:{q:4.6,c:4.8,L:70,b:0},ch:0},
 {A:{q:7.5,c:7.3,L:120,b:0},B:{q:5.0,c:7.3,L:360,b:0},ch:0}];
/* the challenge: new data, best-of-16, true quality ≥ 7.6 with the fewest balanced pairs. The held-out accuracy is shown, not graded:
   500 pairs move it by about ±4 points (review S13); 7.6 leaves 50 pairs below the goal and 100 above it by more than the seed-to-seed
   spread (numbers test, multi-seed check) */
const CH_N=16, CH_Q=7.6, CH_BAL=[0,25,50,100,200], CH_STEPS=[50,100,200,400], CH_START={bal:0,steps:200};
function chEval(bal,steps){const s={ch:true,bal}, r=run(s), b=bonAt(s,steps), q=b[NS.indexOf(CH_N)].q, acc=r.acc[steps];
  const okQ=q>=CH_Q-1e-12; return {bal,steps,acc,q,okQ,ok:okQ,chg:(bal!==CH_START.bal?1:0)+(steps!==CH_START.steps?1:0)};}
/* the challenge has been submitted and meets the goal right now */
const chPassed=()=>mode==='challenge'&&chSubmitted&&chEval(chSt.bal,chSt.steps).ok;
/* the fixed example 2 under the challenge's reward model, at the steps picked: does it still score the padded B higher? */
function chEx2Wrong(){const w=run({ch:true,bal:chSt.bal}).W[chSt.steps], e=EX[1], a=score(w,e.A), b=score(w,e.B); return sg2(a)!==sg2(b)&&b>a;}   // as the pair card rounds it
function chBest(){let b=null; CH_BAL.forEach(bal=>CH_STEPS.forEach(st=>{const e=chEval(bal,st); if(e.ok&&(!b||e.chg<b.chg||(e.chg===b.chg&&e.bal<b.bal))) b=e;})); return b;}

/* =====================================================================
   formatting
   ===================================================================== */
const MINUS='−';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fint=A.fmt.int;
const sg2=v=>{const s=fixd(Math.abs(v),2); return (v<0&&+s!==0?MINUS:'')+s;};
const pP=v=>fixd(v*100,1)+'%', pR=v=>Math.round(v*100)+'%', q2=v=>fixd(v,2), gd=v=>(v<0?MINUS:'')+String(Math.abs(v));
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const DEF=(text,tip)=>A.ui.def(text,tip);
const J_=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
/* a lab is linked when it is open; a planned one is named in plain text (S10) */
const lk=id=>{const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; if(!l) return id; return c.labStatus&&c.labStatus(l)==='live'&&A.shell.labLink?A.shell.labLink(id):esc(L(l.s||{en:id,zh:id}));};
const L8='Llama‑3.1‑8B';
const SRC={
 trl:'https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/reward_trainer.py',
 trlc:'https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/reward_config.py',
 hf:'https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/modeling_layers.py',
 hh:'https://github.com/anthropics/hh-rlhf/blob/c72f5cee8eb7b4d2ea5617657f4430d5e333af07/README.md',
 rb:'https://github.com/allenai/reward-bench/blob/51fcd846e09998c3a5bdd15f5ec9932645f25b4d/scripts/run_rm.py',
 ae:'https://github.com/tatsu-lab/alpaca_eval/blob/cd543a149df89434d8a54582c0151c0b945c3d20/README.md',
 igpt:'https://arxiv.org/abs/2203.02155', summ:'https://arxiv.org/abs/2009.01325', gao:'https://arxiv.org/abs/2210.10760',
 orl:'https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/README.md',
 hfb:'https://github.com/huggingface/blog/blob/998ee73e0594a21b42df14632263266a2831dafe/rlhf.md',
 rlb:'https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/book/chapters/05-reward-models.md'};

/* =====================================================================
   module state
   ===================================================================== */
const BASE={rel:'tied',tau:1,N:1000,bal:0,ni:4,pair:0,shift:0,d:2};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let tShow=T, playing=false, spd='mid', raf=0, lastT=0, relang=false, prog=1;
const watched=new Set();
let wTried=new Set(['tied']), fixTried=new Set([0]), shiftMoved=false, dMoved=false, nMax=4, hackSeen=0;   // hackSeen: the largest n (index) reached in part 6
let restored=false;   // restore() ran before this mount (round 3 #6)
let saved=null, chSubmitted=false, chReveal=false, chSt={...CH_START}, sbTouched=false, drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false;
const shownPrev={};
const runS=()=>mode==='challenge'?{ch:true,bal:chSt.bal}:S;
const Tn=()=>mode==='challenge'?chSt.steps:T;
const R=()=>run(runS());
const runKey=()=>dsKey(runS())+'|'+runS().bal+'|'+Tn();
const tNow=()=>Math.max(0,Math.min(Tn(),Math.round(tShow)));
const wNow=()=>R().W[tNow()];

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 /* G3 (the owner's review of 2026-10-08): what a reward model is, why it is needed and how it is used are in part 1's text; the tip
    only adds the size of the score layer and a real pairing of sizes */
 rm:()=>tx(`Reward model: an LLM that reads a prompt and a response and gives one score, trained on preference pairs. Its score layer is tiny: for ${L8}, the LM head’s 525M parameters become 4,096. It can be smaller than the chat model: InstructGPT’s 175B model was trained with a 6B reward model.`,
           `奖励模型：读进提问和回答、给出一个分数的 LLM，用偏好对训练出来。它的打分层很小，对 ${L8} 来说，LM head 的 525M 个参数换成了 4,096 个。它也可以比聊天模型小，InstructGPT 的 175B 模型就是用一个 6B 的奖励模型训练的。`),
 pair:()=>tx('A preference pair: two responses to the same prompt, and the one a person chose. The chosen response and the rejected response.','偏好对：同一个提问的两个回答，标注的人选了其中一个。被选的叫选中的回答，另一个叫落选的回答。'),
 bt:()=>tx('Bradley–Terry model: the chance that a person prefers A to B depends only on the difference of their scores, through σ. It goes back to Bradley and Terry, 1952.','Bradley–Terry 模型：人选 A 而不选 B 的概率，只取决于两者分数的差，经过 σ 变成概率。它来自 Bradley 和 Terry 1952 年的论文。'),
 sigma:()=>{const s0=A.ui.math('\\sigma(0) = 50%'), s1=A.ui.math('\\sigma(1) \\approx 73%'); return tx(`σ (sigmoid) turns any number into a probability between 0 and 1: ${s0}, ${s1}, and large numbers come close to 100%.`,`σ（sigmoid）把任意一个数变成 0 到 1 之间的概率：${s0}，${s1}，数越大越接近 100%。`);},
 held:()=>tx('Held-out pairs: 500 preference pairs from the same kind of data that training never sees. Accuracy is the share of them in which the reward model scores the chosen response higher than the rejected one.','留出的偏好对：同一类数据里另外 500 对，训练时从来不用。准确率是其中奖励模型给选中的回答打分比落选的高的比例。'),
 tau:()=>tx('How noisy people are (τ): the larger τ, the more often people pick the worse response. With τ = 1, a 1-point better answer is picked 73% of the time; with τ = 3, only 58%. Set for this lab.','标注的人有多不一致（τ）：τ 越大，人越常选中差的那个回答。τ = 1 时，好 1 分的回答有 73% 的时候被选中；τ = 3 时只有 58%。这是这个 Lab 设定的。'),
 bal:()=>tx('Length-balanced pairs: two versions of the same response, one concise and one padded with filler to 3 times the length. The content is the same, so the padded one is worse; people nearly always pick the concise one.','长短平衡的偏好对：同一个回答的两个版本，一个简洁，一个加了废话、长了 3 倍。内容一样，加了废话的更差，标注的人几乎总选简洁的那个。'),
 bon:()=>tx('Best-of-n: the chat model writes n responses to one prompt, the reward model scores them, and the highest-scored one is kept. It changes no parameter of the chat model.','best-of-n：聊天模型对同一个提问写 n 个回答，奖励模型给它们打分，留下分数最高的那个。它不改聊天模型的任何参数。'),
 hack:()=>tx('Reward hacking: optimising against a reward model finds what the score rewards but people do not want, so the score keeps rising while the true quality falls.','reward hacking：照着奖励模型的分数去优化，会找到分数喜欢、人却不想要的东西，于是分数一直涨，真实质量却在降。'),
 q:()=>tx('True quality: how good a response really is, out of 10. In this lab it is known for every response; people judge by it, with noise; the reward model never sees it.','真实质量：一个回答到底有多好，满分 10 分。在这个 Lab 里，每个回答的真实质量都是已知的。标注的人按它来选，但会有随机的分歧；奖励模型看不到它。'),
 /* one feature each (R2.11): the weight bars' tooltips, and the aligned list of part 5's term “three features” */
 feat:i=>[tx('This feature is what the small reward model reads of the content: the true quality plus a reading error (standard deviation 1 point).','这一项是小奖励模型从文字里读出的内容好坏，等于真实质量加上一个读错的量（标准差 1 分）。'),
   tx('The length feature grows with the log of the length: doubling the length adds 1.7.','长度特征随长度的对数增长：长度翻倍，这一项加 1.7。'),
   tx('This feature is +0.5 with bullet points and −0.5 without.','用了列表时这一项是 +0.5，没用时是 −0.5。')][i]
};
const TD=(k,x)=>`<span class="t rm-td" tabindex="0" data-ctl="${k}">${x}</span>`;
const FX=t=>`<div class="fx">${A.ui.math(t)}</div>`;
/* H2 (the owner, 2026-10-09): a formula with its name in words on the left, its chain broken at “=” onto lines that line up */
/* the owner's review of 2026-10-09 (no Show the math row scrolls sideways): a line too long for its box wraps before a top-level “+” or
   “−”, each new line under the line's first term; {} gives the piece's leading operator the spacing of a binary operator. On a phone
   the name goes on its own line above the chain (reward-model.css). */
const topSplit=t=>{const out=[]; let d=0, s=0; for(let i=0;i<t.length;i++){const c=t[i]; if('{(['.includes(c)) d++; else if('})]'.includes(c)) d--;
  else if(!d&&c===' '&&i>s&&(t.startsWith(' + ',i)||t.startsWith(' - ',i))){out.push(t.slice(s,i)); s=i+1;}} out.push(t.slice(s)); return out;};
const WRP=t=>topSplit(t).map((p,i)=>`<span>${A.ui.math(i?`{} ${p}`:p)}</span>`).join('');
const EQ=(name,lines,op='=')=>`<div class="rm-eq">${lines.map((l,i)=>{const [o,t]=Array.isArray(l)?l:[op,l]; return `<div class="rm-eq-n">${i?'':name}</div><div class="rm-eq-o">${o?A.ui.math(o):''}</div><div class="fx rm-wr">${WRP(t)}</div>`;}).join('')}</div>`;
/* H1 (the owner, 2026-10-09): the scores and their gap typeset, never written as strings */
const RA=()=>A.ui.math('r_A'), RB=()=>A.ui.math('r_B'), GAPM=()=>A.ui.math('r_A - r_B');
const BTF=()=>A.lang()==='zh'?'P(\\text{人选 A}) = \\sigma(r_A - r_B)':'P(\\text{person picks A}) = \\sigma(r_A - r_B)';
const LOSSF=()=>A.lang()==='zh'?'\\text{loss} = -\\log\\, \\sigma(r_{\\text{选中}} - r_{\\text{落选}})':'\\text{loss} = -\\log\\, \\sigma(r_{\\text{chosen}} - r_{\\text{rejected}})';
/* Δφ (defined beside the formula) keeps the line inside a phone's width */
const GRADF=()=>A.lang()==='zh'?'w \\leftarrow w + \\eta \\cdot \\text{mean}\\big(\\sigma(-(r_{\\text{选中}} - r_{\\text{落选}})) \\cdot Δ\\phi\\big)'
  :'w \\leftarrow w + \\eta \\cdot \\text{mean}\\big(\\sigma(-(r_{\\text{chosen}} - r_{\\text{rejected}})) \\cdot Δ\\phi\\big)';
const BONF=()=>(A.lang()==='zh'?'P(\\text{挑中第\u00a0}j\\text{\u00a0个})':'P(\\text{pick the\u00a0}j\\text{-th})')+' = \\frac{j^n - (j-1)^n}{M^n}';
/* the symbols of a formula, one per line (R2.11): [symbol, meaning] */
const DEFS=rows=>`<ul class="rm-defs small">${rows.map(([k,v])=>`<li><b>${k}</b><span>${v}</span></li>`).join('')}</ul>`;
const SCOREF=()=>A.lang()==='zh'?'r = w_1 \\cdot \\text{内容} + w_2 \\cdot \\text{长度} + w_3 \\cdot \\text{列表}':'r = w_1 \\cdot \\text{content} + w_2 \\cdot \\text{length} + w_3 \\cdot \\text{list}';

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'a',t:{en:'Under 65%',zh:'不到 65%'}},{v:'b',t:{en:'65% to 80%',zh:'65% 到 80%'}},{v:'c',t:{en:'80% to 95%',zh:'80% 到 95%'}},{v:'d',t:{en:'Over 95%',zh:'95% 以上'}}];
const PRED_RIGHT='c';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const dflt=()=>run({...BASE});
const dB=()=>bonAt({...BASE},T);
const WOPT=[{v:'tied',t:{en:'Longer answers were often chosen',zh:'数据里长回答常被选中'},d:{en:'the initial setup: longer answers are usually more complete',zh:'初始设置：越完整的回答往往越长'}},
  {v:'none',t:{en:'Length unrelated to quality',zh:'长度和质量无关的数据'},d:{en:'the same people, but length no longer follows quality',zh:'同样的标注，只是长度不再跟着质量走'}}];
const wRes=v=>tx(`length weight ${sg2(run({...BASE,rel:v}).W[T][1])}`,`长度的权重 ${sg2(run({...BASE,rel:v}).W[T][1])}`);
const FOPT=[0,50,200];
const fRes=v=>tx(`n = 256: ${q2(bonAt({...BASE,bal:v},T)[8].q)} pts`,`n = 256：${q2(bonAt({...BASE,bal:v},T)[8].q)} 分`);
const fTitle=v=>v===0?tx('No balanced pairs (the initial setup)','不加（初始设置）'):tx(`Add ${v} balanced pairs`,`加 ${v} 对长短平衡的`);
const fDesc=v=>v===0?tx('only the original 1,000 pairs','只用原来的 1,000 对'):tx(`${fint(1000+v)} pairs in all`,`一共 ${fint(1000+v)} 对`);
const levers=(id,aria,val,opts)=>`<div class="rm-lv" id="${id}">`+A.ui.levers({aria,value:val,options:opts})+`</div>`;
const wCtl=()=>levers('rm-ic-w',tx('Which data to train on, pick one','用哪批数据训练，选一个'),S.rel,WOPT.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(wTried.has(o.v)?wRes(o.v):'')})));
const fCtl=()=>levers('rm-ic-fix',tx('How many balanced pairs to add, pick one','加多少对长短平衡的数据，选一个'),String(S.bal),FOPT.map(v=>({v:String(v),title:fTitle(v),desc:fDesc(v),result:esc(fixTried.has(v)?fRes(v):'')})));
const dCtl=id=>A.ui.range({id,key:'d',label:tx('A’s score minus B’s','A 的分数减 B 的分数'),min:-4,max:4,step:0.5,value:S.d,out:sg2(S.d).replace(/\.00$/,''),inline:true});
const shiftCtl=id=>A.ui.range({id,key:'shift',label:tx('Add the same number to every score','给所有分数都加上同一个数'),min:-3,max:3,step:0.5,value:S.shift,out:(S.shift>0?'+':'')+sg2(S.shift).replace(/\.00$/,''),inline:true});
const nCtl=id=>A.ui.range({id,key:'ni',label:tx('n: responses to pick from','n：从几个回答里挑'),min:0,max:NS.length-1,step:1,value:S.ni,out:String(NS[S.ni]),inline:true});
const playBtn=()=>`<div class="stepnav rm-cplay"><button type="button" class="btn sm" data-act="play" id="rm-ic-play">${playLabel()}</button></div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  /* the owner's review of 2026-10-08: no PPO here (the reader has not met it); the lab is about learning a preference from labelled
     pairs, and part 1's text says what a reward model is, why it is needed and how it is used once trained (G3) */
  lead:()=>tx('People find it hard to score one response but easy to compare two. A reward model learns their preference from such labelled comparisons.','人很难给一个回答打分，比较两个回答却很容易。奖励模型就从标注好的比较里学到人的偏好。'),
  enter:()=>({...BASE,pair:0}),
  body:()=>tx(`<p>Asking people to score a response is hard: is it worth 7 or 8? Another person, or the same person on another day, says something else. Comparing is easy: shown two responses to the same prompt, people quickly say which is better. Such a labelled comparison is a ${DEF('preference pair',TIP.pair())}.</p>`+
              `<p>A ${DEF('reward model',TIP.rm())} learns people’s preference from these pairs. It is itself an LLM, usually the chat model after ${lk('sft')}, sometimes a smaller one. Only its last layer differs: it reads a prompt and a response and outputs one score, not a probability for every next token.</p>`+
              `<p>People can label only so many pairs, but a reward model scores any new response automatically. Its scores pick the best of several responses (as this lab does later), or become the score that later training raises.</p>`+
              `<p>The chart shows an example: the person chose A. This lab trains a small reward model with 3 parameters for real, in your browser.</p>`,
             `<p>请人给一个回答打绝对分很难：这个回答该给 7 分还是 8 分？换一个人，或者同一个人换一天，给的分都不一样。比较却容易，同一个提问的两个回答摆在一起，人很快就能说出哪个更好。这样标注好的一次比较，叫一个${DEF('偏好对',TIP.pair())}。</p>`+
             `<p>${DEF('奖励模型',TIP.rm())}就从这些偏好对里学到人的偏好。它本身也是一个 LLM，通常就是 ${lk('sft')} 之后的聊天模型，有时用一个小一些的。不同的只是最后一层，它不再给下一个 token 算概率，而是读完提问和回答，只输出一个分数。</p>`+
             `<p>人能标的偏好对有限，有了奖励模型，聊天模型写出的任何新回答都能自动打分。打出的分可以从几个回答里挑出最好的（这个 Lab 后面就这样用），也可以在后续训练里，当作要让模型提高的分数。</p>`+
             `<p>图上是一个示例，标注的人选了 A。这个 Lab 用一个只有 3 个参数的小奖励模型，在你的浏览器里真正训练它。</p>`)},
 {id:'bt',title:{en:'From a score gap to a choice',zh:'分差怎样变成“选它”的概率'},gate:()=>!!pred,
  lead:()=>tx('The gap between two scores, passed through σ, is the chance a person picks A.','两个分数的差，经过 σ，就是人选 A 的概率。'),
  enter:()=>({...BASE,pair:0,d:2}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>The reward model gives A a score ${RA()} and B a score ${RB()}. The ${DEF('Bradley–Terry model',TIP.bt())} says the chance that a person picks A depends only on the gap:</p>${FX(BTF())}${FX('\\sigma(x) = \\frac{1}{1 + e^{-x}}')}<p>${TD('sigma','σ')} turns the gap into a probability. With a gap of 0 it gives 50%: either answer is as likely. With a gap of 1 it gives about 73%. (${lk('llm-judge')} named this model; here it gets trained.)</p><p>A’s score is 2 higher than B’s. About how likely is a person to pick A?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it; the result shows below.</p>':''}`,
              `<p>奖励模型给 A 打一个分 ${RA()}，给 B 打一个分 ${RB()}。${DEF('Bradley–Terry 模型',TIP.bt())}假设，人选 A 的概率只取决于两个分数的差：</p>${FX(BTF())}${FX('\\sigma(x) = \\frac{1}{1 + e^{-x}}')}<p>${TD('sigma','σ')} 把分差变成概率。分差为 0 时是 50%，两个回答被选中的机会一样。分差为 1 时约是 73%。（${lk('llm-judge')}里提到过这个模型，这里要真正训练它。）</p><p>A 的分数比 B 高 2，人选 A 的概率大约是多少？</p>${ch}${pred?'<p class="muted small">已记下，下面会显示结果，之前还可以改。</p>':''}`);},
  always:true,obs:()=>obsBt()},
 {id:'train',title:{en:'Train on preference pairs',zh:'用偏好对训练'},
  lead:()=>tx('Each step raises the chosen response’s score and lowers the rejected one’s.','每一步都让选中的回答分数升高、落选的降低。'),
  enter:()=>({...BASE,pair:0}),expect:()=>watched.has(keyOf({...BASE},T)),showMe:()=>finishRun(),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The training data are 1,000 preference pairs. At every step the reward model scores both responses of every pair, and its parameters move a little toward “chosen up, rejected down”, which makes this loss smaller (log is the natural logarithm):</p>${FX(LOSSF())}<p>The further the chosen response’s score is above the rejected one’s, the smaller the loss.</p><p>Press Replay and watch how the loss and the accuracy on ${DEF('500 held-out pairs',TIP.held())} change.</p>`+playBtn(),
              `<p>训练数据是 1,000 个偏好对。每一步，奖励模型给每一对的两个回答各打一个分，再把参数往“选中的升、落选的降”的方向挪一点，让这个 loss 变小（log 是自然对数）：</p>${FX(LOSSF())}<p>选中的回答分数比落选的高得越多，loss 就越小。</p><p>点“重放”，看 loss 和${DEF('留出的 500 对',TIP.held())}上的准确率怎么变。</p>`+playBtn()),
  obs:()=>obsTrain()},
 {id:'shift',title:{en:'Only differences mean something',zh:'只有分差有意义'},
  lead:()=>tx('Add the same number to every score, and nothing changes.','给所有分数加上同一个数，什么都不变。'),
  enter:()=>({...BASE,pair:0,shift:0}),expect:()=>shiftMoved,showMe:()=>setShift(2),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The loss and σ only ever see the gap ${GAPM()}. Move every score by the same amount and see what changes:</p>`+shiftCtl('rm-ic-shift'),`<p>loss 和 σ 里只出现分差 ${GAPM()}。把所有分数挪同样多，看看什么会变：</p>`+shiftCtl('rm-ic-shift')),
  obs:()=>obsShift()},
 {id:'weights',title:{en:'What it learned',zh:'它学到了什么'},
  lead:()=>tx('Length got a weight as large as content’s, because longer answers were often chosen.','长度拿到了和内容一样大的权重，因为数据里长回答常被选中。'),
  enter:()=>({...BASE,pair:0}),expect:()=>wTried.has('none'),showMe:()=>setRel('none'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The small reward model’s score is a weighted sum of ${TD('feat','three features')}:</p>${FX(SCOREF())}<p>People judged only the ${DEF('true quality',TIP.q())}, never the length. But the small reward model cannot see the true quality: it reads the content from the text, and makes reading errors.</p><p>In these pairs, more complete answers tend to be longer, so length is a second clue to which response gets chosen. That is why length got a weight as large as content’s (the bars in the chart).</p><p>Train once more on data where length does not follow quality, and compare:</p>`+wCtl(),
              `<p>小奖励模型的分数，是${TD('feat','三个特征')}的加权和：</p>${FX(SCOREF())}<p>标注的人只看${DEF('真实质量',TIP.q())}，从来不看长度。可小奖励模型看不到真实质量，只能从文字里读内容，还会读错。</p><p>在这批数据里，越完整的回答往往越长，所以长度也成了猜“哪个会被选中”的线索。于是长度拿到了和内容一样大的权重（图里的横条）。</p><p>换一批长度和质量无关的数据，再训练一次，比一比：</p>`+wCtl()),
  obs:()=>obsWeights()},
 {id:'hack',title:{en:'Pick answers with it: the score rises, the quality falls',zh:'拿它挑回答：分数在涨，质量在降'},
  lead:()=>tx('Best-of-n keeps the top-scored of n responses. With large n, the long empty ones win.','best-of-n 从 n 个回答里留分数最高的。n 一大，赢的是又长又空的回答。'),
  enter:()=>({...BASE,pair:1,ni:0}),expect:()=>nMax>=6,showMe:()=>setN(6),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The simplest use of a trained reward model is ${DEF('best-of-n',TIP.bon())}: the chat model writes n responses to one prompt, the reward model scores them, and the top one is kept.</p><p>In this lab, the chat model’s responses come from a fixed set of 2,000. 8% of them are padded: the same content with filler, 3 times as long. Filler adds nothing to the content reading, and costs 2.5 points of true quality. Raise n from 1 to 256:</p>`+nCtl('rm-ic-n'),
              `<p>训练好的奖励模型，最简单的用法是 ${DEF('best-of-n',TIP.bon())}：聊天模型对同一个提问写 n 个回答，奖励模型给它们打分，留下分数最高的那个。</p><p>这个 Lab 里，聊天模型能写出的回答固定是 2,000 个，其中 8% 加了废话：内容一样，长度是 3 倍。废话不会让内容的读数变高，真实质量却低 2.5 分。把 n 从 1 调到 256：</p>`+nCtl('rm-ic-n')),
  obs:()=>obsHack()},
 {id:'fix',title:{en:'Add length-balanced pairs',zh:'补上长短平衡的数据'},
  lead:()=>tx('Pairs in which the shorter response is better pull the length weight down.','补上“短的更好”的偏好对，长度的权重就降下来。'),
  enter:()=>({...BASE,pair:1,ni:8}),expect:()=>fixTried.size>1,showMe:()=>setBal(50),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>To stop the reward model from liking length, show it pairs in which length is not quality: ${DEF('length-balanced pairs',TIP.bal())}. Each one is the same response twice, concise and padded; people nearly always pick the concise one.</p><p>Add some to the original 1,000 pairs and train again. Pick an amount and see how the true quality of the pick at n = 256 changes.</p>`+fCtl(),
              `<p>要让奖励模型不再偏爱长度，就给它看长度不等于质量的例子：${DEF('长短平衡的偏好对',TIP.bal())}。每一对都是同一个回答的两个版本，一个简洁，一个加了废话，标注的人几乎总选简洁的那个。</p><p>在原来的 1,000 对上加一些，再训练一次。选一个数量，看看 n = 256 时挑出的回答，真实质量怎么变。</p>`+fCtl()),
  obs:()=>obsFix()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('A reward model learns scores from comparisons, learns only the preferences in its data, and gets gamed when you optimise against it too hard.','奖励模型从比较里学打分，只学数据里有的偏好。按它的分数优化得太狠，它就会被钻空子。'),
  enter:()=>({...BASE,pair:1,ni:8}),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsBt(){
  if(!pred) return '';
  const d=S.d, p=sig(d);
  return tx(`<p><b>A gap of 2 gives ${A.ui.math(`\\sigma(2) = ${pR(sig(2))}`)}: a person picks A ${pR(sig(2))} of the time.</b></p>`,`<p><b>分差 2：${A.ui.math(`\\sigma(2) = ${pR(sig(2))}`)}，人有 ${pR(sig(2))} 的时候选 A。</b></p>`)+verdict()+
    tx(`<p>Why not near 100%: σ flattens toward both ends. From a gap of 0 to 1 the chance rises by 23 points; from 1 to 2, by only 15 more. However large the gap, it only comes closer to 100%.</p><p>Drag the gap and follow the point on the curve. Now: a gap of ${gd(d)} gives ${pR(p)}.</p>`,
       `<p>为什么不接近 100%：σ 越往两头越平。分差从 0 到 1，概率升了 23 个百分点；从 1 到 2，只再升 15 个百分点。分差再大，也只是越来越接近 100%。</p><p>拖动分差，看曲线上的点。现在分差是 ${gd(d)}，概率是 ${pR(p)}。</p>`)+dCtl('rm-ic-d');
}
function obsTrain(){
  /* polish r4: “the reward model now says a person picks A 90% of the time” left the result box: the example on the stage says it
     (“分差 … 2.15：奖励模型认为人有 90% 的时候选 A”) */
  const r=dflt();
  return tx(`<p><b>After ${T} steps the loss falls from ${q2(r.LOSS[0])} to ${q2(r.LOSS[T])}, and on the 500 held-out pairs the reward model is right ${pR(r.acc[T])} of the time.</b></p><p>In the example pair, A’s score rose and B’s fell.</p><p>${pR(r.acc[T])} may sound low. But people disagree with each other: even guessing these 500 labels from the ${DEF('true quality',TIP.q())} gets only ${pR(r.ceil)} of them right. And 500 pairs are a small sample: another 500 would move the accuracy by about ±${fixd(196*Math.sqrt(r.acc[T]*(1-r.acc[T])/NH),0)} points (the lab ${lk('eval-ci')} shows why).</p>`,
            `<p><b>训练 ${T} 步后，loss 从 ${q2(r.LOSS[0])} 降到 ${q2(r.LOSS[T])}。在留出的 500 对里，奖励模型判对了 ${pR(r.acc[T])}。</b></p><p>示例里，A 的分数升了，B 的降了。</p><p>${pR(r.acc[T])} 听起来不高。可标注的人自己也会不一致：就算按${DEF('真实质量',TIP.q())}去猜这 500 对标注，也只能猜对 ${pR(r.ceil)}。500 对也只是一个小样本，换一批 500 对，准确率大约会上下变动 ±${fixd(196*Math.sqrt(r.acc[T]*(1-r.acc[T])/NH),0)} 个百分点（原因见${lk('eval-ci')}）。</p>`);
}
function exScores(){const w=dflt().W[T]; return [score(w,EX[0].A),score(w,EX[0].B)];}
function obsShift(){
  const [a,b]=exScores(), c=S.shift, p=sig(a-b);
  if(!shiftMoved) return '';
  const cc=(c>=0?'+':'')+sg2(c).replace(/\.00$/,'');
  return tx(`<p><b>Every score ${cc}: A goes from ${sg2(a)} to ${sg2(a+c)}, B from ${sg2(b)} to ${sg2(b+c)}. The gap stays ${sg2(a-b)}, and the reward model’s probability that a person picks A stays ${pR(p)}.</b></p><p>The loss and the accuracy do not change either, because both use only gaps. So a score has no fixed zero: “2 points” says nothing until you compare it with another response’s score, and scores of two different reward models cannot be compared.</p><p>To keep scores easy to use, some training adds a small term that pulls the average score toward 0. In TRL that is <code>center_rewards_coefficient</code> (recommended 0.01).</p>`,
            `<p><b>所有分数都 ${cc}：A 从 ${sg2(a)} 变成 ${sg2(a+c)}，B 从 ${sg2(b)} 变成 ${sg2(b+c)}。分差还是 ${sg2(a-b)}，奖励模型算出的概率还是 ${pR(p)}。</b></p><p>loss 和准确率也都不变，因为它们只用到分差。所以分数没有固定的零点：“2 分”本身说明不了好坏，要和另一个回答的分数比才有意义。两个不同的奖励模型，分数也不能直接比。</p><p>为了让分数好用，有的训练会在 loss 里加一小项，把平均分往 0 拉。TRL 里就是 <code>center_rewards_coefficient</code>（推荐 0.01）。</p>`);
}
function obsWeights(){
  const a=run({...BASE}).W[T], b=run({...BASE,rel:'none'}).W[T];
  if(!wTried.has('none')) return '';
  return tx(`<p><b>On data where length does not follow quality, the length weight falls from ${sg2(a[1])} to ${sg2(b[1])}: almost nothing.</b> The content weight rises from ${sg2(a[0])} to ${sg2(b[0])}.</p><p>A reward model learns only the preferences that are in its data, including the ones nobody meant to put there. The LLM judge in ${lk('llm-judge')} shows the same length preference. In AlpacaEval’s human labels, when one of two outputs was more than 30 characters longer, people picked the longer one 64% of the time, so real preference data does carry length.</p>`,
            `<p><b>换成长度和质量无关的数据，长度的权重从 ${sg2(a[1])} 降到 ${sg2(b[1])}，几乎是 0。</b>内容的权重从 ${sg2(a[0])} 升到 ${sg2(b[0])}。</p><p>奖励模型只学数据里有的偏好，也包括谁都没打算放进去的那些。LLM 裁判也有同样的长度偏好，见“${lk('llm-judge')}”这个 Lab。在 AlpacaEval 的人工标注里，两个输出长度相差超过 30 个字符时，人有 64% 的时候选了更长的那个，所以真实的偏好数据里确实带着长度。</p>`);
}
function obsHack(){
  const b=dB(), k=peakOf(b), i=S.ni, x=b[i];
  if(i<=k) return i===0?tx(`<p><b>n = 1, a random response: true quality ${q2(x.q)} out of 10, score ${sg2(x.s)}.</b> Raise n.</p>`,`<p><b>n = 1，相当于随便拿一个回答：真实质量 ${q2(x.q)} 分，分数 ${sg2(x.s)}。</b>把 n 调大。</p>`)
    :tx(`<p><b>n = ${NS[i]}: the pick’s true quality is ${q2(x.q)} out of 10 (n = 1 gives ${q2(b[0].q)}), and its score is ${sg2(x.s)}.</b> Keep raising n.</p>`,`<p><b>n = ${NS[i]}：挑中的回答的真实质量是 ${q2(x.q)} 分（n = 1 时是 ${q2(b[0].q)} 分），分数是 ${sg2(x.s)}。</b>继续调大 n。</p>`);
  return tx(`<p><b>n = ${NS[i]}: the score of the pick is still rising (${sg2(b[k].s)} at n = ${NS[k]}, ${sg2(x.s)} now), but its true quality has fallen from ${q2(b[k].q)} to ${q2(x.q)}.</b></p><p>Why: the more responses you draw, the more likely one of them is a padded one. Its content reading is the same, it is 3 times as long, and length weighs as much as content, so it gets the top score and is picked. This is ${DEF('reward hacking',TIP.hack())}: the score keeps rising while the true quality falls (red in the chart).</p>`,
            `<p><b>n = ${NS[i]}：挑中的回答的分数还在涨（n = ${NS[k]} 时 ${sg2(b[k].s)}，现在 ${sg2(x.s)}），真实质量却从 ${q2(b[k].q)} 分降到了 ${q2(x.q)} 分。</b></p><p>原因：抽的回答越多，越可能抽到加了废话的那种。它的内容读数不变，长度是 3 倍，而长度的权重和内容一样大，于是它分数最高、被挑中。这叫 ${DEF('reward hacking',TIP.hack())}：分数一直涨，真实质量却在降（图里红色的一段）。</p>`);
}
function obsFix(){
  const v=S.bal, r0=dflt(), b0=dB(), r=run({...BASE,bal:v}), b=bonAt({...BASE,bal:v},T);
  if(v===0) return tx('<p><b>The initial setup: no balanced pairs.</b> Pick an option to add some.</p>','<p><b>初始设置：不加长短平衡的数据。</b>选一个选项加一些。</p>');
  if(v===50) return tx(`<p><b>With 50 balanced pairs, the length weight falls from ${sg2(r0.W[T][1])} to ${sg2(r.W[T][1])}, and the true quality now rises all the way: ${q2(b[8].q)} at n = 256, against ${q2(b0[8].q)} in the initial setup.</b></p><p>Why: in these pairs only the length differs, and the longer one nearly always loses, so every step pushes the length weight down. A padded response still gains a little (in example 2, B still scores above A), but the gain is much smaller, and padded responses are picked far less often.</p>`,
                         `<p><b>加 50 对长短平衡的数据，长度的权重从 ${sg2(r0.W[T][1])} 降到 ${sg2(r.W[T][1])}，真实质量一路上升：n = 256 时是 ${q2(b[8].q)} 分，初始设置是 ${q2(b0[8].q)} 分。</b></p><p>原因：这些偏好对里只有长度不同，长的那个几乎总是输，所以每一步都把长度的权重往下压。加了废话的回答还能多得一点分（示例 2 里 B 仍比 A 高），但多出的这点分小多了，它们被挑中的次数也少了很多。</p>`);
  /* B2 of the review: only the accuracy cost is robust; the n = 256 quality of 200 against 50 pairs is this batch's chance (numbers test, multi-seed check) */
  return tx(`<p><b>With 200 balanced pairs, the length weight turns negative (${sg2(r.W[T][1])}), and the held-out accuracy falls from ${pR(r0.acc[T])} to ${pR(r.acc[T])}.</b></p><p>Why: in the original data, longer answers really are more complete, so length carries useful information about quality. Add too many balanced pairs and the reward model turns against long responses, so it gets more ordinary pairs wrong.</p><p class="small muted">On the right of the options, 200 pairs end a little below 50 pairs in true quality. That is chance in this batch of data: redo it with another batch, and 200 pairs usually end a little higher.</p>`,
          `<p><b>加 200 对，长度的权重变成了负数（${sg2(r.W[T][1])}），留出准确率从 ${pR(r0.acc[T])} 降到 ${pR(r.acc[T])}。</b></p><p>原因：在原来的数据里，长回答本来就更完整，所以长度带着关于质量的有用信息。长短平衡的数据加得太多，奖励模型反过来嫌长回答，在普通的偏好对上就更容易判错。</p><p class="small muted">选项右边的真实质量，加 200 对比加 50 对略低。这只是这批数据碰巧如此，换一批数据重做，加 200 对多半反而略高。</p>`);
}
/* G2 (the owner's review of 2026-10-08): the recap states each rule and what follows from it, never a detail of an earlier part */
function remember(){
  const r=dflt();
  return tx(`<ul class="rm-remlist"><li><b>A reward model learns scores from comparisons.</b> Training pushes the chosen responses above the rejected ones. Only the gap between two scores means something, so a score on its own says nothing, and the scores of two reward models cannot be compared.</li>`+
    `<li><b>It learns only the preferences in its data, wanted or not.</b> If longer answers were often chosen, it learns to like length, even though people never judged by length. So check the data for such clues, and add pairs that break them, such as length-balanced pairs.</li>`+
    `<li><b>Optimise against it too hard and it gets gamed.</b> The harder you select for a high score, the more likely the pick is a response that scores high but is worse. Check the picks themselves, not only their score.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “A reward score is an absolute quality score.” Only differences matter. Add the same number to every score, and no probability, loss or accuracy changes.</p><p>✗ “A reward model that is right only ${pR(r.acc[T])} of the time is useless.” People disagree too: knowing the true quality gets only ${pR(r.ceil)} on these labels, and in AlpacaEval’s human labels one person agrees with the majority of three others only 65.7% of the time.</p><p>✗ “The higher the reward, the better.” Once the reward model is gamed, the score keeps rising while the true quality falls.</p></div>${A.shell.toScenarios()}`,
    `<ul class="rm-remlist"><li><b>奖励模型从比较里学打分。</b>训练让选中的回答分数高过落选的。只有两个分数的差有意义，所以一个分数单独拿出来说明不了好坏，两个奖励模型的分数也不能直接比。</li>`+
    `<li><b>它只学数据里有的偏好，不管是不是想要的。</b>数据里长回答常被选中，它就会偏爱长回答，哪怕标注的人从来不看长度。所以要检查数据里有没有这种线索，再补上能打破它的偏好对，比如长短平衡的偏好对。</li>`+
    `<li><b>按它的分数优化得太狠，它就会被钻空子。</b>越是使劲挑高分，越容易挑中分数高、实际更差的回答。所以要直接检查挑出来的回答，不能只看分数。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “奖励模型的分数是绝对的质量分。”只有分差有意义。所有分数都加上同一个数，概率、loss 和准确率一个都不变。</p><p>✗ “准确率只有 ${pR(r.acc[T])} 的奖励模型没用。”标注的人之间也不一致：就算知道真实质量，在这批标注上也只能对 ${pR(r.ceil)}。在 AlpacaEval 的人工标注里，一个人和另外三个人多数意见一致的比例只有 65.7%。</p><p>✗ “奖励越高越好。”奖励模型被钻了空子以后，分数还在涨，真实质量却在降。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'def',p:{rel:'tied',tau:1,N:1000},t:{en:'Longer answers were often chosen',zh:'数据里长回答常被选中'},d:{en:'Complete answers tend to be longer, as in real chat data',zh:'完整的回答往往更长，和真实的对话数据一样'}},
 {id:'flat',p:{rel:'none',tau:1,N:1000},t:{en:'Length unrelated to quality',zh:'长度和质量无关的数据'},d:{en:'Responses of every quality come in every length',zh:'各种质量的回答，长短都有'}},
 {id:'noisy',p:{rel:'tied',tau:3,N:1000},t:{en:'People often disagree',zh:'标注的人常常意见不一'},d:{en:'Hard questions, or a hurried labelling job',zh:'问题难，或者标注做得匆忙'}},
 {id:'small',p:{rel:'tied',tau:1,N:50},t:{en:'Only 50 pairs',zh:'只有 50 对数据'},d:{en:'A first try before paying for more labels',zh:'花钱多标之前，先试一小批'}}];
const presetNow=()=>PRESETS.findIndex(p=>p.p.rel===S.rel&&p.p.tau===S.tau&&p.p.N===S.N&&S.bal===0);
/* the insights (S10 of the review): one finding in bold, then the trend and why; the exact values are on the cards and the charts */
function scenHTML(i){
  const p=PRESETS[i], s={...BASE,...p.p}, r=run(s), b=bonAt(s,T), k=peakOf(b), r0=dflt();
  if(p.id==='def') return tx(`<p><b>Length got a weight as large as content’s, and best-of-n’s true quality peaks at n = ${NS[k]}.</b></p><p>Longer answers tend to be more complete, so length became a clue to the labels. Beyond that n, best-of-n picks padded responses more and more often.</p>`,
                             `<p><b>长度拿到了和内容一样大的权重，best-of-n 挑出的回答在 n = ${NS[k]} 时真实质量最高。</b></p><p>长回答往往更完整，所以长度成了猜标注的一条线索。n 再大，best-of-n 就越来越常挑中加了废话的回答。</p>`);
  if(p.id==='flat') return tx(`<p><b>The length weight is almost 0 (initial setup ${sg2(r0.W[T][1])}), and best-of-n’s picks keep improving as n grows.</b></p><p>With no link between length and quality in the data, the reward model learns no length preference, and padding earns nothing. The held-out accuracy is a little lower (${pR(r.acc[T])}), because without length, content is the only clue.</p>`,
                              `<p><b>长度的权重几乎是 0（初始设置是 ${sg2(r0.W[T][1])}），n 越大，best-of-n 挑出的回答越好。</b></p><p>数据里长度和质量没有关系，奖励模型就学不到长度偏好，加废话也拿不到高分。留出准确率比初始设置低一点（${pR(r.acc[T])}），因为没有长度可用，内容就是唯一的线索。</p>`);
  if(p.id==='noisy') return tx(`<p><b>The noisier the labels, the less the reward model learns: held-out accuracy falls from ${pR(r0.acc[T])} to ${pR(r.acc[T])}.</b></p><p>Even guessing these labels from the true quality gets only ${pR(r.ceil)} right, so the reward model is close to that ceiling. But the length preference is still there: this time length even outweighs content, so best-of-n turns bad at a smaller n.</p>`,
                               `<p><b>标注越不一致，奖励模型学到的就越少：留出准确率从 ${pR(r0.acc[T])} 降到 ${pR(r.acc[T])}。</b></p><p>就算按真实质量去猜这批标注，也只能猜对 ${pR(r.ceil)}，奖励模型离这个上限只差一点。可长度偏好还在，这一次长度的权重甚至比内容还大，所以 best-of-n 更早开始变差。</p>`);
  return tx(`<p><b>With only 50 pairs, the weights follow chance in the sample.</b></p><p>This time length took almost all the weight, and at large n best-of-n picks worse responses than a random pick. The held-out accuracy (${pR(r.acc[T])}) does not show it, because on ordinary pairs “the longer one is better” is often right.</p>`,
            `<p><b>只有 50 对时，权重跟着样本里的偶然走。</b></p><p>这一次长度几乎拿走了全部权重，n 一大，best-of-n 挑出的回答比随便拿一个还差。留出准确率（${pR(r.acc[T])}）看不出这个问题，因为在普通的偏好对上，猜“长的好”常常是对的。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=e=>tx(`${e.bal} balanced pairs, ${e.steps} steps`,`加 ${e.bal} 对，训练 ${e.steps} 步`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chSt.bal,chSt.steps), best=chBest();
  let h;
  const good=tx(`<p class="okline rm-mt"><b>Goal met: best-of-16 picks responses of true quality ${q2(me.q)}.</b>`,`<p class="okline rm-mt"><b>达成目标：best-of-16 挑出的回答，真实质量是 ${q2(me.q)} 分。</b>`);
  /* Codex R10B (round 3): the goal is an average; a single counterexample can still be judged wrong, and the fixed example 2 is one */
  const avg=me.ok?`<p class="small rm-avg">${tx(`The goal looks at the average true quality of the responses best-of-16 picks; a single example can still be judged wrong.${chEx2Wrong()?' The fixed Example 2 in the chart is one: the reward model still scores the padded response B higher.':''}`,`目标看的是 best-of-16 挑出的回答平均的真实质量；单个例子仍可能判错。${chEx2Wrong()?'图里固定的示例 2 就是一个：奖励模型仍给加了废话的 B 打了更高的分。':''}`)}</p>`:'';
  if(me.ok&&me.bal===best.bal&&me.steps===best.steps) h=good+tx(` And it is the smallest change: ${best.bal} balanced pairs, the training left as it was.</p><p>Fewer pairs leave the length weight high enough for padded responses to win at n = 16. The length weight is learned from the data, and more training steps do not change it.</p>`,
     `而且改动已经是最少的：加 ${best.bal} 对，训练步数不变。</p><p>再少，长度的权重还是够大，n = 16 时加了废话的回答照样能赢。长度的权重是从数据里学来的，多训练几步也改变不了。</p>`);
  else if(me.ok) h=good+`</p>`+tx(`<p>But a smaller change also works: ${chName(best)}. ${me.bal>best.bal?'Every balanced pair is one more label to pay for.':'The number of steps did not need to change: on these data, the number of pairs decides whether the goal is met.'}</p>`,`<p>不过改得更少也行：${chName(best)}。${me.bal>best.bal?'每多一对长短平衡的数据，就要多请人标一对。':'训练步数不用改。在这批数据上，达不达标看的是加了多少对。'}</p>`);
  else h=tx(`<p class="badline rm-mt"><b>Not yet: best-of-16 picks responses of true quality ${q2(me.q)}, below ${CH_Q}.</b></p><p class="small">Hint: padded responses win because length has a large weight. Balanced pairs pull it down.</p>`,`<p class="badline rm-mt"><b>还没达到：best-of-16 挑出的回答，真实质量只有 ${q2(me.q)} 分，低于 ${CH_Q}。</b></p><p class="small">提示：加了废话的回答能赢，是因为长度的权重大。长短平衡的数据能把它压下来。</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  h+=avg;
  if(!chReveal) return h;
  /* brief (the owner's review of batch 8, 2026-10-07): the options that meet the goal, your pick and the initial setup; with more than 8 of
     them, your pick, the initial setup and the smallest change (chBest) stay, and the rest of the passing options wait behind a closed list */
  const rows=[]; CH_BAL.forEach(bal=>CH_STEPS.forEach(st=>rows.push(chEval(bal,st))));
  const crows=rows.map(e=>({label:`${e.ok?'✓ ':''}${chName(e)}${tx(` (accuracy ${pP(e.acc)})`,`（准确率 ${pP(e.acc)}）`)}`,value:e.q,text:q2(e.q),ok:e.ok,me:e.bal===chSt.bal&&e.steps===chSt.steps,
      best:e.bal===best.bal&&e.steps===best.steps,e,
      tags:(e.bal===CH_START.bal&&e.steps===CH_START.steps?[{text:tx('initial setup','初始设置')}]:[]).concat(e.bal===chSt.bal&&e.steps===chSt.steps?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  const keep=crows.filter(r=>r.ok||r.me||r.tags.length), shown=keep.length>8?crows.filter(r=>r.me||r.best||r.tags.length):keep;
  h+=A.ui.cmpChart({brief:true,title:tx(`True quality of the best-of-16 pick, out of 10 (goal: at least ${CH_Q})`,`best-of-16 挑出回答的真实质量（满分 10，目标：至少 ${CH_Q}）`),
    aria:tx('True quality: ','真实质量：')+shown.map(r=>`${chName(r.e)}${tx(': ','：')}${q2(r.e.q)}${tx(', accuracy ',' 分，准确率 ')}${pP(r.e.acc)}${r.ok?tx(', meets the goal','，达标'):''}`).join(tx('; ','；')),
    rows:crows,
    goal:CH_Q,goalLabel:tx(`goal: ${CH_Q}`,`目标：${CH_Q}`),max:10,okText:tx('✓ meets the goal','✓ 达标'),noText:tx('below the goal','未达标'),goalText:tx(`goal: ${CH_Q}`,`目标：${CH_Q}`)});
  const s0=chEval(0,200);
  h+=tx(`<p class="small">In the initial setup the pick’s true quality is ${q2(s0.q)}: length had a large weight, and padded responses won. ${best.bal} balanced pairs bring it to ${q2(best.q)}.</p><p class="small">The more pairs you add, the lower the length weight, and the held-out accuracy drifts down a little too. These accuracies come from this batch’s 500 held-out pairs, and another 500 would move them by about ±4 points, so only the trend counts, not a goal.</p><p class="small muted">Note: in this lab the true quality of every response is known. A real team does not know it, so it checks the picks with people or a stronger judge, on responses the reward model never saw.</p>`,
        `<p class="small">初始设置下，挑出的回答真实质量是 ${q2(s0.q)} 分：长度的权重大，加了废话的回答赢了。加 ${best.bal} 对，就升到 ${q2(best.q)} 分。</p><p class="small">加得越多，长度的权重压得越低，留出准确率也跟着往下走一点。这些准确率是在这批留出的 500 对上算的，换一批会上下变动约 4 个百分点，所以只看趋势，不算条件。</p><p class="small muted">注意：在这个 Lab 里，每个回答的真实质量都是已知的。真实的团队并不知道，所以要请人或者用更强的裁判，在奖励模型没见过的回答上检查挑出来的结果。</p>`);
  return h;
}

/* =====================================================================
   header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Eight short parts. See how a score gap becomes a choice, train a small reward model on 1,000 pairs, see what it learned, and use it to pick answers until it gets gamed. Start here if this is new to you.',
            '一共 8 个部分。先看分差怎样变成“选它”的概率，再在 1,000 个偏好对上训练一个小奖励模型，看它学到了什么，最后拿它挑回答，直到它被钻空子。第一次学从这里开始。'),
  sandbox:tx('Four kinds of preference data, from length tied to quality to only 50 pairs. Pick one to see what the reward model learns and what best-of-n picks.',
             '四种偏好数据，从长度跟着质量走到只有 50 对。选一种，看奖励模型学到什么、best-of-n 挑出什么。'),
  challenge:tx(`New data in which length follows quality very closely. Add the fewest length-balanced pairs so that best-of-16 picks responses of true quality at least ${CH_Q}.`,
               `一批新的数据，长度和质量绑得很紧。加最少的长短平衡的数据，让 best-of-16 挑出的回答真实质量至少 ${CH_Q} 分。`)
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser trains a small reward model with 3 parameters by real gradient steps, on seeded, simulated preference pairs. Every number on the page comes from it; ${J_('math','“Show the math”')} works them out. Facts about real systems are quoted with their sources.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Every response has a true quality q out of 10 (mean 6, standard deviation 1.5), a length (how many tokens), and whether it uses a list. In the default data, ${A.ui.math('\\ln(\\text{length}) = \\ln 150 + 0.35 \\cdot \\frac{q - 6}{1.5} + 0.25 z')}, with z a standard normal, so complete answers tend to be longer.</li>`+
 `<li>People pick A with chance ${A.ui.math('\\sigma\\left(\\frac{q_A - q_B}{τ}\\right)')}, with τ = 1. Length plays no part.</li>`+
 `<li>The reward model’s score is ${A.ui.math('r = w \\cdot \\phi')}, with the features ${A.ui.math('\\phi = \\left[\\frac{c - 6}{1.5}, \\frac{\\ln(\\text{length} / 150)}{0.4}, \\text{list} - 0.5\\right]')}. Here ${A.ui.math('c = q + e')}, and e is a standard normal reading error. The loss is the mean of ${A.ui.math('-\\log\\, \\sigma(r_{\\text{chosen}} - r_{\\text{rejected}})')}, as in TRL’s RewardTrainer; full-batch gradient descent, learning rate 0.15, ${T} steps, from w = 0.</li>`+
 `<li>Best-of-n: the chat model’s responses are a fixed set of 2,000, 8% padded (3 times as long, same content, true quality 2.5 lower). The chart shows the exact average over all ways of drawing n with replacement.</li>`+
 `<li>Seeds (mulberry32, Box–Muller normals): training pairs 13, held-out pairs 113, balanced pairs 13 + 7919, the chat model’s responses 31; the challenge’s data 45 and 46. Same seed, same run.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>In this lab a response is a few features, not text. A real reward model reads all the tokens and gives one score, at the last token.</li>`+
 `<li>The small reward model is linear with 3 parameters; a real one is a whole LLM. Its learning rate and step count say nothing about a real model’s.</li>`+
 `<li>In this lab every response’s true quality is known. Nobody knows it in practice; the example texts are illustrative and their features are set for this lab.</li>`+
 `<li>The chat model here is a fixed set of 2,000 responses described by a few features; a real one writes new text every time.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>TRL v1.14.1: ${cite(SRC.trl,'<code>reward_trainer.py</code>')} (the loss line, <code>num_labels = 1</code>, the accuracy metric), ${cite(SRC.trlc,'<code>reward_config.py</code>')} (<code>center_rewards_coefficient</code>, recommended 0.01).</li>`+
 `<li>Transformers v5.18.0 ${cite(SRC.hf,'<code>modeling_layers.py</code>')}: the score layer <code>nn.Linear(hidden_size, num_labels, bias=False)</code>, read at the last non-padding token.</li>`+
 `<li>${cite(SRC.hh,'hh-rlhf README')}: one “chosen” and one “rejected” text per line. Part of the helpfulness data was collected by rejection sampling: a model wrote several responses and an early preference model picked the best, mostly best-of-16.</li>`+
 `<li>${cite(SRC.ae,'AlpacaEval README')}: when one output is more than 30 characters longer, people prefer it 64% of the time; one person agrees with the majority of the other three 65.7% of the time.</li>`+
 `<li>${cite(SRC.rb,'RewardBench <code>run_rm.py</code>')}: accuracy counts 1 when the chosen response scores higher.</li>`+
 `<li>What a reward model is and how it is used: ${cite(SRC.orl,'OpenRLHF README')} trains its reward model from an SFT checkpoint (<code>Llama-3-8b-sft-mixture</code>) and loads it with one output (<code>num_labels=1</code>); ${cite(SRC.hfb,'“Illustrating RLHF”')} says reward models are LMs whose size can differ from the text model’s (a 175B LM with a 6B reward model) and that about 50,000 labelled preference pairs is the usual scale; ${cite(SRC.rlb,'the RLHF book, chapter 5')}: a reward model reranks several completions to pick the top one (best-of-N), or provides the reward for later training.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-08. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器在带种子的模拟偏好对上，用真实的梯度下降训练一个只有 3 个参数的小奖励模型。页面上的每个数都由它算出，${J_('math','“展开公式”')}里有逐行算式。关于真实系统的说法都标了出处。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>每个回答有一个满分 10 分的真实质量 q（平均 6 分，标准差 1.5），还有一个长度（多少个 token），以及是否用了列表。默认数据里，${A.ui.math('\\ln(\\text{长度}) = \\ln 150 + 0.35 \\cdot \\frac{q - 6}{1.5} + 0.25 z')}，z 服从标准正态分布，所以完整的回答往往更长。</li>`+
 `<li>人以 ${A.ui.math('\\sigma\\left(\\frac{q_A - q_B}{τ}\\right)')} 的概率选 A，τ = 1。长度不起作用。</li>`+
 `<li>奖励模型的分数是 ${A.ui.math('r = w \\cdot \\phi')}，三个特征是 ${A.ui.math('\\phi = \\left[\\frac{c - 6}{1.5}, \\frac{\\ln(\\text{长度} / 150)}{0.4}, \\text{列表} - 0.5\\right]')}。其中 ${A.ui.math('c = q + e')}，e 是服从标准正态分布的读数误差。loss 是 ${A.ui.math('-\\log\\, \\sigma(r_{\\text{选中}} - r_{\\text{落选}})')} 的平均，和 TRL 的 RewardTrainer 一样。全批量梯度下降，学习率 0.15，${T} 步，从 w = 0 开始。</li>`+
 `<li>best-of-n：聊天模型的回答固定为 2,000 个，其中 8% 加了废话（长度是 3 倍，内容一样，真实质量低 2.5 分）。图上画的是精确的平均值：从这 2,000 个回答里有放回地抽 n 个，所有抽法都算进去。</li>`+
 `<li>种子（mulberry32，正态随机数用 Box–Muller）：训练用的偏好对 13，留出的偏好对 113，长短平衡的偏好对 13 + 7919，聊天模型的回答 31；挑战题的数据 45 和 46。种子相同，结果就相同。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>在这个 Lab 里，一个回答只是几个特征，不是文字。真实的奖励模型读完整段 token，只在最后一个 token 上打一个分。</li>`+
 `<li>小奖励模型是只有 3 个参数的线性模型，真实的奖励模型是一整个 LLM。它的学习率和步数，对真实模型没有参考意义。</li>`+
 `<li>在这个 Lab 里，每个回答的真实质量都是已知的，现实里没人知道。示例的文字是示意的，它们的特征值是这个 Lab 设定的。</li>`+
 `<li>这里的聊天模型只是固定的 2,000 个回答，每个回答只有几个特征。真实的聊天模型每次都会写出新的文字。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>TRL v1.14.1：${cite(SRC.trl,'<code>reward_trainer.py</code>')}（loss 的那一行、<code>num_labels = 1</code>、准确率指标），${cite(SRC.trlc,'<code>reward_config.py</code>')}（<code>center_rewards_coefficient</code>，推荐 0.01）。</li>`+
 `<li>Transformers v5.18.0 ${cite(SRC.hf,'<code>modeling_layers.py</code>')}：打分层是 <code>nn.Linear(hidden_size, num_labels, bias=False)</code>，在最后一个非 padding 的 token 上取值。</li>`+
 `<li>${cite(SRC.hh,'hh-rlhf 的 README')}：每行一对文字，一个 “chosen”，一个 “rejected”。helpfulness 数据有一部分是这样收集的：模型写出多个回答，用一个早期的偏好模型挑出最好的，大多是 best-of-16（拒绝采样）。</li>`+
 `<li>${cite(SRC.ae,'AlpacaEval 的 README')}：两个输出长度相差超过 30 个字符时，人有 64% 的时候选更长的；一个人和另外三个人多数意见一致的比例是 65.7%。</li>`+
 `<li>${cite(SRC.rb,'RewardBench 的 <code>run_rm.py</code>')}：选中的回答分数更高，就记 1，准确率是这些 1 的比例。</li>`+
 `<li>奖励模型是什么、怎么用：${cite(SRC.orl,'OpenRLHF 的 README')} 从 SFT 之后的检查点（<code>Llama-3-8b-sft-mixture</code>）训练奖励模型，加载时只有一个输出（<code>num_labels=1</code>）。${cite(SRC.hfb,'“Illustrating RLHF”')}写到，奖励模型也是语言模型，大小可以和生成文字的模型不同（175B 的语言模型配 6B 的奖励模型），常见的数据量约是 50,000 个标注好的偏好对。${cite(SRC.rlb,'RLHF book 第 5 章')}写到，奖励模型的用法是给几个回答重新排序、挑出最高的一个（best-of-N），或者给后续的训练提供奖励。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-08。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A preference pair is two responses to the same prompt and the one a person chose. ${lk('llm-judge')} compares responses in pairs and names the Bradley–Terry model, which this lab trains.</p><p>The reward model usually starts from the model after SFT, with its output layer replaced by a score layer that outputs one number (see ${lk('sft')}).</p>`,
                    `<p>偏好对是同一个提问的两个回答，加上标注的人选了哪一个。${lk('llm-judge')}讲过两两比较，也提到了 Bradley–Terry 模型，这个 Lab 就训练它。</p><p>奖励模型通常从 SFT 之后的模型出发，把输出层换成只输出一个数的打分层（见 ${lk('sft')}）。</p>`);
const OUTCOME=()=>tx('explain how pairwise preferences train a reward model, and spot and fix the length preference it learns from its data.','讲清偏好对怎样训练出奖励模型，并能发现、修正它从数据里学到的长度偏好。');
const EXITS=()=>{
  /* the new case's two reward models (review S15): the initial data plus 25 or 100 balanced pairs, which no part shows */
  const r=dflt(), b=dB(), k=peakOf(b), rA=run({...BASE,bal:25}), bA=bonAt({...BASE,bal:25},T), rB=run({...BASE,bal:100}), bB=bonAt({...BASE,bal:100},T);
  const p2=bon2();
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does adding the same number to every score change neither the training nor the predictions?','为什么给所有分数加上同一个数，训练和预测都不变？'),
   o:[tx('The loss and the probability use only the gap between two scores, and the added number cancels in the subtraction.','loss 和概率只用到两个分数的差，加上的数在相减时抵消了。'),
      tx('The reward model brings the average score back to 0 at every step, so the added number disappears.','奖励模型每一步都会把平均分调回 0，加上的数就消失了。'),
      tx('The added number is too small next to the scores to make a visible difference.','加上的数和分数比太小，看不出影响。')],a:0,
   n:[null,tx('Nothing in the Bradley–Terry loss pulls the average anywhere. TRL can add such a term (center_rewards_coefficient), but only if you ask for it.','Bradley–Terry 的 loss 里没有任何一项去拉平均分。TRL 可以加这样一项（center_rewards_coefficient），但要你自己打开。'),
     tx('Size has nothing to do with it: adding 1,000 would change nothing either, because only differences enter the loss.','和大小无关：加 1,000 也一样什么都不变，因为 loss 里只有分差。')],
   why:tx('Moving every score by the same amount leaves every gap as it was, so the probability, the loss and the accuracy all stay the same.','所有分数挪同样多，每个分差都不变，所以概率、loss 和准确率全都不变。')},
  {q:tx('People in this lab judge only the true quality. Why does the reward model still give length a large weight?','这个 Lab 里标注的人只看真实质量。为什么奖励模型还是给长度很大的权重？'),
   o:[tx('It reads content with errors, and in the data more complete answers tend to be longer, so length is a second clue to the labels.','奖励模型读内容会读错，而数据里越完整的回答往往越长，长度就成了猜对标注的另一条线索。'),
      tx('Longer answers have more tokens, so they get larger gradients during training.','长回答的 token 多，训练时梯度就更大。'),
      tx('The reward model starts with a preference for length, and training does not remove it.','奖励模型一开始就偏爱长度，训练也去不掉。')],a:0,
   n:[null,tx('A response is scored once, whatever its length; the gradient depends on the features’ differences, not on a token count.','不管多长，一个回答只打一个分。梯度取决于特征的差，和 token 的个数无关。'),
     tx('It starts at w = 0, with no preference at all. Trained on data where length does not follow quality, it gives length almost no weight.','它从 w = 0 开始，什么偏好都没有。换成长度不跟着质量走的数据训练，长度几乎拿不到权重。')],
   why:tx(`The length weight is learned from the data: ${sg2(r.W[T][1])} on data where longer answers were often chosen, ${sg2(run({...BASE,rel:'none'}).W[T][1])} on data where length is unrelated to quality.`,`长度的权重是从数据里学来的。在长回答常被选中的数据上，它是 ${sg2(r.W[T][1])}。换成长度和质量无关的数据，它只有 ${sg2(run({...BASE,rel:'none'}).W[T][1])}。`)}],
 transfer:[
  {q:tx(`Use this lab’s reward model for best-of-n. When 8% of the chat model’s responses are padded, the pick’s true quality is highest at n = ${NS[k]}. Now only 2% of them are padded, and the reward model is the same. At what n is the true quality of the pick highest now?`,`用这个 Lab 的奖励模型做 best-of-n。聊天模型的回答有 8% 加了废话时，n = ${NS[k]} 挑出的回答真实质量最高。现在只有 2% 加了废话，奖励模型不变。这时 n 等于多少，挑出的回答真实质量最高？`),
   o:[tx(`Earlier, below n = ${NS[k]}`,`更早，n 小于 ${NS[k]}`),tx(`About the same, near n = ${NS[k]}`,`差不多，在 n = ${NS[k]} 附近`),tx(`Later, above n = ${NS[k]}`,`更晚，n 大于 ${NS[k]}`)],a:2,
   why:tx(`Later: n = ${NS[p2.k]} in this lab, at ${q2(p2.b[p2.k].q)}. With fewer padded responses, it takes more draws before one of them shows up among the n, so n has to grow larger before the picks start getting worse. The reward model still prefers them; they are just rarer.`,`更晚：这个 Lab 算出来是 n = ${NS[p2.k]}，${q2(p2.b[p2.k].q)} 分。加了废话的回答越少，要抽更多次才会有一个混进这 n 个里，所以 n 要调得更大，挑出的回答才开始变差。奖励模型照样偏爱它们，只是它们更少见了。`)},
  {q:tx(`You have two reward models. On the same held-out pairs, one is right ${pR(rA.acc[T])} of the time and the other ${pR(rB.acc[T])}. You will use one of them for best-of-256. Which one picks better responses?`,`你有两个奖励模型。在同一批留出的偏好对上，一个的准确率是 ${pR(rA.acc[T])}，另一个是 ${pR(rB.acc[T])}。你要用其中一个做 best-of-256。哪个挑出的回答更好？`),
   o:[tx('The more accurate one','准确率高的那个更好'),tx('Not necessarily; check the picks themselves','不一定，要直接检查挑出来的回答'),tx('Both pick equally well','两个一样好')],a:1,
   n:[tx('Accuracy only says how often it ranks ordinary pairs the right way round.','准确率只说明它在普通的偏好对上排得对不对。'),null,tx('The two models have different weights, so their top-scored responses differ.','两个模型的权重不一样，分数最高的回答也就不一样。')],
   why:tx(`Not necessarily; check the picks themselves. Accuracy is measured on ordinary pairs, and those do not contain the extreme responses best-of-n seeks out. In this lab, the ${pR(rA.acc[T])} one is the initial data plus 25 balanced pairs, and at n = 256 it picks responses of true quality ${q2(bA[8].q)}. The ${pR(rB.acc[T])} one has 100 balanced pairs and picks ${q2(bB[8].q)}.`,`不一定，要直接检查挑出来的回答。准确率是在普通的偏好对上测的，测不到 best-of-n 专门去找的那些极端回答。在这个 Lab 里，准确率 ${pR(rA.acc[T])} 的那个是在初始数据上加了 25 对长短平衡的数据，n = 256 时挑出的回答真实质量是 ${q2(bA[8].q)} 分。准确率 ${pR(rB.acc[T])} 的那个加了 100 对，是 ${q2(bB[8].q)} 分。`)}],
 practice:[
  {id:'rl-for-llms',d:tx('Question 8: symptoms of reward hacking during RLHF, and how to mitigate it. This lab shows the symptom with best-of-n (the score rises, the true quality falls, the picks get longer) and one fix, length-balanced pairs.','第 8 题：RLHF 训练中 reward hacking 的症状和缓解办法。这个 Lab 用 best-of-n 演示了症状（分数涨、真实质量降、挑中的回答变长），还演示了一种修法，就是补上长短平衡的偏好对。')},
  {id:'ml-fundamentals',d:tx('Question 4: the logistic-regression loss and its gradient. This lab’s reward model is logistic regression on the feature differences of each pair (the label: the chosen one wins), so the same derivation gives its gradient.','第 4 题：logistic 回归的 loss 和梯度。这个 Lab 的奖励模型，就是在每一对的特征差上做 logistic 回归（标签是“选中的赢”），推导完全一样。')}]};
};
/* the new case: the same reward model, a bank with 2% padded responses (same seed, so the other responses are the same) */
let P2=null;
function bon2(){if(P2) return P2; const r=mulberry(31), out=[]; for(let i=0;i<M;i++){const pd=r()<0.02; out.push(resp(r,RELS.tied,pd));}
  const save=BANK; BANK=out; const b=bon(dflt().W[T]); BANK=save; return P2={b,k:peakOf(b)};}

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Reward model','奖励模型'),title:l?L(l.q):tx('How does a reward model turn preferences into scores?','奖励模型怎么把偏好变成分数？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: preference pairs','前置知识：偏好对'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage rm-stage" id="rm-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="rm-st-title"></div><div class="meter-sub" id="rm-st-sub"></div></div><div class="stage-actions"><span class="status" id="rm-status"></span></div></div>`+
   `<div class="stage-graph graph rm-graph" id="rm-graph">`+
    `<div class="rm-phone" id="rm-phone"></div>`+
    `<div class="rm-pair" id="rm-pair"></div>`+
    `<div class="rm-sig" id="rm-sig"></div>`+
    `<div class="rm-w" id="rm-w"></div>`+
    `<div class="rm-tr" id="rm-tr"><div class="rm-cap" id="rm-tr-cap"></div><div class="rm-charts" id="rm-tr-charts"></div><div class="rm-player" id="rm-player"></div></div>`+
    `<div class="rm-bon" id="rm-bon"></div>`+
   `</div>`+
   `<p class="rm-scope small muted">${tx('Toy run: a small reward model with 3 parameters, trained in your browser on simulated pairs; example texts are illustrative.','玩具实跑：3 个参数的小奖励模型在浏览器里用模拟的偏好对训练；示例文字只是示意。')}</p>`+
   `<div class="metrics rm-metrics" id="rm-metrics"></div>`+
   `<details class="math" id="rm-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="rm-mathbody"></div></details></div>`;
}
const ASSUME=()=>[
 [tx('True quality','真实质量'),tx('out of 10; mean 6, standard deviation 1.5','满分 10 分，平均 6 分，标准差 1.5')],
 [tx('How noisy people are, τ','标注的人有多不一致 τ'),tx('1 (3 in “People often disagree”): a 1-point better answer is picked 73% of the time','1（“常常意见不一”里是 3）：好 1 分的回答有 73% 的时候被选中')],
 [tx('Content reading error','内容读数的误差'),tx('standard deviation 1 point: the small reward model reads content imperfectly, like a real one','标准差 1 分：小奖励模型读内容会读错，真实的也一样')],
 [tx('Length and quality','长度和质量'),tx('each standard deviation of quality makes answers 42% longer in the default data (0.35 in the log), 82% in the challenge’s','默认数据里，质量每高一个标准差，回答长 42%（对数里加 0.35）；挑战题的数据里长 82%')],
 [tx('Padded responses','加了废话的回答'),tx('8% of the chat model’s 2,000 responses; 3 times as long, same content, true quality 2.5 lower','聊天模型的 2,000 个回答里占 8%，长度是 3 倍，内容一样，真实质量低 2.5 分')],
 [tx('Training','训练'),tx(`full-batch gradient descent, learning rate 0.15, ${T} steps; set so a run settles`,`全批量梯度下降，学习率 0.15，${T} 步。这样设是为了让一次训练能收敛`)]];
function controlsHTML(){
  return `<aside class="panel controls" id="rm-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="rm-lock" hidden></p>`+
   `<div class="lockable" id="rm-ctl-rel-wrap">`+A.ui.seg({id:'rm-ctl-rel',key:'rel',label:tx('Length in the data','数据里的长度'),options:[['tied',tx('follows quality','跟着质量走')],['none',tx('unrelated','和质量无关')]],value:S.rel,note:tx('Whether more complete answers are longer in the training pairs.','训练用的偏好对里，越完整的回答是不是越长。')})+`</div>`+
   `<div class="lockable" id="rm-ctl-tau-wrap">`+A.ui.seg({id:'rm-ctl-tau',key:'tau',label:tx('People','标注的人'),options:[['1',tx('mostly agree (τ = 1)','大多一致（τ = 1）')],['3',tx('often disagree (τ = 3)','常有分歧（τ = 3）')]],value:String(S.tau),note:tx('τ sets how often people pick the worse response; assumed.','τ 决定人有多常选到更差的那个，这是假设。')})+`</div>`+
   `<div class="lockable" id="rm-ctl-N-wrap">`+A.ui.seg({id:'rm-ctl-N',key:'N',label:tx('Preference pairs','偏好对'),options:[['50','50'],['1000','1,000']],value:String(S.N)})+`</div>`+
   `<div class="lockable" id="rm-ctl-bal-wrap">`+A.ui.seg({id:'rm-ctl-bal',key:'bal',label:tx('Length-balanced pairs added','另加长短平衡的偏好对'),options:[0,25,50,100,200].map(v=>[String(v),String(v)]),value:String(S.bal)})+`</div>`+
   `<div class="lockable" id="rm-ctl-n-wrap">`+A.ui.range({id:'rm-ctl-n',key:'ni',label:tx('Best-of-n: n','best-of-n 的 n'),min:0,max:NS.length-1,step:1,value:S.ni,out:String(NS[S.ni])})+`</div>`+
   `<div class="rm-grp" id="rm-ctl-z"><div class="rm-gh">${tx('Values this lab assumes','本 Lab 假设的值')}</div><ul class="rm-asm">${ASSUME().map(([k,v])=>`<li><b>${esc(k)}</b> ${esc(v)}</li>`).join('')}</ul></div>`+
   `<div class="lockable" id="rm-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

def bt_loss(w, chosen, rejected):
    """chosen, rejected: (pairs, features)"""
    m = (chosen - rejected) @ w        # score gaps
    return np.mean(np.logaddexp(0, -m))  # -log sigmoid

def bt_grad(w, chosen, rejected):
    d = chosen - rejected
    s = 1 / (1 + np.exp(d @ w))        # sigmoid(-gap)
    return -(s[:, None] * d).mean(axis=0)

def train(chosen, rejected, steps=200, lr=0.15):
    w = np.zeros(chosen.shape[1])
    for _ in range(steps):
        w -= lr * bt_grad(w, chosen, rejected)
    return w

rng = np.random.default_rng(0)
q = rng.normal(6, 1.5, size=(2000, 2))      # true quality A, B
ln_len = 0.35 * (q - 6) / 1.5 + 0.25 * rng.normal(size=q.shape)
c = q + rng.normal(size=q.shape)             # content reading
f = np.stack([(c - 6) / 1.5, ln_len / 0.4], axis=-1)
a_wins = rng.random(2000) < 1 / (1 + np.exp(q[:, 1] - q[:, 0]))
ch = np.where(a_wins[:, None], f[:, 0], f[:, 1])
rj = np.where(a_wins[:, None], f[:, 1], f[:, 0])
w = train(ch, rj)
print(np.round(w, 2), round(bt_loss(w, ch, rj), 3))`;
const CODE_OUT=`# prints [0.63 0.52] 0.497: length weighs almost as much as content`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="rm-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="rm-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The Bradley–Terry loss and its gradient, trained on 2,000 simulated pairs in which longer answers are more complete. Two features here: content and length.','Bradley–Terry 的 loss 和梯度，在 2,000 个模拟的偏好对上训练，数据里越完整的回答越长。这里只用两个特征：内容和长度。')}</p><pre class="code" id="rm-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="rm-prod">`+
   tx(`<li>TRL’s RewardTrainer (${cite(SRC.trl,'<code>trl/trainer/reward_trainer.py</code>')}) loads the model with <code>AutoModelForSequenceClassification</code> and <code>num_labels = 1</code> (“the only output of the model is the reward score”). Its loss: <code>-nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()</code>; its accuracy metric counts <code>rewards_chosen &gt; rewards_rejected</code>.</li>`+
      `<li>In Transformers (${cite(SRC.hf,'<code>modeling_layers.py</code>')}) the head is <code>nn.Linear(hidden_size, num_labels, bias=False)</code>, read at the last non-padding token: for ${L8}, 4,096 parameters in place of the 525M of the LM head.</li>`+
      `<li>The hh-rlhf data set (${cite(SRC.hh,'README')}) stores one “chosen” and one “rejected” text per line. Part of its helpfulness data was collected by rejection sampling: a model wrote several responses and an early preference model picked the best, mostly best-of-16.</li>`+
      `<li>RewardBench (${cite(SRC.rb,'<code>scripts/run_rm.py</code>')}) scores the chosen and the rejected text and records 1 when the chosen one is higher: the accuracy of this lab, on curated pairs.</li>`+
      `<li>TRL’s <code>center_rewards_coefficient</code> (${cite(SRC.trlc,'<code>reward_config.py</code>')}, recommended 0.01) adds a term that pulls rewards toward mean zero.</li>`,
      `<li>TRL 的 RewardTrainer（${cite(SRC.trl,'<code>trl/trainer/reward_trainer.py</code>')}）用 <code>AutoModelForSequenceClassification</code> 加载模型，<code>num_labels = 1</code>（注释写着：模型唯一的输出就是奖励分数）。loss 是 <code>-nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()</code>；准确率指标数的是 <code>rewards_chosen &gt; rewards_rejected</code>。</li>`+
      `<li>Transformers 里（${cite(SRC.hf,'<code>modeling_layers.py</code>')}）打分层是 <code>nn.Linear(hidden_size, num_labels, bias=False)</code>，在最后一个非 padding 的 token 上取值：${L8} 用 4,096 个参数，换掉了 LM head 的 525M 个参数。</li>`+
      `<li>hh-rlhf 数据集（${cite(SRC.hh,'README')}）每行存一个 “chosen” 和一个 “rejected” 文本。helpfulness 数据有一部分是这样收集的：模型写出多个回答，用一个早期的偏好模型挑出最好的，大多是 best-of-16（拒绝采样）。</li>`+
      `<li>RewardBench（${cite(SRC.rb,'<code>scripts/run_rm.py</code>')}）给选中和落选的文本各打一个分，选中的更高就记 1：也就是这个 Lab 的准确率，只是用的是精心挑过的偏好对。</li>`+
      `<li>TRL 的 <code>center_rewards_coefficient</code>（${cite(SRC.trlc,'<code>reward_config.py</code>')}，推荐 0.01）在 loss 里加一项，把奖励的平均值往 0 拉。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.summ,'Learning to summarize from human feedback')}<span>${tx('Stiennon et al., 2020, arXiv 2009.01325','Stiennon 等，2020，arXiv 2009.01325')}</span></li>`+
   `<li>${cite(SRC.igpt,'Training language models to follow instructions with human feedback')}<span>${tx('Ouyang et al., 2022, arXiv 2203.02155 (InstructGPT)','Ouyang 等，2022，arXiv 2203.02155（InstructGPT）')}</span></li>`+
   `<li>${cite(SRC.gao,'Scaling Laws for Reward Model Overoptimization')}<span>${tx('Gao et al., 2022, arXiv 2210.10760: the same rise-then-fall shape, for best-of-n and for training against the score','Gao 等，2022，arXiv 2210.10760：best-of-n 和照着分数训练时，同样的先升后降')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="rm-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="rm-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="rm-app m2" id="rm-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="rm-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   /* G1 (the owner's review of 2026-10-08): a lead-in that says what to pick and what to watch, not how the options combine */
   `<p class="small">${tx('Pick a scenario and see how much weight the reward model gives length, and how good the responses best-of-n picks really are.','选一个场景，看看奖励模型给长度多大的权重，best-of-n 挑出的回答真实质量有多高。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="rm-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`To change something else, use ${J_('controls','Settings')}.`,`想改别的，在“${J_('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal with the smallest change, and why.','提交后会告诉你是否达标、改动是不是最少，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Best-of-16 at true quality ${CH_Q} or more, with the fewest new labels`,`best-of-16 的真实质量至少 ${CH_Q} 分，新标注越少越好`)}</h3>`+
   `<p>${tx(`A new batch of 1,000 preference pairs, in which length follows quality very closely: the better an answer, the longer it is. You train the small reward model on it and use it for best-of-16 on the same chat model’s responses. The goal: the responses best-of-16 picks have a true quality of at least ${CH_Q} out of 10. You may add length-balanced pairs (a concise response and the same response padded to 3 times the length; people nearly always pick the concise one) and change the number of training steps. Every balanced pair is one more label to pay for: add as few as you can.`,
            `现在有一批新的 1,000 个偏好对。这批数据里，长度和质量绑得很紧，越好的回答越长。你用它训练小奖励模型，再拿它给同一个聊天模型的回答做 best-of-16。目标是让 best-of-16 挑出的回答，真实质量至少 ${CH_Q} 分（满分 10）。你可以另加长短平衡的偏好对（同一个回答的简洁版和加了废话、长了 3 倍的版本，标注的人几乎总选简洁的），也可以改训练步数。每一对长短平衡的数据都要花钱请人标，加得越少越好。`)}</p>`+
   A.ui.seg({id:'rm-ic-chB',key:'chB',label:tx('Length-balanced pairs added','另加长短平衡的偏好对'),options:CH_BAL.map(v=>[String(v),String(v)]),value:String(chSt.bal),inline:true})+
   A.ui.seg({id:'rm-ic-chS',key:'chS',label:tx('Training steps','训练步数'),options:CH_STEPS.map(v=>[String(v),String(v)]),value:String(chSt.steps),inline:true})+
   `<div class="goal mono rm-goal" id="rm-ch-g1"><span>${tx('Best-of-16 true quality','best-of-16 的真实质量')}</span><span><b id="rm-ch-q"></b>${tx(` (goal: at least ${CH_Q})`,`（目标：至少 ${CH_Q}）`)}</span></div>`+
   `<div class="goal mono rm-goal" id="rm-ch-g2"><span>${tx('Held-out accuracy','留出准确率')}</span><span><b id="rm-ch-acc"></b>${tx(' (shown, not a goal)','（只作参考，不算条件）')}</span></div>`+
   submit+`<div id="rm-ch-res" class="rm-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
const WHY={bt:{en:'Pick an answer to continue.',zh:'选一个答案才能继续。'},train:{en:'Play the training to the end to continue.',zh:'把训练播放完才能继续。'},shift:{en:'Move the slider to continue.',zh:'拖动滑块才能继续。'},
  weights:{en:'Train on the other data to continue.',zh:'换另一批数据训练才能继续。'},hack:{en:'Raise n to 64 or more to continue.',zh:'把 n 调到 64 或更大才能继续。'},fix:{en:'Add some balanced pairs to continue.',zh:'加一些长短平衡的数据才能继续。'}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#rm-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function setOut(id,txt){const el=$(id,app); if(!el) return; const o=el.closest('.inline-ctl,.ctl'); const out=o&&o.querySelector('output'); if(out&&out.textContent!==txt) out.textContent=txt;}
function syncRange(id,v){const el=$(id,app); if(el&&+el.value!==v&&document.activeElement!==el){el.value=v; A.ui.setRange(el);}}
function updateRail(){
  const rail=$('#rm-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step]||s.id==='fix'||s.id==='hack');
    if(s.id==='weights') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===S.rel)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=wTried.has(v)?wRes(v):'';});
    if(s.id==='fix') $$('[data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===S.bal)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=fixTried.has(v)?fRes(v):'';});
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; const ae=document.activeElement, keep=ae&&box.contains(ae)&&ae.id;
          if(keep&&ae.type==='range'&&box._h){/* the learner is dragging a slider inside the box: update only its text (contract §3) */
            const tmp=document.createElement('div'); tmp.innerHTML=full; $$('p',box).forEach((p,i)=>{const n=$$('p',tmp)[i]; if(n&&p.innerHTML!==n.innerHTML) p.innerHTML=n.innerHTML;}); setOut('#'+keep,(S.d>0?'':'')+sg2(S.d).replace(/\.00$/,'')); box._h=full;}
          else {box.innerHTML=full; box._h=full; box._step=step; A.ui.initRanges(box); if(again) A.ui.obsBump(box);}}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    syncRange('#rm-ic-n-in',S.ni); setOut('#rm-ic-n-in',String(NS[S.ni]));
    syncRange('#rm-ic-shift-in',S.shift); setOut('#rm-ic-shift-in',(S.shift>0?'+':'')+sg2(S.shift).replace(/\.00$/,''));
    syncRange('#rm-ic-d-in',S.d); setOut('#rm-ic-d-in',sg2(S.d).replace(/\.00$/,''));
    /* the phone's copies above the picture (#rm-phone): their number follows the drag too, not only the fill */
    syncRange('#rm-ph-n-in',S.ni); setOut('#rm-ph-n-in',String(NS[S.ni]));
    syncRange('#rm-ph-shift-in',S.shift); setOut('#rm-ph-shift-in',(S.shift>0?'+':'')+sg2(S.shift).replace(/\.00$/,''));
    const pb=$('#rm-ic-play',rail); if(pb) pb.textContent=playLabel();
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#rm-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p>${tx('Your own settings: they match no scenario. The chart shows them.','你自己的设置，不对应任何一个场景。图上显示的就是它。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chSt.bal,chSt.steps), a=$('#rm-ch-acc',rail), q=$('#rm-ch-q',rail); if(!a) return;
    const hide=!chSubmitted, later=tx('after you submit','提交后显示');
    a.textContent=hide?later:pP(e.acc); q.textContent=hide?later:q2(e.q);
    $('#rm-ch-g1',rail).classList.toggle('met',!hide&&e.okQ); $('#rm-ch-g1',rail).classList.toggle('rm-over',!hide&&!e.okQ);
    $$('.seg-ctl[data-key="chB"]',rail).forEach(x=>A.ui.setPressed(x,String(chSt.bal))); $$('.seg-ctl[data-key="chS"]',rail).forEach(x=>A.ui.setPressed(x,String(chSt.steps)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#rm-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const curStep=()=>mode==='guided'?STEPS[step].id:'';
/* length-balanced pairs are named only from part 7, where they are defined (review S7) */
const balSeen=()=>mode!=='guided'||step>=SIX('fix');
/* which panels the stage shows (the main picture first, R7.12) */
function panels(){
  const id=curStep();
  if(mode==='challenge') return {pair:true,scores:true,sig:false,w:true,tr:chSubmitted,trc:true,bon:chSubmitted,hideRes:!chSubmitted};
  if(mode!=='guided') return {pair:true,scores:true,sig:false,w:true,tr:true,trc:true,bon:true};
  return {pair:true,scores:!(id==='mission'||id==='bt'),sig:id==='bt'&&!!pred||id==='shift',w:['train','weights','hack','fix','remember'].includes(id),tr:['train','weights','fix','remember'].includes(id),trc:id!=='fix',bon:['hack','fix','remember'].includes(id)};
}
const EXT=[
 {p:{en:'How do I remove limescale from a kettle?',zh:'水壶里的水垢怎么去掉？'},
  A:{en:'Fill it with equal parts white vinegar and water, boil, and leave it for an hour. Pour it out and rinse twice. Then boil one kettle of plain water and pour it away, so no vinegar taste is left.',zh:'倒入白醋和水各一半，煮开后放一个小时。倒掉，用清水冲两遍。再烧一壶清水倒掉，就不会留下醋味。'},
  B:{en:'Soak it in vinegar.',zh:'用醋泡一下就行。'}},
 {p:{en:'How do I make boiled eggs easy to peel?',zh:'怎样让煮鸡蛋好剥壳？'},
  A:{en:'Put them in ice water for 5 minutes right after boiling, then crack the wide end first.',zh:'煮好后马上放进冰水里泡 5 分钟，再从大的一头敲开剥。'},
  B:{en:'Great question! Many people struggle with this, and there are many opinions… Put them in ice water for 5 minutes right after boiling, then crack the wide end first. I hope this helps, and remember that every kitchen is different…',zh:'这是一个很好的问题！很多人都遇到过，说法也很多……煮好后马上放进冰水里泡 5 分钟，再从大的一头敲开剥。希望对你有帮助，每个厨房的情况都不一样……'}}];
function pairHTML(){
  const P=panels(), e=EX[S.pair], t=EXT[S.pair], w=wNow(), sh=isP('shift')?S.shift:0;
  const ra=score(w,e.A)+sh, rb=score(w,e.B)+sh, mx=Math.max(e.A.L,e.B.L)*1.05;
  /* who chose and who scored (Codex review b9v3, 可选 9): each tag names the person who labelled the pair; the reward model's verdict
     is a line of its own under the two responses, so neither reads as the other */
  const ck=e.ch===0?'A':'B', rk=e.ch===0?'B':'A';
  const pairTip=esc(tx(`In this preference pair, ${ck} is the chosen response and ${rk} the rejected one: the person who labelled the pair picked ${ck}.`,`在这个偏好对里，${ck} 是选中的回答，${rk} 是落选的回答。标注的人在两个回答里选了 ${ck}。`));
  const tag=k=>k===ck?`<span class="rm-tag" data-tipx="${pairTip}">${tx(`the person chose ${k}`,`标注的人选了 ${k}`)}</span>`:`<span class="rm-tag rm-tag-r" data-tipx="${pairTip}">${tx(`the person did not choose ${k}`,`标注的人没选 ${k}`)}</span>`;
  const box=(k,x,txt,r)=>`<div class="rm-ans${k===ck?' rm-chosen':''}"><div class="rm-ah"><b>${k}</b>${tag(k)}<span class="rm-sc mono"${P.scores?'':' hidden'} data-tipx="${esc(tx(`The reward model’s score for ${k}, at training step ${tNow()}.`,`训练到第 ${tNow()} 步时，奖励模型给 ${k} 的分数。`))}">${A.ui.math(`r_${k} = ${sg2(r)}`)}</span></div>`+
    `<p class="rm-at">${esc(L(txt))}</p><div class="rm-len"><span class="rm-lbar"><i style="width:${(x.L/mx*100).toFixed(1)}%"></i></span><span class="mono small">${tx(`${fint(x.L)} tokens`,`${fint(x.L)} 个 token`)}</span></div></div>`;
  /* the reward model's verdict, from the scores as shown */
  const rC=ck==='A'?ra:rb, rR=ck==='A'?rb:ra, vd=!P.scores?'':(sg2(rC)===sg2(rR)?tx('The reward model gives the two responses the same score.','奖励模型给两个回答打的分一样。')
    :(rC>rR?tx(`The reward model also scores ${ck} higher.`,`奖励模型也给 ${ck} 打了更高的分。`):tx(`The reward model, though, scores ${rk} higher.`,`奖励模型却给 ${rk} 打了更高的分。`)));
  const gap=ra-rb, foot=P.scores?tx(`Score gap ${A.ui.math(`r_A - r_B = ${sg2(gap)}`)}: the reward model says a person picks A ${pR(sig(gap))} of the time.`,`分差 ${A.ui.math(`r_A - r_B = ${sg2(gap)}`)}：奖励模型认为人有 ${pR(sig(gap))} 的时候选 A。`)
    :tx('The reward model has not been trained yet: no scores.','奖励模型还没训练，还没有分数。');
  /* Codex R10B (round 3): the challenge is met on the average quality of the best-of-16 picks, and this one fixed example can still be
     judged wrong. The pass state says so on the example itself instead of changing the example or the result to agree */
  const vk=sg2(rC)===sg2(rR)?'tie':(rC>rR?'agree':'disagree'), still=vk==='disagree'&&chPassed();
  const mark=still?` <span class="rm-tag rm-wrong" data-tipx="${esc(tx('The challenge is met: the goal is the average true quality of the responses best-of-16 picks. This one example is still judged wrong.','挑战已经达标：目标看的是 best-of-16 挑出的回答平均的真实质量。这个例子仍然判错。'))}">${tx('still judged wrong','此例仍判错')}</span>`:'';
  const note=still?`<p class="rm-foot small rm-still-n">${tx('The goal looks at the average true quality of the responses best-of-16 picks; a single example can still be judged wrong, and this one is.','目标看的是 best-of-16 挑出的回答平均的真实质量；单个例子仍可能判错，这个例子就是。')}</p>`:'';
  return `<div class="rm-cap">${tx(`Example ${S.pair+1} (illustrative): ${esc(L(t.p))}`,`示例 ${S.pair+1}（示意）：${esc(L(t.p))}`)}</div><div class="rm-anss">${box('A',e.A,t.A,ra)}${box('B',e.B,t.B,rb)}</div>`+
    (vd?`<p class="rm-verdict" data-verdict="${vk}">${vd}${mark}</p>`:'')+`<p class="rm-foot small">${foot}</p>`+note;
}
/* σ curve with the current gap (part 2: the slider; part 4: the example's gap) */
function drawSig(){
  const host=$('#rm-sig',app), W=Math.max(220,Math.round(host.clientWidth||320)), H=120, m={l:34,r:12,t:10,b:24};
  const X=x=>m.l+(W-m.l-m.r)*(x+6)/12, Y=p=>m.t+(H-m.t-m.b)*(1-p);
  const d=isP('shift')?exScores()[0]-exScores()[1]:S.d, p=sig(d);
  let path=''; for(let i=0;i<=120;i++){const x=-6+i*0.1; path+=(i?'L':'M')+X(x).toFixed(1)+' '+Y(sig(x)).toFixed(1);}
  const cap=isP('shift')?tx('Adding the same number to both scores leaves the point on the σ curve where it is','两个分数都加上同一个数，σ 曲线上的点不动'):A.ui.math(tx('P(\\text{person picks A}) = \\sigma(\\text{gap})','P(\\text{人选 A}) = \\sigma(\\text{分差})'));
  host.innerHTML=`<div class="rm-cap">${cap}</div><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx(`Sigma curve; at a gap of ${sg2(d)} the chance is ${pR(p)}`,`σ 曲线；分差 ${sg2(d)} 时概率是 ${pR(p)}`))}">`+
    `<line class="rm-ax" x1="${m.l}" x2="${W-m.r}" y1="${Y(0)}" y2="${Y(0)}"/><line class="rm-ax" x1="${m.l}" x2="${m.l}" y1="${Y(1)}" y2="${Y(0)}"/>`+
    [0,0.5,1].map(v=>`<text class="rm-tk" x="${m.l-4}" y="${Y(v)+3}" text-anchor="end">${v*100}%</text><line class="rm-grid" x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`).join('')+
    [-6,-4,-2,0,2,4,6].map(v=>`<text class="rm-tk" x="${X(v)}" y="${H-8}" text-anchor="middle">${v<0?MINUS+(-v):v}</text>`).join('')+
    `<path class="rm-line" d="${path}"/><line class="rm-guide" x1="${X(d)}" x2="${X(d)}" y1="${Y(0)}" y2="${Y(p)}"/><circle class="rm-dot" cx="${X(d)}" cy="${Y(p)}" r="5"/>`+
    `<text class="rm-lab" x="${Math.min(W-m.r-4,X(d)+8)}" y="${Math.max(m.t+10,Y(p)-6)}" text-anchor="${X(d)>W-90?'end':'start'}">${pR(p)}</text></svg>`+
    `<div class="rm-axl small muted">${tx('gap ','分差 ')}${GAPM()}</div>`;
}
const FEATN=()=>[tx('Content','内容'),tx('Length','长度'),tx('List','列表')];
function drawW(){
  const host=$('#rm-w',app), w=wNow(), mx=1.2, big=Math.abs(w[1])>=Math.max(Math.abs(w[0]),Math.abs(w[2]))-1e-9&&Math.abs(w[1])>0.05;
  const html=`<div class="rm-cap">${tx(`What the reward model learned: one weight per feature (step ${tNow()})`,`奖励模型学到的：每个特征一个权重（第 ${tNow()} 步）`)}</div><div class="rm-wrows">`+
    w.map((v,i)=>{const pc=Math.min(1,Math.abs(v)/mx)*50; return `<div class="rm-wrow${i===1&&big?' on':''}" data-tipx="${esc(FEATN()[i]+tx(`: weight ${sg2(v)}. `,`：权重 ${sg2(v)}。`)+TIP.feat(i))}"><span class="rm-wl">${esc(FEATN()[i])}</span><span class="rm-wt"><b class="rm-w0"></b><i class="${v<0?'neg':''}" style="${v<0?`right:50%;width:${pc.toFixed(2)}%`:`left:50%;width:${pc.toFixed(2)}%`}"></i></span><em class="mono">${sg2(v)}</em></div>`;}).join('')+`</div>`+
    `<div class="rm-wleg small muted">${tx('Bars grow right for a positive weight, left for a negative one; the thick outline marks length when it weighs most.','权重为正时横条往右，为负时往左；长度的权重最大时，它的横条加粗框。')}</div>`;
  if(host._h!==html){host.innerHTML=html; host._h=html;}
}
function miniChart(o){/* x 0..xmax, series [{vals, cls, label}] */
  const W=o.W, H=o.H, m={l:40,r:o.r||14,t:10,b:22};
  const X=x=>m.l+(W-m.l-m.r)*(o.xlog?Math.log2(x)/8:x/o.xmax), Y=v=>m.t+(H-m.t-m.b)*(1-(v-o.y0)/(o.y1-o.y0));
  let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria)}">`;
  o.yt.forEach(v=>{s+=`<line class="rm-grid" x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="rm-tk" x="${m.l-4}" y="${Y(v)+3}" text-anchor="end">${o.yf(v)}</text>`;});
  o.xt.forEach(v=>{s+=`<text class="rm-tk" x="${X(v)}" y="${H-7}" text-anchor="middle">${v}</text>`;});
  if(o.goal!=null) s+=`<line class="rm-goalln" x1="${m.l}" x2="${W-m.r}" y1="${Y(o.goal)}" y2="${Y(o.goal)}"/>`;
  o.series.forEach(se=>{let d=''; se.pts.forEach(([x,v],i)=>{d+=(i?'L':'M')+X(x).toFixed(1)+' '+Y(v).toFixed(1);}); if(d) s+=`<path class="${se.cls}" d="${d}"/>`;});
  if(o.mark!=null) s+=`<line class="rm-guide" x1="${X(o.mark)}" x2="${X(o.mark)}" y1="${m.t}" y2="${H-m.b}"/>`;
  (o.dots||[]).forEach(([x,v,c])=>{s+=`<circle class="${c}" cx="${X(x)}" cy="${Y(v)}" r="4"/>`;});
  return s+`</svg>`;
}
/* polish round 4 (2026-10-17, B4): on a wide stage the blocks sit in two columns and the charts are drawn shorter, so the stage fits a
   1440 × 900 screen. The heights (and whether best-of-n's two charts sit side by side) are CSS variables on the chart's box, set in
   reward-model.css and read when a chart is drawn; without them (phones, narrower stages) everything is drawn as before. */
const cssNum=(host,v,d)=>{const x=parseFloat(getComputedStyle(host).getPropertyValue(v)); return x>0?x:d;};
function drawTr(){
  const r=R(), Tm=Tn(), t=tNow(), host=$('#rm-tr-charts',app);
  host.hidden=!panels().trc;   // shown before it is measured: hidden in part 7, it measured 0 px wide and part 8 drew its charts 150 px wide
  const W=Math.max(150,Math.floor(((host.clientWidth||300)-12)/2)), H=cssNum(host,'--rm-trh',118);
  const lp=[], ap=[]; const stepN=Math.max(1,Math.round(Tm/100)); for(let i=0;i<=t;i+=stepN){lp.push([i,r.LOSS[i]]); ap.push([i,r.acc[i]]);} if(lp.length&&lp[lp.length-1][0]!==t){lp.push([t,r.LOSS[t]]); ap.push([t,r.acc[t]]);}
  const xt=Tm===400?[0,200,400]:(Tm===50?[0,25,50]:[0,Tm/2,Tm]);
  $('#rm-tr-cap',app).textContent=tx(`Training on ${fint(r.N+r.bal)} pairs: step ${t} of ${Tm}`,`在 ${fint(r.N+r.bal)} 个偏好对上训练：第 ${t} / ${Tm} 步`);
  if(host.hidden){drawPlayer(); return;}
  host.innerHTML=`<figure class="rm-fig"><figcaption>${tx(`Loss (training pairs): ${q2(r.LOSS[t])}`,`loss（训练用的偏好对）：${q2(r.LOSS[t])}`)}</figcaption>`+miniChart({W,H,xmax:Tm,y0:0.3,y1:0.7,yt:[0.3,0.5,0.7],yf:v=>fixd(v,1),xt,series:[{pts:lp,cls:'rm-line'}],dots:[[t,r.LOSS[t],'rm-dot']],aria:tx(`Loss ${q2(r.LOSS[t])} at step ${t}`,`第 ${t} 步的 loss 是 ${q2(r.LOSS[t])}`)})+`</figure>`+
    `<figure class="rm-fig"><figcaption>${tx(`Accuracy on 500 held-out pairs: ${pP(r.acc[t])}`,`留出的 500 对上的准确率：${pP(r.acc[t])}`)}</figcaption>`+miniChart({W,H,xmax:Tm,y0:0.4,y1:0.9,yt:[0.5,0.7,0.9],yf:v=>Math.round(v*100)+'%',xt,series:[{pts:ap,cls:'rm-line'}],dots:[[t,r.acc[t],'rm-dot']],aria:tx(`Accuracy ${pP(r.acc[t])} at step ${t}`,`第 ${t} 步的准确率是 ${pP(r.acc[t])}`)})+`</figure>`;
  drawPlayer();
}
function drawBon(){
  const host=$('#rm-bon',app), cols=cssNum(host,'--rm-bcols',1), W=cols>1?Math.max(150,Math.floor(((host.clientWidth||300)-12)/2)):Math.max(220,Math.round(host.clientWidth||320)), b=bonAt(runS(),tNow()), k=peakOf(b), ch=mode==='challenge';
  /* part 6: the curves reach only as far as the reader has raised n, and turn red once n passes the peak (review S11, R3.8, R5.11) */
  const lim=isP('hack')?Math.max(hackSeen,S.ni):NS.length-1, ni=ch?NS.indexOf(CH_N):S.ni, hackOn=k<NS.length-1&&lim>k;
  const sp=b.slice(0,lim+1).map(x=>[x.n,x.s]), qp=b.slice(0,lim+1).map(x=>[x.n,x.q]);
  const qn=hackOn?qp.slice(0,k+1):qp, qr=hackOn?qp.slice(k):[];
  const smin=Math.min(0,...b.map(x=>x.s)), smax=Math.max(4,...b.map(x=>x.s));
  const top=miniChart({W,H:cssNum(host,'--rm-bsh',104),xlog:true,xmax:256,y0:Math.floor(smin),y1:Math.ceil(smax),yt:[Math.floor(smin),Math.ceil(smax)],yf:v=>v<0?MINUS+(-v):String(v),xt:[1,4,16,64,256],series:[{pts:sp,cls:'rm-line'}],mark:NS[ni],dots:[[NS[ni],b[ni].s,'rm-dot']],aria:tx('Reward-model score of the pick against n','挑中的回答的奖励模型分数随 n 的变化')});
  const bot=miniChart({W,H:cssNum(host,'--rm-bqh',118),xlog:true,xmax:256,y0:5,y1:9,yt:[5,7,9],yf:v=>String(v),xt:[1,4,16,64,256],goal:ch?CH_Q:null,series:[{pts:qn,cls:'rm-line'},{pts:qr,cls:'rm-line rm-bad'}],mark:NS[ni],dots:[[NS[ni],b[ni].q,ni>k&&hackOn?'rm-dot rm-bad':'rm-dot']],aria:tx('True quality of the pick against n','挑中的回答的真实质量随 n 的变化')});
  host.innerHTML=`<div class="rm-cap">${tx(`Best-of-n with this reward model (n = ${NS[ni]} marked)`,`用这个奖励模型做 best-of-n（标出的是 n = ${NS[ni]}）`)}</div>`+
    `<figure class="rm-fig1"><figcaption>${tx(`Reward-model score of the pick: ${sg2(b[ni].s)}`,`挑中的回答的分数：${sg2(b[ni].s)}`)}</figcaption>${top}</figure>`+
    `<figure class="rm-fig1"><figcaption>${tx(`True quality of the pick, out of 10: ${q2(b[ni].q)}`,`挑中的回答的真实质量（满分 10）：${q2(b[ni].q)}`)}</figcaption>${bot}</figure>`+
    `<ul class="rm-bleg small"><li><i class="rm-sw"></i>${tx('average over all draws','所有抽法的平均')}</li>${hackOn?`<li><i class="rm-sw rm-bad"></i>${tx(`the score still rises, the true quality falls (after n = ${NS[k]})`,`分数还在涨、真实质量在降（n = ${NS[k]} 之后）`)}</li>`:''}${ch?`<li><i class="rm-sw rm-gsw"></i>${tx(`goal: ${CH_Q}`,`目标：${CH_Q}`)}</li>`:''}<li>${tx('x axis: n, responses drawn','横轴：n，抽几个回答')}</li></ul>`;
}
function drawPlayer(){
  const pl=$('#rm-player',app); if(!pl) return; const t=tNow(), Tm=Tn();
  const sig_=[playing,spd,t,Tm,A.lang()].join('|'); if(pl._sig===sig_) return; pl._sig=sig_;
  const scrub=pl.querySelector('input[data-key="scrub"]');
  if(scrub&&pl._built===A.lang()+Tm){scrub.value=t; A.ui.setRange(scrub); pl.querySelector('[data-act="play"]').textContent=playLabel(); $$('.seg-ctl[data-key="spd"]',pl).forEach(g=>A.ui.setPressed(g,spd)); pl.querySelector('.rm-count').textContent=tx(`step ${t} of ${Tm}`,`第 ${t} / ${Tm} 步`); return;}
  pl._built=A.lang()+Tm;
  pl.innerHTML=`<button type="button" class="btn ghost sm rm-play" data-act="play" data-tipx="${esc(tx('Play the training at the chosen speed; press again to pause. At the end, the button plays it again from step 0.','按选定的速度播放训练；再按一次暂停。播完后，这个按钮从第 0 步重新播放。'))}">${playLabel()}</button>`+
    A.ui.seg({id:'rm-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd})+
    `<div class="rm-scrub"><input type="range" min="0" max="${Tm}" step="1" value="${t}" data-key="scrub" aria-label="${esc(tx('Training step shown','显示的训练步'))}"><span class="rm-count mono">${tx(`step ${t} of ${Tm}`,`第 ${t} / ${Tm} 步`)}</span></div>`;
  A.ui.initRanges(pl);
}
function playLabel(){return playing?tx('❚❚ Pause','❚❚ 暂停'):(tNow()>=Tn()?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));}
function statusOf(){
  if(mode==='challenge'){const e=chEval(chSt.bal,chSt.steps);
    if(!chSubmitted) return {cls:'neutral',txt:tx('Submit to see the result','提交后显示结果'),short:tx('not submitted','待提交'),tip:tx('The accuracy and the true quality of the picks show once you submit.','提交以后才显示准确率和挑中的回答的真实质量。')};
    return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),tip:tx(`Green when best-of-16’s true quality is at least ${CH_Q}. Now: ${q2(e.q)}.`,`best-of-16 的真实质量至少 ${CH_Q} 分时，徽章是绿色。现在是 ${q2(e.q)} 分。`)};}
  const P=panels(), r=R(), t=tNow();
  if(!P.scores) return {cls:'neutral',txt:tx('Not trained yet','还没训练'),short:tx('not trained','还没训练'),tip:tx('The reward model has no scores until it is trained.','奖励模型训练之前还没有分数。')};
  if(P.bon){const b=bonAt(runS(),t), x=b[S.ni], k=peakOf(b), bad=S.ni>k&&k<NS.length-1;
    return {cls:bad?'bad':'neutral',txt:tx(`n = ${NS[S.ni]}: quality ${q2(x.q)}`,`n = ${NS[S.ni]}：质量 ${q2(x.q)} 分`),short:tx(`quality ${q2(x.q)}`,`质量 ${q2(x.q)} 分`),tip:tx('The true quality of the best-of-n pick, out of 10; red once it falls while the score still rises.','best-of-n 挑中的回答的真实质量（满分 10）。分数还在涨、质量却在降时变红。')};}
  return {cls:'neutral',txt:tx(`Accuracy ${pR(r.acc[t])}`,`准确率 ${pR(r.acc[t])}`),short:tx(`accuracy ${pR(r.acc[t])}`,`准确率 ${pR(r.acc[t])}`),tip:tx('Accuracy on the 500 held-out pairs at the training step shown.','显示的这一步，在留出的 500 对上的准确率。')};
}
function drawHead(){
  const t=$('#rm-st-title',app), sub=$('#rm-st-sub',app), st=$('#rm-status',app), r=R();
  t.textContent=mode==='challenge'?tx('New data: 1,000 pairs, length tied closely to quality','新的数据：1,000 对，长度和质量绑得很紧'):tx(`A small reward model trained on ${fint(r.N)} preference pairs`,`在 ${fint(r.N)} 个偏好对上训练的小奖励模型`);
  /* a setting is named on the stage only once a part has explained it (review S7): the data's length from part 5, balanced pairs from part 7 */
  const s=runS(), parts=[], g=mode==='guided';
  if(!s.ch&&(!g||step>=SIX('weights'))){parts.push(s.rel==='tied'?tx('length follows quality','长度跟着质量走'):tx('length unrelated to quality','长度和质量无关')); parts.push(s.tau===1?tx('people mostly agree','标注的人大多一致'):tx('people often disagree','标注的人常有分歧'));}
  if(balSeen()) parts.push(s.bal?tx(`${s.bal} length-balanced pairs added`,`另加 ${s.bal} 对长短平衡的偏好对`):tx('no length-balanced pairs','没有长短平衡的偏好对'));
  sub.textContent=parts.join(tx(', ','，'));
  const x=statusOf(); st.className='status '+x.cls; st.textContent=x.txt; st.dataset.tipx=x.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').replace('−','-').match(/^[<>]?(-?\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const r=R(), t=tNow(), w=wNow(), P=panels(), m=[];
  m.push(balSeen()?{k:'pairs',label:tx('Pairs used','用了多少对'),v:fint(r.N+r.bal),sub:tx(`${fint(r.N)} + ${r.bal} balanced`,`${fint(r.N)} 对 + ${r.bal} 对长短平衡的`),tip:tx('Preference pairs the reward model was trained on: the data set’s pairs plus any length-balanced pairs added.','奖励模型训练用的偏好对：数据本来的对数，加上另加的长短平衡的对数。')}
    :{k:'pairs',label:tx('Pairs used','用了多少对'),v:fint(r.N+r.bal),sub:tx('training pairs','训练用的偏好对'),tip:tx('Preference pairs the reward model was trained on.','奖励模型训练用的偏好对。')});
  if(P.scores&&!P.hideRes) m.push({k:'acc',pts:true,label:tx('Held-out accuracy','留出准确率'),v:pP(r.acc[t]),sub:tx(`500 pairs, step ${t}`,`500 对，第 ${t} 步`),tip:TIP.held()});
  if(P.w) m.push({k:'wl',label:tx('Length weight','长度的权重'),v:sg2(w[1]),sub:tx(`content: ${sg2(w[0])}`,`内容的权重：${sg2(w[0])}`),tip:tx('How much one unit of the length feature adds to the score (doubling the length adds 1.7 units).','长度特征每多 1，分数加多少（长度翻倍，这个特征加 1.7）。')});
  if(P.bon&&!(isP('hack')&&hackSeen<6)){const b=bonAt(runS(),t), k=peakOf(b); m.push({k:'bestn',label:tx('Best n for true quality','真实质量最高时的 n'),v:String(NS[k]),sub:tx(`quality ${q2(b[k].q)} out of 10`,`质量 ${q2(b[k].q)} 分（满分 10）`),tip:tx('The n at which best-of-n picks responses of the highest true quality. 256 means it is still rising at the end of the chart.','best-of-n 选出回答的真实质量最高时的 n。256 表示到图的右端还在上升。')});}
  return m;
}
function drawMetrics(){
  const host=$('#rm-metrics',app), defs=metricsDef(), sg=defs.map(m=>m.k).join();
  if(host._sig!==sg){host._sig=sg; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!playing&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>=0&&b>=0) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(!playing){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
function mathHTML(){
  const r=R(), t=tNow(), w=wNow(), e=EX[S.pair], fa=phi(e.A), fb=phi(e.B), ra=score(w,e.A), rb=score(w,e.B), hd=s=>`<p class="rm-mh"><b>${esc(s)}</b></p>`, f3=v=>{const s=fixd(Math.abs(v),3); return (v<0&&+s!==0?MINUS:'')+s;}, fp=v=>v<0?`(${f3(v)})`:f3(v);
  const nc=r.acc[t]*NH;
  /* H2 (the owner, 2026-10-09): a few short blocks, each a lead line, the named formula, the reader's numbers under it and the result */
  const ft=v=>v<0?`({${f3(v)}})`:f3(v), sc=(k,fx)=>EQ(tx(`score of ${k}, ${A.ui.math('r_'+k)}`,`${k} 的分数 ${A.ui.math('r_'+k)}`),
    [SCOREF().replace(/^r = /,''),`${ft(w[0])} \\times ${ft(fx[0])} + ${ft(w[1])} \\times ${ft(fx[1])} + ${ft(w[2])} \\times ${ft(fx[2])}`,f3(k==='A'?ra:rb)]);
  const CH=tx('chosen','选中'), RJ=tx('rejected','落选');
  let h=hd(tx(`Score of example ${S.pair+1}, step ${t}`,`示例 ${S.pair+1} 的分数，第 ${t} 步`))+
    `<p>${tx('A response’s score is a weighted sum of its three features. The weights are what training has learned so far:','一个回答的分数，是它三个特征的加权和。权重是训练到这一步学到的：')}</p>`+
    sc('A',fa)+sc('B',fb)+
    DEFS([[A.ui.math('w_1, w_2, w_3'),tx(`the three weights the reward model has learned, now ${f3(w[0])}, ${f3(w[1])} and ${f3(w[2])}`,`奖励模型学到的三个权重，现在是 ${f3(w[0])}、${f3(w[1])}、${f3(w[2])}`)],
      [tx('content','内容'),tx(`${A.ui.math('\\frac{c - 6}{1.5}')}, where c is what the small reward model reads of the content`,`${A.ui.math('\\frac{c - 6}{1.5}')}，c 是从文字里读出的内容好坏`)],
      [tx('length','长度'),A.ui.math(tx('\\frac{\\ln(\\text{tokens} / 150)}{0.4}','\\frac{\\ln(\\text{token 数} / 150)}{0.4}'))],
      [tx('list','列表'),tx('+0.5 with bullet points, −0.5 without','用了列表是 +0.5，没用是 −0.5')]])+
    hd(tx('Bradley–Terry','Bradley–Terry'))+`<p>${tx('The gap between the two scores, passed through σ, is the chance the reward model gives that a person picks A:','两个分数的差经过 σ，就是奖励模型认为人选 A 的概率：')}</p>`+
    EQ(A.ui.math(tx('P(\\text{person picks A})','P(\\text{人选 A})')),['\\sigma(r_A - r_B)',`\\sigma(${f3(ra)} - ${ft(rb)})`,`${fixd(sig(ra-rb)*100,1)}%`])+
    hd(tx('Loss over the training pairs','训练用的偏好对上的 loss'))+`<p>${tx('Each training pair’s loss is small when the chosen response scores well above the rejected one, and large when the order is wrong:','每个训练用的偏好对，选中的回答分数比落选的高得越多，loss 越小；排反了，loss 就大：')}</p>`+
    EQ(tx('loss of one pair','一对的 loss'),[`-\\log\\, \\sigma(r_{\\text{${CH}}} - r_{\\text{${RJ}}})`])+
    `<p>${tx(`${A.ui.math('r_{\\text{chosen}}')} and ${A.ui.math('r_{\\text{rejected}}')} are the scores of the chosen and the rejected response of one pair. At step ${t}, the mean over the ${fint(r.N+r.bal)} training pairs is <b>${f3(r.LOSS[t])}</b>.`,`${A.ui.math('r_{\\text{选中}}')} 和 ${A.ui.math('r_{\\text{落选}}')} 是同一对里选中和落选的回答的分数。第 ${t} 步，训练用的 ${fint(r.N+r.bal)} 对上的平均 loss 是 <b>${f3(r.LOSS[t])}</b>。`)}</p>`+
    hd(tx('Gradient step','梯度下降的一步'))+`<p>${tx('Each step moves the weights so that every pair’s chosen response scores higher than its rejected one; pairs ranked wrong push hardest:','每一步都挪动权重，让每一对里选中的回答分数更高；排反了的对推得最用力：')}</p>`+FX(GRADF())+
    DEFS([[tx('Δφ','Δφ'),tx(`${A.ui.math('\\phi_{\\text{chosen}} - \\phi_{\\text{rejected}}')}: how the two responses of one pair differ in their three features (content, length, list). A response’s score is ${A.ui.math('r = w \\cdot \\phi')}`,`${A.ui.math('\\phi_{\\text{选中}} - \\phi_{\\text{落选}}')}，同一对里两个回答的三个特征（内容、长度、列表）之差。一个回答的分数是 ${A.ui.math('r = w \\cdot \\phi')}`)],
      [tx('η = 0.15','η = 0.15'),tx('the learning rate: how far each step moves the weights','学习率，每一步把权重挪多远')],
      [tx('mean','mean'),tx(`the average over all ${fint(r.N+r.bal)} training pairs`,`对训练用的 ${fint(r.N+r.bal)} 个偏好对取平均`)]])+
    hd(tx('Held-out accuracy','留出准确率'))+(t===0?`<p>${tx(`At step 0 every weight is 0, so both responses of every pair score 0. A tie counts as half right: ${NH} × 0.5 / ${NH} = <b>0.500</b>.`,`第 0 步时所有权重都是 0，每一对的两个回答都是 0 分。打平算对一半：${NH} × 0.5 / ${NH} = <b>0.500</b>。`)}</p>`
      :`<p>${tx(`${Math.round(nc)} of ${NH} held-out pairs score the chosen response higher`,`${NH} 对留出的偏好对里，有 ${Math.round(nc)} 对选中的回答分数更高`)}${tx(': ','：')}${Math.round(nc)} / ${NH} = <b>${fixd(r.acc[t],3)}</b></p>`);
  const b=bonAt(runS(),t), ni=mode==='challenge'?NS.indexOf(CH_N):S.ni;
  h+=hd(tx('Best-of-n','best-of-n'))+`<p>${tx(`Rank the M = ${fint(M)} responses by score, from the lowest (j = 1) to the highest. Draw n of them with replacement and keep the top-scored one. The chance that it is the j-th:`,`把 M = ${fint(M)} 个回答按分数从低到高排，j = 1 是分数最低的。从中有放回地抽 n 个，留下分数最高的那个。它是第 j 个的概率是：`)}</p>`+FX(BONF())+
    `<p>${tx(`So at n = ${NS[ni]} the pick’s average true quality is <b>${fixd(b[ni].q,3)}</b>. Of the picks, ${pR(b[ni].pad)} are padded responses.`,`所以 n = ${NS[ni]} 时，挑中的回答平均真实质量是 <b>${fixd(b[ni].q,3)}</b>。挑中的回答里，有 ${pR(b[ni].pad)} 是加了废话的。`)}</p>`;
  return h;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const P=panels();
  $('#rm-graph',app).classList.toggle('rm-runfirst',isP('train'));   // G6: in part 3 the run is the picture; on a phone it leads the stage
  drawHead();
  const pr=$('#rm-pair',app), ph=pairHTML(); if(pr._h!==ph){pr.innerHTML=ph; pr._h=ph;}
  const sg_=$('#rm-sig',app); sg_.hidden=!P.sig; if(P.sig) drawSig();
  const wv=$('#rm-w',app); wv.hidden=!P.w; if(P.w) drawW();
  const tr=$('#rm-tr',app); tr.hidden=!P.tr; if(P.tr) drawTr();
  const bn=$('#rm-bon',app); bn.hidden=!P.bon; if(P.bon) drawBon();
  drawMetrics();
  const mb=$('#rm-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* while the run plays, only the moving parts redraw */
function renderFrame(){
  if(!app) return; const P=panels();
  const pr=$('#rm-pair',app), ph=pairHTML(); if(pr._h!==ph){pr.innerHTML=ph; pr._h=ph;}
  if(P.w) drawW(); if(P.tr) drawTr(); if(P.bon) drawBon(); drawHead(); drawMetrics(); A.dock.refresh();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change the balanced pairs and the training steps in the task card. The data are fixed, 1,000 new pairs, and best-of-16 uses the same chat model’s responses.','挑战中只能在任务卡里改长短平衡的数据和训练步数。数据是固定的 1,000 对新的偏好对，best-of-16 用的也是同一个聊天模型的回答。');
  if(mode!=='guided') return '';
  return tx('In the guided parts, each part’s card sets the data. To change it yourself, go to Scenarios.','分步引导里，用哪批数据由每一部分的任务卡决定。想自己改，去“应用场景”。');
}
function syncControls(){
  const c=$('#rm-controls',app), g=mode==='guided', ch=mode==='challenge';
  $$('#rm-controls .seg-ctl[data-key="rel"]',app).forEach(x=>A.ui.setPressed(x,S.rel));
  $$('#rm-controls .seg-ctl[data-key="tau"]',app).forEach(x=>A.ui.setPressed(x,String(S.tau)));
  $$('#rm-controls .seg-ctl[data-key="N"]',app).forEach(x=>A.ui.setPressed(x,String(S.N)));
  $$('#rm-controls .seg-ctl[data-key="bal"]',app).forEach(x=>A.ui.setPressed(x,String(S.bal)));
  syncRange('#rm-ctl-n-in',S.ni); setOut('#rm-ctl-n-in',String(NS[S.ni]));
  c.classList.toggle('locked',ch);
  ['rel','tau','N','bal','n','reset'].forEach(k=>{const e=$(`#rm-ctl-${k}-wrap`,app); if(e) e.hidden=g||ch;});
  const ln=$('#rm-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; A.announce(statusOf().txt);},400);}

/* =====================================================================
   the training run: play / pause / speed / replay (R3.9, R3.20)
   ===================================================================== */
/* G6 (the owner's review of 2026-10-08): a training curve of a few hundred steps plays in about 3 s at the default speed (Medium) and
   about 1.5 s at Fast; Scenarios and Settings show the end state at once */
const DUR={slow:8000,mid:3000,fast:1500};
const keyOf=(s,Tm)=>dsKey(s)+'|'+s.bal+'|'+Tm;
function tick(ts){
  raf=0; if(!app||!playing) return;
  const dt=lastT?ts-lastT:16; lastT=ts; const Tm=Tn();
  prog=Math.min(1,prog+dt/DUR[spd]); tShow=Tm*prog*prog;   // the first half of the time shows the first quarter of the steps
  if(tShow>=Tm){playing=false; tShow=Tm; watched.add(runKey()); wantObs=true; render(); wantObs=false; revealSoonIf(); return;}
  renderFrame(); raf=requestAnimationFrame(tick);
}
function play(){if(tNow()>=Tn()) tShow=0; prog=Math.sqrt(Math.max(0,tShow)/Tn()); playing=true; lastT=0; if(A.reduceMotion()){finishRun(); return;} if(!raf) raf=requestAnimationFrame(tick); drawPlayer(); const pb=$('#rm-ic-play',app); if(pb) pb.textContent=playLabel();}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; if(app){drawPlayer(); const pb=$('#rm-ic-play',app); if(pb) pb.textContent=playLabel();}}
function finishRun(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; tShow=Tn(); watched.add(runKey()); wantObs=true; render(); revealSoon(); wantObs=false;}
function revealSoonIf(){if(isP('train')){revealSoon();}}
/* show a run: finished at once if it was watched (or nothing animates here), else play it (R3.20) */
function showRun(auto){pause(); if(watched.has(runKey())||!auto){tShow=Tn(); if(!auto) watched.add(runKey()); render(); return;} tShow=0; render(); play();}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p); if(!RELS[S.rel]||S.rel==='strong') S.rel='tied'; if(![1,3].includes(S.tau)) S.tau=1; if(![50,1000].includes(S.N)) S.N=1000; if(![0,25,50,100,200].includes(S.bal)) S.bal=0; S.ni=Math.max(0,Math.min(NS.length-1,S.ni|0)); render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
/* G6: a new run plays only where it can be seen. On a phone the stage sits above the card the reader is clicking in, so parts 5 and 7
   show the new run's end state at once there (its result is in the card; ↺ Replay plays it) */
const runSeen=()=>innerWidth>760;
function setRel(v){wTried.add(v); S.rel=v; showRun(runSeen());}
function setBal(v){fixTried.add(v); S.bal=v; showRun(runSeen());}
function setShift(v){shiftMoved=true; setState({shift:v});}
function setN(i){nMax=Math.max(nMax,i); if(isP('hack')) hackSeen=Math.max(hackSeen,i); setState({ni:i});}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='bt'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the four answers in the card.','向下滚到卡片里的四个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='hack') return nCtl('rm-ph-n');
  if(id==='shift') return shiftCtl('rm-ph-shift');
  return '';
}
function renderPhone(){const el=$('#rm-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; pause(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#rm-stage',app); if(st) st.scrollTop=0;
  if(s.id==='train'){tShow=watched.has(keyOf({...BASE},T))?T:0;}
  else tShow=T;
  if(s.id==='hack') nMax=Math.max(nMax,S.ni);
  renderRail();
  if(s.id==='train'&&!watched.has(keyOf({...BASE},T))) play();
}
function revealStep(){A.ui.reveal($('#rm-rail .step.current',app)||$('#rm-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), and stop when the reader has scrolled on (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick_=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick_,120); return;}
    const st=$('#rm-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#rm-rail .step.current',app)||$('#rm-rail .railcard',app)||$('#rm-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick_,A.reduceMotion()?30:150);}
/* the charts are drawn for their box's width and, on a wide stage, the heights in reward-model.css: the stage changes width without a
   window resize when the page puts the step list beside it or leaves the primer, so they are drawn again when its width changes */
let stRO=null, stW=0;
function watchStage(){if(stRO){stRO.disconnect(); stRO=null;} const st=app&&$('#rm-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w;
    const P=panels(); if(P.sig) drawSig(); if(P.tr) drawTr(); if(P.bon) drawBon();}));
  stRO.observe(st);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#rm-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#rm-insight',app)); else revealBox($('#rm-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; pause(); mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chSt={...CH_START}; S={...BASE,pair:1};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE,pair:1};}
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  tShow=Tn(); watched.add(runKey());
  renderRail();
}
const token=()=>[S.rel,S.tau,S.N,S.bal,S.ni].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [rel,tau,N,bal,ni]=p;
  if(!['tied','none'].includes(rel)||!['1','3'].includes(tau)||!['50','1000'].includes(N)||!['0','25','50','100','200'].includes(bal)||!/^[0-8]$/.test(ni)) return null;
  return {...BASE,rel,tau:+tau,N:+N,bal:+bal,ni:+ni,pair:1};
}

/* =====================================================================
   interview drill: one pair's loss and gradient, seeded
   ===================================================================== */
function makeDrill(seed){const rnd=mulberry(seed*104729+7), r1=v=>Math.round(v*10)/10; const rc=r1(-1+rnd()*3), rr=r1(-1+rnd()*3);
  const d=+(rc-rr).toFixed(1); return {seed,rc,rr,d,loss:Math.log1p(Math.exp(-d)),g:sig(-d)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#rm-drill-q',app), a=$('#rm-drill-a',app);
  const n=v=>v<0?MINUS+fixd(-v,1):fixd(v,1);
  qd.innerHTML=tx(`A reward model scores the chosen response <b>${n(d.rc)}</b> and the rejected one <b>${n(d.rr)}</b>. What is the Bradley–Terry loss of this pair, and its gradient with respect to each score?`,
                  `奖励模型给选中的回答打 <b>${n(d.rc)}</b> 分，给落选的打 <b>${n(d.rr)}</b> 分。这一对的 Bradley–Terry loss 是多少？它对两个分数的梯度各是多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  /* H2 (the owner, 2026-10-09): three short blocks, each named on the left with the numbers under the formula */
  const nt=v=>v<0?`(${n(v)})`:n(v), G=tx('gap','分差');
  const blocks=EQ(`${G}`,[tx('r_{\\text{chosen}} - r_{\\text{rejected}}','r_{\\text{选中}} - r_{\\text{落选}}'),`${n(d.rc)} - ${nt(d.rr)}`,n(d.d)])+
    EQ('loss',[`-\\log\\, \\sigma(\\text{${G}})`,`\\log(1 + e^{-\\text{${G}}})`,`\\log(1 + e^{-${nt(d.d)}})`,fixd(d.loss,3)])+
    EQ(A.ui.math(tx('\\frac{∂\\, \\text{loss}}{∂ r_{\\text{chosen}}}','\\frac{∂\\, \\text{loss}}{∂ r_{\\text{选中}}}')),[`-\\sigma(-\\text{${G}})`,`-${fixd(d.g,3)}`])+
    EQ(A.ui.math(tx('\\frac{∂\\, \\text{loss}}{∂ r_{\\text{rejected}}}','\\frac{∂\\, \\text{loss}}{∂ r_{\\text{落选}}}')),[`+\\sigma(-\\text{${G}})`,`+${fixd(d.g,3)}`]);
  a.innerHTML=tx(`<p>First the gap between the two scores, then the loss of the pair, then how the loss changes with each score:</p>${blocks}<p>The two gradients are equal and opposite: training pushes the chosen score up and the rejected one down by the same amount. The push ${A.ui.math('\\sigma(-\\text{gap})')} is large when the pair is ranked wrong and fades once the chosen one is well ahead.</p>`,
                 `<p>先算两个分数的差，再算这一对的 loss，最后算 loss 随每个分数怎么变：</p>${blocks}<p>两个梯度大小相等、方向相反：训练把选中的分数往上推、落选的往下推，推得一样多。推力 ${A.ui.math('\\sigma(-\\text{分差})')} 在排反了的时候大，选中的领先很多以后就变小。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const TERM={sigma:()=>`<div>${TIP.sigma()}</div><div class="fx">${A.ui.math('\\sigma(x) = \\frac{1}{1 + e^{-x}}')}</div>`,
  feat:()=>`<div>${esc(tx('The small reward model sees three features of a response:','小奖励模型能看到回答的三个特征：'))}</div><ul class="rm-tipl">${FEATN().map((k,i)=>`<li><b>${esc(k)}</b><span>${esc(TIP.feat(i))}</span></li>`).join('')}</ul>`};
const termTipHTML=k=>TERM[k]?TERM[k]():'';
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  return '';
}
function jump(k){
  if(!app) return;
  if(k==='math'){const d=$('#rm-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#rm-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...BASE,pair:S.pair}; showRun(false);}); return;}
  if(a==='play'){if(playing) pause(); else play(); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chSt.bal,chSt.steps).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#rm-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#rm-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#rm-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='scrub'){pause(); tShow=v; if(v>=Tn()) watched.add(runKey()); renderFrame(); if(v>=Tn()) render(); return;}
    if(k==='d'){dMoved=true; byLearner(()=>setState({d:v})); return;}
    if(k==='shift'){byLearner(()=>setShift(v)); return;}
    if(k==='ni'){bySettings(e.target,()=>setN(v)); return;}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#rm-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#rm-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#rm-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); revealSoon(); return;}
    if(d.lever!=null&&isP('weights')){const f=()=>byLearner(()=>setRel(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.lever!=null&&isP('fix')){const f=()=>byLearner(()=>setBal(+d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='spd'){spd=d.v; A.ui.setPressed(seg,spd); const pl=$('#rm-player',app); if(pl) pl._sig=''; drawPlayer(); return;}
      if(k==='rel') bySettings(b,()=>{S.rel=d.v; showRun(false);});
      else if(k==='tau') bySettings(b,()=>{S.tau=+d.v; showRun(false);});
      else if(k==='N') bySettings(b,()=>{S.N=+d.v; showRun(false);});
      else if(k==='bal') bySettings(b,()=>{S.bal=+d.v; showRun(false);});
      else if(k==='chB') byLearner(()=>{chSt.bal=+d.v; pause(); tShow=Tn(); render();});
      else if(k==='chS') byLearner(()=>{chSt.steps=+d.v; pause(); tShow=Tn(); render();});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>{Object.assign(S,p.p,{bal:0}); showRun(false);}); /* G6: Scenarios show the end state at once */ requestAnimationFrame(()=>{if(app) revealBox($('#rm-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#rm-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12, under the rule of 07:13 UTC): four sections of text and formulas read before
   the guided parts: why comparisons (the limits of demonstrations are the SFT primer's, linked), what a reward model is and reads and
   outputs, the Bradley–Terry model and the loss maximum likelihood gives, and why a reward model is only an approximation (it learns
   whatever separates chosen from rejected answers, and it is not held to anything out of distribution). As in the rest of this lab
   (the owner's review of 2026-10-08, item 1), nothing here explains anything with the later training methods: the reward model's use is
   stated plainly (it scores new answers, so the chat model can be trained further toward higher scores). What the guided parts show
   stays with them: σ at a given gap, the training run, “only differences matter”, which cue the small model learned, best-of-n and
   length-balanced pairs; no number of theirs is here (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=A.shell.labLink(id); return tx(`the ${a} lab`,`“${a}”`);};
// a paragraph per item; a formula block (a <div>) stands on its own
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return /^<div/.test(h)?h:`<p>${h}</p>`;}).join('')});
// formulas on their own line, side by side, wrapping under each other in a narrow column, so nothing scrolls sideways
const PRFX=(...xs)=>`<div class="rm-prfx">${xs.map(x=>`<span>${prM(x)}</span>`).join('')}</div>`;
const prPick=()=>tx('P(\\text{person picks A})','P(\\text{人选 A})');
const prCh=()=>tx('r_{\\text{chosen}}','r_{\\text{选中}}'), prRj=()=>tx('r_{\\text{rejected}}','r_{\\text{落选}}');
const prBT='Bradley–⁠Terry';   // a word joiner after the dash: a narrow column does not break the name (reader, optional 8)
if(A.registerPrimer) A.registerPrimer(ID,{min:6,sections:[
 {h:()=>tx('Demonstrations cannot show which answer is better (preference data)','示范写不出“哪个更好”（偏好数据）'),parts:[prP(
   ()=>tx(`After SFT (supervised fine-tuning, training on answers people wrote as demonstrations), a model can answer prompts; from here on it is called the chat model. SFT can only make it imitate demonstrations, and demonstrations are costly to write and never say which of two answers is better (see ${prLK('sft')}).`,
          `做完 SFT（监督微调，用人写的示范回答训练模型），模型已经能按提问作答，下面就叫它聊天模型。SFT 只能让它模仿示范，而示范写起来又难又贵，也从不说明两个回答哪个更好（见${prLK('sft')}）。`),
   ()=>tx('Comparisons give exactly that. Have the chat model write two answers to the same prompt and ask a person to pick the better one. A comparison only asks which is better: both answers may be good, and no single right answer is needed. Reading the two answers is enough, and the labeller need not be able to write them, so it is fast and cheap. And the answers compared are the model’s own, so the mistakes it actually makes, such as making up facts or stating with confidence what it is unsure of, are among them, and the person’s pick marks those answers as the worse one.',
          '比较给出的正是这一点。让聊天模型对同一个提问写出两个回答，请人选出更好的那个。比较只问哪个更好，两个回答可以都不错，不需要一个标准答案；读完两个回答就能判断，标注的人不必自己写得出来，所以又快又便宜。而且拿来比较的是模型自己写的回答，它实际会犯的错，比如编造事实、把没把握的事说得很肯定，都在里面，人的选择会把这样的回答标成更差的一个。'),
   ()=>tx('One such record holds the prompt, the two answers and which one the person picked; it is called a preference pair. The answer the person picked is the chosen answer, the other one the rejected answer.',
          '这样得到的一条数据包括提问、两个回答，以及人选了哪一个，叫一个偏好对（preference pair）。人选的那个是选中的回答（chosen），另一个是落选的回答（rejected）。'))]},
 {h:()=>tx('Learning people’s preference as a scoring function (the reward model)','把人的偏好学成打分函数（奖励模型）'),parts:[prP(
   ()=>tx('A preference pair only says which of the two answers people compared is better. But to use people’s preference to improve the chat model, whether by picking one of several answers it wrote or by training it further, you need to know how good every new answer it writes is. Nobody has compared those answers, and there are far more of them than people could compare.',
          '偏好对只说明人比较过的那两个回答里哪个更好。可要拿人的偏好去改进聊天模型，不论是从它写的几个回答里挑出一个，还是接着训练它，都得知道它新写出的每个回答有多好。这些回答没有人比较过，数量也远多于人能比较的。'),
   ()=>tx('So people’s judgement is first learned as a model, the reward model: it reads a prompt and an answer and outputs one number, the reward, which is the answer’s score; the more people prefer the answer, the larger the number should be. It is trained on preference pairs. Once trained, it scores whatever new answers the chat model writes, with no person involved. So the whole chain runs: collect preference pairs, train the reward model on them, then use its scores to train the chat model further, so that its answers score higher.',
          '所以先把人的判断学成一个模型，叫奖励模型（reward model）：它读进一个提问和一个回答，输出一个数，叫奖励（reward），也就是这个回答的分数；人越偏爱这个回答，这个数应当越大。它用偏好对来训练。训练好以后，聊天模型新写出的回答都由它打分，不必再请人。整个流程因此是：收集偏好对，用它训练奖励模型，再用奖励模型的分数接着训练聊天模型，让它写出的回答分数更高。'),
   ()=>tx(`To judge whether an answer is good you first have to understand the prompt and the answer, which takes the same command of language the chat model has. The simplest way is to initialise the reward model from the chat model SFT produced: that ability is inherited, and only the output has to change. A language model multiplies the last token’s vector by the LM head to give every token in the vocabulary a logit (see ${prLK('architecture')}); the reward model replaces the LM head with a linear layer that outputs a single number. Under the causal mask only the last token has seen the whole prompt and answer, so the number is computed from its vector. Write the prompt as ${prM('x')} and the answer as ${prM('y')}: the reward model gives one number, ${prM('r(x, y)')}.`,
          `要判断一个回答好不好，先得读懂提问和回答，需要和聊天模型一样的语言能力。最省事的办法是直接拿 SFT 得到的聊天模型来初始化奖励模型，这份能力就继承下来了，要改的只有输出。语言模型在最后一个 token 的向量上乘 LM head，给词表里的每个 token 一个 logit（见${prLK('architecture')}）；奖励模型把 LM head 换成只输出一个数的线性层。因果 mask 下只有最后一个 token 看过整段提问和回答，所以这个数从它的向量算出来。记提问为 ${prM('x')}、回答为 ${prM('y')}，奖励模型给出的就是一个数 ${prM('r(x, y)')}。`))]},
 {h:()=>tx(`From comparisons to scores (the ${prBT} model)`,`从比较到分数（${prBT} 模型）`),parts:[prP(
   ()=>tx('To train a reward model you have no scores, only preference pairs: which of two answers a person picked. To learn scores from comparisons you first have to say how scores relate to choices. The relation can only be a probability, not “the higher score is always picked”: when two answers are about equally good, different people often pick differently, and the larger the gap, the more people pick the same one. Once you know, for any two scores, the chance a person picks each answer, you can ask: which scores best explain the choices people made in the data?',
          '训练奖励模型时手里没有分数，只有偏好对：两个回答里人选了哪一个。要从比较学出分数，先得说清分数和选择的关系。这个关系只能是概率，而不是“分数高的一定被选中”：两个回答差不多好时，不同的人常常选得不一样；差得越多，选同一个的人越多。知道了给定两个分数时人选每个回答的概率，就能问：哪一组分数最能解释数据里人已经做出的选择？'),
   ()=>tx(`The ${prBT} model assumes that each answer has a score ${prM('r')}, and that the chance a person picks A over B depends only on how much higher A’s score is, the gap ${prM('r_A - r_B')}. Chess’s Elo ratings follow the same idea: each player’s chance of winning a game is set by the difference of the two ratings. What maps the gap to a probability is the sigmoid function ${prM('\\sigma')}:`,
          `${prBT} 模型的假设是：每个回答有一个分数 ${prM('r')}，人选 A 而不选 B 的概率只看 A 的分数高出多少，也就是分差 ${prM('r_A - r_B')}。国际象棋的 Elo 等级分是同一个思路：两人对弈各自赢的机会，由两人等级分之差决定。把分差映射成概率的，是 sigmoid 函数 ${prM('\\sigma')}：`),
   ()=>PRFX(`${prPick()} = \\sigma(r_A - r_B)`,'\\sigma(z) = \\frac{1}{1 + e^{-z}}'),
   ()=>tx(`Why ${prM('\\sigma')}? It has the properties this probability should have: it lies between 0 and 1 and rises with the gap, and swapping A and B swaps the two probabilities so that they add up to 1, because ${prM('\\sigma(-z) = 1 - \\sigma(z)')}. Other functions have these properties too, the cumulative distribution function of the normal distribution for one; ${prM('\\sigma')} is used because it reads intuitively and is easy to work with once you take its log. Multiplying the numerator and the denominator of ${prM('\\sigma(r_A - r_B)')} by ${prM('e^{r_A}')} gives`,
          `为什么用 ${prM('\\sigma')}？它满足这个概率该有的性质：取值在 0 和 1 之间，随分差单调增大；把 A 和 B 对调，两个概率跟着互换，加起来是 1，因为 ${prM('\\sigma(-z) = 1 - \\sigma(z)')}。满足这些性质的函数不止 ${prM('\\sigma')} 一个，比如正态分布的累积分布函数也满足；选 ${prM('\\sigma')} 是因为它有一个直观的读法，取了对数也好算。把 ${prM('\\sigma(r_A - r_B)')} 的分子分母同乘 ${prM('e^{r_A}')}，得到`),
   ()=>PRFX(`${prPick()} = \\frac{e^{r_A}}{e^{r_A} + e^{r_B}}`),
   ()=>tx(`Think of ${prM('e^{r}')} as each answer’s “strength”: the chance of picking A is A’s share of the two strengths, a softmax over just two options (see ${prLK('architecture')}).`,
          `把 ${prM('e^{r}')} 看作每个回答的“强度”，人选 A 的概率就是 A 的强度在两者之和里占的比例，也就是只有两个选项的 softmax（见${prLK('architecture')}）。`),
   ()=>tx(`Training looks for parameters ${prM('\\theta')} whose scores best explain the choices in the data. For one preference pair, the probability this model gives to the choice recorded in the data is ${prM(`\\sigma(${prCh()} - ${prRj()})`)}, where ${prM(prCh())} and ${prM(prRj())} are the reward model’s scores for the pair’s chosen and rejected answers. Assuming people’s choices in different pairs are independent, the likelihood of the whole data set is the product of these probabilities. As with the next-token loss of pre-training and SFT (see ${prLK('training-memory')}), maximising the likelihood is minimising its negative log; averaged over the pairs, that is the reward model’s loss:`,
          `训练就是找一组参数 ${prM('\\theta')}，让奖励模型给的分数最能解释数据里的选择。对一个偏好对，按这个模型，人做出数据里那个选择的概率是 ${prM(`\\sigma(${prCh()} - ${prRj()})`)}，${prM(prCh())} 和 ${prM(prRj())} 是奖励模型给这一对里选中的和落选的回答的分数。假设各个偏好对里人的选择相互独立，整批数据的似然就是这些概率的乘积。和预训练、SFT 用的下一个 token 的 loss 一样（见${prLK('training-memory')}），最大化似然就是最小化它的负对数；再按偏好对取平均，就是奖励模型的 loss：`),
   ()=>PRFX(`L(\\theta) = -\\,\\mathbb{E}\\big[\\log \\sigma(${prCh()} - ${prRj()})\\big]`),
   ()=>tx(`Here ${prM('\\mathbb{E}')} averages over all the preference pairs in the data, log is the natural logarithm, and ${prM(prCh())} and ${prM(prRj())} are computed by the same reward model with the same parameters ${prM('\\theta')}. Gradient descent (see ${prLK('backprop')}) uses this loss to adjust all of these parameters together: the Transformer itself and the score layer at the end.`,
          `其中 ${prM('\\mathbb{E}')} 表示对数据里的全部偏好对取平均，log 是自然对数，${prM(prCh())} 和 ${prM(prRj())} 由同一个奖励模型、同一组参数 ${prM('\\theta')} 算出。梯度下降（见${prLK('backprop')}）用这个 loss 一起调整这组参数：Transformer 本身，加上最后的打分层。`))]},
 {h:()=>tx('A reward model only approximates people (the cues it learns, out-of-distribution answers)','奖励模型只是人的近似（学到的线索、分布外的回答）'),parts:[prP(
   ()=>tx(`The reward model ${prM('r(x, y)')} is not people’s preference itself, only an approximation fitted on a finite set of preference pairs. What it learns, and which answers it can be trusted on, both depend on the answers it saw in training.`,
          `奖励模型 ${prM('r(x, y)')} 不是人的偏好本身，只是在有限的偏好对上拟合出来的近似。它学到了什么、在哪些回答上可信，都取决于训练时见过哪些回答。`),
   ()=>tx(`The loss only asks ${prM('r')} to score the chosen answer higher; it does not care how ${prM('r')} manages it. Any feature of an answer that, in these preference pairs, often comes with the chosen answer helps ${prM('r')} bring the loss down, and so it is learned into ${prM('r')}, even if people paid it no attention when they chose. So when ${prM('r')} guesses people’s choices right on pairs it has not seen, that does not mean it values what people value.`,
          `loss 只要求 ${prM('r')} 给选中的回答更高的分，不管 ${prM('r')} 凭什么做到。回答里的任何特征，只要在这批偏好对里常和选中的回答一起出现，就能帮 ${prM('r')} 把 loss 降下去，也就会被学进 ${prM('r')}，即使人做选择时并不在意它。所以 ${prM('r')} 在没见过的偏好对上也能猜中人的选择，不等于它看重的就是人看重的东西。`),
   ()=>tx(`And the only scores in the loss are those of the answers in the preference pairs. Those answers were all written by the chat model after SFT, so they have a certain style and range. Answers very unlike them are called out-of-distribution. The loss never constrained their scores, so what ${prM('r')} gives them depends only on how the fitted function extrapolates. The features people did not care about but that were learned into ${prM('r')} go wrong most easily here: in the training data they always came with good answers, but an out-of-distribution answer can have those features and no good content, and ${prM('r')} can still score it above a truly good answer.`,
          `loss 里出现的也只有偏好对里那些回答的分数。这些回答都是 SFT 之后的聊天模型写的，有一定的风格和范围。和它们很不一样的回答，叫分布外（out of distribution）的回答。loss 从来没有约束过它们的分数，${prM('r')} 给它们多少分，全看拟合出的函数怎样外推。那些人并不在意、却被学进 ${prM('r')} 的特征，在这里最容易出错：在训练数据里，它们总和好回答一起出现；分布外的回答却可以只有这些特征，没有好内容，${prM('r')} 给它的分照样可以比真正好的回答还高。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  mount(el,tok){
    wantObs=false; root=el; pause();
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true; tShow=T; watched.add(runKey());}
    if(mode==='challenge'&&!saved) saved={...BASE};
    if(mode==='guided'&&!p&&step===0&&!relang&&!restored) S={...STEPS[0].enter()};
    /* a reload in the training part before its run was watched: the run starts again, as on entering the part */
    const replay=restored&&mode==='guided'&&STEPS[step].id==='train'&&!watched.has(keyOf({...BASE},T)); if(replay) tShow=0; restored=false;
    el.innerHTML=viewHTML(); app=$('#rm-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#rm-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    fillRail();
    render(); watchStage();
    if(replay) play();
  },
  /* round 3 #6 (the lab state contract): a reload continues where the learner was. The snapshot holds what the learner did: the mode,
     the part, the setup, the prediction, what each part has seen (observed, the data and fixes tried, the sliders moved, the largest n),
     the challenge's picks and submission, whether Scenarios was touched, the drill. Not the playhead or the watched runs. */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},wTried:[...wTried],fixTried:[...fixTried],shiftMoved,dMoved,nMax,hackSeen,
    saved:saved?{...saved}:null,chSt:{...chSt},chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,passed:chPassed()}),
  restore(o){if(!o||o.v!==1) return;
    const half=(v,lo,hi)=>typeof v==='number'&&v>=lo&&v<=hi&&Number.isInteger(v*2);
    const okS=x=>!!x&&typeof x==='object'&&['tied','none'].includes(x.rel)&&[1,3].includes(x.tau)&&[50,1000].includes(x.N)&&FOPT.concat([25,100]).includes(x.bal)
      &&Number.isInteger(x.ni)&&x.ni>=0&&x.ni<NS.length&&[0,1].includes(x.pair)&&half(x.shift,-3,3)&&half(x.d,-4,4);
    const pickS=x=>{const r={...BASE}; for(const k in BASE) r[k]=x[k]; return r;}, arr=a=>Array.isArray(a)?a:[], idx=(v,n,d)=>Number.isInteger(v)&&v>=0&&v<n?v:d;
    if(!okS(o.S)) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=idx(o.step,STEPS.length,0); S=pickS(o.S);
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    wTried=new Set(['tied',...arr(o.wTried).filter(v=>WOPT.some(x=>x.v===v))]); fixTried=new Set([0,...arr(o.fixTried).filter(v=>FOPT.includes(v))]);
    shiftMoved=o.shiftMoved===true; dMoved=o.dMoved===true; nMax=idx(o.nMax,NS.length,4); hackSeen=idx(o.hackSeen,NS.length,0);
    saved=mode==='challenge'&&okS(o.saved)?pickS(o.saved):null;
    const c=o.chSt&&typeof o.chSt==='object'?o.chSt:{}; chSt=CH_BAL.includes(c.bal)&&CH_STEPS.includes(c.steps)?{bal:c.bal,steps:c.steps}:{...CH_START};
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true;
    if(observed[SIX('train')]) watched.add(keyOf({...BASE},T));
    tShow=mode==='guided'?T:Tn(); if(mode!=='guided') watched.add(runKey());
    restored=true;},
  leave(){clearTimeout(annT); stageStop(); pause(); if(stRO){stRO.disconnect(); stRO=null;} root=null; app=null;},
  onLang(){if(root){const d=$('#rm-math',app), open=d&&d.open, was=playing; pause(); relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#rm-math',app).open=true; if(was) play();}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#rm-graph',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#rm-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      const P=panels(), r=R(), t=tNow();
      if(mode==='challenge'&&!chSubmitted){vis.innerHTML=''; read.innerHTML=esc(tx(`Length weight ${sg2(r.W[t][1])}, content weight ${sg2(r.W[t][0])}; the result shows after you submit`,`长度的权重 ${sg2(r.W[t][1])}，内容的权重 ${sg2(r.W[t][0])}；提交后显示结果`)); return;}
      if(mode==='challenge'){const e=chEval(chSt.bal,chSt.steps); vis.innerHTML=`<div class="rm-dk"><i style="width:${(e.q*10).toFixed(1)}%"></i><b style="left:${CH_Q*10}%"></b></div>`;
        read.innerHTML=esc(tx(`Best-of-16: true quality ${q2(e.q)} (goal ${CH_Q}); held-out accuracy ${pP(e.acc)}`,`best-of-16 的真实质量 ${q2(e.q)} 分（目标 ${CH_Q}），留出准确率 ${pP(e.acc)}`)); return;}
      if(P.bon){const b=bonAt(runS(),t), x=b[S.ni]; vis.innerHTML=`<div class="rm-dk"><i style="width:${(x.q*10).toFixed(1)}%"></i></div>`;
        read.innerHTML=esc(tx(`Best-of-${NS[S.ni]}: the pick’s true quality is ${q2(x.q)} out of 10, its score ${sg2(x.s)}`,`best-of-${NS[S.ni]}：挑中的回答的真实质量 ${q2(x.q)} 分（满分 10），分数 ${sg2(x.s)}`)); return;}
      if(!P.scores){vis.innerHTML=''; read.innerHTML=esc(tx('The reward model is not trained yet.','奖励模型还没训练。')); return;}
      vis.innerHTML=`<div class="rm-dk"><i style="width:${(r.acc[t]*100).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(tx(`Step ${t}: held-out accuracy ${pP(r.acc[t])}, length weight ${sg2(r.W[t][1])}`,`第 ${t} 步：留出准确率 ${pP(r.acc[t])}，长度的权重 ${sg2(r.W[t][1])}`));}
  }
});

/* =====================================================================
   teaser data and read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__rm=Object.freeze({
  run:(s,ts)=>{const r=run(s); const pick=ts||[0,1,50,100,200,400]; return deep({W:pick.map(t=>r.W[t]),LOSS:pick.map(t=>r.LOSS[t]),acc:pick.map(t=>r.acc[t]),ceil:r.ceil,N:r.N,bal:r.bal,np:r.P.length,nb:r.P.filter(x=>x[3]).length,ch0:r.P.filter(x=>x[2]===0).length});},
  bon:(s,t)=>deep(bonAt(s,t)), chEval:(b,st)=>deep(chEval(b,st)), chBest:()=>deep(chBest()), bon2:()=>deep(bon2()),
  ex:()=>deep(EX), drill:s=>deep(makeDrill(s)), rand:(seed,n)=>{const r=mulberry(seed), o=[]; for(let i=0;i<n;i++) o.push(r()); return o;},
  bankStats:()=>{const B=bank(); return {n:B.length,pad:B.filter(x=>x.pad).length,q:B.reduce((s,x)=>s+x.q,0)/B.length};},
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transfer:e.transfer.map(q=>({q:q.q,o:q.o,why:q.why})),explain:e.explain.map(q=>({q:q.q,o:q.o,why:q.why}))});},
  consts:Object.freeze({QM,QS,SE,PADQ,PADL,L0,LS,ETA,TMAX,T,NH,M,PPAD,NS:NS.slice(),RELS:{...RELS},BASE:{...BASE},CH_N,CH_Q,CH_BAL:CH_BAL.slice(),CH_STEPS:CH_STEPS.slice(),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:PRESETS.map(p=>({id:p.id,...p.p}))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},tShow,playing,spd,watched:[...watched],wTried:[...wTried],fixTried:[...fixTried],shiftMoved,nMax,hackSeen,chSt:{...chSt},chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
