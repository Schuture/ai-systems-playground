/* scaling-law (world training): How big a model should a compute budget buy?
   Fidelity: Analytical. Every number on the page comes from three formulas:
   - training compute C = 6·N·D FLOPs (2 per parameter per token forward, 4 backward);
   - the fitted loss L(N, D) = E + A/N^α + B/D^β, with Epoch AI's refit of the Chinchilla runs (Besiroglu et al. 2024):
     A 482.00572, B 2085.43420, E 1.81686, α 0.34781, β 0.36585, as printed in epoch-research/analyzing-chinchilla
     data_analysis.ipynb (cell "Parameter estimates and their standard errors");
   - serving costs about 2·N FLOPs per served token (the page writes “served tokens” in words, not a letter: T would clash
     with the trillion suffix, and the brief prefers words for tokens served).
   For a budget C the best size is N_opt = G·(C/6)^a with a = β/(α+β), G = (αA/(βB))^(1/(α+β)), D_opt = C/(6·N_opt).
   For a target loss L*, a model of N parameters needs D = (B/(L* − E − A/N^α))^(1/β) tokens.
   GPU-days on an H100 SXM (989.5 TFLOP/s dense BF16, as labs/roofline.js) at an assumed MFU of 40% (the integrator's decision
   for batch 8: this lab turns FLOPs straight into GPU time, with no separate waiting).
   The fit describes one family of training runs (Chinchilla's data and tokenizer): the page uses it only to compare choices,
   and says where it extrapolates (FITR).
   window.__sl is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='scaling-law';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const FIT={E:1.81686,A:482.00572,B:2085.43420,al:0.34781,be:0.36585};      // Epoch AI's refit (used everywhere)
const FIT0={E:1.69,A:406.4,B:410.7,al:0.34,be:0.28};                       // the paper's own fit (fidelity note only)
const H100=989.5e12, DAY=86400;
/* the range of the runs behind the fit: Epoch's 245 points extracted from the Chinchilla paper's figure
   (epoch-research/analyzing-chinchilla data/svg_extracted_data.csv): N up to 16.18B, C up to 1.296e22 FLOPs, D = C ÷ 6N up to 318B,
   D/N up to 341. Everything past it on this page (budgets to 1e25, sizes to 1T, Llama 3 8B's 1,875 per parameter) is extrapolation. */
const FITR={N:16.18e9,C:1.296e22,D:317.8e9,r:341.1};
/* attention over the context, which 6N leaves out: nanoGPT's estimate_mfu adds 12 × layers × d × context FLOPs per token. For Llama 3
   8B (32 layers, d = 4,096, trained on 8,192-token sequences per its model card; 8,030,261,248 parameters as app.js MODELS) that is
   0.267 of 6N, about a quarter more. */
const ATT8=12*32*4096*8192/(6*8030261248);
const lossOf=(N,D,f)=>{f=f||FIT; return f.E+f.A/Math.pow(N,f.al)+f.B/Math.pow(D,f.be);};
const termsOf=(N,D)=>({e:FIT.E,n:FIT.A/Math.pow(N,FIT.al),d:FIT.B/Math.pow(D,FIT.be)});
function optOf(C,f){f=f||FIT; const a=f.be/(f.al+f.be), G=Math.pow(f.al*f.A/(f.be*f.B),1/(f.al+f.be)), N=G*Math.pow(C/6,a), D=C/6/N; return {N,D,L:lossOf(N,D,f),r:D/N};}
const dNeed=(N,Lt)=>{const r=Lt-FIT.E-FIT.A/Math.pow(N,FIT.al); return r>0?Math.pow(FIT.B/r,1/FIT.be):Infinity;};
const nMin=Lt=>Math.pow(FIT.A/(Lt-FIT.E),1/FIT.al);                       // below this size no amount of data reaches L*
const trainFor=(N,Lt)=>6*N*dNeed(N,Lt);
const totFor=(N,Lt,T)=>trainFor(N,Lt)+2*N*T;
/* the size with the least training + serving compute for a target loss: a grid over log N, then golden-section refinement */
function totBest(Lt,T){
  const lo=Math.log10(nMin(Lt))+1e-4, hi=13.5, f=x=>totFor(Math.pow(10,x),Lt,T);
  let bx=lo, bv=Infinity; for(let i=0;i<=800;i++){const x=lo+(hi-lo)*i/800, v=f(x); if(v<bv){bv=v; bx=x;}}
  let a=Math.max(lo,bx-(hi-lo)/800), b=Math.min(hi,bx+(hi-lo)/800); const g=(Math.sqrt(5)-1)/2;
  for(let k=0;k<80;k++){const c=b-g*(b-a), d=a+g*(b-a); if(f(c)<f(d)) b=d; else a=c;}
  const N=Math.pow(10,(a+b)/2), D=dNeed(N,Lt); return {N,D,train:6*N*D,serve:2*N*T,tot:6*N*D+2*N*T,r:D/N};
}
const gpuDays=(C,u)=>C/(H100*u)/DAY;
const ev=s=>{const C=s.C, N=s.N, D=C/(6*N), Lv=lossOf(N,D), o=optOf(C);
  return {C,N,D,L:Lv,r:D/N,o,gd:gpuDays(C,s.u),serve:2*N*s.T,tot:C+2*N*s.T,t:termsOf(N,D),capOk:!s.cap||D<=s.cap*(1+1e-9)};};

/* part 1's three plans: one budget split three ways. The lab's own example, computed with the fit; not published models. */
const PLANS={big:{N:280e9,D:350e9},mid:{N:70e9,D:1.4e12},small:{N:17.5e9,D:5.6e12}};
const PLANK=['big','mid','small'], PLAN_C=6*70e9*1.4e12;               // 5.88e23 FLOPs each
/* real models (reachable sources only; see the fidelity card): nominal sizes as the model cards name them */
const LLAMA8=8e9, TOK15=15e12;
const REAL={
 smol:{N:135e6,D:600e9,name:'SmolLM 135M',src:'smol'},
 l1s:{N:7e9,D:1e12,name:'Llama 1 7B',src:'l1'},
 l1b:{N:65e9,D:1.4e12,name:'Llama 1 65B',src:'l1'},
 l8:{N:LLAMA8,D:TOK15,name:'Llama 3 8B',src:'l3'},
 l70:{N:70e9,D:TOK15,name:'Llama 3 70B',src:'l3'}};
const L5=lossOf(LLAMA8,TOK15);                                          // part 5's target: Llama 3 8B's fitted loss

/* controls */
const BUD=[1e21,1e22,1e23,1e24,1e25];                                   // the iso-FLOP curves drawn
const LC_MIN=20, LC_MAX=25, LC_STEP=0.25;                               // budget slider: 10^20 … 10^25 in quarter decades
const LN_MIN=8, LN_MAX=12, LN_STEP=0.05;                                // model-size slider: 100M … 1T in 1/20 decades
const LD_MIN=10, LD_MAX=14, LD_STEP=0.05;                               // tokens slider (part 3): 10B … 100T
const TS=[0,1e12,1e13,1e14];                                            // served tokens
const CAPS=[0,1e11,1e12];                                               // usable data: unlimited, 100B, 1T
const VIEWS=['iso','nd','tot'];
const snap=(v,a,s)=>Math.round((v-a)/s);
/* the challenge */
const CH={Ns:[1e9,3e9,8e9,20e9,70e9],Ds:[1e11,3e11,1e12,3e12,1e13],Lt:2.10,T:2e12,start:{n:2,d:0}};
function chEval(ni,di){const N=CH.Ns[ni], D=CH.Ds[di], Lv=lossOf(N,D), train=6*N*D, serve=2*N*CH.T; return {ni,di,N,D,L:Lv,ok:Lv<=CH.Lt,train,serve,tot:train+serve,r:D/N};}
function chAll(){const a=[]; CH.Ns.forEach((_,i)=>CH.Ds.forEach((_,j)=>a.push(chEval(i,j)))); return a;}
const chBest=()=>chAll().filter(e=>e.ok).sort((a,b)=>a.tot-b.tot)[0];
const chTrainBest=()=>chAll().filter(e=>e.ok).sort((a,b)=>a.train-b.train)[0];
/* the result's summary (the owner's review b8, R1): the reader's plan, the cheapest plan and one counterexample, the training-only
   best (the “20 tokens per parameter” answer); the one change that helps most; and why a failing plan fails */
const chSame=(a,b)=>!!a&&!!b&&a.ni===b.ni&&a.di===b.di;
function chOne(me){const c=[];
  CH.Ns.forEach((_,i)=>{if(i!==me.ni) c.push(chEval(i,me.di));}); CH.Ds.forEach((_,j)=>{if(j!==me.di) c.push(chEval(me.ni,j));});
  /* a failing plan: the single change (the size or the tokens alone) that meets the goal with the least compute;
     a passing one: the single change that still meets it and saves the most */
  return c.filter(e=>e.ok&&(!me.ok||e.tot<me.tot*(1-1e-9))).sort((a,b)=>a.tot-b.tot)[0]||null;}
function chGap(me){if(me.ok) return null; const t=termsOf(me.N,me.D);
  return me.N<nMin(CH.Lt)?'nmin':(t.d>=t.n?'data':'model');}
const chCounter=me=>{const tb=chTrainBest(), b=chBest(); return chSame(tb,me)||chSame(tb,b)?null:tb;};

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const SUP='⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup=n=>String(n).split('').map(c=>c==='-'?'⁻':SUP[+c]).join('');
/* FLOPs: 10ⁿ when exact, else two significant digits m.m×10ⁿ */
function fF(x){if(!(x>0)) return '0'; let e=Math.floor(Math.log10(x)+1e-9), m=x/Math.pow(10,e); let ms=fixd(m,1); if(ms==='10.0'){e++; ms='1.0';}
  return Math.abs(m-1)<0.0005?`10${sup(e)}`:`${ms}×10${sup(e)}`;}
const fFv=x=>{if(!(x>0)) return 0; let e=Math.floor(Math.log10(x)+1e-9), m=Math.round(x/Math.pow(10,e)*10)/10; return m*Math.pow(10,e);};   // the value as shown
const trim0=t=>String(t).replace(/(\.\d*?)0+([A-Z]?)$/,'$1$2').replace(/\.([A-Z]?)$/,'$1');   // 70.0B → 70B, 1.40T → 1.4T
const fP=n=>trim0(A.fmt.big?A.fmt.big(n):fixd(n/1e9,2)+'B');            // parameters: 8B, 9.06B, 17.5B
function fTok(d){if(!isFinite(d)) return '∞'; const u=d>=1e12?[1e12,'T']:(d>=1e9?[1e9,'B']:[1e6,'M']); const v=d/u[0], dp=Math.max(0,2-Math.floor(Math.log10(v)+1e-9)); return trim0(fixd(v,dp))+u[1];}
const fTokS=d=>({1e11:'100B',3e11:'300B',1e12:'1T',3e12:'3T',1e13:'10T',1e14:'100T',2e12:'2T',0:'0'}[d]||fTok(d));
const fL=v=>fixd(v,3);
const dL=(a,b)=>fixd(Math.round(a*1000)/1000-Math.round(b*1000)/1000,3);   // a loss difference from the values as shown (R6.6)
function fR(r){if(!isFinite(r)) return '∞'; return r<100?fixd(r,1):fint(Math.round(r));}
function fGD(d){if(d<10) return fixd(d,1); const e=Math.floor(Math.log10(d)), p=Math.pow(10,e-1); return fint(Math.round(d/p)*p);}
const fPct=u=>fixd(u*100,0)+'%';
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
/* a formula on its own line. The owner's review of 2026-10-09 (no Show the math row scrolls sideways): a chain a = b = c is its left
   side, then one piece per “=” (or “≈”) that wraps onto a new line under the previous “=” when the box is too narrow; formulas set side
   by side with “, \quad” wrap between them. {} keeps a piece's leading “=” spaced as a relation. The text stays the same (tests read it). */
const relSplit=t=>{const out=[]; let d=0, s=0; for(let i=0;i<t.length;i++){const c=t[i]; if('{(['.includes(c)) d++; else if('})]'.includes(c)) d--;
  else if(!d&&c===' '&&i>s&&(t.startsWith(' = ',i)||t.startsWith(' \\approx ',i))){out.push(t.slice(s,i)); s=i+1;}} out.push(t.slice(s)); return out;};
const M=t=>{
  if(t.includes(', \\quad ')){const q=t.split(', \\quad '); return `<div class="sl-fx sl-wr sl-wq">${q.map((p,i)=>`<span>${A.ui.math(i<q.length-1?p+',':p)}</span>`).join('')}</div>`;}
  const ps=relSplit(t), lead=/^(=|\\approx)/.test(ps[0]);
  if(ps.length<(lead?2:3)) return `<div class="sl-fx">${A.ui.math(t)}</div>`;
  const lhs=lead?'':ps.shift();
  return `<div class="sl-fx sl-ch">${lhs?`<span class="sl-chl">${A.ui.math(lhs)}</span>`:''}<span class="sl-wr">${ps.map((p,i)=>`<span>${A.ui.math(lhs||i?`{} ${p}`:p)}</span>`).join('')}</span></div>`;};
const LFX='L(N, D) = E + \\frac{A}{N^{α}} + \\frac{B}{D^{β}}';                     // the fitted loss, written the same way everywhere
/* training plus serving, with the served tokens in words (brief, integrator decision 4) */
const TOTX=()=>tx('\\text{total} = 6 N D + 2 N \\times \\text{served tokens}','\\text{总算力} = 6 N D + 2 N \\times \\text{服务的 token 数}');

/* =====================================================================
   module state
   ===================================================================== */
const BASE={C:1e22,N:optOf(1e22).N*0.25,T:0,u:0.4,cap:0};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, view='iso';
let budgetMoved=false, lv3=null, lv3Tried=new Set(), tSeen=new Set([0]), remView='nd', saved=null;
let chN=CH.start.n, chD=CH.start.d, chSubmitted=false, chReveal=false, sbTouched=false, allOpen=false;
let drillSeed=1, drillOpen=false, annT=null, userAct=false, wantObs=false;
let tw=null, raf=0, tweens=0, drawC=null;                             // a short transition of the budget curve (R3.20), not a replay
const shownPrev={};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 param:()=>(A.terms&&A.terms.param?A.terms.param():tx('A parameter is one learned number in the model’s matrices.','参数是模型矩阵里一个学到的数值。'))+tx(' N counts them all.',' N 是它们的总数。'),
 token:()=>(A.terms&&A.terms.token?A.terms.token():'')+tx(' D counts the tokens the model reads in training; each is read once in this lab.',' D 是训练时模型读过的 token 总数，这个 Lab 里每个 token 只读一次。'),
 flop:()=>tx('A FLOP is one floating-point operation, an add or a multiply. Multiplying a parameter by a number and adding it to a running sum counts as 2 FLOPs.','FLOP 是一次浮点运算，一次加法或一次乘法。把一个参数乘上一个数值、再加进累加和，算 2 个 FLOP。'),
 loss:()=>tx('The loss here is the cross-entropy of predicting the next token: the negative log of the probability the model gave the right next token, averaged over many tokens. Lower is better.','这里的 loss 是预测下一个 token 的交叉熵：模型给正确的下一个 token 的概率，取负对数，再对很多 token 求平均。越低越好。'),
 fit:()=>tx('A fit picks the constants of a formula so that it passes as close as possible to measured points, here the final losses of many training runs of different sizes.','拟合就是选出公式里的常数，让公式尽量贴近测出来的点。这里的点是很多次不同规模的训练最后得到的 loss。'),
 mfu:()=>tx(`MFU: the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time; time spent waiting for transfers or idling in the pipeline counts.`,`MFU 是模型真正需要的运算（每个 token 6 × 参数量次）占同一段时间里 GPU 峰值算力的比例。等通信、流水线空转的时间都算在里面。`),
 iso:()=>tx('An iso-FLOP curve keeps the compute fixed: every point on it costs the same 6ND, so a bigger model on it gets fewer tokens.','iso-FLOP 曲线上的算力是固定的：曲线上每个点的 6ND 都一样，所以越大的模型分到的 token 越少。'),
 gpud:()=>tx('A GPU-day is one GPU busy for one day. 1,000 GPU-days are, for example, 100 GPUs for 10 days.','1 个 GPU 天就是一张 GPU 跑一天。1,000 个 GPU 天，比如是 100 张 GPU 跑 10 天。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const planT=k=>{const p=PLANS[k]; return tx(`${fP(p.N)} parameters, ${fTok(p.D)} tokens`,`${fP(p.N)} 个参数，${fTok(p.D)} 个 token`);};
const PRED=PLANK.map(k=>({v:k,t:{en:'',zh:''},f:()=>planT(k)}));
const PRED_RIGHT='mid';
const predT=v=>planT(v);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const bCtl=(id,inline)=>A.ui.range({id,key:'lc',label:tx('Compute budget','算力预算'),min:0,max:(LC_MAX-LC_MIN)/LC_STEP,value:snap(Math.log10(S.C),LC_MIN,LC_STEP),out:fF(S.C)+' FLOP',inline});
const nCtl=(id,inline)=>A.ui.range({id,key:'ln',label:tx('Model size (parameters)','模型大小（参数量）'),min:0,max:(LN_MAX-LN_MIN)/LN_STEP,value:snap(Math.log10(S.N),LN_MIN,LN_STEP),out:fP(S.N),inline});
const dCtl=(id,inline)=>A.ui.range({id,key:'ld',label:tx('Training tokens','训练 token 数'),min:0,max:(LD_MAX-LD_MIN)/LD_STEP,value:snap(Math.log10(S.C/(6*S.N)),LD_MIN,LD_STEP),out:fTok(S.C/(6*S.N)),inline});
const budSeg=(id,inline)=>A.ui.seg({id,key:'bud',label:tx('Compute budget','算力预算'),options:BUD.map(c=>[String(Math.round(Math.log10(c))),fF(c)]),value:String(Math.round(Math.log10(S.C)*100)/100),inline});
const tSeg=(id,inline)=>A.ui.seg({id,key:'T',label:tx('Tokens it will serve (assumed)','它要服务的 token 数（假设值）'),options:TS.map(t=>[String(TS.indexOf(t)),t===0?tx('0 (training only)','0（只算训练）'):fTokS(t)]),value:String(TS.indexOf(S.T)),inline});
const lv3Res=v=>{const r=PLANS[v]; return `loss ${fL(lossOf(r.N,r.D))}`;};
const lv3Now=()=>PLANK.find(k=>near(S,PLANS[k]))||null;
function near(s,r){return Math.abs(s.N/r.N-1)<1e-6&&Math.abs(s.C/(6*r.N*r.D)-1)<1e-6;}
const lv3Ctl=()=>`<div class="sl-lv" id="sl-ic-lv3">`+A.ui.levers({aria:tx('Which plan to put into the formula, pick one','把哪种方案代进公式，选一种'),value:lv3Now(),options:PLANK.map(k=>({v:k,title:tx(`${fP(PLANS[k].N)} parameters`,`${fP(PLANS[k].N)} 个参数`),desc:tx(`trained on ${fTok(PLANS[k].D)} tokens`,`训 ${fTok(PLANS[k].D)} 个 token`),result:esc(lv3Tried.has(k)?lv3Res(k):'')}))})+`</div>`;
const planState=k=>{const r=PLANS[k]; return {C:6*r.N*r.D,N:r.N};};
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'nd',gate:()=>!!pred,skip:true,
  lead:()=>tx('Same compute: a big model on little data, or a small one on a lot?','同样的算力：大模型少数据，还是小模型多数据？'),
  enter:()=>({...BASE,...planState('mid')}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(o.f())}</button>`).join('')}</div>`;
   const li=PLANK.map(k=>`<li>${esc(tx(`${fP(PLANS[k].N)} parameters, trained on ${fTok(PLANS[k].D)} tokens`,`${fP(PLANS[k].N)} 个参数，训 ${fTok(PLANS[k].D)} 个 token`))}</li>`).join('');
   return tx(`<p>Training a model is paid for in compute. The same compute can train a big model on a little data, or a small model on a lot of data.</p><p>The three plans below take the same compute, about ${fF(PLAN_C)} FLOPs each (part 2 shows how it is counted). N is the number of ${DEF('parameters',TIP.param())}, D the number of training ${DEF('tokens',TIP.token())}:</p><ul class="sl-plans">${li}</ul><p>From one plan to the next, the parameters shrink to a quarter and the tokens grow four times.</p><p>Go with your gut; part 3 checks it with this lab’s loss formula. Which plan ends training with the lowest ${DEF('loss',TIP.loss())}?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>训练一个模型，花的是算力。同样多的算力，可以训一个大模型、少用点数据，也可以训一个小模型、多用点数据。</p><p>下面三种方案花的算力一样多，都是约 ${fF(PLAN_C)} 个 FLOP（第 2 部分讲怎么算）。N 是${DEF('参数',TIP.param())}量，D 是训练用的 ${DEF('token',TIP.token())} 数：</p><ul class="sl-plans">${li}</ul><p>从上往下，每换一种方案，参数量变成四分之一，token 数变成 4 倍。</p><p>这一题凭直觉猜，第 3 部分会用这个 Lab 的 loss 公式揭晓。训练结束时，哪种方案的 ${DEF('loss',TIP.loss())} 最低？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'six',title:{en:'Compute = 6 × parameters × tokens',zh:'算力 = 6 × 参数量 × token 数'},view:'nd',expect:()=>budgetMoved,showMe:()=>setBudget(1e24),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Each token uses each parameter for 2 FLOPs forward and 4 backward.','每个 token 用到每个参数：前向 2 个 FLOP，反向 4 个。'),
  enter:()=>({...BASE,C:1e22,N:optOf(1e22).N}),
  body:()=>tx(`<p>In training, every token goes through every parameter three times over. In the forward pass each parameter does one multiply-add with the token’s numbers: 2 ${DEF('FLOPs',TIP.flop())} (the lab ${labLink('roofline')} counts them). The backward pass works out two gradients, one passed back to the layer before and one for the parameter itself, so it costs twice the forward pass: 4 FLOPs.</p>`+M('C \\approx 6 \\times N \\times D')+`<p>N is the number of parameters, D the number of training tokens, and C the total FLOPs. Each diagonal in the chart is one budget: along it, twice the parameters means half the tokens.</p><p>Drag the budget to see how many ${DEF('GPU-days',TIP.gpud())} it takes on H100s. GPU-days = GPUs × days, all GPUs counted together: 8 GPUs for 1 day are 8 GPU-days.</p>`,
              `<p>训练时，每个 token 要把模型的每个参数用三遍。前向传播里，每个参数和这个 token 的数值做一次乘加，是 2 个 ${DEF('FLOP',TIP.flop())}（${labLink('roofline')} 那个 Lab 讲过怎么数）。反向传播要算两份梯度，一份传回前一层，一份留给参数自己，所以是前向的两倍，4 个 FLOP。</p>`+M('C \\approx 6 \\times N \\times D')+`<p>N 是参数量，D 是训练用的 token 数，C 是总的 FLOP 数。图里每条斜线是一笔预算：在同一条线上，参数翻一倍，token 就少一半。</p><p>拖动预算，看它在 H100 上要多少 ${DEF('GPU 天',TIP.gpud())}。GPU 天 = 用的 GPU 数 × 天数，所有卡加在一起算：8 张卡跑 1 天是 8 GPU 天。</p>`)+bCtl('sl-ic-lc',true),
  obs:()=>obsSix()},
 {id:'terms',title:{en:'The loss formula: three terms',zh:'loss 公式：三项相加'},view:'terms',expect:()=>PLANK.every(k=>lv3Tried.has(k)),showMe:()=>{setLv3('big'); setLv3('small'); setLv3('mid');},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Loss = what no model removes + the cost of a small model + the cost of little data.','loss = 谁也去不掉的部分 + 模型小的代价 + 数据少的代价。'),
  enter:()=>{lv3=null; lv3Tried=new Set(); return {...BASE,...planState('big')};},
  body:()=>{const g=10**(-FIT.al), d=10**(-FIT.be);
   return tx(`<p>In the Chinchilla paper, DeepMind trained many models, of different sizes and on different numbers of tokens, and ${DEF('fitted',TIP.fit())} their final losses with one formula. The formula adds up three terms:</p>`+M(LFX)+
               `<ul class="sl-terml"><li><b>What no model removes</b> is the first term, E. Language itself is uncertain: however big the model and however much the data, the loss stays above E.</li>`+
               `<li><b>The cost of a small model</b> is the second term, ${A.ui.math('\\frac{A}{N^{\\alpha}}')}. N, the number of parameters, is in its denominator, so the more parameters, the smaller this term.</li>`+
               `<li><b>The cost of little data</b> is the third term, ${A.ui.math('\\frac{B}{D^{\\beta}}')}. D, the number of training tokens, is in its denominator, so the more tokens, the smaller this term.</li></ul>`+
               `<p>E, A, B, α and β are all fitted constants. The exponents α and β set how fast the two costs fall: the larger the exponent, the faster its term falls as the parameters or the tokens grow. In the fit this Lab uses (see the end of the page), α is about ${fixd(FIT.al,2)} and β about ${fixd(FIT.be,2)}, and both terms fall ever more slowly: every tenfold in parameters leaves about ${Math.round(g*100)}% of the second term, every tenfold in tokens about ${Math.round(d*100)}% of the third, so each drop is smaller than the one before.</p><p>The three plans below use the same compute. Pick one, put it into the formula and see its loss. The sliders in the chart move too, but the result covers only these three plans.</p>`,
             `<p>DeepMind 在 Chinchilla 论文里训了很多大小、token 数各不相同的模型，再用一个公式${DEF('拟合',TIP.fit())}它们最后的 loss。这个公式是三项相加：</p>`+M(LFX)+
               `<ul class="sl-terml"><li><b>谁也去不掉的部分</b>是第一项 E。语言本身有不确定性，模型再大、数据再多，loss 也低不过 E。</li>`+
               `<li><b>模型小的代价</b>是第二项 ${A.ui.math('\\frac{A}{N^{\\alpha}}')}。N 是参数量，在分母上，所以参数越多，这一项越小。</li>`+
               `<li><b>数据少的代价</b>是第三项 ${A.ui.math('\\frac{B}{D^{\\beta}}')}。D 是训练用的 token 数，也在分母上，所以 token 越多，这一项越小。</li></ul>`+
               `<p>E、A、B、α、β 都是拟合出来的常数。α 和 β 是两个指数，管这两项降得多快：指数越大，参数或 token 变多时，对应的那一项降得越快。这个 Lab 用的拟合里（见页末），α 约是 ${fixd(FIT.al,2)}，β 约是 ${fixd(FIT.be,2)}，两项都越降越慢：参数量每变成 10 倍，第二项都只剩约 ${Math.round(g*100)}%；token 数每变成 10 倍，第三项都只剩约 ${Math.round(d*100)}%，所以每次降下来的量越来越少。</p><p>下面三种方案用的算力一样多。选一种代进公式，看看它的 loss 是多少。图里的滑块也能拖，但结果只算这三种方案。</p>`)+lv3Ctl();},
  obs:()=>obsTerms()},
 {id:'iso',title:{en:'Fix the budget, move along the curve',zh:'固定预算，沿着曲线走'},view:'iso',expect:()=>atOpt(),showMe:()=>setN(optOf(S.C).N),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Too big a model gets too little data. Too small a model gains little from the extra tokens.','模型太大，分到的数据太少。模型太小，多读的数据也帮不上多少。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>Now the compute is fixed. Each curve in the chart is one budget: across, the number of parameters; up, the loss after training. Once the size is set, so are the tokens: ${A.ui.math('D = \\frac{C}{6N}')}, with C the budget in FLOPs. Every point on a curve costs the same, so it is called an ${DEF('iso-FLOP curve',TIP.iso())}.</p><p>To the right the model is bigger and gets fewer tokens, so the data term grows. To the left it gets more tokens, but the model is too small. In between is a lowest point: the best size for that budget.</p><p>Drag the model size to the lowest point of the 10²² FLOP curve.</p>`,
              `<p>现在把算力固定。图里每条曲线是一笔预算：横轴是参数量，纵轴是训完的 loss。模型大小定了，token 数也就定了：${A.ui.math('D = \\frac{C}{6N}')}，C 是这笔预算的 FLOP 数。曲线上每个点花的算力都一样，所以叫 ${DEF('iso-FLOP 曲线',TIP.iso())}。</p><p>往右，模型更大，分到的 token 更少，数据那一项变大。往左，token 多了，模型却太小。中间有一个最低点，就是这笔预算下最好的模型大小。</p><p>拖动模型大小，找到 10²² FLOP 这条曲线的最低点。</p>`)+nCtl('sl-ic-ln',true)+budSeg('sl-ic-bud',true),
  obs:()=>obsIso()},
 {id:'serve',title:{en:'Past the optimum, on purpose',zh:'故意训过头'},view:'tot',expect:()=>tSeen.has(1e14),showMe:()=>setT(1e14),label:{en:'Result',zh:'结果'},
  lead:()=>tx('A model is trained once and used many times: a smaller one is cheaper every time.','模型训一次、用很多次：小模型每次推理都更便宜。'),
  enter:()=>{tSeen=new Set([0]); return {...BASE,C:6*LLAMA8*TOK15,N:LLAMA8,T:0};},
  body:()=>tx(`<p>Meta trained Llama 3’s 8B model on over 15T tokens, close to 1,900 per parameter, far past the roughly 20 per parameter that is cheapest when only training counts. The reason is that a trained model then serves users: each token it generates takes about 2N FLOPs (one multiply-add per parameter), so a smaller model is cheaper every time it runs.</p><p>Training plus serving takes:</p>`+M(TOTX())+`<p>Every point on a curve is one model size, trained on just enough tokens to reach Llama 3 8B’s loss (in this lab’s formula, only for comparing); up is the compute it takes. The dashed curve counts training only. Pick how many tokens the model will serve: a solid curve that adds serving appears; see where its lowest point moves.</p>`,
              `<p>Meta 用超过 15T 个 token 训练了 Llama 3 的 8B 模型，每个参数将近 1,900 个，远远超过只算训练时最省的约 20 个。原因是模型训完还要服务用户：每生成一个 token 约要 2N 个 FLOP（每个参数一次乘加），模型越小，每次推理越便宜。</p><p>训练加服务，一共要花的算力是：</p>`+M(TOTX())+`<p>曲线上每一点是一个模型大小，token 数取到刚好做到 Llama 3 8B 的 loss（按这个 Lab 的公式算，只用来比较）；纵轴是要花的算力。虚线只算训练。选一个服务量，图里会多出一条加上服务的实线，看看它的最低点往哪边移。</p>`)+tSeg('sl-ic-T',true),
  obs:()=>obsServe()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>({nd:'nd',iso:'iso',tot:'tot'}[remView]||'nd'),
  lead:()=>tx('Training costs about 6ND. As the budget grows, grow parameters and tokens together. For heavy use, train smaller and longer.','训练算力约是 6ND。预算变大，参数和 token 一起长。要大量服务的模型，训得小一点、久一点。'),
  enter:()=>{remView='nd'; return {...BASE,C:1e23,N:optOf(1e23).N};},
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
const atOpt=()=>Math.abs(Math.log10(S.N/optOf(S.C).N))<=LN_STEP*0.75;
function obsSix(){
  if(!budgetMoved) return '';
  const r=ev(S), fl=fleetOf(r.gd);
  return tx(`<p><b>${fF(S.C)} FLOPs take about ${fGD(r.gd)} H100 ${DEF('GPU-days',TIP.gpud())}: ${fint(fl.g)} H100s for about ${fl.en}.</b> This assumes an ${DEF('MFU',TIP.mfu())} of ${fPct(S.u)}: training gets ${fPct(S.u)} of the H100’s peak on average (see ${J('mfu','Settings')}).</p><p>The three rings in the chart are three training plans of different sizes. Each takes ${fF(PLAN_C)} FLOPs, so they sit on one diagonal.</p>`,
            `<p><b>${fF(S.C)} FLOP 在 H100 上约要 ${fGD(r.gd)} 个 ${DEF('GPU 天',TIP.gpud())}，相当于 ${fint(fl.g)} 张 H100 跑${fl.zh}。</b>这里假设 ${DEF('MFU',TIP.mfu())} 是 ${fPct(S.u)}，也就是训练平均用上了 H100 峰值算力的 ${fPct(S.u)}（见${J('mfu','自由调参')}）。</p><p>图里的三个圆圈是三种训练方案，模型大小各不相同。每种都是 ${fF(PLAN_C)} FLOP，所以它们在同一条斜线上。</p>`);
}
/* a fleet size that makes the wall-clock time readable */
function fleetOf(gd){const g=[8,64,1024,16384].find(n=>gd/n<=60)||16384, d=gd/g;
  return d>=1?{g,en:`${fGD(d)} days`,zh:`约 ${fGD(d)} 天`}:{g,en:`${fGD(d*24)} hours`,zh:`约 ${fGD(d*24)} 小时`};}
function obsTerms(){
  if(!lv3Tried.size) return '';
  const lo=k=>lossOf(PLANS[k].N,PLANS[k].D), tm=k=>termsOf(PLANS[k].N,PLANS[k].D), nm=k=>fP(PLANS[k].N);
  if(!PLANK.every(k=>lv3Tried.has(k))){const done=PLANK.filter(k=>lv3Tried.has(k));
    return tx(`<p><b>${done.map(k=>`${nm(k)}: loss ${fL(lo(k))}`).join('; ')}.</b> Now pick the other ${done.length===1?'two plans':'plan'}.</p>`,`<p><b>${done.map(k=>`${nm(k)} 的 loss 是 ${fL(lo(k))}`).join('；')}。</b>再选剩下的${done.length===1?'两种':'那一种'}方案。</p>`);}
  const b=tm('big'), m=tm('mid'), s=tm('small');
  return tx(`<p><b>The ${nm('mid')} plan has the lowest loss, ${fL(lo('mid'))}. The ${nm('big')} plan ends at ${fL(lo('big'))} and the ${nm('small')} plan at ${fL(lo('small'))}.</b></p>`,`<p><b>${nm('mid')} 的方案 loss 最低，是 ${fL(lo('mid'))}。${nm('big')} 的方案是 ${fL(lo('big'))}，${nm('small')} 的方案是 ${fL(lo('small'))}。</b></p>`)+verdict()+
    tx(`<p>Why: it is the size of the two terms. The ${nm('big')} plan has many parameters and little data, so its data term is about ${fixd(b.d/b.n,1)} times its model term. The ${nm('small')} plan is the other way round. The ${nm('mid')} plan keeps the two terms about the same size, and their sum is the smallest.</p>`,
       `<p>原因在于两项的大小。${nm('big')} 的方案参数多、数据少，数据那一项约是模型那一项的 ${fixd(b.d/b.n,1)} 倍。${nm('small')} 的方案正好反过来。${nm('mid')} 的方案让两项差不多大，加起来最小。</p>`);
}
function obsIso(){
  const r=ev(S), o=r.o, d=dL(r.L,o.L), tiny=+d<0.001, big=S.N>o.N;
  /* one slider step off the lowest point the shown difference is 0.000: say so instead of a zero (R6.10) */
  const off=tiny?(big?tx(`<p><b>This model is a little bigger than the lowest point: its loss is less than 0.001 above it.</b> Drag a little to the left.</p>`,`<p><b>这个模型比最低点稍大一点，loss 只比最低点高不到 0.001。</b>再往左拖一点。</p>`)
      :tx(`<p><b>This model is a little smaller than the lowest point: its loss is less than 0.001 above it.</b> Drag a little to the right.</p>`,`<p><b>这个模型比最低点稍小一点，loss 只比最低点高不到 0.001。</b>再往右拖一点。</p>`))
    :(big?tx(`<p><b>This model is too big for the budget: its loss is ${d} above the lowest point.</b> It gets too few tokens, so the data term grows. Drag left.</p>`,`<p><b>这个模型对这笔预算来说太大，loss 比最低点高 ${d}。</b>它分到的 token 太少，数据那一项变大了。往左拖。</p>`)
      :tx(`<p><b>This model is too small for the budget: its loss is ${d} above the lowest point.</b> It gets plenty of tokens, but its model term stays large, and the extra tokens cannot make up for it. Drag right.</p>`,`<p><b>这个模型对这笔预算来说太小，loss 比最低点高 ${d}。</b>它分到的 token 很多，可模型那一项太大，多读的 token 也补不回来。往右拖。</p>`));
  const head=atOpt()?tx(`<p><b>At ${fF(S.C)} FLOPs the best model has about ${fP(o.N)} parameters and ${fTok(o.D)} tokens: about ${Math.round(o.r)} tokens per parameter.</b></p>`,`<p><b>${fF(S.C)} FLOP 时，最好的模型约 ${fP(o.N)} 个参数、${fTok(o.D)} 个 token：每个参数约 ${Math.round(o.r)} 个 token。</b></p>`)
    :off;
  if(!atOpt()) return head;
  const rs=BUD.map(c=>optOf(c).r), lo=Math.floor(Math.min(...rs)), hi=Math.ceil(Math.max(...rs));
  return head+tx(`<p>Try the other budgets: from 10²¹ to 10²⁵ FLOPs the lowest point always sits between ${lo} and ${hi} tokens per parameter. That is the “20 tokens per parameter” people quote. Both the parameters and the tokens grow about as the square root of the budget.</p>`,
                 `<p>换别的预算试试：从 10²¹ 到 10²⁵ FLOP，最低点都在每个参数 ${lo} 到 ${hi} 个 token 之间。这就是常说的“每个参数 20 个 token”。参数量和 token 数都大约随预算的平方根增长。</p>`);
}
function obsServe(){
  const b=totBest(L5,S.T), b0=totBest(L5,0);
  if(S.T===0) return tx(`<p><b>Counting training only, the cheapest way to reach this loss is a ${fP(b0.N)} model trained on ${fTok(b0.D)} tokens, close to 20 tokens per parameter. Llama 3 8B’s training takes about ${fixd(6*LLAMA8*TOK15/b0.train,1)} times the compute of that model.</b></p>`,
                         `<p><b>只算训练时，做到这个 loss 最省的是一个 ${fP(b0.N)} 的模型，训 ${fTok(b0.D)} 个 token，接近每个参数 20 个 token。Llama 3 8B 的训练算力是这个模型的约 ${fixd(6*LLAMA8*TOK15/b0.train,1)} 倍。</b></p>`);
  const head=tx(`<p><b>Serving ${fTokS(S.T)} tokens, the cheapest model for this loss has about ${fP(b.N)} parameters, trained on ${fTok(b.D)} tokens (${fR(b.r)} per parameter).</b></p>`,`<p><b>要服务 ${fTokS(S.T)} 个 token 时，做到这个 loss 最省的模型约 ${fP(b.N)} 个参数，训 ${fTok(b.D)} 个 token（每个参数 ${fR(b.r)} 个）。</b></p>`);
  const why=tx(`<p>The more tokens it serves, the bigger serving’s share of the total, and the further the cheapest size moves left: a smaller model trained longer. ${S.T===1e14?'At 100T the cheapest size is already close to Llama 3 8B itself.':''}</p><p>Small open models do the same: Hugging Face trained SmolLM 135M on 600B tokens, over 4,000 per parameter.</p>`,
               `<p>要服务的 token 越多，服务花的算力占得越重，最省的模型就越往左移：模型更小、训得更久。${S.T===1e14?'到 100T 时，最省的模型已经很接近 Llama 3 8B 本身。':''}</p><p>小的开源模型也这样做：Hugging Face 的 SmolLM 135M 训了 600B 个 token，每个参数超过 4,000 个。</p>`);
  return head+why;
}
function remBtn(k){const on=remView===k;
  const tip={nd:tx('Shows published models at their parameters and tokens, and the 20-tokens line.','在图里显示几个公开模型的参数和 token，以及每个参数 20 个 token 的线。'),iso:tx('Shows the iso-FLOP curve at a budget of 10²³ FLOPs.','在图里显示预算 10²³ FLOP 的 iso-FLOP 曲线。'),tot:tx('Shows training plus serving compute for a model that will serve 100T tokens.','在图里显示要服务 100T 个 token 时，训练加服务的算力。')}[k];
  return ` <button type="button" class="linkbtn sl-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const d23=gpuDays(1e23,BASE.u), b100=totBest(L5,1e14), b0=totBest(L5,0), rs=BUD.map(c=>optOf(c).r);
  return tx(`<ul class="sl-remlist"><li><b>Training compute is about 6 × parameters × tokens.</b> Divide it by what the GPUs really deliver, and you know how long the training takes.${remBtn('nd')}</li>`+
    `<li><b>As the budget grows, grow parameters and tokens together.</b> Whatever the budget, the best plan sits at about ${Math.floor(Math.min(...rs))} to ${Math.ceil(Math.max(...rs))} tokens per parameter. So ten times the budget means about three times the parameters and three times the tokens.${remBtn('iso')}</li>`+
    `<li><b>A model that will serve many tokens is worth training smaller and longer.</b> Every served token costs compute in proportion to the parameters. The more tokens the model serves, the more a smaller model saves, and the extra training soon pays for itself.${remBtn('tot')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “The bigger the model, the better.” For a fixed budget, too big a model gets too few tokens. The cost of little data then outweighs what the size gains, and the loss ends higher.</p><p>✗ “20 tokens per parameter is a rule.” 20 is only this fit’s best ratio when training alone counts. Count serving too, and the more tokens the model will serve, the higher the cheapest ratio, up to thousands of tokens per parameter.</p><p>✗ “The loss can go down to 0.” E in the formula stays: in this fit, however big the model and however much the data, the loss stays above ${fixd(FIT.E,2)}.</p></div>${A.shell.toScenarios()}`,
    `<ul class="sl-remlist"><li><b>训练算力约等于 6 × 参数量 × token 数。</b>再除以 GPU 每秒真正做到的运算，就能估出要训多久。${remBtn('nd')}</li>`+
    `<li><b>预算变大时，参数量和 token 数要一起长。</b>不管预算多大，最好的方案都在每个参数约 ${Math.floor(Math.min(...rs))} 到 ${Math.ceil(Math.max(...rs))} 个 token。所以预算变成 10 倍，参数量和 token 数各长约 3 倍。${remBtn('iso')}</li>`+
    `<li><b>要服务很多 token 的模型，值得训得小一点、久一点。</b>每服务一个 token，花的算力都和参数量成正比。要服务的 token 越多，小模型省下的越多，多训的那些 token 很快就赚回来了。${remBtn('tot')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “模型越大越好。”算力固定时，太大的模型分到的 token 太少。这时数据少的代价超过了模型大的好处，loss 反而更高。</p><p>✗ “每个参数 20 个 token 是规定。”20 倍只是这个拟合在只算训练时给出的最好比例。把服务的算力也算进来，要服务的 token 越多，最省的比例越高，可以到每个参数上千个 token。</p><p>✗ “loss 可以一直降到 0。”公式里的 E 去不掉：在这个拟合里，模型再大、数据再多，loss 也低不过 ${fixd(FIT.E,2)}。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 epoch:'https://github.com/epoch-research/analyzing-chinchilla/blob/92258837425e1b5f2851d624287f0120583a3d0e/data_analysis.ipynb',
 epochCsv:'https://github.com/epoch-research/analyzing-chinchilla/blob/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv',
 nanoMfu:'https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py',
 nanoNb:'https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/scaling_laws.ipynb',
 l1:'https://github.com/meta-llama/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/MODEL_CARD.md',
 l3:'https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/MODEL_CARD.md',
 l31:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/MODEL_CARD.md',
 dclm:'https://github.com/mlfoundations/dclm/blob/361714bdd60bb9b7f4b2d8354cebbf0dec0c329e/README.md',
 smol:'https://github.com/huggingface/blog/blob/998ee73e0594a21b42df14632263266a2831dafe/smollm.md',
 kaplan:'https://arxiv.org/abs/2001.08361',
 hoff:'https://arxiv.org/abs/2203.15556',
 besi:'https://arxiv.org/abs/2404.10102'
};
const PRESETS=[
 {id:'lab',t:{en:'A lab with 10²² FLOPs',zh:'一个有 10²² FLOP 的实验室'},d:{en:'What size and how many tokens?',zh:'该训多大、训多少 token？'},s:()=>({C:1e22,N:optOf(1e22).N,T:0,cap:0}),v:'iso'},
 {id:'serve',t:{en:'A product that will serve 100T tokens',zh:'一个要服务 100T 个 token 的产品'},d:{en:'Training plus serving, not training alone',zh:'算训练加服务，而不只算训练'},s:()=>({C:1e23,N:optOf(1e23).N,T:1e14,cap:0}),v:'tot'},
 {id:'data',t:{en:'A team with only 100B tokens of data',zh:'只有 100B 个 token 数据的团队'},d:{en:'The budget is 10²² FLOPs, the data runs out first',zh:'预算是 10²² FLOP，数据先不够了'},s:()=>({C:1e22,N:1e22/6e11,T:0,cap:1e11}),v:'iso'},
 {id:'redo',t:{en:'A model too big for 10²¹ FLOPs',zh:'10²¹ FLOP 下，模型太大'},d:{en:'Four times the best size, a quarter of the tokens',zh:'参数是最好大小的四倍，token 只有四分之一'},s:()=>({C:1e21,N:optOf(1e21).N*4,T:0,cap:0}),v:'iso'}
];
const same=(a,b)=>Math.abs(a/b-1)<1e-6||(a===0&&b===0);
const presetNow=()=>PRESETS.findIndex(p=>{const s=p.s(); return same(s.C,S.C)&&same(s.N,S.N)&&s.T===S.T&&s.cap===S.cap;});
function scenHTML(i){
  const p=PRESETS[i], r=ev(S), o=r.o;
  if(p.id==='lab') return tx(`<p><b>10²² FLOPs buy a model of about ${fP(o.N)} parameters trained on ${fTok(o.D)} tokens. On H100s at an MFU of ${fPct(S.u)} that is about ${fGD(r.gd)} GPU-days.</b></p><p>A bigger model would get too few tokens. A smaller one keeps too large a model term, and the extra tokens it reads cannot make up for it. So the iso-FLOP curve is lowest here.</p>`,
                             `<p><b>10²² FLOP 最好训一个约 ${fP(o.N)} 个参数的模型，训 ${fTok(o.D)} 个 token。按 MFU ${fPct(S.u)} 算，在 H100 上约要 ${fGD(r.gd)} 个 GPU 天。</b></p><p>模型再大，分到的 token 就不够。模型再小，模型那一项又太大，多读的 token 也补不回来。所以 iso-FLOP 曲线在这里最低。</p>`);
  if(p.id==='serve'){const b=totBest(r.L,S.T);
    return tx(`<p><b>The best loss 10²³ FLOPs can reach is ${fL(r.L)}. To serve 100T tokens, the cheapest way to that loss is not the ${fP(o.N)} model of 20 tokens per parameter but a ${fP(b.N)} model trained on ${fTok(b.D)} tokens.</b></p><p>The small model’s training takes about ${fixd(b.train/S.C,1)} times the compute of the ${fP(o.N)} model’s, but the ${fP(o.N)} model costs about ${fixd(o.N/b.N,1)} times as much on every served token. Over 100T served tokens, serving is most of the bill.</p>`,
              `<p><b>10²³ FLOP 最好能把 loss 做到 ${fL(r.L)}。要服务 100T 个 token 时，做到这个 loss 最省的不是每个参数 20 个 token 的 ${fP(o.N)} 模型，而是一个 ${fP(b.N)} 的模型，训 ${fTok(b.D)} 个 token。</b></p><p>小模型的训练算力是 ${fP(o.N)} 模型的约 ${fixd(b.train/S.C,1)} 倍，可 ${fP(o.N)} 的模型每服务一个 token，算力是小模型的约 ${fixd(o.N/b.N,1)} 倍。服务 100T 个 token，大头是服务。</p>`);}
  if(p.id==='data') return tx(`<p><b>With only 100B tokens, 10²² FLOPs are best spent on a ${fP(S.N)} model that uses all of them. Its loss is ${fL(r.L)}, only ${dL(r.L,o.L)} above the lowest point with unlimited data.</b></p><p>Smaller models on this curve would need more than 100B tokens (the dotted part). Reading the same data several times is the other way out; the lab ${labLink('data-pipeline')} shows what repeating costs.</p>`,
                              `<p><b>只有 100B 个 token 时，10²² FLOP 最好用来训一个 ${fP(S.N)} 的模型，把这些 token 全用上。它的 loss 是 ${fL(r.L)}，只比数据不限时的最低点高 ${dL(r.L,o.L)}。</b></p><p>这条曲线上更小的模型需要超过 100B 个 token（点线那段）。另一条路是把同一批数据多读几遍，${labLink('data-pipeline')}那个 Lab 讲了重复读的代价。</p>`);
  return tx(`<p><b>At 10²¹ FLOPs the best model has about ${fP(o.N)} parameters. With four times the parameters and a quarter of the tokens, the loss is ${dL(r.L,o.L)} above the lowest point.</b></p><p>The budget changes, the lesson does not: too big a model for its compute is starved of data.</p>`,
            `<p><b>10²¹ FLOP 时，最好的模型约 ${fP(o.N)} 个参数。参数变成四倍、token 只剩四分之一时，loss 比最低点高 ${dL(r.L,o.L)}。</b></p><p>预算变了，道理不变。对这笔算力来说太大的模型，分到的数据会不够。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=(ni,di)=>tx(`${fP(CH.Ns[ni])} parameters on ${fTokS(CH.Ds[di])} tokens`,`${fP(CH.Ns[ni])} 个参数，训 ${fTokS(CH.Ds[di])} 个 token`);
const pctLess=(a,b)=>{const x=(1-b/a)*100; return x<1?tx('less than 1%','不到 1%'):`${Math.round(x)}%`;};
/* why a failing plan fails, in one sentence, from the two terms of part 3 */
function gapTxt(me){const g=chGap(me), n=fP(me.N), d=fTokS(me.D);
  if(g==='nmin') return tx(`Your plan falls short on the cost of a small model. A ${n} model is too small: however much data it reads, its loss cannot reach 2.10.`,`你的方案差在模型小的代价。${n} 的模型太小，数据再多，loss 也到不了 2.10。`);
  if(g==='data') return tx(`Your plan falls short on the cost of little data. Trained on only ${d} tokens, that term is larger than the cost of a small model.`,`你的方案差在数据少的代价。只训 ${d} 个 token，这一项比模型小的代价还大。`);
  return tx(`Your plan falls short on the cost of a small model. A ${n} model needs a great many tokens to make up for its size.`,`你的方案差在模型小的代价。${n} 的模型偏小，要读很多 token 才补得回来。`);}
function gapDir(me){const g=chGap(me);
  return g==='data'?tx('More tokens push that term down, but the training compute grows with them.','多训一些 token 能把这一项压下去，可训练算力也跟着涨。')
    :(g==='nmin'?tx('Only a bigger model can get there.','只有换大一点的模型才行。'):tx('A bigger model needs far fewer tokens.','换个大一点的模型，需要的 token 会少很多。'));}
function oneTxt(me,oc){
  const what=oc.ni!==me.ni?tx(`a ${fP(oc.N)} model instead of ${fP(me.N)}`,`把模型从 ${fP(me.N)} 换成 ${fP(oc.N)}`):tx(`${fTokS(oc.D)} training tokens instead of ${fTokS(me.D)}`,`把训练 token 从 ${fTokS(me.D)} 换成 ${fTokS(oc.D)}`);
  return me.ok?tx(`The best single change is ${what}: about ${pctLess(me.tot,oc.tot)} less compute in all.`,`只改一项的话，${what}最划算，总算力能少约 ${pctLess(me.tot,oc.tot)}。`)
    :tx(`Of the single changes that meet the goal, ${what} takes the least compute.`,`只改一项就能达标的改法里，${what}最省。`);}
/* the summary: the reader's plan, the cheapest plan and the counterexample, each as a bar of training plus serving compute */
function sumHTML(me){
  const best=chBest(), isBest=chSame(me,best), ctr=chCounter(me);
  const rows=[{e:me,role:isBest?'mebest':'me'}].concat(isBest?[]:[{e:best,role:'best'}]).concat(ctr&&!chSame(ctr,me)?[{e:ctr,role:'ctr'}]:[]);
  const n=rows.length, sameN=rows.every(x=>x.e.ni===me.ni), sameD=rows.every(x=>x.e.di===me.di), MX=Math.max(...rows.map(x=>x.e.tot));
  const shared=tx(`What the ${n===3?'three':'two'} plans share: the loss must be 2.10 or lower, then the model serves 2T tokens${sameN?`; each has ${fP(me.N)} parameters`:''}${sameD?`; each trains on ${fTokS(me.D)} tokens`:''}.`,
                  `${n===3?'三':'两'}种方案都要把 loss 做到不高于 2.10，训完都要服务 2T 个 token${sameN?`，模型都是 ${fP(me.N)} 个参数`:''}${sameD?`，都训 ${fTokS(me.D)} 个 token`:''}。`);
  const lab={me:tx('Your plan','你的方案'),mebest:tx('Your plan, the cheapest one','你的方案，就是最省的'),
    best:DEF(tx('Cheapest plan','最省的方案'),tx('Of the plans that meet the goal, the one with the least training plus serving compute.','达标的方案里，训练加服务的算力最少的那一种。')),
    ctr:DEF(tx('Counterexample','反例'),tx('The plan that is cheapest when only training counts: close to 20 tokens per parameter.','只算训练时最省的方案，接近每个参数 20 个 token。'))};
  const row=x=>{const e=x.e, mine=x.role==='me'||x.role==='ctr';
    const chips=(sameN?'':`<span class="sl-chip${mine&&e.ni!==best.ni?' on':''}">${esc(tx(`${fP(e.N)} parameters`,`${fP(e.N)} 个参数`))}</span>`)+
      (sameD?'':`<span class="sl-chip${mine&&e.di!==best.di?' on':''}">${esc(tx(`${fTokS(e.D)} tokens`,`训 ${fTokS(e.D)} 个 token`))}</span>`);
    const tr=e.train/MX*100, sv=e.serve/MX*100;
    const trTip=tx(`Training: 6 × ${fP(e.N)} × ${fTokS(e.D)} = ${fF(e.train)} FLOPs.`,`训练：6 × ${fP(e.N)} × ${fTokS(e.D)} = ${fF(e.train)} FLOP。`), svTip=tx(`Serving: 2 × ${fP(e.N)} × 2T = ${fF(e.serve)} FLOPs.`,`服务：2 × ${fP(e.N)} × 2T = ${fF(e.serve)} FLOP。`);
    const sub=e.ok?tx(`loss ${fL(e.L)}, meets the goal`,`loss ${fL(e.L)}，达标`)+(x.role==='best'&&me.ok?tx(`; about ${pctLess(me.tot,e.tot)} less than yours`,`，比你的方案省约 ${pctLess(me.tot,e.tot)}`):'')
      :`<span class="badc">${tx(`loss ${fL(e.L)}, above 2.10: misses the goal`,`loss ${fL(e.L)}，高于 2.10，没达标`)}</span>`;
    return `<li class="sl-c3r ${x.role}${e.ok?'':' miss'}" data-role="${x.role}"><div class="sl-c3h"><span class="sl-c3t">${lab[x.role]}</span>${chips}</div>`+
      `<div class="sl-c3trk"><i class="sl-sg tr" style="width:${tr.toFixed(2)}%" tabindex="0" data-tipx="${esc(trTip)}"></i><i class="sl-sg sv" style="left:${tr.toFixed(2)}%;width:${sv.toFixed(2)}%" tabindex="0" data-tipx="${esc(svTip)}"></i></div>`+
      `<b class="sl-c3d mono" data-tipx="${esc(tx(`${fF(e.train)} + ${fF(e.serve)} = ${fF(e.tot)} FLOPs.`,`${fF(e.train)} + ${fF(e.serve)} = ${fF(e.tot)} FLOP。`))}" tabindex="0">${fF(e.tot)}</b><div class="sl-c3s">${sub}</div></li>`;};
  let h=`<div class="sl-sum"><div class="sl-scap">${tx('Training plus serving compute (FLOPs): your plan against the cheapest one','训练加服务的算力（FLOP）：你的方案和最省的方案')}</div>`+
    `<p class="small sl-same">${shared}</p><ul class="sl-c3">${rows.map(row).join('')}</ul>`+
    `<div class="sl-c3leg small"><span><i class="sl-sw2 tr"></i>${tx('training','训练')}</span><span><i class="sl-sw2 sv"></i>${tx('serving 2T tokens','服务 2T 个 token')}</span></div>`;
  /* the main gap of a passing plan, and the one change that helps most: one sentence each, computed */
  if(me.ok&&!isBest){const dT=me.train-best.train, dS=me.serve-best.serve;
    const isTb=chSame(me,chTrainBest());   // the “20 tokens per parameter” answer: say so, it is the lesson of the challenge
    h+=`<p class="sl-gap">${isTb?tx(`Your plan is the cheapest one when only training counts, ${Math.round(me.r)} tokens per parameter, close to 20. `,`你的方案是只算训练时最省的，每个参数 ${Math.round(me.r)} 个 token，接近 20。`):''}${dS>=dT?tx(`The cheapest plan takes about ${pctLess(me.tot,best.tot)} less compute in all. Most of the difference is serving: a bigger model costs more on every served token.`,`最省的方案总算力比你的少约 ${pctLess(me.tot,best.tot)}。你多花的算力，大部分在服务上。模型越大，每服务一个 token 越贵。`)
      :tx(`The cheapest plan takes about ${pctLess(me.tot,best.tot)} less compute in all. Most of the difference is training: your plan trains on more than the goal needs.`,`最省的方案总算力比你的少约 ${pctLess(me.tot,best.tot)}。你多花的算力，大部分在训练上，训得比目标需要的多。`)}</p>`;}
  if(!isBest){const oc=chOne(me);
    h+=`<p class="sl-next">${oc?(chSame(oc,best)&&!me.ok?oneTxt(me,oc)+tx(' That is the cheapest plan.','这就是最省的方案。'):oneTxt(me,oc))
      :(me.ok?tx(`No single change saves compute. The cheapest plan changes both at once: ${fP(best.N)} parameters and ${fTokS(best.D)} training tokens.`,`只改一项省不下来。最省的方案同时改了两项：模型换成 ${fP(best.N)}，训练 token 换成 ${fTokS(best.D)}。`)
        :tx(`No single change meets the goal. The cheapest plan changes both at once: ${fP(best.N)} parameters and ${fTokS(best.D)} training tokens.`,`只改一项还达不到目标。最省的方案同时改了两项：模型换成 ${fP(best.N)}，训练 token 换成 ${fTokS(best.D)}。`))}</p>`;}
  if(ctr&&!chSame(ctr,me)){const b=best;
    h+=`<p class="sl-why small">${tx(`The counterexample is the cheapest plan when only training counts: ${fP(ctr.N)} on ${fTokS(ctr.D)} tokens, ${Math.round(ctr.r)} tokens per parameter, close to 20. It meets the goal too, but every served token costs ${fixd(ctr.N/b.N,1)} times what the ${fP(b.N)} model costs. Over 2T served tokens, the ${fP(b.N)} model trained longer takes about ${pctLess(ctr.tot,b.tot)} less compute in all.`,
      `反例是只算训练时最省的方案：${fP(ctr.N)} 的模型训 ${fTokS(ctr.D)} 个 token，每个参数 ${Math.round(ctr.r)} 个，接近 20。它也能达标，可每服务一个 token，它的算力是 ${fP(b.N)} 模型的 ${fixd(ctr.N/b.N,1)} 倍。服务 2T 个 token 下来，训得更久的 ${fP(b.N)} 模型总算力少约 ${pctLess(ctr.tot,b.tot)}。`)}</p>`;}
  return h+`</div>`;}
/* every plan that meets the goal, closed by default; the cheapest marked, the reader's own row in bold */
function allHTML(me){
  const ok=chAll().filter(e=>e.ok).sort((a,b)=>a.tot-b.tot), best=ok[0];
  const tr=e=>`<tr class="${chSame(e,best)?'ok':''}${chSame(e,me)?' me':''}"><td>${fP(e.N)}</td><td>${fTokS(e.D)}</td><td>${fL(e.L)}</td><td>${fF(e.tot)}${chSame(e,best)?' ✓':''}</td></tr>`;
  return `<details class="sl-all"><summary>${tx(`See all settings that pass (${ok.length})`,`看全部可行的设置（${ok.length} 种）`)}</summary><div class="sl-allb">`+
    `<p class="small">${tx(`Of the 25 settings, ${ok.length} reach a loss of 2.10 or lower. They are sorted by training plus serving compute, least first.${me.ok?'':' Your plan misses the goal, so it is not in the list.'}`,`25 种设置里，有 ${ok.length} 种能把 loss 做到不高于 2.10。下面按训练加服务的算力从少到多排。${me.ok?'':'你的方案没达标，不在表里。'}`)}</p>`+
    `<div class="sl-tblw"><table class="sl-tbl"><thead><tr><th>${tx('Size','参数量')}</th><th>${tx('Tokens','训练 token')}</th><th>loss</th><th>${tx('Training + serving (FLOPs)','训练加服务<br>（FLOP）')}</th></tr></thead><tbody>${ok.map(tr).join('')}</tbody></table></div>`+
    `<p class="small muted">${tx('✓ the least compute; bold: your plan.','✓ 是算力最少的，加粗的一行是你的方案。')}</p></div></details>`;}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chN,chD), best=chBest();
  let h;
  if(me.ok&&chSame(me,best)) h=tx(`<p class="okline sl-mt"><b>Goal met, with the least compute.</b> The loss is ${fL(me.L)}, below 2.10; training plus serving takes ${fF(me.tot)} FLOPs.</p>`,`<p class="okline sl-mt"><b>达成目标，而且是最省的方案。</b>loss 是 ${fL(me.L)}，低于 2.10。训练加服务一共 ${fF(me.tot)} FLOP。</p>`);
  else if(me.ok) h=tx(`<p class="okline sl-mt"><b>Goal met, but a cheaper plan exists.</b> The loss is ${fL(me.L)}, below 2.10; training plus serving takes ${fF(me.tot)} FLOPs.</p>`,`<p class="okline sl-mt"><b>达成目标，不过还有更省的方案。</b>loss 是 ${fL(me.L)}，低于 2.10。训练加服务一共 ${fF(me.tot)} FLOP。</p>`);
  else h=tx(`<p class="badline sl-mt"><b>Not yet: the loss is ${fL(me.L)}, above 2.10.</b></p>`,`<p class="badline sl-mt"><b>还没达到：loss 是 ${fL(me.L)}，高于 2.10。</b></p>`)+
    `<p class="small sl-hint">${A.lang()==='zh'?gapTxt(me)+gapDir(me):gapTxt(me)+' '+gapDir(me)}</p>`+
    (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the cheapest plan','直接看最省的方案')}</button>`);
  if(!chReveal) return h;
  h+=sumHTML(me)+allHTML(me);
  h+=tx('<p class="small muted">Note: the loss here is this lab’s fitted formula, a yardstick from one family of training runs. A real team would check the trained model on its own evals, and the served tokens are a guess about the future.</p>',
        '<p class="small muted">注意：这里的 loss 是这个 Lab 的拟合公式算的，只是从一组训练里得到的尺子。真实的团队要在自己的评测上检查训出来的模型，而且要服务多少 token 只是对未来的估计。</p>');
  return h;
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Guess first which of three ways to spend the same compute ends with the lowest loss; then see how training compute is counted, what the loss formula is made of, where the best size for a budget lies, and why today’s models are trained far longer than that. Start here if this is new to you.',
            '一共 6 个部分。先凭直觉猜：同一笔算力分成三种方案，哪种的 loss 最低。再看训练算力怎么算、loss 公式由哪几项组成、一笔预算下最好的模型多大，最后看为什么今天的模型训得比这长得多。第一次学从这里开始。'),
  sandbox:tx('Four real situations: a budget of 10²² FLOPs, a product that will serve 100T tokens, a team with only 100B tokens of data, and a model too big for a smaller budget. Pick one to see how the best size changes.',
             '四个真实情况，分别是一笔 10²² FLOP 的预算、一个要服务 100T 个 token 的产品、只有 100B 个 token 数据的团队，以及更小的预算下一个太大的模型。选一个，看最好的模型大小怎么变。'),
  challenge:tx('The model must reach a loss of 2.10 or lower in this lab’s formula, and then serve 2T tokens. Pick its size and its training tokens so that training plus serving takes the least compute.',
               '模型的 loss 要做到 2.10 以下（用这个 Lab 的公式算），训完还要服务 2T 个 token。选模型大小和训练的 token 数，让训练加服务的总算力最少。')
});
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> every number on this page comes from three formulas; ${J('math','“Show the math”')} works them out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Training compute C ≈ 6ND: 2 FLOPs per parameter per token forward, 4 backward. It leaves out the attention arithmetic that grows with the context length. The longer the context and the smaller the model, the bigger its share: by nanoGPT’s count, Llama 3 8B trained on 8K-token sequences does about ${Math.round(ATT8*100)}% more than 6N per token.</li>`+
 `<li>The loss is ${A.ui.math(LFX)}. E is what no model removes. ${A.ui.math('A / N^{\\alpha}')} is the cost of a small model, with N the number of parameters. ${A.ui.math('B / D^{\\beta}')} is the cost of little data, with D the number of training tokens. The exponents α and β set how fast the two fall. The constants are Epoch AI’s refit of the Chinchilla runs: E ${FIT.E}, A ${FIT.A}, B ${FIT.B}, α ${FIT.al}, β ${FIT.be}. The paper’s own constants (E 1.69, A 406.4, B 410.7, α 0.34, β 0.28) put the best ratio at ${Math.round(optOf(1e21,FIT0).r)} to ${Math.round(optOf(1e25,FIT0).r)} tokens per parameter, which contradicts the paper’s own 20; the refit gives ${Math.round(optOf(1e25).r)} to ${Math.round(optOf(1e21).r)}.</li>`+
 `<li>Serving costs 2N FLOPs per token.</li></ul></div>`+
 `<div><div class="eyebrow">Assumed</div><ul>`+
 `<li>The fit describes one family of runs, on Chinchilla’s data and tokenizer. The page uses it only to compare choices; its loss is not any other model’s reported loss.</li><li>The runs the fit was made from go up to ${fP(FITR.N)} parameters, ${fF(FITR.C)} FLOPs, ${fTok(FITR.D)} training tokens and ${Math.round(FITR.r)} tokens per parameter. This page draws budgets up to 10²⁵ FLOPs and sizes up to 1T parameters, and Llama 3 8B has ${fint(TOK15/LLAMA8)} tokens per parameter: all of that is extrapolation.</li>`+
 `<li>Every token is read once (no repeated data).</li>`+
 `<li>The three plans with the same compute (280B, 70B and 17.5B parameters) are this lab’s own example.</li>`+
 `<li>The MFU is ${fPct(BASE.u)}: on average, training gets ${fPct(BASE.u)} of the H100’s peak. The peak is 989.5 TFLOP/s dense BF16, the same as in the lab ${labLink('roofline')}.</li>`+
 `<li>How many tokens a model will serve is a guess about its future; the page picks 1T to 100T.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Refit constants: ${cite(SRC.epoch,'epoch-research/analyzing-chinchilla <code>data_analysis.ipynb</code>')} (Besiroglu et al., arXiv 2404.10102); the range of the fitted runs from the same repository’s ${cite(SRC.epochCsv,'<code>data/svg_extracted_data.csv</code>')} (245 points). The paper’s constants, as quoted in ${cite(SRC.nanoNb,'karpathy/nanoGPT <code>scaling_laws.ipynb</code>')}.</li>`+
 `<li>6ND: the same notebook writes <code>approx_flops = 6*D*N</code> and counts the backward pass as twice the forward pass; nanoGPT’s ${cite(SRC.nanoMfu,'<code>estimate_mfu</code>')} counts 6N per token plus attention.</li>`+
 `<li>About 20 tokens per parameter: the Chinchilla paper’s own compute-optimal tables, as quoted in the same notebook, sit at 19 to 29; DCLM’s compute-optimal “1x” runs use 20 (412M parameters on 8.2B tokens, ${cite(SRC.dclm,'mlfoundations/dclm <code>README.md</code>')}).</li>`+
 `<li>Llama 1: 7B and 13B on 1T tokens, 33B and 65B on 1.4T (${cite(SRC.l1,'meta-llama/llama <code>MODEL_CARD.md</code>')}). SmolLM: 135M and 360M on 600B tokens (${cite(SRC.smol,'huggingface/blog <code>smollm.md</code>')}).</li>`+
 `<li>The Llama 3 card says “over 15 trillion tokens” and 1.3M H100 GPU hours for the 8B (GPU hours = GPUs × hours, all GPUs counted together; 8 GPUs for 1 hour are 8 GPU hours) (${cite(SRC.l3,'meta-llama/llama3 <code>MODEL_CARD.md</code>')}). The Llama 3.1 card says “~15 trillion tokens” and 1.46M for the 8B (${cite(SRC.l31,'llama-models <code>MODEL_CARD.md</code>')}). The cards count the whole job’s GPU time, more than 6ND at an MFU of ${fPct(BASE.u)} predicts (about ${fixd(6*LLAMA8*TOK15/(H100*BASE.u)/3600/1e6,1)}M hours).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上的数字都由三个公式算出。${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>训练算力 C ≈ 6ND：每个 token 对每个参数，前向 2 个 FLOP，反向 4 个。它没有算 attention 随上下文长度增长的那部分计算。上下文越长、模型越小，这部分占的比例越大。按 nanoGPT 的算法，Llama 3 8B 用 8K 的序列训练时，每个 token 的计算量比 6N 多约 ${Math.round(ATT8*100)}%。</li>`+
 `<li>loss 的公式是 ${A.ui.math(LFX)}。E 是谁也去不掉的部分。${A.ui.math('A / N^{\\alpha}')} 是模型小的代价，N 是参数量。${A.ui.math('B / D^{\\beta}')} 是数据少的代价，D 是训练 token 数。指数 α 和 β 管这两项降得多快。常数用的是 Epoch AI 对 Chinchilla 训练数据的重新拟合：E ${FIT.E}，A ${FIT.A}，B ${FIT.B}，α ${FIT.al}，β ${FIT.be}。论文原来的常数（E 1.69，A 406.4，B 410.7，α 0.34，β 0.28）算出的最好比例是每个参数 ${Math.round(optOf(1e21,FIT0).r)} 到 ${Math.round(optOf(1e25,FIT0).r)} 个 token，和论文自己说的 20 对不上；重新拟合后是 ${Math.round(optOf(1e25).r)} 到 ${Math.round(optOf(1e21).r)}。</li>`+
 `<li>服务每个 token 要 2N 个 FLOP。</li></ul></div>`+
 `<div><div class="eyebrow">假设</div><ul>`+
 `<li>这个拟合描述的是一组训练，用的是 Chinchilla 的数据和 tokenizer。页面只用它比较不同的选择，它算出的 loss 不是别的模型公布的 loss。</li><li>拟合用到的训练里，模型最大 ${fP(FITR.N)} 个参数，算力最多 ${fF(FITR.C)} FLOP，训练 token 最多 ${fTok(FITR.D)} 个，每个参数最多 ${Math.round(FITR.r)} 个 token。这一页的预算画到 10²⁵ FLOP，模型画到 1T 个参数，Llama 3 8B 每个参数有 ${fint(TOK15/LLAMA8)} 个 token。这些都超出了拟合的范围，是外推。</li>`+
 `<li>每个 token 只读一次（不重复使用数据）。</li>`+
 `<li>算力相同的三种方案（280B、70B 和 17.5B 个参数）是本 Lab 自己举的例子。</li>`+
 `<li>MFU 取 ${fPct(BASE.u)}，也就是训练平均用上 H100 峰值算力的 ${fPct(BASE.u)}。峰值是 BF16 稠密 989.5 TFLOP/s，和 ${labLink('roofline')} 那个 Lab 用的一样。</li>`+
 `<li>模型将来要服务多少 token 只是估计，页面取 1T 到 100T。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>重新拟合的常数：${cite(SRC.epoch,'epoch-research/analyzing-chinchilla 的 <code>data_analysis.ipynb</code>')}（Besiroglu 等，arXiv 2404.10102）。拟合用到的训练的范围，来自同一个仓库的 ${cite(SRC.epochCsv,'<code>data/svg_extracted_data.csv</code>')}（245 个点）。论文原来的常数引自 ${cite(SRC.nanoNb,'karpathy/nanoGPT 的 <code>scaling_laws.ipynb</code>')}。</li>`+
 `<li>6ND：同一个 notebook 写的是 <code>approx_flops = 6*D*N</code>，并把反向传播算成前向的两倍；nanoGPT 的 ${cite(SRC.nanoMfu,'<code>estimate_mfu</code>')} 按每个 token 6N 再加 attention 算。</li>`+
 `<li>每个参数约 20 个 token：同一个 notebook 引用的 Chinchilla 论文算力最优表格，落在 19 到 29 之间；DCLM 算力最优的“1x”训练用的是 20（412M 个参数、8.2B 个 token，${cite(SRC.dclm,'mlfoundations/dclm 的 <code>README.md</code>')}）。</li>`+
 `<li>Llama 1：7B 和 13B 训了 1T 个 token，33B 和 65B 训了 1.4T（${cite(SRC.l1,'meta-llama/llama 的 <code>MODEL_CARD.md</code>')}）。SmolLM：135M 和 360M 训了 600B 个 token（${cite(SRC.smol,'huggingface/blog 的 <code>smollm.md</code>')}）。</li>`+
 `<li>Llama 3 的模型卡写的是“over 15 trillion tokens”，8B 用了 1.3M 个 H100 GPU 小时（${cite(SRC.l3,'meta-llama/llama3 的 <code>MODEL_CARD.md</code>')}）。GPU 小时 = 用的 GPU 数 × 小时数，所有卡加在一起算（8 张卡跑 1 小时是 8 GPU 小时）。Llama 3.1 的模型卡写的是“~15 trillion tokens”，8B 用了 1.46M（${cite(SRC.l31,'llama-models 的 <code>MODEL_CARD.md</code>')}）。模型卡算的是整个训练任务的 GPU 时间，比按 6ND、MFU ${fPct(BASE.u)} 算出的约 ${fixd(6*LLAMA8*TOK15/(H100*BASE.u)/3600/1e6,1)}M 小时多。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const PRIMER=()=>tx(`<p>The number of parameters N counts every learned number in the model (the lab ${labLink('transformer')} counts them for Llama). A forward pass costs about 2 FLOPs per parameter per token: one multiply and one add (the lab ${labLink('roofline')}). The loss measures how badly the model predicts the next token; training pushes it down (the lab ${labLink('train-debug')}).</p>`,
  `<p>参数量 N 是模型里所有学到的参数的总数（${labLink('transformer')} 那个 Lab 数过 Llama 的参数）。前向传播里，每个 token 对每个参数约做 2 个 FLOP：一次乘法、一次加法（见 ${labLink('roofline')}）。loss 衡量模型预测下一个 token 有多差，训练就是把它往下压（见 ${labLink('train-debug')}）。</p>`);
const OUTCOME=()=>tx('turn a compute budget into parameters and tokens with 6ND, find the size with the lowest loss for a budget, and explain why a model that will serve many tokens is trained smaller and longer.','用 6ND 把一笔算力换成参数量和 token 数，找出一笔预算下 loss 最低的模型大小，并讲清为什么要服务很多 token 的模型会训得小一点、久一点。');
const EXITS=()=>{
  const o23=optOf(1e23), o25=optOf(1e25), r4=Math.pow(4,FIT.be/(FIT.al+FIT.be)), b5=totBest(L5,0);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does the best model for a budget grow only about as the square root of the budget, with the tokens growing about as fast?','为什么一笔预算下最好的模型只随预算的平方根增长，token 数也差不多一样快？'),
   o:[tx('The model term and the data term shrink at almost the same rate (α and β are close), so as the budget grows, parameters and tokens should grow by about the same factor.','模型那一项和数据那一项缩得差不多快（α 和 β 很接近），所以预算变大时，参数和 token 最好按差不多一样的倍数一起长。'),
      tx('6ND fixes the product of N and D, so N and D must always be equal.','6ND 固定了 N 和 D 的乘积，所以 N 和 D 必须一样大。'),
      tx('Bigger models need fewer tokens, so the tokens stay almost fixed while the model takes all the extra compute.','模型越大需要的 token 越少，所以 token 数几乎不变，多出来的算力都给了模型。')],a:0,
   n:[null,tx('The product is fixed, not each factor: at 10²² FLOPs the best model has about 20 tokens per parameter, so N and D differ by a factor of 20.','固定的是乘积，不是每一个：10²² FLOP 时最好的模型每个参数约 20 个 token，N 和 D 差了 20 倍。'),
      tx('The best tokens grow with the budget: from 10²¹ to 10²⁵ FLOPs they go from about 60B to about 5T.','最好的 token 数随预算一起增长：预算从 10²¹ 到 10²⁵ FLOP，最好的 token 数从约 60B 涨到约 5T。')],
   why:tx('At the lowest point, moving a little compute from parameters to tokens no longer lowers the loss. Because the two terms fall at almost the same rate, that balance keeps N and D growing together, each by about the square root of the budget’s growth.','在最低点，把一点算力从参数挪给 token，loss 不会再降。两项下降得差不多快，所以这个平衡让 N 和 D 一起长，各自大约随预算的平方根增长。')},
  {q:tx('Why do teams train small models far past 20 tokens per parameter?','为什么团队会把小模型训到远超每个参数 20 个 token？'),
   o:[tx('Once it reaches the same loss, the smaller model is cheaper on every token it serves.','做到同样的 loss 以后，小模型每服务一个 token 都更省。'),
      tx('Past 20 tokens per parameter, the loss falls faster.','过了每个参数 20 个 token，loss 降得更快。'),
      tx('For the same loss, the smaller model needs less training compute.','做到同样的 loss，小模型的训练算力更少。')],a:0,
   n:[null,tx('The other way round: past the best point each extra token lowers the loss less and less, so training alone gets more expensive.','正好相反：过了最低点，每多训一个 token，loss 降得越来越少，只算训练会越来越贵。'),
      tx(`The other way round: below the training optimum, a smaller model needs more training compute for the same loss. For Llama 3 8B’s loss, for example, the cheapest model counting training only has ${fP(b5.N)} parameters, far more than 8B.`,`正好相反。比训练最优点小的模型，要做到同样的 loss，训练算力更多。比如要做到 Llama 3 8B 的 loss，只算训练时最省的是 ${fP(b5.N)} 的模型，比 8B 大得多。`)],
   why:tx('Serving costs about 2N FLOPs per token. Training is paid once, but serving is paid on every token, for as long as the model is used. So moving from the training optimum to a smaller model costs some extra training, and then saves 2 × ΔN FLOPs on every served token (ΔN: the parameters it no longer has).','服务时，每个 token 约要 2N 个 FLOP。训练只付一次，服务却每个 token 都要付，一直付到模型不用为止。所以从训练最优点换成小一点的模型，训练要多花一些，之后每服务一个 token 都省下 2 × ΔN 个 FLOP（ΔN 是少掉的参数量）。')}],
 transfer:[
  {q:tx(`At 10²³ FLOPs the best model has about ${fP(o23.N)} parameters. The budget grows to 4 times as much. About how much bigger should the best model be?`,`10²³ FLOP 时，最好的模型约 ${fP(o23.N)} 个参数。预算变成 4 倍，最好的模型大约要变成几倍？`),
   o:[tx('About 4 times','约 4 倍'),tx('About 2 times','约 2 倍'),tx('About the same','差不多不变')],a:1,
   why:tx(`The best size grows about as the square root of the budget, and the square root of 4 is 2: in this fit it is ${fixd(r4,2)} times, and the tokens also grow about 2 times. So both the model and the data double.`,`最好的大小大约随预算的平方根增长，4 的平方根是 2：这个拟合里是 ${fixd(r4,2)} 倍，token 数也大约变成 2 倍。所以模型和数据都翻一倍左右。`)},
  /* the owner's review b8 (6): the question says the compute is unlimited, so both N and D are free and only E can be left */
  {q:tx(`Suppose the compute is unlimited: the model can have as many parameters as you like and read as many training tokens as you like. In this lab’s fit, how low can the loss go?`,`假设算力不受限制，模型可以要多大有多大，训练 token 也可以要多少有多少。按这个 Lab 的拟合，loss 最低能降到多少？`),
   o:[tx('0','0'),tx(`About ${fixd(FIT.E,2)}`,`约 ${fixd(FIT.E,2)}`),tx(`About ${fL(o25.L)}, the best loss at 10²⁵ FLOPs`,`约 ${fL(o25.L)}，也就是 10²⁵ FLOP 时的最低 loss`)],a:1,
   why:tx(`With unlimited compute, N and D can both grow without end, so the cost of a small model and the cost of little data both go to 0. Only E is left, ${fixd(FIT.E,2)} in this fit. A fixed budget is different: at 10²⁵ FLOPs, N and D cannot both grow, and the best loss stops at about ${fL(o25.L)}.`,`算力不受限制，N 和 D 就能一直变大，模型小的代价和数据少的代价都趋于 0。最后只剩 E，这个拟合里是 ${fixd(FIT.E,2)}。算力固定就不一样了。比如 10²⁵ FLOP 时，N 和 D 不能同时变大，loss 最低只到约 ${fL(o25.L)}。`)}],
 practice:[
  {id:'llm-fundamentals',d:tx('Question 9: with C ≈ 6ND and D ≈ 20N, derive N and D for 10²³ FLOPs, then say how the choice of N changes if serving cost counts too.','第 9 题：用 C ≈ 6ND 和 D ≈ 20N，算出 10²³ FLOP 下的 N 和 D，再说明如果把服务成本也算进来，N 的选择会怎么变。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Scaling laws','Scaling law'),title:l?L(l.q):tx('How big a model should a compute budget buy?','一笔算力预算，该训多大的模型？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: parameters, FLOPs and loss','前置知识：参数、FLOP 和 loss'),primerPop:PRIMER(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage sl-stage" id="sl-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="sl-st-title"></div><div class="meter-sub" id="sl-st-sub"></div></div><div class="stage-actions"><span class="status" id="sl-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph sl-graph" id="sl-graph">`+
    `<div class="sl-vsw" id="sl-vsw"></div>`+
    `<div class="sl-chart" id="sl-chart"></div>`+
    `<div class="sl-terms" id="sl-terms" hidden></div>`+
    `<div class="sl-leg" id="sl-leg"></div>`+
   `</div>`+
   `<div class="metrics sl-metrics" id="sl-metrics"></div>`+
   `<details class="math" id="sl-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="sl-mathbody"></div></details></div>`;
}
const NOTE={
 u:()=>tx('An assumed value. MFU: the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time (H100: 989.5 TFLOP/s, dense BF16); time spent waiting for transfers or idling in the pipeline counts. It depends on the kernels, the parallelism and the communication between GPUs, so no formula gives it. To compare with measurements: torchtitan trains Llama 3.1 8B on 8 or 128 H100s at 33% to 39% MFU, and Megatron-LM reports 41% to 47% on H100s. MFU changes only the GPU-days, never the best size or the loss.',
          '这是假设值。MFU 是模型真正需要的运算（每个 token 6 × 参数量次）占同一段时间里 GPU 峰值算力的比例。等通信、流水线空转的时间都算在里面。H100 的峰值是 BF16 稠密 989.5 TFLOP/s。MFU 取决于 kernel、并行方式和 GPU 之间的通信，没法用公式算出来。可以对照实测的数。torchtitan 在 8 张或 128 张 H100 上训 Llama 3.1 8B，MFU 是 33% 到 39%。Megatron-LM 在 H100 上报告的是 41% 到 47%。MFU 只影响 GPU 天，不影响最好的大小和 loss。'),
 T:()=>tx('A guess about the model’s future: how many tokens it will generate for users over its whole life. Each costs about 2N FLOPs.','这是对模型将来的估计：它一辈子要为用户生成多少 token。每个 token 约要 2N 个 FLOP。'),
 cap:()=>tx('The usable training data, read once. With a limit, sizes that would need more tokens than this are drawn dotted.','能用来训练的数据，每个 token 只读一次。有上限时，需要的 token 超过上限的那些大小画成点线。')
};
function controlsHTML(){
  return `<aside class="panel controls" id="sl-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="sl-lock" hidden></p>`+
   `<div class="lockable" id="sl-ctl-lc-wrap">${bCtl('sl-ctl-lc',false)}</div>`+
   `<div class="lockable" id="sl-ctl-ln-wrap">${nCtl('sl-ctl-ln',false)}</div>`+
   `<div class="lockable" id="sl-ctl-T-wrap">${A.ui.seg({id:'sl-ctl-T',key:'T',label:tx('Tokens it will serve (assumed)','它要服务的 token 数（假设值）'),options:TS.map(t=>[String(TS.indexOf(t)),t===0?'0':fTokS(t)]),value:String(TS.indexOf(S.T)),note:NOTE.T()})}</div>`+
   `<div class="lockable" id="sl-ctl-cap-wrap">${A.ui.seg({id:'sl-ctl-cap',key:'cap',label:tx('Usable training data','可用的训练数据'),options:CAPS.map((c,i)=>[String(i),c===0?tx('unlimited','不限'):fTokS(c)]),value:String(CAPS.indexOf(S.cap)),note:NOTE.cap()})}</div>`+
   `<div class="sl-grp lockable" id="sl-ctl-as"><div class="sl-gh">${tx('Assumed value','假设值')}</div>`+
    A.ui.range({id:'sl-ctl-u',key:'u',label:'MFU',min:20,max:60,step:5,value:Math.round(S.u*100),out:fPct(S.u),note:NOTE.u()})+`</div>`+
   `<div class="lockable" id="sl-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

E, A, B, al, be = 1.81686, 482.00572, 2085.4342, 0.34781, 0.36585

def loss(N, D):           # fitted loss of N params on D tokens
    return E + A / N**al + B / D**be

def best_size(C):         # lowest point of the iso-FLOP curve
    N = np.logspace(7, 13, 20001)
    D = C / (6 * N)       # C = 6ND
    i = np.argmin(loss(N, D))
    return N[i], D[i]

def gpu_days(C, mfu=0.40, peak=989.5e12):    # H100 BF16
    return C / (peak * mfu) / 86400

for C in [1e21, 1e22, 1e23, 1e24, 1e25]:
    N, D = best_size(C)
    print(f"{C:.0e}: {N/1e9:.2f}B params, {D/1e9:.0f}B tokens,"
          f" {D/N:.1f} tok/param, {gpu_days(C):,.0f} GPU-days")

# 1e+22: 9.06B params, 184B tokens, 20.3 tok/param, 292 GPU-days`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="sl-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="sl-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The fitted loss, the lowest point of an iso-FLOP curve found on a fine grid, and the GPU-days of a budget.','拟合的 loss、在细网格上找 iso-FLOP 曲线的最低点，以及一笔预算要多少 GPU 天。')}</p><pre class="code" id="sl-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="sl-prod-card"><h3>${A.t('cProd')}</h3><ul class="sl-prod">`+
   tx(`<li>Open model cards report the two inputs of 6ND. Meta’s Llama 3 card gives “over 15 trillion tokens” and the GPU hours per model: 1.3M H100 hours for the 8B, 6.4M for the 70B (${cite(SRC.l3,'meta-llama/llama3 <code>MODEL_CARD.md</code>')}). The Llama 3.1 card gives 1.46M, 7.0M and 30.84M for the 8B, 70B and 405B (${cite(SRC.l31,'<code>MODEL_CARD.md</code>')}).</li>`+
      `<li>Training code measures how much of the peak it gets. nanoGPT’s ${cite(SRC.nanoMfu,'<code>estimate_mfu</code>')} first counts the FLOPs per token: 6N, plus the attention term that grows with the context length. Times the tokens processed per second, that is the FLOP/s actually done; divided by the GPU’s peak, it is the MFU.</li>`+
      `<li>Teams fit their own scaling laws before a big run: they train small models along a few iso-FLOP curves on their own data and tokenizer, fit L(N, D), and read off the size for the big budget. Constants from another team’s data, like the ones on this page, do not carry over.</li>`,
      `<li>开源模型的模型卡会给出 6ND 的两个输入。Meta 的 Llama 3 模型卡写了“over 15 trillion tokens”，以及每个模型的 GPU 小时：8B 用了 1.3M 个 H100 小时，70B 用了 6.4M（${cite(SRC.l3,'meta-llama/llama3 的 <code>MODEL_CARD.md</code>')}）。Llama 3.1 的模型卡给的是 8B、70B、405B 分别 1.46M、7.0M、30.84M（${cite(SRC.l31,'<code>MODEL_CARD.md</code>')}）。</li>`+
      `<li>训练代码会测自己用上了多少峰值算力。nanoGPT 的 ${cite(SRC.nanoMfu,'<code>estimate_mfu</code>')} 先算每个 token 要多少 FLOP，也就是 6N，再加上 attention 随上下文长度增长的那一项。乘上每秒处理的 token 数，就是每秒实际做了多少 FLOP。最后除以 GPU 的峰值，得到 MFU。</li>`+
      `<li>团队在大规模训练之前，会先拟合自己的 scaling law：用自己的数据和 tokenizer，沿着几条 iso-FLOP 曲线训一批小模型，拟合出 L(N, D)，再读出大预算下该训多大。别人数据上拟合的常数，比如这一页上的，不能直接搬过来用。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.kaplan,'Scaling Laws for Neural Language Models')}<span>${tx('Kaplan et al., 2020 (arXiv 2001.08361): loss falls as a power law in parameters, data and compute.','Kaplan 等，2020（arXiv 2001.08361）：loss 随参数量、数据量和算力按幂律下降。')}</span></li>`+
   `<li>${cite(SRC.hoff,'Training Compute-Optimal Large Language Models')}<span>${tx('Hoffmann et al., 2022 (arXiv 2203.15556): the Chinchilla paper; parameters and tokens should grow together.','Hoffmann 等，2022（arXiv 2203.15556）：Chinchilla 论文，参数量和 token 数应该一起增长。')}</span></li>`+
   `<li>${cite(SRC.besi,'Chinchilla Scaling: A replication attempt')}<span>${tx('Besiroglu et al., 2024 (arXiv 2404.10102): refits the Chinchilla data; the constants this lab uses.','Besiroglu 等，2024（arXiv 2404.10102）：重新拟合 Chinchilla 的数据，这个 Lab 用的就是它的常数。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="sl-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="sl-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="sl-app m2" id="sl-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="sl-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how the best model size changes.','选一个场景，看看最好的模型大小怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="sl-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Reach a loss of 2.10, then serve 2T tokens, with the least compute','loss 做到 2.10，再服务 2T 个 token，总算力最少')}</h3>`+
   `<p>${tx('You will train a model whose loss, in this lab’s formula, must be 2.10 or lower. After training it will serve 2T tokens to users. Pick one model size and one number of training tokens below, so that training plus serving takes the least compute.',
            '你要训练一个模型，loss 要做到 2.10 以下（用这个 Lab 的公式算）。训完以后，它要为用户服务 2T 个 token。从下面选一个模型大小，再选训练用的 token 数，让训练加服务的总算力最少。')}</p>`+
   `<ul class="sl-asm small"><li>${tx('Training costs 6 × parameters × tokens FLOPs.','训练的算力是 6 × 参数量 × token 数。')}</li><li>${tx('Serving costs 2 × parameters FLOPs per token, for 2T tokens.','服务每个 token 要 2 × 参数量个 FLOP，一共 2T 个 token。')}</li></ul>`+
   A.ui.seg({id:'sl-ic-chn',key:'chn',label:tx('Model size (parameters)','模型大小（参数量）'),options:CH.Ns.map((n,i)=>[String(i),fP(n)]),value:String(chN),inline:true})+
   A.ui.seg({id:'sl-ic-chd',key:'chd',label:tx('Training tokens','训练 token 数'),options:CH.Ds.map((d,i)=>[String(i),fTokS(d)]),value:String(chD),inline:true})+
   `<div class="goal mono sl-goal" id="sl-ch-g1"><span>${tx('Loss','loss')}</span><span><b id="sl-ch-l"></b>${tx(' (goal: 2.10 or lower)','（目标：不高于 2.10）')}</span></div>`+
   `<div class="goal mono sl-goal" id="sl-ch-g2"><span>${tx('Training + serving','训练加服务')}</span><span><b id="sl-ch-c"></b>${tx(' FLOPs (less is better)',' FLOP（越少越好）')}</span></div>`+
   submit+`<div id="sl-ch-res" class="sl-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},six:{en:'Move the budget to continue.',zh:'拖动一次预算，才能继续。'},
    terms:{en:'Put all three plans into the formula to continue.',zh:'三种方案都代进公式，才能继续。'},iso:{en:'Find the lowest point of the curve to continue.',zh:'找到曲线的最低点，才能继续。'},serve:{en:'Pick 100T served tokens to continue.',zh:'选服务 100T 个 token，才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#sl-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#sl-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&((s.id==='six'&&budgetMoved)||(s.id==='terms'&&lv3Tried.size>0)||s.id==='iso'||s.id==='serve'||!!observed[step]);
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv3Now())); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lv3Tried.has(v)?lv3Res(v):'';});
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#sl-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chN,chD), l=$('#sl-ch-l',rail), c=$('#sl-ch-c',rail); if(!l) return;
    l.textContent=fL(e.L); c.textContent=fF(e.tot);
    const g1=$('#sl-ch-g1',rail); g1.classList.toggle('met',e.ok); g1.classList.toggle('sl-over',!e.ok);
    $$('.seg-ctl[data-key="chn"]',rail).forEach(g=>A.ui.setPressed(g,String(chN))); $$('.seg-ctl[data-key="chd"]',rail).forEach(g=>A.ui.setPressed(g,String(chD)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#sl-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);
      const dd=$('details.sl-all',res); if(dd){dd.open=allOpen; dd.addEventListener('toggle',()=>{allOpen=dd.open;});}}   // the list of passing plans stays as the reader left it
  }
}

/* =====================================================================
   stage: one SVG chart in three views (log axes), the loss terms bar
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':view);
/* the filled dot is the reader's own pick; parts 1 and 2 (no pick yet, R3.8) and part 6 (the lab sets the state) draw none, except
   part 5's and part 6's Llama 3 8B, which is labelled as such */
const meDot=()=>!(mode==='guided'&&['mission','six','remember'].includes(STEPS[step].id));
const llamaDot=()=>isP('serve')||isP('remember');
const lossHidden=()=>isP('mission')||isP('six');   // parts 1 and 2: no loss, size or ratio before part 3 answers the prediction (R3.8)
function chartBox(){const el=$('#sl-chart',app), w=Math.max(280,Math.round((el&&el.clientWidth)||560)), phone=w<500; return {w,h:phone?250:300,ml:phone?48:56,mr:phone?14:22,mt:14,mb:phone?40:44,phone};}
const lg=Math.log10;
function axes(b,x0,x1,y0,y1,ylog){
  const X=v=>b.ml+(lg(v)-x0)/(x1-x0)*(b.w-b.ml-b.mr), Y=ylog?(v=>b.h-b.mb-(lg(v)-y0)/(y1-y0)*(b.h-b.mt-b.mb)):(v=>b.h-b.mb-(v-y0)/(y1-y0)*(b.h-b.mt-b.mb));
  return {X,Y};
}
const xTickLab=e=>{const n=Math.pow(10,e); return n>=1e12?(n/1e12)+'T':(n>=1e9?(n/1e9)+'B':(n/1e6)+'M');};
function gridSVG(b,{X,Y},xs,ys,ylab,yfmt,xlab){
  let s='';
  xs.forEach(e=>{const x=X(Math.pow(10,e)); s+=`<line class="sl-gl" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${b.mt}" y2="${b.h-b.mb}"/><text class="sl-tk" x="${x.toFixed(1)}" y="${b.h-b.mb+14}" text-anchor="middle">${xTickLab(e)}</text>`;});
  ys.forEach(v=>{const y=Y(v); s+=`<line class="sl-gl" x1="${b.ml}" x2="${b.w-b.mr}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/><text class="sl-tk" x="${b.ml-6}" y="${(y+3.5).toFixed(1)}" text-anchor="end">${yfmt(v)}</text>`;});
  s+=`<text class="sl-al" x="${((b.ml+b.w-b.mr)/2).toFixed(1)}" y="${b.h-6}" text-anchor="middle">${esc(xlab)}</text>`;
  /* H7 (2026-10-09): x 12, not 9, so the rotated title's ascent stays inside the picture */
  s+=`<text class="sl-al" x="12" y="${((b.mt+b.h-b.mb)/2).toFixed(1)}" text-anchor="middle" transform="rotate(-90 12 ${((b.mt+b.h-b.mb)/2).toFixed(1)})">${esc(ylab)}</text>`;
  return s;
}
const path=pts=>pts.length?'M'+pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join('L'):'';
function clipPts(pts,b){return pts.filter(p=>p[1]>=b.mt-0.5&&p[1]<=b.h-b.mb+0.5&&p[0]>=b.ml-0.5&&p[0]<=b.w-b.mr+0.5);}
function lab(x,y,t,cls,anchor){return `<text class="sl-lab ${cls||''}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor||'middle'}">${esc(t)}</text>`;}
/* the curve's current budget: during a transition, an in-between budget (log scale) */
const curC=()=>drawC!=null?drawC:S.C;
/* view 1: parameters across, training tokens up; budgets are diagonals */
function ndSVG(b){
  const x0=8,x1=12,y0=10,y1=14, ax=axes(b,x0,x1,y0,y1,true), {X,Y}=ax, v=viewNow(), ch=v==='ch';
  const g=mode==='guided'?STEPS[step].id:'';
  let s=`<svg class="sl-svg" width="${b.w}" height="${b.h}" viewBox="0 0 ${b.w} ${b.h}" role="img" aria-label="${esc(tx('Parameters against training tokens','参数量和训练 token 数'))}">`+
    gridSVG(b,ax,[8,9,10,11,12],[1e10,1e11,1e12,1e13,1e14],tx('training tokens D','训练 token 数 D'),v=>xTickLab(lg(v)),tx('parameters N','参数量 N'));
  s+=`<clipPath id="sl-clip"><rect x="${b.ml}" y="${b.mt}" width="${b.w-b.ml-b.mr}" height="${b.h-b.mt-b.mb}"/></clipPath><g clip-path="url(#sl-clip)">`;
  const showBud=!ch;
  const dl=BUD.slice(); if(showBud&&!BUD.some(C=>Math.abs(lg(C)-lg(curC()))<0.01)) dl.push(curC());
  if(showBud) dl.forEach(C=>{const on=Math.abs(lg(C)-lg(curC()))<0.01;
    const a=[Math.pow(10,x0),C/6/Math.pow(10,x0)], z=[Math.pow(10,x1),C/6/Math.pow(10,x1)];
    s+=`<path class="sl-iso${on?' on':''}" d="M${X(a[0]).toFixed(1)},${Y(a[1]).toFixed(1)}L${X(z[0]).toFixed(1)},${Y(z[1]).toFixed(1)}"/>`;});
  const show20=!ch&&!['mission','six','terms'].includes(g);
  if(show20) s+=`<path class="sl-r20" d="M${X(5e8).toFixed(1)},${Y(1e10).toFixed(1)}L${X(1e12).toFixed(1)},${Y(2e13).toFixed(1)}"/>`;
  if(ch){const pts=[]; for(let e=8;e<=12.0001;e+=0.02){const N=Math.pow(10,e), D=dNeed(N,CH.Lt); if(isFinite(D)&&D<1e15) pts.push([X(N),Y(D)]);} s+=`<path class="sl-goalc" d="${path(pts)}"/>`;}
  if(S.cap&&!ch) s+=`<path class="sl-cap" d="M${X(1e8).toFixed(1)},${Y(S.cap).toFixed(1)}L${X(1e12).toFixed(1)},${Y(S.cap).toFixed(1)}"/>`;
  s+=`</g>`;
  /* labels of the diagonals at their lower-right end, inside the box */
  /* circles the diagonal labels must not touch (R7.8): part 1's plans or the published models, and the reader's dot */
  const pl=placer(b,[]), plansOn=!ch&&['mission','six','terms'].includes(g);
  if(plansOn) PLANK.forEach(k=>pl.dot(X(PLANS[k].N),Y(PLANS[k].D),5)); else if(!ch) Object.keys(REAL).forEach(k=>pl.dot(X(REAL[k].N),Y(REAL[k].D),5));
  if(meDot()&&!ch) pl.dot(X(S.N),Y(S.C/(6*S.N)),6.5);
  if(showBud) dl.forEach(C=>{let N=Math.min(1e12,C/6/1e10), D=C/6/N; const x=X(N), y=Y(D); const on=Math.abs(lg(C)-lg(curC()))<0.01;
    if(!BUD.includes(C)&&BUD.some(c=>Math.abs(lg(c)-lg(C))<0.45)) return;   // an in-between budget too close to a labelled one stays unlabelled
    /* above the line's end; when a ring, the dot or another label is there, further up the same line, then below its end */
    const sl=(b.h-b.mt-b.mb)/(b.w-b.ml-b.mr), lx=Math.min(x-4,b.w-b.mr-4), ly=Math.min(y-5,b.h-b.mb-5);
    const sp=pl.put(lx,ly,fF(C),[['end',0,0],['end',-30,-Math.round(30*sl)],['end',-60,-Math.round(60*sl)],['end',0,18],['start',8,0]],true);
    s+=lab(lx+sp[1],ly+sp[2],fF(C),'sl-isol'+(on?' on':''),sp[0]);});
  if(ch){const gt=tx('plans above the|dashed line pass','虚线上方的方案达标'), gl=b.phone?gt.split('|'):[gt.split('|').join(' ')], gx=(b.w-b.mr-4).toFixed(1);
    s+=`<text class="sl-lab sl-goall" x="${gx}" y="${b.mt+12}" text-anchor="end">`+gl.map((l,i)=>`<tspan x="${gx}" dy="${i?13:0}">${esc(l)}</tspan>`).join('')+`</text>`;}   // the empty top-right corner: no plan has more than 10T tokens, and the line is far left there (R7.8)
  if(S.cap&&!ch) s+=lab(X(1e8)+4,Y(S.cap)-5,tx(`usable data: ${fTokS(S.cap)} tokens`,`可用数据：${fTokS(S.cap)} 个 token`),'sl-capl','start');
  /* part 1's plans (parts 1 to 3), or published models */
  const lp=[]; if(showBud) dl.forEach(C=>{for(let e=x0;e<=x1+1e-9;e+=0.01){const N=Math.pow(10,e), yy=Y(C/6/N); if(yy>=b.mt&&yy<=b.h-b.mb) lp.push([X(N),yy]);}});
  const pp=placer(b,lp); pl.boxes().forEach(q=>pp.box(q)); if(meDot()&&!ch) pp.dot(X(S.N),Y(S.C/(6*S.N)),6.5); if(plansOn) PLANK.forEach(k=>pp.dot(X(PLANS[k].N),Y(PLANS[k].D),5));
  if(!ch&&['mission','six','terms'].includes(g)) PLANK.forEach(k=>{const p=PLANS[k], x=X(p.N), y=Y(p.D);
    const tip=tx(`One of three plans with the same compute: ${fP(p.N)} parameters on ${fTok(p.D)} tokens, ${fF(6*p.N*p.D)} FLOPs.`,`算力相同的三种方案之一：${fP(p.N)} 个参数、${fTok(p.D)} 个 token，${fF(6*p.N*p.D)} FLOP。`);
    /* the first spot clear of the diagonals, the rings, the dot and the diagonal labels (R7.8); the 280B label tries left first, away from the 10²⁴ label */
    const cands=k==='big'?[['end',-8,12],['end',-8,9],['end',-8,15],['middle',0,-9],['start',8,4]]:[['start',8,4],['start',8,12],['end',-8,4],['middle',0,-9],['middle',0,18]];
    const sp=pp.put(x,y,fP(p.N),cands,true);
    s+=`<g class="sl-real sl-plan" tabindex="0" data-tipx="${esc(tip)}"><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"/>${lab(x+sp[1],y+sp[2],fP(p.N),'sl-reall',sp[0])}</g>`;});
  else if(!ch) Object.keys(REAL).forEach(k=>{const r=REAL[k], x=X(r.N), y=Y(r.D);
    /* Llama 3 8B and 70B share a row (15T tokens): when their labels would touch (phones), the 70B label goes one line higher (R7.8) */
    const tight=X(REAL.l70.N)-X(REAL.l8.N)<(lw(REAL.l8.name)+lw(REAL.l70.name))/2+6;
    const pos={smol:[x+6,y+15,'start'],l1s:[x-8,y+4,'end'],l1b:[x+8,y+4,'start'],l8:[x,y-9,'middle'],l70:[x,tight?y-22:y-9,'middle']}[k];
    const src={smol:tx('Hugging Face blog','Hugging Face 博客'),l1:tx('Llama 1 model card','Llama 1 模型卡'),l3:tx('Llama 3 model card','Llama 3 模型卡')}[r.src];
    const tip=tx(`${r.name}: ${fP(r.N)} parameters, ${r.src==='l3'?'over 15T':fTok(r.D)} training tokens, ${fR(r.D/r.N)} per parameter (${src}).`,`${r.name}：${fP(r.N)} 个参数，${r.src==='l3'?'超过 15T':fTok(r.D)} 个训练 token，每个参数 ${fR(r.D/r.N)} 个（${src}）。`);
    s+=`<g class="sl-real" tabindex="0" data-tipx="${esc(tip)}"><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"/>${lab(pos[0],pos[1],r.name,'sl-reall',pos[2])}</g>`;});
  /* the reader's choice */
  if(meDot()){const N=ch?CH.Ns[chN]:S.N, D=ch?CH.Ds[chD]:S.C/(6*S.N), x=X(N), y=Y(D);
    if(x>=b.ml&&x<=b.w-b.mr&&y>=b.mt&&y<=b.h-b.mb) s+=`<circle class="sl-me${ch?(chEval(chN,chD).ok?' ok':' bad'):''}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6.5"/>`;}
  return s+'</svg>';
}
/* view 2: iso-FLOP curves, model size across, fitted loss up */
function isoSVG(b){
  const x0=8,x1=12,y0=1.85,y1=3.0, ax=axes(b,x0,x1,y0,y1,false), {X,Y}=ax, Cc=curC();
  let s=`<svg class="sl-svg" width="${b.w}" height="${b.h}" viewBox="0 0 ${b.w} ${b.h}" role="img" aria-label="${esc(tx('Iso-FLOP curves: model size against loss','iso-FLOP 曲线：模型大小和 loss'))}">`+
    gridSVG(b,ax,[8,9,10,11,12],[2.0,2.2,2.4,2.6,2.8,3.0],tx('loss (fitted)','loss（拟合）'),v=>fixd(v,1),tx('parameters N','参数量 N'));
  s+=`<clipPath id="sl-clip2"><rect x="${b.ml}" y="${b.mt}" width="${b.w-b.ml-b.mr}" height="${b.h-b.mt-b.mb}"/></clipPath><g clip-path="url(#sl-clip2)">`;
  const curve=C=>{const p=[]; for(let e=x0;e<=x1+1e-9;e+=0.02){const N=Math.pow(10,e); p.push([X(N),Y(lossOf(N,C/6/N))]);} return p;};
  /* the path of the lowest points, from 10²⁰ to 10²⁶ */
  const mp=[]; for(let e=20;e<=26.0001;e+=0.1){const o=optOf(Math.pow(10,e)); mp.push([X(o.N),Y(o.L)]);}
  s+=`<path class="sl-minp" d="${path(mp)}"/>`;
  const list=BUD.slice(); if(!BUD.some(C=>Math.abs(lg(C)-lg(Cc))<0.01)) list.push(Cc);
  list.forEach(C=>{const on=Math.abs(lg(C)-lg(Cc))<0.01;
    if(on&&S.cap){const nb=C/6/S.cap, p=curve(C), lo=p.filter(q=>q[0]<=X(nb)), hi=p.filter(q=>q[0]>=X(nb)); s+=`<path class="sl-cur on sl-dotted" d="${path(lo)}"/><path class="sl-cur on" d="${path(hi)}"/>`;}
    else s+=`<path class="sl-cur${on?' on':''}" d="${path(curve(C))}"/>`;});
  s+=`</g>`;
  const pli=placer(b,[]); {const yy=Y(lossOf(S.N,Cc/6/S.N)); if(meDot()) pli.dot(X(S.N),yy,6.5);}   // the lowest points' labels keep clear of the reader's dot (R7.8)
  list.forEach(C=>{const o=optOf(C), on=Math.abs(lg(C)-lg(Cc))<0.01, x=X(o.N), y=Y(o.L); if(y<b.mt||y>b.h-b.mb) return;
    s+=`<circle class="sl-min${on?' on':''}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" tabindex="0" data-tipx="${esc(tx(`Lowest point at ${fF(C)} FLOPs: ${fP(o.N)} parameters, ${fTok(o.D)} tokens, loss ${fL(o.L)}.`,`${fF(C)} FLOP 的最低点：${fP(o.N)} 个参数，${fTok(o.D)} 个 token，loss ${fL(o.L)}。`))}"/>`;
    if(BUD.includes(C)||on){const sp=pli.put(x,y,fF(C),[['middle',0,16],['middle',0,-9],['start',8,4]],true); s+=lab(x+sp[1],y+sp[2],fF(C),'sl-isol'+(on?' on':''),sp[0]);}});
  {const o=optOf(1e20); s+=lab(X(o.N)-8,Y(o.L)-6,tx('lowest points','最低点'),'sl-minl','start');}
  /* the data limit is named in the legend (“dotted: needs over 100B tokens”), not on the crowded curves (R7.8) */
  const r=ev({...S,C:Cc}), x=X(S.N), y=Y(lossOf(S.N,Cc/6/S.N));
  if(meDot()&&y>=b.mt&&y<=b.h-b.mb&&x>=b.ml&&x<=b.w-b.mr) s+=`<circle class="sl-me" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6.5"/>`;
  void r;
  return s+'</svg>';
}
/* label placement (R7.8): an estimated box for an 11 px label (CJK about 11 px a character); a spot is free when it stays inside the
   plot and clear of the curves' points, the circles and the labels already placed */
const lw=t=>[...String(t)].reduce((w,c)=>w+(/[　-鿿＀-￯]/.test(c)?11:6.4),0);
const boxOf=(x,y,t,an)=>{const w=lw(t), x0=an==='end'?x-w:(an==='middle'?x-w/2:x); return [x0-2,y-10,x0+w+2,y+3];};
function placer(b,pts){const taken=[], dots=[];
  const hit=q=>q[0]<b.ml||q[2]>b.w-b.mr||q[1]<b.mt||q[3]>b.h-b.mb||pts.some(p=>p[0]>=q[0]&&p[0]<=q[2]&&p[1]>=q[1]&&p[1]<=q[3])||taken.some(o=>o[0]<q[2]&&q[0]<o[2]&&o[1]<q[3]&&q[1]<o[3])
    ||dots.some(([cx,cy,r])=>{const nx=Math.max(q[0],Math.min(cx,q[2])), ny=Math.max(q[1],Math.min(cy,q[3])); return (nx-cx)**2+(ny-cy)**2<(r+1)**2;});
  return {dot:(x,y,r)=>{dots.push([x,y,r]);}, take:(x,y,t,an)=>{taken.push(boxOf(x,y,t,an));}, boxes:()=>taken.slice(), box:q=>{taken.push(q);},
    put:(x,y,t,cands,fallback)=>{let c=cands.find(([an,dx,dy])=>!hit(boxOf(x+dx,y+dy,t,an))); if(!c&&fallback) c=cands[0]; if(c) taken.push(boxOf(x+c[1],y+c[2],t,c[0])); return c||null;}};}
/* view 3: same loss, training (+ serving) compute against model size */
function totSVG(b){
  const Lt=ev(S).L, x0=Math.max(8,Math.floor(lg(nMin(Lt))*4)/4), x1=12, T=S.T;
  const best0=totBest(Lt,0), bestT=totBest(Lt,T), lo=Math.floor(lg(best0.train)*2)/2-0.5, hi=lo+3;
  const ax=axes(b,x0,x1,lo,hi,true), {X,Y}=ax;
  let s=`<svg class="sl-svg" width="${b.w}" height="${b.h}" viewBox="0 0 ${b.w} ${b.h}" role="img" aria-label="${esc(tx('Compute to reach the same loss, by model size','做到同样 loss 所需的算力，按模型大小'))}">`;
  const ys=[]; for(let e=Math.ceil(lo);e<=hi;e++) ys.push(Math.pow(10,e));
  const xs=[]; for(let e=Math.ceil(x0);e<=x1;e++) xs.push(e);
  s+=gridSVG(b,ax,xs,ys,tx('compute (FLOPs)','算力（FLOP）'),v=>`10${sup(Math.round(lg(v)))}`,tx('parameters N','参数量 N'));
  s+=`<clipPath id="sl-clip3"><rect x="${b.ml}" y="${b.mt}" width="${b.w-b.ml-b.mr}" height="${b.h-b.mt-b.mb}"/></clipPath><g clip-path="url(#sl-clip3)">`;
  const tr=[], to=[]; for(let e=x0;e<=x1+1e-9;e+=0.01){const N=Math.pow(10,e), c=trainFor(N,Lt); if(!isFinite(c)) continue; tr.push([X(N),Y(c)]); to.push([X(N),Y(c+2*N*T)]);}
  s+=`<path class="sl-trn" d="${path(tr.filter(p=>p[1]>-1e4))}"/>`;
  if(T>0) s+=`<path class="sl-tot" d="${path(to.filter(p=>p[1]>-1e4))}"/>`;
  s+=`</g>`;
  const pl=placer(b,tr.concat(T>0?to:[]));
  const N=S.N, c=trainFor(N,Lt)+2*N*T, x=X(N), y=Y(c), meOn=isFinite(c)&&y>=b.mt&&y<=b.h-b.mb&&x>=b.ml;
  const ring=(N2,c2)=>[X(N2),Y(c2)], r0=ring(best0.N,best0.train), rT=ring(bestT.N,bestT.tot);
  pl.dot(r0[0],r0[1],4.5); if(T>0) pl.dot(rT[0],rT[1],4.5); if(meOn) pl.dot(x,y,6.5);
  const lt=x1-0.02, lc=trainFor(Math.pow(10,lt),Lt);
  {const t=tx('training only','只算训练'), ex=X(Math.pow(10,lt))-2, sp=pl.put(ex,Y(lc),t,[['end',0,-8]],T===0);   // with serving on and the two curves too close at the right end, the dashed curve is named by the text above the chart
    if(sp) s+=lab(ex,Y(lc)-8,t,'sl-trnl','end');}
  if(T>0){const ct=lc+2*Math.pow(10,lt)*T, t=tx('training + serving','训练加服务'); if(Y(ct)>b.mt+8){s+=lab(X(Math.pow(10,lt)),Y(ct)-7,t,'sl-totl','end'); pl.take(X(Math.pow(10,lt)),Y(ct)-7,t,'end');}}
  /* the rings: each labelled at the first free spot around it (below first) */
  const far=[['middle',0,-26,1],['middle',0,34,1],['end',-14,-24,1],['start',14,-24,1],['end',-14,34,1],['start',14,34,1],['middle',0,-40,1],['middle',0,48,1],['end',-14,-38,1],['start',14,-38,1]];
  const lead=(rx,ry,sp)=>{if(!sp[3]) return ''; const x2=rx+sp[1], y2=sp[2]>0?ry+sp[2]-11:ry+sp[2]+4, d=Math.hypot(x2-rx,y2-ry)||1;
    return `<line class="sl-lead" x1="${(rx+(x2-rx)/d*6).toFixed(1)}" y1="${(ry+(y2-ry)/d*6).toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`;};
  const mk=(p,cls,t,cands,tip)=>{const [rx,ry]=p; if(ry<b.mt||ry>b.h-b.mb) return ''; const opt=T>0&&cls==='sl-min', sp=pl.put(rx,ry,t,cands.concat(far),!opt);   // with serving on, the training-only ring may go unlabelled (the curve's own label and the tooltip name it)
    return `<circle class="${cls}" cx="${rx.toFixed(1)}" cy="${ry.toFixed(1)}" r="4.5" tabindex="0" data-tipx="${esc(tip)}"/>`+(sp?lead(rx,ry,sp)+lab(rx+sp[1],ry+sp[2],t,'sl-minl',sp[0]):'');};
  const ringT=T>0?mk(rT,'sl-min on',tx(`cheapest, serving ${fTokS(T)}`,`服务 ${fTokS(T)} 时最省`),[['start',10,18],['end',-10,18],['middle',0,-12],['start',10,-10],['middle',0,24]],tx(`Serving ${fTokS(T)} tokens, the cheapest size for loss ${fL(Lt)} is ${fP(bestT.N)} parameters on ${fTok(bestT.D)} tokens: ${fF(bestT.tot)} FLOPs in all.`,`服务 ${fTokS(T)} 个 token 时，做到 loss ${fL(Lt)} 最省的是 ${fP(bestT.N)} 个参数、${fTok(bestT.D)} 个 token：一共 ${fF(bestT.tot)} FLOP。`)):'';   // placed first: with serving on, it is the point of the chart
  s+=mk(r0,'sl-min',tx('cheapest, training only','只算训练时最省'),[['middle',0,19],['middle',0,-10],['start',9,4],['end',-9,4]],tx(`Counting training only, the cheapest size for loss ${fL(Lt)} is ${fP(best0.N)} parameters on ${fTok(best0.D)} tokens: ${fF(best0.train)} FLOPs.`,`只算训练时，做到 loss ${fL(Lt)} 最省的是 ${fP(best0.N)} 个参数、${fTok(best0.D)} 个 token：${fF(best0.train)} FLOP。`));
  s+=ringT;
  if(meOn){const t=llamaDot()?'Llama 3 8B':tx('your pick','你的选择');
    s+=`<circle class="sl-me" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6.5"/>`;
    /* the first spot around the dot clear of both curves and of the other labels; none fits: the legend names the dot */
    const sp=pl.put(x,y,t,[['start',12,4],['start',12,-10],['start',12,18],['end',-10,17],['end',-10,-10],['middle',0,24],['middle',0,-13],['start',14,-52,1]]);
    if(sp) s+=(sp[3]?`<line class="sl-lead" x1="${(x+3).toFixed(1)}" y1="${(y-7).toFixed(1)}" x2="${(x+sp[1]).toFixed(1)}" y2="${(y+sp[2]+3).toFixed(1)}"/>`:'')+lab(x+sp[1],y+sp[2],t,'sl-mel',sp[0]);}
  return s+'</svg>';
}
/* part 3: one row per term. E is the same for every model, so it has no bar; the two other terms share one scale (0 to 0.4) */
const TW=0.4, twid=v=>Math.min(100,v/TW*100).toFixed(1)+'%';
function termsHTML(){
  const D=S.C/(6*S.N), t=termsOf(S.N,D), tot=t.e+t.n+t.d;
  const row=(cls,name,v,tip,bar)=>`<div class="sl-trow ${cls}" tabindex="0" data-tipx="${esc(tip)}"><span class="sl-tn">${name}</span><span class="sl-ttr">${bar}</span><b class="mono">${fL(v)}</b></div>`;
  const cap=tx(`${fP(S.N)} parameters on ${fTok(D)} tokens: the loss is the sum of three parts`,`${fP(S.N)} 个参数、${fTok(D)} 个 token：loss 是三部分之和`);
  return `<div class="sl-tcap">${esc(cap)}</div><div class="sl-trows">`+
    row('e',tx('What no model removes, ','谁也去不掉的部分 ')+A.ui.math('E'),t.e,tx(`What no model removes. The next token is uncertain in the text itself, so no model and no data remove this part of the loss. In this fit E = ${FIT.E}.`,`谁也去不掉的部分。文本里的下一个 token 本来就不确定，所以任何模型、任何数据都去不掉这部分 loss。这个拟合里 E = ${FIT.E}。`),`<em>${esc(tx('the same for every model and every amount of data','对任何模型、任何数据量都一样'))}</em>`)+
    row('n',tx('Cost of a small model, ','模型小的代价 ')+A.ui.math('A / N^{\\alpha}'),t.n,tx(`The cost of a small model. N, the number of parameters, is in the denominator, so the more parameters, the smaller this term. With A = ${FIT.A}, α = ${FIT.al} and N = ${fP(S.N)}, it is ${fL(t.n)}.`,`模型小的代价。N 是参数量，在分母上，所以参数越多，这一项越小。A = ${FIT.A}、α = ${FIT.al}、N = ${fP(S.N)} 时，这一项是 ${fL(t.n)}。`),`<i style="width:${twid(t.n)}"></i>`)+
    row('d',tx('Cost of little data, ','数据少的代价 ')+A.ui.math('B / D^{\\beta}'),t.d,tx(`The cost of little data. D, the number of training tokens, is in the denominator, so the more tokens, the smaller this term. With B = ${FIT.B}, β = ${FIT.be} and D = ${fTok(D)}, it is ${fL(t.d)}.`,`数据少的代价。D 是训练用的 token 数，在分母上，所以 token 越多，这一项越小。B = ${FIT.B}、β = ${FIT.be}、D = ${fTok(D)} 时，这一项是 ${fL(t.d)}。`),`<i style="width:${twid(t.d)}"></i>`)+
    `</div><div class="sl-tsum mono">${tx('loss','loss')} = ${fL(t.e)} + ${fL(t.n)} + ${fL(t.d)} = <b>${fL(tot)}</b></div>`+
    `<div class="sl-tsl">${nCtl('sl-st-ln',true)}${dCtl('sl-st-ld',true)}</div>`;
}
function legendHTML(v){
  const it=(cls,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sl-sw ${cls}"></i>${t}</span>`;   // t is HTML: a formula in it is typeset (H1)
  if(v==='nd'||v==='ch'){const g=mode==='guided'?STEPS[step].id:'';
    return (v==='ch'?it('goal',tx('dashed: loss exactly 2.10','虚线：loss 正好 2.10'),tx('Along the dashed line the loss is exactly 2.10. Plans on it or above it (more tokens) meet the goal.','虚线上的 loss 正好是 2.10。在它上面或上方（token 更多）的方案达标。'))
      :it('iso',tx('diagonal: one budget, ','斜线：一笔预算，')+A.ui.math('C = 6ND'),tx('Every point on one diagonal costs the same compute. The thick one is the budget you picked.','同一条斜线上的点算力都一样。粗线是你选的预算。')))+
      (v!=='ch'&&!['mission','six','terms'].includes(g)?it('r20',tx('dashed: 20 tokens per parameter','虚线：每个参数 20 个 token'),tx('The dashed line marks 20 training tokens per parameter.','虚线是每个参数 20 个训练 token 的位置。')):'')+
      (v==='ch'?'':(['mission','six','terms'].includes(g)?it('real',tx('ring: a plan with the same compute','圆圈：算力相同的方案'),tx('Each ring is one of three training plans, at its parameters and training tokens. All three sit on one diagonal: the same compute.','每个圆圈是一种训练方案，画在它的参数量和训练 token 数上。三个圆圈在同一条斜线上：算力一样。'))
        :it('real',tx('ring: a published model','圆圈：公开的模型'),tx('Each ring is a published model at its parameters and training tokens, as its model card or blog post gives them.','每个圆圈是一个公开的模型，画在它的参数量和训练 token 数上，数字来自它的模型卡或博客。'))))+
      (meDot()?it('me',tx('dot: your pick','圆点：你的选择'),tx('The filled dot is the size and tokens you picked.','实心圆点是你选的大小和 token 数。')):'');}
  if(v==='iso') return it('cur',tx('thick curve is your budget, thin ones the other budgets','粗曲线是你的预算，细曲线是其他预算'),tx('The thick curve is the budget you picked; the thin ones are the other budgets.','粗曲线是你选的预算，细曲线是其他预算。'))+
      it('minp',tx('dashed line and rings: each curve’s lowest point','虚线和空心圆：各条曲线的最低点'),tx('The dashed line joins the lowest point of every budget: the best size grows with the budget.','虚线连起每笔预算的最低点：预算越大，最好的模型越大。'))+
      (S.cap?it('dot',tx(`dotted: needs over ${fTokS(S.cap)} tokens`,`点线：需要超过 ${fTokS(S.cap)} 个 token`),tx('On the dotted part, the model would need more training tokens than the usable data.','点线那段的模型需要的训练 token 超过了可用的数据。')):'')+
      (meDot()?it('me',tx('dot: your pick','圆点：你的选择'),tx('The filled dot is the size you picked on the thick curve.','实心圆点是你在粗曲线上选的大小。')):'');
  if(v==='tot') return it('trn',tx('dashed: training only','虚线：只算训练'),tx('The dashed curve is the training compute each size needs to reach the same loss.','虚线是每种大小做到同样 loss 要花的训练算力。'))+
      (S.T>0?it('tot',tx('solid: training + serving','实线：训练加服务'),tx('The solid curve adds the serving compute, 2N FLOPs per served token.','实线再加上服务的算力：每服务一个 token 要 2N 个 FLOP。')):'')+
      it('min',tx('ring: the cheapest size','圆圈：最省的大小'),tx('A ring marks the size where a curve is lowest.','圆圈标出曲线最低的那个大小。'))+
      it('me',tx(llamaDot()?'dot: Llama 3 8B':'dot: your pick',llamaDot()?'圆点：Llama 3 8B':'圆点：你的选择'),llamaDot()?tx('The filled dot is Llama 3 8B: 8B parameters.','实心圆点是 Llama 3 8B，8B 个参数。'):tx('The filled dot is the current model size.','实心圆点是现在的模型大小。'));
  return '';
}
function drawChart(){
  const v=viewNow(), host=$('#sl-chart',app), tb=$('#sl-terms',app), lgd=$('#sl-leg',app);
  const isTerms=v==='terms';
  host.hidden=isTerms; tb.hidden=!isTerms;
  if(isTerms){const h=termsHTML(); if(tb._h!==h){const act=document.activeElement, inT=act&&tb.contains(act)&&act.matches('input'); if(!inT){tb.innerHTML=h; tb._h=h; A.ui.initRanges(tb);} else updTermsInPlace();} lgd.innerHTML=''; return;}
  const b=chartBox(); const svg=v==='iso'?isoSVG(b):(v==='tot'?totSVG(b):ndSVG(b));
  if(host._h!==svg){host.innerHTML=svg; host._h=svg;}
  const lh=legendHTML(v); if(lgd._h!==lh){lgd.innerHTML=lh; lgd._h=lh;}
}
function updTermsInPlace(){const tb=$('#sl-terms',app); const D=S.C/(6*S.N), t=termsOf(S.N,D);
  const rows=$$('.sl-trow',tb); [t.e,t.n,t.d].forEach((v,i)=>{const r=rows[i]; if(!r) return; r.querySelector('b').textContent=fL(v); const bar=r.querySelector('i'); if(bar) bar.style.width=twid(v);});
  const sm=$('.sl-tsum',tb); if(sm) sm.innerHTML=`${tx('loss','loss')} = ${fL(t.e)} + ${fL(t.n)} + ${fL(t.d)} = <b>${fL(t.e+t.n+t.d)}</b>`;
  const cp=$('.sl-tcap',tb); if(cp) cp.textContent=tx(`${fP(S.N)} parameters on ${fTok(D)} tokens: the loss is the sum of three parts`,`${fP(S.N)} 个参数、${fTok(D)} 个 token：loss 是三部分之和`);}
function vswHTML(){
  if(mode!=='sandbox') return '';
  return A.ui.seg({id:'sl-vsw-seg',key:'view',label:tx('Chart','图'),options:[['iso',tx('Iso-FLOP curves','iso-FLOP 曲线')],['nd',tx('Parameters and tokens','参数和 token')],['tot',tx('Training + serving','训练加服务')]],value:view});
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chN,chD); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the loss is 2.10 or lower. Now: ${fL(e.L)}, with ${fF(e.tot)} FLOPs of training plus serving.`,`loss 不高于 2.10 时，徽章就是绿色。现在是 ${fL(e.L)}，训练加服务 ${fF(e.tot)} FLOP。`)};}
  if(lossHidden()) return isP('mission')?{cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The losses stay hidden until part 3 puts the three plans into the formula.','loss 先隐藏，等第 3 部分把三种方案代进公式再显示。')}
    :{cls:'neutral',txt:tx('Losses in part 3','loss 在第 3 部分揭晓'),short:tx('Loss: part 3','第 3 部分揭晓'),tip:tx('This part only counts compute. The losses appear in part 3, which puts the three plans into the formula.','这一部分只算算力。loss 在第 3 部分揭晓，那里会把三种方案代进公式。')};
  if(v==='iso'&&S.cap){const nb=S.C/6/S.cap, o=optOf(S.C), best=Math.max(nb,o.N), at=Math.abs(lg(S.N/best))<=LN_STEP*0.75;
    /* with a data limit, the best size uses all of it when the unlimited best would need more (R2.9: the badge matches the card) */
    if(at&&nb>o.N) return {cls:'neutral',txt:tx('All the data used: best size under the limit','数据用完了：有上限时最好的大小'),short:tx('Best under the limit','上限内最好'),
      tip:tx(`With ${fTokS(S.cap)} tokens of usable data, ${fP(nb)} parameters use all of it: the best size under the limit. Smaller models would need more data than there is.`,`可用的数据只有 ${fTokS(S.cap)} 个 token，${fP(nb)} 个参数正好把它全用上，这是数据有上限时最好的大小。更小的模型需要的数据超过上限。`)};
    if(S.N<nb*(1-1e-6)) return {cls:'neutral',txt:tx('Needs more data than there is','需要的数据超过上限'),short:tx('Not enough data','数据不够'),
      tip:tx(`At ${fF(S.C)} FLOPs, ${fP(S.N)} parameters would read ${fTok(S.C/6/S.N)} tokens, more than the ${fTokS(S.cap)} there are.`,`${fF(S.C)} FLOP 时，${fP(S.N)} 个参数要读 ${fTok(S.C/6/S.N)} 个 token，超过了可用的 ${fTokS(S.cap)}。`)};}
  if(v==='iso'){const at=atOpt(), big=S.N>optOf(S.C).N;
    return {cls:'neutral',txt:at?tx('At the lowest point','在最低点'):(big?tx('Too big for the budget','对预算来说太大'):tx('Too small for the budget','对预算来说太小')),short:at?tx('Lowest point','最低点'):(big?tx('Too big','太大'):tx('Too small','太小')),
      tip:tx(`At ${fF(S.C)} FLOPs the lowest point is at ${fP(optOf(S.C).N)} parameters; you picked ${fP(S.N)}.`,`${fF(S.C)} FLOP 的最低点在 ${fP(optOf(S.C).N)} 个参数；你选的是 ${fP(S.N)}。`)};}
  if(v==='tot'){const Lt=ev(S).L, b=totBest(Lt,S.T), rel=S.N/b.N, at=Math.abs(lg(rel))<0.1;
    return {cls:'neutral',txt:at?tx('About the cheapest','接近最省'):(rel>1?tx('Bigger than the cheapest','比最省的大'):tx('Smaller than the cheapest','比最省的小')),short:at?tx('Near cheapest','接近最省'):(rel>1?tx('Bigger','偏大'):tx('Smaller','偏小')),
      tip:tx(`For loss ${fL(Lt)} and ${fTokS(S.T)} served tokens, the cheapest size is about ${fP(b.N)}; the current model has ${fP(S.N)} parameters.`,`loss ${fL(Lt)}、服务 ${fTokS(S.T)} 个 token 时，最省的大小约 ${fP(b.N)}；现在的模型是 ${fP(S.N)} 个参数。`)};}
  const r=ev(S);
  return {cls:'neutral',txt:tx(`${fR(r.r)} tokens per parameter`,`每个参数 ${fR(r.r)} 个 token`),short:tx(`${fR(r.r)} tok/param`,`每参数 ${fR(r.r)}`),tip:tx(`${fTok(r.D)} training tokens ÷ ${fP(S.N)} parameters.`,`${fTok(r.D)} 个训练 token ÷ ${fP(S.N)} 个参数。`)};
}
function drawHead(){
  const v=viewNow(), t=$('#sl-st-title',app), sub=$('#sl-st-sub',app), st=$('#sl-status',app), r=ev(S);
  const title={nd:tx('Parameters and training tokens: each diagonal is one budget','参数量和训练 token：每条斜线是一笔预算'),iso:tx('One budget, many sizes: which ends with the lowest loss?','同一笔预算，换不同大小：谁的 loss 最低？'),
    tot:tx('Same loss: training plus serving compute by model size','同样的 loss：训练加服务的算力，按模型大小'),terms:tx('What the fitted loss is made of','拟合的 loss 由哪几部分组成'),ch:tx('Your plan: parameters and training tokens against the 2.10 line','你的方案：参数量、训练 token 和 2.10 那条线')}[v];
  t.textContent=title;
  sub.textContent=v==='ch'?chName(chN,chD):(v==='tot'?tx(`target loss ${fL(r.L)}, serving ${S.T?fTokS(S.T):'0'} tokens`,`目标 loss ${fL(r.L)}，服务 ${S.T?fTokS(S.T):'0'} 个 token`)
    :(isP('mission')?tx(`three plans, ${fF(PLAN_C)} FLOPs each`,`三种方案，每种 ${fF(PLAN_C)} FLOP`):tx(`budget ${fF(S.C)} FLOPs`,`预算 ${fF(S.C)} FLOP`)));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function metricsDef(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chN,chD);
    return [{k:'L',label:tx('Loss (fitted)','loss（拟合）'),v:fL(e.L),raw:e.L,dp:3,sub:tx('goal: 2.10 or lower','目标：不高于 2.10'),tip:tx(`${fP(e.N)} parameters on ${fTokS(e.D)} tokens: loss ${fL(e.L)} in this lab’s formula.`,`${fP(e.N)} 个参数、${fTokS(e.D)} 个 token：用这个 Lab 的公式算，loss 是 ${fL(e.L)}。`)},
      {k:'tr',label:tx('Training','训练'),v:fF(e.train),raw:fFv(e.train),sub:'6ND FLOP',tip:tx(`6 × ${fP(e.N)} × ${fTokS(e.D)} = ${fF(e.train)} FLOPs.`,`6 × ${fP(e.N)} × ${fTokS(e.D)} = ${fF(e.train)} FLOP。`)},
      {k:'sv',label:tx('Serving 2T tokens','服务 2T 个 token'),v:fF(e.serve),raw:fFv(e.serve),sub:tx('FLOP, 2N per token','FLOP，每个 token 2N'),tip:tx(`2 × ${fP(e.N)} × 2T = ${fF(e.serve)} FLOPs.`,`2 × ${fP(e.N)} × 2T = ${fF(e.serve)} FLOP。`)},
      {k:'tot',label:tx('Training + serving','训练加服务'),v:fF(e.tot),raw:fFv(e.tot),sub:tx('FLOP, less is better','FLOP，越少越好'),tip:tx(`${fF(e.train)} + ${fF(e.serve)} = ${fF(e.tot)} FLOPs.`,`${fF(e.train)} + ${fF(e.serve)} = ${fF(e.tot)} FLOP。`)}];}
  const r=ev(S), hide=lossHidden();
  const m=[{k:'r',label:tx('Tokens per parameter','每个参数的 token'),v:fR(r.r),raw:r.r,dp:r.r<100?1:0,sub:tx(`${fTok(r.D)} tokens`,`${fTok(r.D)} 个 token`),tip:tx(`${fTok(r.D)} training tokens ÷ ${fP(S.N)} parameters = ${fR(r.r)}.`,`${fTok(r.D)} 个训练 token ÷ ${fP(S.N)} 个参数 = ${fR(r.r)}。`)},
    {k:'C',label:tx('Training compute','训练算力'),v:fF(S.C),raw:fFv(S.C),sub:tx('FLOP, C = 6ND','FLOP，C = 6ND'),tip:hide?(isP('mission')?tx(`Each of the three plans: ${A.ui.math(`C = 6 \\times N \\times D = ${fFT(PLAN_C)}`)} FLOPs.`,`三种方案都是 ${A.ui.math(`C = 6 \\times N \\times D = ${fFT(PLAN_C)}`)} FLOP。`):tx(`The budget you picked: ${fF(S.C)} FLOPs, however it is split between parameters and tokens.`,`你选的预算是 ${fF(S.C)} FLOP，不管怎么分给参数和 token，都是这么多。`)):tx(`6 × ${fP(S.N)} × ${fTok(r.D)} = ${fF(S.C)} FLOPs.`,`6 × ${fP(S.N)} × ${fTok(r.D)} = ${fF(S.C)} FLOP。`)},
    {k:'gd',label:tx('H100 GPU-days','H100 GPU 天'),v:fGD(r.gd),raw:r.gd,dp:r.gd<10?1:0,sub:tx(`at an MFU of ${fPct(S.u)}`,`按 MFU ${fPct(S.u)}`),tip:tx(`${fF(S.C)} FLOPs ÷ (989.5 TFLOP/s × MFU ${fPct(S.u)}) ÷ 86,400 s = about ${fGD(r.gd)} GPU-days. `,`${fF(S.C)} FLOP ÷（989.5 TFLOP/s × MFU ${fPct(S.u)}）÷ 86,400 秒 ≈ ${fGD(r.gd)} 个 GPU 天。`)+TIP.mfu()},
    {k:'L',label:tx('Loss (fitted)','loss（拟合）'),v:hide?'—':fL(r.L),raw:hide?null:r.L,dp:3,sub:hide?tx('shown in part 3','第 3 部分揭晓'):tx('this lab’s formula','这个 Lab 的公式'),tip:hide?tx('Hidden until part 3.','第 3 部分之前先隐藏。'):tx(`L(${fP(S.N)}, ${fTok(r.D)}) = ${fL(r.L)} in the fitted formula; a yardstick for comparing choices.`,`拟合公式算出 L(${fP(S.N)}, ${fTok(r.D)}) = ${fL(r.L)}，只用来比较不同的选择。`)}];
  if(hide) m.shift();
  if(v==='tot'||(mode==='sandbox'&&S.T>0)) m.push({k:'tot',label:tx('Training + serving','训练加服务'),v:fF(r.tot),raw:fFv(r.tot),sub:tx(`FLOP, serving ${S.T?fTokS(S.T):'0'}`,`FLOP，服务 ${S.T?fTokS(S.T):'0'}`),tip:tx(`${fF(S.C)} + 2 × ${fP(S.N)} × ${S.T?fTokS(S.T):'0'} = ${fF(r.tot)} FLOPs.`,`${fF(S.C)} + 2 × ${fP(S.N)} × ${S.T?fTokS(S.T):'0'} = ${fF(r.tot)} FLOP。`)});
  return m;
}
function drawMetrics(){
  const host=$('#sl-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev&&m.raw!=null&&prev.v!==m.v&&prev.raw>0){dt=A.fmt.delta(prev.raw,m.raw,{dp:m.dp!=null?m.dp:6})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.raw!=null?{v:m.v,raw:m.raw}:null;});
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4). Each formula is followed by its terms
   one per line, the symbols on the left and the same shape with the numbers in place on the right (the owner's review b8, item 7) */
const sci=x=>{const e=Math.floor(Math.log10(x)+1e-9); return `${fixd(x/Math.pow(10,e),3)}×10${sup(e)}`;};
/* a FLOP count as fF shows it (5.9×10²³), in TeX for a typeset hover (H1, 2026-10-09) */
const fFT=x=>fF(x).replace(/×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/,(_,e)=>' \\times 10^{'+[...e].map(c=>c==='⁻'?'-':'⁰¹²³⁴⁵⁶⁷⁸⁹'.indexOf(c)).join('')+'}');
const sciT=x=>{const e=Math.floor(Math.log10(x)+1e-9), m=fixd(x/Math.pow(10,e),3).replace(/\.?0+$/,''); return m==='1'?`10^{${e}}`:`${m} \\times 10^{${e}}`;};   // 8.000×10⁹ → 8×10⁹: shorter lines on a phone
const r3=v=>Math.round(v*1000)/1000;
/* the sum of the terms as shown: “=” when the shown terms add up to the shown loss, else “≈” (R6.6) */
const sumEq=(parts,tot)=>fixd(parts.reduce((x,y)=>x+r3(y),0),3)===fL(tot)?'=':'\\approx';
function lossRows(N,D,Lv){const tm=termsOf(N,D);
  return M(LFX)+`<p class="sl-mlab">${tx('Each term on its own line:','每一项单独算：')}</p>`+
    M(`E = ${FIT.E}`)+
    M(`\\frac{A}{N^{α}} = \\frac{${FIT.A}}{(${sciT(N)})^{${FIT.al}}} = ${fL(tm.n)}`)+
    M(`\\frac{B}{D^{β}} = \\frac{${FIT.B}}{(${sciT(D)})^{${FIT.be}}} = ${fL(tm.d)}`)+
    M(`L = ${fixd(FIT.E,3)} + ${fL(tm.n)} + ${fL(tm.d)} ${sumEq([FIT.E,tm.n,tm.d],Lv)} ${fL(Lv)}`);}
const SRVT=()=>tx('\\text{served tokens}','\\text{服务的 token 数}'), TOTW=()=>tx('\\text{total}','\\text{总算力}');
function costRows(N,D,T,train,serve,tot){
  return M(TOTX())+M(`6 N D = 6 \\times ${sciT(N)} \\times ${sciT(D)} = ${sciT(train)}`)+
    M(`2 N \\times ${SRVT()}`)+`<div class="sl-fx sl-fxc">${A.ui.math(`= 2 \\times ${sciT(N)} \\times ${T?sciT(T):'0'} = ${T?sciT(serve):'0'}`)}</div>`+   // the words make this line long: the numbers go on the next line
    M(`${TOTW()} = ${sciT(train)} + ${T?sciT(serve):'0'} = ${sciT(tot)}`);}
function mathHTML(){
  const v=viewNow(), t=T_, cm=tx(', ','，');
  if(v==='ch'){const e=chEval(chN,chD);
    return `<p>N = ${sci(e.N)}${tx(' parameters',' 个参数')}${cm}D = ${sci(e.D)}${tx(' training tokens',' 个训练 token')}</p>`+lossRows(e.N,e.D,e.L)+costRows(e.N,e.D,CH.T,e.train,e.serve,e.tot);}
  const gd=gpuDays(S.C,S.u);
  const gdRow=()=>M(tx('\\text{GPU-days} = \\frac{C}{989.5 \\times 10^{12} \\times \\text{MFU} \\times 86400}','\\text{GPU 天} = \\frac{C}{989.5 \\times 10^{12} \\times \\text{MFU} \\times 86400}'))+
    `<p>${t('u','MFU')} = ${fixd(S.u,2)}${tx(' (assumed)','（假设值）')}</p>`+
    M(`${tx('\\text{GPU-days}','\\text{GPU 天}')} = \\frac{${sciT(S.C)}}{989.5 \\times 10^{12} \\times ${fixd(S.u,2)} \\times 86400} \\approx ${gd<10?fixd(gd,1):fint(Math.round(gd))}`);
  /* parts 1 and 2: the three plans' 6ND only (part 2 adds the GPU-days of its budget); no size, ratio or loss before part 3 (R3.8) */
  if(lossHidden()){const plans=M('C \\approx 6 \\times N \\times D')+PLANK.map(k=>{const p=PLANS[k]; return `<p class="sl-mlab">${esc(planT(k))}</p>`+M(`C = 6 \\times ${sciT(p.N)} \\times ${sciT(p.D)} = ${sciT(6*p.N*p.D)}`);}).join('');
    return isP('mission')?plans+`<p>${tx('The losses appear in part 3.','loss 在第 3 部分给出。')}</p>`:plans+gdRow();}
  const r=ev(S), rows=[];
  rows.push(M('C = 6 \\times N \\times D')+`<p>${t('N','N')} = ${sci(S.N)}${cm}${t('C','C')} = ${sci(S.C)}</p>`+M(`D = \\frac{C}{6 N} = \\frac{${sciT(S.C)}}{6 \\times ${sciT(S.N)}} = ${sciT(r.D)}`));
  rows.push(lossRows(S.N,r.D,r.L));
  if(v==='iso'||v==='nd'){const a=FIT.be/(FIT.al+FIT.be), G=Math.pow(FIT.al*FIT.A/(FIT.be*FIT.B),1/(FIT.al+FIT.be));
    rows.push(`<p>${tx('The lowest point of the iso-FLOP curve (set the derivative of L along it to 0):','iso-FLOP 曲线的最低点（沿曲线对 L 求导、令导数为 0）：')}</p>`+M('N_{\\text{opt}} = G \\times (C / 6)^{a}, \\quad a = \\frac{β}{α + β}, \\quad G = (\\frac{αA}{βB})^{1/(α+β)}')+
      M(`a = \\frac{β}{α + β} = \\frac{${FIT.be}}{${FIT.al} + ${FIT.be}} = ${fixd(a,5)}`)+
      M(`G = (\\frac{αA}{βB})^{1/(α+β)} = (\\frac{${FIT.al} \\times ${FIT.A}}{${FIT.be} \\times ${FIT.B}})^{1/${fixd(FIT.al+FIT.be,5)}} = ${fixd(G,5)}`)+
      M(`N_{\\text{opt}} = G \\times (C / 6)^{a} = ${fixd(G,5)} \\times (${sciT(S.C/6)})^{${fixd(a,5)}} = ${sciT(r.o.N)}`)+
      M(`D_{\\text{opt}} = \\frac{C}{6 N_{\\text{opt}}} = \\frac{${sciT(S.C)}}{6 \\times ${sciT(r.o.N)}} = ${sciT(r.o.D)}`)+
      `<p>${tx(`${fR(r.o.r)} tokens per parameter`,`每个参数 ${fR(r.o.r)} 个 token`)}</p>`);}
  rows.push(gdRow());
  if(v==='tot'){const b=totBest(r.L,S.T), tn=FIT.A/Math.pow(b.N,FIT.al);
    rows.push(M(tx('\\text{total}(N) = 6 N \\times D_{\\text{need}}(N) + 2 N \\times \\text{served tokens}','\\text{总算力}(N) = 6 N \\times D_{\\text{need}}(N) + 2 N \\times \\text{服务的 token 数}'))+M('D_{\\text{need}}(N) = (\\frac{B}{L^{*} - E - A/N^{α}})^{1/β}')+
      `<p>L* = ${fixd(r.L,5)}${cm}${t('T',tx('served tokens','服务的 token 数'))} = ${S.T?fTokS(S.T):'0'}${tx('. Searching over N, the cheapest size is ','。在 N 上搜索，最省的大小是 ')}N = ${sci(b.N)}${tx(':','：')}</p>`+
      M(`D_{\\text{need}} = (\\frac{${FIT.B}}{${fixd(r.L,5)} - ${fixd(FIT.E,5)} - ${fixd(tn,5)}})^{1/${FIT.be}} = ${sciT(b.D)}`)+
      costRows(b.N,b.D,S.T,b.train,b.serve,b.tot).replace(M(TOTX()),''));}
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true; if(isP('serve')) tSeen.add(S.T);}
  const v=viewNow();
  drawHead();
  const vs=$('#sl-vsw',app), vh=vswHTML(); if(vs._h!==vh){vs.innerHTML=vh; vs._h=vh;} vs.hidden=!vh;
  drawChart();
  $('#sl-graph',app).dataset.v=v;
  drawMetrics();
  const mb=$('#sl-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the model size and the training tokens, in the task card. Fixed: the goal (loss 2.10 or lower) and 2T served tokens.','挑战中只能在任务卡里改模型大小和训练 token 数。其余固定：目标是 loss 不高于 2.10，要服务 2T 个 token。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return tx('Fixed in this part: the three plans. You can change the settings in Scenarios.','这一部分固定：三种方案。在应用场景里可以改这些设置。');
  if(id==='six') return tx('In this part only the budget changes.','这一部分只改预算。');
  if(id==='terms') return tx('In this part you pick a plan in the card, or change the parameters and the tokens next to the bar.','这一部分在卡片里选方案，也可以在图里改参数量和 token 数。');
  if(id==='iso') return tx('In this part you change the model size and the budget. Nothing is served.','这一部分改模型大小和预算，不算服务。');
  if(id==='serve') return tx('Fixed in this part: Llama 3 8B, with 8B parameters and 15T tokens. Only the served tokens change.','这一部分固定：Llama 3 8B，8B 个参数、15T 个 token。只改服务的 token 数。');
  return tx('Settings are fixed on this part; you can change them in Scenarios.','这一部分的设置固定；在应用场景里可以改。');
}
function syncControls(){
  const D=S.C/(6*S.N);
  /* every slider showing this value gets its thumb, its fill and its number, the one being dragged too (the owner's review b8: the
     dragged slider used to keep its old fill and number, because a focused slider was skipped whole, not only its value) */
  const setR=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o&&o.textContent!==out) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  setR('lc',snap(Math.log10(S.C),LC_MIN,LC_STEP),fF(S.C)+' FLOP'); setR('ln',snap(Math.log10(S.N),LN_MIN,LN_STEP),fP(S.N)); setR('ld',snap(Math.log10(D),LD_MIN,LD_STEP),fTok(D)); setR('u',Math.round(S.u*100),fPct(S.u));
  $$('.seg-ctl[data-key="T"]',app).forEach(g=>A.ui.setPressed(g,String(TS.indexOf(S.T)))); $$('.seg-ctl[data-key="cap"]',app).forEach(g=>A.ui.setPressed(g,String(CAPS.indexOf(S.cap))));
  $$('.seg-ctl[data-key="bud"]',app).forEach(g=>A.ui.setPressed(g,String(Math.round(Math.log10(S.C)*100)/100)));
  $$('.seg-ctl[data-key="view"]',app).forEach(g=>A.ui.setPressed(g,view));
  const g=mode==='guided', ch=mode==='challenge', id=g?STEPS[step].id:'', hide=(sel,h)=>{const e=$(sel,app); if(e) e.hidden=h;};
  $('#sl-controls',app).classList.toggle('locked',ch);
  hide('#sl-ctl-lc-wrap',ch||(g&&!['six','iso'].includes(id))); hide('#sl-ctl-ln-wrap',ch||(g&&!['iso','terms'].includes(id))); hide('#sl-ctl-T-wrap',ch||(g&&id!=='serve'));
  ['#sl-ctl-cap-wrap','#sl-ctl-reset-wrap'].forEach(sel=>hide(sel,g||ch)); hide('#sl-ctl-as',ch||(g&&id!=='six'));
  const ln=$('#sl-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const v=viewNow();
  if(v==='ch'){const e=chEval(chN,chD); A.announce(tx(`${chName(chN,chD)}: loss ${fL(e.L)}, ${fF(e.tot)} FLOPs in all.`,`${chName(chN,chD)}：loss ${fL(e.L)}，一共 ${fF(e.tot)} FLOP。`)); return;}
  if(lossHidden()) return; const r=ev(S); A.announce(tx(`${fF(S.C)} FLOPs, ${fP(S.N)} parameters, ${fTok(r.D)} tokens: loss ${fL(r.L)}.`,`${fF(S.C)} FLOP，${fP(S.N)} 个参数，${fTok(r.D)} 个 token：loss ${fL(r.L)}。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
/* a budget change draws the curve moving to its new place over 350 ms (R3.20: only when the picture changes; reduced motion jumps) */
function tweenTo(from,to){
  if(raf) cancelAnimationFrame(raf); raf=0; drawC=null;
  if(A.reduceMotion()||!isFinite(from)||Math.abs(lg(from)-lg(to))<1e-6||!app) return;
  tweens++; const t0=performance.now(), a=lg(from), b=lg(to);
  const tick=t=>{raf=0; if(!app) return; const k=Math.min(1,(t-t0)/350), e=k<0.5?2*k*k:1-Math.pow(-2*k+2,2)/2; drawC=k>=1?null:Math.pow(10,a+(b-a)*e); drawChart(); if(k<1) raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick);
}
function setState(p,opt){
  const was=S.C;
  Object.assign(S,p);
  const free=isP('terms'); S.C=Math.min(free?6e26:Math.pow(10,LC_MAX),Math.max(free?6e18:Math.pow(10,LC_MIN),S.C)); S.N=Math.min(1e12,Math.max(1e8,S.N)); if(!TS.includes(S.T)) S.T=0; if(!CAPS.includes(S.cap)) S.cap=0; S.u=Math.min(0.6,Math.max(0.2,Math.round(S.u*20)/20));
  render();
  if(!(opt&&opt.noTween)&&Math.abs(lg(was)-lg(S.C))>1e-6&&(viewNow()==='iso'||viewNow()==='nd')) tweenTo(was,S.C);
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setBudget(C){budgetMoved=true; const o=optOf(C); setState(isP('six')?{C,N:o.N}:{C,N:isP('iso')?S.N:o.N});}
function setN(N){setState({N});}
function setT(T){tSeen.add(T); setState({T});}
function setLv3(k){lv3=k; lv3Tried.add(k); setState(planState(k));}
function setRem(k){if(!['nd','iso','tot'].includes(k)) return; remView=k;
  S={...BASE}; if(k==='nd') S={...S,C:1e23,N:optOf(1e23).N}; if(k==='iso') S={...S,C:1e23,N:optOf(1e23).N}; if(k==='tot') S={...S,C:6*LLAMA8*TOK15,N:LLAMA8,T:1e14};
  renderRail(); if(innerWidth<=760) A.ui.reveal($('#sl-stage',app));}
function enterStep(i){
  wantObs=false; if(raf) cancelAnimationFrame(raf); raf=0; drawC=null; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#sl-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#sl-rail .step.current',app)||$('#sl-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#sl-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#sl-rail .step.current',app)||$('#sl-rail .railcard',app)||$('#sl-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#sl-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#sl-insight',app)); else revealBox($('#sl-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; if(raf) cancelAnimationFrame(raf); raf=0; drawC=null;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; allOpen=false; chN=CH.start.n; chD=CH.start.d;}
  if(m==='sandbox'){sbTouched=false; if(was==='guided'){const p=PRESETS[0]; S={...BASE,...p.s()}; view=p.v;}}
  A.shell.setMode(app,m,DESCS());
  const st=$('#sl-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[Math.round(lg(S.C)*100),Math.round(lg(S.N)*1000),TS.indexOf(S.T),Math.round(S.u*100),CAPS.indexOf(S.cap),VIEWS.indexOf(view)].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==6||p.some(x=>!Number.isInteger(x))) return null;
  const [c,n,ti,u,ci,vi]=p;
  if(c<LC_MIN*100||c>LC_MAX*100||n<8000||n>12000||ti<0||ti>=TS.length||u<20||u>60||ci<0||ci>=CAPS.length||vi<0||vi>=VIEWS.length) return null;
  return {s:{C:Math.pow(10,c/100),N:Math.pow(10,n/1000),T:TS[ti],u:u/100,cap:CAPS[ci]},v:VIEWS[vi]};
}
/* presets keep exact values; a parsed token is matched back to a preset when it rounds to one */
function snapPreset(s){for(const p of PRESETS){const q=p.s(); if(Math.abs(lg(q.C)-lg(s.C))<0.006&&Math.abs(lg(q.N)-lg(s.N))<0.0006&&q.T===s.T&&q.cap===s.cap) return {...s,C:q.C,N:q.N};} return s;}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+3), pk=a=>a[Math.floor(r()*a.length)];
  const C=pk([3e21,3e22,3e23,3e24]), u=pk([0.35,0.4,0.45]), o=optOf(C); return {seed,C,u,o,gd:gpuDays(C,u)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#sl-drill-q',app), a=$('#sl-drill-a',app);
  qd.innerHTML=tx(`You have <b>${fF(d.C)}</b> FLOPs of training compute on H100s (peak 989.5 TFLOP/s), and training reaches an MFU of <b>${fPct(d.u)}</b>. Using this lab’s fit, how big should the model be, how many tokens should it see, and how many GPU-days will it take?`,
                  `你有 <b>${fF(d.C)}</b> FLOP 的训练算力，用 H100 训练（峰值 989.5 TFLOP/s），MFU 是 <b>${fPct(d.u)}</b>。按这个 Lab 的拟合，模型该多大、训多少 token、要多少 GPU 天？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const aE=FIT.be/(FIT.al+FIT.be), G=Math.pow(FIT.al*FIT.A/(FIT.be*FIT.B),1/(FIT.al+FIT.be));
  a.innerHTML=tx(`<p><b>About ${fP(d.o.N)} parameters on ${fTok(d.o.D)} tokens (${fR(d.o.r)} per parameter), about ${fGD(d.gd)} GPU-days.</b></p>`+
    `<p>The best size, with ${A.ui.math('a = \\frac{\\beta}{\\alpha + \\beta}')} = ${fixd(aE,4)} and G = ${fixd(G,4)} from the fit:</p>`+M(`N_{\\text{opt}} = G \\times \\left(\\frac{C}{6}\\right)^{a} = ${sciT(d.o.N)}`)+`<p>The tokens:</p>`+M(`D = \\frac{C}{6N} = ${sciT(d.o.D)}`)+`<p>The GPU-days (all GPUs together):</p>`+M(`\\text{GPU-days} = \\frac{${sciT(d.C)}}{989.5 \\times 10^{12} \\times ${fixd(d.u,2)} \\times 86,400} = ${fGD(d.gd)}`),
    `<p><b>约 ${fP(d.o.N)} 个参数、${fTok(d.o.D)} 个 token（每个参数 ${fR(d.o.r)} 个），约 ${fGD(d.gd)} 个 GPU 天。</b></p>`+
    `<p>最好的模型大小，拟合给出 ${A.ui.math('a = \\frac{\\beta}{\\alpha + \\beta}')} = ${fixd(aE,4)}，G = ${fixd(G,4)}：</p>`+M(`N_{\\text{opt}} = G \\times \\left(\\frac{C}{6}\\right)^{a} = ${sciT(d.o.N)}`)+`<p>token 数：</p>`+M(`D = \\frac{C}{6N} = ${sciT(d.o.D)}`)+`<p>GPU 天（所有卡加在一起）：</p>`+M(`\\text{GPU 天} = \\frac{${sciT(d.C)}}{989.5 \\times 10^{12} \\times ${fixd(d.u,2)} \\times 86,400} = ${fGD(d.gd)}`));
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={C:{en:'Compute budget',zh:'算力预算'},N:{en:'Model size (parameters)',zh:'模型大小（参数量）'},T:{en:'Tokens it will serve',zh:'它要服务的 token 数'},u:{en:'MFU',zh:'MFU'}};
const TERM={
 N:()=>tx(`N is the number of parameters, ${fP(S.N)} here.`,`N 是参数量，这里是 ${fP(S.N)}。`),
 C:()=>tx(`C is the training compute, ${fF(S.C)} FLOPs here.`,`C 是训练算力，这里是 ${fF(S.C)} FLOP。`),
 T:()=>tx(`The served tokens: how many tokens the model will generate for users, ${S.T?fTokS(S.T):'0'} here (assumed).`,`服务的 token 数：模型要为用户生成多少 token，这里是 ${S.T?fTokS(S.T):'0'}（假设值）。`),
 u:()=>tx(`MFU is the share of the GPU’s peak that training really gets, ${fPct(S.u)} here (assumed).`,`MFU 是训练真正用上的那部分 GPU 峰值算力的比例，这里是 ${fPct(S.u)}（假设值）。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: change the plan in the task card.','挑战中固定不变：请在任务卡里改方案。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(k==='C'&&['six','iso'].includes(id)) return ''; if(k==='N'&&['iso','terms'].includes(id)) return ''; if(k==='T'&&id==='serve') return ''; if(k==='u'&&id==='six') return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='mfu') return ctlLockMsg('u')||tx('Click to jump to “MFU” in Settings.','点击跳到“自由调参”里的“MFU”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({C:$('#sl-ic-lc',app)||$('#sl-ic-bud',app)||$('#sl-ctl-lc',app),N:$('#sl-ic-ln',app)||$('#sl-st-ln',app)||$('#sl-ctl-ln',app),T:$('#sl-ic-T',app)||$('#sl-ctl-T',app),u:$('#sl-ctl-u',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#sl-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#sl-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='mfu'){if(ctlLockMsg('u')){if(src) A.tip.show(src,`<div>${esc(jumpAct('mfu'))}</div>`); return;} const c=$('#sl-ctl-u',app); A.tip.hide(); if(c&&c.getClientRects().length) A.ui.jumpToEl(c); else A.ui.jumpToEl($('#sl-controls',app)); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden in this view.','在现在的视图里，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...BASE,...PRESETS[0].s()}; view=PRESETS[0].v; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chN,chD).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#sl-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#sl-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='lc') bySettings(e.target,()=>{const C=Math.pow(10,LC_MIN+v*LC_STEP); if(isP('six')){budgetMoved=true; setState({C,N:optOf(C).N});} else setState({C});});
    else if(k==='ln') bySettings(e.target,()=>{const N=Math.pow(10,LN_MIN+v*LN_STEP); if(isP('terms')){const D=S.C/(6*S.N); setState({N,C:6*N*D});} else setState({N});});
    else if(k==='ld') bySettings(e.target,()=>{const D=Math.pow(10,LD_MIN+v*LD_STEP); setState({C:6*S.N*D},{noTween:true});});
    else if(k==='u') bySettings(e.target,()=>setState({u:v/100}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#sl-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#sl-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#sl-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('terms')){const f=()=>byLearner(()=>setLv3(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='T') bySettings(b,()=>setT(TS[+d.v]));
      else if(k==='cap') bySettings(b,()=>setState({cap:CAPS[+d.v]}));
      else if(k==='bud') byLearner(()=>{const C=Math.pow(10,+d.v); setState({C});});
      else if(k==='view'){view=d.v; byLearner(()=>render());}
      else if(k==='chn') byLearner(()=>{chN=+d.v; render();});
      else if(k==='chd') byLearner(()=>{chD=+d.v; render();});
      if(seg.closest('#sl-rail')){wantObs=true; revealSoon(); wantObs=false;}
      return;}
    /* a scenario sets the whole state: no ▲▼ against the previous scenario (R6.6), so this is not counted as the learner's own change */
    if(d.preset!=null){const p=PRESETS[+d.preset]; view=p.v; sbTouched=true; setState({...p.s()}); requestAnimationFrame(()=>{if(app) revealBox($('#sl-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#sl-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let lastW=0;
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S as it was (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount.
     S keeps its exact numbers (a part's budget such as 6 × 8B × 15T is not on the token's grid), checked against setState's limits */
  snap:()=>({v:1,mode,step,S:{...S},view,pred,observed:{...observed},saved:saved?{...saved}:null,chN,chD,chSubmitted,chReveal,allOpen,sbTouched,budgetMoved,lv3,
    lv3Tried:[...lv3Tried],tSeen:[...tSeen],remView,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(chN,chD).ok}),
  restore(o){if(!o||o.v!==1) return;
    const num=(v,lo,hi)=>typeof v==='number'&&Number.isFinite(v)&&v>=lo&&v<=hi;
    const cfg=x=>x&&typeof x==='object'&&num(x.C,6e18,6e26)&&num(x.N,1e8,1e12)&&TS.includes(x.T)&&num(x.u,0.2,0.6)&&CAPS.includes(x.cap)?{C:x.C,N:x.N,T:x.T,u:x.u,cap:x.cap}:null;
    const p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p}; view=VIEWS.includes(o.view)?o.view:'iso';
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    const ch=mode==='challenge', ix=(v,arr,d)=>ch&&Number.isInteger(v)&&v>=0&&v<arr.length?v:d;
    chN=ix(o.chN,CH.Ns,CH.start.n); chD=ix(o.chD,CH.Ds,CH.start.d);
    chSubmitted=ch&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; allOpen=chSubmitted&&o.allOpen===true; sbTouched=o.sbTouched===true;
    budgetMoved=o.budgetMoved===true; lv3=PLANK.includes(o.lv3)?o.lv3:null;
    lv3Tried=new Set(Array.isArray(o.lv3Tried)?o.lv3Tried.filter(x=>PLANK.includes(x)):[]); if(lv3) lv3Tried.add(lv3);
    tSeen=new Set([0]); if(Array.isArray(o.tSeen)) o.tSeen.forEach(x=>{if(TS.includes(x)) tSeen.add(x);});
    remView=['nd','iso','tot'].includes(o.remView)?o.remView:'nd';
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el; if(raf) cancelAnimationFrame(raf); raf=0; drawC=null;
    const p=tok?parse(tok):null; if(p&&!(fromSnap&&token()===tok)){S={...BASE,...snapPreset(p.s)}; view=p.v; mode='sandbox'; sbTouched=true;}   // the address's settings, unless the saved state is that same setup with its exact numbers
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#sl-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#sl-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fromSnap=false;
    fillRail();
    render(); lastW=innerWidth;
  },
  leave(){clearTimeout(annT); stageStop(); if(raf) cancelAnimationFrame(raf); raf=0; drawC=null; root=null; app=null;},
  onLang(){if(root){const d=$('#sl-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#sl-math',app).open=true;}},
  onResize(){if(app&&innerWidth!==lastW){lastW=innerWidth; const h=$('#sl-chart',app); if(h) h._h=null; render();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#sl-chart',app).hidden?$('#sl-terms',app):$('#sl-chart',app)),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#sl-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      vis.innerHTML=miniVis();
      if(v==='ch'){const e=chEval(chN,chD); read.innerHTML=esc(tx(`${chName(chN,chD)}: loss ${fL(e.L)}, ${fF(e.tot)} FLOPs in all`,`${chName(chN,chD)}：loss ${fL(e.L)}，一共 ${fF(e.tot)} FLOP`)); return;}
      const r=ev(S);
      if(isP('mission')){read.innerHTML=esc(tx(`Three plans, ${fF(PLAN_C)} FLOPs each: 280B, 70B or 17.5B parameters`,`三种方案，每种 ${fF(PLAN_C)} FLOP：280B、70B 或 17.5B 个参数`)); return;}
      if(lossHidden()){read.innerHTML=esc(tx(`${fF(S.C)} FLOPs: about ${fGD(r.gd)} H100 GPU-days at an MFU of ${fPct(S.u)}`,`${fF(S.C)} FLOP：按 MFU ${fPct(S.u)} 算，约 ${fGD(r.gd)} 个 H100 GPU 天`)); return;}
      if(v==='tot'){const b=totBest(r.L,S.T); read.innerHTML=esc(tx(`Serving ${S.T?fTokS(S.T):'0'} tokens: the cheapest size for loss ${fL(r.L)} is ${fP(b.N)}`,`服务 ${S.T?fTokS(S.T):'0'} 个 token：loss ${fL(r.L)} 时最省的大小是 ${fP(b.N)}`)); return;}
      read.innerHTML=esc(tx(`${fF(S.C)} FLOPs: ${fP(S.N)} parameters on ${fTok(r.D)} tokens, loss ${fL(r.L)}`,`${fF(S.C)} FLOP：${fP(S.N)} 个参数、${fTok(r.D)} 个 token，loss ${fL(r.L)}`));}
  }
});
/* the mini result bar's picture: the model size on a 100M–1T log track, the lowest point of the budget as a ring */
function miniVis(){const p=v=>Math.max(0,Math.min(100,(lg(v)-8)/4*100)); const v=viewNow();
  if(lossHidden()) return `<div class="sl-dk">${PLANK.map(k=>`<b style="left:${p(PLANS[k].N).toFixed(1)}%"></b>`).join('')}</div>`;   // parts 1–2: the three plans only, no best size (R3.8)
  if(v==='ch') return `<div class="sl-dk"><i style="left:${p(CH.Ns[chN]).toFixed(1)}%"></i></div>`;
  if(v==='tot'){const b=totBest(ev(S).L,S.T); return `<div class="sl-dk"><b style="left:${p(b.N).toFixed(1)}%"></b><i style="left:${p(S.N).toFixed(1)}%"></i></div>`;}
  return `<div class="sl-dk"><b style="left:${p(optOf(S.C).N).toFixed(1)}%"></b><i style="left:${p(S.N).toFixed(1)}%"></i></div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__sl=Object.freeze({
  FIT:{...FIT},FIT0:{...FIT0},lossOf:(N,D,orig)=>lossOf(N,D,orig?FIT0:FIT),opt:(C,orig)=>deep(optOf(C,orig?FIT0:FIT)),dNeed,totBest:(Lt,T)=>deep(totBest(Lt,T)),gpuDays,L5:()=>L5,
  chEval:(n,d)=>deep(chEval(n,d)),chAll:()=>deep(chAll()),chBest:()=>deep(chBest()),chTrainBest:()=>deep(chTrainBest()),drill:s=>deep(makeDrill(s)),
  chOne:(n,d)=>{const o=chOne(chEval(n,d)); return o?deep(o):null;},
  /* G7: the single-change sentence the result shows for a plan, in the page's language, and its values as the result shows them */
  chNextText:(n,d)=>{const x=document.createElement('div'); x.innerHTML=sumHTML(chEval(n,d)); const p=x.querySelector('.sl-next'); return p?p.textContent:'';},
  chShown:(n,d)=>{const e=chEval(n,d); return {tot:fFv(e.tot),totS:fF(e.tot),L:fL(e.L),ok:e.ok};},chGap:(n,d)=>chGap(chEval(n,d)),chCounter:(n,d)=>{const o=chCounter(chEval(n,d)); return o?deep(o):null;},nMin,
  fmt:{fF,fP,fTok,fGD,fR,fL},
  consts:Object.freeze({BUD:BUD.slice(),TS:TS.slice(),CAPS:CAPS.slice(),CH:deep(CH),BASE:{...BASE},REAL:deep(REAL),PLANS:deep(PLANS),PLAN_C,STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,H100,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s(),v:p.v})))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},view:viewNow(),budgetMoved,lv3Tried:[...lv3Tried],tSeen:[...tSeen],remView,chN,chD,chSubmitted,chReveal,tweens,tweening:raf!==0,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transferQ:e.transfer.map(q=>q.q),transferWhy:e.transfer.map(q=>q.why)});}
});
})();
