/* optimizers (roots, ML basics): What does Adam do that SGD does not?
   Fidelity: Toy run. Real optimizer updates run in the page on a two-parameter loss, deterministic (no noise, no data):
   - the valley f(x, y) = ½(x² + 25 y²), start (−9, 1.5): the gradient is (x, 25 y), so y is the steep direction and x the flat one;
   - the challenge's valley f = ½(x² + 100 y²), start (−5, 0.8), never shown before the challenge (R3.8);
   - a curved valley for one scenario: f = (1 − x)² + 10 (y − x²)², a gentler Rosenbrock function (10 instead of 100), start (−1.2, 1).
   Updates (PyTorch's conventions): SGD θ −= lr·g; momentum (heavy ball, as torch.optim.SGD with dampening 0, not Nesterov) b = β·b + g,
   θ −= lr·b; Adam m = β1·m + (1−β1)·g, v = β2·v + (1−β2)·g², m̂ = m ÷ (1−β1^t), v̂ = v ÷ (1−β2^t) (bias correction, can be switched off),
   θ −= lr·m̂ ÷ (√v̂ + ε); AdamW also θ −= lr·wd·θ (decoupled, as torch.optim.AdamW). Optional warmup (linear) then cosine to 10%.
   A run "reaches the bottom" at the first step whose loss is below 0.001; it "diverges" once the loss passes 10⁸; runs stop at 200 steps.
   The paths and loss curves play back on their own when the chart comes into view (R3.9, owner check 2026-10-03): play / pause,
   replay, three speeds (steps per second), the step slider as a second control; with reduced motion the chart shows the end state.
   Part 7 (AdamW): one weight of size 1.0 whose loss gradient flips sign every step (±s: the loss pulls it nowhere on average). Once
   m and v have settled, the share of the weight that weight decay removes in one step is lr·wd for AdamW, whatever s, and
   lr·wd·θ ÷ √(s² + (wd·θ)²) ÷ θ for Adam + L2 (the decay enters the gradient, so Adam divides it by √v); lr 0.01, wd 0.1, s 10 and 0.01.
   window.__op is a read-only hook for tests/optimizers_numbers.py and tests/optimizers_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='optimizers';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const quad=(a,b)=>({f:p=>0.5*(a*p[0]*p[0]+b*p[1]*p[1]),g:p=>[a*p[0],b*p[1]]});
const SURF={
 v25:{...quad(1,25),a:1,b:25,start:[-9,1.5],view:[-10,5,-2.5,2.5],levels:[0.1,0.5,2,8,32,128],min:[0,0]},
 v100:{...quad(1,100),a:1,b:100,start:[-5,0.8],view:[-6,2.5,-1.8,1.8],levels:[0.1,0.5,2,8,32,128],min:[0,0]},
 curve:{f:p=>(1-p[0])**2+10*(p[1]-p[0]*p[0])**2,g:p=>[-2*(1-p[0])-40*p[0]*(p[1]-p[0]*p[0]),20*(p[1]-p[0]*p[0])],start:[-1.2,1],view:[-2,2,-0.6,2.4],levels:[0.03,0.2,1,3,8,20],min:[1,1]}
};
const N=200, GOAL=1e-3, BLOW=1e8;
/* the learning rate at step t: warmup (linear from lr/warm) then cosine down to 10% of the peak by step N */
function lrAt(c,t){if(!c.warm) return c.lr; if(t<=c.warm) return c.lr*t/c.warm; const p=(t-c.warm)/(N-c.warm); return c.lr*(0.1+0.9*0.5*(1+Math.cos(Math.PI*p)));}
/* heavy ball on one direction of curvature lam: θ(t+1) = (1 + β − lr·lam) θ(t) − β θ(t−1); the size of the larger root is the factor
   that direction shrinks by per step (|1 − lr·lam| for SGD, √β once the roots are complex, i.e. once the direction swings) */
function hbRate(lam,lr,beta){const a=1+beta-lr*lam; if(!beta) return Math.abs(1-lr*lam); const d=a*a-4*beta; return d<0?Math.sqrt(beta):Math.max(Math.abs(a+Math.sqrt(d)),Math.abs(a-Math.sqrt(d)))/2;}
const RC=new Map();
/* one run: the points, the loss at every step, the update of every step, the first step under the goal, divergence */
function run(c){
  const key=JSON.stringify(c); if(RC.has(key)) return RC.get(key);
  const S=SURF[c.surf||'v25'], b1=c.b1==null?0.9:c.b1, b2=c.b2==null?0.999:c.b2, eps=c.eps==null?1e-8:c.eps, bc=c.bc!==false, scale=c.scale||1;
  let x=S.start.slice(); const pts=[x.slice()], loss=[S.f(x)*scale], d=[[0,0]], lrs=[0];
  let v=[0,0], m=[0,0], s=[0,0], hit=null, div=null;
  for(let t=1;t<=N;t++){
    const g=S.g(x).map(q=>q*scale), lr=lrAt(c,t); let u;
    if(c.opt==='sgd') u=g.map(q=>lr*q);
    else if(c.opt==='mom'){v=v.map((q,i)=>c.beta*q+g[i]); u=v.map(q=>lr*q);}
    else {m=m.map((q,i)=>b1*q+(1-b1)*g[i]); s=s.map((q,i)=>b2*q+(1-b2)*g[i]*g[i]);
      const mh=m.map(q=>bc?q/(1-b1**t):q), sh=s.map(q=>bc?q/(1-b2**t):q);
      u=mh.map((q,i)=>lr*q/(Math.sqrt(sh[i])+eps));
      if(c.opt==='adamw'&&c.wd) u=u.map((q,i)=>q+lr*c.wd*x[i]);}
    x=x.map((q,i)=>q-u[i]); const f=S.f(x)*scale;
    pts.push(x.slice()); loss.push(f); d.push(u); lrs.push(lr);
    if(!isFinite(f)||f>BLOW){div=t; break;}
    if(hit==null&&f<GOAL*scale) hit=t;
  }
  const r={c:{...c},pts,loss,d,lrs,hit,div,n:pts.length-1};
  RC.set(key,r); return r;
}
/* the optimizers as the lab names them; mem = extra numbers kept per parameter */
const OPT={sgd:{en:'SGD',zh:'SGD',mem:0},mom:{en:'Momentum',zh:'Momentum',mem:1},adam:{en:'Adam',zh:'Adam',mem:2},adamw:{en:'AdamW',zh:'AdamW',mem:2}};
const SGD05={opt:'sgd',lr:0.05}, MOM09={opt:'mom',lr:0.05,beta:0.9}, ADAM05={opt:'adam',lr:0.5};
const LR_PART=[...Array(20).keys()].map(i=>Math.round((i+1)*5)/1000);          // part 3: 0.005 … 0.100
const BETAS=[...Array(20).keys()].map(i=>Math.round(i*5)/100);                  // part 4: 0 … 0.95
const LR_ADAM=[0.05,0.1,0.2,0.3,0.5,1.0];                                         // part 5
const LR_ALL=[0.001,0.002,0.003,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.085,0.09,0.1,0.2,0.3,0.5,1.0];
/* the challenge: a valley 4 times steeper across, a new start; optimizer × learning rate, within 100 steps */
const CH={opts:['sgd','mom','adam'],lrs:[0.003,0.01,0.03,0.1,0.3,1.0],max:100,start:{opt:'sgd',lr:0.01},beta:0.9};
const chCfg=(o,lr)=>({opt:o,lr,beta:CH.beta,surf:'v100'});
function chEval(o,lr){const r=run(chCfg(o,lr)); return {o,lr,r,steps:r.hit,ok:r.hit!=null&&r.hit<=CH.max};}
const chChanges=(o,lr)=>(o!==CH.start.opt?1:0)+(lr!==CH.start.lr?1:0);
/* part 7: the share of a weight that weight decay removes in one step, once m and v have settled (see the header) */
const WD={lr:0.01,wd:0.1,th:1,big:10,small:0.01};
const wdPull=(kind,s)=>kind==='adamw'?WD.lr*WD.wd:WD.lr*WD.wd/Math.sqrt(s*s+(WD.wd*WD.th)**2);
const WDK=['l2','adamw'];
/* playback speeds, in steps per second (time-based, R3.9): medium plays a 100-step run in 2.5 s */
const SPEEDS={slow:40,med:22,fast:10};   // seconds for a whole replay of the longest run on screen (R3.9)
/* fewest settings moved first (R4.5), then the fewest extra numbers per parameter, then the fewest steps */
function chBest(){let b=null; CH.opts.forEach(o=>CH.lrs.forEach(lr=>{const e=chEval(o,lr); if(!e.ok) return; const k=[chChanges(o,lr),OPT[o].mem,e.steps];
  if(!b||k[0]<b.k[0]||(k[0]===b.k[0]&&(k[1]<b.k[1]||(k[1]===b.k[1]&&k[2]<b.k[2])))) b={o,lr,k,steps:e.steps};})); return b;}

/* =====================================================================
   formatting
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fLR=v=>v>=1?fixd(v,1):String(+v.toFixed(3));
const fB=v=>fixd(v,2);
const MINUS='−';
const sgn=s=>String(s).replace(/^-/,MINUS);
const f2=v=>sgn(fixd(v,2));
const fLoss=v=>{if(!isFinite(v)||v>=1e6) return tx('over 1M','超过 1M'); if(v>=100) return fixd(v,0); if(v>=1) return fixd(v,1); if(v>=0.001) return String(+v.toPrecision(2)); return v.toExponential(1).replace('e-','e−');};
const optName=o=>L(OPT[o]);
/* the result of a run in a few words: steps to the bottom, or why not */
const resShort=r=>r.hit!=null?tx(`${r.hit} steps`,`${r.hit} 步`):(r.div!=null?tx('diverges','发散'):tx(`not in ${N} steps`,`${N} 步内到不了`));
const resLong=r=>r.hit!=null?tx(`the loss is below 0.001 after ${r.hit} steps`,`${r.hit} 步后 loss 低于 0.001`):(r.div!=null?tx(`it diverges, and the loss passes 100M at step ${r.div}`,`它发散了，第 ${r.div} 步 loss 就超过 100M`):tx(`the loss is still above 0.001 after ${N} steps`,`${N} 步后 loss 还在 0.001 以上`));
/* a setting and its result inside a list: “label: result” in English, “label（result）” in Chinese (one colon per sentence, R6.7) */
const pairTxt=(a,b)=>tx(`${a}: ${b}`,`${a}（${b}）`);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const math=t=>A.ui.math?A.ui.math(t):esc(t);
/* the valleys as typeset formulas (H1, the owner's decision of 2026-10-09) */
const VAL=b=>math(`\\frac{1}{2}(x^2 + ${b}\\,y^2)`), ROS=()=>math('(1 - x)^2 + 10\\,(y - x^2)^2');

/* =====================================================================
   module state
   ===================================================================== */
const BASE={opt:'sgd',lr:0.05,beta:0.9,b2:0.999,bc:true,wd:0,warm:0,surf:'v25'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lrP=0.05, betaP=0, lrA=0.05, bcP='on', seenDiv=false, betaMoved=false, lrAMoved=false, bcTried=new Set(['on']);
let K=1, saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null;
let drillSeed=2, drillOpen=false;
let wdP='l2', wdTried=new Set(['l2']);
let playing=false, raf=0, spd='med', pendWant=false, runSig='', io=null, inView=false, onMap=[];   // onMap: the runs whose paths are on the valley now (R3.20)
const endOf=r=>r.hit!=null?r.hit:r.n;   // the last step drawn for a run: where it reached the bottom, diverged, or step 200
const shownPrev={};

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
/* each definition says the idea first, in plain words, and gives the formula after it (R6.27) */
const TIP={
 grad:()=>tx('The gradient says, for every parameter, which way is uphill and how steep it is: how much the loss rises when that parameter grows a little. Here it is (x, 25 y): at the same distance from the bottom, the valley is 25 times steeper across than along.',
             '梯度（gradient）告诉你每个参数往哪边是上坡、坡有多陡：这个参数增加一点时，loss 涨多少。这里梯度是 (x, 25 y)：离谷底同样远时，横穿山谷的方向比沿着山谷陡 25 倍。'),
 lr:()=>tx('The learning rate (lr) sets how big each step is. In SGD the step is lr times the gradient, so a steep direction gets a big step and a flat one a small step.',
           '学习率（learning rate，lr）决定每一步迈多大。SGD 的一步是 lr 乘以梯度，所以陡的方向一步迈得大，平的方向迈得小。'),
 sgd:()=>tx(`SGD (stochastic gradient descent) moves every parameter downhill, by a step in proportion to its gradient. This valley has no noise, so here it is plain gradient descent. Each step:<div class="fx">${math('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot g')}</div>`,
            `SGD（随机梯度下降）让每个参数往下坡走，步子和它的梯度成正比。这条山谷没有噪声，所以这里就是普通的梯度下降。每一步：<div class="fx">${math('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot g')}</div>`),
 mom:()=>tx(`Momentum keeps a running direction. Steps that keep pointing the same way add up and build speed; steps that go back and forth cancel out. In the heavy-ball form PyTorch uses:<div class="fx">${math('b \\leftarrow \\beta b + g,\\quad \\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot b')}</div>b is the running direction (PyTorch’s momentum_buffer).`,
            `momentum（动量）记着一个累积起来的方向。一直朝同一边的步子会叠加，越走越快。来回反向的步子会互相抵消。PyTorch 用的 heavy-ball 形式是：<div class="fx">${math('b \\leftarrow \\beta b + g,\\quad \\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot b')}</div>b 是累积的方向，PyTorch 叫它 momentum_buffer。`),
 adam:()=>tx(`Adam gives each parameter its own step size: it divides the gradient by the typical size of that parameter’s recent gradients. So while its gradient keeps the same direction, a parameter moves about lr per step whether its gradients are big or small; when the direction keeps flipping, the steps are much smaller.<div class="fx">${math('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot \\frac{m}{\\sqrt{v}}')}</div>m and v are running averages of g and ${math('g^2')}.`,
             `Adam 给每个参数各自的步长：它把梯度除以这个参数最近梯度的典型大小。所以只要梯度方向稳定，不管梯度大还是小，每一步都大约走一个学习率。方向来回变时，步子会小得多。<div class="fx">${math('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot \\frac{m}{\\sqrt{v}}')}</div>m 和 v 分别是 g 和 ${math('g^2')} 的滑动平均。`),
 bc:()=>tx(`At the start, Adam’s running averages m and v have not built up yet, so both are too small. Bias correction scales them back up, so the first steps are not distorted. At step t, the corrected averages are:<div class="fx">${math('\\hat{m} = \\frac{m}{1 - \\beta_1^t},\\quad \\hat{v} = \\frac{v}{1 - \\beta_2^t}')}</div>`,
           `刚开始时，Adam 的两个滑动平均 m 和 v 还没攒起来，都偏小。偏差校正（bias correction）把它们按比例放大回来，让最初几步不失真。第 t 步校正后的两个平均值是：<div class="fx">${math('\\hat{m} = \\frac{m}{1 - \\beta_1^t},\\quad \\hat{v} = \\frac{v}{1 - \\beta_2^t}')}</div>`),
 loss:()=>tx(`The loss here is ${math('f(x, y) = \\frac{1}{2}(x^2 + 25\\,y^2)')}: a long narrow valley whose bottom, with a loss of 0, is at x = 0, y = 0.`,
             `这里的 loss 是 ${math('f(x, y) = \\frac{1}{2}(x^2 + 25\\,y^2)')}：一条又长又窄的山谷，谷底在 x = 0、y = 0，那里 loss 为 0。`),
 param:()=>kt('param')||tx('A parameter is one number the model learns.','参数是模型要学的一个数。'),
 wd:()=>tx('Weight decay pulls every weight a little toward 0 at each step, so the weights stay small. It is a common way to keep a model from fitting noise in its training data.','weight decay 每一步把每个权重往 0 拉一点，让权重保持小。这是防止模型去拟合训练数据里噪声的常用办法。'),
 l2:()=>tx(`L2 regularization adds ${math('\\frac{1}{2} \\cdot \\mathrm{wd} \\cdot \\theta^2')} to the loss, so its pull toward 0, ${math('\\mathrm{wd} \\cdot \\theta')}, arrives as part of the gradient. wd is the weight decay and θ the weight.`,`L2 正则在 loss 里加上 ${math('\\frac{1}{2} \\cdot \\mathrm{wd} \\cdot \\theta^2')}，所以它往 0 的拉力 ${math('\\mathrm{wd} \\cdot \\theta')} 是作为梯度的一部分进来的。wd 是 weight decay，θ 是权重。`)
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'fewer',t:{en:'Fewer steps: it gets to the bottom faster',zh:'步数更少：更快到谷底'}},
  {v:'more',t:{en:'The same or more steps: it zig-zags harder',zh:'步数一样或更多：来回摆得更厉害'}},
  {v:'never',t:{en:'Never: the loss grows step after step',zh:'永远到不了：loss 一步比一步大'}}];
const PRED_RIGHT='never';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const isWD=()=>isP('adamw');
const lrCtl=(id,inline)=>A.ui.range({id,key:'lrP',label:tx('SGD learning rate','SGD 的学习率'),min:0,max:LR_PART.length-1,value:Math.max(0,LR_PART.indexOf(lrP)),out:fLR(lrP),inline});
const betaCtl=(id,inline)=>A.ui.range({id,key:'betaP',label:tx('Momentum β','momentum β'),min:0,max:BETAS.length-1,value:Math.max(0,BETAS.indexOf(betaP)),out:fB(betaP),inline});
const lrACtl=(id,inline)=>A.ui.range({id,key:'lrA',label:tx('Adam learning rate','Adam 的学习率'),min:0,max:LR_ADAM.length-1,value:Math.max(0,LR_ADAM.indexOf(lrA)),out:fLR(lrA),inline});
const BCL=[{v:'on',t:{en:'With bias correction (initial setup)',zh:'有偏差校正（初始设置）'},d:{en:'m and v are scaled up while they are still building up',zh:'m 和 v 还没攒够时，按比例放大'}},{v:'off',t:{en:'Without bias correction',zh:'没有偏差校正'},d:{en:'m and v are used as they are, too small at first',zh:'直接用 m 和 v，刚开始它们偏小'}}];
const bcRes=v=>{const r=run({opt:'adam',lr:lrA,bc:v==='on'}); return tx(`step 1: ${fB(Math.abs(r.d[1][0])/lrA)} × lr`,`第 1 步：lr 的 ${fB(Math.abs(r.d[1][0])/lrA)} 倍`);};
const bcCtl=()=>`<div id="op-ic-bc">`+A.ui.levers({aria:tx('Bias correction, pick one','偏差校正，选一个'),value:bcP,options:BCL.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(bcTried.has(o.v)?bcRes(o.v):'')}))})+`</div>`;
/* part 7: the share of a weight that weight decay removes per step, as a percentage with 2 significant digits */
const fPct=v=>`${(v*100).toPrecision(2)}%`;
const WDL=[{v:'l2',t:{en:'Adam + L2 (initial setup)',zh:'Adam + L2（初始设置）'},d:{en:'the pull toward 0 is added to the gradient, as PyTorch’s Adam does with weight_decay',zh:'往 0 的拉力加进梯度里，PyTorch 的 Adam 设了 weight_decay 就是这样'}},
  {v:'adamw',t:{en:'AdamW',zh:'AdamW'},d:{en:'apart from Adam’s step, every weight shrinks by the learning rate times the weight decay',zh:'在 Adam 那一步之外，每个权重再单独缩小一个比例：学习率乘以 weight decay'}}];
const wdRes=v=>v==='adamw'?tx(`both: ${fPct(wdPull('adamw'))} per step`,`两个都是每步 ${fPct(wdPull('adamw'))}`)
  :tx(`big gradients ${fPct(wdPull('l2',WD.big))}, small ${fPct(wdPull('l2',WD.small))}`,`梯度大的 ${fPct(wdPull('l2',WD.big))}，梯度小的 ${fPct(wdPull('l2',WD.small))}`);
const wdCtl=()=>`<div id="op-ic-wd">`+A.ui.levers({aria:tx('Weight decay, pick one','weight decay 的做法，选一个'),value:wdP,options:WDL.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(wdTried.has(o.v)?wdRes(o.v):'')}))})+`</div>`;
/* which runs the stage shows in each part */
function runsNow(){
  if(mode==='challenge') return [{c:chCfg(S.opt,S.lr),st:S.opt}];
  if(mode==='sandbox') return [{c:cfgOf(S),st:S.opt}];
  const id=STEPS[step].id;
  if(id==='mission') return [{c:SGD05,st:'sgd'},{c:MOM09,st:'mom'},{c:ADAM05,st:'adam'}];
  if(id==='predict') return [{c:SGD05,st:'sgd'}];
  if(id==='sgd') return [{c:{opt:'sgd',lr:lrP},st:'sgd'}];
  if(id==='mom') return [{c:SGD05,st:'sgd'},{c:{opt:'mom',lr:0.05,beta:betaP},st:'mom'}];
  if(id==='adam') return [{c:SGD05,st:'sgd'},{c:{opt:'adam',lr:lrA},st:'adam'}];
  if(id==='bias') return [{c:{opt:'adam',lr:lrA},st:'adam'}].concat(bcP==='off'?[{c:{opt:'adam',lr:lrA,bc:false},st:'nobc'}]:[]);
  if(id==='adamw') return [{c:ADAM05,st:'adam'}];   // not drawn: part 7 has its own picture (drawWD)
  return [{c:SGD05,st:'sgd'},{c:{opt:'mom',lr:0.05,beta:0.6},st:'mom'},{c:ADAM05,st:'adam'}];
}
const cfgOf=s=>{const c={opt:s.opt,lr:s.lr,surf:s.surf}; if(s.opt==='mom') c.beta=s.beta; if(s.opt==='adam'||s.opt==='adamw'){c.b2=s.b2; if(!s.bc) c.bc=false; if(s.b2===0.95) c.eps=1e-5;} if(s.opt==='adamw') c.wd=s.wd; if(s.warm) c.warm=s.warm; return c;};
/* the run's name with its settings in brackets, for use inside a sentence (R6.15) */
const runLabelP=x=>x.st==='nobc'?tx(`Adam (no bias correction, learning rate ${fLR(x.c.lr)})`,`Adam（无偏差校正，学习率 ${fLR(x.c.lr)}）`):tx(`${optName(x.c.opt)} (${x.c.opt==='mom'?`β ${fB(x.c.beta)}, `:''}learning rate ${fLR(x.c.lr)})`,`${optName(x.c.opt)}（${x.c.opt==='mom'?`β ${fB(x.c.beta)}，`:''}学习率 ${fLR(x.c.lr)}）`);
const runLabel=x=>x.st==='nobc'?tx('Adam, no bias correction','Adam，无偏差校正'):`${optName(x.c.opt)}${x.c.opt==='mom'?` β ${fB(x.c.beta)}`:''}${tx(', ','，')}lr ${fLR(x.c.lr)}`;
/* an optional “in numbers” box: the arithmetic stays out of the explanation (R6.27) */
const more=(en,zh,ben,bzh)=>`<details class="op-more"><summary>${tx(en,zh)}</summary>${tx(ben,bzh)}</details>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx(`One valley, one start, three optimizers: SGD needs ${run(SGD05).hit} steps, momentum ${run(MOM09).hit}, Adam ${run(ADAM05).hit}. What does each add?`,`同一条山谷、同一个出发点、三个优化器：SGD 要 ${run(SGD05).hit} 步，momentum 要 ${run(MOM09).hit} 步，Adam 要 ${run(ADAM05).hit} 步。后两个各多做了什么？`),
  body:()=>{const a=run(SGD05).hit,b=run(MOM09).hit,c=run(ADAM05).hit;
   return tx(`<p>A model with two ${DEF('parameters',TIP.param())}, x and y, starts at the ring marked start. Its ${DEF('loss',TIP.loss())} is a long narrow valley: steep across it (y), almost flat along it (x). Each optimizer follows the ${DEF('gradient',TIP.grad())} downhill until the loss is below 0.001.</p><p>${DEF('SGD',TIP.sgd())} at learning rate ${fLR(SGD05.lr)} needs ${a} steps: it drops into the valley within a few steps, then crawls along it to the bottom. ${DEF('Momentum',TIP.mom())} (β ${fB(MOM09.beta)}, the same learning rate) needs ${b}. ${DEF('Adam',TIP.adam())} needs ${c}, at learning rate ${fLR(ADAM05.lr)}. The two learning rates mean different things: SGD’s multiplies the gradient, while Adam’s is roughly how far each step goes, so the two numbers cannot be compared directly (part 5 has the details). What does each one add, and when is it worth it?</p>`,
             `<p>一个模型只有两个${DEF('参数',TIP.param())}：x 和 y。它从标着“出发点”的圆圈出发。它的 ${DEF('loss',TIP.loss())} 是一条又长又窄的山谷：横穿山谷（y 方向）很陡，沿着山谷（x 方向）几乎是平的。每个优化器都顺着${DEF('梯度',TIP.grad())}往下走，直到 loss 低于 0.001。</p><p>学习率为 ${fLR(SGD05.lr)} 的 ${DEF('SGD',TIP.sgd())} 要走 ${a} 步：它几步就落进山谷中间，然后沿着山谷慢慢往谷底爬。${DEF('momentum',TIP.mom())}（β ${fB(MOM09.beta)}，学习率也是 ${fLR(MOM09.lr)}）要 ${b} 步。${DEF('Adam',TIP.adam())} 用学习率 ${fLR(ADAM05.lr)}，要 ${c} 步。这两个学习率的意思不一样：SGD 的学习率是梯度要乘的倍数，Adam 的学习率大约是每一步走多远，所以这两个数不能直接比（第 5 部分细讲）。这两个改进各多做了什么？什么时候值得用？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: what happens to SGD if you double its learning rate?','先猜：把 SGD 的学习率翻倍，会怎样？'),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>SGD with ${DEF('learning rate',TIP.lr())} 0.05 reaches the bottom in ${run(SGD05).hit} steps. You double the learning rate to 0.10. What happens?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>${DEF('学习率',TIP.lr())}为 0.05 的 SGD 走 ${run(SGD05).hit} 步到谷底。现在把学习率翻倍，调到 0.10。会怎样？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'sgd',title:{en:'SGD: one learning rate for every direction',zh:'SGD：所有方向共用一个学习率'},
  lead:()=>tx('Drag the learning rate up to 0.10 and watch the path cross the valley.','把学习率拖到 0.10，看路径怎样横穿山谷。'),
  expect:()=>seenDiv,showMe:()=>setLrP(0.1),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>SGD’s step is the learning rate times the gradient, so the steep direction gets the big steps. At the start y is much closer to the bottom than x, yet SGD’s first step moves y ${fixd(firstRatio(),1)} times as far as x.</p>`,`<p>SGD 的一步是学习率乘以梯度，所以陡的方向迈的步子大。出发时 y 离谷底比 x 近得多，SGD 第一步在 y 方向走的距离却是 x 方向的 ${fixd(firstRatio(),1)} 倍。</p>`)+lrCtl('op-ic-lr',true),
  obs:()=>obsSgd()},
 {id:'mom',title:{en:'Momentum: the flat direction builds up speed',zh:'momentum：平的方向越走越快'},
  lead:()=>tx('Raise momentum β from 0 and watch the path speed up along the valley.','把 momentum 的 β 从 0 调高，看路径沿着山谷越走越快。'),
  expect:()=>betaMoved&&betaP>0,showMe:()=>setBeta(0.6),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>${T_('beta','Momentum')} keeps a running direction: each step adds the new gradient to what is left of the earlier ones. Along the valley every gradient points the same way, so the steps add up and build speed. Across the valley the gradient flips back and forth, so those pushes cancel.</p><p>β sets how much of the running direction each step keeps; β = 0 is plain SGD. The learning rate stays at 0.05.</p>`,
              `<p>${T_('beta','momentum')} 记着一个累积起来的方向：每一步都把新的梯度加到之前留下的方向上。沿着山谷，每一步的梯度都指向同一边，所以越加越大，路径越走越快。横穿山谷时，梯度一会儿朝这边、一会儿朝那边，这些推力互相抵消。</p><p>β 决定每一步保留多少之前的方向；β = 0 就是普通的 SGD。学习率保持 0.05。</p>`)+betaCtl('op-ic-beta',true),
  obs:()=>obsMom()},
 {id:'adam',title:{en:'Adam: each parameter gets its own step size',zh:'Adam：每个参数有自己的步长'},
  lead:()=>tx('Adam starts at SGD’s learning rate, 0.05: compare their steps, then raise Adam’s learning rate.','Adam 先用和 SGD 一样的学习率 0.05：比比两者每一步走多远，再把 Adam 的学习率调高。'),
  expect:()=>lrAMoved,showMe:()=>setLrA(0.5),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>${T_('lrA','Adam')} gives each parameter its own step size: it divides the gradient by the typical size of that parameter’s recent gradients. The steep y has big gradients and the flat x small ones, yet after the division both move about the same distance each step, close to the learning rate.</p><p>So the learning rate means something else. SGD’s step is the learning rate times the gradient; Adam’s step is usually about the learning rate, almost regardless of the gradient’s size. The bars in the chart show how far each step goes in x and in y.</p>`,
              `<p>${T_('lrA','Adam')} 给每个参数各自的步长：它把梯度除以这个参数最近梯度的典型大小。陡的 y 梯度大，平的 x 梯度小，可是除过之后，两个方向每一步走的距离差不多，都接近学习率。</p><p>所以学习率的意思变了。SGD 的一步是学习率乘以梯度；Adam 的一步通常约为一个学习率，基本不看梯度有多大。图里的横条是每一步在 x 和 y 方向各走了多远。</p>`)+lrACtl('op-ic-lrA',true),
  obs:()=>obsAdam()},
 {id:'bias',title:{en:'Bias correction keeps the first steps right',zh:'偏差校正：让最初几步不失真'},
  lead:()=>tx('Switch bias correction off and compare Adam’s first steps.','关掉偏差校正，比较 Adam 最初几步走多远。'),
  expect:()=>bcTried.has('off'),showMe:()=>setBc('off'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Adam’s running averages m and v start at 0, so at first they have not built up and are too small. ${T_('bc','Bias correction')} scales them back up, so the first steps are not distorted.</p><p>Pick with or without bias correction and see how many times lr the first step moves.</p>`,`<p>Adam 的两个滑动平均 m 和 v 都从 0 开始，所以刚开始还没攒起来，都偏小。${T_('bc','偏差校正')}把它们按比例放大回来，让最初几步不失真。</p><p>选有或没有偏差校正，看看第 1 步走了 lr 的几倍。</p>`)+bcCtl(),
  obs:()=>obsBias()},
 {id:'adamw',title:{en:'AdamW: weight decay on its own',zh:'AdamW：把 weight decay 单独做'},
  lead:()=>tx('Weight decay keeps weights small. Compare Adam + L2 with AdamW: which weights does it really shrink?','weight decay 让权重保持小。比较 Adam + L2 和 AdamW：它真正缩小了哪些权重？'),
  expect:()=>wdTried.has('adamw'),showMe:()=>setWd('adamw'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>${DEF('Weight decay',TIP.wd())} pulls every weight a little toward 0 at each step. The classic way adds that pull to the gradient, as ${DEF('L2',TIP.l2())}. Adam then divides it by the typical size of the weight’s recent gradients, like the rest of the gradient: a weight with big gradients barely shrinks, and a weight with small gradients shrinks a lot.</p><p>AdamW (${cite(SRC.lh,'Loshchilov and Hutter')}) keeps the pull out of the gradient: apart from Adam’s step, it shrinks every weight by the same fraction, lr × weight decay. The chart takes two weights of size 1.0 whose gradients flip sign every step, ±${WD.big} and ±${WD.small}, so the loss pulls them nowhere and only weight decay moves them.</p>`,
              `<p>${DEF('weight decay',TIP.wd())} 每一步把每个权重往 0 拉一点。传统的做法是把这股拉力加进梯度里，也就是 ${DEF('L2',TIP.l2())}。可是 Adam 会把梯度除以这个权重最近梯度的典型大小，拉力也跟着被除：梯度大的权重几乎不缩小，梯度小的权重却缩小得很多。</p><p>AdamW（${cite(SRC.lh,'Loshchilov 和 Hutter')}）不把拉力放进梯度。在 Adam 那一步之外，它单独把每个权重缩小同样的比例，也就是 lr × weight decay。图里取两个大小为 1.0 的权重，它们的梯度每一步都变号，分别是 ±${WD.big} 和 ±${WD.small}，所以 loss 不往任何方向拉它们，只有 weight decay 在动它们。</p>`)+wdCtl(),
  obs:()=>obsWd()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('SGD is capped by the steepest direction; momentum speeds up the flat one; Adam sizes each step; AdamW decays every weight alike.','SGD 被最陡的方向限住；momentum 让平的方向越走越快；Adam 给每个参数定步长；AdamW 让每个权重同样地衰减。'),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
/* SGD's first step on the valley: y's length over x's (25 × 1.5 ÷ 9, the same at every learning rate) */
const firstRatio=()=>{const d=run(SGD05).d[1]; return Math.abs(d[1])/Math.abs(d[0]);};
/* the factor y is multiplied by on every SGD step: 1 − 25 lr */
const yFac=lr=>1-25*lr;
function obsSgd(){
  const r=run({opt:'sgd',lr:lrP}), f=yFac(lrP), fs=sgn(fixd(f,3).replace(/\.?0+$/,'')||'0');
  if(lrP>0.08) return tx(`<p><b>At learning rate ${fLR(lrP)}, SGD never reaches the bottom: ${resLong(r)}.</b></p>`,`<p><b>学习率为 ${fLR(lrP)} 时，SGD 永远到不了谷底：${resLong(r)}。</b></p>`)+verdict()+
    tx(`<p>Why: across the valley each step is now so long that y jumps past the middle and lands farther away on the other side. So the path bounces higher up the walls with every step. On this valley SGD is safe only below 0.08, and it is the steepest direction that sets this limit for every parameter.</p>`,
       `<p>原因：横穿山谷的方向，现在每一步都迈得太大，y 跳过山谷中间，落到对面更远的地方。所以路径每一步都在谷壁上弹得更高。在这条山谷上，SGD 的学习率只有低于 0.08 才安全，这个上限由最陡的方向定下，对所有参数都一样。</p>`)+
    more('In numbers','用数字看',`<p>Each step multiplies y’s distance from the middle by 1 − 25 × ${fLR(lrP)} = ${fs}. The minus sign flips y to the other side; a size above 1 makes it grow. The limit 0.08 = 2 ÷ 25 is where this factor reaches −1.</p>`,`<p>每一步把 y 离山谷中间的距离乘以 1 − 25 × ${fLR(lrP)} = ${fs}。负号让 y 翻到另一边；绝对值大于 1，距离就越来越大。上限 0.08 = 2 ÷ 25，正是这个因子等于 −1 的地方。</p>`);
  if(Math.abs(lrP-0.08)<1e-9) return tx(`<p><b>At exactly 0.08, y lands exactly as far on the other side at every step: the path bounces between the two walls forever, and ${resLong(r)}.</b> Go one notch higher.</p>`,`<p><b>正好 0.08 时，y 每一步都落到对面同样远的地方：路径在两边谷壁之间永远来回弹，${resLong(r)}。</b>再往上调一格。</p>`);
  const why=lrP<=0.04?tx(`<p>Why: along the flat valley the gradient is small, so each step covers only ${fixd(lrP*100,1)}% of x’s remaining way to the bottom, and x crawls. A larger learning rate speeds it up, until the steep direction stops it.</p>`,`<p>原因：沿着平的山谷，梯度很小，每一步只走完 x 剩下路程的 ${fixd(lrP*100,1)}%，所以 x 走得很慢。学习率大一些会更快，直到陡的方向拦住它。</p>`)
    :tx(`<p>Why: across the valley y now jumps past the middle at every step, which is the zig-zag, but each jump is shorter than the last, so it settles. Along the valley x still crawls. The flat direction decides how many steps it takes; the steep one decides how large the learning rate may get.</p>`,`<p>原因：横穿山谷的方向，y 现在每一步都跳过山谷中间，这就是来回横跳；不过每一跳都比上一跳短，所以还能停下来。沿着山谷，x 还是走得很慢。要走多少步，由平的方向决定；学习率能调多大，由陡的方向决定。</p>`);
  return tx(`<p><b>At learning rate ${fLR(lrP)}, ${resLong(r)}.</b>${seenDiv?'':' Keep going up to 0.10 to check your prediction.'}</p>`,`<p><b>学习率为 ${fLR(lrP)} 时，${resLong(r)}。</b>${seenDiv?'':'继续调到 0.10，核对你的预测。'}</p>`)+(seenDiv?verdict():'')+why;
}
/* momentum at lr 0.05: the β with the fewest steps among the slider's stops */
const bestBeta=()=>{let b=null; BETAS.forEach(be=>{const h=run({opt:'mom',lr:0.05,beta:be}).hit; if(h!=null&&(b==null||h<b.h)) b={be,h};}); return b.be;};
/* per-step factors, shown with 2 decimals unless that hides the difference (R6.10) */
const pairF=(a,b)=>{let d=2; if(fixd(a,2)===fixd(b,2)) d=3; return [fixd(a,d),fixd(b,d)];};
function obsMom(){
  const r=run({opt:'mom',lr:0.05,beta:betaP}), s0=run(SGD05).hit, bb=bestBeta();
  if(betaP===0) return tx(`<p><b>With β = 0, momentum is plain SGD: ${resLong(r)}.</b> Raise β.</p>`,`<p><b>β = 0 时，momentum 就是普通的 SGD：${resLong(r)}。</b>把 β 调高。</p>`);
  const head=tx(`<p><b>With β = ${fB(betaP)}, ${resLong(r)}, against ${s0} steps for plain SGD at the same learning rate.</b></p>`,`<p><b>β = ${fB(betaP)} 时，${resLong(r)}；同样学习率的普通 SGD 要 ${s0} 步。</b></p>`);
  if(betaP>=0.85) return head+tx(`<p>Why: this much momentum keeps too much of the old direction. The path now overshoots the bottom along the valley as well as across it, and it swings back and forth before it settles. On this valley the fewest steps come near β ${fB(bb)}.</p>`,
    `<p>原因：动量这么大，旧的方向留得太多。路径不光横穿山谷时冲过中间，沿着山谷也冲过了谷底，要来回摆好几次才停下。在这条山谷上，β 在 ${fB(bb)} 附近步数最少。</p>`);
  const [mx,sx]=pairF(hbRate(1,0.05,betaP),hbRate(1,0.05,0)), slower=hbRate(25,0.05,betaP)>hbRate(25,0.05,0)+1e-9;
  const ySent=slower?tx('Across the valley the gradient flips at every step, and momentum carries y a little past the middle each time, so that swing takes longer to die out than with plain SGD.','横穿山谷时，梯度每一步都变向，momentum 每次都把 y 带过山谷中间一点，所以这种摆动比普通 SGD 消失得慢。')
    :tx('Across the valley the swing still dies out within a few steps, as with plain SGD.','横穿山谷时，摆动还是几步之内就消失了，和普通 SGD 一样。');
  const tail=betaP<bb-1e-9?tx(`A larger β speeds up x more but makes that swing last longer; near β ${fB(bb)} the two balance and the steps are fewest.`,`β 再大一些，x 会走得更快，但那种摆动也拖得更久；β 在 ${fB(bb)} 附近两者平衡，步数最少。`)
    :(Math.abs(betaP-bb)<1e-9?tx('At this β the two balance, so the steps are fewest.','在这个 β 下两者正好平衡，所以步数最少。')
    :tx(`Past β ${fB(bb)}, the swing across the valley is what holds the run back, and it gets slower as β grows.`,`β 超过 ${fB(bb)} 以后，拖慢整次运行的是横穿山谷的摆动，而且 β 越大越慢。`));
  const little=0.95-hbRate(1,0.05,betaP)<0.02;   // β 0.05: 0.947 against 0.950, only a little faster
  return head+tx(`<p>Why: along the valley every gradient points the same way, so momentum adds them up and x moves ${little?'a little':'much'} faster than with plain SGD. ${ySent} ${tail}</p>`,
    `<p>原因：沿着山谷，每一步的梯度都指向同一边，momentum 把它们加起来，所以 x 比普通 SGD 走得${little?'快一点':'快得多'}。${ySent}${tail}</p>`)+
    more('In numbers','用数字看',`<p>Late in the run, each step multiplies x’s distance from the bottom by ${mx}; plain SGD multiplies it by ${sx}.</p>`,`<p>到了运行后段，每一步把 x 离谷底的距离乘以 ${mx}；普通 SGD 是乘以 ${sx}。</p>`);
}
function obsAdam(){
  const r=run({opt:'adam',lr:lrA}), s=run(SGD05), d=r.d[1], g=s.d[1];
  const head=lrA===SGD05.lr?tx(`<p><b>At the same learning rate, 0.05, Adam’s first step moves x and y by ${fB(Math.abs(d[0]))} each; SGD’s moves x by ${fB(Math.abs(g[0]))} and y by ${fB(Math.abs(g[1]))}. With steps this short, ${resLong(r)}.</b></p>`,`<p><b>学习率同为 0.05 时，Adam 第一步在 x 和 y 方向各走 ${fB(Math.abs(d[0]))}；SGD 第一步在 x 方向走 ${fB(Math.abs(g[0]))}，在 y 方向走 ${fB(Math.abs(g[1]))}。Adam 的步子这么短，${resLong(r)}。</b></p>`)
    :tx(`<p><b>At learning rate ${fLR(lrA)}, Adam’s first step moves x and y by ${fB(Math.abs(d[0]))} each, and ${resLong(r)}. SGD at 0.05 needs ${s.hit} steps.</b></p>`,`<p><b>学习率为 ${fLR(lrA)} 时，Adam 第一步在 x 和 y 方向各走 ${fB(Math.abs(d[0]))}，${resLong(r)}。学习率 0.05 的 SGD 要 ${s.hit} 步。</b></p>`);
  return head+tx(`<p>Why Adam needs a larger learning rate on this valley: SGD’s step is the learning rate times the gradient, and the gradients here are big, several units. So at the same learning rate SGD moves much farther per step, and Adam needs about 0.5 to beat it.</p><p>In LLM training the two learning rates come out the other way round, for two reasons. Each weight’s gradient is tiny, often far below 0.001, so SGD would need a huge learning rate to move the weights at all. And Adam’s learning rate is roughly how far a weight moves per step; most weights are only about 0.02, so one step may move a weight only a small fraction of that, such as 0.0003. That is why Adam’s learning rates look much smaller than SGD’s in practice.</p>`,
    `<p>为什么在这条山谷上 Adam 要更大的学习率：SGD 的一步是学习率乘以梯度，而这里的梯度很大，有好几个单位。所以学习率相同时，SGD 每一步走得远得多，Adam 要把学习率调到 0.5 左右才比它快。</p><p>训练大模型时，两个学习率的大小正好反过来，原因有两个。一是每个权重的梯度都很小，常常远小于 0.001，所以 SGD 要用很大的学习率才推得动权重。二是 Adam 的学习率大约就是权重每一步走多远，而大多数权重只有 0.02 左右，一步只能挪动它的一小部分，比如 0.0003。所以实践中 Adam 的学习率看起来比 SGD 的小得多。</p>`);
}
/* the largest single step among steps 1–10, in learning rates: which step, which direction */
function biggest(r,lr){let b={t:1,i:0,k:0}; for(let t=1;t<=Math.min(10,r.n);t++) for(let i=0;i<2;i++){const k=Math.abs(r.d[t][i])/lr; if(k>b.k) b={t,i,k};} return b;}
function obsBias(){
  const on=run({opt:'adam',lr:lrA}), off=run({opt:'adam',lr:lrA,bc:false});
  if(bcP==='on') return tx(`<p><b>With bias correction, Adam’s first step is exactly the learning rate (${fB(Math.abs(on.d[1][0])/lrA)} × lr).</b> Switch it off to compare.</p>`,`<p><b>有偏差校正时，Adam 第一步正好走一个学习率（lr 的 ${fB(Math.abs(on.d[1][0])/lrA)} 倍）。</b>关掉它来比较。</p>`);
  const bg=biggest(off,lrA);
  return tx(`<p><b>Without bias correction, Adam’s first steps are several times too long: step 1 is ${fB(Math.abs(off.d[1][0])/lrA)} times the learning rate, and step ${bg.t} reaches ${fixd(bg.k,1)} times.</b> With it, step 1 is exactly the learning rate.</p>`,`<p><b>没有偏差校正时，Adam 最初几步迈得太大：第 1 步是学习率的 ${fB(Math.abs(off.d[1][0])/lrA)} 倍，第 ${bg.t} 步达到 ${fixd(bg.k,1)} 倍。</b>有校正时，第 1 步正好是一个学习率。</p>`)+
    tx(`<p>Why: Adam’s step is the average gradient divided by the gradient’s typical size, ${math('\\frac{m}{\\sqrt{v}}')}. At the start both averages are too small, but v averages over far more steps than m (about 1,000 against 10), so it is much further from its full size. Dividing by a number that is too small makes the step too long. Bias correction scales each average up by what it still lacks, so with it, step 1 is exactly one learning rate.</p>`,
       `<p>原因：Adam 的一步，是梯度的平均值除以梯度的典型大小，也就是 ${math('\\frac{m}{\\sqrt{v}}')}。刚开始这两个平均值都偏小，但 v 要平均的步数比 m 多得多（约 1,000 步对 10 步），所以离它该有的大小差得更远。除数偏小，步子就偏大。偏差校正按两者各自还差多少，把它们放大回来，所以有校正时，第 1 步就正好是一个学习率。</p>`)+
    more('In numbers: step 1','用数字看第 1 步',`<p>After one step, m holds 10% of the gradient and v 0.1% of its square, so ${math('\\sqrt{v}')} is ${fB(100*Math.sqrt(0.001))}% of the gradient’s size. Then ${math(`\\frac{m}{\\sqrt{v}} = \\frac{10\\%}{${fB(100*Math.sqrt(0.001))}\\%} = ${fB(0.1/Math.sqrt(0.001))}`)} instead of 1. ${J('scen','Warmup')} in real training also keeps the early steps small while v is still a poor estimate.</p>`,
      `<p>走完一步时，m 只攒到梯度的 10%，v 只攒到梯度平方的 0.1%，所以 ${math('\\sqrt{v}')} 只有梯度大小的 ${fB(100*Math.sqrt(0.001))}%。于是 ${math(`\\frac{m}{\\sqrt{v}} = \\frac{10\\%}{${fB(100*Math.sqrt(0.001))}\\%} = ${fB(0.1/Math.sqrt(0.001))}`)}，而不是 1。真实训练里的 ${J('scen','warmup')} 也是为了在 v 还估不准时，让前面的步子小一些。</p>`);
}
function obsWd(){
  const pb=fPct(wdPull('l2',WD.big)), ps=fPct(wdPull('l2',WD.small)), pw=fPct(wdPull('adamw')), ratio=Math.round(wdPull('l2',WD.small)/wdPull('l2',WD.big)/10)*10;
  if(wdP==='l2') return tx(`<p><b>With Adam + L2, weight decay shrinks the weight with big gradients by only ${pb} per step, and the weight with small gradients by ${ps}: about ${ratio} times as much.</b> Pick AdamW to compare.</p>`,
    `<p><b>用 Adam + L2 时，weight decay 每一步只把梯度大的权重缩小 ${pb}，却把梯度小的权重缩小 ${ps}，差了约 ${ratio} 倍。</b>选 AdamW 来比较。</p>`);
  return tx(`<p><b>With AdamW, weight decay shrinks both weights by ${pw} per step, whatever their gradients.</b> With Adam + L2 it was ${pb} and ${ps}.</p><p>Why: AdamW applies the decay on its own, outside the division by the gradients’ size, so every weight loses the same fraction: ${math('\\mathrm{lr} \\times \\text{weight decay}')}, here ${fLR(WD.lr)} × ${WD.wd} = ${pw}.</p>`,
    `<p><b>用 AdamW 时，weight decay 每一步把两个权重都缩小 ${pw}，和它们的梯度大小无关。</b>用 Adam + L2 时是 ${pb} 和 ${ps}。</p><p>原因：AdamW 单独做衰减，不经过“除以梯度大小”这一步，所以每个权重都缩小同样的比例，也就是 ${math('\\mathrm{lr} \\times \\text{weight decay}')}，这里是 ${fLR(WD.lr)} × ${WD.wd} = ${pw}。</p>`)+
    tx(`<p>AdamW is now the most common optimizer for training large language models, and Llama 2 was trained with it too (${cite(SRC.llama2,'Llama 2 paper')}); its settings are in scenario 1.</p>`,
       `<p>现在训练大模型最常用的就是 AdamW，Llama 2 也是用它训练的（${cite(SRC.llama2,'Llama 2 论文')}）；它的具体设置见应用场景 1。</p>`);
}
function remember(){
  const s=run(SGD05).hit, m6=run({opt:'mom',lr:0.05,beta:0.6}).hit, m9=run(MOM09).hit, a=run(ADAM05).hit, ratio=Math.round(wdPull('l2',WD.small)/wdPull('l2',WD.big)/10)*10;
  return tx(`<ul class="op-remlist"><li><b>SGD’s learning rate is capped by the steepest direction.</b> Too large a rate makes the path bounce higher up the steep walls at every step until it diverges. A rate small enough for the walls leaves the flat direction crawling, so plain SGD is slow on a long, narrow valley.</li>`+
    `<li><b>Momentum speeds up the flat direction.</b> Along the valley the gradient keeps pointing the same way, so the steps add up and build speed. Too much momentum overshoots: the path swings past the bottom and needs more steps, so β has to be tuned as well.</li>`+
    `<li><b>Adam gives each parameter its own step size.</b> It divides each gradient by that parameter’s typical gradient size, so at the start a steep direction and a flat one move by the same amount, one learning rate. So Adam’s learning rate is how far a step goes, not a multiplier of the gradient. Most weights of a real model are about 0.02 and a step should move them only a small fraction of that, so LLMs use small values such as 0.0003.</li>`+
    `<li><b>AdamW applies weight decay apart from the gradient step.</b> With Adam + L2, the pull toward 0 joins the gradient and is divided by Adam too, so weights with small gradients shrink much faster than weights with big ones. AdamW shrinks every weight by the same fraction. LLMs such as Llama 2 are trained with AdamW.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Adam always converges faster.” On this valley momentum with β 0.6 needed ${m6} steps and Adam ${a}. Adam pays off when gradient sizes differ a lot between parameters, or change during training, so that no single learning rate suits them all.</p><p>✗ “A smaller learning rate is always safer.” Below 0.08 SGD cannot diverge on this valley, but it can waste the run: at 0.01 its loss is still above 0.001 after ${N} steps.</p><p>✗ “Adam with weight decay is AdamW.” PyTorch’s Adam with <code>weight_decay</code> set adds the decay to the gradient as L2, so Adam divides it too. Only AdamW applies it on its own.</p></div>${A.shell.toScenarios()}`,
    `<ul class="op-remlist"><li><b>SGD 的学习率被最陡的方向限住。</b>学习率太大，路径会在陡的谷壁之间越弹越高，最后发散。学习率小到谷壁受得住时，沿着平的方向又走得很慢，所以在又长又窄的山谷里，普通 SGD 很慢。</li>`+
    `<li><b>momentum 让平的方向越走越快。</b>沿着山谷，梯度一直指向同一边，步子越叠越大。动量太大会冲过头，路径冲过谷底再折回来，反而要更多步，所以 β 也要调。</li>`+
    `<li><b>Adam 给每个参数各自的步长。</b>它把每个参数的梯度除以这个参数梯度的典型大小，所以一开始，陡的方向和平的方向走得一样远，都是一个学习率。因此 Adam 的学习率是一步走多远，不是梯度的倍数。真实模型的权重大多只有 0.02 左右，一步只该挪动它的一小部分，所以大模型的学习率是 0.0003 这样的小数。</li>`+
    `<li><b>AdamW 把 weight decay 和梯度那一步分开做。</b>用 Adam + L2 时，往 0 的拉力加进了梯度，也跟着被 Adam 除，所以梯度小的权重比梯度大的缩小得快得多。AdamW 让每个权重缩小同样的比例。Llama 2 这样的大模型用的就是 AdamW。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “Adam 总是收敛得更快。”在这条山谷上，β 0.6 的 momentum 只要 ${m6} 步，Adam 要 ${a} 步。参数之间梯度大小差得很多，或者梯度大小在训练中不断变化，一个学习率照顾不了所有参数时，Adam 才划算。</p><p>✗ “学习率越小越安全。”在这条山谷上，学习率低于 0.08 的 SGD 不会发散，但可能白白浪费训练：学习率 0.01 时，走了 ${N} 步 loss 还在 0.001 以上。</p><p>✗ “Adam 加上 weight decay 就是 AdamW。”PyTorch 的 Adam 设了 <code>weight_decay</code> 时，是把衰减当作 L2 加进梯度，它也会跟着被 Adam 除。只有 AdamW 才把它单独做。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 adam:'https://arxiv.org/abs/1412.6980', lh:'https://arxiv.org/abs/1711.05101', sut:'https://proceedings.mlr.press/v28/sutskever13.html',
 llama2:'https://arxiv.org/abs/2307.09288', resnet:'https://arxiv.org/abs/1512.03385', rosen:'https://doi.org/10.1093/comjnl/3.3.175',
 tsgd:'https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py', tadam:'https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py', tadamw:'https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py'
};
const PRESETS=[
 {id:'llama',t:{en:'Pretrain an LLM (Llama 2’s settings)',zh:'预训练大模型（Llama 2 的设置）'},d:{en:'Billions of parameters, noisy gradients, a long run',zh:'参数量以 B 计，梯度有噪声，训练很长'},s:{opt:'adamw',lr:0.5,b2:0.95,bc:true,wd:0.1,warm:20,surf:'v25',beta:0.9}},
 {id:'resnet',t:{en:'Train an image classifier (ResNet’s settings)',zh:'训练图像分类器（ResNet 的设置）'},d:{en:'A convolutional network from 2015, trained with SGD and momentum',zh:'2015 年的卷积网络，用 SGD 加 momentum 训练'},s:{opt:'mom',lr:0.02,beta:0.9,b2:0.999,bc:true,wd:0,warm:0,surf:'v25'}},
 {id:'adam',t:{en:'Train a new model without tuning (Adam’s defaults)',zh:'不调参训练新模型（Adam 的默认值）'},d:{en:'The paper’s and PyTorch’s defaults for a new model',zh:'论文和 PyTorch 给新模型的默认值'},s:{opt:'adam',lr:0.5,b2:0.999,bc:true,wd:0,warm:0,surf:'v25',beta:0.9}},
 {id:'curve',t:{en:'A curved valley',zh:'弯曲的山谷'},d:{en:'The valley bends, so the downhill direction keeps turning',zh:'山谷是弯的，下坡的方向一直在转'},s:{opt:'mom',lr:0.01,beta:0.9,b2:0.999,bc:true,wd:0,warm:0,surf:'curve'}}
];
const sameS=(a,b)=>a.opt===b.opt&&a.lr===b.lr&&a.surf===b.surf&&(a.opt!=='mom'||a.beta===b.beta)&&(!/adam/.test(a.opt)||(a.b2===b.b2&&a.bc===b.bc))&&(a.opt!=='adamw'||a.wd===b.wd)&&(a.warm||0)===(b.warm||0);
const presetNow=()=>PRESETS.findIndex(p=>sameS(p.s,S));
function scenHTML(i){
  const p=PRESETS[i], r=run(cfgOf({...S,...p.s}));
  if(p.id==='llama') return tx(`<p><b>Llama 2 was pretrained with AdamW. With its settings on this valley, a peak learning rate of 0.5 and 20 warmup steps, ${resLong(r)}.</b></p><p>Llama 2 used β1 0.9, β2 0.95, ε 10⁻⁵ and weight decay 0.1; its learning rate warmed up over 2,000 steps, then followed a cosine down to 10% of the peak. With β2 at 0.95 instead of 0.999, v forgets old gradients faster, so the step size keeps up when the gradients change. Warmup keeps the first steps small while v is still a poor estimate. Settings from the ${cite(SRC.llama2,'Llama 2 paper')}, section 2.2. Its peak learning rate was 0.0003 for the 7B model, far below the 0.5 used here. That is not because the gradients are smaller: Adam’s learning rate is roughly how far each step goes. A real model’s weights are mostly around 0.02, so one step may move a weight only a small fraction of that. This valley is about 9 units long, so here a step can be 0.5.</p>`,
                                `<p><b>Llama 2 用 AdamW 预训练。按它的设置在这条山谷上跑，峰值学习率取 0.5，warmup 20 步，${resLong(r)}。</b></p><p>Llama 2 的设置是 β1 0.9、β2 0.95、ε 10⁻⁵、weight decay 0.1；学习率先 warmup 2,000 步，再按余弦降到峰值的 10%。β2 取 0.95 而不是 0.999，v 忘掉旧梯度更快，梯度变了，步长也能跟上。warmup 让前几步走得小一些，因为那时 v 还估不准。设置来自 ${cite(SRC.llama2,'Llama 2 论文')}第 2.2 节。7B 模型的峰值学习率是 0.0003，远小于这里的 0.5。这不是因为梯度小：Adam 的学习率，大约就是每一步走多远。真实模型的权重大多只有 0.02 左右，一步只能挪动它很小的一部分；这条山谷长约 9 个单位，所以这里一步可以走 0.5。</p>`);
  if(p.id==='resnet') return tx(`<p><b>ResNet was trained with SGD and momentum 0.9 (learning rate 0.1, divided by 10 when the error stopped falling, weight decay 0.0001). Here, at learning rate 0.02, ${resLong(r)}.</b></p><p>Momentum 0.9 is the usual choice for SGD. On this short valley it overshoots a little. Real networks have noisy gradients and far more directions, and there averaging the gradients of about 10 steps pays off. Settings from ${cite(SRC.resnet,'He et al., 2016')}, section 3.4.</p>`,
                                 `<p><b>ResNet 用 SGD 加 momentum 0.9 训练（学习率 0.1，错误率不再下降时除以 10，weight decay 0.0001）。这里学习率取 0.02，${resLong(r)}。</b></p><p>momentum 0.9 是 SGD 的常用值。在这条短山谷上，它会稍微冲过头。真实网络的梯度有噪声，方向也多得多，把大约 10 步的梯度平均起来就很划算。设置来自 ${cite(SRC.resnet,'He 等，2016')} 第 3.4 节。</p>`);
  if(p.id==='adam') return tx(`<p><b>Adam’s defaults are β1 0.9, β2 0.999, ε 10⁻⁸ with bias correction, in the paper and in PyTorch. Here, at learning rate 0.5, ${resLong(r)}.</b></p><p>The default learning rate is 0.001: each parameter moves about 0.001 per step, which suits the weights of a real network. This valley spans about 9 units, so here it is 0.5. Defaults from ${cite(SRC.adam,'Kingma and Ba, 2015')}, Algorithm 1, and ${cite(SRC.tadam,'<code>torch/optim/adam.py</code>')}.</p>`,
                               `<p><b>Adam 的默认值是 β1 0.9、β2 0.999、ε 10⁻⁸，带偏差校正，论文和 PyTorch 都一样。这里学习率取 0.5，${resLong(r)}。</b></p><p>默认学习率是 0.001，也就是每个参数一步大约走 0.001，这适合真实网络的权重。这条山谷长约 9 个单位，所以这里用 0.5。默认值来自 ${cite(SRC.adam,'Kingma 和 Ba，2015')} 的算法 1，以及 ${cite(SRC.tadam,'<code>torch/optim/adam.py</code>')}。</p>`);
  const a=run({opt:'adam',lr:0.2,surf:'curve'});
  return tx(`<p><b>In a curved valley, momentum 0.9 at learning rate 0.01 swings out of the bend before it settles: ${resLong(r)}. Adam at 0.2 takes ${resShort(a)}.</b></p><p>The downhill direction keeps turning, while the running direction momentum keeps still points the old way, so the path overshoots the bend and climbs the outer wall. The loss is a gentler version of ${cite(SRC.rosen,'Rosenbrock’s function')}: ${math('f = (1 - x)^2 + 10\\,(y - x^2)^2')}, with its bottom at (1, 1).</p>`,
            `<p><b>在弯曲的山谷里，momentum 0.9（学习率 0.01）会先冲出弯道，再停下来：${resLong(r)}。学习率 0.2 的 Adam 要 ${resShort(a)}。</b></p><p>下坡的方向一直在转，momentum 记着的方向却还朝着原来那边，所以路径冲过弯道，爬上外侧的谷壁。这个 loss 是 ${cite(SRC.rosen,'Rosenbrock 函数')}的温和版：${math('f = (1 - x)^2 + 10\\,(y - x^2)^2')}，谷底在 (1, 1)。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chSet=(o,lr)=>tx(`${optName(o)}${o==='mom'?' 0.9':''}, lr ${fLR(lr)}`,`${optName(o)}${o==='mom'?' 0.9':''}，lr ${fLR(lr)}`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.opt,S.lr), best=chBest(), c=chChanges(S.opt,S.lr);
  let h;
  const good=tx(`<p class="okline op-mt"><b>Goal met: ${chSet(S.opt,S.lr)} gets the loss below 0.001 in ${me.steps} steps.</b>`,`<p class="okline op-mt"><b>达成目标：${chSet(S.opt,S.lr)} 在 ${me.steps} 步内把 loss 降到 0.001 以下。</b>`);
  if(me.ok&&S.opt===best.o&&S.lr===best.lr) h=good+tx(' And that is the smallest change: you switched only the optimizer and kept the learning rate. Of the optimizers that pass, momentum keeps the fewest extra numbers: one per parameter.</p>','而且这是改动最少的做法：只换了优化器，学习率不用动。在能达标的优化器里，momentum 只给每个参数多存一个值，额外内存最少。</p>');
  else if(me.ok&&c===2) h=good+`</p>`+tx(`<p>But one change is enough: keep the learning rate at ${fLR(CH.start.lr)} and switch only to ${chSet(best.o,best.lr)} (${best.steps} steps).</p>`,`<p>不过改一处就够了：学习率保持 ${fLR(CH.start.lr)}，只换成 ${chSet(best.o,best.lr)}（${best.steps} 步）。</p>`);
  else if(me.ok) h=good+`</p>`+tx(`<p>But Adam keeps two extra numbers per parameter: ${chSet(best.o,best.lr)} also passes with one (${best.steps} steps).</p>`,`<p>不过 Adam 要给每个参数多存两个数值：${chSet(best.o,best.lr)} 只多存一个也能达标（${best.steps} 步）。</p>`);
  else h=tx(`<p class="badline op-mt"><b>Not yet: with ${chSet(S.opt,S.lr)}, ${resLong(me.r)}.</b></p>`,`<p class="badline op-mt"><b>还没达到：${chSet(S.opt,S.lr)} 时，${resLong(me.r)}。</b></p>`)+
      tx(`<p class="small">Hint: this valley is 4 times steeper across than the lab’s. How large may SGD’s learning rate be now, and how fast does x shrink at that rate? Which optimizer moves x faster at the same learning rate?</p>`,`<p class="small">提示：这条山谷横穿方向比 Lab 里的陡 4 倍。SGD 的学习率现在最大能取多少？在这个学习率下 x 缩得有多快？哪个优化器能在同样的学习率下让 x 走得更快？</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const rows=[]; CH.opts.forEach(o=>CH.lrs.forEach(lr=>rows.push(chEval(o,lr))));
  /* the owner's review of batch 8 (2026-10-07): brief, so the chart keeps the settings that meet the goal, the pick and the initial
     setup, and a note under it counts the rest */
  const isMe=e=>e.o===S.opt&&e.lr===S.lr, isStart=e=>e.o===CH.start.opt&&e.lr===CH.start.lr;
  h+=A.ui.cmpChart({title:tx('Steps until the loss is below 0.001 on the steeper valley (goal: 100 or fewer)','在更陡的山谷上，把 loss 降到 0.001 以下要几步（目标：不超过 100 步）'),
    aria:tx('Steps until the loss is below 0.001: ','把 loss 降到 0.001 以下的步数：')+rows.filter(e=>e.ok||isMe(e)||isStart(e)).map(e=>pairTxt(chSet(e.o,e.lr),resShort(e.r))).join(tx('; ','；')),brief:true,
    rows:rows.map(e=>({label:(e.ok?'✓ ':'')+chSet(e.o,e.lr),value:e.steps!=null?e.steps:N,text:resShort(e.r),ok:e.ok,me:e.o===S.opt&&e.lr===S.lr,
      tags:(e.o===CH.start.opt&&e.lr===CH.start.lr?[{text:tx('initial setup','初始设置')}]:[]).concat(e.o===S.opt&&e.lr===S.lr?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:CH.max,goalLabel:tx('goal: 100 steps','目标：100 步'),max:N*1.9,okText:tx('✓ meets the goal','✓ 达到目标'),noText:tx('misses it','没达到'),goalText:tx('goal: 100 steps','目标：100 步')});
  const st=chEval(CH.start.opt,CH.start.lr);
  h+=tx(`<p class="small">In the initial setup, SGD at learning rate 0.01, ${resLong(st.r)}. SGD cannot pass at any learning rate. Above 0.02 (2 ÷ 100) it diverges. At 0.02 or less, x shrinks by at most 2% per step, too slow for 100 steps.</p>`,`<p class="small">初始设置是学习率 0.01 的 SGD，${resLong(st.r)}。SGD 在这里怎么调都不行。学习率超过 0.02（2 ÷ 100）就发散。不超过 0.02 时，x 每步最多只缩小 2%，100 步走不到谷底。</p>`);
  h+=tx(`<p class="small">Why momentum works: along the valley every gradient points the same way, so momentum adds them up and x speeds up well past SGD’s pace. Across the valley it carries y past the middle, so the path swings from wall to wall on its way to the bottom. Adam passes too, at learning rates from 0.1 up, but its learning rate has to be tuned again, and it keeps two numbers per parameter (m and v), one more than momentum.</p>`,`<p class="small">momentum 为什么行：沿着山谷，每一步的梯度都指向同一边，momentum 把它们叠加起来，x 越走越快，远远超过 SGD 的速度。横穿山谷时，它会把 y 带过山谷中间，所以路径在两边谷壁之间来回摆，一路走到谷底。Adam 在学习率 0.1 及以上也能达标，但学习率要另调，而且每个参数要存 m 和 v 两个值，比 momentum 多一个。</p>`);
  h+=tx('<p class="small muted">Note: this valley has no noise and only two parameters. On a real model, check the optimizer and the learning rate on a short run of your own training before a long one.</p>','<p class="small muted">注意：这条山谷没有噪声，也只有两个参数。真实模型上，先用自己训练的短跑检查优化器和学习率，再开长跑。</p>');
  return h;
}

/* =====================================================================
   header copy, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${({7:'Seven',8:'Eight'})[STEPS.length]||STEPS.length} short parts. Predict first, then raise SGD’s learning rate, add momentum, see how Adam sizes each parameter’s step, and why LLMs use AdamW. Start here if this is new to you.`,`一共 ${STEPS.length} 个部分。先预测，再调高 SGD 的学习率，加上 momentum，然后看 Adam 怎样给每个参数定步长，最后看大模型为什么用 AdamW。第一次学从这里开始。`),
  sandbox:tx('Three real settings (Llama 2, ResNet, Adam’s defaults) on the same valley, plus a curved valley. Pick one, then change anything under Settings.','三种真实设置（Llama 2、ResNet、Adam 的默认值）都在同一条山谷上跑，另加一条弯曲的山谷。选一个，再在自由调参里随意改。'),
  challenge:tx('A valley 4 times steeper across than the lab’s. Pick the optimizer and the learning rate so that the loss falls below 0.001 within 100 steps.','一条横穿方向比 Lab 里的陡 4 倍的山谷。选好优化器和学习率，让 loss 在 100 步内降到 0.001 以下。')
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> every path on this page is the optimizer’s real update rule, run step by step on a loss with two parameters; ${J('math','“Show the math”')} gives the formulas.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>SGD, momentum in the heavy-ball form of PyTorch’s SGD (dampening 0, not Nesterov), Adam with bias correction that can be switched off, and AdamW with decoupled weight decay, as in PyTorch.</li>`+
 `<li>The loss ${VAL(25)} from (−9, 1.5). A run reaches the bottom at the first step with a loss below 0.001, diverges once the loss passes 100M, and stops at 200 steps.</li>`+
 `<li>Part 7: one weight of 1.0 whose gradient flips sign every step, ±${WD.big} or ±${WD.small}, so only weight decay moves it on average; lr ${fLR(WD.lr)}, weight decay ${WD.wd}. The bars give the share removed per step once m and v have settled (Show the math has the formula).</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul><li>Two parameters instead of billions, and exact gradients instead of noisy ones from a batch of data. Noise is a main reason real training uses momentum and Adam, so the step counts here only compare the optimizers on this valley.</li>`+
 `<li>The learning rates suit this valley (it spans about 9 units); Llama 2 7B, a real network, used a peak of 0.0003.</li><li>Weight decay pulls every parameter toward 0, where this valley’s bottom also is, so here it does not move the bottom.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>Adam: ${cite(SRC.adam,'Kingma and Ba, ICLR 2015')}. AdamW: ${cite(SRC.lh,'Loshchilov and Hutter, ICLR 2019')}. Momentum: ${cite(SRC.sut,'Sutskever et al., ICML 2013')}.</li>`+
 `<li>Code: PyTorch ${cite(SRC.tsgd,'<code>torch/optim/sgd.py</code>')}, ${cite(SRC.tadam,'<code>adam.py</code>')}, ${cite(SRC.tadamw,'<code>adamw.py</code>')}. Settings in Scenarios: ${cite(SRC.llama2,'Llama 2')}, ${cite(SRC.resnet,'ResNet')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>这一页上的每条路径，都是优化器真实的更新规则在一个两参数的 loss 上一步步跑出来的；${J('math','“展开公式”')}里有公式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>SGD；PyTorch 的 SGD 用的 heavy-ball 形式的 momentum（dampening 为 0，不是 Nesterov）；可以关掉偏差校正的 Adam；以及和 PyTorch 一样做解耦 weight decay 的 AdamW。</li>`+
 `<li>loss 是 ${VAL(25)}，从 (−9, 1.5) 出发。loss 第一次低于 0.001 的那一步算到达谷底；loss 超过 100M 算发散；最多跑 200 步。</li>`+
 `<li>第 7 部分：一个大小为 1.0 的权重，梯度每一步都变号（±${WD.big} 或 ±${WD.small}），所以平均下来只有 weight decay 在动它；lr ${fLR(WD.lr)}，weight decay ${WD.wd}。横条是 m 和 v 稳定之后每一步缩小的比例（公式在“展开公式”里）。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul><li>只有两个参数，而真实模型的参数量以 B 计；梯度是精确的，而不是从一批数据算出、带噪声的。噪声正是真实训练要用 momentum 和 Adam 的主要原因之一，所以这里的步数只用来比较优化器在这条山谷上的表现。</li>`+
 `<li>学习率是按这条山谷（长约 9 个单位）取的；真实网络 Llama 2 7B 的峰值是 0.0003。</li><li>weight decay 把每个参数往 0 拉，这条山谷的谷底也在 0，所以这里它不会挪动谷底。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>Adam：${cite(SRC.adam,'Kingma 和 Ba，ICLR 2015')}。AdamW：${cite(SRC.lh,'Loshchilov 和 Hutter，ICLR 2019')}。momentum：${cite(SRC.sut,'Sutskever 等，ICML 2013')}。</li>`+
 `<li>代码：PyTorch 的 ${cite(SRC.tsgd,'<code>torch/optim/sgd.py</code>')}、${cite(SRC.tadam,'<code>adam.py</code>')}、${cite(SRC.tadamw,'<code>adamw.py</code>')}。应用场景里的设置：${cite(SRC.llama2,'Llama 2')}、${cite(SRC.resnet,'ResNet')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>Training changes every parameter a little at a time to lower the loss. Backprop gives the gradient: for every parameter, how much the loss rises when that parameter grows a little. The optimizer decides how far to move each parameter against its gradient.</p><p>The lab ${labLink('backprop')} shows how the gradient is computed. This lab starts from the gradient.</p>`,
                    `<p>训练每次把每个参数改一点，让 loss 降下来。反向传播给出梯度：每个参数增加一点时，loss 涨多少。优化器决定每个参数逆着梯度走多远。</p><p>${labLink('backprop')}这个 Lab 讲梯度怎么算。这个 Lab 从梯度开始。</p>`);
const OUTCOME=()=>tx('predict when SGD diverges from the learning rate, explain what momentum and Adam add and when each pays off, and why LLMs train with AdamW.','由学习率预测 SGD 什么时候发散，讲清 momentum 和 Adam 各多做了什么、什么时候划算，以及大模型为什么用 AdamW 训练。');
const EXITS=()=>{
  const a1=run({opt:'adam',lr:0.5}), a10=run({opt:'adam',lr:0.5,scale:10});
  return {core:()=>chSubmitted,
 explain:[
  {q:tx('On the valley, SGD at learning rate 0.10 never reaches the bottom. Why?','在这条山谷上，学习率 0.10 的 SGD 永远到不了谷底。为什么？'),
   o:[tx('Across the valley each step multiplies y by 1 − 25 × 0.10 = −1.5, so y flips to the other wall and grows every step.','横穿山谷的方向，每一步把 y 乘以 1 − 25 × 0.10 = −1.5，所以 y 翻到另一边谷壁，而且每一步都变大。'),
      tx('Along the valley each step moves x by 0.10 times its gradient, which is too far, so x jumps past the bottom and lands further away.','沿着山谷的方向，每一步让 x 走梯度的 0.10 倍，走得太远，所以 x 跳过谷底，落得更远。'),
      tx('A larger learning rate adds more noise to every step, so the path wanders around the valley and never settles at the bottom.','学习率越大，每一步的噪声越大，所以路径在山谷里乱走，停不到谷底。')],a:0,
   n:[null,tx('Along the valley x is multiplied by 1 − 0.10 = 0.9 each step: it still shrinks smoothly. The trouble is the steep direction.','沿山谷方向，x 每一步乘以 1 − 0.10 = 0.9：它还在平稳地缩小。出问题的是陡的方向。'),
     tx('This lab’s gradients are exact, with no noise at all. The path still blows up, because of the steep direction’s curvature.','这个 Lab 的梯度是精确的，完全没有噪声。路径照样发散，原因是陡的方向的曲率。')],
   why:tx(`SGD multiplies the gap to the bottom in each direction by ${math('1 - \\mathrm{lr} \\cdot \\lambda')}, where λ is that direction’s curvature. With curvature 25 that factor drops below −1 once lr passes 2 ÷ 25 = 0.08, and the gap grows.`,`SGD 在每个方向上，把离谷底的距离乘以 ${math('1 - \\mathrm{lr} \\cdot \\lambda')}，λ 是这个方向的曲率。曲率为 25 时，lr 超过 2 ÷ 25 = 0.08，这个因子就小于 −1，距离越来越大。`)},
  {q:tx('Without bias correction, why are Adam’s first steps several times larger than the learning rate?','没有偏差校正时，Adam 的最初几步为什么比学习率大好几倍？'),
   o:[tx(`v starts at 0 and averages over about 1,000 steps, so early on ${math('\\sqrt{v}')} is far too small, and the step, ${math('\\frac{m}{\\sqrt{v}}')}, comes out too large.`,`v 从 0 开始，平均的是大约 1,000 步的梯度平方，所以早期 ${math('\\sqrt{v}')} 偏小很多，步子 ${math('\\frac{m}{\\sqrt{v}}')} 就偏大。`),
      tx('m starts at 0 and averages over about 10 steps, so early on m is too large and pushes each step past the learning rate.','m 从 0 开始，平均的是大约 10 步的梯度，所以早期 m 偏大，把每一步推得超过学习率。'),
      tx('Bias correction is a warmup: without it the learning rate starts at its peak instead of rising from near 0.','偏差校正就是 warmup：没有它，学习率从峰值开始，而不是从接近 0 慢慢升上来。')],a:0,
   n:[null,tx('Starting from 0 makes m too small, not too large. Alone, that would make the first steps smaller; it is v’s shortfall that wins.','从 0 开始会让 m 偏小，而不是偏大。单看它，最初几步应该更小；是 v 偏小得更多占了上风。'),
     tx('Bias correction does not change the learning rate: it scales m and v back up. Warmup is a different technique: it changes the learning rate on a schedule.','偏差校正不改学习率，它放大的是 m 和 v。warmup 是另一种做法，按计划调学习率。')],
   why:tx(`v averages the squared gradient over about 1,000 steps; m averages the gradient over only about 10. So at the start v is much further from its full size than m is. A step is ${math('\\frac{m}{\\sqrt{v}}')}, and a divisor that is too small makes the step too long. Bias correction scales m and v back up to their full size, so step 1 is exactly one learning rate. The numbers are in part 6, under “In numbers: step 1”.`,`v 平均的是梯度的平方，要平均大约 1,000 步；m 平均的是梯度，只平均大约 10 步。所以刚开始，v 离它该有的大小比 m 远得多。一步是 ${math('\\frac{m}{\\sqrt{v}}')}，除数偏小，步子就偏大。偏差校正把 m 和 v 各自放大回该有的大小，所以第 1 步正好是一个学习率。具体数字在第 6 部分的“用数字看第 1 步”里。`)}],
 transfer:[
  {q:tx(`Another valley is 50 times steeper across than along: ${VAL(50)}. On this lab’s valley (25 times) SGD diverged above 0.08. Above about which learning rate does SGD diverge now?`,`另一条山谷横穿方向比沿山谷陡 50 倍：${VAL(50)}。在这个 Lab 的山谷（陡 25 倍）上，SGD 超过 0.08 就发散。现在大约超过多少就发散？`),
   o:[tx('About 0.02','约 0.02'),tx('About 0.04','约 0.04'),tx('About 0.08','约 0.08'),tx('About 0.16','约 0.16')],a:1,
   why:tx(`The limit is ${math('2 / \\lambda')}, with λ the steepest curvature: 2 ÷ 50 = 0.04, half of 2 ÷ 25 = 0.08. Twice the steepness halves the safe learning rate, and the flat direction then crawls twice as slowly.`,`上限是 ${math('2 / \\lambda')}，λ 是最陡的曲率：2 ÷ 50 = 0.04，是 2 ÷ 25 = 0.08 的一半。陡一倍，安全的学习率就减半，平的方向也就慢一倍。`)},
  {q:tx(`Adam at learning rate 0.5 on this lab’s valley moves x by ${fB(Math.abs(a1.d[1][0]))} in its first step. Now the loss is multiplied by 10, so every gradient is 10 times larger. About how far does x move in the first step?`,`学习率 0.5 的 Adam 在这个 Lab 的山谷上，第一步让 x 走 ${fB(Math.abs(a1.d[1][0]))}。现在把 loss 乘以 10，每个梯度都变成 10 倍。第一步 x 大约走多远？`),
   o:[tx(`About ${fB(Math.abs(a1.d[1][0]))}, the same`,`约 ${fB(Math.abs(a1.d[1][0]))}，不变`),tx(`About ${fixd(Math.abs(a1.d[1][0])*Math.sqrt(10),1)}, 3.16 times`,`约 ${fixd(Math.abs(a1.d[1][0])*Math.sqrt(10),1)}，是 3.16 倍`),tx(`About ${fixd(Math.abs(a1.d[1][0])*10,1)}, 10 times`,`约 ${fixd(Math.abs(a1.d[1][0])*10,1)}，是 10 倍`)],a:0,
   why:tx(`m grows 10 times and ${math('\\sqrt{v}')} grows 10 times, so ${math('\\frac{m}{\\sqrt{v}}')} does not change: the first step is still ${fB(Math.abs(a10.d[1][0]))}. SGD’s step would be 10 times larger and would diverge at 0.05.`,`m 变成 10 倍，${math('\\sqrt{v}')} 也变成 10 倍，所以 ${math('\\frac{m}{\\sqrt{v}}')} 不变：第一步还是 ${fB(Math.abs(a10.d[1][0]))}。SGD 的步子会变成 10 倍，在 0.05 就发散。`)}],
 practice:[
  {id:'ml-fundamentals',d:tx(`Questions 6 to 8: derive why Adam divides m by ${math('1 - \\beta_1^t')} and why its first step is about lr in every coordinate, why L2 in Adam differs from AdamW, and why warmup helps Adam.`,`第 6 到 8 题：推导 Adam 为什么把 m 除以 ${math('1 - \\beta_1^t')}、为什么第一步在每个坐标上都约为 lr，Adam 里的 L2 为什么和 AdamW 不同，以及 warmup 为什么对 Adam 有用。`)},
  {id:'training-loop-debug',d:tx('Part 1: a training loop with momentum SGD hides six bugs. One scales the gradient so the effective learning rate is 32 times too large, as if the learning rate were set far too high.','第 1 部分：一个用 momentum SGD 的训练循环藏了六个 bug。其中一个把梯度放大，有效学习率大了 32 倍，相当于学习率设得太大。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Optimizers','优化器'),title:l?L(l.q):tx('What does Adam do that SGD does not?','Adam 比 SGD 多做了什么？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: the gradient','前置知识：梯度'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
/* the player under the valley (R3.9): the paths and the loss curves play on their own when the chart comes into view; play / pause /
   replay and the speed come first, the step slider with ◂ ▸ is the second way to look at one step */
const SPDL={slow:{en:'Slow',zh:'慢'},med:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
function playerHTML(){
  return `<div class="op-player" id="op-player"><div class="op-prow"><button type="button" class="btn sm op-play" data-act="play" aria-pressed="false" data-tipx="${esc(tx('Play the paths at the chosen speed; press again to pause. Once they have reached the end, the button plays them again from step 1.','按选定的速度播放路径；再按一次暂停。走到最后一步后，这个按钮从第 1 步重新播放。'))}">${tx('▶ Play','▶ 播放')}</button>`+
   `<span class="op-spd" tabindex="0" data-tipx="${esc(tx(`A whole replay takes about ${SPEEDS.slow} s at Slow, ${SPEEDS.med} s at Medium and ${SPEEDS.fast} s at Fast.`,`整段回放：慢速约 ${SPEEDS.slow} 秒，中速约 ${SPEEDS.med} 秒，快速约 ${SPEEDS.fast} 秒。`))}">`+
   `<span class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="spd">${Object.keys(SPEEDS).map(v=>`<button type="button" data-v="${v}" aria-pressed="${spd===v}">${esc(L(SPDL[v]))}</button>`).join('')}</span></span></div>`+
   `<div class="op-scrub"><button type="button" class="btn ghost sm" data-act="k-" aria-label="${esc(tx('One step back','后退一步'))}" data-tipx="${esc(tx('Shows the step before, and pauses.','显示前一步，并暂停播放。'))}">◂</button>`+
   `<input type="range" id="op-k" min="1" max="${N}" step="1" value="${K}" aria-label="${esc(tx('Step shown','显示第几步'))}"><button type="button" class="btn ghost sm" data-act="k+" aria-label="${esc(tx('One step forward','前进一步'))}" data-tipx="${esc(tx('Shows the next step, and pauses.','显示下一步，并暂停播放。'))}">▸</button><output class="mono" id="op-k-out"></output></div></div>`;
}
function stageHTML(){
  return `<div class="panel stage op-stage" id="op-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="op-st-title"></div><div class="meter-sub" id="op-st-sub"></div></div><span class="status" id="op-status" tabindex="0"></span></div>`+
   `<div class="legend-row" id="op-legend"></div>`+
   `<div class="stage-graph graph op-graph" id="op-graph">`+
    `<div class="op-cap" id="op-map-cap"></div><svg class="op-map" id="op-map" role="img" xmlns="http://www.w3.org/2000/svg"></svg>`+
    playerHTML()+
    `<div class="op-cap" id="op-bars-cap"></div><div class="op-bars" id="op-bars"></div>`+
    `<div class="op-cap" id="op-loss-cap"></div><svg class="op-loss" id="op-loss" role="img" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="op-wd" id="op-wd"></div>`+
   `</div>`+
   `<div class="metrics op-metrics" id="op-metrics"></div>`+
   `<details class="math" id="op-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="op-mathbody"></div></details></div>`;
}
const segRow=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">`+A.ui.seg({id,key,label,options:opts,value:val,note})+`</div>`;
function controlsHTML(){
  return `<aside class="panel controls" id="op-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="op-lock" hidden></p>`+
   segRow('op-ctl-opt','opt',tx('Optimizer','优化器'),[['sgd','SGD'],['mom','Momentum'],['adam','Adam'],['adamw','AdamW']],S.opt,tx('SGD keeps no extra numbers per parameter, momentum one, Adam and AdamW two.','SGD 不给参数多存数值，momentum 多存一个，Adam 和 AdamW 多存两个。'))+
   `<div class="lockable" id="op-ctl-lr-wrap">`+A.ui.range({id:'op-ctl-lr',key:'lr',label:tx('Learning rate','学习率'),min:0,max:LR_ALL.length-1,value:Math.max(0,LR_ALL.indexOf(S.lr)),out:fLR(S.lr),note:tx('SGD and momentum multiply the gradient by it; for Adam and AdamW it is the typical size of each parameter’s step.','SGD 和 momentum 用它乘以梯度；对 Adam 和 AdamW，它是每个参数一步的典型大小。')})+`</div>`+
   `<div class="lockable" id="op-ctl-beta-wrap">`+A.ui.range({id:'op-ctl-beta',key:'beta',label:tx('Momentum β','momentum β'),min:0,max:BETAS.length-1,value:Math.max(0,BETAS.indexOf(S.beta)),out:fB(S.beta),note:tx('How much of the running sum each step keeps. 0.9 is the usual choice: it averages over about 10 steps.','每一步保留多少累加的速度。常用 0.9：大约平均 10 步的梯度。')})+`</div>`+
   segRow('op-ctl-b2','b2',tx('Adam β2','Adam 的 β2'),[['0.95','0.95'],['0.999','0.999']],String(S.b2),tx('How long v remembers: 0.999 averages over about 1,000 steps, 0.95 over about 20. Large LLM runs often use 0.95; ε is then 10⁻⁵ as in Llama 2, else 10⁻⁸.','v 记多久：0.999 大约平均 1,000 步，0.95 大约 20 步。大模型训练常用 0.95；此时 ε 取 10⁻⁵（和 Llama 2 一样），否则取 10⁻⁸。'))+
   segRow('op-ctl-bc','bc',tx('Bias correction','偏差校正'),[['on',tx('on','开')],['off',tx('off','关')]],S.bc?'on':'off',tx('On in every real Adam; switching it off shows why it exists.','真实的 Adam 都开着；关掉它是为了看它为什么存在。'))+
   segRow('op-ctl-wd','wd',tx('Weight decay (AdamW)','weight decay（AdamW）'),[['0','0'],['0.1','0.1']],String(S.wd),tx('Each step shrinks every parameter by lr × this. Llama 2 used 0.1. Here the bottom is at 0, so it does not move the bottom.','每一步把每个参数缩小 lr × 这个值。Llama 2 用 0.1。这里谷底在 0，所以它不会挪动谷底。'))+
   segRow('op-ctl-warm','warm',tx('Warmup, then cosine decay','warmup，然后余弦衰减'),[['0',tx('off','关')],['20',tx('20 steps','20 步')]],String(S.warm),tx('Raises the learning rate linearly over the first steps, then lowers it along a cosine to 10% by step 200. Llama 2 warmed up for 2,000 steps.','前几步把学习率线性升上去，然后按余弦降到第 200 步时的 10%。Llama 2 warmup 了 2,000 步。'))+
   segRow('op-ctl-surf','surf',tx('Loss','loss'),[['v25',tx('narrow valley','窄山谷')],['curve',tx('curved valley','弯山谷')]],S.surf,tx(`The narrow valley is ${VAL(25)}. The curved one is ${ROS()}: a gentler Rosenbrock function.`,`窄山谷是 ${VAL(25)}。弯山谷是 ${ROS()}：温和版的 Rosenbrock 函数。`))+
   `<div class="lockable" id="op-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np
A, B = 1.0, 25.0                      # loss = (A x**2 + B y**2) / 2
grad = lambda p: np.array([A, B]) * p
loss = lambda p: 0.5 * (A * p[0]**2 + B * p[1]**2)

def train(step, n=200):
    p, s = np.array([-9.0, 1.5]), {}  # start; s holds m, v, ...
    for t in range(1, n + 1):
        p = p - step(grad(p), s, t)
        if loss(p) < 1e-3: return t   # steps to the bottom
        if loss(p) > 1e8: return "diverges"
    return "not in %d steps" % n

def sgd(lr):
    return lambda g, s, t: lr * g

def momentum(lr, beta):              # heavy ball, as torch SGD
    def step(g, s, t):
        s["b"] = beta * s.get("b", 0) + g
        return lr * s["b"]
    return step

def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def step(g, s, t):
        s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
        s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g
        m, v = s["m"] / (1 - b1**t), s["v"] / (1 - b2**t)
        return lr * m / (np.sqrt(v) + eps)
    return step

print(train(sgd(0.05)), train(sgd(0.10)))
print(train(momentum(0.05, 0.6)), train(adam(0.5)))
# 104 diverges
# 30 74`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="op-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="op-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s three update rules on its valley, counting the steps to a loss below 0.001.','这个 Lab 的三条更新规则，在它的山谷上数到 loss 低于 0.001 要几步。')}</p><pre class="code" id="op-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="op-prod">`+
   tx(`<li>PyTorch’s ${cite(SRC.tsgd,'<code>torch/optim/sgd.py</code>')}: with <code>momentum</code> set, it keeps a buffer per parameter, updated by <code>buf.mul_(momentum).add_(grad, alpha=1 - dampening)</code> (dampening is 0 by default), and steps by the learning rate times <code>buf</code>; <code>nesterov=True</code> switches to Nesterov’s form.</li>`+
      `<li>${cite(SRC.tadam,'<code>torch/optim/adam.py</code>')} keeps <code>exp_avg</code> (m) and <code>exp_avg_sq</code> (v) per parameter, with defaults <code>betas=(0.9, 0.999)</code> and <code>eps=1e-8</code>, and divides by <code>bias_correction1 = 1 - beta1**step</code> and its β2 twin for bias correction. Its <code>weight_decay</code> adds the decay to the gradient (L2).</li>`+
      `<li>${cite(SRC.tadamw,'<code>torch/optim/adamw.py</code>')}: AdamW is Adam with <code>decoupled_weight_decay=True</code> and <code>weight_decay=1e-2</code> by default. The step then starts with <code>param.mul_(1 - lr * weight_decay)</code>, outside m and v.</li>`,
      `<li>PyTorch 的 ${cite(SRC.tsgd,'<code>torch/optim/sgd.py</code>')}：设了 <code>momentum</code> 时，为每个参数存一个缓冲，用 <code>buf.mul_(momentum).add_(grad, alpha=1 - dampening)</code> 更新（dampening 默认为 0），每步走学习率乘以 <code>buf</code>；<code>nesterov=True</code> 换成 Nesterov 的形式。</li>`+
      `<li>${cite(SRC.tadam,'<code>torch/optim/adam.py</code>')} 为每个参数存 <code>exp_avg</code>（m）和 <code>exp_avg_sq</code>（v），默认 <code>betas=(0.9, 0.999)</code>、<code>eps=1e-8</code>，并除以 <code>bias_correction1 = 1 - beta1**step</code> 和对应的 β2 项做偏差校正。它的 <code>weight_decay</code> 把衰减项加进梯度（L2）。</li>`+
      `<li>${cite(SRC.tadamw,'<code>torch/optim/adamw.py</code>')}：AdamW 就是设了 <code>decoupled_weight_decay=True</code> 的 Adam，默认 <code>weight_decay=1e-2</code>。每一步先做 <code>param.mul_(1 - lr * weight_decay)</code>，不经过 m 和 v。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.adam,'Adam: A Method for Stochastic Optimization')}<span>${tx('Kingma and Ba, ICLR 2015: the running means m and v, bias correction (section 3), and the defaults β1 0.9, β2 0.999, ε 10⁻⁸','Kingma 和 Ba，ICLR 2015：滑动平均 m 和 v、偏差校正（第 3 节），以及默认值 β1 0.9、β2 0.999、ε 10⁻⁸')}</span></li>`+
   `<li>${cite(SRC.lh,'Decoupled Weight Decay Regularization')}<span>${tx('Loshchilov and Hutter, ICLR 2019: in Adam, L2 regularization and weight decay are not the same; AdamW decays the weights outside m and v','Loshchilov 和 Hutter，ICLR 2019：在 Adam 里，L2 正则和 weight decay 不是一回事；AdamW 在 m 和 v 之外衰减权重')}</span></li>`+
   `<li>${cite(SRC.sut,'On the importance of initialization and momentum in deep learning')}<span>${tx('Sutskever, Martens, Dahl and Hinton, ICML 2013: momentum, and Nesterov’s form of it, for training deep networks','Sutskever、Martens、Dahl、Hinton，ICML 2013：用 momentum 及其 Nesterov 形式训练深度网络')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="op-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="op-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="op-app m2" id="op-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="op-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how the optimizer’s path reaches the bottom of the valley.','选一个场景，看看优化器沿什么路径走到谷底。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="op-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Reach the bottom of a steeper valley in 100 steps','100 步内走到更陡的山谷谷底')}</h3>`+
   `<p>${tx(`A new valley, ${VAL(100)}, is 4 times steeper across than the lab’s, and the run starts at (−5, 0.8). The initial setup is SGD at learning rate 0.01. Choose the optimizer and the learning rate so that the loss falls below 0.001 within 100 steps:`,`一条新山谷 ${VAL(100)}，横穿方向比 Lab 里的陡 4 倍，从 (−5, 0.8) 出发。初始设置是学习率 0.01 的 SGD。选好优化器和学习率，让 loss 在 100 步内降到 0.001 以下：`)}</p>`+
   A.ui.seg({id:'op-ic-chO',key:'chO',label:tx('Optimizer','优化器'),options:CH.opts.map(o=>[o,o==='mom'?tx('Momentum 0.9','Momentum 0.9'):optName(o)]),value:S.opt,inline:true})+
   A.ui.seg({id:'op-ic-chL',key:'chL',label:tx('Learning rate','学习率'),options:CH.lrs.map(v=>[String(v),fLR(v)]),value:String(S.lr),inline:true})+
   `<div class="goal mono op-chgoal" id="op-ch-g"><span>${tx('Steps to a loss below 0.001','loss 降到 0.001 以下的步数')}</span><span><b id="op-ch-n"></b>${tx(' (goal: 100 or fewer)','（目标：不超过 100）')}</span></div>`+
   submit+`<div id="op-ch-res" class="op-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},sgd:{en:'Drag the learning rate above 0.08 to continue.',zh:'把学习率拖到 0.08 以上才能继续。'},
    mom:{en:'Raise β above 0 to continue.',zh:'把 β 调到 0 以上才能继续。'},adam:{en:'Change Adam’s learning rate to continue.',zh:'改一下 Adam 的学习率才能继续。'},bias:{en:'Switch bias correction off to continue.',zh:'关掉偏差校正才能继续。'},adamw:{en:'Pick AdamW to continue.',zh:'选 AdamW 才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#op-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail); if(nav) nav.insertAdjacentHTML('afterend','<p class="op-why small muted" hidden></p>');}
  A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#op-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]||(s.id==='sgd'&&lrP!==0.05)||(s.id==='mom'&&betaMoved)||(s.id==='adam'&&lrAMoved)||s.id==='bias'||s.id==='adamw')&&(!s.gate||s.gate());
    const lv=s.id==='adamw'?{p:wdP,t:wdTried,r:wdRes}:{p:bcP,t:bcTried,r:bcRes};
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv.p)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lv.t.has(v)?lv.r(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.op-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#op-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.opt,S.lr), nEl=$('#op-ch-n',rail); if(!nEl) return;
    nEl.textContent=resShort(e.r);
    const g=$('#op-ch-g',rail); g.classList.toggle('met',e.ok); g.classList.toggle('op-over',!e.ok);
    $$('.seg-ctl[data-key="chO"]',rail).forEach(x=>A.ui.setPressed(x,S.opt)); $$('.seg-ctl[data-key="chL"]',rail).forEach(x=>A.ui.setPressed(x,String(S.lr)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#op-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing: the valley with the paths, the step bars, the loss chart
   ===================================================================== */
const STY={sgd:{dash:'',w:2.2,cls:'op-s0'},mom:{dash:'7 4',w:2.2,cls:'op-s1'},adam:{dash:'1.5 3.5',w:2.6,cls:'op-s2'},adamw:{dash:'1.5 3.5',w:2.6,cls:'op-s2'},nobc:{dash:'9 3 2 3',w:2,cls:'op-s3'}};
const surfNow=()=>mode==='challenge'?'v100':(mode==='sandbox'?S.surf:'v25');
/* contour lines by marching squares, cached per surface */
const CONT={};
function contours(id){
  if(CONT[id]) return CONT[id];
  const s=SURF[id], [x0,x1,y0,y1]=s.view, nx=80, ny=40, out=[];
  const X=i=>x0+(x1-x0)*i/nx, Y=j=>y0+(y1-y0)*j/ny, F=[];
  for(let j=0;j<=ny;j++){F.push([]); for(let i=0;i<=nx;i++) F[j].push(s.f([X(i),Y(j)]));}
  s.levels.forEach(lv=>{const segs=[];
    for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){
      const c=[[X(i),Y(j),F[j][i]],[X(i+1),Y(j),F[j][i+1]],[X(i+1),Y(j+1),F[j+1][i+1]],[X(i),Y(j+1),F[j+1][i]]], pts=[];
      for(let k=0;k<4;k++){const a=c[k], b=c[(k+1)%4]; if((a[2]<lv)!==(b[2]<lv)){const t=(lv-a[2])/(b[2]-a[2]); pts.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}}
      if(pts.length>=2) segs.push([pts[0],pts[1]]); if(pts.length===4) segs.push([pts[2],pts[3]]);}
    out.push({lv,segs});});
  CONT[id]=out; return out;
}
function mapGeom(id){const [x0,x1,y0,y1]=SURF[id].view, W=600, H=Math.min(300,Math.round(W*(y1-y0)/(x1-x0))); return {W,H,X:x=>(x-x0)/(x1-x0)*W,Y:y=>H-(y-y0)/(y1-y0)*H};}
const fx=v=>(Math.round(v*10)/10).toFixed(1);
const clampP=v=>Math.max(-1e4,Math.min(1e4,v));
/* R5.9: names sit on the charts beside what they name: each path on the map, each loss line at its own end, the goal line on itself */
const shortName=(x,runs)=>x.st==='nobc'?tx('no correction','无校正'):(runs.some(q=>q.st==='nobc')&&x.c.opt==='adam'?tx('with correction','有校正'):optName(x.c.opt));
const textW=(t,fs)=>[...String(t)].reduce((a,ch)=>a+(/[\u2e80-\uffff]/.test(ch)?1:0.62),0)*fs;
const fsOf=(svg,sel,d)=>{const e=svg.querySelector(sel), v=e?parseFloat(getComputedStyle(e).fontSize):NaN; return v>0?v:d;};
/* a number set in optimizers.css (a CSS variable without a unit), or d when it is not set */
const cssNum=(el,v,d)=>{const x=el?parseFloat(getComputedStyle(el).getPropertyValue(v)):NaN; return x>0?x:d;};
/* each path's name goes beside the first point of its path with room around it: away from the other paths, the start and bottom
   marks and the names already placed; it appears once the path gets there (cached per set of runs, language and font size) */
function mapLabels(svg,g,runs,s){
  const fs=fsOf(svg,'.op-tl',13), key=JSON.stringify([runs.map(x=>[x.c,x.st]),surfNow(),A.lang(),fs]);
  if(svg._lk===key) return svg._lab;
  const P=runs.map(x=>{const r=run(x.c); return r.pts.slice(0,endOf(r)+1).map(q=>[g.X(clampP(q[0])),g.Y(clampP(q[1]))]);});
  const sx=g.X(s.start[0]), sy=g.Y(s.start[1]), mx=g.X(s.min[0]), my=g.Y(s.min[1]);
  const box=(x,y,w)=>({x0:x,x1:x+w,y0:y-fs*0.8,y1:y+fs*0.2});
  const busy=[box(sx+8,sy-8,textW(tx('start','出发点'),fs)),box(mx+8,my+16,textW(tx('bottom','谷底'),fs)),{x0:sx-7,x1:sx+7,y0:sy-7,y1:sy+7},{x0:mx-7,x1:mx+7,y0:my-7,y1:my+7}];
  const dP=(b,q)=>Math.hypot(Math.max(b.x0-q[0],0,q[0]-b.x1),Math.max(b.y0-q[1],0,q[1]-b.y1));
  const dB=(a,b)=>Math.hypot(Math.max(a.x0-b.x1,0,b.x0-a.x1),Math.max(a.y0-b.y1,0,b.y0-a.y1));
  const out=[];
  runs.forEach((x,i)=>{const nm=shortName(x,runs), w=textW(nm,fs); let best=null;
    for(let t=1;t<P[i].length;t++){const px=P[i][t][0], py=P[i][t][1]; if(px<0||px>g.W||py<0||py>g.H) continue;
      for(const [dx,dy] of [[7,-7],[7,7+fs*0.8],[-7-w,-7],[-7-w,7+fs*0.8]]){const b=box(px+dx,py+dy,w);
        if(b.x0<2||b.x1>g.W-2||b.y0<2||b.y1>g.H-2) continue;
        let d=Infinity;
        P.forEach((pp,j)=>{for(let u=0;u<pp.length;u++){if(j===i&&Math.abs(u-t)<=1) continue; const v=dP(b,pp[u]); if(v<d) d=v;}});
        busy.concat(out.map(o=>o.b)).forEach(o=>{const v=dB(b,o); if(v<d) d=v;});
        const sc=Math.min(d,fs*1.5);
        if(!best||sc>best.sc+1e-6) best={sc,t,b,y:py+dy};}}
    if(best) out.push({nm,t:best.t,b:best.b,x:best.b.x0,y:best.y});});
  svg._lk=key; svg._lab=out; return out;
}
/* the valley: the contours are drawn once per surface and language; each frame redraws only the paths up to step K (R3.9) */
function drawMap(){
  const id=surfNow(), g=mapGeom(id), svg=$('#op-map',app), runs=runsNow(), s=SURF[id], key=id+'|'+A.lang();
  if(svg._key!==key){svg._key=key;
    let h=`<defs><clipPath id="op-clip"><rect x="0" y="0" width="${g.W}" height="${g.H}"/></clipPath><marker id="op-ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L8 4L0 8z" class="op-ahd"/></marker></defs>`;
    h+=`<rect x="0" y="0" width="${g.W}" height="${g.H}" class="op-bg"/>`;
    contours(id).forEach((c,i)=>{h+=`<path class="op-ct${i===0?' in':''}" d="${c.segs.map(q=>`M${fx(g.X(q[0][0]))} ${fx(g.Y(q[0][1]))}L${fx(g.X(q[1][0]))} ${fx(g.Y(q[1][1]))}`).join('')}"/>`;});
    h+=`<g clip-path="url(#op-clip)" id="op-dyn"></g><circle class="op-start" cx="${fx(g.X(s.start[0]))}" cy="${fx(g.Y(s.start[1]))}" r="5"/><text class="op-tl" x="${fx(g.X(s.start[0])+8)}" y="${fx(g.Y(s.start[1])-8)}">${esc(tx('start','出发点'))}</text>`+
      `<path class="op-min" d="M${fx(g.X(s.min[0])-5)} ${fx(g.Y(s.min[1])-5)}l10 10M${fx(g.X(s.min[0])-5)} ${fx(g.Y(s.min[1])+5)}l10 -10"/><text class="op-tl" x="${fx(g.X(s.min[0])+8)}" y="${fx(g.Y(s.min[1])+16)}">${esc(tx('bottom','谷底'))}</text><g id="op-mlab"></g>`;
    svg.setAttribute('viewBox',`0 0 ${g.W} ${g.H}`); svg.innerHTML=h; svg._lk=null;
    const tip=tx('Contour lines of the loss: each closed line joins points of equal loss, closer lines mean a steeper wall. Each optimizer’s path starts at the ring marked start and grows step by step; the arrow is the step shown under the chart.','loss 的等高线：每条线连起 loss 相同的点，线越密，谷壁越陡。每个优化器的路径从标着“出发点”的圆圈出发，一步一步画出来；箭头是图下显示的那一步。');
    svg.dataset.tipx=tip; svg.setAttribute('aria-label',tip); svg.setAttribute('tabindex','0');
    const mcap=$('#op-map-cap',app), mch=(id==='curve'?tx(`The curved valley ${ROS()}, seen from above: each line is one loss level`,`从上往下看弯山谷 ${ROS()}：每条线是一个 loss 高度`)
      :(id==='v100'?tx(`The steeper valley ${VAL(100)}, seen from above: each line is one loss level`,`从上往下看更陡的山谷 ${VAL(100)}：每条线是一个 loss 高度`):tx(`The valley ${VAL(25)}, seen from above: each line is one loss level`,`从上往下看山谷 ${VAL(25)}：每条线是一个 loss 高度`)))+tx('; the arrow is the step shown below','；箭头是图下显示的那一步'); if(mcap._h!==mch){mcap.innerHTML=mch; mcap._h=mch;}}
  let h='';
  runs.forEach(x=>{const r=run(x.c), st=STY[x.st]||STY.sgd, end=endOf(r), p=r.pts.slice(0,Math.min(K,end)+1);
    h+=`<polyline class="op-path ${st.cls}" points="${p.map(q=>`${fx(g.X(clampP(q[0])))},${fx(g.Y(clampP(q[1])))}`).join(' ')}" style="stroke-width:${st.w}${st.dash?`;stroke-dasharray:${st.dash}`:''}"/>`;
    if(K<=end){const a=r.pts[K-1], b=r.pts[K]; h+=`<line class="op-arrow" x1="${fx(g.X(clampP(a[0])))}" y1="${fx(g.Y(clampP(a[1])))}" x2="${fx(g.X(clampP(b[0])))}" y2="${fx(g.Y(clampP(b[1])))}" marker-end="url(#op-ah)"/><circle class="op-dot" cx="${fx(g.X(clampP(b[0])))}" cy="${fx(g.Y(clampP(b[1])))}" r="3.2"/>`;}});
  $('#op-dyn',svg).innerHTML=h;
  const mg=$('#op-mlab',svg); if(mg){const lh=mapLabels(svg,g,runs,s).filter(o=>K>=o.t).map(o=>`<text class="op-tl op-pl" x="${fx(o.x)}" y="${fx(o.y)}">${esc(o.nm)}</text>`).join(''); if(mg._h!==lh){mg.innerHTML=lh; mg._h=lh;}}
}
/* the step chosen under the chart: its size in x and in y for every run (R6.25: per-parameter step sizes) */
function drawBars(){
  const runs=runsNow(), rows=runs.map(x=>{const r=run(x.c); return {x,r,d:K<=endOf(r)?r.d[K]:null};});
  const mx=Math.max(1e-9,...rows.filter(q=>q.d).map(q=>Math.max(Math.abs(q.d[0]),Math.abs(q.d[1])))), w=v=>(Math.min(1,Math.abs(v)/mx)*100).toFixed(2)+'%';
  $('#op-bars-cap',app).textContent=surfNow()==='curve'?tx(`How far step ${K} goes in x and in y`,`第 ${K} 步在 x 和 y 方向各走了多远`):tx(`How far step ${K} goes in each direction: x is along the valley, y across it`,`第 ${K} 步在每个方向上走了多远：x 沿着山谷，y 横穿山谷`);
  const v=q=>Math.abs(q)>=100?tx('over 100','超过 100'):fixd(Math.abs(q),Math.abs(q)<0.01?4:(Math.abs(q)<1?3:2));
  $('#op-bars',app).innerHTML=rows.map(q=>{const st=STY[q.x.st]||STY.sgd, lab=esc(runLabel(q.x));
    if(!q.d){const tip=q.r.hit!=null?tx(`${runLabel(q.x)} reached the bottom at step ${q.r.hit}: its loss fell below 0.001, so it takes no more steps.`,`${runLabel(q.x)} 在第 ${q.r.hit} 步到了谷底：loss 降到了 0.001 以下，之后不再走。`)
        :tx(`${runLabel(q.x)} diverged at step ${q.r.n}: its loss passed 100M.`,`${runLabel(q.x)} 在第 ${q.r.n} 步发散了：loss 超过了 100M。`);
      return `<div class="op-brow" tabindex="0" data-tipx="${esc(tip)}"><span class="op-bl">${lab}</span><span class="op-bnone small muted">${esc(q.r.hit!=null?tx(`reached the bottom at step ${q.r.hit}`,`第 ${q.r.hit} 步到了谷底`):tx(`diverged at step ${q.r.n}`,`第 ${q.r.n} 步就发散了`))}</span></div>`;}
    const tip=tx(`${runLabel(q.x)}, step ${K}: x moved by ${v(q.d[0])} and y by ${v(q.d[1])}.`,`${runLabel(q.x)}，第 ${K} 步：x 走了 ${v(q.d[0])}，y 走了 ${v(q.d[1])}。`);
    return `<div class="op-brow" tabindex="0" data-tipx="${esc(tip)}"><span class="op-bl">${lab}</span><span class="op-bk mono">x</span><span class="op-btr"><i class="op-b ${st.cls}" style="width:${w(q.d[0])}"></i></span><span class="op-bv mono">${v(q.d[0])}</span>`+
      `<span></span><span class="op-bk mono">y</span><span class="op-btr"><i class="op-b ${st.cls}" style="width:${w(q.d[1])}"></i></span><span class="op-bv mono">${v(q.d[1])}</span></div>`;}).join('');
}
function drawLoss(){
  /* polish round 5: on a wide stage optimizers.css puts the chart in a column beside the step sizes and gives it a height (--op-lh); it is
     then drawn for that column's width at that height, one unit to a pixel, so its labels keep their size; elsewhere it stays 600 × 150,
     scaled to the stage's width as before */
  const svg=$('#op-loss',app), lh=cssNum(svg,'--op-lh',0), cw=lh?Math.round(svg.getBoundingClientRect().width):0, fit=cw>=240;
  const runs=runsNow(), W=fit?cw:600, H=fit?lh:150, top=1e3, bot=1e-5, X=t=>t/N*(W-74)+58, Y=v=>{const l=Math.log10(Math.max(bot,Math.min(top,v))); return 8+(Math.log10(top)-l)/(Math.log10(top)-Math.log10(bot))*(H-26);};
  /* the top label's baseline sits at least one font size below the top, so it stays inside the chart (H7, kit_text_bounds) */
  const fs0=fsOf(svg,'.op-ax',fit?13:(matchMedia('(max-width:560px)').matches?16:11));
  let h=`<rect x="58" y="8" width="${W-74}" height="${H-26}" class="op-bg"/>`;
  [1e3,1,1e-3].forEach(v=>{h+=`<text class="op-ax" x="54" y="${fx(Math.max(Y(v)+3,fs0+1))}" text-anchor="end">${v===1e3?'1,000':(v===1?'1':'0.001')}</text>`;});
  [0,50,100,150,200].forEach(t=>{h+=`<text class="op-ax" x="${fx(X(t))}" y="${H-4}" text-anchor="middle">${t}</text>`;});
  h+=`<line class="op-goal" x1="58" x2="${W-16}" y1="${fx(Y(GOAL))}" y2="${fx(Y(GOAL))}"/>`;
  runs.forEach(x=>{const r=run(x.c), st=STY[x.st]||STY.sgd, pts=[], end=Math.min(K,endOf(r)); for(let t=0;t<=end;t++) pts.push(`${fx(X(t))},${fx(Y(r.loss[t]))}`);
    h+=`<polyline class="op-lline ${st.cls}" points="${pts.join(' ')}" style="stroke-width:${st.w-0.4}${st.dash?`;stroke-dasharray:${st.dash}`:''}"/>`;});
  h+=`<line class="op-kline" x1="${fx(X(K))}" x2="${fx(X(K))}" y1="8" y2="${H-18}"/>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.innerHTML=h;
  const fs=fsOf(svg,'.op-ax',11), PL=58, PR=W-16, yG=Y(GOAL);
  const gT=tx('goal 0.001','目标 0.001'), gW=textW(gT,fs), gY=yG-fs*0.3-2; let lab='';
  const LL=runs.map(x=>{const r=run(x.c), e=Math.min(K,endOf(r)); return {nm:shortName(x,runs),r,xe:X(e),ye:Y(r.loss[e]),div:r.div!=null&&K>=r.div,hit:r.hit!=null&&K>=r.hit};});
  let ty=8+fs+2;   // a diverged run: its name and “↑ diverges” at the top, right of where it diverged
  LL.filter(o=>o.div).forEach(o=>{const t=`${o.nm} ${tx('↑ diverges','↑ 发散')}`, w=textW(t,fs); lab+=`<text class="op-ax op-dv op-ll" x="${fx(Math.max(PL+4,Math.min(PR-4-w,X(o.r.div)+4)))}" y="${fx(ty)}">${esc(t)}</text>`; ty+=fs+3;});
  /* a run that got below the goal: its name just under the goal line, from its line's end to the right */
  const lo=LL.filter(o=>o.hit&&!o.div).map(o=>({o,w:textW(o.nm,fs),x:o.xe+3})).sort((a,b)=>a.x-b.x);
  for(let i=1;i<lo.length;i++) lo[i].x=Math.max(lo[i].x,lo[i-1].x+lo[i-1].w+fs*0.6);
  const ox=lo.length?lo[lo.length-1].x+lo[lo.length-1].w-(PR-3):0; if(ox>0) lo.forEach(q=>{q.x-=ox;});
  lo.forEach(q=>{lab+=`<text class="op-ax op-ll" x="${fx(q.x)}" y="${fx(yG+fs*0.8+2)}">${esc(q.o.nm)}</text>`;});
  /* a run still above the goal: its name beside its line's end (left of it at the right edge), kept above the goal line */
  const yMin=8+fs, yMax=yG-fs*0.3-2, hi=LL.filter(o=>!o.hit&&!o.div).map(o=>{const w=textW(o.nm,fs); let x=o.xe+4, y=o.ye+fs*0.35; if(x+w>PR-3){x=o.xe-4-w; y=o.ye+fs+2;} return {o,w,x,y:Math.max(yMin,Math.min(yMax,y))};}).sort((a,b)=>a.y-b.y);
  for(let i=1;i<hi.length;i++) if(hi[i].y-hi[i-1].y<fs*1.25) hi[i].y=hi[i-1].y+fs*1.25;
  const oy=hi.length?hi[hi.length-1].y-yMax:0; if(oy>0) hi.forEach(q=>{q.y-=oy;});
  /* the goal's name sits on its line, at the left or else the right end, wherever no line's name is; if both are taken, those names move up */
  const hit=(x0,y)=>hi.filter(q=>q.x<x0+gW+4&&x0<q.x+q.w+4&&Math.abs(q.y-y)<fs*1.25);
  let gx=PL+4; if(hit(gx,gY).length&&!hit(PR-4-gW,gY).length) gx=PR-4-gW; else hit(gx,gY).forEach(q=>{q.y=Math.max(yMin,gY-fs*1.25);});
  lab+=`<text class="op-ax op-gl" x="${fx(gx)}" y="${fx(gY)}">${esc(gT)}</text>`;
  hi.forEach(q=>{lab+=`<text class="op-ax op-ll" x="${fx(q.x)}" y="${fx(q.y)}">${esc(q.o.nm)}</text>`;});
  svg.insertAdjacentHTML('beforeend',lab);
  const tip=tx('The loss after each step, on a log scale: each grid label is 1,000 times the one below. The dashed line is the goal, 0.001; a run’s line ends at the step that gets below it, and its name sits at that end. The thin vertical line is the step shown under the valley.','每一步之后的 loss，对数刻度：每个刻度是下一个的 1,000 倍。虚线是目标 0.001；每条线停在降到它以下的那一步，名字就标在线的末端。细竖线是山谷图下显示的那一步。');
  svg.dataset.tipx=tip; svg.setAttribute('aria-label',tip); svg.setAttribute('tabindex','0');
  $('#op-loss-cap',app).textContent=tx('Loss after each step (log scale, 200 steps)','每一步之后的 loss（对数刻度，200 步）');
}
/* part 7: the share of each weight that weight decay removes in one step, Adam + L2 and (once picked) AdamW, on one linear scale */
const wdName=k=>k==='l2'?'Adam + L2':'AdamW';
function drawWD(){
  const host=$('#op-wd',app), kinds=WDK.filter(k=>wdTried.has(k)), mx=Math.max(...kinds.flatMap(k=>[wdPull(k,WD.big),wdPull(k,WD.small)]));
  const row=(k,s)=>{const v=wdPull(k,s), tip=tx(`${wdName(k)}: at each step, weight decay removes ${fPct(v)} of the weight whose gradients are ±${s}.`,`${wdName(k)}：梯度为 ±${s} 的权重，weight decay 每一步让它缩小 ${fPct(v)}。`);
    return `<div class="op-wrow" tabindex="0" data-tipx="${esc(tip)}"><span class="op-wl">${esc(tx(`gradients ±${s}`,`梯度 ±${s}`))}</span><span class="op-btr"><i class="op-b ${k==='l2'?'op-s0':'op-s2'}" style="width:${Math.max(0.6,v/mx*100).toFixed(2)}%"></i></span><span class="op-bv mono">${fPct(v)}</span></div>`;};
  const html=`<div class="op-cap">${esc(tx('How much weight decay shrinks each weight in one step, as a share of the weight (both weights are 1.0)','weight decay 一步让每个权重缩小多少，按占权重的比例算（两个权重都是 1.0）'))}</div>`+
    kinds.map(k=>`<div class="op-wgrp${k===wdP?' on':''}"><div class="op-wh">${esc(k==='l2'?tx('Adam + L2: the pull is divided too','Adam + L2：拉力也被除了'):tx('AdamW: the same share for every weight','AdamW：每个权重缩小同样的比例'))}</div>${row(k,WD.big)}${row(k,WD.small)}</div>`).join('')+
    (wdTried.has('adamw')?'':`<p class="small muted">${esc(tx('Pick AdamW in the card to add its bars.','在卡片里选 AdamW，图上会加上它的横条。'))}</p>`);
  if(host._h!==html){host.innerHTML=html; host._h=html;}
}
const wdTip=()=>wdP==='l2'?tx(`Adam + L2: weight decay shrinks the weight with gradients ±${WD.big} by ${fPct(wdPull('l2',WD.big))} per step, and the one with ±${WD.small} by ${fPct(wdPull('l2',WD.small))}.`,`Adam + L2：weight decay 每一步把梯度 ±${WD.big} 的权重缩小 ${fPct(wdPull('l2',WD.big))}，把梯度 ±${WD.small} 的权重缩小 ${fPct(wdPull('l2',WD.small))}。`)
  :tx(`AdamW: weight decay shrinks both weights by ${fPct(wdPull('adamw'))} per step.`,`AdamW：weight decay 每一步把两个权重都缩小 ${fPct(wdPull('adamw'))}。`);
function statusOf(){
  if(isWD()) return {cls:'neutral',txt:wdP==='l2'?tx('Uneven decay','衰减不均'):tx('Even decay','衰减均匀'),tip:wdTip()};
  const runs=runsNow(), main=run(runs[runs.length-1].c);
  if(mode==='challenge'){const e=chEval(S.opt,S.lr); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),tip:tx(`Green when the loss falls below 0.001 within 100 steps. Now: ${resShort(e.r)}.`,`loss 在 100 步内降到 0.001 以下时，徽章是绿色。现在是：${resShort(e.r)}。`)};}
  if(mode==='guided'&&isP('predict')) return {cls:'neutral',txt:tx('Predict first','先预测'),tip:tx('The result at learning rate 0.10 stays hidden until you answer part 2.','在第 2 部分作答前，学习率 0.10 的结果先隐藏。')};
  if(runs.length>1){const rs=runs.map(x=>({x,r:run(x.c)})), hits=rs.filter(q=>q.r.hit!=null), dv=rs.some(q=>q.r.div!=null);
    const all=rs.map(q=>pairTxt(runLabel(q.x),resShort(q.r))).join(tx('; ','；'));
    if(!hits.length) return {cls:dv?'bad':'neutral',txt:resShort(main),tip:tx(`No run gets the loss below 0.001: ${all}.`,`没有一个运行把 loss 降到 0.001 以下：${all}。`)};
    const best=Math.min(...hits.map(q=>q.r.hit)), w=hits.filter(q=>q.r.hit===best), nobc=runs.some(x=>x.st==='nobc');
    const nm=x=>x.st==='nobc'?tx('no correction','无校正'):(nobc&&x.c.opt==='adam'?tx('with correction','有校正'):optName(x.c.opt));
    return {cls:dv?'bad':'neutral',txt:w.length>1?tx(`Tie: ${best} steps`,`打平：${best} 步`):tx(`Fastest: ${nm(w[0].x)}`,`最快：${nm(w[0].x)}`),
      tip:tx(`Fastest means the fewest steps to a loss below 0.001. ${all}.`,`最快指把 loss 降到 0.001 以下的步数最少。${all}。`)};}
  return {cls:main.div!=null?'bad':'neutral',txt:resShort(main),tip:tx(`${runLabel(runs[runs.length-1])}: ${resLong(main)}.`,`${runLabel(runs[runs.length-1])}：${resLong(main)}。`)};
}
function drawHead(){
  const t=$('#op-st-title',app), sub=$('#op-st-sub',app), st=$('#op-status',app), runs=runsNow();
  t.textContent=isWD()?tx('Weight decay on two weights: how much each shrinks per step','weight decay 作用在两个权重上：每一步各缩小多少'):(mode==='challenge'?tx('The steeper valley: steps to a loss below 0.001','更陡的山谷：loss 降到 0.001 以下要几步'):tx('Each optimizer’s path down the valley','每个优化器沿山谷往下走的路径'));
  sub.textContent=isWD()?tx(`${wdName(wdP)}, lr ${fLR(WD.lr)}, weight decay ${WD.wd}`,`${wdName(wdP)}，lr ${fLR(WD.lr)}，weight decay ${WD.wd}`):runs.map(runLabel).join(tx('; ','；'));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function legendHTML(){
  return '';   // R5.9: no legend row; the names are on the charts (drawMap, drawLoss) and part 7's groups carry their own titles
  const it=(cls,dash,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><svg class="op-sw" width="22" height="8" aria-hidden="true"><line x1="1" y1="4" x2="21" y2="4" class="${cls}" ${dash?`stroke-dasharray="${dash}"`:''}/></svg><span>${esc(txt)}</span></span>`;
  if(isWD()) return WDK.filter(k=>wdTried.has(k)).map(k=>`<span data-tipx="${esc(tx(`${wdName(k)}: the bar style it has in the chart.`,`${wdName(k)}：它在图里用这种横条。`))}" tabindex="0"><span class="op-swb op-b ${k==='l2'?'op-s0':'op-s2'}" aria-hidden="true"></span><span>${esc(k==='l2'?tx('solid: Adam + L2','实心：Adam + L2'):tx('dotted: AdamW','点状：AdamW'))}</span></span>`).join('');
  const runs=runsNow(), seen=new Set();
  let h=runs.map(x=>{const st=STY[x.st]||STY.sgd; if(seen.has(x.st)) return ''; seen.add(x.st);
    const nm={sgd:tx('solid: SGD','实线：SGD'),mom:tx('dashed: momentum','虚线：momentum'),adam:tx('dotted: Adam','点线：Adam'),adamw:tx('dotted: AdamW','点线：AdamW'),nobc:tx('dash-dot: Adam without bias correction','点划线：无偏差校正的 Adam')}[x.st];
    return it('op-swl '+st.cls,st.dash,nm,tx(`${runLabel(x)}: the line style it has in both charts.`,`${runLabel(x)}：它在两张图里都用这种线型。`));}).join('');
  h+=`<span data-tipx="${esc(tx('The thin arrow is the step shown under the chart; the small dot is where it lands.','细箭头是图下显示的那一步；小圆点是它落下的位置。'))}" tabindex="0"><span class="op-swa" aria-hidden="true">→</span><span>${esc(tx('arrow: the step shown','箭头：显示的那一步'))}</span></span>`;
  h+=`<span data-tipx="${esc(tx('The dashed line across the loss chart is the goal: a loss of 0.001.','loss 图上横穿的虚线是目标：loss 0.001。'))}" tabindex="0">${'<svg class="op-sw" width="22" height="8" aria-hidden="true"><line x1="1" y1="4" x2="21" y2="4" class="op-swg" stroke-dasharray="3 3"/></svg>'}<span>${esc(tx('thin dashed: goal 0.001','细虚线：目标 0.001'))}</span></span>`;
  return h;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
function metricsDef(){
  if(isWD()) return [WD.big,WD.small].map((s,i)=>({k:'wd'+i+wdP,label:tx(`Weight with gradients ±${s}`,`梯度 ±${s} 的权重`),v:tx(`${fPct(wdPull(wdP,s))} per step`,`每步 ${fPct(wdPull(wdP,s))}`),sub:wdName(wdP),
    tip:tx(`With ${wdName(wdP)}, weight decay shrinks this weight by ${fPct(wdPull(wdP,s))} of its size at each step.`,`用 ${wdName(wdP)} 时，weight decay 每一步把这个权重缩小 ${fPct(wdPull(wdP,s))}。`)}));
  const runs=runsNow();
  return runs.map((x,i)=>{const r=run(x.c);
    const sub=r.hit!=null?tx('to a loss below 0.001','到 loss 0.001 以下'):(r.div!=null?tx(`loss passed 100M at step ${r.div}`,`第 ${r.div} 步 loss 超过 100M`):tx(`loss still above 0.001 at step ${N}`,`第 ${N} 步 loss 仍在 0.001 以上`));
    if(mode==='guided'&&isP('predict')) return {k:'r'+i,label:runLabel(x),v:resShort(r),sub,tip:tx(`${runLabel(x)}: ${resLong(r)}.`,`${runLabel(x)}：${resLong(r)}。`)};
    return {k:'r'+i+x.st,label:runLabel(x),v:resShort(r),sub,tip:tx(`${runLabel(x)}: ${resLong(r)}.`,`${runLabel(x)}：${resLong(r)}。`)};})
   .concat((()=>{const x=runs[runs.length-1], r=run(x.c), k=Math.min(K,endOf(r));
     return [{k:'kloss',label:tx(`Loss at step ${k}`,`第 ${k} 步的 loss`),v:fLoss(r.loss[k]),sub:runLabel(x),tip:k===K?tx(`The loss of ${runLabelP(x)} after step ${k}, the step shown under the valley.`,`${runLabelP(x)}走完第 ${k} 步后的 loss。第 ${k} 步就是山谷图下方正在显示的那一步。`)
       :(r.hit!=null?tx(`${runLabelP(x)} reached a loss below 0.001 at step ${k} and stopped there; this is its loss after that last step.`,`${runLabelP(x)}在第 ${k} 步降到了 0.001 以下，停在那里。这是它走完最后一步后的 loss。`)
       :tx(`${runLabelP(x)} stopped at step ${k}${r.div!=null?', where its loss passed 100M':''}; this is its loss after that last step.`,`${runLabelP(x)}停在第 ${k} 步${r.div!=null?'，loss 在这一步超过了 100M':''}。这是它走完最后一步后的 loss。`))}];})());
}
function drawMetrics(){
  const host=$('#op-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    const isN=t=>/^\d+ (steps|步)$/.test(t); if(userAct&&m.k!=='kloss'&&prev!=null&&prev!==m.v&&isN(prev)&&isN(m.v)) dt=A.fmt.delta(numOf(prev),numOf(m.v),{})||'';
    /* a badge stays while its value stands: the playback's frames redraw the cards without clearing it */
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(prev!==m.v){clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* "Show the math": every symbol of a formula has its definition on hover (R1.4); the learning rate is written lr, as in the cards (R2.8) */
function symTips(){const sf=surfNow(), b=sf==='v100'?100:25;
  const along=sf==='curve'?tx('x is one of the two parameters.','x 是两个参数之一。'):tx('x is the parameter along the valley, the flat direction.','x 是沿山谷方向的参数，平的方向。');
  const across=sf==='curve'?tx('y is the other parameter.','y 是另一个参数。'):tx('y is the parameter across the valley, the steep direction.','y 是横穿山谷方向的参数，陡的方向。');
  return {f:tx('f is the loss. It is 0 at the bottom.','f 是 loss，谷底处为 0。'),x:along,y:across,
    'θ':tx('θ is one parameter: x or y here. Each formula runs for each parameter on its own.','θ 是一个参数：这里是 x 或 y。每个公式对每个参数各算一遍。'),
    lr:tx('lr is the learning rate, the same lr as in the cards and Settings.','lr 是学习率，和卡片、自由调参里的 lr 是同一个。'),
    g:tx('g is the gradient for that parameter: how much the loss rises when it grows a little.','g 是这个参数的梯度：它增加一点时，loss 涨多少。'),
    t:tx('t is the step number: 1, 2, 3 …','t 是步数：1、2、3……'),
    'ε':tx('ε is 10⁻⁸ (10⁻⁵ with β2 0.95, as in Llama 2). It keeps the division safe while v is near 0.','ε 是 10⁻⁸（β2 为 0.95 时取 10⁻⁵，和 Llama 2 一样）。v 接近 0 时，它让除法不出问题。'),
    'λ':sf==='curve'?tx('λ is the curvature of a direction: how fast its gradient grows with distance. On the curved valley it changes from place to place.','λ 是一个方向的曲率：梯度随距离增长得多快。弯山谷上它处处不同。')
      :tx(`λ is the curvature of a direction: how fast its gradient grows with distance. Here it is ${b} across the valley and 1 along it.`,`λ 是一个方向的曲率：梯度随距离增长得多快。这里横穿山谷是 ${b}，沿山谷是 1。`)};}
function mathT(tex,extra){const d={...symTips(),...(extra||{})};
  return math(tex).replace(/<mi( mathvariant="normal")?>([^<]+)<\/mi>/g,(m0,mv,sym)=>d[sym]?`<mi${mv||''} class="op-sym" tabindex="0" data-tipx="${esc(d[sym])}">${sym}</mi>`:m0);}
function mathHTML(){
  if(isWD()){const d={'θ':tx('θ is the weight, 1.0 here.','θ 是权重，这里是 1.0。'),wd:tx(`wd is the weight decay, ${WD.wd} here.`,`wd 是 weight decay，这里是 ${WD.wd}。`),s:tx(`s is the size of the weight’s gradients: ${WD.big} or ${WD.small} here.`,`s 是这个权重梯度的大小：这里是 ${WD.big} 或 ${WD.small}。`)};
    return `<p>Adam + L2 ${mathT('g \\leftarrow g + \\mathrm{wd} \\cdot \\theta',d)}${tx(', then Adam’s usual step on this g.','，然后用这个 g 照常走 Adam 的一步。')}</p>`+
      `<p>AdamW ${tx('besides Adam’s usual step:','除了 Adam 照常的一步，还有')} ${mathT('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot \\mathrm{wd} \\cdot \\theta',d)}</p>`+
      `<p>${tx('Share removed per step once m and v have settled, for gradients ±s: Adam + L2','m 和 v 稳定之后，梯度为 ±s 时每一步缩小的比例。Adam + L2 是')} ${mathT('\\frac{\\mathrm{lr} \\cdot \\mathrm{wd}}{\\sqrt{s^2 + (\\mathrm{wd} \\cdot \\theta)^2}}',d)}${tx(', AdamW','，AdamW 是')} ${mathT('\\mathrm{lr} \\cdot \\mathrm{wd}',d)}${tx('.','。')}</p>`;}
  const runs=runsNow(), r0=run(runs[runs.length-1].c), k=Math.min(K,endOf(r0)), p=r0.pts[k-1]||r0.pts[0], d=r0.d[k]||[0,0], t_=T_;
  const vMom={b:tx('b is momentum’s running direction: the sum of past gradients, each shrunk by β per step (PyTorch’s momentum_buffer). It starts at 0.','b 是 momentum 累积的方向：过去的梯度加起来，每过一步乘一次 β（PyTorch 叫 momentum_buffer），从 0 开始。'),'β':tx('β is momentum: the share of b each step keeps, 0.9 by default.','β 是 momentum：每一步保留多少 b，默认 0.9。')};
  const vAdam={m:tx('m is Adam’s running mean of the gradient. It starts at 0.','m 是 Adam 的梯度滑动平均，从 0 开始。'),v:tx('v is Adam’s running mean of the squared gradient. It starts at 0.','v 是 Adam 的梯度平方滑动平均，从 0 开始。'),'β':tx('β1 (0.9) and β2 (0.999, or 0.95 in Llama 2) set how fast m and v forget old gradients.','β1（0.9）和 β2（0.999，Llama 2 用 0.95）决定 m 和 v 忘掉旧梯度有多快。')};
  const rows=[`<p>${tx('Loss','loss')} ${mathT(surfNow()==='curve'?'f = (1-x)^2 + 10\\,(y-x^2)^2':(surfNow()==='v100'?'f = \\frac{1}{2}(x^2 + 100\\,y^2)':'f = \\frac{1}{2}(x^2 + 25\\,y^2)'))}</p>`,
    `<p>${t_('lrP','SGD')} ${mathT('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot g')}</p>`,
    `<p>${t_('beta','Momentum')} ${mathT('b \\leftarrow \\beta b + g,\\quad \\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot b',vMom)}</p>`,
    `<p>${t_('lrA','Adam')} ${mathT('m \\leftarrow \\beta_1 m + (1-\\beta_1) g,\\quad v \\leftarrow \\beta_2 v + (1-\\beta_2) g^2',vAdam)}</p>`,
    `<p>${t_('bc',tx('Bias correction','偏差校正'))} ${mathT('\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot \\frac{m / (1-\\beta_1^t)}{\\sqrt{v / (1-\\beta_2^t)} + \\epsilon}',vAdam)}</p>`,
    (mode!=='guided'||step>=SIX('sgd')?`<p>${tx('SGD is stable on a direction of curvature λ only if','SGD 在曲率为 λ 的方向上稳定的条件是')} ${mathT('|1 - \\mathrm{lr} \\cdot \\lambda| < 1')}${tx(', that is','，即')} ${mathT('\\mathrm{lr} < 2/\\lambda')}${{v25:tx('. Here λ is 25 across the valley, so lr < 0.08.','。这里横穿山谷的 λ 是 25，所以 lr < 0.08。'),v100:tx('. Here λ is 100 across the valley, so lr < 0.02.','。这里横穿山谷的 λ 是 100，所以 lr < 0.02。'),curve:tx('. On the curved valley λ changes from place to place.','。弯山谷上 λ 处处不同。')}[surfNow()]}</p>`:''),
    `<p><b>${esc(runLabel(runs[runs.length-1]))}${tx(`, step ${k}: `,`，第 ${k} 步：`)}</b>${tx('from','从')} (${f2(p[0])}, ${f2(p[1])}) ${tx('by','走')} (${sgn(fixd(-d[0],3))}, ${sgn(fixd(-d[1],3))})${tx(', learning rate ','，学习率 ')}${fLR(+r0.lrs[k].toPrecision(3))}${tx('.','。')}</p>`];
  return rows.join('');
}
function drawStage(){
  const wd=isWD(); $('#op-graph',app).classList.toggle('op-wdmode',wd);
  if(wd){drawWD(); return;}
  drawMap(); drawBars(); drawLoss();
  const ki=$('#op-k',app), ko=$('#op-k-out',app), km=kMax();
  if(ki){if(+ki.max!==km) ki.max=km; if(+ki.value!==K) ki.value=K; A.ui.setRange&&A.ui.setRange(ki);} if(ko) ko.textContent=tx(`step ${K}`,`第 ${K} 步`);
  syncPlayer();}
/* ---------- playback (R3.9): time-based; a whole replay of the longest run takes SPEEDS[spd] seconds ---------- */
const kMax=()=>isWD()?1:Math.max(1,...runsNow().map(x=>endOf(run(x.c))));
function syncPlayer(){if(!app) return; const km=kMax();
  $$('[data-act="play"]',app).forEach(b=>{b.textContent=playing?tx('❚❚ Pause','❚❚ 暂停'):(K>=km?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放')); b.setAttribute('aria-pressed',String(playing));});
  $$('.seg-ctl[data-key="spd"]',app).forEach(g=>A.ui.setPressed(g,spd));}
function stopPlay(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0;}
/* one frame: the charts and the step's numbers; at the end also the math and the mini result bar */
function frame(done){if(!app) return; drawStage(); drawMetrics();
  if(done){const mb=$('#op-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;} A.dock.refresh();}}
function loop(){
  stopPlay(); if(!app||isWD()) return; pendWant=false; playing=true; let last=0, acc=K;
  const tick=ts=>{if(!playing||!app) return; if(!last) last=ts; acc+=Math.min(100,ts-last)/1000*kMax()/SPEEDS[spd]; last=ts;
    const km=kMax(), k=Math.min(km,Math.max(1,Math.floor(acc))); if(k!==K){K=k; frame(false);}
    if(K>=km){stopPlay(); markWatched(); frame(true); return;}
    raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick);
}
function play(){if(K>=kMax()) K=1; loop(); frame(false);}
function pause(){stopPlay(); frame(true);}
/* new runs play from step 1 when the valley is in view; out of view they show the end and play once it comes into view;
   with reduced motion they show the end, and the play button still plays them */
/* the same path gets the same key in every mode: the valley filled in, values equal to run()'s defaults left out */
const PATH_DEF={b1:0.9,b2:0.999,eps:1e-8,bc:true,scale:1,warm:0,wd:0};
const pathKey=c=>{const o={...c,surf:c.surf||surfNow()}; return Object.keys(o).filter(k=>PATH_DEF[k]!==o[k]).sort().map(k=>k+':'+o[k]).join(',');};
/* R3.20: the valleys the reader has watched to the end (played, or stepped to the last step), each by the paths it draws, not by part
   or mode. A setup whose paths were all drawn together on a valley watched to the end (A → B → A, Back to a part, a scenario, the
   challenge) shows its end at once; a new combination of paths still plays. A run stopped part-way, or shown at its end while out of
   view, does not count */
const watched=[];
const wasWatched=keys=>keys.length>0&&watched.some(w=>keys.every(k=>w.includes(k)));
function markWatched(){if(!app||isWD()||K<kMax()) return; const keys=runsNow().map(x=>pathKey(x.c)); if(!wasWatched(keys)) watched.push(keys);}
/* R3.20: when every run of the new setup is already drawn (the next part, a mode switch, fewer runs), or was drawn on a valley watched
   to the end before, show the end at once; Replay plays it again */
function autoPlay(){
  stopPlay(); pendWant=false;
  const keys=isWD()?[]:runsNow().map(x=>pathKey(x.c)), same=keys.length>0&&keys.every(k=>onMap.includes(k)), seen=wasWatched(keys); onMap=keys;
  if(isWD()){K=1; return;}
  /* G6 (the owner, 2026-10-08): Scenarios and Settings, where readers switch often, show the end state at once; Replay still plays */
  if(A.reduceMotion()||same||seen||mode==='sandbox'){K=kMax(); return;}
  if(inView){K=1; loop(); return;}
  K=kMax(); pendWant=true;
}
function watchView(){
  if(io){io.disconnect(); io=null;} inView=false;
  const m=$('#op-map',app); if(!m||!('IntersectionObserver' in window)){inView=true; return;}
  io=new IntersectionObserver(es=>{es.forEach(e=>{inView=e.isIntersecting&&e.intersectionRatio>=0.5;});
    if(inView&&pendWant&&!A.reduceMotion()&&!isWD()&&app){K=1; loop(); frame(false);}},{threshold:[0,0.5,1]});
  io.observe(m);
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!CH.opts.includes(S.opt)) S.opt=CH.start.opt; if(!CH.lrs.includes(S.lr)) S.lr=CH.start.lr;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const sig=JSON.stringify([mode,mode==='guided'?step:0,surfNow(),runsNow().map(x=>x.c)]); if(sig!==runSig){runSig=sig; autoPlay();}
  drawHead(); drawStage(); drawMetrics();
  const lg=$('#op-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;} lg.hidden=!lh;
  const mb=$('#op-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the optimizer and the learning rate, in the task card. Everything else is fixed: the steeper valley ${VAL(100)}, the start (−5, 0.8), momentum β 0.9, Adam’s defaults.`,`挑战中只能在任务卡里改优化器和学习率。其余固定：更陡的山谷 ${VAL(100)}、出发点 (−5, 0.8)、momentum β 0.9、Adam 的默认值。`);
  if(mode!=='guided') return '';
  const sc=`<button type="button" class="linkbtn" data-mode-go="sandbox">${tx('Scenarios','应用场景')}</button>`;
  if(isWD()) return tx(`Fixed in this part: two weights of 1.0, lr ${fLR(WD.lr)} and weight decay ${WD.wd}; the one choice is in the card. In ${sc} you can run AdamW on the valley.`,`这一部分固定：两个权重都是 1.0，lr ${fLR(WD.lr)}，weight decay ${WD.wd}；要选的只有卡片里那一项。在${sc}里可以让 AdamW 在山谷上跑。`);
  return tx(`Fixed in this part: each part sets its own runs on the valley ${VAL(25)}, and its one control is in the card. You can change every setting in ${sc}.`,`这一部分固定：每一部分在山谷 ${VAL(25)} 上设好自己的运行，要改的那一个设置在卡片里。在${sc}里可以改每个设置。`);
}
function syncControls(){
  const setR=(key,list,val,fmt)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{const v=Math.max(0,list.indexOf(val)); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl'); const out=o&&o.querySelector('output'); if(out) out.textContent=fmt(val);});
  setR('lrP',LR_PART,lrP,fLR); setR('betaP',BETAS,betaP,fB); setR('lrA',LR_ADAM,lrA,fLR); setR('lr',LR_ALL,S.lr,fLR); setR('beta',BETAS,S.beta,fB);
  [['opt',S.opt],['b2',String(S.b2)],['bc',S.bc?'on':'off'],['wd',String(S.wd)],['warm',String(S.warm)],['surf',S.surf]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,v)));
  const sb=mode==='sandbox', adam=/adam/.test(S.opt);
  const hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  ['#op-ctl-opt-wrap','#op-ctl-lr-wrap','#op-ctl-surf-wrap','#op-ctl-warm-wrap','#op-ctl-reset-wrap'].forEach(i=>hide(i,!sb));
  hide('#op-ctl-beta-wrap',!sb||S.opt!=='mom'); hide('#op-ctl-b2-wrap',!sb||!adam); hide('#op-ctl-bc-wrap',!sb||!adam); hide('#op-ctl-wd-wrap',!sb||S.opt!=='adamw');
  $('#op-controls',app).classList.toggle('locked',!sb);
  const ln=$('#op-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln._h!==msg){ln.innerHTML=msg; ln._h=msg;}
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; if(isWD()){A.announce(wdTip()); return;} const runs=runsNow(); A.announce(runs.map(x=>pairTxt(runLabel(x),resShort(run(x.c)))).join(tx('; ','；')));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p); if(!LR_ALL.includes(S.lr)) S.lr=0.05; if(!BETAS.includes(S.beta)) S.beta=0.9; render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLrP(v){lrP=v; if(v>0.08) seenDiv=true; render();}
function setBeta(v){betaP=v; betaMoved=true; render();}
function setLrA(v){lrA=v; lrAMoved=true; render();}
function setBc(v){bcP=v; bcTried.add(v); render();}
function setWd(v){wdP=v; wdTried.add(v); render();}
/* the step slider and ◂ ▸ pause the playback */
function setK(k){stopPlay(); pendWant=false; K=Math.max(1,Math.min(kMax(),k)); markWatched(); frame(true);}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i)); K=1;
  const id=STEPS[step].id;
  if(id==='sgd'&&!seenDiv) lrP=0.05;
  if(id==='mom'&&!betaMoved) betaP=0;
  if(id==='adam'&&!lrAMoved) lrA=0.05;
  if(id==='bias') lrA=0.5;
  const st=$('#op-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#op-rail .step.current',app)||$('#op-rail',app));}
/* keepStage (CONTRACT §11, R3.2, R9.7; as labs/moe.js; polish round 5): after Next, Back or a part's title the kit's reveal scrolls the
   new card's top up under the top bar; near the end of the bench that pushes the sticky stage up with it (part 6 entered from part 5 at
   1440 × 900: the stage's top at −176 px, the top of the valley off the screen). Once the page stops scrolling this pulls the stage back
   under the top bar without pushing the card's top off the screen, and does nothing once the reader has scrolled the card away */
let stageT=0;
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#op-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#op-rail .step.current',app)||$('#op-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* the loss chart is drawn for its column's width on a wide stage (drawLoss): the stage changes width without a window resize when the
   page puts the step list beside it or leaves the primer, so it is drawn again when it does (as labs/ppo.js) */
let stRO=null, stW=0;
function watchStage(){if(stRO){stRO.disconnect(); stRO=null;} const st=app&&$('#op-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w; drawStage();}));
  stRO.observe(st);}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#op-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#op-insight',app)); else revealBox($('#op-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; K=1;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...BASE,opt:CH.start.opt,lr:CH.start.lr};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#op-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const OPTS=['sgd','mom','adam','adamw'], SURFS=['v25','curve'];
const token=()=>[OPTS.indexOf(S.opt),LR_ALL.indexOf(S.lr),BETAS.indexOf(S.beta),S.b2===0.95?0:1,S.bc?1:0,S.wd?1:0,S.warm?1:0,SURFS.indexOf(S.surf)].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==8||p.some(v=>!Number.isInteger(v))) return null;
  const [o,l,b,b2,bc,wd,wm,sf]=p; if(!OPTS[o]||LR_ALL[l]==null||BETAS[b]==null||!SURFS[sf]) return null;
  return {opt:OPTS[o],lr:LR_ALL[l],beta:BETAS[b],b2:b2?0.999:0.95,bc:!!bc,wd:wd?0.1:0,warm:wm?20:0,surf:SURFS[sf]};
}

/* =====================================================================
   interview drill: a seeded valley; the largest safe SGD learning rate and the steps at a given rate
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*104729+7), b=[10,20,40,50,80][Math.floor(r()*5)], frac=[0.5,0.8,0.9][Math.floor(r()*3)], lr=Math.round(2/b*frac*1000)/1000, x0=-[4,6,8][Math.floor(r()*3)];
  const steps=Math.ceil(Math.log(Math.sqrt(2*GOAL)/Math.abs(x0))/Math.log(1-lr));
  return {seed,b,lr,x0,max:2/b,xf:1-lr,yf:1-b*lr,steps};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#op-drill-q',app), a=$('#op-drill-a',app);
  q.innerHTML=tx(`The loss is ${VAL(d.b)} and SGD starts at x = ${sgn(d.x0)}. What is the largest learning rate that does not diverge? At learning rate ${d.lr}, about how many steps does x alone need to bring its part of the loss below 0.001?`,
                 `loss 是 ${VAL(d.b)}，SGD 从 x = ${sgn(d.x0)} 出发。不发散的最大学习率是多少？学习率为 ${d.lr} 时，单看 x，大约要几步才能让它那部分 loss 低于 0.001？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>Below 2 ÷ ${d.b} = ${fixd(d.max,3)}; about ${d.steps} steps.</b></p><p>Each step multiplies y by 1 − ${d.b} × ${d.lr} = ${sgn(fixd(d.yf,3))} and x by 1 − ${d.lr} = ${fixd(d.xf,3)}. y stays bounded while:</p><div class="op-fx">${math(`|1 - ${d.b}\\,\\eta| < 1, \\quad \\eta < \\frac{2}{${d.b}}`)}</div><p>x’s part of the loss, ${math('\\frac{1}{2}x^2')}, is below 0.001 once ${math('|x| < \\sqrt{0.002} = 0.0447')}. After n steps, x has shrunk to:</p><div class="op-fx">${math(`${fixd(d.xf,3)}^n \\times ${Math.abs(d.x0)} < 0.0447`)}</div><p>so:</p><div class="op-fx">${math(`n > \\frac{ln(0.0447 / ${Math.abs(d.x0)})}{ln ${fixd(d.xf,3)}}`)}</div><p>Rounded up, that is ${d.steps - 1 + (d.steps>0?1:0)} steps.</p>`,
                 `<p><b>小于 2 ÷ ${d.b} = ${fixd(d.max,3)}；大约 ${d.steps} 步。</b></p><p>每一步把 y 乘以 1 − ${d.b} × ${d.lr} = ${sgn(fixd(d.yf,3))}，把 x 乘以 1 − ${d.lr} = ${fixd(d.xf,3)}。y 不会越来越大的条件是：</p><div class="op-fx">${math(`|1 - ${d.b}\\,\\eta| < 1, \\quad \\eta < \\frac{2}{${d.b}}`)}</div><p>x 那部分 loss ${math('\\frac{1}{2}x^2')} 低于 0.001，要 ${math('|x| < \\sqrt{0.002} = 0.0447')}。走 n 步后，x 缩小到：</p><div class="op-fx">${math(`${fixd(d.xf,3)}^n \\times ${Math.abs(d.x0)} < 0.0447`)}</div><p>所以：</p><div class="op-fx">${math(`n > \\frac{ln(0.0447 / ${Math.abs(d.x0)})}{ln ${fixd(d.xf,3)}}`)}</div><p>向上取整是 ${d.steps} 步。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={lrP:{en:'SGD learning rate',zh:'SGD 的学习率'},beta:{en:'Momentum β',zh:'momentum β'},lrA:{en:'Adam learning rate',zh:'Adam 的学习率'},bc:{en:'Bias correction',zh:'偏差校正'}};
const TERM={lrP:()=>TIP.sgd()+' '+TIP.lr(),beta:()=>TIP.mom(),lrA:()=>TIP.adam(),bc:()=>TIP.bc()};
const termTipHTML=k=>TERM[k]?`<div>${TERM[k]()}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
const partOf={lrP:'sgd',beta:'mom',lrA:'adam',bc:'bias'};
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: only the optimizer and the learning rate in the task card change.','挑战中固定不变：只能改任务卡里的优化器和学习率。');
  if(mode==='guided'&&!isP(partOf[k])) return tx(`Fixed in this part; part ${SIX(partOf[k])+1} lets you change it.`,`这一部分固定不变；第 ${SIX(partOf[k])+1} 部分可以改。`);
  return '';
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='scen') return tx('Warmup: at the start of training, the learning rate rises slowly from near 0 to its peak. Click to open scenario 1: Llama 2’s settings use it.','warmup：训练刚开始的一段时间里，把学习率从接近 0 慢慢升到峰值。点击打开应用场景 1，Llama 2 的设置里用了它。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(sbName(k))}”.`,`点击跳到“${L(sbName(k))}”。`);
}
/* in Scenarios the part's control is not there: a term jumps to the matching setting, or to the optimizer when that setting is hidden */
const SBW={lrP:'#op-ctl-lr-wrap',lrA:'#op-ctl-lr-wrap',beta:'#op-ctl-beta-wrap',bc:'#op-ctl-bc-wrap'};
const sbVisible=k=>{const e=app&&$(SBW[k],app); return !!e&&!e.hidden;};
function sbName(k){if(mode!=='sandbox') return CTLN[k]; if(!sbVisible(k)) return {en:'Optimizer',zh:'优化器'}; return (k==='lrP'||k==='lrA')?{en:'Learning rate',zh:'学习率'}:CTLN[k];}
const ctlEl=k=>mode==='sandbox'?(sbVisible(k)?$(SBW[k],app):$('#op-ctl-opt-wrap',app)):({lrP:$('#op-ic-lr',app),beta:$('#op-ic-beta',app),lrA:$('#op-ic-lrA',app),bc:$('#op-ic-bc',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#op-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#op-controls',app)); return;}
  if(k==='scen'){A.tip.hide(); setMode('sandbox'); byLearner(()=>setState({...PRESETS[0].s})); A.ui.reveal($('#op-rail',app)); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){byLearner(()=>{S={...BASE}; render();}); return;}
  if(a==='k-'){A.tip.quiet&&A.tip.quiet(); setK(K-1); return;}
  if(a==='k+'){A.tip.quiet&&A.tip.quiet(); setK(K+1); return;}
  if(a==='play'){if(playing) pause(); else play(); if(A.tip.quiet) A.tip.quiet(); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.opt,S.lr).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#op-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();} finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#op-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const t=e.target; if(t.id==='op-k'){setK(+t.value); return;} const k=t.dataset&&t.dataset.key; if(!k) return; const v=+t.value;
    if(k==='lrP') byLearner(()=>setLrP(LR_PART[v]));
    else if(k==='betaP') byLearner(()=>setBeta(BETAS[v]));
    else if(k==='lrA') byLearner(()=>setLrA(LR_ADAM[v]));
    else if(k==='lr') bySettings(t,()=>setState({lr:LR_ALL[v]}));
    else if(k==='beta') bySettings(t,()=>setState({beta:BETAS[v]}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#op-rail')){wantObs=true; revealSoon(); wantObs=false;}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(d.go!=null){enterStep(+d.go); revealStep(); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); keepStage(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#op-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('bias')){byLearner(()=>setBc(d.lever)); revealSoon(); return;}
    if(d.lever!=null&&isP('adamw')){byLearner(()=>setWd(d.lever)); revealSoon(); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null&&seg.dataset.key==='spd'){spd=d.v; syncPlayer(); return;}
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='chO') byLearner(()=>setState({opt:v}));
      else if(k==='chL') byLearner(()=>setState({lr:+v}));
      else if(k==='opt') bySettings(b,()=>setState({opt:v}));
      else if(k==='b2') bySettings(b,()=>setState({b2:+v}));
      else if(k==='bc') bySettings(b,()=>setState({bc:v==='on'}));
      else if(k==='wd') bySettings(b,()=>setState({wd:+v}));
      else if(k==='warm') bySettings(b,()=>setState({warm:+v}));
      else if(k==='surf') bySettings(b,()=>setState({surf:v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#op-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#op-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps it, address token included (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount.
     The playhead K is not kept: after a reload a valley in watched (seen to the end) shows its end at once, another one plays */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,sbTouched,lrP,betaP,lrA,bcP,seenDiv,betaMoved,lrAMoved,
    bcTried:[...bcTried],wdP,wdTried:[...wdTried],spd,watched:watched.slice(-40).map(w=>w.slice()),drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.opt,S.lr).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse([OPTS.indexOf(x.opt),LR_ALL.indexOf(x.lr),BETAS.indexOf(x.beta),x.b2===0.95?0:(x.b2===0.999?1:'x'),x.bc?1:0,x.wd?1:0,x.warm?1:0,SURFS.indexOf(x.surf)].join('_')):null;
    const p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    lrP=LR_PART.includes(o.lrP)?o.lrP:0.05; betaP=BETAS.includes(o.betaP)?o.betaP:0; lrA=LR_ADAM.includes(o.lrA)?o.lrA:0.05;
    bcP=['on','off'].includes(o.bcP)?o.bcP:'on'; seenDiv=o.seenDiv===true; betaMoved=o.betaMoved===true; lrAMoved=o.lrAMoved===true;
    bcTried=new Set(['on']); if(Array.isArray(o.bcTried)) o.bcTried.forEach(x=>{if(x==='on'||x==='off') bcTried.add(x);}); bcTried.add(bcP);
    wdP=WDK.includes(o.wdP)?o.wdP:'l2'; wdTried=new Set(['l2']); if(Array.isArray(o.wdTried)) o.wdTried.forEach(x=>{if(WDK.includes(x)) wdTried.add(x);}); wdTried.add(wdP);
    spd=Object.prototype.hasOwnProperty.call(SPEEDS,o.spd)?o.spd:'med';
    watched.length=0; if(Array.isArray(o.watched)) o.watched.slice(-40).forEach(w=>{if(Array.isArray(w)&&w.length&&w.length<12&&w.every(k=>typeof k==='string'&&k.length<200&&/^[\w.:,+-]*$/.test(k))) watched.push(w.slice());});
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:2; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    const resume=playing; stopPlay();
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    fromSnap=false;
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#op-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#op-rail',app));},token});
    bind(app);
    renderDrill();
    fillRail();
    render();
    watchView(); watchStage(); if(resume) loop();
  },
  leave(){clearTimeout(annT); stopPlay(); stageStop(); if(stRO){stRO.disconnect(); stRO=null;} if(io){io.disconnect(); io=null;} inView=false; pendWant=false; runSig=''; onMap=[]; root=null; app=null;},
  onLang(){if(root){const d=$('#op-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#op-math',app).open=true;}},
  onResize(){if(app) drawStage();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(isWD()?$('#op-wd',app):$('#op-map',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#op-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), runs=runsNow();
      status.className='status '+s.cls; status.textContent=s.txt;
      if(isWD()){vis.innerHTML=''; read.innerHTML=esc(wdTip()); return;}
      vis.innerHTML=`<svg class="op-dk" viewBox="0 0 120 24" preserveAspectRatio="none" aria-hidden="true">${runs.map(x=>{const r=run(x.c), st=STY[x.st]||STY.sgd, pts=[]; for(let t=0,e=r.hit!=null?r.hit:r.n;t<=e;t+=(t+2>e&&t<e?e-t:2)) pts.push(`${(t/N*118+1).toFixed(1)},${(2+(3-Math.log10(Math.max(1e-4,Math.min(1e3,r.loss[t]))))/7*20).toFixed(1)}`); return `<polyline class="op-lline ${st.cls}" points="${pts.join(' ')}" style="stroke-width:1.4${st.dash?`;stroke-dasharray:${st.dash}`:''}"/>`;}).join('')}</svg>`;
      if(mode==='guided'&&isP('predict')){read.innerHTML=esc(tx(`SGD at lr 0.05: ${resShort(run(SGD05))} to a loss below 0.001`,`lr 0.05 的 SGD：${resShort(run(SGD05))}到 loss 0.001 以下`)); return;}
      const many=runs.length>1&&new Set(runs.map(x=>x.st)).size===runs.length;
      read.innerHTML=esc(runs.map(x=>pairTxt(many?(x.st==='nobc'?tx('Adam, no bias correction','Adam，无偏差校正'):optName(x.c.opt)):runLabel(x),resShort(run(x.c)))).join(tx('; ','；')));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__op=Object.freeze({
  run:c=>{const r=run(c); return deep({hit:r.hit,div:r.div,n:r.n,pts:r.pts,loss:r.loss.map(v=>isFinite(v)?v:null),d:r.d,lrs:r.lrs});},
  chEval:(o,lr)=>{const e=chEval(o,lr); return {steps:e.steps,ok:e.ok,div:e.r.div};}, chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)),
  rate:(lam,lr,beta)=>hbRate(lam,lr,beta), bestBeta:()=>bestBeta(), biggest:(lr,bc)=>deep(biggest(run({opt:'adam',lr,bc}),lr)),
  wdPull:(k,s)=>wdPull(k,s), kMax:()=>kMax(),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});},
  consts:Object.freeze({N,GOAL,SURF:deep(Object.fromEntries(Object.entries(SURF).map(([k,v])=>[k,{start:v.start,view:v.view,min:v.min}]))),CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),PRED_RIGHT,STEP_IDS:STEPS.map(s=>s.id),LR_PART,BETAS,LR_ADAM,LR_ALL,CODE,WD:deep(WD),SPEEDS:deep(SPEEDS)}),
  state:()=>({...S,mode,step,pred,observed:{...observed},lrP,betaP,lrA,bcP,wdP,seenDiv,K,playing,spd,chSubmitted,chReveal})
});
})();
