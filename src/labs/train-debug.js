/* train-debug (roots, ML basics): Your loss will not go down: where is the bug?
   Fidelity: Toy run. A small classifier really trains in the page: 2-D points in 3 groups (4 in the challenge), a 2 → 16 → C ReLU
   network, plain gradient descent on the whole training set for 300 steps, float64, everything seeded.
   - Bugs are switches in the training step: labels shuffled apart from the inputs (a fresh permutation every step), zero_grad() missing
     (each step adds its gradient to the stored one and steps with the sum), softmax applied before the loss's own softmax. The learning
     rate is a setting: 0.003, 0.3 (healthy) or 10.
   - The script's softmax is written out as exp(z) / Σ exp(z), without subtracting the largest logit. When a logit leads the right
     class's by more than about 745, that class's chance rounds to 0 and the loss is inf; a logit above about 709 overflows exp and
     gives NaN (said in Fidelity, with the link to the softmax lab).
   - The three checks: 1 the loss at step 0 against ln C (and whether the run ever leaves it); 2 overfit a tiny batch (3 examples per
     class, their real labels, no data loader, a fresh model, the run's own step and learning rate, 200 steps, pass below 0.1);
     3 the gradient norm (the L2 norm of the gradient the step uses, over all parameters).
   Words (EN / ZH): loss; learning rate / 学习率; gradient norm / 梯度范数; validation / 验证集; part / 部分; step / 步 (training steps only).
   window.__td is a read-only hook for tests/train-debug_*.py and the home tile (teasers/train-debug.js). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='train-debug';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine (pure, seeded, float64)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* one normal number from two uniforms (Box–Muller, cosine branch) */
const gauss=r=>{const u1=1-r(), u2=r(); return Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2);};
const HID=16, STEPS=300, NTR=100, NVA=50, OF_K=3, OF_STEPS=200, OF_PASS=0.1, INIT_SEED=7, SHUF_SEED=11, SNAPS=[0,30,100,300];
const LRS=[0.003,0.3,10], LR0=0.3, SMALL_K=5;
/* C groups of points around a circle of radius 2 (spread S), 100 per group for training and 50 for validation; both splits are
   standardized with the training split's mean and standard deviation */
const DSPEC={3:{S:1.0,seed:1},4:{S:0.8,seed:2}};
const DATA={};
function dataOf(C){
  if(DATA[C]) return DATA[C];
  const sp=DSPEC[C], r=mulberry(sp.seed), ctr=[...Array(C).keys()].map(c=>{const a=Math.PI/2+2*Math.PI*c/C; return [2*Math.cos(a),2*Math.sin(a)];});
  const blob=n=>{const X=[],y=[]; for(let c=0;c<C;c++) for(let i=0;i<n;i++){X.push([ctr[c][0]+sp.S*gauss(r),ctr[c][1]+sp.S*gauss(r)]); y.push(c);} return {X,y};};
  const tr=blob(NTR), va=blob(NVA);
  const mu=[0,1].map(j=>tr.X.reduce((a,x)=>a+x[j],0)/tr.X.length), sd=[0,1].map(j=>Math.sqrt(tr.X.reduce((a,x)=>a+(x[j]-mu[j])**2,0)/tr.X.length));
  const st=X=>X.map(x=>[(x[0]-mu[0])/sd[0],(x[1]-mu[1])/sd[1]]);
  return (DATA[C]={C,Xtr:st(tr.X),ytr:tr.y,Xva:st(va.X),yva:va.y});
}
/* the first k training examples of each group (all of them when k is 0) */
const subset=(D,k)=>{if(!k) return {X:D.Xtr,y:D.ytr}; const X=[],y=[]; D.Xtr.forEach((x,i)=>{if(i%NTR<k){X.push(x); y.push(D.ytr[i]);}}); return {X,y};};
function initP(C){const r=mulberry(INIT_SEED), W1=[[],[]], W2=[];
  for(let j=0;j<2;j++) for(let k=0;k<HID;k++) W1[j].push(gauss(r));
  for(let k=0;k<HID;k++){W2.push([]); for(let c=0;c<C;c++) W2[k].push(0.01*gauss(r));}
  return {W1,b1:new Array(HID).fill(0),W2,b2:new Array(C).fill(0)};}
const cloneP=p=>({W1:p.W1.map(a=>a.slice()),b1:p.b1.slice(),W2:p.W2.map(a=>a.slice()),b2:p.b2.slice()});
/* the model on one point: logits */
function logits(p,x){const C=p.b2.length, z=new Array(C);
  const h=new Array(HID); for(let k=0;k<HID;k++){const a=x[0]*p.W1[0][k]+x[1]*p.W1[1][k]+p.b1[k]; h[k]=a>0?a:0;}
  for(let c=0;c<C;c++){let s=p.b2[c]; for(let k=0;k<HID;k++) s+=h[k]*p.W2[k][c]; z[c]=s;} return {h,z};}
/* mean cross-entropy over (X, y), its accuracy and (optionally) its gradient. softmax is written out without subtracting the largest
   logit. twice: the model's output goes through softmax before the loss applies its own */
function lossGrad(p,X,y,twice,needGrad){
  const C=p.b2.length, N=X.length;
  const g=needGrad?{W1:[new Array(HID).fill(0),new Array(HID).fill(0)],b1:new Array(HID).fill(0),W2:p.W2.map(()=>new Array(C).fill(0)),b2:new Array(C).fill(0)}:null;
  let Ls=0, ok=0; const q=new Array(C), dz=new Array(C);
  for(let n=0;n<N;n++){const x=X[n], {h,z}=logits(p,x);
    let se=0; for(let c=0;c<C;c++){q[c]=Math.exp(z[c]); se+=q[c];} for(let c=0;c<C;c++) q[c]/=se;
    let am=0; for(let c=1;c<C;c++) if(z[c]>z[am]) am=c; if(am===y[n]) ok++;
    if(!twice){Ls+=-Math.log(q[y[n]]); if(needGrad) for(let c=0;c<C;c++) dz[c]=(q[c]-(c===y[n]?1:0))/N;}
    else {let s2=0; const r2=new Array(C); for(let c=0;c<C;c++){r2[c]=Math.exp(q[c]); s2+=r2[c];} for(let c=0;c<C;c++) r2[c]/=s2;
      Ls+=-Math.log(r2[y[n]]);
      if(needGrad){let dot=0; const dq=new Array(C); for(let c=0;c<C;c++){dq[c]=(r2[c]-(c===y[n]?1:0))/N; dot+=dq[c]*q[c];} for(let c=0;c<C;c++) dz[c]=q[c]*(dq[c]-dot);}}
    if(needGrad){
      for(let c=0;c<C;c++){g.b2[c]+=dz[c]; for(let k=0;k<HID;k++) g.W2[k][c]+=h[k]*dz[c];}
      for(let k=0;k<HID;k++){if(h[k]<=0) continue; let s=0; for(let c=0;c<C;c++) s+=p.W2[k][c]*dz[c]; g.b1[k]+=s; g.W1[0][k]+=x[0]*s; g.W1[1][k]+=x[1]*s;}}
  }
  return {loss:Ls/N,acc:ok/N,g};
}
const each=(p,f)=>{p.W1.forEach((a,j)=>a.forEach((_,k)=>f(a,k))); p.b1.forEach((_,k)=>f(p.b1,k)); p.W2.forEach((a,k)=>a.forEach((_,c)=>f(a,c))); p.b2.forEach((_,c)=>f(p.b2,c));};
const flat=p=>{const o=[]; each(p,(a,i)=>o.push(a[i])); return o;};
/* one run. cfg: {C, lr, bug ('none'|'nozero'|'twice'|'shuffle'), k (examples per group, 0 = all), steps}. Full-batch gradient descent.
   tr[t], va[t], acc[t]: the training loss, validation loss and validation accuracy of the weights before step t (t = 0 … steps);
   gn[t]: the norm of the gradient step t uses. The run stops at the first NaN training loss (dead = that step); inf is recorded. */
const RUNS=new Map();
function run(cfg){
  const c={C:3,lr:LR0,bug:'none',k:0,steps:STEPS,...cfg}, key=[c.C,c.lr,c.bug,c.k,c.steps].join('|');
  if(RUNS.has(key)) return RUNS.get(key);
  const D=dataOf(c.C), T=subset(D,c.k), rs=mulberry(SHUF_SEED);
  const p=initP(c.C), tr=[], va=[], acc=[], gn=[], snaps={}; let G=null, dead=-1, inf=-1;
  for(let t=0;t<=c.steps;t++){
    if(SNAPS.includes(t)) snaps[t]=cloneP(p);
    let y=T.y;
    if(c.bug==='shuffle'){const pm=[...T.y.keys()]; for(let i=pm.length-1;i>0;i--){const j=Math.floor(rs()*(i+1)); const s=pm[i]; pm[i]=pm[j]; pm[j]=s;} y=pm.map(i=>T.y[i]);}
    const r=lossGrad(p,T.X,y,c.bug==='twice',t<c.steps), v=lossGrad(p,D.Xva,D.yva,c.bug==='twice',false);
    tr.push(r.loss); va.push(v.loss); acc.push(v.acc);
    if(Number.isNaN(r.loss)){dead=t; break;}
    if(inf<0&&!isFinite(r.loss)) inf=t;
    if(t===c.steps) break;
    if(c.bug==='nozero'&&G){const a=flat(G), b=flat(r.g); let i=0; each(G,(arr,j)=>{arr[j]=a[i]+b[i]; i++;});}
    else G=r.g;
    gn.push(Math.sqrt(flat(G).reduce((s,v)=>s+v*v,0)));
    const gv=flat(G); let i=0; each(p,(arr,j)=>{arr[j]-=c.lr*gv[i]; i++;});
  }
  const out={cfg:c,tr,va,acc,gn,dead,inf,snaps,last:p};
  RUNS.set(key,out); return out;
}
const lnC=C=>Math.log(C);
/* the smallest and largest chance the model gives any class on any of the points X: how close its guesses are to 1/C.
   A loss at ln C alone does not show this (chances of 2/3 and 1/6 for the right class also average ln 3) */
function chanceRange(p,X){let lo=1,hi=0; for(const x of X){const {z}=logits(p,x), m=Math.max(...z); let se=0; const e=z.map(v=>{const t=Math.exp(v-m); se+=t; return t;}); for(const t of e){const q=t/se; if(q<lo) lo=q; if(q>hi) hi=q;}} return {lo,hi};}
const lastFinite=a=>{for(let i=a.length-1;i>=0;i--) if(isFinite(a[i])) return {i,v:a[i]}; return {i:-1,v:NaN};};
/* check 1: the loss at step 0 against ln C, and whether the run ever leaves ln C */
function check1(r){const C=r.cfg.C, l=lnC(C), e=lastFinite(r.tr), start=r.tr[0];
  const stays=r.dead<0&&r.inf<0&&Math.abs(e.v-l)<0.05&&r.tr.every(v=>Math.abs(v-l)<0.1);
  return {start,lnC:l,startOk:Math.abs(start-l)<0.1,end:e.v,stays,cat:stays?'stays':'leaves'};}
/* check 2: a fresh model learns 3 examples per group with their real labels (no data loader: the shuffle bug is not on this path),
   with the run's own step and learning rate */
function check2(cfg){const c={C:3,lr:LR0,bug:'none',...cfg}, r=run({C:c.C,lr:c.lr,bug:c.bug==='shuffle'?'none':c.bug,k:OF_K,steps:OF_STEPS});
  const e=r.tr[r.tr.length-1], pass=isFinite(e)&&e<OF_PASS;
  return {r,end:e,min:Math.min(...r.tr.filter(isFinite)),pass,dead:r.dead,inf:r.inf,cat:pass?'pass':'fail'};}
/* check 3: the gradient norm over the run. nan: the loss stops being a number within 20 steps; grows: the last 50 finite steps
   average more than 1.5 times the step-0 norm; stays: they average more than 0.2 of it; fades: it shrinks toward 0 */
function check3(r){const g0=r.gn[0], fin=r.gn.filter(isFinite), tail=fin.slice(-50), tm=tail.reduce((a,b)=>a+b,0)/Math.max(1,tail.length), mx=Math.max(...fin);
  const bad=r.inf>=0?r.inf:r.dead, cat=bad>=0&&bad<=20?'nan':(tm>1.5*g0?'grows':(tm/g0>0.2?'stays':'fades'));
  return {g0,tail:tm,max:mx,end:fin[fin.length-1],bad,cat};}
/* what the curve shows on its own: it blows up, it stays at ln C, or it falls */
function curveCat(r){const l=lnC(r.cfg.C); if(r.dead>=0||r.inf>=0||Math.max(...r.tr)>1.5*l) return 'blows'; return check1(r).stays?'flat':'falls';}
/* the curve's shape, as the chart draws it (free to read in the challenge, R4.5):
   now: it blows up (leaves the chart, inf or NaN) within the first 20 steps; late: it blows up after that, having fallen first;
   flat: it never leaves ln C; slow: it is still falling (the last 100 steps give more than a tenth of the whole drop);
   stall: it levelled off far above 0 (above 0.4 × ln C); falls: it levelled off low (the healthy run) */
function curveShape(r){const l=lnC(r.cfg.C), tr=r.tr, off=tr.findIndex(v=>isFinite(v)&&v>1.5*l), bad=[r.inf,r.dead,off].filter(x=>x>=0);
  if(bad.length) return Math.min(...bad)<=20?'now':'late';
  if(check1(r).stays) return 'flat';
  const n=tr.length-1, drop=tr[0]-tr[n];
  if(tr[n-100]-tr[n]>0.1*drop) return 'slow';
  return tr[n]>0.4*l?'stall':'falls';}
/* the decision regions: the class the model predicts on a G × G grid over [−3, 3]² (null when a parameter is NaN) */
const GRID=40, EXT=3;
function regions(p){if(flat(p).some(v=>!isFinite(v))) return null; const out=[];
  for(let i=0;i<GRID;i++){const row=[]; for(let j=0;j<GRID;j++){const x=[-EXT+(j+0.5)*2*EXT/GRID,EXT-(i+0.5)*2*EXT/GRID], z=logits(p,x).z; let am=0; for(let c=1;c<z.length;c++) if(z[c]>z[am]) am=c; row.push(am);} out.push(row);}
  return out;}
/* the largest chance the model gives any class anywhere on that grid: below 0.5 the regions are noise (A19) */
function topChance(p){if(flat(p).some(v=>!isFinite(v))) return NaN; let mx=0;
  for(let i=0;i<GRID;i++) for(let j=0;j<GRID;j++){const z=logits(p,[-EXT+(j+0.5)*2*EXT/GRID,EXT-(i+0.5)*2*EXT/GRID]).z, m=Math.max(...z); let se=0; z.forEach(v=>{se+=Math.exp(v-m);}); mx=Math.max(mx,1/se);}
  return mx;}

/* the bugs, the runs the parts use, and the challenge's mystery runs */
const BUGS=['none','nozero','twice','shuffle'];
const KINDS=['high','low','nozero','twice','shuffle'];                       // the five things that can be wrong
const kindCfg=(k,C)=>({C,lr:k==='high'?10:(k==='low'?0.003:LR0),bug:['nozero','twice','shuffle'].includes(k)?k:'none'});
const kindOf=cfg=>cfg.bug!=='none'?cfg.bug:(cfg.lr===10?'high':(cfg.lr===0.003?'low':'none'));
const sigOf=(k,C)=>{const cf=kindCfg(k,C), r=run(cf); return {kind:k,curve:curveShape(r),c1:check1(r).cat,c2:check2(cf).cat,c3:check3(r).cat};};
const SETS=[[],['c3'],['c1'],['c2'],['c1','c3'],['c2','c3'],['c1','c2'],['c1','c2','c3']];
/* the fewest checks that tell the hidden bug apart from the other four, the curve's shape being free to read (R4.5) */
function minChecks(k,C){const all=KINDS.map(x=>sigOf(x,C)), me=all.find(s=>s.kind===k);
  for(const s of SETS){const same=all.filter(o=>o.kind!==k&&o.curve===me.curve&&s.every(c=>o[c]===me[c])); if(!same.length) return s;} return ['c1','c2','c3'];}
/* does this set of opened checks tell the hidden bug apart? */
const tellsApart=(k,C,opened)=>{const all=KINDS.map(x=>sigOf(x,C)), me=all.find(s=>s.kind===k); return !all.some(o=>o.kind!==k&&o.curve===me.curve&&opened.every(c=>o[c]===me[c]));};
/* the fewest checks that single out the hidden bug without the curve: what would confirm a reading of the curve */
function confirmOf(k,C){const all=KINDS.map(x=>sigOf(x,C)), me=all.find(s=>s.kind===k);
  for(const s of SETS.slice(1)){if(!all.some(o=>o.kind!==k&&s.every(c=>o[c]===me[c]))) return s;} return ['c1','c2','c3'];}
/* the first check whose result differs between two of the five (null when none does) */
const pairCheck=(a,b,C)=>{const sa=sigOf(a,C), sb=sigOf(b,C); return ['c3','c1','c2'].find(c=>sa[c]!==sb[c])||null;};
const MYST=['twice','nozero','shuffle','low','high'];                        // the challenge's mystery runs, in order; 4 groups of points

/* =====================================================================
   formatting
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const f3=v=>Number.isNaN(v)?'NaN':(!isFinite(v)?'inf':fixd(v,3));
const fPct=v=>fixd(v*100,1)+'%', fPct0=v=>fixd(v*100,0)+'%';
const fLr=v=>v===0.003?'0.003':(v===10?'10':fixd(v,1));
const fG=v=>!isFinite(v)?(Number.isNaN(v)?'NaN':'inf'):(v>=100?A.fmt.int(Math.round(v)):(v>=1?fixd(v,2):fixd(v,3)));
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
/* a term whose hover holds a formula on its own line (R6.29): the kit's .t shows termTip's HTML; it has no click action */
const DFX=(k,x)=>`<span class="t td-dfx" tabindex="0" data-ctl="dfx" data-term="${k}">${x}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const code=s=>`<code>${s}</code>`;

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={lr:LR0,bug:'none',k:0};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let stepsShown=0, stepsPrev=0, lrPick='0.3', lrTried=new Set(['0.3']), bugPick='none', bugTried=new Set(['none']), chkPick=null, chkTried=new Set();
let saved=null, sbTouched=false, userAct=false, annT=null, drillSeed=1, drillOpen=false;
let chIdx=0, chOpen=[], chAns=null, chSubmitted=false, chReveal=false;
const shownPrev={};
const cfgNow=()=>({C:3,...S});
const R=()=>run(cfgNow());

/* =====================================================================
   words and definitions (R2.8)
   ===================================================================== */
const KNAME={none:{en:'No bug',zh:'没有 bug'},high:{en:'Learning rate too high',zh:'学习率太高'},low:{en:'Learning rate too low',zh:'学习率太低'},
  nozero:{en:'zero_grad() missing',zh:'漏了 zero_grad()'},twice:{en:'Softmax applied twice',zh:'softmax 用了两次'},shuffle:{en:'Labels shuffled apart from the inputs',zh:'标签和输入分开打乱'}};
const kname=k=>L(KNAME[k]);
/* hovers with a formula on its own line (R6.29), shown through DFX */
const DFXT={
 ce:()=>`<div>${esc(tx('Cross-entropy is the loss of a classifier: the lower the chance the model gives the right answer, the higher the cost.','cross-entropy（交叉熵）是分类器的 loss：模型给正确答案的概率越低，代价越大。'))}</div>`+
   `<div class="fx">${A.ui.math(tx('L = -\\ln\\, p_{\\text{right class}}','L = -\\ln\\, p_{\\text{正确类}}'))}</div>`+
   `<div>${esc(tx('Then it is averaged over the examples. A 100% chance for the right class costs 0. A chance of 1/3 costs','再对所有样本求平均。给正确类的概率是 100% 时，代价是 0。只给 1/3 时，代价是'))} ${A.ui.math('-\\ln\\, \\frac{1}{3} = 1.099')}${tx('.','。')}</div>`,
 lnc:()=>`<div>${esc(tx('If a model gives each of the C classes the same chance, 1/C, its cross-entropy is ln C:','如果模型给 C 个类别的概率都是 1/C，它的 cross-entropy 就是 ln C：'))}</div>`+
   `<div class="fx">${A.ui.math('-\\ln\\, \\frac{1}{C} = \\ln\\, C')}</div>`+
   `<div>${esc(tx('With 3 classes it is 1.099, with 4 classes 1.386. A freshly initialized classifier should start near it.','3 个类别时是 1.099，4 个类别时是 1.386。刚初始化的分类器应该从这附近开始。'))}</div>`
};
const TIP={
 lr:()=>tx('The learning rate sets how big each training step is: at every step, each weight goes down by the learning rate times its gradient.',
           '学习率决定每一步迈多大：每一步，权重都减去“学习率 × 梯度”。'),
 gn:()=>tx('The gradient norm is the length of the gradient the step uses, over all the model’s parameters: the square root of the sum of every gradient value squared. PyTorch’s clip_grad_norm_ returns it.',
           '梯度范数是这一步用到的梯度的长度，把模型所有参数的梯度放在一起算：每个梯度值平方、求和、再开方。PyTorch 的 clip_grad_norm_ 会返回它。'),
 zg:()=>tx('In PyTorch, loss.backward() adds each new gradient to the one already stored in every parameter’s .grad. optimizer.zero_grad() clears it before the next backward pass; without it, every step uses the sum of all the gradients so far.',
           '在 PyTorch 里，loss.backward() 会把新的梯度加到每个参数的 .grad 里已有的梯度上。optimizer.zero_grad() 在下一次反向传播之前把它清零；漏了它，每一步用的都是到目前为止所有梯度的和。'),
 val:()=>tx('The validation set is 150 points the model never trains on (50 per group). Its loss shows whether what the model learned holds for new points.',
            '验证集是模型从不拿来训练的 150 个点（每组 50 个）。它上面的 loss 说明模型学到的东西对新的点是否也成立。'),
 overfit:()=>tx('Overfitting: the model fits its training examples ever better while new examples get worse, because it learns their noise instead of the pattern. The training loss falls while the validation loss rises.',
                '过拟合：模型越来越贴合训练样本，新样本上却越来越差，因为它学到的是这些样本的噪声，而不是规律。训练 loss 下降，验证集 loss 上升。'),
 logit:()=>tx('A logit is the model’s raw score for one class, before softmax. Softmax turns the scores of all the classes into chances that add up to 100%.',
              'logit 是模型给某个类别的原始分数，还没经过 softmax。softmax 把所有类别的分数变成加起来是 100% 的概率。'),
 illus:(n=3*NTR)=>tx(`Everything here is computed in the page: a network with 16 hidden units really trains on ${n} points with plain gradient descent, seeded, so the page shows the same runs every time. Real models and datasets are far larger; the three checks carry over, but some curves change with other optimizers (see the notes at the end of the page).`,
              `这里的一切都在页面里实际算出：一个有 16 个隐藏单元的网络，用普通的梯度下降在 ${n} 个点上真实训练，随机数带种子，所以页面每次显示同样的运行。真实的模型和数据大得多；三个检查照样适用，但换了优化器，有些曲线会变（见页末的“简化、假设和来源”）。`)
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'shuffle',t:{en:'The labels are shuffled apart from the inputs',zh:'标签和输入被分开打乱了'}},
  {v:'high',t:{en:'The learning rate is far too high',zh:'学习率太高了'}},
  {v:'nozero',t:{en:'zero_grad() is missing, so the gradients add up',zh:'漏了 zero_grad()，梯度越加越大'}}];
const PRED_RIGHT='shuffle';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS_[step]&&STEPS_[step].id===id;
/* the run each part shows */
const LRV=[{v:'0.003',lr:0.003,t:{en:'0.003: 100 times lower',zh:'0.003：调低到 1/100'},d:{en:'Someone lowered it to be safe.',zh:'有人为了稳妥，把学习率调得很低。'}},
  {v:'0.3',lr:0.3,t:{en:'0.3: the initial setup',zh:'0.3：初始设置'},d:{en:'The healthy run, drawn as the thin line in the chart.',zh:'健康的运行，就是图里的细线。'}},
  {v:'10',lr:10,t:{en:'10: about 33 times higher',zh:'10：调高到约 33 倍'},d:{en:'Someone raised it to train faster.',zh:'有人想训练得快一点，把学习率调得很高。'}}];
const BGV=[{v:'none',t:{en:'No bug (initial setup)',zh:'没有 bug（初始设置）'},d:{en:'The healthy run, drawn as the thin line in the chart.',zh:'健康的运行，就是图里的细线。'}},
  {v:'nozero',t:{en:'zero_grad() missing',zh:'漏了 zero_grad()'},d:{en:'The script lacks the line that clears the gradient at the start of each step.',zh:'脚本里少了每一步开头清零梯度的那一行。'}},
  {v:'twice',t:{en:'Softmax applied twice',zh:'softmax 用了两次'},d:{en:'The model applies a softmax to its output, and the loss function applies its own again.',zh:'模型输出前多做了一次 softmax，loss 函数自己还会再做一次。'}},
  {v:'shuffle',t:{en:'Labels shuffled apart from the inputs',zh:'标签和输入分开打乱'},d:{en:'The data loader shuffles the inputs and the labels in two different orders, so the labels no longer match the inputs.',zh:'数据加载器把输入和标签各自打乱，顺序不一样，标签就配错了输入。'}}];
/* the three checks, named by what they do (R6.32): what is run, and what a healthy result looks like */
const CKV=[{v:'c1',t:{en:'1. Does the loss stay at ln C?',zh:'1. loss 是不是停在 ln C'},d:{en:'The loss should start near ln 3 = 1.099. If it never leaves it, the model is no better than guessing 1/3 for each class.',zh:'一开始的 loss 应该接近 ln 3 = 1.099。如果之后一直停在那里，模型就没比“每类都猜 1/3”好。'}},
  {v:'c2',t:{en:'2. Train on just 9 examples',zh:'2. 只用 9 个样本训练'},d:{en:'A fresh model learns 9 examples with their real labels for 200 steps (the usual name: “overfit a tiny batch”). If the code is fine, its loss gets close to 0.',zh:'让一个新模型只学 9 个带正确标签的样本，学 200 步（业内叫“过拟合一个小 batch”）。代码没问题的话，它的 loss 会降到接近 0。'}},
  {v:'c3',t:{en:'3. Watch the gradient’s size',zh:'3. 看梯度有多大'},d:{en:'Normally the gradient gets smaller as the loss falls. If it does not, something is wrong.',zh:'正常时，梯度会随着 loss 下降而变小。不是这样，就说明出了问题。'}}];
const mkLever=(id,list,val,res)=>`<div class="td-lv" id="${id}">`+A.ui.levers({aria:tx('Pick one','选一个'),value:val,options:list.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(res(o.v))}))})+`</div>`;
const endWord=r=>r.dead>=0?tx(`NaN at step ${r.dead}`,`第 ${r.dead} 步变成 NaN`):(r.inf>=0?tx(`inf at step ${r.inf}`,`第 ${r.inf} 步变成 inf`):tx(`${f3(r.tr[r.tr.length-1])} at step ${STEPS}`,`第 ${STEPS} 步 ${f3(r.tr[r.tr.length-1])}`));
const lrRes=v=>lrTried.has(v)||v===lrPick?endWord(run({C:3,lr:+v})):'';
const bugRes=v=>bugTried.has(v)||v===bugPick?endWord(run({C:3,bug:v})):'';
const CK_CATCH={c1:['shuffle'],c2:['high','low','nozero','twice'],c3:['high','low','nozero','twice','shuffle']};
const ckRes=v=>chkTried.has(v)?(v==='c3'?tx('tells 4 apart','分开 4 种'):tx(`flags ${CK_CATCH[v].length} of 5`,`5 种里标出 ${CK_CATCH[v].length} 种`)):'';
const stepsCtl=()=>A.ui.seg({id:'td-ic-steps',key:'steps',label:tx('Steps trained','训练了几步'),options:SNAPS.map(s=>[String(s),String(s)]),value:String(stepsShown),inline:true});
const STEPS_=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('A loss stuck at 1.10 for 300 steps: what does 1.10 mean?','loss 停在 1.10 整整 300 步。1.10 说明了什么？'),
  enter:()=>({lr:LR0,bug:'shuffle',k:0}),
  body:()=>{const r=run({C:3,bug:'shuffle'}); return tx(
   `<p>A teammate trains a small classifier to sort 2-D points into 3 classes (the circles, squares and triangles in the chart). Its training loss starts at ${f3(r.tr[0])} and is still ${f3(r.tr[STEPS])} after ${STEPS} steps.</p><p>That is almost exactly ${DFX('lnc','ln 3 = 1.099')} (the dotted line). The loss here is the ${DFX('ce','cross-entropy')}, and a model that answers “1/3 for each class” for every point scores exactly ln 3.</p><p>So after ${STEPS} steps the model is still no better than that guess. Something in the training script is wrong. What?</p>`,
   `<p>同事训练了一个小分类器，要把平面上的点分成 3 类（图里的圆、方块和三角）。它的训练 loss 一开始是 ${f3(r.tr[0])}，训练了 ${STEPS} 步还是 ${f3(r.tr[STEPS])}，几乎没动。</p><p>这个数非常接近 ${DFX('lnc','ln 3 = 1.099')}（图里的点线）。这里的 loss 是 ${DFX('ce','cross-entropy')}，如果模型对每个点都回答“三类各 1/3”，它的 loss 正好是 ln 3。</p><p>也就是说，训练了 ${STEPS} 步，模型还没比“每类都猜 1/3”好。训练脚本里一定有地方写错了。错在哪里？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: which bug keeps the loss at ln 3 from the first step to the last?','先猜：哪个 bug 让 loss 从第一步到最后一步都停在 ln 3？'),
  enter:()=>({lr:LR0,bug:'shuffle',k:0}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Here are three common bugs. Which one keeps the loss at ln 3 = 1.099 from the first step to the last? Pick one.</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 5 shows the result.</p>':''}`,
             `<p>下面是三个常见的 bug。哪一个会让 loss 从头到尾都停在 ln 3 = 1.099？选一个。</p>${ch}${pred?'<p class="muted small">已记下。到第 5 部分揭晓之前，你还可以改。</p>':''}`);}},
 {id:'healthy',title:{en:'A healthy run first',zh:'先看健康的运行'},
  lead:()=>tx('The same script without a bug: watch the loss fall and the regions form.','同一个脚本去掉 bug：看 loss 怎么降，预测区域怎么成形。'),
  enter:()=>{stepsShown=0; return {...BASE};},expect:()=>stepsShown===STEPS,showMe:()=>setSteps(STEPS),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Before hunting the bug, see what training looks like without one. The model and the data are the teammate’s; the ${T_('lr','learning rate')} is 0.3. At every step the model computes the gradient of the loss on all 300 training points and moves its weights a little in the direction that lowers the loss. Pick a step to see the run up to there:</p>`,
              `<p>找 bug 之前，先看看没有 bug 时训练是什么样子。模型和数据都和同事的一样，${T_('lr','学习率')}是 0.3。每一步，模型都用全部 300 个训练点算一次梯度，再把权重往让 loss 变小的方向挪一点。选一个步数，看训练到那里时的样子：</p>`)+stepsCtl(),
  obs:()=>obsHealthy()},
 {id:'lr',title:{en:'Learning rate: too high, too low',zh:'学习率：太高，太低'},
  lead:()=>tx('Two settings, two signatures: a jump to NaN, and a slow, steady fall.','两个设置，两种症状：跳到 NaN，和一条慢慢、稳稳下降的线。'),
  enter:()=>{lrPick='0.3'; lrTried=new Set(['0.3']); return {...BASE};},expect:()=>lrTried.has('10')&&lrTried.has('0.003'),showMe:()=>{setLr(['10','0.003'].find(v=>!lrTried.has(v))||'10');},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Nothing in the code changes here, only the ${T_('lr','learning rate')}. The thin line in the chart is the healthy run at learning rate 0.3, for comparison.</p><p>Pick a learning rate and see how its loss curve differs from the thin line. Try both 0.003 and 10.</p>`,
              `<p>这一部分代码一行不改，只改${T_('lr','学习率')}。图里的细线是学习率 0.3 时健康的运行，用来对照。</p><p>选一个学习率，看看 loss 曲线和细线差在哪。0.003 和 10 都要试一下。</p>`)+mkLever('td-ic-lr',LRV,lrPick,lrRes),
  obs:()=>obsLr()},
 {id:'bugs',title:{en:'Three bugs in the code',zh:'代码里的三个 bug'},
  lead:()=>tx('Each bug bends the curve its own way; the healthy run is the thin line.','每个 bug 让曲线弯成自己的样子；细线是健康的运行。'),
  enter:()=>{bugPick='none'; bugTried=new Set(['none']); return {...BASE};},expect:()=>['nozero','twice','shuffle'].every(v=>bugTried.has(v)),showMe:()=>{setBug(['nozero','twice','shuffle'].find(v=>!bugTried.has(v))||'shuffle');},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The learning rate is back at 0.3. Now one line of the script is wrong.</p><p>Pick a way it can be wrong and see how its loss curve differs from the thin line. Try all three.</p>`,
              `<p>学习率回到 0.3。现在脚本里有一行写错了。</p><p>选一种写错的方式，看看 loss 曲线和细线差在哪。三种都要试一下。</p>`)+mkLever('td-ic-bug',BGV,bugPick,bugRes),
  obs:()=>obsBugs()},
 {id:'checks',title:{en:'Three quick checks',zh:'三个快速检查'},
  lead:()=>tx('Each check takes seconds and catches different bugs.','三个检查各只要几秒，能认出的问题各不相同。'),
  enter:()=>{chkPick=null; chkTried=new Set(); return {...BASE};},expect:()=>chkTried.size>=2,showMe:()=>{setChk(['c1','c3','c2'].find(v=>!chkTried.has(v))||'c2');},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The curve alone often narrows the bug down to two or three. Three quick checks, a few seconds each, tell you which one. The table in the chart runs every check on six runs: the healthy one, two with a bad learning rate and three with a wrong line in the script. Pick a check to see which problems it catches:</p>`,
              `<p>光看曲线，常常只能把范围缩小到两三个 bug。下面三个检查各只要几秒，能帮你确定是哪一个。图里的表格把每个检查都在六次运行上跑了一遍：一次健康的，两次学习率不对的，三次脚本里有一行写错的。选一个检查，看它能认出哪些问题：</p>`)+mkLever('td-ic-chk',CKV,chkPick,ckRes),
  obs:()=>obsChecks()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Compare with ln C, overfit a tiny batch, watch the gradient norm.','先和 ln C 比，再过拟合一个小 batch，再看梯度范数。'),
  enter:()=>({...BASE}),
  body:()=>remember()}
];
const SIX=id=>STEPS_.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsHealthy(){
  const r=run({C:3}), t=stepsShown, l=r.tr[t];
  if(t===0) return tx(`<p><b>At step 0 the untrained model’s loss is ${f3(l)}, close to ln 3 = 1.099.</b> Its last layer’s weights start near 0, so it guesses about 1/3 for each class at every point. A correct start looks like this. Move on through the run.</p>`,
                      `<p><b>第 0 步，还没训练的模型 loss 是 ${f3(l)}，接近 ln 3 = 1.099。</b>它最后一层的权重一开始接近 0，所以对每个点都猜三类各约 1/3。正常的开头就该是这样。接着往后走。</p>`);
  if(t<STEPS) return tx(`<p><b>After ${t} steps the training loss is down to ${f3(l)} (it started at ${f3(r.tr[0])}).</b> The regions in the chart already split the plane between the 3 classes. Go on to step ${STEPS}.</p>`,
                        `<p><b>训练 ${t} 步后，训练 loss 降到 ${f3(l)}（一开始是 ${f3(r.tr[0])}）。</b>图里的预测区域已经把平面分给了 3 类。继续走到第 ${STEPS} 步。</p>`);
  return tx(`<p><b>After ${STEPS} steps the training loss has fallen from ${f3(r.tr[0])} to ${f3(l)}, and ${fPct(r.acc[STEPS])} of the validation points are classified right.</b></p><p>The ${DEF('validation',TIP.val())} loss (dashed) falls with it and then stays flat, so the model learned where the 3 classes lie, not the 300 points by heart. Every other run is compared with this curve: a fast drop, then a slow tail.</p>`,
            `<p><b>训练 ${STEPS} 步后，训练 loss 从 ${f3(r.tr[0])} 降到了 ${f3(l)}，验证集里 ${fPct(r.acc[STEPS])} 的点分对了。</b></p><p>${DEF('验证集',TIP.val())}的 loss（虚线）也跟着降下来，之后基本不变。所以模型学到的是 3 类点各自在哪，而不是死记这 300 个点。后面每次运行都拿这条曲线对照：先快速下降，再慢慢变平。</p>`);
}
function obsLr(){
  const h=run({C:3}), r=run({C:3,lr:+lrPick});
  if(lrPick==='0.3') return tx(`<p><b>Learning rate 0.3 is the healthy run: its loss is ${f3(h.tr[STEPS])} at step ${STEPS}.</b> Pick the higher or the lower rate to compare.</p>`,`<p><b>学习率 0.3 就是那次健康的运行，第 ${STEPS} 步的 loss 是 ${f3(h.tr[STEPS])}。</b>选一个更高或更低的学习率来对比。</p>`);
  if(lrPick==='10'){const pk=Math.max(...r.tr.filter(isFinite)), c3=check3(r);
    const pi=r.tr.indexOf(pk);
    return tx(`<p><b>At learning rate 10 the loss shoots out of the chart within a few steps and turns NaN at step ${r.dead}.</b> The healthy run (thin line) ends at ${f3(h.tr[STEPS])}.</p><p>Why: every step is 33 times longer than in the healthy run, so it jumps right over the lowest point of the loss and lands higher. The next gradient is larger, so the next step is longer still, and the run flies further away each time. On the way the gradient norm climbs from ${fG(c3.g0)} to ${fG(c3.max)}.</p>`+
               `<details class="td-more"><summary>Why inf first, then NaN?</summary><p>Every overshoot makes the logits larger. The loss turns inf at step ${r.inf}: a wrong class’s logit leads the right class’s by more than about 745, so the right class’s chance rounds to 0. It jumps to ${f3(pk)} at step ${pi}. It turns NaN at step ${r.dead}: a logit passes about 709, exp overflows float64, and every number after that is NaN.</p></details>`,
              `<p><b>学习率为 10 时，loss 几步之内就冲出了图，第 ${r.dead} 步变成 NaN。</b>健康的运行（细线）最后降到 ${f3(h.tr[STEPS])}。</p><p>原因：每一步的长度是健康时的 33 倍，一下就跨过了 loss 的最低点，落到更高的地方。下一步的梯度更大，步子也跟着更大，于是越冲越远。一路上，梯度范数从 ${fG(c3.g0)} 涨到了 ${fG(c3.max)}。</p>`+
               `<details class="td-more"><summary>为什么先是 inf，再是 NaN？</summary><p>每冲过头一次，logit 就变大一些。第 ${r.inf} 步，某个错误类别的 logit 比正确类别的高出约 745 以上，正确类别的概率被舍入成 0，loss 变成 inf。第 ${pi} 步 loss 跳到 ${f3(pk)}。第 ${r.dead} 步，某个 logit 超过约 709，exp 在 float64 里溢出，之后算出的数全是 NaN。</p></details>`);}
  const c3=check3(r);
  return tx(`<p><b>At learning rate 0.003 the loss keeps falling, slowly: it is down to ${f3(r.tr[STEPS])} at step ${STEPS} and still falling.</b> The healthy run is at ${f3(h.tr[STEPS])} by then.</p><p>Why: the gradient points the right way and stays large, but each step is only a hundredth of the healthy one. Nothing in the code is broken: many more steps, or a larger rate, would bring the loss down.</p><p>Its validation accuracy (${fPct(r.acc[STEPS])}) is even a little above the healthy run’s (${fPct(h.acc[STEPS])}). Accuracy only asks whether the right class is on top; the loss also counts how sure the model is. So it is the loss that tells these two runs apart.</p>`,
            `<p><b>学习率为 0.003 时，loss 一直在慢慢下降，到第 ${STEPS} 步才降到 ${f3(r.tr[STEPS])}，还在往下走。</b>健康的运行这时已经是 ${f3(h.tr[STEPS])}。</p><p>原因：梯度的方向是对的，大小也一直不小，只是每一步只迈了健康时的 1/100。代码什么都没坏，多训练很多步，或者把学习率调大，loss 就能降下去。</p><p>有意思的是，它的验证集准确率（${fPct(r.acc[STEPS])}）比健康的运行（${fPct(h.acc[STEPS])}）还高一点。准确率只看正确类别是不是排第一，loss 还看模型有多确定。所以要分辨这两次运行，得看 loss。</p>`);
}
function obsBugs(){
  const h=run({C:3}), r=run({C:3,bug:bugPick});
  if(bugPick==='none') return tx(`<p><b>No bug: this is the healthy run, with a loss of ${f3(h.tr[STEPS])} at step ${STEPS}.</b> Pick a bug to compare.</p>`,`<p><b>没有 bug：这就是那次健康的运行，第 ${STEPS} 步的 loss 是 ${f3(h.tr[STEPS])}。</b>选一个 bug 来对比。</p>`);
  if(bugPick==='nozero'){const c3=check3(r);
    return tx(`<p><b>Without zero_grad() the loss first falls even faster than the healthy run, then swings, and turns NaN at step ${r.dead}.</b> The validation loss (dashed) starts climbing much earlier.</p><p>Why: ${DEF('backward() adds each new gradient to the one left from the last step',TIP.zg())}, and zero_grad() is the line that clears the old one first. Without it, every step uses the sum of all the gradients so far, so the steps keep growing until they overshoot. That is why the gradient norm grows (to a peak of ${fG(c3.max)}) instead of shrinking.</p>`,
              `<p><b>漏了 zero_grad() 时，loss 一开始降得比健康的运行还快，后来却来回摆动，第 ${r.dead} 步变成 NaN。</b>验证集的 loss（虚线）更早就开始往上走了。</p><p>原因：${DEF('backward() 会把新算出的梯度加到上一步留下的梯度上',TIP.zg())}，zero_grad() 就是先把旧梯度清零的那一行。少了它，每一步用的都是之前所有梯度的总和，步子越迈越大，最后冲过了头。所以梯度范数不是越来越小，而是一路变大（峰值 ${fG(c3.max)}）。</p>`);}
  if(bugPick==='twice'){const floor=Math.log(1+2/Math.E);
    return tx(`<p><b>With softmax applied twice, the loss drops fast, then stalls at ${f3(r.tr[STEPS])}.</b> The healthy run reaches ${f3(h.tr[STEPS])}. Still, ${fPct(r.acc[STEPS])} of the validation points are classified right, because accuracy only asks whether the right class is on top.</p><p>Why: the loss function applies its own softmax. It should get logits, but it gets chances already squeezed between 0 and 1. Then the right class’s chance can never pass 57.6%, so the loss can never go below ${f3(floor)}. Near that floor the gradient is almost 0, and training stalls.</p>`,
              `<p><b>softmax 用了两次时，loss 先快速下降，然后停在 ${f3(r.tr[STEPS])}，降不下去了。</b>健康的运行能降到 ${f3(h.tr[STEPS])}。不过验证集里仍有 ${fPct(r.acc[STEPS])} 的点分对了，因为准确率只看正确类别是不是排第一。</p><p>原因：loss 函数自己会做一次 softmax。它本该拿到 logit，现在拿到的却是已经压到 0 到 1 之间的概率。这样一来，正确类别的概率最高只能到 57.6%，loss 也就永远低不过 ${f3(floor)}。loss 接近这个下限时，梯度几乎变成 0，训练就停住了。</p>`);}
  return tx(`<p><b>With the labels shuffled, the loss stays at ln 3 from the first step to the last (${f3(r.tr[0])} at step 0 and ${f3(r.tr[STEPS])} at step ${STEPS}). This is the teammate’s run.</b></p>`,`<p><b>标签被打乱时，loss 从头到尾都停在 ln 3 附近（第 0 步是 ${f3(r.tr[0])}，第 ${STEPS} 步还是 ${f3(r.tr[STEPS])}）。这正是同事的那次运行。</b></p>`)+verdict()+
    tx(`<p>Why: at every step each input gets a random label, so the inputs say nothing about their labels. The best the model can do is guess 1/3 for each class at every point, and that costs exactly ln 3.</p>`,`<p>原因：每一步，每个输入都被配上一个随机的标签，输入和标签之间没有任何关系。模型能做到的最好，就是对每个点都猜三类各 1/3，而这样的 loss 正好是 ln 3。</p>`);
}
function obsChecks(){
  if(!chkPick) return '';
  if(chkPick==='c1') return tx(`<p><b>Check 1 flags only the shuffled labels: all six runs start near ln 3, and only that one never leaves it.</b></p><p>A loss that stays at ln C means the model is no better than guessing 1/C for each class. The usual causes are labels that do not match their inputs, or gradients that never reach the weights.</p>`,
                                   `<p><b>检查 1 只标出了打乱的标签：六次运行都从 ln 3 附近开始，只有它一直没离开。</b></p><p>loss 一直停在 ln C，说明模型没比“每类都猜 1/C”好。常见的原因是标签和输入对不上，或者梯度根本没传到权重。</p>`);
  if(chkPick==='c2') return tx(`<p><b>Of the five broken runs, only the one with shuffled labels passes check 2.</b></p><p>The check skips the data loader: it hands 9 examples with their real labels straight to a fresh model, with the run’s own step and learning rate. Working code learns 9 examples easily. If the model cannot, the bug is in the model, the step or the learning rate; if it can, the bug is in how the data reaches the model.</p><p class="small">The table’s column for check 2 shows the loss on those 9 examples after 200 steps; below 0.1 counts as learned.</p>`,
                                   `<p><b>五次弄坏的运行里，只有打乱标签的那次通过了检查 2。</b></p><p>这个检查跳过数据加载器，直接把 9 个带正确标签的样本交给一个新模型，训练步和学习率都和原来一样。正常的代码很容易把 9 个样本学会。连这都学不会，问题就在模型、训练步或学习率里；学会了，问题就在数据送进模型的路上。</p><p class="small">表格里检查 2 那一列的数，是这 9 个样本训练 200 步后的 loss，低于 0.1 算学会了。</p>`);
  return tx(`<p><b>Check 3 watches the gradient’s size next to the loss, and that tells apart the four problems in the model, the step or the learning rate.</b></p><p>For example, at learning rate 0.003 the loss is still high but the gradient stays large: the direction is right, the steps are just too short.</p>`+
            `<details class="td-more"><summary>What the gradient does with each problem</summary><ul><li>Learning rate 10: it explodes within a few steps.</li><li>Learning rate 0.003: it never fades.</li><li>zero_grad() missing: it keeps growing.</li><li>Softmax twice: it is nearly 0 while the loss is still high.</li></ul><p>With shuffled labels the gradient does not fade either. That run is told apart by check 1: its loss stays flat at ln 3.</p></details>`,
            `<p><b>检查 3 看梯度怎么变，再对照 loss，就能把模型、训练步和学习率上的四种问题一一分开。</b></p><p>比如学习率为 0.003 时，loss 还很高，梯度却一直不小：方向没错，只是步子太短。</p>`+
            `<details class="td-more"><summary>四种问题下，梯度各是什么样</summary><ul><li>学习率 10：几步之内就爆炸。</li><li>学习率 0.003：一直不变小。</li><li>漏了 zero_grad()：越来越大。</li><li>softmax 用了两次：loss 还很高，梯度就已经接近 0。</li></ul><p>标签被打乱时，梯度也不变小。认出那一次靠的是检查 1：它的 loss 一直平在 ln 3。</p></details>`);
}
function remember(){
  const sh=run({C:3,bug:'shuffle'}), lo=run({C:3,lr:0.003}), tw=run({C:3,bug:'twice'}), hi=run({C:3,lr:10}), nz=run({C:3,bug:'nozero'}), of=run({C:3,k:SMALL_K});
  const ofMin=Math.min(...of.va), shR=chanceRange(sh.last,dataOf(3).Xtr);
  return tx(`<ul class="td-remlist"><li><b>Compare the loss with ln C first.</b> A loss that stays at ln C means the model is no better than guessing 1/C for each class: something stops it from learning at all. The loss alone does not prove the model guesses 1/C everywhere, so look at its chances and its validation accuracy too.</li>`+
    `<li><b>Train on a handful of examples to tell a code bug from a data bug.</b> When the model, the step or the learning rate is broken, the model cannot learn even a few examples with their real labels. When it learns them, the problem is in how the data reaches it.</li>`+
    `<li><b>Watch the gradient’s size, not only the loss.</b> A loss that ends high can mean opposite things. If the gradient stays large, the direction is right and the steps are just too short, as with too low a learning rate. If the gradient is nearly 0 while the loss is still high, training has stalled, as with softmax applied twice.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “A flat loss means the learning rate is too low.” Too low a learning rate makes the loss fall slowly, but it keeps falling. A loss flat at ln C means the model is no better than guessing 1/C for each class, as with the shuffled labels.</p><p>✗ “A NaN means bad data.” The data in this lab never changed, yet a learning rate that is too high and a missing zero_grad() both end in NaN, because both make the steps too big.</p><p>✗ “Validation loss rising while the training loss falls is a bug in the code.” The scenario “Only 15 training examples” runs perfectly correct code. Its training loss falls to ${f3(of.tr[STEPS])}, while its validation loss climbs back from its low of ${f3(ofMin)} to ${f3(of.va[STEPS])}. That is ${DEF('overfitting',TIP.overfit())}: too few examples, so the model learns them by heart. It is not a bug.</p></div>${A.shell.toScenarios()}`,
    `<ul class="td-remlist"><li><b>先拿 loss 和 ln C 比。</b>loss 一直停在 ln C，说明模型没比“每类都猜 1/C”好，有什么东西让它完全学不到。光看 loss 还不能断定模型真在每个点上都猜 1/C，最好再看一眼它给的概率和验证集准确率。</li>`+
    `<li><b>只拿几个样本训练一下，就能分清是代码的问题还是数据的问题。</b>模型、训练步或学习率出了问题时，模型连几个带正确标签的样本都学不会。学得会，问题就出在数据送进模型的路上。</li>`+
    `<li><b>除了 loss，也要看梯度有多大。</b>loss 最后还很高，可能说明两种正好相反的情况。梯度一直不小，说明方向是对的，只是步子太短，比如学习率太低。loss 还很高，梯度却已经接近 0，说明训练停住了，比如 softmax 用了两次。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “loss 不动了，就是学习率太低。”学习率太低时，loss 降得慢，但一直在降。loss 平平地停在 ln C，说明模型没比“每类都猜 1/C”好，比如标签被打乱了。</p><p>✗ “出现 NaN，就是数据有问题。”这个 Lab 里的数据从来没变过，学习率太高和漏了 zero_grad() 却都会让 loss 变成 NaN，因为两者都让步子迈得太大。</p><p>✗ “训练 loss 在降、验证集 loss 在升，就是代码有 bug。”应用场景“只有 15 个训练样本”用的是完全正确的代码。它的训练 loss 降到 ${f3(of.tr[STEPS])}，验证集 loss 却从最低的 ${f3(ofMin)} 回升到 ${f3(of.va[STEPS])}。这是${DEF('过拟合',TIP.overfit())}：样本太少，模型把它们背了下来，并不是 bug。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'healthy',t:{en:'A healthy run',zh:'一次健康的运行'},d:{en:'No bug: the reference every other curve is read against',zh:'没有 bug，其他运行都拿它对照'},s:{lr:0.3,bug:'none',k:0}},
 {id:'high',t:{en:'NaN after a learning-rate change',zh:'改了学习率之后出现 NaN'},d:{en:'Someone raised the learning rate to train faster',zh:'有人为了训练更快调高了学习率'},s:{lr:10,bug:'none',k:0}},
 {id:'nozero',t:{en:'A refactor dropped one line',zh:'重构时删掉了一行'},d:{en:'optimizer.zero_grad() fell out of the training loop',zh:'训练循环里少了 optimizer.zero_grad()'},s:{lr:0.3,bug:'nozero',k:0}},
 {id:'twice',t:{en:'The loss stalls far above 0',zh:'loss 停在离 0 很远的地方'},d:{en:'The model’s forward pass ends with a softmax',zh:'模型输出前多做了一次 softmax'},s:{lr:0.3,bug:'twice',k:0}},
 {id:'shuffle',t:{en:'The loss sits at 1.10 from the first step',zh:'loss 从第一步起就停在 1.10'},d:{en:'A new data loader shuffles inputs and labels separately',zh:'新的数据加载器把输入和标签分开打乱'},s:{lr:0.3,bug:'shuffle',k:0}},
 {id:'small',t:{en:'Only 15 training examples',zh:'只有 15 个训练样本'},d:{en:'Correct code, but only 5 examples per class',zh:'代码完全正确，但每类只有 5 个样本'},s:{lr:0.3,bug:'none',k:SMALL_K}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.lr===S.lr&&p.s.bug===S.bug&&p.s.k===S.k);
function scenHTML(i){
  const p=PRESETS[i], r=run({C:3,...p.s}), h=run({C:3}), c1=check1(r), c2=check2({C:3,...p.s}), c3=check3(r);
  /* the result in one sentence with its subject (R8.6): whose loss, at which step */
  const hE=f3(h.tr[STEPS]), vm=Math.min(...r.va);
  const head={
   healthy:tx(`<p><b>Without a bug, the training loss falls from ${f3(r.tr[0])} to ${f3(r.tr[STEPS])} by step ${STEPS}.</b></p>`,`<p><b>没有 bug 时，训练 loss 从 ${f3(r.tr[0])} 降到第 ${STEPS} 步的 ${f3(r.tr[STEPS])}。</b></p>`),
   high:tx(`<p><b>With the learning rate raised to 10, the loss shoots out of the chart within a few steps and turns NaN at step ${r.dead}.</b> The healthy run ends at ${hE}.</p>`,`<p><b>学习率调到 10 后，loss 几步之内就冲出了图，第 ${r.dead} 步变成 NaN。</b>健康的运行最后是 ${hE}。</p>`),
   nozero:tx(`<p><b>Without zero_grad(), the loss falls at first, then swings and turns NaN at step ${r.dead}.</b> The healthy run ends at ${hE}.</p>`,`<p><b>少了 zero_grad() 后，loss 先降后摆，第 ${r.dead} 步变成 NaN。</b>健康的运行最后是 ${hE}。</p>`),
   twice:tx(`<p><b>With the extra softmax, the training loss stalls at ${f3(r.tr[STEPS])} and goes no lower.</b> The healthy run ends at ${hE}.</p>`,`<p><b>多做一次 softmax 后，训练 loss 停在 ${f3(r.tr[STEPS])}，再也降不下去。</b>健康的运行最后是 ${hE}。</p>`),
   shuffle:tx(`<p><b>With the labels shuffled, the training loss stays at ${f3(r.tr[STEPS])}, right on ln 3, from the first step to the last.</b></p>`,`<p><b>标签被打乱后，训练 loss 从头到尾都是 ${f3(r.tr[STEPS])}，正好停在 ln 3。</b></p>`),
   small:tx(`<p><b>Trained on only 15 examples, the model’s training loss is ${f3(r.tr[STEPS])} at step ${STEPS}; the healthy run, trained on 300, is at ${hE}.</b></p>`,`<p><b>只用 15 个样本训练时，第 ${STEPS} 步的训练 loss 是 ${f3(r.tr[STEPS])}；用 300 个样本的健康运行是 ${hE}。</b></p>`)}[p.id];
  const why={
   healthy:tx(`<p>All three checks pass. The loss starts near ln 3; trained on just 9 examples, the model gets their loss down to ${f3(c2.end)}; and the gradient shrinks as the loss falls.</p>`,`<p>三个检查都通过了。第 0 步的 loss 接近 ln 3；只用 9 个样本训练时，模型能把 loss 降到 ${f3(c2.end)}；梯度也随着 loss 一起变小。</p>`),
   high:tx(`<p>The learning rate is too high: every step overshoots and the next one flies further, until every number is NaN. Check 3 catches it first: the gradient explodes within a few steps. The fix is a lower rate. To find the largest rate that still trains well, run a learning-rate range test: start from a tiny rate, raise it a little every step, and see where the loss starts to get worse (Smith, 2017).</p>`,`<p>学习率太高，每一步都冲过了头，而且越冲越远，最后算出的数全变成 NaN。检查 3 最早发现它：梯度几步之内就爆炸了。办法是把学习率调低。想找到还能稳定训练的最大学习率，可以做一次学习率范围测试：从很小的学习率开始，每一步把它调大一点，看 loss 从哪里开始变差（Smith，2017）。</p>`),
   nozero:tx(`<p>Without zero_grad(), each step’s gradient piles onto all the earlier ones, so the steps keep growing. Check 3 shows it: the gradient grows instead of shrinking. The fix is to call optimizer.zero_grad() before loss.backward() in every step.</p>`,`<p>少了 zero_grad()，每一步的梯度都叠加在之前所有梯度上，步子越来越大。检查 3 能看出来：梯度越来越大，而不是变小。修法是在每一步的 loss.backward() 之前加回 optimizer.zero_grad()。</p>`),
   twice:tx(`<p>The loss function’s own softmax gets chances instead of logits, so the model can never be sure of an answer, and the loss can go no lower than ${f3(Math.log(1+2/Math.E))}. Check 3 shows it: the gradient is nearly 0 while the loss is still high. The fix is to let the model output logits; F.cross_entropy applies the softmax itself.</p>`,`<p>loss 函数自己的 softmax 拿到的是概率而不是 logit，所以模型怎么学都没法很有把握，loss 最低只能到 ${f3(Math.log(1+2/Math.E))}。检查 3 能看出来：loss 还很高，梯度就已经接近 0。修法是让模型直接输出 logit，F.cross_entropy 会自己做 softmax。</p>`),
   shuffle:tx(`<p>Check 1 finds the loss never leaves ln 3. Check 2 finds the model can still learn 9 examples with their real labels. So the model and the step are fine, and the problem is in how the data reaches the model. The fix is to shuffle the inputs and the labels with one shared order.</p>`,`<p>检查 1 发现 loss 从没离开 ln 3；检查 2 却显示，模型学得会 9 个带正确标签的样本。所以模型和训练步都没问题，问题出在数据送进模型的路上。修法是用同一个顺序一起打乱输入和标签。</p>`),
   small:tx(`<p>With so few examples the model learns these 15 points by heart and gets worse on new ones: its validation loss climbs back from its low of ${f3(vm)} to ${f3(r.va[STEPS])}. That is ${DEF('overfitting',TIP.overfit())}, not a bug, and all three checks pass. More data, a smaller model or stopping earlier helps.</p>`,`<p>样本太少，模型把这 15 个点背了下来，对新的点反而越来越不准：验证集 loss 从最低的 ${f3(vm)} 回升到 ${f3(r.va[STEPS])}。这是${DEF('过拟合',TIP.overfit())}，不是 bug，三个检查也都通过了。多一些数据、用小一点的模型，或者早点停止训练，都能缓解。</p>`)}[p.id];
  return head+why;
}

/* =====================================================================
   challenge
   ===================================================================== */
const CHK_NAME={c1:{en:'Check 1: does the loss stay at ln C?',zh:'检查 1：loss 是不是停在 ln C'},c2:{en:'Check 2: train on just 12 examples',zh:'检查 2：只用 12 个样本训练'},c3:{en:'Check 3: watch the gradient’s size',zh:'检查 3：看梯度有多大'}};
const mystK=()=>MYST[chIdx%MYST.length];
const mystCfg=()=>kindCfg(mystK(),4);
const chMin=()=>minChecks(mystK(),4);
/* the curve's shape in words: as a clause after “the loss” (SHAPE_S) and as a table cell (SHAPE_W) */
const SHAPE_S=(sh,C)=>({now:tx('blows up within the first few steps','头几步就爆掉'),late:tx('falls first, then blows up','先降一阵，再爆掉'),flat:tx(`stays flat at ln ${C}`,`一直平在 ln ${C}`),
  stall:tx('drops fast, then stalls far above 0','先快速下降，再停在离 0 很远的地方'),slow:tx(`falls slowly and is still falling at step ${STEPS}`,`慢慢下降，到第 ${STEPS} 步还在降`),falls:tx('drops fast, then flattens out low','先快速下降，再在低处变平')}[sh]);
const ckList=a=>a.map(c=>L(CHK_NAME[c]).split(/[:：]/)[0]).join(tx(' and ','和'));
function chResultHTML(){
  if(!chSubmitted||!chAns) return '';
  const k=mystK(), mn=chMin(), right=chAns===k, need=mn.length, used=chOpen.length, why=chWhy(k), enough=tellsApart(k,4,chOpen);
  const shK=sigOf(k,4).curve, curveSays=tx(`the loss ${SHAPE_S(shK,4)}, which none of the other four does`,`loss ${SHAPE_S(shK,4)}，另外四种都不是这样`);
  const head=`<p class="okline td-mt"><b>${tx(`Right: ${esc(kname(k).toLowerCase())}.`,`对：${esc(kname(k))}。`)}</b>`;
  let h;
  if(right&&enough&&used<=need) h=head+tx(' ','')+(need?tx(`And you opened only what was needed: ${ckList(mn)}.`,`而且你只打开了需要的检查：${ckList(mn)}。`)
    :tx(`The curve already tells it apart: ${curveSays}. ${ckList(confirmOf(k,4))} would confirm it.`,`光看曲线就能把它分出来：${curveSays}。${ckList(confirmOf(k,4))}可以确认它。`))+`</p>`+why;
  else if(right&&!enough) h=head+`</p><p>`+tx(`But the checks you opened could not yet tell it from another bug whose curve has the same shape: ${ckList(mn)} would have.`,`不过你打开的检查还分不开它和另一个曲线形状相同的 bug：${ckList(mn)}才能分开。`)+`</p>`+why;
  else if(right) h=head+`</p><p>`+(need?tx(`But fewer checks would have done: ${ckList(mn)} alone tells it apart from the other four.`,`不过用不了这么多检查：只看${ckList(mn)}就能把它和另外四种分开。`)
    :tx(`But fewer checks would have done: the curve alone tells it apart, because ${curveSays}. The ${used>1?'checks':'check'} you opened ${used>1?'agree':'agrees'} with it.`,`不过用不了这么多检查：光看曲线就能把它分出来，因为${curveSays}。你打开的检查和它一致。`))+`</p>`+why;
  else {const sa=sigOf(chAns,4).curve, pc=pairCheck(chAns,k,4), an=esc(kname(chAns));
    h=`<p class="badline td-mt"><b>${tx(`Not “${an}”.`,`不是“${an}”。`)}</b></p><p class="small">`+
     (sa!==shK?tx(`Hint: with “${an}” the loss ${SHAPE_S(sa,4)}; here it ${SHAPE_S(shK,4)}.`,`提示：如果是“${an}”，loss 会${SHAPE_S(sa,4)}；这里它${SHAPE_S(shK,4)}。`):tx('Hint: the curve alone cannot tell your answer from the real bug.','提示：光看曲线，分不开你选的 bug 和脚本里真正的 bug。'))+
     (pc?tx(` ${ckList([pc])} would ${sa!==shK?'also ':''}tell your answer apart from the real bug.`,`${ckList([pc])}${sa!==shK?'也':''}能把你选的 bug 和脚本里真正的 bug 分开。`):'')+`</p>`+
     (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);}
  if(right) chReveal=true;
  /* the owner's review of batch 8 (2026-10-07): no list of all five problems. A right answer already says why; “Show the answer”
     after a wrong one names the bug in this run, puts its curve next to the pick's, and says why */
  if(chReveal&&!right) h+=candHTML(4,k,chAns)+why;
  return h;
}
function chWhy(k){
  const r=run(mystCfg()), c3=check3(r), c1=check1(r), c2=check2(mystCfg());
  /* one idea per sentence (R6.28): what the curve did, what the gradient says, how the look-alike differs */
  const m={twice:tx(`<p>Why: the loss drops fast, then stalls at ${f3(r.tr[STEPS])}, far above 0. By then the gradient has faded to ${fG(c3.tail)}, so training has stopped. With a learning rate too low the loss also stays high, but its gradient stays large.</p>`,`<p>原因：loss 先快速下降，再停在 ${f3(r.tr[STEPS])}，离 0 还很远。这时梯度已经降到 ${fG(c3.tail)}，所以训练停住了。学习率太低时 loss 也降不下去，但它的梯度一直很大。</p>`),
   nozero:tx(`<p>Why: the loss falls first, then blows up, and turns NaN at step ${r.dead}. The gradient keeps growing, up to ${fG(c3.max)}, because each step adds the new gradient to the old ones. With a learning rate too high the loss blows up too, but within the first few steps.</p>`,`<p>原因：loss 先降一阵，再爆掉，第 ${r.dead} 步变成 NaN。梯度越来越大，最高到 ${fG(c3.max)}，因为每一步都把新梯度加到旧的上面。学习率太高时 loss 也会爆掉，但在头几步就爆了。</p>`),
   shuffle:tx(`<p>Why: the loss starts at ${f3(c1.start)} and is still ${f3(c1.end)} at step ${STEPS}, close to ln 4 = 1.386. Yet a fresh model learns 12 examples with their real labels (check 2 gets to ${f3(c2.end)}). So the model and the step work, and the problem is in how the data reaches them.</p>`,`<p>原因：loss 从 ${f3(c1.start)} 开始，到第 ${STEPS} 步还是 ${f3(c1.end)}，一直在 ln 4 = 1.386 附近。可是给新模型 12 个带正确标签的样本，它学会了（检查 2 降到 ${f3(c2.end)}）。所以模型和训练步没问题，坏在数据送进模型的路上。</p>`),
   low:tx(`<p>Why: the loss falls slowly and is still falling at step ${STEPS}, at ${f3(r.tr[STEPS])}. The gradient stays large (${fG(c3.tail)} over the last 50 steps), so the direction is right and the steps are just short. With softmax twice the loss also stays high, but its gradient fades to almost 0.</p>`,`<p>原因：loss 慢慢下降，到第 ${STEPS} 步是 ${f3(r.tr[STEPS])}，还在降。梯度一直很大（最后 50 步平均 ${fG(c3.tail)}），说明方向是对的，只是步子太短。softmax 用两次时 loss 也降不下去，但它的梯度会降到接近 0。</p>`),
   high:tx(`<p>Why: the loss blows up at once, with inf at step ${r.inf} and NaN at step ${r.dead}. Every step is far too long, so the gradient explodes within the first steps. Without zero_grad() the loss blows up too, but only after falling for a while.</p>`,`<p>原因：loss 一开始就爆掉，第 ${r.inf} 步变成 inf，第 ${r.dead} 步变成 NaN。每一步都迈得太大，梯度在头几步就爆炸了。漏了 zero_grad() 也会爆掉，但要等 loss 先降一阵。</p>`)}[k];
  return m+tx('<p class="small muted">Note: here exactly one thing is wrong and it is one of five you met. A real script can hide two bugs at once; fix the one a check points at, then run the checks again.</p>','<p class="small muted">注意：这里只有一处错，而且是你见过的五种之一。真实的脚本可能同时藏着两个 bug；先修检查指出的那个，再把检查跑一遍。</p>');
}
/* the challenge's five candidates after an answer: one short line each, how the loss moves (R6.28: the curve decides here) */
function candHTML(C,hl,pick){
  return `<div class="td-cands"><p class="small"><b>${tx(`The answer is “${esc(kname(hl))}”.`,`答案是“${esc(kname(hl))}”。`)}</b></p><ul>`+
    [hl,pick].map(k=>`<li class="${k===hl?'td-me':''}"><b>${esc(kname(k))}</b>${tx(': the loss ','：loss ')}${esc(SHAPE_S(sigOf(k,C).curve,C))}${k===hl?tx(' ← this run',' ← 就是这次'):tx(' ← your answer',' ← 你选的')}</li>`).join('')+`</ul></div>`;
}
/* the five candidates (and in the guided part 6 the healthy run): what the curve does and what each check says */
const SHAPE_W=(sh,C)=>({now:tx('blows up at once','一开始就爆掉'),late:tx('learns, then blows up','先学一阵，再爆掉'),flat:tx(`flat at ln ${C}`,`平在 ln ${C}`),stall:tx('drops fast, stalls far above 0','快速下降，停在离 0 很远处'),slow:tx('falls slowly, still falling','慢慢下降，一直在降'),falls:tx('drops fast, then flattens low','快速下降，在低处变平')}[sh]);
function c1Word(r){const c=check1(r); return c.stays?tx(`starts at ${f3(c.start)}, stays at ln ${r.cfg.C}`,`开始 ${f3(c.start)}，一直停在 ln ${r.cfg.C}`):tx(`starts at ${f3(c.start)}, then leaves ln ${r.cfg.C}`,`开始 ${f3(c.start)}，之后离开 ln ${r.cfg.C}`);}
/* check 2 in words: whether the small batch was learned, and the loss it reached on it after its steps (review: “✗ 0.654” read as a ratio) */
function c2Word(cfg){const c=check2(cfg); if(c.pass) return tx(`✓ learned: loss ${f3(c.end)}`,`✓ 学会了：loss ${f3(c.end)}`);
  return c.dead>=0?tx(`✗ the loss turns NaN at step ${c.dead}`,`✗ 第 ${c.dead} 步 loss 变成 NaN`):(c.inf>=0&&!isFinite(c.end)?tx('✗ the loss turns inf','✗ loss 变成 inf'):tx(`✗ not learned: loss still ${f3(c.end)}`,`✗ 没学会：loss 还有 ${f3(c.end)}`));}
function c3Word(r){const c=check3(r); return {nan:tx(`explodes: ${fG(c.g0)} → ${fG(c.max)}, then the loss turns NaN`,`爆炸：${fG(c.g0)} → ${fG(c.max)}，之后 loss 变成 NaN`),grows:tx(`grows: ${fG(c.g0)} → a peak of ${fG(c.max)}`,`变大：${fG(c.g0)} → 峰值 ${fG(c.max)}`),stays:tx(`does not fade: ${fG(c.g0)} → ${fG(c.tail)} (last 50 steps)`,`没有消失：${fG(c.g0)} → ${fG(c.tail)}（最后 50 步）`),fades:tx(`fades toward 0: ${fG(c.g0)} → ${fG(c.tail)} (last 50 steps)`,`降到接近 0：${fG(c.g0)} → ${fG(c.tail)}（最后 50 步）`)}[c.cat];}
function sigTableHTML(C,hl,right){
  const kinds=(C===3?['none']:[]).concat(KINDS);
  const col=chkPick&&mode==='guided'?chkPick:'', nOf=C*OF_K, H_=[tx('Curve','曲线'),tx('1. Does the loss stay at ln C?','1. loss 是不是停在 ln C'),tx(`2. Train on just ${nOf} examples`,`2. 只用 ${nOf} 个样本训练`),tx('3. Gradient’s size','3. 梯度有多大')];
  const rows=kinds.map(k=>{const cf=k==='none'?{C}:kindCfg(k,C), r=run(cf);
    return `<tr class="${k===hl&&right?'td-me':''}"><th scope="row">${esc(kname(k))}</th><td data-label="${esc(H_[0])}">${esc(SHAPE_W(curveShape(r),C))}</td><td data-label="${esc(H_[1])}" class="${col==='c1'?'td-hl':''}">${esc(c1Word(r))}</td><td data-label="${esc(H_[2])}" class="${col==='c2'?'td-hl':''}">${esc(c2Word(cf))}</td><td data-label="${esc(H_[3])}" class="${col==='c3'?'td-hl':''}">${esc(c3Word(r))}</td></tr>`;}).join('');
  return `<div class="td-tblwrap"><table class="td-tbl"><caption>${tx(`Six runs on the three checks (${C} classes)`,`六次运行在三个检查上的结果（${C} 类点）`)}${col?`<span class="td-capk">${tx('The framed column is the check you picked in the card.','加框的一列，是你在卡片里选的检查。')}</span>`:''}</caption><thead><tr><th scope="col">${tx('Run','运行')}</th><th scope="col">${H_[0]}</th><th scope="col" class="${col==='c1'?'td-hl':''}">${H_[1]}</th><th scope="col" class="${col==='c2'?'td-hl':''}">${H_[2]}</th><th scope="col" class="${col==='c3'?'td-hl':''}">${H_[3]}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* =====================================================================
   copy the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS_.length===7?'Seven':STEPS_.length} short parts. Work out what a loss stuck at ln C means and guess the bug. Then break a healthy run five ways yourself, and learn to tell the breaks apart with three checks. Start here if this is new to you.`,
            `一共 ${STEPS_.length} 个部分。先弄懂 loss 停在 ln C 不动意味着什么，再猜是哪个 bug。然后你亲手用五种方式弄坏一次正常的训练，学会用三个检查把它们分开。第一次学从这里开始。`),
  sandbox:tx('Six situations, from a healthy run to NaN after a learning-rate change and a model with too few examples. Pick one to see its curve, its regions and the three checks.',
             '六种情况，从健康的运行，到改了学习率后出现 NaN，再到样本太少的模型。选一个，看它的曲线、区域和三个检查。'),
  challenge:tx('A teammate trains on a new dataset with 4 classes, and the script has one of the five problems. Read the loss curve, open a check only if you need it, then pick the problem.',
               '同事在一个 4 类的新数据集上训练，脚本里有五种问题中的一种。先看 loss 曲线，需要时再打开检查，然后选出是哪一种。')
});
const SRC={
 clip:'https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py',
 anom:'https://github.com/pytorch/pytorch/blob/main/torch/autograd/anomaly_mode.py',
 optim:'https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py',
 recipe:'https://karpathy.github.io/2019/04/25/recipe/',
 clr:'https://arxiv.org/abs/1506.01186'
};
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> a small classifier really trains in the page, so every curve, region and check is computed, not drawn by hand.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Data: 3 groups of 2-D points (4 in the challenge) around a circle, 100 per group for training and 50 for validation, standardized with the training split’s mean and spread.</li>`+
 `<li>Model: 2 inputs, 16 ReLU units, one logit per class. Training: plain gradient descent on the whole training set, ${STEPS} steps, cross-entropy loss.</li>`+
 `<li>The bugs are switches in the training step. Shuffled labels draw a fresh permutation of the labels every step. The missing zero_grad() adds each gradient to the stored one. With softmax twice, the loss gets chances instead of logits.</li>`+
 `<li>The checks: the loss at step 0 against ln C; a fresh model on 3 examples per group with their real labels, the run’s own step and learning rate, ${OF_STEPS} steps, passing below ${OF_PASS}; the gradient norm of every step.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The script’s softmax is written as ${A.ui.math('\\frac{\\exp(z_c)}{\\sum_j \\exp(z_j)}')}, without subtracting the largest logit, as many from-scratch scripts are. So when a wrong class’s logit leads the right class’s by more than about 745, the right class’s chance rounds to 0 and the loss is inf; a logit above about 709 overflows exp and gives NaN. PyTorch’s F.cross_entropy subtracts it and stays finite; there a too-high rate spikes the loss, and NaN comes later from float32 or float16 overflow (see ${labLink('softmax-numerics')}).</li>`+
 `<li>Real runs use mini-batches, Adam and far larger models, so the curves are noisier. The three checks carry over, but some curves change: Adam keeps each weight’s step near the learning rate whatever the gradient’s size, so a missing zero_grad() can make the loss drift back up instead of turning NaN, while the gradient norm still grows (check 3). Numbers are float64 and seeded: data seed 1 (2 for 4 groups), weights seed ${INIT_SEED}, shuffling seed ${SHUF_SEED}, mulberry32.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Checks 1 and 2 come from Karpathy, ${cite(SRC.recipe,'A Recipe for Training Neural Networks')} (2019): verify the loss at init, overfit a tiny batch.</li>`+
 `<li>Learning-rate range test: Smith, ${cite(SRC.clr,'Cyclical Learning Rates for Training Neural Networks')} (WACV 2017).</li>`+
 `<li>PyTorch: ${cite(SRC.optim,'<code>torch/optim/optimizer.py</code>')} (zero_grad), ${cite(SRC.clip,'<code>torch/nn/utils/clip_grad.py</code>')} (the total gradient norm), ${cite(SRC.anom,'<code>torch/autograd/anomaly_mode.py</code>')} (detect_anomaly).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>一个小分类器在页面里真实训练，所以每条曲线、每个区域、每个检查都是算出来的，不是手画的。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>数据：绕着一个圆的 3 组二维点（挑战题是 4 组），每组 100 个用来训练，50 个用来验证；用训练集的均值和标准差做标准化。</li>`+
 `<li>模型：2 个输入，16 个 ReLU 单元，每个类别一个 logit。训练：在整个训练集上做普通的梯度下降，${STEPS} 步，loss 是 cross-entropy。</li>`+
 `<li>每个 bug 是训练步里的一个开关。打乱标签：每一步都重新抽一个标签的排列。漏了 zero_grad()：每个梯度都加到已存的梯度上。softmax 用两次：loss 拿到的是概率，而不是 logit。</li>`+
 `<li>三个检查：第 0 步的 loss 和 ln C 比；一个新初始化的模型在每组 3 个带真实标签的样本上，用这次运行自己的训练步和学习率跑 ${OF_STEPS} 步，低于 ${OF_PASS} 算通过；每一步的梯度范数。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>脚本里的 softmax 写成 ${A.ui.math('\\frac{\\exp(z_c)}{\\sum_j \\exp(z_j)}')}，没有先减去最大的 logit，很多从零写的脚本都这样。所以某个错误类别的 logit 比正确类别的高出大约 745 以上时，正确类别的概率被舍入成 0，loss 就是 inf；某个 logit 超过大约 709 时，exp 溢出，得到 NaN。PyTorch 的 F.cross_entropy 会先减去它，结果一直有限；在那里学习率太高会让 loss 暴涨，NaN 要等 float32 或 float16 溢出才出现（见 ${labLink('softmax-numerics')}）。</li>`+
 `<li>真实训练用 mini-batch、Adam 和大得多的模型，所以曲线噪声更大。三个检查照样适用，但有些曲线会变：不管梯度多大，Adam 都让每个权重的步子保持在学习率附近，所以漏了 zero_grad() 时，loss 可能是慢慢回升，而不是变成 NaN，梯度范数仍然越来越大（检查 3）。数值是 float64，带种子：数据种子 1（4 组时是 2），权重种子 ${INIT_SEED}，打乱的种子 ${SHUF_SEED}，生成器是 mulberry32。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>检查 1 和检查 2 来自 Karpathy 的 ${cite(SRC.recipe,'A Recipe for Training Neural Networks')}（2019）：先核对初始化时的 loss，再过拟合一个小 batch。</li>`+
 `<li>学习率范围测试：Smith，${cite(SRC.clr,'Cyclical Learning Rates for Training Neural Networks')}（WACV 2017）。</li>`+
 `<li>PyTorch：${cite(SRC.optim,'<code>torch/optim/optimizer.py</code>')}（zero_grad），${cite(SRC.clip,'<code>torch/nn/utils/clip_grad.py</code>')}（梯度的总范数），${cite(SRC.anom,'<code>torch/autograd/anomaly_mode.py</code>')}（detect_anomaly）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A classifier ends with one ${DEF('logit',TIP.logit())} per class; ${DEF('softmax',kt('softmax'))} turns them into chances, and the cross-entropy loss is minus the log of the right class’s chance. Backprop works out the gradient of that loss for every weight, and the optimizer then lowers each weight by the learning rate times its gradient.</p><p>The labs ${labLink('backprop')} and ${labLink('optimizers')} build these pieces; this lab breaks them on purpose.</p>`,
                    `<p>分类器最后为每个类别输出一个 ${DEF('logit',TIP.logit())}；${DEF('softmax',kt('softmax'))} 把它们变成概率，cross-entropy loss 就是正确类别概率的对数取负。反向传播算出 loss 对每个权重的梯度，优化器再把每个权重减去“学习率 × 梯度”。</p><p>${labLink('backprop')}和${labLink('optimizers')}这两个 Lab 搭起了这些部件；这个 Lab 故意把它们弄坏。</p>`);
const OUTCOME=()=>tx(`judge a loss curve against ${DFX('lnc','ln C')}, know which check to run first, and name the bug.`,`用 ${DFX('lnc','ln C')} 判断一条 loss 曲线正不正常，知道先做哪个检查，并说出是哪个 bug。`);
const EXITS=()=>{
  const r3=run({C:3,lr:3}), h3=run({C:3}), lo3=run({C:3,lr:0.003}), cat3=curveCat(r3), aT=cat3==='blows'?0:(r3.tr[STEPS]<run({C:3}).tr[STEPS]*1.5?2:1);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('A 3-class model’s loss starts at 1.10 and is still 1.10 after 300 steps. What does that tell you?','一个 3 类模型的 loss 从 1.10 开始，300 步后还是 1.10。这说明了什么？'),
   o:[tx('1.10 is ln 3, the cross-entropy of giving every class 1/3: after 300 steps the model is still no better than that guess.','1.10 就是 ln 3，也就是给每个类别 1/3 时的 cross-entropy：训练了 300 步，模型还是不比这样猜好。'),
      tx('1.10 is the lowest loss a 3-class model can reach, so the model already learned all it can and the curve has simply converged.','1.10 是 3 类模型能达到的最低 loss，所以模型已经学到了能学的一切，曲线只是收敛了。'),
      tx('1.10 means the learning rate is too small: the weights barely move each step, so the loss stays close to where it started.','1.10 说明学习率太小：权重每一步几乎不动，所以 loss 停在起始值附近。')],a:0,
   n:[null,tx(`The lowest possible loss is 0, a sure right answer on every example; the healthy run in this lab reached ${f3(h3.tr[STEPS])}.`,`可能的最低 loss 是 0，也就是每个样本都百分之百答对；这个 Lab 里健康的运行降到了 ${f3(h3.tr[STEPS])}。`),
     tx(`A small learning rate still moves the loss: at 0.003 it fell slowly and steadily, from ${f3(lo3.tr[0])} to ${f3(lo3.tr[STEPS])}. A loss that does not move from ln 3 at all points to the data or the update instead.`,`学习率小，loss 也还是会动：0.003 时它慢慢、稳稳地从 ${f3(lo3.tr[0])} 降到 ${f3(lo3.tr[STEPS])}。一点都不离开 ln 3，问题更可能在数据或更新本身。`)],
   why:tx(`Cross-entropy is −ln of the chance given to the right class. Giving each of C classes 1/C costs ${A.ui.math('-\\ln\\, \\frac{1}{C} = \\ln\\, C')} on every example, so a loss that stays at ln C means the model does no better than guessing 1/C for every class. It is a signal, not proof that the model guesses 1/C for every class: chances of 2/3 and 1/6 for the right class on two examples also average ln 3. So check the chances and the accuracy next.`,`cross-entropy 是给正确类别的概率取 −ln。给 C 个类别各 1/C，每个样本的代价都是 ${A.ui.math('-\\ln\\, \\frac{1}{C} = \\ln\\, C')}，所以 loss 一直停在 ln C，说明模型并不比“每类都猜 1/C”好。这只是一个信号，不能证明模型真的每类都猜 1/C：两个样本上给正确类别 2/3 和 1/6，平均下来也是 ln 3。所以接着要看概率和准确率。`)},
  {q:tx('With shuffled labels, check 2 passes (a fresh model learns 9 examples with their real labels), yet the full run stays at ln C. What does that tell you?','标签被打乱时，检查 2 通过了（新模型学会了 9 个带正确标签的样本），完整的训练却一直停在 ln C。这说明什么？'),
   o:[tx('The model and the training step are fine, because they learn when given the right labels. So the bug is in how the data reaches the model.','模型和训练步都没问题：给它正确的标签，它就学得会。所以 bug 出在数据送进模型的路上。'),
      tx('Nothing: 9 examples are too few to catch any bug, so check 2 passes whatever the code does.','什么也说明不了：9 个样本太少，查不出任何 bug，不管代码对不对，检查 2 都会通过。'),
      tx('The learning rate is the bug: check 2 uses a larger learning rate than the full run, so it passes anyway.','bug 是学习率：检查 2 用的学习率比完整训练大，所以它照样能通过。')],a:0,
   n:[null,tx('Check 2 fails for four of the five problems: whenever the model, the step or the learning rate is broken, the model cannot learn even 9 examples.','五种问题中有四种都让检查 2 不通过：只要模型、训练步或学习率出了问题，模型连 9 个样本都学不会。'),
     tx('Check 2 uses the run’s own learning rate. That is why it failed at learning rate 0.003 and at 10.','检查 2 用的就是这次训练自己的学习率，所以学习率是 0.003 或 10 时，它都没通过。')],
   why:tx('Check 2 skips the data loader and hands 9 examples with their real labels straight to a fresh model; the model, the step and the learning rate are the full run’s. It learned them, so those three work. The full run still learns nothing, so the problem can only be in the part the check skipped: how the inputs and labels are paired on the way in.','检查 2 跳过数据加载器，把 9 个带正确标签的样本直接交给一个新模型；模型、训练步和学习率都和完整训练一样。它学会了，说明这三样都没问题。完整的训练却什么都没学到，问题只能出在检查跳过的那一段：输入和标签在送进来的路上是怎么配对的。')}],
 transfer:[
  {q:tx('This lab’s 3-class model starts near ln 3 = 1.099 when freshly initialized. A 10-class classifier gets the same small last layer at initialization. About what is its loss at step 0?','这个 Lab 里的 3 类模型刚初始化时，loss 在 ln 3 = 1.099 附近。换成一个 10 类的分类器，最后一层同样初始化得很小，它第 0 步的 loss 大约是多少？'),
   o:[tx('About 1.1','约 1.1'),tx('About 2.3','约 2.3'),tx('About 3.7','约 3.7'),tx('About 10','约 10')],a:1,
   why:tx(`Guessing 1/10 for every class gives a loss of ${A.ui.math(`-\\ln\\, \\frac{1}{10} = \\ln\\, 10 = ${fixd(Math.log(10),3)}`)}. It grows with the log of the class count, not with the count: 10 classes cost about twice as much as 3 (${fixd(Math.log(10)/Math.log(3),2)} times).`,`每类都猜 1/10，loss 就是 ${A.ui.math(`-\\ln\\, \\frac{1}{10} = \\ln\\, 10 = ${fixd(Math.log(10),3)}`)}。它随类别数的对数增长，而不是随类别数本身：10 个类别大约是 3 个类别的 2 倍（${fixd(Math.log(10)/Math.log(3),2)} 倍）。`)},
  {q:tx('In this lab learning rate 0.3 is healthy and 10 gives NaN. What happens at 3, ten times the healthy rate?','在这个 Lab 里，学习率 0.3 是健康的，10 会出现 NaN。学习率为 3，也就是健康值的 10 倍时，会怎样？'),
   o:[tx('NaN within a few steps','几步之内就 NaN'),tx('It swings and ends stuck high','来回摆动，最后停在高处'),tx('It trains, ending near the healthy loss','能训练，最后接近健康的 loss')],a:aT,
   why:tx(`In this lab it trains: the loss is ${f3(r3.tr[STEPS])} at step ${STEPS}, against ${f3(run({C:3}).tr[STEPS])} at 0.3. “Too high” is a threshold, not a ratio: below it, a larger rate takes bigger steps and still converges, here to a slightly lower loss; above it, the steps overshoot and grow. A learning-rate range test finds that threshold for a real model.`,`在这个 Lab 里它能训练：第 ${STEPS} 步 loss 是 ${f3(r3.tr[STEPS])}，0.3 时是 ${f3(run({C:3}).tr[STEPS])}。“太高”是一道门槛，不是一个倍数：低于它，更大的学习率迈的步子更大，照样收敛，这里还收敛到稍低一点的 loss；高于它，步子会冲过头并越来越大。学习率范围测试可以为真实的模型找出这道门槛。`)}],
 practice:[
  {id:'training-loop-debug',d:tx('Parts 2 and 3: predict each bug’s loss curve, then write the sanity checks, including this lab’s loss-at-init and overfit-a-tiny-batch checks.','第 2、3 部分：预测每个 bug 的 loss 曲线，再写出健全性检查，其中包括这个 Lab 的“初始化时的 loss”和“过拟合一个小 batch”。')},
  {id:'transformer-bug-hunt',d:tx('Part 1: four bugs in a small Transformer; one in the training loop keeps the loss near its starting value. Use the three checks to find it.','第 1 部分：一个小 Transformer 里有四个 bug；训练循环里的那个让 loss 一直停在起始值附近。用这三个检查找出它。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:OUTCOME(),short:l&&l.s?L(l.s):tx('Training diagnosis','训练诊断'),title:l?L(l.q):tx('Your loss will not go down: where is the bug?','loss 降不下去，bug 在哪？'),
    minutes:15,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: loss, gradients, learning rate','前置知识：loss、梯度、学习率'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
const scopeTxt=n=>tx(`A real training run in the page: ${n} points, 16 hidden units, seeded.`,`页面里真实的一次训练：${n} 个点，16 个隐藏单元，带种子。`);
function stageHTML(){
  return `<div class="panel stage td-stage" id="td-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="td-st-title"></div><div class="meter-sub" id="td-st-sub"></div></div><div class="stage-actions"><span class="status" id="td-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph td-graph" id="td-graph">`+
    `<div class="td-pane"><div class="td-cap" id="td-curve-cap"></div><div class="td-curve" id="td-curve"></div></div>`+
    `<div class="td-pane"><div class="td-cap" id="td-reg-cap"></div><div class="td-reg" id="td-reg"></div></div>`+
    `<div class="td-tblhost" id="td-tbl" hidden></div>`+
   `</div>`+
   `<p class="td-scope small muted" id="td-scope" tabindex="0" data-tipx="${esc(TIP.illus())}">${scopeTxt(3*NTR)}</p>`+
   `<div class="metrics td-metrics" id="td-metrics"></div>`+
   `<details class="math" id="td-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="td-mathbody"></div></details></div>`;
}
function controlsHTML(){
  const bugOpts=BUGS.map(b=>[b,{none:tx('none','没有'),nozero:tx('no zero_grad()','漏了 zero_grad()'),twice:tx('softmax twice','softmax 两次'),shuffle:tx('labels shuffled','标签打乱')}[b]]);
  return `<aside class="panel controls" id="td-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="td-lock" hidden></p>`+
   `<div class="lockable" id="td-ctl-bug-wrap">`+A.ui.seg({id:'td-ctl-bug',key:'bug',label:tx('Bug in the training script','训练脚本里的 bug'),options:bugOpts,value:S.bug,note:tx('Pick a bug and see how the loss curve changes. Each is a switch in the training step. The learning rate below is a setting, not a bug.','选一个 bug，看看 loss 曲线怎么变。每个 bug 都是训练步里的一个开关。下面的学习率是设置，不是 bug。')})+`</div>`+
   `<div class="lockable" id="td-ctl-lr-wrap">`+A.ui.seg({id:'td-ctl-lr',key:'lr',label:tx('Learning rate','学习率'),options:LRS.map(v=>[String(v),fLr(v)]),value:String(S.lr),note:tx('0.3 trains well here. It is chosen, not computed: the right size depends on the model, the data and the optimizer, so real runs find it with a learning-rate range test.','0.3 在这里训练得很好。它是选出来的，不是算出来的：合适的大小取决于模型、数据和优化器，所以真实训练用学习率范围测试来找它。')})+`</div>`+
   `<div class="lockable" id="td-ctl-k-wrap">`+A.ui.seg({id:'td-ctl-k',key:'k',label:tx('Training examples','训练样本'),options:[['0',tx('300 (100 per group)','300（每组 100）')],[String(SMALL_K),tx('15 (5 per group)','15（每组 5）')]],value:String(S.k),note:tx('Fewer examples than the model can memorize lead to overfitting: the training loss falls while the validation loss rises.','样本少到模型能背下来，就会过拟合：训练 loss 下降，验证集 loss 上升。')})+`</div>`+
   `<div class="td-grp lockable" id="td-ctl-fixed"><div class="td-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl-note">${tx(`${STEPS} steps of plain gradient descent on the whole training set; 16 hidden ReLU units; weights drawn from a seeded normal distribution (the last layer scaled by 0.01, so the untrained model starts near ln C); 150 validation points. These are fixed so that every run starts from the same weights and the only difference is the bug or the setting you pick. The size is chosen to train in well under a second in the page.`,`在整个训练集上做 ${STEPS} 步普通的梯度下降；16 个 ReLU 隐藏单元；权重从带种子的正态分布里抽（最后一层乘以 0.01，所以未训练的模型从 ln C 附近开始）；150 个验证点。这些都固定不变，这样每次运行都从同样的权重出发，唯一的区别就是你选的 bug 或设置。规模选得很小，在页面里训练不到一秒。`)}</div></div>`+
   `<div class="lockable" id="td-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import math
import torch.nn.functional as F

def check_init_loss(model, x, y, n_classes):
    loss = F.cross_entropy(model(x), y).item()  # logits in
    return abs(loss - math.log(n_classes)) < 0.1  # near ln C?

def train_step(model, opt, x, y):
    opt.zero_grad()              # without it, .grad keeps adding up
    loss = F.cross_entropy(model(x), y)  # no softmax before this
    loss.backward()
    opt.step()
    return loss.item()

def check_overfit(model, opt, x, y, steps=200):
    # a fresh model and the run's own step and learning rate,
    # a few examples with their real labels, no data loader
    for _ in range(steps):
        loss = train_step(model, opt, x, y)
    return loss < 0.1            # near 0, or the step is broken

def grad_norm(model):            # after backward(), before step()
    sq = [p.grad.pow(2).sum().item() for p in model.parameters()
          if p.grad is not None]
    return math.sqrt(sum(sq))    # explodes, grows, or ~0 too early?

# in the loop: stop at the first non-finite loss
# loss = train_step(model, opt, xb, yb)
# if not math.isfinite(loss): raise RuntimeError(f"loss {loss}")`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="td-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="td-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The three checks and a training step without the bugs, in PyTorch.','三个检查，以及一个没有这些 bug 的训练步，用 PyTorch 写。')}</p><pre class="code" id="td-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="td-prod">`+
   tx(`<li>${cite(SRC.optim,'<code>torch/optim/optimizer.py</code>')}: <code>Optimizer.zero_grad(set_to_none=True)</code> clears every parameter’s <code>.grad</code>; <code>loss.backward()</code> adds to it, which is why a missing call makes the gradients pile up.</li>`+
      `<li>${cite(SRC.clip,'<code>torch/nn/utils/clip_grad.py</code>')}: <code>clip_grad_norm_</code> returns the total gradient norm before clipping, so logging its return value is check 3 for free; with <code>error_if_nonfinite=True</code> it raises an error once that norm is inf or NaN.</li>`+
      `<li>${cite(SRC.anom,'<code>torch/autograd/anomaly_mode.py</code>')}: <code>torch.autograd.detect_anomaly()</code> (or <code>set_detect_anomaly(True)</code>) raises an error at the first backward step that returns NaN and shows the forward line that created it. It is slow, so use it only to locate a NaN.</li>`,
      `<li>${cite(SRC.optim,'<code>torch/optim/optimizer.py</code>')}：<code>Optimizer.zero_grad(set_to_none=True)</code> 清空每个参数的 <code>.grad</code>；<code>loss.backward()</code> 是往里加的，所以漏了这一行，梯度就会越堆越多。</li>`+
      `<li>${cite(SRC.clip,'<code>torch/nn/utils/clip_grad.py</code>')}：<code>clip_grad_norm_</code> 返回裁剪之前的总梯度范数，把返回值记下来就免费得到了检查 3；设了 <code>error_if_nonfinite=True</code>，这个范数一变成 inf 或 NaN 就报错。</li>`+
      `<li>${cite(SRC.anom,'<code>torch/autograd/anomaly_mode.py</code>')}：<code>torch.autograd.detect_anomaly()</code>（或 <code>set_detect_anomaly(True)</code>）在第一个返回 NaN 的反向计算处报错，并给出前向里创建它的那一行。它很慢，只在定位 NaN 时用。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.recipe,'A Recipe for Training Neural Networks')}<span>${tx(`Karpathy, 2019: verify the loss at init (${A.ui.math('-\\log \\frac{1}{C}')} for a softmax), overfit a tiny batch, and other habits that catch silent bugs`,`Karpathy，2019：核对初始化时的 loss（softmax 时是 ${A.ui.math('-\\log \\frac{1}{C}')}），过拟合一个小 batch，以及其他能抓住无声 bug 的习惯`)}</span></li>`+
   `<li>${cite(SRC.clr,'Cyclical Learning Rates for Training Neural Networks')}<span>${tx('Smith, WACV 2017: the learning-rate range test, which raises the rate during a short run and reads off where the accuracy stops rising','Smith，WACV 2017：学习率范围测试，在一次短训练里逐步调高学习率，看准确率在哪里不再上升')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="td-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="td-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="td-app m2" id="td-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="td-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how its loss curve moves.','选一个场景，看看它的 loss 曲线怎么走。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="td-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const k=mystK(), n=chOpen.length;
  const ckBtns=['c1','c2','c3'].map(c=>`<button type="button" class="btn ${chOpen.includes(c)?'':'ghost '}sm td-ckbtn" data-ck="${c}" aria-pressed="${chOpen.includes(c)}"${chOpen.includes(c)?' disabled':''}>${esc(L(CHK_NAME[c]))}${chOpen.includes(c)?' ✓':''}</button>`).join('');
  const ans=`<div class="choices td-ans" role="group" aria-label="${esc(tx('Your answer, pick one','你的答案，选一个'))}">${KINDS.map(b=>`<button type="button" class="choice" data-ans="${b}" aria-pressed="${chAns===b}">${esc(kname(b))}</button>`).join('')}</div>`;
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit"${chAns?'':' disabled'}>${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you named the right bug, and whether fewer checks would have done.','提交后会告诉你 bug 对不对，以及能不能少用几个检查。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Find the problem in a teammate’s script, with as few checks as you can','找出同事脚本里的问题，检查用得越少越好')}</h3>`+
   `<p>${tx(`A teammate trains the same small model on a new dataset with 4 classes of points. Their script has one thing wrong, and it is one of the five problems you met in the guided parts (script ${chIdx%MYST.length+1} of ${MYST.length}).`,
            `同事用同样的小模型，在一个有 4 类点的新数据集上训练。他的脚本里错了一处，是你在分步引导里见过的五种问题之一（第 ${chIdx%MYST.length+1} 个脚本，共 ${MYST.length} 个）。`)}</p>`+
   `<p>${tx('Read the loss curve first. If it does not settle the question, open a check below. Each check you open counts as one used, and the fewer you need, the better.',
            '先看图里的 loss 曲线。光看曲线定不下来时，再打开下面的检查。每打开一个检查，就算用了一次，用得越少越好。')}</p>`+
   `<div class="td-cks">${ckBtns}</div>`+
   `<div class="goal mono td-goal"><span>${tx('Checks opened','已打开的检查')}</span><span><b id="td-ch-n">${n}</b>${tx(' of 3',' / 3')}</span></div>`+
   `<p class="small"><b>${tx('Which problem do you think it is? Pick one as your answer.','你认为是哪一种问题？选一个作为答案。')}</b></p>`+ans+
   submit+`<div id="td-ch-res" class="td-ch-res" aria-live="polite"></div>`+
   `<p class="small muted td-mt"><button type="button" class="linkbtn" data-act="next-myst">${tx('Try another teammate’s script','换一个同事的脚本')}</button></p></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS_[step];
    return A.ui.steps({steps:STEPS_.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS_.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS_[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},healthy:{en:`Go to step ${STEPS} to continue.`,zh:`走到第 ${STEPS} 步才能继续。`},
    lr:{en:'Try both learning rates to continue.',zh:'两个学习率都试过才能继续。'},bugs:{en:'Try all three bugs to continue.',zh:'三个 bug 都试过才能继续。'},checks:{en:'Pick two checks to continue.',zh:'选两个检查才能继续。'}}[STEPS_[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS_.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#td-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="td-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="td-why" hidden></p>');}
  A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#td-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS_[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.gate||s.gate())&&((s.id==='healthy')||(s.id==='lr'&&lrPick!=='0.3')||(s.id==='bugs'&&bugPick!=='none')||(s.id==='checks'&&!!chkPick)||!!observed[step]);
    $$('#td-ic-lr [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lrPick)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lrRes(v);});
    $$('#td-ic-bug [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===bugPick)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=bugRes(v);});
    $$('#td-ic-chk [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===chkPick)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=ckRes(v);});
    $$('.seg-ctl[data-key="steps"]',rail).forEach(g=>A.ui.setPressed(g,String(stepsShown)));
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.td-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#td-insight',rail); if(ins){const html=cur>=0?scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const n=$('#td-ch-n',rail); if(n) n.textContent=String(chOpen.length);
    $$('[data-ans]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.ans===chAns)));
    const sb=$('[data-act="submit"]',rail); if(sb) sb.disabled=!chAns;
    const res=$('#td-ch-res',rail), html=chResultHTML(); if(res&&res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='challenge'?'ch':(isP('checks')?'tbl':'run');
/* what the stage shows: the run, the comparison run (thin line), how many steps of it, the number of classes */
function stageRun(){
  if(mode==='challenge') return {r:run(mystCfg()),cmp:null,upTo:STEPS,C:4};
  const r=R(), cmp=(isP('lr')&&lrPick!=='0.3')||(isP('bugs')&&bugPick!=='none')||(mode==='sandbox'&&presetNow()!==0&&!(S.lr===LR0&&S.bug==='none'&&S.k===0))?run({C:3}):null;
  return {r,cmp,upTo:isP('healthy')?stepsShown:STEPS,C:3};
}
/* the plot sits below a band of two text lines (PT) that holds the blow-up notes, so no curve crosses them (R7.8) */
const W=400, Hc=232, PL=34, PR=8, PT=32, PB=30;
/* rough width of a line of 11 px monospace text in the chart (CJK characters take a full em) */
const tw11=t=>[...t].reduce((a,c)=>a+(/[\u2e80-\u9fff\uff00-\uffef]/.test(c)?11:6.7),0);
/* x of a note that starts near x0 and stays inside the plot */
const noteX=(x0,t,end)=>{const w=tw11(t), x=end?x0-w:x0; return Math.max(PL+2,Math.min(x,W-PR-2-w));};
function curveSVG(){
  const {r,cmp,upTo,C}=stageRun(), l=lnC(C), yMax=C===3?1.6:2.0, cap=yMax*1.08, X=t=>PL+(W-PL-PR)*t/STEPS, Y=v=>PT+(Hc-PT-PB)*(1-Math.min(v,cap)/yMax);
  const showLn=mode!=='challenge'||chOpen.includes('c1'), lnT=`ln ${C} = ${fixd(l,3)}`;
  const line=(a,n)=>{let d='', on=false; for(let t=0;t<=Math.min(n,a.length-1);t++){const v=a[t]; if(!isFinite(v)){on=false; continue;} d+=(on?'L':'M')+X(t).toFixed(1)+' '+Y(v).toFixed(1); on=true;} return d;};
  let h=`<svg viewBox="0 0 ${W} ${Hc}" class="td-svg" role="img" aria-label="${esc(curveAria(r,upTo,C))}"><defs><clipPath id="td-clip"><rect x="${PL}" y="${PT-2}" width="${W-PL-PR}" height="${Hc-PB-PT+2}"/></clipPath></defs>`;
  for(let i=0;i<=4;i++){const v=yMax*i/4, y=Y(v); h+=`<line class="td-grid" x1="${PL}" x2="${W-PR}" y1="${y}" y2="${y}"/><text class="td-ax" x="${PL-6}" y="${y+3.5}" text-anchor="end">${fixd(v,1)}</text>`;}
  /* the last step's label ends at its tick, so it stays inside the chart (H7, kit_text_bounds) */
  [0,100,200,300].forEach(t=>{h+=`<text class="td-ax" x="${X(t)}" y="${Hc-PB+16}" text-anchor="${t===300?'end':'middle'}">${t}</text>`;});
  h+=`<text class="td-ax" x="${(PL+W-PR)/2}" y="${Hc-3}" text-anchor="middle">${esc(tx('training step','训练步'))}</text>`;
  h+=`<g clip-path="url(#td-clip)">`;
  if(showLn) h+=`<line class="td-ln" x1="${PL}" x2="${W-PR}" y1="${Y(l)}" y2="${Y(l)}"/>`;
  if(cmp) h+=`<path class="td-cmp" d="${line(cmp.tr,STEPS)}"/>`;
  h+=`<path class="td-va" d="${line(r.va,upTo)}"/><path class="td-tr" d="${line(r.tr,upTo)}"/>`;
  h+=`</g>`;
  if(showLn) h+=`<text class="td-lnl t td-dfx" data-ctl="dfx" data-term="lnc" x="${PL+8}" y="${(Y(l)-11).toFixed(1)}">${lnT}</text>`;
  /* where the run leaves the chart, and where it stops being a number: notes in the band above the plot */
  const out=r.tr.findIndex((v,t)=>t<=upTo&&isFinite(v)&&v>cap);
  if(out>=0){const t=tx(`↑ training loss up to ${f3(Math.max(...r.tr.filter(isFinite)))}`,`↑ 训练 loss 最高到 ${f3(Math.max(...r.tr.filter(isFinite)))}`); h+=`<text class="td-mk" x="${noteX(X(out)+4,t,false).toFixed(1)}" y="12">${esc(t)}</text>`;}
  const bad=r.inf>=0?r.inf:r.dead;
  if(bad>=0&&bad<=upTo){const x=X(bad), t=r.inf>=0&&r.dead>=0?tx(`inf at step ${r.inf}, NaN from step ${r.dead}`,`第 ${r.inf} 步 inf，第 ${r.dead} 步起 NaN`):(r.dead>=0?tx(`NaN from step ${r.dead}`,`第 ${r.dead} 步起 NaN`):tx(`inf from step ${r.inf}`,`第 ${r.inf} 步起 inf`)), late=bad>=STEPS*0.6;
    h+=`<line class="td-nan" x1="${x}" x2="${x}" y1="${PT-4}" y2="${Hc-PB}"/><text class="td-mk" x="${noteX(late?x-4:x+4,t,late).toFixed(1)}" y="26">${esc(t)}</text>`;}
  if(isP('healthy')&&upTo<STEPS) h+=`<line class="td-now" x1="${X(upTo)}" x2="${X(upTo)}" y1="${PT}" y2="${Hc-PB}"/>`;
  /* each curve is named beside its end, inside the plot, where the name covers no curve, no other name and no marker line
     (review 2026-10-03: one legend above the chart was hard to match to the lines) */
  if(upTo>0){
    const pts=[], boxes=showLn?[[PL+6,Y(l)-22,tw11(lnT)+4,15]]:[], vx=[];
    if(bad>=0&&bad<=upTo) vx.push(X(bad));
    if(isP('healthy')&&upTo<STEPS) vx.push(X(upTo));
    const trace=(a,n)=>{const m=Math.min(n,a.length-1); for(let t=0;t<=m;t++){const u=a[t]; if(!isFinite(u)) continue; const x0=X(t), y0=Y(u); if(y0>=PT-2) pts.push([x0,y0]);
      const v=t<m?a[t+1]:NaN; if(!isFinite(v)) continue; const y1=Y(v), k=Math.ceil(Math.abs(y1-y0)/3); for(let i=1;i<k;i++){const y=y0+(y1-y0)*i/k; if(y>=PT-2) pts.push([x0+(X(t+1)-x0)*i/k,y]);}}};
    if(cmp) trace(cmp.tr,STEPS); trace(r.va,upTo); trace(r.tr,upTo);
    const endOf=(a,n)=>{for(let t=Math.min(n,a.length-1);t>=0;t--){const v=a[t]; if(isFinite(v)&&v<=cap) return t;} return -1;};
    const cost=bx=>{let c=0; if(bx[0]<PL+2||bx[0]+bx[2]>W-PR-1||bx[1]<PT-1||bx[1]+bx[3]>Hc-PB+2) c+=1000;
      for(const o of boxes) if(bx[0]<o[0]+o[2]&&o[0]<bx[0]+bx[2]&&bx[1]<o[1]+o[3]&&o[1]<bx[1]+bx[3]) c+=100;
      for(const x of vx) if(bx[0]-2<x&&x<bx[0]+bx[2]+2) c+=100;
      if(showLn&&bx[1]<Y(l)+1&&Y(l)-1<bx[1]+bx[3]) c+=3;
      for(const [x,y] of pts) if(x>bx[0]-2&&x<bx[0]+bx[2]+2&&y>bx[1]-2&&y<bx[1]+bx[3]+2) c+=1;
      return c;};
    const name=(a,n,cls,txt,tip)=>{const e=endOf(a,n); if(e<0) return; const w=tw11(txt), ex=X(e), ey=Y(a[e]), cand=[];
      const left=(xe,y)=>cand.push([Math.min(xe,W-PR-1)-3-w,y]), right=(x,y)=>cand.push([x,y]);
      left(ex,ey-6); left(ex,ey+14); right(ex+6,ey+4); right(ex+6,ey-7); right(ex+6,ey+15); right(ex+6,ey+28); right(ex+6,ey+41);
      [0.8,0.6,0.4].forEach(f=>{const t=Math.round(e*f), v=a[t]; if(t>0&&isFinite(v)&&v<=cap){left(X(t),Y(v)-6); left(X(t),Y(v)+14);}});
      let best=null, bc=Infinity; for(const [x,y] of cand){const bx=[x-1,y-10,w+2,13], c=cost(bx); if(c<bc){bc=c; best=[x,y,bx];}}
      if(!best||bc>=1000) return; boxes.push(best[2]);
      h+=`<text class="td-cl ${cls}" x="${best[0].toFixed(1)}" y="${best[1].toFixed(1)}" data-tipx="${esc(tip)}">${esc(txt)}</text>`;};
    name(r.tr,upTo,'td-cl-tr',tx('training loss','训练 loss'),tx('The training loss is the model’s cross-entropy on the training examples, computed before each update.','训练 loss 是模型在训练样本上的 cross-entropy，每一步更新之前算一次。'));
    name(r.va,upTo,'td-cl-va',tx('validation loss','验证集 loss'),TIP.val());
    if(cmp) name(cmp.tr,STEPS,'td-cl-cmp',tx('healthy run','健康的运行'),tx('The training loss of the healthy run (learning rate 0.3, no bug), for comparison.','健康运行（学习率 0.3，没有 bug）的训练 loss，用来对照。'));
  }
  return h+`</svg>`;
}
function curveAria(r,upTo,C){const e=lastFinite(r.tr.slice(0,upTo+1)); return tx(`Training loss from ${f3(r.tr[0])} at step 0 to ${f3(e.v)} at step ${e.i}${r.dead>=0&&r.dead<=upTo?`, NaN from step ${r.dead}`:''}; ln ${C} = ${fixd(lnC(C),3)}.`,`训练 loss 从第 0 步的 ${f3(r.tr[0])} 到第 ${e.i} 步的 ${f3(e.v)}${r.dead>=0&&r.dead<=upTo?`，第 ${r.dead} 步起 NaN`:''}；ln ${C} = ${fixd(lnC(C),3)}。`);}
const SHAPES=['circle','square','triangle','diamond'];
function marker(c,x0,y0,s){
  const f=v=>(+v).toFixed(1), x=+x0, y=+y0;
  if(c===0) return `<circle class="td-pt" cx="${f(x)}" cy="${f(y)}" r="${s}"/>`;
  if(c===1) return `<rect class="td-pt" x="${f(x-s)}" y="${f(y-s)}" width="${2*s}" height="${2*s}"/>`;
  if(c===2) return `<path class="td-pt" d="M${f(x)} ${f(y-s*1.2)}L${f(x+s*1.15)} ${f(y+s*0.85)}L${f(x-s*1.15)} ${f(y+s*0.85)}Z"/>`;
  return `<path class="td-pt" d="M${f(x)} ${f(y-s*1.3)}L${f(x+s*1.2)} ${f(y)}L${f(x)} ${f(y+s*1.3)}L${f(x-s*1.2)} ${f(y)}Z"/>`;
}
function regSVG(){
  const {r,upTo,C}=stageRun(), D=dataOf(C), snapT=isP('healthy')?upTo:STEPS, p=r.snaps[snapT]||r.last, G=r.dead>=0&&r.dead<=snapT?null:regions(p);
  const Z=200, cell=Z/GRID, P=v=>((v+EXT)/(2*EXT))*Z, Q=v=>((EXT-v)/(2*EXT))*Z;
  let h=`<svg viewBox="0 0 ${Z} ${Z}" class="td-svg td-regsvg" role="img" aria-label="${esc(tx('Where the model predicts each class, with the training points marked by shape','模型在哪里预测哪个类别，训练点按形状标出'))}"><defs>`+
   `<pattern id="td-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" class="td-hl1"/></pattern>`+
   `<pattern id="td-dots" width="5" height="5" patternUnits="userSpaceOnUse"><circle cx="2.5" cy="2.5" r="0.9" class="td-dot"/></pattern></defs>`;
  h+=`<rect class="td-regbg" x="0" y="0" width="${Z}" height="${Z}"/>`;
  const tc=G?topChance(p):NaN, faint=tc<0.5;
  if(G){h+=`<g${faint?' class="td-faint"':''}>`; for(let i=0;i<GRID;i++){let j=0; while(j<GRID){const c=G[i][j]; let e=j; while(e+1<GRID&&G[i][e+1]===c) e++; if(c>0) h+=`<rect class="td-rc${c}" x="${(j*cell).toFixed(2)}" y="${(i*cell).toFixed(2)}" width="${((e-j+1)*cell+0.3).toFixed(2)}" height="${(cell+0.3).toFixed(2)}"/>`; j=e+1;}} h+=`</g>`;}
  const T=subset(D,isP('healthy')||mode!=='sandbox'?0:S.k);
  T.X.forEach((x,i)=>{h+=marker(T.y[i],P(x[0]).toFixed(1),Q(x[1]).toFixed(1),T.X.length<50?3.4:2.2);});
  if(!G) h+=`<rect class="td-nanbg" x="${Z*0.14}" y="${Z/2-19}" width="${Z*0.72}" height="38" rx="3"/><text class="td-nant" x="${Z/2}" y="${Z/2-3}" text-anchor="middle">${esc(tx('the loss is NaN:','loss 是 NaN：'))}</text><text class="td-nant" x="${Z/2}" y="${Z/2+11}" text-anchor="middle">${esc(tx('no usable model','没有可用的模型'))}</text>`;
  h+=`</svg>`+regKeyHTML(C);
  if(faint) h+=`<p class="td-regnote">${tx(`Across the whole map, the model never gives any class more than ${fixd(tc*100,0)}%, so the coloured regions behind the points are only noise.`,`在整张图里，模型给任何一类的概率都不超过 ${fixd(tc*100,0)}%，所以背景的分区只是噪声。`)}</p>`;
  return h;
}
/* the key under the region map: each fill shown with the shape of the class it stands for (review 2026-10-03: the key sat above the chart) */
function regKeyHTML(C){
  const nm=[tx('circle','圆'),tx('square','方块'),tx('triangle','三角'),tx('diamond','菱形')];
  return `<div class="td-regkey"><p>${tx('A point’s shape is its true class; the background is the class the model predicts there:','点的形状是它的真实类别，背景是模型在那里预测的类别：')}</p><div class="td-rks">`+
    nm.slice(0,C).map((n,c)=>`<span class="td-rk"><svg viewBox="0 0 16 16" class="td-rksv" aria-hidden="true">${c?`<rect class="td-rc${c}" x="0" y="0" width="16" height="16"/>`:''}${marker(c,8,8.5,3.6)}</svg>${esc(n)}</span>`).join('')+`</div></div>`;
}
function statusOf(){
  if(mode==='challenge') return {cls:'neutral',txt:tx('Teammate’s run','同事的运行'),short:tx('Teammate','同事的运行'),tip:tx('The script behind this run has one of the five problems. Open checks in the task card if you need them, then pick the problem.','这次运行的脚本里有五种问题之一。需要时在任务卡里打开检查，再选出是哪一种。')};
  if(viewNow()==='tbl') return {cls:'neutral',txt:tx('Six runs','六次运行'),short:tx('Six runs','六次运行'),tip:tx('The healthy run and the five broken ones, each with its three checks.','健康的运行和五次弄坏的运行，各自的三个检查。')};
  const {r,C}=stageRun(), cc=curveCat(r), w={blows:{en:'Blows up',zh:'爆掉了'},flat:{en:`Stuck at ln ${C}`,zh:`停在 ln ${C}`},falls:{en:'Loss falls',zh:'loss 在下降'}}[cc];
  const tip={blows:tx('The loss leaves the chart or stops being a number (inf or NaN).','loss 冲出了图，或者不再是有限的数值（inf 或 NaN）。'),flat:tx(`The loss stays within 0.1 of ln ${C} for the whole run: the model does no better than guessing 1/${C} for every class.`,`整次运行里 loss 都在 ln ${C} 上下 0.1 以内：模型并不比“每类都猜 1/${C}”好。`),falls:tx('The loss ends below where it started. Whether it falls far enough is what the checks and the healthy run tell.','loss 最后比开头低。降得够不够，要看检查和健康的运行。')}[cc];
  return {cls:cc==='blows'?'bad':'neutral',txt:L(w),short:L(w),tip};
}
function drawHead(){
  const t=$('#td-st-title',app), sub=$('#td-st-sub',app), st=$('#td-status',app), v=viewNow();
  if(v==='tbl'){t.textContent=tx('The three checks on six runs','六次运行上的三个检查'); sub.textContent=tx('3 groups of points, learning rate 0.3 unless the row says otherwise','3 组点；除非那一行另有说明，学习率都是 0.3');}
  else if(v==='ch'){t.textContent=tx('Your teammate’s run: training and validation loss, 4 classes','同事的运行：训练和验证集的 loss，4 个类别'); sub.textContent=tx(`${STEPS} steps; the learning rate and the problem are not shown`,`训练 ${STEPS} 步；学习率和问题都不告诉你`);}
  else {t.textContent=tx('Training and validation loss, 3 classes','训练和验证集 loss，3 个类别');
    const hide=mode==='guided'&&step<=1;
    sub.textContent=hide?tx(`A run with a hidden bug, ${STEPS} steps`,`一次藏着 bug 的运行，${STEPS} 步`):tx(`Learning rate ${fLr(S.lr)}, bug: ${S.bug==='none'?'none':kname(S.bug).toLowerCase()}, ${S.k?`${S.k*3} training examples`:'300 training examples'}`,`学习率 ${fLr(S.lr)}，bug：${S.bug==='none'?'没有':kname(S.bug)}，${S.k?S.k*3:300} 个训练样本`);}
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const sc=$('#td-scope',app); if(sc){const n=v==='ch'?4*NTR:(v==='tbl'||!S.k?3*NTR:3*S.k); sc.textContent=scopeTxt(n); sc.dataset.tipx=TIP.illus(n);}
}
/* metric cards: the result, then the three checks; ▲▼ only after the learner's own change (R6.6) */
const spark=(a,log)=>{const v=a.filter(isFinite); if(v.length<2) return ''; const f=log?(x=>Math.log10(Math.max(x,1e-6))):(x=>x), lo=Math.min(...v.map(f)), hi=Math.max(...v.map(f)), n=a.length;
  let d='', on=false; a.forEach((x,i)=>{if(!isFinite(x)){on=false; return;} const X=(i/(n-1)*100).toFixed(1), Y=(22-((f(x)-lo)/(hi-lo||1))*20).toFixed(1); d+=(on?'L':'M')+X+' '+Y; on=true;});
  return `<svg class="td-spark" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true"><path d="${d}"/></svg>`;};
function metricsDef(){
  const v=viewNow();
  if(v==='tbl') return [];
  const {r,upTo,C}=stageRun(), t=upTo, cfg=mode==='challenge'?mystCfg():cfgNow(), ch=mode==='challenge', open=c=>!ch||chOpen.includes(c), nOf=C*OF_K;
  const e=lastFinite(r.tr.slice(0,t+1)), dead=r.dead>=0&&r.dead<=t;
  const m=[{k:'tl',label:tx(`Training loss, step ${t}`,`第 ${t} 步的训练 loss`),v:dead?'NaN':f3(r.tr[t]),sub:tx(`started at ${f3(r.tr[0])}`,`开始时 ${f3(r.tr[0])}`),num:true,tip:t===0?tx(`Before any training, the training loss is ${f3(r.tr[0])}.`,`模型还没训练时，训练 loss 是 ${f3(r.tr[0])}。`):tx(`After ${t} steps of training, the training loss is ${dead?'NaN, because the loss became NaN at step '+r.dead:f3(r.tr[t])}. At step 0 it was ${f3(r.tr[0])}.`,`训练到第 ${t} 步时，训练 loss 是 ${dead?'NaN（loss 在第 '+r.dead+' 步就变成了 NaN）':f3(r.tr[t])}。第 0 步时是 ${f3(r.tr[0])}。`)},
    {k:'acc',label:tx('Validation accuracy','验证集准确率'),v:dead?'—':fPct(r.acc[t]),sub:tx(`${C*NVA} points not trained on`,`${C*NVA} 个没参与训练的点`),num:true,tip:tx(`At step ${t}, the share of the ${C*NVA} validation points the model gets right: a point counts as right when the class with the highest score is its true class. Guessing 1/${C} for every class gets about ${fPct(1/C)} right.`,`第 ${t} 步时，模型在 ${C*NVA} 个验证点里猜对的比例：得分最高的类别正好是真实类别，就算猜对。每类都猜 1/${C} 时，大约能猜对 ${fPct(1/C)}。`)}];
  if((isP('healthy')&&t<STEPS)||(mode==='guided'&&step<=1)) return m;
  const c1=check1(r), c2=check2(cfg), c3=check3(r), lnT=`ln ${C} = ${fixd(c1.lnC,3)}`, end=lastFinite(r.tr).v;
  const c1L=tx('Check 1: does the loss stay at ln C?','检查 1：loss 是不是停在 ln C'), c2L=tx(`Check 2: train on just ${nOf} examples`,`检查 2：只用 ${nOf} 个样本训练`), c3L=tx('Check 3: the gradient’s size','检查 3：梯度有多大');
  const c1ok=c1.startOk&&!c1.stays;
  const c1tip=c1.stays?tx(`The loss never leaves ${lnT}, so the model does no better than guessing 1/${C} for every class. The start, ${f3(c1.start)}, is right: that guess costs ${lnT}.`,`loss 一直没离开 ${lnT}，说明模型并不比“每类都猜 1/${C}”好。开头的 ${f3(c1.start)} 倒是对的：每类都猜 1/${C} 时，loss 就是 ${lnT}。`)
    :(!c1.startOk?tx(`At step 0 the loss is ${f3(c1.start)}, far from ${lnT}, the loss of guessing 1/${C} for every class. Look at the initialization, or at how the loss is computed.`,`第 0 步的 loss 是 ${f3(c1.start)}，离 ${lnT}（每类都猜 1/${C} 时的 loss）很远。该去查初始化，或者 loss 是怎么算的。`)
    :tx(`At step 0 the loss is ${f3(c1.start)}, near ${lnT}, the loss of guessing 1/${C} for every class, so the start is right. The loss then leaves it.`,`第 0 步的 loss 是 ${f3(c1.start)}，接近 ${lnT}（每类都猜 1/${C} 时的 loss），所以开头是对的。之后 loss 离开了它。`));
  m.push(open('c1')?{k:'c1',chk:1,label:c1L,v:f3(c1.start),sub:c1.stays?tx(`${lnT}, never leaves it`,`${lnT}，一直不离开`):tx(`${lnT}, then leaves it`,`${lnT}，之后离开`),ok:c1ok,tip:c1tip}:{k:'c1',chk:1,closed:true,label:c1L});
  const c2how=tx(`A freshly initialized model learns just ${nOf} examples (3 per group) with their real labels, through this run’s own step and learning rate: ${c2.dead>=0?`its loss turns NaN at step ${c2.dead}`:`after ${OF_STEPS} steps its loss is ${f3(c2.end)}`}.`,`一个新初始化的模型，用这次运行的训练步和学习率，只学 ${nOf} 个带真实标签的样本（每组 3 个），${c2.dead>=0?`loss 在第 ${c2.dead} 步就变成了 NaN`:`学 ${OF_STEPS} 步之后，loss 是 ${f3(c2.end)}`}。`);
  const c2low=!ch&&cfg.lr===0.003&&cfg.bug==='none';
  const c2tip=c2.pass?c2how+tx(' It learned them, so nothing is wrong with the model or the step itself.','它学会了，说明模型和训练步本身没有问题。')
    :(c2low?c2how+tx(` In ${OF_STEPS} steps it does not learn even ${nOf} examples. Nothing in the code is broken: the learning rate is too low (check 3 shows the gradient is still large).`,`${OF_STEPS} 步下来，连 ${nOf} 个样本都学不会。代码并没有坏，是学习率太低了（检查 3 显示梯度还很大）。`)
    :c2how+tx(` If it cannot learn even ${nOf} examples, the problem is in the model, the step or the learning rate, not in the data.`,`连 ${nOf} 个样本都学不会，问题就在模型、训练步或学习率里，不在数据里。`));
  m.push(open('c2')?{k:'c2',chk:2,label:c2L,v:(c2.pass?'✓ ':'✗ ')+'loss '+(c2.dead>=0?'NaN':f3(c2.end)),sub:tx(`after ${OF_STEPS} steps; below ${OF_PASS} means learned`,`训练 ${OF_STEPS} 步后；低于 ${OF_PASS} 算学会了`),ok:c2.pass,spark:spark(c2.r.tr,false),tip:c2tip}:{k:'c2',chk:2,closed:true,label:c2L});
  const c3ok=c3.cat==='fades'&&curveCat(r)==='falls'&&end<0.5;
  const c3tip={nan:tx(`The gradient norm explodes from ${fG(c3.g0)} at step 0 to ${fG(c3.max)} within a few steps, and the loss turns NaN at step ${r.dead}.`,`梯度范数几步之内就从第 0 步的 ${fG(c3.g0)} 爆炸到 ${fG(c3.max)}，loss 在第 ${r.dead} 步变成了 NaN。`),
    grows:tx(`The gradient norm does not shrink: it grows from ${fG(c3.g0)} at step 0 to a peak of ${fG(c3.max)}.`,`梯度范数没有变小，反而从第 0 步的 ${fG(c3.g0)} 一路涨到峰值 ${fG(c3.max)}。`),
    stays:tx(`The gradient norm does not fade: it is still ${fG(c3.tail)} on average over the last 50 steps (${fG(c3.g0)} at step 0).`,`梯度范数没有降到接近 0：最后 50 步平均还有 ${fG(c3.tail)}（第 0 步是 ${fG(c3.g0)}）。`),
    fades:c3ok?tx(`The gradient norm fades from ${fG(c3.g0)} at step 0 to ${fG(c3.tail)} over the last 50 steps as the loss falls, which is what healthy training looks like.`,`梯度范数随着 loss 下降，从第 0 步的 ${fG(c3.g0)} 降到最后 50 步平均 ${fG(c3.tail)}，正是健康训练该有的样子。`)
      :tx(`The gradient norm has faded to ${fG(c3.tail)} while the loss is still ${f3(end)}, far above 0.`,`loss 还有 ${f3(end)}，离 0 很远，梯度范数却已经降到 ${fG(c3.tail)}。`)}[c3.cat]+tx(' The sparkline uses a log scale.','小图是对数刻度。');
  m.push(open('c3')?{k:'c3',chk:3,label:c3L,v:c3Word(r).split(/[:：]/)[0],sub:c3Word(r).split(/[:：]\s*/)[1]||'',ok:c3ok,spark:spark(r.gn,true),tip:c3tip}:{k:'c3',chk:3,closed:true,label:c3L});
  return m;
}
/* what a ▲▼ compares with (R6.6): in part 3 the step shown before; in parts 4 and 5 and in Scenarios the healthy run; elsewhere none */
function deltaBase(k){
  if(isP('healthy')) return {v:shownPrev[k],name:{en:`step ${stepsPrev}`,zh:`第 ${stepsPrev} 步`}};
  if(isP('lr')||isP('bugs')||mode==='sandbox'){const h=run({C:3}); return {v:k==='tl'?f3(h.tr[STEPS]):(k==='acc'?fPct(h.acc[STEPS]):null),name:{en:'the healthy run',zh:'健康的运行'}};}
  return null;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
function drawMetrics(){
  const host=$('#td-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k+(m.closed?'x':'')).join();
  host.hidden=!defs.length;
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric${m.chk?' td-chk':''}" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="td-sp"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label;
    if(m.closed){el.classList.add('td-closed'); el.querySelector('.v').innerHTML=`<button type="button" class="btn ghost sm" data-ck="${m.k}">${tx('Open','打开')}</button>`; el.querySelector('.sub').textContent=tx('not opened yet','还没打开'); el.querySelector('.td-sp').innerHTML=''; el.dataset.tipx=tx('Not opened yet. Each check you open counts as one used; the fewer, the better.','还没打开。每打开一个检查，就算用了一次，用得越少越好。'); d.textContent=''; return;}
    el.classList.remove('td-closed'); el.classList.toggle('td-ok',!!m.ok); el.classList.toggle('td-bad',m.chk&&!m.ok);
    el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.querySelector('.td-sp').innerHTML=m.spark||''; el.dataset.tipx=m.tip;
    let dt=''; const base=deltaBase(m.k);
    if(userAct&&m.num&&base&&base.v!=null&&base.v!==m.v){const a=numOf(base.v), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=(A.fmt.delta(a,b,m.k==='acc'?{pts:true,dp:1,base:base.name}:{dp:3,base:base.name})||'').replace(/ pts\b/,' percentage points').replace(/ pt\b/,' percentage point');}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
function mathHTML(){
  const {r,C}=stageRun(), l=lnC(C), floor=Math.log(1+(C-1)/Math.E), hideLr=mode==='challenge'||(mode==='guided'&&step<=1);
  /* H2 (the owner's decision of 2026-10-09): a few short blocks, each a lead line in words, the formula with its name on the left and,
     where the run gives them, the numbers and the result */
  const M=t=>A.ui.math(t), FX=(name,tex)=>`<div class="td-mfx td-math-line"><span class="td-mname">${name}</span>${M(tex)}</div>`, BK=(lead,body)=>`<div class="td-mblk"><p class="td-mlead">${lead}</p>${body}</div>`;
  return BK(tx('<b>Cross-entropy.</b> The loss of one example is high when the model gives the right class a low chance.','<b>cross-entropy。</b>模型给正确类别的概率越低，一个样本的 loss 越高。'),
      FX(tx('loss of one example','一个样本的 loss'),'L = -\\ln\\, p_y')+FX(tx('chance of class c','第 c 类的概率'),'p_c = \\frac{e^{z_c}}{\\sum_j e^{z_j}}')+
      `<p class="td-msym">${tx(`${M('y')} is the right class, and ${M('z_c')} is the model’s logit for class c. The training loss is the average of L over the examples.`,`${M('y')} 是正确的类别，${M('z_c')} 是模型给第 c 类的 logit。训练 loss 是 L 在所有样本上的平均。`)}</p>`)+
    BK(tx(`<b>Guessing.</b> A model that gives each of the ${C} classes the same chance, 1/${C}, has this loss on every example:`,`<b>瞎猜。</b>模型给 ${C} 个类别的概率都是 1/${C} 时，每个样本的 loss 都是：`),
      `<div class="td-mfx td-math-line">${M(`-\\ln\\, \\frac{1}{${C}} = \\ln\\, ${C} = ${fixd(l,4)}`)}</div>`+
      (mode==='challenge'&&!chOpen.includes('c1')?'':`<p>${tx(`The run starts at ${f3(r.tr[0])}.`,`这次运行从 ${f3(r.tr[0])} 开始。`)}</p>`))+
    BK(tx('<b>One training step.</b> Gradient descent moves every weight a little against its gradient.','<b>一个训练步。</b>梯度下降把每个权重往梯度的反方向挪一点。'),
      FX(tx('one step','一步'),'w \\leftarrow w - \\eta \\, g')+
      `<p class="td-msym">${tx(`${M('\\eta')} is the learning rate, now ${T_('lr',hideLr?'?':fLr(S.lr))}, and ${M('g')} is the weight’s gradient. Without zero_grad(), g is the sum of every gradient so far.`,`${M('\\eta')} 是学习率，现在是 ${T_('lr',hideLr?'?':fLr(S.lr))}；${M('g')} 是这个权重的梯度。漏了 zero_grad() 时，g 是到目前为止所有梯度的和。`)}</p>`)+
    BK(tx(`<b>Softmax twice.</b> The second softmax gets chances between 0 and 1 instead of logits, so the right class’s chance has a ceiling, and the loss a floor:`,`<b>softmax 用两次。</b>第二次 softmax 拿到的是 0 到 1 之间的概率，而不是 logit。所以正确类别的概率有上限，loss 也有下限：`),
      FX(tx('highest chance','最高的概率'),`p_y \\leq \\frac{e^1}{e^1 + ${C-1} \\cdot e^0} = \\frac{e}{e + ${C-1}}`)+
      FX(tx('lowest loss','最低的 loss'),`-\\ln \\frac{e}{e + ${C-1}} = \\ln\\left(1 + \\frac{${C-1}}{e}\\right) = ${fixd(floor,4)}`))+
    BK(tx('<b>Gradient norm.</b> One number for how large the whole gradient is.','<b>梯度范数。</b>用一个数值表示整个梯度有多大。'),
      FX(tx('gradient norm','梯度范数'),'‖g‖ = \\sqrt{\\sum_i g_i^2}')+
      `<p class="td-msym">${tx(`The sum runs over all ${2*HID+HID+HID*C+C} parameters; ${M('g_i')} is the gradient of parameter i.`,`对全部 ${2*HID+HID+HID*C+C} 个参数求和，${M('g_i')} 是第 i 个参数的梯度。`)}</p>`);
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS_[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const v=viewNow(), tb=$('#td-tbl',app), panes=$$('.td-pane',app);
  tb.hidden=v!=='tbl'; panes.forEach(p=>{p.hidden=v==='tbl';});
  if(v==='tbl'){const h=sigTableHTML(3,null,false); if(tb._h!==h){tb.innerHTML=h; tb._h=h;}}
  else {const {C}=stageRun();
    $('#td-curve-cap',app).textContent=tx('Loss per training step','每个训练步的 loss');
    $('#td-reg-cap',app).textContent=isP('healthy')?tx(`The model after ${stepsShown} steps: predicted class`,`训练 ${stepsShown} 步后的模型：预测的类别`):tx(`The model after the run: predicted class (${C} groups)`,`训练结束后的模型：预测的类别（${C} 组）`);
    const ch=curveSVG(); const ce=$('#td-curve',app); if(ce._h!==ch){ce.innerHTML=ch; ce._h=ch;}
    const rh=regSVG(); const re=$('#td-reg',app); if(re._h!==rh){re.innerHTML=rh; re._h=rh;}}
  drawMetrics();
  const mb=$('#td-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you only open checks and pick the problem, in the task card. The teammate’s run cannot be changed, and its learning rate is not shown.','挑战里只能在任务卡里打开检查、选出问题。同事的运行不能改，它的学习率也不告诉你。');
  if(mode!=='guided') return '';
  const id=STEPS_[step].id;
  if(id==='mission'||id==='predict') return tx('Fixed in this part: the mission’s run, with its bug hidden. You can change everything in Scenarios.','这一部分固定：任务里的那次运行，bug 藏起来了。在应用场景里可以改所有设置。');
  if(id==='healthy') return tx('Fixed in this part: learning rate 0.3, no bug, 300 training examples. Move through the run in the card.','这一部分固定：学习率 0.3，没有 bug，300 个训练样本。在卡片里沿着这次运行往前走。');
  if(id==='lr') return tx('In this part you pick the learning rate in the card; no bug, 300 training examples. You can change the settings in Scenarios.','这一部分只在卡片里选学习率；没有 bug，300 个训练样本。在应用场景里可以改这些设置。');
  if(id==='bugs') return tx('In this part you pick the bug in the card, at learning rate 0.3 with 300 training examples. You can change the settings in Scenarios.','这一部分只在卡片里选 bug，学习率 0.3，300 个训练样本。在应用场景里可以改这些设置。');
  if(id==='remember') return tx('Fixed in this part: the healthy run. You can change the settings in Scenarios.','这一部分固定：健康的运行。在应用场景里可以改这些设置。');
  return tx('Fixed in this part: the table runs every check on six fixed runs. You can change the settings in Scenarios.','这一部分固定：表格在六次固定的运行上跑每个检查。在应用场景里可以改这些设置。');
}
function syncControls(){
  $$('.seg-ctl[data-key="bug"]',app).forEach(g=>A.ui.setPressed(g,S.bug));
  $$('.seg-ctl[data-key="lr"]',app).forEach(g=>A.ui.setPressed(g,String(S.lr)));
  $$('.seg-ctl[data-key="k"]',app).forEach(g=>A.ui.setPressed(g,String(S.k)));
  const c=$('#td-controls',app), sb=mode==='sandbox';
  c.classList.toggle('locked',!sb);
  ['#td-ctl-bug-wrap','#td-ctl-lr-wrap','#td-ctl-k-wrap','#td-ctl-reset-wrap'].forEach(id=>{const e=$(id,app); if(e) e.hidden=!sb;});
  const ln=$('#td-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const {r,upTo}=stageRun(); A.announce(curveAria(r,upTo,stageRun().C));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p); if(!LRS.includes(S.lr)) S.lr=LR0; if(!BUGS.includes(S.bug)) S.bug='none'; if(![0,SMALL_K].includes(S.k)) S.k=0; render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setSteps(t){stepsPrev=stepsShown; stepsShown=t; render();}
function setLr(v){lrPick=v; lrTried.add(v); S={...BASE,lr:+v}; render();}
function setBug(v){bugPick=v; bugTried.add(v); S={...BASE,bug:v}; render();}
function setChk(v){chkPick=v; chkTried.add(v); render();}
function enterStep(i){
  step=Math.max(0,Math.min(STEPS_.length-1,i));
  const s=STEPS_[step]; if(s.enter) S={...BASE,...s.enter()};
  const st=$('#td-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#td-rail .step.current',app)||$('#td-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>{if(!app) return; if(mode==='guided') revealBox($('#td-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#td-insight',app)); else revealBox($('#td-ch-res',app));};
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chAns=null; chOpen=[];}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#td-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const tokOf=s=>[LRS.indexOf(s.lr),BUGS.indexOf(s.bug),s.k].join('_');
const token=()=>tokOf(S);
function parse(tok){const p=String(tok||'').split('_').map(Number); if(p.length!==3) return null; const [li,bi,k]=p;
  if(!Number.isInteger(li)||li<0||li>2||!Number.isInteger(bi)||bi<0||bi>3||![0,SMALL_K].includes(k)) return null; return {lr:LRS[li],bug:BUGS[bi],k};}

/* =====================================================================
   interview drill: seeded, from the engine (R6.6)
   ===================================================================== */
function makeDrill(seed){const r=mulberry(seed*104729+7), C=[2,5,10,100,1000,32000][Math.floor(r()*6)], k=['c1','c2','c3'][Math.floor(r()*3)]; return {seed,C,ln:Math.log(C)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#td-drill-q',app), a=$('#td-drill-a',app);
  qd.innerHTML=tx(`A classifier over <b>${A.fmt.int(d.C)}</b> classes is freshly initialized with a small last layer. Its first logged loss is <b>${fixd(d.ln+0.004,3)}</b>, and after 1,000 steps it is <b>${fixd(d.ln-0.002,3)}</b>. What do these numbers tell you, and which check do you run first?`,
                  `一个 <b>${A.fmt.int(d.C)}</b> 类的分类器刚初始化，最后一层很小。它记录的第一个 loss 是 <b>${fixd(d.ln+0.004,3)}</b>，1,000 步之后是 <b>${fixd(d.ln-0.002,3)}</b>。这些数说明了什么？你先跑哪个检查？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>Both are ln ${A.fmt.int(d.C)} = ${fixd(d.ln,3)}: the start is right, and since then the model has done no better than guessing 1/${A.fmt.int(d.C)} for every class.</b></p><p>Guessing 1/${A.fmt.int(d.C)} for every class costs ${A.ui.math(`-\\ln\\, \\frac{1}{${A.fmt.int(d.C)}} = ${fixd(d.ln,3)}`)}, so the initialization is fine (check 1 passes at step 0) but the run never left it. Run check 2 next: overfit a tiny batch, a few examples with their real labels, through the same step. If that works, the labels are not reaching the model paired with their inputs (shuffled, misaligned, or a wrong column); if it fails too, the weights are not being updated: a missing backward() or frozen parameters leave the gradient norm (check 3) at 0, and a missing step() leaves it unchanged from step to step, because the weights never move.</p>`,
    `<p><b>两个都是 ln ${A.fmt.int(d.C)} = ${fixd(d.ln,3)}：开头是对的，之后模型一直不比“每类都猜 1/${A.fmt.int(d.C)}”好。</b></p><p>每类都猜 1/${A.fmt.int(d.C)}，代价是 ${A.ui.math(`-\\ln\\, \\frac{1}{${A.fmt.int(d.C)}} = ${fixd(d.ln,3)}`)}，所以初始化没问题（第 0 步检查 1 通过），但这次运行从没离开它。接着跑检查 2：用同一个训练步过拟合一个小 batch（几个带真实标签的样本）。能学会，说明标签没有和输入正确配对地送到模型（打乱了、错位了，或者用错了列）；也学不会，说明权重没有被更新：漏了 backward() 或参数被冻结，梯度范数（检查 3）是 0；漏了 step()，权重一直不动，梯度范数每一步都一样。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const TERM={lr:()=>`<div>${esc(TIP.lr())}</div><div class="fx">${A.ui.math('w \\leftarrow w - \\eta \\, g')}</div>`};
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='lr'){if(mode==='challenge') return tx('Not shown in the challenge, because the learning rate may itself be the problem.','挑战里不显示学习率，因为它本身就可能是问题所在。');
    if(mode==='guided') return isP('lr')?tx('Click to jump to the learning rates in the card.','点击跳到卡片里的学习率。'):tx('Fixed in this part; part 4 and Scenarios let you change it.','这一部分固定不变；第 4 部分和应用场景里可以改。');
    return tx('Click to jump to “Learning rate”.','点击跳到“学习率”。');}
  return '';
}
function jump(k,src){
  if(!app) return;
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#td-controls',app)); return;}
  if(k==='lr'){if(mode==='sandbox'){A.ui.jumpToEl($('#td-ctl-lr',app)); return;} if(isP('lr')){A.ui.jumpToEl($('#td-ic-lr',app)); return;} if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){holdPlace(btn,()=>byLearner(()=>setState({...BASE}))); return;}
  if(a==='submit'){if(!chAns) return; chSubmitted=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='next-myst'){chIdx=(chIdx+1)%MYST.length; chOpen=[]; chAns=null; chSubmitted=false; chReveal=false; renderRail();
    requestAnimationFrame(()=>{if(app) A.ui.reveal(innerWidth<=760?$('#td-stage',app):$('#td-rail .railcard',app));}); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#td-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
function openCheck(c){if(mode!=='challenge'||chOpen.includes(c)) return; chOpen.push(c); renderRail();}
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS_[step]; if(s.showMe){byLearner(()=>s.showMe()); revealSoon();} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#td-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null){const box=b.closest('.td-lv'); if(!box) return; const f=()=>byLearner(()=>{if(box.id==='td-ic-lr') setLr(d.lever); else if(box.id==='td-ic-bug') setBug(d.lever); else setChk(d.lever);});
      if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.ck){openCheck(d.ck); return;}
    if(d.ans){chAns=d.ans; if(chSubmitted) chReveal=chReveal||chAns===mystK(); render(); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='steps'){byLearner(()=>setSteps(+d.v)); return;}
      const f=()=>byLearner(()=>setState(k==='lr'?{lr:+d.v}:(k==='k'?{k:+d.v}:{bug:d.v})));
      if(b.closest('#td-controls')) holdPlace(b,f); else f(); return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#td-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#td-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the steps shown, the
   choices tried, the mystery run with the checks opened and the answer, the drill; the runs are recomputed from these */
const okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o);
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,stepsShown,lrPick,lrTried:[...lrTried],bugPick,bugTried:[...bugTried],chkPick,chkTried:[...chkTried],
    chIdx,chOpen:chOpen.slice(),chAns,chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chAns===mystK()&&tellsApart(mystK(),4,chOpen)};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const cfg=x=>okObj(x)?parse(tokOf(x)):null, inV=(arr,v)=>arr.some(x=>x.v===v);
    const tried=(a,arr,cur,first)=>{const t=new Set(Array.isArray(a)?a.filter(v=>inV(arr,v)):[]); if(first!=null&&!t.size) t.add(first); if(cur!=null) t.add(cur); return t;};
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS_.length?o.step:0;
    S={...(cfg(o.S)||BASE)}; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS_.length&&o.observed[k]===true) observed[k]=true;
    stepsShown=SNAPS.includes(o.stepsShown)?o.stepsShown:0; stepsPrev=stepsShown;
    lrPick=inV(LRV,o.lrPick)?o.lrPick:'0.3'; lrTried=tried(o.lrTried,LRV,lrPick,'0.3');
    bugPick=BUGS.includes(o.bugPick)?o.bugPick:'none'; bugTried=new Set(['none',...(Array.isArray(o.bugTried)?o.bugTried.filter(v=>BUGS.includes(v)):[]),bugPick]);
    chkPick=inV(CKV,o.chkPick)?o.chkPick:null; chkTried=tried(o.chkTried,CKV,chkPick,null);
    saved=mode==='challenge'?cfg(o.saved):null;
    chIdx=Number.isInteger(o.chIdx)&&o.chIdx>=0&&o.chIdx<MYST.length?o.chIdx:0;
    chOpen=Array.isArray(o.chOpen)?[...new Set(o.chOpen.filter(c=>inV(CKV,c)))]:[]; chAns=KINDS.includes(o.chAns)?o.chAns:null;
    chSubmitted=mode==='challenge'&&!!chAns&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tokOf(p))) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#td-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#td-rail',app));},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS_[step]; if(s.enter&&step===0) S={...BASE,...s.enter()};}
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const d=$('#td-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#td-math',app).open=true;}},
  termTip(k,el){if(k==='dfx'){const f=el&&DFXT[el.getAttribute('data-term')]; return f?f():'';} return TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(viewNow()==='tbl'?$('#td-tbl',app):$('#td-curve',app)),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#td-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(viewNow()==='tbl'){vis.innerHTML=''; read.innerHTML=esc(chkPick?tx(`Check ${chkPick.slice(1)}: ${ckRes(chkPick)||L(CKV.find(o=>o.v===chkPick).t)}`,`检查 ${chkPick.slice(1)}：${ckRes(chkPick)||L(CKV.find(o=>o.v===chkPick).t)}`):tx('The three checks on the healthy run and the five broken ones','三个检查跑在健康的运行和五次弄坏的运行上')); return;}
      const {r,upTo,C}=stageRun(), e=lastFinite(r.tr.slice(0,upTo+1));
      vis.innerHTML=`<div class="td-dk">${spark(r.tr.slice(0,upTo+1),false)}</div>`;
      const lnT=mode!=='challenge'||chOpen.includes('c1');
      read.innerHTML=esc(r.dead>=0&&r.dead<=upTo?tx(`Training loss: ${f3(r.tr[0])} at step 0, NaN from step ${r.dead}`,`训练 loss：第 0 步 ${f3(r.tr[0])}，第 ${r.dead} 步起 NaN`):tx(`Training loss: ${f3(r.tr[0])} at step 0, ${f3(e.v)} at step ${e.i}${lnT?` (ln ${C} = ${fixd(lnC(C),3)})`:''}`,`训练 loss：第 0 步 ${f3(r.tr[0])}，第 ${e.i} 步 ${f3(e.v)}${lnT?`（ln ${C} = ${fixd(lnC(C),3)}）`:''}`));}
  }
});

/* =====================================================================
   read-only hook for tests and the home tile
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o,(k,v)=>typeof v==='number'&&!isFinite(v)?(Number.isNaN(v)?'NaN':(v>0?'inf':'-inf')):v));
window.__td=Object.freeze({
  data:C=>deep(dataOf(C)), init:C=>deep(initP(C)),
  run:cfg=>{const r=run(cfg); return deep({tr:r.tr,va:r.va,acc:r.acc,gn:r.gn,dead:r.dead,inf:r.inf});},
  runRaw:cfg=>run(cfg),
  check1:cfg=>deep(check1(run(cfg))), check2:cfg=>{const c=check2(cfg); return deep({end:c.end,pass:c.pass,dead:c.dead,inf:c.inf,cat:c.cat});}, check3:cfg=>deep(check3(run(cfg))),
  curveCat:cfg=>curveCat(run(cfg)), curveShape:cfg=>curveShape(run(cfg)), sig:(k,C)=>sigOf(k,C), minChecks, tellsApart, confirmOf, pairCheck, regions:cfg=>regions(run(cfg).last),
  consts:Object.freeze({HID,STEPS,NTR,NVA,OF_K,OF_STEPS,OF_PASS,INIT_SEED,SHUF_SEED,LRS:LRS.slice(),LR0,SMALL_K,MYST:MYST.slice(),KINDS:KINDS.slice(),DSPEC:deep(DSPEC),STEP_IDS:STEPS_.map(s=>s.id),PRED_RIGHT,GRID,EXT}),
  drill:s=>deep(makeDrill(s)), exits:()=>{const e=EXITS(); return {transferA:e.transfer.map(q=>q.a)};},
  state:()=>({...S,mode,step,pred,observed:{...observed},stepsShown,lrPick,lrTried:[...lrTried],bugPick,bugTried:[...bugTried],chkPick,chkTried:[...chkTried],chIdx,chOpen:chOpen.slice(),chAns,chSubmitted,chReveal,myst:mystK(),chMin:chMin()})
});
})();
