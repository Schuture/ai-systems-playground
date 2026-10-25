/* checkpointing (world training): Your run crashed at step 90,000: how do you resume it?
   Fidelity: Simulated (toy run). A small model is really trained in the browser, deterministically (specs/checkpointing.md):
   - data: 3 spiral arms in 2D, 1,440 training and 300 validation points (mulberry32 seeds 11 and 23);
   - model: MLP 2 → 64 ReLU → 3, softmax cross-entropy, dropout 0.1 on the hidden layer (RNG seed 99, 64 draws per sample);
   - AdamW (β1 0.9, β2 0.999, ε 1e-8, weight decay 0.01), learning rate: 30 warmup steps to 0.05, cosine to 0.005 at step 600;
   - batch 24, one seeded shuffle per epoch (seed 7000 + epoch): 60 steps per epoch, 600 steps = 10 epochs;
   - a checkpoint every 100 steps holds weights, optimizer state (m, v, step count), scheduler step, data position, RNG state.
   Resuming without a part: optimizer → m = v = 0 and step count 0; scheduler → a new scheduler counts from 0 (warmup again);
   data position → the loader restarts the current epoch from its first batch; RNG → the generator is seeded again with 99.
   With all parts kept, the resumed run equals the uninterrupted one bit for bit (checked by tests/checkpointing_numbers.py).
   Save-interval model (first order, Young 1974): wasted share = δ/T + (T/2 + R)/M; optimum T* = √(2δM).
   window.__ckp is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='checkpointing';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const NB='‑';

/* =====================================================================
   engine: a small model trained for real
   ===================================================================== */
const E=Object.freeze({NTR:1440,NEV:300,B:24,H:64,C:3,STEPS:600,WARM:30,PEAK:0.05,MINR:0.1,B1:0.9,B2:0.999,EPS:1e-8,WD:0.01,PD:0.1,NOISE:0.2,
  SD:11,SE:23,SI:5,SR:99,SS:7000,EVERY:5,SAVE:100,CRASH:450,CK:400,BAD0:470,BADN:5,BADX:8});
const EPOCH=E.NTR/E.B, NP=2*E.H+E.H+E.C*E.H+E.C;
const EP_CK=Math.floor(E.CK/EPOCH)+1; /* the epoch (counted from 1) that the step-400 checkpoint is in: 7 */
function mul(seed){let a=seed>>>0; const f=()=>{a=(a+0x6D2B79F5)>>>0; let t=Math.imul(a^(a>>>15),1|a); t=(t+Math.imul(t^(t>>>7),61|t))^t; return ((t^(t>>>14))>>>0)/4294967296;};
  f.get=()=>a; f.set=v=>{a=v>>>0;}; return f;}
function gen(n,seed){const r=mul(seed), X=new Float64Array(2*n), Y=new Int32Array(n);
  for(let i=0;i<n;i++){const y=i%E.C, t=r(), rad=0.15+1.35*t, ang=2*Math.PI*y/E.C+t*3.4;
    const nx=(r()+r()+r()-1.5)*E.NOISE, ny=(r()+r()+r()-1.5)*E.NOISE;
    X[2*i]=rad*Math.cos(ang)+nx; X[2*i+1]=rad*Math.sin(ang)+ny; Y[i]=y;}
  return {X,Y};}
let DATA=null;
function data(){if(!DATA){const t=gen(E.NTR,E.SD), v=gen(E.NEV,E.SE); DATA={X:t.X,Y:t.Y,XE:v.X,YE:v.Y,perm:{}};} return DATA;}
function perm(e){const D=data(); if(D.perm[e]) return D.perm[e];
  const r=mul(E.SS+e), p=new Int32Array(E.NTR); for(let i=0;i<E.NTR;i++) p[i]=i;
  for(let i=E.NTR-1;i>0;i--){const j=Math.floor(r()*(i+1)); const t=p[i]; p[i]=p[j]; p[j]=t;}
  D.perm[e]=p; return p;}
function init(){const r=mul(E.SI), f=new Float64Array(NP), H=E.H, a1=1/Math.sqrt(2), a2=1/Math.sqrt(H);
  for(let i=0;i<H;i++) for(let j=0;j<2;j++) f[i*2+j]=(2*r()-1)*a1*1.7;
  const o=3*H; for(let c=0;c<E.C;c++) for(let i=0;i<H;i++) f[o+c*H+i]=(2*r()-1)*a2*1.7;
  return f;}
const lrAt=s=>s<E.WARM?E.PEAK*(s+1)/E.WARM:E.PEAK*(E.MINR+(1-E.MINR)*0.5*(1+Math.cos(Math.PI*((s-E.WARM)/(E.STEPS-E.WARM)))));
const LRX=lrAt(E.STEPS-1-E.CK)/lrAt(E.STEPS-1); /* a new scheduler at step 400, against the uninterrupted one, at the last step: about 8 times */
/* one pass over n samples: mean cross-entropy, and the gradient if g is given */
const hb=new Float64Array(E.H), pb=new Float64Array(E.H), lo=new Float64Array(E.C), ex=new Float64Array(E.C), dl=new Float64Array(E.C);
function pass(f,xs,ys,n,mask,g){
  const H=E.H, C=E.C, o2=3*H, ob=o2+C*H; let Ls=0;
  for(let k=0;k<n;k++){const x0=xs[2*k], x1=xs[2*k+1];
    for(let i=0;i<H;i++){const z=f[i*2]*x0+f[i*2+1]*x1+f[2*H+i]; pb[i]=z; let a=z>0?z:0; if(mask) a=a*mask[k*H+i]; hb[i]=a;}
    let mx=-Infinity; for(let c=0;c<C;c++){let s=0; for(let i=0;i<H;i++) s+=f[o2+c*H+i]*hb[i]; lo[c]=s+f[ob+c]; if(lo[c]>mx) mx=lo[c];}
    let se=0; for(let c=0;c<C;c++){ex[c]=Math.exp(lo[c]-mx); se+=ex[c];}
    const y=ys[k]; Ls+=-(lo[y]-mx-Math.log(se));
    if(!g) continue;
    for(let c=0;c<C;c++) dl[c]=ex[c]/se-(c===y?1:0);
    for(let c=0;c<C;c++){for(let i=0;i<H;i++) g[o2+c*H+i]+=dl[c]*hb[i]/n; g[ob+c]+=dl[c]/n;}
    for(let i=0;i<H;i++){let dh=0; for(let c=0;c<C;c++) dh+=dl[c]*f[o2+c*H+i]; if(mask) dh*=mask[k*H+i]; if(pb[i]<=0) continue;
      g[i*2]+=dh*x0/n; g[i*2+1]+=dh*x1/n; g[2*H+i]+=dh/n;}}
  return Ls/n;}
const evalLoss=f=>{const D=data(); return pass(f,D.XE,D.YE,E.NEV,null,null);};
const fresh=()=>({f:init(),m:new Float64Array(NP),v:new Float64Array(NP),t:0,sched:0,epoch:0,pos:0,rng:E.SR,step:0});
const copyS=s=>({f:Float64Array.from(s.f),m:Float64Array.from(s.m),v:Float64Array.from(s.v),t:s.t,sched:s.sched,epoch:s.epoch,pos:s.pos,rng:s.rng,step:s.step});
/* train from state st to step `until`; bad: steps whose batch is corrupted; skip: steps whose update is skipped */
function train(st0,until,bad,skip){
  const D=data(), st=copyS(st0), r=mul(st.rng), B=E.B, H=E.H;
  const xs=new Float64Array(2*B), ys=new Int32Array(B), mask=new Float64Array(B*H), g=new Float64Array(NP), keep=1/(1-E.PD);
  const ev={}, snaps={}, seen={};
  while(st.step<until){const s=st.step;
    if(s%E.EVERY===0) ev[s]=evalLoss(st.f);
    if(s%E.SAVE===0){const c=copyS(st); c.rng=r.get(); snaps[s]=c;}
    const p=perm(st.epoch);
    for(let k=0;k<B;k++){const id=p[st.pos*B+k]; xs[2*k]=D.X[2*id]; xs[2*k+1]=D.X[2*id+1]; ys[k]=D.Y[id]; const key=st.epoch*E.NTR+id; seen[key]=(seen[key]||0)+1;}
    st.pos++; if(st.pos===EPOCH){st.pos=0; st.epoch++;}
    if(bad&&bad.has(s)) for(let k=0;k<B;k++){xs[2*k]*=E.BADX; xs[2*k+1]*=E.BADX; ys[k]=(ys[k]+1)%E.C;}
    for(let k=0;k<B;k++) for(let i=0;i<H;i++) mask[k*H+i]=r()>=E.PD?keep:0;
    if(skip&&skip.has(s)){st.step++; st.sched++; continue;}
    g.fill(0); pass(st.f,xs,ys,B,mask,g);
    const lr=lrAt(st.sched); st.t++; const t=st.t, c1=1-Math.pow(E.B1,t), c2=1-Math.pow(E.B2,t), f=st.f, m=st.m, v=st.v;
    for(let i=0;i<NP;i++){m[i]=E.B1*m[i]+(1-E.B1)*g[i]; v[i]=E.B2*v[i]+(1-E.B2)*g[i]*g[i];
      const mh=m[i]/c1, vh=v[i]/c2; f[i]=f[i]-lr*E.WD*f[i]-lr*mh/(Math.sqrt(vh)+E.EPS);}
    st.sched++; st.step++;}
  ev[st.step]=evalLoss(st.f); const c=copyS(st); c.rng=r.get(); snaps[st.step]=c;
  return {ev,snaps,seen};}
/* what a checkpoint holds: o = optimizer state, s = scheduler step, d = data position, r = RNG state (weights always) */
const PARTS=['o','s','d','r'];
function restore(ck,keep){const st=copyS(ck);
  if(!keep.o){st.m=new Float64Array(NP); st.v=new Float64Array(NP); st.t=0;}
  if(!keep.s) st.sched=0;
  if(!keep.d) st.pos=0;
  if(!keep.r) st.rng=E.SR;
  return st;}
const CACHE={};
function base(){if(!CACHE.base) CACHE.base=train(fresh(),E.STEPS,null,null); return CACHE.base;}
const maskKey=k=>PARTS.map(p=>k[p]?1:0).join('');
/* the resumed run from step 400 with the given parts kept, and its comparison with the run that never stopped */
function resumed(keep){const key='r'+maskKey(keep); if(CACHE[key]) return CACHE[key];
  const b=base(), run=train(restore(b.snaps[E.CK],keep),E.STEPS,null,null);
  let maxGap=0, maxRel=0, worseAll=true, same=true;
  for(const k in run.ev){const s=+k; if(s<E.CK) continue; const d=run.ev[k]-b.ev[k]; if(d!==0) same=false;
    if(d>maxGap){maxGap=d;} const rel=d/b.ev[k]; if(rel>maxRel) maxRel=rel; if(s>=E.CK+20&&d<=0) worseAll=false;}
  /* data: per epoch, samples read twice, and samples of a complete epoch never read */
  const before=train(fresh(),E.CK,null,null).seen, all={};
  for(const k in before) all[k]=(all[k]||0)+before[k]; for(const k in run.seen) all[k]=(all[k]||0)+run.seen[k];
  let twice=0, never=0; for(const k in all) if(all[k]>=2) twice++;
  for(let e=0;e<E.STEPS/EPOCH;e++) for(let i=0;i<E.NTR;i++) if(!all[e*E.NTR+i]) never++;
  const out={ev:run.ev,maxGap,maxRel,same,worseAll,final:run.ev[E.STEPS],final0:b.ev[E.STEPS],twice,never};
  CACHE[key]=out; return out;}
/* part 5: five corrupted batches at steps 470–474; checkpoints at 400 and 500 */
const BAD=new Set(Array.from({length:E.BADN},(_,i)=>E.BAD0+i));
function spikeRun(){if(!CACHE.spike) CACHE.spike=train(fresh(),E.STEPS,BAD,null); return CACHE.spike;}
const ALL={o:1,s:1,d:1,r:1};
function remedy(opt){const key='sp'+opt; if(CACHE[key]) return CACHE[key];
  const sp=spikeRun(); let run, from;
  if(opt==='latest'){from=500; run={ev:{}}; for(const k in sp.ev) if(+k>=500) run.ev[k]=sp.ev[k];}
  else {from=E.CK; run=train(restore(sp.snaps[E.CK],ALL),E.STEPS,BAD,opt==='skip'?BAD:null);}
  let peak=0; for(const k in sp.ev) if(+k>=E.BAD0&&sp.ev[k]>peak) peak=sp.ev[k];
  const out={ev:run.ev,from,final:run.ev[E.STEPS],peak,pre:sp.ev[E.BAD0]};
  CACHE[key]=out; return out;}

/* =====================================================================
   save-interval model (first order): wasted share = δ/T + (T/2 + R)/M
   ===================================================================== */
const P70=70553706496, P8=8030261248, P405=405853388800, BPC=14; /* checkpoint: BF16 weights 2 + FP32 copy 4 + m 4 + v 4 */
const SIZE70=P70*BPC, SIZE8=P8*BPC, SIZE405=P405*BPC;
const WINT=Object.freeze({bw:10e9,dAsync:10,R:600,M:12*3600});
const waste=(d,T,M,R)=>({save:d/T,lost:T/(2*M),restart:R/M,total:d/T+T/(2*M)+R/M});
const tOpt=(d,M)=>Math.sqrt(2*d*M);
const INTS=[5,10,15,20,30,45,60,90,120,180,240,360,480]; /* minutes */
const CASES={
 c70:{size:SIZE70,M:WINT.M,R:WINT.R,model:`Llama${NB}3.1${NB}70B`,gpus:1024},
 c8:{size:SIZE8,M:32*7*24*3600,R:WINT.R,model:`Llama${NB}3.1${NB}8B`,gpus:8}};
const dOf=(c,mode)=>mode==='async'?WINT.dAsync:CASES[c].size/WINT.bw;
/* the challenge: Llama-3.1-405B, M = 3 h, R = 10 min, 50 GB/s, 30 TB of storage, roll back at least 1 h */
const CH=Object.freeze({size:SIZE405,M:3*3600,R:600,bw:50e9,dAsync:10,store:30e12,back:60,goal:0.12,ints:[5,10,15,20,30,60],keeps:[2,3,4,5,6],start:{mode:'sync',T:60,k:2}});
function chEval(mode,T,k){const d=mode==='async'?CH.dAsync:CH.size/CH.bw, w=waste(d,T*60,CH.M,CH.R), store=k*CH.size, back=(k-1)*T;
  const fits=store<=CH.store, rb=back>=CH.back; return {mode,T,k,d,w:w.total,parts:w,store,back,fits,rb,ok:fits&&rb&&w.total<=CH.goal+1e-12};}
function chAll(){const a=[]; ['sync','async'].forEach(m=>CH.ints.forEach(T=>CH.keeps.forEach(k=>a.push(chEval(m,T,k))))); return a;}
function chBest(){return chAll().filter(e=>e.fits&&e.rb).sort((a,b)=>a.w-b.w||a.k-b.k)[0];}
/* every plan that meets the goal, least waste first: the only plans the result lists (R1) */
function chPass(){return chAll().filter(e=>e.ok).sort((a,b)=>a.w-b.w||a.k-b.k||a.T-b.T);}
/* the plans one setting away from e (save mode, interval or copies kept) that meet the goal, least waste first */
function chOne(e){const c=[];
  ['sync','async'].forEach(m=>{if(m!==e.mode) c.push({f:'mode',v:m,e:chEval(m,e.T,e.k)});});
  CH.ints.forEach(T=>{if(T!==e.T) c.push({f:'T',v:T,e:chEval(e.mode,T,e.k)});});
  CH.keeps.forEach(k=>{if(k!==e.k) c.push({f:'k',v:k,e:chEval(e.mode,e.T,k)});});
  return c.filter(x=>x.e.ok).sort((a,b)=>a.e.w-b.e.w||a.e.k-b.e.k);}
/* the fewest checkpoints that reach back 1 hour at interval T (minutes): 13 at 5 min, 7 at 10, 5 at 15 */
const chKmin=T=>Math.max(2,Math.ceil(CH.back/T)+1);
/* the two limits on the challenge's curve (R15): the most checkpoints the storage holds (kf), and the shortest interval at which kf of
   them reach back CH.back (tmin, minutes). A shorter interval breaks the storage limit or the rollback limit, whatever is kept */
function chLimit(){const kf=Math.min(Math.max(...CH.keeps),Math.floor(CH.store/CH.size)); return {kf,tmin:CH.back/Math.max(1,kf-1)};}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(v,d=1)=>fixd(v*100,d)+'%';
const TB=b=>fixd(b/1e12,2)+' TB';
const GB=b=>fixd(b/1e9,1)+' GB';
/* sizes on cards: TB from half a terabyte up, else GB; in formula lines one more digit, so that each line can be recomputed from the
   numbers it shows (R6.6): sizeX gives the shown text and the shown value (0.1 GB steps), δ and T* are then computed from that value */
const sizeCard=b=>b>=5e11?TB(b):GB(b);
const sizeX=b=>({t:b>=5e11?fixd(b/1e12,4)+' TB':fixd(b/1e9,1)+' GB',v:Math.round(b/1e8)*1e8});
const fnum=(v,maxd=3)=>{const t=fixd(v,maxd); return t.indexOf('.')<0?t:t.replace(/0+$/,'').replace(/\.$/,'');};
const dShown=(b,bw)=>sizeX(b).v/bw;
const mins=s=>{const m=s/60; return m>=90?tx(`${fixd(m/60,1)} h`,`${fixd(m/60,1)} 小时`):tx(`${fixd(m,m<10?1:0)} min`,`${fixd(m,m<10?1:0)} 分钟`);};
const minsT=T=>T>=60&&T%60===0?tx(`${T/60} h`,`${T/60} 小时`):tx(`${T} min`,`${T} 分钟`);
const lossF=v=>v>=1?fixd(v,2):fixd(v,3);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
const MX=t=>`<div class="ckp-fx">${A.ui.math(t)}</div>`;
const big=n=>A.fmt.big?A.fmt.big(n):fint(n);

/* =====================================================================
   module state
   ===================================================================== */
/* view: 'loss' (resume), 'files' (atomic save), 'waste' (interval), 'spike' (roll back) */
const BASE={view:'loss',o:0,s:0,d:0,r:0,atom:'plain',mode:'sync',T:180,cs:'c70',fix:'latest'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=1, drillOpen=false;
let partsTried=new Set(), atomTried=new Set(['plain']), tMoved=false, asyncTried=false, fixTried=new Set(['latest']), remView='weights';
let chS={...CH.start}, allOpen=false;
/* the whole run plays at Fast by default (R3.9): 40 / 20 / 10 s */
let playT=1, playing=false, spd='fast', raf=0, lastT=0, relang=false, from0=true;
const watched=new Set();
const DUR={slow:40000,mid:20000,fast:10000};
const shownPrev={};
const keepOf=s=>({o:s.o,s:s.s,d:s.d,r:s.r});
const sigOf=s=>s.view==='spike'?'sp:'+s.fix:(s.view==='loss'?'ls:'+maskKey(keepOf(s)):'');
const animated=()=>S.view==='loss'||S.view==='spike';

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 ckpt:()=>tx('A checkpoint is a save of the training run: everything needed to carry on from that step as if nothing had happened.','检查点（checkpoint）就是训练的存档：从这一步接着训所需要的全部东西，接上以后就像没停过一样。'),
 adamw:()=>tx('AdamW keeps two running averages for every parameter, m (of the gradient) and v (of the squared gradient), and divides each update by the square root of v; see the optimizers lab.','AdamW 给每个参数记两个滑动平均：m（梯度的）和 v（梯度平方的），每次更新都要除以 v 的平方根。详见“优化器”Lab。'),
 sched:()=>tx('The learning-rate schedule sets the learning rate from the step number: here it rises over the first 30 steps (warmup), then falls along a cosine to a tenth of its peak at step 600.','学习率调度按步数给出学习率。这里前 30 步从 0 升到峰值（预热），然后沿余弦曲线下降，到第 600 步降到峰值的十分之一。'),
 dropout:()=>tx('Dropout switches off a random tenth of the hidden units at every training step; which ones comes from a random-number generator.','dropout 在每一步训练时随机关掉十分之一的隐藏单元；关掉哪些，由随机数生成器决定。'),
 epoch:()=>tx('An epoch is one pass over all 1,440 training samples in a shuffled order: 60 steps of 24 samples here.','一个 epoch 是把 1,440 个训练样本打乱顺序后全部过一遍：这里每步 24 个，一个 epoch 60 步。'),
 val:()=>tx('The validation loss is measured every 5 steps on 300 points never used in training, with dropout off, so it moves only when the weights do.','验证集 loss 每 5 步测一次，用的是训练时从来不用的 300 个点，并且关掉 dropout，所以只有权重变了它才会变。'),
 rename:()=>tx('On a POSIX file system, renaming within one file system is atomic: other programs see either the old name or the new one, never half a rename.','在 POSIX 文件系统里，同一个文件系统内的改名是原子操作。别的程序看到的要么是旧名字，要么是新名字，不会看到改了一半。'),
 bpc:()=>tx('A checkpoint holds the BF16 weights (2 bytes per parameter), the FP32 copy of the weights (4) and Adam’s m and v (4 each): 14 bytes. The FP32 copy and Adam’s m and v together are the optimizer state. The FP32 gradients are not saved, because the next step computes them anew; with them, training holds 18 bytes per parameter.','存档里有 BF16 权重（每个参数 2 字节）、FP32 权重副本（4 字节）、Adam 的 m 和 v（各 4 字节），共 14 字节。FP32 权重副本和 Adam 的 m、v 合起来就是优化器状态。FP32 梯度不用存，因为下一步会重新算。训练时算上梯度，每个参数是 18 字节。'),
 wbw:()=>tx('An assumption: the speed at which the whole job writes a checkpoint to the shared file system. BLOOM wrote its 2.3 TB checkpoint from 384 GPUs in about 40 s, about 57 GB/s. Shared file systems differ a lot, so this lab takes a slower 10 GB/s.','这是整个任务往共享文件系统写存档的速度，是假设的值。BLOOM 用 384 张卡约 40 秒写完 2.3 TB 的存档，约每秒 57 GB。共享文件系统之间差别很大，所以这个 Lab 取慢一些的每秒 10 GB。'),
 mtbf:()=>tx('M, the mean time between failures of the whole job: one failed GPU or machine stops all of them. OPT-175B’s team, on 1,024 A100s, reports about 2 machines going down every day, so M ≈ 12 hours here.','M 是整个任务平均多久出一次故障：一张卡或一台机器坏了，所有卡都得停。OPT-175B 团队在 1,024 张 A100 上训练时，大约每天坏 2 台机器，所以这里取 M ≈ 12 小时。'),
 restart:()=>tx('R, the time a restart takes: swap the failed machine, start every process, load the checkpoint. This lab assumes 10 minutes for an automated restart; OPT-175B and BLOOM also report restarts that took hours when a person had to step in.','R 是重启一次要的时间：换掉坏机器，拉起所有进程，读回存档。这个 Lab 假设自动重启要 10 分钟；OPT-175B 和 BLOOM 也记录过要人来处理、停了几个小时的情况。'),
 m8:()=>tx('An assumption scaled from BLOOM, which saw 1–2 GPU failures a week on 384 GPUs. At 1.5 a week, 8 GPUs fail about once every 32 weeks.','这是按 BLOOM 折算的假设。BLOOM 在 384 张卡上，平均每周有 1–2 张 GPU 出故障。按每周 1.5 张算，8 张卡约 32 周才出一次故障。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'same',t:{en:'It is exactly the same as if nothing had crashed',zh:'和没崩时一模一样'}},
  {v:'bump',t:{en:'It jumps up for a moment, then soon rejoins the uninterrupted curve',zh:'先升高一截，很快回到没崩时的曲线上'}},
  {v:'worse',t:{en:'It jumps up and stays mostly above the uninterrupted curve, still behind at step 600',zh:'先升高一截，之后大多比没崩时高，到第 600 步也没追上'}},
  {v:'broken',t:{en:'Training breaks: the loss goes back to where it started',zh:'训练彻底坏了，loss 回到刚开始的水平'}}];
const PRED_RIGHT='worse';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const PNAME={o:{en:'Optimizer state',zh:'优化器状态'},s:{en:'Where the learning rate is',zh:'学习率走到哪一步'},d:{en:'Where the data loader is',zh:'数据读到哪里'},r:{en:'Random-number state',zh:'随机数状态'}};
const PDESC={o:{en:'In real training it is the FP32 copy of the weights plus Adam’s m and v, 12 bytes per parameter. This small model runs in FP64 throughout and has no FP32 copy, so here it is m, v and AdamW’s step count. Without it AdamW starts again from empty m and v.',zh:'真实训练里，它是 FP32 权重副本加上 Adam 的 m 和 v，每个参数 12 字节。这个小模型全程用 FP64，没有 FP32 副本，所以这里只存 m、v 和 AdamW 走过的步数。没有它，AdamW 从空的 m 和 v 重新开始。'},
  s:{en:'The scheduler’s step number. Without it a new scheduler counts from step 0: the learning rate warms up again to its peak.',zh:'学习率调度走到的步数。没有它，新建的调度器从第 0 步算起，学习率重新预热，一直升回峰值。'},
  d:{en:'Which batches of this epoch have been read. Without it, this lab’s loader starts the epoch again from its first batch.',zh:'这个 epoch 已经读过哪些 batch。没有它，这个 Lab 的数据加载器从这个 epoch 的第一个 batch 重新读。'},
  r:{en:'Where dropout’s random-number generator is. Without it the generator is seeded again as at the start of the run.',zh:'dropout 用的随机数生成器走到了哪里。没有它，生成器按训练开始时的种子重来。'}};
const partsCtl=()=>`<div class="ckp-parts" id="ckp-ic-parts" role="group" aria-label="${esc(tx('What the checkpoint holds; these stack','存档里有什么，可以叠加'))}">`+
  `<div class="ckp-part on fixed"><span class="ckp-sw" aria-hidden="true">✓</span><span><b>${tx('Weights','权重')}</b><span class="small">${tx('Always saved: without them there is nothing to resume.','一定会存：没有它就谈不上接着训。')}</span></span></div>`+
  PARTS.map(p=>`<button type="button" class="ckp-part" data-part="${p}" aria-pressed="${!!S[p]}"><span class="ckp-sw" aria-hidden="true">${S[p]?'✓':''}</span><span><b>${esc(L(PNAME[p]))}</b><span class="small">${esc(L(PDESC[p]))}</span></span></button>`).join('')+`</div>`;
const ATOM={plain:{t:{en:'Overwrite one file (initial setup)',zh:'直接覆盖同一个文件（初始设置）'},d:{en:'Every save writes over ckpt.pt, the only checkpoint. nanoGPT saves this way.',zh:'每次存档都写进同一个文件 ckpt.pt，覆盖上一份。nanoGPT 就是这样存的。'}},
  atomic:{t:{en:'Temporary name, rename, then “latest”',zh:'先写临时文件，再改名，最后更新 latest'},d:{en:'Write step-500.tmp; when it is complete, rename it step-500; only then write “500” into the record file latest.',zh:'先写到 step-500.tmp；全部写完再改名为 step-500；最后才在记录文件 latest 里写上 500。'}}};
const redone=a=>a==='atomic'?500-E.CK:500;
const atomLever=()=>`<div class="ckp-lv">`+A.ui.levers({aria:tx('How to save, pick one','怎么存，选一种'),value:'atom:'+S.atom,
  options:['plain','atomic'].map(a=>({v:'atom:'+a,title:L(ATOM[a].t),desc:L(ATOM[a].d),result:esc(atomTried.has(a)||a===S.atom?tx(`${redone(a)} steps redone`,`重算 ${redone(a)} 步`):'')}))})+`</div>`;
const FIX={latest:{t:{en:'Resume from the latest checkpoint, step 500 (initial setup)',zh:'从最新的存档（第 500 步）接着训（初始设置）'},d:{en:'The only choice if you keep one checkpoint.',zh:'只留一份存档时，只能这样。'}},
  back:{t:{en:'Roll back to step 400, read the data as before',zh:'退回第 400 步，照常读数据'},d:{en:'Needs the step-400 checkpoint kept as well.',zh:'要把第 400 步的存档也留着。'}},
  skip:{t:{en:'Roll back to step 400 and skip the batches of steps 470–474',zh:'退回第 400 步，跳过第 470–474 步的数据'},d:{en:'Those five steps make no update; the run moves on to the next batch.',zh:'这 5 步不更新参数，直接读下一个 batch。'}}};
/* an option's loss at step 600 shows only once its run has been watched to the end (R3.8) */
const fixShown=o=>(fixTried.has(o)||o===S.fix)&&watched.has('sp:'+o);
const fixRes=o=>{const r=remedy(o); return tx(`loss at step 600: ${lossF(r.final)}`,`第 600 步 loss ${lossF(r.final)}`);};
const fixLever=()=>`<div class="ckp-lv">`+A.ui.levers({aria:tx('What to do after the spike, pick one','突刺之后怎么办，选一种'),value:'fix:'+S.fix,
  options:['latest','back','skip'].map(o=>({v:'fix:'+o,title:L(FIX[o].t),desc:L(FIX[o].d),result:esc(fixShown(o)?fixRes(o):'')}))})+`</div>`;
const intCtl=(id,inline)=>A.ui.range({id,key:'T',label:tx('Save every','每隔多久存一次'),min:0,max:INTS.length-1,value:Math.max(0,INTS.indexOf(S.T)),out:minsT(S.T),inline});
const modeCtl=(id,inline)=>A.ui.seg({id,key:'mode',label:tx('How to save','怎么存'),options:[['sync',tx('Synchronous','同步')],['async',tx('Asynchronous','异步')]],value:S.mode,inline});
const MODEDEF=()=>tx(`<ul class="ckp-defs"><li><b>Synchronous save</b><span>Training stops until the checkpoint is written; the job ${DEF('writes 10 GB/s',TIP.wbw())}, so each save stops it for about ${fixd(SIZE70/WINT.bw,0)} s.</span></li><li><b>Asynchronous save</b><span>Training stops only while the state is copied into CPU memory (${WINT.dAsync} s here); the write to disk runs in the background.</span></li></ul>`,
  `<ul class="ckp-defs"><li><b>同步存档</b><span>训练停下来，等存档写完再继续，按${DEF('每秒写 10 GB',TIP.wbw())} 算，每次停约 ${fixd(SIZE70/WINT.bw,0)} 秒。</span></li><li><b>异步存档</b><span>训练只在把状态拷进 CPU 内存时停一下（这里假设 ${WINT.dAsync} 秒），写盘在后台进行。</span></li></ul>`);

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},gate:()=>!!pred,skip:true,
  lead:()=>tx('Crash at step 450; the checkpoint at 400 holds only weights.','第 450 步崩了；第 400 步的存档只有权重。'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Large training runs always get interrupted: with thousands of GPUs, some GPU fails, a link drops or a process dies. A ${DEF('checkpoint',TIP.ckpt())} is the run’s save; after a crash you resume from the latest one.</p>`+
     `<p>This lab really trains a small model in your browser: 600 steps of ${DEF('AdamW',TIP.adamw())}, a ${DEF('learning rate that warms up, then falls',TIP.sched())}, a shuffled batch of data per step, and ${DEF('dropout',TIP.dropout())}. It saves a checkpoint every 100 steps.</p>`+
     `<p>At step 450 the process crashes. The latest checkpoint is from step 400, and it holds only the model’s weights. You resume from it and train to step 600. This is a gut guess; the next parts explain it. What does the ${DEF('validation loss',TIP.val())} do?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the result.</p>':''}`,
     `<p>大规模训练一定会中断：几千张卡一起跑，总会有卡坏掉、网络断开、进程挂掉。${DEF('检查点',TIP.ckpt())}就是训练的存档，崩了以后从最近一次存档接着训。</p>`+
     `<p>这个 Lab 在你的浏览器里真的训练一个小模型：一共 600 步，优化器是 ${DEF('AdamW',TIP.adamw())}，${DEF('学习率先预热、再逐渐下降',TIP.sched())}，每一步读一批打乱的数据，还用 ${DEF('dropout',TIP.dropout())}。每 100 步存一次档。</p>`+
     `<p>训练到第 450 步，进程崩了。最近的存档在第 400 步，里面只有模型的权重。你从这份存档接着训到第 600 步。这一题凭直觉猜，后面几部分会讲清楚为什么。${DEF('验证集 loss',TIP.val())} 会怎样？</p>${ch}${pred?'<p class="muted small">已记下。第 2 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'hold',title:{en:'What a checkpoint must hold',zh:'存档里要有什么'},expect:()=>PARTS.every(p=>S[p]),showMe:()=>setParts({o:1,s:1,d:1,r:1}),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Weights alone are not enough: add the other four.','只有权重不够：把另外四样加回去。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>The chart plays the run: it crashes at step 450 and resumes from the step-400 checkpoint, which holds only the weights. The gray line is the loss without the crash; the dashed line is the loss of the resumed run.</p>`+
     `<p>A checkpoint that resumes exactly holds four more things. Add them back, several at once if you like, and watch the dashed line.</p>`,
     `<p>图里在播放这次训练。第 450 步崩了，从第 400 步的存档接着训，这份存档里只有权重。灰线是没崩时的 loss，虚线是接着训的 loss。</p>`+
     `<p>一份能原样接上的存档，除了权重还有四样。把它们加回存档，可以同时加几样，看看虚线怎么变。</p>`)+partsCtl(),
  obs:()=>obsHold()},
 {id:'atomic',title:{en:'A save is complete, or it does not count',zh:'存档要么完整，要么不算'},expect:()=>atomTried.has('atomic'),showMe:()=>setAtom('atomic'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('A crash in the middle of a save must not destroy the last good checkpoint.','存到一半崩了，也不能毁掉上一份好的存档。'),
  enter:()=>({...BASE,view:'files',o:1,s:1,d:1,r:1}),
  body:()=>tx(`<p>Saving takes time: a large model’s checkpoint is a terabyte or more, written over tens of seconds to minutes. If the machine dies halfway, the file is half written.</p>`+
     `<p>Here the crash comes while the step-500 checkpoint is being written, at 60%. Pick a way to save and see how many steps are redone after the crash. The second way relies on ${DEF('a rename being atomic',TIP.rename())}.</p>`,
     `<p>存档要花时间：大模型的一份存档有上 TB，要写几十秒到几分钟。写到一半机器坏了，文件就只写了一半。</p>`+
     `<p>这里的故障发生在第 500 步存档写到 60% 的时候。选一种存法，看看崩了以后要重算多少步。第二种存法靠的是“${DEF('改名是原子操作',TIP.rename())}”这一点。</p>`)+atomLever(),
  obs:()=>obsAtomic()},
 {id:'interval',title:{en:'How often to save',zh:'多久存一次'},expect:()=>tMoved&&asyncTried,showMe:()=>{asyncTried=true; tMoved=true; setState({mode:'async',T:nearInt(tOpt(dOf('c70','async'),CASES.c70.M)/60)});},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Too often, training keeps stopping for saves; too rarely, each failure repeats much work.','存得太勤，训练老要停下来等存档；存得太少，一次故障重算很多。'),
  enter:()=>({...BASE,view:'waste',mode:'sync',T:180,cs:'c70'}),
  body:()=>tx(`<p>Saving often stops training for every save; saving rarely means more work redone after each failure. The best interval lies in between.</p><p>Here it is found at real size: Llama${NB}3.1${NB}70B on 1,024 GPUs, a checkpoint of ${DEF('14 bytes per parameter',TIP.bpc())}, ${TB(SIZE70)} in all, a failure about every ${DEF('12 hours',TIP.mtbf())} (the time between failures, M), and 10 minutes for each ${DEF('restart',TIP.restart())} (R).</p>`,
     `<p>存得勤，训练要常停下来等存档；存得少，一次故障要重算的工作就多。最省的间隔在两者之间。</p><p>这里按真实规模来找：1,024 张卡训练 Llama${NB}3.1${NB}70B，存档每个参数 ${DEF('14 字节',TIP.bpc())}，一份 ${TB(SIZE70)}；平均每 ${DEF('12 小时',TIP.mtbf())}出一次故障（故障间隔 M），每次${DEF('重启',TIP.restart())}要 10 分钟（重启时间 R）。</p>`)+MODEDEF()+tx('<p>Drag the interval to find the one that wastes least, then switch to asynchronous saving.</p>','<p>拖动间隔，找到浪费最少的那一档，再换成异步存档看看。</p>')+intCtl('ckp-ic-T',true)+modeCtl('ckp-ic-mode',true),
  obs:()=>obsInterval()},
 {id:'spike',title:{en:'Loss spikes: roll back and skip',zh:'loss 突刺：退回去，跳过坏数据'},expect:()=>fixTried.has('skip'),showMe:()=>setFix('skip'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Bad data, not a broken machine: keep older checkpoints to roll back to.','坏的是数据，不是机器：要留着更早的存档才能退回去。'),
  enter:()=>({...BASE,view:'spike',o:1,s:1,d:1,r:1,fix:'latest'}),
  body:()=>tx(`<p>Sometimes nothing crashes, and yet the loss shoots up and stays high for a long time. A common cause is a stretch of corrupted data.</p>`+
     `<p>Here the batches of steps 470 to 474 are corrupted: this lab scales their inputs by 8 and gives them wrong labels. The checkpoint at step 500 is saved as usual, and you notice the spike at step 520. Pick what to do and see the loss at step 600.</p>`,
     `<p>有时候机器没坏，loss 却突然冲高，很久都下不来。常见的原因是一段坏数据。</p>`+
     `<p>这里第 470 到 474 步读到的 batch 是坏的：这个 Lab 把它们的输入放大 8 倍，标签也全标错。第 500 步照常存了档，你在第 520 步才发现突刺。选一种办法，看看第 600 步的 loss 是多少。</p>`)+fixLever(),
  obs:()=>obsSpike()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Save everything, save atomically, save at the right interval, keep more than one.','存全、存完整、存得不勤不懒、多留几份。'),
  enter:()=>remState(remView),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const nearInt=m=>INTS.reduce((a,b)=>Math.abs(b-m)<Math.abs(a-m)?b:a,INTS[0]);

function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
function obsHold(){
  const w0=resumed({}), k=keepOf(S), r=resumed(k), n=PARTS.filter(p=>S[p]).length;
  if(!watched.has('ls:0000')) return '';
  let s='';
  if(n===0){s=tx(`<p><b>With the weights only, the resumed run jumps up first and then stays above the gray line almost all the way: at step 600 its loss is ${lossF(w0.final)}, against ${lossF(w0.final0)} without the crash.</b></p>`,
               `<p><b>只用权重接着训，虚线先往上跳，之后几乎一直在灰线上面。到第 600 步，loss 是 ${lossF(w0.final)}，没崩时是 ${lossF(w0.final0)}。</b></p>`)+verdict()+
     tx(`<p>The learning rate is what pushes it up most. The new scheduler starts its warmup again; about ten steps later the learning rate passes the uninterrupted run’s and keeps climbing to the peak. AdamW’s m and v are empty as well, so every parameter takes a full step and the loss jumps. At step 600 the learning rate is still ${fixd(LRX,0)} times the uninterrupted run’s.</p>`,
        `<p>把它推高的主要是学习率。新的调度器从预热重新开始，十来步后学习率就超过了没崩时的值，一直升到峰值。AdamW 的 m 和 v 又是空的，每个参数的更新量都接近学习率本身，loss 就跳了上去。到第 600 步，学习率还是没崩时的 ${fixd(LRX,0)} 倍。</p>`);
   return s;}
  if(r.same) return tx(`<p><b>All four added back: the dashed line lies exactly on the gray one.</b></p><p>Every step after the resume reads the same batch, switches off the same units and makes the same update as the run that never stopped. The loss agrees to the last digit.</p>`,
                       `<p><b>四样都加回去，虚线和灰线完全重合。</b></p><p>接着训的每一步，读的 batch、关掉的单元、做的更新，都和没崩时一模一样，loss 连最后一位小数都相同。</p>`);
  const miss=PARTS.filter(p=>!S[p]).map(p=>L(PNAME[p])).join(tx(', ','、'));
  s=tx(`<p><b>Still missing: ${esc(miss)}. ${r.maxRel>0.05?`The resumed loss rises up to ${fixd(r.maxRel*100,0)}% above the uninterrupted run’s.`:'The curves no longer coincide, though the gap is small.'}</b></p>`,
       `<p><b>还缺：${esc(miss)}。${r.maxRel>0.05?`接着训的 loss 最多比没崩时高 ${fixd(r.maxRel*100,0)}%。`:'两条线不再重合，只是差得不多。'}</b></p>`);
  if(!S.s) s+=tx('<p>The learning rate warms up again to its peak, so every step is large. The dashed line swings up and down, so how large the largest gap is comes down to chance, and at step 600 the loss is still above the gray line. Only once “where the learning rate is” is back does the gap clearly shrink.</p>',
                 '<p>学习率重新预热，又升回了峰值，每一步都迈得很大。所以虚线忽高忽低，最大差距是多少要看运气，第 600 步的 loss 也还在灰线上面。把“学习率走到哪一步”加回去，差距才会明显变小。</p>');
  else if(!S.o) s+=tx('<p>AdamW starts from empty m and v: its first few updates are far too large, the loss jumps, then comes back once m and v have filled up again.</p>','<p>AdamW 从空的 m 和 v 开始，头几步的更新太大，loss 先往上跳。等 m 和 v 重新攒起来，才回到原来的水平。</p>');
  else if(!S.d) s+=tx(`<p>The data is wrong as well: ${fint(r.twice)} samples were read twice in epoch ${EP_CK}, and ${fint(r.never)} samples of the last epoch were never read.</p>`,`<p>数据也不对了。第 ${EP_CK} 个 epoch 里有 ${fint(r.twice)} 个样本读了两遍，最后一个 epoch 里有 ${fint(r.never)} 个样本一遍都没读到。</p>`);
  else s+=tx('<p>Dropout now switches off different units than the uninterrupted run did. The run is healthy, but it cannot be reproduced: run it again from step 0 and you get the gray line, not this one.</p>','<p>dropout 关掉的单元和没崩时不一样了。训练本身没问题，只是没法复现：从第 0 步重跑一遍，得到的是灰线，不是这条虚线。</p>');
  return s;
}
function obsAtomic(){
  if(S.atom==='plain') return tx(`<p><b>The only checkpoint is broken. As soon as the overwrite began, step 400’s contents in ckpt.pt were gone, and step 500 got only 60% written. The run starts again from step 0: ${redone('plain')} steps redone.</b></p><p>Now pick the second way.</p>`,
     `<p><b>唯一的存档坏了。覆盖一开始，ckpt.pt 里第 400 步的内容就没了，第 500 步又只写进了 60%。只能从第 0 步重训，重算 ${redone('plain')} 步。</b></p><p>再选第二种存法试试。</p>`);
  return tx(`<p><b>step-500.tmp was never finished, so it does not count. latest still says 400, so the run resumes from step 400 and redoes only ${redone('atomic')} steps instead of ${redone('plain')}.</b></p>`+
    `<p>Real frameworks follow the same rule: write first, mark complete last.</p><ul class="ckp-defs ckp-al">`+
    `<li><b>Megatron-LM</b><span>writes <code>latest_<wbr>checkpointed_<wbr>iteration.txt</code> only after every process has finished writing.</span></li>`+
    `<li><b>PyTorch DCP</b><span>writes <code>.metadata<wbr>.tmp</code>, syncs it to disk, then renames it.</span></li>`+
    `<li><b>llm.c</b><span>writes a DONE file last.</span></li></ul><p>Keep the previous checkpoint until the new one is complete.</p>`,
    `<p><b>step-500.tmp 没写完，不算数。latest 里记的还是 400，所以重启后从第 400 步接着训，只重算 ${redone('atomic')} 步，不用重算 ${redone('plain')} 步。</b></p>`+
    `<p>真实的框架都守同一条规矩：先写完，最后再标记“完成”。</p><ul class="ckp-defs ckp-al">`+
    `<li><b>Megatron-LM</b><span>要等所有进程都写完，才写 <code>latest_<wbr>checkpointed_<wbr>iteration.txt</code>。</span></li>`+
    `<li><b>PyTorch 分布式存档</b><span>先写 <code>.metadata<wbr>.tmp</code>，刷到磁盘后再改名。</span></li>`+
    `<li><b>llm.c</b><span>最后写一个 DONE 文件。</span></li></ul><p>新存档写完之前，上一份一定要留着。</p>`);
}
/* the three terms of the wasted share, each with its meaning (owner b8: “公式三项的意义的对应”); the swatches are the picture's */
const TERMS=[{k:'save',tex:'\\frac{\\delta}{T}',inl:'\\delta / T',sw:'ckp-sw-save'},{k:'lost',tex:'\\frac{T}{2M}',inl:'T / 2M',sw:'ckp-sw-lost'},{k:'restart',tex:'\\frac{R}{M}',inl:'R / M',sw:'ckp-sw-rst'}];
const TNAME={save:{en:'Saving',zh:'存档'},lost:{en:'Redone work',zh:'重算'},restart:{en:'Restarts',zh:'重启'}};
const TMEAN={save:{en:'A save every T stops training for δ each time.',zh:'每隔 T 存一次档，每次训练停 δ。'},
  lost:{en:'A failure comes every M on average and loses the work since the last save, on average half an interval.',zh:'平均每隔 M 出一次故障，丢掉上次存档以后的工作，平均是半个间隔。'},
  restart:{en:'Each failure also costs R to restart.',zh:'每次故障还要花 R 重启。'}};
/* the interval picture's numbers now: the 70B or the 8B of Guided and Scenarios, or the 405B plan of the challenge */
function wasteNow(){
  if(mode==='challenge'){const d=chS.mode==='async'?CH.dAsync:CH.size/CH.bw; return {d,T:chS.T,M:CH.M,R:CH.R,md:chS.mode,ymax:0.4,fine:false,w:waste(d,chS.T*60,CH.M,CH.R)};}
  const c=CASES[S.cs], d=dOf(S.cs,S.mode); return {d,T:S.T,M:c.M,R:c.R,md:S.mode,ymax:S.cs==='c8'?0.05:0.4,fine:S.cs==='c8',w:waste(d,S.T*60,c.M,c.R)};}
const termPct=(v,fine)=>fine?(v<0.00005?tx('under 0.01%','不到 0.01%'):pct(v,2)):(v<0.0005?tx('under 0.1%','不到 0.1%'):pct(v));
const mSay=M=>M>=7*24*3600?tx(`${Math.round(M/604800)} weeks`,`${Math.round(M/604800)} 周`):tx(`${fnum(M/3600)} hours`,`${fnum(M/3600)} 小时`);
/* a term's hover: what it is, then its share with the denominator and the units (R2.11); the share as shown, never more digits */
function termTip(k,q){const Ts=q.T*60, sh=v=>{const t=termPct(v,q.fine); return /^[0-9]/.test(t)?` = ${t}`:tx(`, ${t}`,`，${t}`);};
  if(k==='save') return tx(`Saving: every ${minsT(q.T)} training stops for ${fnum(q.d,1)} s. Share of the run = ${fnum(q.d,1)} s ÷ ${fint(Ts)} s${sh(q.w.save)}.`,
                           `存档：每 ${minsT(q.T)}存一次，每次训练停 ${fnum(q.d,1)} 秒。占整次训练的比例 = ${fnum(q.d,1)} 秒 ÷ ${fint(Ts)} 秒${sh(q.w.save)}。`);
  if(k==='lost') return tx(`Redone work: a failure comes every ${mSay(q.M)} on average and loses on average half an interval, ${mins(Ts/2)}. Share of the run = ${fint(Ts/2)} s ÷ ${fint(q.M)} s${sh(q.w.lost)}.`,
                           `重算：平均每 ${mSay(q.M)}出一次故障，每次平均丢掉半个间隔，也就是 ${mins(Ts/2)}。占整次训练的比例 = ${fint(Ts/2)} 秒 ÷ ${fint(q.M)} 秒${sh(q.w.lost)}。`);
  return tx(`Restarts: each failure costs ${fnum(q.R/60)} minutes to restart. Share of the run = ${fint(q.R)} s ÷ ${fint(q.M)} s${sh(q.w.restart)}.`,
            `重启：每次故障要花 ${fnum(q.R/60)} 分钟重启。占整次训练的比例 = ${fint(q.R)} 秒 ÷ ${fint(q.M)} 秒${sh(q.w.restart)}。`);}
/* the rail's table: each term, what it means, its value now */
const termsTable=q=>`<table class="ckp-tt"><tbody>`+TERMS.map(t=>`<tr data-term="${t.k}"><td class="ckp-tt-k"><i class="sw ${t.sw}" aria-hidden="true"></i>${A.ui.math(t.tex)}</td><td><b>${esc(L(TNAME[t.k]))}</b>${esc(L(TMEAN[t.k]))}</td><td class="mono ckp-tt-v">${esc(termPct(q.w[t.k],q.fine))}</td></tr>`).join('')+`</tbody></table>`;
const WASTE_TEX=()=>tx('\\text{wasted} \\approx \\frac{\\delta}{T} + \\frac{T}{2M} + \\frac{R}{M}','\\text{浪费的比例} \\approx \\frac{\\delta}{T} + \\frac{T}{2M} + \\frac{R}{M}');
function obsInterval(){
  const c=CASES.c70, d=dOf('c70',S.mode), w=waste(d,S.T*60,c.M,c.R), to=tOpt(d,c.M), wo=waste(d,to,c.M,c.R), other=S.mode==='sync'?'async':'sync', d2=dOf('c70',other), to2=tOpt(d2,c.M);
  /* when the chosen interval is already within 10% of the best one, say so instead of repeating the same interval and share */
  const atBest=Math.abs(S.T*60/to-1)<0.1;
  return tx(`<p><b>${S.mode==='sync'?'Synchronous':'Asynchronous'}, every ${minsT(S.T)}: ${pct(w.total)} of the run is wasted. ${atBest?'This is about the interval that wastes least.':`Saving about every ${mins(to)} wastes least, ${pct(wo.total)}.`}</b></p>`,
            `<p><b>${S.mode==='sync'?'同步':'异步'}存档、每 ${minsT(S.T)}存一次时，整次训练有 ${pct(w.total)} 的时间浪费掉了。${atBest?'这差不多就是最省的间隔。':`大约每 ${mins(to)}存一次最省，浪费 ${pct(wo.total)}。`}</b></p>`)+
    MX(WASTE_TEX())+termsTable(wasteNow())+
    tx('<p>The longer the interval, the less saving and the more redone work; restarts do not change. The waste is least where the first two terms are equal, at the interval</p>',
       '<p>间隔越长，存档越少、重算越多，重启不变。前两项相等时浪费最少，这时的间隔是：</p>')+
    MX('T^{*} = \\sqrt{2 \\delta M}')+
    tx(`<p>Asynchronous saving cuts δ, so it pays to save more often: about every ${mins(S.mode==='sync'?to2:to)} instead of every ${mins(S.mode==='sync'?to:to2)}.</p>`,
       `<p>异步存档把 δ 压小了，所以值得存得更勤，大约每 ${mins(S.mode==='sync'?to2:to)}一次，而不是每 ${mins(S.mode==='sync'?to:to2)}一次。</p>`);
}
function obsSpike(){
  const r=remedy(S.fix), sp=remedy('latest'), b=base();
  if(S.fix==='latest') return tx(`<p><b>The step-500 checkpoint already holds the damage: the loss is ${lossF(r.final)} at step 600, against ${lossF(b.ev[E.STEPS])} in a run without bad data.</b></p><p>AdamW’s v took in the huge gradients of the bad batches; every later update is divided by ${A.ui.math('\\sqrt{v}')}, so training crawls. If you keep only the latest checkpoint, this is all you have.</p>`,
     `<p><b>第 500 步的存档已经带着损伤：到第 600 步 loss 是 ${lossF(r.final)}，没有坏数据的训练是 ${lossF(b.ev[E.STEPS])}。</b></p><p>坏数据的梯度特别大，被记进了 AdamW 的 v；之后每次更新都要除以 ${A.ui.math('\\sqrt{v}')}，训练就走得很慢。只留最新一份存档的话，手里只有它。</p>`);
  if(S.fix==='back') return tx(`<p><b>Rolling back alone changes nothing: the same bad batches come again at steps 470–474, and the spike repeats exactly.</b></p><p>The data order is part of the checkpoint, so a resumed run reads the same batches. Rolling back helps only together with skipping them.</p>`,
     `<p><b>只退回去没用：第 470–474 步又读到同样的坏数据，突刺原样重演。</b></p><p>数据顺序也存在存档里，所以接着训会读到同样的 batch。退回去，还得跳过它们才行。</p>`);
  return tx(`<p><b>Rolled back to step 400 and skipped the five bad batches: no spike, and the loss at step 600 is ${lossF(r.final)}, against ${lossF(sp.final)} when resuming from step 500.</b></p><p>This needs the step-400 checkpoint, so keep more than the latest one. What real runs do:</p>`+
    `<ul class="ckp-defs ckp-al"><li><b>PaLM</b><span>restarts from an earlier checkpoint and skips the suspect batches.</span></li><li><b>Megatron-LM</b><span>lists the iterations to skip in <code>iterations_to_skip</code>.</span></li><li><b>OPT-175B</b><span>after rolling back, also lowers the learning rate.</span></li></ul>`,
     `<p><b>退回第 400 步、跳过 5 个坏 batch 后，没有突刺，第 600 步的 loss 是 ${lossF(r.final)}。要是从第 500 步接着训，loss 是 ${lossF(sp.final)}。</b></p><p>这要求第 400 步的存档还在，所以存档不能只留最新一份。实际训练中的做法：</p>`+
    `<ul class="ckp-defs ckp-al"><li><b>PaLM</b><span>从更早的存档重启，跳过可疑的 batch。</span></li><li><b>Megatron-LM</b><span>用 <code>iterations_to_skip</code> 指定要跳过的步。</span></li><li><b>OPT-175B</b><span>退回去以后，还把学习率调低。</span></li></ul>`);
}
const REMS=['weights','atomic','interval'];
function remState(k){if(k==='atomic') return {...BASE,view:'files',atom:'atomic',o:1,s:1,d:1,r:1}; if(k==='interval') return {...BASE,view:'waste',mode:'async',T:15}; return {...BASE};}
function remBtn(k){const on=remView===k;
  const tip={weights:tx('Shows the run resumed with the weights only.','在图里显示只用权重接着训的结果。'),atomic:tx('Shows the atomic save after the crash.','在图里显示原子存档在故障之后的样子。'),interval:tx('Shows the 70B’s wasted time against the save interval.','在图里显示 70B 浪费的时间随存档间隔怎么变。')}[k];
  return ` <button type="button" class="linkbtn ckp-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const w0=resumed({}), c=CASES.c70, ds=dOf('c70','sync'), da=dOf('c70','async');
  return tx(`<ul class="ckp-remlist"><li><b>Save the whole training state, not just the weights.</b> With the weights only, AdamW starts again from empty m and v and the learning rate warms up again, so the resumed run ends worse. With the optimizer state, the scheduler step, the data position and the RNG state as well, the resumed run matches step for step.${remBtn('weights')}</li>`+
    `<li><b>Write the new checkpoint completely before you mark it as the latest.</b> Overwrite the only checkpoint, and a crash halfway through the save leaves nothing usable. Write a temporary file, rename it, and update “latest” last: a complete checkpoint is then always on disk.${remBtn('atomic')}</li>`+
    `<li><b>Pick the interval from the save time and the failure rate.</b> The longer each save stops training, the less often to save. The more often failures come, the more often to save. An asynchronous save stops training only briefly, so it can save far more often.${remBtn('interval')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Saving the weights is enough.” The optimizer state, the learning-rate step, the data position and the RNG state decide what the next steps do; without them the resumed run is a different run, and often a worse one.</p><p>✗ “The more often you save, the safer.” Every synchronous save stops training. Save too often, and the stops add up to more than the failures cost.</p><p>✗ “After a crash, train again from the start.” Never: resume from the latest complete checkpoint, and keep an older one in case the latest holds a problem.</p></div>${A.shell.toScenarios()}`,
    `<ul class="ckp-remlist"><li><b>存整个训练状态，不只是权重。</b>只存权重时，AdamW 要从空的 m 和 v 重新开始，学习率也会重新预热，接着训的 loss 就变差了。把优化器状态、学习率走到的步数、数据位置和随机数状态也存上，接着训的每一步都和没崩时一样。${remBtn('weights')}</li>`+
    `<li><b>新存档完整写好，再把它标成“最新”。</b>直接覆盖唯一的存档，存到一半崩了，就一份能用的存档都不剩。先写临时文件，再改名，最后才更新 latest，磁盘上就总有一份完整的存档。${remBtn('atomic')}</li>`+
    `<li><b>按存档要停多久、多久出一次故障来定间隔。</b>每次存档停得越久，就该存得越少。故障来得越勤，就该存得越勤。异步存档只停很短的时间，所以可以存得勤得多。${remBtn('interval')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “存权重就够了。”优化器状态、学习率走到的步数、数据位置和随机数状态，决定了接下来每一步怎么走；少了它们，接着训就成了另一次训练，往往还更差。</p><p>✗ “存得越勤越安全。”每次同步存档都要停下训练。存得太勤，停下来的时间加起来，比故障造成的浪费还多。</p><p>✗ “崩了就从头再训。”不要：从最近一份完整的存档接着训，并且多留一份更早的，以防最新那份本身有问题。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 meg:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/checkpointing.py',
 megcfg:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/config/training_config.py',
 ds:'https://github.com/microsoft/DeepSpeed/blob/a77aeb676507beb9fe0604bc31c2bf075daf7c46/deepspeed/runtime/engine.py',
 dcp:'https://github.com/pytorch/pytorch/blob/e417f6dfacaf7f00dce343adc0d48d8fcc8fb2e2/torch/distributed/checkpoint/filesystem.py',
 tt:'https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/torchtitan/components/checkpointer/base.py',
 ttdoc:'https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/docs/checkpoint.md',
 async:'https://github.com/pytorch/tutorials/blob/11512db7cbbcc4b530ecfc63205d3fed13fa190d/recipes_source/distributed_async_checkpoint_recipe.rst',
 llmc:'https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/train_gpt2.cu',
 nano:'https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py',
 opt:'https://github.com/facebookresearch/metaseq/blob/f7ffa5fd61cf90f498a36d365c13dd7f1a912ff7/projects/OPT/chronicles/56_percent_update.md',
 opt27:'https://github.com/facebookresearch/metaseq/blob/f7ffa5fd61cf90f498a36d365c13dd7f1a912ff7/projects/OPT/chronicles/27_percent_update.md',
 bloom:'https://github.com/bigscience-workshop/bigscience/blob/9e64edec2ad1fb9263bf39ba616a84ff64b4d947/train/tr11-176B-ml/README.md',
 bloomc:'https://github.com/bigscience-workshop/bigscience/blob/9e64edec2ad1fb9263bf39ba616a84ff64b4d947/train/tr11-176B-ml/chronicles.md',
 mle:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/training/fault-tolerance/README.md',
 mlei:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/training/instabilities/README.md'
};
const PRESETS=[
 {id:'big',t:{en:`Llama${NB}3.1${NB}70B on 1,024 GPUs`,zh:`1,024 张卡训练 Llama${NB}3.1${NB}70B`},d:{en:'Failures every half day: asynchronous saving pays',zh:'每半天就出一次故障：异步存档划算'},s:{...BASE,view:'waste',cs:'c70',mode:'async',T:15}},
 {id:'small',t:{en:`Fine-tuning Llama${NB}3.1${NB}8B on one 8-GPU machine`,zh:`一台 8 卡机器微调 Llama${NB}3.1${NB}8B`},d:{en:'Failures are rare: saving every few hours is fine',zh:'很少出故障：隔几小时存一次就行'},s:{...BASE,view:'waste',cs:'c8',mode:'sync',T:360}},
 {id:'data',t:{en:'The checkpoint lacks the data position',zh:'存档没存数据读到哪里'},d:{en:'The other four parts are saved; on resuming, the epoch starts over',zh:'其余四样都存了，接着训时这个 epoch 从头读'},s:{...BASE,view:'loss',o:1,s:1,d:0,r:1}},
 {id:'spike',t:{en:'A stretch of bad data makes the loss spike',zh:'一段坏数据让 loss 冲高'},d:{en:'Two checkpoints kept: roll back past the spike',zh:'留了两份存档：退到突刺之前'},s:{...BASE,view:'spike',o:1,s:1,d:1,r:1,fix:'skip'}}];
const KEYS=['view','o','s','d','r','mode','T','cs','fix'];
const presetNow=()=>PRESETS.findIndex(p=>p.s.view===S.view&&KEYS.every(k=>{if(p.s.view==='waste'&&['o','s','d','r','fix'].includes(k)) return true; if(p.s.view!=='waste'&&['mode','T','cs'].includes(k)) return true; if(p.s.view!=='spike'&&k==='fix') return true; return p.s[k]===S[k];}));
function scenHTML(i){
  const p=PRESETS[i];
  if(p.id==='big'||p.id==='small'){const c=CASES[S.cs], d=dOf(S.cs,S.mode), w=waste(d,S.T*60,c.M,c.R), to=tOpt(d,c.M);
    if(p.id==='big'){const sh=[w.save,w.lost,w.restart].map(x=>x/w.total), even=sh.every(x=>x>0.25&&x<0.45), rmax=w.restart>=w.save&&w.restart>=w.lost;
      return tx(`<p><b>Asynchronous saves every ${minsT(S.T)}: ${pct(w.total)} of the run is wasted${even?`, split about evenly between saves, redone work and restarts${rmax?', restarts a little more':''}`:''}.</b></p><p>Each save stops training for only ${d} s, so saving every quarter of an hour costs little, and a failure loses only a few minutes of work. Written synchronously, each save of ${TB(c.size)} would stop training for ${fixd(dOf('c70','sync'),0)} s.</p>`,
      `<p><b>异步存档、每 ${minsT(S.T)}一次时，整次训练有 ${pct(w.total)} 的时间白费。${even?`存档、重算和重启大约各占三分之一${rmax?'，重启稍多一点':''}。`:''}</b></p><p>每次存档只让训练停 ${d} 秒，所以每刻钟存一次也不贵。一次故障也只丢几分钟的工作。如果改成同步写，每存一次 ${TB(c.size)}，训练都要停 ${fixd(dOf('c70','sync'),0)} 秒。</p>`);}
    return tx(`<p><b>One machine fails rarely: synchronous saves every ${minsT(S.T)} waste only ${pct(w.total,2)} of the run.</b></p><p>The 8B’s checkpoint is only about ${Math.round(c.size/1e10)*10} GB, written in about ten seconds. One machine ${DEF('fails only every few months',TIP.m8())}, so the best interval is as long as about ${fixd(to/3600,0)} hours: saving rarely is fine.</p>`,
      `<p><b>一台机器很少出故障：同步存档每 ${minsT(S.T)}一次，只浪费整次训练的 ${pct(w.total,2)}。</b></p><p>8B 的存档只有约 ${Math.round(c.size/1e10)*10} GB，十来秒就写完。一台机器${DEF('要几个月才坏一次',TIP.m8())}，所以最好的间隔长达约 ${fixd(to/3600,0)} 小时，存得少一点完全没问题。</p>`);}
  if(p.id==='data'){const r=resumed(keepOf(S)), sh=E.CK-(EP_CK-1)*EPOCH, near=Math.abs(r.final/r.final0-1)<0.15;
    return tx(`<p><b>This checkpoint holds the weights, the optimizer state, the learning-rate step and the RNG state, but not where the data loader is. So on resuming, the loader starts epoch ${EP_CK} again from its first batch.</b></p>`+
      `<p>The first ${sh} steps of epoch ${EP_CK} are read again: ${fint(r.twice)} samples were read twice in epoch ${EP_CK}. From then on the run is ${sh} steps behind in the data, so when it stops at step 600, ${fint(r.never)} samples of the last epoch were never read.</p>`+
      `<p>${near?'The loss at step 600 is about the same as without the crash, so the loss curve hides the problem. ':''}Save the loader’s position, or derive it from the step as Megatron-LM does from <code>consumed_train_samples</code>.</p>`,
      `<p><b>这份存档存了权重、优化器状态、学习率走到的步数和随机数状态，唯独没存数据读到哪里。所以接着训时，数据加载器把第 ${EP_CK} 个 epoch 从第一个 batch 重新读起。</b></p>`+
      `<p>第 ${EP_CK} 个 epoch 的前 ${sh} 步又读了一遍，有 ${fint(r.twice)} 个样本读了两遍。之后的训练在数据上一直落后 ${sh} 步，到第 600 步停下时，最后一个 epoch 还有 ${fint(r.never)} 个样本没读到。</p>`+
      `<p>${near?'第 600 步的 loss 和没崩时差不多，光看 loss 曲线发现不了这个问题。':''}要么把加载器的位置存下来，要么像 Megatron-LM 那样用 <code>consumed_train_samples</code> 从步数推算出来。</p>`);}
  const r=remedy('skip'), l=remedy('latest');
  return tx(`<p><b>Rolled back two checkpoints’ worth, to step 400, skipping the bad batches: the loss at step 600 is ${lossF(r.final)}, against ${lossF(l.final)} from the latest checkpoint.</b></p><p>BLOOM’s team kept the last 15–20 checkpoints for exactly this; torchtitan keeps the latest 10 by default.</p>`,
    `<p><b>退回第 400 步、跳过坏数据：第 600 步的 loss 是 ${lossF(r.final)}；从最新的存档接着训是 ${lossF(l.final)}。</b></p><p>BLOOM 团队正是为此保留最近 15–20 份存档；torchtitan 默认保留最近 10 份。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=e=>tx(`${e.mode==='sync'?'synchronous':'asynchronous'}, every ${minsT(e.T)}, keep ${e.k}`,`${e.mode==='sync'?'同步':'异步'}，每 ${minsT(e.T)}一次，留 ${e.k} 份`);
const chSame=(e,s)=>e.mode===s.mode&&e.T===s.T&&e.k===s.k;
/* the challenge's result, as the train-capstone lab's (owner b8, R1): the verdict first; then your plan next to the plan that wastes least,
   the settings they share written once and the ones that differ marked, what fails in red; one sentence on the main gap and one on the
   single change that helps most; the passing plans only, behind a closed disclosure. Waste stays at one decimal: it is compared with
   the 12% goal. */
const CHF=[{k:'mode',chip:e=>e.mode==='sync'?tx('synchronous','同步'):tx('asynchronous','异步'),same:e=>e.mode==='sync'?tx('save synchronously','同步存档'):tx('save asynchronously','异步存档')},
  {k:'T',chip:e=>tx(`every ${minsT(e.T)}`,`每 ${minsT(e.T)}一次`),same:e=>tx(`save every ${minsT(e.T)}`,`每 ${minsT(e.T)}存一次`)},
  {k:'k',chip:e=>tx(`keep ${e.k}`,`留 ${e.k} 份`),same:e=>tx(`keep ${e.k}`,`留 ${e.k} 份`)}];
const chChange=(f,v)=>f==='mode'?(v==='async'?tx('saving asynchronously','改成异步存档'):tx('saving synchronously','改成同步存档')):f==='T'?tx(`saving every ${minsT(v)}`,`把间隔改成 ${minsT(v)}`):tx(`keeping ${v}`,`改成留 ${v} 份`);
const PARTN={save:{en:'saving',zh:'存档'},lost:{en:'redone work',zh:'重算'}};
function chPlanRow(e,role,best,diff){
  const lab={me:tx('Your plan','你的方案'),best:tx('Least waste','最省的方案'),both:tx('Your plan, the least waste','你的方案，就是最省的')}[role];
  const fs=diff.length?diff:CHF;
  const chips=fs.map(f=>`<span class="ckp-chip${role==='me'&&e[f.k]!==best[f.k]?' on':''}">${esc(f.chip(e))}</span>`).join('');
  const m=[[!e.fits,tx(`storage ${TB(e.store)}`,`存储 ${TB(e.store)}`),tx(`${e.k} × ${TB(CH.size)} = ${TB(e.store)}; the limit is 30 TB.`,`${e.k} × ${TB(CH.size)} = ${TB(e.store)}，上限 30 TB。`)],
    [!e.rb,tx(`reaches back ${minsT(e.back)}`,`能退回 ${minsT(e.back)}`),tx(`Right after a save the oldest of ${e.k} is (${e.k} − 1) × ${minsT(e.T)} = ${minsT(e.back)} old; at least 1 hour is needed.`,`刚存完时，${e.k} 份里最老的一份是 (${e.k} − 1) × ${minsT(e.T)} = ${minsT(e.back)}之前的，至少要 1 小时。`)],
    [e.w>CH.goal+1e-12,tx(`wasted ${pct(e.w)}`,`浪费 ${pct(e.w)}`),tx(`Saving ${pct(e.parts.save)} + redone work ${pct(e.parts.lost)} + restarts ${pct(e.parts.restart)}; the goal is at most 12%.`,`存档 ${pct(e.parts.save)} + 重算 ${pct(e.parts.lost)} + 重启 ${pct(e.parts.restart)}，目标是不超过 12%。`)]];
  return `<li class="ckp-plan ${role==='me'?(e.ok?'':'bad'):'best'}" data-role="${role}"><div class="ckp-plan-h"><span class="ckp-plan-t">${lab}</span>${chips}</div>`+
    `<div class="ckp-plan-m">${m.map(([bad,t,tip])=>`<span class="${bad?'bad':''}" tabindex="0" data-tipx="${esc(tip)}">${esc(t)}</span>`).join('')}</div></li>`;
}
function chSummaryHTML(me,best){
  const isBest=chSame(me,best), diff=CHF.filter(f=>me[f.k]!==best[f.k]), same=CHF.filter(f=>!diff.includes(f));
  let h=`<div class="ckp-sum"><div class="ckp-sum-cap">${isBest?tx('Your plan','你的方案'):tx('Your plan and the plan that wastes least','你的方案和最省的方案')}</div>`;
  if(!isBest) h+=`<p class="ckp-sum-same">${same.length?tx(`Both plans ${same.map(f=>f.same(best)).join(' and ')}.`,`两种方案都${same.map(f=>f.same(best)).join('、')}。`):tx('The two plans differ in all three settings.','两种方案的三项设置都不一样。')}</p>`;
  h+=`<ul class="ckp-plans">${isBest?chPlanRow(me,'both',best,[]):chPlanRow(me,'me',best,diff)+chPlanRow(best,'best',best,diff)}</ul>`;
  if(!isBest){
    /* the main gap: the term that grows most against the best plan */
    const ext=me.w-best.w, k=['save','lost'].reduce((a,b)=>(me.parts[b]-best.parts[b])>(me.parts[a]-best.parts[a])?b:a);
    if(ext>0.0005){const why=k==='save'?(me.mode==='sync'?tx(`every synchronous save stops training for about ${fixd(CH.size/CH.bw/60,0)} minutes`,`每次同步存档，训练要停约 ${fixd(CH.size/CH.bw/60,0)} 分钟`):tx(`saving every ${minsT(me.T)} adds up`,`每 ${minsT(me.T)}存一次，次数太多`))
        :tx(`saving every ${minsT(me.T)}, a failure loses ${mins(me.T*30)} of work on average`,`每 ${minsT(me.T)}存一次，一次故障平均要重算 ${mins(me.T*30)}`);
      /* the gap between the two shares as shown, so that the reader’s own subtraction agrees */
      const gp=fixd(parseFloat(pct(me.w))-parseFloat(pct(best.w)),1);
      h+=`<p class="ckp-gap">${tx(`Your plan wastes ${gp} points more, most of it on ${L(PARTN[k])}: ${why}.`,`你的方案多白费 ${gp} 个百分点，大部分在${L(PARTN[k])}上：${why}。`)}</p>`;}
    /* the single change that helps most */
    /* for a plan that meets the goal, only a change whose waste, as shown (one decimal), is lower than the plan's own (rule G7) */
    const oc=chOne(me).filter(x=>!me.ok||parseFloat(pct(x.e.w))<parseFloat(pct(me.w)))[0];
    const dl=diff.map(f=>chChange(f.k,best[f.k])), dn=tx(dl.length>1?dl.slice(0,-1).join(', ')+' and '+dl[dl.length-1]:(dl[0]||''),dl.join('、'));
    h+=`<p class="ckp-next">${oc?(me.ok?tx(`The best single change is ${chChange(oc.f,oc.v)}: ${pct(oc.e.w)} wasted.`,`只改一项的话，${chChange(oc.f,oc.v)}最省，浪费 ${pct(oc.e.w)}。`)
        :tx(`One change meets the goal: ${chChange(oc.f,oc.v)}, ${pct(oc.e.w)} wasted.`,`只改一项就能达标：${chChange(oc.f,oc.v)}，浪费 ${pct(oc.e.w)}。`))
      :(me.ok?tx(`No single change wastes less. The least-waste plan changes ${diff.length} settings at once: ${dn}.`,`只改一项省不下来。最省的方案同时改了 ${diff.length} 项：${dn}。`)
        :tx(`No single change meets the goal. The least-waste plan changes ${diff.length} settings at once: ${dn}.`,`只改一项还达不到目标。最省的方案同时改了 ${diff.length} 项：${dn}。`))}</p>`;}
  const kfit=Math.floor(CH.store/CH.size), tmin=Math.ceil(CH.back/(kfit-1));
  h+=`<p class="ckp-why small">${tx(`Why that plan: 30 TB holds only ${kfit} checkpoints, and for ${kfit} to reach back 1 hour the interval cannot be shorter than ${tmin} minutes. An asynchronous save stops training for only ${CH.dAsync} s, so the shortest interval allowed wastes least. On the chart, the lowest point of the curve (by waste alone: ${mins(tOpt(CH.dAsync,CH.M))}) lies in the pale range, where those limits are broken; the green dot marks ${minsT(best.T)}, the least within the limits.`,
    `为什么是它：30 TB 只放得下 ${kfit} 份存档，${kfit} 份要能退回 1 小时，间隔就不能短于 ${tmin} 分钟。异步存档每次只停 ${CH.dAsync} 秒，所以允许的最短间隔最省。图上曲线的最低点（只看时间浪费：${mins(tOpt(CH.dAsync,CH.M))}）落在淡色的那一段里，那里达不到这两条要求；绿点标的 ${minsT(best.T)}，才是本题可行最省。`)}</p>`;
  return h+chAllHTML(me,best)+`</div>`;
}
/* the passing plans only, closed by default (R1) */
function chAllHTML(me,best){
  const ps=chPass(), wmin=CH.size/CH.bw/60;
  const row=e=>`<li class="${chSame(e,me)?'me':''}" data-ok="1">${esc(chName(e))}${tx(': ','：')}${esc(tx(`wasted ${pct(e.w)}, ${TB(e.store)}`,`浪费 ${pct(e.w)}，存储 ${TB(e.store)}`))}${chSame(e,best)?esc(tx(' (least waste)','（最省）')):''}${chSame(e,me)?esc(tx(' (your plan)','（你的方案）')):''}</li>`;
  return `<details class="ckp-all"><summary>${tx(`See all settings that pass (${ps.length})`,`看全部可行的设置（${ps.length} 种）`)}</summary><div class="ckp-allb">`+
    `<ol>${ps.map(row).join('')}</ol>`+
    `<p class="small">${tx(`Whatever you choose, restarts alone take about ${pct(CH.R/CH.M)}. No synchronous plan passes: each save stops training for about ${fixd(CH.size/CH.bw/60,0)} minutes.`,`不管怎么选，光重启就要占约 ${pct(CH.R/CH.M)}。同步存档的方案没有一种达标，每存一次训练要停约 ${fixd(CH.size/CH.bw/60,0)} 分钟。`)}</p>`+
    `<p class="small muted">${tx(`Note: this challenge uses the first-order formula, which leaves out two things. First, an asynchronous save must finish writing to disk before the next save starts: ${TB(CH.size)} at 50 GB/s takes about ${fixd(wmin,0)} minutes, far less than ${best.T} minutes, so the answer does not change. Second, an asynchronous save keeps a copy of each GPU’s share in CPU memory, which this challenge assumes fits.`,
      `注意：这里用的是一阶公式，它没算两件事。一是异步存档要在下一次存档之前写完盘。${TB(CH.size)} 按每秒 50 GB 写，约 ${fixd(wmin,0)} 分钟，比 ${best.T} 分钟短得多，所以不改变结论。二是异步存档要先把每张卡上的那部分存档拷到 CPU 内存里，这个挑战假设 CPU 内存放得下。`)}</p>`+
    `</div></details>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chS.mode,chS.T,chS.k), best=chBest();
  let h;
  if(me.ok&&chSame(best,me)) h=tx(`<p class="okline ckp-mt"><b>Goal met: ${pct(me.w)} wasted, and no plan within the limits wastes less.</b></p>`,`<p class="okline ckp-mt"><b>达成目标：浪费 ${pct(me.w)}，这已经是浪费最少的方案。</b></p>`);
  else if(me.ok) h=tx(`<p class="okline ckp-mt"><b>Goal met: ${pct(me.w)} wasted. But a plan within the limits wastes less.</b></p>`,`<p class="okline ckp-mt"><b>达成目标：浪费 ${pct(me.w)}。不过还有浪费更少的方案。</b></p>`);
  else {const why=!me.fits?tx(`the ${me.k} checkpoints need ${TB(me.store)}, over the 30 TB`,`${me.k} 份存档要 ${TB(me.store)}，超过了 30 TB`):!me.rb?tx(`the oldest checkpoint is only ${minsT(me.back)} old right after a save, less than 1 hour`,`刚存完时最老的一份只是 ${minsT(me.back)}之前的，不到 1 小时`):tx(`${pct(me.w)} of the run is wasted, over 12%`,`浪费了 ${pct(me.w)}，超过 12%`);
    h=tx(`<p class="badline ckp-mt"><b>Not yet: ${why}.</b></p>`,`<p class="badline ckp-mt"><b>还没达到：${why}。</b></p>`)+
      tx('<p class="small">Hint: with a failure every 3 hours, the longer the interval, the more work each failure loses. A synchronous save stops training for almost 2 minutes. With a short interval you must keep more checkpoints to reach back 1 hour, and how many fit in 30 TB?</p>','<p class="small">提示：每 3 小时就出一次故障，间隔越长，每次丢的工作越多。同步写一份要停将近 2 分钟。间隔短了，要退回 1 小时就得多留几份，而 30 TB 只放得下几份？</p>')+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the plan that wastes least','直接看最省的方案')}</button>`);}
  if(!chReveal) return h;
  return h+chSummaryHTML(me,best);
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Predict first, then resume a crashed run for real: see what a checkpoint must hold, how to save so that a crash cannot destroy it, how often to save, and how to roll back after a loss spike. Start here if this is new to you.',
            '一共 6 个部分。先预测，再真的从存档接着训一次：看存档里要有什么、怎么存才不会被故障毁掉、多久存一次，以及 loss 突刺后怎么退回去。第一次学从这里开始。'),
  sandbox:tx('Four real situations, from a 70B run on 1,024 GPUs to a bad stretch of data. Pick one to see what goes wrong and what the checkpoint does about it.','四个真实情况，从 1,024 张卡训练 70B，到一段坏数据。选一个，看会出什么问题、存档怎么应对。'),
  challenge:tx(`Plan the checkpoints of a Llama${NB}3.1${NB}405B run that fails every 3 hours, choosing the interval, synchronous or asynchronous, and how many to keep, within 30 TB and with a checkpoint at least 1 hour old.`,
               `给一次平均每 3 小时出一次故障的 Llama${NB}3.1${NB}405B 训练定存档方案，定下间隔、同步还是异步、留几份；只有 30 TB，还要能退回 1 小时之前。`)
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> the loss curves come from a small model trained in your browser; the save-interval numbers come from a formula with the assumptions below.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The small model: an MLP with one hidden layer of 64 units and dropout 0.1, classifying 1,440 points on three spiral arms; AdamW (β1 0.9, β2 0.999, weight decay 0.01), 30 warmup steps to a learning rate of 0.05, then a cosine to 0.005 at step 600; batches of 24 in a shuffled order per epoch. Everything is seeded, so the same settings always give the same curve.</li>`+
 `<li>Resuming: a missing optimizer state means AdamW starts with m = v = 0; a missing scheduler means a new one counts from step 0; a missing data position means the loader starts the current epoch again; a missing RNG state means the generator is seeded again.</li>`+
 `<li>Interval: the first-order rule ${A.ui.math('\\text{wasted} \\approx \\frac{\\delta}{T} + \\frac{T/2 + R}{M}')}, whose minimum is ${A.ui.math('T^{*} = \\sqrt{2 \\delta M}')} (Young 1974). Checkpoint size: 14 bytes per parameter (BF16 weights 2, FP32 copy of the weights 4, Adam’s m and v 4 each), as BLOOM counts it; the FP32 gradients are not saved.</li></ul></div>`+
 `<div><div class="eyebrow">Assumed</div><ul>`+
 `<li>The whole job writes 10 GB/s. BLOOM wrote 2.3 TB from 384 GPUs in about 40 s, about 57 GB/s, but shared file systems differ a lot, so this lab takes a slower value. An asynchronous save stops training for 10 s. A restart takes 10 minutes.</li>`+
 `<li>M = 12 hours for 1,024 GPUs, from OPT-175B’s “~2 machines go down every day” on 1,024 A100s. For one 8-GPU machine, BLOOM’s “1-2 GPU failures a week” on 384 GPUs scaled down.</li>`+
 `<li>The challenge’s numbers (failures every 3 hours, 50 GB/s, 30 TB) are part of its task.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>What frameworks save, and the record written last: ${cite(SRC.meg,'Megatron-LM checkpointing.py')}, ${cite(SRC.ds,'DeepSpeed engine.py')}, ${cite(SRC.dcp,'PyTorch DCP filesystem.py')}, ${cite(SRC.tt,'torchtitan checkpointer')}, ${cite(SRC.llmc,'llm.c train_gpt2.cu')}, ${cite(SRC.nano,'nanoGPT train.py')}.</li>`+
 `<li>Real runs: ${cite(SRC.opt,'OPT-175B chronicles')}, ${cite(SRC.bloom,'BLOOM-176B README')} and ${cite(SRC.bloomc,'chronicles')}, ${cite(SRC.mle,'ml-engineering: fault tolerance')}, ${cite(SRC.mlei,'ml-engineering: instabilities')} (PaLM’s remedy, quoted there).</li>`+
 `<li>Cited by title only: J. W. Young, “A first order approximation to the optimum checkpoint interval” (1974).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>loss 曲线来自在你浏览器里真的训练的一个小模型；存档间隔的数字来自一个公式和下面的假设。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>小模型：一个隐藏层 64 个单元的 MLP，dropout 0.1，给三条螺旋臂上的 1,440 个点分类；AdamW（β1 0.9，β2 0.999，weight decay 0.01），前 30 步把学习率预热到 0.05，再沿余弦曲线降到第 600 步的 0.005；每步 24 个样本，每个 epoch 重新打乱。全部用固定的随机种子，同样的设置永远得到同样的曲线。</li>`+
 `<li>接着训：缺优化器状态，AdamW 从 m = v = 0 开始；缺调度器，新的调度器从第 0 步算起；缺数据位置，加载器把当前 epoch 从头读；缺随机数状态，生成器重新播种。</li>`+
 `<li>间隔：一阶公式是 ${A.ui.math('\\text{浪费} \\approx \\frac{\\delta}{T} + \\frac{T/2 + R}{M}')}，最小值在 ${A.ui.math('T^{*} = \\sqrt{2 \\delta M}')}（Young 1974）。存档大小：每个参数 14 字节（BF16 权重 2，FP32 权重副本 4，Adam 的 m 和 v 各 4），和 BLOOM 的算法一样。FP32 梯度不存。</li></ul></div>`+
 `<div><div class="eyebrow">假设</div><ul>`+
 `<li>整个任务每秒写 10 GB。BLOOM 用 384 张卡约 40 秒写完 2.3 TB，约每秒 57 GB，但共享文件系统之间差别很大，这里取慢一些的值。异步存档每次停 10 秒。重启一次 10 分钟。</li>`+
 `<li>1,024 张卡时 M = 12 小时，来自 OPT-175B 在 1,024 张 A100 上“大约每天坏 2 台机器”。一台 8 卡机器的 M 按 BLOOM 在 384 张卡上“每周 1–2 次 GPU 故障”折算。</li>`+
 `<li>挑战题里的数（每 3 小时一次故障、每秒 50 GB、30 TB）是题目给定的条件。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>各框架存了什么、最后写哪个记录：${cite(SRC.meg,'Megatron-LM checkpointing.py')}、${cite(SRC.ds,'DeepSpeed engine.py')}、${cite(SRC.dcp,'PyTorch DCP filesystem.py')}、${cite(SRC.tt,'torchtitan checkpointer')}、${cite(SRC.llmc,'llm.c train_gpt2.cu')}、${cite(SRC.nano,'nanoGPT train.py')}。</li>`+
 `<li>真实训练：${cite(SRC.opt,'OPT-175B 训练日志')}、${cite(SRC.bloom,'BLOOM-176B README')} 和${cite(SRC.bloomc,'训练记录')}、ml-engineering 里的${cite(SRC.mle,'容错')}和${cite(SRC.mlei,'训练不稳定')}两章（其中转述了 PaLM 的做法）。</li>`+
 `<li>只引了标题：J. W. Young，“A first order approximation to the optimum checkpoint interval”（1974）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('say what a checkpoint must hold to resume exactly, save it so a crash cannot corrupt it, and work out how often to save.','说出一份存档要有哪些东西才能原样接上，怎样存才不怕存到一半崩掉，并算出该多久存一次。');
const EXITS=()=>{
  const c=CASES.c70, d=dOf('c70','sync'), to=tOpt(d,c.M), to4=tOpt(d,c.M/4);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does a run resumed without the optimizer state jump up first?','为什么不带优化器状态接着训，loss 会先往上跳？'),
   o:[tx(`AdamW divides every update by ${A.ui.math('\\sqrt{v}')}. With m and v emptied, the first steps estimate them from one or two fresh gradients, so every parameter moves a full learning rate; normally the steps are much smaller.`,`AdamW 每次更新都要除以 ${A.ui.math('\\sqrt{v}')}。m 和 v 清空以后，头几步只凭刚算出的一两个梯度来估计它们，每个参数都走满一个学习率那么大的一步；平时的步子要小得多。`),
      tx('Without the optimizer state the learning rate falls back to 0 for the warmup, so the model stops moving and the loss stays where it was.','没有优化器状态，学习率会回到预热前的 0，模型几乎不动，loss 停在原地。'),
      tx('Without the optimizer state the weights go back to their initial values, so the first 400 steps are lost.','没有优化器状态，权重就退回到训练开始时的初始值，前面 400 步白练了。')],a:0,
   n:[null,tx('The learning rate comes from the scheduler’s step, which is a separate part; and a learning rate near 0 would freeze the loss, not make it jump.','学习率来自调度器的步数，那是另一样东西。而且学习率接近 0 只会让 loss 不动，不会让它往上跳。'),
     tx('The weights are saved on their own and restored exactly. What is lost is m, v and AdamW’s step count, which set the size of each update.','权重是单独存的，接着训时原样恢复。丢的是 m、v 和 AdamW 的步数，它们决定每一步走多大。')],
   why:tx(`AdamW’s step is ${A.ui.math('\\eta \\cdot \\frac{\\hat{m}}{\\sqrt{\\hat{v}}}')}, with η the learning rate. Late in training the gradients mostly point in random directions, so m stays small next to ${A.ui.math('\\sqrt{v}')} and the steps are small. Right after a reset both are built from the same few gradients, so their ratio is about ±1 for every parameter: the first steps are too large until m and v fill up again.`,`AdamW 一步走 ${A.ui.math('\\eta \\cdot \\frac{\\hat{m}}{\\sqrt{\\hat{v}}}')}，η 是学习率。训练后期梯度的方向大多是乱的，m 比 ${A.ui.math('\\sqrt{v}')} 小得多，步子就小。刚清空时，两者都由同样的几个梯度算出，比值对每个参数都约是 ±1，所以头几步走得太大，要等 m 和 v 重新攒起来。`)},
  {q:tx('Why write a checkpoint to a temporary name and rename it when it is complete?','为什么存档要先写到临时名字下，写完再改名？'),
   o:[tx('A rename within one file system is atomic: a reader sees the old checkpoint or the complete new one, never half of it, so a crash during the write leaves the previous checkpoint usable.','同一个文件系统里的改名是原子的：读的人看到的要么是旧存档，要么是完整的新存档，不会只看到一半；所以写的时候崩了，上一份存档照样能用。'),
      tx('Writing to a temporary name is faster, because the file system does not have to check the old file first.','写到临时名字下更快，因为文件系统不用先检查旧文件。'),
      tx('The temporary file is compressed, and the rename decompresses it, which halves the disk space.','临时文件是压缩过的，改名时才解压，所以能省一半磁盘空间。')],a:0,
   n:[null,tx('The write takes as long either way; what changes is what a crash in the middle leaves behind.','两种写法花的时间一样；变的是中途崩了会留下什么。'),
     tx('Nothing is compressed; the rename only changes the name, which is why it can be atomic.','什么都没压缩；改名只改名字，所以它才能是原子的。')],
   why:tx('Overwriting the only checkpoint erases the old one as soon as the write begins, so a crash in the middle leaves no usable checkpoint at all. With a temporary name, an atomic rename and a “latest” record written last, the newest complete checkpoint is always on disk.','直接覆盖唯一的存档，一开始写，旧的就没了。中途崩了，新的又没写完，一份能用的存档都不剩。先写临时名字、原子改名、最后才写 latest，磁盘上就永远有一份完整的最新存档。')}],
 transfer:[
  {q:tx(`A 70B run whose synchronous saves stop training for ${fixd(d,0)} s and that fails every 12 hours on average wastes least by saving about every ${mins(to)}. On a cluster four times as large, failures come every 3 hours instead; everything else stays the same. About how often should it save now?`,
        `一次 70B 的训练，每次同步存档停 ${fixd(d,0)} 秒，平均每 12 小时出一次故障，大约每 ${mins(to)}存一次最省。换到大 4 倍的集群上，故障变成平均每 3 小时一次，其余不变。现在大约多久存一次最好？`),
   o:[tx('About every 12 minutes','大约每 12 分钟'),tx('About every 25 minutes','大约每 25 分钟'),tx('About every 50 minutes, unchanged','大约每 50 分钟，不变')],a:1,
   why:tx(`The best interval grows with the square root of M: a quarter of M gives half the interval, ${A.ui.math(`\\sqrt{2 \\times ${fixd(d,0)}\\ \\text{s} \\times 10,800\\ \\text{s}}`)}, about ${mins(to4)}.`,`最好的间隔和 M 的平方根成正比：M 变成四分之一，间隔变成一半，也就是 ${A.ui.math(`\\sqrt{2 \\times ${fixd(d,0)}\\ \\text{秒} \\times 10,800\\ \\text{秒}}`)}，大约 ${mins(to4)}。`)},
  {q:tx(`In this lab an epoch is ${EPOCH} steps. The step-400 checkpoint has no data position, so the loader rereads epoch ${EP_CK} from its start: ${E.CK-(EP_CK-1)*EPOCH} steps of data are read a second time. Another run has 1,000 steps per epoch and a checkpoint at step 2,300 that again lacks only the data position. About how many steps of data does the resumed run read a second time?`,`这个 Lab 里，一个 epoch 有 ${EPOCH} 步。第 400 步的存档没有存数据位置，加载器就从第 ${EP_CK} 个 epoch 的开头重读，多读了 ${E.CK-(EP_CK-1)*EPOCH} 步的数据。换一次训练，每个 epoch 有 1,000 步，存档在第 2,300 步，同样只缺数据位置。接着训时，大约会有多少步的数据再读一遍？`),
   o:[tx('About 300 steps','约 300 步'),tx('About 1,000 steps','约 1,000 步'),tx('About 2,300 steps','约 2,300 步')],a:0,
   why:tx('Step 2,300 is 300 steps into the third epoch. The loader starts that epoch again, so the batches of those 300 steps are read twice; and if the run was planned as whole epochs, it ends 300 steps short of the last epoch’s end, so those batches are never read.','第 2,300 步是第 3 个 epoch 的第 300 步。加载器把这个 epoch 从头读，所以这 300 步的 batch 读了两遍。如果训练原本按整数个 epoch 安排，结束时最后一个 epoch 也会差 300 步没读完，那些 batch 一遍都没读到。')}],
 practice:[
  {id:'ckpt-code-review',d:tx('Rank the defects in a pull request that adds save, retention and resume: non-atomic writes, the newest checkpoint chosen wrongly, a sampler that cannot resume.','给一个加了存档、保留和恢复的 PR 排缺陷：写入不是原子的、选错最新存档、数据采样不能接着读。')},
  {id:'data-batcher-ckpt',d:tx('Part 3: make a weighted data batcher checkpointable, so that a restored batcher yields exactly the batches the original would have.','第 3 部分：让加权数据批处理器可以存档，恢复后给出的 batch 和原来的一模一样。')},
  {id:'distributed-training',d:tx('System design: put the save interval, asynchronous saving and the recovery after a failure into your plan.','系统设计：把存档间隔、异步存档和故障后的恢复放进你的方案。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Checkpoints','检查点与容错'),title:l&&l.q?L(l.q):tx('Your run crashed at step 90,000: how do you resume it?','训练到第 90,000 步崩了，怎么接着训？'),
    minutes:10,fid:'Simulated',fidPop:FIDPOP(),descs:DESCS(),mode,
    primer:tx('Primer: optimizers','前置知识：优化器'),primerPop:tx(`<p>This lab assumes you know what AdamW keeps per parameter, its running averages m and v: see ${labLink('optimizers')}. What the optimizer state weighs is in ${labLink('training-memory')}; reading loss curves is in ${labLink('train-debug')}.</p>`,
      `<p>这个 Lab 默认你知道 AdamW 给每个参数记什么：两个滑动平均 m 和 v，见${labLink('optimizers')}。优化器状态有多大，见${labLink('training-memory')}；怎么看 loss 曲线，见${labLink('train-debug')}。</p>`)});
}
function stageHTML(){
  return `<div class="panel stage ckp-stage" id="ckp-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="ckp-st-title"></div><div class="meter-sub" id="ckp-st-sub"></div></div><div class="stage-actions"><span class="status" id="ckp-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph ckp-graph" id="ckp-graph"><div class="ckp-box" id="ckp-box"></div><div class="ckp-chart" id="ckp-chart"></div><div class="ckp-terms" id="ckp-terms" hidden></div><div class="ckp-leg" id="ckp-leg"></div><div class="ckp-player" id="ckp-player"></div></div>`+
   `<div class="metrics ckp-metrics" id="ckp-metrics"></div>`+
   `<details class="math" id="ckp-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ckp-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="ckp-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ckp-lock" hidden></p>`+
   `<div class="lockable" id="ckp-ctl-view-wrap">${A.ui.seg({id:'ckp-ctl-view',key:'view',label:tx('Picture','看哪张图'),options:[['loss',tx('Resume','接着训')],['spike',tx('Spike','突刺')],['waste',tx('Interval','间隔')]],value:S.view==='files'?'loss':S.view})}</div>`+
   `<div class="lockable" id="ckp-ctl-parts-wrap"><div class="ctl"><div class="ctl-label">${tx('What the checkpoint holds (they stack)','存档里有什么（可以叠加）')}</div>${PARTS.map(p=>`<button type="button" class="btn ghost sm ckp-tg" data-part="${p}" aria-pressed="${!!S[p]}">${esc(L(PNAME[p]))}</button>`).join('')}<p class="ctl-note">${tx('The weights are always saved. Missing parts are restored as this lab’s resume script would: empty m and v, a scheduler from step 0, the epoch from its first batch, the generator from its seed.','权重一定会存。缺的部分按这个 Lab 的恢复脚本处理：m 和 v 为空，调度器从第 0 步算起，epoch 从第一个 batch 读起，随机数生成器从种子重来。')}</p></div></div>`+
   `<div class="lockable" id="ckp-ctl-fix-wrap">${A.ui.seg({id:'ckp-ctl-fix',key:'fix',label:tx('After the spike','突刺之后'),options:[['latest',tx('Latest (500)','最新（500）')],['back',tx('Back to 400','退回 400')],['skip',tx('400, skip','400，跳过')]],value:S.fix})}</div>`+
   `<div class="lockable" id="ckp-ctl-cs-wrap">${A.ui.seg({id:'ckp-ctl-cs',key:'cs',label:tx('Run','哪次训练'),options:[['c70',tx('70B, 1,024 GPUs','70B，1,024 张卡')],['c8',tx('8B, 8 GPUs','8B，8 张卡')]],value:S.cs,note:tx('M (time between failures): 12 hours for 1,024 GPUs, from OPT-175B’s log; for 8 GPUs, BLOOM’s failure rate scaled down (an assumption).','M（平均多久出一次故障）：1,024 张卡取 12 小时，来自 OPT-175B 的训练日志；8 张卡按 BLOOM 的故障率折算（假设）。')})}</div>`+
   `<div class="lockable" id="ckp-ctl-mode-wrap">${modeCtl('ckp-ctl-mode',false)}</div>`+
   `<div class="lockable" id="ckp-ctl-T-wrap">${intCtl('ckp-ctl-T',false)}<p class="ctl-note">${tx('Assumed: the job writes 10 GB/s, an asynchronous save stops training for 10 s, a restart takes 10 minutes. All three depend on the file system and the cluster; measure them on yours.','假设：整个任务每秒写 10 GB，异步存档每次停 10 秒，重启一次 10 分钟。这三项都取决于文件系统和集群，要在自己的环境里实测。')}</p></div>`+
   `<div class="lockable" id="ckp-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import os, random, torch, numpy as np

def save(run_dir, step, model, opt, sched, loader):
    state = {"step": step, "model": model.state_dict(),
             "optimizer": opt.state_dict(),     # m, v, step
             "scheduler": sched.state_dict(),   # learning rate
             "data": loader.state_dict(),       # position
             "rng": {"python": random.getstate(),
                     "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all()}}
    final = os.path.join(run_dir, f"step-{step}.pt")
    tmp = final + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush(); os.fsync(f.fileno())   # really on disk
    os.replace(tmp, final)                # atomic rename
    latest = os.path.join(run_dir, "latest")
    with open(latest + ".tmp", "w") as f:
        f.write(str(step)); f.flush(); os.fsync(f.fileno())
    os.replace(latest + ".tmp", latest)   # record written last

def load(run_dir, model, opt, sched, loader):
    step = int(open(os.path.join(run_dir, "latest")).read())
    s = torch.load(os.path.join(run_dir, f"step-{step}.pt"),
                   weights_only=False)   # own file: holds RNG states too
    model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"])
    sched.load_state_dict(s["scheduler"]); loader.load_state_dict(s["data"])
    random.setstate(s["rng"]["python"]); np.random.set_state(s["rng"]["numpy"])
    torch.set_rng_state(s["rng"]["torch"])
    torch.cuda.set_rng_state_all(s["rng"]["cuda"])
    return step`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ckp-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ckp-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Save and load all five parts, with a temporary file, an atomic rename and a “latest” record written last. The loader must be stateful (for example torchdata’s StatefulDataLoader).','存和读全部五样：先写临时文件、原子改名，最后写 latest。数据加载器要能存状态（例如 torchdata 的 StatefulDataLoader）。')}</p><pre class="code" id="ckp-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ckp-prod">`+
   tx(`<li>Megatron-LM saves the arguments, the iteration, the model, the optimizer, the scheduler and the Python, NumPy, torch and CUDA RNG states, and after a barrier writes the iteration into <code>latest_checkpointed_iteration.txt</code> (${cite(SRC.meg,'checkpointing.py')}). <code>async_save</code> and <code>iterations_to_skip</code> are in ${cite(SRC.megcfg,'training_config.py')}.</li>`+
      `<li>DeepSpeed’s <code>save_checkpoint</code> must be called on every rank and writes a <code>latest</code> file naming the tag (${cite(SRC.ds,'engine.py')}).</li>`+
      `<li>PyTorch Distributed Checkpoint writes each rank’s shard, then <code>.metadata.tmp</code>, syncs it and renames it (${cite(SRC.dcp,'filesystem.py')}); <code>async_save</code> first copies the state into CPU buffers (${cite(SRC.async,'async checkpoint recipe')}).</li>`+
      `<li>torchtitan saves the model, optimizer, scheduler, data loader and train state every 500 steps by default, keeps the latest 10 and never fewer than 2, “as the last one may be in the process of being saved” (${cite(SRC.tt,'base.py')}).</li>`,
      `<li>Megatron-LM 存参数配置、步数、模型、优化器、调度器，以及 Python、NumPy、torch 和 CUDA 的随机数状态；所有进程写完后，才把步数写进 <code>latest_checkpointed_iteration.txt</code>（${cite(SRC.meg,'checkpointing.py')}）。<code>async_save</code> 和 <code>iterations_to_skip</code> 见 ${cite(SRC.megcfg,'training_config.py')}。</li>`+
      `<li>DeepSpeed 的 <code>save_checkpoint</code> 要在每个进程上调用，最后写一个 <code>latest</code> 文件，记下这份存档的名字（${cite(SRC.ds,'engine.py')}）。</li>`+
      `<li>PyTorch 的分布式存档（DCP）先写每个进程自己的分片，再写 <code>.metadata.tmp</code>，刷盘后改名（${cite(SRC.dcp,'filesystem.py')}）；<code>async_save</code> 先把状态拷进 CPU 缓冲区（${cite(SRC.async,'异步存档教程')}）。</li>`+
      `<li>torchtitan 默认每 500 步存一次模型、优化器、调度器、数据加载器和训练状态，保留最近 10 份，至少 2 份，因为“最新那份可能还在写”（${cite(SRC.tt,'base.py')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${tx('A first order approximation to the optimum checkpoint interval','A first order approximation to the optimum checkpoint interval')}<span>${tx('J. W. Young, Communications of the ACM, 1974: the rule ','J. W. Young，Communications of the ACM，1974：')}${A.ui.math('T^{*} = \\sqrt{2 \\delta M}')}${tx(' (cited by title).',' 这条规则（只引标题）。')}</span></li>`+
   `<li>${cite(SRC.opt27,'OPT-175B chronicles')}<span>${tx('Meta’s notes from training OPT-175B on 1,024 A100s: 40+ restarts in two weeks, rolling back and lowering the learning rate.','Meta 在 1,024 张 A100 上训练 OPT-175B 的记录：两周里重启了 40 多次，退回存档、调低学习率。')}</span></li>`+
   `<li>${cite(SRC.mle,'ml-engineering: fault tolerance')}<span>${tx('Stas Bekman’s guide: BLOOM saved 2.3 TB in about 40 s every 3 hours, 0.37% of the training time.','Stas Bekman 的指南：BLOOM 每 3 小时存一次，2.3 TB 约 40 秒写完，占训练时间的 0.37%。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ckp-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ckp-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ckp-app m2" id="ckp-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ckp-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see what goes wrong and what the checkpoint does about it.','选一个场景，看看会出什么问题、存档怎么应对。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ckp-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Plan the checkpoints of a 405B run: waste at most 12%','给 405B 的训练定存档方案：浪费不超过 12%')}</h3>`+
   `<p>${tx(`A team trains Llama${NB}3.1${NB}405B. Failures come every 3 hours on average, and each restart takes 10 minutes. One checkpoint is ${TB(CH.size)}. Written synchronously at 50 GB/s, each save stops training for about ${fixd(CH.size/CH.bw,0)} s. Written asynchronously, it stops training for 10 s.`,
            `一个团队在训练 Llama${NB}3.1${NB}405B。平均每 3 小时出一次故障，每次重启要 10 分钟。一份存档 ${TB(CH.size)}。同步写时按每秒 50 GB 算，每存一次，训练要停约 ${fixd(CH.size/CH.bw,0)} 秒。异步写时只停 10 秒。`)}</p>`+
   `<p>${tx('The storage holds at most 30 TB of checkpoints. To handle loss spikes, right after each save the oldest checkpoint you keep must be at least 1 hour old. Choose the save interval, synchronous or asynchronous, and how many to keep, so that at most 12% of the run is wasted, the less the better.','存储最多放 30 TB 的存档。为了处理 loss 突刺，每次存完新档，手里最老的那份必须是至少 1 小时前存的。选好存档间隔、同步还是异步、留几份，让白费的时间不超过整次训练的 12%，越少越好。')}</p>`+
   A.ui.seg({id:'ckp-ic-chT',key:'chT',label:tx('Save every','每隔多久存一次'),options:CH.ints.map(T=>[String(T),minsT(T)]),value:String(chS.T),inline:true})+
   A.ui.seg({id:'ckp-ic-chm',key:'chm',label:tx('How to save','怎么存'),options:[['sync',tx('Synchronous','同步')],['async',tx('Asynchronous','异步')]],value:chS.mode,inline:true})+
   A.ui.seg({id:'ckp-ic-chk',key:'chk',label:tx('Checkpoints kept','留几份'),options:CH.keeps.map(k=>[String(k),String(k)]),value:String(chS.k),inline:true})+
   `<div class="goal mono ckp-goal" id="ckp-ch-g1"><span>${tx('Storage','存储')}</span><span><b id="ckp-ch-s"></b>${tx(' (at most 30 TB)','（不超过 30 TB）')}</span></div>`+
   `<div class="goal mono ckp-goal" id="ckp-ch-g2"><span>${tx('Oldest after a save','刚存完时最老的一份')}</span><span><b id="ckp-ch-b"></b>${tx(' (at least 1 h)','（至少 1 小时）')}</span></div>`+
   `<div class="goal mono ckp-goal" id="ckp-ch-g3"><span>${tx('Time wasted','浪费的时间')}</span><span><b id="ckp-ch-w"></b>${tx(' (at most 12%)','（不超过 12%）')}</span></div>`+
   submit+`<div id="ckp-ch-res" class="ckp-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},hold:{en:'Add all four back to continue, or press Show me.',zh:'把四样都加回去才能继续，也可以点“演示给我看”。'},
    atomic:{en:'Pick the second way of saving to continue.',zh:'选第二种存法才能继续。'},interval:{en:'Move the interval and try asynchronous saving to continue.',zh:'拖一下间隔，再试试异步存档，才能继续。'},spike:{en:'Pick the last option to continue.',zh:'选最后一项才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#ckp-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function obsReady(){const s=STEPS[step]; if(!s.obs) return false; if(s.id==='hold') return watched.has('ls:0000'); if(s.id==='spike') return watched.has('sp:latest'); return true;}
function updateRail(){
  const rail=$('#ckp-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=obsReady();
    $$('[data-lever]',rail).forEach(b=>{const [k,x]=b.dataset.lever.split(':'); let on=false, res='';
      if(k==='atom'){on=x===S.atom; res=atomTried.has(x)||on?tx(`${redone(x)} steps redone`,`重算 ${redone(x)} 步`):'';}
      else if(k==='fix'){on=x===S.fix; res=fixShown(x)?fixRes(x):'';}
      b.setAttribute('aria-pressed',String(on)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=res;});
    $$('#ckp-ic-parts [data-part]',rail).forEach(b=>{const on=!!S[b.dataset.part]; b.setAttribute('aria-pressed',String(on)); b.classList.toggle('on',on); b.querySelector('.ckp-sw').textContent=on?'✓':'';});
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    $$('[data-nav="showme"]',rail).forEach(sm=>{sm.hidden=!!observed[step];});
    $$('.ckp-rem',rail).forEach(b=>{const on=b.dataset.rem===remView; b.setAttribute('aria-pressed',String(on)); b.textContent=on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看');});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ckp-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chS.mode,chS.T,chS.k), s=$('#ckp-ch-s',rail); if(!s) return;
    s.textContent=TB(e.store); $('#ckp-ch-b',rail).textContent=minsT(e.back); $('#ckp-ch-w',rail).textContent=pct(e.w);
    const g=(id,ok)=>{const el=$(id,rail); el.classList.toggle('met',ok); el.classList.toggle('ckp-miss',!ok);};
    g('#ckp-ch-g1',e.fits); g('#ckp-ch-g2',e.rb); g('#ckp-ch-g3',e.w<=CH.goal+1e-12);
    $$('.seg-ctl[data-key="chT"]',rail).forEach(x=>A.ui.setPressed(x,String(chS.T))); $$('.seg-ctl[data-key="chm"]',rail).forEach(x=>A.ui.setPressed(x,chS.mode)); $$('.seg-ctl[data-key="chk"]',rail).forEach(x=>A.ui.setPressed(x,String(chS.k)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#ckp-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);
      const dd=$('details.ckp-all',res); if(dd){dd.open=allOpen; dd.addEventListener('toggle',()=>{allOpen=dd.open;});}}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
const hidden0=()=>mode==='guided'&&isP('mission');
const stageView=()=>mode==='challenge'?'challenge':S.view;
/* the loss picture: x = step 0..600, y = validation loss on a log axis */
function lossSeries(){
  const b=base();
  if(S.view==='spike'){const sp=spikeRun(), r=remedy(S.fix); return {ref:b.ev,run:sp.ev,cut:E.STEPS,res:r.ev,from:r.from,crash:null,ymax:Math.max(20,r.peak*1.3)};}
  const r=resumed(keepOf(S)); return {ref:b.ev,run:b.ev,cut:E.CRASH,res:r.ev,from:E.CK,crash:E.CRASH,ymax:1.6};
}
/* time line of the whole-run animation: phase A draws the run to the crash (or to the end), phase B draws the resumed run */
function phases(se){const a=se.cut, b=E.STEPS-se.from, pause=40; const tot=a+pause+b; return {a:a/tot,p:(a+pause)/tot,tot};}
function drawLoss(){
  const host=$('#ckp-chart',app), W=Math.max(260,Math.round(host.clientWidth||520)), DS=S.view==='loss'?44:0, H=(innerWidth<=760?220:260)+DS;
  const se=lossSeries(), ph=phases(se), hid=hidden0();
  /* right margin wide enough for the longest curve name at the line ends (10px mono: about 6.1px a letter, 10.5px a Chinese character) */
  const endT=S.view==='spike'?[tx('no bad data','没有坏数据'),tx('with bad data','有坏数据'),tx('resumed','接着训')]:[tx('no crash','没崩时'),tx('resumed','接着训')];
  const fz=innerWidth<=760?1.1:1, tw=t=>[...t].reduce((a,c)=>a+(c.charCodeAt(0)>0x2e7f?10.5:6.1),0)*fz;
  const top=14, bot=H-30-DS, lx=40, rx=W-Math.max(innerWidth<=760?58:84,Math.ceil(Math.max(...endT.map(tw)))+10), ylo=Math.log10(0.02), yhi=Math.log10(se.ymax);
  const X=s=>lx+(s/E.STEPS)*(rx-lx), Y=v=>bot-((Math.log10(Math.max(v,0.02))-ylo)/(yhi-ylo))*(bot-top);
  const pa=from0?Math.min(1,playT/ph.a):1, sA=Math.round(se.cut*pa);
  const pb=playT<=ph.p?0:Math.min(1,(playT-ph.p)/(1-ph.p)), sB=se.from+Math.round((E.STEPS-se.from)*(from0?pb:Math.min(1,playT)));
  const showB=!hid&&(from0?playT>ph.p:true);
  let g='';
  const ticks=[0.03,0.1,0.3,1,3,10].filter(v=>Math.log10(v)<=yhi);
  ticks.forEach(v=>{g+=`<line x1="${lx}" x2="${rx}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="ckp-grid"/><text x="${lx-4}" y="${(Y(v)+3.5).toFixed(1)}" class="ckp-ax ckp-r">${v}</text>`;});
  [0,100,200,300,400,500,600].forEach(s=>{g+=`<text x="${X(s).toFixed(1)}" y="${bot+14}" class="ckp-ax ckp-mid">${s}</text>`;});
  g+=`<text x="${((lx+rx)/2).toFixed(1)}" y="${bot+27}" class="ckp-ax ckp-mid">${esc(tx('training step','训练步数'))}</text>`;
  const path=(ev,a,b)=>{const pts=[]; for(let s=a;s<=b;s+=E.EVERY){if(ev[s]!=null) pts.push(`${X(s).toFixed(1)},${Y(ev[s]).toFixed(1)}`);} return pts.join(' ');};
  /* reference: what the run would have done */
  if(!hid&&S.view==='loss') g+=`<polyline points="${path(se.ref,0,E.STEPS)}" class="ckp-ref"/>`;
  if(S.view==='spike') g+=`<polyline points="${path(se.ref,0,E.STEPS)}" class="ckp-ref"/>`;
  /* the actual run, to the crash */
  const runEnd=Math.min(sA,se.cut); g+=`<polyline points="${path(se.run,0,runEnd-runEnd%E.EVERY)}" class="ckp-run${S.view==='spike'?' ckp-runsp':''}"/>`;
  if(S.view==='spike'&&sA>=E.BAD0){g+=`<rect x="${X(E.BAD0).toFixed(1)}" y="${top}" width="${Math.max(2,X(E.BAD0+E.BADN)-X(E.BAD0)).toFixed(1)}" height="${bot-top}" class="ckp-badzone"/><text x="${(X(E.BAD0)+4).toFixed(1)}" y="${top+10}" class="ckp-ax ckp-bad">${esc(tx('bad data','坏数据'))}</text>`;}
  /* checkpoint markers */
  const cks=S.view==='spike'?[400,500]:[100,200,300,400];
  cks.forEach(s=>{if(s<=Math.max(sA,se.from)) g+=`<rect x="${(X(s)-3.5).toFixed(1)}" y="${bot-3.5}" width="7" height="7" class="ckp-ck${s===se.from&&showB?' on':''}"/>`;});
  /* the crash */
  if(se.crash&&sA>=se.crash){const cy=Y(se.run[se.crash]); g+=`<line x1="${X(se.crash).toFixed(1)}" x2="${X(se.crash).toFixed(1)}" y1="${top}" y2="${bot}" class="ckp-crashl"/><text x="${X(se.crash).toFixed(1)}" y="${(cy+5).toFixed(1)}" class="ckp-x">✕</text><text x="${(X(se.crash)+5).toFixed(1)}" y="${top+10}" class="ckp-ax ckp-bad">${esc(tx('crash','崩了'))}</text>`;}
  /* resumed run, dashed; red where it is above the reference */
  if(showB){const end=sB-sB%E.EVERY;
    let red=''; for(let s=se.from;s<end;s+=E.EVERY){const a=se.res[s], b=se.res[s+E.EVERY], ra=se.ref[s], rb=se.ref[s+E.EVERY]; if(a==null||b==null) continue; if(a>ra+1e-12||b>rb+1e-12) red+=`<polygon points="${X(s).toFixed(1)},${Y(ra).toFixed(1)} ${X(s+E.EVERY).toFixed(1)},${Y(rb).toFixed(1)} ${X(s+E.EVERY).toFixed(1)},${Y(Math.max(b,rb)).toFixed(1)} ${X(s).toFixed(1)},${Y(Math.max(a,ra)).toFixed(1)}" class="ckp-worse"/>`;}
    g+=red+`<polyline points="${path(se.res,se.from,end)}" class="ckp-res"/>`;}
  /* curve names at their ends (R5.9) */
  const lab=[];
  if(!hid&&(S.view==='spike'||S.view==='loss')) lab.push({y:Y(se.ref[E.STEPS]),t:S.view==='spike'?tx('no bad data','没有坏数据'):tx('no crash','没崩时'),c:'ckp-lref'});
  if(S.view==='spike'&&sA>=E.STEPS) lab.push({y:Y(se.run[E.STEPS]),t:tx('with bad data','有坏数据'),c:'ckp-lrun'});
  if(showB&&sB>=E.STEPS) lab.push({y:Y(se.res[E.STEPS]),t:tx('resumed','接着训'),c:'ckp-lres'});
  lab.sort((a,b)=>a.y-b.y); for(let i=1;i<lab.length;i++) if(lab[i].y-lab[i-1].y<12) lab[i].y=lab[i-1].y+12;
  /* never down into the step numbers under the axis (a phone's shorter chart): lift the stack from the bottom */
  for(let i=lab.length-1;i>=0;i--){const lim=i===lab.length-1?bot-3:lab[i+1].y-12; if(lab[i].y>lim) lab[i].y=lim;}
  lab.forEach(l=>{g+=`<text x="${rx+5}" y="${(l.y+3.5).toFixed(1)}" class="ckp-ax ${l.c}">${esc(l.t)}</text>`;});
  g+=`<text x="${lx-30}" y="${top-3}" class="ckp-ax">${esc(tx('validation loss (log scale)','验证集 loss（对数坐标）'))}</text>`;
  if(DS) g+=dataStrip(X,lx,bot,Math.min(sA,E.CK),showB?sB:E.CK);
  host.innerHTML=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(lossAria(se))}">${g}</svg>`;
}
/* the data each step reads (owner b8, scenario 3), on the loss picture's step axis: one cell per epoch of 60 steps. Up to the checkpoint
   at step 400 the run's own order; from the resume on, the resumed run's. Without the data position the loader starts epoch 7 again:
   its first 40 steps are read twice (red), and the run stops 40 steps short of the last epoch's end (red hatching, past step 600).
   upTo: drawn up to this step before the resume; resTo: the resumed run drawn up to this step (= 400 while it is hidden). */
function dataStrip(X,lx,bot,upTo,resTo){
  const ry=bot+45, rh=12, sh=E.CK-(EP_CK-1)*EPOCH, keepD=!!S.d;
  let g=`<g class="ckp-ds"><defs><pattern id="ckp-nhatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" class="ckp-nbg"/><line x1="0" y1="0" x2="0" y2="5" class="ckp-nl"/></pattern></defs>`;
  g+=`<text x="${innerWidth<=760?4:lx}" y="${bot+41}" class="ckp-ax">${esc(tx('Data read per step (one cell per epoch)','每步读到的数据（每格一个 epoch）'))}</text>`;
  const cell=(a,b,ep)=>{if(b<=a) return ''; const x1=X(a), x2=X(b); let c=`<rect x="${x1.toFixed(1)}" y="${ry}" width="${(x2-x1).toFixed(1)}" height="${rh}" class="ckp-ep${ep%2}" data-ep="${ep+1}"/>`;
    if(x2-x1>=14) c+=`<text x="${((x1+x2)/2).toFixed(1)}" y="${ry+rh-2.5}" class="ckp-epn">${ep+1}</text>`; return c;};
  for(let e=0;e<E.STEPS/EPOCH;e++) g+=cell(e*EPOCH,Math.min((e+1)*EPOCH,upTo),e);
  if(resTo>E.CK){const off=keepD?0:sh;
    for(let e=0;e<E.STEPS/EPOCH;e++){const a=Math.max(E.CK,e*EPOCH+off), b=Math.min((e+1)*EPOCH+off,resTo); g+=cell(a,b,e);}
    if(!keepD){const b2=Math.min(E.CK+sh,resTo);
      g+=`<rect x="${X(E.CK).toFixed(1)}" y="${ry}" width="${(X(b2)-X(E.CK)).toFixed(1)}" height="${rh}" class="ckp-twice"/><text x="${((X(E.CK)+X(b2))/2).toFixed(1)}" y="${ry+rh+12}" class="ckp-ax ckp-mid ckp-dsl">${esc(tx('read twice','读了两遍'))}</text>`;
      if(resTo>=E.STEPS) g+=`<rect x="${X(E.STEPS).toFixed(1)}" y="${ry}" width="${(X(E.STEPS+sh)-X(E.STEPS)).toFixed(1)}" height="${rh}" class="ckp-never"/><text x="${(X(E.STEPS)+1).toFixed(1)}" y="${ry+rh+12}" class="ckp-ax ckp-dsl">${esc(tx('unread','没读到'))}</text>`;}}
  return g+`</g>`;
}
function lossAria(se){if(hidden0()) return tx('The run up to the crash at step 450; the resumed run is shown after you predict.','训练一直画到第 450 步崩溃；接着训的曲线在你预测后才显示。');
  if(S.view==='spike'){const r=remedy(S.fix); return tx(`Validation loss with bad data at steps 470–474; ${L(FIX[S.fix].t)}: loss ${lossF(r.final)} at step 600.`,`第 470–474 步有坏数据时的验证集 loss；${L(FIX[S.fix].t)}：第 600 步 loss ${lossF(r.final)}。`);}
  const r=resumed(keepOf(S)); return tx(`Validation loss: crash at step 450, resumed from step 400; at step 600 the resumed run is at ${lossF(r.final)}, without the crash ${lossF(r.final0)}.`,`验证集 loss：第 450 步崩了，从第 400 步接着训；第 600 步接着训的是 ${lossF(r.final)}，没崩时是 ${lossF(r.final0)}。`);}
function drawBox(){
  const host=$('#ckp-box',app), v=S.view;
  if(v!=='loss'||mode==='challenge'){if(host._h!==''){host.innerHTML=''; host._h='';} host.hidden=true; return;}
  host.hidden=false;
  const cell=(on,name,tip)=>`<span class="ckp-cell${on?' on':''}" tabindex="0" data-tipx="${esc(tip)}"><i aria-hidden="true">${on?'✓':'∅'}</i>${esc(name)}${on?'':`<em class="ckp-no">${esc(tx(' (not saved)','（没存）'))}</em>`}</span>`;
  const h=`<div class="ckp-boxt">${tx('Checkpoint of step 400 holds:','第 400 步的存档里有：')}</div><div class="ckp-cells">`+cell(true,tx('Weights','权重'),tx('The weights are always saved.','权重一定会存。'))+
    PARTS.map(p=>cell(!!S[p],L(PNAME[p]),(S[p]?tx('Saved: ','已存：'):tx('Missing: ','没存：'))+L(PDESC[p]))).join('')+`</div>`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
}
function drawFiles(){
  const host=$('#ckp-chart',app), plain=S.atom==='plain';
  const f=(name,cls,txt,tip)=>`<div class="ckp-file ${cls}" tabindex="0" data-tipx="${esc(tip)}"><b class="mono">${esc(name)}</b><span>${esc(txt)}</span></div>`;
  const row=(t,files)=>`<div class="ckp-frow"><div class="ckp-ft">${esc(t)}</div><div class="ckp-files">${files}</div></div>`;
  let h='';
  if(plain){
    h+=row(tx('1. After the save at step 400','1. 第 400 步存完'),f('ckpt.pt','ok',tx('step 400, complete','第 400 步，完整'),tx('The only checkpoint, complete.','唯一的一份存档，完整。')));
    h+=row(tx('2. Saving step 500: the machine dies at 60%','2. 第 500 步存到 60%，机器坏了'),f('ckpt.pt','bad',tx('first 60% of step 500; step 400 erased','只写进第 500 步的前 60%，第 400 步的已清空'),tx('Overwriting empties the old file as soon as the write begins. Step 500 got only 60% written, so the file cannot be loaded.','覆盖时，一开始写就把旧文件清空了。第 500 步又只写进 60%，所以这个文件读不出来。')));
    h+=row(tx('3. After the restart','3. 重启以后'),`<div class="ckp-verdict bad">${esc(tx(`No usable checkpoint: train again from step 0 (${redone('plain')} steps redone).`,`没有能用的存档，只能从第 0 步重训（重算 ${redone('plain')} 步）。`))}</div>`);
  } else {
    h+=row(tx('1. After the save at step 400','1. 第 400 步存完'),f('step-400/','ok',tx('complete','完整'),tx('Written as step-400.tmp, then renamed.','先写成 step-400.tmp，写完再改名。'))+f('latest','rec','400',tx('The record of the newest complete checkpoint, written last.','记录最新的完整存档，最后才写。')));
    h+=row(tx('2. Saving step 500: the machine dies at 60%','2. 第 500 步存到 60%，机器坏了'),f('step-400/','ok',tx('complete','完整'),tx('Untouched by the new save.','新存档不会碰它。'))+f('step-500.tmp','bad',tx('60%, never renamed','60%，没改名'),tx('Half written. It keeps its temporary name, so nothing will load it.','只写了一半。它还是临时名字，所以不会有人去读它。'))+f('latest','rec','400',tx('Still 400: it is updated only after step 500 is complete.','还是 400：第 500 步写完以后才会更新它。')));
    h+=row(tx('3. After the restart','3. 重启以后'),`<div class="ckp-verdict ok">${esc(tx(`latest says 400: resume from step 400 (${redone('atomic')} steps redone).`,`latest 写着 400：从第 400 步接着训（重算 ${redone('atomic')} 步）。`))}</div>`);
  }
  const key='files'+S.atom+A.lang(); if(host._h!==key){host.innerHTML=`<div class="ckp-fs" role="img" aria-label="${esc(tx('Files on disk before, during and after the crash.','故障之前、之中和之后磁盘上的文件。'))}">${h}</div>`; host._h=key;}
}
/* the interval picture: wasted share against the interval, both save modes; under it a strip over one mean time between failures
   with that one failure (12 hours for the 70B, 3 hours in the challenge), or, for the 8B whose failures are months apart, 24 hours
   without one. The strip's caption has its own line under the axis title (R7.8). */
function stripOf(){
  if(mode==='challenge') return {span:CH.M,fail:true};
  if(S.cs==='c8') return {span:24*3600,fail:false};
  return {span:CASES[S.cs].M,fail:true};
}
function drawWaste(){
  const host=$('#ckp-chart',app), W=Math.max(260,Math.round(host.clientWidth||520)), H0=innerWidth<=760?224:254;
  let dS,dA,M,R,Tm,md;
  if(mode==='challenge'){dS=CH.size/CH.bw; dA=CH.dAsync; M=CH.M; R=CH.R; Tm=chS.T; md=chS.mode;}
  else {const c=CASES[S.cs]; dS=dOf(S.cs,'sync'); dA=dOf(S.cs,'async'); M=c.M; R=c.R; Tm=S.T; md=S.mode;}
  const lx=40, rx=W-22, x0=Math.log(5), x1=Math.log(480), ymax=mode==='challenge'?0.4:(S.cs==='c8'?0.05:0.4);
  const X=m=>lx+((Math.log(m)-x0)/(x1-x0))*(rx-lx);
  /* in the challenge, once the answer is shown (R15): the plan that wastes least within the storage and rollback limits (chBest) is labelled
     above the plot, right of its line, on one row or, when that would run past the right edge, two (broken after the colon). The chart
     grows by that band, so that no curve can cross the label and the plot keeps its size */
  const chk=mode==='challenge', lim=chk&&chReveal?chLimit():null, fz=innerWidth<=760?1.1:1, wOf=t=>[...t].reduce((a,ch)=>a+(/[\u3000-\u9fff\uff00-\uffef]/.test(ch)?10:6),0)*fz;
  let bl=null;
  if(lim){const b=chBest(), xb=X(b.T);
    const lab=b.mode===md?tx(`least within the limits: ${minsT(b.T)}`,`本题可行最省：${minsT(b.T)}`):tx(`least within the limits: ${b.mode==='async'?'asynchronous':'synchronous'}, ${minsT(b.T)}`,`本题可行最省：${b.mode==='async'?'异步':'同步'}，${minsT(b.T)}`);
    const ci=Math.max(lab.indexOf(':'),lab.indexOf('：'))+1, rows=wOf(lab)>W-2-(xb+5)&&ci>0?[lab.slice(0,ci),lab.slice(ci).trim()]:[lab];
    const lw=Math.max(...rows.map(wOf)), lxb=Math.max(lx,Math.min(xb+5,W-2-lw)); bl={b,xb,rows,lxb,lw};}
  const band=bl?Math.round(bl.rows.length*13*fz):0, H=H0+band, top=12+band, bot=H-76;
  const Y=v=>bot-(Math.min(v,ymax)/ymax)*(bot-top);
  let g=`<defs><pattern id="ckp-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" class="ckp-hbg"/><line x1="0" y1="0" x2="0" y2="5" class="ckp-hl"/></pattern></defs>`;
  for(let k=0;k<=4;k++){const v=ymax*k/4; g+=`<line x1="${lx}" x2="${rx}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="ckp-grid"/><text x="${lx-4}" y="${(Y(v)+3.5).toFixed(1)}" class="ckp-ax ckp-r">${fixd(v*100,ymax<0.1?1:0)}%</text>`;}
  /* fewer ticks on a narrow chart; the first label starts at its tick so that it clears the 0% of the y axis (R7.8) */
  (W<480?[5,30,120,480]:[5,15,30,60,120,240,480]).forEach(m=>{g+=`<text x="${(m===5?X(m)-3:X(m)).toFixed(1)}" y="${bot+13}" class="ckp-ax${m===5?'':' ckp-mid'}">${esc(m<60?m+tx(' min',' 分钟'):(m/60)+tx(' h',' 小时'))}</text>`;});
  g+=`<text x="${((lx+rx)/2).toFixed(1)}" y="${bot+26}" class="ckp-ax ckp-mid">${esc(tx('save interval (log scale)','存档间隔（对数坐标）'))}</text>`;
  const curve=d=>{const p=[]; for(let i=0;i<=80;i++){const m=Math.exp(x0+(x1-x0)*i/80); p.push(`${X(m).toFixed(1)},${Y(waste(d,m*60,M,R).total).toFixed(1)}`);} return p.join(' ');};
  g+=`<polyline points="${curve(dS)}" class="ckp-wc${md==='sync'?' on':''}"/><polyline points="${curve(dA)}" class="ckp-wc ckp-wca${md==='async'?' on':''}"/>`;
  /* in the challenge, once the answer is shown: the intervals that break the storage or rollback limit are pale, and the plan that wastes
     least within the limits (chBest) is marked, a line from the top of the plot down to its dot (R15) */
  let bm=null;
  if(lim){const xm=X(Math.min(480,Math.max(5,lim.tmin))); g+=`<rect x="${lx}" y="${top}" width="${Math.max(0,xm-lx).toFixed(1)}" height="${bot-top}" class="ckp-nofit"/>`;
    bm={...bl,yb:Y(waste(bl.b.mode==='async'?dA:dS,bl.b.T*60,M,R).total)};}
  const d=md==='async'?dA:dS, to=tOpt(d,M)/60;
  if(to>=5&&to<=480){const wo=waste(d,to*60,M,R).total, xo=X(to), yo=Y(wo), f=(xo-lx)/(rx-lx);
    /* the label goes under the dot; when a curve or the ring of the chosen interval would run through it there (the asynchronous
       curve under the 70B's synchronous minimum, the steep U of the challenge on a phone), it goes above (a little higher if the curve
       still touches it), then to the left or the right of the dot; under the axis it cannot go. It leans left near the right edge (R7.8). The label's box is estimated:
       10 px per Chinese character, 6 px for the others, at the axis labels' 10 px. */
    const lab=chk?tx(`by waste alone: ${mins(to*60)}`,`只看时间浪费：${mins(to*60)}`):tx(`least: ${mins(to*60)}`,`最少：${mins(to*60)}`), lw=wOf(lab);
    const od=md==='async'?dS:dA, inv=px=>Math.exp(x0+(px-lx)/(rx-lx)*(x1-x0)), xn=X(Tm), yn=Y(waste(d,Tm*60,M,R).total);
    const lean=f>0.8?'end':(f<0.12?'start':'middle'), at=a=>a==='end'?xo+6:(a==='start'?xo-6:xo);
    /* how many points of the two curves (and the ring) fall inside the label's box; Infinity when the box leaves the plot */
    const hits=(a,v)=>{const ax=at(a), la=a==='end'?ax-lw:(a==='start'?ax:ax-lw/2), t=v-10.5*fz, b=v+2.5*fz;
      if(la<lx-2||la+lw>W-2||b>bot||t<0) return Infinity;
      let n=Math.abs(xn-Math.min(Math.max(xn,la),la+lw))<5.5&&yn>t-5.5&&yn<b+5.5?1:0;
      if(bm&&((bm.xb>la-3&&bm.xb<la+lw+3&&t<bm.yb)||(la<bm.lxb+bm.lw&&la+lw>bm.lxb&&t<top))) n+=20;
      for(let k=0;k<=12;k++){const px=Math.min(rx,Math.max(lx,la+1+(lw-2)*k/12)); for(const dd of [d,od]){const yy=Y(waste(dd,inv(px)*60,M,R).total); if(yy>t&&yy<b) n++;}}
      return n;};
    const cand=[[lean,yo+16],[lean,yo-9],[lean,yo-14],['end',yo+16],['start',yo+16],['end',yo-9],['start',yo-9],['end',yo-14],['start',yo-14]].map(([a,v])=>[a,v,hits(a,v)]);
    const [anc,ly]=cand.find(c=>c[2]===0)||cand.reduce((m,c)=>c[2]<m[2]?c:m), lxx=at(anc);
    g+=`<circle cx="${xo.toFixed(1)}" cy="${yo.toFixed(1)}" r="4" class="ckp-opt"/><text x="${lxx.toFixed(1)}" y="${ly.toFixed(1)}" class="ckp-ax" text-anchor="${anc}">${esc(lab)}</text>`;}
  if(bm) g+=`<line x1="${bm.xb.toFixed(1)}" x2="${bm.xb.toFixed(1)}" y1="${bm.lxb>=bm.xb+4?3:top}" y2="${(bm.yb-6).toFixed(1)}" class="ckp-best-l"/><circle cx="${bm.xb.toFixed(1)}" cy="${bm.yb.toFixed(1)}" r="4" class="ckp-best"/>`+bm.rows.map((r,i)=>`<text x="${bm.lxb.toFixed(1)}" y="${(12+(13*(i+1)-7)*fz).toFixed(1)}" class="ckp-ax ckp-best-t">${esc(r)}</text>`).join('');
  const wn=waste(d,Tm*60,M,R).total; g+=`<circle cx="${X(Tm).toFixed(1)}" cy="${Y(wn).toFixed(1)}" r="5.5" class="ckp-now"/>`;
  /* the strip */
  const st=stripOf(), sy=bot+47, sh=12, span=st.span, F=0.608*span, SX=t=>lx+(t/span)*(rx-lx), hrs=span/3600;
  g+=`<rect x="${lx}" y="${sy}" width="${rx-lx}" height="${sh}" class="ckp-train"/>`;
  /* each save stops training for δ: a synchronous one is a block of that width (at least 1.5 px), an asynchronous one a thin, lighter tick */
  const asy=md==='async';
  let last=0; for(let t=Tm*60;t<span;t+=Tm*60){if(t<=F) last=t; g+=`<rect x="${SX(t).toFixed(1)}" y="${sy}" width="${Math.max(asy?1:1.5,SX(t+d)-SX(t)).toFixed(1)}" height="${sh}" class="ckp-save${asy?' ckp-savea':''}"/>`;}
  if(st.fail){g+=`<rect x="${SX(last).toFixed(1)}" y="${sy}" width="${Math.max(1,SX(F)-SX(last)).toFixed(1)}" height="${sh}" class="ckp-lost"/><rect x="${SX(F).toFixed(1)}" y="${sy}" width="${Math.max(1,SX(F+R)-SX(F)).toFixed(1)}" height="${sh}" class="ckp-rst"/>`;
    g+=`<line x1="${SX(F).toFixed(1)}" x2="${SX(F).toFixed(1)}" y1="${sy-4}" y2="${sy+sh+4}" class="ckp-fail"/>`;}
  g+=`<text x="${lx}" y="${sy-5}" class="ckp-ax">${esc(st.fail?tx(`${hrs} hours, one failure (illustration)`,`${hrs} 小时里的一次故障（示意）`):tx(`${hrs} hours, no failure (illustration)`,`${hrs} 小时里没有故障（示意）`))}</text>`;
  host.innerHTML=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx(`Wasted share against the save interval: now ${pct(wn)} at every ${minsT(Tm)}.`,`浪费的比例随存档间隔的变化：现在每 ${minsT(Tm)}存一次，浪费 ${pct(wn)}。`))}">${g}</svg>`;
}
/* the wasted share split into the formula's three terms, on the chart's own 0–40% scale (owner b8, part 4): saving shrinks when the
   saves are asynchronous, redone work grows with the interval, restarts stay. Each part has a hover with its share worked out. */
/* “=” when the three shares as shown add up to the total as shown, “≈” when rounding makes them differ by a digit */
function addsUp(q){const dg=q.fine?2:1, u=t=>/^[0-9]/.test(t)?Math.round(parseFloat(t)*10**dg):0;
  return TERMS.reduce((a,t)=>a+u(termPct(q.w[t.k],q.fine)),0)===u(pct(q.w.total,dg));}
function drawTerms(){
  const host=$('#ckp-terms',app); if(!host) return;
  const v=stageView();
  if(v!=='waste'&&v!=='challenge'){if(host._h!==''){host.innerHTML=''; host._h='';} host.hidden=true; return;}
  host.hidden=false;
  const q=wasteNow(), seg=k=>Math.max(0,Math.min(100,q.w[k]/q.ymax*100)), cls={save:'save',lost:'lost',restart:'rst'};
  const aria=tx(`Wasted ${pct(q.w.total,q.fine?2:1)}: saving ${termPct(q.w.save,q.fine)}, redone work ${termPct(q.w.lost,q.fine)}, restarts ${termPct(q.w.restart,q.fine)}.`,
                `白费 ${pct(q.w.total,q.fine?2:1)}：存档 ${termPct(q.w.save,q.fine)}，重算 ${termPct(q.w.lost,q.fine)}，重启 ${termPct(q.w.restart,q.fine)}。`);
  const item=(t,i)=>`<span tabindex="0" data-term="${t.k}" data-tipx="${esc(termTip(t.k,q))}">${i?'<em class="ckp-op" aria-hidden="true">+</em>':''}<i class="sw ${t.sw}" aria-hidden="true"></i>${esc(L(TNAME[t.k]))} ${A.ui.math(t.inl)} <b>${esc(termPct(q.w[t.k],q.fine))}</b></span>`;
  const h=`<div class="ckp-terms-h">${esc(tx('The wasted time is the sum of three parts','白费的时间是三项之和'))}</div>`+
    `<div class="ckp-tbar" role="img" aria-label="${esc(aria)}">${TERMS.map(t=>`<i class="${cls[t.k]}" data-seg="${t.k}" style="width:${seg(t.k).toFixed(2)}%"></i>`).join('')}</div>`+
    `<div class="ckp-tscale" aria-hidden="true"><span>0%</span><span>${fixd(q.ymax*100,0)}%</span></div>`+
    `<div class="ckp-tlist">${TERMS.map(item).join('')}<strong class="ckp-tsum mono"><em class="ckp-op" aria-hidden="true">${addsUp(q)?'=':'≈'}</em> ${esc(pct(q.w.total,q.fine?2:1))}</strong></div>`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
}
function drawLegend(){
  const lg=$('#ckp-leg',app), v=mode==='challenge'?'waste':S.view;
  const it=(sw,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${sw}"></i>${esc(t)}</span>`;
  let h='';
  if(v==='loss'&&hidden0()){
    h=it('ckp-sw-run',tx('solid: the run until the crash','实线：崩之前的训练'),tx('What actually ran.','实际跑出来的。'))+
      it('ckp-sw-ck',tx('square: checkpoint','方块：存档'),tx('A checkpoint is saved every 100 steps.','每 100 步存一次档。'));
  } else if(v==='loss'||v==='spike'){
    h=it('ckp-sw-ref',v==='spike'?tx('gray line: no bad data','灰线：没有坏数据'):tx('gray line: no crash','灰线：没崩时'),tx('The same run with nothing going wrong.','同一次训练，什么问题都没出。'))+
      it('ckp-sw-run',v==='spike'?tx('solid: with bad data','实线：有坏数据'):tx('solid: the run until the crash','实线：崩之前的训练'),tx('What actually ran.','实际跑出来的。'))+
      it('ckp-sw-res',tx('dashed: resumed','虚线：接着训'),tx('Resumed from the checkpoint marked on the axis.','从横轴上标出的那份存档接着训。'))+
      it('ckp-sw-worse',tx('red: worse than the gray line','红色：比灰线差'),tx('Where the resumed run’s loss is above the gray line.','接着训的 loss 高于灰线的地方。'))+
      it('ckp-sw-ck',tx('square: checkpoint','方块：存档'),tx('A checkpoint is saved every 100 steps; the filled one is the one resumed from.','每 100 步存一次档；实心的是接着训用的那份。'));
  } else if(v==='waste'){
    const st=stripOf(), hrs=st.span/3600, asy=(mode==='challenge'?chS.mode:S.mode)==='async';
    h=it('ckp-sw-sync',tx('solid: synchronous','实线：同步'),tx('Wasted share when every save stops training until it is written.','每次存档都停下训练、直到写完时，浪费的比例。'))+
      it('ckp-sw-async',tx('dashed: asynchronous','虚线：异步'),tx('Wasted share when a save stops training only for the copy to CPU memory.','存档只在拷进 CPU 内存时停一下，浪费的比例。'))+
      it('ckp-sw-now',tx('ring: your interval','圆圈：你选的间隔'),tx('The interval and save mode now chosen.','现在选的间隔和存法。'))+
      it('ckp-sw-train',tx('strip: training','条带：训练'),st.fail?tx(`The strip shows ${hrs} hours, the average time between two failures.`,`条带画的是 ${hrs} 小时，也就是平均两次故障之间的时间。`):tx(`The strip shows ${hrs} hours. One 8-GPU machine fails only about every ${Math.round(CASES.c8.M/604800)} weeks, so there is no failure in it.`,`条带画的是 ${hrs} 小时。一台 8 卡机器平均约 ${Math.round(CASES.c8.M/604800)} 周才出一次故障，所以这段时间里没有故障。`))+
      (asy?it('ckp-sw-savea',tx('thin ticks: asynchronous saves','细刻度：异步存档'),tx('Each asynchronous save stops training only while the state is copied into CPU memory.','每次异步存档，训练只在把状态拷进 CPU 内存时停一下。'))
          :it('ckp-sw-save',tx('thick ticks: synchronous saves','粗刻度：同步存档'),tx('Each synchronous save stops training until the whole checkpoint is written.','每次同步存档，训练都停下，等存档全部写完。')))+
      (st.fail?it('ckp-sw-fail',tx('red line: the failure','红线：故障'),tx('The work since the last save is lost here; the restart follows.','上次存档以后的工作在这里丢了，之后是重启。')):'')+
      (mode==='challenge'&&chReveal?(()=>{const l=chLimit(), b=chBest(), tm=fnum(l.tmin);
        return it('ckp-sw-nofit',tx('pale: breaks the storage or rollback limit','淡色：达不到存储或回退的要求'),tx(`Intervals shorter than ${tm} minutes. 30 TB holds at most ${l.kf} checkpoints, and ${l.kf} checkpoints saved more often than every ${tm} minutes do not reach back 1 hour.`,`间隔短于 ${tm} 分钟的那一段。30 TB 最多放 ${l.kf} 份存档；存得比每 ${tm} 分钟一次还勤，${l.kf} 份就退不回 1 小时。`))+
          it('ckp-sw-best',tx('green dot: least within the limits','绿点：本题可行最省'),tx(`Of the plans that keep both limits, the one that wastes least saves ${b.mode==='async'?'asynchronously':'synchronously'} every ${minsT(b.T)} and keeps ${b.k}, wasting ${pct(b.w)}. The lowest point of the curve looks only at the time wasted, and it lies in the pale range.`,`满足这两条要求的方案里，浪费最少的是${b.mode==='async'?'异步':'同步'}存档、每 ${minsT(b.T)}存一次、留 ${b.k} 份，浪费 ${pct(b.w)}。曲线的最低点只看时间浪费，它落在淡色的那一段里。`));})():'');
  } else h='';
  if(lg._h!==h){lg.innerHTML=h; lg._h=h;}
}
function drawPlayer(){
  const pl=$('#ckp-player',app); if(!pl) return;
  const show=animated()&&mode!=='challenge'&&!hidden0();
  pl.hidden=!show; if(!show) return;
  if(!pl._b){pl._b=1; pl.innerHTML=`<button type="button" class="btn ghost sm" data-act="play"></button>`+A.ui.seg({id:'ckp-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd,inline:false})+
    `<input type="range" class="ckp-scrub" data-key="scrub" min="0" max="1000" value="${Math.round(playT*1000)}" aria-label="${esc(tx('Position in the replay','回放进度'))}"><span class="ckp-phase small mono" id="ckp-phase"></span>`;}
  const label=playing?tx('❚❚ Pause','❚❚ 暂停'):(playT>=1?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  const b=pl.querySelector('[data-act="play"]'); if(b.textContent!==label) b.textContent=label;
  $$('.seg-ctl[data-key="spd"]',pl).forEach(gx=>A.ui.setPressed(gx,spd));
  const sc=pl.querySelector('[data-key="scrub"]'); const v=Math.round(playT*1000); if(+sc.value!==v) sc.value=v;
  const ph=$('#ckp-phase',pl), t=phaseText(); if(ph.textContent!==t) ph.textContent=t;
}
function phaseText(){const se=lossSeries(), p=phases(se);
  if(from0&&playT<p.a) return tx(`step ${Math.round(se.cut*playT/p.a)}`,`第 ${Math.round(se.cut*playT/p.a)} 步`);
  if(from0&&playT<p.p) return se.crash?tx('crash at step 450','第 450 步崩了'):tx('the spike','突刺');
  const u=from0?(playT-p.p)/(1-p.p):playT; return tx(`resumed, step ${se.from+Math.round((E.STEPS-se.from)*Math.min(1,u))}`,`接着训，第 ${se.from+Math.round((E.STEPS-se.from)*Math.min(1,u))} 步`);}
function statusOf(){
  if(mode==='challenge'){const e=chEval(chS.mode,chS.T,chS.k); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),tip:tx(`Green when the plan fits in 30 TB, reaches back 1 hour and wastes at most 12%. Now: ${pct(e.w)} wasted.`,`方案放得下 30 TB、能退回 1 小时、浪费不超过 12% 时，徽章是绿色。现在浪费 ${pct(e.w)}。`)};}
  if(hidden0()) return {cls:'neutral',txt:tx('Predict first','先预测'),tip:tx('The resumed run is hidden until you answer in part 1.','接着训的曲线先隐藏，等你在第 1 部分作答。')};
  const v=S.view;
  if(v==='files') return S.atom==='plain'?{cls:'bad',txt:tx('Checkpoint lost','存档坏了'),tip:tx('The only checkpoint was emptied when the overwrite began, and the new one is only 60% written.','覆盖一开始，唯一的存档就清空了，新的又只写了 60%。')}:{cls:'ok',txt:tx('Resumes from 400','从 400 接着训'),tip:tx('The previous complete checkpoint survived.','上一份完整的存档还在。')};
  if(v==='waste'){const c=CASES[S.cs], w=waste(dOf(S.cs,S.mode),S.T*60,c.M,c.R).total; return {cls:'neutral',txt:tx(`${pct(w)} wasted`,`浪费 ${pct(w)}`),tip:tx('Share of the run lost to saving, redoing work and restarts.','整次训练里花在存档、重算和重启上的比例。')};}
  if(playing||playT<1) return {cls:'neutral',txt:tx('Playing','播放中'),tip:tx('The run is playing.','正在播放这次训练。')};
  if(v==='spike'){const r=remedy(S.fix), ok=S.fix==='skip'; return {cls:ok?'ok':'bad',txt:ok?tx('No spike','没有突刺'):tx('Spike','突刺'),tip:tx(`Loss at step 600: ${lossF(r.final)}.`,`第 600 步的 loss：${lossF(r.final)}。`)};}
  const r=resumed(keepOf(S)); return r.same?{cls:'ok',txt:tx('Exact resume','原样接上'),tip:tx('The resumed run equals the uninterrupted one at every step.','接着训的每一步都和没崩时相同。')}:{cls:'bad',txt:tx('Not the same run','不是同一次训练'),tip:tx('The resumed run differs from the uninterrupted one.','接着训和没崩时不一样。')};
}
function drawHead(){
  const t=$('#ckp-st-title',app), sub=$('#ckp-st-sub',app), st=$('#ckp-status',app), v=stageView();
  t.textContent=v==='challenge'?tx('How much of the 405B run is wasted?','405B 的训练有多少时间白费？'):v==='loss'?tx('Resumed from step 400: the same run as without the crash?','从第 400 步接着训，还和没崩时一样吗？'):v==='spike'?tx('After a loss spike: where to resume from?','loss 突刺之后，从哪里接着训？'):v==='files'?tx('A crash during the save: which checkpoint survives?','存档时崩了，哪份存档还能用？'):tx('How often should the run save?','多久存一次最省？');
  sub.textContent=v==='challenge'?tx(`Llama 405B, failures every 3 h, ${chS.mode==='sync'?'synchronous':'asynchronous'}, every ${minsT(chS.T)}, keep ${chS.k}`,`405B，每 3 小时一次故障，${chS.mode==='sync'?'同步':'异步'}，每 ${minsT(chS.T)}一次，留 ${chS.k} 份`):
    v==='waste'?(S.cs==='c70'?tx(`70B on 1,024 GPUs, ${S.mode==='sync'?'synchronous':'asynchronous'}, every ${minsT(S.T)}`,`70B，1,024 张卡，${S.mode==='sync'?'同步':'异步'}，每 ${minsT(S.T)}一次`):tx(`8B on 8 GPUs, ${S.mode==='sync'?'synchronous':'asynchronous'}, every ${minsT(S.T)}`,`8B，8 张卡，${S.mode==='sync'?'同步':'异步'}，每 ${minsT(S.T)}一次`)):
    v==='files'?L(ATOM[S.atom].t):v==='spike'?L(FIX[S.fix].t):tx('The small model of this lab, 600 steps','这个 Lab 的小模型，共 600 步');
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function metricsDef(){
  const v=stageView(), hide=hidden0();
  if(v==='challenge'){const e=chEval(chS.mode,chS.T,chS.k);
    return [{k:'w',label:tx('Time wasted','浪费的时间'),v:pct(e.w),sub:tx('of the whole run','占整次训练'),tip:tx(`Saving ${pct(e.parts.save)} + redone work ${pct(e.parts.lost)} + restarts ${pct(e.parts.restart)}.`,`存档 ${pct(e.parts.save)} + 重算 ${pct(e.parts.lost)} + 重启 ${pct(e.parts.restart)}。`)},
      {k:'st',label:tx('Storage','存储'),v:TB(e.store),sub:tx('of 30 TB','共 30 TB'),tip:tx(`${e.k} × ${TB(CH.size)}.`,`${e.k} × ${TB(CH.size)}。`)},
      {k:'bk',label:tx('Oldest checkpoint','最老的存档'),v:minsT(e.back),sub:tx('old, right after a save','之前的，刚存完时'),tip:tx(`(${e.k} − 1) × ${minsT(e.T)}.`,`(${e.k} − 1) × ${minsT(e.T)}。`)}];}
  if(v==='waste'){const c=CASES[S.cs], d=dOf(S.cs,S.mode), w=waste(d,S.T*60,c.M,c.R), to=tOpt(d,c.M), dx=S.mode==='sync'?dShown(c.size,WINT.bw):d;
    return [{k:'w',label:tx('Time wasted','浪费的时间'),v:pct(w.total,S.cs==='c8'?2:1),sub:tx('of the whole run','占整次训练'),tip:tx(`Saving ${termPct(w.save,S.cs==='c8')} + redone work ${termPct(w.lost,S.cs==='c8')} + restarts ${termPct(w.restart,S.cs==='c8')}.`,`存档 ${termPct(w.save,S.cs==='c8')} + 重算 ${termPct(w.lost,S.cs==='c8')} + 重启 ${termPct(w.restart,S.cs==='c8')}。`)},
      {k:'d',label:tx('Training stops per save','每次存档停多久'),v:tx(`${fixd(d,0)} s`,`${fixd(d,0)} 秒`),sub:S.mode==='sync'?tx(`${sizeCard(c.size)} at 10 GB/s`,`${sizeCard(c.size)}，每秒 10 GB`):tx('copy to CPU memory','拷进 CPU 内存'),tip:S.mode==='sync'?tx(`${sizeX(c.size).t} ÷ 10 GB/s = ${fnum(dx)} s.`,`${sizeX(c.size).t} ÷ 每秒 10 GB = ${fnum(dx)} 秒。`):tx('Assumed: the time to copy the state into CPU memory.','假设：把状态拷进 CPU 内存的时间。')},
      {k:'to',label:tx('Best interval','最好的间隔'),v:mins(to),sub:'',subH:A.ui.math('T^{*} = \\sqrt{2 \\delta M}'),tip:A.ui.math(`T^{*} = \\sqrt{2 \\times ${fnum(dx)}\\ \\text{${tx('s','秒')}} \\times ${fint(c.M)}\\ \\text{${tx('s','秒')}}} = ${fint(Math.round(Math.sqrt(2*dx*c.M)))}\\ \\text{${tx('s','秒')}}`)}];}
  if(v==='files') return [{k:'rd',label:tx('Steps redone','重算的步数'),v:fint(redone(S.atom)),sub:S.atom==='plain'?tx('from step 0','从第 0 步起'):tx('from step 400','从第 400 步起'),tip:tx('Steps between the checkpoint the run resumes from and the crash at step 500.','接着训用的那份存档到第 500 步故障之间的步数。')},
    {k:'ok',label:tx('Usable checkpoint','能用的存档'),v:S.atom==='plain'?tx('none','没有'):tx('step 400','第 400 步'),sub:tx('after the crash','故障之后'),tip:tx('What the restarted job can load.','重启后能读回来的存档。')}];
  if(v==='spike'){const r=remedy(S.fix), b=base(), done=!playing&&playT>=1;
    return [{k:'f',label:tx('Loss at step 600','第 600 步的 loss'),v:done?lossF(r.final):'—',sub:tx(`without bad data ${lossF(b.ev[E.STEPS])}`,`没有坏数据时 ${lossF(b.ev[E.STEPS])}`),tip:tx('Validation loss at the end of the run.','训练结束时的验证集 loss。')},
      {k:'pk',label:tx('Highest loss in the spike','突刺的最高点'),v:lossF(r.peak),sub:tx(`before it ${lossF(r.pre)}`,`突刺之前 ${lossF(r.pre)}`),tip:tx('The highest validation loss after the bad batches.','坏数据之后验证集 loss 的最高点。')},
      {k:'ck',label:tx('Checkpoints needed','要留几份存档'),v:S.fix==='latest'?'1':'2',sub:S.fix==='latest'?tx('step 500','第 500 步'):tx('steps 400 and 500','第 400 和 500 步'),tip:tx('To roll back to step 400 you must still have it after step 500 is saved.','要退回第 400 步，存完第 500 步以后它还得在。')}];}
  const r=resumed(keepOf(S)), done=!playing&&playT>=1&&!hide;
  return [{k:'g',label:tx('Resumed vs no crash','接着训比没崩时'),v:!done?'—':r.same?tx('identical','完全相同'):tx(`up to +${fixd(r.maxRel*100,0)}%`,`最多高 ${fixd(r.maxRel*100,0)}%`),sub:tx('validation loss, steps 400–600','验证集 loss，第 400–600 步'),tip:hide?tx('Hidden until you answer in part 1.','在第 1 部分作答之前先隐藏。'):tx('The largest amount by which the resumed run’s loss exceeds the uninterrupted run’s, as a share of the uninterrupted run’s loss at that step.','接着训的 loss 比没崩时高出的最大值，按那一步没崩时的 loss 算百分比。')},
    {k:'dt',label:tx('Samples read twice / never','读了两遍 / 没读到的样本'),v:!done?'—':`${fint(r.twice)} / ${fint(r.never)}`,sub:tx('of 1,440 per epoch','每个 epoch 共 1,440 个'),tip:tx('Read twice within one epoch; never read in an epoch the uninterrupted run completed.','在同一个 epoch 里读了两遍的样本数；在没崩时读完的 epoch 里一遍都没读到的样本数。')},
    {k:'rd',label:tx('Steps redone','重算的步数'),v:fint(E.CRASH-E.CK),sub:tx('from step 400 to the crash at step 450','从第 400 步到第 450 步崩溃'),tip:tx('Work done after the checkpoint is lost and redone.','存档之后做的工作丢了，要重算。')}];
}
function drawMetrics(){
  const host=$('#ckp-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`); el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; {const sb=el.querySelector('.sub'); if(m.subH){if(sb._h!==m.subH){sb.innerHTML=m.subH; sb._h=m.subH;}} else {sb._h=null; sb.textContent=m.sub;}} el.dataset.tipx=m.tip;
    const d=el.querySelector('.d'), prev=shownPrev[m.k]; let dt='';
    if(userAct&&prev!=null&&prev!==m.v&&/%$/.test(m.v)&&/%$/.test(prev)&&m.k==='w'){const a=parseFloat(prev), b=parseFloat(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,{pts:true,dp:1})||'';}
    if(userAct){clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2600);} else {d.classList.remove('on'); d.textContent='';}}
    shownPrev[m.k]=m.v;});
}
function mathHTML(){
  if(hidden0()) return `<p>${tx('The formulas appear once you have answered part 1.','在第 1 部分作答之后，这里给出公式。')}</p>`;
  const rows=[];
  rows.push(`<p><b>${tx('AdamW, one step for every parameter','AdamW 对每个参数走一步')}</b>${tx(': m and v are running averages of the gradient g and of its square; the step divides one by the square root of the other.','：m 和 v 是梯度 g 和梯度平方的滑动平均，每一步拿 m 除以 v 的平方根。')}</p>`+MX('m \\leftarrow \\beta_1 m + (1-\\beta_1) g, \\quad v \\leftarrow \\beta_2 v + (1-\\beta_2) g^2')+MX('w \\leftarrow w - \\eta \\lambda w - \\eta \\frac{m / (1-\\beta_1^t)}{\\sqrt{v / (1-\\beta_2^t)} + \\epsilon}')+
    `<p>${tx(`Here ${A.ui.math('\\beta_1 = 0.9')}, ${A.ui.math('\\beta_2 = 0.999')} and ${A.ui.math('\\lambda = 0.01')}; η is the learning rate and t the number of steps taken. Right after ${A.ui.math('m = v = 0')} and ${A.ui.math('t = 0')}, the first step is ${A.ui.math('\\eta \\cdot \\text{sign}(g)')} for every parameter.`,`这里 ${A.ui.math('\\beta_1 = 0.9')}，${A.ui.math('\\beta_2 = 0.999')}，${A.ui.math('\\lambda = 0.01')}；η 是学习率，t 是已经走过的步数。${A.ui.math('m = v = 0')}、${A.ui.math('t = 0')} 之后的第一步，对每个参数都是 ${A.ui.math('\\eta \\cdot \\text{sign}(g)')}。`)}</p>`);
  rows.push(`<p><b>${tx('Learning rate at scheduler step s','调度器第 s 步的学习率')}</b>${tx(': it warms up for 30 steps, then falls along a cosine.','：先预热 30 步，再沿余弦曲线降下来。')}</p>`+MX('\\eta(s) = 0.05 \\cdot \\frac{s+1}{30} \\quad (s < 30)')+MX('\\eta(s) = 0.05 \\cdot (0.1 + 0.45 \\cdot (1 + \\cos \\frac{\\pi (s-30)}{570})) \\quad (s \\geq 30)')+
    `<p>${tx(`At step 400: ${A.ui.math(`\\eta(400) = ${fixd(lrAt(400),4)}`)}. A new scheduler at step 400 starts again from ${A.ui.math(`\\eta(0) = ${fixd(lrAt(0),4)}`)} and is at ${A.ui.math(`\\eta(199) = ${fixd(lrAt(199),4)}`)} when the run ends at step 600, against ${A.ui.math(`\\eta(599) = ${fixd(lrAt(599),4)}`)}.`,`第 400 步时 ${A.ui.math(`\\eta(400) = ${fixd(lrAt(400),4)}`)}。第 400 步新建的调度器从 ${A.ui.math(`\\eta(0) = ${fixd(lrAt(0),4)}`)} 重新开始，到第 600 步结束时是 ${A.ui.math(`\\eta(199) = ${fixd(lrAt(199),4)}`)}。没崩时是 ${A.ui.math(`\\eta(599) = ${fixd(lrAt(599),4)}`)}。`)}</p>`);
  const c=CASES.c70, sx=sizeX(SIZE70), dx=dShown(SIZE70,WINT.bw);
  rows.push(`<p><b>${tx('Wasted share and the best interval','浪费的比例和最好的间隔')}</b>${tx(': saving often wastes time on saves, saving rarely wastes redone work; the best interval T* balances the two.','：存得勤，时间花在存档上；存得少，时间花在重算上。最好的间隔 T* 让两者平衡。')}</p>`+MX('w(T) \\approx \\frac{\\delta}{T} + \\frac{T}{2M} + \\frac{R}{M}')+`<p>${tx('T is the interval between saves, δ how long one save stops training, M the mean time between failures and R the time to restart. The smallest w is at:','T 是两次存档的间隔，δ 是每次存档让训练停多久，M 是平均多久出一次故障，R 是重启要的时间。w 最小的地方是：')}</p>`+MX('T^{*} = \\sqrt{2 \\delta M}')+
    `<p>${T_('size',tx('Checkpoint','存档'))} = ${fint(P70)} × 14 ${tx('bytes','字节')} = ${sx.t}</p>`+
    `<p>${T_('delta','δ')} = ${sx.t} ÷ ${tx('10 GB/s','每秒 10 GB')} = ${fnum(dx)} ${tx('s','秒')}</p>`+
    `<p>${T_('M','M')} = 12 ${tx('h','小时')} = ${fint(c.M)} ${tx('s','秒')}</p>`+
    MX(`T^{*} = \\sqrt{2 \\times ${fnum(dx)} \\times ${fint(c.M)}} = ${fint(Math.round(Math.sqrt(2*dx*c.M)))}\\ \\text{${tx('s','秒')}}`));
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawBox();
  const v=stageView();
  if(v==='loss'||v==='spike') drawLoss(); else if(v==='files') drawFiles(); else drawWaste();
  drawTerms(); drawLegend(); drawPlayer(); drawMetrics();
  const mb=$('#ckp-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function frame(){if(!app) return; drawLoss(); drawPlayer(); drawMetrics(); const st=$('#ckp-status',app), s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the interval, the save mode and the checkpoints kept, in the task card. Everything else is fixed: Llama${NB}3.1${NB}405B, failures every 3 hours, restarts of 10 minutes, 50 GB/s, 30 TB.`,`挑战中只能在任务卡里改间隔、存法和留几份。其余设置固定：Llama${NB}3.1${NB}405B，每 3 小时一次故障，重启 10 分钟，每秒 50 GB，30 TB。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='interval') return tx('Fixed in this part: the 70B on 1,024 GPUs. Change the interval and the save mode in the card.','这一部分固定：1,024 张卡训练 70B。在卡片里改间隔和存法。');
  return tx('In this part you choose in the card. You can change every setting in Scenarios.','这一部分在卡片里选。在应用场景里可以改所有设置。');
}
function syncControls(){
  $$('input[data-key="T"]',app).forEach(i=>{const v=Math.max(0,INTS.indexOf(S.T)); if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=minsT(S.T); i.setAttribute('aria-valuetext',minsT(S.T));});
  $$('.seg-ctl[data-key="mode"]',app).forEach(g=>A.ui.setPressed(g,S.mode)); $$('.seg-ctl[data-key="cs"]',app).forEach(g=>A.ui.setPressed(g,S.cs)); $$('.seg-ctl[data-key="fix"]',app).forEach(g=>A.ui.setPressed(g,S.fix)); $$('.seg-ctl[data-key="view"]',app).forEach(g=>A.ui.setPressed(g,S.view==='files'?'loss':S.view));
  $$('#ckp-controls [data-part]',app).forEach(b=>b.setAttribute('aria-pressed',String(!!S[b.dataset.part])));
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#ckp-controls',app).classList.toggle('locked',ch);
  const v=S.view;
  hide('#ckp-ctl-view-wrap',g||ch); hide('#ckp-ctl-parts-wrap',g||ch||v!=='loss'); hide('#ckp-ctl-fix-wrap',g||ch||v!=='spike');
  hide('#ckp-ctl-cs-wrap',g||ch||v!=='waste'); hide('#ckp-ctl-mode-wrap',ch||v!=='waste'||(g&&!isP('interval'))); hide('#ckp-ctl-T-wrap',ch||v!=='waste'||(g&&!isP('interval'))); hide('#ckp-ctl-reset-wrap',g||ch);
  const ln=$('#ckp-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.txt+tx('. ','。')+s.tip);},400);}

/* =====================================================================
   replay: only when the picture would differ, and only the part that differs (R3.20)
   ===================================================================== */
function tick(t){
  raf=0; if(!app||!playing) return;
  const dt=lastT?t-lastT:16; lastT=t;
  const dur=from0?DUR[spd]:DUR[spd]*(E.STEPS-lossSeries().from)/(E.STEPS+40);
  playT=Math.min(1,playT+dt/dur);
  if(playT>=1){playing=false; playT=1; watched.add(sigOf(S)); wantObs=true; render(); wantObs=false; revealSoonIf(); return;}
  frame(); raf=requestAnimationFrame(tick);
}
function play(whole){if(playT>=1||whole){playT=0;} if(whole!=null) from0=!!whole; playing=true; lastT=0; if(A.reduceMotion()){finishPlay(); return;} if(!raf) raf=requestAnimationFrame(tick); frame();}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; if(app) frame();}
function finishPlay(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; playT=1; if(animated()) watched.add(sigOf(S)); wantObs=true; render(); revealSoon(); wantObs=false;}
/* after a change: play the resumed part if this curve has not been watched; a watched curve shows its end at once */
function maybeReplay(){
  if(!animated()||hidden0()){if(!playing) playT=1; return false;}
  const sg=sigOf(S);
  if(!watched.has(sg)){const first=!watched.size||(S.view==='spike'&&![...watched].some(w=>w.startsWith('sp:')));
    if(playing) pause(); from0=first; playT=0; play(first); return true;}
  if(playing) pause(); playT=1; return false;
}
function revealSoonIf(){if(mode==='guided'){wantObs=true; updateRail(); wantObs=false; revealSoon();}}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  const was=sigOf(S)+S.view;
  Object.assign(S,p);
  if(!INTS.includes(S.T)) S.T=180; ['o','s','d','r'].forEach(k=>{S[k]=S[k]?1:0;});
  if(!['loss','spike','waste','files'].includes(S.view)) S.view='loss';
  const changed=sigOf(S)+S.view!==was;
  if(changed) maybeReplay(); else if(!playing) playT=playT;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setParts(p){PARTS.forEach(k=>{if(p[k]!=null&&(!!p[k])!==(!!S[k])) partsTried.add(k);}); setState(p);}
function setAtom(a){atomTried.add(a); setState({atom:a});}
function setFix(f){fixTried.add(f); setState({fix:f});}
function setRem(k){if(!REMS.includes(k)) return; remView=k; if(playing) pause(); S={...remState(k)}; playT=1; from0=true; renderRail(); if(innerWidth<=760) A.ui.reveal($('#ckp-stage',app));}
function enterStep(i){
  wantObs=false; if(playing) pause(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#ckp-stage',app); if(st) st.scrollTop=0;
  if(!maybeReplay()) playT=1;
  renderRail();
}
function revealStep(){A.ui.reveal($('#ckp-rail .step.current',app)||$('#ckp-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* keepStage as in labs/llm-judge.js (R9.7): stop once the reader scrolls away from the rail's card */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#ckp-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ckp-rail .step.current',app)||$('#ckp-rail .railcard',app)||$('#ckp-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#ckp-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ckp-insight',app)); else revealBox($('#ckp-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; if(playing) pause();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; allOpen=false; chS={...CH.start};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#ckp-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  if(m==='sandbox'){if(!maybeReplay()) playT=1;} else playT=1;
  renderRail();
}
const VIEWS=['loss','spike','waste','files'], FIXES=['latest','back','skip'];
const token=()=>[VIEWS.indexOf(S.view),S.o,S.s,S.d,S.r,FIXES.indexOf(S.fix),S.mode==='async'?1:0,INTS.indexOf(S.T),S.cs==='c8'?1:0,S.atom==='atomic'?1:0].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==10||p.some(x=>!Number.isInteger(x))) return null;
  const [vi,o,s,d,r,fi,ma,ti,ci,ai]=p; if(vi<0||vi>3||fi<0||fi>2||ti<0||ti>=INTS.length||[o,s,d,r,ma,ci,ai].some(x=>x!==0&&x!==1)) return null;
  return {view:VIEWS[vi],o,s,d,r,fix:FIXES[fi],mode:ma?'async':'sync',T:INTS[ti],cs:ci?'c8':'c70',atom:ai?'atomic':'plain'};
}

/* =====================================================================
   interview drill: seeded, from the interval model
   ===================================================================== */
function drill(seed){let a=seed*7919+3; const r=()=>{a=(a*1103515245+12345)%2147483648; return a/2147483648;}, pk=x=>x[Math.floor(r()*x.length)];
  const P=pk([[P8,`Llama${NB}3.1${NB}8B`],[P70,`Llama${NB}3.1${NB}70B`],[P405,`Llama${NB}3.1${NB}405B`]]), bw=pk([10,20,50])*1e9, Mh=pk([3,6,12,24]);
  const size=P[0]*BPC, d=size/bw, M=Mh*3600, to=tOpt(d,M), w=waste(d,to,M,600);
  return {name:P[1],P:P[0],bw,Mh,size,d,to,w:w.total};}
function renderDrill(){
  if(!app) return; const q=drill(drillSeed), qd=$('#ckp-drill-q',app), a=$('#ckp-drill-a',app);
  qd.innerHTML=tx(`You train <b>${q.name}</b> (${big(q.P)} parameters) with AdamW in mixed precision. The job writes <b>${q.bw/1e9} GB/s</b>, saves synchronously, fails every <b>${q.Mh} hours</b> on average, and a restart takes 10 minutes. How large is a checkpoint, how often should you save, and how much of the run is wasted?`,
                  `你用 AdamW 混合精度训练 <b>${q.name}</b>（${big(q.P)} 个参数）。任务每秒写 <b>${q.bw/1e9} GB</b>，同步存档，平均每 <b>${q.Mh} 小时</b>出一次故障，重启一次 10 分钟。一份存档多大？多久存一次最好？整次训练浪费多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const sx=sizeX(q.size), dx=sx.v/q.bw, M=q.Mh*3600, Tx=Math.round(Math.sqrt(2*dx*M));
  a.innerHTML=tx(`<p><b>${sizeCard(q.size)} per checkpoint; save about every ${mins(q.to)}; about ${pct(q.w)} of the run wasted.</b></p><p>Size: ${fint(q.P)} × 14 bytes = ${sx.t} (BF16 weights 2, FP32 copy of the weights 4, Adam’s m and v 4 each).</p><p>δ = ${sx.t} ÷ ${q.bw/1e9} GB/s = ${fnum(dx)} s, and M = ${q.Mh} h = ${fint(M)} s.</p>`+MX(`T^{*} = \\sqrt{2 \\delta M} = \\sqrt{2 \\times ${fnum(dx)} \\times ${fint(M)}} = ${fint(Tx)}\\ \\text{s}`)+MX(`\\text{wasted} \\approx \\frac{\\delta}{T^{*}} + \\frac{T^{*}}{2M} + \\frac{R}{M} = ${pct(q.w).replace('%','\\%')}`),
    `<p><b>一份存档 ${sizeCard(q.size)}，大约每 ${mins(q.to)}存一次最好，整次训练约浪费 ${pct(q.w)}。</b></p><p>大小：${fint(q.P)} × 14 字节 = ${sx.t}（BF16 权重 2，FP32 权重副本 4，Adam 的 m 和 v 各 4）。</p><p>δ = ${sx.t} ÷ 每秒 ${q.bw/1e9} GB = ${fnum(dx)} 秒，M = ${q.Mh} 小时 = ${fint(M)} 秒。</p>`+MX(`T^{*} = \\sqrt{2 \\delta M} = \\sqrt{2 \\times ${fnum(dx)} \\times ${fint(M)}} = ${fint(Tx)}\\ \\text{秒}`)+MX(`\\text{浪费} \\approx \\frac{\\delta}{T^{*}} + \\frac{T^{*}}{2M} + \\frac{R}{M} = ${pct(q.w).replace('%','\\%')}`));
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const TERM={
 size:()=>tx(`One checkpoint of Llama 3.1 70B: ${fint(P70)} parameters × 14 bytes = ${TB(SIZE70)}.`,`Llama 3.1 70B 的一份存档：${fint(P70)} 个参数 × 14 字节 = ${TB(SIZE70)}。`),
 delta:()=>tx('δ, the time training stands still for one save: the checkpoint size divided by the assumed 10 GB/s for a synchronous save, 10 s for an asynchronous one.','δ 是每次存档训练停下的时间：同步存档是存档大小除以假设的每秒 10 GB，异步存档是 10 秒。'),
 M:()=>TIP.mtbf()
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='T') return (mode==='guided'&&!isP('interval'))||mode==='challenge'?tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。'):tx('Click to jump to “Save every”.','点击跳到“每隔多久存一次”。');
  return tx('Comes from the lab’s stated assumptions; nothing to change here.','来自这个 Lab 写明的假设，这里不能改。');
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#ckp-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#ckp-controls',app)); return;}
  if(k==='T'){const el=$('#ckp-ic-T',app)||$('#ckp-ctl-T',app); if(el&&el.getClientRects().length){A.ui.jumpToEl(el); return;}}
  if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S=mode==='sandbox'?{...PRESETS[0].s}:{...BASE}; playT=1; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chS.mode,chS.T,chS.k).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(playing) pause(); else play(playT>=1?true:undefined); A.tip.quiet(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ckp-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#ckp-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='T') bySettings(e.target,()=>{if(isP('interval')) tMoved=true; setState({T:INTS[v]});});
    else if(k==='scrub'){if(playing) pause(); playT=v/1000; if(playT>=1){watched.add(sigOf(S)); render(); return;} frame();}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&k!=='scrub'&&e.target.closest('#ckp-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ckp-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ckp-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.part){const k=d.part; bySettings(b,()=>setParts({[k]:S[k]?0:1})); return;}
    if(d.lever!=null&&mode==='guided'){const [k,x]=d.lever.split(':'); const f=()=>byLearner(()=>{if(k==='atom') setAtom(x); else if(k==='fix') setFix(x);});
      if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='mode') bySettings(b,()=>{if(d.v==='async'&&isP('interval')) asyncTried=true; setState({mode:d.v});});
      else if(k==='cs') bySettings(b,()=>setState({cs:d.v}));
      else if(k==='fix') bySettings(b,()=>setState({fix:d.v}));
      else if(k==='view') bySettings(b,()=>setState({view:d.v}));
      else if(k==='spd'){spd=d.v; lastT=0; if(app) frame();}
      else if(k==='chT') byLearner(()=>{chS.T=+d.v; render();});
      else if(k==='chm') byLearner(()=>{chS.mode=d.v; render();});
      else if(k==='chk') byLearner(()=>{chS.k=+d.v; render();});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; for(const k in shownPrev) shownPrev[k]=null; /* a scenario is another run: no ▲▼ against the last one (R6.6) */
      byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#ckp-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ckp-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
/* the lab state contract (round 3 #6): what the learner did here, and putting it back before the first mount after a reload */
const sOk=x=>{if(!x||typeof x!=='object') return null; const b=(v,a,c)=>v===a?0:(v===c?1:-1);
  return parse([VIEWS.indexOf(x.view),x.o,x.s,x.d,x.r,FIXES.indexOf(x.fix),b(x.mode,'sync','async'),INTS.indexOf(x.T),b(x.cs,'c70','c8'),b(x.atom,'plain','atomic')].join('_'));};
const setOf=(a,ok)=>Array.isArray(a)?a.filter(x=>ok.includes(x)):[];
function snap(){return {v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,partsTried:[...partsTried],atomTried:[...atomTried],
  tMoved,asyncTried,fixTried:[...fixTried],remView,sbTouched,chS:{...chS},chSubmitted,chReveal,allOpen,watched:[...watched],spd,drill:[drillSeed,drillOpen],
  passed:mode==='challenge'&&chSubmitted&&chEval(chS.mode,chS.T,chS.k).ok};}
function restoreState(o){if(!o||o.v!==1) return;
  mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
  step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
  const s=sOk(o.S); S=s?{...BASE,...s}:{...BASE};
  pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
  observed={}; if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
  const sv=mode==='challenge'?sOk(o.saved):null; saved=sv?{...BASE,...sv}:null;
  partsTried=new Set(setOf(o.partsTried,PARTS)); atomTried=new Set(['plain',...setOf(o.atomTried,['atomic'])]); fixTried=new Set(['latest',...setOf(o.fixTried,FIXES)]);
  tMoved=o.tMoved===true; asyncTried=o.asyncTried===true; remView=REMS.includes(o.remView)?o.remView:'weights'; sbTouched=o.sbTouched===true;
  const c=o.chS&&typeof o.chS==='object'?o.chS:{}; chS=['sync','async'].includes(c.mode)&&CH.ints.includes(c.T)&&CH.keeps.includes(c.k)?{mode:c.mode,T:c.T,k:c.k}:{...CH.start};
  chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; allOpen=chReveal&&o.allOpen===true;
  watched.clear(); setOf(o.watched,['sp:latest','sp:back','sp:skip',...Array.from({length:16},(_,i)=>'ls:'+i.toString(2).padStart(4,'0'))]).forEach(k=>watched.add(k));
  spd=Object.keys(DUR).includes(o.spd)?o.spd:'fast';
  if(Array.isArray(o.drill)){drillSeed=Number.isInteger(o.drill[0])&&o.drill[0]>=1&&o.drill[0]<1e6?o.drill[0]:1; drillOpen=o.drill[1]===true;}}

A.registerLab({
  id:ID, snap, restore:restoreState,
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...BASE,...p}; mode='sandbox'; sbTouched=true; playT=1;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#ckp-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ckp-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    render();
    if(!relang&&!playing&&animated()&&!hidden0()&&!watched.has(sigOf(S))) maybeReplay();
  },
  leave(){clearTimeout(annT); stageStop(); if(raf) cancelAnimationFrame(raf); raf=0; playing=false; root=null; app=null;},
  onLang(){if(root){const d=$('#ckp-math',app), open=d&&d.open, was=playing; if(raf) cancelAnimationFrame(raf); raf=0; playing=false; relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#ckp-math',app).open=true; if(was){playing=true; lastT=0; raf=requestAnimationFrame(tick);}}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#ckp-chart',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#ckp-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.txt; vis.innerHTML='';
      const v=stageView();
      if(hidden0()){read.innerHTML=esc(tx('Crash at step 450; the resumed run is shown after you predict','第 450 步崩了；接着训的结果等你预测后揭晓')); return;}
      if(v==='challenge'){const e=chEval(chS.mode,chS.T,chS.k); read.innerHTML=esc(tx(`405B plan: ${pct(e.w)} wasted, ${TB(e.store)} stored`,`405B 的方案：浪费 ${pct(e.w)}，存储 ${TB(e.store)}`)); return;}
      if(v==='waste'){const c=CASES[S.cs], w=waste(dOf(S.cs,S.mode),S.T*60,c.M,c.R).total; read.innerHTML=esc(tx(`Saving every ${minsT(S.T)}: ${pct(w)} of the run wasted`,`每 ${minsT(S.T)}存一次，整次训练浪费 ${pct(w)}`)); return;}
      if(v==='files'){read.innerHTML=esc(tx(`Crash during the save: ${redone(S.atom)} steps redone`,`存档时崩了：重算 ${redone(S.atom)} 步`)); return;}
      if(v==='spike'){const r=remedy(S.fix); read.innerHTML=esc(playT<1?tx('The run is playing','这次训练正在播放'):tx(`After the spike: loss ${lossF(r.final)} at step 600`,`突刺之后：第 600 步 loss ${lossF(r.final)}`)); return;}
      const r=resumed(keepOf(S)); read.innerHTML=esc(playT<1?tx('Resumed run playing','接着训的曲线正在播放'):r.same?tx('Resumed from 400: identical to the run without a crash','从第 400 步接着训：和没崩时完全相同'):tx(`Resumed from 400: up to ${fixd(r.maxRel*100,0)}% above the run without a crash`,`从第 400 步接着训：最多比没崩时高 ${fixd(r.maxRel*100,0)}%`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const ser=r=>JSON.parse(JSON.stringify(r));
window.__ckp=Object.freeze({
  E, base:()=>ser({ev:base().ev}), resumed:k=>ser(resumed(k)), remedy:o=>ser(remedy(o)), lrAt,
  waste, tOpt, dOf, chEval:(m,T,k)=>ser(chEval(m,T,k)), chAll:()=>ser(chAll()), chBest:()=>ser(chBest()), chLimit:()=>ser(chLimit()), chPass:()=>ser(chPass()), chOne:(m,T,k)=>ser(chOne(chEval(m,T,k))),
  /* G7: the single-change sentence the result shows for a plan, in the page's language, and its waste as the result shows it */
  chNextText:(m,T,k)=>{const x=document.createElement('div'); x.innerHTML=chSummaryHTML(chEval(m,T,k),chBest()); const p=x.querySelector('.ckp-next'); return p?p.textContent:'';}, chShown:(m,T,k)=>{const e=chEval(m,T,k); return {w:pct(e.w),ok:e.ok,fits:e.fits,rb:e.rb};}, wasteNow:()=>ser(wasteNow()), drill:s=>ser(drill(s)),
  consts:Object.freeze({SIZE70,SIZE8,SIZE405,WINT:{...WINT},CASES:ser(CASES),CH:ser(CH),INTS:INTS.slice(),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:ser(PRESETS.map(p=>({id:p.id,...p.s}))),DUR:{...DUR}}),
  state:()=>({...S,save:S.mode,mode,step,pred,observed:{...observed},playT,playing,spd,from0,watched:[...watched],chS:{...chS},chSubmitted,chReveal,remView,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return ser({transferA:e.transfer.map(q=>q.a),transfer:e.transfer.map(q=>({q:q.q,o:q.o,a:q.a,why:q.why})),explainA:e.explain.map(q=>q.a)});}
});
})();
