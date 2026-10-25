/* train-capstone (world training): How do you plan the memory, parallelism and time of a 70B run?
   Fidelity: Analytical. Every number comes from plan() below; tests/train-capstone_numbers.py recomputes it independently.
   - compute: C = 6·N·D; a step does k·N FLOPs per token (k = 6, or 8 with full recomputation) at an assumed 50% of the H100's
     989.5 TFLOP/s while the GPU computes; the micro-batch is one 4,096-token sequence unless Settings say otherwise;
   - memory per GPU: 18 bytes per parameter ÷ (t·p), the ZeRO stage over the d data-parallel copies (ZeRO-1 6 + 12/d, ZeRO-2 2 + 16/d,
     ZeRO-3 18/d); ZeRO-3 also holds the BF16 weights of two gathered layers, the one being computed and the next one gathered ahead
     (2 × 2N/(L·t), as in the parallelism lab), kept apart from the model states; activations s·b·h·(18 + 4·I/h) ÷ t per layer and
     micro-batch (Megatron-LM), min(m, p) micro-batches in flight (1F1B);
     with recomputation each layer keeps its input (2·s·b·h ÷ t) and one layer is rebuilt in full; GB = 10⁹ bytes, 80 GB per H100;
   - step: (m + p − 1) × (compute + tensor-parallel all-reduces) + the data-parallel transfer that no compute covers;
     tensor parallel: 4 all-reduces per layer and micro-batch (6 with recomputation), 2(t − 1)/t × bytes ÷ 450 GB/s (NVLink);
     data parallel in BF16, counted in passes (an all-gather or a reduce-scatter; an all-reduce is two): ZeRO-0/1 2 passes once per
     step, hidden only by the last micro-batch's backward pass; ZeRO-2 m + 1, hidden by every backward pass; ZeRO-3 3m, hidden by all
     compute. One pass (integrator decision 8, the scaling book's model): inside one machine (t·d ≤ 8) (d − 1)/d × 2N/(t·p) ÷ 450 GB/s;
     across machines 2N/p ÷ 400 GB/s, the stage's whole array through one machine's network (8 × 400 Gb/s), or ÷ 12.5 GB/s on Ethernet;
   - failures: a machine fails once every 64 days on average (OPT-175B's logbook: ~2 of 128 machines a day); job MTBF M = 64 days ÷
     machines; a checkpoint is 14 bytes per parameter written at 10 GB/s; an asynchronous save stops training for 10 s; a restart takes
     10 minutes (integrator decision 11, shared with the checkpointing lab); wasted share ≈ δ/τ + (τ/2 + R)/M for the interval τ;
     days = steps × step ÷ (1 − waste).
   window.__tcap is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='train-capstone';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
/* nv: NVLink per GPU; node: one 8-GPU machine's InfiniBand (8 × 400 Gb/s); eth: one machine's 100 Gb/s Ethernet */
const GPU={peak:989.5e12,mem:80e9,nv:450e9,node:400e9,eth:100e9/8,perNode:8};
const U0=0.5, SEQ=4096, VOCAB=128256;
const CK={bpp:14,write:10e9,asyncS:10,restartH:10/60};
const MD={
 '8b':{id:'8b',name:'Llama‑3.1‑8B',N:8030261248,L:32,h:4096,I:14336},
 '70b':{id:'70b',name:'Llama‑3.1‑70B',N:70553706496,L:80,h:8192,I:28672},
 '405b':{id:'405b',name:'Llama‑3.1‑405B',N:405853388800,L:126,h:16384,I:53248}
};
const BASE={model:'70b',D:15e12,G:1024,t:8,p:8,z:1,rc:0,b:1,B:1024,net:'ib',ivl:3,asy:0,node:64};
function plan(c){
  const md=MD[c.model], N=md.N, Lr=md.L, h=md.h, I=md.I;
  const tp=c.t*c.p; if(c.G%tp) return null; const d=c.G/tp;
  if(c.B%(d*c.b)) return null; const m=c.B/(d*c.b);
  const pg=N/tp;
  /* zx: ZeRO-3 holds the BF16 weights of two layers (2 bytes × N/L ÷ t each), the one being computed and the next one gathered ahead;
     they are not model states, so states below leaves them out (the metric card, the memory column and Show the math list them apart) */
  const w=2*pg*(c.z===3?1/d:1), g=4*pg*(c.z>=2?1/d:1), o=12*pg*(c.z>=1?1/d:1), zx=c.z===3?4*N/(Lr*c.t):0;
  const tok=c.b*SEQ, full=tok*h*(18+4*I/h)/c.t, keep=2*tok*h/c.t, Ls=Lr/c.p, inf=Math.min(m,c.p);
  const act=c.rc?inf*Ls*keep+full:inf*Ls*full;
  const mem=w+g+o+zx+act;
  const k=c.rc?8:6;
  const cmb=k*(N/c.p)*tok/c.t/(GPU.peak*U0);
  const arB=tok*h*2, ar=c.t>1?2*(c.t-1)/c.t*arB/GPU.nv:0, nar=c.rc?6:4, tpmb=nar*Ls*ar;
  const per=cmb+tpmb, bk=per*(c.rc?3/4:2/3);
  /* one data-parallel pass: across machines, the machine sends the stage's whole BF16 array (all t tensor-parallel slices) through
     its network; inside one machine, each GPU moves (d − 1)/d of its own slice over NVLink */
  const cross=c.t*d>GPU.perNode, Bx=cross?2*N/c.p:2*pg, bw=cross?(c.net==='eth'?GPU.eth:GPU.node):GPU.nv, one=cross?Bx/bw:(d-1)/d*Bx/bw;
  let np,win; if(c.z<=1){np=2; win=bk;} else if(c.z===2){np=m+1; win=m*bk;} else {np=3*m; win=m*per;}
  const tdp=d>1?np*one:0, dpx=Math.max(0,tdp-win);
  const step=(m+c.p-1)*per+dpx;
  const seg={useful:m*cmb*6/k,recomp:m*cmb*(k-6)/k,tp:m*tpmb,bubble:(c.p-1)*per,dp:dpx};
  const Btok=c.B*SEQ, mfu=6*N*Btok/(c.G*GPU.peak*step);
  const steps=c.D/Btok, train=steps*step;
  const ckB=CK.bpp*N, dSync=ckB/CK.write, dS=c.asy?CK.asyncS:dSync, dH=dS/3600;
  const machines=c.G/GPU.perNode, M=c.node*24/machines;
  const wSave=dH/c.ivl, wRedo=(c.ivl/2)/M, wRestart=CK.restartH/M, waste=wSave+wRedo+wRestart;
  const valid=waste<0.5, good=1-waste, wall=valid?train/good:NaN;
  return {md,N,d,m,pg,w,g,o,zx,states:w+g+o,act,full,keep,Ls,inf,mem,fits:mem<=GPU.mem,k,cmb,arB,ar,nar,tpmb,per,bk,cross,Bx,bw,one,np,win,tdp,dpx,step,seg,
    logits:SEQ*c.b*VOCAB*4/c.t,
    Btok,mfu,steps,train,ckB,dSync,dS,machines,M,wSave,wRedo,wRestart,waste,valid,good,wall,days:wall/86400,gpuh:c.G*wall/3600,
    C:6*N*c.D,eff:mfu*good,gpus:c.G};
}
/* part 2: the first estimate, total work ÷ (GPUs × peak × an assumed utilization) */
const simpleDays=(u,c)=>6*MD[c.model].N*c.D/(c.G*GPU.peak*u)/86400;
/* published GPU hours (Llama 3.1 model card) against 6ND ÷ peak */
const REAL=[{id:'8b',h:1.46e6},{id:'70b',h:7.0e6},{id:'405b',h:30.84e6}].map(r=>({...r,u:6*MD[r.id].N*15e12/(r.h*3600*GPU.peak)}));

/* the challenge: Llama‑3.1‑70B on 512 H100s (64 machines), 2T tokens, within 60 days, fewest GPU hours */
const CH={base:{model:'70b',D:2e12,G:512,b:1,B:1024,net:'ib',node:64},Ts:[1,2,4,8],Ps:[1,2,4,8],Zs:[0,1,2,3],RCs:[0,1],IVs:[0.25,1,3],ASs:[0,1],
  start:{t:1,p:1,z:0,rc:0,ivl:3,asy:0},days:60,tol:0.01};
const chCfg=o=>({...CH.base,t:o.t,p:o.p,z:o.z,rc:o.rc,ivl:o.ivl,asy:o.asy});
function chEval(o){const r=plan(chCfg(o)); if(!r) return null; return {...o,r,fits:r.fits,ok:r.fits&&r.valid&&r.days<=CH.days};}
let CHALL=null;
function chAll(){if(CHALL) return CHALL; const a=[];
  CH.Ts.forEach(t=>CH.Ps.forEach(p=>CH.Zs.forEach(z=>CH.RCs.forEach(rc=>CH.IVs.forEach(ivl=>CH.ASs.forEach(asy=>{const e=chEval({t,p,z,rc,ivl,asy}); if(e) a.push(e);}))))));
  return (CHALL=a);}
const chBest=()=>chAll().filter(e=>e.ok).sort((a,b)=>a.r.gpuh-b.r.gpuh)[0];
const chMinimal=e=>!!e&&e.ok&&e.r.gpuh<=chBest().r.gpuh*(1+CH.tol);

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=x=>(x<0.1&&x>0?fixd(x*100,1):fixd(x*100,x<0.2?1:1))+'%';
const pct0=x=>fixd(x*100,0)+'%';
const fGB=b=>{const g=b/1e9; return (g>=1000?fint(Math.round(g)):fixd(g,1))+' GB';};
const gb2=b=>{const g=b/1e9; return (g>=1000?fint(Math.round(g)):fixd(g,2))+' GB';};   /* for a sum the reader can check: 1.24 + 3.53 + 85.90 = 90.67 */
const fTB=b=>fixd(b/1e12,2)+' TB';
const fS=s=>(s>=100?fixd(s,0):(s>=10?fixd(s,1):fixd(s,2)))+' s';
const fDays=x=>!isFinite(x)?'—':(x>=100?fint(Math.round(x)):fixd(x,1));
const dayU=x=>tx(`${fDays(x)} days`,`${fDays(x)} 天`);
const big=n=>A.fmt.big?A.fmt.big(n):fint(n);
const fGpuh=x=>!isFinite(x)?'—':big(Math.round(x));
const fH=hrs=>hrs<1?tx(`${fint(Math.round(hrs*60))} min`,`${fint(Math.round(hrs*60))} 分钟`):tx(`${fixd(hrs,hrs<10?1:0)} h`,`${fixd(hrs,hrs<10?1:0)} 小时`);
const fIvl=v=>v===0.25?tx('15 min','15 分钟'):(v===1?tx('1 hour','1 小时'):tx(`${v} hours`,`${v} 小时`));
const SUP=n=>String(n).split('').map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('');
const sci=x=>{const e=Math.floor(Math.log10(x)); return `${fixd(x/10**e,2)} × 10${SUP(e)}`;};
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
/* every batch 8 lab is live: link them all by id (integrator decision 5) */
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const LQ=id=>`“${labLink(id)}”`;   /* Chinese quotes the lab's name */
const M_=t=>`<div class="tcap-fx">${A.ui.math(t)}</div>`;
/* a long formula that wraps (the owner's review of 2026-10-09: no Show the math row scrolls sideways): its name, “=”, then the right
   side in pieces split before an operator; on a narrow box the pieces wrap, each new line under the first piece, and on a phone the
   name goes on its own line above (train-capstone.css). {} gives a piece's leading operator the spacing of a binary operator. A piece
   given as a list is one term that moves to the next line whole, and breaks before its own “×” only when that line is too narrow. */
const MWp=(p,first)=>`<span>${A.ui.math(first?p:`{} ${p}`)}</span>`;
const MW=(name,ps)=>`<div class="tcap-fx tcap-w"><span class="tcap-wl">${A.ui.math(name)}</span><span class="tcap-wo">${A.ui.math('=')}</span><span class="tcap-wr">${ps.map((p,i)=>Array.isArray(p)?`<span class="tcap-wg">${p.map((q,j)=>MWp(q,!i&&!j)).join('')}</span>`:MWp(p,!i)).join('')}</span></div>`;
/* H2 (the owner, 2026-10-09): the reader's numbers on their own typeset line under the formula, the result in bold after it */
const MV=(t,res)=>`<div class="tcap-math-line tcap-sub">${A.ui.math(t)}${res?` <span class="tcap-res">= <b>${res}</b></span>`:''}</div>`;
const sciT=x=>{const e=Math.floor(Math.log10(x)); return `${fixd(x/10**e,2)} \\times 10^{${e}}`;};
const LEAD=(name,why)=>`<p class="tcap-lead"><b>${name}</b>${why}</p>`;
const sig=(v,n)=>{if(!isFinite(v)||v===0) return '0'; const e=Math.floor(Math.log10(Math.abs(v))); return fixd(v,Math.max(0,n-1-e));};   /* n significant digits, for Show the math */
/* a non-breaking hyphen keeps “ZeRO-2” on one line in the narrow chart rows */
const zName=z=>z===0?tx('no ZeRO','模型状态不分（不用 ZeRO）'):`ZeRO\u2011${z}`;
const layoutName=c=>{const parts=[]; if(c.t>1) parts.push(tx(`TP ${c.t}`,`张量并行 ${c.t} 份`)); if(c.p>1) parts.push(tx(`PP ${c.p}`,`流水线 ${c.p} 段`));
  const d=c.G/(c.t*c.p); parts.push(tx(`DP ${fint(d)}`,`数据并行 ${fint(d)} 份`)); return parts.join(' × ')+tx(` with ${zName(c.z)}`,`，${zName(c.z)}`);};
const everyIvl=v=>v===1?tx('every hour','每小时'):tx(`every ${fIvl(v)}`,`每 ${fIvl(v)}`);
const saveName=c=>tx(`save ${everyIvl(c.ivl)}, ${c.asy?'async':'sync'}`,`${everyIvl(c.ivl)}${c.asy?'异步':'同步'}存一次档`);

/* =====================================================================
   module state
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let u2=0.4, tried2=new Set([0.4]), lv3='none', tried3=new Set(['none']), lv4='start', tried4=new Set(['start']), lv5='3s', tried5=new Set(['3s']);
let ch={...CH.start}, saved=null, chSubmitted=false, chReveal=false, allOpen=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=1, drillOpen=false;
const shownPrev={};
const SRC={
 card:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/MODEL_CARD.md',
 book:'https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/training.md',
 bookGpu:'https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/gpus.md',
 megReadme:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/README.md',
 megDist:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/docs/source/api-guide/dist_optimizer.md',
 megMem:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/theoretical_memory_usage.py',
 megFlops:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/training.py',
 bridge:'https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ec4ec358c3622134905144ca193516b194feddd5/docs/performance-summary-archive.md',
 titan:'https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/benchmarks/llama3_h100_202412_torchtitan.md',
 mlePerf:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/training/performance/README.md',
 mleAcc:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/compute/accelerator/README.md',
 mleFt:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/training/fault-tolerance/README.md',
 opt:'https://github.com/facebookresearch/metaseq/blob/f7ffa5fd61cf90f498a36d365c13dd7f1a912ff7/projects/OPT/chronicles/final_update.md',
 bloom:'https://github.com/bigscience-workshop/bigscience/blob/9e64edec2ad1fb9263bf39ba616a84ff64b4d947/train/tr11-176B-ml/README.md',
 bloomBlog:'https://github.com/huggingface/blog/blob/998ee73e0594a21b42df14632263266a2831dafe/bloom-megatron-deepspeed.md',
 nccl:'https://github.com/NVIDIA/nccl-tests/blob/a3d4589e8f32f8363c334c52eedad75b44ed8b2d/doc/PERFORMANCE.md',
 foundry:'https://github.com/mosaicml/llm-foundry/blob/0cdb2f42962fcf861e58b236e9bd1c9964ba99a4/scripts/train/benchmarking/README.md',
 dcpAsync:'https://github.com/pytorch/tutorials/blob/11512db7cbbcc4b530ecfc63205d3fed13fa190d/recipes_source/distributed_async_checkpoint_recipe.rst',
 ttcfg:'https://github.com/pytorch/torchtitan/blob/main/torchtitan/config/configs.py',
 megarg:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/arguments.py',
 p1910:'https://arxiv.org/abs/1910.02054', p2104:'https://arxiv.org/abs/2104.04473', p2407:'https://arxiv.org/abs/2407.21783', p2205:'https://arxiv.org/abs/2205.05198'
};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 states:()=>tx('Model states are what training keeps for every parameter, 18 bytes in all: the BF16 weight (2 bytes), its FP32 gradient (4) and the optimizer state (12). The optimizer state is the FP32 copy of the weight and Adam’s m and v, 4 bytes each.',
               '模型状态是训练时每个参数都要存的东西，一共 18 字节：BF16 权重 2 字节，FP32 梯度 4 字节，优化器状态 12 字节。优化器状态由 FP32 权重副本、Adam 的 m 和 v 组成，各 4 字节。'),
 act:()=>tx('Activations are the intermediate results each layer computes in the forward pass and keeps until the backward pass has used them. This lab uses Megatron-LM’s per-layer formula with sequence parallelism. That formula already assumes attention’s scores are recomputed in the backward pass, which is called selective recomputation. “Every layer” under Recomputation is full recomputation.',
            '激活是前向时每一层算出的中间结果，要一直留到反向用完。这个 Lab 用 Megatron-LM 带序列并行的每层公式。这个公式已经假设 attention 的分数在反向时重算，这叫部分重算。设置里的“每层都重算”是整层重算。'),
 mb:()=>tx('A micro-batch is one piece of a step’s batch; here it is one sequence of 4,096 tokens. The gradients of all the micro-batches are added up before the weights are updated once.','micro-batch 是把一步的数据切开后的一小份，这里是一条 4,096 个 token 的序列。所有 micro-batch 的梯度加起来以后，权重才更新一次。'),
 mfu:()=>tx('MFU (model FLOPs utilization): the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time. Time spent waiting for transfers or idling in the pipeline counts. Work done twice, such as a recomputed forward pass, does not count as the model’s own.',
            'MFU（model FLOPs utilization）：模型真正需要的运算（每个 token 6 × 参数量次）占同一段时间里 GPU 峰值算力的比例。等通信、流水线空转的时间都算在里面。重复做的运算（比如重算的那遍前向）不算模型本身的运算。'),
 flop:()=>tx('A FLOP is one floating-point operation, a multiply or an add. TFLOP/s is 10¹² of them per second.','FLOP 是一次浮点运算，一次乘法或一次加法。TFLOP/s 是每秒 10¹² 次。'),
 peak:()=>tx('989.5 TFLOP/s is the H100 SXM’s dense BF16 peak, the same figure as the lab Roofline and picotron’s MFU code. ml-engineering measured the best plain BF16 matrix multiply on an H100 at about 80% of it.','989.5 TFLOP/s 是 H100 SXM 稠密 BF16 的峰值，和 Roofline 那个 Lab、picotron 算 MFU 的代码用的一样。ml-engineering 实测，H100 上最好的 BF16 矩阵乘法约是它的 80%。'),
 bubble:()=>tx('The bubble is the time a pipeline’s GPUs sit idle in a step: at the start they wait for the first micro-batch, at the end for the last gradients. It is '+A.ui.math('\\frac{p - 1}{m + p - 1}')+' of the step for p stages and m micro-batches.','bubble 是一步里流水线上的卡空等的时间：开头等第一个 micro-batch 传过来，结尾等最后的梯度传回来。p 段、m 个 micro-batch 时，它占一步的 '+A.ui.math('\\frac{p - 1}{m + p - 1}')+'。'),
 allreduce:()=>tx('An all-reduce adds up a tensor held by each of several GPUs and gives every GPU the sum. In the ring algorithm each GPU sends and receives '+A.ui.math('\\frac{2(n - 1)}{n}')+' times the tensor’s size (nccl-tests).','all-reduce 把几张卡上各一份的张量加起来，再让每张卡都拿到这个和。用环形算法时，每张卡要收发张量大小的 '+A.ui.math('\\frac{2(n - 1)}{n}')+' 倍（nccl-tests）。'),
 nvlink:()=>tx('Inside one H100 machine the 8 GPUs talk over NVLink, 450 GB/s each way per GPU. Between machines they go through InfiniBand, and a machine’s 8 GPUs share 400 GB/s, 8 links of 400 Gb/s (the scaling book, gpus.md).','一台 H100 机器里的 8 张卡走 NVLink，每张卡每个方向 450 GB/s。机器之间走 InfiniBand，一台机器的 8 张卡共用 400 GB/s 的网络带宽，也就是 8 条 400 Gb/s 的链路（scaling book 的 gpus.md）。'),
 recompute:()=>tx('Recomputation (activation checkpointing): the forward pass keeps only each layer’s input, and the backward pass runs the layer’s forward again to get the rest. It saves most of the activations and costs one more forward pass: 8 FLOPs per parameter per token instead of 6.','重算（activation checkpointing）：前向时每层只存它的输入，反向用到时再把这一层的前向算一遍，拿回其余的中间结果。这样省掉大部分激活，代价是多算一遍前向：每个 token、每个参数做 8 次运算，而不是 6 次。'),
 ckpt:()=>tx('A checkpoint is a saved copy of everything training needs to resume. Here it holds 14 bytes per parameter: the BF16 weight (2) and the optimizer state (12: the FP32 copy and Adam’s m and v). BLOOM-176B counts the same 14. Gradients are not saved.','检查点（存档）是接着训需要的全部东西的一份拷贝。这里每个参数 14 字节：BF16 权重 2 字节，优化器状态 12 字节（FP32 权重副本、Adam 的 m 和 v）。BLOOM-176B 也是这样数的。梯度不用存。'),
 mtbf:()=>tx('The mean time between failures of the whole job: if each machine fails once every 64 days on average and the job needs all 128 machines, something fails every 64 days ÷ 128 = 12 hours.','整个任务平均多久出一次故障：每台机器平均 64 天坏一次，任务要用全部 128 台，所以平均 64 天 ÷ 128 = 12 小时就有一台坏。'),
 gpuh:()=>tx('GPU hours are the number of GPUs times the hours they are held, idle or not. Clouds bill by them, and model cards report them.','GPU 小时是卡数乘以占用的小时数，不管卡在不在算。云厂商按它收钱，模型卡也用它报告训练用了多少。'),
 goodput:()=>tx('The share of time training is the share of the wall-clock time that goes to new training, after saving checkpoints, redoing work lost in failures and restarting.','往前训练的时间占比，是去掉存档、故障后重算和重启以后，真正在往前训练的时间占总时间的比例。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'lt1',t:{en:'Under a month',zh:'不到 1 个月'}},{v:'1to3',t:{en:'1 to 3 months',zh:'1 到 3 个月'}},{v:'3to12',t:{en:'3 to 12 months',zh:'3 到 12 个月'}},{v:'gt12',t:{en:'Over a year',zh:'1 年以上'}}];
const PRED_RIGHT='3to12';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
/* a part already finished keeps the learner's choices when they come back to it (R3.20: nothing resets or replays) */
const done=id=>!!observed[STEPS.findIndex(s=>s.id===id)];
const MISSION={...BASE};
const LV2=[{v:0.4,t:{en:'40% (initial setup)',zh:'四成（初始设置）'},d:{en:'What the question assumed',zh:'题目里假设的'}},
  {v:0.3,t:{en:'30%',zh:'三成'},d:{en:'A run that waits more',zh:'等得更多的训练'}},
  {v:0.5,t:{en:'50%',zh:'五成'},d:{en:'A run that wastes less',zh:'浪费更少的训练'}}];
const LV3=[{v:'none',t:{en:'No split (initial setup)',zh:'不切（初始设置）'},d:{en:'Every GPU holds the whole model and runs 1 of the 1,024 sequences',zh:'每张卡存一整份模型，各算 1,024 条序列中的 1 条'},s:{t:1,p:1,z:0}},
  {v:'z3',t:{en:'ZeRO-3 alone',zh:'只用 ZeRO-3'},d:{en:'1,024 data-parallel copies split all the model states',zh:'1,024 份数据并行，把模型状态全部分开存'},s:{t:1,p:1,z:3}},
  {v:'tp8z3',t:{en:'TP 8 + ZeRO-3',zh:'张量并行 8 份 + ZeRO-3'},d:{en:'Each machine’s 8 GPUs split every layer; 128 data-parallel copies split the states',zh:'每台机器的 8 张卡切开每一层；128 份数据并行分开存模型状态'},s:{t:8,p:1,z:3}},
  {v:'tp8pp8z1',t:{en:'TP 8 × PP 8 + ZeRO-1',zh:'张量并行 8 份 × 流水线 8 段 + ZeRO-1'},d:{en:'8 machines form a pipeline of 10 layers each; 16 such pipelines split the optimizer states',zh:'8 台机器连成一条流水线，每段 10 层；16 条流水线分开存优化器状态'},s:{t:8,p:8,z:1}}];
const LV4=[{v:'start',t:{en:'TP 8 × PP 8 + ZeRO-1 (initial setup)',zh:'张量并行 8 份 × 流水线 8 段 + ZeRO-1（初始设置）'},d:{en:'A layout that fits, without recomputation',zh:'放得下的切法，不重算'},s:{t:8,p:8,z:1,rc:0}},
  {v:'rc',t:{en:'The same, with recomputation',zh:'同样的切法，加上重算'},d:{en:'Every layer recomputes its forward pass in the backward pass',zh:'每一层在反向时把前向再算一遍'},s:{t:8,p:8,z:1,rc:1}},
  {v:'tp4pp16',t:{en:'TP 4 × PP 16 + ZeRO-1',zh:'张量并行 4 份 × 流水线 16 段 + ZeRO-1'},d:{en:'Half the tensor parallelism, twice the stages',zh:'张量并行少一半，流水线多一倍'},s:{t:4,p:16,z:1,rc:0}},
  {v:'tp8z3',t:{en:'TP 8 + ZeRO-3, no pipeline',zh:'张量并行 8 份 + ZeRO-3，不分段'},d:{en:'128 data-parallel copies gather each layer’s weights for every micro-batch',zh:'128 份数据并行，每个 micro-batch 都要收集各层的权重'},s:{t:8,p:1,z:3,rc:0}},
  {v:'tp8z1',t:{en:'TP 8 + ZeRO-1, no pipeline',zh:'张量并行 8 份 + ZeRO-1，不分段'},d:{en:'No pipeline and so no bubble; each GPU keeps more of the model states',zh:'没有流水线，也就没有 bubble；每张卡存的模型状态更多'},s:{t:8,p:1,z:1,rc:0}}];
const LV5=[{v:'3s',t:{en:'Every 3 hours, sync (initial setup)',zh:'每 3 小时同步存一次（初始设置）'},d:{en:'Close to BLOOM-176B’s interval',zh:'和 BLOOM-176B 的间隔差不多'},s:{ivl:3,asy:0}},
  {v:'1s',t:{en:'Every hour, sync',zh:'每小时同步存一次'},d:{en:'Three times as many saves',zh:'存档次数变成 3 倍'},s:{ivl:1,asy:0}},
  {v:'15s',t:{en:'Every 15 minutes, sync',zh:'每 15 分钟同步存一次'},d:{en:'Twelve times as many saves',zh:'存档次数变成 12 倍'},s:{ivl:0.25,asy:0}},
  {v:'15a',t:{en:'Every 15 minutes, async',zh:'每 15 分钟异步存一次'},d:{en:'As often, but training stops only for the copy to CPU memory',zh:'一样勤，但训练只在拷到 CPU 内存时停一下'},s:{ivl:0.25,asy:1}}];
const lvOf=(arr,v)=>arr.find(o=>o.v===v);
const p3r=v=>plan({...MISSION,...lvOf(LV3,v).s});
const p4r=v=>plan({...MISSION,...lvOf(LV4,v).s});
const p5r=v=>plan({...MISSION,...lvOf(LV5,v).s});
const lv2Res=v=>dayU(simpleDays(v,MISSION));
const lv3Res=v=>{const r=p3r(v); return tx(`${fGB(r.mem)} per GPU, ${r.fits?'fits':'too big'}`,`每张卡 ${fGB(r.mem)}，${r.fits?'放得下':'放不下'}`);};
const lv4Res=v=>`MFU ${pct(p4r(v).mfu)}`;
const lv5Res=v=>tx(`${pct(p5r(v).waste)} wasted`,`白费 ${pct(p5r(v).waste)}`);
const levers=(id,aria,val,opts,tried,res)=>`<div class="tcap-lv" id="${id}">`+A.ui.levers({aria,value:String(val),options:opts.map(o=>({v:String(o.v),title:L(o.t),desc:L(o.d),result:esc(tried.has(o.v)||o.v===val?res(o.v):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},gate:()=>!!pred,skip:true,
  lead:()=>tx('1,024 H100s, Llama‑3.1‑70B, 15T tokens: how long will it take?','1,024 张 H100 用 15T 个 token 训练 70B：要训多久？'),
  enter:()=>({...MISSION}),
  body:()=>{const ch2=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You are to train Llama‑3.1‑70B on 1,024 H100s, in 128 machines of 8 GPUs. The model has about 70.6B parameters, and it must see 15T tokens.</p>`+
     `<p>Training takes about 6 operations per parameter per token, so the whole run is:</p>`+M_('C \\approx 6 \\times N \\times D')+
     `<p>N is the number of parameters and D the number of tokens. Each H100 peaks at 989.5 T${DEF('FLOP',TIP.flop())}/s. Averaged over a whole run, the GPUs use only about 40% of that peak.</p><p>About how long will the run take?</p>${ch2}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the answer.</p>':''}`,
     `<p>你要用 1,024 张 H100 训练 Llama‑3.1‑70B。这些卡装在 128 台机器里，每台 8 张。这个模型有约 70.6B 个参数，训练要用 15T 个 token。</p>`+
     `<p>训练时，每个 token 对每个参数大约要做 6 次运算，所以整次训练的运算量是：</p>`+M_('C \\approx 6 \\times N \\times D')+
     `<p>N 是参数量，D 是 token 数。每张 H100 的峰值是 989.5 TFLOP/s，也就是每秒 989.5 × 10¹² 次${DEF('浮点运算',TIP.flop())}。整次训练平均下来，GPU 大约只能用上峰值的四成。</p><p>大约要训多久？</p>${ch2}${pred?'<p class="muted small">已记下。第 2 部分给出答案之前，还可以改。</p>':''}`);}},
 {id:'total',title:{en:'The total: the work divided by what the GPUs really do',zh:'先算总账：总运算量除以每秒真正做到的运算'},expect:()=>tried2.size>1,showMe:()=>setLv2(0.3),label:{en:'Result',zh:'结果'},
  lead:()=>tx('The days follow from the total work and the share of the peak the GPUs really reach.','天数取决于总运算量，以及 GPU 真正用上了峰值的几成。'),
  enter:()=>{if(!done('total')){u2=0.4; tried2=new Set([0.4]);} return {...MISSION};},
  body:()=>tx(`<p>First the work. The forward pass does 2 operations per parameter per token, and the backward pass twice as many: 6 in all (the lab ${labLink('scaling-law')} explains why). So the whole run is:</p>`+M_('C = 6 \\times 70.6\\text{B} \\times 15\\text{T} \\approx 6.35 \\times 10^{24}\\ \\text{FLOP}')+
    `<p>Then the speed. The 1,024 GPUs’ ${DEF('peak',TIP.peak())} adds up, but in training the GPUs use only part of it. The share they use is the ${DEF('MFU',TIP.mfu())}: the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time. Time spent waiting for transfers or idling in the pipeline counts.</p>`+M_('\\text{training time} = \\frac{C}{G \\times \\text{peak} \\times \\text{MFU}}')+
    `<p>G is the number of GPUs, and each H100’s peak is 989.5 TFLOP/s. Pick an MFU and see how many days the run takes.</p>`+levers('tcap-ic-lv2',tx('MFU, pick one','MFU，选一个'),u2,LV2,tried2,lv2Res),
    `<p>先看要做多少运算。前向时，每个 token 对每个参数做 2 次运算，反向再做两倍，一共 6 次（为什么是 6，见${LQ('scaling-law')}）。所以整次训练的运算量是：</p>`+M_('C = 6 \\times 70.6\\text{B} \\times 15\\text{T} \\approx 6.35 \\times 10^{24}\\ \\text{FLOP}')+
    `<p>再看每秒能做多少。1,024 张卡的${DEF('峰值',TIP.peak())}加起来很大，可训练时 GPU 只能用上其中一部分。用上的比例叫 ${DEF('MFU',TIP.mfu())}：模型真正需要的运算（每个 token 6 × 参数量次），占同一段时间里 GPU 峰值算力的比例。等通信、流水线空转的时间都算在里面。</p>`+M_('\\text{训练时间} = \\frac{C}{G \\times \\text{峰值} \\times \\text{MFU}}')+
    `<p>G 是卡数。每张 H100 的峰值是 989.5 TFLOP/s。选一个 MFU，看看要训多少天。</p>`+levers('tcap-ic-lv2',tx('MFU, pick one','MFU，选一个'),u2,LV2,tried2,lv2Res)),
  obs:()=>obs2()},
 {id:'memory',title:{en:'Fit first: split 1.27 TB',zh:'先放得下：把 1.27 TB 切开'},expect:()=>tried3.has('tp8z3')||tried3.has('tp8pp8z1'),showMe:()=>setLv3('tp8pp8z1'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('1.27 TB of model states, 80 GB per GPU: which cut fits?','模型状态 1.27 TB，一张卡 80 GB：怎么切才放得下？'),
  enter:()=>{if(!done('memory')){lv3='none'; tried3=new Set(['none']);} return {...MISSION,...lvOf(LV3,lv3).s};},
  body:()=>tx(`<p>An H100 has only 80 GB of memory, but a 70B model’s ${DEF('model states',TIP.states())} alone take 1.27 TB, 18 bytes per parameter (${labLink('training-memory')}), and each GPU also keeps the ${DEF('activations',TIP.act())} it computes. So the model must be cut up and spread over many GPUs.</p><p>There are three ways to cut, and they stack:</p><ul class="tcap-ul">`+
    `<li><b>Tensor parallelism</b> splits every layer’s matrices into t pieces on t GPUs of one machine, which add up their results with an ${DEF('all-reduce',TIP.allreduce())} (${labLink('model-parallel')}).</li>`+
    `<li><b>Pipeline parallelism</b> puts the 80 layers into p stages on different GPUs; a step’s ${DEF('micro-batches',TIP.mb())} flow through in turn, and GPUs at both ends wait: the ${DEF('bubble',TIP.bubble())}.</li>`+
    `<li><b>Data parallelism</b>: every t × p GPUs together compute one whole model, so the 1,024 GPUs form several copies, and each copy trains on its own part of the step’s 1,024 sequences. ZeRO splits the model states among the copies: ZeRO-1 the optimizer state, ZeRO-2 the gradients too, ZeRO-3 the weights too, gathering each layer’s weights just before computing it. PyTorch’s FSDP is ZeRO-3 (${labLink('parallelism')}).</li></ul>`+
    `<p>Pick a layout and see how much memory each GPU needs.</p>`+levers('tcap-ic-lv3',tx('How to split, pick one','怎么切，选一种'),lv3,LV3,tried3,lv3Res),
    `<p>一张 H100 只有 80 GB 显存，可 70B 模型光${DEF('模型状态',TIP.states())}就有 1.27 TB（每个参数 18 字节，见${LQ('training-memory')}），这还没算${DEF('激活',TIP.act())}。所以要把模型切开，分到很多张卡上。</p><p>有三种切法，可以叠起来用：</p><ul class="tcap-ul">`+
    `<li><b>张量并行</b>：把每一层的矩阵切成 t 份，放在同一台机器的 t 张卡上，靠 ${DEF('all-reduce',TIP.allreduce())} 把结果加起来（见${LQ('model-parallel')}）。</li>`+
    `<li><b>流水线并行</b>：把 80 层分成 p 段，放在不同的卡上，一步的数据切成很多个 ${DEF('micro-batch',TIP.mb())} 依次流过，两头有卡空等，这叫 ${DEF('bubble',TIP.bubble())}。</li>`+
    `<li><b>数据并行</b>：每 t × p 张卡合起来算一份完整的模型，1,024 张卡就分成好几份，各算每步 1,024 条序列里的不同几条。ZeRO 再把模型状态分到这几份上：ZeRO-1 分优化器状态，ZeRO-2 再分梯度，ZeRO-3 连权重也分，算到哪一层，就先把那一层的权重收集齐。PyTorch 的 FSDP 就是 ZeRO-3（见${LQ('parallelism')}）。</li></ul>`+
    `<p>选一种切法，看看每张卡要放多少显存。</p>`+levers('tcap-ic-lv3',tx('How to split, pick one','怎么切，选一种'),lv3,LV3,tried3,lv3Res)),
  obs:()=>obs3()},
 {id:'time',title:{en:'Where the time goes',zh:'时间花在哪'},expect:()=>tried4.size>=3,showMe:()=>setLv4('tp8z3'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('In one step, how much of each GPU’s time is real computing?','一步里，每张卡有多少时间在真正地算？'),
  enter:()=>{if(!done('time')){lv4='start'; tried4=new Set(['start']);} return {...MISSION,...lvOf(LV4,lv4).s};},
  body:()=>tx(`<p>Once it fits, look at speed. The bar in the chart is one GPU’s time in one step: besides computing, it waits for three things.</p><ul class="tcap-ul">`+
    `<li><b>Tensor-parallel all-reduces</b>: 2 in the forward and 2 in the backward of every layer, over the machine’s ${DEF('NVLink',TIP.nvlink())}.</li>`+
    `<li><b>The bubble</b> at the two ends of the pipeline.</li>`+
    `<li><b>Data-parallel syncing</b>. ZeRO-1 syncs once a step, alongside only the last micro-batch’s backward pass; the part that this computing hides costs no extra time. ZeRO-3 gathers the weights again for every micro-batch.</li></ul>`+
    `<p>Even while computing, the GPUs do not reach the peak (a plain BF16 matrix multiply reaches about 80%); this lab assumes 50% of the peak while computing. Counting the waits too, the MFU of one step, training N parameters on G GPUs, is:</p>`+M_('\\text{MFU} = \\frac{6 \\times N \\times \\text{tokens per step}}{G \\times \\text{peak} \\times \\text{step time}}')+
    `<p>One more option saves memory: ${DEF('recomputation',TIP.recompute())}, which keeps fewer activations and recomputes them in the backward pass; it costs 8N operations per token instead of 6N. Pick a layout and see the MFU of one step.</p>`+levers('tcap-ic-lv4',tx('Layout, pick one','切法，选一种'),lv4,LV4,tried4,lv4Res),
    `<p>放得下以后，再看快不快。图里的横条是一张卡一步的时间，除了在算，它还要等三样东西：</p><ul class="tcap-ul">`+
    `<li><b>张量并行的 all-reduce</b>：每一层前向 2 次、反向 2 次，走机器里的 ${DEF('NVLink',TIP.nvlink())}。</li>`+
    `<li><b>bubble</b>：流水线两头的空等。</li>`+
    `<li><b>数据并行的同步</b>：ZeRO-1 每步同步一次，只能和最后一个 micro-batch 的反向同时进行，能藏进这段计算里的部分就不另占时间；ZeRO-3 每个 micro-batch 都要重新收集权重。</li></ul>`+
    `<p>就算在算，GPU 也到不了峰值（单纯的 BF16 矩阵乘法约到 80%），这个 Lab 假设跑到峰值的 50%。把等待也算进去，N 个参数的模型在 G 张卡上训练，一步的 MFU 是：</p>`+M_('\\text{MFU} = \\frac{6 \\times N \\times \\text{每步的 token 数}}{G \\times \\text{峰值} \\times \\text{每步耗时}}')+
    `<p>还有一种省显存的办法，叫${DEF('重算',TIP.recompute())}：前向少存激活，反向时再把前向算一遍。代价是每个 token 要做 8N 次运算，而不是 6N。选一种切法，看看一步的 MFU 是多少。</p>`+levers('tcap-ic-lv4',tx('Layout, pick one','切法，选一种'),lv4,LV4,tried4,lv4Res)),
  obs:()=>obs4()},
 {id:'failures',title:{en:'Failures: more GPUs, more often',zh:'算上故障：卡越多，越常出事'},expect:()=>tried5.has('15a'),showMe:()=>setLv5('15a'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('128 machines, one fails every 12 hours: how often should you save?','128 台机器，平均 12 小时坏一台：多久存一次档？'),
  enter:()=>{if(!done('failures')){lv5='3s'; tried5=new Set(['3s']);} return {...MISSION,...lvOf(LV5,lv5).s};},
  body:()=>{const r=plan(MISSION), wTip=tx('This lab assumes the whole job writes its checkpoint to the shared file system at 10 GB/s. BLOOM-176B wrote its 2.3 TB checkpoint from 384 GPUs in about 40 s, about 57 GB/s, but shared file systems differ a lot, so this lab takes a slower 10 GB/s.','这个 Lab 假设整个任务每秒往共享文件系统写 10 GB。BLOOM-176B 用 384 张卡约 40 秒写完 2.3 TB 的存档，每秒约 57 GB。可各家的共享文件系统差别很大，这里取慢一些的每秒 10 GB。');
   return tx(`<p>A run this long will lose machines, and each failure wastes time. This part keeps the initial layout (MFU ${pct(r.mfu)}); what you choose is how often and how to save. In OPT-175B’s run about 2 of 128 machines went down every day, so this lab takes one failure every 64 days per machine. The more machines, the more often the job fails; it fails once every M on average:</p>`+M_('M = \\frac{64\\ \\text{days}}{128\\ \\text{machines}} = 12\\ \\text{hours}')+
    `<p>When a machine fails, the whole job resumes from the latest ${DEF('checkpoint',TIP.ckpt())} (${labLink('checkpointing')}): swapping the machine and loading the checkpoint take an assumed 10 minutes, and the steps since the last save are lost, half an interval on average.</p>`+
    `<p>Saving costs time too. <b>Synchronous saving</b> stops training until the checkpoint of about 1 TB is written, about ${fixd(r.dSync,0)} s at ${DEF('10 GB/s',wTip)}; <b>asynchronous saving</b> stops it only for the copy to CPU memory, an assumed ${fixd(CK.asyncS,0)} s, and writes in the background.</p>`+M_('\\text{wasted} \\approx \\frac{δ}{τ} + \\frac{τ/2 + R}{M}')+
    `<p>δ is the time one save stops training, τ the interval between saves, R the restart time. Pick a way to save and see how much of the time is wasted.</p>`+levers('tcap-ic-lv5',tx('Saving, pick one','怎么存档，选一种'),lv5,LV5,tried5,lv5Res),
    `<p>训练这么久，一定会坏机器，每坏一次都要白费时间。这一部分沿用初始的切法（MFU ${pct(r.mfu)}），要选的是多久存一次档、怎么存。OPT-175B 训练时，128 台机器每天约坏 2 台，这里就按每台 64 天坏一次算。机器越多，整个任务越常出事，平均每隔 M 出一次：</p>`+M_('M = \\frac{64\\ \\text{天}}{128\\ \\text{台}} = 12\\ \\text{小时}')+
    `<p>机器坏了，整个任务从最近一次${DEF('存档',TIP.ckpt())}接着训（存档也叫检查点，见${LQ('checkpointing')}）：换机器、读回存档假设要 10 分钟，上次存档以后算的步也白算了，平均是半个存档间隔。</p>`+
    `<p>存档本身也花时间。<b>同步存档</b>要停下训练，等约 1 TB 的存档写完，按${DEF('每秒写 10 GB',wTip)} 算约 ${fixd(r.dSync,0)} 秒；<b>异步存档</b>只在拷到 CPU 内存时停一下，这里假设 ${fixd(CK.asyncS,0)} 秒，写盘交给后台。</p>`+M_('\\text{白费的比例} \\approx \\frac{δ}{τ} + \\frac{τ/2 + R}{M}')+
    `<p>δ 是每次存档让训练停多久，τ 是存档间隔，R 是重启要多久。选一种存档方式，看看白费了多少时间。</p>`+levers('tcap-ic-lv5',tx('Saving, pick one','怎么存档，选一种'),lv5,LV5,tried5,lv5Res));},
  obs:()=>obs5()},
 {id:'real',title:{en:'Against real runs',zh:'对照真实的训练'},label:{en:'Observed',zh:'观察到'},
  lead:()=>tx('Meta’s published GPU hours: what share of the peak did they really get?','Meta 公布的 GPU 小时，说明真实训练做到了峰值的几成？'),
  enter:()=>({...MISSION,ivl:0.25,asy:1}),
  body:()=>tx(`<p>The Llama 3.1 model card lists the GPU hours each model took, 7.0M for 70B, which gives a check on this plan: the work 6ND, divided by what those GPU hours could do at the H100’s peak, is the share of the peak the real run used on average.</p>`+
    `<p>Some published layouts:</p><ul class="tcap-al">`+
    `<li><b>Llama 3 405B</b><span>16,384 H100s: TP 8 × PP 16 × ZeRO-1 DP 128 (quoted by the scaling book)</span></li>`+
    `<li><b>Llama 3.1 70B</b><span>256 H100s: TP 8 × FSDP 32 (torchtitan’s benchmark)</span></li>`+
    `<li><b>Llama 3 70B</b><span>64 H100s: TP 4 × PP 8 × DP 2 (NVIDIA’s Megatron-Bridge recipe)</span></li></ul>`+
    `<p>Megatron-LM reports 41% to 47% MFU on H100s, above this plan’s ${pct(plan(MISSION).mfu)}, partly because it overlaps the tensor-parallel all-reduces with computing, while this lab counts them all as waiting.</p>`,
    `<p>Llama 3.1 的模型卡公布了每个模型用的 GPU 小时，70B 是 7.0M，可以拿来检验这个计划：运算量 6ND 除以这些 GPU 小时按峰值能做的运算，就是真实训练平均用上了峰值的几成。</p>`+
    `<p>几种公开的切法：</p><ul class="tcap-al">`+
    `<li><b>Llama 3 405B</b><span>16,384 张 H100：张量并行 8 × 流水线 16 段 × ZeRO-1 数据并行 128 份（scaling book 引用）</span></li>`+
    `<li><b>Llama 3.1 70B</b><span>256 张 H100：张量并行 8 × FSDP 32 份（torchtitan 的测试）</span></li>`+
    `<li><b>Llama 3 70B</b><span>64 张 H100：张量并行 4 × 流水线 8 段 × 数据并行 2 份（NVIDIA 的 Megatron-Bridge 配方）</span></li></ul>`+
    `<p>Megatron-LM 报告它在 H100 上的 MFU 是 41% 到 47%，比这个计划的 ${pct(plan(MISSION).mfu)} 高，一个原因是它让张量并行的 all-reduce 和计算重叠，这个 Lab 把它们全算成等待。</p>`),
  obs:()=>obs6()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Total work first, then fit the model in memory, then the waits, then failures.','先算总账，再把模型放进显存，再看等待，最后把故障算进来。'),
  enter:()=>({...MISSION,ivl:0.25,asy:1}),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const right=pred===PRED_RIGHT, d=simpleDays(0.4,MISSION);
  return right?tx(`<p class="okline"><b>Your prediction “${predT(pred)}” is right: at 40% the run takes about ${fDays(d)} days, half a year.</b></p>`,`<p class="okline"><b>你预测的“${predT(pred)}”是对的：按四成算，要训约 ${fDays(d)} 天，差不多半年。</b></p>`)
             :tx(`<p class="badline"><b>You predicted “${predT(pred)}”; the answer is “${predT(PRED_RIGHT)}”: at 40% the run takes about ${fDays(d)} days, half a year.</b></p>`,`<p class="badline"><b>你预测的是“${predT(pred)}”，答案是“${predT(PRED_RIGHT)}”：按四成算，要训约 ${fDays(d)} 天，差不多半年。</b></p>`);
}
function obs2(){
  const d4=simpleDays(0.4,MISSION), du=simpleDays(u2,MISSION), gh=1024*d4*24;
  let h=(pred?verdict():tx(`<p><b>At 40% the run takes about ${fDays(d4)} days.</b></p>`,`<p><b>按四成算，要训约 ${fDays(d4)} 天。</b></p>`))+
    tx(`<p>1,024 GPUs held that long add up to ${big(Math.round(gh))} ${DEF('GPU hours',TIP.gpuh())}. GPU hours = GPUs × hours, all GPUs counted together: 8 GPUs for 1 hour are 8 GPU hours.</p>`,`<p>1,024 张卡训这么久，一共是 ${big(Math.round(gh))} ${DEF('GPU 小时',TIP.gpuh())}。GPU 小时 = 用的 GPU 数 × 小时数，所有卡加在一起算：8 张卡跑 1 小时是 8 GPU 小时。</p>`);
  if(u2!==0.4) h+=tx(`<p>At ${pct0(u2)} it takes ${fDays(du)} days. The time is inversely proportional to the MFU: ${u2<0.4?'a quarter less MFU means a third more days':'a quarter more MFU means a fifth fewer days'}.</p>`,
                      `<p>MFU 是${u2===0.3?'三成':'五成'}时要 ${fDays(du)} 天。时间和 MFU 成反比：${u2<0.4?'MFU 少四分之一，天数就多三分之一':'MFU 多四分之一，天数就少五分之一'}。</p>`);
  else h+=tx('<p>Pick another MFU to see how the days change.</p>','<p>换一个 MFU，看看天数怎么变。</p>');
  return h+tx(`<p class="small muted">Where the other 60% goes is part 4.</p>`,`<p class="small muted">剩下那六成去了哪里，第 4 部分再看。</p>`);
}
function obs3(){
  const r=p3r(lv3);
  if(lv3==='none') return tx(`<p><b>No split: each GPU would need ${fGB(r.mem)}, ${fixd(r.mem/GPU.mem,0)} times its 80 GB.</b></p>`,`<p><b>不切时，每张卡要存 ${fGB(r.mem)}，是 80 GB 的 ${fixd(r.mem/GPU.mem,0)} 倍。</b></p>`);
  if(lv3==='z3') return tx(`<p><b>Still too big: ${fGB(r.mem)} per GPU.</b></p><p>ZeRO-3 leaves each GPU only about ${fGB(r.states)} of model states, plus about ${fGB(r.zx)} for the BF16 weights of the two layers it gathers (the one being computed and the next, gathered ahead), but the activations are ${fGB(r.act)}. One micro-batch going through all 80 layers leaves them on the GPU, and ZeRO cannot split them. Tensor parallelism, which cuts every layer into pieces, is what shrinks the activations.</p>`,
                                `<p><b>还是放不下：每张卡 ${fGB(r.mem)}。</b></p><p>ZeRO-3 让每张卡的模型状态只剩约 ${fGB(r.states)}，再加上临时收集齐的两层 BF16 权重（正在算的这一层和提前收集的下一层），约 ${fGB(r.zx)}，可激活有 ${fGB(r.act)}。一个 micro-batch 走过全部 80 层，激活都留在这张卡上，ZeRO 一点也分不掉。张量并行把每一层切成几份，激活才跟着变少。</p>`);
  const other=lv3==='tp8z3'?'tp8pp8z1':'tp8z3', ro=p3r(other), both=tried3.has(other);
  if(lv3==='tp8z3') return tx(`<p><b>It fits: ${fGB(r.mem)} per GPU.</b></p><p>Tensor parallelism cuts every layer’s activations into 8, and ZeRO-3 spreads the model states over all 1,024 GPUs.</p>`,`<p><b>放得下：每张卡 ${fGB(r.mem)}。</b></p><p>张量并行把每一层的激活切成 8 份，ZeRO-3 把模型状态分到全部 1,024 张卡上。</p>`)+
    (both?tx(`<p>Both ways fit. Which one is faster is part 4.</p>`,`<p>两种都放得下。哪种更快，第 4 部分再看。</p>`):'');
  return tx(`<p><b>It fits: ${fGB(r.mem)} per GPU.</b></p><p>The pipeline gives each GPU 10 of the 80 layers, and tensor parallelism cuts them into 8, so each GPU keeps only 1/64 of the weights and gradients. But ZeRO-1 splits only the optimizer states, so each GPU still keeps ${fGB(r.states)} of model states, far more than the ${fGB(ro.states)} with TP 8 + ZeRO-3. The activations are as large as with that layout, ${fGB(r.act)}: ${r.inf} micro-batches are in the pipeline at once, and the marked GPU, in the first stage, keeps the activations of every one of them on its 10 layers, as much as one micro-batch through all 80 layers. So each GPU needs more memory than that layout’s ${fGB(ro.mem)}.</p>`,`<p><b>放得下：每张卡 ${fGB(r.mem)}。</b></p><p>流水线让每张卡只负责 80 层里的 10 层，张量并行再把它们切成 8 份，每张卡只存 1/64 的权重和梯度。可 ZeRO-1 只分优化器状态，每张卡的模型状态还有 ${fGB(r.states)}，比“张量并行 8 份 + ZeRO-3”的 ${fGB(ro.states)} 多得多。激活倒和那种切法一样多，${fGB(r.act)}：流水线上同时有 ${r.inf} 个 micro-batch，标出的那张卡在第一段，要给每个都存这 10 层的激活，加起来和一个 micro-batch 走完 80 层一样多。所以每张卡要的显存比那种切法的 ${fGB(ro.mem)} 多。</p>`)+
    (both?tx(`<p>Both ways fit. Which one is faster is part 4.</p>`,`<p>两种都放得下。哪种更快，第 4 部分再看。</p>`):'');
}
function obs4(){
  const r=p4r(lv4), s=p4r('start');
  if(lv4==='start') return tx(`<p><b>The initial setup reaches an MFU of ${pct(r.mfu)}.</b></p><p>Each GPU computes for ${pct0(r.seg.useful/r.step)} of the step, at 50% of the peak while computing; the MFU is the two multiplied. Over 60% of the peak goes unused, lost in these two places.</p>`,
                                 `<p><b>初始设置的 MFU 是 ${pct(r.mfu)}。</b></p><p>一步里，每张卡有 ${pct0(r.seg.useful/r.step)} 的时间在算，算的时候跑到峰值的 50%，两者相乘就是 MFU。峰值没用上的那六成多，就丢在这两处。</p>`);
  const base=tx(`<p><b>MFU ${pct(s.mfu)} → ${pct(r.mfu)}.</b></p>`,`<p><b>MFU 从 ${pct(s.mfu)} 变成 ${pct(r.mfu)}。</b></p>`);
  if(lv4==='rc') return base+tx(`<p>Every token now does one more forward pass, 8N operations instead of 6N, but MFU counts only 6N. The extra forward pass does its tensor-parallel all-reduces too, so the all-reduces grow by half. The memory it saves is of no use here, because the layout already fits.</p>`,`<p>每个 token 多算了一遍前向，运算量从 6N 变成 8N，可 MFU 只数 6N。多算的这遍前向也要做张量并行的 all-reduce，all-reduce 就多了一半。它省下的显存这里用不上，因为这种切法本来就放得下。</p>`);
  if(lv4==='tp4pp16') return base+tx(`<p>Almost no change. Tensor parallelism goes from 8 ways to 4, so the all-reduces take less of the step. The pipeline goes from 8 stages to 16, so the bubble grows to ${pct(r.seg.bubble/r.step)} of the step. What one side saves, the other spends.</p>`,`<p>几乎没变。张量并行从 8 份减到 4 份，all-reduce 占的时间变少了。流水线从 8 段加到 16 段，bubble 涨到一步的 ${pct(r.seg.bubble/r.step)}。一头省下的，另一头又花掉了。</p>`);
  if(lv4==='tp8z3') return base+tx(`<p>No pipeline means no bubble, but ZeRO-3 gathers every layer’s weights again for each micro-batch and sends the gradients back. A machine’s 8 GPUs share 400 GB/s of network, and with every layer cut into 8 a micro-batch is computed too quickly to hide that much traffic. For ${pct0(r.seg.dp/r.step)} of a step, this GPU waits for the network.</p>`,`<p>不分段就没有 bubble，可 ZeRO-3 每个 micro-batch 都要重新收集各层的权重，再把梯度发回去。一台机器的 8 张卡共用 400 GB/s 的网络，而每层切成 8 份以后，一个 micro-batch 很快就算完了，藏不住这么多通信。一步里有 ${pct0(r.seg.dp/r.step)} 的时间，这张卡在等网络。</p>`);
  return base+tx(`<p>Higher than the initial setup: no pipeline, so no bubble, and ZeRO-1 syncs once per step. The price is memory: ${fGB(r.mem)} per GPU, against ${fGB(s.mem)} with the pipeline.</p>`,`<p>比初始设置还高：不分段就没有 bubble，ZeRO-1 每步只同步一次。代价是显存：每张卡 ${fGB(r.mem)}，有流水线时只要 ${fGB(s.mem)}。</p>`);
}
function obs5(){
  const r=p5r(lv5), s=p5r('3s');
  if(lv5==='3s') return tx(`<p><b>Saving every 3 hours wastes ${pct(r.waste)} of the time.</b></p><p>Each failure loses 1.5 hours of work on average, and a failure comes every 12 hours.</p>`,`<p><b>每 3 小时存一次，白费 ${pct(r.waste)} 的时间。</b></p><p>每次故障平均白算 1.5 小时，而 12 小时就出一次事。</p>`);
  const base=tx(`<p><b>Wasted: ${pct(s.waste)} → ${pct(r.waste)}.</b></p>`,`<p><b>白费的比例从 ${pct(s.waste)} 降到 ${pct(r.waste)}。</b></p>`);
  if(lv5==='1s') return base+tx(`<p>A shorter interval loses less work in each failure. Saving three times as often, the saves still take only ${pct(r.wSave)} of the time.</p>`,`<p>间隔短了，每次故障白算的少了。存档次数变成 3 倍，存档本身也只占 ${pct(r.wSave)} 的时间。</p>`);
  if(lv5==='15s') return base+tx(`<p>Saving this often backfires: the saves themselves take ${pct(r.wSave)} of the time, so more is wasted than when saving every hour (${pct(p5r('1s').waste)}).</p>`,`<p>存得太勤反而亏了：存档本身就占了 ${pct(r.wSave)} 的时间，所以比每小时存一次（白费 ${pct(p5r('1s').waste)}）还多。</p>`);
  return base+tx(`<p>An asynchronous save stops training for only ${fixd(r.dS,0)} s, so saving this often is cheap. What is left are three small parts of about the same size: saving, work redone and restarts. Each restart costs 10 minutes, and only faster recovery shrinks that.</p>`,`<p>异步存档每次只停 ${fixd(r.dS,0)} 秒，所以存得这么勤也不贵。剩下的三项差不多大：存档、故障后重算和重启。重启每次要 10 分钟，这一项只能靠更快地换机器来省。</p>`);
}
function obs6(){
  const r=plan(S), r70=REAL[1];
  return tx(`<p><b>The real 70B run used ${pct(r70.u)} of the peak on average; this lab’s plan reaches ${pct(r.eff)} (MFU ${pct(r.mfu)} × ${pct(r.good)} of the time spent training).</b></p><p>The card calls its figure “the total GPU time required for training each model”, so the share is not an MFU, only a check of the order of size. And the order of size agrees: on average, the real 70B and 405B runs used a quarter to a third of the peak.</p>`,
            `<p><b>真实的 70B 训练平均用上了峰值的 ${pct(r70.u)}，这个 Lab 的计划是 ${pct(r.eff)}（MFU ${pct(r.mfu)} × 往前训练的时间占比 ${pct(r.good)}）。</b></p><p>模型卡的数字是“训练每个模型所需的全部 GPU 时间”，所以这个比例不是 MFU，只能对照量级。量级对得上：真实的 70B 和 405B 训练，平均只用上了峰值的四分之一到三分之一。</p>`);
}
function remember(){
  const s4=p4r('start'), z3=p3r('z3'), w3=p5r('3s'), w15=p5r('15a'), lo=Math.min(...LV4.map(o=>p4r(o.v).mfu)), hi=Math.max(...LV4.map(o=>p4r(o.v).mfu));
  return tx(`<ul class="tcap-ul">`+
    `<li><b>First the total.</b> The total work divided by what the GPUs really do per second gives the training time:${M_('\\text{training time} = \\frac{6ND}{G \\times \\text{peak} \\times \\text{MFU}}')}The training time is inversely proportional to the MFU: if the MFU drops from 40% to 30%, the run takes a third longer.</li>`+
    `<li><b>Fit first, then compare speed.</b> ZeRO splits only the model states, not the activations: those need tensor parallelism or recomputation. Usually several layouts fit, and their speeds can differ a lot.</li>`+
    `<li><b>More machines, more failures.</b> The job’s time between failures is one machine’s divided by the number of machines. So the more machines, the more often to save, and asynchronously, so that the saves do not hold up training.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div>`+
    `<p>✗ “Twice the GPUs, half the time.” With the same tokens per step, each pipeline gets fewer micro-batches, so the bubble grows. With more machines, failures come more often too.</p>`+
    `<p>✗ “MFU can reach 100%.” Even computing nonstop, matrix multiplies do not reach the peak: ml-engineering measured the best BF16 matrix multiply on an H100 at about 80% of it.</p>`+
    `<p>✗ “If it fits in memory, the layout is fine.” Several layouts can all fit while the best reaches about ${fixd(hi/lo,1)} times the MFU of the worst: waiting for transfers and idling in the pipeline pull the MFU down.</p></div>`+A.shell.toScenarios(),
   `<ul class="tcap-ul">`+
    `<li><b>先算总账。</b>总运算量除以每秒真正做到的运算，就是要训的时间：${M_('\\text{训练时间} = \\frac{6ND}{G \\times \\text{峰值} \\times \\text{MFU}}')}训练时间和 MFU 成反比。MFU 从四成降到三成，天数就多三分之一。</li>`+
    `<li><b>先放得下，再比快慢。</b>ZeRO 只分模型状态，分不了激活。激活要靠张量并行或重算来压。放得下的切法常常不止一种，它们的快慢可以差很多。</li>`+
    `<li><b>机器越多，故障越勤。</b>整个任务多久出一次故障，等于一台机器的故障间隔除以机器数。所以机器越多，越要勤存档，而且要用异步存档，存档才不拖慢训练。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div>`+
    `<p>✗ “卡数翻倍，时间减半。”每步的 token 数不变，卡多了，每条流水线分到的 micro-batch 就少了，bubble 变大。机器多了，故障也更勤。</p>`+
    `<p>✗ “MFU 能做到 100%。”就算一刻不停地算，矩阵乘法也到不了峰值：ml-engineering 实测，H100 上最好的 BF16 矩阵乘法约是峰值的 80%。</p>`+
    `<p>✗ “显存放得下就行。”几种切法都放得下时，MFU 最高的可以是最低的约 ${fixd(hi/lo,1)} 倍。等通信、流水线空转，都会把 MFU 拉低。</p></div>`+A.shell.toScenarios());
}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'8b',t:{en:'A small team trains 8B',zh:'小团队训练 8B'},d:{en:'64 GPUs, 1T tokens, FSDP only',zh:'64 张卡，1T 个 token，只用 FSDP'},s:{model:'8b',D:1e12,G:64,t:1,p:1,z:3,rc:0,b:1,B:1024,net:'ib',ivl:1,asy:1,node:64}},
 {id:'70b',t:{en:'70B on 1,024 GPUs',zh:'1,024 张卡训练 70B'},d:{en:'The plan from the guided parts, saving every 15 minutes asynchronously',zh:'分步引导里的计划，每 15 分钟异步存档'},s:{...BASE,ivl:0.25,asy:1}},
 {id:'405b',t:{en:'405B laid out like Llama 3',zh:'像 Llama 3 那样训练 405B'},d:{en:'16,384 GPUs, 16M tokens per step, BLOOM’s failure rate',zh:'16,384 张卡，每步 16M 个 token，按 BLOOM 的故障率'},s:{model:'405b',D:15e12,G:16384,t:8,p:16,z:1,rc:0,b:1,B:4096,net:'ib',ivl:0.25,asy:1,node:224}},
 {id:'eth',t:{en:'Ordinary Ethernet between machines',zh:'机器之间只有普通以太网'},d:{en:'The 70B plan on a cluster built for serving',zh:'把 70B 的计划搬到为推理搭的集群上'},s:{...BASE,ivl:0.25,asy:1,net:'eth'}},
 {id:'fail',t:{en:'New machines that fail often',zh:'新机器故障多，还在同步存档'},d:{en:'Each machine fails once a month; saving every hour, synchronously',zh:'每台机器一个月坏一次；每小时同步存一次'},s:{...BASE,ivl:1,asy:0,node:32}}
];
const SKEYS=['model','D','G','t','p','z','rc','b','B','net','ivl','asy','node'];
const presetNow=()=>PRESETS.findIndex(p=>SKEYS.every(k=>p.s[k]===S[k]));
function scenHTML(i){
  const p=PRESETS[i], r=plan(S);
  if(!r) return '';
  if(p.id==='8b') return tx(`<p><b>${dayU(r.days)} and ${fGB(r.mem)} per GPU, at an MFU of ${pct(r.mfu)}.</b></p><p>8B is small enough that FSDP alone (ZeRO-3) splits it, with no tensor or pipeline parallelism. Each micro-batch gathers all the weights again, but 8B’s BF16 weights are only 16 GB: a machine gathers them much faster than it computes the micro-batch, so the traffic hides behind the computing.</p><p>${pct(r.mfu)} is this lab’s ceiling: the GPUs run at 50% of the peak while computing and never wait. torchtitan measured 33% to 39% for Llama 3.1 8B with FSDP on 8 or 128 H100s.</p>`,
                              `<p><b>${dayU(r.days)}，每张卡 ${fGB(r.mem)}，MFU ${pct(r.mfu)}。</b></p><p>8B 够小，只用 FSDP（也就是 ZeRO-3）就能分开，不用张量并行和流水线并行。每个 micro-batch 都要重新收集一遍权重，可 8B 的 BF16 权重只有 16 GB，一台机器收齐它们比算完这个 micro-batch 快得多，所以通信全藏在了计算后面。</p><p>${pct(r.mfu)} 是这个 Lab 的上限：GPU 算的时候能跑到峰值的 50%，又没有任何等待。torchtitan 在 8 张或 128 张 H100 上用 FSDP 训练 Llama 3.1 8B，实测 MFU 是 33% 到 39%。</p>`);
  if(p.id==='70b') return tx(`<p><b>${dayU(r.days)}, ${fGpuh(r.gpuh)} GPU hours, MFU ${pct(r.mfu)}.</b></p><p>This is the guided plan, TP 8 × PP 8 with ZeRO-1, saving asynchronously every 15 minutes. Llama 3.1’s model card reports 7.0M GPU hours for its 70B model, about ${fixd(7e6/r.gpuh,1)} times this plan. One possible reason is that the card counts all the time the GPUs were held for this model.</p>`,
                               `<p><b>${dayU(r.days)}，${fGpuh(r.gpuh)} GPU 小时，MFU ${pct(r.mfu)}。</b></p><p>这是分步引导里的计划：张量并行 8 份 × 流水线 8 段 + ZeRO-1，每 15 分钟异步存一次档。Llama 3.1 的模型卡说 70B 用了 7.0M GPU 小时，约是这个计划的 ${fixd(7e6/r.gpuh,1)} 倍。一个可能的原因是，模型卡把为这个模型占着卡的时间都算了进去。</p>`);
  if(p.id==='405b'){const card=30.84e6/r.gpuh;
    return tx(`<p><b>${dayU(r.days)}, MFU ${pct(r.mfu)}.</b></p><p>The scaling book quotes this layout for Llama 3: 16 pipeline stages and 16M tokens per step. Each pipeline gets few micro-batches for that many stages, so the bubble takes ${pct(r.seg.bubble/r.step)} of a step.</p><p>Failure rates differ a lot between clusters, so this scenario uses BLOOM-176B’s lower rate. Even so, 2,048 machines lose one every ${fH(r.M)}.</p><p>Llama 3.1’s model card reports 30.84M GPU hours for 405B, about ${pct0(1-card)} fewer than this estimate. At this lab’s default OPT-175B rate, the estimate would be close to twice the card’s figure.</p>`,
              `<p><b>${dayU(r.days)}，MFU ${pct(r.mfu)}。</b></p><p>scaling book 引用了 Llama 3 的这种切法：16 段流水线，每步 16M 个 token。段数这么多，每条流水线分到的 micro-batch 却不多，所以 bubble 占了一步的 ${pct(r.seg.bubble/r.step)}。</p><p>故障率在不同的集群上差别很大，这个场景用 BLOOM-176B 记下的较低的故障率。就算这样，2,048 台机器平均每 ${fH(r.M)}也要坏一台。</p><p>Llama 3.1 的模型卡说 405B 一共用了 30.84M GPU 小时，比这个估算少约 ${pct0(1-card)}。如果按这个 Lab 默认的 OPT-175B 故障率，估出来的 GPU 小时会接近模型卡的两倍。</p>`);}
  if(p.id==='eth') {const rb=plan({...S,net:'ib'});
    return tx(`<p><b>MFU ${pct(rb.mfu)} → ${pct(r.mfu)}: ${dayU(r.days)} instead of ${fDays(rb.days)}.</b></p><p>Here a machine’s 8 GPUs share one 100 Gb/s Ethernet link, a thirty-second of InfiniBand’s bandwidth.</p><p>The data-parallel sync of each step can no longer hide behind the last backward pass. For ${pct0(r.seg.dp/r.step)} of a step, this GPU waits for it. Tensor parallelism stays on NVLink inside the machine, so it is unchanged.</p>`,
              `<p><b>MFU 从 ${pct(rb.mfu)} 降到 ${pct(r.mfu)}：要 ${dayU(r.days)}，而不是 ${fDays(rb.days)} 天。</b></p><p>这里一台机器的 8 张卡共用一条 100 Gb/s 的以太网，带宽只有 InfiniBand 的 1/32。</p><p>每步的数据并行同步，再也藏不到最后一个反向后面。一步里有 ${pct0(r.seg.dp/r.step)} 的时间，这张卡在等它。张量并行还在机器里的 NVLink 上，所以不受影响。</p>`);}
  const rf=plan({...S,ivl:0.25,asy:1});
  return tx(`<p><b>${pct(r.waste)} of the time is wasted: a failure comes every ${fH(r.M)}.</b></p><p>A new generation of GPUs fails more often when it first comes out; ml-engineering advises keeping 5% to 10% spare machines. With failures this frequent, the half hour of work lost on average in each failure costs more. Saving every 15 minutes asynchronously would cut the waste to ${pct(rf.waste)}.</p>`,
            `<p><b>白费了 ${pct(r.waste)} 的时间：平均每 ${fH(r.M)}就出一次事。</b></p><p>新一代 GPU 刚上市时坏得多，ml-engineering 建议多备 5% 到 10% 的机器。故障这么勤，每次平均白算的半小时就更划不来。改成每 15 分钟异步存一次，白费的时间能降到 ${pct(rf.waste)}。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=e=>tx(`TP ${e.t} × PP ${e.p} × DP ${512/(e.t*e.p)}, ${zName(e.z)}${e.rc?', recompute':''}, ${e.ivl===1?'1 hour':fIvl(e.ivl)} ${e.asy?'async':'sync'}`,
                    `张量并行 ${e.t} × 流水线 ${e.p} 段 × 数据并行 ${512/(e.t*e.p)} 份，${zName(e.z)}${e.rc?'，重算':''}，${everyIvl(e.ivl)}${e.asy?'异步':'同步'}存档`);
const chMe=()=>chEval(ch);
/* ---------- the result after Submit: a summary first, the full list behind “See all plans” (Codex review b9v3, 应修 5) ---------- */
const CHK=['t','p','z','rc','ivl','asy'], CHO={t:CH.Ts,p:CH.Ps,z:CH.Zs,rc:CH.RCs,ivl:CH.IVs,asy:CH.ASs};
const chPick=e=>({t:e.t,p:e.p,z:e.z,rc:e.rc,ivl:e.ivl,asy:e.asy});
const chSame=(a,b)=>CHK.every(k=>a[k]===b[k]);
const chSameLayout=(a,b)=>a.t===b.t&&a.p===b.p&&a.z===b.z&&a.rc===b.rc;
/* the representative counterexample: the recommended plan with only tensor parallelism changed, to 8 ways (each GPU computes too little
   to hide ZeRO-3's traffic); to 1 way (the activations do not fit) when the learner's own layout is the 8-way one */
function chCounter(me){const b=chBest(); let c=chEval({...chPick(b),t:8}); if(me&&chSameLayout(c,me)) c=chEval({...chPick(b),t:1}); return c;}
/* the single change to the learner's plan that meets the goal with the fewest GPU hours. For a plan that meets it, only a change
   whose GPU hours, as the result shows them (2 significant digits), are lower than the plan's own (rule G7: never “650K → 650K”) */
function chOneChange(me){let b=null;
  CHK.forEach(k=>CHO[k].forEach(v=>{if(v===me[k]) return; const e=chEval({...chPick(me),[k]:v});
    if(e&&e.ok&&(!me.ok||roughV(e.r.gpuh)<roughV(me.r.gpuh))&&(!b||e.r.gpuh<b.e.r.gpuh)) b={k,v,e};}));
  return b;}
/* a plan that meets the goal, for which no single change shows fewer GPU hours: does some change still save a little, unseen at the
   precision shown? (“省不了多少” rather than “省不下来”) */
function chOneTiny(me){return !!me&&me.ok&&CHK.some(k=>CHO[k].some(v=>{if(v===me[k]) return false; const e=chEval({...chPick(me),[k]:v}); return !!e&&e.ok&&e.r.gpuh<me.r.gpuh;}));}
/* where a plan's days go: the step's parts scaled to the training time, then saving and failures (the stage's step and run bars) */
const dayParts=r=>{const td=r.train/86400, W=r.wall/86400; return {u:td*r.seg.useful/r.step,rc:td*r.seg.recomp/r.step,tp:td*r.seg.tp/r.step,bb:td*r.seg.bubble/r.step,dp:td*r.seg.dp/r.step,fx:W*r.waste};};
/* GPU hours the way a person says them: 2 significant digits with K or M (R6.18's K/M/B) */
const roughV=x=>{if(!isFinite(x)||x<=0) return NaN; const e=Math.floor(Math.log10(x)); return Math.round(x/10**(e-1))*10**(e-1);};   // the value as shown
const rough=x=>{if(!isFinite(x)||x<=0) return '—'; const r=roughV(x); return r>=1e6?`${fixd(r/1e6,r>=1e7?0:1)}M`:`${fint(Math.round(r/1e3))}K`;};
/* no hover with the exact value (the owner, 2026-10-07: “不需要有鼠标悬浮，把那个精确的数字展开给读者”); the full list keeps exact GPU hours */
const roughH=x=>rough(x);
const C3SEG=()=>[{k:'u',n:tx('Computing','在算')},{k:'rc',n:tx('Recomputing','重算')},{k:'tp',n:tx('Tensor-parallel all-reduces','张量并行的 all-reduce')},
  {k:'bb',n:tx('Bubble','bubble')},{k:'dp',n:tx('Waiting for data-parallel sync','等数据并行同步')},{k:'fx',n:tx('Saving and failures','存档和故障')}];
const C3CLS={u:'u',rc:'rc',tp:'tp',bb:'bb',dp:'dp',fx:'sv'};
/* the extra time of a plan, named for a sentence (“goes to …” / “花在……上”) */
const GAPN={rc:{en:'recomputation',zh:'重算'},tp:{en:'waiting for the tensor-parallel all-reduces',zh:'等张量并行的 all-reduce'},bb:{en:'the pipeline’s bubble',zh:'流水线的 bubble'},
  dp:{en:'waiting for data-parallel sync',zh:'等数据并行同步'},fx:{en:'saving and failures',zh:'存档和故障'}};
/* the settings a row can differ in; saving is one setting (how often, sync or async) */
const CHG=()=>[
 {v:e=>e.t,chip:e=>tx(`TP ${e.t}`,`张量并行 ${e.t} 份`),same:e=>tx(`tensor parallelism ${e.t} ways`,`张量并行都切 ${e.t} 份`)},
 {v:e=>e.p,chip:e=>e.p===1?tx('no pipeline','不分段'):tx(`PP ${e.p}`,`流水线 ${e.p} 段`),same:e=>e.p===1?tx('no pipeline','都不分段'):tx(`${e.p} pipeline stages`,`都分 ${e.p} 段流水线`)},
 {v:e=>e.z,chip:e=>e.z===0?tx('no ZeRO','不用 ZeRO'):`ZeRO‑${e.z}`,same:e=>e.z===0?tx('no ZeRO','都不用 ZeRO'):tx(`ZeRO‑${e.z}`,`都用 ZeRO‑${e.z}`)},
 {v:e=>e.rc,chip:e=>e.rc?tx('recompute every layer','每层都重算'):tx('no recomputation','不重算'),same:e=>e.rc?tx('recomputation in every layer','都每层重算'):tx('no recomputation','都不重算')},
 {v:e=>e.ivl+'/'+e.asy,chip:e=>tx(`save ${everyIvl(e.ivl)}, ${e.asy?'async':'sync'}`,`${everyIvl(e.ivl)}${e.asy?'异步':'同步'}存档`),same:e=>tx(`a save ${everyIvl(e.ivl)}, ${e.asy?'async':'sync'}`,`都${everyIvl(e.ivl)}${e.asy?'异步':'同步'}存一次档`)}];
/* the one change, in words */
function changeName(k,a,b){
  if(k==='t') return tx(`cutting tensor parallelism ${b} ways instead of ${a}`,`把张量并行从 ${a} 份改成 ${b} 份`);
  if(k==='p') return b===1?tx('dropping the pipeline','去掉流水线'):(a===1?tx(`adding a ${b}-stage pipeline`,`加上 ${b} 段流水线`):tx(`using ${b} pipeline stages instead of ${a}`,`把流水线从 ${a} 段改成 ${b} 段`));
  if(k==='z') return a===0?tx(`adding ZeRO‑${b}`,`用上 ZeRO‑${b}`):(b===0?tx('dropping ZeRO','不用 ZeRO'):tx(`ZeRO‑${b} instead of ZeRO‑${a}`,`把 ZeRO‑${a} 换成 ZeRO‑${b}`));
  if(k==='rc') return b?tx('recomputing every layer','每层都重算'):tx('dropping recomputation','去掉重算');
  if(k==='ivl') return tx(`saving every ${fIvl(b)} instead of every ${fIvl(a)}`,`把存档间隔从 ${fIvl(a)}改成 ${fIvl(b)}`);
  return b?tx('saving asynchronously','改用异步存档'):tx('saving synchronously','改用同步存档');
}
const savePct=(a,b)=>{const s=(1-b/a)*100; return s<1?tx('less than 1%','不到 1%'):`${Math.round(s)}%`;};
/* tier 1: the main gap of a plan that misses the goal, and what to change (R3.10: the result, then at most two sentences of reason) */
function gapHint(me){
  const r=me.r;
  if(!me.fits){const k=['o','g','w','zx','act'].reduce((a,b)=>r[b]>r[a]?b:a), v=fGB(r[k]);
    return ({o:tx(`The largest part is the optimizer state, ${v} per GPU. From ZeRO-1 on, it is split over the data-parallel copies.`,`最大的一块是优化器状态，每张卡 ${v}。从 ZeRO-1 起，它会分到各份数据并行上。`),
      g:tx(`The largest part is the FP32 gradients, ${v} per GPU. From ZeRO-2 on, they are split over the data-parallel copies.`,`最大的一块是 FP32 梯度，每张卡 ${v}。从 ZeRO-2 起，它会分到各份数据并行上。`),
      w:tx(`The largest part is the BF16 weights, ${v} per GPU. Tensor parallelism and pipeline stages cut them; of the ZeRO stages only ZeRO-3 splits them.`,`最大的一块是 BF16 权重，每张卡 ${v}。张量并行和流水线能把它切小，ZeRO 里只有 ZeRO-3 会分它。`),
      zx:tx(`The largest part is the BF16 weights of the two layers ZeRO-3 gathers (the one being computed and the next), ${v} per GPU. More tensor parallelism makes them smaller.`,`最大的一块是 ZeRO-3 临时收集的两层 BF16 权重（正在算的这一层和提前收集的下一层），每张卡 ${v}。张量并行切得越细，这两层越小。`),
      act:tx(`The largest part is the activations, ${v} per GPU. ZeRO does not touch them. Tensor parallelism cuts them; recompute only if nothing else fits.`,`最大的一块是激活，每张卡 ${v}。ZeRO 管不到激活，张量并行能把它切小。别的办法都放不下时，才用重算。`)})[k];}
  if(!r.valid) return tx('Save more often, and asynchronously.','存档要更勤，而且用异步。');
  const P=dayParts(r), k=Object.keys(GAPN).reduce((a,b)=>P[b]>P[a]?b:a);
  const head=tx(`Computing itself takes only about ${fDays(P.u)} days. Of the rest, the most goes to ${L(GAPN[k])}.`,`真正在算的只有约 ${fDays(P.u)} 天。其余的时间里，花得最多的是${L(GAPN[k])}。`);
  const tail=({rc:tx('Recomputation runs the forward pass twice; use it only when nothing else fits.','重算多做了一遍前向，别的办法都放不下时才用。'),
    tp:tx('The more ways tensor parallelism cuts each layer, the less each GPU computes, but the all-reduces do not shrink with it.','张量并行切得越细，每张卡算得越少，all-reduce 却不会跟着变短。'),
    bb:tx('The more pipeline stages, the longer the GPUs idle at the start and the end of each step.','流水线段数越多，每步开头和结尾空等得越久。'),
    dp:me.z===3?tx('ZeRO-3 exchanges data with the other machines for every micro-batch. The less each GPU computes, the less of that traffic it hides.','ZeRO-3 每个 micro-batch 都要和别的机器收发一遍。每张卡算得越少，这些通信越藏不住。')
      :tx('The data-parallel sync hides only behind the last backward passes of the step.','数据并行同步只能藏在一步最后的反向后面。'),
    fx:tx('Save often, and asynchronously.','存档要勤，而且用异步。')})[k];
  return A.lang()==='zh'?head+tail:head+' '+tail;
}
/* the summary: your plan, the recommended plan and one counterexample, as bars of days split by where the time goes */
function c3HTML(me,best){
  const isBest=chSame(me,best), ctr=chCounter(me);
  const rows=[{e:me,role:isBest?'mebest':'me'}].concat(isBest?[]:[{e:best,role:'best'}]).concat(ctr?[{e:ctr,role:'ctr'}]:[]);
  const n=rows.length, G=CHG(), diff=G.filter(g=>!rows.every(x=>g.v(x.e)===g.v(rows[0].e))), same=G.filter(g=>!diff.includes(g));
  const ran=rows.filter(x=>x.e.fits&&x.e.r.valid), MX=Math.ceil(Math.max(70,...ran.map(x=>x.e.r.days))/10)*10, gx=(CH.days/MX*100).toFixed(2);
  const segs=C3SEG(), used=segs.filter(s=>ran.some(x=>dayParts(x.e.r)[s.k]>0.05));
  const sameTxt=tx(`What the ${n===3?'three':'two'} plans share: 512 H100s, 2T tokens${same.map(g=>', '+g.same(rows[0].e)).join('')}.`,
                   `${n===3?'三':'两'}种方案都用 512 张 H100 训 2T 个 token${same.map(g=>'，'+g.same(rows[0].e)).join('')}。`);
  const lab={me:tx('Your plan','你的方案'),mebest:tx('Your plan, the recommended one','你的方案，就是推荐方案'),best:DEF(tx('Recommended','推荐方案'),tx('The plan that meets the goal with the fewest GPU hours.','达标的方案里，GPU 小时最少的那一种。')),ctr:tx('Counterexample','反例')};
  const row=x=>{const e=x.e, r=e.r, ran1=e.fits&&r.valid, miss=!e.ok;
    const chips=diff.map(g=>`<span class="tcap-chip${(x.role==='me'||x.role==='ctr')&&g.v(e)!==g.v(best)?' on':''}">${esc(g.chip(e))}</span>`).join('');
    let mid, sub;
    if(!e.fits){mid=`<div class="tcap-c3x bad">${tx(`Does not fit: ${fGB(r.mem)} per GPU`,`每张卡要 ${fGB(r.mem)}，放不下`)}</div>`; sub='';}
    else {const P=dayParts(r);
      mid=`<div class="tcap-c3trk"><div class="tcap-c3fill" style="width:${(Math.min(r.days,MX)/MX*100).toFixed(2)}%">`+
        segs.filter(s=>P[s.k]>0).map(s=>{const w=P[s.k]/r.days*100; return `<i class="tcap-sg ${C3CLS[s.k]}" style="width:${w.toFixed(3)}%"${w>=3?` data-tipx="${esc(`${s.n}${tx(': about ',`：约 `)}${dayU(P[s.k])}`)}"`:''}></i>`;}).join('')+
        `</div><i class="tcap-c3g" style="left:${gx}%"></i></div><b class="tcap-c3d mono${miss?' bad':''}">${ran1?dayU(r.days):'—'}</b>`;
      sub=tx(`about ${roughH(r.gpuh)} GPU hours`,`约 ${roughH(r.gpuh)} GPU 小时`)+
        (x.role==='best'&&me.fits&&me.r.valid?tx(`, ${savePct(me.r.gpuh,r.gpuh)} fewer than yours`,`，比你的方案省 ${savePct(me.r.gpuh,r.gpuh)}`):'')+
        (miss?`<span class="tcap-c3m bad">${tx(', over the 60-day goal','，超过了 60 天的目标')}</span>`:'');}
    return `<li class="tcap-c3r ${x.role}${miss?' miss':''}" data-role="${x.role}"><div class="tcap-c3h"><span class="tcap-c3t">${lab[x.role]}</span>${chips}</div>${mid}${sub?`<div class="tcap-c3s">${sub}</div>`:''}</li>`;};
  let h=`<div class="tcap-sum"><div class="tcap-cap">${tx(`How many days does each plan take, and where does the time go?`,`${n===3?'三':'两'}种方案各要训几天？时间花在了哪里？`)}</div>`+
    `<p class="small tcap-same">${sameTxt}</p><ul class="tcap-c3">${rows.map(row).join('')}</ul>`+
    `<div class="tcap-c3leg small">${used.map(s=>`<span><i class="tcap-sw ${C3CLS[s.k]}"></i>${esc(s.n)}</span>`).join('')}<span><i class="tcap-sw goal"></i>${tx('goal: 60 days','目标：60 天')}</span></div>`;
  /* the main gap, the best single change, and why the counterexample fails: one sentence each */
  if(me.fits&&me.r.valid&&!chMinimal(me)){const Pm=dayParts(me.r), Pb=dayParts(best.r), ext=me.r.days-best.r.days;
    if(ext>0.05){const ks=Object.keys(GAPN), k=ks.reduce((a,b)=>Pm[b]-Pb[b]>Pm[a]-Pb[a]?b:a), sh=(Pm[k]-Pb[k])/ext;
      h+=`<p class="tcap-gap">${tx(`Your plan takes ${fDays(ext)} more days than the recommended one. ${sh>=0.5?'Most of them go to':'The largest share goes to'} ${L(GAPN[k])}.`,
        `你的方案比推荐方案多训 ${fDays(ext)} 天，${sh>=0.5?`多出来的大部分花在${L(GAPN[k])}上`:`多出来的时间里，花得最多的是${L(GAPN[k])}`}。`)}</p>`;}}
  if(!chMinimal(me)){const oc=chOneChange(me), nm=oc?changeName(oc.k,me[oc.k],oc.v):'';
    /* a saving under 1% is written with the two values as the result shows them, which differ (chOneChange) */
    const sv=oc&&me.ok?((1-oc.e.r.gpuh/me.r.gpuh)*100<1?tx(`about ${roughH(oc.e.r.gpuh)} GPU hours instead of ${roughH(me.r.gpuh)}`,`GPU 小时从约 ${roughH(me.r.gpuh)} 降到约 ${roughH(oc.e.r.gpuh)}`)
      :tx(`${savePct(me.r.gpuh,oc.e.r.gpuh)} fewer GPU hours`,`GPU 小时能少 ${savePct(me.r.gpuh,oc.e.r.gpuh)}`)):'';
    /* no single change helps: say what the recommended plan changes together (rule G7) */
    const bd=CHK.filter(k=>me[k]!==best[k]).map(k=>changeName(k,me[k],best[k])), bl=tx(bd.length>1?bd.slice(0,-1).join(', ')+' and '+bd[bd.length-1]:(bd[0]||''),bd.join('、'));
    h+=`<p class="tcap-next">${oc?(me.ok?tx(`The best single change is ${nm}: ${sv}.`,`只改一项的话，${nm}最划算，${sv}。`)
        :tx(`Of the single changes that meet the goal, ${nm} uses the fewest GPU hours.`,`只改一项就达标的改法里，${nm}最省。`))
      :(me.ok?(chOneTiny(me)?tx(`No single change saves much. The recommended plan changes ${bd.length} settings at once: ${bl}.`,`只改一项省不了多少。推荐方案同时改了 ${bd.length} 项：${bl}。`)
          :tx(`No single change saves GPU hours. The recommended plan changes ${bd.length} settings at once: ${bl}.`,`只改一项省不下来。推荐方案同时改了 ${bd.length} 项：${bl}。`))
        :tx(`No single change meets the goal. The recommended plan changes ${bd.length} settings at once: ${bl}.`,`只改一项还达不到目标。推荐方案同时改了 ${bd.length} 项：${bl}。`))}</p>`;}
  if(ctr){const b=best;
    h+=`<p class="tcap-why small">${ctr.t>b.t?tx(`The counterexample changes one setting of the recommended plan: tensor parallelism ${ctr.t} ways. Each GPU computes a micro-batch in less time, but a machine still receives as many weights from the others, so ${pct0(ctr.r.seg.dp/ctr.r.step)} of each step goes to waiting for data-parallel sync.`,
        `反例和推荐方案只差一项，它的张量并行切 ${ctr.t} 份。每张卡算一个 micro-batch 的时间短了，一台机器要从别的机器收的权重却一样多。所以一步里有 ${pct0(ctr.r.seg.dp/ctr.r.step)} 的时间在等数据并行同步。`)
      :tx(`The counterexample changes one setting of the recommended plan: no tensor parallelism. Each GPU then keeps the uncut activations of all ${MD['70b'].L} layers and needs ${fGB(ctr.r.mem)} in all, more than 80 GB.`,
        `反例和推荐方案只差一项，它不切张量并行。每张卡要存 ${MD['70b'].L} 层没切开的激活，一共要 ${fGB(ctr.r.mem)}，放不下。`)}</p>`;}
  return h+`</div>`;
}
/* the full ranked list, closed by default: every layout that fits, each with its cheapest way of saving, and your own plan */
function allHTML(me,best){
  const lay={}; chAll().forEach(e=>{const k=[e.t,e.p,e.z,e.rc].join(); if(!e.fits||!e.r.valid) {if(!(k in lay)) lay[k]=null; return;} if(!lay[k]||e.r.gpuh<lay[k].r.gpuh) lay[k]=e;});
  const per=Object.values(lay).filter(Boolean), nFit=per.length, nOut=Object.keys(lay).length-nFit;
  const svs=new Set(per.map(e=>e.ivl+'/'+e.asy)), s0=per[0];
  let rows=per.slice(), mine=null;
  if(me.fits&&me.r.valid&&!per.some(e=>chSame(e,me))){mine=me; rows.push(me);}
  rows.sort((a,b)=>a.r.gpuh-b.r.gpuh);
  const zc=z=>z===0?tx('none','不用'):String(z);
  const tr=e=>`<tr class="${chMinimal(e)?'ok':''}${chSame(e,me)?' me':''}"${chSame(e,me)?' data-me="1"':''}><td>${e.t}</td><td>${e.p}</td><td>${zc(e.z)}</td><td>${e.rc?tx('yes','是'):tx('no','否')}</td>`+
    `<td class="${e.ok?'':'bad'}">${fDays(e.r.days)}${chMinimal(e)?' ✓':''}</td><td>${fint(Math.round(e.r.gpuh))}</td></tr>`;
  const note=svs.size===1?tx(` Each uses its cheapest way of saving, ${everyIvl(s0.ivl)}, ${s0.asy?'asynchronously':'synchronously'}, which wastes ${pct(s0.r.waste)} of the time.`,`每种都配上最省的存档方式，${everyIvl(s0.ivl)}${s0.asy?'异步':'同步'}存一次档，白费的时间只有 ${pct(s0.r.waste)}。`):'';
  const meLine=!me.fits?tx(` Your plan needs ${fGB(me.r.mem)} per GPU and is not in the list.`,`你的方案每张卡要 ${fGB(me.r.mem)}，放不下，不在表里。`):(mine?tx(' Your plan saves differently, so it has a row of its own.','你的方案存档方式不同，单独占一行。'):'');
  const t2=Object.values(lay).filter(e=>e&&e.t===8&&!e.rc).sort((x,y)=>x.r.gpuh-y.r.gpuh)[0], b=best;
  return `<details class="tcap-all"><summary>${tx(`See all plans (${nFit} layouts that fit)`,`看全部方案（${nFit} 种放得下的切法）`)}</summary><div class="tcap-allb">`+
    `<p class="small">${tx(`They are sorted by GPU hours, fewest first.${note} The other ${nOut} layouts need more than 80 GB per GPU.${meLine}`,
      `下面按 GPU 小时从少到多排。${note}另外 ${nOut} 种切法每张卡超过 80 GB。${meLine}`)}</p>`+
    `<div class="tcap-tblw"><table class="tcap-tbl"><thead><tr><th>${tx('TP','张量<br>并行')}</th><th>${tx('PP','流水线<br>段数')}</th><th>ZeRO</th><th>${tx('Re‑<br>compute','重算')}</th><th>${tx('Days','天数')}</th><th>${tx('GPU hours','GPU<br>小时')}</th></tr></thead><tbody>${rows.map(tr).join('')}</tbody></table></div>`+
    `<p class="small muted tcap-tleg">${tx('✓ fewest GPU hours (within 1%); red: more than 60 days; bold: your plan.','✓ 是 GPU 小时最少的（相差不到 1%），红色是超过了 60 天，加粗的一行是你的方案。')}</p>`+
    tx(`<p class="small">The cheapest plans cut tensor parallelism only ${b.t} ways, use no pipeline, and let ZeRO-3 spread the model states over all 512 GPUs. Each GPU computes ${t2.t/b.t} times as much per layer as with ${t2.t} ways, and each all-reduce moves less data, so the all-reduces take only ${pct(b.r.seg.tp/b.r.step)} of the step instead of ${pct(t2.r.seg.tp/t2.r.step)}.</p>`+
       `<p class="small muted">Note: this lab puts every tensor-parallel all-reduce on the critical path and assumes the GPUs run at 50% of the peak while computing. Real frameworks overlap part of the communication, so measure the top two or three plans on your cluster before you choose.</p>`,
       `<p class="small">最省的方案只切 ${b.t} 份张量并行，不用流水线，用 ZeRO-3 把模型状态分到全部 512 张卡上。每张卡每层算的是切 ${t2.t} 份时的 ${t2.t/b.t} 倍，每次 all-reduce 要传的数据却更少，所以 all-reduce 只占一步的 ${pct(b.r.seg.tp/b.r.step)}，切 ${t2.t} 份时要占 ${pct(t2.r.seg.tp/t2.r.step)}。</p>`+
       `<p class="small muted">注意：这个 Lab 把张量并行的每次 all-reduce 都算在关键路径上，并假设 GPU 算的时候能跑到峰值的 50%。真实的框架会把一部分通信藏到计算后面，所以选之前，把最好的两三种在自己的集群上测一下。</p>`)+
    `</div></details>`;
}
/* opening the full list scrolls its own box (never the page) so that the learner's row is in view */
function showMyRow(dd){const w=$('.tcap-tblw',dd), tr=$('tr[data-me]',dd); if(!w||!tr) return; const th=$('thead',dd), top=tr.offsetTop-(th?th.offsetHeight:0);
  if(top<w.scrollTop||top+tr.offsetHeight>w.scrollTop+w.clientHeight) w.scrollTop=Math.max(0,top-(w.clientHeight-tr.offsetHeight)/2);}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chMe(), best=chBest();
  if(!me) return '';
  let h;
  if(me.ok&&chMinimal(me)) h=tx(`<p class="okline tcap-mt"><b>Goal met: ${dayU(me.r.days)}, about ${roughH(me.r.gpuh)} GPU hours.</b> No plan that meets the goal uses 1% fewer GPU hours.</p>`,
                               `<p class="okline tcap-mt"><b>达成目标：${dayU(me.r.days)}训完，约 ${roughH(me.r.gpuh)} GPU 小时。</b>达标的方案里，没有哪种能再省 1% 的 GPU 小时。</p>`);
  else if(me.ok) h=tx(`<p class="okline tcap-mt"><b>Goal met: ${dayU(me.r.days)}, about ${roughH(me.r.gpuh)} GPU hours.</b> But a cheaper plan exists.</p>`,
                      `<p class="okline tcap-mt"><b>达标了：${dayU(me.r.days)}训完，约 ${roughH(me.r.gpuh)} GPU 小时。</b>不过还有更省的方案。</p>`);
  else {
    const why=!me.fits?tx(`each GPU would need ${fGB(me.r.mem)}, more than its 80 GB`,`每张卡要 ${fGB(me.r.mem)}，超过了 80 GB`):(me.r.valid?tx(`it takes ${dayU(me.r.days)}, more than 60`,`要 ${dayU(me.r.days)}，超过了 60 天`):tx('failures would waste more than half the time','故障会让一半以上的时间白费'));
    h=tx(`<p class="badline tcap-mt"><b>Not yet: ${why}.</b></p>`,`<p class="badline tcap-mt"><b>还没达到：${why}。</b></p>`)+`<p class="small tcap-hint">${gapHint(me)}</p>`+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the recommended plan','直接看推荐方案')}</button>`);
  }
  if(me.ok) chReveal=true;
  if(!chReveal) return h;
  return h+c3HTML(me,best)+allHTML(me,best);
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Estimate how long a 70B run takes, then fit the model in memory, find where the time goes, add failures, and compare with real runs. Start here.`,
            `一共 ${STEPS.length} 个部分。先估一次 70B 训练要多久，再把模型放进显存、找出时间花在哪里、把故障算进来，最后和真实的训练对照。第一次学从这里开始。`),
  sandbox:tx('Five real situations, from 8B on 64 GPUs to 405B on 16,384. Pick one to see the days, the GPU hours, the MFU and the memory of each plan.',
             '五个真实情况，从 64 张卡训练 8B，到 16,384 张卡训练 405B。选一个，看这种计划要几天、多少 GPU 小时、MFU 多少、显存放不放得下。'),
  challenge:tx('Train Llama‑3.1‑70B on 512 H100s with 2T tokens within 60 days, using as few GPU hours as you can.',
               '用 512 张 H100 和 2T 个 token，在 60 天内训练 Llama‑3.1‑70B，GPU 小时越少越好。')
});
const FIDPOP=()=>{const lg=plan(MISSION).logits;
 return tx(
 `<p><b>Formula:</b> every number comes from the formulas of the training labs combined; ${J('math','“Show the math”')} works them out.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Work: 6 FLOPs per parameter per token, 8 with full recomputation (one more forward pass); MFU counts 6. Peak: H100 SXM dense BF16, 989.5 TFLOP/s.</li>`+
 `<li>Memory: 18 bytes per parameter of model states (BF16 weights 2, FP32 gradients 4, optimizer state 12), split by tensor × pipeline parallelism and by the ZeRO stage. ZeRO-3 also holds the BF16 weights of the two layers it gathers, the one being computed and the next. 1F1B keeps ${A.ui.math('\\min(m, p)')} micro-batches on the first stage. GB = 10⁹ bytes; 80 GB per GPU.</li>`+
 `<li>Activations: Megatron-LM’s per-layer formula with sequence parallelism. It already assumes attention’s scores are recomputed in the backward pass (selective recomputation); “every layer” under Recomputation is full recomputation.</li>`+
 `<li>Time: the pipeline’s m + p − 1 slots; 4 tensor-parallel all-reduces per layer and micro-batch (6 with recomputation) over NVLink; the data-parallel traffic that no compute covers.</li>`+
 `<li>Data-parallel traffic between machines: a machine’s 8 GPUs share 400 GB/s of network. An all-gather or reduce-scatter across machines takes about ${A.ui.math('\\frac{\\text{the whole array}}{400\\ \\text{GB/s}}')}, where the whole array is the stage’s BF16 weights or gradients. An all-reduce takes twice that. Inside one machine each GPU moves ${A.ui.math('\\frac{d - 1}{d}')} of its slice over NVLink.</li>`+
 `<li>Failures: wasted share ${A.ui.math('\\approx \\frac{δ}{τ} + \\frac{τ/2 + R}{M}')}, a first-order estimate.</li></ul></div>`+
 `<div><div class="eyebrow">Assumed or simplified</div><ul>`+
 `<li>50% of the peak while computing (an assumption: real kernels reach less than the 80% of a plain matrix multiply); latency, stage-to-stage sends and attention’s s² FLOPs are left out.</li>`+
 `<li>One micro-batch is one 4,096-token sequence unless Settings say otherwise; the embedding table and the LM head are spread evenly over the stages.</li>`+
 `<li>Every transfer is in BF16, 2 bytes per value, as the ZeRO paper assumes. torchtitan and Megatron-LM reduce gradients in FP32 by default, which doubles the bytes of the gradient traffic.</li>`+
 `<li>The logits are left out of the activation memory. Each token has 128,256 FP32 logits, split across the tensor-parallel GPUs. At the default tensor parallel 8 they take ${fixd(lg/1e9,2)} GB per micro-batch.</li>`+
 `<li>ZeRO-3 gathers the weights for every micro-batch; ZeRO-2 sends the gradients after every micro-batch. ZeRO-0/1 sync once per step and overlap only the last backward pass. Megatron-LM also overlaps tensor-parallel communication, which this lab does not.</li>`+
 `<li>Each machine fails once every 64 days, from OPT-175B’s logbook: about 2 of 128 machines a day. BLOOM-176B reported fewer, 1 to 2 GPUs a week of 384. The checkpoint assumptions are the same as in the lab ${labLink('checkpointing')}. A checkpoint is 14 bytes per parameter, written to the shared file system at 10 GB/s. An asynchronous save stops training for 10 s, and a restart takes 10 minutes. All are assumptions.</li>`+
 `<li>No memory is reserved for the CUDA context or communication buffers, so plans close to 80 GB are optimistic. The ZeRO paper counts 16 bytes per parameter (16-bit gradients); this lab keeps the 18 of the other training labs.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>6ND and LLaMA-3 70B’s 6.3 × 10²⁴ FLOPs: ${cite(SRC.book,'scaling book, training.md')}. MFU: ${cite(SRC.mlePerf,'ml-engineering, training/performance')}; 80% matmul ceiling: ${cite(SRC.mleAcc,'ml-engineering, compute/accelerator')}.</li>`+
 `<li>18 bytes and ZeRO-1’s ${A.ui.math('6 + \\frac{12}{d}')}: ${cite(SRC.megDist,'Megatron-LM dist_optimizer.md')}; activations: ${cite(SRC.megMem,'theoretical_memory_usage.py')}; all-reduce cost: ${cite(SRC.nccl,'nccl-tests')}; NVLink per GPU, the network per machine and collectives across machines, Llama 3’s layout and 16M-token batch: ${cite(SRC.bookGpu,'scaling book, gpus.md')}. Gradients reduced in FP32 by default: ${cite(SRC.ttcfg,'torchtitan configs.py')}, ${cite(SRC.megarg,'Megatron-LM arguments.py')}.</li>`+
 `<li>GPU hours: ${cite(SRC.card,'Llama 3.1 model card')}. MFU of real runs: ${cite(SRC.megReadme,'Megatron-LM README')}, ${cite(SRC.titan,'torchtitan benchmarks')}, ${cite(SRC.foundry,'llm-foundry benchmarks')}. Layouts: ${cite(SRC.bridge,'Megatron-Bridge performance archive')}.</li>`+
 `<li>Failures and checkpoints: ${cite(SRC.opt,'OPT-175B chronicles')}, ${cite(SRC.bloom,'BLOOM-176B training README')}, ${cite(SRC.bloomBlog,'BLOOM blog')}, ${cite(SRC.mleFt,'ml-engineering, fault tolerance')}, ${cite(SRC.dcpAsync,'PyTorch async checkpoint recipe')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>所有数字都由几个训练 Lab 的公式组合算出，${J('math','“展开公式”')}里有算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>运算量：每个 token、每个参数 6 次。整层重算时多算一遍前向，是 8 次，MFU 仍只数 6 次。峰值按 H100 SXM 稠密 BF16 的 989.5 TFLOP/s 算。</li>`+
 `<li>显存：模型状态每个参数 18 字节，其中 BF16 权重 2 字节，FP32 梯度 4 字节，优化器状态 12 字节。模型状态按“张量并行 × 流水线段数”和 ZeRO 阶段分开存。ZeRO-3 还要放临时收集的两层 BF16 权重，正在算的这一层和提前收集的下一层。1F1B 让第 1 段同时存 ${A.ui.math('\\min(m, p)')} 个 micro-batch。GB 是 10⁹ 字节，每张卡 80 GB。</li>`+
 `<li>激活：用 Megatron-LM 带序列并行的每层公式。这个公式已经假设 attention 的分数在反向时重算（部分重算）。设置里的“每层都重算”是整层重算。</li>`+
 `<li>时间：流水线一步有 m + p − 1 格。每层每个 micro-batch 要做 4 次张量并行的 all-reduce（重算时 6 次），走 NVLink。数据并行要传的数据，算的时候藏不住的部分就是等待。</li>`+
 `<li>机器之间的数据并行：一台机器的 8 张卡共用 400 GB/s 的网络带宽。跨机器做一次 all-gather 或 reduce-scatter，大约要 ${A.ui.math('\\frac{\\text{整份数据}}{400\\ \\text{GB/s}}')}。整份数据是这一段流水线的全部 BF16 权重或梯度。all-reduce 要做两遍。同一台机器里走 NVLink，每张卡传自己那份的 ${A.ui.math('\\frac{d - 1}{d}')}。</li>`+
 `<li>故障：白费的比例 ${A.ui.math('\\approx \\frac{δ}{τ} + \\frac{τ/2 + R}{M}')}，是一阶近似。</li></ul></div>`+
 `<div><div class="eyebrow">假设和简化</div><ul>`+
 `<li>GPU 算的时候能跑到峰值的 50%。这是假设，真实的 kernel 达不到单纯矩阵乘法的 80%。没有算延迟、流水线段之间的传输和 attention 随长度平方增长的运算。</li>`+
 `<li>除非自由调参里改了，一个 micro-batch 就是一条 4,096 个 token 的序列。embedding 表和 LM head 平均分到各段。</li>`+
 `<li>所有传输都按 BF16 算，每个值 2 字节，和 ZeRO 论文的假设一样。torchtitan 和 Megatron-LM 默认用 FP32 做梯度的归约，梯度要传的字节数会翻倍。</li>`+
 `<li>激活里没有算 logits。每个 token 有 128,256 个 FP32 的 logits，按张量并行切开。默认切 8 份时，每个 micro-batch 的 logits 占 ${fixd(lg/1e9,2)} GB。</li>`+
 `<li>ZeRO-3 每个 micro-batch 都重新收集权重，ZeRO-2 每个 micro-batch 算完就发梯度。不分和 ZeRO-1 每步只同步一次，只和最后一个反向重叠。Megatron-LM 还能让张量并行的通信和计算重叠，这个 Lab 没有算。</li>`+
 `<li>每台机器平均 64 天坏一次，这是按 OPT-175B 的日志算的：128 台机器每天约坏 2 台。BLOOM-176B 坏得少一些，384 张卡每周坏 1 到 2 张。存档的假设和${LQ('checkpointing')}一样。存档每个参数 14 字节，每秒往共享文件系统写 10 GB。异步存档每次让训练停 10 秒，重启要 10 分钟。这些都是假设。</li>`+
 `<li>没有给 CUDA 上下文和通信缓冲留显存，所以接近 80 GB 的方案偏乐观。ZeRO 的论文按每个参数 16 字节算（梯度用 16 位），这个 Lab 和其他训练 Lab 一样用 18 字节。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>6ND 和 LLaMA-3 70B 的 6.3 × 10²⁴ 次运算，见 ${cite(SRC.book,'scaling book 的 training.md')}。MFU 见 ${cite(SRC.mlePerf,'ml-engineering 的 training/performance')}。矩阵乘法约 80% 的上限，见 ${cite(SRC.mleAcc,'ml-engineering 的 compute/accelerator')}。</li>`+
 `<li>18 字节和 ZeRO-1 的 ${A.ui.math('6 + \\frac{12}{d}')}，见 ${cite(SRC.megDist,'Megatron-LM 的 dist_optimizer.md')}。激活见 ${cite(SRC.megMem,'theoretical_memory_usage.py')}。all-reduce 的耗时见 ${cite(SRC.nccl,'nccl-tests')}。每张卡的 NVLink、每台机器的网络和跨机器的通信，以及 Llama 3 的切法和每步 16M 个 token，见 ${cite(SRC.bookGpu,'scaling book 的 gpus.md')}。梯度默认用 FP32 归约，见 ${cite(SRC.ttcfg,'torchtitan 的 configs.py')} 和 ${cite(SRC.megarg,'Megatron-LM 的 arguments.py')}。</li>`+
 `<li>GPU 小时见 ${cite(SRC.card,'Llama 3.1 的模型卡')}。真实训练的 MFU 见 ${cite(SRC.megReadme,'Megatron-LM 的 README')}、${cite(SRC.titan,'torchtitan 的测试')}和 ${cite(SRC.foundry,'llm-foundry 的测试')}。切法见 ${cite(SRC.bridge,'Megatron-Bridge 的性能记录')}。</li>`+
 `<li>故障和存档：${cite(SRC.opt,'OPT-175B 的训练日志')}、${cite(SRC.bloom,'BLOOM-176B 的训练说明')}、${cite(SRC.bloomBlog,'BLOOM 的博客')}、${cite(SRC.mleFt,'ml-engineering 的 fault tolerance')}、${cite(SRC.dcpAsync,'PyTorch 异步存档教程')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最后核对：2026-10-04。原型，尚未经过专家审阅。</p></div></div>`);};
const PRIPOP=()=>tx(`<p>This lab puts the training labs together. The 18 bytes per parameter come from ${labLink('training-memory')}; ZeRO and FSDP from ${labLink('parallelism')}; tensor parallelism, the pipeline bubble and 1F1B from ${labLink('model-parallel')}; 6ND from ${labLink('scaling-law')}; the H100’s peak from ${labLink('roofline')}. Saving and resuming are covered in ${labLink('checkpointing')}.</p>`,
  `<p>这个 Lab 把几个训练 Lab 合在一起。每个参数 18 字节见${LQ('training-memory')}；ZeRO 和 FSDP 见${LQ('parallelism')}；张量并行、流水线的 bubble 和 1F1B 见${LQ('model-parallel')}；6ND 见${LQ('scaling-law')}；H100 的峰值见${LQ('roofline')}。怎么存档、怎么接着训，见${LQ('checkpointing')}。</p>`);
const OUTCOME=()=>tx('estimate how many days a large run takes, pick a layout that fits and runs fast, and say where the time goes, failures included.','估出一次大规模训练要几天，选一种放得下又跑得快的切法，并说清时间花在了哪里，包括故障。');
const EXITS=()=>{
  const r1=plan({...MISSION}), r2=plan({...MISSION,G:2048}), f1=plan({...MISSION}), f2=plan({...MISSION,node:32});
  const tipB=tx(`On 1,024 GPUs each pipeline gets ${r1.m} micro-batches, on 2,048 only ${r2.m}. The bubble grows from ${pct(r1.seg.bubble/r1.step)} to ${pct(r2.seg.bubble/r2.step)} of a step.`,
                `1,024 张卡时，每条流水线分到 ${r1.m} 个 micro-batch，2,048 张时只有 ${r2.m} 个。bubble 从一步的 ${pct(r1.seg.bubble/r1.step)} 涨到 ${pct(r2.seg.bubble/r2.step)}。`),
        tipW=tx(`256 machines lose one every ${fH(r2.M)} on average. The wasted time grows from ${pct(r1.waste)} to ${pct(r2.waste)}.`,
                `256 台机器平均每 ${fH(r2.M)}就坏一台。白费的时间从 ${pct(r1.waste)} 涨到 ${pct(r2.waste)}。`);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does doubling the GPUs not halve the training time?','为什么卡数翻倍，训练时间却减不到一半？'),
   o:[tx('The step’s sequences stay the same, so each pipeline gets half the micro-batches and the bubble grows. Twice the machines also fail twice as often.','每步的序列数不变，每条流水线分到的 micro-batch 少了一半，bubble 变大。机器多了一倍，故障也勤了一倍。'),
      tx('Each GPU now gets half the sequences, so its matrix multiplications are smaller and it sits partly idle. The GPUs no longer reach 50% of the peak while computing.','卡多了一倍，每张卡算的序列少了一半，矩阵乘法变小，GPU 有一部分时间空着。算的时候，它们到不了峰值的 50%。'),
      tx('Each GPU now syncs gradients with twice as many GPUs, so it sends twice the data. The network becomes the bottleneck.','卡多了一倍，每张卡要和两倍的卡同步梯度，要传的数据也翻倍。网络成了瓶颈。')],a:0,
   n:[null,tx('A micro-batch is still one 4,096-token sequence, so the matrices are as large as before and the GPUs still run at 50% of the peak. What shrinks is the number of micro-batches each pipeline gets.','一个 micro-batch 还是一条 4,096 个 token 的序列，矩阵和原来一样大，GPU 算的时候照样能跑到峰值的 50%。变少的是每条流水线分到的 micro-batch 个数。'),
     tx('In a ring all-reduce each GPU sends about twice its data whatever the number of GPUs, '+A.ui.math('\\frac{2(n - 1)}{n}')+' times. What grows is the bubble and the failures.','环形 all-reduce 里，每张卡要发的数据约是它那份的 '+A.ui.math('\\frac{2(n - 1)}{n}')+' 倍，和卡数几乎无关。变大的是 bubble 和故障。')],
   why:tx('The work is fixed. What scales badly is the waiting in each step and the failures each day.','总运算量不变。变差的是每一步的等待和每天的故障。')},
  {q:tx('Why is MFU far below 100% even when everything fits?','为什么就算全都放得下，MFU 也远到不了 100%？'),
   o:[tx('Even while computing, the GPUs do not reach the peak. On top of that, each GPU waits in the bubble and for the all-reduces.','就算在算，GPU 也到不了峰值。此外，每张卡还要在 bubble 里空等，还要等 all-reduce 做完。'),
      tx('Training reads the weights from memory all the time. MFU is measured against the memory bandwidth, so all that reading lowers it.','训练一直在从显存读权重。MFU 是按显存带宽算的，读得这么多，MFU 就低。'),
      tx('Recomputation adds one more forward pass per token. MFU counts that pass twice, so even without waiting it cannot reach 100%.','重算让每个 token 多算一遍前向。MFU 把这遍前向算了两次，所以就算不等，也到不了 100%。')],a:0,
   n:[null,tx('MFU divides the model’s FLOPs by GPUs × peak FLOP/s × time; memory bandwidth is not in it.','MFU 是模型的运算量除以“卡数 × 峰值 FLOP/s × 时间”，里面没有显存带宽。'),
     tx('MFU counts only the model’s own 6 operations per parameter per token. Recomputation makes the step longer and so lowers MFU, but the layouts without it stay far below 100% too.','MFU 只数模型本身每个 token、每个参数的 6 次运算。重算让一步变长，MFU 会因此变低，可不重算的切法，MFU 也远到不了 100%。')],
   why:tx('MFU is the model’s own arithmetic as a share of the peak over the same time. Every idle moment and every repeated operation lowers it.','MFU 是模型本身的运算占同一段时间里峰值算力的比例。每一段空等、每一遍重复的运算，都会让它变低。')}],
 transfer:[
  {q:tx(`1,024 GPUs train with tensor parallel 8, 8 pipeline stages and ZeRO-1, saving every 3 hours synchronously. The run takes ${fDays(r1.days)} days. Now the GPUs double to 2,048, still with 1,024 sequences per step. About how many days does it take?`,
        `1,024 张卡用张量并行 8 份、流水线 8 段和 ZeRO-1 训练，每 3 小时同步存一次档，要训 ${fDays(r1.days)} 天。现在卡数翻倍到 2,048 张，每步还是 1,024 条序列。大约要训多少天？`),
   o:[tx(`About ${fint(Math.round(r1.days/2))} (half)`,`约 ${fint(Math.round(r1.days/2))} 天（正好一半）`),tx(`About ${fint(Math.round(r2.days))}`,`约 ${fint(Math.round(r2.days))} 天`),tx(`About ${fDays(r1.days)} (no change)`,`约 ${fDays(r1.days)} 天（不变）`)],a:1,
   why:tx(`About ${fint(Math.round(r2.days))} days, only a third less. The step’s sequences stay the same, so ${DEF('each pipeline gets half the micro-batches',tipB)} and the bubble almost doubles. Twice the machines also fail twice as often, so ${DEF('the wasted time',tipW)} almost doubles too.`,
           `约 ${fint(Math.round(r2.days))} 天，只少了三分之一。每步的序列数没变，卡多了一倍，${DEF('每条流水线分到的 micro-batch 少了一半',tipB)}，bubble 几乎翻倍。机器也多了一倍，故障勤了一倍，${DEF('白费的时间',tipW)}也几乎翻倍。`),go:()=>openCase({...MISSION,G:2048}),goLabel:tx('Open it in Scenarios','在应用场景里打开')},
  {q:tx(`1,024 GPUs save a checkpoint every 3 hours, synchronously. When each machine fails once every 64 days on average, ${pct(f1.waste)} of the time is wasted. If each machine failed once every 32 days, about how much would be wasted?`,
        `1,024 张卡每 3 小时同步存一次档。每台机器平均 64 天坏一次时，白费 ${pct(f1.waste)} 的时间。如果每台机器 32 天就坏一次，大约会白费多少？`),
   o:[tx(`About ${pct0(f1.waste)} (no change)`,`约 ${pct0(f1.waste)}（不变）`),tx(`About ${pct0(f2.waste)}`,`约 ${pct0(f2.waste)}`),tx(`About ${pct0(4*f1.waste)}`,`约 ${pct0(4*f1.waste)}`)],a:1,
   why:tx(`${pct(f2.waste)}, almost twice as much. The saves cost the same, but the job fails every ${fH(f2.M)}, so the work redone and the restarts double:`,
           `${pct(f2.waste)}，差不多翻倍。存档的开销没变，可任务每 ${fH(f2.M)}就出一次事，白算的和重启的时间都翻倍：`)+
       M_(`\\frac{τ/2 + R}{M} = \\frac{1.5 + 1/6}{${fixd(f2.M,0)}} \\approx ${fixd((f2.wRedo+f2.wRestart)*100,1)}\\%`),go:()=>openCase({...MISSION,node:32}),goLabel:tx('Open it in Scenarios','在应用场景里打开')}],
 practice:[
  {id:'distributed-training',d:tx('A system design: train a model that does not fit on one accelerator. Parts 1–3 and 6(a): compute and time from 6ND, a memory plan per accelerator, a tensor, pipeline and data-parallel layout on the right links, and a checkpoint interval from the cluster’s failure rate.','系统设计：为单卡放不下的模型设计训练方案。第 1–3 部分和 6(a)：用 6ND 估计算量和时间，每张卡的显存方案，把张量、流水线和数据并行放到合适的链路上，再按集群的故障率定存档间隔。')}]};
};
function openCase(c){setMode('sandbox'); byLearner(()=>setState({...c})); A.ui.reveal($('#tcap-stage',app));}

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Training capstone','训练综合挑战'),title:l?L(l.q):tx('How do you plan the memory, parallelism and time of a 70B run?','训练一个 70B 模型，怎么规划显存、并行和时间？'),
    minutes:20,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: the training labs','前置知识：前面几个训练 Lab'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage tcap-stage" id="tcap-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="tcap-st-title"></div><div class="meter-sub" id="tcap-st-sub"></div></div><div class="stage-actions"><span class="status" id="tcap-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph tcap-graph" id="tcap-graph">`+
    `<div class="tcap-phone" id="tcap-phone"></div>`+
    `<div class="tcap-top" id="tcap-top"><div class="tcap-clbox" id="tcap-clbox"><div class="tcap-cap" id="tcap-cl-cap"></div><div class="tcap-cl" id="tcap-cl"></div><div class="tcap-cl-note small muted" id="tcap-cl-note"></div></div>`+
     `<div class="tcap-membox" id="tcap-membox"><div class="tcap-cap" id="tcap-mem-cap"></div><div class="tcap-mem" id="tcap-mem"></div></div></div>`+
    `<div class="tcap-barbox" id="tcap-stepbox"><div class="tcap-cap" id="tcap-step-cap"></div><div class="tcap-bar" id="tcap-step"></div><ul class="tcap-list" id="tcap-step-list"></ul></div>`+
    `<div class="tcap-barbox" id="tcap-runbox"><div class="tcap-cap" id="tcap-run-cap"></div><div class="tcap-bar" id="tcap-run"></div><ul class="tcap-list" id="tcap-run-list"></ul></div>`+
    `<div class="tcap-barbox" id="tcap-realbox"><div class="tcap-cap" id="tcap-real-cap"></div><div class="tcap-real" id="tcap-real"></div></div>`+
   `</div>`+
   `<div class="metrics tcap-metrics" id="tcap-metrics"></div>`+
   `<details class="math" id="tcap-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="tcap-mathbody"></div></details></div>`;
}
const NOTE={
 u0:()=>tx('An assumed value: the share of the H100’s 989.5 TFLOP/s peak the GPU reaches while it computes. A plain BF16 matrix multiply reaches about 80% (ml-engineering’s measurement); attention, normalization and smaller kernels reach less. This lab takes 50%.',
          '这是假设值：GPU 在算的时候，能跑到 H100 峰值 989.5 TFLOP/s 的多少。单纯的 BF16 矩阵乘法约能到 80%（ml-engineering 实测），attention、归一化和较小的 kernel 达不到。这个 Lab 取 50%。'),
 bw:()=>tx('Nominal values from the scaling book’s H100 cluster. Inside a machine each GPU has 450 GB/s of NVLink each way. Between machines, a machine’s 8 GPUs share 400 GB/s of InfiniBand, 8 links of 400 Gb/s. A data-parallel transfer across machines takes about the whole array ÷ 400 GB/s. The Ethernet option gives a machine 100 Gb/s, 12.5 GB/s, a thirty-second of InfiniBand’s. Measured collectives reach less.',
          '这是标称值，来自 scaling book 对 H100 集群的描述。机器里，每张卡的 NVLink 每个方向 450 GB/s。机器之间，一台机器的 8 张卡共用 400 GB/s 的网络带宽，也就是 8 条 400 Gb/s 的 InfiniBand。跨机器的数据并行传一遍，大约要“整份数据 ÷ 400 GB/s”。选“普通以太网”时，一台机器只有 100 Gb/s，也就是 12.5 GB/s，是 InfiniBand 的 1/32。实测的通信达不到这些数。'),
 ck:()=>tx(`Assumed values, the same as in the lab ${labLink('checkpointing')}. A checkpoint holds 14 bytes per parameter, 0.99 TB for 70B. The job writes it to the shared file system at 10 GB/s, so a synchronous save stops training for about 99 s. BLOOM-176B wrote its 2.3 TB checkpoint from 384 GPUs in about 40 s, about 57 GB/s, but shared file systems differ a lot, so this lab takes a slower 10 GB/s. An asynchronous save stops training for 10 s, the time to copy the state into CPU memory. A restart (swap the machine, load the checkpoint) takes 10 minutes; in practice it depends on whether a spare machine is ready.`,
          `这些是假设值，和${LQ('checkpointing')}一样。存档每个参数 14 字节，70B 一共 0.99 TB。整个任务每秒往共享文件系统写 10 GB，所以同步存一次档，训练要停约 99 秒。BLOOM-176B 用 384 张卡约 40 秒写完 2.3 TB 的存档，每秒约 57 GB。可各家的共享文件系统差别很大，这里取慢一些的每秒 10 GB。异步存档每次让训练停 10 秒，这是把状态拷到 CPU 内存的时间。重启（换机器、读回存档）按 10 分钟算。真实要多久，要看有没有现成的备用机器。`)
};
const segRow=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">${A.ui.seg({id,key,label,options:opts,value:String(val),note})}</div>`;
const GS=[64,128,256,512,1024,2048,4096,8192,16384];
const DS=[1e12,2e12,15e12];
function controlsHTML(){
  return `<aside class="panel controls" id="tcap-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="tcap-lock" hidden></p>`+
   segRow('tcap-ctl-model','model',tx('Model','模型'),[['8b','8B'],['70b','70B'],['405b','405B']],S.model)+
   segRow('tcap-ctl-D','D',tx('Training tokens','训练的 token 数'),DS.map(v=>[String(v),`${v/1e12}T`]),S.D)+
   segRow('tcap-ctl-G','G',tx('H100 GPUs (8 per machine)','H100 的卡数（每台机器 8 张）'),GS.map(v=>[String(v),fint(v)]),S.G)+
   segRow('tcap-ctl-B','B',tx('Sequences per step (4,096 tokens each)','每步多少条序列（每条 4,096 个 token）'),[['1024','1,024'],['4096','4,096']],S.B)+
   segRow('tcap-ctl-t','t',tx('Tensor parallel: GPUs per layer','张量并行：一层切到几张卡'),[1,2,4,8].map(v=>[String(v),String(v)]),S.t)+
   segRow('tcap-ctl-p','p',tx('Pipeline stages','流水线段数'),[1,2,4,8,16].map(v=>[String(v),String(v)]),S.p)+
   segRow('tcap-ctl-z','z',tx('ZeRO over the data-parallel copies','数据并行上的 ZeRO'),[['0',tx('none','不分')],['1','ZeRO-1'],['2','ZeRO-2'],['3','ZeRO-3']],S.z)+
   segRow('tcap-ctl-rc','rc',tx('Recomputation','重算'),[['0',tx('off','不重算')],['1',tx('every layer','每层都重算')]],S.rc)+
   segRow('tcap-ctl-b','b',tx('Sequences per micro-batch','每个 micro-batch 几条序列'),[1,2,4,8].map(v=>[String(v),String(v)]),S.b)+
   segRow('tcap-ctl-ivl','ivl',tx('Save a checkpoint every','多久存一次档'),[['0.25',tx('15 min','15 分钟')],['1',tx('1 hour','1 小时')],['3',tx('3 hours','3 小时')]],S.ivl)+
   segRow('tcap-ctl-asy','asy',tx('Saving','存档方式'),[['0',tx('sync','同步')],['1',tx('async','异步')]],S.asy)+
   segRow('tcap-ctl-net','net',tx('Network between machines','机器之间的网络'),[['ib','InfiniBand'],['eth',tx('Ethernet','普通以太网')]],S.net)+
   segRow('tcap-ctl-node','node',tx('Each machine fails once every','每台机器平均多久坏一次'),[['32',tx('32 days','32 天')],['64',tx('64 days (OPT)','64 天（OPT）')],['224',tx('224 days (BLOOM)','224 天（BLOOM）')]],S.node)+
   `<div class="tcap-grp lockable" id="tcap-ctl-as"><div class="tcap-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><div class="lbl">${tx('Share of the peak while computing','算的时候能到峰值的多少')} <b class="mono">50%</b></div><p class="ctl-note">${NOTE.u0()}</p></div>`+
    `<div class="ctl"><div class="lbl">${tx('NVLink per GPU / network per machine','每张卡的 NVLink / 每台机器的网络')} <b class="mono">450 / 400 GB/s</b></div><p class="ctl-note">${NOTE.bw()}</p></div>`+
    `<div class="ctl"><div class="lbl">${tx('Checkpoints and restarts','存档和重启')} <b class="mono">${tx('10 GB/s, 10 s, 10 min','10 GB/s，10 秒，10 分钟')}</b></div><p class="ctl-note">${NOTE.ck()}</p></div></div>`+
   `<div class="lockable" id="tcap-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`PEAK, U0, SEQ = 989.5e12, 0.5, 4096     # H100 BF16; 50% assumed
def plan(N, L, h, I, D, G, t, p, B=1024, nv=450e9, net=400e9):
    d = G // (t * p); m = B // d          # DP copies, micro-batches
    pg = N / (t * p)                       # parameters per GPU
    states = pg * (6 + 12 / d)             # ZeRO-1: optimizer state split
    act = min(m, p) * L / p * SEQ * h * (18 + 4 * I / h) / t
    comp = 6 * N / p * SEQ / t / (PEAK * U0)       # one micro-batch
    ar = 2 * (t - 1) / t * SEQ * h * 2 / nv        # one all-reduce
    per = comp + 4 * L / p * ar
    if t * d > 8: one = 2 * N / p / net            # one pass across machines
    else: one = (d - 1) / d * 2 * pg / nv          # one pass inside a machine
    sync = 2 * one if d > 1 else 0                 # all-reduce: 2 passes
    step = (m + p - 1) * per + max(0, sync - per * 2 / 3)
    mfu = 6 * N * B * SEQ / (G * PEAK * step)
    days = D / (B * SEQ) * step / 86400
    return (states + act) / 1e9, mfu, days

def wasted(tau, delta, M, R=1 / 6):        # hours
    return delta / tau + (tau / 2 + R) / M

gb, mfu, days = plan(70.55e9, 80, 8192, 28672, 15e12, 1024, 8, 8)
M = 64 * 24 / (1024 / 8)                    # job MTBF, hours
w = wasted(0.25, 10 / 3600, M)              # async save: 10 s
print(f'{gb:.1f} GB/GPU  MFU {mfu:.1%}  {days / (1 - w):.0f} days')`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="tcap-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="tcap-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The planner in plain Python, for the guided plan with ZeRO-1: memory per GPU, MFU, and days including failures with saving every 15 minutes asynchronously.','用纯 Python 写的规划器，按分步引导里的 ZeRO-1 计划算：每张卡的显存、MFU，以及每 15 分钟异步存档、算上故障以后的天数。')}</p><pre class="code" id="tcap-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="tcap-prod-card"><h3>${A.t('cProd')}</h3><ul class="tcap-prod">`+
   tx(`<li>Megatron-LM sets the layout with <code>--tensor-model-parallel-size</code>, <code>--pipeline-model-parallel-size</code> and <code>--use-distributed-optimizer</code> (ZeRO-1, ${cite(SRC.megDist,'dist_optimizer.md')}); <code>--recompute-granularity</code> picks full or selective recomputation. Its FLOPs counter in ${cite(SRC.megFlops,'<code>training.py</code>')} counts each GEMM three times (forward, weight gradient, input gradient).</li>`+
      `<li>torchtitan combines FSDP, tensor, pipeline and context parallelism and publishes its Llama 3.1 layouts and throughput (${cite(SRC.titan,'benchmarks')}); NVIDIA’s recipes list TP, PP, CP and virtual pipeline sizes per model and GPU count (${cite(SRC.bridge,'Megatron-Bridge archive')}).</li>`+
      `<li>Model cards report training cost in GPU hours: Llama 3.1 lists 1.46M (8B), 7.0M (70B) and 30.84M (405B) H100 hours, 39.3M in all (${cite(SRC.card,'MODEL_CARD.md')}).</li>`,
      `<li>Megatron-LM 用 <code>--tensor-model-parallel-size</code>、<code>--pipeline-model-parallel-size</code> 和 <code>--use-distributed-optimizer</code>（ZeRO-1，${cite(SRC.megDist,'dist_optimizer.md')}）设定切法；<code>--recompute-granularity</code> 选全部重算还是部分重算。它在 ${cite(SRC.megFlops,'<code>training.py</code>')} 里数运算量时，每个矩阵乘法数三遍（前向、权重的梯度、输入的梯度）。</li>`+
      `<li>torchtitan 把 FSDP、张量、流水线和上下文并行组合起来，并公开了 Llama 3.1 的切法和吞吐（${cite(SRC.titan,'benchmarks')}）；NVIDIA 的配方按模型和卡数列出张量、流水线、上下文并行和虚拟流水线的大小（${cite(SRC.bridge,'Megatron-Bridge 的记录')}）。</li>`+
      `<li>模型卡用 GPU 小时报告训练成本：Llama 3.1 列出 8B 用了 1.46M、70B 用了 7.0M、405B 用了 30.84M H100 小时，一共 39.3M（${cite(SRC.card,'MODEL_CARD.md')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.p2104,'Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM')}<span>${tx('Narayanan et al., 2021 (arXiv 2104.04473): combining tensor, pipeline and data parallelism.','Narayanan 等，2021（arXiv 2104.04473）：把张量、流水线和数据并行组合起来。')}</span></li>`+
   `<li>${cite(SRC.p1910,'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models')}<span>${tx('Rajbhandari et al., 2019 (arXiv 1910.02054): the three ZeRO stages.','Rajbhandari 等，2019（arXiv 1910.02054）：ZeRO 的三个阶段。')}</span></li>`+
   `<li>${cite(SRC.p2205,'Reducing Activation Recomputation in Large Transformer Models')}<span>${tx('Korthikanti et al., 2022 (arXiv 2205.05198): the activation formula and selective recomputation.','Korthikanti 等，2022（arXiv 2205.05198）：激活的公式和部分重算。')}</span></li>`+
   `<li>${cite(SRC.p2407,'The Llama 3 Herd of Models')}<span>${tx('Llama Team, 2024 (arXiv 2407.21783): the 16K-GPU training run. Cited by title only; no number on this page comes from it.','Llama 团队，2024（arXiv 2407.21783）：16K 张卡的训练。只引标题，页面上的数字都不出自它。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="tcap-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="tcap-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="tcap-app m2" id="tcap-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="tcap-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many days the plan takes.','选一个场景，看看这种计划要训多少天。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="tcap-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
const chSeg=(id,key,label,opts,val)=>A.ui.seg({id,key,label,options:opts,value:String(val),inline:true});
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const okP=p=>512%(ch.t*p)===0;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Train 70B on 512 H100s within 60 days','60 天内用 512 张 H100 训完 70B')}</h3>`+
   `<p>${tx('You are to train Llama‑3.1‑70B on 512 H100s (64 machines of 8) with 2T tokens, and finish within 60 days. Choose tensor parallelism, pipeline parallelism, the ZeRO stage over the data-parallel copies, whether to recompute, and how often to save a checkpoint, synchronously or not. Every GPU must fit in 80 GB, the run must finish in time, and it should use as few GPU hours as possible.',
            '你要用 512 张 H100（64 台机器，每台 8 张）和 2T 个 token 训练 Llama‑3.1‑70B，要在 60 天内训完。请选张量并行、流水线并行、数据并行上的 ZeRO 阶段、要不要重算，以及多久存一次档、同步还是异步。每张卡的显存要放得下，按时训完，GPU 小时越少越好。')}</p>`+
   `<ul class="tcap-asm small"><li>${tx('Each step trains 1,024 sequences of 4,096 tokens. A micro-batch is one sequence.','每步训练 1,024 条序列，每条 4,096 个 token。一个 micro-batch 就是一条序列。')}</li><li>${tx('Between machines, InfiniBand: a machine’s 8 GPUs share 400 GB/s.','机器之间是 InfiniBand，一台机器的 8 张卡共用 400 GB/s 的网络带宽。')}</li><li>${tx('Each machine fails once every 64 days on average.','每台机器平均 64 天坏一次。')}</li><li>${tx('GPU hours = GPUs × hours, all 512 GPUs counted together: 512 GPUs for one day are 12,288 GPU hours.','GPU 小时 = 用的 GPU 数 × 小时数，512 张卡加在一起算：512 张卡跑 1 天是 12,288 GPU 小时。')}</li></ul>`+
   chSeg('tcap-ic-cht','cht',tx('Tensor parallel: GPUs per layer','张量并行：一层切到几张卡'),CH.Ts.map(v=>[String(v),String(v)]),ch.t)+
   chSeg('tcap-ic-chp','chp',tx('Pipeline stages','流水线段数'),CH.Ps.map(v=>[String(v),String(v),!okP(v)]),ch.p)+
   chSeg('tcap-ic-chz','chz',tx('ZeRO over the data-parallel copies','数据并行上的 ZeRO'),[['0',tx('none','不分')],['1','ZeRO-1'],['2','ZeRO-2'],['3','ZeRO-3']],ch.z)+
   chSeg('tcap-ic-chrc','chrc',tx('Recomputation','重算'),[['0',tx('off','不重算')],['1',tx('every layer','每层都重算')]],ch.rc)+
   chSeg('tcap-ic-chivl','chivl',tx('Save a checkpoint every','多久存一次档'),[['0.25',tx('15 min','15 分钟')],['1',tx('1 hour','1 小时')],['3',tx('3 hours','3 小时')]],ch.ivl)+
   chSeg('tcap-ic-chasy','chasy',tx('Saving','存档方式'),[['0',tx('sync','同步')],['1',tx('async','异步')]],ch.asy)+
   `<div class="goal mono tcap-goal" id="tcap-ch-g1"><span>${tx('Memory per GPU','每张卡的显存')}</span><span><b id="tcap-ch-mem"></b>${tx(' (goal: at most 80 GB)','（目标：不超过 80 GB）')}</span></div>`+
   `<div class="goal mono tcap-goal" id="tcap-ch-g2"><span>${tx('Days to finish','要训的天数')}</span><span><b id="tcap-ch-days"></b>${tx(' (goal: at most 60)','（目标：不超过 60 天）')}</span></div>`+
   `<div class="goal mono tcap-goal" id="tcap-ch-g3"><span>${tx('GPU hours','GPU 小时')}</span><span><b id="tcap-ch-gh"></b>${tx(' (fewer is better)','（越少越好）')}</span></div>`+
   submit+`<div id="tcap-ch-res" class="tcap-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},total:{en:'Try another MFU to continue.',zh:'换一个 MFU 试试，才能继续。'},
    memory:{en:'Find a split that fits to continue.',zh:'找到一种放得下的切法，才能继续。'},time:{en:'Try two more layouts to continue.',zh:'再试两种切法，才能继续。'},failures:{en:'Try “Every 15 minutes, async” to continue.',zh:'试一下“每 15 分钟异步存一次”，才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#tcap-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
const LVSET={'tcap-ic-lv2':[()=>u2,()=>tried2,lv2Res,v=>+v],'tcap-ic-lv3':[()=>lv3,()=>tried3,lv3Res,v=>v],'tcap-ic-lv4':[()=>lv4,()=>tried4,lv4Res,v=>v],'tcap-ic-lv5':[()=>lv5,()=>tried5,lv5Res,v=>v]};
function updateRail(){
  const rail=$('#tcap-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    Object.keys(LVSET).forEach(id=>{const [cur,tr,res,cv]=LVSET[id]; $$(`#${id} [data-lever]`,rail).forEach(b=>{const v=cv(b.dataset.lever); b.setAttribute('aria-pressed',String(v===cur())); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tr().has(v)||v===cur()?res(v):'';});});
    if(box){const html=s.obs?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#tcap-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chMe(), mm=$('#tcap-ch-mem',rail); if(!mm||!e) return;
    mm.textContent=fGB(e.r.mem); $('#tcap-ch-days',rail).textContent=e.r.valid?dayU(e.r.days):'—'; $('#tcap-ch-gh',rail).textContent=fGpuh(e.r.gpuh);
    const g1=$('#tcap-ch-g1',rail), g2=$('#tcap-ch-g2',rail), dOk=e.r.valid&&e.r.days<=CH.days;
    g1.classList.toggle('met',e.fits); g1.classList.toggle('tcap-over',!e.fits); g2.classList.toggle('met',dOk); g2.classList.toggle('tcap-over',!dOk);
    const res=$('#tcap-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);
      const dd=$('details.tcap-all',res); if(dd){dd.open=allOpen; dd.addEventListener('toggle',()=>{allOpen=dd.open; if(dd.open) showMyRow(dd);}); if(allOpen) showMyRow(dd);}}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
/* which boxes each part shows: the cluster as machines or as a layout, the memory column, the step bar, the run bar, real runs */
function secs(){
  if(mode!=='guided') return {cl:'layout',mem:true,step:true,run:'full',real:false};
  return ({mission:{cl:'machines',est:true},total:{cl:'machines',run:'simple',est:true},memory:{cl:'layout',mem:true},time:{cl:'layout',mem:true,step:true},
    failures:{cl:'layout',mem:true,step:true,run:'full'},real:{real:true,run:'full'},remember:{cl:'layout',mem:true,step:true,run:'full'}})[STEPS[step].id];
}
const predHidden=()=>isP('mission');
function drawMachines(){
  $('#tcap-top',app).classList.remove('wide');
  const G=S.G, nm=G/8, per=Math.min(nm,128), cols=nm>=64?16:(nm>=16?8:Math.max(1,Math.min(nm,8))), rows=Math.ceil(per/cols);
  const gw=22, gh=12, gx=4, gy=4, W=cols*(gw+gx), H=rows*(gh+gy);
  let s='';
  for(let i=0;i<per;i++){const x=(i%cols)*(gw+gx), y=Math.floor(i/cols)*(gh+gy);
    s+=`<rect class="tcap-mach" x="${x}" y="${y}" width="${gw}" height="${gh}" rx="2"/>`;
    for(let j=0;j<8;j++) s+=`<rect class="tcap-gpu" x="${x+1.5+(j%4)*5}" y="${y+1.5+Math.floor(j/4)*5}" width="4" height="4"/>`;}
  $('#tcap-cl',app).innerHTML=`<svg viewBox="0 0 ${W} ${H}" class="tcap-svg tcap-svg-m" role="img" aria-label="${esc(tx(`${fint(nm)} machines of 8 GPUs`,`${fint(nm)} 台机器，每台 8 张卡`))}">${s}</svg>`;
  $('#tcap-cl-cap',app).textContent=tx(`${fint(G)} H100s: ${fint(nm)} machines of 8`,`${fint(G)} 张 H100：${fint(nm)} 台机器，每台 8 张`);
  $('#tcap-cl-note',app).textContent=nm>128?tx(`The first 128 of ${fint(nm)} machines are drawn; each box is one machine, each square one GPU.`,`${fint(nm)} 台机器只画出前 128 台；每个框是一台机器，每个小方块是一张卡。`):tx('Each box is one machine, each square one GPU.','每个框是一台机器，每个小方块是一张卡。');
}
function drawLayout(r){
  /* one cell per tensor-parallel group (its t GPUs as small squares, 4 to a row), one column per stage, one row per copy */
  const d=r.d, rowsN=Math.min(d,4), p=S.p, t=S.t;
  $('#tcap-top',app).classList.toggle('wide',p>8);   /* a long pipeline takes the full width; the memory column moves under it */
  /* refit step 2 (2026-10-24): the stage numbers above the columns are 11 px now (9 before), so they sit 2 px lower (y 11) and the rows
     with them (top 16), and the numbers stay inside the drawing */
  const sq=4, sg=1.5, per=Math.min(t,4), cw=24, ch2=t>4?15:10, cx=4, lw=40, top=16;
  /* the "… N copies in all" line sets the width when it is longer than the drawing (about 5.6 px a Latin character, 9.5 px a CJK one
     or an ellipsis: set for 9 px text; the 11 px text of the refit still ends 14 px or more inside it) */
  const more=d>rowsN?tx(`… ${fint(d)} copies in all`,`…… 一共 ${fint(d)} 份`):'', tw=[...more].reduce((a,c)=>a+(/[\u2026\u3000-\u9fff\uff00-\uffef]/.test(c)?9.5:5.6),0);
  const W=Math.max(lw+p*(cw+cx),more?Math.ceil(lw+tw+6):0), H=top+rowsN*(ch2+5)+(more?14:0);   /* the last line's 11 px text ends inside (13 and H − 2 before) */
  let s='';
  for(let j=0;j<p;j++){const x=lw+j*(cw+cx); if(p<=8||j%4===0||j===p-1) s+=`<text class="tcap-tl" x="${x+cw/2}" y="11" text-anchor="middle">${j+1}</text>`;}
  for(let i=0;i<rowsN;i++){const y=top+i*(ch2+5);
    s+=`<text class="tcap-tl" x="${lw-5}" y="${y+ch2/2+3}" text-anchor="end">${tx('copy','份')} ${i+1}</text>`;
    for(let j=0;j<p;j++){const x=lw+j*(cw+cx), me=i===0&&j===0, x0=x+(cw-(per*sq+(per-1)*sg))/2;
      s+=`<rect class="tcap-tpg${me?' me':''}" x="${x}" y="${y}" width="${cw}" height="${ch2}" rx="2"/>`;
      for(let k=0;k<t;k++) s+=`<rect class="tcap-gpu${me&&k===0?' me':''}" x="${(x0+(k%4)*(sq+sg)).toFixed(2)}" y="${y+3+Math.floor(k/4)*(sq+sg)}" width="${sq}" height="${sq}"/>`;}}
  if(more) s+=`<text class="tcap-tl" x="${lw}" y="${H-3}">${esc(more)}</text>`;
  $('#tcap-cl',app).innerHTML=`<svg viewBox="0 0 ${W} ${H}" class="tcap-svg" style="width:${Math.round(W*1.15)}px" role="img" aria-label="${esc(layoutName(S))}">${s}</svg>`;
  $('#tcap-cl-cap',app).textContent=tx(`${fint(S.G)} GPUs = TP ${t} × PP ${p} × DP ${fint(d)}`,`${fint(S.G)} 张卡 = 张量并行 ${t} × 流水线 ${p} 段 × 数据并行 ${fint(d)} 份`);
  /* the memory column sits to the right of the cluster, or under it on a phone and for a long pipeline */
  const below=p>8||matchMedia('(max-width:760px)').matches;
  $('#tcap-cl-note',app).textContent=tx(`Columns are pipeline stages and rows are data-parallel copies${d>rowsN?', the first 4 drawn':''}. Each outlined group is one tensor-parallel group inside one machine. The marked GPU’s memory is ${below?'below':'on the right'}.`,
    `每一列是流水线的一段，每一行是一份数据并行${d>rowsN?'，只画出前 4 份':''}。每个框是同一台机器里的一组张量并行。标出的那张卡，显存画在${below?'下面':'右边'}。`);
}
const MEMB=[{k:'w',cls:'w',en:'BF16 weights',zh:'BF16 权重'},{k:'g',cls:'g',en:'FP32 gradients',zh:'FP32 梯度'},{k:'o',cls:'o',en:'Optimizer state',zh:'优化器状态'},{k:'zx',cls:'zx',en:'Two gathered layers (ZeRO-3)',zh:'临时收集的两层（ZeRO-3）',
  tip:{en:'Before computing a layer, ZeRO-3 gathers its BF16 weights from the other data-parallel copies. To hide that transfer behind the compute, it gathers the next layer’s while computing this one, so each GPU holds two layers at once and frees each when it is done.',zh:'ZeRO-3 算一层之前，要先从别的几份数据并行那里收集齐这一层的 BF16 权重。为了让通信藏在计算后面，算这一层时已经在收集下一层，所以每张卡同时放着两层，每层算完就释放。'}},{k:'act',cls:'a',en:'Activations',zh:'激活'}];
function drawMem(r){
  const top=Math.max(GPU.mem,r.mem), H=150, sc=v=>v/top*H, over=r.mem>GPU.mem;
  let y=H, s='';
  MEMB.forEach(b=>{const v=r[b.k]; if(!v) return; const hh=sc(v); y-=hh; s+=`<rect class="tcap-mb ${b.cls}" x="0" y="${y.toFixed(2)}" width="30" height="${Math.max(hh,0).toFixed(2)}"/>`;});
  const ly=H-sc(GPU.mem);
  if(over) s+=`<rect class="tcap-mb over" x="0" y="0" width="30" height="${ly.toFixed(2)}"/>`;
  s+=`<line class="tcap-cap80" x1="-4" x2="34" y1="${ly.toFixed(2)}" y2="${ly.toFixed(2)}"/>`;
  const list=MEMB.filter(b=>r[b.k]>0).reverse().map(b=>`<li${b.tip?` tabindex="0" data-tipx="${esc(L(b.tip))}"`:''}><i class="tcap-sw ${b.cls}"></i><span>${esc(L(b))}</span><b class="mono">${fGB(r[b.k])}</b></li>`).join('')+
    `<li class="tcap-tot"><i class="tcap-sw line"></i><span>${over?tx('Total','合计'):tx('Total, under the 80 GB line','合计，在 80 GB 的线以下')}</span><b class="mono">${fGB(r.mem)}</b></li>`+
    (over?`<li class="tcap-ml-over"><i class="tcap-sw over"></i><span>${tx('Over 80 GB by','比 80 GB 多出')}</span><b class="mono">${fGB(r.mem-GPU.mem)}</b></li>`:'');
  $('#tcap-mem',app).innerHTML=`<svg viewBox="-6 -2 44 ${H+4}" class="tcap-col" aria-hidden="true">${s}</svg><ul class="tcap-ml">${list}</ul>`;
  $('#tcap-mem-cap',app).textContent=over?tx(`The marked GPU needs ${fGB(r.mem)}: it does not fit`,`标出的那张卡要 ${fGB(r.mem)}：放不下`):tx(`The marked GPU’s memory: ${fGB(r.mem)} of 80 GB`,`标出的那张卡的显存：80 GB 里用了 ${fGB(r.mem)}`);
}
const STEPSEG=[{k:'useful',cls:'u',en:'Computing',zh:'在算',tip:{en:'The model’s own forward and backward passes. This lab assumes the GPU runs at 50% of the peak while computing.',zh:'模型本身的前向和反向。这个 Lab 假设 GPU 算的时候能跑到峰值的 50%。'}},
  {k:'recomp',cls:'rc',en:'Recomputing',zh:'重算',tip:{en:'The extra forward pass of recomputation. MFU does not count it.',zh:'重算多做的那遍前向，MFU 不算它。'}},
  {k:'tp',cls:'tp',en:'Tensor-parallel all-reduces',zh:'张量并行的 all-reduce',tip:{en:'The GPU waits while each layer’s partial results are summed over NVLink.',zh:'每一层的部分结果在 NVLink 上相加时，卡在等。'}},
  {k:'bubble',cls:'bb',en:'Bubble',zh:'bubble',tip:{en:'Idle at the start and the end of the pipeline: '+A.ui.math('\\frac{p - 1}{m + p - 1}')+' of the step.',zh:'流水线开头和结尾的空等：占一步的 '+A.ui.math('\\frac{p - 1}{m + p - 1}')+'。'}},
  {k:'dp',cls:'dp',en:'Waiting for data-parallel sync',zh:'等数据并行同步',tip:{en:'The part of the data-parallel traffic that no compute covers.',zh:'数据并行的通信里，计算藏不住的那一部分。'}}];
function drawBar(host,listHost,segs,total,fmtV,share){
  const bar=$(host,app), list=$(listHost,app);
  const keys=segs.map(g=>g.k).join(); if(bar._keys!==keys){bar.innerHTML=segs.map(g=>`<i class="tcap-sg ${g.cls}" data-k="${g.k}"></i>`).join(''); bar._keys=keys;}
  segs.forEach(g=>{const el=bar.querySelector(`[data-k="${g.k}"]`); el.style.width=(total>0?g.v/total*100:0).toFixed(3)+'%';});
  list.innerHTML=segs.filter(g=>g.v>0||g.always).map(g=>`<li tabindex="0" data-tipx="${esc(L(g.tip))}"><i class="tcap-sw ${g.cls}"></i><span>${esc(L(g))}</span><b class="mono">${fmtV(g.v)}</b><em class="mono">${share?pct(g.v/total):''}</em></li>`).join('');
}
function drawStep(r){
  $('#tcap-step-cap',app).textContent=tx(`One step of one GPU: ${fS(r.step)}, MFU ${pct(r.mfu)}`,`一张卡的一步：${fS(r.step)}，MFU ${pct(r.mfu)}`);
  drawBar('#tcap-step','#tcap-step-list',STEPSEG.map(g=>({...g,v:r.seg[g.k],always:g.k==='useful'})),r.step,fS,true);
}
const RUNSEG=[{k:'train',cls:'u',en:'New training',zh:'往前训练',tip:{en:'Steps that move the run forward.',zh:'让训练往前走的那些步。'}},
  {k:'save',cls:'sv',en:'Saving checkpoints',zh:'存档',tip:{en:'Training stopped while a checkpoint is saved: '+A.ui.math('\\frac{δ}{τ}')+' of the time.',zh:'存档时训练停下的时间，占 '+A.ui.math('\\frac{δ}{τ}')+'。'}},
  {k:'redo',cls:'rd',en:'Work redone after failures',zh:'故障后重算',tip:{en:'The steps since the last checkpoint, lost in each failure: '+A.ui.math('\\frac{τ/2}{M}')+' of the time.',zh:'每次故障都会丢掉上次存档以后算的步，占 '+A.ui.math('\\frac{τ/2}{M}')+'。'}},
  {k:'restart',cls:'rs',en:'Restarts',zh:'重启',tip:{en:'Swapping the machine and loading the checkpoint, 10 minutes a failure: '+A.ui.math('\\frac{R}{M}')+' of the time.',zh:'换机器、读回存档，每次故障要 10 分钟，占 '+A.ui.math('\\frac{R}{M}')+'。'}}];
function drawRun(r,simple){
  if(simple){const d=simpleDays(u2,S);
    $('#tcap-run-cap',app).textContent=tx(`The whole run at MFU ${pct0(u2)}: ${dayU(d)}`,`整次训练，MFU 按${u2===0.3?'三':(u2===0.5?'五':'四')}成算：${dayU(d)}`);
    drawBar('#tcap-run','#tcap-run-list',[{...RUNSEG[0],v:d,always:true}],d,x=>dayU(x),false); return;}
  if(!r.valid){$('#tcap-run-cap',app).textContent=tx('Failures come too often: more than half the time would be wasted, beyond this estimate','故障太勤：一半以上的时间都会白费，超出了这个估算的范围');
    drawBar('#tcap-run','#tcap-run-list',RUNSEG.map(g=>({...g,v:0})),1,()=>'—',false); return;}
  const W=r.wall/86400;
  $('#tcap-run-cap',app).textContent=tx(`The whole run: ${dayU(r.days)}, ${pct(r.good)} of it new training`,`整次训练：${dayU(r.days)}，其中 ${pct(r.good)} 在往前训练`);
  drawBar('#tcap-run','#tcap-run-list',RUNSEG.map(g=>({...g,v:g.k==='train'?r.train/86400:W*(g.k==='save'?r.wSave:(g.k==='redo'?r.wRedo:r.wRestart)),always:g.k==='train'})),W,x=>dayU(x),true);
}
function drawReal(){
  const r=plan(S), rows=REAL.map(x=>({lab:MD[x.id].name,u:x.u,me:false})).concat([{lab:tx('This lab’s 70B plan','这个 Lab 的 70B 计划'),u:r.eff,me:true}]);
  // the note's two formulas carry their full stops (tcap-fxs, nw), so a narrow line never begins with “.” or “。” (refit step 3)
  $('#tcap-real-cap',app).textContent=tx('What share of the H100’s peak did each run get over all its GPU hours?','每次训练在它的全部 GPU 小时里，做到了 H100 峰值的几成？');
  $('#tcap-real',app).innerHTML=`<ul class="tcap-rl">`+rows.map(x=>`<li class="${x.me?'me':''}"><span>${esc(x.lab)}</span><div class="tcap-rt"><i style="width:${(x.u/0.5*100).toFixed(2)}%"></i></div><b class="mono">${pct(x.u)}</b></li>`).join('')+`</ul>`+
    `<p class="small muted">${tx(`Llama 3.1 rows: <span class="tcap-fxs">${A.ui.math('\\frac{6 \\times \\text{parameters} \\times 15\\text{T tokens}}{\\text{the model card’s H100 hours} \\times 3,600 \\times 989.5\\ \\text{TFLOP/s}}')}.</span> This lab’s row: <span class="nw">${A.ui.math('\\text{MFU} \\times \\text{share of time training}')}.</span> The scale ends at 50%.`,`Llama 3.1 三行：<span class="tcap-fxs">${A.ui.math('\\frac{6 \\times \\text{参数量} \\times 15\\text{T 个 token}}{\\text{模型卡的 H100 小时} \\times 3,600 \\times 989.5\\ \\text{TFLOP/s}}')}。</span>这个 Lab 一行：<span class="nw">${A.ui.math('\\text{MFU} \\times \\text{往前训练的时间占比}')}。</span>刻度到 50% 为止。`)}</p>`;
}
function statusOf(){
  if(predHidden()) return {cls:'neutral',txt:tx('Predict first','先预测'),tip:tx('The days stay hidden until you answer the question in part 1.','在第 1 部分作答之前，天数先隐藏。')};
  const v=secs();
  if(v.est) return {cls:'neutral',txt:tx('Estimate','估算'),tip:tx('A first estimate from 6ND and an assumed MFU; the layout comes in part 3.','先用 6ND 和假设的 MFU 估一次；切法在第 3 部分再定。')};
  const r=plan(S);
  return {cls:r.fits?'ok':'bad',txt:r.fits?tx('Fits in 80 GB','放得下'):tx('Out of memory','显存不够'),tip:r.fits?tx(`Each GPU uses ${fGB(r.mem)} of its 80 GB.`,`每张卡用了 80 GB 里的 ${fGB(r.mem)}。`):tx(`Each GPU would need ${fGB(r.mem)}, more than its 80 GB.`,`每张卡要 ${fGB(r.mem)}，超过了 80 GB。`)};
}
function drawHeadline(){
  const t=$('#tcap-st-title',app), sub=$('#tcap-st-sub',app), st=$('#tcap-status',app), md=MD[S.model], v=secs();
  if(mode==='challenge') t.textContent=tx('Your plan: 70B on 512 H100s, 2T tokens','你的计划：512 张 H100 训练 70B，2T 个 token');
  else if(v.real) t.textContent=tx('Real runs against this lab’s plan','真实的训练和这个 Lab 的计划');
  else t.textContent=tx(`Plan: ${md.name} on ${fint(S.G)} H100s, ${S.D/1e12}T tokens`,`计划：${fint(S.G)} 张 H100 训练 ${md.name}，${S.D/1e12}T 个 token`);
  sub.textContent=v.cl==='machines'?tx('Not split yet','还没决定怎么切'):(layoutName(S)+(S.rc?tx(', recomputation','，重算'):'')+(v.run==='full'?tx(`; ${saveName(S)}`,`；${saveName(S)}`):''));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
/* the number a card shows, with its K/M/B suffix (B1: 4.93M is 4.93 × 10⁶, not 4.93); a value in scientific notation gets no delta */
const numOf=v=>{const s=String(v).replace(/,/g,''); if(/×/.test(s)) return NaN; const m=s.match(/(\d+(?:\.\d+)?)([KMB])?/); return m?parseFloat(m[1])*({K:1e3,M:1e6,B:1e9}[m[2]]||1):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=secs(), hid=predHidden();
  if(v.est){
    const d=simpleDays(u2,S);
    return [{k:'days',label:tx('Days to train','要训的天数'),v:hid?'—':fDays(d),sub:hid?tx('shown in part 2','第 2 部分揭晓'):tx(`at MFU ${pct0(u2)}`,`MFU 按 ${pct0(u2)} 算`),tip:hid?tx('Hidden until you answer the question.','作答之前先隐藏。'):tx(`The total work, 6 × ${big(MD[S.model].N)} parameters × ${sci(S.D)} tokens, ÷ (${fint(S.G)} GPUs × 989.5 TFLOP/s × ${pct0(u2)}) = ${fDays(d)} days.`,`总运算量 6 × ${big(MD[S.model].N)} 个参数 × ${sci(S.D)} 个 token，÷（${fint(S.G)} 张卡 × 989.5 TFLOP/s × ${pct0(u2)}）= ${fDays(d)} 天。`)},
      {k:'gpuh',label:tx('GPU hours','GPU 小时'),v:hid?'—':fGpuh(d*24*S.G),sub:hid?tx('shown in part 2','第 2 部分揭晓'):tx(`${fint(S.G)} GPUs`,`${fint(S.G)} 张卡`),tip:hid?tx('Hidden until you answer the question.','作答之前先隐藏。'):tx(`${fint(S.G)} GPUs × ${fDays(d)} days × 24 hours.`,`${fint(S.G)} 张卡 × ${fDays(d)} 天 × 24 小时。`)},
      {k:'work',label:tx('Total work','总运算量'),v:sci(6*MD[S.model].N*S.D),sub:'FLOP',tip:tx('6 × parameters × tokens.','6 × 参数量 × token 数。')}];}
  const r=plan(S);
  const mem={k:'mem',label:tx('Memory per GPU','每张卡的显存'),v:fGB(r.mem),sub:r.fits?tx('fits in 80 GB','放得下 80 GB'):tx('over 80 GB','超过 80 GB'),tip:tx(`${gb2(r.states)} of model states + ${r.zx?`${gb2(r.zx)} for the two layers ZeRO-3 gathers + `:''}${gb2(r.act)} of activations = ${gb2(r.mem)}.`,`模型状态 ${gb2(r.states)} + ${r.zx?`ZeRO-3 临时收集的两层 ${gb2(r.zx)} + `:''}激活 ${gb2(r.act)} = ${gb2(r.mem)}。`)};
  const mfu={k:'mfu',label:'MFU',v:pct(r.mfu),sub:tx('share of the peak, averaged over a step','一步里平均用上的峰值比例'),tip:tx(`The model’s own work in a step, against what the GPUs could do at peak in that time: 6 × ${big(MD[S.model].N)} parameters × ${fint(r.Btok)} tokens ÷ (${fint(S.G)} GPUs × 989.5 TFLOP/s × ${fS(r.step)}) = ${pct(r.mfu)}.`,`一步里模型本身的运算量，和这段时间里所有卡按峰值能做的运算比：6 × ${big(MD[S.model].N)} 个参数 × ${fint(r.Btok)} 个 token ÷（${fint(S.G)} 张卡 × 989.5 TFLOP/s × ${fS(r.step)}）= ${pct(r.mfu)}。`)};
  const days={k:'days',label:tx('Days to train','要训的天数'),v:r.valid?fDays(r.days):'—',sub:r.valid?tx('failures included','算上故障'):tx('too many failures','故障太多'),tip:tx(`${fint(Math.round(r.steps))} steps × ${fS(r.step)} ÷ ${pct(r.good)} of the time training.`,`${fint(Math.round(r.steps))} 步 × ${fS(r.step)} ÷ 往前训练的时间占比 ${pct(r.good)}。`)};
  const gpuh={k:'gpuh',label:tx('GPU hours','GPU 小时'),v:r.valid?fGpuh(r.gpuh):'—',sub:tx(`${fint(S.G)} GPUs`,`${fint(S.G)} 张卡`),tip:TIP.gpuh()};
  const good={k:'good',label:tx('Share of time training','往前训练的时间占比'),v:pct(r.good),sub:tx(`a failure every ${fH(r.M)}`,`平均每 ${fH(r.M)}一次故障`),tip:TIP.goodput()};
  if(v.real) return [days,gpuh,mfu,good];
  if(!v.step) return [mem,
    {k:'states',label:tx('Model states per GPU','每张卡的模型状态'),v:fGB(r.states),sub:tx(`of ${fTB(18*r.N)} in all`,`总共 ${fTB(18*r.N)}`),tip:tx('The BF16 weights, FP32 gradients and optimizer state this GPU keeps, after tensor, pipeline and ZeRO splitting.','这张卡在张量并行、流水线和 ZeRO 切开以后，存的 BF16 权重、FP32 梯度和优化器状态。')+(r.zx?tx(` The BF16 weights of the two layers ZeRO-3 gathers are not counted here but listed on their own (${fGB(r.zx)}).`,`ZeRO-3 临时收集的两层 BF16 权重不算在这里，另列一项（${fGB(r.zx)}）。`):'')},
    {k:'act',label:tx('Activations per GPU','每张卡的激活'),v:fGB(r.act),sub:tx(`${r.inf} ${r.inf===1?'micro-batch':'micro-batches'} at once`,`同时存 ${r.inf} 个 micro-batch`),tip:tx(`The marked GPU keeps the activations of ${r.inf} ${r.inf===1?'micro-batch':'micro-batches'} on its ${fixd(r.Ls,r.Ls%1?3:0)} layers.`,`标出的那张卡要存 ${r.inf} 个 micro-batch 在它那 ${fixd(r.Ls,r.Ls%1?3:0)} 层上的激活。`)}];
  if(!v.run) return [mem,mfu,{k:'step',label:tx('Time per step','每步耗时'),v:fS(r.step),sub:tx(`${fint(S.B)} sequences of 4,096 tokens`,`每步 ${fint(S.B)} 条序列，每条 4,096 个 token`),tip:tx('The pipeline’s m + p − 1 slots, plus the data-parallel sync that no compute covers.','流水线的 m + p − 1 格，加上计算藏不住的数据并行同步。')}];
  return [days,gpuh,mfu,mem,good];
}
function drawMetrics(){
  const host=$('#tcap-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v)),pts:/%$/.test(m.v)})||'';}
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
function mathHTML(){
  if(predHidden()) return `<p>${tx('The formulas appear once you have answered the question in part 1.','在第 1 部分作答以后，这里给出公式。')}</p>`;
  const v=secs(), md=MD[S.model], rows=[], t=T_;
  if(v.real){const r=plan(S), r70=REAL[1], n70=MD['70b'].N;
    rows.push(LEAD(tx('Share of the peak a real run got','真实训练用上的峰值比例'),tx(': the work 6ND, divided by what the published GPU hours could do at the peak.','：运算量 6ND，除以公布的 GPU 小时按峰值能做的运算。'))+M_(tx('\\text{share} = \\frac{6 \\times N \\times D}{\\text{GPU hours} \\times 3600 \\times \\text{peak}}','\\text{比例} = \\frac{6 \\times N \\times D}{\\text{GPU 小时} \\times 3600 \\times \\text{峰值}}'))+
      `<p>Llama 3.1 70B${tx(':','：')}</p>`+MV(`= \\frac{6 \\times ${fint(n70)} \\times ${sciT(15e12)}}{${sciT(r70.h)} \\times 3,600 \\times 989.5 \\times 10^{12}}`,pct(r70.u)));
    rows.push(LEAD(tx('This lab’s plan','这个 Lab 的计划'),tx(': its MFU, times the share of time that goes to new training.','：它的 MFU，乘以真正往前训练的时间占比。'))+MW(tx('\\text{this lab’s plan}','\\text{这个 Lab 的计划}'),['\\text{MFU}',tx('\\times \\text{share of time training}','\\times \\text{往前训练的时间占比}')])+`<p>= ${pct(r.mfu)} × ${pct(r.good)} = <b>${pct(r.eff)}</b></p>`);
    return rows.join('');}
  rows.push(LEAD(tx('Total work','总运算量'),tx(': every token uses every parameter for about 6 FLOPs, forward and backward.','：每个 token 用到每个参数，前向加反向约 6 次运算。'))+M_('C = 6 \\times N \\times D')+`<p>${tx(`N = ${big(md.N)} parameters, D = ${sci(S.D)} tokens.`,`N = ${big(md.N)} 是参数量，D = ${sci(S.D)} 是 token 数。`)}</p>`+MV(`= 6 \\times ${fint(md.N)} \\times ${sciT(S.D)}`,`${sci(6*md.N*S.D)} FLOP`));
  if(v.est){const d=simpleDays(u2,S); rows.push(LEAD(tx('Training time','训练时间'),tx(`: the work, divided by what the ${fint(S.G)} GPUs really do per second (G is the number of GPUs).`,`：总运算量，除以 ${fint(S.G)} 张卡每秒真正做到的运算（G 是卡数）。`))+M_(tx('\\text{training time} = \\frac{C}{G \\times 989.5\\ \\text{TFLOP/s} \\times \\text{MFU}}','\\text{训练时间} = \\frac{C}{G \\times 989.5\\ \\text{TFLOP/s} \\times \\text{MFU}}'))+MV(`= \\frac{${sciT(6*md.N*S.D)}}{${fint(S.G)} \\times 989.5 \\times 10^{12} \\times ${u2}} = ${fixd(d*86400/1e6,2)} \\times 10^{6}\\ \\text{s}`,dayU(d))); return rows.join('');}
  const r=plan(S), Ls=fixd(r.Ls,r.Ls%1?3:0);
  /* memory: bytes per parameter by ZeRO stage over d copies, and the activations of min(m, p) micro-batches (S9) */
  const bppS=S.z===0?'18':(S.z===1?`6 + \\frac{12}{${fint(r.d)}}`:(S.z===2?`2 + \\frac{16}{${fint(r.d)}}`:`\\frac{18}{${fint(r.d)}}`)), bpp=(r.w+r.g+r.o)/r.pg;
  const sym=(k,en,zh)=>`<li>${A.ui.math(k)}${tx(` ${en}`,`：${zh}`)}</li>`;
  rows.push(LEAD(tx('Memory per GPU','每张卡的显存'),r.zx?tx(': the GPU’s share of the model states, plus the two layers ZeRO-3 gathers and the activations it keeps.','：这张卡分到的模型状态，加上 ZeRO-3 临时收集的两层和它留着的激活。')
    :tx(': the GPU’s share of the model states, plus the activations it keeps.','：这张卡分到的模型状态，加上它留着的激活。'))+
    MW(tx('\\text{memory}','\\text{显存}'),[tx('\\frac{N}{t \\times p} \\times \\text{bytes per parameter}','\\frac{N}{t \\times p} \\times \\text{每个参数的字节数}')].concat(r.zx?['+ 2 \\times \\frac{2 \\times N}{L \\times t}']:[],S.rc
      ?[['+ \\min(m, p) \\times \\frac{L}{p}','\\times \\frac{2 \\times s \\times b \\times h}{t}'],tx('+ \\text{one full layer}','+ \\text{一整层}')]
      :[['+ \\min(m, p) \\times \\frac{L}{p}','\\times \\frac{s \\times b \\times h \\times (18 + 4I/h)}{t}']]))+
    `<ul class="tcap-syms small">`+sym(`N = ${big(md.N)}`,'parameters','参数量')+sym(`t = ${S.t}`,'tensor-parallel GPUs per layer','张量并行的份数')+sym(`p = ${S.p}`,'pipeline stages','流水线段数')+sym(`d = ${fint(r.d)}`,'data-parallel copies','数据并行的份数')+
      sym(`m = ${r.m}`,'micro-batches per pipeline in a step','每条流水线每步的 micro-batch 数')+sym(`L = ${md.L}`,'layers','层数')+sym('s = 4,096','tokens per sequence','每条序列的 token 数')+sym(`b = ${S.b}`,S.b===1?'sequence per micro-batch':'sequences per micro-batch','每个 micro-batch 的序列数')+
      sym(`h = ${fint(md.h)}`,'hidden size','隐藏维度')+sym(`I = ${fint(md.I)}`,'the MLP’s intermediate size','MLP 的中间维度')+`</ul>`+
    `<p>${tx('Bytes per parameter with ','每个参数的字节数，')}${zName(S.z)}${tx(':','：')}</p>`+MV(`${bppS} = ${sig(bpp,3)}`)+
    `<p>${tx('Model states:','模型状态：')}</p>`+MV(`${big(r.pg)} \\times ${sig(bpp,3)}\\ \\text{${tx('bytes','字节')}} = \\text{${fGB(r.w+r.g+r.o)}}`)+(r.zx?`<p>${tx('Plus the BF16 weights of the two layers ZeRO-3 gathers, the one being computed and the next (2 bytes per parameter):','再加 ZeRO-3 临时收集的两层 BF16 权重，正在算的这一层和提前收集的下一层（每个参数 2 字节）：')}</p>`+MV(`2 \\times \\frac{2 \\times ${big(md.N)}}{${md.L} \\times ${S.t}}\\ \\text{${tx('bytes','字节')}} = \\text{${fGB(r.zx)}}`):'')+
    `<p>${tx('Activations:','激活：')}</p>`+MV(`${r.inf} \\times ${Ls} \\times ${S.rc?`\\text{${sig(r.keep/1e9,4)} GB} + \\text{${sig(r.full/1e9,4)} GB}`:`\\text{${sig(r.full/1e9,4)} GB}`} = \\text{${fGB(r.act)}}`)+
    `<p>${tx('In all: ','合计：')}${gb2(r.states)} + ${r.zx?`${gb2(r.zx)} + `:''}${gb2(r.act)} = <b>${gb2(r.mem)}</b></p>`);
  if(v.step){
    rows.push(LEAD(tx('One micro-batch','一个 micro-batch'),tx(': each stage computes it, then waits for the tensor-parallel all-reduces.','：每段流水线先算它，再等张量并行的 all-reduce。'))+MW(tx('\\text{micro-batch time}','\\text{一个 micro-batch 的时间}'),['\\frac{k \\times N/p \\times s \\times b}{t \\times 989.5\\ \\text{TFLOP/s} \\times 50\\%}',tx('+ n \\times \\frac{L}{p} \\times \\text{all-reduce time}','+ n \\times \\frac{L}{p} \\times \\text{一次 all-reduce 的时间}')])+
      `<p class="small">${tx(`k = ${r.k} operations per parameter per token (8 with recomputation). n = ${r.nar} all-reduces per layer (6 with recomputation).`,`k = ${r.k} 是每个 token、每个参数的运算次数（重算时 8），n = ${r.nar} 是每层的 all-reduce 次数（重算时 6）。`)}</p>`+(S.t>1?MW(tx('\\text{all-reduce time}','\\text{一次 all-reduce 的时间}'),['\\frac{2(t - 1)}{t}',tx('\\times \\frac{2 \\times s \\times b \\times h\\ \\text{bytes}}{450\\ \\text{GB/s}}','\\times \\frac{2 \\times s \\times b \\times h\\ \\text{字节}}{450\\ \\text{GB/s}}')])+`<p>= ${sig(r.ar*1e3,3)} ${tx('ms (over NVLink)','毫秒（走 NVLink）')}</p>`:'')+
      `<p>${tx('Compute plus all-reduces:','计算加 all-reduce：')} ${sig(r.cmb,4)} s + ${sig(r.tpmb,4)} s = <b>${sig(r.per,4)} s</b></p>`);
    const net=S.net==='eth'?'12.5':'400';
    const dpT=r.d<=1?tx('There is no data parallelism, so there is nothing to sync.','没有数据并行，也就不用同步。'):
      (r.cross?tx(`The data-parallel copies are on different machines. One pass sends the stage’s whole BF16 array, ${A.ui.math('\\frac{2N}{p}')} = ${fGB(r.Bx)}, through one machine’s ${net} GB/s: ${sig(r.one,3)} s.`,`这几份数据并行在不同的机器上。每传一遍，要把这一段的全部 BF16 数据 ${A.ui.math('\\frac{2N}{p}')} = ${fGB(r.Bx)} 送过一台机器的 ${net} GB/s 网络，约 ${sig(r.one,3)} 秒。`)
              :tx(`The data-parallel copies are in one machine. In each pass every GPU moves ${A.ui.math('\\frac{d - 1}{d}')} of its ${fGB(r.Bx)} over NVLink: ${sig(r.one,3)} s.`,`这几份数据并行在同一台机器里。每传一遍，每张卡走 NVLink 传自己那 ${fGB(r.Bx)} 的 ${A.ui.math('\\frac{d - 1}{d}')}，约 ${sig(r.one,3)} 秒。`));
    const winT=r.d<=1?'':(S.z===0?tx(' Without ZeRO a step makes 2 passes, covered only by the last micro-batch’s backward pass.','不用 ZeRO 时，每步传 2 遍，只能藏在最后一个 micro-batch 的反向后面。'):
      (S.z===1?tx(' ZeRO-1 also makes 2 passes a step, covered only by the last micro-batch’s backward pass.','ZeRO-1 每步也传 2 遍，只能藏在最后一个 micro-batch 的反向后面。'):
      (S.z===2?tx(' ZeRO-2 makes m + 1 passes a step (m is the micro-batches per step), covered by every micro-batch’s backward pass.','ZeRO-2 每步传 m + 1 遍（m 是每步的 micro-batch 数），能藏在每个 micro-batch 的反向后面。'):
               tx(' ZeRO-3 makes 3m passes a step, covered by all the computing.','ZeRO-3 每步传 3m 遍，能藏在全部计算后面。'))));
    rows.push(LEAD(tx('One step','一步'),tx(': the pipeline runs m + p − 1 micro-batch slots, then waits for whatever data-parallel sync the computing did not cover.','：流水线要走 m + p − 1 格，再加上计算没藏住的那部分数据并行同步。'))+MW(tx('\\text{step}','\\text{每步}'),[tx('(m + p - 1) \\times \\text{micro-batch time}','(m + p - 1) \\times \\text{一个 micro-batch 的时间}'),tx('+ \\max(0, \\text{sync time} - \\text{covered})','+ \\max(0, \\text{同步的时间} - \\text{藏得住的})')])+
      `<p class="small">${dpT}${winT}</p>`+
      MV(`(${r.m} + ${S.p} - 1) \\times ${sig(r.per,4)}\\ \\text{s} + \\max(0, ${sig(r.tdp,3)} - ${sig(r.win,3)})\\ \\text{s}`,`${sig(r.step,4)} s`));
    rows.push(LEAD('MFU',tx(': the share of the GPUs’ peak the step really uses.','：这一步真正用上的峰值比例。'))+M_(tx('\\text{MFU} = \\frac{6 \\times N \\times \\text{tokens per step}}{G \\times 989.5\\ \\text{TFLOP/s} \\times \\text{step}}','\\text{MFU} = \\frac{6 \\times N \\times \\text{每步的 token 数}}{G \\times 989.5\\ \\text{TFLOP/s} \\times \\text{每步耗时}}'))+
      MV(`= \\frac{6 \\times ${big(md.N)} \\times ${fint(r.Btok)}}{${fint(S.G)} \\times 989.5 \\times 10^{12} \\times ${sig(r.step,4)}\\ \\text{s}}`,pct(r.mfu)));}
  if(v.run==='full'){
    rows.push(LEAD(tx('Time between failures','多久坏一次'),tx(': the more machines, the sooner one of them fails.','：机器越多，越快有一台坏掉。'))+M_(tx('M = \\frac{\\text{days per machine failure}}{\\text{machines}}','M = \\frac{\\text{每台机器坏一次的天数}}{\\text{机器台数}}'))+`<p>= ${S.node} × 24 ${tx('h','小时')} ÷ ${fint(r.machines)} = ${fH(r.M)}</p>`);
    const dT=S.asy?tx('An asynchronous save stops training for 10 s.','异步存档每次让训练停 10 秒。'):tx(`A synchronous save writes ${fTB(r.ckB)} at 10 GB/s.`,`同步存档要以每秒 10 GB 写完 ${fTB(r.ckB)}。`);
    rows.push(LEAD(tx('Time wasted','白费的时间'),tx(': saves stop training, and each failure redoes half an interval and restarts.','：存档要停训练，每次故障还要重算半个间隔、再重启。'))+M_(tx('\\text{wasted} \\approx \\frac{δ}{τ} + \\frac{τ/2 + R}{M}','\\text{白费的比例} \\approx \\frac{δ}{τ} + \\frac{τ/2 + R}{M}'))+
      `<p class="small">${tx('δ is how long one save stops training, τ the interval between saves, R the restart time. ','δ 是每次存档让训练停多久，τ 是存档间隔，R 是重启要多久。')}${dT}</p>`+
      `<p>${tx(`δ = ${fixd(r.dS,r.dS>=10?1:2)} s = ${sig(r.dS/3600,3)} h, `,`δ = ${fixd(r.dS,r.dS>=10?1:2)} 秒 = ${sig(r.dS/3600,3)} 小时，`)}${t('T',`τ = ${tx(`${S.ivl} h`,`${S.ivl} 小时`)}`)}${tx(', R = 1/6 h, M = ','，R = 1/6 小时，M = ')}${fH(r.M)}${tx(':','：')}</p>`+MV(`\\frac{${sig(r.dS/3600,3)}}{${S.ivl}} + \\frac{${S.ivl/2} + ${fixd(1/6,3)}}{${fixd(r.M,r.M%1?2:0)}}`,pct(r.waste)));
    if(r.valid) rows.push(LEAD(tx('Days','天数'),tx(': the steps’ time, stretched by the share wasted.','：所有步数的时间，再按白费的比例拉长。'))+M_(tx('\\text{days} = \\frac{\\text{steps} \\times \\text{step}}{1 - \\text{wasted}}','\\text{天数} = \\frac{\\text{步数} \\times \\text{每步耗时}}{1 - \\text{白费的比例}}'))+`<p>= ${fint(Math.round(r.steps))} × ${sig(r.step,4)} s ÷ ${fixd(r.good,4)} ÷ 86,400 = <b>${dayU(r.days)}</b></p>`);}
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge') S={...chCfg(ch)};
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const v=secs(), r=plan(S);
  $('#tcap-top',app).hidden=!v.cl&&!v.mem;
  $('#tcap-clbox',app).hidden=!v.cl; $('#tcap-membox',app).hidden=!v.mem;
  $('#tcap-stepbox',app).hidden=!v.step||predHidden(); $('#tcap-runbox',app).hidden=!v.run||predHidden(); $('#tcap-realbox',app).hidden=!v.real;
  $('#tcap-graph',app).dataset.v=v.cl||'real';
  drawHeadline();
  if(v.cl==='machines') drawMachines(); else if(v.cl==='layout'&&r) drawLayout(r);
  if(v.mem&&r) drawMem(r);
  if(v.step&&r) drawStep(r);
  if(v.run&&!predHidden()) drawRun(r,v.run==='simple');
  if(v.real) drawReal();
  drawMetrics();
  const mb=$('#tcap-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the six choices in the task card. Everything else is fixed: Llama‑3.1‑70B, 2T tokens, 512 H100s, 1,024 sequences of 4,096 tokens per step, InfiniBand, a machine failure every 64 days.','挑战中只能在任务卡里改那六项。其余设置固定：Llama‑3.1‑70B，2T 个 token，512 张 H100，每步 1,024 条 4,096 个 token 的序列，InfiniBand，每台机器 64 天坏一次。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'||id==='total') return tx('Fixed in this part: Llama‑3.1‑70B, 15T tokens, 1,024 H100s.','这一部分固定：Llama‑3.1‑70B，15T 个 token，1,024 张 H100。');
  if(id==='memory'||id==='time') return tx('Fixed in this part: Llama‑3.1‑70B, 15T tokens, 1,024 H100s, 1,024 sequences per step. Pick the layout in the card.','这一部分固定：Llama‑3.1‑70B，15T 个 token，1,024 张 H100，每步 1,024 条序列。在卡片里选切法。');
  if(id==='failures') return tx('Fixed in this part: TP 8 × PP 8 with ZeRO-1, and a machine failure every 64 days. Pick how to save in the card.','这一部分固定：张量并行 8 份 × 流水线 8 段 + ZeRO-1，每台机器 64 天坏一次。在卡片里选怎么存档。');
  return tx('Fixed in this part. You can change every setting in Scenarios.','这一部分固定不变。在应用场景里可以改所有设置。');
}
function okT(t,p,G){return G%(t*p)===0&&S.B%((G/(t*p))*S.b)===0;}
function syncControls(){
  const set=(key,val)=>$$(`.seg-ctl[data-key="${key}"]`,app).forEach(g=>A.ui.setPressed(g,String(val)));
  ['model','D','G','B','t','p','z','rc','b','ivl','asy','net','node'].forEach(k=>set(k,S[k]));
  ['t','p','z','rc','ivl','asy'].forEach(k=>set('ch'+k,ch[k]));
  $$('#tcap-controls .seg-ctl[data-key="t"] button',app).forEach(b=>{b.disabled=!okT(+b.dataset.v,S.p,S.G);});
  $$('#tcap-controls .seg-ctl[data-key="p"] button',app).forEach(b=>{b.disabled=!okT(S.t,+b.dataset.v,S.G);});
  $$('#tcap-controls .seg-ctl[data-key="b"] button',app).forEach(b=>{const d=S.G/(S.t*S.p); b.disabled=!(Number.isInteger(d)&&S.B%(d*+b.dataset.v)===0);});
  $$('#tcap-rail .seg-ctl[data-key="chp"] button',app).forEach(b=>{b.disabled=512%(ch.t*+b.dataset.v)!==0;});
  const g=mode==='guided', c=mode==='challenge';
  $('#tcap-controls',app).classList.toggle('locked',c||g);
  $$('#tcap-controls .lockable',app).forEach(e=>{e.hidden=g||c;});
  const ln=$('#tcap-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||predHidden()) return; const r=plan(S); if(!r) return;
  A.announce(secs().run==='full'&&r.valid?tx(`${fGB(r.mem)} per GPU, MFU ${pct(r.mfu)}, ${dayU(r.days)}.`,`每张卡 ${fGB(r.mem)}，MFU ${pct(r.mfu)}，${dayU(r.days)}。`):tx(`${fGB(r.mem)} per GPU.`,`每张卡 ${fGB(r.mem)}。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
/* keep a Settings change valid: when t × p does not divide the GPUs, or the copies get no whole micro-batch, fall back step by step */
function fixup(){
  if(!okT(S.t,S.p,S.G)){for(const p of [S.p,8,4,2,1]) if(okT(S.t,p,S.G)){S.p=p; break;} if(!okT(S.t,S.p,S.G)) for(const t of [8,4,2,1]) if(okT(t,S.p,S.G)){S.t=t; break;}}
  const d=S.G/(S.t*S.p); if(!Number.isInteger(d)||S.B%(d*S.b)){for(const b of [4,2,1]) if(Number.isInteger(d)&&S.B%(d*b)===0){S.b=b; break;}}
  if(!plan(S)){S.b=1; for(const t of [8,4,2,1]) for(const p of [16,8,4,2,1]) if(!plan(S)&&okT(t,p,S.G)){S.t=t; S.p=p;}}
}
function setState(p){Object.assign(S,p); if(mode==='sandbox') fixup(); render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv2(v){u2=v; tried2.add(v); render();}
function setLv3(v){lv3=v; tried3.add(v); setState({...lvOf(LV3,v).s});}
function setLv4(v){lv4=v; tried4.add(v); setState({...lvOf(LV4,v).s});}
function setLv5(v){lv5=v; tried5.add(v); setState({...lvOf(LV5,v).s});}
function phoneHTML(){
  if(mode==='challenge') return `<p class="small tcap-ph">${tx('Task: train 70B on 512 H100s within 60 days, with the fewest GPU hours.','任务：60 天内用 512 张 H100 训完 70B，GPU 小时越少越好。')}</p><button type="button" class="btn ghost sm" data-act="to-task" data-tipx="${esc(tx('Scrolls down to the six choices in the task card.','向下滚到任务卡里的六项选择。'))}">${tx('Choose the layout below ↓','去下面选切法 ↓')}</button>`;
  if(mode==='sandbox') return `<p class="small tcap-ph">${tx('Five real situations: pick one to see its days, GPU hours and memory.','五个真实情况：选一个，看它要几天、多少 GPU 小时、显存放不放得下。')}</p><button type="button" class="btn ghost sm" data-act="to-task" data-tipx="${esc(tx('Scrolls down to the list of scenarios.','向下滚到场景列表。'))}">${tx('Pick a scenario below ↓','去下面选一个场景 ↓')}</button>`;
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the four answers in the card.','向下滚到卡片里的四个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='total'||id==='memory'||id==='time'||id==='failures') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the options in the card.','向下滚到卡片里的选项。'))}">${tx('Pick one below ↓','在下面选一个 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#tcap-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#tcap-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#tcap-rail .step.current',app)||$('#tcap-rail',app));}
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
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#tcap-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#tcap-rail .step.current',app)||$('#tcap-rail .railcard',app)||$('#tcap-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#tcap-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#tcap-insight',app)); else revealBox($('#tcap-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
/* after Submit or “Show the recommended plan” (应修 5): a result taller than the space under the bars starts right under them, so the
   verdict, the main gap and the next step share one screen in both languages; a shorter one is brought into view as before */
function revealRes(){const box=app&&$('#tcap-ch-res',app); if(!box||box.hidden) return; const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16&&Math.abs(r.top-top)>=8){window.scrollBy({top:r.top-top,behavior:A.reduceMotion()?'auto':'smooth'}); return;}
  revealBox(box);}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; allOpen=false; ch={...CH.start}; S={...chCfg(ch)};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[1].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#tcap-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const MI=['8b','70b','405b'], IVS=[0.25,1,3], NODES=[32,64,224];
const tokOf=s=>[MI.indexOf(s.model),DS.indexOf(s.D),GS.indexOf(s.G),s.t,s.p,s.z,s.rc,s.b,s.B===4096?1:0,s.net==='eth'?1:0,IVS.indexOf(s.ivl),s.asy,NODES.indexOf(s.node)].join('_');
const token=()=>tokOf(S);
function parse(tok){
  const q=String(tok||'').split('_').map(Number); if(q.length!==13||q.some(x=>!Number.isInteger(x))) return null;
  const [mi,di,gi,t,p,z,rc,b,bi,ne,ii,as,ni]=q;
  if(mi<0||mi>2||di<0||di>2||gi<0||gi>=GS.length||![1,2,4,8].includes(t)||![1,2,4,8,16].includes(p)||z<0||z>3||(rc!==0&&rc!==1)||![1,2,4,8].includes(b)||(bi!==0&&bi!==1)||(ne!==0&&ne!==1)||ii<0||ii>2||(as!==0&&as!==1)||ni<0||ni>2) return null;
  const c={model:MI[mi],D:DS[di],G:GS[gi],t,p,z,rc,b,B:bi?4096:1024,net:ne?'eth':'ib',ivl:IVS[ii],asy:as,node:NODES[ni]};
  return plan(c)?c:null;
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+3), pk=a=>a[Math.floor(r()*a.length)];
  let G=pk([256,512,1024,2048]), D=pk([1e12,2e12,15e12]), u=pk([0.3,0.35,0.4,0.45]);
  if(G===1024&&D===15e12) D=2e12;   /* never part 1's own question (R3.8) */
  const N=MD['70b'].N, days=6*N*D/(G*GPU.peak*u)/86400, M=64*24/(G/8), dS=CK.bpp*N/CK.write, tau=Math.sqrt(2*(dS/3600)*M);
  return {seed,G,D,u,days,gpuh:days*24*G,M,dS,tau};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#tcap-drill-q',app), a=$('#tcap-drill-a',app);
  qd.innerHTML=tx(`Llama‑3.1‑70B (70.6B parameters) is trained on <b>${d.D/1e12}T</b> tokens with <b>${fint(d.G)}</b> H100s (989.5 TFLOP/s each) at an MFU of <b>${pct0(d.u)}</b>. How many days does it take, and how many GPU hours? If each 8-GPU machine fails once every 64 days and a synchronous save stops training for ${fixd(d.dS,0)} s, about how often should you save?`,
                  `用 <b>${fint(d.G)}</b> 张 H100（每张 989.5 TFLOP/s）用 <b>${d.D/1e12}T</b> 个 token 训练 Llama‑3.1‑70B（70.6B 个参数），MFU 是 <b>${pct0(d.u)}</b>。要训多少天、多少 GPU 小时？如果每台 8 卡机器平均 64 天坏一次，同步存一次档要让训练停 ${fixd(d.dS,0)} 秒，大约多久存一次档好？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>${dayU(d.days)}, ${fGpuh(d.gpuh)} GPU hours. Save about every ${fH(d.tau)}.</b></p>`+
    `<p>Work: 6 × 70,553,706,496 × ${sci(d.D)} = ${sci(6*MD['70b'].N*d.D)} FLOP. Speed: ${fint(d.G)} × 989.5 × 10¹² × ${d.u} FLOP/s. Days = ${A.ui.math('\\frac{\\text{work}}{\\text{speed} \\times 86,400}')} = ${fDays(d.days)}.</p>`+
    `<p>Failures: M = 64 days × 24 ÷ ${fint(d.G/8)} machines = ${fH(d.M)}. With δ = ${fixd(d.dS,0)} s = ${sig(d.dS/3600,3)} h, the interval τ that makes ${A.ui.math('\\frac{δ}{τ} + \\frac{τ}{2M}')} smallest is, in hours:</p>`+M_(`τ = \\sqrt{2δM} = \\sqrt{2 \\times ${sig(d.dS/3600,3)} \\times ${fixd(d.M,1)}} \\approx ${fixd(d.tau,2)}`)+`<p>That is about ${fH(d.tau)}.</p>`,
    `<p><b>${dayU(d.days)}，${fGpuh(d.gpuh)} GPU 小时。约每 ${fH(d.tau)}存一次档。</b></p>`+
    `<p>运算量：6 × 70,553,706,496 × ${sci(d.D)} = ${sci(6*MD['70b'].N*d.D)} FLOP。速度：${fint(d.G)} × 989.5 × 10¹² × ${d.u} FLOP/s。天数 = ${A.ui.math('\\frac{\\text{运算量}}{\\text{速度} \\times 86,400}')} = ${fDays(d.days)}。</p>`+
    `<p>故障：M = 64 天 × 24 ÷ ${fint(d.G/8)} 台 = ${fH(d.M)}。δ = ${fixd(d.dS,0)} 秒 = ${sig(d.dS/3600,3)} 小时。按小时算，让 ${A.ui.math('\\frac{δ}{τ} + \\frac{τ}{2M}')} 最小的存档间隔是：</p>`+M_(`τ = \\sqrt{2δM} = \\sqrt{2 \\times ${sig(d.dS/3600,3)} \\times ${fixd(d.M,1)}} \\approx ${fixd(d.tau,2)}`)+`<p>也就是约 ${fH(d.tau)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={G:{en:'H100 GPUs',zh:'H100 的卡数'},t:{en:'Tensor parallel',zh:'张量并行'},p:{en:'Pipeline stages',zh:'流水线段数'},m:{en:'Pipeline stages',zh:'流水线段数'},T:{en:'Save a checkpoint every',zh:'多久存一次档'}};   /* key T: the interval τ (Settings key ivl) */
const TERM={
 G:()=>tx(`G is the number of GPUs, ${fint(S.G)} here.`,`G 是卡数，这里是 ${fint(S.G)}。`),
 t:()=>tx(`t is the number of GPUs one layer is split across (tensor parallelism), ${S.t} here.`,`t 是一层切到的卡数（张量并行），这里是 ${S.t}。`),
 p:()=>tx(`p is the number of pipeline stages, ${S.p} here.`,`p 是流水线的段数，这里是 ${S.p}。`),
 m:()=>{const r=plan(S); return r?tx(`m is the number of micro-batches each pipeline runs per step: 1,024 sequences ÷ ${fint(r.d)} copies = ${r.m}. It follows from the layout.`,`m 是每条流水线每步要算的 micro-batch 数：1,024 条序列 ÷ ${fint(r.d)} 份 = ${r.m}。它由切法决定。`):'';},
 T:()=>tx(`τ is the interval between checkpoints, ${fIvl(S.ivl)} here.`,`τ 是两次存档之间的间隔，这里是 ${fIvl(S.ivl)}。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return k==='t'||k==='p'||k==='T'||k==='m'?(k==='G'?tx('Fixed during the challenge: 512 H100s.','挑战中固定不变：512 张 H100。'):''):tx('Fixed during the challenge: 512 H100s.','挑战中固定不变：512 张 H100。');
  if(mode!=='guided') return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>mode==='challenge'?({t:$('#tcap-ic-cht',app),p:$('#tcap-ic-chp',app),m:$('#tcap-ic-chp',app),T:$('#tcap-ic-chivl',app)}[k]):({G:$('#tcap-ctl-G',app),t:$('#tcap-ctl-t',app),p:$('#tcap-ctl-p',app),m:$('#tcap-ctl-p',app),T:$('#tcap-ctl-ivl',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#tcap-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#tcap-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...PRESETS[1].s}; render();}); return;}
  if(a==='submit'){chSubmitted=true; const e=chMe(); if(e&&e.ok) chReveal=true; renderRail(); requestAnimationFrame(revealRes); return;}
  if(a==='reveal'){chReveal=true; render(); requestAnimationFrame(revealRes); return;}
  if(a==='to-q'){const q=$('#tcap-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#tcap-rail .tcap-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-task'){const q=$('#tcap-rail .railcard',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#tcap-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#tcap-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
const NUMK=new Set(['D','G','B','t','p','z','rc','b','ivl','asy','node']);
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#tcap-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#tcap-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null){const lv=b.closest('.tcap-lv'); if(!lv) return; const id=lv.id;
      const f=()=>byLearner(()=>{if(id==='tcap-ic-lv2') setLv2(+d.lever); else if(id==='tcap-ic-lv3') setLv3(d.lever); else if(id==='tcap-ic-lv4') setLv4(d.lever); else setLv5(d.lever);});
      if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(/^ch/.test(k)){const kk=k.slice(2), val=+v; if(kk==='p'&&512%(ch.t*val)) return;
        byLearner(()=>{ch={...ch,[kk]:val}; if(512%(ch.t*ch.p)) ch.p=1; render();}); return;}
      if(SKEYS.includes(k)) bySettings(b,()=>setState({[k]:NUMK.has(k)?+v:v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#tcap-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#tcap-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the levers tried, the
   challenge's choices and result, the drill; the plans are recomputed from these */
const pickS=o=>Object.fromEntries(SKEYS.map(k=>[k,o[k]]));
const okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o);
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:pickS(S),pred,observed:{...observed},saved:saved?pickS(saved):null,u2,tried2:[...tried2],lv3,tried3:[...tried3],lv4,tried4:[...tried4],lv5,tried5:[...tried5],
    ch:{...ch},chSubmitted,chReveal,allOpen,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&!!(chEval(ch)||{}).ok};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const cfg=x=>okObj(x)?parse(tokOf(x)):null, inV=(arr,v)=>arr.some(x=>x.v===v);
    const tried=(a,arr,cur,first)=>{const t=new Set(Array.isArray(a)?a.filter(v=>inV(arr,v)):[]); if(!t.size) t.add(first); t.add(cur); return t;};
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...(cfg(o.S)||BASE)}; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    u2=inV(LV2,o.u2)?o.u2:0.4; tried2=tried(o.tried2,LV2,u2,0.4);
    lv3=inV(LV3,o.lv3)?o.lv3:'none'; tried3=tried(o.tried3,LV3,lv3,'none');
    lv4=inV(LV4,o.lv4)?o.lv4:'start'; tried4=tried(o.tried4,LV4,lv4,'start');
    lv5=inV(LV5,o.lv5)?o.lv5:'3s'; tried5=tried(o.tried5,LV5,lv5,'3s');
    const c=okObj(o.ch)?o.ch:{}, k=(key,arr)=>arr.includes(c[key])?c[key]:CH.start[key];
    ch={t:k('t',CH.Ts),p:k('p',CH.Ps),z:k('z',CH.Zs),rc:k('rc',CH.RCs),ivl:k('ivl',CH.IVs),asy:k('asy',CH.ASs)}; if(512%(ch.t*ch.p)) ch.p=1; if(!chEval(ch)) ch={...CH.start};
    saved=mode==='challenge'?cfg(o.saved):null; if(mode==='challenge') S={...chCfg(ch)};
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; allOpen=chSubmitted&&o.allOpen===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tokOf(p))) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#tcap-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#tcap-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#tcap-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#tcap-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#tcap-graph',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#tcap-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.txt;
      if(predHidden()){vis.innerHTML=''; read.innerHTML=esc(tx('1,024 H100s, 70B, 15T tokens: answer first','1,024 张 H100 训练 70B：先作答')); return;}
      const v=secs(), r=plan(S);
      if(v.est){const d=simpleDays(u2,S); vis.innerHTML=''; read.innerHTML=esc(tx(`At MFU ${pct0(u2)}: ${dayU(d)}`,`MFU 按 ${pct0(u2)} 算：${dayU(d)}`)); return;}
      vis.innerHTML=`<div class="tcap-dk${r.fits?'':' bad'}"><i style="width:${Math.min(100,r.mem/GPU.mem*100).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(v.run==='full'&&r.valid?tx(`${fGB(r.mem)} per GPU, MFU ${pct(r.mfu)}, ${dayU(r.days)}`,`每张卡 ${fGB(r.mem)}，MFU ${pct(r.mfu)}，${dayU(r.days)}`):(v.step?tx(`${fGB(r.mem)} per GPU, MFU ${pct(r.mfu)}`,`每张卡 ${fGB(r.mem)}，MFU ${pct(r.mfu)}`):tx(`${fGB(r.mem)} per GPU of 80 GB`,`每张卡 80 GB 里用 ${fGB(r.mem)}`)));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const pub=r=>{if(!r) return null; const o={...r}; delete o.md; return deep(o);};
window.__tcap=Object.freeze({
  plan:c=>pub(plan({...BASE,...c})), simpleDays:(u,c)=>simpleDays(u,{...BASE,...c}), real:()=>deep(REAL),
  chAll:()=>chAll().map(e=>deep({t:e.t,p:e.p,z:e.z,rc:e.rc,ivl:e.ivl,asy:e.asy,fits:e.fits,ok:e.ok,days:e.r.days,gpuh:e.r.gpuh,mem:e.r.mem,min:chMinimal(e)})),
  drill:s=>deep(makeDrill(s)),
  /* the challenge's summary after Submit (应修 5): the counterexample, the best single change, where the days go, the rounding */
  chSummary:o=>{const me=chEval({...CH.start,...o}); if(!me) return null; const c=chCounter(me), oc=chOneChange(me);
    return deep({counter:c?{...chPick(c),days:c.r.days,gpuh:c.r.gpuh,ok:c.ok,fits:c.fits,mem:c.r.mem,dpShare:c.r.seg.dp/c.r.step}:null,one:oc?{k:oc.k,v:oc.v,gpuh:oc.e.r.gpuh}:null,parts:dayParts(me.r),bestParts:dayParts(chBest().r)});},
  rough:x=>rough(x),
  /* G7: the single-change sentence the result shows for a plan, in the page's language, and the GPU hours as shown */
  chNextText:o=>{const me=chEval({...CH.start,...o}); if(!me) return null; const d=document.createElement('div'); d.innerHTML=c3HTML(me,chBest()); const p=d.querySelector('.tcap-next'); return p?p.textContent:'';},
  chShownH:o=>{const me=chEval({...CH.start,...o}); return me?roughH(me.r.gpuh):null;},
  consts:Object.freeze({GPU:{...GPU},U0,SEQ,CK:{...CK},MD:deep(Object.fromEntries(Object.entries(MD).map(([k,v])=>[k,{N:v.N,L:v.L,h:v.h,I:v.I}]))),CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,
    PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),LV3:deep(LV3.map(o=>({v:o.v,...o.s}))),LV4:deep(LV4.map(o=>({v:o.v,...o.s}))),LV5:deep(LV5.map(o=>({v:o.v,...o.s})))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},u2,lv3,lv4,lv5,tried2:[...tried2],tried3:[...tried3],tried4:[...tried4],tried5:[...tried5],ch:{...ch},chSubmitted,chReveal,allOpen,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transferQ:e.transfer.map(q=>q.q),transferWhy:e.transfer.map(q=>q.why)});}
});
})();
