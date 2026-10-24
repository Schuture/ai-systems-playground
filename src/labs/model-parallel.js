/* model-parallel (world training): How do pipeline and tensor parallelism cut one model apart?
   Fidelity: Analytical. Every number comes from the engine below (tests/model-parallel_numbers.py recomputes it independently):
   - compute: 6 × (parameters of the layers) × tokens ÷ (989.5 TFLOP/s × 50%), 50% being the share of the peak while computing
     (integrator decision 2026-10-04; MFU appears only as a measured result); forward 1/3, backward 2/3;
   - pipeline: p stages of L/p layers, m micro-batches; GPipe and 1F1B simulated (forward 1 unit, backward 2 units); step
     (m + p − 1)(F + B); bubble (p − 1)/(m + p − 1) (zero-bubble README); GPU 1 holds m (GPipe) or min(m, p) (1F1B) micro-batches;
   - tensor parallel: per layer and micro-batch 2 all-reduces forward + 2 backward of b·s·h·2 bytes; ring time 2(t−1)/t × bytes ÷ bw
     (nccl-tests); NVLink 450 GB/s each way per GPU; across machines each GPU of a machine runs its own tensor-parallel group, so the
     machine's 400 GB/s (8 × 400 Gb/s) is shared 8 ways, 50 GB/s each (scaling book); on the critical path;
   - data parallel: BF16 gradients (2 bytes, decision 10) all-reduced over d copies, exposed: inside one machine (t·d ≤ 8) a ring
     over NVLink, across machines 2 × B ÷ 400 GB/s with B the stage's whole gradient held by one machine (decision 8);
   - memory: 18 bytes per parameter ÷ (t·p) (ZeRO-1: 6 + 12/d), activations s·b·h·(18 + 4·I/h) ÷ t per layer and micro-batch
     (Megatron theoretical_memory_usage.py: sequence parallelism and selective recomputation) × L/p layers × micro-batches in
     flight; logits left out. GB = 10⁹ bytes.
   window.__mp is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='model-parallel';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const GPU={peak:989.5e12,mem:80e9,nv:450e9,ib:50e9,node:400e9,perNode:8};   // ib: one GPU's own 400 Gb/s port; node: a machine's 8 ports
const SHARE=0.50, SEQ=4096, GBY=1e9;                  // SHARE: the share of the peak while computing (an assumption, not MFU)
const MD={
 '8b':{id:'8b',name:'Llama‑3.1‑8B',N:8030261248,L:32,h:4096,I:14336,kv:8,hd:128,V:128256},
 '70b':{id:'70b',name:'Llama‑3.1‑70B',N:70553706496,L:80,h:8192,I:28672,kv:8,hd:128,V:128256},
 '405b':{id:'405b',name:'Llama‑3.1‑405B',N:405853388800,L:126,h:16384,I:53248,kv:8,hd:128,V:128256}
};
const layerP=md=>2*md.h*md.h+2*md.h*md.kv*md.hd+3*md.h*md.I+2*md.h;
const actK=md=>18+4*md.I/md.h;                       // Megatron: s·b·h·(18 + 4·ffn/h) bytes per layer and micro-batch
const embShare=md=>2*md.V*md.h/md.N;                 // the embedding table and the LM head, as a share of all parameters (70B: 2.98%)
/* one configuration: model, t (TP), p (PP), d (DP), m (micro-batches per pipeline), b (sequences per micro-batch), seq, place
   ('in' = the TP group inside one machine, 'out' = one GPU in each machine), zero (1 = ZeRO-1 over DP), sched ('gpipe'|'1f1b') */
function cost(c){
  const md=MD[c.model], Pl=layerP(md), tok=c.b*c.seq, Ls=md.L/c.p;
  const flopMb=6*Pl*tok*Ls/c.t;                        // per micro-batch, per GPU of a stage, forward + backward
  const comp=flopMb/(GPU.peak*SHARE);
  const arBytes=tok*md.h*2, bw=c.place==='out'?GPU.ib:GPU.nv;
  const ar=c.t>1?2*(c.t-1)/c.t*arBytes/bw:0;
  const comm=4*Ls*ar;                                  // per micro-batch per stage: 2 forward + 2 backward per layer
  const per=comp+comm, pipe=(c.m+c.p-1)*per;
  const pg=md.N/(c.t*c.p);
  /* data parallel (integrator decisions 8 and 10): gradients move in BF16, 2 bytes per value. A data-parallel group inside one
     machine (t·d ≤ 8) is a ring over NVLink: 2 × bytes × (d − 1)/d ÷ 450 GB/s. A group that spans machines follows the scaling
     book: an all-reduce of B bytes takes 2 × B ÷ 400 GB/s, 400 GB/s being one machine's network bandwidth, and B the whole array
     one machine holds once: the stage's gradient, all t tensor-parallel shards together (t × pg × 2 bytes). */
  const dpIn=c.t*c.d<=GPU.perNode, dpBytes=dpIn?2*pg:2*c.t*pg;
  const dp=c.d>1?(dpIn?2*(c.d-1)/c.d*dpBytes/GPU.nv:2*dpBytes/GPU.node):0;
  const step=pipe+dp;
  const bpp=c.zero?6+12/c.d:18, states=bpp*pg;
  const inflight=c.sched==='gpipe'?c.m:Math.min(c.m,c.p);
  const actMb=Ls*tok*md.h*actK(md)/c.t, act=inflight*actMb;
  const mem=states+act;
  return {md,Pl,tok,Ls,flopMb,comp,arBytes,bw,ar,comm,per,pipe,pg,dpIn,dpBytes,dp,step,bpp,states,inflight,actMb,act,mem,fits:mem<=GPU.mem,
    bubble:(c.p-1)/(c.m+c.p-1),tpShare:c.m*comm/step,weights:2*pg,gpus:c.t*c.p*c.d};
}
/* one layer of one micro-batch (parts 4 and 5): compute and the four all-reduces */
function layer(t,place){const md=MD['70b'], c=cost({model:'70b',t,p:md.L,d:1,m:1,b:1,seq:SEQ,place,zero:0,sched:'1f1b'});
  return {comp:c.comp,ar:c.ar,comm:c.comm,total:c.comp+c.comm,share:c.comm/(c.comp+c.comm),bytes:c.arBytes,bw:c.bw,params:c.Pl/t,c};}
/* the schedule: forward 1 unit, backward 2 units; Megatron's 1F1B order (p − i − 1 warm-up forwards on stage i) */
const SIMS={};
function sim(p,m,sched){
  const key=`${sched}|${p}|${m}`; if(SIMS[key]) return SIMS[key];
  const order=[];
  for(let i=0;i<p;i++){const o=[];
    if(sched==='gpipe'){for(let j=0;j<m;j++) o.push(['F',j]); for(let j=0;j<m;j++) o.push(['B',j]);}
    else {const w=Math.min(p-i-1,m); for(let j=0;j<w;j++) o.push(['F',j]); for(let k=0;k<m-w;k++){o.push(['F',w+k]); o.push(['B',k]);} for(let j=m-w;j<m;j++) o.push(['B',j]);}
    order.push(o);}
  const eF=[...Array(p)].map(()=>new Array(m).fill(-1)), eB=[...Array(p)].map(()=>new Array(m).fill(-1));
  const idx=new Array(p).fill(0), free=new Array(p).fill(0), ops=[...Array(p)].map(()=>[]);
  let prog=true;
  while(prog){prog=false;
    for(let i=0;i<p;i++){while(idx[i]<order[i].length){const [k,j]=order[i][idx[i]];
      const dep=k==='F'?(i===0?0:eF[i-1][j]):(i===p-1?eF[i][j]:eB[i+1][j]); if(dep<0) break;
      const s=Math.max(free[i],dep), e=s+(k==='F'?1:2); ops[i].push({k,j,s,e}); (k==='F'?eF:eB)[i][j]=e; free[i]=e; idx[i]++; prog=true;}}}
  const T=Math.max(...free);
  /* GPU 1's activations in flight: a forward started and its backward not yet finished */
  let cur=0, peak=0; const ev=[]; ops[0].forEach(o=>{if(o.k==='F') ev.push([o.s,1]); else ev.push([o.e,-1]);});
  ev.sort((a,b)=>a[0]-b[0]||a[1]-b[1]); ev.forEach(e=>{cur+=e[1]; peak=Math.max(peak,cur);});
  const busy=ops.reduce((a,r)=>a+r.reduce((x,o)=>x+o.e-o.s,0),0);
  return (SIMS[key]={p,m,sched,ops,T,peak,idle:1-busy/(p*T)});
}
/* what a sim shows up to time x (units): idle share so far, GPU 1's activations now and the most so far */
function simAt(s,x){
  if(x<=0) return {idle:0,now:0,peak:0};
  let busy=0; s.ops.forEach(r=>r.forEach(o=>{busy+=Math.max(0,Math.min(o.e,x)-o.s);}));
  let now=0, pk=0; const ev=[]; s.ops[0].forEach(o=>{if(o.k==='F'&&o.s<x) ev.push([o.s,1]); if(o.k==='B'&&o.e<=x) ev.push([o.e,-1]);});
  ev.sort((a,b)=>a[0]-b[0]||a[1]-b[1]); ev.forEach(e=>{now+=e[1]; pk=Math.max(pk,now);});
  return {idle:Math.max(0,1-busy/(s.p*Math.min(x,s.T))),now,peak:pk};
}

/* the challenge: Llama‑3.1‑70B on 32 GPUs (4 machines), 256 sequences of 4,096 tokens per step, ZeRO-1 over DP, 1F1B */
const CH={G:32,B:256,Ts:[1,2,4,8],Ps:[1,2,4,8],Ms:[8,16,32,64,128,256],start:{t:8,p:1,m:8},tol:0.01};
function chCfg(t,p,m){const d=CH.G/(t*p); if(!Number.isInteger(d)||d<1) return null; const b=CH.B/(d*m); if(!Number.isInteger(b)||b<1) return null;
  return {model:'70b',t,p,d,m,b,seq:SEQ,place:'in',zero:1,sched:'1f1b'};}
function chEval(t,p,m){const c=chCfg(t,p,m); if(!c) return null; const r=cost(c); return {t,p,m,d:c.d,b:c.b,r,fits:r.fits,step:r.step};}
function chAll(){const a=[]; CH.Ts.forEach(t=>CH.Ps.forEach(p=>CH.Ms.forEach(m=>{const e=chEval(t,p,m); if(e) a.push(e);}))); return a;}
function chBest(){return chAll().filter(e=>e.fits).sort((a,b)=>a.step-b.step)[0];}
const chMinimal=e=>!!e&&e.fits&&e.step<=chBest().step*(1+CH.tol);

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=x=>x===0?'0%':(x<0.1?fixd(x*100,1)+'%':fixd(x*100,0)+'%');
const fK=md=>{const k=actK(md); return Number.isInteger(k)?String(k):fixd(k,2);};
const fGB=b=>{const g=b/GBY; return (g>=100?fixd(g,0):fixd(g,1))+' GB';};
const fMs=s=>{const ms=s*1e3; return (ms>=100?fixd(ms,0):(ms>=10?fixd(ms,1):fixd(ms,2)))+' ms';};
const fS=s=>(s>=100?fixd(s,0):fixd(s,1))+' s';
const fL=x=>String(+x.toFixed(3));
const fMB=b=>fixd(b/1e6,1)+' MB';
const fX=x=>String(+x.toFixed(5));                    // an exact short value such as 6.375 bytes per parameter (R6.6)
const fS2=s=>fixd(s,2)+' s';                          // two decimals, so that parts add up to the shown total (R6.6)
const SH=()=>pct(SHARE);                              // “50%”, the share of the peak while computing
const big=n=>A.fmt.big?A.fmt.big(n):fint(n);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const labLinkZ=id=>{const h=labLink(id); return /^[A-Za-z0-9]/.test(h.replace(/<[^>]+>/g,''))?' '+h:h;};   // after a Chinese character
const M=t=>`<div class="mp-fx">${A.ui.math(t)}</div>`;
const tname=t=>t===1?tx('no tensor parallelism','不做张量并行'):tx(`TP ${t}`,`张量并行 ${t} 份`);
const layoutName=(t,p,d)=>p===1?tx(`no pipeline × TP ${t} × DP ${d}`,`不分流水线 × 张量并行 ${t} 份 × 数据并行 ${d} 份`):tx(`PP ${p} × TP ${t} × DP ${d}`,`流水线 ${p} 段 × 张量并行 ${t} 份 × 数据并行 ${d} 份`);   // R4: pipeline first
const layoutSym=(t,p,d,m)=>p===1?tx(`no pipeline (p = 1) × TP ${t} × DP ${d}, m = ${m} micro-batches per copy`,`不分流水线（p = 1）× 张量并行 ${t} 份 × 数据并行 ${d} 份，每份切成 m = ${m} 个 micro-batch`):tx(`p = ${p} stages × TP ${t} × DP ${d}, m = ${m} micro-batches per pipeline`,`p = ${p} 段流水线 × 张量并行 ${t} 份 × 数据并行 ${d} 份，每条流水线 m = ${m} 个 micro-batch`);

/* =====================================================================
   module state
   ===================================================================== */
const BASE={model:'70b',t:1,p:8,d:1,m:1,b:1,seq:SEQ,place:'in',zero:0,sched:'gpipe'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lv3='gpipe', tried3=new Set(['gpipe']), lv5='in', tried5=new Set(['in']), mSeen=new Set([1]), tSeen=new Set([1]);
let quiet=false;   // true while a scenario sets the state: no ▲▼ badges for it (R6.6)
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=1, drillOpen=false;
let ph=0, playing=false, spd='mid', raf=0, lastT=0, relang=false, dockK='';
const watched=new Set();                                // R3.20: pictures already played to the end
const DUR={slow:20000,mid:8000,fast:4500};                 // owner b8 (1): the old Fast is the new Medium (default); Slow is the old Medium
const shownPrev={};
const SRC={
 meg:'https://github.com/NVIDIA/Megatron-LM',
 megSched:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/core/pipeline_parallel/schedules.py',
 megLayers:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/core/tensor_parallel/layers.py',
 megMem:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/theoretical_memory_usage.py',
 megReadme:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/README.md',
 megCP:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/docs/user-guide/features/context_parallel.md',
 bridge:'https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ec4ec358c3622134905144ca193516b194feddd5/docs/parallelisms.md',
 nccl:'https://github.com/NVIDIA/nccl-tests/blob/a3d4589e8f32f8363c334c52eedad75b44ed8b2d/doc/PERFORMANCE.md',
 book:'https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/gpus.md',
 titan:'https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/benchmarks/llama3_h100_202412_torchtitan.md',
 mamf:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/compute/accelerator/README.md',
 zb:'https://github.com/sail-sg/zero-bubble-pipeline-parallelism',
 pipe:'https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.pipelining.md',
 p1909:'https://arxiv.org/abs/1909.08053', p2104:'https://arxiv.org/abs/2104.04473', p1811:'https://arxiv.org/abs/1811.06965', p2205:'https://arxiv.org/abs/2205.05198'
};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 states:()=>tx('Model states are everything training keeps for each parameter, 18 bytes in all: the BF16 weights (2 bytes), the FP32 gradients (4), an FP32 copy of the weights (4) and Adam’s m and v (4 each). The FP32 copy, m and v together are the optimizer state, 12 bytes per parameter (Megatron-LM’s distributed-optimizer notes).',
               '模型状态是训练时每个参数都要存的东西，一共 18 字节：BF16 权重 2 字节，FP32 梯度 4 字节，FP32 权重副本 4 字节，Adam 的 m 和 v 各 4 字节。FP32 权重副本、m 和 v 合起来叫优化器状态，每个参数 12 字节（Megatron-LM 分布式优化器的文档）。'),
 stage:()=>tx('A stage is the run of consecutive layers that one GPU holds in a pipeline: here 10 of the 80 layers.','段（stage）是流水线里一张卡负责的那几层连续的层：这里是 80 层里的 10 层。'),
 mb:()=>tx('A micro-batch is one piece of the step’s batch. The pipeline passes micro-batches down one after another; the gradients of all of them are added up before the weights are updated once.',
           'micro-batch 是把一步的 batch 切开后的一小份。流水线一份接一份地往下传，所有份的梯度加起来以后，权重才更新一次。'),
 bubble:()=>tx('The bubble is the time a GPU in a pipeline sits idle during a step, waiting for data from the GPUs before it or for gradients from the GPUs after it.',
               'bubble 是一步里流水线上的卡空等的时间。卡在等前面的卡把数据传过来，或者等后面的卡把梯度传回来。'),
 act:()=>tx('Activations are the intermediate results a layer computes in the forward pass. The backward pass needs them, so they stay in memory until that micro-batch’s backward has passed the layer. This lab assumes selective recomputation: attention’s intermediate results are not kept but computed again in the backward pass, and everything else is kept.',
            '激活是前向时每一层算出的中间结果。反向要用到它们，所以它们一直留在显存里，直到这个 micro-batch 的反向经过这一层。这个 Lab 假设用选择性重算，attention 里的中间结果不存，反向时再算，其余的都存着。'),
 allreduce:()=>tx('An all-reduce adds up one tensor held by each of several GPUs and gives every GPU the sum. In the ring algorithm each GPU sends and receives '+A.ui.math('\\frac{2(t - 1)}{t}')+' times the tensor’s size (nccl-tests).',
                  'all-reduce 把几张卡上各一份的张量加起来，再让每张卡都拿到这个和。用环形算法时，每张卡要收发张量大小的 '+A.ui.math('\\frac{2(t - 1)}{t}')+' 倍（nccl-tests）。'),
 nvlink:()=>tx('NVLink connects the 8 GPUs inside one H100 machine: each GPU can send 450 GB/s each way (the scaling book, gpus.md). Measured all-reduces reach less; the scaling book says it is hard to exceed 370 GB/s.',
               'NVLink 把一台 H100 机器里的 8 张卡连在一起，每张卡每个方向能发 450 GB/s（scaling book 的 gpus.md）。实测的 all-reduce 达不到这么多，scaling book 说很难超过 370 GB/s。'),
 ib:()=>tx('Between machines the GPUs talk over an InfiniBand network. In the reference H100 cluster the 8 GPUs of a machine share 400 GB/s of network bandwidth: each GPU has its own 400 Gb/s port, 50 GB/s each way (the scaling book, gpus.md).',
           '机器之间的卡通过 InfiniBand 网络通信。在参考的 H100 集群里，一台机器的 8 张卡共用 400 GB/s 的网络带宽，每张卡有自己的一个 400 Gb/s 网口，每个方向 50 GB/s（scaling book 的 gpus.md）。'),
 dp:()=>tx('Data parallelism runs several copies of the same model (here, of the same pipeline), each on its own share of the step’s data; at the end of the step the copies average their gradients with an all-reduce.',
           '数据并行是同一个模型（这里是同一条流水线）跑好几份，每份算这一步里自己那一份数据；一步结束时，各份用 all-reduce 把梯度平均一次。'),
 zero1:()=>tx('ZeRO-1 (Megatron-LM’s distributed optimizer): the data-parallel copies split the optimizer state among themselves, the 12 bytes per parameter of the FP32 copy and Adam’s m and v. So each keeps '+A.ui.math('6 + \\frac{12}{d}')+' bytes per parameter instead of 18, with d the number of data-parallel copies (here d = '+S.d+').',
              'ZeRO-1（Megatron-LM 的分布式优化器）让各份数据并行分开存优化器状态，也就是 FP32 权重副本和 Adam 的 m、v 这 12 字节。所以每个参数只占 '+A.ui.math('6 + \\frac{12}{d}')+' 字节，而不是 18 字节；d 是数据并行的份数，这里 d = '+S.d+'。'),
 /* integrator decision 2026-10-04: one name and one default for the assumption; MFU only as a result */
 share:()=>tx(`An assumed value: how much of the H100’s peak, 989.5 TFLOP/s, a GPU reaches while it is computing. Plain BF16 matrix multiplies reach about 80% (measured by ml-engineering); attention, normalization and smaller kernels do not, so this lab uses ${SH()}.`,
              `这是假设值：GPU 在算的时候，能跑到 H100 峰值 989.5 TFLOP/s 的多少。单纯的 BF16 矩阵乘法约能到 80%（ml-engineering 实测）；attention、归一化和较小的 kernel 达不到，这个 Lab 取 ${SH()}。`),
 mfu:()=>tx(`MFU: the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time; time spent waiting for transfers or idling in the pipeline counts. Measured: torchtitan 33% to 39% (Llama 3.1 8B with FSDP on 8 or 128 H100s), Megatron-LM’s README 41% to 47% on H100. In this lab a GPU reaches ${SH()} while computing and hides no communication, so its MFU tops out at ${SH()}. Recomputing every layer makes a GPU do 8N FLOPs per token instead of 6N (N is the parameter count), but MFU still counts 6N: the extra forward pass is not the model’s own arithmetic.`,
            `MFU：模型真正需要的运算（每个 token 6 × 参数量次）占同一段时间里 GPU 峰值算力的比例。等通信、流水线空转的时间都算在里面。实测值：torchtitan 是 33% 到 39%（Llama 3.1 8B，FSDP，8 张或 128 张 H100），Megatron-LM 的 README 在 H100 上是 41% 到 47%。这个 Lab 里 GPU 算的时候能跑到峰值的 ${SH()}，通信又一点不藏，所以 MFU 最高也只有 ${SH()}。每一层都重算时，每个 token 要做 8N 次运算（N 是参数量），MFU 仍按 6N 算，因为多做的那一遍前向不是模型本身要的运算。`)
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'all',t:{en:'All 8',zh:'8 张都在干活'}},{v:'some',t:{en:'2 to 7',zh:'2 到 7 张'}},{v:'one',t:{en:'Only 1',zh:'只有 1 张'}}];
const PRED_RIGHT='one';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const MS2=[1,2,4,8,16,32,64], TS=[1,2,4,8];
/* owner b8 (5): each option carries its worked example of the bubble formula (p = 8 in part 2), computed from the engine */
const bubbleEx=(p,m)=>tx(`p = ${p}, m = ${m}:`,`p = ${p}，m = ${m}：`)+`<div class="fx">${A.ui.math(`\\text{bubble} = \\frac{${p} - 1}{${m} + ${p} - 1} = \\frac{${p-1}}{${m+p-1}} \\approx ${pct(sim(p,m,'gpipe').idle).replace('%','\\%')}`)}</div>`;
const mCtl=(id,inline)=>MS2.reduce((h,m)=>h.replace(`data-v="${m}"`,`data-v="${m}" data-tipx="${esc(bubbleEx(8,m))}"`),A.ui.seg({id,key:'m',label:tx('Micro-batches per step, m','micro-batch 数 m（每步切成几份）'),options:MS2.map(m=>[String(m),String(m)]),value:String(S.m),inline}));
const tCtl=(id,inline)=>A.ui.seg({id,key:'t',label:tx('GPUs one layer is split across, t','一层切到几张卡上（t）'),options:TS.map(t=>[String(t),String(t)]),value:String(S.t),inline});
const LV3=[{v:'gpipe',t:{en:'GPipe (initial setup)',zh:'GPipe（初始设置）'},d:{en:'All m = 16 forwards first, then all 16 backwards',zh:'m = 16 个前向全部做完，再做 16 个反向'}},
  {v:'1f1b',t:{en:'1F1B',zh:'1F1B'},d:{en:'GPU 1 first does p = 8 forwards, then one forward and one backward in turn',zh:'第 1 张卡先做 p = 8 个前向，然后一个前向、一个反向交替'}}];
const LV5=[{v:'in',t:{en:'In one machine (initial setup)',zh:'在一台机器里（初始设置）'},d:{en:'The 8 GPUs exchange results over NVLink, 450 GB/s each way',zh:'8 张卡之间走 NVLink，每个方向 450 GB/s'}},
  {v:'out',t:{en:'One GPU in each of 8 machines',zh:'分在 8 台机器上，每台 1 张'},d:{en:'The 8 GPUs exchange results over InfiniBand, 50 GB/s each way',zh:'8 张卡之间走 InfiniBand，每个方向 50 GB/s'}}];
const p3=v=>sim(8,16,v);
const lv3Res=v=>tx(`GPU 1 keeps ${v==='gpipe'?'m':'p'} = ${p3(v).peak}`,`第 1 张卡存 ${v==='gpipe'?'m':'p'} = ${p3(v).peak} 份`);
const lv5Res=v=>tx(`all-reduces ${pct(layer(8,v).share)} of the time`,`all-reduce 占 ${pct(layer(8,v).share)}`);
const levers=(id,aria,val,opts,tried,res)=>`<div class="mp-lv" id="${id}">`+A.ui.levers({aria,value:val,options:opts.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(tried.has(o.v)||o.v===val?res(o.v):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'gantt',gate:()=>!!pred,skip:true,
  lead:()=>tx('70B has 80 layers. Put 10 on each of 8 GPUs and send one batch: how many GPUs work at a time?','70B 有 80 层，8 张卡各放 10 层。一次只送一批数据，同一时刻有几张卡在干活？'),
  enter:()=>({...BASE,m:1}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   const st8=cost({...BASE,model:'70b',t:1,p:8,d:1,m:1,zero:0});   // S14: 8 stages alone still hold 1/8 of the model states per GPU
   return tx(`<p>You are training Llama‑3.1‑70B. Its ${DEF('model states',TIP.states())} come to about ${fixd(18*MD['70b'].N/1e12,2)} TB, far more than one 80 GB H100 holds.</p>`+
     `<p>One way is to cut the model by layer: 10 layers on each of 8 GPUs, GPU 1 holding layers 1–10, GPU 2 layers 11–20, and so on. This is pipeline parallelism, and each GPU’s run of layers is one ${DEF('stage',TIP.stage())}. The number of stages is called p; here p = 8. Even so, each GPU still has to hold about ${fGB(st8.states)} of model states and it still does not fit; part 6 spreads them over more GPUs.</p>`+
     `<p>In a training step the data goes down from GPU 1 to GPU 8 (the forward pass). Once GPU 8 has the loss, the gradients go back up from GPU 8 to GPU 1 (the backward pass). Each GPU can start its part only when the stage before it has finished. If the whole batch goes in at once, how many of the 8 GPUs are working at any moment?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the result.</p>':''}`,
     `<p>你要训练 Llama‑3.1‑70B。它的${DEF('模型状态',TIP.states())}加起来约 ${fixd(18*MD['70b'].N/1e12,2)} TB，一张 80 GB 的 H100 远远放不下。</p>`+
     `<p>一种办法是按层切：8 张卡各放 10 层，第 1 张卡放第 1–10 层，第 2 张卡放第 11–20 层，依次往下。这叫流水线并行（pipeline parallelism），每张卡负责的那几层叫一${DEF('段',TIP.stage())}。流水线的段数记作 p，这里 p = 8。这样切完，每张卡还要存约 ${fGB(st8.states)} 的模型状态，仍然放不下，第 6 部分会再把它们分到更多卡上。</p>`+
     `<p>训练一步时，数据先从第 1 张卡往下走到第 8 张卡，这是前向。第 8 张卡算出 loss 以后，梯度再从第 8 张卡一段一段传回第 1 张卡，这是反向。每张卡都要等上一段算完，才能开始自己的部分。如果整批数据一次送进去，任何时刻，8 张卡里有几张在干活？</p>${ch}${pred?'<p class="muted small">已记下。第 2 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'gpipe',title:{en:'Micro-batches fill the pipeline',zh:'micro-batch 把流水线填满'},view:'gantt',expect:()=>[...mSeen].some(m=>m>=32),showMe:{m:32},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Cut the batch into m pieces: while one piece moves on, the next one follows.','把一批数据切成 m 份：前一份往下走，后一份马上跟上。'),
  enter:()=>({...BASE,m:1}),
  body:()=>tx(`<p>With the whole batch at once, GPU 1 finishes its forward and then can only wait for the other 7 GPUs and for the gradients to come back.</p>`+
    `<p>The fix is to cut the batch into m pieces, each a ${DEF('micro-batch',TIP.mb())}, so m is the number of micro-batches. GPU 1 hands the first piece to GPU 2 as soon as it is done and starts the second, so several GPUs work at the same time. This order is called GPipe: every micro-batch first does its forward pass, then the backward passes follow one after another.</p><p>Raise the number of micro-batches m and watch the ${DEF('bubble',TIP.bubble())}:</p>`,
    `<p>整批一次送进去时，第 1 张卡算完前向，就只能等后面 7 张卡算完、梯度传回来。</p>`+
    `<p>办法是把这一批切成 m 份，每份叫一个 ${DEF('micro-batch',TIP.mb())}，m 就是 micro-batch 数。第 1 张卡算完第 1 份就交给第 2 张卡，自己接着算第 2 份，几张卡就能同时干活。这种排法叫 GPipe：所有 micro-batch 先做完前向，再一份接一份做反向。</p><p>把 micro-batch 数 m 调大，看 ${DEF('bubble',TIP.bubble())} 怎么变：</p>`)+mCtl('mp-ic-m',true),
  obs:()=>obsGpipe()},
 {id:'1f1b',title:{en:'1F1B: the same bubble, far fewer activations kept',zh:'1F1B：bubble 一样，激活少存很多'},view:'gantt',expect:()=>tried3.has('1f1b'),showMe:()=>setLv3('1f1b'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('A different order keeps the bubble but frees activations early.','换一种顺序：bubble 不变，激活提早释放。'),
  enter:()=>{lv3='gpipe'; tried3=new Set(['gpipe']); return {...BASE,m:16,sched:'gpipe'};},
  body:()=>tx(`<p>1F1B (one forward, one backward) changes the order: GPU 1 first does p forwards to fill the p stages, then alternates one forward and one backward, and each backward frees the ${DEF('activations',TIP.act())} of its own micro-batch. GPipe starts the backward passes only after all m forwards, so GPU 1 holds the activations of all m micro-batches at once.</p>`+
    `<p>Activations are what each layer computes in the forward pass; the backward pass needs them, so they stay in memory until it has used them.</p><p>With p = 8 stages and m = 16 micro-batches, pick a schedule and see how many micro-batches of activations GPU 1 holds at once.</p>`,
    `<p>1F1B（one forward, one backward）换了顺序：第 1 张卡先做 p 个前向，把 p 段流水线填满，之后前向、反向轮流做，每做完一个反向，就释放那个 micro-batch 的${DEF('激活',TIP.act())}。GPipe 要等 m 个前向全做完才开始反向，所以第 1 张卡要同时存着全部 m 份激活。</p>`+
    `<p>激活是前向时每一层算出的结果，反向要用，用完才能释放。</p><p>流水线段数 p = 8、micro-batch 数 m = 16 时，选一种排法，看看第 1 张卡最多要同时存几份激活。</p>`)+
    levers('mp-ic-lv3',tx('Pipeline schedule, pick one','流水线的排法，选一种'),lv3,LV3,tried3,lv3Res),
  obs:()=>obs1f1b()},
 {id:'tp',title:{en:'Tensor parallelism: cut inside a layer',zh:'张量并行：在层里面切'},view:'layer',expect:()=>tSeen.has(8),showMe:{t:8},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Each GPU computes a slice of every matrix; the slices are summed twice per layer.','每张卡算每个矩阵的一块，每层合并两次结果。'),
  enter:()=>({...BASE,t:1}),
  body:()=>tx(`<p>Pipeline parallelism cuts between layers, and each GPU still computes whole layers. ${DEF('Tensor parallelism',A.terms&&A.terms.tp?A.terms.tp():'')} cuts inside a layer: every matrix multiply is shared by t GPUs, and each stores and computes only ${A.ui.math('\\frac{1}{t}')} of it.</p>`+
    `<p>Take the ${DEF('MLP block',A.terms&&A.terms.mlp?A.terms.mlp():'')} in the chart. Its gate and up matrices are cut by columns: each GPU gets the full input and computes only its own slice of the intermediate result. The down matrix is cut by rows: each GPU turns its slice into a partial result as large as the whole output, and the t partial results add up to the correct output.</p>`+
    `<p>Adding them up is an ${DEF('all-reduce',TIP.allreduce())}. The attention block hands its heads out to the GPUs (see ${labLink('mha')}) and is summed once at its end too, so each layer needs 2 all-reduces in the forward pass. In the backward pass the gradients flow the other way: a matrix cut by columns gives each GPU only part of the gradient of its input, which has to be summed too, so each layer needs 2 more in the backward pass.</p>`,
    `<p>流水线并行在层和层之间切，每张卡算的还是完整的层。${DEF('张量并行',A.terms&&A.terms.tp?A.terms.tp():'')}在层里面切：每个矩阵乘法由 t 张卡分担，每张卡只存、只算其中 ${A.ui.math('\\frac{1}{t}')}。</p>`+
    `<p>以图里的 ${DEF('MLP 块',A.terms&&A.terms.mlp?A.terms.mlp():'')}为例。gate 和 up 两个矩阵按列切，每张卡拿到完整的输入，只算出中间结果里属于自己的那一段。down 矩阵按行切，每张卡用自己那一段算出一份和输出一样大的部分结果，t 份加起来才是正确的输出。</p>`+
    `<p>把它们加起来的这一步叫 ${DEF('all-reduce',TIP.allreduce())}。attention 块按 head 分给各张卡（见${labLink('mha')}），最后也要合并一次，所以前向每层要做 2 次 all-reduce。反向时梯度倒着走。按列切的矩阵，每张卡只能算出输入梯度的一部分，这几部分也要加起来，所以反向每层还要再做 2 次。</p>`)+tCtl('mp-ic-t',true),
  obs:()=>obsTp()},
 {id:'link',title:{en:'Why it stays inside one machine',zh:'为什么只在一台机器里用'},view:'layer',expect:()=>tried5.has('out'),showMe:()=>setLv5('out'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('The same 8 GPUs: in one machine, or one in each of 8 machines?','同样 8 张卡：放在一台机器里，还是分到 8 台机器上？'),
  enter:()=>{lv5='in'; tried5=new Set(['in']); return {...BASE,t:8,place:'in'};},
  body:()=>tx(`<p>Inside one H100 machine the 8 GPUs are linked by ${DEF('NVLink',TIP.nvlink())}: each GPU sends 450 GB/s each way. Between machines they go over an ${DEF('InfiniBand network',TIP.ib())}, where the 8 GPUs of a machine share 400 GB/s: each GPU gets 50 GB/s each way, a ninth of NVLink.</p>`+
    `<p>Tensor parallelism does its all-reduces in every layer, and the next piece of work waits for each one to finish. Pick where the same 8 GPUs sit and see how much of the layer’s time goes to waiting for the all-reduces.</p>`,
    `<p>一台 H100 机器里的 8 张卡用 ${DEF('NVLink',TIP.nvlink())} 连着，每张卡每个方向 450 GB/s。机器和机器之间走 ${DEF('InfiniBand 网络',TIP.ib())}，一台机器的 8 张卡共用 400 GB/s 的网络带宽，每张卡每个方向分到 50 GB/s，只有 NVLink 的九分之一。</p>`+
    `<p>张量并行每一层都要做 all-reduce，而且后面的计算要等它做完才能开始。同样 8 张卡，选一种放法，看看这一层的时间里有多少花在等 all-reduce 上。</p>`)+
    levers('mp-ic-lv5',tx('Where the 8 GPUs sit, pick one','8 张卡放在哪里，选一种'),lv5,LV5,tried5,lv5Res),
  obs:()=>obsLink()},
 {id:'3d',title:{en:'Stacking the three',zh:'三种切法叠起来'},view:'cluster',label:{en:'Result',zh:'结果'},
  lead:()=>tx('Tensor parallel inside a machine, pipeline across machines, data parallel between pipelines.','机器里张量并行，机器之间流水线并行，流水线之间数据并行。'),
  enter:()=>({model:'70b',t:8,p:4,d:2,m:128,b:1,seq:SEQ,place:'in',zero:1,sched:'1f1b'}),
  body:()=>tx(`<p>Large runs stack the three cuts, giving the faster links to the cuts that talk more often. The chart shows one layout of Llama‑3.1‑70B on 8 machines, 64 H100s:</p><ul class="mp-ul">`+
    `<li>Tensor parallelism all-reduces in every layer, so it stays among the 8 GPUs of one machine, on NVLink.</li>`+
    `<li>The pipeline talks only at stage boundaries, so it can cross machines: 4 machines form one pipeline, p = 4 stages of 20 layers each.</li>`+
    `<li>${DEF('Data parallelism',TIP.dp())} talks once per step, so it goes between two pipelines, over the network between machines: there are 2 such pipelines, each taking half of the step’s 256 sequences, one sequence per micro-batch. The two pipelines also split the optimizer state in two with ${DEF('ZeRO-1',TIP.zero1())} (see ${labLink('parallelism')}).</li></ul>`,
    `<p>大模型训练把三种切法叠起来，通信越频繁的切法，放在越快的链路上。图里是 Llama‑3.1‑70B 在 8 台机器、64 张 H100 上的一种排法：</p><ul class="mp-ul">`+
    `<li>张量并行每一层都要做 all-reduce，所以放在一台机器的 8 张卡之间，走 NVLink。</li>`+
    `<li>流水线只在段的边界上通信，可以跨机器：4 台机器连成一条流水线，段数 p = 4，每段 20 层。</li>`+
    `<li>${DEF('数据并行',TIP.dp())}一步才通信一次，所以放在两条流水线之间，走机器之间的网络：这样的流水线有 2 条，每步的 256 条序列各算一半，每个 micro-batch 一条序列。两条流水线还用 ${DEF('ZeRO-1',TIP.zero1())} 把优化器状态分成两半存（见${labLinkZ('parallelism')}）。</li></ul>`),
  obs:()=>obs3d()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:'gantt',
  lead:()=>tx('Pipeline: mind the bubble. Tensor: mind the link. Stack them by speed of link.','流水线看 bubble，张量并行看链路，按链路快慢把它们叠起来。'),
  enter:()=>({...BASE,m:32}),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
const runDone=()=>ph>=1-1e-9;
function obsGpipe(){
  if(!runDone()) return '';
  const s=sim(8,S.m,'gpipe'), b=s.idle, b1=sim(8,1,'gpipe').idle;
  let h='';
  if(S.m===1) h=tx(`<p><b>With one batch (m = 1), only one GPU works at any moment; each GPU waits ${pct(b)} of the step.</b></p>`,`<p><b>只送一批（m = 1）时，任何时刻只有 1 张卡在干活，每张卡有 ${pct(b)} 的时间在等。</b></p>`)+verdict();
  if(S.m===1&&mSeen.size<2) return h+tx('<p>Now raise m.</p>','<p>再把 m 调大试试。</p>');   // the rule waits until a second m has been tried
  if(S.m!==1) h=tx(`<p><b>With p = 8 stages and m = ${S.m} micro-batches, each GPU waits ${pct(b)} of the step (${pct(b1)} with m = 1).</b></p>`,`<p><b>流水线段数 p = 8、micro-batch 数 m = ${S.m} 时，每张卡有 ${pct(b)} 的时间在等（m = 1 时是 ${pct(b1)}）。</b></p>`)+(S.m>=32?'':tx('<p>Raise it further.</p>','<p>再调大试试。</p>'));
  /* B2: where each GPU idles (GPipe, forward 1, backward 2): GPU 1 only in the middle, GPU 8 at the start and the end; p − 1 slots each */
  h+=tx('<p>A step is m + p − 1 slots long (one slot is one forward plus one backward), and each GPU works in only m of them, so every GPU idles p − 1 slots, whatever m is. The share of the step it waits is:</p>','<p>一步一共 m + p − 1 格（一格是一次前向加一次反向），每张卡只有 m 格在干活，所以每张卡都空等 p − 1 格，和 m 无关。空等占一步的比例是：</p>')+
    M('\\text{bubble} = \\frac{p - 1}{m + p - 1}')+
    tx(`<p>p is the number of stages and m the number of micro-batches. Here p = 8, so the top stays 7 while the bottom grows with m. The m you have tried:</p>`,`<p>p 是流水线段数，m 是 micro-batch 数。这里 p = 8，分子一直是 7，分母随 m 变大。你试过的几种 m：</p>`)+wxTable()+
    tx(`<p>So the larger m, the smaller the idle share, but it never reaches 0, because those p − 1 slots are always there.</p>`,`<p>所以 m 越大，空等的占比越小，但永远到不了 0，因为那 p − 1 格总在那里。</p>`);
  return h;
}
/* owner b8 (5): the formula with this part's numbers, one row per m the learner has tried (the current one in bold) */
function wxTable(){
  const ms=[...new Set([...mSeen,S.m])].sort((a,b)=>a-b);
  return `<table class="mp-wx"><thead><tr><th>${tx('micro-batches m','micro-batch 数 m')}</th><th>${A.ui.math('\\text{bubble} = \\frac{p - 1}{m + p - 1}')}</th></tr></thead><tbody>`+
    ms.map(m=>`<tr${m===S.m?' class="cur"':''}><td>${A.ui.math(`m = ${m}`)}</td><td>${A.ui.math(`\\frac{8 - 1}{${m} + 8 - 1} = \\frac{7}{${m+7}}`)}${tx(', about ','，约 ')}${pct(sim(8,m,'gpipe').idle)}</td></tr>`).join('')+`</tbody></table>`;
}
function obs1f1b(){
  if(!runDone()) return '';
  const g=p3('gpipe'), f=p3('1f1b'), mb=cost({...BASE,m:16,sched:'1f1b'}).actMb;
  if(lv3==='gpipe') return tx(`<p><b>GPipe: GPU 1 holds the activations of all m = ${g.peak} micro-batches at once, and each GPU waits ${pct(g.idle)} of the step.</b></p><p>Now pick 1F1B.</p>`,`<p><b>GPipe：第 1 张卡同时存着全部 m = ${g.peak} 个 micro-batch 的激活，每张卡有 ${pct(g.idle)} 的时间在等。</b></p><p>再选 1F1B 试试。</p>`);
  return tx(`<p><b>1F1B: GPU 1 holds at most p = ${f.peak} micro-batches of activations instead of m = ${g.peak}, and the bubble is still ${pct(f.idle)}.</b></p>`+
    `<p>The bubble stays, because 1F1B only changes the order: each GPU still idles p − 1 slots in total.</p>`+
    `<p>Still, these ${f.peak} micro-batches of activations take about ${fGB(f.peak*mb)}, more than an 80 GB card holds. Part 6 adds tensor parallelism, which shrinks each one to an eighth. These numbers already assume selective recomputation; recomputing every layer would keep even less (see ${labLink('training-memory')}).</p>`,
    `<p><b>1F1B：第 1 张卡最多只存 p = ${f.peak} 个 micro-batch 的激活，而不是 m = ${g.peak} 个。bubble 还是 ${pct(f.idle)}。</b></p>`+
    `<p>bubble 没变，因为 1F1B 只换了顺序，每张卡空等的加起来还是 p − 1 格。</p>`+
    `<p>可是这 ${f.peak} 份激活约 ${fGB(f.peak*mb)}，还是放不进 80 GB 的卡。第 6 部分加上张量并行，每份只剩八分之一。这里已经按选择性重算来算，每一层都重算的话还能再少存（见${labLinkZ('training-memory')}）。</p>`);
}
function obsTp(){
  const r=layer(S.t,'in'), r1=layer(1,'in');
  if(S.t===1) return tx(`<p><b>On one GPU, one layer of one 4,096-token micro-batch takes ${fMs(r1.comp)} to compute (forward and backward).</b> Split it across more GPUs.</p>`,`<p><b>一张卡算一层（一个 4,096 个 token 的 micro-batch，前向加反向）要 ${fMs(r1.comp)}。</b>把它切到更多卡上试试。</p>`);
  return tx(`<p><b>Split ${S.t} ways, each GPU computes the layer in ${fMs(r.comp)} (${fMs(r1.comp)} on one GPU), but then waits ${fMs(r.comm)} for the 4 all-reduces: ${pct(r.share)} of the layer’s time.</b></p>`+
    `<p>Each all-reduce carries the layer’s whole output for the micro-batch, 4,096 tokens × 8,192 dimensions × 2 bytes = ${fMB(r.bytes)}, however finely the layer is cut. So the finer the cut, the less each GPU computes, while the communication stays nearly the same. Its time is:</p>`,
    `<p><b>切成 ${S.t} 份，每张卡算这一层只要 ${fMs(r.comp)}（一张卡时是 ${fMs(r1.comp)}），但之后要等 4 次 all-reduce，共 ${fMs(r.comm)}，占这一层时间的 ${pct(r.share)}。</b></p>`+
    `<p>每次 all-reduce 传的是这个 micro-batch 在这一层的整份输出：4,096 个 token × 8,192 维 × 2 字节 = ${fMB(r.bytes)}，层切得再细也是这么大。所以切得越细，每张卡算得越少，通信却几乎不变。一次 all-reduce 的时间是：</p>`)+
    M(tx('\\text{time} = \\frac{2(t - 1)}{t} \\times \\frac{\\text{bytes}}{\\text{bandwidth}}','\\text{时间} = \\frac{2(t - 1)}{t} \\times \\frac{\\text{字节数}}{\\text{带宽}}'));
}
function obsLink(){
  const a=layer(8,'in'), b=layer(8,'out');
  if(lv5==='in') return tx(`<p><b>In one machine: the 4 all-reduces of a layer take ${fMs(a.comm)}, ${pct(a.share)} of the layer’s time.</b> Now spread the GPUs over 8 machines.</p>`,`<p><b>在一台机器里：一层的 4 次 all-reduce 共 ${fMs(a.comm)}，占这一层时间的 ${pct(a.share)}。</b>再把卡分到 8 台机器上试试。</p>`);
  /* S9c: the exact values stay on the cards; the sentence gives the ratio and a share in words */
  const k=Math.round(b.comm/a.comm), part=b.share>=0.6&&b.share<2/3?tx('nearly two thirds','将近三分之二'):(b.share>0.5?tx('more than half','一半多'):pct(b.share));
  return tx(`<p><b>Across 8 machines the all-reduces take ${k} times as long as before, longer than the layer’s compute. Of the layer’s time, each GPU spends ${part} waiting for the network.</b></p>`+
    `<p>That is why tensor parallelism stays inside one machine, and the cut between machines is left to pipeline parallelism. A pipeline sends one micro-batch’s activations across each stage boundary, instead of an all-reduce in every layer.</p>`,
    `<p><b>分到 8 台机器上以后，all-reduce 的时间变成原来的 ${k} 倍，比这一层的计算还长。这一层的时间里，每张卡有${part}在等网络。</b></p>`+
    `<p>所以张量并行只在一台机器里面用，机器之间交给流水线并行。流水线只在段和段的边界上传一次 micro-batch 的激活，不用每一层都做 all-reduce。</p>`);
}
function obs3d(){
  const r=cost(S);
  const wait=r.bubble+r.tpShare;                          // S9a: one number, the two kinds of waiting together (cards show each)
  return tx(`<p><b>Each GPU uses only ${fGB(r.mem)}, so it fits. The bubble takes ${pct(r.bubble)} of the step and waiting for tensor-parallel all-reduces ${pct(r.tpShare)}, together about ${pct(wait)}.</b></p>`+
    `<p class="small mp-two">Two more cuts:</p><ul class="mp-ul small"><li>Context parallelism: when sequences get very long, one sequence is spread over several GPUs.</li><li>Expert parallelism: an ${labLink('moe')} model puts different experts on different GPUs.</li></ul>`,
    `<p><b>每张卡只用 ${fGB(r.mem)}，放得下。一步里，bubble 占 ${pct(r.bubble)}，等张量并行的通信占 ${pct(r.tpShare)}，加起来约 ${pct(wait)}。</b></p>`+
    `<p class="small mp-two">还有两种切法：</p><ul class="mp-ul small"><li>上下文并行（context parallelism）：序列很长时，把一条序列切到几张卡上。</li><li>专家并行（expert parallelism）：${labLink('moe')} 模型把不同的专家放到不同的卡上。</li></ul>`);
}
function remember(){
  /* G2 (the owner, 2026-10-08): each point keeps its rule and says why it matters or what follows; no point leans on a number or a
     setting of an earlier part. The chart beside it is part 2's pipeline at m = 32, so point 1 may point at it. */
  return tx(`<ul class="mp-remlist"><li><b>Pipeline parallelism pays with the bubble.</b> Each GPU always idles p − 1 slots, and every row of the chart keeps that hatched time. The more micro-batches, the smaller its share of the step, so make the number of micro-batches m much larger than the number of stages p.</li>`+
    `<li><b>1F1B keeps the bubble and saves memory.</b> GPipe makes GPU 1 keep the activations of all m micro-batches; 1F1B keeps at most p. So with 1F1B you can raise m to shrink the bubble without keeping more activations.</li>`+
    `<li><b>Tensor parallelism talks in every layer, so it stays inside one machine.</b> The network between machines is far slower than NVLink, so across machines every all-reduce takes several times as long. Between machines, use the pipeline and data parallelism, which talk far less often.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Pipeline parallelism is free.” It has the bubble, which more micro-batches shrink but never remove, and GPU 1 keeps activations for several micro-batches at once.</p>`+
    `<p>✗ “The finer the tensor-parallel cut, the faster.” One layer does finish sooner, but with a fixed number of GPUs the whole run is not necessarily faster. The finer the cut, the less each GPU computes while each all-reduce stays the same size, so waiting takes a growing share. That waiting is time the GPUs could have spent on other data.</p>`+
    `<p>✗ “More micro-batches are always better.” The step’s batch is set by what the optimizer needs, so more micro-batches means smaller ones. A micro-batch has at least one sequence, and smaller matrix multiplies run less efficiently (Megatron-LM’s README). This lab does not model that loss.</p></div>${A.shell.toScenarios()}`,
    `<ul class="mp-remlist"><li><b>流水线并行的代价是 bubble。</b>每张卡总要空等 p − 1 格，图里每一行都留着这几格斜纹。micro-batch 越多，这几格在一步里占得越少，所以 micro-batch 数 m 要比段数 p 大得多。</li>`+
    `<li><b>1F1B 不改 bubble，只省显存。</b>GPipe 让第 1 张卡存下全部 m 份激活，1F1B 最多只存 p 份。所以用 1F1B 时，把 m 调大来压小 bubble，激活也不会跟着变多。</li>`+
    `<li><b>张量并行每一层都要通信，所以只在一台机器里用。</b>机器之间的网络比 NVLink 慢得多，跨机器做张量并行，每次 all-reduce 都要多等好几倍的时间。机器之间交给流水线和数据并行，它们通信的次数少得多。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “流水线并行没有代价。”它有 bubble，micro-batch 切得再多也降不到 0。第 1 张卡还要同时存好几份激活。</p>`+
    `<p>✗ “张量并行切得越细越快。”一层是算得更快了，可卡数固定时，整体不一定更快。切得越细，每张卡算得越少，每次 all-reduce 却一样大，等通信的时间占得越来越多。这些等通信的时间，本来可以拿去算别的数据。</p>`+
    `<p>✗ “micro-batch 越多越好。”一步的 batch 由优化器的需要决定，micro-batch 越多，每份就越小。一份至少要有一条序列，而且矩阵乘法越小，GPU 跑得越没效率（Megatron-LM 的 README）。这个 Lab 没有算这部分损失。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'one',t:{en:'70B on one machine',zh:'一台机器训练 70B'},d:{en:'8 GPUs, tensor parallelism alone',zh:'8 张卡，只用张量并行'},s:{model:'70b',t:8,p:1,d:1,m:16,b:1,seq:SEQ,place:'in',zero:1,sched:'1f1b'}},
 {id:'2d',t:{en:'70B on 256 GPUs, like torchtitan',zh:'像 torchtitan 那样，256 张卡训练 70B'},d:{en:'Tensor parallel in each machine, 32 data-parallel copies',zh:'机器里张量并行，32 份数据并行'},s:{model:'70b',t:8,p:1,d:32,m:16,b:1,seq:SEQ,place:'in',zero:1,sched:'1f1b'}},
 {id:'llama3',t:{en:'405B on 16,384 GPUs, like Llama 3',zh:'像 Llama 3 那样，16,384 张卡训练 405B'},d:{en:'16 pipeline stages (p = 16), tensor parallel 8, 128 data-parallel copies',zh:'流水线 16 段（p = 16），张量并行 8 份，数据并行 128 份'},s:{model:'405b',t:8,p:16,d:128,m:32,b:1,seq:SEQ,place:'in',zero:1,sched:'1f1b'}},
 {id:'long',t:{en:'405B with 128K-token sequences',zh:'405B，每条序列 128K 个 token'},d:{en:'Long-context training on 512 GPUs',zh:'512 张卡上的长上下文训练'},s:{model:'405b',t:8,p:8,d:8,m:1,b:1,seq:131072,place:'in',zero:1,sched:'1f1b'}}
];
const SKEYS=['model','t','p','d','m','b','seq','place','zero','sched'];
const presetNow=()=>PRESETS.findIndex(p=>SKEYS.every(k=>p.s[k]===S[k]));
function scenHTML(i){
  const p=PRESETS[i], r=cost(S);
  if(p.id==='one') return tx(`<p><b>It does not fit: each GPU would need ${fGB(r.mem)}, of which ${fGB(r.states)} are model states.</b></p><p>Tensor parallelism splits the 70B model’s 1.27 TB of model states only 8 ways, and with a single copy ZeRO-1 has nothing to split. More GPUs are needed: pipeline stages across machines, or data-parallel copies that split the states further (FSDP shards all of them).</p>`,
                             `<p><b>放不下：每张卡要 ${fGB(r.mem)}，其中模型状态就有 ${fGB(r.states)}。</b></p><p>张量并行只能把 70B 那 1.27 TB 的模型状态分成 8 份。数据并行只有一份，没有别的副本来分担，ZeRO-1 也省不下显存。要用更多卡：在机器之间加流水线，或者加数据并行的份数，把状态再分开（FSDP 会把它们全部分开存）。</p>`);
  if(p.id==='2d') return tx(`<p><b>It fits: ${fGB(r.mem)} per GPU, with no pipeline and so no bubble. The price is the gradient all-reduce across 32 machines at the end of each step: ${fS(r.dp)} here.</b></p><p>torchtitan benchmarked Llama 3.1 70B on 256 H100s as FSDP 32 × TP 8 (${cite(SRC.titan,'torchtitan benchmarks, Table 4')}). FSDP shards the weights and gradients as well, so with the same layout it needs less memory for the model states than ZeRO-1 does here. torchtitan’s H100s are the 96 GB version.</p>`,
                            `<p><b>放得下：每张卡 ${fGB(r.mem)}，没有流水线，也就没有 bubble。代价是每步结束时跨 32 台机器做一次梯度 all-reduce，这里要 ${fS(r.dp)}。</b></p><p>torchtitan 在 256 张 H100 上测过 Llama 3.1 70B，用的是 FSDP 32 × 张量并行 8（${cite(SRC.titan,'torchtitan 的测试，表 4')}）。FSDP 连权重和梯度也分开存，同样的排法下，它存模型状态用的显存比这里的 ZeRO-1 少。torchtitan 用的 H100 是 96 GB 的版本。</p>`);
  if(p.id==='llama3') return tx(`<p><b>It fits (${fGB(r.mem)} per GPU), but with plain 1F1B each GPU waits ${pct(r.bubble)} of the step: each of the 128 pipelines gets only m = 32 micro-batches for its p = 16 stages.</b></p><p>The scaling book quotes Llama 3’s layout: 16-way pipeline parallelism, 8-way tensor parallelism inside a node, 128-way ZeRO-1 data parallelism, 16M tokens per step (${cite(SRC.book,'scaling book, gpus.md')}). This lab assumes 4,096-token sequences. A schedule that gives each GPU several smaller chunks of layers (interleaved 1F1B) shrinks the bubble; on 405B, torchtitan measured 128 instead of 100 tokens per GPU per second with it (${cite(SRC.titan,'Table 5')}).</p>`,
                                `<p><b>放得下（每张卡 ${fGB(r.mem)}），但用普通的 1F1B 时，每张卡有 ${pct(r.bubble)} 的时间在等。128 条流水线每条只分到 m = 32 个 micro-batch，段数却有 p = 16。</b></p><p>scaling book 引用了 Llama 3 的排法：流水线 16 段，机器里张量并行 8 份，ZeRO-1 数据并行 128 份，每步 16M 个 token（${cite(SRC.book,'scaling book 的 gpus.md')}）。这个 Lab 假设每条序列 4,096 个 token。如果让每张卡负责几小段不相连的层（交错式 1F1B），bubble 会变小：torchtitan 在 405B 上测到，换成它以后，每张卡每秒处理的 token 从 100 个涨到 128 个（${cite(SRC.titan,'表 5')}）。</p>`);
  return tx(`<p><b>It does not fit: one 128K-token sequence alone needs ${fGB(r.act)} of activations on GPU 1, ${fGB(r.mem)} in all.</b></p><p>Activations grow linearly with the sequence length. Tensor parallelism already splits them 8 ways and the pipeline only separates the layers, so neither stops them from growing with the length. This lab also leaves out attention’s compute, which grows with the square of the length.</p>`+
            `<p>Context parallelism cuts the whole sequence across GPUs (${cite(SRC.megCP,'Megatron-LM, context_parallel.md')}). torchtitan benchmarked 405B on 262,144-token sequences with every layer recomputed (full activation checkpointing) plus 8-way context parallelism (${cite(SRC.titan,'Table 6')}).</p>`,
            `<p><b>放不下：一条 128K 个 token 的序列，光激活就要在第 1 张卡上占 ${fGB(r.act)}，一共 ${fGB(r.mem)}。</b></p><p>激活随序列长度线性增长。张量并行已经把它分成了 8 份，流水线只是把层分开，都挡不住它随长度变大。这个 Lab 也没有算 attention 的计算，它随长度的平方增长。</p>`+
            `<p>上下文并行把整条序列切开，分到几张卡上（${cite(SRC.megCP,'Megatron-LM 的 context_parallel.md')}）。torchtitan 在 405B 上测过 262,144 个 token 的序列，每一层都重算，再加上 8 份上下文并行（${cite(SRC.titan,'表 6')}）。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=e=>tx(`PP ${e.p} × TP ${e.t} × DP ${e.d}, m = ${e.m}`,`流水线 ${e.p} 段 × 张量并行 ${e.t} 份 × 数据并行 ${e.d} 份，m = ${e.m}`);
/* R1 (owner b8): the verdict first; then your layout next to the fastest one, what they share written once and only the settings
   that differ shown; red for what fails; one sentence on the main gap and one on the single change that helps most; the layouts
   that pass behind a closed disclosure. Times are rounded the way a person says them; there is no hover that only expands a number
   to its exact value (the owner, 2026-10-07). Memory stays as the cards write it, since it is compared with the 80 GB limit. */
const pctR=x=>`${Math.max(1,Math.round(x*100))}%`;   // a share the way a person says it
const rS=x=>{const v=x>=0.95?String(Math.round(x)):fixd(x,1); return tx(`about ${v} s`,`约 ${v} 秒`);};
const exS=rS, exGB=fGB;
/* where a step's time goes: computing, waiting for the tensor-parallel all-reduces, the bubble, the data-parallel sync (they add up) */
const partsOf=e=>{const r=e.r; return {u:r.comp*e.m,tp:r.comm*e.m,bb:r.per*(e.p-1),dp:r.dp};};
const PARTN=()=>[{k:'u',n:tx('computing','在算')},{k:'tp',n:tx('tensor-parallel all-reduces','等张量并行的 all-reduce')},{k:'bb',n:tx('bubble','bubble')},{k:'dp',n:tx('data-parallel sync','等数据并行同步')}];
/* the m the task card takes when a new t or p no longer divides the batch: the nearest valid m by ratio, the smaller one on a tie.
   The click handler uses this same function, so “the card then sets m = …” always says what the card does. */
function chNearM(t,p,m){if(chCfg(t,p,m)) return m; const ok=CH.Ms.filter(x=>chCfg(t,p,x)); return ok.length?ok.reduce((a,x)=>Math.abs(Math.log(x/m))<Math.abs(Math.log(a/m))?x:a,ok[0]):null;}
/* every single change the task card allows from layout e (a new t, p or m), with the layout it leads to */
function chChanges(e){
  const out=[];
  CH.Ts.forEach(t=>{if(t!==e.t&&Number.isInteger(CH.G/(t*e.p))){const m=chNearM(t,e.p,e.m); if(m) out.push({k:'t',v:t,e:chEval(t,e.p,m)});}});
  CH.Ps.forEach(p=>{if(p!==e.p&&Number.isInteger(CH.G/(e.t*p))){const m=chNearM(e.t,p,e.m); if(m) out.push({k:'p',v:p,e:chEval(e.t,p,m)});}});
  CH.Ms.forEach(m=>{if(m!==e.m&&chCfg(e.t,e.p,m)) out.push({k:'m',v:m,e:chEval(e.t,e.p,m)});});
  return out.filter(c=>c.e);
}
/* G7 (Codex on b9 preview v5): a step time as the task card shows it. A change counts as faster only when this shown value drops
   (33.4 s → 33.2 s); 33.2 s → 33.3 s is slower, and a change that leaves the shown value as it is does not help either. */
const shownStep=s=>parseFloat(fS(s).replace(/,/g,''));
/* the single change the result names. A layout that does not fit: of the changes that make it fit, the one with the shortest step.
   A layout that fits but is slow: the change with the shortest step that still fits, and only when the shown step time drops;
   otherwise none, and the result says that no single change helps. */
function chOneChange(e){
  const c=chChanges(e).filter(x=>x.e.fits).sort((a,b)=>a.e.step-b.e.step)[0]||null;
  if(!c||!e.fits) return c;
  return shownStep(c.e.step)<shownStep(e.step)?c:null;
}
const CHG=()=>[{k:'p',chip:e=>e.p===1?tx('no pipeline (p = 1)','不分段（p = 1）'):tx(`p = ${e.p} stages`,`流水线 p = ${e.p} 段`),same:e=>e.p===1?tx('both without a pipeline','都不分段'):tx(`both with p = ${e.p} stages`,`流水线都是 p = ${e.p} 段`)},
  {k:'t',chip:e=>e.t===1?tx('no tensor parallelism','不做张量并行'):tx(`TP ${e.t}`,`张量并行 ${e.t} 份`),same:e=>e.t===1?tx('both without tensor parallelism','都不做张量并行'):tx(`both with tensor parallelism ${e.t} ways`,`张量并行都切 ${e.t} 份`)},
  {k:'d',chip:e=>tx(`DP ${e.d}`,`数据并行 ${e.d} 份`),same:e=>tx(`both with ${e.d} data-parallel ${e.d===1?'copy':'copies'}`,`数据并行都是 ${e.d} 份`)},
  {k:'m',chip:e=>tx(`m = ${e.m}`,`m = ${e.m}`),same:e=>tx(`both with m = ${e.m} micro-batches`,`都切成 m = ${e.m} 个 micro-batch`)}];
/* one change in words; p = 1 and t = 1 are named as the chips name them (“不分段”, “不做张量并行”), not as “1 段” or “1 份” */
const chgName=(k,a,b)=>({t:a===1?tx(`tensor parallelism ${b} ways instead of none`,`改用张量并行 ${b} 份`):(b===1?tx(`no tensor parallelism instead of ${a} ways`,`改成不做张量并行`):tx(`tensor parallelism ${b} ways instead of ${a}`,`把张量并行从 ${a} 份改成 ${b} 份`)),
  p:a===1?tx(`a pipeline of ${b} stages instead of none`,`改用 ${b} 段流水线`):(b===1?tx(`no pipeline instead of ${a} stages`,`改成不分流水线`):tx(`${b} pipeline stages instead of ${a}`,`把流水线从 ${a} 段改成 ${b} 段`)),
  m:tx(`m = ${b} micro-batches instead of ${a}`,`把 micro-batch 数从 m = ${a} 改成 m = ${b}`)})[k];
/* G7: the settings a target layout has and layout `from` does not, as targets (d follows from t and p), joined the way each
   language lists them */
function chTargets(from,to){
  const it=[];
  if(to.p!==from.p) it.push(to.p===1?tx(`no pipeline instead of ${from.p} stages`,'改成不分流水线'):from.p===1?tx(`a pipeline of ${to.p} stages instead of none`,`改用 ${to.p} 段流水线`):tx(`${to.p} pipeline stages instead of ${from.p}`,`把流水线改成 ${to.p} 段`));
  if(to.t!==from.t) it.push(to.t===1?tx(`no tensor parallelism instead of ${from.t} ways`,'改成不做张量并行'):from.t===1?tx(`tensor parallelism ${to.t} ways instead of none`,`改用张量并行 ${to.t} 份`):tx(`tensor parallelism ${to.t} ways instead of ${from.t}`,`把张量并行改成 ${to.t} 份`));
  if(to.m!==from.m) it.push(tx(`m = ${to.m} micro-batches instead of ${from.m}`,`把 micro-batch 数改成 m = ${to.m}`));
  const n=it.length, list=A.lang()==='zh'?it.join('、'):(n<3?it.join(' and '):it.slice(0,-1).join(', ')+' and '+it[n-1]);
  return {n,list,many:tx(['','one setting','two settings','three settings'][n],['','一项','两项','三项'][n])};
}
/* the layout that meets the goal with the fewest settings changed from `from` (then the faster one) */
function chNearestGoal(from){
  const dn=e=>['p','t','m'].filter(k=>e[k]!==from[k]).length;
  return chAll().filter(chMinimal).sort((a,b)=>dn(a)-dn(b)||a.step-b.step)[0];
}
/* a step time with the digits of the task card, in the reader's language */
const fSw=s=>{const v=fS(s).replace(/ s$/,''); return tx(`${v} s`,`${v} 秒`);};
/* the sentence after the main gap (G7, Codex on b9 preview v5): name a single change only when it really helps as the page shows it,
   with the step time before and after (or, for a layout that does not fit, what the change costs); otherwise say that no single
   change helps and name what the fastest layout changes together */
function chNextHTML(me,best){
  const oc=chOneChange(me), sp=A.lang()==='zh'?'':' ';
  if(!oc){const g=chTargets(me,best), all=tx(`The fastest layout changes ${g.many} at once: ${g.list}.`,`最快的排法要同时${g.list}。`);
    return (me.fits?tx('No single change makes a step shorter.','不管只改哪一项，每步都不会变短。'):tx('No single change makes it fit.','只改一项还放不下。'))+sp+all;}
  const ch=chgName(oc.k,me[oc.k],oc.v), top=chMinimal(oc.e), cm=oc.k!=='m'&&oc.e.m!==me.m;
  const card=cm?tx(` (the card then sets m = ${oc.e.m})`,`（任务卡会把 m 随之改成 ${oc.e.m}）`):'';
  if(me.fits){
    const h=tx(`The best single change is ${ch}${card}: a step goes from ${fSw(me.step)} to ${fSw(oc.e.step)}${top?', which meets the goal.':', still short of the goal.'}`,
               `只改一项的话，${ch}最划算${card}。每步从 ${fSw(me.step)}缩短到 ${fSw(oc.e.step)}，${top?'改完就达标了':'但还没达标'}。`);
    if(top) return h;
    const g=chTargets(oc.e,chNearestGoal(oc.e));
    return h+sp+(g.n>1?tx(`To meet it, also change ${g.many} at once: ${g.list}.`,`要达标，还要同时${g.list}。`):tx(`To meet it, also change to ${g.list}.`,`要达标，还要${g.list}。`));}
  return tx(`Of the single changes that make it fit, ${ch}${card} gives the shortest step. Each GPU then uses ${fGB(oc.e.r.mem)} and a step takes ${fSw(oc.e.step)}${top?', which meets the goal.':`, about ${pctR(oc.e.step/best.step-1)} slower than the fastest layout.`}`,
            `只改一项就放得下的改法里，${ch}每步最短${card}。改完每张卡用 ${fGB(oc.e.r.mem)}，每步 ${fSw(oc.e.step)}，${top?'就达标了':`比最快的排法慢约 ${pctR(oc.e.step/best.step-1)}`}。`);
}
function chSumHTML(me,best){
  const isBest=chMinimal(me)&&me.t===best.t&&me.p===best.p&&me.m===best.m, rows=[{e:me,role:isBest?'mebest':'me'}].concat(isBest?[]:[{e:best,role:'best'}]);
  const G=CHG(), diff=rows.length===1?G:G.filter(g=>!rows.every(x=>x.e[g.k]===rows[0].e[g.k])), same=G.filter(g=>!diff.includes(g));
  const MX=Math.max(...rows.filter(x=>x.e.fits).map(x=>x.e.step)), PN=PARTN();
  const sameTxt=rows.length===2?tx(`Both layouts train Llama‑3.1‑70B on 32 H100s with 256 sequences per step, ZeRO-1 and 1F1B${same.map(g=>', '+g.same(me)).join('')}.`,
                   `两种排法都在 32 张 H100 上训练 Llama‑3.1‑70B，每步 256 条序列，都用 ZeRO-1 和 1F1B${same.map(g=>'，'+g.same(me)).join('')}。`)
    :tx('Llama‑3.1‑70B on 32 H100s, 256 sequences per step, ZeRO-1, 1F1B.','条件：32 张 H100 训练 Llama‑3.1‑70B，每步 256 条序列，ZeRO-1，1F1B。');
  const lab={me:tx('Your layout','你的排法'),mebest:tx('Your layout, the fastest one','你的排法，就是最快的'),best:DEF(tx('Fastest','最快的排法'),tx('The layout that fits in 80 GB with the shortest step. Layouts within 1% of it meet the goal too.','放得下 80 GB 的排法里，每步最短的那种。和它相差不到 1% 的排法也达标。'))};
  const row=x=>{const e=x.e, r=e.r, chips=diff.map(g=>`<span class="mp-chip${x.role==='me'&&e[g.k]!==best[g.k]?' on':''}">${esc(g.chip(e))}</span>`).join('');
    let mid;
    if(!e.fits) mid=`<div class="mp-c3x bad">${tx(`Does not fit: ${fGB(r.mem)} per GPU, more than 80 GB`,`放不下：每张卡要 ${fGB(r.mem)}，超过了 80 GB`)}</div>`;
    else {const P=partsOf(e), slow=!chMinimal(e);
      mid=`<div class="mp-c3b"><div class="mp-c3trk"><div class="mp-c3fill" style="width:${(e.step/MX*100).toFixed(2)}%">`+
        PN.filter(q=>P[q.k]>0).map(q=>`<i class="mp-sg ${q.k}" style="width:${(P[q.k]/e.step*100).toFixed(3)}%" data-tipx="${esc(`${q.n}${tx(': about ','：约 ')}${pct(P[q.k]/e.step)}${tx(' of the step',' 的时间')}`)}"></i>`).join('')+
        `</div></div><b class="mp-c3d mono${slow?' bad':''}">${exS(e.step)}</b></div>`+
        `<div class="mp-c3s">${tx('each GPU uses ','每张卡用 ')}${exGB(r.mem)}${slow?`<span class="mp-c3m bad">${tx(`, each step about ${pctR(e.step/best.step-1)} slower than the fastest`,`，每步比最快的慢约 ${pctR(e.step/best.step-1)}`)}</span>`:''}</div>`;}
    return `<li class="mp-c3r ${x.role}" data-role="${x.role}"><div class="mp-c3h"><span class="mp-c3t">${lab[x.role]}</span>${chips}</div>${mid}</li>`;};
  let h=`<div class="mp-chsum"><div class="mp-c3cap">${tx(rows.length===2?'How long does a step take with each layout, and where does the time go?':'How long does a step take, and where does the time go?',rows.length===2?'两种排法每步各要多久？时间花在了哪里？':'每步要多久？时间花在了哪里？')}</div><p class="small mp-same">${sameTxt}</p><ul class="mp-c3">${rows.map(row).join('')}</ul>`+
    `<div class="mp-c3leg small">${PN.map(q=>`<span><i class="mp-sw3 ${q.k}"></i>${esc(q.n)}</span>`).join('')}</div>`;
  /* the main gap (a layout that fits but is slower) and the best single change (any layout that misses) */
  if(me.fits&&!chMinimal(me)){const Pm=partsOf(me), Pb=partsOf(best), ext=me.step-best.step, k=Object.keys(Pm).reduce((a,b)=>Pm[b]-Pb[b]>Pm[a]-Pb[a]?b:a), nm=PN.find(q=>q.k===k).n;
    h+=`<p class="mp-gap">${tx(`Your layout takes ${rS(ext)} longer per step than the fastest. ${(Pm[k]-Pb[k])/ext>=0.5?'Most of that goes to':'The largest share goes to'} ${nm}.`,
      `你的排法每步比最快的多${rS(ext)}，${(Pm[k]-Pb[k])/ext>=0.5?`多出来的大部分花在${nm}上`:`多出来的时间里，花得最多的是${nm}`}。`)}</p>`;}
  if(!chMinimal(me)) h+=`<p class="mp-next">${chNextHTML(me,best)}</p>`;
  return h+`</div>`;
}
/* the layouts that pass, closed by default; the reason for TP 4 and the overlap caveat with them */
function chAllHTML(){
  const pass=chAll().filter(chMinimal).sort((a,b)=>a.step-b.step), rows=chAll().filter(e=>e.fits), t8=rows.filter(e=>e.t===8).sort((a,b)=>a.step-b.step)[0], b=chBest(), half=b.r.tpShare/t8.r.tpShare;
  const halfW=half>0.4&&half<0.6?tx('only about half','只有 8 份时的一半左右'):tx(`${fixd(half,1)} times`,`是 8 份时的 ${fixd(half,1)} 倍`);   // S9h: 0.47 here; the numbers test checks it
  return `<details class="mp-all"><summary>${tx(`See all layouts that pass (${pass.length})`,`看全部达标的排法（${pass.length} 种）`)}</summary><div class="mp-allb"><ul class="mp-pass">`+
    pass.map(e=>`<li>${esc(chName(e))}${tx(': ','：每步')}${exS(e.step)}${tx(' per step, ','，每张卡 ')}${exGB(e.r.mem)}${tx(' per GPU','')}</li>`).join('')+`</ul>`+
    tx(`<p class="small">Both use tensor parallelism 4, not 8. With 4, each GPU computes twice as much per layer while each all-reduce stays the same size, so the all-reduces take ${halfW} the time they take with 8. The pipeline and ZeRO-1 make up the memory, and with many micro-batches the bubble stays small.</p>`+
       `<p class="small muted">Note: this lab puts every all-reduce on the critical path. Megatron-LM can overlap tensor-parallel communication with compute (--tp-comm-overlap), which makes tensor parallelism 8 cheaper than shown. Measure both on your cluster before you choose.</p>`,
       `<p class="small">两种都用张量并行 4 份，而不是 8 份。切成 4 份时，每张卡每层的计算多了一倍，每次 all-reduce 却一样大，所以 all-reduce 占的时间${halfW}。显存由流水线和 ZeRO-1 补上。micro-batch 切得多，bubble 也很小。</p>`+
       `<p class="small muted">注意：这个 Lab 把每次 all-reduce 都算在关键路径上。Megatron-LM 可以让张量并行的通信和计算重叠（--tp-comm-overlap），那样张量并行 8 份会比这里算的便宜。选之前，在自己的集群上把两种都测一下。</p>`)+`</div></details>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.t,S.p,S.m), best=chBest();
  if(!me) return '';
  let h;
  if(me.fits&&chMinimal(me)) h=tx(`<p class="okline mp-mt"><b>Goal met: it fits (${exGB(me.r.mem)} per GPU), and a step takes ${exS(me.step)}.</b> No layout that fits is more than 1% faster.</p>`,`<p class="okline mp-mt"><b>达成目标：放得下（每张卡 ${exGB(me.r.mem)}），每步${exS(me.step)}。</b>放得下的排法里，没有哪种比它快 1% 以上。</p>`);
  else if(me.fits) h=tx(`<p class="badline mp-mt"><b>It fits, but it is not the fastest: ${exS(me.step)} per step, about ${pctR(me.step/best.step-1)} slower than the fastest layout.</b></p>`,`<p class="badline mp-mt"><b>放得下，但不是最快的：每步${exS(me.step)}，比最快的排法慢约 ${pctR(me.step/best.step-1)}。</b></p>`);
  else h=tx(`<p class="badline mp-mt"><b>Not yet: each GPU would need ${exGB(me.r.mem)}, more than its 80 GB.</b></p><p class="small">Hint: the larger tensor parallelism times the pipeline stages, the less model state each GPU stores. The more micro-batches, the smaller each one and the fewer activations; the finer the tensor-parallel cut, the fewer activations too. Among the layouts that fit, which one spends the least time in all-reduces?</p>`,
              `<p class="badline mp-mt"><b>还没达到：每张卡要 ${exGB(me.r.mem)}，超过了 80 GB。</b></p><p class="small">提示：流水线的段数和张量并行的份数乘起来越大，每张卡存的模型状态越少。micro-batch 切得越多，每份越小，激活也越少。张量并行切得越细，激活也越少。在放得下的排法里，哪种花在 all-reduce 上的时间最少？</p>`)+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the fastest layout','直接看最快的排法')}</button>`);
  if(me.fits) chReveal=true;
  if(!chReveal) return h;
  return h+chSumHTML(me,best)+chAllHTML();
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Predict first, then see why a pipeline leaves GPUs idle, how micro-batches and 1F1B shrink the cost, how tensor parallelism cuts inside a layer and why it stays in one machine, and how real runs stack the three. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看流水线为什么让卡空等、micro-batch 和 1F1B 怎么减小代价，张量并行怎么在层里面切、为什么只在一台机器里用，最后看真实的训练怎么把三种叠起来。第一次学从这里开始。`),
  sandbox:tx('Four real situations, from 70B on one machine to 405B on 16,384 GPUs. Pick one to see whether it fits in memory, how big the bubble is and how much time goes to communication.',
             '四个真实情况，从一台机器训练 70B，到 16,384 张卡训练 405B。选一个，看显存放不放得下、bubble 有多大、通信占了多少时间。'),
  challenge:tx('Train Llama‑3.1‑70B on 4 machines (32 H100s), 256 sequences per step. Choose pipeline stages, tensor parallelism and micro-batches so that every GPU fits in 80 GB and a step takes within 1% of the fastest layout.',
               '在 4 台机器（32 张 H100）上训练 Llama‑3.1‑70B，每步 256 条序列。选好流水线段数、张量并行和 micro-batch 数，让每张卡都放得下 80 GB，每步耗时和最快的排法相差不到 1%。')
});
const SRC_DT={titan:'https://github.com/pytorch/torchtitan/blob/main/torchtitan/config/configs.py',meg:'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py'};
const FIDPOP=()=>{const emb=Math.round(embShare(MD['70b'])*100), lg=fGB(SEQ*MD['70b'].V*4);
 return tx(
 `<p><b>Formula:</b> every number comes from a formula or a small simulation of the schedule; ${J('math','“Show the math”')} works them out.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Compute: 6 FLOPs per parameter per token (forward 2, backward 4) for the layers’ parameters. While computing, a GPU reaches ${SH()} of the H100’s 989.5 TFLOP/s dense BF16. ${TIP.share()}</li>`+
 `<li>Pipeline: GPipe and 1F1B are simulated stage by stage, 1F1B in Megatron-LM’s order (GPU i starts with p − i − 1 forwards). Both give the bubble ${A.ui.math('\\frac{p - 1}{m + p - 1}')}, as the zero-bubble README writes it.</li>`+
 `<li>Tensor parallelism: Megatron-LM’s split, column then row, 2 all-reduces in the forward and 2 in the backward per layer, each of 2 bytes × tokens × hidden size; ring all-reduce time from nccl-tests. Across machines every GPU of a machine runs its own tensor-parallel group, so each gets 50 GB/s of the machine’s 400 GB/s.</li>`+
 `<li>Data parallelism: the gradients are all-reduced in BF16, 2 bytes per value, as the ZeRO paper assumes. Inside one machine a ring over NVLink; across machines the scaling book’s rule: the stage’s whole gradient goes twice through each machine’s 400 GB/s.</li>`+
 `<li>Memory: 18 bytes per parameter of model states (${A.ui.math('6 + \\frac{12}{d}')} with ZeRO-1), and Megatron-LM’s per-layer activation formula, ${A.ui.math('\\frac{s \\cdot b \\cdot h \\cdot (18 + 4 \\cdot \\frac{I}{h})}{t}')} bytes. That formula assumes sequence parallelism and selective recomputation (attention’s intermediate results are not kept but computed again in the backward pass).</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>No overlap: every all-reduce waits on the critical path, and the data-parallel gradients are all-reduced after the step. Real frameworks overlap much of both.</li>`+
 `<li>${TIP.mfu()}</li>`+
 `<li>torchtitan and Megatron-LM reduce gradients in FP32 by default, which doubles the gradient bytes to move (${cite(SRC_DT.titan,'torchtitan configs.py')}, ${cite(SRC_DT.meg,'Megatron-LM arguments.py')}).</li>`+
 `<li>Bandwidths are nominal (450 GB/s per GPU inside a machine, 400 GB/s per machine between machines); measured all-reduces reach less. Latency, the stage-to-stage sends of the pipeline (one 67 MB tensor per micro-batch per boundary) and the attention’s s² FLOPs are left out.</li>`+
 `<li>The embedding table and the LM head (together about ${emb}% of 70B’s parameters) count in memory, spread evenly, but not in compute. Layers are spread evenly even when they do not divide (405B’s 126 layers on 16 stages).</li>`+
 `<li>The activations leave out the logits. At the default setting, one 4,096-token micro-batch, they take about ${lg} in FP32 on the last stage’s GPU (split t ways with tensor parallelism).</li>`+
 `<li>The chart draws the backward pass exactly twice as long as the forward pass.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Bubble: ${cite(SRC.zb,'sail-sg zero-bubble README')}; 1F1B order: ${cite(SRC.megSched,'Megatron-LM schedules.py')}; column and row splits: ${cite(SRC.megLayers,'Megatron-LM tensor_parallel/layers.py')}; activations: ${cite(SRC.megMem,'theoretical_memory_usage.py')}.</li>`+
 `<li>All-reduce cost: ${cite(SRC.nccl,'nccl-tests PERFORMANCE.md')}. NVLink 450 GB/s, InfiniBand 400 GB/s per machine, collectives across machines, Llama 3’s layout: ${cite(SRC.book,'scaling book, gpus.md')}. Layouts and MFU measurements: ${cite(SRC.titan,'torchtitan benchmarks')}, ${cite(SRC.megReadme,'Megatron-LM README')}. A plain BF16 matrix multiply at 80% of the peak: ${cite(SRC.mamf,'ml-engineering, accelerator README')}.</li>`+
 `<li>Model shapes: config.json of each model on Hugging Face.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>所有数字都由公式或对排法的小模拟算出；${J('math','“展开公式”')}里有算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>计算：层的参数每个 token 算 6 个 FLOP（前向 2，反向 4）。GPU 算的时候能跑到 H100 稠密 BF16 峰值 989.5 TFLOP/s 的 ${SH()}。${TIP.share()}</li>`+
 `<li>流水线：GPipe 和 1F1B 都逐段模拟，1F1B 按 Megatron-LM 的顺序排（第 i 张卡先做 p − i − 1 个前向）。两种的 bubble 都是 ${A.ui.math('\\frac{p - 1}{m + p - 1}')}，和 zero-bubble 的 README 写法一样。</li>`+
 `<li>张量并行：Megatron-LM 的切法，先按列再按行，每层前向 2 次、反向 2 次 all-reduce，每次 2 字节 × token 数 × 隐藏维度。环形 all-reduce 的时间按 nccl-tests 的公式算。跨机器时，每台机器的 8 张卡各在做自己那一组的 all-reduce，所以每张卡分到这台机器 400 GB/s 里的 50 GB/s。</li>`+
 `<li>数据并行：梯度用 BF16 做 all-reduce，每个梯度占 2 字节，和 ZeRO 论文的假设一样。在一台机器里按 NVLink 上的环形算法算。跨机器时按 scaling book 的算法，这一段的整块梯度要传两遍，走每台机器 400 GB/s 的网络。</li>`+
 `<li>显存：模型状态每个参数 18 字节（ZeRO-1 时 ${A.ui.math('6 + \\frac{12}{d}')}）。激活用 Megatron-LM 的每层公式，${A.ui.math('\\frac{s \\cdot b \\cdot h \\cdot (18 + 4 \\cdot \\frac{I}{h})}{t}')} 字节。这个公式假设了序列并行和选择性重算，attention 的中间结果不存，反向时再算。</li></ul></div>`+
 `<div><div class="eyebrow">简化</div><ul>`+
 `<li>不做重叠：每次 all-reduce 都在关键路径上等，数据并行的梯度在一步结束后再做 all-reduce。真实的框架会把这两种通信大部分藏到计算后面。</li>`+
 `<li>${TIP.mfu()}</li>`+
 `<li>torchtitan 和 Megatron-LM 默认用 FP32 做梯度的归约，梯度要传的字节数会翻倍（${cite(SRC_DT.titan,'torchtitan 的 configs.py')}、${cite(SRC_DT.meg,'Megatron-LM 的 arguments.py')}）。</li>`+
 `<li>带宽用标称值（机器里每张卡 450 GB/s，机器之间每台机器 400 GB/s），实测的 all-reduce 达不到。没有算延迟、流水线段之间的传输（每个 micro-batch 在每个边界传一份 67 MB 的张量）和 attention 随长度平方增长的计算。</li>`+
 `<li>embedding 表和 LM head（加起来约占 70B 参数的 ${emb}%）只算进显存，平均分到每张卡上，不算进计算。层数除不尽时也按平均分（405B 的 126 层分 16 段）。</li>`+
 `<li>激活里没有算 logits。默认设置下，一个 4,096 个 token 的 micro-batch，logits 用 FP32 存，在最后一段的卡上约占 ${lg}（张量并行时再分成 t 份）。</li>`+
 `<li>图里的反向一律画成前向的 2 倍长。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>bubble 的公式出自 ${cite(SRC.zb,'sail-sg zero-bubble 的 README')}。1F1B 的顺序出自 ${cite(SRC.megSched,'Megatron-LM 的 schedules.py')}。按列、按行切出自 ${cite(SRC.megLayers,'Megatron-LM 的 tensor_parallel/layers.py')}。激活的公式出自 ${cite(SRC.megMem,'theoretical_memory_usage.py')}。</li>`+
 `<li>all-reduce 的耗时：${cite(SRC.nccl,'nccl-tests 的 PERFORMANCE.md')}。NVLink 450 GB/s、每台机器 InfiniBand 400 GB/s、跨机器的集合通信，以及 Llama 3 的排法：${cite(SRC.book,'scaling book 的 gpus.md')}。真实的排法和 MFU 的实测值出自 ${cite(SRC.titan,'torchtitan 的测试')}和 ${cite(SRC.megReadme,'Megatron-LM 的 README')}。单纯的 BF16 矩阵乘法能到峰值的 80%：${cite(SRC.mamf,'ml-engineering 的 accelerator README')}。</li>`+
 `<li>模型结构：各模型在 Hugging Face 上的 config.json。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);};
const PRIPOP=()=>tx(`<p>A training step runs the forward pass, the backward pass and then the optimizer update. With FSDP or ZeRO-3 (${labLink('parallelism')}) every GPU still computes whole layers on its own share of the data; only the storage of the model states is spread out. This lab cuts the computation itself.</p><p>A layer of Llama‑3.1‑70B holds an attention block and an MLP block (${labLink('architecture')}); its 64 query heads (${labLink('mha')}) are what tensor parallelism shares out.</p>`,
  `<p>训练一步要做前向、反向，再做一次优化器更新。用 FSDP 或 ZeRO-3 时（见${labLinkZ('parallelism')}），每张卡算的仍然是完整的层，只是模型状态分开存。这个 Lab 要切开的是计算本身。</p><p>Llama‑3.1‑70B 的每一层有一个 attention 块和一个 MLP 块（见${labLink('architecture')}）；张量并行分给各张卡的，是它的 64 个 query head（见${labLink('mha')}）。</p>`);
const OUTCOME=()=>tx('work out a pipeline’s bubble, explain why tensor parallelism stays inside one machine, and lay out a model too big for one GPU.','根据段数和 micro-batch 数算出流水线的 bubble，讲清张量并行为什么只在一台机器里用，并为一个单卡放不下的模型安排流水线、张量和数据并行。');
const EXITS=()=>{
  const b832=sim(8,32,'gpipe').idle, b1664=(16-1)/(64+16-1);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does the bubble shrink with more micro-batches but never reach 0?','为什么 micro-batch 越多，bubble 越小，却永远降不到 0？'),
   o:[tx('Each GPU idles p − 1 slots in total, whatever m is. More micro-batches keep each GPU busy for longer, so the idle share falls, but it never reaches 0.','每张卡空等的时间加起来总是 p − 1 格，和 m 无关。micro-batch 越多，每张卡忙的时间越长，空等的占比就越小，但不会变成 0。'),
      tx('Each extra micro-batch removes one slot of waiting, so the bubble falls step by step until the micro-batches outnumber the stages, and after that it is gone.','每多一个 micro-batch 就少一格空等，所以 bubble 一格一格变小，等 micro-batch 比段数多以后就没有了。'),
      tx('More micro-batches mean more sends between GPUs, and the time spent sending cancels what the extra micro-batches save.','micro-batch 越多，卡之间传的次数越多。传数据多花的时间，抵掉了多切几份省下的时间。')],a:0,
   n:[null,tx('Each GPU’s idle time stays p − 1 slots; more micro-batches do not remove any of it. The extra micro-batches only make the busy time longer, so the idle share shrinks.','每张卡空等的时间始终是 p − 1 格，多切几个 micro-batch 也省不掉。多切的 micro-batch 只让忙的时间变长，所以空等的占比变小了。'),
     tx('This lab counts no time for sends between stages, and still the bubble stays: '+A.ui.math('\\frac{p - 1}{m + p - 1}')+' has no communication in it.','这个 Lab 没有算段之间传数据的时间，bubble 照样降不到 0：'+A.ui.math('\\frac{p - 1}{m + p - 1}')+' 里根本没有通信。')],
   /* S9i: the formula on its own line (the kit puts why inside a <p>: close it around the math block) */
   why:tx('The bubble is:','bubble 的公式是：')+'</p>'+M('\\text{bubble} = \\frac{p - 1}{m + p - 1}')+'<p>'+tx('p is the number of pipeline stages and m the number of micro-batches. The top is the number of slots each GPU always idles; the bottom grows with m.','p 是流水线段数，m 是 micro-batch 数。分子是每张卡固定要空等的格数，分母随 m 变大。')},
  {q:tx('Why is tensor parallelism kept inside one machine?','为什么张量并行只在一台机器里用？'),
   o:[tx('It needs all-reduces in every layer, and the next piece of work waits for each one; between machines each GPU has only 50 GB/s each way, a ninth of NVLink, so the waiting outgrows the compute.','它每一层都要做 all-reduce，后面的计算要等它做完。机器之间每张卡每个方向只有 50 GB/s，是 NVLink 的九分之一，所以等通信的时间超过了计算。'),
      tx('Across machines each all-reduce has to carry more data, because the partial results must be copied to every machine.','跨机器时，每次 all-reduce 要传的数据更多，因为部分结果要复制到每台机器上。'),
      tx('Tensor parallelism needs the GPUs to share one memory, and only GPUs in the same machine can do that.','张量并行要求几张卡共用一块显存，只有同一台机器里的卡才行。')],a:0,
   n:[null,tx('The data is the same, 67 MB per all-reduce for one 4,096-token micro-batch of 70B; what changes is the bandwidth it goes over.','传的数据一样多，70B 一个 4,096 个 token 的 micro-batch 每次 67 MB；变的是走哪条链路、带宽多大。'),
     tx('Each GPU keeps its own '+A.ui.math('\\frac{1}{t}')+' of the parameters in its own memory; the GPUs exchange results with all-reduces and share no memory.','每张卡在自己的显存里存自己那 '+A.ui.math('\\frac{1}{t}')+' 的参数；卡之间靠 all-reduce 交换结果，并不共用显存。')],
   why:tx('Tensor parallelism trades memory and compute per GPU for communication in every layer, on the critical path. That trade only pays off over the fastest link.','张量并行用“每一层都要通信、而且要等”换来每张卡更少的显存和计算。只有在最快的链路上，这笔交换才划算。')}],
 transfer:[
  {q:tx(`p = 8 pipeline stages with m = 32 micro-batches leave a bubble of ${pct(b832)}. A larger model needs p = 16 stages, and each step is cut into m = 64 micro-batches. About how large is the bubble now?`,`流水线段数 p = 8、micro-batch 数 m = 32 时，bubble 是 ${pct(b832)}。一个更大的模型要 p = 16 段，每步切成 m = 64 个 micro-batch。这时 bubble 大约是多少？`),
   o:[tx('About 10%','约 10%'),tx('About 20%','约 20%'),tx('About 40%','约 40%')],a:1,
   why:tx(`Both the stages and the micro-batches doubled, so each GPU’s idle slots and busy slots both about doubled, and the share stays about the same: 15 ÷ (64 + 15) ≈ ${pct(b1664)}.`,`段数和 micro-batch 数都翻了一倍，每张卡空等的格数和干活的格数也都差不多翻了一倍，占比几乎不变：15 ÷ (64 + 15) ≈ ${pct(b1664)}。`)},
  {q:tx('With p = 8 pipeline stages and m = 16 micro-batches, GPU 1 holds 16 micro-batches of activations under GPipe and 8 under 1F1B. With p = 16 stages and m = 64 micro-batches, how many times less does 1F1B keep on GPU 1 than GPipe?','流水线段数 p = 8、micro-batch 数 m = 16 时，GPipe 让第 1 张卡同时存 16 份激活，1F1B 只存 8 份。换成 p = 16 段、m = 64 个 micro-batch，1F1B 存的是 GPipe 的几分之一？'),
   o:[tx('Half, as before','和原来一样，一半'),tx('A quarter','四分之一'),tx('An eighth','八分之一')],a:1,
   why:tx('GPipe keeps all m micro-batches, 64; 1F1B keeps at most p, 16. So 1F1B keeps a quarter.','GPipe 存全部 m 份，也就是 64 份；1F1B 最多存 p 份，也就是 16 份。所以是四分之一。')}],
 practice:[
  {id:'matmul-roofline',d:tx('Parts 3 and 4: pipeline against tensor parallelism for one input on two devices, then sharding a feed-forward network by columns and rows.','第 3、4 部分：两台设备处理同一个输入时，流水线并行和张量并行比一比；再把一个前馈网络按列、按行切开。')},
  {id:'distributed-training',d:tx('A system design: train a model that does not fit on one accelerator; place pipeline, tensor and data parallelism on the right links and account for the bubble.','系统设计：训练一个单卡放不下的模型；把流水线、张量和数据并行放到合适的链路上，并算上 bubble。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Pipeline & tensor','流水线与张量并行'),title:l?L(l.q):tx('How do pipeline and tensor parallelism cut one model apart?','流水线并行和张量并行怎么切开一个模型？'),
    minutes:15,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: what one GPU computes in training','前置知识：训练时每张卡算什么'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage mp-stage" id="mp-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="mp-st-title"></div><div class="meter-sub" id="mp-st-sub"></div></div><div class="stage-actions"><span class="status" id="mp-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph mp-graph" id="mp-graph">`+
    `<div class="mp-phone" id="mp-phone"></div>`+
    `<div class="mp-clbox" id="mp-clbox" hidden><div class="mp-cap" id="mp-cl-cap"></div><div class="mp-cl" id="mp-cl"></div><div class="mp-mem" id="mp-mem"></div></div>`+
    `<div class="mp-gbox" id="mp-gbox"><div class="mp-cap" id="mp-g-cap"></div><div class="mp-gleg" id="mp-gleg"></div><div class="mp-gantt" id="mp-gantt"></div><div class="mp-player" id="mp-player"></div></div>`+
    `<div class="mp-lbox" id="mp-lbox" hidden><div class="mp-cap" id="mp-l-cap"></div><div class="mp-tp" id="mp-tp"></div><div class="mp-cap" id="mp-lt-cap"></div><div class="mp-lt" id="mp-lt"></div><div class="mp-ltleg" id="mp-ltleg"></div></div>`+
   `</div>`+
   `<div class="metrics mp-metrics" id="mp-metrics"></div>`+
   `<details class="math" id="mp-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="mp-mathbody"></div></details></div>`;
}
const NOTE={
 share:()=>TIP.share(),
 bw:()=>tx('Nominal values: NVLink 450 GB/s each way per GPU inside a machine. Between machines the 8 GPUs of a machine share 400 GB/s of InfiniBand (one 400 Gb/s port per GPU, 50 GB/s each way); a data-parallel all-reduce across machines uses the machine’s whole 400 GB/s. They come from the scaling book’s description of the reference H100 cluster; measured all-reduces reach less, so real communication takes longer than shown.',
           '这是标称值：机器里每张卡 NVLink 每个方向 450 GB/s。机器之间，一台机器的 8 张卡共用 400 GB/s 的 InfiniBand（每张卡一个 400 Gb/s 网口，每个方向 50 GB/s），跨机器的数据并行 all-reduce 按整台机器的 400 GB/s 算。这些数来自 scaling book 对参考 H100 集群的描述。实测的 all-reduce 达不到，所以真实的通信比这里算的更久。')
};
const segRow=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">${A.ui.seg({id,key,label,options:opts,value:String(val),note})}</div>`;
function controlsHTML(){
  return `<aside class="panel controls" id="mp-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="mp-lock" hidden></p>`+
   segRow('mp-ctl-model','model',tx('Model','模型'),[['8b','8B'],['70b','70B'],['405b','405B']],S.model)+
   segRow('mp-ctl-p','p',tx('Pipeline stages p','流水线段数 p'),[1,2,4,8,16].map(v=>[String(v),String(v)]),S.p)+
   segRow('mp-ctl-t','t',tx('Tensor parallel t: GPUs per layer','张量并行 t：一层切到几张卡'),TS.map(v=>[String(v),String(v)]),S.t)+
   segRow('mp-ctl-d','d',tx('Data-parallel copies d','数据并行份数 d'),[1,2,4,8,16,32,64,128].map(v=>[String(v),String(v)]),S.d)+
   segRow('mp-ctl-m','m',tx('Micro-batches per pipeline m','每条流水线的 micro-batch 数 m'),[1,2,4,8,16,32,64,128,256].map(v=>[String(v),String(v)]),S.m,tx('Each micro-batch is one sequence here.','这里每个 micro-batch 是一条序列。'))+
   segRow('mp-ctl-seq','seq',tx('Tokens per sequence','每条序列的 token 数'),[['4096','4K'],['32768','32K'],['131072','128K']],S.seq)+
   segRow('mp-ctl-sched','sched',tx('Schedule','排法'),[['gpipe','GPipe'],['1f1b','1F1B']],S.sched)+
   segRow('mp-ctl-place','place',tx('Where a tensor-parallel group sits','张量并行的一组卡放在哪'),[['in',tx('one machine','一台机器里')],['out',tx('one per machine','每台机器 1 张')]],S.place)+
   segRow('mp-ctl-zero','zero',tx('Split the optimizer state across data-parallel copies (ZeRO-1)','优化器状态按数据并行份数分开存（ZeRO-1）'),[['0',tx('no','不分')],['1',tx('yes','分')]],S.zero)+
   `<div class="mp-grp lockable" id="mp-ctl-as"><div class="mp-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><div class="lbl">${tx('Share of the peak while computing','算的时候能到峰值的多少')} <b class="mono">${SH()}</b></div><p class="ctl-note">${esc(NOTE.share())}</p></div>`+
    `<div class="ctl"><div class="lbl">${tx('Link bandwidth per GPU','每张卡的链路带宽')} <b class="mono">450 / 50 GB/s</b></div><p class="ctl-note">${esc(NOTE.bw())}</p></div></div>`+
   `<div class="lockable" id="mp-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`def schedule(p, m):                 # Megatron-style 1F1B
    todo = []
    for i in range(p):
        w = min(p - i - 1, m)           # warm-up forwards
        o = [('F', j) for j in range(w)]
        for k in range(m - w):
            o += [('F', w + k), ('B', k)]
        todo.append(o + [('B', j) for j in range(m - w, m)])
    end, free, rows = {}, [0] * p, [[] for _ in range(p)]
    while any(todo):
        for i in range(p):
            while todo[i]:
                k, j = todo[i][0]
                dep = ((k, i - 1, j) if k == 'F' and i > 0 else
                       ('F', i, j) if k == 'B' and i == p - 1 else
                       ('B', i + 1, j) if k == 'B' else None)
                if dep and dep not in end: break
                s = max(free[i], end.get(dep, 0))
                e = s + (1 if k == 'F' else 2)    # B = 2 F
                end[(k, i, j)] = free[i] = e
                rows[i].append((s, e, k, j))
                todo[i].pop(0)
    for i, r in enumerate(rows):
        line = ['.'] * max(free)
        for s, e, k, j in r:
            line[s:e] = [k.lower() if k == 'B' else k] * (e - s)
        print(f'GPU {i + 1}', ''.join(line))
    print('bubble', round((p - 1) / (m + p - 1), 3))
schedule(4, 8)`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="mp-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="mp-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A 1F1B schedule in plain Python: it places every forward (F) and backward (b) of p stages and m micro-batches, prints one row per GPU (a dot is idle time) and the bubble.','用纯 Python 排一个 1F1B：把 p 段、m 个 micro-batch 的每个前向（F）和反向（b）排上时间线，每张卡打印一行（点是空等），最后打印 bubble。')}</p><pre class="code" id="mp-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="mp-prod-card"><h3>${A.t('cProd')}</h3><ul class="mp-prod">`+
   tx(`<li>Megatron-LM sets each cut with its own size: <code>pipeline_model_parallel_size</code>, <code>virtual_pipeline_model_parallel_size</code> (several chunks of layers per GPU, the interleaved schedule), <code>tensor_model_parallel_size</code>, <code>context_parallel_size</code> and <code>expert_model_parallel_size</code> (${cite(SRC.bridge,'Megatron-Bridge, parallelisms.md')}). Its guide says to keep tensor parallelism within a node, up to 8, and use pipeline parallelism across nodes.</li>`+
      `<li>The 1F1B order of this lab is Megatron-LM’s: in ${cite(SRC.megSched,'<code>schedules.py</code>')}, <code>num_warmup_microbatches</code> is the number of stages minus the GPU’s stage minus 1. The column and row splits are <code>ColumnParallelLinear</code> and <code>RowParallelLinear</code> in ${cite(SRC.megLayers,'<code>tensor_parallel/layers.py</code>')}.</li>`+
      `<li>PyTorch ships the schedules in <code>torch.distributed.pipelining</code>: <code>ScheduleGPipe</code>, <code>Schedule1F1B</code>, <code>ScheduleInterleaved1F1B</code> and zero-bubble variants (${cite(SRC.pipe,'distributed.pipelining.md')}). torchtitan combines them with FSDP, tensor and context parallelism and publishes measured layouts for Llama 3.1 8B, 70B and 405B (${cite(SRC.titan,'benchmarks')}).</li>`,
      `<li>Megatron-LM 每种切法各有一个参数：<code>pipeline_model_parallel_size</code>、<code>virtual_pipeline_model_parallel_size</code>（每张卡放几小段层，也就是交错式排法）、<code>tensor_model_parallel_size</code>、<code>context_parallel_size</code> 和 <code>expert_model_parallel_size</code>（${cite(SRC.bridge,'Megatron-Bridge 的 parallelisms.md')}）。它的指南说：张量并行留在一台机器里，最多 8 份；机器之间用流水线并行。</li>`+
      `<li>这个 Lab 的 1F1B 顺序就是 Megatron-LM 的：${cite(SRC.megSched,'<code>schedules.py</code>')} 里，<code>num_warmup_microbatches</code> 等于段数减去这张卡所在的段，再减 1。按列切和按行切是 ${cite(SRC.megLayers,'<code>tensor_parallel/layers.py</code>')} 里的 <code>ColumnParallelLinear</code> 和 <code>RowParallelLinear</code>。</li>`+
      `<li>PyTorch 在 <code>torch.distributed.pipelining</code> 里提供这些排法：<code>ScheduleGPipe</code>、<code>Schedule1F1B</code>、<code>ScheduleInterleaved1F1B</code>，还有 zero-bubble 的变体（${cite(SRC.pipe,'distributed.pipelining.md')}）。torchtitan 把它们和 FSDP、张量并行、上下文并行组合起来，并公开了 Llama 3.1 8B、70B、405B 的实测排法（${cite(SRC.titan,'benchmarks')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.p1909,'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism')}<span>${tx('Shoeybi et al., 2019 (arXiv 1909.08053): the column-then-row split of tensor parallelism.','Shoeybi 等，2019（arXiv 1909.08053）：张量并行先按列、再按行的切法。')}</span></li>`+
   `<li>${cite(SRC.p2104,'Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM')}<span>${tx('Narayanan et al., 2021 (arXiv 2104.04473): combining pipeline, tensor and data parallelism; the interleaved 1F1B schedule.','Narayanan 等，2021（arXiv 2104.04473）：把流水线、张量和数据并行组合起来；交错式 1F1B 排法。')}</span></li>`+
   `<li>${cite(SRC.p1811,'GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism')}<span>${tx('Huang et al., 2019 (arXiv 1811.06965): micro-batches in a pipeline.','Huang 等，2019（arXiv 1811.06965）：在流水线里用 micro-batch。')}</span></li>`+
   `<li>${cite(SRC.p2205,'Reducing Activation Recomputation in Large Transformer Models')}<span>${tx('Korthikanti et al., 2022 (arXiv 2205.05198): sequence parallelism and the activation formula used here.','Korthikanti 等，2022（arXiv 2205.05198）：序列并行，以及这里用的激活公式。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="mp-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="mp-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="mp-app m2" id="mp-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="mp-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see whether each GPU has enough memory, then where a step’s time goes.','选一个场景，看看每张卡的显存够不够，再看一步的时间花在了哪里。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="mp-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const d=CH.G/(S.t*S.p), okM=m=>!!chCfg(S.t,S.p,m);
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Lay out 70B on 4 machines','把 70B 排到 4 台机器上')}</h3>`+
   `<p>${tx('You train Llama‑3.1‑70B on 4 machines with 8 H100s each, 32 GPUs. Every step trains on 256 sequences of 4,096 tokens. Choose how many pipeline stages, how many GPUs share each layer (tensor parallelism), and how many micro-batches each pipeline cuts its share into; the GPUs left over make data-parallel copies. Every GPU must fit in 80 GB, and a step must take within 1% of the fastest layout that fits.',
            '你要在 4 台机器上训练 Llama‑3.1‑70B，每台 8 张 H100，一共 32 张卡。每步训练 256 条序列，每条 4,096 个 token。请选：流水线分几段、一层切到几张卡上（张量并行）、每条流水线把自己那份数据切成几个 micro-batch。剩下的卡做数据并行。每张卡都要放得下 80 GB，每步耗时和放得下的排法里最快的那种相差不到 1%。')}</p>`+
   `<ul class="mp-asm small"><li>${tx('The data-parallel copies split the optimizer state (ZeRO-1); the schedule is 1F1B.','各份数据并行把优化器状态分开存（ZeRO-1）；排法用 1F1B。')}</li><li>${tx('A tensor-parallel group stays inside one machine.','张量并行的一组卡总在同一台机器里。')}</li></ul>`+
   A.ui.seg({id:'mp-ic-chp',key:'chp',label:tx('Pipeline stages p','流水线段数 p'),options:CH.Ps.map(v=>[String(v),String(v),!Number.isInteger(CH.G/(S.t*v))]),value:String(S.p),inline:true})+
   A.ui.seg({id:'mp-ic-cht',key:'cht',label:tx('Tensor parallel t: GPUs per layer','张量并行 t：一层切到几张卡'),options:CH.Ts.map(v=>[String(v),String(v),!Number.isInteger(CH.G/(v*S.p))]),value:String(S.t),inline:true})+
   A.ui.seg({id:'mp-ic-chm',key:'chm',label:tx('Micro-batches per pipeline m','每条流水线的 micro-batch 数 m'),options:CH.Ms.map(v=>[String(v),String(v),!okM(v)]),value:String(S.m),inline:true,note:tx(`With d = ${d} data-parallel ${d===1?'copy':'copies'}, each pipeline gets ${CH.B/d} sequences per step.`,`数据并行 d = ${d} 份，每条流水线每步分到 ${CH.B/d} 条序列。`)})+
   `<div class="goal mono mp-goal" id="mp-ch-g1"><span>${tx('Memory per GPU','每张卡的显存')}</span><span><b id="mp-ch-mem"></b>${tx(' (goal: at most 80 GB)','（目标：不超过 80 GB）')}</span></div>`+
   `<div class="goal mono mp-goal" id="mp-ch-g2"><span>${tx('Time per step','每步耗时')}</span><span><b id="mp-ch-step"></b>${tx(' (goal: within 1% of the fastest)','（目标：和最快的相差不到 1%）')}</span></div>`+
   submit+`<div id="mp-ch-res" class="mp-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},gpipe:{en:'Set the number of micro-batches m to 32 or more to continue.',zh:'把 micro-batch 数 m 调到 32 或更多，才能继续。'},
    '1f1b':{en:'Pick “1F1B” to continue.',zh:'选“1F1B”才能继续。'},tp:{en:'Split the layer across 8 GPUs to continue.',zh:'把这一层切到 8 张卡上，才能继续。'},link:{en:'Pick “One GPU in each of 8 machines” to continue.',zh:'选“分在 8 台机器上”才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#mp-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function obsShown(s){
  if(!s.obs) return false;
  if(s.id==='gpipe'||s.id==='1f1b') return runDone();
  if(s.id==='tp') return S.t!==1||!!observed[step];
  return true;
}
function updateRail(){
  const rail=$('#mp-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    $$('#mp-ic-lv3 [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv3)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tried3.has(v)||v===lv3?lv3Res(v):'';});
    $$('#mp-ic-lv5 [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv5)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tried5.has(v)||v===lv5?lv5Res(v):'';});
    if(box){const html=obsShown(s)?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#mp-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.t,S.p,S.m), mm=$('#mp-ch-mem',rail), st=$('#mp-ch-step',rail); if(!mm||!e) return;
    mm.textContent=fGB(e.r.mem); st.textContent=fS(e.step);
    const g1=$('#mp-ch-g1',rail); g1.classList.toggle('met',e.fits); g1.classList.toggle('mp-over',!e.fits);
    const res=$('#mp-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage: views
   ===================================================================== */
const viewNow=()=>mode==='guided'?STEPS[step].view:'cluster';
const predHidden=()=>isP('mission');
const redIdle=()=>mode==='guided'&&(isP('gpipe')||isP('1f1b'));
const simNow=()=>sim(S.p,S.m,S.sched);
const simKey=()=>`${S.sched}|${S.p}|${S.m}`;
/* the machine that holds pipeline dp's stage pp (both from 0), numbered from 1 as under the cluster's cells */
const machOf=(dp,pp)=>Math.floor(S.t*(dp+S.d*pp)/GPU.perNode)+1;
function ganttCap(){
  if(predHidden()) return tx('8 GPUs, 10 layers each: one row per GPU, time runs to the right','8 张卡，各放 10 层：每行是一张卡，往右是时间');
  const s=simNow();
  const one=mode!=='guided'||S.d>1, per=S.t>1?tx(`${S.t} GPUs per stage`,`每段 ${S.t} 张卡`):'';
  return tx(`One training step: p = ${S.p} stages, m = ${S.m} ${S.m===1?'micro-batch':'micro-batches'}, ${S.sched==='gpipe'?'GPipe':'1F1B'}`,`一步训练：流水线段数 p = ${S.p}，micro-batch 数 m = ${S.m}，${S.sched==='gpipe'?'GPipe':'1F1B'}`)+
    (one||per?tx(` (${[one?'one pipeline':'',per].filter(Boolean).join('; ')})`,`（${[one?'其中一条流水线':'',per].filter(Boolean).join('；')}）`):'');
}
function legendHTML(g,mix){
  const one=g===1;
  return `<span tabindex="0" data-tipx="${esc(one?tx('An outlined cell is one micro-batch’s forward pass on that GPU; the number says which micro-batch.','空心块是某个 micro-batch 在这张卡上的前向，块里的数字是第几个 micro-batch。'):tx(`An outlined block is the forward passes of ${g} micro-batches in a row on that GPU; point at a block to see which ones.`,`空心块是这张卡连着做的 ${g} 个 micro-batch 的前向，指到块上能看到是哪几个。`))}"><i class="sw mp-sw-f"></i>${esc(tx('outlined: forward','空心块：前向'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(one?tx('A solid cell is one micro-batch’s backward pass; it is drawn twice as long as the forward pass.','实心块是某个 micro-batch 的反向，画成前向的 2 倍长。'):tx(`A solid block is the backward passes of ${g} micro-batches in a row; a backward is drawn twice as long as a forward.`,`实心块是连着做的 ${g} 个 micro-batch 的反向，一个反向画成前向的 2 倍长。`))}"><i class="sw mp-sw-b"></i>${esc(tx('solid: backward','实心块：反向'))}</span>`+
    (mix?`<span tabindex="0" data-tipx="${esc(tx(`In 1F1B a GPU does one forward, then one backward, in turn. A block half outlined and half solid holds ${g} forwards and ${g} backwards done that way.`,`1F1B 里，一张卡做一个前向、再做一个反向，轮流来。上半空心、下半实心的块，就是这样轮流做的 ${g} 个前向和 ${g} 个反向。`))}"><i class="sw mp-sw-m"></i>${esc(tx('half and half: forward and backward in turn','半空半实：前向、反向轮流'))}</span>`:'')+
    `<span tabindex="0" data-tipx="${esc(tx('Hatched time: the GPU is idle, waiting for data or gradients. This is the bubble.','斜纹：这张卡空着，在等数据或梯度，这就是 bubble。'))}"><i class="sw mp-sw-i${redIdle()?' red':''}"></i>${esc(tx('hatched: idle (bubble)','斜纹：空等（bubble）'))}</span>`+
    (one?'':`<span class="mp-gn" tabindex="0" data-tipx="${esc(tx(`With m = ${S.m} micro-batches one cell each would be too thin to tell apart, so each block holds ${g} micro-batches in a row. The time axis is unchanged, and the idle time is drawn to scale.`,`micro-batch 数 m = ${S.m}，一个一个画会细得分不开，所以每块画连着的 ${g} 个。时间轴不变，空等的时间照原样画。`))}">${esc(tx(`one block = ${g} micro-batches`,`一块 = ${g} 个 micro-batch`))}</span>`);
}
/* owner b8 (2), (4): blocks must stay apart at every m. u = pixels per time unit (a forward is 1 unit). When a forward would be
   narrower than MINW px, g micro-batches in a row are drawn as one block (g a power of 2). A run of work with no idle time in it is
   cut into pure forward runs, pure backward runs and 1F1B's alternation (one forward, one backward, in turn); each is cut into blocks
   of g micro-batches; a block narrower than MINW joins its neighbour in the same run. Idle time is never hidden inside a block. */
const MINW=6;
const groupOf=(u,m)=>{let g=1; while(g<m&&g*u<MINW) g*=2; return g;};
function blocksOf(ops,g,u){
  const runs=[]; let cur=null;
  ops.forEach(o=>{if(cur&&o.s===cur[cur.length-1].e) cur.push(o); else {cur=[o]; runs.push(cur);}});
  const out=[];
  runs.forEach(run=>{
    if(g===1){run.forEach(o=>out.push([o])); return;}
    const grp=[]; run.forEach(o=>{const l=grp[grp.length-1]; if(l&&l[0].k===o.k) l.push(o); else grp.push([o]);});
    const segs=[]; let i=0;
    while(i<grp.length){if(grp[i].length===1){let j=i; while(j+1<grp.length&&grp[j+1].length===1) j++;
        if(j-i+1>=4){segs.push({mix:true,ops:[].concat(...grp.slice(i,j+1))}); i=j+1; continue;}}
      segs.push({mix:false,ops:grp[i]}); i++;}
    const ch=[]; segs.forEach(sg=>{const n=sg.mix?2*g:g; for(let k=0;k<sg.ops.length;k+=n) ch.push(sg.ops.slice(k,k+n));});
    const w=c=>(c[c.length-1].e-c[0].s)*u;
    for(let k=0;k<ch.length&&ch.length>1;){if(w(ch[k])>=MINW){k++; continue;}
      if(k>0&&(k===ch.length-1||w(ch[k-1])<=w(ch[k+1]))){ch[k-1]=ch[k-1].concat(ch[k]); ch.splice(k,1);} else {ch[k+1]=ch[k].concat(ch[k+1]); ch.splice(k,1);}}
    ch.forEach(c=>out.push(c));});
  return out;
}
const rangeOf=(c,k)=>{const js=c.filter(o=>o.k===k).map(o=>o.j+1); if(!js.length) return ''; const a=Math.min(...js), b=Math.max(...js); return a===b?String(a):`${a}–${b}`;};
function blockTip(c,i){
  const f=rangeOf(c,'F'), b=rangeOf(c,'B'), gpu=S.t>1?tx(`Stage ${i+1}: `,`第 ${i+1} 段：`):tx(`GPU ${i+1}: `,`第 ${i+1} 张卡：`);
  if(f&&b) return gpu+tx(`forwards of micro-batches ${f} and backwards of ${b}, in turn.`,`第 ${f} 个 micro-batch 的前向和第 ${b} 个的反向，轮流做。`);
  return gpu+(f?tx(`the forward passes of micro-batches ${f}.`,`第 ${f} 个 micro-batch 的前向。`):tx(`the backward passes of micro-batches ${b}.`,`第 ${b} 个 micro-batch 的反向。`));
}
/* outlines shrink with the block: 1.5 px from 12 px wide, 1 px from 6 px, none below; busy blocks keep a 1 px gap */
const olw=px=>px>=12?'':(px>=6?' o1':' o0');
function rowHTML(ops,T,u,g,i,red,dense){
  let cells='', last=0;
  const P=x=>(x/T*100).toFixed(3), idle=(a,b)=>`<i class="mp-ci${red?' red':''}${olw((b-a)*u)}" style="left:${P(a)}%;width:${P(b-a)}%"></i>`;
  blocksOf(ops,g,u).forEach(c=>{const s=c[0].s, e=c[c.length-1].e, px=(e-s)*u;
    if(s>last) cells+=idle(last,s);
    const kf=c.every(o=>o.k==='F'), kb=c.every(o=>o.k==='B'), cls=kf?'mp-cf':(kb?'mp-cb':'mp-cm');
    let lab='';
    if(!dense){if(g===1){if(u>=16) lab=String(c[0].j+1);} else if(kf||kb){const r=rangeOf(c,kf?'F':'B'); if(px>=r.length*6.6+8) lab=r;}}
    cells+=`<i class="${cls}${olw(px)}" style="left:${P(s)}%;width:${px>=3?`calc(${P(e-s)}% - 1px)`:`${P(e-s)}%`}"${g>1?` data-tipx="${esc(blockTip(c,i))}"`:''}>${lab}</i>`; last=e;});
  if(last<T) cells+=idle(last,T);
  return cells;
}
function drawGantt(){
  const host=$('#mp-gantt',app), cap=$('#mp-g-cap',app), lg=$('#mp-gleg',app);
  cap.textContent=ganttCap();
  const p=S.p, md=MD[S.model];
  /* the track is the host minus the 96 px label column; u pixels per time unit decide how many micro-batches one block holds */
  const s=predHidden()?null:simNow(), W=Math.max(1,host.clientWidth-96), u=s?W/s.T:1, g=s?groupOf(u,S.m):1;
  const mix=!!s&&g>1&&S.sched==='1f1b'&&s.ops.some(r=>blocksOf(r,g,u).some(c=>!c.every(o=>o.k===c[0].k)));
  const lh=predHidden()?'':legendHTML(g,mix); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const sig=[predHidden(),simKey(),S.t,S.d,redIdle(),A.lang(),host.clientWidth>0?Math.round(host.clientWidth/8):0].join('|');
  if(host._sig!==sig){host._sig=sig;
    const rows=[];
    /* one GPU per stage: "GPU 1"; with tensor parallelism a stage is t GPUs of one machine (A7): "Stage 1, machine 1" when it fits
       the label column, else "Machine 1" (measured below); the caption says how many GPUs a stage has */
    const lab=(i,long)=>{const a=Math.round(i*md.L/p)+1, b=Math.round((i+1)*md.L/p), ly=p<=8?`<small>${tx(`layers ${a}–${b}`,`第 ${a}–${b} 层`)}</small>`:'';
      if(S.t===1) return tx(`GPU ${i+1}`,`第 ${i+1} 张卡`)+ly;
      const mc=machOf(0,i); return `<span class="mp-gl1">${long?tx(`Stage ${i+1}, machine ${mc}`,`第 ${i+1} 段，机器 ${mc}`):tx(`Machine ${mc}`,`机器 ${mc}`)}</span>`+ly;};
    if(predHidden()){for(let i=0;i<p;i++) rows.push(`<div class="mp-grow"><span class="mp-glab">${lab(i)}</span><span class="mp-gtrk"></span></div>`);}
    else for(let i=0;i<p;i++) rows.push(`<div class="mp-grow"><span class="mp-glab">${lab(i,true)}</span><span class="mp-gtrk">${rowHTML(s.ops[i],s.T,u,g,i,redIdle(),p>8)}</span></div>`);
    host.innerHTML=`<div class="mp-grows${p>8?' dense':''}">${rows.join('')}<div class="mp-mask" id="mp-mask"></div><div class="mp-head" id="mp-head"></div></div>`;
    /* the text's own width against the label's content box (an ellipsis would hide the machine) */
    const tooLong=e=>{const w=e.clientWidth-(parseFloat(getComputedStyle(e).paddingRight)||0); return [...e.querySelectorAll('.mp-gl1,small')].some(x=>{const r=document.createRange(); r.selectNodeContents(x); return r.getBoundingClientRect().width>w+0.5;});};
    if(S.t>1&&!predHidden()){const ls=[...host.querySelectorAll('.mp-glab')]; if(ls.some(tooLong)) ls.forEach((e,i)=>{e.innerHTML=lab(i,false);});}
    host.dataset.tipx=predHidden()?tx('Each row is one GPU of the pipeline and the 10 layers it holds.','每一行是流水线上的一张卡，以及它放的 10 层。'):S.t>1?tx(`Each row is one stage of the pipeline: ${S.t} GPUs in one machine split its layers with tensor parallelism. Time runs to the right. The whole step is m + p − 1 = ${S.m+S.p-1} forward-and-backward slots long, of which each stage is busy in m = ${S.m}.`,`每一行是流水线的一段：一台机器里的 ${S.t} 张卡用张量并行分担这一段的层。往右是时间。整步一共 m + p − 1 = ${S.m+S.p-1} 格（每格一次前向加一次反向），每段只有 m = ${S.m} 格在干活。`):tx(`Each row is one GPU; time runs to the right. The whole step is m + p − 1 = ${S.m+S.p-1} forward-and-backward slots long, of which each GPU is busy in m = ${S.m}.`,`每一行是一张卡，往右是时间。整步一共 m + p − 1 = ${S.m+S.p-1} 格（每格一次前向加一次反向），每张卡只有 m = ${S.m} 格在干活。`);
    host.tabIndex=0; host.setAttribute('role','img'); host.setAttribute('aria-label',host.dataset.tipx);}
  drawHead();
  drawPlayer();
}
/* the playhead: a mask hides what has not happened yet */
function drawHead(){
  const mk=$('#mp-mask',app), hd=$('#mp-head',app); if(!mk) return;
  const x=predHidden()?0:ph*100;
  mk.style.left=`calc(var(--mp-lab) + (100% - var(--mp-lab)) * ${(x/100).toFixed(4)})`;
  hd.style.left=mk.style.left; hd.hidden=predHidden()||ph>=1;
}
function drawPlayer(){
  const pl=$('#mp-player',app); if(!pl) return;
  if(predHidden()){pl.hidden=true; return;} pl.hidden=false;
  const s=simNow(), done=ph>=1, sig=[playing,spd,Math.round(ph*1000),A.lang(),simKey()].join('|'); if(pl._sig===sig) return; pl._sig=sig;
  const btnTxt=playing?tx('❚❚ Pause','❚❚ 暂停'):(done?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  const cnt=tx(`slot ${fixd(Math.min(ph,1)*(S.m+S.p-1),0)} of ${S.m+S.p-1}`,`第 ${fixd(Math.min(ph,1)*(S.m+S.p-1),0)} / ${S.m+S.p-1} 格`);
  const scrub=pl.querySelector('input[data-key="scrub"]');
  if(scrub&&pl._built===A.lang()){scrub.value=Math.round(ph*1000); A.ui.setRange(scrub); pl.querySelector('[data-act="play"]').textContent=btnTxt;
    $$('.seg-ctl[data-key="spd"]',pl).forEach(g=>A.ui.setPressed(g,spd)); pl.querySelector('.mp-count').textContent=cnt; return;}
  pl._built=A.lang();
  pl.innerHTML=`<button type="button" class="btn ghost sm mp-play" data-act="play" data-tipx="${esc(tx('Play this step at the chosen speed; press again to pause. At the end, the button plays it again from the start.','按选定的速度播放这一步；再按一次暂停。播完以后，这个按钮从头重放。'))}">${btnTxt}</button>`+
    A.ui.seg({id:'mp-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd})+
    `<div class="mp-scrub"><input type="range" min="0" max="1000" step="1" value="${Math.round(ph*1000)}" data-key="scrub" aria-label="${esc(tx('Time into the step','这一步播到哪里'))}"><span class="mp-count mono">${cnt}</span></div>`;
  A.ui.initRanges(pl);
}
/* parts 4 and 5: the MLP block cut across t GPUs, and one layer's timeline */
function drawLayer(){
  const t=S.t, place=S.place, r=layer(t,place), r1=layer(1,'in');
  $('#mp-l-cap',app).textContent=t===1?tx('The MLP block of one layer on one GPU','一层的 MLP 块，放在一张卡上'):tx(`The MLP block of one layer, cut across ${t} GPUs`,`一层的 MLP 块，切到 ${t} 张卡上`);
  const strips=(n,dir)=>{let s=''; for(let i=0;i<n;i++) s+=`<i class="mp-st"><b>${n>1?i+1:''}</b></i>`; return `<span class="mp-mat ${dir}" style="--n:${n}">${s}</span>`;};
  const gpuTip=t===1?'':tx(` Each numbered slice belongs to one GPU: GPU k keeps slice k of both weight matrices, 1/${t} of the block’s parameters.`,`编号相同的切片属于同一张卡：第 k 张卡存两个权重矩阵的第 k 片，是这个块参数的 1/${t}。`);
  const blk=(cls,label,inner,tip)=>`<span class="mp-blk ${cls}" tabindex="0" data-tipx="${esc(tip)}"><span class="mp-bl">${label}</span>${inner}</span>`;
  const h=blk('x',tx('input','输入'),`<span class="mp-act"></span>`,tx('The input to the MLP block: 4,096 tokens × 8,192 dimensions. Every GPU holds the whole of it.','MLP 块的输入：4,096 个 token × 8,192 维。每张卡都有一整份。'))+
    `<span class="mp-op">×</span>`+blk('w1',tx('gate & up, cut by columns','gate 和 up，按列切'),strips(t,'cols'),tx('The gate and up matrices (8,192 × 28,672 each), cut by columns: each GPU computes its own slice of the 28,672-wide middle from the full input. No communication is needed for this.','gate 和 up 两个矩阵（各 8,192 × 28,672）按列切：每张卡用完整的输入，算出 28,672 维的中间层里自己那一段。这一步不需要通信。')+gpuTip)+
    `<span class="mp-op">→</span>`+blk('w2',tx('down, cut by rows','down，按行切'),strips(t,'rows'),tx('The down matrix (28,672 × 8,192), cut by rows: each GPU multiplies its slice of the middle by its rows and gets a partial result as large as the whole output.','down 矩阵（28,672 × 8,192）按行切：每张卡用自己那段中间结果乘自己那几行，得到一份和整个输出一样大的部分结果。')+gpuTip)+
    `<span class="mp-op">→</span>`+blk('ps',t>1?tx(`${t} partial results`,`${t} 份部分结果`):tx('output','输出'),`<span class="mp-parts" style="--n:${t}">${Array.from({length:t},()=>'<i></i>').join('')}</span>`,t>1?tx(`Each GPU holds one partial result of 4,096 × 8,192 numbers; the true output is their sum.`,`每张卡有一份 4,096 × 8,192 的部分结果；真正的输出是它们的和。`):tx('One GPU computes the whole output itself.','一张卡自己算出整份输出。'))+
    (t>1?`<span class="mp-op">→</span>`+blk('ar'+(place==='out'?' red':''),'all-reduce',`<span class="mp-sum">Σ</span>`,tx(`The all-reduce adds the ${t} partial results and gives every GPU the sum: ${fMB(r.bytes)} per GPU each time, over ${place==='out'?'InfiniBand':'NVLink'}.`,`all-reduce 把 ${t} 份部分结果加起来，再让每张卡都拿到这个和：每次每张卡 ${fMB(r.bytes)}，走 ${place==='out'?'InfiniBand':'NVLink'}。`)):'');
  const tp=$('#mp-tp',app); if(tp._h!==h){tp.innerHTML=h; tp._h=h;}
  /* the timeline of one layer: forward (attention, all-reduce, MLP, all-reduce) then backward, to one scale (one GPU's whole layer) */
  const md=MD['70b'], Pl=layerP(md), fa=(2*md.h*md.h+2*md.h*md.kv*md.hd)/Pl, scale=Math.max(r1.total,r.total);
  const segAt=sc=>(w,cls,tip)=>`<i class="${cls}" style="width:${(w/sc*100).toFixed(3)}%" tabindex="0" data-tipx="${esc(tip)}"></i>`;
  const cf=r.comp/3, cb=r.comp*2/3, arT=r.ar, red=place==='out';
  const ctip=(x,w)=>tx(`${w}: ${fMs(x)} of compute on each GPU.`,`${w}：每张卡计算 ${fMs(x)}。`), atip=tx(`One all-reduce of ${fMB(r.bytes)}: ${fMs(arT)} over ${red?'InfiniBand':'NVLink'}, while the GPU waits.`,`一次 ${fMB(r.bytes)} 的 all-reduce：走 ${red?'InfiniBand':'NVLink'} 要 ${fMs(arT)}，这段时间这张卡在等。`);
  const arC='mp-ar'+(red?' red':'');
  const build=seg=>seg(cf*fa,'mp-cmp',ctip(cf*fa,tx('forward, attention','前向，attention')))+(t>1?seg(arT,arC,atip):'')+seg(cf*(1-fa),'mp-cmp',ctip(cf*(1-fa),tx('forward, MLP','前向，MLP')))+(t>1?seg(arT,arC,atip):'')+
    seg(cb*(1-fa),'mp-cmp b',ctip(cb*(1-fa),tx('backward, MLP','反向，MLP')))+(t>1?seg(arT,arC,atip):'')+seg(cb*fa,'mp-cmp b',ctip(cb*fa,tx('backward, attention','反向，attention')))+(t>1?seg(arT,arC,atip):'');
  const lt=build(segAt(scale)), lz=t>1?build(segAt(r.total)):'';
  $('#mp-lt-cap',app).textContent=t===1?tx(`One layer, one micro-batch of 4,096 tokens, on one GPU: ${fMs(r.total)} in all`,`一层、一个 4,096 个 token 的 micro-batch，在一张卡上：一共 ${fMs(r.total)}`):tx(`The same layer on each of the ${t} GPUs: ${fMs(r.total)} in all`,`同一层在 ${t} 张卡的每一张上：一共 ${fMs(r.total)}`);
  const ref=`<span class="mp-ltref" style="width:${(r1.total/scale*100).toFixed(3)}%" tabindex="0" data-tipx="${esc(tx(`The dashed outline is the same layer on one GPU: ${fMs(r1.total)}.`,`虚线框是同一层放在一张卡上的长度：${fMs(r1.total)}。`))}"></span>`;
  /* S13: the same per-GPU layer again, scaled to its own length, so that the all-reduces are wide enough to see (R5.10) */
  const zcap=t>1?tx(`Zoomed to each GPU’s own ${fMs(r.total)}. The four all-reduces take ${fMs(r.comm)} of it, ${pct(r.share)}.`,`放大到每张卡自己的 ${fMs(r.total)}。其中 4 次 all-reduce 共 ${fMs(r.comm)}，占 ${pct(r.share)}。`):'—';
  const lh=`<div class="mp-ltbar">${ref}<span class="mp-ltsegs">${lt}</span></div><div class="mp-ltax"><span>0</span><span>${fMs(scale)}</span></div>`+
    `<div class="mp-ltz"${t>1?'':' aria-hidden="true"'}><div class="mp-ltzc">${esc(zcap)}</div><div class="mp-ltbar z"><span class="mp-ltsegs">${lz}</span></div><div class="mp-ltax"><span>0</span><span>${t>1?fMs(r.total):''}</span></div></div>`;
  const ltel=$('#mp-lt',app); if(ltel._h!==lh){ltel.innerHTML=lh; ltel._h=lh;}
  const lg=`<span><i class="sw mp-sw-cmp"></i>${esc(tx('compute (forward, then backward)','计算（先前向，后反向）'))}</span>`+(t>1?`<span><i class="sw mp-sw-ar${red?' red':''}"></i>${esc(red?tx('all-reduce over InfiniBand: the GPU waits','all-reduce 走 InfiniBand：卡在等'):tx('all-reduce over NVLink','all-reduce 走 NVLink'))}</span>`:'')+
    `<span><i class="sw mp-sw-ref"></i>${esc(tx('dashed: the layer on one GPU','虚线框：一张卡算这一层'))}</span><span><i class="sw mp-sw-w"></i>${esc(tx('blue: weights','蓝色：权重'))}</span>`;
  const lge=$('#mp-ltleg',app); if(lge._h!==lg){lge.innerHTML=lg; lge._h=lg;}
}
/* part 6, Scenarios, challenge: the cluster (rows = pipelines, columns = stages, a cell = one TP group) and the memory bar */
function drawCluster(){
  const r=cost(S), t=S.t, p=S.p, d=S.d, md=MD[S.model];
  const cap=tx(`${md.name} on ${fint(r.gpus)} H100s: ${p} ${p===1?'stage':'stages'} across, ${d} ${d===1?'pipeline':'pipelines'} down, ${t} ${t===1?'GPU':'GPUs'} per cell`,`${md.name}，${fint(r.gpus)} 张 H100：横向 ${p} 段，纵向 ${d} 条流水线，每格 ${t} 张卡`);
  const rowsShown=Math.min(d,8), colsShown=Math.min(p,16), machine=machOf;
  let h=`<div class="mp-clg" style="--c:${colsShown}">`;
  for(let i=0;i<rowsShown;i++) for(let j=0;j<colsShown;j++){
    const m0=machine(i,j), tip=tx(`Pipeline ${i+1}, stage ${j+1} (layers ${Math.round(j*md.L/p)+1}–${Math.round((j+1)*md.L/p)}): ${t} ${t===1?'GPU':'GPUs'} in machine ${m0} split each of these layers${t>1?' with tensor parallelism':''}.`,`第 ${i+1} 条流水线的第 ${j+1} 段（第 ${Math.round(j*md.L/p)+1}–${Math.round((j+1)*md.L/p)} 层）：机器 ${m0} 里的 ${t} 张卡${t>1?'用张量并行':''}分担这几层。`);
    h+=`<span class="mp-cell" tabindex="0" data-tipx="${esc(tip)}"><span class="mp-gp" style="--t:${t}">${'<i></i>'.repeat(t)}</span>${rowsShown*colsShown<=32?`<small><span class="l">${tx('machine','机器')} ${m0}</span><span class="s">${m0}</span></small>`:''}</span>`;}
  h+=`</div>`;
  if(d>rowsShown) h+=`<p class="mp-more small muted">${tx(`… and ${d-rowsShown} more pipelines like these`,`……下面还有 ${d-rowsShown} 条同样的流水线`)}</p>`;
  /* B1/B3: the two links the card's rule is about, named where the machines are drawn */
  h+=`<p class="mp-more mp-links small muted">${tx(`Inside a machine: NVLink, ${GPU.nv/1e9} GB/s each way per GPU. Between machines: InfiniBand, ${GPU.ib/1e9} GB/s each way per GPU.`,`机器里：NVLink，每张卡每个方向 ${GPU.nv/1e9} GB/s。机器之间：InfiniBand，每张卡每个方向 ${GPU.ib/1e9} GB/s。`)}</p>`;
  const cl=$('#mp-cl',app); if(cl._h!==h){cl.innerHTML=h; cl._h=h;}
  /* R7.4, R7.8: “machine 3” under each cell when it fits; else only the number, which the caption then explains; else nothing (the
     cell's hover still names its machine). Measured on every draw, so a resize or a language switch picks the form again. */
  const g=$('.mp-clg',cl), tooWide=()=>[...cl.querySelectorAll('.mp-cell small')].some(e=>e.scrollWidth>e.clientWidth+0.5);
  g.classList.remove('short','none'); if(tooWide()){g.classList.add('short'); if(tooWide()){g.classList.remove('short'); g.classList.add('none');}}
  const capEl=$('#mp-cl-cap',app), capTxt=cap+(g.classList.contains('short')?tx('; the number under a cell is its machine','，格子下的数字是机器编号'):''); if(capEl.textContent!==capTxt) capEl.textContent=capTxt;
  /* memory per GPU */
  const lim=GPU.mem, mx=Math.max(lim*1.25,r.mem), w=x=>(x/mx*100).toFixed(2), over=!r.fits;
  const mh=`<div class="mp-mcap">${tx('Memory per GPU (a GPU of stage 1, the most loaded)','每张卡的显存（第 1 段的卡，最满的那张）')}</div><div class="mp-mbar${over?' over':''}">`+
    `<i class="mp-mw" style="width:${w(r.weights)}%" tabindex="0" data-tipx="${esc(tx(`BF16 weights: ${fGB(r.weights)}, 2 bytes for each of the ${big(r.pg)} parameters on this GPU.`,`BF16 权重：${fGB(r.weights)}，这张卡上 ${big(r.pg)} 个参数各 2 字节。`))}"></i>`+
    `<i class="mp-mo" style="width:${w(r.states-r.weights)}%" tabindex="0" data-tipx="${esc(tx(`FP32 gradients and optimizer state: ${fGB(r.states-r.weights)}, ${fX(r.bpp-2)} bytes per parameter (the gradient 4, the FP32 copy and Adam’s m and v 12${S.zero&&d>1?` split over ${d} data-parallel copies by ZeRO-1`:''}).`,`FP32 梯度和优化器状态：${fGB(r.states-r.weights)}，每个参数 ${fX(r.bpp-2)} 字节（梯度 4 字节，FP32 权重副本和 Adam 的 m、v 共 12 字节${S.zero&&d>1?`，由 ZeRO-1 分到 ${d} 份数据并行上`:''}）。`))}"></i>`+
    `<i class="mp-ma" style="width:${w(r.act)}%" tabindex="0" data-tipx="${esc(tx(`Activations: ${fGB(r.act)}, ${r.inflight} ${r.inflight===1?'micro-batch':'micro-batches'} in flight × ${fGB(r.actMb)} each.`,`激活：${fGB(r.act)}，同时存 ${r.inflight} 个 micro-batch，每个 ${fGB(r.actMb)}。`))}"></i>`+
    `<b class="mp-mlim" style="left:${w(lim)}%"><em>80 GB</em></b></div><div class="mp-mleg"><span><i class="sw mp-sw-w"></i>${esc(tx('BF16 weights','BF16 权重'))}</span><span><i class="sw mp-sw-o"></i>${esc(tx('FP32 gradients + optimizer state','FP32 梯度 + 优化器状态'))}</span><span><i class="sw mp-sw-a"></i>${esc(tx('activations','激活'))}</span><span class="mp-mtot${over?' bad':''}">${esc(tx(`${fGB(r.mem)} of 80 GB`,`${fGB(r.mem)} / 80 GB`))}</span></div>`;
  const me=$('#mp-mem',app); if(me._h!==mh){me.innerHTML=mh; me._h=mh;}
}
function statusOf(){
  const v=viewNow();
  if(predHidden()) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The schedule stays hidden until you answer the question in the card.','先在卡片里作答，再显示这一步怎么排。')};
  if(v==='gantt'){const at=simAt(simNow(),ph*simNow().T); const b=ph>=1?simNow().idle:at.idle;
    return {cls:'neutral',txt:tx(`Idle ${pct(b)}`,`空等 ${pct(b)}`),short:tx(`Idle ${pct(b)}`,`空等 ${pct(b)}`),tip:ph>=1?tx(`Over the whole step, each GPU is idle ${pct(b)} of the time.`,`整步下来，每张卡有 ${pct(b)} 的时间空等。`):tx(`So far in this step, the GPUs have been idle ${pct(b)} of the time.`,`这一步到现在，各张卡有 ${pct(b)} 的时间在空等。`)};}
  if(v==='layer'){const r=layer(S.t,S.place); const bad=r.share>0.5;
    return {cls:bad?'bad':'neutral',txt:S.t===1?tx('No communication','不用通信'):tx(`Waiting ${pct(r.share)}`,`等通信 ${pct(r.share)}`),short:S.t===1?tx('No comms','不用通信'):tx(`Waiting ${pct(r.share)}`,`等 ${pct(r.share)}`),tip:tx(`Share of the layer’s time each GPU waits for all-reduces: ${pct(r.share)}.`,`每张卡在这一层里等 all-reduce 的时间占 ${pct(r.share)}。`)};}
  const r=cost(S);
  return {cls:r.fits?'ok':'bad',txt:r.fits?tx('Fits in 80 GB','放得下'):tx('Out of memory','显存不够'),short:r.fits?tx('Fits','放得下'):tx('Out of memory','显存不够'),tip:tx(`Each GPU needs ${fGB(r.mem)} of its 80 GB.`,`每张卡要用 80 GB 里的 ${fGB(r.mem)}。`)};
}
function drawHeadline(){
  const v=viewNow(), t=$('#mp-st-title',app), sub=$('#mp-st-sub',app), st=$('#mp-status',app), md=MD[S.model];
  if(v==='gantt'){t.textContent=tx(`Pipeline of ${S.p} GPUs: who is busy when?`,`${S.p} 张卡的流水线：谁在什么时候干活？`); sub.textContent=mode==='guided'&&STEPS[step].id==='mission'?tx(`${md.name}: p = ${S.p} stages of ${md.L/S.p} layers, the whole batch at once`,`${md.name}：p = ${S.p} 段，每段 ${md.L/S.p} 层，整批一次送进去`):tx(`${md.name}: p = ${S.p} stages of ${md.L/S.p} layers, m = ${S.m} ${S.m===1?'micro-batch':'micro-batches'}`,`${md.name}：p = ${S.p} 段，每段 ${md.L/S.p} 层，m = ${S.m} 个 micro-batch`);}
  else if(v==='layer'){t.textContent=tx('Tensor parallelism inside one layer','一层里面的张量并行'); sub.textContent=tx(`Llama‑3.1‑70B, ${S.t} ${S.t===1?'GPU':'GPUs'}${S.t>1?(S.place==='out'?', one in each of 8 machines':', in one machine'):''}`,`Llama‑3.1‑70B，${S.t} 张卡${S.t>1?(S.place==='out'?'，分在 8 台机器上':'，在一台机器里'):''}`);}
  else {t.textContent=mode==='challenge'?tx('Your layout of 70B on 32 H100s','你在 32 张 H100 上的 70B 排法'):tx('How the GPUs are laid out','这些卡怎么排'); sub.textContent=layoutSym(S.t,S.p,S.d,S.m);}
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=viewNow();
  if(v==='gantt'){
    if(predHidden()) return [{k:'idle',label:tx('Idle share of the step','这一步空等的比例'),v:'—',sub:tx('shown in part 2','第 2 部分揭晓'),tip:tx('Hidden until you answer the question.','作答之前先隐藏。')},{k:'act',label:tx('Activations GPU 1 holds','第 1 张卡存的激活'),v:'—',sub:tx('micro-batches','个 micro-batch'),tip:tx('Hidden until you answer the question.','作答之前先隐藏。')}];
    const s=simNow(), done=ph>=1, at=simAt(s,ph*s.T), mb=cost(S).actMb, pk=done?s.peak:at.peak;
    return [{k:'idle',label:done?tx('Idle share of the step','这一步空等的比例'):tx('Idle so far','到现在空等的比例'),v:pct(done?s.idle:at.idle),sub:done?tx(`whole step, p = ${S.p} GPUs`,`整步，p = ${S.p} 张卡`):tx('live, as it plays','播放中实时计算'),tip:tx(`The share of GPU time spent idle${done?' over the whole step':' so far'}: ${pct(done?s.idle:at.idle)}.`,`${done?'整步':'到现在'}所有卡空等的时间占 ${pct(done?s.idle:at.idle)}。`)},
      {k:'act',label:tx('Activations GPU 1 holds','第 1 张卡存的激活'),v:String(pk),sub:mode==='guided'&&STEPS[step].id==='gpipe'?tx(`${pk===1?'micro-batch':'micro-batches'} at most; part 3 explains`,`个 micro-batch（最多），第 3 部分讲`):tx(`${pk===1?'micro-batch':'micro-batches'} at most, about ${fGB(pk*mb)}`,`个 micro-batch（最多），约 ${fGB(pk*mb)}`),tip:tx(`At most ${pk} micro-batches${done?'':' so far'} have their activations in GPU 1’s memory at once; each takes about ${fGB(mb)} on its ${MD[S.model].L/S.p} layers.`,`${done?'':'到现在，'}第 1 张卡最多同时存着 ${pk} 个 micro-batch 的激活；每个在它那 ${MD[S.model].L/S.p} 层上约占 ${fGB(mb)}。`)},
      {k:'slot',label:tx('Step length, m + p − 1','这一步的长度 m + p − 1'),v:String(S.m+S.p-1),sub:tx('forward-and-backward slots','格（每格一次前向加一次反向）'),tip:tx(`m + p − 1 = ${S.m} + ${S.p} − 1 = ${S.m+S.p-1} slots; each GPU works in ${S.m} of them.`,`m + p − 1 = ${S.m} + ${S.p} − 1 = ${S.m+S.p-1} 格；每张卡只有 ${S.m} 格在干活。`)}];}
  if(v==='layer'){const r=layer(S.t,S.place);
    return [{k:'comp',label:tx('Compute per GPU','每张卡的计算'),v:fMs(r.comp),sub:tx('one layer, forward + backward','一层，前向加反向'),tip:tx(`6 × ${big(r.c.Pl)} parameters × 4,096 tokens ÷ ${S.t} GPUs ÷ (989.5 TFLOP/s × ${SH()}) = ${fMs(r.comp)}. ${SH()} is an assumption: while computing, a GPU reaches ${SH()} of the peak.`,`6 × ${big(r.c.Pl)} 个参数 × 4,096 个 token ÷ ${S.t} 张卡 ÷（989.5 TFLOP/s × ${SH()}）= ${fMs(r.comp)}。其中 ${SH()} 是假设值，GPU 算的时候能跑到峰值的 ${SH()}。`)},
      {k:'comm',label:tx('All-reduces per layer','每层的 all-reduce'),v:S.t===1?'0':fMs(r.comm),sub:S.t===1?tx('none on one GPU','一张卡不用'):tx(`4 × ${fMB(r.bytes)} over ${S.place==='out'?'InfiniBand':'NVLink'}`,`4 次，各 ${fMB(r.bytes)}，走 ${S.place==='out'?'InfiniBand':'NVLink'}`),tip:S.t===1?tx('One GPU holds the whole layer, so nothing needs to be added up across GPUs.','一张卡放得下整层，不用和别的卡相加，也就没有 all-reduce。'):tx(`4 all-reduces of ${fMB(r.bytes)}, each ${fMs(r.ar)}.`,`4 次 ${fMB(r.bytes)} 的 all-reduce，每次 ${fMs(r.ar)}。`)},
      {k:'share',label:tx('Time spent waiting','等通信的时间'),v:pct(r.share),sub:tx('of the layer’s time','占这一层的时间'),tip:S.t===1?tx('On one GPU no time goes to all-reduces.','一张卡时，没有时间花在 all-reduce 上。'):tx(`${fMs(r.comm)} ÷ (${fMs(r.comp)} + ${fMs(r.comm)}) = ${pct(r.share)}.`,`${fMs(r.comm)} ÷（${fMs(r.comp)} + ${fMs(r.comm)}）= ${pct(r.share)}。`)},
      {k:'par',label:tx('Parameters per GPU','每张卡的参数'),v:big(r.params),sub:tx('of this layer’s 856M','这一层共 856M'),tip:tx(`${big(r.c.Pl)} ÷ ${S.t} = ${big(r.params)} parameters of this layer on each GPU.`,`${big(r.c.Pl)} ÷ ${S.t} = 每张卡 ${big(r.params)} 个这一层的参数。`)}];}
  const r=cost(S);
  return [{k:'mem',label:tx('Memory per GPU','每张卡的显存'),v:fGB(r.mem),sub:r.fits?tx('fits in 80 GB','放得下 80 GB'):tx('over 80 GB','超过 80 GB'),tip:tx(`${fGB(r.states)} of model states + ${fGB(r.act)} of activations = ${fGB(r.mem)}.`,`模型状态 ${fGB(r.states)} + 激活 ${fGB(r.act)} = ${fGB(r.mem)}。`)},
    {k:'bub',label:tx('Bubble','bubble'),v:pct(r.bubble),sub:tx('of the step, idle','一步里空等'),tip:tx('Idle slots ÷ all slots of a GPU:','每张卡空等的格数 ÷ 全部格数：')+`<div class="fx">${A.ui.math(`\\frac{p - 1}{m + p - 1} = \\frac{${S.p} - 1}{${S.m} + ${S.p} - 1} = ${pct(r.bubble).replace('%','\\%')}`)}</div>`},
    {k:'tp',label:tx('Tensor-parallel all-reduces','张量并行的 all-reduce'),v:pct(r.tpShare),sub:tx('of the step','占一步'),tip:tx(`Each GPU waits ${pct(r.tpShare)} of the step for tensor-parallel all-reduces.`,`每张卡有 ${pct(r.tpShare)} 的时间在等张量并行的 all-reduce。`)},
    {k:'step',label:tx('Time per step','每步耗时'),v:fS(r.step),sub:tx(`${fint(r.gpus)} GPUs`,`${fint(r.gpus)} 张卡`),tip:tx(`The pipeline’s ${fS2(r.pipe)} plus ${fS2(r.dp)} for the data-parallel gradient all-reduce.`,`流水线的 ${fS2(r.pipe)}，加上数据并行梯度 all-reduce 的 ${fS2(r.dp)}。`)}];
}
function drawMetrics(){
  const host=$('#mp-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  /* R6.6: a badge only for the learner's own change, against the values shown just before it in the same view. A new part,
     mode or view starts without badges; a state the lab sets (a scenario, entering a part) clears them. */
  const ctx=mode+':'+step+':'+viewNow(), fresh=host._ctx!==ctx; host._ctx=ctx;
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!quiet&&!fresh&&prev!=null&&prev!==m.v&&!playing){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v)),pts:/%$/.test(m.v)})||'';}
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(userAct||fresh||(prev!=null&&prev!==m.v)){clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
const f4=x=>{if(x===0) return '0'; const d=Math.max(0,3-Math.floor(Math.log10(Math.abs(x)))); return fixd(x,Math.min(d,7));};
function mathHTML(){
  const v=viewNow(), t=T_, SHm=SH().replace('%','\\%'), shN=fL(SHARE);
  if(predHidden()) return `<p>${tx('The formulas appear once you have answered the question in part 1.','在第 1 部分作答以后，这里给出公式。')}</p>`;
  const rows=[];
  if(v==='gantt'){const s=simNow(), c=cost(S);
    rows.push(M('\\text{bubble} = \\frac{p - 1}{m + p - 1}')+`<p>${t('p','p')} = ${S.p}${tx(', ','，')}${t('m','m')} = ${S.m}${tx('. ','。')}${S.p-1} ÷ ${S.m+S.p-1} = <b>${pct(s.idle)}</b></p>`);
    rows.push(`<p>${tx('One slot is the time one stage takes for one forward and one backward of one micro-batch.','一格是一段流水线给一个 micro-batch 做一次前向和一次反向的时间。')}</p>`+M(tx(`\\text{one slot} = \\frac{6 \\times N_{\\text{layer}} \\times \\frac{L}{p} \\times \\text{tokens}}{989.5\\ \\text{TFLOP/s} \\times ${SHm}}`,`\\text{一格的时间} = \\frac{6 \\times N_{\\text{层}} \\times \\frac{L}{p} \\times \\text{token 数}}{989.5\\ \\text{TFLOP/s} \\times ${SHm}}`))+
      `<p>= 6 × ${fint(c.Pl)} × ${fL(c.Ls)} × ${fint(c.tok)} ÷ (989.5 × 10¹² × ${shN}) = ${f4(c.per)} s</p>`+`<p>${tx(`${SH()} is an assumption: while computing, a GPU reaches ${SH()} of the peak.`,`其中 ${SH()} 是假设值，GPU 算的时候能跑到峰值的 ${SH()}。`)}</p>`+
      M(tx('\\text{step} = (m + p - 1) \\times \\text{one slot}','\\text{每步耗时} = (m + p - 1) \\times \\text{一格的时间}'))+`<p>= ${S.m+S.p-1} × ${f4(c.per)} s = ${fS(c.pipe)}</p>`);
    rows.push(`<p>${tx('Activations GPU 1 holds follow Megatron-LM’s per-layer formula.','第 1 张卡存的激活按 Megatron-LM 的每层公式算。')}</p>`+M(tx('\\text{act}','\\text{激活}')+' = n \\times \\frac{L}{p} \\times s \\times b \\times h \\times (18 + 4\\frac{I}{h})')+
      `<p>${S.sched==='gpipe'?tx(`GPipe: ${A.ui.math(`n = m = ${S.m}`)}.`,`GPipe 时 ${A.ui.math(`n = m = ${S.m}`)}。`):tx(`1F1B: ${A.ui.math(`n = \\min(m, p) = ${c.inflight}`)}.`,`1F1B 时 ${A.ui.math(`n = \\min(m, p) = ${c.inflight}`)}。`)}</p><p>= ${c.inflight} × ${fL(c.Ls)} × ${fint(S.seq)} × 1 × ${fint(MD[S.model].h)} × ${fK(MD[S.model])} = <b>${fGB(c.act)}</b></p>`);
    return rows.join('');}
  if(v==='layer'){const r=layer(S.t,S.place);
    rows.push(M(tx(`\\text{compute time} = \\frac{6 \\times N_{\\text{layer}} \\times \\text{tokens}}{t \\times 989.5\\ \\text{TFLOP/s} \\times ${SHm}}`,`\\text{计算时间} = \\frac{6 \\times N_{\\text{层}} \\times \\text{token 数}}{t \\times 989.5\\ \\text{TFLOP/s} \\times ${SHm}}`))+`<p>= 6 × ${fint(r.c.Pl)} × 4,096 ÷ (${t('t',S.t)} × 989.5 × 10¹² × ${shN}) = <b>${fMs(r.comp)}</b></p>`);
    if(S.t>1) rows.push(M(tx('\\text{one all-reduce}','\\text{一次 all-reduce 的时间}')+' = \\frac{2(t - 1)}{t} \\times \\frac{s \\times h \\times 2}{'+tx('\\text{bandwidth}','\\text{带宽}')+'}')+`<p>= 2 × ${S.t-1} ÷ ${S.t} × ${fint(r.bytes)} B ÷ ${S.place==='out'?'50':'450'} GB/s = ${f4(r.ar*1e3)} ms</p><p>${tx(`Each layer does four, 4 × ${f4(r.ar*1e3)} ms = <b>${fMs(r.comm)}</b>.`,`每层要做 4 次，4 × ${f4(r.ar*1e3)} ms = <b>${fMs(r.comm)}</b>。`)}</p>`);
    return rows.join('');}
  const r=cost(S), md=MD[S.model], lc=r.comp/r.Ls;
  /* one slot: the stage's L/p layers, each its compute plus its 4 tensor-parallel all-reduces (part 4's two times); check b8v1, R6.6 */
  const slotRow=S.t>1?M(tx('\\text{one slot} = \\frac{L}{p} \\times (\\text{layer compute} + 4 \\times \\text{all-reduce})','\\text{一格的时间} = \\frac{L}{p} \\times (\\text{计算} + 4 \\times \\text{all-reduce})'))+`<p>= ${fL(r.Ls)} × (${f4(lc*1e3)} ms + 4 × ${f4(r.ar*1e3)} ms) = ${f4(r.per)} s</p>`
    :M(tx('\\text{one slot} = \\frac{L}{p} \\times \\text{layer compute}','\\text{一格的时间} = \\frac{L}{p} \\times \\text{一层的计算}'))+`<p>= ${fL(r.Ls)} × ${f4(lc*1e3)} ms = ${f4(r.per)} s</p>`;
  rows.push(M(tx('\\text{model states}','\\text{模型状态}')+' = \\frac{N}{t \\times p} \\times '+(S.zero?'(6 + \\frac{12}{d})':'18'))+`<p>= ${fint(md.N)} ÷ (${S.t} × ${S.p}) × ${fX(r.bpp)} = ${fGB(r.states)}</p>`);
  rows.push(M(tx('\\text{act}','\\text{激活}')+' = n \\times \\frac{L}{p} \\times \\frac{s \\times b \\times h \\times (18 + 4\\frac{I}{h})}{t}')+`<p>${S.sched==='gpipe'?tx(`GPipe: ${A.ui.math(`n = m = ${S.m}`)}.`,`GPipe 时 ${A.ui.math(`n = m = ${S.m}`)}。`):tx(`1F1B: ${A.ui.math(`n = \\min(m, p) = ${r.inflight}`)}.`,`1F1B 时 ${A.ui.math(`n = \\min(m, p) = ${r.inflight}`)}。`)}</p><p>= ${r.inflight} × ${fL(r.Ls)} × ${fint(S.seq)} × ${S.b} × ${fint(md.h)} × ${fK(md)} ÷ ${S.t} = ${fGB(r.act)}</p><p>${tx(`Model states plus activations: ${fGB(r.states)} + ${fGB(r.act)} = <b>${fGB(r.mem)}</b>.`,`模型状态加激活，${fGB(r.states)} + ${fGB(r.act)} = <b>${fGB(r.mem)}</b>。`)}</p>`);
  if(S.d>1){
    rows.push(`<p>${tx('Gradients travel in BF16, 2 bytes each.','梯度用 BF16 传，每个梯度占 2 字节。')} ${r.dpIn?tx(`The ${S.d} data-parallel copies sit in one machine, so the gradient all-reduce is a ring over NVLink.`,`数据并行的 ${S.d} 份都在同一台机器里，梯度的 all-reduce 走 NVLink，用环形算法。`):tx('The data-parallel copies sit in different machines. An all-reduce across machines moves the stage’s whole gradient twice through each machine’s 400 GB/s.','数据并行的各份分在不同的机器上。跨机器的 all-reduce 要把这一段的整块梯度传两遍，走每台机器 400 GB/s 的网络。')}</p>`+
      (r.dpIn?M(tx('\\text{gradient all-reduce}','\\text{梯度的 all-reduce}')+' = \\frac{2(d - 1)}{d} \\times \\frac{2N / (t \\times p)}{450\\ \\text{GB/s}}')+`<p>= 2 × ${S.d-1} ÷ ${S.d} × (2 × ${fint(md.N)} ÷ (${S.t} × ${S.p})) B ÷ 450 GB/s = ${f4(r.dp)} s</p>`
             :M(tx('\\text{gradient all-reduce}','\\text{梯度的 all-reduce}')+' = 2 \\times \\frac{2N / p}{400\\ \\text{GB/s}}')+`<p>= 2 × (2 × ${fint(md.N)} ÷ ${S.p}) B ÷ 400 GB/s = ${f4(r.dp)} s</p>`));
    /* two short lines instead of one long one, so that nothing is cut at 390 px */
    rows.push(slotRow);
    rows.push(M(tx('\\text{pipeline} = (m + p - 1) \\times \\text{one slot}','\\text{流水线} = (m + p - 1) \\times \\text{一格的时间}'))+`<p>= ${S.m+S.p-1} × ${f4(r.per)} s = ${f4(r.pipe)} s</p>`);
    rows.push(M(tx('\\text{step} = \\text{pipeline} + \\text{gradient all-reduce}','\\text{每步耗时} = \\text{流水线} + \\text{梯度的 all-reduce}'))+`<p>= ${f4(r.pipe)} s + ${f4(r.dp)} s = <b>${f4(r.step)} s</b></p>`);
  }
  else rows.push(slotRow+M(tx('\\text{step} = (m + p - 1) \\times \\text{one slot}','\\text{每步耗时} = (m + p - 1) \\times \\text{一格的时间}'))+`<p>${tx('One data-parallel copy: no gradient all-reduce.','数据并行只有一份，不用做梯度的 all-reduce。')}</p><p>= ${S.m+S.p-1} × ${f4(r.per)} s = <b>${f4(r.step)} s</b></p>`);
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){const c=chCfg(S.t,S.p,S.m)||chCfg(S.t,S.p,CH.B/(CH.G/(S.t*S.p))); if(c) Object.assign(S,c); else Object.assign(S,chCfg(CH.start.t,CH.start.p,CH.start.m));}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const v=viewNow();
  $('#mp-gbox',app).hidden=!(v==='gantt'||(v==='cluster'&&S.p>1));
  $('#mp-lbox',app).hidden=v!=='layer';
  $('#mp-clbox',app).hidden=v!=='cluster';
  $('#mp-graph',app).dataset.v=v;
  drawHeadline();
  if(!$('#mp-gbox',app).hidden) drawGantt();
  if(v==='layer') drawLayer();
  if(v==='cluster') drawCluster();
  drawMetrics();
  const mb=$('#mp-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); dockK=''; A.dock.refresh(); announce();
}
/* only the playhead moved: the mask, the player, the live cards, the badge; B2 (one clock): the phone's mini result bar reads the
   same idle share as the cards, so it moves with them while it plays instead of jumping at the end */
function renderPlay(){if(!app) return; drawHead(); drawPlayer(); drawMetrics(); drawHeadline(); if(ph>=1){dockK=''; updateRail(); A.dock.refresh(); return;}
  const k=[viewNow(),simKey(),statusOf().short].join('|'); if(k!==dockK){dockK=k; A.dock.render();}}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the pipeline stages, tensor parallelism and the micro-batches, in the task card. Everything else is fixed: Llama‑3.1‑70B, 32 H100s, 256 sequences of 4,096 tokens per step, ZeRO-1, 1F1B.','挑战中只能在任务卡里改流水线段数、张量并行和 micro-batch 数。其余设置固定：Llama‑3.1‑70B，32 张 H100，每步 256 条 4,096 个 token 的序列，ZeRO-1，1F1B。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return tx('Fixed in this part: Llama‑3.1‑70B, p = 8 stages of 10 layers, the whole batch at once. Answer in the card.','这一部分固定：Llama‑3.1‑70B，流水线段数 p = 8，每段 10 层，整批一次送进去。在卡片里作答。');
  if(id==='gpipe') return tx('Fixed in this part: Llama‑3.1‑70B, p = 8 stages of 10 layers, GPipe. Only the number of micro-batches m changes, in the card.','这一部分固定：Llama‑3.1‑70B，流水线段数 p = 8，每段 10 层，GPipe。只在卡片里改 micro-batch 数 m。');
  if(id==='1f1b') return tx('Fixed in this part: p = 8 stages, m = 16 micro-batches. Pick the schedule in the card.','这一部分固定：流水线段数 p = 8，micro-batch 数 m = 16。在卡片里选排法。');
  if(id==='tp'||id==='link') return tx('Fixed in this part: one layer of Llama‑3.1‑70B and one micro-batch of 4,096 tokens. Change it in the card.','这一部分固定：Llama‑3.1‑70B 的一层，一个 4,096 个 token 的 micro-batch。在卡片里改。');
  return tx('Fixed in this part. You can change every setting in Scenarios.','这一部分固定不变。在应用场景里可以改所有设置。');
}
function syncControls(){
  const set=(key,val)=>$$(`.seg-ctl[data-key="${key}"]`,app).forEach(g=>A.ui.setPressed(g,String(val)));
  ['model','t','p','d','m','seq','sched','place','zero'].forEach(k=>set(k,S[k])); set('cht',S.t); set('chp',S.p); set('chm',S.m);
  const g=mode==='guided', ch=mode==='challenge';
  $('#mp-controls',app).classList.toggle('locked',ch||g);
  $$('#mp-controls .lockable',app).forEach(e=>{e.hidden=g||ch;});
  const ln=$('#mp-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||predHidden()) return; const v=viewNow();
  if(v==='gantt'){const s=simNow(); if(ph>=1) A.announce(tx(`p = ${S.p} stages, m = ${S.m} micro-batches: each GPU idle ${pct(s.idle)} of the step.`,`流水线段数 p = ${S.p}，micro-batch 数 m = ${S.m}：每张卡空等 ${pct(s.idle)}。`)); return;}
  if(v==='layer'){const r=layer(S.t,S.place); A.announce(tx(`TP ${S.t}: all-reduces ${pct(r.share)} of the layer’s time.`,`张量并行 ${S.t} 份：all-reduce 占这一层时间的 ${pct(r.share)}。`)); return;}
  const r=cost(S); A.announce(tx(`${fGB(r.mem)} per GPU, ${fS(r.step)} per step.`,`每张卡 ${fGB(r.mem)}，每步 ${fS(r.step)}。`));},400);}

/* =====================================================================
   the player (R3.9, R3.20)
   ===================================================================== */
function tick(tm){
  raf=0; if(!app||!playing) return;
  const dt=lastT?tm-lastT:16; lastT=tm;
  ph=Math.min(1,ph+dt/DUR[spd]);
  if(ph>=1){playing=false; ph=1; watched.add(simKey()); wantObs=true; renderPlay(); wantObs=false; revealSoonIf(); return;}
  renderPlay(); raf=requestAnimationFrame(tick);
}
function play(fromStart){if(fromStart||ph>=1) ph=0; playing=true; lastT=0; if(A.reduceMotion()){finishRun(); return;} if(!raf) raf=requestAnimationFrame(tick); drawPlayer();}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; if(app) drawPlayer();}
function finishRun(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; ph=1; watched.add(simKey()); renderPlay();}
/* a new picture plays from the start; one already watched is shown finished (R3.20) */
function cue(){
  if(predHidden()){pause(); ph=0; return;}
  /* only parts 2 and 3 are about the schedule: they play a new picture once; everywhere else the step is shown finished */
  const auto=mode==='guided'&&(isP('gpipe')||isP('1f1b'));
  if(watched.has(simKey())||!auto){pause(); ph=1; return;}
  ph=0; if(!relang) play(true);
}
let lastKey='';
function cueIfNew(){const k=simKey()+'|'+(predHidden()?'h':''); if(k===lastKey) return; lastKey=k; cue();}
function revealSoonIf(){if(mode==='guided'&&(isP('gpipe')||isP('1f1b'))){wantObs=true; updateRail(); wantObs=false; revealSoon();}}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  if(isP('gpipe')) mSeen.add(S.m);
  if(isP('tp')) tSeen.add(S.t);
  cueIfNew(); render();
}
function byLearner(fn,o){userAct=true; quiet=!!(o&&o.badges===false); if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false; quiet=false;}}
function setLv3(v){lv3=v; tried3.add(v); setState({sched:v});}
function setLv5(v){lv5=v; tried5.add(v); setState({place:v});}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='gpipe') return mCtl('mp-ph-m',true);
  if(id==='tp') return tCtl('mp-ph-t',true);
  if(id==='1f1b'||id==='link') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the two options in the card.','向下滚到卡片里的两个选项。'))}">${tx('Pick one below ↓','在下面选一个 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#mp-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  if(s.id==='gpipe') mSeen.add(S.m); if(s.id==='tp') tSeen.add(S.t);
  const st=$('#mp-stage',app); if(st) st.scrollTop=0;
  cueIfNew(); renderRail();
}
function revealStep(){A.ui.reveal($('#mp-rail .step.current',app)||$('#mp-rail',app));}
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
    const st=$('#mp-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#mp-rail .step.current',app)||$('#mp-rail .railcard',app)||$('#mp-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#mp-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#mp-insight',app)); else revealBox($('#mp-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; pause();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...chCfg(CH.start.t,CH.start.p,CH.start.m)};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#mp-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  cueIfNew(); renderRail();
}
const MI=['8b','70b','405b'], SQ=[4096,32768,131072];
const token=(s=S)=>[MI.indexOf(s.model),s.t,s.p,s.d,s.m,SQ.indexOf(s.seq),s.sched==='gpipe'?0:1,s.place==='out'?1:0,s.zero].join('_');
function parse(tok){
  const q=String(tok||'').split('_').map(Number); if(q.length!==9||q.some(x=>!Number.isInteger(x))) return null;
  const [mi,t,p,d,m,si,sc,pl,z]=q;
  if(mi<0||mi>2||![1,2,4,8].includes(t)||![1,2,4,8,16].includes(p)||![1,2,4,8,16,32,64,128].includes(d)||![1,2,4,8,16,32,64,128,256].includes(m)||si<0||si>2||(sc!==0&&sc!==1)||(pl!==0&&pl!==1)||(z!==0&&z!==1)) return null;
  return {model:MI[mi],t,p,d,m,b:1,seq:SQ[si],sched:sc?'1f1b':'gpipe',place:pl?'out':'in',zero:z};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*104729+11), pk=a=>a[Math.floor(r()*a.length)];
  const p=pk([4,8,16]), m=pk([8,16,32,64]), t=pk([2,4,8]), place=pk(['in','out']);
  return {seed,p,m,t,place,bubble:(p-1)/(m+p-1),gp:m,f1:Math.min(m,p),l:layer(t,place)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#mp-drill-q',app), a=$('#mp-drill-a',app);
  qd.innerHTML=tx(`Llama‑3.1‑70B is trained with p = <b>${d.p}</b> pipeline stages and m = <b>${d.m}</b> micro-batches per step, and each layer is split <b>${d.t}</b> ways with tensor parallelism ${d.place==='out'?'<b>across machines</b> (InfiniBand, 50 GB/s each way)':'<b>inside one machine</b> (NVLink, 450 GB/s each way)'}. What share of the step does each GPU sit in the bubble? How many micro-batches of activations does GPU 1 hold under GPipe and under 1F1B? How long do one layer’s all-reduces take for a 4,096-token micro-batch?`,
                  `训练 Llama‑3.1‑70B，流水线段数 p = <b>${d.p}</b>，每步 micro-batch 数 m = <b>${d.m}</b>。每层用张量并行切成 <b>${d.t}</b> 份，${d.place==='out'?'<b>跨机器</b>（InfiniBand，每个方向 50 GB/s）':'<b>在一台机器里</b>（NVLink，每个方向 450 GB/s）'}。每张卡在 bubble 里空等多少？GPipe 和 1F1B 下，第 1 张卡各存几份激活？一个 4,096 个 token 的 micro-batch，一层的 all-reduce 要多久？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>Bubble ${pct(d.bubble)}; GPipe holds ${d.gp}, 1F1B ${d.f1}; the all-reduces take ${fMs(d.l.comm)} per layer, ${pct(d.l.share)} of the layer’s time.</b></p>`+
    `<p>Bubble:</p>`+M(`\\text{bubble} = \\frac{p - 1}{m + p - 1} = \\frac{${d.p-1}}{${d.m+d.p-1}} = ${pct(d.bubble).replace('%','\\%')}`)+`<p>Activations: GPipe keeps all ${A.ui.math(`m = ${d.m}`)}. 1F1B keeps ${A.ui.math(`\\min(m, p) = ${d.f1}`)}.</p>`+
    `<p>All-reduce: 4,096 × 8,192 × 2 = ${fint(d.l.bytes)} bytes; 2 × ${d.t-1} ÷ ${d.t} × ${fint(d.l.bytes)} ÷ ${d.place==='out'?'50':'450'} GB/s = ${fMs(d.l.ar)} each, 4 per layer = ${fMs(d.l.comm)}. The layer’s compute: ${fMs(d.l.comp)} per GPU.</p>`,
    `<p><b>bubble ${pct(d.bubble)}；GPipe 存 ${d.gp} 份，1F1B 存 ${d.f1} 份；一层的 all-reduce 共 ${fMs(d.l.comm)}，占这一层时间的 ${pct(d.l.share)}。</b></p>`+
    `<p>bubble：</p>`+M(`\\text{bubble} = \\frac{p - 1}{m + p - 1} = \\frac{${d.p-1}}{${d.m+d.p-1}} = ${pct(d.bubble).replace('%','\\%')}`)+`<p>激活：GPipe 存全部 ${A.ui.math(`m = ${d.m}`)} 份。1F1B 存 ${A.ui.math(`\\min(m, p) = ${d.f1}`)} 份。</p>`+
    `<p>all-reduce：4,096 × 8,192 × 2 = ${fint(d.l.bytes)} 字节；2 × ${d.t-1} ÷ ${d.t} × ${fint(d.l.bytes)} ÷ ${d.place==='out'?'50':'450'} GB/s = 每次 ${fMs(d.l.ar)}，每层 4 次共 ${fMs(d.l.comm)}。这一层的计算：每张卡 ${fMs(d.l.comp)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={p:{en:'Pipeline stages',zh:'流水线段数'},m:{en:'Micro-batches',zh:'micro-batch 数'},t:{en:'Tensor parallel',zh:'张量并行'}};
const TERM={
 p:()=>tx(`p is the number of pipeline stages, ${S.p} here.`,`p 是流水线的段数，这里是 ${S.p}。`),
 m:()=>tx(`m is the number of micro-batches per step and pipeline, ${S.m} here.`,`m 是每条流水线每步的 micro-batch 数，这里是 ${S.m}。`),
 t:()=>tx(`t is the number of GPUs one layer is split across, ${S.t} here.`,`t 是一层切到的卡数，这里是 ${S.t}。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return k==='m'||k==='p'||k==='t'?'':tx('Fixed during the challenge.','挑战中固定不变。');
  if(mode!=='guided') return '';
  if(k==='m'&&isP('gpipe')) return ''; if(k==='t'&&isP('tp')) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({p:mode==='challenge'?$('#mp-ic-chp',app):$('#mp-ctl-p',app),m:$('#mp-ic-m',app)||$('#mp-ic-chm',app)||$('#mp-ctl-m',app),t:$('#mp-ic-t',app)||$('#mp-ic-cht',app)||$('#mp-ctl-t',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#mp-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#mp-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...PRESETS[0].s}; cueIfNew(); render();}); return;}
  if(a==='submit'){chSubmitted=true; const e=chEval(S.t,S.p,S.m); if(e&&e.fits) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(playing) pause(); else play(ph>=1); A.tip.quiet(); return;}
  if(a==='to-q'){const q=$('#mp-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#mp-rail .mp-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#mp-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#mp-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k!=='scrub') return; pause(); ph=(+e.target.value)/1000; if(ph>=1){finishRun(); revealSoonIf(); return;} renderPlay();});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#mp-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); if(s.id==='gpipe'){if(playing){/* the run plays at the learner's speed */}} revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#mp-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('1f1b')){const f=()=>byLearner(()=>setLv3(d.lever)); if(innerWidth<=760) holdPlace(b,f); else f(); return;}
    if(d.lever!=null&&isP('link')){const f=()=>byLearner(()=>setLv5(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='spd'){spd=v; lastT=0; drawPlayer(); return;}
      if(k==='m') bySettings(b,()=>setState({m:+v}));
      else if(k==='t') bySettings(b,()=>setState({t:+v}));
      else if(k==='p') bySettings(b,()=>setState({p:+v}));
      else if(k==='d') bySettings(b,()=>setState({d:+v}));
      else if(k==='model') bySettings(b,()=>setState({model:v}));
      else if(k==='seq') bySettings(b,()=>setState({seq:+v}));
      else if(k==='sched') bySettings(b,()=>setState({sched:v}));
      else if(k==='place') bySettings(b,()=>setState({place:v}));
      else if(k==='zero') bySettings(b,()=>setState({zero:+v}));
      else if(k==='cht'||k==='chp'||k==='chm'){const t=k==='cht'?+v:S.t, p=k==='chp'?+v:S.p; let m=k==='chm'?+v:S.m;
        if(!Number.isInteger(CH.G/(t*p))) return;
        m=chNearM(t,p,m);   /* the same rule the result's “the card then sets m = …” is computed with */
        byLearner(()=>{Object.assign(S,chCfg(t,p,m)); cueIfNew(); fillRail(); render();});}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s}),{badges:false}); requestAnimationFrame(()=>{if(app) revealBox($('#mp-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#mp-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the choices tried, the
   pictures already played to the end, the playing speed, the challenge's result, the drill. The playhead is not kept: a picture
   still playing is recorded as played to its end, a picture already played shows finished, and one not played yet plays from its
   start. */
const SK=Object.keys(BASE), okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), pickS=o=>Object.fromEntries(SK.map(k=>[k,o[k]]));
const cfgOf=x=>{const c=okObj(x)?parse(token(x)):null; return c?pickS(c):null;};
A.registerLab({
  id:ID,
  snap(){const e=mode==='challenge'&&chSubmitted?chEval(S.t,S.p,S.m):null;
    return {v:1,mode,step,S:pickS(S),pred,observed:{...observed},saved:saved?pickS(saved):null,lv3,tried3:[...tried3],lv5,tried5:[...tried5],mSeen:[...mSeen],tSeen:[...tSeen],spd,watched:[...new Set([...watched,...(playing?[simKey()]:[])])],
      chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,passed:!!(e&&e.fits&&chMinimal(e))};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[], inV=(L,v)=>L.some(x=>x.v===v);
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    if(mode==='challenge'){const x=okObj(o.S)?o.S:{}, c=CH.Ts.includes(x.t)&&CH.Ps.includes(x.p)&&CH.Ms.includes(x.m)?chCfg(x.t,x.p,x.m):null; S={...(c||chCfg(CH.start.t,CH.start.p,CH.start.m))};}
    else S={...(cfgOf(o.S)||BASE)};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    lv3=inV(LV3,o.lv3)?o.lv3:'gpipe'; tried3=new Set(['gpipe',...arr(o.tried3,v=>inV(LV3,v)),lv3]);
    lv5=inV(LV5,o.lv5)?o.lv5:'in'; tried5=new Set(['in',...arr(o.tried5,v=>inV(LV5,v)),lv5]);
    mSeen=new Set([1,...arr(o.mSeen,v=>[1,2,4,8,16,32,64,128,256].includes(v))]); tSeen=new Set([1,...arr(o.tSeen,v=>[1,2,4,8].includes(v))]);
    spd=['slow','mid','fast'].includes(o.spd)?o.spd:'mid';
    watched.clear(); arr(o.watched,v=>typeof v==='string'&&/^(gpipe|1f1b)\|\d{1,2}\|\d{1,3}$/.test(v)).slice(0,200).forEach(v=>watched.add(v));
    ph=0; playing=false;
    saved=mode==='challenge'?cfgOf(o.saved):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#mp-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#mp-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    if(!relang) lastKey='';
    cueIfNew();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); pause(); root=null; app=null;},
  onLang(){if(root){const d=$('#mp-math',app), open=d&&d.open, was=playing; pause(); relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#mp-math',app).open=true; if(was) play(false);}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='layer'?$('#mp-tp',app):(v==='cluster'?$('#mp-cl',app):$('#mp-gantt',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#mp-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      if(predHidden()){vis.innerHTML=''; read.innerHTML=esc(tx('8 GPUs, 10 layers each: answer the question first','8 张卡各放 10 层：先作答')); return;}
      if(v==='gantt'){const sm=simNow(), done=ph>=1, b=done?sm.idle:simAt(sm,ph*sm.T).idle; vis.innerHTML=`<div class="mp-dk"><i style="width:${((1-b)*100).toFixed(1)}%"></i></div>`;
        read.innerHTML=esc(tx(`p = ${S.p} stages, m = ${S.m}: ${done?'each GPU idle':'idle so far'} ${pct(b)}`,`p = ${S.p} 段，m = ${S.m}：${done?'每张卡空等':'到现在空等'} ${pct(b)}`)); return;}
      if(v==='layer'){const r=layer(S.t,S.place); vis.innerHTML=`<div class="mp-dk"><i style="width:${((1-r.share)*100).toFixed(1)}%"></i></div>`;
        read.innerHTML=esc(tx(`TP ${S.t}${S.t>1?(S.place==='out'?' across machines':' in one machine'):''}: all-reduces ${pct(r.share)} of a layer’s time`,`张量并行 ${S.t} 份${S.t>1?(S.place==='out'?'，跨机器':'，一台机器里'):''}：all-reduce 占一层时间的 ${pct(r.share)}`)); return;}
      const r=cost(S); vis.innerHTML=`<div class="mp-dk${r.fits?'':' bad'}"><i style="width:${Math.min(100,r.mem/GPU.mem*100).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(tx(`${fGB(r.mem)} per GPU, ${fS(r.step)} per step`,`每张卡 ${fGB(r.mem)}，每步 ${fS(r.step).replace(/ s$/,' 秒')}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__mp=Object.freeze({
  cost:c=>{const r=cost({...BASE,...c}); const o={...r}; delete o.md; return deep(o);}, layer:(t,pl)=>{const r=layer(t,pl); return deep({comp:r.comp,ar:r.ar,comm:r.comm,total:r.total,share:r.share,bytes:r.bytes,params:r.params});},
  sim:(p,m,s)=>{const r=sim(p,m,s); return deep({T:r.T,peak:r.peak,idle:r.idle,ops:r.ops});}, chEval:(t,p,m)=>{const e=chEval(t,p,m); return e?deep({t:e.t,p:e.p,m:e.m,d:e.d,b:e.b,fits:e.fits,step:e.step,mem:e.r.mem,tpShare:e.r.tpShare,bubble:e.r.bubble}):null;},
  chAll:()=>chAll().map(e=>deep({t:e.t,p:e.p,m:e.m,d:e.d,b:e.b,fits:e.fits,step:e.step,mem:e.r.mem,min:chMinimal(e)})), oneChange:(t,p,m)=>{const e=chEval(t,p,m); const c=e?chOneChange(e):null; return c?deep({k:c.k,v:c.v,t:c.e.t,p:c.e.p,m:c.e.m,step:c.e.step,mem:c.e.r.mem,fits:c.e.fits,min:chMinimal(c.e)}):null;},
  /* G7: the result's last sentence for a pick, as plain text in the current language ('' for a pick that meets the goal) */
  nextText:(t,p,m)=>{const e=chEval(t,p,m); return e?(chMinimal(e)?'':chNextHTML(e,chBest()).replace(/<[^>]+>/g,'')):null;}, drill:s=>{const d=makeDrill(s); return deep({p:d.p,m:d.m,t:d.t,place:d.place,bubble:d.bubble,gp:d.gp,f1:d.f1,comm:d.l.comm,share:d.l.share});},
  consts:Object.freeze({GPU:{...GPU},SHARE,SEQ,MD:deep(MD),CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),DUR:{...DUR}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},lv3,lv5,tried3:[...tried3],tried5:[...tried5],chSubmitted,chReveal,ph,playing,spd,watched:[...watched],stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),text:[...e.explain.flatMap(q=>[q.q,...q.o,...q.n.filter(Boolean),q.why]),...e.transfer.flatMap(q=>[q.q,...q.o,q.why])].join('\n')});}
});
})();
