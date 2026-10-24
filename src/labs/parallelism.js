/* parallelism (world training): How do you train a model that does not fit on one GPU? (ZeRO and FSDP)
   Fidelity: Analytical. Memory per GPU and the data moved are exact formulas; the step time is a two-stream schedule (one compute
   stream, one transfer stream, per layer) with stated assumptions:
   - model states, 18 bytes per parameter (Megatron-LM dist_optimizer.md, Hugging Face model_memory_anatomy.md): BF16 weights 2,
     FP32 gradients 4, FP32 copy of the weights 4, Adam's m 4, v 4. Per GPU, N parameters, d data-parallel GPUs: DP 18N; ZeRO-1
     6N + 12N/d; ZeRO-2 2N + 16N/d; ZeRO-3 18N/d; HSDP 18N/8 (ZeRO-3 inside each 8-GPU machine, a copy per machine). ZeRO-3 and HSDP
     also hold the full BF16 weights of the layers in flight: 2 average layers (2N/L bytes each) with prefetching, 1 without;
   - activations, full recomputation: each layer keeps its input (2h bytes per token) and the layer being recomputed
     a = 2 × (6h + 2·kv) + 8i bytes per token (41h for 8B, 40.5h for 70B): tokens × (2hL + a). Logits in FP32, tokens × V × 4 bytes,
     as training-memory counts them;
   - the bytes each GPU sends, ring collectives (NCCL PERFORMANCE.md): all-reduce 2(d − 1)/d of the data, reduce-scatter and all-gather
     (d − 1)/d; every transfer in BF16, 2 bytes per value (integrator decision 10, as the ZeRO paper assumes): DP and ZeRO-1/2 2N
     values per step, ZeRO-3 3N;
   - transfer time (integrator decision 8, the scaling book's gpus.md): inside one machine, each GPU's bytes ÷ 450 GB/s (NVLink);
     a group that spans machines moves the whole array (all its shards together) over each machine's network, an all-gather or
     reduce-scatter in bytes ÷ the machine's bandwidth, an all-reduce in twice that. Per machine: 400 GB/s (8 × 400 Gb/s, one NIC
     per GPU), 50 GB/s (a single 400 Gb/s NIC), 12.5 GB/s (100 Gb/s Ethernet);
   - compute (integrator decisions 1 and 9) = 8N × tokens ÷ (989.5 TFLOP/s × the share of the peak while computing, 50% by default):
     full recomputation adds one forward pass; forward a quarter, backward with the recomputation three quarters, evenly over the
     layers. MFU is a result and counts 6N: 6N × tokens ÷ (step × 989.5 TFLOP/s).
   Words (EN / ZH): shard / 分片, model states / 模型状态, waiting / 等数据, part / 部分.
   window.__zr is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='parallelism';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2').replace(/([一-鿿]) +(?=[一-鿿（])/g,'$1');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const NB='‑';
const MOD={
 '8b':{id:'8b',name:'Llama'+NB+'3.1'+NB+'8B',P:8030261248,L:32,h:4096,i:14336,kv:1024,V:128256},
 '70b':{id:'70b',name:'Llama'+NB+'3.1'+NB+'70B',P:70553706496,L:80,h:8192,i:28672,kv:1024,V:128256}
};
const PEAK=989.5e12, CAP=80e9, GPN=8;
const BW_IN=450e9;                                   // NVLink inside one H100 machine, per GPU (scaling book, gpus.md)
/* the network of one 8-GPU machine, shared by its 8 GPUs (integrator decision 8): one 400 Gb/s NIC per GPU; a single 400 Gb/s NIC;
   one 100 Gb/s Ethernet link */
const NET={ib:400e9,one:50e9,eth:12.5e9};
const NETS=['ib','one','eth'];
const STS=['dp','z1','z2','z3','hsdp'];
const NS=[8,16,32,64];
const TS=[1024,2048,4096,8192,16384,32768,65536,131072];
const US=[30,35,40,45,50,55,60,65,70,75,80];         // the share of the peak while computing, % (integrator decision 1: 50% default)
const BASE={m:'8b',N:8,st:'dp',T:4096,ov:1,net:'ib',u:50};
const FPT=8, MFPT=6;                                  // FLOPs per parameter per token: done with full recomputation 8; MFU counts 6
const ring=n=>(n-1)/n;
function mem(c){
  const m=MOD[c.m], P=m.P, g=Math.min(c.N,GPN);
  const sh={dp:[1,1,1],z1:[1,1,c.N],z2:[1,c.N,c.N],z3:[c.N,c.N,c.N],hsdp:[g,g,g]}[c.st];
  const w=2*P/sh[0], gr=4*P/sh[1], cp=4*P/sh[2], mm=4*P/sh[2], vv=4*P/sh[2], o=cp+mm+vv;
  const inf=(c.st==='z3'||c.st==='hsdp')?(c.ov?2:1)*2*P/m.L:0;
  /* the layer being recomputed, BF16, per token: x, normed x, q, k, v, attention output, x2, normed x2, then gate, SiLU, up, product */
  const aL=2*(6*m.h+2*m.kv)+8*m.i, perTok=2*m.h*m.L+aL, act=c.T*perTok;
  const lg=c.T*m.V*4;                                // the logits in FP32, as training-memory counts them
  const states=w+gr+o, total=states+inf+act+lg;
  return {w,gr,cp,mm,vv,o,inf,act,lg,aL,perTok,states,total,fits:total<=CAP,over:Math.max(0,total-CAP),sh};
}
/* the bytes each GPU sends in one step, and the time of each transfer */
function comm(c){
  const m=MOD[c.m], P=m.P, one=c.N<=GPN, g=Math.min(c.N,GPN), R=Math.max(1,c.N/GPN), W=NET[c.net], lay=2*P/m.L;
  let ag=0,agb=0,rs=0,ar=0,arx=0;
  if(c.st==='dp'){ar=2*lay*ring(c.N);}
  else if(c.st==='z1'||c.st==='z2'){ag=lay*ring(c.N); rs=lay*ring(c.N);}
  else if(c.st==='z3'){ag=lay*ring(c.N); agb=ag; rs=ag;}
  else {ag=lay*ring(g); agb=ag; rs=ag; arx=R>1?2*(lay/g)*ring(R):0;}
  const per={ag,agb,rs,ar,arx}, L0=m.L, bytes=L0*(ag+agb+rs+ar+arx), dpBytes=2*2*P*ring(c.N);
  /* per layer: inside one machine, each GPU's bytes over NVLink; a group that spans machines moves the whole layer over each
     machine's network, an all-reduce twice (decision 8); HSDP keeps its all-gathers and reduce-scatters inside the machine */
  const inM=one||c.st==='hsdp', x=(v,k)=>v?(inM?v/BW_IN:k*lay/W):0;
  const t={ag:x(ag,1),agb:x(agb,1),rs:x(rs,1),ar:x(ar,2),arx:arx?2*lay/W:0};
  const time=L0*(t.ag+t.agb+t.rs+t.ar+t.arx);
  /* what each machine moves over the network in one step, by the same rule (an all-reduce counts the array twice) */
  const netB=one?0:L0*lay*(c.st==='z3'?3:2), netT=one?0:netB/W;
  return {per,t,bytes,dpBytes,ratio:bytes/dpBytes,time,netB,netT,W,one};
}
/* one training step on one GPU: a compute stream and a transfer stream, layer by layer */
function sim(c){
  const m=MOD[c.m], Lr=m.L, P=m.P, C=FPT*P*c.T/(PEAK*c.u/100), tF=C/4/Lr, tB=3*C/4/Lr, k=comm(c), t=k.t;
  const ev=[]; let tc=0, tn=0;
  const put=(r,kind,i,s,d)=>{ev.push({r,k:kind,i,t0:s,t1:s+d}); return s+d;};
  const fwdAG=c.st!=='dp', bwdAG=c.st==='z3'||c.st==='hsdp';
  const red=(i,s)=>{if(c.st==='dp') return put('n','AR',i,s,t.ar); let e=put('n','RS',i,s,t.rs); if(t.arx>0) e=put('n','ARx',i,e,t.arx); return e;};
  const doneB={};
  if(!fwdAG){for(let i=0;i<Lr;i++) tc=put('c','F',i,tc,tF);}
  else if(c.ov){
    const done={}; tn=put('n','AG',0,0,t.ag); done[0]=tn;
    for(let i=0;i<Lr;i++){const s=Math.max(tc,done[i]);
      if(i+1<Lr){tn=put('n','AG',i+1,Math.max(tn,s),t.ag); done[i+1]=tn;}
      else if(bwdAG){tn=put('n','AGb',Lr-1,Math.max(tn,s),t.agb); doneB[Lr-1]=tn;}
      tc=put('c','F',i,s,tF);}
  } else {
    for(let i=0;i<Lr;i++){const s=Math.max(tc,tn), e=put('n','AG',i,s,t.ag); tc=put('c','F',i,e,tF); tn=tc;}
  }
  if(c.ov){
    if(bwdAG&&doneB[Lr-1]==null){tn=put('n','AGb',Lr-1,Math.max(tn,tc),t.agb); doneB[Lr-1]=tn;}
    for(let i=Lr-1;i>=0;i--){
      const s=bwdAG?Math.max(tc,doneB[i]):tc;
      if(bwdAG&&i>0){tn=put('n','AGb',i-1,Math.max(tn,s),t.agb); doneB[i-1]=tn;}
      tc=put('c','B',i,s,tB);
      tn=red(i,Math.max(tn,tc));
    }
  } else {
    for(let i=Lr-1;i>=0;i--){
      let s=Math.max(tc,tn); if(bwdAG) s=put('n','AGb',i,s,t.agb);
      tc=put('c','B',i,s,tB); tc=tn=red(i,tc);
    }
  }
  const step=Math.max(tc,tn), wait=Math.max(0,step-C);
  /* the gaps in the compute row, where this GPU waits for data */
  const cs=ev.filter(e=>e.r==='c').sort((a,b)=>a.t0-b.t0), gaps=[]; let x=0;
  cs.forEach(e=>{if(e.t0-x>step*1e-6) gaps.push({t0:x,t1:e.t0}); x=Math.max(x,e.t1);});
  if(step-x>step*1e-6) gaps.push({t0:x,t1:step});
  /* MFU, a result: the model's own 6N FLOPs per token over the GPU's peak in the same time (integrator decisions 2 and 9) */
  const mfu=step>0?MFPT*P*c.T/(step*PEAK):0;
  return {C,tF,tB,step,wait,share:step>0?wait/step:0,mfu,ev,gaps,fwdEnd:cs.filter(e=>e.k==='F').reduce((a,e)=>Math.max(a,e.t1),0),k};
}
function evalC(c){const me=mem(c), s=sim(c); return {c:{...c},me,s,k:s.k,ok:me.fits&&s.share<=CH.wait+1e-12};}

/* the challenge: Llama-3.1-70B on 64 GPUs in 8 machines with one 400 Gb/s NIC each; pick the sharding and the tokens per GPU per step */
const CH={m:'70b',N:64,net:'one',ov:1,u:50,sts:['z1','z2','z3','hsdp'],Ts:[2048,4096,8192,16384,32768,65536],wait:0.10,start:{st:'z2',T:4096}};
const chCfg=(st,T)=>({m:CH.m,N:CH.N,st,T,ov:CH.ov,net:CH.net,u:CH.u});
function chAll(){const a=[]; CH.sts.forEach(st=>CH.Ts.forEach(T=>a.push(evalC(chCfg(st,T))))); return a;}
/* among the passing options, the shortest step (R4.5) */
function chBest(){return chAll().filter(e=>e.ok).sort((a,b)=>a.s.step-b.s.step)[0];}
/* the one change that helps most (the owner's review of 2026-10-07, R1): of the options that differ from the reader's in exactly one
   setting, the passing one with the shortest step; if none passes and the reader's does not fit, the single change that fits with
   the least waiting (pass:false, so the result says what it still misses); otherwise null */
function chOne(st,T){
  const me=evalC(chCfg(st,T)), one=chAll().filter(e=>(e.c.st===st)!==(e.c.T===T));
  const pass=one.filter(e=>e.ok).sort((a,b)=>a.s.step-b.s.step);
  if(pass.length) return {e:pass[0],pass:true};
  const fit=one.filter(e=>e.me.fits).sort((a,b)=>a.s.share-b.s.share);
  return !me.me.fits&&fit.length?{e:fit[0],pass:false}:null;
}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fGB=b=>fixd(b/1e9,1)+' GB';
const fG=b=>fixd(b/1e9,1);
const fSec=t=>{const v=t<1?fixd(t,2):(t<10?fixd(t,1):fixd(t,0)); return tx(v+' s',v+' 秒');};
const fPctW=s=>s<0.01?tx('under 1%','不到 1%'):fixd(s*100,0)+'%';
const fX=r=>fixd(r,1);
const fTok=T=>fint(T);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const labLinkZ=id=>{const h=labLink(id); return /^[A-Za-z0-9]/.test(h.replace(/<[^>]+>/g,''))?' '+h:h;};   // after a Chinese character
const M=t=>`<div class="zr-fx">${A.ui.math(t)}</div>`;
/* every formula carries its name on the left, in words (the owner's review of 2026-10-07: “6N+12N/d 是什么含义，是显存占用量吗，
   为什么没有名字”) */
const MS=()=>A.lang()==='zh'?'\\text{每张卡存模型状态的字节数}':'\\text{bytes of model states per GPU}';
/* a named formula: the name and its sign, then the right-hand side; the right-hand side moves under the name when the card is too
   narrow for both, instead of scrolling */
const MN=(name,rhs,rel)=>`<div class="zr-fx zr-fxn">${A.ui.math(name+' '+(rel||'='))}${A.ui.math(rhs)}</div>`;
/* the reader's numbers under a named formula (H2, 2026-10-09): its own line, not a .zr-fx (those carry a name on the left) */
const MV=t=>`<div class="zr-math-line zr-numl">${A.ui.math(t)}</div>`;
const big=n=>A.ui.big?A.ui.big(n):(A.fmt.big?A.fmt.big(n):fint(n));
const stName=(st,N)=>({dp:tx('Data parallel','数据并行'),z1:'ZeRO-1',z2:'ZeRO-2',z3:'ZeRO-3',hsdp:tx('HSDP','HSDP')})[st];
const stLong=(st,N)=>{const n=N||8;
  return {dp:tx('Data parallel (a full copy on every GPU)','数据并行（每张卡一整份）'),
    z1:tx(`ZeRO-1 (optimizer state split over ${n} GPUs)`,`ZeRO-1（优化器状态分到 ${n} 张卡上）`),
    z2:tx(`ZeRO-2 (gradients split too)`,`ZeRO-2（梯度也分）`),
    z3:tx(`ZeRO-3 (weights split too, over ${n} GPUs)`,`ZeRO-3（权重也分到 ${n} 张卡上）`),
    hsdp:tx('HSDP (ZeRO-3 inside each machine, a copy per machine)','HSDP（机器内 ZeRO-3，每台机器一整份）')}[st];};
const netName=n=>({ib:tx('400 GB/s per machine','每台机器 400 GB/s'),one:tx('50 GB/s per machine','每台机器 50 GB/s'),eth:tx('12.5 GB/s per machine (Ethernet)','每台机器 12.5 GB/s（以太网）')})[n];

/* =====================================================================
   module state
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lv5='noov', lv5Tried=new Set(['noov']), lv6='z3', lv6Tried=new Set(['z3']), remView='dp';
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, relang=false;
let drillSeed=1, drillOpen=false;
/* the timeline's player: one step, a mechanism replay, Medium by default (R3.9) */
let pT=0, playing=false, spd='mid', raf=0, lastT=0, curSig='';
/* R3.20: a timeline the reader has already watched to the end (in any part, scenario, the challenge, or on a return) shows its end
   at once; only a new, different timeline plays, and ▶ Replay replays on demand */
const watched=new Set();                 // timelines already played to the end (R3.20)
const DUR={slow:16000,mid:8000,fast:4000};   // B6: Medium within a few seconds (was 24 / 12 / 5 s)
const shownPrev={};
const P1={...BASE}, P2={...BASE,st:'dp'}, P3={...BASE,st:'z1'}, P4={...BASE,st:'z2'}, P5={...BASE,st:'z3',ov:0}, P6={...BASE,N:64,st:'z3',net:'one'};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 states:()=>tx('Model states are everything training keeps for each parameter, 18 bytes in all: the BF16 weights (2 bytes), the FP32 gradients (4), an FP32 copy of the weights (4) and Adam’s m and v (4 each). The FP32 copy, m and v together are the optimizer state, 12 bytes per parameter.',
               '模型状态是训练时每个参数都要存的东西，一共 18 字节：BF16 权重 2 字节，FP32 梯度 4 字节，FP32 权重副本 4 字节，Adam 的 m 和 v 各 4 字节。FP32 权重副本、m 和 v 合起来叫优化器状态，每个参数 12 字节。'),
 dp:()=>tx('Data parallelism: every GPU runs the whole model on a different part of the batch and computes the gradients of all parameters. The GPUs then average their gradients, and every GPU applies the same update.',
           '数据并行：每张卡用整个模型处理这一批数据里不同的一部分，算出全部参数的梯度。然后各卡把梯度取平均，每张卡做同样的一次更新。'),
 act:()=>tx('Activations are the intermediate results the forward pass keeps for the backward pass; they grow with the tokens each GPU processes per step. This lab assumes full recomputation. In the forward pass each layer keeps only its input. In the backward pass, each layer first redoes its own forward pass.',
            '激活是前向时留下、反向要用的中间结果，它随每张卡每步处理的 token 数增长。这个 Lab 假设开了重算。前向时，每层只留下自己的输入。反向算到哪一层，就把这一层的前向重算一遍。'),
 lg:()=>tx('The logits are the model’s last output. Each token gets one row of 128,256 values, one for each word in Llama 3’s vocabulary. As in the lab “Training memory”, they are kept in FP32, 4 bytes each, so they too grow with the tokens each GPU processes.',
           'logits 是模型最后的输出。每个 token 对应一行，一行有 128,256 个值，Llama 3 词表里的每个词各占一个。和“训练显存”那个 Lab 一样，logits 按 FP32 存，每个值 4 字节，所以也随每张卡处理的 token 数增长。'),
 ar:()=>tx('All-reduce: every GPU hands in its own copy of some data (here, its gradients), and at the end every GPU holds the sum (or average) over all GPUs. In a ring of d GPUs, each GPU sends '+A.ui.math('\\frac{2(d - 1)}{d}')+' times the data’s size.',
           'all-reduce：每张卡交出自己的一份数据（这里是梯度），结束时每张卡都拿到所有卡加起来的结果（或平均值）。d 张卡用环形实现时，每张卡要发出数据大小的 '+A.ui.math('\\frac{2(d - 1)}{d}')+' 倍。'),
 rs:()=>tx('Reduce-scatter: the data is cut into d pieces, one for each GPU. At the end each GPU holds only the sum over all GPUs of its own piece. Each GPU sends '+A.ui.math('\\frac{d - 1}{d}')+' of the data, half of an all-reduce.',
           'reduce-scatter：数据切成 d 份，每张卡负责其中一份。结束时，每张卡只拿到自己那一份在所有卡上的加和。每张卡发出的数据是总量的 '+A.ui.math('\\frac{d - 1}{d}')+'，是 all-reduce 的一半。'),
 ag:()=>tx('All-gather: each GPU hands in its own piece, and at the end every GPU holds all d pieces. Each GPU sends '+A.ui.math('\\frac{d - 1}{d}')+' of the full data. An all-reduce is a reduce-scatter followed by an all-gather.',
           'all-gather：每张卡交出自己的那一份，结束时每张卡都有全部 d 份。每张卡发出完整数据的 '+A.ui.math('\\frac{d - 1}{d}')+'。一次 all-reduce 就是一次 reduce-scatter 接一次 all-gather。'),
 shard:()=>tx('To shard is to split something into equal pieces and keep one piece on each GPU, instead of a full copy on each.','分片：把一样东西平均切成几份，每张卡只存其中一份，而不是每张卡各存一整份。'),
 nvlink:()=>tx('NVLink connects the GPUs inside one machine, much faster than the network between machines. This lab uses 450 GB/s per GPU, the figure NVIDIA gives for an H100 machine; the scaling book notes that in practice it is hard to exceed 370 GB/s.','NVLink 是同一台机器里 GPU 之间的高速连接，比机器之间的网络快得多。这个 Lab 按每张卡 450 GB/s 算，这是 NVIDIA 给 H100 机器标的数。scaling book 也提到，实际很难超过 370 GB/s。'),
 net:()=>tx('Between machines the data goes over the network, and the 8 GPUs of a machine share its bandwidth. Large H100 clusters usually give every GPU its own 400 Gb/s NIC, 400 GB/s per machine in all. torchtitan’s test cluster also has 400 Gb/s per GPU between hosts.',
            '机器之间的数据走网络，一台机器的 8 张卡共用这台机器的网络带宽。大的 H100 集群通常每张卡配一块 400 Gb/s 的网卡，一台机器一共 400 GB/s。torchtitan 测试用的集群，机器之间也是每张卡 400 Gb/s。'),
 pre:()=>tx('Prefetching: while a layer computes, the GPU already fetches the next layer’s weights, so they are there when the computation reaches it. It holds two layers’ full weights at a time instead of one.','预取：一层还在计算时，就先把下一层的权重取过来，等计算走到下一层时，权重已经到了。代价是同时要放两层的完整权重，而不是一层。'),
 hsdp:()=>tx('HSDP (hybrid sharded data parallel): ZeRO-3 inside each machine, over NVLink, and a full copy of the model states in every machine. Between machines only the gradients are all-reduced, once per step, each GPU taking its own 1/8. PyTorch FSDP calls it HYBRID_SHARD.',
             'HSDP（混合分片数据并行）：每台机器内部做 ZeRO-3，走 NVLink，每台机器各存一整份模型状态。机器之间只对梯度做一次 all-reduce，每张卡负责其中的 1/8。PyTorch FSDP 里叫 HYBRID_SHARD。'),
 /* integrator decision 1: one name and one default for the assumption */
 share:()=>tx('An assumed value: how much of the H100’s peak, 989.5 TFLOP/s, a GPU reaches while it is computing. Plain BF16 matrix multiplies reach about 80% (measured by ml-engineering). Attention, normalization and smaller kernels do not reach that, so this lab uses 50% by default.',
              '这是假设值，指 GPU 在算的时候能跑到 H100 峰值 989.5 TFLOP/s 的多少。单纯的 BF16 矩阵乘法约能到 80%（ml-engineering 实测）。attention、归一化和较小的 kernel 达不到这么高，所以这个 Lab 默认取 50%。'),
 /* integrator decisions 2 and 9: MFU only as a result; it counts 6N, while the GPU does 8N with full recomputation */
 mfu:()=>{const top=fixd(S.u*MFPT/FPT,1);
   return tx(`MFU: the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time. Time spent waiting for data counts. This lab recomputes, so a GPU really does 8 × parameters FLOPs per token, and the extra forward pass does not count. So even with no waiting, the MFU here is 3/4 of the ${S.u}%, ${top}%. Measured: torchtitan 33% to 39% (Llama 3.1 8B with FSDP on 8 or 128 H100s); Megatron-LM’s README 41% to 47% on H100.`,
             `MFU：模型真正需要的运算（每个 token 6 × 参数量次）占同一段时间里 GPU 峰值算力的比例。等数据的时间也算在里面。这个 Lab 开了重算，每个 token 实际要算 8 × 参数量次，多出的那次前向不算进 MFU。所以就算一点不等数据，这里的 MFU 也只有 ${S.u}% 的 3/4，也就是 ${top}%。实测值：torchtitan 是 33% 到 39%（Llama 3.1 8B，FSDP，8 张或 128 张 H100），Megatron-LM 的 README 在 H100 上是 41% 到 47%。`);}
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'q18',t:{en:'About 18 GB: the 8 GPUs split it evenly',zh:'约 18 GB，8 张卡平分'}},
            {v:'mid',t:{en:'Somewhere in between: part of it is saved',zh:'介于两者之间，省掉一部分'}},
            {v:'q145',t:{en:'Still about 145 GB: each GPU keeps a full copy',zh:'还是约 145 GB，每张卡各存一整份'}}];
const PRED_RIGHT='q145';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const segZ=(id,key,opts,val)=>A.ui.seg({id,key,label:'',options:opts,value:val,inline:true});
const LV5=[{v:'noov',t:{en:'Fetch when needed (initial setup)',zh:'用到才取（初始设置）'},d:{en:'A layer’s weights are fetched only when the computation reaches it; the GPU waits meanwhile',zh:'计算走到哪一层，才去取这一层的权重，取的时候 GPU 只能等'}},
           {v:'ov',t:{en:'Prefetch the next layer',zh:'提前取下一层（预取）'},d:{en:'While a layer computes, the next layer’s weights are already on their way',zh:'一层还在计算时，下一层的权重已经在路上'}}];
const LV6=[{v:'z3',t:{en:'ZeRO-3 over all 64 GPUs, one NIC per machine (initial setup)',zh:'ZeRO-3 分到 64 张卡，每台机器一块网卡（初始设置）'},d:{en:'4,096 tokens per GPU per step; 50 GB/s per machine',zh:'每张卡每步 4,096 个 token，每台机器 50 GB/s'}},
           {v:'tok',t:{en:'More tokens per GPU per step',zh:'每张卡每步多处理一些 token'},d:{en:'16,384 instead of 4,096: four times the computation, the same data to move',zh:'从 4,096 加到 16,384，计算变成 4 倍，要传的数据不变'}},
           {v:'net',t:{en:'One NIC per GPU',zh:'每张卡配一块网卡'},d:{en:'8 NICs of 400 Gb/s: 400 GB/s per machine, as large H100 clusters usually have',zh:'8 块 400 Gb/s 的网卡，每台机器 400 GB/s，大的 H100 集群通常这样配'}}];
const lv6Cfg=v=>v==='tok'?{...P6,T:16384}:(v==='net'?{...P6,net:'ib'}:{...P6});
const lvRes5=v=>{const s=sim({...P5,ov:v==='ov'?1:0}); return tx(`waits ${fPctW(s.share)}`,`等数据 ${fPctW(s.share)}`);};
const lvRes6=v=>{const e=evalC(lv6Cfg(v)); return tx(`waits ${fPctW(e.s.share)}`,`等数据 ${fPctW(e.s.share)}`);};
const lv5Ctl=()=>`<div class="zr-lv" id="zr-ic-lv5">`+A.ui.levers({aria:tx('When to fetch a layer’s weights, pick one','什么时候取一层的权重，选一种'),value:lv5,options:LV5.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(lv5Tried.has(o.v)||o.v===lv5?lvRes5(o.v):'')}))})+`</div>`;
const lv6Ctl=()=>`<div class="zr-lv" id="zr-ic-lv6">`+A.ui.levers({aria:tx('Three ways, pick one','三种做法，选一种'),value:lv6,options:LV6.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(lv6Tried.has(o.v)||o.v===lv6?lvRes6(o.v):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},gate:()=>!!pred,skip:true,
  lead:()=>tx('Data parallel on 8 GPUs: how much does each GPU hold?','8 张卡做数据并行，每张卡要存多少？'),
  enter:()=>({...P1}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You are training ${MOD['8b'].name}. Training keeps 18 bytes of ${DEF('model states',TIP.states())} for every parameter: the BF16 weights 2 bytes, the FP32 gradients 4 bytes, the optimizer state 12 bytes. ${MOD['8b'].name} has 8.03B parameters, about 145 GB of model states, and one H100 has 80 GB.</p><p>You have one machine with 8 H100s and use ${DEF('data parallelism',TIP.dp())}. Each GPU takes a different part of the data and computes its own gradients. Then the 8 GPUs average their gradients, and each GPU updates its parameters with the average.</p><p>How much does each GPU have to hold?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the result.</p>':''}`,
             `<p>你要训练 ${MOD['8b'].name}。训练时，每个参数要存 18 字节的${DEF('模型状态',TIP.states())}：BF16 权重 2 字节，FP32 梯度 4 字节，优化器状态 12 字节。${MOD['8b'].name} 有 8.03B 个参数，模型状态一共约 145 GB，而一张 H100 只有 80 GB。</p><p>你有一台装了 8 张 H100 的机器，打算用${DEF('数据并行',TIP.dp())}。每张卡拿不同的数据，各自算出梯度。然后 8 张卡把梯度取平均，每张卡都用这个平均梯度更新自己的参数。</p><p>这样做，每张卡要存多少？</p>${ch}${pred?'<p class="muted small">已记下。第 2 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'dp',title:{en:'Data parallel: a full copy on each GPU',zh:'数据并行：每张卡一整份'},expect:()=>watched.has(sigOf(P2)),showMe:()=>finishPlay(),label:{en:'Result',zh:'结果'},
  lead:()=>tx('8 GPUs hold 8 identical copies of the model states.','8 张卡存了 8 份一模一样的模型状态。'),
  enter:()=>({...P2}),
  body:()=>tx(`<p>Each GPU runs the whole forward and backward pass on its own data, so each needs all the weights. It computes the gradients of every parameter, and its update needs the whole optimizer state.</p><p>The 8 GPUs average their gradients with an ${DEF('all-reduce',TIP.ar())}: every GPU hands in its gradients, and at the end every GPU holds the average. It does not wait for the whole backward pass. As soon as a layer’s backward pass is done, that layer’s gradients start moving, while the GPU goes on with the other layers.</p>`,
              `<p>每张卡都要用自己的数据把前向、反向完整跑一遍，所以每张卡都要有全部权重。它算出的是全部参数的梯度，更新时也要用到全部优化器状态。</p><p>8 张卡要把梯度取平均，用的是 ${DEF('all-reduce',TIP.ar())}：每张卡交出自己的梯度，结束时每张卡都拿到平均值。它不用等整个反向算完。一层的反向一算完，这一层的梯度就开始传，GPU 同时接着算别的层。</p>`),
  obs:()=>obsDP()},
 {id:'z1',title:{en:'ZeRO-1: split the optimizer state',zh:'ZeRO-1：把优化器状态分开存'},expect:()=>S.st==='z1',showMe:{st:'z1'},label:{en:'Result',zh:'结果'},
  lead:()=>tx('12 of the 18 bytes are the optimizer state.','18 字节里有 12 字节是优化器状态。'),
  enter:()=>({...P2}),
  body:()=>tx(`<p>The 8 copies are identical, so ZeRO’s idea is direct: each GPU keeps only 1/8 and gets the rest from the others when it needs it. Start with the optimizer state, 12 of the 18 bytes, used only in the update.</p><p>ZeRO-1 ${DEF('shards',TIP.shard())} the optimizer state 8 ways. Each GPU keeps one piece and updates only the parameters that piece belongs to.</p><p>So each GPU needs only the averaged gradients of its own 1/8, and the gradients’ all-reduce becomes a ${DEF('reduce-scatter',TIP.rs())}. After the update, before the next forward pass reaches a layer, an ${DEF('all-gather',TIP.ag())} brings every GPU that layer’s new weights.</p>`,
              `<p>8 张卡上的东西一模一样，所以 ZeRO 的想法很直接：每张卡只存 1/8，要用的时候再从别的卡取。先从优化器状态下手，它占 18 字节里的 12 字节，而且只在更新参数时才用到。</p><p>ZeRO-1 把优化器状态${DEF('分片',TIP.shard())}存，分成 8 份。每张卡只存一份，也只更新这一份对应的参数。</p><p>所以每张卡只需要自己那 1/8 参数的平均梯度，梯度的 all-reduce 就换成了 ${DEF('reduce-scatter',TIP.rs())}。更新完，下一步的前向算到某一层之前，再用 ${DEF('all-gather',TIP.ag())} 把这一层的新权重取齐。</p>`)+
   `<div class="zr-ic">`+segZ('zr-ic-z1','st13',[['dp',tx('Full copy on each GPU','每张卡一整份')],['z1',tx('Split 8 ways (ZeRO-1)','分成 8 份（ZeRO-1）')]],S.st)+`</div>`,
  obs:()=>obsZ1()},
 {id:'z2',title:{en:'ZeRO-2: split the gradients too',zh:'ZeRO-2：梯度也分开存'},expect:()=>S.st==='z2',showMe:{st:'z2'},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Each GPU keeps only the gradients of its own 1/8.','每张卡只留自己那 1/8 参数的梯度。'),
  enter:()=>({...P3}),
  body:()=>tx(`<p>After the reduce-scatter, each GPU uses only the averaged gradients of its own 1/8 of the parameters. The other 7/8 sit there unused. ZeRO-2 frees them: once a layer’s gradients have been sent, each GPU keeps only its own share.</p>`,
              `<p>reduce-scatter 之后，每张卡只用得上自己那 1/8 参数的平均梯度，另外 7/8 的梯度放着没用。ZeRO-2 把它们也释放掉：一层的梯度传完，每张卡只留自己那一份。</p>`)+
   `<div class="zr-ic">`+segZ('zr-ic-z2','st24',[['z1',tx('Full gradients (ZeRO-1)','完整的梯度（ZeRO-1）')],['z2',tx('Split 8 ways (ZeRO-2)','分成 8 份（ZeRO-2）')]],S.st)+`</div>`,
  obs:()=>obsZ2()},
 {id:'z3',title:{en:'ZeRO-3 (FSDP): split the weights too',zh:'ZeRO-3（FSDP）：权重也分开存'},expect:()=>lv5Tried.has('ov'),showMe:()=>setLv5('ov'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Down to 1/8, at the price of half again as much data to move.','省到 1/8，代价是多传一半的数据。'),
  enter:()=>{lv5='noov'; lv5Tried=new Set(['noov']); return {...P5};},
  body:()=>tx(`<p>After ZeRO-2 each GPU still holds a full copy of the BF16 weights. ZeRO-3 splits them 8 ways too. Before a layer computes, an all-gather collects its weights from all GPUs; they are freed right after, and the backward pass gathers them again. PyTorch’s FSDP (Fully Sharded Data Parallel) does this by default.</p><p>The price is more traffic, all of it in BF16. Data parallelism’s all-reduce moves about two copies of the gradients; ZeRO-3 moves three (each layer’s weights twice, the gradients once), 1.5 times as much. A layer cannot compute while its weights are on their way. When should a layer’s weights be fetched? Pick one and see how much of the time the GPU waits for data.</p>`,
              `<p>ZeRO-2 之后，每张卡还存着一整份 BF16 权重。ZeRO-3 把权重也分成 8 份：一层要算之前，先用 all-gather 从所有卡取齐这一层的权重，算完马上释放，反向时再取一次。PyTorch 的 FSDP（完全分片的数据并行）默认就这样做。</p><p>代价是多传数据，都按 BF16 传。数据并行的 all-reduce 相当于把梯度传两遍；ZeRO-3 每层的权重要取两遍，梯度再传一遍，一共三遍，是数据并行的 1.5 倍。权重还在路上时，这一层没法算。什么时候去取这一层的权重？选一种取法，看看 GPU 有多少时间在等数据。</p>`)+lv5Ctl(),
  obs:()=>obsZ3()},
 {id:'net',title:{en:'Across machines',zh:'跨机器'},expect:()=>lv6Tried.size>=2,showMe:()=>setLv6('net'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('When each machine’s network is slow, ZeRO-3 across machines waits.','每台机器的网络慢时，ZeRO-3 跨机器要等数据。'),
  enter:()=>{lv6='z3'; lv6Tried=new Set(['z3']); return {...P6};},
  body:()=>tx(`<p>The same 8B, now on 8 machines, 64 GPUs. Inside a machine the GPUs talk over ${DEF('NVLink',TIP.nvlink())}, 450 GB/s per GPU. Between machines the data goes over the ${DEF('network',TIP.net())}, and the 8 GPUs of a machine share its bandwidth.</p><p>The 8 GPUs need the same layer’s weights. So the data enters the machine once, and NVLink hands it to all 8. An all-gather across machines takes about</p>`,
              `<p>还是这个 8B，换成 8 台机器、64 张卡。机器内的卡之间走 ${DEF('NVLink',TIP.nvlink())}，每张卡 450 GB/s。机器之间走${DEF('网络',TIP.net())}，一台机器的 8 张卡共用这台机器的网络带宽。</p><p>8 张卡要取的是同一层的权重。这一层的数据只要进这台机器一次，再用 NVLink 分给 8 张卡。所以跨机器的一次 all-gather，大约要花这么长时间：</p>`)+
   (A.lang()==='zh'?MN('\\text{一次 all-gather 的时间}','\\frac{\\text{这一层的字节数}}{\\text{一台机器的网络带宽}}','\\approx'):MN('\\text{time of one all-gather}','\\frac{\\text{bytes of the layer}}{\\text{network bandwidth of one machine}}','\\approx'))+
   tx(`<p>These 8 machines have a slow network: one 400 Gb/s NIC each, 50 GB/s. Pick one of the three ways and see how much of the time the GPU waits for data. The first is the setup now; the other two are ways to improve it.</p>`,
      `<p>这 8 台机器的网络不快，每台只有一块 400 Gb/s 的网卡，也就是 50 GB/s。选一种做法，看看 GPU 有多少时间在等数据。第一种就是现在这样，另外两种是改进的办法。</p>`)+lv6Ctl(),
  obs:()=>obsNet()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Sharding saves memory and costs transfers. The transfers are free only while the computation covers them.','分片省显存，但要多传数据。只有计算盖得住时，多传的才不花时间。'),
  enter:()=>{remView='dp'; return {...P2};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
/* “this GPU waits 45% of the step” / “this GPU hardly waits for data” */
const waitSay=s=>s<0.01?tx('this GPU hardly waits for data','这张卡几乎不用等数据'):tx(`this GPU waits for data ${fPctW(s)} of the step`,`这张卡有 ${fPctW(s)} 的时间在等数据`);
function obsDP(){
  if(!watched.has(sigOf(P2))) return '';
  const e=evalC(P2);
  return tx(`<p><b>Each GPU needs ${fGB(e.me.total)}, and the model states alone are ${fGB(e.me.states)}. That is over 80 GB, so it does not fit.</b></p>`,`<p><b>每张卡要 ${fGB(e.me.total)}，光是模型状态就有 ${fGB(e.me.states)}，超过了 80 GB，放不下。</b></p>`)+verdict()+
    tx(`<p>Data parallelism splits the data, not the model: the 8 GPUs hold 8 identical copies of the weights, gradients and optimizer state. The gradients’ all-reduce runs during the backward pass, so the GPU hardly waits. Each GPU sends ${fGB(e.k.bytes)} per step. The gradients are kept in FP32, but they travel in BF16, 2 bytes per value; every transfer in this lab is counted that way.</p>`,
       `<p>数据并行分的是数据，不是模型。8 张卡上存着 8 份一模一样的权重、梯度和优化器状态。梯度的 all-reduce 和反向同时进行，所以 GPU 几乎不用等。每张卡每步要发出 ${fGB(e.k.bytes)} 的数据。梯度存的是 FP32，在卡之间传的时候却按 BF16 传，每个数值 2 字节；这个 Lab 里所有的传输都这样算。</p>`);
}
function obsZ1(){
  const e=evalC({...P2,st:S.st});
  if(S.st!=='z1') return tx(`<p><b>A full copy of the model states on each GPU: with the activations and logits, ${fGB(e.me.total)} per GPU, over 80 GB.</b> Split the optimizer state 8 ways.</p>`,`<p><b>每张卡存一整份模型状态，加上激活和 logits，一共要 ${fGB(e.me.total)}，超过 80 GB。</b>把优化器状态分成 8 份试试。</p>`);
  const d=evalC(P2);
  return tx(`<p><b>ZeRO-1: ${fGB(e.me.total)} per GPU, and it fits (data parallel needed ${fGB(d.me.total)}).</b></p><p>The data each GPU moves is unchanged, ${fGB(e.k.bytes)} per step: a reduce-scatter and an all-gather each move half as much as an all-reduce. In this lab ZeRO-1 costs no extra transfer time.</p>`,
            `<p><b>ZeRO-1：每张卡 ${fGB(e.me.total)}，放得下（数据并行时要 ${fGB(d.me.total)}）。</b></p><p>每张卡要传的数据没有变，每步还是 ${fGB(e.k.bytes)}。reduce-scatter 和 all-gather 各传 all-reduce 一半的量。在这个 Lab 里，ZeRO-1 不多花传输时间。</p>`)+
    MN(MS(),'6N + \\frac{12N}{d}')+tx(`<p>6N is the BF16 weights (2N) and the FP32 gradients (4N): every GPU keeps them whole. ${A.ui.math('\\frac{12N}{d}')} is the optimizer state, the FP32 copy, m and v (12N), split over the d GPUs. N is the number of parameters, d the number of GPUs. Activations are not in it.</p><p>Megatron-LM’s distributed optimizer documents the same: ${A.ui.math('6 + \\frac{12}{d}')} bytes per parameter.</p>`,
                                                `<p>6N 是 BF16 权重（2N）和 FP32 梯度（4N），每张卡各存一整份。${A.ui.math('\\frac{12N}{d}')} 是优化器状态，也就是 FP32 权重副本、m 和 v（共 12N），分到 d 张卡上。N 是参数量，d 是卡数。激活不算在里面。</p><p>Megatron-LM 的分布式优化器文档写的也是这个数：每个参数 ${A.ui.math('6 + \\frac{12}{d}')} 字节。</p>`);
}
function obsZ2(){
  const e=evalC({...P2,st:S.st}), z1=evalC(P3);
  if(S.st!=='z2') return tx(`<p><b>ZeRO-1: ${fGB(z1.me.total)} per GPU.</b> Split the gradients 8 ways too.</p>`,`<p><b>ZeRO-1：每张卡 ${fGB(z1.me.total)}。</b>把梯度也分成 8 份试试。</p>`);
  return tx(`<p><b>ZeRO-2: ${fGB(e.me.total)} per GPU, down from ${fGB(z1.me.total)} with ZeRO-1.</b></p><p>The transfers are the same as ZeRO-1’s, so the timeline does not change. Each layer’s gradients go out by reduce-scatter anyway, and ZeRO-2 only frees what the GPU no longer needs.</p>`,
            `<p><b>ZeRO-2：每张卡 ${fGB(e.me.total)}，ZeRO-1 时是 ${fGB(z1.me.total)}。</b></p><p>传输和 ZeRO-1 完全一样，所以时间线没有变。每层的梯度本来就用 reduce-scatter 传出去，ZeRO-2 只是把用不上的那部分释放掉。</p>`)+
    MN(MS(),'2N + \\frac{16N}{d}')+tx(`<p>2N is the BF16 weights, still whole on every GPU. ${A.ui.math('\\frac{16N}{d}')} is the FP32 gradients (4N) and the optimizer state (12N), split over the d GPUs. Activations are not in it.</p>`,
                                       `<p>2N 是 BF16 权重，每张卡还是一整份。${A.ui.math('\\frac{16N}{d}')} 是 FP32 梯度（4N）和优化器状态（12N），分到 d 张卡上。激活不算在里面。</p>`);
}
function obsZ3(){
  const e=evalC({...P5,ov:lv5==='ov'?1:0}), d=evalC(P2), n=evalC({...P5,ov:0});
  const head=tx(`<p><b>ZeRO-3: only ${fGB(e.me.total)} per GPU, and each step moves 1.5 times the data of plain data parallelism.</b></p>`,`<p><b>ZeRO-3：每张卡只要 ${fGB(e.me.total)}，每步要传的数据是数据并行的 1.5 倍。</b></p>`);
  if(lv5==='noov') return head+tx(`<p><b>Fetching when needed, this GPU spends ${fPctW(e.s.share)} of the step waiting for data</b>, the red parts of the timeline. Without prefetching, transfers and computation do not overlap, so the GPU stops for every transfer: before every layer for its weights, and after each layer’s backward pass for the gradients’ reduce-scatter. Now try prefetching.</p>`,
                                   `<p><b>用到才取时，这张卡有 ${fPctW(e.s.share)} 的时间在等数据</b>，就是时间线里红色的部分。不预取时，传输和计算不重叠，每传一次，GPU 都要停下来等：每一层开始前等这一层的权重，反向的每一层算完，还要等梯度的 reduce-scatter 传完。再试试预取。</p>`);
  return head+tx(`<p><b>With ${DEF('prefetching',TIP.pre())}, the waiting drops from ${fPctW(n.s.share)} to ${fPctW(e.s.share)} of the step:</b> the extra transfers hide behind the computation. The price is a second layer’s weights in memory, about ${fGB(e.me.inf/2)}.</p>`,
                 `<p><b>用了${DEF('预取',TIP.pre())}，等数据的时间从这一步的 ${fPctW(n.s.share)} 降到 ${fPctW(e.s.share)}。</b>多传的数据藏在计算后面，几乎不花时间。代价是显存里要多放一层的权重，约 ${fGB(e.me.inf/2)}。</p>`)+
    MN(MS(),'\\frac{18N}{d}')+tx('<p>18N bytes are the weights, gradients and optimizer state of N parameters, split over d GPUs. Not in it: the activations and the full weights of the layers in use.</p>',
                                  '<p>18N 字节是 N 个参数的权重、梯度和优化器状态，全部分到 d 张卡上。不算在里面的有激活，还有正在用的那几层的完整权重。</p>');
}
function obsNet(){
  const e=evalC(lv6Cfg(lv6)), z=evalC(P6);
  if(lv6==='z3') return tx(`<p><b>With one NIC per machine, this GPU waits for data ${fPctW(z.s.share)} of the step.</b> The transfers take ${fSec(z.k.time)}, the ${DEF('computation',TIP.share())} only ${fSec(z.s.C)}, so the computation can no longer cover them. Try the other two.</p>`,
                           `<p><b>每台机器只有一块网卡时，这张卡有 ${fPctW(z.s.share)} 的时间在等数据。</b>传输要 ${fSec(z.k.time)}，${DEF('计算',TIP.share())}只有 ${fSec(z.s.C)}，计算已经盖不住传输了。试试另外两种做法。</p>`);
  if(lv6==='tok') return tx(`<p><b>16,384 tokens per GPU: the waiting drops from ${fPctW(z.s.share)} to ${fPctW(e.s.share)} of the step.</b></p><p>The data to move depends only on the model, not on the tokens, while the computation grows with the tokens. Four times the tokens give the transfers enough computation to hide behind.</p><p>It has two prices. Each GPU’s activations and logits grow four times as well, to ${fGB(e.me.act+e.me.lg)}. And the model sees four times as much data before each update.</p>`,
                          `<p><b>每张卡每步 16,384 个 token：等数据的时间从 ${fPctW(z.s.share)} 降到 ${fPctW(e.s.share)}。</b></p><p>要传的数据只取决于模型，和 token 数无关。计算却随 token 数增长。token 变成 4 倍，传输就有足够的计算可以藏在后面。</p><p>代价有两个。每张卡的激活和 logits 也变成 4 倍，一共 ${fGB(e.me.act+e.me.lg)}。模型也要处理 4 倍的数据，才更新一次参数。</p>`);
  return tx(`<p><b>One NIC per GPU: the waiting drops from ${fPctW(z.s.share)} to ${fPctW(e.s.share)} of the step.</b></p><p>With 8 NICs a machine has 400 GB/s, eight times as much. The transfers take only ${fSec(e.k.time)} and hide behind the ${fSec(e.s.C)} of computation again. Large H100 clusters are usually built this way. torchtitan’s test cluster also has 400 Gb/s per GPU between hosts.</p><p>Another way is to split the model itself across GPUs; the lab “${labLink('model-parallel')}” covers it.</p>`,
            `<p><b>每张卡配一块网卡：等数据的时间从 ${fPctW(z.s.share)} 降到 ${fPctW(e.s.share)}。</b></p><p>8 块网卡让一台机器有 400 GB/s，是原来的 8 倍。传输只要 ${fSec(e.k.time)}，又藏到了 ${fSec(e.s.C)} 的计算后面。大的 H100 集群通常这样配。torchtitan 测试用的集群，机器之间也是每张卡 400 Gb/s。</p><p>还有一条路是把模型本身切开，分到几张卡上，那是“${labLink('model-parallel')}”这个 Lab 讲的内容。</p>`);
}
const REMS=['dp','z3','net'];
function remBtn(k){const on=remView===k;
  const tip={dp:tx('Shows data parallelism on 8 GPUs.','在图里显示 8 张卡的数据并行。'),z3:tx('Shows ZeRO-3 with prefetching.','在图里显示带预取的 ZeRO-3。'),net:tx('Shows ZeRO-3 over 64 GPUs with one NIC per machine and 4,096 tokens per GPU.','在图里显示 64 张卡上的 ZeRO-3：每台机器一块网卡，每张卡 4,096 个 token。')}[k];
  return ` <button type="button" class="linkbtn zr-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const d=evalC(P2), z1=evalC(P3), z3=evalC({...P5,ov:1}), z3n=evalC(P5), nt=evalC(P6), tk=evalC(lv6Cfg('tok'));
  return tx(`<ul class="zr-remlist"><li><b>Data parallelism splits the data, not the model.</b> Each GPU holds a full copy of the model states, so more GPUs process more data while each GPU needs just as much memory.${remBtn('dp')}</li>`+
    `<li><b>Shard what is identical, and fetch it when needed.</b> ZeRO-3 splits the weights, the gradients and the optimizer state (PyTorch’s FSDP does this by default), so with d GPUs each holds ${A.ui.math('\\frac{1}{d}')} of the model states.${remBtn('z3')}</li>`+
    `<li><b>The extra transfers cost time only when the computation cannot cover them.</b> When it cannot, give each GPU more tokens per step, or give the machines a faster network. Either way the waiting drops.${remBtn('net')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Data parallelism makes a model fit that does not fit on one GPU.” Every GPU still keeps everything: its model states alone are as large as on a single GPU.</p><p>✗ “ZeRO-3 saves memory for free.” It moves half again as much data. Without prefetching, the GPU waits for each layer’s weights before it can compute that layer.</p><p>✗ “More GPUs make every step faster.” With more GPUs, each GPU computes no less and moves about as much data. More GPUs process more tokens per step, but each step is no faster.</p></div>${A.shell.toScenarios()}`,
    `<ul class="zr-remlist"><li><b>数据并行分的是数据，不是模型。</b>每张卡都存着一整份模型状态。所以卡越多，能同时算的数据越多，每张卡要的显存却一点不少。${remBtn('dp')}</li>`+
    `<li><b>一样的东西分开存，用的时候再取。</b>ZeRO-3 把权重、梯度和优化器状态都分开存，PyTorch 的 FSDP 默认就这样做。d 张卡时，每张卡只存 ${A.ui.math('\\frac{1}{d}')} 的模型状态。${remBtn('z3')}</li>`+
    `<li><b>多传的数据，只有计算盖不住时才花时间。</b>盖不住时，可以让每张卡每步多算一些 token，也可以给机器配更快的网络。这两种办法都能让等数据的时间降下来。${remBtn('net')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “数据并行能让一张卡放不下的模型放下。”每张卡还是什么都存，光模型状态就和只用一张卡时一样多。</p><p>✗ “ZeRO-3 省显存，不花代价。”它要多传一半的数据。不预取的话，GPU 每算一层，都要先等这一层的权重传过来。</p><p>✗ “卡越多，每步越快。”卡多了，每张卡要算的不会变少，要传的数据也差不多一样多。所以卡越多，每一步处理的 token 越多，但每一步不会更快。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 fsdp:'https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/api.py',
 fsdp2:'https://github.com/pytorch/pytorch/blob/93cf7fb9e32e18f37b7369666c3ca941fe276164/docs/source/distributed.fsdp.fully_shard.md',
 fsdp2py:'https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py',
 dsz:'https://github.com/microsoft/DeepSpeed/blob/a77aeb676507beb9fe0604bc31c2bf075daf7c46/docs/_tutorials/zero.md',
 dsc:'https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_pages/config-json.md',
 nccl:'https://github.com/NVIDIA/nccl-tests/blob/a3d4589e8f32f8363c334c52eedad75b44ed8b2d/doc/PERFORMANCE.md',
 tt:'https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/benchmarks/llama3_h100_202412_torchtitan.md',
 ttf:'https://github.com/pytorch/torchtitan/blob/e3929368edd2f2457456913890ba2b4c0f27da31/docs/fsdp.md',
 meg:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/docs/user-guide/features/dist_optimizer.md',
 hf:'https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/docs/source/en/model_memory_anatomy.md',
 mle:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/training/model-parallelism/README.md',
 sbg:'https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/gpus.md',
 zero:'https://arxiv.org/abs/1910.02054',
 fsdpp:'https://arxiv.org/abs/2304.11277',
 dscfg:'https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/config.py',
 ttcfg:'https://github.com/pytorch/torchtitan/blob/main/torchtitan/config/configs.py',
 megarg:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/megatron/training/arguments.py',
 mamf:'https://github.com/stas00/ml-engineering/blob/edb6919fe61301bcb96edc08899d1c4dd5edfb28/compute/accelerator/README.md',
 megr:'https://github.com/NVIDIA/Megatron-LM/blob/ef8cd10b8e34a94c9ad7452e7851f58337f353a0/README.md'
};
const PRESETS=[
 {id:'one',t:{en:'Llama‑3.1‑8B on one 8-GPU machine',zh:'一台 8 卡机器训练 Llama‑3.1‑8B'},d:{en:'The common small setup: does ZeRO-1 already fit?',zh:'最常见的小规模训练：ZeRO-1 就够了吗？'},s:{m:'8b',N:8,st:'z1',T:4096,ov:1,net:'ib',u:50}},
 {id:'long',t:{en:'Long context: one 128K-token sequence per GPU',zh:'长上下文：每张卡一条 128K token 的序列'},d:{en:'Here the activations and logits, not the model states, decide',zh:'这时决定显存的是激活和 logits，不是模型状态'},s:{m:'8b',N:8,st:'z3',T:131072,ov:1,net:'ib',u:50}},
 {id:'hsdp',t:{en:'8 machines, 64 GPUs: shard inside, copy across',zh:'8 台机器、64 张卡：机器内分片，机器间复制'},d:{en:'Keep the all-gathers and reduce-scatters inside each machine',zh:'让 all-gather 和 reduce-scatter 都留在机器内'},s:{m:'8b',N:64,st:'hsdp',T:4096,ov:1,net:'ib',u:50}},
 {id:'eth',t:{en:'A team with ordinary Ethernet between machines',zh:'机器之间只有普通以太网的团队'},d:{en:'4 machines, 32 GPUs, 100 Gb/s per machine',zh:'4 台机器、32 张卡，每台机器 100 Gb/s'},s:{m:'8b',N:32,st:'z1',T:4096,ov:1,net:'eth',u:50}}
];
const same=(a,b)=>['m','N','st','T','ov','net','u'].every(k=>a[k]===b[k]);
const presetNow=()=>PRESETS.findIndex(p=>same(p.s,S));
function scenHTML(i){
  const p=PRESETS[i], e=evalC(S);
  if(p.id==='one'){const d=evalC({...S,st:'dp'}), z3=evalC({...S,st:'z3'});
    return tx(`<p><b>With ZeRO-1, each GPU needs ${fGB(e.me.total)} and fits, and ${waitSay(e.s.share)}.</b></p><p>Data parallel would need ${fGB(d.me.total)} per GPU. ZeRO-3 would go down to ${fGB(z3.me.total)}, at the price of half again as much data to move. The memory it saves could let each GPU process more tokens per step, or recompute fewer layers.</p>`,
              `<p><b>用 ZeRO-1，每张卡要 ${fGB(e.me.total)}，放得下，${waitSay(e.s.share)}。</b></p><p>数据并行每张卡要 ${fGB(d.me.total)}。ZeRO-3 能降到 ${fGB(z3.me.total)}，代价是多传一半的数据。省下的显存可以让每张卡每步处理更多 token，或者少重算一些层。</p>`);}
  if(p.id==='long')
    return tx(`<p><b>With ZeRO-3, each GPU still needs ${fGB(e.me.total)}. It does not fit.</b></p><p>Sharding shrinks only the model states, here down to ${fGB(e.me.states)}. The activations and logits grow with the tokens each GPU processes, and every GPU keeps its own. For one 128K-token sequence the two take ${fGB(e.me.act+e.me.lg)}. The lab “${labLink('training-memory')}” explains them.</p><p>In practice one sequence is split over several GPUs, which is called context parallelism (CP). torchtitan trains Llama 3.1 8B on 131,072-token sequences with 8 GPUs as FSDP 2 × CP 4, so 4 GPUs share one sequence (${cite(SRC.tt,'benchmarks')}). The lab “${labLink('model-parallel')}” covers it too.</p><p>Take the step time here as a lower bound. This lab leaves out the arithmetic of attention, which at 128K is about twice the rest, so a real step takes about three times as long.</p>`,
              `<p><b>用 ZeRO-3，每张卡还是要 ${fGB(e.me.total)}，放不下。</b></p><p>分片只能缩小模型状态，这里只剩 ${fGB(e.me.states)}。激活和 logits 随每张卡处理的 token 数增长，每张卡都要存自己的那份。一条 128K 的序列，光这两样就要 ${fGB(e.me.act+e.me.lg)}。“${labLink('training-memory')}”那个 Lab 专门讲激活。</p><p>业界的做法是把一条序列切到几张卡上，叫上下文并行（CP）。torchtitan 用 8 张卡训练 Llama 3.1 8B、序列长 131,072 个 token 时，分成 FSDP 2 × CP 4，4 张卡分一条序列（${cite(SRC.tt,'测试结果')}）。“${labLink('model-parallel')}”那个 Lab 也讲到了它。</p><p>这里的每步时间只能当下限看。这个 Lab 没有算注意力的运算，128K 时它约是其余部分的 2 倍，实际一步要长到约 3 倍。</p>`);
  if(p.id==='hsdp'){const z=evalC({...S,st:'z3'});
    return tx(`<p><b>HSDP: ${fGB(e.me.total)} per GPU, and ${waitSay(e.s.share)}. ZeRO-3 over all 64 GPUs needs only ${fGB(z.me.total)} per GPU and hardly waits either.</b></p><p>HSDP keeps the weights’ all-gathers and the gradients’ reduce-scatters inside each machine, on NVLink. Only one all-reduce of the gradients crosses the network, about ${fGB(e.k.netB)} per machine per step, a third less than ZeRO-3’s ${fGB(z.k.netB)}.</p><p>On this network both hide their transfers, so HSDP’s extra memory buys no time here. The third it saves on the network counts when the network is slow. PyTorch FSDP calls it HYBRID_SHARD: full sharding within a node, parameters replicated across nodes (${cite(SRC.fsdp,'torch/distributed/fsdp/api.py')}). torchtitan sets it with a 2D device mesh (${cite(SRC.ttf,'docs/fsdp.md')}).</p>`,
              `<p><b>HSDP：每张卡 ${fGB(e.me.total)}，${waitSay(e.s.share)}。ZeRO-3 分到全部 64 张卡时，每张卡只要 ${fGB(z.me.total)}，也几乎不用等。</b></p><p>HSDP 把权重的 all-gather 和梯度的 reduce-scatter 都留在机器内，走 NVLink。过网络的只有梯度的一次 all-reduce，每台机器每步约 ${fGB(e.k.netB)}，比 ZeRO-3 的 ${fGB(z.k.netB)} 少三分之一。</p><p>在这个网络上，两种做法都藏得住传输，HSDP 多占的显存没有换来时间。网络慢时，少传的这三分之一才有用。PyTorch FSDP 把这种做法叫 HYBRID_SHARD，在机器内做 FULL_SHARD，在机器之间复制参数（${cite(SRC.fsdp,'torch/distributed/fsdp/api.py')}）。torchtitan 用二维的 device mesh 来设置它（${cite(SRC.ttf,'docs/fsdp.md')}）。</p>`);}
  const ib=evalC({...S,net:'ib'});
  return tx(`<p><b>Over 100 Gb/s Ethernet per machine, this GPU waits for data ${fPctW(e.s.share)} of the step. The transfers take ${fSec(e.k.time)}, the computation only ${fSec(e.s.C)}.</b></p><p>The 8 GPUs of a machine share one 100 Gb/s link, 12.5 GB/s, 1/32 of the 400 GB/s per machine of a large H100 cluster. On that network the same run would ${ib.s.share<0.01?'hardly wait for data':`wait ${fPctW(ib.s.share)}`}.</p><p>With a network this slow, the transfers cannot hide. The way out is more computation between two synchronizations: more tokens per GPU per step, or the gradients of several batches added up before one synchronization. With ZeRO-1, the second moves no extra data (ml-engineering).</p>`,
            `<p><b>每台机器只有 100 Gb/s 以太网时，这张卡有 ${fPctW(e.s.share)} 的时间在等数据。传输要 ${fSec(e.k.time)}，计算只有 ${fSec(e.s.C)}。</b></p><p>一台机器的 8 张卡共用这一条以太网，每台机器只有 12.5 GB/s，是大的 H100 集群（每台 400 GB/s）的 1/32。换成那样的网络，同一个训练${ib.s.share<0.01?'几乎不用等数据':`只等 ${fPctW(ib.s.share)}`}。</p><p>网络这么慢，传输完全藏不住。办法是两次同步之间多算一些：每张卡每步多处理一些 token，或者攒几批的梯度再同步一次。用 ZeRO-1 时，攒几批再同步不会多传数据（ml-engineering）。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=(st,T)=>tx(`${stName(st)}, ${fTok(T)} tokens per GPU`,`${stName(st)}，每张卡 ${fTok(T)} 个 token`);
/* round numbers the way a person says them (R1); no hover that only repeats a number's exact value (the owner's clarification of
   2026-10-07 23:37 UTC): the value is plain text, red when it fails */
const rGB=b=>{const v=b/1e9; return (v>=10?fixd(v,0):fixd(v,1))+' GB';};
const rSec=t=>{const v=t>=5?fixd(t,0):(t>=1?fixd(t,1):fixd(t,2)); return tx(v+' s',v+' 秒');};
const rPct=x=>x<0.01?tx('under 1%','不到 1%'):fixd(x*100,0)+'%';
const sv=(txt,_exact,bad)=>`<span class="zr-sv${bad?' zr-bad':''}">${txt}</span>`;
/* the main gap of a plan, in one computed sentence */
function chGap(me,best){
  if(!me.me.fits){
    if(me.me.states>=me.me.act+me.me.lg){const v=sv(tx('about ','约 ')+rGB(me.me.states),fGB(me.me.states));
      return ({z1:tx(`ZeRO-1 splits only the optimizer state, so the model states alone take ${v} per GPU.`,`ZeRO-1 只分了优化器状态，每张卡光模型状态就要${v}。`),
        z2:tx(`ZeRO-2 still keeps the full BF16 weights on every GPU, so the model states alone take ${v} per GPU.`,`ZeRO-2 每张卡还存着一整份 BF16 权重，光模型状态就要${v}。`),
        hsdp:tx(`HSDP splits the model states only over the 8 GPUs of a machine, so they alone take ${v} per GPU.`,`HSDP 只在一台机器的 8 张卡之间分，每张卡光模型状态就要${v}。`)})[me.c.st]||'';}
    const v=sv(tx('about ','约 ')+rGB(me.me.act+me.me.lg),fGB(me.me.act+me.me.lg));
    return tx(`With ${fTok(me.c.T)} tokens per GPU, the activations and logits alone take ${v}, and sharding does not split them.`,`每张卡 ${fTok(me.c.T)} 个 token 时，光激活和 logits 就要${v}，分片分不了它们。`);}
  if(me.s.share>CH.wait){const c=sv(tx('about ','约 ')+rSec(me.s.C),fSec(me.s.C)), k=sv(tx('about ','约 ')+rSec(me.k.time),fSec(me.k.time));
    return tx(`With ${fTok(me.c.T)} tokens per GPU, a step computes for only ${c} but its transfers take ${k}, so the computation cannot hide them.`,`每张卡 ${fTok(me.c.T)} 个 token 时，一步只算${c}，传输却要${k}，计算盖不住传输。`);}
  const a=sv(tx('about ','约 ')+rSec(me.s.step),fSec(me.s.step)), b=sv(tx('about ','约 ')+rSec(best.s.step),fSec(best.s.step));
  return tx(`Your step takes ${a}, the best one ${b}. More tokens per step make each step longer, and the same tokens give the model fewer updates.`,`你的每步要${a}，最好的设置只要${b}。每步的 token 越多，每步越长，同样多的 token 让模型更新的次数也越少。`);
}
/* the one change that helps most, in one sentence */
function chNext(me,best){
  const o=chOne(me.c.st,me.c.T);
  const what=e=>e.c.st!==me.c.st?tx(`switching to ${stName(e.c.st)}`,`把分法换成 ${stName(e.c.st)}`):tx(`${e.c.T>me.c.T?'raising':'lowering'} the tokens per GPU to ${fTok(e.c.T)}`,`把每张卡的 token ${e.c.T>me.c.T?'加':'减'}到 ${fTok(e.c.T)}`);
  if(o&&o.pass){const isB=o.e.c.st===best.c.st&&o.e.c.T===best.c.T;
    return tx(`The one change that helps most is ${what(o.e)}: it meets the goal${isB?', and it is the best setting':''}.`,`只改一项的话，${what(o.e)}最管用，这样就达标了${isB?'，而且正是最好的设置':''}。`);}
  if(o){const w=sv(tx('about ','约 ')+rPct(o.e.s.share),fPctW(o.e.s.share));
    return tx(`No single change meets the goal. ${what(o.e)[0].toUpperCase()+what(o.e).slice(1)} makes it fit, but each GPU then waits ${w} of the step; the best setting also uses ${fTok(best.c.T)} tokens per GPU.`,
              `只改一项还达不到目标。${what(o.e)}就放得下了，但等数据要占${w}，所以最好的设置还把每张卡的 token 定在 ${fTok(best.c.T)}。`);}
  return tx(`No single change meets the goal. The best setting changes both at once: ${stName(best.c.st)} and ${fTok(best.c.T)} tokens per GPU.`,`只改一项还达不到目标。最好的设置同时改了两项：分法换成 ${stName(best.c.st)}，每张卡的 token 定在 ${fTok(best.c.T)}。`);
}
/* the summary (R1, after the train-capstone lab): the reader's setting beside the best one, what they share written once, only the
   differing settings shown, highlighted where the reader's differs; what fails is red */
function chSumHTML(me,best){
  const rows=[{e:me,role:'me'},{e:best,role:'best'}], sameSt=me.c.st===best.c.st, sameT=me.c.T===best.c.T;
  const share=tx(`Both train Llama‑3.1‑70B on 64 H100s in 8 machines, 50 GB/s of network per machine, with prefetching${sameSt?`, ${stName(me.c.st)}`:''}${sameT?`, ${fTok(me.c.T)} tokens per GPU`:''}.`,
                 `两种设置都用 8 台机器、64 张 H100 训练 Llama‑3.1‑70B，每台机器的网络 50 GB/s，开了预取${sameSt?`，都用 ${stName(me.c.st)}`:''}${sameT?`，每张卡都是 ${fTok(me.c.T)} 个 token`:''}。`);
  const lab={me:tx('Your setting','你的设置'),best:DEF(tx('Best setting','最好的设置'),tx('Of the settings that meet both conditions, the one with the shortest step.','满足两个条件的设置里，每步最短的那一种。'))};
  const row=x=>{const e=x.e, on=k=>x.role==='me'&&(k==='st'?!sameSt:!sameT);
    const chips=(sameSt?'':`<span class="zr-chip${on('st')?' on':''}">${esc(stName(e.c.st))}</span>`)+(sameT?'':`<span class="zr-chip${on('T')?' on':''}">${esc(tx(`${fTok(e.c.T)} tokens per GPU`,`每张卡 ${fTok(e.c.T)} 个 token`))}</span>`);
    let v;
    if(!e.me.fits) v=sv(tx(`${rGB(e.me.total)} per GPU: does not fit in 80 GB`,`每张卡约 ${rGB(e.me.total)}，超过 80 GB，放不下`),fGB(e.me.total),true);
    else {const wb=e.s.share>CH.wait;
      v=sv(tx(`${rGB(e.me.total)} per GPU`,`每张卡约 ${rGB(e.me.total)}`),fGB(e.me.total))+
        sv(tx(`waits ${rPct(e.s.share)}${wb?', over 10%':''}`,`等数据${rPct(e.s.share)==='不到 1%'?'':'约 '}${rPct(e.s.share)}${wb?'，超过 10%':''}`),fPctW(e.s.share)+(e.s.share<0.01?'':` (${fixd(e.s.share*100,1)}%)`),wb)+
        sv(tx(`${rSec(e.s.step)} per step`,`每步约 ${rSec(e.s.step)}`),fSec(e.s.step));}
    return `<li class="zr-c2r ${x.role}${e.ok?'':' miss'}"><div class="zr-c2h"><span class="zr-c2t">${lab[x.role]}</span>${chips}</div><div class="zr-c2v">${v}</div></li>`;};
  return `<div class="zr-sum"><div class="zr-sumcap">${tx('Your setting and the best one: where they differ','你的设置和最好的设置差在哪里')}</div><p class="small zr-same">${share}</p>`+
    `<ul class="zr-c2">${rows.map(row).join('')}</ul><p class="zr-gap">${chGap(me,best)}</p><p class="zr-next">${chNext(me,best)}</p></div>`;
}
/* the settings that pass, and only those, behind a disclosure closed by default (R1) */
function chPassHTML(me){
  const ps=chAll().filter(e=>e.ok).sort((a,b)=>a.s.step-b.s.step), b=ps[0];
  const li=e=>{const mine=e.c.st===me.c.st&&e.c.T===me.c.T, isB=e===b;
    return `<li class="${mine?'me':''}"><b>${esc(chName(e.c.st,e.c.T))}</b>${isB?`<span class="zr-chip on">${tx('shortest step','每步最短')}</span>`:''}${mine?`<span class="zr-chip">${tx('yours','你的')}</span>`:''}<br>`+
      sv(tx(`${rGB(e.me.total)} per GPU`,`每张卡约 ${rGB(e.me.total)}`),fGB(e.me.total))+sv(tx(`waits ${rPct(e.s.share)}`,`等数据${rPct(e.s.share)==='不到 1%'?'':'约 '}${rPct(e.s.share)}`),fPctW(e.s.share)+(e.s.share<0.01?'':` (${fixd(e.s.share*100,1)}%)`))+sv(tx(`${rSec(e.s.step)} per step`,`每步约 ${rSec(e.s.step)}`),fSec(e.s.step))+`</li>`;};
  return `<details class="zr-all"><summary>${tx(`See all settings that pass (${ps.length})`,`看全部可行的设置（${ps.length} 种）`)}</summary><div class="zr-allb">`+
    `<p class="small">${tx(`Only these ${ps.length} of the 24 combinations fit in 80 GB and wait at most 10%. All are ZeRO-3: the other ways to split keep too much on each GPU.`,`24 种组合里，只有这 ${ps.length} 种放得下 80 GB、等数据不超过 10%。它们都是 ZeRO-3，别的分法每张卡存得太多。`)}</p>`+
    `<ul class="zr-pass">${ps.map(li).join('')}</ul>`+
    tx('<p class="small muted">Note: this assumes a GPU reaches 50% of the peak while computing, and 50 GB/s of network per machine. Measure both on your own cluster. A faster GPU or a slower network needs more tokens per GPU to hide the same transfers.</p>',
       '<p class="small muted">注意：这里假设 GPU 算的时候能跑到峰值的 50%，每台机器的网络是 50 GB/s。这两个数要在自己的集群上测。GPU 更快或网络更慢时，要更多的 token 才能盖住同样的传输。</p>')+
    `</div></details>`;
}
let chAllOpen=false;
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=evalC(chCfg(S.st,S.T)), best=chBest(), isBest=me.ok&&me.c.st===best.c.st&&me.c.T===best.c.T;
  const vals=e=>tx(`${rGB(e.me.total)} per GPU, waiting ${rPct(e.s.share)}, ${rSec(e.s.step)} per step`,`每张卡约 ${rGB(e.me.total)}，等数据${rPct(e.s.share)==='不到 1%'?'':'约 '}${rPct(e.s.share)}，每步约 ${rSec(e.s.step)}`);
  let h;
  if(isBest) h=tx(`<p class="okline zr-mt"><b>Goal met: ${vals(me)}.</b> And no passing option has a shorter step.</p>`,`<p class="okline zr-mt"><b>达成目标：${vals(me)}。</b>而且在达标的方案里，这一步最短。</p>`);
  else if(me.ok) h=tx(`<p class="okline zr-mt"><b>Goal met: ${vals(me)}.</b> But a shorter step also works.</p>`,`<p class="okline zr-mt"><b>达标了：${vals(me)}。</b>不过还有每步更短的设置。</p>`);
  else {const why=!me.me.fits?tx(`it needs about ${rGB(me.me.total)} per GPU, over 80 GB`,`每张卡要约 ${rGB(me.me.total)}，超过 80 GB`):tx(`this GPU waits about ${rPct(me.s.share)} of the step, over 10%`,`这一步里约 ${rPct(me.s.share)} 的时间在等数据，超过 10%`);
    h=tx(`<p class="badline zr-mt"><b>Not yet: ${why}.</b></p>`,`<p class="badline zr-mt"><b>还没达到：${why}。</b></p>`);
    if(!chReveal) h+=tx('<p class="small">Hint: first check which sharding fits at all. With 18 bytes per parameter, work out how much each sharding leaves on one GPU. Then think: the data to move does not depend on the tokens, but the computation grows with them.</p>',
         '<p class="small">提示：先看哪种分法放得下。每个参数 18 字节，算一算每种分法在一张卡上留下多少。再想想，要传的数据和 token 数无关，计算却和 token 数成正比。</p>')+
      `<button type="button" class="linkbtn" data-act="reveal">${tx('Show the best setting','直接看最好的设置')}</button>`;}
  if(!chReveal) return h;
  return h+(isBest?'':chSumHTML(me,best))+chPassHTML(me);
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Predict first, then watch data parallelism copy the model onto every GPU, ZeRO split the optimizer state, the gradients and the weights, and what the extra transfers cost across machines. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看数据并行怎样把模型复制到每张卡上，ZeRO 怎样把优化器状态、梯度和权重分开存，以及跨机器时多传的数据要花多少时间。第一次学从这里开始。`),
  sandbox:tx('Four real situations, from one 8-GPU machine to a team with only Ethernet between machines. Pick one to see what each GPU holds and how long it waits for data.',
             '四个真实情况，从一台 8 卡机器到机器之间只有以太网的团队。选一个，看每张卡存了什么、要等多久的数据。'),
  challenge:tx('Train Llama‑3.1‑70B on 64 H100s in 8 machines with a slow network between them. Pick the sharding and the tokens per GPU per step so that it fits in 80 GB and each GPU waits for data at most 10% of the step, with the shortest step you can.',
               '用 8 台机器、64 张 H100 训练 Llama‑3.1‑70B，机器之间的网络不快。选好分法和每张卡每步的 token 数，让每张卡放得下 80 GB、等数据的时间不超过每步的 10%，同时每步尽量短。')
});
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> memory per GPU and the data moved are exact formulas; the step time comes from a schedule of one compute stream and one transfer stream, layer by layer. ${J('math','“Show the math”')} works the numbers out.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Model states, 18 bytes per parameter: BF16 weights 2, FP32 gradients 4, FP32 copy of the weights 4, Adam’s m and v 4 each, as Megatron-LM and Hugging Face count them. The ZeRO paper counts 16, because it keeps the gradients in 16 bits.</li>`+
 `<li>Sharding as each stage says: ZeRO-1 the optimizer state (FP32 copy, m, v), ZeRO-2 also the gradients, ZeRO-3 also the weights (DeepSpeed’s ZeRO tutorial); PyTorch FSDP’s default, FULL_SHARD, is ZeRO-3. ZeRO-3 also holds the full weights of the layers in flight: two average layers with prefetching, one without.</li>`+
 `<li>Activations with full recomputation: each layer’s input (2h bytes per token) plus one layer’s activations while it is recomputed (counted tensor by tensor for a Llama layer in BF16 with FlashAttention: 41h bytes per token for 8B, 40.5h for 70B). The logits in FP32, tokens × 128,256 × 4 bytes, as the lab “Training memory” counts them.</li>`+
 `<li>Data sent: ring collectives as in NCCL’s performance notes. An all-reduce sends ${A.ui.math('\\frac{2(d - 1)}{d}')} of the data per GPU, a reduce-scatter or all-gather ${A.ui.math('\\frac{d - 1}{d}')}. Data parallel and ZeRO-1/2 move 2 × N values per step, ZeRO-3 3 × N, 1.5 times as much (ml-engineering). ZeRO-1 and ZeRO-2 gather the updated weights before each layer’s forward pass (as FSDP’s SHARD_GRAD_OP does).</li>`+
 `<li>Transfer time, as the scaling book models it. Inside one machine it is ${A.ui.math('\\frac{\\text{each GPU’s bytes}}{450\\ \\text{GB/s}}')}, over NVLink. Across machines, an all-gather or reduce-scatter takes about ${A.ui.math('\\frac{\\text{the whole array’s bytes}}{\\text{one machine’s network bandwidth}}')}, and an all-reduce twice that. HSDP’s all-gathers and reduce-scatters stay on NVLink.</li>`+
 `<li>Compute: with full recomputation, 8 × parameters FLOPs per token (one more forward pass), at the assumed share of the peak while computing; a quarter forward, three quarters backward with the recomputation. MFU, a result, counts 6 × parameters.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>Every transfer is counted in BF16, 2 bytes per value, as the ZeRO paper assumes; then ZeRO-3 moves 1.5 times the data of data parallelism. torchtitan and Megatron-LM reduce the gradients in FP32 by default, which doubles the bytes of the gradients’ transfer.</li>`+
 `<li>One transfer stream; the transfers of a layer run one after another. Real NCCL can run several at once and use other algorithms (trees, NVLink switches), faster or slower than this.</li>`+
 `<li>The arithmetic of attention is left out. With 4,096-token sequences it would add about 7% for 8B; at 128K about twice the rest (scenario 2).</li>`+
 `<li>The optimizer step, temporary buffers and memory fragmentation are left out. GPU memory is the vendor’s 80 GB, in GB (10⁹ bytes).</li></ul></div>`+
 `<div><div class="eyebrow">Assumed</div><ul>`+
 `<li>The share of the peak while computing: 50% of 989.5 TFLOP/s by default. Plain BF16 matrix multiplies reach about 80% (ml-engineering); attention, normalization and smaller kernels do not.</li>`+
 `<li>Network per machine, shared by its 8 GPUs: 400 GB/s with a 400 Gb/s NIC per GPU (torchtitan’s H100 cluster); 50 GB/s with a single 400 Gb/s NIC (part 6 and the challenge, the lab’s assumption); 12.5 GB/s with 100 Gb/s Ethernet (scenario 4, the lab’s assumption).</li>`+
 `<li>Inside a machine, NVLink at 450 GB/s per GPU, NVIDIA’s figure as the scaling book quotes it; the scaling book adds that in practice it is hard to exceed 370 GB/s.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>18 bytes per parameter: Megatron-LM ${cite(SRC.meg,'<code>dist_optimizer.md</code>')} (18, and ${A.ui.math('6 + \\frac{12}{d}')} with the distributed optimizer); Hugging Face ${cite(SRC.hf,'<code>model_memory_anatomy.md</code>')}.</li>`+
 `<li>Sharding strategies: DeepSpeed ${cite(SRC.dsz,'ZeRO tutorial')} and ${cite(SRC.dscfg,'<code>deepspeed/runtime/zero/config.py</code>')}; PyTorch FSDP ${cite(SRC.fsdp,'<code>torch/distributed/fsdp/api.py</code>')} (ShardingStrategy) and the ${cite(SRC.fsdp2,'<code>fully_shard</code> docs')}.</li>`+
 `<li>Transfers: ${cite(SRC.nccl,'nccl-tests <code>doc/PERFORMANCE.md</code>')} (ring factors); ${cite(SRC.mle,'ml-engineering, model parallelism')} (2 × and 3 × the parameters per step); ${cite(SRC.sbg,'the scaling book, <code>gpus.md</code>')} (NVLink, collectives across machines).</li>`+
 `<li>Gradients are reduced in FP32 by default in torchtitan (${cite(SRC.ttcfg,'<code>config/configs.py</code>')}) and Megatron-LM (${cite(SRC.megarg,'<code>arguments.py</code>')}).</li>`+
 `<li>Compute and MFU: ${cite(SRC.mamf,'ml-engineering, accelerators')} (BF16 matrix multiplies at about 80% of the H100’s peak); ${cite(SRC.tt,'torchtitan Llama 3 H100 benchmarks')} (400 Gb/s per GPU between hosts; Llama 3.1 8B with FSDP on 8 or 128 H100s, 33% to 39% MFU; context parallelism at 128K); ${cite(SRC.megr,'Megatron-LM README')} (41% to 47% MFU on H100).</li>`+
 `<li>Peak 989.5 TFLOP/s dense BF16 (H100 SXM) and model shapes as in the site’s other labs (config.json of each model).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04 (links pinned to the commits read that day, except the PyTorch <code>api.py</code> and <code>_fully_shard.py</code>, torchtitan’s <code>configs.py</code> and DeepSpeed’s <code>config.py</code>, read from the main branch). Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>每张卡的显存和要传的数据都是精确公式。每步的时间由一个排程算出，排程里有一条计算流和一条传输流，一层一层排。${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>模型状态每个参数 18 字节：BF16 权重 2、FP32 梯度 4、FP32 权重副本 4、Adam 的 m 和 v 各 4，和 Megatron-LM、Hugging Face 的算法一样。ZeRO 论文算的是 16 字节，因为它的梯度用 16 位存。</li>`+
 `<li>按各阶段的规则分开：ZeRO-1 分优化器状态（FP32 权重副本、m、v），ZeRO-2 再分梯度，ZeRO-3 再分权重（DeepSpeed 的 ZeRO 教程）。PyTorch FSDP 默认的 FULL_SHARD 就是 ZeRO-3。ZeRO-3 还要放正在用的几层的完整权重。预取时放两层（按平均一层的大小算），不预取时放一层。</li>`+
 `<li>激活按开了重算来算。每层留下自己的输入，每个 token 2h 字节。再加上正在重算的那一层的激活，按 Llama 一层在 BF16、FlashAttention 下逐个张量数出来，8B 每个 token 41h 字节，70B 是 40.5h。logits 按 FP32 算，token 数 × 128,256 × 4 字节，和“训练显存”那个 Lab 一样。</li>`+
 `<li>发出的数据按 NCCL 性能说明里的环形实现算。all-reduce 每张卡发出数据的 ${A.ui.math('\\frac{2(d - 1)}{d}')}，reduce-scatter 和 all-gather 各 ${A.ui.math('\\frac{d - 1}{d}')}。数据并行和 ZeRO-1/2 每步传 2 × N 个值，ZeRO-3 传 3 × N 个值，是 1.5 倍（ml-engineering）。ZeRO-1 和 ZeRO-2 在每层前向之前取回更新后的权重，和 FSDP 的 SHARD_GRAD_OP 一样。</li>`+
 `<li>传输时间按 scaling book 的算法。机器内走 NVLink，时间是 ${A.ui.math('\\frac{\\text{每张卡的字节数}}{450\\ \\text{GB/s}}')}。跨机器时，一次 all-gather 或 reduce-scatter 约要 ${A.ui.math('\\frac{\\text{整份数据的字节数}}{\\text{一台机器的网络带宽}}')}，all-reduce 是它的两倍。HSDP 的 all-gather 和 reduce-scatter 都走 NVLink。</li>`+
 `<li>计算：开了重算，每个 token 要算 8 × 参数量次（多一次前向），按假设的“算的时候能到峰值的多少”算时间。前向占四分之一，反向连同重算占四分之三。MFU 是结果，只按 6 × 参数量算。</li></ul></div>`+
 `<div><div class="eyebrow">简化</div><ul>`+
 `<li>所有传输都按 BF16 算，每个值 2 字节，和 ZeRO 论文的假设一样，这样 ZeRO-3 要传的才是数据并行的 1.5 倍。torchtitan 和 Megatron-LM 默认用 FP32 做梯度的归约，梯度要传的字节数会翻倍。</li>`+
 `<li>只有一条传输流，同一层的几次传输一个接一个。真实的 NCCL 可以同时跑几个，还有别的算法（树形、NVLink 交换机上的），可能比这里快，也可能慢。</li>`+
 `<li>没有算注意力的运算。序列长 4,096 个 token 时，8B 的运算只多约 7%。到 128K，注意力约是其余部分的 2 倍（见应用场景 2）。</li>`+
 `<li>不算优化器那一步的时间、临时缓冲区和显存碎片。GPU 显存按厂商标的 80 GB 算，单位是 GB（10⁹ 字节）。</li></ul></div>`+
 `<div><div class="eyebrow">假设</div><ul>`+
 `<li>GPU 算的时候能跑到峰值 989.5 TFLOP/s 的 50%，这是默认值。单纯的 BF16 矩阵乘法约能到 80%（ml-engineering）。attention、归一化和较小的 kernel 达不到这么高。</li>`+
 `<li>每台机器的网络，8 张卡共用。每张卡一块 400 Gb/s 网卡时是 400 GB/s，和 torchtitan 的 H100 集群一样。整台机器只有一块 400 Gb/s 网卡时是 50 GB/s，用在第 6 部分和挑战里，是这个 Lab 的假设。100 Gb/s 以太网是 12.5 GB/s，用在应用场景 4，也是这个 Lab 的假设。</li>`+
 `<li>机器内 NVLink 每张卡 450 GB/s，这是 scaling book 引的 NVIDIA 的数。scaling book 也提到，实际很难超过 370 GB/s。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>每个参数 18 字节：Megatron-LM 的 ${cite(SRC.meg,'<code>dist_optimizer.md</code>')}（18；用分布式优化器时 ${A.ui.math('6 + \\frac{12}{d}')}）；Hugging Face 的 ${cite(SRC.hf,'<code>model_memory_anatomy.md</code>')}。</li>`+
 `<li>分片策略：DeepSpeed 的 ${cite(SRC.dsz,'ZeRO 教程')}和 ${cite(SRC.dscfg,'<code>deepspeed/runtime/zero/config.py</code>')}；PyTorch FSDP 的 ${cite(SRC.fsdp,'<code>torch/distributed/fsdp/api.py</code>')}（ShardingStrategy）和 ${cite(SRC.fsdp2,'<code>fully_shard</code> 文档')}。</li>`+
 `<li>传输：${cite(SRC.nccl,'nccl-tests 的 <code>doc/PERFORMANCE.md</code>')}（环形实现的系数）；${cite(SRC.mle,'ml-engineering 的模型并行一章')}（每步传参数量的 2 倍和 3 倍）；${cite(SRC.sbg,'scaling book 的 <code>gpus.md</code>')}（NVLink，跨机器的集合通信）。</li>`+
 `<li>torchtitan 和 Megatron-LM 默认用 FP32 做梯度的归约，见 torchtitan 的 ${cite(SRC.ttcfg,'<code>config/configs.py</code>')} 和 Megatron-LM 的 ${cite(SRC.megarg,'<code>arguments.py</code>')}。</li>`+
 `<li>计算和 MFU：${cite(SRC.mamf,'ml-engineering 的加速器一章')}（BF16 矩阵乘法约能到 H100 峰值的 80%）；${cite(SRC.tt,'torchtitan 的 Llama 3 H100 测试')}（机器之间每张卡 400 Gb/s；Llama 3.1 8B 用 FSDP 在 8 张或 128 张 H100 上，MFU 33% 到 39%；128K 时用上下文并行）；${cite(SRC.megr,'Megatron-LM 的 README')}（H100 上 MFU 41% 到 47%）。</li>`+
 `<li>峰值 989.5 TFLOP/s（H100 SXM 稠密 BF16）和模型形状与本站其他 Lab 相同（来自各模型的 config.json）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04（链接固定在当天读到的提交上；PyTorch 的 <code>api.py</code>、<code>_fully_shard.py</code>，torchtitan 的 <code>configs.py</code> 和 DeepSpeed 的 <code>config.py</code> 读的是主分支）。原型，尚未经专家审阅。</p></div></div>`);
const PRIMER=()=>tx(`<p>Training keeps 18 bytes for every parameter: the BF16 weights (2), the FP32 gradients (4), an FP32 copy of the weights (4) and Adam’s m and v (4 each). The last three together are the optimizer state. On top of that come the activations and the logits, which grow with the tokens each GPU processes.</p><p>Time here is work divided by speed, as in the lab “${labLink('roofline')}”:</p>${M('\\text{compute time} = \\frac{\\text{FLOPs}}{\\text{the GPU’s FLOP/s}}')}${M('\\text{transfer time} = \\frac{\\text{bytes}}{\\text{the link’s bandwidth}}')}<p>The lab “${labLink('training-memory')}” takes the 18 bytes apart one by one.</p>`,
 `<p>训练时每个参数要存 18 字节：BF16 权重（2）、FP32 梯度（4）、FP32 权重副本（4），以及 Adam 的 m 和 v（各 4）。后三样合起来叫优化器状态。此外还有激活和 logits，它们随每张卡处理的 token 数增长。</p><p>这里的时间都是用量除以速度算出来的，和“${labLink('roofline')}”那个 Lab 讲的一样：</p>${M('\\text{计算时间} = \\frac{\\text{计算量}}{\\text{GPU 的 FLOP/s}}')}${M('\\text{传输时间} = \\frac{\\text{数据量}}{\\text{链路带宽}}')}<p>“${labLink('training-memory')}”那个 Lab 会把这 18 字节一样一样拆开讲。</p>`);
const OUTCOME=()=>tx('work out what each GPU holds under data parallelism, each ZeRO stage and FSDP, and tell whether the extra transfers will cost time.','算出数据并行、各个 ZeRO 阶段和 FSDP 下每张卡要存多少，并判断多传的数据会不会拖慢训练。');
const EXITS=()=>{
  const z2=mem({...BASE,N:16,st:'z2'}), z28=mem({...BASE,N:8,st:'z2'}), d8=comm({...BASE,N:8,st:'dp'}), d64=comm({...BASE,N:64,st:'dp'});
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does ZeRO-3 move 1.5 times the data of plain data parallelism?','为什么 ZeRO-3 要传的数据是数据并行的 1.5 倍？'),
   o:[tx('Each layer’s weights are gathered twice, before the forward pass and again before the backward pass, and the gradients are reduce-scattered once; each of the three moves half of what the gradients’ all-reduce moves.','每层的权重要取两次，前向之前一次、反向之前一次，梯度再 reduce-scatter 一次。这三次，每次都是梯度 all-reduce 的一半。'),
      tx('With the weights split, every GPU must also send the optimizer state to the others after each update, which adds half the data.','权重分开后，每次更新完，每张卡还要把优化器状态发给别的卡，多出了一半的数据。'),
      tx('ZeRO-3 sends the weights in FP32 instead of BF16, so each transfer is larger.','ZeRO-3 用 FP32 而不是 BF16 传权重，所以每次传得更多。')],a:0,
   n:[null,tx('The optimizer state never travels: each GPU updates only its own share and keeps the optimizer state for it. What travels are the BF16 weights and gradients.','优化器状态从来不传。每张卡只更新自己那份参数，也只存这份的优化器状态。传的是 BF16 的权重和梯度。'),
     tx('The weights travel in BF16, 2 bytes each, like the gradients; the extra half comes from gathering them a second time before the backward pass.','权重按 BF16 传，每个 2 字节，和梯度一样。多出的一半，来自反向前再取一次权重。')],
   why:tx('An all-reduce is a reduce-scatter plus an all-gather. Plain data parallelism does one all-reduce of the gradients: two halves. ZeRO-3 frees each layer’s weights after use, so it gathers them in the forward pass and again in the backward pass, then reduce-scatters the gradients: three halves.','一次 all-reduce 就是一次 reduce-scatter 加一次 all-gather。数据并行对梯度做一次 all-reduce，相当于两个“一半”。ZeRO-3 用完就释放每层的权重，所以前向取一次、反向再取一次，再加梯度的一次 reduce-scatter，一共三个“一半”。')},
  {q:tx('Why does ZeRO-2 save memory over ZeRO-1 without moving any more data?','为什么 ZeRO-2 比 ZeRO-1 省显存，却不多传数据？'),
   o:[tx('With ZeRO-1 the gradients already leave each GPU by reduce-scatter, and each GPU uses only its own share; ZeRO-2 just frees the shares it no longer needs.','ZeRO-1 的梯度本来就用 reduce-scatter 传出去，每张卡只用得上自己那一份。ZeRO-2 只是把用不上的部分释放掉。'),
      tx('ZeRO-2 compresses the gradients to 8 bits before sending them, so the same transfer carries more.','ZeRO-2 先把梯度压成 8 位再传，所以同样的传输能带更多。'),
      tx('ZeRO-2 sends the gradients only every other step.','ZeRO-2 每隔一步才传一次梯度。')],a:0,
   n:[null,tx('Nothing is compressed: the gradients travel exactly as with ZeRO-1, in the same reduce-scatter.','没有任何压缩。梯度和 ZeRO-1 时一模一样，用同样的 reduce-scatter 传。'),
     tx('Every step still averages every gradient; skipping steps would change the training itself.','每一步都还是要把所有梯度取平均。隔步才传，训练本身就变了。')],
   why:tx('The reduce-scatter that ZeRO-1 uses already gives each GPU the summed gradients of its own share only. The full gradients a GPU computed are useless once they are sent, so ZeRO-2 frees them layer by layer: less memory, the same transfers.','ZeRO-1 用的 reduce-scatter 本来就只给每张卡它那一份梯度的加和。一张卡自己算出的完整梯度，传出去以后就没用了，所以 ZeRO-2 一层一层把它们释放掉。显存少了，传输还是那些。')}],
 transfer:[
  {q:tx(`On 8 GPUs, ZeRO-2 leaves ${fGB(z28.states)} of model states on each GPU for Llama‑3.1‑8B. With the same ZeRO-2 on 16 GPUs, about how much is left on each?`,`用 ZeRO-2 训练 Llama‑3.1‑8B，8 张卡时每张卡的模型状态是 ${fGB(z28.states)}。同样用 ZeRO-2，换成 16 张卡，每张卡大约剩多少？`),
   o:[tx('About 16 GB: half as much','约 16 GB：少一半'),tx('About 24 GB: the weights do not shrink','约 24 GB：权重那部分不变'),tx('About 32 GB: the same','约 32 GB：不变')],a:1,
   why:tx(`ZeRO-2 keeps the 2 bytes of weights whole on every GPU, about 16 GB, however many GPUs there are. Only the gradients and the optimizer state, 16 bytes per parameter, are split, and doubling the GPUs halves that part. So each GPU keeps ${fGB(z2.states)}, not half of ${fGB(z28.states)}.`,`ZeRO-2 每张卡还存着完整的权重，每个参数 2 字节，约 16 GB，卡再多也不变。只有梯度和优化器状态（每个参数 16 字节）被分开，卡数翻倍，这部分减半。所以每张卡剩 ${fGB(z2.states)}，而不是 ${fGB(z28.states)} 的一半。`)},
  {q:tx(`With data parallelism on 8 GPUs, each GPU sends ${fGB(d8.bytes)} per step for Llama‑3.1‑8B. On 64 GPUs, about how much does each GPU send?`,`用数据并行训练 Llama‑3.1‑8B，8 张卡时每张卡每步发出 ${fGB(d8.bytes)}。换成 64 张卡，每张卡每步大约发出多少？`),
   o:[tx('About 3.5 GB: one eighth','约 3.5 GB：是原来的 1/8'),tx('About the same, a little more','和原来差不多，略多一点'),tx('About 225 GB: eight times as much','约 225 GB：是原来的 8 倍')],a:1,
   why:tx(`In a ring all-reduce each GPU sends ${A.ui.math('\\frac{2(d - 1)}{d}')} times the gradients’ size: 7/8 of twice the size on 8 GPUs, 63/64 of it on 64. So each GPU sends ${fGB(d64.bytes)}, a little more than ${fGB(d8.bytes)}. Adding GPUs does not reduce what each one has to move.`,`环形 all-reduce 里，每张卡发出梯度大小的 ${A.ui.math('\\frac{2(d - 1)}{d}')} 倍。8 张卡时是两倍大小的 7/8，64 张卡时是 63/64。所以每张卡发出 ${fGB(d64.bytes)}，比 ${fGB(d8.bytes)} 略多。卡多了，每张卡要传的数据并不会变少。`)}],
 practice:[
  {id:'distributed-training',d:tx('The part “Memory per accelerator”: why the data-parallel replicas should not each keep the full optimizer state, what sharding it (ZeRO, FSDP) saves, and the extra all-gather of weights that sharding the weights too costs. The note counts 16 bytes per parameter (BF16 gradients); this lab counts 18.','“每块加速器的显存”那一段：为什么数据并行的各个副本不必各存一整份优化器状态，把它分片（ZeRO、FSDP）能省多少，以及连权重也分片要多付的那次权重 all-gather。笔记按每个参数 16 字节算（梯度用 BF16），这个 Lab 按 18 字节算。')},
  {id:'matmul-roofline',d:tx('Part 4: compare token sharding, which keeps a full copy of the weights on every device like data parallelism, with tensor sharding and its ring all-reduce of '+A.ui.math('\\frac{2(p - 1)}{p}')+', and say which is compute-, memory- or communication-bound.','第 4 部分：比较 token 切分（每块设备各存一整份权重，和数据并行一样）与张量切分（环形 all-reduce 发出 '+A.ui.math('\\frac{2(p - 1)}{p}')+'），判断各自是计算、显存还是通信受限。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('ZeRO and FSDP','ZeRO 与 FSDP'),title:l?L(l.q):tx('How do you train a model that does not fit on one GPU?','一张卡放不下的模型，怎么训练？'),
    minutes:15,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: training memory','前置知识：训练显存'),primerPop:PRIMER(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage zr-stage" id="zr-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="zr-st-title"></div><div class="meter-sub" id="zr-st-sub"></div></div><div class="stage-actions"><span class="status" id="zr-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph zr-graph" id="zr-graph">`+
    `<div class="zr-membox"><div class="zr-cap" id="zr-mem-cap"></div><div class="zr-cols" id="zr-cols"></div><div class="zr-leg" id="zr-mem-leg"></div></div>`+
    `<div class="zr-tlbox" id="zr-tlbox"><div class="zr-cap" id="zr-tl-cap"></div><div class="zr-tl" id="zr-tl"></div><div class="zr-now" id="zr-now" aria-live="off"></div><div class="zr-leg" id="zr-tl-leg"></div><div class="zr-player" id="zr-player"></div></div>`+
   `</div>`+
   `<div class="metrics zr-metrics" id="zr-metrics"></div>`+
   `<details class="math" id="zr-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="zr-mathbody"></div></details></div>`;
}
const NOTE={
 u:()=>esc(TIP.share())+tx(' ','')+tx(`The higher it is, the shorter the computation, and the less time there is to hide the transfers. The ${DEF('MFU',TIP.mfu())} that training reports quote is a result, not this assumption.`,
          `它越高，计算越短，能藏住传输的时间就越少。训练报告里常说的 ${DEF('MFU',TIP.mfu())} 是结果，不是这个假设。`),
 net:()=>`<ul class="zr-netl">`+
   `<li>${tx('400 GB/s: one 400 Gb/s NIC per GPU, as in torchtitan’s H100 cluster.','400 GB/s：每张卡一块 400 Gb/s 的网卡，和 torchtitan 的 H100 集群一样。')}</li>`+
   `<li>${tx('50 GB/s: a single 400 Gb/s NIC for the whole machine.','50 GB/s：整台机器只有一块 400 Gb/s 的网卡。')}</li>`+
   `<li>${tx('12.5 GB/s: ordinary 100 Gb/s Ethernet.','12.5 GB/s：普通的 100 Gb/s 以太网。')}</li></ul>`+
   tx('The 8 GPUs of a machine share it. Inside a machine, NVLink gives each GPU 450 GB/s (NVIDIA’s figure, as the scaling book gives it).','一台机器的 8 张卡共用这些带宽。机器内部走 NVLink，每张卡 450 GB/s（NVIDIA 给的数，见 scaling book）。'),
 T:()=>tx('Tokens each GPU processes per step. The activations, the logits and the computation grow with it. The data to move does not.','每张卡每步处理的 token 数。激活、logits 和计算都随它增长，要传的数据不随它变。')
};
function controlsHTML(){
  const opt=(v,t)=>[v,t];
  return `<aside class="panel controls" id="zr-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="zr-lock" hidden></p>`+
   `<div class="lockable" id="zr-ctl-m-wrap">${A.ui.seg({id:'zr-ctl-m',key:'m',label:tx('Model','模型'),options:[opt('8b',MOD['8b'].name),opt('70b',MOD['70b'].name)],value:S.m})}</div>`+
   `<div class="lockable" id="zr-ctl-N-wrap">${A.ui.seg({id:'zr-ctl-N',key:'N',label:tx('H100 GPUs (8 per machine)','H100 卡数（每台机器 8 张）'),options:NS.map(n=>[String(n),String(n)]),value:String(S.N)})}</div>`+
   `<div class="lockable" id="zr-ctl-st-wrap">${A.ui.seg({id:'zr-ctl-st',key:'st',label:tx('How to split','分法'),options:STS.map(s=>[s,stName(s)]),value:S.st,note:tx('Data parallel keeps a full copy on every GPU. ZeRO-1 splits the optimizer state, ZeRO-2 also the gradients, ZeRO-3 also the weights. HSDP runs ZeRO-3 inside each machine and keeps one copy per machine.','数据并行在每张卡上存一整份。ZeRO-1 分优化器状态，ZeRO-2 再分梯度，ZeRO-3 再分权重。HSDP 在机器内做 ZeRO-3，每台机器存一整份。')})}</div>`+
   `<div class="lockable" id="zr-ctl-T-wrap">${A.ui.range({id:'zr-ctl-T',key:'T',label:tx('Tokens per GPU per step','每张卡每步的 token 数'),min:0,max:TS.length-1,step:1,value:TS.indexOf(S.T),out:fTok(S.T),note:NOTE.T()})}</div>`+
   `<div class="lockable" id="zr-ctl-ov-wrap">${A.ui.seg({id:'zr-ctl-ov',key:'ov',label:tx('Transfers during computation','传输和计算同时进行'),options:[['1',tx('Yes (prefetch, buckets)','是（预取、边算边传）')],['0',tx('No','否')]],value:String(S.ov)})}</div>`+
   `<div class="lockable" id="zr-ctl-net-wrap">${A.ui.seg({id:'zr-ctl-net',key:'net',label:tx('Network per machine','每台机器的网络'),options:[['ib','400 GB/s'],['one','50 GB/s'],['eth',tx('12.5 GB/s (Ethernet)','12.5 GB/s（以太网）')]],value:S.net,note:NOTE.net()})}</div>`+
   `<div class="zr-grp lockable" id="zr-ctl-as"><div class="zr-gh">${tx('Assumed value','假设值')}</div>`+
    A.ui.range({id:'zr-ctl-u',key:'u',label:tx('Share of the peak while computing','算的时候能到峰值的多少'),min:0,max:US.length-1,step:1,value:US.indexOf(S.u),out:S.u+'%',note:NOTE.u()})+`</div>`+
   `<div class="lockable" id="zr-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`# Per-GPU memory and traffic for data parallel and ZeRO 1-3.
N = 8_030_261_248          # Llama-3.1-8B parameters
d = 8                      # data-parallel GPUs

def model_states(stage, N, d):
    w, g, o = 2 * N, 4 * N, 12 * N       # BF16 weights, FP32 grads, optimizer state
    if stage >= 1: o /= d                # ZeRO-1: optimizer state
    if stage >= 2: g /= d                # ZeRO-2: gradients too
    if stage >= 3: w /= d                # ZeRO-3: weights too
    return w + g + o

def bytes_sent(stage, N, d):
    ring = (d - 1) / d                   # all-gather, reduce-scatter
    halves = 3 if stage == 3 else 2      # all-reduce = 2 halves
    return halves * 2 * N * ring         # BF16: 2 bytes per value

for s in range(4):
    print(s, round(model_states(s, N, d) / 1e9, 1),
          round(bytes_sent(s, N, d) / 1e9, 1))
# 0 144.5 28.1 / 1 60.2 28.1 / 2 32.1 28.1 / 3 18.1 42.2`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="zr-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="zr-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Each GPU’s model states and the bytes it sends per step, for data parallel (stage 0) and ZeRO-1, -2 and -3.','每张卡的模型状态和每步发出的字节数，分别算数据并行（stage 0）和 ZeRO-1、2、3。')}</p><pre class="code" id="zr-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="zr-prod-card"><h3>${A.t('cProd')}</h3><ul class="zr-prod">`+
   tx(`<li>PyTorch FSDP turns the stages into sharding strategies (${cite(SRC.fsdp,'<code>torch/distributed/fsdp/api.py</code>')}):<ul class="zr-sub">`+
        `<li><code>FULL_SHARD</code> (ZeRO-3): parameters, gradients and optimizer state are all sharded. The weights are gathered before the forward and again before the backward, and the gradients are reduce-scattered.</li>`+
        `<li><code>SHARD_GRAD_OP</code> (ZeRO-2): the full parameters stay from the forward through the backward.</li>`+
        `<li><code>NO_SHARD</code> (data parallel): nothing is sharded, and the gradients are all-reduced.</li>`+
        `<li><code>HYBRID_SHARD</code> (HSDP): <code>FULL_SHARD</code> inside a machine, a copy per machine.</li></ul></li>`+
      `<li>FSDP2’s <code>fully_shard</code> (${cite(SRC.fsdp2py,'<code>_fully_shard.py</code>')}) replaces the strategies with <code>reshard_after_forward</code>. <code>True</code> frees the weights after the forward and gathers them again in the backward. <code>False</code> keeps them. torchtitan’s ${cite(SRC.ttf,'<code>docs/fsdp.md</code>')} maps them to ZeRO-3 and ZeRO-2, and a 2D mesh to HSDP.</li>`+
      `<li>DeepSpeed picks the stage with <code>zero_optimization.stage</code>, 0 to 3. With ZeRO-3, <code>overlap_comm</code> is on by default, so the gradient reduction runs during the backward. <code>stage3_prefetch_bucket_size</code> defaults to 5e7 elements and sets the size of the prefetch buffer. These defaults come from ${cite(SRC.dscfg,'<code>deepspeed/runtime/zero/config.py</code>')}; the ${cite(SRC.dsc,'config docs')} still give older values.</li>`+
      `<li>torchtitan trains Llama 3.1 70B on 256 H100s as FSDP 32 × tensor parallel 8, and 405B on 512 with pipeline parallelism added (${cite(SRC.tt,'benchmarks')}). With hundreds of GPUs and sharding alone, each GPU must still process enough tokens per step to hide the transfers. Times hundreds of GPUs, the global batch gets too large (${cite(SRC.sbg,'scaling book')}). Tensor parallelism lowers the number of data-parallel groups.</li>`,
      `<li>PyTorch FSDP 把这几个阶段做成了分片策略（${cite(SRC.fsdp,'<code>torch/distributed/fsdp/api.py</code>')}）。<ul class="zr-sub">`+
        `<li><code>FULL_SHARD</code>（ZeRO-3）：参数、梯度和优化器状态都分片。前向之前和反向之前各取一次权重，梯度用 reduce-scatter。</li>`+
        `<li><code>SHARD_GRAD_OP</code>（ZeRO-2）：从前向到反向一直留着完整的参数。</li>`+
        `<li><code>NO_SHARD</code>（数据并行）：不分片，梯度用 all-reduce。</li>`+
        `<li><code>HYBRID_SHARD</code>（HSDP）：机器内 <code>FULL_SHARD</code>，机器之间复制。</li></ul></li>`+
      `<li>FSDP2 的 <code>fully_shard</code>（${cite(SRC.fsdp2py,'<code>_fully_shard.py</code>')}）用 <code>reshard_after_forward</code> 代替这些策略。设成 <code>True</code>，前向后释放权重，反向时再取。设成 <code>False</code>，权重一直留着。torchtitan 的 ${cite(SRC.ttf,'<code>docs/fsdp.md</code>')} 把这两种设置分别对应到 ZeRO-3 和 ZeRO-2，二维的 mesh 对应 HSDP。</li>`+
      `<li>DeepSpeed 用 <code>zero_optimization.stage</code> 选阶段，取 0 到 3。用 ZeRO-3 时，<code>overlap_comm</code> 默认打开，梯度的归约和反向同时进行。<code>stage3_prefetch_bucket_size</code> 默认 5e7 个元素，决定预取的缓冲区多大。这些默认值以 ${cite(SRC.dscfg,'<code>deepspeed/runtime/zero/config.py</code>')} 为准，${cite(SRC.dsc,'配置文档')}里还是旧值。</li>`+
      `<li>torchtitan 在 256 张 H100 上训练 Llama 3.1 70B，用的是 FSDP 32 × 张量并行 8。405B 用 512 张卡，还加了流水线并行（${cite(SRC.tt,'测试结果')}）。卡多到几百张时，如果只靠分片，每张卡每步都要处理足够多的 token 才藏得住传输。乘上几百张卡，总的 batch 就太大了（${cite(SRC.sbg,'scaling book')}）。加上张量并行，数据并行的路数就降下来了。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.zero,'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models')}<span>${tx('Rajbhandari et al., 2020 (arXiv 1910.02054): the three stages of sharding.','Rajbhandari 等，2020（arXiv 1910.02054），提出了分片的三个阶段。')}</span></li>`+
   `<li>${cite(SRC.fsdpp,'PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel')}<span>${tx('Zhao et al., 2023 (arXiv 2304.11277): how PyTorch implements and overlaps it.','Zhao 等，2023（arXiv 2304.11277），讲 PyTorch 怎样实现它、怎样让传输和计算重叠。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="zr-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="zr-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="zr-app m2" id="zr-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="zr-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much memory each GPU needs and how much of the time it waits for data.','选一个场景，看看每张卡要放多少显存、有多少时间在等数据。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="zr-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Train a 70B on 64 GPUs without waiting','用 64 张卡训练 70B，又不让卡干等')}</h3>`+
   `<p>${tx('You are to train Llama‑3.1‑70B on 64 H100s: 8 machines with 8 GPUs each. Choose how to split the model states and how many tokens each GPU processes per step. Two conditions: each GPU must fit in 80 GB, and it may wait for data at most 10% of each step. Among the plans that meet both, pick the one with the shortest step.',
            '你要用 64 张 H100 训练 Llama‑3.1‑70B，一共 8 台机器，每台 8 张卡。请选好模型状态怎么分，以及每张卡每步处理多少 token。有两个条件：每张卡放得下 80 GB，等数据的时间不超过每步的 10%。在满足两个条件的方案里，选每步最短的。')}</p>`+
   `<ul class="zr-asm small"><li>${tx('Llama‑3.1‑70B has 70.6B parameters in 80 layers. Each parameter needs 18 bytes of model states.','Llama‑3.1‑70B 有 70.6B 个参数、80 层，每个参数要存 18 字节的模型状态。')}</li>`+
   `<li>${tx('The network of these 8 machines is slow. Each machine has a single 400 Gb/s NIC, so its 8 GPUs share 50 GB/s.','这 8 台机器的网络不快。每台机器只有一块 400 Gb/s 的网卡，8 张卡共用 50 GB/s。')}</li>`+
   `<li>${tx(`While computing, a GPU reaches ${DEF('50% of its peak',TIP.share())}. Transfers run during the computation, with prefetching on.`,`GPU ${DEF('算的时候能跑到峰值的 50%',TIP.share())}。传输和计算同时进行，也开了预取。`)}</li>`+
   `<li>${tx('Activations assume recomputation, as in the rest of this lab. The logits are kept in FP32.','激活和这个 Lab 的其他部分一样，按开了重算来计算。logits 用 FP32 存。')}</li></ul>`+
   A.ui.seg({id:'zr-ic-chs',key:'chs',label:tx('How to split','分法'),options:CH.sts.map(s=>[s,stName(s)]),value:S.st,inline:true})+
   A.ui.seg({id:'zr-ic-cht',key:'cht',label:tx('Tokens per GPU per step','每张卡每步的 token 数'),options:CH.Ts.map(T=>[String(T),fTok(T)]),value:String(S.T),inline:true})+
   `<div class="goal mono zr-goal" id="zr-ch-g1"><span>${tx('Memory per GPU','每张卡的显存')}</span><span><b id="zr-ch-m"></b>${tx(' (at most 80 GB)','（不超过 80 GB）')}</span></div>`+
   `<div class="goal mono zr-goal" id="zr-ch-g2"><span>${tx('Waiting for data','等数据的时间')}</span><span><b id="zr-ch-w"></b>${tx(' (at most 10%)','（不超过 10%）')}</span></div>`+
   `<div class="goal mono zr-goal" id="zr-ch-g3"><span>${tx('Time per step','每步的时间')}</span><span><b id="zr-ch-t"></b>${tx(' (shorter is better)','（越短越好）')}</span></div>`+
   submit+`<div id="zr-ch-res" class="zr-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},dp:{en:'Let the step finish playing to continue (Fast takes a few seconds).',zh:'等这一步播完才能继续（选“快”只要几秒）。'},
    z1:{en:'Split the optimizer state 8 ways to continue.',zh:'把优化器状态分成 8 份，才能继续。'},z2:{en:'Split the gradients 8 ways to continue.',zh:'把梯度分成 8 份，才能继续。'},
    z3:{en:'Try “Prefetch the next layer” to continue.',zh:'试一下“提前取下一层（预取）”，才能继续。'},net:{en:'Try one of the two ways to improve it to continue.',zh:'再试一种改进的办法，才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#zr-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#zr-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(s.id==='dp'?watched.has(sigOf(P2)):true);
    $$('#zr-ic-lv5 [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv5)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lv5Tried.has(v)||v===lv5?lvRes5(v):'';});
    $$('#zr-ic-lv6 [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv6)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lv6Tried.has(v)||v===lv6?lvRes6(v):'';});
    $$('.seg-ctl[data-key="st13"],.seg-ctl[data-key="st24"]',rail).forEach(g=>A.ui.setPressed(g,S.st));
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    $$('[data-nav="showme"]',rail).forEach(sm=>{sm.hidden=!!observed[step];});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#zr-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=evalC(chCfg(S.st,S.T)), m=$('#zr-ch-m',rail), w=$('#zr-ch-w',rail), t=$('#zr-ch-t',rail); if(!m) return;
    m.textContent=fGB(e.me.total); w.textContent=fPctW(e.s.share); t.textContent=fSec(e.s.step);
    $('#zr-ch-g1',rail).classList.toggle('met',e.me.fits); $('#zr-ch-g1',rail).classList.toggle('zr-over',!e.me.fits);
    $('#zr-ch-g2',rail).classList.toggle('met',e.s.share<=CH.wait); $('#zr-ch-g2',rail).classList.toggle('zr-over',e.s.share>CH.wait);
    $$('.seg-ctl[data-key="chs"]',rail).forEach(g=>A.ui.setPressed(g,S.st)); $$('.seg-ctl[data-key="cht"]',rail).forEach(g=>A.ui.setPressed(g,String(S.T)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#zr-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);
      const dd=$('details.zr-all',res); if(dd){dd.open=chAllOpen; dd.addEventListener('toggle',()=>{chAllOpen=dd.open;});}}
  }
}

/* =====================================================================
   stage: memory columns
   ===================================================================== */
const hidePred=()=>mode==='guided'&&isP('mission');
const SCALE=160e9;                              // the columns show 0–160 GB; 80 GB is the GPU
function memCols(){
  const host=$('#zr-cols',app), me=mem(S), n=Math.min(S.N,8), hide=hidePred();
  const pct=b=>Math.max(0,Math.min(100,b/SCALE*100));
  const blocks=[{k:'w',b:me.w,cls:'zr-b-w',sh:me.sh[0]},{k:'gr',b:me.gr,cls:'zr-b-g',sh:me.sh[1]},{k:'cp',b:me.cp,cls:'zr-b-c',sh:me.sh[2]},{k:'mm',b:me.mm,cls:'zr-b-m',sh:me.sh[2]},{k:'vv',b:me.vv,cls:'zr-b-v',sh:me.sh[2]},{k:'inf',b:me.inf,cls:'zr-b-inf',sh:1},{k:'act',b:me.act,cls:'zr-b-act',sh:1},{k:'lg',b:me.lg,cls:'zr-b-lg',sh:1}].filter(x=>x.b>0);
  /* integrator decision 6: one name for each block, in the legend too */
  const name={w:tx('BF16 weights','BF16 权重'),gr:tx('FP32 gradients','FP32 梯度'),cp:tx('FP32 copy','FP32 权重副本'),mm:tx('Adam’s m','Adam 的 m'),vv:tx('Adam’s v','Adam 的 v'),inf:tx('weights in flight','正在用的权重'),act:tx('activations','激活'),lg:'logits'};
  let cols='';
  for(let g=0;g<n;g++){
    let y=0, bl='';
    if(!hide) blocks.forEach(x=>{const h=pct(x.b), lab=x.sh>1?(S.st==='hsdp'?`${(g%8)+1}/${x.sh}`:`${g+1}/${x.sh}`):'';
      const tip=x.sh>1?tx(`GPU ${g+1} keeps only its own 1/${x.sh} of the ${name[x.k]}, ${fGB(x.b)}.`,`GPU ${g+1} 只存${name[x.k]}里属于自己的 1/${x.sh}，共 ${fGB(x.b)}。`)
        :(x.k==='inf'?tx(`The full weights of the layers being used right now, ${fGB(x.b)}. ZeRO-3 gathers them before a layer runs and frees them after.`,`正在用的几层的完整权重，共 ${fGB(x.b)}。ZeRO-3 在一层运行前把它们取齐，用完就释放。`)
         :(x.k==='act'?tx(`Activations of ${fTok(S.T)} tokens, ${fGB(x.b)}. Every GPU keeps its own, and sharding does not split them.`,`${fTok(S.T)} 个 token 的激活，共 ${fGB(x.b)}。每张卡存自己的那份，分片分不了它们。`)
          :(x.k==='lg'?tx(`The logits of ${fTok(S.T)} tokens in FP32, ${fGB(x.b)}. Every GPU keeps its own, and sharding does not split them.`,`${fTok(S.T)} 个 token 的 logits，用 FP32 存，共 ${fGB(x.b)}。每张卡存自己的那份，分片也分不了它们。`)
           :tx(`GPU ${g+1} holds all of the ${name[x.k]}, ${fGB(x.b)}, the same as every other GPU.`,`GPU ${g+1} 上有一整份${name[x.k]}，共 ${fGB(x.b)}，和其他每张卡上的一模一样。`))));
      bl+=`<i class="zr-blk ${x.cls}${x.sh>1?' sh':''}" style="bottom:${y.toFixed(2)}%;height:${h.toFixed(2)}%" data-tipx="${esc(tip)}">${lab&&h>=6?`<em>${lab}</em>`:''}</i>`; y+=h;});
    const over=!hide&&me.total>CAP, top=pct(me.total);
    cols+=`<div class="zr-col${over?' over':''}"><div class="zr-colin">${bl}${over?`<b class="zr-ovr" style="bottom:${pct(CAP).toFixed(2)}%;height:${Math.max(0,top-pct(CAP)).toFixed(2)}%"></b>`:''}${hide?`<span class="zr-q">?</span>`:''}${!hide&&me.total>SCALE?`<span class="zr-clip">▲ ${fG(me.total)}</span>`:''}</div><span class="zr-gl mono">GPU ${g+1}</span></div>`;
  }
  const line=`<div class="zr-cap80" style="bottom:calc(${pct(CAP).toFixed(2)}% + 22px)"><span>${tx('80 GB: one H100','80 GB：一张 H100')}</span></div>`;
  const more=S.N>8?`<div class="zr-more">${tx(`× ${S.N/8} machines: ${S.N} GPUs; the other ${S.N-8} hold the same`,`× ${S.N/8} 台机器，共 ${S.N} 张卡，另外 ${S.N-8} 张存的一样`)}</div>`:'';
  const h=`<div class="zr-colsin">${cols}${line}</div>${more}`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
  $('#zr-mem-cap',app).textContent=hide?tx(`What each of the ${n} GPUs holds: answer the question in the card first`,`8 张卡各存多少：先回答卡片里的问题`)
    :tx(`What each GPU holds (memory per GPU, ${me.fits?'within':'over'} 80 GB)`,`每张卡存了什么（每张卡的显存${me.fits?'在 80 GB 以内':'超过了 80 GB'}）`);
  /* top to bottom, as the blocks are stacked in the columns beside it */
  const lg=[];
  lg.push(['zr-sw-ovr',tx('over 80 GB: does not fit','超过 80 GB：放不下'),tx('The part above the 80 GB line does not fit on the GPU.','80 GB 线以上的部分，GPU 放不下。')]);
  lg.push(['zr-sw-sh',tx('3/8: this GPU’s share','3/8：这张卡的那一份'),tx('A block marked 3/8 is split 8 ways, and this GPU keeps the 3rd piece.','标着 3/8 的块被分成了 8 份，这张卡存第 3 份。')]);
  lg.push(['zr-sw-lg','logits',TIP.lg()]);
  lg.push(['zr-sw-act',tx('activations','激活'),TIP.act()]);
  if(me.inf>0) lg.push(['zr-sw-inf',tx('weights in flight','正在用的权重'),tx('ZeRO-3 gathers the full weights of the layer about to run (and, with prefetching, the next one) and frees them after use.','ZeRO-3 把马上要算的那一层（预取时还有下一层）的完整权重取齐，用完就释放。')]);
  lg.push(['zr-sw-v',tx('Adam’s v','Adam 的 v'),tx('Adam’s second moment, a running average of the squared gradients. FP32, 4 bytes per parameter, part of the optimizer state.','Adam 的二阶矩，是梯度平方的滑动平均。它用 FP32 存，每个参数 4 字节，属于优化器状态。')]);
  lg.push(['zr-sw-m',tx('Adam’s m','Adam 的 m'),tx('Adam’s first moment, a running average of the gradients. FP32, 4 bytes per parameter, part of the optimizer state.','Adam 的一阶矩，是梯度的滑动平均。它用 FP32 存，每个参数 4 字节，属于优化器状态。')]);
  lg.push(['zr-sw-c',tx('FP32 copy of the weights','FP32 权重副本'),tx('An FP32 copy of each weight, 4 bytes per parameter. Each update is applied to it. Part of the optimizer state.','每个权重再用 FP32 存一份，每个参数 4 字节。每次更新都加在这一份上。它属于优化器状态。')]);
  lg.push(['zr-sw-g',tx('FP32 gradients','FP32 梯度'),tx('The gradients in FP32, 4 bytes per parameter.','梯度用 FP32 存，每个参数 4 字节。')]);
  lg.push(['zr-sw-w',tx('BF16 weights','BF16 权重'),tx('The weights in BF16, 2 bytes per parameter.','权重用 BF16 存，每个参数 2 字节。')]);
  const lh=hide?'':lg.map(x=>`<span tabindex="0" data-tipx="${esc(x[2])}"><i class="sw ${x[0]}"></i>${esc(x[1])}</span>`).join('');
  const lgEl=$('#zr-mem-leg',app); if(lgEl._h!==lh){lgEl.innerHTML=lh; lgEl._h=lh;}
}

/* =====================================================================
   stage: the timeline of one step, and its player
   ===================================================================== */
const sigOf=c=>[c.m,c.N,c.st==='z2'?'z1':c.st,c.T,c.ov,c.N>8?c.net:'-',c.u].join('|');
let SIM=null;
function drawTimeline(){
  const box=$('#zr-tlbox',app), hide=hidePred(); box.hidden=hide; if(hide) return;
  const c={...S}, sg=sigOf(c);
  if(!SIM||SIM._sig!==sg){SIM=sim(c); SIM._sig=sg;}
  const s=SIM, T=s.step, P=x=>(x/T*100).toFixed(3);
  const bar=e=>`<i class="zr-e zr-e-${e.k}" style="left:${P(e.t0)}%;width:${P(e.t1-e.t0)}%"></i>`;
  const cs=s.ev.filter(e=>e.r==='c'), ns=s.ev.filter(e=>e.r==='n');
  const gaps=s.gaps.map(g=>`<i class="zr-e zr-wait" style="left:${P(g.t0)}%;width:${P(g.t1-g.t0)}%"></i>`).join('');
  const ticks=(()=>{const raw=T/4, p=Math.pow(10,Math.floor(Math.log10(raw))), st=[1,2,2.5,5,10].map(m=>m*p).find(v=>v>=raw)||raw; let h=''; for(let v=0;v<=T*1.0001;v+=st) h+=`<span style="left:${P(v)}%">${fSec(v).replace(/^0\.00 /,'0 ')}</span>`; return h;})();
  const h=`<div class="zr-rowlab">${tx('compute','计算')}</div><div class="zr-row zr-rc" data-tipx="${esc(tx(`The GPU’s computation, layer by layer: forward, then backward. Each layer of the backward first does its forward pass again. Red is where the GPU waits for data, ${fPctW(s.share)} of this step.`,`GPU 的计算一层一层地进行，先前向，再反向。反向的每一层先把自己的前向重算一遍。红色是 GPU 在等数据，占这一步的 ${fPctW(s.share)}。`))}">${cs.map(bar).join('')}${gaps}</div>`+
    `<div class="zr-rowlab">${tx('transfer','传输')}</div><div class="zr-row zr-rn" data-tipx="${esc(tx(`The transfers between GPUs: ${fGB(s.k.bytes)} sent by this GPU in this step, taking ${fSec(s.k.time)} in all.`,`GPU 之间的传输：这一步里这张卡发出 ${fGB(s.k.bytes)}，一共要 ${fSec(s.k.time)}。`))}">${ns.map(bar).join('')}</div>`+
    `<div class="zr-cur" id="zr-cur"></div><div class="zr-head" id="zr-head"></div><div class="zr-ax">${ticks}</div>`;
  const tl=$('#zr-tl',app); if(tl._h!==h){tl.innerHTML=h; tl._h=h;}
  $('#zr-tl-cap',app).textContent=tx(`One training step on one GPU (every GPU does the same): ${fSec(s.step)}, of which ${fPctW(s.share)} waiting for data`,`一张卡上的一步训练（每张卡都一样）：${fSec(s.step)}，其中 ${fPctW(s.share)} 在等数据`);
  const kinds=new Set(ns.map(e=>e.k)); const lg=[['zr-sw-F',tx('forward','前向'),tx('Forward pass of one layer.','一层的前向计算。')],['zr-sw-B',tx('backward','反向'),tx('Backward pass of one layer, with its forward pass done again first: about three times the forward.','一层的反向计算，包括先重算一遍这一层的前向，约是前向的三倍。')]];
  if(kinds.has('AG')||kinds.has('AGb')) lg.push(['zr-sw-AG','all-gather',TIP.ag()]);
  if(kinds.has('RS')) lg.push(['zr-sw-RS','reduce-scatter',TIP.rs()]);
  if(kinds.has('AR')) lg.push(['zr-sw-AR','all-reduce',TIP.ar()]);
  if(kinds.has('ARx')) lg.push(['zr-sw-ARx',tx('all-reduce across machines','跨机器 all-reduce'),tx('HSDP: each GPU all-reduces its 1/8 of the gradients with the GPUs holding the same share in the other machines.','HSDP：每张卡和其他机器上负责同一份的卡，对这 1/8 的梯度做 all-reduce。')]);
  lg.push(['zr-sw-wait',tx('waiting for data','等数据'),tx('The GPU has nothing to compute until a transfer finishes.','传输完成之前，GPU 没有可算的东西，只能等。')]);
  lg.push(['zr-sw-head',tx('playhead','播放位置'),tx('Where the replay is now.','回放现在播到的位置。')]);
  const lh=lg.map(x=>`<span tabindex="0" data-tipx="${esc(x[2])}"><i class="sw ${x[0]}"></i>${esc(x[1])}</span>`).join('');
  const lgEl=$('#zr-tl-leg',app); if(lgEl._h!==lh){lgEl.innerHTML=lh; lgEl._h=lh;}
  if(curSig!==sg){curSig=sg; pause(); if(watched.has(sg)) pT=T; else {pT=0; if(!relang) playSoon();}}
  pT=Math.min(pT,T); drawHead(); drawPlayer();
}
/* The caption above the legend (the owner's review of 2026-10-07: it flickered). It used to be one string rebuilt every frame, and
   its transfer half existed only while a transfer was in flight, about 1 ms of a 4–12 ms layer, so at Medium it showed for one or
   two frames per layer and vanished. Now it is two lines whose sentence is built once per phase (the forward pass, the backward
   pass, the end of the step) and kept while that phase lasts; during playback only the numbers in it change, in place, and the
   box has a fixed height (R7.9, R7.11). The transfer line names the latest transfer of its kind, so it never blinks off between
   two transfers. In a step that waits (1% or more), the time waited so far is a red number that grows as the playhead crosses
   the red gaps (R5.11), instead of a red sentence that flashed on for a frame or two. */
const secNum=t=>t<1?fixd(t,2):(t<10?fixd(t,1):fixd(t,0));
const NUMS=new Set(['t','l','w','k']);
function nowFrame(ph,showW){
  const Lr=MOD[S.m].L, st=S.st, n=k=>`<span class="zr-nn" data-n="${k}"></span>`;
  if(ph==='D'){
    const c=tx(`The step took ${n('t')} s, ${esc(fPctW(SIM.share))} of it waiting.`,`这一步用了 ${n('t')} 秒，其中 ${esc(fPctW(SIM.share))} 的时间在等数据。`);
    const m=showW?tx('Red is where this GPU waits: the transfers the computation did not hide.','红色是这张卡在等数据，也就是计算没藏住的传输。')
                 :tx('The transfers ran during the computation, so this GPU hardly waited.','传输和计算同时进行，所以这张卡几乎不用等。');
    return [c,m];}
  const pass=ph==='F'?tx('forward','前向'):tx('backward','反向');
  const c=tx(`${n('t')} s: ${pass}, layer ${n('l')}/${Lr}.`,`${n('t')} 秒：${pass}，第 ${n('l')} / ${Lr} 层。`)+
    (showW?`<span class="zr-waitn">${tx(` Waited ${n('w')} s.`,`已等数据 ${n('w')} 秒。`)}</span>`:'');
  let m;
  if(ph==='F') m=st==='dp'?tx('Transfers: none in the forward pass.','传输：前向时卡之间不传数据。')
    :((st==='z1'||st==='z2')?tx(`All-gather: layer ${n('k')}’s updated weights arrive from the other GPUs.`,`all-gather：从别的卡取来第 ${n('k')} 层更新后的权重。`)
      :tx(`All-gather: layer ${n('k')}’s weights arrive from the other GPUs.`,`all-gather：从别的卡取来第 ${n('k')} 层的权重。`));
  else m=st==='dp'?tx(`All-reduce: layer ${n('k')}’s gradients are averaged over all GPUs.`,`all-reduce：第 ${n('k')} 层的梯度在所有卡上求平均。`)
    :(st==='hsdp'?tx(`Reduce-scatter: layer ${n('k')}’s gradients in the machine, then an all-reduce across machines.`,`reduce-scatter：第 ${n('k')} 层的梯度先在机器内求和，再跨机器 all-reduce。`)
      :(st==='z3'?tx(`Reduce-scatter: layer ${n('k')}’s gradients. Weights are gathered again for the backward pass.`,`reduce-scatter：第 ${n('k')} 层的梯度，每张卡留下自己那份的加和。反向前，每层的权重还要再取一次。`)
        :tx(`Reduce-scatter: layer ${n('k')}’s gradients; each GPU keeps the sum of its share.`,`reduce-scatter：第 ${n('k')} 层的梯度，每张卡留下自己那份的加和。`)));
  return [c,m];
}
function nowNums(ph){
  const ev=SIM.ev, t=pT, o={t:secNum(ph==='D'?SIM.step:t)};
  if(ph==='D') return o;
  /* the layer this GPU computes, or, in a gap, the layer it waits to compute */
  const cs=ev.filter(e=>e.r==='c'), c=cs.find(e=>e.t0<=t&&t<e.t1)||cs.find(e=>e.t0>t)||cs[cs.length-1];
  o.l=String(c.i+1);
  /* the latest transfer of this phase's kind that has started, or the first one still to come */
  const kinds=ph==='F'?['AG']:(S.st==='dp'?['AR']:['RS']), ks=ev.filter(e=>e.r==='n'&&kinds.includes(e.k));
  let k=null; ks.forEach(e=>{if(e.t0<=t+1e-15) k=e;}); if(!k) k=ks[0];
  o.k=k?String(k.i+1):'';
  let w=0; SIM.gaps.forEach(g=>{if(g.t0<t) w+=Math.min(t,g.t1)-g.t0;}); o.w=secNum(w);
  return o;
}
function drawNow(){
  const now=$('#zr-now',app); if(!now||!SIM) return;
  const T=SIM.step, done=pT>=T-1e-12, ph=done?'D':(pT<SIM.fwdEnd-1e-15?'F':'B'), showW=SIM.share>=0.01;
  const key=[ph,S.st,showW?1:0,MOD[S.m].L,A.lang(),SIM._sig].join('|');
  if(now._key!==key){const [c,m]=nowFrame(ph,showW); now._key=key; now._gen=(now._gen||0)+1;
    now.innerHTML=`<div class="zr-nl zr-nl-c">${c}</div><div class="zr-nl zr-nl-n">${m}</div>`;
    now.dataset.phase=ph; now.dataset.gen=String(now._gen);}
  const o=nowNums(ph);
  now.querySelectorAll('[data-n]').forEach(e=>{const v=o[e.dataset.n]; if(v!=null&&NUMS.has(e.dataset.n)&&e.textContent!==v) e.textContent=v;});
}
function drawHead(){
  if(!SIM||!app) return; const T=SIM.step, cur=$('#zr-cur',app), hd=$('#zr-head',app); if(!cur) return;
  const x=T>0?pT/T*100:100; cur.style.left=x.toFixed(3)+'%'; cur.style.width=(100-x).toFixed(3)+'%'; hd.style.left=x.toFixed(3)+'%';
  hd.hidden=pT>=T&&!playing;
  drawNow();
}
function drawPlayer(){
  const pl=$('#zr-player',app); if(!pl||!SIM) return; const T=SIM.step;
  const sig=[playing,spd,A.lang(),pT>=T].join('|'); const scrub=pl.querySelector('input[data-key="scrub"]');
  if(scrub&&pl._built===A.lang()){scrub.value=Math.round(pT/T*1000); A.ui.setRange(scrub);
    if(pl._sig!==sig){pl._sig=sig; const pb=pl.querySelector('[data-act="play"]'); pb.textContent=playing?tx('❚❚ Pause','❚❚ 暂停'):(pT>=T?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放')); $$('.seg-ctl[data-key="spd"]',pl).forEach(g=>A.ui.setPressed(g,spd));}
    return;}
  pl._built=A.lang(); pl._sig=sig;
  pl.innerHTML=`<button type="button" class="btn ghost sm zr-play" data-act="play" data-tipx="${esc(tx('Play the step at the chosen speed; press again to pause. At the end, the button replays it.','按选定的速度播放这一步，再按一次就暂停。播完后，这个按钮会重放。'))}">${playing?tx('❚❚ Pause','❚❚ 暂停'):(pT>=T?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放'))}</button>`+
    `<button type="button" class="btn ghost sm zr-stp" data-act="back1" aria-label="${esc(tx('Back one layer','后退一层'))}" data-tipx="${esc(tx('Back to the start of the previous layer’s computation.','退回上一层计算开始的地方。'))}">◂</button>`+
    `<button type="button" class="btn ghost sm zr-stp" data-act="fwd1" aria-label="${esc(tx('Forward one layer','前进一层'))}" data-tipx="${esc(tx('On to the end of the next layer’s computation.','前进到下一层计算结束的地方。'))}">▸</button>`+
    A.ui.seg({id:'zr-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd})+
    `<div class="zr-scrub"><input type="range" min="0" max="1000" step="1" value="${Math.round(pT/T*1000)}" data-key="scrub" aria-label="${esc(tx('Position in the step','这一步里的位置'))}"></div>`;
  A.ui.initRanges(pl);
}
function tick(t){
  raf=0; if(!app||!playing||!SIM) return;
  const dt=lastT?t-lastT:16; lastT=t; const T=SIM.step;
  pT=Math.min(T,pT+dt/DUR[spd]*T);
  if(pT>=T){playing=false; pT=T; watched.add(SIM._sig); drawHead(); drawPlayer(); render(); if(isP('dp')){wantObs=true; updateRail(); wantObs=false; revealSoon();} return;}
  drawHead(); drawPlayer(); raf=requestAnimationFrame(tick);
}
function play(){if(!SIM) return; if(pT>=SIM.step) pT=0; playing=true; lastT=0; if(A.reduceMotion()){finishPlay(); return;} if(!raf) raf=requestAnimationFrame(tick); drawPlayer();}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; if(app) drawPlayer();}
function finishPlay(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; if(!SIM) return; pT=SIM.step; watched.add(SIM._sig); wantObs=true; render(); revealSoon(); wantObs=false;}
let playT=0;
function playSoon(){clearTimeout(playT); playT=setTimeout(()=>{if(app&&SIM&&!watched.has(SIM._sig)&&!playing&&pT<SIM.step) play();},60);}
function stepLayer(dir){
  if(!SIM) return; pause(); const cs=SIM.ev.filter(e=>e.r==='c');
  if(dir>0){const nx=cs.find(e=>e.t1>pT+1e-9); pT=nx?nx.t1:SIM.step;} else {const pv=[...cs].reverse().find(e=>e.t0<pT-1e-9); pT=pv?pv.t0:0;}
  if(pT>=SIM.step-1e-12){pT=SIM.step; if(!watched.has(SIM._sig)){watched.add(SIM._sig); render(); return;}}
  drawHead(); drawPlayer();
}

/* =====================================================================
   stage: head, metrics, math
   ===================================================================== */
function statusOf(){
  if(hidePred()) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The memory stays hidden until you answer in part 1.','在第 1 部分作答之前，先不显示显存。')};
  const e=evalC(S);
  if(mode==='challenge') return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when each GPU fits in 80 GB and waits at most 10% of the step. Now: ${fGB(e.me.total)}, waiting ${fPctW(e.s.share)}.`,`每张卡放得下 80 GB、等数据不超过每步的 10% 时，徽章是绿色。现在：${fGB(e.me.total)}，等数据 ${fPctW(e.s.share)}。`)};
  if(!e.me.fits) return {cls:'bad',txt:tx('Does not fit','放不下'),short:tx('Does not fit','放不下'),tip:tx(`Each GPU needs ${fGB(e.me.total)}, over the H100’s 80 GB.`,`每张卡要 ${fGB(e.me.total)}，超过了 H100 的 80 GB。`)};
  return {cls:'ok',txt:tx('Fits','放得下'),short:tx('Fits','放得下'),tip:tx(`Each GPU needs ${fGB(e.me.total)}, within the H100’s 80 GB.`,`每张卡要 ${fGB(e.me.total)}，在 H100 的 80 GB 以内。`)};
}
function drawHeadline(){
  const t=$('#zr-st-title',app), sub=$('#zr-st-sub',app), st=$('#zr-status',app);
  t.textContent=tx(`${MOD[S.m].name} on ${S.N} H100s${S.N>8?` (${S.N/8} machines)`:''}: what each GPU holds, and one training step`,`${S.N} 张 H100${S.N>8?`（${S.N/8} 台机器）`:''}训练 ${MOD[S.m].name}：每张卡存什么，一步怎么走`);
  sub.textContent=hidePred()?tx('Data parallel, 4,096 tokens per GPU per step','数据并行，每张卡每步 4,096 个 token'):tx(`${stLong(S.st,S.N)}, ${fTok(S.T)} tokens per GPU per step`,`${stLong(S.st,S.N)}，每张卡每步 ${fTok(S.T)} 个 token`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  if(hidePred()) return [{k:'mem',label:tx('Memory per GPU','每张卡的显存'),v:'—',sub:tx('shown in part 2','第 2 部分揭晓'),tip:tx('Hidden until you answer in part 1.','在第 1 部分作答之前先隐藏。')}];
  const e=evalC(S), k=e.k, W=fixd(k.W/1e9,k.W%1e9?1:0);
  const kin=k.one?tx('inside the machine','都在机器内'):(S.st==='hsdp'?tx('only the gradients cross machines','只有梯度跨机器'):tx('across machines','跨机器'));
  const netTip=k.one?'':tx(' ','')+tx(`Each machine moves about ${fGB(k.netB)} over its network per step, ${fSec(k.netT)} at ${W} GB/s.`,`每台机器每步约有 ${fGB(k.netB)} 要过网络，按 ${W} GB/s 算要 ${fSec(k.netT)}。`);
  return [
   {k:'mem',label:tx('Memory per GPU','每张卡的显存'),v:fGB(e.me.total),sub:e.me.fits?tx('within 80 GB','在 80 GB 以内'):tx(`${fG(e.me.over)} GB over 80 GB`,`超出 80 GB ${fG(e.me.over)} GB`),
    tip:tx(`Model states ${fGB(e.me.states)}${e.me.inf?`, weights in flight ${fGB(e.me.inf)}`:''}, activations ${fGB(e.me.act)}, logits ${fGB(e.me.lg)}. ${fGB(e.me.total)} per GPU in all.`,`模型状态 ${fGB(e.me.states)}${e.me.inf?`，正在用的权重 ${fGB(e.me.inf)}`:''}，激活 ${fGB(e.me.act)}，logits ${fGB(e.me.lg)}。每张卡一共 ${fGB(e.me.total)}。`)},
   {k:'bytes',label:tx('Data sent per GPU per step','每张卡每步发出的数据'),v:fGB(k.bytes),sub:tx(`${fX(k.ratio)} × data parallel, ${kin}`,`数据并行的 ${fX(k.ratio)} 倍，${kin}`),
    tip:tx(`This GPU sends ${fGB(k.bytes)} in one step. Plain data parallel on the same GPUs sends ${fGB(k.dpBytes)}.`,`这一步里这张卡发出 ${fGB(k.bytes)}。同样的卡做数据并行时，发出 ${fGB(k.dpBytes)}。`)+netTip},
   {k:'step',label:tx('Time per step','每步的时间'),v:fSec(e.s.step),sub:tx(`computation ${fSec(e.s.C)}`,`其中计算 ${fSec(e.s.C)}`),
    tip:tx(`Computation ${fSec(e.s.C)}, with a GPU at ${S.u}% of its peak while computing. Transfers ${fSec(k.time)} in all, partly during the computation. The step ends when both are done. MFU for this step: ${fixd(e.s.mfu*100,1)}%.`,
           `计算要 ${fSec(e.s.C)}，按 GPU 算的时候能跑到峰值的 ${S.u}% 算。传输一共要 ${fSec(k.time)}，一部分和计算同时进行。两者都完成，这一步才结束。这一步的 MFU 是 ${fixd(e.s.mfu*100,1)}%。`)},
   {k:'wait',label:tx('Waiting for data','等数据的时间'),v:fPctW(e.s.share),sub:tx('of this step, on this GPU','占这张卡这一步的比例'),
    tip:e.s.share<0.01?tx('In this step, this GPU almost never waits for data: under 1% of the step.','这一步里，这张卡几乎不用等数据，不到这一步的 1%。')
      :tx(`In this step, this GPU has nothing to compute for ${fSec(e.s.wait)}, ${fPctW(e.s.share)} of the step.`,`这一步里，这张卡有 ${fSec(e.s.wait)} 没有可算的东西，占这一步的 ${fPctW(e.s.share)}。`)}];
}
function drawMetrics(){
  const host=$('#zr-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    el.classList.toggle('zr-bad',m.k==='mem'&&!hidePred()&&!mem(S).fits);
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&!/under|不到/.test(prev+m.v)){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v)),pts:m.k==='wait'})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
function mathHTML(){
  if(hidePred()) return `<p>${tx('The formulas appear once you have answered in part 1.','在第 1 部分作答后，这里给出公式。')}</p>`;
  const e=evalC(S), m=MOD[S.m], P=m.P, n=S.N, t=T_, zh=A.lang()==='zh', k=e.k;
  const sh=e.me.sh, rows=[], W=fixd(k.W/1e9,k.W%1e9?1:0), tok=t('T',tx('tokens','token 数')), X=A.ui.math, Xt=v=>`\\text{${v}}`;
  /* H2 (the owner, 2026-10-09): short blocks, each a lead line in words, the named formula, the reader's numbers on the line under it */
  const LEAD=(name,why)=>`<p><b>${name}</b>${why}</p>`, sep=tx(', ','，'), end=tx('.','。');
  /* integrator decision 4: N is the parameter count, d the number of GPUs */
  const form={dp:'18N',z1:'6N + \\frac{12N}{d}',z2:'2N + \\frac{16N}{d}',z3:'\\frac{18N}{d}',hsdp:'\\frac{18N}{8}'}[S.st];
  rows.push(LEAD(tx('Model states per GPU','每张卡的模型状态'),tx(': each GPU keeps its pieces of the weights, the gradients and the optimizer state.','：每张卡存它分到的那几份权重、梯度和优化器状态。'))+MN(MS(),'\\frac{2N}{s_w} + \\frac{4N}{s_g} + \\frac{12N}{s_o}')+
            `<p>${tx(`2N is the BF16 weights, 4N the FP32 gradients, 12N the optimizer state (the FP32 copy, m and v). ${X('s_w')}, ${X('s_g')} and ${X('s_o')} are how many pieces each of them is split into. Activations are not in it.`,`2N 是 BF16 权重，4N 是 FP32 梯度，12N 是优化器状态（FP32 权重副本、m 和 v）。${X('s_w')}、${X('s_g')}、${X('s_o')} 是这三样各被分成几份。激活不算在里面。`)}</p>`+
            `<p>${S.st==='hsdp'?tx('HSDP splits them only over the 8 GPUs of a machine, so each s is 8:','HSDP 只在一台机器的 8 张卡之间分，所以每个 s 都是 8：'):tx(`${stName(S.st)} with d GPUs:`,`${stName(S.st)}，d 张卡：`)}</p>`+MN(MS(),form)+
            `<p>${t('P','N')} = ${fint(P)}${sep}${X(`s_w = ${sh[0]}`)}${sep}${X(`s_g = ${sh[1]}`)}${sep}${X(`s_o = ${sh[2]}`)}${end}</p>`+
            MV(`= ${fG(e.me.w)} + ${fG(e.me.gr)} + ${fG(e.me.o)} = ${Xt(fGB(e.me.states))}`));
  if(e.me.inf) rows.push(LEAD(tx('Weights in flight','正在用的权重'),tx(`: a layer computes with its full BF16 weights, gathered just before it runs${S.ov?'; with overlap the next layer’s arrive at the same time':''}.`,`：每一层算之前，要先把它完整的 BF16 权重收齐${S.ov?'；开了重叠，下一层的权重也同时在收':''}。`))+
            MN(tx('\\text{weights in flight}','\\text{正在用的权重}'),`${S.ov?2:1} \\times \\frac{2N}{L}`)+
            MV(`= ${S.ov?2:1} \\times \\frac{2 \\times ${fint(P)}}{${m.L}} \\ ${Xt(tx('bytes','字节'))} = ${Xt(fGB(e.me.inf))}`));
  rows.push(LEAD(tx('Activations (recomputation)','激活（开了重算）'),tx(': each layer keeps only its input, and the layer being recomputed holds its full activations for a moment.','：每层只留输入，正在重算的那一层暂时放着它全部的激活。'))+(zh?MN('\\text{每张卡的激活字节数}','\\text{token 数} \\times (2hL + a)'):MN('\\text{bytes of activations per GPU}','\\text{tokens} \\times (2hL + a)'))+
            `<p>${tx(`h is the hidden size and L the number of layers. Each layer keeps its input, 2h bytes per token. a is the activations of the layer being recomputed, ${fint(e.me.aL)} bytes per token.`,`h 是隐藏层的宽度，L 是层数。每层留下的输入，每个 token 占 2h 字节。a 是正在重算的那一层的激活，每个 token ${fint(e.me.aL)} 字节。`)}</p>`+
            `<p>${tok} = ${fint(S.T)}${sep}h = ${fint(m.h)}${sep}L = ${m.L}${end}</p>`+
            MV(`= ${fint(S.T)} \\times (2 \\times ${fint(m.h)} \\times ${m.L} + ${fint(e.me.aL)}) \\ ${Xt(tx('bytes','字节'))} = ${Xt(fGB(e.me.act))}`));
  rows.push(LEAD(tx('Logits (FP32)','logits（FP32）'),tx(': every token gets one logit per word of the vocabulary, 4 bytes each.','：每个 token 对词表里每个词都有一个 logit，每个 4 字节。'))+(zh?MN('\\text{logits 的字节数}','\\text{token 数} \\times V \\times 4'):MN('\\text{bytes of the logits}','\\text{tokens} \\times V \\times 4'))+
            `<p>${tx(`V is the vocabulary, ${fint(m.V)} words.`,`V 是词表大小，${fint(m.V)} 个词。`)}</p>`+
            MV(`= ${fint(S.T)} \\times ${fint(m.V)} \\times 4 \\ ${Xt(tx('bytes','字节'))} = ${Xt(fGB(e.me.lg))}`));
  rows.push(LEAD(tx('Total','合计'),tx(': the parts above add up, against the 80 GB of an H100.','：上面几项加起来，和一张 H100 的 80 GB 比。'))+
            MV(`${fG(e.me.states)}${e.me.inf?` + ${fG(e.me.inf)}`:''} + ${fG(e.me.act)} + ${fG(e.me.lg)} = ${Xt(fGB(e.me.total))} ${e.me.fits?'\\leq':'>'} ${Xt('80 GB')}`));
  rows.push(LEAD(tx('Data sent per GPU per step','每张卡每步发出的数据'),tx(': what each GPU sends to the others in one step, in BF16.','：一步里每张卡发给别的卡的数据，按 BF16 算。'))+MN(zh?'\\text{每张卡每步发出的字节数}':'\\text{bytes each GPU sends per step}',(S.st==='dp'?'2 \\times 2N \\times \\frac{d-1}{d}':(S.st==='z3'?'3 \\times 2N \\times \\frac{d-1}{d}':(S.st==='hsdp'?'3 \\times 2N \\times \\frac{7}{8} + 2 \\times \\frac{2N}{8} \\times \\frac{R-1}{R}':'2 \\times 2N \\times \\frac{d-1}{d}')))));
  rows.push(`<p>d = ${n}${S.st==='hsdp'?`${sep}R = ${Math.max(1,n/8)} ${tx('machines','台机器')}`:''}${tx(': ','：')}<b>${fGB(k.bytes)}</b></p>`+
            `<p>${tx(`Data parallel on the same GPUs sends ${X(`2 \\times 2N \\times \\frac{${n} - 1}{${n}}`)} = ${fGB(k.dpBytes)}.`,`同样的卡做数据并行，发出 ${X(`2 \\times 2N \\times \\frac{${n} - 1}{${n}}`)} = ${fGB(k.dpBytes)}。`)}</p>`);
  rows.push(LEAD(tx('Compute time','计算时间'),tx(': the FLOPs of one step, divided by what the GPU really does per second.','：一步的运算量，除以 GPU 每秒真正做到的运算量。'))+(zh?MN('\\text{每步的计算时间}','\\frac{8N \\times \\text{token 数}}{989.5\\ \\text{TFLOP/s} \\times u}'):MN('\\text{compute time per step}','\\frac{8N \\times \\text{tokens}}{989.5\\ \\text{TFLOP/s} \\times u}')));
  rows.push(`<p>${tx(`With recomputation a GPU does 8N FLOPs per token. 2N of them are the forward pass done again.`,`开了重算，每个 token 要算 8N 次。其中 2N 次是重算的那次前向。`)}</p>`+
            `<p>${tx(`u is the share of the peak while computing, ${t('u',S.u+'%')} here.`,`u 是 GPU 算的时候能跑到峰值的多少，这里是 ${t('u',S.u+'%')}。`)}</p>`+
            MV(`= \\frac{8 \\times ${fint(P)} \\times ${fint(S.T)}}{989.5 \\times 10^{12} \\times ${S.u}\\%} = ${Xt(fSec(e.s.C))}`));
  rows.push(LEAD(tx('Transfer time','传输时间'),k.one?tx(': inside one machine, each GPU’s data goes over NVLink at 450 GB/s.','：在一台机器里，每张卡的数据走 NVLink，每秒 450 GB。'):tx(`: across machines, a layer’s whole array goes over each machine’s network. An all-gather or a reduce-scatter counts once, an all-reduce twice. Each machine has ${W} GB/s.`,`：跨机器时，一层的整块数据都要过每台机器的网络。all-gather 和 reduce-scatter 各算一次，all-reduce 算两次。每台机器的网络是 ${W} GB/s。`)));
  if(k.one) rows.push(MN(tx('\\text{transfer time}','\\text{传输时间}'),tx('\\frac{\\text{bytes sent}}{450\\ \\text{GB/s}}','\\frac{\\text{发出的字节数}}{450\\ \\text{GB/s}}'))+MV(`= \\frac{${Xt(fGB(k.bytes))}}{450\\ \\text{GB/s}} = ${Xt(fSec(k.time))}`));
  else if(S.st==='hsdp'){const inB=m.L*(k.per.ag+k.per.agb+k.per.rs);
    rows.push(MN(tx('\\text{transfer time}','\\text{传输时间}'),tx(`\\frac{\\text{bytes inside the machine}}{450\\ \\text{GB/s}} + 2 \\times \\frac{2N}{${W}\\ \\text{GB/s}}`,`\\frac{\\text{机器内的字节数}}{450\\ \\text{GB/s}} + 2 \\times \\frac{2N}{${W}\\ \\text{GB/s}}`))+
              MV(`= \\frac{${Xt(fGB(inB))}}{450\\ \\text{GB/s}} + 2 \\times \\frac{${Xt(fGB(2*P))}}{${W}\\ \\text{GB/s}} = ${Xt(fSec(k.time))}`));}
  else rows.push(MN(tx('\\text{transfer time}','\\text{传输时间}'),`${S.st==='z3'?3:2} \\times \\frac{2N}{${W}\\ \\text{GB/s}}`)+
              MV(`= ${S.st==='z3'?3:2} \\times \\frac{${Xt(fGB(2*P))}}{${W}\\ \\text{GB/s}} = ${Xt(fSec(k.time))}`));
  rows.push(LEAD(tx('Step and waiting','每步和等数据'),S.ov?tx(': the schedule overlaps the two layer by layer.','：排程里两者一层一层地重叠。'):tx(': the schedule does not overlap the two.','：排程里两者完全不重叠。'))+`<p>${tx('Step','每步')} = <b>${fSec(e.s.step)}</b></p>`+
            MN(tx('\\text{waiting}','\\text{等数据的比例}'),tx('\\frac{\\text{step} - \\text{compute}}{\\text{step}}','\\frac{\\text{每步} - \\text{计算}}{\\text{每步}}'))+
            MV(`= \\frac{${Xt(fSec(e.s.step))} - ${Xt(fSec(e.s.C))}}{${Xt(fSec(e.s.step))}} = ${Xt(fPctW(e.s.share))}`));
  rows.push(LEAD(DEF('MFU',TIP.mfu()),tx(': the share of the peak the step really uses, counting 6N per token (the recomputed forward pass is not counted).','：这一步真正用上的峰值比例，每个 token 按 6N 算（重算的那次前向不算）。'))+
            MN(tx('\\text{peak share (MFU)}','\\text{峰值比例（MFU）}'),tx('\\frac{6N \\times \\text{tokens}}{\\text{step} \\times 989.5\\ \\text{TFLOP/s}}','\\frac{6N \\times \\text{token 数}}{\\text{每步} \\times 989.5\\ \\text{TFLOP/s}}'))+
            MV(`= \\frac{6 \\times ${fint(P)} \\times ${fint(S.T)}}{${Xt(fSec(e.s.step))} \\times 989.5 \\times 10^{12}} = ${fixd(e.s.mfu*100,1)}\\%`));
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){Object.assign(S,{m:CH.m,N:CH.N,net:CH.net,ov:CH.ov,u:CH.u}); if(!CH.sts.includes(S.st)) S.st=CH.start.st; if(!CH.Ts.includes(S.T)) S.T=CH.start.T;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHeadline(); memCols(); drawTimeline(); drawMetrics();
  const mb=$('#zr-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the sharding and the tokens per GPU, in the task card. Everything else is fixed: Llama‑3.1‑70B, 64 GPUs in 8 machines, a 50 GB/s network per machine, a GPU at ${CH.u}% of its peak while computing, transfers during the computation.`,`挑战中只能在任务卡里改分法和每张卡的 token 数。其余设置固定：Llama‑3.1‑70B，8 台机器共 64 张卡，每台机器的网络 50 GB/s，GPU 算的时候能跑到峰值的 ${CH.u}%，传输和计算同时进行。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='net') return tx(`In this part you pick one of three ways in the card. Llama‑3.1‑8B on 64 GPUs in 8 machines, a GPU at ${BASE.u}% of its peak while computing. You can change the settings in Scenarios.`,`这一部分只在卡片里三选一。模型是 Llama‑3.1‑8B，8 台机器共 64 张卡，GPU 算的时候能跑到峰值的 ${BASE.u}%。在应用场景里可以改这些设置。`);
  return tx(`Fixed in this part: Llama‑3.1‑8B on one machine with 8 H100s, 4,096 tokens per GPU per step, a GPU at ${BASE.u}% of its peak while computing. You can change the settings in Scenarios.`,`这一部分固定：Llama‑3.1‑8B，一台机器 8 张 H100，每张卡每步 4,096 个 token，GPU 算的时候能跑到峰值的 ${BASE.u}%。在应用场景里可以改这些设置。`);
}
function syncControls(){
  const setR=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  setR('T',TS.indexOf(S.T),fTok(S.T)); setR('u',US.indexOf(S.u),S.u+'%');
  $$('.seg-ctl[data-key="m"]',app).forEach(g=>A.ui.setPressed(g,S.m)); $$('.seg-ctl[data-key="N"]',app).forEach(g=>A.ui.setPressed(g,String(S.N)));
  $$('.seg-ctl[data-key="st"]',app).forEach(g=>A.ui.setPressed(g,S.st)); $$('.seg-ctl[data-key="ov"]',app).forEach(g=>A.ui.setPressed(g,String(S.ov))); $$('.seg-ctl[data-key="net"]',app).forEach(g=>A.ui.setPressed(g,S.net));
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#zr-controls',app).classList.toggle('locked',ch);
  ['#zr-ctl-m-wrap','#zr-ctl-N-wrap','#zr-ctl-st-wrap','#zr-ctl-T-wrap','#zr-ctl-ov-wrap','#zr-ctl-net-wrap','#zr-ctl-as','#zr-ctl-reset-wrap'].forEach(id=>hide(id,g||ch));
  const dm=$('#zr-ctl-as .def',app); if(dm) dm.dataset.tipx=TIP.mfu();
  const ln=$('#zr-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||hidePred()) return; const e=evalC(S);
  A.announce(tx(`${stName(S.st)} on ${S.N} GPUs: ${fGB(e.me.total)} per GPU, waiting ${fPctW(e.s.share)} of the step.`,`${S.N} 张卡、${stName(S.st)}：每张卡 ${fGB(e.me.total)}，等数据 ${fPctW(e.s.share)}。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  if(!MOD[S.m]) S.m='8b'; if(!NS.includes(S.N)) S.N=8; if(!STS.includes(S.st)) S.st='dp'; if(!TS.includes(S.T)) S.T=4096; S.ov=S.ov?1:0; if(!NET[S.net]) S.net='ib'; if(!US.includes(S.u)) S.u=BASE.u;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv5(v){lv5=v; lv5Tried.add(v); S={...P5,ov:v==='ov'?1:0}; render();}
function setLv6(v){lv6=v; lv6Tried.add(v); S={...lv6Cfg(v)}; render();}
function setRem(k){if(!REMS.includes(k)) return; remView=k; S=k==='dp'?{...P2}:(k==='z3'?{...P5,ov:1}:{...P6}); renderRail(); if(innerWidth<=760) A.ui.reveal($('#zr-stage',app));}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#zr-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#zr-rail .step.current',app)||$('#zr-rail',app));}
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
    const st=$('#zr-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#zr-rail .step.current',app)||$('#zr-rail .railcard',app)||$('#zr-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#zr-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#zr-insight',app)); else revealBox($('#zr-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chAllOpen=false; S={...S,...CH.start,m:CH.m,N:CH.N,net:CH.net,ov:CH.ov,u:CH.u};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#zr-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.m,s.N,s.st,TS.indexOf(s.T),s.ov,s.net,s.u].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const [m,N,st,ti,ov,net,u]=p, Nn=+N, Ti=+ti, un=+u;
  if(!MOD[m]||!NS.includes(Nn)||!STS.includes(st)||!(Ti>=0&&Ti<TS.length)||(ov!=='0'&&ov!=='1')||!NET[net]||!US.includes(un)) return null;
  return {m,N:Nn,st,T:TS[Ti],ov:+ov,net,u:un};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*104729+11), pk=a=>a[Math.floor(r()*a.length)];
  const Pb=pk([3,7,13,34]), N=pk([8,16,32]), st=pk(['z1','z2','z3']);
  const P=Pb*1e9, f=(N-1)/N, states={z1:6*P+12*P/N,z2:2*P+16*P/N,z3:18*P/N}[st], bytes=(st==='z3'?3:2)*2*P*f;
  return {seed,Pb,N,st,states,bytes,dp:18*P,dpBytes:2*2*P*f,fits:states<=CAP};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#zr-drill-q',app), a=$('#zr-drill-a',app);
  qd.innerHTML=tx(`A model of <b>${d.Pb}B</b> parameters trains with mixed-precision Adam (18 bytes per parameter) on <b>${d.N}</b> GPUs of 80 GB, using <b>${d.st.toUpperCase().replace('Z','ZeRO-')}</b>. How much memory do the model states take per GPU, does that fit, and how many bytes does each GPU send per step, compared with plain data parallel?`,
                  `一个 <b>${d.Pb}B</b> 参数的模型，用混合精度 Adam（每个参数 18 字节）在 <b>${d.N}</b> 张 80 GB 的卡上训练，用 <b>${d.st.toUpperCase().replace('Z','ZeRO-')}</b>。每张卡的模型状态占多少显存？放得下吗？每张卡每步发出多少字节，和数据并行比呢？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const form=A.ui.math({z1:'6N + \\frac{12N}{d}',z2:'2N + \\frac{16N}{d}',z3:'\\frac{18N}{d}'}[d.st]), kk=d.st==='z3'?'3':'2', bx=A.ui.math(`${kk} \\times 2N \\times \\frac{d - 1}{d}`), bn=A.ui.math(`${kk} \\times ${2*d.Pb}\\ \\text{GB} \\times \\frac{${d.N-1}}{${d.N}}`);
  a.innerHTML=tx(`<p><b>${fGB(d.states)} of model states per GPU, before activations; it ${d.fits?'fits in':'does not fit in'} 80 GB. ${fGB(d.bytes)} sent per step, ${d.st==='z3'?'1.5 times':'the same as'} data parallel’s ${fGB(d.dpBytes)}.</b></p><p>Model states: ${form} with N = ${d.Pb}B and d = ${d.N}. Data parallel would need ${fGB(d.dp)}.</p><p>Bytes: ${bx} = ${bn}.</p>`,
    `<p><b>每张卡的模型状态是 ${fGB(d.states)}，还没算激活，80 GB 的卡${d.fits?'放得下':'放不下'}。每步发出 ${fGB(d.bytes)}，${d.st==='z3'?`是数据并行（${fGB(d.dpBytes)}）的 1.5 倍`:`和数据并行一样多（${fGB(d.dpBytes)}）`}。</b></p><p>模型状态按 ${form} 算，N = ${d.Pb}B，d = ${d.N}。数据并行要 ${fGB(d.dp)}。</p><p>字节数按 ${bx} 算，也就是 ${bn}。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={T:{en:'Tokens per GPU per step',zh:'每张卡每步的 token 数'},u:{en:'Share of the peak while computing',zh:'算的时候能到峰值的多少'},P:{en:'Model',zh:'模型'}};
const TERM={
 P:()=>tx(`N is the number of parameters: ${fint(MOD[S.m].P)} for ${MOD[S.m].name}, from its config.`,`N 是参数量。${MOD[S.m].name} 有 ${fint(MOD[S.m].P)} 个参数，来自它的配置。`),
 T:()=>tx(`The tokens each GPU processes per step, ${fint(S.T)} here.`,`每张卡每步处理的 token 数，这里是 ${fint(S.T)}。`),
 u:()=>tx(`An assumed value: while computing, a GPU reaches ${S.u}% of the H100’s peak, 989.5 TFLOP/s.`,`这是假设值：GPU 算的时候能跑到 H100 峰值 989.5 TFLOP/s 的 ${S.u}%。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return k==='T'?tx('Change the tokens per GPU in the task card.','请在任务卡里改每张卡的 token 数。'):tx(`Fixed during the challenge: Llama‑3.1‑70B, a GPU at ${CH.u}% of its peak while computing.`,`挑战中固定不变：Llama‑3.1‑70B，GPU 算的时候能跑到峰值的 ${CH.u}%。`);
  if(mode!=='guided') return '';
  return tx('Fixed in this part. You can change it in Scenarios.','这一部分固定不变，在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({T:$('#zr-ctl-T',app),u:$('#zr-ctl-u',app),P:$('#zr-ctl-m',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#zr-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#zr-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S=mode==='sandbox'?{...PRESETS[0].s}:{...BASE}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(evalC(chCfg(S.st,S.T)).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(playing) pause(); else play(); A.tip.quiet(); return;}
  if(a==='fwd1'){stepLayer(1); A.tip.quiet(); return;}
  if(a==='back1'){stepLayer(-1); A.tip.quiet(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#zr-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#zr-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='T') bySettings(e.target,()=>setState({T:TS[v]}));
    else if(k==='u') bySettings(e.target,()=>setState({u:US[v]}));
    else if(k==='scrub'&&SIM){pause(); pT=v/1000*SIM.step; if(v>=1000&&!watched.has(SIM._sig)){watched.add(SIM._sig); render(); return;} drawHead(); drawPlayer();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#zr-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#zr-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('z3')&&b.closest('#zr-ic-lv5')){const f=()=>byLearner(()=>setLv5(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.lever!=null&&isP('net')&&b.closest('#zr-ic-lv6')){const f=()=>byLearner(()=>setLv6(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='m') bySettings(b,()=>setState({m:d.v}));
      else if(k==='N') bySettings(b,()=>setState({N:+d.v}));
      else if(k==='st') bySettings(b,()=>setState({st:d.v}));
      else if(k==='ov') bySettings(b,()=>setState({ov:+d.v}));
      else if(k==='net') bySettings(b,()=>setState({net:d.v}));
      else if(k==='st13'||k==='st24'){wantObs=true; byLearner(()=>setState({st:d.v})); revealSoon(); wantObs=false;}
      else if(k==='spd'){spd=d.v; lastT=0; drawPlayer();}
      else if(k==='chs') byLearner(()=>setState({st:d.v}));
      else if(k==='cht') byLearner(()=>setState({T:+d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#zr-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#zr-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12 07:13 UTC): four text-only sections read before the guided parts: why one
   GPU is not enough, data parallelism, the three collectives defined by what each GPU holds before and after, and the links inside
   and between machines. d is the number of GPUs, as in the lab's own formulas. What each GPU
   holds, sharding, what each collective sends and how they combine (the challenge asks for them) stay with the lab, so no memory per
   GPU, bytes, waiting share or all-reduce = reduce-scatter + all-gather is here; the numbers are the datasheets' (989.5 TFLOP/s, 3.35 TB/s, NVLink 450 GB/s, 400 Gb/s NICs) and Meta's 15T tokens
   (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=labLink(id); return tx(a,`“${a}”`);};
// a paragraph per item; a formula block (a <div>) stands on its own
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return /^<div/.test(h)?h:`<p>${h}</p>`;}).join('')});
// formulas on their own line, side by side, wrapping under each other in a narrow column (one:true: the pieces of one formula, kept
// close), so nothing scrolls sideways
const prFX=(xs,one)=>`<div class="zr-prfx${one?' one':''}">${xs.map(x=>`<span>${prM(x)}</span>`).join('')}</div>`;
const prTo=(a,b)=>prFX([a,'\\to\\ '+b],true);   // what the GPUs hold before → after a collective
A.registerPrimer(ID,{min:8,sections:[
 {h:()=>tx('One GPU can neither hold it nor finish it: model states and training compute','一张卡放不下，也训不完：模型状态和训练的计算量'),parts:[prP(
   ()=>tx(`Training a model repeats one step many times: run the forward pass on a batch of data, compute the gradients of all the parameters, then update the parameters (see the lab ${prLK('backprop')}). To train Llama 3.1 8B (about 8 billion parameters) on one H100, what this step uses has to fit in the GPU’s memory, and all the training data has to be processed in a reasonable time. One GPU can do neither.`,
          `训练一个模型，就是把同一步重复很多次：用一批数据跑一遍前向，求出全部参数的梯度，再更新参数（见${prLK('backprop')}）。拿一张 H100 训练 Llama 3.1 8B（约 80 亿个参数），这一步用到的东西要放得进显存，全部训练数据也要在合理的时间里算完。这两件事，一张卡都做不到。`),
   ()=>tx(`Memory first. The 8B model’s weights in BF16 (2 bytes per number) take about 16 GB, so for inference alone one H100’s 80 GB holds them. In training, each parameter also carries a gradient and optimizer state; with the weights these make up the model states (see the lab ${prLK('training-memory')}). The model states are several times larger than the weights alone, over 80 GB for the 8B model already, and that is before the activations.`,
          `先看显存。8B 模型的权重用 BF16（每个数值 2 字节）存，约 16 GB，只做推理的话，一张 H100 的 80 GB 放得下。训练时，每个参数还要带上梯度和优化器状态，和权重合起来就是模型状态（见${prLK('training-memory')}）。模型状态比权重本身大好几倍，8B 模型的已经超过 80 GB，这还没算激活。`),
   ()=>tx(`Then compute. Training takes about ${prM('6N')} floating-point operations per token, where ${prM('N')} is the number of parameters. The forward pass does one multiply and one add per parameter, ${prM('2N')} (see the lab ${prLK('prefill-decode')}). In the backward pass, every matrix multiply ${prM('y_j = A x_j')} (${prM('x_j')} is token ${prM('j')}’s vector) needs two gradients: the gradient of its input, ${prM('A^{⊤} \\frac{\\partial L}{\\partial y_j}')}, passed back to the layer before; and the gradient of its weights, to which every token adds ${prM('\\frac{\\partial L}{\\partial y_j} x_j^{⊤}')}, kept for the update. Each takes one multiply and one add for every number in ${prM('A')}, as many as the forward ${prM('A x_j')}, so the two together take ${prM('4N')} per token. Llama 3.1 8B was pretrained on about 15 trillion tokens. The total FLOPs ${prM('F')} divided by the H100’s peak compute ${prM('P')} (see the lab ${prLK('roofline')}) is a lower bound on the time one GPU needs:`,
          `再看算力。训练时每个 token 约要 ${prM('6N')} 次浮点运算，${prM('N')} 是参数量。前向每个参数做一次乘法、一次加法，是 ${prM('2N')} 次（见${prLK('prefill-decode')}）。反向时，每个矩阵乘法 ${prM('y_j = A x_j')}（${prM('x_j')} 是第 ${prM('j')} 个 token 的向量）要算两样梯度：对输入的梯度 ${prM('A^{⊤} \\frac{\\partial L}{\\partial y_j}')} 传给前一层；对权重的梯度要给每个 token 加上一份 ${prM('\\frac{\\partial L}{\\partial y_j} x_j^{⊤}')}，留给更新用。两样都要对 ${prM('A')} 里的每个数做一次乘法、一次加法，和前向算 ${prM('A x_j')} 一样多，合起来每个 token ${prM('4N')} 次。Llama 3.1 8B 预训练用了约 15 万亿个 token。总运算量 ${prM('F')} 除以 H100 的峰值算力 ${prM('P')}（见${prLK('roofline')}），就是一张卡所需时间的下限：`),
   ()=>prFX([tx('F \\approx 6N \\times \\text{tokens}','F \\approx 6N \\times \\text{token 数}'),'\\approx 6 \\times (8 \\times 10^{9}) \\times (1.5 \\times 10^{13})','\\approx 7.2 \\times 10^{23}\\ \\text{FLOP}'],true),
   ()=>prFX(['t \\ge \\frac{F}{P}','\\approx \\frac{7.2 \\times 10^{23}\\ \\text{FLOP}}{989.5 \\times 10^{12}\\ \\text{FLOP/s}}',tx('\\approx 7.3 \\times 10^{8}\\ \\text{s}','\\approx 7.3 \\times 10^{8}\\ \\text{秒}'),tx('\\approx 23\\ \\text{years}','\\approx 23\\ \\text{年}')],true),
   ()=>tx('That assumes the peak every moment; in practice it takes longer. The memory does not fit and the time is far too long, so training uses many GPUs together and splits each step’s work among them.',
          '这是每一刻都跑满峰值的情况，实际只会更久。显存放不下，时间也等不起，所以要用很多张卡一起训练，把每一步的工作分给它们。'))]},
 {h:()=>tx('Data parallelism: each GPU uses different data, and the gradients are averaged before the update','数据并行：每张卡用不同的数据，更新前取平均'),parts:[prP(
   ()=>tx(`Each training step uses a batch of sequences. The loss is the average loss over all the tokens of the batch, and the update uses the gradient of that average. A sequence’s forward and backward passes need nothing from the other sequences, so a batch can be split up and computed on several GPUs at once. The most direct multi-GPU training does just that: each of ${prM('d')} GPUs holds a full copy of the model, uses a different part of the batch, runs its own forward and backward pass, and then the gradients are averaged. This is data parallelism.`,
          `训练的每一步用一批（batch）序列。loss 是这一批里所有 token 的 loss 的平均，更新用的是这个平均 loss 的梯度。一条序列的前向和反向用不到别的序列的结果，所以一批序列可以拆开，同时放在几张卡上算。最直接的多卡训练就是这样做的：${prM('d')} 张卡上各有一份完整的模型，每张卡用这一批里不同的一份数据，各自跑前向和反向，再把算出的梯度取平均。这叫数据并行（data parallelism）。`),
   ()=>tx(`Split the batch evenly into ${prM('d')} parts. GPU ${prM('i')} computes part ${prM('i')}: the average loss of that part, ${prM('L_i(\\theta)')}, and its gradient ${prM('g_i')}. Every part has the same number of tokens, so the batch’s average loss is the average of the ${prM('d')} parts’ average losses; differentiation is linear, so the batch’s gradient is the average of the ${prM('d')} gradients:`,
          `把这一批平均分成 ${prM('d')} 份，第 ${prM('i')} 张卡算第 ${prM('i')} 份，得到这一份的平均 loss ${prM('L_i(\\theta)')} 和它的梯度 ${prM('g_i')}。每份的 token 数一样，所以整批的平均 loss 是 ${prM('d')} 份平均 loss 的平均；求导是线性的，所以整批的梯度也是 ${prM('d')} 个梯度的平均：`),
   ()=>prFX(['L(\\theta) = \\frac{1}{d}\\sum_{i=1}^{d} L_i(\\theta)','\\frac{\\partial L}{\\partial \\theta} = \\frac{1}{d}\\sum_{i=1}^{d} g_i','g_i = \\frac{\\partial L_i}{\\partial \\theta}']),
   ()=>tx(`The GPUs start from the same initial parameters, and every step each makes the same update with this average gradient, so after every step each GPU’s parameters are what one GPU would get training on the whole batch. Only the speed differs. ${prM('d')} GPUs computing at once process ${prM('d')} times as many tokens in the same time, while the total number of tokens to process stays the same, so the training time drops to about ${prM('1/d')}, provided the averaging itself takes little time. The pretraining that needs at least 23 years on one GPU has a lower bound of about 8 days on 1,024 GPUs.`,
          `各卡从同一份初始参数出发，每一步都用这个平均梯度做同样的更新，所以每一步之后，各卡的参数都和一张卡用整批数据训练得到的相同。不同的只是速度：${prM('d')} 张卡同时算，同样的时间里处理的 token 是一张卡的 ${prM('d')} 倍，而要处理的 token 总数不变，所以训练时间缩到约 ${prM('1/d')}，前提是取平均本身不太花时间。一张卡至少要约 23 年的那次预训练，用 1,024 张卡，下限就降到约 8 天。`))]},
 {h:()=>tx('How GPUs exchange data: all-reduce, reduce-scatter, all-gather','卡之间怎么交换数据：all-reduce、reduce-scatter、all-gather'),parts:[prP(
   ()=>tx(`Data parallelism averages the GPUs’ gradients before the update. Multi-GPU training and inference also often need two other exchanges: each GPU needs only one piece of the sum of the GPUs’ data, or each GPU holds one piece and every GPU needs all the pieces put together. Each of these exchanges is done by all ${prM('d')} GPUs together, each handing in data and getting back a result by the same rule. This is collective communication. On NVIDIA GPUs it is done by the NCCL library, which PyTorch’s torch.distributed calls.`,
          `数据并行在更新前要把各卡的梯度取平均。除了取平均，多卡训练和推理里，卡之间还常要做另外两种交换：每张卡只要各卡数据之和里的一段；或者各卡各有一段，每张卡都要拿到拼起来的完整一份。这几种交换都由 ${prM('d')} 张卡一起完成，每张卡按同一个规则交出数据、拿回结果，叫集合通信（collective communication）。在 NVIDIA GPU 上，集合通信由通信库 NCCL 来做，PyTorch 的 torch.distributed 调用的就是它。`),
   ()=>tx(`A collective is defined by what each GPU holds before it and after it. Number the GPUs 1 to ${prM('d')}, each holding a vector of the same length; in the formulas, the left of the arrow is what the GPUs hold before, the right what they hold after. The “reduce” in the names means combining the numbers at the same position on all GPUs into one, here by adding them.`,
          `一种集合通信，由做之前和做完以后每张卡上有什么来定义。把 ${prM('d')} 张卡编号为 1 到 ${prM('d')}，各卡上是一样长的向量；式子里箭头左边是做之前各卡上的数据，右边是做完以后的。名字里的 reduce（归约）指把各卡同一位置上的数合成一个，这里是相加。`),
   ()=>tx(`<b>All-reduce</b>: before, GPU ${prM('i')} holds ${prM('x_i')}; after, every GPU holds the sum of the ${prM('d')} vectors. Data parallelism averages with it: ${prM('x_i')} is the gradient ${prM('g_i')}, and the sum is divided by ${prM('d')}.`,
          `<b>all-reduce</b>：做之前，卡 ${prM('i')} 上是 ${prM('x_i')}；做完以后，每张卡上都是这 ${prM('d')} 个向量的和。数据并行取平均用的就是它：${prM('x_i')} 是梯度 ${prM('g_i')}，加完再除以 ${prM('d')}。`),
   ()=>prTo(tx('\\text{GPU}\\ i\\text{:}\\ x_i','\\text{卡}\\ i\\text{：}\\ x_i'),tx('\\text{every GPU:}\\ \\sum_{j=1}^{d} x_j','\\text{每张卡：}\\ \\sum_{j=1}^{d} x_j')),
   ()=>tx(`<b>Reduce-scatter</b>: the vector on every GPU is cut into ${prM('d')} pieces, ${prM('x_i = (x_i^{(1)}, \\ldots, x_i^{(d)})')}. After it, GPU ${prM('i')} holds only piece ${prM('i')} of the sum and none of the others; “scatter” means the result is spread over the GPUs, one piece each. It is used when each GPU needs only its own piece of the sum.`,
          `<b>reduce-scatter</b>：每张卡上的向量都切成 ${prM('d')} 段，${prM('x_i = (x_i^{(1)}, \\ldots, x_i^{(d)})')}。做完以后，卡 ${prM('i')} 上只有和的第 ${prM('i')} 段，没有别的段；scatter（分散）指结果分散在各卡上，每张卡一段。每张卡只需要和里属于自己的那一段时，用它。`),
   ()=>prTo(tx('\\text{GPU}\\ i\\text{:}\\ x_i','\\text{卡}\\ i\\text{：}\\ x_i'),tx('\\text{GPU}\\ i\\text{:}\\ \\sum_{j=1}^{d} x_j^{(i)}','\\text{卡}\\ i\\text{：}\\ \\sum_{j=1}^{d} x_j^{(i)}')),
   ()=>tx(`<b>All-gather</b>: before, GPU ${prM('i')} holds only one piece ${prM('y^{(i)}')}; after, every GPU holds the whole vector, the ${prM('d')} pieces in order. Nothing is added; the pieces are only collected (gathered). It is used when each GPU holds one piece and every GPU needs the whole.`,
          `<b>all-gather</b>：做之前，卡 ${prM('i')} 上只有一段 ${prM('y^{(i)}')}；做完以后，每张卡上都有 ${prM('d')} 段按顺序拼成的整个向量。这里不做加法，只是收集（gather）。各卡各存着一段，而每张卡又要用到完整的一份时，用它。`),
   ()=>prTo(tx('\\text{GPU}\\ i\\text{:}\\ y^{(i)}','\\text{卡}\\ i\\text{：}\\ y^{(i)}'),tx('\\text{every GPU:}\\ (y^{(1)}, \\ldots, y^{(d)})','\\text{每张卡：}\\ (y^{(1)}, \\ldots, y^{(d)})')))]},
 {h:()=>tx('Communication takes time: NVLink and InfiniBand','通信要花时间：NVLink 和 InfiniBand'),parts:[prP(
   ()=>tx(`Data parallelism saves time only if the averaging takes little time. But a collective moves data from one GPU to others. As with reading GPU memory, a transfer takes about its bytes divided by the bandwidth, the most bytes per second the connection can move (see the lab ${prLK('roofline')}). Until the data arrives, the computation that needs it waits. So how fast multi-GPU training runs depends not only on how fast each GPU computes but also on how fast the GPUs are connected, and the bandwidth between GPUs in one machine is far from that between machines.`,
          `数据并行要省时间，前提是取平均不太花时间。可集合通信要把数据从一张卡传到别的卡上。和从显存读数据一样，传一次的时间约是传的字节数除以带宽，带宽即每秒最多能传的字节数（见${prLK('roofline')}）。数据传到之前，要用它的计算只能等着。所以多卡训练快不快，不只看每张卡算得多快，也看卡和卡之间连得多快；而同一台机器里的卡之间，和不同机器的卡之间，带宽差得很远。`),
   ()=>tx('Training H100s usually come 8 to a machine, as in NVIDIA’s DGX H100 server. All 8 GPUs are connected to the machine’s NVSwitch chips, so any two of them talk over NVLink, NVIDIA’s direct connection between GPUs. Over NVLink each H100 can send 450 GB/s and receive 450 GB/s at the same time. For comparison, a GPU reads its own memory at 3.35 TB/s, more than seven times NVLink.',
          '训练用的 H100 一般 8 张装在一台机器里，比如 NVIDIA 的 DGX H100 服务器。8 张卡都接在机器里的 NVSwitch 交换芯片上，任意两张卡之间都走 NVLink，这是 NVIDIA 专为 GPU 之间直连做的连接。H100 每张卡通过 NVLink 发出和收进各能到 450 GB/s。作为对照，一张卡读自己显存的带宽是 3.35 TB/s，是 NVLink 的七倍多。'),
   ()=>tx(`Data that leaves the machine goes over the network. Large training clusters often use InfiniBand, a fast network built for data centers: each machine connects through its network cards (NICs) to switches, and the switches to the other machines. Network speeds are counted in bits, 8 bits to a byte, so one 400 Gb/s InfiniBand NIC moves 50 GB/s in each direction. H100 training machines usually give every GPU its own such NIC, 8 per machine, ${prM('8 \\times 50 = 400')} GB/s in all. Even so, a GPU’s bandwidth out of its machine is only a ninth of its NVLink.`,
          `数据要出这台机器，就得走网络。大的训练集群常用 InfiniBand，一种为数据中心设计的高速网络：每台机器通过网卡连到交换机上，交换机再连到别的机器。网络的速度按位（bit）计，8 位是 1 字节，所以一块 400 Gb/s 的 InfiniBand 网卡每个方向能传 50 GB/s。H100 训练机器通常给每张卡配一块这样的网卡，一台机器 8 块，一共 ${prM('8 \\times 50 = 400')} GB/s。就算这样，一张卡通往机器外面的带宽也只有 NVLink 的九分之一。`),
   ()=>tx('Beyond 8 GPUs, some data has to cross between machines, and the network is often the slowest part of its way. So when training spreads over several machines, whether a way of splitting the work pays off depends largely on whether the data it exchanges can stay inside a machine on NVLink or has to go over the network.',
          '卡超过 8 张，就一定有数据要跨机器传，网络往往是这一路上最慢的一段。所以把训练分到多台机器上时，一种分法划不划算，很大程度上要看它交换的数据能留在一台机器里走 NVLink，还是得过网络。'))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the choices tried, the
   timelines already played to the end, the playing speed, the challenge's result, the drill. The playhead is not kept: a timeline
   still playing is recorded as played to its end, a timeline already played shows finished, and one not played yet plays from its
   start. */
const SK=Object.keys(BASE), okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), pickS=o=>Object.fromEntries(SK.map(k=>[k,o[k]]));
const cfgOf=x=>okObj(x)?parse(token(x)):null;
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:pickS(S),pred,observed:{...observed},saved:saved?pickS(saved):null,lv5,lv5Tried:[...lv5Tried],lv6,lv6Tried:[...lv6Tried],remView,spd,watched:[...new Set([...watched,...(playing&&SIM&&SIM._sig?[SIM._sig]:[])])],
    chSubmitted,chReveal,chAllOpen,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&evalC(chCfg(S.st,S.T)).ok};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[], inV=(L,v)=>L.some(x=>x.v===v);
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...(cfgOf(o.S)||BASE)};
    if(mode==='challenge'){const st=CH.sts.includes(S.st)?S.st:CH.start.st, T=CH.Ts.includes(S.T)?S.T:CH.start.T; S={...S,st,T,m:CH.m,N:CH.N,net:CH.net,ov:CH.ov,u:CH.u};}
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    lv5=inV(LV5,o.lv5)?o.lv5:'noov'; lv5Tried=new Set(['noov',...arr(o.lv5Tried,v=>inV(LV5,v)),lv5]);
    lv6=inV(LV6,o.lv6)?o.lv6:'z3'; lv6Tried=new Set(['z3',...arr(o.lv6Tried,v=>inV(LV6,v)),lv6]);
    remView=REMS.includes(o.remView)?o.remView:'dp'; spd=['slow','mid','fast'].includes(o.spd)?o.spd:'mid';
    watched.clear(); arr(o.watched,v=>typeof v==='string'&&v.length<80&&v.split('|').length===7&&!!MOD[v.split('|')[0]]).slice(0,200).forEach(v=>watched.add(v));
    pT=0; playing=false;
    saved=mode==='challenge'?cfgOf(o.saved):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; chAllOpen=chSubmitted&&o.chAllOpen===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el; curSig=''; SIM=null;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#zr-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#zr-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); clearTimeout(playT); stageStop(); pause(); root=null; app=null;},
  onLang(){if(root){const d=$('#zr-math',app), open=d&&d.open, was=playing, t0=pT, sg=curSig; pause(); relang=true; try{this.mount(root,null);} finally{relang=false;} if(sg===curSig){pT=t0; drawHead(); drawPlayer();} if(open) $('#zr-math',app).open=true; if(was) play();}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#zr-cols',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#zr-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(hidePred()){vis.innerHTML=''; read.innerHTML=esc(tx('Data parallel on 8 GPUs: answer in the card first','8 张卡做数据并行：先在卡片里作答')); return;}
      const e=evalC(S), w=Math.min(100,e.me.total/SCALE*100), c=CAP/SCALE*100;
      vis.innerHTML=`<div class="zr-dk"><i style="width:${w.toFixed(1)}%" class="${e.me.fits?'':'over'}"></i><b style="left:${c.toFixed(1)}%"></b></div>`;
      read.innerHTML=esc(tx(`${stName(S.st)}, ${S.N} GPUs: ${fGB(e.me.total)} per GPU, waits ${fPctW(e.s.share)}`,`${S.N} 张卡、${stName(S.st)}：每张卡 ${fGB(e.me.total)}，等数据 ${fPctW(e.s.share)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__zr=Object.freeze({
  mem:o=>deep(mem({...BASE,...o})), comm:o=>deep(comm({...BASE,...o})), sim:o=>{const s=sim({...BASE,...o}); return deep({C:s.C,tF:s.tF,tB:s.tB,step:s.step,wait:s.wait,share:s.share,mfu:s.mfu,n:s.ev.length,gaps:s.gaps.length});},
  evalC:o=>{const e=evalC({...BASE,...o}); return deep({total:e.me.total,lg:e.me.lg,fits:e.me.fits,share:e.s.share,step:e.s.step,mfu:e.s.mfu,bytes:e.k.bytes,ok:e.ok});},
  chAll:()=>chAll().map(e=>({st:e.c.st,T:e.c.T,total:e.me.total,share:e.s.share,step:e.s.step,ok:e.ok})), chBest:()=>{const b=chBest(); return {st:b.c.st,T:b.c.T};},
  drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({MOD:deep(MOD),PEAK,CAP,BW_IN,NET:{...NET},NS:NS.slice(),TS:TS.slice(),US:US.slice(),BASE:{...BASE},CH:deep(CH),FPT,MFPT,STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),P2:{...P2},P5:{...P5},P6:{...P6},DUR:{...DUR}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},lv5,lv5Tried:[...lv5Tried],lv6,lv6Tried:[...lv6Tried],remView,chSubmitted,chReveal,pT,playing,spd,step_s:SIM?SIM.step:0,watched:[...watched],sig:curSig,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});},
  /* the one change that helps most, for the challenge's result (R1) */
  chOne:(st,T)=>{const o=chOne(st,T); return o?{st:o.e.c.st,T:o.e.c.T,pass:o.pass}:null;},
  /* G7: the single-change sentence the result shows for a pick, in the page's language (empty for the best setting) */
  chNextText:(st,T)=>{const me=evalC(chCfg(st,T)), best=chBest(); return me.ok&&me.c.st===best.c.st&&me.c.T===best.c.T?'':chNext(me,best).replace(/<[^>]+>/g,'');},
  chShown:(st,T)=>{const e=evalC(chCfg(st,T)); return {gb:rGB(e.me.total),wait:rPct(e.s.share),step:rSec(e.s.step),fits:e.me.fits,ok:e.ok};},
  /* every sentence the playback caption can show for the current setup, with the widest numbers, measured in a hidden copy of the
     box: the phases whose text does not fit (R7.9: the box never grows, and its first line never wraps) */
  nowFit:()=>{const now=app&&$('#zr-now',app); if(!now||!SIM||!now.clientWidth) return null;
    const cl=now.cloneNode(false); cl.removeAttribute('id'); Object.assign(cl.style,{position:'absolute',visibility:'hidden',width:now.clientWidth+'px'}); now.parentNode.appendChild(cl);
    const out=[]; ['F','B','D'].forEach(ph=>{const [c,m]=nowFrame(ph,true); cl.innerHTML=`<div class="zr-nl zr-nl-c">${c}</div><div class="zr-nl zr-nl-n">${m}</div>`;
      cl.querySelectorAll('[data-n]').forEach(e=>{e.textContent={t:'99.9',l:'80',w:'9.99',k:'80'}[e.dataset.n]||'';});
      const lc=cl.querySelector('.zr-nl-c'); if(cl.scrollHeight>cl.clientHeight+1||lc.scrollWidth>lc.clientWidth+1) out.push(ph+': '+cl.innerText.replace(/\n/g,' | '));});
    cl.remove(); return out;}
});
})();
