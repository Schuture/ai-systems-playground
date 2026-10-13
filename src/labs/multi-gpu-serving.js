/* multi-gpu-serving (world inference): How is one model served across 8 GPUs?
   Analytical lab (fidelity "Formula"): the step model of the other inference labs (S2), divided over the GPUs of a tensor-parallel
   group, plus the time of the all-reduces each layer needs (a fixed cost per message plus the ring's bytes over NVLink or InfiniBand),
   pipeline stages and replicas. tests/multi-gpu-serving_numbers.py recomputes every number independently.
   Spec: specs/multi-gpu-serving.md; sources: sources/b10src/multi-gpu-serving/README.md. Prefix mgs. */
(function(){
'use strict';
const ID='multi-gpu-serving';

/* ENGINE BEGIN — pure functions, no DOM */
const MGS=(function(){
const GiB=2**30, USABLE=80*GiB*0.9, OVH=2*GiB, BS=3.35e12*0.6, PER_NODE=8;
const LINK={nvl:450e9, ib:50e9}, LAT={nvl:18e-6, ib:60e-6};
const MODELS={
 '70b':{n:'70B',L:80,d:8192,kvh:8,hd:128,ffn:28672,V:128256,wb:2,peak:989.5e12},
 '70b8':{n:'70B',L:80,d:8192,kvh:8,hd:128,ffn:28672,V:128256,wb:1,peak:1979e12},
 '405b8':{n:'405B',L:126,d:16384,kvh:8,hd:128,ffn:53248,V:128256,wb:1,peak:1979e12},
 '405b':{n:'405B',L:126,d:16384,kvh:8,hd:128,ffn:53248,V:128256,wb:2,peak:989.5e12}};
function shape(k){const M=MODELS[k], d=M.d, pa=2*d*d+2*d*M.kvh*M.hd, pm=3*d*M.ffn, pl=pa+pm+2*d, pls=M.L*pl+d, ph=M.V*d;
  return {M,pa,pm,pl,pls,ph,pt:pls+2*ph,kv:2*M.L*M.kvh*M.hd*2};}
/* one all-reduce of S bytes over a tensor-parallel group of tp GPUs: a fixed cost plus the ring's bytes (nccl-tests: 2(n-1)/n of the
   data per GPU). Across k nodes: reduce inside each node over NVLink, exchange 1/8 of the data across nodes (every GPU on its own
   NIC), gather back inside the node (ml-engineering's hierarchical model). */
function arTime(S,tp){
  if(tp<=1) return {t:0,lat:0,bwIn:0,bwX:0,cross:false};
  if(tp<=PER_NODE){const bwIn=2*(tp-1)/tp*S/LINK.nvl; return {t:LAT.nvl+bwIn,lat:LAT.nvl,bwIn,bwX:0,cross:false};}
  const g=PER_NODE, k=tp/g, bwIn=2*(g-1)/g*S/LINK.nvl, bwX=2*(k-1)/k*(S/g)/LINK.ib;
  return {t:LAT.ib+bwIn+bwX,lat:LAT.ib,bwIn,bwX,cross:true};}
const nodeOf=i=>Math.floor(i/PER_NODE);
/* c = {m: model key, g: GPUs (8 or 16), tp, pp, u: users in all, ctx: tokens of context each user's next token reads} */
function evalSetup(c){
  const s=shape(c.m), M=s.M, FS=M.peak*0.5, n=c.tp*c.pp, R=Math.max(1,Math.floor(c.g/n)), u=Math.ceil(c.u/R);
  const w=M.wb*s.pt/n, kvDiv=Math.min(c.tp,M.kvh)*c.pp, kvTok=s.kv/kvDiv;
  const room=USABLE-OVH-w, need=u*c.ctx*kvTok, fitW=room>0, fits=fitW&&need<=room;
  /* decode: the replica's users go in min(pp, u) micro-batches; every stage reads its weights once per micro-batch */
  const mb=Math.min(c.pp,u), b=Math.ceil(u/mb);
  const F=(2*s.pls*b+2*s.ph*b+b*4*M.L*M.d*(c.ctx+1))/(c.pp*c.tp);
  const By=M.wb*(s.pls+s.ph)/n+b*c.ctx*kvTok;
  const tc=F/FS, tm=By/BS, tcm=Math.max(tc,tm);
  const nAR=c.tp>1?2*M.L/c.pp:0, S=b*M.d*2, ar=arTime(S,c.tp), stage=tcm+nAR*ar.t;
  const sends=[];
  if(c.pp>1) for(let i=0;i<c.pp;i++){const x=nodeOf(i*c.tp)!==nodeOf(((i+1)%c.pp)*c.tp), bytes=i<c.pp-1?S:b*4;
    sends.push({x,bytes,t:(x?LAT.ib:LAT.nvl)+bytes/(x?LINK.ib:LINK.nvl)});}
  const tSend=sends.reduce((a,x)=>a+x.t,0), tpot=c.pp*stage+tSend, tComm=c.pp*nAR*ar.t+tSend;
  /* one layer of one micro-batch, split for the drawing: attention (its weights and the KV cache) vs MLP (its weights) by bytes */
  const lt=tcm/(M.L/c.pp), ab=M.wb*s.pa/c.tp+b*c.ctx*(s.kv/M.L)/Math.min(c.tp,M.kvh), mbb=M.wb*s.pm/c.tp, sh=ab/(ab+mbb);
  /* a new prompt of ctx tokens prefilled alone on one replica (its first token); stages one after the other */
  const Fp=(2*s.pls*c.ctx+2*s.ph+2*M.L*M.d*c.ctx*(c.ctx+1))/c.tp, Bp=M.wb*(s.pls+s.ph)/c.tp+c.ctx*s.kv/Math.min(c.tp,M.kvh);
  const Sp=c.ctx*M.d*2, arp=arTime(Sp,c.tp);
  let ttft=Math.max(Fp/FS,Bp/BS)+(c.tp>1?2*M.L*arp.t:0);
  for(let i=0;i<sends.length-1;i++) ttft+=(sends[i].x?LAT.ib:LAT.nvl)+Sp/(sends[i].x?LINK.ib:LINK.nvl);
  return {c:{...c},R,u,w,room,need,fitW,fits,kvTok,mb,b,F,By,tc,tm,tcm,nAR,S,ar,stage,sends,tSend,tpot,tComm,share:tComm/tpot,
    tps:fits?c.u/tpot/c.g:null,ttft,layer:{attn:lt*sh,mlp:lt*(1-sh),ar:ar.t},FS,s};}
function mulberry(seed){let h=seed>>>0; return ()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};}
/* MoE routing (the lab's assumption): each token picks 8 of 256 experts uniformly at random; 32 experts per GPU on 8 GPUs */
const EP={E:256,K:8,G:8,seed:11};
function route(nTok){const rnd=mulberry(EP.seed), per=new Array(EP.G).fill(0), per0=EP.E/EP.G;
  for(let t=0;t<nTok;t++){const ch=[]; while(ch.length<EP.K){const e=Math.floor(rnd()*EP.E); if(!ch.includes(e)) ch.push(e);} for(const e of ch) per[Math.floor(e/per0)]++;}
  return per;}
return {GiB,USABLE,OVH,BS,LINK,LAT,PER_NODE,MODELS,EP,shape,arTime,evalSetup,mulberry,route};
})();
/* ENGINE END */
window.MGS_ENGINE=MGS;

const A=window.ASP; if(!A) return;
const $=(s,r)=>A.$(s,r), $$=(s,r)=>A.$$(s,r), esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const zh=()=>A.lang()==='zh';
const tx=(en,z)=>zh()?zfix(z):en;
const L=o=>zh()?zfix(A.L(o)):A.L(o);
const DEF=(t,tip)=>A.ui.def(t,tip);
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):esc(t||id);
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id];   // a lab that is not live is named in plain text (R6.39)
  if(!l) return esc(id); if(C.labStatus&&C.labStatus(l)!=='live') return esc(L(l.s||{en:id,zh:id}));
  return labLink(id);};
const M=t=>A.ui.math(t);
const J_=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const cite=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const M70='Llama‑3.1‑70B', M405='Llama‑3.1‑405B', MDS='DeepSeek‑V3';

/* number formats */
/* fixed decimals: numbers side by side keep the same digits (R6.6), so 23.0 ms and 5.0 GiB keep their .0 */
const fx=(x,d)=>{const t=Math.abs(x)<0.5*10**-d?(0).toFixed(d):x.toFixed(d), [i,f]=t.split('.'); return i.replace(/\B(?=(\d{3})+(?!\d))/g,',')+(f?'.'+f:'');};
const f1=x=>fx(x,1), fint=n=>A.fmt.int(n);
const fMs=t=>zh()?`${f1(t*1e3)} 毫秒`:`${f1(t*1e3)} ms`;
const fUs=t=>zh()?`${Math.round(t*1e6)} 微秒`:`${Math.round(t*1e6)} µs`;
const fGi=b=>`${f1(b/MGS.GiB)} GiB`;
const fP=x=>`${Math.round(x*100)}%`;
const fTok=x=>fint(Math.round(x));
const fKiB=b=>b>=2**20?`${A.fmt.num(b/2**20,b>=10*2**20?0:1)} MiB`:`${fint(Math.round(b/1024))} KiB`;

/* ---------- definitions ---------- */
const TIP={
 tp:()=>A.terms.tp()+tx(` Each layer of ${M70}, used here, has 64 query heads and eight KV heads: split over eight GPUs, each GPU computes eight query heads and keeps one KV head.`,`这里的 ${M70} 每层有 64 个 query head、八个 KV head：分到八张卡上，每张卡算八个 query head，存一个 KV head。`),
 ar:()=>tx('All-reduce: every GPU holds a partial result of the same shape; they are added up value by value, and every GPU gets the sum. Tensor parallelism does it twice per layer: after the attention block and after the MLP block.','all-reduce：每张卡上有一份形状相同的部分结果，把它们逐个数值加起来，每张卡都拿到这个总和。张量并行每层做两次：attention 块之后一次，MLP 块之后一次。'),
 lat:()=>tx('The fixed cost of a message: before any byte moves, the GPUs must launch the exchange and wait for each other. For a small all-reduce it is most of the time. This lab takes 18 µs inside a node, from a measurement on eight H200s, and assumes 60 µs across nodes; the notes at the end of the page show where both come from.','一次通信的固定开销：一个字节还没传，卡之间先要发起通信、互相等齐。小的 all-reduce，大部分时间花在这上面。这个 Lab 在机器内按 18 微秒算（来自八张 H200 上的实测），跨机器按 60 微秒算（这个 Lab 的假设）；两个数怎么来的，见页末的“简化、假设和来源”。'),
 mem:()=>tx('Memory left for weights and KV cache = 80 GiB × 90% − 2 GiB = 70 GiB. The H100 is sold as 80 GB, counted here as GiB; 90% is vLLM’s default share up to v0.19 (92% from v0.20.0), and the runtime keeps about 2 GiB.','留给权重和 KV cache 的显存 = 80 GiB × 90% − 2 GiB = 70 GiB。H100 标的是 80 GB，这里按 GiB 计；90% 是 vLLM 在 v0.19 及以前的默认值（v0.20.0 起是 92%），运行时还要占约 2 GiB。'),
 routed:()=>tx(`Routed experts: the experts the router chooses from for each token. ${MDS} also has one shared expert that every token uses.`,`路由专家：路由器为每个 token 从中挑选的专家。${MDS} 另有一个共享专家，每个 token 都用。`),
 pp:()=>tx('Pipeline parallelism cuts the model by layers: each group of GPUs (a stage) keeps a run of consecutive layers. A token passes the stages one after another; between two stages only that token’s hidden vector moves.','流水线并行按层切：每一组卡（一段）存连续的几十层。一个 token 依次走过每一段，两段之间只传这个 token 的隐藏向量。'),
 rep:()=>tx('A replica is one whole copy of the model on its own GPUs (data parallelism). Replicas never talk to each other during a step; each serves its share of the users.','副本：在自己的几张卡上放一整份模型（数据并行）。副本之间在一步里不通信，各自服务分给它的用户。'),
 kvrep:()=>tx(`${M70} has only eight KV heads. With sixteen GPUs, vLLM keeps each KV head on two GPUs (vllm/model_executor/models/llama.py: “Number of KV heads is less than TP size, so we replicate the KV heads”), so each GPU still holds an eighth of every user’s KV cache.`,`${M70} 只有八个 KV head。十六张卡时，vLLM 让每个 KV head 在两张卡上各存一份（vllm/model_executor/models/llama.py：“Number of KV heads is less than TP size, so we replicate the KV heads”），所以每张卡上仍是每个用户 KV cache 的八分之一。`),
 tpot:()=>tx('Time per output token (TPOT): once the answer has started, how long one token takes on average. Here every decode step gives every user one token, so it is the time of one decode step (one round of the pipeline).','每个输出 token 的耗时（TPOT）：回答开始以后，平均多久出一个 token。这里每一步 decode 给每个用户各出一个 token，所以它就是一步 decode 的时间（流水线时是走完一轮的时间）。'),
 ttft:()=>tx('Time to first token (TTFT): from the moment a request arrives to its first answer token. Here: a new 4,096-token prompt prefilled alone on one replica, with no queue.','首 token 延迟（TTFT）：从请求到达，到它的第一个回答 token。这里算的是一个 4,096 个 token 的新 prompt 单独在一个副本上做 prefill，不排队。'),
 tps:()=>tx('Output tokens per second per GPU: all the users’ tokens in one second, divided by the GPUs. It says how much a GPU earns, while TPOT says how fast each user reads.','每张卡每秒出的 token：一秒里所有用户拿到的 token，除以卡数。它说明一张卡干了多少活；TPOT 说明每个用户读得多快。'),
 nvl:()=>tx('NVLink on an H100 SXM (the fourth generation): 450 GB/s in one direction per GPU (the 900 GB/s often quoted counts both directions); all eight GPUs of a node reach each other through NVSwitch.','H100 SXM 的 NVLink（第四代）：每张卡单向 450 GB/s（常说的 900 GB/s 是两个方向加起来），一台机器里的八张卡经 NVSwitch 互相连通。'),
 ib:()=>tx('Across nodes: one 400 Gb/s InfiniBand NIC per GPU, 50 GB/s in one direction, a ninth of NVLink.','跨机器：每张卡一个 400 Gb/s 的 InfiniBand 网卡，单向 50 GB/s，是 NVLink 的九分之一。'),
 ep:()=>tx('Expert parallelism (EP): the experts of each MoE layer are spread over GPUs, each GPU keeping a few whole experts. A token’s hidden vector is sent to the GPUs of the experts it chose, and their results come back.','专家并行（EP）：把每个 MoE 层的专家分到不同的卡上，每张卡存几个完整的专家。token 的隐藏向量发到它选中的专家所在的卡，算完再把结果发回来。'),
 a2a:()=>tx('All-to-all: every GPU sends a different piece of data to every other GPU at the same time. MoE does it twice per layer: dispatch (tokens to their experts) and combine (results back).','all-to-all：每张卡同时给其他每张卡各发一份不同的数据。MoE 每层做两次：dispatch（把 token 发给专家）和 combine（把结果收回来）。'),
 moe:()=>tx(`Mixture of experts (MoE): the MLP block is replaced by many experts, and a router picks a few of them for each token. ${MDS} has 256 routed experts per MoE layer and picks 8 for each token.`,`混合专家（MoE）：MLP 块换成很多个专家，路由器给每个 token 选其中几个。${MDS} 每个 MoE 层有 256 个路由专家，每个 token 选 8 个。`)};
const SRC={
 vllm:'https://github.com/vllm-project/vllm/blob/ed7ae705a1f366c2008964af562a7543e91d934e/docs/serving/parallelism_scaling.md',
 llama:'https://github.com/vllm-project/vllm/blob/ed7ae705a1f366c2008964af562a7543e91d934e/vllm/model_executor/models/llama.py',
 vep:'https://github.com/vllm-project/vllm/blob/ed7ae705a1f366c2008964af562a7543e91d934e/docs/serving/expert_parallel_deployment.md',
 net:'https://github.com/stas00/ml-engineering/blob/fb75494c0d9e536f048e27be2869a77fe65ad9f4/network/README.md',
 nccl:'https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/doc/PERFORMANCE.md',
 ds:'https://github.com/deepseek-ai/open-infra-index/blob/56d86855fcf6e08fdfd45ce6280bd24322c93351/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md',
 dscfg:'https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/configs/config_671B.json',
 deepep:'https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/README.md',
 eplb:'https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913d/README.md',
 p_mega:'https://arxiv.org/abs/1909.08053', p_pope:'https://arxiv.org/abs/2211.05102', p_ds:'https://arxiv.org/abs/2412.19437'};
const dTP=t=>DEF(t,TIP.tp()), dAR=t=>DEF(t,TIP.ar()), dLat=t=>DEF(t,TIP.lat()), dPP=t=>DEF(t,TIP.pp()), dRep=t=>DEF(t,TIP.rep()), dTPOT=t=>DEF(t,TIP.tpot());

/* ---------- the settings ---------- */
const MKEYS=['70b','70b8','405b8','405b'];
const MNAME={'70b':{en:`${M70}, BF16`,zh:`${M70}，BF16`},'70b8':{en:`${M70}, FP8`,zh:`${M70}，FP8`},'405b8':{en:`${M405}, FP8`,zh:`${M405}，FP8`},'405b':{en:`${M405}, BF16`,zh:`${M405}，BF16`}};
const MSHORT={'70b':'70B BF16','70b8':'70B FP8','405b8':'405B FP8','405b':'405B BF16'};
const BASE={m:'70b',g:8,tp:8,pp:1,u:32,ctx:4096};
const USERS=[8,32,128,256], CTXS=[1024,4096,16384], TPS=[1,2,4,8,16], PPS=[1,2,4];
/* 8 GPUs: keep the TP size up to 8 and the most pipeline stages that still fit */
function fit8(){const tp=Math.min(S.tp,8); let pp=S.pp; while(tp*pp>8) pp/=2; return {g:8,tp,pp:Math.max(1,pp)};}
function norm(s){const o={...s}; if(o.tp*o.pp>16) o.pp=Math.max(1,16/o.tp); if(o.tp*o.pp>8) o.g=16; return o;}
const E=c=>MGS.evalSetup(c);
const setupName=c=>{const R=Math.max(1,Math.floor(c.g/(c.tp*c.pp)));
  if(c.tp*c.pp===1) return tx('each GPU keeps a whole copy','每张卡各放一整份模型');
  const core=c.pp>1?tx(`TP ${c.tp} × ${c.pp} pipeline stages`,`TP ${c.tp} × 流水线 ${c.pp} 段`):`TP ${c.tp}`;
  return R>1?tx(`${R} replicas × ${core}`,`${R} 个副本 × ${core}`):core;};
const gpusName=g=>g>8?tx('16 H100s in 2 nodes','2 台机器、16 张 H100'):tx('8 H100s in 1 node','1 台机器、8 张 H100');

/* ---------- guided state ---------- */
let S={...BASE}, mode='guided', step=0, pred=null, root=null, app=null;
let p3=null, p5=8, p6=null, p7=32, tried={}, saved=null, chSubmitted=false, chReveal=false, annT=null, stageT=0, userAct=false, spdUser=null;
const watched=new Set();
let PLAY={on:false,raf:0,last:0,key:''}, ph=1, lastKey='', nPlays=0;
const PRED=[{v:'2x',t:{en:'About twice as fast',zh:'快了约一倍'}},{v:'bit',t:{en:'A little faster',zh:'快一点'}},{v:'same',t:{en:'About the same',zh:'差不多'}},{v:'slow',t:{en:'Slower than TP 8',zh:'反而比 TP 8 慢'}}];
const PRED_RIGHT='slow';
const V1={m:'70b',g:8,tp:1,pp:1,u:32,ctx:4096};
const V8={...BASE}, V16={...BASE,g:16,tp:16}, VPP={...BASE,g:16,tp:8,pp:2}, VR2={...BASE,g:16,tp:8,pp:1};
const P3={2:{...BASE,tp:2},4:{...BASE,tp:4},8:{...BASE}};
const P6={tp16:V16,pp:VPP,rep:VR2};
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const isP=id=>mode==='guided'&&STEPS[step].id===id;
const seenP=id=>mode!=='guided'||step>=SIX(id);
function viewCfg(){
  if(mode!=='guided') return norm(S);
  const id=STEPS[step].id;
  if(id==='mission') return V1;
  if(id==='size') return P3[p3||8];
  if(id==='cross') return p5===16?V16:V8;
  if(id==='pp') return P6[p6||'tp16'];
  if(id==='moe'||id==='remember') return V8;
  return V8;}
const cur=()=>E(viewCfg());
const viewKey=c=>[c.m,c.g,c.tp,c.pp,c.u,c.ctx].join('|');
const RUN_SECS={slow:12,med:6,fast:3};
const spdNow=()=>spdUser||'med';   // a step-by-step mechanism replay defaults to medium (R3.9)

/* ---------- numbers the copy quotes, all from the engine ---------- */
const R1=()=>E(V1), R8=()=>E(V8), R16=()=>E(V16), RPP=()=>E(VPP), RR2=()=>E(VR2);
const oneGpuRead=()=>{const s=MGS.shape('70b'); return 2*(s.pls+s.ph)/MGS.BS;};
/* the same work as TP 8 serving 32 users (the weights read each step plus their KV cache), on one GPU if it fit: the baseline of part 2 */
const oneGpuAll=()=>{const s=MGS.shape('70b'); return (2*(s.pls+s.ph)+32*4096*s.kv)/MGS.BS;};

/* ---------- the guided parts ---------- */
const STEPS=[
 {id:'mission',title:{en:'One GPU cannot hold a 70B model',zh:'一张卡放不下 70B'},lead:{en:'32 users, 8 H100s, one model.',zh:'32 个用户，8 张 H100，一个模型。'},
  body:()=>{const s=MGS.shape('70b'), wb=2*s.pt;
   return tx(`<p>In BF16, ${M70}’s weights take ${fGi(wb)}, and one H100 has only ${DEF('70 GiB',TIP.mem())} for weights and KV cache. The weights alone do not fit (red in the chart).</p><p>Even if they fit, one GPU would be slow. Every decode step reads the weights once (${labLink('prefill-decode')}); the embedding table is only looked up a few rows, so about ${A.fmt.num(2*(s.pls+s.ph)/1e9,0)} GB are read, about ${fMs(oneGpuRead())} on one GPU.</p><p>So the model is split over several GPUs: each keeps a part and reads only its part every step. How should one node of 8 H100s split it for its 32 users?</p><p>Below the memory chart is a timeline of one of the 80 layers: each GPU first reads memory and computes, then merges its results with the other GPUs (the merge, called all-reduce, comes in part 2). Right now one GPU cannot even hold the weights, so the timeline only says “does not fit”.</p>`,
              `<p>BF16 的 ${M70} 权重要 ${fGi(wb)}，一张 H100 留给权重和 KV cache 的只有 ${DEF('70 GiB',TIP.mem())}。光是权重就放不下（图里的红色）。</p><p>就算放得下，一张卡也太慢。decode 每一步都要把权重读一遍（见 ${labLink('prefill-decode')}）；embedding 表每步只查用到的几行，不用整个读，所以要读的是约 ${A.fmt.num(2*(s.pls+s.ph)/1e9,0)} GB，一张卡要约 ${fMs(oneGpuRead())}。</p><p>所以要把模型分到几张卡上：每张卡只存一部分，每一步也只读自己那部分。一台 8 张 H100 的机器要服务 32 个用户，该怎么分？</p><p>显存图下面是 80 层中一层的时间线：每张卡先读显存和计算，再和其他卡合并结果（合并叫 all-reduce，第 2 部分讲）。现在一张卡连权重都放不下，所以这条时间线上只标着“放不下”。</p>`);}},
 {id:'tp',title:{en:'Tensor parallelism: each GPU keeps a slice of every matrix',zh:'张量并行：每张卡存每个矩阵的一块'},lead:{en:'TP 8: two merges per layer.',zh:'TP 8：每层合并两次。'},
  body:()=>tx(`<p>${dTP('Tensor parallelism (TP)')}, a kind of model parallelism, cuts every matrix of every layer into slices, one per GPU. Here the 8 GPUs form one group, written TP 8 from now on. The “${labLink('model-parallel')}” lab shows the cut for training; serving cuts the same way.</p><p>Attention is cut by head, so each user’s KV cache is split too. A GPU holding a slice computes only a partial result, so after the attention block and again after the MLP block the GPUs add their partial results together: an ${dAR('all-reduce')}, twice per layer.</p><p>In decode each merge is small: each user adds only one hidden vector of 8,192 values in BF16 (16 KiB). The chart plays one of the 80 layers on each GPU: first reading memory and computing, then the merge, which all GPUs do together.</p>`,
              `<p>${dTP('张量并行（TP）')}是模型并行的一种：把每一层的每个矩阵切成几块，每张卡存一块。这里 8 张卡组成一组，下面记作 TP 8。训练时怎么切，“${labLink('model-parallel')}”这个 Lab 讲过；推理时切法一样。</p><p>attention 按 head 切，所以每个用户的 KV cache 也分开了。每张卡只有矩阵的一块，算出来的只是部分结果；attention 块和 MLP 块算完以后，各张卡都要把部分结果加起来，这叫 ${dAR('all-reduce')}，每层两次。</p><p>decode 时每次合并的数据很少：每个用户只有一个 8,192 维的隐藏向量，用 BF16 存（16 KiB）。图里播的是 80 层中的一层：每张卡先读显存和计算，再和其他卡一起合并。</p>`),
  obs:()=>{const r=R8(), one=oneGpuAll();
   return tx(`<p><b>A token takes ${fMs(r.tpot)} here, about ${A.fmt.num(one/r.tpot,1)} times faster than one GPU doing the same work.</b></p><p>One GPU would need ${fMs(one)} to read the weights and the users’ KV cache (if it fit). On 8 GPUs, reading and computing take an eighth of that; the rest is the merges, about a fifth of the step.</p><p>Each merge carries only ${fKiB(r.S)}, which NVLink moves almost at once; most of a merge is its ${dLat('fixed cost')} of about 18 µs, paid before any byte moves.</p>`,
              `<p><b>这里每个 token 要 ${fMs(r.tpot)}，是一张卡做同样的活的约 ${A.fmt.num(one/r.tpot,1)} 倍快。</b></p><p>一张卡读一遍权重和这些用户的 KV cache（假设放得下）要 ${fMs(one)}。分到 8 张卡上，读显存和计算正好是它的八分之一；剩下的是合并，约占一步的五分之一。</p><p>每次合并只有 ${fKiB(r.S)}，NVLink 几乎一下就传完了；合并的时间大部分是${dLat('固定开销')}，约 18 微秒，一个字节还没传就先花掉了。</p>`);},always:true},
 {id:'size',title:{en:'How big should the group be?',zh:'一组用几张卡？'},lead:{en:'One group, or several replicas.',zh:'一个组，还是几个副本。'},expect:()=>Object.keys(tried).filter(k=>/^s/.test(k)).length>=2,
  body:()=>tx(`<p>The 8 GPUs can also form smaller groups, each holding a whole copy of the model: ${dRep('replicas')} (data parallelism), each serving its share of the users. “4 replicas, TP 2 each” means every 2 GPUs form a group with a whole copy of the model, and each group serves 8 of the 32 users.</p><p class="small">Pick a way to split the 8 GPUs and watch the memory per GPU, the ${dTPOT('time per output token')} and the tokens each GPU produces per second.</p>`,
              `<p>8 张卡也可以分成更小的组，每组放一整份模型，这叫${dRep('副本')}（数据并行），各服务一部分用户。比如“4 个副本，每个 TP 2”，就是每两张卡一组，各放一整份模型，每组服务 32 个用户里的 8 个。</p><p class="small">选一种分法，看看每张卡的显存、${dTPOT('每个输出 token 的耗时')}和每张卡每秒出的 token 怎么变。</p>`)
   +A.ui.levers({aria:tx('How to split 8 GPUs','8 张卡怎么分'),value:p3==null?null:String(p3),options:[2,4,8].map(t=>({v:String(t),title:t===8?tx('One TP 8 group','一个 TP 8 的组'):tx(`${8/t} replicas, TP ${t} each`,`${8/t} 个副本，每个 TP ${t}`),desc:'',result:sizeRes(t)}))}),
  obs:()=>{const t=p3, r=E(P3[t]);
   if(!r.fits) return tx(`<p class="badline"><b>Does not fit: each GPU keeps ${fGi(r.w)} of weights and has ${fGi(r.room)} left for its users’ KV cache.</b></p><p>The KV cache of the users given to this replica needs ${fGi(r.need)} on each GPU. Two GPUs are the fewest that hold the weights at all, and then almost nothing is left for the users.</p>`,
                         `<p class="badline"><b>放不下：每张卡要存 ${fGi(r.w)} 的权重，只剩 ${fGi(r.room)} 给它的用户存 KV cache。</b></p><p>分到这个副本的用户，KV cache 在每张卡上要 ${fGi(r.need)}。两张卡是能放下权重的最少卡数，这时几乎没有地方留给用户了。</p>`);
   const o=E(P3[t===8?4:8]), r4=t===8?o:r, r8=t===8?r:o;
   return tx(`<p><b>${t===8?`One group of eight: ${fMs(r.tpot)} per output token, against ${fMs(o.tpot)} for two replicas of four GPUs.`:`Two replicas of four GPUs: ${fMs(r.tpot)} per output token, against ${fMs(o.tpot)} for one group of eight.`}</b></p><p>Decode time goes mostly into reading the weights, and every replica reads all of them every step, however few users it has. One group of eight reads them once for all the users, spread over its GPUs, so each GPU also produces more tokens per second (metric cards).</p><p>Its price is the merges: the larger the group, the larger their share of the step (about ${fP(r4.share)} with four GPUs, ${fP(r8.share)} with eight).</p>`,
              `<p><b>${t===8?`一个八张卡的组：每个输出 token ${fMs(r.tpot)}；两个四张卡的副本是 ${fMs(o.tpot)}。`:`两个四张卡的副本：每个输出 token ${fMs(r.tpot)}；一个八张卡的组是 ${fMs(o.tpot)}。`}</b></p><p>decode 的时间主要花在读权重上。每个副本每一步都要把全部权重读一遍，不管它分到几个用户；八张卡的组只为全部用户读一遍，还分给了八张卡，所以每张卡每秒出的 token 也更多（见指标卡）。</p><p>代价是合并：组越大，合并在一步里占的比例越高（四张卡约 ${fP(r4.share)}，八张卡约 ${fP(r8.share)}）。</p>`);}},
 {id:'predict',title:{en:'Predict: TP 16 across two nodes',zh:'先猜：TP 16 跨两台机器'},lead:{en:'Is 16 GPUs faster than 8?',zh:'16 张卡会比 8 张快吗？'},gate:()=>!!pred,
  body:()=>{const r=R8();
   return tx(`<p>Add a second node and make one group of all sixteen GPUs: each GPU reads half the weights. With the group of eight (the chart), a step is ${fMs(r.tcm)} of reading and computing plus ${fMs(r.tComm)} of merges.</p><p>The two nodes are linked by ${DEF('InfiniBand',TIP.ib())}, a ninth of ${DEF('NVLink',TIP.nvl())}’s speed. In this lab a merge across nodes has a fixed cost of about 60 µs, more than three times the 18 µs inside a node.</p><p>With TP 16 across the two nodes, what happens to the time per output token?</p>`+predHTML(),
              `<p>再加一台机器，十六张卡组成一个组：每张卡要读的权重减半。八张卡的组（见图）一步是读显存和计算 ${fMs(r.tcm)}，加上合并 ${fMs(r.tComm)}。</p><p>两台机器之间走 ${DEF('InfiniBand',TIP.ib())}，速度是 ${DEF('NVLink',TIP.nvl())} 的九分之一。在这个 Lab 里，一次跨机器的合并固定开销约 60 微秒，是机器内 18 微秒的三倍多。</p><p>跨两台机器用 TP 16 时，每个输出 token 的耗时会怎样？</p>`+predHTML());}},
 {id:'cross',title:{en:'Tensor parallelism across nodes',zh:'跨机器的张量并行'},lead:{en:'One node against two.',zh:'一台机器和两台比。'},expect:()=>!!tried.x16,
  body:()=>verdictHTML()+tx('<p class="small">Pick a layout and watch where one layer’s time goes.</p>','<p class="small">选一种放法，看看一层的时间花在哪里。</p>')
   +A.ui.levers({aria:tx('Layout','放法'),value:String(p5),options:[{v:'8',title:tx('TP 8 in one node','TP 8，一台机器'),desc:'',result:tried.x8||p5===8?fMs(R8().tpot):''},{v:'16',title:tx('TP 16 across two nodes','TP 16，跨两台机器'),desc:'',result:tried.x16?fMs(R16().tpot):''}]}),
  obs:()=>{const r=R16(), q=R8();
   if(p5!==16) return tx(`<p><b>In one node: ${fMs(q.tpot)} per output token, merges ${fP(q.share)} of the step.</b> Now try two nodes.</p>`,`<p><b>一台机器：每个输出 token ${fMs(q.tpot)}，合并占一步的 ${fP(q.share)}。</b>再试试两台机器。</p>`);
   return tx(`<p class="badline"><b>Across two nodes a token takes ${fMs(r.tpot)}, slower than ${fMs(q.tpot)} in one node.</b></p><p>Reading and computing fell from ${fMs(q.tcm)} to ${fMs(r.tcm)}, not to half: with only eight KV heads, ${DEF('each KV head is kept on two GPUs',TIP.kvrep())}, so each GPU reads as much KV cache as before.</p><p>The merges now take more than half the step (red in the chart). Each pays a fixed cost of about 60 µs across nodes, this lab’s assumption (the fidelity note says where the result flips), so a NIC with more bandwidth alone would hardly help.</p><p>A prefill is different: it computes long enough that halving the compute saves more than its larger merges cost, so TP 16 gives a sooner first token here (metric cards).</p>`,
              `<p class="badline"><b>跨两台机器，每个 token 要 ${fMs(r.tpot)}，比一台机器的 ${fMs(q.tpot)}还慢。</b></p><p>读显存和计算从 ${fMs(q.tcm)}降到 ${fMs(r.tcm)}，没能减半：模型只有八个 KV head，${DEF('每个 KV head 要在两张卡上各存一份',TIP.kvrep())}，每张卡要读的 KV cache 没有变少。</p><p>合并占了一步的一半以上（图里的红色）。每次跨机器都有约 60 微秒的固定开销，这是这个 Lab 的假设（页末的“简化、假设和来源”里写了取多少时结论会反过来），所以只换带宽更大的网卡也帮不了多少。</p><p>prefill 不一样：它要算很久，把计算减半省下的时间，比更大的合并多花的还多，所以在这里 TP 16 的首 token 反而更快（见指标卡）。</p>`);}},
 {id:'pp',title:{en:'Pipeline stages and replicas across nodes',zh:'跨机器：流水线和副本'},lead:{en:'Three ways to use 16 GPUs.',zh:'16 张卡的三种用法。'},expect:()=>!!(tried.pp&&tried.rep),
  body:()=>tx(`<p>${dPP('Pipeline parallelism')} cuts by layer: the first node keeps the first half of the layers, the second node the other half, with TP 8 inside each node (the “${labLink('model-parallel')}” lab). A token passes the two stages one after the other; between them only its hidden vector moves, 16 KiB per user.</p><p>The pipeline does not cut the weights read for each token: the two stages together read as many as one group of eight. What it buys is twice the memory, and the users split into two batches that the stages work on at once.</p><p>The other way is two replicas: each node keeps a whole copy with TP 8 and serves half the users. <span class="small">Pick a way to use the sixteen GPUs and compare.</span></p>`,
              `<p>${dPP('流水线并行')}按层切：第一台机器存前一半的层，第二台存后一半，每台机器里再用 TP 8（见“${labLink('model-parallel')}”这个 Lab）。一个 token 依次走过两段，两段之间只传它的隐藏向量，每个用户 16 KiB。</p><p>流水线不会减少每个 token 要读的权重：两段读的加起来，和一个八张卡的组一样多。它换来的是显存翻倍，用户分成两批，两段可以同时各处理一批。</p><p>另一种是两个副本：每台机器各放一整份模型，用 TP 8，各服务一半的用户。<span class="small">选一种十六张卡的用法，比一比。</span></p>`)
   +A.ui.levers({aria:tx('How to use 16 GPUs','16 张卡怎么用'),value:p6,options:[{v:'tp16',title:tx('One TP 16 group','一个 TP 16 的组'),desc:'',result:tried.tp16||p6==='tp16'?fMs(R16().tpot):''},{v:'pp',title:tx('TP 8 × 2 pipeline stages','TP 8 × 流水线 2 段'),desc:'',result:tried.pp?fMs(RPP().tpot):''},{v:'rep',title:tx('2 replicas, TP 8 each','2 个副本，每个 TP 8'),desc:'',result:tried.rep?fMs(RR2().tpot):''}]}),
  obs:()=>{const b=RPP(), c=RR2(), q=R8();
   return tx(`<p><b>The pipeline (${fMs(b.tpot)}) and the two replicas (${fMs(c.tpot)}) are about equally fast, and both beat one group of sixteen.</b></p><p>Both are also a little faster than one node (${fMs(q.tpot)}): the KV cache is spread over sixteen GPUs, so each GPU reads half as much. The pipeline adds only two small messages across nodes per step.</p><p>When the model fits in one node, replicas are the simplest. When it does not (${M405} in BF16 needs ${fGi(2*MGS.shape('405b').pt)} of weights), vLLM’s advice is TP inside a node and pipeline stages across nodes.</p>`,
              `<p><b>流水线（${fMs(b.tpot)}）和两个副本（${fMs(c.tpot)}）差不多快，都比一个十六张卡的组快。</b></p><p>两种也都比一台机器（${fMs(q.tpot)}）快一点：KV cache 分到了十六张卡上，每张卡要读的只有一半。流水线每步只多了两次跨机器的小消息。</p><p>模型在一台机器里放得下时，最简单的是多放几个副本。放不下时（BF16 的 ${M405} 光权重就要 ${fGi(2*MGS.shape('405b').pt)}），vLLM 的建议是机器内用 TP，机器之间用流水线。</p>`);}},
 {id:'moe',title:{en:'MoE: experts on different GPUs',zh:'MoE：专家分到不同的卡上'},lead:{en:'Each token visits the GPUs of its experts.',zh:'每个 token 要去找它选中的专家。'},expect:()=>!!tried.m,
  body:()=>tx(`<p>A ${DEF('mixture-of-experts (MoE)',TIP.moe())} model replaces the MLP block with many experts, and each token uses only a few (${LL('moe')}). ${MDS} has 256 ${DEF('routed experts',TIP.routed())} in each MoE layer and picks 8 for every token.</p><p>${DEF('Expert parallelism (EP)',TIP.ep())} puts different experts on different GPUs; here eight GPUs keep 32 experts each. In every MoE layer a token’s hidden vector goes to the GPUs of its experts and the results come back: an ${DEF('all-to-all',TIP.a2a())}, twice per layer.</p><p>This part looks only at how the work of one layer is shared, not at whether the model fits: at 671B parameters, ${MDS} does not fit in one node of H100s. <span class="small">Each token picks 8 experts, so it brings 8 pieces of work, one token–expert pair each; a pair is computed by the GPU that holds its expert, so a GPU’s load is how many pairs in this step chose an expert on that GPU. Pick how many tokens are in the step (in decode, roughly how many users are being served at once) and watch the busiest GPU against the average (the chart).</span></p>`+moeSeg(),
              `<p>${DEF('混合专家（MoE）',TIP.moe())}模型把 MLP 块换成很多个专家，每个 token 只用其中几个（见 ${LL('moe')}）。${MDS} 每个 MoE 层有 256 个${DEF('路由专家',TIP.routed())}，每个 token 选 8 个。</p><p>${DEF('专家并行（EP）',TIP.ep())}把专家分到不同的卡上：这里八张卡，每张放 32 个。每个 MoE 层，token 的隐藏向量要发到它选中的专家所在的卡，算完再把结果发回来，这叫 ${DEF('all-to-all',TIP.a2a())}，每层两次。</p><p>这一部分只看一层的活怎么分，不管放不放得下：${MDS} 有 671B 个参数，一台 H100 的机器放不下。<span class="small">每个 token 选 8 个专家，就带来 8 份活，每份叫一个 token–专家对；一个 token–专家对，由放着那个专家的卡来算；所以一张卡要算多少对，就看这一步里有多少对选中的是这张卡上的专家。选这一步有多少个 token（decode 时大约就是同时在服务的用户数），看看最忙的卡比平均忙多少（见图）。</span></p>`+moeSeg()),
  obs:()=>{const per=MGS.route(p7), avg=p7*MGS.EP.K/MGS.EP.G, mx=Math.max(...per), k=mx/avg;
   return tx(`<p><b>With ${fint(p7)} tokens, the busiest GPU has ${A.fmt.num(k,2)} times the average work.</b></p><p>The layer waits for the busiest GPU; the others sit idle meanwhile. The more tokens in the step, the more the random ups and downs even out. This lab picks experts uniformly at random; vLLM’s documentation says real routing “can be highly skewed”.</p><p>${MDS}’s own serving copies its busiest experts (${cite(SRC.eplb,'EPLB')}), and in decode spreads one unit over 18 nodes and serves a great many tokens at once; with only 2 routed experts per GPU, each expert gets many tokens, and the load evens out (${cite(SRC.ds,'DeepSeek’s inference overview')}). vLLM can also serve MoE models with expert parallelism; across nodes its documentation recommends ${cite(SRC.deepep,'DeepEP')}’s kernels for the all-to-all.</p>`,
              `<p><b>${fint(p7)} 个 token 时，最忙的卡要做平均的 ${A.fmt.num(k,2)} 倍。</b></p><p>这一层要等最忙的那张卡算完、发完，其他卡这时只能闲着。一步里的 token 越多，随机的起伏越会互相抵消。这个 Lab 让每个 token 均匀随机地选专家；vLLM 的文档说，真实的路由“可能非常不均匀”（can be highly skewed）。</p><p>${MDS} 自己的推理服务把负载重的专家多复制几份（${cite(SRC.eplb,'EPLB')}）；decode 时一个部署单元跨 18 台机器，同时服务的 token 很多；每张卡只放 2 个路由专家，所以分到每个专家的 token 也多，各张卡的忙闲容易拉平（${cite(SRC.ds,'DeepSeek 的推理系统介绍')}）。vLLM 也能用专家并行跑 MoE 模型；跨机器时，它的文档推荐用 ${cite(SRC.deepep,'DeepEP')} 的 kernel 做 all-to-all。</p>`);}},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Four rules, two misconceptions.',zh:'四条规律，两个误解。'},
  body:()=>{const r8=R8(), r16=R16(), s70=MGS.shape('70b');
   return tx(`<ol class="mgs-rem"><li><p><b>First check that it fits, then how fast it is.</b> ${M70} in BF16 needs ${fGi(2*s70.pt)} of weights, more than one GPU holds. With two GPUs almost nothing is left for KV cache, so 70B needs at least four to leave room for its users.</p></li><li><p><b>A larger TP group reads less per GPU, but its merges cost the same each time.</b> Decode merges are small, so their time is mostly the fixed cost. Under this lab’s assumed 60 µs per merge across nodes, ${M70} gets its tokens slower from two nodes with TP 16 than from one node; at a smaller cost this flips (the fidelity note says where).</p></li><li><p><b>Across nodes, use pipeline stages or replicas.</b> They send only a few small messages across per step. A much larger model can still win with TP across nodes: in this lab ${M405} in BF16 is faster with one group of sixteen than with the pipeline (Scenarios).</p></li><li><p><b>An MoE layer waits for its busiest GPU.</b> With ${fint(32)} tokens in a step the busiest GPU does about ${A.fmt.num(Math.max(...MGS.route(32))/32,1)} times the average; with thousands of tokens the load nearly evens out. Real routing is less even, so vLLM and DeepSeek copy busy experts.</p></li></ol>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “More GPUs, faster tokens.” In this lab, ${M70}’s users get a token every ${fMs(r16.tpot)} from two nodes and every ${fMs(r8.tpot)} from one.</p><p>✗ “A merge moves only a few hundred KiB, so it costs nothing.” Each merge first pays a fixed cost, about 18 µs in a node here, and a step has 160 of them.</p></div>`+
     `<div class="obs"><div class="eyebrow">What real systems do</div><ul class="mgs-sys"><li>vLLM: “use tensor parallelism” when the model fits in one node; across nodes, the tensor parallel size is the GPUs per node and the pipeline parallel size the nodes (${cite(SRC.vllm,'parallelism and scaling')}).</li><li>vLLM: when the TP size exceeds the KV heads, the KV heads are replicated (${cite(SRC.llama,'llama.py')}).</li><li>${MDS}: expert parallelism with redundant copies of busy experts; prefill units span 4 nodes, decode units 18 (${cite(SRC.ds,'inference overview')}).</li><li>vLLM: <code>--enable-expert-parallel</code>; across nodes its documentation recommends DeepEP kernels, and it has a load balancer for experts (${cite(SRC.vep,'expert parallel deployment')}).</li></ul></div>`+A.shell.toScenarios(),
              `<ol class="mgs-rem"><li><p><b>先看放不放得下，再看快不快。</b>BF16 的 ${M70} 权重要 ${fGi(2*s70.pt)}，一张卡放不下。两张卡时几乎没有地方留给 KV cache，所以 70B 至少要四张卡才留得出用户的空间。</p></li><li><p><b>TP 组越大，每张卡读得越少，但每次合并的开销不变。</b>decode 的合并消息很小，时间主要是固定开销。按这个 Lab 假设的跨机器每次 60 微秒，${M70} 用两台机器的 TP 16 出 token 比一台机器还慢；这笔开销小一些，结论就会反过来（页末的“简化、假设和来源”里写了界线）。</p></li><li><p><b>跨机器用流水线或副本。</b>它们每步只有几次小消息跨机器。比 70B 大得多的模型，跨机器的 TP 也可能划算：在这个 Lab 里，BF16 的 ${M405} 用十六张卡的组比用流水线快（见应用场景）。</p></li><li><p><b>MoE 层要等最忙的那张卡。</b>一步只有 ${fint(32)} 个 token 时，最忙的卡要做平均的约 ${A.fmt.num(Math.max(...MGS.route(32))/32,1)} 倍；token 多到上千个，忙闲就几乎拉平了。真实的路由更不均匀，所以 vLLM 和 DeepSeek 都会把忙的专家多复制几份。</p></li></ol>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “卡越多越快。”在这个 Lab 里，${M70} 的用户从两台机器每 ${fMs(r16.tpot)}拿到一个 token，从一台机器只要 ${fMs(r8.tpot)}。</p><p>✗ “合并只传几百 KiB，不花时间。”每次合并都要先付一笔固定开销，这里机器内约 18 微秒，而一步有 160 次。</p></div>`+
     `<div class="obs"><div class="eyebrow">真实系统怎么做</div><ul class="mgs-sys"><li>vLLM：模型在一台机器里放得下，就“用张量并行”（use tensor parallelism）；跨机器时，TP 设成每台机器的卡数，流水线段数设成机器数（${cite(SRC.vllm,'parallelism and scaling')}）。</li><li>vLLM：TP 的卡数比 KV head 多时，KV head 复制到几张卡上（${cite(SRC.llama,'llama.py')}）。</li><li>${MDS}：用专家并行，并把忙的专家多复制几份；prefill 的部署单元跨 4 台机器，decode 的跨 18 台（${cite(SRC.ds,'推理系统介绍')}）。</li><li>vLLM：<code>--enable-expert-parallel</code>；跨机器时它的文档推荐 DeepEP 的 kernel，还有专家的负载均衡器（${cite(SRC.vep,'expert parallel deployment')}）。</li></ul></div>`+A.shell.toScenarios());}}
];
function sizeRes(t){if(!tried['s'+t]&&p3!==t) return ''; const r=E(P3[t]); return r.fits?fMs(r.tpot):tx('does not fit','放不下');}
function predHTML(){
  return `<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`+
   (pred?`<p class="small muted">${tx('Saved. You can still change it before the next part shows the result.','已记下，进入下一部分看结果之前还可以改。')}</p>`:'');}
function verdictHTML(){
  if(!pred) return '';
  const right=pred===PRED_RIGHT, ans=tx('TP 16 is slower than TP 8.','TP 16 反而比 TP 8 慢。');
  return `<p class="${right?'okline':'badline'}">${right?tx('Your prediction was right: ','你猜对了：')+ans:tx(`Your prediction: “${esc(L(PRED.find(o=>o.v===pred).t))}”. The answer: `,`你的预测是“${esc(L(PRED.find(o=>o.v===pred).t))}”。答案是：`)+ans}</p>`;}
function moeSeg(){return A.ui.seg({id:'mgs-ic-moe',key:'p7',label:tx('Tokens in the step','这一步的 token 数'),options:[[ '32','32'],['256','256'],['2048','2,048']],value:String(p7),inline:true});}
function moeHTML(){
  const per=MGS.route(p7), avg=p7*MGS.EP.K/MGS.EP.G, mx=Math.max(...per), top=Math.max(mx,avg)*1.08;
  const bars=per.map((v,i)=>`<div class="mgs-ep-row" data-tipx="${esc(tx(`GPU ${i+1}: ${fint(v)} pairs`,`卡 ${i+1}：${fint(v)} 对`))}"><span class="mgs-ep-l">${tx('GPU','卡')} ${i+1}</span><div class="mgs-ep-tr"><i class="${v===mx?'bad':''}" style="width:${(100*v/top).toFixed(1)}%"></i><b class="mgs-ep-avg" style="left:${(100*avg/top).toFixed(1)}%"></b></div><span class="mgs-ep-v mono">${fint(v)}</span></div>`).join('');
  return `<div class="mgs-ep" id="mgs-ep"><div class="mgs-bh">${tx(`Token–expert pairs each GPU computes (${fint(p7)} tokens × 8 experts)`,`每张卡要算的 token–专家对（${fint(p7)} 个 token × 8 个专家）`)}</div>${bars}`+
   `<div class="mgs-leg"><span><i class="mgs-sw ep"></i>${tx('token–expert pairs this GPU computes','这张卡要算的 token–专家对')}</span><span><i class="mgs-sw bad"></i>${tx('the busiest GPU','最忙的卡')}</span><span><i class="mgs-sw avgl"></i>${tx(`average: ${fint(avg)}`,`平均：${fint(avg)}`)}</span></div></div>`;}
const moeK=()=>{const per=MGS.route(p7); return Math.max(...per)/(p7*MGS.EP.K/MGS.EP.G);};

/* ---------- scenarios ---------- */
const PRESETS=[
 {id:'tp8',s:{m:'70b',g:8,tp:8,pp:1,u:32,ctx:4096},t:{en:'70B on one node, TP 8',zh:'70B，一台机器，TP 8'},d:{en:'--tensor-parallel-size 8',zh:'--tensor-parallel-size 8'}},
 {id:'fp8',s:{m:'70b8',g:8,tp:8,pp:1,u:32,ctx:4096},t:{en:'70B in FP8, TP 8',zh:'FP8 的 70B，TP 8'},d:{en:'half the weight bytes',zh:'权重的字节少一半'}},
 {id:'tp16',s:{m:'70b',g:16,tp:16,pp:1,u:32,ctx:4096},t:{en:'70B on two nodes, TP 16',zh:'70B，两台机器，TP 16'},d:{en:'one group across InfiniBand',zh:'一个组跨 InfiniBand'}},
 {id:'pp',s:{m:'70b',g:16,tp:8,pp:2,u:32,ctx:4096},t:{en:'70B on two nodes, TP 8 × 2 stages',zh:'70B，两台机器，TP 8 × 流水线 2 段'},d:{en:'vLLM’s advice across nodes',zh:'vLLM 跨机器的建议'}},
 {id:'405f',s:{m:'405b8',g:8,tp:8,pp:1,u:32,ctx:4096},t:{en:'405B in FP8 on one node, TP 8',zh:'FP8 的 405B，一台机器，TP 8'},d:{en:'fits in 8 GPUs at 1 byte per parameter',zh:'每个参数 1 字节，8 张卡放得下'}},
 {id:'405b',s:{m:'405b',g:16,tp:8,pp:2,u:32,ctx:4096},t:{en:'405B in BF16, TP 8 × 2 stages',zh:'BF16 的 405B，TP 8 × 流水线 2 段'},d:{en:'too big for one node',zh:'一台机器放不下'}}];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k]));
function scenHTML(i){const p=PRESETS[i], r=E(p.s), t=fMs(r.tpot), sh=fP(r.share);
  const T={
   tp8:tx(`<p><b>${t} per output token; the merges take ${sh} of the step.</b></p><p>The usual way to serve 70B on one node: every GPU keeps an eighth of each matrix and an eighth of every user’s KV cache.</p>`,`<p><b>每个输出 token ${t}，合并占一步的 ${sh}。</b></p><p>一台机器跑 70B 的常见做法：每张卡存每个矩阵的八分之一，以及每个用户 KV cache 的八分之一。</p>`),
   fp8:(()=>{const b=E(PRESETS[0].s); return tx(`<p><b>${t} per output token, against ${fMs(b.tpot)} in BF16.</b></p><p>FP8 halves the weight bytes each GPU reads, but the KV cache stays in BF16, so reading and computing fall from ${fMs(b.tcm)} to ${fMs(r.tcm)}.</p><p>The merges still carry BF16 values and cost the same. The faster the reading, the larger the merges’ share of the step: ${sh} here, against ${fP(b.share)}.</p>`,`<p><b>每个输出 token ${t}，BF16 时是 ${fMs(b.tpot)}。</b></p><p>FP8 让每张卡要读的权重少了一半，KV cache 还是 BF16，所以读显存和计算从 ${fMs(b.tcm)}降到 ${fMs(r.tcm)}。</p><p>合并传的还是 BF16 的数值，开销不变。读得越快，合并占的比例越大：这里是 ${sh}，BF16 时是 ${fP(b.share)}。</p>`);})(),
   tp16:tx(`<p><b>${t} per output token; the merges take ${sh} of the step.</b></p><p>Each GPU reads less, but each of its 160 merges per step crosses InfiniBand, with a fixed cost more than three times the one inside a node.</p>`,`<p><b>每个输出 token ${t}，合并占一步的 ${sh}。</b></p><p>每张卡读得少了，可每步 160 次合并都要跨 InfiniBand，每次的固定开销是机器内的三倍多。</p>`),
   pp:tx(`<p><b>${t} per output token; the merges and the transfers take ${sh} of the step.</b></p><p>Only two small messages per step cross the nodes. Each node’s TP 8 merges stay on NVLink.</p>`,`<p><b>每个输出 token ${t}，合并和段间传输占一步的 ${sh}。</b></p><p>每步只有两次小消息跨机器，两台机器各自的 TP 8 合并都走 NVLink。</p>`),
   '405f':tx(`<p><b>${t} per output token; each GPU keeps ${fGi(r.w)} of weights.</b></p><p>At one byte per parameter, this model fits in one node. The official FP8 release keeps 81.3B of its parameters in BF16, which needs 56.7 GiB per GPU; this lab counts every parameter at one byte.</p><p>Even with the official release, the KV cache of these users (${fGi(r.need)} per GPU) still fits.</p>`,`<p><b>每个输出 token ${t}，每张卡存 ${fGi(r.w)} 的权重。</b></p><p>每个参数一个字节时，这个模型放得进一台机器。官方的 FP8 版本有 81.3B 个参数仍是 BF16，每张卡要 56.7 GiB；这个 Lab 每个参数都按一个字节算。</p><p>按官方版本算，这些用户的 KV cache（每张卡 ${fGi(r.need)}）也还放得下。</p>`),
   '405b':(()=>{const o=E({...p.s,tp:16,pp:1}), q=E({...p.s,u:128}); return tx(`<p><b>${t} per output token: the two stages together read as many weights as one group of eight would.</b></p><p>In this lab one group of sixteen across the two nodes is faster (${fMs(o.tpot)}): each GPU of this model reads so many bytes per step that halving them saves more than the slow merges across nodes cost.</p><p>The group of sixteen pays in memory: the model has only eight KV heads, so each GPU still keeps an eighth of every user’s KV cache. In the pipeline each stage holds only half the layers, so each GPU keeps a sixteenth. In ${J_('controls','Settings')} set the users to 128: the group of sixteen no longer fits, and the pipeline still does (${fMs(q.tpot)}).</p>`,`<p><b>每个输出 token ${t}：两段读的权重加起来，和一个八张卡的组一样多。</b></p><p>在这个 Lab 里，跨两台机器的十六张卡的组更快（${fMs(o.tpot)}）：这个模型每张卡每一步要读的字节很多，减半省下的时间比跨机器的慢合并多花的还多。</p><p>十六张卡的组的代价是显存：模型只有八个 KV head，每张卡仍要存每个用户 KV cache 的八分之一。流水线每段只存一半的层，每张卡只存十六分之一。在“${J_('controls','自由调参')}”里把用户数改成 128：十六张卡的组放不下了，流水线还放得下（${fMs(q.tpot)}）。</p>`);})()};
  return T[p.id];}
function sandboxHTML(){const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how long each output token takes and where the time goes.','选一个场景，看看每个输出 token 要多久、时间花在哪里。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="mgs-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change the model, the GPUs, the split, the users or their context under ${J_('controls','Settings')}.`,`想改模型、卡数、分法、用户数或上下文，在“${J_('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;}

/* ---------- challenge ---------- */
/* B1: the reader also picks one node or two; with 128 users one node of TP 8 is the best answer, which no guided part prints */
const CH={m:'70b',u:128,ctx:4096,goal:0.025}, CH_START={g:16,tp:16,pp:1};
const CH_OPTS=()=>{const o=[]; for(const g of [8,16]) for(const tp of TPS) for(const pp of [1,2,4,8,16]) if(tp<=g&&tp*pp<=g) o.push({g,tp,pp}); return o;};
const chCfg=o=>({m:CH.m,g:o.g,u:CH.u,ctx:CH.ctx,tp:o.tp,pp:o.pp});
const chRes=o=>E(chCfg(o));
const chOk=o=>{const r=chRes(o); return r.fits&&r.tpot<=CH.goal+1e-12;};
const chKey=o=>o.g+'-'+o.tp+'x'+o.pp;
const chPick=()=>({g:S.g,tp:S.tp,pp:S.pp});
const chName=o=>tx(`${o.g} GPUs: ${setupName(chCfg(o))}`,`${o.g} 张卡：${setupName(chCfg(o))}`);
const chBest=()=>CH_OPTS().filter(chOk).reduce((a,o)=>!a||chRes(o).tps>chRes(a).tps?o:a,null);
function chFix(){if(S.tp>S.g) S.tp=S.g; while(S.pp>1&&S.tp*S.pp>S.g) S.pp/=2;}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button>`;
  const tpOpts=TPS.map(t=>[String(t),String(t),t>S.g]), ppOpts=[1,2,4,8,16].map(p=>[String(p),String(p),S.tp*p>S.g]);
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Give 128 users a token every 25 ms or faster, from one node or two','让一台或两台机器给 128 个用户每 25 毫秒内出一个 token')}</h3>`+
   tx(`<p>One node of eight H100s serves ${M70} in BF16 to 128 users at once, each with 4,096 tokens of context; a second node can be added.</p><p>Every user must get an output token at least every 25 ms. Choose how many GPUs to use, the TP size and the pipeline stages; the GPUs left over make replicas, which split the users evenly.</p><p>Among the setups that pass, the more tokens per second each GPU produces, the better.</p>`,
      `<p>一台机器有八张 H100，用 BF16 跑 ${M70}，同时服务 128 个用户，每人的上下文有 4,096 个 token；必要时可以再加一台。</p><p>每个用户至少每 25 毫秒要拿到一个输出 token。选用几张卡、TP 的卡数和流水线的段数；剩下的卡组成副本，用户在副本之间平分。</p><p>达标的设置里，每张卡每秒出的 token 越多越好。</p>`)+
   A.ui.seg({id:'mgs-ic-chg',key:'chg',label:tx('GPUs','卡数'),options:[['8',tx('8 (one node)','8（一台机器）')],['16',tx('16 (two nodes)','16（两台机器）')]],value:String(S.g),inline:true})+
   A.ui.seg({id:'mgs-ic-chtp',key:'chtp',label:tx('TP size (GPUs per group)','TP 的卡数（每组几张）'),options:tpOpts,value:String(S.tp),inline:true})+
   A.ui.seg({id:'mgs-ic-chpp',key:'chpp',label:tx('Pipeline stages','流水线的段数'),options:ppOpts,value:String(S.pp),inline:true})+
   `<p class="small muted" id="mgs-ch-rep"></p>`+
   `<div class="goal mono mgs-goal" id="mgs-ch-g"><span>${tx('Time per output token','每个输出 token 的耗时')}</span><span><b id="mgs-ch-v"></b>${tx(' (goal: 25 ms or less)','（目标：不超过 25 毫秒）')}</span></div><div class="progress" id="mgs-ch-pr" aria-hidden="true"><i></i></div>`+
   submit+`<div id="mgs-ch-res" class="mgs-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chSubmitted) return '';
  const pick=chPick(), r=chRes(pick), ok=chOk(pick), opts=CH_OPTS(), best=chBest();
  let head;
  if(!ok){
    const hint=!r.fitW?tx(`The weights alone need ${fGi(r.w)} per GPU, more than the 70 GiB there is.`,`光是权重每张卡就要 ${fGi(r.w)}，超过了能用的 70 GiB。`)
      :!r.fits?tx(`After its weights each GPU has ${fGi(r.room)} left, but its users’ KV cache needs ${fGi(r.need)}.`,`每张卡存下权重后只剩 ${fGi(r.room)}，可它的用户的 KV cache 要 ${fGi(r.need)}。`)
      :r.ar.cross?tx(`With only eight KV heads, each GPU still reads an eighth of every user’s KV cache, and every merge crosses the nodes: ${fP(r.share)} of the step.`,`模型只有八个 KV head，每张卡仍要读每个用户 KV cache 的八分之一；每次合并还都要跨机器，占了一步的 ${fP(r.share)}。`)
      :tx(`Each GPU reads too many bytes per step: with small groups, every replica or stage reads a large share of the weights for its users.`,`每张卡每一步要读的字节太多：组小时，每个副本或每一段都要为自己的用户读很大一份权重。`);
    head=`<p class="badline"><b>${r.fits?tx(`Not met: ${fMs(r.tpot)} per output token.`,`没达标：每个输出 token ${fMs(r.tpot)}。`):tx('Not met: it does not fit.','没达标：放不下。')}</b>${zh()?'':' '}${hint}</p>`+(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);}
  else if(chKey(best)===chKey(pick)) head=`<p class="okline"><b>${tx(`Met: ${fMs(r.tpot)} per output token, and the most tokens per second per GPU (${fTok(r.tps)}).`,`达标：每个输出 token ${fMs(r.tpot)}，而且每张卡每秒出的 token 最多（${fTok(r.tps)} 个）。`)}</b></p>`;
  else {const b=chRes(best); head=`<p class="okline"><b>${tx(`Met: ${fMs(r.tpot)} per output token.`,`达标：每个输出 token ${fMs(r.tpot)}。`)}</b></p><p>${tx(`But another setup gets more tokens per second from each GPU (the chart): ${fTok(b.tps)} against your ${fTok(r.tps)}.`,`不过另一种分法每张卡每秒出的 token 更多（见图）：${fTok(b.tps)} 个，你的是 ${fTok(r.tps)} 个。`)}</p>`;}
  if(!ok&&!chReveal) return head;
  const mx=Math.max(...opts.map(o=>{const x=chRes(o); return x.fits?x.tpot:0;}))*1e3;
  const rows=opts.map(o=>{const x=chRes(o), me=chKey(o)===chKey(pick), st=chKey(o)===chKey(CH_START), pass=chOk(o);
    return {label:chName(o),value:x.fits?x.tpot*1e3:0,text:x.fits?fMs(x.tpot):tx('does not fit','放不下'),ok:pass,me,best:best&&chKey(o)===chKey(best),
      tags:[...(me?[{text:tx('your pick','你的选择'),me:true}]:[]),...(st?[{text:tx('initial setup','初始设置')}]:[]),...(pass?[{text:tx(`${fTok(x.tps)} tokens/s per GPU`,`每张卡每秒 ${fTok(x.tps)} 个 token`)}]:[])]};});
  const b=chRes(best), r2=chRes({g:16,tp:8,pp:1});
  const why=tx(`<p><b>One node is enough: one group of eight meets the goal (${fMs(b.tpot)}) and gets the most from each GPU.</b> Smaller groups on one node miss it: each replica or stage reads a large share of the weights for its users.</p><p>A second node gives each token sooner (the chart), but the same users are spread over twice the GPUs, so each GPU produces fewer tokens per second: ${fTok(r2.tps)} for two replicas of eight, against ${fTok(b.tps)}.</p><p>One group of sixteen misses the goal: each GPU still reads an eighth of every user’s KV cache, and every merge crosses the nodes. Caveat: here nothing overlaps and every user has the same context; engines overlap some communication with compute, so measure your own before you choose.</p>`,
                `<p><b>一台机器就够了：一个八张卡的组就能达标（${fMs(b.tpot)}），每张卡出的 token 也最多。</b>一台机器里分成更小的组都不达标：每个副本或每一段都要为自己的用户读很大一份权重。</p><p>加一台机器，每个 token 会更快（见图），可同样多的用户分到了两倍的卡上，每张卡每秒出的 token 就少了：两个八张卡的副本是 ${fTok(r2.tps)} 个，一台机器是 ${fTok(b.tps)} 个。</p><p>一个十六张卡的组不达标：每张卡仍要读每个用户 KV cache 的八分之一，每次合并还都要跨机器。注意：这里没有任何事重叠，每个用户的上下文也一样长；真实的引擎会让一部分通信和计算重叠，选之前先在自己的引擎上量一量。</p>`);
  return head+A.ui.cmpChart({title:tx(`${M70} for 128 users: which setups give every output token within 25 ms?`,`${M70} 服务 128 个用户：哪些分法能让每个输出 token 不超过 25 毫秒？`),aria:tx('Time per output token for each setup','每种设置每个输出 token 的耗时'),rows,goal:25,goalLabel:tx('goal 25 ms','目标 25 毫秒'),max:Math.ceil(mx/10)*10,brief:true,okText:tx('meets the 25 ms goal','达到 25 毫秒的目标'),noText:tx('misses it or does not fit','没达到或放不下'),
    hiddenText:n=>tx(n===1?'The other setup misses the goal or does not fit and is not shown.':`The other ${n} setups miss the goal or do not fit and are not shown.`,`其余 ${n} 种没达到目标或放不下，图里没有列出。`)})+why;}

/* ---------- capability layer ---------- */
const EXITS=()=>{const r8=R8(), t16=E({m:'405b',g:16,tp:16,pp:1,u:64,ctx:4096}), tpp=E({m:'405b',g:16,tp:8,pp:2,u:64,ctx:4096});
 /* T1: a 32-layer model, hidden size 4,096, 16 users, TP 2 in a node */
 const t32=64*MGS.arTime(16*4096*2,2).t;
 return {
 explain:[{q:tx('Two pipeline stages on two nodes have twice the GPUs of one group of eight on one node. Why do they not halve the time per output token?','两台机器上的流水线 2 段，卡数是一台机器上八张卡的组的两倍。为什么每个输出 token 的耗时没有减半？'),
   o:[tx('A token passes the stages one after the other, and together they read as many weights as the group of eight; only the KV cache per GPU halves.','一个 token 要依次走过两段，两段读的权重加起来和八张卡的组一样多；只有每张卡的 KV cache 减半了。'),
      tx('Sending the hidden vectors between the stages is so slow that it cancels the saving.','段与段之间传隐藏向量太慢，抵掉了省下的时间。'),
      tx('Every stage keeps a whole copy of the weights, so twice the bytes are read.','每一段都要存一整份权重，读的字节翻倍了。')],a:0,
   why:tx('The pipeline buys memory, not speed for one token: the users split into batches that the stages work on at once.','流水线换来的是显存，不是单个 token 的速度：用户分成几批，各段同时各处理一批。'),
   n:[null,tx('Between stages only one hidden vector per user moves; it takes a tiny part of the step.','段与段之间每个用户只传一个隐藏向量，只占一步里很小的一部分。'),tx('Each stage keeps only its own layers; no stage holds the whole model.','每一段只存自己那几十层，没有哪一段存着整个模型。')]},
  {q:tx('In this lab, one group of eight GPUs serving all the users gives each token sooner than two replicas of four GPUs, and each GPU produces more tokens per second. Why?','在这个 Lab 里，一个八张卡的组服务全部用户，比两个四张卡的副本每个 token 更快，每张卡每秒出的 token 也更多。为什么？'),
   o:[tx('Decode time goes mostly into reading the weights: each replica reads all of them every step, while the group of eight reads them once for all the users.','decode 主要花在读权重上：每个副本每一步都要把全部权重读一遍，八张卡的组只为全部用户读一遍。'),
      tx('The smaller groups’ merges are slower.','小组的合并更慢。'),
      tx('Replicas must copy their KV caches to each other every step.','副本之间每一步都要互相同步 KV cache。')],a:0,
   why:tx('Fewer users per replica do not shrink the weights to read. A larger group spreads one read over more GPUs.','每个副本的用户少了，要读的权重并不会变少。组大，就是把同一次读分给更多张卡。'),
   n:[null,tx('They are a little cheaper: in a smaller ring each GPU sends a smaller multiple of the message, and each replica merges the hidden vectors of only half the users.','小组的合并还便宜一点：环越小，每张卡要发的字节相对消息越少，而且每个副本只合并一半用户的隐藏向量。'),tx('Replicas never talk to each other during a step.','副本之间在一步里不通信。')]}],
 transfer:[{q:tx(`A model with 32 layers runs TP 2 inside a node. About how long do its merges take per step?<br>Hidden size 4,096; 16 users.<br>For comparison, ${M70} with TP 8:<br>160 merges per step, about ${fMs(r8.tComm)}.`,`一个 32 层的模型在一台机器里用 TP 2。它每步的合并大约要多久？<br>隐藏维度 4,096；16 个用户。<br>对照：TP 8 的 ${M70}，<br>每步合并 160 次，约 ${fMs(r8.tComm)}。`),
   o:[tx(`About ${fMs(t32)}`,`约 ${fMs(t32)}`),tx('About 0.1 ms: with two GPUs there is almost nothing to send','约 0.1 毫秒：两张卡几乎不用传什么'),tx(`About ${fMs(r8.tComm)}: the same as ${M70}`,`约 ${fMs(r8.tComm)}：和 ${M70} 一样`)],a:0,
   why:tx('It merges twice per layer too, so fewer layers mean fewer merges; each still costs about the same fixed time, however small the message.','它也是每层合并两次，层少，合并就少；每次合并的固定开销差不多，消息再小也省不掉。')},
  {q:tx(`A team serves ${M405} in BF16 to 64 users on two nodes; it does not fit in one. Which gives each token sooner: one group of sixteen GPUs, or a group of eight in each node as two pipeline stages?<br>Two stages, read and compute: ${fMs(tpp.c.pp*tpp.tcm)}<br>TP 16, read and compute: ${fMs(t16.tcm)}<br>TP 16 merges: ${t16.nAR} per step, ${fUs(t16.ar.t)} each`,`一个团队用两台机器跑 BF16 的 ${M405}，服务 64 个用户；一台机器放不下。十六张卡组成一个组，和每台机器八张卡、分成流水线两段，哪个每个输出 token 更快？<br>两段加起来，读和算要 ${fMs(tpp.c.pp*tpp.tcm)}<br>TP 16 读和算要 ${fMs(t16.tcm)}<br>TP 16 每步合并 ${t16.nAR} 次，每次 ${fUs(t16.ar.t)}`),
   o:[tx('One group of sixteen: halving a long read saves more than its merges cost','十六张卡的组：把很长的读取时间减半，省下的比合并多花的多'),tx('The pipeline: TP never pays off across nodes','流水线：跨机器的 TP 永远不划算'),tx('The same: both use sixteen GPUs','一样：都用十六张卡')],a:0,
   why:tx(`Here the group of sixteen takes ${fMs(t16.tpot)} against ${fMs(tpp.tpot)} for the pipeline: with a model this large the read is long, so halving it pays even with slow merges.`,`这里十六张卡的组要 ${fMs(t16.tpot)}，流水线要 ${fMs(tpp.tpot)}：模型这么大，读的时间很长，即使合并慢，减半也划算。`)}],
 practice:[{id:'matmul-roofline',d:tx('Roofline and sharding estimates for a feed-forward layer split over two accelerators.','把前馈层切到两块加速器上，用 roofline 估算时间。')},{id:'inference-serving',d:tx('Design an LLM serving system, from GPUs to batching.','设计大模型推理服务，从 GPU 到批处理。')}]};};

/* ---------- copy: header ---------- */
const DESCS=()=>({guided:tx('8 parts, from a model too big for one GPU to tensor, pipeline and expert parallelism.','8 个部分，从一张卡放不下的模型，讲到张量、流水线和专家并行。'),
  sandbox:tx('Common setups, and every setting.','几种常见的分法，所有设置都能改。'),challenge:tx(`One or two nodes of H100s run ${M70} in BF16 for 128 users, each needing a token within 25 ms.`,`一台或两台 H100 机器用 BF16 跑 ${M70}，服务 128 个用户，每个用户 25 毫秒内要拿到下一个 token。`)});
const OUTCOME=()=>tx('choose the tensor-parallel size, pipeline stages and replicas for a model, and price the communication each choice adds.','为一个模型选好张量并行的卡数、流水线的段数和副本数，并算出每种选择多出来的通信时间。');
/* how much the conclusions lean on the two fixed costs (S10, O4): all from the engine's own pieces */
const SENS=()=>{const r8=R8(), r16=R16(), r4=E(P3[4]), bw16=r16.ar.bwIn+r16.ar.bwX, sym=32768*1.75/7.34e9;
  return {flip:(r8.tpot-r16.tcm)/r16.nAR-bw16, at30:r16.tcm+r16.nAR*(30e-6+bw16), sym, sym8:r8.tcm+r8.nAR*(sym+r8.ar.bwIn), sym4:r4.tcm+r4.nAR*(sym+r4.ar.bwIn)};};
const FIDPOP=()=>{const v=SENS(), us1=x=>tx(`${f1(x*1e6)} µs`,`${f1(x*1e6)} 微秒`);
 return tx(`<p><b>Formula.</b> Each decode step is priced by the step model of the other inference labs: the slower of its arithmetic at 50% of the dense peak (989.5 TFLOP/s in BF16, 1,979 with FP8 weights and activations) and its memory traffic at 60% of 3.35 TB/s, both this lab’s assumptions. Each GPU of a tensor-parallel group does 1/n of the arithmetic and reads 1/n of the weights; the KV cache splits by KV head, so beyond 8 GPUs it does not shrink. The weights read per step leave out the embedding table, of which only a few rows are looked up.</p>`+
  `<p>Each layer adds 2 all-reduces that add up the users’ hidden vectors (2 bytes per value). One all-reduce takes a fixed cost plus the time to send its bytes:</p><div class="fx">${M('t_{\\text{AR}} = t_0 + \\frac{2(n-1)}{n} \\cdot \\frac{S}{B}')}</div><p>S is the bytes merged, n the GPUs in the group, B each GPU’s bandwidth (${cite(SRC.nccl,'nccl-tests')}). Inside a node, t₀ is 18 µs and B is NVLink’s 450 GB/s. Across nodes, t₀ is taken as 60 µs, this lab’s assumption. The data is first reduced inside each node, each GPU sends 1/8 of it across on its own 50 GB/s InfiniBand NIC, and the result is gathered inside the node (${cite(SRC.net,'ml-engineering')}).</p>`+
  `<p><b>Where the two fixed costs come from.</b> ml-engineering measured a 32 KiB all-reduce on 8 H200s (NVLink 4, as on the H100) at 3.15 GB/s of bus bandwidth; undoing the bus-bandwidth factor gives the time:</p><div class="fx">${M('\\frac{32,768\\ \\text{B} \\times 1.75}{3.15\\ \\text{GB/s}} \\approx 18.2\\ \\mu\\text{s}')}</div><p>That is the regular NCCL column; with NCCL’s symmetric memory the same table shows 7.34 GB/s, about ${us1(v.sym)}. Then TP 8 would take ${fMs(v.sym8)} per token and two TP 4 replicas ${fMs(v.sym4)}: parts 3 and 5 reach the same conclusions. For small all-reduces across nodes the same notes measured B200 machines, 3 to 5 times slower on 4 nodes than in one; there is no H100 figure for two nodes, so 60 µs is an assumption. Under this lab’s arithmetic, TP 16 is slower than TP 8 for 32 users of 70B whenever the fixed cost across nodes is above about ${us1(v.flip)}; at 30 µs, TP 16 would take about ${fMs(v.at30)} and be faster. A transfer between pipeline stages pays the same fixed cost (18 or 60 µs) plus its bytes.</p>`+
  `<p>Pipeline stages split the users into micro-batches; each stage reads its weights once per micro-batch, and each step sends the hidden vectors to the next stage and the new tokens back.</p><p>Left out: overlap of communication with compute (engines overlap some), kernel launch costs, prefill mixed into decode steps (${LL('pd-disagg')}), uneven context lengths (every user reads 4,096 tokens unless changed), the embedding table and LM head placed on one stage (here split evenly). FP8 counts 1 byte per parameter; the official 405B FP8 release keeps 81.3B parameters in BF16 (56.7 GiB per GPU on 8 GPUs instead of 47.2), which still fits. The MoE part routes tokens uniformly at random, a lab assumption.</p>`,
  `<p><b>公式计算。</b>每一步 decode 按其他推理 Lab 的步模型算：算力按稠密峰值的 50% 算（BF16 是 989.5 TFLOP/s；权重和激活都用 FP8 时是 1,979），带宽按 3.35 TB/s 的 60% 算，取两者中较慢的那个，这两个百分比都是这个 Lab 的假设。张量并行组里的每张卡做 1/n 的计算、读 1/n 的权重；KV cache 按 KV head 切，超过 8 张卡就不再变小。每步要读的权重不含 embedding 表：它每步只查用到的几行。</p>`+
  `<p>每层另加 2 次 all-reduce，把所有用户的隐藏向量加起来（每个数值 2 字节）。一次 all-reduce 的时间是固定开销加上传字节的时间：</p><div class="fx">${M('t_{\\text{AR}} = t_0 + \\frac{2(n-1)}{n} \\cdot \\frac{S}{B}')}</div><p>其中 S 是要合并的字节数，n 是组里的卡数，B 是每张卡的带宽（${cite(SRC.nccl,'nccl-tests')}）。机器内，t₀ 是 18 微秒，B 是 NVLink 的 450 GB/s。跨机器时，t₀ 按 60 微秒算，这是这个 Lab 的假设。数据先在机器内归约，每张卡再用自己 50 GB/s 的 InfiniBand 网卡跨机器传 1/8，最后在机器内汇集（${cite(SRC.net,'ml-engineering')}）。</p>`+
  `<p><b>两个固定开销怎么来的。</b>ml-engineering 在 8 张 H200（NVLink 4，和 H100 一样）上量到，32 KiB 的 all-reduce 总线带宽是 3.15 GB/s；撤掉总线带宽的修正系数，换算成时间：</p><div class="fx">${M('\\frac{32,768\\ \\text{B} \\times 1.75}{3.15\\ \\text{GB/s}} \\approx 18.2\\ \\text{微秒}')}</div><p>这是 NCCL 普通模式那一列；打开 NCCL 的 symmetric memory，同一张表是 7.34 GB/s，约 ${us1(v.sym)}。按它算，TP 8 每个 token ${fMs(v.sym8)}，两个 TP 4 副本 ${fMs(v.sym4)}，第 3、5 部分的结论都不变。跨机器的小 all-reduce，同一份资料量的是 B200 机器：4 台机器比一台机器内慢 3 到 5 倍；没有 H100 两台机器的数，所以 60 微秒是假设。按这个 Lab 的算法，只要跨机器的固定开销超过约 ${us1(v.flip)}，32 个用户时 70B 的 TP 16 就比 TP 8 慢；如果只有 30 微秒，TP 16 约 ${fMs(v.at30)}，反而更快。流水线段间的传输也付同样的固定开销（18 或 60 微秒），再加上传字节的时间。</p>`+
  `<p>流水线把用户分成几批，每一段每处理一批就读一遍自己的权重；每一步把隐藏向量传给下一段，再把新 token 传回第一段。</p><p>没有算进去的：通信和计算的重叠（真实的引擎会重叠一部分）、kernel 启动的开销、混进 decode 步的 prefill（见 ${LL('pd-disagg')}）、上下文长短不一（不改设置时每个用户都读 4,096 个 token）、embedding 表和 LM head 放在某一段上（这里平均分）。FP8 每个参数按 1 字节算；官方的 405B FP8 版本有 81.3B 个参数仍是 BF16（8 张卡时每张 56.7 GiB，不是 47.2），也还放得下。MoE 部分让 token 均匀随机地选专家，这是这个 Lab 的假设。</p>`);};
const PRIPOP=()=>tx(`<p>A decode step reads all the weights and the users’ KV cache once and writes one token per user, so memory bandwidth is the limit (${labLink('prefill-decode')}). Tensor and pipeline parallelism for training are taught in the “${labLink('model-parallel')}” lab; this lab uses the same cuts for serving.</p>`,
 `<p>decode 每一步把全部权重和用户的 KV cache 读一遍，给每个用户写一个 token，所以带宽是瓶颈（见 ${labLink('prefill-decode')}）。训练时的张量并行和流水线并行，“${labLink('model-parallel')}”这个 Lab 讲过；这个 Lab 在推理时用同样的切法。</p>`);

/* ---------- views ---------- */
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Multi-GPU serving','多卡推理'),title:l?L(l.q):tx('How is one model served across 8 GPUs?','一个模型怎么在 8 张卡上服务？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: decode and TP','前置知识：decode 和张量并行'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const spdCtl=()=>`<div class="mgs-speed"><span class="mgs-speed-l">${tx('Speed','速度')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="spd">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${spdNow()===v}">${x}</button>`).join('')}</div></div>`;
function stageHTML(){
  const lg=(tip,cls,t)=>`<span data-tipx="${esc(tip)}"><i class="mgs-sw ${cls}"></i>${t}</span>`;
  return `<div class="panel stage mgs-stage" id="mgs-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="mgs-st-title"></div><div class="meter-sub" id="mgs-st-sub"></div></div><div class="stage-actions"><span class="status" id="mgs-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph mgs-graph" id="mgs-graph">`+
    `<div class="mgs-blk" id="mgs-gb"><div class="mgs-bh" id="mgs-g-h"></div><div class="mgs-gpus" id="mgs-gpus"></div>`+
    `<div class="legend-row mgs-legend">${lg(tx('The model’s weights this GPU keeps.','这张卡存的那部分权重。'),'w',tx('weights','权重'))}${lg(tx('The KV cache this GPU keeps for its users.','这张卡为它的用户存的 KV cache。'),'kv',tx('KV cache','KV cache'))}${lg(tx('About 2 GiB the runtime keeps on every GPU.','运行时在每张卡上占的约 2 GiB。'),'oh',tx('runtime overhead','运行时开销'))}${lg(tx('A GPU whose weights and KV cache do not fit in the 70 GiB it can use.','权重和 KV cache 超过了能用的 70 GiB 的卡。'),'bad',tx('does not fit','放不下'))}</div></div>`+
    `<div class="mgs-blk" id="mgs-lb"><div class="mgs-bh" id="mgs-l-h"></div><div class="mgs-layer" id="mgs-layer"></div>`+
    `<div class="legend-row mgs-legend">${lg(tx('Each GPU reads its weights and KV cache and computes its slice.','每张卡读自己的权重和 KV cache，算自己那一块。'),'cm',tx('read and compute','读显存和计算'))}${lg(TIP.ar(),'ar',tx('merge (all-reduce)','合并（all-reduce）'))}${lg(tx('Merges that take half the step or more.','占了一步一半以上时间的合并。'),'bad',tx('merges dominate','合并占大头'))}</div>`+
    `<div class="mgs-run" id="mgs-run"><button type="button" class="btn sm" data-act="play" id="mgs-play"></button><button type="button" class="btn ghost sm" data-act="replay" id="mgs-replay">${tx('Replay','重放')}</button>${spdCtl()}<input type="range" class="mgs-prog" id="mgs-prog" min="0" max="1000" value="1000" aria-label="${esc(tx('Position in the layer','播放位置'))}"></div>`+
    `<p class="mgs-sum" id="mgs-sum"></p></div>`+
    `<div class="mgs-blk" id="mgs-swb" hidden><div class="mgs-bh" id="mgs-sw-h"></div><div id="mgs-sweep"></div><p class="small muted mgs-sw-note" id="mgs-sw-n"></p></div>`+
    `<div class="mgs-blk" id="mgs-epb" hidden><div id="mgs-ep-host"></div></div>`+
   `</div>`+
   `<div class="metrics mgs-metrics" id="mgs-metrics"></div>`+
   `<details class="math" id="mgs-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="mgs-mathbody"></div></details></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="mgs-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="mgs-lock" hidden></p><div id="mgs-ctls">`+
   A.ui.seg({id:'mgs-ctl-m',key:'m',label:tx('Model','模型'),options:MKEYS.map(k=>[k,MSHORT[k]]),value:S.m,note:tx(`${M70} or ${M405}. FP8 here means 1 byte per parameter and FP8 matrix multiplies; the KV cache stays in BF16.`,`${M70} 或 ${M405}。这里的 FP8 指每个参数 1 字节、矩阵乘法也用 FP8；KV cache 仍是 BF16。`)})+
   A.ui.seg({id:'mgs-ctl-g',key:'g',label:tx('GPUs','卡数'),options:[['8',tx('8 (one node)','8（一台机器）')],['16',tx('16 (two nodes)','16（两台机器）')]],value:String(S.g)})+
   A.ui.seg({id:'mgs-ctl-tp',key:'tp',label:tx('TP size (GPUs per group)','TP 的卡数（每组几张）'),options:TPS.map(t=>[String(t),String(t)]),value:String(S.tp),note:tx('Above 8, a group spans two nodes.','超过 8 张，一个组就跨两台机器。')})+
   A.ui.seg({id:'mgs-ctl-pp',key:'pp',label:tx('Pipeline stages','流水线的段数'),options:PPS.map(t=>[String(t),String(t)]),value:String(S.pp),note:tx('The GPUs left over make replicas, and the users are split evenly between them.','剩下的卡组成副本，用户在副本之间平分。')})+
   A.ui.seg({id:'mgs-ctl-u',key:'u',label:tx('Users served at once','同时服务的用户数'),options:USERS.map(t=>[String(t),String(t)]),value:String(S.u)})+
   A.ui.seg({id:'mgs-ctl-c',key:'ctx',label:tx('Context per user (tokens)','每个用户的上下文（token）'),options:CTXS.map(t=>[String(t),fint(t)]),value:String(S.ctx),note:tx('Every user’s next token reads this many tokens of KV cache; real users vary, this lab gives all the same.','每个用户的下一个 token 要读这么多 token 的 KV cache；真实的用户长短不一，这个 Lab 让所有人都一样。')})+
   `<p class="ctl-note">${tx('Assumed: each merge’s fixed cost is 18 µs inside a node and 60 µs across nodes; NVLink 450 GB/s and InfiniBand 50 GB/s per GPU, one way. See the notes at the end of the page for why.','假设：每次合并的固定开销机器内 18 微秒、跨机器 60 微秒；每张卡 NVLink 单向 450 GB/s，InfiniBand 单向 50 GB/s。为什么这样取，见页末的“简化、假设和来源”。')}</p>`+
   `<button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`# One decode step of Llama-3.1-70B (BF16) split over GPUs
L, d = 80, 8192                 # layers, hidden size
# weight bytes read per step (the embedding table is not read)
W_READ = 139.006e9
KV_TOK = 2 * L * 8 * 128 * 2    # bytes of KV cache per token
BS = 3.35e12 * 0.6              # bytes/s each GPU reads (60%)
NVL, IB = 450e9, 50e9           # bytes/s per GPU, one way

def all_reduce(nbytes, tp):
    if tp <= 8:                 # one node: fixed cost + ring
        return 18e-6 + 2 * (tp - 1) / tp * nbytes / NVL
    k = tp // 8                 # nodes: reduce in node,
    return (60e-6 + 2 * 7 / 8 * nbytes / NVL    # 1/8 across
            + 2 * (k - 1) / k * nbytes / 8 / IB)

def tpot(tp, users=32, ctx=4096):
    kv_split = min(tp, 8)       # only 8 KV heads
    read = W_READ / tp + users * ctx * KV_TOK / kv_split
    merges = 2 * L * all_reduce(users * d * 2, tp)
    return read / BS + merges   # decode: memory-bound

for tp in (8, 16):
    print(f"TP {tp}: {tpot(tp) * 1e3:.1f} ms per token")`;
const CODE_OUT=`# TP 8: 14.5 ms per token
# TP 16: 17.1 ms per token`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="mgs-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="mgs-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The time per output token of a TP group: what each GPU reads, plus 160 merges.','一个 TP 组每个输出 token 的耗时：每张卡要读的字节，加上 160 次合并。')}</p><pre class="code" id="mgs-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="mgs-prod">`+
   tx(`<li>vLLM: ${cite(SRC.vllm,'<code>docs/serving/parallelism_scaling.md</code>')}. <code>--tensor-parallel-size</code> inside a node; across nodes, the TP size is the GPUs per node and <code>--pipeline-parallel-size</code> the nodes. Its TP is Megatron-LM’s algorithm.</li><li>vLLM: ${cite(SRC.llama,'<code>vllm/model_executor/models/llama.py</code>')}. The attention layer splits the heads over the TP group and replicates the KV heads when there are fewer of them than GPUs.</li><li>vLLM: ${cite(SRC.vep,'<code>docs/serving/expert_parallel_deployment.md</code>')}. <code>--enable-expert-parallel</code>; several all-to-all backends, and for multi-node prefill and decode its documentation recommends DeepEP’s; EPLB for skewed experts.</li><li>DeepSeek: ${cite(SRC.deepep,'DeepEP')} (all-to-all kernels for MoE dispatch and combine) and ${cite(SRC.eplb,'EPLB')} (redundant experts).</li>`,
      `<li>vLLM：${cite(SRC.vllm,'<code>docs/serving/parallelism_scaling.md</code>')}。机器内用 <code>--tensor-parallel-size</code>；跨机器时 TP 设成每台机器的卡数，<code>--pipeline-parallel-size</code> 设成机器数。它的 TP 用的是 Megatron-LM 的算法。</li><li>vLLM：${cite(SRC.llama,'<code>vllm/model_executor/models/llama.py</code>')}。attention 层把 head 分给 TP 组里的卡；KV head 比卡少时，把 KV head 复制几份。</li><li>vLLM：${cite(SRC.vep,'<code>docs/serving/expert_parallel_deployment.md</code>')}。<code>--enable-expert-parallel</code>；有几种 all-to-all 后端，跨机器的 prefill 和 decode，它的文档推荐 DeepEP 的；专家负载不均时用 EPLB。</li><li>DeepSeek：${cite(SRC.deepep,'DeepEP')}（MoE 的 dispatch 和 combine 用的 all-to-all kernel）和 ${cite(SRC.eplb,'EPLB')}（冗余专家）。</li>`)+`</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.p_mega,'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism')}<span>${tx('Shoeybi et al., 2019, arXiv 1909.08053','Shoeybi 等，2019，arXiv 1909.08053')}</span></li>`+
   `<li>${cite(SRC.p_pope,'Efficiently Scaling Transformer Inference')}<span>${tx('Pope et al., 2022, arXiv 2211.05102','Pope 等，2022，arXiv 2211.05102')}</span></li>`+
   `<li>${cite(SRC.p_ds,'DeepSeek-V3 Technical Report')}<span>${tx('DeepSeek-AI, 2024, arXiv 2412.19437','DeepSeek-AI，2024，arXiv 2412.19437')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="mgs-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="mgs-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="mgs-app m2" id="mgs-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="mgs-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- the stage ---------- */
const NS='http://www.w3.org/2000/svg';
function groupsOf(c){/* every block of tp consecutive GPUs inside one node: {a, z, rep, st} */
  const n=c.tp*c.pp, R=Math.max(1,Math.floor(c.g/n)), out=[];
  for(let r=0;r<R;r++) for(let s=0;s<c.pp;s++){const a=r*n+s*c.tp, z=a+c.tp-1; out.push({a,z,rep:r,st:s});}
  return out;}
function gpuDiagram(r){
  const host=$('#mgs-gpus',app), c=r.c, nodes=c.g>8?2:1, W=Math.max(260,Math.round(host.clientWidth||600));
  /* two nodes sit side by side when there is room (the stage must fit the screen), else one above the other */
  const side=nodes>1&&W>=400, pad=6, gap=side?3:(W<480?4:8), ngap=side?14:0;
  const cw=side?((W-ngap)/2-2*pad-7*gap)/8:(W-2*pad-7*gap)/8, BH=nodes>1?(W<480?30:40):(W<480?40:52), NH=BH+40;
  const nodeW=side?(W-ngap)/2:W, IBH=nodes>1?(side?16:22):0, H=(side?NH:nodes*NH)+IBH+4;
  const nodeX=k=>side?k*(nodeW+ngap):0, nodeY=k=>side?2:k*(NH+22)+2;
  const gx=i=>nodeX(Math.floor(i/8))+pad+(i%8)*(cw+gap), gy=i=>nodeY(Math.floor(i/8))+19, FULL=80*MGS.GiB;
  const bad=!r.fits;
  let s=`<svg class="mgs-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(tx('GPUs, their memory and their groups','每张卡的显存和分组'))}">`;
  for(let k=0;k<nodes;k++){const x=nodeX(k), y=nodeY(k);
    s+=`<rect class="mgs-node" x="${x+1}" y="${y}" width="${nodeW-2}" height="${NH}" rx="6"/><text class="mgs-nt" x="${x+pad}" y="${y+13}">${esc(tx(`Node ${k+1}: 8 H100s on NVLink`,`机器 ${k+1}：8 张 H100，NVLink 互连`))}</text>`;
    for(let i=0;i<8;i++){const gi=k*8+i, gx0=gx(gi), y0=gy(gi), used=gi<c.g;
      const hW=Math.min(1,r.w/FULL), hO=MGS.OVH/FULL, hK=Math.min(Math.max(0,1-hW-hO),r.need/FULL);
      s+=`<g class="mgs-gpu${used?'':' off'}" data-gpu="${gi}"><rect class="mgs-gbox${bad?' bad':''}" x="${gx0}" y="${y0}" width="${cw}" height="${BH}" rx="3"/>`;
      if(used){let yy=y0+BH; const seg=(h,cls)=>{const hh=h*BH; yy-=hh; return hh>0.2?`<rect class="mgs-m ${cls}" x="${gx0+1}" y="${yy}" width="${cw-2}" height="${hh}"/>`:'';};
        s+=seg(hW,'w')+seg(Math.min(hO,1-hW),'oh')+seg(hK,'kv');}
      s+=`<text class="mgs-gn" x="${gx0+cw/2}" y="${y0+BH-4}" text-anchor="middle">${gi+1}</text></g>`;}
    }
  /* groups: a bracket under each block of tp GPUs in a node, with its name when it fits */
  const G=groupsOf(c);
  for(const g of G){for(let k=0;k<nodes;k++){const a=Math.max(g.a,k*8), z=Math.min(g.z,k*8+7); if(a>z) continue;
      const y=gy(a)+BH+6, x0=gx(a), x1=gx(z)+cw;
      s+=`<path class="mgs-br" d="M${x0} ${y-3}V${y}H${x1}V${y-3}"/>`;
      if(c.tp*c.pp>1){let lab=groupLabel(c,g,k,false); if(lab.length*6.3>x1-x0+6) lab=groupLabel(c,g,k,true);
        if(lab&&lab.length*6.3<=x1-x0+6) s+=`<text class="mgs-bt" x="${(x0+x1)/2}" y="${y+12}" text-anchor="middle">${esc(lab)}</text>`;}}}
  if(c.tp*c.pp===1){const y=gy(0)+BH+18; s+=`<text class="mgs-bt" x="${W/2}" y="${y}" text-anchor="middle">${esc(tx('each GPU keeps a whole copy','每张卡各放一整份模型'))}</text>`;}
  if(nodes>1){const cross=r.ar.cross||r.sends.some(x=>x.x), cl=`mgs-ib${cross&&r.share>=0.5?' bad':''}`, lab=esc(side?tx('InfiniBand between the nodes, 50 GB/s per GPU','两台机器之间走 InfiniBand，每张卡 50 GB/s'):tx('InfiniBand between nodes, 50 GB/s a GPU','机器之间走 InfiniBand，每张卡 50 GB/s'));
    if(side){const xm=W/2, y=NH/2; s+=`<line class="${cl}" x1="${xm-ngap/2+1}" y1="${y}" x2="${xm+ngap/2-1}" y2="${y}"/><path class="${cl}" d="M${xm} ${y}V${NH+8}"/><text class="mgs-ibt" x="${xm}" y="${NH+15}" text-anchor="middle">${lab}</text>`;}
    else {const y=nodeY(0)+NH, xl=pad+12; s+=`<line class="${cl}" x1="${xl}" y1="${y}" x2="${xl}" y2="${y+22}"/><text class="mgs-ibt" x="${xl+8}" y="${y+15}">${lab}</text>`;}}
  /* the ring of each TP group, shown while it merges */
  if(c.tp>1&&r.fits) for(const g of G){const k0=nodeOf8(g.a), k1=nodeOf8(g.z);
      if(k0===k1||side){const x0=gx(g.a)-2, x1=gx(g.z)+cw+2, y0=gy(g.a)-3, y1=gy(g.a)+BH+3;
        s+=`<rect class="mgs-ring${r.share>=0.5?' bad':''}" x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" rx="6"/>`;}
      else {const x0=gx(0)-2, x1=gx(7)+cw+2, y0=gy(g.a)-3, y1=gy(g.z)+BH+3;
        s+=`<rect class="mgs-ring${r.share>=0.5?' bad':''}" x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" rx="6"/>`;}}
  s+='</svg>';
  if(host._h!==s){host.innerHTML=s; host._h=s;}
  $('#mgs-g-h',app).textContent=tx(`Memory of each GPU (80 GiB) and how the GPUs are grouped`,`每张卡的显存（80 GiB）和卡的分组`);}
const nodeOf8=i=>Math.floor(i/8);
function groupLabel(c,g,k,short){
  const R=Math.max(1,Math.floor(c.g/(c.tp*c.pp)));
  if(c.tp>8) return short?tx('TP 16, half','TP 16 的一半'):tx(`TP 16 group (GPUs ${k*8+1}–${k*8+8} of 16)`,`TP 16 的组（第 ${k*8+1}–${k*8+8} 张）`);
  const parts=[], per=MGS.MODELS[c.m].L/c.pp;
  if(R>1) parts.push(short?tx(`rep. ${g.rep+1}`,`副本 ${g.rep+1}`):tx(`replica ${g.rep+1}`,`副本 ${g.rep+1}`));
  if(c.pp>1) parts.push(short?tx(`stage ${g.st+1}`,`第 ${g.st+1} 段`):tx(`stage ${g.st+1}: layers ${g.st*per+1}–${(g.st+1)*per}`,`第 ${g.st+1} 段：第 ${g.st*per+1}–${(g.st+1)*per} 层`));
  if(!parts.length) parts.push(short?`TP ${c.tp}`:tx(`one TP ${c.tp} group`,`一个 TP ${c.tp} 的组`));
  return parts.join(tx(', ','，'));}
function layerHTML(r){
  const c=r.c, Lm=MGS.MODELS[c.m].L;
  $('#mgs-l-h',app).textContent=c.pp>1?tx(`One layer of one stage, on each GPU (each stage has ${Lm/c.pp} layers)`,`一段里的一层，每张卡上（每段 ${Lm/c.pp} 层）`):tx(`One layer of the ${Lm}, on each GPU`,`${Lm} 层中的一层，每张卡上`);
  const host=$('#mgs-layer',app);
  if(!r.fitW){const h=`<p class="mgs-nofit">${tx(`Does not fit: each GPU would need ${fGi(r.w)} for the weights alone, and it can use 70 GiB.`,`放不下：光是权重每张卡就要 ${fGi(r.w)}，能用的只有 70 GiB。`)}</p>`; if(host._h!==h){host.innerHTML=h; host._h=h;} return;}
  const lt=r.layer, tot=lt.attn+lt.mlp+(c.tp>1?2*lt.ar:0), pc=x=>(100*x/tot).toFixed(2), red=r.share>=0.5?' bad':'';
  const segs=[['cm',lt.attn,tx('attention block','attention 块')],...(c.tp>1?[['ar'+red,lt.ar,tx('merge','合并')]]:[]),['cm',lt.mlp,tx('MLP block','MLP 块')],...(c.tp>1?[['ar'+red,lt.ar,tx('merge','合并')]]:[])];
  let x=0; const marks=[];
  const bar=segs.map(([cls,t,lab])=>{const h=`<i class="mgs-seg ${cls}" style="left:${pc(x)}%;width:${pc(t)}%" data-tipx="${esc(`${lab}${tx(': ','：')}${fUs(t)}`)}"></i>`; marks.push(x/tot); x+=t; return h;}).join('');
  const totR=segs.reduce((a,x)=>a+Math.round(x[1]*1e6),0), lab=`<div class="mgs-lscale mono"><span>0</span><span>${tx(`${totR} µs`,`${totR} 微秒`)}</span></div>`;
  const h=`<div class="mgs-lbar" id="mgs-lbar">${bar}<b class="mgs-head" id="mgs-head"></b></div>${lab}`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
  host._marks=marks; host._segs=segs.map(s=>s[0]);}
function sumText(r){
  const c=r.c, Lm=MGS.MODELS[c.m].L;
  if(!r.fitW) return '';
  const parts=[c.pp*r.tcm,c.pp*r.nAR*r.ar.t,r.tSend];
  /* one decimal when the shown parts add up to the shown total, else two */
  let d=1; const sh=(x,k)=>+(x*1e3).toFixed(k); if(r.fits&&Math.abs(parts.reduce((a,x)=>a+sh(x,1),0)-sh(r.tpot,1))>1e-9) d=2;
  const m=x=>tx(`${fx(x*1e3,d)} ms`,`${fx(x*1e3,d)} 毫秒`), tot=parts.reduce((a,x)=>a+sh(x,d),0)/1e3;
  const base=c.tp>1?tx(`A step through all ${Lm} layers: about ${m(parts[0])} of reading and computing${c.pp>1?',':' and'} ${r.nAR*c.pp} merges (${m(parts[1])})`,`一步要走完 ${Lm} 层：读显存和计算约 ${m(parts[0])}，合并 ${r.nAR*c.pp} 次，约 ${m(parts[1])}`)
    :tx(`A step through all ${Lm} layers: about ${m(parts[0])} of reading and computing, with no merges`,`一步要走完 ${Lm} 层：读显存和计算约 ${m(parts[0])}，不用合并`);
  const pp=c.pp>1?tx(` and ${r.sends.length} transfers between stages (${m(parts[2])})`,`，再加 ${r.sends.length} 次段间传输（${m(parts[2])}）`):'';
  const end=r.fits?tx(`. Each output token: ${m(tot)}.`,`。每个输出 token：${m(tot)}。`):tx(`. But the users’ KV cache does not fit (${fGi(r.need)} needed per GPU, ${fGi(r.room)} left).`,`。可是用户的 KV cache 放不下（每张卡要 ${fGi(r.need)}，只剩 ${fGi(r.room)}）。`);
  return base+pp+end;}
function sweepRows(){const c=viewCfg(); return TPS.filter(t=>t>1).map(t=>({t,r:E({m:c.m,g:Math.max(t,1),tp:t,pp:1,u:c.u,ctx:c.ctx})}));}
function sweepHTML(){
  const c=viewCfg(), rows=sweepRows(), mx=Math.max(...rows.filter(x=>x.r.fits).map(x=>x.r.tpot),1e-9);
  $('#mgs-sw-h',app).textContent=tx(`One TP group alone serving all ${c.u} users: time per output token`,`只用一个 TP 组服务全部 ${c.u} 个用户：每个输出 token 要多久`);
  $('#mgs-sw-n',app).textContent=tx('Each row is one group serving all users, the rest idle; splitting into replicas is faster.','每行一个组服务全部用户，其余的卡闲着；分成几个副本会更快。');
  return `<div class="mgs-swp">${rows.map(({t,r})=>{const on=c.tp===t&&c.pp===1&&Math.floor(c.g/t)===1;
    return `<div class="mgs-sw-row${on?' on':''}"><span class="mgs-sw-l">TP ${t}${t>8?tx(' (2 nodes)','（两台机器）'):''}</span><div class="mgs-sw-tr">${r.fits?`<i class="cm" style="width:${(100*r.tcm/mx).toFixed(1)}%"></i><i class="ar${r.share>=0.5?' bad':''}" style="width:${(100*r.tComm/mx).toFixed(1)}%"></i>`:''}</div><span class="mgs-sw-v mono">${r.fits?fMs(r.tpot):tx('does not fit','放不下')}</span></div>`;}).join('')}</div>`;}
function metricsDef(r){
  const no=tx('does not fit','放不下');
  return [{k:'tpot',label:tx('Per output token','每个输出 token'),v:r.fits?fMs(r.tpot):no,sub:tx('one step, every user','一步，每个用户一个'),tip:TIP.tpot()},
   {k:'tps',label:tx('Tokens/s per GPU','每张卡每秒的 token'),v:r.fits?fTok(r.tps):no,sub:tx(`${r.c.u} users, ${r.c.g} GPUs`,`${r.c.u} 个用户，${r.c.g} 张卡`),tip:TIP.tps()},
   {k:'mem',label:tx('Weights per GPU','每张卡的权重'),v:fGi(r.w),sub:r.fitW?tx(`+ KV cache ${fGi(r.need)}`,`另有 KV cache ${fGi(r.need)}`):tx('70 GiB usable','能用 70 GiB'),tip:tx('The weights each GPU keeps, and the KV cache its users need on it. ','每张卡存的权重，以及它的用户在这张卡上要的 KV cache。')+TIP.mem()},
   {k:'comm',label:tx('Merges’ share','合并占的比例'),v:r.fits?fP(r.share):no,sub:tx('red at 50% or more','50% 以上标红'),tip:tx('The time spent in all-reduces (and transfers between pipeline stages), as a share of the whole step.','花在 all-reduce（以及流水线段间传输）上的时间，占整一步的比例。')},
   {k:'ttft',label:tx('First token','首 token'),v:r.fits?fMs(r.ttft):no,sub:tx(`a ${fint(r.c.ctx)}-token prompt`,`${fint(r.c.ctx)} 个 token 的 prompt`),tip:TIP.ttft()}];}
const numOf=s=>{const m=String(s).replace(/,/g,'').match(/-?[\d.]+/); return m?parseFloat(m[0]):NaN;};
const shownPrev={};
function drawMetrics(r){
  const host=$('#mgs-metrics',app), defs=metricsDef(r), sg=defs.map(m=>m.k).join();
  if(host._sig!==sg){host._sig=sg; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; const vv=el.querySelector('.v'), mm=/^([\d.,]+) (\S+)$/.exec(m.v), vh=mm?`${esc(mm[1])}<span class="u"> ${esc(mm[2])}</span>`:esc(m.v); if(vv._h!==vh){vv.innerHTML=vh; vv._h=vh;} vv.classList.toggle('txt',!/^[\d.,]/.test(m.v)); el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.k==='comm'?{pts:true,dp:0}:{dp:m.k==='tps'?0:1})||'';}
    clearTimeout(d._t);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on');},2600);} else if(userAct){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});}

/* Show the math */
const EQ=rows=>`<div class="mgs-eqs">${rows.map(([l,r,rel])=>`<span class="mgs-eq-l">${l?M(l):''}</span><span class="mgs-eq-e">${M(rel||'=')}</span><span class="mgs-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="mgs-mb"><p>${lead}</p>${body}${res?`<p class="mgs-mb-res">${res}</p>`:''}</div>`;
function mathHTML(){
  if(isP('moe')) return '';
  /* every substitution computes to its shown result: results are worked out from the rounded values on the page (R6.6) */
  const r=cur(), c=r.c, Md=MGS.MODELS[c.m], s=r.s, n=c.tp*c.pp, GB=1e9, GiB=MGS.GiB;
  const d2=x=>fx(x,2), g2=b=>d2(b/GB), T=(en,z)=>tx(`\\text{${en}}`,`\\text{${z}}`), msx=T('ms','毫秒'), usx=T('µs','微秒');
  const peak=Md.peak/1e12, peakS=fx(peak,peak%1?1:0), wAll=Md.wb*s.pt, wRead=Md.wb*(s.pls+s.ph)/n, kvRead=r.b*c.ctx*r.kvTok;
  const wGB=+g2(r.w), kvGB=+g2(r.need);
  const one=n===1;
  let h=MB(one?tx(`<b>What each GPU keeps.</b> Each GPU keeps all the weights, and the KV cache of its ${r.u} user${r.u>1?'s':''}:`,`<b>每张卡存什么。</b>每张卡存一整份权重，以及它的 ${r.u} 个用户的 KV cache：`)
       :tx(`<b>What each GPU keeps.</b> The weights split evenly over the ${n} GPUs of a replica; the KV cache splits by KV head (at most 8 ways)${c.pp>1?' and by stage':''}:`,`<b>每张卡存什么。</b>权重平均分到一个副本的 ${n} 张卡上；KV cache 按 KV head 分（最多分 8 份）${c.pp>1?'，再按段分':''}：`),
     EQ([[T('weights per GPU','每张卡的权重'),one?`${g2(wAll)}\\ \\text{GB} = ${f1(r.w/GiB)}\\ \\text{GiB}`:`\\frac{${g2(wAll)}\\ \\text{GB}}{${n}} = ${g2(r.w)}\\ \\text{GB} = ${f1(r.w/GiB)}\\ \\text{GiB}`],
         [T('KV cache per GPU','每张卡的 KV cache'),(()=>{const num=`${r.u} \\times ${fint(c.ctx)} \\times ${fint(s.kv)}\\ \\text{B}`, dv=Math.min(c.tp,8)*c.pp; return dv>1?`\\frac{${num}}{${Math.min(c.tp,8)}${c.pp>1?` \\times ${c.pp}`:''}}`:num;})()],
         ['',`${g2(r.need)}\\ \\text{GB} = ${f1(r.need/GiB)}\\ \\text{GiB}`]]),
     (r.fits?tx(`Together ${d2(wGB+kvGB)} GB = ${f1((wGB+kvGB)*GB/GiB)} GiB: it fits in the 70 GiB each GPU can use.`,`加起来 ${d2(wGB+kvGB)} GB = ${f1((wGB+kvGB)*GB/GiB)} GiB，放得进每张卡能用的 70 GiB。`):tx(`Together they do not fit in the 70 GiB each GPU can use.`,`加起来超过了每张卡能用的 70 GiB。`)));
  if(!r.fitW) return h;
  const aS=+g2(wRead), bS=+g2(kvRead), tm=(aS+bS)*GB/MGS.BS, Fs=+(r.F/1e12).toFixed(r.F<1e12?3:2), tc=Fs*1e12/(peak*1e12*0.5);
  const tmS=+d2(tm*1e3), tcS=+d2(tc*1e3), tcmS=Math.max(tmS,tcS);
  h+=MB(tx(`<b>Reading and computing, per GPU.</b> A decode step takes the slower of the two. Each step reads the weights, except the embedding table (only a few of its rows are looked up), and ${c.pp>1?`the KV cache of one batch: each stage handles ${r.b} users at a time:`:`the KV cache of its ${r.b} user${r.b>1?'s':''}:`}`,`<b>每张卡读和算。</b>一步 decode 的时间取两者中较慢的那个。每步要读权重（不含 embedding 表，它只查用到的几行）和 KV cache；${c.pp>1?`每一段一次处理 ${r.b} 个用户，读的是这一批的 KV cache：`:`这里是 ${r.b} 个用户的 KV cache：`}`),
     EQ([[`t_{${T('read','读')}}`,`\\frac{${g2(wRead)}\\ \\text{GB} + ${g2(kvRead)}\\ \\text{GB}}{3.35\\ \\text{TB/s} \\times 60\\%} = ${d2(tm*1e3)}\\ ${msx}`],
         [`t_{${T('compute','算')}}`,`\\frac{${Fs}\\ \\text{TFLOP}}{${peakS}\\ \\text{TFLOP/s} \\times 50\\%} = ${d2(tc*1e3)}\\ ${msx}`]]),
     tx(`The larger, ${d2(tcmS)} ms, is the time${c.pp>1?' of one stage for one batch of users':''}.`,`较大的 ${d2(tcmS)} 毫秒就是${c.pp>1?'一段处理一批用户':'这部分'}的时间。`));
  let arS=0;
  if(c.tp>1){const k=c.tp>8, Sb=r.b*Md.d*2, bwIn=+d2(r.ar.bwIn*1e6), bwX=+d2(r.ar.bwX*1e6), t0=k?60:18; arS=+d2(t0+bwIn+bwX);
    h+=MB(tx(`<b>One merge (all-reduce).</b> The hidden vectors of ${r.b} users. A fixed cost, then the ring’s bytes${k?'; across the two nodes only 1/8 of the data leaves each node, on each GPU’s InfiniBand NIC':''}:`,`<b>一次合并（all-reduce）。</b>${r.b} 个用户的隐藏向量。先付固定开销，再传环上的字节${k?'；跨两台机器时，每台机器只用每张卡的 InfiniBand 网卡送出 1/8 的数据':''}：`),
      EQ([['S',`${r.b} \\times ${fint(Md.d)} \\times 2\\ \\text{B} = ${fKiB(Sb).replace(/ (\S+)$/,'\\ \\text{$1}')}`],
          ...(k?[[`t_{\\text{AR}}`,`60\\ ${usx} + \\frac{2 \\times 7}{8} \\times \\frac{S}{450\\ \\text{GB/s}} + \\frac{S}{8 \\times 50\\ \\text{GB/s}}`],['',`60 + ${d2(bwIn)} + ${d2(bwX)} = ${d2(arS)}\\ ${usx}`]]
               :[[`t_{\\text{AR}}`,`18\\ ${usx} + \\frac{2 \\times (${c.tp} - 1)}{${c.tp}} \\times \\frac{S}{450\\ \\text{GB/s}}`],['',`18 + ${d2(bwIn)} = ${d2(arS)}\\ ${usx}`]])]),'');}
  const sendS=+d2(r.tSend*1e3), tpS=c.pp*(tcmS+r.nAR*arS/1e3)+sendS;
  h+=MB(tx(`<b>Each output token.</b> ${c.pp>1?`${c.pp} stages, each with ${r.nAR} merges, and ${r.sends.length} transfers between stages:`:c.tp>1?`${r.nAR} merges per step:`:'No merges:'}`,`<b>每个输出 token。</b>${c.pp>1?`${c.pp} 段，每段合并 ${r.nAR} 次，再加 ${r.sends.length} 次段间传输：`:c.tp>1?`每步合并 ${r.nAR} 次：`:'不用合并：'}`),
     EQ([[T('TPOT','TPOT'),c.pp>1?`${c.pp} \\times (${d2(tcmS)}\\ ${msx} + ${r.nAR} \\times ${d2(arS)}\\ ${usx})`:c.tp>1?`${d2(tcmS)}\\ ${msx} + ${r.nAR} \\times ${d2(arS)}\\ ${usx}`:`${d2(tcmS)}\\ ${msx}`],
         ...(c.pp>1?[['',`${d2(sendS)}\\ ${msx}`,'+']]:[]),['',`${d2(tpS)}\\ ${msx}`]]),
     c.pp>1?tx(`The last term is the ${r.sends.length} transfers between stages.`,`最后一项是 ${r.sends.length} 次段间传输。`):'');
  if(r.fits) h+=MB(tx('<b>Tokens per second per GPU.</b> Every user gets one token per step:','<b>每张卡每秒出的 token。</b>每一步每个用户各得一个 token：'),
     EQ([[T('tokens/s per GPU','每张卡每秒的 token'),`\\frac{${c.u}}{${d2(tpS)}\\ ${msx} \\times ${c.g}} = ${fTok(c.u/(+d2(tpS)/1e3)/c.g)}`]]),'');
  return h;}

/* ---------- stage head, status ---------- */
function statusOf(r){
  if(isP('moe')) return {cls:'',short:tx(`busiest ${A.fmt.num(moeK(),2)}× avg.`,`最忙 ${A.fmt.num(moeK(),2)} 倍平均`)};
  if(PLAY.on) return {cls:'',short:tx('playing','播放中')};
  if(!r.fits) return {cls:'bad',short:tx('does not fit','放不下')};
  if(mode==='challenge') return r.tpot<=CH.goal+1e-12?{cls:'ok',short:tx('goal met','达到目标')}:{cls:'bad',short:tx('too slow','太慢')};
  if(r.share>=0.5) return {cls:'bad',short:tx('merges dominate','合并占大头')};
  return {cls:'',short:fMs(r.tpot)};}
function drawHead(r){
  const c=r.c;
  if(isP('moe')){
    $('#mgs-st-title',app).textContent=tx(`One MoE layer of ${MDS}: 8 GPUs with 32 routed experts each`,`${MDS} 的一个 MoE 层：8 张卡各放 32 个路由专家`);
    $('#mgs-st-sub',app).textContent=tx(`${fint(p7)} tokens in the step, each picking 8 experts: how many token–expert pairs each GPU computes. This part is not timed.`,`这一步有 ${fint(p7)} 个 token，每个选 8 个专家：每张卡要算多少 token–专家对。这一部分不计时。`);
    const b=$('#mgs-status',app); b.className='status'; b.textContent=statusOf(r).short; b.dataset.tipx=tx('The busiest GPU’s work against the average; the layer waits for the busiest.','最忙的卡的活和平均比；这一层要等最忙的那张卡。'); return;}
  $('#mgs-st-title',app).textContent=(PLAY.on?tx('Playing one layer: ','正在播放一层：'):'')+tx(`${gpusName(c.g)}, ${setupName(c)}`,`${gpusName(c.g)}，${setupName(c)}`);
  $('#mgs-st-sub',app).textContent=tx(`${L(MNAME[c.m])}; ${c.u} users, ${fint(c.ctx)} tokens of context each; one decode step.`,`${L(MNAME[c.m])}；${c.u} 个用户，每人 ${fint(c.ctx)} 个 token 的上下文；一步 decode。`);
  const st=statusOf(r), b=$('#mgs-status',app); b.className='status '+st.cls; b.textContent=st.short;
  b.dataset.tipx=PLAY.on?tx('One layer is playing.','正在播放一层。'):!r.fits?tx('The weights and the users’ KV cache do not fit in the 70 GiB a GPU can use.','权重和用户的 KV cache 超过了一张卡能用的 70 GiB。'):mode==='challenge'?tx('The goal: an output token every 25 ms or faster.','目标：每个输出 token 不超过 25 毫秒。'):r.share>=0.5?tx('The merges take half the step or more.','合并占了一步一半以上的时间。'):TIP.tpot();}
function drawRunCtl(r){
  const pb=$('#mgs-play',app); if(!pb) return;
  const can=r.fitW&&!isP('moe');
  $('#mgs-run',app).hidden=!can;
  pb.textContent=PLAY.on?tx('❚❚ Pause','❚❚ 暂停'):(ph>=1?tx('▶ Play','▶ 播放'):tx('▶ Continue','▶ 继续'));
  pb.dataset.tipx=tx(`Plays one layer at the chosen speed: about ${RUN_SECS[spdNow()]} s.`,`按选定的速度播放一层：约 ${RUN_SECS[spdNow()]} 秒。`);
  $('#mgs-replay',app).dataset.tipx=tx('Plays this layer again from the start.','从头再播放一次这一层。');
  const pr=$('#mgs-prog',app); pr.value=String(Math.round(1000*Math.min(1,ph)));
  $$('.seg-ctl[data-key="spd"]',app).forEach(g=>A.ui.setPressed(g,spdNow()));}
/* the moving parts: the play head, the GPUs computing, the rings merging */
function phaseAt(p){const host=$('#mgs-layer',app), mk=host&&host._marks, sg=host&&host._segs; if(!mk) return '';
  let i=0; for(let j=0;j<mk.length;j++) if(p>=mk[j]) i=j; return sg[i]||'';}
function moveHead(){
  const hd=$('#mgs-head',app); if(hd) hd.style.left=(100*Math.min(1,ph)).toFixed(2)+'%';
  const p=PLAY.on||ph<1?phaseAt(ph):'', merging=/^ar/.test(p), computing=p==='cm';
  const svg=$('#mgs-gpus svg',app); if(!svg) return;
  svg.classList.toggle('mgs-busy',computing); svg.classList.toggle('mgs-merging',merging);
  if(merging) $$('.mgs-ring',svg).forEach(e=>{e.style.strokeDashoffset=String(-(ph*600)%100);});}
function render(){
  if(!app) return; const r=cur(), k=viewKey(r.c);
  if(k!==lastKey){lastKey=k; if(!PLAY.on) ph=1;}
  const moe=isP('moe');
  $('#mgs-gb',app).hidden=moe; $('#mgs-lb',app).hidden=moe; $('#mgs-epb',app).hidden=!moe; $('#mgs-metrics',app).hidden=moe; $('#mgs-math',app).hidden=moe;
  if(moe){const h=moeHTML(), host=$('#mgs-ep-host',app); if(host._h!==h){host.innerHTML=h; host._h=h;}}
  drawHead(r); if(!moe){gpuDiagram(r); layerHTML(r); $('#mgs-sum',app).textContent=sumText(r);}
  const swb=$('#mgs-swb',app), showSw=mode==='sandbox'; swb.hidden=!showSw;
  if(showSw){const h=sweepHTML(), sw=$('#mgs-sweep',app); if(sw._h!==h){sw.innerHTML=h; sw._h=h;}}
  drawRunCtl(r); moveHead(); if(!moe) drawMetrics(r);
  const mb=$('#mgs-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  // the address names the settings in Scenarios only; leaving Scenarios drops them, so a reload in another mode puts back that mode (round 3 #6)
  if((!PLAY.on||mode!=='sandbox')&&/^#?multi-gpu-serving(~|$)/.test(location.hash||'#multi-gpu-serving')) A.hash.write(token());
  A.dock.refresh(); announce(r);}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app||PLAY.on) return; if(isP('moe')){A.announce(tx(`${fint(p7)} tokens: the busiest GPU has ${A.fmt.num(moeK(),2)} times the average work.`,`${fint(p7)} 个 token：最忙的卡要做平均的 ${A.fmt.num(moeK(),2)} 倍。`)); return;} A.announce(r.fits?tx(`${fMs(r.tpot)} per output token; merges ${fP(r.share)} of the step.`,`每个输出 token ${fMs(r.tpot)}；合并占一步的 ${fP(r.share)}。`):tx('Does not fit.','放不下。'));},400);}

/* ---------- the player (as labs/ppo.js): play / pause, 重放, speed, progress; R3.20 ---------- */
function stopPlay(){if(PLAY.raf) cancelAnimationFrame(PLAY.raf); PLAY.on=false; PLAY.raf=0;}
function playFrom(p0){
  stopPlay(); const r=cur(); if(!r.fitW){ph=1; render(); return;} ph=p0; nPlays++;
  const secs=RUN_SECS[spdNow()];
  PLAY={on:true,raf:0,last:0,key:viewKey(r.c)};
  const tick=ts=>{if(!PLAY.on||!app){PLAY.raf=0; return;}
    if(PLAY.last){ph+=Math.min(0.1,(ts-PLAY.last)/1000)/secs; if(ph>=1){ph=1; watched.add(PLAY.key); PLAY.on=false; PLAY.raf=0; render(); return;} moveHead(); const pr=$('#mgs-prog',app); if(pr) pr.value=String(Math.round(1000*ph));}
    PLAY.last=ts; PLAY.raf=requestAnimationFrame(tick);};
  PLAY.raf=requestAnimationFrame(tick); render();}
/* play the layer only when its picture is new (R3.20) */
function showRun(src){
  const r=cur(), k=viewKey(r.c);
  if(PLAY.on&&PLAY.key===k) return;
  stopPlay();
  if(!r.fitW||watched.has(k)||A.reduceMotion()||src==='quiet'||isP('moe')){ph=1; if(!isP('moe')) watched.add(k); render(); return;}
  playFrom(0);}

/* ---------- the rail ---------- */
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!s.expect())||(!!s.gate&&!s.gate());}
const WHY={size:{en:'Try two ways to split to continue.',zh:'试两种分法才能继续。'},predict:{en:'Pick an answer to continue.',zh:'选一个答案才能继续。'},cross:{en:'Try TP 16 across two nodes to continue.',zh:'试一下跨两台机器的 TP 16 才能继续。'},
  pp:{en:'Try the pipeline and the two replicas to continue.',zh:'试一下流水线和两个副本才能继续。'},moe:{en:'Try another number of tokens to continue.',zh:'换一个 token 数才能继续。'}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(L(x.lead))})),m2:true,current:step,body:s.body(),back:step>0,showMe:false,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function renderRail(){const rail=$('#mgs-rail',app); if(!rail) return; rail.innerHTML=railFull(); render();}
function updateRail(){
  const rail=$('#mgs-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    const show=!!s.obs&&(s.always?!(PLAY.on&&!watched.has(viewKey(viewCfg()))):(s.expect?(s.id==='size'?p3!=null:s.id==='cross'?true:s.id==='pp'?p6!=null:s.id==='moe'?true:s.expect()):true));
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${tx('Observed','观察到')}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}}}
    if(s.id==='size') $$('[data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===p3)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=sizeRes(v);});
    if(s.id==='cross') $$('[data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===p5)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=(v===8?(tried.x8||p5===8):tried.x16)?fMs((v===8?R8():R16()).tpot):'';});
    if(s.id==='pp') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===p6)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tried[v]?fMs(E(P6[v]).tpot):'';});
    if(s.id==='moe') $$('.seg-ctl[data-key="p7"]',rail).forEach(g=>A.ui.setPressed(g,String(p7)));
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#mgs-insight',rail); if(ins){const html=c>=0?`<div class="eyebrow">${esc(L(PRESETS[c].t))}</div>`+scenHTML(c):''; ins.hidden=!html; if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const r=chRes(chPick()), ok=chOk(chPick()), o=$('#mgs-ch-v',rail); if(!o) return;
    o.textContent=r.fits?fMs(r.tpot):tx('does not fit','放不下');
    const pr=$('#mgs-ch-pr i',rail); if(pr) pr.style.width=(r.fits?Math.min(100,100*CH.goal/r.tpot):0).toFixed(0)+'%';
    $('#mgs-ch-g',rail).classList.toggle('met',ok); $('#mgs-ch-g',rail).classList.toggle('mgs-over',!ok);
    const n=S.tp*S.pp, R=Math.max(1,Math.floor(S.g/n));
    $('#mgs-ch-rep',rail).textContent=R>1?tx(`${R} replicas of ${n} GPUs, ${Math.ceil(CH.u/R)} users each.`,`${R} 个副本，每个 ${n} 张卡、${Math.ceil(CH.u/R)} 个用户。`):tx(`One copy on ${n} GPUs serves all ${CH.u} users.`,`一份模型放在 ${n} 张卡上，服务全部 ${CH.u} 个用户。`);
    $$('.seg-ctl[data-key="chg"]',rail).forEach(g=>A.ui.setPressed(g,String(S.g)));
    $$('.seg-ctl[data-key="chtp"]',rail).forEach(g=>{A.ui.setPressed(g,String(S.tp)); $$('button[data-v]',g).forEach(b=>{b.disabled=+b.dataset.v>S.g;});});
    $$('.seg-ctl[data-key="chpp"]',rail).forEach(g=>{A.ui.setPressed(g,String(S.pp)); $$('button[data-v]',g).forEach(b=>{b.disabled=S.tp*(+b.dataset.v)>S.g;});});
    const res=$('#mgs-ch-res',rail), html=chResultHTML(); if(res._h!==html){res.innerHTML=html; res._h=html;}}
}
function syncControls(){
  const c=$('#mgs-controls',app); if(!c) return; const lock=$('#mgs-lock',app), wrap=$('#mgs-ctls',app), free=mode==='sandbox';
  wrap.hidden=!free; lock.hidden=free; lock.textContent=mode==='guided'?tx('Each part sets its own layout; every setting is free in Scenarios.','每一部分有自己的设置；在应用场景里所有设置都能改。'):tx('In the challenge, the choices are in the task card.','挑战题的选择都在任务卡里。');
  if(!free) return;
  const n=norm(S);
  A.ui.setPressed($('.seg-ctl[data-key="m"]',c),n.m); A.ui.setPressed($('.seg-ctl[data-key="g"]',c),String(n.g)); A.ui.setPressed($('.seg-ctl[data-key="tp"]',c),String(n.tp));
  A.ui.setPressed($('.seg-ctl[data-key="pp"]',c),String(n.pp)); A.ui.setPressed($('.seg-ctl[data-key="u"]',c),String(n.u)); A.ui.setPressed($('.seg-ctl[data-key="ctx"]',c),String(n.ctx));}

/* ---------- navigation ---------- */
function enterStep(i){step=Math.max(0,Math.min(STEPS.length-1,i)); renderRail(); showRun(['mission','moe'].includes(STEPS[step].id)?'quiet':'');}
function revealStep(){A.ui.reveal($('#mgs-rail .step.current',app)||$('#mgs-rail',app));}
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2, R9.7), as labs/llm-judge.js */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#mgs-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#mgs-rail .step.current',app)||$('#mgs-rail .railcard',app)||$('#mgs-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function setMode(m){
  if(m===mode) return; const was=mode; mode=m; stopPlay();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH,...CH_START}; delete S.goal;}
  if(m==='sandbox'&&was==='guided') S={...BASE};
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail(); showRun('quiet');}
const token=(s=S)=>{const n=norm(s); return [n.m,n.g,n.tp,n.pp,n.u,n.ctx].join('_');};
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==6) return null;
  const [m,g,tp,pp,u,ctx]=p, o={m,g:+g,tp:+tp,pp:+pp,u:+u,ctx:+ctx};
  if(!MKEYS.includes(m)||![8,16].includes(o.g)||!TPS.includes(o.tp)||!PPS.includes(o.pp)||!USERS.includes(o.u)||!CTXS.includes(o.ctx)||o.tp*o.pp>o.g) return null;
  return o;}

/* ---------- interview drill ---------- */
let drillSeed=1, drillOpen=false;
function makeDrill(seed){const rnd=MGS.mulberry(seed*7919+5), tps=[4,8], us=[8,16,32,64];   // every pick fits at 4,096 tokens (O7)
  return {tp:tps[Math.floor(rnd()*2)],u:us[Math.floor(rnd()*4)]};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#mgs-drill-q',app), a=$('#mgs-drill-a',app);
  q.innerHTML=tx(`${M70} (80 layers, hidden size 8,192, BF16 activations) is served with TP ${d.tp} inside one node to ${d.u} users. In one decode step, how many all-reduces run, how many bytes does each carry, and about how long do they take in all? Take 450 GB/s for NVLink and a fixed cost of 18 µs per all-reduce.`,
                 `${M70}（80 层，隐藏维度 8,192，激活用 BF16）在一台机器里用 TP ${d.tp}，服务 ${d.u} 个用户。一步 decode 里有几次 all-reduce？每次传多少字节？加起来大约多久？NVLink 按 450 GB/s，每次 all-reduce 的固定开销按 18 微秒算。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  /* S27: the shown time per all-reduce has two decimals, and the total is computed from it */
  const S0=d.u*8192*2, ar=MGS.arTime(S0,d.tp), arS=fx(ar.t*1e6,2), tot=fx(160*(+arS)/1e3,2), usx=tx('\\mu\\text{s}','\\text{微秒}'), msx=tx('\\text{ms}','\\text{毫秒}');
  const eq=EQ([['S',`${d.u} \\times 8,192 \\times 2 = ${fint(S0)}\\ \\text{B}`],['t_{\\text{AR}}',`18\\ ${usx} + \\frac{2 \\times (${d.tp} - 1)}{${d.tp}} \\times \\frac{${fint(S0)}\\ \\text{B}}{450\\ \\text{GB/s}} = ${arS}\\ ${usx}`],[tx('\\text{total}','\\text{合计}'),`160 \\times ${arS}\\ ${usx} = ${tot}\\ ${msx}`]]);
  a.innerHTML=tx(`<p><b>160 all-reduces of ${fKiB(S0)} each, about ${tot} ms per step.</b></p>${eq}<p>Two per layer (after the attention block and after the MLP block), 80 layers. At these sizes most of each all-reduce is its fixed cost: even a message half this size would leave the total almost unchanged.</p>`,
                 `<p><b>160 次 all-reduce，每次 ${fKiB(S0)}，每步约 ${tot} 毫秒。</b></p>${eq}<p>每层两次（attention 块之后、MLP 块之后），一共 80 层。这么小的消息，每次 all-reduce 的时间大部分是固定开销：就算消息再小一半，总时间也几乎不变。</p>`);}

/* ---------- jumps ---------- */
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  return '';}
function jump(k){
  if(!app) return;
  if(k==='math'){const d=$('#mgs-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#mgs-controls',app));}}

/* ---------- events ---------- */
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
function setS(p){Object.assign(S,p); S=norm(S); byLearner(()=>{render(); showRun('settings');});}
function act(a,btn){
  if(a==='play'){if(PLAY.on){stopPlay(); render(); return;} playFrom(ph>=1?0:ph); return;}
  if(a==='replay'){playFrom(0); return;}
  if(a==='reset'){setS({...BASE}); return;}
  if(a==='submit'){chSubmitted=true; renderRail(); requestAnimationFrame(()=>revealBox($('#mgs-ch-res',app))); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#mgs-code',app); try{navigator.clipboard.writeText(pre.textContent);}catch(e){} btn.textContent=tx('Copied','已复制'); setTimeout(()=>{btn.textContent=A.t('copyCode');},1600);}}
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)||b.disabled) return; const d=b.dataset;
    if(!d.jump&&b.closest('#mgs-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#mgs-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('size')){p3=+d.lever; tried['s'+p3]=true; byLearner(()=>{render(); showRun();}); requestAnimationFrame(()=>revealBox($('#mgs-rail [data-obs]',app))); return;}
    if(d.lever!=null&&isP('cross')){p5=+d.lever; tried['x'+p5]=true; byLearner(()=>{render(); showRun();}); requestAnimationFrame(()=>revealBox($('#mgs-rail [data-obs]',app))); return;}
    if(d.lever!=null&&isP('pp')){p6=d.lever; tried[p6]=true; byLearner(()=>{render(); showRun();}); requestAnimationFrame(()=>revealBox($('#mgs-rail [data-obs]',app))); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='spd'){spdUser=v; if(PLAY.on) playFrom(ph); else render(); return;}
      if(k==='p7'){p7=+v; if(p7!==32) tried.m=true; byLearner(()=>render()); requestAnimationFrame(()=>revealBox($('#mgs-rail [data-obs]',app))); return;}
      if(k==='chg') return byLearner(()=>{S.g=+v; chFix(); render(); showRun('settings');});
      if(k==='chtp') return byLearner(()=>{S.tp=+v; chFix(); render(); showRun('settings');});
      if(k==='chpp') return byLearner(()=>{S.pp=+v; render(); showRun('settings');});
      if(k==='m') return setS({m:v}); if(k==='g') return setS(+v===8?fit8():{g:16});
      if(k==='tp') return setS({tp:+v}); if(k==='pp') return setS({pp:+v}); if(k==='u') return setS({u:+v}); if(k==='ctx') return setS({ctx:+v});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; S={...p.s}; byLearner(()=>{render(); showRun('settings');}); requestAnimationFrame(()=>{if(app) revealBox($('#mgs-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#mgs-rail',app)); return;}
    if(d.act) act(d.act,b);});
  el.addEventListener('input',e=>{const t=e.target;
    if(t.id==='mgs-prog'){stopPlay(); ph=Math.min(1,(+t.value)/1000); if(ph>=1) watched.add(viewKey(cur().c)); moveHead(); drawRunCtl(cur());}});
  el.addEventListener('change',e=>{const t=e.target; if(t.id==='mgs-prog'&&+t.value>=1000){ph=1; render();}});
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));}

/* ---------- registration ---------- */
const sOf=s=>({m:s.m,g:s.g,tp:s.tp,pp:s.pp,u:s.u,ctx:s.ctx});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:sOf(S),pred,p3,p5,p6,p7,tried:{...tried},saved:saved?sOf(saved):null,chSubmitted,chReveal,spdUser,watched:[...watched],
    drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chOk(chPick())}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=mode==='challenge'?{...p,m:CH.m,u:CH.u,ctx:CH.ctx}:p; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    p3=[2,4,8].includes(o.p3)?o.p3:null; p5=[8,16].includes(o.p5)?o.p5:8; p6=P6[o.p6]?o.p6:null; p7=[32,256,2048].includes(o.p7)?o.p7:32;
    tried={}; if(o.tried&&typeof o.tried==='object') for(const k in o.tried) if(/^(s[248]|x8|x16|tp16|pp|rep|m)$/.test(k)&&o.tried[k]===true) tried[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sv:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; spdUser=RUN_SECS[o.spdUser]?o.spdUser:null;
    watched.clear(); if(Array.isArray(o.watched)) o.watched.slice(-200).forEach(k=>{const q=typeof k==='string'?k.split('|'):[]; if(q.length===6&&parse(q.join('_'))&&k===viewKey(parse(q.join('_')))) watched.add(k);});
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    root=el; const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#mgs-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#mgs-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    const rail=$('#mgs-rail',app); rail.innerHTML=railFull();
    ph=1; render();
    if(mode==='guided'&&STEPS[step].id!=='mission') showRun();},
  leave(){stopPlay(); clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#mgs-math',app), open=d&&d.open, wasPlaying=PLAY.on; stopPlay(); this.mount(root,null); if(open) $('#mgs-math',app).open=true; if(wasPlaying) showRun();}},
  onResize(){if(app){const g=$('#mgs-gpus',app); if(g) g._h=''; render();}},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#mgs-gpus',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#mgs-graph',app),
    render({status,vis,read}){if(!app) return; const r=cur(), s=statusOf(r);
      status.className='status '+s.cls; status.textContent=s.short;
      if(isP('moe')){const per=MGS.route(p7), mx=Math.max(...per);
        vis.innerHTML=`<div class="mgs-dk mgs-dk-ep">${per.map(v=>`<i class="${v===mx?'bad':'cm'}" style="height:${(100*v/mx).toFixed(0)}%"></i>`).join('')}</div>`;
        read.textContent=tx(`One MoE layer, ${fint(p7)} tokens: the busiest GPU has ${A.fmt.num(moeK(),2)} times the average work.`,`一个 MoE 层，${fint(p7)} 个 token：最忙的卡要做平均的 ${A.fmt.num(moeK(),2)} 倍。`); return;}
      vis.innerHTML=`<div class="mgs-dk"><i class="cm" style="width:${r.fits?(100*(1-r.share)).toFixed(0):0}%"></i><i class="ar${r.share>=0.5?' bad':''}" style="width:${r.fits?(100*r.share).toFixed(0):0}%"></i></div>`;
      read.textContent=r.fits?tx(`${setupName(r.c)}: ${fMs(r.tpot)} per output token; merges take ${fP(r.share)} of the step.`,`${setupName(r.c)}：每个输出 token ${fMs(r.tpot)}，合并占一步的 ${fP(r.share)}。`):tx(`${setupName(r.c)}: does not fit.`,`${setupName(r.c)}：放不下。`);}}
});

/* ---------- read-only hook for tests ---------- */
window.__mgs=Object.freeze({
  engine:MGS, STEP_IDS:STEPS.map(s=>s.id), PRESETS:PRESETS.map(p=>({id:p.id,s:{...p.s}})), CH:{...CH}, CH_START:{...CH_START}, PRED_RIGHT,
  view:()=>viewCfg(),
  state:()=>({...S,mode,step,pred,p3,p5,p6,p7,tried:{...tried},chSubmitted,chReveal,playing:PLAY.on,ph,plays:nPlays,watched:[...watched],spd:spdNow(),spdUser}),
  chOpts:()=>CH_OPTS().map(o=>{const r=chRes(o); return {...o,ok:chOk(o),fits:r.fits,tpot:r.tpot,tps:r.tps};}),
  chBest:()=>{const b=chBest(); return b&&{...b};}, sens:()=>SENS(),
  exits:()=>{const e=EXITS(); return {transferA:e.transfer.map(q=>q.a)};}
});

})();
