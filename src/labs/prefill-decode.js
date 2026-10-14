/* prefill-decode (the "SDE → LLM serving" path, stop 3 of 7): Why is the first token slow and the rest fast?
   Fidelity: Formula. Every time on the page comes from teasers/set-a.js's step model (same constants, same formulas), so the lab's
   numbers equal the home tile's: Llama-3.1-8B in BF16 on one H100 SXM, a step takes the longer of its arithmetic (at 50% of the dense
   BF16 peak) and its memory traffic (at 60% of the peak bandwidth).
   - Prefill of an N-token prompt: step({tok:N, logits:1, attn:attn(0,N), kv:N}); its time is the time to first token (no queue here).
   - A decode step for B requests at context c: step({tok:B, logits:B, attn:B*attn(c,c+1), kv:B*c}); its time is the time per output
     token. The stage prices the first decode step, whose context is the prompt; the whole answer adds up every step as the context grows.
   - The most requests that fit: the KV cache memory lab's engine (A.kv.compute with its defaults: 80 GB read as 80 GiB, 90% usable,
     2 GiB runtime overhead), so both labs of the path give the same number.
   Words (EN / ZH): prefill; decode step / decode 步; time to first token (TTFT) / 首 token 延迟（TTFT）; time per output token (TPOT) /
   每个输出 token 的耗时（TPOT）; arithmetic / 计算; memory / 读写显存 (the bytes a step moves); batch; part / 部分 for guided cards.
   window.__pd is a read-only hook for tests/prefill-decode_numbers.py and tests/prefill-decode_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='prefill-decode';
/* Chinese keeps a space between Chinese characters and Latin letters or digits (“8 个 KV head”), as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const MODEL='Llama‑3.1‑8B', GPUN='H100 SXM';      // no-break hyphens and space: a model or GPU name never breaks

/* =====================================================================
   engine: teasers/set-a.js's step model, the same constants and formulas
   ===================================================================== */
const H100={name:GPUN,peak:1979e12/2,bw:3.35e12,mem:80};       // dense BF16 = the datasheet's 1,979 TFLOPS "with sparsity" ÷ 2; 3.35 TB/s; 80 GB
const A100={name:'A100 80GB SXM',peak:312e12,bw:2.039e12,mem:80};   // the exits' new case (datasheet: 312 TFLOPS dense BF16, 2,039 GB/s)
const EFF={mfu:0.5,mbu:0.6};                                    // assumed: 50% of peak FLOP/s, 60% of peak bandwidth (as set-a)
const LL={name:'Llama-3.1-8B',layers:32,hidden:4096,heads:32,kvHeads:8,headDim:128,vocab:128256,ffn:14336,maxCtx:131072};
const P_EMB=LL.vocab*LL.hidden, P_HEAD=LL.vocab*LL.hidden;
const P_LAYER=2*LL.hidden*LL.hidden+2*LL.hidden*LL.kvHeads*LL.headDim+3*LL.hidden*LL.ffn+2*LL.hidden;
const P_LAYERS=LL.layers*P_LAYER+LL.hidden, P_TOTAL=P_LAYERS+P_EMB+P_HEAD;
const KV_TOK=2*LL.layers*LL.kvHeads*LL.headDim*2;              // K and V, BF16: 131,072 bytes = 128 KiB per token
const W_READ=2*(P_LAYERS+P_HEAD);                               // bytes one step reads: every weight but the embedding table (a lookup)
const W_MEM=2*P_TOTAL;                                          // bytes the weights take in GPU memory
const attn=(a,b)=>2*LL.layers*LL.hidden*(b*(b+1)-a*(a+1));      // QKᵀ and AV FLOPs, all layers, for query positions a..b-1
function stepOn(g,o){
  const fl=2*P_LAYERS*o.tok+2*P_HEAD*o.logits+o.attn, by=W_READ+KV_TOK*o.kv, fs=g.peak*EFF.mfu, bs=g.bw*EFF.mbu;
  const tc=fl/fs, tm=by/bs;
  return {t:Math.max(tc,tm),tc,tm,fl,by,tw:W_READ/bs,tkv:KV_TOK*o.kv/bs,ai:fl/by,cb:tc>=tm,tok:o.tok,kv:o.kv};
}
const prefill=(N,g)=>stepOn(g||H100,{tok:N,logits:1,attn:attn(0,N),kv:N});
const decode=(B,c,g)=>stepOn(g||H100,{tok:B,logits:B,attn:B*attn(c,c+1),kv:B*c});
const RIDGE=(H100.peak*EFF.mfu)/(H100.bw*EFF.mbu);
/* the ridge for any efficiencies from 20% to 100% (the fidelity note: the phases stay on their sides) */
const RIDGE_LO=H100.peak*0.2/H100.bw, RIDGE_HI=H100.peak/(H100.bw*0.2);             // FLOPs per byte where the two limits meet: ≈ 246
/* the KV cache memory lab's defaults: the share of memory a server uses and the runtime overhead */
const KVD=A.kv&&A.kv.DEFAULT?A.kv.DEFAULT:{util:0.9,overhead:2};
const UTIL=KVD.util, OH=KVD.overhead;
function memOf(c,B){
  if(A.kv&&A.kv.compute) return A.kv.compute({model:'llama31-8b',gpu:'h100',n:1,ctx:c,users:B||1,wp:'bf16',kp:'bf16',attn:'gqa',util:UTIL,overhead:OH});
  const cap=H100.mem*A.GiB, usable=cap*UTIL, oh=OH*A.GiB, kvPerReq=KV_TOK*c, budget=usable-W_MEM-oh;
  return {cap,usable,oh,wPerGpu:W_MEM,kvPerReq,kvPerGpu:kvPerReq*(B||1),need:W_MEM+kvPerReq*(B||1)+oh,budget,maxUsers:budget>0?Math.floor(budget/kvPerReq+1e-9):0};
}
const memLimit=c=>memOf(c,1).maxUsers;
const N_STOPS=[512,1024,2048,4096,8192,16384,32768], M_STOPS=[20,50,100,200,300,500,1000];
const BASE={N:2048,M:300,B:1};
const TL_K=24;                                                  // decode ticks drawn on the timeline
/* one request with an N-token prompt and an M-token answer, decoding together with B − 1 others at the same context */
const AC=new Map();
function compute(s){
  const key=s.N+'|'+s.M+'|'+s.B; if(AC.has(key)) return AC.get(key);
  const N=s.N, M=s.M, B=s.B, pre=prefill(N), dec=decode(B,N);
  let sum=0; const ticks=[]; let t=pre.t;
  for(let k=0;k<M-1;k++){const d=decode(B,N+k).t; sum+=d; if(k<TL_K){t+=d; ticks.push(t);}}
  const lim=memLimit(N);
  const r={N,M,B,pre,dec,ttft:pre.t,tpot:dec.t,tps:1/dec.t,thr:B/dec.t,dsum:sum,e2e:pre.t+sum,avg:M>1?sum/(M-1):dec.t,ticks,lim};
  AC.set(key,r); return r;
}
/* the challenge: 4K tokens of context each, at most 20 ms per output token (50 tokens per second each); choose the batch size */
const CH={N:4096,M:300,goal:0.020,start:1,rows:[1,16,32,44,46,47,48,64]};
const chPoint=B=>{const d=decode(B,CH.N); return {B,t:d.t,tps:1/d.t,thr:B/d.t,ok:d.t<=CH.goal,d};};
const chMax=()=>memLimit(CH.N);
function chBest(){let best=0; for(let B=1;B<=chMax();B++){if(chPoint(B).ok) best=B;} return best;}
/* part 6, the experiment: one number of the H100 doubled, everything else the same (the model, the 50% and 60%, the load). An idealised
   variant, not a real GPU; it exists only in this part (Scenarios and the challenge keep the real H100). One request at a time, priced
   as compute() prices it: TTFT = the prefill, TPOT = the first decode step, whole answer = TTFT + every decode step as the context grows */
const XV=[{id:'orig',g:H100,t:{en:'Original H100',zh:'原版 H100'}},
  {id:'c2',g:{...H100,peak:2*H100.peak},t:{en:'Compute × 2',zh:'算力 × 2'}},
  {id:'b2',g:{...H100,bw:2*H100.bw},t:{en:'Bandwidth × 2',zh:'带宽 × 2'}}];
const XL=[{id:'chat',N:2048,M:300,t:{en:'Chat',zh:'聊天'}},{id:'doc',N:32768,M:20,t:{en:'Long document',zh:'长文档'}}];
const xVar=id=>XV.find(v=>v.id===id)||XV[0], xLoad=id=>XL.find(l=>l.id===id)||XL[0];
const XC=new Map();
function xRun(vid,lid){
  const key=vid+'|'+lid; if(XC.has(key)) return XC.get(key);
  const v=xVar(vid), l=xLoad(lid), g=v.g, N=l.N, M=l.M, pre=prefill(N,g), dec=decode(1,N,g);
  let sum=0, t=pre.t; const times=[];                            // when each later token appears
  for(let k=0;k<M-1;k++){const d=decode(1,N+k,g).t; sum+=d; t+=d; times.push(t);}
  const r={v:v.id,l:l.id,N,M,pre,dec,ttft:pre.t,tpot:dec.t,dsum:sum,e2e:pre.t+sum,avg:M>1?sum/(M-1):dec.t,times,g:{peak:g.peak,bw:g.bw}};
  XC.set(key,r); return r;
}

/* =====================================================================
   formatting (every derived number is computed from the value as shown, or the line gives the digits it needs, R6.6)
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const NS=' ';
/* prefill times, the time to first token: one decimal in ms below 1 s, two decimals in s from 1 s (60.0 ms, 1.49 s) */
const fPre=sec=>{const v=sec*1e3; return v<999.95?fixd(v,1)+NS+'ms':fixd(v/1e3,2)+NS+'s';};
const preMs=sec=>{const v=sec*1e3; return v<999.95?Math.round(v*10)/10:Math.round(v/10)*10;};   // the shown value, in ms
/* a decode step: two decimals in ms (7.60 ms), three below 1 ms (0.033 ms) */
const fDec=sec=>{const v=sec*1e3; return v<0.9995?fixd(v,3)+NS+'ms':fixd(v,2)+NS+'ms';};
const decMs=sec=>{const v=sec*1e3; return v<0.9995?Math.round(v*1000)/1000:Math.round(v*100)/100;};
const fDec3=sec=>fixd(sec*1e3,3)+NS+'ms';                       // three decimals, for a line that divides by it
/* a whole answer: whole ms below 1 s, two decimals below 10 s, then one */
const fTot=sec=>sec<0.9995?fint(sec*1e3)+NS+'ms':(sec<9.995?fixd(sec,2)+NS+'s':fixd(sec,1)+NS+'s');
/* tokens per second, from the exact time: whole numbers from 100, one decimal below (49.9, not 50: the goal is at least 50) */
const fTps=v=>v>=99.5?fint(v):fixd(Math.floor(v*10+1e-9)/10,1);
const fThr=v=>fint(v);
const ratioS=q=>q>=9.95?fint(q):fixd(q,1);
const tokK=n=>n>=1024&&n%1024===0?(n/1024)+'K':fint(n);
const tokFull=n=>{const k=tokK(n); return k===fint(n)?k:tx(`${k} (${fint(n)})`,`${k}（${fint(n)}）`);};
const tokFullN=n=>{const k=tokK(n); return k===fint(n)?tx(`${k} tokens`,`${k} 个 token`):tx(`${k} (${fint(n)}) tokens`,`${k}（${fint(n)}）个 token`);};
const gbS=b=>{const v=b/1e9; return (v>=9.995?fixd(v,1):(v>=0.0995?fixd(v,2):fixd(v,3)))+NS+'GB';};
const gbD=b=>gbS(b).replace(/GB$/,()=>DEF('GB',TIP.gb()));        // the unit defined on hover where GB first appears (R6.12)
const an=n=>/^8/.test(String(n))?tx('an','an'):tx('a','a');     // "an 8K-token prompt", "a 2K-token prompt"
const gib1=b=>fixd(b/A.GiB,1)+NS+'GiB';
const gib2=b=>fixd(b/A.GiB,2)+NS+'GiB';
const flS=f=>f>=1e12?fixd(f/1e12,2)+NS+'TFLOP':fixd(f/1e9,1)+NS+'GFLOP';
const tfS=v=>fixd(v/1e12,2);                                    // 494.75 (TFLOP/s)
const tbS=v=>fixd(v/1e12,2);                                    // 2.01 (TB/s)
const reqN=n=>tx(n===1?'1 request':`${fint(n)} requests`,`${fint(n)} 个请求`);
const userN=n=>tx(n===1?'1 user':`${fint(n)} users`,`${fint(n)} 个用户`);
const tokN=n=>tx(n===1?'1 token':`${fint(n)} tokens`,`${fint(n)} 个 token`);
const pct0=x=>fint(x*100)+'%';
/* numbers in prose (R6.20): rounded for reading, about two significant digits (1.5 s, 270 ms, 60 ms, 7.6 ms); the exact values stay on
   the stage, the metric cards and Show the math */
const fR=sec=>{const v=sec*1e3; if(v>=9950) return fint(Math.round(v/1e3))+NS+'s'; if(v>=950) return fixd(v/1e3,1)+NS+'s'; if(v>=99.5) return fint(Math.round(v/10)*10)+NS+'ms'; if(v>=9.95) return fint(Math.round(v))+NS+'ms'; return fixd(v,1)+NS+'ms';};
/* how many times as long each request (or user) waits for its next token as alone, in words (q: the time per output token over one
   request's): 64 requests at 2K wait about twice as long, 220 about 5 times */
const waitS=(q,user)=>{const e=user?'user':'request', z=user?'用户':'请求';
  return q<1.5?tx(`Each ${e} waits only a little longer for its next token than alone.`,`每个${z}等下一个 token 的时间只比单独跑时长一点。`)
    :(q<2.5?tx(`Each ${e} waits about twice as long for its next token as alone.`,`每个${z}等下一个 token 的时间约是单独跑时的两倍。`)
      :tx(`Each ${e} waits about ${fint(Math.round(q))} times as long for its next token as alone.`,`每个${z}等下一个 token 的时间约是单独跑时的 ${fint(Math.round(q))} 倍。`));};
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const KVH=()=>(A.terms&&A.terms.kvhead?A.terms.kvhead():tx('A KV head is one set of keys and values that several query heads share.','KV head 是几个 query head 共用的一组 key 和 value。'));
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let saved=null, chSubmitted=false, chReveal=false, drillSeed=2, drillOpen=false, annT=null, userAct=false, wantObs=false, sbTouched=false;
let roofOpen=false, roofSeen=false, clampNote=false;
/* part 6: the variant and the load picked (both reset when the learner leaves the part), the prediction, and whether compute × 2 was seen */
let xv='orig', xl='chat', xpred=null, xSeenC2=false, xK=5;   // xK: a thin tick every xK tokens on the three timelines
const shownPrev={};
const R=()=>compute(S);
const B0=()=>compute({N:2048,M:300,B:1});       // the mission's request, the baseline every part compares with

/* =====================================================================
   definitions shown on hover (R2.8): a term is defined where a reader first meets it
   ===================================================================== */
const TIP={
 prefill:()=>tx('Prefill is the first step of answering a request: the model takes in the whole prompt in one pass, writes every prompt token’s keys and values to the KV cache, and produces the first token of the answer.',
                'prefill 是回答一个请求的第一步：模型一次读入整个 prompt，把每个 prompt token 的 key 和 value 写进 KV cache，并产出回答的第一个 token。'),
 decode:()=>tx('Decode is everything after prefill: each decode step runs the newest token through the model and produces one more token, until the answer is done.',
               'decode 是 prefill 之后的全部过程：每个 decode 步把最新的一个 token 过一遍模型，多产出一个 token，直到回答写完。'),
 dstep:()=>tx('A decode step is one pass of the model that produces one new token for each request in it. It computes only the newest token, and reads the keys and values of the earlier tokens from the KV cache.',
              'decode 步是把模型跑一遍、给其中每个请求各多产出一个 token 的过程。它只算最新的那个 token，之前 token 的 key 和 value 从 KV cache 里读。'),
 ttft:()=>tx('Time to first token (TTFT): how long a user waits from sending the prompt to seeing the first token of the answer. This lab has no queue, so it is the prefill’s time.',
             '首 token 延迟（TTFT，time to first token）：从发出 prompt 到看到回答的第一个 token 要等多久。这个 Lab 里没有排队，所以它就是 prefill 的耗时。'),
 tpot:()=>tx('Time per output token (TPOT): the time between two tokens of the answer once it has started. Each of them takes one decode step, so in this lab it is the time of one decode step.',
             '每个输出 token 的耗时（TPOT，time per output token）：回答开始以后，两个 token 之间隔多久。之后每个 token 都要一个 decode 步，所以在这个 Lab 里它就是一个 decode 步的耗时。'),
 overlap:()=>tx('A GPU reads memory and does arithmetic at the same time: while it computes with one chunk of the weights, it is already reading the next. So a step takes about as long as the slower of the two, not their sum. This is this lab’s estimate; real kernels overlap the two only in part.',
                'GPU 读写显存和做计算是同时进行的：它用已读到的一段权重计算时，已经在读下一段。所以一步的耗时约等于两者中较慢的那个，而不是两者相加。这是这个 Lab 的估计；真实 kernel 只能让两者部分重叠。'),
 wread:()=>tx(`In every step the GPU streams every parameter it multiplies (all but the embedding table: ${gbS(W_READ)}, ${gib1(W_READ)}, for ${MODEL}) from GPU memory into its compute units once. That one read serves every token the step computes, so it costs the same for 1 token or 1,000.`,
              `每一步，GPU 都要把参与乘法的全部参数（embedding 表以外的全部参数，${MODEL} 是 ${gbS(W_READ)}，即 ${gib1(W_READ)}）从显存读进计算单元一次。这一次读，供这一步算的所有 token 共用：算 1 个 token 和算 1,000 个 token 读的一样多。`),
 flop:()=>tx('A FLOP is one floating-point operation; a multiply and an add count as 2. A GFLOP is 10^9 FLOPs and a TFLOP 10^12, so TFLOP/s measures how fast a GPU computes.',
             'FLOP 是一次浮点运算，一次乘法加一次加法算 2 个 FLOP。GFLOP 是 10^9 个 FLOP，TFLOP 是 10^12 个，所以 TFLOP/s 衡量 GPU 算得多快。'),
 gb:()=>tx('GB and TB here are 10^9 and 10^12 bytes, to match the bandwidth in TB/s. 15.0 GB is 14.0 GiB, the unit the KV cache labs use for memory.',
           '这里的 GB、TB 是 10^9、10^12 字节，和带宽的 TB/s 一致。15.0 GB 就是 14.0 GiB，KV cache 那两个 Lab 用 GiB 表示显存。'),
 ctx:()=>tx('A decode step’s context is the tokens before its newest one: the prompt plus the answer tokens written so far. Their keys and values are what the step reads from the KV cache. The first decode step’s context is the prompt.',
            'decode 步的上下文是它最新那个 token 之前的全部 token：prompt 加上已经写出的回答。这一步要从 KV cache 里读的，就是它们的 key 和 value。第一个 decode 步的上下文就是 prompt。'),
 batch:()=>tx('A batch is the set of requests one step serves together. A decode step for a batch of B requests computes the newest token of each and reads each one’s own KV cache, but reads the weights only once for all of them.',
              'batch 是一步里一起处理的一组请求。B 个请求的 decode 步算出每个请求的最新 token，读每个请求自己的 KV cache，但权重只为它们一起读一遍。'),
 thr:()=>tx('Throughput is the tokens per second the GPU writes for all the requests together.',
            '吞吐是 GPU 每秒为所有请求一共写出的 token 数。')+FXT(tx('\\text{throughput} = \\frac{\\text{batch size}}{\\text{TPOT}}','\\text{吞吐} = \\frac{\\text{batch 大小}}{\\text{TPOT}}'))+tx('TPOT is the time per output token.','TPOT 是每个输出 token 的耗时。'),
 ai:()=>tx('Arithmetic intensity says how many FLOPs a step does for each byte it moves to or from GPU memory. The Roofline paper calls it operational intensity.','算术强度说的是，一步每和显存之间搬运 1 个字节，要做多少个 FLOP。Roofline 论文叫它 operational intensity。')+FXT(tx('\\text{arithmetic intensity} = \\frac{\\text{FLOP}}{\\text{bytes moved}}','\\text{算术强度} = \\frac{\\text{FLOP 数}}{\\text{搬运的字节数}}'))+tx('Below the ridge, memory sets the time (memory-bound). Above it, arithmetic does (compute-bound).','低于拐点时，读写显存决定耗时（受显存带宽限制）。高于拐点时，计算决定耗时（受算力限制）。'),
 ridge:()=>tx('The ridge is the arithmetic intensity at which the two limits take equal time.','拐点是两种限制耗时相等时的算术强度。')+FXT(tx('\\text{ridge} = \\frac{\\text{FLOP/s the GPU reaches}}{\\text{bandwidth}}','\\text{拐点} = \\frac{\\text{GPU 能达到的 FLOP/s}}{\\text{带宽}}'))+tx(`With this lab’s efficiencies: ${tfS(H100.peak*EFF.mfu)} TFLOP/s ÷ ${tbS(H100.bw*EFF.mbu)} TB/s = ${fint(RIDGE)} FLOPs per byte.`,`按这个 Lab 的效率：${tfS(H100.peak*EFF.mfu)} TFLOP/s ÷ ${tbS(H100.bw*EFF.mbu)} TB/s = ${fint(RIDGE)}，即每字节 ${fint(RIDGE)} 个 FLOP。`),
 gib:()=>kt('gib'), kib:()=>kt('kib'), kv:()=>kt('kvcache')
};
/* H1 (the owner, 2026-10-09): every formula typeset with the kit's TeX subset, a formula the reader needs on its own line with its name */
const FX=(tex,c)=>`<div class="${c||'pd-fx'}">${A.ui.math(tex)}</div>`;
const FXT=tex=>FX(tex,'fx');

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'both',t:{en:'Both grow about 4 times',zh:'两个都变成约 4 倍'}},
  {v:'first',t:{en:'The first token takes about 4 times as long; later tokens barely change',zh:'首 token 约慢到 4 倍，之后的 token 几乎不变'}},
  {v:'later',t:{en:'Later tokens take about 4 times as long; the first token barely changes',zh:'之后的 token 约慢到 4 倍，首 token 几乎不变'}},
  {v:'neither',t:{en:'Neither changes much',zh:'两个都变化不大'}}];
const PRED_RIGHT='first';
const predT=v=>L(PRED.find(o=>o.v===v).t);
/* part 6's prediction: twice the FLOP/s, the same bandwidth (the answer is hidden until a pick, R3.8) */
const XPRED=[{v:'both',t:{en:'Both the first token and every later token, about twice as fast',zh:'首 token 和之后的每个 token 都快一倍左右'}},
  {v:'first',t:{en:'Mainly the first token; later tokens barely change',zh:'主要是首 token 变快，之后的 token 几乎不变'}},
  {v:'later',t:{en:'Mainly the later tokens; the first token barely changes',zh:'主要是之后的 token 变快，首 token 几乎不变'}},
  {v:'neither',t:{en:'Neither changes much',zh:'两个都变化不大'}}];
const XPRED_RIGHT='first';
const xpredT=v=>L(XPRED.find(o=>o.v===v).t);
const isX=()=>mode==='guided'&&STEPS[step]&&STEPS[step].id==='double';
const xOpen=()=>isX()&&!!xpred;                                   // the variants are shown only after the prediction
const xName=id=>L(xVar(id).t);
/* part 3 is prefill's: the decode half of the chart waits for part 4, which reveals the prediction's answer (R3.8) */
const decMasked=()=>mode==='guided'&&STEPS[step].id==='prefill';
/* before the prediction, the prompt length is fixed at the mission's 2K (R3.8) */
const preLock=()=>mode==='guided'&&!pred&&step<=1;
const nCtl=(id,inline)=>A.ui.range({id,key:'N',label:tx('Prompt length','prompt 长度'),min:0,max:N_STOPS.length-1,value:Math.max(0,N_STOPS.indexOf(S.N)),out:tx(`${tokK(S.N)} tokens`,`${tokK(S.N)} 个 token`),inline});
const mCtl=(id,inline)=>A.ui.range({id,key:'M',label:tx('Answer length','回答长度'),min:0,max:M_STOPS.length-1,value:Math.max(0,M_STOPS.indexOf(S.M)),out:tx(`${fint(S.M)} tokens`,`${fint(S.M)} 个 token`),inline});
/* the batch size: a slider of whole requests with one-step buttons, since a single request can decide a goal */
function bCtl(id,key,max,label,inline){
  return `<div class="${inline?'inline-ctl':'ctl'} pd-bctl" id="${id}"><label for="${id}-in"><span>${esc(label)}</span> <output></output></label>`+
    `<div class="pd-brow"><button type="button" class="btn ghost sm pd-bstep" data-act="bminus" data-for="${key}" aria-label="${esc(tx('One fewer','少一个'))}">−</button>`+
    `<input type="range" id="${id}-in" data-key="${key}" min="1" max="${max}" step="1" value="${Math.min(S.B,max)}">`+
    `<button type="button" class="btn ghost sm pd-bstep" data-act="bplus" data-for="${key}" aria-label="${esc(tx('One more','多一个'))}">+</button></div>`+
    `<div class="ctl-note pd-bnote"></div></div>`;
}
const DEEP=()=>`<div class="pd-deep"><span class="lbl">${tx('Go deeper (optional)','深入（可选）')}</span><button type="button" class="btn ghost sm" data-act="roof" aria-pressed="${roofOpen}">${tx('The roofline','Roofline 图')}${roofSeen?' <span class="pd-seen">✓</span>':''}</button></div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'tl',
  lead:()=>{const b=B0(); return tx(`First token after ${fPre(b.ttft)}, then one every ${fDec(b.tpot)}: why so different?`,`首 token 要等 ${fPre(b.ttft)}，之后每 ${fDec(b.tpot)} 一个：为什么差这么多？`);},
  enter:()=>({...BASE}),
  body:()=>{const b=B0();
   return tx(`<p>${MODEL} runs on one ${GPUN} in BF16. It gets a prompt of 2,048 tokens (2K) and writes a 300-token answer: the first token takes a while, then the rest come quickly (${J('tl','the timeline in the chart')}).</p><p>Generation has two phases. ${DEF('Prefill',TIP.prefill())} takes in the whole prompt in one step and ends with the first token: that wait is the ${DEF('time to first token (TTFT)',TIP.ttft())}. ${DEF('Decode',TIP.decode())} then adds one token per step: the ${DEF('time per output token (TPOT)',TIP.tpot())}.</p><p>So prefill handles 2,048 tokens in ${fPre(b.ttft)}, while decode needs ${fDec(b.tpot)} for each single token. Why?</p>`,
             `<p>一张 ${GPUN} 用 BF16 跑 ${MODEL}。它收到一个 2,048 个 token（2K）的 prompt，要写一个 300 个 token 的回答：第一个 token 要等一会儿才出现，之后的 token 一个接一个很快就出来了（见${J('tl','图里的时间线')}）。</p><p>生成分两个阶段。${DEF('prefill',TIP.prefill())} 在一步里读入整个 prompt，以第一个 token 结束：这段等待就是${DEF('首 token 延迟（TTFT）',TIP.ttft())}。之后是 ${DEF('decode',TIP.decode())}：每一步多写一个 token，每步的耗时就是${DEF('每个输出 token 的耗时（TPOT）',TIP.tpot())}。</p><p>也就是说，prefill 用 ${fPre(b.ttft)} 处理了 2,048 个 token，decode 却要 ${fDec(b.tpot)} 才写出一个。为什么？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'tl',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: the prompt grows 4 times. What happens to each time?','先猜：prompt 变成 4 倍长，两个时间各会怎样？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>The prompt grows from 2K to 8K tokens, 4 times as long. What happens to the time to first token and to the time per output token?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 4 shows the result.</p>':''}`,
             `<p>prompt 从 2K 个 token 变成 8K 个，长到原来的 4 倍。首 token 延迟和每个输出 token 的耗时会怎样？</p>${ch}${pred?'<p class="muted small">已记下。第 4 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'prefill',title:{en:'Prefill: the whole prompt at once',zh:'prefill：一次算完整个 prompt'},focus:'pre',
  lead:()=>tx('Drag the prompt length: arithmetic sets the time to first token.','拖动 prompt 长度：计算决定首 token 延迟。'),
  enter:()=>({...BASE}),expect:()=>S.N!==2048,showMe:{N:8192},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>A GPU reads memory and does arithmetic at the same time, so ${DEF('a step takes about as long as the slower of the two',TIP.overlap())}; ${J('lim','the chart')} draws both for each phase.</p><p>Prefill ${DEF('reads the weights',TIP.wread())} once and multiplies every prompt token by them, about 14 ${DEF('GFLOP',TIP.flop())} per token. Drag the prompt length:</p>`,
              `<p>GPU 读写显存和做计算是同时进行的，所以${DEF('一步的耗时约等于两者中较慢的那个',TIP.overlap())}；${J('lim','图里')}把每个阶段的这两项都画了出来。</p><p>prefill 把${DEF('权重读一遍',TIP.wread())}，再让每个 prompt token 都和它们相乘，每个 token 约 14 ${DEF('GFLOP',TIP.flop())}。拖动 prompt 长度：</p>`)+nCtl('pd-ic-N',true),
  obs:()=>obsPrefill()},
 {id:'decode',title:{en:'Decode: one token per step',zh:'decode：每步一个 token'},focus:'dec',
  lead:()=>tx('Each step reads all the weights for one token: memory sets the time.','每一步为了一个 token 把全部权重读一遍：读写显存决定耗时。'),
  enter:()=>({B:1,M:300}),expect:()=>S.N!==2048,always:true,showMe:{N:8192},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Each ${DEF('decode step',TIP.dstep())} computes only the newest token, yet it still reads all ${gbD(W_READ)} of weights and the KV cache of its ${DEF('context',TIP.ctx())}: the prompt plus the tokens written so far. One token’s arithmetic is tiny next to those reads.</p><p>Drag the prompt length again:</p>`,
              `<p>每个 ${DEF('decode 步',TIP.dstep())}只算最新的一个 token，却仍要把全部 ${gbD(W_READ)} 的权重读一遍，还要读它的${DEF('上下文',TIP.ctx())}的 KV cache，也就是 prompt 加上已经写出的 token。和这些读取相比，一个 token 的计算量微不足道。</p><p>再拖一次 prompt 长度：</p>`)+nCtl('pd-ic-N',true),
  obs:()=>obsDecode()},
 {id:'batch',title:{en:'Batch the decode',zh:'把 decode 攒成 batch'},focus:'dec',
  lead:()=>tx('Many requests share each step’s read of the weights.','多个请求共用每一步读的那一遍权重。'),
  enter:()=>({...BASE}),expect:()=>S.B>1,showMe:{B:64},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>A server decodes many requests at once: one decode step computes the newest token of each of B requests, a ${DEF('batch',TIP.batch())}. The step still reads the weights once; each request adds only its own KV cache. Drag the number of requests:</p>`,
              `<p>服务端会同时 decode 很多个请求：一个 decode 步算出 B 个请求各自的最新 token，这 B 个请求就是一个 ${DEF('batch',TIP.batch())}。这一步仍然只读一遍权重，每个请求只多加上它自己的 KV cache。拖动请求数：</p>`)+
    bCtl('pd-ic-B','B',memLimit(S.N),tx('Requests decoding together','一起 decode 的请求数'),true)+DEEP(),
  obs:()=>obsBatch()},
 {id:'double',title:{en:'Experiment: twice the compute, or twice the bandwidth?',zh:'实验：算力翻倍，还是带宽翻倍？'},focus:'xa',always:true,
  lead:()=>tx('Doubling a GPU’s arithmetic or its bandwidth: which part of the answer gets faster?','把 GPU 的算力或带宽翻倍：回答的哪一段会变快？'),
  enter:()=>{xv='orig'; xl='chat'; const l=xLoad(xl); return {N:l.N,M:l.M,B:1};},
  gate:()=>!!xpred,expect:()=>xv!=='orig',showMe:()=>setX({v:'c2'}),label:{en:'Result',zh:'结果'},
  body:()=>xBody(),
  obs:()=>xObs()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:'',
  lead:()=>tx('Two phases, two limits, and why servers batch decode.','两个阶段，两种限制，以及服务端为什么把 decode 攒成 batch。'),
  enter:()=>({...BASE}),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);

/* part 3: the prefill, from the learner's own prompt length. Explanations say the trend and the reason in words; the exact times are on
   the metric cards, the two bars and Show the math (R6.20) */
function obsPrefill(){
  const r=R(), b=B0(), q=preMs(r.ttft)/preMs(b.ttft), p=r.N/2048, rel=q/p;
  if(r.N===2048) return tx(`<p><b>Prompt of 2K tokens: the first token comes after about ${fR(r.ttft)}.</b> Drag the prompt length to see how prefill grows.</p>`,
                           `<p><b>prompt 为 2K 个 token：约 ${fR(r.ttft)} 后出现第一个 token。</b>拖动 prompt 长度，看 prefill 怎么变。</p>`);
  const long=r.N>=8192;
  /* the trend against the mission's 2K: in step with the prompt (4K: 2.1 times the time for twice the prompt), faster for long prompts */
  const trend=p<1?tx('the time shrinks about in step with the prompt','时间和 prompt 差不多同比例缩短')
    :(rel<1.08?tx('the time grows about in step with the prompt','时间和 prompt 差不多同比例增长')
      :(rel<1.4?tx('the time grows a little faster than the prompt','时间比 prompt 涨得还快一点'):tx('the time grows clearly faster than the prompt','时间比 prompt 涨得明显更快')));
  return tx(`<p><b>Prompt of ${tokK(r.N)} tokens: the first token comes after about ${fR(r.ttft)} instead of about ${fR(b.ttft)} at 2K, so ${trend}.</b></p><p>Why: arithmetic sets prefill’s time. Every prompt token is multiplied by all the weights, which are read only once, so the arithmetic grows with the prompt and stays longer than the reads (the two bars).${long?' Long prompts also cost more per token, since each token attends to every earlier one.':''}</p>`,
            `<p><b>prompt 为 ${tokK(r.N)} 个 token：约 ${fR(r.ttft)} 后才出现第一个 token，2K 时约 ${fR(b.ttft)}，${trend}。</b></p><p>原因：prefill 的耗时由计算决定。每个 prompt token 都要和全部权重相乘，而权重只读一遍，所以计算的时间随 prompt 增长，而且一直比读写显存的时间长（图里的两根条）。${long?'prompt 越长，每个 token 要做 attention 的前文就越多，所以平均每个 token 也更贵。':''}</p>`);
}
/* the verdict on the prediction, which is about 2K → 8K; the numbers are on the stage and the metric cards */
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right</span>: that is what happens from 2K to 8K tokens.</p>`,`<p><span class="okc">你的预测“${pt}”是对的</span>：从 2K 到 8K 个 token 正是这样。</p>`)
    :tx(`<p>You predicted “${pt}”. From 2K to 8K tokens, the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。从 2K 到 8K 个 token，正确答案是“${rt}”。</p>`);
}
/* part 4: the decode step, then the verdict, then why (R3.10: result, verdict, reason) */
function obsDecode(){
  const r=R(), b=B0();
  if(r.N===2048) return tx(`<p><b>Prompt of 2K tokens: each later token takes about ${fR(r.tpot)}.</b> Drag the prompt length to 8K to check your prediction.</p>`,
                           `<p><b>prompt 为 2K 个 token：之后每个 token 约要 ${fR(r.tpot)}。</b>把 prompt 长度拖到 8K，核对你的预测。</p>`);
  const d=decMs(r.tpot)-decMs(b.tpot), rel=Math.abs(d)/decMs(b.tpot), up=r.N>2048;
  /* how much a later token changes, in words: 8K is 5% more, 32K 26% (R6.20) */
  const chg=rel<0.1?(d>=0?tx('barely more','几乎没变'):tx('barely less','几乎没变')):(rel<0.35?tx('only a little more','只多了一点'):tx('more','多了一些'));
  const kvSmall=KV_TOK*r.N/W_READ<0.15;
  const lead=tx(`<p><b>Prompt of ${tokK(r.N)} tokens: each later token takes about ${fR(r.tpot)} instead of about ${fR(b.tpot)} at 2K, ${chg}, while the first token ${up?'grows':'shrinks'} with the prompt.</b></p>`,
                `<p><b>prompt 为 ${tokK(r.N)} 个 token：之后每个 token 约 ${fR(r.tpot)}，2K 时约 ${fR(b.tpot)}，${chg}；首 token 却随 prompt 一起变${up?'长':'短'}了。</b></p>`);
  const why=tx(`<p>Why: memory sets a decode step’s time. To compute one token, the step reads all the weights, which takes far longer than the token’s arithmetic. A longer prompt only adds its KV cache to that read, ${kvSmall?'small next to the weights':'still much smaller than the weights'} (the KV cache part of the memory bar).</p>`,
               `<p>原因：decode 步的耗时由读写显存决定。为了算一个 token，这一步要把全部权重读一遍，这比这个 token 的计算慢得多。prompt 变长，只是给这次读取加上它的 KV cache，${kvSmall?'和权重比起来很小':'仍比权重小得多'}（读写显存条里 KV cache 的那一段）。</p>`);
  return lead+verdict()+why;
}
/* part 5: the batch, from the learner's own batch size */
function obsBatch(){
  const r=R(), one=compute({...S,B:1});
  if(r.B===1) return tx(`<p><b>One request: each decode step takes about ${fR(r.tpot)} and writes just one token.</b> Drag the number of requests up.</p>`,
                        `<p><b>一个请求：每个 decode 步约 ${fR(r.tpot)}，只写出一个 token。</b>把请求数往上拖。</p>`);
  const k=Math.round(r.thr)/Math.round(one.thr), kt2=decMs(r.tpot)/decMs(one.tpot);
  const full=r.B>=r.lim;
  return tx(`<p><b>${reqN(r.B)} at ${tokK(r.N)} tokens each: the GPU writes about ${ratioS(k)} times as many tokens per second as for one request.</b> ${waitS(kt2)}</p><p>Why: the step reads the weights once for all ${fint(r.B)} requests, so they share that read. But each request adds a read of its own KV cache, so the step grows with every request, and the gain stays below ${fint(r.B)} times.${full?` That is the most that fit: at ${tokK(r.N)} tokens each, GPU memory holds no more KV caches.`:''}</p>`,
            `<p><b>${reqN(r.B)}，每个 ${tokK(r.N)} 个 token：GPU 每秒写出的 token 约是一个请求时的 ${ratioS(k)} 倍。</b>${waitS(kt2)}</p><p>原因：这一步只为这 ${fint(r.B)} 个请求读一遍权重，大家共用这次读取。但每个请求都要多读一份自己的 KV cache，每多一个请求，这一步就长一点，所以涨不到 ${fint(r.B)} 倍。${full?`这已经是能放下的最多请求：每个 ${tokK(r.N)} 个 token 时，显存放不下更多的 KV cache 了。`:''}</p>`);
}
/* part 6, the experiment. Times in one unit per load (R6.12, review A4): the chat's first token in ms and its whole answer in s; the long
   document's first token and whole answer both in s, 3 significant digits (0.747 s next to 1.49 s, never 746.7 ms next to 1.49 s) */
const xs3=sec=>sec<0.9995?fixd(sec,3)+NS+'s':(sec<9.995?fixd(sec,2)+NS+'s':fixd(sec,1)+NS+'s');
const xFt=sec=>xl==='doc'?xs3(sec):fPre(sec);                  // the first token (the prefill)
const xWa=sec=>xl==='doc'?xs3(sec):fTot(sec);                  // the whole answer
const xRd=sec=>xl==='doc'?(sec>=0.995?fixd(sec,1):fixd(sec,2))+NS+'s':fR(sec);   // rounded, in prose (R6.20): 1.7 s, 0.93 s
/* how much faster a time got, from the values as shown (R6.6): q = the original's time ÷ the variant's; each option by its own ratio (R6.10b) */
const shownS=s=>toBase(numOf(s),unitOf(s));                        // a shown time (“2.31 s”, “0.929 s”, “30.0 ms”) in seconds
const fasterW=q=>q>=1.98?tx('2× faster','快一倍'):(q>=1.9?tx('about 2× faster','快了约一倍'):(q>=1.3?tx(`${fixd(q,1)}× faster`,`快到 ${fixd(q,1)} 倍`):(q>=1.03?tx('a little faster','快了一点'):(q>1.0005?tx('about the same','几乎不变'):tx('the same','不变')))));
const asFast=q=>q>=1.98&&q<2.02?tx('twice as fast','快了一倍'):tx(`${ratioS(q)} times as fast`,`快到 ${ratioS(q)} 倍`);
const xQ=(vid,l)=>{const o=xRun('orig',l||xl), r=xRun(vid,l||xl);
  return {o,r,q1:shownS(xFt(o.ttft))/shownS(xFt(r.ttft)),q2:decMs(o.tpot)/decMs(r.tpot),qe:shownS(xWa(o.e2e))/shownS(xWa(r.e2e))};};
/* a GPU option: its whole answer on its title row, right-aligned under the header “whole answer”, so the three options read as a
   comparison table (R3.6); below at full width, the initial setup, or what the doubling did (the phase that gained, then the whole answer) */
const xRes=vid=>xpred?xWa(xRun(vid,xl).e2e):'';
function xDesc(vid){
  if(vid==='orig') return tx('initial setup','初始设置');
  const {q1,q2,qe}=xQ(vid);
  const a=q1>=q2?tx(`first token ${fasterW(q1)}`,`首 token ${fasterW(q1)}`):tx(`later tokens ${fasterW(q2)}`,`之后的 token ${fasterW(q2)}`);
  return a+tx(`, whole answer ${fasterW(qe)}`,`，整个回答${fasterW(qe)}`);
}
const xLvTip=vid=>vid==='orig'?tx(`The ${GPUN} with its real numbers, as in the other parts.`,`真实参数的 ${GPUN}，和其他部分一样。`)
  :tx(`An idealised ${GPUN} with ${vid==='c2'?'twice the FLOP/s and the same memory bandwidth':'twice the memory bandwidth and the same FLOP/s'}; everything else stays the same.`,`理想化的 ${GPUN}：${vid==='c2'?'FLOP/s 翻倍，显存带宽不变':'显存带宽翻倍，FLOP/s 不变'}，其余都不变。`);
const xLevers=()=>`<div class="pd-xlv" id="pd-ic-xv"><div class="pd-xlv-h" aria-hidden="true"><span>${tx('GPU','GPU')}</span><span>${tx('whole answer','整个回答')}</span></div>`+
  A.ui.levers({aria:tx('The GPU, pick one','GPU，选一张'),value:xv,options:XV.map(v=>({v:v.id,title:L(v.t),desc:xDesc(v.id),result:esc(xRes(v.id))}))})
    .replace(/data-lever="(\w+)"/g,(m,v)=>`${m} data-tipx="${esc(xLvTip(v))}"`)+`</div>`;
/* the load: the one request of this part, in words with its numbers (the switch's note) and short (the mini result bar) */
const xLoadD=id=>{const l=xLoad(id);
  return id==='chat'?tx(`${an(tokK(l.N))==='an'?'An':'A'} ${tokK(l.N)}-token prompt and a ${fint(l.M)}-token answer, one user.`,`prompt ${tokK(l.N)} 个 token，回答 ${fint(l.M)} 个 token，一个用户。`)
    :tx(`A question about a long document: ${an(tokK(l.N))} ${tokK(l.N)}-token prompt and a ${fint(l.M)}-token answer, one user.`,`针对一篇长文档提问：prompt ${tokK(l.N)} 个 token，回答 ${fint(l.M)} 个 token，一个用户。`);};
const xLoadS=id=>{const l=xLoad(id); return tx(`${tokK(l.N)} prompt, ${fint(l.M)}-token answer`,`prompt ${tokK(l.N)} 个 token、回答 ${fint(l.M)} 个 token`);};
const xLoadCtl=()=>A.ui.seg({id:'pd-ic-xl',key:'xl',label:tx('Load','负载'),options:XL.map(l=>[l.id,L(l.t)]),value:xl,inline:true,note:`<span class="pd-xl-note">${esc(xLoadD(xl))}</span>`});
function xBody(){
  if(!xpred){
    const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${XPRED.map(o=>`<button type="button" class="choice" data-xpred="${o.v}" aria-pressed="false">${esc(L(o.t))}</button>`).join('')}</div>`;
    const lc=xLoad('chat');
    return tx(`<p>Take the ${GPUN} and double one number, keeping everything else the same: how fast it does arithmetic (its FLOP/s), or how fast it reads GPU memory (its bandwidth). The request is one user’s chat: ${an(tokK(lc.N))} ${tokK(lc.N)}-token prompt and a ${fint(lc.M)}-token answer.</p><p>Twice the arithmetic per second, same memory bandwidth. What gets faster?</p>${ch}`,
              `<p>把 ${GPUN} 的一项指标翻倍，其余都不变：它每秒能做多少计算（FLOP/s），或者它读显存有多快（带宽）。请求是一个用户的一次聊天，prompt 有 ${tokK(lc.N)} 个 token，回答 ${fint(lc.M)} 个 token。</p><p>每秒的计算量翻倍，显存带宽不变。什么会变快？</p>${ch}`);}
  /* after the pick: short, so the card with its result and Next fits on a 1440 × 1000 screen (R3.10, review A8); single choice said (R2.5) */
  return tx(`<p class="small muted">Your prediction: “${esc(xpredT(xpred))}”.</p><p class="small">Pick a GPU and see which gets faster: the first token or the later tokens. Then try the other load.</p>`,
            `<p class="small muted">你的预测：“${esc(xpredT(xpred))}”。</p><p class="small">选一张 GPU，看看首 token 和之后的 token 哪个变快了。然后换另一个负载试试。</p>`)+xLevers()+xLoadCtl();
}
/* the verdict on part 6's prediction, shown with compute × 2 (either load: its first token gains, its later tokens do not) */
function xVerdict(){
  const pt=esc(xpredT(xpred)), rt=esc(xpredT(XPRED_RIGHT));
  return xpred===XPRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>Not quite: with twice the arithmetic, the right answer is “${rt}”.</p>`,`<p>不太对。算力翻倍时，正确答案是“${rt}”。</p>`);   // the card quotes the prediction just above (R3.8; shorter card, review A8)
}
/* the result: which phase gained, then (with compute × 2) the verdict, then why the other phase did not (R3.10). Trend and principle in
   words, with one rounded pair that the stage shows (the whole answer); the exact times are on the timelines, the cards and Show the math (R6.20) */
function xObs(){
  if(!xpred) return '';
  const doc=xl==='doc';
  if(xv==='orig'){const o=xRun('orig',xl), sh=o.ttft/o.e2e;
    const mix=sh<0.1?tx('almost all of it decode','几乎全是 decode'):(sh>0.6?tx('most of it the prefill','大部分是 prefill'):tx('part prefill, part decode','prefill 和 decode 各占一部分'));
    return tx(`<p><b>The original ${GPUN} takes about ${xRd(o.e2e)} for the whole answer, ${mix}.</b> ${xSeenC2?'Pick an option to compare.':'Pick “Compute × 2” to check your prediction.'}</p>`,
              `<p><b>原版 ${GPUN} 写完整个回答约要 ${xRd(o.e2e)}，${mix}。</b>${xSeenC2?'选一个选项来比较。':'选“算力 × 2”，核对你的预测。'}</p>`);}
  const {o,r,q1,q2,qe}=xQ(xv);
  /* the whole answer, as a rounded pair (or one value when the rounding is the same) */
  const tot=qe<1.03?(xRd(o.e2e)===xRd(r.e2e)?tx(`but the whole answer still takes about ${xRd(r.e2e)}`,`整个回答却仍要约 ${xRd(r.e2e)}`):tx(`but the whole answer barely changes, about ${xRd(r.e2e)} instead of about ${xRd(o.e2e)}`,`整个回答却几乎不变，约 ${xRd(r.e2e)}，原来约 ${xRd(o.e2e)}`))
    :(qe<1.3?tx(`but the whole answer drops only from about ${xRd(o.e2e)} to about ${xRd(r.e2e)}`,`整个回答却只从约 ${xRd(o.e2e)} 降到约 ${xRd(r.e2e)}`)
      :tx(`and the whole answer drops from about ${xRd(o.e2e)} to about ${xRd(r.e2e)}`,`整个回答从约 ${xRd(o.e2e)} 降到约 ${xRd(r.e2e)}`));
  const twist=tx('Which doubling helps the whole answer more depends on whether a request is mostly prompt or mostly answer.','哪种翻倍对整个回答更有用，要看请求主要是 prompt 还是回答。');
  if(xv==='c2'){
    const res=tx(`<p><b>With twice the arithmetic, the first token comes ${asFast(q1)}, ${tot}.</b></p>`,`<p><b>算力翻倍，首 token ${asFast(q1)}，${tot}。</b></p>`);
    /* two short sentences (review A8); the step's time stays, not everything in it (review A9: its arithmetic bar does shrink) */
    const why=doc?tx(`<p>Why: this request is mostly prompt, and arithmetic sets the prefill’s time, so twice the arithmetic ${qe>=1.7?'nearly halves':'shortens'} the whole answer. ${twist}</p>`,
                     `<p>原因：这个请求主要是 prompt，而 prefill 的耗时由计算决定，算力翻倍，整个回答就${qe>=1.7?'几乎减半':'变短了'}。${twist}</p>`)
      :tx(`<p>Why: prefill does thousands of tokens’ arithmetic for each read of the weights, so arithmetic sets its time, and it ${r.pre.cb?'halves':'shrinks'}. A decode step reads every weight for one token, so memory traffic sets its time, which stays the same, and a chat is almost all decode.</p>`,
          `<p>原因：prefill 每读一遍权重，就要为几千个 token 做计算，计算决定它的耗时，所以它${r.pre.cb?'减半':'变短'}。decode 步为了一个 token 把全部权重读一遍，读写显存决定它的耗时，这个耗时不变，而聊天几乎全是 decode。</p>`);
    return res+xVerdict()+why;
  }
  const moved=xFt(o.ttft)===xFt(r.ttft)?tx('The first token does not move.','首 token 没有变。'):tx('The first token barely moves.','首 token 几乎没变。');
  const res=tx(`<p><b>With twice the bandwidth, every later token comes ${asFast(q2)}, ${tot}.</b> ${moved}</p>`,`<p><b>带宽翻倍，之后每个 token 都${asFast(q2)}，${tot}。</b>${moved}</p>`);
  if(doc) return res+tx(`<p>Why: the answer is short, so decode is only a small part of this request; most of it is the prefill, whose arithmetic twice the bandwidth leaves as it was. ${twist}</p>`,
                        `<p>原因：回答很短，decode 只占这个请求的一小部分；大部分时间是 prefill，而带宽翻倍不改变它的计算。${twist}</p>`);
  return res+tx(`<p>Why: a decode step reads every weight for one token, so memory traffic sets its time, and twice the bandwidth halves every step. Prefill’s time is its arithmetic, which twice the bandwidth leaves as it was.</p><p>A chat is mostly answer, so the bandwidth sets most of its time. Now switch the load to the long document, which is mostly prompt.</p>`,
                `<p>原因：decode 步为了一个 token 把全部权重读一遍，读写显存决定它的耗时，所以带宽翻倍，每一步都减半。prefill 的耗时是它的计算，带宽翻倍不改变它。</p><p>聊天主要是回答，所以它的大部分时间由带宽决定。现在把负载换成长文档，它主要是 prompt。</p>`);
}
/* part 6's state change: the variant or the load (the load sets the prompt and the answer of the one request) */
function setX(p){
  if(p.v&&XV.some(v=>v.id===p.v)) xv=p.v;
  if(p.l&&XL.some(l=>l.id===p.l)) xl=p.l;
  if(xv==='c2'&&xpred) xSeenC2=true;
  const l=xLoad(xl); S={...S,N:l.N,M:l.M,B:1};
  render();
}
function remember(){
  const a=compute({N:2048,M:300,B:1}), l=compute({N:32768,M:300,B:1}), b64=compute({N:2048,M:300,B:64});
  const pre2=stepOn({...H100,peak:H100.peak*2},{tok:2048,logits:1,attn:attn(0,2048),kv:2048}), dec2=stepOn({...H100,peak:H100.peak*2},{tok:1,logits:1,attn:attn(2048,2049),kv:2048});
  const kL=preMs(l.ttft)/preMs(a.ttft), k64=decMs(b64.tpot)/decMs(a.tpot);
  /* twice the FLOP/s halves the 2K prefill and leaves a decode step as it was: the words follow the engine (Show the math has the numbers) */
  const halves=Math.abs(preMs(pre2.t)/preMs(a.ttft)-0.5)<0.02, same=decMs(dec2.t)===decMs(a.tpot);
  return tx(`<ul><li><b>Two phases.</b> Prefill takes in the whole prompt in one step and ends with the first token (TTFT); decode then adds one token per step (TPOT).</li><li><b>Two limits.</b> A step takes the longer of its arithmetic and its memory traffic. Prefill multiplies every prompt token by weights it reads once, so arithmetic sets its time, and it grows with the prompt. A decode step reads all the weights and the KV cache for one token, so memory sets its time, and a longer prompt adds only a little to it. So a faster GPU helps where its bigger number is the limit: more bandwidth for answer-heavy chats, more FLOP/s for long prompts.</li><li><b>Batching decode shares the weight read.</b> So the more requests decode together, the more tokens per second the GPU writes, and each user waits only a little longer per token. How many can decode together depends on how many KV caches fit in GPU memory.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “A GPU with more TFLOP/s makes every token faster.” Memory sets a decode step’s time. Twice the FLOP/s speeds up only the prefill: the wait for the first token ${halves?'halves':'gets shorter'}, but every later token stays ${same?'exactly as slow':'about as slow'}.</p><p>✗ “A long prompt slows down every token.” Mostly the first: at 32K tokens the first token takes about ${ratioS(kL)} times as long as at 2K, while each later token takes only a little longer.</p><p>✗ “Batching makes each user B times slower.” The weight read is shared: with 64 requests at 2K tokens each, every request waits ${k64<1.5?'only a little longer per token than alone':(k64<2.5?'only about twice as long per token as alone':`only about ${ratioS(k64)} times as long per token as alone`)}, not 64 times as long.</p></div>${DEEP()}${A.shell.toScenarios()}`,
    `<ul><li><b>两个阶段。</b>prefill 在一步里读入整个 prompt，以第一个 token 结束（TTFT）；之后 decode 每一步多写一个 token（TPOT）。</li><li><b>两种限制。</b>一步的耗时取计算和读写显存中较长的那个。prefill 让每个 prompt token 都和只读一遍的权重相乘，所以计算决定它的耗时，它随 prompt 增长。decode 步为了一个 token 把全部权重和 KV cache 读一遍，所以读写显存决定它的耗时，prompt 变长，它只多一点。所以，更快的 GPU 要看它加强的是哪一项：回答长的聊天靠带宽，prompt 长的请求靠算力。</li><li><b>把 decode 攒成 batch，大家共用读权重的那一遍。</b>所以一起 decode 的请求越多，GPU 每秒写出的 token 就越多，每个用户每个 token 只多等一点。能一起 decode 多少个请求，要看显存放得下多少份 KV cache。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “GPU 的 TFLOP/s 越高，每个 token 都越快。”decode 步的耗时由读写显存决定。FLOP/s 翻倍只让 prefill 变快，首 token 的等待${halves?'减半':'变短'}，之后的每个 token 却${same?'一点也不会变快':'几乎不会变快'}。</p><p>✗ “prompt 越长，每个 token 都越慢。”主要慢的是第一个：32K 个 token 时，首 token 约是 2K 时的 ${ratioS(kL)} 倍，之后每个 token 只慢一点。</p><p>✗ “攒 B 个请求，每个用户就慢 B 倍。”读权重的那一遍是共用的：64 个请求、每个 2K 个 token 时，每个请求每个 token 等的时间${k64<1.5?'只比单独跑时长一点':(k64<2.5?'只是单独跑时的两倍左右':`只是单独跑时的 ${ratioS(k64)} 倍左右`)}，而不是 64 倍。</p></div>${DEEP()}${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real workloads, one plain phrase each (R2.1)
   ===================================================================== */
const PRESETS=[
 {id:'chat',t:{en:'Chat reply',zh:'聊天回复'},d:{en:'a 1K-token prompt and a 300-token answer, one user',zh:'prompt 1K 个 token，回答 300 个 token，一个用户'},s:{N:1024,M:300,B:1}},
 {id:'doc',t:{en:'Long document summary',zh:'长文档摘要'},d:{en:'a 32K-token document and a 300-token summary, one user',zh:'32K 个 token 的文档，摘要 300 个 token，一个用户'},s:{N:32768,M:300,B:1}},
 {id:'code',t:{en:'Code completion',zh:'代码补全'},d:{en:'4K tokens of code and a 20-token completion, one user',zh:'4K 个 token 的代码，补全 20 个 token，一个用户'},s:{N:4096,M:20,B:1}},
 {id:'busy',t:{en:'Busy chat server',zh:'繁忙的聊天服务'},d:{en:'64 users decoding at once, 2K tokens of context each',zh:'64 个用户同时 decode，每人 2K 个 token 的上下文'},s:{N:2048,M:300,B:64}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.N===S.N&&p.s.M===S.M&&p.s.B===S.B);
function scenHTML(i){
  const p=PRESETS[i], r=compute(p.s), share=Math.round(r.ttft/r.e2e*100), name=esc(L(p.t));
  const head=tx(`<p><b>${name}, by this lab’s estimate: prefill is ${share}% of the wait for the whole ${fint(r.M)}-token answer.</b></p>`,
                `<p><b>${name}，按这个 Lab 的估计：整个 ${fint(r.M)} 个 token 的回答要等的时间里，prefill 占 ${share}%。</b></p>`);
  /* why, in words: the times themselves are on the metric cards and the timeline (R6.20) */
  let why='';
  if(p.id==='chat') why=tx(`<p>A 1K-token prompt is quick to compute, but every token of the answer needs its own decode step, which reads all the weights for that one token: decode is almost all of the wait.</p>`,
                         `<p>1K 个 token 的 prompt 算得很快；回答的每个 token 却都要一个自己的 decode 步，为这一个 token 把全部权重读一遍：几乎全部等待都花在 decode 上。</p>`);
  if(p.id==='doc') why=tx(`<p>Prefill multiplies every token of the document by the weights, and each also attends to every earlier one: about ${fR(r.ttft)} before anything appears. Each later token is still quick, since its step only reads the weights and the document’s KV cache.</p>`,
                          `<p>prefill 要让文档的每个 token 都和权重相乘，每个 token 还要对前面所有 token 做 attention：约 ${fR(r.ttft)} 之后才有第一个 token。之后每个 token 仍然很快，因为这一步只是读权重，再读文档的 KV cache。</p>`);
  if(p.id==='code') why=tx(`<p>A short answer after a long prompt: only ${fint(r.M-1)} decode steps follow the prefill, so the prefill is almost half of the wait. Code completion servers watch the time to first token closely.</p>`,
                           `<p>prompt 长、回答短：prefill 之后只有 ${fint(r.M-1)} 个 decode 步，所以 prefill 几乎占了一半的等待。代码补全服务特别在意首 token 延迟。</p>`);
  if(p.id==='busy'){const one=compute({...p.s,B:1}), k=Math.round(r.thr)/Math.round(one.thr), kt=decMs(r.tpot)/decMs(one.tpot);
    why=tx(`<p>The ${fint(r.B)} users share each decode step’s read of the weights. ${waitS(kt,true)} Together they get about ${ratioS(k)} times as many tokens per second as one user. The first token takes as long as for one user, since this lab runs each prefill alone.</p>`,
           `<p>这 ${fint(r.B)} 个用户共用每个 decode 步读权重的那一遍。${waitS(kt,true)}合起来每秒拿到的 token 约是一个用户时的 ${ratioS(k)} 倍。首 token 和一个用户时一样久，因为这个 Lab 让每次 prefill 单独跑。</p>`);}
  return head+why;
}

/* =====================================================================
   challenge
   ===================================================================== */
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chPoint(S.B), best=chPoint(chBest()), nxt=chPoint(chBest()+1);
  let h;
  if(me.ok&&me.B===best.B) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${userN(me.B)}, each gets a token every ${fDec(me.t)}.</b> And that is the most: one more user pushes every token over 20 ms.</p>`,
                                `<p class="okline" style="margin-top:12px"><b>达成目标：${userN(me.B)}，每人每 ${fDec(me.t)} 拿到一个 token。</b>而且这已经是最多：再多一个用户，每个 token 就超过 20 ms。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${userN(me.B)}, each gets a token every ${fDec(me.t)}.</b> But more users fit: up to ${fint(best.B)} stay within 20 ms, and every user added raises the tokens per second in all.</p>`,
                      `<p class="okline" style="margin-top:12px"><b>达成目标：${userN(me.B)}，每人每 ${fDec(me.t)} 拿到一个 token。</b>不过还能多服务几个：最多 ${fint(best.B)} 个用户时仍在 20 ms 以内，而且每多一个用户，合起来每秒的 token 就多一些。</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: with ${userN(me.B)}, each token takes ${fDec(me.t)}, over 20 ms.</b></p><p class="small">Hint: a decode step reads the weights once for everyone, and every user adds the same read of its own KV cache. How many users’ reads fit in what is left of 20 ms after the weights?</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：${userN(me.B)}时，每个 token 要 ${fDec(me.t)}，超过了 20 ms。</b></p><p class="small">提示：一个 decode 步为所有人只读一遍权重，每个用户再多读一份自己的 KV cache，每份一样大。20 ms 扣掉读权重的时间，还能放下几个用户的读取？</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const rows=CH.rows.slice(); if(!rows.includes(me.B)) rows.push(me.B); rows.sort((a,b)=>a-b);
  const pts=rows.map(chPoint), base=chPoint(CH.start);
  /* brief (the owner's review of batch 8): the chart keeps the user counts that meet the goal, the pick and the initial setup, and
     says how many it left out; the title and the aria name only what it shows */
  const shownPts=pts.filter(p=>p.ok||p.B===me.B||p.B===CH.start);
  h+=A.ui.cmpChart({title:tx(`Tokens per second each user gets, ${LL.name} on one H100 SXM with 4K tokens of context each (goal: at least 50)`,`一张 H100 SXM 跑 ${LL.name}、每人 4K 个 token 上下文时，每个用户每秒拿到的 token（目标：至少 50 个）`).replace(/Llama-3\.1-8B/g,MODEL),
    aria:tx('Tokens per second each user gets: ','每个用户每秒拿到的 token：')+shownPts.map(p=>`${userN(p.B)}${tx(': ','：')}${fTps(p.tps)}`).join(tx('; ','；')),
    brief:true,hiddenText:n=>tx(`The other ${n} user counts give each user fewer than 50 tokens per second and are not shown.`,`其余 ${n} 档用户数下，每人每秒不到 50 个 token，图里没有列出。`),
    rows:pts.map(p=>({label:tx(`${userN(p.B)}: ${fDec(p.t)} per token`,`${userN(p.B)}：每个 token ${fDec(p.t)}`),value:p.tps,text:fTps(p.tps),ok:p.ok,me:p.B===me.B,
      tags:(p.B===CH.start?[{text:tx('initial setup','初始设置')}]:[]).concat(p.B===me.B?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:50,goalLabel:tx('goal: 50','目标 50'),max:140,okText:tx('50 or more per second','每秒 50 个或更多'),noText:tx('fewer than 50','不到 50 个'),goalText:tx('goal: 50 tokens per second','目标：每秒 50 个 token')});
  /* baseline → effect → reason (R3.10), in words: the times are on the chart, the cards and Show the math (R6.20); the first user count
     that misses is not on the brief chart, so its time is on hover */
  const kThr=Math.round(best.thr)/Math.round(base.thr);
  h+=tx(`<p class="small">Initial setup, 1 user: each token takes far less than 20 ms, but each step writes just one token. Every user added lengthens each step by the same small amount, the read of one more KV cache: ${userN(best.B)} still fit within 20 ms, ${DEF(fint(nxt.B)+' do not',`With ${userN(nxt.B)} each token takes ${fDec(nxt.t)}.`)}.</p>`,
        `<p class="small">初始设置是 1 个用户：每个 token 远不到 20 ms，但每一步只写出一个 token。每多一个用户，每一步就多读一份 KV cache，长出一小段，而且每次都一样长：${userN(best.B)}还在 20 ms 以内，${DEF(fint(nxt.B)+' 个就超了',`${userN(nxt.B)}时，每个 token 要 ${fDec(nxt.t)}。`)}。</p>`);
  h+=tx(`<p class="small">Why: every step reads the weights once, however many users it serves, so batching shares that read. With ${userN(best.B)} the GPU writes about ${ratioS(kThr)} times as many tokens per second as with one, and each user adds only the read of their own KV cache.</p>`,
        `<p class="small">原因：不管服务多少用户，每一步都只读一遍权重，batch 让大家共用这次读取。${userN(best.B)}时，GPU 每秒写出的 token 约是 1 个用户时的 ${ratioS(kThr)} 倍，而每个用户只多出读自己那份 KV cache 的时间。</p>`);
  const b70=chBestAt(0.7);
  h+=tx(`<p class="small muted">Note: the 60% of peak bandwidth is this lab’s assumption; if the kernels reached 70%, ${fint(b70)} users would fit in 20 ms. Before sizing a real server, measure the time per output token at the batch size you plan. Real servers also run new requests’ prefills on the same GPU between decode steps, which lengthens some of them; the ${labLink('continuous-batching')} lab shows how a server schedules the two.</p>`,
        `<p class="small muted">注意：峰值带宽的 60% 是这个 Lab 的假设；如果 kernel 能用到 70%，20 ms 内能服务 ${fint(b70)} 个用户。给真实服务定规模之前，要按计划的 batch 大小实测每个输出 token 的耗时。真实服务还会在 decode 步之间，在同一张 GPU 上跑新请求的 prefill，让其中一些步变长；怎样把两者一起调度，在 ${labLink('continuous-batching')} 这个 Lab 里讲。</p>`);
  return h;
}
/* the caveat's number: the most users within 20 ms if the kernels reached another share of the bandwidth */
function chBestAt(mbu){const g={peak:H100.peak,bw:H100.bw*mbu/EFF.mbu}; let best=0; for(let B=1;B<=chMax();B++){if(decode(B,CH.N,g).t<=CH.goal) best=B;} return best;}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===7?'Seven':fint(STEPS.length)} short parts. Predict first, then see why prefill is limited by arithmetic and decode by memory traffic, why servers decode many requests at once, and which half of an answer a faster GPU speeds up. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看为什么 prefill 受计算限制、decode 受读写显存限制，服务端为什么一次 decode 很多个请求，以及更快的 GPU 让回答的哪一段变快。第一次学从这里开始。`),
  sandbox:tx(`Four real workloads for ${MODEL} on one ${GPUN}, from a 20-token code completion to a server decoding 64 chats at once. Pick one to see where its time goes.`,
             `四个真实负载，都是一张 ${GPUN} 跑 ${MODEL}，从只补全 20 个 token 的代码补全，到同时 decode 64 个对话的服务端。选一个，看时间花在哪里。`),
  challenge:tx(`One ${GPUN} serves ${MODEL} chats with 4K tokens of context each, and every user must get at least 50 tokens per second. Change only the number of users, and serve as many as you can.`,
               `一张 ${GPUN} 跑 ${MODEL} 的对话，每个对话有 4K 个 token 的上下文，每个用户每秒至少要拿到 50 个 token。只改同时服务的用户数，尽量多服务一些用户。`)
});
const SRC={
 cfg:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 a100:'https://www.nvidia.com/en-us/data-center/a100/',
 dbx:'https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices',
 roof:'https://dl.acm.org/doi/10.1145/1498765.1498785',
 dist:'https://arxiv.org/abs/2401.09670',
 split:'https://arxiv.org/abs/2311.18677',
 vstats:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py',
 vlog:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py',
 vbench:'https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py',
 sglm:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py',
 sglb:'https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py'
};
/* the prefill-decode tile, as a link to the map where it lives */
const TILE_A=()=>`<a href="#map~inference" data-tipx="${esc(tx('Click to open the tree at Inference serving and see this tile.','点击回到知识树的“推理服务”这一枝，看这个方块。'))}">${esc(tx('“Why is the first token slow and the rest fast?”','“为什么首 token 要等很久，之后却一个接一个很快？”'))}</a>`;
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> every time on this page comes from one step model, the one the home-page tile ${TILE_A()} uses; ${J('math','“Show the math”')} works it out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>${MODEL} in BF16 on one ${GPUN}: 32 ${DEF('layers',kt('layer'))}, ${DEF('hidden size',kt('hidden'))} 4,096, 8 ${DEF('KV heads',KVH())} of 128 numbers, ${A.fmt.big(P_TOTAL)} ${DEF('parameters',kt('param'))}.</li>`+
 `<li>A step takes the longer of its arithmetic and its memory traffic, because the GPU does both at the same time. Arithmetic is 2 FLOPs per multiplied parameter per token, the ${DEF('LM head',kt('lmhead'))} once per new token, plus attention. Memory traffic is every parameter but the ${DEF('embedding table',kt('embedding'))}, read once per step however many tokens it computes, plus 128 KiB of KV cache per token: written by prefill, read by decode.</li>`+
 `<li>Prefill takes the whole prompt in one step, which also produces the first token. Decode serves B requests at the same context, one new token each per step.</li>`+
 `<li>The most requests that fit comes from the ${labLink('kv-cache')} lab’s model: the 80 GB read as 80 GiB, ${pct0(UTIL)} of it usable, ${OH} GiB of runtime overhead, ${gbS(W_MEM)} of weights.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>50% of the dense BF16 peak and 60% of the peak bandwidth are this lab’s assumptions (${J('eff','see Settings')}); real kernels reach different shares at different sizes. They set the times, not which limit sets each phase: for any shares from 20% to 100%, the ridge stays between ${fint(RIDGE_LO)} and ${fint(RIDGE_HI)} FLOPs per byte, while the prefill of 2K tokens does ${fint(prefill(2048).ai)} and one decode step ${fixd(decode(1,2048).ai,2)}.</li>`+
 `<li>No queue: the time to first token is the prefill alone. A real server also waits for a free place in the batch, and runs other requests’ prefills between decode steps. Both add to the times a user sees, but neither changes which limit sets a prefill or a decode step.</li>`+
 `<li>The requests in a batch all have the same context; with mixed contexts a step reads the sum of their KV caches instead. The memory traffic of activations and the time to launch kernels are left out; the 60% comes from measured step times, which include them.</li>`+
 `<li>The time per output token is the first decode step’s. The whole answer adds up every step as the context grows by one token, which adds only 128 KiB per token to read.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${MODEL}: ${cite(SRC.cfg,'config.json')}.</li>`+
 `<li>${GPUN}, from ${cite(SRC.h100,'NVIDIA’s product page')}: BF16 1,979 TFLOPS “with sparsity”, so 989.5 dense; 3.35 TB/s; 80 GB.</li>`+
 `<li>${cite(SRC.dbx,'Databricks (2023)')}: the time to first token and time per output token; “Generating the first token is typically compute-bound, while subsequent decoding is memory-bound”; 60% of peak bandwidth at batch size 1 on two H100 80GB GPUs.</li>`+
 `<li>${cite(SRC.roof,'Williams, Waterman and Patterson (2009)')}: the roofline and its ridge point.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-28. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上的每个时间都来自同一个耗时模型，也就是首页方块${TILE_A()}用的那一个；${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>一张 ${GPUN} 用 BF16 跑 ${MODEL}：32 ${DEF('层',kt('layer'))}，${DEF('隐藏维度',kt('hidden'))} 4,096，8 个 ${DEF('KV head',KVH())}，每个 128 维，一共 ${A.fmt.big(P_TOTAL)} 个${DEF('参数',kt('param'))}。</li>`+
 `<li>一步的耗时取计算和读写显存中较长的那个，因为 GPU 同时做这两件事。计算按每个参与乘法的参数每个 token 2 FLOP，${DEF('LM head',kt('lmhead'))} 每个新 token 算一次，再加 attention。读写显存算除 ${DEF('embedding 表',kt('embedding'))}外的全部参数，每一步读一遍，不管这一步算多少个 token；再加每个 token 128 KiB 的 KV cache：prefill 写，decode 读。</li>`+
 `<li>prefill 在一步里算完整个 prompt，同时产出首 token。decode 让 B 个请求在同样长的上下文上，每一步各多写一个 token。</li>`+
 `<li>最多能放下几个请求，用的是 ${labLink('kv-cache')}这个 Lab 的模型：80 GB 按 80 GiB 计，可用 ${pct0(UTIL)}，运行时开销 ${OH} GiB，权重 ${gbS(W_MEM)}。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>稠密 BF16 峰值的 50% 和峰值带宽的 60% 是这个 Lab 的假设（${J('eff','见“自由调参”')}）；真实 kernel 在不同规模下能用到的比例不一样。它们决定时间长短，不改变每个阶段受哪个限制：两个比例在 20% 到 100% 之间怎么取，拐点都在每字节 ${fint(RIDGE_LO)} 到 ${fint(RIDGE_HI)} 个 FLOP 之间，而 2K 个 token 的 prefill 是 ${fint(prefill(2048).ai)}，一个 decode 步是 ${fixd(decode(1,2048).ai,2)}。</li>`+
 `<li>不排队：首 token 延迟只算 prefill。真实服务还要等 batch 里空出位置，并在 decode 步之间穿插别的请求的 prefill。两者都会让用户多等，但都不改变 prefill 和 decode 步各受哪个限制。</li>`+
 `<li>一个 batch 里的请求上下文一样长；长短不一时，一步读的是各请求 KV cache 的总和。不计激活值读写显存的流量，也不计启动 kernel 的时间；60% 来自实测的每步耗时，里面已经包含了它们。</li>`+
 `<li>每个输出 token 的耗时取第一个 decode 步的。整个回答把每一步加起来，上下文每步多一个 token，只多读 128 KiB。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${MODEL}：${cite(SRC.cfg,'config.json')}。</li>`+
 `<li>${GPUN}：${cite(SRC.h100,'NVIDIA 产品页')}，BF16 1,979 TFLOPS 标着“with sparsity”，所以稠密是 989.5；3.35 TB/s；80 GB。</li>`+
 `<li>${cite(SRC.dbx,'Databricks（2023）')}：首 token 延迟和每个输出 token 的耗时；“生成第一个 token 通常受算力限制，之后的 decode 受显存带宽限制”；在两张 H100 80GB 上，batch 大小为 1 时用到峰值带宽的 60%。</li>`+
 `<li>${cite(SRC.roof,'Williams、Waterman 和 Patterson（2009）')}：roofline 模型和它的拐点（ridge point）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-28。原型，尚未经专家审阅。</p></div></div>`);
/* the kit's primer word for word (CONTRACT §9), plus one line on what this lab adds */
const PRIPOP=()=>A.t('priPop')+tx(`<p>The lab ${labLink('kv-cache')} works out how much memory the KV cache takes: 128 KiB per token for ${MODEL}. This lab adds how long each step takes to read it, and the weights.</p>`,
                                  `<p>KV cache 占多少显存，在 ${labLink('kv-cache')}这个 Lab 里算过：${MODEL} 每个 token 128 KiB。这个 Lab 再看每一步读它、读权重要多久。</p>`);
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers */
const OUTCOME=()=>tx('estimate a request’s time to first token and time per output token from the model and the GPU, and say which limit sets each.','从模型和 GPU 的参数估算一个请求的首 token 延迟和每个输出 token 的耗时，并说清各自受哪个限制。');
const EXITS=()=>{
  const a=compute({N:2048,M:300,B:1}), b64=compute({N:2048,M:300,B:64}), k64=Math.round(b64.thr)/Math.round(a.thr);
  const kvW=a.pre.tkv/a.ttft, kst=decMs(b64.tpot)/decMs(a.tpot);   // writing the 2K prompt's KV cache: 0.2% of prefill; 64 requests: a step 2.1 times as long
  const ad=decode(1,2048,A100), ap=prefill(2048,A100), fsA=A100.peak*EFF.mfu, bsA=A100.bw*EFF.mbu, g2=b=>fixd(b/1e9,2)+NS+'GB';   // two decimals, so the sum and the division redo by hand
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('A 4 times longer prompt makes the first token about 4 times slower, but later tokens only a little slower. Why?','prompt 长到 4 倍，首 token 慢到约 4 倍，之后的 token 却只慢一点点。为什么？'),
   o:[tx('Prefill multiplies every prompt token by weights it reads once, so its arithmetic grows with the prompt; a decode step reads all the weights for one token, and the prompt adds only its KV cache to that read.','prefill 让每个 prompt token 都和只读一遍的权重相乘，所以计算量随 prompt 增长；decode 步为了一个 token 把全部权重读一遍，prompt 只给这次读取加上它的 KV cache。'),
     tx('Prefill writes the whole prompt’s KV cache to GPU memory, and writing is slower than reading; a decode step only reads the KV cache, so the prompt hardly costs it anything.','prefill 要把整个 prompt 的 KV cache 写进显存，而写比读慢；decode 步只读 KV cache，所以 prompt 几乎不给它添成本。'),
     tx('Prefill runs the prompt’s tokens through the model one after another, so its time adds up with the prompt; a decode step handles one token, so it does not depend on the prompt.','prefill 把 prompt 的 token 一个接一个送进模型，所以耗时随 prompt 累加；decode 步只处理一个 token，所以和 prompt 无关。')],a:0,
   n:[null,tx(`Writing the prompt’s KV cache is small: at 2K tokens it takes ${kvW<0.01?'less than 1%':pct0(kvW)} of prefill’s time. Prefill’s time is its arithmetic.`,`写 prompt 的 KV cache 量很小：2K 个 token 时，它${kvW<0.01?'还不到 prefill 耗时的 1%':`只占 prefill 耗时的 ${pct0(kvW)}`}。prefill 的耗时是它的计算。`),
     tx('Prefill takes in the whole prompt in one step: all its tokens are multiplied by the weights together, which is why it is fast per token. Its time grows because its arithmetic grows.','prefill 在一步里读入整个 prompt：所有 token 一起和权重相乘，所以平均到每个 token 很快。它的耗时增长，是因为计算量在增长。')],
   why:tx('Prefill’s arithmetic, every prompt token times all the weights, takes far longer than reading the weights once, so its time grows with the prompt. A decode step reads all the weights for one token, so memory sets its time, and a longer prompt only adds its KV cache to that read, which is small next to the weights.',
          'prefill 的计算是让每个 prompt token 都乘一遍全部权重，远比把权重读一遍费时，所以它的耗时随 prompt 增长。decode 步为一个 token 把全部权重读一遍，读写显存决定它的耗时；prompt 变长，只给这次读取加上它的 KV cache，和权重比起来很小。')},
  /* the premise matches the lab's own busy server (64 requests at 2K: about 30 times the tokens per second, not 64): the gain is real but less than B-fold */
  {q:tx(`Decoding ${fint(b64.B)} requests together gives about ${ratioS(k64)} times as many tokens per second as one request, not ${fint(b64.B)} times. Why both?`,
        `${fint(b64.B)} 个请求一起 decode，每秒的 token 数约是一个请求时的 ${ratioS(k64)} 倍，而不是 ${fint(b64.B)} 倍。为什么涨了，又涨不到 ${fint(b64.B)} 倍？`),
   o:[tx('The step reads the weights once for all the requests, which raises the tokens per second; but each request adds a read of its own KV cache, so the step keeps getting longer.','这一步为所有请求只读一遍权重，所以每秒的 token 数涨了；但每个请求都要多读一份自己的 KV cache，所以这一步也一直在变长。'),
     tx('The requests share the GPU’s arithmetic, which raises the tokens per second; but each request gets a smaller share of it, so every token is computed more slowly.','这些请求共用 GPU 的算力，所以每秒的 token 数涨了；但每个请求分到的算力更少，所以每个 token 都算得更慢。'),
     tx('The requests share one KV cache, which raises the tokens per second; but the shared cache grows with every request, so reading it takes longer.','这些请求共用一份 KV cache，所以每秒的 token 数涨了；但这份共用的 cache 随请求数变大，所以读它也更久。')],a:0,
   n:[null,tx('Decode is not short of arithmetic: one token’s arithmetic is a tiny part of a decode step. The step grows because of memory reads: each request’s own KV cache.','decode 不缺算力：一个 token 的计算只占一个 decode 步里很小的一部分。这一步变长，是因为要读的显存变多了：每个请求都有自己的 KV cache。'),
     tx('Each request keeps its own KV cache, the keys and values of its own context. What the requests share is the read of the weights.','每个请求都有自己的 KV cache，也就是它自己上下文的 key 和 value。这些请求共用的是读权重的那一遍。')],
   why:tx(`A decode step’s time is its memory traffic: the weights, read once whatever the batch, plus each request’s own KV cache. So ${fint(b64.B)} requests get ${fint(b64.B)} tokens from each step, while the step gets only ${kst<2.5?'about twice as long':`about ${ratioS(kst)} times as long`}. The more requests, the more of each step goes to reading their KV caches.`,
          `decode 步的耗时就是读写显存的时间：权重不管 batch 多大都只读一遍，再加上每个请求自己的 KV cache。所以 ${fint(b64.B)} 个请求每一步拿到 ${fint(b64.B)} 个 token，这一步的耗时却只变成原来的${kst<2.5?'两倍左右':` ${ratioS(kst)} 倍左右`}。请求越多，每一步里花在读 KV cache 上的时间就越多。`)}],
 transfer:[
  {q:tx(`An A100 80GB SXM reaches 312 TFLOP/s of dense BF16 and 2,039 GB/s of memory bandwidth (NVIDIA’s datasheet); the ${GPUN} reaches 989.5 TFLOP/s and 3,350 GB/s. With this lab’s efficiencies, about how long does one decode step take for ${MODEL} at 2K tokens of context, one request?`,
        `A100 80GB SXM 的稠密 BF16 算力是 312 TFLOP/s，显存带宽 2,039 GB/s（NVIDIA 规格表）；${GPUN} 是 989.5 TFLOP/s 和 3,350 GB/s。按这个 Lab 的效率，${MODEL} 在 2K 个 token 的上下文上、一个请求时，一个 decode 步大约要多久？`),
   o:[tx('Under 10 ms','不到 10 ms'),tx('10 to 18 ms','10 到 18 ms'),tx('18 to 30 ms','18 到 30 ms'),tx('More than 30 ms','超过 30 ms')],a:1,
   /* a worked computation, one step per sentence (R6.6, R6.20) */
   why:tx(`Memory sets a decode step’s time, so it follows the bandwidth. It reads ${g2(W_READ)} of weights and 2,048 × 128 KiB = ${gbS(KV_TOK*2048)} of KV cache: ${g2(ad.by)}. At 2,039 GB/s × 60% = ${fixd(bsA/1e12,3)} TB/s, that takes ${fDec(ad.tm)} (${fDec(a.tpot)} on the ${GPUN}). The arithmetic, ${flS(ad.fl)} ÷ (312 × 50% = ${fint(fsA/1e12)} TFLOP/s) = ${fDec(ad.tc)}, stays far below. Source: ${cite(SRC.a100,'NVIDIA A100 datasheet')}.`,
          `decode 步的耗时由读写显存决定，所以跟着带宽走。这一步要读 ${g2(W_READ)} 的权重和 2,048 × 128 KiB = ${gbS(KV_TOK*2048)} 的 KV cache，一共 ${g2(ad.by)}。按 2,039 GB/s × 60% = ${fixd(bsA/1e12,3)} TB/s 读，要 ${fDec(ad.tm)}，${GPUN} 上是 ${fDec(a.tpot)}。计算 ${flS(ad.fl)} ÷（312 × 50% = ${fint(fsA/1e12)} TFLOP/s）= ${fDec(ad.tc)}，远低于此。来源：${cite(SRC.a100,'NVIDIA A100 规格表')}。`)},
  {q:tx(`On the same A100 80GB SXM, about how long does the prefill of a 2K-token prompt take, that is the time to first token?`,`还是这张 A100 80GB SXM，2K 个 token 的 prompt 做 prefill 大约要多久，也就是首 token 延迟是多少？`),
   o:[tx('Under 80 ms','不到 80 ms'),tx('80 to 140 ms','80 到 140 ms'),tx('140 to 250 ms','140 到 250 ms'),tx('More than 250 ms','超过 250 ms')],a:2,
   why:tx(`Arithmetic sets prefill’s time, so it follows the FLOP/s. At 312 × 50% = ${fint(fsA/1e12)} TFLOP/s, the prefill’s ${flS(ap.fl)} take ${fPre(ap.tc)}, ${fixd(989.5/312,1)} times the ${GPUN}’s ${fPre(a.ttft)}, as 989.5 ÷ 312 = ${fixd(989.5/312,1)}. Reading the weights and writing the KV cache, ${fDec(ap.tm)}, stays far below.`,
          `prefill 的耗时由计算决定，所以跟着 FLOP/s 走。按 312 × 50% = ${fint(fsA/1e12)} TFLOP/s 算，${flS(ap.fl)} 的 prefill 要 ${fPre(ap.tc)}，是 ${GPUN} 上 ${fPre(a.ttft)} 的 ${fixd(989.5/312,1)} 倍，因为 989.5 ÷ 312 = ${fixd(989.5/312,1)}。读权重、写 KV cache 要 ${fDec(ap.tm)}，远低于此。`)}],
 practice:[
  {id:'matmul-roofline',d:tx('Part 1: for four matmul shapes, count the FLOPs, the bytes moved and the arithmetic intensity, and say which are memory-bound; shape A, 8 tokens times a 4,096 × 4,096 weight, has the shape of a decode step’s matmul.','第 1 部分：四种形状的矩阵乘法，算出 FLOPs、搬运的字节数和算术强度，判断哪些受显存带宽限制；形状 A（8 个 token 乘一个 4,096 × 4,096 的权重）正是 decode 步里矩阵乘法的形状。')},
  {id:'inference-serving',d:tx('Items 1 and 4: what the time to first token is made of, then the first deep dive, separating prefill from decode work on the GPUs.','第 1 项和第 4 项：首 token 延迟由哪些部分组成；再做第一个深入话题，把 prefill 和 decode 的工作在 GPU 上分开。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Prefill vs decode','Prefill 和 decode'),title:l?L(l.q):tx('Why is the first token slow and the rest fast?','为什么首 token 要等很久，之后却一个接一个很快？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),primer:A.t('priBtn').replace(/\s*▾$/,''),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  const ph=(k,name)=>`<div class="pd-ph" data-ph="${k}"><div class="pd-ph-h"><b>${name}</b><span class="pd-ph-d"></span><span class="pd-ph-t mono"></span></div>`+
    `<div class="pd-rows"><div class="pd-row" data-k="c" tabindex="0"><span class="pd-rl">${tx('arithmetic','计算')}</span><div class="pd-bar"><span class="pd-ghost" hidden></span><i class="pd-bc"></i><b class="pd-goal" hidden></b><em class="pd-settag" hidden></em></div><span class="pd-rv mono"></span></div>`+
    `<div class="pd-row" data-k="m" tabindex="0"><span class="pd-rl">${tx('memory','读写显存')}</span><div class="pd-bar"><span class="pd-ghost" hidden></span><i class="pd-bw"></i><i class="pd-bk"></i><b class="pd-goal" hidden></b><em class="pd-settag" hidden></em></div><span class="pd-rv mono"></span></div>`+
    `<div class="pd-mask" hidden></div></div></div>`;
  /* part 6: the same request on three GPUs, one row each, on one time scale; a row is also the control that picks its GPU */
  const xrow=v=>`<button type="button" class="pd-xrow" data-xv="${v.id}" aria-pressed="false"><span class="pd-xrow-h"><span class="pd-xrow-n"><i class="pd-xdot" aria-hidden="true"></i><b>${esc(L(v.t))}</b></span>`+
    `<span class="pd-xrow-v mono"><span class="pd-xrow-f"></span><span class="pd-xrow-e"></span></span></span>`+
    `<span class="pd-xtrack"><i class="pd-xband"></i><i class="pd-xpre"></i><i class="pd-xfirst"></i><span class="pd-xticks"></span><span class="pd-xmask" hidden></span></span></button>`;
  const xa=`<div class="pd-xa" id="pd-xa" hidden><div class="pd-xa-top"><div class="pd-cap" id="pd-xa-cap"></div>`+
    `<span class="pd-xa-ideal" tabindex="0" data-tipx="${esc(tx(`Each variant is an idealised ${GPUN} with one number doubled, its dense BF16 FLOP/s or its memory bandwidth; the model, the 50% and 60% efficiencies and the request stay the same. A real new GPU changes several numbers at once. Keeping the efficiencies does not change which limit sets each phase: twice the FLOP/s moves the ridge from ${fint(RIDGE)} to ${fint(2*RIDGE)} FLOPs per byte, still far below the 2K prefill’s ${fint(prefill(2048).ai)}; twice the bandwidth moves it to ${fint(RIDGE/2)}, still far above a decode step’s ${fixd(decode(1,2048).ai,2)}.`,`两个变体都是理想化的 ${GPUN}：各自只把一项指标翻倍（稠密 BF16 的 FLOP/s，或显存带宽），模型、50% 和 60% 的效率、请求都不变。真实的新一代 GPU 会同时改变好几项指标。效率不变，也不改变每个阶段受哪个限制。算力翻倍，拐点从每字节 ${fint(RIDGE)} 个 FLOP 升到 ${fint(2*RIDGE)} 个，仍远低于 2K prefill 的 ${fint(prefill(2048).ai)} 个；带宽翻倍，拐点降到 ${fint(RIDGE/2)} 个，仍远高于 decode 步的 ${fixd(decode(1,2048).ai,2)} 个。`))}">${tx('Idealised: one number doubled, everything else the same.','理想化实验：只把一项指标翻倍，其余不变。')}</span></div>`+
    `<div class="pd-xrows" role="group" aria-label="${esc(tx('The GPU, pick one','GPU，选一张'))}">${XV.map(xrow).join('')}</div><div class="pd-xaxis mono" aria-hidden="true"></div>`+
    `<button type="button" class="btn sm pd-xgo" data-act="xq" hidden></button>`+
    `<div class="pd-xzoom" id="pd-xzoom" hidden><div class="pd-xzoom-cap"></div>${XV.map(v=>`<div class="pd-xzrow" data-xv="${v.id}"><span class="pd-xzn">${esc(L(v.t))}</span><span class="pd-xztrack"><i class="pd-xzpre"></i><i class="pd-xzfirst"></i><span class="pd-xzv mono"></span></span></div>`).join('')}<div class="pd-xzaxis mono" aria-hidden="true"></div></div></div>`;
  return `<div class="panel stage pd-stage" id="pd-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="pd-st-title"></div><div class="meter-sub" id="pd-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="pd-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="pd-legend"></div>`+
   `<div class="stage-graph graph pd-graph" id="pd-graph">`+
    `<div class="pd-tl" id="pd-tl"><div class="pd-cap" id="pd-tl-cap"></div><div class="pd-tl-labs"><span class="pd-tl-lpre"></span><span class="pd-tl-ldec"></span></div>`+
    `<div class="pd-tl-track" id="pd-tl-track" tabindex="0"><i class="pd-tl-pre"><span class="pd-tl-pin"></span></i><i class="pd-tl-first"></i><span class="pd-tl-ticks"></span><span class="pd-tl-mask" hidden></span></div>`+
    `<div class="pd-tl-axis mono" aria-hidden="true"></div></div>`+xa+
    `<div class="pd-lim" id="pd-lim"><div class="pd-cap" id="pd-lim-cap"></div>${ph('pre',tx('Prefill','prefill'))}${ph('dec',tx('Decode step','decode 步'))}</div>`+
    `<div class="pd-roof" id="pd-roof" hidden></div>`+
    `<div class="pd-mem" id="pd-mem" hidden><div class="pd-cap" id="pd-mem-cap"></div><div class="bar thin pd-membar" id="pd-membar" tabindex="0"></div><div class="tz-leg pd-memleg" id="pd-memleg"></div></div>`+
   `</div>`+
   /* the numbers' scope, next to the numbers (GPT-6 review of V17 #2): an estimate for one GPU, with no queue or network, each prefill on its own */
   `<p class="pd-scope small muted" tabindex="0" data-tipx="${esc(tx('Every time here comes from this lab’s step model: the arithmetic at 50% of the GPU’s peak FLOP/s, the memory traffic at 60% of its peak bandwidth, whichever is longer. A real server also queues requests, sends tokens over the network, and runs other requests’ prefills between decode steps.','这里的每个时间都来自这个 Lab 的耗时模型：计算按 GPU 峰值 FLOP/s 的 50%，读写显存按峰值带宽的 60%，取较长的那个。真实的服务还有排队、网络传输，而且会在 decode 步之间插入别的请求的 prefill。'))}">${tx('Estimates for one GPU: no queue and no network, and each prefill runs on its own.','单卡估算：不含排队和网络，每次 prefill 单独算。')}</p>`+
   `<div class="metrics pd-metrics" id="pd-metrics"></div>`+
   `<details class="math" id="pd-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="pd-mathbody"></div></details></div>`;
}
const EFF_NOTE=()=>tx(`This lab assumes a step reaches 50% of the ${GPUN}’s dense BF16 peak (989.5 TFLOP/s), that is ${tfS(H100.peak*EFF.mfu)} TFLOP/s; how much of the peak a real kernel reaches depends on the kernel and on how many tokens a step computes. It matters only when arithmetic sets the time, as in prefill.`,
  `本 Lab 假设一步能用到 ${GPUN} 稠密 BF16 峰值（989.5 TFLOP/s）的 50%，即 ${tfS(H100.peak*EFF.mfu)} TFLOP/s；真实 kernel 能用到峰值的多少，取决于 kernel 本身和一步算多少个 token。只有计算决定耗时时它才起作用，比如 prefill。`);
const BW_NOTE=()=>tx(`This lab assumes a step reaches 60% of the peak memory bandwidth (3.35 TB/s), that is ${tbS(H100.bw*EFF.mbu)} TB/s. Kernels never stream memory at the full peak; Databricks (2023) measured 60% at batch size 1 on two H100 80GB GPUs. It sets every decode step’s time, so a server that reaches more gets proportionally faster tokens.`,
  `本 Lab 假设一步能用到峰值显存带宽（3.35 TB/s）的 60%，即 ${tbS(H100.bw*EFF.mbu)} TB/s。kernel 读显存从来达不到满峰值；Databricks（2023）在两张 H100 80GB 上实测，batch 大小为 1 时是 60%。每个 decode 步的耗时都由它决定，所以能用到更多带宽的服务端，每个 token 会按比例更快。`);
function controlsHTML(){
  return `<aside class="panel controls" id="pd-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="pd-lock" hidden></p>`+
   `<div class="lockable" id="pd-ctl-N-wrap">`+nCtl('pd-ctl-N',false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The tokens prefill takes in: the new message and the conversation before it. The first decode step reads their KV cache, so this is also its context.','prefill 读入的 token：新消息加上之前的对话。第一个 decode 步要读它们的 KV cache，所以这也是它的上下文。')}</div></div>`)+`</div>`+
   `<div class="ctl-note" id="pd-Nlock" hidden>${tx('The prompt length can change after you answer part 2.','在第 2 部分作答后，才能改 prompt 长度。')}</div>`+
   `<div class="lockable">`+mCtl('pd-ctl-M',false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The tokens the model writes: the first comes from prefill, each later one from a decode step.','模型要写出的 token：第一个来自 prefill，之后每个来自一个 decode 步。')}</div></div>`)+`</div>`+
   `<div class="lockable">`+bCtl('pd-ctl-B','B',memLimit(S.N),tx('Requests decoding together (batch)','一起 decode 的请求数（batch）'),false)+`</div>`+
   `<div class="pd-grp lockable" id="pd-ctl-eff"><div class="pd-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('Arithmetic: 50% of the peak FLOP/s','计算：峰值 FLOP/s 的 50%')}</span><div class="ctl-note">${EFF_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Memory: 60% of the peak bandwidth','读写显存：峰值带宽的 60%')}</span><div class="ctl-note">${BW_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${A.t('cUtil')}${tx(': ','：')}${pct0(UTIL)}</span><div class="ctl-note">${A.t('utilNote')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx(`Runtime overhead: ${OH} GiB`,`运行时开销：${OH} GiB`)}</span><div class="ctl-note">${A.t('ohNote')}</div></div></div>`+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`# Llama-3.1-8B in BF16 on one H100 SXM: this lab's step model
L, D, KV_HEADS, HEAD_DIM = 32, 4096, 8, 128
VOCAB, FFN = 128_256, 14_336
PER_LAYER = 2*D*D + 2*D*KV_HEADS*HEAD_DIM + 3*D*FFN + 2*D
P_LAYERS = L*PER_LAYER + D        # multiplied by every token
P_HEAD = VOCAB*D                  # LM head: once per new token
W_READ = 2*(P_LAYERS + P_HEAD)    # bytes of weights read per step
KV_TOK = 2*L*KV_HEADS*HEAD_DIM*2  # 131,072 bytes per token

FLOPS = 1979e12/2 * 0.5           # 50% of the dense BF16 peak
BW = 3.35e12 * 0.6                # 60% of the peak bandwidth

def attn(a, b):   # QK^T and AV FLOPs for query positions a..b-1
    return 2*L*D*(b*(b + 1) - a*(a + 1))

def step(tokens, logits, attn_flops, kv_tokens):
    flops = 2*P_LAYERS*tokens + 2*P_HEAD*logits + attn_flops
    nbytes = W_READ + KV_TOK*kv_tokens
    return max(flops/FLOPS, nbytes/BW)   # the slower one

def ttft(n):          # prefill: the whole prompt in one step
    return step(n, 1, attn(0, n), n)

def tpot(c, b=1):     # one decode step: b requests at context c
    return step(b, b, b*attn(c, c + 1), b*c)

print(f"TTFT {ttft(2048)*1e3:.1f} ms, TPOT {tpot(2048)*1e3:.2f} ms")
for b in (1, 16):  # b requests at 2,048 tokens each share a step
    print(f"{b} x 2048: {b/tpot(2048, b):.0f} tokens/s")
# TTFT 60.0 ms, TPOT 7.60 ms
# 1 x 2048: 132 tokens/s
# 16 x 2048: 1666 tokens/s`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="pd-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="pd-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s step model: the time to first token and the time per output token, each the slower of the arithmetic and the memory traffic.','这个 Lab 的耗时模型：首 token 延迟和每个输出 token 的耗时，各取计算和读写显存中较慢的那个。')}</p><pre class="code" id="pd-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="pd-prod">`+
   tx(`<li>vLLM measures both per request in ${cite(SRC.vstats,'<code>vllm/v1/metrics/stats.py</code>')}: the time to first token from the request’s arrival, so queueing counts, and the mean time per output token, leaving out the token prefill produced:${FX('\\text{TPOT} = \\frac{\\text{decode time}}{\\text{output tokens} - 1}')}</li>`+
      `<li>${cite(SRC.vlog,'<code>vllm/v1/metrics/loggers.py</code>')} exports them as the Prometheus histograms <code>vllm:time_to_first_token_seconds</code>, <code>vllm:request_time_per_output_token_seconds</code> and <code>vllm:inter_token_latency_seconds</code>; the benchmark in ${cite(SRC.vbench,'<code>vllm/benchmarks/serve.py</code>')} reports TPOT as (latency − TTFT) ÷ (output tokens − 1).</li>`+
      `<li>SGLang exports <code>sglang:time_to_first_token_seconds</code> and <code>sglang:inter_token_latency_seconds</code> from ${cite(SRC.sglm,'<code>python/sglang/srt/observability/metrics_collector.py</code>')}; its benchmark, ${cite(SRC.sglb,'<code>python/sglang/benchmark/serving.py</code>')}, computes TPOT the same way.</li>`,
      `<li>vLLM 在 ${cite(SRC.vstats,'<code>vllm/v1/metrics/stats.py</code>')} 里为每个请求测这两个量：首 token 延迟从请求到达算起，所以排队也算在内。平均每个输出 token 的耗时不算 prefill 产出的那个 token：${FX('\\text{TPOT} = \\frac{\\text{decode 时间}}{\\text{输出 token 数} - 1}')}</li>`+
      `<li>${cite(SRC.vlog,'<code>vllm/v1/metrics/loggers.py</code>')} 把它们导出成 Prometheus 直方图 <code>vllm:time_to_first_token_seconds</code>、<code>vllm:request_time_per_output_token_seconds</code> 和 <code>vllm:inter_token_latency_seconds</code>；${cite(SRC.vbench,'<code>vllm/benchmarks/serve.py</code>')} 里的 benchmark 按（总延迟 − TTFT）÷（输出 token 数 − 1）报告 TPOT。</li>`+
      `<li>SGLang 在 ${cite(SRC.sglm,'<code>python/sglang/srt/observability/metrics_collector.py</code>')} 里导出 <code>sglang:time_to_first_token_seconds</code> 和 <code>sglang:inter_token_latency_seconds</code>；它的 benchmark ${cite(SRC.sglb,'<code>python/sglang/benchmark/serving.py</code>')} 也这样算 TPOT。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.roof,'Roofline: An Insightful Visual Performance Model for Multicore Architectures')}<span>${tx('Williams, Waterman and Patterson, CACM 2009: attainable FLOP/s against arithmetic intensity, and the ridge point','Williams、Waterman、Patterson，CACM 2009：能达到的 FLOP/s 与算术强度的关系，以及拐点')}</span></li>`+
   `<li>${cite(SRC.dist,'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving')}<span>${tx('Zhong et al., OSDI 2024: prefill and decoding on different GPUs, with TTFT and TPOT targets','Zhong 等，OSDI 2024：把 prefill 和 decode 放到不同的 GPU 上，分别满足 TTFT 和 TPOT 目标')}</span></li>`+
   `<li>${cite(SRC.split,'Splitwise: Efficient Generative LLM Inference Using Phase Splitting')}<span>${tx('Patel et al., ISCA 2024: a compute-intensive prompt phase and a memory-intensive token phase, run on separate machines','Patel 等，ISCA 2024：计算密集的 prompt 阶段和访存密集的 token 阶段，放到不同的机器上跑')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="pd-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="pd-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="pd-app m2" id="pd-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="pd-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much of the wait for the whole answer is prefill.','选一个场景，看看整个回答要等的时间里，prefill 占多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="pd-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${DEEP()}${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Serve the most users on one ${GPUN} at 50 tokens per second each`,`一张 ${GPUN}，每人每秒 50 个 token，最多能服务几个用户`)}</h3>`+
   `<p>${tx(`One ${GPUN} serves ${MODEL} chats with 4K (4,096) tokens of context each. Every user must get at least 50 tokens per second, that is at most 20 ms per output token. You may change only the number of users decoding together; serve as many at once as you can:`,
            `一张 ${GPUN} 跑 ${MODEL} 的对话，每个对话有 4K（4,096）个 token 的上下文。每个用户每秒至少要拿到 50 个 token，也就是每个输出 token 不超过 20 ms。你只能改一起 decode 的用户数，尽量同时多服务一些：`)}</p>`+
   bCtl('pd-ic-chB','chB',chMax(),tx('Users decoding together','一起 decode 的用户数'),true)+
   `<div class="goal mono" id="pd-ch-goal"><span>${tx('Time per output token now','现在每个输出 token 的耗时')}</span><span><b id="pd-ch-now"></b>${tx(' / at most 20 ms',' / 最多 20 ms')}</span></div><div class="progress" id="pd-ch-prog" aria-hidden="true"><i></i></div>`+
   submit+`<div id="pd-ch-res" class="pd-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function renderRail(){const rail=$('#pd-rail',app); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';   // the next lab in every mode (R3.11)
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#pd-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step])&&(!s.gate||s.gate());
    /* part 6: the GPU options and the load follow the state (a click on a timeline row, Show me), and each option keeps its result */
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===xv)); const rr=b.querySelector('.lv-r'), dd=b.querySelector('.lv-d');
      if(rr) rr.textContent=xRes(v); if(dd) dd.textContent=xDesc(v);});
    const xs=$('.seg-ctl[data-key="xl"]',rail); if(xs){A.ui.setPressed(xs,xl); const nt=$('.pd-xl-note',rail); if(nt) nt.textContent=xLoadD(xl);}
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-act="roof"]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(roofOpen));});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#pd-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the four scenarios above.','从上面四个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
    $$('[data-act="roof"]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(roofOpen));});
  } else {
    const p=chPoint(S.B), now=$('#pd-ch-now',rail); if(!now) return;
    now.textContent=fDec(p.t);
    $('#pd-ch-prog i',rail).style.width=Math.min(100,p.t/CH.goal*100).toFixed(1)+'%';
    $('#pd-ch-goal',rail).classList.toggle('met',p.ok); $('#pd-ch-prog',rail).classList.toggle('met',p.ok); $('#pd-ch-goal',rail).classList.toggle('pd-over',!p.ok); $('#pd-ch-prog',rail).classList.toggle('pd-over',!p.ok);
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#pd-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
function statusOf(){
  const r=R();
  if(mode==='challenge'){const ok=r.tpot<=CH.goal; return {cls:ok?'ok':'bad',txt:ok?tx(`Within 20 ms: ${fDec(r.tpot)}`,`在 20 ms 以内：${fDec(r.tpot)}`):tx(`Over 20 ms: ${fDec(r.tpot)}`,`超过 20 ms：${fDec(r.tpot)}`),short:ok?tx('Within 20 ms','20 ms 以内'):tx('Over 20 ms','超过 20 ms'),
    tip:tx(`Green when each user’s time per output token is at most 20 ms, that is at least 50 tokens per second, with ${userN(r.B)} at 4K tokens of context each.`,`${userN(r.B)}、每人 4K 个 token 上下文时，每个用户每个输出 token 的耗时不超过 20 ms（每秒至少 50 个 token），徽章就是绿色。`)};}
  if(isX()){
    if(!xpred) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),
      tip:tx('The two doubled GPUs stay hidden until you answer the question in this part.','回答这一部分的问题之前，两个翻倍的 GPU 先隐藏。')};
    const x=xRun(xv,xl), o=xRun('orig',xl);
    return {cls:'neutral',txt:tx(`${xName(xv)}: whole answer ${xWa(x.e2e)}`,`${xName(xv)}：整个回答 ${xWa(x.e2e)}`),short:L(xLoad(xl).t),
      tip:xv==='orig'?tx(`The GPU picked, and the time of the whole ${fint(x.M)}-token answer on it: prefill, then every decode step.`,`选中的 GPU，以及它写完整个 ${fint(x.M)} 个 token 的回答要多久：prefill 加上每一个 decode 步。`)
        :tx(`The GPU picked, an idealised variant of the ${GPUN}, and the time of the whole ${fint(x.M)}-token answer on it; the original ${GPUN} takes ${xWa(o.e2e)}.`,`选中的 GPU（${GPUN} 的理想化变体），以及它写完整个 ${fint(x.M)} 个 token 的回答要多久；原版 ${GPUN} 要 ${xWa(o.e2e)}。`)};}
  if(decMasked()) return {cls:'neutral',txt:tx(`Prefill: ${fPre(r.ttft)}`,`prefill：${fPre(r.ttft)}`),short:tx('Prefill','prefill'),
    tip:tx(`The time to first token, prefill’s time for ${an(tokK(r.N))} ${tokK(r.N)}-token prompt, by this lab’s estimate. The decode times appear in part 4, where they answer your prediction.`,`首 token 延迟，也就是 ${tokK(r.N)} 个 token 的 prompt 做 prefill 的耗时（按这个 Lab 的估计）。decode 的耗时在第 4 部分出现，那里揭晓你的预测。`)};
  const q=preMs(r.ttft)/decMs(r.tpot);
  return {cls:'neutral',txt:tx(`First token: ${ratioS(q)} times a later one`,`首 token 是之后每个的 ${ratioS(q)} 倍`),short:tx(`${tokK(r.N)} prompt`,`prompt ${tokK(r.N)}`),
    tip:tx(`How many times as long the first token takes as each later token, by this lab’s estimate: ${fPre(r.ttft)} ÷ ${fDec(r.tpot)} = ${ratioS(q)}.`,`按这个 Lab 的估计，首 token 的耗时是之后每个 token 的几倍：${fPre(r.ttft)} ÷ ${fDec(r.tpot)} = ${ratioS(q)}。`)};
}
function drawHead(){
  const r=R(), t=$('#pd-st-title',app), sub=$('#pd-st-sub',app), st=$('#pd-status',app);
  t.textContent=isX()?tx(`${MODEL} in BF16: the ${GPUN} and two idealised variants`,`BF16 的 ${MODEL}：${GPUN} 和两个理想化的变体`):tx(`${MODEL} on one ${GPUN}, BF16`,`一张 ${GPUN} 跑 ${MODEL}，BF16`);
  sub.textContent=mode==='challenge'?(r.B===1?tx('1 user, 4K tokens of context','1 个用户，4K 个 token 的上下文'):tx(`${userN(r.B)} decoding together, 4K tokens of context each`,`${userN(r.B)}一起 decode，每人 4K 个 token 的上下文`))
    :(r.B===1?tx(`One request: ${an(tokK(r.N))} ${tokK(r.N)}-token prompt, a ${fint(r.M)}-token answer`,`一个请求：prompt ${tokK(r.N)} 个 token，回答 ${fint(r.M)} 个 token`)
      :tx(`One of ${fint(r.B)} requests decoding together: ${an(tokK(r.N))} ${tokK(r.N)}-token prompt, a ${fint(r.M)}-token answer`,`一起 decode 的 ${fint(r.B)} 个请求之一：prompt ${tokK(r.N)} 个 token，回答 ${fint(r.M)} 个 token`));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function legendHTML(){
  const it=(cls,txt,tip,jump)=>`<span data-tipx="${esc(tip)}" tabindex="0"${jump?` data-jump="${jump}"`:''}><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  const ch=mode==='challenge', x=isX(), k=xK;
  const pre=it('pd-sw-pre',tx('prefill','prefill'),tx('The gray block on the timeline is the prefill: the GPU computes every prompt token in one step before the first token appears.','时间线上的灰色块是 prefill：GPU 先在一步里算完所有 prompt token，之后才出现首 token。'))+
    it('pd-sw-first',tx('thick line: first token','粗线：首 token'),tx('The thick line on the timeline is the first token: it appears when prefill ends.','时间线上的粗线是首 token：prefill 结束时出现。'));
  const bars=it('pd-sw-ar',tx('arithmetic','计算'),tx('A gray bar is a step’s arithmetic time: its FLOPs divided by the FLOP/s the GPU reaches.','灰色条是一步的计算时间：它的 FLOP 数除以 GPU 能达到的 FLOP/s。'))+
    it('w',tx('weights read','读权重'),tx(`The blue part of a memory bar is the time to read the weights: ${gbS(W_READ)} once per step, however many tokens the step computes.`,`读写显存条的蓝色部分是读权重的时间：每一步读一遍 ${gbS(W_READ)}，不管这一步算多少个 token。`))+
    it('kv','KV cache',tx('The amber part of a memory bar is the KV cache’s time: prefill writes the prompt’s keys and values, a decode step reads its context’s, 128 KiB per token.','读写显存条的琥珀色部分是 KV cache 的时间：prefill 写入 prompt 的 key 和 value，decode 步读出它上下文的 key 和 value，每个 token 128 KiB。'));
  /* part 6: a tick every k tokens (k keeps the fastest timeline's ticks apart), the faded timelines, the original's outlines */
  if(x) return pre+it('pd-sw-tick',k===1?tx('thin tick: a later token','细线：之后的一个 token'):tx(`thin tick: every ${k} tokens`,`细线：每 ${k} 个 token 一条`),
      k===1?tx('Each thin tick on a timeline is one more token of the answer: one decode step each.','时间线上的每条细线是回答的又一个 token：每个都要一个 decode 步。')
        :tx(`Each thin tick on a timeline marks ${k} more tokens of the answer, one decode step each; the last tick is the answer’s last token.`,`时间线上每条细线表示回答又多了 ${k} 个 token，每个 token 一个 decode 步；最后一条细线是回答的最后一个 token。`))+
    it('pd-sw-xoff',tx('faded timeline: not the GPU picked','半透明的时间线：没选中的 GPU'),tx('The timeline of the GPU picked is drawn in full; the other two are faded.','选中的 GPU 的时间线画成实心，另外两条半透明。'))+bars+
    it('pd-sw-ghost',tx('hatched: the original H100','斜纹：原版 H100'),tx('A hatched bar behind a bar is the same bar on the original H100, on the same scale: where the hatching shows, the bar got shorter.','一根条后面的斜纹条是原版 H100 上同一根条，比例尺相同：露出斜纹的地方，就是这根条变短了。'))+
    it('pd-sw-fade',tx('faded bar: does not set the time','半透明的条：不决定耗时'),tx('In each phase the longer bar sets the step’s time; the shorter one is drawn faded, because the GPU does it at the same time.','每个阶段里，较长的那条决定这一步的耗时；较短的那条画成半透明，因为 GPU 同时在做它。'));
  return pre+it('pd-sw-tick',tx('thin tick: a later token','细线：之后的一个 token'),tx('Each thin tick on the timeline is one more token of the answer: one decode step each.','时间线上的每条细线是回答的又一个 token：每个都要一个 decode 步。'))+bars+
    (ch?it('lim',tx('dashed line: 20 ms goal','虚线：目标 20 ms'),tx('The dashed line is the goal: at most 20 ms per output token, that is at least 50 tokens per second for each user.','虚线是目标：每个输出 token 最多 20 ms，也就是每个用户每秒至少 50 个 token。'))
       :it('pd-sw-fade',tx('faded: does not set the time','半透明：不决定耗时'),tx('In each phase the longer bar sets the step’s time; the shorter one is drawn faded, because the GPU does it at the same time.','每个阶段里，较长的那条决定这一步的耗时；较短的那条画成半透明，因为 GPU 同时在做它。')));
}
/* nice axis steps for the timeline, in ms */
function niceStep(span,n){const raw=span/(n||5), p=Math.pow(10,Math.floor(Math.log10(raw))), m=raw/p; return (m<1.5?1:(m<3.5?2:(m<7.5?5:10)))*p;}
/* part 6: three timelines, one per GPU, on one time scale: the original's whole answer (a shorter bar is visibly shorter). The GPU picked
   is drawn in full, the others faded; before the prediction only the original is shown (R3.8) */
function drawX(){
  const box=$('#pd-xa',app), tl=$('#pd-tl',app); if(!box) return false;
  const on=isX(); box.hidden=!on; if(tl) tl.hidden=on;
  if(!on) return false;
  const o=xRun('orig',xl), span=o.e2e*1.03, open=!!xpred, W=$('.pd-xtrack',box).clientWidth||500;
  /* one tick every k tokens, the same k on every row (so the faster row's ticks sit closer): the fastest row's ticks stay 4 px apart */
  const px=Math.min(...XV.map(v=>xRun(v.id,xl).tpot))/span*W;
  xK=[1,2,5,10,20,50,100].find(k=>k*px>=4)||100;
  const X=v=>Math.max(0,Math.min(100,v/span*100)), pos=v=>X(v).toFixed(3)+'%';
  $('#pd-xa-cap',app).textContent=tx('The same request on three GPUs, on one time scale','同一个请求放到三张 GPU 上，用同一条时间轴');
  /* before the prediction: the question is the way in (R7.3, review A1); a masked row and this button bring it on screen */
  const go=$('.pd-xgo',box); go.hidden=open; go.textContent=tx('Answer the question to see the two doubled GPUs ↓','先回答问题，再看两张翻倍的 GPU ↓');
  XV.forEach(v=>{
    const row=$(`.pd-xrow[data-xv="${v.id}"]`,box), r=xRun(v.id,xl), hid=!open&&v.id!=='orig', sel=v.id===xv, tr=$('.pd-xtrack',row), mk=$('.pd-xmask',tr), nm=xName(v.id);
    row.setAttribute('aria-pressed',String(sel)); row.classList.toggle('pd-xsel',sel); row.classList.toggle('pd-xoff',open&&!sel); row.classList.toggle('pd-xhid',hid);
    mk.hidden=!hid;
    if(hid){mk.textContent=tx('shown after your prediction','预测之后显示'); $('.pd-xpre',tr).style.width='0%'; $('.pd-xfirst',tr).style.left='0%'; $('.pd-xband',tr).style.width='0%'; $('.pd-xticks',tr).innerHTML='';
      $('.pd-xrow-f',row).textContent=''; $('.pd-xrow-e',row).textContent='';
      row.dataset.tipx=tx('This GPU is shown once you answer the question in this part. Click to jump to the question.','回答这一部分的问题之后，才显示这张 GPU。点击跳到问题。'); row.setAttribute('aria-label',nm+tx(': ','：')+row.dataset.tipx); return;}
    $('.pd-xpre',tr).style.width=pos(r.ttft); $('.pd-xfirst',tr).style.left=pos(r.ttft);
    const band=$('.pd-xband',tr); band.style.left=pos(r.ttft); band.style.width=Math.max(0,X(r.e2e)-X(r.ttft)).toFixed(3)+'%';
    const last=r.times.length-1;
    $('.pd-xticks',tr).innerHTML=r.times.map((t,i)=>(i+1)%xK===0||i===last?`<i style="left:${pos(t)}"></i>`:'').join('');
    $('.pd-xrow-f',row).textContent=tx(`first token ${xFt(r.ttft)}`,`首 token ${xFt(r.ttft)}`);
    $('.pd-xrow-e',row).textContent=tx(`whole answer ${xWa(r.e2e)}`,`整个回答 ${xWa(r.e2e)}`);
    const act=sel?'':tx(' Click to see it in the bars and the cards.','点击，在时间条和指标卡里看它。');
    row.dataset.tipx=(v.id==='orig'?tx(`The ${GPUN}: `,`${GPUN}：`):tx(`${nm}, an idealised ${GPUN} with ${v.id==='c2'?'twice the FLOP/s':'twice the bandwidth'}: `,`${nm}，${v.id==='c2'?'FLOP/s':'带宽'}翻倍的理想化 ${GPUN}：`))+
      tx(`the first token after ${xFt(r.ttft)}, then one every ${fDec(r.tpot)} at first; the whole ${fint(r.M)}-token answer takes ${xWa(r.e2e)}.`,`${xFt(r.ttft)} 后出首 token，之后起初每 ${fDec(r.tpot)} 一个；整个 ${fint(r.M)} 个 token 的回答要 ${xWa(r.e2e)}。`)+act;
    row.setAttribute('aria-label',row.dataset.tipx);});
  const spanMs=span*1e3, st=niceStep(spanMs,Math.max(2,Math.min(5,Math.floor(W/78)))), inS=spanMs>=2000, lab=v=>inS?fixd(v/1000,st>=1000?0:1):fint(v);
  const vs=[]; for(let v=0;v<=spanMs*0.93;v+=st) vs.push(v);
  $('.pd-xaxis',box).innerHTML=vs.map((v,i)=>`<span style="left:${(v/spanMs*100).toFixed(3)}%">${lab(v)}${i===vs.length-1?NS+(inS?'s':'ms'):''}</span>`).join('');
  /* the first token, zoomed in, when it is a sliver of the time scale (the chat): the prefill blocks on their own axis, so compute × 2's
     halving is seen, not read (review A2); the three timelines above keep their one scale */
  const zb=$('#pd-xzoom',box), small=o.ttft/span<0.1; zb.hidden=!small;
  if(small){
    const zs=[50,100,200,500,1000].find(z=>z>=Math.max(...XV.map(v=>xRun(v.id,xl).ttft))*1e3*1.3)||1000, Z=v=>Math.min(100,v*1e3/zs*100).toFixed(3)+'%';
    $('.pd-xzoom-cap',zb).textContent=tx(`The first token, zoomed in: 0 to ${fint(zs)} ms`,`放大看首 token：0 到 ${fint(zs)} ms`);
    XV.forEach(v=>{const zr=$(`.pd-xzrow[data-xv="${v.id}"]`,zb), r=xRun(v.id,xl), hid=!open&&v.id!=='orig';
      zr.hidden=hid; zr.classList.toggle('pd-xoff',open&&v.id!==xv); zr.classList.toggle('pd-xsel',v.id===xv);
      if(hid) return;
      $('.pd-xzpre',zr).style.width=Z(r.ttft); $('.pd-xzfirst',zr).style.left=Z(r.ttft);
      const val=$('.pd-xzv',zr); val.textContent=xFt(r.ttft); val.style.left=`calc(${Z(r.ttft)} + 6px)`;});
    $('.pd-xzaxis',zb).innerHTML=[0,zs/2,zs].map((v,i)=>`<span style="left:${(v/zs*100).toFixed(3)}%">${fint(v)}${i===2?NS+'ms':''}</span>`).join('');
  }
  return true;
}
function drawTL(){
  if(drawX()) return;
  const r=R(), tl=$('#pd-tl',app); if(!tl) return;
  const masked=decMasked(), K=r.ticks.length;
  /* part 3 keeps one time scale, up to the longest prompt's first token, so the prefill block visibly grows with the prompt */
  const span=masked?Math.max(prefill(32768).t*1.07,r.ttft*1.05):(K?r.ticks[K-1]*1.015:r.ttft*1.2);
  const X=v=>Math.max(0,Math.min(100,v/span*100)), pos=v=>X(v).toFixed(3)+'%';
  $('.pd-tl-pre',tl).style.width=pos(r.ttft); $('.pd-tl-first',tl).style.left=pos(r.ttft);
  /* the label inside the prefill block: the long one if it fits, else "2K", else none (measured, so it never runs into the first-token line) */
  const pin=$('.pd-tl-pin',tl), pw=X(r.ttft)/100*(tl.clientWidth||500), fits=s=>{pin.textContent=s; return pin.offsetWidth<=pw-8;};
  if(!fits(tx(`${fint(r.N)} prompt tokens, one step`,`${fint(r.N)} 个 prompt token，一步`))&&!fits(tokK(r.N))) pin.textContent='';
  const tk=$('.pd-tl-ticks',tl), mk=$('.pd-tl-mask',tl);
  tk.innerHTML=masked?'':r.ticks.map(t=>`<i style="left:${pos(t)}"></i>`).join('');
  mk.hidden=!masked; if(masked){mk.style.left=pos(r.ttft); mk.style.width=(100-X(r.ttft)).toFixed(3)+'%'; mk.textContent=tx('decode: part 4','decode：第 4 部分');}
  const tw=$('#pd-tl-track',app).clientWidth||500, spanMs=span*1e3, st=niceStep(spanMs,Math.max(2,Math.min(5,Math.floor(tw/78)))), inS=spanMs>=2000, lab=v=>inS?fixd(v/1000,st>=1000?0:1):fint(v);
  const vs=[]; for(let v=0;v<=spanMs*0.93;v+=st) vs.push(v);
  $('.pd-tl-axis',tl).innerHTML=vs.map((v,i)=>`<span style="left:${(v/spanMs*100).toFixed(3)}%">${lab(v)}${i===vs.length-1?NS+(inS?'s':'ms'):''}</span>`).join('');
  $('.pd-tl-lpre',tl).textContent=tx(`prefill → first token at ${fPre(r.ttft)}`,`prefill → ${fPre(r.ttft)} 出首 token`);
  $('.pd-tl-ldec',tl).textContent=masked?tx('decode: shown in part 4','decode：第 4 部分显示'):tx(`decode: 1 token every ${fDec(r.tpot)}`,`decode：每 ${fDec(r.tpot)} 一个 token`);
  const cap=$('#pd-tl-cap',app);
  cap.textContent=masked?tx(`The time to first token for ${an(tokK(r.N))} ${tokK(r.N)}-token prompt, on one scale up to 32K tokens`,`${tokK(r.N)} 个 token 的 prompt 的首 token 延迟，时间轴按 32K 个 token 的长度画`)
    :(K+1>=r.M?tx(`One request: all ${fint(r.M)} tokens of its answer`,`一个请求：回答的全部 ${fint(r.M)} 个 token`)
      :(mode==='challenge'?tx(`One user’s request: its first ${K+1} tokens`,`一个用户的请求：前 ${K+1} 个 token`)
        :tx(`One request: the first ${K+1} of its ${fint(r.M)} tokens (whole answer ${fTot(r.e2e)})`,`一个请求：${fint(r.M)} 个 token 里的前 ${K+1} 个（整个回答 ${fTot(r.e2e)}）`)));
  const track=$('#pd-tl-track',app);
  track.dataset.tipx=masked?tx(`The prefill of ${an(fint(r.N))} ${fint(r.N)}-token prompt takes ${fPre(r.ttft)}; then the first token appears. The decode steps after it are shown from part 4 on.`,`${fint(r.N)} 个 token 的 prompt 做 prefill 要 ${fPre(r.ttft)}，之后出现首 token。后面的 decode 步从第 4 部分起显示。`)
    :tx(`The prefill of all ${fint(r.N)} prompt tokens takes ${fPre(r.ttft)}, one step; the first token appears at its end (the thick line). Each thin tick is one more token, one decode step each, ${fDec(r.tpot)} apart at first.`,`${fint(r.N)} 个 prompt token 的 prefill 在一步里完成，要 ${fPre(r.ttft)}；它结束时出现首 token（粗线）。每条细线是又一个 token，各用一个 decode 步，开始时相隔 ${fDec(r.tpot)}。`);
  track.setAttribute('aria-label',track.dataset.tipx);
}
function drawPhase(ph,o,F,opt){
  const box=$(`.pd-ph[data-ph="${ph}"]`,app); if(!box) return;
  opt=opt||{};
  box.hidden=!!opt.hide; if(opt.hide) return;
  const mask=$('.pd-mask',box), masked=!!opt.masked;
  mask.hidden=!masked; box.classList.toggle('pd-masked',masked);
  if(masked){mask.textContent=tx('Decode: shown in part 4','decode：第 4 部分显示'); $('.pd-ph-t',box).textContent=''; $('.pd-ph-d',box).textContent=tx('one token per step','每步一个 token');
    $$('.pd-row',box).forEach(rw=>{rw.dataset.tipx=tx('Decode’s bars are shown from part 4 on, where they answer your prediction.','decode 的两根时间条从第 4 部分起显示，那里揭晓你的预测。'); rw.querySelector('.pd-rv').textContent=''; rw.querySelectorAll('i').forEach(i=>{i.style.width='0%';}); const g=rw.querySelector('.pd-ghost'); if(g) g.hidden=true;});
    return;}
  const max=opt.scale||Math.max(o.tc,o.tm), P=v=>Math.max(0,Math.min(100,v/max*100));
  const cRow=$('.pd-row[data-k="c"]',box), mRow=$('.pd-row[data-k="m"]',box), cb=o.tc>=o.tm;
  $('.pd-bc',cRow).style.width=P(o.tc).toFixed(3)+'%';
  $('.pd-bw',mRow).style.width=P(o.tw).toFixed(3)+'%'; const bk=$('.pd-bk',mRow); bk.style.left=P(o.tw).toFixed(3)+'%'; bk.style.width=(P(o.tm)-P(o.tw)).toFixed(3)+'%';
  cRow.classList.toggle('pd-set',cb); mRow.classList.toggle('pd-set',!cb);
  const fade=opt.fade!=null?opt.fade:!opt.scale;              // the challenge's fixed scale draws both bars in full; part 6 keeps the fading
  cRow.classList.toggle('pd-fade',!cb&&fade); mRow.classList.toggle('pd-fade',cb&&fade);
  $('.pd-rv',cRow).textContent=F(o.tc); $('.pd-rv',mRow).textContent=F(o.tm);
  /* part 6: the original's bars as dashed outlines on the same scale, so a bar that halved shows it */
  const gh=opt.ghost; [[cRow,gh&&gh.tc],[mRow,gh&&gh.tm]].forEach(([rw,v])=>{const g=$('.pd-ghost',rw); if(!g) return; g.hidden=!gh; if(gh) g.style.width=P(v).toFixed(3)+'%';});
  /* the longer bar says that it sets the time, inside the bar (not in the challenge, where the goal line matters); with an outline
     around it, at the bar's own end */
  [[cRow,cb,o.tc],[mRow,!cb,o.tm]].forEach(([rw,on,v])=>{const tg=$('.pd-settag',rw); tg.hidden=!on||!!opt.goal; tg.textContent=tx('sets the time','决定耗时'); tg.style.right=gh?`calc(${(100-P(v)).toFixed(3)}% + 4px)`:'';});
  $('.pd-ph-t',box).textContent=F(o.t);
  $$('.pd-goal',box).forEach(g=>{g.hidden=!opt.goal; if(opt.goal) g.style.left=P(opt.goal).toFixed(3)+'%';});
  return {cRow,mRow,cb};
}
/* part 6: the two limits of each phase for the GPU picked, on the original's scale, with the original's bars as dashed outlines */
function drawLimX(){
  const o=xRun('orig',xl), r=xRun(xv,xl), diff=xv!=='orig', nm=xName(xv), pk=r.g.peak*EFF.mfu, bw=r.g.bw*EFF.mbu;
  $('#pd-lim-cap',app).textContent=tx('Why: each step takes the longer of its arithmetic and its memory traffic','原因：每一步的耗时取计算和读写显存中较长的那个');
  const was=(F,a,b)=>diff&&F(a)!==F(b)?tx(` (original: ${F(b)})`,`（原版 ${F(b)}）`):'', fP=xFt;
  const onG=tx(diff?` on ${nm}`:'',diff?`（${nm}）`:'');
  const pre=drawPhase('pre',r.pre,fP,{scale:o.pre.t,fade:true,ghost:diff?o.pre:null}), bp=$('.pd-ph[data-ph="pre"]',app);
  $('.pd-ph-d',bp).textContent=tx(`one step over ${tokN(r.N)}`,`一步算完 ${tokN(r.N)}`);
  $('.pd-ph-t',bp).textContent=fP(r.pre.t)+was(fP,r.pre.t,o.pre.t);
  const og=(F,v)=>diff?tx(`; the hatched bar behind it is the original’s ${F(v)}`,`；后面的斜纹条是原版的 ${F(v)}`):'';
  pre.cRow.dataset.tipx=tx(`Prefill’s arithmetic${onG}: ${flS(r.pre.fl)} at ${tfS(pk)} TFLOP/s, so ${fP(r.pre.tc)}${og(fP,o.pre.tc)}.${pre.cb?' It is the longer of the two, so it sets prefill’s time.':''}`,
    `prefill 的计算${onG}：${flS(r.pre.fl)}，按 ${tfS(pk)} TFLOP/s 算，要 ${fP(r.pre.tc)}${og(fP,o.pre.tc)}。${pre.cb?'它是两者中较长的，所以决定 prefill 的耗时。':''}`);
  pre.mRow.dataset.tipx=tx(`Prefill’s memory traffic${onG}: ${gbS(r.pre.by)} at ${tbS(bw)} TB/s, so ${fPre(r.pre.tm)}${og(fPre,o.pre.tm)}.${!pre.cb?' It is the longer of the two, so it sets prefill’s time.':''}`,
    `prefill 的读写显存${onG}：${gbS(r.pre.by)}，按 ${tbS(bw)} TB/s 算，要 ${fPre(r.pre.tm)}${og(fPre,o.pre.tm)}。${!pre.cb?'它是两者中较长的，所以决定 prefill 的耗时。':''}`);
  const dec=drawPhase('dec',r.dec,fDec,{scale:o.dec.t,fade:true,ghost:diff?o.dec:null}), bd=$('.pd-ph[data-ph="dec"]',app);
  $('.pd-ph-d',bd).textContent=tx(`1 new token, ${tokK(r.N)} tokens of context`,`1 个新 token，上下文 ${tokK(r.N)} 个 token`);
  $('.pd-ph-t',bd).textContent=fDec(r.dec.t)+was(fDec,r.dec.t,o.dec.t);
  dec.cRow.dataset.tipx=tx(`A decode step’s arithmetic${onG}: ${flS(r.dec.fl)} at ${tfS(pk)} TFLOP/s, so ${fDec(r.dec.tc)}${og(fDec,o.dec.tc)}.${dec.cb?' It is the longer of the two, so it sets the time per output token.':''}`,
    `decode 步的计算${onG}：${flS(r.dec.fl)}，按 ${tfS(pk)} TFLOP/s 算，要 ${fDec(r.dec.tc)}${og(fDec,o.dec.tc)}。${dec.cb?'它是两者中较长的，所以决定每个输出 token 的耗时。':''}`);
  dec.mRow.dataset.tipx=tx(`A decode step’s memory traffic${onG}: ${gbS(r.dec.by)} at ${tbS(bw)} TB/s, so ${fDec(r.dec.tm)}${og(fDec,o.dec.tm)}.${!dec.cb?' It is the longer of the two, so it sets the time per output token.':''}`,
    `decode 步的读写显存${onG}：${gbS(r.dec.by)}，按 ${tbS(bw)} TB/s 算，要 ${fDec(r.dec.tm)}${og(fDec,o.dec.tm)}。${!dec.cb?'它是两者中较长的，所以决定每个输出 token 的耗时。':''}`);
}
function drawLim(){
  const r=R(), ch=mode==='challenge', lim=$('#pd-lim',app), rf=$('#pd-roof',app), showRoof=roofOpen&&!ch&&!isX();
  lim.hidden=showRoof; rf.hidden=!showRoof;
  if(showRoof){drawRoof(); return;}
  if(isX()){drawLimX(); return;}
  $('#pd-lim-cap',app).textContent=ch?tx('Why: a decode step’s time is the longer of its arithmetic and its memory traffic, here against the 20 ms goal','原因：decode 步的耗时取计算和读写显存中较长的那个，这里和 20 ms 的目标对比')
    :tx('Why: each step takes the longer of its arithmetic and its memory traffic','原因：每一步的耗时取计算和读写显存中较长的那个');
  const pre=drawPhase('pre',r.pre,fPre,{hide:ch});
  if(pre){const box=$('.pd-ph[data-ph="pre"]',app);
    $('.pd-ph-d',box).textContent=tx(`one step over ${tokN(r.N)}`,`一步算完 ${tokN(r.N)}`);
    pre.cRow.dataset.tipx=tx(`Prefill’s arithmetic multiplies ${fint(r.N)} prompt tokens by the layers, about 14 GFLOP each, runs the LM head once and adds attention over the prompt. That is ${flS(r.pre.fl)} at ${tfS(H100.peak*EFF.mfu)} TFLOP/s (50% of the ${GPUN}’s dense BF16 peak), so ${fPre(r.pre.tc)}.${pre.cb?' It is the longer of the two, so it sets prefill’s time.':''}`,
      `prefill 的计算：${fint(r.N)} 个 prompt token 各过一遍所有层，每个约 14 GFLOP；LM head 算一次，再加对 prompt 的 attention。一共 ${flS(r.pre.fl)}，按 ${tfS(H100.peak*EFF.mfu)} TFLOP/s（${GPUN} 稠密 BF16 峰值的 50%）算，要 ${fPre(r.pre.tc)}。${pre.cb?'它是两者中较长的，所以决定 prefill 的耗时。':''}`);
    pre.mRow.dataset.tipx=tx(`Prefill’s memory traffic reads the weights once, ${gbS(W_READ)}, and writes the prompt’s KV cache, ${fint(r.N)} × 128 KiB = ${gbS(KV_TOK*r.N)}. That is ${gbS(r.pre.by)} at ${tbS(H100.bw*EFF.mbu)} TB/s (60% of 3.35 TB/s), so ${fPre(r.pre.tm)}.`,
      `prefill 的读写显存：权重读一遍 ${gbS(W_READ)}，再写入 prompt 的 KV cache，${fint(r.N)} × 128 KiB = ${gbS(KV_TOK*r.N)}。一共 ${gbS(r.pre.by)}，按 ${tbS(H100.bw*EFF.mbu)} TB/s（3.35 TB/s 的 60%）算，要 ${fPre(r.pre.tm)}。`);}
  const masked=decMasked(), scale=ch?Math.max(0.030,r.dec.t*1.08):0;
  const dec=drawPhase('dec',r.dec,fDec,{masked,scale,goal:ch?CH.goal:0});
  const box=$('.pd-ph[data-ph="dec"]',app);
  if(dec){
    $('.pd-ph-d',box).textContent=r.B===1?tx(`1 new token, ${tokK(r.N)} tokens of context`,`1 个新 token，上下文 ${tokK(r.N)} 个 token`):tx(`1 new token for each of ${reqN(r.B)}, ${tokK(r.N)} tokens of context each`,`${reqN(r.B)}各 1 个新 token，每个上下文 ${tokK(r.N)} 个 token`);
    dec.cRow.dataset.tipx=tx(`A decode step’s arithmetic runs ${r.B===1?'1 new token':`${fint(r.B)} new tokens`} through the layers and the LM head, ${fixd(2*(P_LAYERS+P_HEAD)/1e9,1)} GFLOP per token, and adds attention over ${tokK(r.N)} tokens of context${r.B>1?' each':''}. That is ${flS(r.dec.fl)} at ${tfS(H100.peak*EFF.mfu)} TFLOP/s, so ${fDec(r.dec.tc)}.`,
      `decode 步的计算：${r.B===1?'1 个新 token 过':`${fint(r.B)} 个新 token 各过`}一遍所有层和 LM head，每个 token ${fixd(2*(P_LAYERS+P_HEAD)/1e9,1)} GFLOP，再加对${r.B>1?'各自':''} ${tokK(r.N)} 个 token 上下文的 attention。一共 ${flS(r.dec.fl)}，按 ${tfS(H100.peak*EFF.mfu)} TFLOP/s 算，要 ${fDec(r.dec.tc)}。`);
    dec.mRow.dataset.tipx=tx(`A decode step’s memory traffic reads all the weights once, ${gbS(W_READ)} (${fDec(r.dec.tw)}), and the KV cache of ${r.B===1?'its':'each request’s'} ${fint(r.N)} tokens of context, ${r.B===1?'':`${fint(r.B)} × `}${fint(r.N)} × 128 KiB = ${gbS(KV_TOK*r.N*r.B)} (${fDec(r.dec.tkv)}). That is ${gbS(r.dec.by)} at ${tbS(H100.bw*EFF.mbu)} TB/s, so ${fDec(r.dec.tm)}.${!dec.cb?' It is the longer of the two, so it sets the time per output token.':''}`,
      `decode 步的读写显存：全部权重 ${gbS(W_READ)} 读一遍（${fDec(r.dec.tw)}），再读${r.B===1?'':'每个请求'} ${fint(r.N)} 个 token 上下文的 KV cache，${r.B===1?'':`${fint(r.B)} × `}${fint(r.N)} × 128 KiB = ${gbS(KV_TOK*r.N*r.B)}（${fDec(r.dec.tkv)}）。一共 ${gbS(r.dec.by)}，按 ${tbS(H100.bw*EFF.mbu)} TB/s 算，要 ${fDec(r.dec.tm)}。${!dec.cb?'它是两者中较长的，所以决定每个输出 token 的耗时。':''}`);
  }
  if(ch){$('.pd-ph-t',box).textContent=fDec(r.dec.t)+tx(' (goal: at most 20 ms)','（目标：最多 20 ms）');}
}
/* the GPU memory the batch takes: the KV cache memory lab's bar, drawn by the kit (weights, runtime overhead, KV caches, free) */
function drawMem(){
  const box=$('#pd-mem',app); if(!box) return;
  const r=R(), show=!roofOpen&&(r.B>1||(mode==='guided'&&STEPS[step].id==='batch'))&&!decMasked();
  box.hidden=!show; if(!show||!A.kv||!A.kv.drawBar) {if(!A.kv||!A.kv.drawBar) box.hidden=true; return;}
  const m=memOf(r.N,r.B), bar=$('#pd-membar',app);
  A.kv.drawBar(bar,A.kv.perGpuBar(m),{noText:true,capw:100,perUser:m.kvPerReq});
  $('#pd-mem-cap',app).textContent=tx(`GPU memory: the weights and the KV caches of ${reqN(r.B)} (at most ${fint(r.lim)} fit at ${tokK(r.N)} tokens each)`,`显存：权重和 ${reqN(r.B)}的 KV cache（每个 ${tokK(r.N)} 个 token 时最多放得下 ${fint(r.lim)} 个）`);
  bar.dataset.tipx=tx(`The ${GPUN}’s 80 GB, read as 80 GiB: the weights take ${gib1(W_MEM)} (${gbS(W_MEM)}), the runtime overhead ${OH} GiB, and the KV caches of ${reqN(r.B)} ${gib2(m.kvPerGpu)} (${fint(r.B)} × ${fint(r.N)} × 128 KiB); the server uses at most ${pct0(UTIL)}, ${gib1(m.usable)}.`,
    `${GPUN} 的 80 GB 按 80 GiB 计：权重占 ${gib1(W_MEM)}（${gbS(W_MEM)}），运行时开销 ${OH} GiB，${reqN(r.B)}的 KV cache 占 ${gib2(m.kvPerGpu)}（${fint(r.B)} × ${fint(r.N)} × 128 KiB）；服务端最多用 ${pct0(UTIL)}，即 ${gib1(m.usable)}。`);
  bar.setAttribute('role','img'); bar.setAttribute('aria-label',bar.dataset.tipx);
  const lg=$('#pd-memleg',app), leg=(cls,t,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i>${esc(t)}</span>`;
  const h=leg('w',tx('weights','权重'),tx(`The weights stay in GPU memory: ${gbS(W_MEM)} = ${gib1(W_MEM)}, the embedding table included.`,`权重一直占着显存：${gbS(W_MEM)} = ${gib1(W_MEM)}，包括 embedding 表。`))+
    leg('oh',tx('runtime overhead','运行时开销'),A.t('tipLgOH'))+
    leg('kv',bar._ticks?tx('KV cache, one slice per request','KV cache，每个请求一段'):tx('KV caches of the requests','各请求的 KV cache'),tx(`Each request’s KV cache: ${fint(r.N)} tokens × 128 KiB = ${gib2(KV_TOK*r.N)}.`,`每个请求的 KV cache：${fint(r.N)} 个 token × 128 KiB = ${gib2(KV_TOK*r.N)}。`))+
    leg('free',tx('free','空闲'),tx('Free memory: usable GPU memory that nothing takes yet.','空闲：可用显存里还没被占用的部分。'))+
    leg('lim',tx(`dashed line: ${pct0(UTIL)} usable`,`虚线：可用 ${pct0(UTIL)}`),A.t('utilNote'));
  if(lg._h!==h){lg.innerHTML=h; lg._h=h;}
}
/* Deep dive: the roofline. Log–log: arithmetic intensity (FLOPs per byte) against the FLOP/s a step reaches */
function roofPts(){const r=R(); return {pre:r.pre,dec:r.dec,r};}
function drawRoof(){
  const box=$('#pd-roof',app); if(!box) return;
  /* the viewBox follows the panel's width, so the labels keep their size on a phone (a fixed 560 would shrink them to 6 px) */
  const cw=box.clientWidth||(box.parentNode&&box.parentNode.clientWidth)||560, W=Math.max(300,Math.min(640,Math.round(cw))), narrow=W<480;
  const {pre,dec,r}=roofPts(), H=narrow?230:260, ml=50, mr=narrow?18:14, mt=14, mb=42, x0=0, x1=5, y0=0, y1=3;
  const X=ai=>ml+(Math.log10(ai)-x0)/(x1-x0)*(W-ml-mr), Y=tf=>H-mb-(Math.log10(tf)-y0)/(y1-y0)*(H-mt-mb);
  const pk=H100.peak*EFF.mfu/1e12, bw=H100.bw*EFF.mbu/1e12, rid=pk/bw;
  let g='';
  for(let d=x0;d<=x1;d++){const x=X(10**d); g+=`<line class="pd-rg" x1="${x}" y1="${mt}" x2="${x}" y2="${H-mb}"/><text class="pd-rt" x="${x}" y="${H-mb+14}" text-anchor="${d===x1?'end':'middle'}">${narrow&&d===4?'':fint(10**d)}</text>`;}
  for(let d=y0;d<=y1;d++){const y=Y(10**d); g+=`<line class="pd-rg" x1="${ml}" y1="${y}" x2="${W-mr}" y2="${y}"/><text class="pd-rt" x="${ml-6}" y="${y+4}" text-anchor="end">${fint(10**d)}</text>`;}
  const roof=`M${X(1)},${Y(bw)} L${X(rid)},${Y(pk)} L${X(1e5)},${Y(pk)}`;
  const pP={x:X(pre.ai),y:Y(pre.fl/pre.t/1e12)}, pD={x:X(dec.ai),y:Y(dec.fl/dec.t/1e12)};
  const tip=(t)=>` tabindex="0" data-tipx="${esc(t)}"`;
  const tPre=tx(`Prefill of ${fint(r.N)} tokens does ${flS(pre.fl)} and moves ${gbS(pre.by)}, ${fint(pre.ai)} FLOPs per byte, right of the ridge. Arithmetic sets its time, at ${tfS(pre.fl/pre.t)} TFLOP/s.`,`${fint(r.N)} 个 token 的 prefill 要算 ${flS(pre.fl)}，搬运 ${gbS(pre.by)}，每字节 ${fint(pre.ai)} 个 FLOP，在拐点右边。计算决定它的耗时，达到 ${tfS(pre.fl/pre.t)} TFLOP/s。`);
  const tDec=tx(`A decode step for ${reqN(r.B)} at ${fint(r.N)} tokens does ${flS(dec.fl)} and moves ${gbS(dec.by)}, ${fixd(dec.ai,dec.ai<10?2:1)} FLOPs per byte, left of the ridge. Memory sets its time, and it reaches only ${tfS(dec.fl/dec.t)} TFLOP/s.`,`${reqN(r.B)}、每个 ${fint(r.N)} 个 token 的 decode 步要算 ${flS(dec.fl)}，搬运 ${gbS(dec.by)}，每字节 ${fixd(dec.ai,dec.ai<10?2:1)} 个 FLOP，在拐点左边。读写显存决定它的耗时，只达到 ${tfS(dec.fl/dec.t)} TFLOP/s。`);
  /* the decode point's label: above and left of it, clear of the slope and the ridge, unless the point sits at the left edge */
  const dTxt=r.B===1?tx('decode, 1 request','decode，1 个请求'):tx(`decode, ${fint(r.B)} requests`,`decode，${fint(r.B)} 个请求`), dW=[...dTxt].reduce((a,ch)=>a+(ch.codePointAt(0)>=0x2e80?11:6.8),0);
  const dLab=pD.x-12-dW>ml?`<text class="pd-rt pd-rlab" x="${pD.x-12}" y="${pD.y-9}" text-anchor="end">${esc(dTxt)}</text>`:`<text class="pd-rt pd-rlab" x="${pD.x+10}" y="${pD.y+17}">${esc(dTxt)}</text>`;
  const svg=`<svg class="pd-rsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx('Roofline: prefill right of the ridge, decode left of it.','Roofline 图：prefill 在拐点右边，decode 在左边。'))}">`+g+
    `<text class="pd-rt pd-rax" x="${(ml+W-mr)/2}" y="${H-6}" text-anchor="middle">${esc(narrow?tx('FLOPs per byte moved (log scale)','每字节的 FLOP 数（对数轴）'):tx('FLOPs per byte moved (arithmetic intensity, log scale)','每搬运 1 字节的 FLOP 数（算术强度，对数轴）'))}</text>`+
    `<text class="pd-rt pd-rax" x="12" y="${(mt+H-mb)/2}" text-anchor="middle" transform="rotate(-90 12 ${(mt+H-mb)/2})">${esc(narrow?tx('TFLOP/s reached','达到的 TFLOP/s'):tx('TFLOP/s reached (log scale)','达到的 TFLOP/s（对数轴）'))}</text>`+
    (narrow?'':`<text class="pd-rt pd-rreg" x="${X(1)+6}" y="${mt+11}">${esc(tx('← memory sets the time','← 读写显存决定耗时'))}</text>`+
    `<text class="pd-rt pd-rreg" x="${X(1e5)-4}" y="${mt+11}" text-anchor="end">${esc(tx('arithmetic sets the time →','计算决定耗时 →'))}</text>`)+
    `<g class="pd-rrg"${tip(TIP.ridge())}><line class="pd-rridge" x1="${X(rid)}" y1="${mt}" x2="${X(rid)}" y2="${H-mb}"/><line class="pd-rhit" x1="${X(rid)}" y1="${mt}" x2="${X(rid)}" y2="${H-mb}"/><text class="pd-rt pd-rlab" x="${X(rid)+5}" y="${mt+11}">${esc(tx(`ridge: ${fint(rid)}`,`拐点：${fint(rid)}`))}</text></g>`+
    `<path class="pd-rroof" d="${roof}"/>`+
    `<g class="pd-rpt"${tip(tPre)}><circle cx="${pP.x}" cy="${pP.y}" r="6"/><text class="pd-rt pd-rlab" x="${pP.x}" y="${pP.y+20}" text-anchor="middle">${esc(tx(`prefill, ${tokK(r.N)}`,`prefill，${tokK(r.N)}`))}</text></g>`+
    `<g class="pd-rpt pd-rpt-d"${tip(tDec)}><rect x="${pD.x-5.5}" y="${pD.y-5.5}" width="11" height="11"/>${dLab}</g></svg>`;
  box.innerHTML=`<div class="pd-rhead"><span class="pd-cap">${tx(`Deep dive: the roofline. The most FLOP/s a step can reach at each ${DEF('arithmetic intensity',TIP.ai())}, with this lab’s 50% and 60%`,`深入：roofline 图。每种${DEF('算术强度',TIP.ai())}下一步最多能达到的 FLOP/s（按这个 Lab 的 50% 和 60%）`)}</span><button type="button" class="btn ghost sm" data-act="roof-close">${tx('Back to the two limits','回到两种限制')}</button></div>`+svg+
    `<div class="tz-leg pd-rleg"><span><i class="pd-sw-roof"></i>${tx('line: the most FLOP/s reached','折线：最多能达到的 FLOP/s')}</span><span><i class="sw lim"></i>${tx('dashed line: the ridge','虚线：拐点')}</span><span><i class="pd-sw-dot"></i>${tx('dot: prefill','圆点：prefill')}</span><span><i class="pd-sw-sq"></i>${tx('square: a decode step','方块：decode 步')}</span></div>`+
    /* the deep dive's point in words (R6.20); each point's tooltip and Show the math give the exact intensities */
    `<p class="stage-note">${tx('Prefill multiplies each parameter it reads by every prompt token: many FLOPs per byte, right of the ridge. A decode step multiplies each parameter by one token per request: few FLOPs per byte, left of the ridge. Batching moves it right.','prefill 读进来的每个参数都要和每个 prompt token 相乘：每字节的 FLOP 很多，在拐点右边。decode 步只让它和每个请求的一个 token 相乘：每字节的 FLOP 很少，在拐点左边。攒 batch 会让它往右移。')}</p>`;
}
/* metric cards: ▲▼ only after the learner's own change, from the numbers as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
const unitOf=v=>{const m=String(v).match(/(ms|s)$/); return m?m[1]:'';};
const toBase=(v,u)=>({ms:v/1000,s:v}[u]!=null?{ms:v/1000,s:v}[u]:v);
/* part 6: a card's change against the original H100, from the values as shown (R6.6): ▼ and the ratio of the two times */
function xCmp(a,b){
  if(a===b) return tx('same as the original','和原版一样');
  if(unitOf(a)!==unitOf(b)) return '';
  return A.fmt.delta(numOf(a),numOf(b),{dp:Math.max(decOf(a),decOf(b)),base:{en:'the original',zh:'原版'}});
}
const xDiv=(a,b)=>{const x=shownS(a), y=shownS(b), u=v=>unitOf(v), same=u(a)===u(b), ms=v=>u(v)==='ms'?v:fint(Math.round(shownS(v)*1e3))+NS+'ms';
  return `${same?b:ms(b)} ÷ ${same?a:ms(a)} = ${fixd(y/x,2)}`;};
function metricsX(){
  const o=xRun('orig',xl), r=xRun(xv,xl), diff=xv!=='orig'&&!!xpred, nm=xName(xv);
  const here=(F,v,w)=>diff?tx(` Here: ${nm}, ${F(v)}; the original ${GPUN} takes ${F(w)}, and ${xDiv(F(w),F(v))}.`,`这里：${nm}，${F(v)}；原版 ${GPUN} 要 ${F(w)}，${xDiv(F(w),F(v))}。`)
    :tx(` Here: the ${GPUN}, ${F(v)}.`,`这里：${GPUN}，${F(v)}。`);
  return [
   {k:'ttft',label:tx('Time to first token (TTFT)','首 token 延迟（TTFT）'),v:xFt(r.ttft),sub:tx(`prefill of ${tokN(r.N)}, no queue`,`${tokN(r.N)}的 prefill，不含排队`),cmp:diff?xCmp(xFt(o.ttft),xFt(r.ttft)):'',
    tip:TIP.ttft()+tx(' ','')+here(xFt,r.ttft,o.ttft)},
   {k:'tpot',label:tx('Time per output token (TPOT)','每个输出 token 的耗时（TPOT）'),v:fDec(r.tpot),sub:tx('the first decode step','第一个 decode 步'),cmp:diff?xCmp(fDec(o.tpot),fDec(r.tpot)):'',
    tip:TIP.tpot()+tx(` The first decode step after the prompt, at ${fint(r.N)} tokens of context.`,`取的是 prompt 之后的第一个 decode 步（上下文 ${fint(r.N)} 个 token）。`)+here(fDec,r.tpot,o.tpot)},
   {k:'e2e',label:tx('Whole answer','整个回答'),v:xWa(r.e2e),sub:tokN(r.M),cmp:diff?xCmp(xWa(o.e2e),xWa(r.e2e)):'',
    tip:tx(`The whole ${fint(r.M)}-token answer: the prefill, ${xFt(r.ttft)}, plus ${fint(r.M-1)} decode steps of ${fDec(r.avg)} on average, as the context grows from ${fint(r.N)} to ${fint(r.N+r.M-2)} tokens: ${xWa(r.e2e)} in all.`,`整个 ${fint(r.M)} 个 token 的回答：prefill ${xFt(r.ttft)}，加上 ${fint(r.M-1)} 个 decode 步（上下文从 ${fint(r.N)} 个 token 涨到 ${fint(r.N+r.M-2)} 个，平均每步 ${fDec(r.avg)}），一共 ${xWa(r.e2e)}。`)+
      (diff?tx(` The original ${GPUN} takes ${xWa(o.e2e)}, and ${xDiv(xWa(o.e2e),xWa(r.e2e))}.`,`原版 ${GPUN} 要 ${xWa(o.e2e)}，${xDiv(xWa(o.e2e),xWa(r.e2e))}。`):'')}];
}
function metricsDef(){
  if(isX()) return metricsX();
  const r=R(), hid=decMasked(), q=v=>hid?'?':v, later=tx('shown in part 4','第 4 部分显示');
  return [
   {k:'ttft',label:tx('Time to first token (TTFT)','首 token 延迟（TTFT）'),v:fPre(r.ttft),sub:tx(`prefill of ${tokN(r.N)}, no queue`,`${tokN(r.N)}的 prefill，不含排队`),
    tip:TIP.ttft()+tx(' ','')+tx(`Here: prefill of ${fint(r.N)} tokens, ${fPre(r.ttft)}, set by ${r.pre.cb?'arithmetic':'memory traffic'}.`,`这里：${fint(r.N)} 个 token 的 prefill，${fPre(r.ttft)}，由${r.pre.cb?'计算':'读写显存'}决定。`)},
   {k:'tpot',label:tx('Time per output token (TPOT)','每个输出 token 的耗时（TPOT）'),v:q(fDec(r.tpot)),sub:hid?later:(r.B===1?tx('the first decode step','第一个 decode 步'):tx(`the first decode step, ${reqN(r.B)}`,`第一个 decode 步，${reqN(r.B)}`)),hid,
    tip:hid?tx('Hidden until part 4, where it answers your prediction.','第 4 部分揭晓你的预测之前先隐藏。'):TIP.tpot()+tx(' ','')+tx(`Here: the first decode step after the prompt, at ${fint(r.N)} tokens of context${r.B>1?`, for ${reqN(r.B)}`:''}, ${fDec(r.tpot)}.`,`这里：prompt 之后的第一个 decode 步，上下文 ${fint(r.N)} 个 token${r.B>1?`，${reqN(r.B)}`:''}，${fDec(r.tpot)}。`)},
   {k:'tps',label:tx('Tokens per second, each request','每个请求每秒的 token'),v:q(fTps(r.tps)),sub:hid?later:tx('the inverse of TPOT','TPOT 的倒数'),hid,
    tip:hid?tx('Hidden until part 4, where it answers your prediction.','第 4 部分揭晓你的预测之前先隐藏。'):tx('How many tokens per second each request gets once its answer has started.','回答开始以后，每个请求每秒拿到几个 token。')+FXT(tx('\\text{tokens per second, each request} = \\frac{1}{\\text{TPOT}}','\\text{每个请求每秒的 token} = \\frac{1}{\\text{TPOT}}'))+tx(`Here: 1 ÷ ${fDec3(r.tpot)} = ${fTps(r.tps)}.`,`这里：1 ÷ ${fDec3(r.tpot)} = ${fTps(r.tps)}。`)},
   {k:'thr',label:tx('Tokens per second, all requests','全部请求每秒的 token'),v:q(fThr(r.thr)),sub:hid?later:tx(`decode only, ${reqN(r.B)}`,`只算 decode，${reqN(r.B)}`),hid,
    tip:hid?tx('Hidden until part 4, where it answers your prediction.','第 4 部分揭晓你的预测之前先隐藏。'):TIP.thr()+tx(' ','')+tx(`Here: ${fint(r.B)} ÷ ${fDec3(r.tpot)} = ${fThr(r.thr)}.`,`这里：${fint(r.B)} ÷ ${fDec3(r.tpot)} = ${fThr(r.thr)}。`)},
   {k:'e2e',label:tx('Whole answer','整个回答'),v:q(fTot(r.e2e)),sub:hid?later:tokN(r.M),hid,
    tip:hid?tx('Hidden until part 4, where it answers your prediction.','第 4 部分揭晓你的预测之前先隐藏。'):tx(`The whole ${fint(r.M)}-token answer takes the prefill, ${fPre(r.ttft)}, plus ${fint(r.M-1)} decode steps of ${fDec(r.avg)} on average, as the context grows from ${fint(r.N)} to ${fint(r.N+r.M-2)} tokens: ${fTot(r.e2e)} in all.`,`整个 ${fint(r.M)} 个 token 的回答要多久：prefill ${fPre(r.ttft)}，加上 ${fint(r.M-1)} 个 decode 步（上下文从 ${fint(r.N)} 个 token 涨到 ${fint(r.N+r.M-2)} 个，平均每步 ${fDec(r.avg)}），一共 ${fTot(r.e2e)}。`)}].filter(m=>!(mode==='challenge'&&m.k==='e2e'));   // the challenge is about the time per output token at 4K only
}
function drawMetrics(){
  const host=$('#pd-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip; el.classList.toggle('pd-hid',!!m.hid);
    /* part 6: the change against the original stays on the card while that GPU is picked (it is a comparison, not a ▲▼ after a change) */
    if(m.cmp!=null){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=m.cmp; d.classList.toggle('on',!!m.cmp); shownPrev[m.k]=m.v; return;}
    let dt='';
    const prev=shownPrev[m.k];
    if(userAct&&!m.hid&&prev!=null&&prev!==m.v){const ua=unitOf(prev), ub=unitOf(m.v), same=ua===ub, a=same?numOf(prev):toBase(numOf(prev),ua), b=same?numOf(m.v):toBase(numOf(m.v),ub);
      if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,same?{dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.hid?null:m.v;});
}
/* Show the math: every line redoes from the numbers it shows (R6.6), and every term has its own tooltip (R1.4) */
/* part 6's Show the math: the two limits of each phase on the GPU picked, every line from the numbers it shows (R6.6) */
/* H1/H2 (the owner, 2026-10-09): Show the math as short blocks, like a page of a good textbook. Each block has one lead line in words,
   then a small table: the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the
   result, each on its own line with the "=" signs lined up. groups: [{name, terms:[TeX per column], rows:[{l, c:[html per column]} | {r:html}]}] */
const MX=x=>A.ui.math(x);
function mathBlock(lead,groups,note){const n=Math.max(...groups.map(g=>g.terms.length)), eq=`<td class="pd-meq">${MX('=')}</td>`;
  const cells=(a,f)=>a.map((x,i)=>`<td${i===a.length-1&&a.length<n?` colspan="${n-i}"`:''}>${f(x)}</td>`).join('');
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?cells(r.c,x=>x):`<td colspan="${n}" class="pd-mres">${r.r}</td>`}</tr>`;
  const body=groups.map(g=>`<tr class="pd-mname"><th scope="row">${g.name}</th>${eq}${cells(g.terms,MX)}</tr>${g.rows.map(tr).join('')}`).join('');
  return `<div class="pd-mb">${lead?`<p class="pd-mlead">${lead}</p>`:''}<div class="pd-mfx"><table class="pd-mt">${body}</table></div>${note?`<p class="pd-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.pd-mt'); if(!tb) return; const e0=tb.querySelector('td.pd-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('pd-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('pd-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="pd-mc${r?'':' pd-mc0'}">${x}</span>`;
    h+=`<tr class="pd-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="pd-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="pd-mband${g.tall?' pd-mtall':''}"><td class="pd-meq"><span class="pd-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="pd-mfl">${units.map(u=>`<span class="pd-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="pd-meq">${E}</td><td class="pd-mres">${r.l?`<span class="pd-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='pd-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.pd-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.pd-mfx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
/* one step's time: compute time and memory time, the longer of the two (the GPU does both at once) */
function stepMath(lead,o,flDesc,byDesc,ce,be,ceN,beN,dp,resName,resV){
  const TF=v=>fixd(v/1e12,v<1e12?4:3)+NS+'TFLOP', GB=v=>fixd(v/1e9,3)+NS+DEF('GB',TIP.gb());
  return mathBlock(lead,[
    {name:tx('Compute time','计算时间'),terms:[tx('\\text{FLOP}','\\text{运算量}'),'\\div',tx('\\text{FLOP/s reached}','\\text{能达到的 FLOP/s}')],
      rows:[{r:`(${flDesc}) ÷ ${ce}`},{c:[TF(o.fl),'÷',ceN]},{r:`${fixd(o.tc*1e3,dp)}${NS}ms`}]},
    {name:tx('Memory time','读写显存时间'),terms:[tx('\\text{bytes moved}','\\text{搬运的字节数}'),'\\div',tx('\\text{bandwidth reached}','\\text{能达到的带宽}')],
      rows:[{r:`(${byDesc}) ÷ ${be}`},{c:[GB(o.by),'÷',beN]},{r:`${fixd(o.tm*1e3,dp)}${NS}ms`}]},
    {name:`${resName} (${DEF(tx('step time','这一步的耗时'),TIP.overlap())})`,terms:[tx('\\max(\\text{compute time},\\ \\text{memory time})','\\max(\\text{计算时间},\\ \\text{读写显存时间})')],
      rows:[{c:[`max(${fixd(o.tc*1e3,dp)}${NS}ms, ${fixd(o.tm*1e3,dp)}${NS}ms)`]},{r:`<b>${resV}</b>${o.cb?tx(', set by arithmetic','，由计算决定'):tx(', set by memory traffic','，由读写显存决定')}`}]}]);}
const sizeMath=t=>mathBlock(tx('Every step reads all the weights once.','每一步都要把全部权重读一遍。'),
    [{name:tx('Weights read per step','每一步读的权重'),terms:[tx('\\text{parameters}','\\text{参数数}'),'\\times',tx('\\text{bytes per parameter}','\\text{每个参数的字节数}')],
      rows:[{c:[t('wr',fint(P_LAYERS+P_HEAD)),'×','2']},{r:`${fint(W_READ)} ${tx('bytes','字节')} (<b>${gbS(W_READ)}</b>)`}]}])+
  mathBlock(tx('Every token keeps a key and a value for each KV head in every layer, 2 bytes per number in BF16.','每个 token 在每一层、每个 KV head 上存一个 key 和一个 value，BF16 下每个数值占 2 字节。'),
    [{name:tx('KV cache per token','每个 token 的 KV cache'),terms:['2','\\times',tx('\\text{layers}','\\text{层数}'),'\\times',tx('\\text{KV heads}','\\text{KV head 数}'),'\\times','\\text{head_dim}','\\times',tx('\\text{bytes per number}','\\text{每个数值的字节数}')],
      rows:[{c:[t('kv','2'),'×','32','×','8','×','128','×','2']},{r:`131,072 ${tx('bytes','字节')} (<b>128${NS}${DEF('KiB',TIP.kib())}</b>)`}]}]);
function mathX(){
  const r=xRun(xv,xl), o=xRun('orig',xl), t=T_, N=r.N, pk=r.g.peak*EFF.mfu, bw=r.g.bw*EFF.mbu, nm=xName(xv);
  const WL=t('wl','2 × '+fint(P_LAYERS)), WH=t('wh','2 × '+fint(P_HEAD)), ceN=tfS(pk)+NS+'TFLOP/s', beN=tbS(bw)+NS+'TB/s', ce=t(xv==='c2'?'xpk':'effc',ceN), be=t(xv==='b2'?'xbw':'effm',beN);
  const rows=[];
  rows.push(mathBlock(tx(`<b>${nm}</b>: what the GPU reaches, at this lab’s efficiencies.`,`<b>${nm}</b>：按这个 Lab 的效率，GPU 能达到多少。`),
    [{name:tx('FLOP/s reached','能达到的 FLOP/s'),terms:[tx('\\text{peak FLOP/s}','\\text{峰值 FLOP/s}'),'\\times','50\\%'],rows:[{c:[`${xv==='c2'?'2 × ':''}${fixd(H100.peak/1e12,1)}${NS}TFLOP/s`,'×','50%']},{r:ce}]},
     {name:tx('Bandwidth reached','能达到的带宽'),terms:[tx('\\text{peak bandwidth}','\\text{峰值带宽}'),'\\times','60\\%'],rows:[{c:[`${xv==='b2'?'2 × ':''}${fixd(H100.bw/1e12,2)}${NS}TB/s`,'×','60%']},{r:be}]}]));
  rows.push(stepMath(tx(`<b>Prefill of ${fint(N)} tokens.</b> The GPU computes and moves memory at the same time, so the step takes the longer of the two.`,`<b>${fint(N)} 个 token 的 prefill。</b>GPU 计算和读写显存同时进行，所以这一步取两者中较长的那个。`),r.pre,
    `${WL} × ${t('N',fint(N))} + ${WH} + ${t('attpre',`2 × 32 × 4,096 × ${fint(N)} × ${fint(N+1)}`)}`,`${t('wr2',fint(W_READ))} + ${t('N',fint(N))} × ${t('kvt','131,072')}`,ce,be,ceN,beN,2,'TTFT',xFt(r.ttft)));
  rows.push(stepMath(tx(`<b>One decode step at ${fint(N)} tokens of context.</b>`,`<b>一个 decode 步，上下文 ${fint(N)} 个 token。</b>`),r.dec,
    `${WL} + ${WH} + ${t('attdec',`4 × 32 × 4,096 × ${fint(N+1)}`)}`,`${t('wr2',fint(W_READ))} + ${t('N',fint(N))} × ${t('kvt','131,072')}`,ce,be,ceN,beN,3,'TPOT',fDec(r.tpot)));
  /* the prefill plus the decode steps' total, both as shown, so the sum redoes by hand (R6.6); the steps and their average in brackets */
  const tS=Math.round(r.ttft*1e5)/100, dS=Math.round(r.dsum*1e5)/100;
  rows.push(mathBlock(tx('The whole answer is the prefill plus every decode step after it.','整个回答的耗时是 prefill 加上之后所有的 decode 步。'),
    [{name:tx('Whole answer','整个回答'),terms:['\\text{TTFT}','+',tx('\\text{all decode steps}','\\text{所有 decode 步}')],
      rows:[{c:[`${fixd(tS,2)}${NS}ms`,'+',`${fixd(dS,2)}${NS}ms${tx(` (${t('M',fint(r.M-1))} steps)`,`（${t('M',fint(r.M-1))} 步）`)}`]},{r:`${fixd(tS+dS,2)}${NS}ms (<b>${xWa(r.e2e)}</b>)`}]}]));
  if(xv!=='orig') rows.push(`<div class="pd-mb"><p class="pd-mlead">${tx(`Against the original ${GPUN}`,`和原版 ${GPUN} 比`)}</p><p class="pd-mvs">TTFT ${xDiv(xFt(o.ttft),xFt(r.ttft))}<br>TPOT ${xDiv(fDec(o.tpot),fDec(r.tpot))}<br>${tx('whole answer ','整个回答 ')}${xDiv(xWa(o.e2e),xWa(r.e2e))}</p></div>`);
  return rows.join('');
}
function mathHTML(){
  if(isX()) return mathX();
  const r=R(), t=T_, N=r.N, B=r.B, pk=H100.peak*EFF.mfu, bw=H100.bw*EFF.mbu, hid=decMasked();
  const TF=v=>fixd(v/1e12,v<1e12?4:3)+NS+'TFLOP', GB=v=>fixd(v/1e9,3)+NS+DEF('GB',TIP.gb());
  const rows=[sizeMath(t)];
  const WL=t('wl','2 × '+fint(P_LAYERS)), WH=t('wh','2 × '+fint(P_HEAD)), ceN=tfS(pk)+NS+'TFLOP/s', beN=tbS(bw)+NS+'TB/s', ce=t('effc',ceN), be=t('effm',beN);
  rows.push(stepMath(tx(`<b>Prefill of ${fint(N)} tokens.</b> The GPU computes and moves memory at the same time, so the step takes the longer of the two.`,`<b>${fint(N)} 个 token 的 prefill。</b>GPU 计算和读写显存同时进行，所以这一步取两者中较长的那个。`),r.pre,
    `${WL} × ${t('N',fint(N))} + ${WH} + ${t('attpre',`2 × 32 × 4,096 × ${fint(N)} × ${fint(N+1)}`)}`,`${t('wr2',fint(W_READ))} + ${t('N',fint(N))} × ${t('kvt','131,072')}`,ce,be,ceN,beN,2,'TTFT',fPre(r.ttft)));
  if(hid) rows.push(`<p class="muted">${tx('The decode step’s lines appear in part 4, where they answer your prediction.','decode 步的算式在第 4 部分出现，那里揭晓你的预测。')}</p>`);
  else {
    rows.push(stepMath(tx(`<b>One decode step, ${reqN(B)} at ${fint(N)} tokens of context.</b>${B>1?' The weights are read once for all of them.':''}`,`<b>一个 decode 步，${reqN(B)}，上下文 ${fint(N)} 个 token。</b>${B>1?'权重只为它们一起读一遍。':''}`),r.dec,
      `${t('B',fint(B))} × (${WL} + ${WH}) + ${t('B',fint(B))} × ${t('attdec',`4 × 32 × 4,096 × ${fint(N+1)}`)}`,`${t('wr2',fint(W_READ))} + ${t('B',fint(B))} × ${t('N',fint(N))} × ${t('kvt','131,072')}`,ce,be,ceN,beN,3,'TPOT',fDec(r.tpot)));
    rows.push(mathBlock(tx('Once the answer has started, each request gets one token per decode step.','回答开始以后，每个请求每个 decode 步拿到一个 token。'),
      [{name:tx('Tokens per second, each request','每个请求每秒的 token'),terms:['\\frac{1}{\\text{TPOT}}'],rows:[{c:[`1 ÷ ${fDec3(r.tpot)}`]},{r:`<b>${fTps(r.tps)}</b>`}]},
       {name:tx('Tokens per second, all requests','全部请求每秒的 token'),terms:[tx('\\frac{\\text{batch size}}{\\text{TPOT}}','\\frac{\\text{batch 大小}}{\\text{TPOT}}')],rows:[{c:[`${t('B',fint(B))} ÷ ${fDec3(r.tpot)}`]},{r:`<b>${fThr(r.thr)}</b>`}]}]));
    /* the challenge checks the goal at 4K tokens of context: an answer's growing context (its later steps) is not part of it */
    if(mode!=='challenge') rows.push(mathBlock(tx('The whole answer is the prefill plus every decode step after it, which grow a little slower as the context grows.','整个回答的耗时是 prefill 加上之后所有的 decode 步。上下文越长，decode 步越慢一点。'),
      [{name:tx('Whole answer','整个回答'),terms:['\\text{TTFT}','+',tx('\\text{decode steps}','\\text{decode 步数}'),'\\times',tx('\\text{mean step}','\\text{平均每步}')],
        rows:[{c:[`${fixd(r.ttft*1e3,2)}${NS}ms`,'+',t('M',fint(r.M-1)),'×',t('avg',fixd(r.avg*1e3,3)+NS+'ms')]},{r:`${fixd(r.e2e*1e3,1)}${NS}ms (<b>${fTot(r.e2e)}</b>)`}]}]));
    rows.push(mathBlock(tx(`${DEF('Arithmetic intensity',TIP.ai())} tells whether arithmetic or memory sets a step’s time. Above the ${DEF('ridge',TIP.ridge())}, arithmetic does.`,`${DEF('算术强度',TIP.ai())}决定一步的耗时由计算还是读写显存决定。高于${DEF('拐点',TIP.ridge())}时，由计算决定。`),
      [{name:tx('Arithmetic intensity','算术强度'),terms:[tx('\\frac{\\text{FLOP}}{\\text{bytes}}','\\frac{\\text{FLOP}}{\\text{字节}}')],
        rows:[{l:'prefill',c:[`${TF(r.pre.fl)} ÷ ${GB(r.pre.by)}`]},{r:`<b>${fint(r.pre.ai)}</b>`},{l:'decode',c:[`${TF(r.dec.fl)} ÷ ${GB(r.dec.by)}`]},{r:`<b>${fixd(r.dec.ai,2)}</b>`}]},
       {name:tx('Ridge','拐点'),terms:[tx('\\frac{\\text{FLOP/s reached}}{\\text{bandwidth reached}}','\\frac{\\text{能达到的 FLOP/s}}{\\text{能达到的带宽}}')],rows:[{c:[`${ce} ÷ ${be}`]},{r:`<b>${fint(RIDGE)}</b>`}]}]));
    const m=memOf(N,1), budget=m.usable-W_MEM-m.oh;
    rows.push(mathBlock(tx('Every request’s KV cache must fit in the GPU memory the weights leave.','每个请求的 KV cache 都要放进权重之外的显存。'),
      [{name:tx('Requests that fit','能放下的请求数'),terms:[tx('\\frac{\\text{memory} \\times \\text{share used} - \\text{weights} - \\text{overhead}}{\\text{tokens} \\times \\text{KV cache per token}}','\\frac{\\text{显存} \\times \\text{可用比例} - \\text{权重} - \\text{开销}}{\\text{token 数} \\times \\text{每个 token 的 KV cache}}')],
        rows:[{c:[`(${t('mem','80 GiB')} × ${t('util',pct0(UTIL))} − ${t('wmem',gib2(W_MEM))} − ${t('oh',OH+NS+'GiB')}) ÷ (${t('N',fint(N))} × 128 KiB)`]},{c:[`${gib2(budget)} ÷ ${gib2(KV_TOK*N)}`]},{r:`<b>${fint(r.lim)}</b>`}]}]));
  }
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){S.N=CH.N; S.M=CH.M; if(roofOpen) roofOpen=false;}
  S.B=Math.max(1,Math.min(S.B,memLimit(S.N)));
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const gr=$('#pd-graph',app); gr.dataset.focus=mode==='guided'?(STEPS[step].focus||''):(mode==='challenge'?'dec':'');
  drawTL(); drawLim(); drawMem(); drawMetrics();
  const lg=$('#pd-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#pd-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;} fitMath(mb);
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  const r=R(), lim=r.lim;
  $$('input[data-key="N"]',app).forEach(i=>{const v=N_STOPS.indexOf(S.N); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=tx(`${tokK(S.N)} tokens`,`${tokK(S.N)} 个 token`); i.setAttribute('aria-valuetext',tx(`${fint(S.N)} tokens`,`${fint(S.N)} 个 token`));});
  $$('input[data-key="M"]',app).forEach(i=>{const v=M_STOPS.indexOf(S.M); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=tx(`${fint(S.M)} tokens`,`${fint(S.M)} 个 token`); i.setAttribute('aria-valuetext',tx(`${fint(S.M)} tokens`,`${fint(S.M)} 个 token`));});
  $$('input[data-key="B"],input[data-key="chB"]',app).forEach(i=>{const ch=i.dataset.key==='chB';
    if(+i.max!==lim) i.max=lim; if(+i.value!==S.B) i.value=S.B; A.ui.setRange(i);
    const w=i.closest('.pd-bctl'), o=w.querySelector('output'); o.textContent=ch?userN(S.B):reqN(S.B); i.setAttribute('aria-valuetext',o.textContent);
    w.querySelector('[data-act="bminus"]').disabled=S.B<=1; w.querySelector('[data-act="bplus"]').disabled=S.B>=lim;
    const note=w.querySelector('.pd-bnote');
    note.innerHTML=ch?tx(`From 1 to ${fint(lim)}: at 4K tokens each, GPU memory holds the KV caches of at most ${fint(lim)} users.`,`从 1 到 ${fint(lim)}：每人 4K 个 token 时，显存最多放得下 ${fint(lim)} 个用户的 KV cache。`)
      :tx(`At most ${fint(lim)} at ${tokK(S.N)} tokens each: every request’s KV cache must fit in the memory the weights leave (the ${labLink('kv-cache')} lab’s model).`,`每个 ${tokK(S.N)} 个 token 时最多 ${fint(lim)} 个：每个请求的 KV cache 都要放进权重之外的显存（按 ${labLink('kv-cache')}这个 Lab 的模型）。`);
    note.classList.toggle('warn',clampNote&&S.B===lim);});
  const lock=mode==='challenge'||isX(), pl=preLock(), c=$('#pd-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#pd-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=isX()?tx(`In this experiment you change only the GPU and the load, in the card. Everything else is fixed: ${MODEL} in BF16, one user, the same 50% and 60% efficiencies on every GPU.`,`这个实验里只能在卡片里换 GPU 和负载。其余设置固定：BF16 的 ${MODEL}，一个用户，每张 GPU 都按同样的 50% 和 60% 效率算。`)
    :tx(`In the challenge you can change only the number of users, in the task card. Everything else is fixed: ${MODEL} on one ${GPUN}, 4K tokens of context each.`,`挑战中只能在任务卡里改用户数。其余设置固定：一张 ${GPUN} 跑 ${MODEL}，每人 4K 个 token 的上下文。`);
  const nw=$('#pd-ctl-N-wrap',app); if(nw) nw.hidden=pl||lock; const nl=$('#pd-Nlock',app); if(nl) nl.hidden=!pl||lock;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const r=R();
  if(isX()){const x=xRun(xv,xl); A.announce(tx(`${xName(xv)}: first token after ${fPre(x.ttft)}, then one every ${fDec(x.tpot)}; the whole answer takes ${fTot(x.e2e)}.`,`${xName(xv)}：${fPre(x.ttft)} 后出首 token，之后每 ${fDec(x.tpot)} 一个；整个回答要 ${fTot(x.e2e)}。`)); return;}
  A.announce(decMasked()?tx(`Prompt of ${fint(r.N)} tokens: first token after ${fPre(r.ttft)}.`,`prompt ${fint(r.N)} 个 token：${fPre(r.ttft)} 后出首 token。`)
    :tx(`Prompt of ${fint(r.N)} tokens, ${reqN(r.B)}: first token after ${fPre(r.ttft)}, then one every ${fDec(r.tpot)}.`,`prompt ${fint(r.N)} 个 token，${reqN(r.B)}：${fPre(r.ttft)} 后出首 token，之后每 ${fDec(r.tpot)} 一个。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  if(preLock()&&p.N!=null) p={...p,N:S.N};
  const was=S.B; Object.assign(S,p);
  if(!N_STOPS.includes(S.N)) S.N=BASE.N; if(!M_STOPS.includes(S.M)) S.M=BASE.M;
  if(mode==='challenge'){S.N=CH.N; S.M=CH.M;}
  const lim=memLimit(S.N); S.B=Math.max(1,Math.min(Math.round(S.B)||1,lim));
  clampNote=p.N!=null&&was>lim&&S.B===lim;
  render();
}
/* a change the learner made: metric cards may show a ▲▼ badge */
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  xv='orig'; xl='chat';                                          // part 6's GPU and load exist only in that part (its enter sets them again)
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  S.B=Math.max(1,Math.min(S.B,memLimit(S.N)));
  const st=$('#pd-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg={prefill:'#pd-ic-N',decode:'#pd-ic-N',batch:'#pd-ic-B',double:'#pd-ic-xv'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#pd-rail .step.current',app)||$('#pd-rail',app));}
/* the result of a part comes into view after a click that produced it: scroll just enough to show all of it (R3.10) */
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* after a click in the rail, keep the sticky stage's head below the top bar (r12 N4), as labs/kv-basics.js does */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#pd-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#pd-rail .step.current',app)||$('#pd-rail .railcard',app)||$('#pd-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#pd-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#pd-insight',app)); else revealBox($('#pd-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
/* part 6: after a pick in the card, bring its result into view on a wide screen, never so far that the option just clicked leaves the
   screen; on a phone the page does not move (the three timelines and the rows that pick a GPU are on the first screen) */
function xKeep(ctl){requestAnimationFrame(()=>{if(!app||innerWidth<=760) return; const ob=$('#pd-rail [data-obs]',app), box=$('#pd-rail .step.current .stepnav',app)||ob; if(!ob||ob.hidden||!box||!ctl.isConnected) return;   // the result and the card's Next (review A8)
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), safe=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:52)+8;
  const r=box.getBoundingClientRect(), c=ctl.getBoundingClientRect(); if(r.bottom<=innerHeight-8) return;
  const dy=Math.min(r.bottom-innerHeight+16,c.top-safe); if(dy>=2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});});}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='guided'&&STEPS[step].id==='double'){xv='orig'; xl='chat'; S={...BASE};}   // part 6's variants never reach Scenarios or the challenge
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; roofOpen=false; S={N:CH.N,M:CH.M,B:CH.start};}
  if(m==='sandbox') sbTouched=false;
  A.shell.setMode(app,m,DESCS());
  const st=$('#pd-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.N,s.M,s.B].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==3) return null;
  const [N,M,B]=p.map(Number);
  if(!N_STOPS.includes(N)||!M_STOPS.includes(M)||!Number.isInteger(B)||B<1||B>memLimit(N)) return null;
  return {N,M,B};
}

/* =====================================================================
   interview drill: seeded, from the engine; every line redoes from the numbers it shows (R6.6)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DRILL_N=[1024,8192,16384], DRILL_B=[4,8,16,32];   // always a batch, and never 4K: no drill gives away the prediction or the challenge
function makeDrill(seed){const rnd=mulberry(seed*7919+17), pick=a=>a[Math.floor(rnd()*a.length)];
  const N=pick(DRILL_N); let B=pick(DRILL_B); B=Math.min(B,memLimit(N));
  return {seed,N,B,pre:prefill(N),dec:decode(B,N)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#pd-drill-q',app), a=$('#pd-drill-a',app);
  const others=d.B-1;
  qd.innerHTML=tx(`<b>${MODEL}</b> runs in BF16 on one ${GPUN}, at 50% of its 989.5 TFLOP/s dense peak and 60% of its 3.35 TB/s bandwidth. A request brings ${an(fint(d.N))} <b>${fint(d.N)}</b>-token prompt${others?`, and its answer decodes together with <b>${fint(others)}</b> other request${others===1?'':'s'} at the same context`:''}. Estimate its time to first token and its time per output token, and say which limit sets each.`,
                  `<b>${MODEL}</b> 用 BF16 跑在一张 ${GPUN} 上，按稠密峰值 989.5 TFLOP/s 的 50%、带宽 3.35 TB/s 的 60% 计。一个请求带来 <b>${fint(d.N)}</b> 个 token 的 prompt${others?`，它的回答和另外 <b>${fint(others)}</b> 个上下文一样长的请求一起 decode`:''}。估算它的首 token 延迟和每个输出 token 的耗时，并说出各自受哪个限制。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const pk=H100.peak*EFF.mfu, bw=H100.bw*EFF.mbu, p=d.pre, q=d.dec;
  a.innerHTML=tx(`<p><b>TTFT ≈ ${fPre(p.t)} (${p.cb?'arithmetic':'memory'}); TPOT ≈ ${fDec(q.t)} (${q.cb?'arithmetic':'memory'}).</b></p>`+
    `<p>Per step the GPU reads 2 × (6,979,588,096 + 525,336,576) = ${fint(W_READ)} bytes of weights; the layers cost 2 × 6,979,588,096 FLOPs per token and the LM head 2 × 525,336,576 per new token.</p>`+
    `<p>Prefill: arithmetic = 13,959,176,192 × ${fint(d.N)} + 1,050,673,152 + 2 × 32 × 4,096 × ${fint(d.N)} × ${fint(d.N+1)} = ${fint(p.fl)} FLOP ÷ ${tfS(pk)} TFLOP/s = ${fixd(p.tc*1e3,2)} ms; memory = ${fint(W_READ)} + ${fint(d.N)} × 131,072 = ${fint(p.by)} bytes ÷ ${tbS(bw)} TB/s = ${fixd(p.tm*1e3,2)} ms. The longer, ${p.cb?'arithmetic':'memory'}, is the TTFT.</p>`+
    `<p>Decode step for ${fint(d.B)}: arithmetic = ${fint(d.B)} × 15,009,849,344 + ${fint(d.B)} × 4 × 32 × 4,096 × ${fint(d.N+1)} = ${fint(q.fl)} FLOP ÷ ${tfS(pk)} TFLOP/s = ${fixd(q.tc*1e3,3)} ms; memory = ${fint(W_READ)} + ${fint(d.B)} × ${fint(d.N)} × 131,072 = ${fint(q.by)} bytes ÷ ${tbS(bw)} TB/s = ${fixd(q.tm*1e3,2)} ms. The longer, ${q.cb?'arithmetic':'memory'}, is the TPOT.</p>`,
    `<p><b>TTFT ≈ ${fPre(p.t)}（${p.cb?'计算':'读写显存'}决定）；TPOT ≈ ${fDec(q.t)}（${q.cb?'计算':'读写显存'}决定）。</b></p>`+
    `<p>每一步 GPU 读 2 ×（6,979,588,096 + 525,336,576）= ${fint(W_READ)} 字节的权重；每个 token 过各层要 2 × 6,979,588,096 FLOP，每个新 token 的 LM head 要 2 × 525,336,576 FLOP。</p>`+
    `<p>prefill：计算 = 13,959,176,192 × ${fint(d.N)} + 1,050,673,152 + 2 × 32 × 4,096 × ${fint(d.N)} × ${fint(d.N+1)} = ${fint(p.fl)} FLOP ÷ ${tfS(pk)} TFLOP/s = ${fixd(p.tc*1e3,2)} ms；读写显存 = ${fint(W_READ)} + ${fint(d.N)} × 131,072 = ${fint(p.by)} 字节 ÷ ${tbS(bw)} TB/s = ${fixd(p.tm*1e3,2)} ms。较长的${p.cb?'计算':'读写显存'}就是 TTFT。</p>`+
    `<p>${fint(d.B)} 个请求的 decode 步：计算 = ${fint(d.B)} × 15,009,849,344 + ${fint(d.B)} × 4 × 32 × 4,096 × ${fint(d.N+1)} = ${fint(q.fl)} FLOP ÷ ${tfS(pk)} TFLOP/s = ${fixd(q.tc*1e3,3)} ms；读写显存 = ${fint(W_READ)} + ${fint(d.B)} × ${fint(d.N)} × 131,072 = ${fint(q.by)} 字节 ÷ ${tbS(bw)} TB/s = ${fixd(q.tm*1e3,2)} ms。较长的${q.cb?'计算':'读写显存'}就是 TPOT。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={N:{en:'Prompt length',zh:'prompt 长度'},M:{en:'Answer length',zh:'回答长度'},B:{en:'Requests decoding together',zh:'一起 decode 的请求数'}};
const TERM={
 N:()=>tx(`${fint(S.N)} is the prompt length: the tokens prefill takes in, and the first decode step’s context.`,`${fint(S.N)} 是 prompt 长度：prefill 读入的 token 数，也是第一个 decode 步的上下文。`),
 B:()=>tx(`${fint(S.B)} is the number of requests decoding together: each decode step writes one token for each of them.`,`${fint(S.B)} 是一起 decode 的请求数：每个 decode 步给它们各写一个 token。`),
 M:()=>tx(`${fint(S.M-1)} is the answer length, ${fint(S.M)} tokens, minus the first token, which prefill produces: the answer’s decode steps.`,`${fint(S.M-1)} 是回答长度 ${fint(S.M)} 个 token 减去 prefill 产出的首 token：回答要跑的 decode 步数。`),
 avg:()=>tx(`${fixd(R().avg*1e3,3)} ms is the average decode step of this answer: its context grows by one token per step, from ${fint(S.N)} to ${fint(S.N+S.M-2)} tokens, so each step reads 128 KiB more than the last.`,`${fixd(R().avg*1e3,3)} ms 是这个回答平均每个 decode 步的耗时：上下文每步多一个 token，从 ${fint(S.N)} 个涨到 ${fint(S.N+S.M-2)} 个，每一步比上一步多读 128 KiB。`),
 wr:()=>tx(`${MODEL} multiplies ${fint(P_LAYERS)} parameters for every token (its 32 layers and the final RMSNorm) and ${fint(P_HEAD)} in the LM head for every new token, 2 bytes each in BF16. The embedding table is only looked up, one row per token, so a step does not read it whole.`,`${MODEL} 每个 token 要用 ${fint(P_LAYERS)} 个参数做乘法（32 层加最后的 RMSNorm），每个新 token 还要用 LM head 的 ${fint(P_HEAD)} 个，BF16 下每个 2 字节。embedding 表只需每个 token 查一行，所以一步不用把它整张读一遍。`),
 wr2:()=>tx(`${fint(W_READ)} bytes are the weights every step reads once: all the parameters it multiplies, 2 bytes each.`,`${fint(W_READ)} 字节是每一步都要读一遍的权重：参与乘法的全部参数，每个 2 字节。`),
 kv:()=>tx('Every token keeps a key and a value of 128 numbers for each of the 8 KV heads in each of the 32 layers, 2 bytes per number in BF16 (from the config.json).','每个 token 在 32 层、每层 8 个 KV head 上各存一个 key 和一个 value，都是 128 维，BF16 下每个数值 2 字节（来自 config.json）。'),
 kvt:()=>tx('131,072 bytes (128 KiB) is the KV cache of one token: prefill writes it for every prompt token, and a decode step reads it for every token of its context.','131,072 字节（128 KiB）是一个 token 的 KV cache：prefill 为每个 prompt token 写入一份，decode 步为上下文里的每个 token 读一份。'),
 wl:()=>tx(`2 FLOPs, a multiply and an add, for each of the ${fint(P_LAYERS)} parameters of the layers: every token computed pays this.`,`层里的 ${fint(P_LAYERS)} 个参数，每个 2 FLOP（一次乘、一次加）：每算一个 token 都要这么多。`),
 wh:()=>tx(`2 FLOPs for each of the LM head’s ${fint(P_HEAD)} parameters (4,096 × 128,256): it scores every vocabulary token, once per new token only.`,`LM head 的 ${fint(P_HEAD)} 个参数（4,096 × 128,256），每个 2 FLOP：它给词表里每个 token 打分，只为每个新 token 算一次。`),
 attpre:()=>tx(`Attention over the prompt: token i meets the keys of tokens 1 to i, so ${fint(S.N)} × ${fint(S.N+1)} ÷ 2 query–key pairs, at 4 × 32 × 4,096 FLOPs a pair (score and value, 32 layers): 2 × 32 × 4,096 × ${fint(S.N)} × ${fint(S.N+1)}.`,`对 prompt 的 attention：第 i 个 token 和第 1 到第 i 个 token 的 key 相遇，一共 ${fint(S.N)} × ${fint(S.N+1)} ÷ 2 对 query 和 key，每对 4 × 32 × 4,096 FLOP（得分和 value，32 层）：2 × 32 × 4,096 × ${fint(S.N)} × ${fint(S.N+1)}。`),
 attdec:()=>tx(`Each new token’s query meets the keys of its ${fint(S.N)} tokens of context and its own: ${fint(S.N+1)} pairs at 4 × 32 × 4,096 FLOPs a pair.`,`每个新 token 的 query 和它上下文里 ${fint(S.N)} 个 token 以及它自己的 key 相遇：${fint(S.N+1)} 对，每对 4 × 32 × 4,096 FLOP。`),
 effc:()=>tx(`${tfS(H100.peak*EFF.mfu)} TFLOP/s is 50% of the ${GPUN}’s dense BF16 peak, 989.5 TFLOP/s (its datasheet’s 1,979 TFLOPS with sparsity, halved). The 50% is this lab’s assumption.`,`${tfS(H100.peak*EFF.mfu)} TFLOP/s 是 ${GPUN} 稠密 BF16 峰值 989.5 TFLOP/s（规格表上 with sparsity 的 1,979 TFLOPS 减半）的 50%。50% 是本 Lab 的假设。`),
 effm:()=>tx(`${tbS(H100.bw*EFF.mbu)} TB/s is 60% of the ${GPUN}’s peak memory bandwidth, 3.35 TB/s. The 60% is this lab’s assumption.`,`${tbS(H100.bw*EFF.mbu)} TB/s 是 ${GPUN} 峰值显存带宽 3.35 TB/s 的 60%。60% 是本 Lab 的假设。`),
 mem:()=>tx(`The ${GPUN}’s 80 GB, read as 80 GiB, as the KV cache memory lab does: a card’s GB figure is close to that many GiB.`,`${GPUN} 的 80 GB，和 KV cache 显存那个 Lab 一样按 80 GiB 计：显卡标的 GB 接近同样多的 GiB。`),
 util:()=>A.t('utilNote'),
 oh:()=>A.t('ohNote'),
 /* part 6: the doubled GPU's rates and its average decode step */
 xpk:()=>{const g=xRun(xv,xl).g; return tx(`${tfS(g.peak*EFF.mfu)} TFLOP/s is what a step reaches on ${xName(xv)}: twice the ${GPUN}’s dense BF16 peak, 2 × ${fixd(H100.peak/1e12,1)} TFLOP/s, at the same 50%. It is an idealised GPU: only this number is doubled.`,`${tfS(g.peak*EFF.mfu)} TFLOP/s 是${xName(xv)}上一步能达到的算力：${GPUN} 稠密 BF16 峰值的两倍，即 2 × ${fixd(H100.peak/1e12,1)} TFLOP/s，同样按 50% 算。这是理想化的 GPU，只有这一项翻倍。`);},
 xbw:()=>{const g=xRun(xv,xl).g; return tx(`${tbS(g.bw*EFF.mbu)} TB/s is what a step reaches on ${xName(xv)}: twice the ${GPUN}’s peak bandwidth, 2 × ${fixd(H100.bw/1e12,2)} TB/s, at the same 60%. It is an idealised GPU: only this number is doubled.`,`${tbS(g.bw*EFF.mbu)} TB/s 是${xName(xv)}上一步能达到的带宽：${GPUN} 峰值带宽的两倍，即 2 × ${fixd(H100.bw/1e12,2)} TB/s，同样按 60% 算。这是理想化的 GPU，只有这一项翻倍。`);},
 xavg:()=>{const r=xRun(xv,xl); return tx(`${fixd(r.avg*1e3,4)} ms is the average decode step of this answer on ${xName(xv)}: its context grows by one token per step, from ${fint(r.N)} to ${fint(r.N+r.M-2)} tokens, so each step reads 128 KiB more than the last.`,`${fixd(r.avg*1e3,4)} ms 是${xName(xv)}上这个回答平均每个 decode 步的耗时：上下文每步多一个 token，从 ${fint(r.N)} 个涨到 ${fint(r.N+r.M-2)} 个，每一步比上一步多读 128 KiB。`);},
 wmem:()=>tx(`The weights in GPU memory: all ${fint(P_TOTAL)} parameters, the embedding table included, 2 bytes each: ${gbS(W_MEM)} = ${gib2(W_MEM)}.`,`显存里的权重：全部 ${fint(P_TOTAL)} 个参数，包括 embedding 表，每个 2 字节：${gbS(W_MEM)} = ${gib2(W_MEM)}。`)
};
const termText=k=>{const tm=TERM[k]; return tm?tm():'';};
const termTipHTML=k=>TERM[k]?`<div>${esc(termText(k))}</div><div class="act">${jumpAct(k)}</div>`:'';
const FIXED=new Set(['wr','wr2','kv','kvt','wl','wh','attpre','attdec','mem','wmem']);
const DER={avg:'N',effc:'eff',effm:'eff',util:'eff',oh:'eff'};
const locked=k=>(mode==='challenge'&&(k==='N'||k==='M'))||(k==='N'&&preLock());
function ctlValue(k){return {N:tx(`${tokK(S.N)} tokens`,`${tokK(S.N)} 个 token`),M:tx(`${fint(S.M)} tokens`,`${fint(S.M)} 个 token`),B:reqN(S.B)}[k];}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='tl') return tx('Click to scroll to the timeline in the chart.','点击滚到图里的时间线。');
  if(k==='lim') return tx('Click to scroll to the two bars of each phase in the chart.','点击滚到图里每个阶段的两根时间条。');
  if(k==='xa') return tx('Click to scroll to the three timelines in the chart.','点击滚到图里的三条时间线。');
  /* part 6: the load switch sets the prompt and the answer; the GPU options set the doubled rates; the 50% and 60% stay */
  if(isX()&&(k==='N'||k==='M'||k==='xavg')) return tx('Click to jump to “Load” in the card.','点击跳到卡片里的“负载”。');
  if(isX()&&(k==='xpk'||k==='xbw')) return tx('Click to jump to the GPU options in the card.','点击跳到卡片里的 GPU 选项。');
  if(isX()&&(k==='eff'||k==='effc'||k==='effm')) return tx('Fixed in this experiment: 50% of the peak FLOP/s and 60% of the peak bandwidth, on every GPU.','这个实验里固定不变：每张 GPU 都按峰值 FLOP/s 的 50%、峰值带宽的 60% 算。');
  if(k==='eff') return tx('Click to jump to “Assumed values” under Settings.','点击跳到“自由调参”里的“假设值”。');
  if(FIXED.has(k)) return tx(`Fixed in this lab: it always uses ${MODEL} on one ${GPUN}.`,`这个 Lab 里固定不变：始终是一张 ${GPUN} 跑 ${MODEL}。`);
  if(DER[k]) return jumpAct(DER[k]);
  if(!CTLN[k]) return '';
  if(k==='N'&&preLock()) return tx('The prompt length can change after you answer part 2.','在第 2 部分作答后，才能改 prompt 长度。');
  if(locked(k)) return tx(`Fixed during the challenge: ${k==='N'?'4K tokens of context each':'the answer length does not change the time per output token'}.`,`挑战中固定不变：${k==='N'?'每人 4K 个 token 的上下文':'回答长度不影响每个输出 token 的耗时'}。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({N:$('#pd-ic-N',app)||$('#pd-ctl-N',app),M:$('#pd-ctl-M',app),B:$('#pd-ic-B',app)||$('#pd-ic-chB',app)||$('#pd-ctl-B',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#pd-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(isX()&&(k==='N'||k==='M'||k==='xavg'||k==='xpk'||k==='xbw')){const el=$(k==='xpk'||k==='xbw'?'#pd-ic-xv':'#pd-ic-xl',app); A.tip.hide(); if(el) A.ui.jumpToEl(el); return;}
  if(isX()&&(k==='eff'||k==='effc'||k==='effm')){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  if(k==='tl'||k==='lim'||k==='xa'){if(k==='lim'&&roofOpen){roofOpen=false; render();} const el=$(k==='tl'?'#pd-tl':(k==='xa'?'#pd-xa':'#pd-lim'),app); A.tip.hide(); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='controls'){const c=$('#pd-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='eff'){const el=$('#pd-ctl-eff',app); if(el&&!el.closest('.locked')) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('Fixed during the challenge: 50% of the peak FLOP/s and 60% of the peak bandwidth.','挑战中固定不变：峰值 FLOP/s 的 50%，峰值带宽的 60%。'))}</div>`); return;}
  if(FIXED.has(k)){if(src) A.tip.show(src,termTipHTML(k)); return;}
  if(DER[k]){jump(DER[k],src); return;}
  if(!CTLN[k]) return;
  if(locked(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{if(mode==='challenge') setState({B:CH.start}); else {const keep=preLock()?S.N:BASE.N; S={...BASE,N:keep}; render();}}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(S.B).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='xq'){const c=$('#pd-rail .step.current .choices',app); if(c) A.ui.jumpToEl(c); return;}   // part 6: to the prediction's question
  if(a==='bminus'||a==='bplus'){const d=a==='bplus'?1:-1, inRail=!!btn.closest('#pd-rail');
    const f=()=>byLearner(()=>setState({B:S.B+d})); if(inRail){wantObs=true; f(); revealSoon(); wantObs=false;} else bySettings(btn,()=>setState({B:S.B+d})); return;}
  if(a==='roof'){roofOpen=!roofOpen; if(roofOpen) roofSeen=true; if(mode==='guided'){const box=$('#pd-rail [data-obs]',app); renderRail(); void box;} else renderRail();
    if(roofOpen){const el=$('#pd-roof',app); if(el&&innerWidth<=760) el.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(a==='roof-close'){roofOpen=false; renderRail(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#pd-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* a change under Settings never moves the page (R3.2): scroll anchoring is paused and the control is put back where it was */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#pd-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='N') bySettings(e.target,()=>setState({N:N_STOPS[v]}));
    else if(k==='M') bySettings(e.target,()=>setState({M:M_STOPS[v]}));
    else if(k==='B'||k==='chB') bySettings(e.target,()=>setState({B:v}));});
  /* a slider in the card brings its result into view; a change under Settings never scrolls the page (R3.2) */
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#pd-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#pd-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#pd-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    /* part 6: the prediction (then the three GPUs appear), a GPU from the card or from its timeline, the load */
    if(d.xpred){if(xpred) return; xpred=d.xpred; renderRail(); const fb=$('#pd-ic-xv [data-lever="orig"]',app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.xv!=null){if(!isX()) return; if(b.classList.contains('pd-xhid')){act('xq',b); return;} byLearner(()=>setX({v:d.xv})); return;}
    /* on a phone the card sits below the stage, which changes height with the GPU: the option clicked stays under the finger (R3.2) */
    if(d.lever!=null&&isX()){const f=()=>byLearner(()=>setX({v:d.lever})); if(innerWidth<=760) holdPlace(b,f); else f(); xKeep(b); return;}
    if(d.v!=null&&b.closest('.seg-ctl[data-key="xl"]')&&isX()){const f=()=>byLearner(()=>setX({l:d.v})); if(innerWidth<=760) holdPlace(b,f); else f(); xKeep(b); return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#pd-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#pd-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12, under the rule of 07:13 UTC): four text-only sections read top to bottom
   before the guided parts. They give the motivation and the mechanism in symbols (N parameters, P prompt tokens, B requests: what a
   step reads and computes); how strong each effect is stays with the guided parts, so no time, ratio or batch size of theirs is here.
   Hardware numbers are the datasheet's (989.5 TFLOP/s, 3.35 TB/s), so the ridge here is the peak one, ≈ 295 (A.registerPrimer)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=labLink(id); return tx(a,`“${a}”`);};
const prP=(...ps)=>({html:()=>ps.map(p=>`<p>${p()}</p>`).join('')});
const prFx=(en,zh)=>()=>`<span class="pd-prfx">${prM(tx(en,zh))}</span>`;   // a formula on its own line
A.registerPrimer(ID,{min:8,sections:[
 {h:()=>tx('What a large-model inference service does','大模型推理服务在做什么'),parts:[prP(
   ()=>tx(`A large model works on text cut into tokens (a word, part of a word, or a punctuation mark), and it writes its answer one token at a time: each step predicts the next token from all the tokens before it, appends it, and predicts again. This is autoregressive generation (the lab ${prLK('architecture')} shows how one token is computed from the ones before it).`,
          `大模型把文字切成 token（一个词、词的一部分或一个标点）来处理，回答也是一个 token 一个 token 写出来的：每一步根据前面的全部 token 预测下一个 token，把它接到后面，再预测下一个。这叫自回归生成（一个 token 怎样从前面的 token 算出来，见${prLK('architecture')}）。`),
   ()=>tx('Each request involves two stretches of tokens. One is the prompt, the user’s input to the model: the question, the conversation so far, an attached document; all its tokens are known as soon as the request arrives. The other is the answer, which the model has to write as above, one token after another.',
          '每个请求涉及两段 token。一段是 prompt，即用户给模型的输入：问题、之前的对话、附上的文档；请求一到，这一段的全部 token 就都知道了。另一段是回答，要由模型按上面的办法一个 token 接一个 token 地写出来。'),
   ()=>tx('An inference service is the system that runs the model on GPUs to answer these requests. One serving GPU handles many users’ requests at the same time, and it faces two goals that do not agree. Each user wants their own answer quickly. The operator wants every GPU to write as many tokens per second as it can: a GPU costs about the same every second, busy or not (its rent or its depreciation, plus power), so the more tokens it writes per second, the less each token costs.',
          '推理服务（inference serving）就是在 GPU 上运行模型、回答这些请求的系统。一张做推理服务的 GPU 同时处理很多用户的请求，它面对两个并不一致的目标。每个用户都希望自己的回答来得快。运营方希望每张 GPU 每秒写出的 token 越多越好：一张 GPU 不管忙不忙，每秒的成本（租金或折旧，加上电费）都差不多，每秒写出的 token 越多，摊到每个 token 上的成本就越低。'))]},
 {h:()=>tx('Three numbers that measure a service: TTFT, TPOT and throughput','衡量推理服务的三个数：TTFT、TPOT、吞吐'),parts:[prP(
   ()=>tx('“A quick answer” means two things, because the answer is shown while it is being written. After sending a request, the user first waits for the answer’s first token: this wait is the time to first token (TTFT), and it decides how responsive the service feels. Then the tokens appear one after another; the time between two of them is the time per output token (TPOT), and it decides how fast the text streams.',
          '“回答来得快”说的其实是两件事，因为回答是边写边显示的。用户发出请求以后，先要等回答的第一个 token 出现，这段时间叫首 token 延迟（TTFT，time to first token），它决定服务用起来反应快不快。之后 token 一个接一个出现，相邻两个 token 之间隔的时间叫每个输出 token 的耗时（TPOT，time per output token），它决定文字流出来有多快。'),
   ()=>tx(`An answer of ${prM('n')} tokens shows its first token after TTFT, and each of the other ${prM('n-1')} tokens one TPOT after the one before, so the whole answer takes:`,
          `一个 ${prM('n')} 个 token 的回答，第一个 token 在 TTFT 时出现，后面 ${prM('n-1')} 个 token 各隔一个 TPOT，所以写完整个回答要：`),
   prFx('T_{\\text{answer}} = \\text{TTFT} + (n-1) \\cdot \\text{TPOT}','T_{\\text{回答}} = \\text{TTFT} + (n-1) \\cdot \\text{TPOT}'),
   ()=>tx('With the same total, a blank screen for several seconds and then the whole answer at once is a very different thing to use from text that starts at once and keeps coming at a steady pace. So a service usually sets one limit on TTFT and another on TPOT, rather than one on the total.',
          '可同样的总时间，先对着空白等好几秒再一下子出完，和马上开始、稳定地往外出，用起来完全是两回事。所以服务一般给 TTFT 和 TPOT 各定一个上限，而不只看总时间。'),
   ()=>tx('TTFT and TPOT are what one user sees. The operator looks at the whole GPU: how many tokens per second it writes for all users together, its throughput. The cost of each token is then:',
          'TTFT 和 TPOT 是从一个用户的角度看的。运营方看的是整张 GPU：它每秒一共为所有用户写出多少个 token，这叫吞吐（throughput）。每个 token 的成本就是：'),
   prFx('\\text{cost per token} = \\frac{\\text{GPU cost per second}}{\\text{throughput}}','\\text{每个 token 的成本} = \\frac{\\text{GPU 每秒的成本}}{\\text{吞吐}}'))]},
 {h:()=>tx('Prefill and decode: what each step reads and computes','prefill 和 decode：每步读什么、算什么'),parts:[prP(
   ()=>tx('What sets these three numbers is how much work the GPU does in each step, one run of the model on the GPU. A step has to read the numbers it uses from GPU memory into the compute units, above all the weights, the model’s parameters (the numbers in its matrices); and it has to multiply and add, where one multiply or one add is one floating-point operation (FLOP). Reading takes about the bytes moved divided by the memory bandwidth (the most bytes per second GPU memory can move); computing takes about the FLOPs divided by the peak compute (the most FLOPs per second the GPU can do). The GPU reads and computes at the same time, so the longer of the two sets the step’s time.',
          '这三个数由什么决定，要看 GPU 每一步做了多少事；模型在 GPU 上算一遍就是一步。一步既要把用到的数从显存读进计算单元，主要是权重，即模型的参数（各个矩阵里的数）；也要做乘法和加法，一次乘法或一次加法叫一次浮点运算（FLOP）。读完要的时间约是读写的字节数除以显存带宽（每秒最多读写多少字节），算完要的时间约是浮点运算数除以峰值算力（每秒最多做多少次浮点运算）。GPU 边读边算，两段里较长的那段决定这一步的耗时。'),
   ()=>tx(`Comparing the two times is the same as comparing two ratios: the step’s arithmetic intensity, its FLOPs divided by the bytes it moves, and the GPU’s ridge point, its peak compute divided by its memory bandwidth. Above the ridge, computing takes longer than reading and the step is compute-bound; below it, reading takes longer and the step is memory-bound (the basics of ${prLK('roofline')} derive this). An H100 SXM (an NVIDIA data-center GPU) has its ridge at ${prM('989.5\\ \\text{TFLOP/s} \\div 3.35\\ \\text{TB/s} \\approx 295')} FLOPs per byte. In real runs neither the compute nor the bandwidth is used in full, and the compute usually falls further short, so the real ridge is somewhat lower, but still a few hundred.`,
          `比较这两段时间，等于比较两个比值：这一步的算术强度，即浮点运算数除以读写的字节数；和 GPU 的拐点，即峰值算力除以显存带宽。算术强度高于拐点，算完比读完久，这一步受算力限制；低于拐点，读完比算完久，受显存带宽限制（推导见${prLK('roofline')}的知识梳理）。H100 SXM（NVIDIA 的数据中心 GPU）的拐点是 ${prM('989.5\\ \\text{TFLOP/s} \\div 3.35\\ \\text{TB/s} \\approx 295')}，即每字节 295 次浮点运算。实际运行时，算力和带宽都用不满，通常算力差得更多，所以实际的拐点会低一些，但仍有几百。`),
   ()=>tx(`The prompt’s ${prM('L')} tokens are all there as soon as the request arrives, so they need not be computed one by one: they go through the model together, in one step called prefill. Say the model has ${prM('N')} parameters (Llama 3.1 8B has about 8 billion). Stored in BF16, a floating-point format of 2 bytes per number, the weights take about ${prM('2N')} bytes, and the step reads them once. The model’s arithmetic is mostly vectors multiplied by weight matrices, so each parameter does one multiply and one add for each token: each token takes about ${prM('2N')} FLOPs, and the ${prM('L')} tokens about ${prM('2N L')}. Every prompt token has to be computed, not only the last: the model has dozens of layers, and in each layer every token uses two vectors, k and v, that each earlier token computed in that layer (attention is the computation that combines them). The output at the prompt’s last token gives the answer’s first token, so TTFT is mostly the prefill’s time (plus any time the request waits in a queue).`,
          `请求一到，prompt 的 ${prM('L')} 个 token 就都在了，不必一个一个地算，可以放进同一步里一起算：这一步叫 prefill。设模型有 ${prM('N')} 个参数（Llama 3.1 8B 约 80 亿个），用 BF16（每个数占 2 字节的浮点格式）存，权重约 ${prM('2N')} 字节，这一步把它们读一遍。模型里的计算主要是向量乘权重矩阵，每个参数对每个 token 做一次乘法、一次加法，所以每个 token 约 ${prM('2N')} 次浮点运算，${prM('L')} 个 token 一共约 ${prM('2N L')} 次。prompt 的每个 token 都要算，不只是最后一个：模型由几十层组成，每个 token 在每一层都要用到它前面所有 token 在这一层算出的两个向量 k 和 v（注意力就是拿它们来运算）。prompt 最后一个 token 的输出给出回答的第一个 token，所以 TTFT 主要就是 prefill 的耗时（请求要排队的话，再加上排队的时间）。`),
   ()=>tx(`The tokens after it can only come one at a time: predicting a token needs the one before it. This phase is decode. Each decode step writes one token, so TPOT is the time of one decode step. Counting a decode step the way we counted prefill: its FLOPs are one token’s worth, about ${prM('2N')}; its reads are not smaller at all, the weights are still about ${prM('2N')} bytes, plus this request’s KV cache, the k and v of every earlier token in every layer, computed in earlier steps and kept in GPU memory so they need not be computed again (see ${prLK('kv-basics')}).`,
          `之后的 token 只能一个一个地写：要预测下一个 token，得先有上一个。这个阶段叫 decode，每个 decode 步写出一个 token，所以 TPOT 就是一个 decode 步的耗时。照 prefill 的办法给一个 decode 步计数：浮点运算只有一个 token 的份，约 ${prM('2N')} 次；读取却一点没少，权重照样约 ${prM('2N')} 字节，另外还有这个请求的 KV cache，即前面所有 token 在每一层算出的 k 和 v，它们在之前的步骤里算过，存在显存里，就不必重算（见${prLK('kv-basics')}）。`),
   ()=>tx('From the counts above, with only the weights counted as bytes read (with the KV cache, decode’s intensity is lower still), the two kinds of step have these arithmetic intensities:',
          '按上面的计数，读的字节只算权重（算上 KV cache，decode 的算术强度只会更低），两种步的算术强度是：'),
   prFx('I_{\\text{prefill}} \\approx \\frac{2N L}{2N} = L','I_{\\text{prefill}} \\approx \\frac{2N L}{2N} = L'),
   prFx('I_{\\text{decode}} \\approx \\frac{2N}{2N} = 1','I_{\\text{decode}} \\approx \\frac{2N}{2N} = 1'),
   ()=>tx(`Once a prompt has more than a few hundred tokens, prefill’s intensity ${prM('L')} is above the ridge, so prefill is compute-bound; decode’s is about 1, far below it, so decode is memory-bound.`,
          `prompt 的 token 数只要超过几百，prefill 的算术强度 ${prM('L')} 就在拐点之上，受算力限制；decode 的算术强度约为 1，远在拐点之下，受显存带宽限制。`))]},
 {h:()=>tx('Put many requests’ decode into one batch','把多个请求的 decode 合成一个 batch'),parts:[prP(
   ()=>tx('A memory-bound decode is a poor deal for the operator: a decode step spends almost all its time reading, the GPU’s compute sits almost idle all that time, and every second still costs the same.',
          'decode 受显存带宽限制，对运营方很不划算：一个 decode 步的时间几乎都花在读取上，这段时间里 GPU 的算力几乎都闲着，可每秒的成本一分不少。'),
   ()=>tx(`But many requests are decoding on the same GPU at once. Put ${prM('B')} of them into one decode step, and these ${prM('B')} requests are a batch: the vectors of their ${prM('B')} newest tokens are stacked into a matrix and multiplied by the same weights, so vector-times-matrix becomes matrix-times-matrix. With ${prM('\\text{KV}_i')} the bytes of request ${prM('i')}’s KV cache, one decode step for the batch does:`,
          `可同一张 GPU 上同时有很多请求在 decode。把 ${prM('B')} 个请求的 decode 放进同一步，这 ${prM('B')} 个请求叫一个 batch：${prM('B')} 个最新 token 的向量排成一个矩阵，和同一份权重相乘，向量乘矩阵变成了矩阵乘矩阵。记请求 ${prM('i')} 的 KV cache 有 ${prM('\\text{KV}_i')} 字节，这个 batch 的一个 decode 步：`),
   prFx('\\text{bytes read} \\approx 2N + \\text{KV}_1 + \\cdots + \\text{KV}_B','\\text{读的字节} \\approx 2N + \\text{KV}_1 + \\cdots + \\text{KV}_B'),
   prFx('\\text{FLOPs} \\approx 2N \\times B','\\text{浮点运算} \\approx 2N \\times B'),
   ()=>tx('Throughput is the tokens each step writes divided by the step’s time:','吞吐是每一步写出的 token 数除以一步的耗时：'),
   prFx('\\text{throughput} = \\frac{B}{\\text{time of one decode step}}','\\text{吞吐} = \\frac{B}{\\text{一个 decode 步的耗时}}'),
   ()=>tx(`The numerator grows ${prM('B')}-fold, the denominator much less: the step’s biggest read, the weights, is shared by the ${prM('B')} requests; each request’s own KV cache is usually much smaller than the weights; and the extra arithmetic runs on compute units that were idle. So throughput rises and each token costs less. The price falls on each user: every request in the batch gets one token per step, so its TPOT grows by as much as the step does. This is the pull between the speed users want and the throughput the operator wants.`,
          `分子变成 ${prM('B')} 倍，分母却涨得少得多：一步里最大的一笔读取，也就是权重，由 ${prM('B')} 个请求共用；每个请求自己的 KV cache 通常比权重小得多；多出来的运算用的又是本来闲着的算力。所以吞吐上去了，每个 token 的成本降了。代价落在每个用户身上：batch 里每个请求每一步拿到一个 token，一步变长多少，它的 TPOT 就变长多少。这就是用户要的快和运营方要的多之间的拉扯。`),
   ()=>tx(`Even while TPOT stays within its limit, a batch cannot grow without end, because memory sets a hard limit: every request’s KV cache has to stay in GPU memory next to the weights, and it grows with the request’s tokens (the prompt plus the answer written so far). Once the weights’ ${prM('2N')} bytes are in, the rest of GPU memory holds the KV caches of only so many requests, and that is the most a batch can take.`,
          `就算 TPOT 还在上限以内，batch 也不能无限加大，显存有硬上限：每个请求的 KV cache 都要一直存在显存里，和权重放在一起，而且这个请求的 token（prompt 加上已经写出的回答）越多，它越大。显存放下权重的约 ${prM('2N')} 字节以后，剩下的地方放得下多少个请求的 KV cache，一个 batch 最多就有多少个请求。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:{N:S.N,M:S.M,B:S.B},pred,observed:{...observed},saved:saved?{N:saved.N,M:saved.M,B:saved.B}:null,chSubmitted,chReveal,
    roofOpen,roofSeen,xv,xl,xpred,xSeenC2,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPoint(S.B).ok}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=p; if(mode==='challenge'){S.N=CH.N; S.M=CH.M; S.B=Math.max(1,Math.min(S.B,memLimit(CH.N)));}
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sv:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    roofOpen=o.roofOpen===true&&mode!=='challenge'; roofSeen=o.roofSeen===true||roofOpen;
    const inX=mode==='guided'&&STEPS[step].id==='double';   // part 6's GPU and load exist only in that part
    xv=inX&&XV.some(v=>v.id===o.xv)?o.xv:'orig'; xl=inX&&XL.some(l=>l.id===o.xl)?o.xl:'chat';
    xpred=XPRED.some(x=>x.v===o.xpred)?o.xpred:null; xSeenC2=o.xSeenC2===true;
    sbTouched=o.sbTouched===true; if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true; resumed=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!resumed?parse(tok):null; resumed=false;   /* after restore() the address names the saved settings: nothing to apply */ if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#pd-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#pd-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    const rail=$('#pd-rail',app); rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):''); A.ui.initRanges(rail);   // a language switch re-mounts: keep the exits (Batch 2)
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#pd-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#pd-math',app).open=true;}},
  onResize(){if(app){drawTL(); drawMem(); if(isX()){const lg=$('#pd-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}}}},   // part 6: the tick spacing follows the width
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(isX()?$('#pd-xa',app):$('#pd-tl',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#pd-graph',app),
    render({status,vis,read}){if(!app) return; const r=R(), s=statusOf(), masked=decMasked();
      status.className='status '+s.cls; status.textContent=s.short;
      /* part 6: the GPU picked, on the original's scale (before the prediction, the original) */
      if(isX()){const o=xRun('orig',xl), x=xRun(xv,xl), sp=o.e2e*1.03, X=v=>Math.min(100,v/sp*100).toFixed(2)+'%', k=Math.max(1,Math.ceil(x.times.length/12));
        const line=(r,cls)=>`<i class="pd-dk-pre ${cls}" style="width:${X(r.ttft)}"></i><i class="pd-dk-first ${cls}" style="left:${X(r.ttft)}"></i>`+r.times.filter((_,i)=>(i+1)%k===0||i===r.times.length-1).map(t=>`<i class="pd-dk-tick ${cls}" style="left:${X(t)}"></i>`).join('');
        vis.innerHTML=`<div class="pd-dk pd-dkx">${xv!=='orig'?line(o,'pd-dk-o'):''}${line(x,xv!=='orig'?'pd-dk-v':'')}</div>`;
        read.innerHTML=esc(tx(`${xName(xv)}, ${xLoadS(xl)}: whole answer ${xWa(x.e2e)}`,`${xName(xv)}，${xLoadS(xl)}：整个回答 ${xWa(x.e2e)}`)+(xv!=='orig'?tx(` (original ${xWa(o.e2e)})`,`（原版 ${xWa(o.e2e)}）`):''));   // the original in the sentence itself: phones hide .dk-mem (review A6)
        return;}
      const span=masked?prefill(32768).t*1.07:(r.ticks.length?r.ticks[r.ticks.length-1]*1.015:r.ttft*1.2), X=v=>Math.min(100,v/span*100).toFixed(2)+'%';
      vis.innerHTML=`<div class="pd-dk"><i class="pd-dk-pre" style="width:${X(r.ttft)}"></i><i class="pd-dk-first" style="left:${X(r.ttft)}"></i>${masked?'':r.ticks.filter((_,i)=>i%2===0).map(t=>`<i class="pd-dk-tick" style="left:${X(t)}"></i>`).join('')}</div>`;
      if(mode==='challenge'){read.innerHTML=esc(tx(`${userN(r.B)} at 4K tokens each: ${fDec(r.tpot)} per output token`,`${userN(r.B)}、每人 4K 个 token：每个输出 token ${fDec(r.tpot)}`))+`<span class="dk-mem">${esc(tx(' (the goal is at most 20 ms)','（目标是最多 20 ms）'))}</span>`; return;}
      read.innerHTML=masked?esc(tx(`Prefill of ${tokK(r.N)} tokens: first token after ${fPre(r.ttft)}`,`${tokK(r.N)} 个 token 的 prefill：${fPre(r.ttft)} 后出首 token`))
        :esc(tx(`${tokK(r.N)}-token prompt: first token after ${fPre(r.ttft)}, then one every ${fDec(r.tpot)}`,`prompt ${tokK(r.N)} 个 token：${fPre(r.ttft)} 后出首 token，之后每 ${fDec(r.tpot)} 一个`))+(r.B>1?`<span class="dk-mem">${esc(tx(` (${reqN(r.B)} together)`,`（${reqN(r.B)}一起）`))}</span>`:'');}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__pd=Object.freeze({
  prefill:(N,g)=>deep(prefill(N,g==='a100'?A100:H100)), decode:(B,c,g)=>deep(decode(B,c,g==='a100'?A100:H100)), compute:s=>deep(compute(s)),
  memLimit, chPoint:B=>deep(chPoint(B)), chBest, chBestAt, drill:s=>deep(makeDrill(s)),
  fmt:Object.freeze({pre:fPre,dec:fDec,tot:fTot,tps:fTps,thr:fThr,gb:gbS}),
  consts:Object.freeze({H100:Object.freeze({peak:H100.peak,bw:H100.bw,mem:H100.mem}),A100:Object.freeze({peak:A100.peak,bw:A100.bw}),EFF:Object.freeze({...EFF}),LL:Object.freeze({...LL}),
    P_EMB,P_HEAD,P_LAYER,P_LAYERS,P_TOTAL,KV_TOK,W_READ,W_MEM,RIDGE,UTIL,OH,N_STOPS:N_STOPS.slice(),M_STOPS:M_STOPS.slice(),BASE:{...BASE},TL_K,
    CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,
    XV:XV.map(v=>({id:v.id,peak:v.g.peak,bw:v.g.bw})),XL:XL.map(l=>({id:l.id,N:l.N,M:l.M})),XPRED_RIGHT}),
  xrun:(v,l)=>deep(xRun(v,l)),
  state:()=>({...S,mode,step,pred,observed:{...observed},chSubmitted,chReveal,roofOpen,roofSeen,stageBusy:stageT!==0,xv,xl,xpred,xK})
});
})();
