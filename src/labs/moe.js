/* moe (the trunk "How LLMs work"): How can a 47B model run like a 13B one?
   Fidelity: Formula. Mixtral-8x7B's parameters are counted component by component from its config.json (46,702,792,704 in all,
   12,879,925,248 for one token: Mistral AI's 46.7B and 12.9B). Every time on the page comes from prefill-decode's step model (the
   same constants as teasers/set-a.js): a step takes the longer of its arithmetic, at 50% of the H100 SXM's dense BF16 peak, and its
   memory reads, at 60% of the peak bandwidth.
   - A decode step for B requests at context c reads the shared weights (all but the embedding table), the experts its tokens picked,
     and the KV caches. Experts read per layer: E·(1 − (1 − k/E)^B), the average under uniform independent routing (an assumption).
   - Memory: the KV cache memory lab's defaults (80 GB read as 80 GiB, 90% usable, 2 GiB runtime overhead); FP8 = 1 byte per parameter.
   - Part 6: Mixtral next to dense models (Llama 2 13B and 70B, a hypothetical dense 46.7B; Mistral 7B in the quality bars only).
   - Part 7: expert parallelism (8 GPUs, one expert of every layer each); the layer waits for its busiest GPU.
   Words (EN / ZH): expert / 专家, router / 路由器, mixture of experts (MoE) / 混合专家（MoE）, as in the lab Model parameters;
   decode step / decode 步; part / 部分 for guided cards; parameters / 参数.
   window.__moe is a read-only hook for tests/moe_numbers.py and tests/moe_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='moe';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const MODEL='Mixtral‑8x7B', GPUN='H100 SXM';      // no-break hyphen and space: a name never breaks

/* =====================================================================
   engine
   ===================================================================== */
/* Mixtral-8x7B config.json (mistralai/Mixtral-8x7B-v0.1): hidden_size 4096, intermediate_size 14336, num_hidden_layers 32,
   num_attention_heads 32, num_key_value_heads 8, num_local_experts 8, num_experts_per_tok 2, vocab_size 32000, untied LM head */
const MX={V:32000,d:4096,L:32,H:32,KV:8,hd:128,f:14336,E:8,k:2,aux:0.02};
const P_ATTN=2*MX.d*MX.d+2*MX.d*MX.KV*MX.hd;     // Wq, Wo: d × d; Wk, Wv: d × (8 × 128)
const P_NORM=2*MX.d, P_ROUTER=MX.d*MX.E, P_EXPERT=3*MX.d*MX.f;   // two RMSNorms; the router; one expert's gate, up and down
const P_EMB=MX.V*MX.d, P_HEAD=MX.V*MX.d, P_FNORM=MX.d;
const P_LAYER=P_ATTN+P_NORM+P_ROUTER+MX.E*P_EXPERT;
const P_TOTAL=P_EMB+P_HEAD+P_FNORM+MX.L*P_LAYER;             // 46,702,792,704
const P_EXPS=MX.L*MX.E*P_EXPERT;                             // every expert of every layer
const P_SHARED=P_TOTAL-P_EXPS;                               // attention, norms, routers, embedding table, LM head
const P_SREAD=P_SHARED-P_EMB;                                // read by every decode step (the embedding table is a lookup)
const P_ACTIVE=P_TOTAL-MX.L*(MX.E-MX.k)*P_EXPERT;            // 12,879,925,248: what one token uses
const P_MULT=P_SREAD+MX.L*MX.k*P_EXPERT;                     // multiplied by every token
const P_SLOT=MX.L*P_EXPERT;                                  // one expert position across the 32 layers
const KV_TOK=2*MX.L*MX.KV*MX.hd*2;                           // K and V in BF16: 131,072 bytes = 128 KiB per token
/* prefill-decode's step model: the same GPU and the same assumed shares (teasers/set-a.js) */
const H100={peak:1979e12/2,bw:3.35e12,mem:80}, EFF={mfu:0.5,mbu:0.6};
const FS=H100.peak*EFF.mfu, BS=H100.bw*EFF.mbu;
const KVD=A.kv&&A.kv.DEFAULT?A.kv.DEFAULT:{util:0.9,overhead:2};
const UTIL=KVD.util, OH=KVD.overhead;
const BPP={bf16:2,fp8:1}, PN={bf16:'BF16',fp8:'FP8'};
/* experts a batch of B tokens reads in one layer, on average, if each token picks its k of E at random (uniform, independent) */
const touched=(B,E,k)=>E*(1-Math.pow(1-k/E,B));
const attnF=c=>4*MX.L*MX.d*(c+1);                            // one new token's attention at context c (QKᵀ and AV, all layers)
function decode(B,c,p,T){
  const bpp=BPP[p||'fp8'], Tt=T!=null?T:touched(B,MX.E,MX.k);
  const fl=B*(2*P_MULT+attnF(c)), byS=bpp*P_SREAD, byE=bpp*MX.L*Tt*P_EXPERT, byK=B*c*KV_TOK, by=byS+byE+byK;
  const tc=fl/FS, tm=by/BS;
  return {B,c,p:p||'fp8',T:Tt,fl,byS,byE,byK,by,tc,tm,t:Math.max(tc,tm),cb:tc>=tm};
}
function memOf(p,c){
  const cap=H100.mem*A.GiB, usable=cap*UTIL, oh=OH*A.GiB, w=BPP[p]*P_TOTAL, budget=usable-w-oh, per=c*KV_TOK;
  return {cap,usable,oh,w,budget,per,fits:budget>0,max:budget>0?Math.floor(budget/per+1e-9):0};
}
const C_STOPS=[512,1024,2048,4096,8192,16384,32768];
const BASE={p:'fp8',c:2048,B:1};
const AC=new Map();
function compute(s){
  const key=s.p+'|'+s.c+'|'+s.B; if(AC.has(key)) return AC.get(key);
  const m=memOf(s.p,s.c), B=Math.max(1,Math.min(s.B,Math.max(1,m.max)));
  const d=m.fits?decode(B,s.c,s.p):null, lo=m.fits?decode(B,s.c,s.p,MX.k):null, hi=m.fits?decode(B,s.c,s.p,MX.E):null;
  const r={p:s.p,c:s.c,B,m,d,lo,hi,tps:d?1/d.t:0,thr:d?B/d.t:0};
  AC.set(key,r); return r;
}
/* the challenge: FP8 on one H100, 4K tokens of context per user, a token at least every 30 ms; choose the number of users */
const CH={p:'fp8',c:4096,goal:0.030,start:1,rows:[1,8,16,24,25,26,32,53]};
const chPoint=B=>{const d=decode(B,CH.c,CH.p); return {B,t:d.t,tps:1/d.t,thr:B/d.t,ok:d.t<=CH.goal+1e-12,d};};
const chMax=()=>memOf(CH.p,CH.c).max;
function chBest(){let best=0; for(let B=1;B<=chMax();B++){if(chPoint(B).ok) best=B;} return best;}
/* part 7: expert parallelism. 8 H100 SXM hold one expert of every layer each; a prefill of 4,096 tokens sends 8,192 token copies
   to experts; each GPU computes the copies its expert got (the time of one layer, FP8 weights); the layer waits for the busiest GPU.
   Copies are whole: what the popular expert leaves is split as evenly as whole numbers allow (6,144 = 5 × 878 + 2 × 877). */
const EP={N:4096,hot:2};                                     // the popular expert is expert 3 (index 2)
const EP_COPIES=EP.N*MX.k;
const SKEW=[{id:'bal',share:1/8},{id:'q',share:1/4},{id:'h',share:1/2}];
function epRun(id){
  const sk=SKEW.find(x=>x.id===id)||SKEW[0], hot=Math.round(sk.share*EP_COPIES), restN=EP_COPIES-hot, base=Math.floor(restN/(MX.E-1)), ext=restN-base*(MX.E-1);
  let j=0; const loads=Array.from({length:MX.E},(_,e)=>{if(e===EP.hot) return hot; const v=base+(j<ext?1:0); j++; return v;});
  const gpu=loads.map(n=>{const tc=n*2*P_EXPERT/FS, tm=BPP.fp8*P_EXPERT/BS; return {n,tc,tm,t:Math.max(tc,tm)};});
  const t=Math.max(...gpu.map(g=>g.t)), bal=Math.max(EP_COPIES/MX.E*2*P_EXPERT/FS,BPP.fp8*P_EXPERT/BS);
  const busy=gpu.reduce((a,g)=>a+g.t,0)/(MX.E*t);
  return {id:sk.id,share:sk.share,loads,gpu,t,bal,ratio:t/bal,busy,idle:1-busy};
}
/* part 6: the dense models a learner would pick instead (Mistral 7B, from Mistral-7B-v0.1's config.json, only in the quality bars). Llama 2 13B and 70B from their config.json (meta-llama, read from the NousResearch
   mirrors); a dense model with Mixtral's 46.7B parameters is hypothetical. Arithmetic per token = 2 FLOPs per multiplied parameter (every
   parameter but the embedding table for a dense model; the shared parts and 2 experts per layer for Mixtral). Weights read per decode
   step, FP8 (1 byte per parameter): a dense model reads the same at any batch; Mixtral reads 2 experts per layer for 1 request and the
   experts CMP_B requests pick. Memory: every parameter, 1 byte. The KV cache is left out: it depends on the attention design, not on MoE. */
const DENSE={l13:{V:32000,d:5120,L:40,H:40,KV:40,f:13824},l70:{V:32000,d:8192,L:80,H:64,KV:8,f:28672},m7:{V:32000,d:4096,L:32,H:32,KV:8,f:14336}};
function denseP(c){const hd=c.d/c.H, layer=2*c.d*c.d+2*c.d*c.KV*hd+3*c.d*c.f+2*c.d, emb=c.V*c.d; return {total:2*emb+c.L*layer+c.d, read:emb+c.L*layer+c.d, kv:2*c.L*c.KV*hd*2};}
/* the Mixtral paper (arXiv 2401.04088), Table 2, as printed (%): Mistral AI re-ran every Llama 2 number with its own pipeline */
const BENCH_COLS=['MMLU','HellaSwag','WinoGrande','PIQA','ARC-Easy','ARC-Challenge','NaturalQuestions','TriviaQA','HumanEval','MBPP','MATH','GSM8K'];
const BENCH={l13:[55.6,80.7,72.9,80.8,75.2,48.8,16.7,64.0,18.9,35.4,6.0,34.3],mx:[70.6,84.4,77.2,83.6,83.1,59.7,30.6,71.5,40.2,60.7,28.4,74.4],l70:[69.9,85.4,80.4,82.6,79.9,56.5,25.4,73.0,29.3,49.8,13.8,69.6],m7:[62.5,81.0,74.2,82.2,80.5,54.9,23.2,62.5,26.2,50.2,12.7,50.0]};
const BENCH_SHOW=[0,11,8,1];                                 // MMLU, GSM8K, HumanEval, HellaSwag (one where Llama 2 70B is higher)
/* the settings of the paper's §3 for the four shown */
const BENCH_SHOT={0:{en:'5-shot',zh:'5-shot'},1:{en:'0-shot',zh:'0-shot'},8:{en:'0-shot',zh:'0-shot'},11:{en:'8-shot, majority vote over 8 samples',zh:'8-shot，8 次采样投票取多数'}};
const beats=id=>BENCH.mx.filter((v,i)=>v>BENCH[id][i]).length;   // benchmarks where Mixtral scores higher
const CMP_B=64, CMP_IDS=['l13','mx','l70','d47'], CMP_N=4096, CMP_C=2048;
/* per model, FP8 weights: pre = the prefill of a CMP_N-token prompt at the step model's arithmetic rate (the matrix multiplications only);
   t1, tB = the weight reads of one decode step for 1 and for CMP_B requests; kv = KV cache bytes per token (BF16); fit = the requests of
   CMP_C tokens one H100 holds next to the weights (the KV cache lab's memory model); kvB = the KV cache one step reads at CMP_B × CMP_C.
   The hypothetical dense 46.7B keeps Mixtral's attention, so its KV cache is Mixtral's. */
function cmpModels(){
  const a=denseP(DENSE.l13), b=denseP(DENSE.l70), T=touched(CMP_B,MX.E,MX.k), room=H100.mem*A.GiB*UTIL-OH*A.GiB;
  const mk=(id,total,mult,read1,readB,kv)=>{const fl=2*mult, left=room-total;
    return {id,total,mult,fl,read1,readB,mem:total,kv,pre:CMP_N*fl/FS,t1:read1/BS,tB:readB/BS,fit:left>0?Math.floor(left/(CMP_C*kv)+1e-9):0,kvB:CMP_B*CMP_C*kv};};
  return [mk('l13',a.total,a.read,a.read,a.read,a.kv),mk('mx',P_TOTAL,P_MULT,P_MULT,P_SREAD+MX.L*T*P_EXPERT,KV_TOK),mk('l70',b.total,b.read,b.read,b.read,b.kv),mk('d47',P_TOTAL,P_TOTAL-P_EMB,P_TOTAL-P_EMB,P_TOTAL-P_EMB,KV_TOK)];
}
const M7=(()=>{const q=denseP(DENSE.m7); return {total:q.total,mult:q.read,fl:2*q.read};})();
/* Llama 2 70B in the curve's terms, like Mixtral's line: FP8 weights read in every decode step plus its own KV cache (320 KiB per
   token); the line ends at the most requests one H100 holds next to the weights (the KV cache lab's memory model) */
const L70=(()=>{const q=denseP(DENSE.l70); return {total:q.total,read:q.read,kv:q.kv,L:DENSE.l70.L,d:DENSE.l70.d,ctx:4096};})();   // ctx: max_position_embeddings
const l70Takes=c=>c+1<=L70.ctx;                              // the newest token must fit in its 4,096-token window: 2K yes, 4K no
function l70Step(B,c){const fl=B*(2*L70.read+4*L70.L*L70.d*(c+1)), by=L70.read+B*c*L70.kv; return {B,c,fl,by,t:Math.max(fl/FS,by/BS)};}
const l70Fit=c=>{const left=H100.mem*A.GiB*UTIL-OH*A.GiB-L70.total; return left>0?Math.floor(left/(c*L70.kv)+1e-9):0;};
let lastCurve=null;                                          // what the curve drew, for the tests
const cmpOf=id=>cmpModels().find(m=>m.id===id);
/* seeded draws for the pictures only (the numbers use the averages above) */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function sampleRoute(B,seed){
  const rnd=mulberry(seed*9973+B*31+7), loads=Array(MX.E).fill(0), picks=[];
  for(let i=0;i<B;i++){const a=Math.floor(rnd()*MX.E); let b=Math.floor(rnd()*(MX.E-1)); if(b>=a) b++; picks.push([a,b]); loads[a]++; loads[b]++;}
  return {picks,loads,hit:loads.filter(x=>x>0).length};
}
/* part 3: six tokens of a sentence and made-up router scores (seeded, illustrative: not Mixtral's router) */
const SENT=['The','capital','of','France','is','Paris'];
function routeTok(i){
  const rnd=mulberry(i*7919+101), z=[];
  for(let e=0;e<MX.E;e++){const u=Math.max(1e-9,rnd()), v=rnd(); z.push(Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)*1.15);}
  const mx=Math.max(...z), ex=z.map(v=>Math.exp(v-mx)), s=ex.reduce((a,b)=>a+b,0), pr=ex.map(v=>v/s);
  const top=pr.map((v,e)=>[v,e]).sort((a,b)=>b[0]-a[0]).slice(0,MX.k).map(x=>x[1]), tw=top.reduce((a,e)=>a+pr[e],0);
  return {i,tok:SENT[i],pr,top,w:top.map(e=>pr[e]/tw)};
}

/* =====================================================================
   formatting (numbers in cards and math as shown; prose rounds, R6.20)
   ===================================================================== */
const fint=A.fmt.int, big=A.fmt.big, BIG=n=>A.ui.big?A.ui.big(n):big(n);
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const NS=' ';
const ms2=sec=>fixd(sec*1e3,2)+NS+'ms';                      // a decode step: 6.48 ms
const msR=sec=>{const v=sec*1e3; return v>=9.95?fint(Math.round(v))+NS+'ms':fixd(v,1)+NS+'ms';};   // prose: 6.5 ms, 22 ms
const gib1=b=>fixd(b/A.GiB,1)+NS+'GiB';
const gbS=b=>{const v=b/1e9; return (v>=9.995?fixd(v,1):fixd(v,2))+NS+'GB';};
const tokK=n=>n>=1024&&n%1024===0?(n/1024)+'K':fint(n);
const reqN=n=>tx(n===1?'1 request':`${fint(n)} requests`,`${fint(n)} 个请求`);
const userN=n=>tx(n===1?'1 user':`${fint(n)} users`,`${fint(n)} 个用户`);
const T1=v=>fixd(v,1);                                       // experts per layer: 7.2
const ratioS=q=>q>=9.95?fint(q):fixd(q,1);
const rs=q=>Math.abs(q-Math.round(q))<1e-9?fint(Math.round(q)):ratioS(q);   // part 7's ratios are exact: 2×, 4×
const pct0=x=>fint(x*100)+'%';
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
const expN=e=>tx(`expert ${e+1}`,`专家 ${e+1}`);
const secS=s=>fixd(s,2)+NS+'s';                              // a prefill: 0.21 s
const ms1=sec=>fixd(sec*1e3,1)+NS+'ms';                      // part 6's read times, from GB with one decimal (R6.6): 23.2 ms
const gb3=b=>fixd(b/1e9,3)+NS+'GB';                          // a GB value that is divided by 2.01 TB/s next to it (R6.6)
const bB=n=>fixd(n/1e9,2)+'B';                               // parameters a token multiplies: 12.75B (2 × 12.75B = 25.5 GFLOP)
const whoN=n=>mode==='challenge'?userN(n):reqN(n);           // the challenge speaks of users (R2.8)
const one1=()=>mode==='challenge'?tx('user','用户'):tx('request','请求');
/* a link to another part of the tour; from Scenarios or the challenge it opens the guided tour at that part. PLp: the same in
   parentheses, which stay on the link's line */
const PL=(id,t)=>{const i=SIX(id); return `<button type="button" class="linkbtn" data-go="${i}" data-tipx="${esc(tx(`Click to go to part ${i+1}, “${L(STEPS[i].title)}”.`,`点击去第 ${i+1} 部分“${L(STEPS[i].title)}”。`))}">${t||tx(`part ${i+1}`,`第 ${i+1} 部分`)}</button>`;};
const PLp=(id,t)=>tx(`<span class="moe-nw">(${PL(id,t)})</span>`,`<span class="moe-nw">（${PL(id,t)}）</span>`);

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let saved=null, chSubmitted=false, chReveal=false, drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false, sbTouched=false;
let animShown='', animN=0;   // R3.20: the routing picture whose dots have run; the same picture again (another part, mode or option with the same routes, a language switch) is shown finished
const watched=new Set();      // R3.20: the routing pictures already seen to the end (their dots ran out, or the picture was drawn finished), by their routes, not by part; coming back to one (A → B → A, a later part) draws it finished
let runSig='';                // the picture whose dots are running
let rt=0, skew='bal', seed=1, animT=0, animGo=false, cmp='l13', cmpSeen=new Set(['l13']), bench=0;
const shownPrev={};
const R=()=>compute(S);

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 moe:()=>tx(`A mixture-of-experts (MoE) layer replaces the MLP block with several MLP blocks, the experts, and a small router that sends each token to a few of them. Only the chosen experts compute that token. ${MODEL} has 8 experts in each of its 32 layers and sends each token to 2.`,
            `混合专家（MoE，mixture of experts）层把 MLP 块换成好几个 MLP 块（叫作专家），再加一个小路由器，把每个 token 只送到其中几个专家。只有被选中的专家为这个 token 计算。${MODEL} 每层有 8 个专家，每个 token 送到其中 2 个。`),
 expert:()=>tx(`An expert is one MLP block of an MoE layer, with its own gate, up and down matrices. Each of ${MODEL}’s experts is the size of Mistral‑7B’s MLP block: 3 × 4,096 × 14,336 = 176,160,768 parameters.`,
               `专家就是 MoE 层里的一个 MLP 块，有自己的 gate、up、down 三个矩阵。${MODEL} 的每个专家和 Mistral‑7B 的 MLP 块一样大：3 × 4,096 × 14,336 = 176,160,768 个参数。`),
 router:()=>tx(`The router is a small matrix in each layer, 4,096 × 8 in ${MODEL}, that gives a token one score per expert. Softmax turns the scores into shares; the token goes to the 2 experts with the largest shares, and their outputs are added, weighted by those two shares rescaled to add up to 100%.`,
               `路由器是每层里的一个小矩阵（${MODEL} 里是 4,096 × 8），给一个 token 的每个专家打一个分。softmax 把得分变成份额；token 被送到份额最大的 2 个专家，它们的输出按这两个份额（重新缩放到加起来是 100%）加权相加。`),
 active:()=>tx(`Active parameters are the parameters one token uses: all of them except the experts its router did not pick. ${MODEL}’s 12.9B count the embedding table and the LM head too, as Mistral AI does, although a token looks up only its own row of the table.`,
               `激活参数是一个 token 用到的参数：除了路由器没选中的专家，其余参数都算。${MODEL} 的 12.9B 和 Mistral AI 一样把 embedding 表和 LM head 也算在内，虽然一个 token 只查 embedding 表里自己的那一行。`),
 flop:()=>tx('A FLOP is one floating-point operation; a multiply and an add count as 2. A GFLOP is 10⁹ FLOPs and a TFLOP 10¹².','FLOP 是一次浮点运算，一次乘法加一次加法算 2 个 FLOP。GFLOP 是 10⁹ 个 FLOP，TFLOP 是 10¹² 个。'),
 gb:()=>tx('GB here is 10⁹ bytes, to match the bandwidth in TB/s (10¹² bytes per second).','这里的 GB 是 10⁹ 字节，和带宽的 TB/s（每秒 10¹² 字节）一致。'),
 fp8:()=>tx('FP8 stores each parameter in 1 byte instead of BF16’s 2. This lab counts every parameter at 1 byte; a real FP8 checkpoint of Mixtral keeps the embedding table, the LM head and the routers in BF16, about 0.25 GiB more.',
            'FP8 每个参数只占 1 字节，BF16 占 2 字节。这个 Lab 把每个参数都按 1 字节算；真实的 Mixtral FP8 checkpoint 会把 embedding 表、LM head 和路由器留在 BF16，大约多 0.25 GiB。'),
 dstep:()=>tx('A decode step is one pass of the model that produces one new token for each request in it. It computes only the newest token of each request, and reads from GPU memory the weights it multiplies and the keys and values of the earlier tokens (the KV cache).',
              'decode 步是把模型跑一遍、给其中每个请求各多产出一个 token 的过程。它只算每个请求最新的那个 token，并从显存里读它要乘的权重，以及之前各 token 的 key 和 value（KV cache）。'),
 batch:()=>tx('A batch is the set of requests one step serves together. A decode step for a batch of B requests reads each weight it needs once for all of them.','batch 是一步里一起处理的一组请求。B 个请求的 decode 步，需要的每个参数只为它们一起读一遍。'),
 read:()=>tx('An expert is read in a decode step when at least one token of the step picked it: its weights come from GPU memory once, however many of the step’s tokens use it. An expert no token picked is not read at all.',
             '一个 decode 步里，只要有一个 token 选了某个专家，这个专家就要被读：它的权重从显存读一遍，不管这一步里有几个 token 用它。没有 token 选的专家，这一步完全不读。'),
 uniform:()=>tx(`This lab assumes each token picks its 2 experts at random, every pair equally likely, independently of the other tokens. A token then skips a given expert with probability 6/8, and all B tokens skip it with probability ${A.ui.math('\\left(\\frac{6}{8}\\right)^{B}')}, so a layer reads ${A.ui.math('8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{B}\\right)')} experts on average. Real routers favour some experts over others; for the same number of picks, uniform routing reaches the most different experts, so a real batch reads somewhat fewer.`,
                `这个 Lab 假设每个 token 随机选 2 个专家，每一对的可能性相同，和别的 token 无关。于是一个 token 躲过某个专家的概率是 6/8，B 个 token 都躲过它的概率是 ${A.ui.math('\\left(\\frac{6}{8}\\right)^{B}')}，所以一层平均要读 ${A.ui.math('8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{B}\\right)')} 个专家。真实的路由器会偏爱某些专家；选择次数相同时，均匀路由选到的不同专家最多，所以真实的 batch 读到的专家会少一些。`),
 mult:()=>tx(`A token multiplies ${bB(P_MULT)} of its 12.9B active parameters: all but the embedding table, of which it only looks up its own row. Each costs 2 FLOPs, a multiply and an add: 2 × ${bB(P_MULT)} = 25.5 GFLOP.`,
             `一个 token 要乘的是 12.9B 个激活参数里的 ${bB(P_MULT)} 个：embedding 表不乘，它只查表里自己那一行。每个参数 2 FLOP（一次乘法加一次加法）：2 × ${bB(P_MULT)} = 25.5 GFLOP。`),
 ep:()=>tx('Expert parallelism puts different experts on different GPUs: here each of 8 GPUs holds one expert of every layer, and each token is sent to the GPUs that hold the experts it picked. The GPUs exchange the tokens before and after the experts (an all-to-all), which this lab leaves out.',
           '专家并行（expert parallelism）把不同的专家放到不同的 GPU 上：这里 8 张 GPU 各存每一层的一个专家，每个 token 被送到存着它所选专家的 GPU 上。GPU 之间在专家前后要交换 token（all-to-all），这个 Lab 不计这部分。'),
 lbl:()=>tx(`A load-balancing loss is an extra training loss that grows when the router sends more tokens to some experts than to others, so the router learns to spread them. ${MODEL}’s config.json weights it by 0.02 (router_aux_loss_coef); the formula is the Switch Transformer’s.`,
            `负载均衡 loss 是训练时额外加的一项 loss：路由器把 token 偏向某些专家时它就变大，于是路由器学会把 token 分散开。${MODEL} 的 config.json 给它的权重是 0.02（router_aux_loss_coef）；公式来自 Switch Transformer。`),
 gib:()=>kt('gib'), kib:()=>kt('kib'), kv:()=>kt('kvcache'), mlp:()=>kt('mlp'), softmax:()=>kt('softmax'), param:()=>kt('param')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'lt40',t:{en:'Under 40 GiB',zh:'不到 40 GiB'}},{v:'40-80',t:{en:'40 to 80 GiB',zh:'40 到 80 GiB'}},
  {v:'80-100',t:{en:'80 to 100 GiB',zh:'80 到 100 GiB'}},{v:'gt100',t:{en:'Over 100 GiB',zh:'超过 100 GiB'}}];
const PRED_RIGHT='80-100';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const stepId=()=>mode==='guided'&&STEPS[step]?STEPS[step].id:'';
const memHidden=()=>mode==='guided'&&step<SIX('mem');     // the prediction's answer waits for part 4 (R3.8)
const isEP=()=>stepId()==='imb';
const isRouter=()=>stepId()==='router';
/* the batch size: a slider of whole requests with one-step buttons */
function bCtl(id,key,max,label,inline){
  return `<div class="${inline?'inline-ctl':'ctl'} moe-bctl" id="${id}"><label for="${id}-in"><span>${esc(label)}</span> <output></output></label>`+
    `<div class="moe-brow"><button type="button" class="btn ghost sm moe-bstep" data-act="bminus" data-for="${key}" aria-label="${esc(tx('One fewer','少一个'))}">−</button>`+
    `<input type="range" id="${id}-in" data-key="${key}" min="1" max="${Math.max(1,max)}" step="1" value="${Math.min(S.B,Math.max(1,max))}">`+
    `<button type="button" class="btn ghost sm moe-bstep" data-act="bplus" data-for="${key}" aria-label="${esc(tx('One more','多一个'))}">+</button></div>`+
    `<div class="ctl-note moe-bnote"></div></div>`;
}
const pCtl=(id,inline)=>A.ui.seg({id,key:'p',label:tx('Weights stored in','权重的存储格式'),options:[['bf16',tx('BF16, 2 bytes','BF16，2 字节')],['fp8',tx('FP8, 1 byte','FP8，1 字节')]],value:S.p,inline});
const cCtl=(id,inline)=>A.ui.range({id,key:'c',label:tx('Context per request','每个请求的上下文'),min:0,max:C_STOPS.length-1,value:Math.max(0,C_STOPS.indexOf(S.c)),out:tx(`${tokK(S.c)} tokens`,`${tokK(S.c)} 个 token`),inline});
const skewLevers=()=>A.ui.levers({aria:tx('How the router spreads the tokens, pick one','路由器怎么分配 token，选一种'),value:skew,
  options:SKEW.map(x=>({v:x.id,title:skewTitle(x.id),desc:skewDesc(x.id),result:esc(skewRes(x.id))}))});
const skewTitle=id=>({bal:tx('Balanced','均衡'),q:tx('One popular expert','一个专家很热门'),h:tx('Every token picks it','每个 token 都选它')}[id]);
const skewDesc=id=>{const r=epRun(id), hot=fint(Math.round(r.loads[EP.hot]));
  return id==='bal'?tx(`initial setup: every expert gets ${fint(EP_COPIES/MX.E)} of the ${fint(EP_COPIES)} copies`,`初始设置：${fint(EP_COPIES)} 份里每个专家分到 ${fint(EP_COPIES/MX.E)} 份`)
    :tx(`expert 3 gets ${hot} copies, ${id==='q'?'a quarter':'half'} of them`,`专家 3 分到 ${hot} 份，占${id==='q'?'四分之一':'一半'}`);};
const skewRes=id=>{const r=epRun(id); return id==='bal'?tx('1× (initial)','1×（初始）'):tx(`${rs(r.ratio)}× as long`,`耗时变成 ${rs(r.ratio)} 倍`);};
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'flow',
  lead:()=>tx('46.7B parameters, but each token computes with only 12.9B: how?','46.7B 个参数，每个 token 却只用 12.9B 个来计算：怎么做到的？'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>${MODEL} holds 46.7B ${DEF('parameters',TIP.param())}, yet Mistral AI says it runs at the speed and cost of a 12.9B model. It is a ${DEF('mixture of experts (MoE)',TIP.moe())}: in every layer, the ${DEF('MLP block',TIP.mlp())} is replaced by 8 ${DEF('experts',TIP.expert())}, and a small ${DEF('router',TIP.router())} sends each token to 2 of them <span class="moe-nw">(${J('flow','the picture in the chart')})</span>.</p><p>So each token computes with only 12.9B of the 46.7B parameters, its ${DEF('active parameters',TIP.active())}. What does that save, and when does the saving disappear?</p>`,
              `<p>${MODEL} 有 46.7B 个${DEF('参数',TIP.param())}，Mistral AI 却说它的速度和成本与 12.9B 的模型相当。它是一个${DEF('混合专家（MoE）',TIP.moe())}模型：每一层的 ${DEF('MLP 块',TIP.mlp())}换成了 8 个${DEF('专家',TIP.expert())}，再由一个小${DEF('路由器',TIP.router())}把每个 token 送到其中 2 个（见${J('flow','图里的示意')}）。</p><p>所以每个 token 只用 46.7B 里的 12.9B 个参数来计算，这就是它的${DEF('激活参数',TIP.active())}。这省下了什么？什么时候又省不下来？</p>`)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'par',gate:()=>!!pred,skip:true,why:()=>tx('Pick an answer, or skip this part, to go on.','选一个答案（或者跳过这一部分），才能进入下一部分。'),
  lead:()=>tx('Guess first: how much GPU memory do the weights need?','先猜：这些权重要占多少显存？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Each token computes with 12.9B of ${MODEL}’s 46.7B parameters. Stored as BF16, 2 bytes per parameter, how much GPU memory do its weights need, in ${DEF('GiB',TIP.gib())}? One ${GPUN} has 80 GB.</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 4 shows the result.</p>':''}`,
             `<p>每个 token 只用 ${MODEL} 46.7B 个参数里的 12.9B 个来计算。用 BF16 存（每个参数 2 字节），它的权重要占多少显存（按 ${DEF('GiB',TIP.gib())} 算）？一张 ${GPUN} 有 80 GB。</p>${ch}${pred?'<p class="muted small">已记下。第 4 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'router',title:{en:'The router picks 2 of 8',zh:'路由器从 8 个里选 2 个'},focus:'flow',
  lead:()=>tx('Route a few tokens: each one goes to its own 2 experts.','送几个 token 过去：每个 token 去自己的 2 个专家。'),
  enter:()=>{rt=0; return {...BASE};},expect:()=>rt>=3,showMe:()=>{rt=SENT.length; render();},label:{en:'Observed',zh:'观察到'},
  why:()=>tx(`Route ${3-rt} more token${3-rt===1?'':'s'} to go on.`,`再送出 ${3-rt} 个 token，才能进入下一部分。`),
  body:()=>tx(`<p>In every layer, the router gives each token a score for each of the 8 experts, and ${DEF('softmax',TIP.softmax())} turns the scores into shares. The token goes to the 2 experts with the largest shares, and their two outputs are added, weighted by those shares.</p><p>Send the tokens of a sentence through one layer, one by one, and see which 2 experts each token goes to.</p><div class="moe-rt"><button type="button" class="btn sm" data-act="route"${rt>=SENT.length?' disabled':''}>${rt>=SENT.length?tx('All six tokens routed','六个 token 都送完了'):tx(`Route “${SENT[rt]}”`,`送出“${SENT[rt]}”`)}</button><span class="small muted">${tx(`${rt} of 6 tokens routed`,`已送出 6 个 token 里的 ${rt} 个`)}</span></div>`,
              `<p>每一层里，路由器给每个 token 的 8 个专家各打一个分，${DEF('softmax',TIP.softmax())} 把得分变成份额。token 被送到份额最大的 2 个专家，这两个专家的输出按份额加权相加。</p><p>把一句话的 token 一个一个送过这一层，看看每个 token 去了哪 2 个专家。</p><div class="moe-rt"><button type="button" class="btn sm" data-act="route"${rt>=SENT.length?' disabled':''}>${rt>=SENT.length?tx('All six tokens routed','六个 token 都送完了'):tx(`Route “${SENT[rt]}”`,`送出“${SENT[rt]}”`)}</button><span class="small muted">${tx(`${rt} of 6 tokens routed`,`已送出 6 个 token 里的 ${rt} 个`)}</span></div>`),
  obs:()=>obsRouter()},
 {id:'mem',title:{en:'Compute versus memory',zh:'计算和显存'},focus:'par',
  lead:()=>tx('Each token computes with 12.9B parameters, but where do all 46.7B go?','每个 token 用 12.9B 个参数计算，那 46.7B 个都放在哪？'),
  enter:()=>({...BASE,p:'bf16'}),expect:()=>S.p==='fp8',always:true,showMe:{p:'fp8'},label:{en:'Result',zh:'结果'},why:()=>tx('Switch to FP8 to go on.','切到 FP8，才能进入下一部分。'),
  body:()=>tx(`<p>A token’s arithmetic follows ${DEF('the parameters it multiplies',TIP.mult())}: 2 ${DEF('FLOPs',TIP.flop())} for each, about 25.5 GFLOP (the embedding table is only looked up). GPU memory is another matter. Switch how the weights are stored:</p>`,
              `<p>一个 token 的计算量跟着它${DEF('要乘的参数',TIP.mult())}走（embedding 表只查表、不做乘法）：每个参数 2 ${DEF('FLOP',TIP.flop())}，约 25.5 GFLOP。显存却是另一回事。切换权重的存储格式：</p>`)+pCtl('moe-ic-p',true),
  obs:()=>obsMem()},
 {id:'batch',title:{en:'Batch the decode',zh:'把 decode 攒成 batch'},focus:'read',
  lead:()=>tx('More requests in a step pick more experts, and every picked expert is read.','一步里的请求越多，被选中的专家就越多，而每个被选中的专家都要读。'),
  enter:()=>({...BASE}),expect:()=>S.B>1,showMe:{B:64},label:{en:'Observed',zh:'观察到'},why:()=>tx('Drag the number of requests above 1 to go on.','把请求数拖到 1 以上，才能进入下一部分。'),
  body:()=>tx(`<p>A server decodes many requests at once: each ${DEF('decode step',TIP.dstep())} computes the newest token of each of B requests, a ${DEF('batch',TIP.batch())}. Memory reads set a decode step’s time (${labLink('prefill-decode')}), and a step reads every expert that ${DEF('any of its tokens picked',TIP.read())}. Drag the number of requests:</p>`,
              `<p>服务端会同时 decode 很多个请求：每个 ${DEF('decode 步',TIP.dstep())}算出 B 个请求各自的最新 token，这 B 个请求就是一个 ${DEF('batch',TIP.batch())}。decode 步的耗时由读显存决定（见 ${labLink('prefill-decode')}），而一步要读${DEF('它的 token 选中的每一个专家',TIP.read())}。拖动请求数：</p>`)+
    bCtl('moe-ic-B','B',memOf('fp8',S.c).max,tx('Requests decoding together','一起 decode 的请求数'),true),
  obs:()=>obsBatch()},
 {id:'adv',title:{en:'Then where is the advantage?',zh:'那优势在哪？'},focus:'cmp',
  lead:()=>tx('Against a dense model of the same quality, Mixtral computes and reads far less; against one that computes as much, it needs more memory.','和质量相当的稠密模型比，Mixtral 算得少得多，读得也少；和计算量相同的比，它更占显存。'),
  enter:()=>{cmp='l13'; cmpSeen=new Set(['l13']); return {...BASE};},expect:()=>cmpSeen.has('l70'),always:true,showMe:()=>setCmp('l70'),label:{en:'Result',zh:'结果'},
  why:()=>tx('Look at Llama 2 70B, the dense model of the same quality, to go on.','看一看质量相当的稠密模型 Llama 2 70B，才能进入下一部分。'),
  body:()=>tx(`<p>A busy server reads almost every expert in every step. So where is the advantage of MoE?</p><p class="small">Pick a dense model to compare with Mixtral and see where MoE wins and where it loses.</p>`,
              `<p>服务端一忙起来，每一步几乎要读遍所有专家。那 MoE 的优势在哪？</p><p class="small">选一个稠密模型和 Mixtral 比，看看 MoE 赢在哪里、输在哪里。</p>`)+cmpLevers(),
  obs:()=>obsCmp()},
 {id:'imb',title:{en:'When the router plays favourites',zh:'路由器偏心的时候'},focus:'ep',
  lead:()=>tx('Spread over 8 GPUs, a layer waits for its busiest expert.','专家分到 8 张 GPU 上以后，一层要等最忙的那个专家。'),
  enter:()=>{skew='bal'; return {...BASE};},expect:()=>skew!=='bal',showMe:()=>setSkew('h'),label:{en:'Result',zh:'结果'},why:()=>tx('Pick a spread where one expert is popular to go on.','选一个让某个专家很热门的分法，才能进入下一部分。'),
  body:()=>tx(`<p>Large deployments spread the experts over GPUs (${DEF('expert parallelism',TIP.ep())}): here 8 ${GPUN} hold one expert of every layer each. In a prefill of 4,096 tokens, each token goes to 2 experts, so the experts get 8,192 token copies in all, and each GPU computes the copies its expert got. Arithmetic sets a prefill’s time, so the layer is done when the busiest GPU is done.</p><p class="small">Pick how the router spreads the tokens and see how long the layer takes.</p>`,
              `<p>大规模部署会把专家分到多张 GPU 上（${DEF('专家并行',TIP.ep())}）：这里 8 张 ${GPUN} 各存每一层的一个专家。prefill 4,096 个 token 时，每个 token 去 2 个专家，所以专家们一共收到 8,192 份 token 副本，每张 GPU 算它那个专家分到的副本。prefill 的耗时由计算决定，所以最忙的 GPU 算完，这一层才算完。</p><p class="small">选一种分配方式，看看这一层要算多久。</p>`)+skewLevers(),
  obs:()=>obsEP()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:'',
  lead:()=>tx('Memory holds every expert; arithmetic and reads follow the experts picked.','显存要放下每个专家；计算和读显存跟着被选中的专家走。'),
  enter:()=>({...BASE,B:64}),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);

/* part 6: Mixtral against the dense model the learner picks; every number from cmpModels() or the paper's Table 2 */
const CMPN={l13:'Llama 2 13B',mx:'Mixtral‑8x7B',l70:'Llama 2 70B',m7:'Mistral 7B'};
const cmpName=id=>id==='d47'?tx('Dense 46.7B','稠密 46.7B'):CMPN[id];
const cmpTitle=id=>({l13:'Llama 2 13B',l70:'Llama 2 70B',d47:tx('Dense 46.7B','稠密 46.7B')}[id]);
const cmpDesc=id=>({l13:tx(`same arithmetic per token: a dense model with ${big(cmpOf('l13').total)} parameters, all used by every token`,`每个 token 计算量相同：${big(cmpOf('l13').total)} 个参数的稠密模型，每个 token 用全部参数`),
  l70:tx(`about the same quality: a dense model with ${big(cmpOf('l70').total)} parameters`,`质量相当：${big(cmpOf('l70').total)} 个参数的稠密模型`),
  d47:tx('same memory, hypothetical: Mixtral’s parameters, all used by every token; nobody trained it','显存相同，假想的：参数和 Mixtral 一样多，每个 token 用全部参数；没有人训练过它')}[id]);
/* the option's result says who is compared with whom (R6.6) and follows the benchmark picked on the stage (R1.9) */
const cmpRes=id=>{const m=cmpOf('mx'), b=cmpOf(id), bi=BENCH_SHOW[bench], bn=BENCH_COLS[bi], d=fixd(BENCH.mx[bi]-BENCH.l13[bi],1);
  return id==='l13'?tx(`Mixtral +${d} points on ${bn}`,`Mixtral 的 ${bn} 高 ${d} 分`):tx(`${fixd(b.fl/m.fl,1)}× Mixtral’s arithmetic`,`计算量是 Mixtral 的 ${fixd(b.fl/m.fl,1)} 倍`);};
const cmpLevers=()=>A.ui.levers({aria:tx('Compare Mixtral with, pick one','拿 Mixtral 和谁比，选一个'),value:cmp,options:['l13','l70','d47'].map(id=>({v:id,title:cmpTitle(id),desc:cmpDesc(id),result:esc(cmpRes(id))}))});
function setCmp(id){if(!['l13','l70','d47'].includes(id)) return; cmp=id; cmpSeen.add(id); render();}
function obsCmp(){
  const m=cmpOf('mx'), b=cmpOf(cmp), bi=BENCH_SHOW[bench], bn=BENCH_COLS[bi], q=x=>fixd(x,1)+'%', fr=fixd(b.fl/m.fl,1);
  if(cmp==='l13') return tx(`<p><b>${MODEL} computes about as much per token as Llama 2 13B, yet scores higher on ${bn}: ${q(BENCH.mx[bi])} against ${q(BENCH.l13[bi])}.</b></p>`+
      `<p>Part of the gap is training: Mistral 7B, a dense model from the same team, scores ${q(BENCH.m7[bi])} on ${bn} with about half the arithmetic of either model. Mixtral adds ${fixd(BENCH.mx[bi]-BENCH.m7[bi],1)} points over it with ${fixd(P_MULT/M7.mult,1)} times its arithmetic, so not all of that is MoE either. Yet with the arithmetic per token fixed, the ${cite(SRC.switch,'Switch Transformers paper')} finds that more experts give a better model.</p>`+
      `<p>The price is memory: ${MODEL} holds ${fixd(m.mem/b.mem,1)} times the weights of Llama 2 13B.</p>`,
    `<p><b>Mixtral 每个 token 的计算量和 Llama 2 13B 差不多，${bn} 却更高：${q(BENCH.mx[bi])} 对 ${q(BENCH.l13[bi])}。</b></p>`+
      `<p>差距不全是 MoE 的功劳：同一个团队的稠密模型 Mistral 7B 计算量约是它们的一半，${bn} 就有 ${q(BENCH.m7[bi])}；Mixtral 比它高 ${fixd(BENCH.mx[bi]-BENCH.m7[bi],1)} 分，计算量也是它的 ${fixd(P_MULT/M7.mult,1)} 倍，所以这 ${fixd(BENCH.mx[bi]-BENCH.m7[bi],1)} 分也不全来自 MoE。而 ${cite(SRC.switch,'Switch Transformer 论文')}固定每个 token 的计算量，发现专家越多，模型越好。</p>`+
      `<p>代价是显存：Mixtral 的权重是 Llama 2 13B 的 ${fixd(m.mem/b.mem,1)} 倍。</p>`);
  if(cmp==='l70'){const same=BENCH.mx[bi]-BENCH.l70[bi]<=1.5;
    return tx(`<p><b>Llama 2 70B ${same?`scores about the same as ${MODEL} on ${bn} (${q(BENCH.l70[bi])} against ${q(BENCH.mx[bi])})`:`scores lower than ${MODEL} on ${bn} (${q(BENCH.l70[bi])} against ${q(BENCH.mx[bi])}) and about the same over all 12 benchmarks`}, yet it computes ${fr} times as much per token and reads more weights in every decode step.</b></p>`+
      `<p>Why: Llama 2 70B multiplies and reads all of its parameters for every token. ${MODEL} multiplies only 2 experts per layer, and even a busy decode step reads no more than its own 46.7B parameters; it also needs less GPU memory.</p>`+
      `<p>In time: Mixtral prefills a ${tokK(CMP_N)}-token prompt in about ${secS(m.pre)} instead of ${secS(b.pre)}, and reads its weights in a busy decode step in about ${msR(m.tB)} instead of ${msR(b.tB)}.</p>`+
      `<p>So the advantage of MoE is quality per unit of arithmetic. It is largest where arithmetic sets the time, in training and in prefill, and it holds in decode against a dense model of the same quality.</p>`,
    `<p><b>Llama 2 70B 的 ${bn} 分数${same?`和 ${MODEL} 差不多（${q(BENCH.l70[bi])} 对 ${q(BENCH.mx[bi])}）`:`比 ${MODEL} 低（${q(BENCH.l70[bi])} 对 ${q(BENCH.mx[bi])}），12 个基准总体上差不多`}，每个 token 的计算量却是 Mixtral 的 ${fr} 倍，每个 decode 步读的权重也更多。</b></p>`+
      `<p>原因：Llama 2 70B 每个 token 都要乘、都要读全部参数；Mixtral 每层只乘 2 个专家，再忙的 decode 步也最多读 46.7B 个参数，占的显存也更少。</p>`+
      `<p>换成时间：Mixtral prefill ${tokK(CMP_N)} token 约 ${secS(m.pre)}，而不是 ${secS(b.pre)}；繁忙的 decode 步读权重约 ${msR(m.tB)}，而不是 ${msR(b.tB)}。</p>`+
      `<p>所以 MoE 的优势是“每单位计算换来的质量”：在由计算决定耗时的训练和 prefill 里最大，decode 时和质量相当的稠密模型比也占优。</p>`);}
  return tx(`<p><b>This hypothetical dense model has ${MODEL}’s 46.7B parameters and needs the same memory, but computes ${fr} times as much per token.</b></p>`+
      `<p>Would it score higher? Most likely, yes, by a margin that depends on the MoE design: the ${cite(SRC.dsmoe,'DeepSeekMoE paper')} treats such a dense model as “the strict upper bound” of MoE. DeepSeekMoE’s own design nearly reaches it, while ${DEF('a conventional top-2 MoE like Mixtral’s stays clearly below it',DSM_TIP())} (at about 2B parameters and 100B training tokens).</p>`+
      `<p>So the dense model would buy some quality with ${fr} times the arithmetic; per unit of arithmetic, MoE wins. In decode at a large batch both read about the same weights per step, so there MoE saves nothing; for one request ${MODEL} reads about a quarter as much.</p>`,
    `<p><b>这个假想的稠密模型和 ${MODEL} 一样有 46.7B 个参数，需要同样的显存，每个 token 的计算量却是 Mixtral 的 ${fr} 倍。</b></p>`+
      `<p>它的分数会更高吗？多半会，高多少取决于 MoE 的设计：${cite(SRC.dsmoe,'DeepSeekMoE 论文')}把这样的稠密模型看作 MoE 的上限（“strict upper bound”），DeepSeekMoE 自己的设计几乎追平，而${DEF('像 Mixtral 这样常规的 top-2 MoE 明显低于它',DSM_TIP())}（约 2B 个参数、100B 个训练 token 的实验）。</p>`+
      `<p>所以这个稠密模型用 ${fr} 倍的计算量换来一些质量，按每单位计算，MoE 更划算。decode 时 batch 大的话，两者每一步读的权重差不多，MoE 什么也没省；只有一个请求时，Mixtral 只读约四分之一。</p>`);
}
/* DeepSeekMoE's Tables 1 and 2 (arXiv 2401.06066), for the dense 46.7B answer */
const DSM_TIP=()=>tx('Pile loss in the DeepSeekMoE paper (lower is better): a dense model of the same total size (Dense×16) 1.806, DeepSeekMoE 1.808, GShard 1.867. GShard routes like Mixtral, to the top 2 of 16 full-size experts with no shared expert; HellaSwag 55.1, 54.8 and 50.5. All at about 2B parameters and 100B training tokens (Tables 1 and 2).',
  'DeepSeekMoE 论文里的 Pile loss（越低越好）：总参数相同的稠密模型（Dense×16）1.806，DeepSeekMoE 1.808，GShard 1.867。GShard 的路由和 Mixtral 一样，从 16 个完整大小的专家里选前 2 个，没有共享专家；HellaSwag 分别是 55.1、54.8 和 50.5。规模都约是 2B 个参数、100B 个训练 token（表 1 和表 2）。');
/* part 3: the router, from the tokens routed so far */
function obsRouter(){
  if(!rt) return '';
  const cur=routeTok(rt-1), used=new Set(); for(let i=0;i<rt;i++) routeTok(i).top.forEach(e=>used.add(e));
  const [a,b]=cur.top, pa=fint(Math.round(cur.w[0]*100)), pb=fint(100-Math.round(cur.w[0]*100));
  const lead=tx(`<p><b>“${cur.tok}” goes to ${expN(a)} and ${expN(b)}, weighted ${pa}% and ${pb}%.</b> ${rt===1?'The other 6 experts do nothing for it.':`So far, ${rt} tokens have used ${used.size} different experts of this layer.`}</p>`,
                `<p><b>“${cur.tok}”被送到${expN(a)}和${expN(b)}，权重分别是 ${pa}% 和 ${pb}%。</b>${rt===1?'另外 6 个专家不为它做任何计算。':`到现在，${rt} 个 token 一共用到了这一层的 ${used.size} 个不同的专家。`}</p>`);
  const why=rt<3?tx('<p>Route a few more tokens to see which experts they pick.</p>','<p>再送几个 token，看它们选哪些专家。</p>')
    :tx(`<p>Why: each token computes with only its own 2 experts, a quarter of the layer’s expert parameters. But the next token may pick any 2, so over a few tokens most experts of the layer get used.</p>`,
        `<p>原因：每个 token 只用自己的 2 个专家来计算，只占这一层专家参数的四分之一。但下一个 token 可能选任意 2 个，所以几个 token 下来，这一层的大多数专家都会被用到。</p>`);
  return lead+why;
}
/* the verdict on the prediction (BF16: 87.0 GiB) */
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), r=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${r}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${r}”。</p>`);
}
function obsMem(){
  const bf=memOf('bf16',S.c), f8=memOf('fp8',S.c);
  if(S.p==='bf16') return tx(`<p><b>In BF16 the weights of all 46.7B parameters take ${gib1(bf.w)}, more than the ${gib1(bf.usable)} one ${GPUN} can use, although each token uses only 12.9B of them.</b></p>`,
                            `<p><b>用 BF16 存，全部 46.7B 个参数的权重要占 ${gib1(bf.w)}，超过一张 ${GPUN} 能用的 ${gib1(bf.usable)}，虽然每个 token 只用其中 12.9B 个。</b></p>`)+verdict()+
    tx(`<p>Why: any token may pick any expert, so every expert must wait in GPU memory; only the arithmetic shrinks. Now switch to ${DEF('FP8',TIP.fp8())}.</p>`,`<p>原因：任何 token 都可能选任何专家，所以每个专家都得在显存里候着；省下来的只是计算。现在切到 ${DEF('FP8',TIP.fp8())}。</p>`);
  return tx(`<p><b>In ${DEF('FP8',TIP.fp8())} the weights take ${gib1(f8.w)} and fit on one ${GPUN}, with ${gib1(f8.budget)} left for the KV cache.</b> In BF16 they took ${gib1(bf.w)}.</p>`,
            `<p><b>用 ${DEF('FP8',TIP.fp8())} 存，权重占 ${gib1(f8.w)}，一张 ${GPUN} 放得下，还剩 ${gib1(f8.budget)} 给 KV cache。</b>BF16 时要 ${gib1(bf.w)}。</p>`)+verdict()+
    tx(`<p>Why: halving the bytes per parameter halves the memory of every expert, used or not. The rest of this lab runs ${MODEL} in FP8 on one ${GPUN}.</p>`,`<p>原因：每个参数的字节数减半，每个专家的显存都减半，不管用没用到。这个 Lab 后面都用 FP8 在一张 ${GPUN} 上跑 ${MODEL}。</p>`);
}
/* part 5: the batch, from the learner's own batch size (trend and principle in words, R6.20) */
function obsBatch(){
  const r=R(), one=compute({...S,B:1});
  // BF16 picked in this part: the weights do not fit, so no decode step runs (round 3: this threw on r.d.t)
  if(!r.m.fits) return tx(`<p><b>${PN[r.p]}: the weights alone need ${gib1(r.m.w)}, more than the ${gib1(r.m.usable)} one ${GPUN} can use, so not one decode step runs.</b> Switch back to FP8 to see what the number of requests does.</p>`,
                          `<p><b>${PN[r.p]}：光是权重就要 ${gib1(r.m.w)}，超过一张 ${GPUN} 能用的 ${gib1(r.m.usable)}，所以一个 decode 步也跑不了。</b>换回 FP8，再看请求数的影响。</p>`);
  if(r.B===1) return tx(`<p><b>One request: each decode step reads 2 experts per layer and takes about ${msR(r.d.t)}.</b> Drag the number of requests up.</p>`,
                        `<p><b>一个请求：每个 decode 步每层读 2 个专家，约 ${msR(r.d.t)}。</b>把请求数往上拖。</p>`);
  const q=r.d.t/one.d.t, all=r.d.T>=7.95;
  const res=tx(`<p><b>${reqN(r.B)} at ${tokK(r.c)} tokens each: a step reads about ${T1(r.d.T)} of the 8 experts per layer and takes about ${msR(r.d.t)}, ${ratioS(q)} times as long as for one request.</b></p>`,
               `<p><b>${reqN(r.B)}，每个 ${tokK(r.c)} 个 token：一步每层约读 8 个专家里的 ${T1(r.d.T)} 个，约 ${msR(r.d.t)}，是一个请求时的 ${ratioS(q)} 倍。</b></p>`);
  const why=all?tx(`<p>Why: with this many tokens, almost every expert is picked by someone, so each step reads nearly all 46.7B parameters, like a dense model of that size. From here, each extra request adds only the read of its own KV cache.</p>`+
                     `<p>So against a dense model of ${MODEL}’s own size the saving is gone. Against Llama 2 70B, a dense model of about the same quality, a step still reads fewer weights, and each token computes about a fifth as much: ${PL('adv','the next part')} compares them.</p>`,
                   `<p>原因：token 这么多，几乎每个专家都有人选，所以每一步都要读几乎全部 46.7B 个参数，和同样大的稠密模型一样。再往上，每多一个请求只多读它自己的 KV cache。</p>`+
                     `<p>所以和同样大的稠密模型比，节省没有了。但和质量相当的稠密模型 Llama 2 70B 比，一步读的权重仍然更少，每个 token 的计算量也只有 Llama 2 70B 的约五分之一：${PL('adv','下一部分')}会比较它们。</p>`)
    :tx(`<p>Why: each request’s token picks its own 2 experts, so the step reads the union of their picks. A few requests already cover most of the 8, and the step grows much faster than for a dense model, whose weights are read once whatever the batch.</p>`,
        `<p>原因：每个请求的 token 各选自己的 2 个专家，这一步要读的是它们选中的专家的并集。几个请求就已经覆盖了 8 个里的大多数，所以这一步变长得比稠密模型快得多：稠密模型不管 batch 多大，权重都只读一遍。</p>`);
  return res+why;
}
/* part 6: the router's spread over 8 GPUs */
function obsEP(){
  const r=epRun(skew), b=epRun('bal');
  if(skew==='bal') return tx(`<p><b>Balanced: every GPU gets ${fint(EP_COPIES/MX.E)} copies and the layer takes ${ms2(r.t)}.</b> Pick a spread where one expert is popular.</p>`,
                             `<p><b>均衡时：每张 GPU 分到 ${fint(EP_COPIES/MX.E)} 份，这一层要 ${ms2(r.t)}。</b>选一个让某个专家很热门的分法。</p>`);
  const k=Math.round(r.ratio), kw=k===2?tx('twice as long','耗时变成 2 倍'):tx(`${k} times as long`,`耗时变成 ${k} 倍`);
  const res=tx(`<p><b>Expert 3 gets ${fint(Math.round(r.loads[EP.hot]))} copies: the layer takes ${ms2(r.t)} instead of ${ms2(b.t)}, ${kw}, while the other 7 GPUs wait.</b></p>`,
               `<p><b>专家 3 分到 ${fint(Math.round(r.loads[EP.hot]))} 份：这一层要 ${ms2(r.t)}，均衡时是 ${ms2(b.t)}，${kw}，另外 7 张 GPU 都在等。</b></p>`);
  const why=tx(`<p>Why: each GPU’s time follows the copies it got, and the layer waits for the slowest. Training adds a ${DEF('load-balancing loss',TIP.lbl())} so the router spreads tokens more evenly; servers can also keep extra copies of popular experts.</p>`,
               `<p>原因：每张 GPU 的耗时跟着它分到的份数走，而这一层要等最慢的那张。所以训练时加一项${DEF('负载均衡 loss',TIP.lbl())}，让路由器把 token 分得更匀；服务端也可以给热门专家多放几份副本。</p>`);
  return res+why;
}
function remember(){
  const bf=memOf('bf16',2048), one=compute({...BASE}), b64=compute({...BASE,B:64}), mxc=cmpOf('mx'), l70=cmpOf('l70'), k=fixd(l70.fl/mxc.fl,1), x5=fint(Math.round(b64.d.t/one.d.t));
  /* small pictures, one per point (R6.23): the BF16 weights against one GPU; 2 of 8 experts; 1 request against 64; the arithmetic of the two models */
  const pic=(html,tip)=>`<span class="moe-rm-pic" tabindex="0" role="img" aria-label="${esc(tip)}" data-tipx="${esc(tip)}">${html}</span>`;
  const sq=(n,lab)=>`<span class="moe-rm-row">${lab?`<em class="mono">${esc(lab)}</em>`:''}<span class="moe-rm-ex">${Array.from({length:MX.E},(_,i)=>`<i${i<n?' class="on"':''}></i>`).join('')}</span></span>`;
  const sc=bf.cap*1.125, P=v=>(v/sc*100).toFixed(1)+'%';
  const p1=pic(`<span class="moe-rm-bar"><i class="moe-rm-w" style="width:${P(bf.usable)}"></i><i class="moe-over" style="left:${P(bf.usable)};width:${P(bf.w-bf.usable)}"></i><b style="left:${P(bf.usable)}"></b><u style="left:${P(bf.cap)}"></u></span><span class="moe-rm-cap mono">${gib1(bf.w)}</span>`,
    tx(`The BF16 weights, ${gib1(bf.w)}, against one ${GPUN}: blue fits in the ${gib1(bf.usable)} it can use (the dashed line), red does not; the thin line is its 80 GiB.`,`BF16 权重 ${gib1(bf.w)}，对比一张 ${GPUN}：蓝色放得进它能用的 ${gib1(bf.usable)}（虚线），红色放不下；细线是它的 80 GiB。`));
  const p2=pic(sq(2)+`<span class="moe-rm-cap mono">${tx('2 of 8','8 选 2')}</span>`,tx('One token, one layer: 2 of the 8 experts compute it.','一个 token、一层：8 个专家里有 2 个为它计算。'));
  const p3=pic(sq(2,'1')+sq(8,'64')+`<span class="moe-rm-cap mono">${tx('requests','请求数')}</span>`,tx(`One layer of one decode step: 1 request reads 2 of the 8 experts; 64 requests read all 8 (on average ${T1(b64.d.T)}).`,`一个 decode 步里的一层：1 个请求读 8 个专家里的 2 个；64 个请求读满 8 个（平均 ${T1(b64.d.T)} 个）。`));
  const p4=pic(`<span class="moe-rm-row"><em class="mono">8x7B</em><span class="moe-rm-f"><i class="mx" style="width:${(mxc.fl/l70.fl*100).toFixed(1)}%"></i></span></span><span class="moe-rm-row"><em class="mono">70B</em><span class="moe-rm-f"><i style="width:100%"></i></span></span><span class="moe-rm-cap mono">GFLOP</span>`,
    tx(`Arithmetic per token: ${MODEL} ${fixd(mxc.fl/1e9,1)} GFLOP, Llama 2 70B ${fixd(l70.fl/1e9,1)} GFLOP, ${k} times as much.`,`每个 token 的计算量：${MODEL} ${fixd(mxc.fl/1e9,1)} GFLOP，Llama 2 70B ${fixd(l70.fl/1e9,1)} GFLOP，是 Mixtral 的 ${k} 倍。`));
  const li=(pc,t)=>`<li>${pc}<div class="moe-rm-t">${t}</div></li>`;
  return tx(`<ul class="moe-rm">`+
      li(p1,`<b>Size GPU memory for every parameter, not only the active ones.</b> Any token may pick any expert, so every expert must sit in GPU memory. MoE saves arithmetic, not memory: stored in BF16, ${MODEL} does not fit on one ${GPUN} ${PLp('mem')}.`)+
      li(p2,`<b>Count a token’s arithmetic from the parameters it uses.</b> Each token computes only with the experts its router picks ${PLp('router')}, like a dense model of 12.9B.`)+
      li(p3,`<b>Predict decode time from the experts a batch reads.</b> One request reads 2 experts per layer; a batch of a few requests already reads most of them ${PLp('batch')}, and a busy server reads nearly the whole model in every step.`)+
      li(p4,`<b>Compare MoE with a dense model of the same quality, not the same size.</b> A dense model of about the same quality, such as Llama 2 70B, computes about ${k} times as much per token and reads more weights in every step ${PLp('adv')}. That is what MoE buys; the price is memory for every expert.`)+`</ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “MoE needs only the memory of the active parameters.” Every expert must sit in GPU memory, because any token may pick it. So the memory follows all 46.7B parameters, although each token computes with only 12.9B of them.</p><p>✗ “8x7B means 56B parameters.” Only the MLP blocks are copied into experts; attention, the embedding table and the LM head are shared, so ${MODEL} holds 46.7B.</p><p>✗ “MoE decodes faster at every batch size.” With 64 requests in a batch, every step reads every expert and takes about ${x5} times as long as with one request.</p><p>✗ “MoE saves nothing, because a big batch reads every expert.” A big batch removes only the decode-time saving in weight reads against a dense model of ${MODEL}’s own size. Even when a step reads every expert, it reads ${gbS(mxc.readB)} of weights; Llama 2 70B, which scores about the same, reads ${gbS(l70.readB)} in every step.</p></div>${A.shell.toScenarios()}`,
    `<ul class="moe-rm">`+
      li(p1,`<b>按全部参数准备显存，而不只是激活参数。</b>任何 token 都可能选任何专家，所以每个专家都得放在显存里。MoE 省的是计算，不是显存。用 BF16 存的话，一张 ${GPUN} 就放不下 ${MODEL}${PLp('mem')}。`)+
      li(p2,`<b>按一个 token 用到的参数算它的计算量。</b>每个 token 只用路由器选中的专家计算${PLp('router')}，和 12.9B 的稠密模型一样。`)+
      li(p3,`<b>按一个 batch 读到的专家估算 decode 的耗时。</b>一个请求每层读 2 个专家；几个请求合起来就读到了大多数专家${PLp('batch')}，繁忙的服务端每一步几乎要把整个模型读一遍。`)+
      li(p4,`<b>拿 MoE 和质量相当的稠密模型比，而不是和同样大的比。</b>质量相当的稠密模型（比如 Llama 2 70B），每个 token 的计算量约是 ${MODEL} 的 ${k} 倍，每一步读的权重也更多${PLp('adv')}。这就是 MoE 换来的好处，代价是每个专家都要占显存。`)+`</ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “MoE 只需要放下激活参数的显存。”任何 token 都可能选中任何专家，所以每个专家都得放在显存里。显存要按全部 46.7B 个参数算，虽然每个 token 只用其中的 12.9B 个计算。</p><p>✗ “8x7B 就是 56B 个参数。”只有 MLP 块被复制成了专家；attention、embedding 表和 LM head 是共享的，所以 ${MODEL} 有 46.7B 个参数。</p><p>✗ “MoE 在任何 batch 大小下 decode 都更快。”batch 里有 64 个请求时，每一步都要读遍所有专家，耗时约是只有一个请求时的 ${x5} 倍。</p><p>✗ “大 batch 要读遍每个专家，所以 MoE 什么也没省。”大 batch 只抹掉了它相对同样大小（46.7B）的稠密模型在 decode 时读权重的节省。就算一步读遍所有专家，读的权重也只有 ${gbS(mxc.readB)}；分数相当的 Llama 2 70B 每一步都要读 ${gbS(l70.readB)}。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real deployments, one plain phrase each (R2.1)
   ===================================================================== */
const PRESETS=[
 {id:'one',t:{en:'One user chatting',zh:'一个用户在聊天'},d:{en:'one request decoding, 2K tokens of context',zh:'一个请求在 decode，上下文 2K 个 token'},s:{p:'fp8',c:2048,B:1}},
 {id:'team',t:{en:'A small team',zh:'一个小团队'},d:{en:'8 requests decoding together, 2K tokens each',zh:'8 个请求一起 decode，每个 2K 个 token'},s:{p:'fp8',c:2048,B:8}},
 {id:'busy',t:{en:'Busy chat server',zh:'繁忙的聊天服务'},d:{en:'64 requests decoding together, 2K tokens each',zh:'64 个请求一起 decode，每个 2K 个 token'},s:{p:'fp8',c:2048,B:64}},
 {id:'docs',t:{en:'Questions about long documents',zh:'针对长文档提问'},d:{en:'4 requests with 16K tokens of context each',zh:'4 个请求，每个 16K 个 token 的上下文'},s:{p:'fp8',c:16384,B:4}},
 {id:'bf16',t:{en:'The weights as released, in BF16',zh:'按发布时的 BF16 权重'},d:{en:'one request, 2K tokens, on one H100 SXM',zh:'一个请求，2K 个 token，一张 H100 SXM'},s:{p:'bf16',c:2048,B:1}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.p===S.p&&p.s.c===S.c&&p.s.B===S.B);
function scenHTML(i){
  const p=PRESETS[i], r=compute(p.s), name=esc(L(p.t));
  if(!r.m.fits) return tx(`<p><b>${name}: the weights alone need ${gib1(r.m.w)}, more than the ${gib1(r.m.usable)} one ${GPUN} can use.</b></p><p>Every expert must be in GPU memory, used or not. The weights are stored in FP8 on one GPU in the other scenarios; in BF16 they would need two GPUs.</p>`,
                          `<p><b>${name}：光是权重就要 ${gib1(r.m.w)}，超过一张 ${GPUN} 能用的 ${gib1(r.m.usable)}。</b></p><p>每个专家都得放在显存里，不管用没用到。别的场景都用 FP8 在一张 GPU 上存权重；用 BF16 就要两张 GPU。</p>`);
  const head=tx(`<p><b>${name}: each decode step reads about ${T1(r.d.T)} of the 8 experts per layer and takes about ${msR(r.d.t)}.</b></p>`,
                `<p><b>${name}：每个 decode 步每层约读 8 个专家里的 ${T1(r.d.T)} 个，约 ${msR(r.d.t)}。</b></p>`);
  let why='';
  if(p.id==='one') why=tx('<p>One token picks 2 experts per layer, so the step reads about a quarter of the expert parameters: the model decodes like a dense model of 12.9B.</p>','<p>一个 token 每层选 2 个专家，这一步只读约四分之一的专家参数：它 decode 起来像 12.9B 的稠密模型。</p>');
  if(p.id==='team') why=tx('<p>Eight tokens together pick almost every expert, so the step reads most of the model: each user waits more than three times as long per token as alone, while the GPU writes more tokens per second in all.</p>','<p>8 个 token 合起来几乎选遍了所有专家，这一步要读模型的大部分：每个用户每个 token 等的时间是单独用时的 3 倍多，GPU 每秒一共写出的 token 却更多了。</p>');
  if(p.id==='busy') why=tx(`<p>Every expert is read in every step, as for a dense model of 46.7B; the extra requests add only their KV caches. The GPU writes the most tokens per second in all, and each user waits the longest per token.</p><p>A dense model of about the same quality, Llama 2 70B, would read about 1.5 times as many weights in every step and compute over five times as much per token ${PLp('adv')}.</p>`,`<p>每一步都要读全部专家，和 46.7B 的稠密模型一样；多出来的请求只多读它们的 KV cache。GPU 每秒一共写出的 token 最多，每个用户每个 token 等得也最久。</p><p>质量相当的稠密模型 Llama 2 70B 每一步要读约 1.5 倍的权重，每个 token 的计算量是 Mixtral 的五倍多${PLp('adv')}。</p>`);
  if(p.id==='docs') why=tx('<p>Four long requests read about five and a half experts per layer, and each also reads a long KV cache; at 16K tokens each, GPU memory holds the KV caches of at most 13 requests.</p>','<p>4 个长请求每层约读 5 个半专家，每个还要读很长的 KV cache；每个 16K 个 token 时，显存最多放得下 13 个请求的 KV cache。</p>');
  return head+why;
}

/* =====================================================================
   challenge
   ===================================================================== */
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chPoint(S.B), best=chPoint(chBest()), nxt=chPoint(chBest()+1);
  let h;
  if(me.ok&&me.B===best.B) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${userN(me.B)}, each gets a token every ${ms2(me.t)}.</b> And that is the most: one more user pushes every token over 30 ms.</p>`,
                                `<p class="okline" style="margin-top:12px"><b>达成目标：${userN(me.B)}，每人每 ${ms2(me.t)} 拿到一个 token。</b>而且这已经是最多：再多一个用户，每个 token 就超过 30 ms。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${userN(me.B)}, each gets a token every ${ms2(me.t)}.</b> But more users fit: up to ${fint(best.B)} stay within 30 ms, and every user added raises the tokens per second in all.</p>`,
                      `<p class="okline" style="margin-top:12px"><b>达成目标：${userN(me.B)}，每人每 ${ms2(me.t)} 拿到一个 token。</b>不过还能多服务几个：最多 ${fint(best.B)} 个用户时仍在 30 ms 以内，而且每多一个用户，合起来每秒的 token 就多一些。</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: with ${userN(me.B)}, each token takes ${ms2(me.t)}, over 30 ms.</b></p><p class="small">Hint: once a step reads all 8 experts per layer, what does each extra user add to the step? How many of those fit in what is left of 30 ms?</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：${userN(me.B)}时，每个 token 要 ${ms2(me.t)}，超过了 30 ms。</b></p><p class="small">提示：一步每层已经读满 8 个专家以后，每多一个用户，这一步还多读什么？30 ms 剩下的时间里还放得下几份？</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const rows=CH.rows.slice(); if(!rows.includes(me.B)) rows.push(me.B); rows.sort((a,b)=>a-b);
  const pts=rows.map(chPoint);
  h+=A.ui.cmpChart({title:tx(`Each user’s time per output token, ${MODEL} in FP8 on one ${GPUN}, 4K tokens of context each (goal: at most 30 ms)`,`一张 ${GPUN} 用 FP8 跑 ${MODEL}、每人 4K 个 token 上下文时，每个用户每个输出 token 的耗时（目标：最多 30 ms）`),
    brief:true,hiddenText:n=>tx(`The other ${n} user counts take over 30 ms per token and are not shown.`,`另外 ${n} 种用户数下，每个 token 都超过 30 ms，图里没有列出。`),
    aria:tx('Time per output token, by the number of users: ','不同用户数下每个输出 token 的耗时：')+pts.filter(p=>p.ok||p.B===me.B||p.B===CH.start).map(p=>`${fint(p.B)}${tx(': ','：')}${ms2(p.t)}`).join(tx('; ','；')),
    rows:pts.map(p=>({label:tx(`${userN(p.B)}, ${fint(p.thr)} tokens per second in all`,`${userN(p.B)}，合计每秒 ${fint(p.thr)} 个 token`),value:p.t*1e3,text:ms2(p.t),ok:p.ok,me:p.B===me.B,
      tags:(p.B===CH.start?[{text:tx('initial setup','初始设置')}]:[]).concat(p.B===me.B?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:30,goalLabel:tx('goal: 30 ms','目标 30 ms'),max:50,okText:tx('30 ms or less','不超过 30 ms'),noText:tx('over 30 ms','超过 30 ms'),goalText:tx('goal: at most 30 ms','目标：最多 30 ms')});
  h+=tx(`<p class="small">Initial setup, 1 user: a step reads only 2 experts per layer and takes far less than 30 ms. A few users make it jump, because together they pick almost every expert. After that, each user adds only the read of their own KV cache: ${userN(best.B)} still fit within 30 ms, ${fint(nxt.B)} do not.</p>`,
        `<p class="small">初始设置是 1 个用户：一步每层只读 2 个专家，远不到 30 ms。几个用户就让它猛涨，因为他们合起来几乎选遍了所有专家。之后每多一个用户，只多读他自己的 KV cache：${userN(best.B)}还在 30 ms 以内，${fint(nxt.B)} 个就超了。</p>`);
  h+=tx(`<p class="small">Why: memory reads set a decode step’s time. The experts cost the same once all 8 are read, so the room left under 30 ms goes to the users’ KV caches, and each user adds the same amount.</p>`,
        `<p class="small">原因：decode 步的耗时由读显存决定。8 个专家都读到以后，专家这部分就不再变，30 ms 剩下的空间都留给用户的 KV cache，每个用户增加的量都一样。</p>`);
  h+=tx(`<p class="small">Llama 2 70B, of about the same quality, could not serve these users: it takes at most ${fint(L70.ctx)} tokens in all, which each user’s context already fills. At any context, its weights alone take ${msR(L70.read/BS)} per decode step, over the goal.</p>`,
        `<p class="small">质量相当的 Llama 2 70B 服务不了这些用户：它最多只能处理 ${fint(L70.ctx)} 个 token，每个用户的上下文已经占满了。不管上下文多长，光是读它的权重，每个 decode 步就要 ${msR(L70.read/BS)}，超过了目标。</p>`);
  h+=tx(`<p class="small muted">Note: this assumes uniform routing and 60% of the peak bandwidth, a share measured at batch size 1 that falls as the batch grows. A real router is not uniform, and a real FP8 checkpoint keeps a few parts in BF16; measure the time per output token at the batch size you plan before sizing a server.</p>`,
        `<p class="small muted">注意：这里假设路由均匀、带宽用到峰值的 60%；这个比例是 batch 为 1 时测的，batch 越大越低。真实的路由器并不均匀，真实的 FP8 checkpoint 也有几部分留在 BF16；给服务定规模之前，要按计划的 batch 大小实测每个输出 token 的耗时。</p>`);
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`Eight short parts. Predict first, then watch the router send each token to 2 of 8 experts, see what that saves and what it does not, how a batch of requests reads more experts, where the advantage lies against dense models of the same compute, memory or quality, and why a popular expert slows a layer down. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看路由器怎样把每个 token 送到 8 个专家里的 2 个，这省下了什么、没省下什么，一批请求为什么要读更多专家，和计算量、显存或质量相当的稠密模型比优势在哪，以及一个热门专家为什么会拖慢一整层。第一次学从这里开始。`),
  sandbox:tx(`Five real deployments of ${MODEL} on one ${GPUN}, from one user chatting to a busy server, and one that does not fit. Pick one to see how many experts each decode step reads.`,
             `五个真实部署，都是一张 ${GPUN} 跑 ${MODEL}，从一个用户聊天到繁忙的服务端，还有一个放不下的。选一个，看每个 decode 步要读几个专家。`),
  challenge:tx(`One ${GPUN} serves ${MODEL} in FP8 with 4K tokens of context per user, and every user must get a token at least every 30 ms. Change only the number of users, and serve as many as you can.`,
               `一张 ${GPUN} 用 FP8 跑 ${MODEL}，每个用户 4K 个 token 的上下文，每人每 30 ms 至少要拿到一个 token。只改用户数，尽量多服务一些用户。`)
});
const SRC={
 cfg:'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json',
 blog:'https://mistral.ai/news/mixtral-of-experts',
 paper:'https://arxiv.org/abs/2401.04088',
 shazeer:'https://arxiv.org/abs/1701.06538',
 switch:'https://arxiv.org/abs/2101.03961',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 fp8:'https://huggingface.co/neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8',
 dsv3:'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json',
 l13:'https://huggingface.co/NousResearch/Llama-2-13b-hf/blob/main/config.json',
 l70:'https://huggingface.co/NousResearch/Llama-2-70b-hf/blob/main/config.json',
 hf:'https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py',
 vmx:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py',
 vfm:'https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe',
 valign:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py',
 vker:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py',
 dsmoe:'https://arxiv.org/abs/2401.06066',
 m7:'https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json',
 db:'https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices'
};
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> the parameters are counted from ${MODEL}’s config.json, and every time comes from the step model of ${labLink('prefill-decode')}; ${J('math','“Show the math”')} works it out line by line. The pictures of single tokens being routed are illustrative.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>${MODEL}: 32 layers, hidden size 4,096, 8 experts per layer of 176M parameters each, 2 per token, 8 KV heads; 46.7B parameters in all, 12.9B per token.</li>`+
 `<li>A step takes the longer of its arithmetic (2 FLOPs per multiplied parameter per token, plus attention) and its memory reads (every weight it multiplies, once per step, plus 128 KiB of KV cache per token of context).</li>`+
 `<li>A decode step reads the shared parts (attention, norms, routers, LM head) and every expert at least one of its tokens picked.</li>`+
 `<li>GPU memory: the ${labLink('kv-cache')} lab’s model, 80 GB read as 80 GiB, ${pct0(UTIL)} usable, ${OH} GiB of runtime overhead.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>Routing is uniform and independent: a layer reads ${A.ui.math('8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{B}\\right)')} experts on average for B tokens. Real routers favour some experts over others, and for the same number of picks uniform routing reaches the most different experts, so a real batch reads somewhat fewer; the numbers are estimates. (Consecutive tokens of one text often pick the same expert, the Mixtral paper finds; that matters in a prefill, not in a decode batch of different requests.)</li>`+
 `<li>50% of the dense BF16 peak and 60% of the peak bandwidth are assumptions, as in the other labs. FP8 weights count 1 byte per parameter; the arithmetic is priced at the BF16 rate, which changes no decode step here, since memory reads set all of them. The prefill spread over 8 GPUs by expert parallelism is set by arithmetic: at the H100’s dense FP8 rate, 1,979 TFLOP/s, each of its times would halve, while its ratios and waiting shares would stay the same.</li>`+
 `<li>A real FP8 checkpoint keeps the embedding table, the LM head and the routers in BF16, 0.25 GiB more: at 2K or 4K tokens per request one request fewer fits, and every decode step also reads the LM head in BF16, about 0.07 ms more. No conclusion changes: the challenge’s answer stays 25 users.</li>`+
 `<li>The bars that set Mixtral against dense models count weights only: the KV cache comes on top in every decode step and depends on the attention design, not on MoE (per token: Llama 2 13B 800 KiB, Llama 2 70B 320 KiB, ${MODEL} 128 KiB); the note under the reads and the requests that fit show what it changes. Arithmetic counts the matrix multiplications, 2 FLOPs per multiplied parameter; attention over the context adds a few percent to every model. Mistral 7B is shown in the quality bars only.</li>`+
 `<li>The expert-parallel picture leaves out the all-to-all traffic between GPUs and the attention blocks; only the experts’ time is shown.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${MODEL}: ${cite(SRC.cfg,'config.json')}; Mistral AI, ${cite(SRC.blog,'“Mixtral of experts”')}: “46.7B total parameters but only uses 12.9B parameters per token”.</li>`+
 `<li>Jiang et al., ${cite(SRC.paper,'Mixtral of Experts (2024)')}: the router picks 2 of 8 experts per token and layer; routing analysis (§5); on expert parallelism (EP), the paper notes that “EP introduces challenges in load balancing, as it is essential to distribute the workload evenly across the GPUs”.</li>`+
 `<li>${GPUN}: ${cite(SRC.h100,'NVIDIA')}, BF16 1,979 TFLOPS “with sparsity”, so 989.5 dense; FP8 3,958 with sparsity, so 1,979 dense; 3.35 TB/s; 80 GB. Databricks, ${cite(SRC.db,'“LLM Inference Performance Engineering: Best Practices”')}: “At batch size 1, we can achieve a higher MBU of 60% on 2xH100-80GBs”, and “MBU decreases as batch size increases”.</li>`+
 `<li>FP8 checkpoint: ${cite(SRC.fp8,'neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8')}.</li>`+
 `<li>Llama 2 13B and 70B: config.json of ${cite(SRC.l13,'Llama-2-13b-hf')} and ${cite(SRC.l70,'Llama-2-70b-hf')} (meta-llama, read from the NousResearch mirrors); both take at most 4,096 tokens (max_position_embeddings), ${MODEL} 32,768. Mistral 7B: ${cite(SRC.m7,'Mistral-7B-v0.1 config.json')}, 7.24B parameters.</li>`+
 `<li>Benchmark scores: the Mixtral paper, Table 2, with the settings of its §3 (MMLU 5-shot; HellaSwag and HumanEval 0-shot; GSM8K 8-shot, maj@8); Mistral AI re-ran every model “with our own evaluation pipeline for fair comparison”. The caption: “Mixtral outperforms or matches Llama 2 70B performance on almost all popular benchmarks while using 5x fewer active parameters during inference”.</li>`+
 `<li>${cite(SRC.paper,'Mixtral paper')}, Table 2: ${MODEL} scores higher than Llama 2 13B on all ${beats('l13')} benchmarks, and higher than Llama 2 70B on ${beats('l70')} of the 12. Mistral 7B, faded, is a dense model from the same team.</li>`+
 `<li>Fedus et al., ${cite(SRC.switch,'Switch Transformers')} §3.1, T5-based models trained on the C4 corpus: “when keeping the FLOPS per token fixed, having more parameters (experts) speeds up training”.</li>`+
 `<li>Dai et al., ${cite(SRC.dsmoe,'DeepSeekMoE (2024)')} §4.3: “DeepSeekMoE nearly approaches the performance of Dense×16, which sets the strict upper bound of MoE models in terms of the model capacity”. Dense×16 runs all 1.89B of its FFN parameters for every token; DeepSeekMoE 2B holds the same 1.89B of expert parameters and runs 0.24B of them (0.3B of its 2.0B parameters in all); both were trained on 100B tokens. GShard, a conventional top-2 MoE of the same size (0.3B of 2.0B active), stays below both: Pile loss 1.867 against 1.808 and 1.806 (Tables 1 and 2).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-30. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>参数按 ${MODEL} 的 config.json 逐项数出来，每个时间都来自 ${labLink('prefill-decode')} 的耗时模型；${J('math','“展开公式”')}里有逐行算式。单个 token 被路由的图是示意。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>${MODEL}：32 层，隐藏维度 4,096，每层 8 个专家，每个 176M 个参数，每个 token 选 2 个，8 个 KV head；一共 46.7B 个参数，每个 token 用 12.9B 个。</li>`+
 `<li>一步的耗时取计算（每个参与乘法的参数每个 token 2 FLOP，再加 attention）和读显存（参与乘法的权重每一步读一遍，再加上下文里每个 token 128 KiB 的 KV cache）中较长的那个。</li>`+
 `<li>decode 步要读共享的部分（attention、norm、路由器、LM head），以及至少被它的一个 token 选中的每个专家。</li>`+
 `<li>显存：用 ${labLink('kv-cache')}这个 Lab 的模型，80 GB 按 80 GiB 计，可用 ${pct0(UTIL)}，运行时开销 ${OH} GiB。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>路由是均匀且相互独立的：B 个 token 时一层平均读 ${A.ui.math('8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{B}\\right)')} 个专家。真实的路由器会偏爱某些专家，而选择次数相同时，均匀路由选到的不同专家最多，所以真实的 batch 读到的专家会少一些；这些数都是估计值。（Mixtral 论文发现同一段文字里相邻的 token 常选同一个专家；这影响的是 prefill，而不是由不同请求组成的 decode batch。）</li>`+
 `<li>稠密 BF16 峰值的 50% 和峰值带宽的 60% 是假设，和别的 Lab 一样。FP8 权重按每个参数 1 字节算；计算仍按 BF16 的速度算，这不改变这里的任何 decode 步，因为它们都由读显存决定。专家分到 8 张 GPU 上的那次 prefill，耗时由计算决定。按 H100 稠密 FP8 的速度（1,979 TFLOP/s）算，它的每个时间都会减半，倍数和等待的比例则不变。</li>`+
 `<li>真实的 FP8 checkpoint 把 embedding 表、LM head 和路由器留在 BF16，多 0.25 GiB：每个请求 2K 或 4K 个 token 时少放一个请求，每个 decode 步也要按 BF16 读 LM head，约多 0.07 ms。结论都不变：挑战的答案仍是 25 个用户。</li>`+
 `<li>拿 Mixtral 和稠密模型比的那组条形只算权重。每个 decode 步还要另外读 KV cache，它取决于注意力的设计，和 MoE 无关（每个 token 分别是 Llama 2 13B 800 KiB，Llama 2 70B 320 KiB，${MODEL} 128 KiB）；读权重那一组下面的说明和放得下的请求数给出了它的影响。计算量只算矩阵乘法，每个参与乘法的参数 2 FLOP；对上下文的 attention 让每个模型都多百分之几。Mistral 7B 只出现在质量那一组条形里。</li>`+
 `<li>专家并行那张图不计 GPU 之间的 all-to-all 通信和 attention 块，只画专家的耗时。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${MODEL} 的 ${cite(SRC.cfg,'config.json')}；Mistral AI 在 ${cite(SRC.blog,'“Mixtral of experts”')} 里写道“46.7B total parameters but only uses 12.9B parameters per token”。</li>`+
 `<li>Jiang 等，${cite(SRC.paper,'Mixtral of Experts（2024）')}：路由器每层为每个 token 从 8 个专家里选 2 个；路由分析（第 5 节）；关于专家并行（EP），论文写道“EP introduces challenges in load balancing, as it is essential to distribute the workload evenly across the GPUs”。</li>`+
 `<li>${GPUN}：${cite(SRC.h100,'NVIDIA')}，BF16 1,979 TFLOPS 标着“with sparsity”，所以稠密是 989.5；FP8 带稀疏是 3,958，稠密是 1,979；3.35 TB/s；80 GB。Databricks 的${cite(SRC.db,'“LLM Inference Performance Engineering: Best Practices”')}：“At batch size 1, we can achieve a higher MBU of 60% on 2xH100-80GBs”，以及“MBU decreases as batch size increases”。</li>`+
 `<li>FP8 checkpoint：${cite(SRC.fp8,'neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8')}。</li>`+
 `<li>Llama 2 13B 和 70B：${cite(SRC.l13,'Llama-2-13b-hf')} 和 ${cite(SRC.l70,'Llama-2-70b-hf')} 的 config.json（meta-llama 的模型，读自 NousResearch 的镜像）；两者最多只能处理 4,096 个 token（max_position_embeddings），${MODEL} 是 32,768 个。Mistral 7B：${cite(SRC.m7,'Mistral-7B-v0.1 的 config.json')}，7.24B 个参数。</li>`+
 `<li>基准分数：Mixtral 论文表 2，设置见它的第 3 节（MMLU 5-shot；HellaSwag 和 HumanEval 0-shot；GSM8K 8-shot，maj@8）；Mistral AI 用自己的评测流程重跑了每个模型（“with our own evaluation pipeline for fair comparison”）。表注原文：“Mixtral outperforms or matches Llama 2 70B performance on almost all popular benchmarks while using 5x fewer active parameters during inference”。</li>`+
 `<li>${cite(SRC.paper,'Mixtral 论文')}表 2：${MODEL} 在全部 ${beats('l13')} 个基准上都高于 Llama 2 13B，在 12 个里有 ${beats('l70')} 个高于 Llama 2 70B。半透明的 Mistral 7B 是同一个团队的稠密模型。</li>`+
 `<li>Fedus 等，${cite(SRC.switch,'Switch Transformers')} 第 3.1 节，基于 T5、在 C4 语料上训练的模型：“when keeping the FLOPS per token fixed, having more parameters (experts) speeds up training”。</li>`+
 `<li>Dai 等，${cite(SRC.dsmoe,'DeepSeekMoE（2024）')}第 4.3 节：“DeepSeekMoE nearly approaches the performance of Dense×16, which sets the strict upper bound of MoE models in terms of the model capacity”。Dense×16 每个 token 都用它全部 1.89B 个 FFN 参数；DeepSeekMoE 2B 有同样多的 1.89B 个专家参数，每个 token 用其中 0.24B（全部 2.0B 个参数里用 0.3B）；两者都训练了 100B 个 token。同样大小的常规 top-2 MoE GShard（2.0B 里激活 0.3B）比两者都差：Pile loss 1.867，对 1.808 和 1.806（表 1 和表 2）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-30。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A dense model uses every parameter for every token. Most of them sit in the MLP blocks; the attention blocks, the embedding table and the LM head hold the rest. The lab ${labLink('transformer')} counts them from config.json.</p><p>A decode step reads every weight it multiplies from GPU memory once, and those reads set its time (${labLink('prefill-decode')}).</p>`,
                     `<p>稠密模型的每个 token 都要用到全部参数。大部分参数在 MLP 块里，其余在 attention 块、embedding 表和 LM head 里。${labLink('transformer')}这个 Lab 按 config.json 把它们数了一遍。</p><p>decode 步把它要乘的全部参数从显存读一遍，读显存的时间决定它的耗时（见 ${labLink('prefill-decode')}）。</p>`);
/* the capability layer (Batch 2) */
const OUTCOME=()=>tx('estimate a mixture-of-experts model’s GPU memory, its arithmetic per token and its decode step time at a given batch size, and say why they differ.','估算混合专家（MoE）模型的显存、每个 token 的计算量和给定 batch 下 decode 步的耗时，并说清它们为什么不一样。');
const EXITS=()=>{
  const b8=compute({...BASE,B:8}), dsE=256, dsK=8, ds8=touched(8,dsE,dsK), ds64=touched(64,dsE,dsK), top1=P_TOTAL-MX.L*(MX.E-1)*P_EXPERT;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx(`Each token of ${MODEL} computes with 12.9B parameters. Why must all 46.7B still sit in GPU memory?`,`${MODEL} 的每个 token 只用 12.9B 个参数计算。为什么全部 46.7B 个仍要放在显存里？`),
   o:[tx('The router picks 2 experts for each token in each layer, and the next token may pick any other 2, so every expert must be ready; only the arithmetic is limited to the 2 picked.','路由器在每一层为每个 token 选 2 个专家，下一个 token 可能选另外任意 2 个，所以每个专家都得随时可用；只有计算量限于被选中的 2 个。'),
      tx('Every token runs through all 8 experts, and the router only decides how much of each output to keep, so every expert computes every token.','每个 token 都要过全部 8 个专家，路由器只决定每个输出保留多少，所以每个专家都要为每个 token 计算。'),
      tx('The 8 experts are copies of one MLP block, kept so that 8 tokens can be computed at the same time, one copy per token.','8 个专家是同一个 MLP 块的副本，留着是为了让 8 个 token 同时计算，每个 token 用一份。')],a:0,
   n:[null,tx('Only the 2 picked experts compute a token; the other 6 do nothing for it. That is why a token’s arithmetic is that of 12.9B parameters.','只有被选中的 2 个专家为一个 token 计算，另外 6 个什么都不做。所以一个 token 的计算量只是 12.9B 个参数的。'),
     tx('The experts are different MLP blocks with their own parameters, trained to handle different tokens. Copies would not need a router.','专家是参数各不相同的 MLP 块，训练出来处理不同的 token。如果只是副本，就不需要路由器了。')],
   why:tx('Which experts a token needs is known only when the router scores it, and different tokens pick different pairs. So every expert must be in GPU memory, while each token computes with only its own 2.','一个 token 要哪两个专家，只有路由器打完分才知道，而不同 token 选的组合不同。所以每个专家都得放在显存里，而每个 token 只用自己的 2 个计算。')},
  {q:tx(`At batch 1, a decode step of ${MODEL} reads about as much as a dense 12.9B model; with 64 requests, about as much as a dense 46.7B model. Why?`,`batch 为 1 时，${MODEL} 的一个 decode 步读的量和 12.9B 的稠密模型差不多；64 个请求时，却和 46.7B 的稠密模型差不多。为什么？`),
   o:[tx('A step reads every expert that at least one of its tokens picked. One token picks 2 per layer, but 64 tokens together pick almost all 8, so the step reads almost every parameter.','一步要读至少被它的一个 token 选中的每个专家。一个 token 每层选 2 个，64 个 token 合起来几乎选遍 8 个，所以这一步几乎要读全部参数。'),
      tx('With 64 tokens the router has 64 times the work, and its arithmetic grows until it takes as long as reading every expert.','64 个 token 时路由器的工作量是 64 倍，它的计算一直增长，直到和读全部专家一样久。'),
      tx('The KV caches of 64 requests grow as large as the experts, so reading them takes as long as reading all 8 experts would.','64 个请求的 KV cache 长到和专家一样大，所以读它们和读全部 8 个专家一样久。')],a:0,
   n:[null,tx('The router is tiny: 32,768 parameters per layer, about 65,000 FLOPs per token per layer, under 0.01% of the token’s arithmetic. Its work is a rounding error next to the experts it picks.','路由器很小：每层 32,768 个参数，每个 token 每层约 65,000 FLOP，不到这个 token 计算量的 0.01%。和它选中的专家比，它的工作量可以忽略。'),
     tx('The KV caches do add reads, but far less than the experts: at 2K tokens each, 64 requests keep about 17 GB of KV cache, while the experts hold 45.1 GB. The step grows because more experts are picked.','KV cache 确实会增加读取量，但远比专家少：每个 2K 个 token 时，64 个请求的 KV cache 约 17 GB，专家却有 45.1 GB。这一步变长，是因为被选中的专家变多了。')],
   why:tx(`An expert is read once per step if any token of the step picked it. One token reads 2 of 8 per layer; 8 tokens already about ${T1(b8.d.T)} of 8; 64 tokens all of them. So a busy server reads nearly the whole model every step, as a dense model of that size would.`,
          `只要一步里有一个 token 选了某个专家，它就要读一遍。一个 token 每层读 8 个里的 2 个；8 个 token 已经约读 ${T1(b8.d.T)} 个；64 个 token 就全读了。所以繁忙的服务端每一步几乎都要把整个模型读一遍，和同样大的稠密模型一样。`)},
  {q:tx(`A busy server’s decode step reads almost every expert of ${MODEL}. Why is Mixtral still a better choice there than Llama 2 70B?`,`繁忙的服务端上，${MODEL} 的每个 decode 步几乎要读遍所有专家。为什么这时它仍然比 Llama 2 70B 更划算？`),
   o:[tx('It scores about as well as Llama 2 70B, yet each token multiplies far fewer parameters, and even a step that reads every expert reads fewer weights than Llama 2 70B does; the price is the memory for all 8 experts.','它的分数和 Llama 2 70B 差不多，每个 token 要乘的参数却少得多；就算一步读遍所有专家，读的权重也比 Llama 2 70B 少；代价是 8 个专家都要占显存。'),
      tx('It is not: once a step reads every expert, Mixtral costs as much per step as any dense model, so a busy server should pick the dense model with the same arithmetic per token instead.','它不划算：一步读遍所有专家以后，Mixtral 每一步的成本就和任何稠密模型一样，所以繁忙的服务端应该改用每个 token 计算量相同的稠密模型。'),
      tx('At a large batch the router sends each token to one expert instead of two, so every step reads half the experts and costs about half of what Llama 2 70B costs.','batch 大的时候，路由器只把每个 token 送到 1 个专家而不是 2 个，所以每一步只读一半的专家，成本约是 Llama 2 70B 的一半。')],a:0,
   n:[null,tx('Reading every expert removes only the decode-time saving in weight reads against a dense model of Mixtral’s own size. The dense model with the same arithmetic, Llama 2 13B, scores far lower; the one with the same quality, Llama 2 70B, reads and computes more.','读遍所有专家，只抹掉了它相对同样大小的稠密模型在 decode 时读权重的节省。计算量相同的 Llama 2 13B 分数低得多；质量相当的 Llama 2 70B 读得更多，算得也更多。'),
     tx('The router always sends each token to 2 experts. The batch changes only how many different experts a step reads, and that number grows toward all 8.','路由器始终把每个 token 送到 2 个专家。batch 改变的只是一步里要读多少个不同的专家，而这个数会涨到全部 8 个。')],
   why:tx(`Compare it with a dense model of the same quality: Llama 2 70B multiplies and reads ${big(cmpOf('l70').mult)} parameters for every token, while even a ${MODEL} step that reads every expert reads only ${big(cmpOf('mx').readB)}, and each of its tokens multiplies far fewer. The advantage is quality per unit of arithmetic; the cost is memory.`,
          `和质量相当的稠密模型比：Llama 2 70B 每个 token 都要乘、都要读 ${big(cmpOf('l70').mult)} 个参数；${MODEL} 就算一步读遍所有专家，也只读 ${big(cmpOf('mx').readB)} 个，每个 token 要乘的参数也少得多。优势是每单位计算换来的质量，代价是显存。`)}],
 transfer:[
  {q:tx(`At a batch of 8 requests, ${MODEL}’s decode step reads about ${T1(b8.d.T)} of its 8 experts per layer, ${pct0(b8.d.T/8)}. DeepSeek‑V3 sends each token to 8 of 256 routed experts. At the same batch of 8, about what share of its routed experts does a step read per layer (uniform routing)?`,
        `batch 为 8 个请求时，${MODEL} 的 decode 步每层约读 8 个专家里的 ${T1(b8.d.T)} 个，即 ${pct0(b8.d.T/8)}。DeepSeek‑V3 把每个 token 送到 256 个路由专家里的 8 个。同样 8 个请求，按均匀路由算，它的一步每层大约读多大比例的路由专家？`),
   o:[tx('Under 10%','不到 10%'),tx('10% to 40%','10% 到 40%'),tx('40% to 80%','40% 到 80%'),tx('Over 80%','超过 80%')],a:1,
   why:tx(`${DEF(`About ${pct0(ds8/dsE)}`,`A token skips a given expert with probability 248/256, all 8 tokens with ${A.ui.math('\\left(\\frac{248}{256}\\right)^{8}')} = ${fixd(Math.pow(248/256,8),2)}, so a step reads 256 × (1 − ${fixd(Math.pow(248/256,8),2)}) ≈ ${fint(Math.round(ds8))} routed experts per layer.`)}: 8 tokens make 64 picks among 256 experts, so at most a quarter can be read, and some picks land on the same expert. Many small experts keep a small batch’s reads small. Source: ${cite(SRC.dsv3,'DeepSeek‑V3’s config.json')}, which also has a shared expert that every token uses.`,
          `${DEF(`约 ${pct0(ds8/dsE)}`,`一个 token 躲过某个专家的概率是 248/256，8 个 token 都躲过它的概率是 ${A.ui.math('\\left(\\frac{248}{256}\\right)^{8}')} = ${fixd(Math.pow(248/256,8),2)}，所以一步每层读 256 × (1 − ${fixd(Math.pow(248/256,8),2)}) ≈ ${fint(Math.round(ds8))} 个路由专家。`)}：8 个 token 在 256 个专家里一共选 64 次，所以最多读到四分之一，而且有些会选到同一个专家。专家多而小，小 batch 读的就少。来源：${cite(SRC.dsv3,'DeepSeek‑V3 的 config.json')}，它另有一个每个 token 都用的共享专家。`)},
  {q:tx(`${MODEL} sends each token to 2 experts per layer and uses 12.9B parameters per token; one expert per layer adds up to 5.64B parameters over the 32 layers. If it sent each token to only 1 expert, as Switch Transformers do, about how many parameters would each token use?`,
        `${MODEL} 每层把每个 token 送到 2 个专家，每个 token 用 12.9B 个参数；每层一个专家，32 层加起来是 5.64B 个参数。如果它像 Switch Transformer 那样每个 token 只送到 1 个专家，每个 token 大约用多少参数？`),
   o:[tx('About 6.4B','约 6.4B'),tx('About 7.2B','约 7.2B'),tx('About 12.9B','约 12.9B'),tx('About 23.3B','约 23.3B')],a:1,
   why:tx(`About ${big(top1)}: one expert per layer fewer, 12.9B − 5.64B. Not half of 12.9B, because the shared parts (attention, the embedding table, the LM head, the routers) stay the same. That is ${DEF('Mistral‑7B’s size plus the routers',`${fixd(P_ACTIVE/1e9,2)}B − ${fixd(P_SLOT/1e9,2)}B = ${fixd(top1/1e9,2)}B; Mistral‑7B has ${fixd(M7.total/1e9,2)}B parameters, and the 32 routers add ${fixd(MX.L*P_ROUTER/1e6,2)}M.`)}.`,
          `约 ${big(top1)}：每层少一个专家，12.9B − 5.64B。不是 12.9B 的一半，因为共享的部分（attention、embedding 表、LM head、路由器）不变。这正是${DEF('Mistral‑7B 的大小再加上路由器',`${fixd(P_ACTIVE/1e9,2)}B − ${fixd(P_SLOT/1e9,2)}B = ${fixd(top1/1e9,2)}B；Mistral‑7B 有 ${fixd(M7.total/1e9,2)}B 个参数，32 个路由器再加 ${fixd(MX.L*P_ROUTER/1e6,2)}M。`)}。`)}],
 practice:[
  {id:'llm-fundamentals',d:tx('Question 12: total and active parameters of a mixture of experts, why an auxiliary load-balancing loss is needed, and what the capacity factor controls.','第 12 题：混合专家模型的总参数量和激活参数量，为什么需要辅助的负载均衡 loss，以及容量因子（capacity factor）控制什么。')},
  {id:'matmul-roofline',d:tx('Part 1: shape A, 8 tokens times a 4,096 × 4,096 weight, is the shape of an expert’s matmul when only a few tokens reach it; count its FLOPs and bytes and say why it is memory-bound.','第 1 部分：形状 A（8 个 token 乘一个 4,096 × 4,096 的权重）正是只有几个 token 到达时一个专家的矩阵乘法形状；算出它的 FLOPs 和字节数，说明它为什么受显存带宽限制。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'MoE',title:l?L(l.q):tx('How can a 47B model run like a 13B one?','47B 的模型怎么跑得像 13B 一样快？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: where a model’s parameters live','前置知识：模型的参数在哪'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage moe-stage" id="moe-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="moe-st-title"></div><div class="meter-sub" id="moe-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="moe-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="moe-legend"></div>`+
   `<div class="stage-graph graph moe-graph" id="moe-graph">`+
    `<div class="moe-flow" id="moe-flow"><div class="moe-cap-row"><div class="moe-cap" id="moe-flow-cap"></div><span class="moe-cap-side"><span class="moe-badge" id="moe-flow-badge" tabindex="0"></span><button type="button" class="moe-replay" id="moe-replay" data-act="replay" hidden data-tipx="${esc(tx('Play the routing again: the dots run from the tokens through the router to their experts.','把路由再播一遍：圆点从 token 经过路由器走到它们的专家。'))}">${tx('⟲ Replay','⟲ 重放')}</button></span></div><div class="moe-svgbox" id="moe-flow-svg"></div></div>`+
    `<div class="moe-par" id="moe-par"><div class="moe-cap" id="moe-par-cap"></div><div class="moe-prows" id="moe-prows"></div>`+
     `<div class="moe-gpu" id="moe-gpu"><div class="moe-cap2" id="moe-gpu-cap"></div><div class="bar thin moe-membar" id="moe-membar" tabindex="0"></div><div class="tz-leg moe-memleg" id="moe-memleg"></div></div></div>`+
    `<div class="moe-read" id="moe-read"><div class="moe-cap" id="moe-read-cap"></div><div class="moe-rbar" id="moe-rbar" tabindex="0"></div><div class="moe-rlab mono" id="moe-rlab"></div><div class="moe-svgbox" id="moe-curve"></div></div>`+
    `<div class="moe-ep" id="moe-ep"><div class="moe-cap" id="moe-ep-cap"></div><div class="moe-eprows" id="moe-eprows"></div></div>`+
    `<div class="moe-cmp" id="moe-cmp"><div class="moe-cap" id="moe-cmp-cap"></div><div id="moe-cmp-body"></div></div>`+
   `</div>`+
   `<p class="moe-scope small muted" id="moe-scope" tabindex="0"></p>`+
   `<div class="metrics moe-metrics" id="moe-metrics"></div>`+
   `<details class="math" id="moe-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="moe-mathbody"></div></details></div>`;
}
/* the conditions under the chart, true for what the chart shows in this part (R2.7, R6.10) */
function scopeOf(){
  const id=stepId(), r=R();
  if(id==='adv') return [tx('Scores from the Mixtral paper; arithmetic, bytes and times counted from config.json with this lab’s step model; weights only, in FP8.','分数来自 Mixtral 论文；计算量、字节数和耗时按 config.json 和这个 Lab 的耗时模型算出；只算权重，FP8。'),
    tx(`Scores: the Mixtral paper’s Table 2, every model run by Mistral AI with its own pipeline. Arithmetic: 2 FLOPs per multiplied parameter. Times: the arithmetic at 50% of one ${GPUN}’s dense BF16 peak, the reads at 60% of its peak bandwidth. The KV cache is not in the bars; the note under the reads gives it, and the memory bars show how many requests fit.`,`分数：Mixtral 论文表 2，每个模型都由 Mistral AI 用自己的评测流程跑出。计算量：每个参与乘法的参数 2 FLOP。耗时：计算按一张 ${GPUN} 稠密 BF16 峰值的 50%，读显存按峰值带宽的 60%。条形里不含 KV cache：读权重那一组下面的说明给出了它，显存那一组给出放得下几个请求。`)];
  if(id==='imb') return [tx('Estimate for one MoE layer over 8 GPUs during a 4,096-token prefill: the experts’ arithmetic only, without the all-to-all or attention.','8 张 GPU 上 prefill 4,096 个 token 时一个 MoE 层的估算：只算专家的计算，不含 all-to-all 和 attention。'),
    tx(`Each GPU’s time is the arithmetic of the token copies its expert got, at 50% of the ${GPUN}’s dense BF16 peak; the router’s shares are set by the option in the card. The exchange of tokens between the GPUs (all-to-all) and the attention blocks are left out.`,`每张 GPU 的耗时是它那个专家分到的 token 副本的计算量，按 ${GPUN} 稠密 BF16 峰值的 50% 算；路由器的份额由卡片里的选项设定。GPU 之间交换 token（all-to-all）和 attention 块都不计。`)];
  if(id==='mission'||id==='predict'||id==='router') return [tx(`Parameters counted from ${MODEL}’s config.json; the routing pictures are illustrative.`,`参数按 ${MODEL} 的 config.json 逐项数出；路由的图只是示意。`),
    tx('Every parameter count comes from config.json, component by component; “Show the math” adds them up. Which experts a token picks here is made up for the picture: the rule (softmax, then the top 2) is real, the scores are not.','每个参数量都按 config.json 逐项数出，“展开公式”里把它们加起来。这里 token 选哪两个专家是为画图编的：规则（先 softmax，再取前 2 个）是真的，得分不是。')];
  if(id==='mem'||!r.m.fits) return [tx(`Weights counted from config.json; one GPU’s memory as in the KV cache lab: ${pct0(UTIL)} usable, ${OH} GiB of runtime overhead.`,`权重按 config.json 数出；一张 GPU 的显存按 KV cache 那个 Lab 的算法：可用 ${pct0(UTIL)}，运行时开销 ${OH} GiB。`),
    tx(`The weights take the parameter count times the bytes per parameter. The server uses at most ${pct0(UTIL)} of the GPU’s 80 GiB (vLLM’s gpu_memory_utilization), and ${OH} GiB of that go to the runtime; the rest holds the KV cache.`,`权重 = 参数量 × 每个参数的字节数。服务端最多用 GPU 80 GiB 的 ${pct0(UTIL)}（vLLM 的 gpu_memory_utilization），其中 ${OH} GiB 给运行时；剩下的放 KV cache。`)];
  return [tx('Estimates for one GPU with uniform routing: no queue, decode steps only.','单卡估算，假设路由均匀：不含排队，只算 decode 步。'),
    tx('Every time here comes from this lab’s step model: the arithmetic at 50% of the GPU’s dense BF16 peak, the memory reads at 60% of its peak bandwidth, whichever is longer, with routing assumed uniform. A real server also queues requests and runs prefills between decode steps.','这里的每个时间都来自这个 Lab 的耗时模型：计算按 GPU 稠密 BF16 峰值的 50%，读显存按峰值带宽的 60%，取较长的那个，并假设路由均匀。真实服务还有排队，并会在 decode 步之间插入 prefill。')];
}
const EFF_NOTE=()=>tx(`This lab assumes a step reaches 50% of the ${GPUN}’s dense BF16 peak (989.5 TFLOP/s), that is 494.75 TFLOP/s, as in the other labs. It matters only when arithmetic sets a step’s time, as in a prefill (${PL('imb')}, and the prefill times beside the dense models); every decode step here is set by its memory reads.`,
  `本 Lab 假设一步能用到 ${GPUN} 稠密 BF16 峰值（989.5 TFLOP/s）的 50%，即 494.75 TFLOP/s，和别的 Lab 一样。只有计算决定耗时的时候它才起作用，比如 prefill（${PL('imb')}，以及和稠密模型比较时的 prefill 时间）；这里的每个 decode 步都由读显存决定。`);
const BW_NOTE=()=>tx(`This lab assumes a step reaches 60% of the peak memory bandwidth (3.35 TB/s), that is 2.01 TB/s, as in the other labs. Databricks measured about 60% at batch size 1 on two H100s, and less as the batch grows, so the times at large batches here are optimistic. It sets every decode step’s time here.`,
  `本 Lab 假设一步能用到峰值显存带宽（3.35 TB/s）的 60%，即 2.01 TB/s，和别的 Lab 一样。Databricks 在两张 H100 上、batch 为 1 时实测约 60%，batch 越大越低，所以这里大 batch 的耗时偏乐观。这里每个 decode 步的耗时都由它决定。`);
const RT_NOTE=()=>tx(`This lab assumes every token picks its 2 experts at random, independently of the others. A token then skips a given expert with probability 6/8, and all B tokens skip it with probability ${A.ui.math('\\left(\\frac{6}{8}\\right)^{B}')}, so a layer reads ${A.ui.math('8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{B}\\right)')} experts on average. It is assumed because real routing depends on the text. Real routers favour some experts over others, and for the same number of picks uniform routing reaches the most different experts, so a real batch reads somewhat fewer. (Consecutive tokens of one text often pick the same expert, the Mixtral paper finds; that matters in a prefill, not in a decode batch of different requests.)`,
  `本 Lab 假设每个 token 随机选 2 个专家，和别的 token 无关。于是一个 token 躲过某个专家的概率是 6/8，B 个 token 都躲过它的概率是 ${A.ui.math('\\left(\\frac{6}{8}\\right)^{B}')}，所以一层平均读 ${A.ui.math('8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{B}\\right)')} 个专家。之所以假设，是因为真实的路由取决于文字内容。真实的路由器会偏爱某些专家，而选择次数相同时，均匀路由选到的不同专家最多，所以真实的 batch 读到的专家会少一些。（Mixtral 论文发现同一段文字里相邻的 token 常选同一个专家；这影响的是 prefill，而不是由不同请求组成的 decode batch。）`);
function controlsHTML(){
  return `<aside class="panel controls" id="moe-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="moe-lock" hidden></p>`+
   `<div class="lockable" id="moe-ctl-p-wrap">`+pCtl('moe-ctl-p',false).replace(/<\/div>$/,`<div class="ctl-note">${tx(`How many bytes each parameter takes in GPU memory. ${MODEL} is released in BF16; in FP8 its weights take half the memory and each decode step reads half the bytes.`,`每个参数在显存里占几个字节。${MODEL} 发布时是 BF16；用 FP8 存，权重只占一半显存，每个 decode 步读的字节也减半。`)}</div></div>`)+`</div>`+
   `<div class="lockable">`+cCtl('moe-ctl-c',false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The tokens before each request’s newest one: its prompt and the answer so far. A decode step reads their KV cache, 128 KiB per token.','每个请求最新 token 之前的全部 token：prompt 加上已经写出的回答。decode 步要读它们的 KV cache，每个 token 128 KiB。')}</div></div>`)+`</div>`+
   `<div class="lockable">`+bCtl('moe-ctl-B','B',R().m.max,tx('Requests decoding together (batch)','一起 decode 的请求数（batch）'),false)+`</div>`+
   `<div class="moe-grp lockable" id="moe-ctl-eff"><div class="moe-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('Routing: uniform and independent','路由：均匀且相互独立')}</span><div class="ctl-note">${RT_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Arithmetic: 50% of the peak FLOP/s','计算：峰值 FLOP/s 的 50%')}</span><div class="ctl-note">${EFF_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Memory reads: 60% of the peak bandwidth','读显存：峰值带宽的 60%')}</span><div class="ctl-note">${BW_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${A.t('cUtil')}${tx(': ','：')}${pct0(UTIL)}</span><div class="ctl-note">${A.t('utilNote')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx(`Runtime overhead: ${OH} GiB`,`运行时开销：${OH} GiB`)}</span><div class="ctl-note">${A.t('ohNote')}</div></div></div>`+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

def moe_layer(x, w_router, experts, k=2):
    """x: (tokens, d); w_router: (d, E); experts: E triples
    (w_gate, w_up, w_down) of shapes (d, f), (d, f), (f, d)."""
    logits = x @ w_router                     # a score per expert
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)             # softmax over experts
    top = np.argsort(-p, axis=-1)[:, :k]      # the k best experts
    w = np.take_along_axis(p, top, -1)
    w /= w.sum(-1, keepdims=True)             # the k shares add to 1
    y = np.zeros_like(x)
    for e, (wg, wu, wd) in enumerate(experts):
        rows, slot = np.nonzero(top == e)     # tokens that picked e
        if rows.size == 0:
            continue                          # no token: never read
        g = x[rows] @ wg
        h = g / (1 + np.exp(-g)) * (x[rows] @ wu)   # SiLU(gate) * up
        y[rows] += w[rows, slot, None] * (h @ wd)
    return y, top

rng = np.random.default_rng(0)
d, f, E, T = 64, 224, 8, 6                    # a tiny Mixtral-like layer
x = rng.standard_normal((T, d))
w_router = rng.standard_normal((d, E)) * 0.1
shapes = ((d, f), (d, f), (f, d))
experts = [tuple(rng.standard_normal(s) * 0.05 for s in shapes)
           for _ in range(E)]
y, top = moe_layer(x, w_router, experts)
print(y.shape, "experts read:", len(np.unique(top)), "of", E)
# (6, 64) experts read: 6 of 8`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="moe-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="moe-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('An MoE layer in NumPy: softmax over the router’s scores, the top 2 experts per token, and their outputs added with the two shares. An expert no token picked is skipped, which is why a small batch reads fewer weights.','用 NumPy 写的 MoE 层：对路由器的得分做 softmax，每个 token 取前 2 个专家，再按两个份额把它们的输出相加。没有 token 选的专家直接跳过，所以小 batch 读的权重更少。')}</p><pre class="code" id="moe-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="moe-prod">`+
   tx(`<li>Hugging Face transformers, ${cite(SRC.hf,'<code>models/mixtral/modeling_mixtral.py</code>')}: <code>MixtralSparseMoeBlock</code> calls <code>MixtralTopKRouter</code> (softmax over the 8 scores, the top 2, rescaled to add up to 1) and <code>MixtralExperts</code>, which loops only over the experts at least one token picked. <code>load_balancing_loss_func</code> is the Switch Transformer’s auxiliary loss.</li>`+
      `<li>vLLM, ${cite(SRC.vmx,'<code>vllm/model_executor/models/mixtral.py</code>')}: <code>MixtralMoE</code> shards every expert across the GPUs and runs one fused kernel; it can also keep redundant copies of experts to balance expert-parallel load.</li>`+
      `<li>vLLM, ${cite(SRC.vfm,'<code>vllm/model_executor/layers/fused_moe/</code>')}: ${cite(SRC.valign,'<code>moe_align_block_size.py</code>')} sorts the tokens by expert and pads each expert’s count to the kernel’s block size; ${cite(SRC.vker,'<code>fused_moe.py</code>')} holds the Triton <code>fused_moe_kernel</code> that computes every expert in one launch.</li>`,
      `<li>Hugging Face transformers，${cite(SRC.hf,'<code>models/mixtral/modeling_mixtral.py</code>')}：<code>MixtralSparseMoeBlock</code> 调用 <code>MixtralTopKRouter</code>（对 8 个得分做 softmax，取前 2 个，重新缩放到加起来是 1）和 <code>MixtralExperts</code>，后者只循环至少被一个 token 选中的专家。<code>load_balancing_loss_func</code> 就是 Switch Transformer 的辅助 loss。</li>`+
      `<li>vLLM，${cite(SRC.vmx,'<code>vllm/model_executor/models/mixtral.py</code>')}：<code>MixtralMoE</code> 把每个专家切分到各张 GPU 上，用一个融合 kernel 计算；它还可以给专家放冗余副本，平衡专家并行的负载。</li>`+
      `<li>vLLM，${cite(SRC.vfm,'<code>vllm/model_executor/layers/fused_moe/</code>')}：${cite(SRC.valign,'<code>moe_align_block_size.py</code>')} 把 token 按专家排序，并把每个专家的 token 数补齐到 kernel 的块大小；${cite(SRC.vker,'<code>fused_moe.py</code>')} 里的 Triton <code>fused_moe_kernel</code> 一次启动就算完所有专家。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.shazeer,'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer')}<span>${tx('Shazeer et al., ICLR 2017: a trainable gating network picks a sparse combination of experts for each example','Shazeer 等，ICLR 2017：可训练的门控网络为每个样本选出一个稀疏的专家组合')}</span></li>`+
   `<li>${cite(SRC.switch,'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity')}<span>${tx('Fedus, Zoph and Shazeer, JMLR 2022: route each token to a single expert, an auxiliary load-balancing loss, and the expert capacity','Fedus、Zoph、Shazeer，JMLR 2022：每个 token 只送到一个专家，辅助负载均衡 loss，以及专家容量')}</span></li>`+
   `<li>${cite(SRC.paper,'Mixtral of Experts')}<span>${tx('Jiang et al., 2024: 8 experts per layer, 2 per token, 47B parameters with 13B active; routing analysis','Jiang 等，2024：每层 8 个专家，每个 token 用 2 个，47B 个参数里 13B 个激活；路由分析')}</span></li>`+
   `<li>${cite(SRC.dsmoe,'DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models')}<span>${tx('Dai et al., 2024: many small experts plus shared experts come close to a dense model of the same total size, the upper bound; a conventional top-2 MoE stays further below','Dai 等，2024：许多小专家加上共享专家，接近总参数相同的稠密模型（上限）；常规的 top-2 MoE 离它更远')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="moe-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="moe-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="moe-app m2" id="moe-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="moe-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many experts each decode step reads.','选一个场景，看看每个 decode 步要读几个专家。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="moe-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Serve the most users on one ${GPUN}, a token every 30 ms each`,`一张 ${GPUN}，每人每 30 ms 一个 token，最多能服务几个用户`)}</h3>`+
   `<p>${tx(`One ${GPUN} serves ${MODEL} in FP8, with 4K (4,096) tokens of context per user. Every user must get a token at least every 30 ms, about 33 tokens per second. You may change only the number of users decoding together; serve as many at once as you can:`,
            `一张 ${GPUN} 用 FP8 跑 ${MODEL}，每个用户有 4K（4,096）个 token 的上下文。每个用户至少每 30 ms 要拿到一个 token，约每秒 33 个。你只能改一起 decode 的用户数，尽量同时多服务一些：`)}</p>`+
   bCtl('moe-ic-chB','chB',chMax(),tx('Users decoding together','一起 decode 的用户数'),true)+
   `<div class="goal mono" id="moe-ch-goal"><span>${tx('Time per output token now','现在每个输出 token 的耗时')}</span><span><b id="moe-ch-now"></b>${tx(' / at most 30 ms',' / 最多 30 ms')}</span></div><div class="progress" id="moe-ch-prog" aria-hidden="true"><i></i></div>`+
   submit+`<div id="moe-ch-res" class="moe-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
/* why Next is greyed out: a line under the buttons and on the phone's part bar, and the same text as the button's tooltip (R2.4) */
function nextWhy(){if(mode!=='guided'||!isNextDisabled()) return ''; const s=STEPS[step]; return s.why?s.why():'';}
function railWhy(rail){if(mode!=='guided') return; const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
  if(nav) nav.insertAdjacentHTML('afterend','<p class="moe-why moe-why-off" aria-live="polite"></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="moe-why moe-why-off"></p>');}
/* Scenarios after the learner's own change: what was just set (R1.9) */
function curHTML(){const r=R();
  if(!r.m.fits) return tx(`<p><b>${PN[r.p]}: the weights alone need ${gib1(r.m.w)}, more than the ${gib1(r.m.usable)} one ${GPUN} can use.</b></p>`,`<p><b>${PN[r.p]}：光是权重就要 ${gib1(r.m.w)}，超过一张 ${GPUN} 能用的 ${gib1(r.m.usable)}。</b></p>`);
  return tx(`<p><b>You set ${reqN(r.B)} at ${tokK(r.c)} tokens each: each decode step reads about ${T1(r.d.T)} of the 8 experts per layer and takes about ${msR(r.d.t)}.</b></p>`,`<p><b>你设的是 ${reqN(r.B)}，每个 ${tokK(r.c)} 个 token：每个 decode 步每层约读 8 个专家里的 ${T1(r.d.T)} 个，约 ${msR(r.d.t)}。</b></p>`);}
const railExtra=()=>(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');
function renderRail(){const rail=$('#moe-rail',app); if(!rail) return;
  rail.innerHTML=railHTML()+railExtra(); railWhy(rail); A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#moe-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step]||(s.id==='router'&&rt>0))&&(!s.gate||s.gate());
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===(s.id==='adv'?cmp:skew))); const rr=b.querySelector('.lv-r'); if(rr&&s.id==='adv') rr.textContent=cmpRes(v);});
    const ps=$('.seg-ctl[data-key="p"]',rail); if(ps) A.ui.setPressed(ps,S.p);
    const rb=$('[data-act="route"]',rail); if(rb){rb.disabled=rt>=SENT.length; rb.textContent=rt>=SENT.length?tx('All six tokens routed','六个 token 都送完了'):tx(`Route “${SENT[rt]}”`,`送出“${SENT[rt]}”`);
      const nt=rb.nextElementSibling; if(nt) nt.textContent=tx(`${rt} of 6 tokens routed`,`已送出 6 个 token 里的 ${rt} 个`);}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.moe-why',rail).forEach(w=>{if(why){w.textContent=why; w.classList.remove('moe-why-off');} else w.classList.add('moe-why-off');});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#moe-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?curHTML()+`<p class="small">${tx('This matches none of the five scenarios.','这和五个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the five scenarios above.','从上面五个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const p=chPoint(S.B), now=$('#moe-ch-now',rail); if(!now) return;
    now.textContent=ms2(p.t);
    $('#moe-ch-prog i',rail).style.width=Math.min(100,p.t/CH.goal*100).toFixed(1)+'%';
    [$('#moe-ch-goal',rail),$('#moe-ch-prog',rail)].forEach(e=>{e.classList.toggle('met',p.ok); e.classList.toggle('moe-over',!p.ok);});
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#moe-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
/* the rule for the experts one step reads, in words, where it first appears (R2.7); Show the math typesets it */
function skipWords(B){const q=Math.pow(1-MX.k/MX.E,B), a=q>=0.005?fint(Math.round(q*100))+'%':tx('under 1%','不到 1%'), c=q>=0.005?fint(100-Math.round(q*100))+'%':tx('over 99%','超过 99%'), t=fixd(touched(B,MX.E,MX.k),2);
  return B===1?tx('One token picks 2 of the 8 experts, so a step reads 2.','一个 token 从 8 个专家里选 2 个，所以一步读 2 个。')
    :q<0.005?tx(`Each of the step’s tokens skips a given expert with probability 6/8, so all ${fint(B)} skip it with probability ${A.ui.math(`\\left(\\frac{6}{8}\\right)^{${B}}`)}, under 1%. A layer then reads almost all of its 8 experts on average: ${t}.`,`这一步的每个 token 躲过某个专家的概率是 6/8，所以 ${fint(B)} 个都躲过它的概率是 ${A.ui.math(`\\left(\\frac{6}{8}\\right)^{${B}}`)}，不到 1%。于是一层平均几乎读到全部 8 个专家：${t} 个。`)
    :tx(`Each of the step’s tokens skips a given expert with probability 6/8, so all ${fint(B)} skip it with probability ${A.ui.math(`\\left(\\frac{6}{8}\\right)^{${B}}`)} = ${a}. A layer then reads the other ${c} of its 8 experts on average: ${t}.`,`这一步的每个 token 躲过某个专家的概率是 6/8，所以 ${fint(B)} 个都躲过它的概率是 ${A.ui.math(`\\left(\\frac{6}{8}\\right)^{${B}}`)} = ${a}。于是一层平均读到 8 个专家里另外的 ${c}：${t} 个。`);}
function statusOf(){
  const r=R();
  if(mode==='challenge'){const ok=r.d.t<=CH.goal+1e-12; return {cls:ok?'ok':'bad',txt:ok?tx(`Within 30 ms: ${ms2(r.d.t)}`,`在 30 ms 以内：${ms2(r.d.t)}`):tx(`Over 30 ms: ${ms2(r.d.t)}`,`超过 30 ms：${ms2(r.d.t)}`),short:ok?tx('Within 30 ms','30 ms 以内'):tx('Over 30 ms','超过 30 ms'),
    tip:tx(`Green when each user’s time per output token is at most 30 ms, with ${userN(r.B)} at 4K tokens of context each.`,`${userN(r.B)}、每人 4K 个 token 上下文时，每个用户每个输出 token 的耗时不超过 30 ms，徽章就是绿色。`)};}
  const id=stepId();
  if(isEP()){const e=epRun(skew); return {cls:'neutral',txt:tx(`Layer: ${rs(e.ratio)}× the balanced time`,`这一层：均衡时的 ${rs(e.ratio)} 倍`),short:tx(`${rs(e.ratio)}× balanced`,`均衡的 ${rs(e.ratio)} 倍`),
    tip:tx('How many times as long one MoE layer takes as with balanced routing: it waits for its busiest GPU.','一个 MoE 层的耗时是均衡路由时的几倍：它要等最忙的那张 GPU。')};}
  if(id==='adv'){const m=cmpOf('mx'), b=cmpOf(cmp), k=fixd(b.fl/m.fl,1), bi=BENCH_SHOW[bench], bn=BENCH_COLS[bi], d=fixd(BENCH.mx[bi]-BENCH.l13[bi],1);
    return cmp==='l13'?{cls:'neutral',txt:tx(`Same arithmetic; Mixtral +${d} points on ${bn}`,`计算量相同；Mixtral 的 ${bn} 高 ${d} 分`),short:tx('vs 13B','比 13B'),tip:tx(`Mixtral against Llama 2 13B, the dense model with the same arithmetic per token: ${fixd(BENCH.mx[bi],1)}% against ${fixd(BENCH.l13[bi],1)}% on ${bn}.`,`Mixtral 和每个 token 计算量相同的稠密模型 Llama 2 13B 比：${bn} 是 ${fixd(BENCH.mx[bi],1)}% 对 ${fixd(BENCH.l13[bi],1)}%。`)}
      :{cls:'neutral',txt:tx(`${cmpName(cmp)} computes ${k}× as much`,`${cmpName(cmp)}的计算量是 Mixtral 的 ${k} 倍`),short:tx(`vs ${cmp==='l70'?'70B':'dense 46.7B'}`,`比${cmp==='l70'?'70B':'稠密 46.7B'}`),tip:tx(`${cmpName(cmp)}’s arithmetic per token against Mixtral’s: ${fixd(b.fl/1e9,1)} ÷ ${fixd(m.fl/1e9,1)} GFLOP = ${k}.`,`${cmpName(cmp)}和 Mixtral 每个 token 的计算量之比：${fixd(b.fl/1e9,1)} ÷ ${fixd(m.fl/1e9,1)} GFLOP = ${k}。`)};}
  if(id==='mission'||id==='predict'||id==='router') return {cls:'neutral',txt:tx('12.9B of 46.7B per token','每个 token 用 46.7B 里的 12.9B'),short:tx('2 of 8 experts','8 选 2'),
    tip:tx('Each token computes with its 2 experts per layer and the shared parts: 12.9B of the 46.7B parameters.','每个 token 每层用自己的 2 个专家和共享的部分计算：46.7B 个参数里的 12.9B 个。')};
  if(!r.m.fits) return {cls:'bad',txt:tx(`Does not fit: ${gib1(r.m.w)} of weights`,`放不下：权重 ${gib1(r.m.w)}`),short:tx('Does not fit','放不下'),
    tip:tx(`The weights alone take ${gib1(r.m.w)}; one ${GPUN} can use ${gib1(r.m.usable)} (80 GiB × ${pct0(UTIL)}).`,`光是权重就要 ${gib1(r.m.w)}；一张 ${GPUN} 能用 ${gib1(r.m.usable)}（80 GiB × ${pct0(UTIL)}）。`)};
  if(id==='mem') return {cls:'ok',txt:tx(`Fits: ${gib1(r.m.w)} of weights`,`放得下：权重 ${gib1(r.m.w)}`),short:tx('Fits','放得下'),
    tip:tx(`The weights take ${gib1(r.m.w)} of the ${gib1(r.m.usable)} one ${GPUN} can use, leaving ${gib1(r.m.budget)} for the KV cache.`,`权重占一张 ${GPUN} 可用的 ${gib1(r.m.usable)} 里的 ${gib1(r.m.w)}，剩 ${gib1(r.m.budget)} 给 KV cache。`)};
  return {cls:'neutral',txt:tx(`${T1(r.d.T)} of 8 experts read per layer`,`每层读 8 个专家里的 ${T1(r.d.T)} 个`),short:tx(`${T1(r.d.T)} of 8 experts`,`8 个里读 ${T1(r.d.T)} 个`),
    tip:tx(`How many of the 8 experts in each layer a decode step reads, on average, with ${whoN(r.B)} in the step. ${skipWords(r.B)}`,`一步里有 ${whoN(r.B)}时，一个 decode 步平均读每层 8 个专家里的几个。${skipWords(r.B)}`)};
}
function drawHead(){
  const r=R(), t=$('#moe-st-title',app), sub=$('#moe-st-sub',app), st=$('#moe-status',app), id=stepId();
  if(id==='adv'){t.textContent=tx(`${MODEL} and the dense models you might pick instead`,`${MODEL} 和你可能改用的稠密模型`); sub.textContent=tx(`Per token or per decode step, weights in FP8; ${CMP_B} requests stand for a busy server`,`按每个 token 或每个 decode 步算，权重都用 FP8；${CMP_B} 个请求代表繁忙的服务端`); const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip; return;}
  t.textContent=isEP()?tx(`${MODEL} over 8 ${GPUN}, one expert per GPU`,`8 张 ${GPUN} 跑 ${MODEL}，每张 GPU 一个专家`):(id==='mission'||id==='predict'||id==='router'?tx(`${MODEL}: 32 layers, 8 experts in each`,`${MODEL}：32 层，每层 8 个专家`):tx(`${MODEL} in ${PN[r.p]} on one ${GPUN}`,`一张 ${GPUN} 用 ${PN[r.p]} 跑 ${MODEL}`));
  sub.textContent=isEP()?tx('A prefill of 4,096 tokens, one MoE layer','prefill 4,096 个 token，一个 MoE 层')
    :(id==='router'?tx('One layer, the tokens of one sentence','一层，一句话里的 token'):(id==='mission'||id==='predict'?tx('One token through one layer','一个 token 经过一层')
      :(r.m.fits?tx(`${whoN(r.B)} decoding together, ${tokK(r.c)} tokens of context each`,`${whoN(r.B)}一起 decode，每个 ${tokK(r.c)} 个 token 的上下文`):tx('The weights do not fit in one GPU','一张 GPU 放不下权重'))));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
/* the router picture: tokens (left) → router (middle) → the 8 experts of one layer (right). Experts are weights: a blue fill is an
   expert read in this step, a blue outline one that is not read. Tokens and lines are grayscale (R5.6). */
function flowModel(){
  const id=stepId();
  if(isRouter()){const toks=SENT.map((w,i)=>({label:w,routed:i<rt,cur:i===rt-1})); const loads=Array(MX.E).fill(0), picks=[];
    for(let i=0;i<rt;i++){const r=routeTok(i); picks.push({i,top:r.top,w:r.w}); r.top.forEach(e=>loads[e]++);}
    return {kind:'router',toks,picks,loads,cur:rt>0?routeTok(rt-1):null};}
  if(isEP()){const e=epRun(skew); return {kind:'ep',toks:[{label:tx('4,096 tokens','4,096 个 token'),short:tx('4K tokens','4K 个 token'),routed:true}],loads:e.loads,picks:[]};}
  if(id==='mission'||id==='predict'){const r=routeTok(5); return {kind:'one',toks:[{label:SENT[5],routed:true,cur:true}],picks:[{i:0,top:r.top,w:r.w}],loads:r.top.reduce((a,e)=>{a[e]++; return a;},Array(MX.E).fill(0)),cur:null};}
  const r=R(); if(!r.m.fits) return {kind:'none',toks:[],picks:[],loads:Array(MX.E).fill(0)};
  const sm=sampleRoute(r.B,seed), SHOW=8, toks=[];
  for(let i=0;i<Math.min(r.B,SHOW);i++) toks.push({label:mode==='challenge'?tx(`user ${i+1}`,`用户 ${i+1}`):tx(`req ${i+1}`,`请求 ${i+1}`),routed:true});
  if(r.B>SHOW) toks[SHOW-1]={label:tx(`+${fint(r.B-SHOW+1)} more`,`另 ${fint(r.B-SHOW+1)} 个`),routed:true,more:true};
  return {kind:'batch',toks,picks:sm.picks.slice(0,SHOW-(r.B>SHOW?1:0)).map((p,i)=>({i,top:p})),loads:sm.loads,hit:sm.hit};
}
function drawFlow(){
  const box=$('#moe-flow-svg',app); if(!box) return;
  const fp=$('#moe-flow',app); fp.hidden=stepId()==='adv'; if(fp.hidden) return;
  const fm=flowModel(), cw=box.clientWidth||560, W=Math.max(300,Math.min(680,Math.round(cw))), narrow=W<460, compact=fm.kind==='batch'||fm.kind==='ep'||fm.kind==='none', H=compact?(narrow?186:172):(narrow?236:224);
  const chipW=fm.kind==='ep'?(narrow?78:104):(narrow?64:84), xR=chipW+(narrow?22:48), wR=narrow?58:88, xE=xR+wR+(narrow?22:48), wE=W-xE-4, top=14, rowH=(H-top-8)/MX.E;
  const eY=e=>top+e*rowH+rowH/2, chips=fm.toks, nT=chips.length, tH=compact?16:18, tGap=Math.min(8,(H-top-8-nT*tH)/Math.max(1,nT)), tY=i=>top+(H-top-8-(nT*tH+(nT-1)*tGap))/2+i*(tH+tGap)+tH/2;
  const maxL=Math.max(1,...fm.loads), read=fm.loads.map((v,e)=>v>0&&(fm.kind!=='router'||!fm.cur||fm.cur.top.includes(e))), past=fm.loads.map((v,e)=>fm.kind==='router'&&v>0&&!read[e]);
  let g='';
  /* lines: tokens → router, router → experts (width by load) */
  chips.forEach((c,i)=>{if(!c.routed) return; g+=`<line class="moe-ln${c.cur?' moe-ln-cur':''}" x1="${4+chipW}" y1="${tY(i)}" x2="${xR}" y2="${H/2}"/>`;});
  fm.loads.forEach((v,e)=>{if(!v) return; const cur=fm.cur&&fm.cur.top.includes(e); const wd=fm.kind==='ep'?1+5*v/maxL:(fm.kind==='batch'?1+3*Math.min(1,v/Math.max(4,maxL)):1.4);
    g+=`<line class="moe-ln${cur||fm.kind==='one'?' moe-ln-cur':''}" x1="${xR+wR}" y1="${H/2}" x2="${xE}" y2="${eY(e)}" style="stroke-width:${wd.toFixed(2)}"/>`;});
  /* token chips */
  chips.forEach((c,i)=>{const y=tY(i)-tH/2; g+=`<g class="moe-chip${c.cur?' moe-chip-cur':''}${c.routed?'':' moe-chip-wait'}${c.more?' moe-chip-more':''}"${c.short?` data-short="${esc(c.short)}"`:''}><rect x="4" y="${y}" width="${chipW}" height="${tH}" rx="9"/><text x="${4+chipW/2}" y="${y+tH/2+4}" text-anchor="middle">${esc(c.label)}</text></g>`;});
  /* router: in part 3 its 8 shares for the current token, one row per expert */
  g+=`<g class="moe-router"><rect x="${xR}" y="${top}" width="${wR}" height="${H-top-8}" rx="8"/>`;
  if(fm.kind==='router'&&fm.cur){fm.cur.pr.forEach((p,e)=>{const on=fm.cur.top.includes(e), bw=Math.max(2,(wR-16)*p/Math.max(...fm.cur.pr)); g+=`<rect class="moe-sh${on?' on':''}" x="${xR+8}" y="${eY(e)-4}" width="${bw.toFixed(1)}" height="8" rx="2"/>`;});
    g+=`<text class="moe-rlabel" x="${xR+wR/2}" y="${top-2}" text-anchor="middle">${esc(tx('router','路由器'))}</text>`;}
  else g+=`<text class="moe-rname" x="${xR+wR/2}" y="${top-2}" text-anchor="middle">${esc(tx('router','路由器'))}</text>`;
  g+=`</g>`;
  /* experts */
  for(let e=0;e<MX.E;e++){const y=eY(e)-rowH/2+2, h=rowH-4, on=read[e], v=fm.loads[e];
    const lab=fm.kind==='ep'?tx(`GPU ${e+1}, expert ${e+1}`,`GPU ${e+1}，专家 ${e+1}`):expN(e), shortLab=fm.kind==='ep'?`GPU ${e+1}`:'';
    const cnt=fm.kind==='ep'?fint(Math.round(v)):(fm.kind==='router'&&fm.cur&&fm.cur.top.includes(e)?fint(Math.round(fm.cur.w[fm.cur.top.indexOf(e)]*100))+'%':(v?(fm.kind==='batch'?'×'+fint(v):''):''));
    g+=`<g class="moe-ex${on?' on':''}${past[e]?' past':''}"><rect x="${xE}" y="${y}" width="${wE}" height="${h}" rx="4"/>`;
    g+=`<text class="moe-exname" x="${xE+8}" y="${y+h/2+(compact?3.5:4)}"${shortLab?` data-short="${esc(shortLab)}"`:''}>${esc(lab)}</text>${cnt?`<text class="moe-excnt" x="${xE+wE-7}" y="${y+h/2+(compact?3.5:4)}" text-anchor="end">${esc(cnt)}</text>`:''}</g>`;}
  /* moving dots for the animation (placed by animate(); none when motion is reduced) */
  g+=`<g class="moe-dots"></g>`;
  const aria=fm.kind==='router'?tx(`Router picture: ${rt} of 6 tokens routed.`,`路由示意：6 个 token 已送出 ${rt} 个。`):tx(`Router picture: ${fm.loads.filter(v=>v>0).length} of 8 experts read.`,`路由示意：8 个专家里读了 ${fm.loads.filter(v=>v>0).length} 个。`);
  box.innerHTML=`<svg class="moe-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria)}">${g}</svg>`;
  box._geo={chipW,xR,wR,xE,H,tY,eY,fm};
  fitText(box);
  /* part 7: if one GPU row needs its short label, every row takes it, so the rows read alike */
  if(fm.kind==='ep'){const ns=[...box.querySelectorAll('.moe-exname[data-short]')]; if(ns.some(n=>n.textContent===n.getAttribute('data-short'))){ns.forEach(n=>{n.textContent=n.getAttribute('data-short'); n.style.fontSize='';}); fitText(box);}}
  /* caption and badge */
  const cap=$('#moe-flow-cap',app), bd=$('#moe-flow-badge',app), rp=$('#moe-replay',app);
  if(rp) rp.hidden=A.reduceMotion()||!flowPaths(fm).length;
  cap.textContent=fm.kind==='router'?tx('One layer: each token goes to the 2 experts with the largest shares (bars in the router)','一层：每个 token 去份额最大的 2 个专家（路由器里的横条）')
    :(fm.kind==='ep'?tx('One MoE layer over 8 GPUs: the token copies each expert gets','8 张 GPU 上的一个 MoE 层：每个专家分到的 token 副本数')
      :(fm.kind==='one'?tx('One layer: the router sends the token to 2 of 8 experts','一层：路由器把 token 送到 8 个专家里的 2 个')
        :(fm.kind==='none'?tx('No step runs: the weights do not fit','跑不起来：权重放不下'):tx(`One layer in one decode step: each ${one1()}’s new token picks 2 experts; ${fm.hit} of 8 are read in this draw`,`一个 decode 步里的一层：每个${one1()}的新 token 各选 2 个专家；这次抽到 ${fm.hit} 个专家要读`))));
  const badge=fm.kind==='batch'?tx('one random draw','随机抽一次'):(fm.kind==='ep'?tx('fixed shares','设定的份额'):tx('illustrative scores','示意的得分'));
  bd.textContent=badge; bd.hidden=fm.kind==='none';
  bd.dataset.tipx=fm.kind==='batch'?tx(`Which experts the requests pick is drawn at random here, each pair equally likely. The numbers on the page use the average over all draws, ${fixd(R().d?R().d.T:0,1)} experts per layer; the 32 layers route independently, so over the whole model the count is close to that average.`,`这里每个请求选哪两个专家是随机抽的，每一对的可能性相同。页面上的数用的是所有抽法的平均值，每层 ${fixd(R().d?R().d.T:0,1)} 个专家；32 层各自独立路由，所以整个模型加起来很接近这个平均值。`)
    :(fm.kind==='ep'?tx('The shares are set by the option you pick in the card.','份额由卡片里选的选项设定。')
      :tx(`The router’s scores here are made up for illustration (seeded, the same every time). The rule is real: softmax, then the top 2.`,`这里路由器的得分是为示意编的（固定种子，每次一样）。规则是真的：先 softmax，再取前 2 个。`));
  box.dataset.tipx=fm.kind==='one'?tx('The gray chip is one token; the router sends it to the two filled experts, and the other six do nothing for it.','灰色胶囊是一个 token；路由器把它送到两个填满的专家，另外六个专家不为它做任何计算。'):fm.kind==='router'?tx('Each gray chip is a token of the sentence; the bars in the router are its shares for the 8 experts; the two filled experts are the ones it goes to.','每个灰色胶囊是这句话里的一个 token；路由器里的横条是它给 8 个专家的份额；两个填满的专家就是它要去的。'):(fm.kind==='ep'?tx('Each row is one GPU holding one expert; the number is the token copies it gets.','每一行是存着一个专家的一张 GPU；数字是它分到的 token 副本数。'):(fm.kind==='none'?tx('No decode step runs here: the weights do not fit in one GPU’s memory, so no token reaches the router and no expert is read.','这里跑不了 decode 步：一张 GPU 的显存放不下权重，所以没有 token 去路由器，也没有专家被读。'):tx(`Each gray chip is a ${one1()}’s newest token; a filled expert is read from GPU memory in this step, an outlined one is not; the number on an expert counts the tokens that picked it.`,`每个灰色胶囊是一个${one1()}的最新 token；填满的专家这一步要从显存读，只有边框的不读；专家上的数字是选了它的 token 数。`)));
}
/* part 6: four blocks on their own scales, one bar per model: quality (the paper's Table 2), arithmetic per token, weights read per
   decode step (solid: 1 request; outline: CMP_B requests), weights in GPU memory. Mixtral and the model it is compared with are drawn
   in full, the others faded. Scores and arithmetic in gray, weights in blue (R5.6). */
function drawCmp(){
  const box=$('#moe-cmp',app), on=stepId()==='adv'; box.hidden=!on; if(!on) return;
  const M=cmpModels(), bi=BENCH_SHOW[bench], bn=BENCH_COLS[bi], sh=L(BENCH_SHOT[bi]), pair=id=>id==='mx'||id===cmp, GF=m=>fixd(m.fl/1e9,1);
  $('#moe-cmp-cap',app).textContent=tx(`${MODEL} next to dense models, weights in FP8`,`${MODEL} 和稠密模型并排比较，权重都用 FP8`);
  const row=(id,v,max,txt,sub,tip,cls,v2)=>`<div class="moe-cm-row${pair(id)?' on':''}${id==='mx'?' mx':''}${id==='m7'?' m7':''}" data-m="${id}" tabindex="0" data-tipx="${esc(tip)}"><span class="moe-cm-n">${esc(cmpName(id))}</span>`+
    `<span class="moe-cm-bar">${v2!=null?`<i class="moe-cm-b2 ${cls}" style="width:${(v2/max*100).toFixed(2)}%"></i>`:''}${v!=null?`<i class="${cls}" style="width:${(v/max*100).toFixed(2)}%"></i>`:'<em>'+esc(tx('not measured','没有测过'))+'</em>'}</span><span class="moe-cm-v mono">${esc(txt)}${sub?`<small>${esc(sub)}</small>`:''}</span></div>`;
  const blk=(title,body,extra,note)=>`<div class="moe-cm-blk"><div class="moe-cm-h">${title}</div>${extra||''}${body}${note?`<p class="moe-cm-note">${note}</p>`:''}</div>`;
  const q=M.map(m=>{const s=m.id==='d47'?null:BENCH[m.id][bi]; return row(m.id,s,100,s==null?'':fixd(s,1)+'%','',
    s==null?tx('A dense model with Mixtral’s 46.7B parameters is hypothetical: nobody trained one for the paper, so it has no score.','参数和 Mixtral 一样多的稠密模型是假想的：论文没有训练它，所以没有分数。')
      :tx(`${cmpName(m.id)} scores ${fixd(s,1)}% on ${bn} (${sh}) in the Mixtral paper’s Table 2; Mistral AI ran every model with its own pipeline.`,`Mixtral 论文表 2 里，${cmpName(m.id)}的 ${bn} 分数是 ${fixd(s,1)}%（${sh}）；Mistral AI 用自己的评测流程跑了每个模型。`),'moe-cm-q');}).join('')+
    row('m7',BENCH.m7[bi],100,fixd(BENCH.m7[bi],1)+'%','',tx(`Mistral 7B, a dense model from Mistral AI, the team behind Mixtral: ${bB(M7.total)} parameters, ${fixd(M7.fl/1e9,1)} GFLOP per token, about half of Llama 2 13B’s. It scores ${fixd(BENCH.m7[bi],1)}% on ${bn} (${sh}) in the same Table 2: part of Mixtral’s lead over Llama 2 13B comes from training, not from MoE.`,`Mistral 7B 是 Mixtral 同一个团队 Mistral AI 的稠密模型：${bB(M7.total)} 个参数，每个 token ${fixd(M7.fl/1e9,1)} GFLOP，约是 Llama 2 13B 的一半。同一张表 2 里它的 ${bn} 分数是 ${fixd(BENCH.m7[bi],1)}%（${sh}）：Mixtral 领先 Llama 2 13B 的分数里，有一部分来自训练，而不是 MoE。`),'moe-cm-q');
  const seg=A.ui.seg({id:'moe-bench',key:'bench',label:tx('Benchmark','基准'),options:BENCH_SHOW.map((c,i)=>[String(i),BENCH_COLS[c]]),value:String(bench),inline:false});
  const fmx=Math.max(...M.map(m=>m.fl)), rmx=Math.max(...M.map(m=>m.readB)), mmx=Math.max(...M.map(m=>m.mem));
  const multTip=m=>m.id==='mx'?tx(`${MODEL} multiplies ${bB(m.mult)} parameters per token: the shared parts and 2 experts per layer (the embedding table is only looked up). 2 FLOPs each: ${GF(m)} GFLOP.`,`${MODEL} 每个 token 要乘 ${bB(m.mult)} 个参数：共享部分加上每层 2 个专家（embedding 表只查表）。每个 2 FLOP：${GF(m)} GFLOP。`)
    :m.id==='d47'?tx(`A dense 46.7B would multiply ${bB(m.mult)} parameters per token, all but the embedding table. 2 FLOPs each: ${GF(m)} GFLOP.`,`稠密 46.7B 每个 token 要乘 ${bB(m.mult)} 个参数，除了 embedding 表全都要乘。每个 2 FLOP：${GF(m)} GFLOP。`)
    :tx(`${cmpName(m.id)} multiplies ${bB(m.mult)} of its ${big(m.total)} parameters per token, all but the embedding table, which is only looked up. 2 FLOPs each: ${GF(m)} GFLOP.`,`${cmpName(m.id)}每个 token 要乘它 ${big(m.total)} 个参数里的 ${bB(m.mult)} 个，除了只查表的 embedding 表全都要乘。每个 2 FLOP：${GF(m)} GFLOP。`);
  const f=M.map(m=>row(m.id,m.fl,fmx,GF(m)+' GFLOP',secS(m.pre),multTip(m)+tx(` A prefill of a ${tokK(CMP_N)}-token prompt is set by arithmetic: ${fint(CMP_N)} × ${GF(m)} GFLOP ÷ 494.75 TFLOP/s = ${secS(m.pre)}; attention over the prompt adds a few percent.`,`一个 ${tokK(CMP_N)} token 的 prompt 做 prefill 由计算决定：${fint(CMP_N)} × ${GF(m)} GFLOP ÷ 494.75 TFLOP/s = ${secS(m.pre)}；对 prompt 的 attention 再多百分之几。`),'moe-cm-f')).join('');
  const r=M.map(m=>{const two=m.read1!==m.readB; return row(m.id,m.read1,rmx,two?`${fixd(m.read1/1e9,1)} → ${fixd(m.readB/1e9,1)} GB`:gbS(m.read1),two?`${fixd(m.t1*1e3,1)} → ${ms1(m.tB)}`:ms1(m.t1),
    two?tx(`${MODEL} reads ${gb3(m.read1)} of weights in a decode step for 1 request (2 experts per layer) and ${gb3(m.readB)} for ${CMP_B} requests (almost all 8). At 2.01 TB/s that takes ${ms2(m.t1)} and ${ms2(m.tB)}. Its KV cache comes on top: ${fint(m.kv/1024)} KiB per token.`,`${MODEL} 一个请求时每个 decode 步读 ${gb3(m.read1)} 权重（每层 2 个专家），${CMP_B} 个请求时读 ${gb3(m.readB)}（几乎全部 8 个）。按 2.01 TB/s 要 ${ms2(m.t1)} 和 ${ms2(m.tB)}。KV cache 另算：每个 token ${fint(m.kv/1024)} KiB。`)
      :tx(`${cmpName(m.id)} reads every weight it multiplies in every decode step, at any batch: ${gb3(m.read1)} ÷ 2.01 TB/s = ${ms2(m.t1)}. Its KV cache comes on top: ${fint(m.kv/1024)} KiB per token${m.id==='d47'?' if it kept Mixtral’s attention':''}.`,`${cmpName(m.id)}每个 decode 步都要把它乘的全部权重读一遍，不管 batch 多大：${gb3(m.read1)} ÷ 2.01 TB/s = ${ms2(m.t1)}。KV cache 另算：每个 token ${fint(m.kv/1024)} KiB${m.id==='d47'?'（假设它沿用 Mixtral 的 attention）':''}。`),'moe-cm-w',two?m.readB:null);}).join('');
  const kvG=id=>fint(Math.round(cmpOf(id).kvB/1e9));
  const kvNote=tx(`Weights only. The KV cache comes on top and depends on the attention design, not on MoE: at ${CMP_B} requests of ${tokK(CMP_C)} tokens it adds about ${kvG('l13')} GB per step for Llama 2 13B (no GQA), ${kvG('l70')} GB for Llama 2 70B and ${kvG('mx')} GB for Mixtral.`,`只算权重。KV cache 还要另外读，它取决于注意力的设计，和 MoE 无关：${CMP_B} 个请求、每个 ${tokK(CMP_C)} 个 token 时，每一步还要多读约 ${kvG('l13')} GB（Llama 2 13B，没有 GQA）、${kvG('l70')} GB（Llama 2 70B）和 ${kvG('mx')} GB（Mixtral）。`);
  const g=M.map(m=>row(m.id,m.mem,mmx,gib1(m.mem),tx(`${fint(m.fit)} fit`,`放得下 ${fint(m.fit)} 个`),tx(`${cmpName(m.id)}: ${big(m.total)} parameters × 1 byte = ${gib1(m.mem)} of GPU memory for the weights. Next to them, one ${GPUN} (${gib1(H100.mem*A.GiB*UTIL)} usable, ${OH} GiB of overhead) holds the KV caches of ${fint(m.fit)} requests of ${tokK(CMP_C)} tokens, at ${fint(m.kv/1024)} KiB per token${m.id==='d47'?' (if it kept Mixtral’s attention)':''}.`,`${cmpName(m.id)}：${big(m.total)} 个参数 × 1 字节 = ${gib1(m.mem)} 显存。除去权重，一张 ${GPUN}（可用 ${gib1(H100.mem*A.GiB*UTIL)}，开销 ${OH} GiB）放得下 ${fint(m.fit)} 个 ${tokK(CMP_C)} token 请求的 KV cache，每个 token ${fint(m.kv/1024)} KiB${m.id==='d47'?'（假设它沿用 Mixtral 的 attention）':''}。`),'moe-cm-w')).join('');
  /* the note on the paper's Table 2 that stood under the quality bars is with the sources at the end of the page (FIDPOP; polish round 4,
     checklist A3: sources go to the page's end), so the four blocks fit in one screen */
  $('#moe-cmp-body',app).innerHTML=blk(tx(`Quality: ${bn} score`,`质量：${bn} 分数`),q,seg)+
    blk(tx(`Arithmetic per token, and the prefill of a ${tokK(CMP_N)}-token prompt`,`每个 token 的计算量，以及 prefill 一个 ${tokK(CMP_N)} token 的 prompt 要多久`),f)+
    blk(tx(`Weights read per decode step, and the time to read them: 1 request, and ${CMP_B} requests`,`每个 decode 步读的权重和读它们的时间：1 个请求，以及 ${CMP_B} 个请求`),r,'',kvNote)+
    blk(tx(`Weights in GPU memory, and the requests of ${tokK(CMP_C)} tokens one ${GPUN} holds next to them`,`权重占的显存，以及一张 ${GPUN} 还放得下几个 ${tokK(CMP_C)} token 的请求`),g);
}
/* no label may overflow its capsule or box (R7.8): a shorter label if it has one, then a smaller font, never below 8.5 px */
function fitText(box){
  const fit=(tx_,room)=>{if(!tx_||!tx_.getComputedTextLength||room<=0) return; let w=tx_.getComputedTextLength(); if(!w) return;
    const sh=tx_.getAttribute('data-short')||(tx_.parentNode&&tx_.parentNode.getAttribute&&tx_.parentNode.getAttribute('data-short'));
    if(w>room&&sh){tx_.textContent=sh; w=tx_.getComputedTextLength();}
    let fs=parseFloat(getComputedStyle(tx_).fontSize)||11; while(w>room&&fs>8.5){fs-=0.5; tx_.style.fontSize=fs+'px'; w=tx_.getComputedTextLength();}};
  box.querySelectorAll('.moe-chip').forEach(gc=>{const rc=gc.querySelector('rect'), tt=gc.querySelector('text'); if(rc&&tt) fit(tt,(+rc.getAttribute('width'))-10);});
  box.querySelectorAll('.moe-ex').forEach(gx=>{const rc=gx.querySelector('rect'), nm=gx.querySelector('.moe-exname'), ct=gx.querySelector('.moe-excnt'); if(!rc||!nm) return;
    const cw2=ct&&ct.getComputedTextLength?ct.getComputedTextLength():0; fit(nm,(+rc.getAttribute('width'))-16-(cw2?cw2+8:0));});
}
/* the dots that travel token → router → expert, once after a change; reduced motion: none */
/* the routes the dots run along: in part 3 only the token just routed; one key per picture, so the same picture is not played twice */
function flowPaths(fm){const out=[]; fm.picks.forEach((p,j)=>{if(fm.kind==='router'&&p.i!==rt-1) return; p.top.forEach(e=>out.push({ti:fm.kind==='router'?p.i:(fm.kind==='one'?0:j),e,j}));}); return out;}
const flowSig=fm=>fm.kind+'|'+(fm.kind==='router'?rt:'')+'|'+fm.toks.length+'|'+flowPaths(fm).map(q=>q.ti+'>'+q.e).join(',');
/* a picture drawn without its dots (on opening the lab, a language switch) is on screen all the same: the same picture later is not played */
function noteShown(){const box=$('#moe-flow-svg',app), fp=$('#moe-flow',app); if(box&&box._geo&&fp&&!fp.hidden){const sg=flowSig(box._geo.fm); if(animT&&sg!==runSig){cancelAnimationFrame(animT); animT=0;} animShown=sg; watched.add(sg);}}
function animate(force){
  const box=$('#moe-flow-svg',app); if(!box||!box._geo) return;
  const G=box._geo, fm=G.fm, dots=$('.moe-dots',box), sig=flowSig(fm); if(!force&&sig===animShown) return;
  cancelAnimationFrame(animT); animT=0; animShown=sig;
  if(!force&&watched.has(sig)) return;   // R3.20: a picture already seen to the end is drawn finished at once; ⟲ Replay (force) runs it again
  if(A.reduceMotion()||!dots||!fm.picks.length){watched.add(sig); return;}
  const paths=flowPaths(fm).map(q=>({x0:4+G.chipW,y0:G.tY(q.ti),x1:G.xR+G.wR/2,y1:G.H/2,x2:G.xE+6,y2:G.eY(q.e),d:q.j*70}));
  if(!paths.length){watched.add(sig); return;} animN++; runSig=sig;
  dots.innerHTML=paths.map(()=>'<circle r="4"/>').join(''); const cs=[...dots.children], t0=performance.now(), T=900;
  const tick=now=>{let alive=false; paths.forEach((p,i)=>{const u=Math.max(0,Math.min(1,(now-t0-p.d)/T)); if(u<1) alive=true;
      const a=Math.min(1,u*2), b=Math.max(0,u*2-1), x=u<0.5?p.x0+(p.x1-p.x0)*a:p.x1+(p.x2-p.x1)*b, y=u<0.5?p.y0+(p.y1-p.y0)*a:p.y1+(p.y2-p.y1)*b;
      cs[i].setAttribute('cx',x.toFixed(1)); cs[i].setAttribute('cy',y.toFixed(1)); cs[i].style.opacity=u>=1?0:1;});
    if(alive&&app) animT=requestAnimationFrame(tick); else {animT=0; if(dots.isConnected) dots.innerHTML=''; if(app) watched.add(sig);}};   // ran to the end: watched (a run cut part-way never gets here)
  animT=requestAnimationFrame(tick);
}
/* the token whose experts the parameter bars fill: the one the router picture shows (Paris in parts 1 and 2, the token just routed in
   part 3, request 1's token in part 4) */
function parTok(){const id=stepId(), each=tx('the bar assumes the same pick in every layer (each layer really picks on its own).','这里假设每一层都选它（实际上每层各自选）。');
  if(id==='router'&&rt>0){const r=routeTok(rt-1); return {lab:tx(`used by “${r.tok}”`,`“${r.tok}”用到`),top:r.top,why:tx(`“${r.tok}” picked it; `,`“${r.tok}”选了它；`)+each};}
  if(mode==='guided'&&['mission','predict','router'].includes(id)) return {lab:tx('used by “Paris”','“Paris”用到'),top:routeTok(5).top,why:tx('“Paris” picked it; ','“Paris”选了它；')+each};
  const fits=R().m.fits; return {lab:fits?tx('used by req 1','请求 1 用到'):tx('used by one token','一个 token 用到'),top:sampleRoute(1,seed).picks[0],why:(fits?tx('Req 1’s token picked it; ','请求 1 的 token 选了它；'):tx('One token would pick it; ','一个 token 会选它；'))+each};}
/* parameters: held in memory vs used by one token, one scale; from part 4 on (and in Scenarios), the GPU's memory in GiB */
function drawPar(){
  const box=$('#moe-par',app), id=stepId(), show=(mode==='guided'&&['mission','predict','router','mem'].includes(id))||(mode==='sandbox'&&!R().m.fits);
  box.hidden=!show; if(!show) return;
  $('#moe-par-cap',app).textContent=tx('Parameters: all of them sit in GPU memory; one token computes with only some','参数：全部都在显存里；一个 token 只用其中一部分计算');
  const X=v=>(v/P_TOTAL*100).toFixed(3)+'%';
  const seg=(cls,a,b,tip)=>`<i class="${cls}" style="left:${X(a)};width:${X(b-a)}" data-tipx="${esc(tip)}" tabindex="0"></i>`;
  const expTip=e=>tx(`Expert ${e+1} of every layer: 32 × 176,160,768 = ${big(P_SLOT)} parameters.`,`每一层的专家 ${e+1}：32 × 176,160,768 = ${big(P_SLOT)} 个参数。`);
  const shTip=tx(`The shared parts every token uses: the attention blocks, the RMSNorms, the routers, the embedding table and the LM head, ${big(P_SHARED)} parameters.`,`每个 token 都用的共享部分：attention 块、RMSNorm、路由器、embedding 表和 LM head，一共 ${big(P_SHARED)} 个参数。`);
  const pt=parTok(), pick=pt.top;
  let a='', b='';
  a+=seg('moe-pw',0,P_SHARED,shTip); b+=seg('moe-pw',0,P_SHARED,shTip);
  for(let e=0;e<MX.E;e++){const x0=P_SHARED+e*P_SLOT; a+=seg('moe-pw moe-pe',x0,x0+P_SLOT,expTip(e)); b+=pick.includes(e)?seg('moe-pw moe-pe',x0,x0+P_SLOT,expTip(e)+tx(' ','')+pt.why):seg('moe-po',x0,x0+P_SLOT,tx(`Expert ${e+1}: not used by this token.`,`专家 ${e+1}：这个 token 不用它。`));}
  $('#moe-prows',app).innerHTML=`<div class="moe-prow"><span class="moe-pl">${tx('in GPU memory','放在显存里')}</span><div class="moe-pbar">${a}</div><span class="moe-pv mono">${BIG(P_TOTAL)}</span></div>`+
    `<div class="moe-prow"><span class="moe-pl">${esc(pt.lab)}</span><div class="moe-pbar">${b}</div><span class="moe-pv mono">${BIG(P_ACTIVE)}</span></div>`;
  /* the GPU's memory: the KV cache lab's bar, only once the prediction's answer may show (R3.8) */
  const gp=$('#moe-gpu',app), on=!memHidden()&&(id==='mem'||mode==='sandbox'); gp.hidden=!on; if(!on) return;
  const m=memOf(S.p,S.c), bar=$('#moe-membar',app);
  $('#moe-gpu-cap',app).innerHTML=tx(`One ${GPUN}: 80 GB, read as 80 ${DEF('GiB',TIP.gib())}; the weights in ${PN[S.p]} take ${gib1(m.w)}`,`一张 ${GPUN}：80 GB，按 80 ${DEF('GiB',TIP.gib())} 计；${PN[S.p]} 权重占 ${gib1(m.w)}`);
  if(A.kv&&A.kv.drawBar) A.kv.drawBar(bar,{cap:m.cap,usable:m.usable,w:m.w,oh:m.oh,kv:0,need:m.w+m.oh},{noText:true,capw:100});
  bar.dataset.tipx=tx(`The weights take ${gib1(m.w)} (${fint(P_TOTAL)} × ${BPP[S.p]} ${BPP[S.p]===1?'byte':'bytes'} = ${gbS(m.w)}), the runtime overhead ${OH} GiB; the server uses at most ${pct0(UTIL)} of 80 GiB, ${gib1(m.usable)}.${m.fits?` ${gib1(m.budget)} are left for the KV cache.`:' The weights do not fit.'}`,
    `权重占 ${gib1(m.w)}（${fint(P_TOTAL)} × ${BPP[S.p]} 字节 = ${gbS(m.w)}），运行时开销 ${OH} GiB；服务端最多用 80 GiB 的 ${pct0(UTIL)}，即 ${gib1(m.usable)}。${m.fits?`剩下 ${gib1(m.budget)} 给 KV cache。`:'权重放不下。'}`);
  bar.setAttribute('role','img'); bar.setAttribute('aria-label',bar.dataset.tipx);
  const lg=$('#moe-memleg',app), leg=(cls,t,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i>${esc(t)}</span>`;
  const h=leg('w',tx('weights','权重'),tx(`All ${big(P_TOTAL)} parameters in ${PN[S.p]}: ${gib1(m.w)}.`,`全部 ${big(P_TOTAL)} 个参数（${PN[S.p]}）：${gib1(m.w)}。`))+leg('oh',tx('runtime overhead','运行时开销'),A.t('tipLgOH'))+
    leg('free',tx('free','空闲'),tx('Usable memory nothing takes yet: room for the KV cache.','可用显存里还没被占用的部分：留给 KV cache 的空间。'))+(m.fits?'':leg('over',tx('over the usable memory','超出可用显存'),tx('The part of the weights that does not fit in the usable memory.','权重里超出可用显存的那一部分。')))+
    leg('lim',tx(`dashed line: ${pct0(UTIL)} usable`,`虚线：可用 ${pct0(UTIL)}`),A.t('utilNote'))+
    leg('res',tx('past it: kept free','虚线右边：留空'),tx(`The other ${pct0(1-UTIL)} of the GPU’s memory, past the dashed line, is kept free as a margin: the server does not use it.`,`虚线右边是 GPU 显存里另外的 ${pct0(1-UTIL)}，留作余量，服务端不用它。`));
  if(lg._h!==h){lg.innerHTML=h; lg._h=h;}
}
/* what one decode step reads, and its time against the batch */
function drawRead(){
  const box=$('#moe-read',app), id=stepId(), r=R(), show=r.m.fits&&(mode==='challenge'||mode==='sandbox'||(mode==='guided'&&(id==='batch'||id==='remember')));
  box.hidden=!show; if(!show) return;
  const d=r.d, hi=r.hi, scale=decode(r.m.max,r.c,r.p,MX.E).by;   // the scale: every expert read and the most requests that fit
  $('#moe-read-cap',app).textContent=tx(`What one decode step reads from GPU memory, ${whoN(r.B)} at ${tokK(r.c)} tokens each`,`一个 decode 步从显存读的东西：${whoN(r.B)}，每个 ${tokK(r.c)} 个 token`);
  const X=v=>(v/scale*100).toFixed(3)+'%';
  const seg=(cls,a,w,tip)=>`<i class="${cls}" style="left:${X(a)};width:${X(w)}" data-tipx="${esc(tip)}" tabindex="0"></i>`;
  $('#moe-rbar',app).innerHTML=seg('moe-rw',0,d.byS,tx(`Shared weights every step reads: attention, norms, routers and the LM head, ${gbS(d.byS)}.`,`每一步都要读的共享权重：attention、norm、路由器和 LM head，${gbS(d.byS)}。`))+
    seg('moe-rw moe-re',d.byS,d.byE,tx(`Experts read: ${fixd(d.T,2)} of 8 per layer on average, 32 × ${fixd(d.T,2)} × 176,160,768 × ${BPP[r.p]} ${BPP[r.p]===1?'byte':'bytes'} = ${gbS(d.byE)}.`,`读到的专家：每层平均 8 个里的 ${fixd(d.T,2)} 个，32 × ${fixd(d.T,2)} × 176,160,768 × ${BPP[r.p]} 字节 = ${gbS(d.byE)}。`))+
    seg('moe-ro',d.byS+d.byE,hi.byE-d.byE,hi.byE-d.byE<5e6?tx('Experts not read in this step: almost none, since the batch picks nearly every expert.','这一步没读的专家：几乎没有，因为这个 batch 几乎选遍了所有专家。'):tx(`Experts not read in this step: ${gbS(hi.byE-d.byE)} that stay in GPU memory.`,`这一步没读的专家：${gbS(hi.byE-d.byE)}，留在显存里。`))+
    seg('moe-rk',d.byS+hi.byE,d.byK,tx(`KV caches: ${fint(r.B)} × ${fint(r.c)} tokens × 128 KiB = ${gbS(d.byK)}.`,`KV cache：${fint(r.B)} × ${fint(r.c)} 个 token × 128 KiB = ${gbS(d.byK)}。`));
  $('#moe-rbar',app).dataset.tipx=tx(`What one decode step reads from GPU memory, ${gbS(d.by)} in all: the shared weights, the experts its tokens picked, and the KV caches. Blue is weights, amber the KV cache, and the outline the experts no token picked.`,`一个 decode 步从显存读的东西，一共 ${gbS(d.by)}：共享权重、它的 token 选中的专家，以及 KV cache。蓝色是权重，琥珀色是 KV cache，边框是没有 token 选中的专家。`);
  $('#moe-rbar',app).setAttribute('aria-label',tx(`A decode step reads ${gbS(d.by)}: ${gbS(d.byS)} shared weights, ${gbS(d.byE)} of experts, ${gbS(d.byK)} of KV cache.`,`一个 decode 步读 ${gbS(d.by)}：共享权重 ${gbS(d.byS)}，专家 ${gbS(d.byE)}，KV cache ${gbS(d.byK)}。`));
  $('#moe-rlab',app).innerHTML=tx(`${gb3(d.by)} per step at ${T_('effm','2.01 TB/s')}: ${ms2(d.t)} per token`,`每步 ${gb3(d.by)}，按 ${T_('effm','2.01 TB/s')} 读：每个 token ${ms2(d.t)}`);
  drawCurve();
}
/* the step time against the number of requests: the MoE curve between the 2-expert and the 8-expert lines (both straight: KV only) */
function drawCurve(){
  const box=$('#moe-curve',app); if(!box) return;
  /* a wide stage draws the curve shorter (--moe-ch, set in moe.css) so parts 5 and 8 fit in one screen (polish round 4). On a wider curve
     the plot starts at 57 and the time axis's labels end at 45 (ml − 12), where they were: the ring round the point at one request
     (radius 9, at the plot's left edge) then clears them whatever its value (the shorter curve had brought it onto the “10”) */
  const r=R(), cw=box.clientWidth||520, W=Math.max(300,Math.min(680,Math.round(cw))), narrow=W<460, chV=parseFloat(getComputedStyle(box).getPropertyValue('--moe-ch')), H=narrow?170:(chV>0?chV:158), ml=narrow?44:57, lx=narrow?ml-5:ml-12, mr=narrow?10:14, mt=12, mb=30;
  const Bmax=r.m.max, ch=mode==='challenge', can=curveCan(), f70=l70Fit(r.c), show70=!ch&&r.p==='fp8'&&f70>0&&l70Takes(r.c), B70=Math.min(f70,Bmax), t70a=l70Step(1,r.c).t*1e3, t70b=l70Step(Math.max(1,B70),r.c).t*1e3;
  const ymax=Math.max(ch?40:0,show70?t70b*1.14:0,decode(Bmax,r.c,r.p,MX.E).t*1e3*1.08);
  const X=B=>ml+(B-1)/Math.max(1,Bmax-1)*(W-ml-mr), Y=v=>H-mb-(v/ymax)*(H-mt-mb);
  const pts=[], n=Math.min(Bmax,160); for(let i=0;i<n;i++){const B=1+Math.round(i*(Bmax-1)/Math.max(1,n-1)); pts.push([X(B),Y(decode(B,r.c,r.p).t*1e3)]);}
  const line=(T)=>`M${X(1)},${Y(decode(1,r.c,r.p,T).t*1e3)} L${X(Bmax)},${Y(decode(Bmax,r.c,r.p,T).t*1e3)}`;
  const ty=niceStep(ymax,4); let g='';
  const tops=[]; for(let v=0;v<=ymax;v+=ty) tops.push(v);
  tops.forEach((v,i)=>{g+=`<line class="moe-cg" x1="${ml}" y1="${Y(v)}" x2="${W-mr}" y2="${Y(v)}"/><text class="moe-ct" x="${lx}" y="${Y(v)+4}" text-anchor="end">${fint(v)}${i===tops.length-1?' ms':''}</text>`;});
  const tx0=niceStep(Bmax,narrow?4:6); for(let B=0;B<=Bmax;B+=tx0){if(B<1) continue; g+=`<text class="moe-ct" x="${X(B)}" y="${H-mb+14}" text-anchor="middle">${fint(B)}</text>`;}
  g+=`<text class="moe-ct" x="${X(1)}" y="${H-mb+14}" text-anchor="middle">1</text>`;
  g+=`<text class="moe-ct moe-cax" x="${(ml+W-mr)/2}" y="${H-3}" text-anchor="middle">${esc(ch?tx('users decoding together','一起 decode 的用户数'):tx('requests decoding together','一起 decode 的请求数'))}</text>`;
  g+=`<path class="moe-cref moe-cref8" d="${line(MX.E)}"/><path class="moe-cref" d="${line(MX.k)}"/>`;
  if(show70){const x1=X(1).toFixed(1), x2=X(B70), y1=Y(t70a).toFixed(1), y2=Y(t70b);
    g+=`<line class="moe-c70" x1="${x1}" y1="${y1}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/><circle class="moe-c70e" cx="${x2.toFixed(1)}" cy="${y2.toFixed(1)}" r="2.5"/>`+
      `<text class="moe-ct moe-c70l" x="${(x2+6).toFixed(1)}" y="${(y2-6).toFixed(1)}">${esc(tx(`${narrow?'70B':'Llama 2 70B'}: ${fint(f70)} fit${f70===1?'s':''}`,`${narrow?'70B':'Llama 2 70B'}：放得下 ${fint(f70)} 个`))}</text>`;}
  lastCurve={c:r.c,p:r.p,Bmax,l70:show70?{B0:1,t0:t70a/1e3,B1:B70,t1:t70b/1e3,fit:f70}:null};
  if(ch) g+=`<line class="moe-cgoal" x1="${ml}" y1="${Y(30)}" x2="${W-mr}" y2="${Y(30)}"/><text class="moe-ct moe-cgl" x="${W-mr-2}" y="${Y(30)-4}" text-anchor="end">${esc(tx('goal: 30 ms','目标 30 ms'))}</text>`;
  g+=`<path class="moe-cmoe" d="M${pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' L')}"/>`;
  const px=X(r.B), py=Y(r.d.t*1e3);
  if(can) g+=`<circle class="moe-cring" cx="${px}" cy="${py}" r="9"/>`;
  g+=`<circle class="moe-cpt" cx="${px}" cy="${py}" r="5"/>`;
  const l8=Y(decode(Bmax,r.c,r.p,MX.E).t*1e3);
  const lw=narrow?86:98, atRight=px<ml+lw+24, Bend=1+(lw+6)/Math.max(1,W-ml-mr)*(Bmax-1);   // "all 8 experts" goes to the end of its line the point is not near
  const a8=atRight?{x:W-mr-2,y:l8-6,a:'end'}:{x:ml+6,y:Y(decode(Math.min(Bmax,Bend),r.c,r.p,MX.E).t*1e3)-6,a:'start'};
  const a2={x:W-mr-2,y:Y(decode(Math.max(1,Bmax-((narrow?62:70)+4)/Math.max(1,W-ml-mr)*(Bmax-1)),r.c,r.p,MX.k).t*1e3)+13,a:'end'};   // under its line all along the label
  const s8=tx('all 8 experts','全部 8 个专家'), s2=tx('2 experts','2 个专家');
  g+=`<text class="moe-ct" x="${a8.x.toFixed(1)}" y="${a8.y.toFixed(1)}"${a8.a==='end'?' text-anchor="end"':''}>${esc(s8)}</text><text class="moe-ct" x="${a2.x.toFixed(1)}" y="${a2.y.toFixed(1)}" text-anchor="end">${esc(s2)}</text>`;
  /* the point's value label (R7.8): the places tried are above-right, below-right, then further above or below, then the same on the
     left (left first near the right end); widths are estimated from the font: 0.6 em per Latin character, 1 em per Chinese one */
  const fz=10.5, tw=s=>[...String(s)].reduce((a,ch)=>a+(ch.charCodeAt(0)>=0x2e80?fz:fz*0.6),0);
  const bx=(x,y,s,a)=>{const w=tw(s), x0=a==='end'?x-w:x; return {x0,x1:x0+w,y0:y-fz*0.86,y1:y+fz*0.29};};
  const boxHit=(p,q)=>p.x0<q.x1+1&&q.x0<p.x1+1&&p.y0<q.y1+1&&q.y0<p.y1+1;
  const segHit=(b,x1,y1,x2,y2)=>{const lo=Math.max(Math.min(x1,x2),b.x0-1), hi=Math.min(Math.max(x1,x2),b.x1+1); if(lo>hi) return false;
    const yAt=x=>x2===x1?y1:y1+(y2-y1)*(x-x1)/(x2-x1), ya=yAt(lo), yb=yAt(hi); return Math.max(ya,yb)>=b.y0-1&&Math.min(ya,yb)<=b.y1+1;};
  const lineT=T=>[X(1),Y(decode(1,r.c,r.p,T).t*1e3),X(Bmax),Y(decode(Bmax,r.c,r.p,T).t*1e3)];
  const thick=[]; for(let i=1;i<pts.length;i++) thick.push([pts[i-1][0],pts[i-1][1],pts[i][0],pts[i][1]]);
  const segs=[lineT(MX.E),lineT(MX.k)]; if(show70) segs.push([X(1),Y(t70a),X(B70),Y(t70b)]); if(ch) segs.push([ml,Y(30),W-mr,Y(30)]);
  const boxes=[bx(a8.x,a8.y,s8,a8.a),bx(a2.x,a2.y,s2,'end'),{x0:px-10,x1:px+10,y0:py-10,y1:py+10}];
  if(show70) boxes.push(bx(X(B70)+6,Y(t70b)-6,tx(`${narrow?'70B':'Llama 2 70B'}: ${fint(f70)} fit${f70===1?'s':''}`,`${narrow?'70B':'Llama 2 70B'}：放得下 ${fint(f70)} 个`),'start'));
  if(ch) boxes.push(bx(W-mr-2,Y(30)-4,tx('goal: 30 ms','目标 30 ms'),'end'));
  const vs=ms2(r.d.t), DY=[-10,14,18,-16,24,-24,32], side=(a,x)=>DY.map(dy=>[a,x,Math.max(mt+8,Math.min(H-mb-4,py+dy))]), R_=px>W*0.7?side('end',px-12).concat(side('start',px+12)):side('start',px+12).concat(side('end',px-12));
  /* a score per position: the thick line, the ring and the other labels count 3 each, a dotted line 1; the lowest score wins, in order */
  const inside=b=>b.x0>=ml-2&&b.x1<=W-mr+2&&b.y0>=1&&b.y1<=H-mb-1, score=b=>3*boxes.filter(q=>boxHit(b,q)).length+3*(thick.some(s=>segHit(b,...s))?1:0)+segs.filter(s=>segHit(b,...s)).length;
  const cands=R_.filter(c=>inside(bx(c[1],c[2],vs,c[0]))).map(c=>({c,s:score(bx(c[1],c[2],vs,c[0]))})), pick=cands.length?cands.reduce((m,x)=>x.s<m.s?x:m,cands[0]).c:R_[0];
  g+=`<text class="moe-ct moe-clab" x="${pick[1].toFixed(1)}" y="${pick[2].toFixed(1)}" text-anchor="${pick[0]}">${esc(vs)}</text>`;
  g+=`<g class="moe-cmark" hidden><line class="moe-cmk" x1="0" y1="${mt}" x2="0" y2="${H-mb}"/><circle class="moe-cmd" r="4.5" cx="0" cy="0"/></g>`;
  box.innerHTML=`<svg class="moe-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx(`Time per decode step against the number of requests: ${ms2(r.d.t)} at ${fint(r.B)}.`,`decode 步耗时随请求数的变化：${fint(r.B)} 个时 ${ms2(r.d.t)}。`))}">${g}</svg>`;
  box._cg={W,ml,mr,Bmax,X,Y,c:r.c,p:r.p};
  box.classList.toggle('moe-can',can); box.tabIndex=0;
  box.dataset.tipx=curveTip(r.B,can);
}
/* the curve's readout: what a decode step reads and takes at N requests; a click sets the batch there when the batch can change */
const curveCan=()=>mode==='sandbox'||mode==='challenge'||stepId()==='batch';
function curveTip(B,can){const d=decode(B,R().c,R().p), who=whoN(B);
  return tx(`At ${who}: ${T1(d.T)} of 8 experts read per layer, ${gb3(d.by)} per step, ${ms2(d.t)} per token.`,`${who}时：每层读 8 个专家里的 ${T1(d.T)} 个，每步 ${gb3(d.by)}，每个 token ${ms2(d.t)}。`)+(can?(B===S.B?tx(` Now set to ${who}.`,`现在设的就是 ${who}。`):tx(` Click to set ${who}.`,`点击就改成 ${who}。`)):tx(' Fixed in this part; the number of requests can change in Scenarios.','这一部分固定不变；请求数在“应用场景”里可以改。'));}
function curveAt(box,clientX){const g=box._cg, sv=box.querySelector('svg'); if(!g||!sv) return null; const rc=sv.getBoundingClientRect(), x=(clientX-rc.left)*g.W/rc.width;
  return Math.max(1,Math.min(g.Bmax,Math.round(1+(x-g.ml)/Math.max(1,g.W-g.ml-g.mr)*(g.Bmax-1))));}
function curveShow(box,B){const g=box._cg, mk=box.querySelector('.moe-cmark'); if(!g||!mk) return; const d=decode(B,g.c,g.p), x=g.X(B).toFixed(1), y=g.Y(d.t*1e3).toFixed(1);
  mk.removeAttribute('hidden'); mk.querySelector('line').setAttribute('x1',x); mk.querySelector('line').setAttribute('x2',x); const c=mk.querySelector('circle'); c.setAttribute('cx',x); c.setAttribute('cy',y);
  box._hB=B; box.dataset.tipx=curveTip(B,curveCan()); if(A.tip&&A.tip.show) A.tip.show(box,`<div>${esc(box.dataset.tipx)}</div>`);}
function niceStep(span,n){const raw=span/(n||5), p=Math.pow(10,Math.floor(Math.log10(raw))), m=raw/p; return (m<1.5?1:(m<3.5?2:(m<7.5?5:10)))*p;}
/* part 6: each GPU's busy time, and how long it waits for the busiest */
function drawEP(){
  const box=$('#moe-ep',app), on=isEP(); box.hidden=!on; if(!on) return;
  const e=epRun(skew), b=epRun('bal'), mx=Math.max(e.t,b.t)*1.02, P=v=>(v/mx*100).toFixed(3)+'%';
  $('#moe-ep-cap',app).textContent=tx(`Each GPU’s time for one MoE layer: the layer ends at ${ms2(e.t)} (balanced: ${ms2(b.t)})`,`每张 GPU 算一个 MoE 层的时间：这一层在 ${ms2(e.t)} 结束（均衡时 ${ms2(b.t)}）`);
  $('#moe-eprows',app).innerHTML=e.gpu.map((gp,i)=>`<div class="moe-eprow" tabindex="0" data-tipx="${esc(tx(`GPU ${i+1} computes ${fint(Math.round(gp.n))} token copies with expert ${i+1}: ${ms2(gp.t)}.${gp.t<e.t-1e-9?` Then it waits ${ms2(e.t-gp.t)} for the busiest GPU.`:' It is the busiest GPU: the layer ends when it does.'}`,`GPU ${i+1} 用专家 ${i+1} 算 ${fint(Math.round(gp.n))} 份 token 副本：${ms2(gp.t)}。${gp.t<e.t-1e-9?`然后等最忙的 GPU ${ms2(e.t-gp.t)}。`:'它是最忙的 GPU：它算完，这一层才结束。'}`))}"><span class="moe-epl mono">GPU ${i+1}</span><div class="moe-eptrack"><i class="moe-epbusy" style="width:${P(gp.t)}"></i><i class="moe-epidle" style="left:${P(gp.t)};width:${P(e.t-gp.t)}"></i><b class="moe-epbal" style="left:${P(b.t)}"></b></div><span class="moe-epv mono">${ms2(gp.t)}</span></div>`).join('');
}
function legendHTML(){
  const it=(cls,txt,tip,jump)=>`<span data-tipx="${esc(tip)}" tabindex="0"${jump?` data-jump="${jump}"`:''}><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  const id=stepId(), r=R();
  const chip=it('moe-sw-chip',tx('gray chip: a token','灰色胶囊：一个 token'),tx('Each gray chip is one token on its way to the router.','每个灰色胶囊是一个正要去路由器的 token。'));
  const early=mode==='guided'&&['mission','predict','router','mem'].includes(id);
  const exR=early?it('w',tx('filled: an expert the token uses','填满：这个 token 用的专家'),tx('Each expert is a block of weights in GPU memory. A filled expert computes the token just routed.','每个专家是显存里的一块权重。填满的专家为刚送出的这个 token 计算。')):it('w',isEP()?tx('filled: an expert with work','填满：有活干的专家'):tx('filled: an expert read','填满：要读的专家'),isEP()?tx('Each expert is a block of weights; filled, it has token copies to compute.','每个专家是一块权重；填满表示它有 token 副本要算。'):tx('Each expert is a block of weights, stored in GPU memory. A filled expert is read from memory in this step.','每个专家是一块权重，放在显存里。填满的专家这一步要从显存读。'));
  const exO=early?it('moe-sw-out',tx('outline: an expert the token does not use','边框：这个 token 不用的专家'),tx('An outlined expert stays in GPU memory but does nothing for this token.','只有边框的专家留在显存里，但不为这个 token 做任何计算。')):it('moe-sw-out',tx('outline: an expert not read','边框：不读的专家'),tx('An outlined expert is in GPU memory, but no token picked it, so this step does not read it.','只有边框的专家在显存里，但没有 token 选它，所以这一步不读它。'));
  if(id==='adv') return it('moe-sw-q',tx('a dense model’s score or arithmetic','稠密模型的分数或计算量'),tx('A gray bar is a dense model’s benchmark score or its arithmetic per token.','灰色条是稠密模型的基准分数，或者它每个 token 的计算量。'))+
    it('moe-sw-mx',tx('Mixtral’s score or arithmetic','Mixtral 的分数或计算量'),tx(`${MODEL}’s own score and arithmetic are drawn in the text colour, so they stand out from the dense models’.`,`${MODEL} 自己的分数和计算量用文字的颜色画，和稠密模型的区分开。`))+
    it('w',tx('blue: weights','蓝色：权重'),tx('A blue bar is weights: those read in one decode step, or those held in GPU memory, in FP8.','蓝色条是权重：一个 decode 步读的，或者放在显存里的（FP8）。'))+
    it('moe-sw-out',tx(`outline: read only at ${CMP_B} requests`,`边框：${CMP_B} 个请求时才读的`),tx(`For ${MODEL}, the solid part is what a decode step reads for 1 request, and the outlined part what it reads in addition for ${CMP_B} requests.`,`对 ${MODEL} 来说，实心部分是一个请求时一个 decode 步读的，边框部分是 ${CMP_B} 个请求时多读的。`))+
    it('moe-sw-pair',tx('faded: not in this comparison','半透明：不在这次比较里'),tx('Mixtral and the dense model picked in the card are drawn in full; the others are faded.','Mixtral 和卡片里选中的稠密模型画成实心，其余的半透明。'));
  if(isEP()) return chip+exR+it('moe-sw-busy',tx('solid bar: computing','实心条：在计算'),tx('How long each GPU computes its expert’s token copies.','每张 GPU 算它那个专家分到的 token 副本要多久。'))+it('moe-sw-idle',tx('hatched: waiting','斜纹：在等'),tx('The time a GPU waits for the busiest one before the layer can end.','一张 GPU 等最忙的那张算完、这一层才能结束的时间。'))+it('lim',tx('dashed line: balanced','虚线：均衡时'),tx('Where the layer would end if every expert got the same number of copies.','如果每个专家分到的副本一样多，这一层会在这里结束。'));
  const base=chip+exR+exO;
  const past=isRouter()&&rt>1?it('moe-sw-past',tx('faded: used by an earlier token','半透明：前面的 token 用过的专家'),tx('An expert drawn faded was picked by an earlier token of the sentence, not by the token just routed.','半透明的专家是这句话里前面的 token 选的，不是刚送出的这个 token 选的。')):'';
  const cnt=`<span data-tipx="${esc(tx('The number on an expert counts the tokens of this step that picked it; the expert is read once, however many picked it.','专家上的数字是这一步里选了它的 token 数；不管几个 token 选它，它都只读一遍。'))}" tabindex="0"><b class="mono moe-lgx">×N</b> <span>${esc(tx('picked by N of the step’s tokens','这一步有 N 个 token 选它'))}</span></span>`;
  const po=()=>it('moe-sw-po',tx('dashed box: parameters a token does not use','虚线框：token 不用的参数'),tx('In the parameter bars, a dashed box is an expert that stays in GPU memory but does not compute this token.','参数条里的虚线框，是留在显存里、却不为这个 token 计算的专家。'));
  if(!r.m.fits&&(id==='mem'||mode==='sandbox')) return it('moe-sw-out',tx('outline: an expert in GPU memory','边框：显存里的一个专家'),tx('The weights do not fit, so no step runs: every expert is drawn as an outline, none of them read.','权重放不下，就跑不了任何一步：每个专家都只画边框，一个也不读。'))+
    it('w',tx('blue: parameters','蓝色：参数'),tx('In the parameter bars, blue is parameters: all of them in GPU memory in the upper bar, those one token computes with in the lower one.','参数条里的蓝色是参数：上面一条是显存里的全部参数，下面一条是一个 token 计算时用到的。'))+po();
  if(id==='mission'||id==='predict'||id==='router'||id==='mem') return base+past+(id==='mem'&&r.m.fits?cnt:'')+po();
  const f70=l70Fit(r.c), ctx70=mode!=='challenge'&&r.m.fits&&!l70Takes(r.c)?`<span data-tipx="${esc(tx(`Llama 2 70B takes at most ${fint(L70.ctx)} tokens in all (max_position_embeddings in its config.json), so it cannot serve these requests: their context alone reaches that limit, with no room left for a new token. The curve draws no line for it; the guided part “${L(STEPS[SIX('adv')].title)}” compares the two at 2K tokens per request.`,`Llama 2 70B 最多只能处理 ${fint(L70.ctx)} 个 token（config.json 的 max_position_embeddings），所以服务不了这些请求：光是它们的上下文就到了这个上限，新 token 已经没有位置。曲线上不画它；分步引导的“${L(STEPS[SIX('adv')].title)}”在每个请求 2K 个 token 时比较了两者。`))}" tabindex="0"><span>${esc(tx(`no Llama 2 70B line: it takes at most ${fint(L70.ctx)} tokens`,`没画 Llama 2 70B：它最多只能处理 ${fint(L70.ctx)} 个 token`))}</span></span>`:'';
  const l70=mode!=='challenge'&&r.m.fits&&f70>0&&l70Takes(r.c)?it('moe-sw-70',tx('dash-dot line: Llama 2 70B with its KV cache','点划线：Llama 2 70B（含 KV cache）'),tx(`Llama 2 70B, a dense model of about the same quality, counted like the thick line: every decode step reads its ${gbS(L70.read)} of weights and its KV cache, ${fint(L70.kv/1024)} KiB per token of context. At ${tokK(r.c)} tokens each, one request takes ${ms1(l70Step(1,r.c).t)}, and one ${GPUN} holds only ${fint(f70)} of its requests next to the weights, so the line ends there. The guided part “${L(STEPS[SIX('adv')].title)}” compares the two.`,`Llama 2 70B 是质量相当的稠密模型，这里和粗线算法一样：每个 decode 步要读它的 ${gbS(L70.read)} 权重，再加它的 KV cache（上下文每个 token ${fint(L70.kv/1024)} KiB）。每个请求 ${tokK(r.c)} 个 token 时，1 个请求要 ${ms1(l70Step(1,r.c).t)}，而一张 ${GPUN} 除去权重只放得下它 ${fint(f70)} 个请求，所以线到那里为止。分步引导的“${L(STEPS[SIX('adv')].title)}”比较了两者。`)):ctx70;
  return base+cnt+it('kv','KV cache',tx('The amber part of the read bar: each request’s KV cache, 128 KiB per token of context.','读取条里的琥珀色部分：每个请求的 KV cache，上下文每个 token 128 KiB。'))+l70+
    it('moe-sw-line',tx('thick line: this MoE','粗线：这个 MoE'),tx('The thick curve is the decode step’s time at each number of requests, with the experts the batch picks.','粗曲线是每种请求数下 decode 步的耗时，按 batch 选中的专家算。'))+
    it('moe-sw-ref',tx('dotted lines: 2 or all 8 experts read','点线：读 2 个或全部 8 个专家'),tx('The lower dotted line is the time if a step read only 2 experts per layer, as one token needs; the upper one if it read all 8, as a dense model of 46.7B reads everything.','下面的点线是一步每层只读 2 个专家（一个 token 的需要）时的耗时；上面的是读满 8 个（像 46.7B 的稠密模型那样全读）时的耗时。'))+
    (mode==='challenge'?it('lim',tx('dashed line: 30 ms goal','虚线：目标 30 ms'),tx('At most 30 ms per output token for each user.','每个用户每个输出 token 最多 30 ms。')):'');
}
/* metric cards: ▲▼ only after the learner's own change, from the numbers as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const r=R(), id=stepId();
  if(isEP()){const e=epRun(skew), b=epRun('bal');
    return [{k:'hot',label:tx('Copies on the busiest GPU','最忙的 GPU 分到的副本'),v:fint(Math.round(Math.max(...e.loads))),sub:tx(`of ${fint(EP_COPIES)}; balanced ${fint(EP_COPIES/MX.E)}`,`共 ${fint(EP_COPIES)} 份；均衡时 ${fint(EP_COPIES/MX.E)}`),tip:tx('The token copies the busiest expert’s GPU computes in one layer.','最忙的专家所在的 GPU 在一层里要算的 token 副本数。')},
      {k:'lt',label:tx('One MoE layer','一个 MoE 层'),v:ms2(e.t),sub:tx(`balanced: ${ms2(b.t)}`,`均衡时：${ms2(b.t)}`),tip:tx(`The layer ends when the busiest GPU ends: ${fint(Math.round(Math.max(...e.loads)))} copies × 2 × 176,160,768 FLOPs ÷ 494.75 TFLOP/s = ${ms2(e.t)}.`,`最忙的 GPU 算完，这一层才结束：${fint(Math.round(Math.max(...e.loads)))} 份 × 2 × 176,160,768 FLOP ÷ 494.75 TFLOP/s = ${ms2(e.t)}。`)},
      {k:'idle',pts:true,label:tx('GPU time spent waiting','GPU 花在等待上的时间'),v:pct0(e.idle),sub:tx('of all 8 GPUs’ time in the layer','占 8 张 GPU 在这一层的总时间'),tip:tx('The share of the 8 GPUs’ time in this layer during which they wait for the busiest one.','8 张 GPU 在这一层的时间里，等最忙的那张的时间占多少。')}];}
  if(id==='adv'){const m=cmpOf('mx'), b=cmpOf(cmp), bi=BENCH_SHOW[bench], bn=BENCH_COLS[bi], bs=cmp==='d47'?tx('not measured','没有测过'):fixd(BENCH[cmp][bi],1)+'%';
    const sh=L(BENCH_SHOT[bi]);
    return [{k:'cq'+bi,label:tx(`${bn} score`,`${bn} 分数`),v:fixd(BENCH.mx[bi],1)+'%',sub:`${cmpName(cmp)}${tx(': ','：')}${bs}`,tip:tx(`${MODEL}’s ${bn} score (${sh}) in the Mixtral paper’s Table 2, next to ${cmpName(cmp)}’s.`,`Mixtral 论文表 2 里 ${MODEL} 的 ${bn} 分数（${sh}），旁边是${cmpName(cmp)}的。`)},
      {k:'cf',label:tx('Arithmetic per token, GFLOP','每个 token 的计算量（GFLOP）'),v:fixd(m.fl/1e9,1),sub:`${cmpName(cmp)}${tx(': ','：')}${fixd(b.fl/1e9,1)}`,tip:tx(`Arithmetic per token is 2 FLOPs per multiplied parameter: ${MODEL} multiplies ${bB(m.mult)} per token, ${cmpName(cmp)} ${bB(b.mult)}.`,`每个 token 的计算量是每个参与乘法的参数 2 FLOP：${MODEL} 每个 token 乘 ${bB(m.mult)} 个参数，${cmpName(cmp)}乘 ${bB(b.mult)} 个。`)},
      {k:'cr',label:tx(`Weights read per step, ${CMP_B} requests`,`${CMP_B} 个请求时每步读的权重`),v:gbS(m.readB),sub:`${cmpName(cmp)}${tx(': ','：')}${gbS(b.readB)}`,tip:tx(`A decode step for ${CMP_B} requests reads ${gb3(m.readB)} of ${MODEL}’s weights in FP8, ${ms2(m.tB)} at 2.01 TB/s; ${cmpName(cmp)} reads ${gb3(b.readB)}, ${ms2(b.tB)}. The KV cache comes on top for every model.`,`${CMP_B} 个请求一起 decode 时，一个 decode 步读 ${gb3(m.readB)} 的 ${MODEL} 权重（FP8），按 2.01 TB/s 要 ${ms2(m.tB)}；${cmpName(cmp)}读 ${gb3(b.readB)}，要 ${ms2(b.tB)}。每个模型还要另外读 KV cache。`)},
      {k:'cm',label:tx('Weights in GPU memory','权重占的显存'),v:gib1(m.mem),sub:`${cmpName(cmp)}${tx(': ','：')}${gib1(b.mem)}`,tip:cmp==='d47'?tx(`Every parameter at 1 byte (FP8): ${MODEL} and the dense 46.7B both hold ${big(m.total)}.`,`每个参数 1 字节（FP8）：${MODEL} 和稠密 46.7B 都是 ${big(m.total)} 个。`):tx(`Every parameter at 1 byte (FP8): ${MODEL} ${big(m.total)}, ${cmpName(cmp)} ${big(b.total)}.`,`每个参数 1 字节（FP8）：${MODEL} ${big(m.total)} 个，${cmpName(cmp)} ${big(b.total)} 个。`)}];}
  const setA=id==='mission'||id==='predict'||id==='router'||id==='mem'||!r.m.fits;
  if(setA){const hid=memHidden(), m=memOf(S.p,S.c);
    return [{k:'tot',label:tx('Parameters in all','参数总数'),v:big(P_TOTAL),sub:tx('all 8 experts in every layer','每层全部 8 个专家'),tip:tx(`${MODEL}’s parameters, counted from its config.json. Mistral AI gives the same 46.7B.`,`${MODEL} 的参数是按 config.json 数出来的，和 Mistral AI 给出的 46.7B 一致。`)},
      {k:'act',label:tx('Parameters per token','每个 token 用的参数'),v:big(P_ACTIVE),sub:tx('2 of 8 experts per layer','每层 8 个专家里的 2 个'),tip:TIP.active()},
      {k:'mem',label:tx('Weights in GPU memory','权重占的显存'),v:hid?'?':gib1(m.w),sub:hid?tx('shown in part 4','第 4 部分显示'):tx(`${PN[S.p]}; one ${GPUN} can use ${gib1(m.usable)}`,`${PN[S.p]}；一张 ${GPUN} 可用 ${gib1(m.usable)}`),hid,
       tip:hid?tx('Hidden until part 4, where it answers your prediction.','第 4 部分揭晓你的预测之前先隐藏。'):tx(`${fint(P_TOTAL)} parameters × ${BPP[S.p]} ${BPP[S.p]===1?'byte':'bytes'} = ${gbS(m.w)} = ${gib1(m.w)}.`,`${fint(P_TOTAL)} 个参数 × ${BPP[S.p]} 字节 = ${gbS(m.w)} = ${gib1(m.w)}。`)}];}
  const d=r.d;
  return [{k:'T',label:tx('Experts read per layer','每层读的专家'),v:T1(d.T)+tx(' of 8',' / 8'),sub:tx(`${whoN(r.B)}, uniform routing`,`${whoN(r.B)}，均匀路由`),tip:skipWords(r.B)},
    {k:'by',label:tx('Read per decode step','每个 decode 步读'),v:gbS(d.by),sub:tx('weights and KV caches','权重和 KV cache'),tip:tx(`One decode step reads ${gbS(d.by)}: ${gbS(d.byS)} of shared weights + ${gbS(d.byE)} of experts + ${gbS(d.byK)} of KV cache.`,`一个 decode 步读 ${gbS(d.by)}：共享权重 ${gbS(d.byS)} + 专家 ${gbS(d.byE)} + KV cache ${gbS(d.byK)}。`)},
    {k:'tpot',label:tx('Time per output token (TPOT)','每个输出 token 的耗时（TPOT）'),v:ms2(d.t),sub:tx('one decode step','一个 decode 步'),tip:tx(`Each ${one1()} gets one token per decode step. Memory reads set the step: ${gb3(d.by)} ÷ 2.01 TB/s = ${ms2(d.tm)}; the arithmetic takes ${ms2(d.tc)}.`,`每个${one1()}每个 decode 步拿到一个 token。读显存决定这一步：${gb3(d.by)} ÷ 2.01 TB/s = ${ms2(d.tm)}；计算要 ${ms2(d.tc)}。`)},
    {k:'thr',label:tx(`Tokens per second, all ${one1()}s`,`全部${one1()}每秒的 token`),v:fint(r.thr),sub:tx(`${fixd(r.tps,1)} per ${one1()}`,`每个${one1()} ${fixd(r.tps,1)} 个`),tip:tx(`All ${one1()}s together get ${fint(r.thr)} tokens per second: ${fint(r.B)} ${r.B===1?'token':'tokens'} per step ÷ ${fixd(d.t*1e3,3)} ms per step.`,`全部${one1()}合计每秒 ${fint(r.thr)} 个 token：每步 ${fint(r.B)} 个 token ÷ 每步 ${fixd(d.t*1e3,3)} ms。`)}];
}
function drawMetrics(){
  const host=$('#moe-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip; el.classList.toggle('moe-hid',!!m.hid);
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!m.hid&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&String(prev).replace(/[\d.,\s]/g,'')===String(m.v).replace(/[\d.,\s]/g,'')) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v)),pts:!!m.pts})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.hid?null:m.v;});
}
/* Show the math: every line redoes from the numbers it shows (R6.6), and every term has its own tooltip (R1.4) */
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each block is one lead line in
   words, then rows that line up at “=”: the quantity's name, its formula typeset, the reader's numbers, the result (R6.29). A cell is
   TeX, or {h:html} for the numbers, whose factors keep their terms (R1.4) */
/* a row written as one chain (name = formula = numbers = result) breaks at each “=” into rows of three cells, so every step sits under
   the one before it and a long chain does not run past the panel (H2: chains break at “=”, aligned) */
const MOsplit=rows=>rows.flatMap(r=>r.length>3&&r.every((c,i)=>i%2===0||c==='=')&&r.length%2===1?r.slice(1).reduce((o,c,i)=>{if(i%2===1) o.push([o.length?'':r[0],'=',c]); return o;},[]):[r]);
/* the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): a long formula is cut before each
   “+” or “−” outside braces and brackets into pieces that wrap onto the next line, indented under its first term; the reader's numbers wrap at
   their spaces the same way. A piece that starts with an operator gets a medium space on each side of it (Chromium drops the space
   around an mrow's leading operator; the kit's .kv-fx adds one too) */
const MOcut=tex=>{const out=[]; let d=0, last=0;
  for(let i=0;i<tex.length;i++){const c=tex[i];
    if(c==='{'||c==='(') d++; else if(c==='}'||c===')') d--;
    else if(tex.startsWith('\\left',i)){d++; i+=4;} else if(tex.startsWith('\\right',i)){d--; i+=5;}
    else if(d===0&&i>last&&(tex.startsWith(' + ',i)||tex.startsWith(' - ',i))){out.push(tex.slice(last,i)); last=i+1;}}
  out.push(tex.slice(last)); return out;};
const MOfx=tex=>{const p=MOcut(tex); return p.length<2?null:p.map((x,k)=>`<span class="moe-fp">${A.ui.math(k?x.replace(/^([+-]) /,'\\:$1\\: '):x)}</span>`).join('<wbr>');};
/* the last cell of a short row spans the rest (not a fixed column), so the rows still line up when the name moves above them */
const MOG=rows_=>{const rows=MOsplit(rows_), n=Math.max(...rows.map(r=>r.length));
  return `<div class="moe-mg" style="--n:${n};--n1:${n-1}">`+rows.map(r=>r.map((c,i)=>{const w=i===r.length-1&&r.length<n, fx=i>0&&typeof c==='string'?MOfx(c):null;
    return `<span class="moe-mc${i===0?' moe-mn':''}${w?' moe-mw':''}${c&&c.h!=null?' moe-mh':''}${fx?' moe-fxc':''}"${w?` style="grid-column:span ${n-i}"`:''}>${c==null||c===''?'':(c.h!=null?c.h:fx||A.ui.math(c))}</span>`;}).join('')).join('')+'</div>';};
const MOB=(lead,...grids)=>`<div class="moe-mb"><p class="moe-ml">${lead}</p>${grids.map(MOG).join('')}</div>`;
function mathHTML(){
  const r=R(), t=T_, H=h=>({h}), X=x=>'\\text{'+x+'}', B_=x=>`<b>${x}</b>`;
  let h=MOB(tx('One layer’s parameters: attention, one expert, the router and two RMSNorms.','一层的参数：attention、一个专家、路由器和两个 RMSNorm。'),
    [[X('attention'),'=',H(t('attn','2 × 4,096 × 4,096 + 2 × 4,096 × 1,024')),'=',H('41,943,040')],[X(tx('one expert','一个专家')),'=',H(t('exp','3 × 4,096 × 14,336')),'=',H('176,160,768')],
     [X(tx('router','路由器')),'=',H(t('rtr','4,096 × 8')),'=',H('32,768')],[X(tx('two RMSNorms','两个 RMSNorm')),'=',H('2 × 4,096'),'=',H('8,192')]]);
  h+=MOB(tx('The model has 32 such layers with 8 experts each, plus the embedding and the output head. Each token uses only 2 of the 8 experts in a layer:','模型有 32 个这样的层，每层 8 个专家，再加上 embedding 和输出头。每个 token 在每层只用 8 个专家中的 2 个：'),
    [[X(tx('parameters in all','参数总数')),'=',X(tx('embedding and head','embedding 和输出头'))+' + 4,096 + 32 \\times ('+X(tx('layer','一层'))+' + E \\times '+X(tx('expert','专家'))+')'],
     ['','=',H(`${t('emb','2 × 32,000 × 4,096')} + 4,096 + 32 × (41,943,040 + 8,192 + 32,768 + ${t('E','8')} × 176,160,768)`)],['','=',H(`${B_(fint(P_TOTAL))} (${big(P_TOTAL)})`)]],
    [[X(tx('parameters per token','每个 token 用的参数')),'=',X(tx('parameters in all','参数总数'))+' - 32 \\times (E - k) \\times '+X(tx('expert','专家'))],
     ['','=',H(`${fint(P_TOTAL)} − 32 × (${t('E','8')} − ${t('k','2')}) × 176,160,768`)],['','=',H(`${B_(fint(P_ACTIVE))} (${big(P_ACTIVE)})`)]]);
  if(!memHidden()){const m=memOf(S.p,S.c);
    h+=MOB(tx('Every expert must sit in GPU memory, so the weights count all parameters. What is left of the usable memory holds the KV cache:','每个专家都要放在显存里，所以权重按全部参数算。可用显存减去权重和开销，剩下的放 KV cache：'),
      [[X(tx('weights','权重')),'=',X(tx('parameters in all','参数总数'))+' \\times '+X(tx('bytes each','每个的字节数')),'=',H(`${fint(P_TOTAL)} × ${t('p',BPP[S.p]+NS+tx(BPP[S.p]===1?'byte':'bytes','字节'))}`),'=',H(`${gbS(m.w)} = ${B_(gib1(m.w))}`)],
       [X(tx('usable','可用')),'=',X(tx('GPU memory','显存'))+' \\times '+X(tx('utilization','利用率')),'=',H(`80 GiB × ${pct0(UTIL)}`),'=',H(gib1(m.usable))]].concat(m.fits?
      [[X(tx('left for the KV cache','留给 KV cache')),'=',X(tx('usable','可用'))+' - '+X(tx('weights','权重'))+' - '+X(tx('overhead','开销')),'=',H(`${gib1(m.usable)} − ${gib1(m.w)} − ${OH} GiB`),'=',H(gib1(m.budget))]]:[]));}
  if(stepId()==='adv'){const M=cmpModels(), g=id=>M.find(m=>m.id===id), a=g('l13'), c=g('l70'), m=g('mx'), dd=g('d47');
    h+=MOB(tx('The dense models’ parameters, from their config files:','两个稠密模型的参数，按各自的 config 算：'),
      [[X('Llama 2 13B'),'=',H(t('l13c','2 × 32,000 × 5,120 + 40 × (4 × 5,120 × 5,120 + 3 × 5,120 × 13,824 + 2 × 5,120) + 5,120')),'=',H(`${B_(fint(a.total))} (${big(a.total)})`)],
       [X('Llama 2 70B'),'=',H(t('l70c','2 × 32,000 × 8,192 + 80 × (2 × 8,192 × 8,192 + 2 × 8,192 × 1,024 + 3 × 8,192 × 28,672 + 2 × 8,192) + 8,192')),'=',H(`${B_(fint(c.total))} (${big(c.total)})`)]]);
    h+=MOB(tx('Each multiplied parameter costs a multiply and an add per token:','每个参与乘法的参数，每个 token 要做一次乘法和一次加法：'),
      [[X(tx('arithmetic per token','每个 token 的计算量')),'=','2 \\times '+X(tx('multiplied parameters','参与乘法的参数'))]],
      [[X('Llama 2 13B'),'=',H(`2 × ${fint(a.mult)}`),'=',H(`${fixd(a.fl/1e9,1)} GFLOP`)],[X(MODEL.replace(/[‑]/g,'-')),'=',H(`2 × ${fint(m.mult)}`),'=',H(`${fixd(m.fl/1e9,1)} GFLOP`)],
       [X('Llama 2 70B'),'=',H(`2 × ${fint(c.mult)}`),'=',H(`${fixd(c.fl/1e9,1)} GFLOP`)],[X(tx('dense 46.7B','稠密 46.7B')),'=',H(`2 × ${fint(dd.mult)}`),'=',H(`${fixd(dd.fl/1e9,1)} GFLOP`)]]);
    h+=MOB(tx(`${MODEL} reads the shared weights and only the experts it uses (FP8, 1 byte each):`,`${MODEL} 每个 decode 步读共享的权重，再加上用到的专家（FP8，每个 1 字节）：`),
      [[X(tx('1 request','1 个请求')),'=',H(`${fint(P_SREAD)} + 32 × 2 × 176,160,768`),'=',H(gbS(m.read1))],[X(tx(`${CMP_B} requests`,`${CMP_B} 个请求`)),'=',H(`${fint(P_SREAD)} + 32 × ${fixd(touched(CMP_B,MX.E,MX.k),3)} × 176,160,768`),'=',H(gbS(m.readB))]]);
    h+=MOB(tx(`A prefill of a ${tokK(CMP_N)}-token prompt is set by arithmetic:`,`prefill 一个 ${tokK(CMP_N)} token 的 prompt，耗时由计算决定：`),
      [[X(tx('prefill time','prefill 耗时')),'=','\\frac{'+fint(CMP_N)+' \\times '+X(tx('arithmetic per token','每个 token 的计算量'))+'}{'+X(tx('compute speed','算力'))+'}'],['',H(tx('with','其中')),H(t('effc','494.75 TFLOP/s'))]],
      [[X('Llama 2 13B'),'=',H(secS(a.pre))],[X(MODEL.replace(/[‑]/g,'-')),'=',H(secS(m.pre))],[X('Llama 2 70B'),'=',H(secS(c.pre))],[X(tx('dense 46.7B','稠密 46.7B')),'=',H(secS(dd.pre))]]);
    h+=MOB(tx('A decode step of few requests is set by reading the weights:','请求少时，一个 decode 步的耗时由读权重决定：'),
      [[X(tx('read time','读权重的时间')),'=','\\frac{'+X(tx('weights read','读的权重'))+'}{'+X(tx('memory speed','带宽'))+'}'],['',H(tx('with','其中')),H(t('effm','2.01 TB/s'))]],
      [[X('Llama 2 13B'),'=',H(`${gb3(a.read1)} → ${ms2(a.t1)}`)],[X(MODEL.replace(/[‑]/g,'-')),'=',H(`${gb3(m.read1)} → ${ms2(m.t1)}`)],[X(tx(`${CMP_B} requests`,`${CMP_B} 个请求`)),'=',H(`${gb3(m.readB)} → ${ms2(m.tB)}`)],[X('Llama 2 70B'),'=',H(`${gb3(c.read1)} → ${ms2(c.t1)}`)]]);
    const room=H100.mem*A.GiB*UTIL-OH*A.GiB, fitS=x=>`${fixd((room-x.mem)/A.GiB,1)} GiB ÷ ${fixd(CMP_C*x.kv/A.GiB,2)} GiB = ${fint(x.fit)}`;
    h+=MOB(tx(`How many requests of ${tokK(CMP_C)} tokens fit next to the weights on one H100:`,`一张 H100 上放完权重，还放得下多少个 ${tokK(CMP_C)} token 的请求：`),
      [[X(tx('requests that fit','放得下的请求')),'=','\\frac{'+X(gib1(room+OH*A.GiB))+' - '+X(OH+' GiB')+' - '+X(tx('weights','权重'))+'}{'+fint(CMP_C)+' \\times '+X(tx('KV cache per token','每个 token 的 KV cache'))+'}']],
      [[X('Llama 2 13B'),'=',H(fitS(a))],[X(MODEL.replace(/[‑]/g,'-')),'=',H(fitS(m))],[X('Llama 2 70B'),'=',H(fitS(c))]]);
    return h;}
  if(isEP()){const e=epRun(skew), b=epRun('bal');
    return h+MOB(tx(`Part ${SIX('imb')+1}, one MoE layer on the busiest GPU. The layer ends when the busiest GPU ends:`,`第 ${SIX('imb')+1} 部分，一个 MoE 层在最忙的 GPU 上。最忙的 GPU 算完，这一层才算完：`),
      [[X(tx('layer time','一层的耗时')),'=','\\frac{'+X(tx('expert copies','专家副本数'))+' \\times 2 \\times '+X(tx('expert','专家'))+'}{'+X(tx('compute speed','算力'))+'}']],
      [[X(tx('busiest GPU','最忙的 GPU')),'=',H(`${fint(Math.round(Math.max(...e.loads)))} ${tx('copies','份')} × 2 × 176,160,768 FLOP ÷ ${t('effc','494.75 TFLOP/s')}`),'=',H(B_(ms2(e.t)))],
       [X(tx('balanced','均衡时')),'=',H(`${fint(EP_COPIES/MX.E)} × 2 × 176,160,768 ÷ 494.75 TFLOP/s`),'=',H(ms2(b.t))]]);}
  if(r.m.fits&&!(mode==='guided'&&['mission','predict','router','mem'].includes(stepId()))){const d=r.d, bp=BPP[r.p];
    h+=MOB(tx(`Each of the ${t('B','B = '+fint(r.B))} requests picks k = ${t('k','2')} of the E = ${t('E','8')} experts in a layer. An expert no request picks is not read, which happens with chance`,`batch 里 ${t('B','B = '+fint(r.B))} 个请求，每个在每层从 E = ${t('E','8')} 个专家里选 k = ${t('k','2')} 个。没被任何请求选中的专家不用读，这种情况的概率是`)+` ${A.ui.math('\\left(1 - \\frac{k}{E}\\right)^{B}')}${tx('. So each layer reads, on average:','。所以每层平均要读：')}`,
      [[X(tx('experts read per layer','每层读的专家')),'=','E \\times \\left(1 - \\left(1 - \\frac{k}{E}\\right)^{B}\\right)'],['','=',`8 \\times \\left(1 - \\left(1 - \\frac{2}{8}\\right)^{${fint(r.B)}}\\right)`],['','=',H(B_(fixd(d.T,3)))]]);
    h+=MOB(tx('A decode step reads the shared weights, the experts it uses and every request’s KV cache:','一个 decode 步要读共享的权重、用到的专家，还有每个请求的 KV cache：'),
      [[X(tx('bytes read','读的字节')),'=',X(tx('bytes each','每个的字节数'))+' \\times ('+X(tx('shared','共享'))+' + 32 \\times '+X(tx('experts read','读的专家'))+' \\times '+X(tx('expert','专家'))+') + B \\times '+X(tx('context','上下文'))+' \\times '+X(tx('KV per token','每个 token 的 KV'))],
       ['','=',H(`${bp} × (${t('sread','1,474,564,096')} + 32 × ${fixd(d.T,3)} × 176,160,768) + ${t('B',fint(r.B))} × ${t('c',fint(r.c))} × ${t('kvt','131,072')}`)],['','=',H(`${fint(Math.round(d.by))} ${tx('bytes','字节')} = ${B_(gbS(d.by))}`)]]);
    h+=MOB(tx('The step takes the longer of computing and reading memory:','这一步的耗时，取计算和读显存中较长的那个：'),
      [[X(tx('compute time','计算时间')),'=','\\frac{'+X(tx('arithmetic','计算量'))+'}{'+X(tx('compute speed','算力'))+'}','=',H(`${t('B',fint(r.B))} × (2 × ${t('mult','12,748,853,248')} + 4 × 32 × 4,096 × ${fint(r.c+1)}) = ${fixd(d.fl/1e12,3)} TFLOP ÷ ${t('effc','494.75 TFLOP/s')}`),'=',H(`${fixd(d.tc*1e3,3)} ms`)],
       [X(tx('read time','读显存时间')),'=','\\frac{'+X(tx('bytes read','读的字节'))+'}{'+X(tx('memory speed','带宽'))+'}','=',H(`${fixd(d.by/1e9,3)} GB ÷ ${t('effm','2.01 TB/s')}`),'=',H(`${fixd(d.tm*1e3,3)} ms`)],
       [X('TPOT'),'=','\\max('+X(tx('compute','计算'))+',\\ '+X(tx('read','读显存'))+')','=',H(d.cb?tx('compute time','计算时间'):tx('read time','读显存时间')),'=',H(B_(ms2(d.t)))]]);
    h+=MOB(tx('Every request gets one token per step:','每一步，每个请求得到一个 token：'),
      [[X(tx('tokens per second, all requests','全部请求每秒的 token')),'=','\\frac{B}{\\text{TPOT}}','=',H(`${fint(r.B)} ÷ ${fixd(d.t*1e3,3)} ms`),'=',H(B_(fint(r.thr)))]]);}
  return h;
}
function render(){
  if(!app) return;
  if(mode==='challenge'){S.p=CH.p; S.c=CH.c;}
  const mx=memOf(S.p,S.c).max; S.B=Math.max(1,Math.min(S.B,Math.max(1,mx)));
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const gr=$('#moe-graph',app); gr.dataset.focus=mode==='guided'?(STEPS[step].focus||''):'';
  drawFlow(); drawPar(); drawRead(); drawEP(); drawCmp(); drawMetrics();
  const lg=$('#moe-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#moe-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  const sc=$('#moe-scope',app), [st,stip]=scopeOf(); if(sc.textContent!==st) sc.textContent=st; sc.dataset.tipx=stip;
  syncControls(); updateRail();
  if(animGo){animGo=false; animate();} else noteShown();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  const r=R(), lim=Math.max(1,r.m.max);
  $$('.seg-ctl[data-key="p"]',app).forEach(sg=>A.ui.setPressed(sg,S.p));
  $$('input[data-key="c"]',app).forEach(i=>{const v=C_STOPS.indexOf(S.c); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=tx(`${tokK(S.c)} tokens`,`${tokK(S.c)} 个 token`); i.setAttribute('aria-valuetext',tx(`${fint(S.c)} tokens`,`${fint(S.c)} 个 token`));});
  $$('input[data-key="B"],input[data-key="chB"]',app).forEach(i=>{const ch=i.dataset.key==='chB', L2=ch?chMax():lim;
    if(+i.max!==L2) i.max=L2; if(+i.value!==S.B) i.value=S.B; A.ui.setRange(i); i.disabled=!r.m.fits;
    const w=i.closest('.moe-bctl'), o=w.querySelector('output'); o.textContent=ch?userN(S.B):reqN(S.B); i.setAttribute('aria-valuetext',o.textContent);
    w.querySelector('[data-act="bminus"]').disabled=S.B<=1||!r.m.fits; w.querySelector('[data-act="bplus"]').disabled=S.B>=L2||!r.m.fits;
    w.querySelector('.moe-bnote').innerHTML=!r.m.fits?tx('The weights do not fit, so no request can run.','权重放不下，一个请求也跑不了。')
      :(ch?tx(`From 1 to ${fint(L2)}: at 4K tokens each, GPU memory holds the KV caches of at most ${fint(L2)} users.`,`从 1 到 ${fint(L2)}：每人 4K 个 token 时，显存最多放得下 ${fint(L2)} 个用户的 KV cache。`)
        :tx(`At most ${fint(L2)} at ${tokK(S.c)} tokens each: every request’s KV cache must fit next to the weights.`,`每个 ${tokK(S.c)} 个 token 时最多 ${fint(L2)} 个：每个请求的 KV cache 都要放在权重之外的显存里。`));});
  const lock=mode==='challenge'||(mode==='guided'&&stepId()!=='batch'&&stepId()!=='mem'), c=$('#moe-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#moe-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=mode==='challenge'?tx(`In the challenge you can change only the number of users, in the task card. Everything else is fixed: ${MODEL} in FP8 on one ${GPUN}, 4K tokens of context each.`,`挑战中只能在任务卡里改用户数。其余设置固定：一张 ${GPUN} 用 FP8 跑 ${MODEL}，每人 4K 个 token 的上下文。`)
    :tx('In this part the settings are fixed; the card holds what you can change. Every setting is free in Scenarios.','这一部分的设置是固定的，能改的都在卡片里。在“应用场景”里每个设置都能改。');
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const r=R();
  if(isEP()){const e=epRun(skew); A.announce(tx(`One MoE layer takes ${ms2(e.t)}, ${rs(e.ratio)} times the balanced time.`,`一个 MoE 层要 ${ms2(e.t)}，是均衡时的 ${rs(e.ratio)} 倍。`)); return;}
  if(!r.m.fits){A.announce(tx(`The weights take ${gib1(r.m.w)} and do not fit.`,`权重要 ${gib1(r.m.w)}，放不下。`)); return;}
  A.announce(tx(`${whoN(r.B)}: ${T1(r.d.T)} of 8 experts per layer, ${ms2(r.d.t)} per token.`,`${whoN(r.B)}：每层 8 个专家读 ${T1(r.d.T)} 个，每个 token ${ms2(r.d.t)}。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  const was=S.B; Object.assign(S,p);
  if(!C_STOPS.includes(S.c)) S.c=BASE.c; if(!BPP[S.p]) S.p=BASE.p;
  if(mode==='challenge'){S.p=CH.p; S.c=CH.c;}
  const lim=Math.max(1,memOf(S.p,S.c).max); S.B=Math.max(1,Math.min(Math.round(S.B)||1,lim));
  if(S.B!==was||p.p||p.c) {seed=1; animGo=true;}
  render();
}
function setSkew(id){if(!SKEW.some(x=>x.id===id)) return; skew=id; animGo=true; render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.id!=='imb') skew='bal';
  if(s.id!=='adv'){cmp='l13'; bench=0;}
  const st=$('#moe-stage',app); if(st) st.scrollTop=0;
  animGo=true; renderRail();
  const tg={router:'.moe-rt',mem:'#moe-ic-p',batch:'#moe-ic-B',adv:'.levers',imb:'.levers'}[s.id];
  if(tg){const el=$(`#moe-rail ${tg}`,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#moe-rail .step.current',app)||$('#moe-rail',app));}
/* keepStage (CONTRACT §11, R3.2, R9.7; as labs/spec-decoding.js; polish round 4): after Next, Back or a part's title the kit's reveal
   scrolls the new card's top up under the top bar; near the end of the bench that pushes the sticky stage up with it (part 7 entered
   from part 6 at 1440 × 900: the stage's top at −149 px, its head under the top bar). Once the page stops scrolling this pulls the
   stage back under the top bar without pushing the card's top off the screen, and does nothing once the reader has scrolled the card
   away */
let stageT=0;
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#moe-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#moe-rail .step.current',app)||$('#moe-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#moe-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#moe-insight',app)); else revealBox($('#moe-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={p:CH.p,c:CH.c,B:CH.start};}
  cmp='l13'; bench=0;
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  skew='bal';
  A.shell.setMode(app,m,DESCS());
  const st=$('#moe-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  animGo=true; renderRail();
}
const token=()=>[S.p,S.c,S.B].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==3) return null;
  const pr=p[0], c=+p[1], B=+p[2];
  if(!BPP[pr]||!C_STOPS.includes(c)||!Number.isInteger(B)||B<1||B>Math.max(1,memOf(pr,c).max)) return null;
  return {p:pr,c,B};
}

/* =====================================================================
   interview drill: seeded, from the engine (never 4K: the challenge's context)
   ===================================================================== */
const DRILL_C=[1024,8192], DRILL_B=[2,4,16,32];
function makeDrill(sd){const rnd=mulberry(sd*7919+17), pick=a=>a[Math.floor(rnd()*a.length)];
  const c=pick(DRILL_C); let B=pick(DRILL_B); B=Math.min(B,memOf('fp8',c).max); return {sd,c,B,d:decode(B,c,'fp8')};}
function renderDrill(){
  if(!app) return; const q=makeDrill(drillSeed), qd=$('#moe-drill-q',app), a=$('#moe-drill-a',app), d=q.d;
  qd.innerHTML=tx(`<b>${MODEL}</b> runs in FP8 on one ${GPUN}, at 60% of its 3.35 TB/s bandwidth. <b>${fint(q.B)}</b> requests decode together, each with <b>${fint(q.c)}</b> tokens of context. Assuming uniform routing, about how many experts per layer does a decode step read, how many bytes, and how long does it take?`,
                  `<b>${MODEL}</b> 用 FP8 跑在一张 ${GPUN} 上，按带宽 3.35 TB/s 的 60% 计。<b>${fint(q.B)}</b> 个请求一起 decode，每个有 <b>${fint(q.c)}</b> 个 token 的上下文。假设路由均匀，一个 decode 步每层大约读几个专家、读多少字节、要多久？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>About ${T1(d.T)} experts per layer, ${gbS(d.by)}, ${ms2(d.t)} per token (memory reads set it).</b></p>`+
    `<p>Experts read per layer:</p><div class="fx moe-fxp">${A.ui.math(`\\text{experts read per layer} = 8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{${q.B}}\\right) = ${fixd(d.T,3)}`)}</div>`+
    `<p>Bytes = 1,474,564,096 (shared weights, 1 byte each) + 32 × ${fixd(d.T,3)} × 176,160,768 (experts) + ${fint(q.B)} × ${fint(q.c)} × 131,072 (KV cache) = ${fint(Math.round(d.by))} bytes = ${gbS(d.by)}.</p>`+
    `<p>Time = ${fixd(d.by/1e9,3)} GB ÷ 2.01 TB/s = ${fixd(d.tm*1e3,3)} ms; the arithmetic, ${fixd(d.fl/1e12,3)} TFLOP ÷ 494.75 TFLOP/s = ${fixd(d.tc*1e3,3)} ms, is shorter. If it read only 2 experts per layer it would take ${ms2(decode(q.B,q.c,'fp8',2).t)}; all 8, ${ms2(decode(q.B,q.c,'fp8',8).t)}.</p>`,
    `<p><b>每层约读 ${T1(d.T)} 个专家，${gbS(d.by)}，每个 token ${ms2(d.t)}（由读显存决定）。</b></p>`+
    `<p>每层读的专家：</p><div class="fx moe-fxp">${A.ui.math(`\\text{每层读的专家} = 8 \\times \\left(1 - \\left(\\frac{6}{8}\\right)^{${q.B}}\\right) = ${fixd(d.T,3)}`)}</div>`+
    `<p>字节数 = 1,474,564,096（共享权重，每个 1 字节）+ 32 × ${fixd(d.T,3)} × 176,160,768（专家）+ ${fint(q.B)} × ${fint(q.c)} × 131,072（KV cache）= ${fint(Math.round(d.by))} 字节 = ${gbS(d.by)}。</p>`+
    `<p>耗时 = ${fixd(d.by/1e9,3)} GB ÷ 2.01 TB/s = ${fixd(d.tm*1e3,3)} ms；计算 ${fixd(d.fl/1e12,3)} TFLOP ÷ 494.75 TFLOP/s = ${fixd(d.tc*1e3,3)} ms，更短。如果每层只读 2 个专家，要 ${ms2(decode(q.B,q.c,'fp8',2).t)}；读满 8 个，要 ${ms2(decode(q.B,q.c,'fp8',8).t)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={p:{en:'Weights stored in',zh:'权重的存储格式'},c:{en:'Context per request',zh:'每个请求的上下文'},B:{en:'Requests decoding together',zh:'一起 decode 的请求数'}};
const TERM={
 attn:()=>tx('One attention block: Wq and Wo are 4,096 × 4,096; Wk and Wv are 4,096 × 1,024, since the 32 query heads share 8 KV heads of 128 dimensions (config.json).','一个 attention 块：Wq 和 Wo 是 4,096 × 4,096；Wk 和 Wv 是 4,096 × 1,024，因为 32 个 query head 共用 8 个 128 维的 KV head（来自 config.json）。'),
 exp:()=>tx('One expert: gate and up are 4,096 × 14,336, down is 14,336 × 4,096 (hidden_size and intermediate_size in config.json), the size of Mistral‑7B’s MLP block.','一个专家：gate 和 up 是 4,096 × 14,336，down 是 14,336 × 4,096（config.json 的 hidden_size 和 intermediate_size），和 Mistral‑7B 的 MLP 块一样大。'),
 rtr:()=>tx('The router of one layer: a 4,096 × 8 matrix, one column per expert.','一层的路由器：4,096 × 8 的矩阵，每个专家一列。'),
 emb:()=>tx('The embedding table and the LM head, 32,000 × 4,096 each (vocab_size and hidden_size; untied, so both are counted).','embedding 表和 LM head，各 32,000 × 4,096（vocab_size 和 hidden_size；两者不共享参数，所以都要算）。'),
 E:()=>tx('8 experts in each layer (num_local_experts in config.json).','每层 8 个专家（config.json 的 num_local_experts）。'),
 k:()=>tx('2 experts per token in each layer (num_experts_per_tok in config.json).','每层每个 token 选 2 个专家（config.json 的 num_experts_per_tok）。'),
 sread:()=>tx('1,474,564,096 parameters every decode step reads besides the experts: the attention blocks, RMSNorms, routers and the LM head. The embedding table is left out: a token only looks up its own row.','除了专家，每个 decode 步都要读的 1,474,564,096 个参数：attention 块、RMSNorm、路由器和 LM head。embedding 表不算：一个 token 只查自己那一行。'),
 mult:()=>tx('12,748,853,248 parameters each token multiplies: the shared parts that are read and 2 experts in each of the 32 layers. Each costs 2 FLOPs.','每个 token 要乘的 12,748,853,248 个参数：要读的共享部分，加上 32 层里每层 2 个专家。每个参数 2 FLOP。'),
 kvt:()=>tx('131,072 bytes (128 KiB): one token’s keys and values, 2 × 32 layers × 8 KV heads × 128 × 2 bytes in BF16.','131,072 字节（128 KiB）：一个 token 的 key 和 value，2 × 32 层 × 8 个 KV head × 128 × 2 字节（BF16）。'),
 effc:()=>tx(`494.75 TFLOP/s is 50% of the ${GPUN}’s dense BF16 peak, 989.5 TFLOP/s. The 50% is this lab’s assumption, as in the other labs.`,`494.75 TFLOP/s 是 ${GPUN} 稠密 BF16 峰值 989.5 TFLOP/s 的 50%。50% 是这个 Lab 的假设，和别的 Lab 一样。`),
 effm:()=>tx(`2.01 TB/s is 60% of the ${GPUN}’s peak bandwidth, 3.35 TB/s. The 60% is this lab’s assumption, as in the other labs.`,`2.01 TB/s 是 ${GPUN} 峰值带宽 3.35 TB/s 的 60%。60% 是这个 Lab 的假设，和别的 Lab 一样。`),
 p:()=>tx(`${BPP[S.p]} ${BPP[S.p]===1?'byte':'bytes'} per parameter: the weights are stored in ${PN[S.p]}.`,`每个参数 ${BPP[S.p]} 字节：权重用 ${PN[S.p]} 存。`),
 c:()=>tx(`${fint(S.c)} tokens of context per request: their KV cache is read in every decode step.`,`每个请求 ${fint(S.c)} 个 token 的上下文：每个 decode 步都要读它们的 KV cache。`),
 B:()=>tx(`${fint(S.B)} requests decode together: each decode step writes one token for each of them.`,`${fint(S.B)} 个请求一起 decode：每个 decode 步给它们各写一个 token。`),
 l13c:()=>tx('Llama 2 13B’s config.json: hidden size 5,120, intermediate size 13,824, 40 layers, 40 heads with a key and value each (no GQA), vocabulary 32,000, untied LM head.','Llama 2 13B 的 config.json：隐藏维度 5,120，中间维度 13,824，40 层，40 个 head，每个 head 都有自己的 key 和 value（没有 GQA），词表 32,000，LM head 不和 embedding 表共享参数。'),
 l70c:()=>tx('Llama 2 70B’s config.json: hidden size 8,192, intermediate size 28,672, 80 layers, 64 query heads sharing 8 KV heads, vocabulary 32,000, untied LM head.','Llama 2 70B 的 config.json：隐藏维度 8,192，中间维度 28,672，80 层，64 个 query head 共用 8 个 KV head，词表 32,000，LM head 不和 embedding 表共享参数。')
};
const termText=k=>{const tm=TERM[k]; return tm?tm():'';};
const termTipHTML=k=>TERM[k]?`<div>${esc(termText(k))}</div><div class="act">${jumpAct(k)}</div>`:'';
const FIXED=new Set(['attn','exp','rtr','emb','E','k','sread','mult','kvt','l13c','l70c']);
const locked=k=>(mode==='challenge'&&(k==='p'||k==='c'))||(mode==='guided'&&!((stepId()==='batch'&&k==='B')||(stepId()==='mem'&&k==='p')));
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='flow') return tx('Click to scroll to the router picture in the chart.','点击滚到图里的路由示意。');
  if(k==='effc'||k==='effm'||k==='eff') return mode==='challenge'?tx('Fixed during the challenge: 50% of the peak FLOP/s and 60% of the peak bandwidth.','挑战中固定不变：峰值 FLOP/s 的 50%，峰值带宽的 60%。'):tx('Click to jump to “Assumed values” under Settings.','点击跳到“自由调参”里的“假设值”。');
  if(FIXED.has(k)) return tx(`Fixed in this lab: it always uses ${MODEL}’s config.json.`,`这个 Lab 里固定不变：始终按 ${MODEL} 的 config.json。`);
  if(!CTLN[k]) return '';
  if(locked(k)) return mode==='challenge'?tx(`Fixed during the challenge: ${k==='p'?'FP8 weights':'4K tokens of context each'}.`,`挑战中固定不变：${k==='p'?'FP8 权重':'每人 4K 个 token 的上下文'}。`)
    :tx(`Fixed in this part; change it in Scenarios.`,`这一部分固定不变；在“应用场景”里可以改。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({p:$('#moe-ic-p',app)||$('#moe-ctl-p',app),c:$('#moe-ctl-c',app),B:$('#moe-ic-B',app)||$('#moe-ic-chB',app)||$('#moe-ctl-B',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#moe-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='flow'){const el=$('#moe-flow',app); A.tip.hide(); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#moe-controls',app)); return;}
  if(k==='effc'||k==='effm'||k==='eff'){const el=$('#moe-ctl-eff',app); if(mode!=='challenge'&&el&&!el.closest('.locked')) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  if(FIXED.has(k)){if(src) A.tip.show(src,termTipHTML(k)); return;}
  if(!CTLN[k]) return;
  if(locked(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{if(mode==='challenge') setState({B:CH.start}); else setState({...BASE});}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(S.B).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='replay'){animate(true); return;}
  if(a==='route'){if(rt>=SENT.length) return; wantObs=true; byLearner(()=>{rt++; animGo=true; render();}); revealSoon(); wantObs=false; return;}
  if(a==='bminus'||a==='bplus'){const d=a==='bplus'?1:-1, inRail=!!btn.closest('#moe-rail');
    const f=()=>byLearner(()=>setState({B:S.B+d})); if(inRail){wantObs=true; f(); revealSoon(); wantObs=false;} else bySettings(btn,()=>setState({B:S.B+d})); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#moe-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* a change under Settings never moves the page (R3.2) */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();} finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#moe-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='c') bySettings(e.target,()=>setState({c:C_STOPS[v]}));
    else if(k==='B'||k==='chB') bySettings(e.target,()=>setState({B:v}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#moe-rail')){wantObs=true; revealSoon(); wantObs=false;}});
  el.addEventListener('mousemove',e=>{const bx=e.target.closest&&e.target.closest('#moe-curve'); if(!bx) return; const B=curveAt(bx,e.clientX); if(B!=null&&B!==bx._hB) curveShow(bx,B);});
  el.addEventListener('mouseleave',e=>{if(!e.target||e.target.id!=='moe-curve') return; const mk=e.target.querySelector('.moe-cmark'); if(mk) mk.setAttribute('hidden',''); e.target._hB=null;},true);
  el.addEventListener('click',e=>{const bx=e.target.closest&&e.target.closest('#moe-curve'); if(!bx) return; const B=curveAt(bx,e.clientX); if(B==null) return;
    if(!curveCan()){if(A.tip&&A.tip.show) A.tip.show(bx,`<div>${esc(curveTip(B,false))}</div>`); return;}
    byLearner(()=>setState({B})); const nb=$('#moe-curve',app); if(nb) curveShow(nb,B);});
  el.addEventListener('keydown',e=>{const bx=e.target.closest&&e.target.closest('#moe-curve'); if(!bx||!curveCan()||(e.key!=='ArrowLeft'&&e.key!=='ArrowRight')) return; e.preventDefault();
    byLearner(()=>setState({B:S.B+(e.key==='ArrowRight'?1:-1)})); const nb=$('#moe-curve',app); if(nb){nb.focus({preventScroll:true}); curveShow(nb,S.B);}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(d.go!=null){const i=+d.go; if(mode!=='guided'){step=Math.max(0,Math.min(STEPS.length-1,i)); setMode('guided'); A.ui.reveal($('#moe-rail',app));} else {enterStep(i); revealStep(); keepStage();} return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); keepStage(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#moe-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&stepId()==='adv'){byLearner(()=>setCmp(d.lever)); requestAnimationFrame(()=>{if(app) revealBox($('#moe-rail [data-obs]',app));}); return;}
    if(d.v!=null&&b.closest('.seg-ctl[data-key="bench"]')){bench=Math.max(0,Math.min(BENCH_SHOW.length-1,+d.v)); byLearner(()=>render()); return;}
    if(d.lever!=null&&isEP()){byLearner(()=>setSkew(d.lever)); requestAnimationFrame(()=>{if(app) revealBox($('#moe-rail [data-obs]',app));}); return;}
    if(d.v!=null){const sg=b.closest('.seg-ctl[data-key="p"]'); if(sg){if(mode==='challenge') return; wantObs=!!b.closest('#moe-rail'); bySettings(b,()=>setState({p:d.v})); if(wantObs) revealSoon(); wantObs=false; return;}}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#moe-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#moe-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S as it was (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount.
     watched (the routing pictures already seen to the end) is kept so that a picture seen before the reload is drawn finished after it;
     a picture whose dots are still running counts as seen */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,sbTouched,rt,skew,cmp,cmpSeen:[...cmpSeen],bench,
    watched:[...watched,...(animT&&runSig&&!watched.has(runSig)?[runSig]:[])].slice(-60),drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPoint(S.B).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse([x.p,x.c,x.B].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    rt=Number.isInteger(o.rt)&&o.rt>=0&&o.rt<=SENT.length?o.rt:0; skew=SKEW.some(x=>x.id===o.skew)?o.skew:'bal';
    const okC=x=>['l13','l70','d47'].includes(x); cmp=okC(o.cmp)?o.cmp:'l13';
    cmpSeen=new Set(['l13']); if(Array.isArray(o.cmpSeen)) o.cmpSeen.forEach(x=>{if(okC(x)) cmpSeen.add(x);}); cmpSeen.add(cmp);
    bench=Number.isInteger(o.bench)&&o.bench>=0&&o.bench<BENCH_SHOW.length?o.bench:0;
    watched.clear(); if(Array.isArray(o.watched)) o.watched.slice(-60).forEach(x=>{if(typeof x==='string'&&x.length<600&&/^[a-z]+\|\d*\|\d+\|[\d>,]*$/.test(x)) watched.add(x);});
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#moe-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#moe-rail',app));},token});
    bind(app); renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    fromSnap=false;
    const rail=$('#moe-rail',app); rail.innerHTML=railHTML()+railExtra(); railWhy(rail); A.ui.initRanges(rail);
    render();
  },
  leave(){clearTimeout(annT); stageStop(); cancelAnimationFrame(animT); animT=0; animShown=''; root=null; app=null;},
  onLang(){if(root){const d=$('#moe-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#moe-math',app).open=true;}},
  onResize(){if(app){drawFlow(); if(!$('#moe-read',app).hidden) drawCurve();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(stepId()==='adv'?$('#moe-cmp',app):$('#moe-flow',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#moe-graph',app),
    render({status,vis,read}){if(!app) return; const r=R(), s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(isEP()){const e=epRun(skew), mx=Math.max(...e.gpu.map(g=>g.t));
        vis.innerHTML=`<div class="moe-dk">${e.gpu.map(g=>`<i class="moe-dk-e on" style="opacity:${(0.35+0.65*g.t/mx).toFixed(2)}"></i>`).join('')}</div>`;
        read.innerHTML=esc(tx(`One MoE layer: ${ms2(e.t)} (balanced ${ms2(e.bal)})`,`一个 MoE 层：${ms2(e.t)}（均衡时 ${ms2(e.bal)}）`)); return;}
      if(stepId()==='adv'){const m=cmpOf('mx'), b=cmpOf(cmp), bi=BENCH_SHOW[bench];
        vis.innerHTML=`<div class="moe-dk moe-dk2"><i class="moe-dk-f" style="width:${(m.fl/b.fl*100).toFixed(1)}%"></i></div>`;
        read.innerHTML=esc(tx(`${MODEL} vs ${cmpName(cmp)}: ${fixd(m.fl/1e9,1)} vs ${fixd(b.fl/1e9,1)} GFLOP per token`,`${MODEL} 比${cmpName(cmp)}：每个 token ${fixd(m.fl/1e9,1)} 比 ${fixd(b.fl/1e9,1)} GFLOP`))+`<span class="dk-mem">${esc(cmp==='d47'?'':tx(`, ${BENCH_COLS[bi]} ${fixd(BENCH.mx[bi],1)}% vs ${fixd(BENCH[cmp][bi],1)}%`,`，${BENCH_COLS[bi]} ${fixd(BENCH.mx[bi],1)}% 比 ${fixd(BENCH[cmp][bi],1)}%`))}</span>`; return;}
      const fm=flowModel(), on=fm.loads.map(v=>v>0);
      vis.innerHTML=`<div class="moe-dk">${on.map(x=>`<i class="moe-dk-e${x?' on':''}"></i>`).join('')}</div>`;
      const id=stepId();
      if(id==='mission'||id==='predict'||id==='router'){read.innerHTML=esc(tx('Each token computes with 12.9B of 46.7B parameters','每个 token 用 46.7B 个参数里的 12.9B 个计算')); return;}
      if(!r.m.fits){read.innerHTML=esc(tx(`${PN[r.p]} weights: ${gib1(r.m.w)}, over the ${gib1(r.m.usable)} usable`,`${PN[r.p]} 权重：${gib1(r.m.w)}，超过可用的 ${gib1(r.m.usable)}`)); return;}
      if(id==='mem'){read.innerHTML=esc(tx(`${PN[r.p]} weights: ${gib1(r.m.w)} of ${gib1(r.m.usable)} usable`,`${PN[r.p]} 权重：可用 ${gib1(r.m.usable)} 里占 ${gib1(r.m.w)}`)); return;}
      const who=mode==='challenge'?userN(r.B):reqN(r.B);
      read.innerHTML=mode==='challenge'?esc(tx(`${who}: ${ms2(r.d.t)} per token`,`${who}：每个 token ${ms2(r.d.t)}`))+`<span class="dk-mem">${esc(tx(' (the goal is at most 30 ms)','（目标是最多 30 ms）'))}</span>`
        :esc(tx(`${who} at ${tokK(r.c)} tokens each: ${T1(r.d.T)} of 8 experts per layer, ${ms2(r.d.t)} per token`,`${who}，每个 ${tokK(r.c)} 个 token：每层读 8 个专家里的 ${T1(r.d.T)} 个，每个 token ${ms2(r.d.t)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__moe=Object.freeze({
  decode:(B,c,p,T)=>deep(decode(B,c,p,T)), memOf:(p,c)=>deep(memOf(p,c)), compute:s=>deep(compute(s)), touched,
  chPoint:B=>deep(chPoint(B)), chBest, chMax, epRun:id=>deep(epRun(id)), routeTok:i=>deep(routeTok(i)), sampleRoute:(B,sd)=>deep(sampleRoute(B,sd)), drill:s=>deep(makeDrill(s)),
  exits:()=>{const x=EXITS(); return {explain:x.explain.map(q=>({q:q.q,o:q.o,a:q.a})),transfer:x.transfer.map(q=>({q:q.q,o:q.o,a:q.a,why:q.why}))};},
  cmpModels:()=>deep(cmpModels()), denseP:c=>deep(denseP(c)), beats, l70Step:(B,c)=>deep(l70Step(B,c)), l70Fit, l70Takes, curveLines:()=>deep(lastCurve),
  consts:Object.freeze({BENCH:deep(BENCH),BENCH_COLS:BENCH_COLS.slice(),BENCH_SHOW:BENCH_SHOW.slice(),DENSE:deep(DENSE),CMP_B,MX:{...MX},H100:{...H100},EFF:{...EFF},UTIL,OH,KV_TOK,P_ATTN,P_NORM,P_ROUTER,P_EXPERT,P_EMB,P_HEAD,P_FNORM,P_LAYER,P_TOTAL,P_EXPS,P_SHARED,P_SREAD,P_ACTIVE,P_MULT,P_SLOT,
    C_STOPS:C_STOPS.slice(),BASE:{...BASE},CH:deep(CH),EP:{...EP},EP_COPIES,SKEW:deep(SKEW),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,SENT:SENT.slice()}),
  state:()=>({...S,mode,step,pred,observed:{...observed},chSubmitted,chReveal,rt,skew,seed,cmp,cmpSeen:[...cmpSeen],bench,plays:animN})
});
})();
