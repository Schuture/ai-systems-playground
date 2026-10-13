/* roofline (the root "ML basics", world roots): Is your kernel limited by compute or by memory?
   Fidelity: Formula. Every number comes from one model: a kernel needs FLOPs and bytes moved to or from GPU memory (HBM); the GPU reads
   and computes at the same time, so it takes max(FLOPs ÷ peak FLOP/s, bytes ÷ peak bandwidth) and reaches min(peak, I × bandwidth),
   where I = FLOPs ÷ bytes (arithmetic intensity). The ridge is peak ÷ bandwidth. 100% of the datasheet peaks: the roofline is the bound.
   - Constants shared with labs/prefill-decode.js: H100 SXM 989.5 TFLOP/s dense BF16 (1,979 with sparsity ÷ 2), 3.35 TB/s; A100 80GB SXM
     312 TFLOP/s, 2,039 GB/s; Llama‑3.1‑8B's shapes, its W_READ (every parameter but the embedding table: 7,504,924,672) and its prefill
     formula. prefill-decode prices the same steps at 50% / 60% of the peaks; the Fidelity popover says so with its numbers.
   - Kernels: the decode step's weight matmuls (batch B: 2 FLOPs per parameter per token, 2 bytes per parameter in BF16, 1 in FP8 weights
     multiplied in BF16), prefill of N tokens, attention during decode (B users at context c; I ≈ 4, the GQA group size), a square matmul n
     (I = n/3), an elementwise residual add (1 FLOP per 6 bytes). The challenge's H200 (989.5 TFLOP/s, 4.8 TB/s) appears nowhere else.
   Words (EN / ZH): arithmetic intensity / 算术强度 (FLOP per byte / 每字节的 FLOP 数); ridge / 拐点; memory-bound / 受显存带宽限制;
   compute-bound / 受算力限制; peak / 峰值; part / 部分 for guided cards; step / 步 for the engine.
   window.__rf is a read-only hook for tests/roofline_numbers.py and tests/roofline_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='roofline';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const MODEL='Llama‑3.1‑8B';                        // no-break hyphens: a model name never breaks

/* =====================================================================
   engine
   ===================================================================== */
const GPUS={
 a100:{id:'a100',name:'A100',full:'A100 80GB SXM',peak:312e12,bw:2.039e12},
 h100:{id:'h100',name:'H100',full:'H100 SXM',peak:1979e12/2,bw:3.35e12},
 h200:{id:'h200',name:'H200',full:'H200 SXM',peak:1979e12/2,bw:4.8e12}};
const LL={layers:32,hidden:4096,heads:32,kvHeads:8,headDim:128,vocab:128256,ffn:14336};
const P_EMB=LL.vocab*LL.hidden, P_HEAD=LL.vocab*LL.hidden;
const P_LAYER=2*LL.hidden*LL.hidden+2*LL.hidden*LL.kvHeads*LL.headDim+3*LL.hidden*LL.ffn+2*LL.hidden;
const P_LAYERS=LL.layers*P_LAYER+LL.hidden, P_TOTAL=P_LAYERS+P_EMB+P_HEAD;   // 8,030,261,248
const W_PAR=P_LAYERS+P_HEAD;                                                  // 7,504,924,672 parameters a step reads
const KV_TOK=2*LL.layers*LL.kvHeads*LL.headDim*2;                             // 131,072 bytes (128 KiB) per token, BF16
const attn=(a,b)=>2*LL.layers*LL.hidden*(b*(b+1)-a*(a+1));                    // QKᵀ and AV FLOPs, all layers, query positions a..b-1
const EW_N=2048*LL.hidden;                                                    // the residual add: 2,048 tokens × 4,096
const WB={bf16:2,fp8:1};
/* one kernel's FLOPs and bytes, and what the bytes are (weights, KV cache, other) */
function kern(s){
  const wb=WB[s.p]||2;
  if(s.k==='dec') return {k:'dec',fl:2*W_PAR*s.B,by:wb*W_PAR,mem:'w'};
  if(s.k==='pre') return {k:'pre',fl:2*P_LAYERS*s.N+2*P_HEAD+attn(0,s.N),by:wb*W_PAR+KV_TOK*s.N,mem:'w',wby:wb*W_PAR,kvby:KV_TOK*s.N};
  if(s.k==='att') return {k:'att',fl:s.B*attn(s.c,s.c+1),by:s.B*KV_TOK*s.c,mem:'kv'};
  if(s.k==='mm') return {k:'mm',fl:2*s.n**3,by:2*3*s.n*s.n,mem:'x'};
  return {k:'ew',fl:EW_N,by:6*EW_N,mem:'x'};
}
/* the roofline: time, TFLOP/s reached, share of the peak, bound */
function roof(kk,g){
  const I=kk.fl/kk.by, tc=kk.fl/g.peak, tm=kk.by/g.bw, t=Math.max(tc,tm), att=kk.fl/t;
  return {...kk,I,tc,tm,t,att,share:att/g.peak,ridge:g.peak/g.bw,bound:tc>=tm?'compute':'memory',gpu:g.id};
}
const ev=s=>roof(kern(s),GPUS[s.gpu]||GPUS.h100);
const BST=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024];
const NST=[128,256,512,1024,2048,4096,8192,16384,32768];
const CST=[512,1024,2048,4096,8192,16384,32768];
const MST=[64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192];
const KINDS=['dec','pre','att','mm','ew'];
const BASE={k:'dec',B:1,N:2048,c:4096,n:256,gpu:'h100',p:'bf16'};
/* part 5: decode at batch 16 and prefill of 2,048 tokens, each option from the same initial setup */
const LV=[{id:'base',gpu:'h100',p:'bf16',t:{en:'H100, BF16 weights (initial setup)',zh:'H100，BF16 权重（初始设置）'},d:{en:'No change: the setup to compare against',zh:'不做改动，作为比较的基准'}},
  {id:'a100',gpu:'a100',p:'bf16',t:{en:'An A100 instead',zh:'换成 A100'},d:{en:'312 TFLOP/s and 2.04 TB/s: a third of the H100’s FLOP/s and 61% of its bandwidth',zh:'312 TFLOP/s、2.04 TB/s：算力是 H100 的三分之一，带宽是它的 61%'}},
  {id:'fp8',gpu:'h100',p:'fp8',t:{en:'FP8 weights on the H100',zh:'H100 上用 FP8 权重'},d:{en:'1 byte per weight instead of 2, still multiplied in BF16',zh:'每个权重 1 字节而不是 2 字节，乘法仍用 BF16'}}];
const lvOf=id=>LV.find(v=>v.id===id)||LV[0];
const DUO_B=16, DUO_N=2048;
const duoOf=id=>{const v=lvOf(id); return {dec:ev({...BASE,k:'dec',B:DUO_B,gpu:v.gpu,p:v.p}),pre:ev({...BASE,k:'pre',N:DUO_N,gpu:v.gpu,p:v.p})};};
/* part 6 and the scenarios: five kernels of serving Llama‑3.1‑8B on one H100 */
const PRESETS=[
 {id:'chat1',s:{k:'dec',B:1},t:{en:'Chat with one user',zh:'只和一个用户聊天'},d:{en:'One request at a time, like a personal assistant serving a single user',zh:'一次只服务一个请求，像只为一个用户服务的个人助手'}},
 {id:'busy',s:{k:'dec',B:256},t:{en:'A busy server',zh:'繁忙的服务端'},d:{en:'One decode step for 256 requests at once',zh:'一个 decode 步同时处理 256 个请求'}},
 {id:'prompt',s:{k:'pre',N:2048},t:{en:'A long prompt arrives',zh:'来了一个长 prompt'},d:{en:'Prefill of a 2,048-token prompt in one step',zh:'一步 prefill 完一个 2,048 个 token 的 prompt'}},
 {id:'attn',s:{k:'att',B:64,c:4096},t:{en:'Attention during decode',zh:'decode 时的 attention'},d:{en:'64 users, each reading their own 4K-token KV cache',zh:'64 个用户，每人读自己 4K 个 token 的 KV cache'}},
 {id:'resid',s:{k:'ew'},t:{en:'A residual add',zh:'一次残差相加'},d:{en:'Add two 2,048 × 4,096 tensors, number by number',zh:'把两个 2,048 × 4,096 的张量逐个数值相加'}}];
const presetState=p=>({...BASE,...p.s,gpu:'h100',p:'bf16'});
/* the challenge: a GPU the lab never showed */
const CH={gpu:'h200',Bs:[1,16,64,256,512],Ps:['bf16','fp8'],start:{B:1,p:'bf16'},goal:0.5};
const chEval=(B,p)=>{const r=ev({...BASE,k:'dec',B,p,gpu:CH.gpu}); return {B,p,r,share:r.share,ok:r.share>=CH.goal-1e-12};};
const chChanges=(B,p)=>(B!==CH.start.B?1:0)+(p!==CH.start.p?1:0);
/* the fewest requests first (each request in a batch is a user waiting, with a KV cache in memory), then the fewest changes */
function chBest(){let b=null; CH.Bs.forEach(B=>CH.Ps.forEach(p=>{const e=chEval(B,p); if(!e.ok) return; const c=chChanges(B,p);
  if(!b||B<b.B||(B===b.B&&c<b.c)) b={B,p,c};})); return b;}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sig=(v,n)=>{if(v===0) return '0'; const d=Math.max(0,n-1-Math.floor(Math.log10(Math.abs(v)))); return fixd(v,d);};
/* a time, 3 significant digits, in s, ms or µs */
const fTime=t=>t>=1?sig(t,3)+' s':(t>=1e-3?sig(t*1e3,3)+' ms':(t>=1e-6?sig(t*1e6,3)+' µs':sig(t*1e9,3)+' ns'));
/* FLOP per byte: two decimals under 1, one under 100, whole numbers above */
const fI=I=>I<1?fixd(I,2):(I<100?fixd(I,1):fint(Math.round(I)));
/* TFLOP/s reached: the peaks keep their datasheet digits (989.5) */
const fTF=f=>{const v=f/1e12; return v>=10?fixd(v,1):(v>=1?fixd(v,2):fixd(v,3));};
/* the share of the peak: never rounded up to 100% (R6.10) */
const fSh=x=>{const v=x*100; if(x>=1-1e-12) return '100%'; if(v>=99.95) return '>99.9%'; if(v>=10) return fixd(v,1)+'%'; if(v>=1) return fixd(v,1)+'%'; if(v>=0.01) return fixd(v,2)+'%'; return '<0.01%';};
/* bytes and FLOPs in decimal units (bandwidth is in decimal TB/s) */
const fBy=b=>b>=1e9?sig(b/1e9,3)+' GB':(b>=1e6?sig(b/1e6,3)+' MB':sig(b/1e3,3)+' kB');
const fFl=f=>f>=1e12?sig(f/1e12,3)+' TFLOP':(f>=1e9?sig(f/1e9,3)+' GFLOP':(f>=1e6?sig(f/1e6,3)+' MFLOP':fint(f)+' FLOP'));
const fBW=g=>g.bw>=3e12?fixd(g.bw/1e12,2)+' TB/s':fixd(g.bw/1e12,2)+' TB/s';
const fPk=g=>fixd(g.peak/1e12,g.peak%1e12===0?0:1)+' TFLOP/s';
const fR=r=>fint(Math.round(r));                  // the ridge, whole FLOP per byte moved from memory
const fX=x=>x>=10?fint(Math.round(x)):fixd(x,1);   // "1.6 times"
const tokN=n=>tx(`${fint(n)} tokens`,`${fint(n)} 个 token`);
const tokK=n=>n>=1024&&n%1024===0?`${n/1024}K`:fint(n);
const reqN=n=>tx(n===1?'1 request':`${fint(n)} requests`,`${fint(n)} 个请求`);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
/* a term whose hover holds a formula on its own line (R6.29): the kit's .t shows termTip's HTML; it has no click action */
const DFX=(k,x)=>`<span class="t rf-dfx" tabindex="0" data-ctl="dfx" data-term="${k}">${x}</span>`;
const DFXT={
 roof:()=>`<div>${esc(tx('The roofline is the chart’s thick line: the most FLOP/s a kernel can reach at each arithmetic intensity.','roofline 是图里的那条粗线：在每种算术强度下，kernel 最多能达到的 FLOP/s。'))}</div>`+
   `<div class="fx rf-fxw">${A.ui.math(tx('\\text{FLOP/s reached} = \\min(\\text{peak},\\ \\text{intensity} \\times \\text{bandwidth})','\\text{达到的 FLOP/s} = \\min(\\text{峰值算力},\\ \\text{算术强度} \\times \\text{带宽})'))}</div>`+
   `<div>${esc(tx('The arithmetic intensity is the FLOPs per byte moved from memory. The slanted part is set by the bandwidth, the flat part by the peak FLOP/s.','算术强度就是每从显存搬运 1 字节做的 FLOP 数。斜的一段由显存带宽决定，平的一段由峰值算力决定。'))}</div>`
};
/* the time formula, typeset (R6.29): the 讲清楚 answer and the time hover use the same form */
const FX_T=()=>tx('\\text{time} = \\max\\left(\\frac{\\text{FLOPs}}{\\text{peak FLOP/s}},\\ \\frac{\\text{bytes}}{\\text{bandwidth}}\\right)','\\text{耗时} = \\max\\left(\\frac{\\text{FLOP 数}}{\\text{峰值算力}},\\ \\frac{\\text{字节数}}{\\text{显存带宽}}\\right)');
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const BIG=n=>A.ui.big?A.ui.big(n):esc(A.fmt.big(n));
const nbh=s=>String(s);

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seen64=false, seenPast=false, mmPast=false, lv='base', lvTried=new Set(['base']);
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null;
let prev=null, prevGpu=null, drillSeed=2, drillOpen=false;
const shownPrev={};
const R=()=>ev(S);
const B0=()=>ev({...BASE});                          // the mission: decode at batch 1, H100, BF16

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 flop:()=>tx('A FLOP is one floating-point operation, an add or a multiply. Multiplying a weight by a number and adding it to a running sum counts as 2 FLOPs.',
             'FLOP 是一次浮点运算：一次加法或一次乘法。把一个权重乘上一个数值，再加到累加和里，算 2 个 FLOP。'),
 hbm:()=>tx('HBM (high-bandwidth memory) is the GPU’s main memory, where the weights and the KV cache live. The math units can work only on numbers that have been read from it, at the GPU’s memory bandwidth.',
            'HBM（高带宽显存）是 GPU 的主显存，权重和 KV cache 都存在这里。计算单元只能处理从它读进来的数值，读的速度就是 GPU 的显存带宽。'),
 wread:()=>tx(`${MODEL} has ${A.fmt.big(P_TOTAL)} parameters. A decode step reads ${A.fmt.big(W_PAR)} of them: all but the embedding table’s ${A.fmt.big(P_EMB)}, from which it only looks up one row per token. At 2 bytes each (BF16) that is 15.0 GB, where 1 GB is 10^9 bytes; the whole model is 16.1 GB.`,
              `${MODEL} 有 ${A.fmt.big(P_TOTAL)} 个参数。decode 步读其中的 ${A.fmt.big(W_PAR)} 个：除了 embedding 表的 ${A.fmt.big(P_EMB)} 个参数以外的全部，embedding 表每个 token 只查一行。每个参数 2 字节（BF16），一共 15.0 GB（1 GB = 10^9 字节）；整个模型是 16.1 GB。`),
 ai:()=>tx('Arithmetic intensity is a kernel’s FLOPs divided by the bytes it moves to or from GPU memory: how much arithmetic it does with each byte it reads or writes. The Roofline paper calls it operational intensity.',
           '算术强度是一个 kernel 的 FLOP 数除以它和显存之间搬运的字节数：每读写 1 字节，它做多少运算。Roofline 论文把它叫作 operational intensity。'),
 ridge:()=>tx('The ridge is the GPU’s peak FLOP/s divided by its memory bandwidth: the arithmetic intensity at which computing takes exactly as long as moving the bytes. Left of it a kernel is memory-bound, right of it compute-bound.',
              '拐点是 GPU 的峰值算力除以显存带宽：算术强度等于它时，计算和搬运字节的耗时正好一样长。在它左边，kernel 受显存带宽限制；在它右边，受算力限制。'),
 kernel:()=>tx('A kernel is one program the GPU runs over a whole tensor, such as one matrix multiply. A model’s step is a sequence of kernels.',
               'kernel 是 GPU 在一整个张量上运行的一个程序，比如一次矩阵乘法。模型的一步由一串 kernel 组成。'),
 batch:()=>tx('The batch is how many requests one decode step serves at once. Each request adds one token to the step: every weight read from memory is multiplied by every request’s token.',
              'batch 是一个 decode 步同时服务的请求数。每个请求给这一步加 1 个 token：从显存读进来的每个权重，都要和每个请求的 token 相乘。'),
 overlap:()=>tx('The GPU reads the next tiles of weights while its math units work on the tiles already read, so the two happen at the same time. The step therefore takes the longer of the two times, not their sum.',
                'GPU 一边读下一块权重，一边计算已经读进来的那几块，两件事同时进行。所以一步的耗时取两者中较长的那个，而不是两者相加。'),
 fp8:()=>tx('FP8 stores each weight in 1 byte instead of 2. Here the kernel converts each weight back to BF16 before multiplying (weight-only quantization), so the arithmetic and its peak stay those of BF16.',
            'FP8 用 1 字节而不是 2 字节存一个权重。这里 kernel 在相乘之前把每个权重转回 BF16（只量化权重），所以计算和它的峰值仍按 BF16 算。'),
 roofline:()=>tx('The roofline is the chart’s thick line: the most FLOP/s a kernel can reach at each arithmetic intensity (FLOPs per byte moved from memory), the smaller of the peak FLOP/s and the intensity times the memory bandwidth. Its slanted part is set by the bandwidth, its flat part by the peak FLOP/s.',
                 'roofline 是图里的那条粗线：在每种算术强度（每从显存搬运 1 字节做的 FLOP 数）下，kernel 最多能达到的 FLOP/s，也就是峰值算力和“算术强度 × 显存带宽”中较小的那个。斜的一段由带宽决定，平的一段由峰值算力决定。'),
 ridgePd:()=>TIP.ridge()+tx(' This lab takes the datasheet peaks at 100%. Prefill vs decode assumes kernels reach 50% of the peak FLOP/s and 60% of the bandwidth, which puts the H100’s ridge at about 246.',
                 '这个 Lab 按规格表峰值的 100% 算。“Prefill 和 decode”这个 Lab 假设 kernel 只达到峰值算力的 50%、带宽的 60%，那样 H100 的拐点约是 246。'),
 tflops:()=>tx('TFLOP/s reached is the kernel’s FLOPs divided by its time. It can never exceed the peak, nor the arithmetic intensity times the bandwidth: the roof in the chart.',
               '达到的 TFLOP/s 是 kernel 的 FLOP 数除以它的耗时。它不会超过峰值，也不会超过算术强度乘以带宽：就是图里的屋顶线。')
};
const kname=s=>{
  if(s.k==='dec') return tx(`decode, batch ${fint(s.B)}`,`decode，batch ${fint(s.B)}`);
  if(s.k==='pre') return tx(`prefill, ${tokK(s.N)} tokens`,`prefill，${tokK(s.N)} 个 token`);
  if(s.k==='att') return tx(`attention, ${fint(s.B)} × ${tokK(s.c)}`,`attention，${fint(s.B)} × ${tokK(s.c)}`);
  if(s.k==='mm') return tx(`matmul, n = ${fint(s.n)}`,`矩阵乘法，n = ${fint(s.n)}`);
  return tx('residual add','残差相加');
};
/* on a narrow chart whose title names the kernel, the dot's label keeps only its size */
const kshort=s=>s.k==='att'?`${fint(s.B)} × ${tokK(s.c)}`:s.k==='dec'?`batch ${fint(s.B)}`:(s.k==='pre'?tokN(s.N):(s.k==='mm'?`n = ${fint(s.n)}`:kname(s)));
/* a kernel described in a full phrase, for titles and tooltips */
const kfull=s=>{
  if(s.k==='dec') return tx(`The weight matmuls of one decode step of ${MODEL} for ${reqN(s.B)}`,`decode 步的权重矩阵乘法：${MODEL}，${reqN(s.B)}`);
  if(s.k==='pre') return tx(`Prefill of a ${fint(s.N)}-token prompt on ${MODEL}`,`${MODEL} prefill 一个 ${fint(s.N)} 个 token 的 prompt`);
  if(s.k==='att') return tx(`Attention in one decode step for ${fint(s.B)} users with ${fint(s.c)} tokens of context each`,`一个 decode 步里 ${fint(s.B)} 个用户的 attention，每人 ${fint(s.c)} 个 token 的上下文`);
  if(s.k==='mm') return tx(`A square matmul, (${fint(s.n)} × ${fint(s.n)}) times (${fint(s.n)} × ${fint(s.n)}), in BF16`,`一个 (${fint(s.n)} × ${fint(s.n)}) 乘 (${fint(s.n)} × ${fint(s.n)}) 的 BF16 方阵乘法`);
  return tx('A residual add of two 2,048 × 4,096 tensors, number by number, in BF16','两个 2,048 × 4,096 的 BF16 张量逐个数值相加（残差相加）');
};
const gpuOf=s=>GPUS[s.gpu]||GPUS.h100;
const precTxt=p=>p==='fp8'?tx('FP8 weights','FP8 权重'):tx('BF16 weights','BF16 权重');
const boundTxt=r=>r.bound==='compute'?tx('compute-bound','受算力限制'):tx('memory-bound','受显存带宽限制');

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'same',t:{en:'About as long (less than twice)',zh:'差不多一样长（不到 2 倍）'}},
  {v:'few',t:{en:'Several times as long (2 to 20 times)',zh:'长好几倍（2 到 20 倍）'}},
  {v:'x64',t:{en:'About 64 times as long (more than 20 times)',zh:'长约 64 倍（超过 20 倍）'}}];
const PRED_RIGHT='same';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const preLock=()=>mode==='guided'&&!pred&&step<=1;
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const bIdx=B=>{const i=BST.indexOf(B); return i<0?0:i;};
const mIdx=n=>{const i=MST.indexOf(n); return i<0?0:i;};
const bCtl=(id,inline,note)=>A.ui.range({id,key:'B',label:tx('Batch: requests per step','batch：每步的请求数'),min:0,max:BST.length-1,value:bIdx(S.B),out:fint(S.B),inline,note:note||''});
const mCtl=(id,inline)=>A.ui.range({id,key:'n',label:tx('Matrix size n','矩阵大小 n'),min:0,max:MST.length-1,value:mIdx(S.n),out:fint(S.n),inline});
const lvRes=id=>{const d=duoOf(id); return tx(`decode ${fTime(d.dec.t)}, prefill ${fTime(d.pre.t)}`,`decode ${fTime(d.dec.t)}，prefill ${fTime(d.pre.t)}`);};
const lvCtl=()=>`<div class="rf-lv" id="rf-ic-lv">`+A.ui.levers({aria:tx('One change, pick one','一种改动，选一种'),value:lv,options:LV.map(v=>({v:v.id,title:L(v.t),desc:L(v.d),result:esc(lvTried.has(v.id)||v.id===lv?lvRes(v.id):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'one',
  lead:()=>{const b=B0(); return tx(`At batch 1 a decode step takes ${fTime(b.t)}, yet the math units are busy only ${fSh(b.share)} of that time. Why?`,`batch 为 1 时，一个 decode 步要 ${fTime(b.t)}，计算单元却只有 ${fSh(b.share)} 的时间在忙。为什么？`);},
  enter:()=>({...BASE}),
  body:()=>{const b=B0(), g=GPUS.h100;
   return tx(`<p>You use ${MODEL} on one H100 to answer a single user. Each new token takes one ${DEF('decode step','A decode step generates the next token for every request in the batch. It runs every layer of the model once, so it reads every weight from GPU memory once.')}.</p><p>In each decode step the GPU does two things at the same time. It reads all ${DEF('15.0 GB',TIP.wread())} of weights from ${DEF('GPU memory (HBM)',TIP.hbm())}. And it does one multiply-add of each weight with the new token, ${fFl(b.fl)} in all (2 ${DEF('FLOPs',TIP.flop())} per weight).</p><p>The H100 can compute ${T_('peak',fPk(g))}, but it can read only ${T_('bw',fBW(g))} from memory. So reading the weights takes ${fTime(b.t)}, while the arithmetic needs only ${fTime(b.tc)}. Of the step’s ${fTime(b.t)}, the math units are busy only ${fSh(b.share)} (the two lanes under the ${DFX('roof','roofline chart')}).</p><p>Which of the two sets a ${DEF('kernel',TIP.kernel())}’s time? And how could this step make fuller use of the GPU?</p>`,
             `<p>你在一张 H100 上用 ${MODEL} 为一个用户生成回答。每生成一个 token，就要跑一个 ${DEF('decode 步','decode 步为 batch 里每个请求生成下一个 token。它把模型的每一层跑一遍，所以要把每个权重从显存读一遍。')}。</p><p>每个 decode 步，GPU 同时做两件事。一件是从${DEF('显存（HBM）',TIP.hbm())}读出全部 ${DEF('15.0 GB',TIP.wread())} 的权重。另一件是让每个权重和这个新 token 做一次乘加，一共 ${fFl(b.fl)}（每个权重 2 个 ${DEF('FLOP',TIP.flop())}）。</p><p>H100 的峰值算力是 ${T_('peak',fPk(g))}，从显存读数据却只有 ${T_('bw',fBW(g))}。所以读权重要 ${fTime(b.t)}，计算只要 ${fTime(b.tc)}。在这一步的 ${fTime(b.t)} 里，计算单元只有 ${fSh(b.share)} 的时间在忙（见 ${DFX('roof','roofline 图')}下面的两条时间线）。</p><p>一个 ${DEF('kernel',TIP.kernel())} 的耗时，由这两件事里的哪一件决定？怎样才能让这一步把 GPU 用得更充分？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},view:'one',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: at batch 64, how long do one decode step’s weight matmuls take?','先猜：batch 为 64 时，一个 decode 步的权重矩阵乘法要多久？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Now 64 users chat at once. One decode step can make one token for each of them at the same time: a ${DEF('batch',TIP.batch())} of 64.</p><p>The step does 64 times the arithmetic of batch 1. Here, too, count only the weight matmuls. Each user also reads their own KV cache; that is another kernel, which part 6 comes back to.</p><p>At batch 1 the weight matmuls take ${fTime(B0().t)}. How long do they take at batch 64?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>现在有 64 个用户同时在聊。一个 decode 步可以一次为他们每人生成一个 token，这就是 ${DEF('batch',TIP.batch())} 为 64。</p><p>这一步的运算量是 batch 为 1 时的 64 倍。这里同样只算权重矩阵乘法。每个用户还要读自己的 KV cache，那是另一个 kernel，第 6 部分会讲到。</p><p>batch 为 1 时，权重矩阵乘法要 ${fTime(B0().t)}。batch 为 64 时要多久？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'batch',title:{en:'Batch the decode step',zh:'给 decode 步加大 batch'},view:'one',
  lead:()=>tx('Drag the batch to 64, then past the ridge, and watch how the dot and the two lanes change.','先把 batch 拖到 64，再拖过拐点，看圆点和两条时间线怎么变。'),
  enter:()=>({...BASE}),expect:()=>seen64,showMe:{B:64},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The dot is this decode step on the ${DFX('roof','roofline')}. Across is its ${DEF('arithmetic intensity',TIP.ai())}, the FLOPs per byte moved from memory. Up is the TFLOP/s it reaches.</p><p>Each extra request multiplies every weight read by one more token. So the step’s FLOPs per byte moved from memory equal the ${T_('B','batch')}.</p>`,
              `<p>圆点就是这个 decode 步在 ${DFX('roof','roofline')} 上的位置。横坐标是它的${DEF('算术强度',TIP.ai())}，也就是每从显存搬运 1 字节做的 FLOP 数。纵坐标是它达到的 TFLOP/s。</p><p>每多一个请求，读进来的每个权重就要多和一个 token 相乘。所以这一步每从显存搬运 1 字节做的 FLOP 数，正好等于 ${T_('B','batch')}。</p>`)+bCtl('rf-ic-B',true),
  obs:()=>obsBatch()},
 {id:'matmul',title:{en:'Grow a matmul past the ridge',zh:'把矩阵乘法加大到拐点右边'},view:'one',
  lead:()=>tx('Drag the square matmul’s n up until the dot reaches the flat roof.','把方阵乘法的 n 往大拖，直到圆点到达平的屋顶。'),
  enter:()=>({...BASE,k:'mm',n:256}),expect:()=>mmPast,showMe:{n:1024},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The bigger the matrices, the more arithmetic each byte moved takes part in.</p><p>A matmul of two ${T_('n',A.ui.math('n \\times n'))} matrices in BF16 reads both and writes the result: ${A.ui.math('3n^2')} numbers of 2 bytes each. It does ${A.ui.math('2n^3')} FLOPs. Dividing the two:</p><div class="fx rf-fxp">${A.ui.math('\\text{FLOPs per byte moved from memory}')}<div class="rf-fxr">${A.ui.math('= \\frac{2n^3}{2 \\times 3n^2}')}</div><div class="rf-fxr">${A.ui.math('= \\frac{n}{3}')}</div></div><p>The ${DEF('ridge',TIP.ridge())} is where this number reaches ${fR(GPUS.h100.peak/GPUS.h100.bw)}.</p>`,
              `<p>矩阵越大，搬进来的每个字节就能参与越多次运算。</p><p>两个 ${T_('n',A.ui.math('n \\times n'))} 矩阵用 BF16 相乘，要读入两个矩阵、写出一个结果，一共 ${A.ui.math('3n^2')} 个数值，每个 2 字节。它要做 ${A.ui.math('2n^3')} 个 FLOP。两者相除：</p><div class="fx rf-fxp">${A.ui.math('\\text{每从显存搬运 1 字节做的 FLOP 数}')}<div class="rf-fxr">${A.ui.math('= \\frac{2n^3}{2 \\times 3n^2}')}</div><div class="rf-fxr">${A.ui.math('= \\frac{n}{3}')}</div></div><p>这个数达到 ${fR(GPUS.h100.peak/GPUS.h100.bw)} 时，就到了${DEF('拐点',TIP.ridge())}。</p>`)+mCtl('rf-ic-n',true),
  obs:()=>obsMatmul()},
 {id:'lever',title:{en:'A faster GPU or FP8 weights?',zh:'换更快的 GPU，还是用 FP8 权重？'},view:'duo',
  lead:()=>tx('One change at a time, on a decode step and a prefill: which one gets faster?','一次只改一处，看 decode 步和 prefill 哪个变快。'),
  enter:()=>{lv='base'; lvTried=new Set(['base']); return {...BASE};},expect:()=>lv!=='base',showMe:()=>setLv('a100'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The chart now has two dots, both on the H100. One is a decode step at batch ${DUO_B}, left of the ridge. The other is the prefill of a ${fint(DUO_N)}-token prompt, right of the ridge.</p><p>Pick a change and see how long the decode step and the prefill take. Each option starts from this setup and changes only the one thing it names.</p>`,
              `<p>图里现在有两个圆点，都在 H100 上。一个是 batch 为 ${DUO_B} 的 decode 步，在拐点左边。另一个是一次 prefill，处理一个 ${fint(DUO_N)} 个 token 的 prompt，在拐点右边。</p><p>选一种改动，看看 decode 步和 prefill 各要多久。每个选项都从现在的设置出发，只改它写明的那一处。</p>`)+lvCtl(),
  obs:()=>obsLever()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:'map',
  lead:()=>tx('FLOPs per byte moved from memory place a kernel; the ridge says which limit sets its time.','每从显存搬运 1 字节做的 FLOP 数决定 kernel 的位置；拐点决定是哪个限制在起作用。'),
  enter:()=>({...BASE}),
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
/* part 3: the result at the learner's batch, the verdict once 64 has been reached, then why (R3.10) */
function obsBatch(){
  const r=R(), b=B0(), B=S.B, rid=r.ridge;
  if(B===1) return tx(`<p><b>At batch 1 each step takes ${fTime(r.t)} and uses ${fSh(r.share)} of the peak FLOP/s.</b> Drag the batch to 64 to check your prediction.</p>`,`<p><b>batch 为 1 时，每步要 ${fTime(r.t)}，用上了峰值算力的 ${fSh(r.share)}。</b>把 batch 拖到 64，核对你的预测。</p>`);
  if(B<64&&!seen64) return tx(`<p><b>At batch ${fint(B)} the step still takes ${fTime(r.t)}. The share of the peak FLOP/s it uses rose from ${fSh(b.share)} to ${fSh(r.share)}.</b> Drag on to 64 to check your prediction.</p>`,`<p><b>batch 为 ${fint(B)} 时，这一步还是要 ${fTime(r.t)}。用上的峰值算力从 ${fSh(b.share)} 升到了 ${fSh(r.share)}。</b>继续拖到 64，核对你的预测。</p>`);
  if(r.bound==='memory') return tx(`<p><b>At batch ${fint(B)} the weight matmuls still take ${fTime(r.t)}, as at batch 1, while doing ${fint(B)} times the work. The share of the peak FLOP/s they use rose from ${fSh(b.share)} to ${fSh(r.share)}.</b></p>`,`<p><b>batch 为 ${fint(B)} 时，权重矩阵乘法还是要 ${fTime(r.t)}，和 batch 为 1 时一样，干的活却是 ${fint(B)} 倍。用上的峰值算力从 ${fSh(b.share)} 升到了 ${fSh(r.share)}。</b></p>`)+verdict()+
    tx(`<p>Why: whatever the batch, the step reads the weights only once. Each request only adds arithmetic on the weights already read. As long as that arithmetic is faster than the read, it hides inside the read’s time. The point where both take as long is the ${DEF('ridge',TIP.ridgePd())}: batch ${fR(rid)} on the H100.${seenPast?'':' Drag past it.'}</p>`,`<p>原因：不管 batch 多大，这一步都只把权重读一遍。每个请求只是在已经读进来的权重上多做一些运算。只要这些运算比读显存快，它们就藏在读显存的时间里。两者一样长的地方就是${DEF('拐点',TIP.ridgePd())}，在 H100 上是 batch 为 ${fR(rid)}。${seenPast?'':'把 batch 拖过它试试。'}</p>`);
  return tx(`<p><b>At batch ${fint(B)} the weight matmuls are compute-bound. They take ${fTime(r.t)}, ${fX(r.t/b.t)} times as long as at batch 1, and use 100% of the peak FLOP/s.</b></p>`,`<p><b>batch 为 ${fint(B)} 时，权重矩阵乘法变成受算力限制。它要 ${fTime(r.t)}，是 batch 为 1 时的 ${fX(r.t/b.t)} 倍，用上了峰值算力的 100%。</b></p>`)+verdict()+
    tx(`<p>Why: past the ridge the arithmetic takes longer than the read. So every extra request adds its own compute time. Batching still pays: all the requests share one step, which is far faster than running them one at a time.</p>`,`<p>原因：过了拐点，运算比读显存还久。所以每多一个请求，就多一份它自己的计算时间。攒 batch 仍然划算：所有请求共用一步，比一个一个地跑快得多。</p>`);
}
/* part 4: the square matmul */
function obsMatmul(){
  const r=R(), n=S.n, nr=3*r.ridge;
  if(r.bound==='memory') return tx(`<p><b>At n = ${fint(n)} the matmul does ${fI(r.I)} FLOPs per byte moved from memory. That is still left of the ridge, so it is memory-bound and uses ${fSh(r.share)} of the peak FLOP/s.</b>${mmPast?'':' Drag n up until the dot reaches the flat roof.'}</p>`,`<p><b>n = ${fint(n)} 时，每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP。这还在拐点左边，所以受显存带宽限制，用上了峰值算力的 ${fSh(r.share)}。</b>${mmPast?'':'把 n 往大拖，直到圆点到达平的屋顶。'}</p>`);
  return tx(`<p><b>At n = ${fint(n)} the matmul does ${fI(r.I)} FLOPs per byte moved from memory. That is right of the ridge, so it is compute-bound and uses 100% of the peak FLOP/s.</b></p><p>Why: the FLOPs grow as ${A.ui.math('n^3')}, the bytes only as ${A.ui.math('n^2')}. So a bigger matmul does more arithmetic with every byte. It crosses the ridge at n ≈ ${fint(Math.round(nr))}, marked on the chart.</p><p>A decode step’s matmuls are different. Each weight matrix meets only one row per request, so each request adds only about 1 FLOP per byte moved from memory. Their dot sits at about the batch size on the x-axis. Prefill multiplies each weight by thousands of prompt tokens, so its matmuls sit far right.</p>`,
            `<p><b>n = ${fint(n)} 时，每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP。这已经在拐点右边，所以受算力限制，用上了峰值算力的 100%。</b></p><p>原因：FLOP 数按 ${A.ui.math('n^3')} 增长，字节数只按 ${A.ui.math('n^2')} 增长。所以矩阵越大，每个字节上做的运算越多。n ≈ ${fint(Math.round(nr))} 时越过拐点，图上标出了这个位置。</p><p>decode 步的矩阵乘法不一样。每个权重矩阵只和每个请求的一行相乘，所以每多一个请求，每从显存搬运 1 字节只多做约 1 个 FLOP。它的圆点停在横轴上约等于 batch 的位置。prefill 让每个权重和几千个 prompt token 相乘，所以它的矩阵乘法在最右边。</p>`);
}
/* part 5: the same two kernels under each change (R3.6: initial setup → now, and why) */
function obsLever(){
  const b=duoOf('base'), d=duoOf(lv);
  if(lv==='base') return tx(`<p><b>Initial setup: on the H100, the decode step at batch ${DUO_B} takes ${fTime(b.dec.t)} and the prefill ${fTime(b.pre.t)}.</b> Pick a change to compare.</p>`,`<p><b>初始设置下，在 H100 上，batch 为 ${DUO_B} 的 decode 步要 ${fTime(b.dec.t)}，prefill 要 ${fTime(b.pre.t)}。</b>选一种改动来比较。</p>`);
  if(lv==='a100'){const a=GPUS.a100, h=GPUS.h100;
    return tx(`<p><b>On the A100 the decode step goes from ${fTime(b.dec.t)} to ${fTime(d.dec.t)}, ${fX(d.dec.t/b.dec.t)} times as long. The prefill goes from ${fTime(b.pre.t)} to ${fTime(d.pre.t)}, ${fX(d.pre.t/b.pre.t)} times as long.</b></p><p>Why: the decode step is left of the ridge, so its time follows the bandwidth. The H100 reads memory ${fX(h.bw/a.bw)} times as fast as the A100. The prefill is right of the ridge, so its time follows the FLOP/s. The H100 computes ${fX(h.peak/a.peak)} times as fast. The A100’s ridge sits at ${fR(a.peak/a.bw)} FLOPs per byte moved from memory (the thin dashed roof is the H100’s, from before).</p>`,
              `<p><b>换成 A100 后，decode 步从 ${fTime(b.dec.t)} 变成 ${fTime(d.dec.t)}，是原来的 ${fX(d.dec.t/b.dec.t)} 倍。prefill 从 ${fTime(b.pre.t)} 变成 ${fTime(d.pre.t)}，是原来的 ${fX(d.pre.t/b.pre.t)} 倍。</b></p><p>原因：decode 步在拐点左边，耗时跟着带宽走。H100 读显存的速度是 A100 的 ${fX(h.bw/a.bw)} 倍。prefill 在拐点右边，耗时跟着算力走。H100 的算力是 A100 的 ${fX(h.peak/a.peak)} 倍。A100 的拐点在每从显存搬运 1 字节做 ${fR(a.peak/a.bw)} 个 FLOP 的位置（细虚线是 H100 原来的屋顶）。</p>`);}
  return tx(`<p><b>With FP8 weights the decode step goes from ${fTime(b.dec.t)} to ${fTime(d.dec.t)}. The prefill stays at ${fTime(d.pre.t)}.</b></p><p>Why: FP8 stores each weight in 1 byte instead of 2. So the decode step reads half the bytes, and its FLOPs per byte moved from memory double. It is left of the ridge, so its time halves too. The prefill was already limited by arithmetic. The ${DEF('FP8 weights',TIP.fp8())} here are still multiplied in BF16, so the same FLOPs take the same time.</p>`,
            `<p><b>用 FP8 权重时，decode 步从 ${fTime(b.dec.t)} 变成 ${fTime(d.dec.t)}。prefill 还是 ${fTime(d.pre.t)}。</b></p><p>原因：FP8 每个权重只占 1 字节，而不是 2 字节。所以 decode 步要读的字节减半，每从显存搬运 1 字节做的 FLOP 数翻倍。它在拐点左边，所以耗时也减半。prefill 本来就受算力限制。这里的 ${DEF('FP8 权重',TIP.fp8())}仍按 BF16 相乘，同样多的 FLOP 还是要同样长的时间。</p>`);
}
/* part 6: four points, each with this lab's example (R6.23), three misconceptions, the way to the scenarios */
function remember(){
  const b=B0(), at=ev({...BASE,k:'att',B:64,c:4096}), kv=kvStep(), da=duoOf('a100'), du=duoOf('base');
  return tx(`<ul class="rf-remlist"><li><b>The FLOPs per byte moved from memory tell where a kernel sits.</b> A decode step for one request does about one FLOP per byte it reads, so it sits far left of the ridge and its math units wait almost all the time. To put them to work, each byte read has to feed more arithmetic.</li>`+
    `<li><b>Left of the ridge, more arithmetic costs no time.</b> So until a decode step reaches the ridge, adding requests to its batch is almost free, because the weights are read once for the whole batch. Past the ridge, each extra request adds its own compute time.</li>`+
    `<li><b>The two sides of the ridge need different fixes.</b> A memory-bound kernel such as a decode step gets faster with fewer bytes to read, for example FP8 weights, or with more bandwidth. A compute-bound kernel such as a prefill gets faster only with more FLOP/s. So find out which side of the ridge a kernel is on before you pick a fix.</li>`+
    `<li><b>Batching shares the weights, not the KV cache.</b> Each user’s attention reads their own KV cache. It does about ${fI(at.I)} FLOPs per byte moved from memory at any batch: the third dot from the left in the chart. See ${labLink('kv-basics')} for where that cache comes from, and ${labLink('gqa-mla')} for how GQA shrinks it.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Decode speeds up in proportion to a GPU’s TFLOP/s.” A decode step follows the bandwidth. The A100 has a third of the H100’s FLOP/s, yet the same decode step takes only ${fX(da.dec.t/du.dec.t)} times as long on it: the ratio of their bandwidths.</p><p>✗ “Low GPU utilization means a bug.” At batch 1, the decode step uses only ${fSh(b.share)} of the H100’s peak FLOP/s, and that is already the hardware’s limit: the step keeps waiting for the weights to arrive from memory. The fix is to serve more tokens with each weight read, that is batching, not a faster kernel.</p><p>✗ “Batching slows each request down in proportion to the batch size.” The weights are read once for the whole batch, so the weight matmuls take as long for many requests as for one. What grows is the KV cache read, because each user reads their own. In the scenario “Attention during decode”, a step for 64 users takes ${DEF(`about ${fixd(kv.x,0)} times as long`,kvTip(kv))} as a step for one user.</p></div>${A.shell.toScenarios()}`,
    `<ul class="rf-remlist"><li><b>看每从显存搬运 1 字节做多少 FLOP，就知道 kernel 在图上的位置。</b>只服务一个请求时，decode 步每读 1 字节只做约 1 个 FLOP，所以它远在拐点左边，计算单元几乎一直在等数据。想用上算力，就要让读进来的每个字节参与更多运算。</li>`+
    `<li><b>在拐点左边，多做的运算不花时间。</b>所以 decode 步到达拐点之前，往 batch 里多加请求几乎不多花时间，因为整个 batch 只读一遍权重。过了拐点，每多一个请求，就多一份计算时间。</li>`+
    `<li><b>拐点两边，要用不同的办法提速。</b>decode 步这类受显存带宽限制的 kernel，要靠少读字节（比如用 FP8 权重）或更高的带宽来提速。prefill 这类受算力限制的 kernel，只有算力更高才会变快。所以先看清 kernel 在拐点哪一边，再选办法。</li>`+
    `<li><b>攒 batch 共享的是权重，不是 KV cache。</b>每个用户的 attention 都要读自己的 KV cache。它每从显存搬运 1 字节约做 ${fI(at.I)} 个 FLOP，batch 多大都一样，就是图里左边第三个圆点。这份 cache 从哪来，见${labLink('kv-basics')}；GQA 怎样把它变小，见 ${labLink('gqa-mla')}。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “decode 的速度和 GPU 的 TFLOP/s 成正比。”decode 步的速度跟着带宽走。A100 的算力只有 H100 的三分之一，同一个 decode 步在 A100 上的耗时却只是 H100 上的 ${fX(da.dec.t/du.dec.t)} 倍，正好是两者的带宽之比。</p><p>✗ “GPU 利用率低，说明有 bug。”batch 为 1 时，decode 步只用上 H100 峰值算力的 ${fSh(b.share)}，这已经是硬件的极限：这一步一直在等权重从显存读进来。办法是让每次读进来的权重服务更多 token，也就是攒 batch，而不是换一个更快的 kernel。</p><p>✗ “攒 batch 会让每个请求按 batch 的大小成比例地变慢。”整个 batch 只读一遍权重，所以权重矩阵乘法服务很多请求和服务一个请求一样快。变长的是读 KV cache，因为每个用户都要读自己的那一份。在应用场景“decode 时的 attention”里，为 64 个用户跑一步，耗时是只服务一个用户时的${DEF(`约 ${fixd(kv.x,0)} 倍`,kvTip(kv))}。</p></div>${A.shell.toScenarios()}`);
}
/* misconception 3: a decode step with each user's KV cache read added (the two kernels this lab prices), 64 users against one,
   4K tokens of context each (the scenario "Attention during decode"), on one H100 at 100% of the peaks */
function kvStep(){const c=4096, w1=ev({...BASE,B:1}), w64=ev({...BASE,B:64}), a1=ev({...BASE,k:'att',B:1,c}), a64=ev({...BASE,k:'att',B:64,c});
  return {c,w1,w64,a1,a64,t1:w1.t+a1.t,t64:w64.t+a64.t,x:(w64.t+a64.t)/(w1.t+a1.t)};}
const kvTip=k=>tx(`This counts the two kernels of a decode step that this lab prices, the weight matmuls and the attention, with ${tokK(k.c)} tokens of context per user on one H100. For 64 users the weights take ${fTime(k.w64.t)} and their 64 KV caches ${fTime(k.a64.t)}: ${fTime(k.t64)} per step. For one user, ${fTime(k.w1.t)} and ${fTime(k.a1.t)}: ${fTime(k.t1)}. Each user waits about ${fixd(k.x,1)} times as long per token, while the step makes 64 times as many tokens.`,
  `这里算的是这个 Lab 给 decode 步计时的两个 kernel，权重矩阵乘法和 attention；每个用户 ${tokK(k.c)} 个 token 的上下文，一张 H100。64 个用户时，权重要 ${fTime(k.w64.t)}，64 份 KV cache 要 ${fTime(k.a64.t)}，一步共 ${fTime(k.t64)}。一个用户时是 ${fTime(k.w1.t)} 加 ${fTime(k.a1.t)}，共 ${fTime(k.t1)}。每个用户每个 token 要等约 ${fixd(k.x,1)} 倍的时间，而这一步产出的 token 是 64 倍。`);

/* =====================================================================
   scenarios
   ===================================================================== */
const presetNow=()=>PRESETS.findIndex(p=>{const t=presetState(p); return KINDS.includes(S.k)&&S.k===t.k&&S.gpu===t.gpu&&S.p===t.p&&(t.k==='dec'?S.B===t.B:t.k==='pre'?S.N===t.N:t.k==='att'?S.B===t.B&&S.c===t.c:t.k==='ew');});
function scenHTML(i){
  const p=PRESETS[i], r=ev(presetState(p)), rid=fR(r.ridge);
  const head=tx(`<p><b>${esc(kfull(presetState(p)))}: ${fTime(r.t)}, using ${fSh(r.share)} of the H100’s peak FLOP/s.</b> The kernel does ${fI(r.I)} FLOPs per byte moved from memory, ${r.bound==='compute'?'right':'left'} of the ridge (${rid}), so it is ${boundTxt(r)}.</p>`,
                `<p><b>${presetState(p).k==='dec'?`${presetState(p).B===1?'只服务 1 个请求时':`同时服务 ${fint(presetState(p).B)} 个请求时`}，${MODEL} 的一个 decode 步里，权重矩阵乘法要 ${fTime(r.t)}，${r.share<0.1?'只':''}用上了 H100 峰值算力的 ${fSh(r.share)}。`:`${esc(kfull(presetState(p)))}，要 ${fTime(r.t)}，用上了 H100 峰值算力的 ${fSh(r.share)}。`}</b>它每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，在拐点（${rid}）${r.bound==='compute'?'右':'左'}边，所以${boundTxt(r)}。</p>`);
  if(p.id==='chat1') return head+tx(`<p>With one request, each weight meets only one token. So the step does 1 FLOP per byte moved from memory and spends most of its time waiting for the weights. This is also why a local model that serves one person runs at the speed of its memory bandwidth, not of its compute.</p>`,
                                     `<p>只有一个请求时，每个权重只和一个 token 相乘。所以这一步每从显存搬运 1 字节只做 1 个 FLOP，大部分时间都在等权重读进来。这也是为什么只给一个人用的本地模型，速度由显存带宽决定，而不是由算力决定。</p>`);
  if(p.id==='busy') return head+tx(`<p>The 256 requests share each weight read, so the step sits just left of the ridge. Its weight matmuls take as long as for one request, yet serve all 256. Each user also reads their own KV cache, which is another kernel (see the scenario “Attention during decode”).</p>`,
                                    `<p>256 个请求共用每一次读权重，所以这一步刚好在拐点左边。它的权重矩阵乘法和只服务一个请求时一样快，却同时服务了 256 个请求。每个用户还要读自己的 KV cache，那是另一个 kernel（见应用场景“decode 时的 attention”）。</p>`);
  if(p.id==='prompt') return head+tx(`<p>Prefill multiplies each weight by every token of the prompt, so it sits far right of the ridge, on the flat roof. Here it is priced at 100% of the peak FLOP/s: ${fTime(r.t)}. The lab ${labLink('prefill-decode')} assumes real kernels reach only 50%, so there this prefill takes twice as long.</p>`,
                                      `<p>prefill 让每个权重和 prompt 里的每个 token 都相乘，所以它在拐点右边很远的地方，贴着平的屋顶。这里按峰值算力的 100% 算，它要 ${fTime(r.t)}。“${labLink('prefill-decode')}”这个 Lab 假设真实的 kernel 只能用上峰值算力的 50%，所以那里这次 prefill 要两倍的时间。</p>`);
  if(p.id==='attn') return head+tx(`<p>Each user’s query heads read that user’s own keys and values once. Every 4 query heads share one KV head (${DEF('GQA',kt('gqa'))}), so the kernel does about 4 FLOPs per byte moved from memory, at any batch.</p><p>The ${DEF('64 caches',`64 users × 4,096 tokens × 128 KiB per token = ${fBy(r.by)} of KV cache, against ${fBy(2*W_PAR)} of weights.`)} hold more than twice the bytes of the weights, so this attention takes longer than the weight matmuls of the same step. That is what makes a bigger batch slower per token.</p>`,
                                    `<p>每个用户的 query head，要把这个用户自己的 key 和 value 读一遍。每 4 个 query head 共用一个 KV head（${DEF('GQA',kt('gqa'))}），所以这个 kernel 每从显存搬运 1 字节约做 4 个 FLOP，batch 多大都一样。</p><p>${DEF('64 份 cache',`64 个用户 × 4,096 个 token × 每个 token 128 KiB = ${fBy(r.by)} 的 KV cache，权重是 ${fBy(2*W_PAR)}。`)} 加起来，字节数是权重的两倍多。所以这部分 attention 比同一步里的权重矩阵乘法还要久。batch 越大，每个 token 越慢，原因就在这里。</p>`);
  return head+tx(`<p>A residual add works number by number. For each single FLOP it reads two numbers and writes one, so it does 1 FLOP per 6 bytes moved from memory. On any GPU it stays memory-bound. So engines fuse such ops into a neighbouring kernel. ${cite(SRC.vllm,'vLLM')}, for example, adds the residual inside its RMSNorm kernel, which then does not read the sum back from memory. ${labLink('flash-attention')} fuses the steps of attention in the same way.</p>`,
                 `<p>残差相加是逐个数值的运算。每读两个数值、写一个数值，它才做 1 个 FLOP，也就是每从显存搬运 6 字节做 1 个 FLOP。换哪张 GPU，它都受显存带宽限制。所以推理引擎会把这类运算融合进相邻的 kernel。比如 ${cite(SRC.vllm,'vLLM')} 在 RMSNorm kernel 里顺便做残差相加，相加的结果就不用再从显存读一遍。${labLink('flash-attention')} 用同样的办法，把 attention 的几步融合在一起。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chSet=(B,p)=>tx(`batch ${fint(B)}, ${p==='fp8'?'FP8':'BF16'} weights`,`batch ${fint(B)}，${p==='fp8'?'FP8':'BF16'} 权重`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.B,S.p), best=chBest(), g=GPUS.h200, rid=g.peak/g.bw;
  let h;
  const good=`<p class="okline rf-mt">`+tx(`<b>Goal met: the decode step’s weight matmuls reach ${fSh(me.share)} of the H200’s peak with ${reqN(S.B)} in the batch.</b>`,`<b>达成目标：batch 里有 ${reqN(S.B)}时，decode 步的权重矩阵乘法用上了 H200 峰值算力的 ${fSh(me.share)}。</b>`);
  if(me.ok&&S.B===best.B&&S.p===best.p) h=good+tx(' And that is the fewest requests that can do it.</p>','而且这已经是能做到的最少请求数。</p>');
  else if(me.ok) h=good+tx(' But fewer requests would do.</p>','不过更少的请求也够。</p>')+tx(`<p>FP8 weights double the FLOPs per byte moved from memory, and with them ${fint(best.B)} requests already use ${fSh(chEval(best.B,best.p).share)}.</p>`,`<p>FP8 权重让每从显存搬运 1 字节做的 FLOP 数翻倍，用它的话，${fint(best.B)} 个请求就能用上 ${fSh(chEval(best.B,best.p).share)}。</p>`);
  else h=tx(`<p class="badline rf-mt"><b>Not yet: the decode step reaches ${fSh(me.share)} of the H200’s peak, not 50%.</b></p>`,`<p class="badline rf-mt"><b>还没达到：decode 步只用上了 H200 峰值算力的 ${fSh(me.share)}，不到 50%。</b></p>`)+
      tx(`<p class="small">Hint: the H200 has the H100’s peak FLOP/s but more bandwidth, so its ${DEF('ridge',`${fixd(g.peak/1e12,1)} ÷ ${fixd(g.bw/1e12,1)} ≈ ${fR(rid)} FLOPs per byte moved from memory.`)} sits further left, at ${fR(rid)}. To use half of the peak FLOP/s, the step needs only half that many FLOPs per byte moved from memory. With BF16 weights, how many does each request add? And with FP8 weights?</p>`,
         `<p class="small">提示：H200 的峰值算力和 H100 一样，带宽却更高，所以它的${DEF('拐点',`${fixd(g.peak/1e12,1)} ÷ ${fixd(g.bw/1e12,1)} ≈ 每从显存搬运 1 字节做 ${fR(rid)} 个 FLOP。`)}更靠左，在 ${fR(rid)}。要用上一半的峰值算力，每从显存搬运 1 字节只需做它一半的 FLOP。用 BF16 权重时，每多一个请求，这个数加几？用 FP8 权重时呢？</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const rows=[]; CH.Ps.forEach(p=>CH.Bs.forEach(B=>rows.push(chEval(B,p))));
  const isStart=e=>e.B===CH.start.B&&e.p===CH.start.p, isMe=e=>e.B===S.B&&e.p===S.p, shown=rows.filter(e=>e.ok||isMe(e)||isStart(e));
  h+=A.ui.cmpChart({title:tx(`Share of the H200’s peak FLOP/s that one decode step of ${MODEL} uses (goal: at least 50%)`,`${MODEL} 的一个 decode 步用上的 H200 峰值算力（目标是至少 50%）`),
    aria:tx('Share of the H200’s peak FLOP/s: ','用上的 H200 峰值算力：')+shown.map(e=>`${chSet(e.B,e.p)}${tx(': ','：')}${fSh(e.share)}`).join(tx('; ','；')),brief:true,
    rows:rows.map(e=>({label:tx(`${e.ok?'✓ ':''}batch ${fint(e.B)}, ${e.p==='fp8'?'FP8':'BF16'}`,`${e.ok?'✓ ':''}batch ${fint(e.B)}，${e.p==='fp8'?'FP8':'BF16'}`),value:e.share*100,text:fSh(e.share),ok:e.ok,me:e.B===S.B&&e.p===S.p,
      tags:(e.B===CH.start.B&&e.p===CH.start.p?[{text:tx('initial setup','初始设置')}]:[]).concat(e.B===S.B&&e.p===S.p?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:50,goalLabel:tx('goal: 50%','目标：50%'),max:118,okText:tx('✓ meets the goal','✓ 达到目标'),noText:tx('misses it','没达到'),goalText:tx('goal: 50%','目标：50%')});
  const st=chEval(CH.start.B,CH.start.p);
  h+=tx(`<p class="small">The initial setup does ${fint(Math.round(st.r.I))} FLOP per byte moved from memory, far left of the H200’s ridge at ${fR(rid)}.</p><p class="small">FP8 weights read half the bytes, so they double the FLOPs per byte moved from memory of any batch. With them the goal takes batch ${fint(best.B)}; with BF16 weights it takes batch ${fint(CH.Bs.find(B=>chEval(B,'bf16').ok))}.</p>`,
        `<p class="small">初始设置每从显存搬运 1 字节只做 ${fint(Math.round(st.r.I))} 个 FLOP，远在 H200 的拐点 ${fR(rid)} 左边。</p><p class="small">FP8 权重读的字节减半，所以不管 batch 多大，每从显存搬运 1 字节做的 FLOP 数都翻倍。用它的话，batch 为 ${fint(best.B)} 就能达标；用 BF16 权重要 batch 为 ${fint(CH.Bs.find(B=>chEval(B,'bf16').ok))}。</p>`);
  h+=tx('<p class="small muted">Note: this counts the weight matmuls only. 64 users also read their own KV caches, at about 4 FLOPs per byte moved from memory whatever the batch. So the whole step sits further left than the dot (see the scenario “Attention during decode”). And before deploying FP8 weights, check on an evaluation set that answer quality does not drop.</p>',
        '<p class="small muted">注意：这里只算了权重的矩阵乘法。64 个用户还要读各自的 KV cache，它每从显存搬运 1 字节只做约 4 个 FLOP，batch 多大都一样。所以整个 decode 步其实比圆点更靠左（见应用场景“decode 时的 attention”）。另外，部署 FP8 权重之前，要先在评测集上确认回答质量没有下降。</p>');
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===6?'Six':fint(STEPS.length)} short parts. Predict first, then batch a decode step up to the ridge, grow a matmul past it, and try a faster GPU against FP8 weights. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再把 decode 步的 batch 加到拐点，把矩阵乘法加大到拐点右边，最后比较换更快的 GPU 和用 FP8 权重。第一次学从这里开始。`),
  sandbox:tx('Five kernels from serving an LLM, from a chat with one user to a residual add. Pick one to see where it sits on the roofline and what limits it.',
             '服务一个大模型时的五种 kernel，从只和一个用户聊天到一次残差相加。选一个，看它在 roofline 上的位置，以及是什么限制了它。'),
  challenge:tx(`One H200 serves ${MODEL}. Choose the batch and the weight precision so that the weight matmuls of one decode step use at least 50% of the GPU’s peak compute, with as few requests in the batch as possible.`,
               `一张 H200 跑 ${MODEL}。选好 batch 和权重精度，让一个 decode 步的权重矩阵乘法至少用上这张 GPU 峰值算力的 50%，同时 batch 里的请求尽量少。`)
});
const SRC={
 roof:'https://dl.acm.org/doi/10.1145/1498765.1498785',
 book:'https://jax-ml.github.io/scaling-book/roofline/',
 ncu:'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html',
 pope:'https://arxiv.org/abs/2211.05102',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 a100:'https://www.nvidia.com/en-us/data-center/a100/',
 h200:'https://www.nvidia.com/en-us/data-center/h200/',
 cfg:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 vllm:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py'
};
const FIDPOP=()=>{const h=GPUS.h100, pd60=2*W_PAR/(h.bw*0.6), pd50=ev({...BASE,k:'pre',N:2048}).fl/(h.peak*0.5), rd=(h.peak*0.5)/(h.bw*0.6);
 return tx(
 `<p><b>Formula:</b> every number on this page comes from one model of a kernel’s time; ${J('math','“Show the math”')} works it out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>A kernel takes the longer of its compute time and its memory time, because the GPU reads and computes at the same time:<div class="fx">${A.ui.math(FX_T())}</div>The FLOP/s it reaches are its FLOPs divided by that time.</li>`+
 `<li>The peaks are datasheet numbers, dense BF16: A100 80GB SXM 312 TFLOP/s and 2,039 GB/s; H100 SXM 989.5 TFLOP/s (the datasheet’s 1,979 “with sparsity” ÷ 2) and 3.35 TB/s; H200 SXM the same 989.5 TFLOP/s and 4.8 TB/s.</li>`+
 `<li>${MODEL}’s decode step is its weight matmuls: ${A.fmt.big(W_PAR)} parameters (all but the embedding table, which is looked up one row per token), 2 bytes each in BF16, 2 FLOPs per parameter per token. Prefill is counted as in ${labLink('prefill-decode')}: the weight matmuls for every prompt token, the attention over the prompt, and the KV cache it writes. Attention during decode reads each user’s KV cache, 128 KiB per token.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>100% of the peaks: the roofline is the bound no kernel beats, and real kernels reach less. ${labLink('prefill-decode')} prices the same steps at 50% of the peak FLOP/s and 60% of the peak bandwidth, so its ridge sits at about ${fR(rd)} and its times are longer: ${fTime(pd50)} for the 2,048-token prefill (${fTime(pd50/2)} here) and ${fTime(pd60)} for a decode step’s weights (${fTime(2*W_PAR/h.bw)} here). The weight matmuls of a decode step at batch 256 sit between the two ridges, so they are memory-bound here and compute-bound there. ${labLink('prefill-decode')} also counts every user’s KV cache read in its decode step, which keeps that step memory-bound.</li>`+
 `<li>FP8 weights take 1 byte per parameter and are multiplied in BF16 (weight-only quantization), so the compute roof stays at the BF16 peak. With FP8 arithmetic too (the H100 does 1,979 TFLOP/s dense in FP8) the compute roof doubles as well.</li>`+
 `<li>Left out of the decode step: the activations each matmul reads and writes (about 9% more bytes when 256 requests share a step, so about 9% fewer FLOPs per byte moved from memory), the norms and other elementwise ops, which are kernels of their own, and the KV cache, which is the kernel “attention during decode”. Kernel launch overheads and on-chip caches are not modeled.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>The roofline: ${cite(SRC.roof,'Williams, Waterman and Patterson, CACM 2009')}; ${cite(SRC.book,'How to Scale Your Model, “All About Rooflines”')}. Fusing the residual add into RMSNorm: ${cite(SRC.vllm,'vLLM’s layernorm.py')}.</li>`+
 `<li>GPUs: NVIDIA’s ${cite(SRC.a100,'A100')}, ${cite(SRC.h100,'H100')} and ${cite(SRC.h200,'H200')} datasheets. Model shapes: ${cite(SRC.cfg,'Llama 3.1 8B config.json')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上的每个数，都来自同一个计算 kernel 耗时的模型；${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>GPU 一边读一边算，所以一个 kernel 的耗时取计算时间和读写显存时间中较长的那个：<div class="fx">${A.ui.math(FX_T())}</div>它达到的 FLOP/s，就是 FLOP 数除以这个耗时。</li>`+
 `<li>峰值来自厂商的规格表，按稠密 BF16：A100 80GB SXM 312 TFLOP/s、2,039 GB/s；H100 SXM 989.5 TFLOP/s（规格表上“with sparsity”的 1,979 ÷ 2）、3.35 TB/s；H200 SXM 同样是 989.5 TFLOP/s，带宽 4.8 TB/s。</li>`+
 `<li>${MODEL} 的 decode 步就是它的权重矩阵乘法：${A.fmt.big(W_PAR)} 个参数（除 embedding 表以外的全部，embedding 表每个 token 只查一行），BF16 每个 2 字节，每个参数每个 token 2 个 FLOP。prefill 的算法和“${labLink('prefill-decode')}”这个 Lab 一样：每个 prompt token 的权重矩阵乘法、对 prompt 的 attention，以及它写入的 KV cache。decode 时的 attention 读每个用户的 KV cache，每个 token 128 KiB。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>按峰值的 100% 算：roofline 是任何 kernel 都超不过的上限，真实的 kernel 达不到。“${labLink('prefill-decode')}”这个 Lab 按峰值算力的 50%、峰值带宽的 60% 计时，所以那里的拐点约是 ${fR(rd)}，耗时也更长：2,048 个 token 的 prefill 是 ${fTime(pd50)}（这里 ${fTime(pd50/2)}），decode 步读权重是 ${fTime(pd60)}（这里 ${fTime(2*W_PAR/h.bw)}）。batch 为 256 的 decode 步，它的权重矩阵乘法落在两个拐点之间，所以在这里受显存带宽限制，在那里受算力限制。“${labLink('prefill-decode')}”这个 Lab 的 decode 步还要读每个用户的 KV cache，所以那一步在那里仍受显存带宽限制。</li>`+
 `<li>FP8 权重每个参数 1 字节，按 BF16 相乘（只量化权重），所以算力的屋顶仍是 BF16 的峰值。如果运算也用 FP8（H100 的稠密 FP8 是 1,979 TFLOP/s），算力的屋顶也会翻倍。</li>`+
 `<li>decode 步里没算的：每次矩阵乘法读写的激活值（256 个请求共用一步时约多 9% 的字节，每从显存搬运 1 字节做的 FLOP 数也就约少 9%）；RMSNorm 和其他逐个数值的运算，它们是单独的 kernel；KV cache，它是“decode 时的 attention”这个 kernel。kernel 启动开销和片上缓存都没有建模。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>roofline：${cite(SRC.roof,'Williams、Waterman 和 Patterson，CACM 2009')}；${cite(SRC.book,'How to Scale Your Model 的“All About Rooflines”一章')}。把残差相加融合进 RMSNorm：${cite(SRC.vllm,'vLLM 的 layernorm.py')}。</li>`+
 `<li>GPU：NVIDIA 的 ${cite(SRC.a100,'A100')}、${cite(SRC.h100,'H100')} 和 ${cite(SRC.h200,'H200')} 规格表。模型形状：${cite(SRC.cfg,'Llama 3.1 8B 的 config.json')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);};
const PRIPOP=()=>tx(`<p>An LLM answers in two phases. Prefill runs the whole prompt through the model in one step. Decode then generates one token per step, and every decode step reads all of the model’s weights from GPU memory. ${labLink('prefill-decode')} shows both phases with times.</p><p>This lab gives the general tool behind them: count a kernel’s FLOPs and the bytes it moves, and the roofline tells you which of the two sets its time.</p><p>This lab’s decode step is the weight matmuls alone, at 100% of the datasheet peaks, the bound no kernel beats. ${labLink('prefill-decode')} also reads each user’s KV cache and assumes kernels reach 50% of the peak FLOP/s and 60% of the bandwidth, so its steps take longer.</p>`,
                    `<p>大模型分两个阶段回答。prefill 在一步里把整个 prompt 送进模型。之后 decode 每一步生成一个 token，每个 decode 步都要把模型的全部权重从显存读一遍。“${labLink('prefill-decode')}”这个 Lab 用耗时讲了这两个阶段。</p><p>这个 Lab 讲它背后的通用工具：数出一个 kernel 的 FLOP 数和它搬运的字节数，roofline 就告诉你是哪一个决定了它的耗时。</p><p>这里的 decode 步只算权重的矩阵乘法，按规格表峰值的 100% 算，是任何 kernel 都超不过的上限。“${labLink('prefill-decode')}”这个 Lab 还读每个用户的 KV cache，并假设 kernel 只达到峰值算力的 50%、带宽的 60%，所以那里每一步更久。</p>`);
const OUTCOME=()=>tx('place a kernel on the roofline from its FLOPs and bytes, and say whether a bigger batch, FP8 weights or a faster GPU will speed it up.','根据 FLOP 数和读写的字节数，把一个 kernel 放到 roofline 上，并判断加大 batch、FP8 权重或换更快的 GPU 能不能让它变快。');
/* the new cases use numbers the page shows (R3.19): batch 32 on the H100 is a stop of part 3's slider */
const X32=()=>ev({...BASE,B:32});
const XMM={m:64,k:4096,n:4096};
const xmmI=()=>{const {m,k,n}=XMM; return 2*m*k*n/(2*(m*k+k*n+m*n));};
const EXITS=()=>{
  const b=B0(), h=GPUS.h100, x32=X32(), xi=xmmI();
  return {
 core:()=>chSubmitted, explain:[
  {q:tx(`At batch 1, a decode step of ${MODEL} on an H100 reaches ${fSh(b.share)} of the peak FLOP/s. Why so little?`,`batch 为 1 时，${MODEL} 的一个 decode 步在 H100 上只用上了峰值算力的 ${fSh(b.share)}。为什么这么少？`),
   o:[tx('Each weight read from memory is used for only one multiply-add with the one token, while the H100 could do about 295 FLOPs in the time one byte arrives. So the step spends almost all its time waiting for the weights.','从显存读进来的每个权重只和这一个 token 做一次乘加；而在 1 字节读进来的时间里，H100 能做约 295 个 FLOP。所以这一步几乎全部时间都在等权重。'),
      tx('An 8B model’s matrices are too narrow to fill the H100’s thousands of cores, so most cores have no work. A wider model would fill them and reach a higher share of the peak FLOP/s at batch 1.','8B 模型的矩阵太窄，填不满 H100 的几千个核心，大部分核心没活干。换一个更宽的模型就能填满，batch 为 1 时也能用上更多的峰值算力。'),
      tx('Each decode step must wait for the previous token, so the GPU sits idle between steps; the share is low because of that idle time, not because of anything inside the step.','每个 decode 步都要等上一个 token，GPU 在两步之间是空闲的；用上的算力少，是因为这段空闲时间，而不是因为步里的什么事。')],a:0,
   n:[null,tx(`Width does not help: a wider model has more weights to read too, so the FLOPs per byte moved from memory stay at one per request. A model twice as large would reach the same ${fSh(b.share)} at batch 1, in twice the time.`,`宽度没用：更宽的模型要读的权重也更多，所以每个请求每从显存搬运 1 字节还是只做一个 FLOP。大一倍的模型在 batch 为 1 时也只用上同样的 ${fSh(b.share)}，耗时还翻倍。`),
     tx(`The share is measured inside one step: in the lab’s own step, the math units are busy ${fSh(b.share)} of the ${fTime(b.t)} the step takes. Waiting between steps would lower it further, but it is not the cause.`,`这个比例是在一步之内量的。在这个 Lab 的一步里，${fTime(b.t)} 中只有 ${fSh(b.share)} 的时间计算单元在忙。两步之间的等待会让它更低，但不是原因。`)],
   why:tx(`The step does only 1 FLOP per byte moved from memory, while the H100 can do about ${DEF(`${fR(h.peak/h.bw)} FLOPs`,`${fixd(h.peak/1e12,1)} TFLOP/s ÷ ${fixd(h.bw/1e12,2)} TB/s ≈ ${fR(h.peak/h.bw)}: the H100’s ridge.`)} in the time it reads 1 byte. The math units spend most of the time waiting for data, so the memory read sets the time.`,`这一步每从显存搬运 1 字节只做 1 个 FLOP，而 H100 在读 1 字节的时间里能做约 ${DEF(`${fR(h.peak/h.bw)} 个 FLOP`,`${fixd(h.peak/1e12,1)} TFLOP/s ÷ ${fixd(h.bw/1e12,2)} TB/s ≈ ${fR(h.peak/h.bw)}，这就是 H100 的拐点。`)}。计算单元大部分时间都在等数据，所以耗时由读显存决定。`)},
  {q:tx('Why does a bigger batch barely change the decode step’s time, until the batch reaches the ridge?','为什么在 batch 达到拐点之前，加大 batch 几乎不改变 decode 步的耗时？'),
   o:[tx('The step reads the weights once whatever the batch, and each request only adds arithmetic on the weights already read; until that arithmetic takes as long as the read, it hides behind it.','不管 batch 多大，这一步都只读一遍权重，每个请求只是在已经读进来的权重上多做运算；在这些运算和读显存一样长之前，它们都藏在读显存的时间里。'),
      tx('The GPU keeps the weights in its on-chip cache after the first request, so the other requests in the batch read them for free.','GPU 在第一个请求之后把权重留在片上缓存里，所以 batch 里其他请求读权重不花时间。'),
      tx('The GPU runs the requests of a batch on separate groups of cores, each with its own share of the memory bandwidth, so they finish together.','GPU 把 batch 里的请求放在不同的几组核心上跑，每组分到自己的一份显存带宽，所以它们同时结束。')],a:0,
   n:[null,tx('15.0 GB of weights do not fit in a GPU’s on-chip cache, which holds tens of MB. The batch shares one read because every tile of weights, as it streams in, is multiplied by all the requests’ tokens before the next tile arrives.','15.0 GB 的权重放不进只有几十 MB 的片上缓存。batch 能共用一次读取，是因为每一块权重读进来时，先和所有请求的 token 都乘完，下一块才进来。'),
     tx('Splitting the bandwidth among the requests would make each request read the weights at a fraction of the speed, so a bigger batch would take longer. The requests share one read of the weights instead.','如果把带宽分给各个请求，每个请求读权重的速度就只有几分之一，batch 越大就越慢。实际上各个请求共用同一次读权重。')],
   why:tx(`The time is the longer of computing and reading memory:<span class="rf-fxl">${A.ui.math(FX_T())}</span>Only the compute time grows with the batch. On the H100 the two are equal at batch 295, the ridge. Past it, each extra request adds its own compute time.`,`耗时取计算和读显存中较长的那个：<span class="rf-fxl">${A.ui.math(FX_T())}</span>其中只有计算时间随 batch 增长。在 H100 上，两者在 batch 为 295 时相等，这就是拐点。过了拐点，每多一个请求，就多一份计算时间。`)}],
 transfer:[
  {q:tx(`At batch 32, the weight matmuls of ${MODEL}’s decode step on an H100 do about ${fSh(x32.share)} of the FLOPs per second the H100 can do at its peak. A model with twice as many parameters, also in BF16, runs at the same batch on the same H100. About what share of the peak FLOP/s do its weight matmuls reach?`,`batch 为 32 时，${MODEL} 的 decode 步在 H100 上做权重矩阵乘法，每秒完成的 FLOP 约是 H100 峰值算力的 ${fSh(x32.share)}。换一个参数量多一倍的模型，同样用 BF16，在同一张 H100 上以同样的 batch 运行。它的权重矩阵乘法每秒完成的 FLOP，大约是峰值算力的多少？`),
   o:[tx(`Still about ${fSh(x32.share)}, but each step takes about twice as long`,`还是约 ${fSh(x32.share)}，但每步耗时约翻倍`),tx(`About ${fSh(2*x32.share)}, because the arithmetic doubles`,`约 ${fSh(2*x32.share)}，因为运算量翻倍了`),tx(`About ${fSh(x32.share/2)}, because the bytes to read double`,`约 ${fSh(x32.share/2)}，因为要读的字节翻倍了`)],a:0,
   why:tx(`Twice the parameters means twice the FLOPs and twice the bytes, so the FLOPs per byte moved from memory stay at the batch size, and the share of the peak FLOP/s stays at ${DEF(`about ${fSh(x32.share)}`,`32 FLOPs per byte moved from memory ÷ the ridge’s ${fR(h.peak/h.bw)} ≈ ${fSh(x32.share)}.`)}. The step still waits for the read, now of twice the bytes, so it takes about twice as long.`,`参数翻倍，FLOP 数和字节数都翻倍，所以每从显存搬运 1 字节做的 FLOP 数还是等于 batch 大小，用上的峰值算力也还是${DEF(`约 ${fSh(x32.share)}`,`32 ÷ 拐点 ${fR(h.peak/h.bw)} ≈ ${fSh(x32.share)}。`)}。这一步仍然在等读显存，只是读的字节翻倍，所以耗时约翻倍。`)},
  {q:tx(`A square matmul of two ${A.ui.math('n \\times n')} matrices in BF16 does about ${A.ui.math('\\frac{n}{3}')} FLOPs per byte moved from memory. One layer’s Wq at batch 64 is a (64 × 4,096) times (4,096 × 4,096) matmul in BF16. About how many FLOPs per byte moved from memory, and which side of the H100’s ridge (${fR(h.peak/h.bw)})?`,`两个 ${A.ui.math('n \\times n')} 的 BF16 方阵相乘，每从显存搬运 1 字节约做 ${A.ui.math('\\frac{n}{3}')} 个 FLOP。一层的 Wq 在 batch 为 64 时，是一个 (64 × 4,096) 乘 (4,096 × 4,096) 的 BF16 矩阵乘法。它每从显存搬运 1 字节大约做多少个 FLOP？在 H100 拐点（${fR(h.peak/h.bw)}）的哪一边？`),
   o:[tx('About 62: memory-bound','约 62：受显存带宽限制'),tx('About 1,365, like n = 4,096: compute-bound','约 1,365，和 n = 4,096 一样：受算力限制'),tx('About 21, like n = 64: memory-bound','约 21，和 n = 64 一样：受显存带宽限制')],a:0,
   why:tx(`The bytes are mostly the weight matrix, and each of its numbers is used for one multiply-add per row of the input, so every row of the input adds about one FLOP per byte moved from memory: ${DEF(`about ${fint(Math.round(xi))} in all`,`2 × 64 × 4,096 × 4,096 FLOPs ÷ (2 × (64 × 4,096 + 4,096 × 4,096 + 64 × 4,096)) bytes = ${fixd(xi,1)} FLOPs per byte moved from memory.`)}. That is left of the ridge, so the matmul is memory-bound, like the decode step it belongs to.`,
          `字节数主要是那个权重矩阵，它的每个数值对输入的每一行做一次乘加，所以输入每多一行，每从显存搬运 1 字节就多做约 1 个 FLOP，加起来${DEF(`约 ${fint(Math.round(xi))} 个`,`2 × 64 × 4,096 × 4,096 个 FLOP ÷ (2 × (64 × 4,096 + 4,096 × 4,096 + 64 × 4,096)) 字节 = 每从显存搬运 1 字节做 ${fixd(xi,1)} 个 FLOP。`)}。这在拐点左边，所以它受显存带宽限制，和它所属的 decode 步一样。`)}],
 practice:[
  {id:'matmul-roofline',d:tx('Part 1: for four matmul shapes on a 312 TFLOP/s, 1.5 TB/s accelerator, give the FLOPs, the bytes, the FLOPs per byte moved from memory, the bound and the time; then show that growing m, k or n alone can only move a matmul toward compute-bound.','第 1 部分：在 312 TFLOP/s、1.5 TB/s 的加速器上，给四种形状的矩阵乘法算出 FLOP 数、字节数、每从显存搬运 1 字节做的 FLOP 数、受什么限制和耗时；再说明单独增大 m、k 或 n 只会让矩阵乘法往受算力限制的方向走。')},
  {id:'inference-batching',d:tx('The replica’s run_batch takes a fixed 100 ms whether 1 or 64 of its slots are filled, which is this lab’s decode step left of the ridge, where its time does not change with the batch. Size the replica pool and the batching policy from it.','题目里 run_batch 不管填了 1 个还是 64 个位置都固定花 100 ms，正是这个 Lab 里拐点左边、decode 步耗时不随 batch 变的那一段。据此估算副本数量并设计攒 batch 的策略。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('GPU & roofline','GPU 与 roofline'),title:l?L(l.q):tx('Is your kernel limited by compute or by memory?','你的 kernel 卡在算力还是显存带宽？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: prefill and decode','前置知识：prefill 和 decode'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage rf-stage" id="rf-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="rf-st-title"></div><div class="meter-sub" id="rf-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="rf-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph rf-graph" id="rf-graph"><div class="rf-phone" id="rf-phone"></div><div class="rf-plot" id="rf-plot"></div><div class="rf-time" id="rf-time"></div></div>`+
   `<p class="rf-scope small muted" id="rf-scope" tabindex="0" data-tipx="${esc(TIP.overlap())}"></p>`+
   `<div class="metrics rf-metrics" id="rf-metrics"></div>`+
   `<details class="math" id="rf-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="rf-mathbody"></div></details></div>`;
}
const opt=(v,t)=>[v,t];
function controlsHTML(){
  return `<aside class="panel controls" id="rf-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="rf-lock" hidden></p>`+
   `<div class="lockable" id="rf-ctl-k-wrap">`+A.ui.seg({id:'rf-ctl-k',key:'k',label:tx('Kernel','kernel'),options:[opt('dec',tx('Decode step','decode 步')),opt('pre','Prefill'),opt('att',tx('Attention in decode','decode 时的 attention')),opt('mm',tx('Square matmul','方阵乘法')),opt('ew',tx('Residual add','残差相加'))],value:S.k,note:tx(`The kernel on the chart. All but the matmul and the add are ${MODEL}’s.`,`图上的 kernel。除了矩阵乘法和残差相加，其余都是 ${MODEL} 的。`)})+`</div>`+
   `<div class="lockable" id="rf-ctl-B-wrap">`+bCtl('rf-ctl-B',false,tx('Requests served by one decode step. Each adds one token per weight read.','一个 decode 步同时服务的请求数。每个请求让每次读权重多算一个 token。'))+`</div>`+
   `<div class="lockable" id="rf-ctl-N-wrap">`+A.ui.range({id:'rf-ctl-N',key:'N',label:tx('Prompt length','prompt 长度'),min:0,max:NST.length-1,value:Math.max(0,NST.indexOf(S.N)),out:fint(S.N),note:tx('Tokens the prefill step processes at once.','prefill 这一步一次处理的 token 数。')})+`</div>`+
   `<div class="lockable" id="rf-ctl-c-wrap">`+A.ui.range({id:'rf-ctl-c',key:'c',label:tx('Context per user','每个用户的上下文'),min:0,max:CST.length-1,value:Math.max(0,CST.indexOf(S.c)),out:fint(S.c),note:tx('Tokens in each user’s KV cache, all read once per step.','每个用户 KV cache 里的 token 数，每一步都要全部读一遍。')})+`</div>`+
   `<div class="lockable" id="rf-ctl-n-wrap">`+mCtl('rf-ctl-n',false).replace(/<\/div>$/,`<div class="ctl-note">${tx(`Both matrices are ${A.ui.math('n \\times n')}, in BF16.`,`两个矩阵都是 ${A.ui.math('n \\times n')}，BF16。`)}</div></div>`)+`</div>`+
   `<div class="lockable" id="rf-ctl-gpu-wrap">`+A.ui.seg({id:'rf-ctl-gpu',key:'gpu',label:'GPU',options:[opt('a100','A100'),opt('h100','H100')],value:S.gpu,note:tx('A100 80GB SXM: 312 TFLOP/s, 2.04 TB/s. H100 SXM: 989.5 TFLOP/s, 3.35 TB/s.','A100 80GB SXM：312 TFLOP/s，2.04 TB/s。H100 SXM：989.5 TFLOP/s，3.35 TB/s。')})+`</div>`+
   `<div class="lockable" id="rf-ctl-p-wrap">`+A.ui.seg({id:'rf-ctl-p',key:'p',label:tx('Weight precision','权重精度'),options:[opt('bf16','BF16'),opt('fp8','FP8')],value:S.p,note:tx('Bytes per weight: 2 or 1. FP8 weights are multiplied in BF16 here, so the peak stays the same.','每个权重的字节数：2 或 1。这里 FP8 权重仍按 BF16 相乘，所以峰值不变。')})+`<p class="ctl-note rf-pnote" id="rf-pnote" hidden>${tx('This kernel reads no weights, so the weight precision does not change it.','这个 kernel 不读权重，所以权重精度不影响它。')}</p></div>`+
   `<div class="rf-grp lockable" id="rf-ctl-asm"><div class="rf-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('100% of the peaks','按峰值的 100%')}</span><div class="ctl-note">${tx('The peak FLOP/s and bandwidth are the datasheet’s, not measured: the roofline is the best case. Real kernels reach less, depending on their shapes and how well they are tuned, so a kernel close to the ridge can land on its other side. Prefill vs decode, for example, assumes 50% of the peak FLOP/s and 60% of the bandwidth.','峰值算力和带宽来自规格表，不是实测的：roofline 是最好情况。真实的 kernel 达不到，具体多少取决于形状和调优程度，所以离拐点很近的 kernel 可能落到另一边。比如“Prefill 和 decode”这个 Lab 假设只达到峰值算力的 50%、带宽的 60%。')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Reading and computing overlap','读显存和计算同时进行')}</span><div class="ctl-note">${TIP.overlap()}</div></div></div>`+
   `<div class="lockable" id="rf-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`# The roofline: a kernel takes the longer of its compute and its
# memory traffic, because the GPU reads and computes at once.
GPUS = {"A100": (312e12, 2.039e12),   # peak FLOP/s, bytes/s
        "H100": (989.5e12, 3.35e12)}

def roofline(flops, nbytes, gpu="H100"):
    peak, bw = GPUS[gpu]
    intensity = flops / nbytes              # FLOPs per byte moved
    t = max(flops / peak, nbytes / bw)      # seconds
    reached = flops / t                     # = min(peak, I * bw)
    bound = "compute" if intensity >= peak / bw else "memory"
    return intensity, reached, t, bound

def matmul(m, k, n, b=2):                   # (m x k) @ (k x n)
    return 2 * m * k * n, b * (m * k + k * n + m * n)

W = 7_504_924_672        # Llama-3.1-8B parameters a decode step reads
def decode(batch, wbytes=2):                # weight matmuls only
    return 2 * W * batch, wbytes * W

for name, (f, b) in [("matmul m=1", matmul(1, 4096, 4096)),
                     ("matmul m=4096", matmul(4096, 4096, 4096)),
                     ("decode B=1", decode(1))]:
    i, reached, t, bound = roofline(f, b)
    print(f"{name:13} I={i:7.1f} {reached / 1e12:6.1f} TFLOP/s"
          f" {t * 1e6:8.1f} us {bound}")
# matmul m=1    I=    1.0    3.3 TFLOP/s     10.0 us memory
# matmul m=4096 I= 1365.3  989.5 TFLOP/s    138.9 us compute
# decode B=1    I=    1.0    3.4 TFLOP/s   4480.6 us memory`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="rf-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="rf-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s model: intensity, FLOP/s reached, time and bound, for two matmuls and a decode step. Change the batch in decode() to see the rest.','这个 Lab 的模型：两个矩阵乘法和一个 decode 步的算术强度、达到的 FLOP/s、耗时和受什么限制。改 decode() 里的 batch 看其他情况。')}</p><pre class="code" id="rf-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="rf-prod">`+
   tx(`<li>NVIDIA Nsight Compute draws this chart for a real kernel, from its measured FLOPs and bytes and the GPU’s own roofs; its ${cite(SRC.ncu,'Profiling Guide')} explains the roofline charts.</li>`+
      `<li>${cite(SRC.book,'How to Scale Your Model, chapter “All About Rooflines”')} works the same model for matmuls, with communication between chips as a third roof.</li>`+
      `<li>Serving engines act on it: they batch many requests into one decode step, store weights in FP8, and fuse elementwise ops into a neighbouring kernel, the three fixes this lab discusses.</li>`,
      `<li>NVIDIA Nsight Compute 会为真实的 kernel 画这张图，用的是它实测的 FLOP 数和字节数，以及这张 GPU 自己的屋顶；它的 ${cite(SRC.ncu,'Profiling Guide')} 讲了这些 roofline 图。</li>`+
      `<li>${cite(SRC.book,'How to Scale Your Model 的“All About Rooflines”一章')}用同一个模型分析矩阵乘法，并把芯片之间的通信当作第三个屋顶。</li>`+
      `<li>推理引擎就是照着它做的：把很多请求攒进同一个 decode 步，用 FP8 存权重，把逐个数值的运算融合进相邻的 kernel，正是这个 Lab 讨论的三种办法。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.roof,'Roofline: An Insightful Visual Performance Model for Multicore Architectures')}<span>${tx('Williams, Waterman and Patterson, CACM 2009: attainable FLOP/s against operational intensity, the ridge point, and ceilings for each optimization','Williams、Waterman、Patterson，CACM 2009：能达到的 FLOP/s 和 operational intensity 的关系、拐点，以及每种优化对应的天花板')}</span></li>`+
   `<li>${cite(SRC.pope,'Efficiently Scaling Transformer Inference')}<span>${tx('Pope et al., MLSys 2023: decode at small batch is limited by loading the weights, and a larger batch raises the FLOPs per byte moved from memory until compute takes over','Pope 等，MLSys 2023：小 batch 的 decode 受读权重限制，加大 batch 会提高每从显存搬运 1 字节做的 FLOP 数，直到计算成为瓶颈')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="rf-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="rf-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="rf-app m2" id="rf-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="rf-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much of the H100’s peak compute its kernel uses. Each one runs on one H100 with BF16 weights.','选一个场景，看看这个 kernel 能用上多少 H100 峰值算力。每个场景都在一张 H100 上，用 BF16 权重。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="rf-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const g=GPUS.h200;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('One decode step at 50% of an H200’s peak compute, with as few requests as possible','用尽量少的请求，让一个 decode 步用上 H200 峰值算力的 50%')}</h3>`+
   `<p>${tx(`You run ${MODEL} on one H200. Its peak is ${fPk(g)}, and it reads memory at ${fixd(g.bw/1e12,1)} TB/s.`,
            `你用一张 H200 跑 ${MODEL}。它的峰值算力是 ${fPk(g)}，读显存的带宽是 ${fixd(g.bw/1e12,1)} TB/s。`)}</p>`+
   `<p>${tx('The goal: the weight matmuls of one decode step use at least 50% of the GPU’s peak FLOP/s, with as few requests in the batch as possible. Pick the batch and the weight precision:',
            '目标是让一个 decode 步的权重矩阵乘法，至少用上这张 GPU 峰值算力的 50%。同时，batch 里的请求越少越好。请选好 batch 和权重精度：')}</p>`+
   A.ui.seg({id:'rf-ic-chB',key:'chB',label:tx('Batch: requests per step','batch：每步的请求数'),options:CH.Bs.map(B=>[String(B),fint(B)]),value:String(S.B),inline:true})+
   A.ui.seg({id:'rf-ic-chP',key:'chP',label:tx('Weight precision','权重精度'),options:[['bf16','BF16'],['fp8','FP8']],value:S.p,inline:true})+
   `<div class="goal mono rf-goal" id="rf-ch-g1"><span>${tx('Share of the H200’s peak FLOP/s used','用上的 H200 峰值算力')}</span><span><b id="rf-ch-s"></b>${tx(' (goal: at least 50%)','（目标：至少 50%）')}</span></div>`+
   `<div class="goal mono rf-goal" id="rf-ch-g2"><span>${tx('Requests in the batch','batch 里的请求数')}</span><span><b id="rf-ch-b"></b>${tx(' (as few as possible)','（越少越好）')}</span></div>`+
   submit+`<div id="rf-ch-res" class="rf-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},batch:{en:'Drag the batch to 64 to continue.',zh:'把 batch 拖到 64 才能继续。'},
    matmul:{en:'Drag n until the dot reaches the flat roof to continue.',zh:'把 n 拖到圆点到达平的屋顶才能继续。'},lever:{en:'Pick a change to continue.',zh:'选一种改动才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#rf-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="rf-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="rf-why" hidden></p>');}
  A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#rf-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]||(s.id==='batch'&&S.B!==1)||(s.id==='matmul'&&S.n!==256)||(s.id==='lever'&&lv!=='base')||userAct)&&(!s.gate||s.gate());
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvTried.has(v)||v===lv?lvRes(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.rf-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#rf-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.B,S.p), sv=$('#rf-ch-s',rail), bv=$('#rf-ch-b',rail); if(!sv) return;
    sv.textContent=fSh(e.share); bv.textContent=fint(S.B);
    const g1=$('#rf-ch-g1',rail), o1=e.ok; g1.classList.toggle('met',o1); g1.classList.toggle('rf-over',!o1);
    $$('.seg-ctl[data-key="chB"]',rail).forEach(g=>A.ui.setPressed(g,String(S.B))); $$('.seg-ctl[data-key="chP"]',rail).forEach(g=>A.ui.setPressed(g,S.p));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#rf-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?STEPS[step].view:(mode==='challenge'?'ch':'one');
const gpuNow=()=>mode==='challenge'?GPUS.h200:(viewNow()==='duo'?GPUS[lvOf(lv).gpu]:(viewNow()==='map'?GPUS.h100:gpuOf(S)));
/* the dots on the chart: the kernel; part 5's two; part 6's five */
function dotsNow(){
  const v=viewNow();
  if(prFig) return [];   // the primer's picture: the roofs, the axes and the ridge, no kernel
  if(v==='duo'){const d=duoOf(lv); return [{key:'dec',r:d.dec,lab:kname({k:'dec',B:DUO_B}),s:{...BASE,k:'dec',B:DUO_B}},{key:'pre',r:d.pre,lab:kname({k:'pre',N:DUO_N}),s:{...BASE,k:'pre',N:DUO_N}}];}
  if(v==='map') return PRESETS.map(p=>{const s=presetState(p); return {key:p.id,r:ev(s),lab:kname(s),short:s.k==='dec'?`decode ${fint(s.B)}`:(s.k==='pre'?'prefill':(s.k==='att'?'attention':kname(s))),s};});
  if(v==='ch'){const s={...BASE,k:'dec',B:S.B,p:S.p,gpu:CH.gpu}; return [{key:'ch',r:ev(s),lab:kname(s)+(S.p==='fp8'?tx(', FP8','，FP8'):''),short:kshort(s),s}];}
  return [{key:'main',r:R(),lab:kname(S)+(S.p==='fp8'&&(S.k==='dec'||S.k==='pre')?tx(', FP8','，FP8'):''),short:kshort(S),s:{...S}}];
}
const X0=-1, X1=4, Y0=-1, Y1=3.5;     // log10 of FLOP per byte moved from memory (0.1 … 10,000) and of TFLOP/s (0.1 … 3,162)
let plotW=640, drawn={prev:false,ghost:false};
/* every label keeps clear of the dots, the dashed rings, the ridge line and the labels placed before it (R7.8) */
function drawPlot(){
  const host=$('#rf-plot',app); if(!host) return;
  const cw=host.clientWidth||600, W=Math.max(300,Math.min(760,Math.round(cw))), narrow=W<480, H=narrow?250:300, ml=narrow?40:50, mr=narrow?10:16, mt=22, mb=40;
  plotW=W;
  const X=I=>ml+(Math.log10(I)-X0)/(X1-X0)*(W-ml-mr), Y=tf=>H-mb-(Math.log10(tf)-Y0)/(Y1-Y0)*(H-mt-mb);
  const g=gpuNow(), pk=g.peak/1e12, bw=g.bw/1e12, rid=pk/bw, v=viewNow();
  let s='';
  const xt=[0.1,1,10,100,1000,10000], yt=[0.1,1,10,100,1000];
  xt.forEach(x=>{const px=X(x); s+=`<line class="rf-grid" x1="${px.toFixed(1)}" y1="${mt}" x2="${px.toFixed(1)}" y2="${H-mb}"/>`+(narrow&&x===10000?'':`<text class="rf-tick" x="${px.toFixed(1)}" y="${H-mb+14}" text-anchor="${x===10000?'end':(x===0.1?'start':'middle')}">${x<1?'0.1':fint(x)}</text>`);});
  yt.forEach(y=>{const py=Y(y); s+=`<line class="rf-grid" x1="${ml}" y1="${py.toFixed(1)}" x2="${W-mr}" y2="${py.toFixed(1)}"/><text class="rf-tick" x="${ml-5}" y="${(py+3.5).toFixed(1)}" text-anchor="end">${y<1?'0.1':fint(y)}</text>`;});
  s+=`<text class="rf-ax rf-axx" x="${((ml+W-mr)/2).toFixed(1)}" y="${H-6}" text-anchor="middle" tabindex="0" data-tipx="${esc(TIP.ai())}">${esc(narrow?tx('FLOPs per byte moved from memory','每从显存搬运 1 字节做多少 FLOP'):tx('Arithmetic intensity: FLOPs per byte moved from memory (log scale)','算术强度：每从显存搬运 1 字节做多少 FLOP（对数轴）'))}</text>`;
  s+=`<text class="rf-ax" x="11" y="${((mt+H-mb)/2).toFixed(1)}" text-anchor="middle" transform="rotate(-90 11 ${((mt+H-mb)/2).toFixed(1)})">${esc(narrow?tx('TFLOP/s reached','达到的 TFLOP/s'):tx('TFLOP/s reached (log scale)','达到的 TFLOP/s（对数轴）'))}</text>`;
  const path=gg=>{const p=gg.peak/1e12, b=gg.bw/1e12, r=p/b; return `M${X(0.1).toFixed(1)},${Y(0.1*b).toFixed(1)} L${X(r).toFixed(1)},${Y(p).toFixed(1)} L${X(1e4).toFixed(1)},${Y(p).toFixed(1)}`;};
  const twd=t=>[...String(t)].reduce((a,ch)=>a+(ch.codePointAt(0)>=0x2e80?11:6.6),0);
  const hit=(b,list)=>list.some(o=>b[0]<o[2]&&o[0]<b[2]&&b[1]<o[3]&&o[1]<b[3]);
  const inPlot=b=>b[0]>=ml+2&&b[2]<=W-mr+1&&b[1]>=mt-2&&b[3]<=H-mb-2;
  /* what the labels keep clear of */
  const dots=dotsNow(), pos=d=>[X(Math.max(0.1,Math.min(1e4,d.r.I))),Y(Math.max(0.1,d.r.att/1e12))], rx=X(rid);
  const rings=prev&&v!=='map'?prev.map(p=>{const cur=dots.find(d=>d.key===p.key); if(!cur||(Math.abs(cur.r.I-p.I)<1e-9&&Math.abs(cur.r.att-p.att)<1)) return null; return [X(p.I),Y(p.att/1e12)];}).filter(Boolean):[];
  const obst=dots.map(d=>{const [px,py]=pos(d); return [px-8,py-8,px+8,py+8];}).concat(rings.map(([x,y])=>[x-9,y-9,x+9,y+9]),[[rx-3,Y(pk)-2,rx+3,H-mb]]);
  const placed=[];
  let lines=[];
  const take=cands=>{for(const c of cands){if(!inPlot(c.b)||hit(c.b,obst)||hit(c.b,placed)||hit(c.b,lines)) continue; placed.push(c.b); return c;} return null;};
  /* a label along a flat roof at height yl: above it from the right end leftwards, then below it */
  const flat=(t,yl,xmin,below)=>{const tw=twd(t), out=[]; for(const dy of below?[-6,20]:[-6]) for(let xa=W-mr-2;xa-tw>=xmin;xa-=6) out.push({x:xa,y:yl+dy,b:[xa-tw-2,yl+dy-11,xa+2,yl+dy+3]}); return out;};
  /* the roof before the last GPU change (part 5's A100: the H100's) */
  const ghost=v==='duo'&&lv==='a100'?GPUS.h100:(v==='one'&&mode==='sandbox'&&prevGpu&&prevGpu!==g.id?GPUS[prevGpu]:null);
  drawn={prev:rings.length>0,ghost:!!ghost};
  /* the roof lines too, so that no label sits across a line: the slanted part in short pieces, the flat part in one */
  const lineBoxes=gg=>{const p=gg.peak/1e12, b=gg.bw/1e12, xr=X(p/b), out=[]; for(let x=X(0.1);x<xr;x+=5){const y=Y(Math.pow(10,(x-ml)/(W-ml-mr)*(X1-X0)+X0)*b); out.push([x-2.5,y-2.5,x+2.5,y+2.5]);}
    out.push([xr-2,Y(p)-2,W-mr,Y(p)+2]); return out;};
  const roofObst=lineBoxes(g).concat(ghost?lineBoxes(ghost):[]);
  if(ghost) s+=`<path class="rf-ghost" d="${path(ghost)}"/>`;
  /* the ridge: dashed, with its value; in part 4 also the n where the matmul crosses it */
  const rlab=tx(`ridge ${fR(rid)}`,`拐点 ${fR(rid)}`); placed.push([rx+2,H-mb-17,rx+6+twd(rlab),H-mb-2]);
  if(v==='one'&&S.k==='mm') placed.push([rx+2,H-mb-30,rx+6+twd('n ≈ 2,655'),H-mb-15]);
  s+=`<g class="rf-rg" tabindex="0" data-tipx="${esc(TIP.ridge()+' '+tx(`The ${g.name}’s ridge: ${fixd(pk,pk%1?1:0)} ÷ ${fixd(bw,2)} = ${fixd(rid,1)} FLOPs per byte moved from memory.`,`${g.name} 的拐点：${fixd(pk,pk%1?1:0)} ÷ ${fixd(bw,2)} = 每从显存搬运 1 字节做 ${fixd(rid,1)} 个 FLOP。`))}"><line class="rf-ridge" x1="${rx.toFixed(1)}" y1="${(Y(pk)).toFixed(1)}" x2="${rx.toFixed(1)}" y2="${H-mb}"/><line class="rf-hit" x1="${rx.toFixed(1)}" y1="${(Y(pk)).toFixed(1)}" x2="${rx.toFixed(1)}" y2="${H-mb}"/>`+
    `<text class="rf-rlab" x="${(rx+4).toFixed(1)}" y="${(H-mb-6).toFixed(1)}">${esc(rlab)}</text>${v==='one'&&S.k==='mm'?`<text class="rf-rlab" x="${(rx+4).toFixed(1)}" y="${(H-mb-19).toFixed(1)}">n ≈ ${fint(Math.round(3*rid))}</text>`:''}</g>`;
  /* the roof: one thick line; the peak labelled along its flat part, clear of the dots */
  s+=`<path class="rf-roof" d="${path(g)}"/>`;
  lines=roofObst;
  /* the dashed rings where the kernels were before the learner's last change */
  rings.forEach(([x,y])=>{s+=`<circle class="rf-prev" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="8"/>`;});
  /* the dots' labels: on the dot's own side of the ridge, near the dot, else further away with a leader line */
  let leads='', labs='', marks='';
  dots.forEach(d=>{const [px,py]=pos(d), left=px<rx, lo=left?ml+2:rx+5, hi=left?rx-5:W-mr, t=d.short&&(narrow||twd(d.lab)>hi-lo)?d.short:d.lab, tw=twd(t);
    const tip=dotTip(d);
    marks+=`<g class="rf-dot${v==='map'?' rf-map':''}" tabindex="0" data-tipx="${esc(tip)}"><circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${v==='map'?5.5:6.5}"/><circle class="rf-hit" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="13"/></g>`;
    const cands=[], DY=[16,-12,30,-26,44,-40,58,72,86];
    for(const dy of DY) for(const side of [1,-1]){
      const x0=Math.max(lo,Math.min(hi-tw,side>0?px+10:px-10-tw)); if(x0<lo-0.5) continue;
      cands.push({x:x0,y:py+dy,b:[x0,py+dy-10,x0+tw,py+dy+2]});}
    for(const dy of DY) for(const side of [1,-1]){const x0=Math.max(ml+2,Math.min(W-mr-tw,side>0?px+10:px-10-tw)); cands.push({x:x0,y:py+dy,b:[x0,py+dy-10,x0+tw,py+dy+2]});}
    const c=take(cands); if(!c) return;
    const nx=Math.max(c.b[0],Math.min(c.b[2],px)), ny=Math.max(c.b[1],Math.min(c.b[3],py)), dd=Math.hypot(nx-px,ny-py);
    if(dd>16){const r0=(v==='map'?5.5:6.5)+2; leads+=`<line class="rf-lead" x1="${(px+(nx-px)/dd*r0).toFixed(1)}" y1="${(py+(ny-py)/dd*r0).toFixed(1)}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}"/>`;}
    labs+=`<text class="rf-dlab" x="${c.x.toFixed(1)}" y="${c.y.toFixed(1)}">${esc(t)}</text>`;
  });
  /* the peak, along the flat roof; the old roof's name along its own flat part */
  const plab=narrow?fPk(g):tx(`${g.name} peak: ${fPk(g)}`,`${g.name} 峰值 ${fPk(g)}`), pc=take(flat(plab,Y(pk),rx-24,true).concat(flat(plab,Y(pk),ml+2,true)))||{x:W-mr-2,y:Y(pk)-6};
  s+=`<text class="rf-rooflab" x="${pc.x.toFixed(1)}" y="${pc.y.toFixed(1)}" text-anchor="end">${esc(plab)}</text>`;
  if(ghost){const gl=tx(`${ghost.name} (before)`,`${ghost.name}（之前）`), gy=Y(ghost.peak/1e12), gc=take(flat(gl,gy,X(ghost.peak/ghost.bw)-24,true));
    if(gc) s+=`<text class="rf-glab" x="${gc.x.toFixed(1)}" y="${gc.y.toFixed(1)}" text-anchor="end">${esc(gl)}</text>`;}
  /* the bandwidth roof: its label runs along the slanted part, where no dot or ring sits; under the line when the old roof runs just above it */
  const sl=Math.atan2(Y(1)-Y(10),X(10)-X(1))*180/Math.PI, ca=Math.cos(sl*Math.PI/180), sa=Math.sin(sl*Math.PI/180);
  const under=!!ghost&&ghost.bw>g.bw, slc=[];
  /* the full label first, then the bandwidth alone if the full one finds no free stretch */
  for(const mlab of narrow?[fBW(g)]:[tx(`memory: ${fBW(g)}`,`显存带宽 ${fBW(g)}`),fBW(g)]){const mlen=twd(mlab);
    for(const off of under?[17,-13]:[-13,17]) for(const I0 of [0.115,0.2,0.3,0.45,0.7,1,1.6,2.5,4,6,10,16,25,40]){const lx=X(I0), ly=Y(I0*bw)+off, qb=[];
      if(lx+mlen*ca>rx-8) continue;
      for(let q=0;q<4;q++){const x0=lx+q/4*mlen*ca, y0=ly-q/4*mlen*sa; qb.push([x0,y0-mlen/4*sa-11,x0+mlen/4*ca,y0+3]);}
      slc.push({lx,ly,qb,mlab,len:mlen});}}
  /* the label runs parallel to its own roof, so only the old roof can cross it: test the old roof's points against the rotated label */
  const gP=[]; if(ghost){const gp=ghost.peak/1e12, gb=ghost.bw/1e12, gx=X(gp/gb); for(let x=X(0.1);x<W-mr;x+=4) gP.push([x,x<gx?Y(Math.pow(10,(x-ml)/(W-ml-mr)*(X1-X0)+X0)*gb):Y(gp)]);}
  const crossG=c=>gP.some(([px,py])=>{const dx=px-c.lx, dy=py-c.ly, u=dx*ca-dy*sa, v=-dx*sa-dy*ca; return u>-3&&u<c.len+3&&v>-6&&v<14;});
  const free=c=>c.qb.every(b=>inPlot(b)&&!hit(b,obst)&&!hit(b,placed));
  let mc=slc.find(c=>free(c)&&!crossG(c))||slc.find(free)||slc[0];
  if(mc){mc.qb.forEach(b=>placed.push(b));
    s+=`<text class="rf-rooflab" x="${mc.lx.toFixed(1)}" y="${mc.ly.toFixed(1)}" transform="rotate(${(-sl).toFixed(1)} ${mc.lx.toFixed(1)} ${mc.ly.toFixed(1)})">${esc(mc.mlab)}</text>`;}
  if(!narrow) s+=`<text class="rf-reg" x="${(ml+6).toFixed(1)}" y="${mt-8}">${esc(tx('← memory-bound','← 受显存带宽限制'))}</text><text class="rf-reg" x="${(W-mr).toFixed(1)}" y="${mt-8}" text-anchor="end">${esc(tx('compute-bound →','受算力限制 →'))}</text>`;
  s+=leads+marks+labs;
  const aria=dots.map(d=>dotTip(d)).join(' ');
  host.innerHTML=`<svg class="rf-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(tx(`Roofline of the ${g.name}. `,`${g.name} 的 roofline。`)+aria)}">${s}</svg>`+plotKeyHTML();
}
function dotTip(d){
  const r=d.r, s=d.s, g=GPUS[r.gpu];
  return tx(`${kfull(s)}${s.p==='fp8'&&(s.k==='dec'||s.k==='pre')?', FP8 weights':''}, on the ${g.name}: ${fFl(r.fl)} and ${fBy(r.by)} moved, ${fI(r.I)} FLOPs per byte moved from memory, ${r.bound==='compute'?'right':'left'} of the ridge (${fR(r.ridge)}). It takes ${fTime(r.t)} and reaches ${fTF(r.att)} TFLOP/s, ${fSh(r.share)} of the peak FLOP/s.`,
            `${s.k==='dec'?`decode 步的权重矩阵乘法（${MODEL}，${reqN(s.B)}${s.p==='fp8'?'，FP8 权重':''}）`:kfull(s)+(s.p==='fp8'&&s.k==='pre'?'，FP8 权重':'')}，在 ${g.name} 上：${fFl(r.fl)}，搬运 ${fBy(r.by)}，每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，在拐点（${fR(r.ridge)}）${r.bound==='compute'?'右':'左'}边。它要 ${fTime(r.t)}，达到 ${fTF(r.att)} TFLOP/s，是峰值算力的 ${fSh(r.share)}。`);
}
/* under the chart: two lanes on one time axis for each kernel, the longer one sets the step's time; the shorter one's rest is idle */
function laneHTML(d){
  const r=d.r, T=r.t, mw=r.tm/T*100, cw=r.tc/T*100, memCls=r.mem==='w'?'rf-w':(r.mem==='kv'?'rf-kv':'rf-x');
  const memLab=r.mem==='w'?tx('Reading weights','读权重'):(r.mem==='kv'?tx('Reading KV cache','读 KV cache'):tx('Memory traffic','读写显存'));
  const memTip=tx(`${fBy(r.by)} ÷ ${fBW(GPUS[r.gpu])} = ${fTime(r.tm)}: the time to move the kernel’s bytes to or from memory.`,`${fBy(r.by)} ÷ ${fBW(GPUS[r.gpu])} = ${fTime(r.tm)}：把这个 kernel 的字节搬进搬出显存要的时间。`)+
    (r.k==='pre'?tx(` The prefill reads ${fBy(r.wby)} of weights and writes ${fBy(r.kvby)} of KV cache for the prompt.`,`prefill 读 ${fBy(r.wby)} 的权重，并为 prompt 写入 ${fBy(r.kvby)} 的 KV cache。`):'');
  const cmpTip=tx(`${fFl(r.fl)} ÷ ${fPk(GPUS[r.gpu])} = ${fTime(r.tc)}: the time the math units need at the peak.`,`${fFl(r.fl)} ÷ ${fPk(GPUS[r.gpu])} = ${fTime(r.tc)}：计算单元按峰值算需要的时间。`);
  const idleTip=tx('Hatched: this unit is idle, waiting for the other one to finish.','斜纹：这一部分空闲着，在等另一部分做完。');
  const lane=(lab,cls,w,val,tip)=>`<div class="rf-lane"><span class="rf-ln">${esc(lab)}</span><div class="rf-trk" tabindex="0" data-tipx="${esc(tip)}"><i class="rf-bar ${cls}" style="width:${Math.max(0.6,w).toFixed(2)}%"></i>${w<99.95?`<i class="rf-idle" style="left:${Math.max(0.6,w).toFixed(2)}%;width:${(100-Math.max(0.6,w)).toFixed(2)}%" data-tipx="${esc(idleTip)}"></i>`:''}</div><span class="rf-lval mono">${val}</span></div>`;
  return `<div class="rf-tk"><div class="rf-tcap">${esc(tx(`${d.lab}: ${fTime(T)}, set by ${r.bound==='compute'?'computing':'memory'}`,`${d.lab}：${fTime(T)}，由${r.bound==='compute'?'计算':'读写显存'}决定`))}</div>`+
    lane(memLab,'rf-mem '+memCls,mw,fTime(r.tm),memTip)+lane(tx('Computing','计算'),'rf-cmp',cw,fTime(r.tc),cmpTip)+`</div>`;
}
function drawTime(){
  const host=$('#rf-time',app); if(!host) return;
  const v=viewNow();
  if(v==='map'){host.innerHTML=`<p class="rf-mapnote small">${esc(tx('The five kernels of Scenarios, all on one H100. Four sit left of the ridge, where memory sets their time; the cards below give each one’s time.','应用场景里的五种 kernel，都在一张 H100 上。四个在拐点左边，耗时由读写显存决定；下面的卡片是每个 kernel 的耗时。'))}</p>`; host._h=''; return;}
  const h=dotsNow().map(laneHTML).join('')+laneKeyHTML();
  if(host._h!==h){host.innerHTML=h; host._h=h;}
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.B,S.p); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when one decode step reaches at least 50% of the H200’s peak. Now: ${fSh(e.share)} with ${reqN(S.B)}.`,`一个 decode 步用上了 H200 峰值算力的 50% 以上时，徽章就是绿色。现在是 ${fSh(e.share)}，batch 里有 ${reqN(S.B)}。`)};}
  if(v==='duo'){const d=duoOf(lv); return {cls:'neutral',txt:tx('decode: memory, prefill: compute','decode 受带宽，prefill 受算力'),short:tx('Two kernels','两个 kernel'),
    tip:tx(`The decode step at batch ${DUO_B} is ${boundTxt(d.dec)}; the prefill of ${fint(DUO_N)} tokens is ${boundTxt(d.pre)}.`,`batch 为 ${DUO_B} 的 decode 步${boundTxt(d.dec)}；${fint(DUO_N)} 个 token 的 prefill ${boundTxt(d.pre)}。`)};}
  if(v==='map') return {cls:'neutral',txt:tx('Five kernels','五种 kernel'),short:tx('Five kernels','五种 kernel'),tip:tx('The five kernels of Scenarios on one H100: four are left of the ridge.','应用场景里的五种 kernel，都在一张 H100 上：四个在拐点左边。')};
  const r=R();
  return {cls:'neutral',txt:r.bound==='compute'?tx('Compute-bound','受算力限制'):tx('Memory-bound','受显存带宽限制'),short:r.bound==='compute'?tx('Compute-bound','受算力限制'):tx('Memory-bound','受带宽限制'),
    tip:r.bound==='compute'?tx(`The kernel does ${fI(r.I)} FLOPs per byte moved from memory, right of the ${GPUS[r.gpu].name}’s ridge at ${fR(r.ridge)}: computing takes longer than moving its bytes, so the peak FLOP/s sets its time.`,`这个 kernel 每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，在 ${GPUS[r.gpu].name} 的拐点 ${fR(r.ridge)} 右边：计算比搬运字节更久，所以峰值算力决定它的耗时。`)
      :tx(`The kernel does ${fI(r.I)} FLOPs per byte moved from memory, left of the ${GPUS[r.gpu].name}’s ridge at ${fR(r.ridge)}: moving its bytes takes longer than computing, so the memory bandwidth sets its time.`,`这个 kernel 每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，在 ${GPUS[r.gpu].name} 的拐点 ${fR(r.ridge)} 左边：搬运字节比计算更久，所以显存带宽决定它的耗时。`)};
}
function drawHead(){
  const v=viewNow(), t=$('#rf-st-title',app), sub=$('#rf-st-sub',app), st=$('#rf-status',app), g=gpuNow();
  t.textContent=v==='ch'?tx(`Where one decode step of ${MODEL} sits on the H200’s roofline`,`${MODEL} 的一个 decode 步在 H200 roofline 上的位置`):
    (v==='duo'?tx(`A decode step and a prefill on the ${g.name}, ${precTxt(lvOf(lv).p)}`,`${g.name} 上的 decode 步和 prefill，${precTxt(lvOf(lv).p)}`):
    (v==='map'?tx('Five kernels on the H100’s roofline','H100 roofline 上的五种 kernel'):kfull(S)));
  sub.textContent=v==='ch'?tx(`H200: ${fPk(g)}, ${fixd(g.bw/1e12,1)} TB/s; batch ${fint(S.B)}, ${precTxt(S.p)}`,`H200：${fPk(g)}，${fixd(g.bw/1e12,1)} TB/s；batch ${fint(S.B)}，${precTxt(S.p)}`)
    :(v==='duo'?tx(`decode at batch ${DUO_B}, prefill of ${fint(DUO_N)} tokens`,`batch 为 ${DUO_B} 的 decode，${fint(DUO_N)} 个 token 的 prefill`):(v==='map'?tx('BF16 weights; each dot is one kernel','BF16 权重；每个圆点是一个 kernel'):tx(`${g.full}: ${fPk(g)}, ${fBW(g)}${(S.k==='dec'||S.k==='pre')?'; '+precTxt(S.p):''}`,`${g.full}：${fPk(g)}，${fBW(g)}${(S.k==='dec'||S.k==='pre')?'；'+precTxt(S.p):''}`)));
  const dec=v==='ch'||v==='duo'||(v==='one'&&S.k==='dec');
  if(dec) sub.textContent+=tx('; KV cache read not included','；不含读 KV cache');
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const sc=$('#rf-scope',app); if(sc) sc.textContent=scopeTxt();
}
/* under the chart, next to the times: what they assume, and how Prefill vs decode's times differ (R2.7) */
function scopeTxt(){
  const v=viewNow(), k=v==='one'?S.k:v;
  if(k==='duo') return tx('Best case: 100% of the datasheet peaks, and the decode step’s weight matmuls only. Prefill vs decode assumes 50% of the peak FLOP/s and 60% of the bandwidth and adds the KV cache, so its times are longer.',
    '最好情况：按规格表峰值的 100%，decode 步只算权重的矩阵乘法。“Prefill 和 decode”这个 Lab 假设只达到峰值算力的 50%、带宽的 60%，还加上 KV cache，所以那里的耗时更长。');
  if(k==='dec'||k==='ch') return tx('Best case: 100% of the datasheet peaks, and the weight matmuls only. Each user’s KV cache is read by another kernel, attention. Prefill vs decode assumes 50% of the peak FLOP/s and 60% of the bandwidth and adds that KV cache, so its steps take longer.',
    '最好情况：按规格表峰值的 100%，只算权重的矩阵乘法。每个用户的 KV cache 由另一个 kernel（attention）读。“Prefill 和 decode”这个 Lab 假设只达到峰值算力的 50%、带宽的 60%，还加上 KV cache，所以那里每一步更久。');
  if(k==='pre') return tx('Best case: 100% of the datasheet peaks, reading and computing at the same time. Prefill vs decode assumes 50% of the peak FLOP/s, so its prefill takes twice as long.',
    '最好情况：按规格表峰值的 100%，读显存和计算同时进行。“Prefill 和 decode”这个 Lab 假设只达到峰值算力的 50%，所以那里的 prefill 要两倍的时间。');
  if(k==='map') return tx('Best case: 100% of the datasheet peaks. The two decode dots are their weight matmuls only; the attention dot is the KV cache read of a decode step.',
    '最好情况：按规格表峰值的 100%。两个 decode 圆点只算权重的矩阵乘法；attention 圆点是 decode 步里读 KV cache 的那部分。');
  return tx('Best case: 100% of the datasheet peaks, reading and computing at the same time.','最好情况：按规格表峰值的 100%，读显存和计算同时进行。');
}
/* each sub-chart carries its own key (review 2026-10-03: one legend above the chart was hard to match):
   under the roofline, what its lines and dots mean, the ring and the old roof only once drawn (R5.7); under the lanes, the hatch */
const keyIt=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
function plotKeyHTML(){
  let h=keyIt('rf-sw-roof',tx('thick line: the most FLOP/s a kernel can reach','粗线：kernel 最多能达到的 FLOP/s'),tx('The thick line is the most FLOP/s a kernel can reach at each arithmetic intensity. The slanted part is limited by the memory bandwidth, the flat part by the peak FLOP/s.','粗线是在每种算术强度下，kernel 能达到的最高 FLOP/s。斜的一段受显存带宽限制，平的一段受峰值算力限制。'))+
    keyIt('rf-sw-ridge',tx('vertical dashed line: the ridge','竖虚线：拐点'),TIP.ridge())+(prFig?'':
    keyIt('rf-sw-dot',tx('dot: a kernel','圆点：一个 kernel'),tx('Each dot is one kernel, placed by its FLOPs per byte moved from memory (across) and the TFLOP/s it reaches (up). It always sits on the roof in this lab, because the lab assumes 100% of the peaks.','每个圆点是一个 kernel，横向是它每从显存搬运 1 字节做多少 FLOP，纵向是它达到的 TFLOP/s。在这个 Lab 里它总在屋顶上，因为这里按峰值的 100% 算。')));
  if(drawn.prev) h+=keyIt('rf-sw-prev',tx('dashed ring: before your last change','虚线圈：上次改动之前'),tx('The dashed ring marks where the dot was before your last change, so you can see how far it moved.','虚线圈标出你上次改动之前圆点的位置，好看出它移动了多远。'));
  if(drawn.ghost) h+=keyIt('rf-sw-ghost',tx('thin dashed roof: the GPU before','细虚线屋顶：之前的 GPU'),tx('After a change of GPU, the thin dashed line is the roof of the GPU you had before.','换了 GPU 之后，细虚线是之前那张 GPU 的屋顶。'));
  return `<div class="legend-row rf-key" id="rf-pkey">${h}</div>`;
}
function laneKeyHTML(){
  return `<div class="legend-row rf-key" id="rf-lkey">${keyIt('rf-sw-idle',tx('hatched: idle, waiting for the other lane to finish','斜纹：空闲，在等另一条做完'),tx('Hatched: the shorter of the two lanes is idle for the rest of the step, waiting for the longer one.','斜纹：两条时间线里较短的那条，在这一步剩下的时间里空闲，等较长的那条。'))}</div>`;
}
/* metric cards: ▲▼ only after the learner's own change, from the values as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=viewNow();
  if(v==='duo'){const d=duoOf(lv), b0=duoOf('base');
    return [{k:'dt',label:tx('Decode step','decode 步'),v:fTime(d.dec.t),base:fTime(b0.dec.t),sub:tx(`batch ${DUO_B}, per step`,`batch ${DUO_B}，每步`),tip:tx(`The decode step at batch ${DUO_B} takes ${fTime(d.dec.t)}: ${boundTxt(d.dec)}.`,`batch 为 ${DUO_B} 的 decode 步要 ${fTime(d.dec.t)}：${boundTxt(d.dec)}。`)},
      {k:'pt',label:tx('Prefill','prefill'),v:fTime(d.pre.t),base:fTime(b0.pre.t),sub:tx(`${fint(DUO_N)} tokens`,`${fint(DUO_N)} 个 token`),tip:tx(`The prefill of ${fint(DUO_N)} tokens takes ${fTime(d.pre.t)}: ${boundTxt(d.pre)}.`,`${fint(DUO_N)} 个 token 的 prefill 要 ${fTime(d.pre.t)}：${boundTxt(d.pre)}。`)},
      {k:'ds',label:tx('Decode: share of the peak FLOP/s','decode：用上的峰值算力'),v:fSh(d.dec.share),base:fSh(b0.dec.share),sub:tx('the FLOP/s reached as a share of the peak FLOP/s','达到的 FLOP/s 占峰值算力的比例'),pts:true,tip:tx(`The decode step reaches ${fTF(d.dec.att)} TFLOP/s, ${fSh(d.dec.share)} of the ${GPUS[d.dec.gpu].name}’s peak.`,`decode 步达到 ${fTF(d.dec.att)} TFLOP/s，是 ${GPUS[d.dec.gpu].name} 峰值算力的 ${fSh(d.dec.share)}。`)+(lv==='a100'?tx(` The share is higher than on the H100 because the A100’s peak is lower; its decode step still takes ${fX(d.dec.t/b0.dec.t)} times as long.`,`比例比 H100 上高，是因为 A100 的峰值更低；它的 decode 步仍然要 H100 上的 ${fX(d.dec.t/b0.dec.t)} 倍时间。`):'')},
      {k:'ri',label:tx('Ridge','拐点'),v:fR(d.dec.ridge),base:fR(b0.dec.ridge),sub:tx('FLOPs per byte moved from memory','每从显存搬运 1 字节做的 FLOP 数'),tip:tx(`The ${GPUS[d.dec.gpu].name}’s ridge: its peak FLOP/s divided by its bandwidth.`,`${GPUS[d.dec.gpu].name} 的拐点：峰值算力除以带宽。`)}];}
  if(v==='map') return PRESETS.map(p=>{const s=presetState(p), r=ev(s); return {k:'m_'+p.id,label:L(p.t),v:fTime(r.t),sub:tx(`${fI(r.I)} FLOPs per byte moved from memory`,`每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP`),tip:dotTip({r,s})};});
  const r=v==='ch'?chEval(S.B,S.p).r:R(), g=GPUS[r.gpu];
  return [{k:'t',label:tx('Time','耗时'),v:fTime(r.t),sub:tx('the longer of the two lanes','两条时间线中较长的'),tip:tx(`The kernel takes ${fTime(r.t)}: the longer of ${fTime(r.tm)} for memory and ${fTime(r.tc)} for computing.`,`这个 kernel 要 ${fTime(r.t)}：读写显存 ${fTime(r.tm)} 和计算 ${fTime(r.tc)} 中较长的那个。`)+(r.t<5e-6?tx(' This is the best case, a lower bound: launching a kernel alone usually takes a few microseconds.','这是最好情况下的耗时，实际只会更长：光是启动一个 kernel，通常就要几微秒。'):'')},
    {k:'i',label:tx('FLOPs per byte moved from memory','每从显存搬运 1 字节做的 FLOP 数'),v:fI(r.I),sub:tx(`ridge: ${fR(r.ridge)}`,`拐点：${fR(r.ridge)}`),tip:tx(`${fFl(r.fl)} ÷ ${fBy(r.by)} = ${fI(r.I)} FLOPs per byte moved from memory, its arithmetic intensity.`,`${fFl(r.fl)} ÷ ${fBy(r.by)} = 每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，就是它的算术强度。`)},
    {k:'tf',label:tx('TFLOP/s reached','达到的 TFLOP/s'),v:fTF(r.att),sub:tx(`peak: ${fixd(g.peak/1e12,g.peak%1e12?1:0)}`,`峰值：${fixd(g.peak/1e12,g.peak%1e12?1:0)}`),tip:tx(`${fFl(r.fl)} ÷ ${fTime(r.t)} = ${fTF(r.att)} TFLOP/s.`,`${fFl(r.fl)} ÷ ${fTime(r.t)} = ${fTF(r.att)} TFLOP/s。`)},
    {k:'sh',label:tx('Share of the peak FLOP/s used','用上的峰值算力'),v:fSh(r.share),sub:tx('the FLOP/s reached as a share of the peak FLOP/s','达到的 FLOP/s 占峰值算力的比例'),pts:true,tip:tx(`This kernel does ${fTF(r.att)} TFLOP per second, ${fSh(r.share)} of the ${g.name}’s peak of ${fPk(g)}.`,`这个 kernel 每秒完成 ${fTF(r.att)} TFLOP，是 ${g.name} 峰值算力 ${fPk(g)} 的 ${fSh(r.share)}。`)}];
}
let noDelta=false;
function drawMetrics(){
  const host=$('#rf-metrics',app), defs=metricsDef(), sig_=defs.map(m=>m.k).join();
  if(host._sig!==sig_){host._sig=sig_; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const pv=m.base!=null?m.base:shownPrev[m.k];
    if(userAct&&!noDelta&&pv!=null&&pv!==m.v){const a=numOf(pv), b=numOf(m.v), sameU=String(pv).replace(/[\d.,<> ]/g,'')===String(m.v).replace(/[\d.,<> ]/g,''); const bo=m.base!=null?{base:{en:'initial setup',zh:'初始设置'}}:{}; if(isFinite(a)&&isFinite(b)&&sameU) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:Math.max(decOf(pv),decOf(m.v)),...bo}:bo)||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each block is one lead line in
   words, the quantity's name and its formula, the reader's numbers under it, then the result. The rows of a formula form a grid, so
   each number sits under the term it replaces and the “=” signs line up (R6.29); every input the reader can change is a term (R1.4) */
const MX=x=>'\\text{'+x+'}';
const MCL=c=>c==null||c===''?'':(c.h!=null?c.h:A.ui.math(c));
/* a long sum (MS: its formula row and the reader's numbers under it) is cut before each “+” into pieces; each piece keeps its numbers
   under its terms, and the pieces wrap onto the next line, indented under the first term, when the box is narrow (the owner's review
   of 2026-10-09: a row the reader has to scroll sideways is not read). The rows after the sum (“= result”) follow on lines of their
   own, their “=” under the sum's */
const MS=(...rows)=>({seg:rows});
const MSG=(rows,rest)=>{const m=rows[0].length, cut=[1]; for(let i=2;i<m;i++) if(rows[0][i]==='+') cut.push(i); cut.push(m);
  const pc=(rs,a,b,cl)=>`<span class="rf-sg${cl||''}" style="grid-template-columns:repeat(${b-a},auto)">`+rs.map(r=>r.slice(a,b).map(c=>`<span class="rf-mc">${MCL(c)}</span>`).join('')).join('')+'</span>';
  return `<div class="rf-mg" style="--n:2;--n1:1"><span class="rf-mc rf-mn rf-sn">${MCL(rows[0][0])}</span><span class="rf-mc rf-sw">`+
    cut.slice(0,-1).map((a,k)=>pc(rows,a,cut[k+1])).join('')+rest.map(r=>pc([r],1,r.length,' rf-sl')).join('')+'</span></div>';};
/* the last cell of a short row spans the rest (not a fixed column), so the rows still line up when the name moves above them */
const MG=rows=>{const k=rows.findIndex(r=>r.seg); if(k>=0) return MSG(rows[k].seg,rows.slice(k+1));
  const n=Math.max(...rows.map(r=>r.length));
  return `<div class="rf-mg" style="--n:${n};--n1:${n-1}">`+rows.map(r=>r.map((c,i)=>{const w=i===r.length-1&&r.length<n;
    return `<span class="rf-mc${i===0?' rf-mn':''}${w?' rf-mw':''}"${w?` style="grid-column:span ${n-i}"`:''}>${MCL(c)}</span>`;}).join('')).join('')+'</div>';};
const MB=(lead,...grids)=>`<div class="rf-mb"><p class="rf-ml">${lead}</p>${grids.map(MG).join('')}</div>`;
function mathOne(s,r){
  const g=GPUS[r.gpu], wb=WB[s.p]||2, pk_=wb===1?'p8':'p', X=MX, N=(k,x)=>({h:T_(k,A.ui.math(X(x)))}), F=(a,b)=>`\\frac{${a}}{${b}}`;
  const FL=X(tx('FLOPs','FLOP 数')), BY=X(tx('bytes','字节数')), WP=X(tx('weights','权重参数')), BPW=X(tx('bytes per weight','每个权重的字节数'));
  const PL={h:'<span class="rf-par">(</span>'}, PR={h:'<span class="rf-par">)</span>'};
  const PK=X(tx('peak FLOP/s','峰值算力')), BW=X(tx('bandwidth','显存带宽')), TM=X(tx('time','耗时')), RE=X(tx('FLOP/s reached','达到的 FLOP/s'));
  let fl, by;
  if(s.k==='dec'){
    fl=[[FL,'=','2','\\times',WP,'\\times',X('batch')],['','=','2','\\times',X(fint(W_PAR)),'\\times',N('B',fint(s.B))],['','=',X(fFl(r.fl))]];
    by=[[BY,'=',BPW,'\\times',WP],['','=',N(pk_,String(wb)),'\\times',X(fint(W_PAR))],['','=',X(fBy(r.by))]];}
  else if(s.k==='pre'){
    const PT=X(tx('prompt tokens','prompt token 数'));
    fl=[MS([FL,'=','2','\\times',X(tx('layer weights','各层权重')),'\\times',PT,'+','2','\\times',X(tx('output head','输出头权重')),'+',X('attention')],
        ['','=','2','\\times',X(fint(P_LAYERS)),'\\times',N('N',fint(s.N)),'+','2','\\times',X(fint(P_HEAD)),'+',`2 \\times 32 \\times 4,096 \\times ${fint(s.N)} \\times ${fint(s.N+1)}`]),
        ['','=',X(fFl(r.fl))]];
    by=[MS([BY,'=',BPW,'\\times',WP,'+',X(tx('KV bytes per token','每个 token 的 KV 字节数')),'\\times',PT],
        ['','=',N(pk_,String(wb)),'\\times',X(fint(W_PAR)),'+',X(fint(KV_TOK)),'\\times',N('N',fint(s.N))]),['','=',X(fBy(r.by))]];}
  else if(s.k==='att'){
    fl=[[FL,'=',X('batch'),'\\times','4','\\times',X(tx('layers','层数')),'\\times',X(tx('width','宽度')),'\\times',X(tx('(context + 1)','（上下文 + 1）'))],
        ['','=',N('B',fint(s.B)),'\\times','4','\\times','32','\\times','4,096','\\times',N('c',`(${fint(s.c)} + 1)`)],['','=',X(fFl(r.fl))]];
    by=[[BY,'=',X('batch'),'\\times',X(tx('KV bytes per token','每个 token 的 KV 字节数')),'\\times',X(tx('context','上下文'))],
        ['','=',N('B',fint(s.B)),'\\times',X(fint(KV_TOK)),'\\times',N('c',fint(s.c))],['','=',X(fBy(r.by))]];}
  else if(s.k==='mm'){
    fl=[[FL,'=','2','\\times','n^3'],['','=','2','\\times',{h:T_('n',A.ui.math(`${fint(s.n)}^3`))}],['','=',X(fFl(r.fl))]];
    by=[[BY,'=','2','\\times','3','\\times','n^2'],['','=','2','\\times','3','\\times',{h:T_('n',A.ui.math(`${fint(s.n)}^2`))}],['','=',X(fBy(r.by))]];}
  else {
    fl=[[FL,'=',X(tx('numbers','数值的数量'))],['','=','2,048 \\times 4,096'],['','=',X(fFl(r.fl))]];
    by=[[BY,'=','3','\\times','2','\\times',X(tx('numbers','数值的数量'))],['','=','3','\\times','2','\\times',X(fint(EW_N))],['','=',X(fBy(r.by))]];}
  return MB(tx('First, how much arithmetic the kernel does and how many bytes it moves to or from memory.','先算这个 kernel 要做多少运算，要和显存来回搬多少字节。'),fl,by)+
    MB(tx('Divided, they give the arithmetic intensity: the FLOPs per byte moved from memory. Compared with the GPU’s ridge, it says which side of the ridge the kernel is on.','两者相除，就是算术强度：每从显存搬运 1 字节做的 FLOP 数。拿它和 GPU 的拐点比，就知道 kernel 在拐点哪一边。'),
      [[X(tx('arithmetic intensity','算术强度')),'=',F(FL,BY)],['','=',F(X(fFl(r.fl)),X(fBy(r.by)))],['','=',X(fI(r.I))]],
      [[{h:T_('peak',A.ui.math(X(tx('ridge','拐点'))))},'=',F(PK,BW)],['','=',F(X(fPk(g)),X(fBW(g)))],['','=',X(fixd(r.ridge,1))]])+
    MB(tx('The GPU reads and computes at the same time, so the kernel takes the longer of the two times.','GPU 一边读一边算，所以耗时取两个时间中较长的那个。'),
      [[{h:T_('fx',A.ui.math(TM))},'=','\\max',PL,F(FL,PK),',',F(BY,BW),PR],
       ['','=','\\max',PL,F(X(fFl(r.fl)),X(fPk(g))),',',F(X(fBy(r.by)),X(fBW(g))),PR],
       ['','=','\\max',PL,X(fTime(r.tc)),',',X(fTime(r.tm)),PR],['','=',X(fTime(r.t))]])+
    MB(tx('The FLOPs it does per second, and what share of the GPU’s peak that is.','它每秒完成的 FLOP，以及这占峰值算力的多少。'),
      [[RE,'=',F(FL,TM)],['','=',F(X(fFl(r.fl)),X(fTime(r.t)))],['','=',X(fTF(r.att)+' TFLOP/s')]],
      [[X(tx('share of the peak FLOP/s','占峰值算力的比例')),'=',F(RE,PK)],['','=',F(X(fTF(r.att)+' TFLOP/s'),X(fPk(g)))],['','=',X(fSh(r.share))]]);
}
function mathHTML(){
  const v=viewNow(), hd=s=>`<p class="rf-mh"><b>${esc(s)}</b></p>`;
  if(v==='duo'){const d=duoOf(lv), o=lvOf(lv); return hd(tx(`Decode, batch ${DUO_B}`,`decode，batch ${DUO_B}`))+mathOne({...BASE,k:'dec',B:DUO_B,p:o.p,gpu:o.gpu},d.dec)+hd(tx(`Prefill, ${fint(DUO_N)} tokens`,`prefill，${fint(DUO_N)} 个 token`))+mathOne({...BASE,k:'pre',N:DUO_N,p:o.p,gpu:o.gpu},d.pre);}
  if(v==='map') return PRESETS.map(p=>{const s=presetState(p); return hd(L(p.t))+mathOne(s,ev(s));}).join('');
  if(v==='ch'){const s={...BASE,k:'dec',B:S.B,p:S.p,gpu:CH.gpu}; return mathOne(s,ev(s));}
  return mathOne(S,R());
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!CH.Bs.includes(S.B)) S.B=CH.start.B; if(!CH.Ps.includes(S.p)) S.p=CH.start.p; S.k='dec';}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  if(isP('batch')&&S.B>=64) seen64=true; if(isP('batch')&&S.B>=384) seenPast=true; if(isP('matmul')&&R().bound==='compute') mmPast=true;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawPlot(); drawTime(); drawMetrics();
  const mb=$('#rf-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the batch and the weight precision, in the task card. Everything else is fixed: one decode step of ${MODEL} on one H200.`,`挑战中只能在任务卡里改 batch 和权重精度。其余设置固定：${MODEL} 的一个 decode 步，一张 H200。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'||id==='predict') return tx(`Fixed in this part: a decode step of ${MODEL} at batch 1, on one H100, BF16 weights. Part 3 lets you change the batch.`,`这一部分固定：${MODEL} 的 decode 步，batch 为 1，一张 H100，BF16 权重。第 3 部分可以改 batch。`);
  if(id==='batch') return tx('Fixed in this part: the decode step on one H100, BF16 weights. You can change them in Scenarios.','这一部分固定：decode 步，一张 H100，BF16 权重。在应用场景里可以改。');
  if(id==='matmul') return tx('Fixed in this part: a square matmul in BF16 on one H100. You can change the kernel and the GPU in Scenarios.','这一部分固定：一张 H100 上的 BF16 方阵乘法。在应用场景里可以改 kernel 和 GPU。');
  if(id==='lever') return tx('In this part you pick one change in the card. You can change the settings in Scenarios.','这一部分只在卡片里选一种改动。在应用场景里可以改这些设置。');
  return tx('Fixed in this part: the five kernels of Scenarios on one H100. You can change the settings in Scenarios.','这一部分固定：应用场景里的五种 kernel，一张 H100。在应用场景里可以改这些设置。');
}
function syncControls(){
  const setR=(key,idx,txt)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==idx) i.value=idx; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=txt; i.setAttribute('aria-valuetext',txt);});
  setR('B',bIdx(S.B),fint(S.B)); setR('n',mIdx(S.n),fint(S.n)); setR('N',Math.max(0,NST.indexOf(S.N)),fint(S.N)); setR('c',Math.max(0,CST.indexOf(S.c)),fint(S.c));
  $$('.seg-ctl[data-key="k"]',app).forEach(g=>A.ui.setPressed(g,S.k)); $$('.seg-ctl[data-key="gpu"]',app).forEach(g=>A.ui.setPressed(g,S.gpu)); $$('.seg-ctl[data-key="p"]',app).forEach(g=>A.ui.setPressed(g,S.p));
  const c=$('#rf-controls',app), ch=mode==='challenge', gd=mode==='guided';
  c.classList.toggle('locked',ch);
  const hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  const liveB=!gd||isP('batch'), liveN=!gd||isP('matmul');
  if(gd){hide('#rf-ctl-k-wrap',true); hide('#rf-ctl-B-wrap',!isP('batch')); hide('#rf-ctl-n-wrap',!isP('matmul')); ['#rf-ctl-N-wrap','#rf-ctl-c-wrap','#rf-ctl-gpu-wrap','#rf-ctl-p-wrap','#rf-ctl-asm'].forEach(x=>hide(x,true)); hide('#rf-ctl-reset-wrap',!(liveB&&isP('batch'))&&!(liveN&&isP('matmul')));}
  else if(ch){['#rf-ctl-k-wrap','#rf-ctl-B-wrap','#rf-ctl-N-wrap','#rf-ctl-c-wrap','#rf-ctl-n-wrap','#rf-ctl-gpu-wrap','#rf-ctl-p-wrap','#rf-ctl-asm','#rf-ctl-reset-wrap'].forEach(x=>hide(x,true));}
  else {hide('#rf-ctl-k-wrap',false); hide('#rf-ctl-B-wrap',!(S.k==='dec'||S.k==='att')); hide('#rf-ctl-N-wrap',S.k!=='pre'); hide('#rf-ctl-c-wrap',S.k!=='att'); hide('#rf-ctl-n-wrap',S.k!=='mm');
    hide('#rf-ctl-gpu-wrap',false); hide('#rf-ctl-p-wrap',false); hide('#rf-ctl-asm',false); hide('#rf-ctl-reset-wrap',false);
    const pw=!(S.k==='dec'||S.k==='pre'); const seg=$('#rf-ctl-p',app); if(seg) seg.hidden=pw; const pn=$('#rf-pnote',app); if(pn) pn.hidden=!pw;}
  const ln=$('#rf-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const v=viewNow();
  if(v==='ch'){const e=chEval(S.B,S.p); A.announce(tx(`Batch ${S.B}, ${precTxt(S.p)}: ${fSh(e.share)} of the H200’s peak.`,`batch ${S.B}，${precTxt(S.p)}：用上了 H200 峰值算力的 ${fSh(e.share)}。`)); return;}
  if(v!=='one') return;
  const r=R(); A.announce(tx(`${kname(S)}: ${fTime(r.t)}, ${fI(r.I)} FLOPs per byte moved from memory, ${boundTxt(r)}, ${fSh(r.share)} of the peak FLOP/s.`,`${kname(S)}：${fTime(r.t)}，每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，${boundTxt(r)}，用上了峰值算力的 ${fSh(r.share)}。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
/* the dots as they are now, kept as the "before" ring when the learner changes something (R3.3) */
const snapDots=()=>dotsNow().map(d=>({key:d.key,I:d.r.I,att:d.r.att}));
function setState(p){
  if(preLock()&&p.B!=null) p={...p,B:S.B};
  const before=userAct?snapDots():null, g0=S.gpu;
  Object.assign(S,p);
  if(!BST.includes(S.B)) S.B=1; if(!NST.includes(S.N)) S.N=2048; if(!CST.includes(S.c)) S.c=4096; if(!MST.includes(S.n)) S.n=256;
  if(!KINDS.includes(S.k)) S.k='dec'; if(!['a100','h100'].includes(S.gpu)) S.gpu='h100'; if(!WB[S.p]) S.p='bf16';
  if(before) prev=before; if(userAct&&g0!==S.gpu) prevGpu=g0;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(id){const before=userAct?snapDots():null; lv=id; lvTried.add(id); if(before) prev=before; render();}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i)); prev=null; prevGpu=null;
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  const st=$('#rf-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg=(innerWidth<=760?{batch:'#rf-ph-B',matmul:'#rf-ph-n'}:{})[s.id]||{batch:'#rf-ic-B',matmul:'#rf-ic-n',lever:'#rf-ic-lv'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#rf-rail .step.current',app)||$('#rf-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), as labs/sampling.js does */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#rf-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#rf-rail .step.current',app)||$('#rf-rail .railcard',app)||$('#rf-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#rf-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#rf-insight',app)); else revealBox($('#rf-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; prev=null; prevGpu=null;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,k:'dec',B:CH.start.B,p:CH.start.p};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#rf-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[S.k,S.B,S.N,S.c,S.n,S.gpu,S.p].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const [k,B,N,c,n,gpu,pr]=[p[0],+p[1],+p[2],+p[3],+p[4],p[5],p[6]];
  if(!KINDS.includes(k)||!BST.includes(B)||!NST.includes(N)||!CST.includes(c)||!MST.includes(n)||!['a100','h100'].includes(gpu)||!WB[pr]) return null;
  return {k,B,N,c,n,gpu,p:pr};
}
/* phones: the part's control at the top of the chart, so the control and what it changes share one screen */
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='predict'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='lever') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the three options in the card.','向下滚到卡片里的三个选项。'))}">${tx('Pick a change below ↓','在下面选一种改动 ↓')}</button>`;
  if(id==='batch') return bCtl('rf-ph-B',true);
  if(id==='matmul') return mCtl('rf-ph-n',true);
  return '';
}
function renderPhone(){const el=$('#rf-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}

/* =====================================================================
   interview drill: seeded, from the engine; every line redoes from the numbers it shows (R6.6)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DM=[1,8,32,128,512,2048], DK=[1024,4096,8192], DG=['a100','h100'];
function makeDrill(seed){const rnd=mulberry(seed*104729+7), pk=a=>a[Math.floor(rnd()*a.length)];
  const m=pk(DM), k=pk(DK), n=pk(DK), gpu=pk(DG), g=GPUS[gpu], fl=2*m*k*n, by=2*(m*k+k*n+m*n), r=roof({k:'mm',fl,by,mem:'x'},g);
  return {seed,m,k,n,gpu,fl,by,r};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), g=GPUS[d.gpu], qd=$('#rf-drill-q',app), a=$('#rf-drill-a',app);
  const art=x=>/^(8|11|18)/.test(String(x))?'an':'a';
  qd.innerHTML=tx(`An ${g.full} (${fPk(g)}, ${fBW(g)}) multiplies ${art(d.m)} <b>${fint(d.m)} × ${fint(d.k)}</b> matrix by ${art(d.k)} <b>${fint(d.k)} × ${fint(d.n)}</b> matrix in BF16. How many FLOPs per byte moved from memory, which limit sets its time, and how long does it take at best?`,
                  `一张 ${g.full}（${fPk(g)}，${fBW(g)}）用 BF16 计算一个 <b>${fint(d.m)} × ${fint(d.k)}</b> 矩阵乘一个 <b>${fint(d.k)} × ${fint(d.n)}</b> 矩阵。每从显存搬运 1 字节做多少个 FLOP，是哪个限制决定耗时，最快要多久？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const r=d.r;
  a.innerHTML=tx(`<p><b>${fI(r.I)} FLOPs per byte moved from memory, ${boundTxt(r)}: ${fTime(r.t)} at best.</b></p><p>FLOPs = 2 × ${fint(d.m)} × ${fint(d.k)} × ${fint(d.n)} = ${fint(d.fl)}. Bytes = 2 × (${fint(d.m)} × ${fint(d.k)} + ${fint(d.k)} × ${fint(d.n)} + ${fint(d.m)} × ${fint(d.n)}) = ${fint(d.by)}. Intensity = ${fint(d.fl)} ÷ ${fint(d.by)} = ${fixd(r.I,1)}; the ridge is ${fixd(g.peak/1e12,1)} ÷ ${fixd(g.bw/1e12,3)} = ${fixd(r.ridge,1)}.</p><p>Compute ${fTime(r.tc)}, memory ${fTime(r.tm)}: the time is the longer, ${fTime(r.t)}, at ${fSh(r.share)} of the peak FLOP/s.</p>`,
    `<p><b>每从显存搬运 1 字节做 ${fI(r.I)} 个 FLOP，${boundTxt(r)}：最快 ${fTime(r.t)}。</b></p><p>FLOP 数 = 2 × ${fint(d.m)} × ${fint(d.k)} × ${fint(d.n)} = ${fint(d.fl)}。字节数 = 2 × (${fint(d.m)} × ${fint(d.k)} + ${fint(d.k)} × ${fint(d.n)} + ${fint(d.m)} × ${fint(d.n)}) = ${fint(d.by)}。算术强度 = ${fint(d.fl)} ÷ ${fint(d.by)} = ${fixd(r.I,1)}；拐点是 ${fixd(g.peak/1e12,1)} ÷ ${fixd(g.bw/1e12,3)} = ${fixd(r.ridge,1)}。</p><p>计算 ${fTime(r.tc)}，读写显存 ${fTime(r.tm)}：耗时取较长的 ${fTime(r.t)}，用上了峰值算力的 ${fSh(r.share)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={B:{en:'Batch',zh:'batch'},n:{en:'Matrix size n',zh:'矩阵大小 n'},N:{en:'Prompt length',zh:'prompt 长度'},c:{en:'Context per user',zh:'每个用户的上下文'},p:{en:'Weight precision',zh:'权重精度'},p8:{en:'Weight precision',zh:'权重精度'},peak:{en:'GPU',zh:'GPU'},bw:{en:'GPU',zh:'GPU'}};
const TERM={
 B:()=>`<div>${esc(TIP.batch())}</div>`,
 n:()=>`<div>${esc(tx('The size of both square matrices. The matmul does','两个方阵的大小。这个矩阵乘法做'))} ${A.ui.math('2n^3')} ${esc(tx('FLOPs and moves','个 FLOP，搬运'))} ${A.ui.math('3n^2')} ${esc(tx('numbers of 2 bytes, so:','个数值、每个 2 字节，所以：'))}</div><div class="fx">${A.ui.math(tx('\\text{FLOPs per byte moved} = \\frac{2n^3}{2 \\times 3n^2} = \\frac{n}{3}','\\text{每搬运 1 字节做的 FLOP 数} = \\frac{2n^3}{2 \\times 3n^2} = \\frac{n}{3}'))}</div>`,
 N:()=>`<div>${esc(tx('The number of prompt tokens the prefill processes in one step. Every weight is multiplied by each of them.','prefill 一步处理的 prompt token 数。每个权重都要和它们每一个相乘。'))}</div>`,
 c:()=>`<div>${esc(tx('The tokens in each user’s KV cache. The step reads all of them, 128 KiB per token in BF16, and does about 4 FLOPs per byte read.','每个用户 KV cache 里的 token 数。这一步要把它们全部读一遍，BF16 每个 token 128 KiB，每读 1 字节约做 4 个 FLOP。'))}</div>`,
 p:()=>`<div>${esc(tx('2 bytes per weight: the weights are stored in BF16. FP8 weights would take 1 byte each.','每个权重 2 字节：权重用 BF16 存储。换成 FP8 权重，每个只占 1 字节。'))}</div>`,
 p8:()=>`<div>${esc(tx('1 byte per weight: the weights are stored in FP8. ','每个权重 1 字节：权重用 FP8 存储。')+TIP.fp8())}</div>`,
 W:()=>`<div>${esc(TIP.wread())}</div>`,
 peak:()=>`<div>${esc(tx('The peak FLOP/s is the most arithmetic the GPU can do per second, from its datasheet: dense BF16 on the tensor cores. The H100’s is 989.5 TFLOP/s, half of the datasheet’s 1,979 “with sparsity”.','峰值算力是 GPU 每秒最多能做的运算，来自规格表：张量核心上的稠密 BF16。H100 是 989.5 TFLOP/s，即规格表上“with sparsity”的 1,979 的一半。'))}</div>`,
 bw:()=>`<div>${esc(tx('The memory bandwidth is how many bytes per second the GPU can read from or write to HBM, from its datasheet. The H100 SXM’s is 3.35 TB/s.','显存带宽是 GPU 每秒能从 HBM 读或写多少字节，来自规格表。H100 SXM 是 3.35 TB/s。'))}</div>`,
 fx:()=>`<div>${esc(TIP.overlap())}</div><div class="fx">${A.ui.math?A.ui.math(FX_T()):''}</div>`
};
const termTipHTML=k=>{if(!TERM[k]) return ''; const a=jumpAct(k); return `${TERM[k]()}`+(a?`<div class="act">${esc(a)}</div>`:'');};
function ctlLockMsg(k){
  if(k==='W') return tx(`Fixed by the model: ${MODEL}’s parameters do not change in this lab.`,`由模型决定：在这个 Lab 里 ${MODEL} 的参数不变。`);
  if(k==='fx') return '';
  if(mode==='challenge') return k==='B'||k==='p'||k==='p8'?'':tx('Fixed during the challenge: one decode step of '+MODEL+' on one H200.','挑战中固定不变：'+MODEL+' 的一个 decode 步，一张 H200。');
  if(mode!=='guided') return '';
  if(k==='B'&&isP('batch')) return ''; if(k==='n'&&isP('matmul')) return '';
  if(k==='B'&&step<SIX('batch')) return isP('predict')&&!pred?tx('The batch can change after you answer part 2.','在第 2 部分作答后，才能改 batch。'):tx('Fixed at 1 in this part; part 3 lets you change it.','这一部分固定为 1；第 3 部分可以改。');
  if(k==='peak'||k==='bw') return tx('Fixed in this part: the H100. In Scenarios you can pick another GPU under Settings.','这一部分固定为 H100。在应用场景里，可以在“自由调参”里换 GPU。');
  if(k==='p'||k==='p8') return tx('Fixed in this part. In Scenarios you can change the weight precision under Settings.','这一部分固定不变。在应用场景里，可以在“自由调参”里改权重精度。');
  return tx('Fixed in this part. In Scenarios you can change it under Settings.','这一部分固定不变。在应用场景里，可以在“自由调参”里改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math'||k==='fx'){const d=app&&$('#rf-math',app); return d&&d.open&&k==='fx'?'':tx('Click to open “Show the math”.','点击展开“展开公式”。');}
  if(!CTLN[k]&&k!=='W') return '';
  const m=ctlLockMsg(k); if(m) return m;
  if(mode==='challenge') return tx(`Click to jump to “${L(CTLN[k])}” in the task card.`,`点击跳到任务卡里的“${L(CTLN[k])}”。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({B:$('#rf-ic-B',app)||$('#rf-ic-chB',app)||$('#rf-ctl-B',app),n:$('#rf-ic-n',app)||$('#rf-ctl-n',app),N:$('#rf-ctl-N',app),c:$('#rf-ctl-c',app),p:$('#rf-ic-chP',app)||$('#rf-ctl-p',app),p8:$('#rf-ic-chP',app)||$('#rf-ctl-p',app),peak:$('#rf-ctl-gpu',app),bw:$('#rf-ctl-gpu',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'||k==='fx'){const d=$('#rf-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#rf-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]&&k!=='W') return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(ctlLockMsg(k)||tx('Fixed in this view.','这里固定不变。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{if(mode==='guided') setState({B:isP('batch')?1:S.B,n:isP('matmul')?256:S.n}); else setState({...BASE});}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.B,S.p).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#rf-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#rf-ic-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#rf-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#rf-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='B') bySettings(e.target,()=>setState({B:BST[v]}));
    else if(k==='n') bySettings(e.target,()=>setState({n:MST[v]}));
    else if(k==='N') bySettings(e.target,()=>setState({N:NST[v]}));
    else if(k==='c') bySettings(e.target,()=>setState({c:CST[v]}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#rf-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#rf-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#rf-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('lever')){const f=()=>byLearner(()=>setLv(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='k') bySettings(b,()=>setState({k:d.v}));
      else if(k==='gpu') bySettings(b,()=>setState({gpu:d.v}));
      else if(k==='p') bySettings(b,()=>setState({p:d.v}));
      else if(k==='chB') byLearner(()=>setState({B:+d.v}));
      else if(k==='chP') byLearner(()=>setState({p:d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; noDelta=true; try{byLearner(()=>setState(presetState(p)));} finally{noDelta=false;} requestAnimationFrame(()=>{if(app) revealBox($('#rf-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#rf-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (primer, approved outline of 2026-10-12 07:13 UTC): four sections read before the guided parts. A kernel reads from HBM
   and computes on the chip; the two limits P and W and t ≈ max(F/P, B/W); arithmetic intensity and the ridge; the roofline. They
   give the mechanism only: how strong each effect is stays with the guided parts. The one picture is the lab's own roofline
   (drawPlot), drawn with the primer's own state (the H100, no kernel dot, no lanes), whatever the reader set in the lab
   ===================================================================== */
let prFig=false;   // dotsNow and plotKeyHTML draw no kernel while the primer's picture is drawn
/* drawPlot into a detached box of width w with the primer's state; the reader's state and the module's drawing state put back after */
function prRoofSVG(w){
  const sv={S,mode,step,lv,prev,prevGpu,app,plotW,drawn}, box=document.createElement('div');
  box.innerHTML='<div id="rf-plot"></div>'; const host=box.firstChild; Object.defineProperty(host,'clientWidth',{value:w});
  S={...BASE}; mode='sandbox'; step=0; lv='base'; prev=null; prevGpu=null; app=box; prFig=true;
  try{drawPlot(); return host.innerHTML;}
  finally{({S,mode,step,lv,prev,prevGpu,app,plotW,drawn}=sv); prFig=false;}
}
/* the primer's copy gets its own ids, so the lab's lookups (#rf-pkey) never land in the primer; nothing in it takes focus */
const prIds=h=>h.replace(/(\b(?:id|for|aria-labelledby|aria-describedby|aria-controls)="|url\(#|href="#)rf-/g,'$1pr-rf-').replace(/tabindex="0"/g,'tabindex="-1"');
const prRoofHTML=w=>`<div class="rf-plot pr-rfbox">${prIds(prRoofSVG(w))}</div>`;
/* drawn again at the figure's own width once it is on the page, and whenever that width changes */
function prRoofMount(fig){const box=fig.querySelector('.pr-rfbox'); if(!box) return; let w0=0, ro=null;
  const draw=()=>{if(!box.isConnected){if(ro) ro.disconnect(); return;} const w=Math.round(box.clientWidth); if(w<60||w===w0) return; w0=w; box.innerHTML=prIds(prRoofSVG(w));};
  if(typeof ResizeObserver==='function'){ro=new ResizeObserver(draw); ro.observe(box);} draw();}
const M_=A.ui.math;
const prP=(...ps)=>({html:()=>ps.map(p=>`<p>${p()}</p>`).join('')});
const prX=src=>`<span class="rf-prx">${M_(src)}</span>`;
const pk0=fixd(GPUS.h100.peak/1e12,1), bw0=fixd(GPUS.h100.bw/1e12,2), rid0=fR(GPUS.h100.peak/GPUS.h100.bw);
if(A.registerPrimer) A.registerPrimer(ID,{min:6,sections:[
 {h:()=>tx('A kernel on the GPU: read data from memory (HBM), compute in the compute units','GPU 上的一个 kernel：从显存（HBM）读数据，在计算单元上算'),parts:[
   prP(()=>tx('Every step a large model computes on the GPU (generating one token, say) is a sequence of kernels run one after another. A kernel is one program the GPU runs over a whole tensor: one matrix multiplication is a kernel, one attention is another, and so is a residual add (two tensors added number by number) or an RMSNorm (each vector divided by its root mean square). A step takes as long as its kernels added up, so a faster model needs faster kernels, and making a kernel faster starts with knowing what limits it.',
              '大模型在 GPU 上每算一步（比如生成一个 token），都是一串 kernel 依次执行。kernel 是 GPU 在一整个张量上运行的一个程序：一次矩阵乘法是一个 kernel，一次注意力是一个，残差相加（两个张量逐个数值相加）、RMSNorm（把每个向量除以它的均方根）这类运算也各是一个。一步的耗时就是这些 kernel 的耗时加起来，所以模型要变快，就得让 kernel 变快；而要让一个 kernel 变快，先得知道是什么限制了它。'),
       ()=>tx('A kernel does three things. It reads its inputs from the GPU’s memory: weights, and activations (the intermediate results a layer computes and hands on). It computes in the chip’s compute units; matrix multiplications run on special units among them, the tensor cores. And it writes its result back to memory. The GPU’s memory is HBM (high-bandwidth memory). It is large, 80 GB on an H100, and the model’s weights are kept there. But it sits beside the compute chip, not on it, so every byte has to cross the memory bus between the two before the compute units can use it.',
              '一个 kernel 做三件事：从显存读入输入，也就是权重和激活值（每一层算出来、交给下一层的中间结果）；在芯片上的计算单元里算，矩阵乘法由其中专门的 tensor core 来做；再把结果写回显存。GPU 的显存叫 HBM（高带宽显存），容量大，H100 上有 80 GB，模型的权重就存在这里。可它在计算芯片旁边，而不在芯片上，每个字节都要先经过两者之间的总线，才能被计算单元用上。'),
       ()=>tx('The chip has memory of its own, SRAM (registers and caches). It is far faster than HBM but far smaller: the largest part of it, the H100’s L2 cache, holds 50 MB. A model of 8 billion parameters stored in BF16 (a floating-point format of 2 bytes per number) has more than ten GB of weights, hundreds of times the L2 cache, so they cannot fit. So a large tensor streams in from HBM again every time it is used, and a kernel’s time depends on how long this moving takes and how long the computing takes.',
              '芯片上也有自己的存储，叫 SRAM（寄存器和缓存），读写比 HBM 快得多，容量却小得多：其中最大的 L2 缓存，在 H100 上也只有 50 MB。一个 80 亿参数的模型用 BF16（每个数 2 字节的浮点格式）存，权重有十几 GB，比 L2 缓存大几百倍，放不进去。所以大的张量每用一次，都要从 HBM 重新读进芯片；kernel 的耗时，就看搬运和计算各要多久。'))]},
 {h:()=>tx('Two limits: peak compute (FLOP/s) and memory bandwidth (B/s)','两个上限：峰值算力（FLOP/s）和显存带宽（B/s）'),parts:[
   prP(()=>tx(`How long each takes depends on how fast the GPU can do it. Computing is counted in FLOPs: one FLOP is one floating-point add or multiply, so each multiply-add of a matrix multiplication is 2 FLOPs. The most FLOPs a GPU can do per second is its peak compute, ${M_('P')}. For BF16 matrix multiplications on an H100 (the SXM version, as everywhere here), ${M_(`P = ${pk0}\\ \\text{TFLOP/s}`)}, where 1 TFLOP is ${M_('10^{12}')} FLOPs.`,
              `搬运和计算各要多久，取决于 GPU 做这两件事最快能多快。计算按 FLOP 来数：一次浮点加法或乘法是 1 个 FLOP，矩阵乘法里每个乘加算 2 个 FLOP。GPU 每秒最多能做的 FLOP 数叫峰值算力，记作 ${M_('P')}。H100（SXM 版，下同）用 BF16 做矩阵乘法，${M_(`P = ${pk0}\\ \\text{TFLOP/s}`)}，1 TFLOP 是 ${M_('10^{12}')} 个 FLOP。`),
       ()=>tx(`The most bytes the GPU can move per second between HBM and the chip is its memory bandwidth, ${M_('W')}. The H100’s HBM3 memory (the third generation of HBM) gives ${M_(`W = ${bw0}\\ \\text{TB/s}`)}, where 1 TB is ${M_('10^{12}')} bytes.`,
              `GPU 每秒最多能在 HBM 和芯片之间搬运的字节数叫显存带宽，记作 ${M_('W')}。H100 用的是 HBM3（第三代 HBM），${M_(`W = ${bw0}\\ \\text{TB/s}`)}，1 TB 是 ${M_('10^{12}')} 字节。`),
       ()=>tx(`Say a kernel does ${M_('F')} FLOPs and moves ${M_('B')} bytes between memory and the chip. Computing alone would take ${M_('F/P')}, moving alone ${M_('B/W')}. The kernel works through its tensors in small tiles: while the compute units work on the tiles already read, the next tiles are on their way from HBM, so the two overlap. Its time is therefore not their sum but the longer of the two:`,
              `设一个 kernel 要做 ${M_('F')} 个 FLOP，在显存和芯片之间搬运 ${M_('B')} 个字节。光算要 ${M_('F/P')}，光搬运要 ${M_('B/W')}。kernel 把张量分成小块来处理：计算单元算已经读进来的几块时，后面几块正在从 HBM 读进来，两件事重叠进行。所以耗时不是两者相加，而是取较长的那个：`),
       ()=>prX('t \\approx \\max\\left(\\frac{F}{P},\\ \\frac{B}{W}\\right)'),
       ()=>tx('The faster of the two finishes its share and then waits for the other. This is the time when both run at their limits; real kernels often fall short of them, and then take longer.','较快的那件事做完自己那份以后，只能空等另一件。这是计算和搬运都跑满上限时的耗时；实际的 kernel 常常跑不满，耗时只会更长。'))]},
 {h:()=>tx('Arithmetic intensity and the ridge point: operations per byte moved','算术强度和拐点：每读写 1 字节做几次运算'),parts:[
   prP(()=>tx(`Which side sets a kernel’s time comes down to comparing the computing time ${M_('F/P')} with the moving time ${M_('B/W')}. Multiplying both by ${M_('P/B')} turns the comparison into:`,
              `kernel 的耗时由哪一边决定，要比较计算时间 ${M_('F/P')} 和搬运时间 ${M_('B/W')}。两边都乘以 ${M_('P/B')}，比较就变成：`),
       ()=>prX('\\frac{F}{P} > \\frac{B}{W}\\;\\;⇔\\;\\;\\frac{F}{B} > \\frac{P}{W}'),
       ()=>tx('The left side depends only on the kernel, the right side only on the GPU.','左边只和 kernel 有关，右边只和 GPU 有关。'),
       ()=>tx(`The left side, ${M_('I = F/B')}, is the arithmetic intensity: how many FLOPs the kernel does for each byte it moves to or from memory. The right side, ${M_('I^{*} = P/W')}, is the ridge point: how many FLOPs the GPU can do in the time it takes to move one byte. In the chart of the last section the speed limit bends there from a slanted line to a flat one, hence the name; it is not an inflection point in the calculus sense. On an H100, with ${M_(`P = ${pk0}\\ \\text{TFLOP/s}`)} and ${M_(`W = ${bw0}\\ \\text{TB/s}`)}:`,
              `左边的 ${M_('I = F/B')} 叫算术强度（arithmetic intensity）：kernel 每从显存读写 1 字节，做多少个 FLOP。右边的 ${M_('I^{*} = P/W')} 叫拐点（ridge point）：GPU 在搬运 1 字节的时间里，能做多少个 FLOP。最后一节画出的速度上限在这里从斜线折成平线，所以叫拐点；它不是微积分里凹凸改变的那种拐点。H100 的 ${M_(`P = ${pk0}\\ \\text{TFLOP/s}`)}，${M_(`W = ${bw0}\\ \\text{TB/s}`)}：`),
       ()=>prX(tx(`I^{*} = \\frac{${pk0}\\ \\text{TFLOP/s}}{${bw0}\\ \\text{TB/s}} \\approx ${rid0}\\ \\text{FLOP/byte}`,`I^{*} = \\frac{${pk0}\\ \\text{TFLOP/s}}{${bw0}\\ \\text{TB/s}} \\approx ${rid0}\\ \\text{FLOP/字节}`)),
       ()=>tx(`When ${M_('I < I^{*}')}, moving takes longer than computing, and the kernel is memory-bound. When ${M_('I > I^{*}')}, computing takes longer, and it is compute-bound.`,
              `${M_('I < I^{*}')} 时，搬运比计算久，kernel 受显存带宽限制（memory-bound）；${M_('I > I^{*}')} 时，计算比搬运久，kernel 受算力限制（compute-bound）。`)),
   prP(()=>tx(`A kernel’s arithmetic intensity takes only counting its FLOPs and bytes. Take multiplying every number of a BF16 vector of length ${M_('n')} by the same constant. Each position takes one multiplication, ${M_('n')} FLOPs in all. It reads ${M_('n')} numbers and writes ${M_('n')}, at 2 bytes each: ${M_('2n')} bytes read and ${M_('2n')} written.`,
              `一个 kernel 的算术强度，数一数它的 FLOP 和字节就能得到。比如把一个长度为 ${M_('n')} 的 BF16 向量的每个数都乘以同一个常数：每个位置做 1 次乘法，一共 ${M_('n')} 个 FLOP；读入 ${M_('n')} 个数，写出 ${M_('n')} 个数，每个数 2 字节，也就是读 ${M_('2n')} 字节、写 ${M_('2n')} 字节。`),
       ()=>prX('I = \\frac{n}{2n + 2n} = \\frac{1}{4}'),
       ()=>tx(`It does not depend on ${M_('n')}, and it is below 1. This kernel is not a matrix multiplication and cannot use the tensor cores, so the compute it can use is lower than ${M_('P')} and so is its ridge, but still far above 1/4: however long the vector, this kernel is memory-bound.`,
              `它和 ${M_('n')} 无关，比 1 还小。这个 kernel 不是矩阵乘法，用不上 tensor core，能用的算力比 ${M_('P')} 低，拐点也更低，但仍远大于 1/4：向量再长，这个 kernel 也受显存带宽限制。`))]},
 {h:()=>tx('The roofline: the most FLOP/s a kernel can reach','roofline 图：kernel 最多能达到的 FLOP/s'),parts:[
   prP(()=>tx(`Knowing which side of the ridge a kernel is on, we also want to know how much of the GPU’s compute it puts to use. A kernel reaches ${M_('F/t')} FLOP/s. ${M_('P')} and ${M_('W')} are limits that real kernels often fall short of, so ${M_('t')} is at least ${M_('\\max(F/P,\\ B/W)')}. Since ${M_('F')} divided by ${M_('F/P')} is ${M_('P')} and ${M_('F')} divided by ${M_('B/W')} is ${M_('I \\cdot W')}, the most a kernel can reach is the smaller of the two rates:`,
              `知道了 kernel 在拐点哪一边，还想知道它用上了多少算力。kernel 达到的 FLOP/s 是 ${M_('F/t')}。${M_('P')} 和 ${M_('W')} 都是上限，实际的计算和搬运常常跑不满，所以 ${M_('t')} 至少是 ${M_('\\max(F/P,\\ B/W)')}。又因为 ${M_('F')} 除以 ${M_('F/P')} 是 ${M_('P')}，${M_('F')} 除以 ${M_('B/W')} 是 ${M_('I \\cdot W')}，kernel 最多能达到两个速度里较小的那个：`),
       ()=>prX('\\frac{F}{t} \\le \\min\\left(P,\\ I \\cdot W\\right)'),
       ()=>tx(`Drawn as a chart, this upper limit is a bent line that looks like a roof, hence the name roofline. Arithmetic intensities run from below 1 to the thousands and the FLOP/s reached span several orders of magnitude too, so the chart puts the arithmetic intensity ${M_('I')} across and the FLOP/s reached up, both on log scales. There ${M_('I \\cdot W')} is a straight slanted line, since ${M_('\\log(I \\cdot W) = \\log I + \\log W')}: the roof set by the memory bandwidth ${M_('W')}. The peak compute ${M_('P')} is a flat line: the roof set by compute. The two meet at ${M_('I = I^{*}')}, the ridge.`,
              `把这个上限画出来，是一条像屋顶的折线，所以叫 roofline 图。算术强度小的不到 1，大的上千，达到的 FLOP/s 也跨好几个数量级，所以横轴取算术强度 ${M_('I')}，纵轴取达到的 FLOP/s，两条轴都用对数刻度。这样 ${M_('I \\cdot W')} 是一条斜的直线，因为 ${M_('\\log(I \\cdot W) = \\log I + \\log W')}：这是显存带宽 ${M_('W')} 定的屋顶。峰值算力 ${M_('P')} 是一条水平线：这是算力定的屋顶。两段在 ${M_('I = I^{*}')} 相交，就是拐点。`)),
   {fig:[{html:()=>prRoofHTML(640),cap:()=>tx('The H100’s roofline: the slanted part is set by the memory bandwidth, the flat part by the peak compute; the dashed vertical line is the ridge.','H100 的 roofline：斜的一段由显存带宽决定，平的一段由峰值算力决定，竖虚线是拐点。'),ms:1e9}],cls:'rf-prfig',mount:prRoofMount},
   prP(()=>tx('Any kernel can be marked on the chart by its intensity and the FLOP/s it actually reaches. If the slower of its two jobs runs at its limit, the mark sits on the roof; if not, it sits below it, never above.',
              '每个 kernel 都可以按它的算术强度和实际达到的 FLOP/s 标在图上。较慢的那件事跑满了上限，这个点就正好在屋顶上；跑不满，它就落在屋顶下方，不会高过屋顶。'),
       ()=>tx(`Left of the ridge, ${M_('B/W')} is larger than ${M_('F/P')}, so at best ${M_('t = B/W')}: the time is set only by the bytes moved, whatever the FLOPs. Right of the ridge, at best ${M_('t = F/P')}: the time is set only by the FLOPs, whatever the bytes.`,
              `在拐点左边，${M_('B/W')} 比 ${M_('F/P')} 大，耗时最短也是 ${M_('t = B/W')}，只由搬运的字节数决定，和 FLOP 数无关。在拐点右边，耗时最短是 ${M_('t = F/P')}，只由 FLOP 数决定，和搬运的字节数无关。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S as it was (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount.
     The faint dots of the setup before the last change (prev, prevGpu) are a comparison of the moment, like the ▲ badges: not kept */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,sbTouched,seen64,seenPast,mmPast,lv,lvTried:[...lvTried],
    drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.B,S.p).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse([x.k,x.B,x.N,x.c,x.n,x.gpu,x.p].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p}; prev=null; prevGpu=null;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    seen64=o.seen64===true; seenPast=o.seenPast===true; mmPast=o.mmPast===true;
    lv=LV.some(x=>x.id===o.lv)?o.lv:'base'; lvTried=new Set(['base']); if(Array.isArray(o.lvTried)) o.lvTried.forEach(x=>{if(LV.some(l=>l.id===x)) lvTried.add(x);}); lvTried.add(lv);
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:2; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#rf-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#rf-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    fromSnap=false;
    fillRail();
    render();
    requestAnimationFrame(()=>{if(app) drawPlot();});
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#rf-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#rf-math',app).open=true;}},
  onResize(){if(app) drawPlot();},
  termTip(k,el){if(k==='dfx'){const f=el&&DFXT[el.getAttribute('data-term')]; return f?f():'';} return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#rf-plot',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#rf-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      if(v==='duo'){const d=duoOf(lv); vis.innerHTML=''; read.innerHTML=esc(tx(`Decode at batch ${DUO_B}: ${fTime(d.dec.t)}; prefill of ${fint(DUO_N)} tokens: ${fTime(d.pre.t)}`,`batch 为 ${DUO_B} 的 decode：${fTime(d.dec.t)}；${fint(DUO_N)} 个 token 的 prefill：${fTime(d.pre.t)}`)); return;}
      if(v==='map'){vis.innerHTML=''; read.innerHTML=esc(tx('On one H100, only the prefill reaches the flat roof; the other four wait for memory','在一张 H100 上，只有 prefill 到达平的屋顶；其余四个都在等显存')); return;}
      if(v==='ch'){const e=chEval(S.B,S.p); vis.innerHTML=miniBar(e.share,true); read.innerHTML=esc(tx(`H200, batch ${S.B}, ${precTxt(S.p)}: ${fSh(e.share)} of the peak FLOP/s (goal: 50%)`,`H200，batch ${S.B}，${precTxt(S.p)}：峰值算力的 ${fSh(e.share)}（目标是 50%）`)); return;}
      const r=R(); vis.innerHTML=miniBar(r.share);
      read.innerHTML=esc(tx(`${kname(S)} on the ${GPUS[r.gpu].name}: ${fTime(r.t)}, ${fSh(r.share)} of the peak FLOP/s`,`${GPUS[r.gpu].name} 上的${kname(S)}：${fTime(r.t)}，用上了峰值算力的 ${fSh(r.share)}`));}
  }
});
/* the mini result bar's picture: the share of the peak, with the goal's tick in the challenge */
function miniBar(share,goal){return `<div class="rf-dk"><i style="width:${Math.max(1,share*100).toFixed(1)}%"></i>${goal?'<b style="left:50%"></b>':''}</div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__rf=Object.freeze({
  ev:s=>deep(ev({...BASE,...s})), duo:id=>deep(duoOf(id)), chEval:(B,p)=>deep(chEval(B,p)), chBest, chChanges,
  drill:s=>deep(makeDrill(s)), xmmI, x32:()=>deep(X32()), kv:()=>deep(kvStep()),
  fmt:Object.freeze({time:fTime,I:fI,TF:fTF,share:fSh,bytes:fBy,flops:fFl}),
  consts:Object.freeze({GPUS:deep(GPUS),W_PAR,P_TOTAL,P_EMB,P_LAYERS,P_HEAD,KV_TOK,EW_N,BST:BST.slice(),MST:MST.slice(),NST:NST.slice(),CST:CST.slice(),BASE:{...BASE},
    STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...presetState(p)}))),CH:deep(CH),DUO_B,DUO_N,CODE}),
  state:()=>({...S,mode,step,pred,observed:{...observed},seen64,seenPast,mmPast,lv,lvTried:[...lvTried],chSubmitted,chReveal,stageBusy:stageT!==0,prev:prev?deep(prev):null,prevGpu})
});
})();
