/* kv-basics (Transformer series ③): Why does generation keep every token's keys and values?
   A live toy: a 2-layer transformer (d = 8, 2 heads of head_dim 4 per layer, RMSNorm, an 8 → 16 → 8 MLP) with made-up fixed
   weights writes the attention lab's sentence one word per decode step. Without a cache every step recomputes the keys and values
   of all tokens so far in both layers; with the cache only the newest token's. The recompute check (kept vs recomputed keys and
   values) is computed here, in floating point: with the causal mask it is exactly 0; with the mask off, layer 2 differs.
   The real-model part prices one decode step of Llama-3.1-8B on one H100 SXM with teasers/set-a.js's step model (same constants,
   same formulas), so the numbers agree with the home tile "Why is the first token slow and the rest fast?": a step with the cache is
   the tile's decode step, the first step of an answer with the cache is the tile's prefill; a step without the cache does the prefill's
   arithmetic but reads only the weights, since nothing is cached (review 8).
   Words (EN / ZH): decode step / decode 步; token-pass / token 前向 (one token through all layers); KV cache; causal mask / 因果 mask.
   window.__kvb is a read-only hook for tests/kv-basics_numbers.py and tests/kv-basics_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='kv-basics';

/* =====================================================================
   engine 1: the toy transformer
   ===================================================================== */
const NT=9, D=8, NH=2, HD=4, NL=2, FF=16, SEED=11;
const SCALE={tok:1,pos:0.5,q:0.6,k:0.6,v:0.6,o:0.5,w1:0.6,w2:0.4};
const WORDS={en:['<s>','The','cat','sat','on','the','mat','because','it'],zh:['<s>','小猫','坐','在','垫子','上','，','因为','它']};
const PER_TOK=NL*2*NH*HD;                 // numbers one token keeps in the toy's cache: 2 layers × K and V × 2 heads × 4 = 32
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* made-up, fixed weights: drawn once from a seeded generator and rounded to two decimals, in this order */
const W=(()=>{const rnd=mulberry(SEED), r2=v=>Math.round(v*100)/100;
  const M=(r,c,s)=>Array.from({length:r},()=>Array.from({length:c},()=>r2((rnd()*2-1)*s)));
  const w={tok:M(NT,D,SCALE.tok),pos:M(NT,D,SCALE.pos),layers:[]};
  for(let l=0;l<NL;l++) w.layers.push({q:M(D,D,SCALE.q),k:M(D,D,SCALE.k),v:M(D,D,SCALE.v),o:M(D,D,SCALE.o),w1:M(D,FF,SCALE.w1),w2:M(FF,D,SCALE.w2)});
  return w;})();
const mv=(x,M)=>{const n=M[0].length, out=new Array(n); for(let j=0;j<n;j++){let s=0; for(let i=0;i<x.length;i++) s+=x[i]*M[i][j]; out[j]=s;} return out;};
const rms=x=>{let s=0; for(const v of x) s+=v*v; const r=Math.sqrt(s/x.length+1e-6); return x.map(v=>v/r);};
/* One full forward pass over the first s tokens. Each layer: RMSNorm, q/k/v, attention per head (causal: token i sees tokens 0..i;
   mask off: every token sees all s), the output matrix, a residual add, RMSNorm, the MLP (ReLU), a residual add.
   The loops run in the same order for a token whether it is the newest or an earlier one, so with the causal mask a token's keys and
   values come out bit for bit the same in every later step: the check below prints a computed 0, not a typed one. */
const FWD=new Map();
function forward(s,mask){
  const key=s+'|'+(mask?1:0); if(FWD.has(key)) return FWD.get(key);
  let h=[]; for(let i=0;i<s;i++) h.push(W.tok[i].map((v,c)=>v+W.pos[i][c]));
  const K=[],V=[];
  for(let l=0;l<NL;l++){const P=W.layers[l];
    const a=h.map(rms), q=a.map(x=>mv(x,P.q)), k=a.map(x=>mv(x,P.k)), v=a.map(x=>mv(x,P.v));
    K.push(k); V.push(v);
    const nh=[];
    for(let i=0;i<s;i++){
      const last=mask?i:s-1, o=new Array(D).fill(0);
      for(let hh=0;hh<NH;hh++){const off=hh*HD, sc=[]; let m=-Infinity;
        for(let j=0;j<=last;j++){let d=0; for(let c=0;c<HD;c++) d+=q[i][off+c]*k[j][off+c]; d/=Math.sqrt(HD); sc.push(d); if(d>m) m=d;}
        let z=0; const e=sc.map(x=>{const y=Math.exp(x-m); z+=y; return y;});
        for(let j=0;j<=last;j++){const wgt=e[j]/z; for(let c=0;c<HD;c++) o[off+c]+=wgt*v[j][off+c];}}
      const ao=mv(o,P.o), h1=h[i].map((x,c)=>x+ao[c]);
      const b=rms(h1), f1=mv(b,P.w1).map(x=>x>0?x:0), f2=mv(f1,P.w2);
      nh.push(h1.map((x,c)=>x+f2[c]));}
    h=nh;}
  const r={K,V,h}; FWD.set(key,r); return r;
}
/* The recompute check at step n: for every earlier token j, its key and value as computed in step n against those computed in step
   j + 1, the step that took token j in (that is what a cache keeps). Largest absolute difference per layer and per token. */
function check(n,mask){
  const out={n,mask,per:[[],[]],max:[0,0],at:[-1,-1]};
  if(n<2) return out;
  const now=forward(n,mask);
  for(let j=0;j<n-1;j++){const first=forward(j+1,mask);
    for(let l=0;l<NL;l++){let m=0;
      for(let c=0;c<D;c++){m=Math.max(m,Math.abs(now.K[l][j][c]-first.K[l][j][c]),Math.abs(now.V[l][j][c]-first.V[l][j][c]));}
      out.per[l].push(m); if(m>out.max[l]){out.max[l]=m; out.at[l]=j;}}}
  return out;
}
/* for scale: the average size (ignoring the sign) of the numbers in one layer's keys and values at step n */
function absMean(n,mask,l){if(n<1) return 0; const f=forward(n,mask); let s=0,c=0; for(const M of [f.K[l],f.V[l]]) for(const r of M) for(const v of r){s+=Math.abs(v); c++;} return s/c;}
/* the work, counted: step s costs s token-passes without the cache, 1 with it */
const tri=n=>n>0?n*(n+1)/2:0;
function counts(n,cache){return {n,cache,passes:cache?n:tri(n),reads:cache?tri(n-1):0,nums:cache?n*PER_TOK:0,p100:cache?100:tri(100),stepCost:s=>cache?1:s};}

/* =====================================================================
   engine 2: Llama-3.1-8B on one H100 SXM, with teasers/set-a.js's step model (same constants and formulas)
   ===================================================================== */
const H100={peak:1979e12/2,bw:3.35e12,mem:80};      // dense BF16 = the datasheet's 1,979 TFLOPS "with sparsity" ÷ 2; 3.35 TB/s; 80 GB
const EFF={mfu:0.5,mbu:0.6};                         // assumed: 50% of peak FLOP/s, 60% of peak bandwidth (as set-a)
const LL={name:'Llama-3.1-8B',layers:32,hidden:4096,heads:32,kvHeads:8,headDim:128,vocab:128256,ffn:14336,maxCtx:131072};
const P_EMB=LL.vocab*LL.hidden, P_HEAD=LL.vocab*LL.hidden;
const P_LAYER=2*LL.hidden*LL.hidden+2*LL.hidden*LL.kvHeads*LL.headDim+3*LL.hidden*LL.ffn+2*LL.hidden;
const P_LAYERS=LL.layers*P_LAYER+LL.hidden, P_TOTAL=P_LAYERS+P_EMB+P_HEAD;
const KV_TOK=2*LL.layers*LL.kvHeads*LL.headDim*2;    // 131,072 bytes = 128 KiB per token
const W_READ=2*(P_LAYERS+P_HEAD);                    // bytes one step reads: every weight but the embedding table (a lookup)
const W_MEM=2*P_TOTAL;                               // bytes the weights take in GPU memory
const attnF=(a,b)=>2*LL.layers*LL.hidden*(b*(b+1)-a*(a+1));   // QKᵀ and AV FLOPs, all layers, for query positions a..b-1
function stepT(o){const fl=2*P_LAYERS*o.tok+2*P_HEAD*o.logits+o.attn, by=W_READ+KV_TOK*o.kv;
  const tc=fl/(H100.peak*EFF.mfu), tm=by/(H100.bw*EFF.mbu); return {t:Math.max(tc,tm),tc,tm,fl,by};}
/* context C = the tokens one decode step works on, the newest included. Both steps read the weights once (the GPU streams every
   multiplied parameter from memory once per step, however many tokens it computes).
   - With the cache the step computes the newest token and reads the keys and values of the other C − 1 from the cache: the tile's
     decode step after C − 1 tokens.
   - Without the cache nothing is kept, so nothing is read from a cache (r14, review 8 point 4): the step computes all C tokens again,
     the tile's prefill arithmetic, and its memory traffic is the weights alone. The keys and values it computes are activations, used
     and dropped within the step; like the other activations, their traffic is left out (the Fidelity card says so).
   - The first step of an answer with the cache is a prefill: it computes the prompt once and writes its keys and values to the
     cache, exactly the tile's prefill. */
const nocache=C=>stepT({tok:C,logits:1,attn:attnF(0,C),kv:0});
const prefill=P=>stepT({tok:P,logits:1,attn:attnF(0,P),kv:P});
const cached=C=>stepT({tok:1,logits:1,attn:attnF(C-1,C),kv:C-1});
function real(C){const w=cached(C), wo=nocache(C); return {C,kvTok:KV_TOK,kv:C*KV_TOK,w,wo,wMem:W_MEM};}
/* a whole answer: prompt P, answer A. With the cache: a prefill of the P prompt tokens, then one token per step.
   Without it: step k computes P + k − 1 tokens. */
function scen(P,Aa){const first=prefill(P).t; let w=first; for(let C=P+1;C<P+Aa;C++) w+=cached(C).t;
  let wo=0; for(let C=P;C<P+Aa;C++) wo+=nocache(C).t;
  return {P,A:Aa,first,firstNo:nocache(P).t,with:w,without:wo,avg:Aa>1?(w-first)/(Aa-1):0,endTok:P+Aa-1,endBytes:(P+Aa-1)*KV_TOK};}
const CTX=[256,512,1024,1536,2048,3072,4096,8192,16384,32768,65536,131072];
const CH_STOPS=CTX.slice(0,10), CH_START=4096, CH_GOAL=20;      // challenge: 20 tokens per second, 50 ms per token

/* =====================================================================
   formatting (every derived number is computed from the value as shown, R6.6)
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
/* a time: one decimal in ms below 1 s, two decimals in s from 1 s (both keep their trailing zero: "8.0 ms", "12.80 s") */
function msOf(sec){const v=sec*1e3; return v<999.95?{v:+v.toFixed(1),u:'ms'}:{v:+(v/1000).toFixed(2),u:'s'};}
function tS(sec){const m=msOf(sec); return m.u==='ms'?fixd(m.v,1)+' ms':fixd(m.v,2)+' s';}
const shownSec=sec=>{const m=msOf(sec); return m.u==='ms'?m.v/1000:m.v;};
/* tokens per second, from the time as shown: whole numbers from 100, one decimal from 1, two below 1 */
function tpsV(sec){const v=1/shownSec(sec); return v>=99.5?Math.round(v):(v>=0.995?+v.toFixed(1):+v.toFixed(2));}
function tpsS(sec){const v=tpsV(sec); return v>=99.5?fint(v):(v>=0.995?fixd(v,1):fixd(v,2));}
/* a whole answer's time: whole ms below 1 s, two decimals below 10 s, one below 60 s, then minutes */
function totV(sec){if(sec<0.9995) return {v:Math.round(sec*1e3),u:'ms'}; if(sec<9.995) return {v:+sec.toFixed(2),u:'s'}; if(sec<59.95) return {v:+sec.toFixed(1),u:'s'}; return {v:+(sec/60).toFixed(1),u:'min'};}
function totS(sec){const t=totV(sec); return t.u==='ms'?fint(t.v)+' ms':(t.u==='min'?fixd(t.v,1)+' min':fixd(t.v,sec<9.995?2:1)+' s');}
const totSec=sec=>{const t=totV(sec); return t.u==='ms'?t.v/1000:(t.u==='min'?t.v*60:t.v);};
/* a ratio of two shown values: one decimal below 10, whole numbers from 10 */
function ratioS(r){return r>=9.95?fint(r):fixd(r,1);}
/* numbers in prose (R6.20): rounded for reading, about two significant digits (7.5 ms, 270 ms, 13 s, 25 min); the exact values stay on
   the stage, the metric cards and Show the math (a tooltip that says how a number is made may give its parts) */
const fR=sec=>{const v=sec*1e3; if(v>=9950) return fint(Math.round(v/1e3))+'\u00a0s'; if(v>=950) return fixd(v/1e3,1)+'\u00a0s'; if(v>=99.5) return fint(Math.round(v/10)*10)+'\u00a0ms'; if(v>=9.95) return fint(Math.round(v))+'\u00a0ms'; return fixd(v,1)+'\u00a0ms';};
const totR=sec=>{if(sec<59.95) return fR(sec); const m=sec/60; return (m>=9.95?fint(Math.round(m)):fixd(m,1))+'\u00a0min';};
/* how many times as long, in words, from a ratio of shown values: "over 30 times" / "30 多倍" */
const ratioW=q=>{if(q<9.95) return tx(`${fixd(q,1)} times`,`${fixd(q,1)} 倍`); const f=q>=100?Math.floor(q/100)*100:Math.floor(q/10)*10;
  return q>f+1e-9?tx(`over ${fint(f)} times`,`${fint(f)} 多倍`):tx(`${fint(f)} times`,`${fint(f)} 倍`);};
/* memory: MiB below 1 GiB (one decimal unless whole), GiB from 1 GiB (two decimals unless whole) */
function memS(b){const g=b/A.GiB; if(g>=1) return (Number.isInteger(g)?fint(g):fixd(g,2))+' GiB'; const m=b/A.MiB; return (Number.isInteger(m)?fint(m):fixd(m,1))+' MiB';}
const gib1=b=>fixd(b/A.GiB,1)+' GiB';
/* decimal bytes, 3 significant digits (GB and MB are 10⁹ and 10⁶ bytes), shown next to GiB and MiB where bytes first become GiB (R6.12) */
const sig3=v=>fixd(v,v>=100?0:(v>=10?1:2));
const gbS=b=>b>=1e9?sig3(b/1e9)+'\u00a0GB':sig3(b/1e6)+'\u00a0MB';
/* R6.18: a count of more than 6 digits: 3 significant digits with M or B (the kit's helper; no exact-value hover, the owner's clarification of 2026-10-07) */
const bigN=n=>A.ui.big?A.ui.big(n):fint(n);
const bigS=n=>A.fmt.big?A.fmt.big(n):fint(n);
/* context lengths: 256, 1,000, 1.5K, 8K; "8K (8,192)" in sentences */
function ctxK(C){if(C>=1024&&C%512===0) return String(C/1024)+'K'; return fint(C);}
function ctxFull(C){const k=ctxK(C); return k===fint(C)?k:tx(`${k} (${fint(C)})`,`${k}（${fint(C)}）`);}
const ctxFullTok=C=>{const k=ctxK(C); return k===fint(C)?tx(`${k} tokens`,`${k} 个 token`):tx(`${k} (${fint(C)}) tokens`,`${k}（${fint(C)}）个 token`);};
const tokN=n=>tx(n===1?'1 token':`${fint(n)} tokens`,`${fint(n)} 个 token`);
const passN=n=>tx(n===1?'1 token-pass':`${fint(n)} token-passes`,`${fint(n)} 次 token 前向`);
const stepN=n=>tx(n===1?'1 decode step':`${fint(n)} decode steps`,`${fint(n)} 个 decode 步`);
const ctxTok=C=>tx(`${ctxK(C)} tokens`,`${ctxK(C)} 个 token`);
const diffS=v=>v===0?'0':(v<0.005?'<0.01':fixd(v,2));
const NB=s=>String(s).replace(/Llama-3\.1-8B/g,'Llama‑3.1‑8B').replace(/H100 SXM/g,'H100 SXM');
const wd=(j,lang)=>WORDS[lang||A.lang()][j];
const q=s=>tx(`“${s}”`,`“${s}”`);
const J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const cite=(url,txt)=>`<a href="${url}" target="_blank" rel="noopener">${txt}</a>`;

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const DEFAULT={view:'toy',n:0,cache:false,mask:true,ctx:8192};
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let saved=null, chSubmitted=false, chReveal=false, drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false, tzN=6, sbTouched=false, maskBack=false;
const shownPrev={};
/* another lab by its short name, with its whole question on hover (the kit's helper) */
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
/* the kit's structural definitions (A.terms, CONTRACT §11): the same words on every lab of the site; this lab adds its model after them */
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const TIP={
 decode:()=>tx('A decode step is one run of the model that produces one new token. The engine repeats it until the answer is complete. When the prompt is long, the first run takes in all of it and is usually called prefill; here the prompt is only <s>, so that run is decode step 1.',
               'decode 步是把模型跑一遍、生成一个新 token 的过程，引擎一步接一步地跑，直到回答写完。prompt 很长时，第一次运行要读入整个 prompt，通常叫 prefill；这里的 prompt 只有 <s>，所以它就是第 1 个 decode 步。'),
 start:()=>tx('<s> is the start token: a fixed first token that the model takes in before any word, so that decode step 1 has something to run on.',
              '<s> 是起始 token：模型在读入任何词之前先读入的一个固定 token，这样第 1 个 decode 步才有东西可算。'),
 pass:()=>tx('A token-pass is one token going through all the layers of the model once.','一次 token 前向是一个 token 从第一层走到最后一层，把所有层都过一遍。'),
 mask:()=>kt('mask')+tx(' So a later token cannot change what an earlier token computes, in any layer.','所以后面来的 token 改变不了前面的 token 在任何一层算出的东西。'),
 enc:()=>tx('An encoder such as BERT takes in a whole text at once, with no causal mask: every token attends to every other token, earlier and later. It does not write text token by token.',
            'BERT 这类 encoder 一次读入整段文字，没有因果 mask：每个 token 都会看其他所有 token，前面的和后面的都看。它不一个 token 一个 token 地写文字。'),
 kv:()=>kt('kvcache')+tx(' It grows by one token’s worth with every decode step.','每个 decode 步它都多存一个 token 的量。'),
 layer:()=>kt('layer')+tx(' The small model here has 2 layers; Llama-3.1-8B has 32.','这里的小模型有 2 层，Llama-3.1-8B 有 32 层。'),
 /* units: GiB is the kit's; KiB and MiB follow its wording (R6.12) */
 gib:()=>kt('gib'),
 kib:()=>kt('kib'),
 mib:()=>kt('kib')+tx(' A MiB is about 5% more than an MB (10^6 bytes).','MiB 比 MB（10^6 字节）大约 5%。'),
 /* review 8 points 2 and 5: why a step takes the slower of memory and arithmetic, and what "reading the weights" is */
 overlap:()=>tx('A GPU reads memory and does arithmetic at the same time: while it computes with one chunk of the weights, it is already reading the next. So a step takes about as long as the slower of the two, not their sum. This is this lab’s estimate; real kernels overlap the two only in part.',
                'GPU 读显存和做计算是同时进行的：它用已读到的一段权重计算时，已经在读下一段。所以一步的耗时约等于两者中较慢的那个，而不是两者相加。这是这个 Lab 的估计；真实 kernel 只能让两者部分重叠。'),
 wread:()=>tx(`In every decode step the GPU streams every parameter it multiplies (all but the embedding table: ${gib1(W_READ)} for Llama-3.1-8B) from GPU memory into its compute units once. That one read serves every token the step computes, so it costs the same for 1 token or 100, while the arithmetic grows with the tokens. A step without the cache recomputes all the earlier tokens and still reads the weights only once.`,
              `每个 decode 步，GPU 都要把参与乘法的全部参数（embedding 表以外的全部参数，Llama-3.1-8B 是 ${gib1(W_READ)}）从显存读进计算单元一次。这一次读，供这一步算的所有 token 共用：算 1 个 token 和算 100 个 token 读的一样多，计算量却随 token 数增长。不用 cache 的一步要把之前所有 token 重算一遍，权重也只读一次。`),
 /* after the kit's definition, which says only "head": this lab never defines a query head (r12 site #8) */
 kvh:()=>(A.terms&&A.terms.kvhead?A.terms.kvhead():'')+tx(' Llama-3.1-8B’s 32 heads per layer share 8 KV heads, one for every 4 heads, so each token keeps 8 keys and 8 values per layer; the lab “GQA and MLA” (GQA is grouped-query attention, MLA multi-head latent attention) shows how the sharing works.',
            'Llama-3.1-8B 每层 32 个 head 共用 8 个 KV head，每 4 个共用一个，所以每个 token 每层存 8 个 key 和 8 个 value；怎么共用，由“GQA 和 MLA”（GQA 即 grouped-query attention，MLA 即 multi-head latent attention）这个 Lab 来讲。'),
 prefix:()=>tx('Prefix caching keeps the KV cache of earlier requests, so that a new request that starts with the same tokens reuses it instead of computing that part again.',
               'prefix caching 会留着之前请求的 KV cache，开头相同的新请求直接复用，不再重算这一段。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'lt200',t:{en:'Under 200',zh:'不到 200 次'}},{v:'200',t:{en:'200 to 2,000',zh:'200 到 2,000 次'}},{v:'2000',t:{en:'2,000 to 7,500',zh:'2,000 到 7,500 次'}},{v:'gt7500',t:{en:'More than 7,500',zh:'超过 7,500 次'}}];
const predOf=x=>x<200?'lt200':(x<2000?'200':(x<=7500?'2000':'gt7500'));
/* before the learner answers part 2, the toy does not write (R3.8: no result before the answer) */
const preLock=()=>mode==='guided'&&!pred&&step<=1;
/* the 100-token count is the prediction's answer: hidden until part 3 shows it */
const hide100=()=>mode==='guided'&&(step<2||(step===2&&!observed[2]));
const showCheck=()=>(mode!=='guided'||step>=3)&&S.n>=2;
const sumTxt=n=>n<=5?Array.from({length:n},(_,i)=>i+1).join(' + '):`1 + 2 + … + ${n}`;
const genCtl=()=>`<div class="inline-ctl kvb-gen" id="kvb-ic-gen"><div class="kvb-genrow"><button type="button" class="btn sm" data-act="gen">${tx('Write the next word','生成下一个词')}</button><button type="button" class="btn ghost sm" data-act="restart">${tx('Start over','从头开始')}</button></div>`+
  `<div class="kvb-genout"><span>${tx('Decode steps done','已完成的 decode 步')}</span><output class="mono" data-out="n"></output></div><p class="small muted kvb-genend" data-out="end" hidden></p></div>`;
const segHTML=(id,key,label,opts,val,inline)=>`<div class="${inline?'inline-ctl':'ctl'}" id="${id}"><span class="lbl">${esc(label)}</span><div class="seg-ctl" role="group" aria-label="${esc(label)}" data-key="${key}">${opts.map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${String(v)===String(val)}">${esc(t)}</button>`).join('')}</div></div>`;
const maskCtl=inline=>segHTML(inline?'kvb-ic-mask':'kvb-ctl-mask-seg','mask',tx('Causal mask','因果 mask'),[['1',tx('On','开')],['0',tx('Off','关')]],S.mask?'1':'0',inline);
const cacheCtl=inline=>segHTML(inline?'kvb-ic-cache':'kvb-ctl-cache-seg','cache',tx('KV cache','KV cache'),[['0',tx('Off','不用')],['1',tx('On','用')]],S.cache?'1':'0',inline);
const ctxCtl=(id,stops,inline)=>{const i=nearest(stops,S.ctx);
  return A.ui.range({id,key:stops===CTX?'ctx':'chctx',label:tx('Context','上下文长度'),min:0,max:stops.length-1,value:i,out:ctxTok(S.ctx),inline});};
function nearest(arr,v){let bi=0,bd=Infinity; arr.forEach((a,i)=>{const d=Math.abs(Math.log(a)-Math.log(v)); if(d<bd){bd=d; bi=i;}}); return bi;}

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'Step 9 needs the keys and values of all 9 tokens: compute them again, or keep them?',zh:'第 9 步需要全部 9 个 token 的 key 和 value：重新算，还是存下来？'},enter:()=>({view:'toy',n:0,cache:false,mask:true}),
  body:()=>tx(`<p>An LLM writes text one token at a time: each ${DEF('decode step',TIP.decode())} runs the model once and adds one token. Here a small model takes in the ${DEF('start token &lt;s&gt;',TIP.start())} and then writes “${esc(WORDS.en.slice(1).join(' '))}”, one word per step; in this lab one word is one token.</p><p>At decode step 9 the model runs on “${esc(wd(8,'en'))}” to write the next word. Every ${DEF('layer',TIP.layer())} then needs the keys and values of all 9 tokens, and all but the newest were already computed in earlier steps. Compute them again, or keep them?</p>`,
              `<p>LLM 一次生成一个 token：每个 ${DEF('decode 步',TIP.decode())}把模型跑一遍，多生成一个 token。这里有一个小模型，先读入${DEF('起始 token &lt;s&gt;',TIP.start())}，再一步一个词地生成“${esc(WORDS.zh.slice(1).join(' '))}”；在这个 Lab 里，一个词或一个标点就是一个 token。</p><p>第 9 个 decode 步，模型读入“${esc(wd(8,'zh'))}”，要生成下一个词。这时每一${DEF('层',TIP.layer())}都需要全部 9 个 token 的 key 和 value，除了最新的那个，其余的在之前的步里都算过。是重新算一遍，还是存下来？</p>`)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: keeping nothing, how many token-passes for 100 tokens?',zh:'先猜：什么都不存，生成 100 个 token 要多少次 token 前向？'},enter:()=>({view:'toy',n:0,cache:false,mask:true}),gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Suppose the model keeps nothing, so every decode step computes every key and value it needs from scratch. How many ${DEF('token-passes',TIP.pass())} does it take to write 100 tokens after &lt;s&gt;?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the count.</p>':''}`,
             `<p>假设模型什么都不存，每个 decode 步都从头算出它需要的每个 key 和 value。在 &lt;s&gt; 后面生成 100 个 token，一共要做多少次 ${DEF('token 前向',TIP.pass())}？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'steps',title:{en:'One step at a time',zh:'一步一步生成'},lead:{en:'Write word by word and watch the token-passes add up.',zh:'一个词一个词地生成，看 token 前向怎么累加。'},enter:()=>({view:'toy',cache:false,mask:true,n:observed[2]?Math.max(S.n,4):0}),expect:()=>S.n>=4,showMe:{n:9},label:{en:'Observed',zh:'观察到'},
  body:()=>tx('<p>The model keeps nothing, so each decode step runs every token so far through both layers again. Write the sentence word by word and watch the token-passes add up; each token-pass fills one cell in each layer’s grid. Write at least 4 words to go on.</p>',
              '<p>模型什么都不存，所以每个 decode 步都要把目前所有的 token 重新过一遍两层。一个词一个词地生成这句话，看 token 前向的次数怎么累加；每一次 token 前向，在每一层的格子里各填一格。至少生成 4 个词，才能进入下一部分。</p>')+genCtl(),
  obs:()=>obsSteps()},
 {id:'check',title:{en:'Do earlier keys and values change?',zh:'之前的 K 和 V 会变吗？'},lead:{en:'Compare earlier tokens’ keys and values across steps, then switch off the causal mask.',zh:'比一比之前 token 的 key 和 value 各步变没变，再关掉因果 mask。'},enter:()=>({view:'toy',n:9,cache:false,mask:true}),expect:()=>!S.mask,always:true,showMe:{mask:false},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>By step 9 the model has computed the keys and values of “&lt;s&gt;” once in every step. The row marked ${J('chk','“change”')} under each layer in the chart compares each earlier token’s key and value in step 9 with those from its first step.</p><p>Now switch the ${DEF('causal mask',TIP.mask())} off, so that every token also looks at later tokens, as in ${DEF('an encoder such as BERT',TIP.enc())}, and compare again. When “${esc(wd(1,'en'))}” was first computed, in step 2, no later token existed; in step 9 it would also see every token after it.</p>`,
              `<p>到第 9 步，“&lt;s&gt;”的 key 和 value 每一步都算了一次。图中每层下面标着${J('chk','“变化”')}的那一行，把之前每个 token 在第 9 步算出的 key 和 value，和它第一次算出时的比较。</p><p>现在关掉${DEF('因果 mask',TIP.mask())}，让每个 token 也能看到后面的 token（${DEF('BERT 这类 encoder',TIP.enc())} 就是这样），再比较一次。“${esc(wd(1,'zh'))}”第一次算出是在第 2 步，那时它后面还没有 token；到第 9 步，它会看到后面所有的 token。</p>`)+maskCtl(true),
  obs:()=>obsCheck()},
 {id:'keep',title:{en:'Keep them',zh:'把它们存下来'},lead:{en:'Compute each key and value once and keep it: turn on the KV cache.',zh:'每个 key 和 value 只算一次、存下来：打开 KV cache。'},enter:()=>({view:'toy',n:9,cache:false,mask:true}),expect:()=>S.cache,always:true,showMe:{cache:true},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>So each token’s key and value can be computed once, in the step that takes the token in, and kept for every later step. That store, a key and a value per token in every layer and head, is the ${DEF('KV cache',TIP.kv())}. Turn it on:</p>`,
              `<p>所以每个 token 的 key 和 value 只要在读入它的那一步算一次，之后每一步都接着用。这些存下来的 key 和 value（每个 token 在每一层、每个 head 上各一份）就是 ${DEF('KV cache',TIP.kv())}。把它打开：</p>`)+cacheCtl(true),
  obs:()=>obsKeep()},
 {id:'real',title:{en:'Time and memory on Llama-3.1-8B',zh:'在 Llama-3.1-8B 上：时间和显存'},lead:{en:'At full size: 128 KiB per token, and much shorter steps with the cache.',zh:'全尺寸下每个 token 占 128 KiB，用 cache 的步短得多。'},enter:()=>({view:'real',ctx:8192}),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Per token, Llama-3.1-8B keeps a key and a value for each of its ${DEF('KV heads',TIP.kvh())} in every layer: ${fint(KV_TOK/1024)} ${DEF('KiB',TIP.kib())} in BF16.</p><p>The GPU reads memory and computes at the same time, so ${DEF('a decode step takes about as long as the slower of the two',TIP.overlap())}. Compare it with and without the cache:</p>`,
              `<p>Llama-3.1-8B 每个 token 在每一层、每个 ${DEF('KV head',TIP.kvh())} 上各存一个 key 和一个 value，BF16 下一共 ${fint(KV_TOK/1024)} ${DEF('KiB',TIP.kib())}。</p><p>GPU 读显存和做计算是同时进行的，所以${DEF('一个 decode 步的耗时约等于两者中较慢的那个',TIP.overlap())}。比较用和不用 cache 时的一步：</p>`)+ctxCtl('kvb-ic-ctx',CTX,true),
  obs:()=>obsReal()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},enter:()=>({view:'real',ctx:8192}),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);

function obsSteps(){
  const n=S.n, c=counts(n,S.cache);
  if(S.cache) return tx(`<p><b>${n===0?'No decode step done yet with the KV cache':`After ${stepN(n)} with the KV cache: ${passN(c.passes)}`}</b>${n>1?', one per step':''}.</p><p>This part counts the work without the cache: turn it off under ${J('controls','Settings')}.</p>`,
                        `<p><b>${n===0?'用 KV cache，还没完成任何 decode 步':`用 KV cache 时，${stepN(n)}一共 ${passN(c.passes)}`}</b>${n>1?'，每步一次':''}。</p><p>这一部分数的是不用 cache 时的工作量：在“${J('controls','自由调参')}”里把它关掉。</p>`);
  /* R6.20: the result and the 100-token count in words and one number each; the sums are in the metric card's tooltip and Show the math */
  const head=n===0?tx('<b>No decode step done yet, so no token-pass so far.</b>','<b>还没完成任何 decode 步，所以还没有 token 前向。</b>')
    :(n===1?tx('<b>After 1 decode step: 1 token-pass.</b>','<b>1 个 decode 步之后，一共做了 1 次 token 前向。</b>')
    :tx(`<b>After ${stepN(n)}: ${passN(c.passes)}.</b>`,`<b>${stepN(n)}之后，一共做了 ${passN(c.passes)}。</b>`));
  let pf='';
  if(pred){const right=predOf(tri(100)), pt=esc(L(PRED.find(p=>p.v===pred).t)), rt=esc(L(PRED.find(p=>p.v===right).t));
    pf=pred===right?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
                   :tx(`<p>You predicted “${pt}”; the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);}
  return tx(`<p>${head}</p><p>Writing 100 tokens this way takes ${fint(tri(100))} token-passes.</p>${pf}<p>Why: step n runs all n tokens so far, so the total grows with the square of n. If each step ran only its newest token, 100 tokens would take 100.</p>`,
            `<p>${head}</p><p>这样生成 100 个 token 要 ${fint(tri(100))} 次 token 前向。</p>${pf}<p>原因：第 n 步要把目前的 n 个 token 全部过一遍，所以总次数随 n 的平方增长。如果每步只算最新的那个 token，100 个 token 只要 100 次。</p>`);
}
function obsCheck(){
  const n=S.n;
  if(n<2) return n===0?tx(`<p><b>No decode step done yet, so there is no earlier key or value to compare.</b> Write at least two words under ${J('controls','Settings')}.</p>`,
                          `<p><b>还没完成任何 decode 步，没有之前的 key 和 value 可比。</b>在“${J('controls','自由调参')}”里至少生成两个词。</p>`)
                     :tx(`<p><b>Only 1 decode step done, so there is no earlier key or value to compare yet.</b> Write at least two words under ${J('controls','Settings')}.</p>`,
                          `<p><b>只完成了 1 个 decode 步，还没有之前的 key 和 value 可比。</b>在“${J('controls','自由调参')}”里至少生成两个词。</p>`);
  const c=check(n,S.mask), d1=diffS(c.max[0]), d2=diffS(c.max[1]);
  if(S.mask) return tx(`<p><b>No earlier key or value changed.</b> ${d1===d2?`In both layers the largest change is ${d1}.`:`The largest change is ${d1} in layer 1 and ${d2} in layer 2.`}</p><p>Why: a token enters layer 1 as its own input vector, which never changes. With the causal mask, its layer-1 output mixes only itself and earlier tokens, so that output is fixed too, and the same holds in every later layer. Its keys and values are computed from these fixed vectors, so later steps get exactly the same numbers, and keeping them loses nothing.</p>`,
                          `<p><b>之前的 key 和 value 一个都没变。</b>${d1===d2?`两层的最大变化都是 ${d1}。`:`第 1 层的最大变化是 ${d1}，第 2 层是 ${d2}。`}</p><p>原因：一个 token 进入第 1 层时就是它自己的输入向量，永远不变。有因果 mask 时，它在第 1 层的输出只混合它自己和之前的 token，所以也不会变；后面每一层都是这样。它每一层的 key 和 value 都由这些不变的向量算出，所以之后每一步算出的数值一模一样，存下来不会丢任何东西。</p>`);
  /* R6.20: one number in the sentence (the largest change, on the stage's "change" row); its size against the numbers themselves in
     words, chosen by the ratio at this step (from 1.0 at step 2 to 1.7), and the average itself on hover */
  const at=c.at[1]>=0?wd(c.at[1]):'', mean=absMean(n,S.mask,1), sz=fixd(mean,1), l1=c.max[0]===0, qs=c.max[1]/mean;
  const tipSz=tx(`At step ${n}, the numbers in layer 2’s keys and values average ${sz}, ignoring sign.`,`第 ${n} 步时，第 2 层 key 和 value 里的数值，不计正负号，平均是 ${sz}。`);
  const avgE=DEF('the average size of the numbers in its keys and values',tipSz), avgZ=DEF('这一层 key 和 value 里数值的平均大小',tipSz);
  const cmp=qs>=1.05?tx(`larger than ${avgE}`,`比${avgZ}还大`):(qs>=0.95?tx(`about as large as ${avgE}`,`和${avgZ}差不多大`):tx(`more than half of ${avgE}`,`超过${avgZ}的一半`));
  return tx(`<p><b>With the mask off, layer 2’s earlier keys and values changed; ${l1?'layer 1’s did not':`layer 1’s changed by up to ${d1}`}.</b> In layer 2 the largest change is <span class="badc">${d2}</span> (at “${esc(at)}”), ${cmp}.</p><p>Why: each new token now changes the earlier tokens’ layer-1 outputs, and layer 2 computes its keys and values from them. Layer 1’s come from each token’s own input, so from layer 2 on a kept key or value would be stale.</p>`,
            `<p><b>关掉 mask 后，第 2 层之前 token 的 key 和 value 变了，${l1?'第 1 层的没变':`第 1 层的最多变了 ${d1}`}。</b>第 2 层变化最大的是 <span class="badc">${d2}</span>（在“${esc(at)}”上），${cmp}。</p><p>原因：现在每来一个新 token，前面 token 在第 1 层的输出就会变，而第 2 层的 key 和 value 正是从这个输出算出来的。第 1 层的 key 和 value 直接来自每个 token 自己的输入，所以从第 2 层起，存下来的 key 和 value 就过期了。</p>`);
}
function obsKeep(){
  const n=S.n, last=wd(Math.max(0,n-1));
  if(n<1) return tx(`<p><b>No decode step done yet</b>, so nothing is computed or kept. Write some words under ${J('controls','Settings')}.</p>`,`<p><b>还没完成任何 decode 步</b>，所以什么都没算，也没存。在“${J('controls','自由调参')}”里先生成几个词。</p>`);
  const back=maskBack&&S.mask?tx(' (The causal mask is back on.)','（因果 mask 已重新打开。）'):'';
  if(!S.cache) return n===1?tx(`<p><b>Without the KV cache, step 1 ran 1 token-pass.</b>${back}</p>`,`<p><b>不用 KV cache，第 1 步做了 1 次 token 前向。</b>${back}</p>`)
    :tx(`<p><b>Without the KV cache, step ${n} ran every token so far again: ${passN(n)}.</b>${back}</p>`,`<p><b>不用 KV cache，第 ${n} 步重算了目前所有的 token：${passN(n)}。</b>${back}</p>`);
  const warn=S.mask?'':tx(` With the causal mask off, though, the kept keys and values from layer 2 on are stale: the model no longer computes what it would without the cache.`,`不过因果 mask 关掉了，从第 2 层起存下来的 key 和 value 已经过期：模型算出的结果和不用 cache 时不一样了。`);
  const others=n===2?tx('those of “<s>”','“<s>”的'):tx('the earlier tokens’','之前 token 的');
  /* R6.20: the result (one pair), then the trend with the 100-token pair, then the price with the per-token size; the counts of all
     the steps are on the metric cards, the product 2 × 2 × 2 × 4 in the cache card's tooltip and Show the math */
  const work=n===1?tx(`<b>Step 1 runs 1 token-pass with or without the KV cache</b>: nothing is kept yet, so it computes the keys and values of “${esc(last)}” and keeps them. From the next step on, each step runs only its newest token.`,
                      `<b>第 1 步用不用 KV cache 都只做 1 次 token 前向</b>：cache 里还没有东西，它在每层算出“${esc(last)}”的 key 和 value，然后存下来。从下一步起，每步只算最新的那个 token。`)
    :tx(`<b>With the KV cache, step ${n} ran 1 token-pass instead of ${n}</b>: it computed only the keys and values of “${esc(last)}” and read ${others} from the cache.`,
        `<b>用 KV cache 后，第 ${n} 步只做 1 次 token 前向，而不是 ${n} 次</b>：每层只算“${esc(last)}”的 key 和 value，${others}从 cache 里读。`);
  return tx(`<p>${work}${warn}</p><p>So the work grows in step with the length, not its square: 100 tokens take 100 token-passes, not ${fint(tri(100))}.</p><p>The price is memory: every token keeps a key and a value for each head in each layer, ${PER_TOK} numbers here, and the cache grows with every word written.</p>`,
            `<p>${work}${warn}</p><p>所以工作量随长度线性增长，而不是平方增长：100 个 token 只要 100 次，而不是 ${fint(tri(100))} 次。</p><p>代价是显存：每个 token 在每一层、每个 head 上都要存一个 key 和一个 value，这里一共 ${PER_TOK} 个数值；每多写一个词，cache 就多一份。</p>`);
}
/* how the two times of one decode step are computed, from the values as shown (CONTRACT §9: a "how" link answers on hover).
   Review 8 (points 2 and 4): the GPU reads memory and computes at the same time, so a step takes the slower of the two; the tip
   gives both terms of both steps and says which one decides each time. */
const ms2=t=>fixd(t*1e3,2)+'\u00a0ms';
function howTip(C){
  const r=real(C), GB=v=>fixd(v/1e9,3)+'\u00a0GB', TF=v=>fixd(v/1e12,v<1e12?4:3)+'\u00a0TFLOP';
  const pk=fixd(H100.peak*EFF.mfu/1e12,2), bw=fixd(H100.bw*EFF.mbu/1e12,2), pk0=fixd(H100.peak/1e12,1), bw0=fixd(H100.bw/1e12,2);
  const woMem=r.wo.tm>=r.wo.tc;
  return tx(`The GPU reads memory and does arithmetic at the same time, so a decode step takes about as long as the slower of the two: memory at ${bw} TB/s (60% of the H100 SXM’s ${bw0}), arithmetic at ${pk} TFLOP/s (50% of its ${pk0}). With the cache, at ${fint(C)} tokens the step reads ${GB(r.w.by)} of weights and cache (${ms2(r.w.tm)}) and does ${TF(r.w.fl)} of arithmetic (${ms2(r.w.tc)}), so memory decides: ${tS(r.w.t)}. Without it, the step reads only the weights, ${GB(r.wo.by)} (${ms2(r.wo.tm)}), but does ${TF(r.wo.fl)} (${ms2(r.wo.tc)}), so ${woMem?'memory':'arithmetic'} decides: ${tS(r.wo.t)}. This is this lab’s estimate.`,
            `GPU 读显存和做计算是同时进行的，所以一个 decode 步的耗时约等于两者中较慢的那个：读显存按 ${bw} TB/s 算（H100 SXM 峰值 ${bw0} 的 60%），计算按 ${pk} TFLOP/s 算（峰值 ${pk0} 的 50%）。用 cache、上下文 ${fint(C)} 个 token 时，这一步要读 ${GB(r.w.by)} 的权重和 cache（${ms2(r.w.tm)}），算 ${TF(r.w.fl)}（${ms2(r.w.tc)}），所以读显存决定耗时：${tS(r.w.t)}。不用 cache 时，这一步只读权重 ${GB(r.wo.by)}（${ms2(r.wo.tm)}），却要算 ${TF(r.wo.fl)}（${ms2(r.wo.tc)}），所以${woMem?'读显存':'计算'}决定耗时：${tS(r.wo.t)}。这是这个 Lab 的估计。`);
}
/* a size with its unit defined on hover (R6.12): GiB is the kit's definition, MiB this lab's */
const memD=b=>memS(b).replace(/(GiB|MiB)$/,u=>DEF(u,u==='GiB'?TIP.gib():TIP.mib()));
/* part 6 (R6.20): the result is the two step times, rounded, as one pair; the reason in words, chosen by the state (R6.10); the price
   is the cache's size. The exact times are on the speed bars and the metric cards, the terms behind them in "How these are computed"
   and Show the math, the size of one read of the weights in the tooltip on "reads the weights". */
function obsReal(){
  const C=S.ctx, r=real(C), k=shownSec(r.wo.t)/shownSec(r.w.t), same=k<1.05;
  /* the share of the step's reads that is the cache: tiny up to 16K, a fifth to a third at 32K and 64K, over half at 128K, where the
     cache also outgrows the weights */
  const big=r.kv>r.wMem, kvShare=KV_TOK*(C-1)/r.w.by, half=kvShare>0.5, slow=kvShare>0.15;
  const RW=DEF(tx('reads the weights','读一遍权重'),TIP.wread());
  const how=`${tx(' ','')}<button type="button" class="linkbtn kvb-how" data-jump="math" data-tipx="${esc(howTip(C))}">${tx('How these are computed','这两个耗时是怎么算的')}</button>`;
  const at=tx(`At ${ctxTok(C)} of context`,`上下文 ${ctxTok(C)} 时`);
  const head=same?tx(`<p><b>${at}, one decode step takes about ${fR(r.w.t)} with or without the KV cache, by this lab’s estimate.</b>${how}</p>`,`<p><b>${at}，按这个 Lab 的估计，一个 decode 步用不用 KV cache 都约 ${fR(r.w.t)}。</b>${how}</p>`)
    :tx(`<p><b>${at}, one decode step takes about ${fR(r.w.t)} with the KV cache and about ${fR(r.wo.t)} without it, by this lab’s estimate.</b>${how}</p>`,`<p><b>${at}，按这个 Lab 的估计，一个 decode 步用 KV cache 约 ${fR(r.w.t)}，不用约 ${fR(r.wo.t)}。</b>${how}</p>`);
  const price=tx(`<p>The price is memory: the cache now holds ${memD(r.kv)} (${gbS(r.kv)})${big?', more than all the weights':', and it grows with every token'}.</p>`,
                 `<p>代价是显存：这时 cache 占 ${memD(r.kv)}（${gbS(r.kv)}）${big?'，比全部权重还大':'，而且每多一个 token 就多一份'}。</p>`);
  /* review 8 point 4: both steps read the weights once; say what else each step does and which term decides its time */
  if(same) return head+tx(`<p>Why: every step ${RW} once, however many tokens it computes. At this short context, recomputing every token and reading the cache both take less time than that, so the cache saves nothing yet; the gap opens as the context grows.</p>`,
                          `<p>原因：不管算多少个 token，每一步都要${RW}。上下文这么短时，重算全部 token 和读 cache 都比读权重快，所以 cache 暂时省不了什么；上下文越长，差距越大。</p>`)+price;
  const withWhy=half?tx('With the cache it computes one token, but the cache it reads is now long: over half of the step goes to reading it.','用 cache 时只算一个 token，但要读的 cache 已经很长：一步里一半以上的时间花在读它上。')
    :(slow?tx('With the cache it computes one token, so memory decides, and its time grows only slowly as the cache gets longer.','用 cache 时只算一个 token，所以读显存决定耗时；cache 变长，耗时也只慢慢增加。')
    :tx('With the cache it computes one token, so memory decides and its time barely grows.','用 cache 时只算一个 token，所以读显存决定耗时，几乎不随上下文变长。'));
  return head+tx(`<p>Why: every step ${RW} once, however many tokens it computes. ${withWhy} Without it, the step recomputes the whole context, so arithmetic decides and the time grows with the context.</p>`,
                 `<p>原因：不管算多少个 token，每一步都要${RW}。${withWhy}不用 cache 时要把整段上下文重算一遍，所以计算决定耗时，而且随上下文增长。</p>`)+price;
}
/* part 7 (R6.20): the principles in words, with one number where it helps (128 KiB); the misconception about speed says the trend and
   one ratio, the one the stage's badge shows at 8K; no line longer than before */
function remember(){
  const r=real(8192), top=real(LL.maxCtx), q=shownSec(r.wo.t)/shownSec(r.w.t);
  const topE=top.kv>top.wMem?'more than its weights':'still less than its weights', topZ=top.kv>top.wMem?'比全部权重还大':'仍比全部权重小';
  return tx(`<ul><li>Each decode step needs the keys and values of every token so far, in every layer. With the causal mask they never change, so they are computed once and kept: that is the KV cache.</li><li>Without it, work grows with the length squared; with it, linearly.</li><li>It costs memory. Each token takes<div class="kvb-fx">${A.ui.math('\\text{KV cache per token} = 2 \\times \\text{layers} \\times \\text{KV heads} \\times \\text{head_dim} \\times \\text{bytes per number}')}</div>The 2 counts a key and a value. For Llama-3.1-8B in BF16 that is ${fint(KV_TOK/1024)} KiB per token, and at ${ctxTok(LL.maxCtx)} ${topE}.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “The KV cache stores the model’s outputs.” It stores every layer’s keys and values, not the words written.</p><p>✗ “Without a KV cache it is only a little slower.” Only at the shortest contexts. Without it each step recomputes the whole context, so at ${ctxTok(r.C)} it takes ${ratioW(q)} as long, by this lab’s estimate.</p><p>✗ “Every transformer can cache this way.” In ${DEF('an encoder such as BERT',TIP.enc())} every token also sees later ones, so each new token changes the earlier tokens’ keys and values from layer 2 on, and a stored copy would be out of date.</p></div>${A.shell.toScenarios()}`,
    `<ul><li>每个 decode 步都需要目前所有 token 在每一层的 key 和 value。有因果 mask，它们永远不变，所以只算一次、存下来反复用，这就是 KV cache。</li><li>不存的话，工作量随长度的平方增长；存下来，只随长度线性增长。</li><li>代价是显存。每个 token 占的显存是<div class="kvb-fx">${A.ui.math('\\text{每个 token 的 KV cache} = 2 \\times \\text{层数} \\times \\text{KV head 数} \\times \\text{head_dim} \\times \\text{每个数值的字节数}')}</div>式子里的 2 指 key 和 value 两个向量。Llama-3.1-8B 在 BF16 下每个 token 占 ${fint(KV_TOK/1024)} KiB，到 ${ctxTok(LL.maxCtx)} 时${topZ}。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “KV cache 缓存的是模型的输出。”它存的是每一层里的 key 和 value，而不是写出的词。</p><p>✗ “没有 KV cache 也就慢一点。”只有上下文最短时才是这样。不用 cache 时，每一步都要把整段上下文重算一遍，所以按这个 Lab 的估计，到 ${ctxTok(r.C)} 时，一步的耗时是用 cache 时的 ${ratioW(q)}。</p><p>✗ “所有 Transformer 都能这样缓存。”在 ${DEF('BERT 这类 encoder',TIP.enc())} 里，每个 token 也看后面的 token，所以每来一个新 token，之前 token 从第 2 层起的 key 和 value 都会变，存下来的那份就不能再用了。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real workloads, one plain phrase each (R2.1)
   ===================================================================== */
const PRESETS=[
 {id:'chat',t:{en:'Chat',zh:'聊天'},d:{en:'a 1,000-token prompt and a 500-token answer',zh:'prompt 1,000 个 token，回答 500 个'},P:1000,A:500},
 {id:'doc',t:{en:'Summarizing a long document',zh:'长文档摘要'},d:{en:'a 32K-token document and a 1,000-token summary',zh:'32K 个 token 的文档，摘要 1,000 个 token'},P:32768,A:1000},
 {id:'code',t:{en:'Code completion',zh:'代码补全'},d:{en:'4K tokens of code and a 50-token completion',zh:'4K 个 token 的代码，补全 50 个 token'},P:4096,A:50},
 {id:'turn50',t:{en:'The 50th turn of a long chat',zh:'长对话的第 50 轮'},d:{en:'64K tokens of conversation and a 500-token answer',zh:'对话已有 64K 个 token，回答 500 个 token'},P:65536,A:500}
];
const presetNow=()=>PRESETS.findIndex(p=>p.P===S.ctx);
/* R6.20: the two whole-answer times, rounded, are the result; each carries its exact value and its parts on hover (the stage shows
   one step, not the whole answer); the reason is in words */
function scenHTML(i){
  const p=PRESETS[i], s=scen(p.P,p.A), k=totSec(s.without)/totSec(s.with), last=nocache(s.endTok).t;
  const tipW=tx(`With the KV cache the whole answer takes ${totS(s.with)}, by this lab’s estimate. The model first runs the ${fint(p.P)}-token prompt through once (the prefill), which takes ${tS(s.first)} and gives the first token; each of the other ${fint(p.A-1)} tokens takes one decode step of about ${fixd(s.avg*1e3,2)} ms.`,
                `用 KV cache 时，整个回答要 ${totS(s.with)}（按这个 Lab 的估计）。模型先把 ${fint(p.P)} 个 token 的 prompt 过一遍（这一步叫 prefill），${tS(s.first)} 后得到第一个 token；之后的 ${fint(p.A-1)} 个 token 各用一个 decode 步，每步约 ${fixd(s.avg*1e3,2)} ms。`);
  const tipWo=tx(`Without the KV cache the whole answer takes ${totS(s.without)}, ${ratioS(k)} times as long as with it, by this lab’s estimate. Every step, the first included, runs the whole conversation through the model: ${fint(p.P)} tokens at the first step (${tS(s.firstNo)}), ${fint(s.endTok)} at the last (${tS(last)}).`,
                 `不用 KV cache 时，整个回答要 ${totS(s.without)}，是用 cache 时的 ${ratioS(k)} 倍（按这个 Lab 的估计）。包括第一步在内，每一步都要把整段对话过一遍模型：第一步 ${fint(p.P)} 个 token（${tS(s.firstNo)}），最后一步 ${fint(s.endTok)} 个（${tS(last)}）。`);
  /* the long chat: a server can keep the earlier turns' cache (R6.11: this lab does not) */
  const pc=p.id==='turn50'?tx(` In this lab the earlier turns’ cache is not kept between turns; a server with ${DEF('prefix caching',TIP.prefix())}, such as ${cite(SRC.apc,'vLLM')}, can keep it and skip most of that prefill.`,
                              `在这个 Lab 里，前几轮的 cache 没有留到这一轮；开了 ${DEF('prefix caching',TIP.prefix())} 的服务端（比如 ${cite(SRC.apc,'vLLM')}）可以留着，省掉这次 prefill 的大部分。`):'';
  return tx(`<p><b>${esc(L(p.t))}, by this lab’s estimate: with the KV cache the whole answer takes ${DEF('about '+totR(s.with),tipW)}; without it, ${DEF('about '+totR(s.without),tipWo)}.</b></p><p>With the cache, only the first token waits for the whole prompt to run through the model (the prefill); each later token takes one short step. Without it, every step runs the whole conversation through again, so every token takes at least as long as the first.${pc}</p><p>The cache grows with every token: by the last one it holds ${memD(s.endBytes)} (${gbS(s.endBytes)}).</p><p class="small muted">The chart shows one step at ${fint(p.P)} tokens of context, the prompt’s length; the times above add up the whole answer.</p>`,
            `<p><b>${esc(L(p.t))}：按这个 Lab 的估计，用 KV cache 整个回答大约要 ${DEF(totR(s.with),tipW)}，不用大约要 ${DEF(totR(s.without),tipWo)}。</b></p><p>用 cache 时，只有第一个 token 要等整个 prompt 过一遍模型（这一步叫 prefill），之后每个 token 只要短短的一步。不用 cache 时，每一步都要把整段对话重新过一遍，所以每个 token 都至少和第一个一样慢。${pc}</p><p>KV cache 随每个 token 增长：写到最后一个 token 时，它占 ${memD(s.endBytes)}（${gbS(s.endBytes)}）。</p><p class="small muted">图里画的是上下文为 ${fint(p.P)} 个 token（prompt 的长度）时的一步；上面的时间是整个回答加起来的。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chPoint=C=>{const r=real(C); return {C,t:r.wo.t,tps:tpsV(r.wo.t),ok:tpsV(r.wo.t)>=CH_GOAL,tw:r.w.t};};
const chBest=()=>CH_STOPS.map(chPoint).filter(p=>p.ok).sort((a,b)=>b.C-a.C)[0];
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chPoint(S.ctx), best=chBest();
  /* R6.20: the verdict's one number is the speed, in the goal's unit; the time per token is on the metric card, every stop's on the chart */
  /* the next stop misses the goal, so the brief chart leaves it out: its time is on hover */
  const nx=CH_STOPS[CH_STOPS.indexOf(me.C)+1], nxP=nx?chPoint(nx):null;
  const tooSlow=(en,zh)=>nxP?DEF(tx(en,zh),tx(`At ${fint(nxP.C)} tokens of context each token takes ${tS(nxP.t)}, over 50 ms.`,`上下文 ${fint(nxP.C)} 个 token 时，每个 token 要 ${tS(nxP.t)}，超过了 50 ms。`)):tx(en,zh);
  let h;
  if(me.ok&&me.C===best.C) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: by this lab’s estimate, at ${fint(me.C)} tokens of context it writes ${tpsS(me.t)} tokens per second.</b> And that is the longest context here that keeps up: ${tooSlow('the next one is already too slow','再长一档就太慢了')}.</p>`,
                                `<p class="okline" style="margin-top:12px"><b>达成目标：按这个 Lab 的估计，上下文 ${fint(me.C)} 个 token 时，每秒生成 ${tpsS(me.t)} 个 token。</b>而且这已经是这里能达标的最长上下文：${tooSlow('the next one is already too slow','再长一档就太慢了')}。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: by this lab’s estimate, at ${fint(me.C)} tokens of context it writes ${tpsS(me.t)} tokens per second.</b> But a longer context still keeps up.</p>`,
                      `<p class="okline" style="margin-top:12px"><b>达成目标：按这个 Lab 的估计，上下文 ${fint(me.C)} 个 token 时，每秒生成 ${tpsS(me.t)} 个 token。</b>不过更长的上下文也能达标。</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: by this lab’s estimate, at ${fint(me.C)} tokens of context it writes only ${tpsS(me.t)} tokens per second.</b></p><p class="small">Hint: without the cache, each decode step recomputes every token of the context. How long a context can one step recompute in 50 ms?</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：按这个 Lab 的估计，上下文 ${fint(me.C)} 个 token 时，每秒只能生成 ${tpsS(me.t)} 个 token。</b></p><p class="small">提示：不用 cache 时，每个 decode 步都要把上下文里的每个 token 重算一遍。50 ms 里一步能重算多长的上下文？</p>`)
          +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const pts=CH_STOPS.map(chPoint), base=chPoint(CH_START), lo=Math.min(...pts.map(p=>p.tw)), hi=Math.max(...pts.map(p=>p.tw));
  const rows=pts.map(p=>({label:tx(`${fint(p.C)} tokens: ${tS(p.t)} per token`,`${fint(p.C)} 个 token：每个 ${tS(p.t)}`),value:p.tps,text:tpsS(p.t),ok:p.ok,me:p.C===me.C,
    tags:(p.C===CH_START?[{text:tx('initial setup','初始设置')}]:[]).concat(p.C===me.C?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  /* brief (the owner's review of batch 8): the chart keeps the stops that meet the goal, the pick and the initial setup, and says how
     many it left out; the title and the aria name only what it shows */
  const shownPts=pts.filter(p=>p.ok||p.C===me.C||p.C===CH_START);
  h+=A.ui.cmpChart({title:NB(tx('Estimated tokens per second without the KV cache, Llama-3.1-8B on one H100 SXM (goal: at least 20)','一张 H100 SXM、不用 KV cache 时，Llama-3.1-8B 每秒生成的 token 数估计（目标：至少 20 个）')),
    aria:tx('Tokens per second without the KV cache: ','不用 KV cache 时每秒生成的 token 数：')+shownPts.map(p=>`${fint(p.C)}${tx(' tokens: ',' 个 token：')}${tpsS(p.t)}`).join(tx('; ','；')),
    brief:true,hiddenText:n=>tx(`The other ${n} context lengths give fewer than 20 tokens per second and are not shown.`,`其余 ${n} 个上下文长度每秒不到 20 个 token，图里没有列出。`),
    rows,goal:CH_GOAL,goalLabel:tx('goal: 20','目标 20'),max:150,okText:tx('20 or more per second','每秒 20 个或更多'),noText:tx('fewer than 20','不到 20 个'),goalText:tx('goal: 20 tokens per second','目标：每秒 20 个 token')});
  /* baseline → effect → reason (R3.10) in words and the answer's one number (R6.20): the initial setup and the answer are on the chart */
  const far=base.tps<CH_GOAL/2, little=hi/lo<1.5;
  h+=tx(`<p class="small">Initial setup, ${ctxTok(CH_START)}: ${far?'far too slow':'too slow'}. The longest context here that meets the goal is ${fint(best.C)} tokens. With the KV cache, the time per token would grow ${little?'only a little':'much more slowly'} from the shortest context here to the longest.</p>`,
        `<p class="small">初始设置是 ${ctxTok(CH_START)}，${far?'远远达不到目标':'达不到目标'}。这里能达标的最长上下文是 ${fint(best.C)} 个 token。用 KV cache 的话，从这里最短的上下文到最长的，每个 token 的耗时${little?'只增加一点':'增加得慢得多'}。</p>`);
  /* review 8 points 2, 4, 5: which term decides each step, and what "reading the weights" is (its size and time are in its tooltip) */
  h+=tx(`<p class="small">Why: without the cache, every step computes all the context’s tokens again, so arithmetic decides its time, and it grows with the context. Only at the shortest context does that arithmetic still hide inside the time it takes to ${DEF('read the weights',TIP.wread())}, since the GPU computes while it reads. With the cache, a step computes one token and mostly reads memory, so the context adds little to its time.</p>`,
        `<p class="small">原因：不用 cache 时，每一步都要把上下文里的全部 token 重算一遍，所以计算决定耗时，而且随上下文增长。只有上下文最短时，这点计算还藏在${DEF('读一遍权重',TIP.wread())}所需的时间里，因为 GPU 一边读一边算。用 cache 时，每一步只算一个 token，主要时间花在读显存上，所以上下文变长也只多花一点时间。</p>`);
  h+=tx(`<p class="small muted">Note: this is one conversation on one GPU, with this lab’s estimates of 50% of peak compute and 60% of peak memory bandwidth. Real servers keep the cache and serve many conversations at once, so its memory becomes the limit; the ${labLink('kv-cache')} lab works that out.</p>`,
        `<p class="small muted">注意：这里只算一张 GPU 上的一个对话，并按这个 Lab 的估计（峰值算力的 50%、峰值带宽的 60%）计时。真实服务都保留 cache，并同时服务很多对话，所以 cache 占的显存成了限制；${labLink('kv-cache')}那个 Lab 讲的就是这个。</p>`);
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Seven short parts. Predict first, watch a small model write a sentence while keeping nothing, check whether earlier keys and values ever change, then keep them and see what that saves and costs on Llama-3.1-8B. Start here if this is new to you.',
            '一共 7 个部分。先预测，再看一个小模型什么都不存、一步一个词地写出一句话，检查之前的 key 和 value 会不会变，然后把它们存下来，看看在 Llama-3.1-8B 上省了什么、花了什么。第一次学从这里开始。'),
  sandbox:tx('Four real workloads for Llama-3.1-8B on one H100 SXM, from a 50-token code completion to a chat with 64K tokens of history. Pick one to see how long its answer takes with and without the KV cache.',
             '四个真实负载，都是一张 H100 SXM 跑 Llama-3.1-8B，从补全 50 个 token 的代码，到已有 64K 个 token 的长对话。选一个，看它的回答用和不用 KV cache 各要多久。'),
  challenge:tx('One H100 SXM runs Llama-3.1-8B for one conversation, which needs at least 20 tokens per second. With the KV cache turned off, find the longest context that keeps up.',
               '一张 H100 SXM 跑 Llama-3.1-8B，只服务一个对话，要求每秒至少 20 个 token。关掉 KV cache，找出还能达到这个速度的最长上下文。')
});
const SRC={
 cfg:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 vas:'https://arxiv.org/abs/1706.03762',
 hf:'https://huggingface.co/docs/transformers/cache_explanation',
 cu:'https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py',
 llama:'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
 pope:'https://arxiv.org/abs/2211.05102',
 bert:'https://arxiv.org/abs/1810.04805',
 vllm:'https://vllm.ai/blog/2023-06-20-vllm',
 apc:'https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html'
};
/* the prefill-decode tile, as a link to the map where it lives (A18) */
const TILE_A=()=>`<a href="#map~inference" data-tipx="${esc(tx('Click to open the tree at Inference serving and see this tile.','点击回到知识树的“推理服务”这一枝，看这个方块。'))}">${esc(tx('“Why is the first token slow and the rest fast?”','“为什么首 token 要等很久，之后却一个接一个很快？”'))}</a>`;
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser runs a small transformer for real. The Llama-3.1-8B times come from the step model under “Modeled”; ${J('mathreal','“Show the math”')} works them out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The small model: 2 layers, hidden size d = 8. Each layer has an attention block with 2 heads of head_dim 4 and an MLP (multi-layer perceptron) block of 8 → 16 → 8 numbers with a ReLU in between; an RMSNorm without learned scales comes before each block, and each block’s output is added back to the vector from before its RMSNorm (the residual connection). Keys and values are computed in floating point in every step.</li>`+
 `<li>The recompute check compares the kept keys and values with a full recompute, number by number.</li>`+
 `<li>Llama-3.1-8B’s KV cache per token, from its config.json.</li>`+
 `<li>Time per decode step on one H100 SXM, with the step model that the home-page tile ${TILE_A()} uses. The GPU reads memory and does arithmetic at the same time, so a step takes the longer of the two. The arithmetic is 2 FLOPs per multiplied parameter per token, plus attention. The memory traffic is every parameter but the embedding table, read once per step however many tokens it computes. With the cache, a step also reads 128 KiB of keys and values per earlier token; the first step, the prefill, writes 128 KiB per prompt token instead. Without the cache, nothing is read from or written to a cache.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The small model’s parameters and its 9 input vectors are made-up numbers, fixed on every visit; the same numbers stand for the English and the Chinese sentence.</li>`+
 `<li>The small model has no LM head (LM: language model), the matrix that scores every possible next token; each next word is taken from the sentence instead, because the lab counts work, not guesses.</li>`+
 `<li>Compute at 50% of the dense BF16 peak and memory at 60% of the bandwidth are estimates (${J('eff','see Settings')}).</li>`+
 `<li>A step without the cache is priced as one pass of the whole context through the model. The memory traffic of activations is left out: the vectors each block passes on and, without the cache, the keys and values the step computes and then drops. It grows with the tokens a step computes, so at 256 tokens the step without the cache would take somewhat longer than shown; from 512 tokens on, that step’s arithmetic takes far longer than this traffic, so leaving it out does not change its time. One conversation on one GPU, no batching.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Llama-3.1-8B: ${cite(SRC.cfg,'config.json')} (32 layers, 8 KV heads, head_dim 128).</li>`+
 `<li>H100 SXM, from ${cite(SRC.h100,'NVIDIA’s product page')}: BF16 1,979 TFLOPS “with sparsity”, so 989.5 dense; 3.35 TB/s; 80 GB.</li>`+
 `<li>The causal mask: ${cite(SRC.vas,'Vaswani et al., 2017')}, §3.2.3.</li>`+
 `<li>An encoder such as BERT: ${cite(SRC.bert,'Devlin et al., 2018')}, whose abstract says BERT is conditioned “on both left and right context in all layers”.</li>`+
 `<li>The KV cache in Hugging Face transformers: ${cite(SRC.hf,'“How caching works”')} and <code>DynamicCache</code>.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-25. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器真的在运行一个小 Transformer。Llama-3.1-8B 的耗时来自“建模了什么”里的耗时模型，${J('mathreal','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>小模型：2 层，隐藏维度 d = 8。每层一个 attention 块（2 个 head，head_dim 4）和一个 MLP 块（MLP 即多层感知机；8 → 16 → 8 维，中间用 ReLU）；每个块前有一个不带可学系数的 RMSNorm，块的输出再加回到这个块的 RMSNorm 之前的向量上（残差连接）。每一步的 key 和 value 都用浮点数真的算出来。</li>`+
 `<li>重算检查把存下来的 key 和 value 和全部重算的结果逐个数值比较。</li>`+
 `<li>Llama-3.1-8B 每个 token 的 KV cache，按它的 config.json 算。</li>`+
 `<li>一张 H100 SXM 上每个 decode 步的耗时，用的是首页方块${TILE_A()}的耗时模型。GPU 读显存和做计算是同时进行的，所以一步取两者中较长的那个。计算按每个参与乘法的参数每个 token 2 FLOP，再加 attention。读显存算除 embedding 表外的全部参数，每一步读一遍，不管这一步算多少个 token。用 cache 时，每一步再读之前每个 token 128 KiB 的 key 和 value；第一步 prefill 则把 prompt 每个 token 的 128 KiB 写进 cache。不用 cache 时，既不读也不写 cache。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>小模型的参数和 9 个输入向量都是编出来的数值，每次打开都一样；英文句子和中文句子用的是同一组数。</li>`+
 `<li>小模型没有 LM head（LM 即语言模型，language model），也就是给每个可能的下一个 token 打分的那个矩阵；每一步的下一个词直接取自这句话，因为这个 Lab 数的是工作量，不是猜词。</li>`+
 `<li>计算按稠密 BF16 峰值的 50%、读显存按峰值带宽的 60% 算，都是估计值（${J('eff','见“自由调参”')}）。</li>`+
 `<li>不用 cache 的一步，按“整段上下文一次过一遍模型”计价。不计激活值读写显存的流量，包括各个块传下去的向量；不用 cache 时，还包括这一步算出后又丢掉的 key 和 value。它们随一步算的 token 数增长，所以 256 个 token 时，不用 cache 的一步实际会比这里显示的慢一些；从 512 个 token 起，这一步的计算远比这些流量耗时，不计它们不改变它的耗时。只算一张 GPU 上的一个对话，不做批处理。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>Llama-3.1-8B：${cite(SRC.cfg,'config.json')}（32 层、8 个 KV head、head_dim 128）。</li>`+
 `<li>H100 SXM：${cite(SRC.h100,'NVIDIA 产品页')}，BF16 1,979 TFLOPS 标着“with sparsity”，所以稠密是 989.5；3.35 TB/s；80 GB。</li>`+
 `<li>因果 mask：${cite(SRC.vas,'Vaswani 等，2017')}，§3.2.3。</li>`+
 `<li>BERT 这类 encoder：${cite(SRC.bert,'Devlin 等，2018')}，摘要说 BERT 在每一层都同时看左右两边的上下文。</li>`+
 `<li>Hugging Face transformers 里的 KV cache：${cite(SRC.hf,'“How caching works”')} 和 <code>DynamicCache</code>。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-25。原型，尚未经专家审阅。</p></div></div>`);
/* the primer in the site's structural words (CONTRACT §11): each word carries the kit's definition, as in the Architecture overview lab */
const PRIPOP=()=>tx(
 `<p>In every ${DEF('layer',kt('layer'))}, the ${DEF('attention block',kt('attention'))} turns each token’s vector into a query, a key and a value, and cuts each of them into equal parts, one per ${DEF('head',kt('head'))}. Each head compares its part of the token’s query with its part of the keys of the token itself and every earlier token (the causal mask hides every later token), turns the scores into ${DEF('attention weights',kt('aweight'))} that add up to 100%, and takes that mix of their values. In ${DEF('multi-head attention (MHA)',kt('mha'))} every head keeps its own key and value.</p><p>The ${labLink('mha')} lab works through it from scratch, with the same example sentence (The cat sat on the mat …). The ${labLink('architecture')} lab shows how the whole model is built from such layers.</p>`,
 `<p>每一${DEF('层',kt('layer'))}里，${DEF('attention 块',kt('attention'))}把每个 token 的向量变成一个 query、一个 key 和一个 value，每个都切成相等的几份，每份属于一个 ${DEF('head',kt('head'))}。每个 head 拿自己那份 query，和这个 token 自己及之前每个 token 的 key 的同一份比较（因果 mask 挡住所有后面的 token），把得分换成加起来是 100% 的${DEF('注意力权重',kt('aweight'))}，再按这些注意力权重混合它们的 value 的那一份。在${DEF('多头注意力（MHA）',kt('mha'))}里，每个 head 都有自己的 key 和 value。</p><p>${labLink('mha')}那个 Lab 用这里的同一个例句（小猫 坐 在 垫子 上……）从头讲。整个模型怎样由这样的层搭起来，在${labLink('architecture')}那个 Lab 里讲。</p>`);

/* =====================================================================
   views
   ===================================================================== */
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers */
const OUTCOME=()=>tx('explain why generation keeps every token’s keys and values, and count the compute it saves and the memory it costs.','讲清生成时为什么要存每个 token 的 key 和 value，并算出它省下多少计算、多占多少显存。');
const EXITS=()=>{const m70=DRILL_M.find(m=>m.name==='Llama-3.1-70B'), kv70=2*m70.L*m70.kv*m70.hd*2; return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why can generation store each token’s keys and values instead of computing them again at every step?','生成时，为什么可以把每个 token 的 key 和 value 存下来，而不用每一步重算？'),
   o:[tx('With the causal mask, a token’s key and value depend only on the tokens up to it, so they never change once computed.','有因果 mask，一个 token 的 key 和 value 只取决于它和它前面的 token，算出来以后就不会再变。'),
     tx('Keys and values are part of the weights, so they are loaded once, together with the model.','key 和 value 是权重的一部分，随模型一起载入一次就行。'),
     tx('Later tokens change the earlier keys and values only a little, so reusing them is close enough.','后面的 token 只会让前面的 key 和 value 变一点点，所以复用它们误差不大。')],a:0,
   n:[null,tx('The weights Wk and Wv are fixed, but each token’s key and value are computed from that token’s own vector in each layer, so they are stored per token.','固定的是权重 Wk 和 Wv；每个 token 的 key 和 value 是用它在每一层的向量算出来的，所以要按 token 存。'),
     tx('They do not change at all: the causal mask keeps later tokens out of an earlier token’s computation, so the cache gives exactly the same result.','它们完全不变：因果 mask 让后面的 token 进不了前面 token 的计算，所以用 cache 的结果和重算一模一样。')],
   why:tx('A token’s key and value in a layer are computed from its vector there, which depends only on the tokens up to it. So each step computes the newest token’s key and value and reads the others from the cache.','一个 token 在某一层的 key 和 value 由它在这一层的向量算出，而这个向量只取决于它和它前面的 token。所以每一步只算最新 token 的 key 和 value，其余的从 cache 里读。')},
  {q:tx('What does the KV cache cost?','KV cache 的代价是什么？'),
   o:[tx('Memory: every token keeps a key and a value in every layer, and the cache grows with the context.','显存：每个 token 在每一层都存一份 key 和 value，cache 随上下文增长。'),
     tx('Accuracy: reusing old keys and values makes the output slightly different from computing everything again.','准确度：复用旧的 key 和 value，输出会和全部重算略有不同。'),
     tx('Compute: each step has to redo the attention of every earlier token.','计算量：每一步都要把前面每个 token 的注意力重做一遍。')],a:0,
   n:[null,tx('The output is the same: the stored keys and values are exactly what a full recomputation would give.','输出完全一样：存下的 key 和 value 和全部重算得到的一模一样。'),
     tx('That is the cost without the cache. With it, each step computes only the newest token, though it still reads the whole cache.','那是不用 cache 的代价。用 cache 时每一步只算最新的 token，只是要把整个 cache 读一遍。')],
   why:tx(`The cache saves compute and costs memory: every token of every conversation keeps a key and a value per KV head and layer, ${fint(KV_TOK/1024)} KiB for Llama‑3.1‑8B.`,`cache 省的是计算，花的是显存：每个对话的每个 token 在每一层都要为每个 KV head 存一个 key 和一个 value，Llama‑3.1‑8B 每个 token 要 ${fint(KV_TOK/1024)} KiB。`)}],
 /* a new case: the worked computation stays (R6.6), as short steps (R6.20); the numbers from the drill's Llama-3.1-70B and tri() */
 transfer:[
  {q:tx('Llama‑3.1‑70B has 80 layers, 8 KV heads and head_dim 128. How much KV cache does it keep per token in BF16?','Llama‑3.1‑70B 有 80 层、8 个 KV head，head_dim 是 128。BF16 下每个 token 要存多少 KV cache？'),
   o:['40 KiB','128 KiB','320 KiB','2.5 MiB'],a:2,
   why:tx(`2 (a key and a value) × ${m70.L} layers × ${m70.kv} KV heads × ${m70.hd} numbers × 2 bytes = ${fint(kv70)} bytes = ${fint(kv70/1024)} KiB. It has ${fixd(m70.L/LL.layers,1)} times Llama‑3.1‑8B’s layers, so ${fixd(kv70/KV_TOK,1)} times its ${fint(KV_TOK/1024)} KiB.`,`2（key 和 value）× ${m70.L} 层 × ${m70.kv} 个 KV head × ${m70.hd} 维 × 2 字节 = ${fint(kv70)} 字节 = ${fint(kv70/1024)} KiB。层数是 Llama‑3.1‑8B 的 ${fixd(m70.L/LL.layers,1)} 倍，所以是它 ${fint(KV_TOK/1024)} KiB 的 ${fixd(kv70/KV_TOK,1)} 倍。`)},
  {q:tx('Without the KV cache, how many token-passes does it take to write 1,000 tokens after <s>?','不用 KV cache，在 <s> 后面生成 1,000 个 token，一共要做多少次 token 前向？'),
   o:['1,000','2,000','500,500','1,000,000'],a:2,
   why:tx(`Step k runs all k tokens so far: 1 + 2 + … + 1,000 = 1,000 × 1,001 ÷ 2 = ${fint(tri(1000))}. With the cache it is 1,000, one per step.`,`第 k 步要把目前的 k 个 token 都过一遍：1 + 2 + … + 1,000 = 1,000 × 1,001 ÷ 2 = ${fint(tri(1000))} 次。用 KV cache 只要 1,000 次，每步一次。`)}],
 practice:[
  {id:'attention-kv-cache',d:tx('Part 2: decode one token at a time with a KV cache, matching full attention exactly, and count the cache’s bytes.','第 2 部分：用 KV cache 一次解码一个 token，结果要和整段计算完全一致，再算 cache 占多少字节。')},
  {id:'matmul-roofline',d:tx('Part 1: for four matmul shapes, count the FLOPs and the bytes moved, and say whether compute or memory bandwidth limits each one.','第 1 部分：四种形状的矩阵乘法，各算 FLOPs 和搬运的字节数，判断每一种受算力还是显存带宽限制。')}]};};
function headerHTML(){
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:tx('Why a KV cache','为什么要 KV cache'),title:tx('Why does generation keep every token’s keys and values?','生成时为什么要把 K 和 V 存起来？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: attention and heads','前置知识：attention 和 head'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function toyHTML(){
  const w=WORDS[A.lang()];
  let h=`<div class="kvb-grid" id="kvb-grid"><div class="kvb-row kvb-toks"><span class="kvb-rl"></span>${w.map((x,j)=>`<span class="kvb-tok" data-j="${j}"><i>${j+1}</i><b>${esc(x)}</b></span>`).join('')}</div>`;
  for(let l=0;l<NL;l++){
    h+=`<div class="kvb-layer" data-l="${l}"><div class="kvb-lcap">${l===0?tx('Layer 1 (each row is a decode step, each column a token; step n takes in token n and writes the word after it)','第 1 层（每行是一个 decode 步，每列是一个 token；第 n 步读入第 n 个 token，生成它后面的词）'):tx('Layer 2','第 2 层')}</div>`;
    for(let s=1;s<=NT;s++) h+=`<div class="kvb-row kvb-srow" data-s="${s}"><span class="kvb-rl"${l===0?' tabindex="0"':''}>${tx(`step ${s}`,`第 ${s} 步`)}</span>${w.map((_,j)=>`<span class="kvb-c" data-j="${j}"></span>`).join('')}</div>`;
    h+=`<div class="kvb-row kvb-chk" hidden><span class="kvb-rl" tabindex="0">${tx('change','变化')}</span>${w.map((_,j)=>`<span class="kvb-d mono" data-j="${j}"></span>`).join('')}</div></div>`;
  }
  return h+`</div><p class="stage-note kvb-note">${tx('One word is one token here; each step’s next word is taken from the sentence.','这里一个词或一个标点就是一个 token；每一步的下一个词直接取自这句话。')}</p>`;
}
function realHTML(){
  const ax=[0,50,100,150].map(v=>`<span style="left:${(v/150*100).toFixed(2)}%">${v}</span>`).join(''), mx=[0,20,40,60,80].map((v,i)=>`<span style="left:${v/80*100}%">${v}${i===4?' GiB':''}</span>`).join('');
  return `<div class="kvb-ctitle" id="kvb-sp-title"></div><div class="kvb-sp" id="kvb-sp">`+
    ['w','wo'].map(k=>`<div class="kvb-sprow" data-k="${k}" tabindex="0"><div class="kvb-sph"><span class="kvb-spl"></span><span class="kvb-spv mono"></span></div><div class="kvb-spt"><i></i><b class="kvb-goal" hidden></b></div></div>`).join('')+
    `<div class="kvb-axis mono" aria-hidden="true">${ax}</div><div class="kvb-axlab mono" aria-hidden="true">${tx('tokens per second','每秒 token 数')}</div></div>`+
    `<div id="kvb-memwrap"><div class="kvb-ctitle" id="kvb-mem-title"></div><div class="kvb-mem" id="kvb-mem" tabindex="0"><i class="kvb-mw"><span></span></i><i class="kvb-mk"><span></span></i><i class="kvb-mf"></i></div>`+
    `<div class="kvb-axis mono" aria-hidden="true">${mx}</div><p class="stage-note" id="kvb-real-note">${tx('The datasheet’s 80 GB is taken as 80 GiB.','规格表的 80 GB 按 80 GiB 计。')}</p></div>`;
}
function stageHTML(){
  return `<div class="panel stage kvb-stage" id="kvb-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="kvb-st-title"></div><div class="meter-sub" id="kvb-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="kvb-status" tabindex="0"></span>`+
   `<div class="seg-ctl kvb-vseg" id="kvb-vseg" role="group" aria-label="${esc(tx('Chart','图'))}" data-key="view">${[['toy',tx('Small model','小模型')],['real','Llama-3.1-8B']].map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${S.view===v}" data-tipx="${esc(tx('Switch the chart between the small model and Llama-3.1-8B. It changes only the chart, not your settings.','在小模型和 Llama-3.1-8B 之间切换图。只换图，不改你的设置。'))}">${t}</button>`).join('')}</div></div></div>`+
   `<div class="legend-row" id="kvb-legend"></div>`+
   `<div class="stage-graph graph kvb-graph" id="kvb-graph"><div id="kvb-toy"${S.view==='toy'?'':' hidden'}>${toyHTML()}</div><div id="kvb-real"${S.view==='real'?'':' hidden'}>${realHTML()}</div></div>`+
   `<div class="metrics kvb-metrics" id="kvb-metrics"></div>`+
   `<details class="math" id="kvb-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="kvb-mathbody"></div></details></div>`;
}
const EFF_NOTE=()=>tx('This is an estimate. The GPU reads memory and does arithmetic at the same time, so a decode step takes about as long as the slower of the two: its arithmetic, at 50% of the H100 SXM’s dense BF16 peak (989.5 TFLOP/s), or its memory traffic, at 60% of its peak bandwidth (3.35 TB/s). Every step reads the weights once, however many tokens it computes. Both shares are round numbers this lab assumes: real kernels reach only part of the peak, and how much depends on the kernel and on how many tokens a step handles.',
  '这是估计值。GPU 读显存和做计算是同时进行的，所以一个 decode 步的耗时约等于两者中较慢的那个：计算按 H100 SXM 稠密 BF16 峰值（989.5 TFLOP/s）的 50% 算，读显存按峰值带宽（3.35 TB/s）的 60% 算。不管一步算多少个 token，权重都只读一遍。这两个比例是本 Lab 假设的整数：真实 kernel 只能用到峰值的一部分，用到多少取决于 kernel 和一步处理多少个 token。');
function controlsHTML(){
  return `<aside class="panel controls" id="kvb-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="kvb-lock" hidden></p>`+
   `<div class="kvb-grp lockable" id="kvb-grp-toy"><div class="kvb-gh">${tx('Small model','小模型')}</div>`+
   `<div id="kvb-ctl-n-wrap">`+A.ui.range({id:'kvb-ctl-n',key:'n',label:tx('Decode steps done','已完成的 decode 步'),min:0,max:NT,value:S.n,out:''})+`</div>`+
   `<div class="ctl-note" id="kvb-nlock" hidden>${tx('The small model starts writing after you answer part 2.','在第 2 部分作答后，小模型才开始生成。')}</div>`+
   `<div class="ctl" id="kvb-ctl-cache">`+cacheCtl(false).replace(/^<div class="ctl" id="kvb-ctl-cache-seg">|<\/div>$/g,'')+`<div class="ctl-note">${tx('Off: every decode step computes the keys and values of all tokens so far. On: each token’s are computed once and kept.','不用：每个 decode 步都算出目前所有 token 的 key 和 value。用：每个 token 的只算一次，然后存下来。')}</div></div>`+
   `<div class="ctl" id="kvb-ctl-mask">`+maskCtl(false).replace(/^<div class="ctl" id="kvb-ctl-mask-seg">|<\/div>$/g,'')+`<div class="ctl-note">${tx('On: no token looks at later tokens, as in decoder-only models such as Llama and GPT. Off: every token looks at every other token, as in an encoder such as BERT.','开：没有 token 看后面的 token，Llama、GPT 这类 decoder-only 模型都这样。关：每个 token 都看其他所有 token，像 BERT 这类 encoder。')}</div></div></div>`+
   `<div class="kvb-grp" id="kvb-grp-real"><div class="kvb-gh">${NB(tx('Llama-3.1-8B on one H100 SXM','一张 H100 SXM 跑 Llama-3.1-8B'))}</div>`+
   `<div class="lockable">`+ctxCtl('kvb-ctl-ctx',CTX,false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The tokens one decode step works on: the prompt plus everything written so far. Llama-3.1-8B supports up to 128K (131,072).','一个 decode 步要处理的 token：prompt 加上目前已生成的全部。Llama-3.1-8B 最多支持 128K（131,072）个。')}</div></div>`)+`</div>`+
   `<div class="ctl" id="kvb-ctl-eff"><span class="lbl">${tx('Step time','每一步的耗时')}</span><div class="ctl-note">${EFF_NOTE()}</div></div></div>`+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import torch

torch.manual_seed(0)
d, n_layers = 8, 2
W = [{k: torch.randn(d, d) / d**0.5 for k in "qkvo"}
     for _ in range(n_layers)]

def attend(q, K, V):            # q: (d,)  K, V: (t, d)
    return torch.softmax(K @ q / d**0.5, dim=0) @ V

def with_cache(xs):             # one token-pass per decode step
    cache = [([], []) for _ in W]   # per layer: kept keys, values
    outs = []
    for x in xs:                # x: the newest token's vector
        for w, (Ks, Vs) in zip(W, cache):
            Ks.append(x @ w["k"])       # computed once, kept
            Vs.append(x @ w["v"])
            a = attend(x @ w["q"], torch.stack(Ks), torch.stack(Vs))
            x = x + a @ w["o"]
        outs.append(x)
    return outs

def without_cache(xs):          # step t reruns all t tokens
    outs = []
    for t in range(1, len(xs) + 1):
        h = xs[:t]
        for w in W:
            K, V = h @ w["k"], h @ w["v"]   # recomputed every step
            mask = torch.ones(t, t).tril().bool()   # causal mask
            s = h @ w["q"] @ K.T / d**0.5
            s = s.masked_fill(~mask, float("-inf"))
            h = h + torch.softmax(s, dim=-1) @ V @ w["o"]
        outs.append(h[-1])
    return outs

xs = torch.randn(9, d)
a, b = with_cache(xs), without_cache(xs)
err = max((x - y).abs().max().item() for x, y in zip(a, b))
print(err)  # under 1e-6: float rounding only`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="kvb-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="kvb-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Generation with and without a KV cache, one head per layer: both loops give the same outputs; the one without the cache recomputes every earlier token at every step.','用和不用 KV cache 各生成一遍（每层一个 head）：两种写法输出一样，不用 cache 的那个每一步都把之前的 token 重算一遍。')}</p><pre class="code" id="kvb-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="kvb-prod">`+
   tx(`<li>Hugging Face transformers: the default cache for generation is <code>DynamicCache</code> in ${cite(SRC.cu,'<code>transformers/cache_utils.py</code>')}, one key tensor and one value tensor per layer that grow along the sequence.</li>`+
      `<li>Each attention block, such as <code>LlamaAttention</code> in ${cite(SRC.llama,'<code>models/llama/modeling_llama.py</code>')}, appends the step’s keys and values with <code>past_key_values.update(key_states, value_states, self.layer_idx)</code>.</li>`+
      `<li>vLLM keeps the same keys and values in fixed-size blocks of GPU memory (${cite(SRC.vllm,'vLLM blog, 2023')}); the ${labLink('paged-attention')} lab shows why.</li>`,
      `<li>Hugging Face transformers：生成时默认的 cache 是 ${cite(SRC.cu,'<code>transformers/cache_utils.py</code>')} 里的 <code>DynamicCache</code>，每层一个 key 张量和一个 value 张量，沿序列方向变长。</li>`+
      `<li>每个 attention 块，比如 ${cite(SRC.llama,'<code>models/llama/modeling_llama.py</code>')} 里的 <code>LlamaAttention</code>，用 <code>past_key_values.update(key_states, value_states, self.layer_idx)</code> 把这一步的 key 和 value 追加进去。</li>`+
      `<li>vLLM 把同样的 key 和 value 放在显存里固定大小的 block 中（${cite(SRC.vllm,'vLLM 博客，2023')}）；${labLink('paged-attention')} 那个 Lab 讲为什么。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.hf,'How caching works')}<span>${tx('Hugging Face transformers docs: what the cache keeps for each layer','Hugging Face transformers 文档：cache 在每一层存什么')}</span></li>`+
   `<li>${cite(SRC.pope,'Efficiently Scaling Transformer Inference')}<span>${tx('Pope et al., 2022: the KV cache must stay in memory for the whole decoding','Pope 等，2022：解码期间 KV cache 要一直留在显存里')}</span></li>`+
   `<li>${cite(SRC.vas,'Attention Is All You Need')}<span>${tx('Vaswani et al., 2017, §3.2.3: the decoder’s mask that stops a token from attending to later ones','Vaswani 等，2017，§3.2.3：decoder 用 mask 挡住后面的 token')}</span></li></ul>`+
   `<p class="small muted kvb-see">${tx(`See also: ${cite('https://poloclub.github.io/transformer-explainer/','Transformer Explainer')} (Cho et al., 2024) runs GPT-2 small in your browser. ${cite('https://bbycroft.net/llm','LLM Visualization')} (Brendan Bycroft) is a 3D walkthrough of a tiny GPT running inference, next to the shapes of GPT-2 small and XL.`,
      `另见：${cite('https://poloclub.github.io/transformer-explainer/','Transformer Explainer')}（Cho 等，2024）在浏览器里运行 GPT-2 small。${cite('https://bbycroft.net/llm','LLM Visualization')}（Brendan Bycroft）用 3D 一步步演示一个很小的 GPT 做推理，旁边还画出 GPT-2 small 和 XL 的结构。`)}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="kvb-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="kvb-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="kvb-app m2" id="kvb-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="kvb-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how long the whole answer takes with and without the KV cache.','选一个场景，看看用和不用 KV cache，整个回答各要多久。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="kvb-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${NB(tx('Keep 20 tokens per second on one H100 SXM with the KV cache off','关掉 KV cache，让一张 H100 SXM 保持每秒 20 个 token'))}</h3>`+
   `<p>${NB(tx('The model is Llama-3.1-8B, serving one conversation, and it must write at least 20 tokens per second (at most 50 ms per token). With the KV cache off, how long can the context get? You may change only the context:','模型是 Llama-3.1-8B，只服务一个对话，要求每秒至少生成 20 个 token（每个 token 不超过 50 ms）。关掉 KV cache，上下文最长能到多少？你只能改上下文长度：'))}</p>`+
   ctxCtl('kvb-ic-chctx',CH_STOPS,true)+
   `<div class="goal mono" id="kvb-ch-goal"><span>${tx('Estimated speed now, without the KV cache','现在的速度估计（不用 KV cache）')}</span><span>${tx('','每秒 ')}<b id="kvb-ch-now"></b>${tx(' / 20 tokens per second',' / 20 个 token')}</span></div><div class="progress" id="kvb-ch-prog" aria-hidden="true"><i></i></div>`+
   submit+`<div id="kvb-ch-res" class="kvb-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function renderRail(){const rail=$('#kvb-rail',app); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';   // the next lab in every mode (R3.11)
  rail.innerHTML=NB(railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):''))+nl; A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#kvb-rail',app); if(!rail) return;
  // inline controls mirror the state
  $$('[data-out="n"]',rail).forEach(o=>{o.textContent=tx(`${S.n} of ${NT}`,`${S.n} / ${NT}`);});
  $$('[data-out="end"]',rail).forEach(o=>{o.hidden=S.n<NT; o.textContent=tx(`The sentence ends at “${wd(NT-1)}”.`,`这句话到“${wd(NT-1)}”就结束了。`);});
  $$('[data-act="gen"]',rail).forEach(b=>{b.disabled=S.n>=NT;});
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step]);
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+NB(s.obs());
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#kvb-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+NB(scenHTML(cur)):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the four scenarios above.','从上面四个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const p=chPoint(S.ctx), now=$('#kvb-ch-now',rail); if(!now) return;
    now.textContent=tpsS(p.t);
    $('#kvb-ch-prog i',rail).style.width=Math.min(100,p.tps/CH_GOAL*100)+'%';
    $('#kvb-ch-goal',rail).classList.toggle('met',p.ok); $('#kvb-ch-prog',rail).classList.toggle('met',p.ok);
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#kvb-ch-res',rail), html=NB(chResultHTML()); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
function toyStatus(){
  const c=check(S.n,S.mask);
  if(!S.cache) return {cls:'neutral',txt:tx('Every step recomputes all tokens','每步重算所有 token')};
  if(S.n<2) return {cls:'neutral',txt:tx('KV cache on','KV cache 已打开')};
  if(S.mask||!(c.max[1]>0)) return {cls:'ok',txt:tx('Cache matches a recompute','cache 和重算的一致')};
  return {cls:'bad',txt:tx('Cache is stale in layer 2','cache 在第 2 层过期了')};
}
function realStatus(){
  const r=real(S.ctx);
  if(mode==='challenge'){const p=chPoint(S.ctx); return p.ok?{cls:'ok',txt:tx(`Goal met: ${tpsS(p.t)} tokens per second`,`达成目标：每秒 ${tpsS(p.t)} 个 token`)}:{cls:'bad',txt:tx(`Not yet: ${tpsS(p.t)} tokens per second`,`还没达到：每秒 ${tpsS(p.t)} 个 token`)};}
  const k=shownSec(r.wo.t)/shownSec(r.w.t);
  return k<1.05?{cls:'neutral',txt:tx('Same speed with or without the cache','用不用 cache 一样快')}:{cls:'neutral',txt:tx(`With the cache: ${ratioS(k)} times as fast`,`用 cache 的速度是不用时的 ${ratioS(k)} 倍`)};
}
const STATUS_TIP=()=>S.view==='toy'?(S.cache?tx('This compares the keys and values kept in the KV cache with a full recompute at the latest step. In this lab, with the causal mask, they are equal to the last digit; with it off, layer 2’s differ.','这里把 KV cache 里存着的 key 和 value 与最近一步全部重算的结果比较。在这个 Lab 里，有因果 mask 时一位不差；关掉后第 2 层的不一样。')
    :tx('Without the KV cache nothing is kept: every decode step computes the keys and values of all tokens so far again.','不用 KV cache 时什么都不存：每个 decode 步都把目前所有 token 的 key 和 value 重新算一遍。'))
  :(mode==='challenge'?tx('Green when, by this lab’s estimate, Llama-3.1-8B without the KV cache writes at least 20 tokens per second at this context, that is, at most 50 ms per token.','按这个 Lab 的估计，不用 KV cache 时，Llama-3.1-8B 在这个上下文长度下每秒至少生成 20 个 token（每个 token 不超过 50 ms），就变绿。')
   :(()=>{const r=real(S.ctx), w=tS(r.w.t), wo=tS(r.wo.t), k=ratioS(shownSec(r.wo.t)/shownSec(r.w.t));
      return tx(`How many times as long one decode step takes without the KV cache as with it, at this context, by this lab’s estimate: ${wo} ÷ ${w} = ${k}.`,`在这个上下文长度下，按这个 Lab 的估计，不用 KV cache 时一个 decode 步的耗时是用 cache 时的几倍：${wo} ÷ ${w} = ${k}。`);})());
function drawHead(){
  const t=$('#kvb-st-title',app), sub=$('#kvb-st-sub',app), st=$('#kvb-status',app);
  let s;
  if(S.view==='toy'){
    t.textContent=tx('Small model: 2 layers, 2 heads per layer','小模型：2 层，每层 2 个 head');
    sub.textContent=S.n===0?tx('No decode step done yet','还没开始生成')
      :tx(`${S.n} of ${NT} decode steps done, ${S.cache?'with':'without'} the KV cache${S.mask?'':', causal mask off'}`,`已完成 ${S.n} / ${NT} 个 decode 步，${S.cache?'用':'不用'} KV cache${S.mask?'':'，关掉了因果 mask'}`);
    s=toyStatus();
  } else {
    t.textContent=NB(tx('Llama-3.1-8B on one H100 SXM, one conversation','一张 H100 SXM 跑 Llama-3.1-8B，一个对话'));
    sub.textContent=mode==='sandbox'&&presetNow()>=0?tx(`${ctxFullTok(S.ctx)} of context (the prompt), BF16`,`上下文 ${ctxFullTok(S.ctx)}（prompt 的长度），BF16`):tx(`${ctxFullTok(S.ctx)} of context, BF16`,`上下文 ${ctxFullTok(S.ctx)}，BF16`);
    s=realStatus();
  }
  st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=STATUS_TIP();
  const vs=$('#kvb-vseg',app); if(vs){vs.hidden=mode==='challenge'; A.ui.setPressed(vs,S.view);}
}
function legendHTML(){
  const it=(cls,txt,tip,jump)=>`<span data-tipx="${esc(tip)}" tabindex="0"${jump?` data-jump="${jump}"`:''}><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  if(S.view==='toy'){const c=check(S.n,S.mask), stale=showCheck()&&c.max[1]>0;
    return it('kvb-sw-comp',tx('computed in this step','这一步算出'),tx('A filled cell: in this decode step, the model computed this token’s key and value in this layer.','实心格：这个 decode 步在这一层算出了这个 token 的 key 和 value。'))+
      it('kvb-sw-read',tx('read from the KV cache','从 KV cache 读出'),tx('An amber outline: this decode step read this token’s key and value from the KV cache instead of computing them.','琥珀色框：这个 decode 步没有重算这个 token 的 key 和 value，而是从 KV cache 里读出。'))+
      it('kvb-sw-cur',tx('the latest step','最近一步'),tx('The shaded row is the decode step done last.','带底色的一行是最近完成的 decode 步。'))+
      (S.n>=1&&S.n<NT?it('kvb-sw-new',tx('underlined: the word the latest step wrote','下划线：最近一步生成的词'),tx('The underlined word at the top of the chart is the one the latest decode step wrote: step n takes in token n and writes the word after it.','图顶上带下划线的词，是最近一个 decode 步生成的：第 n 步读入第 n 个 token，生成它后面的词。')):'')+
      (stale?it('kvb-sw-stale',tx('changed, so a kept copy would be stale','变了，存下来的就过期了'),tx('A red number under a layer: that token’s key or value in the latest step differs from the one computed in the step that took the token in.','层下面的红色数字：这个 token 在最近一步算出的 key 或 value，和读入它的那一步算出的不一样。')):'');}
  const goal=`<span data-tipx="${esc(tx('The dashed line is the goal: 20 tokens per second, that is, 50 ms per token.','虚线是目标：每秒 20 个 token，也就是每个 token 50 ms。'))}" tabindex="0"><i class="sw lim"></i><span>${tx('dashed line: goal, 20 per second','虚线：目标，每秒 20 个')}</span></span>`;
  if(mode==='challenge') return goal;
  return it('w',tx('weights','权重'),tx(`The weights are the model’s ${bigS(P_TOTAL)} parameters, 2 bytes each in BF16: ${fint(P_TOTAL)} × 2 bytes = ${gbS(W_MEM)} = ${gib1(W_MEM)}. They stay in GPU memory and are shared by every conversation.`,`权重就是模型的参数，一共 ${bigS(P_TOTAL)} 个，BF16 下每个 2 字节：${fint(P_TOTAL)} × 2 字节 = ${gbS(W_MEM)} = ${gib1(W_MEM)}。它们一直放在显存里，所有对话共用。`))+
    it('kv','KV cache',tx('This conversation’s KV cache: the keys and values of every token of its context, 128 KiB per token.','这个对话的 KV cache：上下文里每个 token 的 key 和 value，每个 token 128 KiB。'))+
    it('free',tx('free','空闲'),tx('Free memory is GPU memory that neither the weights nor this conversation’s KV cache takes.','空闲是权重和这个对话的 KV cache 都没占用的显存。'));
}
function rowTip(s){
  const w=WORDS[A.lang()], inp=w[s-1], out=s<NT?w[s]:null;
  const a=out?tx(`Decode step ${s} writes “${out}”, the word after “${inp}”.`,`第 ${s} 个 decode 步生成“${out}”，也就是“${inp}”后面的词。`)
             :tx(`Decode step ${s} runs on “${inp}” to write the word after it, where this sentence stops.`,`第 ${s} 个 decode 步读入“${inp}”，生成它后面的词；这句话在这里结束。`);
  const b=s===1?tx(' It computes the keys and values of “<s>” in every layer, with or without the KV cache.','不管用不用 KV cache，它都在每一层算出“<s>”的 key 和 value。')
    :s===2?tx(` Without the KV cache it computes the keys and values of both tokens so far in every layer; with the cache, only those of “${inp}”, and it reads those of “<s>” from the cache.`,
              `不用 KV cache 时，它在每一层都要算出目前两个 token 的 key 和 value；用 cache 时只算“${inp}”的，“<s>”的从 cache 里读。`)
    :tx(` Without the KV cache it computes the keys and values of all ${s} tokens so far in every layer; with the cache, only those of “${inp}”, and it reads the other ${s-1} tokens’ from the cache.`,
        `不用 KV cache 时，它在每一层都要算出目前 ${s} 个 token 的 key 和 value；用 cache 时只算“${inp}”的，另外 ${s-1} 个 token 的从 cache 里读。`);
  return a+b;
}
function drawToy(){
  const g=$('#kvb-grid',app); if(!g) return;
  const n=S.n, c=check(n,S.mask), chk=showCheck();
  $$('.kvb-tok',g).forEach(t=>t.classList.toggle('kvb-new',n>=1&&n<NT&&+t.dataset.j===n));
  $$('.kvb-layer',g).forEach(layer=>{const l=+layer.dataset.l;
    $$('.kvb-srow',layer).forEach(row=>{const s=+row.dataset.s, done=s<=n;
      row.classList.toggle('kvb-done',done); row.classList.toggle('kvb-cur',s===n);
      if(l===0){const lb=row.querySelector('.kvb-rl'); lb.dataset.tipx=rowTip(s);}
      $$('.kvb-c',row).forEach(cell=>{const j=+cell.dataset.j, comp=done&&j<s&&(!S.cache||j===s-1), read=done&&S.cache&&j<s-1;
        cell.classList.toggle('kvb-comp',comp); cell.classList.toggle('kvb-read',read);});});
    const cr=layer.querySelector('.kvb-chk'); cr.hidden=!chk;
    if(chk){const lb=cr.querySelector('.kvb-rl');
      lb.dataset.tipx=tx(`For each earlier token: the largest difference between its key and value computed in step ${n} and those from the step that took it in. 0 means every number is exactly the same. A “—” marks the token this step took in; it has no earlier copy to compare with.`,`每个之前的 token：它在第 ${n} 步算出的 key 和 value，和读入它的那一步算出的相比，最大差多少。0 表示每个数值都一模一样。“—”是这一步读入的 token，还没有更早的结果可比。`);
      $$('.kvb-d',cr).forEach(dc=>{const j=+dc.dataset.j;
        if(j<n-1){const v=c.per[l][j]; dc.textContent=diffS(v); dc.classList.toggle('kvb-bad',v>0);}
        else {dc.textContent=j===n-1?'—':''; dc.classList.remove('kvb-bad');}});}});
  g.setAttribute('aria-label',tx(`Small model after ${stepN(n)} ${S.cache?'with':'without'} the KV cache: ${passN(counts(n,S.cache).passes)} so far.`,`小模型完成 ${stepN(n)}（${S.cache?'用':'不用'} KV cache）：目前 ${passN(counts(n,S.cache).passes)}。`));
}
function drawReal(){
  const r=real(S.ctx), sp=$('#kvb-sp',app); if(!sp) return;
  $('#kvb-sp-title',app).textContent=tx(`How fast one conversation gets its tokens at ${ctxTok(S.ctx)} of context, by this lab’s estimate`,`按这个 Lab 的估计，上下文 ${ctxK(S.ctx)} 个 token 时，一个对话每秒拿到几个 token`);
  const ch=mode==='challenge';
  const rows={w:{lab:ch?tx('With the KV cache (for comparison; off in this challenge)','用 KV cache（仅作对比，挑战中关着）'):tx('With the KV cache','用 KV cache'),t:r.w.t},wo:{lab:tx('Without the KV cache','不用 KV cache'),t:r.wo.t}};
  const mwrap=$('#kvb-memwrap',app); if(mwrap) mwrap.hidden=ch;
  $$('.kvb-sprow',sp).forEach(row=>{const o=rows[row.dataset.k], v=tpsV(o.t);
    row.querySelector('.kvb-spl').textContent=o.lab;
    row.querySelector('.kvb-spv').textContent=tx(`${tpsS(o.t)} tokens per second (${tS(o.t)} each)`,`每秒 ${tpsS(o.t)} 个（每个 ${tS(o.t)}）`);
    row.querySelector('.kvb-spt i').style.width=Math.min(100,v/150*100).toFixed(2)+'%';
    const gl=row.querySelector('.kvb-goal'); gl.hidden=mode!=='challenge'; gl.style.left=(CH_GOAL/150*100)+'%';
    row.dataset.tipx=row.dataset.k==='w'?tx(`With the KV cache, one decode step runs only the newest token and reads the other ${fint(S.ctx-1)} tokens’ keys and values from the cache: ${tS(o.t)}, estimated with this lab’s step model.`,`用 KV cache 时，一个 decode 步只算最新的一个 token，另外 ${fint(S.ctx-1)} 个 token 的 key 和 value 从 cache 里读：${tS(o.t)}，按这个 Lab 的耗时模型估计。`)
      :tx(`Without the KV cache, one decode step runs all ${fint(S.ctx)} tokens through the model again: ${tS(o.t)}, estimated with this lab’s step model.`,`不用 KV cache 时，一个 decode 步要把全部 ${fint(S.ctx)} 个 token 重新过一遍模型：${tS(o.t)}，按这个 Lab 的耗时模型估计。`);});
  $('#kvb-mem-title',app).textContent=NB(tx('Memory of the H100 SXM (80 GiB): the weights and this conversation’s KV cache','H100 SXM 的显存（80 GiB）：权重和这个对话的 KV cache'));
  const mem=$('#kvb-mem',app), cap=H100.mem*A.GiB, pw=r.wMem/cap*100, pk=r.kv/cap*100, w=mem.clientWidth||400;
  const mw=mem.querySelector('.kvb-mw'), mk=mem.querySelector('.kvb-mk'), mf=mem.querySelector('.kvb-mf');
  mw.style.width=pw.toFixed(3)+'%'; mk.style.left=pw.toFixed(3)+'%'; mk.style.width=pk.toFixed(3)+'%'; mf.style.left=(pw+pk).toFixed(3)+'%'; mf.style.width=Math.max(0,100-pw-pk).toFixed(3)+'%';
  const fit=(el,txt,pc)=>{el.firstChild.textContent=(w*pc/100>=txt.length*6.6+10)?txt:'';};
  fit(mw,gib1(r.wMem),pw); fit(mk,'KV '+memS(r.kv),pk);
  mem.dataset.tipx=tx(`The H100 SXM’s 80 GiB hold the weights (${gib1(r.wMem)}) and this conversation’s KV cache (${memS(r.kv)} for ${fint(S.ctx)} tokens); ${gib1(cap-r.wMem-r.kv)} stay free.`,`H100 SXM 的 80 GiB 里放着权重（${gib1(r.wMem)}）和这个对话的 KV cache（${fint(S.ctx)} 个 token，${memS(r.kv)}）；还空着 ${gib1(cap-r.wMem-r.kv)}。`);
}
/* metric cards: ▲▼ only after the learner's own change, from the numbers as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
const unitOf=v=>{const m=String(v).match(/(ms|s|min|KiB|MiB|GiB)$/); return m?m[1]:'';};
const toBase=(v,u)=>({ms:v/1000,s:v,min:v*60,KiB:v*1024,MiB:v*1048576,GiB:v*1073741824}[u]||v);
function metricsDef(){
  if(S.view==='toy'){const c=counts(S.n,S.cache), hid=hide100();
    return [
     {k:'passes',label:tx('Token-passes so far','目前的 token 前向'),v:fint(c.passes),sub:tx(`in ${S.n} decode step${S.n===1?'':'s'}`,`${S.n} 个 decode 步合计`),tip:hid?tx('A token-pass is one token going through all the layers once.','一次 token 前向是一个 token 把所有层过一遍。'):tx('A token-pass is one token going through all the layers once. Without the KV cache, decode step n makes n of them; with it, 1.','一次 token 前向是一个 token 把所有层过一遍。不用 KV cache 时，第 n 个 decode 步要做 n 次；用了只要 1 次。')},
     {k:'p100',label:tx('Token-passes to write 100 tokens','生成 100 个 token 的 token 前向'),v:hid?'?':fint(c.p100),
      sub:hid?(pred?(step===2?tx('shown once you have written 4 words','生成 4 个词后显示'):tx('shown in part 3','第 3 部分显示')):tx('shown after you predict','预测后显示')):(S.cache?tx('with the KV cache','用 KV cache'):tx('without the KV cache','不用 KV cache')),
      tip:hid?tx('Hidden until part 3 shows the count, so it does not give away your prediction.','在第 3 部分给出结果之前先隐藏，免得提前透露预测的答案。'):tx(`Starting from <s>, writing 100 tokens takes 1 + 2 + … + 100 = ${fint(tri(100))} token-passes without the KV cache and 100 with it.`,`从 <s> 开始生成 100 个 token：不用 KV cache 要 1 + 2 + … + 100 = ${fint(tri(100))} 次 token 前向，用了只要 100 次。`),hid},
     {k:'reads',label:tx('K and V read from the cache','从 cache 读出的 K 和 V'),v:fint(c.reads),sub:tx(`per layer, in ${S.n} step${S.n===1?'':'s'}`,`每层，${S.n} 步合计`),tip:tx('How many times a decode step read an earlier token’s key and value from the KV cache instead of computing them again, counted in one layer. Without the cache it stays 0.','decode 步从 KV cache 里读出之前 token 的 key 和 value、而不重算的次数，按一层计。不用 cache 时一直是 0。')},
     {k:'nums',label:tx('Numbers in the KV cache','KV cache 里存的数值'),v:fint(c.nums),sub:S.cache?tx(`${S.n} tokens × ${PER_TOK}`,`${S.n} 个 token × ${PER_TOK} 个数值`):tx('nothing is kept','什么都不存'),tip:tx(`Each token keeps a key and a value of 4 numbers for each of the 2 heads in each of the 2 layers: 2 × 2 × 2 × 4 = ${PER_TOK} numbers. Without the cache nothing is kept.`,`每个 token 在 2 层、每层 2 个 head 上各存一个 key 和一个 value，都是 4 维：2 × 2 × 2 × 4 = ${PER_TOK} 个数值。不用 cache 时什么都不存。`)}];}
  const r=real(S.ctx), est=t=>tx(`${tpsS(t)} tokens per second, estimated`,`每秒 ${tpsS(t)} 个 token（估计）`);
  const tw={k:'tw',label:mode==='challenge'?tx('Time per token, with the KV cache (for comparison)','每个 token 的耗时（用 KV cache，仅作对比）'):tx('Time per token, with the KV cache','每个 token 的耗时（用 KV cache）'),v:tS(r.w.t),sub:est(r.w.t),tip:tx('One decode step with the cache: it computes only the newest token but reads the weights and the whole cache, so reading memory decides its time. Estimated with this lab’s step model on one H100 SXM.','用 cache 时的一个 decode 步：只算最新的一个 token，却要把权重和整个 cache 各读一遍，所以读显存决定耗时。按这个 Lab 的耗时模型，在一张 H100 SXM 上估计。')};
  const two={k:'two',label:mode==='challenge'?tx('Time per token, without the KV cache','每个 token 的耗时（不用 KV cache）'):tx('Time per token, without it','每个 token 的耗时（不用）'),v:tS(r.wo.t),sub:est(r.wo.t),tip:tx('One decode step without the cache: it reads only the weights but computes all the context’s tokens again, so, except at the shortest contexts, arithmetic decides its time, which grows with the context. Estimated with this lab’s step model on one H100 SXM.','不用 cache 时的一个 decode 步：只读权重，却要把上下文里的全部 token 重算一遍，所以除了最短的上下文，都是计算决定耗时，耗时随上下文增长。按这个 Lab 的耗时模型，在一张 H100 SXM 上估计。')};
  if(mode==='challenge') return [two,tw];
  return [
   {k:'kvtok',label:tx('KV cache per token','每个 token 的 KV cache'),v:'128 KiB',sub:tx('32 layers, 8 KV heads, BF16','32 层，每层 8 个 KV head，BF16'),tip:tx('Every token of the context keeps a key and a value of 128 numbers for each of the 8 KV heads in each of the 32 layers, 2 bytes per number in BF16: 131,072 bytes = 128 KiB (131 KB).','上下文里的每个 token，在 32 层、每层 8 个 KV head 上各存一个 key 和一个 value，都是 128 维，BF16 下每个数值占 2 字节：131,072 字节 = 128 KiB（131 KB）。')},
   {k:'kvctx',label:tx('KV cache, whole context','整段上下文的 KV cache'),v:memS(r.kv),sub:tokN(S.ctx),tip:tx(`This conversation’s KV cache: 128 KiB per token times the tokens of context, ${fint(S.ctx)} × 131,072 bytes = ${gbS(r.kv)} = ${memS(r.kv)}. It stays in GPU memory for as long as the answer is being written.`,`这个对话的 KV cache：每个 token 128 KiB 乘以上下文的 token 数，${fint(S.ctx)} × 131,072 字节 = ${gbS(r.kv)} = ${memS(r.kv)}。写回答的整个过程中，它都一直占着显存。`)},
   tw,two];
}
function drawMetrics(){
  const host=$('#kvb-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join(); host.classList.toggle('kvb-m2',defs.length===2);
  if(host._sig!==sig){host._sig=sig; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt='';
    const prev=shownPrev[m.k];
    /* same unit: compare the shown numbers at the shown decimals; different units (266.7 ms → 1.49 s): compare in one unit */
    if(userAct&&!m.hid&&prev!=null&&prev!==m.v){const ua=unitOf(prev), ub=unitOf(m.v), same=ua===ub, a=same?numOf(prev):toBase(numOf(prev),ua), b=same?numOf(m.v):toBase(numOf(m.v),ub);
      if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,same?{dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.hid?null:m.v;});
}
/* Show the math: every line redoes from the numbers it shows (R6.6), and every term has its own tooltip (R1.4, review 8 point 3).
   H1/H2 (the owner, 2026-10-09): short blocks, like a page of a good textbook. Each block has one lead line in words, then a small table:
   the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the result, each on its
   own line with the "=" signs lined up. groups: [{name, terms:[TeX per column], rows:[{l, c:[html per column]} | {r:html}]}] */
TIP.gbu=()=>tx('GB and TB here are 10⁹ and 10¹² bytes, to match the bandwidth in TB/s.','这里的 GB、TB 是 10⁹、10¹² 字节，和带宽的 TB/s 一致。');
const MX=x=>A.ui.math(x);
function mathBlock(lead,groups,note){const n=Math.max(...groups.map(g=>g.terms.length)), eq=`<td class="kvb-meq">${MX('=')}</td>`;
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?r.c.map((x,i)=>`<td${i===r.c.length-1&&r.c.length<n?` colspan="${n-i}"`:''}>${x}</td>`).join(''):`<td colspan="${n}" class="kvb-mres">${r.r}</td>`}</tr>`;
  const body=groups.map(g=>`<tr class="kvb-mname"><th scope="row">${g.name}</th>${eq}${g.terms.map((x,i)=>`<td${i===g.terms.length-1&&g.terms.length<n?` colspan="${n-i}"`:''}>${MX(x)}</td>`).join('')}</tr>${g.rows.map(tr).join('')}`).join('');
  return `<div class="kvb-mb">${lead?`<p class="kvb-mlead">${lead}</p>`:''}<div class="kvb-fx"><table class="kvb-mt">${body}</table></div>${note?`<p class="kvb-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.kvb-mt'); if(!tb) return; const e0=tb.querySelector('td.kvb-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('kvb-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('kvb-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="kvb-mc${r?'':' kvb-mc0'}">${x}</span>`;
    h+=`<tr class="kvb-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="kvb-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="kvb-mband${g.tall?' kvb-mtall':''}"><td class="kvb-meq"><span class="kvb-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="kvb-mfl">${units.map(u=>`<span class="kvb-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="kvb-meq">${E}</td><td class="kvb-mres">${r.l?`<span class="kvb-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='kvb-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.kvb-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.kvb-fx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
function mathHTML(){
  const t=(k,x)=>T_(k,x), NS=' ';
  if(S.view==='toy'){const n=S.n, c=check(n,S.mask), rows=[];
    rows.push(mathBlock(tx('Every token keeps a key and a value for each head in every layer.','每个 token 在每一层、每个 head 上都存一个 key 和一个 value。'),
      [{name:tx('KV cache per token in the small model','小模型每个 token 的 KV cache'),terms:['2','\\times',tx('\\text{layers}','\\text{层数}'),'\\times',tx('\\text{heads}','\\text{head 数}'),'\\times','\\text{head_dim}'],
        rows:[{c:[t('skv2','2'),'×',t('slayers','2'),'×',t('sheads','2'),'×',t('shd','4')]},{r:`<b>${PER_TOK} ${tx('numbers','个数值')}</b>`}]}]));
    if(n>=1){const r0=n<=5?Array.from({length:n},(_,i)=>i).join(' + '):`0 + 1 + … + ${n-1}`;
      rows.push(mathBlock(tx(`Without the cache, step k runs all k tokens so far through the model again. Here there are ${stepN(n)}.`,`不用 cache 时，第 k 步要把目前的 k 个 token 重新过一遍模型。这里一共 ${stepN(n)}。`),
        [{name:tx('Token-passes without the cache','不用 cache 的 token 前向次数'),terms:['1 + 2 + \\ldots + n','=','\\frac{n(n + 1)}{2}'],
          rows:[{l:T_('n',`n = ${fint(n)}`),c:[t('sumn',sumTxt(n)),'=',t('gauss',`${n} × ${n+1} ÷ 2`)]},{r:`<b>${passN(tri(n))}</b>`}]}]));
      rows.push(mathBlock(tx('With the cache, each step runs only its newest token, and in each layer reads the earlier tokens’ keys and values from the cache.','用 cache 时，每一步只算最新的一个 token，并在每一层从 cache 里读之前 token 的 key 和 value。'),
        [{name:tx('Token-passes with the cache','用 cache 的 token 前向次数'),terms:['n'],rows:[{c:[t('pass1',tx(`${passN(n)} (one per step)`,`${passN(n)}（每步一次）`))]}]},
         {name:tx('Reads from the cache per layer','每层从 cache 读的次数'),terms:['0 + 1 + \\ldots + (n - 1)'],rows:[{c:[t('reads',r0)]},{r:`<b>${fint(tri(n-1))}</b>`}]}]));}
    if(n>=2){const j=Math.min(1,n-2), now=forward(n,S.mask), first=forward(j+1,S.mask), a=first.K[1][j].slice(0,HD), b=now.K[1][j].slice(0,HD), vec=x=>'('+x.map(v=>fixd(v,4).replace('-','−')).join(', ')+')';
      const dm=Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
      rows.push(`<div class="kvb-mb"><p class="kvb-mlead">${tx(`Layer 2, head 1, the key of “${esc(wd(j))}” (token ${j+1})`,`第 2 层 head 1 里“${esc(wd(j))}”（第 ${j+1} 个 token）的 key`)}</p><p class="kvb-mvec">${tx(`from step ${j+1}, when it was first computed`,`第 ${j+1} 步第一次算出时`)} ${vec(a)}<br>${tx(`from step ${n}, `,`第 ${n} 步，`)}${T_('mask',S.mask?tx('causal mask on','因果 mask 开'):tx('causal mask off','因果 mask 关'))} ${vec(b)}<br>${tx('largest difference','最大差')} = <b>${fixd(dm,4)}</b>${tx('.','。')}${c.max[0]===0&&c.max[1]===0?tx(' In both layers, every earlier key and value is exactly the same.','两层里之前的 key 和 value 每个数值都一模一样。'):tx(` Over all tokens’ keys and values, layer 2 differs by up to ${diffS(c.max[1])} and layer 1 by ${diffS(c.max[0])}.`,`所有 token 的 key 和 value 合起来，第 2 层最多差 ${diffS(c.max[1])}，第 1 层差 ${diffS(c.max[0])}。`)}</p></div>`);}
    rows.push(`<p class="small muted">${tx('The key vectors are the small model’s own numbers, computed in your browser; in this lab, with the causal mask, the two rows agree in every digit.','这些 key 向量是小模型在你的浏览器里算出的数值；在这个 Lab 里，有因果 mask 时，两行每一位都一样。')}</p>`);
    return rows.join('');}
  const r=real(S.ctx), C=S.ctx, TF=v=>fixd(v/1e12,v<1e12?4:3)+NS+'TFLOP', GBv=v=>fixd(v/1e9,3)+NS+DEF('GB',TIP.gbu());
  const peak=H100.peak*EFF.mfu, bw=H100.bw*EFF.mbu, WL=t('wl','2 × '+fint(P_LAYERS)), WH=t('wh','2 × '+fint(P_HEAD)), WR=t('wr',fint(W_READ));
  const ceN=fixd(peak/1e12,2)+NS+'TFLOP/s', beN=fixd(bw/1e12,2)+NS+'TB/s';
  /* the two steps side by side, term for term: what changes between them and what does not (R2.7, review 8 point 4) */
  const step=(lead,o,flDesc,byDesc)=>mathBlock(lead,[
    {name:tx('Compute time','计算时间'),terms:[tx('\\text{FLOP}','\\text{运算量}'),'\\div',tx('\\text{compute rate}','\\text{有效算力}')],
      rows:[{r:`(${flDesc}) ÷ ${t('effc',ceN)}`},{c:[TF(o.fl),'÷',ceN]},{r:`${fixd(o.tc*1e3,2)}${NS}ms`}]},
    {name:tx('Memory time','读显存时间'),terms:[tx('\\text{bytes read}','\\text{读的字节数}'),'\\div',tx('\\text{bandwidth}','\\text{有效带宽}')],
      rows:[{r:`(${byDesc}) ÷ ${t('effm',beN)}`},{c:[GBv(o.by),'÷',beN]},{r:`${fixd(o.tm*1e3,2)}${NS}ms`}]},
    {name:DEF(tx('Step time','一步的耗时'),TIP.overlap()),terms:[tx('\\max(\\text{compute time},\\ \\text{memory time})','\\max(\\text{计算时间},\\ \\text{读显存时间})')],
      rows:[{c:[`max(${fixd(o.tc*1e3,2)}${NS}ms, ${fixd(o.tm*1e3,2)}${NS}ms)`]},{r:`<b>${tS(o.t)}</b>${o.tm>=o.tc?tx(', set by memory','，由读显存决定'):tx(', set by arithmetic','，由计算决定')}`}]}]);
  const rows=[];
  rows.push(mathBlock(tx('Every token keeps a key and a value for each KV head in every layer, 2 bytes per number in BF16.','每个 token 在每一层、每个 KV head 上存一个 key 和一个 value，BF16 下每个数值占 2 字节。'),
    [{name:tx('KV cache per token','每个 token 的 KV cache'),terms:['2','\\times',tx('\\text{layers}','\\text{层数}'),'\\times',tx('\\text{KV heads}','\\text{KV head 数}'),'\\times','\\text{head_dim}','\\times',tx('\\text{bytes per number}','\\text{每个数值的字节数}')],
      rows:[{c:[t('kv2','2'),'×',t('layers','32'),'×',t('kvheads','8'),'×',t('hdim','128'),'×',t('bytes2','2')]},
        {r:`131,072 ${tx('bytes','字节')} (<b>128${NS}${DEF('KiB',TIP.kib())}</b>)`}]}]));
  rows.push(mathBlock(tx('Every token of the context keeps one such entry in the cache.','上下文里的每个 token 都在 cache 里占这么一份。'),
    [{name:tx('KV cache for the context','整段上下文的 KV cache'),terms:[tx('\\text{tokens}','\\text{token 数}'),'\\times',tx('\\text{KV cache per token}','\\text{每个 token 的 KV cache}')],
      rows:[{c:[t('ctx',fint(C)),'×',`131,072 ${tx('bytes','字节')}`]},{r:`${fint(r.kv)} ${tx('bytes','字节')} (${gbS(r.kv)}, <b>${memD(r.kv)}</b>)`}]}]));
  rows.push(step(tx(`<b>With the cache</b>, one decode step at ${fint(C)} tokens. The GPU computes and reads memory at the same time, so the step takes the longer of the two.`,`<b>用 cache</b>，上下文 ${fint(C)} 个 token 时的一个 decode 步。GPU 计算和读显存同时进行，所以一步的耗时取两者中较长的那个。`),r.w,
    `${WL} × ${t('tok1','1')} + ${WH} + ${t('attnpair','4 × 32 × 4,096')} × ${t('pairs1',fint(C))}`,
    `${WR} + ${t('kvread',`${fint(C-1)} × 131,072`)}`));
  rows.push(step(tx(`<b>Without the cache</b>, one decode step at ${fint(C)} tokens: every token of the context is computed again.`,`<b>不用 cache</b>，上下文 ${fint(C)} 个 token 时的一个 decode 步：整段上下文的每个 token 都要重算。`),r.wo,
    `${WL} × ${t('tokall',fint(C))} + ${WH} + ${t('pairsall',`2 × 32 × 4,096 × ${fint(C)} × ${fint(C+1)}`)}`,
    `${WR} + ${t('kvnone','0 × 131,072')}`));
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){S.view='real'; if(!CH_STOPS.includes(S.ctx)) S.ctx=CH_START;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const toy=$('#kvb-toy',app), rl=$('#kvb-real',app); if(toy) toy.hidden=S.view!=='toy'; if(rl) rl.hidden=S.view!=='real';
  drawHead();
  const lg=$('#kvb-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  if(S.view==='toy') drawToy(); else drawReal();
  drawMetrics();
  const mb=$('#kvb-mathbody',app), mh=NB(mathHTML()); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;} fitMath(mb);
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(g=>{const k=g.dataset.key; if(k==='view') A.ui.setPressed(g,S.view); else if(k==='cache') A.ui.setPressed(g,S.cache?'1':'0'); else if(k==='mask') A.ui.setPressed(g,S.mask?'1':'0');});
  $$('input[data-key="n"]',app).forEach(i=>{if(+i.value!==S.n) i.value=S.n; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=tx(`${S.n} of ${NT}`,`${S.n} / ${NT}`);});
  $$('input[data-key="ctx"]',app).forEach(i=>{const v=nearest(CTX,S.ctx); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=ctxTok(S.ctx);});
  $$('input[data-key="chctx"]',app).forEach(i=>{const v=nearest(CH_STOPS,S.ctx); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=ctxTok(S.ctx);});
  const lock=mode==='challenge', pl=preLock(), c=$('#kvb-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#kvb-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=NB(tx('In the challenge you can change only the context. Everything else is fixed: Llama-3.1-8B on one H100 SXM, one conversation, with the KV cache off.','挑战中只能改上下文长度。其余设置固定：一张 H100 SXM 跑 Llama-3.1-8B，只服务一个对话，不用 KV cache。'));
  const nw=$('#kvb-ctl-n-wrap',app); if(nw) nw.hidden=pl; const nl=$('#kvb-nlock',app); if(nl) nl.hidden=!pl;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return;
  if(S.view==='toy'){const c=counts(S.n,S.cache); A.announce(tx(`${stepN(S.n)} ${S.cache?'with':'without'} the KV cache: ${passN(c.passes)}.`,`${stepN(S.n)}（${S.cache?'用':'不用'} KV cache）：${passN(c.passes)}。`));}
  else {const r=real(S.ctx); A.announce(tx(`At ${fint(S.ctx)} tokens of context, estimated: ${tS(r.w.t)} per token with the KV cache, ${tS(r.wo.t)} without.`,`上下文 ${fint(S.ctx)} 个 token 时，估计每个 token 用 KV cache 要 ${tS(r.w.t)}，不用要 ${tS(r.wo.t)}。`));}},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  if(preLock()&&p.n!=null) p={...p,n:0};
  Object.assign(S,p);
  S.n=Math.max(0,Math.min(NT,Math.round(S.n)));
  if(mode==='challenge'){S.view='real'; if(!CH_STOPS.includes(S.ctx)) S.ctx=CH_START;}
  render();
}
/* a change the learner made: metric cards may show a ▲▼ badge */
function byLearner(fn){userAct=true; maskBack=false; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; maskBack=s.id==='keep'&&!S.mask; if(s.enter) Object.assign(S,s.enter());
  const st=$('#kvb-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg={steps:'#kvb-ic-gen',check:'#kvb-ic-mask',keep:'#kvb-ic-cache',real:'#kvb-ic-ctx'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#kvb-rail .step.current',app)||$('#kvb-rail',app));}
/* the result of a part comes into view after a click that produced it: scroll just enough to show all of it, never pushing its top
   under the top bar and the mini result bar (R3.10) */
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  /* taller than the screen: a top already on screen stays put, so the control just used stays put too (r12 N7: on a phone the
     challenge's slider left the screen at every release); otherwise its top comes first */
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;                     // already in view
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;  // down just enough to show its end, or up to its top
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* After Next or Back, a mode switch or a pick that shortens the card, the rail can end above the sticky stage's bottom; the stage then
   slides up under the top bar with the bench's bottom edge (r12 N4: entering part 5 at 1440 x 900 hid its title, badge and legend).
   Once any scroll has settled, scroll up just enough for the stage to stick below the top bar again. The card stays on screen: the
   bench then ends at the bottom of the screen, and the card's top is never pushed below it. Wide screens only (the stage is sticky). */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}   // a reveal still scrolling
    const st=$('#kvb-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#kvb-rail .step.current',app)||$('#kvb-rail .railcard',app)||$('#kvb-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* the learner's own scrolling wins: a wheel, a swipe or a scrolling key cancels a pending correction */
window.addEventListener('wheel',stageStop,{passive:true}); window.addEventListener('touchmove',stageStop,{passive:true});
window.addEventListener('keydown',e=>{if(/^(PageUp|PageDown|ArrowUp|ArrowDown|Home|End| )$/.test(e.key)&&!(e.target.closest&&e.target.closest('input,select,textarea,button,[role="button"]'))) stageStop();});
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#kvb-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#kvb-insight',app)); else revealBox($('#kvb-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,view:'real',ctx:CH_START};}
  if(m==='sandbox'){S.view='real'; sbTouched=false;}
  A.shell.setMode(app,m,DESCS());
  const st=$('#kvb-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.view==='toy'?'t':'r',s.n,s.cache?1:0,s.mask?1:0,s.ctx].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [v,n,c,m,C]=p, o={view:v==='t'?'toy':(v==='r'?'real':null),n:+n,cache:c==='1',mask:m==='1',ctx:+C};
  if(!o.view||!Number.isInteger(o.n)||o.n<0||o.n>NT||!/^[01]$/.test(c)||!/^[01]$/.test(m)||!Number.isInteger(o.ctx)||o.ctx<1||o.ctx>LL.maxCtx) return null;
  return o;
}

/* =====================================================================
   interview drill: seeded, from the engine; every line redoes from the numbers it shows (R6.6)
   ===================================================================== */
const DRILL_M=[{name:'Llama-3.2-1B',L:16,kv:8,hd:64},{name:'Llama-3.1-8B',L:32,kv:8,hd:128},{name:'Mistral-7B-v0.1',L:32,kv:8,hd:128},{name:'Llama-3.1-70B',L:80,kv:8,hd:128}];
const DRILL_P=[2048,4096,8192,32768], DRILL_A=[200,500,1000];
function makeDrill(seed){const r=mulberry(seed*7919+13), pick=a=>a[Math.floor(r()*a.length)];
  const m=pick(DRILL_M), P=pick(DRILL_P), Aa=pick(DRILL_A), per=2*m.L*m.kv*m.hd*2, endTok=P+Aa-1;
  return {seed,m,P,A:Aa,per,endTok,bytes:per*endTok,without:Aa*P+Aa*(Aa-1)/2,with:P+Aa-1};}
/* "a" or "an" before a number, by how it is read aloud (8,192: an eight thousand …) */
const an=n=>{const t=String(n); return /^8/.test(t)||/^1[18](\d{3})*$/.test(t)?'an':'a';};
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#kvb-drill-q',app), a=$('#kvb-drill-a',app);
  qd.innerHTML=tx(`<b>${d.m.name}</b> (${d.m.L} layers, ${d.m.kv} KV heads, head_dim ${d.m.hd}) runs in BF16. It takes in ${an(d.P)} <b>${fint(d.P)}</b>-token prompt and writes ${an(d.A)} <b>${fint(d.A)}</b>-token answer. How big is its KV cache when the answer is done, and how many token-passes does the answer take without a KV cache, and with one?`,
                  `<b>${d.m.name}</b>（${d.m.L} 层，${d.m.kv} 个 KV head，head_dim ${d.m.hd}）用 BF16 运行。它读入 <b>${fint(d.P)}</b> 个 token 的 prompt，写出 <b>${fint(d.A)}</b> 个 token 的回答。回答写完时 KV cache 有多大？这段回答不用 KV cache 要做多少次 token 前向，用了又要多少次？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const kib=d.per/1024;
  a.innerHTML=tx(`<p><b>${memS(d.bytes)}; ${bigN(d.without)} token-passes without a cache, ${fint(d.with)} with one.</b></p><p>KV cache per token = 2 (a key and a value) × ${d.m.L} layers × ${d.m.kv} KV heads × ${d.m.hd} × 2 bytes = ${fint(d.per)} bytes = ${fint(kib)} KiB.</p><p>After the last decode step the cache holds the ${fint(d.P)} + ${fint(d.A)} − 1 = ${fint(d.endTok)} tokens taken in so far (the last token is written but not yet taken in): ${fint(d.endTok)} × ${fint(kib)} KiB = ${memS(d.bytes)}.</p><p>Without a cache, step k runs ${fint(d.P)} + k − 1 tokens (step 1 takes in the whole prompt), so ${fint(d.A)} steps cost ${fint(d.A)} × ${fint(d.P)} + ${fint(d.A)} × ${fint(d.A-1)} ÷ 2 = ${fint(d.A*d.P)} + ${fint(d.A*(d.A-1)/2)} = ${fint(d.without)}.</p><p>With a cache, the prefill runs the ${fint(d.P)} prompt tokens and each later decode step one: ${fint(d.P)} + ${fint(d.A-1)} = ${fint(d.with)}.</p>`,
                 `<p><b>${memS(d.bytes)}；不用 cache 要 ${bigN(d.without)} 次 token 前向，用了只要 ${fint(d.with)} 次。</b></p><p>每个 token 的 KV cache = 2（key 和 value）× ${d.m.L} 层 × ${d.m.kv} 个 KV head × ${d.m.hd} × 2 字节 = ${fint(d.per)} 字节 = ${fint(kib)} KiB。</p><p>最后一个 decode 步之后，cache 里存着已经读入的 ${fint(d.P)} + ${fint(d.A)} − 1 = ${fint(d.endTok)} 个 token（最后一个 token 已写出，还没读入）：${fint(d.endTok)} × ${fint(kib)} KiB = ${memS(d.bytes)}。</p><p>不用 cache 时，第 k 步要算 ${fint(d.P)} + k − 1 个 token（第 1 步读入整个 prompt），所以 ${fint(d.A)} 步一共 ${fint(d.A)} × ${fint(d.P)} + ${fint(d.A)} × ${fint(d.A-1)} ÷ 2 = ${fint(d.A*d.P)} + ${fint(d.A*(d.A-1)/2)} = ${fint(d.without)} 次。</p><p>用 cache 时，prefill 算 ${fint(d.P)} 个 prompt token，之后每个 decode 步 1 个：${fint(d.P)} + ${fint(d.A-1)} = ${fint(d.with)} 次。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={n:{en:'Decode steps done',zh:'已完成的 decode 步'},cache:{en:'KV cache',zh:'KV cache'},mask:{en:'Causal mask',zh:'因果 mask'},ctx:{en:'Context',zh:'上下文长度'}};
/* every term of every formula line has its own tooltip (R1.4, R6.15): what the number is and where it comes from, then what a
   click does. Derived terms explain their derivation here, not in a gray note under the formula (review 8 point 3). */
const TERM={
 n:{en:'The number of decode steps the small model has done; after step n, n tokens have gone through the model.',zh:'小模型已经完成的 decode 步数；第 n 步之后，已经有 n 个 token 过了模型。'},
 cache:{en:'Whether the small model keeps each token’s keys and values after the step that first computes them.',zh:'小模型是否把每个 token 第一次算出的 key 和 value 存下来。'},
 mask:{en:'With the causal mask on, no token looks at later tokens; off, every token looks at every other one.',zh:'因果 mask 开着时，没有 token 看后面的 token；关掉后，每个 token 都看其他所有 token。'},
 ctx:{en:'The tokens one decode step works on: the prompt plus everything written so far.',zh:'一个 decode 步要处理的 token：prompt 加上目前已生成的全部。'},
 model:()=>tx(`From Llama-3.1-8B’s config.json: 32 layers, 8 KV heads, head_dim 128, ${bigS(P_TOTAL)} parameters in all (${fint(P_TOTAL)}).`,`来自 Llama-3.1-8B 的 config.json：32 层，8 个 KV head，head_dim 128，一共 ${bigS(P_TOTAL)} 个参数（${fint(P_TOTAL)} 个）。`),
 gpu:{en:'The H100 SXM’s datasheet: BF16 1,979 TFLOPS with sparsity, so 989.5 TFLOP/s dense, and 3.35 TB/s of memory bandwidth.',zh:'H100 SXM 的规格表：BF16 标着 1,979 TFLOPS（with sparsity），稠密是 989.5 TFLOP/s；显存带宽 3.35 TB/s。'},
 eff:{en:'This lab assumes a step reaches 50% of the peak for arithmetic and 60% of the peak for memory.',zh:'本 Lab 假设一步能用到峰值的 50%（计算）和 60%（读显存）。'},
 /* Llama-3.1-8B's shape (config.json) */
 kv2:{en:'The 2 counts the two vectors a token keeps for each KV head in every layer: a key and a value.',zh:'2 是一个 token 在每一层、每个 KV head 上存的两个向量：一个 key 和一个 value。'},
 layers:{en:'32 is Llama-3.1-8B’s number of layers (num_hidden_layers in its config.json), and every layer keeps its own keys and values.',zh:'32 是 Llama-3.1-8B 的层数（config.json 里的 num_hidden_layers），每一层都存自己的 key 和 value。'},
 kvheads:{en:'8 is Llama-3.1-8B’s number of KV heads per layer (num_key_value_heads in its config.json): its 32 heads share them, one for every 4 heads.',zh:'8 是 Llama-3.1-8B 每层的 KV head 数（config.json 里的 num_key_value_heads）：它的 32 个 head 共用这 8 个，每 4 个共用一个。'},
 hdim:{en:'128 is head_dim, the length of each key and each value: the hidden size d = 4,096 divided among 32 heads (head_dim in its config.json).',zh:'128 是 head_dim，也就是每个 key 和每个 value 的维数：隐藏维度 d = 4,096 分给 32 个 head，每个 head 分到 128 维（config.json 里的 head_dim）。'},
 bytes2:{en:'BF16 stores each number in 2 bytes.',zh:'BF16 下每个数值占 2 字节。'},
 wl:()=>tx(`Llama-3.1-8B multiplies ${fint(P_LAYERS)} parameters (${bigS(P_LAYERS)}) for every token it computes: those of its 32 layers and the final RMSNorm. Each costs 2 FLOPs, a multiply and an add, so one token through the layers takes 2 × ${fint(P_LAYERS)} FLOPs.`,
           `Llama-3.1-8B 每算一个 token，要用 ${fint(P_LAYERS)} 个参数（${bigS(P_LAYERS)}）做乘法：32 层加上最后的 RMSNorm。每个参数 2 FLOP（一次乘、一次加），所以一个 token 过一遍各层要 2 × ${fint(P_LAYERS)} FLOP。`),
 wh:()=>tx(`The LM head has ${fint(P_HEAD)} parameters (4,096 × 128,256): it turns the newest token’s final vector into a logit for every token in the vocabulary, once per step, at 2 FLOPs per parameter.`,
           `LM head 有 ${fint(P_HEAD)} 个参数（4,096 × 128,256）：它把最新一个 token 最后的向量变成词表里每个 token 的 logit，每一步只算一次，每个参数 2 FLOP。`),
 wr:()=>tx(`A decode step reads ${fint(W_READ)} bytes of weights (${gib1(W_READ)}): every parameter it multiplies, at 2 bytes in BF16; the embedding table is only looked up, one row per token. With the ${gib1(2*P_EMB)} embedding table the whole model is ${gib1(W_MEM)}. The GPU streams these parameters from memory into its compute units once per step, and that one read serves every token the step computes.`,
           `一个 decode 步要读 ${fint(W_READ)} 字节的权重（${gib1(W_READ)}）：参与乘法的每个参数，BF16 下各 2 字节；embedding 表只需每个 token 查一行，不用整张读。加上 ${gib1(2*P_EMB)} 的 embedding 表，整个模型是 ${gib1(W_MEM)}。每一步，GPU 都把这些参数从显存读进计算单元一次，这一次读供这一步算的所有 token 共用。`),
 attnpair:{en:'4 × 32 × 4,096 is the arithmetic of one query–key pair: for each of the hidden size’s d = 4,096 numbers, 2 FLOPs for the score (query · key) and 2 for adding the value, in each of the 32 layers.',zh:'4 × 32 × 4,096 是一对 query 和 key 的计算量：隐藏维度 d = 4,096 的每一维，算得分（query · key）2 FLOP，加上 value 2 FLOP，32 层都要算。'},
 /* the two steps, term by term (review 8 points 3 and 4) */
 tok1:{en:'1 is the number of tokens a step with the cache computes: only the newest one, in every layer.',zh:'1 是用 cache 时一步要算的 token 数：只有最新的那一个，每一层都算。'},
 tokall:()=>tx(`${fint(S.ctx)} is the number of tokens a step without the cache computes: all of the context again, in every layer.`,`${fint(S.ctx)} 是不用 cache 时一步要算的 token 数：整段上下文都重算，每一层都算。`),
 pairs1:()=>tx(`With the cache only the newest token’s query is computed, and it meets the keys of all ${fint(S.ctx)} tokens: ${fint(S.ctx)} query–key pairs.`,`用 cache 时只算最新 token 的 query，它和全部 ${fint(S.ctx)} 个 token 的 key 相遇：${fint(S.ctx)} 对 query 和 key。`),
 pairsall:()=>{const C=fint(S.ctx), C1=fint(S.ctx+1);
   return tx(`Without the cache every token’s query is computed again, and token i meets the keys of tokens 1 to i: 1 + 2 + … + ${C} = ${C} × ${C1} ÷ 2 query–key pairs. At 4 × 32 × 4,096 FLOPs a pair, that is 4 × 32 × 4,096 × ${C} × ${C1} ÷ 2 = 2 × 32 × 4,096 × ${C} × ${C1}.`,
             `不用 cache 时，每个 token 的 query 都要重算，第 i 个 token 和第 1 到第 i 个 token 的 key 相遇：一共 1 + 2 + … + ${C} = ${C} × ${C1} ÷ 2 对 query 和 key。每对 4 × 32 × 4,096 FLOP，合起来是 4 × 32 × 4,096 × ${C} × ${C1} ÷ 2 = 2 × 32 × 4,096 × ${C} × ${C1}。`);},
 kvread:()=>tx(`With the cache the step also reads the keys and values of the ${fint(S.ctx-1)} earlier tokens from the cache, 131,072 bytes (128 KiB) each.`,`用 cache 时，这一步还要从 cache 里读之前 ${fint(S.ctx-1)} 个 token 的 key 和 value，每个 token 131,072 字节（128 KiB）。`),
 kvnone:{en:'Without the cache nothing is kept, so the step reads nothing from a cache. The keys and values it computes are used within the step and dropped, like the other activations, whose memory traffic this lab leaves out.',zh:'不用 cache 时什么都不存，所以这一步不从 cache 读任何东西。它算出的 key 和 value 在这一步里用完就丢，和其他激活值一样；激活值读写显存的流量，这个 Lab 不计。'},
 effc:()=>tx(`${fixd(H100.peak*EFF.mfu/1e12,2)} TFLOP/s is 50% of the H100 SXM’s dense BF16 peak, 989.5 TFLOP/s (its datasheet’s 1,979 TFLOPS with sparsity, halved). The 50% is this lab’s assumption.`,`${fixd(H100.peak*EFF.mfu/1e12,2)} TFLOP/s 是 H100 SXM 稠密 BF16 峰值 989.5 TFLOP/s（规格表上 with sparsity 的 1,979 TFLOPS 减半）的 50%。50% 是本 Lab 的假设。`),
 effm:()=>tx(`${fixd(H100.bw*EFF.mbu/1e12,2)} TB/s is 60% of the H100 SXM’s peak memory bandwidth, 3.35 TB/s. The 60% is this lab’s assumption.`,`${fixd(H100.bw*EFF.mbu/1e12,2)} TB/s 是 H100 SXM 峰值显存带宽 3.35 TB/s 的 60%。60% 是本 Lab 的假设。`),
 /* the small model's shape and its counts */
 skv2:{en:'The 2 counts the two vectors a token keeps for each head in every layer: a key and a value.',zh:'2 是一个 token 在每一层、每个 head 上存的两个向量：一个 key 和一个 value。'},
 slayers:{en:'The small model has 2 layers.',zh:'小模型有 2 层。'},
 sheads:{en:'Each layer of the small model has 2 heads, each with its own key and value.',zh:'小模型每层有 2 个 head，各有自己的 key 和 value。'},
 shd:{en:'Each head’s key and value have 4 numbers (head_dim): the hidden size d = 8 split between 2 heads.',zh:'每个 head 的 key 和 value 都是 4 维（head_dim）：隐藏维度 d = 8 分给 2 个 head。'},
 sumn:()=>{const n=S.n; return tx(`Without the cache, step k runs all k tokens so far through the model: 1 token-pass in step 1, 2 in step 2${n>2?`, … ${n} in step ${n}`:''}.`,`不用 cache 时，第 k 步要把目前的 k 个 token 全部过一遍模型：第 1 步 1 次，第 2 步 2 次${n>2?`，……第 ${n} 步 ${n} 次`:''}。`);},
 gauss:()=>{const n=S.n; return tx(`The sum of the whole numbers from 1 to n has a short form, below. Here n is ${n}: ${n} × ${n+1} ÷ 2 = ${fint(tri(n))}.`,`从 1 加到 n 的和有一个简单的算法，见下面的公式。这里 n 是 ${n}：${n} × ${n+1} ÷ 2 = ${fint(tri(n))}。`);},
 pass1:()=>tx(`With the cache each step runs only its newest token: one token-pass per step, ${fint(S.n)} in ${stepN(S.n)}.`,`用 cache 时，每一步只算最新的一个 token：每步 1 次 token 前向，${stepN(S.n)}共 ${fint(S.n)} 次。`),
 reads:()=>{const n=S.n; return tx(`With the cache, step k reads the keys and values of its k − 1 earlier tokens from the cache, in each layer: 0 in step 1${n>1?`, 1 in step 2`:''}${n>2?`, … ${n-1} in step ${n}`:''}.`,`用 cache 时，第 k 步在每一层从 cache 读之前 k − 1 个 token 的 key 和 value：第 1 步 0 次${n>1?'，第 2 步 1 次':''}${n>2?`，……第 ${n} 步 ${n-1} 次`:''}。`);}
};
const termText=k=>{const tm=TERM[k]; return !tm?'':(typeof tm==='function'?tm():L(tm));};
/* H1 (the owner, 2026-10-09): a term whose hover needs a formula shows it typeset, under the words */
const TERMFX={gauss:()=>A.ui.math('1 + 2 + \\ldots + n = \\frac{n(n + 1)}{2}')};
const termTipHTML=k=>TERM[k]?`<div>${esc(termText(k))}</div>${TERMFX[k]?`<div class="fx">${TERMFX[k]()}</div>`:''}<div class="act">${jumpAct(k)}</div>`:'';
/* Llama-3.1-8B's and the H100's constants, the small model's shape, and terms that describe how a step works: a click shows the tip */
const FIXED=new Set(['model','gpu','wl','wh','wr','kv2','layers','kvheads','hdim','bytes2','attnpair']);
const SFIXED=new Set(['skv2','slayers','sheads','shd']);
const HOWSTEP=new Set(['tok1','kvnone']);
/* derived terms jump to the setting they follow */
const DER={tokall:'ctx',pairs1:'ctx',pairsall:'ctx',kvread:'ctx',effc:'eff',effm:'eff',sumn:'n',gauss:'n',pass1:'n',reads:'n'};
const locked=k=>(mode==='challenge'&&(k==='n'||k==='cache'||k==='mask'))||(k==='n'&&preLock());
function ctlValue(k){return {n:tx(`${S.n} of ${NT}`,`${S.n} / ${NT}`),cache:S.cache?tx('On','用'):tx('Off','不用'),mask:S.mask?tx('On','开'):tx('Off','关'),ctx:ctxTok(S.ctx)}[k];}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='mathreal') return S.view==='real'?tx('Click to open “Show the math”.','点击展开“展开公式”。'):tx('Click to switch the chart to Llama-3.1-8B and open “Show the math”.','点击把图切到 Llama-3.1-8B，并展开“展开公式”。');
  if(k==='chk') return tx('Click to scroll to the “change” rows in the chart.','点击滚到图里的“变化”行。');
  if(k==='eff') return tx('Click to jump to “Step time” under Settings.','点击跳到“自由调参”里的“每一步的耗时”。');
  if(FIXED.has(k)) return tx('Fixed in this lab: it always uses Llama-3.1-8B on one H100 SXM.','这个 Lab 里固定不变：始终是一张 H100 SXM 跑 Llama-3.1-8B。');
  if(SFIXED.has(k)) return tx('Fixed in this lab: the small model always has this shape.','这个 Lab 里固定不变：小模型始终是这个形状。');
  if(k==='tok1') return tx('Fixed: this is what a step with the cache computes.','固定不变：用 cache 的一步就是这样算的。');
  if(k==='kvnone') return tx('Fixed: this is what a step without the cache reads.','固定不变：不用 cache 的一步就是这样读的。');
  if(DER[k]) return jumpAct(DER[k]);
  if(!CTLN[k]) return '';
  if(k==='n'&&preLock()) return tx('The small model starts writing after you answer part 2.','在第 2 部分作答后，小模型才开始生成。');
  if(locked(k)) return tx(`Fixed during the challenge: the challenge is about Llama-3.1-8B, so the small model’s “${L(CTLN[k])}” is not used.`,`挑战中固定不变：挑战只看 Llama-3.1-8B，用不到小模型的“${L(CTLN[k])}”。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({n:$('#kvb-ic-gen',app)||$('#kvb-ctl-n',app),cache:$('#kvb-ic-cache',app)||$('#kvb-ctl-cache',app),mask:$('#kvb-ic-mask',app)||$('#kvb-ctl-mask',app),
  ctx:$('#kvb-ic-ctx',app)||$('#kvb-ic-chctx',app)||$('#kvb-ctl-ctx',app)}[k]);
function jump(k,src){
  if(!app) return;
  /* the Fidelity popover's link is about the Llama-3.1-8B times, so it shows that math (r12 N9) */
  if(k==='mathreal'&&S.view!=='real') setState({view:'real'});
  if(k==='math'||k==='mathreal'){const d=$('#kvb-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='chk'){if(S.view!=='toy') setState({view:'toy'}); const rows=$$('#kvb-grid .kvb-chk',app).filter(r=>!r.hidden), el=rows[0]||$('#kvb-grid',app); A.tip.hide();
    if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); (rows.length?rows:[el]).forEach(x=>A.ui.flash(x));} return;}
  if(k==='controls'){const c=$('#kvb-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='eff'){const el=$('#kvb-ctl-eff',app); if(el) A.ui.jumpToEl(el); return;}
  if(FIXED.has(k)||SFIXED.has(k)||HOWSTEP.has(k)){if(src) A.tip.show(src,termTipHTML(k)); return;}
  if(DER[k]){jump(DER[k],src); return;}
  if(!CTLN[k]) return;
  if(locked(k)){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  if((k==='n'||k==='cache'||k==='mask')&&S.view!=='toy') setState({view:'toy'});
  if(k==='ctx'&&S.view!=='real') setState({view:'real'});
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='gen'){if(S.n>=NT) return; wantObs=true; byLearner(()=>setState({view:'toy',n:S.n+1})); revealSoon(); wantObs=false; return;}
  if(a==='restart'){byLearner(()=>setState({view:'toy',n:0})); return;}
  if(a==='reset'){bySettings(btn,()=>{if(mode==='challenge') setState({ctx:CH_START}); else {Object.assign(S,DEFAULT); if(preLock()) S.n=0; render();}}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(S.ctx).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#kvb-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* R3.2 (r15 N2): a change under Settings never moves the page. Nothing scrolls on purpose, but on a phone the stage sits above
   Settings, and a Settings control can switch the chart (the cache and the mask show the small model, the context Llama-3.1-8B);
   the stage then changes height and the browser's scroll anchoring moved the control by a line (49 px at 390 px wide). So anchoring
   is paused for the update and the control just used is put back where it was, as #gqa-mla does. */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#kvb-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return;
    if(k==='n') bySettings(e.target,()=>setState({view:'toy',n:+e.target.value}));
    else if(k==='ctx') bySettings(e.target,()=>setState({view:'real',ctx:CTX[+e.target.value]}));
    else if(k==='chctx') byLearner(()=>setState({ctx:CH_STOPS[+e.target.value]}));});
  /* a slider in the card brings its result into view; a change under Settings never scrolls the page (R3.2, CONTRACT §11) */
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if((k==='n'||k==='ctx'||k==='chctx')&&e.target.closest('#kvb-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    /* a click in the rail can change the card's height (a new part, a shorter result): keep the sticky stage below the top bar (r12 N4) */
    if(!d.jump&&d.ex!=='drill'&&b.closest('#kvb-rail')) keepStage();   // not after the capability card's jump to the drill, below the bench
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!b.closest('#kvb-rail');
      if(k==='view'){setState({view:d.v}); return;}
      wantObs=inRail;
      if(k==='cache') bySettings(b,()=>setState({view:'toy',cache:d.v==='1'}));
      else if(k==='mask') bySettings(b,()=>setState({view:'toy',mask:d.v==='1'}));
      if(inRail) revealSoon(); wantObs=false; return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#kvb-rail [data-pred="${pred}"]`,app); if(fb) fb.focus(); return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({view:'real',ctx:p.P})); requestAnimationFrame(()=>{if(app) revealBox($('#kvb-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#kvb-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}

/* =====================================================================
   registration
   ===================================================================== */
const sOf=s=>({view:s.view,n:s.n,cache:s.cache,mask:s.mask,ctx:s.ctx});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:sOf(S),pred,observed:{...observed},saved:saved?sOf(saved):null,chSubmitted,chReveal,maskBack,sbTouched,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&chPoint(S.ctx).ok}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=p; if(mode!=='guided') S.view='real'; pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sv:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    maskBack=o.maskBack===true&&mode==='guided'&&STEPS[step].id==='keep'; sbTouched=o.sbTouched===true;
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...DEFAULT};
    el.innerHTML=viewHTML(); app=$('#kvb-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#kvb-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    const rail=$('#kvb-rail',app); rail.innerHTML=NB(railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):''))+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):''); A.ui.initRanges(rail);   // a language switch re-mounts: keep the exits (Batch 2)
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#kvb-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#kvb-math',app).open=true;}},
  onResize(){if(app&&S.view==='real') drawReal();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(S.view==='toy'?$('#kvb-grid',app):$('#kvb-sp',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#kvb-graph',app),
    render({status,vis,read}){if(!app) return; const mem=x=>`<span class="dk-mem">${x}</span>`;
      if(S.view==='toy'){const s=toyStatus(), c=counts(S.n,S.cache), ck=check(S.n,S.mask);
        /* the bar is one line on a phone: the stage's status in a few words */
        status.className='status '+s.cls; status.textContent=!S.cache?tx('No KV cache','不用 KV cache'):(s.cls==='neutral'?tx('KV cache on','用 KV cache'):(s.cls==='ok'?tx('Matches a recompute','和重算的一致'):tx('Cache is stale','cache 过期了')));
        let g='<div class="kvb-dkg">'; for(let s2=1;s2<=NT;s2++){for(let j=0;j<NT;j++){const done=s2<=S.n, comp=done&&j<s2&&(!S.cache||j===s2-1), rd=done&&S.cache&&j<s2-1; g+=`<i class="${comp?'kvb-comp':(rd?'kvb-read':'')}"></i>`;}} vis.innerHTML=g+'</div>';
        const chkMain=showCheck()&&(!S.mask||(mode==='guided'&&STEPS[step].id==='check'));
        read.innerHTML=chkMain?esc(S.mask?tx(`Causal mask on: at step ${S.n}, earlier keys and values differ by 0 in both layers`,`因果 mask 开：第 ${S.n} 步时，之前的 key 和 value 两层都差 0`)
                                        :tx(`Causal mask off: at step ${S.n}, layer 2 differs by up to ${diffS(ck.max[1])}, layer 1 by ${diffS(ck.max[0])}`,`因果 mask 关：第 ${S.n} 步时，第 2 层最多差 ${diffS(ck.max[1])}，第 1 层差 ${diffS(ck.max[0])}`))
          :esc(S.n===0?tx('No decode step done yet','还没开始生成'):tx(`Step ${S.n} of ${NT} ${S.cache?'with':'without'} the KV cache: ${passN(c.passes)} so far`,`第 ${S.n} / ${NT} 步，${S.cache?'用':'不用'} KV cache：目前 ${passN(c.passes)}`));
        return;}
      const r=real(S.ctx), s=realStatus();
      vis.innerHTML=`<div class="kvb-dks"><i style="width:${Math.min(100,tpsV(r.w.t)/150*100).toFixed(1)}%"></i><i style="width:${Math.min(100,tpsV(r.wo.t)/150*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const p=chPoint(S.ctx); status.className='status '+s.cls; status.textContent=p.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到');
        read.innerHTML=esc(tx(`Without the KV cache, at ${ctxTok(S.ctx)} of context: an estimated ${tpsS(p.t)} tokens per second`,`不用 KV cache、上下文 ${ctxTok(S.ctx)} 时，估计每秒 ${tpsS(p.t)} 个 token`))+mem(esc(tx(' (the goal is 20)','（目标 20 个）'))); return;}
      status.className='status neutral'; status.textContent=ctxTok(S.ctx);    // two words, as the Context slider reads; the sentence carries the rest
      read.innerHTML=esc(tx(`Estimated at ${ctxTok(S.ctx)} of context: ${tS(r.w.t)} per token with the KV cache, ${tS(r.wo.t)} without`,`上下文 ${ctxTok(S.ctx)} 时，估计每个 token 用 KV cache 要 ${tS(r.w.t)}，不用要 ${tS(r.wo.t)}`));}
  }
});

/* =====================================================================
   home tile: one slider, two small grids, one sentence (R11.5); the counts are a formula
   ===================================================================== */
function tzTri(n,cache){
  // a staircase triangle of n rows in a 100 × 100 box: row s covers columns 0..s-1; with the cache only the diagonal is computed.
  // Up to TZ_CELLS rows every cell is its own square with a gap, so the squares stay visible; beyond that the cells merge into one shape
  const c=100/n; let comp='', read='';
  if(n<=TZ_CELLS){const g=Math.min(1.6,c*0.14), w=(c-g).toFixed(2), sq=(i,j)=>`M${(j*c+g/2).toFixed(2)},${(i*c+g/2).toFixed(2)}h${w}v${w}h-${w}Z`;
    for(let i=0;i<n;i++) for(let j=0;j<=i;j++){if(!cache||j===i) comp+=sq(i,j); else read+=sq(i,j);}
    return {comp,read};}
  if(!cache){let d=`M0,0`; for(let s=1;s<=n;s++){d+=`H${(s*c).toFixed(2)}V${(s*c).toFixed(2)}`;} d+=`H0Z`; comp=d;}
  else {let d=''; for(let s=1;s<=n;s++){d+=`M${((s-1)*c).toFixed(2)},${((s-1)*c).toFixed(2)}h${c.toFixed(2)}v${c.toFixed(2)}h${(-c).toFixed(2)}Z`;} comp=d;
    if(n>1){let r=`M0,${c.toFixed(2)}`; for(let s=2;s<=n;s++){r+=`H${((s-1)*c).toFixed(2)}V${(s*c).toFixed(2)}`;} r+=`H0Z`; read=r;}}
  return {comp,read};
}
const TZ_CELLS=16;   // the tile opens on 6 tokens, where every square shows (product owner, 2026-10-02: at 64 the squares merged into a gray band)
A.registerTeaser(ID,{fid:'formula',render(el){
  const lab=tx('Tokens written','生成的 token 数');
  el.innerHTML=`<div class="tz kvb-tz"><label class="tz-lab"><span>${lab}</span><output class="mono"></output></label>`+
    `<input type="range" min="1" max="64" step="1" value="${tzN}" aria-label="${esc(lab)}">`+
    `<div class="kvb-tzv" aria-hidden="true">${['off','on'].map(k=>`<figure class="kvb-tzf" data-k="${k}"><figcaption>${k==='off'?tx('No KV cache','不用 KV cache'):tx('KV cache','用 KV cache')}</figcaption><svg viewBox="-1 -1 102 102" class="kvb-tzs"><rect x="0" y="0" width="100" height="100" class="kvb-tzbox"/><path class="kvb-tzr"/><path class="kvb-tzc"/></svg><span class="kvb-tzn mono"></span></figure>`).join('')}</div>`+
    `<div class="tz-leg"><span data-tipx="${esc(tx('Each row is a decode step and each column a token; a filled cell is a token run through all the layers in that step.','每行是一个 decode 步，每列是一个 token；实心格是这一步里过了一遍所有层的 token。'))}"><i class="sw kvb-sw-comp"></i>${tx('token run through the layers','过一遍模型的 token')}</span><span data-tipx="${esc(tx('An amber outline: the step read that token’s keys and values from the KV cache instead of computing them again.','琥珀色框：这一步从 KV cache 里读出这个 token 的 key 和 value，没有重算。'))}"><i class="sw kvb-sw-read"></i>${tx('keys and values read from the KV cache','从 KV cache 读出的 key 和 value')}</span></div>`+
    `<p class="tz-read"></p>`+
    `<p class="tz-note">${tx('Without a cache, decode step n runs all n tokens so far through the model again. A token-pass is one token through all the layers. So the count grows with the square of the length: 100 tokens take 5,050 token-passes, and 1,000 tokens take 500,500.','不用 cache 时，第 n 个 decode 步要把目前的 n 个 token 重新过一遍模型。一个 token 走完所有层，算一次 token 前向。所以次数随长度的平方增长：100 个 token 要 5,050 次，1,000 个 token 要 500,500 次。')}</p>`+
    `<p class="tz-note">${tx('With a KV cache, each step runs only the newest token and reads the earlier tokens’ keys and values from the cache. The counts are a formula, for a prompt of one start token.','用 KV cache 时，每一步只算最新的一个 token，之前 token 的 key 和 value 从 cache 里读。这些次数是按公式算的，prompt 只有一个起始 token。')}</p>`+
    `<p class="tz-note">${tx('Keeping them costs memory: 128 KiB per token for Llama-3.1-8B in BF16 (2 × 32 layers × 8 KV heads × 128 numbers × 2 bytes).','存下来要占显存：Llama-3.1-8B 在 BF16 下每个 token 128 KiB（2 × 32 层 × 8 个 KV head × 128 维 × 2 字节）。')}</p></div>`;
  const inp=el.querySelector('input'), out=el.querySelector('output'), rd=el.querySelector('.tz-read');
  const upd=()=>{tzN=+inp.value; A.ui.setRange(inp); const n=tzN, off=tri(n);
    out.textContent=tokN(n); inp.setAttribute('aria-valuetext',out.textContent);
    el.querySelectorAll('.kvb-tzf').forEach(f=>{const on=f.dataset.k==='on', p=tzTri(n,on); f.querySelector('.kvb-tzc').setAttribute('d',p.comp); f.querySelector('.kvb-tzr').setAttribute('d',p.read||'M0,0');
      f.querySelector('svg').classList.toggle('kvb-tzcells',n<=TZ_CELLS);
      f.querySelector('.kvb-tzn').textContent=tx(passN(on?n:off),`${fint(on?n:off)} 次前向`);});
    rd.innerHTML=tx(`Writing ${tokN(n)} takes <b>${fint(off)}</b> ${off===1?'token-pass':'token-passes'} (a token through all layers) without a KV cache, and <b>${fint(n)}</b> with one.`,
                    `生成 ${fint(n)} 个 token：不用 KV cache 要做 <b>${fint(off)}</b> 次 token 前向（一个 token 走完所有层），用了只要 <b>${fint(n)}</b> 次。`);};
  inp.addEventListener('input',upd); upd();
}});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__kvb=Object.freeze({
  weights:()=>deep(W), words:()=>deep(WORDS),
  forward:(s,mask)=>deep(forward(s,mask)), check:(n,mask)=>deep(check(n,mask)), counts:(n,cache)=>{const c=counts(n,cache); return {n:c.n,cache:c.cache,passes:c.passes,reads:c.reads,nums:c.nums,p100:c.p100};},
  real:C=>deep(real(C)), nocache:C=>deep(nocache(C)), cached:C=>deep(cached(C)), prefill:P=>deep(prefill(P)), scen:(P,Aa)=>deep(scen(P,Aa)),
  fmt:Object.freeze({t:tS,tps:tpsS,tpsV,tot:totS,mem:memS,ratio:ratioS,diff:diffS}),
  drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({NT,D,NH,HD,NL,FF,SEED,SCALE:Object.freeze({...SCALE}),PER_TOK,H100:Object.freeze({...H100}),EFF:Object.freeze({...EFF}),LL:Object.freeze({...LL}),
    P_EMB,P_HEAD,P_LAYER,P_LAYERS,P_TOTAL,KV_TOK,W_READ,W_MEM,CTX:CTX.slice(),CH_STOPS:CH_STOPS.slice(),CH_START,CH_GOAL,PRESETS:deep(PRESETS.map(p=>({id:p.id,P:p.P,A:p.A}))),STEP_IDS:STEPS.map(s=>s.id)}),
  state:()=>({...S,mode,step,pred,observed:{...observed},chSubmitted,chReveal,tzN,stageBusy:stageT!==0})
});
})();
