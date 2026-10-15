/* chunked-prefill (the "SDE → LLM serving" path, stop 6 of 7): Why does one huge prompt slow everyone down?
   Fidelity: Toy run. The lab runs teasers/set-a.js's step model step by step, with the same constants and step formula as the home tile
   and labs/prefill-decode.js: Llama-3.1-8B in BF16 on one H100 SXM; a step takes the longer of its arithmetic (at 50% of the dense BF16
   peak) and its memory traffic (at 60% of the peak bandwidth).
   - U users decode at context c, held fixed while the prompt is prefilled (as the tile does); an N-token prompt arrives as a step ends.
   - Chunk C (0 = none): step k prefills prompt tokens a..b-1 next to the U decode tokens:
     step({tok:U+(b-a), logits:U+(b===N?1:0), attn:U*attn(c,c+1)+attn(a,b), kv:U*c+b}). The prompt's first token (TTFT) is the end of its
     last chunk's step. A step without prefill: step({tok:U, logits:U, attn:U*attn(c,c+1), kv:U*c}). After its first token the prompt
     decodes with the users: step({tok:U+1, logits:U+1, attn:U*attn(c,c+1)+attn(N,N+1), kv:U*c+N}).
   - The users' longest wait is the longest step while the prompt is prefilled (every user gets one token when each step ends); their
     time per output token during the prefill is TTFT ÷ steps. TTFT = the prompt's own arithmetic + steps × the users' arithmetic + what
     memory traffic adds in the steps it sets (exact, since the chunks' arithmetic adds up to the whole prompt's).
   Words (EN / ZH): step / 步 (an engine step, never a guided card: those are part / 部分); chunk; prefill; decode step / decode 步;
   arithmetic / 计算; memory traffic / 读写显存 (the bar label is memory / 读写显存, as in prefill-decode); time to first token (TTFT) /
   首 token 延迟（TTFT）.
   window.__ck is a read-only hook for tests/chunked-prefill_numbers.py and tests/chunked-prefill_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='chunked-prefill';
/* Chinese keeps a space between Chinese characters and Latin letters or digits (“16 个用户”), as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const MODEL='Llama‑3.1‑8B', GPUN='H100 SXM';      // no-break hyphens and space: a model or GPU name never breaks

/* =====================================================================
   engine: teasers/set-a.js's step model, the same constants and formulas
   ===================================================================== */
const H100={name:GPUN,peak:1979e12/2,bw:3.35e12,mem:80};       // dense BF16 = the datasheet's 1,979 TFLOPS "with sparsity" ÷ 2; 3.35 TB/s; 80 GB
const EFF={mfu:0.5,mbu:0.6};                                    // assumed: 50% of peak FLOP/s, 60% of peak bandwidth (as set-a)
const LL={name:'Llama-3.1-8B',layers:32,hidden:4096,heads:32,kvHeads:8,headDim:128,vocab:128256,ffn:14336,maxCtx:131072};
const P_EMB=LL.vocab*LL.hidden, P_HEAD=LL.vocab*LL.hidden;
const P_LAYER=2*LL.hidden*LL.hidden+2*LL.hidden*LL.kvHeads*LL.headDim+3*LL.hidden*LL.ffn+2*LL.hidden;
const P_LAYERS=LL.layers*P_LAYER+LL.hidden, P_TOTAL=P_LAYERS+P_EMB+P_HEAD;
const KV_TOK=2*LL.layers*LL.kvHeads*LL.headDim*2;              // K and V, BF16: 131,072 bytes = 128 KiB per token
const W_READ=2*(P_LAYERS+P_HEAD);                               // bytes one step reads: every weight but the embedding table (a lookup)
const W_MEM=2*P_TOTAL;                                          // bytes the weights take in GPU memory
const attn=(a,b)=>2*LL.layers*LL.hidden*(b*(b+1)-a*(a+1));      // QKᵀ and AV FLOPs, all layers, for query positions a..b-1 (causal)
const FS=H100.peak*EFF.mfu, BS=H100.bw*EFF.mbu;                 // the FLOP/s and bytes/s a step reaches
/* one step: U users decode at context c, and prompt tokens a..b-1 of an N-token prompt are prefilled next to them (none when a===b) */
function stepOf(U,c,a,b,N){
  const chunk=b-a, last=chunk>0&&b===N;
  const fD=2*P_LAYERS*U+2*P_HEAD*U+U*attn(c,c+1);             // the users' tokens: layers, LM head, attention over their contexts
  const fM=2*P_LAYERS*chunk+(last?2*P_HEAD:0);                 // the chunk's tokens through the layers (the prompt's LM head in its last chunk)
  const fA=chunk>0?attn(a,b):0;                                 // the chunk's attention to itself and every earlier prompt token
  const fl=fD+fM+fA, by=W_READ+KV_TOK*(U*c+b), tc=fl/FS, tm=by/BS;
  return {t:Math.max(tc,tm),tc,tm,fl,by,fD,fM,fA,tcD:fD/FS,tcM:fM/FS,tcA:fA/FS,tw:W_READ/BS,tkv:KV_TOK*(U*c+b)/BS,cb:tc>=tm,a,b,chunk,last};
}
/* after its first token, the prompt decodes with the users: one more request at context N */
function afterOf(U,c,N){
  const fl=2*P_LAYERS*(U+1)+2*P_HEAD*(U+1)+U*attn(c,c+1)+attn(N,N+1), by=W_READ+KV_TOK*(U*c+N), tc=fl/FS, tm=by/BS;
  return {t:Math.max(tc,tm),tc,tm,fl,by};
}
const RC=new Map();
function run(s){
  const U=s.U, c=s.ctx, N=s.N, C=s.C||0, key=[U,c,N,C].join('_'); if(RC.has(key)) return RC.get(key);
  const chunk=C>0&&C<N?C:N, n=Math.ceil(N/chunk), normal=stepOf(U,c,0,0,N), steps=[]; let t=0;
  for(let k=0;k<n;k++){const a=k*chunk, b=Math.min(N,a+chunk), st=stepOf(U,c,a,b,N); st.k=k; st.start=t; st.end=t+st.t; steps.push(st); t+=st.t;}
  let li=0; steps.forEach((x,i)=>{if(x.t>steps[li].t) li=i;});
  const fP=2*P_LAYERS*N+2*P_HEAD+attn(0,N), tcP=fP/FS;         // the prompt's own arithmetic: the same however it is split
  const extra=steps.reduce((z,x)=>z+Math.max(0,x.tm-x.tc),0), nMem=steps.filter(x=>!x.cb).length;
  const r={U,c,N,C,chunk,n,normal,steps,after:afterOf(U,c,N),ttft:t,longest:steps[li].t,li,avg:t/n,fP,tcP,tcD:normal.tcD,extra,nMem,tcAttP:attn(0,N)/FS};
  RC.set(key,r); if(RC.size>500) RC.delete(RC.keys().next().value);
  return r;
}
/* the most users that fit: the KV cache memory lab's model (80 GB read as 80 GiB, 90% usable, 2 GiB of runtime overhead) */
const KVD=A.kv&&A.kv.DEFAULT?A.kv.DEFAULT:{util:0.9,overhead:2};
const UTIL=KVD.util, OH=KVD.overhead;
const KV_BUDGET=H100.mem*A.GiB*UTIL-W_MEM-OH*A.GiB;             // bytes left for every request's KV cache
const TOK_FIT=Math.floor(KV_BUDGET/KV_TOK+1e-9);                // tokens of KV cache that fit
const maxUsers=(c,N)=>Math.max(0,Math.floor((TOK_FIT-N)/c+1e-9));
const U_STOPS=[1,2,4,8,16,32,64], X_STOPS=[1024,2048,4096,8192,16384], N_STOPS=[2048,4096,8192,16384,32768,65536,122880];
/* chunk sizes; 0 = no chunking. They reach down to 32 tokens, where a step is little more than one read of the weights, so both ends
   show (product owner, 29 Sep: “too large gives one long step, too small reads the weights again and again”); above 1K only 2K is kept,
   which the prediction and the challenge need */
const K_OPTS=[0,2048,1024,512,256,128,64,32];
const BASE={U:16,ctx:4096,N:32768,C:0};
const uFit=(c,N)=>{const m=maxUsers(c,N); let b=U_STOPS[0]; U_STOPS.forEach(u=>{if(u<=m) b=u;}); return b;};
/* the time axis for a workload: the same for every chunk size, so a change of chunk size moves the marks, not the scale. It covers the
   runs from no chunking down to 256-token chunks; smaller chunks need so many steps that their first token lies past the right edge,
   and the chart says so there instead of shrinking every other run */
const AXIS_OPTS=K_OPTS.filter(C=>C===0||C>=256);
const SPAN=new Map();
function spanOf(s){
  const key=[s.U,s.ctx,s.N].join('_'); if(SPAN.has(key)) return SPAN.get(key);
  let m=0; AXIS_OPTS.forEach(C=>{m=Math.max(m,run({...s,C}).ttft);});
  const r0=run({...s,C:0}), x0=-Math.max(0.06*m,4*r0.normal.t), x1=m*1.06;
  const v={x0,x1,max:m}; SPAN.set(key,v); return v;
}
/* what of a run lies past the right edge: its first token (off), and its longest step (longOff) */
const edgeOf=r=>{const x1=spanOf({U:r.U,ctx:r.c,N:r.N}).x1; return {x1,off:r.ttft>x1+1e-12,longOff:r.steps[r.li].end>x1+1e-12};};
/* the largest chunk that still rides free: the smallest chunk whose step is set by arithmetic, early (the first chunk) and late (the last) */
const CROSS=new Map();
function crossOf(s){
  const key=[s.U,s.ctx,s.N].join('_'); if(CROSS.has(key)) return CROSS.get(key);
  const U=s.U, c=s.ctx, N=s.N; let first=N+1, last=N+1;
  for(let C=1;C<=N;C++){if(stepOf(U,c,0,C,N).cb){first=C; break;}}
  for(let C=1;C<=N;C++){if(stepOf(U,c,N-C,N,N).cb){last=C; break;}}
  const v={first,last}; CROSS.set(key,v); return v;
}
/* the challenge: 32 users at 2K, a 16K-token document; every user must get a token at least every 50 ms; choose only the chunk size.
   A setting the guided parts and the scenarios never show (R3.8): the best chunk is 1K here, not the guided parts' 512 */
const CH={U:32,ctx:2048,N:16384,goal:0.050,start:0};
const chPoint=C=>{const r=run({...CH,C}); return {C,wait:r.longest,ttft:r.ttft,n:r.n,ok:r.longest<=CH.goal,r};};
function chBest(){let b=null; K_OPTS.forEach(C=>{const p=chPoint(C); if(p.ok&&(!b||p.ttft<b.ttft)) b=p;}); return b;}

/* =====================================================================
   formatting (every derived number is computed from the value as shown, or the line gives the digits it needs, R6.6)
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const NS=' ';
/* a time: two decimals below 100 ms (11.74 ms), one below 1 s (127.2 ms), whole ms below 10 s (1,494 ms), then seconds (11.47 s) */
const fT=sec=>{const v=sec*1e3; if(v<99.995) return fixd(v,2)+NS+'ms'; if(v<999.95) return fixd(v,1)+NS+'ms'; if(v<9999.5) return fint(v)+NS+'ms'; return fixd(v/1e3,2)+NS+'s';};
const tMs=sec=>{const v=sec*1e3; if(v<99.995) return Math.round(v*100)/100; if(v<999.95) return Math.round(v*10)/10; if(v<9999.5) return Math.round(v); return Math.round(v/10)*10;};   // the shown value, in ms
const tDp=sec=>{const v=sec*1e3; return v<99.995?2:(v<999.95?1:(v<9999.5?0:-1));};   // decimals fT shows; −1: seconds
/* b − a from the two values as shown (R6.6), at the coarser precision of the two: 1,502 − 1,494 = 8 ms */
const fDiff=(a,b)=>{const da=tDp(a), db=tDp(b); if(da<0||db<0) return fixd((tMs(b)-tMs(a))/1e3,2)+NS+'s'; return fixd(tMs(b)-tMs(a),Math.min(da,db))+NS+'ms';};
const f1=sec=>fixd(sec*1e3,1)+NS+'ms';                          // one decimal, for a line that adds or divides
const f2=sec=>fixd(sec*1e3,2)+NS+'ms';
const f3=sec=>fixd(sec*1e3,3)+NS+'ms';
const fClock=sec=>{const v=sec*1e3; return Math.abs(v)<9999.5?fint(v)+NS+'ms':fixd(v/1e3,2)+NS+'s';};
const tokK=n=>n>=1024&&n%1024===0?(n/1024)+'K':fint(n);
const aTok=n=>(/^8/.test(tokK(n))?'an ':'a ')+tokK(n);      // “an 8K-token prompt”, “a 32K-token prompt”
const gbS=b=>{const v=b/1e9; return (v>=9.995?fixd(v,1):(v>=0.0995?fixd(v,2):fixd(v,3)))+NS+'GB';};
const tfS=v=>fixd(v/1e12,2), tbS=v=>fixd(v/1e12,2);
const ratioS=q=>q>=9.95?fint(q):fixd(q,1);
const pct0=x=>fint(x*100)+'%';
/* numbers in prose (R6.20): rounded for reading, about two significant digits (1.5 s, 130 ms, 33 ms, 7.6 ms); exact values stay on the
   stage, the metric cards and Show the math */
const fR=sec=>{const v=sec*1e3; if(v>=9950) return fint(Math.round(v/1e3))+NS+'s'; if(v>=950) return fixd(v/1e3,1)+NS+'s'; if(v>=99.5) return fint(Math.round(v/10)*10)+NS+'ms'; if(v>=9.95) return fint(Math.round(v))+NS+'ms'; return fixd(v,1)+NS+'ms';};
/* how many times as long, in words: over 100 times / 一百多倍 */
const ratioW=q=>q>=200?tx(`over ${fint(Math.floor(q/100)*100)} times`,`${fint(Math.floor(q/100)*100)} 多倍`):(q>=100?tx('over 100 times','一百多倍'):(q>=10?tx(`over ${fint(Math.floor(q/10)*10)} times`,`${fint(Math.floor(q/10)*10)} 多倍`):tx(`${fixd(q,1)} times`,`${fixd(q,1)} 倍`)));
/* how much later the prompt's first token comes than without chunking, in words */
const laterW=(r,r0)=>{const q=r.ttft/r0.ttft-1; return q<0.01?tx('barely later','几乎没有变晚'):(q<0.05?tx('only a little later','只晚了一点'):(q<0.2?tx('noticeably later','晚了一些'):tx('much later','明显变晚')));};
/* why it comes later at all (the exact split is in Show the math) */
const laterWhyW=r=>r.nMem?tx('every extra step computes the users’ tokens once more, and with small chunks many steps each take the full time to read the weights, however little they prefill','多出来的每一步都要再算一遍用户的 token；而且 chunk 小的时候，很多步不管 prefill 多少，都要花满读一遍权重的时间')
  :tx('every extra step computes the users’ tokens once more, a small extra cost','多出来的每一步都要再算一遍用户的 token，这点额外的工作很小');
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const userN=n=>tx(n===1?'1 user':`${fint(n)} users`,`${fint(n)} 个用户`);
const tokN=n=>tx(n===1?'1 token':`${fint(n)} tokens`,`${fint(n)} 个 token`);
const stepN=n=>tx(n===1?'1 step':`${fint(n)} steps`,`${fint(n)} 步`);
const ctxTxt=c=>tx(`${tokK(c)} tokens of context`,`${tokK(c)} 个 token 的上下文`);

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let saved=null, chSubmitted=false, chReveal=false, drillSeed=2, drillOpen=false, annT=null, userAct=false, wantObs=false, sbTouched=false;
let deepOpen=false, deepSeen=false, clampNote=false, speed='med', whyOpen=false;
const play={t:null,playing:false,raf:0,last:0,rate:0};          // t: the moment shown (seconds); null = the whole run
const SPEED={slow:48,med:24,fast:8};                            // seconds to play the whole timeline (time-based, R3.9)
const shownPrev={};
const R=()=>run(S);
const B0=()=>run({...BASE,C:0});                                // the mission: no chunking
const B2=()=>run({...BASE,C:2048});                             // the prediction's case: 2K-token chunks

/* =====================================================================
   definitions shown on hover (R2.8): a term is defined where a reader first meets it; prefill-decode's wording where it defines them
   ===================================================================== */
const TIP={
 prefill:()=>tx('Prefill is the first step of answering a request: the model takes in the whole prompt in one pass, writes every prompt token’s keys and values to the KV cache, and produces the first token of the answer.',
                'prefill 是回答一个请求的第一步：模型一次读入整个 prompt，把每个 prompt token 的 key 和 value 写进 KV cache，并产出回答的第一个 token。'),
 dstep:()=>tx('A decode step is one pass of the model that produces one new token for each request in it. It computes only the newest token, and reads the keys and values of the earlier tokens from the KV cache.',
              'decode 步是把模型跑一遍、给其中每个请求各多产出一个 token 的过程。它只算最新的那个 token，之前 token 的 key 和 value 从 KV cache 里读。'),
 step:()=>tx('A step is one pass of the model over everything the server scheduled together: here, one new token for each user who is decoding and, while a prompt is prefilled, all of it or one chunk of it. Every request in the step gets its output only when the whole step ends.',
             '一步是模型把服务端排在一起的所有工作跑一遍：这里是给每个正在 decode 的用户各算一个新 token，prompt 正在 prefill 时，再加上整个 prompt 或它的一个 chunk。这一步里的每个请求，都要等整步结束才拿到结果。'),
 chunk:()=>tx('Chunked prefill splits a long prompt into chunks of a set number of tokens and prefills one chunk per step, next to the decode tokens of the requests already running. The prompt’s first token comes when its last chunk is done.',
              'chunked prefill 把长 prompt 切成固定 token 数的 chunk，每一步 prefill 一个 chunk，和已经在跑的请求的 decode token 放在同一步里。prompt 的最后一个 chunk 算完，它的首 token 才出来。'),
 ttft:()=>tx('Time to first token (TTFT): how long a user waits from sending the prompt to seeing the first token of the answer. This lab has no queue, so it is the time of the prompt’s prefill steps.',
             '首 token 延迟（TTFT，time to first token）：从发出 prompt 到看到回答的第一个 token 要等多久。这个 Lab 里没有排队，所以它就是 prompt 的 prefill 各步的耗时。'),
 tpot:()=>tx('Time per output token (TPOT): the time between two tokens of an answer once it has started.',
             '每个输出 token 的耗时（TPOT，time per output token）：回答开始以后，两个 token 之间隔多久。'),
 overlap:()=>tx('A GPU reads memory and does arithmetic at the same time: while it computes with one chunk of the weights, it is already reading the next. So a step takes about as long as the slower of the two, not their sum. This is this lab’s estimate; real kernels overlap the two only in part.',
                'GPU 读写显存和做计算是同时进行的：它用已读到的一段权重计算时，已经在读下一段。所以一步的耗时约等于两者中较慢的那个，而不是两者相加。这是这个 Lab 的估计；真实 kernel 只能让两者部分重叠。'),
 wread:()=>tx(`In every step the GPU streams every parameter it multiplies (all but the embedding table: ${gbS(W_READ)}, for ${MODEL}) from GPU memory into its compute units once. That one read serves every token the step computes, so it costs the same for 1 token or 1,000.`,
              `每一步，GPU 都要把参与乘法的全部参数（embedding 表以外的全部参数，${MODEL} 是 ${gbS(W_READ)}）从显存读进计算单元一次。这一次读，供这一步算的所有 token 共用：算 1 个 token 和算 1,000 个 token 读的一样多。`),
 causal:()=>tx('In prefill each prompt token compares its query with the keys of itself and of every earlier prompt token (the causal mask), so a token deep in the prompt does more attention work than one near its start. A chunk near the end of a 32K-token prompt attends to about 32K earlier tokens.',
               'prefill 时，每个 prompt token 都拿它的 query 去和它自己以及前面每个 prompt token 的 key 比较（因果 mask），所以越靠后的 token，attention 的计算越多。32K 个 token 的 prompt 快结束时的一个 chunk，要对前面约 32K 个 token 做 attention。'),
 ctx:()=>tx('A user’s context is every token of the conversation so far. Each decode step reads their keys and values from the KV cache: 128 KiB per token for Llama-3.1-8B.',
            '用户的上下文是这段对话到目前为止的全部 token。每个 decode 步都要从 KV cache 里读它们的 key 和 value：Llama-3.1-8B 每个 token 128 KiB。'),
 flop:()=>tx('A FLOP is one floating-point operation; a multiply and an add count as 2. A GFLOP is 10^9 FLOPs and a TFLOP 10^12, so TFLOP/s measures how fast a GPU computes.',
             'FLOP 是一次浮点运算，一次乘法加一次加法算 2 个 FLOP。GFLOP 是 10^9 个 FLOP，TFLOP 是 10^12 个，所以 TFLOP/s 衡量 GPU 算得多快。'),
 gb:()=>tx('GB and TB here are 10^9 and 10^12 bytes, to match the bandwidth in TB/s. 15.0 GB is 14.0 GiB, the unit the KV cache labs use for memory.',
           '这里的 GB、TB 是 10^9、10^12 字节，和带宽的 TB/s 一致。15.0 GB 就是 14.0 GiB，KV cache 那两个 Lab 用 GiB 表示显存。'),
 kib:()=>kt('kib'), gib:()=>kt('gib'), kv:()=>kt('kvcache')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'a',t:{en:'The users’ longest wait drops a lot; the document’s first token barely moves',zh:'用户最长的等待大幅缩短，文档的首 token 几乎不变'}},
  {v:'b',t:{en:'Both get much better',zh:'两者都大幅改善'}},
  {v:'c',t:{en:'The users’ wait drops; the document’s first token takes about twice as long',zh:'用户的等待缩短，文档的首 token 慢到约 2 倍'}},
  {v:'d',t:{en:'Nothing changes much',zh:'都变化不大'}}];
const PRED_RIGHT='a';
const predT=v=>L(PRED.find(o=>o.v===v).t);
/* parts 1–3 are about the stall: the chunk size stays “none” until part 4, where the prediction's answer is shown (R3.8) */
const cLock=()=>mode==='guided'&&step<=2;
const kLabel=C=>C?tokK(C):tx('none','不切');
const cCtl=(id,inline,key)=>A.ui.seg({id,key:key||'C',label:tx('Chunk size (prompt tokens per step)','chunk 大小（每步 prefill 的 prompt token 数）'),options:K_OPTS.map(C=>[String(C),kLabel(C)]),value:String(S.C),inline});
const playLabel=()=>play.playing?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Replay','▶ 回放');
const STEPTIP=()=>tx('The chart moves to the end of the next step, where every user gets a token.','图上的时刻跳到下一步结束的地方，那时每个用户各拿到一个 token。');
const speedCtl=()=>`<div class="ck-speed" data-tipx="${esc(tx(`Slow plays the whole timeline in ${SPEED.slow} s, Medium in ${SPEED.med} s and Fast in ${SPEED.fast} s.`,`慢速用 ${SPEED.slow} s 放完整条时间线，中速用 ${SPEED.med} s，快速用 ${SPEED.fast} s。`))}"><span class="ck-speed-l">${tx('Speed','速度')}</span>`+
  `<div class="seg-ctl" role="group" aria-label="${esc(tx('Replay speed','回放速度'))}" data-key="speed">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${speed===v}">${x}</button>`).join('')}</div></div>`;
const playBtns=()=>`<button type="button" class="btn sm ck-pbtn" data-act="play">${playLabel()}</button>${speedCtl()}<button type="button" class="btn ghost sm" data-act="stepfwd" data-tipx="${esc(STEPTIP())}">${tx('Next step ▸','下一步 ▸')}</button>`;
const DEEP=()=>`<div class="ck-deepbtn"><span class="lbl">${tx('Go deeper (optional)','深入（可选）')}</span><button type="button" class="btn ghost sm" data-act="deep" aria-pressed="${deepOpen}">${tx('The free chunk','白搭车的 chunk')}${deepSeen?' <span class="ck-seen">✓</span>':''}</button></div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'tl',
  lead:()=>{const b=B0(); return tx(`16 users get a token every ${fT(b.normal.t)}, until a 32K-token document arrives.`,`16 个用户每 ${fT(b.normal.t)} 拿到一个 token，直到一份 32K 个 token 的文档到来。`);},
  enter:()=>({...BASE}),
  body:()=>{const b=B0();
   return tx(`<p>${MODEL} runs on one ${GPUN} in BF16. 16 users are chatting with it, each with 4K tokens of ${DEF('context',TIP.ctx())}: every ${DEF('decode step',TIP.dstep())} gives each of them one more token, a steady stream.</p><p>Then a user sends a 32K-token document to be summarized. Before the answer can start, the server must ${DEF('prefill',TIP.prefill())} it: run the whole document through the model. Without ${DEF('chunking',TIP.chunk())} that is one single ${DEF('step',TIP.step())}, and nobody gets a token until it ends: ${J('tl','the timeline')} shows everyone frozen for about ${fR(b.longest)}.</p><p>Why must they all wait, and how can the server keep their tokens coming?</p>`,
             `<p>一张 ${GPUN} 用 BF16 跑 ${MODEL}。16 个用户正在和它聊天，每人有 4K 个 token 的${DEF('上下文',TIP.ctx())}：每个 ${DEF('decode 步',TIP.dstep())}给他们各多写一个 token，源源不断。</p><p>这时有个用户发来一份 32K 个 token 的文档，要它写摘要。回答开始之前，服务端得先 ${DEF('prefill',TIP.prefill())}：把整份文档过一遍模型。不${DEF('切 chunk',TIP.chunk())} 时，这是完整的${DEF('一步',TIP.step())}，这一步结束之前谁也拿不到 token：${J('tl','时间线')}上，所有人卡住了约 ${fR(b.longest)}。</p><p>为什么所有人都得等？服务端怎样才能让他们的 token 不断？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'tl',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: prefill the document in 2K-token chunks. What changes?','先猜：文档切成 2K 个 token 的 chunk 来 prefill，会怎样？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Now the server splits the document into 2K-token chunks. Each step puts the 16 users’ decode tokens and one chunk together into a single pass of the model: every user gets a token, and the document moves on by one chunk. After 16 such steps the whole document is prefilled. What happens to the users’ longest wait between two tokens, and to the document’s first token?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 4 shows the result.</p>':''}`,
             `<p>现在服务端把文档切成 2K 个 token 的 chunk。每一步把 16 个用户的 decode token 和文档的一个 chunk 放在一起，让模型跑一遍：每个用户拿到一个 token，文档往前推进一个 chunk。这样 16 步之后，整份文档就 prefill 完了。用户两个 token 之间最长的等待会怎样？文档的首 token 呢？</p>${ch}${pred?'<p class="muted small">已记下。第 4 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'stall',title:{en:'The stall: one step for the whole prompt',zh:'卡住：整个 prompt 一步算完'},focus:'lim',always:true,label:{en:'Result',zh:'结果'},
  lead:()=>tx('Replay it: nobody gets a token until the prefill step ends.','回放一遍：prefill 这一步结束之前，谁也拿不到 token。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>The server runs the 16 users’ decode tokens and the document’s whole prefill in the same step, and ${DEF('a step takes about as long as the slower of its arithmetic and its memory traffic',TIP.overlap())}: ${J('lim','the chart')} draws both, for a usual step and for this one. Replay the run:</p>`,
              `<p>服务端把 16 个用户的 decode token 和文档的整个 prefill 放在同一步里跑，而${DEF('一步的耗时约等于计算和读写显存中较慢的那个',TIP.overlap())}：${J('lim','图里')}把平时的一步和这一步的这两项都画了出来。回放一遍：</p>`)+
    `<div class="inline-ctl ck-ic-play" id="ck-ic-play"><span class="lbl">${tx('Replay the stall','回放卡住的这一步')}</span><div class="ck-prow"><button type="button" class="btn sm ck-pbtn" data-act="play">${playLabel()}</button><button type="button" class="btn ghost sm" data-act="stepfwd" data-tipx="${esc(STEPTIP())}">${tx('Next step ▸','下一步 ▸')}</button></div>${speedCtl()}</div>`,
  obs:()=>obsStall()},
 {id:'chunk',title:{en:'Chunk it',zh:'切成 chunk'},focus:'tl',expect:()=>S.C!==0,showMe:{C:2048},label:{en:'Observed',zh:'观察到'},
  lead:()=>tx('Pick a chunk size: each step now prefills one chunk next to the users.','选一个 chunk 大小：每一步只在用户旁边 prefill 一个 chunk。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>With chunked prefill, each step computes the users’ decode tokens together with one chunk of the document, in a single pass of the model (the dashed box on the chart). Each step is short, and every user gets a token when it ends. Pick a chunk size:</p>`,
              `<p>用 chunked prefill 时，每一步把用户的 decode token 和文档的一个 chunk 放在一起，模型只跑一遍（图上的虚线框）。每一步都很短，每一步结束时，每个用户都拿到一个 token。选一个 chunk 大小：</p>`)+cCtl('ck-ic-C',true),
  obs:()=>obsChunk()},
 {id:'free',title:{en:'How small should a chunk be?',zh:'chunk 该切多小？'},focus:'free',always:true,expect:()=>tooSmall(R()),showMe:{C:32},label:{en:'Observed',zh:'观察到'},
  lead:()=>tx('Small chunks ride on spare arithmetic, until they get too small.','小 chunk 搭的是空闲算力的便车，但太小也不行。'),
  enter:()=>({...BASE,C:2048}),
  body:()=>tx(`<p>A usual step mostly waits for memory: it ${DEF('reads all the weights',TIP.wread())} and the users’ KV caches, while its arithmetic is tiny (compare the two bars). A chunk adds arithmetic; as long as the step’s arithmetic stays shorter than its reads, the step takes no longer, and the chunk rides free.</p><p>Try 256 tokens first, then make the chunk even smaller, down to 32 tokens:</p>`,
              `<p>平时的一步主要在等读显存：它${DEF('把权重读一遍',TIP.wread())}，再读用户们的 KV cache，而计算很少（比较图里的两根条）。chunk 加进来的是计算：只要这一步的计算还没超过读取的时间，这一步就不会变长，chunk 等于白搭车。</p><p>先试 256 个 token，再把 chunk 切得更小，一直切到 32 个 token：</p>`)+cCtl('ck-ic-C',true)+DEEP(),
  obs:()=>obsFree()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:'',
  lead:()=>tx('The stall, the fix, and how small a chunk should be.','卡住的原因、解决办法，以及 chunk 该切多小。'),
  enter:()=>({...BASE,C:2048}),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);

/* part 3: the stall, from the mission's run. Explanations say the trend and the reason in words; the numbers are on the stage (R6.20) */
function obsStall(){
  const r=run({...BASE,C:0});
  return tx(`<p><b>One step of about ${fR(r.longest)}, ${ratioW(r.longest/r.normal.t)} a usual step: all that time, none of the 16 users gets a token.</b></p><p>Why: every token of the document is multiplied by all the weights and ${DEF('attends to every earlier one',TIP.causal())}, so the step’s arithmetic is huge, far longer than its memory reads (compare the two bars). The users’ own tokens cost almost nothing, but they are in the same step and come out only when it ends.</p>`,
            `<p><b>这一步要约 ${fR(r.longest)}，是平时一步的${ratioW(r.longest/r.normal.t)}：这段时间里，16 个用户谁也拿不到 token。</b></p><p>原因：文档的每个 token 都要和全部权重相乘，还要${DEF('对它前面的每个 token 做 attention',TIP.causal())}，这一步的计算量巨大，远远超过读写显存的时间（比较图里的两根条）。用户自己的 token 几乎不花时间，可它们和文档在同一步里，只能等这一步整个结束。</p>`);
}
/* the verdict on the prediction, which is about 2K-token chunks; the numbers are on the stage and the metric cards */
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right</span>: that is what 2K-token chunks do.</p>`,`<p><span class="okc">你的预测“${pt}”是对的</span>：切成 2K 个 token 的 chunk 时正是这样。</p>`)
    :tx(`<p>You predicted “${pt}”. With 2K-token chunks, the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。切成 2K 个 token 的 chunk 时，正确答案是“${rt}”。</p>`);
}
/* part 4: the learner's own chunk size, then the verdict, then why (R3.10: result, verdict, reason) */
function obsChunk(){
  const r=R(), r0=run({...S,C:0});
  if(r.C===0) return tx(`<p><b>No chunking: between two tokens the users wait for the whole prefill, about ${fR(r.longest)}.</b> Pick a chunk size to check your prediction.</p>`,`<p><b>不切 chunk：用户两个 token 之间要等整个 prefill，约 ${fR(r.longest)}。</b>选一个 chunk 大小，核对你的预测。</p>`);
  if(r.n===1) return tx(`<p><b>${tokK(r.C)}-token chunks: the whole prompt fits in one chunk, so this is the same as no chunking.</b></p>`,`<p><b>切成 ${tokK(r.C)} 个 token 的 chunk 时，整个 prompt 装得进一个 chunk，和不切一样。</b></p>`)+verdict();
  const lead=tx(`<p><b>${tokK(r.C)}-token chunks: the users’ longest wait drops from about ${fR(r0.longest)} to about ${fR(r.longest)}, and the document’s first token comes ${laterW(r,r0)}.</b></p>`,
                `<p><b>切成 ${tokK(r.C)} 个 token 的 chunk：用户最长的等待从约 ${fR(r0.longest)} 降到约 ${fR(r.longest)}，文档的首 token ${laterW(r,r0)}。</b></p>`);
  const brk=edgeOf(r).longOff?'':tx(' (the bracket on the chart)','（图上的括号）');
  const why=tooSmall(r)?tx(`<p>Why: the document’s arithmetic is the same, but it is now spread over many short steps, so no single step is long. With chunks this small, though, every step takes as long as its memory reads, however little it prefills, and the document needs ${fint(r.n)} of them. So its first token comes much later: most of that time goes to reading the weights again and again.</p>`,
                          `<p>原因：文档的计算总量没变，只是分到了许多短步里，所以没有哪一步特别长。可 chunk 这么小时，每一步不管 prefill 多少，都要花满读显存的时间，而文档要跑 ${fint(r.n)} 步。所以它的首 token 晚了很多，大部分时间都花在一遍又一遍地读权重上。</p>`)
    :tx(`<p>Why: the document’s arithmetic is the same, but it is now spread over many short steps, so no single step is long. The later a chunk, the more earlier tokens it looks back at, so the last chunk’s step is the longest${brk}. The first token comes later because ${laterWhyW(r)}.</p>`,
        `<p>原因：文档的计算总量没变，只是分到了许多短步里，所以没有哪一步特别长。chunk 越靠后，要回看的前文越多，所以最后一个 chunk 的那一步最长${brk}。首 token 会变晚，是因为${laterWhyW(r)}。</p>`);
  return lead+verdict()+why;
}
/* part 5: a small chunk rides on arithmetic a usual step leaves idle; too small a chunk only adds steps. The part is observed once the
   learner has tried a chunk that is too small: every step set by its memory reads, so the users' longest wait cannot shrink any more
   (it is the last step's reads, the same for any smaller chunk) while every extra step reads all the weights again */
const tooSmall=r=>r.n>1&&r.nMem===r.n;
/* the largest chunk offered whose first step still rides free, for the trade-off in words (256 in the mission) */
const freeC=()=>{const x=crossOf(S).first; return K_OPTS.filter(C=>C>0&&C<x&&C<S.N).reduce((m,C)=>Math.max(m,C),0);};
function freeWhyHTML(){
  const fc=freeC(), rf=fc?run({...S,C:fc}):null, r0=run({...S,C:0});
  const here=fc?tx(`, like ${tokK(fc)} tokens here,`,`（这里是 ${tokK(fc)} 个 token）`):'';
  return `<details class="ck-why"><summary>${tx('Why is a chunk that just rides free the best choice?','为什么切到刚好能白搭车的大小最合适？')}</summary>`+
    tx(`<p>Neither a large nor a tiny chunk works well. A large chunk makes each step long: its arithmetic takes far longer than the step’s memory reads, and every user waits for that step. A tiny chunk keeps each step as short as its memory reads, but the document then needs so many steps, each reading all the weights again, that its first token comes very late.</p><p>A chunk just small enough to ride free${here} sits in between. It uses the arithmetic a usual step leaves idle, so the users’ wait stays close to a usual step. It is still large enough that its steps spend little extra time just reading, so the document’s first token comes ${rf?laterW(rf,r0):'only a little later'}, not many times later.</p>`,
       `<p>chunk 太大、太小都不好。chunk 太大，每一步的计算远远超过读显存的时间，这一步就很长，所有用户都得等它。chunk 太小，每一步虽然只花读显存的时间，可文档要跑很多步，每一步都要把权重重新读一遍，所以文档的首 token 来得很晚。</p><p>刚好能白搭车的 chunk${here}在两者之间。它用的是平时一步闲着的算力，所以用户的等待和平时一步差不多。它又没有小到让每一步都在空等读取，多花的时间不多，所以文档的首 token ${rf?laterW(rf,r0):'只晚了一点'}，而不是晚好几倍。</p>`)+`</details>`;
}
function obsFree(){
  const r=R(), r0=run({...S,C:0}), f=r.steps[0], done=!!observed[SIX('free')];
  if(r.C===0||r.n===1) return tx(`<p><b>No chunking: one long step.</b> Pick a chunk size.</p>`,`<p><b>不切 chunk：只有很长的一步。</b>选一个 chunk 大小。</p>`);
  const near=r.longest<r.normal.t*1.5, why=done?freeWhyHTML():'';
  if(tooSmall(r)){
    const lead=tx(`<p><b>${tokK(r.C)}-token chunks: ${near?'the users still wait about as long as in a usual step':'the users’ wait has stopped shrinking'}, but the document’s first token now comes at about ${fR(r.ttft)}, compared with about ${fR(r0.ttft)} without chunking.</b></p>`,
                  `<p><b>切成 ${tokK(r.C)} 个 token 的 chunk：${near?'用户的等待仍和平时一步差不多':'用户的等待已经不再缩短'}，可文档的首 token 要到约 ${fR(r.ttft)} 才出来，不切 chunk 时只要约 ${fR(r0.ttft)}。</b></p>`);
    const cause=tx(`<p>Why: a chunk this small adds less arithmetic than the step’s memory reads take, so the reads set the time of every step (compare the two bars). Making the chunk smaller cannot shorten those reads; it only adds steps. The document now needs ${fint(r.n)} steps, and each one reads all the weights again to prefill just ${fint(r.C)} tokens. So most of the time goes to reading the weights, while the GPU’s arithmetic sits idle.</p>`,
                   `<p>原因：chunk 这么小，加进来的计算比读显存的时间还短，所以每一步的耗时都由读显存决定（比较图里的两根条）。chunk 再切小，也缩短不了这些读取，只会让步数更多。现在文档要跑 ${fint(r.n)} 步，每一步都为了区区 ${fint(r.C)} 个 token 把权重完整读一遍，所以大部分时间都花在读权重上，GPU 的算力却闲着。</p>`);
    return lead+cause+why;
  }
  const lead=tx(`<p><b>${tokK(r.C)}-token chunks: the users wait at most about ${fR(r.longest)}${near?', close to a usual step':''}, and the document’s first token comes ${laterW(r,r0)}.</b></p>`,
                `<p><b>切成 ${tokK(r.C)} 个 token 的 chunk：用户最多等约 ${fR(r.longest)}${near?'，和平时一步差不多':''}；文档的首 token ${laterW(r,r0)}。</b></p>`);
  const first=!f.cb?tx('The first chunk’s step barely gets longer: its arithmetic fits into the time the step spends reading memory anyway.','第一个 chunk 的那一步几乎没变长：它的计算正好塞进这一步本来就要花的读显存时间里。')
    :tx('This chunk is too big to ride free: already in its first step, the arithmetic takes longer than the memory reads.','这个 chunk 太大，搭不了便车：从第一步起，计算就比读显存还久。');
  const later=tx(' Later chunks cost more, because they look back at more earlier tokens: the longest step is the last one.','越靠后的 chunk 越贵，因为要回看的前文越多：最长的是最后一步。');
  const cost=r.nMem?tx('<p>The price of small chunks is the number of steps: every step reads all the weights once, however little it prefills, and those reads pile up before the document’s first token.</p>','<p>chunk 太小的代价是步数：每一步不管 prefill 多少，都要把权重完整读一遍，步数一多，这些读取的时间就都压在文档的首 token 上。</p>')
    :tx('<p>Every step here is set by arithmetic, so splitting finer costs little: each extra step only computes the users’ tokens once more.</p>','<p>这里每一步都由计算决定，所以切得更细代价很小：多出来的每一步只是再顺带算一遍用户的 token。</p>');
  const smallest=K_OPTS.filter(C=>C>0).reduce((m,C)=>Math.min(m,C),Infinity);
  const nudge=done||r.C<=smallest?'':(f.cb?tx('<p>Try a smaller chunk.</p>','<p>试试更小的 chunk。</p>')
    :tx(`<p>Now make the chunk even smaller, down to ${fint(smallest)} tokens: does the users’ wait keep shrinking?</p>`,`<p>再把 chunk 切得更小，一直切到 ${fint(smallest)} 个 token：用户的等待还会继续缩短吗？</p>`));
  return lead+`<p>${first}${later}</p>`+cost+nudge+why;
}
function remember(){
  return tx(`<ul><li><b>The stall.</b> Without chunking, a long prompt’s prefill is one long step, and no user who is decoding gets a token until it ends.</li><li><b>The fix.</b> Chunked prefill puts one chunk into each step, together with the users’ decode tokens. Every step is short and ends with a token for every user, while the long prompt’s first token comes only a little later.</li><li><b>How small.</b> A usual step waits on memory and leaves arithmetic idle, so a small chunk adds almost nothing. Later chunks look back at more earlier tokens and cost more: the last chunk sets the longest wait. But a chunk that is too small cannot shorten the wait any more, because every step already takes as long as reading the weights. It only adds steps, each reading all the weights again, so the long prompt’s first token comes much later.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Chunking makes the long prompt much slower.” Its arithmetic is the same, only spread over more steps; each extra step adds just the users’ few tokens, so its first token comes only a little later.</p><p>✗ “The smaller the chunk, the better.” Once a chunk rides free, making it smaller hardly shortens the users’ wait, but every extra step reads all the weights again, so the first token keeps getting later.</p><p>✗ “Chunking reduces the work.” The prompt’s work is the same however it is split; chunking only spreads it out, and every extra step adds a little work of its own.</p></div>${DEEP()}${A.shell.toScenarios()}`,
            `<ul><li><b>卡住的原因。</b>不切 chunk 时，长 prompt 的 prefill 就是很长的一步，这一步结束之前，正在 decode 的用户谁也拿不到 token。</li><li><b>解决办法。</b>chunked prefill 每一步只放一个 chunk，和用户的 decode token 一起算。每一步都很短，结束时每个用户都拿到一个 token；长 prompt 的首 token 只晚一点点。</li><li><b>切多小。</b>平时的一步在等读显存，算力闲着，所以小 chunk 几乎不增加耗时。越靠后的 chunk 要回看的前文越多，越贵：最长的等待由最后一个 chunk 决定。可 chunk 太小也不行：每一步本来就要花读权重的时间，再切小也缩短不了用户的等待，只会多出很多步，每一步都把权重重新读一遍，所以长 prompt 的首 token 会晚很多。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “切成 chunk，长 prompt 会慢很多。”它的计算量不变，只是分到更多步里；多出来的每一步只顺带算用户那几个 token，所以首 token 只晚一点点。</p><p>✗ “chunk 越小越好。”chunk 小到能白搭车以后，再切小，用户的等待几乎不再缩短；可多出来的每一步都要再读一遍权重，首 token 越来越晚。</p><p>✗ “切成 chunk 能减少计算量。”prompt 的计算量怎么切都一样；切 chunk 只是把它摊开，而多出来的每一步还有一点自己的额外工作。</p></div>${DEEP()}${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real workloads, one plain phrase each (R2.1); they set the workload, not the chunk size
   ===================================================================== */
const PRESETS=[
 {id:'doc',t:{en:'Document summary while users chat',zh:'有人聊天时来了一份长文档'},d:{en:'16 users at 4K tokens each; a 32K-token document to summarize',zh:'16 个用户，每人 4K 个 token 的上下文；一份 32K 个 token 的文档要写摘要'},s:{U:16,ctx:4096,N:32768}},
 {id:'upload',t:{en:'A long upload in a busy service',zh:'繁忙的服务里来了一份超长上传'},d:{en:'32 users at 4K tokens each; a 120K-token upload',zh:'32 个用户，每人 4K 个 token 的上下文；一份 120K 个 token 的上传'},s:{U:32,ctx:4096,N:122880}},
 {id:'chat',t:{en:'Busy chat with short prompts',zh:'繁忙的聊天，prompt 都很短'},d:{en:'64 users at 2K tokens each; a new 2K-token message',zh:'64 个用户，每人 2K 个 token 的上下文；来了一条 2K 个 token 的新消息'},s:{U:64,ctx:2048,N:2048}},
 {id:'code',t:{en:'Code assistant',zh:'代码助手'},d:{en:'8 users at 8K tokens each; a new 8K-token prompt of code',zh:'8 个用户，每人 8K 个 token 的上下文；来了一个 8K 个 token 的代码 prompt'},s:{U:8,ctx:8192,N:8192}}];
const presetNow=()=>PRESETS.findIndex(p=>p.s.U===S.U&&p.s.ctx===S.ctx&&p.s.N===S.N);
function scenHTML(i){
  const p=PRESETS[i], r=R(), r0=run({...S,C:0}), name=esc(L(p.t));
  let head;
  if(r.n>1) head=tx(`<p><b>${name}, with ${tokK(r.C)}-token chunks: the users’ longest wait drops from about ${fR(r0.longest)} to about ${fR(r.longest)}, and the prompt’s first token comes ${laterW(r,r0)}.</b></p>`,
                    `<p><b>${name}，切成 ${tokK(r.C)} 个 token 的 chunk：用户最长的等待从约 ${fR(r0.longest)} 降到约 ${fR(r.longest)}，prompt 的首 token ${laterW(r,r0)}。</b></p>`);
  else head=tx(`<p><b>${name}, ${r.C?'one chunk holds the whole prompt, the same as no chunking':'without chunking'}: between two tokens the users wait for the whole prefill, about ${fR(r.longest)}.</b></p>`,
               `<p><b>${name}，${r.C?'一个 chunk 就装下整个 prompt，和不切一样':'不切 chunk'}：用户两个 token 之间要等整个 prefill，约 ${fR(r.longest)}。</b></p>`);
  let open='';
  if(p.id==='upload') open=tx('A 120K-token prefill costs far more than 4 times a 32K one, because its attention grows with the square of the prompt. ','120K 个 token 的 prefill 远不止 32K 的 4 倍，因为它的 attention 随 prompt 长度的平方增长。');
  if(p.id==='chat') open=tx('A short prompt costs little either way: even without chunking, its prefill is a short step. ','短 prompt 切不切都不贵：就算不切，它的 prefill 也只是短短的一步。');
  if(p.id==='code') open=tx('Even an 8K prompt freezes the users for many usual steps when it is prefilled at once; chunks bring the wait back near a usual step. ','就算只有 8K 个 token，一步 prefill 完也会让用户卡住好多个平时的步长；切成 chunk 后，等待又回到接近平时的一步。');
  let why;
  if(r.n===1) why=tx('The whole prefill runs in one step, and the users’ tokens come out only when it ends.','整个 prefill 在一步里跑完，用户的 token 要等它结束才出来。');
  else if(tooSmall(r)) why=tx(`The prompt’s arithmetic is spread over the steps, but with chunks this small every step takes as long as its memory reads, however little it prefills. The first token comes much later because each of the many steps reads all the weights again.`,
              `prompt 的计算分摊到了各步里；可 chunk 这么小时，每一步不管 prefill 多少，都要花满读显存的时间。首 token 晚了很多，是因为这么多步里，每一步都要把权重重新读一遍。`);
  else why=tx(`The prompt’s arithmetic is spread over the steps. The last chunk looks back at the most earlier tokens, so its step is the longest. The first token comes later because ${laterWhyW(r)}.`,
              `prompt 的计算分摊到了各步里。最后一个 chunk 要回看的前文最多，所以它那一步最长。首 token 会变晚，是因为${laterWhyW(r)}。`);
  return head+`<p>${open}${why}</p>`;
}

/* =====================================================================
   challenge
   ===================================================================== */
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chPoint(S.C), best=chBest(), nxt=chPoint(best.C*2);
  const nm=C=>C?tx(`${tokK(C)}-token chunks`,`${tokK(C)} 个 token 的 chunk`):tx('no chunking','不切 chunk');
  let h;
  if(me.ok&&me.C===best.C) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: with ${nm(me.C)} every user waits at most ${fT(me.wait)}.</b> No larger chunk stays within 50 ms, so this gives the document its earliest first token.</p>`,
                                `<p class="okline" style="margin-top:12px"><b>达成目标：切成 ${nm(me.C)}时，每个用户最多等 ${fT(me.wait)}。</b>再大的 chunk 都会超过 50 ms，所以这已经是文档最早的首 token。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: with ${nm(me.C)} every user waits at most ${fT(me.wait)}.</b> But a larger chunk also stays within 50 ms and starts the answer sooner, since it needs fewer steps.</p>`,
                      `<p class="okline" style="margin-top:12px"><b>达成目标：切成 ${nm(me.C)}时，每个用户最多等 ${fT(me.wait)}。</b>不过更大的 chunk 也不超过 50 ms，步数更少，回答开始得更早。</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: with ${nm(me.C)} the users wait up to ${fT(me.wait)} between two tokens, over 50 ms.</b></p><p class="small">Hint: the last chunk’s step is the longest, and halving the chunk about halves it. Which chunk’s last step fits in 50 ms?</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：${me.C?`切成 ${nm(me.C)}`:'不切 chunk'}时，用户两个 token 之间要等 ${fT(me.wait)}，超过了 50 ms。</b></p><p class="small">提示：最后一个 chunk 的那一步最长，chunk 减半，它也差不多减半。哪种 chunk 的最后一步能放进 50 ms？</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the bars are in ms on a 0–140 ms scale; a wait beyond it is drawn full length and says so ("off the scale") */
  const pts=K_OPTS.map(chPoint), MAXMS=140;
  /* brief (the owner's review of batch 8): the chart keeps the chunk sizes that meet the goal, the pick and the initial setup, and
     says how many it left out; the title and the aria name only what it shows */
  const shownPts=pts.filter(p=>p.ok||p.C===me.C||p.C===CH.start);
  h+=A.ui.cmpChart({title:tx(`The users’ longest wait between tokens, ${LL.name} on one H100 SXM with ${CH.U} users at ${tokK(CH.ctx)} tokens each and a ${tokK(CH.N)}-token document (goal: at most 50 ms)`,`一张 H100 SXM 跑 ${LL.name}、${CH.U} 个用户每人 ${tokK(CH.ctx)} 个 token、来了一份 ${tokK(CH.N)} 个 token 的文档时，用户两个 token 之间最长的等待（目标是最多 50 ms）`).replace(/Llama-3\.1-8B/g,MODEL),
    aria:tx('The users’ longest wait between tokens: ','用户两个 token 之间最长的等待：')+shownPts.map(p=>`${kLabel(p.C)}${tx(': ','：')}${fT(p.wait)}`).join(tx('; ','；')),
    brief:true,hiddenText:n=>tx(n===1?'The remaining chunk size goes over 50 ms and is not shown.':`The other ${n} chunk sizes go over 50 ms and are not shown.`,`其余 ${n} 种 chunk 大小超过了 50 ms，图里没有列出。`),
    rows:pts.map(p=>({label:tx(`${p.C?`${tokK(p.C)}-token chunks`:'No chunking'}: first token at ${fT(p.ttft)}`,`${p.C?`${tokK(p.C)} 个 token 的 chunk`:'不切 chunk'}：首 token 在 ${fT(p.ttft)}`),value:Math.min(p.wait*1e3,MAXMS),text:fT(p.wait)+(p.wait*1e3>MAXMS?' ▸':''),ok:p.ok,me:p.C===me.C,
      tags:(p.C===CH.start?[{text:tx('initial setup','初始设置')}]:[]).concat(p.C===me.C?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:50,goalLabel:tx('goal: 50 ms','目标 50 ms'),max:MAXMS,okText:tx('50 ms or less','50 ms 以内'),noText:tx('over 50 ms','超过 50 ms'),goalText:tx('goal: at most 50 ms','目标：最多 50 ms')})+`<p class="small muted ck-cmpnote">${tx(`▸ The bar runs past the scale, which ends at ${MAXMS} ms.`,`▸ 表示这根条超出了刻度，刻度到 ${MAXMS} ms 为止。`)}</p>`;
  /* baseline → effect → reason (R3.10), in words: the numbers are on the chart (R6.20); the next larger chunk misses the goal and is
     not on the brief chart, so its wait is on hover */
  const nxTip=tx(`With ${nm(nxt.C)} the users wait up to ${fT(nxt.wait)} between two tokens.`,`切成 ${nm(nxt.C)}时，用户两个 token 之间最多要等 ${fT(nxt.wait)}。`);
  h+=tx(`<p class="small">Initial setup, no chunking: one long step, and every user waits for the whole prefill. With ${nm(best.C)} the longest step, the last chunk’s, fits within 50 ms; ${DEF(`with ${nm(nxt.C)} it does not`,nxTip)}.</p>`,
        `<p class="small">初始设置是不切 chunk：只有很长的一步，每个用户都要等整个 prefill。切成 ${nm(best.C)}时，最长的那一步（最后一个 chunk）能放进 50 ms。${DEF(`切成 ${nm(nxt.C)}时就放不进了`,nxTip)}。</p>`);
  h+=tx(`<p class="small">Why: the last chunk attends to the most earlier tokens, so its step is the longest, and it shrinks with the chunk. The shorter the document, the less its last chunk has to look back at, so the larger the chunk that fits. Smaller chunks meet the goal too, but they need more steps, and every step computes the users’ tokens again (and, for tiny chunks, reads all the weights again), so the first token comes later.</p>`,
        `<p class="small">原因：最后一个 chunk 要回看的前文最多，所以它那一步最长，而且随 chunk 变小而变短。文档越短，最后一个 chunk 要回看的前文就越少，能用的 chunk 也就越大。更小的 chunk 也能达标，但步数更多：每多一步，就要再算一遍用户的 token（chunk 很小时还要再读一遍权重），首 token 会更晚。</p>`);
  const bud=best.C+CH.U;
  h+=tx(`<p class="small muted">Note: real servers set a token budget per step instead of a chunk size, and it counts the decode tokens too. In vLLM (max_num_batched_tokens), a ${fint(bud)}-token budget with ${CH.U} users decoding leaves ${fint(best.C)} tokens for the chunk. The 50% and 60% efficiencies are this lab’s assumptions: before you set a budget, measure your server’s time per output token at the 99th percentile while long prompts arrive.</p>`,
        `<p class="small muted">注意：真实服务设的是每步的 token 预算，不是 chunk 大小，而且 decode 的 token 也算在内。在 vLLM 里（max_num_batched_tokens），${CH.U} 个用户在 decode 时，${fint(bud)} 个 token 的预算留给 chunk 的是 ${fint(best.C)} 个。50% 和 60% 这两个效率是这个 Lab 的假设：定预算之前，要在长 prompt 到来时，实测服务的每个输出 token 耗时的 99 分位数。</p>`);
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Predict first, see why one long prefill freezes every user who is decoding, then split it into chunks and find out how small they should be. Start here if this is new to you.',
            '一共 6 个部分。先预测，再看为什么一次长 prefill 会卡住所有正在 decode 的用户，然后把它切成 chunk，看看该切多小。第一次学从这里开始。'),
  sandbox:tx(`Four real workloads for ${MODEL} on one ${GPUN}, from a busy chat with short prompts to a 120K-token upload. Pick one, then try chunk sizes in it.`,
             `四个真实负载，都是一张 ${GPUN} 跑 ${MODEL}，从 prompt 都很短的繁忙聊天，到 120K 个 token 的超长上传。选一个，再在里面试不同的 chunk 大小。`),
  challenge:tx(`One ${GPUN} serves 32 chats with ${MODEL} when a 16K-token document arrives. Change only the chunk size: keep every user’s wait between tokens within 50 ms, and give the document its first token as early as you can.`,
               `一张 ${GPUN} 用 ${MODEL} 服务 32 个对话时，来了一份 16K 个 token 的文档。只改 chunk 大小：让每个用户两个 token 之间的等待不超过 50 ms，并让文档的首 token 尽量早。`)
});
const SRC={
 cfg:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 dbx:'https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices',
 sarathi:'https://arxiv.org/abs/2308.16369',
 serve:'https://www.usenix.org/conference/osdi24/presentation/agrawal',
 fastgen:'https://arxiv.org/abs/2401.08671',
 vsched:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py',
 vcfg:'https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py',
 vargs:'https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py',
 sglargs:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/schedule.py',
 sglpol:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py',
 sglsch:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py'
};
const TILE_A=()=>`<a href="#map~inference" data-tipx="${esc(tx('Click to open the tree at Inference serving and see this tile.','点击回到知识树的“推理服务”这一枝，看这个方块。'))}">${esc(tx('“Why does one huge prompt slow everyone down?”','“一个超长 prompt 为什么会拖慢所有人？”'))}</a>`;
function FIDPOP(){
  const b=B0(), p=run({...BASE,C:0}).steps[0], aiU=b.normal.fl/b.normal.by, aiP=p.fl/p.by, lo=H100.peak*0.2/H100.bw, hi=H100.peak/(H100.bw*0.2);
  return tx(
 `<p><b>Toy run:</b> the page runs one step model step by step, the one the home-page tile ${TILE_A()} and the lab ${labLink('prefill-decode')} use; ${J('math','“Show the math”')} works out the longest step line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>${MODEL} in BF16 on one ${GPUN}: 32 ${DEF('layers',kt('layer'))}, ${DEF('hidden size',kt('hidden'))} 4,096, 8 KV heads of 128 dimensions, ${A.ui.big?A.ui.big(P_TOTAL):A.fmt.big(P_TOTAL)} ${DEF('parameters',kt('param'))}.</li>`+
 `<li>A step takes the longer of its arithmetic and its memory traffic, because the GPU does both at the same time. Arithmetic is 2 FLOPs per multiplied parameter per token, the ${DEF('LM head',kt('lmhead'))} once per new token, plus attention. Memory traffic is every parameter but the ${DEF('embedding table',kt('embedding'))}, read once per step, plus 128 KiB of KV cache for every token of every context in the step.</li>`+
 `<li>Each step gives every user who is decoding one token. While the prompt is prefilled, each step also computes the whole prompt or one chunk of it, whose tokens attend to themselves and to every earlier prompt token. The prompt’s first token comes when its last chunk’s step ends; after it, the prompt decodes with the users.</li>`+
 `<li>How many users fit comes from the lab ${labLink('kv-cache')}’s model: the 80 GB read as 80 GiB, ${pct0(UTIL)} of it usable, ${OH} GiB of runtime overhead, ${gbS(W_MEM)} of weights.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>50% of the dense BF16 peak and 60% of the peak bandwidth are this lab’s assumptions (${J('eff','see Settings')}). They set the times and the size up to which a chunk rides free, not which limit sets a usual step or an unchunked prefill: for any shares from 20% to 100% the ridge stays between ${fint(lo)} and ${fint(hi)} FLOPs per byte, while a usual step of the mission does ${fixd(aiU,1)} and the unchunked prefill ${fint(aiP)}.</li>`+
 `<li>The users’ contexts stay the same while the prompt is prefilled. Really each grows by one token per step; over a prefill of up to a few hundred steps that adds a few percent to what a step reads.</li>`+
 `<li>One prompt arrives, just as a step ends, and nothing else joins: no queue and no other prefill. The chunk size is fixed; real engines set a token budget per step, which counts the decode tokens too, so there a chunk is the budget minus the decode tokens.</li>`+
 `<li>The memory traffic of activations and the time to launch kernels are left out; the 60% comes from measured step times, which include them.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${MODEL}: ${cite(SRC.cfg,'config.json')}.</li>`+
 `<li>${GPUN}, from ${cite(SRC.h100,'NVIDIA’s product page')}: BF16 1,979 TFLOPS “with sparsity”, so 989.5 dense; 3.35 TB/s; 80 GB.</li>`+
 `<li>${cite(SRC.dbx,'Databricks (2023)')}: 60% of peak bandwidth at batch size 1 on two H100 80GB GPUs.</li>`+
 `<li>${cite(SRC.sarathi,'SARATHI (Agrawal et al., 2023)')} and ${cite(SRC.serve,'Sarathi-Serve (OSDI 2024)')}: prefills split into chunks, run in the same batch as ongoing decodes.</li>`+
 `<li>vLLM’s scheduler: ${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-28. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>这一页一步一步地跑一个耗时模型，和首页方块${TILE_A()}、${labLink('prefill-decode')} 这个 Lab 用的是同一个；${J('math','“展开公式”')}里逐行算出最长的那一步。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>一张 ${GPUN} 用 BF16 跑 ${MODEL}：32 ${DEF('层',kt('layer'))}，${DEF('隐藏维度',kt('hidden'))} 4,096，8 个 KV head，每个 128 维，一共 ${A.ui.big?A.ui.big(P_TOTAL):A.fmt.big(P_TOTAL)} 个${DEF('参数',kt('param'))}。</li>`+
 `<li>一步的耗时取计算和读写显存中较长的那个，因为 GPU 同时做这两件事。计算按每个参与乘法的参数每个 token 2 FLOP，${DEF('LM head',kt('lmhead'))} 每个新 token 算一次，再加 attention。读写显存算除 ${DEF('embedding 表',kt('embedding'))}外的全部参数，每一步读一遍；再加这一步里每个上下文的每个 token 128 KiB 的 KV cache。</li>`+
 `<li>每一步给每个正在 decode 的用户各算一个 token。prompt 正在 prefill 时，每一步还要算整个 prompt 或它的一个 chunk，chunk 里的 token 要对它们自己和前面每个 prompt token 做 attention。最后一个 chunk 的那一步结束时，prompt 的首 token 出来；之后 prompt 和用户们一起 decode。</li>`+
 `<li>最多能放下几个用户，用的是 ${labLink('kv-cache')}这个 Lab 的模型：80 GB 按 80 GiB 计，可用 ${pct0(UTIL)}，运行时开销 ${OH} GiB，权重 ${gbS(W_MEM)}。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>稠密 BF16 峰值的 50% 和峰值带宽的 60% 是这个 Lab 的假设（${J('eff','见“自由调参”')}）。它们决定时间长短，以及多大的 chunk 能白搭车，但不改变平时的一步和不切 chunk 的 prefill 各受哪个限制：两个比例在 20% 到 100% 之间怎么取，拐点都在每字节 ${fint(lo)} 到 ${fint(hi)} 个 FLOP 之间，而任务里平时的一步是 ${fixd(aiU,1)}，不切 chunk 的 prefill 是 ${fint(aiP)}。</li>`+
 `<li>prompt 在 prefill 时，用户的上下文保持不变。实际上每一步都会多一个 token；prefill 最多几百步，每一步要读的量多出几个百分点。</li>`+
 `<li>只来一个 prompt，正好在一步结束时到达，之后没有别的请求加入：不排队，也没有别的 prefill。chunk 大小是固定的；真实引擎设的是每步的 token 预算，decode 的 token 也算在内，所以那里的 chunk 是预算减去 decode 的 token。</li>`+
 `<li>不计激活值读写显存的流量，也不计启动 kernel 的时间；60% 来自实测的每步耗时，里面已经包含了它们。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${MODEL}：${cite(SRC.cfg,'config.json')}。</li>`+
 `<li>${GPUN}：${cite(SRC.h100,'NVIDIA 产品页')}，BF16 1,979 TFLOPS 标着“with sparsity”，所以稠密是 989.5；3.35 TB/s；80 GB。</li>`+
 `<li>${cite(SRC.dbx,'Databricks（2023）')}：在两张 H100 80GB 上，batch 大小为 1 时用到峰值带宽的 60%。</li>`+
 `<li>${cite(SRC.sarathi,'SARATHI（Agrawal 等，2023）')}和 ${cite(SRC.serve,'Sarathi-Serve（OSDI 2024）')}：把 prefill 切成 chunk，和正在进行的 decode 放在同一个 batch 里跑。</li>`+
 `<li>vLLM 的调度器：${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-28。原型，尚未经专家审阅。</p></div></div>`);
}
/* the prerequisite is two labs: prefill-decode (the two limits) and continuous batching (a joining prefill runs in a step everyone waits for) */
const PRIPOP=()=>tx(`<p>A request is served in two phases. Prefill runs the whole prompt through the model in one step and ends with the first token; each decode step then adds one token. A step takes the longer of its arithmetic and its memory traffic: prefill is limited by arithmetic, a decode step by reading the weights and the KV cache. The lab ${labLink('prefill-decode')} works both out.</p><p>A server decodes many requests in each step, since they share that step’s read of the weights. A running batch admits a new request between steps, and the new request’s prefill runs inside a step that every running request waits for: the lab ${labLink('continuous-batching')} shows the schedule.</p>`,
                    `<p>服务一个请求分两个阶段。prefill 在一步里把整个 prompt 过一遍模型，以首 token 结束；之后每个 decode 步多写一个 token。一步的耗时取计算和读写显存中较长的那个：prefill 受计算限制，decode 步受读权重和 KV cache 的限制。${labLink('prefill-decode')} 这个 Lab 把两者都算了出来。</p><p>服务端每一步同时 decode 很多个请求，因为它们共用这一步读权重的那一遍。在跑的 batch 在两步之间接纳新请求，新请求的 prefill 就在某一步里跑，在跑的请求都要等这一步：${labLink('continuous-batching')} 这个 Lab 画出了这样的调度。</p>`);
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers */
const OUTCOME=()=>tx('explain why one long prompt stalls every user who is decoding, and choose a chunk size that keeps their tokens coming within a target.',
                     '讲清一个长 prompt 为什么会卡住所有正在 decode 的用户，并选出一个 chunk 大小，让他们的 token 按目标准时到来。');
function exGo(p){setMode('sandbox'); byLearner(()=>setState(p)); A.ui.reveal($('#ck-stage',app));}
/* a new case's options in natural order, with a note on what each wrong one misses */
const EXITS=()=>{
  const a=B0(), sN=run({...BASE,C:256}), f256=sN.steps[0];
  const t1=run({U:64,ctx:4096,N:32768,C:0}), t1s=t1.steps[0];
  const t2=run({U:16,ctx:4096,N:65536,C:2048}), t2l=t2.steps[t2.li], c2=B2(), c2l=c2.steps[c2.li];
  return {
 core:()=>chSubmitted, explain:[
  {q:tx(`Without chunking, a 32K-token prompt makes the 16 users who are decoding wait about ${fixd(a.longest,1)} s for their next token. Why?`,`不切 chunk 时，一个 32K 个 token 的 prompt 让正在 decode 的 16 个用户等了约 ${fixd(a.longest,1)} s 才拿到下一个 token。为什么？`),
   o:[tx('The whole prefill runs in one step, and a step hands out the users’ tokens only when it ends, so they wait for all of the prompt’s arithmetic.','整个 prefill 在一步里算完，而一步要等结束才把用户的 token 交出去，所以他们要等 prompt 的全部计算做完。'),
     tx('The prefill has to read the prompt’s KV cache from GPU memory, and that read is so large that the users’ decode step waits for it to finish.','prefill 要从显存里读 prompt 的 KV cache，这次读取量太大，用户的 decode 步只好等它读完。'),
     tx('The users’ contexts now include the prompt’s 32K tokens, so each of their decode steps reads that much more KV cache and slows down.','用户的上下文里多了 prompt 的 32K 个 token，所以他们每个 decode 步都要多读这么多 KV cache，于是变慢。')],a:0,
   n:[null,tx('Prefill is set by arithmetic, not by memory reads: in that step the arithmetic bar is far longer than the memory bar.','prefill 受计算限制，不是读显存：这一步里，计算条比读写显存条长得多。'),
     tx('Each user attends only to its own context. The prompt belongs to another request, so the users’ decode tokens read only their own 4K tokens, as before.','每个用户只对自己的上下文做 attention。这个 prompt 属于另一个请求，所以用户的 decode token 照旧只读他们自己的 4K 个 token。')],
   why:tx('One step pushes every prompt token through all the weights, and each attends to every earlier one: a huge amount of arithmetic. The users’ tokens are in that same step, so, cheap as they are, they come out only when it ends.',
          '一步要让每个 prompt token 都和全部权重相乘，每个 token 还要对它前面的每个 token 做 attention：计算量巨大。用户的 token 也在这一步里，它们再便宜，也要等这一步结束才出来。')},
  {q:tx(`A step in which 16 users decode at 4K tokens each takes ${fT(a.normal.t)}. With the first 256-token chunk of a new prompt added, it takes ${fT(f256.t)}. Why so little more?`,`16 个用户在 4K 个 token 上 decode 的一步要 ${fT(a.normal.t)}。加进一个新 prompt 的第一个 256 个 token 的 chunk 后，要 ${fT(f256.t)}。为什么只多了这么一点？`),
   o:[tx('Reading the weights and the KV caches sets that step’s time and its arithmetic takes far less, so the chunk’s arithmetic fits in the time the reads take anyway.','决定这一步耗时的是读权重和 KV cache，它的计算用时少得多，所以 chunk 的计算正好塞进读取本来就要花的时间里。'),
     tx('The chunk’s tokens reuse the keys and values already in the users’ KV caches, so the GPU computes almost nothing new for them in that step.','chunk 的 token 直接复用用户 KV cache 里已有的 key 和 value，所以这一步里 GPU 几乎不用为它们算新的东西。'),
     tx('The GPU runs the chunk on other compute units once the users’ tokens are done, so the step itself hardly changes and the users do not wait.','GPU 等用户的 token 算完，再用别的计算单元算这个 chunk，所以这一步本身几乎不变，用户也不用等。')],a:0,
   n:[null,tx('Every prompt token gets its own keys and values: the chunk’s tokens are multiplied by all the weights, the hatched part of the arithmetic bar.','每个 prompt token 都有自己的 key 和 value：chunk 的 token 都要和全部权重相乘，就是计算条里斜纹的那一段。'),
     tx('The chunk runs in the same step as the users’ tokens, and every user gets its token only when that whole step ends.','chunk 和用户的 token 在同一步里跑，每个用户都要等这一整步结束才拿到 token。')],
   why:tx('A step takes the longer of its arithmetic and its memory reads. A usual step is set by reading the weights and the KV caches, and its own arithmetic is tiny. The chunk adds arithmetic, but the total still fits inside the time the reads take anyway, so the step barely gets longer.',
          '一步的耗时取计算和读写显存中较长的那个。平时的一步由读权重和 KV cache 决定，它自己的计算很少。chunk 增加了计算，但加起来仍然塞得进读取本来就要花的时间，所以这一步几乎不变长。')}],
 transfer:[
  {q:tx(`16 users decode at 4K tokens of context each. A 32K-token prompt arrives, not chunked, and they wait about ${fixd(a.longest,1)} s for their next token. Now 64 users decode instead, 4 times as many, and the same prompt arrives. About how long do they wait?`,
        `16 个用户在 decode，每人 4K 个 token 的上下文。这时来了一个 32K 个 token 的 prompt，不切 chunk，他们要等约 ${fixd(a.longest,1)} s 才拿到下一个 token。现在换成 64 个用户，是 16 个的 4 倍，来的还是这个 prompt。他们大约要等多久？`),
   o:[tx('Under 1 s','不到 1 s'),tx('1 to 2 s','1 到 2 s'),tx('2 to 4 s','2 到 4 s'),tx('More than 4 s','超过 4 s')],a:1,
   n:[tx(`The prompt’s own arithmetic already takes ${f1(t1.tcP)}, however many users decode.`,`光是 prompt 自己的计算就要 ${f1(t1.tcP)}，和有多少用户在 decode 无关。`),null,
      tx(`The 64 users add only their own tokens’ arithmetic, ${fT(t1.tcD)}, to the step.`,`64 个用户只给这一步加了他们自己 token 的计算，${fT(t1.tcD)}。`),
      tx('The wait does not grow with the number of users: it is the prompt’s prefill, which is the same work for 16 users or 64.','等待不会随用户数增长：它是 prompt 的 prefill，16 个用户和 64 个用户时的计算量一样。')],
   why:tx(`The step’s time is its arithmetic, and nearly all of it is the prompt’s: ${f1(t1.tcP)}. The 64 users’ tokens add ${fT(t1.tcD)}, so the step takes ${fT(t1s.t)} (with 16 users, ${fT(a.longest)}). Its memory traffic, ${fT(t1s.tm)}, stays far below.`,
          `这一步的耗时就是它的计算，几乎全是 prompt 的：${f1(t1.tcP)}。64 个用户的 token 加了 ${fT(t1.tcD)}，所以这一步要 ${fT(t1s.t)}（16 个用户时是 ${fT(a.longest)}）。它的读写显存只要 ${fT(t1s.tm)}，远低于此。`),
   go:()=>exGo({U:64,ctx:4096,N:32768,C:0}),goLabel:tx('Open it in Scenarios: 64 users','在应用场景里打开：64 个用户')},
  {q:tx(`16 users decode at 4K tokens each while a long prompt is prefilled in 2K-token chunks. With a 32K-token prompt, the users’ longest wait between two tokens is about ${fR(c2l.t)}. Now a 64K-token prompt arrives, twice as long. About how long is the users’ longest wait?`,
        `16 个用户在 decode，每人 4K 个 token。一个长 prompt 切成 2K 个 token 的 chunk 来 prefill。prompt 有 32K 个 token 时，用户两个 token 之间最长约等 ${fR(c2l.t)}。现在来了一个 64K 个 token 的 prompt，长了一倍。用户最长大约要等多久？`),
   o:[tx('Under 100 ms','不到 100 ms'),tx('100 to 150 ms','100 到 150 ms'),tx('150 to 250 ms','150 到 250 ms'),tx('More than 250 ms','超过 250 ms')],a:2,
   n:[tx('Halving the chunk would halve the wait, but here the chunk is the same and the prompt longer.','chunk 减半才会让等待减半，这里 chunk 没变，prompt 变长了。'),
      tx(`That is about the wait with the 32K-token prompt. The last chunk now looks back at twice as many earlier tokens.`,`这差不多是 32K 个 token 时的等待。现在最后一个 chunk 要回看的前文多了一倍。`),null,
      tx(`Only the last chunk’s attention grows with the prompt; its ${fint(2048)} tokens through the weights cost the same as before.`,`只有最后一个 chunk 的 attention 随 prompt 变长而增加；它的 ${fint(2048)} 个 token 和权重相乘的计算不变。`)],
   why:tx(`The longest step is still the last chunk’s. Its tokens now look back at about twice as many earlier tokens, so its attention about doubles, while its multiplications with the weights stay the same. Arithmetic sets the step: about ${fR(t2l.t)}, against about ${fR(c2l.t)} with the 32K-token prompt.`,
          `最长的仍是最后一个 chunk 的那一步。它的 token 要回看的前文多了约一倍，所以 attention 的计算约翻一倍，而和权重相乘的计算不变。这一步由计算决定：约 ${fR(t2l.t)}，32K 个 token 时约 ${fR(c2l.t)}。`),
   go:()=>exGo({U:16,ctx:4096,N:65536,C:2048}),goLabel:tx('Open it in Scenarios: a 64K-token prompt','在应用场景里打开：64K 个 token 的 prompt')}],
 practice:[
  {id:'inference-engine',d:tx('Level 2, then the first follow-up: each step decodes every running sequence, then admits waiting prompts while the token budget allows; extend it so that a long prompt takes what is left of the budget as a chunk instead of waiting for room for all of it.',
                              '第 2 关，再做第一条追问：每一步先给每个在跑的序列 decode 一个 token，再在 token 预算内接纳排队的 prompt；把它扩展成长 prompt 不必等到整个装得下，而是用剩下的预算先 prefill 一个 chunk。')},
  {id:'inference-serving',d:tx('Item 4, first deep dive, the chunked-prefill option: size each step’s prefill budget so a 16,000-token prompt does not hold the decoding requests below the 10-tokens-per-second floor, and weigh it against separate prefill and decode pools.',
                               '第 4 项的第一个深入话题里的 chunked prefill：定好每步的 prefill 预算，让 16,000 个 token 的 prompt 不会把正在 decode 的请求拖到每秒 10 个 token 以下，并和 prefill、decode 分池部署做比较。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'Chunked prefill',title:l?L(l.q):tx('Why does one huge prompt slow everyone down?','一个超长 prompt 为什么会拖慢所有人？'),
    minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: prefill, decode and batching','前置知识：prefill、decode 和 batch'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  const ph=k=>`<div class="ck-ph" data-ph="${k}"><div class="ck-ph-h"><b></b><span class="ck-ph-d"></span><span class="ck-ph-t mono"></span></div>`+
    `<div class="ck-rows"><div class="ck-row" data-k="c" tabindex="0"><span class="ck-rl">${tx('arithmetic','计算')}</span><div class="ck-bar"><i class="ck-bu"></i><i class="ck-bch"></i><b class="ck-goal" hidden></b><em class="ck-settag" hidden></em></div><span class="ck-rv mono"></span></div>`+
    `<div class="ck-row" data-k="m" tabindex="0"><span class="ck-rl">${tx('memory','读写显存')}</span><div class="ck-bar"><i class="ck-bw"></i><i class="ck-bk"></i><b class="ck-goal" hidden></b><em class="ck-settag" hidden></em></div><span class="ck-rv mono"></span></div></div></div>`;
  return `<div class="panel stage ck-stage" id="ck-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="ck-st-title"></div><div class="meter-sub" id="ck-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="ck-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="ck-legend"></div>`+
   `<div class="stage-graph graph ck-graph" id="ck-graph">`+
    `<div class="ck-tl" id="ck-tl"><div class="ck-cap" id="ck-tl-cap"></div><div class="ck-anat" id="ck-anat"></div>`+
     `<div class="ck-lh"><b id="ck-lh-u"></b><span id="ck-lh-u2"></span></div>`+
     `<div class="ck-brk" id="ck-brk" aria-hidden="true"><span class="ck-lg" id="ck-lg"><i></i><em></em></span><span class="ck-wt" id="ck-wt" hidden><i></i><em></em></span></div>`+
     `<div class="ck-track ck-tu" id="ck-tu" tabindex="0" role="img"><i class="ck-lfill"></i><div class="ck-mk" id="ck-mku"></div><i class="ck-wfill" hidden></i><i class="ck-head" hidden></i></div>`+
     `<div class="ck-lh ck-lh-p"><b id="ck-lh-p"></b><span id="ck-lh-p2"></span></div>`+
     `<div class="ck-arr" id="ck-arr" aria-hidden="true"><span class="ck-arr-a"></span><span class="ck-arr-f"></span></div>`+
     `<div class="ck-track ck-tp" id="ck-tp" tabindex="0" role="img"><div class="ck-mk" id="ck-mkp"></div><i class="ck-head" hidden></i></div>`+
     `<div class="ck-axis mono" id="ck-axis" aria-hidden="true"></div><i class="ck-band" id="ck-band" hidden aria-hidden="true"></i></div>`+
    `<div class="ck-play" id="ck-play">${playBtns()}<input type="range" id="ck-scrub" min="0" max="1000" step="1" value="1000" aria-label="${esc(tx('Time in the run','运行中的时刻'))}"></div>`+
    `<p class="ck-clock" id="ck-clock"></p>`+
    `<div class="ck-lim" id="ck-lim"><div class="ck-cap" id="ck-lim-cap"></div><div class="legend-row ck-limleg" id="ck-limleg"></div>${ph('u')}${ph('s')}</div>`+
    `<div class="ck-deep" id="ck-deep" hidden></div>`+
   `</div>`+
   `<p class="ck-scope small muted" id="ck-scope" tabindex="0"></p>`+
   `<div class="metrics ck-metrics" id="ck-metrics"></div>`+
   `<details class="math" id="ck-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ck-mathbody"></div></details></div>`;
}
const EFF_NOTE=()=>tx(`This lab assumes a step reaches 50% of the ${GPUN}’s dense BF16 peak (989.5 TFLOP/s), that is ${tfS(FS)} TFLOP/s; how much of the peak a real kernel reaches depends on the kernel and on how many tokens a step computes. It sets how long a prefill or a chunk takes, and so the chunk size up to which a chunk rides free.`,
  `本 Lab 假设一步能用到 ${GPUN} 稠密 BF16 峰值（989.5 TFLOP/s）的 50%，即 ${tfS(FS)} TFLOP/s；真实 kernel 能用到峰值的多少，取决于 kernel 本身和一步算多少个 token。它决定 prefill 和每个 chunk 要多久，也就决定了多大的 chunk 能白搭车。`);
const BW_NOTE=()=>tx(`This lab assumes a step reaches 60% of the peak memory bandwidth (3.35 TB/s), that is ${tbS(BS)} TB/s. Kernels never stream memory at the full peak; Databricks (2023) measured 60% at batch size 1 on two H100 80GB GPUs. It sets a usual decode step’s time, and so how much arithmetic a chunk can add before the step gets longer.`,
  `本 Lab 假设一步能用到峰值显存带宽（3.35 TB/s）的 60%，即 ${tbS(BS)} TB/s。kernel 读显存从来达不到满峰值；Databricks（2023）在两张 H100 80GB 上实测，batch 大小为 1 时是 60%。它决定平时一个 decode 步的耗时，也就决定了 chunk 能加多少计算而不让这一步变长。`);
const rng=(id,key,label,stops,val,fmt,note)=>A.ui.range({id,key,label,min:0,max:stops.length-1,value:Math.max(0,stops.indexOf(val)),out:fmt(val)})+'';
function controlsHTML(){
  const note=(h,id)=>`<div class="ctl-note"${id?` id="${id}"`:''}>${h}</div>`;
  const wrap=(html,n)=>html.replace(/<\/div>$/,n+'</div>');
  return `<aside class="panel controls" id="ck-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ck-lock" hidden></p>`+
   `<div class="lockable">`+wrap(rng('ck-ctl-U','U',tx('Users decoding','正在 decode 的用户数'),U_STOPS,S.U,userN),note('','ck-Unote'))+`</div>`+
   `<div class="lockable">`+wrap(rng('ck-ctl-X','X',tx('Their context','他们的上下文'),X_STOPS,S.ctx,v=>tx(`${tokK(v)} tokens`,`${tokK(v)} 个 token`)),note(tx('The tokens each user’s conversation holds so far. Every decode step reads their KV cache, 128 KiB per token; in this lab they stay the same while the prompt is prefilled.','每个用户的对话到目前为止的 token。每个 decode 步都要读它们的 KV cache，每个 token 128 KiB；在这个 Lab 里，prompt prefill 期间它们保持不变。')))+`</div>`+
   `<div class="lockable">`+wrap(rng('ck-ctl-N','N',tx('The long prompt','长 prompt'),N_STOPS,S.N,v=>tx(`${tokK(v)} tokens`,`${tokK(v)} 个 token`)),note(tx(`The new request’s prompt, prefilled while the users decode. The longest here is 120K tokens: ${MODEL} reads at most 128K, prompt and answer together.`,`新请求的 prompt，在用户 decode 的同时 prefill。这里最长 120K 个 token：${MODEL} 最多读 128K 个，prompt 和回答合在一起算。`)))+`</div>`+
   `<div class="lockable" id="ck-ctl-C-wrap">`+wrap(cCtl('ck-ctl-C',false),note(tx('Prompt tokens prefilled per step. “None” prefills the whole prompt in one step. Real servers set a token budget per step instead, which counts the decode tokens too (vLLM’s max_num_batched_tokens).','每步 prefill 的 prompt token 数。“不切”就是整个 prompt 一步 prefill 完。真实服务设的是每步的 token 预算，decode 的 token 也算在内（vLLM 的 max_num_batched_tokens）。')))+`</div>`+
   `<div class="ctl-note" id="ck-Clock" hidden>${tx('The chunk size stays “none” in parts 1 to 3; you pick it in part 4.','第 1 到 3 部分里，chunk 大小固定为“不切”；到第 4 部分再选。')}</div>`+
   `<div class="ck-grp lockable" id="ck-ctl-eff"><div class="ck-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('Arithmetic: 50% of the peak FLOP/s','计算：峰值 FLOP/s 的 50%')}</span><div class="ctl-note">${EFF_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Memory: 60% of the peak bandwidth','读写显存：峰值带宽的 60%')}</span><div class="ctl-note">${BW_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${A.t('cUtil')}${tx(': ','：')}${pct0(UTIL)}</span><div class="ctl-note">${A.t('utilNote')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx(`Runtime overhead: ${OH} GiB`,`运行时开销：${OH} GiB`)}</span><div class="ctl-note">${A.t('ohNote')}</div></div></div>`+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`# One engine step under a token budget (chunked prefill):
# decode tokens first, then as much of the waiting prompt as fits.
L, D, KVH, HD, FFN, VOCAB = 32, 4096, 8, 128, 14_336, 128_256
P_LAYERS = L*(2*D*D + 2*D*KVH*HD + 3*D*FFN + 2*D) + D
P_HEAD = VOCAB*D                  # LM head: once per new token
W_READ = 2*(P_LAYERS + P_HEAD)    # bytes of weights read per step
KV_TOK = 2*L*KVH*HD*2             # 131,072 bytes per token
FLOPS = 1979e12/2 * 0.5           # 50% of the H100's dense BF16 peak
BW = 3.35e12 * 0.6                # 60% of its peak bandwidth

def attn(a, b):   # QK^T and AV FLOPs for query positions a..b-1
    return 2*L*D*(b*(b + 1) - a*(a + 1))

def step(users, ctx, a, b, last):  # users decode; prompt tokens a..b-1
    flops = (2*P_LAYERS*(users + b - a) + 2*P_HEAD*(users + last)
             + users*attn(ctx, ctx + 1) + attn(a, b))
    nbytes = W_READ + KV_TOK*(users*ctx + b)
    return max(flops/FLOPS, nbytes/BW)       # the slower one

def serve(users, ctx, prompt, budget):
    done, waits = 0, []           # the prompt's first token: sum(waits)
    while done < prompt:
        end = min(prompt, done + budget - users)  # decodes go first
        waits.append(step(users, ctx, done, end, end == prompt))
        done = end
    return waits

for budget in (32_768 + 16, 8_192, 4_096 + 16):  # 8,192: vLLM on an H100
    waits = serve(16, 4096, 32_768, budget)
    print(f"budget {budget}: steps {len(waits)},"
          f" users wait up to {max(waits)*1e3:.1f} ms")
# budget 32784: steps 1, users wait up to 1494.0 ms
# budget 8192: steps 5, users wait up to 479.2 ms
# budget 4112: steps 8, users wait up to 249.5 ms`;
function connectHTML(){
  const v8=run({...BASE,C:8192-16});
  return `<section class="wrap connect" aria-labelledby="ck-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ck-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A scheduler step with a token budget: the decode tokens go first, and the prompt takes what is left, one chunk per step. The step time is this lab’s model.','带 token 预算的一步调度：decode 的 token 先放进去，prompt 用剩下的预算，每步一个 chunk。每步的耗时用的是这个 Lab 的模型。')}</p><pre class="code" id="ck-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ck-prod">`+
   tx(`<li>vLLM has no separate prefill and decode phases: in ${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')} each step starts with a token budget, schedules the running requests first and then the waiting ones, and cuts each request’s new tokens to what is left of the budget, so a long prompt is prefilled in chunks.</li>`+
      `<li>The budget is <code>max_num_batched_tokens</code>, and <code>enable_chunked_prefill</code> is on by default (${cite(SRC.vcfg,'<code>vllm/config/scheduler.py</code>')}). <code>EngineArgs.get_batch_defaults</code> in ${cite(SRC.vargs,'<code>vllm/engine/arg_utils.py</code>')} sets it to 8,192 tokens for the API server on an H100 or H200 (16,384 for offline use). With 16 users decoding that leaves chunks of 8,176 tokens: in this lab’s mission the users would wait up to ${fT(v8.longest)}.</li>`+
      `<li>SGLang caps each chunk with <code>--chunked-prefill-size</code> (${cite(SRC.sglargs,'<code>python/sglang/srt/arg_groups/fields/schedule.py</code>')}). The decoding requests join a chunk’s batch only with <code>--enable-mixed-chunk</code>, off by default: then <code>PrefillAdder</code> in ${cite(SRC.sglpol,'<code>schedule_policy.py</code>')} subtracts their tokens from the chunk, and the scheduler in ${cite(SRC.sglsch,'<code>scheduler.py</code>')} mixes them in; otherwise it runs a pending prefill batch before a decode batch.</li>`,
      `<li>vLLM 不分 prefill 和 decode 两个阶段：在 ${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')} 里，每一步从一个 token 预算开始，先调度在跑的请求，再调度排队的请求，每个请求的新 token 都截到预算剩下的量，所以长 prompt 会被分成 chunk 来 prefill。</li>`+
      `<li>这个预算就是 <code>max_num_batched_tokens</code>，<code>enable_chunked_prefill</code> 默认打开（${cite(SRC.vcfg,'<code>vllm/config/scheduler.py</code>')}）。${cite(SRC.vargs,'<code>vllm/engine/arg_utils.py</code>')} 里的 <code>EngineArgs.get_batch_defaults</code> 把它设成：H100 或 H200 上的 API 服务 8,192 个 token（离线使用时 16,384 个）。16 个用户在 decode 时，chunk 就是 8,176 个 token：在这个 Lab 的任务里，用户最多要等 ${fT(v8.longest)}。</li>`+
      `<li>SGLang 用 <code>--chunked-prefill-size</code> 限制每个 chunk 的大小（${cite(SRC.sglargs,'<code>python/sglang/srt/arg_groups/fields/schedule.py</code>')}）。只有打开 <code>--enable-mixed-chunk</code>（默认关闭），正在 decode 的请求才会和 chunk 放进同一个 batch：这时 ${cite(SRC.sglpol,'<code>schedule_policy.py</code>')} 里的 <code>PrefillAdder</code> 从 chunk 里扣掉它们的 token，${cite(SRC.sglsch,'<code>scheduler.py</code>')} 里的调度器把它们混进来；不打开时，有待跑的 prefill batch 就先跑它，再跑 decode batch。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.sarathi,'SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills')}<span>${tx('Agrawal et al., 2023: a prefill split into equal-sized chunks, one per batch, with the rest of the batch filled with decodes that ride along cheaply','Agrawal 等，2023：把 prefill 切成等长的 chunk，每个 batch 放一个，其余位置放 decode，让 decode 低成本地搭车')}</span></li>`+
   `<li>${cite(SRC.serve,'Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve')}<span>${tx('Agrawal et al., OSDI 2024: chunked prefills and stall-free schedules that add new requests to a batch without pausing ongoing decodes','Agrawal 等，OSDI 2024：chunked prefill 加上不卡顿的调度，新请求加入 batch 时不暂停正在进行的 decode')}</span></li>`+
   `<li>${cite(SRC.fastgen,'DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference')}<span>${tx('Holmes et al., 2024: Dynamic SplitFuse cuts long prompts into chunks and fills forward passes of a set size with prompt and generation tokens','Holmes 等，2024：Dynamic SplitFuse 把长 prompt 切成 chunk，用 prompt 和生成的 token 填满固定大小的前向')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ck-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ck-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ck-app m2" id="ck-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ck-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how long the users wait at most once the prompt is cut into chunks.','选一个场景，看看把 prompt 切成 chunk 后，用户最长要等多久。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   cCtl('ck-sb-C',true)+
   `<div class="obs" id="ck-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${DEEP()}${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`One ${GPUN}: keep 32 users within 50 ms per token while a 16K-token document is prefilled`,`一张 ${GPUN}：prefill 一份 16K 个 token 的文档时，让 32 个用户每个 token 都不超过 50 ms`)}</h3>`+
   `<p>${tx(`32 users decode at 2K (2,048) tokens of context each on one ${GPUN} with ${MODEL}, and a 16K-token document arrives. Every user must keep getting a token at least every 50 ms. Choose only the chunk size: give the document its first token as early as you can.`,
            `一张 ${GPUN} 跑 ${MODEL}，32 个用户在 decode，每人 2K（2,048）个 token 的上下文，这时来了一份 16K 个 token 的文档。每个用户至少每 50 ms 要拿到一个 token。只改 chunk 大小：让文档的首 token 尽量早。`)}</p>`+
   cCtl('ck-ic-chC',true,'C')+
   `<div class="goal mono" id="ck-ch-goal"><span>${tx('Longest wait now','现在最长的等待')}</span><span><b id="ck-ch-now"></b>${tx(' / at most 50 ms',' / 最多 50 ms')}</span></div><div class="progress" id="ck-ch-prog" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono ck-ch-ttft"><span>${tx('The document’s first token now','现在文档的首 token')}</span><span><b id="ck-ch-ttft"></b></span></div>`+
   submit+`<div id="ck-ch-res" class="ck-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function renderRail(){const rail=$('#ck-rail',app); if(!rail) return; rail.innerHTML=railFull(); A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#ck-rail',app); if(!rail) return;
  $$('[data-act="deep"]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(deepOpen));});
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step]);
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);
          const dw=$('details.ck-why',box); if(dw){dw.open=whyOpen; dw.addEventListener('toggle',()=>{whyOpen=dw.open;});}}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ck-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the four scenarios above.','从上面四个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const p=chPoint(S.C), now=$('#ck-ch-now',rail); if(!now) return;
    now.textContent=fT(p.wait); $('#ck-ch-ttft',rail).textContent=fT(p.ttft);
    $('#ck-ch-prog i',rail).style.width=Math.min(100,p.wait/CH.goal*100).toFixed(1)+'%';
    const g=$('#ck-ch-goal',rail), pr=$('#ck-ch-prog',rail);
    g.classList.toggle('met',p.ok); pr.classList.toggle('met',p.ok); g.classList.toggle('ck-over',!p.ok); pr.classList.toggle('ck-over',!p.ok);
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#ck-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);
      /* a bar past the scale keeps its value inside, at the track's end; every other label the kit places, on one line and clear of
         the goal line (kit polish K1: this lab's own “inside a bar longer than 62%” put the goal line through 24.24 ms and 47.16 ms) */
      $$('.cmp-row',res).forEach(rw=>{const v=rw.querySelector('.cmp-val'); if(v&&/▸/.test(v.textContent)) rw.classList.add('ck-off');});}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
function statusOf(){
  const r=R();
  if(mode==='challenge'){const ok=r.longest<=CH.goal;
    return {cls:ok?'ok':'bad',txt:ok?tx(`Within 50 ms: ${fT(r.longest)}`,`在 50 ms 以内：${fT(r.longest)}`):tx(`Over 50 ms: ${fT(r.longest)}`,`超过 50 ms：${fT(r.longest)}`),short:ok?tx('Within 50 ms','50 ms 以内'):tx('Over 50 ms','超过 50 ms'),
      tip:tx(`Green when every user’s longest wait between two tokens is at most 50 ms while the ${tokK(CH.N)}-token document is prefilled; now it is ${fT(r.longest)}.`,`prefill ${tokK(CH.N)} 个 token 的文档期间，每个用户两个 token 之间最长的等待不超过 50 ms，徽章就是绿色；现在是 ${fT(r.longest)}。`)};}
  const q=tMs(r.longest)/tMs(r.normal.t);
  return {cls:'neutral',txt:tx(`Longest wait: ${ratioS(q)} times a usual step`,`最长的等待是平时一步的 ${ratioS(q)} 倍`),short:tx(`Wait ${fT(r.longest)}`,`最长等 ${fT(r.longest)}`),
    tip:tx(`How many times as long the users’ longest wait between two tokens is as a usual step, while the prompt is prefilled: ${fT(r.longest)} ÷ ${fT(r.normal.t)} = ${ratioS(q)}.`,`prompt prefill 期间，用户两个 token 之间最长的等待是平时一步的几倍：${fT(r.longest)} ÷ ${fT(r.normal.t)} = ${ratioS(q)}。`)};
}
function drawHead(){
  const r=R(), t=$('#ck-st-title',app), sub=$('#ck-st-sub',app), st=$('#ck-status',app);
  t.textContent=tx(`${MODEL} on one ${GPUN}, BF16`,`一张 ${GPUN} 跑 ${MODEL}，BF16`);
  sub.textContent=tx(`${userN(r.U)} decoding, ${tokK(r.c)} tokens each; ${aTok(r.N)}-token prompt ${r.n>1?`in ${tokK(r.chunk)}-token chunks`:(r.C?`in one chunk`:'in one step')}`,
                     `${userN(r.U)}在 decode，每人 ${tokK(r.c)} 个 token；${tokK(r.N)} 个 token 的 prompt ${r.n>1?`切成 ${tokK(r.chunk)} 个 token 的 chunk`:(r.C?'一个 chunk 就装下':'一步 prefill 完')}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const legIt=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
/* the timeline's marks, above the chart */
function legendHTML(){
  const live=play.t!=null, r=R(), E=edgeOf(r);
  return legIt('ck-sw-tick',tx('thin tick: every user gets a token','细线：每个用户各拿到一个 token'),tx('Each thin tick in the users’ row is the end of a step: every user who is decoding gets one more token there. The gaps between the ticks are the users’ waits.','用户那一行的每条细线是一步的结束：每个正在 decode 的用户在那里各多拿到一个 token。细线之间的空白就是用户的等待。'))+
    (!live&&E.longOff?'':legIt('ck-sw-brk',live?tx('shaded: waiting now','阴影：正在等'):tx('bracket: the longest wait','括号：最长的等待'),live?tx('The shaded span is how long the users have waited since their last token, up to the moment shown.','阴影是用户从上一个 token 到图上这一刻已经等了多久。'):tx('The bracket and the shading in the users’ row mark their longest wait between two tokens while the prompt is prefilled.','用户那一行的括号和阴影，标出 prompt prefill 期间他们两个 token 之间最长的等待。')))+
    legIt('ck-sw-seg',tx('bars: prefill steps','横条：prefill 的各步'),tx('In the prompt’s row, each bar is one step of its prefill: the whole prompt, or one chunk. Neighbouring bars alternate in shade so you can tell the steps apart.','prompt 那一行里，每一段横条是它 prefill 的一步：整个 prompt，或者一个 chunk。相邻的横条深浅交替，好区分每一步。'))+
    (bandOn(r)?legIt('ck-sw-band',tx('dashed box: one step','虚线框：一步'),tx('The dashed box spans both rows: the users’ wait and the chunk inside it are one and the same step, computed together in one pass of the model. The two bars under “Why” show that step.','虚线框跨过两行：用户的这段等待和框里的那个 chunk 是同一步，模型一起算一遍。下面“原因”里的两根条画的就是这一步。')):'')+
    (E.off?'':legIt('ck-sw-first',tx('thick line: its first token','粗线：它的首 token'),tx('The thick line in the prompt’s row is its first token: it comes when the last chunk’s step ends. The thin ticks after it are its next tokens, decoded with the users.','prompt 那一行的粗线是它的首 token：最后一个 chunk 的那一步结束时出来。后面的细线是它和用户一起 decode 出的 token。')))+
    (live?legIt('ck-sw-head',tx('line with a triangle: now','带三角的竖线：当前时刻'),tx('The line with a triangle is the moment the replay has reached; the chart shows only what has happened by then.','带三角的竖线是回放走到的时刻；图上只画到这一刻为止发生的事。')):'');
}
/* the two limits' bars, inside their panel */
function limLegendHTML(){
  const ch=mode==='challenge';
  return legIt('ck-sw-ar',tx('the users’ tokens','用户的 token'),tx('The solid gray part of an arithmetic bar is the users’ decode tokens: their FLOPs divided by the FLOP/s the GPU reaches.','计算条的实心灰色部分是用户的 decode token：它们的 FLOP 数除以 GPU 能达到的 FLOP/s。'))+
    legIt('ck-sw-arc',tx('hatched: the prompt’s share','斜纹：prompt 的部分'),tx('The hatched part of an arithmetic bar is the prompt’s share of the step: the chunk’s tokens through the weights and their attention to every earlier prompt token.','计算条的斜纹部分是这一步里 prompt 的计算：chunk 的 token 和权重相乘，以及它们对前面每个 prompt token 的 attention。'))+
    legIt('w',tx('weights read','读权重'),tx(`The blue part of a memory bar is the time to read the weights: ${gbS(W_READ)} once per step, however many tokens the step computes.`,`读写显存条的蓝色部分是读权重的时间：每一步读一遍 ${gbS(W_READ)}，不管这一步算多少个 token。`))+
    legIt('kv','KV cache',tx('The amber part of a memory bar is the KV cache’s time: every user’s context and the prompt’s tokens so far, 128 KiB per token.','读写显存条的琥珀色部分是 KV cache 的时间：每个用户的上下文，加上 prompt 到这一步为止的 token，每个 token 128 KiB。'))+
    (ch?legIt('lim',tx('dashed line: 50 ms goal','虚线：目标 50 ms'),tx('The dashed line is the goal: every user gets a token at least every 50 ms, so no step while the document is prefilled may be longer.','虚线是目标：每个用户至少每 50 ms 拿到一个 token，所以文档 prefill 期间，每一步都不能比它长。'))
       :legIt('ck-sw-fade',tx('faded: does not set the time','半透明：不决定耗时'),tx('In each step the longer bar sets the step’s time; the shorter one is drawn faded, because the GPU does it at the same time.','每一步里，较长的那条决定这一步的耗时；较短的那条画成半透明，因为 GPU 同时在做它。')));
}
/* nice axis steps, in ms */
function niceStep(span,n){const raw=span/(n||5), p=Math.pow(10,Math.floor(Math.log10(raw))), m=raw/p; return (m<1.5?1:(m<3.5?2:(m<7.5?5:10)))*p;}
let tlKey='';
function drawTL(){
  const tl=$('#ck-tl',app); if(!tl) return;
  const r=R(), sp=spanOf(S), x0=sp.x0, x1=sp.x1, W=x1-x0, X=t=>(t-x0)/W*100, P=t=>Math.max(0,Math.min(100,X(t))).toFixed(3)+'%', E=edgeOf(r);
  const key=[r.U,r.c,r.N,r.C,A.lang(),mode].join('|');
  if(key!==tlKey){tlKey=key;
    const tu=[], tp=[], pxw=($('#ck-tu',app).clientWidth||500)/(W*1e3), dn=dt=>dt*1e3*pxw<2.6?' d':'';   // px per ms
    for(let t=0;t>=x0-1e-12;t-=r.normal.t) tu.push(`<i class="ck-tick${dn(r.normal.t)}" style="left:${P(t)}"></i>`);
    r.steps.forEach((s,i)=>{if(s.start>=x1-1e-12) return;       // past the right edge (chunks too small): the label there says when the first token comes
      if(s.end<=x1+1e-12) tu.push(`<i class="ck-tick${dn(s.t)}" style="left:${P(s.end)}"></i>`);
      const tip=r.n<=64?` data-tipx="${esc(tx(`Step ${fint(s.k+1)} of ${fint(r.n)}: prompt tokens ${fint(s.a+1)} to ${fint(s.b)} next to the users’ ${fint(r.U)} tokens, ${fT(s.t)}, set by ${s.cb?'arithmetic':'memory traffic'}.`,`第 ${fint(s.k+1)} 步（共 ${fint(r.n)} 步）：prompt 的第 ${fint(s.a+1)} 到 ${fint(s.b)} 个 token，加上用户的 ${fint(r.U)} 个 token，${fT(s.t)}，由${s.cb?'计算':'读写显存'}决定。`))}"`:'';
      tp.push(`<i class="ck-seg${i%2?' odd':''}" style="left:${P(s.start)};width:${(X(Math.min(s.end,x1))-X(s.start)).toFixed(3)}%"${tip}></i>`);});
    for(let t=r.ttft+r.after.t;t<=x1+1e-12;t+=r.after.t){tu.push(`<i class="ck-tick${dn(r.after.t)}" style="left:${P(t)}"></i>`); tp.push(`<i class="ck-ptick${dn(r.after.t)}" style="left:${P(t)}"></i>`);}
    if(!E.off) tp.push(`<i class="ck-first" style="left:${P(r.ttft)}"></i>`);
    if(r.n===1) tp.push(`<span class="ck-pin" style="left:${P(0)};width:${(X(r.ttft)-X(0)).toFixed(3)}%">${esc(tx(`${fint(r.N)} prompt tokens, one step`,`${fint(r.N)} 个 prompt token，一步`))}</span>`);
    $('#ck-mku',app).innerHTML=tu.join(''); $('#ck-mkp',app).innerHTML=tp.join(''); $('#ck-tp',app).classList.toggle('ck-offr',E.off);
    /* the label inside the one prefill bar: the long one if it fits, else the prompt length, else none */
    const pin=$('.ck-pin',app); if(pin&&pin.clientWidth&&pin.scrollWidth>pin.clientWidth){pin.textContent=tokK(r.N); if(pin.scrollWidth>pin.clientWidth) pin.textContent='';}
    const tw=$('#ck-tu',app).clientWidth||500, spanMs=W*1e3, st=niceStep(Math.max(1,x1*1e3),Math.max(2,Math.min(5,Math.floor(tw/80)))), inS=x1*1e3>=2000, lab=v=>inS?fixd(v/1000,st>=1000?0:1):fint(v);
    const vs=[]; for(let v=0;v<=x1*1e3*0.97;v+=st) vs.push(v);
    $('#ck-axis',app).innerHTML=vs.map((v,i)=>`<span style="left:${(X(v/1e3)).toFixed(3)}%">${lab(v)}${i===vs.length-1?NS+(inS?'s':'ms'):''}</span>`).join('');
    void spanMs;
  }
  /* labels and tooltips (cheap; they follow the language and the state) */
  $('#ck-tl-cap',app).textContent=tx(`What the users see while the ${tokK(r.N)}-token prompt is prefilled`,`${tokK(r.N)} 个 token 的 prompt prefill 期间，用户看到的情况`);
  $('#ck-lh-u',app).textContent=tx(`${userN(r.U)} decoding`,`${userN(r.U)}在 decode`);
  $('#ck-lh-u2',app).textContent=tx(`a token every ${fT(r.normal.t)} before the prompt`,`prompt 来之前每 ${fT(r.normal.t)} 一个 token`);
  $('#ck-lh-p',app).textContent=tx(`${tokK(r.N)}-token prompt`,`${tokK(r.N)} 个 token 的 prompt`);
  $('#ck-lh-p2',app).textContent=r.n>1?tx(`${fint(r.n)} chunks of ${tokK(r.chunk)} tokens`,`${fint(r.n)} 个 chunk，每个 ${tokK(r.chunk)} 个 token`):(r.C?tx('one chunk holds the whole prompt','一个 chunk 就装下整个 prompt'):tx('one step, no chunking','一步算完，不切 chunk'));
  const arr=$('.ck-arr-a',tl), ft=$('.ck-arr-f',tl);
  arr.style.left=P(0); arr.textContent=tx('▾ prompt arrives','▾ prompt 到达');
  ft.style.left=P(r.ttft); ft.classList.toggle('off',E.off);
  ft.textContent=E.off?tx(`first token at ${fT(r.ttft)}, off the chart ▸`,`${fT(r.ttft)} 才出首 token，在图外 ▸`):tx(`first token at ${fT(r.ttft)} ▾`,`${fT(r.ttft)} 出首 token ▾`);
  arr.hidden=false; {const a=arr.getBoundingClientRect(), f=ft.getBoundingClientRect(); if(a.width&&f.width&&a.right+8>f.left) arr.hidden=true;}   // measured after the transforms
  const tu=$('#ck-tu',app), tp=$('#ck-tp',app);
  tu.dataset.tipx=tx(`The ${userN(r.U)}: each thin tick is the end of a step, where every user gets one more token. Before the prompt arrives the ticks are ${fT(r.normal.t)} apart; while it is prefilled the gaps are the prefill steps, the longest ${fT(r.longest)}; after its first token, ${fT(r.after.t)}, because each step now also reads the prompt’s KV cache.`,
                     `${userN(r.U)}：每条细线是一步的结束，每个用户在那里多拿到一个 token。prompt 来之前，细线之间隔 ${fT(r.normal.t)}；prefill 期间，间隔就是 prefill 的各步，最长 ${fT(r.longest)}；prompt 出首 token 之后是 ${fT(r.after.t)}，因为每一步还要多读 prompt 的 KV cache。`);
  const tEnd=E.off?tx(`, past the right edge of the chart`,`，已经在图的右边之外`):tx(' (the thick line)','（粗线）');
  tp.dataset.tipx=r.n>1?(tooSmall(r)?tx(`The ${tokK(r.N)}-token prompt: ${fint(r.n)} chunks of ${fint(r.chunk)} tokens, one per step (shades alternate). Every step takes as long as its memory reads, so the steps are almost the same length. Its first token comes when the last chunk’s step ends, at ${fT(r.ttft)}${tEnd}.`,
                                         `${tokK(r.N)} 个 token 的 prompt：${fint(r.n)} 个 chunk，每个 ${fint(r.chunk)} 个 token，每步一个（深浅交替）。每一步的耗时都由读显存决定，所以各步几乎一样长。最后一个 chunk 的那一步结束时，首 token 在 ${fT(r.ttft)} 出来${tEnd}。`)
    :tx(`The ${tokK(r.N)}-token prompt: ${fint(r.n)} chunks of ${fint(r.chunk)} tokens, one per step (shades alternate); the later a chunk, the longer its step, since its tokens attend to more earlier prompt tokens. Its first token comes when the last chunk’s step ends, at ${fT(r.ttft)}${tEnd}.`,
        `${tokK(r.N)} 个 token 的 prompt：${fint(r.n)} 个 chunk，每个 ${fint(r.chunk)} 个 token，每步一个（深浅交替）；chunk 越靠后，它那一步越长，因为它的 token 要对更多前面的 prompt token 做 attention。最后一个 chunk 的那一步结束时，首 token 在 ${fT(r.ttft)} 出来${tEnd}。`))
    :tx(`The ${tokK(r.N)}-token prompt, prefilled in one step of ${fT(r.ttft)}; its first token comes when that step ends (the thick line).`,`${tokK(r.N)} 个 token 的 prompt 一步 prefill 完，要 ${fT(r.ttft)}；这一步结束时出首 token（粗线）。`);
  tu.setAttribute('aria-label',tu.dataset.tipx); tp.setAttribute('aria-label',tp.dataset.tipx);
  /* the longest wait's bracket */
  const lg=$('#ck-lg',app), ls=r.steps[r.li], ch=mode==='challenge';
  lg.style.left=P(ls.start); lg.style.width=Math.max(0.15,X(ls.end)-X(ls.start)).toFixed(3)+'%';
  lg.classList.toggle('ck-bad',ch&&r.longest>CH.goal); lg.classList.toggle('ck-good',ch&&r.longest<=CH.goal);
  const em=$('em',lg); em.textContent=tx(`longest wait: ${fT(r.longest)}`,`最长的等待：${fT(r.longest)}`);
  fitLabel(em,lg);
  const lf=$('#ck-tu .ck-lfill',app); if(lf){lf.style.left=P(ls.start); lf.style.width=Math.max(0.15,X(ls.end)-X(ls.start)).toFixed(3)+'%';}
  const an=$('#ck-anat',app), ah=anatHTML(r); if(an&&an._h!==ah){an.innerHTML=ah; an._h=ah;}
  drawPlay(); drawBand();
}
/* what one step holds, above the timeline (Batch 3 review: the product owner read “each step prefills 2,048 tokens besides the users’
   decode tokens” as decode, then prefill, taking turns). It is one pass: the users' next tokens and one chunk, computed together */
function anatHTML(r){
  const U=r.U, n=r.n, sq=`<span class="ck-an-sq" aria-hidden="true">${'<i></i>'.repeat(Math.min(U,16))}</span>`;
  const uTip=tx(`Each of the ${userN(U)} puts its one next token into the step.`,`${userN(U)}各把自己的下一个 token 放进这一步。`);
  const cTip=n>1?tx(`One chunk of the prompt: ${fint(r.chunk)} of its ${fint(r.N)} tokens.`,`prompt 的一个 chunk：它 ${fint(r.N)} 个 token 里的 ${fint(r.chunk)} 个。`):tx(`The whole prompt, all ${fint(r.N)} tokens.`,`整个 prompt，全部 ${fint(r.N)} 个 token。`);
  const pTip=tx('The model runs once over everything in the step: it reads the weights once, and every request in the step gets its result when the pass ends. The users do not take turns with the prompt.','模型把这一步里的所有东西一起跑一遍：权重只读一次，这一步结束时，里面的每个请求都拿到结果。用户和 prompt 不是轮流来的。');
  return `<span class="ck-an-h">${n>1?tx('Each step','每一步'):tx('The one step','这一步')}</span>`+
    `<span class="ck-an-it" tabindex="0" data-tipx="${esc(uTip)}">${sq}<em>${tx(`${userN(U)} × 1 token`,`${userN(U)}各 1 个 token`)}</em></span><span class="ck-an-op" aria-hidden="true">+</span>`+
    `<span class="ck-an-it" tabindex="0" data-tipx="${esc(cTip)}"><span class="ck-an-ch" aria-hidden="true"></span><em>${n>1?tx(`one ${tokK(r.chunk)} chunk`,`一个 ${tokK(r.chunk)} 的 chunk`):tx(`the whole ${tokK(r.N)} prompt`,`整个 ${tokK(r.N)} 的 prompt`)}</em></span>`+
    `<span class="ck-an-op" aria-hidden="true">→</span><span class="ck-an-it ck-an-pass" tabindex="0" data-tipx="${esc(pTip)}"><em>${tx('one pass','模型一起算一遍')}</em></span>`;
}
/* the dashed box across both rows: the users' wait and the chunk in it are one step. It marks the step the “Why” bars show */
function drawBand(){
  const band=$('#ck-band',app); if(!band) return; const r=R();
  if(!bandOn(r)){band.hidden=true; return;}
  const s=boxStep(r).s, sp=spanOf(S), X=t=>(Math.min(t,sp.x1)-sp.x0)/(sp.x1-sp.x0), tu=$('#ck-tu',app), tp=$('#ck-tp',app), w=tu.clientWidth;
  band.hidden=false;
  band.style.left=(tu.offsetLeft+X(s.start)*w-2).toFixed(1)+'px'; band.style.width=Math.max(5,(X(s.end)-X(s.start))*w+4).toFixed(1)+'px';
  band.style.top=(tu.offsetTop-3)+'px'; band.style.height=(tp.offsetTop+tp.offsetHeight-tu.offsetTop+6)+'px';
}
/* a label centered on its bracket, kept inside the chart */
function fitLabel(em,box){
  em.style.left='50%'; em.style.transform='translateX(-50%)';
  const tw=box.parentNode.clientWidth, bl=box.offsetLeft, bw=box.offsetWidth, w=em.offsetWidth; if(!tw||!w) return;
  const c=bl+bw/2; let left=c-w/2; left=Math.max(0,Math.min(tw-w,left));
  em.style.left=(left-bl)+'px'; em.style.transform='none';
}
/* the moment shown: the whole run (t = null), or the replay's playhead */
const runEnd=()=>spanOf(S).x1;
function stepAt(r,t){for(let k=0;k<r.steps.length;k++){if(t>=r.steps[k].start-1e-12&&t<r.steps[k].end-1e-12) return k;} return -1;}
function lastTokenBefore(r,t){ // the users' last token at or before t: before the prompt, the ticks are at 0, −T, −2T …
  if(t<0) return -Math.ceil(-t/r.normal.t-1e-9)*r.normal.t;
  if(t<r.ttft){const k=stepAt(r,t); return k>0?r.steps[k-1].end:0;}
  const m=Math.floor((t-r.ttft)/r.after.t+1e-9); return r.ttft+m*r.after.t;}
function drawPlay(){
  if(!app) return; const r=R(), sp=spanOf(S), x0=sp.x0, W=sp.x1-x0, X=t=>(t-x0)/W*100;
  const t=play.t, live=t!=null, cut=live?Math.max(0,Math.min(100,X(t))):100;
  ['#ck-mku','#ck-mkp'].forEach(s=>{const e=$(s,app); if(e) e.style.clipPath=live?`inset(-12px ${(100-cut).toFixed(3)}% -12px 0)`:'';});
  $$('.ck-head',app).forEach(h=>{h.hidden=!live; if(live) h.style.left=cut.toFixed(3)+'%';});
  const af=$('.ck-arr-f',app); if(af) af.style.visibility=live&&t<r.ttft-1e-12?'hidden':'';
  const lg=$('#ck-lg',app), wt=$('#ck-wt',app), wf=$('#ck-tu .ck-wfill',app), lf=$('#ck-tu .ck-lfill',app);
  const lo=edgeOf(r).longOff; if(lg) lg.hidden=live||lo; if(lf) lf.hidden=live||lo;   // the longest step past the right edge: no bracket
  if(wt){wt.hidden=!live||t<0; if(wf) wf.hidden=wt.hidden;
    if(live&&t>=0){const lt=lastTokenBefore(r,t), a=X(lt), w=Math.max(0.15,X(t)-a); wt.style.left=a.toFixed(3)+'%'; wt.style.width=w.toFixed(3)+'%'; if(wf){wf.style.left=a.toFixed(3)+'%'; wf.style.width=w.toFixed(3)+'%';}
      const em=$('em',wt); em.textContent=tx(`waiting ${fClock(t-lt)}`,`已等 ${fClock(t-lt)}`); fitLabel(em,wt);}}
  const sc=$('#ck-scrub',app); if(sc){sc.value=String(live?Math.round((t-x0)/W*1000):1000); A.ui.setRange(sc);}
  $$('[data-act="play"]',app).forEach(b=>{b.textContent=playLabel();});
  const clk=$('#ck-clock',app); if(clk) clk.textContent=clockText(r,t);
  const lgd=$('#ck-legend',app); if(lgd){const h=legendHTML(); if(lgd._h!==h){lgd.innerHTML=h; lgd._h=h;}}
  const k=live?stepAt(r,t):-1; if(k!==play.k){play.k=k; drawLim(); drawBand();}
  const dk=(live?fClock(t):'end')+'|'+mode; if(dk!==play.dk){play.dk=dk; A.dock.render();}
}
function clockText(r,t){
  if(t==null) return (r.n>1?tx(`The whole run: ${fint(r.n)} prefill steps, the longest ${fT(r.longest)}.`,`整次运行：prefill 一共 ${fint(r.n)} 步，最长的一步 ${fT(r.longest)}。`)
    :tx(`The whole run: one prefill step of ${fT(r.longest)}.`,`整次运行：prefill 只有一步，${fT(r.longest)}。`))
    +(edgeOf(r).off?tx(` The prompt’s first token comes at ${fT(r.ttft)}, past the chart’s right edge.`,`prompt 的首 token 要到 ${fT(r.ttft)} 才出来，在图的右边之外。`):'');
  if(t<0) return tx(`Before the prompt arrives, the ${userN(r.U)} get a token every ${fT(r.normal.t)}.`,`prompt 到达之前，${userN(r.U)}每 ${fT(r.normal.t)} 拿到一个 token。`);
  if(t<r.ttft-1e-12){const k=stepAt(r,t), s=r.steps[Math.max(0,k)], lt=lastTokenBefore(r,t);
    return tx(`At ${fClock(t)}: step ${fint(s.k+1)} of ${fint(r.n)}. The users have waited ${fClock(t-lt)} since their last token.`,
              `${fClock(t)}：第 ${fint(s.k+1)} 步（共 ${fint(r.n)} 步）。用户从上一个 token 起已经等了 ${fClock(t-lt)}。`);}
  return tx(`At ${fClock(t)}: the prompt got its first token at ${fT(r.ttft)} and now decodes with the users, a token every ${fT(r.after.t)}.`,`${fClock(t)}：prompt 在 ${fT(r.ttft)} 拿到了首 token，现在和用户一起 decode，每 ${fT(r.after.t)} 一个 token。`);
}
/* the two limits of a step: a usual one (the users alone) and the step being looked at */
function drawPhase(ph,o,opt){
  const box=$(`.ck-ph[data-ph="${ph}"]`,app); if(!box) return null;
  const max=Math.max(o.tc,o.tm,opt.goal?opt.goal*1.12:0), P=v=>Math.max(0,Math.min(100,v/max*100));
  const cRow=$('.ck-row[data-k="c"]',box), mRow=$('.ck-row[data-k="m"]',box), cb=o.tc>=o.tm;
  const bu=$('.ck-bu',cRow), bch=$('.ck-bch',cRow), uPart=o.tcD, pPart=o.tc-o.tcD;
  bu.style.width=P(uPart).toFixed(3)+'%'; bch.style.left=P(uPart).toFixed(3)+'%'; bch.style.width=(P(o.tc)-P(uPart)).toFixed(3)+'%'; bch.hidden=pPart<=0;
  $('.ck-bw',mRow).style.width=P(o.tw).toFixed(3)+'%'; const bk=$('.ck-bk',mRow); bk.style.left=P(o.tw).toFixed(3)+'%'; bk.style.width=(P(o.tm)-P(o.tw)).toFixed(3)+'%';
  cRow.classList.toggle('ck-set',cb); mRow.classList.toggle('ck-set',!cb);
  cRow.classList.toggle('ck-fade',!cb&&!opt.goal); mRow.classList.toggle('ck-fade',cb&&!opt.goal);
  $('.ck-rv',cRow).textContent=fT(o.tc); $('.ck-rv',mRow).textContent=fT(o.tm);
  [[cRow,cb],[mRow,!cb]].forEach(([rw,on])=>{const tg=$('.ck-settag',rw); tg.hidden=!on||!!opt.goal; tg.textContent=tx('sets the time','决定耗时');});
  $$('.ck-goal',box).forEach(g=>{g.hidden=!opt.goal; if(opt.goal) g.style.left=P(opt.goal).toFixed(3)+'%';});
  $('.ck-ph-h b',box).textContent=opt.title; $('.ck-ph-d',box).textContent=opt.desc; $('.ck-ph-t',box).textContent=fT(o.t)+(opt.goal?tx(' (goal: at most 50 ms)','（目标：最多 50 ms）'):'');
  return {cRow,mRow,cb};
}
function boxStep(r){ // which step the second box shows: the playhead's, the first chunk's (part 5, or when the longest lies past the chart's right edge), or the longest
  const k=play.t!=null?stepAt(r,play.t):-1;
  if(k>=0) return {s:r.steps[k],why:'now'};
  if(r.n>1&&((mode==='guided'&&STEPS[step].id==='free')||(mode!=='challenge'&&edgeOf(r).longOff))) return {s:r.steps[0],why:'first'};
  return {s:r.steps[r.li],why:'longest'};
}
/* the dashed box across both rows is drawn when the step it marks starts before the right edge (in the challenge the second box keeps
   the longest step, which may lie past it) */
const bandOn=r=>r.n>1&&boxStep(r).s.start<edgeOf(r).x1-1e-12;
function drawLim(){
  const lim=$('#ck-lim',app), dp=$('#ck-deep',app); if(!lim) return;
  const r=R(), ch=mode==='challenge', show=deepOpen&&!ch;
  lim.hidden=show; dp.hidden=!show; if(show){drawDeep(); return;}
  $('#ck-lim-cap',app).textContent=ch?tx('Why: each step takes the longer of its arithmetic and its memory traffic, here against the 50 ms goal','原因：每一步的耗时取计算和读写显存中较长的那个，这里和 50 ms 的目标对比')
    :tx('Why: each step takes the longer of its arithmetic and its memory traffic','原因：每一步的耗时取计算和读写显存中较长的那个');
  const ll=$('#ck-limleg',app), lh=limLegendHTML(); if(ll._h!==lh){ll.innerHTML=lh; ll._h=lh;}
  const n=r.normal, u=drawPhase('u',n,{title:tx('A usual step','平时的一步'),desc:tx(`${userN(r.U)}’ tokens, ${tokK(r.c)} of context each`,`${userN(r.U)}的 token，每人 ${tokK(r.c)} 的上下文`)});
  const B=boxStep(r), s=B.s, lab=B.why==='now'?tx(`Step ${fint(s.k+1)} of ${fint(r.n)}, now`,`第 ${fint(s.k+1)} 步（共 ${fint(r.n)} 步），当前`):(B.why==='first'?tx('The first chunk’s step','第一个 chunk 的那一步'):(r.n>1?tx('The longest step','最长的一步'):tx('The prefill step','prefill 的那一步')));
  const desc=r.n>1?tx(`the users’ tokens and prompt tokens ${fint(s.a+1)} to ${fint(s.b)}`,`用户的 token，加上 prompt 的第 ${fint(s.a+1)} 到 ${fint(s.b)} 个 token`):tx(`the users’ tokens and all ${fint(r.N)} prompt tokens`,`用户的 token，加上全部 ${fint(r.N)} 个 prompt token`);
  const v=drawPhase('s',s,{title:lab,desc,goal:ch?CH.goal:0});
  if(u){u.cRow.dataset.tipx=tx(`A usual step’s arithmetic: the ${userN(r.U)}’ tokens through the layers and the LM head, ${fixd(2*(P_LAYERS+P_HEAD)/1e9,1)} GFLOP each, plus attention over their contexts: ${f3(n.tc)} at ${tfS(FS)} TFLOP/s (50% of the ${GPUN}’s dense BF16 peak).`,
                                 `平时一步的计算：${userN(r.U)}的 token 各过一遍所有层和 LM head，每个 ${fixd(2*(P_LAYERS+P_HEAD)/1e9,1)} GFLOP，再加对各自上下文的 attention，按 ${tfS(FS)} TFLOP/s（${GPUN} 稠密 BF16 峰值的 50%）算，要 ${f3(n.tc)}。`);
    u.mRow.dataset.tipx=tx(`A usual step’s memory traffic: the weights, ${gbS(W_READ)} (${fT(n.tw)}), and the users’ KV caches, ${fint(r.U)} × ${fint(r.c)} × 128 KiB = ${gbS(KV_TOK*r.U*r.c)} (${fT(n.tkv)}), at ${tbS(BS)} TB/s (60% of 3.35 TB/s): ${fT(n.tm)}. It is the longer of the two, so it sets the step’s time.`,
                         `平时一步的读写显存：权重 ${gbS(W_READ)}（${fT(n.tw)}），加上用户的 KV cache，${fint(r.U)} × ${fint(r.c)} × 128 KiB = ${gbS(KV_TOK*r.U*r.c)}（${fT(n.tkv)}），按 ${tbS(BS)} TB/s（3.35 TB/s 的 60%）算，一共 ${fT(n.tm)}。它是两者中较长的，所以决定这一步的耗时。`);}
  if(v){v.cRow.dataset.tipx=tx(`This step’s arithmetic: the users’ tokens, ${f3(s.tcD)} (solid), and the prompt’s ${fint(s.chunk)} tokens (hatched): ${fT(s.tcM)} through the weights and ${fT(s.tcA)} of attention ${s.a?`to themselves and the ${fint(s.a)} prompt tokens before them`:'among themselves'}. ${fT(s.tc)} in all${s.cb?', the longer of the two, so it sets the time.':'.'}`,
                                 `这一步的计算是用户的 token ${f3(s.tcD)}（实心），加上 prompt 的 ${fint(s.chunk)} 个 token（斜纹）：和权重相乘 ${fT(s.tcM)}，${s.a?`对它们自己和前面 ${fint(s.a)} 个 prompt token 的 attention`:'它们彼此之间的 attention'} ${fT(s.tcA)}。一共 ${fT(s.tc)}${s.cb?'，是两者中较长的，所以决定耗时。':'。'}`);
    v.mRow.dataset.tipx=tx(`This step’s memory traffic: the weights, ${gbS(W_READ)} (${fT(s.tw)}), and the KV cache of the users’ contexts and of prompt tokens 1 to ${fint(s.b)}, ${gbS(KV_TOK*(r.U*r.c+s.b))} (${fT(s.tkv)}): ${fT(s.tm)}${!s.cb?', the longer of the two, so it sets the time.':'.'}`,
                         `这一步的读写显存：权重 ${gbS(W_READ)}（${fT(s.tw)}），加上用户上下文和 prompt 第 1 到 ${fint(s.b)} 个 token 的 KV cache，${gbS(KV_TOK*(r.U*r.c+s.b))}（${fT(s.tkv)}），一共 ${fT(s.tm)}${!s.cb?'，是两者中较长的，所以决定耗时。':'。'}`);}
}
/* Deep dive: the free chunk. One step's time against the chunk size, for the prompt's first chunk and for its last */
function drawDeep(){
  const box=$('#ck-deep',app); if(!box) return;
  const r=R(), U=r.U, c=r.c, N=r.N, x=crossOf(S), CM=Math.min(1024,N);
  const cw=box.clientWidth||(box.parentNode&&box.parentNode.clientWidth)||560, W=Math.max(300,Math.min(640,Math.round(cw))), narrow=W<480;
  const H=narrow?230:250, ml=44, mr=narrow?16:14, mt=16, mb=40;
  const first=C=>stepOf(U,c,0,C,N), last=C=>stepOf(U,c,N-C,N,N);
  let ymax=Math.max(last(CM).t,first(CM).t)*1e3*1.1; const yst=niceStep(ymax,4); ymax=Math.ceil(ymax/yst)*yst;
  const X=C=>ml+C/CM*(W-ml-mr), Y=ms=>H-mb-ms/ymax*(H-mt-mb);
  const pts=f=>{const a=[]; for(let C=0;C<=CM;C+=Math.max(1,Math.round(CM/160))) a.push(C); if(a[a.length-1]!==CM) a.push(CM); return a.map(C=>`${X(C).toFixed(1)},${Y(f(C)*1e3).toFixed(1)}`).join(' ');};
  let g='';
  for(let v=0;v<=ymax+1e-9;v+=yst){const y=Y(v); g+=`<line class="ck-fg" x1="${ml}" y1="${y}" x2="${W-mr}" y2="${y}"/><text class="ck-ft" x="${ml-6}" y="${y+4}" text-anchor="end">${fint(v)}</text>`;}
  const xt=CM>=1024?[0,256,512,768,1024]:[0,Math.round(CM/2),CM];
  xt.forEach(C=>{g+=`<text class="ck-ft" x="${X(C)}" y="${H-mb+14}" text-anchor="${C===CM?'end':(C===0?'start':'middle')}">${fint(C)}</text>`;});
  const fx=Math.min(x.first-1,CM);
  const zone=x.first>1?`<rect class="ck-fzone" x="${X(0)}" y="${mt}" width="${(X(fx)-X(0)).toFixed(1)}" height="${H-mb-mt}"/><text class="ck-ft ck-flab" x="${X(0)+5}" y="${mt+12}">${esc(tx(`free: up to ${fint(fx)} tokens`,`白搭车：不超过 ${fint(fx)} 个 token`))}</text>`:'';
  /* the small sizes crowd the left edge: only the learner's own pick among them */
  const dots=K_OPTS.filter(C=>C>0&&C<=CM&&(C>=256||C===r.C)).map(C=>{const on=C===r.C, fs=first(C), ls=last(C);
    const tip=esc(tx(`${fint(C)}-token chunks: the first chunk’s step takes ${fT(fs.t)} (${fs.cb?'arithmetic':'memory traffic'} sets it), the last chunk’s ${fT(ls.t)} (${ls.cb?'arithmetic':'memory traffic'} sets it).`,`${fint(C)} 个 token 的 chunk：第一个 chunk 的那一步要 ${fT(fs.t)}（由${fs.cb?'计算':'读写显存'}决定），最后一个 chunk 的那一步要 ${fT(ls.t)}（由${ls.cb?'计算':'读写显存'}决定）。`));
    return `<g class="ck-fpt${on?' on':''}" tabindex="0" data-tipx="${tip}"><circle cx="${X(C)}" cy="${Y(fs.t*1e3)}" r="${on?6:4.5}"/><rect x="${X(C)-(on?5.5:4)}" y="${Y(ls.t*1e3)-(on?5.5:4)}" width="${on?11:8}" height="${on?11:8}"/></g>`;}).join('');
  const svg=`<svg class="ck-fsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx(`One step’s time by chunk size: the first chunk’s step stays at the memory time up to ${fint(fx)} tokens; the last chunk’s step is longer at every size.`,`按 chunk 大小画一步的耗时：第一个 chunk 的那一步在 ${fint(fx)} 个 token 以内都等于读写显存的时间；最后一个 chunk 的那一步在任何大小下都更长。`))}">`+g+zone+
    `<text class="ck-ft ck-fax" x="${(ml+W-mr)/2}" y="${H-6}" text-anchor="middle">${esc(tx('chunk size (prompt tokens per step)','chunk 大小（每步 prefill 的 prompt token 数）'))}</text>`+
    `<text class="ck-ft ck-fax" x="12" y="${(mt+H-mb)/2}" text-anchor="middle" transform="rotate(-90 12 ${(mt+H-mb)/2})">${esc(tx('step time (ms)','一步的耗时（ms）'))}</text>`+
    `<polyline class="ck-fmem" points="${pts(C=>first(C).tm)}"/><polyline class="ck-farf" points="${pts(C=>first(C).tc)}"/>`+
    `<polyline class="ck-flast" points="${pts(C=>last(C).t)}"/><polyline class="ck-ffirst" points="${pts(C=>first(C).t)}"/>`+dots+`</svg>`;
  box.innerHTML=`<div class="ck-dhead"><span class="ck-cap">${tx(`Deep dive: the free chunk. How long one step takes, by chunk size, for ${userN(U)} at ${tokK(c)} and ${aTok(N)}-token prompt`,`深入：白搭车的 chunk。${userN(U)}（每人 ${tokK(c)}）、${tokK(N)} 个 token 的 prompt 时，一步的耗时随 chunk 大小的变化`)}</span><button type="button" class="btn ghost sm" data-act="deep-close">${tx('Back to the two limits','回到两种限制')}</button></div>`+svg+
    `<div class="tz-leg ck-fleg"><span><i class="ck-sw-ff"></i>${tx('thick line: step with the first chunk','粗线：放第一个 chunk 的一步')}</span><span><i class="ck-sw-fl"></i>${tx('thick dashed: step with the last chunk','粗虚线：放最后一个 chunk 的一步')}</span><span><i class="ck-sw-fa"></i>${tx('dashed: the first chunk’s arithmetic','虚线：第一个 chunk 那一步的计算')}</span><span><i class="ck-sw-fm"></i>${tx('thin line: its memory traffic','细线：它的读写显存')}</span><span><i class="ck-sw-fz"></i>${tx('shaded: the chunk rides free','阴影：chunk 白搭车')}</span></div>`+
    `<p class="stage-note">${tx('A usual step mostly waits for memory, and its arithmetic sits idle. A small chunk uses that idle arithmetic, so its step barely gets longer: the flat, shaded part. Past the shaded size, arithmetic sets the time and the step grows with the chunk. The last chunk also looks back at every earlier prompt token, so its line (thick dashed) rises much sooner: the longest wait is always the last chunk’s.',
                                 '平时的一步主要在等读显存，算力大半闲着。小 chunk 用的正是这些闲着的算力，所以这一步几乎不变长：就是阴影里平的那一段。chunk 超过阴影标出的大小后，由计算决定耗时，一步随 chunk 变大而变长。最后一个 chunk 还要回看前面所有的 prompt token，所以它那条线（粗虚线）早早就往上走：最长的等待总是出在最后一个 chunk。')}</p>`;
}
/* metric cards: ▲▼ only after the learner's own change, from the numbers as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
const unitOf=v=>{const m=String(v).match(/(ms|s)$/); return m?m[1]:'';};
const toBase=(v,u)=>u==='ms'?v/1000:v;
function metricsDef(){
  const r=R();
  return [
   {k:'wait',label:tx('Users’ longest wait','用户最长的等待'),v:fT(r.longest),sub:tx(`a usual step: ${fT(r.normal.t)}`,`平时一步 ${fT(r.normal.t)}`),
    tip:tx(`The longest time a user waits between two tokens while the prompt is prefilled: the longest step, since every user gets one token when each step ends. Here it is ${r.n>1?`step ${fint(r.li+1)} of ${fint(r.n)}`:'the one prefill step'}, ${fT(r.longest)}, against ${fT(r.normal.t)} for a usual step.`,
           `prompt prefill 期间，一个用户两个 token 之间最长要等多久：也就是最长的那一步，因为每一步结束时每个用户拿到一个 token。这里是${r.n>1?`第 ${fint(r.li+1)} 步（共 ${fint(r.n)} 步）`:'唯一的那一步 prefill'}，${fT(r.longest)}；平时一步是 ${fT(r.normal.t)}。`)},
   {k:'tpot',label:tx('Users’ TPOT in the prefill','prefill 期间用户的 TPOT'),v:fT(r.avg),sub:r.n>1?tx(`average of ${fint(r.n)} steps`,`${fint(r.n)} 步的平均`):tx('one step','只有一步'),
    tip:TIP.tpot()+tx(' ','')+tx(`While the prompt is prefilled, each user gets one token per step, so on average: ${f1(r.ttft)} ÷ ${fint(r.n)} = ${fT(r.avg)}.`,`prompt prefill 期间，每个用户每一步拿到一个 token，所以平均是：${f1(r.ttft)} ÷ ${fint(r.n)} = ${fT(r.avg)}。`)},
   {k:'ttft',label:tx('Prompt’s TTFT','长 prompt 的 TTFT'),v:fT(r.ttft),sub:tx('no queue','不含排队'),
    tip:TIP.ttft()+tx(' ','')+(r.n>1?tx(`Here: ${stepN(r.n)} of prefill, ${fT(r.ttft)}; without chunking, ${fT(run({...S,C:0}).ttft)}.`,`这里：${fint(r.n)} 步 prefill，${fT(r.ttft)}；不切 chunk 时是 ${fT(run({...S,C:0}).ttft)}。`)
      :tx(`Here: one step of prefill, ${fT(r.ttft)}.`,`这里：prefill 只有一步，${fT(r.ttft)}。`))},
   {k:'steps',label:tx('Prefill steps','prefill 的步数'),v:fint(r.n),sub:r.n>1?tx(`${tokK(r.chunk)} tokens each`,`每步 ${tokK(r.chunk)} 个 token`):tx('the whole prompt','整个 prompt'),
    tip:r.n>1?tx(`The prompt’s ${fint(r.N)} tokens in chunks of ${fint(r.chunk)}: ${fint(r.N)} ÷ ${fint(r.chunk)} = ${fint(r.n)} steps, one chunk each.`,`prompt 的 ${fint(r.N)} 个 token，每个 chunk ${fint(r.chunk)} 个：${fint(r.N)} ÷ ${fint(r.chunk)} = ${fint(r.n)} 步，每步一个 chunk。`)
      :tx(`Without chunking the whole prompt, ${fint(r.N)} tokens, is prefilled in one step.`,`不切 chunk 时，整个 prompt（${fint(r.N)} 个 token）一步 prefill 完。`)}];
}
function drawMetrics(){
  const host=$('#ck-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const ua=unitOf(prev), ub=unitOf(m.v), same=ua===ub, a=same?numOf(prev):toBase(numOf(prev),ua), b=same?numOf(m.v):toBase(numOf(m.v),ub);
      if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,same?{dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows (R6.6), and every term has its own tooltip (R1.4).
   H1/H2 (the owner, 2026-10-09): short blocks, like a page of a good textbook. Each block has one lead line in words, then a small table:
   the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the result, each on its
   own line with the "=" signs lined up. groups: [{name, terms:[TeX per column], rows:[{l, c:[html per column]} | {r:html}]}] */
const MX=x=>A.ui.math(x);
function mathBlock(lead,groups,note){const n=Math.max(...groups.map(g=>g.terms.length)), eq=`<td class="ck-meq">${MX('=')}</td>`;
  const cells=(a,f)=>a.map((x,i)=>`<td${i===a.length-1&&a.length<n?` colspan="${n-i}"`:''}>${f(x)}</td>`).join('');
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?cells(r.c,x=>x):`<td colspan="${n}" class="ck-mres">${r.r}</td>`}</tr>`;
  const body=groups.map(g=>`<tr class="ck-mname"><th scope="row">${g.name}</th>${eq}${cells(g.terms,MX)}</tr>${g.rows.map(tr).join('')}`).join('');
  return `<div class="ck-mb">${lead?`<p class="ck-mlead">${lead}</p>`:''}<div class="ck-mfx"><table class="ck-mt">${body}</table></div>${note?`<p class="ck-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.ck-mt'); if(!tb) return; const e0=tb.querySelector('td.ck-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('ck-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('ck-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="ck-mc${r?'':' ck-mc0'}">${x}</span>`;
    h+=`<tr class="ck-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="ck-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="ck-mband${g.tall?' ck-mtall':''}"><td class="ck-meq"><span class="ck-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="ck-mfl">${units.map(u=>`<span class="ck-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="ck-meq">${E}</td><td class="ck-mres">${r.l?`<span class="ck-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='ck-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.ck-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.ck-mfx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
function mathHTML(){
  const r=R(), t=T_, U=r.U, c=r.c, n=r.normal, s=r.steps[r.li];
  const TF=v=>fixd(v/1e12,4)+NS+'TFLOP', GB=v=>fixd(v/1e9,3)+NS+DEF('GB',TIP.gb()), ceN=tfS(FS)+NS+'TFLOP/s', beN=tbS(BS)+NS+'TB/s', ce=t('effc',ceN), be=t('effm',beN);
  const WL=t('wl','13,959,176,192'), WH=t('wh','1,050,673,152'), AU=t('attu',`4 × 32 × 4,096 × ${fint(c+1)}`);
  /* one step's time: compute time and memory time, the longer of the two (the GPU does both at once) */
  const step=(lead,o,flDesc,byDesc,resName)=>mathBlock(lead,[
    {name:tx('Compute time','计算时间'),terms:[tx('\\text{FLOP}','\\text{运算量}'),'\\div',tx('\\text{FLOP/s reached}','\\text{能达到的 FLOP/s}')],
      rows:[{r:`(${flDesc}) ÷ ${ce}`},{c:[TF(o.fl),'÷',ceN]},{r:f3(o.tc)}]},
    {name:tx('Memory time','读写显存时间'),terms:[tx('\\text{bytes moved}','\\text{搬运的字节数}'),'\\div',tx('\\text{bandwidth reached}','\\text{能达到的带宽}')],
      rows:[{r:`(${byDesc}) ÷ ${be}`},{c:[GB(o.by),'÷',beN]},{r:f3(o.tm)}]},
    {name:`${resName} (${DEF(tx('step time','这一步的耗时'),TIP.overlap())})`,terms:[tx('\\max(\\text{compute time},\\ \\text{memory time})','\\max(\\text{计算时间},\\ \\text{读写显存时间})')],
      rows:[{c:[`max(${f3(o.tc)}, ${f3(o.tm)})`]},{r:`<b>${fT(o.t)}</b>${o.cb?tx(', set by arithmetic','，由计算决定'):tx(', set by memory traffic','，由读写显存决定')}`}]}]);
  const rows=[];
  rows.push(mathBlock(tx('Every step reads all the weights once.','每一步都要把全部权重读一遍。'),
    [{name:tx('Weights read per step','每一步读的权重'),terms:[tx('\\text{parameters}','\\text{参数数}'),'\\times',tx('\\text{bytes per parameter}','\\text{每个参数的字节数}')],
      rows:[{c:[t('wr',fint(P_LAYERS+P_HEAD)),'×','2']},{r:`${fint(W_READ)} ${tx('bytes','字节')} (<b>${gbS(W_READ)}</b>)`}]}]));
  rows.push(mathBlock(tx('Every token keeps a key and a value for each KV head in every layer, 2 bytes per number in BF16.','每个 token 在每一层、每个 KV head 上存一个 key 和一个 value，BF16 下每个数值占 2 字节。'),
    [{name:tx('KV cache per token','每个 token 的 KV cache'),terms:['2','\\times',tx('\\text{layers}','\\text{层数}'),'\\times',tx('\\text{KV heads}','\\text{KV head 数}'),'\\times','\\text{head_dim}','\\times',tx('\\text{bytes per number}','\\text{每个数值的字节数}')],
      rows:[{c:[t('kv','2'),'×','32','×','8','×','128','×','2']},{r:`131,072 ${tx('bytes','字节')} (<b>128${NS}${DEF('KiB',TIP.kib())}</b>)`}]}]));
  rows.push(step(tx(`<b>A usual step: ${userN(U)} at ${fint(c)} tokens of context.</b> The GPU computes and moves memory at the same time, so a step takes the longer of the two.`,`<b>平时的一步：${userN(U)}，上下文 ${fint(c)} 个 token。</b>GPU 计算和读写显存同时进行，所以一步取两者中较长的那个。`),n,
    `${t('U',fint(U))} × (${WL} + ${WH}) + ${t('U',fint(U))} × ${AU}`,`${t('wr2',fint(W_READ))} + ${t('U',fint(U))} × ${t('c',fint(c))} × ${t('kvt','131,072')}`,tx('A usual step','平时的一步')));
  const last=s.b===r.N?1:0;
  rows.push(step(`<b>${r.n>1?tx(`The longest step, ${fint(s.k+1)} of ${fint(r.n)}: the users’ tokens and prompt tokens ${fint(s.a+1)} to ${fint(s.b)}.`,`最长的一步，第 ${fint(s.k+1)} 步（共 ${fint(r.n)} 步）：用户的 token，加上 prompt 的第 ${fint(s.a+1)} 到 ${fint(s.b)} 个 token。`):tx(`The prefill step: the users’ tokens and all ${fint(r.N)} prompt tokens.`,`prefill 的那一步：用户的 token，加上全部 ${fint(r.N)} 个 prompt token。`)}</b>`,s,
    `(${t('U',fint(U))} + ${t('C',fint(s.chunk))}) × ${WL} + (${t('U',fint(U))} + ${last}) × ${WH} + ${t('U',fint(U))} × ${AU} + ${t('attc',`2 × 32 × 4,096 × (${fint(s.b)} × ${fint(s.b+1)} − ${fint(s.a)} × ${fint(s.a+1)})`)}`,
    `${t('wr2',fint(W_READ))} + (${t('U',fint(U))} × ${t('c',fint(c))} + ${t('b',fint(s.b))}) × ${t('kvt','131,072')}`,tx('The users’ longest wait','用户最长的等待')));
  /* TTFT, split exactly: the prompt's arithmetic + steps × the users' arithmetic + what memory adds in the steps it sets; the parts as shown add up */
  const pA=+(r.tcP*1e3).toFixed(2), pB=+(r.tcD*1e3).toFixed(4), pC=+(r.extra*1e3).toFixed(2), tot=pA+r.n*pB+pC;
  rows.push(mathBlock(tx('The first token comes when the whole prompt is computed: the prompt’s own arithmetic, the users’ tokens in every one of its steps, and the time memory adds in the steps it sets.','整个 prompt 算完，首 token 才出来。要等的是 prompt 自己的计算、每一步里用户 token 的计算，以及读写显存决定耗时的那些步多出的时间。'),
    [{name:tx('Time to first token','首 token 延迟'),terms:[tx('\\text{prompt’s arithmetic}','\\text{prompt 的计算}'),'+',tx('\\text{steps}','\\text{步数}'),'\\times',tx('\\text{users’ arithmetic}','\\text{用户 token 的计算}'),'+',tx('\\text{memory’s extra}','\\text{读写显存多出的时间}')],
      rows:[{c:[t('tcp',fixd(pA,2)+NS+'ms'),'+',t('n',fint(r.n)),'×',t('tcd',fixd(pB,4)+NS+'ms'),'+',t('extra',fixd(pC,2)+NS+'ms')]},{r:`<b>${fixd(tot,1)}${NS}ms</b>`}]}]));
  rows.push(mathBlock(tx('While the prompt is prefilled, each user gets one token per step.','prompt prefill 期间，每个用户每一步拿到一个 token。'),
    [{name:tx('Steps','步数'),terms:[tx('\\frac{\\text{prompt tokens}}{\\text{chunk size}}','\\frac{\\text{prompt 的 token 数}}{\\text{chunk 大小}}')],rows:[{c:[`${fint(r.N)} ÷ ${fint(r.chunk)}`]},{r:`<b>${fint(r.n)}</b>`}]},
     {name:tx('Users’ time per output token during the prefill','prefill 期间用户每个输出 token 的耗时'),terms:[tx('\\frac{\\text{time to first token}}{\\text{steps}}','\\frac{\\text{首 token 延迟}}{\\text{步数}}')],rows:[{c:[`${fixd(tot,1)}${NS}ms ÷ ${t('n',fint(r.n))}`]},{r:`<b>${fixd(tot/r.n,2)}${NS}ms</b>`}]}]));
  /* in bytes, so the division redoes exactly: (80 GiB × 90% − weights − 2 GiB) ÷ 131,072 */
  const kvNeed=U*c+r.N, usable=Math.round(H100.mem*A.GiB*UTIL);
  rows.push(mathBlock(tx('The KV cache of every user and of the prompt must fit in the GPU memory the weights leave.','每个用户和这个 prompt 的 KV cache，都要放进权重之外的显存。'),
    [{name:tx('Tokens of KV cache that fit','放得下的 KV cache（token 数）'),terms:[tx('\\frac{\\text{usable memory} - \\text{weights} - \\text{overhead}}{\\text{KV cache per token}}','\\frac{\\text{可用显存} - \\text{权重} - \\text{开销}}{\\text{每个 token 的 KV cache}}')],
      rows:[{r:`(${t('mem',fint(usable))} − ${t('wmem',fint(W_MEM))} − ${t('oh',fint(OH*A.GiB))}) ${tx('bytes','字节')} ÷ ${t('kvt','131,072')}`},{r:`<b>${fint(TOK_FIT)}</b>`}]}],
    tx(`This run needs ${fint(U)} × ${fint(c)} + ${fint(r.N)} = ${fint(kvNeed)}. At ${tokK(c)} tokens each, at most ${fint(maxUsers(c,r.N))} users fit.`,`这次运行需要 ${fint(U)} × ${fint(c)} + ${fint(r.N)} = ${fint(kvNeed)} 个。每人 ${tokK(c)} 个 token 时，最多放得下 ${fint(maxUsers(c,r.N))} 个用户。`)));
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){S.U=CH.U; S.ctx=CH.ctx; S.N=CH.N; if(deepOpen) deepOpen=false;}
  if(cLock()) S.C=0;
  const lim=uFit(S.ctx,S.N); if(S.U>lim) S.U=lim;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const lg=$('#ck-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const gr=$('#ck-graph',app); gr.dataset.focus=mode==='guided'?(STEPS[step].focus||''):'';
  drawTL(); drawLim(); drawMetrics();
  const sc=$('#ck-scope',app), r=R();
  sc.textContent=tx(`Estimates for one GPU: no queue and no network; the users’ contexts stay at ${tokK(r.c)} tokens while the prompt is prefilled.`,`单卡估算：不含排队和网络；prompt prefill 期间，用户的上下文保持在 ${tokK(r.c)} 个 token。`);
  sc.dataset.tipx=tx('Every time here comes from this lab’s step model: the arithmetic at 50% of the GPU’s peak FLOP/s, the memory traffic at 60% of its peak bandwidth, whichever is longer. A real server also queues requests, sends tokens over the network, grows every context by one token per step, and may run other requests’ prefills in the same steps.','这里的每个时间都来自这个 Lab 的耗时模型：计算按 GPU 峰值 FLOP/s 的 50%，读写显存按峰值带宽的 60%，取较长的那个。真实的服务还有排队、网络传输，每一步每个上下文都会多一个 token，还可能在同样的步里插入别的请求的 prefill。');
  const mb=$('#ck-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;} fitMath(mb);
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  const setRng=(key,stops,val,fmt)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{const v=stops.indexOf(val); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fmt(val); i.setAttribute('aria-valuetext',fmt(val));});
  setRng('U',U_STOPS,S.U,userN); setRng('X',X_STOPS,S.ctx,v=>tx(`${tokK(v)} tokens`,`${tokK(v)} 个 token`)); setRng('N',N_STOPS,S.N,v=>tx(`${tokK(v)} tokens`,`${tokK(v)} 个 token`));
  $$('.seg-ctl[data-key="C"]',app).forEach(g=>A.ui.setPressed(g,String(S.C)));
  $$('.seg-ctl[data-key="speed"]',app).forEach(g=>A.ui.setPressed(g,speed));
  const m=maxUsers(S.ctx,S.N), un=$('#ck-Unote',app);
  if(un){un.innerHTML=tx(`Chats already writing their answers; each gets one token per step. At ${tokK(S.ctx)} tokens each, next to the ${tokK(S.N)}-token prompt, GPU memory holds the KV caches of at most ${fint(m)} users (the ${labLink('kv-cache')} lab’s model).`,`已经在写回答的对话，每一步各拿到一个 token。每人 ${tokK(S.ctx)} 个 token、再加上 ${tokK(S.N)} 个 token 的 prompt 时，显存最多放得下 ${fint(m)} 个用户的 KV cache（按 ${labLink('kv-cache')}这个 Lab 的模型）。`);
    un.classList.toggle('warn',clampNote);}
  const lock=mode==='challenge', cl=cLock(), c=$('#ck-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#ck-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx(`In the challenge you can change only the chunk size, in the task card. Everything else is fixed: ${MODEL} on one ${GPUN}, ${CH.U} users at ${tokK(CH.ctx)} tokens of context each, a ${tokK(CH.N)}-token document.`,`挑战中只能在任务卡里改 chunk 大小。其余设置固定：一张 ${GPUN} 跑 ${MODEL}，${CH.U} 个用户每人 ${tokK(CH.ctx)} 个 token 的上下文，一份 ${tokK(CH.N)} 个 token 的文档。`);
  const cw=$('#ck-ctl-C-wrap',app); if(cw) cw.hidden=cl||lock; const cn=$('#ck-Clock',app); if(cn) cn.hidden=!cl||lock;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const r=R();
  A.announce(tx(`${userN(r.U)} at ${tokK(r.c)} tokens, ${aTok(r.N)}-token prompt${r.n>1?` in ${tokK(r.chunk)}-token chunks`:' in one step'}: the users wait at most ${fT(r.longest)}; the prompt’s first token at ${fT(r.ttft)}.`,
                `${userN(r.U)}每人 ${tokK(r.c)} 个 token，${tokK(r.N)} 个 token 的 prompt${r.n>1?`切成 ${tokK(r.chunk)} 个 token 的 chunk`:'一步 prefill 完'}：用户最多等 ${fT(r.longest)}；prompt 的首 token 在 ${fT(r.ttft)}。`));},400);}

/* =====================================================================
   replay: the run advances by elapsed time; the speed is the whole timeline's length in real seconds (R3.9)
   ===================================================================== */
function stopPlay(){play.playing=false; play.rate=0; if(play.raf) cancelAnimationFrame(play.raf); play.raf=0; if(app) $$('[data-act="play"]',app).forEach(b=>{b.textContent=playLabel();});}
function onPlayEnd(){play.t=null; stopPlay(); drawPlay(); drawLim(); legendSync();}
const legendSync=()=>{const lg=$('#ck-legend',app); if(lg){const h=legendHTML(); if(lg._h!==h){lg.innerHTML=h; lg._h=h;}}};
function startPlay(rate){
  if(play.playing){stopPlay(); drawPlay(); return;}
  if(A.reduceMotion()){onPlayEnd(); return;}
  const sp=spanOf(S); if(play.t==null||play.t>=sp.x1-1e-9) play.t=sp.x0;
  play.playing=true; play.rate=rate||0; play.last=performance.now(); drawPlay();
  const tick=now=>{if(!app||!play.playing) return; const sp2=spanOf(S), dt=Math.min(0.1,(now-play.last)/1000); play.last=now;   // a long pause (hidden tab) does not jump ahead
    play.t+=dt*(sp2.x1-sp2.x0)/(play.rate||SPEED[speed]); if(play.t>=sp2.x1){onPlayEnd(); return;} drawPlay(); play.raf=requestAnimationFrame(tick);};
  play.raf=requestAnimationFrame(tick);
}
/* one step forward: the arrival, then the end of each prefill step, then the whole run */
function stepFwd(){
  stopPlay(); const r=R(), sp=spanOf(S), marks=[0].concat(r.steps.map(s=>s.end).filter(e=>e<=sp.x1+1e-12));   // only the marks on the chart
  if(play.t==null){play.t=0;}
  else {const nx=marks.find(m=>m>play.t+1e-9); play.t=nx==null?null:nx;}
  drawPlay(); drawLim(); legendSync(); void sp;
}
function playFrom(b,rate){const gr=$('#ck-tl',app);
  if(!play.playing&&b&&b.closest('#ck-rail')&&gr){const rc=gr.getBoundingClientRect(); if(rc.top<60||rc.bottom>window.innerHeight) gr.scrollIntoView({block:'start',behavior:A.reduceMotion()?'auto':'smooth'});}
  startPlay(rate);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  if(!U_STOPS.includes(S.U)) S.U=BASE.U; if(!X_STOPS.includes(S.ctx)) S.ctx=BASE.ctx; if(!N_STOPS.includes(S.N)) S.N=BASE.N; if(!K_OPTS.includes(S.C)) S.C=BASE.C;
  if(mode==='challenge'){S.U=CH.U; S.ctx=CH.ctx; S.N=CH.N;}
  if(cLock()) S.C=0;
  const lim=uFit(S.ctx,S.N); clampNote=S.U>lim; if(S.U>lim) S.U=lim;   // the users that fit next to the prompt: the note says so when it clamps
  stopPlay(); play.t=null; play.k=undefined;
  render();
}
/* a change the learner made: metric cards may show a ▲▼ badge */
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.id!=='free'&&s.id!=='remember') deepOpen=false;
  stopPlay(); play.t=null; play.k=undefined;
  const st=$('#ck-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg={chunk:'#ck-ic-C',free:'#ck-ic-C',stall:'#ck-ic-play'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#ck-rail .step.current',app)||$('#ck-rail',app));}
/* the result of a part comes into view after a click that produced it: scroll just enough to show all of it (R3.10) */
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* after a click in the rail, keep the sticky stage's head below the top bar (r12 N4), as labs/prefill-decode.js does */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#ck-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ck-rail .step.current',app)||$('#ck-rail .railcard',app)||$('#ck-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#ck-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ck-insight',app)); else revealBox($('#ck-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; deepOpen=false; S={U:CH.U,ctx:CH.ctx,N:CH.N,C:CH.start};}
  if(m==='sandbox') sbTouched=false;
  stopPlay(); play.t=null; play.k=undefined;
  A.shell.setMode(app,m,DESCS());
  const st=$('#ck-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.U,s.ctx,s.N,s.C].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==4) return null;
  const [U,ctx,N,C]=p.map(Number);
  if(!U_STOPS.includes(U)||!X_STOPS.includes(ctx)||!N_STOPS.includes(N)||!K_OPTS.includes(C)||U>maxUsers(ctx,N)) return null;
  return {U,ctx,N,C};
}

/* =====================================================================
   interview drill: seeded, from the engine; every step of the run is set by arithmetic, so the first token has a closed form (R6.6)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DRILLS=(()=>{const out=[]; [8,32].forEach(U=>[2048,8192].forEach(ctx=>[16384,65536].forEach(N=>[1024,4096].forEach(C=>{
  const r=run({U,ctx,N,C}); if(U*ctx+N<=TOK_FIT&&r.nMem===0) out.push({U,ctx,N,C});}))));
  return out;})();                                              // never the mission (16 users at 4K, 32K) nor the new cases
function makeDrill(seed){const rnd=mulberry(seed*7919+17), d=DRILLS[Math.floor(rnd()*DRILLS.length)]; return {seed,...d,r:run(d)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#ck-drill-q',app), a=$('#ck-drill-a',app), r=d.r, s=r.steps[r.li], f=r.steps[0];
  qd.innerHTML=tx(`<b>${MODEL}</b> runs in BF16 on one ${GPUN}, at 50% of its 989.5 TFLOP/s dense peak and 60% of its 3.35 TB/s. <b>${fint(d.U)}</b> users decode at <b>${fint(d.ctx)}</b> tokens of context each when a <b>${fint(d.N)}</b>-token prompt arrives, prefilled in <b>${fint(d.C)}</b>-token chunks, one per step next to their decode tokens. How long do the users wait at most between two tokens, and when does the prompt get its first token?`,
                  `<b>${MODEL}</b> 用 BF16 跑在一张 ${GPUN} 上，按稠密峰值 989.5 TFLOP/s 的 50%、带宽 3.35 TB/s 的 60% 计。<b>${fint(d.U)}</b> 个用户在 decode，每人 <b>${fint(d.ctx)}</b> 个 token 的上下文。这时来了一个 <b>${fint(d.N)}</b> 个 token 的 prompt，切成 <b>${fint(d.C)}</b> 个 token 的 chunk，每步一个，和用户的 decode token 一起算。用户两个 token 之间最多等多久？prompt 的首 token 什么时候出来？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const n=r.n, fDs=fint(f.fD), fP=r.fP;
  a.innerHTML=tx(`<p><b>Longest wait ≈ ${fT(s.t)} (the last chunk’s step); first token ≈ ${fT(r.ttft)}.</b></p>`+
    `<p>The users’ tokens per step: ${fint(d.U)} × (13,959,176,192 + 1,050,673,152) + ${fint(d.U)} × 4 × 32 × 4,096 × ${fint(d.ctx+1)} = ${fDs} FLOP (${f3(f.tcD)}). Every step reads 15,009,849,344 bytes of weights and the users’ KV cache.</p>`+
    `<p>Last chunk, prompt tokens ${fint(s.a+1)} to ${fint(s.b)}: arithmetic = ${fDs} + ${fint(s.chunk)} × 13,959,176,192 + 1,050,673,152 + 2 × 32 × 4,096 × (${fint(s.b)} × ${fint(s.b+1)} − ${fint(s.a)} × ${fint(s.a+1)}) = ${fint(s.fl)} FLOP ÷ ${tfS(FS)} TFLOP/s = ${f2(s.tc)}; memory = 15,009,849,344 + (${fint(d.U)} × ${fint(d.ctx)} + ${fint(s.b)}) × 131,072 = ${fint(s.by)} bytes ÷ ${tbS(BS)} TB/s = ${f2(s.tm)}. Arithmetic is longer: ${fT(s.t)}.</p>`+
    `<p>First token: even the first chunk’s step is set by arithmetic (${f2(f.tc)} against ${f2(f.tm)}), and later chunks do more arithmetic, so every step is, and the first token comes after all the arithmetic. The prompt’s own is 13,959,176,192 × ${fint(d.N)} + 1,050,673,152 + 2 × 32 × 4,096 × ${fint(d.N)} × ${fint(d.N+1)} = ${fint(fP)} FLOP, and the users’ tokens add ${fint(n)} × ${fDs} FLOP over the ${fint(n)} steps; in all, ${fint(fP+n*f.fD)} FLOP ÷ ${tfS(FS)} TFLOP/s = ${f1(r.ttft)}.</p>`,
    `<p><b>最长的等待 ≈ ${fT(s.t)}（最后一个 chunk 的那一步）；首 token ≈ ${fT(r.ttft)}。</b></p>`+
    `<p>每一步里用户的 token：${fint(d.U)} ×（13,959,176,192 + 1,050,673,152）+ ${fint(d.U)} × 4 × 32 × 4,096 × ${fint(d.ctx+1)} = ${fDs} FLOP（${f3(f.tcD)}）。每一步都要读 15,009,849,344 字节的权重和用户的 KV cache。</p>`+
    `<p>最后一个 chunk，prompt 的第 ${fint(s.a+1)} 到 ${fint(s.b)} 个 token：计算 = ${fDs} + ${fint(s.chunk)} × 13,959,176,192 + 1,050,673,152 + 2 × 32 × 4,096 ×（${fint(s.b)} × ${fint(s.b+1)} − ${fint(s.a)} × ${fint(s.a+1)}）= ${fint(s.fl)} FLOP ÷ ${tfS(FS)} TFLOP/s = ${f2(s.tc)}；读写显存 = 15,009,849,344 +（${fint(d.U)} × ${fint(d.ctx)} + ${fint(s.b)}）× 131,072 = ${fint(s.by)} 字节 ÷ ${tbS(BS)} TB/s = ${f2(s.tm)}。计算更长：${fT(s.t)}。</p>`+
    `<p>首 token：连第一个 chunk 的那一步都由计算决定（${f2(f.tc)}，读写显存 ${f2(f.tm)}），后面的 chunk 计算更多，所以每一步都由计算决定，首 token 要等全部计算做完。prompt 自己的计算是 13,959,176,192 × ${fint(d.N)} + 1,050,673,152 + 2 × 32 × 4,096 × ${fint(d.N)} × ${fint(d.N+1)} = ${fint(fP)} FLOP，${fint(n)} 步里用户的 token 再加 ${fint(n)} × ${fDs} FLOP，一共 ${fint(fP+n*f.fD)} FLOP ÷ ${tfS(FS)} TFLOP/s = ${f1(r.ttft)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={U:{en:'Users decoding',zh:'正在 decode 的用户数'},c:{en:'Their context',zh:'他们的上下文'},N:{en:'The long prompt',zh:'长 prompt'},C:{en:'Chunk size',zh:'chunk 大小'}};
const TERM={
 U:()=>tx(`${fint(S.U)} is the number of users decoding: each step computes one new token for each of them.`,`${fint(S.U)} 是正在 decode 的用户数：每一步给他们各算一个新 token。`),
 c:()=>tx(`${fint(S.ctx)} is each user’s context: the tokens of the conversation so far, whose KV cache every decode step reads.`,`${fint(S.ctx)} 是每个用户的上下文：对话到目前为止的 token，每个 decode 步都要读它们的 KV cache。`),
 C:()=>{const r=R(); return r.n>1?tx(`${fint(r.chunk)} is the chunk size: the prompt tokens this step prefills next to the users’ tokens.`,`${fint(r.chunk)} 是 chunk 大小：这一步在用户的 token 之外 prefill 的 prompt token 数。`):tx(`${fint(r.N)} is the whole prompt: without chunking, one step prefills all of it.`,`${fint(r.N)} 是整个 prompt：不切 chunk 时，一步把它全部 prefill 完。`);},
 b:()=>{const r=R(), s=r.steps[r.li]; return tx(`${fint(s.b)} prompt tokens have their KV cache in GPU memory by the end of this step: the step reads the earlier ones for attention and writes its own chunk’s.`,`到这一步结束时，已经有 ${fint(s.b)} 个 prompt token 的 KV cache 在显存里：这一步要读前面那些做 attention，再写入自己这个 chunk 的。`);},
 n:()=>{const r=R(); return tx(`${fint(r.n)} is the number of steps the prompt’s prefill takes: ${fint(r.N)} ÷ ${fint(r.chunk)}.`,`${fint(r.n)} 是 prompt 的 prefill 要用的步数：${fint(r.N)} ÷ ${fint(r.chunk)}。`);},
 wr:()=>tx(`${MODEL} multiplies ${fint(P_LAYERS)} parameters for every token (its 32 layers and the final RMSNorm) and ${fint(P_HEAD)} in the LM head for every new token, 2 bytes each in BF16. The embedding table is only looked up, one row per token, so a step does not read it whole.`,`${MODEL} 每个 token 要用 ${fint(P_LAYERS)} 个参数做乘法（32 层加最后的 RMSNorm），每个新 token 还要用 LM head 的 ${fint(P_HEAD)} 个，BF16 下每个 2 字节。embedding 表只需每个 token 查一行，所以一步不用把它整张读一遍。`),
 wr2:()=>tx(`${fint(W_READ)} bytes are the weights every step reads once: all the parameters it multiplies, 2 bytes each.`,`${fint(W_READ)} 字节是每一步都要读一遍的权重：参与乘法的全部参数，每个 2 字节。`),
 kv:()=>tx('Every token keeps a key and a value of 128 numbers for each of the 8 KV heads in each of the 32 layers, 2 bytes per number in BF16 (from the config.json).','每个 token 在 32 层、每层 8 个 KV head 上各存一个 key 和一个 value，都是 128 维，BF16 下每个数值 2 字节（来自 config.json）。'),
 kvt:()=>tx('131,072 bytes (128 KiB) is the KV cache of one token: a step reads it for every token of every context in the step.','131,072 字节（128 KiB）是一个 token 的 KV cache：这一步里每个上下文的每个 token，都要读一份。'),
 wl:()=>tx(`2 FLOPs, a multiply and an add, for each of the ${fint(P_LAYERS)} parameters of the layers: every token computed pays this, 13,959,176,192 FLOPs.`,`层里的 ${fint(P_LAYERS)} 个参数，每个 2 FLOP（一次乘、一次加）：每算一个 token 都要这么多，13,959,176,192 FLOP。`),
 wh:()=>tx(`2 FLOPs for each of the LM head’s ${fint(P_HEAD)} parameters (4,096 × 128,256). It scores every vocabulary token, once per new token only: each user’s token and the prompt’s first token.`,`LM head 的 ${fint(P_HEAD)} 个参数（4,096 × 128,256），每个 2 FLOP。它给词表里每个 token 打分，只为每个新 token 算一次：每个用户的 token，以及 prompt 的首 token。`),
 attu:()=>tx(`Each user’s new token meets the keys of its ${fint(S.ctx)} tokens of context and its own: ${fint(S.ctx+1)} pairs at 4 × 32 × 4,096 FLOPs a pair (score and value, 32 layers).`,`每个用户的新 token 要和它上下文里 ${fint(S.ctx)} 个 token 以及它自己的 key 相遇：${fint(S.ctx+1)} 对，每对 4 × 32 × 4,096 FLOP（得分和 value，32 层）。`),
 attc:()=>{const r=R(), s=r.steps[r.li]; return tx(`Attention of the chunk: prompt token i meets the keys of tokens 1 to i, so prompt tokens ${fint(s.a+1)} to ${fint(s.b)} make (${fint(s.b)} × ${fint(s.b+1)} − ${fint(s.a)} × ${fint(s.a+1)}) ÷ 2 query–key pairs, at 4 × 32 × 4,096 FLOPs a pair. The later the chunk, the more earlier tokens it meets.`,`chunk 的 attention：第 i 个 prompt token 要和第 1 到第 i 个 token 的 key 相遇，所以第 ${fint(s.a+1)} 到 ${fint(s.b)} 个 prompt token 一共（${fint(s.b)} × ${fint(s.b+1)} − ${fint(s.a)} × ${fint(s.a+1)}）÷ 2 对 query 和 key，每对 4 × 32 × 4,096 FLOP。chunk 越靠后，要相遇的前面的 token 越多。`);},
 tcp:()=>{const r=R(); return tx(`${fixd(r.tcP*1e3,2)} ms is the prompt’s own arithmetic: all ${fint(r.N)} tokens through the layers, its first token through the LM head, and the attention of every prompt token to the ones before it. It is the same however the prompt is split.`,`${fixd(r.tcP*1e3,2)} ms 是 prompt 自己的计算：全部 ${fint(r.N)} 个 token 过一遍各层，首 token 过 LM head，再加每个 prompt token 对它前面 token 的 attention。不管怎么切，它都一样。`);},
 tcd:()=>{const r=R(); return tx(`${fixd(r.tcD*1e3,4)} ms is the arithmetic of the users’ tokens in one step; every step of the prefill has it.`,`${fixd(r.tcD*1e3,4)} ms 是一步里用户 token 的计算；prefill 的每一步都有它。`);},
 extra:()=>{const r=R(); return r.nMem?tx(`${fixd(r.extra*1e3,2)} ms is what memory traffic adds: in ${fint(r.nMem)} of the ${fint(r.n)} steps the reads take longer than the arithmetic, and the step lasts as long as the reads.`,`${fixd(r.extra*1e3,2)} ms 是读写显存多出来的时间：${fint(r.n)} 步里有 ${fint(r.nMem)} 步读取比计算更久，这一步就按读取的时间算。`)
   :tx('0 ms: every step of this prefill is set by arithmetic, so memory traffic adds nothing.','0 ms：这次 prefill 的每一步都由计算决定，所以读写显存没有多花时间。');},
 effc:()=>tx(`${tfS(FS)} TFLOP/s is 50% of the ${GPUN}’s dense BF16 peak, 989.5 TFLOP/s (its datasheet’s 1,979 TFLOPS with sparsity, halved). The 50% is this lab’s assumption.`,`${tfS(FS)} TFLOP/s 是 ${GPUN} 稠密 BF16 峰值 989.5 TFLOP/s（规格表上 with sparsity 的 1,979 TFLOPS 减半）的 50%。50% 是本 Lab 的假设。`),
 effm:()=>tx(`${tbS(BS)} TB/s is 60% of the ${GPUN}’s peak memory bandwidth, 3.35 TB/s. The 60% is this lab’s assumption.`,`${tbS(BS)} TB/s 是 ${GPUN} 峰值显存带宽 3.35 TB/s 的 60%。60% 是本 Lab 的假设。`),
 mem:()=>{const u=fint(Math.round(H100.mem*A.GiB*UTIL)); return tx(`${u} bytes are the ${pct0(UTIL)} of the ${GPUN}’s memory the server uses: its 80 GB read as 80 GiB (80 × 2^30 bytes), as the KV cache memory lab does, since a card’s GB figure is close to that many GiB.`,`${u} 字节是服务端能用的 ${GPUN} 显存的 ${pct0(UTIL)}：和 KV cache 显存那个 Lab 一样，80 GB 按 80 GiB（80 × 2^30 字节）计，因为显卡标的 GB 接近同样多的 GiB。`);},
 util:()=>A.t('utilNote'), oh:()=>A.t('ohNote'),
 wmem:()=>tx(`The weights in GPU memory: all ${fint(P_TOTAL)} parameters, the embedding table included, 2 bytes each: ${gbS(W_MEM)} = ${fixd(W_MEM/A.GiB,2)} GiB.`,`显存里的权重：全部 ${fint(P_TOTAL)} 个参数，包括 embedding 表，每个 2 字节，一共 ${gbS(W_MEM)} = ${fixd(W_MEM/A.GiB,2)} GiB。`)
};
const termText=k=>{const tm=TERM[k]; return tm?tm():'';};
const termTipHTML=k=>TERM[k]?`<div>${esc(termText(k))}</div><div class="act">${jumpAct(k)}</div>`:'';
const FIXED=new Set(['wr','wr2','kv','kvt','wl','wh','mem','wmem','tcp','tcd','extra','attu','attc','b','n']);
const DER={effc:'eff',effm:'eff',util:'eff',oh:'eff'};
const locked=k=>(mode==='challenge'&&(k==='U'||k==='c'||k==='N'))||(k==='C'&&cLock());
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='tl') return tx('Click to scroll to the timeline in the chart.','点击滚到图里的时间线。');
  if(k==='lim') return tx('Click to scroll to the two limits of each step in the chart.','点击滚到图里每一步的两种限制。');
  if(k==='eff') return tx('Click to jump to “Assumed values” under Settings.','点击跳到“自由调参”里的“假设值”。');
  if(FIXED.has(k)) return tx(`Fixed in this lab: it follows from ${MODEL} on one ${GPUN} and the settings.`,`这个 Lab 里不能直接改：它由一张 ${GPUN} 跑 ${MODEL} 和当前的设置决定。`);
  if(DER[k]) return jumpAct(DER[k]);
  if(!CTLN[k]) return '';
  if(k==='C'&&cLock()) return tx('The chunk size stays “none” in parts 1 to 3; you pick it in part 4.','第 1 到 3 部分里，chunk 大小固定为“不切”；到第 4 部分再选。');
  if(locked(k)) return tx(`Fixed during the challenge: ${{U:'16 users',c:'4K tokens of context each',N:'a 32K-token document'}[k]}.`,`挑战中固定不变：${{U:'16 个用户',c:'每人 4K 个 token 的上下文',N:'一份 32K 个 token 的文档'}[k]}。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({U:$('#ck-ctl-U',app),c:$('#ck-ctl-X',app),N:$('#ck-ctl-N',app),C:$('#ck-ic-C',app)||$('#ck-ic-chC',app)||$('#ck-sb-C',app)||$('#ck-ctl-C',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#ck-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='tl'||k==='lim'){if(k==='lim'&&deepOpen){deepOpen=false; render();} const el=$(k==='tl'?'#ck-tl':'#ck-lim',app); A.tip.hide(); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='controls'){const c=$('#ck-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='eff'){const el=$('#ck-ctl-eff',app); if(el&&!el.closest('.locked')) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('Fixed during the challenge: 50% of the peak FLOP/s and 60% of the peak bandwidth.','挑战中固定不变：峰值 FLOP/s 的 50%，峰值带宽的 60%。'))}</div>`); return;}
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
  if(a==='reset'){bySettings(btn,()=>{if(mode==='challenge') setState({C:CH.start}); else setState({...BASE});}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(S.C).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){playFrom(btn); return;}
  if(a==='stepfwd'){stepFwd(); return;}
  if(a==='deep'){deepOpen=!deepOpen; if(deepOpen) deepSeen=true; renderRail();
    if(deepOpen){const el=$('#ck-deep',app); if(el&&innerWidth<=760) el.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(a==='deep-close'){deepOpen=false; renderRail(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ck-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#ck-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function scrubTo(v){stopPlay(); const sp=spanOf(S), t=sp.x0+(+v)/1000*(sp.x1-sp.x0); play.t=t>=sp.x1-1e-9?null:t; drawPlay(); drawLim(); legendSync();}
function bind(el){
  el.addEventListener('input',e=>{const tg=e.target; if(tg.id==='ck-scrub'){scrubTo(tg.value); return;}
    const k=tg.dataset&&tg.dataset.key; if(!k) return; const v=+tg.value;
    if(k==='U') bySettings(tg,()=>setState({U:U_STOPS[v]}));
    else if(k==='X') bySettings(tg,()=>setState({ctx:X_STOPS[v]}));
    else if(k==='N') bySettings(tg,()=>setState({N:N_STOPS[v]}));});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ck-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='speed'){speed=d.v; $$('.seg-ctl[data-key="speed"]',app).forEach(g=>A.ui.setPressed(g,speed)); return;}
      if(k==='C'){const inRail=!!b.closest('#ck-rail'), f=()=>setState({C:+d.v});
        if(inRail){wantObs=true; byLearner(f); revealSoon(); wantObs=false;} else bySettings(b,f); return;}
      return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ck-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#ck-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ck-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:{U:S.U,ctx:S.ctx,N:S.N,C:S.C},pred,observed:{...observed},saved:saved?{U:saved.U,ctx:saved.ctx,N:saved.N,C:saved.C}:null,chSubmitted,chReveal,
    deepOpen,deepSeen,whyOpen,speed,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPoint(S.C).ok}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=mode==='challenge'?{U:CH.U,ctx:CH.ctx,N:CH.N,C:p.C}:p; pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sv:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    const sid=mode==='guided'?STEPS[step].id:'';   // the deeper view stays open only where enterStep() and setMode() leave it open
    deepOpen=o.deepOpen===true&&mode!=='challenge'&&(mode!=='guided'||sid==='free'||sid==='remember'); deepSeen=o.deepSeen===true||deepOpen; whyOpen=o.whyOpen===true;
    if(SPEED[o.speed]) speed=o.speed; sbTouched=o.sbTouched===true;
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true; resumed=true;},
  mount(el,tok){
    wantObs=false; root=el; tlKey='';
    const p=tok&&!resumed?parse(tok):null; resumed=false;   /* after restore() the address names the saved settings: nothing to apply */ if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#ck-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ck-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    const rail=$('#ck-rail',app); rail.innerHTML=railFull(); A.ui.initRanges(rail);   // a language switch re-mounts: keep the exits (Batch 2)
    play.k=undefined; render();
  },
  leave(){clearTimeout(annT); stageStop(); stopPlay(); play.t=null; root=null; app=null;},
  onLang(){if(root){const d=$('#ck-math',app), open=d&&d.open, t=play.t; stopPlay(); this.mount(root,null); if(open) $('#ck-math',app).open=true; if(t!=null){play.t=t; drawPlay(); drawLim(); legendSync();}}},
  onResize(){if(app){tlKey=''; drawTL(); if(deepOpen) drawDeep();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#ck-tl',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#ck-graph',app),
    render({status,vis,read}){if(!app) return; const r=R(), s=statusOf(), sp=spanOf(S), X=t=>Math.max(0,Math.min(100,(t-sp.x0)/(sp.x1-sp.x0)*100)).toFixed(2)+'%';
      status.className='status '+s.cls; status.textContent=s.short;
      const on=r.steps.filter(x=>x.start<sp.x1), off=r.ttft>sp.x1+1e-12;   // only what lies on the chart
      const ticks=[]; for(let t=0;t>=sp.x0;t-=r.normal.t) ticks.push(t); on.forEach(x=>{if(x.end<=sp.x1) ticks.push(x.end);}); for(let t=r.ttft+r.after.t;t<=sp.x1;t+=r.after.t) ticks.push(t);
      const every=Math.max(1,Math.ceil(ticks.length/60));
      vis.innerHTML=`<div class="ck-dk">${on.map((x,i)=>`<i class="ck-dk-seg${i%2?' odd':''}" style="left:${X(x.start)};width:${((Math.min(x.end,sp.x1)-x.start)/(sp.x1-sp.x0)*100).toFixed(2)}%"></i>`).join('')}${ticks.filter((_,i)=>i%every===0).map(t=>`<i class="ck-dk-tick" style="left:${X(t)}"></i>`).join('')}${off?'':`<i class="ck-dk-first" style="left:${X(r.ttft)}"></i>`}</div>`;
      read.innerHTML=esc(tx(`${userN(r.U)}, ${aTok(r.N)}-token prompt${r.n>1?` in ${tokK(r.chunk)}-token chunks`:' in one step'}: they wait at most ${fT(r.longest)}`,`${userN(r.U)}，${tokK(r.N)} 个 token 的 prompt${r.n>1?`切成 ${tokK(r.chunk)} 个 token 的 chunk`:'一步 prefill 完'}：最多等 ${fT(r.longest)}`))+
        `<span class="dk-mem">${esc(mode==='challenge'?tx(' (the goal is at most 50 ms)','（目标是最多 50 ms）'):tx(`; first token at ${fT(r.ttft)}`,`；首 token 在 ${fT(r.ttft)}`))}</span>`;}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__ck=Object.freeze({
  run:s=>deep(run(s)), step:(U,c,a,b,N)=>deep(stepOf(U,c,a,b,N)), after:(U,c,N)=>deep(afterOf(U,c,N)), span:s=>deep(spanOf(s)), cross:s=>deep(crossOf(s)),
  maxUsers, chPoint:C=>deep(chPoint(C)), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)), drills:()=>deep(DRILLS),
  fmt:Object.freeze({t:fT,ms:tMs,gb:gbS}),
  consts:Object.freeze({H100:Object.freeze({peak:H100.peak,bw:H100.bw,mem:H100.mem}),EFF:Object.freeze({...EFF}),LL:Object.freeze({...LL}),
    P_EMB,P_HEAD,P_LAYER,P_LAYERS,P_TOTAL,KV_TOK,W_READ,W_MEM,TOK_FIT,UTIL,OH,U_STOPS:U_STOPS.slice(),X_STOPS:X_STOPS.slice(),N_STOPS:N_STOPS.slice(),K_OPTS:K_OPTS.slice(),
    BASE:{...BASE},CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,SPEED:{...SPEED}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},chSubmitted,chReveal,deepOpen,deepSeen,speed,play:{t:play.t,playing:play.playing},stageBusy:stageT!==0})
});
})();
