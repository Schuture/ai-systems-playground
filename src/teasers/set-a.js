/* Set A teasers for the Inference domain: prefill-decode, chunked-prefill, prefix-caching, quantization, spec-decoding.
   Each tile: one control, one small visual with a .tz-leg legend, and one or two sentences (.tz-read, about 35 English
   words at most, guide R11.5) that restate the setup and the result. The assumptions and sources are 2–4 short plain-text
   sentences in .tz-note paragraphs, each answering one question a reader would ask; the tile shows them behind its
   "Assumptions & sources" button.
   Every setting lives in module scope, so a re-render (language switch, reselecting the domain) keeps it.
   window.__tzA is a read-only hook for tests/set-a_numbers.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const fmtInt=A.fmt.int, num=A.fmt.num;

/* ---------- facts ----------
   H100 SXM (NVIDIA datasheet): BF16 Tensor Core 1,979 TFLOPS, marked "with sparsity", so dense is half; HBM3 3.35 TB/s.
   Llama-3.1-8B (meta-llama/llama-models sku_list.py): dim 4096, 32 layers, 32 heads, 8 KV heads, vocab 128,256,
   FFN 14,336 (1.3 × 8/3 × 4096 rounded up to a multiple of 1024); untied output layer. Total 8,030,261,248 parameters,
   which matches the safetensors count on Hugging Face. */
const H100={peak:1979e12/2,bw:3.35e12};
const EFF={mfu:0.5,mbu:0.6};            // assumed: prefill reaches 50% of peak FLOP/s, decode 60% of peak bandwidth
const LL={name:'Llama-3.1-8B',layers:32,hidden:4096,heads:32,kvHeads:8,headDim:128,vocab:128256,ffn:14336};
const P_EMB=LL.vocab*LL.hidden;          // embedding table: a lookup, no multiply, only one row read per token
const P_HEAD=LL.vocab*LL.hidden;         // output layer (not tied to the embedding)
const P_LAYER=2*LL.hidden*LL.hidden+2*LL.hidden*LL.kvHeads*LL.headDim+3*LL.hidden*LL.ffn+2*LL.hidden;
const P_LAYERS=LL.layers*P_LAYER+LL.hidden;   // the 32 blocks + final norm: multiplied by every token
const P_TOTAL=P_LAYERS+P_EMB+P_HEAD;
const KV_TOK=2*LL.layers*LL.kvHeads*LL.headDim*2;   // K and V, BF16: 131,072 B = 128 KiB per token
const W_READ=2*(P_LAYERS+P_HEAD);        // weight bytes one forward step reads from HBM

/* ---------- one step on the GPU: the slower of compute and memory traffic ---------- */
// causal attention FLOPs (QK^T and AV, all layers) for query positions a..b-1; query i sees i+1 keys
const attn=(a,b)=>2*LL.layers*LL.hidden*(b*(b+1)-a*(a+1));
function step(o){  // o: {tok: tokens multiplied by the layers, logits: tokens that need the output layer, attn: FLOPs, kv: KV entries touched}
  const fl=2*P_LAYERS*o.tok+2*P_HEAD*o.logits+o.attn, by=W_READ+KV_TOK*o.kv;
  const tc=fl/(H100.peak*EFF.mfu), tm=by/(H100.bw*EFF.mbu);
  return {t:Math.max(tc,tm),tc,tm,fl,by};
}

/* ---------- formatting ---------- */
const tokK=n=>n>=1024&&n%1024===0?(n/1024)+'K':fmtInt(n);      // tokens: K = 1,024
const NB=' ';                                     // keep a number and its unit on one line
function ms(sec){const v=sec*1e3; if(v>=999.5) return num(v/1000,2)+NB+'s'; if(v>=99.95) return num(v,0)+NB+'ms'; return num(v,1)+NB+'ms';}
const ms0=sec=>fmtInt(sec*1e3)+NB+'ms';             // 1,494 ms: first-token times in whole ms, no false precision (R8 re-check 13a)
const gb=b=>num(b/1e9,1)+NB+'GB';
const gib=b=>num(b/A.GiB,1)+NB+'GiB';
const pct2=x=>String(Number(x.toPrecision(2)));   // two significant digits: 11, 1.9, 0.12
function stored(x){const s=String(x), d=s.replace(/^0\.0*|\./g,'').length; return d<=8?s:'≈'+String(Number(x.toPrecision(6)));}
const storedShort=x=>{const s=stored(x); return s.length<=8?s:'≈'+String(Number(x.toPrecision(4)));};   // fits the 62 px value column
function flash(el){if(!el||A.reduceMotion()) return; el.classList.remove('tz-seta-flash'); void el.offsetWidth; el.classList.add('tz-seta-flash');}
const pos=v=>(Math.max(0,Math.min(100,v))).toFixed(3)+'%';
const pctOf=x=>num(x*100,0)+'%';                  // an assumed efficiency or ratio: 0.5 -> 50%
const TFLOPS=num(H100.peak/1e12,1)+NB+'TFLOP/s', TBPS=num(H100.bw/1e12,2)+NB+'TB/s';   // 989.5 TFLOP/s, 3.35 TB/s
const notes=n=>Array.from({length:n},()=>'<p class="tz-note"></p>').join('');
const setNotes=(el,list)=>el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=list[i]||'';});
const leg=items=>`<div class="tz-leg">${items.map(([cls,label,tip])=>`<span${tip?` data-tipx="${esc(tip)}"`:''}><i class="${cls}"></i>${esc(label)}</span>`).join('')}</div>`;

/* =====================================================================
   1. prefill-decode: the first token waits for the whole prompt; later tokens each re-read the weights
   ===================================================================== */
const PD_N=[512,1024,2048,4096,8192,16384,32768], PD_NEXT=20; let pdI=3;
function pdCompute(N){
  const pre=step({tok:N,logits:1,attn:attn(0,N),kv:N});        // whole prompt in one pass, writes its KV cache
  const dec=step({tok:1,logits:1,attn:attn(N,N+1),kv:N});      // one new token reads the weights and the prompt's KV cache
  return {N,ttft:pre.t,tpot:dec.t,preFlops:pre.fl,preCompute:pre.tc,preMemory:pre.tm,decBytes:dec.by,decCompute:dec.tc,decMemory:dec.tm,
    ratio:pre.t/dec.t,next:PD_NEXT,total:pre.t+PD_NEXT*dec.t};
}
// notes: 4 sentences (guide R11.5); the prompt length and the bytes a decode step reads follow the slider
function pdNotes(r){const N=fmtInt(r.N), dec=gb(r.decBytes);
  return tx([
    `Prefill computes all ${N} prompt tokens in one pass, so it grows with the prompt; decode re-reads the weights for every new token, so it barely changes.`,
    `Prefill runs at ${pctOf(EFF.mfu)} of the H100 SXM’s dense BF16 compute (${TFLOPS}); each decode step reads ${dec} of weights and KV cache at ${pctOf(EFF.mbu)} of ${TBPS}.`,
    `Both efficiencies are assumptions; Databricks measured about 60% of peak bandwidth when decoding one request on H100s. Sources: NVIDIA H100 datasheet; meta-llama/llama-models; Databricks (2023).`],[
    `Prefill 一次算完全部 ${N} 个 prompt token，所以 prompt 越长越慢；decode 每出一个 token 都要重读一遍权重，所以每个 token 的耗时几乎不变。`,
    `Prefill 按 H100 SXM 稠密 BF16 算力（${TFLOPS}）的 ${pctOf(EFF.mfu)} 算；decode 每步读 ${dec}（权重加 KV cache），按 ${TBPS} 带宽的 ${pctOf(EFF.mbu)} 算。`,
    `这两个效率是假设值：Databricks 在 H100 上实测，单个请求 decode 时约用到峰值带宽的 60%。来源：NVIDIA H100 规格表；meta-llama/llama-models；Databricks（2023）。`]);}
A.registerTeaser('prefill-decode',{render(el){
  const lab=tx('Prompt length','Prompt 长度');
  let ticks=''; for(let i=0;i<PD_NEXT;i++) ticks+='<i class="tz-pfd-tick"></i>';
  el.innerHTML=`<div class="tz tz-pfd"><label class="tz-lab"><span>${lab}</span><output class="mono"></output></label>`+
    `<input type="range" min="0" max="${PD_N.length-1}" step="1" value="${pdI}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-pfd-vis" role="img"><div class="tz-pfd-lab"><span class="tz-pfd-lpre"></span></div>`+
    `<div class="tz-pfd-track"><i class="tz-pfd-pre"></i><i class="tz-pfd-mark"></i>${ticks}</div>`+
    `<div class="tz-pfd-lab tz-pfd-lab-r"><span class="tz-pfd-ldec"></span></div></div>`+
    leg([['tz-pfd-sw-pre',tx('prefill: the whole prompt','prefill：整个 prompt'),tx('The GPU computes every prompt token in one pass before the first token appears.','GPU 先一次算完所有 prompt token，之后才出首 token。')],
         ['tz-pfd-sw-mark',tx('thick line: first token','粗线：首 token'),tx('The thick line is the first token: it appears only after prefill has processed the whole prompt.','粗线是首 token：prefill 算完整个 prompt 之后才出现。')],
         ['tz-pfd-sw-tick',tx('thin tick: each later token','细线：之后每个 token'),tx('Each thin tick is a decoded token: every new token takes one full pass over the weights.','细线是 decode 出的 token：每出一个新 token，都要把全部权重读一遍。')]])+
    `<p class="tz-read"></p>${notes(3)}</div>`;
  const q=s=>el.querySelector(s), inp=q('input'), out=q('output'), vis=q('.tz-pfd-vis'), rd=q('.tz-read');
  const upd=()=>{pdI=+inp.value; A.ui.setRange(inp); const r=pdCompute(PD_N[pdI]), X=t=>t/r.total*100, tk=tokK(r.N);
    out.textContent=tx(`${tk} tokens`,`${tk} token`); inp.setAttribute('aria-valuetext',out.textContent);
    q('.tz-pfd-pre').style.width=pos(X(r.ttft)); q('.tz-pfd-mark').style.left=pos(X(r.ttft));
    el.querySelectorAll('.tz-pfd-tick').forEach((t,i)=>{t.style.left=pos(X(r.ttft+(i+1)*r.tpot));});
    q('.tz-pfd-lpre').textContent=tx(`prefill → first token at ${ms(r.ttft)}`,`prefill → ${ms(r.ttft)} 出首 token`);
    q('.tz-pfd-ldec').textContent=tx(`decode: 1 token every ${ms(r.tpot)}`,`decode：每 ${ms(r.tpot)} 一个 token`);
    vis.setAttribute('aria-label',tx(`Timeline: ${ms(r.ttft)} of prefill before the first token, then one token every ${ms(r.tpot)}.`,`时间线：prefill ${ms(r.ttft)} 后出首 token，之后每 ${ms(r.tpot)} 一个 token。`));
    rd.innerHTML=tx(`${LL.name} on one H100 SXM, a single ${tk}-token prompt: first token after <b>${ms(r.ttft)}</b> of prefill, then decode adds one token every <b>${ms(r.tpot)}</b>.`,
      `一张 H100 SXM 跑 ${LL.name}，单个 ${tk} token 的 prompt：prefill 要 <b>${ms(r.ttft)}</b> 才出首 token，之后 decode 每 <b>${ms(r.tpot)}</b> 出一个 token。`);
    setNotes(el,pdNotes(r));};
  inp.addEventListener('input',upd); inp.addEventListener('change',()=>{flash(rd); A.announce(rd.textContent);}); upd();
}});

/* =====================================================================
   2. chunked-prefill: a 32K prompt arrives while 16 users decode; each step = 16 decode tokens + one prefill chunk
   ===================================================================== */
const CP_C=[0,8192,2048,512]; let cpI=0;                 // 0 = no chunking: the whole prompt in one step
const CP={users:16,userCtx:4096,prompt:32768,from:-0.1,to:1.7};
function cpCompute(C){
  const U=CP.users, UC=CP.userCtx, N=CP.prompt, chunk=C||N, n=Math.ceil(N/chunk), dAttn=U*attn(UC,UC+1);
  const normal=step({tok:U,logits:U,attn:dAttn,kv:U*UC});           // the 16 users alone
  const steps=[]; let t=0;
  for(let k=0;k<n;k++){const a=k*chunk, b=Math.min(N,a+chunk);
    const s=step({tok:U+(b-a),logits:U+(b===N?1:0),attn:dAttn+attn(a,b),kv:U*UC+b});
    steps.push({a,b,start:t,end:t+s.t,dur:s.t,compute:s.tc,memory:s.tm}); t+=s.t;}
  const after=step({tok:U+1,logits:U+1,attn:dAttn+attn(N,N+1),kv:U*UC+N});   // the long prompt now decodes too
  return {C,chunk,n,normal:normal.t,normalCompute:normal.tc,normalMemory:normal.tm,after:after.t,steps,ttft:t,longest:Math.max(...steps.map(s=>s.dur))};
}
const CP_BASE=cpCompute(0);
// notes: 4 sentences (guide R11.5), the same for every chunk size; the numbers are the unchunked run's and the users' usual step.
// ZH says 一步 / 每一步 for an engine step everywhere on this tile (guide R2.8).
function cpNotes(){const b=CP_BASE, U=CP.users, ctx=tokK(CP.userCtx);
  return tx([
    `Without chunking, the long prompt is prefilled in one step: for those ${ms0(b.ttft)} none of the ${U} users gets a token.`,
    `With chunking, each step gives the ${U} users one token each and then prefills one chunk; later chunks attend to every earlier prompt token, so the last chunk sets the longest wait.`,
    `A step takes the longer of its compute, at ${pctOf(EFF.mfu)} of the H100 SXM’s dense BF16 peak, and its memory reads, at ${pctOf(EFF.mbu)} of ${TBPS}; with ${ctx} tokens of context per user, a usual step takes ${ms(b.normal)}.`,
    `Sources: NVIDIA H100 datasheet; meta-llama/llama-models; Databricks (2023).`],[
    `不切 chunk 时，长 prompt 在一步里 prefill 完：这 ${ms0(b.ttft)} 里，${U} 个用户都拿不到 token。`,
    `切成 chunk 后，每一步给 ${U} 个用户各生成一个 token，再 prefill 一个 chunk；越靠后的 chunk 要对前面所有 prompt token 做 attention，所以最后一个 chunk 决定最长的等待。`,
    `每一步的耗时取计算（H100 SXM 稠密 BF16 算力的 ${pctOf(EFF.mfu)}）和读显存（${TBPS} 的 ${pctOf(EFF.mbu)}）里较长的那个；每个用户有 ${ctx} token 上下文，所以平时一步 ${ms(b.normal)}。`,
    `来源：NVIDIA H100 规格表；meta-llama/llama-models；Databricks（2023）。`]);}
A.registerTeaser('chunked-prefill',{render(el){
  const lab=tx('Prefill chunk size (tokens per step)','Prefill 的 chunk 大小（每一步 prefill 的 token 数）');
  const opts=CP_C.map(c=>[String(c),c?tokK(c):tx('none','不切')]);
  el.innerHTML=`<div class="tz tz-chunk">${A.ui.seg({key:'tz-chunk',label:lab,options:opts,value:String(CP_C[cpI]),inline:false})}`+
    `<div class="tz-vis tz-chunk-vis" role="img"><div class="tz-chunk-top"><span class="tz-chunk-arr"></span><span class="tz-chunk-ft"></span></div>`+
    `<div class="tz-chunk-track"><span class="tz-chunk-ticks"></span></div><div class="tz-chunk-segs"></div><div class="tz-chunk-axis"></div></div>`+
    leg([['tz-chunk-sw-tick',tx('thin tick: all 16 users get a token','细线：16 个用户各得到一个 token')],
         ['tz-chunk-sw-seg',tx('bar: prefill chunks (shades alternate)','横条：prefill chunk（深浅交替）'),tx('One shade per step that prefills a chunk of the long prompt.','每种深浅是一步，这一步 prefill 长 prompt 的一个 chunk。')]])+
    `<p class="tz-read"></p>${notes(4)}</div>`;
  const q=s=>el.querySelector(s), seg=q('.seg-ctl'), vis=q('.tz-chunk-vis'), rd=q('.tz-read');
  // the label may wrap only before its bracket: "Prefill 的 chunk 大小" / "（每一步 prefill 的 token 数）" (one flex item, same text)
  const lb=q('.ctl>.lbl'), br=lb?lb.textContent.search(/[（(]/):-1;
  if(br>0) lb.innerHTML=`<span>${esc(lb.textContent.slice(0,br))}<span class="tz-chunk-nw">${esc(lb.textContent.slice(br))}</span></span>`;
  const W=CP.to-CP.from, X=t=>(t-CP.from)/W*100;
  q('.tz-chunk-axis').innerHTML=[0,0.5,1,1.5].map(s=>`<span style="left:${pos(X(s))}">${num(s,1)}${NB}s</span>`).join('');
  const upd=()=>{const C=CP_C[cpI], r=cpCompute(C), b=CP_BASE; A.ui.setPressed(seg,String(C));
    let tk=[]; for(let t=0;t>=CP.from;t-=r.normal) tk.push(t);
    r.steps.forEach(s=>tk.push(s.end)); for(let t=r.ttft+r.after;t<=CP.to;t+=r.after) tk.push(t);
    q('.tz-chunk-ticks').innerHTML=tk.filter(t=>t>=CP.from&&t<=CP.to).map(t=>`<i style="left:${pos(X(t))}"></i>`).join('');
    q('.tz-chunk-segs').innerHTML=r.steps.map((s,i)=>`<i class="${i%2?'odd':''}" style="left:${pos(X(s.start))};width:${pos(X(s.end)-X(s.start))}"></i>`).join('');
    const arr=q('.tz-chunk-arr'), ft=q('.tz-chunk-ft');
    arr.style.left=pos(X(0)); arr.textContent=tx('▾ prompt arrives','▾ prompt 到达');
    ft.style.left=pos(X(r.ttft)); ft.textContent=tx('its first token ▾','它的首 token ▾');
    const gap=C?ms(r.longest):ms0(r.longest);             // unchunked, the longest gap is the one prefill step: shown as the sentence shows it
    vis.setAttribute('aria-label',tx(`Timeline of the 16 users' tokens: longest gap ${gap}; the long prompt's first token after ${ms0(r.ttft)}.`,`16 个用户出 token 的时间线：最长间隔 ${gap}；长 prompt 的首 token 在 ${ms0(r.ttft)} 后。`));
    rd.innerHTML=C?tx(`${LL.name}, one H100 SXM, 16 users decoding, a 32K-token prompt prefilled in ${tokK(C)}-token chunks: their longest wait between tokens is <b>${ms(r.longest)}</b> (${ms0(b.ttft)} unchunked).`,
        `一张 H100 SXM 跑 ${LL.name}，16 个用户正在 decode，32K token 的 prompt 切成 ${tokK(C)} token 的 chunk：每个用户两个 token 之间最长等 <b>${ms(r.longest)}</b>（不切时 ${ms0(b.ttft)}）。`)
      :tx(`${LL.name}, one H100 SXM, 16 users decoding, a 32K-token prompt prefilled in one step: they wait <b>${ms0(r.ttft)}</b> between tokens instead of ${ms(r.normal)}.`,
        `一张 H100 SXM 跑 ${LL.name}，16 个用户正在 decode，32K token 的 prompt 一步 prefill 完：每个用户两个 token 之间要等 <b>${ms0(r.ttft)}</b>，平时只要 ${ms(r.normal)}。`);
    setNotes(el,cpNotes());};
  seg.addEventListener('click',e=>{const bt=e.target.closest('button[data-v]'); if(!bt||!seg.contains(bt)) return; const i=CP_C.indexOf(+bt.dataset.v); if(i<0) return;
    cpI=i; upd(); flash(rd); A.announce(rd.textContent);});
  upd();
}});

/* =====================================================================
   3. prefix-caching: N requests share one 10,000-token system prompt, each adds 200 tokens of its own
   ===================================================================== */
const PC_N=[1,10,100,1000]; let pcI=2; const PC={prefix:10000,own:200};
function pcCompute(n){const without=n*(PC.prefix+PC.own), withC=PC.prefix+n*PC.own;
  return {n,without,with:withC,saved:without-withC,pct:(1-withC/without)*100,kvBytes:PC.prefix*KV_TOK};}
// notes: 4 sentences (guide R11.5); N is the number of requests, named where it first appears
function pcNotes(r){const kv=gib(r.kvBytes), per=num(KV_TOK/1024,0)+NB+'KiB', P=fmtInt(PC.prefix), O=fmtInt(PC.own), PO=fmtInt(PC.prefix+PC.own);
  return tx([
    `The cost: the shared prefix’s KV cache stays in GPU memory, ${per} per token, so ${kv} for its ${P} tokens (${LL.name}, BF16).`,
    `For N requests, N × ${PO} tokens are prefilled without prefix caching, and ${P} + N × ${O} with it. Only an identical prefix is reused; vLLM calls this automatic prefix caching (docs: features/automatic_prefix_caching.md).`,
    `This counts tokens, not time: the ${O} new tokens of each request still attend to the ${P} cached ones.`],[
    `代价是共享前缀的 KV cache 要一直留在显存里：每个 token 占 ${per}，这 ${P} 个 token 共 ${kv}（${LL.name}，BF16）。`,
    `N 个请求时，不用 prefix caching 要 prefill N × ${PO} 个 token，用了只要 ${P} + N × ${O} 个。只有完全相同的前缀才能复用；vLLM 叫它 automatic prefix caching（文档：features/automatic_prefix_caching.md）。`,
    `这里数的是 token，不是时间：每个请求新的 ${O} 个 token 仍要对缓存的 ${P} 个 token 做 attention。`]);}
A.registerTeaser('prefix-caching',{render(el){
  const lab=tx('Requests with the same system prompt','用同一个 system prompt 的请求数');
  el.innerHTML=`<div class="tz tz-pfx">${A.ui.seg({key:'tz-pfx',label:lab,options:PC_N.map(n=>[String(n),fmtInt(n)]),value:String(PC_N[pcI]),inline:false})}`+
    `<div class="tz-vis tz-pfx-vis" role="img"><div class="tz-pfx-cap">${tx('Tokens to prefill','要 prefill 的 token 数')}</div>`+
    `<div class="tz-pfx-row"><span>${tx('without prefix caching','不用 prefix caching')}</span><b class="mono tz-pfx-v0"></b></div><div class="tz-pfx-bar tz-pfx-b0"><i class="tz-pfx-rep"></i><i class="tz-pfx-own tz-pfx-own0"></i></div>`+
    `<div class="tz-pfx-row"><span>${tx('with prefix caching','用 prefix caching')}</span><b class="mono tz-pfx-v1"></b></div><div class="tz-pfx-bar tz-pfx-b1"><i class="tz-pfx-sh"></i><i class="tz-pfx-own"></i></div></div>`+
    leg([['tz-pfx-sw-sh',tx('shared system prompt','共享的 system prompt')],['tz-pfx-sw-own',tx('each request’s own tokens','每个请求自己的 token')],
         ['tz-pfx-sw-gap',tx('thin gap: next request','细缝：下一个请求')]])+
    `<p class="tz-read"></p>${notes(3)}</div>`;
  const q=s=>el.querySelector(s), seg=q('.seg-ctl'), vis=q('.tz-pfx-vis'), rd=q('.tz-read'), gap=q('.tz-pfx-sw-gap').parentNode;
  const upd=()=>{const r=pcCompute(PC_N[pcI]); A.ui.setPressed(seg,String(r.n));
    // R6.18: token totals over 6 digits in decimal M (1.02M, 10.2M). No hover with the exact count: these totals are exact at 3 digits,
    // and the product owner found explaining how big 10.2M is unnecessary (2026-10-02)
    const BGt=n=>n>=1e6?A.fmt.big(n):fmtInt(n), BGa=BGt;
    q('.tz-pfx-v0').innerHTML=BGt(r.without); q('.tz-pfx-v1').innerHTML=BGt(r.with);
    const X=v=>v/r.without*100;                           // both bars share one scale: tokens without caching = full width
    const rep=q('.tz-pfx-rep'), lines=r.n>1&&r.n<=100;     // a thin line between repeats of the shared prompt, while they are wide enough to see
    rep.style.width=pos(X(r.n*PC.prefix)); rep.classList.toggle('tz-pfx-lines',lines); rep.style.backgroundSize=lines?`${(100/r.n).toFixed(4)}% 100%`:''; gap.hidden=!lines;
    const o0=q('.tz-pfx-own0'); o0.style.left=pos(X(r.n*PC.prefix)); o0.style.width=pos(X(r.n*PC.own));
    const o1=q('.tz-pfx-b1 .tz-pfx-own'); q('.tz-pfx-sh').style.width=pos(X(PC.prefix)); o1.style.left=pos(X(PC.prefix)); o1.style.width=pos(X(r.n*PC.own));
    vis.setAttribute('aria-label',tx(`Tokens to prefill: ${BGa(r.without)} without prefix caching, ${BGa(r.with)} with it.`,`要 prefill 的 token：不用 prefix caching ${BGa(r.without)} 个，用了 ${BGa(r.with)} 个。`));
    const P=fmtInt(PC.prefix), O=fmtInt(PC.own);           // R8 re-check 13b: one shared prompt, and each request's own tokens said in full
    rd.innerHTML=r.n===1?tx(`1 request with a ${P}-token system prompt and ${O} tokens of its own: <b>${fmtInt(r.without)}</b> tokens to prefill with or without prefix caching, since nothing is cached yet.`,
        `1 个请求用一段 ${P} 个 token 的 system prompt，还有自己的 ${O} 个 token：用不用 prefix caching 都要 prefill <b>${fmtInt(r.without)}</b> 个 token，因为还没有可复用的缓存。`)
      :tx(`${fmtInt(r.n)} requests share one ${P}-token system prompt, and each adds ${O} tokens of its own: <b>${BGt(r.without)}</b> tokens to prefill without prefix caching, <b>${BGt(r.with)}</b> with it (${num(r.pct,0)}% fewer).`,
        `${fmtInt(r.n)} 个请求共用一段 ${P} 个 token 的 system prompt，每个请求还有自己的 ${O} 个 token：不用 prefix caching 要 prefill <b>${BGt(r.without)}</b> 个 token，用了只要 <b>${BGt(r.with)}</b> 个（少 ${num(r.pct,0)}%）。`);
    setNotes(el,pcNotes(r));};
  seg.addEventListener('click',e=>{const bt=e.target.closest('button[data-v]'); if(!bt||!seg.contains(bt)) return; const i=PC_N.indexOf(+bt.dataset.v); if(i<0) return;
    pcI=i; upd(); flash(rd); A.announce(rd.textContent);});
  upd();
}});

/* =====================================================================
   4. quantization: round one value to the nearest representable BF16 / FP8 E4M3 / FP4 E2M1 number (ties to even)
   ===================================================================== */
// OCP OFP8 spec: E4M3 has 3 mantissa bits, bias 7, max 448 (no infinities). OCP MX spec: FP4 E2M1 has bias 1, max 6.
// BF16: 8 exponent bits, 7 mantissa bits, bias 127.
const QZ_F=[{id:'fp4',name:'FP4',sub:'E2M1',e:2,m:1,bias:1,max:6},{id:'fp8',name:'FP8',sub:'E4M3',e:4,m:3,bias:7,max:448},{id:'bf16',name:'BF16',sub:'',e:8,m:7,bias:127,max:3.3895313892515355e38}];
const QZ_MAX=600; let qzV=270;                          // slider value = hundredths, 0.00 … 6.00
function qzRound(v,f){
  if(v===0) return {q:0,tie:false};
  let e=Math.floor(Math.log2(v)); if(Math.pow(2,e)>v) e--; if(Math.pow(2,e+1)<=v) e++;
  e=Math.max(e,1-f.bias);                               // below the smallest normal: subnormal spacing
  const s=Math.pow(2,e-f.m), r=v/s; let n=Math.floor(r); const fr=r-n, tie=fr===0.5;
  if(fr>0.5||(tie&&n%2===1)) n++;
  return {q:Math.min(n*s,f.max),tie};
}
function qzGrid(f,hi){const out=[]; if(f.e>4) return out;   // BF16 is too dense to draw
  for(let E=0;E<(1<<f.e);E++) for(let M=0;M<(1<<f.m);M++){
    if(f.id==='fp8'&&E===15&&M===7) continue;            // E4M3 NaN
    const v=E===0?Math.pow(2,1-f.bias)*M/(1<<f.m):Math.pow(2,E-f.bias)*(1+M/(1<<f.m)); if(v<=hi) out.push(v);}
  return out;}
function qzCompute(i){const v=i/100; return {v,rows:QZ_F.map(f=>{const o=qzRound(v,f); return {id:f.id,name:f.name,sub:f.sub,q:o.q,tie:o.tie,err:v?Math.abs(o.q-v)/v*100:0};})};}
// notes: 4 sentences (guide R11.5); the FP4 values and the steps per power of two come from the format table above
const QZ_NOTES=()=>{const f4=qzGrid(QZ_F[0],QZ_F[0].max).map(String), f8=1<<QZ_F[1].m, b16=1<<QZ_F[2].m;
  return tx([
  `Your browser rounds the value to the nearest number each format can store, ties to even (the rounding the OCP specs require).`,
  `FP4 E2M1 can store only ${f4.slice(0,-1).join(', ')} and ${f4[f4.length-1]}; FP8 E4M3 has ${f8} values between two powers of two, BF16 has ${b16}. The spacing doubles at every power of two, so larger values lose more in absolute terms.`,
  `Real FP8/FP4 inference first divides each block of values by a shared scale (one per 32 values in MXFP8 and MXFP4, one per 16 in NVFP4), so this grid applies after scaling.`],[
  `浏览器把这个数舍入到每种格式能表示的最近值，正好居中时取偶数那档（OCP 规范的舍入方式）。`,
  `FP4 E2M1 只能表示 ${f4.join('、')}；FP8 E4M3 在每两个 2 的幂之间有 ${f8} 档，BF16 有 ${b16} 档。每过一个 2 的幂，间距翻倍，所以数越大，绝对误差越大。`,
  `真实的 FP8/FP4 推理会先把每一块数值除以一个共用的缩放系数（MXFP8 和 MXFP4 每 32 个值一个，NVFP4 每 16 个值一个），这里的网格是缩放之后的。`]);};
A.registerTeaser('quantization',{render(el){
  const lab=tx('Value to store','要存的数');
  const rows=QZ_F.map(f=>{const g=qzGrid(f,6).map(v=>`<i style="left:${pos(v/6*100)}"></i>`).join('');
    return `<div class="tz-quant-row" data-f="${f.id}"><span class="tz-quant-name mono">${f.name}</span><div class="tz-quant-line${f.e>4?' dense':''}">${g}<b class="tz-quant-err"></b><b class="tz-quant-orig"></b><b class="tz-quant-dot"></b></div><span class="tz-quant-val mono"></span></div>`;}).join('');
  el.innerHTML=`<div class="tz tz-quant"><label class="tz-lab"><span>${lab}</span><output class="mono"></output></label>`+
    `<input type="range" min="0" max="${QZ_MAX}" step="1" value="${qzV}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-quant-vis" role="img">${rows}<div class="tz-quant-row"><span></span><div class="tz-quant-axis">${[0,1,2,3,4,5,6].map(v=>`<span style="left:${pos(v/6*100)}">${v}</span>`).join('')}</div><span></span></div></div>`+
    leg([['tz-quant-sw-orig',tx('dashed: your value','虚线：你的数')],['tz-quant-sw-dot',tx('dot: stored value','圆点：存下的数')],['tz-quant-sw-err',tx('bar: error','短条：误差')],
         ['tz-quant-sw-tick',tx('tick: a value the format can store','刻度：格式能表示的值')],['tz-quant-sw-band',tx('band: too dense to draw','色带：密得画不开')]])+
    `<p class="tz-read"></p>${notes(3)}</div>`;
  const q=s=>el.querySelector(s), inp=q('input'), out=q('output'), vis=q('.tz-quant-vis'), rd=q('.tz-read');
  setNotes(el,QZ_NOTES());
  const upd=()=>{qzV=+inp.value; A.ui.setRange(inp); const r=qzCompute(qzV), vs=r.v.toFixed(2); out.textContent=vs; inp.setAttribute('aria-valuetext',vs);
    r.rows.forEach(row=>{const box=q(`.tz-quant-row[data-f="${row.id}"]`), x=r.v/6*100, y=row.q/6*100;
      box.querySelector('.tz-quant-orig').style.left=pos(x); box.querySelector('.tz-quant-dot').style.left=pos(y);
      const er=box.querySelector('.tz-quant-err'); er.style.left=pos(Math.min(x,y)); er.style.width=pos(Math.abs(x-y));
      box.querySelector('.tz-quant-val').textContent=storedShort(row.q);});
    vis.setAttribute('aria-label',tx(`${vs} on each format's grid: `,`${vs} 在各格式网格上的位置：`)+r.rows.map(w=>`${w.name} ${stored(w.q)}`).join(', '));
    const part=(w,first)=>{const val=`<b>${stored(w.q)}</b>`, tie=w.tie?tx(', a tie, rounded to even','，正好居中，取偶数档'):'';
      if(w.err===0) return first?tx(`stays ${val} (exact)`,`还是 ${val}（没有误差）`):tx(`${val} (exact)`,`${val}（没有误差）`);
      if(w.q===0) return tx(`becomes ${val} (value lost${tie})`,`变成 ${val}（这个数丢了${tie}）`);
      return first?tx(`becomes ${val} (${pct2(w.err)}% error${tie})`,`变成 ${val}（误差 ${pct2(w.err)}%${tie}）`):tx(`${val} (${pct2(w.err)}% error${tie})`,`${val}（误差 ${pct2(w.err)}%${tie}）`);};
    const [f4,f8,b16]=r.rows;
    rd.innerHTML=r.rows.every(w=>w.err===0)?tx(`${vs} is stored exactly in FP4, FP8 and BF16: it sits on all three grids.`,`${vs} 在 FP4、FP8 和 BF16 里都能精确存下，它正好落在三种格式的网格上。`)
      :tx(`${vs} stored in FP4 (E2M1) ${part(f4,true)}; in FP8 (E4M3), ${part(f8)}; in BF16, ${part(b16)}.`,
          `${vs} 存成 FP4（E2M1）${part(f4,true)}；存成 FP8（E4M3）是 ${part(f8)}；存成 BF16 是 ${part(b16)}。`);
    };
  inp.addEventListener('input',upd); inp.addEventListener('change',()=>{flash(rd); A.announce(rd.textContent);}); upd();
}});

/* =====================================================================
   5. spec-decoding: expected tokens per target pass = (1 − α^(k+1)) / (1 − α); speedup = that / (k·c + 1)   (Leviathan et al., 2023)
   α is the chance that a guess is accepted given the earlier ones were (the paper's α = E(β) under its i.i.d. assumption)
   ===================================================================== */
const SD={k:4,c:0.17,min:50,max:95,step:5}; let sdA=80;   // α in percent
function sdCompute(p){const a=p/100, k=SD.k, c=SD.c, E=(1-Math.pow(a,k+1))/(1-a), slots=[];
  for(let j=0;j<=k;j++) slots.push(Math.pow(a,j));        // token j+1 of a pass appears only if the first j guesses were accepted
  return {p,a,k,c,E,kept:E-1,speed:E/(k*c+1),slots};}
// notes: 4 sentences (guide R11.5), the same for every α; the guess count and the draft's cost come from SD
function sdNotes(){const k=SD.k, c=pctOf(SD.c);
  return tx([
    `α is the chance the next guess is accepted when the earlier ones were; of ${k} guesses, α + α² + α³ + α⁴ are kept on average, plus one token the target model generates itself.`,
    `Speed assumes a draft step takes about ${c} as long as a target step (a 1B draft for the 8B at batch 1, priced like the serving labs), that acceptances are independent, and that one target pass checks all ${k+1} tokens.`,
    `The output keeps exactly the target model’s distribution: at the first rejected guess the target resamples a replacement token and drops the later guesses. Source: Leviathan, Kalman and Matias, ICML 2023 (arXiv 2211.17192).`],[
    `α 是前面的猜测都被接受时，下一个猜测被接受的概率；${k} 个猜测里平均留下 α + α² + α³ + α⁴ 个，再加上目标模型自己生成的 1 个 token。`,
    `速度按 draft 模型每步耗时约为目标模型的 ${c} 估算（1B 的 draft 配 8B 的目标模型、batch 为 1，按 Serving 各 Lab 的步时模型算）；还假设各次接受相互独立，目标模型跑一次就能检查 ${k+1} 个 token。`,
    `输出分布和目标模型自己生成的完全一样：遇到第一个被拒绝的猜测，目标模型重新采样一个 token 替换它，后面的猜测全部丢弃。来源：Leviathan、Kalman、Matias，ICML 2023（arXiv 2211.17192）。`]);}
A.registerTeaser('spec-decoding',{render(el){
  const lab=tx('Chance each guess is accepted, given the earlier ones were (α)','每个猜测被接受的概率 α（前面都被接受时）');
  const slots=cls=>`<div class="tz-spec-slots ${cls}">${Array.from({length:SD.k+1},()=>'<span><i></i></span>').join('')}</div>`;
  el.innerHTML=`<div class="tz tz-spec"><label class="tz-lab"><span>${lab}</span><output class="mono"></output></label>`+
    `<input type="range" min="${SD.min}" max="${SD.max}" step="${SD.step}" value="${sdA}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-spec-vis" role="img"><div class="tz-spec-cap">${tx('Tokens per target pass (one box per token)','目标模型每跑一次产出的 token（每格一个）')}</div>`+
    `<div class="tz-spec-grid"><span class="tz-spec-name">${tx('No drafts','不猜')}</span>${slots('tz-spec-s0')}<b class="tz-spec-v mono">1</b>`+
    `<span class="tz-spec-name">${tx('4-token drafts','每轮猜 4 个')}</span>${slots('tz-spec-s1')}<b class="tz-spec-v tz-spec-e mono"></b>`+
    `<span aria-hidden="true"></span><div class="tz-spec-pr mono"></div><span aria-hidden="true"></span></div></div>`+
    leg([['tz-spec-sw-fill',tx('fill: chance the token is produced','填充：这个 token 被产出的概率')],['tz-spec-sw-empty',tx('dashed box: not produced','虚线框：不产出')]])+
    `<p class="tz-read"></p>${notes(3)}</div>`;
  const q=s=>el.querySelector(s), inp=q('input'), out=q('output'), vis=q('.tz-spec-vis'), rd=q('.tz-read');
  q('.tz-spec-s0').querySelectorAll('i').forEach((f,j)=>{f.style.height=j?'0%':'100%';});
  const upd=()=>{sdA=+inp.value; A.ui.setRange(inp); const r=sdCompute(sdA), E=num(r.E,2), K=num(r.kept,2), S=num(r.speed,1);
    out.textContent=sdA+'%'; inp.setAttribute('aria-valuetext',out.textContent);
    q('.tz-spec-s1').querySelectorAll('i').forEach((f,j)=>{f.style.height=pos(r.slots[j]*100);});
    q('.tz-spec-e').textContent=E;
    q('.tz-spec-pr').innerHTML=r.slots.map(s=>`<span>${num(s*100,0)}%</span>`).join('');
    vis.setAttribute('aria-label',tx(`Tokens per target pass: 1 without drafts, ${E} with 4-token drafts at α = ${sdA}%.`,`目标模型每跑一次的 token：不猜 1 个；α = ${sdA}%、每轮猜 4 个时平均 ${E} 个。`));
    rd.innerHTML=tx(`At α = ${sdA}%, the target model keeps <b>${K}</b> of ${SD.k} drafted tokens on average: <b>${E}</b> tokens per target pass instead of 1, <b>${S} times</b> as fast as without drafts.`,
      `α = ${sdA}% 时，目标模型平均留下 draft 猜的 ${SD.k} 个 token 里的 <b>${K}</b> 个：每跑一次产出 <b>${E}</b> 个 token（不猜时 1 个），速度约为不猜时的 <b>${S} 倍</b>。`);
    setNotes(el,sdNotes());};
  inp.addEventListener('input',upd); inp.addEventListener('change',()=>{flash(rd); A.announce(rd.textContent);}); upd();
}});

/* ---------- read-only hook for tests ---------- */
window.__tzA=Object.freeze({
  consts:Object.freeze({peak:H100.peak,bw:H100.bw,mfu:EFF.mfu,mbu:EFF.mbu,P_TOTAL,P_LAYERS,P_EMB,P_HEAD,KV_TOK,W_READ,PD_NEXT}),
  pd:Object.freeze({options:PD_N.slice(),compute:pdCompute,state:()=>pdI}),
  cp:Object.freeze({options:CP_C.slice(),setup:Object.assign({},CP),compute:cpCompute,state:()=>cpI}),
  pc:Object.freeze({options:PC_N.slice(),setup:Object.assign({},PC),compute:pcCompute,state:()=>pcI}),
  qz:Object.freeze({max:QZ_MAX,compute:qzCompute,grid:f=>qzGrid(QZ_F.find(x=>x.id===f),6),state:()=>qzV}),
  sd:Object.freeze({setup:Object.assign({},SD),compute:sdCompute,state:()=>sdA})
});
})();
