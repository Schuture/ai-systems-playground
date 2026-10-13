/* The MoE lab's home tile answers the lab's question, "How can a 47B model run like a 13B one?", for 1, 4, 16 or 64 requests decoding
   together. One layer's 8 experts show why (the filled part: the experts a decode step reads, on average); three bars on one shared scale
   give the answer: the time of one decode step, which is each request's time per token, for Mixtral-8x7B and two hypothetical dense models
   that keep Mixtral's attention (so its KV cache) and use every parameter for every token: one with Mixtral's 46.7B parameters (the lab's
   part 6 "Dense 46.7B") and one with the 12.9B that one Mixtral token uses (the question's "13B"; the lab: one request "decodes like a
   dense model of 12.9B"). With 1 request Mixtral reads 2 of 8 experts per layer and its step equals the dense 12.9B's, 3.6 times as fast
   as the dense 46.7B; as requests are added their tokens pick different experts, and at 64 Mixtral reads all 8 and its step equals the
   dense 46.7B's (every step here is limited by its memory reads, not its arithmetic). At 16 and 64 requests a second sentence says
   what it still saves, its arithmetic (each token computes with 12.9B parameters), so the reader does not conclude that MoE is useless
   when many users share a GPU (R6.26, home review v4 A #6).
   Same engine as labs/moe.js (decode(), touched(), cmpModels()'s d47; tests/moe_numbers.py checks they agree): FP8 weights on one
   H100 SXM, 2K tokens of context per request, uniform routing, prefill-decode's step model (the longer of the arithmetic at 50% of the
   dense BF16 peak and the memory reads at 60% of the peak bandwidth). The experts are weights, so blue; a time is not, so the bars are
   grayscale (R5.6). No memory size on the tile: the lab's part 2 asks the learner to predict it (R3.8).
   window.__tzMoe is a read-only hook for tests/moe_numbers.py and tests/moe_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc}=A;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const MODEL='Mixtral‑8x7B';                                   // U+2011 no-break hyphens: a model name never breaks
/* Mixtral-8x7B config.json: hidden 4096, 32 layers, 8 KV heads × 128, intermediate 14336, 8 experts, 2 per token, vocab 32000, untied LM head */
const MX={d:4096,L:32,KV:8,hd:128,f:14336,E:8,k:2,V:32000};
const P_EXPERT=3*MX.d*MX.f, P_EMB=MX.V*MX.d;
const P_SREAD=MX.L*(2*MX.d*MX.d+2*MX.d*MX.KV*MX.hd+2*MX.d+MX.d*MX.E)+MX.d+MX.V*MX.d;   // attention, norms, routers, final norm, LM head
const P_ALL=P_SREAD+MX.L*MX.E*P_EXPERT;                       // all 46.7B but the embedding table, which a token only looks up
const P_MULT=P_SREAD+MX.L*MX.k*P_EXPERT;                      // one token's 12.9B active parameters but the embedding table
const KV_TOK=2*MX.L*MX.KV*MX.hd*2, CTX=2048;                  // 128 KiB of BF16 K and V per token; 2K tokens per request
const FS=1979e12/2*0.5, BS=3.35e12*0.6;                       // 50% of the dense BF16 peak, 60% of 3.35 TB/s
const ATT=4*MX.L*MX.d*(CTX+1);                                // one new token's attention over its context: the same in all three models
const touched=B=>MX.E*(1-Math.pow(1-MX.k/MX.E,B));            // experts per layer that B tokens pick, on average (uniform routing)
/* one decode step for B requests: each request's token multiplies `mult` parameters (2 FLOPs each) and does its attention; the step reads
   `w` bytes of FP8 weights and every request's KV cache, and takes the longer of the two */
function stepOf(B,mult,w){const fl=B*(2*mult+ATT), by=w+B*CTX*KV_TOK; return {fl,by,tc:fl/FS,tm:by/BS,t:Math.max(fl/FS,by/BS)};}
function step(B){const T=touched(B);
  return {B,T,mx:stepOf(B,P_MULT,P_SREAD+MX.L*T*P_EXPERT),d47:stepOf(B,P_ALL,P_ALL),d13:stepOf(B,P_MULT,P_MULT)};}
const NS=[1,4,16,64]; let ni=0;
const IDS=['d47','mx','d13'];                                 // top to bottom: Mixtral between the two dense models
const SCALE=Math.max(...NS.map(B=>{const r=step(B); return Math.max(...IDS.map(id=>r[id].t));}));   // one scale for every setting
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const shown=s=>Math.round(s*1e5)/100;                         // a time in ms as the tile shows it (2 decimals, as in the lab)
const NB='\u00a0';                                            // no-break space: “2 of 8”, “2.4 倍” and “H100 SXM” stay on one line
const nw=s=>`<span class="tz-moe-nw">${s}</span>`;              // a model's name never breaks, in Chinese either
const name=id=>id==='mx'?MODEL:id==='d47'?tx('Dense 46.7B','稠密 46.7B'):tx('Dense 12.9B','稠密 12.9B');
const tip=id=>id==='mx'?tx(`${MODEL} has 46.7B parameters, with 8 experts in every layer; each token uses 2 of them per layer, 12.9B parameters in all.`,`${MODEL} 一共有 46.7B 个参数，每层 8 个专家；每个 token 每层只用其中 2 个，合计 12.9B 个参数。`)
  :id==='d47'?tx(`Hypothetical: a dense model with ${MODEL}’s 46.7B parameters and its attention, so the same KV cache; every token uses every parameter.`,`假想的模型：参数和 ${MODEL} 一样多（46.7B 个），attention 也一样，所以 KV cache 相同；每个 token 都用全部参数。`)
  :tx(`Hypothetical: a dense model with 12.9B parameters, as many as one ${MODEL} token uses, and ${MODEL}’s attention, so the same KV cache; every token uses every parameter.`,`假想的模型：12.9B 个参数，和一个 ${MODEL} token 用到的一样多；attention 和 KV cache 与 ${MODEL} 相同，每个 token 都用全部参数。`);
/* which comparison the sentence makes, from the times as shown (R6.10: the words follow the numbers) */
function verdict(B){const r=step(B), mx=shown(r.mx.t), a=shown(r.d47.t), z=shown(r.d13.t);
  return mx===z?{k:'as13',x:fixd(a/mx,1)}:mx===a?{k:'as47'}:mx/a>=0.98?{k:'near47'}:{k:'times13',x:fixd(mx/z,1)};}
function readHTML(B){const r=step(B), v=verdict(B), T=fixd(r.T,1);
  const d13=nw(tx('dense 12.9B','稠密 12.9B')), d47=nw(tx('dense 46.7B','稠密 46.7B'));
  if(B===1&&v.k==='as13') return tx(`1 request: ${MODEL} reads <b>2${NB}of${NB}8</b> experts per layer, so it decodes <b>as fast as a ${d13}</b>, ${v.x} times a ${nw('dense 46.7B’s')} speed.`,
    `1 个请求时，${MODEL} 每层只读 8 个专家里的 <b>2${NB}个</b>，所以 decode 起来<b>和${d13} 一样快</b>，速度是${d47} 的 ${v.x}${NB}倍。`);
  const res={as13:tx(`decodes <b>as fast as a ${d13}</b>`,`decode 起来<b>和${d13} 一样快</b>`),
    times13:tx(`takes <b>${v.x} times as long</b> per token as a ${d13}`,`每个 token 的耗时是${d13} 的 <b>${v.x}${NB}倍</b>`),
    near47:tx(`decodes <b>almost as slowly as a ${d47}</b>`,`decode 起来<b>几乎和${d47} 一样慢</b>`),
    as47:tx(`decodes <b>as slowly as a ${d47}</b>`,`decode 起来<b>和${d47} 一样慢</b>`)}[v.k];
  /* R6.26: where the batch reads (almost) every expert, the second sentence says what MoE still saves, so “as slowly” does not read as
     “MoE is useless”. Every step on this tile is limited by its memory reads (tests/moe_numbers.py), so less arithmetic does not
     shorten it; at 16 requests the reads still differ a little (7.9 of 8 experts), hence “hardly any” (R6.10) */
  const still={near47:tx(`It <b>still saves arithmetic</b>, but hardly any memory reads: each token computes with only 12.9B parameters.`,
                         `它<b>省下的仍是计算</b>，读显存几乎没省：每个 token 只用 12.9B 个参数来计算。`),
               as47:tx(`It <b>still saves arithmetic</b>, not memory reads: each token computes with only 12.9B parameters.`,
                       `它<b>省下的仍是计算</b>，不是读显存：每个 token 只用 12.9B 个参数来计算。`)}[v.k];
  return tx(`${B} requests pick different experts, so ${MODEL} reads <b>${T}${NB}of${NB}8</b> per layer and ${res}.`,
    `${B} 个请求各选各的专家，所以 ${MODEL} 每层要读 8 个里的 <b>${T}${NB}个</b>，${res}。`)+(still?tx(' ','')+still:'');}
function bars(B){const r=step(B); return IDS.map(id=>({id,name:name(id),t:r[id].t,text:fixd(shown(r[id].t),2)+' ms',w:r[id].t/SCALE*100}));}
function aria(B){const r=step(B), b=bars(B), v=id=>b.find(x=>x.id===id).text;
  return tx(`${B===1?'1 request':`${B} requests`} decoding together: ${MODEL} reads ${fixd(r.T,1)} of 8 experts per layer on average. Time per token: dense 46.7B ${v('d47')}, ${MODEL} ${v('mx')}, dense 12.9B ${v('d13')}.`,
    `${B} 个请求一起 decode：${MODEL} 每层平均读 8 个专家里的 ${fixd(r.T,1)} 个。每个 token 的耗时依次是稠密 46.7B ${v('d47')}，${MODEL} ${v('mx')}，稠密 12.9B ${v('d13')}。`);}
/* the notes behind "Assumptions & sources" (R11.5: at most 4 short notes, one question each): how a time is computed, what a step reads,
   how many experts a batch reads, the sources. With them a reader can redo every time on the tile (review B-08): bytes ÷ (60% of
   3.35 TB/s). The 50% arithmetic rate stays in the lab: no time here depends on it. That the dense models are hypothetical is in their
   names' tooltips and the legend. Plain text, no "^" (review A #8). One array per language, each note through zfix on its own (B1) */
function notes(){
  const N={en:[`A decode step gives every request one token and lasts as long as its memory reads, at an assumed 60% of an H100${NB}SXM’s 3.35${NB}TB/s: here they always take longer than its arithmetic.`,
    `A step reads FP8 weights (of the embedding table only one row per request) and each request’s KV cache of 2K tokens, in BF16 at 128${NB}KiB per token in all three models.`,
    `Each token picks 2 of a layer’s 8 experts, here at random (uniform routing), so N requests read 8 × (1 − (6/8) to the power N) experts per layer on average.`,
    `Sources: ${MODEL}’s config.json; Mistral AI’s Mixtral announcement; NVIDIA’s H100 datasheet.`],
   zh:[`一个 decode 步给每个请求生成一个 token，耗时就是读显存的时间（按一张 H100${NB}SXM 峰值带宽 3.35${NB}TB/s 的 60% 估算），因为这里读显存总比计算久。`,
    `每一步读 FP8 权重（embedding 表不整表读，每个请求只查其中一行）和每个请求 2K 个 token 的 KV cache；KV cache 按 BF16 存，三个模型都是每个 token 128${NB}KiB。`,
    `每个 token 在每层从 8 个专家里选 2 个，这里假设随机选（均匀路由），所以 N 个请求每层平均读 8 ×（1 − 6/8 的 N 次方）个专家。`,
    `来源：${MODEL} 的 config.json；Mistral AI 发布 Mixtral 的文章；NVIDIA H100 规格表。`]};
  return A.lang()==='zh'?N.zh.map(zfix):N.en;}
A.registerTeaser('moe',{fid:'formula',render(el){
  const lab=tx('Requests decoding together','一起 decode 的请求数');
  el.innerHTML=`<div class="tz tz-moe">${A.ui.seg({key:'tz-moe',label:lab,options:NS.map(n=>[String(n),String(n)]),value:String(NS[ni]),inline:false})}`+
    `<div class="tz-vis tz-moe-vis" role="img"><div class="tz-moe-cap" aria-hidden="true">${esc(tx('A layer’s 8 experts: read per step (average)','一层的 8 个专家：每步平均读几个'))}</div>`+
    `<div class="tz-moe-ex" aria-hidden="true">${Array.from({length:MX.E},()=>'<span><i></i></span>').join('')}</div>`+
    `<div class="tz-moe-cap" aria-hidden="true">${esc(tx('Each request’s time per token','每个请求每个 token 的耗时'))}</div>`+
    `<div class="tz-moe-tm">${IDS.map(id=>`<span class="tz-moe-n tz-moe-${id}" tabindex="0" data-tipx="${esc(tip(id))}">${esc(name(id))}</span><span class="tz-moe-tr tz-moe-${id}" aria-hidden="true"><i></i></span><span class="tz-moe-v tz-moe-${id}" aria-hidden="true"></span>`).join('')}</div></div>`+
    `<div class="tz-leg"><span tabindex="0" data-tipx="${esc(tx('An expert is a block of weights; the filled part is how many of a layer’s 8 experts a decode step reads, on average.','专家是一块权重；填满的部分表示一个 decode 步平均读一层 8 个专家里的几个。'))}"><i class="sw w"></i>${esc(tx('experts read','读到的专家'))}</span>`+
    `<span><i class="sw tz-moe-sw-out"></i>${esc(tx('not read','没读的'))}</span><span><i class="sw tz-moe-sw-mx"></i>${MODEL}</span><span><i class="sw tz-moe-sw-dn"></i>${esc(tx('hypothetical dense','假想的稠密模型'))}</span></div>`+
    /* the sentence sits on hidden copies of every setting's, so it keeps the longest one's height at any width and switching never
       moves what is below it (R7.9; only 16 and 64 requests have a second sentence) */
    `<div class="tz-moe-rd"><p class="tz-read"></p>${NS.map(B=>`<p class="tz-moe-ghost" aria-hidden="true">${readHTML(B)}</p>`).join('')}</div>`+
    `${notes().map(()=>'<p class="tz-note"></p>').join('')}</div>`;
  const q=s=>el.querySelector(s), seg=q('.seg-ctl'), rd=q('.tz-read'), vis=q('.tz-moe-vis');
  const upd=()=>{const B=NS[ni], r=step(B); A.ui.setPressed(seg,String(B));
    el.querySelectorAll('.tz-moe-ex i').forEach((x,i)=>{x.style.width=(Math.max(0,Math.min(1,r.T-i))*100).toFixed(1)+'%';});
    bars(B).forEach(b=>{q(`.tz-moe-tr.tz-moe-${b.id} i`).style.width=b.w.toFixed(2)+'%'; q(`.tz-moe-v.tz-moe-${b.id}`).textContent=b.text;});
    vis.setAttribute('aria-label',aria(B)); rd.innerHTML=readHTML(B);
    const ns=notes(); el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=ns[i]||'';});};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||!seg.contains(b)) return; const i=NS.indexOf(+b.dataset.v); if(i<0) return; ni=i; upd(); A.announce(rd.textContent);});
  upd();
}});
window.__tzMoe=Object.freeze({step:B=>JSON.parse(JSON.stringify(step(B))),bars:B=>JSON.parse(JSON.stringify(bars(B))),read:B=>readHTML(B),verdict:B=>verdict(B),
  notes:()=>notes(),NS:NS.slice(),CTX,SCALE});
})();
