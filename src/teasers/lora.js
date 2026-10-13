/* Teaser for the lora lab's home tile: one segmented control picks the method (full fine-tuning, LoRA, QLoRA) for Llama-3.1-8B, one
   sequence of 4,096 tokens, no recomputation, LoRA of rank 16 on all 7 matrices of every layer: the lab's own engine and numbers
   (labs/lora.js, specs/lora.md). One horizontal stacked bar on a fixed axis with a dashed line at one 80 GB H100; the part beyond it
   is hatched red. Bytes: full fine-tuning 18 per parameter; LoRA: frozen weights 2 (BF16), LoRA parameters 16 (FP32 weight, gradient,
   Adam's m and v); QLoRA: the layers' linear weights in NF4 with double quantization, 0.5 + 1/64 + 4/(64 x 256) bytes each, the
   embedding table, LM head and norms in BF16. Activations and logits as labs/training-memory.js. GB = 10^9 bytes.
   The tile does not show the share of trained parameters (R3.8: part 3 of the lab asks for it).
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's choice. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
/* Llama-3.1-8B shapes (app.js MODELS) */
const P=8030261248, NL=32, D=4096, KV=1024, F=14336, V=128256, SEQ=4096, R=16;
const MATS=[[D,D],[D,KV],[D,KV],[D,D],[D,F],[D,F],[F,D]];
const LIN=NL*MATS.reduce((a,m)=>a+m[0]*m[1],0), LORA=NL*R*MATS.reduce((a,m)=>a+m[0]+m[1],0);
const NF4=0.5+1/64+4/(64*256), ACT=SEQ*NL*167936, LOGITS=SEQ*V*4, CAP=80e9, AXIS=180e9;
const METHS=['full','lora','qlora'];
let cur='lora';
function model(me){
  const b=me==='full'?{w:2*P,q:0,s:16*P}:me==='lora'?{w:2*P,q:0,s:16*LORA}:{w:2*(P-LIN),q:LIN*NF4,s:16*LORA};
  b.a=ACT+LOGITS; b.tot=b.w+b.q+b.s+b.a; b.fits=b.tot<=CAP; return b;}
const G=v=>(v/1e9).toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});
const name=me=>({full:tx('Full fine-tuning','全量微调'),lora:'LoRA',qlora:'QLoRA'}[me]);
function readHTML(me){const m=model(me);
  if(me==='full') return tx(`Full fine-tuning of the 8B with mixed-precision AdamW on 4,096 tokens needs ${G(m.tot)} GB, <b>more than one 80 GB H100 holds</b>.`,
                            `8B 用混合精度 AdamW 全量微调一条 4,096 个 token 的序列，要 ${G(m.tot)} GB，<b>一张 80 GB 的 H100 放不下</b>。`);
  if(me==='lora') return tx(`LoRA needs ${G(m.tot)} GB and <b>fits one 80 GB H100</b>: the frozen weights carry no gradients or optimizer state.`,
                            `LoRA 只要 ${G(m.tot)} GB，<b>一张 80 GB 的 H100 放得下</b>。冻结的权重不带梯度，也不带优化器状态。`);
  return tx(`QLoRA stores the frozen weights in 4 bits and needs ${G(m.tot)} GB, <b>most of it activations</b>.`,
            `QLoRA 把冻结的权重存成 4 位，只要 ${G(m.tot)} GB，<b>其中大半是激活</b>。`);}
function notes(){
  return [tx('Formula: Llama‑3.1‑8B, one sequence of 4,096 tokens, no recomputation; LoRA of rank 16 on all 7 matrices of every layer. Full fine-tuning with mixed-precision AdamW keeps 18 bytes per parameter. LoRA keeps the frozen weights in BF16, 2 bytes each, and 16 bytes per LoRA parameter. QLoRA stores the layers’ linear weights in 4-bit NF4, about 0.52 bytes each with their quantization constants.',
             '公式：Llama‑3.1‑8B，一条 4,096 个 token 的序列，不重算；秩 16 的 LoRA 加在每层全部 7 个矩阵上。用混合精度 AdamW 全量微调，每个参数 18 字节。LoRA 冻结的权重用 BF16，每个 2 字节，每个 LoRA 参数 16 字节。QLoRA 把各层线性层的权重存成 4 位的 NF4，连同量化常数每个约 0.52 字节。'),
    tx('In LoRA and QLoRA the gray part is B and A with their gradients and Adam’s m and v: 0.7 GB. Activations and logits are counted as in the lab Training memory.',
       'LoRA 和 QLoRA 那一条里，灰色是 B 和 A，连同它们的梯度和 Adam 的 m、v，共 0.7 GB。激活和 logits 的算法和“训练显存”Lab 一样。'),
    tx('GB here is 10⁹ bytes, and the H100’s 80 GB is read as 80 × 10⁹ bytes, as in the lab. The CUDA context and temporary buffers are left out.',
       '这里的 GB 是 10⁹ 字节，H100 的 80 GB 也按 80 × 10⁹ 字节算，和 Lab 一致。CUDA 上下文和临时缓冲区没有算。')];}
const pct=v=>Math.max(0,Math.min(100,v/AXIS*100));
function visHTML(me){const m=model(me), cx=pct(CAP);
  let acc=0, segs='';
  [['w',m.w],['q',m.q],['s',m.s],['a',m.a]].forEach(([k,v])=>{if(v<=0) return; const w=Math.max(k==='s'?0.8:0,pct(v)); segs+=`<i class="tz-lora-seg ${k}" style="left:${pct(acc).toFixed(2)}%;width:${w.toFixed(2)}%"></i>`; acc+=v;});
  const ov=m.fits?'':`<i class="tz-lora-ov" style="left:${cx.toFixed(2)}%;width:${(pct(m.tot)-cx).toFixed(2)}%"></i>`;
  return `<div class="tz-lora-grid">`+
    `<span class="tz-lora-lab">${esc(`${name(me)} ${G(m.tot)} GB`)}</span>`+
    `<span class="tz-lora-trk">${segs}${ov}<b class="tz-lora-cap" style="left:${cx.toFixed(2)}%"></b></span>`+
    `<em class="tz-lora-capl" style="left:${cx.toFixed(2)}%">${esc(tx('one H100: 80 GB','一张 H100：80 GB'))}</em>`+
    `</div>`;}
function helpOf(me){const m=model(me);
  return tx(`${name(me)} of Llama‑3.1‑8B, 4,096 tokens: ${G(m.tot)} GB. Weights ${G(m.w+m.q)} GB (blue${m.q?', hatched where stored in 4 bits':''}), gradients and optimizer state ${G(m.s)} GB (gray), activations and logits ${G(m.a)} GB (light gray). The dashed line is one 80 GB H100${m.fits?'.':'; the red hatching is the part over it.'}`,
            `${name(me)}，Llama‑3.1‑8B，4,096 个 token：${G(m.tot)} GB。权重 ${G(m.w+m.q)} GB（蓝色${m.q?'，斜纹是存成 4 位的部分':''}），梯度和优化器状态 ${G(m.s)} GB（灰色），激活和 logits ${G(m.a)} GB（浅灰）。虚线是一张 80 GB 的 H100${m.fits?'。':'，红色斜纹是超出的部分。'}`);}
A.registerTeaser('lora',{fid:'formula',render(el){
  el.innerHTML=`<div class="tz tz-lora">`+
    `<div class="seg-ctl tz-lora-segc" role="group" aria-label="${esc(tx('Fine-tuning method','微调方法'))}">`+
      METHS.map(me=>`<button type="button" data-v="${me}" aria-pressed="${me===cur}">${esc(name(me))}</button>`).join('')+`</div>`+
    `<div class="tz-vis tz-lora-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw w" aria-hidden="true"></i>${esc(tx('weights','权重'))}</span><span><i class="sw tz-lora-swq" aria-hidden="true"></i>${esc(tx('4-bit weights','4 位权重'))}</span><span><i class="sw tz-lora-sws" aria-hidden="true"></i>${esc(tx('gradients, optimizer state','梯度和优化器状态'))}</span><span><i class="sw tz-lora-swa" aria-hidden="true"></i>${esc(tx('activations, logits','激活和 logits'))}</span><span><i class="sw over" aria-hidden="true"></i>${esc(tx('over the GPU','超出这张卡'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const seg=el.querySelector('.tz-lora-segc'), vis=el.querySelector('.tz-lora-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{
    seg.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===cur)));
    vis.innerHTML=visHTML(cur);
    const help=helpOf(cur); vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(cur); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  seg.addEventListener('click',ev=>{const b=ev.target.closest('button[data-v]'); if(!b||b.dataset.v===cur) return; cur=b.dataset.v; upd(true);});
  upd(false);
}});
/* read-only hook for tests/lora_numbers.py */
window.__tzLora=Object.freeze({METHS:METHS.slice(),model:me=>({...model(me)}),read:me=>readHTML(me),help:me=>helpOf(me),LORA,LIN,NF4});
})();
