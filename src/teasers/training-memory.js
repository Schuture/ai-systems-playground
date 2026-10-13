/* Teaser for the training-memory lab's home tile: one slider picks a model, four real Llama models (1B, 3B, 8B, 70B). It opens at the
   8B, the lab's own Llama‑3.1‑8B, so the tile's title, the slider and the sentence all name the same 8B model (the owner's review of
   batch 8, 2026-10-07: the tile used to open at a nominal 7B under an 8B title).
   Parameter counts come from each model's config (meta-llama/llama-models models/sku_list.py): the 8B and 70B as app.js MODELS and the
   lab; the 1B as the lab's challenge (tied embedding table); the 3B summed the same way (d 3,072, 28 layers, 24/8 heads, FFN 8,192,
   tied). tests/training-memory_numbers.py re-derives all four matrix by matrix.
   Two bars on one axis, with the lab's bytes per parameter (batch8-brief.md, decision 6): inference holds the BF16 weights (2 bytes);
   training's model states take 18 bytes (BF16 weights 2, FP32 gradients 4, FP32 copy 4, m 4, v 4). A dashed line marks one 80 GB GPU;
   the part of a bar beyond it is hatched red. Activations are left out and the note says so.
   Sizes are written as R6.18 writes them, to 3 significant digits (16.1 GB, 145 GB, 8.03B); no hover expands them to exact values
   (the owner, 2026-10-07 23:37 UTC). GB = 10⁹ bytes; the GPU's 80 GB is read as 80 × 10⁹ bytes, as in the lab.
   R3.8: the tile shows the 8B's model states (about 145 GB), not the peak part 1 of the lab asks for (states + activations + logits).
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const NB='‑';
const MODS=[{k:'1B',name:`Llama${NB}3.2${NB}1B`,P:1235814400},{k:'3B',name:`Llama${NB}3.2${NB}3B`,P:3212749824},
  {k:'8B',name:`Llama${NB}3.1${NB}8B`,P:8030261248},{k:'70B',name:`Llama${NB}3.1${NB}70B`,P:70553706496}];
const DEF_IDX=2, INF=2, PARTS=[2,4,4,4,4], TRAIN=PARTS.reduce((a,b)=>a+b,0), CAP=80e9;
let idx=DEF_IDX;
const fN=n=>n.toLocaleString('en-US');
/* GB to 3 significant digits (R6.18): 2.47, 16.1, 145, 1,270 */
const rGB=b=>{const v=b/1e9; if(v>=1000){const q=Math.pow(10,Math.floor(Math.log10(v))-2); return fN(Math.round(v/q)*q);}
  const d=v>=100?0:v>=10?1:2; return v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});};
const bigP=n=>A.fmt&&A.fmt.big?A.fmt.big(n):fN(n);
function model(i){const o=MODS[i], inf=o.P*INF, tr=o.P*TRAIN, mx=Math.max(tr,CAP)*1.06;
  return {k:o.k,name:o.name,P:o.P,inf,tr,mx,ratio:TRAIN/INF,fits:tr<=CAP,infFits:inf<=CAP};}
const pct=(v,m)=>Math.max(0,Math.min(100,v/m.mx*100));
function readHTML(i){const m=model(i);
  if(m.fits) return tx(`A ${m.k} model needs ${rGB(m.inf)} GB of weights to serve. Training it needs ${rGB(m.tr)} GB for the model states alone, ${m.ratio} times as much, <b>which still fits on one 80 GB GPU</b>.`,
                       `${m.k} 模型推理时只放 ${rGB(m.inf)} GB 权重。训练时，光模型状态就要 ${rGB(m.tr)} GB，是推理时的 ${m.ratio} 倍，<b>一张 80 GB 的卡还放得下</b>。`);
  return tx(`A${m.k==='8B'?'n':''} ${m.k} model needs ${rGB(m.inf)} GB of weights to serve. Training it needs ${rGB(m.tr)} GB for the model states alone, ${m.ratio} times as much, <b>more than one 80 GB GPU holds</b>.`,
            `${m.k} 模型推理时只放 ${rGB(m.inf)} GB 权重。训练时，光模型状态就要 ${rGB(m.tr)} GB，是推理时的 ${m.ratio} 倍，<b>一张 80 GB 的卡放不下</b>。`);}
function notes(){
  return [tx('Formula: inference holds the weights, 2 bytes per parameter in BF16. Training with AdamW in mixed precision holds 18 bytes of model states per parameter: the BF16 weights’ 2, plus 4 each for the FP32 gradients, the FP32 copy of the weights, and Adam’s m and v.',
             '公式：推理只放权重，BF16 下每个参数 2 字节。用 AdamW 做混合精度训练时，模型状态每个参数 18 字节。其中 BF16 权重 2 字节，FP32 梯度、FP32 权重副本、Adam 的 m 和 v 各 4 字节。'),
    tx('The training bar shows only these model states. The activations the forward pass keeps for the backward pass come on top and grow with the tokens processed at once; the lab counts them.',
       '训练那一条只画了这些模型状态。前向留给反向用的激活还要另外加，它们随一次处理的 token 数增长。Lab 里会把它们算进去。'),
    tx('Each stop is a real Llama model, with the parameter count from its config. The 1B and 3B are Llama 3.2, the 8B and 70B Llama 3.1, and the 8B is the lab’s own model. GB here is 10⁹ bytes, and the GPU’s 80 GB is read as 80 × 10⁹ bytes, as in the lab.',
       '每一档都是真实的 Llama 模型，参数量来自它的 config。1B 和 3B 是 Llama 3.2，8B 和 70B 是 Llama 3.1，其中 8B 就是 Lab 里用的那个模型。这里的 GB 是 10⁹ 字节，GPU 的 80 GB 也按 80 × 10⁹ 字节算，和 Lab 一致。')];}
function visHTML(i){const m=model(i), cx=pct(CAP,m);
  let segs='', acc=0; PARTS.forEach((p,k)=>{const w=pct(m.P*p,m); segs+=`<i class="tz-tmem-seg ${k===0?'w':'s'}" style="left:${pct(acc,m).toFixed(2)}%;width:${w.toFixed(2)}%"></i>`; acc+=m.P*p;});
  const ov=m.fits?'':`<i class="tz-tmem-ov" style="left:${cx.toFixed(2)}%;width:${(pct(m.tr,m)-cx).toFixed(2)}%"></i>`;
  return `<div class="tz-tmem-grid">`+
    `<span class="tz-tmem-lab">${esc(tx(`inference ${rGB(m.inf)} GB`,`推理 ${rGB(m.inf)} GB`))}</span><span class="tz-tmem-trk"><i class="tz-tmem-seg w" style="left:0;width:${pct(m.inf,m).toFixed(2)}%"></i></span>`+
    `<span class="tz-tmem-lab">${esc(tx(`training ${rGB(m.tr)} GB`,`训练 ${rGB(m.tr)} GB`))}</span><span class="tz-tmem-trk">${segs}${ov}</span>`+
    `<span class="tz-tmem-capbox"><b class="tz-tmem-cap" style="left:${cx.toFixed(2)}%"></b><em class="tz-tmem-capl${cx>60?' r':''}" style="left:${cx.toFixed(2)}%">${esc(tx('one GPU: 80 GB','一张卡：80 GB'))}</em></span>`+
    `</div>`;}
/* the hover says what the chart shows (which model, what each colour is, the dashed line); its numbers are the ones on the tile */
function helpText(i){const m=model(i);
  return tx(`${m.name}, ${bigP(m.P)} parameters. Inference: ${rGB(m.inf)} GB of weights. Training: ${rGB(m.tr)} GB of model states, the BF16 weights (blue) and the FP32 gradients, FP32 copy and Adam’s m and v (gray). The dashed line is one 80 GB GPU${m.fits?'.':'. The red hatching is the part over it.'}`,
            `${m.name}，${bigP(m.P)} 个参数。推理：${rGB(m.inf)} GB 权重。训练：${rGB(m.tr)} GB 模型状态，蓝色是 BF16 权重，灰色是 FP32 梯度、FP32 权重副本，以及 Adam 的 m 和 v。虚线是一张 80 GB 的卡${m.fits?'。':'，红色斜纹是超出的部分。'}`);}
A.registerTeaser('training-memory',{fid:'formula',render(el){
  const lab=tx('Model size','模型大小');
  el.innerHTML=`<div class="tz tz-tmem">`+
    `<label class="tz-lab" for="tz-tmem-in"><span>${esc(lab)}</span><output id="tz-tmem-out">${MODS[idx].k}</output></label>`+
    `<input type="range" id="tz-tmem-in" min="0" max="${MODS.length-1}" step="1" value="${idx}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-tmem-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw w" aria-hidden="true"></i>${esc(tx('BF16 weights','BF16 权重'))}</span><span><i class="sw tz-tmem-sws" aria-hidden="true"></i>${esc(tx('FP32 gradients, FP32 copy, Adam’s m and v','FP32 梯度、FP32 权重副本、Adam 的 m 和 v'))}</span><span><i class="sw over" aria-hidden="true"></i>${esc(tx('over the GPU','超出这张卡'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-tmem-in'), out=el.querySelector('#tz-tmem-out'), vis=el.querySelector('.tz-tmem-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const m=model(idx); out.textContent=m.k; inp.value=idx; inp.setAttribute('aria-valuetext',m.name); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visHTML(idx);
    const help=helpText(idx);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(idx); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(MODS.length-1,Math.round(+inp.value))); if(v!==idx){idx=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/training-memory_numbers.py */
window.__tzTmem=Object.freeze({MODS:MODS.map(o=>({...o})),SIZES:MODS.map(o=>o.k),DEF_IDX,PARTS:PARTS.slice(),model:i=>({...model(i)}),read:i=>readHTML(i),help:i=>helpText(i),rGB:b=>rGB(b)});
})();
