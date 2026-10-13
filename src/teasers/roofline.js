/* Teaser for the roofline lab's home tile: drag the batch of one Llama‑3.1‑8B decode step on an H100 and watch its dot climb the
   slanted roof of the roofline until it meets the flat one at the ridge (295 FLOPs per byte). One control (batch, the lab's stops
   1 to 1,024), one small visual (a log-log roofline, 120 px tall), one sentence in .tz-read, a one-line legend, plain-text notes.
   The numbers are the lab's own (labs/roofline.js): 7,504,924,672 parameters read per step (every parameter but the embedding
   table, 15.0 GB in BF16), 2 FLOPs per parameter per request, 989.5 TFLOP/s and 3.35 TB/s at 100% of the datasheet peaks.
   The sentence gives the bound and the share of the peak, not the step's time: the lab's part 2 asks the learner to predict the time
   at batch 64 (R3.8: the tile does not answer it). State lives in module scope, so a re-render keeps the learner's batch. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const MODEL='Llama‑3.1‑8B';
const PEAK=1979e12/2, BW=3.35e12, W_PAR=7504924672;
const BST=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024];
let tzB=16;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fint=v=>Math.round(v).toLocaleString('en-US');
const f1=v=>Math.round(v*10)/10;
/* the share of the peak, never rounded up to 100% (as in the lab) */
const fSh=x=>{const v=x*100; if(x>=1-1e-12) return '100%'; if(v>=99.95) return '>99.9%'; if(v>=1) return fixd(v,1)+'%'; return fixd(v,2)+'%';};
function model(B){
  const fl=2*W_PAR*B, by=2*W_PAR, I=fl/by, tc=fl/PEAK, tm=by/BW, t=Math.max(tc,tm), att=fl/t;
  return {B,fl,by,I,tc,tm,t,att,share:att/PEAK,ridge:PEAK/BW,bound:tc>=tm?'compute':'memory'};
}
function readHTML(B){const m=model(B);
  return m.bound==='memory'
    ?tx(`At batch ${fint(B)}, the weight matmuls of one ${MODEL} decode step on an H100 are memory-bound and use <b>${fSh(m.share)}</b> of the peak compute.`,
        `batch 为 ${fint(B)} 时，在 H100 上跑 ${MODEL}，一个 decode 步的权重矩阵乘法受显存带宽限制，只用上峰值算力的 <b>${fSh(m.share)}</b>。`)
    :tx(`At batch ${fint(B)}, the weight matmuls of one ${MODEL} decode step on an H100 are compute-bound and use <b>${fSh(m.share)}</b> of the peak compute.`,
        `batch 为 ${fint(B)} 时，在 H100 上跑 ${MODEL}，一个 decode 步的权重矩阵乘法受算力限制，用上峰值算力的 <b>${fSh(m.share)}</b>。`);}
function notes(B){const m=model(B);
  return [tx(`Formula: a step takes the longer of FLOPs ÷ 989.5 TFLOP/s and bytes ÷ 3.35 TB/s, because the GPU reads and computes at once. At batch ${fint(B)}: ${fixd(m.fl/1e12,2)} TFLOP and 15.0 GB, ${fint(m.I)} FLOPs per byte moved from memory; the ridge is 989.5 ÷ 3.35 ≈ 295.`,
             `公式计算：一步的耗时取 FLOP 数 ÷ 989.5 TFLOP/s 和字节数 ÷ 3.35 TB/s 中较长的那个，因为 GPU 一边读一边算。batch 为 ${fint(B)} 时：${fixd(m.fl/1e12,2)} TFLOP、15.0 GB，每从显存搬运 1 字节做 ${fint(m.I)} 个 FLOP；拐点是 989.5 ÷ 3.35 ≈ 295。`),
    tx(`A decode step reads ${MODEL}’s 7,504,924,672 parameters (all but the embedding table), 2 bytes each in BF16, and does 2 FLOPs per parameter per request. Weights only: each user’s KV cache and the activations are left out.`,
       `decode 步要读 ${MODEL} 的 7,504,924,672 个参数（除 embedding 表以外的全部），BF16 每个 2 字节；每个参数每个请求做 2 个 FLOP。只算权重：每个用户的 KV cache 和激活值都没算。`),
    tx('100% of the H100 SXM datasheet peaks (dense BF16, the datasheet’s 1,979 TFLOP/s with sparsity ÷ 2), so the roofline is the best case. The lab “Prefill vs decode” prices steps at 50% of the peak FLOP/s and 60% of the bandwidth and adds the KV cache, so its times are longer.',
       '按 H100 SXM 规格表峰值的 100% 算（稠密 BF16，即规格表上带稀疏的 1,979 TFLOP/s ÷ 2），所以 roofline 是最好情况。“Prefill 和 decode”这个 Lab 按峰值算力的 50%、带宽的 60% 计时，还加上 KV cache，所以那里的耗时更长。')];}
const X0=-0.3, X1=3.6, Y0=0, Y1=3.3;     // log10: 0.5 … 4,000 FLOPs per byte; 1 … 2,000 TFLOP/s
function svg(W,B){
  const H=120, ml=34, mr=6, mt=8, mb=26, m=model(B), pk=PEAK/1e12, bw=BW/1e12, rid=pk/bw;
  const X=I=>ml+(Math.log10(I)-X0)/(X1-X0)*(W-ml-mr), Y=tf=>H-mb-(Math.log10(tf)-Y0)/(Y1-Y0)*(H-mt-mb);
  let h='';
  [1,10,100,1000].forEach(x=>{const px=X(x); h+=`<line class="tz-rf-grid" x1="${f1(px)}" y1="${mt}" x2="${f1(px)}" y2="${H-mb}"/><text class="tz-rf-tick" x="${f1(px)}" y="${H-mb+11}" text-anchor="middle">${fint(x)}</text>`;});
  [1,10,100,1000].forEach(y=>{const py=Y(y); h+=`<line class="tz-rf-grid" x1="${ml}" y1="${f1(py)}" x2="${W-mr}" y2="${f1(py)}"/><text class="tz-rf-tick" x="${ml-4}" y="${f1(py+3)}" text-anchor="end">${fint(y)}</text>`;});
  h+=`<text class="tz-rf-ax" x="${f1((ml+W-mr)/2)}" y="${H-3}" text-anchor="middle">${esc(W<330?tx('FLOPs per byte moved from memory','每从显存搬运 1 字节做多少 FLOP'):tx('FLOPs per byte moved from memory (= batch)','每从显存搬运 1 字节做多少 FLOP（= batch）'))}</text>`;
  h+=`<text class="tz-rf-ax" x="${ml+4}" y="${mt+9}">${esc(tx('TFLOP/s','TFLOP/s'))}</text>`;
  const rx=X(rid);
  h+=`<line class="tz-rf-ridge" x1="${f1(rx)}" y1="${f1(Y(pk))}" x2="${f1(rx)}" y2="${H-mb}"/><text class="tz-rf-rlab" x="${f1(rx+3)}" y="${f1(H-mb-4)}">${esc(tx(`ridge ${fint(rid)}`,`拐点 ${fint(rid)}`))}</text>`;
  const i0=Math.pow(10,X0);
  h+=`<path class="tz-rf-roof" d="M${f1(X(i0))},${f1(Y(Math.max(1,i0*bw)))} L${f1(rx)},${f1(Y(pk))} L${f1(X(Math.pow(10,X1)))},${f1(Y(pk))}"/>`;
  /* the peak's label sits left of the ridge, above the slanted roof, where the dot never reaches; narrow tiles drop the unit (the y axis names it) */
  h+=`<text class="tz-rf-plab" x="${f1(rx-12)}" y="${f1(Y(pk)-6)}" text-anchor="end">${esc(W<290?tx('peak 989.5','峰值 989.5'):tx('peak 989.5 TFLOP/s','峰值 989.5 TFLOP/s'))}</text>`;
  const px=X(m.I), py=Y(m.att/1e12), lab=tx(`batch ${fint(B)}`,`batch ${fint(B)}`);
  h+=`<circle class="tz-rf-dot" cx="${f1(px)}" cy="${f1(py)}" r="5"/>`;
  /* the batch label stays on the dot's side of the ridge and never crosses a line: right of a dot well left of the ridge (under the slanted roof);
     for a dot just left of the ridge, below-left of it, low enough to clear the slanted roof at the label's left end; for a dot right of the ridge,
     under the flat roof, right of the ridge line (checker RF-N1) */
  const lw=lab.length*5.8+2, far=px+8+lw<=rx-3, sl=((H-mt-mb)/(Y1-Y0))/((W-ml-mr)/(X1-X0));
  h+=far?`<text class="tz-rf-dlab" x="${f1(px+8)}" y="${f1(Math.min(H-mb-4,py+14))}">${esc(lab)}</text>`
    :(px<rx?`<text class="tz-rf-dlab" x="${f1(px-4)}" y="${f1(Math.min(H-mb-4,py+(lw+4)*sl+12))}" text-anchor="end">${esc(lab)}</text>`
    :`<text class="tz-rf-dlab" x="${f1(Math.min(rx+4,W-mr-2-lw))}" y="${f1(Y(pk)+15)}">${esc(lab)}</text>`);
  return {H,h};
}
A.registerTeaser('roofline',{fid:'formula',render(el){
  const i0=Math.max(0,BST.indexOf(tzB));
  el.innerHTML=`<div class="tz tz-rf">`+
    `<label class="tz-lab" for="tz-rf-b"><span>${esc(tx('Batch: requests per decode step','batch：每个 decode 步的请求数'))}</span><output class="mono" for="tz-rf-b"></output></label>`+
    `<input type="range" id="tz-rf-b" min="0" max="${BST.length-1}" step="1" value="${i0}">`+
    `<svg class="tz-vis tz-rf-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"><span><i class="tz-rf-sw tz-rf-sw-roof" aria-hidden="true"></i>${esc(tx('thick line: the H100’s roof','粗线：H100 的屋顶'))}</span><span><i class="tz-rf-sw tz-rf-sw-ridge" aria-hidden="true"></i>${esc(tx('dashed: ridge','虚线：拐点'))}</span><span><i class="tz-rf-sw tz-rf-sw-dot" aria-hidden="true"></i>${esc(tx('dot: the decode step’s weight matmuls','圆点：decode 步的权重矩阵乘法'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('input'), out=el.querySelector('output'), vis=el.querySelector('svg'), rd=el.querySelector('.tz-read');
  const help=tx('The roofline of an H100: the most TFLOP/s a kernel can reach (up) at each number of FLOPs it does per byte moved from memory (across). The dot is one decode step’s weight matmuls; its FLOPs per byte moved equal the batch.',
                'H100 的 roofline：横向是 kernel 每从显存搬运 1 字节做多少 FLOP，纵向是它在这个位置最多能达到的 TFLOP/s。圆点是一个 decode 步的权重矩阵乘法，它每从显存搬运 1 字节做的 FLOP 数就等于 batch。');
  vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
  const draw=()=>{const W=Math.max(240,Math.round(el.clientWidth>40?el.clientWidth:340)), r=svg(W,tzB); vis.setAttribute('viewBox',`0 0 ${W} ${r.H}`); vis.setAttribute('width',W); vis.setAttribute('height',r.H); vis.innerHTML=r.h;};
  const upd=user=>{tzB=BST[+inp.value]||16; if(A.ui.setRange) A.ui.setRange(inp); out.textContent=fint(tzB); inp.setAttribute('aria-valuetext',fint(tzB));
    draw(); el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes(tzB)[i]||'';});
    const h=readHTML(tzB); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>upd(true));
  upd(false);
}});
/* read-only hook for tests/roofline_numbers.py: the tile's numbers */
window.__tzRf=Object.freeze({model:B=>JSON.parse(JSON.stringify(model(B))),read:B=>readHTML(B),notes:B=>notes(B),BST:BST.slice()});
})();
