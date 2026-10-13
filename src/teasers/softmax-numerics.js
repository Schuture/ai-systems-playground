/* Teaser for the softmax-numerics lab's home tile: one slider sets the sampler's temperature T, which divides the model's six logits
   (9.2, 8.9, 8.7, 8.6, 8.4 and 8.2, the lab's mission at T = 1). The bar shows the naive softmax's chances in FP32 (Math.fround after
   every operation, the lab's own engine). Below T ≈ 0.104 the largest logit passes 88.72, e^z overflows FP32 and Inf ÷ Inf is NaN
   (a red hatched bar); only then does a second bar show the softmax with the max subtracted. The two are never shown side by side
   while both work, so the tile does not answer part 2's prediction (R3.8: what subtracting one number from every logit does).
   One control, one small visual (at most two bars), one sentence in .tz-read, a one-line legend, plain-text notes.
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
/* the mission's logits at T = 0.1; at temperature k/100 they are Z10 × 10 / k */
const TOKS=['Paris','the','a','located','home','not'], Z10=[92,89,87,86,84,82];
let k=10;
const F=Math.fround;
function naive(z){const e=z.map(v=>F(Math.exp(F(v)))); let s=0; for(const v of e) s=F(s+v); return {e,s,p:e.map(v=>F(v/s))};}
function stable(z){const zz=z.map(v=>F(v)), m=Math.max(...zz), e=zz.map(v=>F(Math.exp(F(v-m)))); let s=0; for(const v of e) s=F(s+v); return {e,s,p:e.map(v=>F(v/s))};}
function model(t){const z=Z10.map(v=>v*10/t), n=naive(z), st=stable(z);
  return {k:t,temp:t/100,z,top:z[0],n:n.p,st:st.p,sumN:n.s,nan:n.p.some(Number.isNaN),over:n.e.filter(v=>v===Infinity).length};}
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>Number.isNaN(x)?'NaN':fixd(x*100,1)+'%';
const fT=t=>fixd(t/100,2);
const fZ=v=>Number.isInteger(v)?String(v):fixd(v,1);
function readHTML(t){const m=model(t), z=fZ(m.top);
  if(m.nan) return tx(`At temperature ${fT(t)} the largest score becomes ${z}. FP32 cannot hold e<sup>${z}</sup>, so the naive softmax returns <b>NaN</b>. Subtracting the max first gives “Paris” <b>${fP(m.st[0])}</b>.`,
                      `temperature 为 ${fT(t)} 时，最大的得分变成 ${z}，e<sup>${z}</sup> 超出了 FP32 能存的范围，朴素 softmax 返回 <b>NaN</b>。先减去最大值再算，“Paris”得到 <b>${fP(m.st[0])}</b>。`);
  return tx(`At temperature ${fT(t)} the largest score is ${z}, and e<sup>${z}</sup> still fits in FP32, so the naive softmax works: “Paris” gets <b>${fP(m.n[0])}</b>.`,
            `temperature 为 ${fT(t)} 时，最大的得分是 ${z}，e<sup>${z}</sup> 还存得下，朴素 softmax 能算出结果：“Paris” <b>${fP(m.n[0])}</b>。`);}
function notes(){
  return [tx('Toy run: the six made-up scores (logits) of the lab’s mission, 9.2, 8.9, 8.7, 8.6, 8.4 and 8.2, divided by the temperature. Your browser computes in FP32 and rounds after every operation.',
             '玩具实跑：Lab 任务里的六个虚构得分（logit），9.2、8.9、8.7、8.6、8.4 和 8.2，除以 temperature。浏览器用 FP32 计算，每一步运算后都取整。'),
    tx('Naive: exp(z) ÷ (sum of all exp(z)). FP32’s largest number is about 3.4 × 10³⁸, so exp(z) becomes Inf above z ≈ 88.72, and Inf ÷ Inf is NaN.',
       '朴素：exp(z) ÷（所有 exp(z) 之和）。FP32 能存的最大数约 3.4 × 10³⁸，所以 z 超过约 88.72 时 exp(z) 变成 Inf，而 Inf ÷ Inf 是 NaN。'),
    tx('Max subtracted, shown only when the naive softmax fails: exp(z − m) ÷ (sum of all exp(z − m)), with m the largest score, the way PyTorch’s softmax computes it (aten/src/ATen/native/SoftMax.cpp).',
       '先减最大值（只在朴素 softmax 出错时显示）：exp(z − m) ÷（所有 exp(z − m) 之和），m 是最大的得分；PyTorch 的 softmax 就是这样算的（aten/src/ATen/native/SoftMax.cpp）。')];}
function barHTML(p,lab,hide){
  const nan=p.some(Number.isNaN);
  let inner;
  if(nan) inner=`<span class="tz-sm-nan">NaN</span>`;
  else {let x=0; inner=p.map((v,i)=>{if(v<=0) return ''; const s=`<i class="tz-sm-seg" style="left:${(x*100).toFixed(2)}%;width:${(v*100).toFixed(2)}%">${i===0&&v>0.3?`<span>${esc(TOKS[0])} ${fP(v)}</span>`:''}</i>`; x+=v; return s;}).join('');}
  return `<div class="tz-sm-row${hide?' tz-sm-off':''}"${hide?' aria-hidden="true"':''}><span class="tz-sm-lab">${esc(lab)}</span><div class="tz-sm-bar${nan?' tz-sm-bad':''}">${inner}</div></div>`;
}
A.registerTeaser('softmax-numerics',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-sm">`+
    `<label class="tz-lab" for="tz-sm-in"><span>${esc(tx('Temperature','temperature'))}</span><output id="tz-sm-out">${fT(k)}</output></label>`+
    `<input type="range" id="tz-sm-in" min="5" max="100" step="1" value="${k}" aria-label="${esc(tx('Temperature','temperature'))}">`+
    `<div class="tz-vis tz-sm-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-sm-sw" aria-hidden="true"></i>${esc(tx('chance of each token, “Paris” first','每个 token 的概率，“Paris”在最前'))}</span><span><i class="sw tz-sm-sw tz-sm-bad" aria-hidden="true"></i>NaN</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-sm-in'), out=el.querySelector('#tz-sm-out'), vis=el.querySelector('.tz-sm-vis'), rd=el.querySelector('.tz-read');
  const help=tx('The chances of the six tokens laid end to end, from the naive softmax. When it returns NaN (red hatched), a second bar shows the softmax with the max subtracted.','六个 token 的概率首尾相接，来自朴素 softmax。它返回 NaN（红色斜纹）时，第二根条显示先减最大值的 softmax。');
  vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  inp.setAttribute('aria-valuetext',fT(k));
  const upd=user=>{const m=model(k); out.textContent=fT(k); inp.value=k; inp.setAttribute('aria-valuetext',fT(k)); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=barHTML(m.n,tx('naive','朴素'),false)+barHTML(m.st,tx('max subtracted','先减最大值'),!m.nan);
    const h=readHTML(k); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(5,Math.min(100,Math.round(+inp.value))); if(v!==k){k=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/softmax-numerics_numbers.py: the tile's numbers at temperature t/100 */
window.__tzSm=Object.freeze({model:t=>JSON.parse(JSON.stringify(model(t),(key,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v)),read:t=>readHTML(t)});
})();
