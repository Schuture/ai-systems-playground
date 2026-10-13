/* Teaser for the train-debug lab's home tile: pick one bug in a small training script and watch its training loss against
   ln 3 (dotted) and the run without a bug (thin line). The runs are the lab's own (labs/train-debug.js, window.__td): a 2 → 16 → 3
   network trained on 300 seeded 2-D points for 300 steps of plain gradient descent. One control, one small visual (120 px tall),
   one sentence in .tz-read, a one-line legend, and plain-text notes. State lives in module scope, so a re-render (domain or
   language switch) keeps the learner's pick. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh).replace(/ln 3 = /g,'ln\u00a03\u00a0=\u00a0'):en;
/* three picks: none, rate too high, no zero_grad(). The shuffled labels stay out of the tile: part 2 of the lab asks which bug keeps
   the loss at ln 3, and the tile must not answer it first (R3.8) */
const OPTS=['none','high','nozero'];
let tzBug='none';
const cfgOf=b=>b==='high'?{C:3,lr:10}:(b==='none'?{C:3}:{C:3,bug:b});
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const f3=v=>fixd(v,3);
const runOf=b=>window.__td?window.__td.runRaw(cfgOf(b)):null;
function readHTML(b){
  const r=runOf(b), h=runOf('none'); if(!r||!h) return '';
  const end=r.tr[r.tr.length-1], pk=Math.max(...r.tr.filter(isFinite)), lo=Math.min(...r.tr.slice(0,60));
  if(b==='none') return tx(`No bug: the loss falls from ${f3(r.tr[0])} to <b>${f3(end)}</b> in 300 steps, far below ln 3 = 1.099.`,
                           `没有 bug：loss 在 300 步里从 ${f3(r.tr[0])} 降到 <b>${f3(end)}</b>，远低于 ln 3 = 1.099。`);
  if(b==='high') return tx(`Learning rate 10 instead of 0.3: the loss jumps to ${f3(pk)}, then turns <b>NaN</b> at step ${r.dead}.`,
                           `学习率用 10 而不是 0.3：loss 跳到 ${f3(pk)}，第 ${r.dead} 步变成 <b>NaN</b>。`);
  return tx(`Without zero_grad() the loss first falls to ${f3(lo)}, then swings and turns <b>NaN</b> at step ${r.dead}.`,
            `漏了 zero_grad()：loss 先降到 ${f3(lo)}，然后来回摆动，第 ${r.dead} 步变成 <b>NaN</b>。`);
}
function notes(){
  return [tx('Toy run: a network with 16 hidden units really trains in your browser, on 300 seeded 2-D points in 3 groups, with plain gradient descent at learning rate 0.3.',
             '玩具实跑：一个有 16 个隐藏单元的网络在你的浏览器里真实训练，数据是 3 组共 300 个带种子的二维点，用学习率 0.3 的普通梯度下降。'),
    tx('If a model gives each of the 3 classes a chance of 1/3, its loss is ln 3 = 1.099. A loss that stays there means the model does no better than that blind guess.',
       '如果模型给 3 个类别的概率都是 1/3，它的 loss 就是 ln 3 = 1.099。loss 一直停在这里，说明模型并不比这样瞎猜强。'),
    tx('The script computes softmax without subtracting the largest logit. With a too-high learning rate the logits drift so far apart that the right class’s chance rounds to 0 (the loss is inf), and then exp overflows (NaN). The lab’s first two checks follow Karpathy’s 2019 “A Recipe for Training Neural Networks”.',
       '脚本算 softmax 时没有先减去最大的 logit。学习率太高时，logit 之间拉得太开，正确类别的概率被舍入成 0（loss 是 inf），接着 exp 溢出（NaN）。Lab 里的前两个检查来自 Karpathy 2019 年的“A Recipe for Training Neural Networks”。')];
}
function svg(W,b){
  const H=120, PL=26, PR=6, PT=8, PB=18, yMax=1.6, X=t=>PL+(W-PL-PR)*t/300, Y=v=>PT+(H-PT-PB)*(1-Math.min(v,yMax*1.1)/yMax);
  const r=runOf(b), h=runOf('none'); if(!r||!h) return {H,h:''};
  const line=a=>{let d='', on=false; a.forEach((v,t)=>{if(!isFinite(v)){on=false; return;} d+=(on?'L':'M')+X(t).toFixed(1)+' '+Y(v).toFixed(1); on=true;}); return d;};
  let s=`<defs><clipPath id="tz-td-clip"><rect x="${PL}" y="0" width="${W-PL-PR}" height="${H-PB}"/></clipPath></defs>`;
  [0,0.8,1.6].forEach(v=>{s+=`<line class="tz-td-grid" x1="${PL}" x2="${W-PR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/><text x="${PL-4}" y="${(Y(v)+3.5).toFixed(1)}" text-anchor="end">${fixd(v,1)}</text>`;});
  [0,150,300].forEach(t=>{s+=`<text x="${X(t).toFixed(1)}" y="${H-4}" text-anchor="${t===0?'start':(t===300?'end':'middle')}">${t===300?tx('300 steps','300 步'):t}</text>`;});
  s+=`<g clip-path="url(#tz-td-clip)"><line class="tz-td-ln" x1="${PL}" x2="${W-PR}" y1="${Y(Math.log(3)).toFixed(1)}" y2="${Y(Math.log(3)).toFixed(1)}"/>`;
  if(b!=='none') s+=`<path class="tz-td-cmp" d="${line(h.tr)}"/>`;
  s+=`<path class="tz-td-tr" d="${line(r.tr)}"/></g>`;
  if(r.dead>=0){const x=X(r.dead); s+=`<line class="tz-td-nan" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${PT}" y2="${H-PB}"/><text class="tz-td-nant" x="${(r.dead<200?x+4:x-4).toFixed(1)}" y="${PT+10}" text-anchor="${r.dead<200?'start':'end'}">NaN</text>`;}
  return {H,h:s};
}
A.registerTeaser('train-debug',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-td">`+A.ui.seg({id:'tz-td-seg',key:'tztd',label:tx('Bug in the training script','训练脚本里的 bug'),options:[['none',tx('none','没有')],['high',tx('rate too high','学习率太高')],['nozero',tx('no zero_grad()','漏了 zero_grad()')]],value:tzBug,inline:false})+
    `<svg class="tz-vis tz-td-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"><span><i class="tz-td-sw tz-td-sw-tr" aria-hidden="true"></i>${esc(tx('thick line: this run','粗线：这次运行'))}</span><span><i class="tz-td-sw tz-td-sw-cmp" aria-hidden="true"></i>${esc(tx('thin line: no bug','细线：没有 bug'))}</span><span><i class="tz-td-sw tz-td-sw-ln" aria-hidden="true"></i>${esc(tx('dotted line: ln 3','点线：ln 3'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('svg'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  const help=tx('The training loss over 300 steps for the picked bug (thick line), the run without a bug (thin line) and ln 3 = 1.099 (dotted line).','所选 bug 下 300 步的训练 loss（粗线）、没有 bug 的运行（细线）和 ln 3 = 1.099（点线）。');
  vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const draw=()=>{const W=Math.max(240,Math.round(el.clientWidth||300)), r=svg(W,tzBug); vis.setAttribute('viewBox',`0 0 ${W} ${r.H}`); vis.setAttribute('width',W); vis.setAttribute('height',r.H); vis.innerHTML=r.h;};
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzBug); draw(); const h=readHTML(tzBug); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=b.dataset.v; if(OPTS.includes(v)&&v!==tzBug){tzBug=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/train-debug_numbers.py: the tile's sentence */
window.__tzTd=Object.freeze({read:b=>readHTML(b),opts:OPTS.slice()});
})();
