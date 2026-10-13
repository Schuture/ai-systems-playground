/* Teaser for the eval-ci lab's home tile: one slider sets the number of questions (100 → 10,000) on which two models score 70% and
   72% (not the lab's 80% and 82%, so the tile does not answer part 1's prediction, R3.8). Each row draws a score with its 95% error
   bar on a fixed axis; where the two bars overlap, a hatched band spans both rows (at least 8 px wide, so a thin overlap still shows).
   A third row draws B − A's own error bar (each on its own) on its own fixed axis with 0 marked: the sentence's verdict is that bar
   against 0, as in the lab (overlap alone does not mean "no difference").
   Formula only: SE = √(p(1 − p)/n), error bar = p ± 1.96·SE (normal approximation), B − A: ± 1.96·√(SE_A² + SE_B²).
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const NS=[100,200,500,1000,2000,5000,10000], PA=0.70, PB=0.72, Z=1.96, AX=[58,84], DX=[-2,6];
let idx=3;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fN=n=>n.toLocaleString('en-US');
function model(i){const n=NS[i], seA=Math.sqrt(PA*(1-PA)/n), seB=Math.sqrt(PB*(1-PB)/n), hA=Z*seA, hB=Z*seB, hS=Z*Math.sqrt(seA*seA+seB*seB), D=PB-PA;
  return {n,hA,hB,hS,D,overlap:PA+hA>PB-hB,real:D-hS>0};}
const pos=v=>Math.max(0,Math.min(100,(v-AX[0])/(AX[1]-AX[0])*100));
const dpos=v=>Math.max(0,Math.min(100,(v-DX[0])/(DX[1]-DX[0])*100));
function readHTML(i){const m=model(i), n=fN(m.n);
  if(!m.real) return tx(`With ${n} questions, the error bar of B − A crosses 0, so the 2-point gap <b>could be noise</b>.`,
                        `${n} 道题时，B − A 的误差线跨过了 0，这 2 分<b>可能只是噪声</b>。`);
  if(m.overlap) return tx(`With ${n} questions the error bars still overlap, yet B − A’s error bar <b>stays clear of 0</b>: the lead is unlikely to be noise.`,
                          `${n} 道题时，两条误差线还有一点重叠，但 B − A 的误差线已经<b>不碰 0</b>，B 领先不太可能只是噪声。`);
  return tx(`With ${n} questions the error bars no longer overlap, and B − A’s error bar is clear of 0: the gain is <b>beyond noise</b>.`,
            `${n} 道题时，两条误差线不再重叠，B − A 的误差线也不碰 0，高出的 2 分<b>超出了噪声</b>。`);}
function notes(){
  /* the tile's notes hover is plain text (the kit's tznotes), so the formula itself is shown typeset in the lab, not here (R6.29) */
  return [tx('Formula: each error bar is the normal approximation of a 95% interval, worked out from the score and the number of questions. The more questions, the shorter the bar. The lab shows the formula.',
             '公式：每条误差线是 95% 区间的正态近似，由分数和题数算出。题越多，误差线越短。公式在 Lab 里。'),
    tx('Overlapping error bars do not mean there is no difference. The bottom row is B − A’s own error bar, combined from A’s and B’s. From about 4,000 questions on, it stays clear of 0.',
       '两条误差线重叠，不等于没有差别。最下面一行是 B − A 自己的误差线，由 A 和 B 的误差线合成。从大约 4,000 道题起，它就不碰 0 了。'),
    tx('70% and 72% are made-up scores for two models; the lab uses 80% and 82%. The questions are assumed to be drawn independently from many like them.',
       '70% 和 72% 是两个模型的虚构分数；Lab 里用的是 80% 和 82%。假设题目是从大量同类题里各自独立抽出来的。')];}
function rowHTML(name,p,h){const l=pos((p-h)*100), r=pos((p+h)*100), x=pos(p*100);
  return `<div class="tz-evc-row"><span class="tz-evc-lab">${esc(`${name} ${fixd(p*100,1)}% ± ${fixd(h*100,1)}`)}</span><span class="tz-evc-trk">`+
    `<i class="tz-evc-bar" style="left:${l.toFixed(2)}%;width:${(r-l).toFixed(2)}%"></i><i class="tz-evc-dot" style="left:${x.toFixed(2)}%"></i></span></div>`;}
/* B − A on its own axis (points, DX), 0 marked; a bar wider than the axis is cut at the edge with an arrow head */
function diffHTML(m){const D=m.D*100, h=m.hS*100, l=dpos(D-h), r=dpos(D+h), x=dpos(D), z=dpos(0);
  const cut=(D-h<DX[0]?' cl':'')+(D+h>DX[1]?' cr':'');
  return `<div class="tz-evc-row tz-evc-drow"><span class="tz-evc-lab">B − A<br>${esc(`${fixd(D,1)} ± ${fixd(h,1)}`)}</span><span class="tz-evc-trk">`+
    `<b class="tz-evc-zero" style="left:${z.toFixed(2)}%"></b><i class="tz-evc-bar${cut}" style="left:${l.toFixed(2)}%;width:${(r-l).toFixed(2)}%"></i><i class="tz-evc-dot" style="left:${x.toFixed(2)}%"></i></span></div>`+
    `<div class="tz-evc-scale"><span class="tz-evc-sp"></span><span class="tz-evc-sc tz-evc-dsc"><span style="left:${z.toFixed(2)}%">0</span></span></div>`;}
function visHTML(i){const m=model(i), lo=Math.max((PA-m.hA)*100,(PB-m.hB)*100), hi=Math.min((PA+m.hA)*100,(PB+m.hB)*100);
  const band=m.overlap?`<b class="tz-evc-ov" style="left:${((pos(lo)+pos(hi))/2).toFixed(2)}%;width:${(pos(hi)-pos(lo)).toFixed(2)}%"></b>`:'';
  let sc=''; for(let v=60; v<=80; v+=10) sc+=`<span style="left:${pos(v).toFixed(2)}%">${v}%</span>`;
  return `<div class="tz-evc-rows">${rowHTML('A',PA,m.hA)}${rowHTML('B',PB,m.hB)}<div class="tz-evc-band">${band}</div></div><div class="tz-evc-scale"><span class="tz-evc-sp"></span><span class="tz-evc-sc">${sc}</span></div>${diffHTML(m)}`;}
A.registerTeaser('eval-ci',{fid:'formula',render(el){
  const lab=tx('Questions in the benchmark','benchmark 的题数');
  el.innerHTML=`<div class="tz tz-evc">`+
    `<label class="tz-lab" for="tz-evc-in"><span>${esc(lab)}</span><output id="tz-evc-out">${fN(NS[idx])}</output></label>`+
    `<input type="range" id="tz-evc-in" min="0" max="${NS.length-1}" step="1" value="${idx}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-evc-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-evc-swbar" aria-hidden="true"></i>${esc(tx('score and its 95% error bar','分数和它的 95% 误差线'))}</span><span><i class="sw tz-evc-swov" aria-hidden="true"></i>${esc(tx('where the two overlap','两条重叠的部分'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-evc-in'), out=el.querySelector('#tz-evc-out'), vis=el.querySelector('.tz-evc-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const m=model(idx), n=fN(m.n); out.textContent=n; inp.value=idx; inp.setAttribute('aria-valuetext',tx(`${n} questions`,`${n} 道题`)); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visHTML(idx);
    const help=tx(`Two models on ${n} questions: A ${fixd(PA*100,1)}% ± ${fixd(m.hA*100,1)} and B ${fixd(PB*100,1)}% ± ${fixd(m.hB*100,1)} points.${m.overlap?' The hatched band is where the two error bars overlap.':' The error bars do not overlap.'} The bottom row is B − A = 2.0 ± ${fixd(m.hS*100,1)} points; the vertical line there is 0.`,
                  `两个模型做 ${n} 道题：A ${fixd(PA*100,1)}% ± ${fixd(m.hA*100,1)} 分，B ${fixd(PB*100,1)}% ± ${fixd(m.hB*100,1)} 分。${m.overlap?'斜纹是两条误差线重叠的部分。':'两条误差线没有重叠。'}最下面一行是 B − A = 2.0 ± ${fixd(m.hS*100,1)} 分，那里的竖线是 0。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(idx); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(NS.length-1,Math.round(+inp.value))); if(v!==idx){idx=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/eval-ci_numbers.py */
window.__tzEvc=Object.freeze({NS:NS.slice(),DX:DX.slice(),model:i=>({...model(i)}),read:i=>readHTML(i)});
})();
