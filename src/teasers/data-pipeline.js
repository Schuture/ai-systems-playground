/* Teaser for the data-pipeline lab's home tile: one slider sets MinHash's hashes per group r (2 → 12) with 20 groups fixed. The
   S-curve is the chance that a pair of pages of similarity s is caught, P(s) = 1 − (1 − s^r)^b; two marked points read it at a
   repost (s = 0.8) and at a heavily edited version (s = 0.5),
   which still shares about two thirds of its 5-grams and should stay as a page of its own. The tile does not answer part 1's prediction (R3.8): it shows no
   page counts. Formula only. State lives in module scope, so a re-render keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const B=20, RMIN=2, RMAX=12, HI=0.8, LO=0.5;
let r=6;
const P=(s,rr)=>1-Math.pow(1-Math.pow(s,rr),B);
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pc=v=>v>=0.9995?tx('over 99.9%','99.9% 以上'):(v<0.0005?tx('under 0.1%','不到 0.1%'):fixd(v*100,1)+'%');
const pcs=v=>v>=0.9995?'>99.9%':(v<0.0005?'<0.1%':fixd(v*100,1)+'%'); /* short form for the chart's labels, which share one line near the top */
function readHTML(rr){const h=P(HI,rr), l=P(LO,rr), ph=pc(h), pl=pc(l);
  if(l>0.1) return tx(`With 20 groups of ${rr} hashes, a repost (similarity 0.8) is caught ${ph} of the time, but a heavily edited version (0.5) is too, ${pl}: <b>too loose</b>.`,
                      `20 组、每组 ${rr} 个哈希时，转载（相似度 0.8）有${ph}被抓住。可改动很大的版本（相似度 0.5）也有${pl}被当成重复，<b>太松了</b>。`);
  if(h<0.9) return tx(`With 20 groups of ${rr} hashes, heavily edited versions (0.5) are safe at ${pl}, but reposts (0.8) are caught only ${ph} of the time: <b>too strict</b>.`,
                      `20 组、每组 ${rr} 个哈希时，改动很大的版本（0.5）只有${pl}被当成重复。可转载（0.8）也只有${ph}被抓住，<b>太严了</b>。`);
  return tx(`With 20 groups of ${rr} hashes, reposts (0.8) are caught ${ph} of the time and heavily edited versions (0.5) only ${pl}: the curve <b>separates the two</b>.`,
            `20 组、每组 ${rr} 个哈希时，转载（0.8）有${ph}被抓住，改动很大的版本（0.5）只有${pl}。曲线<b>把两种分开了</b>。`);}
function notes(){
  return [tx('Formula: a pair of similarity s is caught with chance P(s) = 1 − (1 − sʳ)ᵇ, here with b = 20 groups. FineWeb uses 14 groups of 8 hashes.','公式：相似度为 s 的一对网页被抓住的机会是 P(s) = 1 − (1 − sʳ)ᵇ，这里 b = 20 组。FineWeb 用的是 14 组、每组 8 个哈希。'),
    tx('Similarity is the share of 5-grams (runs of 5 words) the two pages have in common. At 0.5, two pages of the same length still share about two thirds of their 5-grams.','相似度是两篇网页共有的 5-gram（连续 5 个词）所占的比例。相似度是 0.5 时，两篇仍有大约三分之二的 5-gram 相同。')];}
const W=300, H=130, L0=30, R0=8, T0=8, B0=22, X=s=>L0+s*(W-L0-R0), Y=p=>T0+(1-p)*(H-T0-B0), f1=v=>v.toFixed(1);
function visHTML(rr){
  let d=''; for(let k=0;k<=100;k++){const s=k/100; d+=(k?'L':'M')+f1(X(s))+' '+f1(Y(P(s,rr)));}
  let g=`<line class="tz-dl-ax" x1="${L0}" x2="${W-R0}" y1="${Y(0)}" y2="${Y(0)}"/><line class="tz-dl-gl" x1="${L0}" x2="${W-R0}" y1="${Y(1)}" y2="${Y(1)}"/>`+
    `<text class="tz-dl-t" x="${L0-4}" y="${Y(1)+4}" text-anchor="end">100%</text><text class="tz-dl-t" x="${L0-4}" y="${Y(0)+3}" text-anchor="end">0</text>`;
  [0,0.5,0.8,1].forEach(s=>{g+=`<text class="tz-dl-t" x="${f1(X(s))}" y="${H-8}" text-anchor="middle">${s===0?'0':(s===1?'1':fixd(s,1))}</text>`;});
  g+=`<path class="tz-dl-cur" d="${d}"/>`;
  const pts=[[HI,tx('repost','转载')],[LO,tx('heavily edited','改动很大')]].map(([s,t])=>{const p=P(s,rr); return {s,t,p,y:Math.max(T0+10,Math.min(Y(0)-6,Y(p)-6))};});
  if(Math.abs(pts[0].y-pts[1].y)<6) pts[0].y=pts[1].y+12; /* both near 100%: the repost's label goes one line down, under the flat top */
  pts.forEach(({s,t,p,y})=>{
    g+=`<line class="tz-dl-mk" x1="${f1(X(s))}" x2="${f1(X(s))}" y1="${Y(0)}" y2="${f1(Y(p))}"/><circle class="tz-dl-pt" cx="${f1(X(s))}" cy="${f1(Y(p))}" r="4"/>`+
      `<text class="tz-dl-v" x="${f1(X(s)-6)}" y="${f1(y)}" text-anchor="end">${esc(t)} ${pcs(p)}</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">${g}</svg>`;}
A.registerTeaser('data-pipeline',{fid:'formula',render(el){
  const lab=tx('Hashes per group r (20 groups)','每组的哈希数 r（共 20 组）');
  el.innerHTML=`<div class="tz tz-dl">`+
    `<label class="tz-lab" for="tz-dl-in"><span>${esc(lab)}</span><output id="tz-dl-out">${r}</output></label>`+
    `<input type="range" id="tz-dl-in" min="${RMIN}" max="${RMAX}" step="1" value="${r}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-dl-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-dl-swc" aria-hidden="true"></i>${esc(tx('the curve: chance that two pages are caught as duplicates (x axis: similarity)','曲线：两篇网页被当成重复的机会（横轴是相似度）'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-dl-in'), out=el.querySelector('#tz-dl-out'), vis=el.querySelector('.tz-dl-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{out.textContent=String(r); inp.value=r; inp.setAttribute('aria-valuetext',tx(`${r} hashes per group`,`每组 ${r} 个哈希`)); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visHTML(r);
    const help=tx(`MinHash with 20 groups of ${r}: the curve is the chance that two pages are caught as duplicates, from similarity 0 to 1. At 0.8 it is ${pc(P(HI,r))}; at 0.5 it is ${pc(P(LO,r))}.`,
                  `MinHash 用 20 组、每组 ${r} 个：曲线是两篇网页被当成重复的机会，相似度从 0 到 1。0.8 时是 ${pc(P(HI,r))}，0.5 时是 ${pc(P(LO,r))}。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(r); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(RMIN,Math.min(RMAX,Math.round(+inp.value))); if(v!==r){r=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/data-pipeline_numbers.py */
window.__tzDl=Object.freeze({B,RMIN,RMAX,P,read:rr=>readHTML(rr)});
})();
