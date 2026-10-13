/* Teaser for the eval-capstone lab's home tile: one total against the same items split by category. A formula on a stated example
   (not the lab's own numbers, R3.8): three categories with 2,000 items each and 20% of items with a winner; true differences +4
   (coding, 50% of traffic), +1 (Q&A, 30%), −4 (summaries, 20%); the total is their traffic-weighted average, +1.5 points. Interval:
   the lab's item-by-item one, h = 1.96 · sqrt((q − Δ²) / n); the total's standard error combines the categories' with the weights.
   One control, one small forest plot of steady height, one sentence in .tz-read, a one-line legend, plain-text notes. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const N=2000, Q=0.2, Z=1.96;
const CATS=[{en:'Coding',zh:'编程',w:.5,d:.04},{en:'Q&A',zh:'问答',w:.3,d:.01},{en:'Summaries',zh:'摘要',w:.2,d:-.04}];
function rows(){
  const r=CATS.map(c=>{const se=Math.sqrt((Q-c.d*c.d)/N); return {name:tx(c.en,c.zh),D:c.d,se,lo:c.d-Z*se,hi:c.d+Z*se};});
  const D=CATS.reduce((s,c)=>s+c.w*c.d,0), se=Math.sqrt(CATS.reduce((s,c,i)=>s+(c.w*r[i].se)**2,0));
  return {cats:r,tot:{name:tx('Total','总分'),D,se,lo:D-Z*se,hi:D+Z*se}};
}
let tzMode='tot';
const f1=x=>{const v=Math.abs(x*100).toFixed(1); return (x<0?'−':'+')+v;};
const LO=-8, HI=8, pos=v=>((v*100-LO)/(HI-LO)*100).toFixed(1);
function readHTML(m){
  const R=rows(), s=R.cats[2];
  if(m==='tot') return tx(`One total: the new model is <b>${f1(R.tot.D)} points</b> ahead, interval clear of 0. It looks ready.`,
                          `只看总分：新模型高 <b>${Math.abs(R.tot.D*100).toFixed(1)} 分</b>，区间不含 0，看起来可以上线。`);
  return tx(`Split by category: summaries are <b>${f1(s.D)} points</b>, clearly worse; coding’s gain hid it.`,
            `按类别分开：摘要的差是 <b>${f1(s.D)} 分</b>，确实变差了；编程的进步把它盖住了。`);
}
function visHTML(m){
  const R=rows(), list=m==='tot'?[R.tot]:R.cats;
  const row=r=>{const red=r.hi<0; return `<div class="tz-ecap-row"><span class="tz-ecap-n">${esc(r.name)}</span><span class="tz-ecap-t"><i class="tz-ecap-z" style="left:${pos(0)}%"></i><i class="tz-ecap-b${red?' red':''}" style="left:${pos(r.lo)}%;width:${(pos(r.hi)-pos(r.lo)).toFixed(1)}%"></i><i class="tz-ecap-p${red?' red':''}" style="left:${pos(r.D)}%"></i></span></div>`;};
  return `<div class="tz-ecap-fp">${list.map(row).join('')}</div><div class="tz-ecap-ax mono"><span>−8</span><span>0</span><span>+8</span></div>`;
}
function notes(){
  return [tx('Formula on a stated example: three categories, 2,000 items each; 20% of items have a better answer on one side. True differences +4, +1 and −4 points, weighted by traffic 50%, 30% and 20%.',
             '公式计算，例子是设定的：三类题，每类 2,000 题，五分之一的题有一方答得更好。真实的差是 +4、+1、−4 分，按流量 50%、30%、20% 加权。'),
    tx('Each bar is the 95% interval of the difference between the new and the old model, in points (percentage points), compared item by item.',
       '每条线是新旧模型之差的 95% 区间，单位是分（百分点），按逐题对比算。')];
}
A.registerTeaser('eval-capstone',{fid:'formula',render(el){
  el.innerHTML=`<div class="tz tz-ecap">`+A.ui.seg({id:'tz-ecap-seg',key:'tzecap',label:tx('Read the result as','怎么看结果'),options:[['tot',tx('one total','只看总分')],['cat',tx('by category','按类别分开')]],value:tzMode,inline:false})+
    `<div class="tz-vis tz-ecap-vis" role="img"></div>`+
    `<div class="tz-leg"><span><i class="tz-ecap-sw" aria-hidden="true"></i>${esc(tx('difference, 95% interval','差和 95% 区间'))}</span><span><i class="tz-ecap-sw red" aria-hidden="true"></i>${esc(tx('worse','变差了'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-ecap-vis'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzMode);
    vis.innerHTML=visHTML(tzMode);
    const R=rows(), lab=(tzMode==='tot'?[R.tot]:R.cats).map(r=>tx(`${r.name} ${f1(r.D)} points (${f1(r.lo)} to ${f1(r.hi)})`,`${r.name} ${f1(r.D)} 分（${f1(r.lo)} 到 ${f1(r.hi)}）`)).join(tx('; ','；'));
    vis.setAttribute('aria-label',lab); vis.setAttribute('data-tipx',lab);
    const h=readHTML(tzMode); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=b.dataset.v; if((v==='tot'||v==='cat')&&v!==tzMode){tzMode=v; upd(true);}});
  upd(false);
}});
window.__tzEcap=Object.freeze({rows:()=>JSON.parse(JSON.stringify(rows())),read:m=>readHTML(m)});
})();
