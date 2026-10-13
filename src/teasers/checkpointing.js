/* Teaser for the checkpointing lab's home tile: one slider sets the save interval T (5 minutes to 4 hours). One curve shows the
   wasted share of the run against T, on a log axis; a dot marks the chosen T and a dashed drop line the lowest point.
   Formula only (Young's first-order model, the lab's part 4): wasted share = δ/T + (T/2 + R)/M, lowest at T = √(2δM).
   The sentence answers the tile's question: a crash sends the run back to the latest checkpoint, and half an interval is redone.
   Fixed here, and different from the lab's cases: a save stops training for δ = 60 s, a failure every M = 8 hours on average,
   a restart takes R = 10 minutes. The tile does not answer part 1's prediction (R3.8): that one is about what a checkpoint holds.
   The sentence has the same shape at every stop, so the tile keeps its height. State lives in module scope, so a re-render (domain
   or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const D=60, M=8*3600, R=600, INTS=[5,10,15,20,30,45,60,90,120,180,240];
let idx=8;
const W=320, H=120, ML=34, MR=8, MT=8, MB=20, YMAX=0.3, X0=Math.log(5), X1=Math.log(240);
const waste=Tm=>{const T=Tm*60; return D/T+(T/2+R)/M;};
const tOpt=()=>Math.sqrt(2*D*M)/60;
const pctI=v=>Math.round(v*100)+'%';
const minsT=m=>m<60?tx(`${m} min`,`${m} 分钟`):(m%60===0?tx(`${m/60} h`,`${m/60} 小时`):tx(`${m/60} h`,`${m/60} 小时`));
function model(i){const T=INTS[i], w=waste(T), to=tOpt(), wo=waste(to); return {T,w,to,wo,near:w-wo<0.005};}
const X=m=>ML+(Math.log(m)-X0)/(X1-X0)*(W-ML-MR), Y=v=>H-MB-(Math.min(v,YMAX)/YMAX)*(H-MT-MB);
function visSVG(i){const m=model(i); let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">`;
  [0,0.1,0.2,0.3].forEach(v=>{const y=Y(v); s+=`<line class="tz-ckp-gl" x1="${ML}" x2="${W-MR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/><text class="tz-ckp-tk" x="${ML-4}" y="${(y+3).toFixed(1)}" text-anchor="end">${Math.round(v*100)}%</text>`;});
  [[5,tx('5 min','5 分钟')],[30,tx('30 min','30 分钟')],[60,tx('1 h','1 小时')],[240,tx('4 h','4 小时')]].forEach(([t,l])=>{const x=X(t); s+=`<text class="tz-ckp-tk" x="${x.toFixed(1)}" y="${H-6}" text-anchor="${t===5?'start':(t===240?'end':'middle')}">${esc(l)}</text>`;});
  const pts=[]; for(let k=0;k<=60;k++){const t=Math.exp(X0+(X1-X0)*k/60); pts.push(X(t).toFixed(1)+','+Y(waste(t)).toFixed(1));}
  s+=`<path class="tz-ckp-cur" d="M${pts.join('L')}"/>`;
  const xo=X(m.to); s+=`<line class="tz-ckp-drop" x1="${xo.toFixed(1)}" x2="${xo.toFixed(1)}" y1="${(Y(m.wo)+4).toFixed(1)}" y2="${H-MB}"/>`;
  s+=`<circle class="tz-ckp-dot" cx="${X(m.T).toFixed(1)}" cy="${Y(m.w).toFixed(1)}" r="5"/>`;
  return s+'</svg>';}
function readHTML(i){const m=model(i);
  return tx(`A crash sends you back to the latest checkpoint: saving every ${minsT(m.T)}, you redo ${minsT(m.T/2)} on average and waste about <b>${pctI(m.w)} of the run</b>. Saving about every ${Math.round(m.to)} min wastes least, about ${pctI(m.wo)}.`,
            `崩了只能从最近的存档接着训。每 ${minsT(m.T)}存一次，平均要重算 ${minsT(m.T/2)}的训练，约 <b>${pctI(m.w)} 的时间白费</b>。每 ${Math.round(m.to)} 分钟左右存一次最省，约 ${pctI(m.wo)}。`);}
function notes(){
  return [tx('Formula: wasted share = δ/T + (T/2 + R)/M. Every save stops training for δ; a failure comes every M on average and loses, on average, half an interval of work, plus R to restart.',
             '公式：浪费的比例 = δ/T + (T/2 + R)/M。每次存档训练停 δ；平均每隔 M 出一次故障，平均丢掉半个间隔的工作，再加上重启的 R。'),
    tx('Assumed here: δ = 1 minute, M = 8 hours, R = 10 minutes. Saving often wastes time on the saves; saving rarely wastes time redoing work. The least is at T = √(2δM).',
       '这里假设 δ = 1 分钟，M = 8 小时，R = 10 分钟。存得太勤，时间花在存档上；存得太少，时间花在重算上。最省的间隔是 T = √(2δM)。'),
    tx('A first-order estimate (Young, 1974): it holds while δ and T are much shorter than M.',
       '这是一阶估计（Young，1974），δ 和 T 都比 M 短得多时才成立。')];}
A.registerTeaser('checkpointing',{fid:'formula',render(el){
  const lab=tx('Save every','每隔多久存一次');
  el.innerHTML=`<div class="tz tz-ckp">`+
    `<label class="tz-lab" for="tz-ckp-in"><span>${esc(lab)}</span><output id="tz-ckp-out"></output></label>`+
    `<input type="range" id="tz-ckp-in" min="0" max="${INTS.length-1}" step="1" value="${idx}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-ckp-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-ckp-swc" aria-hidden="true"></i>${esc(tx('wasted share of the run (log axis across)','浪费的时间比例（横轴是对数坐标）'))}</span><span><i class="sw tz-ckp-swd" aria-hidden="true"></i>${esc(tx('your interval','你选的间隔'))}</span><span><i class="sw tz-ckp-swl" aria-hidden="true"></i>${esc(tx('the least','最省'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-ckp-in'), out=el.querySelector('#tz-ckp-out'), vis=el.querySelector('.tz-ckp-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const m=model(idx), t=minsT(m.T); out.textContent=t; inp.value=idx; inp.setAttribute('aria-valuetext',t); if(A.ui&&A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visSVG(idx);
    const help=tx(`The curve is the wasted share of the run for every save interval from 5 minutes to 4 hours. The dot is every ${t}: about ${pctI(m.w)}. The dashed line marks the least, about every ${Math.round(m.to)} min.`,
                  `曲线是从 5 分钟到 4 小时，每种存档间隔下浪费的时间比例。圆点是每 ${t}一次：约 ${pctI(m.w)}。虚线标出最省的间隔，约 ${Math.round(m.to)} 分钟。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(idx); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(INTS.length-1,Math.round(+inp.value))); if(v!==idx){idx=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/checkpointing_numbers.py */
window.__tzCkp=Object.freeze({D,M,R,INTS:INTS.slice(),model:i=>({...model(i)}),read:i=>readHTML(i),notes:()=>notes()});
})();
