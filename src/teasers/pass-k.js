/* Teaser for the pass-k lab's home tile: 8 tasks, 4 tries each, pass@1 always 50%. Pick how the 50% is spread over the tasks and
   watch pass@4: 93.8% when every task passes half the time, 71.9% with 2 always / 4 half / 2 never, 50.0% when half always pass and
   half never do. Exact (closed form): pass@4 = mean over tasks of 1 − (1 − p)^4. The squares are one possible set of tries (a 50% task
   is drawn passing 2 of 4), and a note says so. It uses 50%, not the lab's 60%, so the tile does not answer part 1 first (R3.8).
   One control, one small visual (at most 130 px, same height for every pick), one sentence in .tz-read, a legend, plain-text notes. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const OPTS=['even','mix','split'];
/* each task's chance of passing a try */
const MIX={even:[.5,.5,.5,.5,.5,.5,.5,.5],mix:[1,1,.5,.5,.5,.5,0,0],split:[1,1,1,1,0,0,0,0]};
/* one possible set of tries for a 50% task: 2 of 4 pass, in a different order for each task */
const HALF=[[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1],[1,0,0,1],[0,1,1,0],[1,0,1,0],[0,1,1,0]];
let tzMix='even';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>fixd(x*100,1)+'%';
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
const p1=m=>mean(MIX[m]);
const p4=m=>mean(MIX[m].map(p=>1-Math.pow(1-p,4)));
function readHTML(m){
  const v=fP(p4(m));
  if(m==='even') return tx(`Every task passes half the time, so 4 tries almost always include a pass: pass@4 is <b>${v}</b>.`,
                           `每个任务都有一半的把握，4 次里几乎总有一次通过，pass@4 是 <b>${v}</b>。`);
  if(m==='mix') return tx(`Two tasks never pass, and no retry rescues them: pass@4 stops at <b>${v}</b>.`,
                          `有 2 个任务从来过不了，重试救不回来，pass@4 停在 <b>${v}</b>。`);
  return tx(`Half the tasks always pass and half never do, so retrying adds nothing: pass@4 stays at <b>${v}</b>.`,
            `一半任务次次都过，一半怎么都过不了，重试什么也添不了，pass@4 还是 <b>${v}</b>。`);
}
function notes(){
  return [tx('Formula: pass@4 averages, over the tasks, each task’s chance that at least one of 4 tries passes. pass@1 is 50% in all three.',
             '按公式算：pass@4 是每个任务“4 次里至少一次通过”的机会，再对任务取平均。三种情况的 pass@1 都是 50%。'),
    tx('The squares show one possible set of tries: a task that passes half the time is drawn passing 2 of its 4.',
       '方块只是一种可能的尝试结果：有一半把握的任务，画成 4 次里过 2 次。')];
}
function svg(W,m){
  const H=118, cols=8, rows=4, gap=3, sq=Math.min(15,Math.floor((W*0.5-30)/cols)-gap), cw=sq+gap, LW=24, top=6;
  const gx=LW, gw=cols*cw, bx=gx+gw+18, bw=Math.max(60,W-bx-48);
  let s='';
  [0,1,2,3].forEach(j=>{s+=`<text x="${LW-5}" y="${(top+j*cw+sq/2+3.5).toFixed(1)}" text-anchor="end">${j+1}</text>`;});
  MIX[m].forEach((p,i)=>{const t=p===1?[1,1,1,1]:(p===0?[0,0,0,0]:HALF[i]);
    t.forEach((on,j)=>{s+=`<rect class="tz-pak-sq${on?' on':''}" x="${gx+i*cw}" y="${top+j*cw}" width="${sq}" height="${sq}" rx="1.5"/>`;});});
  const yl=top+rows*cw+14;
  s+=`<text x="${gx}" y="${yl}">${esc(tx('8 tasks; rows are tries','8 个任务；每行是一次尝试'))}</text>`;
  /* two bars: pass@1 (the same for every pick) and pass@4 */
  const bar=(y,lab,v,cls)=>`<text class="tz-pak-bl" x="${bx}" y="${y-5}">${esc(lab)}</text><rect class="tz-pak-trk" x="${bx}" y="${y}" width="${bw}" height="12" rx="2"/><rect class="${cls}" x="${bx}" y="${y}" width="${(bw*v).toFixed(1)}" height="12" rx="2"/><text class="tz-pak-bv" x="${bx+bw+5}" y="${y+10}">${fP(v)}</text>`;
  s+=bar(top+16,'pass@1',p1(m),'tz-pak-b1');
  s+=bar(top+58,'pass@4',p4(m),'tz-pak-b4');
  return {H,h:s};
}
A.registerTeaser('pass-k',{fid:'formula',render(el){
  el.innerHTML=`<div class="tz tz-pak">`+A.ui.seg({id:'tz-pak-seg',key:'tzpak',label:tx('pass@1 is 50% in all three. Which tasks do the passes fall on?','三种分法的 pass@1 都是 50%，成功落在哪些任务上？'),
      options:[['even',tx('all at 50%','都是一半')],['mix',tx('mixed','有好有坏')],['split',tx('all or nothing','全有或全无')]],value:tzMix,inline:false})+
    `<svg class="tz-vis tz-pak-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"><span><i class="tz-pak-sw on" aria-hidden="true"></i>${esc(tx('filled: try passed','实心：这次通过'))}</span><span><i class="tz-pak-sw" aria-hidden="true"></i>${esc(tx('empty: try failed','空心：这次没过'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('svg'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  const help=tx('8 tasks, 4 tries each (columns are tasks, rows are tries), with pass@1 and pass@4 as bars.','8 个任务，每个试 4 次（每列一个任务，每行一次尝试），右边的条是 pass@1 和 pass@4。');
  vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const draw=()=>{const W=Math.max(240,Math.round(el.clientWidth||300)), r=svg(W,tzMix); vis.setAttribute('viewBox',`0 0 ${W} ${r.H}`); vis.setAttribute('width',W); vis.setAttribute('height',r.H); vis.innerHTML=r.h;};
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzMix); draw(); const h=readHTML(tzMix); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=b.dataset.v; if(OPTS.includes(v)&&v!==tzMix){tzMix=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/pass-k_numbers.py: the tile's numbers and sentence */
window.__tzPak=Object.freeze({opts:OPTS.slice(),mix:JSON.parse(JSON.stringify(MIX)),p1:m=>p1(m),p4:m=>p4(m),read:m=>readHTML(m)});
})();
