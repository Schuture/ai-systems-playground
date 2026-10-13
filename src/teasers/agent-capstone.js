/* Teaser for the agent-capstone lab's home tile: one control, “old harness | new harness”, on a small suite of 12 tasks run 3 times
   each (the lab's engine on its own small suite, seed 89; tests/agent-capstone_numbers.py checks every cell against window.__acp).
   The grid shows the 36 runs; the strip shows each harness's success rate with its 95% interval; the sentence gives the conclusion
   first (R11.5): the old harness's rate and how rarely a task succeeds 3 times out of 3; the new harness's rate and that its interval
   clears the old one. The intervals and pass^3, computed here from the cells (mean of the tasks' shares; 1.96 × sample sd ÷ √12;
   tasks with 3 of 3), are in the picture's tooltip.
   It does not answer the lab's part 1 question (which single change helps most), R3.8. State lives in module scope. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
/* o = succeeded, f = failed, u = an unsafe action; one string per run, one letter per task (4 lookups, 5 edits, 3 outside text) */
const RUNS={old:['ooooofoffoff','oooffffffofu','oofffffffufo'],new:['oooooooofofo','oooofoofoooo','ooooofoooooo']};
const KINDS=[4,5,3];
let cur='old';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>fixd(Math.max(0,Math.min(1,x))*100,1)+'%';
function stats(h){
  const R=RUNS[h], n=R[0].length, k=R.length, cnt=[];
  for(let i=0;i<n;i++) cnt.push(R.filter(r=>r[i]==='o').length);
  const s=cnt.map(c=>c/k), m=s.reduce((a,b)=>a+b,0)/n, sd=Math.sqrt(s.reduce((a,x)=>a+(x-m)*(x-m),0)/(n-1)), half=1.96*sd/Math.sqrt(n);
  return {n,k,cnt,succ:m,lo:m-half,hi:m+half,p3:cnt.filter(c=>c===k).length/n,unsafe:R.join('').split('u').length-1};
}
/* S14 (R11.5): the conclusion first, in one or two sentences; the numbers behind it are in the tooltip */
const few=p=>p<0.2?tx('fewer than one task in five succeeds all 3 times','3 次都成功的任务不到五分之一'):tx(`${fP(p)} of the tasks succeed all 3 times`,`3 次都成功的任务占 ${fP(p)}`);
function readHTML(h){const e=stats(h), o=stats('old');
  return h==='old'?tx(`The old harness: only <b>${fP(e.succ)}</b> of the runs succeed, and ${few(e.p3)}.`,
                      `旧 harness 只有 <b>${fP(e.succ)}</b> 的运行成功，${few(e.p3)}。`)
    :(e.lo>o.hi?tx(`The new harness: <b>${fP(e.succ)}</b> succeed, and its interval does not overlap the old one’s, so the gain is not luck.`,`新 harness 的成功率是 <b>${fP(e.succ)}</b>，区间和旧的不重叠，可见这次改进不是运气。`)
               :tx(`The new harness: <b>${fP(e.succ)}</b> succeed, but its interval still overlaps the old one’s.`,`新 harness 的成功率是 <b>${fP(e.succ)}</b>，可区间和旧的还有重叠。`));}
function helpOf(h){const e=stats(h), o=stats('old'), n=stats('new');
  return tx(`Each column is one of 12 tasks, each row one run of all of them: ${e.cnt.reduce((a,b)=>a+b,0)} of 36 runs succeeded${e.unsafe?`, ${e.unsafe} with an unsafe action`:''}. Below, each harness's success rate with its 95% interval: old ${fP(o.lo)} to ${fP(o.hi)}, new ${fP(n.lo)} to ${fP(n.hi)}. pass^3, the share of tasks that succeed all 3 times: old ${fP(o.p3)}, new ${fP(n.p3)}.`,
    `每一列是 12 个任务中的一个，每一行是把它们各跑一次。36 次运行里成功 ${e.cnt.reduce((a,b)=>a+b,0)} 次${e.unsafe?`，${e.unsafe} 次做了不安全的操作`:''}。下面是两个 harness 各自的成功率和 95% 区间，旧的是 ${fP(o.lo)} 到 ${fP(o.hi)}，新的是 ${fP(n.lo)} 到 ${fP(n.hi)}。3 次都成功的任务占多少（pass^3），旧的是 ${fP(o.p3)}，新的是 ${fP(n.p3)}。`);}
function notes(){
  return [tx('Toy run: 12 tasks, each run 3 times, under assumed error rates. The rate, the interval and pass^3 are computed exactly from these runs.',
             '玩具实跑：12 个任务，每个跑 3 次，出错的比例是假设的。成功率、区间和 pass^3 都由这些运行精确算出。'),
          tx('Old: no retries, no check, the whole context kept, no gate on risky actions. New: 2 retries, a check after each step, trim + summary + notes, a person confirms risky actions.',
             '旧：不重试，不检查，上下文全部保留，危险操作不把关。新：重试 2 次，每步检查，上下文裁剪 + 摘要 + 笔记，危险操作要人确认。')];}
function gridHTML(h){
  const R=RUNS[h]; let out='';
  R.forEach(r=>{let i=0; out+='<div class="tz-acp-r">'; KINDS.forEach((cnt,g)=>{if(g) out+='<b></b>'; for(let j=0;j<cnt;j++,i++) out+=`<i class="tz-acp-c ${r[i]}"></i>`;}); out+='</div>';});
  return out;
}
function stripHTML(h){
  const row=(k,lab)=>{const e=stats(k), L=v=>(Math.max(0,Math.min(1,v))*100).toFixed(2)+'%';
    return `<div class="tz-acp-sr${k===h?' on':''}"><span class="tz-acp-sl">${esc(lab)}</span><span class="tz-acp-st"><i class="ci" style="left:${L(e.lo)};width:calc(${L(e.hi)} - ${L(e.lo)})"></i><i class="pt" style="left:${L(e.succ)}"></i></span><span class="tz-acp-sv mono">${fP(e.succ)}</span></div>`;};
  return row('old',tx('old','旧'))+row('new',tx('new','新'))+`<div class="tz-acp-ax mono" aria-hidden="true"><span></span><span class="tz-acp-axs"><span>0%</span><span>50%</span><span>100%</span></span><span></span></div>`;
}
A.registerTeaser('agent-capstone',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-acp">`+
    `<div class="seg-ctl tz-acp-seg" role="group" aria-label="${esc(tx('Harness','harness'))}">`+
      `<button type="button" data-v="old" aria-pressed="${cur==='old'}">${esc(tx('Old harness','旧 harness'))}</button><button type="button" data-v="new" aria-pressed="${cur==='new'}">${esc(tx('New harness','新 harness'))}</button></div>`+
    `<div class="tz-acp-vis" role="img"><div class="tz-acp-grid"></div><div class="tz-acp-strip"></div></div>`+
    `<div class="tz-leg"><span><i class="sw tz-acp-sw-o" aria-hidden="true"></i>${esc(tx('run succeeded','成功的运行'))}</span><span><i class="sw tz-acp-sw-f" aria-hidden="true"></i>${esc(tx('failed','失败'))}</span><span><i class="sw tz-acp-sw-u" aria-hidden="true"></i>${esc(tx('unsafe action','不安全的操作'))}</span><span><i class="sw tz-acp-sw-i" aria-hidden="true"></i>${esc(tx('95% interval','95% 区间'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const seg=el.querySelector('.tz-acp-seg'), vis=el.querySelector('.tz-acp-vis'), grid=el.querySelector('.tz-acp-grid'), strip=el.querySelector('.tz-acp-strip'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{
    seg.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===cur)));
    grid.innerHTML=gridHTML(cur); strip.innerHTML=stripHTML(cur);
    const help=helpOf(cur);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(cur); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  seg.addEventListener('click',ev=>{const b=ev.target.closest('button[data-v]'); if(!b||b.dataset.v===cur) return; cur=b.dataset.v; upd(true);});
  upd(false);
}});
/* read-only hook for tests/agent-capstone_numbers.py */
window.__tzAcp=Object.freeze({RUNS:JSON.parse(JSON.stringify(RUNS)),stats,read:h=>readHTML(h),help:h=>helpOf(h)});
})();
