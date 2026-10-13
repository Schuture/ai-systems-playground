/* Teaser for the context-eng lab's home tile: one choice of what the agent keeps (everything / the last 10 steps / a summary every
   10 steps plus a notes file) on a toy run of 40 steps. Toy numbers, not the lab's coding task (3,000 tokens a step, a 64K window
   instead of 60 uneven steps and 128K), so the tile does not answer part 1's prediction (R3.8). The row shows, at step 36, which
   steps are still in the context (whole, folded into the summary, dropped); the bar shows that step's context against the window;
   the fact written at step 2 is needed at step 36. One control, one small visual, one sentence in .tz-read, a one-line legend,
   plain-text notes. State lives in module scope, so a re-render keeps the choice. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const N=40, H=2000, PER=3000, W=65536, SUMM=2000, EVERY=10, KEEP=10, FACT=2, NEED=36, FETCH=150, SCALE=80000;
const fI=n=>n.toLocaleString('en-US');
let pol='all';
/* the context of the call at step k: the head, the summary if any, every kept step (each 3,000 tokens) */
function ctxAt(p,k){
  if(p==='all') return H+PER*k;
  if(p==='last') return H+PER*Math.min(k,KEEP);
  const folded=Math.floor((k-1)/EVERY)*EVERY;           // steps already folded into the summary before step k
  return H+(folded?SUMM:0)+PER*(k-folded)+(k===NEED?FETCH:0);
}
/* the first step whose context is over the window (0: none) */
function overAt(p){for(let k=1;k<=N;k++) if(ctxAt(p,k)>W) return k; return 0;}
/* the state of step j in the context of the call at step k */
function cellAt(p,j,k){
  if(j>k) return 'none';
  if(p==='all') return 'kept';
  if(p==='last') return j>k-KEEP?'kept':'drop';
  const folded=Math.floor((k-1)/EVERY)*EVERY; return j>folded?'kept':'summ';
}
/* does step 36 still have the fact from step 2? */
function factOk(p){if(overAt(p)) return false; if(p==='notes') return true; return cellAt(p,FACT,NEED)==='kept';}
const OPTS=()=>[{v:'all',l:tx('Keep everything','全部留着')},{v:'last',l:tx('Last 10 steps','只留最近 10 步')},{v:'notes',l:tx('Summary + notes','摘要 + 笔记')}];
function readHTML(p){
  const o=overAt(p);
  if(p==='all') return tx(`Keeping everything, the context passes the 64K window at <b>step ${o}</b>. The run stops there, before step ${NEED} can use the fact from step ${FACT}.`,
                          `全部留着的话，上下文到<b>第 ${o} 步</b>就超过 64K 窗口，任务停在那里，等不到第 ${NEED} 步用上第 ${FACT} 步的那个事实。`);
  if(p==='last') return tx(`Keeping the last 10 steps, the context never passes ${fI(ctxAt('last',N))} tokens, but the fact from step ${FACT} is <b>gone</b> by step ${NEED}.`,
                           `只留最近 10 步，上下文最多 ${fI(ctxAt('last',N))} 个 token，可第 ${FACT} 步的事实到第 ${NEED} 步已经<b>没了</b>。`);
  return tx(`With a summary every 10 steps and a notes file, step ${NEED} sends ${fI(ctxAt('notes',NEED))} tokens and <b>reads the fact back</b> from the notes.`,
            `每 10 步写一次摘要，再记一份笔记：第 ${NEED} 步只发 ${fI(ctxAt('notes',NEED))} 个 token，事实<b>从笔记里读回来</b>。`);
}
function notes(){
  return [tx('Toy numbers, not the lab’s task: each step adds 3,000 tokens, the window is 64K and a summary is 2,000 tokens.',
             '玩具数字，和 Lab 里的任务不一样：每一步加 3,000 个 token，窗口 64K，摘要 2,000 个 token。'),
          tx('Every call sends the whole context again, so whatever is kept is read again at every step.',
             '每次调用都把整个上下文再发一遍，所以留着的东西，每一步都要再读一次。')];
}
function visHTML(p){
  const o=overAt(p), k=o||NEED, c=ctxAt(p,k), cells=[];
  for(let j=1;j<=N;j++) cells.push(`<i class="tz-cte-c ${cellAt(p,j,k)}${j===FACT?' f':''}"></i>`);
  const w=Math.min(100,c/SCALE*100), wl=W/SCALE*100;
  const lab=o?tx(`step ${o}: ${fI(c)} tokens, over the window`,`第 ${o} 步：${fI(c)} 个 token，超出窗口`):tx(`step ${NEED}: ${fI(c)} tokens`,`第 ${NEED} 步：${fI(c)} 个 token`);
  const fact=o?tx(`the fact: the run never got to step ${NEED}`,`那个事实：运行没走到第 ${NEED} 步`):(factOk(p)?tx('the fact: ✓ there','那个事实：✓ 还在'):tx('the fact: ✗ gone','那个事实：✗ 没了'));
  return `<div class="tz-cte-row" aria-hidden="true">${cells.join('')}</div>`+
    `<div class="tz-cte-bar${o?' over':''}" aria-hidden="true"><i style="width:${w.toFixed(1)}%"></i><b style="left:${wl.toFixed(1)}%"></b></div>`+
    `<div class="tz-cte-cap" aria-hidden="true"><span>${esc(lab)}</span><span class="tz-cte-fact ${o?'na':(factOk(p)?'ok':'bad')}">${esc(fact)}</span></div>`;
}
A.registerTeaser('context-eng',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-cte">`+
    `<div class="tz-lab"><span>${esc(tx('What the agent keeps','agent 留下什么'))}</span></div>`+
    `<div class="seg-ctl tz-cte-seg" role="group" aria-label="${esc(tx('What the agent keeps','agent 留下什么'))}">${OPTS().map(o=>`<button type="button" data-v="${o.v}" aria-pressed="${o.v===pol}">${esc(o.l)}</button>`).join('')}</div>`+
    `<div class="tz-cte-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-cte-sw-k" aria-hidden="true"></i>${esc(tx('kept whole','原样留着'))}</span><span><i class="sw tz-cte-sw-s" aria-hidden="true"></i>${esc(tx('in the summary','并进摘要'))}</span><span><i class="sw tz-cte-sw-d" aria-hidden="true"></i>${esc(tx('dropped','删掉了'))}</span><span><i class="sw tz-cte-sw-n" aria-hidden="true"></i>${esc(tx('not reached yet','还没走到'))}</span><span><i class="sw tz-cte-sw-f" aria-hidden="true"></i>${esc(tx(`the fact, step ${FACT}`,`第 ${FACT} 步的事实`))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-cte-vis'), rd=el.querySelector('.tz-read'), seg=el.querySelector('.tz-cte-seg');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{
    seg.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===pol)));
    vis.innerHTML=visHTML(pol);
    const o=overAt(pol), k=o||NEED;
    const help=tx(`One cell per step of a 40-step toy run, as the call at step ${k} sees them. The bar is that call’s context, ${fI(ctxAt(pol,k))} tokens, against the 64K window (the mark).`,
                  `一格是 40 步玩具任务里的一步，按第 ${k} 步那次调用看到的样子画。下面的条是那次调用的上下文，${fI(ctxAt(pol,k))} 个 token，竖线是 64K 窗口。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(pol); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||b.dataset.v===pol) return; pol=b.dataset.v; upd(true);});
  upd(false);
}});
/* read-only hook for tests/context-eng_numbers.py */
window.__tzCte=Object.freeze({ctxAt,overAt,cellAt,factOk,read:p=>readHTML(p),consts:{N,H,PER,W,SUMM,EVERY,KEEP,FACT,NEED,FETCH}});
})();
