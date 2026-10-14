/* Teaser for the agent-reliability lab's home tile: one slider sets the number of steps in a task (1 to 40), each right 90% of the
   time (an assumed rate, not the lab's 95% and 20 steps, so the tile does not answer part 1's prediction, R3.8). The row shows one
   cell per step; the bar shows the chance that every step is right, 0.9ⁿ (the formula, no simulation). One control, one small visual,
   one sentence in .tz-read, a one-line legend, plain-text notes. State lives in module scope, so a re-render keeps the setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const P=0.9, NMAX=40;
let n=10;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>fixd(x*100,1)+'%';
const succ=k=>Math.pow(P,k);
const stepsW=k=>tx(k===1?'1 step':`${k} steps`,`${k} 步`);
function readHTML(k){const s=succ(k), fail=100-Math.round(s*100);
  return tx(`At 90% per step, a ${k===1?'1-step':`${k}-step`} task succeeds <b>${fP(s)}</b> of the time: about ${fail} in 100 tasks fail at some step.`,
            `每步 90% 的把握，${k} 步的任务成功率是 <b>${fP(s)}</b>：100 个任务里，约有 ${fail} 个会在某一步出错。`);}
function notes(){
  return [tx('Formula: the task succeeds only if every step is right, so each extra step multiplies the success by one more 0.9. The lab shows the formula.',
             '公式：每一步都做对，任务才成功，所以每多一步，成功率就多乘一个 0.9。公式在 Lab 里。'),
          tx('Here each step is assumed to be right 90% of the time.','这里假设每步做对的把握是 90%。')];}
function visHTML(k){
  const cells=Array.from({length:NMAX},(_,i)=>`<i class="tz-arl-c${i<k?' on':''}"></i>`).join('');
  return `<div class="tz-arl-row" aria-hidden="true">${cells}</div>`+
    `<div class="tz-arl-bar"><i style="width:${(succ(k)*100).toFixed(2)}%"></i><span>${fP(succ(k))}</span></div>`;
}
A.registerTeaser('agent-reliability',{fid:'formula',render(el){
  el.innerHTML=`<div class="tz tz-arl">`+
    `<label class="tz-lab" for="tz-arl-in"><span>${esc(tx('Steps in the task','任务的步数'))}</span><output id="tz-arl-out"></output></label>`+
    `<input type="range" id="tz-arl-in" min="1" max="${NMAX}" step="1" value="${n}" aria-label="${esc(tx('Steps in the task','任务的步数'))}">`+
    `<div class="tz-arl-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-arl-sw" aria-hidden="true"></i>${esc(tx('one step, 90% right','一步，90% 的把握'))}</span><span><i class="sw tz-arl-sw2" aria-hidden="true"></i>${esc(tx('chance the whole task succeeds','整个任务成功的把握'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-arl-in'), out=el.querySelector('#tz-arl-out'), vis=el.querySelector('.tz-arl-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{out.textContent=stepsW(n); inp.value=n; inp.setAttribute('aria-valuetext',stepsW(n)); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visHTML(n);
    const help=tx(`One cell per step: ${stepsW(n)}, each right 90% of the time. The bar is the chance that all of them are right: ${fP(succ(n))}.`,`每一格是一步：${stepsW(n)}，每步 90% 的把握。下面的条是全部做对的把握：${fP(succ(n))}。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(n); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(1,Math.min(NMAX,Math.round(+inp.value))); if(v!==n){n=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/agent-reliability_numbers.py */
window.__tzArl=Object.freeze({succ,read:k=>readHTML(k),P,NMAX});
})();
