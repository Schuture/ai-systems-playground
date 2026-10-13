/* Teaser for the rlvr lab's home tile: the lab's toy run (labs/rlvr.js), 60 GRPO steps from the SFT model on the lab's main set of
   problems, with a strict or a loose checker. Two stacked bars: the shares of the four ways of writing before training and after 60
   steps. Numbers come from the lab's engine through window.__rlvr.teaser when the lab is in the build; FALLBACK holds the same values
   (tests/rlvr_numbers.py checks they agree). These are not the lab's part 1 prediction (R3.8): part 1 asks which way of writing grows.
   One control, two bars of steady height, one sentence in .tz-read, a one-line legend and plain-text notes. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const P0=[0.68,0.25,0.05,0.02], T=60;
const FALLBACK={strict:{pi:[0.0233,0.0449,0.9287,0.0032],p1:0.7263,p10:0.3141},loose:{pi:[0.0112,0.0164,0.0410,0.9314],p1:0.1854,p10:0.3141}};
const WAYS=[{en:'answer only',zh:'直接写答案'},{en:'step by step',zh:'一步步算'},{en:'solve, then check',zh:'算完再检查'},{en:'candidate numbers',zh:'写候选数字'}];
const wn=i=>tx(WAYS[i].en,WAYS[i].zh);
const OPTS=['strict','loose'];
let tzChk='strict';
function data(chk){
  try{if(window.__rlvr&&window.__rlvr.teaser){const d=window.__rlvr.teaser(chk); return {pi:d.pi,p1:d.p1,p10:d.p10};}}catch(e){/* use FALLBACK */}
  return FALLBACK[chk];
}
const pct=x=>Math.round(x*100)+'%';
function readHTML(chk){
  const d=data(chk);
  if(chk==='strict') return tx(`After ${T} steps with the strict checker, the answers written “solve, then check” grow from ${pct(P0[2])} to <b>${pct(d.pi[2])}</b> of all answers, and the share right in one try from ${pct(d.p10)} to ${pct(d.p1)}.`,
    `严格判分训练 ${T} 步后，用“算完再检查”写的回答从 ${pct(P0[2])} 涨到 <b>${pct(d.pi[2])}</b>，一次答对的比例从 ${pct(d.p10)} 升到 ${pct(d.p1)}。`);
  return tx(`After ${T} steps with the loose checker, the answers that list candidate numbers grow from ${pct(P0[3])} to <b>${pct(d.pi[3])}</b> of all answers, and the share right in one try falls from ${pct(d.p10)} to ${pct(d.p1)}.`,
    `宽松判分训练 ${T} 步后，写好几个候选数字的回答从 ${pct(P0[3])} 涨到 <b>${pct(d.pi[3])}</b>，一次答对的比例反而从 ${pct(d.p10)} 跌到 ${pct(d.p1)}。`);
}
function notes(){
  return [tx(`This is a simulated policy with only 5 parameters: 4 choose the way of writing, 1 decides whether the answer ends with a “####” line. It trains in your browser for ${T} GRPO steps. Each step takes 8 problems with 8 answers each. A right answer scores 1, a wrong one 0. The problems, the ways of writing and their chances of being right are the lab’s assumptions.`,
             `这是一个模拟的策略模型，只有 5 个参数。其中 4 个决定选哪种写法，1 个决定最后写不写 “####” 这一行。它在浏览器里用 GRPO 训练 ${T} 步，每一步做 8 道题，每道题写 8 个回答。答对得 1 分，答错得 0 分。题目、写法和每种写法答对的概率都是这个 Lab 的假设。`),
    tx('Strict checker: only the number after “####” counts. Loose checker: any number in the answer may equal the reference answer. “Right in one try” is always measured with the strict checker.',
       '严格的判分程序：只看 “####” 后面的数。宽松的判分程序：回答里任何一个数等于标准答案就算对。“一次答对”始终按严格判分来算。')];
}
function barHTML(pi,red){
  return `<div class="tz-rlvr-bar">${pi.map((p,i)=>`<i class="tz-rlvr-w${i}${i===3&&red?' red':''}" style="width:${(p*100).toFixed(1)}%"></i>`).join('')}</div>`;
}
function visHTML(chk){
  const d=data(chk), red=chk==='loose';
  return `<div class="tz-rlvr-row"><span class="tz-rlvr-lb">${esc(tx('before','训练前'))}</span>${barHTML(P0,false)}</div>`+
    `<div class="tz-rlvr-row"><span class="tz-rlvr-lb">${esc(tx(`after ${T} steps`,`${T} 步后`))}</span>${barHTML(d.pi,red)}</div>`;
}
A.registerTeaser('rlvr',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-rlvr">`+A.ui.seg({id:'tz-rlvr-seg',key:'tzrlvr',label:tx('The checker','判分程序'),options:[['strict',tx('strict','严格')],['loose',tx('loose','宽松')]],value:tzChk,inline:false})+
    `<div class="tz-vis tz-rlvr-vis" role="img"></div>`+
    `<div class="tz-leg">${[0,1,2,3].map(i=>`<span><i class="tz-rlvr-sw tz-rlvr-w${i}" aria-hidden="true"></i>${esc(wn(i))}</span>`).join('')}</div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-rlvr-vis'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzChk); const d=data(tzChk);
    vis.innerHTML=visHTML(tzChk);
    const lab=tx(`After ${T} steps: ${[0,1,2,3].map(i=>`${wn(i)} ${pct(d.pi[i])}`).join(', ')}.`,`${T} 步后：${[0,1,2,3].map(i=>`${wn(i)} ${pct(d.pi[i])}`).join('，')}。`);
    vis.setAttribute('aria-label',lab); vis.setAttribute('data-tipx',lab);
    const h=readHTML(tzChk); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=b.dataset.v; if(OPTS.includes(v)&&v!==tzChk){tzChk=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/rlvr_numbers.py */
window.__tzRlvr=Object.freeze({fallback:JSON.parse(JSON.stringify(FALLBACK)),data:c=>JSON.parse(JSON.stringify(data(c))),read:c=>readHTML(c),notes:()=>notes(),legend:()=>[0,1,2,3].map(wn),opts:OPTS.slice()});
})();
