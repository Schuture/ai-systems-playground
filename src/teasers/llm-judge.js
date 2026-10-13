/* Teaser for the llm-judge lab's home tile: 100 pairs of equally good, equally long answers, judged once (A shown first) or once in
   each order. The judge is the lab's small judge (labs/llm-judge.js): it favours the answer shown first by 0.5 log-odds, says "equally
   good" within ±0.4, and has the same two kinds of noise; the 100 pairs are seeded (mulberry32, seed 502 × 9973 + 11, stratified like
   the lab's). Once, A first: A wins about 58%; both orders: about 50 / 50. These are not the lab's part 1 numbers (R3.8). One control,
   one stacked bar of steady height, one sentence in .tz-read, a one-line legend, and plain-text notes. State lives in module scope. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const N=100, SEED=502, SN=Math.sqrt(3)/Math.PI, LQ=u=>Math.log(u/(1-u));
const J={pi:0.5,st:0.9,se:0.4,tau:0.4};
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function perm(r,n){const a=[...Array(n).keys()]; for(let i=n-1;i>0;i--){const k=Math.floor(r()*(i+1)), t=a[i]; a[i]=a[k]; a[k]=t;} return a;}
let RES=null;
/* B's result per pair: once with A shown first, and in both orders (two different verdicts: a tie) */
function results(){
  if(RES) return RES;
  const r=mulberry(SEED*9973+11), P=[]; for(let j=0;j<7;j++) P.push(perm(r,N));
  const q=(p,i)=>(p[i]+0.5)/N, once=[], both=[];
  for(let i=0;i<N;i++){
    const g=2*(LQ(q(P[0],i))*SN), et=LQ(q(P[6],i)), eAB=LQ(q(P[4],i)), eBA=LQ(q(P[5],i));
    const vote=(o,e)=>{const s=g+J.pi*o+J.st*et+J.se*e; return s>J.tau?1:(s<-J.tau?0:0.5);};
    const a=vote(1,eBA), b=vote(-1,eAB);
    once.push(b); both.push(a===b?a:0.5);
  }
  const sum=v=>{const W=v.filter(x=>x===1).length, T=v.filter(x=>x===0.5).length, Lo=v.filter(x=>x===0).length; return {W,T,Lo,b:(W+T/2)/N};};
  RES={once:sum(once),both:sum(both)}; return RES;
}
const OPTS=['once','both'];
let tzMode='once';
const pct=x=>(Math.round(x*1000)/10).toLocaleString('en-US',{maximumFractionDigits:1})+'%';
function readHTML(m){
  const s=results()[m], a=1-s.b;
  if(m==='once') return tx(`Two equally good answers, judged once with A first: A wins <b>${pct(a)}</b>, B ${pct(s.b)}.`,
                           `两个回答一样好，只判一次、A 放在前面：A 的胜率 <b>${pct(a)}</b>，B 只有 ${pct(s.b)}。`);
  return tx(`The same pairs, once in each order (two different verdicts count as a tie): A <b>${pct(a)}</b>, B ${pct(s.b)}.`,
            `同样的回答，两种顺序各判一次（两次结论不同算平局）：A <b>${pct(a)}</b>，B ${pct(s.b)}。`);
}
function notes(){
  return [tx('Toy run: 100 seeded pairs of answers that are equally good and equally long. The judge favours the answer shown first, an assumption of the lab.',
             '玩具实跑：100 对带种子的回答，内容一样好、长度也一样。裁判偏向排在前面的回答，这是这个 Lab 的假设。'),
    tx('The win rate counts each win as 1 and each tie as half, out of the 100 pairs. Judging each pair in both orders and counting two different verdicts as a tie is what MT‑Bench does (FastChat).',
       '胜率按每赢一次记 1、每平局一次记一半来数，再除以 100 对。每对按两种顺序各判一次、两次结论不同算平局，是 MT‑Bench（FastChat）的做法。')];
}
function barHTML(m){
  const s=results()[m], seg=(cls,n)=>n?`<i class="${cls}" style="width:${n}%"><span>${n>=9?n:''}</span></i>`:'';
  return `<div class="tz-ljg-bar">${seg('tz-ljg-a',s.Lo)}${seg('tz-ljg-t',s.T)}${seg('tz-ljg-b',s.W)}<b class="tz-ljg-50"></b></div>`+
    `<div class="tz-ljg-ax mono"><span>${tx('A wins','A 赢')}</span><span>50%</span><span>${tx('B wins','B 赢')}</span></div>`;
}
A.registerTeaser('llm-judge',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-ljg">`+A.ui.seg({id:'tz-ljg-seg',key:'tzljg',label:tx('How the judge judges','裁判怎样判'),options:[['once',tx('once, A first','判一次，A 在前')],['both',tx('both orders','两种顺序都判')]],value:tzMode,inline:false})+
    `<div class="tz-vis tz-ljg-vis" role="img"></div>`+
    `<div class="tz-leg"><span><i class="tz-ljg-sw tz-ljg-a" aria-hidden="true"></i>${esc(tx('A wins','A 赢'))}</span><span><i class="tz-ljg-sw tz-ljg-t" aria-hidden="true"></i>${esc(tx('tie','平局'))}</span><span><i class="tz-ljg-sw tz-ljg-b" aria-hidden="true"></i>${esc(tx('B wins','B 赢'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-ljg-vis'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzMode); const s=results()[tzMode];
    vis.innerHTML=barHTML(tzMode);
    const lab=tx(`100 verdicts: A wins ${s.Lo}, ties ${s.T}, B wins ${s.W}.`,`100 个结论：A 赢 ${s.Lo}，平局 ${s.T}，B 赢 ${s.W}。`); vis.setAttribute('aria-label',lab); vis.setAttribute('data-tipx',lab);
    const h=readHTML(tzMode); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=b.dataset.v; if(OPTS.includes(v)&&v!==tzMode){tzMode=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/llm-judge_numbers.py: the tile's counts and sentence */
window.__tzLjg=Object.freeze({results:()=>JSON.parse(JSON.stringify(results())),read:m=>readHTML(m),opts:OPTS.slice()});
})();
