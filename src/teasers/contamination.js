/* Teaser for the contamination lab's home tile: 200 simulated benchmark questions, 40 of them leaked into the training data
   (30 copied, often with a few words changed, 10 reworded). The check flags a question when any run of n words is also found
   in the training data (lm-evaluation-harness and GPT-3 use one found run). The population follows the lab's own rule
   (labs/contamination.js pop(): question lengths 25–55 words, a copy broken by 0–4 changed words, one copy in five only partial,
   reworded and clean questions sharing a few short phrases, some clean ones sharing a 10-word instruction) with its own seed and
   size, so these are not the lab's part 1 numbers (R3.8). One control, one grid of 200 squares, one sentence in .tz-read,
   a one-line legend and plain-text notes. State lives in module scope. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const N=200, B={seed:5,nv:30,np:10,nb:8,kmax:5}, NS=[3,8,13], COLS=25, GP=11;
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
let P=null;
function pop(){if(P) return P; const rnd=mulberry(B.seed);
  const len=[]; for(let i=0;i<N;i++) len.push(25+Math.floor(rnd()*31));
  const idx=[]; for(let i=0;i<N;i++) idx.push(i);
  for(let i=N-1;i>0;i--){const j=Math.floor(rnd()*(i+1)); const t=idx[i]; idx[i]=idx[j]; idx[j]=t;}
  const kind=new Array(N).fill('c'); idx.forEach((i,k)=>{if(k<B.nv) kind[i]='v'; else if(k<B.nv+B.np) kind[i]='p'; else if(k<B.nv+B.np+B.nb) kind[i]='b';});
  const runs=[];
  for(let i=0;i<N;i++){const Li=len[i], t=kind[i]; let sp;
    if(t==='v'){const k=Math.floor(rnd()*B.kmax), pc=rnd()<0.2, Lc=pc?Math.max(8,Math.floor(Li*0.4)):Li;
      const set=new Set(); for(let j=0;j<k;j++) set.add(Math.floor(rnd()*Lc)); const br=[...set].sort((a,b)=>a-b);
      sp=[]; let prev=0; br.forEach(b=>{if(b-prev>0) sp.push(b-prev); prev=b+1;}); if(Lc-prev>0) sp.push(Lc-prev);}
    else {const m=1+Math.floor(rnd()*4); sp=[]; for(let j=0;j<m;j++) sp.push(2+Math.floor(rnd()*3)); if(t==='b') sp.push(10);}
    runs.push(sp);}
  return (P={kind,runs});}
const RES={};
/* flag a question when any run of n words is found; count caught leaks and false alarms */
function check(n){if(RES[n]) return RES[n]; const p=pop(), cls=[]; let caught=0, fa=0, leaked=0;
  for(let i=0;i<N;i++){const lk=p.kind[i]==='v'||p.kind[i]==='p', fl=p.runs[i].some(s=>s>=n);
    if(lk) leaked++; if(lk&&fl) caught++; if(!lk&&fl) fa++; cls.push(lk?(fl?'hit':'miss'):(fl?'fa':'ok'));}
  return (RES[n]={n,caught,leaked,fa,kept:N-caught-fa,cls});}
let tzN=13;
function readHTML(n){const r=check(n);
  return tx(`The check catches <b>${r.caught} of ${r.leaked}</b> leaked questions and ${r.fa?`flags ${r.fa} clean ones by mistake`:'flags no clean question'}.`,
            `${r.leaked} 道泄漏题抓到 <b>${r.caught} 道</b>，${r.fa?`另有 ${r.fa} 道干净题被误报`:'没有误报'}。`);}
function notes(){return [
  tx('Toy run: 200 seeded questions; the lab draws 1,000 by the same rule. 40 of them leaked into the training data.',
     '玩具实跑：200 道带种子的题，Lab 里按同样的规则抽 1,000 道。其中 40 道漏进了训练数据。'),
  tx('Of the leaked questions, 30 were copied (often with a few words changed) and 10 reworded. The rest are clean; some share a short phrase or a common instruction with the web. These are assumptions of the lab.',
     '漏进去的题里，30 道是抄过去的（常被改了几个词），10 道是改写过的。其余是干净题，有些和网上的文字有一句短语或一句常见说明相同。这些是这个 Lab 的假设。'),
  tx('A question is flagged when any run of n words is also found in the training data, after lower-casing and removing punctuation. GPT-3’s authors counted a question as contaminated when any one run was found, with runs of 8 to 13 words depending on the dataset; lm-evaluation-harness, for simplicity, always uses the upper end.',
     '只要有一段连续 n 个词在训练数据里也找得到，就标记这道题（先转小写、去掉标点）。GPT-3 的作者只要找到一段就算污染，片段长度按数据集取 8 到 13 个词；lm-evaluation-harness 为了简单，一律取上限。'),
  tx('Short runs catch more leaks but flag clean questions too; long runs miss copies with a few words changed. A reworded question has no long run in common: at n = 3 it is flagged only because a short phrase happens to match, just like a false alarm on a clean question; with longer runs, none is found.',
     '片段短，抓到的泄漏多，但也会把干净题标出来；片段长，改了几个词的抄题就会漏掉。改写过的题没有长的相同片段：n = 3 时它们被标出来，只是碰巧有短语对上，和干净题被误报一样；n 再长，就一道也查不到。')];}
function gridSVG(n){const r=check(n), rows=N/COLS; let h='';
  for(let i=0;i<N;i++){const x=(i%COLS)*GP, y=Math.floor(i/COLS)*GP; h+=`<rect class="tz-ctm-q ${r.cls[i]}" x="${x}" y="${y}" width="10" height="10" rx="1.5"/>`;}
  return `<svg class="tz-ctm-grid" viewBox="0 0 ${COLS*GP-1} ${rows*GP-1}" aria-hidden="true">${h}</svg>`;}
A.registerTeaser('contamination',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-ctm">`+A.ui.seg({id:'tz-ctm-seg',key:'tzctm',label:tx('Run length n (words)','片段长度 n（词）'),options:NS.map(v=>[String(v),String(v)]),value:String(tzN),inline:false})+
    `<div class="tz-vis tz-ctm-vis" role="img"></div>`+
    `<div class="tz-leg"><span><i class="tz-ctm-sw hit" aria-hidden="true"></i>${esc(tx('leak caught','抓到的泄漏'))}</span><span><i class="tz-ctm-sw miss" aria-hidden="true"></i>${esc(tx('leak missed','漏掉的泄漏'))}</span><span><i class="tz-ctm-sw fa" aria-hidden="true"></i>${esc(tx('false alarm','误报'))}</span><span><i class="tz-ctm-sw ok" aria-hidden="true"></i>${esc(tx('clean','干净'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-ctm-vis'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,String(tzN)); const r=check(tzN);
    vis.innerHTML=gridSVG(tzN);
    const lab=tx(`200 questions, n = ${tzN}: leaks caught ${r.caught}, leaks missed ${r.leaked-r.caught}, false alarms ${r.fa}, clean kept ${N-r.leaked-r.fa}.`,
                 `200 道题，n = ${tzN}：抓到的泄漏 ${r.caught}，漏掉的泄漏 ${r.leaked-r.caught}，误报 ${r.fa}，留下的干净题 ${N-r.leaked-r.fa}。`);
    vis.setAttribute('aria-label',lab); vis.setAttribute('data-tipx',lab);
    const h=readHTML(tzN); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=+b.dataset.v; if(NS.includes(v)&&v!==tzN){tzN=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/contamination_numbers.py: the tile's counts and sentence */
window.__tzCtm=Object.freeze({check:n=>{const r=check(n); return {n:r.n,caught:r.caught,leaked:r.leaked,fa:r.fa,kept:r.kept};},read:n=>readHTML(n),ns:NS.slice()});
})();
