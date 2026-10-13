/* Teaser for the model-parallel lab's home tile: one slider sets the number of micro-batches m (1 → 32) of a 4-stage pipeline
   (the lab uses 8 stages, so the tile does not answer part 1's prediction, R3.8). A tiny GPipe Gantt chart, one row per GPU:
   outlined cells are forwards (1 unit), solid cells backwards (2 units), hatched time idle. The rows keep their height.
   Formula only: bubble = (p − 1) ÷ (m + p − 1) (zero-bubble pipeline README); the drawing places every cell by its dependencies.
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const P=4, MS=[1,2,4,8,16,32];
let idx=4;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=x=>(x<0.1?fixd(x*100,1):fixd(x*100,0))+'%';
/* GPipe: all forwards, then all backwards; F(i, j) after F(i − 1, j); B(i, j) after B(i + 1, j) (the last stage after its own F) */
function gpipe(p,m){
  const eF=[...Array(p)].map(()=>[]), eB=[...Array(p)].map(()=>[]), rows=[];
  for(let i=0;i<p;i++){let t=0; const r=[];
    for(let j=0;j<m;j++){const s=Math.max(t,i?eF[i-1][j]:0); t=s+1; eF[i][j]=t; r.push({k:'F',j,s,e:t});}
    rows.push(r);}
  for(let i=p-1;i>=0;i--){let t=eF[i][m-1];
    for(let j=0;j<m;j++){const s=Math.max(t,i===p-1?eF[i][j]:eB[i+1][j]); t=s+2; eB[i][j]=t; rows[i].push({k:'B',j,s,e:t});}}
  const T=Math.max(...rows.map(r=>r[r.length-1].e));
  return {rows,T,bubble:(p-1)/(m+p-1)};
}
function model(i){const m=MS[i], g=gpipe(P,m); return {m,T:g.T,bubble:g.bubble,rows:g.rows,busy:m*3/g.T};}
function readHTML(i){const r=model(i);
  return tx(`4 stages, ${r.m} ${r.m===1?'micro-batch':'micro-batches'}: each GPU sits idle <b>${pct(r.bubble)}</b> of the step (the bubble).`,
            `4 段流水线、${r.m} 个 micro-batch：每张卡有 <b>${pct(r.bubble)}</b> 的时间在空等（bubble）。`);}
function notes(){
  return [tx('Formula: bubble = (p − 1) ÷ (m + p − 1), with p stages and m micro-batches (the zero-bubble pipeline README). Here p = 4; the lab uses 8.',
             '公式：bubble = (p − 1) ÷ (m + p − 1)，p 是流水线的段数，m 是 micro-batch 的数量（zero-bubble pipeline 的 README）。这里 p = 4，Lab 里用 8。'),
    tx('Assumed: the GPipe order (all forwards, then all backwards); a backward takes twice as long as a forward; the time to pass data between stages is left out.',
       '假设：按 GPipe 的顺序排（先做完所有前向，再做反向）；反向的时间是前向的 2 倍；段与段之间传数据的时间不计。')];}
/* owner b8 (2): cells must stay apart. u = pixels per time unit (a forward is 1 unit, trk = the track's width); when a forward would
   be narrower than 6 px, g micro-batches in a row (a power of 2) are drawn as one block; busy blocks keep a 1 px gap. GPipe's
   forwards and backwards each run without idle time in between, so a block never hides idle time. */
const groupOf=(u,m)=>{let g=1; while(g<m&&g*u<6) g*=2; return g;};
function visHTML(i,trk){const r=model(i), W=x=>(x/r.T*100).toFixed(3), u=(trk||260)/r.T, g=groupOf(u,r.m);
  return {g,html:`<div class="tz-mp-rows">`+r.rows.map((row,k)=>{let cells='', last=0; const runs=[];
    row.forEach(o=>{const l=runs[runs.length-1]; if(l&&l.k===o.k&&l.e===o.s&&l.n<g){l.e=o.e; l.n++;} else runs.push({k:o.k,s:o.s,e:o.e,n:1});});
    runs.forEach(o=>{if(o.s>last) cells+=`<i class="tz-mp-i" style="left:${W(last)}%;width:${W(o.s-last)}%"></i>`; cells+=`<i class="tz-mp-${o.k==='F'?'f':'b'}" style="left:${W(o.s)}%;width:calc(${W(o.e-o.s)}% - 1px)"></i>`; last=o.e;});
    if(last<r.T) cells+=`<i class="tz-mp-i" style="left:${W(last)}%;width:${W(r.T-last)}%"></i>`;
    return `<div class="tz-mp-row"><span class="tz-mp-lab">${esc(tx(`GPU ${k+1}`,`卡 ${k+1}`))}</span><span class="tz-mp-trk">${cells}</span></div>`;}).join('')+`</div>`};}
A.registerTeaser('model-parallel',{fid:'formula',render(el){
  const lab=tx('Micro-batches per step','每步切成几个 micro-batch');
  el.innerHTML=`<div class="tz tz-mp">`+
    `<label class="tz-lab" for="tz-mp-in"><span>${esc(lab)}</span><output id="tz-mp-out">${MS[idx]}</output></label>`+
    `<input type="range" id="tz-mp-in" min="0" max="${MS.length-1}" step="1" value="${idx}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-mp-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-mp-swf" aria-hidden="true"></i>${esc(tx('forward','前向'))}</span><span><i class="sw tz-mp-swb" aria-hidden="true"></i>${esc(tx('backward','反向'))}</span><span><i class="sw tz-mp-swi" aria-hidden="true"></i>${esc(tx('idle','空等'))}</span><span class="tz-mp-gn"></span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-mp-in'), out=el.querySelector('#tz-mp-out'), vis=el.querySelector('.tz-mp-vis'), rd=el.querySelector('.tz-read'), gn=el.querySelector('.tz-mp-gn');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const r=model(idx); out.textContent=String(r.m); inp.value=idx; inp.setAttribute('aria-valuetext',tx(`${r.m} micro-batches`,`${r.m} 个 micro-batch`)); if(A.ui.setRange) A.ui.setRange(inp);
    const v=visHTML(idx,Math.max(0,vis.clientWidth-52)); vis.innerHTML=v.html;
    /* the note keeps its place at every position (hidden at one per block), so the tile's height does not change (R7.9) */
    gn.textContent=tx(`one block = ${Math.max(2,v.g)} micro-batches`,`一块 = ${Math.max(2,v.g)} 个 micro-batch`); gn.style.visibility=v.g>1?'visible':'hidden';
    const help=tx(`One training step of a 4-stage pipeline with ${r.m} ${r.m===1?'micro-batch':'micro-batches'}: one row per GPU, time to the right${v.g>1?`; each block holds ${v.g} micro-batches in a row`:''}. Each GPU is busy ${pct(r.busy)} of the step and idle ${pct(r.bubble)}.`,
                  `4 段流水线、${r.m} 个 micro-batch 的一步训练：每行是一张卡，往右是时间${v.g>1?`，每块是连着的 ${v.g} 个 micro-batch`:''}。每张卡有 ${pct(r.busy)} 的时间在干活，${pct(r.bubble)} 在空等。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(idx); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(MS.length-1,Math.round(+inp.value))); if(v!==idx){idx=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/model-parallel_numbers.py */
window.__tzMp=Object.freeze({P,MS:MS.slice(),model:i=>{const r=model(i); return {m:r.m,T:r.T,bubble:r.bubble,busy:r.busy};},read:i=>readHTML(i)});
})();
