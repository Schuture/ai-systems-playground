/* Home-tile teaser for the Vector index lab: one control (nprobe, the number of IVF cells searched out of the lab's 50),
   the 50 cells as squares (filled when searched), and two bars: the share of the true 10 nearest found and the comparisons
   per question (the 50 centres included) against the 5,000 of exact search. TZ holds the lab engine's measured values (recall %
   and comparisons per question, averaged over 100 questions); tests/vector-index_numbers.py checks them. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const TZ={1:[30.3,153],4:[60.3,457],16:[90.9,1657]};
const NL=50, N=5000, NPS=[1,4,16];
let np=4;
const T={render(el){
  const [rec,cmp]=TZ[np], W=300, H=92, cw=(W-8)/25, I=A.fmt.int;
  const read=b=>{const B=t=>b?`<b>${t}</b>`:t; return tx(`Searching ${B(`${np} of 50`)} cells finds ${B(`${A.fmt.num(rec,1)}%`)} of the true nearest.`,`查 50 个簇里的 ${B(`${np} 个`)}，找回 ${B(`${A.fmt.num(rec,1)}%`)} 的真正近邻。`);};
  // the picture's label adds how the comparisons split: the 50 centres, then the vectors in the cells searched (review b10r1, S1)
  const aria=()=>read(false)+' '+tx(`The question is compared with the 50 centres, then with about ${I(cmp-50)} vectors in ${np===1?'that cell':'those cells'}: ${I(cmp)} comparisons in all.`,`问题先和 50 个中心比，再和${np===1?'这个簇':'这些簇'}里约 ${I(cmp-50)} 个向量比，一共比较 ${I(cmp)} 次。`);
  let h='';
  for(let i=0;i<NL;i++){const x=4+cw*(i%25), y=4+(i<25?0:14); h+=`<rect class="tz-vix-c${i<np?' on':''}" x="${(x+1).toFixed(1)}" y="${y}" width="${(cw-2).toFixed(1)}" height="11" rx="2"/>`;}
  const bw=W-8, bar=(y,v,cls,lab)=>`<text x="4" y="${y-3}">${esc(lab)}</text><rect class="tz-vix-t" x="4" y="${y}" width="${bw}" height="9" rx="3"/><rect class="${cls}" x="4" y="${y}" width="${Math.max(2,bw*v).toFixed(1)}" height="9" rx="3"/>`;
  h+=bar(48,rec/100,'tz-vix-r',tx(`true 10 nearest found: ${A.fmt.num(rec,1)}%`,`找回的真正最近 10 个：${A.fmt.num(rec,1)}%`));
  h+=bar(80,cmp/N,'tz-vix-m',tx(`comparisons: ${A.fmt.int(cmp)} (50 centres included)`,`比较 ${A.fmt.int(cmp)} 次（含 50 个中心）`));
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-vix-np',key:'tzvnp',label:tx('Cells searched, nprobe','要查的簇数 nprobe'),options:NPS.map(v=>[String(v),String(v)]),value:String(np)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(aria())}">${h}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-vix-swon"></i>${tx('cell searched','查了的簇')}</span><span><i class="sw tz-vix-swoff"></i>${tx('cell skipped','跳过的簇')}</span></div>`+
    `<p class="tz-read">${read(true)}</p>`+
    `<p class="tz-note">${tx('Measured in this lab: 5,000 seeded 32-dimensional vectors in 200 loose clusters, an IVF index with 50 cells (k-means), 100 questions; counts include comparing the question with the 50 centres. A full bar is the 5,000 comparisons of exact search.','在这个 Lab 里实测：5,000 个固定种子生成的 32 维向量（200 个松散的团），IVF 索引 50 个簇（k-means），100 个问题；计数包括问题和 50 个中心的比较。条的满格是精确搜索的 5,000 次比较。')}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzvnp"] button').forEach(b=>b.addEventListener('click',()=>{np=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('vector-index',T);
})();
