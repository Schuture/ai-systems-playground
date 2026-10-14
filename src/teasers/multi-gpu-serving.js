/* teaser for multi-gpu-serving: Llama-3.1-70B (BF16) on 8 H100s for 32 users with 4,096 tokens of context each. One TP 8 group,
   two TP 4 replicas or four TP 2 replicas: memory per GPU and the time per output token, from the lab's own engine
   (window.MGS_ENGINE: the S2 step model plus 160 all-reduces per step). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
let v='8';
const tx=(en,zh)=>A.lang()==='zh'?zh:en;
const fMs=t=>tx(`${(t*1e3).toFixed(1)} ms`,`${(t*1e3).toFixed(1)} 毫秒`);   // one decimal kept, so 23.0 sits beside 14.5
function res(tp){const E=window.MGS_ENGINE; return E?E.evalSetup({m:'70b',g:8,tp,pp:1,u:32,ctx:4096}):null;}
function vis(r){
  const E=window.MGS_ENGINE, W=300, H=104, pad=2, gap=4, cw=(W-2*pad-7*gap)/8, BH=74, FULL=80*E.GiB, tp=r.c.tp, bad=!r.fits;
  const hW=Math.min(1,r.w/FULL), hO=E.OVH/FULL, hK=Math.min(Math.max(0,1-hW-hO),r.need/FULL);
  let s=`<svg class="tz-vis tz-mgs-vis" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tx('Memory of each of the 8 GPUs and how they are grouped','8 张卡各自的显存和分组')}">`;
  for(let i=0;i<8;i++){const x=pad+i*(cw+gap); let y=2+BH;
    const seg=(h,cls)=>{const hh=h*BH; y-=hh; return hh>0.2?`<rect class="${cls}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${cw.toFixed(1)}" height="${hh.toFixed(1)}"/>`:'';};
    s+=`<rect class="tz-mgs-box${bad?' bad':''}" x="${x.toFixed(1)}" y="2" width="${cw.toFixed(1)}" height="${BH}" rx="2"/>`+seg(hW,'tz-mgs-w')+seg(hO,'tz-mgs-oh')+(bad?'':seg(hK,'tz-mgs-kv'));
    if(bad) s+=`<rect class="tz-mgs-over" x="${x.toFixed(1)}" y="2" width="${cw.toFixed(1)}" height="${(BH*(1-hW-hO)).toFixed(1)}"/>`;}
  for(let g=0;g<8/tp;g++){const x0=pad+g*tp*(cw+gap), x1=x0+tp*(cw+gap)-gap, yb=BH+8;
    s+=`<path class="tz-mgs-br" d="M${x0.toFixed(1)} ${yb-3}V${yb}H${x1.toFixed(1)}V${yb-3}"/><text x="${((x0+x1)/2).toFixed(1)}" y="${yb+13}" text-anchor="middle">TP ${tp}</text>`;}
  return s+'</svg>';}
function sentence(r){
  const t=r.c.tp, ref=res(8);
  if(!r.fits) return tx(`Four replicas of two GPUs: the weights almost fill each GPU, and its 8 users’ KV cache does not fit.`,`四个两张卡的副本：权重几乎占满了每张卡，分到的 8 个用户的 KV cache 放不下。`);
  if(t===8) return tx(`One group of eight: every user gets a token every ${fMs(r.tpot)}, the merges between GPUs included.`,`一个八张卡的组：每个用户每 ${fMs(r.tpot)}拿到一个 token，卡之间的合并也算在内。`);
  return tx(`Two replicas of four GPUs: each reads all the weights every step, so a token takes ${fMs(r.tpot)} instead of ${fMs(ref.tpot)}.`,`两个四张卡的副本：每个副本每一步都读全部权重，一个 token 要 ${fMs(r.tpot)}，而不是 ${fMs(ref.tpot)}。`);}
function render(el){
  const r=res(+v); if(!r){el.innerHTML=''; return;}
  el.innerHTML=`<div class="tz tz-mgs">`+
   A.ui.seg({id:'tz-mgs-seg',key:'tzmgs',label:tx('How to split 8 GPUs','8 张卡怎么分'),options:[['8',tx('1 × TP 8','1 × TP 8')],['4',tx('2 × TP 4','2 × TP 4')],['2',tx('4 × TP 2','4 × TP 2')]],value:v})+
   vis(r)+
   `<div class="tz-leg"><span><i class="sw w"></i>${tx('weights','权重')}</span><span><i class="sw kv"></i>${tx('KV cache','KV cache')}</span><span><i class="sw oh"></i>${tx('overhead','运行时开销')}</span><span><i class="sw over"></i>${tx('does not fit','放不下')}</span></div>`+
   `<p class="tz-read">${sentence(r)}</p>`+
   `<p class="tz-note">${tx('Computed from formulas: Llama-3.1-70B in BF16 on 8 H100s, 32 users with 4,096 tokens of context each, split evenly between the replicas. A GPU can use 70 GiB: 90% of its 80 GB, counted as GiB (vLLM’s default share up to v0.19; 92% from v0.20.0), minus 2 GiB for the runtime. A decode step reads the weights and the KV cache at 60% of 3.35 TB/s (or computes at 50% of peak, whichever is slower), plus 2 all-reduces per layer, 160 per step, each 18 µs plus the ring’s bytes at NVLink’s 450 GB/s one way. Nothing overlaps.','按公式计算：8 张 H100 上的 BF16 Llama-3.1-70B，32 个用户，每人 4,096 个 token 的上下文，在副本之间平分。每张卡能用 70 GiB：80 GB 按 GiB 计，取 90%（vLLM 在 v0.19 及以前的默认值，v0.20.0 起是 92%），再减去运行时的 2 GiB。一步 decode 按 3.35 TB/s 的 60% 读权重和 KV cache（或按峰值算力的 50% 计算，取较慢的），再加每层 2 次 all-reduce，每步 160 次，每次 18 微秒加上环上的字节按 NVLink 单向 450 GB/s 传。各部分不重叠。')}</p></div>`;
  el.querySelector('.seg-ctl').addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b) return; v=b.dataset.v; render(el);});
}
A.registerTeaser('multi-gpu-serving',{fid:'formula',render});
})();
