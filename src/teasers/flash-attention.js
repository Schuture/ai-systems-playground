/* The FlashAttention lab's home tile: the bytes one Llama-3.1-8B head reads from and writes to HBM in BF16, standard attention against
   FlashAttention (Q tiles of 128 rows), by context length. Same formulas as labs/flash-attention.js (the numbers test checks they agree),
   the papers' count: standard = Q, K, V read, O written, S and P each written and read; FlashAttention = Q read, O written, K and V read
   once per Q tile. Both bars share one log axis from 2 MiB to 256 GiB (ticks ×16), so dragging the context visibly lengthens both bars
   while FlashAttention's stays one step (×2) shorter; what each bar holds is said in the notes (Yixiong's review: "what does each bar
   stand for, and why does only the ratio change?").
   window.__tzFa is a read-only hook for tests/flash-attention_numbers.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc}=A;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const D=128, BR=128, EL=2, NS=[1024,2048,4096,8192,16384,32768,65536,131072];
const M8='Llama‑3.1‑8B';
function traffic(N){const stdBase=EL*4*N*D, sp=EL*4*N*N, faBase=EL*2*N*D, kv=EL*Math.ceil(N/BR)*2*N*D, kv1=EL*2*N*D;
  return {N,stdBase,sp,std:stdBase+sp,faBase,kv,kv1,floor:faBase+kv1,fa:faBase+kv,s:EL*N*N};}
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const MiB=1024*1024, GiB=1024*MiB;
/* the pair in one unit: MiB below 10 GiB, else GiB with one decimal (as the lab) */
const fB=(b,big)=>big?fixd(b/GiB,1)+' GiB':(b/MiB<10?fixd(b/MiB,2):Math.round(b/MiB).toLocaleString('en-US'))+' MiB';
const fS=b=>b>=GiB?fixd(b/GiB,0)+' GiB':Math.round(b/MiB).toLocaleString('en-US')+' MiB';   // S is a power of 2: whole MiB or GiB
const tok=N=>A.fmt&&A.fmt.tok?A.fmt.tok(N):(N/1024)+'K';
let tzI=3;
/* the shared log axis: 2 MiB at the left edge, 256 GiB at the right, 17 doublings */
const AX0=2*MiB, AXD=17, TICKS=[4*MiB,64*MiB,GiB,16*GiB,256*GiB];
const ax=b=>Math.max(0,Math.min(1,Math.log2(b/AX0)/AXD));
const fTick=b=>b>=GiB?(b/GiB)+' GiB':(b/MiB)+' MiB';
function readHTML(i){const r=traffic(NS[i]), big=r.std>=10*GiB;
  return tx(`At ${tok(r.N)} tokens, one head reads and writes <b>${fB(r.std,big)}</b> of HBM with standard attention, and only <b>${fB(r.fa,big)}</b> with FlashAttention.`,
            `${tok(r.N)} token 时，一个 head 要在 HBM 上读写 <b>${fB(r.std,big)}</b>（标准 attention），FlashAttention 只要 <b>${fB(r.fa,big)}</b>。`);}
function notes(i){const r=traffic(NS[i]), s=traffic(131072).s, big=r.std>=10*GiB;
  return A.lang()==='zh'?[
    `每一条是 ${M8} 一个 head 读写 HBM 的字节数，按 BF16 算。HBM 是 GPU 的显存，又大又慢；芯片上的 SRAM 又小又快。标准 attention 读 Q、K、V，写 O，还要把 S 和 P 写到 HBM 再读回来。S 和 P 都给每一对 token 存一个数值。FlashAttention 把 S 和 P 留在 SRAM 里：它读 Q、写 O，每个 128 行的 Q tile 再把 K 和 V 读一遍。`,
    `每个得分，标准 attention 为 S 和 P 搬 8 字节；FlashAttention 重读 K 和 V，摊到每个得分上是 4 字节。所以不管上下文多长，它都只搬一半左右。上下文每翻一倍，两者都变成约 4 倍。FlashAttention 也从不存 S：${tok(r.N)} token 时一个 head 的 S 要 ${fS(r.s)}${r.N<NS[NS.length-1]?`，128K 时要 ${fS(s)}`:''}。`,
    `这里把每次重读 K 和 V 都算作 HBM 流量，和 FlashAttention 论文的算法一样。H100 的 50 MB L2 cache 很可能送出其中不少重读（这是推断，不是这里测的），所以真实流量在 Q、K、V 各读一次、O 写一次的下限（${tok(r.N)} token 时 ${fB(r.floor,big)}）和这里的数之间。FlashAttention 论文在 GPT‑2 medium 上实测是 40.3 GB 对 4.4 GB。Lab 里真的跑一个 12 token 的例子，这里的数字是公式算出的。来源：FlashAttention 论文（Dao 等，2022）、FlashAttention‑2（Dao，2023）、${M8} 的 config.json。`].map(zfix):[
    `Each bar is the bytes one ${M8} head reads from and writes to HBM, in BF16. HBM is the GPU’s large, slow memory; the SRAM on the chip is small and fast. Standard attention reads Q, K and V, writes O, and writes S and P out to HBM and reads them back. S and P each hold one number for every pair of tokens. FlashAttention keeps S and P in SRAM: it reads Q, writes O, and reads K and V once per Q tile of 128 rows.`,
    `Per score, standard attention moves 8 bytes for S and P; FlashAttention’s re-reads of K and V come to 4 bytes per score. So it moves about half at every length. Both grow about 4 times per doubling of the context. FlashAttention also never stores S, which takes ${fS(r.s)} per head at ${tok(r.N)} tokens${r.N<NS[NS.length-1]?` and ${fS(s)} at 128K`:''}.`,
    `This counts every re-read of K and V as HBM traffic, as the FlashAttention paper’s algorithm does. An H100’s 50 MB L2 cache probably serves many of these re-reads (an inference, not measured here), so the real traffic lies between the floor of reading Q, K and V and writing O once (${fB(r.floor,big)} at ${tok(r.N)} tokens) and this number. The FlashAttention paper measured 40.3 GB against 4.4 GB on GPT‑2 medium. The lab runs a 12-token example for real; these numbers are a formula. Sources: the FlashAttention paper (Dao et al., 2022), FlashAttention‑2 (Dao, 2023), ${M8}’s config.json.`];}
A.registerTeaser('flash-attention',{render(el){
  const row=(k,lab)=>`<div class="tz-fa-row" data-k="${k}"><div class="tz-fa-top"><span class="tz-fa-l">${esc(lab)}</span><b class="mono tz-fa-v"></b></div><span class="tz-fa-trk"><i></i></span></div>`;
  el.innerHTML=`<div class="tz tz-fa">`+
    `<label class="tz-lab" for="tz-fa-n"><span>${esc(tx('Context length','上下文长度'))}</span><output class="mono" for="tz-fa-n"></output></label>`+
    `<input type="range" id="tz-fa-n" min="0" max="${NS.length-1}" step="1" value="${tzI}">`+
    `<div class="tz-vis tz-fa-vis" role="img" tabindex="0"><div class="tz-fa-cap">${esc(tx('Bytes one head reads from and writes to HBM, the GPU’s memory','一个 head 读写 HBM（GPU 显存）的字节数'))}</div>`+
     row('std',tx('Standard attention','标准 attention'))+row('fa','FlashAttention')+
     `<div class="tz-fa-ax" aria-hidden="true">${TICKS.map((t,j)=>`<span class="${j===0?'tz-fa-t0':(j===TICKS.length-1?'tz-fa-tn':'')}" style="left:${(ax(t)*100).toFixed(2)}%">${fTick(t)}</span>`).join('')}</div>`+
     `<div class="tz-fa-axn">${esc(tx('Log scale: each tick is 16 times the one before.','对数刻度：每一格是前一格的 16 倍。'))}</div></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const q=s=>el.querySelector(s), inp=q('input'), out=q('output'), vis=q('.tz-fa-vis'), rd=q('.tz-read');
  const upd=user=>{tzI=+inp.value; if(A.ui&&A.ui.setRange) A.ui.setRange(inp);
    const r=traffic(NS[tzI]), big=r.std>=10*GiB;
    out.textContent=tx(`${tok(r.N)} tokens`,`${tok(r.N)} token`); inp.setAttribute('aria-valuetext',out.textContent);
    [['std',r.std],['fa',r.fa]].forEach(([k,v])=>{const e=q(`.tz-fa-row[data-k="${k}"]`); e.querySelector('i').style.width=(ax(v)*100).toFixed(2)+'%'; e.querySelector('.tz-fa-v').textContent=fB(v,big);});
    const lab=tx(`One head at ${tok(r.N)} tokens: standard attention reads and writes ${fB(r.std,big)} of HBM, FlashAttention ${fB(r.fa,big)}. Both bars share one log axis, so a longer context lengthens both.`,`${tok(r.N)} token 时一个 head：标准 attention 读写 HBM ${fB(r.std,big)}，FlashAttention ${fB(r.fa,big)}。两条共用一根对数轴，上下文越长，两条都越长。`);
    vis.setAttribute('aria-label',lab); vis.setAttribute('data-tipx',lab);
    el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes(tzI)[i]||'';});
    const h=readHTML(tzI); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>upd(true));
  upd(false);
}});
window.__tzFa=Object.freeze({traffic:i=>JSON.parse(JSON.stringify(traffic(NS[i]))),read:i=>readHTML(i),notes:i=>notes(i),NS:NS.slice(),bars:i=>{const r=traffic(NS[i]); return {std:ax(r.std),fa:ax(r.fa)};},axis:{lo:AX0,doublings:AXD,ticks:TICKS.map(ax)}});
})();
