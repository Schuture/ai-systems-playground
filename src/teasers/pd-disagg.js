/* teaser for pd-disagg: one 32,768-token document arrives while 8 chats decode. Colocated on 1 GPU, its prefill stalls every chat;
   split onto 1 prefill + 1 decode GPU, the chats keep their pace. Numbers from the lab's own engine (window.PDD_ENGINE). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
let v='co';
const tx=(en,zh)=>A.lang()==='zh'?zh:en;
const fT=x=>x<1?tx(`${Math.round(x*1000)} ms`,`${Math.round(x*1000)} 毫秒`):tx(`${A.fmt.num(x,2)} s`,`${A.fmt.num(x,2)} 秒`);
function runs(){const E=window.PDD_ENGINE; if(!E) return null;
  return {co:E.runOf('one',{mode:'co',n:1,budget:0},0),split:E.runOf('one',{mode:'split',n:2,p:1,link:'nvl'},0)};}
function bars(run){const T=Math.max(run.end,0.1);
  return run.G.map((g,i)=>{const segs=g.steps.map(s=>{const st=s.pt>0&&s.nd>0&&s.t1-s.t0>0.15, cls=st?'tz-pdd-st':(s.pt>0?'tz-pdd-pf':'tz-pdd-dc');
      return `<i class="${cls}" style="left:${(100*s.t0/T).toFixed(2)}%;width:${Math.max(0.3,100*(s.t1-s.t0)/T).toFixed(2)}%"></i>`;}).join('');
    const lab=g.kind==='pre'?tx('prefill GPU','prefill 的卡'):g.kind==='dec'?tx('decode GPU','decode 的卡'):tx('one GPU for both','一张卡两种都做');
    return `<div class="tz-pdd-row"><span class="tz-pdd-l">${lab}</span><div class="tz-pdd-tr">${segs}</div></div>`;}).join('');}
function render(el){
  const R=runs(); if(!R){el.innerHTML=''; return;}
  const run=R[v], g=run.m.chat.gap_max;
  el.innerHTML=`<div class="tz tz-pdd">`+
   A.ui.seg({id:'tz-pdd-seg',key:'tzpdd',label:tx('Where prefill runs','prefill 在哪跑'),options:[['co',tx('same GPU','同一张卡')],['split',tx('its own GPU','单独一张卡')]],value:v})+
   `<div class="tz-vis tz-pdd-vis">${bars(run)}</div>`+
   `<div class="tz-leg"><span><i class="tz-pdd-sw tz-pdd-pf"></i>prefill</span><span><i class="tz-pdd-sw tz-pdd-dc"></i>decode</span><span><i class="tz-pdd-sw tz-pdd-st"></i>${tx('chats stalled','聊天被卡住')}</span></div>`+
   `<p class="tz-read">${v==='co'?tx(`On one GPU, a 32K-token document’s prefill makes every chat wait ${fT(g)} for its next token.`,`同一张卡上，一篇 32K token 文档的 prefill 让每个聊天等 ${fT(g)}才出下一个 token。`)
     :tx(`With prefill on its own GPU, the chats’ longest wait between tokens is ${fT(g)}.`,`prefill 放到单独一张卡上，聊天两个 token 之间最多只等 ${fT(g)}。`)}</p>`+
   `<p class="tz-note">${tx('Toy run: Llama-3.1-8B on H100s, 8 chats and one 32,768-token document, priced by the lab’s step model (50% of peak compute, 60% of peak bandwidth). The split sends the KV cache over NVLink at 450 GB/s. Not a recording.','玩具实跑：H100 上的 Llama-3.1-8B，8 个聊天请求和 1 篇 32,768 个 token 的文档，按 Lab 的步模型计时（算力按峰值的 50%，带宽按 60%）。分开跑时 KV cache 走 NVLink，450 GB/s。不是真实录制。')}</p></div>`;
  el.querySelector('.seg-ctl').addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b) return; v=b.dataset.v; render(el);});
}
A.registerTeaser('pd-disagg',{fid:'toy',render});
})();
