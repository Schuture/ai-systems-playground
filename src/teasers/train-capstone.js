/* Teaser for the train-capstone lab's home tile: one slider sets the number of H100s (256 → 16,384, doubling) that train
   Llama‑3.1‑70B on 2T tokens, the lab's challenge workload (not part 1's 15T, so the tile does not answer part 1's question, R3.8).
   Every size runs the lab's planner for one fixed plan: TP 8 × PP 8 × ZeRO-1, 1,024 sequences of 4,096 tokens per step, saving
   every 15 minutes asynchronously, with the lab's assumptions (50% of the peak while computing, NVLink 450 GB/s per GPU, 400 GB/s
   of network per 8-GPU machine, a machine failure every 64 days, a 10 s stop per save, 10 minutes per restart). With the step's
   sequences fixed, more GPUs give each pipeline fewer micro-batches (a bigger bubble) and more machines fail more often, so
   doubling the GPUs cuts the days by less than half and the GPU hours rise (review S14: the old tile held MFU at 40% and showed
   “twice the GPUs, half the time”, a misconception the lab corrects). State lives in module scope, so a re-render keeps it. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
/* the lab's planner, for the one plan the tile shows */
const N=70553706496, NL=80, H=8192, D=2e12, PEAK=989.5e12, U0=0.5, SEQ=4096, B=1024, TP=8, PP=8;
const NV=450e9, NODE=400e9, NODE_DAYS=64, RESTART_H=1/6, TAU_H=0.25, STOP_S=10;
const GS=[256,512,1024,2048,4096,8192,16384], DMAX=100, HMAX=2.8e6;
let idx=3;
function run(G){
  const d=G/(TP*PP), m=B/d;
  const comp=6*(N/PP)*SEQ/TP/(PEAK*U0), ar=2*(TP-1)/TP*SEQ*H*2/NV, per=comp+4*(NL/PP)*ar;
  /* ZeRO-1: two data-parallel passes a step, hidden only by the last micro-batch's backward pass */
  const one=TP*d>8?2*N/PP/NODE:(d-1)/d*2*N/(TP*PP)/NV, dpx=d>1?Math.max(0,2*one-per*2/3):0;
  const step=(m+PP-1)*per+dpx, M=NODE_DAYS*24/(G/8), waste=STOP_S/3600/TAU_H+(TAU_H/2+RESTART_H)/M;
  const days=D/(B*SEQ)*step/(1-waste)/86400;
  return {days,gpuh:days*24*G,m,bubble:(PP-1)*per/step,M,waste};
}
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fint=v=>Math.round(v).toLocaleString('en-US');
const fDays=x=>x>=10?fixd(x,1):fixd(x,2);
/* GPU hours: thousands as K, millions as M (R6.18) */
const fH=x=>x<1e6?`${fint(x/1e3)}K`:(A.fmt&&A.fmt.big?A.fmt.big(Math.round(x)):`${fixd(x/1e6,2)}M`);
function readHTML(i){const G=GS[i], r=run(G);
  if(i===0) return tx(`${fint(G)} H100s take about <b>${fDays(r.days)} days</b> and ${fH(r.gpuh)} GPU hours.`,
                      `${fint(G)} 张 H100 约要 <b>${fDays(r.days)} 天</b>，一共 ${fH(r.gpuh)} GPU 小时。`);
  const q=run(GS[i-1]), less=Math.round((1-r.days/q.days)*100), more=Math.round((r.gpuh/q.gpuh-1)*100);
  return tx(`${fint(G)} H100s take about <b>${fDays(r.days)} days</b>, only ${less}% less than ${fint(GS[i-1])}, and use ${more}% more GPU hours.`,
            `${fint(G)} 张 H100 约要 <b>${fDays(r.days)} 天</b>，只比 ${fint(GS[i-1])} 张少 ${less}%，GPU 小时却多了 ${more}%。`);}
function notes(){
  return [tx('Each size runs the lab’s planner for one plan: Llama‑3.1‑70B on 2T tokens, TP 8 × PP 8 × ZeRO-1, 1,024 sequences of 4,096 tokens per step, saving every 15 minutes asynchronously.',
             '每种卡数都用 Lab 里的规划器算同一个计划：Llama‑3.1‑70B 喂 2T 个 token，张量并行 8 份 × 流水线 8 段 × ZeRO-1，每步 1,024 条 4,096 个 token 的序列，每 15 分钟异步存一次档。'),
    tx('The lab’s assumptions hold: the GPUs run at 50% of the peak while computing, a machine’s 8 GPUs share 400 GB/s of network, and each machine fails once every 64 days.',
       '假设和 Lab 里一样：GPU 算的时候能跑到峰值的 50%，一台机器的 8 张卡共用 400 GB/s 的网络带宽，每台机器平均 64 天坏一次。'),
    tx('With the step’s sequences fixed, more GPUs give each pipeline fewer micro-batches, so the bubble grows. More machines also fail more often.',
       '每步的序列数不变，卡越多，每条流水线分到的 micro-batch 越少，bubble 就越大。机器越多，故障也越勤。')];}
function visHTML(i){const r=run(GS[i]), w=Math.min(100,r.days/DMAX*100), wh=Math.min(100,r.gpuh/HMAX*100);
  return `<div class="tz-tcap-rows">`+
    `<div class="tz-tcap-row"><span class="tz-tcap-lab">${esc(tx('days','天数'))}</span><span class="tz-tcap-trk"><i class="tz-tcap-d" style="width:${w.toFixed(2)}%"></i></span><b class="tz-tcap-v">${esc(fDays(r.days))}</b></div>`+
    `<div class="tz-tcap-row"><span class="tz-tcap-lab">${esc(tx('GPU hours','GPU 小时'))}</span><span class="tz-tcap-trk"><i class="tz-tcap-h" style="width:${wh.toFixed(2)}%"></i></span><b class="tz-tcap-v">${esc(fH(r.gpuh))}</b></div>`+`</div>`;}
A.registerTeaser('train-capstone',{fid:'formula',render(el){
  const lab=tx('H100 GPUs','H100 的卡数');
  el.innerHTML=`<div class="tz tz-tcap">`+
    `<label class="tz-lab" for="tz-tcap-in"><span>${esc(lab)}</span><output id="tz-tcap-out">${fint(GS[idx])}</output></label>`+
    `<input type="range" id="tz-tcap-in" min="0" max="${GS.length-1}" step="1" value="${idx}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-tcap-vis" role="img" tabindex="0"></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-tcap-in'), out=el.querySelector('#tz-tcap-out'), vis=el.querySelector('.tz-tcap-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const G=GS[idx], r=run(G); out.textContent=fint(G); inp.value=idx; inp.setAttribute('aria-valuetext',tx(`${fint(G)} GPUs`,`${fint(G)} 张卡`)); if(A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visHTML(idx);
    const help=tx(`Llama‑3.1‑70B on 2T tokens with ${fint(G)} H100s: ${fDays(r.days)} days (the bar’s scale ends at ${DMAX}) and ${fH(r.gpuh)} GPU hours (the scale ends at ${fH(HMAX)}).`,
                  `${fint(G)} 张 H100 训练 Llama‑3.1‑70B，喂 2T 个 token：${fDays(r.days)} 天（横条的刻度到 ${DMAX} 天），${fH(r.gpuh)} GPU 小时（刻度到 ${fH(HMAX)}）。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(idx); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(GS.length-1,Math.round(+inp.value))); if(v!==idx){idx=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/train-capstone_numbers.py */
window.__tcapTz=Object.freeze({days:G=>run(G).days,gpuh:G=>run(G).gpuh,GS:GS.slice()});
})();
