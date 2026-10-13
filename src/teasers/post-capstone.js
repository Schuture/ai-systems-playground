/* Teaser for the post-capstone lab's home tile: "SFT, DPO or GRPO: which one fits your task?" One single-choice control says
   what data the team has (example answers, preference pairs, or a checker), and four rows show, for Llama‑3.1‑8B with full
   fine-tuning, which methods can run and how many GPU hours one pass over that data takes. The data and the cost model are the
   lab's own task cards and formulas (labs/post-capstone.js): the support card (3,000 conversations of 512 tokens), the preference
   card (20,000 pairs, a 128-token prompt and two 256-token answers), the challenge card (7,473 problems; the team has only their
   final answers, 8 tokens each, and a checker). SFT runs on all three: on a conversation, on a pair's chosen answer, or on a bare
   final answer (prompt-completion data, as TRL's dataset formats describe); 6N per trained token, 2N per token a frozen model reads, at 50% of 989.5 TFLOP/s; generating
   answers at 64 per GPU at once, one step reading the 16.1 GB of BF16 weights at 3.35 TB/s; GRPO generates 8 answers per prompt;
   PPO trains policy and value model 4 times on each batch; the reward model is trained once over the pairs first. Memory is left
   to the lab. Each row has its own tooltip, the same fields in the same order (R2.11).
   State lives in module scope, so a re-render keeps it. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const N=8030261248, PEAK=989.5e12, U0=0.5, HBM=3.35e12, GEN_B=64, G=8, EP=4, HMAX=4.5;
const DATA={demo:{n:3000,p:128,r:384},pairs:{n:20000,p:128,r:256},checker:{n:7473,p:128,r:256,a:8}};
const METHS=['sft','dpo','ppo','grpo'];
const genRate=GEN_B/Math.max(2*N/HBM,2*N*GEN_B/(PEAK*U0));
/* GPU hours of one pass, or null when the data does not allow the method */
function cost(d,m){const c=DATA[d], t=c.p+c.r, cf=1/(PEAK*U0), rm=6*N*2*t*cf*c.n;
  if(m==='sft') return 6*N*(c.p+(c.a||c.r))*cf*c.n/3600;   /* the completion: a reply, the chosen answer, or a final answer alone */
  if(m==='dpo') return d==='pairs'?(6*N*2*t+2*N*2*t)*cf*c.n/3600:null;
  if(m==='ppo') return d==='pairs'?((c.r/genRate+(2*EP*6*N*t+2*2*N*t)*cf)*c.n+rm)/3600:null;
  if(d==='demo') return null;
  const useRM=d==='pairs';
  return ((G*c.r/genRate+(6*N*G*t+2*N*G*t*(useRM?2:1))*cf)*c.n+(useRM?rm:0))/3600;}
let pick='pairs';
const fixd=(v,k)=>v.toLocaleString('en-US',{minimumFractionDigits:k,maximumFractionDigits:k});
const gh=v=>v>=0.1?fixd(v,2):fixd(v,3);
const NAME=()=>({sft:'SFT',dpo:'DPO',ppo:tx('RM + PPO','奖励模型 + PPO'),grpo:'GRPO'});
const NEED=()=>({demo:tx('needs examples','缺示范回答'),pairs:tx('needs pairs','缺偏好对'),checker:tx('needs a checker','缺判分程序')});
const needOf=m=>m==='sft'?'demo':(m==='grpo'?'checker':'pairs');
const OPTS=()=>[['demo',tx('Examples','示范回答')],['pairs',tx('Preference pairs','偏好对')],['checker',tx('Final answers + checker','最后答案 + 判分程序')]];
function readHTML(){const c=k=>gh(cost(pick,k));
  if(pick==='demo') return tx(`With 3,000 example conversations, only SFT can run: one pass takes <b>${c('sft')} GPU hours</b>.`,`只有 3,000 段示范对话时，只能用 SFT：训练一遍 <b>${c('sft')} GPU 小时</b>。`);
  if(pick==='pairs') return tx(`With 20,000 preference pairs, DPO takes <b>${c('dpo')} GPU hours</b> a pass, reward model + PPO ${c('ppo')}, GRPO ${c('grpo')}. SFT sees only the chosen answers.`,`有 20,000 对偏好时，DPO 训练一遍 <b>${c('dpo')} GPU 小时</b>，奖励模型 + PPO 要 ${c('ppo')}，GRPO 要 ${c('grpo')}。SFT 只能学选中的回答。`);
  return tx(`With final answers and a checker for 7,473 problems, GRPO takes <b>${c('grpo')} GPU hours</b> a pass. SFT is cheaper but teaches no working.`,`只有 7,473 道题的最后答案和判分程序时，GRPO 训练一遍 <b>${c('grpo')} GPU 小时</b>。SFT 便宜得多，可学不到解题过程。`);}
const SFT_ON=()=>({demo:tx('the conversations','示范对话'),pairs:tx('the chosen answers','选中的回答'),checker:tx('the final answers alone','最后答案')});
const rowTip=m=>{const v=cost(pick,m), nm=NAME()[m];
  if(m==='sft') return tx(`${nm} on ${SFT_ON()[pick]}: ${gh(v)} GPU hours for one pass.`,`${nm}（学${SFT_ON()[pick]}）：训练一遍 ${gh(v)} GPU 小时。`);
  return v==null?tx(`${nm}: cannot run, the team has no ${({demo:'examples',pairs:'preference pairs',checker:'checker'})[needOf(m)]}.`,`${nm}：跑不起来，团队手里没有${({demo:'示范回答',pairs:'偏好对',checker:'判分程序'})[needOf(m)]}。`)
    :tx(`${nm}: ${gh(v)} GPU hours for one pass.`,`${nm}：训练一遍 ${gh(v)} GPU 小时。`);};
function visHTML(){const nm=NAME(), nd=NEED();
  return `<div class="tz-pcap-rows">`+METHS.map(m=>{const v=cost(pick,m);
    return `<div class="tz-pcap-row" data-tipx="${esc(rowTip(m))}"><span class="tz-pcap-lab">${esc(nm[m])}</span>`+
      (v==null?`<span class="tz-pcap-no"><i class="sw over"></i>${esc(nd[needOf(m)])}</span>`
        :`<span class="tz-pcap-trk"><i class="tz-pcap-b" style="width:${Math.max(1,Math.min(100,v/HMAX*100)).toFixed(2)}%"></i></span><b class="tz-pcap-v">${esc(gh(v))}</b>`)+`</div>`;}).join('')+`</div>`;}
/* one item per line (R2.11): each note is one line of the tile's “Assumptions and sources” */
function notes(){
  return [tx('Llama‑3.1‑8B with full fine-tuning, one pass over the data of the lab’s example task cards. Memory is checked in the lab, not here.','Llama‑3.1‑8B，全量微调，把 Lab 里示例任务卡上的数据训练一遍。显存不在这里算，Lab 里会算。'),
    tx('Examples: 3,000 support conversations of 512 tokens each.','示范回答：3,000 段客服对话，每段 512 个 token。'),
    tx('Preference pairs: 20,000 pairs, each a 128-token prompt with two 256-token answers.','偏好对：20,000 对，每对是一个 128 个 token 的提问和两个 256 个 token 的回答。'),
    tx('Final answers + checker: 7,473 GSM8K problems with only their final answers (8 tokens each), no worked solutions. SFT on them learns to state answers without the working.','最后答案 + 判分程序：7,473 道 GSM8K 题，只有最后答案（每个 8 个 token），没有解题过程。拿它们做 SFT，模型学到的是不写过程、直接报答案。'),
    tx('GPU hours by the lab’s formulas: a trained token costs 6N and a token read by a frozen model 2N, at 50% of the H100’s peak. Generating answers reads all the weights once per step, for 64 answers per GPU at a time.','GPU 小时按 Lab 里的公式算：训练一个 token 算 6N，冻住的模型读一个 token 算 2N，都按 H100 峰值的 50% 算。生成回答时，每张卡同时生成 64 个，每一步把全部权重读一遍。'),
    tx('GRPO generates 8 answers per prompt. PPO trains its policy and value model 4 times on each batch. A reward model is trained on the pairs first.','GRPO 每个提问生成 8 个回答。PPO 的策略模型和价值模型在每批回答上各训练 4 遍。奖励模型先用偏好对训练。')];}
A.registerTeaser('post-capstone',{fid:'formula',render(el){
  const lab=tx('The data you have','手里的数据');
  el.innerHTML=`<div class="tz tz-pcap">`+
    `<div class="tz-pcap-ctl">${A.ui.seg({id:'tz-pcap-seg',key:'tzpcap',label:lab,options:OPTS(),value:pick,inline:false})}</div>`+
    `<div class="tz-vis tz-pcap-vis" role="group"></div>`+
    `<div class="tz-leg"><span data-tipx="${esc(tx('The bar is the GPU hours of one pass over the data; its scale ends at 4.5.','横条是把数据训练一遍的 GPU 小时，刻度到 4.5。'))}"><i class="sw tz-pcap-sw"></i>${esc(tx('GPU hours, one pass','训练一遍的 GPU 小时'))}</span><span data-tipx="${esc(tx('The method needs data the team does not have.','这个方法要的数据，团队手里没有。'))}"><i class="sw over"></i>${esc(tx('cannot run','跑不起来'))}</span></div>`+
    `<p class="tz-read"></p>${notes().map(()=>'<p class="tz-note"></p>').join('')}</div>`;
  const vis=el.querySelector('.tz-pcap-vis'), rd=el.querySelector('.tz-read'), seg=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{vis.innerHTML=visHTML(); if(seg&&A.ui.setPressed) A.ui.setPressed(seg,pick);
    vis.setAttribute('aria-label',tx(`GPU hours of one pass with ${OPTS().find(o=>o[0]===pick)[1].toLowerCase()}, one row per method`,`用${OPTS().find(o=>o[0]===pick)[1]}训练一遍的 GPU 小时，每个方法一行`));
    const h=readHTML(); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button[data-v]'); if(!b||!el.contains(b)) return; if(b.dataset.v!==pick){pick=b.dataset.v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/post-capstone_numbers.py */
window.__pcapTz=Object.freeze({cost,genRate,DATA:JSON.parse(JSON.stringify(DATA)),pick:()=>pick});
})();
