/* post-capstone (world post): SFT, DPO or GRPO: which one fits your task?
   Fidelity: Analytical (shown as Formula / 公式计算; integrator, 2026-10-05: memory, GPU hours and the metric rule are all formulas). Memory and compute come from batch 9's shared formulas (brief S4–S5); how far each method moves each task's
   metric is this lab's own rule, stated on the page. tests/post-capstone_numbers.py recomputes every number independently.
   - memory per GPU (GB = 10⁹ bytes, 80 GB per H100), the lora lab's model: full fine-tuning 18 bytes per parameter; LoRA 2 bytes per
     frozen parameter + 16 per LoRA parameter (r = 16 on all 7 matrices); QLoRA the linear weights in NF4 with double quantization
     (0.515869 bytes), the embedding table, LM head and norms in BF16; reference model 2 bytes per parameter (none with LoRA / QLoRA:
     the adapter is switched off, as TRL does); reward model 2; value model like the policy. All model bytes split evenly over the
     card's GPUs; each GPU adds the activations (167,936 / 331,776 bytes per token per layer) and logits (4 × 128,256 bytes per token)
     of one 1,024-token sequence;
   - compute: trained tokens × 6N (4N with LoRA / QLoRA), frozen forward passes × 2N, at 50% of 989.5 TFLOP/s; generation: 64 answers
     per GPU at once, one step reads the BF16 weights (2N bytes ÷ 3.35 TB/s) and adds a token to each answer;
   - the metric after x passes over the data (the lab's rule, typeset in Show the math): it climbs from its value before training
     toward a ceiling and closes half of the gap left every h passes. A method is blocked only when the card's data cannot form its
     training examples; SFT on final answers alone (challenge) or on the chosen answers alone (preference card) runs, and by the
     lab's conservative rule gains nothing (its ceiling is its value before training).
   - the GPU hours count all the card's GPUs together; the stage shows their inputs (tokens, work per token, the assumed 50% of the
     peak) and a measured check (TT: torchtune's 70B LoRA run, about 40% of the peak).
   window.__pcap is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='post-capstone';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const NB='‑';

/* =====================================================================
   engine
   ===================================================================== */
const PEAK=989.5e12, U0=0.5, HBM=3.35e12, CAP=80e9, VOC=128256, SEQ=1024, GEN_B=64, G_ANS=8, PPO_EP=4, RANK=16, Z=1.96;
const NF4=0.5+1/64+4/(64*256);
const MOD={
 '8b':{id:'8b',name:`Llama${NB}3.1${NB}8B`,N:8030261248,L:32,d:4096,H:32,KV:8,hd:128,f:14336},
 '70b':{id:'70b',name:`Llama${NB}3.1${NB}70B`,N:70553706496,L:80,d:8192,H:64,KV:8,hd:128,f:28672}
};
function parts(m){
  const q=m.H*m.hd, kv=m.KV*m.hd, sh=[[m.d,q],[m.d,kv],[m.d,kv],[q,m.d],[m.d,m.f],[m.d,m.f],[m.f,m.d]];
  const lin=m.L*sh.reduce((a,x)=>a+x[0]*x[1],0), eo=2*VOC*m.d, nrm=m.N-lin-eo, lp=m.L*RANK*sh.reduce((a,x)=>a+x[0]+x[1],0);
  const aL=2*(2*m.d+2*q+2*kv)+2*(2*m.d+4*m.f);
  return {lin,eo,nrm,lp,aL};
}
const PT={'8b':parts(MOD['8b']),'70b':parts(MOD['70b'])};
/* the task cards (示例, written for this lab). data: demo = whole example answers (conversations, worked solutions), ans = final answers
   only, pairs = preference pairs, checker = a program that checks the final answer. tok.a: a final answer alone ("#### 18" and the end
   token), the completion of answer-only SFT. Rules (this lab's): SFT learns to write what its completions contain, so on the
   challenge card (final answers, no working) and on the preference card (the chosen answers alone, never the rejected one) it gets
   no gain (the ceiling is the value before training): no reachable source gives a number for either. */
const CARDS={
 fmt:{id:'fmt',m:'8b',inst:true,gpus:1,budget:1,data:{demo:1,ans:0,pairs:0,checker:0},n:3000,tok:{p:128,r:384},metric:'fmt',m0:40,target:95,nEval:200,rule:{sft:{mi:98,h:0.5}}},
 pref:{id:'pref',m:'8b',inst:true,gpus:8,budget:3,data:{demo:0,ans:0,pairs:1,checker:0},n:20000,tok:{p:128,r:256},metric:'win',m0:50,target:60,nEval:500,rule:{sft:{mi:50,h:0.5},dpo:{mi:70,h:0.5},ppo:{mi:70,h:0.5},grpo:{mi:70,h:0.5}}},
 math:{id:'math',m:'8b',inst:false,gpus:16,budget:10,data:{demo:1,ans:0,pairs:0,checker:1},n:7473,tok:{p:128,r:256},metric:'acc',m0:39.7,target:70,nEval:1319,rule:{sft:{mi:60,h:0.5},grpo:{mi:79.2,h:1}}},
 big:{id:'big',m:'70b',inst:true,gpus:1,budget:2,data:{demo:1,ans:0,pairs:0,checker:0},n:3000,tok:{p:128,r:384},metric:'fmt',m0:40,target:95,nEval:200,rule:{sft:{mi:98,h:0.5}}},
 ch:{id:'ch',m:'8b',inst:false,gpus:1,budget:3,data:{demo:0,ans:1,pairs:0,checker:1},n:7473,tok:{p:128,r:256,a:8},metric:'acc',m0:39.7,target:65,nEval:1319,rule:{sft:{mi:39.7,h:0.5},grpo:{mi:79.2,h:1}}}
};
const CARD_IDS=['fmt','pref','math','big'];
const METHS=['sft','dpo','ppo','grpo'], MODES=['full','lora','qlora'], PASSES=[0.5,1,2,3,4];
const genRate=m=>GEN_B/Math.max(2*m.N/HBM,2*m.N*GEN_B/(PEAK*U0));   /* tokens per second per GPU */
/* what SFT's completions are on a card: whole example answers, final answers alone, or the chosen answers of the pairs
   (TRL: a preference dataset becomes prompt-completion data by dropping the rejected answer) */
const sftFrom=c=>c.data.demo?'demo':(c.data.ans?'ans':(c.data.pairs?'chosen':''));
/* a method is blocked only when its training examples cannot be formed from the card's data */
function needOf(c,meth){
  if(meth==='sft') return sftFrom(c)?'':'demo';
  if(meth==='dpo'||meth==='ppo') return c.data.pairs?'':'pairs';
  return c.data.checker||c.data.pairs?'':'checker';
}
/* trained model bytes, by block */
function trainedBlocks(mk,mode){const m=MOD[mk], p=PT[mk], N=m.N;
  if(mode==='full') return {w:2*N,g:4*N,o:12*N};
  if(mode==='lora') return {w:2*N,lr:16*p.lp};
  return {q4:p.lin*NF4,w:2*(p.eo+p.nrm),lr:16*p.lp};}
function plan(s){
  const c=CARDS[s.card], m=MOD[c.m], pt=PT[c.m], N=m.N, G=s.G||G_ANS, x=s.x, kT=s.mode==='full'?6:4, cf=1/(PEAK*U0);
  const miss=needOf(c,s.meth), can=!miss, useRM=s.meth==='ppo'||(s.meth==='grpo'&&!c.data.checker);
  /* memory: every model's bytes over the GPUs, then this GPU's activations and logits */
  const add=(acc,b,k)=>{Object.keys(b).forEach(n=>{acc[n]=(acc[n]||0)+b[n]*k;}); return acc;};
  /* the blocks say what they hold (the owner, 2026-10-09: “BF16 权重只有 4.2GB”): the trained model's own bytes (w, q4, g, o, lr;
     PPO's value model adds a second set), and fz, the frozen BF16 copies of other models (a full fine-tuning's reference, the reward model) */
  const held=['policy'], all={};
  add(all,trainedBlocks(c.m,s.mode),1);
  const refOn=!(s.noRef&&s.meth==='grpo');   /* noRef: GRPO with β = 0 loads no reference model (TRL's default) */
  if((s.meth==='dpo'||s.meth==='ppo'||s.meth==='grpo')&&refOn){held.push('ref'); if(s.mode==='full') add(all,{fz:2*N},1);}
  if(s.meth==='ppo'){held.push('value'); add(all,trainedBlocks(c.m,s.mode),1);}
  if(useRM){held.push('rm'); add(all,{fz:2*N},1);}
  const per={}; Object.keys(all).forEach(k=>{per[k]=all[k]/c.gpus;});
  per.a=SEQ*m.L*(s.rc?2*m.d:pt.aL); per.l=SEQ*VOC*4;   /* rc: activations recomputed in the backward pass, 2·d bytes per token per layer kept */
  const ORDER=['w','q4','g','o','lr','fz','a','l'], blocks=ORDER.filter(k=>per[k]).map(k=>[k,per[k]]);
  const mem=blocks.reduce((a,b)=>a+b[1],0), modelBytes=Object.values(all).reduce((a,b)=>a+b,0);
  /* compute, GPU-seconds for one pass over the data, from the tokens of one example: trained (kT·N each), read by a frozen model
     (2N each), generated (at the writing rate); the reward model is trained once, on 2 answers per pair */
  const t=c.tok.p+c.tok.r, gr=genRate(m), from=s.meth==='sft'?sftFrom(c):'', tS=c.tok.p+(from==='ans'?c.tok.a:c.tok.r);
  const tk={train:0,frozen:0,gen:0};
  if(s.meth==='sft') tk.train=tS;
  else if(s.meth==='dpo'){tk.train=2*t; tk.frozen=2*t;}
  else if(s.meth==='ppo'){tk.gen=c.tok.r; tk.train=2*PPO_EP*t; tk.frozen=2*t;}
  else {tk.gen=G*c.tok.r; tk.train=G*t; tk.frozen=G*t*((refOn?1:0)+(useRM?1:0));}
  const rmTok=useRM?2*t*c.n:0;
  const train=kT*N*tk.train*cf, frozen=2*N*tk.frozen*cf, gen=tk.gen/gr, rmTrain=kT*N*rmTok*cf;
  const n=c.n, seg={train:train*n*x,frozen:frozen*n*x,gen:gen*n*x,rm:rmTrain};
  const gpus=seg.train+seg.frozen+seg.gen+seg.rm, gpuh=gpus/3600;
  const rule=c.rule[s.meth], metric=can&&rule?rule.mi-(rule.mi-c.m0)*Math.pow(2,-x/rule.h):c.m0;
  const se=Math.sqrt(metric/100*(1-metric/100)/c.nEval)*100, band=Z*se;
  const fits=mem<=CAP, inBudget=gpuh<=c.budget+1e-9, hits=metric>=c.target-1e-9;
  /* flat: the method runs, but by this lab's rule it leaves the metric where it started */
  return {card:c.id,meth:s.meth,mode:s.mode,x,G,miss,can,useRM,held,blocks,per,mem,modelBytes,fits,seg,gpuSec:gpus,gpuh,wallH:gpuh/c.gpus,inBudget,
    rule:rule||null,metric,band,hits,ok:can&&fits&&inBudget&&hits,genRate:gr,perEx:{train,frozen,gen},kT,from,flat:!!(can&&rule&&rule.mi===c.m0),
    tok:tk,rmTok,trainRate:1/(kT*N*cf)};   /* tok: tokens of one example; trainRate: trained tokens per second per GPU */
}
/* part 3: one pass of SFT and of GRPO over the math card's problems, full fine-tuning on its 16 GPUs */
const P3={sft:{card:'math',meth:'sft',mode:'full',x:1},grpo:{card:'math',meth:'grpo',mode:'full',x:1}};
const p3Ratio=()=>plan(P3.grpo).gpuh/plan(P3.sft).gpuh;
/* the same without the reference model (TRL's GRPO default β = 0) */
const p3NoRef=()=>plan({...P3.grpo,noRef:true}).gpuh/plan(P3.sft).gpuh;
/* KV cache of the answers one GPU generates at once (K and V, BF16), for a prompt + answer of the math card */
const kvGen=mk=>{const m=MOD[mk], c=CARDS.math; return GEN_B*(c.tok.p+c.tok.r)*2*m.L*m.KV*m.hd*2;};
const stepMs=mk=>2*MOD[mk].N/HBM*1e3;
/* a check of the 50% against a measurement (the owner, 2026-10-09: “训练3遍才0.73GPU小时，这是真的吗”): torchtune's README measures
   3,497 tokens per second for Llama 3.1 70B with LoRA on 8 A100s. One A100 could not do that alone (the forward pass, 2N per token,
   would need 493 TFLOP/s against its 312), so it is the total of the 8: 437 tokens per second per GPU, which at 4N per token is 39.5%
   of the A100's 312 TFLOP/s (ml-engineering, nanoGPT). Every GPU-hour figure here would be 0.5 / 0.395 = 1.27 times as large at that rate. */
const TT={tok:3497,gpus:8,peak:312e12};
const ttU=()=>TT.tok/TT.gpus*4*MOD['70b'].N/TT.peak;
/* a run's GPU hours with its computing parts at the measured share of the peak (generating answers is bound by memory, not compute) */
const ttGpuh=r=>((r.seg.train+r.seg.frozen+r.seg.rm)*U0/ttU()+r.seg.gen)/3600;

/* the challenge: the math problems with final answers only, one H100, 3 GPU hours, 39.7% → 65% */
const CH={card:'ch',start:{meth:'sft',mode:'full',x:1},tol:0.01};
let CHALL=null;
function chAll(){if(CHALL) return CHALL; const a=[]; METHS.forEach(meth=>MODES.forEach(mode=>PASSES.forEach(x=>{const r=plan({card:'ch',meth,mode,x}); a.push({meth,mode,x,r,ok:r.ok});}))); return (CHALL=a);}
const chBest=()=>chAll().filter(e=>e.ok).sort((a,b)=>a.r.gpuh-b.r.gpuh)[0];
const chMinimal=e=>!!e&&e.ok&&e.r.gpuh<=chBest().r.gpuh*(1+CH.tol);

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const GB=b=>`${fixd(b/1e9,1)} GB`;
const pc=v=>`${fixd(v,1)}%`;
const gh=v=>v>=10?fixd(v,1):(v>=0.1?fixd(v,2):fixd(v,3));
const ghU=v=>tx(`${gh(v)} GPU hours`,`${gh(v)} GPU 小时`);
const big=n=>A.fmt.big?A.fmt.big(n):fint(n);
const bigH=n=>A.ui.big?A.ui.big(n):esc(big(n));
const passW=x=>x===0.5?tx('half a pass','半遍'):tx(`${x} ${x===1?'pass':'passes'}`,`${x} 遍`);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):esc(t||id);
const MX=t=>`<div class="pcap-fx">${A.ui.math(t)}</div>`;
const M=t=>A.ui.math(t);
/* a chain in Show the math (H2): the quantity's name, then each step on its own line with the “=” signs lined up; the last step is the
   result, in words and units (bold) */
const EQ=(name,steps,res)=>`<div class="pcap-eq">`+steps.map((x,i)=>`<div class="pcap-eqn">${i?'':M(name)}</div><div class="pcap-eqr">${M((/^\+/.test(x)?'':'= ')+x)}</div>`).join('')+
  `<div class="pcap-eqn"></div><div class="pcap-eqr">${M('=')} <b>${res}</b></div></div>`;
/* a chain whose rows already start with “=” or “+” (a long sum broken at “+”), no result row: the interview answer */
const EQL=(name,rows)=>`<div class="pcap-eq">`+rows.map((x,i)=>`<div class="pcap-eqn">${i?'':M(name)}</div><div class="pcap-eqr">${M(x)}</div>`).join('')+`</div>`;
const methName=me=>({sft:'SFT',dpo:'DPO',ppo:tx('reward model + PPO','奖励模型 + PPO'),grpo:'GRPO'}[me]);
const modeName=mo=>({full:tx('full fine-tuning','全量微调'),lora:'LoRA',qlora:'QLoRA'}[mo]);
const cardModel=c=>MOD[c.m].name+(c.inst?`${NB}Instruct`:'');
const SRC={
 trlFmt:'https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/docs/source/dataset_formats.md',
 trlDpo:'https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/docs/source/dpo_trainer.md',
 trlLwr:'https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/docs/source/lora_without_regret.md',
 trlPpo:'https://github.com/huggingface/trl/blob/v1.12.0/trl/experimental/ppo/ppo_trainer.py',
 trlGrpoCfg:'https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/trl/trainer/grpo_config.py',
 trlSftCfg:'https://github.com/huggingface/trl/blob/14c8d7019319d187e3ac3cbee5906ec46d95b89b/trl/trainer/sft_config.py',
 tulu:'https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/docs/tulu3.md',
 tuluReadme:'https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/README.md',
 card:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/MODEL_CARD.md',
 simple:'https://github.com/hkust-nlp/simpleRL-reason/blob/cf1c7858bfc145a27aa733e75d930d34f0d318a6/README.md',
 gsm:'https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/README.md',
 qlora:'https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/README.md',
 ttReadme:'https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/README.md',
 qloraSh:'https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/scripts/finetune_guanaco_65b.sh',
 peftFull:'https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/method_comparison/MetaMathQA/results/full-finetuning--llama-3.2-3B-lr_0.00001.json',
 peftLora:'https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/method_comparison/MetaMathQA/results/lora--llama-3.2-3B-rank32.json',
 pDpo:'https://arxiv.org/abs/2305.18290', pGsm:'https://arxiv.org/abs/2110.14168', pStar:'https://arxiv.org/abs/2203.14465', pMath:'https://arxiv.org/abs/2402.03300', pTulu:'https://arxiv.org/abs/2411.15124', pIgpt:'https://arxiv.org/abs/2203.02155'
};

/* =====================================================================
   words, defined where they first appear (R2.8, R2.10)
   ===================================================================== */
const TIP={
 policy:()=>tx('The policy is the model being trained, the one that writes the answers.','策略模型是正在训练、负责生成回答的模型。'),
 ref:()=>tx('The reference model is a frozen copy of the policy as it was when training began. It is never updated and is used only to measure how far the policy has moved (the KL penalty).','参考模型是训练开始时策略模型的一份冻结副本，不再更新，只用来算策略模型走了多远（KL 惩罚）。'),
 rm:()=>tx('A reward model reads a prompt and an answer and gives a score. It is trained on preference pairs.','奖励模型读“提问 + 回答”，给出一个分数。它是用偏好对训练出来的。'),
 value:()=>tx('The value model (also called the critic) estimates, from each position, how much reward is still to come on average. PPO trains it alongside the policy.','价值模型（也叫 critic）估计从每个位置往后，平均还能拿到多少奖励。PPO 让它和策略模型一起训练。'),
 pair:()=>tx('A preference pair is two answers to the same prompt, of which a person chose one: the chosen and the rejected answer.','偏好对是同一个提问的两个回答，标注的人选了其中一个：选中的回答和落选的回答。'),
 checker:()=>tx('A checker is a program that checks the final answer: 1 point if it is right, 0 if it is wrong.','判分程序是检查最后答案的程序：对得 1 分，错得 0 分。'),
 G:()=>tx(`The group size G is how many answers GRPO generates for one problem. GRPO takes the group’s average reward as the baseline: an answer that scores above the average gets a positive advantage, one below it a negative advantage. The advantage of answer i is ${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r)}')}, where ${A.ui.math('r_i')} is its reward and std is the standard deviation of the group’s rewards. When all G answers get the same reward, each reward equals the average, every numerator is 0, and so every advantage is 0. TRL’s default is G = 8, and this lab uses 8.`,
   `组大小 G 是 GRPO 给一道题生成几个回答。GRPO 拿这组的平均分当基准：得分比平均分高的回答，优势为正，比平均分低的，优势为负。第 i 个回答的优势是 ${A.ui.math('A_i = \\frac{r_i - \\text{mean}(r)}{\\text{std}(r)}')}，${A.ui.math('r_i')} 是它的得分，std 是这组得分的标准差。G 个回答得分全一样时，每个得分都等于平均分，分子都是 0，所以优势全是 0。TRL 默认 G = 8，这个 Lab 也用 8。`),
 roll:()=>tx('Generating answers (a rollout): in RL training the policy writes its own answers, which are then scored and trained on. DPO uses only the answers already written in its preference pairs.','生成回答（rollout）：RL 训练里让策略模型自己写回答，再给这些回答打分、拿来训练。DPO 只用偏好对里已经写好的回答。'),
 noref:()=>tx(`TRL’s GRPO uses β = 0 by default (since v0.19.0) and then loads no reference model, so this part of the bar goes. GRPO would then cost about ${fixd(p3NoRef(),0)} times SFT, still more than 10 times. verl, DeepSeek‑R1 and Tülu 3 keep a small KL term, and this lab keeps the reference model.`,
   `TRL 的 GRPO 从 v0.19.0 起默认 β = 0，这时不加载参考模型，条里的这一段也就没有了。那样 GRPO 约是 SFT 的 ${fixd(p3NoRef(),0)} 倍，仍在 10 倍以上。verl、DeepSeek‑R1 和 Tülu 3 都保留一个小的 KL 项，这个 Lab 也保留参考模型。`),
 pass:()=>tx('One pass means every example of the card’s data is used once. Half a pass uses half of them.','训练一遍，就是把任务卡上的数据每条用一次。半遍只用其中一半。'),
 gpuh:()=>tx('GPU hours are the number of GPUs times the hours they run. 8 GPUs running for 1 hour use 8 GPU hours, and so does 1 GPU running for 8 hours. Every budget and cost in this lab counts all GPUs together, as papers and cloud bills do.',
   'GPU 小时是用的 GPU 数乘以小时数。8 张卡一起跑 1 小时是 8 GPU 小时，1 张卡跑 8 小时也是 8 GPU 小时。这个 Lab 的预算和花费都是所有卡加起来的总数，论文和云厂商的账单也是这样算的。'),
 gen:()=>tx(`Generating answers (a rollout) is the policy writing its own answers. This lab assumes each GPU generates ${GEN_B} answers at once. Each step reads all the BF16 weights from GPU memory once and adds one token to each answer, so for the 8B each GPU generates about ${fint(Math.round(genRate(MOD['8b'])/10)*10)} tokens per second.`,
            `生成回答（rollout）就是策略模型自己写回答。这个 Lab 假设每张卡同时生成 ${GEN_B} 个回答。每一步把全部 BF16 权重从显存读一遍，给每个回答添一个 token，所以 8B 每张卡每秒约生成 ${fint(Math.round(genRate(MOD['8b'])/10)*10)} 个 token。`),
 band:()=>tx('An error bar is the 95% interval drawn around a score. In this lab the score is the true level given by the lab’s rule, so a new batch of examples of the same size would give a score inside the bar 95% of the time. Its formula is in “Show the math”.','误差线画的是一个分数两边的 95% 区间。这个 Lab 里的分数就是按规则算出的真实水平，所以换一批同样多的样本再测，新分数有 95% 的机会落在线内。算式在“展开公式”里。'),
 bytes18:()=>tx('Full fine-tuning keeps 18 bytes for every parameter: the BF16 weight (2), its FP32 gradient (4) and the optimizer state (12, for the FP32 weight copy and Adam’s m and v).','全量微调每个参数要存 18 字节：BF16 权重 2 字节，FP32 梯度 4 字节，优化器状态 12 字节（FP32 权重副本、Adam 的 m 和 v）。')
};
const DATAN={demo:{en:'demonstrations',zh:'示范回答'},ans:{en:'final answers',zh:'最后答案'},pairs:{en:'preference pairs',zh:'偏好对'},checker:{en:'a checker',zh:'判分程序'}};
const dataName=k=>L(DATAN[k]);
/* what one training example is, for a method the card's data allows (strip, metric card, the challenge's data row) */
function exOf(c,meth,r){
  if(meth==='sft'){const f=sftFrom(c);
    return f==='ans'?tx('problem → final answer','题目 → 最后答案'):f==='chosen'?tx('prompt → chosen answer','提问 → 选中的回答'):c.metric==='acc'?tx('problem → worked solution','题目 → 解题过程'):tx('prompt → example reply','提问 → 示范回答');}
  if(meth==='dpo') return tx('trains on the pairs','直接用偏好对训练');
  return r&&r.useRM?tx('reward model trained first','先训练奖励模型'):tx('the checker gives the reward','判分程序给奖励');}
const canTip=(c,meth)=>{const f=meth==='sft'?sftFrom(c):'';
  if(f==='ans') return tx('Each problem with its final answer makes one SFT example. SFT learns to write what its answers contain, so here it learns to give the answer without the working.','每道题配上它的最后答案，就是一条 SFT 的训练样本。回答里写了什么，SFT 就学着写什么，所以它在这里学到的是不写过程、直接报答案。');
  if(f==='chosen') return tx('Each pair makes one SFT example, the prompt with the chosen answer. The rejected answer is dropped, so SFT never sees what users turned down.','每个偏好对都能做成一条 SFT 的训练样本，就是提问配上选中的回答。落选的回答被丢掉，所以 SFT 看不到用户没选的是哪个。');
  return tx('This data can make this method’s training examples.','这套数据能做成这个方法要的训练样本。');};
/* the method runs, but by this lab's rule the metric stays where it started (SFT on final answers alone, or on the chosen answers alone) */
const flatWhy=c=>c.metric==='acc'?tx(`By this lab’s rule, SFT on final answers alone leaves the accuracy at ${pc(c.m0)}: it learns to give the answer without the working. Worked solutions in the data would teach it how to solve the problems.`,`按这个 Lab 的规则，只学最后答案的 SFT 不提高答对率，停在 ${pc(c.m0)}。它学到的是不写过程、直接报答案。数据里有一步步的解题过程，SFT 才学得会怎么解题。`)
  :tx(`By this lab’s rule, SFT on the chosen answers alone leaves the win rate at ${pc(c.m0)}: it never sees the answer users turned down. DPO learns from both answers of each pair.`,`按这个 Lab 的规则，只学选中回答的 SFT 不提高胜率，停在 ${pc(c.m0)}。它看不到用户没选的那个回答，DPO 却从每对的两个回答里学。`);
/* the four methods, defined before any option uses them (R2.10) */
const METH_DEF=()=>tx(
 `<ul class="pcap-defs">`+
 `<li><b>${labLink('sft','SFT')}</b>: pair each prompt with a written answer and train the model to write that answer. The model learns to write what those answers contain. <i>Needs:</i> prompts with answers, such as demonstrations.</li>`+
 `<li><b>${labLink('dpo','DPO')}</b>: train on ${DEF('preference pairs',TIP.pair())} directly, raising the chosen answer against the rejected one while a ${DEF('reference model',TIP.ref())} holds the model back. While training, the model does not ${DEF('generate answers of its own',TIP.roll())}. <i>Needs:</i> preference pairs.</li>`+
 `<li><b>${labLink('reward-model','Reward model')} + ${labLink('ppo','PPO')}</b>: first train a ${DEF('reward model',TIP.rm())} on the pairs. Then the ${DEF('policy',TIP.policy())} generates answers itself and is pushed toward what the reward model scores high, with a ${DEF('value model',TIP.value())} and the reference model alongside. <i>Needs:</i> preference pairs.</li>`+
 `<li><b>${labLink('grpo','GRPO')}</b>: the policy generates a group of ${DEF('G answers',TIP.G())} to each prompt and takes the group’s average score as the baseline, so GRPO needs no value model. <i>Needs:</i> a ${DEF('checker',TIP.checker())} (see ${labLink('rlvr','RLVR')}); without one, a reward model trained on preference pairs first.</li></ul>`,
 `<ul class="pcap-defs">`+
 `<li><b>${labLink('sft','SFT')}</b>：给每个提问配一个写好的回答，训练模型照着写。回答里写了什么，模型就学着写什么。<i>需要：</i>提问和配好的回答，比如示范回答。</li>`+
 `<li><b>${labLink('dpo','DPO')}</b>：直接用${DEF('偏好对',TIP.pair())}训练，让选中的回答比落选的回答更可能出现，同时用${DEF('参考模型',TIP.ref())}拉住它不跑远。训练时，模型不用${DEF('自己生成回答',TIP.roll())}。<i>需要：</i>偏好对。</li>`+
 `<li><b>${labLink('reward-model','奖励模型')} + ${labLink('ppo','PPO')}</b>：先用偏好对训练一个${DEF('奖励模型',TIP.rm())}，再让${DEF('策略模型',TIP.policy())}自己生成回答，朝着奖励模型打高分的方向改。旁边还要有${DEF('价值模型',TIP.value())}和参考模型。<i>需要：</i>偏好对。</li>`+
 `<li><b>${labLink('grpo','GRPO')}</b>：策略模型给每个提问生成${DEF('一组 G 个回答',TIP.G())}，以这组的平均分为基准，所以不需要价值模型。<i>需要：</i>${DEF('判分程序',TIP.checker())}（见 ${labLink('rlvr','RLVR')}）；没有的话，要先用偏好对训练一个奖励模型来打分。</li></ul>`);
const MODE_DEF=()=>tx(
 `<ul class="pcap-defs">`+
 `<li><b>Full fine-tuning</b> trains every parameter and keeps ${DEF('18 bytes for each',TIP.bytes18())}.</li>`+
 `<li><b>${labLink('lora','LoRA')}</b> freezes the original weights (BF16, 2 bytes each). Beside each weight matrix it adds two narrow matrices and trains only them; their width is called the rank r. With LoRA of rank r = 16 on all 7 matrices of each layer (4 in attention, 3 in the MLP), the 8B trains only ${bigH(PT['8b'].lp)} parameters.</li>`+
 `<li><b>QLoRA</b> also stores the frozen weights in 4 bits, about half a byte each. The embedding table and the LM head stay in BF16.</li></ul>`,
 `<ul class="pcap-defs">`+
 `<li><b>全量微调</b>：每个参数都训练，每个参数要存 ${DEF('18 字节',TIP.bytes18())}。</li>`+
 `<li><b>${labLink('lora','LoRA')}</b>：原来的权重冻住不动（BF16，每个 2 字节）。每个权重矩阵旁边加两个窄矩阵，只训练它们，窄矩阵的宽度叫秩 r。每层 7 个矩阵（注意力的 4 个、MLP 的 3 个）都加上秩 r = 16 的 LoRA 时，8B 一共只训练 ${bigH(PT['8b'].lp)} 个参数。</li>`+
 `<li><b>QLoRA</b>：再把冻住的权重存成 4 位，每个约半个字节。embedding 表和 LM head 仍是 BF16。</li></ul>`);

/* =====================================================================
   task cards' words
   ===================================================================== */
const CARDW={
 fmt:{s:{en:'Support format',zh:'客服格式'},t:{en:'Support replies in the house format',zh:'客服回答要符合公司格式'},
   goal:{en:'Make the support assistant answer in the company’s format.',zh:'让客服助手的回答符合公司的格式规范。'},
   data:{en:'3,000 example conversations, written in house; no preference data',zh:'3,000 段自己写的示范对话，没有偏好数据'},
   met:{en:'Format pass rate (a person checks 200 sampled answers)',zh:'格式合格率（人工随机抽检 200 个回答）'},
   k:{en:'Support format',zh:'客服格式'},
   story:{en:'The support team wants the assistant’s replies to follow the company’s format. It has 3,000 example conversations written in house, and no preference data. At acceptance a person samples 200 replies and checks the share in the right format: it must reach at least 95%.',
     zh:'客服团队想让助手的回答符合公司的格式规范。他们手里有 3,000 段自己写的示范对话，没有偏好数据。验收时由人随机抽 200 个回答，看格式合格的比例，至少要到 95%。'}},
 pref:{s:{en:'User preferences',zh:'用户偏好'},t:{en:'Answers users prefer',zh:'回答更合用户口味'},
   goal:{en:'Make users prefer the new version’s answers.',zh:'让用户更喜欢新版本的回答。'},
   data:{en:'20,000 preference pairs (two answers to one prompt, of which a user chose one)',zh:'20,000 对偏好（同一个提问的两个回答，用户选了其中一个）'},
   met:{en:'Win rate against the old version (500 prompts, an LLM as the judge)',zh:'对旧版本的胜率（500 个提问，LLM 当裁判）'},
   k:{en:'User preferences',zh:'用户偏好'},
   story:{en:'The product team wants users to prefer the new version’s answers. It has 20,000 preference pairs: two answers to one prompt, of which a user chose one. At acceptance the old and the new version each answer 500 prompts, an LLM judges, and the new version must win at least 60%.',
     zh:'产品团队想让用户更喜欢新版本的回答。他们手里有 20,000 对偏好，每对是同一个提问的两个回答，用户选了其中一个。验收时拿 500 个提问让新旧版本各答一次，由 LLM 当裁判，新版本至少要赢 60%。'}},
 math:{s:{en:'Math accuracy',zh:'数学答对率'},t:{en:'Math accuracy',zh:'数学题的答对率'},k:{en:'Math accuracy',zh:'数学答对率'},
   story:{en:'The math team wants the model to solve more grade-school math word problems. It has the 7,473 GSM8K training problems, each with a worked solution and the final answer, and a checker. At acceptance the model takes the 1,319 GSM8K test problems and must get at least 70% right.',
     zh:'数学团队想让模型答对更多小学数学应用题。他们手里有 GSM8K 训练集的 7,473 道题，每道都有解题过程和最后的答案，还有一个判分程序。验收看 GSM8K 测试集 1,319 道题，至少要答对 70%。'},
   goal:{en:'Make the model solve more grade-school math problems.',zh:'让模型答对更多小学数学应用题。'},
   data:{en:'7,473 GSM8K training problems, each with a worked solution and the final answer; a checker for the answer',zh:'GSM8K 训练集的 7,473 道题，每道都有解题过程和最后的答案，还有判分程序'},
   met:{en:'Accuracy on the 1,319 GSM8K test problems',zh:'GSM8K 测试集 1,319 道题的答对率'}},
 big:{s:{en:'70B, one GPU',zh:'70B，一张卡'},t:{en:'The house format on 70B, one GPU',zh:'70B 学公司格式，只有一张卡'},k:{en:'70B, house format',zh:'70B 学公司格式'},
   goal:{en:'The same support format, but on the larger 70B model.',zh:'同样的客服格式，但要用更大的 70B 模型。'},
   data:{en:'The same 3,000 example conversations',zh:'同样的 3,000 段示范对话'},
   met:{en:'Format pass rate (a person checks 200 sampled answers)',zh:'格式合格率（人工随机抽检 200 个回答）'}},
 ch:{t:{en:'Math, final answers only, one GPU',zh:'只拿到最后答案的数学题，一张卡'},k:{en:'Math, answers only',zh:'只有答案的数学题'},
   goal:{en:'Raise the math accuracy on one GPU.',zh:'在一张卡上提高数学题的答对率。'},
   data:{en:'The 7,473 GSM8K training problems. The team got only the questions and the final answers, not the worked solutions. It also has a checker.',zh:'GSM8K 训练集的 7,473 道题。团队只拿到题目和最后的答案，没拿到解题过程。另有一个判分程序。'},
   met:{en:'Accuracy on the 1,319 GSM8K test problems',zh:'GSM8K 测试集 1,319 道题的答对率'}}
};
const metricName=c=>({fmt:tx('format pass rate','格式合格率'),win:tx('win rate','胜率'),acc:tx('accuracy','答对率')}[c.metric]);
const gpuW=c=>c.gpus===1?tx('one H100','1 张 H100'):tx(`${c.gpus} H100s`,`${c.gpus} 张 H100`);

/* =====================================================================
   module state
   ===================================================================== */
let S={card:'fmt',meth:'sft',mode:'lora',x:3}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let view1='fmt', tried2=new Set(), lv4='8full', tried4=new Set(['8full']), lv5=1, tried5=new Set([1]), lv6='dpo', tried6=new Set(['dpo']);
let ch={...CH.start}, saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=1, drillOpen=false;
const shownPrev={};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'lt1',t:{en:'Less than SFT',zh:'比 SFT 还少'}},{v:'1to3',t:{en:'1 to 3 times SFT',zh:'是 SFT 的 1 到 3 倍'}},{v:'3to10',t:{en:'3 to 10 times SFT',zh:'是 SFT 的 3 到 10 倍'}},{v:'gt10',t:{en:'More than 10 times SFT',zh:'是 SFT 的 10 倍以上'}}];
const PRED_RIGHT='gt10';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const predHidden=()=>isP('predict')&&!pred;
const LV1=CARD_IDS.slice(0,3).map(k=>({v:k,t:CARDW[k].t,d:CARDW[k].data}));
const LV4=[{v:'8full',t:{en:'8B, full fine-tuning (initial setup)',zh:'8B，全量微调（初始设置）'},d:{en:'The support task on one H100',zh:'客服任务，一张 H100'},s:{card:'fmt',meth:'sft',mode:'full',x:3}},
  {v:'8lora',t:{en:'8B, LoRA',zh:'8B，LoRA'},d:{en:'Only the method changes',zh:'只换方法'},s:{card:'fmt',meth:'sft',mode:'lora',x:3}},
  {v:'70lora',t:{en:'70B, LoRA',zh:'70B，LoRA'},d:{en:'The same task on the 70B model, still one H100',zh:'同样的任务换成 70B，还是一张 H100'},s:{card:'big',meth:'sft',mode:'lora',x:3}},
  {v:'70qlora',t:{en:'70B, QLoRA',zh:'70B，QLoRA'},d:{en:'The frozen weights in 4 bits',zh:'冻住的权重存成 4 位'},s:{card:'big',meth:'sft',mode:'qlora',x:3}}];
const LV5=PASSES.map(x=>({v:x,t:{en:x===1?'1 pass (initial setup)':(x===0.5?'Half a pass':`${x} passes`),zh:x===1?'1 遍（初始设置）':(x===0.5?'半遍':`${x} 遍`)},d:{en:x===0.5?'Half of the 3,000 conversations, once':`Every conversation ${x===1?'once':x+' times'}`,zh:x===0.5?'3,000 段对话里的一半，各用一次':`每段对话用 ${x} 次`}}));
const LV6=[{v:'dpo',t:{en:'DPO (initial setup)',zh:'DPO（初始设置）'},d:{en:'Trains on the pairs directly',zh:'直接用偏好对训练'},s:{card:'pref',meth:'dpo',mode:'full',x:1}},
  {v:'ppo',t:{en:'Reward model + PPO',zh:'奖励模型 + PPO'},d:{en:'Train a reward model on the pairs, then PPO against it',zh:'先用偏好对训练奖励模型，再用 PPO 往高分拉'},s:{card:'pref',meth:'ppo',mode:'full',x:1}},
  {v:'grpo',t:{en:'GRPO with a reward model',zh:'GRPO，用奖励模型打分'},d:{en:'No checker here, so a reward model scores the 8 answers',zh:'这里没有判分程序，只能让奖励模型给 8 个回答打分'},s:{card:'pref',meth:'grpo',mode:'full',x:1}}];
const lvOf=(arr,v)=>arr.find(o=>o.v===v);
const lv4Res=v=>{const r=plan(lvOf(LV4,v).s); return r.fits?tx('fits','放得下'):tx('does not fit','放不下');};
const lv5Res=v=>{const r=plan({card:'fmt',meth:'sft',mode:'lora',x:v}); return pc(r.metric);};
const lv6Res=v=>ghU(plan(lvOf(LV6,v).s).gpuh);
const levers=(id,aria,val,opts,tried,res)=>`<div class="pcap-lv" id="${id}">`+A.ui.levers({aria,value:String(val),options:opts.map(o=>({v:String(o.v),title:L(o.t),desc:L(o.d),result:esc(res&&(tried.has(o.v)||o.v===val)?res(o.v):'')}))})+`</div>`;
/* the lead-in above a group of options says what to pick and what to watch, in this lab's words (the owner, 2026-10-08, rule G1:
   it replaces the old note on single choice; the options stay single choice and keep their radio dots) */
const lead=(en,zh)=>`<p class="pcap-lead">${tx(en,zh)}</p>`;
const LEAD={lv1:()=>lead('Pick a task and see what data its team has and what target it has to reach.','选一个任务，看看这个团队手里有什么数据、要达到什么目标。'),
 lv4:()=>lead('Pick a model and a way to train it, and see how much memory it needs and whether one H100 holds it.','选一种模型和训练方式，看看它要多少显存，一张 H100 放不放得下。'),
 lv5:()=>lead('Pick a number of passes and see whether the format pass rate clears 95% with a margin.','选一个训练遍数，看看格式合格率能不能稳稳过 95%。'),
 lv6:()=>lead('Pick a method and see how many GPU hours one pass takes, and whether it stays within the budget.','选一种方法，看看训练一遍要多少 GPU 小时，超不超预算。')};
const methSeg=(id,key,val,inline)=>A.ui.seg({id,key,label:tx('Method','方法'),options:METHS.map(m=>[m,methName(m)]),value:val,inline});
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('Three teams, three kinds of data: which method for which?','三个团队，三种数据：各用什么方法？'),
  enter:()=>({card:view1,meth:'sft',mode:'lora',x:3}),
  body:()=>tx(`<p>Three teams ask you to post-train their model. Their data differ: one has example conversations, one has users’ choices between two answers, one has math problems a program can check.</p>`+
    `<p>The earlier labs covered the methods ${labLink('sft','SFT')}, ${labLink('dpo','DPO')}, ${labLink('reward-model','reward model')} + ${labLink('ppo','PPO')} and ${labLink('grpo','GRPO')}, and ${labLink('lora','LoRA')}, which saves memory. This lab puts them together, and you choose.</p>`+LEAD.lv1()+
    levers('pcap-ic-lv1',tx('Task to look at','看哪个任务'),view1,LV1,new Set(),null),
    `<p>三个团队请你给他们的模型做后训练。他们手里的数据各不一样：一个有写好的示范对话，一个有用户在两个回答里选过的记录，一个有能用程序判对错的数学题。</p>`+
    `<p>前面几个 Lab 讲了 ${labLink('sft','SFT')}、${labLink('dpo','DPO')}、${labLink('reward-model','奖励模型')} + ${labLink('ppo','PPO')} 和 ${labLink('grpo','GRPO')} 这几种方法，还有省显存的 ${labLink('lora','LoRA')}。这个 Lab 把它们放在一起，由你来选。</p>`+LEAD.lv1()+
    levers('pcap-ic-lv1',tx('Task to look at','看哪个任务'),view1,LV1,new Set(),null)),
  /* polish r4: the picked task's goal, metric, data, model, GPUs and budget are the task card on the stage (above the card on a phone),
     so the card no longer retells them (CARDW[...].story stays for reference); the GPU-hours definition stays (the owner, 2026-10-09, item 1) */
  obs:()=>tx('<p>The task card in the chart lists the team’s goal, data and metric, the model, the number of GPUs and the budget. The later parts use them.</p><p>The budget is in GPU hours, which add up the time of every GPU:</p>','<p>图里的任务卡写着这个团队的目标、数据、指标，用什么模型、几张卡、多少预算，后面都会用到。</p><p>预算用 GPU 小时来算，它把每张卡用的时间加在一起：</p>')+
    MX(tx('\\text{GPU hours} = \\text{GPUs} \\times \\text{hours}','\\text{GPU 小时} = \\text{GPU 数} \\times \\text{小时数}'))+
    tx('<p>8 GPUs running for 1 hour use 8 GPU hours. Every budget and cost in this lab counts all GPUs together, as papers do.</p>','<p>8 张卡一起跑 1 小时，就是 8 GPU 小时。这个 Lab 里的预算和花费，都是所有卡加起来的总数，论文里也是这样算的。</p>')},
 {id:'data',title:{en:'Start from the data',zh:'先看手里的数据'},
  lead:()=>tx('Each method needs its own kind of data.','每个方法要的数据不一样。'),
  enter:()=>({...S,card:S.card==='big'?'fmt':S.card,mode:'full',x:1}),
  expect:()=>tried2.size>=3,
  body:()=>tx(`<p>Each method learns from a different kind of data. Before cost or quality, ask: what does the team actually have?</p>`+METH_DEF()+
    `<p>Pick a task and a method. The chart says whether the team has the data it needs.</p>`,
    `<p>每个方法从不同的数据里学东西。先别管花多少钱、效果多好，先问：团队手里到底有什么数据？</p>`+METH_DEF()+
    `<p>选一个任务、一个方法，图里会告诉你数据够不够。</p>`)+
    A.ui.seg({id:'pcap-ic-card',key:'card',label:tx('Task','任务'),options:CARD_IDS.slice(0,3).map(k=>[k,L(CARDW[k].t)]),value:S.card,inline:true})+methSeg('pcap-ic-meth','meth',S.meth,true),
  obs:()=>{const r=plan(S), c=CARDS[S.card];
    if(!tried2.size) return '';
    const head=r.can?tx(`<p class="okline"><b>${methName(S.meth)} can run on “${esc(L(CARDW[S.card].t))}”.</b></p>`,`<p class="okline"><b>“${esc(L(CARDW[S.card].t))}”可以用${methName(S.meth)}。</b></p>`)
      :tx(`<p class="badline"><b>${methName(S.meth)} cannot run on “${esc(L(CARDW[S.card].t))}”: it needs ${dataName(r.miss)}, and the team has none.</b></p>`,`<p class="badline"><b>“${esc(L(CARDW[S.card].t))}”用不了${methName(S.meth)}。${methName(S.meth)} 需要${dataName(r.miss)}，团队手里没有。</b></p>`);
    const more=S.card==='pref'&&S.meth==='sft'?tx('<p>Each pair gives one training example, the prompt with the chosen answer, so SFT can run. But it never sees the answer that lost, so it cannot learn why users turned it down. Whether a method can run and whether it reaches the target are two different questions.</p>','<p>每个偏好对都能做成一条训练样本，就是提问配上选中的回答，所以 SFT 能跑。可它看不到落选的回答，学不到用户为什么没选它。能不能跑是一回事，能不能达标是另一回事。</p>')
      :S.card==='pref'&&S.meth==='grpo'?tx('<p>It runs, but only with a reward model trained on the pairs first, because there is no checker. Part 6 comes back to this.</p>','<p>能跑，但因为没有判分程序，要先用偏好对训练一个奖励模型来打分。第 6 部分会回到这件事。</p>')
      :S.card==='math'&&S.meth==='sft'?tx('<p>It runs on the worked solutions. GRPO can run too, with the checker. Part 3 compares what the two cost.</p>','<p>可以用解题过程做 SFT。有判分程序，GRPO 也能跑。第 3 部分比一比两者各花多少。</p>'):'';
    return head+more+(tried2.size<3?tx(`<p class="small muted">Try ${3-tried2.size} more task and method ${3-tried2.size===1?'pair':'pairs'}.</p>`,`<p class="small muted">再试 ${3-tried2.size} 种任务和方法的组合。</p>`):'');}},
 {id:'predict',title:{en:'Predict: how much more does GRPO cost?',zh:'预测：GRPO 比 SFT 贵多少？'},gate:()=>!!pred,skip:true,
  lead:()=>tx('7,473 math problems: GRPO or SFT, which costs more?','7,473 道数学题：GRPO 和 SFT 谁更贵？'),
  enter:()=>({...P3.grpo}),
  body:()=>{const ch2=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>The math task can use both. Each problem is about 128 tokens and its worked solution about 256.</p>`+
     `<ul class="pcap-defs"><li><b>SFT</b> trains once on each problem with its solution.</li><li><b>GRPO</b> lets the model generate ${DEF('8 answers',TIP.G())} to each problem, about 256 tokens each. Each answer is trained on together with the problem, and the reference model reads each of them once more.</li>`+
     `<li>The reference model only reads and does not train, so reading a token costs about a third of training it.</li>`+
     `<li>When generating, each step reads all the weights from GPU memory once to add one token to each answer. This lab assumes each GPU generates ${GEN_B} answers at once, so the ${GEN_B} new tokens share that one read, and generating a token takes about as long as training one.</li></ul>`+
     `<p>Both go through all 7,473 problems once, with full fine-tuning on 16 H100s. How many ${DEF('GPU hours',TIP.gpuh())} does GRPO take, compared with SFT?</p>`+ch2+
     (pred?`<p class="small muted">${tx('Saved. You can still change it before you move on.','已记下，进入下一部分之前还可以改。')}</p>`:''),
     `<p>数学任务两种方法都能用。每道题约 128 个 token，解题过程约 256 个 token。</p>`+
     `<ul class="pcap-defs"><li><b>SFT</b>：每道题连同解题过程训练一遍。</li><li><b>GRPO</b>：每道题让模型生成 ${DEF('8 个回答',TIP.G())}，每个约 256 个 token。每个回答都连同题目一起训练，参考模型还要把它们各读一遍。</li>`+
     `<li>参考模型只读、不训练，读一个 token 的花费约是训练它的三分之一。</li>`+
     `<li>生成回答时，每一步要把全部权重从显存读一遍，才能给每个回答添一个 token。这个 Lab 假设每张卡同时生成 ${GEN_B} 个回答，读一遍权重的时间由这 ${GEN_B} 个新 token 分摊，所以生成一个 token 的时间和训练一个 token 差不多。</li></ul>`+
     `<p>两种方法都把 7,473 道题过一遍，都用 16 张 H100 做全量微调。GRPO 花的 ${DEF('GPU 小时',TIP.gpuh())}大约是 SFT 的多少倍？</p>`+ch2+
     (pred?`<p class="small muted">${tx('Saved. You can still change it before you move on.','已记下，进入下一部分之前还可以改。')}</p>`:''));},
  obs:()=>{if(!pred) return ''; const rs=plan(P3.sft), rg=plan(P3.grpo), k=p3Ratio(), right=pred===PRED_RIGHT;
    const v=right?tx(`<p class="okline"><b>Right: GRPO takes about ${fixd(k,0)} times the GPU hours of SFT.</b></p>`,`<p class="okline"><b>答对了：GRPO 花的 GPU 小时约是 SFT 的 ${fixd(k,0)} 倍。</b></p>`)
      :tx(`<p class="badline"><b>You predicted “${esc(predT(pred))}”. The answer is “${esc(predT(PRED_RIGHT))}”: about ${fixd(k,0)} times.</b></p>`,`<p class="badline"><b>你预测的是“${esc(predT(pred))}”，正确答案是“${esc(predT(PRED_RIGHT))}”，实际约 ${fixd(k,0)} 倍。</b></p>`);
    return v+tx(`<p>Training on 8 answers instead of one solution already makes it 8 times. The ${DEF('reference model reading the answers',TIP.noref())} takes time, and so does generating them token by token. These two are the last two parts of the GRPO bar.</p>`,
      `<p>训练 8 个回答而不是一份解题过程，光这一项就是 8 倍。${DEF('参考模型读这些回答',TIP.noref())}要花时间，模型一个 token 一个 token 地生成它们也要花时间。这两项就是图里 GRPO 那根条的后两段。</p>`);}},
 {id:'memory',title:{en:'Memory decides full or LoRA',zh:'显存决定全量还是 LoRA'},
  lead:()=>tx('One H100: what fits?','只有一张 H100，什么放得下？'),
  enter:()=>({...lvOf(LV4,lv4).s}),
  expect:()=>tried4.has('70qlora'),
  body:()=>tx(`<p>The support task runs SFT on one H100 with 80 GB. Before training, decide how much of the model to train:</p>`+MODE_DEF()+LEAD.lv4()+
    levers('pcap-ic-lv4',tx('Model and method','模型和方法'),lv4,LV4,tried4,lv4Res),
    `<p>客服任务要在一张 80 GB 的 H100 上做 SFT。开训之前，先决定训练模型的哪些部分：</p>`+MODE_DEF()+LEAD.lv4()+
    levers('pcap-ic-lv4',tx('Model and method','模型和方法'),lv4,LV4,tried4,lv4Res)),
  obs:()=>{const r=plan(S), base=plan(LV4[0].s), c=CARDS[S.card];
    const head=r.fits?tx(`<p class="okline"><b>${MOD[c.m].name} with ${modeName(S.mode)} needs ${GB(r.mem)} on the H100: it fits.</b></p>`,`<p class="okline"><b>${MOD[c.m].name} 用${modeName(S.mode)}要 ${GB(r.mem)}，一张 H100 放得下。</b></p>`)
      :tx(`<p class="badline"><b>${MOD[c.m].name} with ${modeName(S.mode)} needs ${GB(r.mem)}, more than the H100’s 80 GB.</b></p>`,`<p class="badline"><b>${MOD[c.m].name} 用${modeName(S.mode)}要 ${GB(r.mem)}，超过了 H100 的 80 GB。</b></p>`);
    const why={'8full':tx('<p>Full fine-tuning keeps 18 bytes for every parameter, so the model states alone are far beyond 80 GB.</p>','<p>全量微调每个参数要 18 字节，光模型状态就远远超过 80 GB。</p>'),
      '8lora':tx(`<p>The initial setup needed ${GB(base.mem)}. LoRA keeps the frozen weights at 2 bytes each and gives gradients and Adam’s state only to the narrow matrices, so it fits easily.</p>`,`<p>初始设置要 ${GB(base.mem)}。LoRA 冻住的权重每个只占 2 字节，梯度和 Adam 的状态只给那两个窄矩阵，所以轻松放下。</p>`),
      '70lora':tx('<p>The frozen BF16 weights of 70B alone are larger than the GPU. LoRA does not shrink them.</p>','<p>70B 光是冻住的 BF16 权重就比这张卡大，LoRA 并不能让它们变小。</p>'),
      '70qlora':(q=>tx(`<p>Stored in 4 bits, the frozen linear layers take about a quarter of the room, and the 70B fits on one H100. They hold most of the model’s weights. Only the embedding table and the LM head stay in BF16, which is why that block is small.</p><p>The GPU hours are few because the data is small: ${passW(q.x)} over ${fint(CARDS.big.n)} conversations train only ${big(q.tok.train*CARDS.big.n*q.x)} tokens. The “GPU hours and budget” block of the chart shows how they add up.</p>`,
        `<p>冻住的线性层存成 4 位，只占约四分之一的空间，70B 也能放进一张 H100。模型的大部分权重都在这些线性层里。只有 embedding 表和 LM head 还是 BF16，所以 BF16 那一块很小。</p><p>GPU 小时这么少，是因为数据少：${fint(CARDS.big.n)} 段对话训练${passW(q.x)}，一共只有 ${big(q.tok.train*CARDS.big.n*q.x)} 个 token。图里“GPU 小时和预算”那一栏写了是怎么算出来的。</p>`))(plan({card:'big',meth:'sft',mode:'qlora',x:3}))}[lv4]||'';
    return head+why;}},
 {id:'metric',title:{en:'The metric decides when to stop',zh:'指标决定什么时候停'},
  lead:()=>tx('Train until the format check passes 95%.','训到格式抽检过 95%。'),
  enter:()=>({card:'fmt',meth:'sft',mode:'lora',x:lv5}),
  expect:()=>plan({card:'fmt',meth:'sft',mode:'lora',x:lv5}).hits,
  body:()=>{const r=plan({card:'fmt',meth:'sft',mode:'lora',x:3});
   return tx(`<p>The support task is done when at least 95% of 200 sampled answers pass the format check. How many ${DEF('passes',TIP.pass())} over the 3,000 conversations does SFT need?</p>`+
    `<p>In this lab the pass rate climbs toward a ceiling, and each half pass closes half of the remaining gap (this lab’s rule):</p>`+MX('m(x) = m_\\infty - (m_\\infty - m_0) \\times 2^{-x/h}')+
    `<p class="small muted">${M('m_0')} = 40% is the rate before training, ${M('m_\\infty')} = 98% the ceiling, h = half a pass, and x the passes.</p>`+LEAD.lv5()+
    levers('pcap-ic-lv5',tx('Passes over the data','训练几遍'),lv5,LV5,tried5,lv5Res),
    `<p>客服任务的验收标准：随机抽 200 个回答，至少 95% 格式合格。SFT 要把 3,000 段对话${DEF('训练几遍',TIP.pass())}才够？</p>`+
    `<p>在这个 Lab 里，合格率朝一个上限爬，每多训半遍，就补上剩下差距的一半（这是本 Lab 的规则）：</p>`+MX('m(x) = m_\\infty - (m_\\infty - m_0) \\times 2^{-x/h}')+
    `<p class="small muted">训练前的合格率 ${M('m_0')} = 40%，上限 ${M('m_\\infty')} = 98%，h = 半遍，x 是训练的遍数。</p>`+LEAD.lv5()+
    levers('pcap-ic-lv5',tx('Passes over the data','训练几遍'),lv5,LV5,tried5,lv5Res));},
  obs:()=>{const r=plan({card:'fmt',meth:'sft',mode:'lora',x:lv5}), c5=CARDS.fmt;
    const head=r.hits?tx(`<p class="okline"><b>${passW(lv5)}: ${pc(r.metric)} pass the check, above the 95% target.</b></p>`,`<p class="okline"><b>训练${passW(lv5)}，合格率 ${pc(r.metric)}，过了 95% 的目标。</b></p>`)
      :tx(`<p class="badline"><b>${passW(lv5)}: ${pc(r.metric)} pass the check, below the 95% target.</b></p>`,`<p class="badline"><b>训练${passW(lv5)}，合格率 ${pc(r.metric)}，没到 95% 的目标。</b></p>`);
    const lo=r.metric-r.band, clear=lo>=c5.target;
    const noise=tx(`<p>The check samples only 200 answers, so the score itself is uncertain: its ${DEF('error bar',TIP.band())} is about ±${fixd(r.band,1)} points (see ${labLink('eval-ci')}). A score just under or over 95% does not settle it; leave a margin.</p>`,
      `<p>抽检只看 200 个回答，量出来的分数本身就有误差。${DEF('误差线',TIP.band())}约 ±${fixd(r.band,1)} 个百分点（见${labLink('eval-ci')}）。刚好比 95% 低一点或高一点，都说明不了什么，要留出余量。</p>`)+
      (r.hits?(clear?tx(`<p>Here even the bottom of the error bar, ${pc(lo)}, is above 95%.</p>`,`<p>这里连误差线的下端 ${pc(lo)} 都在 95% 以上。</p>`):tx(`<p>But the bottom of the error bar, ${pc(lo)}, is still below 95%: another sample of 200 could fail the check.</p>`,`<p>可误差线的下端 ${pc(lo)} 还在 95% 以下，换 200 个回答再抽检，可能就不合格。</p>`)):'');
    return head+noise;}},
 {id:'wrong',title:{en:'When the obvious choice is wrong',zh:'当热门的方法不合适'},
  lead:()=>tx('Preferences, no checker: is GRPO a good idea?','只有偏好对、没有判分程序，用 GRPO 好吗？'),
  enter:()=>({...lvOf(LV6,lv6).s}),
  expect:()=>tried6.has('grpo'),
  body:()=>tx(`<p>The task “Answers users prefer” has 20,000 preference pairs, 8 H100s and a budget of 3 GPU hours for the 8 GPUs together. GRPO is popular. Can it raise the win rate here?</p>`+
    `<p>GRPO needs a reward for each answer. No checker can tell whether users like an answer better, so a reward model must be trained on the pairs first, and every answer GRPO generates must be scored by it.</p>`+LEAD.lv6()+
    levers('pcap-ic-lv6',tx('Method','方法'),lv6,LV6,tried6,lv6Res),
    `<p>“回答更合用户口味”这个任务有 20,000 对偏好、8 张 H100，预算是 8 张卡加起来 3 GPU 小时。GRPO 正流行，在这里能用它提高胜率吗？</p>`+
    `<p>GRPO 要给每个回答一个分数。“用户更喜欢”没有判分程序能判，所以要先用偏好对训练一个奖励模型，GRPO 生成的每个回答都要让它打一次分。</p>`+LEAD.lv6()+
    levers('pcap-ic-lv6',tx('Method','方法'),lv6,LV6,tried6,lv6Res)),
  obs:()=>{const r=plan(S), d=plan(LV6[0].s);
    const head=r.inBudget?tx(`<p class="okline"><b>${methName(S.meth)}: ${ghU(r.gpuh)} for one pass, within the 3-hour budget.</b></p>`,`<p class="okline"><b>${methName(S.meth)}：训练一遍要 ${ghU(r.gpuh)}，在 3 GPU 小时的预算以内。</b></p>`)
      :tx(`<p class="badline"><b>${methName(S.meth)}: ${ghU(r.gpuh)} for one pass, over the 3-hour budget.</b></p>`,`<p class="badline"><b>${methName(S.meth)}：训练一遍要 ${ghU(r.gpuh)}，超出了 3 GPU 小时的预算。</b></p>`);
    const why={dpo:tx('<p>DPO generates no answers and needs no reward model: it trains on the pairs directly.</p>','<p>DPO 不生成回答，也不需要奖励模型，直接拿偏好对训练。</p>'),
      ppo:tx(`<p>It costs about ${fixd(r.gpuh/d.gpuh,1)} times DPO. The reward model is trained first, and every batch of answers PPO generates is trained on several times by both the policy and the value model.</p>`,`<p>它约是 DPO 的 ${fixd(r.gpuh/d.gpuh,1)} 倍。先要训练奖励模型，而且 PPO 每生成一批回答，策略模型和价值模型都要在上面训练好几遍。</p>`),
      grpo:tx(`<p>It costs about ${fixd(r.gpuh/d.gpuh,0)} times DPO: 8 answers per prompt to generate, train on and score. In this lab it reaches the same win rate as DPO, because both learn from the same pairs. A reward model also has gaps: the policy can exploit them so that the score rises while the true quality does not. That is called reward hacking (see ${labLink('reward-model')}).</p>`,`<p>它约是 DPO 的 ${fixd(r.gpuh/d.gpuh,0)} 倍，因为每个提问要生成 8 个回答，每个都要训练、都要打分。在这个 Lab 里，它最后的胜率和 DPO 一样，因为两者学的是同一批偏好对。奖励模型也有漏洞，策略模型可能钻它的空子，让分数涨了、真实质量却没涨，这叫 reward hacking（见${labLink('reward-model')}）。</p>`)}[S.meth]||'';
    return head+why;}},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Data first, then the budget, then the metric.','先看数据，再看预算，最后定指标。'),
  enter:()=>({card:'math',meth:'grpo',mode:'full',x:3}),
  /* the words below (no fit / fit, “over ten times”, one more pass) are checked against plan() in tests/post-capstone_numbers.py */
  body:()=>tx(`<ol class="pcap-ul">`+
    `<li><b>Start from the data.</b> The data a team has decides which methods can run at all. With only example answers, use SFT. With preference pairs, DPO is the cheap fit, because it generates no answers. With a program that checks the answer, GRPO can use “right or not” as its reward, and that is exactly what the test measures.`+
    `<p class="pcap-li2">Real post-training often chains them: SFT first to teach the format, then DPO to learn preferences, then RL with a checker. Tülu 3 does exactly this (see “${J('connect','Take it further')}”). This lab looks at one step at a time.</p></li>`+
    `<li><b>Then the budget: memory first, then GPU hours.</b> Besides the weights, full fine-tuning keeps a gradient and the optimizer state for every parameter. Together they are several times the size of the weights, so one GPU soon runs out of memory. LoRA trains only a few narrow matrices, so most of that goes away. When even the frozen weights do not fit, QLoRA stores them in 4 bits. GRPO generates several answers to each problem before it trains on them, so on the same data it costs over ten times the GPU hours of SFT.</li>`+
    `<li><b>Then the metric, with its error bar.</b> When the check uses only a few hundred examples, the score itself is uncertain. Just passing the target is not enough. Stop when the bottom of the error bar clears it too, or another sample of the same size may fail the check.</li></ol>`+
    `<div class="pcap-mis"><p>✗ “RL always beats SFT.” To teach a format, example answers and SFT are enough, fast and cheap. RL would first need a program that checks the format (Tülu 3’s RLVR uses programs to check the constraints in instructions), and then several generated answers per prompt, which costs far more.</p>`+
    `<p>✗ “DPO needs no generation, so it is always the cheaper choice.” For the same number of examples DPO costs more than SFT: each pair trains two answers, and the reference model reads both. It is cheaper only than PPO and GRPO, which generate answers. It also learns only from the answers already in its pairs (${cite(SRC.trlDpo,'TRL docs')}). With a checker but no pairs, using DPO means first letting the checker pair the model’s own right and wrong answers.</p>`+
    `<p>✗ “LoRA is always worse than full fine-tuning.” TRL’s summary of “LoRA Without Regret” says LoRA can match full fine-tuning when configured right, and RL needs only a small rank (${cite(SRC.trlLwr,'lora_without_regret.md')}). It also says LoRA falls behind when the data exceed its capacity.</p></div>`+A.shell.toScenarios(),
    `<ol class="pcap-ul">`+
    `<li><b>先看数据。</b>手里有什么数据，决定了哪些方法跑得起来。只有示范回答，就用 SFT。有偏好对，DPO 不用生成回答，是便宜又合适的选择。有能判对错的程序，GRPO 可以直接拿“答对没有”当奖励，而这正是验收要量的东西。`+
    `<p class="pcap-li2">真实的后训练常把它们串起来：先用 SFT 教会格式，再用 DPO 学偏好，最后用判分程序做 RL。Tülu 3 就是这样做的，见“${J('connect','继续深入')}”。这个 Lab 每次只看其中一步。</p></li>`+
    `<li><b>再看预算。先看显存，再看 GPU 小时。</b>全量微调除了权重，还要给每个参数存梯度和优化器状态。它们比权重本身大好几倍，所以一张卡很快就放不下。LoRA 只训练几个窄矩阵，这一大块就省掉了。连冻住的权重都放不下时，再用 QLoRA 把它们存成 4 位。GRPO 每道题要先生成好几个回答，再拿它们训练，所以同样的数据，它花的 GPU 小时是 SFT 的十几倍。</li>`+
    `<li><b>最后看指标，连同它的误差线。</b>验收只抽几百个样本时，分数本身就有误差。分数刚过目标还不够，误差线的下端也高过目标，才能停下。不然换一批同样多的样本再测，可能就不达标了。</li></ol>`+
    `<div class="pcap-mis"><p>✗“RL 总比 SFT 好。”只是教格式的话，有示范回答、用 SFT 就够了，又快又便宜。要用 RL，先得写一个能判格式对错的程序（Tülu 3 的 RLVR 就用程序检查指令里的约束），再给每个提问生成好几个回答，贵得多。</p>`+
    `<p>✗“DPO 不用生成，所以永远更便宜。”同样多的例子，DPO 比 SFT 贵。每对要训练两个回答，参考模型还要把它们各读一遍。它只比要生成回答的 PPO、GRPO 便宜。它也只能从偏好对里已有的回答学（${cite(SRC.trlDpo,'TRL 文档')}）。只有判分程序、没有偏好对时，要用 DPO，得先用判分程序把模型自己答对和答错的回答配成对。</p>`+
    `<p>✗“LoRA 一定比全量差。”TRL 对“LoRA Without Regret”的总结是：配置得当时，LoRA 能达到全量微调的效果，RL 只需要很小的秩（${cite(SRC.trlLwr,'lora_without_regret.md')}）。它也说了：数据超出 LoRA 的容量时，LoRA 不如全量。</p></div>`+A.shell.toScenarios())}
];
const leadOf=s=>s.lead?s.lead():'';

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'fmt',t:{en:'Support format, one GPU',zh:'客服格式，一张卡'},d:{en:'A small team with example conversations',zh:'小团队，手里有示范对话'},s:{card:'fmt',meth:'sft',mode:'lora',x:4}},
 {id:'dpo',t:{en:'Preferences with DPO',zh:'用 DPO 学偏好'},d:{en:'User choices, no reward model',zh:'有用户的选择，不训练奖励模型'},s:{card:'pref',meth:'dpo',mode:'full',x:1}},
 {id:'ppo',t:{en:'Preferences the RLHF way',zh:'按 RLHF 的老路学偏好'},d:{en:'A reward model, then PPO, as in InstructGPT',zh:'先训练奖励模型，再做 PPO，和 InstructGPT 一样'},s:{card:'pref',meth:'ppo',mode:'full',x:1}},
 {id:'grpo',t:{en:'Math with a checker',zh:'有判分程序的数学题'},d:{en:'GRPO with a right-or-wrong reward',zh:'GRPO，奖励只看对错'},s:{card:'math',meth:'grpo',mode:'full',x:3}},
 {id:'big',t:{en:'70B on one GPU',zh:'一张卡跑 70B'},d:{en:'The support format on the larger model',zh:'更大的模型学客服格式'},s:{card:'big',meth:'sft',mode:'qlora',x:4}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.card===S.card&&p.s.meth===S.meth&&p.s.mode===S.mode&&p.s.x===S.x);
function scenHTML(i){const r=plan(PRESETS[i].s), d=plan(PRESETS[1].s), k=fixd(plan(PRESETS[2].s).gpuh/d.gpuh,0);
  const res={fmt:tx('SFT with LoRA, 4 passes: it fits on one GPU, and the format pass rate clears 95% with room to spare.','用 LoRA 做 SFT，训 4 遍。一张卡放得下，格式合格率稳稳过了 95%。'),
   dpo:tx('DPO with full fine-tuning, 1 pass: the cheapest way to a win rate above 60%.','用 DPO 做全量微调，训 1 遍。花得最少，胜率就过了 60%。'),
   ppo:tx(`A reward model, then PPO, 1 pass: the same win rate as DPO for about ${k} times its GPU hours.`,`先训练奖励模型再做 PPO，训 1 遍。胜率和 DPO 一样，GPU 小时约是 DPO 的 ${k} 倍。`),
   grpo:tx('GRPO with full fine-tuning, 3 passes: the accuracy clears the 70% target within the budget.','用 GRPO 做全量微调，训 3 遍。答对率过了 70% 的目标，也没超预算。'),
   big:tx('70B with QLoRA, 4 passes: it just fits on one GPU, and the format pass rate clears 95%.','70B 用 QLoRA 做 SFT，训 4 遍。一张卡刚好放得下，格式合格率过了 95%。')}[PRESETS[i].id];
  const why={fmt:tx('<p>The team has only conversations and one GPU, so it is SFT, with LoRA to fit the GPU. The whole error bar of the 200-answer check is above 95%, so the margin is enough.</p>','<p>团队只有示范对话、只有一张卡，所以做 SFT，并用 LoRA 放进这张卡。200 个回答抽检的整条误差线都在 95% 以上，余量够了。</p>'),
   dpo:tx('<p>DPO trains on the pairs directly. It generates no answers, so one pass is cheap.</p>','<p>DPO 直接用偏好对训练。它不用自己生成回答，所以训练一遍很便宜。</p>'),
   ppo:tx('<p>This is the classic recipe of InstructGPT: four models in memory, and every batch of answers is generated, scored and trained on several times. In this lab it learns from the same pairs as DPO, so it reaches the same win rate.</p>','<p>这是 InstructGPT 的经典做法。显存里要放四个模型，每批回答都要生成、打分，再训练好几遍。在这个 Lab 里，它学的和 DPO 是同一批偏好对，所以胜率一样。</p>'),
   grpo:tx('<p>The checker gives the reward directly, so no reward model is needed. Generating 8 answers per problem is what makes it cost far more than SFT.</p>','<p>判分程序直接给奖励，不需要奖励模型。每道题要生成 8 个回答，所以它比 SFT 贵得多。</p>'),
   big:tx('<p>70B does not fit on one H100 even with LoRA. Only with the frozen weights in 4 bits does it fit.</p>','<p>70B 用 LoRA 也放不进一张 H100。冻住的权重存成 4 位以后，才放得下。</p>')}[PRESETS[i].id];
  return `<p><b>${res}</b></p>`+why;}

/* =====================================================================
   header, popovers, capability layer
   ===================================================================== */
const DESCS=()=>({
 guided:tx('7 short parts. Three teams, three kinds of data: start from the data, predict what GRPO costs, see memory pick LoRA, and let the metric say when to stop. Start here the first time.','一共 7 个部分。三个团队、三种数据：先从数据选方法，预测 GRPO 要花多少，看显存怎么逼你选 LoRA，再让指标告诉你什么时候停。第一次学从这里开始。'),
 sandbox:tx('Five real situations, each with a sensible setup; change the task, method, LoRA and passes in Settings.','五种真实的情况，各带一套合理的设置；在“自由调参”里可以改任务、方法、LoRA 和训练遍数。'),
 challenge:tx('A tutoring team got only the math problems and their final answers, and has one H100. Raise the accuracy within the budget, using as few GPU hours as possible.','一个辅导团队只拿到了数学题和最后的答案，也只有一张 H100。要在预算内提高答对率，GPU 小时越少越好。')});
const OUTCOME=()=>tx('pick SFT, DPO, reward model + PPO or GRPO for a task from its data, check that it fits in memory and budget, and say how you will know it worked.','按手里的数据给一个任务选 SFT、DPO、奖励模型 + PPO 或 GRPO，算清显存和预算够不够，并说出怎么验收。');
const FIDPOP=()=>{const g=plan({card:'math',meth:'grpo',mode:'full',x:1}), g0=plan({card:'math',meth:'grpo',mode:'full',x:1,noRef:true}),
  q=plan({card:'big',meth:'sft',mode:'qlora',x:1}), qr=plan({card:'big',meth:'sft',mode:'qlora',x:1,rc:true}), rule=MX('m(x) = m_\\infty - (m_\\infty - m_0) \\times 2^{-x/h}');
 return tx(
 `<p><b>Formula:</b> memory and GPU hours come from formulas; how far each method moves each metric is this lab’s own rule.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Memory per GPU, by what each parameter keeps:<ul class="pcap-pl">`+
  `<li>Full fine-tuning: 18 bytes per parameter.</li>`+
  `<li>LoRA: 2 bytes per frozen parameter, 16 per LoRA parameter (rank r = 16 on all 7 matrices).</li>`+
  `<li>QLoRA: 0.516 bytes per frozen linear weight (NF4 with double quantization). The embedding table and the LM head stay in BF16.</li>`+
  `<li>Reference and reward models: 2 bytes per parameter. With LoRA the reference is the base with the adapter switched off (TRL), so it takes no extra memory.</li></ul></li>`+
 `<li>All model bytes are split evenly over the GPUs. GB = 10⁹ bytes.</li>`+
 `<li>Activations: each GPU also holds activations and logits. This lab assumes a GPU processes 1,024 tokens at once and recomputes nothing. Activations grow with the tokens processed at once. Every example on the task cards is at most 512 tokens, so 1,024 tokens hold at least two of them. Recomputing activations in the backward pass (gradient checkpointing) saves most of them. TRL’s SFT turns it on by default (${cite(SRC.trlSftCfg,'sft_config.py')}). With it, the 70B with QLoRA needs about ${GB(qr.mem)} instead of ${GB(q.mem)}. Nothing that fits on this page would stop fitting, or the other way round.</li>`+
 `<li>Compute: a trained token costs 6N (about 4N with LoRA or QLoRA) and a token read by a frozen model 2N, where N is the number of parameters. Both run at 50% of the H100’s 989.5 TFLOP/s.</li>`+
 `<li>GPU hours are the GPUs times the hours, all GPUs together, as in papers: 8 GPUs for 1 hour are 8 GPU hours.</li>`+
 `<li>A check of the 50%: torchtune’s README measures 3,497 tokens per second for Llama 3.1 70B with LoRA on 8 A100s (${cite(SRC.ttReadme,'README')}). One A100 could not do that alone, so it is the total of the 8: about 437 tokens per second per GPU. At 4N per token that is about ${fixd(ttU()*100,0)}% of the A100’s 312 TFLOP/s. So the 50% here is on the fast side: at the measured share, the computing part of every GPU-hour figure would be about ${fixd(U0/ttU(),2)} times as large. The same README measures QLoRA only 2% slower than LoRA (Llama 3.2 3B on an A100).</li>`+
 `<li>Generating answers (rollouts): ${GEN_B} answers per GPU at once. Each step reads the 16.1 GB of BF16 weights from memory and adds a token to each answer. At 3.35 TB/s, one step of the 8B takes about ${fixd(stepMs('8b'),1)} ms.</li>`+
 `<li>GRPO generates 8 answers per problem and trains on each batch of answers once (TRL’s defaults num_generations = 8, num_iterations = 1, ${cite(SRC.trlGrpoCfg,'grpo_config.py')}). PPO generates one answer per prompt and trains 4 times on each batch (TRL’s num_ppo_epochs). The reward model is trained once over the pairs first.</li>`+
 `<li>TRL’s GRPO uses β = 0 by default (since v0.19.0) and then loads no reference model. On the math task the models would then take ${GB(g0.modelBytes)} instead of ${GB(g.modelBytes)}, and the reference model would no longer read the answers. verl, DeepSeek‑R1 and Tülu 3 keep a small KL term, and this lab keeps the reference model.</li></ul></div>`+
 `<div><div class="eyebrow">This lab’s rule</div><ul><li>The metric after x passes, with ${M('m_0')}, ${M('m_\\infty')} and h set per task and method:${rule}</li>`+
 `<li>Only two numbers of the math task are measured: Llama‑3.1‑8B’s 39.7% before training, and the 79.2% SimpleRL-Zoo reached after its RL training (${cite(SRC.simple,'README')}). This lab assumes GRPO reaches at most 79.2%.</li>`+
 `<li>LoRA and QLoRA reach what full fine-tuning reaches.</li>`+
 `<li>SFT learns to write what its answers contain. The math card’s worked solutions show the working. Final answers alone show none, so in this lab SFT on them leaves the accuracy where it was. On the task “Answers users prefer”, SFT trains on the chosen answers only and never sees what users turned down, so in this lab it leaves the win rate where it was too. No source this lab could check gives a number for either case, so the lab takes the most conservative setting: the ceiling ${M('m_\\infty')} equals ${M('m_0')}, the value before training, and training leaves the metric where it was.</li>`+
 `<li>Two papers bear on this, cited by title only under “Take it further”. The GSM8K paper trains on full worked solutions. STaR compares fine-tuning on answers alone with fine-tuning on the reasoning.</li>`+
 `<li>Tokens per example (128 for a prompt, 256 or 384 for an answer, 8 for a final answer alone) are assumptions.</li></ul>`+
 `<div class="eyebrow">Left out</div><ul><li>The KV cache while generating answers, prefill, the checker’s time, PPO’s forward passes of the policy and the value model before each update, and the waiting real systems do (for the longest answer, and when switching between generating and training). Real runs take longer than these ideal figures.</li><li>SimpleRL-Zoo reports about 100 steps in 15 hours on 2 × 8 H100s for its 7B and 14B models.</li><li>QLoRA’s README reports that its Guanaco models needed 24 hours of fine-tuning on one GPU (${cite(SRC.qlora,'README')}). Its 65B script trains 1,875 steps of 16 examples, each at most 528 tokens, and recomputes the activations (${cite(SRC.qloraSh,'finetune_guanaco_65b.sh')}). If that run took the 24 hours, it trained at most about 180 tokens per second, far fewer than the ideal figure here.</li></ul>`+
 `<div class="eyebrow">Versions</div><ul><li>TRL’s PPO is quoted from v1.12.0. Everything else from TRL is quoted from commit 14c8d70.</li></ul></div></div>`,
 `<p><b>公式计算：</b>显存和 GPU 小时由公式算出；每个方法能把指标提多高，是这个 Lab 自己定的规则。</p><div class="pop-grid"><div><div class="eyebrow">模型里算了什么</div><ul>`+
 `<li>每张卡的显存，按每个参数存多少算：<ul class="pcap-pl">`+
  `<li>全量微调：每个参数 18 字节。</li>`+
  `<li>LoRA：每个冻住的参数 2 字节，每个 LoRA 参数 16 字节（秩 r = 16，7 个矩阵都加）。</li>`+
  `<li>QLoRA：冻住的线性层权重每个 0.516 字节（NF4 加双重量化）。embedding 表和 LM head 仍是 BF16。</li>`+
  `<li>参考模型和奖励模型：每个参数 2 字节。用 LoRA 时，参考模型就是关掉 LoRA 的原模型（TRL 的做法），不另占显存。</li></ul></li>`+
 `<li>所有模型的字节平均分到每张卡上。GB 是 10⁹ 字节。</li>`+
 `<li>激活：每张卡还要放下激活和 logits。这个 Lab 假设每张卡一次处理 1,024 个 token，激活不重算。激活随一次处理的 token 数增长。任务卡上每条数据最多 512 个 token，所以 1,024 个 token 至少能放下两条。反向时重算激活（gradient checkpointing）能省下大部分激活。TRL 的 SFT 默认打开它（${cite(SRC.trlSftCfg,'sft_config.py')}）。打开后，70B 的 QLoRA 从 ${GB(q.mem)} 降到约 ${GB(qr.mem)}。这页上放得下的设置照样放得下，放不下的照样放不下。</li>`+
 `<li>算力：训练一个 token 算 6N（LoRA 和 QLoRA 约 4N），冻住的模型读一个 token 算 2N，N 是模型的参数量。这两项都按 H100 峰值 989.5 TFLOP/s 的 50% 算。</li>`+
 `<li>GPU 小时是卡数乘以小时数，所有卡加起来算，和论文一样：8 张卡跑 1 小时是 8 GPU 小时。</li>`+
 `<li>50% 和实测比一比：torchtune 的 README 里，70B 用 LoRA 在 8 张 A100 上每秒训练 3,497 个 token（${cite(SRC.ttReadme,'README')}）。一张 A100 做不到这么快，所以这是 8 张卡的总数，每张卡每秒约 437 个。按每个 token 4N 算，约是 A100 峰值 312 TFLOP/s 的 ${fixd(ttU()*100,0)}%。所以这里的 50% 偏快：按实测的比例，每个 GPU 小时里算力那部分要乘以约 ${fixd(U0/ttU(),2)}。同一份 README 里，QLoRA 只比 LoRA 慢 2%（Llama 3.2 3B，一张 A100）。</li>`+
 `<li>生成回答（rollout）：每张卡同时生成 ${GEN_B} 个回答。每一步把 16.1 GB 的 BF16 权重从显存读一遍，给每个回答添一个 token。显存带宽是 3.35 TB/s，所以 8B 一步约 ${fixd(stepMs('8b'),1)} ms。</li>`+
 `<li>GRPO 每道题生成 8 个回答，每批回答只训练一遍（TRL 的默认值 num_generations = 8、num_iterations = 1，${cite(SRC.trlGrpoCfg,'grpo_config.py')}）。PPO 每个提问生成 1 个回答，每批训练 4 遍（TRL 的 num_ppo_epochs）。奖励模型先在偏好对上训练一遍。</li>`+
 `<li>TRL 的 GRPO 从 v0.19.0 起默认 β = 0，这时不加载参考模型。数学任务上，所有模型占的显存会从 ${GB(g.modelBytes)} 降到 ${GB(g0.modelBytes)}，参考模型也不用再读回答。verl、DeepSeek‑R1 和 Tülu 3 都保留一个小的 KL 项，这个 Lab 也保留参考模型。</li></ul></div>`+
 `<div><div class="eyebrow">本 Lab 的规则</div><ul><li>训练 x 遍以后的指标按下面的式子算，${M('m_0')}、${M('m_\\infty')}、h 按任务和方法设定。${rule}</li>`+
 `<li>只有数学任务的两个数来自实测：Llama‑3.1‑8B 训练前的 39.7%，和 SimpleRL-Zoo 用 RL 训练后的 79.2%（${cite(SRC.simple,'README')}）。这个 Lab 假设 GRPO 最多到 79.2%。</li>`+
 `<li>LoRA 和 QLoRA 的效果和全量微调一样。</li>`+
 `<li>回答里写了什么，SFT 就学着写什么。数学团队的解题过程里有一步步的推理。只有最后答案时，回答里没有推理，所以在这个 Lab 里，这样的 SFT 不提高答对率。在“回答更合用户口味”这个任务上，SFT 只学选中的回答，看不到用户没选的那个，所以在这个 Lab 里它也不提高胜率。这两种情况都找不到能核对、又给出具体数字的来源，所以这个 Lab 取最保守的设定：上限 ${M('m_\\infty')} 就等于训练前的 ${M('m_0')}，训练后和训练前一样。</li>`+
 `<li>有两篇论文和这件事有关，“继续深入”里只引了标题。GSM8K 的论文用完整的解题过程训练。STaR 比较了只学答案和连推理一起学的微调。</li>`+
 `<li>每条数据的 token 数（提问 128，回答 256 或 384，只有最后答案时 8）是假设值。</li></ul>`+
 `<div class="eyebrow">没算的</div><ul><li>生成回答时的 KV cache、读题（prefill）、判分程序的时间、PPO 每次更新前策略模型和价值模型各做的一次前向，以及真实系统的等待（等最长的回答、在生成回答和训练之间切换）。真实训练比这些理想的数慢。</li><li>SimpleRL-Zoo 报告，它的 7B 和 14B 模型用 2 × 8 张 H100 跑约 100 步，花了 15 小时。</li><li>QLoRA 的 README 写着，Guanaco 模型只在一张卡上微调了 24 小时（${cite(SRC.qlora,'README')}）。它训练 65B 的脚本跑 1,875 步，每步 16 条、每条最多 528 个 token，反向时还重算激活（${cite(SRC.qloraSh,'finetune_guanaco_65b.sh')}）。如果这次训练用了 24 小时，每秒最多只训练约 180 个 token，比这里的理想值少得多。</li></ul>`+
 `<div class="eyebrow">版本</div><ul><li>TRL 的 PPO 引自 v1.12.0，TRL 的其余内容引自提交 14c8d70。</li></ul></div></div>`);};
const PRIPOP=()=>tx(`<p>This lab puts the post-training labs together: ${labLink('sft')}, ${labLink('lora')}, ${labLink('reward-model')}, ${labLink('ppo')} and ${labLink('dpo')}. It also uses three things from other labs:</p>`+
  `<ul class="pcap-pl"><li>18 bytes per parameter: the lab “Training memory”.</li><li>The error bar of a metric: ${labLink('eval-ci')}.</li><li>Win rates judged by a model: ${labLink('llm-judge')}.</li></ul>`+
  `<p>${labLink('grpo')} and ${labLink('rlvr')} are the two labs just before this one on the post-training path.</p>`,
  `<p>这个 Lab 把后训练的几个 Lab 串起来：${labLink('sft')}、${labLink('lora')}、${labLink('reward-model')}、${labLink('ppo')} 和 ${labLink('dpo')}。它还用到别的 Lab 讲过的三样东西：</p>`+
  `<ul class="pcap-pl"><li>每个参数 18 字节：见“训练显存”。</li><li>指标的误差线：见“${labLink('eval-ci')}”。</li><li>用模型当裁判算胜率：见“${labLink('llm-judge')}”。</li></ul>`+
  `<p>${labLink('grpo')} 和 ${labLink('rlvr')} 是后训练路线上紧挨在这个 Lab 前面的两个 Lab。</p>`);
const EXITS=()=>{const r8=plan({card:'math',meth:'grpo',mode:'full',x:1}), r16=plan({card:'math',meth:'grpo',mode:'full',x:1,G:16});
  const one1=plan({card:'pref',meth:'dpo',mode:'full',x:1}), memFull=plan({card:'fmt',meth:'dpo',mode:'full',x:1}), memLora=plan({card:'fmt',meth:'dpo',mode:'lora',x:1});
  return {
  explain:[
   {q:tx('Why does GRPO cost so much more than SFT for the same math problems?','同样的数学题，GRPO 为什么比 SFT 贵这么多？'),
    o:[tx('Each problem gets 8 generated answers, each trained on and read by the reference model, and generating them reads all the weights at every step.','每道题要生成 8 个回答，每个都要训练，都要让参考模型读一遍。生成回答时，每一步还要把全部权重读一遍。'),
       tx('The checker is slow: most of the time goes into marking the answers right or wrong.','判分程序很慢，大部分时间花在判对错上。'),
       tx('GRPO trains a value model next to the policy, which doubles the training work.','GRPO 要在策略模型旁边再训练一个价值模型，训练的工作量翻倍。')],a:0,
    n:[ '',tx('The checker only compares each answer’s final result; its time is tiny next to the model’s work, so this lab leaves it out.','判分程序只核对每个回答最后的答案，花的时间和模型的计算比起来太少，这个 Lab 没有算它。'),tx('GRPO has no value model: the group’s average reward is its baseline.','GRPO 没有价值模型：它用这一组回答的平均分当基准。')],
    why:tx('The work grows with the tokens trained on and generated: 8 answers per problem, each trained on, read by the reference model and generated token by token.','工作量跟着训练和生成的 token 走。每道题 8 个回答，每个都要训练、让参考模型读，还要一个 token 一个 token 地生成出来。')},
   {q:tx('Why does LoRA fit 8B on one H100 when full fine-tuning does not?','为什么 8B 全量微调放不进一张 H100，LoRA 却放得下？'),
    o:[tx('Full fine-tuning keeps 18 bytes per parameter. LoRA keeps the frozen weights at 2 bytes and gives gradients and Adam’s state only to the narrow matrices.','全量微调每个参数要 18 字节。LoRA 冻住的权重每个只占 2 字节，梯度和 Adam 的状态只给那两个窄矩阵。'),
       tx('LoRA cuts the model to about 1% of its parameters, so the weights themselves shrink by 99%.','LoRA 把模型的参数砍到原来的 1% 左右，所以权重本身也小了 99%。'),
       tx('LoRA keeps no activations, which were most of the memory.','LoRA 不存激活，而激活本来占了大部分显存。')],a:0,
    n:['',tx('The original weights stay in memory, frozen, at 2 bytes each; only the trained part is small.','原来的权重还在显存里，只是冻住了，每个 2 字节；变小的只是要训练的那部分。'),tx('LoRA keeps the activations; the model still runs forward and backward through every layer.','LoRA 照样要存激活：模型每一层照样要做前向和反向。')],
    why:tx('What full fine-tuning adds per parameter, the FP32 gradient and Adam’s state, is what LoRA removes from the frozen weights.','全量微调给每个参数多存的 FP32 梯度和 Adam 状态，正是 LoRA 从冻住的权重上省掉的。')}],
  transfer:[
   {q:tx(`On the task “Answers users prefer”, DPO with full fine-tuning uses only ${GB(one1.mem)} on each of the 8 H100s, with room to spare. Now only one H100 is left. What happens?`,`“回答更合用户口味”这个任务用 DPO 做全量微调时，8 张 H100 每张只用 ${GB(one1.mem)}，很宽松。现在只剩一张 H100，会怎样？`),
    o:[tx('Full fine-tuning still fits.','全量微调照样放得下。'),tx('Full fine-tuning does not fit; LoRA does, and its reference model costs no extra memory.','全量微调放不下。LoRA 放得下，它的参考模型也不另占显存。'),tx('Even LoRA does not fit, because DPO also holds a full reference model.','连 LoRA 也放不下，因为 DPO 还要放一整份参考模型。')],a:1,
    why:tx(`On one GPU, full DPO holds two models: the policy at 18 bytes per parameter and the reference model at 2 bytes per parameter. Together they take ${GB(memFull.modelBytes)}, and ${GB(memFull.mem)} with the activations and logits, so they do not fit. With LoRA the reference is the base model with the adapter switched off, as TRL does it, so the total is only ${GB(memLora.mem)}.`,`只有一张卡时，全量 DPO 要放下两个模型：策略模型每个参数 18 字节，参考模型每个参数 2 字节。两者一共 ${GB(memFull.modelBytes)}，算上激活和 logits 是 ${GB(memFull.mem)}，放不下。用 LoRA 时，参考模型就是关掉 LoRA 的原模型（TRL 就是这样做的），一共只要 ${GB(memLora.mem)}。`)},
   {q:tx(`On the math task, one pass of GRPO with 8 answers per problem takes about ${gh(r8.gpuh)} GPU hours. With 16 answers per problem instead, about how many?`,`数学任务上，每道题生成 8 个回答时，GRPO 训练一遍约 ${gh(r8.gpuh)} GPU 小时。改成每道题 16 个回答，大约要多少？`),
    o:[tx('About the same','差不多不变'),tx(`About ${fixd(r16.gpuh/r8.gpuh,0)} times as much`,`约是原来的 ${fixd(r16.gpuh/r8.gpuh,0)} 倍`),tx('About 4 times as much','约是原来的 4 倍')],a:1,
    why:tx('Generating, training and the reference model’s reading all grow with the number of answers, so twice the answers cost twice the GPU hours.','生成回答、训练和参考模型读回答，都跟着回答的多少走，所以回答多一倍，GPU 小时也多一倍。')}],
  practice:[{id:'rl-for-llms',d:tx('Question 5 asks you to compare PPO and GRPO in a table and say when you would choose each. Question 7 asks what data DPO needs.','第 5 题要用一张表比较 PPO 和 GRPO，说出什么时候选哪个。第 7 题问 DPO 需要什么数据。')},
    {id:'llm-fundamentals',d:tx('Question 10 asks what pretraining, SFT and preference optimization each optimize.','第 10 题问预训练、SFT 和偏好优化各自优化什么。')}]};};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Post-training capstone','后训练综合挑战'),title:l?L(l.q):tx('SFT, DPO or GRPO: which one fits your task?','SFT、DPO 还是 GRPO：你的任务该选哪个？'),
    minutes:20,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: the post-training labs','前置知识：前面几个后训练 Lab'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage pcap-stage" id="pcap-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="pcap-st-title"></div><div class="meter-sub" id="pcap-st-sub"></div></div><div class="stage-actions"><span class="status" id="pcap-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph pcap-graph" id="pcap-graph">`+
    `<div class="pcap-phone" id="pcap-phone"></div>`+
    `<div class="pcap-card" id="pcap-card"></div>`+
    `<div class="pcap-blk" id="pcap-setbox"><div class="pcap-bl">${tx('From the data to the result','从数据到结果')}</div><ol class="pcap-strip" id="pcap-strip" aria-label="${esc(tx('From the data to the result','从数据到结果'))}"></ol></div>`+
    `<div class="pcap-blk pcap-box" id="pcap-membox"><div class="pcap-bl">${tx('Memory','显存')}</div><div class="pcap-cap" id="pcap-mem-cap"></div><div class="pcap-hbar" id="pcap-mem"></div><ul class="pcap-list" id="pcap-mem-list"></ul></div>`+
    `<div class="pcap-blk pcap-box" id="pcap-costbox"><div class="pcap-bl">${tx('GPU hours and budget','GPU 小时和预算')}</div><div class="pcap-cap" id="pcap-cost-cap"></div><div id="pcap-cost"></div><ul class="pcap-list pcap-leg" id="pcap-cost-list"></ul><div class="pcap-hw" id="pcap-cost-hw"><div class="pcap-hh">${tx('How the GPU hours are counted','GPU 小时是怎么算出来的')}</div><dl class="pcap-how" id="pcap-cost-how"></dl></div></div>`+
    `<div class="pcap-blk pcap-box" id="pcap-metbox"><div class="pcap-bl">${tx('Result','结果')}</div><div class="pcap-cap" id="pcap-met-cap"></div><div class="pcap-met" id="pcap-met"></div><ul class="pcap-list pcap-leg" id="pcap-met-list"></ul></div>`+
    `<p class="pcap-blk pcap-tags small" id="pcap-tags"><span class="pcap-tag" tabindex="0" data-tipx="${esc(tx('Memory and GPU hours come from the formulas in “Show the math”.','显存和 GPU 小时由“展开公式”里的公式算出。'))}">${tx('Memory and GPU hours: formulas','显存和 GPU 小时：公式计算')}</span><span class="pcap-tag" tabindex="0" data-tipx="${esc(tx('How far a method moves a metric is this lab’s own rule; see the notes at the end of the page.','方法能把指标提多高，是本 Lab 自己的规则；见页末的“简化、假设和来源”。'))}">${tx('Metric: this lab’s rule','指标：这个 Lab 的规则')}</span></p>`+
   `</div>`+
   `<div class="metrics pcap-metrics" id="pcap-metrics"></div>`+
   `<details class="math" id="pcap-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="pcap-mathbody"></div></details></div>`;
}
const NOTE={
 u0:()=>tx('An assumed value, the same as in the training labs: the share of the H100’s 989.5 TFLOP/s peak a GPU reaches while it computes.','这是假设值，和训练那几个 Lab 一样。GPU 在算的时候，能跑到 H100 峰值 989.5 TFLOP/s 的 50%。'),
 gen:()=>{const r=fint(Math.round(genRate(MOD['8b'])/10)*10), ms=fixd(stepMs('8b'),1), kv=fixd(kvGen('8b')/1e9,1);
   return tx(`An assumed value. Each GPU generates ${GEN_B} answers at once. Each step reads all the BF16 weights from GPU memory once and adds one token to each answer, so for the 8B a step takes about ${ms} ms and each GPU generates about ${r} tokens per second. How many answers fit at once is decided by the memory left for their KV cache after the weights. These answers are short: the KV cache of ${GEN_B} of them takes only about ${kv} GB, so a real engine can generate far more at once. The more it generates together, the cheaper each token, until the GPU’s compute is used up. Each step also reads that KV cache, which this lab leaves out, so the time spent generating is on the low side here.`,
     `这是假设值。每张卡同时生成 ${GEN_B} 个回答。每一步把全部 BF16 权重从显存读一遍，给每个回答添一个 token，所以 8B 每一步约 ${ms} ms，每张卡每秒约生成 ${r} 个 token。能同时生成多少个回答，由放完权重以后显存里还剩多少地方放 KV cache 决定。这里的回答很短，${GEN_B} 个回答的 KV cache 只占约 ${kv} GB，所以真实的引擎能同时生成的远不止 ${GEN_B} 个。同时生成的越多，每个 token 越便宜，直到算力用满。每一步还要读这些 KV cache，这个 Lab 没有算，所以这里生成回答的时间偏少。`);},
 tok:()=>tx('Assumed sizes for the examples on the task cards: a support conversation 512 tokens; a prompt 128 tokens and an answer 256 tokens. A final answer alone (“#### 18” and the end token) is 8 tokens. Longer examples cost proportionally more.','任务卡上数据的大小是假设值。一段客服对话 512 个 token，一个提问 128 个 token，一个回答 256 个 token。只有最后答案时（“#### 18” 加上结束的 token），回答算 8 个 token。数据越长，花费按比例增加。'),
 rule:()=>tx('How far each method moves each metric is this lab’s own rule. By this rule the metric climbs toward a ceiling, more slowly the closer it gets. “Show the math” puts in the numbers of the current setup. The rule does not say one method is better in general.','每个方法能把指标提多高，是本 Lab 自己定的规则。按这个规则，指标朝一个上限爬，越接近上限涨得越慢。现在这套设置代进去的数，写在“展开公式”里。这个规则也不说明哪个方法在一般情况下更好。')
};
const segRow=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">${A.ui.seg({id,key,label,options:opts,value:String(val),note})}</div>`;
function controlsHTML(){
  return `<aside class="panel controls" id="pcap-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="pcap-lock" hidden></p>`+
   segRow('pcap-ctl-card','card',tx('Task','任务'),CARD_IDS.map(k=>[k,L(CARDW[k].s)]),S.card)+
   segRow('pcap-ctl-meth','meth',tx('Method','方法'),METHS.map(m=>[m,methName(m)]),S.meth)+
   segRow('pcap-ctl-mode','mode',tx('Which parameters to train','训练哪些参数'),MODES.map(m=>[m,modeName(m)]),S.mode)+
   segRow('pcap-ctl-x','x',tx('Passes over the data','训练几遍'),PASSES.map(x=>[String(x),x===0.5?tx('half','半遍'):String(x)]),S.x)+
   `<div class="pcap-grp lockable" id="pcap-ctl-as"><div class="pcap-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><div class="lbl">${tx('Share of the peak while computing','算的时候能到峰值的多少')} <b class="mono">50%</b></div><p class="ctl-note">${NOTE.u0()}</p></div>`+
    `<div class="ctl"><div class="lbl">${tx('Answers generated at once per GPU','每张卡同时生成几个回答')} <b class="mono">${GEN_B}</b></div><p class="ctl-note">${NOTE.gen()}</p></div>`+
    `<div class="ctl"><div class="lbl">${tx('Tokens per example','每条数据的 token 数')} <b class="mono">128 / 256 / 512</b></div><p class="ctl-note">${NOTE.tok()}</p></div>`+
    `<div class="ctl"><div class="lbl">${tx('How the metric moves','指标怎么变')}</div><p class="ctl-note">${NOTE.rule()}</p></div></div>`+
   `<div class="lockable" id="pcap-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`PEAK, U, HBM = 989.5e12, 0.5, 3.35e12   # H100; 50% assumed
N, GPUS, P, R, G = 8.03e9, 16, 128, 256, 8   # 8B, math card
def gpu_hours(method, n=7473, lora=False):
    k = 4 if lora else 6                 # trained tokens: 6N, 4N
    t = P + R                            # tokens per problem
    if method == 'sft':
        flops = k * N * t * n
        gen = 0
    else:                                # GRPO, a checker
        flops = (k + 2) * N * G * t * n  # + reference 2N
        rate = 64 / (2 * N / HBM)        # tokens/s per GPU
        gen = G * R * n / rate
    return (flops / (PEAK * U) + gen) / 3600

def mem_per_gpu(method, lora=False):  # GB, 1,024 tokens
    states = 2 * N + 16 * 41.9e6 if lora else 18 * N
    ref = 0 if lora or method == 'sft' else 2 * N
    act = 1024 * 32 * 167936 + 1024 * 128256 * 4
    return ((states + ref) / GPUS + act) / 1e9

s, g = gpu_hours('sft'), gpu_hours('grpo')
print(f'SFT {s:.3f} h  GRPO {g:.2f} h  ({g / s:.0f}x)')
print(f'GRPO {mem_per_gpu("grpo"):.1f} GB per GPU')`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="pcap-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="pcap-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The planner in plain Python: the GPU hours of one pass of SFT and of GRPO over the 7,473 math problems, and GRPO’s memory per GPU. It prints this page’s numbers.','用纯 Python 写的估算：SFT 和 GRPO 把 7,473 道数学题各过一遍要多少 GPU 小时，GRPO 每张卡要多少显存。打印出来的就是页面上的数。')}</p><pre class="code" id="pcap-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="pcap-prod">`+
   tx(`<li>TRL’s docs list which kind of data each trainer works with (${cite(SRC.trlFmt,'dataset_formats.md')}). SFTTrainer works with plain text, or with prompts paired with answers. DPOTrainer and RewardTrainer work with preference data: a prompt, the chosen answer and the rejected one. GRPOTrainer and RLOOTrainer need only the prompts. So the data you have already narrows the choice of method.</li>`+
      `<li>Tülu 3 runs all three in a row on Llama‑3.1‑8B: SFT, then DPO from the SFT model, then RLVR from the DPO model, with the learning rate falling from 5e-6 to 5e-7 to 3e-7 and KL β = 0.05 in the last stage (${cite(SRC.tulu,'docs/tulu3.md')}, ${cite(SRC.tuluReadme,'README')}).</li>`+
      `<li>The Llama 3.1 model card says the instruct models use “supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF)”, and that the fine-tuning data include public instruction datasets and over 25M synthetic examples (${cite(SRC.card,'MODEL_CARD.md')}).</li>`+
      `<li>With a PEFT adapter, TRL keeps no separate reference model: its PPO trainer sets <code>ref_model = None</code> and switches the adapter off to score with the base (${cite(SRC.trlPpo,'ppo_trainer.py')}). QLoRA’s README reports fine-tuning a 65B model on one 48 GB GPU (${cite(SRC.qlora,'README')}). Its 65B script recomputes the activations in the backward pass (<code>--gradient_checkpointing</code>, ${cite(SRC.qloraSh,'finetune_guanaco_65b.sh')}), and TRL’s SFT does so by default too. This lab does not, so its 70B needs ${GB(plan({card:'big',meth:'sft',mode:'qlora',x:1}).mem)}; with recomputation it would need about ${GB(plan({card:'big',meth:'sft',mode:'qlora',x:1,rc:true}).mem)}.</li>`,
      `<li>TRL 的文档列出了每个 trainer 适配哪种数据（${cite(SRC.trlFmt,'dataset_formats.md')}）。SFTTrainer 用整段文字，或者配好回答的提问。DPOTrainer 和 RewardTrainer 用偏好数据，每条有提问、选中的回答和落选的回答。GRPOTrainer 和 RLOOTrainer 只需要提问。所以手里有什么数据，就已经缩小了能选的方法。</li>`+
      `<li>Tülu 3 在 Llama‑3.1‑8B 上把三种方法依次做完：先 SFT，再从 SFT 的模型做 DPO，最后从 DPO 的模型做 RLVR。三段的学习率依次是 5e-6、5e-7 和 3e-7，最后一段的 KL β = 0.05（${cite(SRC.tulu,'docs/tulu3.md')}、${cite(SRC.tuluReadme,'README')}）。</li>`+
      `<li>Llama 3.1 的模型卡写着，instruct 模型用了“supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF)”，微调数据包括公开的指令数据集和超过 25M 条合成数据（${cite(SRC.card,'MODEL_CARD.md')}）。</li>`+
      `<li>用 PEFT 的 LoRA 时，TRL 不另存参考模型。它的 PPO trainer 把 <code>ref_model</code> 设成 None，算参考模型的概率时关掉 LoRA，用原模型算（${cite(SRC.trlPpo,'ppo_trainer.py')}）。QLoRA 的 README 写着它在一张 48 GB 的卡上微调了 65B 的模型（${cite(SRC.qlora,'README')}）。它微调 65B 的脚本在反向时重算激活（<code>--gradient_checkpointing</code>，${cite(SRC.qloraSh,'finetune_guanaco_65b.sh')}），TRL 的 SFT 默认也这样做。这个 Lab 不重算激活，所以 70B 要 ${GB(plan({card:'big',meth:'sft',mode:'qlora',x:1}).mem)}，重算时约 ${GB(plan({card:'big',meth:'sft',mode:'qlora',x:1,rc:true}).mem)}。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.pIgpt,'Training language models to follow instructions with human feedback')}<span>${tx('Ouyang et al., 2022 (arXiv 2203.02155): SFT, then a reward model, then PPO. Cited by title only.','Ouyang 等，2022（arXiv 2203.02155）：先 SFT，再训练奖励模型，再做 PPO。只引标题。')}</span></li>`+
   `<li>${cite(SRC.pDpo,'Direct Preference Optimization: Your Language Model is Secretly a Reward Model')}<span>${tx('Rafailov et al., 2023 (arXiv 2305.18290). Cited by title only.','Rafailov 等，2023（arXiv 2305.18290）。只引标题。')}</span></li>`+
   `<li>${cite(SRC.pMath,'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models')}<span>${tx('Shao et al., 2024 (arXiv 2402.03300): where GRPO was introduced. Cited by title only.','Shao 等，2024（arXiv 2402.03300）：GRPO 出自这篇。只引标题。')}</span></li>`+
   `<li>${cite(SRC.pGsm,'Training Verifiers to Solve Math Word Problems')}<span>${tx('Cobbe et al., 2021 (arXiv 2110.14168): the GSM8K paper; its models are fine-tuned on the full worked solutions. Cited by title only.','Cobbe 等，2021（arXiv 2110.14168）：GSM8K 的论文，模型用完整的解题过程微调。只引标题。')}</span></li>`+
   `<li>${cite(SRC.pStar,'STaR: Bootstrapping Reasoning With Reasoning')}<span>${tx('Zelikman et al., 2022 (arXiv 2203.14465): compares fine-tuning on final answers alone with fine-tuning on the reasoning. Cited by title only.','Zelikman 等，2022（arXiv 2203.14465）：比较了只学最后答案和连推理一起学的微调。只引标题。')}</span></li>`+
   `<li>${cite(SRC.pTulu,'Tülu 3: Pushing Frontiers in Open Language Model Post-Training')}<span>${tx('Lambert et al., 2024 (arXiv 2411.15124): SFT, DPO and RLVR in one pipeline. Cited by title only; the settings above come from the open-instruct repository.','Lambert 等，2024（arXiv 2411.15124）：SFT、DPO、RLVR 串成一条流水线。只引标题。上面的设置取自 open-instruct 仓库。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="pcap-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="pcap-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="pcap-app m2" id="pcap-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="pcap-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see which method fits it and how much memory and how many GPU hours it takes.','选一个场景，看看它适合哪种方法，显存和 GPU 小时各要多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="pcap-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
const chSeg=(id,key,label,opts,val)=>A.ui.seg({id,key,label,options:opts,value:String(val),inline:true});
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted pcap-sub">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('One H100, 3 GPU hours: math accuracy from 39.7% to 65%','一张 H100、3 GPU 小时：把答对率从 39.7% 提到 65%')}</h3>`+
   `<p>${tx(`A tutoring team wants Llama‑3.1‑8B to solve more GSM8K problems. Today it gets 39.7% of the 1,319 test problems right. GSM8K comes with a worked solution for every problem, but this team got only the 7,473 training problems and their final answers. It also has a checker, one H100 and a budget of 3 ${DEF('GPU hours',TIP.gpuh())}: with one GPU, that is 3 hours.`,
            `一个辅导团队想让 Llama‑3.1‑8B 答对更多 GSM8K 的题。它现在能答对测试集 1,319 道题里的 39.7%。GSM8K 本来每道题都带解题过程，可这个团队只拿到了训练集 7,473 道题的题目和最后的答案。团队另有一个判分程序和一张 H100，预算 3 ${DEF('GPU 小时',TIP.gpuh())}。只有一张卡，3 GPU 小时就是 3 小时。`)}</p>`+
   `<p>${tx('Choose the method, which parameters to train and how many passes. A setup meets the goal only if it passes four checks: the team’s data can make the method’s training examples, the run fits on one GPU, it stays within the budget, and the accuracy reaches 65%. Among the setups that pass all four, the fewer GPU hours, the better.','请选方法、训练哪些参数、训练几遍。一个设置要过四关才算达标：手里的数据能做成这个方法的训练样本，一张卡放得下，不超预算，答对率到 65%。四关都过的设置里，GPU 小时越少越好。')}</p>`+
   chSeg('pcap-ic-chmeth','chmeth',tx('Method','方法'),METHS.map(m=>[m,methName(m)]),ch.meth)+
   chSeg('pcap-ic-chmode','chmode',tx('Which parameters to train','训练哪些参数'),MODES.map(m=>[m,modeName(m)]),ch.mode)+
   chSeg('pcap-ic-chx','chx',tx('Passes over the data','训练几遍'),PASSES.map(x=>[String(x),x===0.5?tx('half','半遍'):String(x)]),ch.x)+
   `<div class="goal mono pcap-goal" id="pcap-ch-g0"><span>${tx('Data','数据')}</span><span><b id="pcap-ch-can"></b></span></div>`+
   `<div class="goal mono pcap-goal" id="pcap-ch-g1"><span>${tx('Memory per GPU','每张卡的显存')}</span><span><b id="pcap-ch-mem"></b>${tx(' (limit 80 GB)','（上限 80 GB）')}</span></div>`+
   `<div class="goal mono pcap-goal" id="pcap-ch-g2"><span>${tx('GPU hours','GPU 小时')}</span><span><b id="pcap-ch-gh"></b>${tx(' (limit 3 GPU hours, fewer is better)','（上限 3 GPU 小时，越少越好）')}</span></div>`+
   `<div class="goal mono pcap-goal" id="pcap-ch-g3"><span>${tx('Test accuracy','测试集答对率')}</span><span><b id="pcap-ch-acc"></b>${tx(' (target at least 65%)','（目标至少 65%）')}</span></div>`+
   submit+`<div id="pcap-ch-res" class="pcap-ch-res" aria-live="polite"></div></div>`;
}
const chMe=()=>{const r=plan({card:'ch',...ch}); return {...ch,r,ok:r.ok};};
const chName=e=>`${methName(e.meth)}${tx(', ','，')}${modeName(e.mode)}${tx(', ','，')}${passW(e.x)}`;
/* the four checks of a setup, in the order of the task card's rows: the data can make the method's training examples, it fits in
   memory, it fits the budget, it reaches the target. n is the check's name, t the sentence when it fails. */
function chChecks(e){const r=e.r;
  return [{k:'data',n:tx('Data: ','数据：'),ok:r.can,t:r.can?'':tx(`${methName(e.meth)} needs ${dataName(r.miss)}, and the team has none, so it cannot run.`,`${methName(e.meth)} 需要${dataName(r.miss)}，团队手里没有，所以跑不起来。`)},
    {k:'mem',n:tx('Memory: ','显存：'),ok:r.fits,t:tx(`each GPU would need ${GB(r.mem)}, more than 80 GB.`,`每张卡要 ${GB(r.mem)}，超过了 80 GB。`)},
    {k:'budget',n:tx('Budget: ','预算：'),ok:r.inBudget,t:tx(`it takes ${ghU(r.gpuh)}, over the budget of 3.`,`要 ${ghU(r.gpuh)}，超出了 3 GPU 小时的预算。`)},
    {k:'target',n:tx('Accuracy: ','答对率：'),ok:r.hits,t:r.flat?tx(`it stays at ${pc(r.metric)}, as before training, below 65%.`,`停在 ${pc(r.metric)}，和训练前一样，没到 65%。`):tx(`it reaches only ${pc(r.metric)}, below 65%.`,`只到 ${pc(r.metric)}，没到 65%。`)}];}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chMe(), best=chBest(); let h;
  if(me.ok&&chMinimal(me)) h=tx(`<p class="okline"><b>Goal met: ${pc(me.r.metric)} on the test, ${ghU(me.r.gpuh)}, ${GB(me.r.mem)} per GPU. No setup that meets the goal uses fewer GPU hours.</b></p>`,`<p class="okline"><b>达成目标：测试集答对 ${pc(me.r.metric)}，${ghU(me.r.gpuh)}，每张卡 ${GB(me.r.mem)}。达标的设置里，没有比它更省 GPU 小时的。</b></p>`);
  else if(me.ok) h=tx(`<p class="okline"><b>Goal met: ${pc(me.r.metric)}, ${ghU(me.r.gpuh)}.</b></p><p>But a cheaper setup exists: ${esc(chName(best))}, ${pc(best.r.metric)} for ${ghU(best.r.gpuh)}.</p>`,`<p class="okline"><b>达标了：${pc(me.r.metric)}，${ghU(me.r.gpuh)}。</b></p><p>不过还有更省的设置：${esc(chName(best))}，${pc(best.r.metric)}，只要 ${ghU(best.r.gpuh)}。</p>`);
  else {
    const ck=chChecks(me), bad=me.r.can?ck.filter(x=>!x.ok):[ck[0]];
    h=tx(`<p class="badline"><b>Not yet: ${bad.length===1?'one of the four checks fails':`${['','','two','three','four'][bad.length]} of the four checks fail`}.</b></p>`,`<p class="badline"><b>还没达到：四关里有${['一','两','三','四'][bad.length-1]}关没过。</b></p>`)+
      `<ul class="pcap-badchk">${bad.map(x=>`<li><b>${esc(x.n)}</b>${x.t}</li>`).join('')}</ul>`+
      (me.meth==='sft'&&me.r.can?tx(`<p>SFT does run: each problem with its final answer makes one training example. But SFT learns to write what its answers contain, so here it learns to give the answer without the working. In this lab that does not raise the accuracy. For SFT to help, the data would need worked solutions for it to imitate.</p>`,
        `<p>SFT 其实能跑。每道题配上它的最后答案，就是一条训练样本。可回答里写了什么，SFT 就学着写什么，所以它学到的是不写过程、直接报答案。在这个 Lab 里，这样训练不提高答对率。要让 SFT 有用，数据里得有一步步的解题过程，让它照着学。</p>`):'')+
      tx(`<p class="small">Hint: first ask what each method can learn from this data. Then make it fit on one GPU. Then train just enough passes.</p>`,`<p class="small">提示：先想清楚每个方法能从这套数据里学到什么，再让它放进一张卡，最后只训练刚好够的遍数。</p>`)+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(me.ok) chReveal=true;
  if(!chReveal) return h;
  /* brief (the owner's review of batch 8, 2026-10-07): every setup goes in, and the chart keeps the ones that pass all four checks, your pick
     and the initial setup; the note counts the rest. Your pick always stays (Codex, forwarded by the owner on 2026-10-08, rule G8), also
     when the team's data cannot make its training examples (DPO, reward model + PPO): that row has no bar, its value says why in a few
     words, and it is drawn as failing (a red ring, red words and its own legend entry) */
  const same=(a,b)=>a.meth===b.meth&&a.mode===b.mode&&a.x===b.x, init=e=>same(e,CH.start), mine=e=>same(e,me);
  const noRun=e=>tx(`cannot run: no ${dataName(e.r.miss)}`,`跑不起来，没有${dataName(e.r.miss)}`);
  const val=e=>e.r.can?tx(`${gh(e.r.gpuh)} h, ${pc(e.r.metric)}${e.r.fits?'':', does not fit'}`,`${gh(e.r.gpuh)} 小时，${pc(e.r.metric)}${e.r.fits?'':'，放不下'}`):noRun(e);
  const order=chAll().slice().sort((a,b)=>(init(b)-init(a))||(b.ok-a.ok)||((a.r.gpuh||0)-(b.r.gpuh||0)));
  const rows=order.map(e=>({label:(chMinimal(e)?'✓ ':'')+chName(e),value:e.r.can?e.r.gpuh:0,text:val(e),ok:e.ok,me:mine(e),
    tags:(init(e)?[{text:tx('initial setup','初始设置')}]:[]).concat(mine(e)?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  const shown=order.filter(e=>e.ok||mine(e)||init(e));
  let chart=A.ui.cmpChart({brief:true,title:tx('GPU hours on one H100, with the test accuracy (budget 3 GPU hours, accuracy target at least 65%)','一张 H100 上要的 GPU 小时，以及测试集答对率（预算 3 GPU 小时，答对率目标至少 65%）'),
    aria:tx('GPU hours and accuracy: ','GPU 小时和答对率：')+shown.map(e=>e.r.can?`${chName(e)}${tx(': ','：')}${ghU(e.r.gpuh)}${tx(', ','，')}${pc(e.r.metric)}`:`${chName(e)}${tx(': cannot run, it needs ','：跑不起来，没有')}${dataName(e.r.miss)}`).join(tx('; ','；')),rows,
    hiddenText:n=>tx(`The other ${n} setups fail at least one of the four checks and are not shown.`,`其余 ${n} 种设置至少有一关没过，图里没有列出。`),
    goal:3,goalLabel:tx('budget: 3 GPU hours','预算：3 GPU 小时'),goalText:tx('budget','预算'),okText:tx('meets every goal','全部达标'),noText:tx('misses at least one','至少一项没达标'),legend:true});
  /* the kit draws the pick as <div class="cmp-row me"> (it never passes, so no " ok"); the legend is the one div.cmp-leg */
  if(!me.r.can) chart=chart.replace('<div class="cmp-row me">','<div class="cmp-row me pcap-norun">')
    .replace(/(<div class="cmp-leg"[^>]*>[\s\S]*?)(<\/div>)/,(m,a,b)=>`${a}<span><i class="pcap-leg-norun"></i>${esc(tx('cannot run','跑不起来'))}</span>${b}`);
  h+=chart;
  const g1=chAll().find(e=>e.meth==='grpo'&&e.mode==='lora'&&e.x===1), g4=chAll().find(e=>e.meth==='grpo'&&e.mode==='lora'&&e.x===4);
  const short=t=>DEF(t,tx(`${pc(g1.r.metric)} after 1 pass.`,`训 1 遍是 ${pc(g1.r.metric)}。`)), over=t=>DEF(t,tx(`${ghU(g4.r.gpuh)} for 4 passes.`,`训 4 遍要 ${ghU(g4.r.gpuh)}。`));
  const clear=best.r.metric-best.r.band>=CARDS.ch.target;
  h+=tx(`<p class="small">SFT does run on the final answers, and with LoRA it costs little. But it learns to give the answer without the working, so in this lab the accuracy stays at ${pc(CARDS.ch.m0)}. Worked solutions would change that. DPO and the reward model need preference pairs, which the team does not have. That leaves GRPO: the model writes its own working, and the checker rewards the right answers.</p>`+
    `<p class="small">Full GRPO does not fit on one GPU. LoRA and QLoRA both fit and, in this lab, cost the same GPU hours. One pass ${short('falls short')}, two passes clear 65%, and four would ${over('break the budget')}.</p>`+
    `<p class="small">Caveat: these accuracies follow this lab’s rule. The test has 1,319 problems, so its error bar is about ±${fixd(best.r.band,1)} points. ${clear?`At ${passW(best.x)} the whole error bar is above 65%, so the margin is enough.`:`At ${passW(best.x)} part of the error bar is still below 65%, so the margin is thin.`}</p>`,
    `<p class="small">SFT 能用最后答案来训练，用 LoRA 也花不了多少。可它学到的是不写过程、直接报答案，所以在这个 Lab 里答对率停在 ${pc(CARDS.ch.m0)}。有解题过程就不一样了。DPO 和奖励模型都要偏好对，团队没有。剩下的是 GRPO，模型自己写解题过程，判分程序给答对的回答奖励。</p>`+
    `<p class="small">全量 GRPO 一张卡放不下。LoRA 和 QLoRA 都放得下，在这个 Lab 里花的 GPU 小时也一样。训 1 遍${short('还差一截')}，训 2 遍就过了 65%，训 4 遍就${over('超预算了')}。</p>`+
    `<p class="small">注意：这些答对率按的是这个 Lab 的规则。测试集有 1,319 道题，误差线约 ±${fixd(best.r.band,1)} 个百分点。${clear?`训${passW(best.x)}时，整条误差线都在 65% 以上，余量够了。`:`训${passW(best.x)}时，误差线还有一截在 65% 以下，余量不多。`}</p>`);
  return h;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},data:{en:'Try three task and method pairs to continue.',zh:'试三种任务和方法的组合，才能继续。'},
    memory:{en:'Try “70B, QLoRA” to continue.',zh:'试一下“70B，QLoRA”，才能继续。'},metric:{en:'Reach the 95% target to continue.',zh:'训到 95% 的目标，才能继续。'},wrong:{en:'Try “GRPO with a reward model” to continue.',zh:'试一下“GRPO，用奖励模型打分”，才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:false,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#pcap-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
const LVSET={'pcap-ic-lv1':[()=>view1,()=>new Set(),null,v=>v],'pcap-ic-lv4':[()=>lv4,()=>tried4,lv4Res,v=>v],'pcap-ic-lv5':[()=>lv5,()=>tried5,lv5Res,v=>+v],'pcap-ic-lv6':[()=>lv6,()=>tried6,lv6Res,v=>v]};
function updateRail(){
  const rail=$('#pcap-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    Object.keys(LVSET).forEach(id=>{const [cur,tr,res,cv]=LVSET[id]; $$(`#${id} [data-lever]`,rail).forEach(b=>{const v=cv(b.dataset.lever); b.setAttribute('aria-pressed',String(v===cur())); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=res&&(tr().has(v)||v===cur())?res(v):'';});});
    $$('.seg-ctl[data-key="card"]',rail).forEach(g=>A.ui.setPressed(g,S.card)); $$('.seg-ctl[data-key="meth"]',rail).forEach(g=>A.ui.setPressed(g,S.meth));
    if(box){const html=s.obs?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#pcap-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chMe(), b0=$('#pcap-ch-can',rail); if(!b0) return;
    b0.textContent=e.r.can?tx(`runs: ${exOf(CARDS.ch,e.meth,e.r)}`,`能跑，${exOf(CARDS.ch,e.meth,e.r)}`):tx(`cannot run: no ${dataName(e.r.miss)}`,`跑不起来，没有${dataName(e.r.miss)}`);
    $('#pcap-ch-mem',rail).textContent=GB(e.r.mem); $('#pcap-ch-gh',rail).textContent=e.r.can?gh(e.r.gpuh):'—'; $('#pcap-ch-acc',rail).textContent=pc(e.r.metric);
    const set=(id,ok)=>{const g=$(id,rail); g.classList.toggle('met',ok===true); g.classList.toggle('pcap-over',ok===false);};
    set('#pcap-ch-g0',e.r.can); set('#pcap-ch-g1',e.r.fits); set('#pcap-ch-g2',e.r.can?e.r.inBudget:null); set('#pcap-ch-g3',e.r.hits);
    const res=$('#pcap-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
function secs(){
  if(mode!=='guided') return {card:true,strip:true,mem:true,cost:'one',met:true};
  return ({mission:{card:true,strip:true},data:{card:true,strip:true},predict:{card:true,strip:true,cost:'p3'},memory:{card:true,strip:true,mem:true,cost:'one'},
    metric:{card:true,strip:true,met:true,cost:'one'},wrong:{card:true,strip:true,mem:true,cost:'one'},remember:{card:true,strip:true,mem:true,cost:'one',met:true}})[STEPS[step].id];
}
const curCard=()=>mode==='challenge'?CARDS.ch:CARDS[S.card];
/* the budget counts all GPUs together (the owner, 2026-10-09: “不知道是所有GPU一起跑的，还是单个GPU算”); the words say what that is on the card's GPUs */
const wallW=(gpuh,n)=>{const h=gpuh/n, m=h*60;
  const hd=Number.isInteger(h)||h>=10?0:1;
  return h>=1?tx(`${fixd(h,hd)} ${h===1?'hour':'hours'}`,`${fixd(h,hd)} 小时`):(m>=1?tx(`${fint(Math.round(m))} minutes`,`${fint(Math.round(m))} 分钟`):tx('under a minute','不到 1 分钟'));};
const budgetW=c=>c.gpus>1?tx(`${gh(c.budget)} ${DEF('GPU hours',TIP.gpuh())}, all ${c.gpus} GPUs together (the ${c.gpus} GPUs running for about ${wallW(c.budget,c.gpus)})`,`${gh(c.budget)} ${DEF('GPU 小时',TIP.gpuh())}，${c.gpus} 张卡加起来算（${c.gpus} 张卡一起跑约 ${wallW(c.budget,c.gpus)}）`)
  :tx(`${gh(c.budget)} ${DEF('GPU hours',TIP.gpuh())} (one GPU, so ${wallW(c.budget,1)})`,`${gh(c.budget)} ${DEF('GPU 小时',TIP.gpuh())}（只有 1 张卡，就是 ${wallW(c.budget,1)}）`);
const cardLine=c=>tx(`${L(CARDW[c.id].k)}: ${gpuW(c)}, budget ${ghU(c.budget)}${c.gpus>1?` for all ${c.gpus} GPUs together`:''}, target ${c.target}%`,`${L(CARDW[c.id].k)}：${gpuW(c)}，预算 ${ghU(c.budget)}${c.gpus>1?`（${c.gpus} 张卡加起来）`:''}，目标 ${c.target}%`);
function drawCard(){
  const c=curCard(), w=CARDW[c.id], host=$('#pcap-card',app);
  /* on a phone the main picture comes first: after part 1 the card shrinks to one line the reader can open */
  const fold=innerWidth<=760&&mode==='guided'&&step>0, was=host.querySelector('details.pcap-cardd'), open=!!(fold&&was&&was.open);
  const full=`<div class="pcap-ct">${esc(L(w.t))}<span class="pcap-ex" data-tipx="${esc(tx('An example written for this lab.','为这个 Lab 写的示例。'))}" tabindex="0">${tx('example','示例')}</span></div>`+
   `<dl class="pcap-dl"><div><dt>${tx('Goal','目标')}</dt><dd>${esc(L(w.goal))}</dd></div>`+
   `<div><dt>${tx('Metric','指标')}</dt><dd>${esc(L(w.met))}${tx(`, from ${c.m0}% to at least ${c.target}%`,`，从 ${c.m0}% 提到至少 ${c.target}%`)}</dd></div>`+
   `<div><dt>${tx('Data','数据')}</dt><dd>${esc(L(w.data))}</dd></div>`+
   `<div><dt>${tx('Model and GPUs','模型和卡')}</dt><dd>${cardModel(c)}${tx(` on ${gpuW(c)} (80 GB each)`,`，${gpuW(c)}（每张 80 GB）`)}</dd></div>`+
   `<div><dt>${tx('Budget','预算')}</dt><dd>${budgetW(c)}</dd></div></dl>`;
  const html=fold?`<details class="pcap-cardd"${open?' open':''}><summary>${esc(cardLine(c))}</summary>${full}</details>`:full;
  if(host._h!==html){host.innerHTML=html; host._h=html;}
}
function stripItems(){
  const c=curCard(), st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), v=secs(), hid=predHidden();
  const names=['demo','ans','pairs','checker'].filter(k=>c.data[k]).map(dataName), have=names.join(tx(', ','、'))||tx('nothing usable','没有能用的');
  /* each data name stays on one line in a narrow chip (“最后答案、判分程序” breaks after 、, not inside a name) */
  const items=[{k:'data',h:tx('Data','数据'),v:have,vh:names.length?names.map(n=>`<span class="pcap-nw">${esc(n)}</span>`).join(tx(', ','、')):'',ok:null,tip:tx('What the team has. Each method needs its own kind of data.','团队手里有的数据。每个方法要的数据不一样。')}];
  if(!isP('mission')) items.push({k:'meth',h:tx('Method','方法'),v:methName(st.meth),sub:r.can?exOf(c,st.meth,r):tx(`needs ${dataName(r.miss)}`,`缺${dataName(r.miss)}`),ok:r.can,tip:r.can?canTip(c,st.meth):tx(`This method needs ${dataName(r.miss)}, which the card does not have.`,`这个方法需要${dataName(r.miss)}，任务卡上没有。`)});
  if(!isP('mission')&&!isP('data')){
    items.push({k:'mode',h:tx('Parameters trained','训练哪些参数'),v:modeName(st.mode),sub:tx(`${GB(r.mem)} per GPU`,`每张卡 ${GB(r.mem)}`),ok:r.fits,tip:r.fits?tx(`Each GPU needs ${GB(r.mem)} of its 80 GB.`,`每张卡要 80 GB 里的 ${GB(r.mem)}。`):tx(`Each GPU would need ${GB(r.mem)}, more than its 80 GB.`,`每张卡要 ${GB(r.mem)}，超过了 80 GB。`)});
    items.push({k:'x',h:tx('Passes','训练几遍'),v:passW(st.x),sub:hid?'—':(r.can?ghU(r.gpuh):'—'),ok:hid||!r.can?null:r.inBudget,tip:hid?tx('Hidden until you answer the question.','作答之前先隐藏。'):r.can?tx(`${ghU(r.gpuh)} against a budget of ${ghU(c.budget)}.`,`${ghU(r.gpuh)}，预算 ${ghU(c.budget)}。`):tx('Cannot run with this data.','这套数据跑不起来。')});
    if(mode!=='guided'||v.met||isP('wrong')) items.push({k:'res',h:tx('Result','结果'),v:r.can?`${metricName(c)} ${pc(r.metric)}`:'—',sub:tx(`target ${c.target}%`,`目标 ${c.target}%`),ok:r.can?r.hits:false,tip:r.can?(r.flat?flatWhy(c):tx(`By this lab’s rule, ${passW(st.x)} of ${methName(st.meth)} gives ${pc(r.metric)}; the target is ${c.target}%.`,`按这个 Lab 的规则，${methName(st.meth)} 训练${passW(st.x)}得到 ${pc(r.metric)}，目标是 ${c.target}%。`)):tx('The method cannot run, so the metric stays where it started.','方法跑不起来，指标停在原地。')});
  }
  return items;
}
function drawStrip(){
  const items=stripItems();
  $('#pcap-strip',app).innerHTML=items.map((it,i)=>`<li class="pcap-chip${it.ok===false?' pcap-bad':(it.ok===true?' pcap-good':'')}" tabindex="0" data-tipx="${esc(it.tip)}"><span class="pcap-chh">${i+1}. ${esc(it.h)}</span><b>${it.vh||esc(it.v)}</b>${it.sub?`<span class="pcap-chs">${esc(it.sub)}</span>`:''}</li>`).join('');
}
/* each memory block is named by what it holds (the owner, 2026-10-09: “BF16权重只有4.2GB，这是LoRA的，不是模型的是吗？”) */
const HELD={policy:{en:'policy',zh:'策略模型'},ref:{en:'reference model',zh:'参考模型'},value:{en:'value model',zh:'价值模型'},rm:{en:'reward model',zh:'奖励模型'}};
function blkName(k,st,r){const two=st.meth==='ppo', mo=st.mode;
  if(k==='w') return mo==='full'?(two?tx('Policy and value model weights (BF16)','策略模型和价值模型的权重（BF16）'):tx('All the model’s weights (BF16)','模型的全部权重（BF16）'))
    :mo==='lora'?(two?tx('Frozen original weights, two copies (BF16)','冻住的原权重，两份（BF16）'):tx('Frozen original weights (BF16)','冻住的原权重（BF16）'))
    :(two?tx('Embedding tables and LM heads, two copies (BF16)','embedding 表和 LM head，两份（BF16）'):tx('Embedding table and LM head (BF16)','embedding 表和 LM head（BF16）'));
  if(k==='q4') return two?tx('Frozen linear layers, two copies (4-bit)','冻住的线性层，两份（4 位）'):tx('Frozen linear layers (4-bit)','冻住的线性层（4 位）');
  if(k==='g') return tx('Gradients (FP32)','梯度（FP32）');
  if(k==='o') return tx('Optimizer state (FP32 weight copy, Adam’s m and v)','优化器状态（FP32 权重副本、Adam 的 m 和 v）');
  if(k==='lr') return tx('LoRA matrices, with their gradients and optimizer state','LoRA 矩阵，连同梯度和优化器状态');
  if(k==='fz'){const f=r.held.filter(h=>h==='rm'||(h==='ref'&&mo==='full'));
    return f.length===2?tx('Reference and reward models (BF16)','参考模型和奖励模型（BF16）'):f[0]==='rm'?tx('Reward model (BF16)','奖励模型（BF16）'):tx('Reference model (BF16)','参考模型（BF16）');}
  return k==='a'?tx('Activations','激活'):'logits';}
function blkTip(k,st,r,v){const c=curCard(), m=MOD[c.m], here=tx(` On this GPU: ${GB(v)}.`,`这张卡上占 ${GB(v)}。`);
  const t={w:st.mode==='full'?tx('Full fine-tuning trains every weight, so the model keeps all its weights in BF16, 2 bytes each.','全量微调训练每个权重，模型的全部权重都存成 BF16，每个 2 字节。')
      :st.mode==='lora'?tx('LoRA leaves the original weights as they are, frozen in BF16, 2 bytes each.','LoRA 不改原来的权重，把它们冻住，仍按 BF16 存，每个 2 字节。')
      :tx(`QLoRA stores only the linear layers in 4 bits. The embedding table, the LM head and the few weights of the RMSNorms stay in BF16. They are ${fixd(PT[c.m].eo/m.N*100,1)}% of the parameters, so this block is small; the model’s weights are mostly in the 4-bit block.`,`QLoRA 只把线性层存成 4 位。embedding 表、LM head 和 RMSNorm 的少量权重仍是 BF16。它们只占参数的 ${fixd(PT[c.m].eo/m.N*100,1)}%，所以这一块很小，模型的权重大部分在 4 位那一块里。`),
    q4:tx('The frozen linear layers of every layer (the 4 attention matrices and the 3 MLP matrices) stored in 4 bits (NF4 with double quantization), about 0.516 bytes per weight. Most of the model’s weights are here.','每一层的线性层（注意力的 4 个矩阵和 MLP 的 3 个矩阵）冻住，存成 4 位（NF4 加双重量化），每个权重约 0.516 字节。模型的大部分权重都在这里。'),
    g:tx('One FP32 gradient, 4 bytes, for every trained weight.','每个要训练的权重有一个 FP32 梯度，4 字节。'),
    o:tx('Adam keeps an FP32 copy of every trained weight and its m and v: 12 bytes per weight.','Adam 给每个要训练的权重存一份 FP32 副本和它的 m、v，一共 12 字节。'),
    lr:tx(`The narrow matrices LoRA trains (rank 16 on all 7 matrices of each layer), with their gradients and Adam’s state: 16 bytes for each of their ${big(PT[c.m].lp)} parameters.`,`LoRA 要训练的窄矩阵（每层 7 个矩阵都加秩 16），连同梯度和 Adam 的状态，${big(PT[c.m].lp)} 个参数每个 16 字节。`),
    fz:tx('Frozen copies that are only read, never trained: 2 bytes per parameter.','只读、不训练的冻结副本，每个参数 2 字节。'),
    a:tx(`What the forward pass keeps for the backward pass, for the ${fint(SEQ)} tokens this GPU works on at once.`,`前向时留给反向用的中间结果，按这张卡一次处理 ${fint(SEQ)} 个 token 算。`),
    l:tx(`The scores over the 128,256-token vocabulary for each of the ${fint(SEQ)} tokens, in FP32.`,`${fint(SEQ)} 个 token 各自对词表里 128,256 个 token 的得分，FP32。`)}[k];
  return t+(A.lang()==='zh'?'':' ')+here.trim();}
function drawMem(){
  const st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), c=curCard(), scale=Math.max(CAP*1.1,r.mem);
  const heldTxt=r.held.filter(h=>!(h==='ref'&&st.mode!=='full')).map(h=>L(HELD[h])).join(tx(', ','、'));
  $('#pcap-mem-cap',app).innerHTML=tx(`Each GPU uses ${GB(r.mem)} of its 80 GB`,`每张卡 80 GB 里用 ${GB(r.mem)}`)+`<span class="pcap-capsub">${esc(tx(`Models in memory are the ${heldTxt}${c.gpus>1?`, split over ${c.gpus} GPUs`:''}${st.mode!=='full'&&r.held.includes('ref')?'. The reference is the base with the adapter off':''}`,`显存里放着${heldTxt}${c.gpus>1?`，平均分到 ${c.gpus} 张卡`:''}${st.mode!=='full'&&r.held.includes('ref')?'。参考模型就是关掉 LoRA 的原模型':''}`))}</span>`;
  let x=0; const segs=r.blocks.map(([k,v])=>{const w=v/scale*100, s=`<i class="pcap-mb ${k}" style="left:${x.toFixed(2)}%;width:${w.toFixed(2)}%"></i>`; x+=w; return s;}).join('');
  const over=r.mem>CAP?`<i class="pcap-mb over" style="left:${(CAP/scale*100).toFixed(2)}%;width:${((r.mem-CAP)/scale*100).toFixed(2)}%"></i>`:'';
  $('#pcap-mem',app).innerHTML=`<div class="pcap-track">${segs}${over}<span class="pcap-cap80" style="left:${(CAP/scale*100).toFixed(2)}%"></span></div>`;
  $('#pcap-mem-list',app).innerHTML=r.blocks.map(([k,v])=>`<li tabindex="0" data-k="${k}" data-tipx="${esc(blkTip(k,st,r,v))}"><i class="pcap-sw ${k}"></i><span>${esc(blkName(k,st,r))}</span><b>${GB(v)}</b></li>`).join('')+
    `<li tabindex="0" data-tipx="${esc(tx('The thick line marks the H100’s 80 GB.','粗线标出 H100 的 80 GB。'))}"><i class="pcap-sw line"></i><span>${tx('80 GB per H100','每张 H100 80 GB')}</span><b></b></li>`+
    (r.mem>CAP?`<li class="pcap-overl" tabindex="0" data-tipx="${esc(tx('The part above 80 GB does not fit.','超出 80 GB 的部分放不下。'))}"><i class="pcap-sw over"></i><span>${tx('Over 80 GB','超出 80 GB')}</span><b>${GB(r.mem-CAP)}</b></li>`:'');
}
const CSEG={train:{en:'Training (forward and backward)',zh:'训练（前向和反向）'},frozen:{en:'Reference and reward model reading',zh:'参考模型和奖励模型读回答'},gen:{en:'Generating answers',zh:'生成回答'},rm:{en:'Training the reward model',zh:'训练奖励模型'}};
function costRow(label,r,scale,me){
  const order=['train','frozen','gen','rm']; let segs='';
  order.forEach(k=>{const v=r.seg[k]/3600; if(v>0) segs+=`<i class="pcap-cs ${k}" style="width:${(v/scale*100).toFixed(2)}%" data-tipx="${esc(tx(`${L(CSEG[k])}: ${ghU(v)}.`,`${L(CSEG[k])}：${ghU(v)}。`))}" tabindex="0"></i>`;});
  return `<div class="pcap-crow${me?' me':''}"><span class="pcap-cl">${label}</span><div class="pcap-ctrack">${segs}</div><b class="pcap-cv">${r.can?ghU(r.gpuh):tx('cannot run','跑不起来')}</b></div>`;
}
/* what makes the GPU hours (the owner, 2026-10-09: “训练3遍才0.73GPU小时，这是真的吗”): the tokens of every pass, the work per token and
   the assumed speed, from plan(); with LoRA or QLoRA also the measured check (torchtune, about 40% of the peak, see TT) */
function howRows(c,r,st){
  const N=MOD[c.m].N, n=c.n*r.x, rows=[], tok=v=>tx(`${big(v)} tokens`,`${big(v)} 个 token`);
  /* one example's trained tokens: a pair of answers for DPO, two models learning it PPO_EP times for PPO, G answers for GRPO */
  const t=c.tok.p+c.tok.r, per=r.tok.train, Ga=Math.round(per/t), pf=fint(per);
  const perW={sft:tx(`each ${pf} tokens`,`每条 ${pf} 个`),dpo:tx(`each a pair of answers, 2 × ${fint(t)} = ${pf} tokens`,`每条 2 个回答，共 2 × ${fint(t)} = ${pf} 个`),
    ppo:tx(`the policy and the value model each learn every example ${PPO_EP} times, 2 × ${PPO_EP} × ${fint(t)} = ${pf} tokens`,`策略模型和价值模型每条各学 ${PPO_EP} 次，共 2 × ${PPO_EP} × ${fint(t)} = ${pf} 个`),
    grpo:tx(`each ${Ga} answers, ${Ga} × ${fint(t)} = ${pf} tokens`,`每条 ${Ga} 个回答，共 ${Ga} × ${fint(t)} = ${pf} 个`)}[st.meth];
  rows.push([tx('Tokens trained','训练的 token'),tx(`${tok(n*per)} (${fint(c.n)} examples per pass, ${perW}, ${passW(r.x)})`,`${tok(n*per)}（每遍 ${fint(c.n)} 条，${perW}，训练${passW(r.x)}）`)]);
  rows.push([tx('Work per trained token','训练一个 token 的计算'),tx(`about ${fixd(r.kT*N/1e9,0)} GFLOP (${r.kT} × ${big(N)} parameters${r.kT===4?', with no gradient for the frozen weights':''})`,`约 ${fixd(r.kT*N/1e9,0)} GFLOP（${r.kT} × ${big(N)} 个参数${r.kT===4?'，冻住的权重不算梯度':''}）`)]);
  if(r.tok.frozen) rows.push([tx('Read by frozen models','冻住的模型读的 token'),tx(`${tok(n*r.tok.frozen)}, about ${fixd(2*N/1e9,0)} GFLOP each (2 × ${big(N)})`,`${tok(n*r.tok.frozen)}，每个约 ${fixd(2*N/1e9,0)} GFLOP（2 × ${big(N)}）`)]);
  rows.push([tx('Speed','计算速度'),tx(`50% of the H100’s 989.5 TFLOP/s peak (assumed): about ${fint(Math.round(r.trainRate/10)*10)} trained tokens per second per GPU`,`按 H100 峰值 989.5 TFLOP/s 的 50% 算（假设），每张卡每秒约训练 ${fint(Math.round(r.trainRate/10)*10)} 个 token`)]);
  if(r.tok.gen) rows.push([tx('Generated','生成的 token'),tx(`${tok(n*r.tok.gen)}, about ${fint(Math.round(r.genRate/10)*10)} per second per GPU`,`${tok(n*r.tok.gen)}，每张卡每秒约生成 ${fint(Math.round(r.genRate/10)*10)} 个`)]);
  if(r.rmTok) rows.push([tx('Reward model','训练奖励模型'),tx(`trained once first: ${tok(r.rmTok)}`,`先训练一次，${tok(r.rmTok)}`)]);
  if(st.mode!=='full') rows.push([tx('Measured check','和实测比'),tx(`torchtune measured Llama 3.1 70B with LoRA at about ${fixd(ttU()*100,0)}% of the peak. At that speed this run would take about ${ghU(ttGpuh(r))}.`,`torchtune 实测，70B 用 LoRA 训练只到峰值的约 ${fixd(ttU()*100,0)}%。按这个速度，这次训练约要 ${ghU(ttGpuh(r))}。`)]);
  return rows.map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');}
function drawCost(v){
  const host=$('#pcap-cost',app), cap=$('#pcap-cost-cap',app), list=$('#pcap-cost-list',app), how=$('#pcap-cost-how',app), c=curCard();
  let over=false;
  if(v.cost==='p3'){
    const rs=plan(P3.sft), rg=plan(P3.grpo), scale=rg.gpuh*1.05;
    cap.innerHTML=tx('One pass over the 7,473 math problems, full fine-tuning on 16 H100s','把 7,473 道数学题过一遍，16 张 H100 做全量微调')+`<span class="pcap-capsub">${tx('GPU hours of the 16 GPUs together','GPU 小时按 16 张卡加起来算')}</span>`;
    host.innerHTML=costRow('SFT',rs,scale)+costRow('GRPO',rg,scale);
    how.innerHTML=''; how._h=''; how.parentElement.hidden=true;
  } else {
    const st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), scale=Math.max(c.budget,r.gpuh)*1.08;
    cap.innerHTML=(r.can?tx(`This run uses ${ghU(r.gpuh)} of a budget of ${ghU(c.budget)}`,`这次训练用 ${ghU(r.gpuh)}，预算 ${ghU(c.budget)}`):tx(`This method cannot run; the budget is ${ghU(c.budget)}`,`这个方法跑不起来，预算 ${ghU(c.budget)}`))+
      `<span class="pcap-capsub">${c.gpus>1?tx(`Both count all ${c.gpus} GPUs together${r.can?`, so the ${c.gpus} GPUs run for about ${wallW(r.gpuh,c.gpus)}`:''}.`,`两个数都是 ${c.gpus} 张卡加起来算的${r.can?`，${c.gpus} 张卡一起跑约 ${wallW(r.gpuh,c.gpus)}`:''}。`)
        :tx(`One GPU${r.can?`, so it runs for about ${wallW(r.gpuh,1)}`:''}.`,`只有 1 张卡${r.can?`，也就是这张卡跑约 ${wallW(r.gpuh,1)}`:''}。`)}</span>`;
    host.innerHTML=`<div class="pcap-cwrap">`+costRow(methName(st.meth),r,scale,true)+`<span class="pcap-budget" style="left:calc(var(--pcl) + (100% - var(--pcl) - var(--pcv)) * ${(c.budget/scale).toFixed(4)})" data-tipx="${esc(tx(`The budget: ${ghU(c.budget)}.`,`预算：${ghU(c.budget)}。`))}" tabindex="0"></span></div>`;
    over=r.can&&!r.inBudget;
    if(over) host.querySelector('.pcap-crow').classList.add('pcap-overb');
    const hh=r.can?howRows(c,r,st):''; if(how._h!==hh){how.innerHTML=hh; how._h=hh;} how.parentElement.hidden=!hh;
  }
  const used=new Set(); (v.cost==='p3'?[plan(P3.sft),plan(P3.grpo)]:[plan(mode==='challenge'?{card:'ch',...ch}:S)]).forEach(r=>Object.keys(r.seg).forEach(k=>{if(r.seg[k]>0) used.add(k);}));
  list.innerHTML=['train','frozen','gen','rm'].filter(k=>used.has(k)).map(k=>`<li tabindex="0" data-tipx="${esc(k==='gen'?TIP.gen():L(CSEG[k]))}"><i class="pcap-sw c-${k}"></i><span>${esc(L(CSEG[k]))}</span></li>`).join('')+
    (v.cost==='p3'?'':`<li tabindex="0" data-tipx="${esc(tx('The dashed line marks the budget.','虚线标出预算。'))}"><i class="pcap-sw dash"></i><span>${tx('budget','预算')}</span></li>`)+
    (over?`<li class="pcap-overl" tabindex="0" data-tipx="${esc(tx('A red ring marks a run over the budget.','红框标出超了预算的训练。'))}"><i class="pcap-sw overb"></i><span>${tx('over the budget','超预算')}</span></li>`:'');
}
const zoomOf=c=>[Math.max(0,c.target-10),Math.min(100,c.target+15)];
function drawMet(){
  const c=curCard(), st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), mn=metricName(c);
  $('#pcap-met-cap',app).textContent=r.flat?tx(`${mn.replace(/^./,x=>x.toUpperCase())}: ${pc(r.metric)}, the same as before training, target ${c.target}%`,`${mn}：${pc(r.metric)}，和训练前一样，目标 ${c.target}%`)
    :tx(`${mn.replace(/^./,x=>x.toUpperCase())}: ${pc(r.metric)}, target ${c.target}%`,`${mn}：${pc(r.metric)}，目标 ${c.target}%`);
  const lo=Math.max(0,r.metric-r.band), hi=Math.min(100,r.metric+r.band);
  $('#pcap-met',app).innerHTML=`<div class="pcap-mtrack"><i class="pcap-mfill${r.hits&&r.can?' ok':''}" style="width:${r.metric.toFixed(2)}%"></i><i class="pcap-mband" style="left:${lo.toFixed(2)}%;width:${(hi-lo).toFixed(2)}%" tabindex="0" data-tipx="${esc(TIP.band())}"></i><span class="pcap-m0" style="left:${c.m0}%" tabindex="0" data-tipx="${esc(tx(`Before training: ${c.m0}%.`,`训练前：${c.m0}%。`))}"></span><span class="pcap-mt" style="left:${c.target}%" tabindex="0" data-tipx="${esc(tx(`Target: ${c.target}%.`,`目标：${c.target}%。`))}"></span></div>`+
    `<div class="pcap-mscale"><span>0%</span><span>50%</span><span>100%</span></div>`+zoomHTML(c,r,lo,hi);
  const below=lo<c.target, [z0,z1]=zoomOf(c), zvis=hi>z0&&lo<z1, zdot=zvis&&r.metric>=z0&&r.metric<=z1;
  /* every mark of the two strips is named here (H6): the zoomed strip's dot is the score after training */
  $('#pcap-met-list',app).innerHTML=`<li><i class="pcap-sw mf${r.hits&&r.can?' ok':''}"></i><span>${tx('after training','训练后')}</span></li><li><i class="pcap-sw mb"></i><span>${tx('error bar (95%)','误差线（95%）')}</span></li><li><i class="pcap-sw m0"></i><span>${tx('before training','训练前')}</span></li><li><i class="pcap-sw dash"></i><span>${tx('target','目标')}</span></li>`+
    (zdot?`<li><i class="pcap-sw zdot"></i><span>${tx('the score after training, in the zoomed strip','放大图里的点：训练后的分数')}</span></li>`:'')+
    (below&&zvis?`<li class="pcap-zbadl"><i class="pcap-sw zbad"></i><span>${tx('the part of the error bar below the target','误差线低于目标的部分')}</span></li>`:'');
}
function zoomHTML(c,r,lo,hi){
  const [z0,z1]=zoomOf(c), zx=v=>Math.max(0,Math.min(100,(v-z0)/(z1-z0)*100)), below=lo<c.target, dot=r.metric>=z0&&r.metric<=z1, vis=hi>z0&&lo<z1, T=c.target;
  /* three states (R6.10): the whole bar below the target, only its bottom below, the whole bar above; no zoomed strip when the bar lies outside it */
  const where=hi<T?tx(`all of it is below the ${T}% target`,`整条都低于目标 ${T}%`):below?tx(`its bottom is below the ${T}% target`,`下端低于目标 ${T}%`):tx(`all of it is above the ${T}% target`,`整条都在目标 ${T}% 以上`);
  const cap=(vis?tx(`Zoomed in on ${z0}% to ${z1}%. `,`放大看 ${z0}% 到 ${z1}%。`):'')+tx(`The error bar runs from ${pc(lo)} to ${pc(hi)}; ${where}.`,`误差线从 ${pc(lo)} 到 ${pc(hi)}，${where}。`);
  if(!vis) return `<div class="pcap-zoom"><p class="pcap-zcap${below?' pcap-zbadt':''}">${cap}</p></div>`;
  return `<div class="pcap-zoom"><div class="pcap-ztrack">`+
    `<i class="pcap-zband" style="left:${zx(lo).toFixed(2)}%;width:${(zx(hi)-zx(lo)).toFixed(2)}%"></i>`+
    (below?`<i class="pcap-zbad" style="left:${zx(lo).toFixed(2)}%;width:${(zx(Math.min(c.target,hi))-zx(lo)).toFixed(2)}%"></i>`:'')+
    (dot?`<i class="pcap-zdot" style="left:${zx(r.metric).toFixed(2)}%"></i>`:'')+
    `<span class="pcap-mt" style="left:${zx(c.target).toFixed(2)}%"></span></div>`+
    `<div class="pcap-mscale"><span>${z0}%</span><span>${z1}%</span></div><p class="pcap-zcap${below?' pcap-zbadt':''}">${cap}</p></div>`;}
function statusOf(){
  if(predHidden()) return {cls:'neutral',txt:tx('Predict first','先预测'),tip:tx('The GPU hours stay hidden until you answer the question in this part.','在这一部分作答之前，GPU 小时先隐藏。')};
  const v=secs(), st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st);
  if(isP('mission')) return {cls:'neutral',txt:tx('Task card','任务卡'),tip:tx('Three tasks. The next part picks a method for each.','这里有三个任务。下一部分给每个任务选方法。')};
  if(!r.can) return {cls:'bad',txt:tx('Cannot run','跑不起来'),tip:tx(`The method needs ${dataName(r.miss)}.`,`这个方法需要${dataName(r.miss)}。`)};
  if(v.cost==='p3') return {cls:'neutral',txt:tx('Cost compared','比较花费'),tip:tx('Both bars are one pass over the math problems.','两根条都是把数学题过一遍。')};
  if(!v.mem&&!v.cost&&!v.met) return {cls:'ok',txt:tx('Can run','能跑'),tip:tx('The data allows this method.','这套数据可以用这个方法。')};
  if(!r.fits) return {cls:'bad',txt:tx('Out of memory','显存不够'),tip:tx(`Each GPU would need ${GB(r.mem)}, more than its 80 GB.`,`每张卡要 ${GB(r.mem)}，超过了 80 GB。`)};
  if(v.cost&&!r.inBudget) return {cls:'bad',txt:tx('Over budget','超预算'),tip:tx(`${ghU(r.gpuh)} against a budget of ${ghU(curCard().budget)}.`,`${ghU(r.gpuh)}，预算是 ${ghU(curCard().budget)}。`)};
  if(v.met&&!r.hits) return {cls:mode==='challenge'?'bad':'neutral',txt:tx('Below target','没到目标'),tip:r.flat?flatWhy(curCard()):tx(`${pc(r.metric)} against a target of ${curCard().target}%.`,`${pc(r.metric)}，目标 ${curCard().target}%。`)};
  if(isP('memory')||(!v.cost&&!v.met)) return {cls:'ok',txt:tx('Fits in 80 GB','放得下'),tip:tx(`Each GPU uses ${GB(r.mem)} of its 80 GB.`,`每张卡用了 80 GB 里的 ${GB(r.mem)}。`)};
  return {cls:'ok',txt:v.met?tx('Target met','达标'):tx('Within budget','预算够'),tip:tx(`${GB(r.mem)} per GPU, ${ghU(r.gpuh)}, ${pc(r.metric)}.`,`每张卡 ${GB(r.mem)}，${ghU(r.gpuh)}，${pc(r.metric)}。`)};
}
function drawHeadline(){
  const t=$('#pcap-st-title',app), sub=$('#pcap-st-sub',app), st=$('#pcap-status',app), c=curCard();
  t.textContent=mode==='challenge'?tx('Your setup: math accuracy on one H100','你的设置：一张 H100 提高数学题的答对率'):tx(`Task: ${L(CARDW[c.id].t)}`,`任务：${L(CARDW[c.id].t)}`);
  const s0=mode==='challenge'?{card:'ch',...ch}:S;
  sub.textContent=isP('mission')||isP('data')?tx(`${cardModel(c)}, ${gpuW(c)}`,`${cardModel(c)}，${gpuW(c)}`):tx(`${methName(s0.meth)}, ${modeName(s0.mode)}, ${passW(s0.x)}`,`${methName(s0.meth)}，${modeName(s0.mode)}，训练${passW(s0.x)}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), c=curCard(), hid=predHidden(), v=secs();
  const can={k:'can',label:tx('Can it run?','能不能跑'),v:r.can?tx('yes','能'):tx('no','不能'),sub:r.can?exOf(c,st.meth,r):tx(`no ${dataName(r.miss)}`,`缺${dataName(r.miss)}`),tip:r.can?canTip(c,st.meth):tx('Whether the card’s data can make this method’s training examples.','任务卡上的数据能不能做成这个方法要的训练样本。')};
  const mem={k:'mem',label:tx('Memory per GPU','每张卡的显存'),v:GB(r.mem),sub:r.fits?tx('fits in 80 GB','放得下 80 GB'):tx('over 80 GB','超过 80 GB'),tip:tx(`All the models’ bytes are split over ${c.gpus>1?`the ${c.gpus} GPUs`:'the one GPU'}, and each GPU adds one sequence’s activations and logits: ${GB(r.mem)} in all.`,`所有模型的字节平均分到 ${c.gpus} 张卡上，每张卡再加一条序列的激活和 logits，一共 ${GB(r.mem)}。`)};
  const gpuh={k:'gpuh',label:tx('GPU hours','GPU 小时'),v:hid?'—':(r.can?gh(r.gpuh):'—'),sub:hid?tx('shown after you answer','作答后揭晓'):tx(`budget ${gh(c.budget)}`,`预算 ${gh(c.budget)}`),tip:hid?tx('Hidden until you answer the question.','作答之前先隐藏。'):TIP.gpuh()};
  const met={k:'met',label:metricName(c),v:pc(r.metric),sub:tx(`target ${c.target}%`,`目标 ${c.target}%`),tip:tx(`By this lab’s rule; the error bar is about ±${fixd(r.band,1)} points.`,`按这个 Lab 的规则算出。误差线约 ±${fixd(r.band,1)} 个百分点。`)};
  if(isP('mission')) return [{k:'bud',fixed:true,label:tx('Budget','预算'),v:gh(c.budget),sub:tx('GPU hours','GPU 小时'),tip:TIP.gpuh()},{k:'tgt',fixed:true,label:tx('Target','目标'),v:`${c.target}%`,sub:metricName(c),tip:tx(`The ${metricName(c)} must reach at least ${c.target}%.`,`${metricName(c)}至少要到 ${c.target}%。`)}];
  if(isP('data')) return [can];
  if(isP('predict')) return [can,gpuh];
  if(isP('memory')) return [can,mem];
  return [can,mem,gpuh,met];
}
let metCard=null;
function drawMetrics(){
  const host=$('#pcap-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join(), cid=curCard().id;
  /* a change badge compares two runs of the same task card only: a new card clears it (its metric and budget differ) */
  if(metCard!==cid){metCard=cid; Object.keys(shownPrev).forEach(k=>{shownPrev[k]=null;});}
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!m.fixed&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v)),pts:/%$/.test(m.v)})||'';}
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
function mathHTML(){
  if(predHidden()) return `<p>${tx('The formulas appear once you have answered the question in this part.','在这一部分作答以后，这里给出公式。')}</p>`;
  if(isP('mission')||isP('data')) return `<p>${tx('This part looks only at the data and the method, so there is nothing to compute yet. From part 3 on, the formulas for memory, GPU hours and the metric appear here.','这一部分只看数据和方法，还没有要算的数。从第 3 部分起，这里给出显存、GPU 小时和指标的算式。')}</p>`;
  const st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), c=curCard(), m=MOD[c.m], rows=[], v=secs(), N=m.N;
  const G2=b=>`${fixd(b/1e9,2)}\\,\\text{GB}`;
  rows.push(`<div class="pcap-mblk"><p class="pcap-ml">${tx('Memory per GPU. Every model’s bytes are split evenly over the GPUs, and each GPU also holds the activations and logits of the tokens it works on.','每张卡的显存。所有模型的字节平均分到每张卡上，每张卡还要放下它正在算的 token 的激活和 logits。')}</p>`+
    EQ(tx('\\text{memory per GPU}','\\text{每张卡的显存}'),[tx('\\frac{\\text{all model bytes}}{\\text{GPUs}} + \\text{activations} + \\text{logits}','\\frac{\\text{所有模型的字节}}{\\text{卡数}} + \\text{激活} + \\text{logits}'),
      `\\frac{${G2(r.modelBytes)}}{${c.gpus}} + ${G2(r.per.a)} + ${G2(r.per.l)}`],GB(r.mem))+
    `<p class="small muted">${tx(`Activations: 1,024 tokens × ${m.L} layers × ${fint(PT[c.m].aL)} bytes. Logits: 1,024 tokens × 128,256 (the vocabulary) × 4 bytes.`,`激活是 1,024 个 token × ${m.L} 层 × ${fint(PT[c.m].aL)} 字节。logits 是 1,024 个 token × 128,256（词表大小）× 4 字节。`)}</p></div>`);
  if(r.can){
    const rm=r.seg.rm>0, tsum=`${fixd(r.perEx.train,4)} + ${fixd(r.perEx.frozen,4)} + ${fixd(r.perEx.gen,4)}`, pe=r.perEx.train+r.perEx.frozen+r.perEx.gen;
    /* two short chains that fit a phone (H2): one example's GPU-seconds, then the GPU hours; the reward model's one-off training on a “+” line */
    rows.push(`<div class="pcap-mblk"><p class="pcap-ml">${tx('GPU hours. One example costs three things, in GPU-seconds: training on it, the frozen models reading it, and generating answers.','GPU 小时。每条数据有三项花费，单位是 GPU 秒：训练它、冻住的模型读它、生成回答。')}</p>`+
      EQ(tx('\\text{per example}','\\text{每条}'),[tx('\\text{training} + \\text{reading} + \\text{generating}','\\text{训练} + \\text{冻住的模型读} + \\text{生成回答}'),tsum],tx(`${fixd(pe,4)} GPU-seconds`,`${fixd(pe,4)} GPU 秒`))+
      `<p class="small">${tx('Times the examples and the passes, that is the GPU-seconds of all GPUs together; 3,600 of them make a GPU hour.','乘以条数和遍数，就是所有卡加起来的 GPU 秒。3,600 GPU 秒是 1 GPU 小时。')}${rm?tx(' The reward model is trained once, before the passes.','奖励模型只在训练前训练一次，单独加上。'):''}</p>`+
      EQ(tx('\\text{GPU hours}','\\text{GPU 小时}'),[tx('\\frac{\\text{per example} \\times \\text{examples} \\times \\text{passes}}{3600}','\\frac{\\text{每条} \\times \\text{条数} \\times \\text{遍数}}{3600}'),...(rm?[tx('+ \\text{reward model}','+ \\text{训练奖励模型}')]:[]),
        `\\frac{${fixd(pe,4)} \\times ${fint(c.n)} \\times ${r.x}}{3600}`,...(rm?[`+ ${gh(r.seg.rm/3600)}`]:[])],ghU(r.gpuh))+`</div>`);
    rows.push(`<div class="pcap-mblk"><p class="pcap-ml">${tx(`Training one example. Its trained tokens times the work per token, at the assumed speed of the GPU. ${r.kT===6?'Full fine-tuning costs 6N per token':'LoRA and QLoRA cost about 4N per token: no gradient for the frozen weights'}, where N is the number of parameters.`,`训练一条数据的时间。要训练的 token 数乘以每个 token 的计算量，再除以假设的计算速度。${r.kT===6?'全量微调每个 token 算 6N':'LoRA 和 QLoRA 不算冻住权重的梯度，每个 token 约算 4N'}，N 是模型的参数量。`)}</p>`+
      EQ(tx('\\text{training one example}','\\text{训练一条}'),[tx(`\\frac{\\text{trained tokens} \\times ${r.kT}N}{50\\% \\times 989.5\\,\\text{TFLOP/s}}`,`\\frac{\\text{训练的 token} \\times ${r.kT}N}{50\\% \\times 989.5\\,\\text{TFLOP/s}}`),
        `\\frac{${fint(r.tok.train)} \\times ${r.kT} \\times ${fixd(N/1e9,2)}\\,\\text{B}}{50\\% \\times 989.5\\,\\text{TFLOP/s}}`],tx(`${fixd(r.perEx.train,4)} GPU-seconds`,`${fixd(r.perEx.train,4)} GPU 秒`))+
      /* only what this run has: frozen reading, generating, and the measured check for LoRA and QLoRA (torchtune measured LoRA) */
      ((n=>n?`<p class="small muted">${n}</p>`:'')([r.tok.frozen?tx('A frozen model reading a token costs 2N.','冻住的模型读一个 token 算 2N。'):'',
        r.tok.gen?tx(`Generating runs at about ${fint(Math.round(r.genRate))} tokens per second per GPU.`,`生成回答按每张卡每秒约 ${fint(Math.round(r.genRate))} 个 token 算。`):'',
        r.kT===4?tx(`torchtune measured Llama 3.1 70B with LoRA at about ${fixd(ttU()*100,0)}% of the peak instead of 50%; at that speed this run would take about ${ghU(ttGpuh(r))}.`,`torchtune 实测，70B 用 LoRA 训练只到峰值的约 ${fixd(ttU()*100,0)}%，不到 50%。按这个速度，这次训练约要 ${ghU(ttGpuh(r))}。`):'']
        .filter(Boolean).join(A.lang()==='zh'?'':' ')))+`</div>`);
    /* the metric only where the stage shows it (SF-2): the result chip */
    const showMet=mode!=='guided'||v.met||isP('wrong');
    if(r.rule&&showMet){
      const p=r.metric/100;
      rows.push(`<div class="pcap-mblk"><p class="pcap-ml">${tx('The metric after x passes, by this lab’s rule. It climbs from where it started toward a ceiling, and each h passes close half of the gap that is left.','训练 x 遍以后的指标，按本 Lab 的规则算。它从训练前的值朝一个上限爬，每训练 h 遍，就补上剩下差距的一半。')}</p>`+
        EQ('m(x)',['m_\\infty - (m_\\infty - m_0) \\times 2^{-x/h}',`${r.rule.mi} - (${r.rule.mi} - ${c.m0}) \\times 2^{-${r.x}/${r.rule.h}}`],pc(r.metric))+
        `<p class="small muted">${tx(`${M('m_0')} is the value before training, ${M('m_\\infty')} the ceiling, and x the passes.`,`${M('m_0')} 是训练前的值，${M('m_\\infty')} 是上限，x 是训练的遍数。`)}</p></div>`);
      rows.push(`<div class="pcap-mblk"><p class="pcap-ml">${tx('The error bar. The check uses only a few hundred examples, so the measured score could land elsewhere in this interval 95% of the time.','误差线。验收只用几百个样本，量出的分数有 95% 的机会落在这个区间里。')}</p>`+
        EQ(tx('\\text{error bar}','\\text{误差线}'),['± 1.96 \\times \\sqrt{\\frac{p(1 - p)}{n}}',`± 1.96 \\times \\sqrt{\\frac{${fixd(p,4)} \\times ${fixd(1-p,4)}}{${fint(c.nEval)}}}`],tx(`±${fixd(r.band,1)} points`,`±${fixd(r.band,1)} 个百分点`))+
        `<p class="small muted">${tx(`p is the score and n the number of examples the check uses (${fint(c.nEval)}).`,`p 是分数，n 是验收用的样本数（${fint(c.nEval)}）。`)}</p></div>`);}
  }
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const v=secs();
  drawHeadline(); drawCard(); drawStrip();
  $('#pcap-membox',app).hidden=!v.mem; $('#pcap-costbox',app).hidden=!v.cost||predHidden(); $('#pcap-metbox',app).hidden=!v.met;
  if(v.mem) drawMem();
  if(v.cost&&!predHidden()) drawCost(v);
  if(v.met) drawMet();
  drawMetrics();
  const mb=$('#pcap-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the three choices in the task card. Everything else is fixed: Llama‑3.1‑8B, the 7,473 problems with their final answers only, one H100, a budget of 3 GPU hours.','挑战中只能改任务卡里的三项。其余设置固定：Llama‑3.1‑8B，只拿到最后答案的 7,473 道题，一张 H100，预算 3 GPU 小时。');
  if(mode!=='guided') return '';
  return tx('The settings are fixed in this part; the choice it asks for is in the card above. You can change every setting in Scenarios.','这一部分的设置是固定的，要你选的在上面的卡片里。到了应用场景，所有设置都可以改。');
}
function syncControls(){
  const set=(key,val)=>$$(`#pcap-controls .seg-ctl[data-key="${key}"]`,app).forEach(g=>A.ui.setPressed(g,String(val)));
  ['card','meth','mode','x'].forEach(k=>set(k,S[k]));
  ['meth','mode','x'].forEach(k=>$$(`#pcap-rail .seg-ctl[data-key="ch${k}"]`,app).forEach(g=>A.ui.setPressed(g,String(ch[k]))));
  const g=mode==='guided', c=mode==='challenge';
  $('#pcap-controls',app).classList.toggle('locked',c||g);
  $$('#pcap-controls .lockable',app).forEach(e=>{e.hidden=g||c;});
  const ln=$('#pcap-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||predHidden()) return; const st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st);
  A.announce(r.can?tx(`${GB(r.mem)} per GPU, ${ghU(r.gpuh)}, ${pc(r.metric)}.`,`每张卡 ${GB(r.mem)}，${ghU(r.gpuh)}，${pc(r.metric)}。`):tx('This method cannot run with this data.','这套数据用不了这个方法。'));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p); render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function phoneHTML(){
  if(mode==='challenge') return `<p class="small pcap-ph">${tx('Task: one H100, 3 GPU hours, math accuracy from 39.7% to 65%.','任务：一张 H100、3 GPU 小时，把答对率从 39.7% 提到 65%。')}</p><button type="button" class="btn ghost sm" data-act="to-task" data-tipx="${esc(tx('Scrolls down to the three choices in the task card.','向下滚到任务卡里的三项选择。'))}">${tx('Choose below ↓','去下面选 ↓')}</button>`;
  if(mode==='sandbox') return `<p class="small pcap-ph">${tx('Five situations: pick one to see its memory, GPU hours and metric.','五种情况：选一个，看它的显存、GPU 小时和指标。')}</p><button type="button" class="btn ghost sm" data-act="to-task" data-tipx="${esc(tx('Scrolls down to the list of scenarios.','向下滚到场景列表。'))}">${tx('Pick a scenario below ↓','去下面选一个场景 ↓')}</button>`;
  const id=STEPS[step].id;
  if(id==='predict'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the four answers in the card.','向下滚到卡片里的四个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(['mission','memory','metric','wrong'].includes(id)) return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the options in the card.','向下滚到卡片里的选项。'))}">${tx('Pick one below ↓','在下面选一个 ↓')}</button>`;
  if(id==='data') return `<button type="button" class="btn ghost sm" data-act="to-seg" data-tipx="${esc(tx('Scrolls down to the task and method in the card.','向下滚到卡片里的任务和方法。'))}">${tx('Pick a task and method below ↓','在下面选任务和方法 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#pcap-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#pcap-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#pcap-rail .step.current',app)||$('#pcap-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* keepStage as labs/llm-judge.js (R9.7): stop when the learner has scrolled the current card away */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#pcap-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#pcap-rail .step.current',app)||$('#pcap-rail .railcard',app)||$('#pcap-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#pcap-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#pcap-insight',app)); else revealBox($('#pcap-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; ch={...CH.start};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#pcap-stage',app); if(st) st.scrollTop=0;
  Object.keys(shownPrev).forEach(k=>{shownPrev[k]=null;});
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[CARD_IDS.indexOf(s.card),METHS.indexOf(s.meth),MODES.indexOf(s.mode),PASSES.indexOf(s.x)].join('_');
function parse(tok){
  const q=String(tok||'').split('_').map(Number); if(q.length!==4||q.some(x=>!Number.isInteger(x))) return null;
  const [ci,mi,oi,xi]=q; if(ci<0||ci>=CARD_IDS.length||mi<0||mi>=METHS.length||oi<0||oi>=MODES.length||xi<0||xi>=PASSES.length) return null;
  return {card:CARD_IDS[ci],meth:METHS[mi],mode:MODES[oi],x:PASSES[xi]};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+3), pk=a=>a[Math.floor(r()*a.length)];
  const meth=pk(['dpo','grpo']), mode=pk(['full','lora']), gpus=pk([1,2,4,8]), x=pk([1,2]);
  const card=meth==='dpo'?'pref':'math', c={...CARDS[card]}, saveG=c.gpus;
  CARDS[card].gpus=gpus; const p=plan({card,meth,mode,x}); CARDS[card].gpus=saveG;
  return {seed,meth,mode,gpus,x,card,mem:p.mem,gpuh:p.gpuh,fits:p.fits,modelBytes:p.modelBytes,act:p.per.a+p.per.l};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#pcap-drill-q',app), a=$('#pcap-drill-a',app);
  const what=d.meth==='dpo'?tx('DPO on 20,000 preference pairs (a 128-token prompt and two 256-token answers each)','用 DPO 训练 20,000 对偏好（每对一个 128 个 token 的提问、两个 256 个 token 的回答）'):tx('GRPO with a checker on 7,473 math problems (each problem 128 tokens, with 8 generated answers of 256 tokens)','用 GRPO 和判分程序训练 7,473 道数学题（每道题 128 个 token，生成 8 个回答，每个 256 个 token）');
  qd.innerHTML=tx(`You run ${what} on Llama‑3.1‑8B with <b>${modeName(d.mode)}</b>, <b>${d.gpus}</b> H100${d.gpus>1?'s':''}, <b>${passW(d.x)}</b>. Does it fit in 80 GB per GPU, and about how many GPU hours does it take?`,
                  `在 Llama‑3.1‑8B 上${what}，用${modeName(d.mode)}，<b>${d.gpus}</b> 张 H100，训练${passW(d.x)}。每张卡 80 GB 放得下吗？大约要多少 GPU 小时？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>${GB(d.mem)} per GPU, ${d.fits?'it fits':'it does not fit'}. About ${ghU(d.gpuh)} in all.</b></p><p>Memory: the models’ ${fixd(d.modelBytes/1e9,1)} GB ÷ ${d.gpus} + ${fixd(d.act/1e9,2)} GB of activations and logits. GPU hours: the work of training and of the reference model’s reading, done at 50% of 989.5 TFLOP/s; N is the 8B parameters${d.meth==='grpo'?'. Generating the answers is added on top, at about 13,350 tokens per second per GPU':''}.</p>`+
    EQL('\\text{GPU hours}',['= \\frac{\\text{work}}{50\\% \\times 989.5\\,\\text{TFLOP/s} \\times 3600}',...(d.meth==='grpo'?['+ \\text{generating time}']:[])])+
    EQL('\\text{work}',[`= \\text{trained tokens} \\times ${d.mode==='full'?6:4}N`,'+ \\text{tokens the reference reads} \\times 2N']),
    `<p><b>每张卡 ${GB(d.mem)}，${d.fits?'放得下':'放不下'}。一共约 ${ghU(d.gpuh)}。</b></p><p>显存：模型一共 ${fixd(d.modelBytes/1e9,1)} GB ÷ ${d.gpus} 张卡，加上 ${fixd(d.act/1e9,2)} GB 的激活和 logits。GPU 小时算的是训练和参考模型读回答的计算量，计算速度按 989.5 TFLOP/s 的 50% 算，N 是 8B 的参数量${d.meth==='grpo'?'。生成回答的时间另外加上，每张卡每秒约生成 13,350 个 token':''}。</p>`+
    EQL('\\text{GPU 小时}',['= \\frac{\\text{计算量}}{50\\% \\times 989.5\\,\\text{TFLOP/s} \\times 3600}',...(d.meth==='grpo'?['+ \\text{生成回答的时间}']:[])])+
    EQL('\\text{计算量}',[`= \\text{训练的 token} \\times ${d.mode==='full'?6:4}N`,'+ \\text{参考模型读的 token} \\times 2N']));
}

/* =====================================================================
   jumps
   ===================================================================== */
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='connect') return tx('Click to jump to “Take it further”, where Tülu 3’s pipeline is quoted.','点击跳到“继续深入”，那里引了 Tülu 3 的流程。');
  return '';
}
function jump(k){
  if(!app) return;
  if(k==='controls'){const c=$('#pcap-controls',app); A.tip.hide(); A.ui.jumpToEl(c);}
  if(k==='connect'){const c=$('.connect',app); A.tip.hide(); if(c) c.scrollIntoView({behavior:A.reduceMotion()?'auto':'smooth',block:'start'});}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...PRESETS[0].s}; render();}); return;}
  if(a==='submit'){chSubmitted=true; const e=chMe(); if(e.ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#pcap-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#pcap-rail .pcap-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-seg'){const q=$('#pcap-ic-card',app)||$('#pcap-rail .seg-ctl',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-task'){const q=$('#pcap-rail .railcard',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#pcap-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#pcap-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function setLever(id,v){
  if(id==='pcap-ic-lv1'){view1=v; S={...S,card:v}; render(); return;}
  if(id==='pcap-ic-lv4'){lv4=v; tried4.add(v); setState({...lvOf(LV4,v).s}); return;}
  if(id==='pcap-ic-lv5'){lv5=+v; tried5.add(+v); setState({card:'fmt',meth:'sft',mode:'lora',x:+v}); return;}
  if(id==='pcap-ic-lv6'){lv6=v; tried6.add(v); setState({...lvOf(LV6,v).s});}
}
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#pcap-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#pcap-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); revealSoon(); return;}
    if(d.lever!=null){const lv=b.closest('.pcap-lv'); if(!lv) return; const id=lv.id;
      const f=()=>byLearner(()=>setLever(id,d.lever));
      if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(/^ch/.test(k)){const kk=k.slice(2); byLearner(()=>{ch={...ch,[kk]:kk==='x'?+v:v}; render();}); return;}
      if(seg.closest('#pcap-rail')&&(k==='card'||k==='meth')){byLearner(()=>{tried2.add(`${k==='card'?v:S.card}|${k==='meth'?v:S.meth}`); setState({[k]:v});}); revealSoon(); return;}
      if(['card','meth','mode','x'].includes(k)) bySettings(b,()=>setState({[k]:k==='x'?+v:v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#pcap-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#pcap-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the task looked at and the
   options tried, the challenge's three choices and result, the drill; the plans are recomputed from these */
const rsObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), rsCfg=x=>rsObj(x)?parse(token(x)):null;
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:{card:S.card,meth:S.meth,mode:S.mode,x:S.x},pred,observed:{...observed},saved:saved?{card:saved.card,meth:saved.meth,mode:saved.mode,x:saved.x}:null,
    view1,tried2:[...tried2],lv4,tried4:[...tried4],lv5,tried5:[...tried5],lv6,tried6:[...tried6],ch:{...ch},chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&chMe().ok};},
  restore(o){if(!rsObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[], inV=(L,v)=>L.some(x=>x.v===v);
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...(rsCfg(o.S)||{card:'fmt',meth:'sft',mode:'lora',x:3})}; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(rsObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    view1=CARD_IDS.includes(o.view1)?o.view1:'fmt';
    tried2=new Set(arr(o.tried2,v=>typeof v==='string'&&v.split('|').length===2&&CARD_IDS.includes(v.split('|')[0])&&METHS.includes(v.split('|')[1])));
    lv4=inV(LV4,o.lv4)?o.lv4:'8full'; tried4=new Set(['8full',...arr(o.tried4,v=>inV(LV4,v)),lv4]);
    lv5=PASSES.includes(o.lv5)?o.lv5:1; tried5=new Set([1,...arr(o.tried5,v=>PASSES.includes(v)),lv5]);
    lv6=inV(LV6,o.lv6)?o.lv6:'dpo'; tried6=new Set(['dpo',...arr(o.tried6,v=>inV(LV6,v)),lv6]);
    const c=rsObj(o.ch)?o.ch:{}; ch={meth:METHS.includes(c.meth)?c.meth:CH.start.meth,mode:MODES.includes(c.mode)?c.mode:CH.start.mode,x:PASSES.includes(c.x)?c.x:CH.start.x};
    saved=mode==='challenge'?rsCfg(o.saved):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...PRESETS[0].s};
    el.innerHTML=viewHTML(); app=$('#pcap-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#pcap-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#pcap-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#pcap-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#pcap-graph',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#pcap-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.txt;
      if(predHidden()){vis.innerHTML=''; read.innerHTML=esc(tx('GRPO or SFT on 7,473 problems: answer first','7,473 道题，GRPO 和 SFT：先作答')); return;}
      const st=mode==='challenge'?{card:'ch',...ch}:S, r=plan(st), c=curCard();
      if(isP('mission')){vis.innerHTML=''; read.innerHTML=esc(tx(`${L(CARDW[c.id].t)}: ${gpuW(c)}, budget ${ghU(c.budget)}, target ${c.target}%`,`${L(CARDW[c.id].t)}：${gpuW(c)}，预算 ${ghU(c.budget)}，目标 ${c.target}%`)); return;}
      /* polish r4 (reader): part 2 judges the data only, as its stage does; memory and GPU hours wait for parts 3 and 4 */
      if(isP('data')){vis.innerHTML=''; read.innerHTML=esc(r.can?tx(`${methName(st.meth)}: ${exOf(c,st.meth,r)}`,`${methName(st.meth)}：${exOf(c,st.meth,r)}`):tx(`${methName(st.meth)}: no ${dataName(r.miss)}`,`${methName(st.meth)}：缺${dataName(r.miss)}`)); return;}
      vis.innerHTML=`<div class="pcap-dk${r.fits?'':' bad'}"><i style="width:${Math.min(100,r.mem/CAP*100).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(!r.can?tx(`${methName(st.meth)} cannot run: no ${dataName(r.miss)}`,`${methName(st.meth)} 跑不起来：缺${dataName(r.miss)}`)
        :(mode!=='guided'||secs().met||isP('wrong'))?tx(`${GB(r.mem)} per GPU, ${ghU(r.gpuh)}, ${metricName(c)} ${pc(r.metric)} (target ${c.target}%)`,`每张卡 ${GB(r.mem)}，${ghU(r.gpuh)}，${metricName(c)} ${pc(r.metric)}（目标 ${c.target}%）`)
        :tx(`${GB(r.mem)} per GPU, ${ghU(r.gpuh)}`,`每张卡 ${GB(r.mem)}，${ghU(r.gpuh)}`));} /* the metric only where the stage shows it */
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__pcap=Object.freeze({
  plan:s=>deep(plan(s)), p3Ratio, p3NoRef, genRate:mk=>genRate(MOD[mk]), kvGen, stepMs, ttU, ttGpuh:s=>ttGpuh(plan(s)),
  chAll:()=>chAll().map(e=>deep({meth:e.meth,mode:e.mode,x:e.x,ok:e.ok,can:e.r.can,fits:e.r.fits,mem:e.r.mem,gpuh:e.r.gpuh,metric:e.r.metric,min:chMinimal(e)})),
  drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({PEAK,U0,HBM,CAP,VOC,SEQ,GEN_B,G_ANS,PPO_EP,RANK,NF4,PT:deep(PT),CARDS:deep(CARDS),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),LV4:deep(LV4.map(o=>({v:o.v,...o.s}))),LV6:deep(LV6.map(o=>({v:o.v,...o.s})))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},view1,lv4,lv5,lv6,tried2:[...tried2],tried4:[...tried4],tried5:[...tried5],tried6:[...tried6],ch:{...ch},chSubmitted,chReveal,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transferQ:e.transfer.map(q=>q.q),transferO:e.transfer.map(q=>q.o),transferWhy:e.transfer.map(q=>q.why)});}
});
})();
