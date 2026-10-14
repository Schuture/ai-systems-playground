/* training-memory (world training): An 8B model's weights take 16 GB; why does training it not fit on an 80 GB GPU?
   Fidelity: Analytical. Every byte comes from the model's config and a stated accounting (specs/training-memory.md):
   - model states, bytes per parameter (mixed precision): BF16 weights 2, FP32 gradients 4, FP32 copy 4, FP32 m 4, FP32 v 4 = 18 (AdamW;
     Megatron-LM's estimator and Hugging Face's guide); 8-bit AdamW keeps m and v in 1 byte each (12); SGD with momentum keeps only m (14);
   - activations, bytes per token per layer, BF16, fused RMSNorm, FlashAttention, no dropout, tensor by tensor:
     attention block: norm input d, norm output d, q (n_q·d_h), k and v (n_kv·d_h each), attention output (n_q·d_h);
     MLP block: norm input d, norm output d, gate f, SiLU(gate) f, up f, SiLU(gate)·up f;  a = 2 × (sum)  (Llama-3.1-8B: 167,936 = 41·d);
   - logits: tokens × vocabulary × 4 bytes (one FP32 copy);
   - full recomputation keeps only each layer's input (2·d bytes per token per layer) and redoes the layer's forward in the backward pass;
   - peak = model states + kept activations + logits, at the end of the forward pass (the curve's other moments are lower);
   - training compute per token: 6N without recomputation, 8N with it (one more forward pass, 2N).
   GB = 10^9 bytes everywhere; a GPU's "80 GB" is read as 80 × 10^9 bytes (read as GiB, no case on the page and not the challenge's best plan changes).
   The challenge's MFU without recomputation (30/38/44/47% for 1/2/4/8 sequences at once) is this lab's assumption, labelled on the
   page; recomputation adds 2N per token at the same speed, so its time is 8N ÷ that share.
   Names (integrator's decision 6, batch8-brief.md): BF16 weights, FP32 gradients, FP32 copy of the weights (中文“FP32 权重副本”),
   Adam's m and v; optimizer state = FP32 copy + m + v (12 bytes); model states = all 18 bytes.
   window.__tmem is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='training-memory';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const NB='‑';

/* =====================================================================
   engine
   ===================================================================== */
/* config.json shapes (meta-llama/llama-models models/sku_list.py; Hugging Face config.json mirrors, specs/transformer-facts.md A1–A4).
   Parameter totals are the kit's (app.js MODELS) and, for the 1B, the sum over its config (tied embedding table). */
const MOD={
 '8b':{name:`Llama${NB}3.1${NB}8B`,P:8030261248,L:32,d:4096,H:32,KV:8,hd:128,f:14336,V:128256},
 '70b':{name:`Llama${NB}3.1${NB}70B`,P:70553706496,L:80,d:8192,H:64,KV:8,hd:128,f:28672,V:128256},
 '1b':{name:`Llama${NB}3.2${NB}1B`,P:1235814400,L:16,d:2048,H:32,KV:8,hd:64,f:8192,V:128256},
 '405b':{name:`Llama${NB}3.1${NB}405B`,P:405853388800,L:126,d:16384,H:128,KV:8,hd:128,f:53248,V:128256}
};
const GPU={h100:{name:'H100',cap:80e9},rtx4090:{name:'RTX 4090',cap:24e9}};
/* bytes per parameter: weights, gradients, FP32 copy, m, v */
const OPT={adamw:[2,4,4,4,4],adam8:[2,4,4,1,1],sgdm:[2,4,4,4,0]};
const OPTS=['adamw','adam8','sgdm'];
const SEQS=[1024,2048,4096,8192,16384,32768], MBS=[1,2,4,8], ACCS=[1,2,4,8];
const bpp=o=>OPT[o].reduce((a,b)=>a+b,0);
/* activations kept per token per layer (bytes), tensor by tensor; and with recomputation only the layer's input */
const aLayer=m=>2*(2*m.d+2*m.H*m.hd+2*m.KV*m.hd)+2*(2*m.d+4*m.f);
const aIn=m=>2*m.d;
function compute(s){
  const m=MOD[s.m], g=GPU[s.gpu||'h100'], T=s.seq*s.mb, b=OPT[s.opt];
  const blocks=b.map(x=>x*m.P), states=blocks.reduce((a,c)=>a+c,0);
  const perTok=m.L*(s.rc?aIn(m):aLayer(m)), acts=T*perTok, logits=T*m.V*4, peak=states+acts+logits;
  return {m,g,T,cap:g.cap,blocks,states,bpp:bpp(s.opt),aL:aLayer(m),aI:aIn(m),perTok,acts,logits,peak,fits:peak<=g.cap,room:g.cap-peak,
    infer:2*m.P,flopsTok:(s.rc?8:6)*m.P,acc:s.acc,rc:s.rc,tokUpd:s.seq*s.mb*s.acc};
}
/* one update over time: k micro-steps (forward, backward), then the optimizer step. Units are compute (a forward pass of T tokens = T).
   Vertices carry the kept activations and the logits; memory = states + acts + lg, linear between vertices. */
function timeline(c){
  const k=c.acc, rc=c.rc, Lr=c.m.L, T=c.T, fw=T, bw=T*(rc?3:2), unit=fw+bw, optD=0.06*k*unit, total=k*unit+optD;
  const aL=c.aL*T, aI=c.aI*T, kept=rc?aI:aL, lg=c.logits, V=[];
  const push=(x,acts,l,ph,li,mi)=>V.push({x:x/total,acts,lg:l,ph,li,mi});
  let x=0;
  for(let i=0;i<k;i++){
    push(x,0,0,'f',0,i);
    push(x+fw*0.9,Lr*kept,0,'f',Lr,i);
    push(x+fw,Lr*kept,lg,'h',Lr,i);
    x+=fw;
    const hb=bw*0.06, slot=(bw-hb)/Lr; let s=x+hb;
    push(s,Lr*kept,0,'b',Lr,i);
    for(let j=Lr;j>=1;j--){
      if(rc) push(s+slot/3,j*aI+aL,0,'r',j,i);
      push(s+slot,(j-1)*kept,0,'b',j-1,i); s+=slot;
    }
    x+=bw;
  }
  push(x,0,0,'o',0,k); push(total,0,0,'o',0,k);
  const tot=v=>c.states+v.acts+v.lg;
  let pk=0; V.forEach((v,i)=>{if(tot(v)>tot(V[pk])+1e-6) pk=i;});
  let oom=null;
  if(c.states>c.cap) oom=0;
  else for(let i=1;i<V.length;i++){const a=tot(V[i-1]), b=tot(V[i]); if(b>c.cap&&a<=c.cap){oom=V[i-1].x+(c.cap-a)/(b-a)*(V[i].x-V[i-1].x); break;}}
  return {V,pk,peakX:V[pk].x,oom,k,rc};
}
function atX(tl,x){
  const V=tl.V; let i=0; while(i<V.length-2&&V[i+1].x<x) i++;
  const a=V[i], b=V[i+1], u=b.x>a.x?Math.max(0,Math.min(1,(x-a.x)/(b.x-a.x))):1;
  return {acts:a.acts+(b.acts-a.acts)*u,lg:a.lg+(b.lg-a.lg)*u,seg:i,ph:a.ph==='h'&&u>0?'h':(b.ph==='r'?'r':a.ph),li:a.li,mi:a.mi};
}
/* the high-water mark up to x, with its split into activations and logits */
function hwX(tl,c,x){
  let best={acts:0,lg:0}, bt=-1;
  for(const v of tl.V){if(v.x>x) break; const t=v.acts+v.lg; if(t>bt){bt=t; best={acts:v.acts,lg:v.lg};}}
  const cur=atX(tl,x); if(cur.acts+cur.lg>bt) best={acts:cur.acts,lg:cur.lg};
  return best;
}

/* the challenge: Llama-3.2-1B from scratch on one RTX 4090, 2,048-token sequences, 32 sequences per update */
const CH={m:'1b',gpu:'rtx4090',seq:2048,upd:32,opts:['adamw','adam8'],rcs:[0,1],mbs:[1,2,4,8],util:{1:0.30,2:0.38,4:0.44,8:0.47},goal:0.95,start:{opt:'adamw',rc:0,mb:1}};
const chState=(o,rc,mb)=>({m:CH.m,gpu:CH.gpu,opt:o,seq:CH.seq,mb,acc:CH.upd/mb,rc});
/* time per update relative to "no recomputation, 1 sequence at once": compute (6N or 8N per token) ÷ MFU without recomputation */
const relTime=(rc,mb)=>((rc?8:6)/CH.util[mb])/(6/CH.util[1]);
function chEval(o,rc,mb){const c=compute(chState(o,rc,mb)), t=relTime(rc,mb); return {opt:o,rc,mb,peak:c.peak,fits:c.fits,t,ok:c.fits&&t<=CH.goal+1e-9};}
function chAll(){const a=[]; CH.opts.forEach(o=>CH.rcs.forEach(r=>CH.mbs.forEach(mb=>a.push(chEval(o,r,mb))))); return a;}
function chBest(){return chAll().filter(e=>e.fits).sort((a,b)=>a.t-b.t||a.peak-b.peak)[0];}
/* R1 (the owner's review of batch 8): the plans that meet the goal, fastest first (the closed list after Submit) */
function chPass(){return chAll().filter(e=>e.ok).sort((a,b)=>a.t-b.t||a.peak-b.peak);}
/* the one change that helps most: among the plans one setting away from the reader's, the fastest that meets the goal and, when the
   reader's plan already meets it, is faster than it as the result shows the time (two decimals, rule G7); null when no single change does */
function chOneChange(o,rc,mb){
  const me=chEval(o,rc,mb), cand=[];
  CH.opts.forEach(x=>{if(x!==o) cand.push({k:'opt',v:x,e:chEval(x,rc,mb)});});
  CH.rcs.forEach(x=>{if(x!==rc) cand.push({k:'rc',v:x,e:chEval(o,x,mb)});});
  CH.mbs.forEach(x=>{if(x!==mb) cand.push({k:'mb',v:x,e:chEval(o,rc,x)});});
  return cand.filter(c=>c.e.ok&&(!me.ok||parseFloat(fixd(c.e.t,2))<parseFloat(fixd(me.t,2)))).sort((a,b)=>a.e.t-b.e.t||a.e.peak-b.e.peak)[0]||null;
}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const GB=b=>{const v=b/1e9; return (Math.abs(v-Math.round(v))<1e-9?fint(Math.round(v)):fixd(v,1))+' GB';};
const gb=b=>fixd(b/1e9,1);
const big=n=>A.fmt.big?A.fmt.big(n):fint(n);
const bigH=n=>A.ui.big?A.ui.big(n):esc(big(n));
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
/* after a Chinese character, a link whose text starts with a Latin letter or digit gets a space (as labs/agent-anatomy.js does) */
const labLinkZ=id=>{const h=labLink(id); return /^[A-Za-z0-9]/.test(h.replace(/<[^>]+>/g,''))?' '+h:h;};
const M=t=>`<div class="tmem-fx">${A.ui.math(t)}</div>`;
/* a sum too long for a phone (the owner's review of 2026-10-09: no Show the math row scrolls sideways): its left side and “=”, then the
   terms split before each top-level “+”, wrapping onto a new line under the first term; {} keeps the “+” spaced as a binary operator */
const topSplit=t=>{const out=[]; let d=0, s=0; for(let i=0;i<t.length;i++){const c=t[i]; if('{(['.includes(c)) d++; else if('})]'.includes(c)) d--;
  else if(!d&&c===' '&&i>s&&(t.startsWith(' + ',i)||t.startsWith(' - ',i))){out.push(t.slice(s,i)); s=i+1;}} out.push(t.slice(s)); return out;};
/* a line of numbers (a big model, 70B or 405B, on a phone): it wraps before each top-level “=” or “+”, each new step under the line's start */
const relSplit=t=>{const out=[]; let d=0, s=0; for(let i=0;i<t.length;i++){const c=t[i]; if('{(['.includes(c)) d++; else if('})]'.includes(c)) d--;
  else if(!d&&c===' '&&i>s&&(t.startsWith(' = ',i)||t.startsWith(' + ',i))){out.push(t.slice(s,i)); s=i+1;}} out.push(t.slice(s)); return out;};
const MC=t=>`<div class="tmem-fx tmem-wr">${relSplit(t).map((p,i)=>`<span>${A.ui.math(i?`{} ${p}`:p)}</span>`).join('')}</div>`;
const MW=(lhs,t)=>`<div class="tmem-fx tmem-w"><span class="tmem-wl">${A.ui.math(lhs+' =')}</span><span class="tmem-wr">${topSplit(t).map((p,i)=>`<span>${A.ui.math(i?`{} ${p}`:p)}</span>`).join('')}</span></div>`;
const optName=o=>({adamw:tx('AdamW','AdamW'),adam8:tx('8-bit AdamW','8 位 AdamW'),sgdm:tx('SGD with momentum','带 momentum 的 SGD')}[o]);
const tokS=n=>fint(n);

/* =====================================================================
   module state
   ===================================================================== */
const BASE={m:'8b',gpu:'h100',opt:'adamw',seq:4096,mb:1,acc:1,rc:0};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let optTried=new Set(['adamw']), rcTried=new Set([0]), acTried=new Set(['a8x1']), acSel='a8x1', tokMoved=false, remView='states';
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=1, drillOpen=false;
/* the replay of one update: a mechanism replay, Medium by default (R3.9): 40 / 20 / 10 s */
let playT=1, playing=false, spd='mid', raf=0, lastT=0, relang=false;
const watched=new Set();
const DUR={slow:40000,mid:20000,fast:10000};
const shownPrev={};
const sigOf=s=>`${s.m}|${s.rc?1:0}|${s.acc}`;

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 full:()=>tx('Full fine-tuning updates every parameter of the model, as pretraining does; methods such as LoRA train only a few added parameters instead.','全量微调会更新模型的每一个参数，和预训练一样；LoRA 这类方法只训练少量新加的参数。'),
 gb:()=>tx('GB here is 10⁹ bytes. A GPU’s “80 GB” is also read as 80 × 10⁹ bytes in this lab.','这里的 GB 是 10⁹ 字节。这个 Lab 也把 GPU 的“80 GB”按 80 × 10⁹ 字节算。'),
 states:()=>tx('Model states are what training keeps for every parameter: the BF16 weights, the FP32 gradients and the optimizer state, 18 bytes in all. The optimizer state is the FP32 copy of the weights plus Adam’s m and v, 12 bytes. They stay in memory for the whole run.','模型状态是训练时每个参数都要带着的东西：BF16 权重、FP32 梯度和优化器状态，一共 18 字节。优化器状态是 FP32 权重副本，加上 Adam 的 m 和 v，一共 12 字节。整个训练过程中，它们一直占着显存。'),
 round:()=>tx('BF16 keeps only 8 significant bits, about 2 to 3 significant decimal digits. Near 0.02, two neighbouring BF16 numbers are about 0.00012 apart. So 0.02 + 0.00003 rounds back to 0.02, and the update is lost.','BF16 的有效数字只有 8 个二进制位，约合 2 到 3 位十进制有效数字。在 0.02 附近，相邻两个 BF16 数相差约 0.00012。所以 0.02 加上 0.00003，舍入后还是 0.02，这次更新就丢了。'),
 m:()=>tx('m is the running average of the gradient (Adam’s first moment). Each step moves it a little toward the new gradient.','m 是梯度的滑动平均（Adam 的一阶矩）。每一步把它往新的梯度挪一点。'),
 v:()=>tx('v is the running average of the squared gradient (Adam’s second moment). The update divides by its square root, so each parameter gets its own step size.','v 是梯度平方的滑动平均（Adam 的二阶矩）。更新时要除以它的平方根，所以每个参数的步长各不相同。'),
 act:()=>tx('Activations are the intermediate results the forward pass keeps so that the backward pass can compute the gradients from them.','激活是前向时留下来的中间结果，反向时要靠它们算出梯度。'),
 logit:()=>A.terms&&A.terms.logit?A.terms.logit()+tx(' Llama 3 gives every token 128,256 logits, one per vocabulary entry; this lab counts them in FP32.','Llama 3 给每个 token 算 128,256 个 logit，词表里每个 token 一个；这个 Lab 按 FP32 算。'):tx('A logit is the score the LM head gives each vocabulary entry.','logit 是 LM head 给词表里每个 token 的打分。'),
 rc:()=>tx('Recomputation (activation checkpointing): each layer keeps only its input; in the backward pass the layer’s forward is run again to get its intermediate results back.','重算（activation checkpointing）：每一层只留下它的输入；反向走到这一层时，把这一层的前向重新跑一遍，拿回中间结果。'),
 accum:()=>tx('Gradient accumulation: the sequences of one update go through the GPU in several smaller groups; each group’s gradients are added into the same gradient memory, and the optimizer updates the parameters once at the end.','梯度累积：一次更新要用的序列分成几小份送进 GPU，每份算出的梯度都加到同一块梯度显存里，最后优化器只更新一次参数。'),
 micro:()=>tx('A micro-batch is the group of sequences that goes through one forward and one backward pass together.','micro-batch 是一起做一次前向和一次反向的那一份序列。'),
 n6:()=>tx('Training compute per token: about 2N for the forward pass (N parameters, one multiply and one add each) and 4N for the backward pass, which works out gradients for both the inputs and the weights. Attention’s own arithmetic is left out.','每个 token 的训练计算：前向约 2N（N 个参数，每个做一次乘法、一次加法），反向约 4N，因为反向既要算输入的梯度，也要算权重的梯度。注意力自己的那部分计算没有算进去。'),
 /* MFU: the integrator's definition (batch8-brief.md, decision 2), then one sentence for this lab */
 mfu:()=>tx('MFU: the model’s own arithmetic (6 × parameters FLOPs per token) as a share of what the GPUs could do at peak in the same time; time spent waiting for transfers or idling in the pipeline counts. The extra forward pass of recomputation is not part of the model’s own arithmetic.',
            'MFU：模型真正需要的运算（每个 token 6 × 参数量次）占同一段时间里 GPU 峰值算力的比例。等通信、流水线空转的时间都算在里面。重算多做的那一遍前向，不算模型真正需要的运算。'),
 mfuWhy:()=>tx('MFU depends on the kernels, the model and the GPU; it has to be measured on your own setup and cannot be worked out from a formula. These numbers only carry the trend: bigger matrix multiplications leave the GPU idle less often.',
               'MFU 取决于 kernel、模型和 GPU，只能在自己的环境里实测，用公式算不出来。这里的数只用来体现趋势：矩阵乘法越大，GPU 越不容易闲着。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
/* the title already says 80 GB is not enough (R3.8, review B2), so the options cover everything above 80 GB, without gaps */
const PRED=[{v:'80-100',t:{en:'80 to 100 GB',zh:'80 到 100 GB'}},{v:'100-130',t:{en:'100 to 130 GB',zh:'100 到 130 GB'}},
  {v:'130-200',t:{en:'130 to 200 GB',zh:'130 到 200 GB'}},{v:'gt200',t:{en:'Over 200 GB',zh:'200 GB 以上'}}];
const PRED_RIGHT='130-200';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const C0=()=>compute(BASE);
const seqCtl=(id,inline)=>A.ui.range({id,key:'seq',label:tx('Tokens per sequence','每条序列的 token 数'),min:0,max:SEQS.length-1,value:Math.max(0,SEQS.indexOf(S.seq)),out:tokS(S.seq),inline});
const mbCtl=(id,inline)=>A.ui.seg({id,key:'mb',label:tx('Sequences at once','每次一起送进 GPU 的序列'),options:MBS.map(k=>[String(k),String(k)]),value:String(S.mb),inline});
const OPTD={adamw:{en:'m and v both in FP32, 4 bytes each.',zh:'m 和 v 都用 FP32 存，各 4 字节。'},
  adam8:{en:'m and v stored as 8-bit integers, each small block with its own scale: 1 byte each. They are expanded back to FP32 only while the update is computed.',zh:'m 和 v 压成 8 位整数存，每一小块各带一个缩放系数，各 1 字节。只在算更新的那一刻临时还原成 FP32。'},
  sgdm:{en:'Keeps only m (FP32, 4 bytes); there is no v.',zh:'只记一个 m（FP32，4 字节），没有 v。'}};
const optRes=o=>tx(`model states ${GB(compute({...BASE,opt:o}).states)}`,`模型状态 ${GB(compute({...BASE,opt:o}).states)}`);
const optLever=()=>`<div class="tmem-lv" id="tmem-ic-opt">`+A.ui.levers({aria:tx('Optimizer, pick one','优化器，选一种'),value:'opt:'+S.opt,
  options:OPTS.map(o=>({v:'opt:'+o,title:optName(o)+(o==='adamw'?tx(' (initial setup)','（初始设置）'):''),desc:L(OPTD[o]),result:esc(optTried.has(o)||o===S.opt?optRes(o):'')}))})+`</div>`;
const rcRes=r=>tx(`peak ${GB(compute({...BASE,rc:r}).peak)}`,`峰值 ${GB(compute({...BASE,rc:r}).peak)}`);
const rcLever=()=>`<div class="tmem-lv" id="tmem-ic-rc">`+A.ui.levers({aria:tx('Recomputation, pick one','重算，选一种'),value:'rc:'+S.rc,
  options:[{v:'rc:0',title:tx('Keep everything (initial setup)','全部留着（初始设置）'),desc:tx('Every layer keeps all its intermediate results until the backward pass uses them.','每一层的中间结果都留到反向用完为止。'),result:esc(rcRes(0))},
    {v:'rc:1',title:tx('Recompute every layer','每一层都重算'),desc:tx('Every layer keeps only its input and runs its forward again in the backward pass.','每一层只留输入，反向时把这一层的前向再跑一遍。'),result:esc(rcTried.has(1)?rcRes(1):'')}]})+`</div>`;
/* part 5: one update of 8 sequences of 4,096 tokens, in 1, 2, 4 or 8 micro-batches; the last option also turns on the two other tricks */
const AC=[{v:'a8x1',mb:8,acc:1,t:{en:'All 8 at once (initial setup)',zh:'8 条一起算（初始设置）'},d:{en:'One forward and one backward pass over all 8 sequences.',zh:'8 条序列一起做一次前向、一次反向。'}},
  {v:'a4x2',mb:4,acc:2,t:{en:'2 micro-batches of 4',zh:'分 2 份，每份 4 条'},d:{en:'Two passes; the gradients of both are added up.',zh:'跑两趟，两份的梯度加在一起。'}},
  {v:'a2x4',mb:2,acc:4,t:{en:'4 micro-batches of 2',zh:'分 4 份，每份 2 条'},d:{en:'Four passes.',zh:'跑四趟。'}},
  {v:'a1x8',mb:1,acc:8,t:{en:'8 micro-batches of 1',zh:'分 8 份，每份 1 条'},d:{en:'Eight passes, one sequence each.',zh:'跑八趟，每趟 1 条。'}},
  {v:'all',mb:1,acc:8,opt:'adam8',rc:1,t:{en:'All three tricks: 8 micro-batches, 8-bit AdamW and recomputation',zh:'三个办法一起用：分 8 份 + 8 位 AdamW + 重算'},d:{en:'This option changes three things at once, to see whether everything on this page together is enough.',zh:'这一项同时改了三处，看看这一页讲过的办法全用上，够不够。'}}];
const acState=v=>{const o=AC.find(a=>a.v===v); return {...BASE,mb:o.mb,acc:o.acc,opt:o.opt||'adamw',rc:o.rc||0};};
const acRes=v=>tx(`peak ${GB(compute(acState(v)).peak)}`,`峰值 ${GB(compute(acState(v)).peak)}`);
const acLever=()=>`<div class="tmem-lv" id="tmem-ic-ac">`+A.ui.levers({aria:tx('How to split the update, pick one','一次更新怎么分，选一种'),value:'ac:'+acSel,
  options:AC.map(o=>({v:'ac:'+o.v,title:L(o.t),desc:L(o.d),result:esc(acTried.has(o.v)||o.v===acSel?acRes(o.v):'')}))})+`</div>`;

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},gate:()=>!!pred,skip:true,
  lead:()=>tx('Inference holds 16 GB of weights. How much does training need?','推理只放 16 GB 的权重。训练要多少显存？'),
  enter:()=>({...BASE}),
  body:()=>{const c=C0(), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You are to ${DEF('fully fine-tune',TIP.full())} ${c.m.name} on one H100 with 80 ${DEF('GB',TIP.gb())} of memory. The model has ${bigH(c.m.P)} parameters. Stored in BF16, 2 bytes each, its weights take ${GB(c.infer)}; for inference, they are most of what sits in GPU memory.</p>`+
     `<p>Each training step sends one sequence of 4,096 tokens through the model and computes the loss, then works out a gradient for every parameter in the backward pass, and finally lets AdamW update the parameters. So besides the weights, the GPU must hold three more things: every parameter’s gradient, the numbers the optimizer keeps for every parameter, and intermediate results. The intermediate results are computed in the forward pass and kept until the backward pass uses them.</p>`+
     `<p>This is a gut guess; the next parts take it apart piece by piece. One 80 GB GPU cannot hold this step. How much memory does it need at its highest point?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the result.</p>':''}`,
     `<p>你要在一张 80 ${DEF('GB',TIP.gb())} 显存的 H100 上${DEF('全量微调',TIP.full())} ${c.m.name}。它有 ${bigH(c.m.P)} 个参数，用 BF16 存，每个参数 2 字节，权重一共 ${GB(c.infer)}。推理时，显存里主要就是这些权重。</p>`+
     `<p>训练的每一步，先把一条 4,096 个 token 的序列送进模型，算出 loss，再反向算出每个参数的梯度，最后由 AdamW 更新参数。所以除了权重，GPU 上还要放三样东西：每个参数的梯度、优化器给每个参数记的数，还有中间结果。中间结果是前向时算出来的，要一直留到反向再用。</p>`+
     `<p>这一题凭直觉猜，后面几部分会一项一项讲清楚。一张 80 GB 的卡放不下这一步，那它最多要用到多少显存？</p>${ch}${pred?'<p class="muted small">已记下。第 2 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'states',title:{en:'Four more numbers per parameter',zh:'每个参数还要带四样东西'},expect:()=>watched.has(sigOf(BASE)),showMe:()=>finishPlay(),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Add FP32 gradients, an FP32 copy of the weights, and Adam’s m and v: 18 bytes per parameter.','加上 FP32 梯度、FP32 权重副本，以及 Adam 的 m 和 v，每个参数 18 字节。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>The chart plays one training step. The large block at the bottom that never moves is the ${DEF('model states',TIP.states())}. Besides its BF16 weight (2 bytes), every parameter carries four more numbers:</p>`+
     `<ul class="tmem-four"><li><b>FP32 gradient</b><span>worked out in the backward pass and kept in FP32, 4 bytes. Gradients are often added together, and adding them in FP32 loses no precision.</span></li>`+
     `<li><b>FP32 copy</b><span>an FP32 copy of the weights, 4 bytes. The updates are added to it, because a small update added to a BF16 weight is often ${DEF('rounded away',TIP.round())} (see ${labLink('number-formats')}).</span></li>`+
     `<li><b>Adam’s m</b><span>the ${DEF('running average of the gradient',TIP.m())}, 4 bytes (FP32).</span></li>`+
     `<li><b>Adam’s v</b><span>the ${DEF('running average of the squared gradient',TIP.v())}, 4 bytes (FP32). For m and v, see ${labLink('optimizers')}.</span></li></ul>`+
     M('2 + 4 + 4 + 4 + 4 = 18 \\, \\text{bytes per parameter}')+
     `<p>The FP32 copy, m and v are what the optimizer uses. Together they are called the optimizer state, 12 bytes per parameter.</p>`+
     `<p>How m and v are stored depends on the optimizer. Pick an optimizer and see how much memory the model states take.</p>`,
     `<p>图里在播放训练的一步。最下面一直不动的那一大块，叫${DEF('模型状态',TIP.states())}。每个参数除了 BF16 权重（2 字节），还带着四样东西：</p>`+
     `<ul class="tmem-four"><li><b>FP32 梯度</b><span>反向时算出来，用 FP32 存，4 字节。梯度常常要一份份加起来，用 FP32 加才不丢精度。</span></li>`+
     `<li><b>FP32 权重副本</b><span>权重的一份 FP32 拷贝，4 字节。更新都加在它上面，因为很小的更新加到 BF16 权重上，常常被${DEF('舍入掉',TIP.round())}（见${labLink('number-formats')}）。</span></li>`+
     `<li><b>Adam 的 m</b><span>${DEF('梯度的滑动平均',TIP.m())}，4 字节（FP32）。</span></li>`+
     `<li><b>Adam 的 v</b><span>${DEF('梯度平方的滑动平均',TIP.v())}，4 字节（FP32）。m 和 v 见${labLink('optimizers')}。</span></li></ul>`+
     M('2 + 4 + 4 + 4 + 4 = 18 \\, \\text{字节/参数}')+
     `<p>FP32 权重副本，加上 Adam 的 m 和 v，是给优化器用的，合称优化器状态，每个参数 12 字节。</p>`+
     `<p>m 和 v 怎么存，取决于优化器。选一种优化器，看看模型状态要占多少显存。</p>`)+optLever(),
  obs:()=>obsStates()},
 {id:'acts',title:{en:'What the forward pass keeps',zh:'前向留给反向的中间结果'},expect:()=>tokMoved,showMe:{seq:16384},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Activations grow with the tokens processed at once.','激活随一次处理的 token 数一起涨。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>Above the model states sit the ${DEF('activations',TIP.act())}: what every layer computes in the forward pass and keeps until the backward pass has used it to work out the gradients (see ${labLink('backprop')}). For example, the ${DEF('MLP block',A.terms&&A.terms.mlp?A.terms.mlp():'')} first widens every token’s vector with two matrices (the gate and up projections in Llama). Both results are kept for the backward pass.</p>`+
     `<p>Every token leaves its own set in every layer, so the more tokens are processed at once, the more activations there are. Longer sequences, or more sequences at once, both make them grow. After the last layer the model also gives each token 128,256 ${DEF('logits',TIP.logit())}, counted here in FP32. The logits grow with the tokens too.</p>`,
     `<p>模型状态上面那一块是${DEF('激活',TIP.act())}，也就是前向时每一层算出的中间结果。它们要一直留到反向，用来算梯度（见${labLink('backprop')}）。比如 ${DEF('MLP 块',A.terms&&A.terms.mlp?A.terms.mlp():'')}先用两个矩阵，把每个 token 的向量变宽（Llama 里叫 gate 投影和 up 投影）。这两个结果都要留到反向。</p>`+
     `<p>每个 token 在每一层都留下自己的一份，所以一次处理的 token 越多，激活越大。序列越长，或者一次送进 GPU 的序列越多，激活都会变大。最后一层之后，模型还要给每个 token 算出 128,256 个 ${DEF('logit',TIP.logit())}，这里按 FP32 算。logits 也跟着 token 数一起涨。</p>`)+
     seqCtl('tmem-ic-seq',true)+mbCtl('tmem-ic-mb',true)+
     tx(`<p class="small muted">The attention weights (one score for every pair of tokens) are not kept: FlashAttention recomputes them in the backward pass instead (see ${labLink('flash-attention')}).</p>`,
        `<p class="small muted">注意力权重（每两个 token 之间一个得分）不在其中。FlashAttention 不存它们，反向时重新算（见 ${labLink('flash-attention')}）。</p>`),
  obs:()=>obsActs()},
 {id:'rc',title:{en:'Recompute instead of storing',zh:'不存了，反向时再算一遍'},expect:()=>rcTried.has(1),showMe:{rc:1},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Keep only each layer’s input; pay one more forward pass.','每层只留输入，代价是多算一遍前向。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p><b>Recomputation</b> (${DEF('activation checkpointing',TIP.rc())}): in the forward pass, each layer keeps only its input and drops its intermediate results as soon as it is done. When the backward pass reaches that layer, it first runs the layer’s forward again from the kept input to get the intermediate results back, then works out the gradients.</p>`+
     `<p>The cost is one more forward pass through the whole model. Training a token takes about 6N operations without it (N is the number of parameters). With it, it takes ${DEF('8N',TIP.n6())}, a third more.</p>`+
     `<p>Pick one and see how the peak memory changes.</p>`,
     `<p><b>重算</b>（${DEF('activation checkpointing',TIP.rc())}）的意思是，前向时每一层只留下自己的输入，层里的中间结果用完就扔。反向走到这一层时，先用留下的输入把这一层的前向重新算一遍，拿回中间结果，再算梯度。</p>`+
     `<p>代价是整个模型多做一遍前向。不重算时，训练一个 token 约要 6N 次运算（N 是参数量）。重算后${DEF('变成 8N',TIP.n6())}，多三分之一。</p>`+
     `<p>选一种做法，看看峰值显存怎么变。</p>`)+rcLever(),
  obs:()=>obsRc()},
 {id:'accum',title:{en:'One update, in smaller pieces',zh:'一次更新，分几份算'},expect:()=>acTried.has('all'),showMe:()=>setAc('all'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Smaller pieces shrink the activations, never the model states.','分得再细，也只缩小激活，模型状态不变。'),
  enter:()=>{acSel='a8x1'; acTried=new Set(['a8x1']); return acState('a8x1');},
  body:()=>tx(`<p>Now one parameter update uses 8 sequences of 4,096 tokens. With ${DEF('gradient accumulation',TIP.accum())}, the 8 sequences go through the GPU in a few smaller groups. The sequences sent at once are called a ${DEF('micro-batch',TIP.micro())}. Each group runs its forward and backward pass, and its gradients are added into the same gradient memory. Only after all 8 does the optimizer update the parameters, once.</p>`+
     `<p>The update comes out the same as with all 8 at once; it just takes more passes. Pick a way to split it and see how the peak memory changes.</p>`,
     `<p>现在一次参数更新要用 8 条 4,096 个 token 的序列。用${DEF('梯度累积',TIP.accum())}时，这 8 条分成几小份送进 GPU，一次送进去的那几条叫一个 ${DEF('micro-batch',TIP.micro())}。每份做完前向和反向，梯度就加到同一块梯度显存里。8 条都算完，优化器才更新一次参数。</p>`+
     `<p>这样得到的更新和 8 条一起算一样，只是多跑了几趟。选一种分法，看看峰值显存怎么变。</p>`)+acLever(),
  obs:()=>obsAc()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Each trick shrinks one kind of memory, never all of them.','每个办法只压一种显存，压不了全部。'),
  enter:()=>remState(remView),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
function fitLine(c){return c.fits?tx(`${GB(c.room)} to spare`,`还剩 ${GB(c.room)}`):tx(`${GB(-c.room)} more than the ${c.g.name} holds`,`比 ${c.g.name} 的 ${GB(c.cap)} 多 ${GB(-c.room)}`);}
function obsStates(){
  if(!watched.has(sigOf(BASE))) return '';
  const c=compute(S), b=C0();
  let s='';
  if(S.opt==='adamw') s=tx(`<p><b>The step peaks at ${GB(b.peak)}: ${GB(-b.room)} more than the H100’s ${GB(b.cap)}. It does not fit.</b></p>`,`<p><b>这一步的峰值是 ${GB(b.peak)}，比 H100 的 ${GB(b.cap)} 多 ${GB(-b.room)}，放不下。</b></p>`);
  else s=tx(`<p><b>With ${optName(S.opt)}, the model states drop from ${GB(b.states)} to ${GB(c.states)}, and the step peaks at ${GB(c.peak)}: ${c.fits?'it fits':'it still does not fit'}.</b></p>`,
            `<p><b>换成${optName(S.opt)}后，模型状态从 ${GB(b.states)} 降到 ${GB(c.states)}，峰值 ${GB(c.peak)}，${c.fits?'装得下了':'还是放不下'}。</b></p>`);
  s+=verdict();
  if(S.opt==='adamw') s+=tx(`<p>${GB(b.states)} of it is the model states, held from the first layer’s forward pass to the end of the update. At 18 bytes per parameter, that is 9 times the weights alone, which is all that inference holds.</p>`,
                            `<p>其中 ${GB(b.states)} 是模型状态，从第一层前向一直占到参数更新完。每个参数 18 字节，是推理时只放权重的 9 倍。</p>`);
  else if(S.opt==='adam8') s+=tx('<p>It saves only on Adam’s m and v, from 8 bytes per parameter to 2. The BF16 weights, the FP32 gradients and the FP32 copy stay as they were.</p>','<p>它省下的只有 Adam 的 m 和 v，每个参数从 8 字节压到 2 字节。BF16 权重、FP32 梯度和 FP32 权重副本都没变。</p>');
  else s+=tx('<p>It saves only the 4 bytes of v. SGD with momentum often trains large Transformers worse than AdamW, which is why AdamW is the usual choice.</p>','<p>它只省下 v 的 4 字节。训练大型 Transformer 时，带 momentum 的 SGD 往往不如 AdamW，所以大家通常还是用 AdamW。</p>');
  return s;
}
function obsActs(){
  const c=compute(S), b=C0();
  if(c.T===b.T) return tx(`<p><b>${tokS(c.T)} tokens at once: the activations take ${GB(c.acts)} and the logits ${GB(c.logits)}.</b> Drag the length, or send more sequences at once.</p>`,`<p><b>一次 ${tokS(c.T)} 个 token：激活占 ${GB(c.acts)}，logits 占 ${GB(c.logits)}。</b>拖动序列长度，或者一次多送几条试试。</p>`);
  const r=c.T/b.T, rs=r>=1?tx(`${fixd(r,r%1?1:0)} times`,`${fixd(r,r%1?1:0)} 倍`):tx(`1/${fixd(1/r,0)}`,`1/${fixd(1/r,0)}`);
  return tx(`<p><b>${tokS(c.T)} tokens at once: the activations take ${GB(c.acts)} (${GB(b.acts)} at 4,096 tokens), and the step peaks at ${GB(c.peak)}.</b></p><p>The tokens became ${rs} as many, and so did the activations and the logits. The ${GB(c.states)} of model states did not move. The activations are what every token leaves in every layer, so:</p>`,
            `<p><b>一次 ${tokS(c.T)} 个 token：激活占 ${GB(c.acts)}（4,096 个 token 时是 ${GB(b.acts)}），峰值 ${GB(c.peak)}。</b></p><p>token 数变成了原来的 ${rs}，激活和 logits 也跟着变成 ${rs}。${GB(c.states)} 的模型状态一点没动。激活是每个 token 在每一层留下的一份，所以：</p>`)+
    M(tx('\\text{activations} = \\text{tokens} \\times \\text{layers} \\times a','\\text{激活} = \\text{token 数} \\times \\text{层数} \\times a'))+
    tx(`<p>Here “tokens” means the tokens processed at once. a is what one token leaves in one layer. ${c.m.name} has ${c.m.L} layers, and a is ${fixd(c.aL/1000,0)} KB. ${J('math','Show the math','Opens the tensor-by-tensor count.')} lists the tensors that make it up.</p>`,
       `<p>这里的 token 数，指一次处理的 token 数。a 是一个 token 在一层里留下的字节数。${c.m.name} 有 ${c.m.L} 层，a 是 ${fixd(c.aL/1000,0)} KB。它由哪些张量组成，见${J('math','“展开公式”','打开逐个张量的清单。')}。</p>`);
}
function obsRc(){
  const c=compute(S), b=C0();
  if(!S.rc) return tx(`<p><b>Keeping everything: the activations take ${GB(b.acts)}, and the step peaks at ${GB(b.peak)}.</b> Now pick recomputation.</p>`,`<p><b>全部留着：激活占 ${GB(b.acts)}，峰值 ${GB(b.peak)}。</b>再选“每一层都重算”试试。</p>`);
  return tx(`<p><b>Recomputing every layer: the activations drop from ${GB(b.acts)} to ${GB(c.acts)}, and the peak from ${GB(b.peak)} to ${GB(c.peak)}. It still does not fit.</b></p>`+
     `<p>With recomputation, the backward pass first redoes each layer’s forward pass to get its activations back, then works out the gradients. Recomputation saves only on the activations. The ${GB(c.states)} of model states stay. It costs a third more compute per token:</p>`,
     `<p><b>每一层都重算：激活从 ${GB(b.acts)} 降到 ${GB(c.acts)}，峰值从 ${GB(b.peak)} 降到 ${GB(c.peak)}，还是放不下。</b></p>`+
     `<p>重算时，反向走到每一层，先把这一层的前向再做一遍，拿回激活，再算梯度。重算省下的只有激活，${GB(c.states)} 的模型状态还在。代价是每个 token 的计算多三分之一：</p>`)+
    M(tx('6N \\to 8N \\, \\text{operations per token}','6N \\to 8N \\, \\text{次运算/token}'));
}
function obsAc(){
  const c=compute(S), b=compute(acState('a8x1')), o=AC.find(a=>a.v===acSel);
  if(acSel==='a8x1') return tx(`<p><b>All 8 at once: 32,768 tokens in one pass, and the step peaks at ${GB(b.peak)}.</b> Now split it.</p>`,`<p><b>8 条一起算：一趟 32,768 个 token，峰值 ${GB(b.peak)}。</b>分成几份试试。</p>`);
  if(acSel==='all'){const s8=compute({...BASE,opt:'adam8'});
    return tx(`<p><b>All three tricks together: the step peaks at ${GB(c.peak)}, still ${GB(-c.room)} more than the H100’s ${GB(c.cap)}.</b></p><p>Even with 8-bit AdamW, the model states alone are ${GB(s8.states)}, more than the whole GPU holds before a single activation is kept. None of the three tricks on this page shrinks the BF16 weights, the FP32 gradients or the FP32 copy.</p><p>With several GPUs, the usual way is to split the model states across them, each GPU holding a part. That is the next lab in this branch, ${labLink('parallelism')}. With one GPU, you can also move the optimizer state and the gradients to CPU memory (DeepSpeed’s ZeRO-Offload), at the cost of moving data between the CPU and the GPU in every step.</p>`,
              `<p><b>三个办法一起用：峰值 ${GB(c.peak)}，还是比 H100 的 ${GB(c.cap)} 多 ${GB(-c.room)}。</b></p><p>就算用了 8 位 AdamW，光模型状态也有 ${GB(s8.states)}，一份激活还没存，就已经超过整张卡了。这一页的三个办法，都压不小 BF16 权重、FP32 梯度和 FP32 权重副本。</p><p>有好几张卡时，通常把模型状态分到多张卡上，每张卡只存一部分。这就是这一枝的下一个 Lab：${labLink('parallelism')}。只有一张卡时，也可以把优化器状态和梯度挪到 CPU 内存里（DeepSpeed 的 ZeRO-Offload），代价是每一步都要在 CPU 和 GPU 之间来回搬数据。</p>`);}
  return tx(`<p><b>${L(o.t)}: the step peaks at ${GB(c.peak)} (${GB(b.peak)} with all 8 at once).</b></p><p>Each pass now holds the activations of ${o.mb===1?'one sequence':o.mb+' sequences'} only, so they shrink with every split. The ${GB(c.states)} of model states stay the same however you split, so the peak never goes below them.</p>`,
            `<p><b>${L(o.t)}：峰值 ${GB(c.peak)}（8 条一起算时是 ${GB(b.peak)}）。</b></p><p>现在每一趟只放 ${o.mb} 条序列的激活，分得越细，激活越小。可是 ${GB(c.states)} 的模型状态怎么分都不变，所以峰值不会低于它。</p>`);
}
const REMS=['states','rc','all'];
function remState(k){if(k==='rc') return {...BASE,rc:1}; if(k==='all') return acState('all'); return {...BASE};}
function remBtn(k){const on=remView===k;
  const tip={states:tx('Shows the 8B’s step with AdamW and one sequence of 4,096 tokens.','在图里显示 8B 用 AdamW、一条 4,096 个 token 的序列时的那一步。'),rc:tx('Shows the 8B’s step with every layer recomputed.','在图里显示 8B 每一层都重算时的那一步。'),all:tx('Shows the 8B’s step with all three tricks together.','在图里显示 8B 三个办法一起用时的那一步。')}[k];
  return ` <button type="button" class="linkbtn tmem-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const b=C0(), r=compute({...BASE,rc:1}), all=compute(acState('all')), a8=compute(acState('a8x1'));
  return tx(`<ul class="tmem-remlist"><li><b>Training holds four more things per parameter than inference.</b> With AdamW the model states are 9 times the BF16 weights. So to tell whether a model can be trained, count 18 bytes per parameter, not 2.${remBtn('states')}</li>`+
    `<li><b>Each memory trick shrinks one item only.</b> Recomputation shrinks only the activations, and 8-bit AdamW only Adam’s m and v. So before picking a trick, look at which item takes the most memory.${remBtn('rc')}</li>`+
    `<li><b>When the model states alone fill one GPU, the usual way is to split them across several GPUs.</b> Even with all three tricks on, the 8B’s model states do not fit in one 80 GB GPU. How to split them: ${labLink('parallelism')}.${remBtn('all')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Memory is about parameters × 2 bytes.” That is inference. Training also holds the FP32 gradients, the FP32 copy of the weights, and Adam’s m and v: 18 bytes per parameter with AdamW.</p><p>✗ “Halve the batch, halve the memory.” Only the activations and logits halve; the model states stay as they are. So when the model states take most of the memory, halving the batch barely lowers the peak.</p><p>✗ “Recomputation is free.” It runs every layer’s forward pass a second time: a third more compute per token.</p></div>${A.shell.toScenarios()}`,
    `<ul class="tmem-remlist"><li><b>训练时，每个参数比推理多带四样东西。</b>用 AdamW 训练时，模型状态是 BF16 权重的 9 倍。所以估算一个模型能不能训，要按每个参数 18 字节算，不能按 2 字节算。${remBtn('states')}</li>`+
    `<li><b>每种省显存的办法只压其中一项。</b>重算只压激活，8 位 AdamW 只压 Adam 的 m 和 v，别的都不动。所以选办法之前，先看清哪一项占的显存最多。${remBtn('rc')}</li>`+
    `<li><b>光模型状态就占满一张卡时，常见的做法是把它们分到多张卡上。</b>三个办法全用上，8B 的模型状态在一张 80 GB 的卡上还是放不下。怎么分到多张卡上，见${labLinkZ('parallelism')}。${remBtn('all')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “显存约等于参数量 × 2 字节。”这是推理。训练还要放 FP32 梯度、FP32 权重副本，以及 Adam 的 m 和 v，用 AdamW 时每个参数 18 字节。</p><p>✗ “batch 减半，显存就减半。”减半的只有激活和 logits，模型状态一点不变。所以模型状态占大头时，batch 减半，峰值也降不了多少。</p><p>✗ “重算不花代价。”它把每一层的前向再跑一遍，每个 token 的计算多三分之一。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 sku:'https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py',
 hfmem:'https://github.com/huggingface/transformers/blob/main/docs/source/en/model_memory_anatomy.md',
 megdo:'https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/dist_optimizer.md',
 mle:'https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md',
 dsmem:'https://github.com/microsoft/DeepSpeed/blob/master/docs/code-docs/source/memory.rst',
 bnb:'https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/README.md',
 ckpt:'https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py',
 snap:'https://github.com/pytorch/pytorch/blob/main/docs/source/torch_cuda_memory.md',
 meg:'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py',
 zero:'https://arxiv.org/abs/1910.02054',
 sel:'https://arxiv.org/abs/2205.05198',
 b8:'https://arxiv.org/abs/2110.02861',
 sub:'https://arxiv.org/abs/1604.06174',
 tt:'https://github.com/pytorch/torchtune/blob/main/README.md',
 ttcfg:'https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full_single_device.yaml',
 bnbx:'https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/explanations/optimizers.mdx'
};
const PRESETS=[
 {id:'ft8',t:{en:'Full fine-tuning of the 8B on one H100',zh:'在一张 H100 上全量微调 8B'},d:{en:'A team wants to fine-tune every parameter on the GPU it has',zh:'团队想用手头的一张卡训练全部参数'},s:{...BASE}},
 {id:'squeeze',t:{en:'Every trick on, still one H100',zh:'能省的都省了，还是一张 H100'},d:{en:'8-bit AdamW and recomputation, one sequence at once',zh:'8 位 AdamW 加上重算，每次送进一条序列'},s:{...BASE,opt:'adam8',rc:1}},
 {id:'long',t:{en:'Fine-tuning on 32K-token documents',zh:'用 32K token 的长文档微调'},d:{en:'Long-context training, one document per pass',zh:'长上下文训练，每趟一篇文档'},s:{...BASE,seq:32768}},
 {id:'big',t:{en:`Full training of Llama${NB}3.1${NB}70B`,zh:`全量训练 Llama${NB}3.1${NB}70B`},d:{en:'About 9 times the 8B’s parameters, same recipe',zh:'参数是 8B 的约 9 倍，训练方法不变'},s:{...BASE,m:'70b'}}
];
const KEYS=['m','opt','seq','mb','acc','rc'];
const presetNow=()=>PRESETS.findIndex(p=>KEYS.every(k=>p.s[k]===S[k]));
function scenHTML(i){
  const p=PRESETS[i], c=compute(S);
  if(p.id==='ft8') return tx(`<p><b>The step peaks at ${GB(c.peak)}, ${GB(-c.room)} more than one H100 holds.</b></p><p>${GB(c.states)} of it is the model states, there before the first token arrives. Shorter sequences cannot fix that.</p>`,
                             `<p><b>这一步的峰值是 ${GB(c.peak)}，比一张 H100 多 ${GB(-c.room)}。</b></p><p>其中 ${GB(c.states)} 是模型状态，第一个 token 进来之前就已经占着了。序列再短也解决不了。</p>`);
  if(p.id==='squeeze') return tx(`<p><b>With 8-bit AdamW and recomputation the step still peaks at ${GB(c.peak)}, ${GB(-c.room)} more than the H100’s ${GB(c.cap)}.</b></p><p>The model states alone are ${GB(c.states)}, more than the H100’s ${GB(c.cap)} before the first token arrives. No setting on this page brings them below it.</p>`,
                                 `<p><b>8 位 AdamW 加上重算，峰值还有 ${GB(c.peak)}，比 H100 的 ${GB(c.cap)} 多 ${GB(-c.room)}。</b></p><p>光模型状态就有 ${GB(c.states)}，第一个 token 还没进来，就已经超过 H100 的 ${GB(c.cap)}。这一页的哪个设置都压不到它以下。</p>`);
  if(p.id==='long'){const r=compute({...S,rc:1});
    return tx(`<p><b>With one 32K-token document, the activations alone take ${GB(c.acts)}, more than the ${GB(c.states)} of model states.</b></p><p>At this length the activations are the biggest item, and recomputation is no longer optional. It brings them down to ${GB(r.acts)}.</p>`,
              `<p><b>一篇 32K token 的文档，光激活就有 ${GB(c.acts)}，比 ${GB(c.states)} 的模型状态还多。</b></p><p>这么长的序列，激活成了最大的一项，重算就成了必选。重算能把激活压到 ${GB(r.acts)}。</p>`);}
  const n=Math.ceil(c.states/c.cap);
  return tx(`<p><b>${c.m.name}’s model states alone are ${GB(c.states)}, so it takes at least ${n} H100s just to hold them.</b></p><p>That is before any activation. Models this size are usually trained with their model states split across many GPUs (see ${labLink('parallelism')}).</p>`,
            `<p><b>${c.m.name} 光模型状态就有 ${GB(c.states)}，至少要 ${n} 张 H100 才放得下。</b></p><p>这还没算任何激活。这么大的模型，训练时通常把模型状态分到很多张卡上（见${labLinkZ('parallelism')}）。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=(o,rc,mb)=>tx(`${optName(o)}, ${rc?'recompute every layer':'no recomputation'}, ${mb} ${mb===1?'sequence':'sequences'} at once`,`${optName(o)}，${rc?'每一层都重算':'不重算'}，每次 ${mb} 条`);
const chSame=(e,o,rc,mb)=>e.opt===o&&e.rc===rc&&e.mb===mb;
/* R1 (the owner's review of batch 8, as labs/train-capstone.js): after Submit, the verdict first; then the reader's plan beside the
   fastest plan, with only the settings they differ in as chips (bold where the reader's differs) and what they share written once;
   each plan's two goals as bars with the goal's dashed line, the part beyond a goal red; one sentence on the main gap and one on the
   single change that helps most; the plans that pass behind a disclosure closed by default. Never all 16 plans. */
const CHS=()=>[
 {k:'opt',chip:e=>optName(e.opt),en:e=>`Both plans use ${optName(e.opt)}.`,zh:e=>`都用${optName(e.opt)}`},
 {k:'rc',chip:e=>e.rc?tx('recompute every layer','每一层都重算'):tx('no recomputation','不重算'),en:e=>e.rc?'Both plans recompute every layer.':'Neither plan recomputes.',zh:e=>e.rc?'都每一层重算':'都不重算'},
 {k:'mb',chip:e=>tx(`${e.mb} ${e.mb===1?'sequence':'sequences'} at once`,`每次 ${e.mb} 条`),en:e=>`Both plans send ${e.mb} ${e.mb===1?'sequence':'sequences'} at once.`,zh:e=>`每次都送 ${e.mb} 条`}];
function chChange(k,v){
  if(k==='opt') return v==='adam8'?tx('switching to 8-bit AdamW','换成 8 位 AdamW'):tx('switching back to AdamW','换回 AdamW');
  if(k==='rc') return v?tx('recomputing every layer','改成每一层都重算'):tx('dropping recomputation','去掉重算');
  return tx(`sending ${v} ${v===1?'sequence':'sequences'} at once`,`改成每次送 ${v} 条`);
}
/* one plan: its chips, then its two goals as bars (peak against 24 GB, time against 0.95 of the reference), each value written beside
   its bar; no hover that only repeats a value more exactly (the owner, 2026-10-07 23:37 UTC) */
function chRow(e,role,diff,best,MX,TM){
  const lab={me:tx('Your plan','你的方案'),mebest:tx('Your plan, the fastest one','你的方案，就是最快的方案'),
    best:DEF(tx('Fastest plan','最快的方案'),tx('Of the plans that fit in 24 GB, the one whose update takes the least time.','放得下 24 GB 的方案里，每次更新用时最少的那一种。'))}[role];
  const shown=CHS().filter(g=>role==='mebest'||diff.includes(g.k));
  const chips=shown.map(g=>`<span class="tmem-chip${role==='me'&&e[g.k]!==best[g.k]?' on':''}">${esc(g.chip(e))}</span>`).join('');
  const bar=(v,goal,mx,txt)=>{const over=v>goal+1e-9, gx=goal/mx*100;
    return `<span class="tmem-c3trk" aria-hidden="true"><i class="tmem-c3fill" style="width:${(Math.min(v,goal)/mx*100).toFixed(2)}%"></i>`+
      (over?`<i class="tmem-c3o bad" style="left:${gx.toFixed(2)}%;width:${((Math.min(v,mx)-goal)/mx*100).toFixed(2)}%"></i>`:'')+
      `<i class="tmem-c3g" style="left:${gx.toFixed(2)}%"></i></span><b class="tmem-c3v${over?' bad':''}">${txt}</b>`;};
  return `<li class="tmem-c3r ${role}" data-role="${role}"><div class="tmem-c3h"><span class="tmem-c3t">${lab}</span>${chips}</div>`+
    `<div class="tmem-c3m" data-goal="mem"><span class="tmem-c3k">${tx('Memory','显存')}</span>${bar(e.peak/1e9,24,MX,GB(e.peak))}</div>`+
    `<div class="tmem-c3m" data-goal="time"><span class="tmem-c3k">${tx('Time','用时')}</span>${bar(e.t,CH.goal,TM,fixd(e.t,2))}</div></li>`;
}
/* the main gap, in one sentence computed from the plan */
function chGap(me,best){
  if(chSame(best,me.opt,me.rc,me.mb)) return tx('8-bit AdamW frees just enough memory for the second sequence’s activations and logits, and it adds no compute per token. With 2 sequences at once, the GPU sits idle less often.',
                                               '8 位 AdamW 腾出的显存，正好放下第二条序列的激活和 logits，又不让每个 token 多算。每次送 2 条，GPU 闲着的时间更少。');
  if(!me.fits){const c=compute(chState(me.opt,me.rc,me.mb)), al=c.acts+c.logits;
    return c.states>=al?tx(`The model states alone take ${GB(c.states)}, so what is left of the 24 GB cannot hold this step’s activations and logits.`,`光模型状态就占了 ${GB(c.states)}，24 GB 剩下的部分放不下这一步的激活和 logits。`)
      :tx(`With ${me.mb} sequences at once, the activations and logits take ${GB(al)}, more than the model states.`,`一次送 ${me.mb} 条序列，激活和 logits 就占了 ${GB(al)}，比模型状态还多。`);}
  if(!me.ok) return me.rc?tx(`Recomputation adds a third more compute per token, so with ${me.mb===1?'1 sequence':me.mb+' sequences'} at once the update is even slower than the reference.`,`重算让每个 token 多算三分之一，所以一次只送 ${me.mb} 条时，更新比参照还慢。`)
    :tx(`It fits, but with only ${me.mb===1?'1 sequence':me.mb+' sequences'} at once the GPU often sits idle, so the update is ${me.t>=1-1e-9?'as slow as':'barely faster than'} the reference.`,`放得下，可一次只送 ${me.mb} 条序列，GPU 常常闲着，所以更新${me.t>=1-1e-9?'和参照一样慢':'只比参照快一点'}。`);
  if(me.rc&&!best.rc) return tx(`Recomputation makes every token cost a third more compute. That costs more time than sending more sequences at once saves, so your update takes ${fixd(me.t,2)} of the reference time and the fastest plan’s ${fixd(best.t,2)}.`,
                                `重算让每个 token 多算三分之一。多花的这些时间，比一次多送几条省下的还多，所以你的方案用时 ${fixd(me.t,2)}，最快的方案只要 ${fixd(best.t,2)}。`);
  return tx(`Your update takes ${fixd(me.t,2)} of the reference time; the fastest plan takes ${fixd(best.t,2)}.`,`你的方案用时 ${fixd(me.t,2)}，最快的方案只要 ${fixd(best.t,2)}。`);
}
/* the one change that helps most (engine: chOneChange) */
function chNext(me,best){
  if(chSame(best,me.opt,me.rc,me.mb)) return '';
  const oc=chOneChange(me.opt,me.rc,me.mb), nd=CHS().filter(g=>me[g.k]!==best[g.k]).length;
  if(oc){const isB=chSame(best,oc.e.opt,oc.e.rc,oc.e.mb);
    return me.ok?tx(`The one change that helps most is ${chChange(oc.k,oc.v)}: an update then takes ${fixd(oc.e.t,2)}.`,`只改一项的话，${chChange(oc.k,oc.v)}最管用，每次更新的用时降到 ${fixd(oc.e.t,2)}。`)
      :tx(`The one change that helps most is ${chChange(oc.k,oc.v)}: that alone meets the goal, at ${fixd(oc.e.t,2)} of the reference time${isB?', and it gives the fastest plan':''}.`,
          `只改一项的话，${chChange(oc.k,oc.v)}最管用，光这一项就能达标，用时是参照的 ${fixd(oc.e.t,2)}${isB?'，这正是最快的方案':''}。`);}
  const ch=CHS().filter(g=>me[g.k]!==best[g.k]).map(g=>chChange(g.k,best[g.k]));
  const en=ch.length>1?ch.slice(0,-1).join(', ')+' and '+ch[ch.length-1]:ch[0], zh=ch.join('、');
  return me.ok?tx(`No single change makes it faster. The fastest plan makes ${nd} changes at once: ${en}.`,`只改一项快不起来。最快的方案同时改了 ${nd} 项：${zh}。`)
    :tx(`No single change meets the goal. The fastest plan makes ${nd} changes at once: ${en}.`,`只改一项还达不到目标。最快的方案同时改了 ${nd} 项：${zh}。`);
}
function chSummary(me,best){
  const isBest=chSame(best,me.opt,me.rc,me.mb), rows=[{e:me,role:isBest?'mebest':'me'}].concat(isBest?[]:[{e:best,role:'best'}]);
  const diff=CHS().filter(g=>me[g.k]!==best[g.k]).map(g=>g.k), same=CHS().filter(g=>!diff.includes(g.k));
  const MX=Math.max(24,...rows.map(x=>x.e.peak/1e9))*1.12, TM=Math.max(1,...rows.map(x=>x.e.t))*1.12;
  let h=`<div class="tmem-sum"><div class="tmem-sumcap">${isBest?tx('Your plan against the two goals','你的方案离两个目标多远'):tx('Your plan and the fastest plan, against the two goals','你的方案和最快的方案，离两个目标各有多远')}</div>`;
  if(!isBest&&same.length) h+=`<p class="small tmem-same">${tx(same.map(g=>g.en(me)).join(' '),'两种方案'+same.map(g=>g.zh(me)).join('，')+'。')}</p>`;
  h+=`<ul class="tmem-c3">${rows.map(x=>chRow(x.e,x.role,diff,best,MX,TM)).join('')}</ul>`+
    `<div class="tmem-c3leg small"><span><i class="sw tmem-sw-cap" aria-hidden="true"></i>${tx('dashed: the goal','虚线是目标')}</span><span><i class="sw tmem-sw-over" aria-hidden="true"></i>${tx('red: beyond the goal','红色是超出目标的部分')}</span></div>`;
  h+=`<p class="tmem-gap">${chGap(me,best)}</p>`;
  const nx=chNext(me,best); if(nx) h+=`<p class="tmem-next">${nx}</p>`;
  const ps=chPass();
  h+=`<details class="tmem-all"><summary>${tx(`See all settings that pass (${ps.length})`,`看全部可行的设置（${ps.length} 种）`)}</summary><div class="tmem-allb"><ul class="tmem-alll">`+
    ps.map(e=>{const mine=chSame(e,me.opt,me.rc,me.mb), fast=chSame(e,best.opt,best.rc,best.mb);
      return `<li${mine?' class="me"':''}>${esc(chName(e.opt,e.rc,e.mb))}${tx(': peak ','：峰值 ')}${GB(e.peak)}${tx(', time ','，用时 ')}${fixd(e.t,2)}${fast?tx(' (fastest)','（最快）'):''}${mine?tx(' (your plan)','（你的方案）'):''}</li>`;}).join('')+`</ul>`+
    tx(`<p class="small">The other ${chAll().length-ps.length} plans either do not fit in 24 GB or take longer than ${fixd(CH.goal,2)}. Recomputation also frees memory, enough for 4 sequences at once, but every token then costs a third more compute, so that plan is slower than the fastest one.</p>`,
       `<p class="small">其余 ${chAll().length-ps.length} 种，不是放不下 24 GB，就是用时超过 ${fixd(CH.goal,2)}。重算也能腾出显存，够一次放 4 条序列。可每个 token 要多算三分之一，所以那种方案比最快的慢。</p>`)+`</div></details>`;
  return h+`</div>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.opt,S.rc,S.mb), best=chBest(), isBest=chSame(best,me.opt,me.rc,me.mb);
  let h;
  if(me.ok&&isBest) h=tx(`<p class="okline tmem-mt"><b>Goal met: the step peaks at ${GB(me.peak)}, and an update takes ${fixd(me.t,2)} of the reference time.</b> Of the plans that meet the goal, this is the fastest.</p>`,
                         `<p class="okline tmem-mt"><b>达成目标：峰值 ${GB(me.peak)}，每次更新的用时是参照的 ${fixd(me.t,2)}。</b>达标的方案里，这是最快的。</p>`);
  else if(me.ok) h=tx(`<p class="okline tmem-mt"><b>Goal met: the step peaks at ${GB(me.peak)}, and an update takes ${fixd(me.t,2)} of the reference time.</b> But a faster plan also meets it.</p>`,
                      `<p class="okline tmem-mt"><b>达标了：峰值 ${GB(me.peak)}，每次更新的用时是参照的 ${fixd(me.t,2)}。</b>不过还有更快的方案。</p>`);
  else h=tx(`<p class="badline tmem-mt"><b>Not yet: ${me.fits?`it fits (${GB(me.peak)}), but an update takes ${fixd(me.t,2)} of the reference time, over ${fixd(CH.goal,2)}`:`the step peaks at ${GB(me.peak)}, over the 24 GB`}.</b></p>`,
            `<p class="badline tmem-mt"><b>还没达到目标：${me.fits?`放得下（${GB(me.peak)}），但每次更新的用时是参照的 ${fixd(me.t,2)}，超过了 ${fixd(CH.goal,2)}`:`一步要 ${GB(me.peak)}，超过了 24 GB`}。</b></p>`)+
      tx('<p class="small">Hint: the more sequences go into the GPU at once, the less it sits idle and the faster the update. But each extra sequence brings its own activations and logits. Two things free memory for them: storing Adam’s m and v in 8 bits, or recomputing the activations. Which of the two adds no compute per token?</p>',
         '<p class="small">提示：一次送进 GPU 的序列越多，GPU 越不容易闲着，更新越快。可每多送一条，就多一份激活和 logits。腾出显存的办法有两个：把 Adam 的 m 和 v 压成 8 位，或者重算激活。哪一个不让每个 token 多算？</p>')+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the fastest plan','直接看最快的方案')}</button>`);
  if(!chReveal) return h;
  h+=chSummary(me,best);
  h+=tx(`<p class="small muted">Note: this challenge counts only memory and compute time. It does not check how 8-bit AdamW affects training; the bitsandbytes authors say their 8-bit optimizers keep 32-bit performance (${cite(SRC.bnb,'README')}), but compare the loss curves on your own task before you switch. A real run also needs some memory for the CUDA context and the allocator, which this lab leaves out.</p>`,
        `<p class="small muted">注意：这个挑战只算显存和计算用时，没有考虑 8 位 AdamW 对训练效果的影响。bitsandbytes 的作者说他们的 8 位优化器能保持 32 位的效果（${cite(SRC.bnb,'README')}），换之前还是要在自己的任务上对比 loss 曲线。真实训练还要给 CUDA 上下文和显存分配器留一点显存，这个 Lab 没有算。</p>`);
  return h;
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Predict first, then watch one training step and see what fills the GPU: the four extra numbers every parameter carries, the activations, and what recomputation and smaller micro-batches can and cannot save. Start here if this is new to you.',
            '一共 6 个部分。先预测，再看训练一步时显存里装了什么：每个参数多带的四样东西、激活，以及重算和分份能省什么、省不了什么。第一次学从这里开始。'),
  sandbox:tx(`Four real situations, from fine-tuning the 8B on one H100 to training the 70B. Pick one to see what fills the memory.`,'四个真实情况，从在一张 H100 上微调 8B，到训练 70B。选一个，看显存都被什么占了。'),
  challenge:tx(`Train Llama${NB}3.2${NB}1B from scratch on one 24 GB RTX 4090. Choose the optimizer, recomputation and the sequences sent at once so that the step fits and each update is as fast as you can make it.`,
               `在一张 24 GB 的 RTX 4090 上从头训练 Llama${NB}3.2${NB}1B。选好优化器、要不要重算、每次送几条序列，让这一步放得下，同时每次更新尽量快。`)
});
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> every byte on this page is counted from the model’s config and the accounting below; ${J('math','“Show the math”')} works it out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Model states: every parameter has a BF16 weight (2 bytes), an FP32 gradient (4 bytes) and an FP32 copy of the weight (4 bytes). The bytes of m and v depend on the optimizer:<ul class="tmem-poplist"><li>AdamW: m and v, 4 bytes each, 18 bytes in all</li><li>8-bit AdamW: m and v, 1 byte each, 12 bytes in all</li><li>SGD with momentum: m only, 4 bytes, 14 bytes in all</li></ul>This is how Megatron-LM (${cite(SRC.megdo,'dist_optimizer.md')}) and Hugging Face (${cite(SRC.hfmem,'model_memory_anatomy.md')}) count them. The ZeRO paper keeps both the weights and the gradients in 16 bits and counts 16 bytes. Training wholly in BF16, with no FP32 copy, needs only 8 bytes per parameter (${cite(SRC.mle,'ml-engineering')}), and the 8B’s model states would be ${GB(MOD['8b'].P*8)}, under 80 GB. But the optimizer then needs Kahan summation or stochastic rounding so that small updates are not lost. This lab does not model it.</li>`+
 `<li>Activations: counted tensor by tensor for one Llama layer in BF16 with FlashAttention and fused RMSNorm, no dropout (the list is in “Show the math”), the same method as Korthikanti et al. (arXiv 2205.05198). Logits: one FP32 copy, tokens × 128,256 × 4 bytes.</li>`+
 `<li>Recomputation keeps each layer’s input; while a layer is recomputed its activations come back for a moment, which is less than the logits that were freed just before, so the peak stays at the end of the forward pass.</li>`+
 `<li>The curve’s time axis is compute: the forward pass 1, the backward pass 2, a recomputed backward pass 3, the optimizer step a short fixed stretch. It is a picture of the formula.</li></ul></div>`+
 `<div><div class="eyebrow">Left out</div><ul>`+
 `<li>The CUDA context, the allocator’s cached but unused memory and fragmentation, temporary buffers in the optimizer step, and the inputs of the final RMSNorm and LM head (under 0.1 GB at 4,096 tokens for the 8B).</li>`+
 `<li>The gradients are counted for the whole step, as gradient accumulation needs. PyTorch’s default frees them between updates and allocates them during the first backward pass, which lowers the first micro-batch’s peak a little.</li>`+
 `<li>A GPU’s “80 GB” is read as 80 × 10⁹ bytes, and the RTX 4090’s “24 GB” as 24 × 10⁹. Read as GiB (85.9 GB and 25.8 GB), the 8B and the 70B still never fit on one H100, and the challenge’s fastest plan stays the same.</li>`+
 `<li>Compute per token (6N, 8N) leaves out attention’s own arithmetic.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Model shapes: config.json of each model and Meta’s ${cite(SRC.sku,'<code>models/sku_list.py</code>')}; parameter counts as in the rest of the site.</li>`+
 `<li>8-bit optimizers: ${cite(SRC.bnb,'bitsandbytes README')} (block-wise quantization). Recomputation: ${cite(SRC.ckpt,'<code>torch/utils/checkpoint.py</code>')}, Megatron-LM ${cite(SRC.meg,'<code>recompute_granularity</code>')}.</li>`+
 `<li>Bytes per parameter and its variants: ${cite(SRC.megdo,'Megatron-LM dist_optimizer.md')}, ${cite(SRC.hfmem,'Hugging Face model_memory_anatomy.md')}, ${cite(SRC.mle,'ml-engineering training/performance')}.</li><li>Three papers, cited by title only. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (arXiv 1910.02054); Reducing Activation Recomputation in Large Transformer Models (arXiv 2205.05198); 8-bit Optimizers via Block-wise Quantization (arXiv 2110.02861).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上的每个字节，都由模型的 config 和下面的计算方式算出。${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>模型状态里，每个参数都有 BF16 权重 2 字节、FP32 梯度 4 字节和 FP32 权重副本 4 字节。m 和 v 的字节数取决于优化器：<ul class="tmem-poplist"><li>AdamW：m 和 v 各 4 字节，一共 18 字节</li><li>8 位 AdamW：m 和 v 各 1 字节，一共 12 字节</li><li>带 momentum 的 SGD：只有 m，4 字节，一共 14 字节</li></ul>这和 Megatron-LM（${cite(SRC.megdo,'dist_optimizer.md')}）、Hugging Face（${cite(SRC.hfmem,'model_memory_anatomy.md')}）的计算方式一致。ZeRO 论文把权重和梯度都用 16 位存，算 16 字节。完全用 BF16 训练、不留 FP32 权重副本时，每个参数只要 8 字节（${cite(SRC.mle,'ml-engineering')}），8B 的模型状态只有 ${GB(MOD['8b'].P*8)}，不到 80 GB。但这时优化器要用 Kahan 求和或随机舍入，小的更新才不会丢。这个 Lab 没有建模这种做法。</li>`+
 `<li>激活：按 Llama 的一层逐个张量数，BF16，用 FlashAttention 和融合的 RMSNorm，没有 dropout（清单在“展开公式”里），和 Korthikanti 等人（arXiv 2205.05198）的数法相同。logits：一份 FP32，token 数 × 128,256 × 4 字节。</li>`+
 `<li>重算时每层留下输入；某一层重算的那一刻，它的激活会短暂回来，但比刚释放的 logits 小，所以峰值仍在前向结束的那一刻。</li>`+
 `<li>曲线的横轴按计算量画：前向 1，反向 2，重算时的反向 3，优化器更新取一小段固定长度。它是公式的示意。</li></ul></div>`+
 `<div><div class="eyebrow">没有算的</div><ul>`+
 `<li>CUDA 上下文、显存分配器缓存着没用的部分和碎片、优化器更新时的临时缓冲区，以及最后的 RMSNorm 和 LM head 的输入（8B 在 4,096 个 token 时不到 0.1 GB）。</li>`+
 `<li>梯度按整步都在算，梯度累积时本来就要这样。PyTorch 默认在两次更新之间释放梯度，第一趟反向时再分配，所以第一个 micro-batch 的峰值会略低一点。</li>`+
 `<li>H100 的“80 GB”按 80 × 10⁹ 字节算，RTX 4090 的“24 GB”按 24 × 10⁹ 字节算。如果按 GiB 算（85.9 GB 和 25.8 GB），8B 和 70B 照样放不进一张 H100，挑战里最快的方案也不变。</li>`+
 `<li>每个 token 的计算（6N、8N）没有算注意力自己的那部分。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>模型形状：各模型的 config.json 和 Meta 的 ${cite(SRC.sku,'<code>models/sku_list.py</code>')}；参数量和全站一致。</li>`+
 `<li>8 位优化器：${cite(SRC.bnb,'bitsandbytes README')}（分块量化）。重算：${cite(SRC.ckpt,'<code>torch/utils/checkpoint.py</code>')}，Megatron-LM 的 ${cite(SRC.meg,'<code>recompute_granularity</code>')}。</li>`+
 `<li>每个参数的字节和几种变体：${cite(SRC.megdo,'Megatron-LM dist_optimizer.md')}，${cite(SRC.hfmem,'Hugging Face model_memory_anatomy.md')}，${cite(SRC.mle,'ml-engineering training/performance')}。</li><li>下面三篇论文只引了标题。ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（arXiv 1910.02054）；Reducing Activation Recomputation in Large Transformer Models（arXiv 2205.05198）；8-bit Optimizers via Block-wise Quantization（arXiv 2110.02861）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('estimate the GPU memory a training run needs from a config, and say which trick shrinks which part.','从模型的 config 估出训练要多少显存，并说清每种省显存的办法省的是哪一项。');
const EXITS=()=>{
  const b=C0(), big=compute({...BASE,m:'405b'}), a405=compute({...BASE,m:'405b'});
  const nG=Math.ceil(big.states/80e9), ratio=a405.acts/b.acts;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does training with AdamW take 18 bytes per parameter when inference takes 2?','为什么用 AdamW 训练时每个参数要 18 字节，推理只要 2 字节？'),
   o:[tx('Training keeps four more numbers per parameter: an FP32 gradient (4 bytes), an FP32 copy of the weights (4 bytes), and Adam’s m and v (4 bytes each).','训练时每个参数还要多放四样：FP32 梯度（4 字节）、FP32 权重副本（4 字节），以及 Adam 的 m 和 v（各 4 字节）。'),
      tx('Training stores every weight in FP32 instead of BF16, and FP32 takes 9 times the bytes of BF16.','训练时每个权重都改用 FP32 存，FP32 占的字节是 BF16 的 9 倍。'),
      tx('Training keeps a copy of the weights for every layer’s forward pass, so the bytes grow with the number of layers.','训练时每一层的前向都要留一份权重拷贝，所以字节数跟着层数涨。')],a:0,
   n:[null,tx('FP32 takes 4 bytes, only twice BF16’s 2. The 18 bytes come from five numbers per parameter, not from one wider number.','FP32 是 4 字节，只是 BF16 的 2 倍。18 字节来自每个参数存的五个数，不是把一个数变宽了。'),
     tx('The forward pass reads the same weights in every step; nothing copies them per layer. What grows with the layers is the activations, which are counted separately.','前向每一步读的都是同一份权重，不会每层拷贝一份。跟着层数涨的是激活，它另算。')],
   why:tx('Inference holds only the weights. Training also needs each parameter’s FP32 gradient, an FP32 copy of the weights so that small updates are not rounded away, and Adam’s m and v. That adds up to 2 + 4 + 4 + 4 + 4 = 18 bytes.','推理只放权重。训练还要放每个参数的 FP32 梯度、FP32 权重副本（免得小的更新被舍入掉），以及 Adam 的 m 和 v。加起来是 2 + 4 + 4 + 4 + 4 = 18 字节。')},
  {q:tx('8-bit AdamW takes the model states from 18 to 12 bytes per parameter. Which bytes does it save?','8 位 AdamW 把模型状态从每个参数 18 字节降到 12 字节。它省的是哪几个字节？'),
   o:[tx('The 8 bytes of Adam’s m and v become 2, each stored in 1 byte. The BF16 weights, the FP32 gradients and the FP32 copy stay as they are.','Adam 的 m 和 v 从 8 字节变成 2 字节，两者各用 1 字节存。BF16 权重、FP32 梯度和 FP32 权重副本都不变。'),
      tx('The FP32 copy is dropped, and the gradients go from 4 bytes to 2: 6 bytes in all.','FP32 权重副本去掉了，梯度也从 4 字节压到 2 字节，一共省 6 字节。'),
      tx('All five numbers shrink a little, each by about a third.','五个数都压小一点，每个大约少三分之一。')],a:0,
   n:[null,tx('The FP32 copy and the FP32 gradients stay. An 8-bit optimizer changes only how Adam’s m and v are stored.','FP32 权重副本和 FP32 梯度都还在。8 位优化器只改 Adam 的 m 和 v 怎么存。'),
     tx('Only Adam’s m and v change, each from 4 bytes to 1. The other three numbers keep their size.','改的只有 Adam 的 m 和 v，各从 4 字节变成 1 字节。另外三个数大小不变。')],
   why:tx('8-bit AdamW changes only how Adam’s m and v are stored. bitsandbytes stores them as 8-bit integers in small blocks, each block with its own scale, and expands them back to FP32 only while it computes the update.','8 位 AdamW 只改 Adam 的 m 和 v 怎么存。bitsandbytes 把它们分成小块，存成 8 位整数，每块带一个缩放系数，只在算更新时临时还原成 FP32。')}],
 transfer:[
  {q:tx(`On this page, ${b.m.name}’s model states took ${GB(b.states)}, so it needs at least 2 H100s of 80 GB just to hold them. ${MOD['405b'].name} has about 50 times as many parameters. With AdamW, about how many 80 GB GPUs does it need just for its model states?`,
        `这一页里，${b.m.name} 的模型状态是 ${GB(b.states)}，所以光放它们就要至少 2 张 80 GB 的 H100。${MOD['405b'].name} 的参数大约是它的 50 倍。用 AdamW 时，光放 405B 的模型状态，大约要多少张 80 GB 的卡？`),
   o:[tx('About 10','约 10 张'),tx('About 90','约 90 张'),tx('About 900','约 900 张')],a:1,
   why:tx(`The model states grow in step with the parameters: about 50 times ${GB(b.states)} is about 7.2 TB, and 7.2 TB ÷ 80 GB is about 90 GPUs (exactly: ${GB(big.states)}, at least ${nG} GPUs). That is before any activation. So in practice, models this size are trained with their model states split across many GPUs.`,
          `模型状态和参数量成正比。${GB(b.states)} 的约 50 倍，大约是 7.2 TB。7.2 TB ÷ 80 GB，大约 90 张（准确地说是 ${GB(big.states)}，至少 ${nG} 张）。这还没算激活，所以训练这么大的模型，实际上都把模型状态分到很多张卡上。`)},
  {q:tx(`On this page, ${b.m.name} keeps ${GB(b.acts)} of activations for 4,096 tokens without recomputation. ${MOD['405b'].name}’s hidden size is 4 times the 8B’s, and it has about 4 times as many layers. For the same 4,096 tokens, about how much more do its activations take?`,
        `这一页里，${b.m.name} 处理 4,096 个 token、不重算时，激活是 ${GB(b.acts)}。${MOD['405b'].name} 的隐藏维度是 8B 的 4 倍，层数也大约是 4 倍。同样 4,096 个 token，它的激活大约是 8B 的几倍？`),
   o:[tx('About 4 times','约 4 倍'),tx('About 15 times','约 15 倍'),tx('About 60 times','约 60 倍')],a:1,
   why:tx(`What one token leaves in one layer grows with the hidden size, and there is one such set per layer, so the activations grow with both: about 4 × 4 = 16 times. The 405B’s MLP is a little narrower relative to its hidden size, so it comes to about ${fixd(ratio,0)} times (${GB(a405.acts)}).`,
          `一个 token 在一层里留下的字节和隐藏维度成正比，而每一层都有一份，所以激活跟两者都成正比：大约 4 × 4 = 16 倍。405B 的 MLP 相对隐藏维度窄一点，所以实际约是 ${fixd(ratio,0)} 倍（${GB(a405.acts)}）。`)}],
 practice:[
  {id:'distributed-training',d:tx('Part 2, the memory plan: count the bytes per parameter for a 70B model trained with Adam in mixed precision. Then work out why a model this size has these states split across many GPUs, and where recomputation fits.','第 2 项“显存方案”：先数一数 70B 模型用 Adam 混合精度训练时，每个参数占多少字节。再想清楚这么大的模型为什么要把这些状态分到多张卡上，以及重算该用在哪里。')},
  {id:'mlp-backprop',d:tx('“Memory and activation checkpointing” in the solution’s extensions: why the backward pass needs every cached activation, and how checkpointing trades that memory for recomputation.','解答末尾的“Memory and activation checkpointing”：反向为什么需要前向缓存的每个激活，以及重算怎样用计算换显存。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Training memory','训练显存'),title:tx(`An 8B model’s weights take 16 GB: why does training it not fit on an 80 GB GPU?`,'8B 模型的权重只有 16 GB，为什么训练它，80 GB 的卡都放不下？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),descs:DESCS(),mode,
    primer:tx('Primer: optimizers','前置知识：优化器'),primerPop:tx(`<p>This lab uses three things explained in other labs:</p><ul><li>Adam’s m and v, kept for every parameter: see ${labLink('optimizers')}</li><li>Why training keeps an FP32 copy of the weights: see ${labLink('number-formats')}</li><li>Why the forward pass keeps intermediate results: see ${labLink('backprop')}</li></ul>`,
      `<p>这个 Lab 用到三个 Lab 讲过的东西：</p><ul><li>Adam 给每个参数记的 m 和 v：见${labLink('optimizers')}</li><li>训练为什么要留一份 FP32 权重副本：见${labLink('number-formats')}</li><li>前向为什么要留中间结果：见${labLink('backprop')}</li></ul>`)});
}
function stageHTML(){
  return `<div class="panel stage tmem-stage" id="tmem-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="tmem-st-title"></div><div class="meter-sub" id="tmem-st-sub"></div></div><div class="stage-actions"><span class="status" id="tmem-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph tmem-graph" id="tmem-graph">`+
    `<div class="tmem-vis" id="tmem-vis"><div class="tmem-colbox"><div class="tmem-cap" id="tmem-col-cap"></div><div class="tmem-col" id="tmem-col"></div></div>`+
    `<div class="tmem-curvebox"><div class="tmem-cap" id="tmem-cv-cap"></div><div class="tmem-curve" id="tmem-curve"></div><div class="tmem-leg" id="tmem-cv-leg"></div><div class="tmem-player" id="tmem-player"></div></div></div>`+
   `</div>`+
   `<div class="metrics tmem-metrics" id="tmem-metrics"></div>`+
   `<details class="math" id="tmem-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="tmem-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="tmem-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="tmem-lock" hidden></p>`+
   `<div class="lockable" id="tmem-ctl-m-wrap">${A.ui.seg({id:'tmem-ctl-m',key:'m',label:tx('Model','模型'),options:[['8b',MOD['8b'].name],['70b',MOD['70b'].name]],value:S.m,note:tx('The GPU is one H100 with 80 GB in Guided and Scenarios.','分步引导和应用场景里，GPU 都是一张 80 GB 的 H100。')})}</div>`+
   `<div class="lockable" id="tmem-ctl-opt-wrap">${A.ui.seg({id:'tmem-ctl-opt',key:'opt',label:tx('Optimizer','优化器'),options:OPTS.map(o=>[o,optName(o)]),value:S.opt})}</div>`+
   `<div class="lockable" id="tmem-ctl-seq-wrap">${seqCtl('tmem-ctl-seq',false)}</div>`+
   `<div class="lockable" id="tmem-ctl-mb-wrap">${mbCtl('tmem-ctl-mb',false)}</div>`+
   `<div class="lockable" id="tmem-ctl-acc-wrap">${A.ui.seg({id:'tmem-ctl-acc',key:'acc',label:tx('Passes per update','每次更新跑几趟'),options:ACCS.map(k=>[String(k),String(k)]),value:String(S.acc),note:tx('Each pass runs the forward and backward computation once, and the gradients are added up. The parameters are updated once, after all the passes. More passes use more sequences per update and make it slower. Each pass sends the same sequences into the GPU, so the peak stays the same.','每趟做一次前向和反向，梯度加在一起，跑完这几趟才更新一次参数。趟数越多，一次更新用到的序列越多，也越慢。每趟送进 GPU 的序列不变，所以峰值不变。')})}</div>`+
   `<div class="lockable" id="tmem-ctl-rc-wrap">${A.ui.seg({id:'tmem-ctl-rc',key:'rc',label:tx('Recomputation','重算'),options:[['0',tx('Off','不重算')],['1',tx('Every layer','每一层都重算')]],value:String(S.rc)})}</div>`+
   `<div class="lockable" id="tmem-ctl-note"><p class="ctl-note">${tx('Logits are counted as one FP32 copy (4 bytes each). This is an assumption: the loss needs them in FP32 for accuracy, and some frameworks briefly hold a BF16 copy and the logits’ gradient as well, so real peaks can be higher. The more tokens and the bigger the vocabulary (128,256 tokens for Llama 3), the larger the logits.','logits 按一份 FP32（每个 4 字节）算，这是假设。算 loss 时要用 FP32 才够准。有的框架还会短暂多放一份 BF16 的 logits 和它的梯度，所以实际的峰值可能更高。token 越多、词表越大，logits 就越大（Llama 3 的词表有 128,256 个 token）。')}</p></div>`+
   `<div class="lockable" id="tmem-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`def train_memory(cfg, bpp=18, tokens=4096, recompute=False):
    """Peak GPU memory of one training step, in bytes.
    bpp: 18 for AdamW (2 w + 4 g + 4 fp32 copy + 4 m + 4 v),
    12 for 8-bit AdamW, 14 for SGD with momentum."""
    d, f = cfg["hidden_size"], cfg["intermediate_size"]
    hd = cfg.get("head_dim", d // cfg["num_attention_heads"])
    q = cfg["num_attention_heads"] * hd
    kv = cfg["num_key_value_heads"] * hd
    attn = 2 * d + q + 2 * kv + q    # norm in/out, q, k, v, out
    mlp = 2 * d + 4 * f              # norm in/out, gate, silu, up, x
    per_layer = 2 * (attn + mlp)     # BF16: 2 bytes each
    if recompute:
        per_layer = 2 * d            # keep only the layer input
    states = bpp * cfg["n_params"]
    acts = tokens * cfg["num_hidden_layers"] * per_layer
    logits = tokens * cfg["vocab_size"] * 4      # one FP32 copy
    return states + acts + logits

llama31_8b = dict(hidden_size=4096, intermediate_size=14336,
    num_attention_heads=32, num_key_value_heads=8,
    num_hidden_layers=32, vocab_size=128256, n_params=8030261248)
print(train_memory(llama31_8b) / 1e9)                 # 168.7
print(train_memory(llama31_8b, 12, recompute=True) / 1e9)   # 99.5`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="tmem-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="tmem-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A memory estimator from a config.json: model states, activations counted tensor by tensor, and the logits.','从 config.json 估出训练显存：模型状态、逐个张量数出的激活，再加上 logits。')}</p><pre class="code" id="tmem-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="tmem-prod-card"><h3>${A.t('cProd')}</h3><ul class="tmem-prod">`+
   tx(`<li>To see the real curve, PyTorch records every allocation with <code>torch.cuda.memory._record_memory_history()</code> and saves it with <code>_dump_snapshot()</code>; the snapshot opens at pytorch.org/memory_viz (${cite(SRC.snap,'torch_cuda_memory.md')}). The rise in the forward pass and the fall in the backward pass in this lab’s chart is what such a trace shows.</li>`+
      `<li>DeepSpeed’s docs give estimators for the model states under each ZeRO stage, and say that activations come on top and can be as large as everything else for long sequences (${cite(SRC.dsmem,'memory.rst')}). Its tables also give the memory when the optimizer state is moved to CPU memory (<code>offload_optimizer=cpu</code>, ZeRO-Offload). What each ZeRO stage splits: ${labLink('parallelism')}.</li>`+
      `<li>Recomputation in PyTorch is <code>torch.utils.checkpoint</code>: the tensors inside the checkpointed function are not kept, and the function runs again in the backward pass (${cite(SRC.ckpt,'checkpoint.py')}). Megatron-LM offers <code>recompute_granularity</code> “full” (the whole layer, as in this lab) or “selective”, which by default recomputes only the core of attention, the part with much memory and little compute (${cite(SRC.meg,'transformer_config.py')}).</li>`+
      `<li>8-bit optimizers: bitsandbytes’ <code>bnb.optim</code> stores Adam’s m and v in 8 bits with block-wise quantization (${cite(SRC.bnb,'README')}).</li>`+
      `<li>torchtune fully fine-tunes Llama${NB}3.1${NB}8B on one 24 GB RTX 4090, with a peak of 18.9 GiB (${cite(SRC.tt,'README')}). Its config (${cite(SRC.ttcfg,'8B_full_single_device.yaml')}) does four things. It keeps the weights in BF16 only, with no FP32 copy. It uses bitsandbytes’ <code>PagedAdamW8bit</code>, with m and v in 8 bits. The BF16 weights and the 8-bit m and v come to ${GB(MOD['8b'].P*4)}, more than the card holds. When the GPU runs out, the paged optimizer moves m and v page by page to CPU memory (${cite(SRC.bnbx,'optimizers.mdx')}). It updates each parameter as soon as its gradient is ready (<code>optimizer_in_bwd</code>), so the full set of gradients is never held at once. And it recomputes the activations.</li>`,
      `<li>想看真实的曲线：PyTorch 用 <code>torch.cuda.memory._record_memory_history()</code> 记下每一次分配，用 <code>_dump_snapshot()</code> 存下来，再到 pytorch.org/memory_viz 打开（${cite(SRC.snap,'torch_cuda_memory.md')}）。这个 Lab 的图里前向上升、反向下降的形状，就是这类记录里看到的样子。</li>`+
      `<li>DeepSpeed 的文档给出了各个 ZeRO 阶段下模型状态的估算器，并说明激活要另外加，长序列时可能和其余部分加起来一样大（${cite(SRC.dsmem,'memory.rst')}）。它的表里也给出了把优化器状态挪到 CPU 内存时的用量（<code>offload_optimizer=cpu</code>，即 ZeRO-Offload）。各个 ZeRO 阶段分的是什么，见${labLinkZ('parallelism')}。</li>`+
      `<li>PyTorch 里的重算是 <code>torch.utils.checkpoint</code>：被包起来的函数里的张量不保存，反向时把函数再跑一遍（${cite(SRC.ckpt,'checkpoint.py')}）。Megatron-LM 的 <code>recompute_granularity</code> 可以选 “full”（整层重算，和这个 Lab 一样）或 “selective”。selective 默认只重算注意力的核心部分，这部分占显存多、计算少（${cite(SRC.meg,'transformer_config.py')}）。</li>`+
      `<li>8 位优化器：bitsandbytes 的 <code>bnb.optim</code> 用分块量化，把 Adam 的 m 和 v 存成 8 位（${cite(SRC.bnb,'README')}）。</li>`+
      `<li>torchtune 在一张 24 GB 的 RTX 4090 上全量微调 Llama${NB}3.1${NB}8B，峰值 18.9 GiB（${cite(SRC.tt,'README')}）。它的配置（${cite(SRC.ttcfg,'8B_full_single_device.yaml')}）做了四件事。权重只用 BF16 存，不留 FP32 权重副本。优化器用 bitsandbytes 的 <code>PagedAdamW8bit</code>，m 和 v 存成 8 位。BF16 权重加上 8 位的 m 和 v 就有 ${GB(MOD['8b'].P*4)}，这张卡放不下。显存不够时，分页优化器会把 m 和 v 一页页挪到 CPU 内存里（${cite(SRC.bnbx,'optimizers.mdx')}）。每个参数的梯度一算好，就马上更新这个参数（<code>optimizer_in_bwd</code>），所以从不同时存着全部梯度。最后，激活也用重算。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.zero,'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models')}<span>${tx('Rajbhandari et al., 2019 (arXiv 1910.02054): the model-state accounting of mixed-precision Adam, and how to split it across GPUs.','Rajbhandari 等，2019（arXiv 1910.02054）：混合精度 Adam 的模型状态怎么算，以及怎样分到多张卡上。')}</span></li>`+
   `<li>${cite(SRC.sel,'Reducing Activation Recomputation in Large Transformer Models')}<span>${tx('Korthikanti et al., 2022 (arXiv 2205.05198): counts the activations of a Transformer layer tensor by tensor and introduces selective recomputation.','Korthikanti 等，2022（arXiv 2205.05198）：逐个张量数出 Transformer 一层的激活，并提出选择性重算。')}</span></li>`+
   `<li>${cite(SRC.b8,'8-bit Optimizers via Block-wise Quantization')}<span>${tx('Dettmers et al., 2021 (arXiv 2110.02861): the method behind bitsandbytes’ 8-bit optimizers.','Dettmers 等，2021（arXiv 2110.02861）：bitsandbytes 8 位优化器背后的方法。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="tmem-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="tmem-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="tmem-app m2" id="tmem-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="tmem-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much memory each GPU needs.','选一个场景，看看每张卡要放多少显存。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="tmem-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const u=CH.util;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Train a 1B model on one 24 GB RTX 4090, as fast as it fits`,'在一张 24 GB 的 RTX 4090 上训练 1B 模型：放得下，还要快')}</h3>`+
   `<p>${tx(`Your team has one RTX 4090 with 24 GB and wants to train ${MOD['1b'].name} (${big(MOD['1b'].P)} parameters) from scratch on sequences of 2,048 tokens. Every parameter update uses the gradients of 32 sequences, which may be worked out in several passes and added up.`,
            `你们只有一张 24 GB 的 RTX 4090，要从头训练 ${MOD['1b'].name}（${big(MOD['1b'].P)} 个参数），每条序列 2,048 个 token。每次更新参数要用 32 条序列的梯度，可以分几趟算完再加起来。`)}</p>`+
   `<p>${tx(`The reference is the simplest plan: no recomputation and 1 sequence at once. Its time per update counts as 1.00. Your plan must fit one step in 24 GB and take at most ${fixd(CH.goal,2)} per update, at least ${Math.round((1-CH.goal)*100)}% faster than the reference. Among the plans that meet the goal, faster is better.`,
            `参照是最简单的跑法：不重算，每次送 1 条序列。它一次更新的用时记作 1.00。你的方案要一步放得下 24 GB，每次更新的用时不超过 ${fixd(CH.goal,2)}，也就是比参照至少快 ${Math.round((1-CH.goal)*100)}%。达标的方案里，越快越好。`)}</p>`+
   `<ul class="tmem-asm small"><li>${tx(`The more sequences go into the GPU at once, the bigger the matrix multiplications and the less the GPU sits idle. Without recomputation, this lab assumes an ${DEF('MFU',TIP.mfu())} of ${Math.round(u[1]*100)}%, ${Math.round(u[2]*100)}%, ${Math.round(u[4]*100)}% and ${Math.round(u[8]*100)}% with 1, 2, 4 and 8 sequences at once.`,`一次送进 GPU 的序列越多，矩阵乘法越大，GPU 越不容易闲着。这个 Lab 假设不重算时，一次送 1、2、4、8 条序列，${DEF('MFU',TIP.mfu())} 分别是 ${Math.round(u[1]*100)}%、${Math.round(u[2]*100)}%、${Math.round(u[4]*100)}%、${Math.round(u[8]*100)}%。`)} ${DEF(tx('Why assumed?','为什么是假设？'),TIP.mfuWhy())}</li><li>${tx('With recomputation, every token costs 2N more operations (N is the number of parameters), 8N instead of 6N. The GPU computes at the same speed, so the time grows by a third.','重算时，每个 token 多做 2N 次运算（N 是参数量），从 6N 变成 8N。GPU 干活的速度不变，所以用时多三分之一。')}</li></ul>`+
   A.ui.seg({id:'tmem-ic-cho',key:'cho',label:tx('Optimizer','优化器'),options:CH.opts.map(o=>[o,optName(o)]),value:S.opt,inline:true})+
   A.ui.seg({id:'tmem-ic-chr',key:'chr',label:tx('Recomputation','重算'),options:[['0',tx('Off','不重算')],['1',tx('Every layer','每一层都重算')]],value:String(S.rc),inline:true})+
   A.ui.seg({id:'tmem-ic-chm',key:'chm',label:tx('Sequences at once','每次送进 GPU 几条'),options:CH.mbs.map(k=>[String(k),String(k)]),value:String(S.mb),inline:true})+
   `<div class="goal mono tmem-goal" id="tmem-ch-g1"><span>${tx('Peak of one step','一步的峰值')}</span><span><b id="tmem-ch-p"></b>${tx(' (goal: at most 24 GB)','（目标：不超过 24 GB）')}</span></div>`+
   `<div class="goal mono tmem-goal" id="tmem-ch-g2"><span>${tx('Time per update','每次更新的用时')}</span><span><b id="tmem-ch-t"></b>${tx(` (goal: at most ${fixd(CH.goal,2)})`,`（目标：不超过 ${fixd(CH.goal,2)}）`)}</span></div>`+
   submit+`<div id="tmem-ch-res" class="tmem-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},states:{en:'Let the step finish playing to continue (“Fast” takes 10 seconds).',zh:'等这一步播完才能继续（选“快”只要 10 秒）。'},
    acts:{en:'Change the length or the sequences at once to continue.',zh:'改一下序列长度或一次送的条数，才能继续。'},rc:{en:'Pick “Recompute every layer” to continue.',zh:'选“每一层都重算”才能继续。'},accum:{en:'Pick the last option, all three tricks, to continue.',zh:'选最后一项“三个办法一起用”，才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#tmem-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#tmem-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(s.id!=='states'||watched.has(sigOf(BASE)));
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever, [k,x]=v.split(':'); let on=false, res='';
      if(k==='opt'){on=x===S.opt; res=optTried.has(x)||on?optRes(x):'';}
      else if(k==='rc'){on=+x===S.rc; res=rcTried.has(+x)||on?rcRes(+x):'';}
      else if(k==='ac'){on=x===acSel; res=acTried.has(x)||on?acRes(x):'';}
      b.setAttribute('aria-pressed',String(on)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=res;});
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    $$('[data-nav="showme"]',rail).forEach(sm=>{sm.hidden=!!observed[step];});
    $$('.tmem-rem',rail).forEach(b=>{const on=b.dataset.rem===remView; b.setAttribute('aria-pressed',String(on)); b.textContent=on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看');});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#tmem-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.opt,S.rc,S.mb), p=$('#tmem-ch-p',rail), t=$('#tmem-ch-t',rail); if(!p) return;
    p.textContent=GB(e.peak); t.textContent=fixd(e.t,2);
    const g1=$('#tmem-ch-g1',rail), g2=$('#tmem-ch-g2',rail); g1.classList.toggle('met',e.fits); g1.classList.toggle('tmem-miss',!e.fits); g2.classList.toggle('met',e.t<=CH.goal+1e-9); g2.classList.toggle('tmem-miss',e.t>CH.goal+1e-9);
    $$('.seg-ctl[data-key="cho"]',rail).forEach(g=>A.ui.setPressed(g,S.opt)); $$('.seg-ctl[data-key="chr"]',rail).forEach(g=>A.ui.setPressed(g,String(S.rc))); $$('.seg-ctl[data-key="chm"]',rail).forEach(g=>A.ui.setPressed(g,String(S.mb)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#tmem-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
const hidden0=()=>mode==='guided'&&isP('mission');
let TL=null, TLc=null, TLkey='';
function ensureTL(){const k=JSON.stringify(S); if(k!==TLkey||!TL){TLkey=k; TLc=compute(S); TL=timeline(TLc);} return {c:TLc,tl:TL};}
const BLK=[{k:'w',en:'BF16 weights',zh:'BF16 权重'},{k:'g',en:'FP32 gradients',zh:'FP32 梯度'},{k:'c',en:'FP32 copy',zh:'FP32 权重副本'},{k:'m',en:'Adam’s m',zh:'Adam 的 m'},{k:'v',en:'Adam’s v',zh:'Adam 的 v'},{k:'a',en:'activations',zh:'激活'},{k:'l',en:'logits',zh:'logits'}];
const blkTip=(k,c,val)=>({w:tx(`BF16 weights: ${GB(val)}, 2 bytes per parameter. Inference holds only these.`,`BF16 权重：${GB(val)}，每个参数 2 字节。推理只放这一样。`),
  g:tx(`FP32 gradients: ${GB(val)}, 4 bytes per parameter, written in the backward pass.`,`FP32 梯度：${GB(val)}，每个参数 4 字节，反向时写进去。`),
  c:tx(`FP32 copy of the weights: ${GB(val)}, 4 bytes per parameter. The updates are added to it.`,`FP32 权重副本：${GB(val)}，每个参数 4 字节，更新都加在它上面。`),
  m:tx(`${S.opt==='sgdm'?'m':'Adam’s m'}, the running average of the gradient: ${GB(val)} (${OPT[S.opt][3]} bytes per parameter).`,`${S.opt==='sgdm'?'m':'Adam 的 m'}，梯度的滑动平均：${GB(val)}（每个参数 ${OPT[S.opt][3]} 字节）。`),
  v:tx(`Adam’s v, the running average of the squared gradient: ${GB(val)} (${OPT[S.opt][4]} bytes per parameter).`,`Adam 的 v，梯度平方的滑动平均：${GB(val)}（每个参数 ${OPT[S.opt][4]} 字节）。`),
  a:tx(`Activations kept for the backward pass: ${GB(val)} at this moment.`,`为反向留着的激活：此刻 ${GB(val)}。`),
  l:tx(`Logits in FP32: ${GB(val)} at this moment.`,`FP32 的 logits：此刻 ${GB(val)}。`)}[k]);
const niceStep=v=>{const t=[5,10,20,25,50,100,200,250,500,1000,2000]; for(const s of t) if(v/s<=6) return s; return 5000;};
/* the GPU column: the high-water mark so far, block by block, labels beside the blocks (R5.9); what is above the GPU's top is red */
function drawColumn(c,tl,x){
  const host=$('#tmem-col',app), W=Math.max(200,Math.min(260,host.clientWidth||220)), H=colH();
  const top=12, bot=H-22, ymax=yMax(c), Y=v=>bot-(v/ymax)*(bot-top);
  const hw=hidden0()?{acts:0,lg:0}:hwX(tl,c,x);
  const vals=[c.blocks[0],c.blocks[1],c.blocks[2],c.blocks[3],c.blocks[4],hw.acts,hw.lg];
  const cx=40, cw=48, ix=10, iw=18;
  let svg=`<svg class="tmem-colsvg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(colAria(c,hw))}">`+
    `<defs><pattern id="tmem-p-g" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="tmem-pbg"/><line x1="0" y1="0" x2="0" y2="6" class="tmem-pln"/></pattern>`+
    `<pattern id="tmem-p-m" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" class="tmem-pbg"/><circle cx="2.5" cy="2.5" r="1.1" class="tmem-pdot"/></pattern>`+
    `<pattern id="tmem-p-v" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="tmem-pbg"/><line x1="0" y1="0" x2="0" y2="6" class="tmem-pln"/><line x1="0" y1="3" x2="6" y2="3" class="tmem-pln"/></pattern>`+
    `<pattern id="tmem-p-l" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="4" class="tmem-pbg"/><line x1="1" y1="0" x2="1" y2="4" class="tmem-pln"/></pattern></defs>`;
  /* inference column (weights only), same scale */
  const iy=Y(c.infer);
  svg+=`<rect x="${ix}" y="${iy.toFixed(1)}" width="${iw}" height="${(bot-iy).toFixed(1)}" class="tmem-b-w" tabindex="0" data-tipx="${esc(tx(`Inference: ${GB(c.infer)}, the weights alone, for comparison.`,`推理：${GB(c.infer)}，只有权重，放在这里作对比。`))}"/><text x="${ix+iw/2}" y="${bot+14}" class="tmem-ax tmem-mid">${esc(tx('infer.','推理'))}</text>`;
  /* training column */
  const capY=Y(c.cap); let y=bot; const labs=[];
  if(hidden0()){svg+=`<text x="${cx+cw/2}" y="${(capY+bot)/2}" class="tmem-q tmem-mid">?</text>`;}
  else vals.forEach((v,i)=>{if(v<=0) return; const h=(v/ymax)*(bot-top), yy=y-h; const k=BLK[i].k;
    svg+=`<rect x="${cx}" y="${yy.toFixed(2)}" width="${cw}" height="${Math.max(0,h).toFixed(2)}" class="tmem-b-${k}" data-tipx="${esc(blkTip(k,c,v))}" tabindex="0"/>`;
    labs.push({y:yy+h/2,t:`${S.opt==='sgdm'&&k==='m'?'m':L(BLK[i])} ${GB(v)}`,k}); y=yy;});
  /* over the top: red */
  if(!hidden0()&&y<capY-0.5){svg+=`<rect x="${cx}" y="${y.toFixed(2)}" width="${cw}" height="${(capY-y).toFixed(2)}" class="tmem-over"/>`;}
  svg+=`<rect x="${cx-3}" y="${capY.toFixed(2)}" width="${cw+6}" height="${(bot-capY).toFixed(2)}" class="tmem-gpu" data-tipx="${esc(tx(`The ${c.g.name}: ${GB(c.cap)} of memory.`,`${c.g.name}：${GB(c.cap)} 显存。`))}" tabindex="0"/>`;
  svg+=`<text x="${cx+cw/2}" y="${bot+14}" class="tmem-ax tmem-mid">${esc(tx('training','训练'))}</text>`;
  /* labels beside the blocks, pushed apart */
  const lx=cx+cw+8, minG=12; labs.sort((a,b)=>b.y-a.y);
  for(let i=1;i<labs.length;i++) if(labs[i-1].y-labs[i].y<minG) labs[i].y=labs[i-1].y-minG;
  const over=labs.length&&labs[labs.length-1].y<top+4; if(over){const d=top+4-labs[labs.length-1].y; labs.forEach(l=>{l.y+=d;}); for(let i=labs.length-2;i>=0;i--) if(labs[i].y-labs[i+1].y<minG) labs[i].y=labs[i+1].y+minG;}
  labs.forEach(l=>{svg+=`<text x="${lx}" y="${(l.y+4).toFixed(1)}" class="tmem-bl">${esc(l.t)}</text>`;});
  /* H7 (2026-10-09): only the size beside the line, so it stays inside the picture; the caption above names the GPU */
  const cap=GB(c.cap);
  svg+=`<text x="${cx-4}" y="${(capY-4).toFixed(1)}" class="tmem-capl" text-anchor="end">${esc(cap)}</text><line x1="${cx-6}" x2="${cx+cw+6}" y1="${capY.toFixed(2)}" y2="${capY.toFixed(2)}" class="tmem-capline"/>`;
  if(!hidden0()&&y<capY-0.5){const ln=A.lang()==='zh'?['放不下']:['out of','memory'], room=capY-y;
    if(room>=ln.length*12+6){const y0=(y+capY)/2-(ln.length-1)*6+4; ln.forEach((t,i)=>{svg+=`<text x="${cx+cw/2}" y="${(y0+i*12).toFixed(1)}" class="tmem-oomt tmem-halo tmem-mid">${esc(t)}</text>`;});}
    else svg+=`<text x="${cx+cw/2}" y="${Math.max(top+10,y-4).toFixed(1)}" class="tmem-oomt tmem-halo tmem-mid">${esc(ln.join(' '))}</text>`;}
  svg+=`</svg>`;
  host.innerHTML=svg;
  $('#tmem-col-cap',app).textContent=hidden0()?tx(`${c.g.name} memory: inference and training`,`${c.g.name} 显存：推理和训练`):tx(`${c.g.name} memory, highest so far in this step`,`${c.g.name} 显存：这一步到此刻的最高点`);
}
function colAria(c,hw){if(hidden0()) return tx(`Inference holds ${GB(c.infer)} of weights; training is hidden until you predict.`,`推理放 ${GB(c.infer)} 的权重；训练的那一栏等你预测后再显示。`);
  return tx(`Highest so far: ${GB(c.states+hw.acts+hw.lg)}, of which model states ${GB(c.states)}, activations ${GB(hw.acts)}, logits ${GB(hw.lg)}. The ${c.g.name} holds ${GB(c.cap)}.`,`到此刻的最高点：${GB(c.states+hw.acts+hw.lg)}，其中模型状态 ${GB(c.states)}，激活 ${GB(hw.acts)}，logits ${GB(hw.lg)}。${c.g.name} 有 ${GB(c.cap)}。`);}
const colH=()=>innerWidth<=760?220:250;
const yMax=c=>Math.max(c.cap,c.peak,c.infer)*1.1;
/* S8: the GPU line's label sits in a right gutter outside the plot, so it never covers the curve */
const capGut=c=>Math.ceil(Math.max(String(c.g.name).length,GB(c.cap).length)*6.4)+10;
function curveStatic(c,tl,W,H){
  const top=12, bot=H-26, lx=34, rx=W-6-capGut(c), ymax=yMax(c), Y=v=>bot-(v/ymax)*(bot-top), X=x=>lx+x*(rx-lx);
  let g='';
  const st=niceStep(ymax/1e9);
  for(let v=0; v<=ymax/1e9; v+=st){const yy=Y(v*1e9); g+=`<line x1="${lx}" x2="${rx}" y1="${yy.toFixed(1)}" y2="${yy.toFixed(1)}" class="tmem-grid"/><text x="${lx-4}" y="${(yy+3.5).toFixed(1)}" class="tmem-ax" text-anchor="end">${fint(v)}</text>`;}
  g+=`<text x="${lx-4}" y="${top-2}" class="tmem-ax" text-anchor="end">GB</text>`;
  const pts=tl.V.map(v=>`${X(v.x).toFixed(2)},${Y(c.states+v.acts+v.lg).toFixed(2)}`).join(' ');
  g+=`<rect x="${lx}" y="${Y(c.states).toFixed(2)}" width="${(rx-lx).toFixed(2)}" height="${(bot-Y(c.states)).toFixed(2)}" class="tmem-band"/>`;
  g+=`<polygon points="${X(0)},${Y(c.states).toFixed(2)} ${pts} ${X(1)},${Y(c.states).toFixed(2)}" class="tmem-area"/>`;
  const capY=Y(c.cap);
  g+=`<clipPath id="tmem-clip-over"><rect x="${lx}" y="0" width="${rx-lx}" height="${Math.max(0,capY).toFixed(2)}"/></clipPath>`;
  g+=`<g clip-path="url(#tmem-clip-over)"><polygon points="${X(0)},${bot} ${X(0)},${Y(c.states).toFixed(2)} ${pts} ${X(1)},${Y(c.states).toFixed(2)} ${X(1)},${bot}" class="tmem-overarea"/></g>`;
  g+=`<polyline points="${pts}" class="tmem-line"/>`;
  g+=`<line x1="${lx}" x2="${rx}" y1="${capY.toFixed(2)}" y2="${capY.toFixed(2)}" class="tmem-capline"/>`;
  g+=`<text x="${rx+5}" y="${(capY-3).toFixed(1)}" class="tmem-capl">${esc(c.g.name)}</text><text x="${rx+5}" y="${(capY+10).toFixed(1)}" class="tmem-capl">${esc(GB(c.cap))}</text>`;
  /* the band label keeps clear of the GPU's dashed line */
  let by=Math.min(bot-4,Y(c.states/2)+4); if(Math.abs(by-4-capY)<12) by=capY+18<=bot-4?capY+18:capY-8;
  g+=`<text x="${lx+4}" y="${by.toFixed(1)}" class="tmem-bandl">${esc(tx(`model states ${GB(c.states)}`,`模型状态 ${GB(c.states)}`))}</text>`;
  /* phases of the first micro-batch along the bottom; then the optimizer step */
  const v0=tl.V, fEnd=v0.find(v=>v.ph==='h').x, bEnd=(tl.k>1?v0.find(v=>v.mi===1).x:v0[v0.length-2].x), oStart=v0[v0.length-2].x;
  const lab=(a,b,t,cls)=>`<rect x="${X(a).toFixed(1)}" y="${bot+4}" width="${Math.max(1,X(b)-X(a)-1).toFixed(1)}" height="6" class="${cls}"/>`+((X(b)-X(a))>34?`<text x="${((X(a)+X(b))/2).toFixed(1)}" y="${bot+21}" class="tmem-ax tmem-mid">${esc(t)}</text>`:'');
  g+=lab(0,fEnd,tx('forward','前向'),'tmem-ph-f')+lab(fEnd,bEnd,tl.rc?tx('backward + recompute','反向（含重算）'):tx('backward','反向'),'tmem-ph-b');
  if(tl.k>1) g+=`<text x="${((X(bEnd)+X(oStart))/2).toFixed(1)}" y="${bot+21}" class="tmem-ax tmem-mid">${esc(tx(`… ${tl.k} micro-batches in all`,`…… 共 ${tl.k} 份`))}</text>`;
  g+=`<rect x="${X(oStart).toFixed(1)}" y="${bot+4}" width="${Math.max(2,X(1)-X(oStart)).toFixed(1)}" height="6" class="tmem-ph-o"/>`;
  return g;
}
function drawCurve(c,tl,x){
  const host=$('#tmem-curve',app), box=$('#tmem-vis',app);
  const W=Math.max(220,Math.round(host.clientWidth||420)), H=colH();
  if(hidden0()){host.innerHTML=`<div class="tmem-curve0" style="height:${H}px"><p>${esc(tx('One training step plays here in part 2, after your prediction.','作答之后，第 2 部分在这里播放训练的一步。'))}</p></div>`; host._k=''; $('#tmem-cv-cap',app).textContent=tx('Memory over one training step','训练一步时显存怎么变'); return;}
  const key=[TLkey,W,H,A.lang()].join('|');
  if(host._k!==key){host._k=key; host.innerHTML=`<svg class="tmem-cvsvg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"><g class="tmem-static">${curveStatic(c,tl,W,H)}</g><g class="tmem-dyn"></g></svg>`;}
  const top=12, bot=H-26, lx=34, rx=W-6-capGut(c), ymax=yMax(c), Y=v=>bot-(v/ymax)*(bot-top), X=u=>lx+u*(rx-lx);
  const cur=atX(tl,x), memNow=c.states+cur.acts+cur.lg;
  let d=`<line x1="${X(x).toFixed(2)}" x2="${X(x).toFixed(2)}" y1="${top}" y2="${bot}" class="tmem-cursor"/><circle cx="${X(x).toFixed(2)}" cy="${Y(memNow).toFixed(2)}" r="3.5" class="tmem-cdot${memNow>c.cap?' bad':''}"/>`;
  if(x>=tl.peakX-1e-9){const pv=c.peak, px=X(tl.peakX), py=Y(pv), right=px>(lx+rx)/2;
    const capY=Y(c.cap), ly=Math.abs(py-capY)<12?Math.min(py,capY)-6:py-6;
    d+=`<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="4.5" class="tmem-pk${pv>c.cap?' bad':''}"/><text x="${(right?px-8:px+8).toFixed(1)}" y="${Math.max(top+10,ly).toFixed(1)}" class="tmem-pkl" text-anchor="${right?'end':'start'}">${esc(tx(`peak ${GB(pv)}`,`峰值 ${GB(pv)}`))}</text>`;}
  if(tl.oom!=null&&x>=tl.oom-1e-9){const ox=X(tl.oom), oy=Y(Math.min(c.cap,c.states+tl.V[0].acts)), at0=tl.oom<=1e-9;
    const oyy=at0?Y(c.states):Y(c.cap), ot=at0?tx('over from the start','一开始就放不下'):tx('from here on it does not fit','从这里起放不下');
    /* S8: the label goes inside the model-states band, just under its top edge, where the curve never is */
    /* H7 (2026-10-09): beside the dot when there is room, otherwise slid back inside the plot (never past its left edge) */
    const ow=ot.length*(A.lang()==='zh'?11.5:6.2), oty=Math.min(bot-4,Y(c.states)+14), flip=ox+8+ow>rx;
    const ox0=Math.max(lx+2,Math.min(flip?ox-8-ow:ox+8,rx-ow));
    d+=`<circle cx="${ox.toFixed(2)}" cy="${oyy.toFixed(2)}" r="5" class="tmem-oomdot"/><text x="${ox0.toFixed(1)}" y="${oty.toFixed(1)}" class="tmem-oomt" text-anchor="start">${esc(ot)}</text>`; void oy;}
  const dyn=host.querySelector('.tmem-dyn'); if(dyn) dyn.innerHTML=d;
  const svg=host.querySelector('svg'); if(svg) svg.setAttribute('aria-label',tx(`Memory over one update: now ${GB(memNow)}; model states ${GB(c.states)}; peak ${GB(c.peak)} at the end of the forward pass; the ${c.g.name} holds ${GB(c.cap)}.`,`一次更新中显存的变化：此刻 ${GB(memNow)}；模型状态 ${GB(c.states)}；峰值 ${GB(c.peak)}，在前向结束时；${c.g.name} 有 ${GB(c.cap)}。`));
  $('#tmem-cv-cap',app).textContent=tl.k>1?tx(`Memory over one update: ${tl.k} micro-batches, then the optimizer step`,`一次更新中显存怎么变：${tl.k} 份 micro-batch，然后更新参数`):tx('Memory over one training step: forward, backward, optimizer step','训练一步时显存怎么变：前向、反向、更新参数');
  void box;
}
function drawLegend(c){
  const lg=$('#tmem-cv-leg',app); if(hidden0()){if(lg._h!==''){lg.innerHTML=''; lg._h='';} return;}
  const h=`<span tabindex="0" data-tipx="${esc(tx('The flat band at the bottom: the model states, held for the whole update.','底部平的那一条：模型状态，整次更新都占着。'))}"><i class="sw tmem-sw-band"></i>${esc(tx('model states','模型状态'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('The shaded area above it: activations and logits, which rise in the forward pass and are freed in the backward pass.','上面的灰色区域：激活和 logits，前向时涨，反向时释放。'))}"><i class="sw tmem-sw-area"></i>${esc(tx('activations and logits','激活和 logits'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx(`The dashed line: all the memory the ${c.g.name} has.`,`虚线：${c.g.name} 的全部显存。`))}"><i class="sw tmem-sw-cap"></i>${esc(tx(`dashed: the ${c.g.name}’s ${GB(c.cap)}`,`虚线：${c.g.name} 的 ${GB(c.cap)}`))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('Red: memory the step needs beyond what the GPU has. A real run stops there with an out-of-memory error.','红色：这一步需要、而 GPU 没有的显存。真实训练到这里会报显存不足（OOM）。'))}"><i class="sw tmem-sw-over"></i>${esc(tx('red: over the GPU','红色：超出 GPU'))}</span>`;
  const key=h; if(lg._h!==key){lg.innerHTML=h; lg._h=key;}
}
function phaseText(c,tl,x){
  if(x>=1-1e-9) return tx('The update is done. The column keeps this step’s peak.','这一次更新做完了。柱子留着这一步的峰值。');
  const p=atX(tl,x), pre=tl.k>1&&p.mi<tl.k?tx(`micro-batch ${p.mi+1} of ${tl.k}: `,`第 ${p.mi+1} / ${tl.k} 份：`):'';
  const Lr=c.m.L;
  if(p.ph==='f'){const li=Math.min(Lr,Math.max(1,Math.ceil(atLayer(tl,x)))); return pre+tx(`forward, layer ${li} of ${Lr}: every layer keeps its activations`,`前向，第 ${li} / ${Lr} 层：每一层留下自己的激活`);}
  if(p.ph==='h') return pre+tx('LM head and loss: the logits appear','LM head 和 loss：logits 出现');
  if(p.ph==='o') return tx('optimizer step: the parameters are updated','优化器更新参数');
  const li=Math.max(1,p.li);
  return pre+(p.ph==='r'?tx(`backward, layer ${li}: first its forward again`,`反向，第 ${li} 层：先重算它的前向`):tx(`backward, layer ${li}: its activations are used and freed`,`反向，第 ${li} 层：激活用完就释放`));
}
function atLayer(tl,x){const V=tl.V; let i=0; while(i<V.length-2&&V[i+1].x<x) i++; const a=V[i], b=V[i+1]; if(a.ph!=='f') return a.li; const u=b.x>a.x?(x-a.x)/(b.x-a.x):1; return a.li+(b.li-a.li)*u;}
function drawPlayer(c,tl){
  const pl=$('#tmem-player',app); if(!pl) return;
  if(hidden0()){if(pl._built!=='none'){pl.innerHTML=''; pl._built='none';} return;}
  const label=playing?tx('❚❚ Pause','❚❚ 暂停'):(playT>=1?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  if(pl._built!==A.lang()){pl._built=A.lang();
    pl.innerHTML=`<div class="tmem-prow"><button type="button" class="btn ghost sm tmem-play" data-act="play" data-tipx="${esc(tx('Play one update at the chosen speed; press again to pause. When it has finished, the button plays it again from the start.','按选定的速度播放一次更新；再按一次暂停。播完后，这个按钮从头再放一遍。'))}"></button>`+
      A.ui.seg({id:'tmem-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd})+`</div>`+
      `<div class="tmem-scrub"><input type="range" min="0" max="1000" step="1" value="${Math.round(playT*1000)}" data-key="scrub" aria-label="${esc(tx('Moment in the update','这次更新进行到哪里'))}"></div><p class="tmem-phase mono" id="tmem-phase"></p>`;
    A.ui.initRanges(pl);}
  const b=pl.querySelector('[data-act="play"]'); if(b.textContent!==label) b.textContent=label;
  $$('.seg-ctl[data-key="spd"]',pl).forEach(g=>A.ui.setPressed(g,spd));
  const sc=pl.querySelector('input[data-key="scrub"]'); const v=Math.round(playT*1000); if(+sc.value!==v){sc.value=v; A.ui.setRange(sc);}
  const ph=pl.querySelector('#tmem-phase'), t=phaseText(c,tl,playT); if(ph.textContent!==t) ph.textContent=t;
}
function statusOf(){
  if(mode==='challenge'){const e=chEval(S.opt,S.rc,S.mb); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when one step fits in 24 GB and an update takes at most ${fixd(CH.goal,2)} of the reference time. Now: ${GB(e.peak)}, time ${fixd(e.t,2)}.`,`一步放得下 24 GB、每次更新的用时不超过参照的 ${fixd(CH.goal,2)} 时，徽章是绿色。现在：${GB(e.peak)}，用时 ${fixd(e.t,2)}。`)};}
  if(hidden0()) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The training column stays hidden until you answer in part 1.','训练那一栏先隐藏，等你在第 1 部分作答。')};
  const {c,tl}=ensureTL(), hw=hwX(tl,c,playT), sofar=c.states+hw.acts+hw.lg;
  if(sofar>c.cap) return {cls:'bad',txt:tx('Out of memory','放不下'),short:tx('Out of memory','放不下'),tip:tx(`This step needs ${GB(sofar)} so far; the ${c.g.name} holds ${GB(c.cap)}.`,`这一步到此刻已经要 ${GB(sofar)}；${c.g.name} 只有 ${GB(c.cap)}。`)};
  if(playT<1) return {cls:'neutral',txt:tx('Playing','播放中'),short:tx('Playing','播放中'),tip:tx(`So far ${GB(sofar)} of the ${GB(c.cap)}.`,`到此刻用了 ${GB(sofar)}，共 ${GB(c.cap)}。`)};
  return {cls:'ok',txt:tx('Fits','装得下'),short:tx('Fits','装得下'),tip:tx(`The step peaks at ${GB(c.peak)}, within the ${c.g.name}’s ${GB(c.cap)}.`,`这一步的峰值是 ${GB(c.peak)}，在 ${c.g.name} 的 ${GB(c.cap)} 以内。`)};
}
function drawHead(){
  const t=$('#tmem-st-title',app), sub=$('#tmem-st-sub',app), st=$('#tmem-status',app), c=compute(S);
  t.textContent=mode==='challenge'?tx(`Does one step fit on the RTX 4090’s 24 GB?`,'一步能不能放进 RTX 4090 的 24 GB？'):tx(`Does one training step fit on one ${c.g.name} (${GB(c.cap)})?`,`训练一步，一张 ${c.g.name}（${GB(c.cap)}）放得下吗？`);
  sub.textContent=tx(`${c.m.name} with ${optName(S.opt)}, ${S.mb===1?`1 sequence of ${tokS(S.seq)} tokens at once`:`${S.mb} sequences of ${tokS(S.seq)} tokens at once`}${S.acc>1?`, ${S.acc} passes per update`:''}${S.rc?', every layer recomputed':''}`,
    `${c.m.name}，用 ${optName(S.opt)}，${S.mb===1?`每次送进 1 条 ${tokS(S.seq)} 个 token 的序列`:`每次送进 ${S.mb} 条序列，每条 ${tokS(S.seq)} 个 token`}${S.acc>1?`，每次更新跑 ${S.acc} 趟`:''}${S.rc?'，每一层都重算':''}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const {c,tl}=ensureTL(), hide=hidden0(), hw=hide?{acts:0,lg:0}:hwX(tl,c,playT), sofar=c.states+hw.acts+hw.lg, done=playT>=1;
  const pk={k:'pk',label:done?tx('Peak memory','峰值显存'):tx('Highest so far','到此刻的最高点'),v:hide?'—':GB(sofar),sub:hide?tx('shown in part 2','第 2 部分揭晓'):tx('of one step','一步里的最高点'),
    tip:hide?tx('Hidden until you answer in part 1.','在第 1 部分作答之前先隐藏。'):tx(`Model states ${G2(c.states)} + activations ${G2(hw.acts)} + logits ${G2(hw.lg)} = ${S2(c.states,hw.acts,hw.lg)}.`,`模型状态 ${G2(c.states)} + 激活 ${G2(hw.acts)} + logits ${G2(hw.lg)} = ${S2(c.states,hw.acts,hw.lg)}。`)};
  const room=c.cap-sofar;
  const rm={k:'rm',label:hide?tx(`Against the ${GB(c.cap)}`,`和 ${GB(c.cap)} 比`):room>=0?tx('Left on the GPU','GPU 还剩'):tx('Missing on the GPU','GPU 还差'),v:hide?'—':GB(Math.abs(room)),sub:hide?tx('shown in part 2','第 2 部分揭晓'):tx(`of the ${c.g.name}’s ${GB(c.cap)}`,`${c.g.name} 共 ${GB(c.cap)}`),
    tip:hide?tx('Hidden until you answer in part 1.','在第 1 部分作答之前先隐藏。'):(room>=0?tx(`${GB(c.cap)} − ${GB(sofar)} = ${GB(room)} left.`,`${GB(c.cap)} − ${GB(sofar)} = 还剩 ${GB(room)}。`):tx(`${GB(sofar)} − ${GB(c.cap)} = ${GB(-room)} more than the GPU has.`,`${GB(sofar)} − ${GB(c.cap)} = 比 GPU 多 ${GB(-room)}。`))};
  const ms={k:'ms',label:tx('Model states','模型状态'),v:hide?'—':GB(c.states),sub:hide?tx('shown in part 2','第 2 部分揭晓'):tx(`${c.bpp} bytes per parameter`,`每个参数 ${c.bpp} 字节`),tip:hide?tx('Hidden until you answer in part 1.','在第 1 部分作答之前先隐藏。'):tx(`${big(c.m.P)} parameters × ${c.bpp} bytes = ${GB(c.states)}.`,`${big(c.m.P)} 个参数 × ${c.bpp} 字节 = ${GB(c.states)}。`)};
  const cp={k:'cp',label:tx('Compute per token','每个 token 的计算'),v:S.rc?'8N':'6N',sub:S.rc?tx('recompute: a third more','重算，多三分之一'):tx('no recomputation','不重算'),tip:tx(`Training compute per token is ${S.rc?'8N with recomputation (one more forward pass, 2N)':'6N: 2N forward and 4N backward'}. N is the number of parameters, ${bigTxt(c.m.P)}.`,`每个 token 的训练计算是 ${S.rc?'8N（重算多一遍前向，2N）':'6N（前向 2N，反向 4N）'}。N 是参数量，${bigTxt(c.m.P)}。`)};
  if(mode==='challenge'){const e=chEval(S.opt,S.rc,S.mb); return [pk,rm,{k:'t',label:tx('Time per update','每次更新的用时'),v:fixd(e.t,2),sub:tx('reference = 1.00','参照 = 1.00'),tip:tx(`Compute per token (${S.rc?8:6}N) ÷ the share of the peak reached (${Math.round(CH.util[S.mb]*100)}%), against the reference, 6N at ${Math.round(CH.util[1]*100)}%:`,`每个 token 的计算（${S.rc?8:6}N）÷ 做到的峰值比例（${Math.round(CH.util[S.mb]*100)}%），再和参照（6N，${Math.round(CH.util[1]*100)}%）比：`)+`<div class="fx">${A.ui.math(`\\frac{${S.rc?8:6} \\div ${Math.round(CH.util[S.mb]*100)}\\%}{6 \\div ${Math.round(CH.util[1]*100)}\\%} = ${fixd(e.t,2)}`)}</div>`}];}
  return [pk,rm,ms,cp];
}
const bigTxt=n=>big(n);
/* S1: parts at 2 decimals, and the total as the exact sum of the parts as shown, so the "=" always adds up */
const G2=v=>`${fixd(v/1e9,2)} GB`;
const S2=(...vs)=>`${fixd(vs.reduce((a,v)=>a+Math.round(v/1e7),0)/100,2)} GB`;
function drawMetrics(){
  const host=$('#tmem-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!playing&&prev!=null&&prev!==m.v&&/GB/.test(m.v)&&/GB/.test(prev)){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    if(userAct||!playing){clearTimeout(d._t); clearTimeout(d._t2);
      if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
      else {d.classList.remove('on'); d.textContent='';}}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4) */
function mathHTML(){
  if(hidden0()) return `<p>${tx('The formulas appear once you have answered part 1.','在第 1 部分作答之后，这里给出公式。')}</p>`;
  const c=compute(S), m=c.m, t=T_, b=OPT[S.opt], q=m.H*m.hd, kv=m.KV*m.hd, X=A.ui.math, Xt=v=>`\\text{${v}}`;
  const rows=[];
  /* H2 (the owner, 2026-10-09): a few short blocks, each a lead line in words, the formula with its name, the reader's numbers under it */
  rows.push(`<p><b>${tx('Model states','模型状态')}</b>${tx(': every parameter’s bytes, times the number of parameters.','：每个参数要存的字节数，乘以参数量。')}</p>`+
    M(tx('\\text{model states} = N \\times (2 + 4 + 4 + b_m + b_v)','\\text{模型状态} = N \\times (2 + 4 + 4 + b_m + b_v)'))+
    `<p>${tx(`${t('N','N')} is the number of parameters. 2, 4 and 4 are the bytes of the BF16 weight, the FP32 gradient and the FP32 copy. ${X('b_m')} and ${X('b_v')} are the bytes of the optimizer’s m and v: ${b[3]} and ${b[4]} for ${optName(S.opt)}.`,
            `${t('N','N')} 是参数量。2、4、4 是 BF16 权重、FP32 梯度和 FP32 权重副本的字节数。${X('b_m')} 和 ${X('b_v')} 是优化器的 m 和 v 的字节数，${optName(S.opt)} 是 ${b[3]} 和 ${b[4]}。`)}</p>`+
    MC(`= ${fint(m.P)} \\times (2 + 4 + 4 + ${b[3]} + ${b[4]}) \\ ${Xt(tx('bytes','字节'))} = ${Xt(GB(c.states))}`));
  rows.push(`<p><b>${tx('Activations per token per layer','每个 token 每层的激活')}</b>${tx(': the tensors one token leaves in one layer, 2 bytes per value (BF16).','：一个 token 在一层里留下的张量，每个值 2 字节（BF16）。')}</p>`+
    MW('a','2 \\times (2d + 2 n_q d_h + 2 n_{kv} d_h) + 2 \\times (2d + 4f)')+
    `<ul class="tmem-tlist"><li>${tx(`attention block: RMSNorm input and output (${X('d')} each), q and the attention output (${X('n_q d_h')} each), k and v (${X('n_{kv} d_h')} each)`,`attention 块：RMSNorm 的输入和输出（各 ${X('d')}），q 和注意力输出（各 ${X('n_q d_h')}），k 和 v（各 ${X('n_{kv} d_h')}）`)}</li><li>${tx(`MLP block: RMSNorm input and output (${X('d')} each), gate, ${X('\\text{SiLU}(\\text{gate})')}, up, and ${X('\\text{SiLU}(\\text{gate}) \\cdot \\text{up}')} (${X('f')} each)`,`MLP 块：RMSNorm 的输入和输出（各 ${X('d')}），gate、${X('\\text{SiLU}(\\text{gate})')}、up 和 ${X('\\text{SiLU}(\\text{gate}) \\cdot \\text{up}')}（各 ${X('f')}）`)}</li></ul>`+
    `<p>${tx(`${t('d','d')} is the hidden size, ${t('f','f')} the MLP block’s intermediate size, ${X('n_q d_h')} the length of all query heads together and ${X('n_{kv} d_h')} that of all KV heads. For ${m.name}:`,
            `${t('d','d')} 是隐藏维度，${t('f','f')} 是 MLP 块的中间维度，${X('n_q d_h')} 是所有 query head 合起来的长度，${X('n_{kv} d_h')} 是所有 KV head 合起来的长度。${m.name} 的数值代进去：`)}</p>`+
    MW('a',`2 \\times (2 \\times ${fint(m.d)} + 2 \\times ${fint(q)} + 2 \\times ${fint(kv)}) + 2 \\times (2 \\times ${fint(m.d)} + 4 \\times ${fint(m.f)})`)+
    M(`= ${fint(c.aL)} \\ ${Xt(tx('bytes','字节'))}`));
  rows.push(`<p><b>${tx('Activations kept','留下的激活')}</b>${S.rc?tx(': with recomputation each layer keeps only its input, 2d bytes per token.','：重算时每层只留输入，每个 token 2d 字节。'):tx(': every token processed at once leaves a in every layer.','：一次处理的每个 token，在每一层都留下 a。')}</p>`+
    M(tx(`\\text{activations} = \\text{tokens at once} \\times \\text{layers} \\times ${S.rc?'2d':'a'}`,`\\text{激活} = \\text{一次处理的 token 数} \\times \\text{层数} \\times ${S.rc?'2d':'a'}`))+
    `<p>${t('T',tx('Tokens at once','一次处理的 token 数'))} = ${S.mb} × ${fint(S.seq)} = ${fint(c.T)}${tx(', ','，')}${t('L',tx('layers','层数'))} = ${m.L}${tx('.','。')}</p>`+
    MC(`= ${fint(c.T)} \\times ${m.L} \\times ${fint(S.rc?c.aI:c.aL)} \\ ${Xt(tx('bytes','字节'))} = ${Xt(GB(c.acts))}`));
  rows.push(`<p><b>${tx('Logits (FP32)','logits（FP32）')}</b>${tx(': every token gets one logit per vocabulary entry, 4 bytes each.','：每个 token 对词表里每一项都有一个 logit，每个 4 字节。')}</p>`+
    M(tx('\\text{logits} = \\text{tokens at once} \\times V \\times 4','\\text{logits} = \\text{一次处理的 token 数} \\times V \\times 4'))+
    `<p>${t('V','V')} ${tx('is the vocabulary size.','是词表大小。')}</p>`+
    MC(`= ${fint(c.T)} \\times ${fint(m.V)} \\times 4 \\ ${Xt(tx('bytes','字节'))} = ${Xt(GB(c.logits))}`));
  rows.push(`<p><b>${tx('Peak, at the end of the forward pass','峰值（前向结束时）')}</b>${tx(': the three add up.','：三样加起来。')}</p>`+
    MC(`${Xt(G2(c.states))} + ${Xt(G2(c.acts))} + ${Xt(G2(c.logits))} = ${Xt(S2(c.states,c.acts,c.logits))}`)+
    `<p>${tx(`The ${c.g.name} holds ${GB(c.cap)}.`,`${c.g.name} 有 ${GB(c.cap)}。`)}</p>`);
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){Object.assign(S,{m:CH.m,gpu:CH.gpu,seq:CH.seq}); if(!CH.mbs.includes(S.mb)) S.mb=1; if(!CH.opts.includes(S.opt)) S.opt='adamw'; S.acc=CH.upd/S.mb;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const {c,tl}=ensureTL();
  drawHead(); drawColumn(c,tl,playT); drawCurve(c,tl,playT); drawLegend(c); drawPlayer(c,tl);
  drawMetrics();
  const mb=$('#tmem-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* during a replay only the moving parts are redrawn */
function frame(){
  if(!app) return; const {c,tl}=ensureTL();
  drawColumn(c,tl,playT); drawCurve(c,tl,playT); drawPlayer(c,tl); drawMetrics();
  const st=$('#tmem-status',app), s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the optimizer, recomputation and the sequences at once, in the task card. Everything else is fixed: ${MOD['1b'].name}, one RTX 4090 with 24 GB, 2,048-token sequences, 32 sequences per update.`,`挑战中只能在任务卡里改优化器、重算和每次送几条。其余设置固定：${MOD['1b'].name}，一张 24 GB 的 RTX 4090，每条序列 2,048 个 token，每次更新 32 条。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='acts') return tx(`Fixed in this part: ${MOD['8b'].name}, AdamW, no recomputation, one H100. Only the length and the sequences at once change.`,`这一部分固定：${MOD['8b'].name}，AdamW，不重算，一张 H100。只改序列长度和一次送几条。`);
  if(id==='states'||id==='rc'||id==='accum') return tx('In this part you pick an option in the card. You can change every setting in Scenarios.','这一部分在卡片里选选项。在应用场景里可以改所有设置。');
  return tx(`Fixed in this part: ${MOD['8b'].name}, AdamW, one 4,096-token sequence, one H100. You can change the settings in Scenarios.`,`这一部分固定：${MOD['8b'].name}，AdamW，一条 4,096 个 token 的序列，一张 H100。在应用场景里可以改这些设置。`);
}
function syncControls(){
  const setR=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  setR('seq',Math.max(0,SEQS.indexOf(S.seq)),tokS(S.seq));
  $$('.seg-ctl[data-key="m"]',app).forEach(g=>A.ui.setPressed(g,S.m)); $$('.seg-ctl[data-key="opt"]',app).forEach(g=>A.ui.setPressed(g,S.opt));
  $$('.seg-ctl[data-key="mb"]',app).forEach(g=>A.ui.setPressed(g,String(S.mb))); $$('.seg-ctl[data-key="acc"]',app).forEach(g=>A.ui.setPressed(g,String(S.acc))); $$('.seg-ctl[data-key="rc"]',app).forEach(g=>A.ui.setPressed(g,String(S.rc)));
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#tmem-controls',app).classList.toggle('locked',ch);
  hide('#tmem-ctl-seq-wrap',ch||(g&&!isP('acts'))); hide('#tmem-ctl-mb-wrap',ch||(g&&!isP('acts')));
  ['#tmem-ctl-m-wrap','#tmem-ctl-opt-wrap','#tmem-ctl-acc-wrap','#tmem-ctl-rc-wrap','#tmem-ctl-reset-wrap'].forEach(id=>hide(id,g||ch));
  hide('#tmem-ctl-note',ch);
  const ln=$('#tmem-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||hidden0()) return; const c=compute(S);
  A.announce(tx(`Peak ${GB(c.peak)} on a ${GB(c.cap)} GPU: ${c.fits?'fits':'out of memory'}.`,`峰值 ${GB(c.peak)}，GPU 有 ${GB(c.cap)}：${c.fits?'装得下':'放不下'}。`));},400);}

/* =====================================================================
   replay: play only when the picture would differ (R3.20)
   ===================================================================== */
function tick(t){
  raf=0; if(!app||!playing) return;
  const dt=lastT?t-lastT:16; lastT=t;
  playT=Math.min(1,playT+dt/DUR[spd]);
  if(playT>=1){playing=false; playT=1; watched.add(sigOf(S)); wantObs=true; render(); wantObs=false; revealSoonIf(); return;}
  frame(); raf=requestAnimationFrame(tick);
}
function play(){if(playT>=1) playT=0; playing=true; lastT=0; if(A.reduceMotion()){finishPlay(); return;} if(!raf) raf=requestAnimationFrame(tick); frame();}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; if(app) frame();}
function finishPlay(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; playT=1; watched.add(sigOf(S)); wantObs=true; render(); revealSoon(); wantObs=false;}
/* after a change: replay if this picture has not been watched, else show the finished step */
function maybeReplay(auto){
  const sg=sigOf(S);
  if(!watched.has(sg)&&auto){playT=0; play(); return true;}
  if(!playing) playT=1;
  return false;
}
function revealSoonIf(){if(mode==='guided'){wantObs=true; updateRail(); wantObs=false; revealSoon();}}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p,auto){
  const was=sigOf(S);
  Object.assign(S,p);
  if(!SEQS.includes(S.seq)) S.seq=4096; if(!MBS.includes(S.mb)) S.mb=1; if(!ACCS.includes(S.acc)&&mode!=='challenge') S.acc=1; S.rc=S.rc?1:0; if(!OPT[S.opt]) S.opt='adamw'; if(!MOD[S.m]) S.m='8b';
  if(isP('acts')&&S.seq*S.mb!==4096) tokMoved=true;
  const changed=sigOf(S)!==was;
  if(changed&&playing) pause();
  if(changed) maybeReplay(!!auto);
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setOpt(o){optTried.add(o); setState({opt:o});}
function setRc(r){rcTried.add(r); setState({rc:r},true);}
function setAc(v){acSel=v; acTried.add(v); const s=acState(v); setState({mb:s.mb,acc:s.acc,opt:s.opt,rc:s.rc},true);}
function setRem(k){if(!REMS.includes(k)) return; remView=k; if(playing) pause(); S={...remState(k)}; playT=1; renderRail(); if(innerWidth<=760) A.ui.reveal($('#tmem-stage',app));}
function enterStep(i){
  wantObs=false; if(playing) pause(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  if(s.id==='rc'){rcTried=new Set([0]);}
  const st=$('#tmem-stage',app); if(st) st.scrollTop=0;
  const auto=s.id==='states';
  if(!maybeReplay(auto)) playT=1;
  renderRail();
}
function revealStep(){A.ui.reveal($('#tmem-rail .step.current',app)||$('#tmem-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* keepStage as in labs/llm-judge.js (R9.7): stop once the reader scrolls away from the rail's card */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#tmem-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#tmem-rail .step.current',app)||$('#tmem-rail .railcard',app)||$('#tmem-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#tmem-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#tmem-insight',app)); else revealBox($('#tmem-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; if(playing) pause();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={m:CH.m,gpu:CH.gpu,seq:CH.seq,opt:CH.start.opt,rc:CH.start.rc,mb:CH.start.mb,acc:CH.upd/CH.start.mb};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#tmem-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  playT=1;
  renderRail();
}
const token=()=>[Object.keys(MOD).indexOf(S.m),OPTS.indexOf(S.opt),SEQS.indexOf(S.seq),MBS.indexOf(S.mb),ACCS.indexOf(S.acc),S.rc].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==6||p.some(x=>!Number.isInteger(x))) return null;
  const [mi,oi,si,bi,ai,r]=p, mk=['8b','70b'][mi];
  if(!mk||oi<0||oi>=OPTS.length||si<0||si>=SEQS.length||bi<0||bi>=MBS.length||ai<0||ai>=ACCS.length||(r!==0&&r!==1)) return null;
  return {m:mk,gpu:'h100',opt:OPTS[oi],seq:SEQS[si],mb:MBS[bi],acc:ACCS[ai],rc:r};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+3), pk=a=>a[Math.floor(r()*a.length)];
  const s={m:pk(['8b','70b']),gpu:'h100',opt:pk(['adamw','adam8']),seq:pk([2048,4096,8192]),mb:pk([1,2]),acc:1,rc:pk([0,1])};
  return {seed,s,c:compute(s)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), s=d.s, c=d.c, qd=$('#tmem-drill-q',app), a=$('#tmem-drill-a',app);
  /* S7: a no-recompute seed gives what the activations need (f and the KV heads; Llama's query heads add up to d) */
  const shape=tx(`${bigH(c.m.P)} parameters, ${c.m.L} layers, hidden size ${fint(c.m.d)}${s.rc?'':`, MLP intermediate size ${fint(c.m.f)}, ${c.m.KV} KV heads of ${c.m.hd} dimensions`}`,
                 `${bigH(c.m.P)} 个参数，${c.m.L} 层，隐藏维度 ${fint(c.m.d)}${s.rc?'':`，MLP 中间维度 ${fint(c.m.f)}，${c.m.KV} 个 KV head，每个 head ${c.m.hd} 维`}`);
  qd.innerHTML=tx(`You fully fine-tune <b>${c.m.name}</b> with <b>${optName(s.opt)}</b>. It has ${shape}. Each step sends <b>${s.mb===1?`1 sequence of ${fint(s.seq)} tokens`:`${s.mb} sequences of ${fint(s.seq)} tokens`}</b> into the GPU at once, ${s.rc?'<b>recomputing every layer</b>':'<b>without recomputation</b>'}. Estimate the peak memory of one step. Just to hold the model states, how many 80 GB GPUs does it take at least?`,
                  `你要全量微调 <b>${c.m.name}</b>，用 <b>${optName(s.opt)}</b>。它有 ${shape}。每一步一次送进 <b>${s.mb===1?`1 条 ${fint(s.seq)} 个 token 的序列`:`${s.mb} 条序列，每条 ${fint(s.seq)} 个 token`}</b>，${s.rc?'<b>每一层都重算</b>':'<b>不重算</b>'}。估算一步的峰值显存。光放模型状态，至少要几张 80 GB 的卡？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const nG=Math.ceil(c.states/80e9), q80=fixd(c.states/80e9,2), pk=S2(c.states,c.acts,c.logits);
  a.innerHTML=tx(`<p><b>The peak is about ${GB(c.peak)}. Just to hold the model states takes at least ${nG} GPUs of 80 GB.</b></p>`+
    `<p>Model states: ${fint(c.m.P)} × ${c.bpp} bytes = ${G2(c.states)}.</p><p>Activations: ${fint(c.T)} tokens × ${c.m.L} layers × ${fint(c.perTok/c.m.L)} bytes = ${G2(c.acts)}.</p><p>Logits: ${fint(c.T)} × 128,256 × 4 bytes = ${G2(c.logits)}.</p>`+
    `<p>Peak: ${G2(c.states)} + ${G2(c.acts)} + ${G2(c.logits)} = ${pk}.</p><p>${G2(c.states)} ÷ 80 GB = ${q80}, rounded up to ${nG} GPUs.</p>`,
    `<p><b>峰值约 ${GB(c.peak)}。光放模型状态，至少要 ${nG} 张 80 GB 的卡。</b></p>`+
    `<p>模型状态：${fint(c.m.P)} × ${c.bpp} 字节 = ${G2(c.states)}。</p><p>激活：${fint(c.T)} 个 token × ${c.m.L} 层 × ${fint(c.perTok/c.m.L)} 字节 = ${G2(c.acts)}。</p><p>logits：${fint(c.T)} × 128,256 × 4 字节 = ${G2(c.logits)}。</p>`+
    `<p>峰值：${G2(c.states)} + ${G2(c.acts)} + ${G2(c.logits)} = ${pk}。</p><p>${G2(c.states)} ÷ 80 GB = ${q80}，向上取整是 ${nG} 张。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={seq:{en:'Tokens per sequence',zh:'每条序列的 token 数'},mb:{en:'Sequences per micro-batch',zh:'每次一起送进 GPU 的序列'},opt:{en:'Optimizer',zh:'优化器'}};
const TERM={
 N:()=>tx(`N is the number of parameters: ${fint(compute(S).m.P)} for ${compute(S).m.name}, from its config.`,`N 是参数量：${compute(S).m.name} 有 ${fint(compute(S).m.P)} 个，来自它的 config。`),
 d:()=>tx(`d is the hidden size, the length of every token’s vector between layers: ${fint(compute(S).m.d)} here (config.json hidden_size).`,`d 是隐藏维度，也就是层与层之间每个 token 的向量长度：这里是 ${fint(compute(S).m.d)}（config.json 的 hidden_size）。`),
 f:()=>tx(`f is the MLP block’s intermediate size: ${fint(compute(S).m.f)} here (config.json intermediate_size).`,`f 是 MLP 块的中间维度：这里是 ${fint(compute(S).m.f)}（config.json 的 intermediate_size）。`),
 L:()=>tx(`The number of layers: ${compute(S).m.L} here.`,`层数：这里是 ${compute(S).m.L}。`),
 V:()=>tx('V is the vocabulary size: 128,256 tokens for Llama 3, so every token gets 128,256 logits.','V 是词表大小：Llama 3 有 128,256 个 token，所以每个 token 有 128,256 个 logit。'),
 T:()=>tx(`Tokens at once = sequences at once × tokens per sequence = ${S.mb} × ${fint(S.seq)} = ${fint(S.mb*S.seq)}.`,`一次处理的 token 数 = 一次送的条数 × 每条的 token 数 = ${S.mb} × ${fint(S.seq)} = ${fint(S.mb*S.seq)}。`)
};
const TCTL={T:'seq'};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div>${jumpAct(TCTL[k]||k)?`<div class="act">${esc(jumpAct(TCTL[k]||k))}</div>`:''}`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: change the plan in the task card.','挑战中固定不变：请在任务卡里改方案。');
  if(mode!=='guided') return '';
  if((k==='seq'||k==='mb')&&isP('acts')) return '';
  if(k==='opt'&&isP('states')) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return k in TERM?tx('Comes from the model’s config; nothing to change here.','来自模型的 config，这里不能改。'):'';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({seq:$('#tmem-ic-seq',app)||$('#tmem-ctl-seq',app),mb:$('#tmem-ic-mb',app)||$('#tmem-ctl-mb',app),opt:$('#tmem-ic-opt',app)||$('#tmem-ctl-opt',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(TCTL[k]) k=TCTL[k];
  if(k==='math'){const d=$('#tmem-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#tmem-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]){if(src) A.tip.show(src,`<div>${esc(jumpAct(k)||tx('Comes from the model’s config.','来自模型的 config。'))}</div>`); return;}
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S=mode==='sandbox'?{...PRESETS[0].s}:{...BASE}; playT=1; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.opt,S.rc,S.mb).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(playing) pause(); else play(); A.tip.quiet(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#tmem-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#tmem-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='seq') bySettings(e.target,()=>setState({seq:SEQS[v]}));
    else if(k==='scrub'){if(playing) pause(); playT=v/1000; if(playT>=1){watched.add(sigOf(S)); render(); return;} frame();}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&k!=='scrub'&&e.target.closest('#tmem-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#tmem-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>{if(typeof s.showMe==='function') s.showMe(); else {if(s.showMe.rc!=null) rcTried.add(s.showMe.rc); setState(s.showMe,true);}}); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#tmem-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&mode==='guided'){const [k,x]=d.lever.split(':'); const f=()=>byLearner(()=>{if(k==='opt') setOpt(x); else if(k==='rc') setRc(+x); else if(k==='ac') setAc(x);});
      if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='mb') bySettings(b,()=>setState({mb:+d.v}));
      else if(k==='m') bySettings(b,()=>setState({m:d.v}));
      else if(k==='opt') bySettings(b,()=>setState({opt:d.v}));
      else if(k==='acc') bySettings(b,()=>setState({acc:+d.v}));
      else if(k==='rc') bySettings(b,()=>setState({rc:+d.v}));
      else if(k==='spd'){spd=d.v; lastT=0; if(app) frame();}
      else if(k==='cho') byLearner(()=>setState({opt:d.v}));
      else if(k==='chr') byLearner(()=>setState({rc:+d.v}));
      else if(k==='chm') byLearner(()=>setState({mb:+d.v,acc:CH.upd/(+d.v)}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s},true)); requestAnimationFrame(()=>{if(app) revealBox($('#tmem-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#tmem-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12, under the rule of 07:13 UTC): four text-only sections read top to bottom
   before the guided parts: what pre-training is and its scale; what one training step keeps in GPU memory (weights, gradients,
   optimizer state, activations; the first three are the model states); mixed precision (why BF16 to compute and FP32 to keep, and
   Adam's per-parameter state); where the activations come from and when a step's memory is highest. How much each item takes stays
   with the guided parts, so there is no byte total per parameter and no GB figure here. Section 1 owns the next-token
   (cross-entropy) loss on the site; the SFT primer links here for it. Hardware numbers are the H100 SXM
   datasheet's; the Llama 3 numbers are Meta's (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=labLink(id); return tx(a,`“${a}”`);};
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return h.startsWith('<div')?h:`<p>${h}</p>`;}).join('')});
const prFx=(en,zh)=>()=>`<div class="tmem-fx tmem-prfx">${prM(tx(en,zh))}</div>`;   // a formula on its own line
/* a sum of named items on its own line: “lhs = a”, then “+ b”, “+ c” as pieces that wrap on a phone. The items are \mathrm, not \text:
   beside an mtext the browser sets “=” and “+” with no space around them */
const prSum=(lhs,terms)=>()=>`<div class="tmem-fx tmem-prsum">${terms.map((t,i)=>`<span>${prM(i?`+ \\; ${t}`:`${lhs} = ${t}`)}</span>`).join('')}</div>`;
A.registerPrimer(ID,{min:9,sections:[
 {h:()=>tx('Pre-training: predicting the next token over trillions of tokens','预训练：在几万亿个 token 上预测下一个 token'),parts:[prP(
   ()=>tx(`A language model reads a piece of text and gives a probability distribution over the next token (see the basics of ${prLK('architecture')}). Its parameters, billions to hundreds of billions of them, are set by training. Pre-training starts from random parameters and trains the model on a huge amount of text to predict the next token.`,
          `语言模型读进一段文字，给出下一个 token 的概率分布（见${prLK('architecture')}的知识梳理）。它的参数有几十亿到几千亿个，要靠训练定下来。预训练就是从随机的参数出发，在海量文字上训练模型预测下一个 token。`),
   ()=>tx(`Write a text as ${prM('x')}, with ${prM('|x|')} tokens; its ${prM('i')}-th token is ${prM('x_i')}, all the tokens before it are ${prM('x_{<i}')}, and all the model’s parameters are ${prM('\\theta')}. Having read ${prM('x_{<i}')}, the model gives a distribution over the next token, ${prM('p_{\\theta}(\\cdot \\mid x_{<i})')} (${prM('x_{<1}')} is empty: in practice a special token marking the start goes before the text, and the model predicts ${prM('x_1')} from it). By the multiplication rule of conditional probability, the probability the model gives the whole text is the product of the probabilities it gives the right token at each position:`,
          `记一段文字为 ${prM('x')}，共 ${prM('|x|')} 个 token，第 ${prM('i')} 个为 ${prM('x_i')}，它前面的全部 token 为 ${prM('x_{<i}')}，模型的全部参数为 ${prM('\\theta')}。模型读完 ${prM('x_{<i}')}，给出下一个 token 的分布 ${prM('p_{\\theta}(\\cdot \\mid x_{<i})')}（${prM('x_{<1}')} 是空的：实际的文字前面有一个表示开头的特殊 token，模型从它出发预测 ${prM('x_1')}）。按条件概率的乘法公式，模型给这段文字的概率，是它在每个位置给正确 token 的概率之积：`),
   prFx('p_{\\theta}(x) = \\Pi_{i=1}^{|x|} \\, p_{\\theta}(x_i \\mid x_{<i})','p_{\\theta}(x) = \\Pi_{i=1}^{|x|} \\, p_{\\theta}(x_i \\mid x_{<i})'),
   ()=>tx(`Training should make the training text as probable as possible: maximum likelihood estimation. Taking the logarithm turns the product into a sum, a minus sign turns it into something to minimise, and dividing by the number of tokens ${prM('|x|')} keeps the loss from growing with the length of the text:`,
          `训练要让训练文字的概率尽量大，也就是最大似然估计。取对数把乘积变成求和，加负号变成要最小化的量，再除以 token 数 ${prM('|x|')}，让 loss 的大小不随文字长短而变：`),
   prFx('L(\\theta) = -\\frac{1}{|x|} \\sum_{i=1}^{|x|} \\log p_{\\theta}(x_i \\mid x_{<i})','L(\\theta) = -\\frac{1}{|x|} \\sum_{i=1}^{|x|} \\log p_{\\theta}(x_i \\mid x_{<i})'),
   ()=>tx(`Each term ${prM('-\\log p_{\\theta}(x_i \\mid x_{<i})')} is the cross-entropy between the distribution that gives the right token probability 1 and the model’s distribution, so this is called the cross-entropy loss: the lower the probability the right token gets, the larger its term. Training brings it down step by step with gradient descent (see the basics of ${prLK('backprop')}). A real step uses not one text but a batch of them, and the loss is the average over all their positions.`,
          `每一项 ${prM('-\\log p_{\\theta}(x_i \\mid x_{<i})')} 是“正确 token 的概率为 1”的分布和模型给出的分布之间的交叉熵，所以它叫交叉熵 loss：正确 token 分到的概率越低，这一项越大。训练用梯度下降一步步把它降下来（见${prLK('backprop')}的知识梳理）。实际的每一步用的不是一段文字，而是一批，loss 取其中所有位置的平均。`),
   ()=>tx('What sets pre-training apart is its scale. Llama 3 was pre-trained on about 15 trillion tokens of text. Llama 3.1 comes in three sizes, 8B, 70B and 405B: about 8, 70 and 405 billion parameters. The 405B was trained on up to 16,000 H100s, NVIDIA’s data-center GPUs.',
          '预训练的特点是规模大。Llama 3 的预训练文字约有 15 万亿个 token。Llama 3.1 有 8B、70B、405B 三个版本，即约 80 亿、700 亿和 4,050 亿个参数；其中 405B 是在多达 1.6 万张 H100（NVIDIA 的数据中心 GPU）上训练的。'),
   ()=>tx('Scale like this raises a chain of engineering questions: how much GPU memory one training step takes; when one GPU cannot hold it, how to split the model across several; how big a model a compute budget should buy, and how much data it needs; and how to carry on when something fails in a run that lasts months. The first has to be answered first, because whether one GPU can hold a step decides whether the model must be split across several: what sits in GPU memory during one training step? The question is not only pre-training’s. Starting from a pre-trained model and training all its parameters further, on much less data, is called full fine-tuning; each of its steps does what a pre-training step does, so it keeps the same things in memory.',
          '这样的规模带来一连串工程问题：训练一步要占多少显存；一张 GPU 放不下时，怎样把模型分到多张 GPU 上；一笔算力该训多大的模型、配多少数据；连续跑几个月，中途出了故障怎样接着训。最先要回答的是第一个，因为一张卡放不放得下，决定了要不要把模型分到多张卡上：训练的一步，显存里要放哪些东西？这个问题不只出在预训练上。从一个预训练好的模型出发，在少得多的数据上继续训练全部参数，叫全量微调；它每一步做的事和预训练一样，显存里要放的东西也一样。'))]},
 {h:()=>tx('One training step: forward, backward, optimizer update','训练的一步：前向、反向、优化器更新'),parts:[prP(
   ()=>tx('To know whether a training step fits, look at which numbers each stage of the step uses and how long they have to stay. A step has three stages. The forward pass runs a batch of text from the first layer to the last and gets the loss. The backward pass walks back from the loss and gets every parameter’s gradient. Finally, the optimizer uses these gradients to update every parameter.',
          '要知道一步训练放不放得下，就要看这一步的每个阶段用到哪些数、这些数要留多久。一步分三段。前向把一批文字从第一层算到最后一层，得到 loss。反向从 loss 往回走，得到每个参数的梯度。最后，优化器用这些梯度更新每个参数。'),
   ()=>tx(`The weights are used in both the forward and the backward pass and are rewritten in the update, so they stay in memory throughout. The gradients come out of the backward pass, one per parameter; usually the optimizer updates all the parameters together once the backward pass has finished and every gradient is there, so when the backward pass ends they are all in memory. The simplest update, gradient descent ${prM('\\theta \\leftarrow \\theta - \\eta \\, \\partial L / \\partial \\theta')} (${prM('\\eta')} is the learning rate), needs nothing but the gradients. The optimizers used to train large models (Adam, for one) also keep a few numbers for every parameter and carry them from one step to the next; these numbers are the optimizer state.`,
          `权重在前向和反向里都要用，更新时被改写，自始至终都在显存里。梯度由反向算出，每个参数一个；通常要等反向走完、全部梯度都算出来，优化器才一起更新，所以反向结束时，它们全部在显存里。最简单的更新是梯度下降 ${prM('\\theta \\leftarrow \\theta - \\eta \\, \\partial L / \\partial \\theta')}（${prM('\\eta')} 是学习率），除了梯度什么也不用。训练大模型用的优化器（比如 Adam），还要为每个参数记几个数，从这一步带到下一步，这些数叫优化器状态。`),
   ()=>tx(`One more thing: the backward pass needs some of the values the forward pass computed; they are kept from the forward pass and freed once the backward pass has used them (the basics of ${prLK('backprop')} call them saved tensors). In training these values are called activations, which has nothing to do with activation functions.`,
          `还有一样：反向要用到前向算出的一些值，它们前向时留下来，反向用完才释放（${prLK('backprop')}的知识梳理里叫 saved tensors）。训练里把这些值叫激活（activations），和激活函数无关。`),
   ()=>tx('What sets their size splits these into two kinds. The weights, the gradients and the optimizer state are one set per parameter: their size depends only on the number of parameters and on the format each is stored in, not on how much text the step processes. They are carried from step to step and usually stay in memory for the whole run; together they are called the model states. The activations depend on the text the step processes. So:',
          '这几样按大小由什么决定，分成两类。权重、梯度和优化器状态都是每个参数一份，大小只取决于参数量和每样用什么格式存，和这一步处理多少文字无关。它们从一步留到下一步，通常整个训练过程都占着显存，合称模型状态（model states）。激活则取决于这一步处理的文字。所以：'),
   ()=>prSum(tx('\\mathrm{model states}','\\mathrm{模型状态}'),[tx('\\mathrm{weights}','\\mathrm{权重}'),tx('\\mathrm{gradients}','\\mathrm{梯度}'),tx('\\mathrm{optimizer state}','\\mathrm{优化器状态}')])(),
   ()=>prSum(tx('\\mathrm{memory in training}','\\mathrm{训练时的显存}'),[tx('\\mathrm{model states}','\\mathrm{模型状态}'),tx('\\mathrm{activations}','\\mathrm{激活}')])(),
   ()=>tx('Of all these, the only one inference has to keep is the weights.','其中只做推理时需要一直留着的，只有权重。'))]},
 {h:()=>tx('Mixed-precision training: why both BF16 and FP32','混合精度训练：为什么既有 BF16，又有 FP32'),parts:[prP(
   ()=>tx(`How large the model states are depends on the number of parameters and on how many bytes each item takes per parameter, and the bytes are set by the item’s number format. A floating-point number has a sign bit, exponent bits and mantissa bits: the exponent bits set how large and how small a number can be, the mantissa bits set its precision (more in ${prLK('number-formats')}). FP32 takes 32 bits: 8 exponent bits and 23 mantissa bits. BF16 takes 16 bits, also with 8 exponent bits, so its range is almost that of FP32, but it has only 7 mantissa bits. A BF16 number takes half the bytes of an FP32 one, and a GPU computes with it much faster: an H100 (SXM) does BF16 matrix multiplication at up to 989.5 TFLOP/s and FP32 arithmetic at up to 67 TFLOP/s (see ${prLK('number-formats')}). So the matrix multiplications of the forward and backward passes run in BF16, and the weights they use are stored in BF16.`,
          `模型状态有多大，看参数量，也看每个参数的每一样占几个字节，而字节数由它的数值格式决定。浮点数由符号位、指数位和尾数位组成：指数位决定能表示多大、多小的数，尾数位决定精度（详见${prLK('number-formats')}）。FP32 占 32 位，有 8 个指数位、23 个尾数位。BF16 占 16 位，指数位同样是 8 个，所以范围和 FP32 几乎一样，尾数位却只有 7 个。BF16 每个数只占 FP32 的一半字节，GPU 算它也快得多：H100（SXM 版）做 BF16 矩阵乘法的峰值是 989.5 TFLOP/s，做 FP32 运算只有 67 TFLOP/s（见${prLK('number-formats')}）。所以前向和反向里的矩阵乘法都用 BF16 算，参与计算的权重也存成 BF16。`),
   ()=>tx(`But BF16 is not precise enough to apply the updates in. A floating-point number is written, as in scientific notation, as “1.something × 2 to some power”, and the leading 1 need not be stored, so BF16’s 7 mantissa bits give 8 significant binary digits; a weight ${prM('w')} and the BF16 numbers next to it on either side are at least ${prM('2^{-8} \\, |w|')}, 1/256 of ${prM('|w|')}, apart (how this bound follows from the format’s bits: see ${prLK('number-formats')}). When a change is added to ${prM('w')}, the result is rounded to the nearest BF16 number, so a change smaller than half that distance, that is, smaller than 1/512 of the weight (about 0.2%), rounds back to ${prM('w')} and is always lost.`,
          `可是在 BF16 上加更新，精度不够。浮点数像科学计数法那样写成“1.几位 × 2 的几次方”，开头的 1 不用存，所以 BF16 的 7 个尾数位给出 8 个有效二进制位；一个权重 ${prM('w')} 和它两边相邻的 BF16 数至少相差 ${prM('2^{-8} \\, |w|')}，即 ${prM('|w|')} 的 1/256（这个界怎样从格式的位算出来，见${prLK('number-formats')}）。${prM('w')} 加上一个改动，结果要舍入到最近的 BF16 数，所以改动不到这个距离的一半，也就是不到权重的 1/512（约 0.2%），就又舍入回 ${prM('w')}，一定丢掉。`),
   ()=>tx(`Large models are almost always trained with Adam or its variant AdamW. Each Adam step is the learning rate ${prM('\\eta')} times a ratio: the running average of the gradient divided by the square root of the running average of the squared gradient. This ratio is usually at most 1, so one step changes a parameter by at most about ${prM('\\eta')}, however large the gradient is (see ${prLK('optimizers')}). Put into this 1/512 bound, such a change is always lost once ${prM('|w|')} is above about ${prM('512 \\, \\eta')}. Pre-training learning rates are usually around ${prM('10^{-4}')}; then every weight larger than about 0.05 has its change rounded away at every step. Late in training the learning rate is lowered further, and at ${prM('10^{-5}')} the same happens to every weight larger than about 0.005. A change lost at every step never adds up, however many steps there are, and the weight never moves.`,
          `训练大模型几乎都用 Adam 或它的变体 AdamW。Adam 每一步走的是学习率 ${prM('\\eta')} 乘一个比值：梯度的滑动平均，除以梯度平方的滑动平均的平方根。这个比值一般不超过 1，所以一步给一个参数的改动最多约为 ${prM('\\eta')}，和梯度本身多大无关（见${prLK('optimizers')}）。代进 1/512 这个界，${prM('|w|')} 大于约 ${prM('512 \\, \\eta')} 时，这样的改动一定被舍掉。预训练的学习率通常在 ${prM('10^{-4}')} 上下，这时大小超过约 0.05 的权重，每一步的改动都被舍掉；训练后期学习率还会调小，降到 ${prM('10^{-5}')} 时，大小超过约 0.005 的权重也是这样。每一步都被舍掉的改动，加多少步也攒不起来，这个权重就一直不动。`),
   ()=>tx(`The fix is to keep every weight a second time, in FP32: the FP32 copy of the weights (master weights). FP32 has 24 significant binary digits, and the matching bound is ${prM('2^{-25} \\, |w|')}, about ${prM('3 \\times 10^{-8}')} of the weight, so changes the size of a learning rate are kept. Each step’s update is added to the FP32 copy, which is then rounded to BF16 for the next step’s forward and backward passes: small changes build up in the copy, and the rounded BF16 weight changes only once they carry it past the midpoint between two neighbouring BF16 numbers. The copy is used only in the update, so it counts as optimizer state.`,
          `解决办法是每个权重再用 FP32 存一份，叫 FP32 权重副本（master weights）。FP32 有 24 个有效二进制位，对应的界是 ${prM('2^{-25} \\, |w|')}，约为权重的 ${prM('3 \\times 10^{-8}')}，学习率大小的改动都留得住。每一步的更新加在 FP32 副本上，再把副本舍入成 BF16，给下一步的前向和反向用：小的改动在副本里一点点攒起来，攒到跨过两个相邻 BF16 数的中点，舍入出来的 BF16 权重才变。这份副本只在更新时用到，算在优化器状态里。`),
   ()=>tx(`The gradients and Adam’s own numbers are stored in FP32 too, for the same reason: they are built up by many small additions. A parameter’s gradient is a sum of many contributions, for example one from every token of the step (see the accumulation in ${prLK('backprop')}), and in BF16 a small contribution added to a sum that is already large is lost the same way. Adam’s two running averages are the two numbers it keeps for every parameter, m and v; each step folds the new gradient into them in a small proportion, and they are carried from step to step, so they are optimizer state too. Computing in BF16 while keeping and updating the parameters in FP32 is called mixed-precision training.`,
          `梯度和 Adam 自己记的数也用 FP32 存，原因相同：它们都是一点点加出来的。一个参数的梯度是许多份贡献加起来的，比如一步里每个 token 一份（见${prLK('backprop')}里的累加）；往一个已经不小的和上加一份小的贡献，在 BF16 里同样会丢。Adam 的那两个滑动平均，就是它为每个参数各记的两个数 m 和 v：每一步只把新梯度按一个小比例并进去，又要从一步带到下一步，也属于优化器状态。用 BF16 计算、用 FP32 保存和更新参数，这种做法叫混合精度训练。`))]},
 {h:()=>tx('Activations: every matrix multiplication keeps its input for the backward pass','激活：每个矩阵乘法的输入都要留到反向'),parts:[prP(
   ()=>tx(`The model states depend only on the number of parameters; the rest are the activations. Which values are they in a Transformer? Most of a Transformer’s computation is matrix multiplication: the vector ${prM('u_j')} of the step’s ${prM('j')}-th token is multiplied by a weight matrix ${prM('A')}, giving ${prM('y_j = A u_j')}. The q, k and v and the MLP are all computed this way (see the basics of ${prLK('architecture')}). Every token of the step uses the same ${prM('A')}, so by the multivariable chain rule its gradient is the sum of the tokens’ contributions:`,
          `模型状态只看参数量，剩下的是激活。在 Transformer 里，它们是哪些值？Transformer 的计算大部分是矩阵乘法：一步里第 ${prM('j')} 个 token 的向量 ${prM('u_j')} 乘一个权重矩阵 ${prM('A')}，得到 ${prM('y_j = A u_j')}。q、k、v 和 MLP 都是这样算的（见${prLK('architecture')}的知识梳理）。一步里的每个 token 都用同一个 ${prM('A')}，按多元链式法则，它的梯度是各个 token 的贡献之和：`),
   prFx('\\frac{\\partial L}{\\partial A} = \\sum_{j} \\frac{\\partial L}{\\partial y_j} \\, u_j^{⊤}','\\frac{\\partial L}{\\partial A} = \\sum_{j} \\frac{\\partial L}{\\partial y_j} \\, u_j^{⊤}'),
   ()=>tx(`Every term uses the input ${prM('u_j')} from the forward pass (a product’s local derivative with respect to the weight is the other input; see ${prLK('backprop')}), so every matrix multiplication keeps its input for every token. Nonlinear functions such as SiLU, the element-wise product in SwiGLU, and RMSNorm keep their inputs too, because their local derivatives need them. Three things therefore set how large the activations are: how many tokens the step processes at once (the number of sequences times their length), how many layers there are, and how wide each layer’s vectors are. Which optimizer is used makes no difference to them.`,
          `式子里每一项都要用到前向时的输入 ${prM('u_j')}（乘法对权重的局部导数就是另一个输入，见${prLK('backprop')}），所以每个矩阵乘法都要为每个 token 留下它的输入；SiLU 这样的非线性函数、SwiGLU 里的逐元素相乘和 RMSNorm 也要留下输入，因为它们的局部导数要用到。这样，激活的多少由三样东西决定：一步同时处理多少 token（序列的条数乘以每条的长度），有多少层，每层的向量有多宽。用哪种优化器，和它无关。`),
   ()=>tx(`When the model only runs inference, these values can go as soon as they have been used: once the operations after a layer have taken its intermediate results, they are no longer needed (the exception is each layer’s k and v during generation, kept for the tokens that come later; see ${prLK('kv-basics')}). Training has to keep them until the backward pass comes back to their layer. The backward pass goes from the last layer to the first and frees each layer’s activations once it has used them, so the first layer’s activations are kept longest. When the forward pass has just ended and the backward pass has not yet started, every layer’s activations are still there. This is when a step’s memory is highest: roughly the model states plus the activations of all the layers.`,
          `只做推理时，这些值用过就可以扔：一层的中间结果交给后面的运算以后，就不再需要（生成时每层的 k 和 v 例外，要留给后面的 token 用，见${prLK('kv-basics')}）。训练却要把它们一直留到反向回到这一层。反向从最后一层往第一层走，每过一层，用完这一层的激活就释放，所以第一层的激活留得最久。前向刚结束、反向还没开始时，每一层的激活都还在。这是一步里显存最高的时刻，大致是模型状态加上所有层的激活。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the choices tried, the
   pictures already played to the end, the playing speed, the challenge's result, the drill. The playhead is not kept: a picture
   still playing is recorded as played to its end, a picture already played shows finished, and one not played yet plays from its
   start. */
const SK=Object.keys(BASE), okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o);
const cfgOf=(x,ch)=>{if(!okObj(x)) return null; const c=Object.fromEntries(SK.map(k=>[k,x[k]]));
  if(ch) return CH.opts.includes(c.opt)&&CH.rcs.includes(c.rc)&&CH.mbs.includes(c.mb)?chState(c.opt,c.rc,c.mb):null;
  return MOD[c.m]&&GPU[c.gpu]&&OPT[c.opt]&&SEQS.includes(c.seq)&&MBS.includes(c.mb)&&ACCS.includes(c.acc)&&(c.rc===0||c.rc===1)?c:null;};
A.registerLab({
  id:ID,
  snap(){const pk=o=>Object.fromEntries(SK.map(k=>[k,o[k]]));
    return {v:1,mode,step,S:pk(S),pred,observed:{...observed},saved:saved?pk(saved):null,optTried:[...optTried],rcTried:[...rcTried],acSel,acTried:[...acTried],tokMoved,remView,spd,watched:[...new Set([...watched,...(playing?[sigOf(S)]:[])])],
      chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.opt,S.rc,S.mb).ok};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    const c=cfgOf(o.S,mode==='challenge'); S=c?{...c}:(mode==='challenge'?chState(CH.start.opt,CH.start.rc,CH.start.mb):{...BASE});
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    optTried=new Set(['adamw',...arr(o.optTried,v=>!!OPT[v])]); rcTried=new Set([0,...arr(o.rcTried,v=>v===0||v===1)]);
    acSel=AC.some(a=>a.v===o.acSel)?o.acSel:'a8x1'; acTried=new Set(['a8x1',...arr(o.acTried,v=>AC.some(a=>a.v===v)),acSel]);
    tokMoved=o.tokMoved===true; remView=REMS.includes(o.remView)?o.remView:'states'; spd=['slow','mid','fast'].includes(o.spd)?o.spd:'mid';
    watched.clear(); arr(o.watched,v=>typeof v==='string'&&/^\w+\|[01]\|\d+$/.test(v)&&!!MOD[v.split('|')[0]]).slice(0,100).forEach(v=>watched.add(v));
    playT=1; playing=false;
    saved=mode==='challenge'?cfgOf(o.saved,false):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) sbTouched=true; S={...p}; mode='sandbox'; playT=1;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#tmem-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#tmem-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    render();
    if(isP('states')&&!watched.has(sigOf(S))&&!playing&&!relang){playT=0; play();}
  },
  leave(){clearTimeout(annT); stageStop(); if(raf) cancelAnimationFrame(raf); raf=0; playing=false; root=null; app=null;},
  onLang(){if(root){const d=$('#tmem-math',app), open=d&&d.open, was=playing; if(raf) cancelAnimationFrame(raf); raf=0; playing=false; relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#tmem-math',app).open=true; if(was) play();}},
  onResize(){if(app){const cv=$('#tmem-curve',app); if(cv) cv._k=''; render();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#tmem-vis',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#tmem-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), c=compute(S);
      status.className='status '+s.cls; status.textContent=s.short;
      if(hidden0()){vis.innerHTML=''; read.innerHTML=esc(tx(`${c.m.name}: weights ${GB(c.infer)}; training memory hidden until you predict`,`${c.m.name}：权重 ${GB(c.infer)}；训练显存等你预测后揭晓`)); return;}
      const {tl}=ensureTL(), hw=hwX(tl,c,playT), sofar=c.states+hw.acts+hw.lg, mx=Math.max(sofar,c.cap)*1.05;
      vis.innerHTML=`<div class="tmem-dk"><i class="s" style="width:${(c.states/mx*100).toFixed(1)}%"></i><i class="a" style="width:${((hw.acts+hw.lg)/mx*100).toFixed(1)}%"></i><b style="left:${(c.cap/mx*100).toFixed(1)}%"></b></div>`;
      read.innerHTML=esc(tx(`${c.m.name}, one step: ${playT>=1?'peak':'so far'} ${GB(sofar)}; the ${c.g.name} holds ${GB(c.cap)}`,`${c.m.name} 训练一步：${playT>=1?'峰值':'到此刻'} ${GB(sofar)}；${c.g.name} 有 ${GB(c.cap)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__tmem=Object.freeze({
  compute:o=>{const c=compute({...BASE,...o}); return deep({...c,m:{...c.m},g:{...c.g}});}, aLayer:k=>aLayer(MOD[k]), aIn:k=>aIn(MOD[k]), bpp,
  timeline:o=>{const c=compute({...BASE,...o}); return deep(timeline(c));},
  /* G7: the single-change sentence the result shows for a plan, in the page's language, and the plan's values as the result shows them */
  chNextText:(o,r,mb)=>{const me=chEval(o,r,mb); return chNext(me,chBest()).replace(/<[^>]+>/g,'');}, chShown:(o,r,mb)=>{const e=chEval(o,r,mb); return {t:fixd(e.t,2),peak:GB(e.peak),fits:e.fits,ok:e.ok};},
  chEval:(o,r,mb)=>deep(chEval(o,r,mb)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), chPass:()=>deep(chPass()), chOneChange:(o,r,mb)=>deep(chOneChange(o,r,mb)), relTime, drill:s=>{const d=makeDrill(s); return deep({s:d.s,peak:d.c.peak,states:d.c.states,acts:d.c.acts,logits:d.c.logits});},
  consts:Object.freeze({MOD:deep(MOD),GPU:deep(GPU),OPT:deep(OPT),SEQS:SEQS.slice(),MBS:MBS.slice(),ACCS:ACCS.slice(),CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),AC:deep(AC),DUR:{...DUR}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},playT,playing,spd,watched:[...watched],acSel,optTried:[...optTried],rcTried:[...rcTried],acTried:[...acTried],tokMoved,remView,chSubmitted,chReveal,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});}
});
})();
