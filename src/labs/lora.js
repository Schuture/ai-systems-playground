/* lora (world post): How does LoRA fine-tune an 8B model on one GPU?
   Fidelity: Analytical. Every number is a parameter or byte count from the model's shapes and the library defaults (specs/lora.md):
   - shapes as app.js MODELS (Llama-3.1-8B: d 4,096, 32 query heads and 8 KV heads of 128, FFN 14,336, 32 layers; 70B: d 8,192,
     64 query heads, 8 KV heads, FFN 28,672, 80 layers; vocabulary 128,256); seven matrices per layer: q, k, v, o, gate, up, down;
   - a LoRA on a d_in × d_out matrix adds r × (d_in + d_out) parameters (PEFT: lora_A is r × in, lora_B is out × r);
   - bytes: full fine-tuning 18 per parameter (batch 8); LoRA: frozen weights 2 (BF16), LoRA parameters 16 (PEFT upcasts adapter
     weights to FP32 by default: FP32 weight, FP32 gradient, Adam's m and v; no separate FP32 copy);
   - QLoRA: the blocks' linear weights in NF4 with double quantization, 0.5 + 1/64 + 4/(64·256) bytes per parameter (bitsandbytes:
     blocksize 64, an FP32 absmax per block, compressed to 8 bits in blocks of 256 with one FP32 constant each); the embedding table,
     the LM head and the RMSNorm weights stay BF16 (transformers converts only nn.Linear and keeps the output embedding out);
   - activations and logits exactly as labs/training-memory.js (BF16, per token per layer; 2d with full recomputation; logits
     tokens × vocabulary × 4); the peak is everything at the end of the forward pass;
   - compute per token: full 6N, LoRA and QLoRA about 4N (no gradients for the frozen weights), recomputation adds 2N.
   GB = 10^9 bytes; a GPU's "24 / 48 / 80 GB" is read as that many 10^9 bytes. No animation in this lab (R3.9 / R3.20 do not apply).
   window.__lora is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='lora';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const NB='‑';

/* =====================================================================
   engine
   ===================================================================== */
const MOD={
 '8b':{name:`Llama${NB}3.1${NB}8B`,P:8030261248,L:32,d:4096,H:32,KV:8,hd:128,f:14336,V:128256},
 '70b':{name:`Llama${NB}3.1${NB}70B`,P:70553706496,L:80,d:8192,H:64,KV:8,hd:128,f:28672,V:128256},
 /* the capability layer's new cases only (never on the stage) */
 '405b':{name:`Llama${NB}3.1${NB}405B`,P:405853388800,L:126,d:16384,H:128,KV:8,hd:128,f:53248,V:128256}
};
const GPUS={g24:{cap:24e9},g48:{cap:48e9},g80:{cap:80e9,name:'H100'}};
const GPUK=['g24','g48','g80'];
const MATS=['q','k','v','o','gate','up','down'];
const TGT={qv:['q','v'],attn:['q','k','v','o'],all:MATS};
const TGTS=['qv','attn','all'];
const RANKS=[4,8,16,32,64,128,256];
const SEQS=[1024,2048,4096,8192,16384,32768];
const METHS=['full','lora','qlora'];
/* bytes per parameter */
const B_FULL=18, B_BF16=2, B_LORA=16;
const NF4=0.5+1/64+4/(64*256);     /* 0.515869140625: 4 bits + an 8-bit absmax per 64 + an FP32 constant per 64 × 256 */
const NF4_NODQ=0.5+4/64;           /* 0.5625 without double quantization */
/* [d_in, d_out] of each matrix (PyTorch nn.Linear(in, out)) */
function shapes(m){const q=m.H*m.hd, kv=m.KV*m.hd; return {q:[m.d,q],k:[m.d,kv],v:[m.d,kv],o:[q,m.d],gate:[m.d,m.f],up:[m.d,m.f],down:[m.f,m.d]};}
const aLayer=m=>2*(2*m.d+2*m.H*m.hd+2*m.KV*m.hd)+2*(2*m.d+4*m.f);
const aIn=m=>2*m.d;
const loraOf=(m,r,tg)=>{const sh=shapes(m); return m.L*r*TGT[tg].reduce((a,k)=>a+sh[k][0]+sh[k][1],0);};
function compute(s){
  const m=MOD[s.m], sh=shapes(m), g=GPUS[s.gpu]||GPUS.g80;
  const lin=m.L*MATS.reduce((a,k)=>a+sh[k][0]*sh[k][1],0), eo=2*m.V*m.d, nrm=m.P-lin-eo;
  const lpCfg=loraOf(m,s.r,s.tg), lp=s.meth==='full'?0:lpCfg;
  const blocks=[];
  if(s.meth==='full') blocks.push(['w',B_BF16*m.P],['g',4*m.P],['c',4*m.P],['mv',8*m.P]);
  else if(s.meth==='lora') blocks.push(['fw',B_BF16*m.P],['lr',B_LORA*lp]);
  else blocks.push(['q4',lin*NF4],['fb',B_BF16*(eo+nrm)],['lr',B_LORA*lp]);
  const acts=s.seq*m.L*(s.rc?aIn(m):aLayer(m)), logits=s.seq*m.V*4;
  blocks.push(['a',acts],['l',logits]);
  const tot=blocks.reduce((a,b)=>a+b[1],0), trained=s.meth==='full'?m.P:lp;
  return {m,sh,g,cap:g.cap,lin,eo,nrm,lp,lpCfg,trained,share:trained/m.P,blocks,acts,logits,tot,fits:tot<=g.cap,room:g.cap-tot,
    fl:(s.meth==='full'?6:4)+(s.rc?2:0),aL:aLayer(m),aI:aIn(m)};
}
const val=(c,k)=>{const b=c.blocks.find(x=>x[0]===k); return b?b[1]:0;};
function matInfo(mk,mat,r){const m=MOD[mk], [i,o]=shapes(m)[mat]; return {din:i,dout:o,W:i*o,BA:r*(i+o),ratio:r*(i+o)/(i*o)};}

/* the challenge: Llama-3.1-70B on one 48 GB GPU, 2,048-token sequences, at least 100M trained parameters, least memory */
const CH={m:'70b',gpu:'g48',seq:2048,minP:1e8,start:{meth:'lora',r:8,tg:'qv',rc:0}};
const chState=(meth,r,tg,rc)=>({m:CH.m,gpu:CH.gpu,seq:CH.seq,meth,r,tg,rc,mat:'q'});
function chEval(meth,r,tg,rc){const c=compute(chState(meth,r,tg,rc)), enough=c.trained>=CH.minP;
  return {meth,r,tg,rc,tot:c.tot,trained:c.trained,fits:c.fits,enough,ok:c.fits&&enough};}
function chAll(){const a=[]; METHS.forEach(me=>[0,1].forEach(rc=>{if(me==='full'){a.push(chEval(me,16,'all',rc)); return;} TGTS.forEach(tg=>RANKS.forEach(r=>a.push(chEval(me,r,tg,rc))));})); return a;}
function chBest(){return chAll().filter(e=>e.ok).sort((a,b)=>a.tot-b.tot)[0];}
const chMinimal=e=>{const b=chBest(); return e.ok&&Math.abs(e.tot-b.tot)<=1e-6*b.tot;};
/* the full list behind “See all settings that pass” (the owner's review of batch 8, 2026-10-07: only the plans that meet the goal):
   every plan that fits and trains at least 100M parameters, least memory first */
function chPass(){return chAll().filter(e=>e.ok).sort((a,b)=>a.tot-b.tot);}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const GB=b=>`${fixd(b/1e9,1)} GB`;
const big=n=>A.fmt.big?A.fmt.big(n):fint(n);
const bigH=n=>n>999999&&A.ui.big?A.ui.big(n):esc(n>999999?big(n):fint(n));
const pctS=v=>{const p=v*100; return p>=10?fixd(p,0)+'%':p>=1?fixd(p,1)+'%':p>=0.1?fixd(p,2)+'%':fixd(p,3)+'%';};
const ratioS=k=>{const r1=Math.round(k*10)/10; return Math.abs(r1-Math.round(r1))<1e-9?fint(Math.round(r1)):fixd(r1,1);};
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
const labLinkZ=id=>{const h=labLink(id); return /^[A-Za-z0-9]/.test(h.replace(/<[^>]+>/g,''))?' '+h:h;};
const MX=t=>`<div class="lora-fx">${A.ui.math(t)}</div>`;
/* H2 (the owner, 2026-10-09): a formula with its name in words on the left, its chain broken at “=” onto lines that line up; a line may be [op, tex] */
/* the owner's review of 2026-10-09 (no Show the math row scrolls sideways): a line too long for its box wraps before a top-level “+” or
   “−”, each new line under the line's first term; {} gives the piece's leading operator the spacing of a binary operator. On a phone
   the name goes on its own line above the chain (lora.css). */
const topSplit=t=>{const out=[]; let d=0, s=0; for(let i=0;i<t.length;i++){const c=t[i]; if('{(['.includes(c)) d++; else if('})]'.includes(c)) d--;
  else if(!d&&c===' '&&i>s&&(t.startsWith(' + ',i)||t.startsWith(' - ',i))){out.push(t.slice(s,i)); s=i+1;}} out.push(t.slice(s)); return out;};
const WRP=t=>topSplit(t).map((p,i)=>`<span>${A.ui.math(i?`{} ${p}`:p)}</span>`).join('');
const EQ=(name,lines,op='=')=>`<div class="lora-eq">${lines.map((l,i)=>{const [o,t]=Array.isArray(l)?l:[op,l]; return `<div class="lora-eq-n">${i?'':name}</div><div class="lora-eq-o">${o?A.ui.math(o):''}</div><div class="lora-fx lora-wr">${WRP(t)}</div>`;}).join('')}</div>`;
/* H1 (the owner, 2026-10-09): d_in, d_out and the shapes typeset, never written as strings */
const DIN=()=>A.ui.math('d_{in}'), DOUT=()=>A.ui.math('d_{out}');
const G2=v=>`${fixd(v/1e9,2)} GB`;
const S2=vs=>`${fixd(vs.reduce((a,v)=>a+Math.round(v/1e7),0)/100,2)} GB`;
const methName=me=>({full:tx('Full fine-tuning','全量微调'),lora:'LoRA',qlora:'QLoRA'}[me]);
const tgName=tg=>({qv:tx('q and v','q 和 v'),attn:tx('the 4 attention matrices','注意力的 4 个矩阵'),all:tx('all 7 matrices','全部 7 个矩阵')}[tg]);
const tgShort=tg=>({qv:tx('q and v','q 和 v'),attn:tx('4 attention','注意力的 4 个'),all:tx('all 7','全部 7 个')}[tg]);
const gpuName=k=>k==='g80'?tx('one 80 GB H100','一张 80 GB 的 H100'):k==='g48'?tx('one 48 GB GPU','一张 48 GB 的卡'):tx('one 24 GB GPU','一张 24 GB 的卡');
const gpuShort=k=>k==='g80'?'H100':k==='g48'?tx('48 GB GPU','48 GB 的卡'):tx('24 GB GPU','24 GB 的卡');
const fitW=c=>c.fits?tx('fits','放得下'):tx('does not fit','放不下');
const loraDesc=s=>tx(`LoRA of rank r = ${s.r} on ${tgName(s.tg)}`,`秩 r = ${s.r} 的 LoRA 加在${tgName(s.tg)}上`);

/* =====================================================================
   module state
   ===================================================================== */
const BASE={m:'8b',meth:'full',r:16,tg:'all',seq:4096,rc:0,gpu:'g80',mat:'q'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let matSeen=new Set(['q']), l3Tried=new Set(['full']), l6Tried=new Set(['lora']), l6Sel='lora', r4Moved=false, seqOOM=false, remView='';
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, drillSeed=1, drillOpen=false;
const shownPrev={};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 full:()=>tx('Full fine-tuning updates every parameter of the model, as pretraining does.','全量微调会更新模型的每一个参数，和预训练一样。'),
 states:()=>tx('Model states are what training keeps for every parameter it updates: the BF16 weights (2 bytes), the FP32 gradients (4) and the optimizer state, which is the FP32 copy of the weights plus Adam’s m and v (4 each): 18 bytes in all.','模型状态是训练时每个要更新的参数都带着的东西：BF16 权重（2 字节）、FP32 梯度（4 字节），以及优化器状态，也就是 FP32 权重副本加上 Adam 的 m 和 v（各 4 字节），一共 18 字节。'),
 gb:()=>tx('GB here is 10⁹ bytes. A GPU’s “80 GB” is also read as 80 × 10⁹ bytes in this lab (and 24 GB, 48 GB likewise).','这里的 GB 是 10⁹ 字节。这个 Lab 也把 GPU 的“80 GB”按 80 × 10⁹ 字节算（24 GB、48 GB 同理）。'),
 lora:()=>tx('LoRA (low-rank adaptation) freezes the model’s weights and trains two small matrices B and A beside each matrix it adapts; their product B·A, scaled, is added to that matrix’s output.','LoRA（low-rank adaptation，低秩适配）把模型的权重冻住，在要改的每个矩阵旁边加两个小矩阵 B 和 A，只训练它们。B·A 乘上缩放系数后，加到这个矩阵的输出上。'),
 rank:()=>tx('The rank of a matrix is how many independent rows (or columns) it has at most. B has only r columns, so B·A has rank at most r, however large it is.','矩阵的秩是它最多有几行（或几列）线性无关。B 只有 r 列，所以不管 B·A 多大，它的秩最多是 r。'),
 alpha:()=>tx('α is a scaling factor in the formula above: the larger α, the more B·A changes the output. PEFT calls it lora_alpha, with defaults r = 8 and α = 8. torchtune (PyTorch’s own fine-tuning library) notes in its configs that it is “usually” '+A.ui.math('\\alpha = 2r')+'.','α 是上面公式里的缩放系数：α 越大，B·A 对输出的改动越大。PEFT 里它叫 lora_alpha，默认 r = 8、α = 8。torchtune（PyTorch 官方的微调库）在配置里注明，通常取 '+A.ui.math('\\alpha = 2r')+'。'),
 peft:()=>tx('PEFT (parameter-efficient fine-tuning) is Hugging Face’s library for fine-tuning methods that train only a few parameters, LoRA among them. The PEFT defaults this page quotes come from its source code.','PEFT（parameter-efficient fine-tuning，参数高效微调）是 Hugging Face 的一个库，实现了 LoRA 这类只训练少量参数的微调方法。这一页引用的 PEFT 默认值，都来自它的源码。'),
 tt:()=>tx('torchtune is PyTorch’s own library for fine-tuning LLMs. This page quotes its single-GPU configs for Llama 3.1 8B and the memory its README reports.','torchtune 是 PyTorch 官方的 LLM 微调库。这一页引用了它给 Llama 3.1 8B 的单卡配置，以及它的 README 报告的显存。'),
 bnb:()=>tx('bitsandbytes is the quantization library QLoRA uses: it stores the weights in 4 bits and turns them back into BF16 for the multiplication.','bitsandbytes 是 QLoRA 用的量化库：它把权重存成 4 位，做乘法前再还原成 BF16。'),
 frozen:()=>tx('Frozen: the weights are used in the forward and backward pass but never updated, so they need no gradient and no optimizer state.','冻结：前向和反向都要用这些权重，但从不更新它们，所以它们不需要梯度，也不需要优化器状态。'),
 act:()=>tx('Activations are the intermediate results the forward pass keeps so that the backward pass can compute the gradients from them.','激活是前向时留下来的中间结果，反向时要靠它们算出梯度。'),
 rc:()=>tx('Recomputation (activation checkpointing): each layer keeps only its input; in the backward pass the layer’s forward is run again to get its intermediate results back. It costs one more forward pass, about 2N per token.','重算（activation checkpointing）：每一层只留下它的输入。反向走到这一层时，把这一层的前向再跑一遍，拿回中间结果。代价是多一遍前向，每个 token 约多 2N 次运算。'),
 qlora:()=>tx('QLoRA is LoRA on a base model whose frozen linear weights are stored in 4 bits (NF4). The LoRA matrices B and A are trained as usual.','QLoRA 就是在一个冻结的线性层权重存成 4 位（NF4）的模型上做 LoRA。LoRA 的 B 和 A 照常训练。'),
 nf4:()=>tx('NF4 (4-bit NormalFloat) is a 4-bit format with 16 levels placed for normally distributed weights. Every block of 64 weights shares one quantization constant (the block’s absmax). With double quantization these constants are themselves stored in 8 bits, every 256 of them sharing one FP32 constant. That makes about 4.13 bits, 0.516 bytes, per weight.','NF4（4-bit NormalFloat）是一种 4 位格式，16 个取值按正态分布的权重来摆放。每 64 个权重共用一个量化常数（这一块的最大绝对值）。开了双重量化，这些量化常数再压成 8 位，每 256 个共用一个 FP32 常数。平均每个权重约 4.13 位，也就是 0.516 字节。'),
 targets:()=>tx('Each layer has 7 matrices: q, k, v and o in the attention block, gate, up and down in the MLP block. A LoRA is added only to the matrices you choose; the others are frozen and untouched.','每层有 7 个矩阵：attention 块的 q、k、v、o，MLP 块的 gate、up、down。LoRA 只加在你选的矩阵上，其余的冻结，完全不变。'),
 /* the owner's review of 2026-10-08: other labs give 8N with recomputation (training-memory, model-parallel, parallelism); that is full
    training. Here LoRA skips the frozen weights' gradients, so its 6N with recomputation is the same rule in a different setting. */
 n4:()=>tx('Compute per token: the forward pass costs about 2N (N parameters, one multiply and one add each). The backward pass computes two kinds of gradients, for the inputs and for the weights, about 2N each. Full fine-tuning needs both: 6N without recomputation, and 8N with it, because recomputation runs every layer’s forward pass once more (2N). LoRA gives the frozen weights no gradient, so its backward pass is only about 2N: about 4N without recomputation and about 6N with it. Derived from this rule, not measured; the LoRA matrices’ own arithmetic and attention’s are left out.',
           '每个 token 的计算：前向约 2N（N 个参数，每个做一次乘法、一次加法）。反向要算两种梯度，输入的梯度和权重的梯度，各约 2N。全量微调两种都要算，不重算是 6N。重算要把每一层的前向再跑一遍，多 2N，所以是 8N。LoRA 不给冻结的权重算梯度，反向只要约 2N，所以不重算约 4N，重算约 6N。这是按这条规则推出来的，不是实测。LoRA 矩阵自己的计算和注意力的计算都没算进去。')
};
const attnDef=t=>DEF(t,A.terms&&A.terms.attention?A.terms.attention():TIP.targets());
const mlpDef=t=>DEF(t,A.terms&&A.terms.mlp?A.terms.mlp():TIP.targets());

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'lt2',t:{en:'Under 2%',zh:'不到 2%'}},{v:'2-10',t:{en:'2% to 10%',zh:'2% 到 10%'}},{v:'10-30',t:{en:'10% to 30%',zh:'10% 到 30%'}},{v:'gt30',t:{en:'Over 30%',zh:'30% 以上'}}];
const PRED_RIGHT='lt2';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const S8=o=>({...BASE,meth:'lora',...o});
const C3=me=>compute({...BASE,meth:me});
const lv3Res=me=>{const c=C3(me); return `${GB(c.tot)}${tx(', ','，')}${fitW(c)}`;};
const L6=[{v:'lora',s:{meth:'lora',rc:1},t:{en:'LoRA, frozen weights in BF16 (initial setup)',zh:'LoRA，冻结的权重用 BF16（初始设置）'},d:{en:'2 bytes for every frozen parameter.',zh:'每个冻结的参数 2 字节。'}},
  {v:'qlora',s:{meth:'qlora',rc:1},t:{en:'QLoRA: the frozen linear weights in 4 bits',zh:'QLoRA：冻结的线性层权重用 4 位'},d:{en:'The 7 matrices of every layer in NF4; the embedding table and the LM head stay BF16.',zh:'每层 7 个矩阵存成 NF4，embedding 表和 LM head 还是 BF16。'}},
  {v:'qnorc',s:{meth:'qlora',rc:0},t:{en:'QLoRA, but without recomputation',zh:'QLoRA，但不重算'},d:{en:'This option also turns recomputation off, to see whether 4 bits can replace it.',zh:'这一项同时关掉了重算，看看 4 位能不能代替重算。'}}];
const S6=v=>({...BASE,m:'70b',r:16,tg:'all',...L6.find(o=>o.v===v).s});
const lv6Res=v=>{const c=compute(S6(v)); return `${GB(c.tot)}${tx(', ','，')}${fitW(c)}`;};
const lever=(id,aria,value,opts)=>`<div class="lora-lv" id="${id}">`+A.ui.levers({aria,value,options:opts})+`</div>`;
const matSeg=(id,inline)=>A.ui.seg({id,key:'mat',label:tx('Matrix in the chart','图里看哪个矩阵'),options:MATS.map(k=>[k,k]),value:S.mat,inline});
const rCtl=(id,inline)=>A.ui.range({id,key:'r',label:tx('Rank r','秩 r'),min:0,max:RANKS.length-1,value:Math.max(0,RANKS.indexOf(S.r)),out:String(S.r),inline});
const tgCtl=(id,inline)=>A.ui.seg({id,key:'tg',label:tx('Which matrices get a LoRA','LoRA 加在哪些矩阵上'),options:TGTS.map(k=>[k,tgShort(k)]),value:S.tg,inline});
const seqCtl=(id,inline)=>A.ui.range({id,key:'seq',label:tx('Tokens per sequence','每条序列的 token 数'),min:0,max:SEQS.length-1,value:Math.max(0,SEQS.indexOf(S.seq)),out:fint(S.seq),inline});
const rcCtl=(id,inline)=>A.ui.seg({id,key:'rc',label:tx('Recomputation','重算'),options:[['0',tx('Off','不重算')],['1',tx('Every layer','每一层都重算')]],value:String(S.rc),inline});

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('Full fine-tuning of the 8B keeps 144.5 GB of model states. Can we train only a small part?','全量微调 8B 光模型状态就要 144.5 GB。能不能只训练一小部分？'),
  enter:()=>({...BASE}),
  body:()=>{const m=MOD['8b'];
   return tx(`<p>${labLink('sft')} (supervised fine-tuning, the previous lab) teaches a base model to answer questions from question-and-answer conversations. This time we do it on one GPU. ${DEF('Full fine-tuning',TIP.full())} of ${m.name} with the usual mixed-precision AdamW keeps 18 bytes of ${DEF('model states',TIP.states())} per parameter (see ${labLink('training-memory')}). That is ${GB(B_FULL*m.P)} before any activations, more than one H100 with 80 ${DEF('GB',TIP.gb())} holds.</p>`+
     `<p>${DEF('LoRA',TIP.lora())} does it differently. It freezes each big matrix W it adapts and does not change a single parameter of it. Beside W it adds two small matrices, A and B, and trains only those two.</p>`+
     `<p>In the forward pass the input x takes two paths (${J('fig','see the chart')}): one through the frozen W, the other through A and then B. The two results are added to give the output h. A squeezes the input into a short list of numbers, and B expands that list to as many numbers as W x has, so A and B are small and there are few parameters to train.</p>`,
     `<p>上一个 Lab 的${labLinkZ('sft')}（监督微调）用一问一答的对话，教基础模型学会回答问题。这一次只用一张卡来做这件事。用常见的混合精度 AdamW ${DEF('全量微调',TIP.full())} ${m.name}，每个参数要带 18 字节的${DEF('模型状态',TIP.states())}（见${labLinkZ('training-memory')}）。还没算激活就已经有 ${GB(B_FULL*m.P)}，一张 80 ${DEF('GB',TIP.gb())} 的 H100 放不下。</p>`+
     `<p>${DEF('LoRA',TIP.lora())} 换了一种做法。它把模型里要改的大矩阵 W 冻住，一个参数都不改，再在 W 旁边加两个小矩阵 A 和 B，只训练这两个。</p>`+
     `<p>前向时，输入 x 走两条路（${J('fig','见图')}）。一条乘冻结的 W，另一条先乘 A、再乘 B。两条路的结果加起来，就是输出 h。A 先把输入压成很短的一组数，B 再把这组数展开成和 W x 一样长，所以 A 和 B 都很小，要训练的参数很少。</p>`);}},
 {id:'lowrank',title:{en:'Two small matrices, B and A',zh:'两个小矩阵：B 和 A'},expect:()=>[...matSeen].some(k=>k!=='q'),showMe:{mat:'down'},
  lead:()=>tx('W stays frozen; B·A is the change to W.','W 冻住不动，B·A 就是对 W 的修改。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>How small are B and A? W turns a ${DIN()}-dimensional input into a ${DOUT()}-dimensional output, so it has ${DOUT()} rows and ${DIN()} columns. B has ${DOUT()} rows and r columns, and A has r rows and ${DIN()} columns. r is called the ${DEF('rank',TIP.rank())}, a small number you choose, here 16.</p>`+
       `<p>In the forward pass the input x goes through the frozen W and, beside it, through A and then B, and the two results are added. As a formula:</p>`+EQ(`output ${A.ui.math('h')}`,['W x + \\frac{α}{r} B A x'])+
       `<p>So B·A is the change to W. It is as large as W, but its rank is at most r. B starts as all zeros, so at step 0 the model is exactly the base model. ${DEF('α',TIP.alpha())} is a scaling factor.</p>`+
       `<p>Every layer has 7 matrices: q, k, v and o in the ${attnDef('attention block')}, gate, up and down in the ${mlpDef('MLP block')}. Pick another one and see how many parameters its W, B and A have.</p>`+matSeg('lora-ic-mat',true),
     `<p>B 和 A 有多大？W 把 ${DIN()} 维的输入变成 ${DOUT()} 维的输出，所以它有 ${DOUT()} 行、${DIN()} 列。B 有 ${DOUT()} 行、r 列，A 有 r 行、${DIN()} 列。r 叫${DEF('秩',TIP.rank())}，是你自己选的一个不大的整数，这里取 16。</p>`+
       `<p>前向时，输入 x 一路乘冻结的 W，一路先乘 A、再乘 B，两路的结果相加。写成公式是：</p>`+EQ(`输出 ${A.ui.math('h')}`,['W x + \\frac{α}{r} B A x'])+
       `<p>所以 B·A 就是对 W 的修改。它和 W 一样大，但秩最多是 r。训练开始时 B 全是 0，所以第 0 步的模型和原来的模型一模一样。${DEF('α',TIP.alpha())} 是缩放系数。</p>`+
       `<p>每层有 7 个矩阵：${attnDef('attention 块')}的 q、k、v、o，${mlpDef('MLP 块')}的 gate、up、down。换一个矩阵，看看它的 W、B 和 A 各有多少参数。</p>`+matSeg('lora-ic-mat',true)),
  obs:()=>{if(S.mat==='q'&&![...matSeen].some(k=>k!=='q')) return ''; const mi=matInfo('8b',S.mat,16);
   return tx(`<p><b>${S.mat}: W has ${bigH(mi.W)} parameters; B and A together have ${bigH(mi.BA)}, ${pctS(mi.ratio)} of W.</b></p><p>The wider the matrix, the smaller B and A’s share: B and A grow with the sides of W, W with their product.</p>`,
             `<p><b>${S.mat}：W 有 ${bigH(mi.W)} 个参数，B 和 A 一共 ${bigH(mi.BA)} 个，是 W 的 ${pctS(mi.ratio)}。</b></p><p>矩阵越大，B 和 A 占的比例越小。B 和 A 的参数只跟 W 的两条边长之和成正比，W 的参数却跟两条边的乘积成正比。</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},label:{en:'Result',zh:'结果'},gate:()=>!!pred,skip:true,
  lead:()=>tx('On all 7 matrices, what share of the parameters is trained?','加在全部 7 个矩阵上，要训练的参数占多少？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>On every matrix, B and A have only a small share of W’s parameters. The largest share is on the narrower k and v, and even there it is only about 2% of W.</p><p>Now add a LoRA of rank r = 16 to all 7 matrices of every layer. About what share of all the model’s parameters gets trained?</p>${ch}`,
             `<p>每个矩阵旁边的 B 和 A，参数都只占 W 的一小部分。占得最多的是窄一些的 k 和 v，也只有 W 的 2% 左右。</p><p>现在把秩 r = 16 的 LoRA 加在每层全部 7 个矩阵上。要训练的参数，大约占模型全部参数的多少？</p>${ch}`);},
  obs:()=>{if(!pred) return ''; const c=compute(S8({})), right=pred===PRED_RIGHT, mlp=matInfo('8b','down',16);
   return tx(`<p class="${right?'okline':'badline'} lora-verdict"><b>You predicted “${esc(predT(pred))}”${right?', and that is right.':`; the answer is “${esc(predT(PRED_RIGHT))}”.`}</b> A LoRA of rank 16 on all 7 matrices trains ${bigH(c.lp)} parameters, ${pctS(c.lp/c.m.P)} of all ${bigH(c.m.P)}.</p><p>On every matrix B and A are under 2% of W. The MLP block’s three matrices hold most of the parameters, and on them B and A are only about ${fixd(mlp.ratio*100,1)}% of W, so the total stays small too.</p>`,
             `<p class="${right?'okline':'badline'} lora-verdict"><b>你预测的是“${esc(predT(pred))}”，${right?'预测对了。':`正确答案是“${esc(predT(PRED_RIGHT))}”。`}</b>秩 16 的 LoRA 加在全部 7 个矩阵上，要训练 ${bigH(c.lp)} 个参数，只占全部 ${bigH(c.m.P)} 个的 ${pctS(c.lp/c.m.P)}。</p><p>每个矩阵的 B 和 A 都不到 W 的 2%。模型的参数大多在 MLP 块的 3 个矩阵里，它们的 B 和 A 只有 W 的约 ${fixd(mlp.ratio*100,1)}%，所以合起来也很少。</p>`);}},
 {id:'memory',title:{en:'Memory: what goes away',zh:'显存：省掉了哪几块'},expect:()=>l3Tried.has('lora'),showMe:{l3:'lora'},
  lead:()=>tx('The frozen weights need no gradients and no optimizer state.','冻结的权重不需要梯度，也不需要优化器状态。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>In full fine-tuning every parameter carries, besides its BF16 weight, an FP32 gradient, an FP32 copy of the weight, and Adam’s m and v. LoRA freezes W, so these parameters need no gradient and no optimizer state.</p>`+
     `<p>B and A are trained. ${DEF('PEFT',TIP.peft())} (Hugging Face’s parameter-efficient fine-tuning library) keeps them in FP32 by default: the weight, its gradient, m and v take 4 bytes each, 16 bytes per LoRA parameter.</p>`+
     `<p>The model is ${MOD['8b'].name}, with one sequence of 4,096 tokens and no recomputation. Pick a fine-tuning method and see how much memory one step needs.</p>`+
     lever('lora-ic-l3',tx('Pick a fine-tuning method','选一种微调方法'),'l3:'+S.meth,[
       {v:'l3:full',title:tx('Full fine-tuning (initial setup)','全量微调（初始设置）'),desc:tx(`Trains all ${big(MOD['8b'].P)} parameters, 18 bytes each.`,`训练全部 ${big(MOD['8b'].P)} 个参数，每个 18 字节。`),result:esc(lv3Res('full'))},
       {v:'l3:lora',title:tx('LoRA, rank r = 16 on all 7 matrices','LoRA：秩 r = 16，加在全部 7 个矩阵上'),desc:tx('Freezes every parameter and trains only B and A.','冻住全部参数，只训练 B 和 A。'),result:esc(l3Tried.has('lora')?lv3Res('lora'):'')}]),
     `<p>全量微调时，每个参数除了 BF16 权重，还要带 FP32 梯度、FP32 权重副本，以及 Adam 的 m 和 v。LoRA 冻住了 W，这些参数就不需要梯度，也不需要优化器状态。</p>`+
     `<p>B 和 A 要训练。${DEF('PEFT',TIP.peft())}（Hugging Face 的参数高效微调库）默认把它们存成 FP32。权重、梯度、m、v 各 4 字节，每个 LoRA 参数一共 16 字节。</p>`+
     `<p>模型是 ${MOD['8b'].name}，一条 4,096 个 token 的序列，不重算。选一种微调方法，看看一步要多少显存。</p>`+
     lever('lora-ic-l3',tx('Pick a fine-tuning method','选一种微调方法'),'l3:'+S.meth,[
       {v:'l3:full',title:tx('Full fine-tuning (initial setup)','全量微调（初始设置）'),desc:tx(`Trains all ${big(MOD['8b'].P)} parameters, 18 bytes each.`,`训练全部 ${big(MOD['8b'].P)} 个参数，每个 18 字节。`),result:esc(lv3Res('full'))},
       {v:'l3:lora',title:tx('LoRA, rank r = 16 on all 7 matrices','LoRA：秩 r = 16，加在全部 7 个矩阵上'),desc:tx('Freezes every parameter and trains only B and A.','冻住全部参数，只训练 B 和 A。'),result:esc(l3Tried.has('lora')?lv3Res('lora'):'')}])),
  obs:()=>{if(!l3Tried.has('lora')) return ''; const f=C3('full'), l=C3('lora');
   if(S.meth==='full') return tx(`<p><b>Back to full fine-tuning: ${GB(f.tot)}, which does not fit one H100.</b></p><p>Its gradients and optimizer state alone are ${GB(16*f.m.P)}.</p>`,`<p><b>回到全量微调：${GB(f.tot)}，一张 H100 放不下。</b></p><p>光梯度和优化器状态就有 ${GB(16*f.m.P)}。</p>`);
   return tx(`<p><b>With LoRA the memory drops from ${GB(f.tot)} to ${GB(l.tot)}: it fits one H100.</b></p><p>The trained parameters shrank by more than 99%, but the memory only to about a quarter. The frozen weights stay in memory, because the forward pass uses them. The activations hardly shrink either, because the backward pass still goes through every layer to reach each layer’s B and A. The biggest block is now the activations.</p>`,
             `<p><b>改成 LoRA：显存从 ${GB(f.tot)} 降到 ${GB(l.tot)}，一张 H100 放得下了。</b></p><p>要训练的参数少了 99% 以上，显存却只降到约四分之一。冻结的权重还得放在显存里，因为前向要用它们。激活也几乎没少，因为反向还是要穿过每一层，才能把梯度送到每一层的 B 和 A。现在最大的一块是激活。</p>`);}},
 {id:'rank',title:{en:'The rank, and which matrices',zh:'秩，以及加在哪些矩阵上'},expect:()=>r4Moved,showMe:{r:64},
  lead:()=>tx('Parameters and their memory grow in step with r.','参数和它们占的显存跟着 r 成比例地涨。'),
  enter:()=>S8({}),
  body:()=>tx(`<p>The larger the rank r, the wider B and A, the richer the changes B·A can represent, and the more parameters there are to train. Each matrix adds:</p>`+EQ('LoRA parameters',['r \\times (d_{in} + d_{out})'])+
     `<p>With PEFT, if you do not say which matrices to adapt, a LoRA on Llama goes on q and v only. It can also go on the 4 matrices of the attention block, or on all 7. Real training configs usually adapt most of them: ${DEF('torchtune',TIP.tt())} (PyTorch’s own fine-tuning library) puts its single-GPU LoRA of the 8B on 6 of the 7, all but k. A matrix without a LoRA does not change at all in training.</p>`+
     rCtl('lora-ic-r',true)+tgCtl('lora-ic-tg',true),
     `<p>秩 r 越大，B 和 A 越宽，B·A 能表示的修改越丰富，要训练的参数也越多。每个矩阵加的参数是：</p>`+EQ('LoRA 参数',['r \\times (d_{in} + d_{out})'])+
     `<p>用 PEFT 微调 Llama 时，如果不指定加在哪些矩阵上，LoRA 只加在 q 和 v 上。也可以加在 attention 块的 4 个矩阵上，或者 7 个矩阵全加。实际的训练配置常加在大部分矩阵上，比如 ${DEF('torchtune',TIP.tt())}（PyTorch 官方的微调库）给 8B 的单卡配置，加了除 k 以外的 6 个。没加 LoRA 的矩阵，训练时完全不变。</p>`+
     rCtl('lora-ic-r',true)+tgCtl('lora-ic-tg',true)),
  obs:()=>{if(!r4Moved) return ''; const c=compute(S), b=compute(S8({})), k=c.lp/b.lp;
   const rel=Math.abs(k-1)<1e-9?tx('as many trained parameters as the initial setup','要训练的参数和初始设置一样多'):k>1?tx(`${ratioS(k)} times as many trained parameters as the initial setup`,`要训练的参数是初始设置的 ${ratioS(k)} 倍`):tx(`only ${fixd(k*100,0)}% as many trained parameters as the initial setup`,`要训练的参数只有初始设置的 ${fixd(k*100,0)}%`);
   return tx(`<p><b>Rank r = ${S.r} on ${tgName(S.tg)}: ${rel}; their block takes ${GB(B_LORA*c.lp)}.</b></p><p>The initial setup is rank 16 on all 7 matrices. Double r and both the parameters and that block double, yet next to the frozen weights and the activations the block stays small. How large a rank a task needs is not something this lab can tell. torchtune’s config notes beside lora_rank that “higher increases accuracy and memory”.</p>`,
             `<p><b>秩 r = ${S.r}，加在${tgName(S.tg)}上：${rel}，LoRA 参数这一块占 ${GB(B_LORA*c.lp)}。</b></p><p>初始设置是秩 16、7 个矩阵全加。r 翻一倍，参数和这一块显存也翻一倍，可和冻结的权重、激活比，它还是很小。一个任务要多大的秩，这个 Lab 回答不了。torchtune 的配置在 lora_rank 旁边写着“higher increases accuracy and memory”（越大，效果越好，显存也越多）。</p>`);}},
 {id:'long',title:{en:'Long sequences still need recomputation',zh:'序列变长：激活还要靠重算'},expect:()=>S.rc===1&&S.seq>=16384,showMe:{seq:16384,rc:1},
  lead:()=>tx('LoRA does not shrink the activations; long sequences stop fitting.','LoRA 不让激活变少；序列一长就放不下。'),
  enter:()=>S8({}),
  body:()=>tx(`<p>${DEF('Activations',TIP.act())} grow with the tokens processed at once. LoRA changes only which parameters are trained; the forward and backward passes go through every layer as before, so the activations are about as large as in full fine-tuning, and this lab counts them as equal.</p>`+
     `<p>Make the sequence longer and see where it stops fitting. Then turn on ${DEF('recomputation',TIP.rc())}: each layer keeps only its input and runs its forward again in the backward pass (as in ${labLink('training-memory')}).</p>`+
     seqCtl('lora-ic-seq',true)+rcCtl('lora-ic-rc',true),
     `<p>${DEF('激活',TIP.act())}和一次处理的 token 数成正比。LoRA 只改了哪些参数要训练，前向和反向还是一层不少地走一遍，所以激活和全量微调差不多一样多，这个 Lab 按一样多算。</p>`+
     `<p>把序列拉长，看从哪里开始放不下。再打开${DEF('重算',TIP.rc())}，每一层只留下输入，反向时把这一层的前向再跑一遍（见${labLinkZ('training-memory')}）。</p>`+
     seqCtl('lora-ic-seq',true)+rcCtl('lora-ic-rc',true)),
  obs:()=>{const c=compute(S);
   if(S.seq<16384&&!seqOOM) return '';
   if(!c.fits) return tx(`<p class="badline"><b>${fint(S.seq)} tokens${S.rc?', every layer recomputed':', no recomputation'}: ${GB(c.tot)}, more than the H100’s 80 GB.</b></p><p>${S.rc?'Even the logits and the layer inputs are too much at this length.':`The activations alone take ${GB(c.acts)}. Turn on recomputation.`}</p>`,
                          `<p class="badline"><b>${fint(S.seq)} 个 token，${S.rc?'每一层都重算':'不重算'}：${GB(c.tot)}，超过了 H100 的 80 GB。</b></p><p>${S.rc?'这么长的序列，logits 和每层的输入也放不下了。':`光激活就有 ${GB(c.acts)}。打开重算试试。`}</p>`);
   /* the owner's review of 2026-10-08: other labs say 8N with recomputation; that is full training, so the setting is named here */
   if(S.rc) return tx(`<p><b>${fint(S.seq)} tokens, every layer recomputed: ${GB(c.tot)}, it fits again.</b></p><p>The price is one more forward pass. LoRA with recomputation costs about 6N operations per token (N is the number of parameters), against about 4N without it.</p><p>Full fine-tuning with recomputation costs 8N (see ${labLink('training-memory')}): its backward pass also computes a gradient for every weight, 2N more than LoRA’s.</p>`,
                      `<p><b>${fint(S.seq)} 个 token，每一层都重算：${GB(c.tot)}，又放得下了。</b></p><p>代价是多跑一遍前向。LoRA 加上重算，每个 token 约要 6N 次运算（N 是参数量），不重算时约 4N。</p><p>全量微调加上重算是 8N（见${labLinkZ('training-memory')}）。它的反向还要给每个权重算梯度，比 LoRA 多 2N。</p>`);
   return tx(`<p><b>${fint(S.seq)} tokens, no recomputation: ${GB(c.tot)}, it fits.</b></p><p>Make it longer.</p>`,`<p><b>${fint(S.seq)} 个 token，不重算：${GB(c.tot)}，放得下。</b></p><p>再拉长一些。</p>`);}},
 {id:'qlora',title:{en:'QLoRA: frozen weights in 4 bits',zh:'QLoRA：冻结的权重存成 4 位'},expect:()=>l6Tried.has('qlora'),showMe:{l6:'qlora'},
  lead:()=>tx('In 4 bits, even the 70B’s frozen weights fit one H100.','冻结的权重存成 4 位，70B 也能放进一张 H100。'),
  enter:()=>S6('lora'),
  body:()=>{const m=MOD['70b'];
   return tx(`<p>Now ${m.name}. The frozen weights still sit in GPU memory. In BF16, 2 bytes each, they alone take more than one H100.</p>`+
     `<p>${DEF('QLoRA',TIP.qlora())} stores the frozen weights of every layer’s 7 matrices in the 4-bit ${DEF('NF4',TIP.nf4())} format: with its quantization constants, about 0.52 bytes per parameter. The embedding table and the LM head stay in BF16. The arithmetic does not use 4 bits: right before a matrix is used, it is turned back into BF16, and the matrix multiplication runs in BF16 (see ${labLink('number-formats')}).</p>`+
     `<p>Each sequence has 4,096 tokens, a LoRA of rank r = 16 sits on all 7 matrices, and every layer is recomputed. Pick a way to store the frozen weights and see how much memory one step of the 70B needs.</p>`+
     lever('lora-ic-l6',tx('Pick how to store the frozen weights','选一种存冻结权重的办法'),'l6:'+l6Sel,L6.map(o=>({v:'l6:'+o.v,title:L(o.t),desc:L(o.d),result:esc(l6Tried.has(o.v)?lv6Res(o.v):'')}))),
     `<p>换成 ${m.name}。冻结的权重还是要放在显存里，BF16 下每个参数 2 字节，光这些就超过一张 H100。</p>`+
     `<p>${DEF('QLoRA',TIP.qlora())} 把每层 7 个矩阵的冻结权重存成 4 位的 ${DEF('NF4',TIP.nf4())}，加上量化常数，每个参数约 0.52 字节。embedding 表和 LM head 还是 BF16。计算并不用 4 位：每个矩阵用到之前，先还原成 BF16，矩阵乘法还是在 BF16 里做（见${labLinkZ('number-formats')}）。</p>`+
     `<p>每条序列 4,096 个 token，秩 r = 16 的 LoRA 加在全部 7 个矩阵上，每一层都重算。选一种存冻结权重的办法，看看 70B 一步要多少显存。</p>`+
     lever('lora-ic-l6',tx('Pick how to store the frozen weights','选一种存冻结权重的办法'),'l6:'+l6Sel,L6.map(o=>({v:'l6:'+o.v,title:L(o.t),desc:L(o.d),result:esc(l6Tried.has(o.v)?lv6Res(o.v):'')}))));},
  obs:()=>{if(!l6Tried.has('qlora')&&!l6Tried.has('qnorc')) return ''; const b=compute(S6('lora')), c=compute(S6(l6Sel));
   if(l6Sel==='lora') return tx(`<p><b>The initial setup: the frozen BF16 weights alone take ${GB(val(b,'fw'))}; in all ${GB(b.tot)}, which does not fit.</b></p>`,`<p><b>初始设置：光冻结的 BF16 权重就有 ${GB(val(b,'fw'))}，一共 ${GB(b.tot)}，放不下。</b></p>`);
   if(l6Sel==='qlora') return tx(`<p><b>With QLoRA the memory drops from ${GB(b.tot)} to ${GB(c.tot)}: the 70B fits one H100.</b></p><p>The 4-bit weights take about a quarter of their BF16 size. QLoRA saves memory, not compute: every matrix is still multiplied in BF16.</p>`,
                              `<p><b>改成 QLoRA：显存从 ${GB(b.tot)} 降到 ${GB(c.tot)}，一张 H100 放得下 70B 了。</b></p><p>4 位的权重只有 BF16 的约四分之一。QLoRA 省的是显存，不是计算：每个矩阵还是在 BF16 里做乘法。</p>`);
   return tx(`<p class="badline"><b>QLoRA without recomputation: ${GB(c.tot)}, which does not fit.</b></p><p>4 bits shrink only the weights. The 70B’s layers are wider and there are 80 of them, so its activations alone take ${GB(c.acts)}: recomputation is still needed.</p>`,
             `<p class="badline"><b>QLoRA 但不重算：${GB(c.tot)}，放不下。</b></p><p>4 位只压了权重。70B 每层更宽，又有 80 层，光激活就有 ${GB(c.acts)}：重算还是少不了。</p>`);}},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('LoRA saves the gradients and optimizer state, not the weights, not the activations.','LoRA 省的是梯度和优化器状态，不是权重，也不是激活。'),
  enter:()=>S8({}),
  /* the recap keeps each rule and says what follows from it; it does not ask the reader to recall a number from an earlier part (the
     owner's review of 2026-10-08). Each button names the case it puts in the chart, where its numbers are (R6.23). */
  body:()=>{
   const item=(k,en,zh)=>`<div class="lora-rem"><p>${tx(en,zh)}</p>${k?`<button type="button" class="btn ghost sm lora-remb" data-rem="${k}" aria-pressed="${remView===k}">${remBtn(k)}</button>`:''}</div>`;
   return item('small',`<b>“LoRA makes the model smaller”: no.</b> The whole model stays in GPU memory and every layer still runs, so a GPU must first hold every frozen weight. After training, B·A can be added into W: the LoRA README says this adds no inference latency, so the served model is as large as before.`,
                  `<b>“LoRA 让模型变小”：不对。</b>整个模型都还在显存里，每一层都还要跑，所以一张卡先得放下全部冻结的权重。训练完，B·A 可以直接加进 W。LoRA 的 README 说，这样不增加推理延迟，上线的模型和原来一样大。`)+
     item('acts',`<b>“LoRA saves the activations”: no.</b> It saves the gradients and optimizer state of the frozen weights. The activations still grow with the sequence, so a long sequence stops fitting. Then what helps is recomputation; a smaller rank does not.`,
                 `<b>“LoRA 省的是激活”：不对。</b>它省的是冻结权重的梯度和优化器状态。激活还是跟着序列长度涨，序列一长就放不下。这时该开的是重算，把秩调小没有用。`)+
     item('rank',`<b>“The larger the rank, the better”: not for free.</b> The parameters and their memory grow in step with r. Whether a larger rank trains a better model, this lab cannot tell; compare on your own task.`,
                 `<b>“秩越大越好”：不一定，而且有代价。</b>参数和它们占的显存跟着 r 成比例地涨。秩大了效果会不会更好，这个 Lab 回答不了，要在自己的任务上比。`)+
     item('q4',`<b>“QLoRA computes in 4 bits”: no.</b> It stores the frozen weights in 4 bits and turns each matrix back into BF16 right before using it. So QLoRA saves memory, enough to fit a much larger model on one GPU, while the compute per token stays the same.`,
                 `<b>“QLoRA 用 4 位做计算”：不对。</b>它只是把冻结的权重存成 4 位，每个矩阵用之前先还原成 BF16 再算。所以 QLoRA 省的是显存，一张卡能放下大得多的模型，每个 token 的计算量却不变。`)+
     item(null,`<b>“Training under 1% of the parameters must be much worse than full fine-tuning”: not necessarily.</b> The ${cite(SRC.lora,'LoRA README')} reports results comparable or superior to full fine-tuning on the GLUE benchmark with RoBERTa and DeBERTa (up to 1.5B parameters). The ${cite(SRC.qlora,'QLoRA README')} says LoRA on a 4-bit base model preserves full 16-bit fine-tuning task performance, from its tests on LLaMA and T5. For another model or task, evaluate on your own.`,
               `<b>“只训练不到 1% 的参数，效果一定比全量微调差很多”：不一定。</b>${cite(SRC.lora,'LoRA 的 README')} 报告，在 GLUE 基准上微调 RoBERTa 和 DeBERTa（最大 1.5B 个参数）时，效果和全量微调相当或更好。${cite(SRC.qlora,'QLoRA 的 README')} 说，在 4 位的基础模型上做 LoRA，保住了 16 位全量微调的效果，这是他们在 LLaMA 和 T5 上测的结果。换了模型和任务，要自己评测。`)+
     A.shell.toScenarios();}}
];
const sIdx=id=>STEPS.findIndex(s=>s.id===id), pn=id=>sIdx(id)+1;
const REMS={small:()=>S8({}),acts:()=>S8({seq:16384}),rank:()=>S8({r:64}),q4:()=>S6('qlora')};
const remBtn=k=>remView===k?tx('In the chart now','图里正显示'):{small:tx('Show the frozen weights','在图里看冻结的权重'),acts:tx('Show a long sequence','在图里看长序列'),rank:tx('Show rank 64','在图里看秩 64'),q4:tx('Show the 70B with QLoRA','在图里看 70B 用 QLoRA')}[k];
const leadOf=s=>typeof s.lead==='function'?s.lead():'';

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'l8',t:{en:`LoRA on the 8B, one H100`,zh:'8B，一张 H100 做 LoRA'},d:{en:'A team fine-tunes on the one 80 GB GPU it has',zh:'团队用手头的一张 80 GB 的卡微调'},s:{...BASE,meth:'lora'}},
 {id:'q8',t:{en:'QLoRA on the 8B, one 24 GB GPU',zh:'8B，一张 24 GB 的卡做 QLoRA'},d:{en:'Fine-tuning on a gaming-class card',zh:'在一张游戏卡级别的显卡上微调'},s:{...BASE,meth:'qlora',rc:1,gpu:'g24'}},
 {id:'q70',t:{en:'QLoRA on the 70B, one H100',zh:'70B，一张 H100 做 QLoRA'},d:{en:'A model one GPU cannot hold in BF16',zh:'用 BF16 存，一张卡放不下的模型'},s:{...BASE,m:'70b',meth:'qlora',rc:1}},
 {id:'f70',t:{en:'Full fine-tuning of the 70B: how many H100s?',zh:'70B 全量微调要多少张 H100'},d:{en:'Training every parameter of the 70B',zh:'训练 70B 的每一个参数'},s:{...BASE,m:'70b',meth:'full',rc:1}}
];
const KEYS=['m','meth','r','tg','seq','rc','gpu'];
const presetNow=()=>PRESETS.findIndex(p=>KEYS.every(k=>p.s[k]===S[k]));
const SRC={
 peftCfg:'https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/src/peft/tuners/lora/config.py',
 peftLayer:'https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/src/peft/tuners/lora/layer.py',
 peftConst:'https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/src/peft/utils/constants.py',
 peftMap:'https://github.com/huggingface/peft/blob/532a05dd505c28993119b7715ee286f4234bf51b/src/peft/mapping_func.py',
 bnbF:'https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py',
 bnbM:'https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/nn/modules.py',
 bnbOps:'https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/backends/cuda/ops.py',
 tfBnb:'https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/integrations/bitsandbytes.py',
 tfBase:'https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/quantizers/base.py',
 qlora:'https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/README.md',
 qlora65:'https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/scripts/finetune_guanaco_65b.sh',
 lora:'https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/README.md',
 tt:'https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/README.md',
 ttLora:'https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/configs/llama3_1/8B_lora_single_device.yaml',
 ttQlora:'https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/configs/llama3_1/8B_qlora_single_device.yaml',
 ttFull:'https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/recipes/configs/llama3_1/8B_full_single_device.yaml',
 ttTut:'https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/docs/source/tutorials/qlora_finetune.rst',
 paLora:'https://arxiv.org/abs/2106.09685',paQlora:'https://arxiv.org/abs/2305.14314',paRs:'https://arxiv.org/abs/2312.03732'
};
const GIB=()=>DEF('GiB',tx('GiB = 2³⁰ = 1,073,741,824 bytes, about 7% more than a GB (10⁹ bytes). torchtune reports memory in GiB; the rest of this page uses GB.','GiB = 2³⁰ = 1,073,741,824 字节，比 GB（10⁹ 字节）大约 7%。torchtune 用 GiB 报显存；这一页别的地方都用 GB。'));
function scenHTML(i){
  const p=PRESETS[i], c=compute(S);
  if(p.id==='l8') return tx(`<p><b>LoRA on ${c.m.name}: ${GB(c.tot)}, ${GB(c.room)} to spare on the H100.</b></p><p>The biggest block is the activations, ${GB(c.acts)}, not the weights. ${DEF('torchtune',TIP.tt())} (PyTorch’s own fine-tuning library) reports in its README a peak of 30.3 ${GIB()} for LoRA on the 8B on one 48 GB A6000. Its settings differ from these: 2 sequences of 2,048 tokens per step, and torch compile on (${cite(SRC.tt,'README')}).</p>`,
                            `<p><b>${c.m.name} 做 LoRA：${GB(c.tot)}，H100 还剩 ${GB(c.room)}。</b></p><p>最大的一块是激活（${GB(c.acts)}），不是权重。${DEF('torchtune',TIP.tt())}（PyTorch 官方的微调库）在 README 里报告，8B 在一张 48 GB 的 A6000 上做 LoRA，峰值是 30.3 ${GIB()}。它的设置和这里不同，每步送 2 条 2,048 个 token 的序列，还打开了 torch compile（${cite(SRC.tt,'README')}）。</p>`);
  if(p.id==='q8'){const lo=compute({...S,meth:'lora'}), sv=fint(Math.round((lo.tot-c.tot)/1e9));
    return tx(`<p><b>QLoRA on ${c.m.name} with recomputation: ${GB(c.tot)}, it fits the 24 GB card.</b></p><p>With recomputation, plain LoRA fits too, at ${GB(lo.tot)}. QLoRA saves about ${sv} GB more, room for longer sequences or more sequences per step.</p><p>${DEF('torchtune',TIP.tt())} (PyTorch’s own fine-tuning library) reports in its README a peak of 7.4 ${GIB()} for QLoRA on the 8B on one 24 GB RTX 4090, with 2 sequences of 2,048 tokens per step (${cite(SRC.tt,'README')}).</p>`,
              `<p><b>${c.m.name} 做 QLoRA，打开重算：${GB(c.tot)}，24 GB 的卡放得下。</b></p><p>同样打开重算，LoRA 也放得下，要 ${GB(lo.tot)}。QLoRA 又省下约 ${sv} GB，可以用来放更长的序列，或者每步多放几条序列。</p><p>${DEF('torchtune',TIP.tt())}（PyTorch 官方的微调库）在 README 里报告，8B 在一张 24 GB 的 RTX 4090 上做 QLoRA，峰值是 7.4 ${GIB()}，每步 2 条 2,048 个 token 的序列（${cite(SRC.tt,'README')}）。</p>`);}
  if(p.id==='q70') return tx(`<p><b>QLoRA on ${c.m.name} with recomputation: ${GB(c.tot)} on the 80 GB H100.</b></p><p>In BF16 its frozen weights alone would take ${GB(B_BF16*c.m.P)}. The QLoRA README makes the same kind of claim for a 65B model: it “reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU” (${cite(SRC.qlora,'README')}).</p>`,
                             `<p><b>${c.m.name} 做 QLoRA，打开重算：${GB(c.tot)}，一张 80 GB 的 H100 放得下。</b></p><p>如果用 BF16，光冻结的权重就要 ${GB(B_BF16*c.m.P)}。QLoRA 的 README 对 65B 的模型说过同样的事：它“reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU”（${cite(SRC.qlora,'README')}）。</p>`);
  const ms=B_FULL*c.m.P, n=Math.ceil(ms/80e9);
  return tx(`<p><b>Full fine-tuning of ${c.m.name}: the model states alone are ${GB(ms)}, at least ${n} H100s just to hold them.</b></p><p>That is before any activation. Training every parameter of a model this size means splitting its model states over many GPUs (see ${labLink('parallelism')}); with QLoRA the same model fits one.</p>`,
            `<p><b>${c.m.name} 全量微调：光模型状态就有 ${GB(ms)}，至少要 ${n} 张 H100 才放得下。</b></p><p>这还没算激活。要训练这么大模型的每一个参数，就得把模型状态分到很多张卡上（见${labLinkZ('parallelism')}）。用 QLoRA，同一个模型一张卡就放得下。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=e=>e.meth==='full'?tx(`full fine-tuning, ${e.rc?'recompute':'no recomputation'}`,`全量微调，${e.rc?'重算':'不重算'}`):
  tx(`${methName(e.meth)}, r = ${e.r} on ${tgShort(e.tg)}, ${e.rc?'recompute':'no recomputation'}`,`${methName(e.meth)}，r = ${e.r}，${tgShort(e.tg)}，${e.rc?'重算':'不重算'}`);
const chSame=(e,s)=>e.meth===s.meth&&e.rc===s.rc&&(e.meth==='full'||(e.r===s.r&&e.tg===s.tg));
/* the summary layer after Submit (review codex-b9v3 应修 5): the plan's memory in three blocks, each with the setting that shrinks it, next to
   the plan with the least memory (once the goal is met, or once the learner asks for the answer); the full list of plans behind a
   disclosure, closed by default */
const chParts=e=>{const c=compute(chState(e.meth,e.r,e.tg,e.rc)), sum=ks=>c.blocks.filter(b=>ks.includes(b[0])).reduce((a,b)=>a+b[1],0);
  return {w:sum(['w','fw','q4','fb']),t:sum(['g','c','mv','lr']),a:sum(['a','l'])};};
const CH_ROWS=[
 {k:'w',name:()=>tx('Weights','权重'),by:()=>tx('the method sets it','由方法决定'),set:e=>e.meth==='qlora'?tx('4-bit, frozen','4 位，冻结'):e.meth==='lora'?tx('BF16, frozen','BF16，冻结'):tx('BF16, trained','BF16，要训练'),same:(a,b)=>a.meth===b.meth},
 {k:'t',name:()=>tx('LoRA parameters, with gradients and optimizer state','LoRA 参数（含梯度和优化器状态）'),by:()=>tx('the rank and the matrices set it','由秩和矩阵决定'),
  set:e=>e.meth==='full'?tx('no LoRA: gradients and optimizer state of every parameter','没有 LoRA，这是全部参数的梯度和优化器状态'):`r = ${e.r}${tx(', ','，')}${tgShort(e.tg)}`,same:(a,b)=>a.meth!=='full'&&b.meth!=='full'&&a.r===b.r&&a.tg===b.tg},
 {k:'a',name:()=>tx('Activations and logits','激活和 logits'),by:()=>tx('recomputation sets it','由重算决定'),set:e=>e.rc?tx('recompute','重算'):tx('no recomputation','不重算'),same:(a,b)=>a.rc===b.rc}];
function chSumHTML(me,best){
  const two=!!best, pm=chParts(me), pb=two?chParts(best):null;
  const cell=(e,p,k)=>`<td><span class="lora-sset">${esc(CH_ROWS.find(r=>r.k===k).set(e))}</span><b class="mono">${esc(GB(p[k]))}</b></td>`;
  let h=`<table class="lora-sum${two?' two':''}"><caption>${esc(two?tx('Where your plan’s memory goes, next to the plan with the least memory','你的方案和最省的方案，显存各花在哪里'):tx('Where your plan’s memory goes','你的方案，显存花在哪里'))}</caption>`+
    `<thead><tr><th scope="col">${esc(tx('Memory block','显存的一块'))}</th><th scope="col">${esc(tx('Your plan','你的方案'))}</th>${two?`<th scope="col">${esc(tx('Least memory','最省的方案'))}</th>`:''}</tr></thead><tbody>`;
  CH_ROWS.forEach(r=>{const diff=two&&!r.same(me,best);
    h+=`<tr${diff?' class="lora-diff"':''}><th scope="row">${esc(r.name())}<span>${esc(r.by())}</span></th>${cell(me,pm,r.k)}${two?cell(best,pb,r.k):''}</tr>`;});
  h+=`<tr class="lora-sumtot"><th scope="row">${esc(tx('In all','合计'))}<span>${esc(tx('goal: at most 48 GB','目标：不超过 48 GB'))}</span></th><td${me.fits?'':' class="lora-bad"'}><b class="mono">${esc(GB(me.tot))}</b></td>${two?`<td><b class="mono">${esc(GB(best.tot))}</b></td>`:''}</tr>`;
  h+=`<tr><th scope="row">${esc(tx('Trained parameters','要训练的参数'))}<span>${esc(tx('goal: at least 100M','目标：至少 100M 个'))}</span></th><td${me.enough?'':' class="lora-bad"'}><b class="mono">${esc(big(me.trained))}</b></td>${two?`<td><b class="mono">${esc(big(best.trained))}</b></td>`:''}</tr></tbody></table>`;
  if(two&&CH_ROWS.some(r=>!r.same(me,best))) h+=`<p class="small muted lora-sumkey">${esc(tx('A line at the left of a row: your setting there differs from the plan with the least memory.','左边带竖线的行，是你的设置和最省的方案不一样的地方。'))}</p>`;
  return h;
}
/* the one main gap, in words: the block where the plan spends the most more than the best plan */
function chGapHTML(me,best){
  if(me.meth==='full'){const x=Math.floor(chParts(me).t/GPUS[CH.gpu].cap/10)*10;
    return tx(`<p>Full fine-tuning trains every parameter. Their gradients and optimizer state alone take more than ${x} times the card’s memory.</p>`,`<p>全量微调要训练全部参数。光是这些参数的梯度和优化器状态，就是这张卡显存的 ${x} 多倍。</p>`);}
  if(me.fits&&!me.enough) return tx('<p>Your plan fits but trains too few parameters. A larger rank, or more matrices, trains more of them, at 16 bytes each.</p>','<p>你的方案放得下，但要训练的参数不够 100M。秩调大一些，或者加在更多矩阵上，就能多训练一些参数，每个多占 16 字节。</p>');
  const pm=chParts(me), pb=chParts(best), k=[['w',pm.w-pb.w],['t',pm.t-pb.t],['a',pm.a-pb.a]].sort((x,y)=>y[1]-x[1])[0][0];
  if(k==='w') return tx('<p>The biggest gap is the weights. LoRA keeps the frozen weights in BF16; QLoRA stores them in 4 bits, about a quarter of the size.</p>','<p>差距最大的是权重。LoRA 把冻结的权重留在 BF16，QLoRA 把它们存成 4 位，只要约四分之一。</p>');
  if(k==='a') return tx('<p>The biggest gap is the activations. Without recomputation every layer keeps its intermediate results for the backward pass; with it, each layer keeps only its input.</p>','<p>差距最大的是激活。不重算时，每一层都要给反向留下中间结果。重算时，每一层只留输入。</p>');
  const x=me.trained/best.trained;
  return tx(`<p>The gap is the LoRA parameters. Your plan trains ${ratioS(x)} times as many as the plan with the least memory, and every LoRA parameter costs 16 bytes wherever it sits.</p>`,
            `<p>差距在 LoRA 参数这一块。你的方案要训练的参数是最省方案的 ${ratioS(x)} 倍，而 LoRA 参数不管加在哪里，每个都是 16 字节。</p>`);
}
/* before the answer is shown: the block to shrink next, and the setting that sets it */
function chNextHTML(me){
  if(me.meth==='full'||me.fits) return chGapHTML(me,null);
  const p=chParts(me), can={w:me.meth!=='qlora',t:me.trained>1e8,a:!me.rc}, by=ks=>ks.slice().sort((x,y)=>p[y]-p[x])[0];
  const ks=['w','t','a'].filter(k=>can[k]); if(!ks.length) return '';
  const k=by(ks), top=by(['w','t','a']);
  const NM={w:tx('the weights','权重'),t:tx('the LoRA parameters','LoRA 参数'),a:tx('the activations and logits','激活和 logits')};
  const BY={w:tx('The method sets it.','它由方法决定。'),t:tx('The rank and the matrices set it.','它由秩和矩阵决定。'),a:tx('Recomputation sets it.','它由重算决定。')};
  const NO={w:tx('The weights are already in 4 bits and cannot shrink further. ','权重已经存成 4 位，不能再小了。'),a:tx('The activations are already recomputed and cannot shrink further. ','激活已经在重算，不能再小了。'),t:''};
  return tx(`<p>${k!==top?NO[top]:''}The block to shrink next is ${NM[k]}, ${GB(p[k])}. ${BY[k]}</p>`,`<p>${k!==top?NO[top]:''}接下来要压的是${NM[k]}这一块，有 ${GB(p[k])}。${BY[k]}</p>`);
}
let allOpen=false;
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.meth,S.r,S.tg,S.rc), best=chBest(), minimal=me.ok&&chMinimal(me);
  let h;
  const good=tx(`<p class="okline lora-mt"><b>Goal met: ${GB(me.tot)} on the 48 GB card, with ${bigH(me.trained)} trained parameters.</b>`,`<p class="okline lora-mt"><b>达成目标：48 GB 的卡用了 ${GB(me.tot)}，要训练 ${bigH(me.trained)} 个参数。</b>`);
  if(minimal) h=good+tx(' And no plan that meets the goal uses less memory.</p>','而且达标的方案里，没有比它更省显存的。</p>');
  else if(me.ok) h=good+tx(' But a plan with less memory also meets it.</p>','不过还有更省的方案。</p>');
  else h=tx(`<p class="badline lora-mt"><b>Not yet. ${!me.fits?`It needs ${GB(me.tot)}, more than the 48 GB.`:`It fits (${GB(me.tot)}), but trains only ${big(me.trained)} parameters, under 100M.`}</b></p>`,
            `<p class="badline lora-mt"><b>还没达到目标。${!me.fits?`要 ${GB(me.tot)}，超过了 48 GB。`:`放得下（${GB(me.tot)}），但只训练 ${big(me.trained)} 个参数，不到 100M。`}</b></p>`);
  /* the summary: the plan next to the best one once revealed; a minimal plan is the best one, so it stands alone */
  h+=chSumHTML(me,chReveal&&!minimal?best:null);
  if(!chReveal) return h+chNextHTML(me)+tx('<p class="small">Hint: every LoRA parameter costs 16 bytes wherever it sits, so the plan with the least memory is the one whose trained parameters land closest above 100M.</p>',
         '<p class="small">提示：LoRA 的参数不管加在哪里，每个都是 16 字节。所以最省显存的方案，是要训练的参数刚好超过 100M 的那个。</p>')+
      `<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;
  if(!minimal) h+=chGapHTML(me,best);
  const rows=chPass();
  let all=A.ui.cmpChart({title:tx('Memory the plans that meet the goal need for the 70B on the 48 GB card (2,048-token sequences)','达标的方案在 48 GB 的卡上微调 70B 要多少显存（每条序列 2,048 个 token）'),
    aria:tx('Memory of the plans that meet the goal: ','达标的方案要的显存：')+rows.map(e=>`${chName(e)}${tx(': ','：')}${GB(e.tot)}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:chName(e),value:e.tot/1e9,text:GB(e.tot),ok:true,me:chSame(e,S),best:chMinimal(e),
      tags:(chMinimal(e)?[{text:tx('least memory','最省')}]:[]).concat(chSame(e,S)?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:48,goalLabel:tx('48 GB: the card','48 GB：这张卡'),okText:tx('✓ fits, at least 100M trained','✓ 放得下，且至少训练 100M 个'),noText:tx('does not fit','放不下'),goalText:tx('48 GB: the card','48 GB：这张卡')});
  /* why the other plans miss: named in words, their memory on hover (R6.20) */
  const st=chEval(CH.start.meth,CH.start.r,CH.start.tg,CH.start.rc), lo=chEval('lora',best.r,best.tg,1), noRc=chEval('qlora',best.r,best.tg,0);
  const more=Math.min(...rows.filter(e=>e.tg!=='all').map(e=>e.trained))/best.trained-1;
  const FR=[[1/5,'a fifth','五分之一'],[1/4,'a quarter','四分之一'],[1/3,'a third','三分之一'],[1/2,'half','一半']], fr=FR.reduce((a,f)=>Math.abs(f[0]-more)<Math.abs(a[0]-more)?f:a);
  const frac=tx(fr[1],fr[2]), fracZ=frac, nfit=e=>DEF(tx('does not fit','放不下'),tx(`It needs ${GB(e.tot)}.`,`要 ${GB(e.tot)}。`));
  all+=tx(`<p class="small">The initial setup (PEFT’s default, r = 8 on q and v, no recomputation) needs ${GB(st.tot)} and trains only ${big(st.trained)} parameters.</p>`+
        `<p class="small">Two settings bring the big blocks down. QLoRA stores the frozen weights in 4 bits. Plain LoRA keeps them in BF16, and even with recomputation it ${nfit(lo)}. Recomputation shrinks the activations, and QLoRA without it ${nfit(noRc)} either. So every plan above is QLoRA with recomputation.</p>`+
        `<p class="small">What is left is the LoRA parameters, 16 bytes each wherever they sit, so the best plan is the one whose trained parameters land just above 100M. On all 7 matrices a rank of 8 is enough. On q and v, or on the 4 attention matrices, the rank has to be much larger, which gives about ${frac} more parameters and so a little more memory.</p>`,
        `<p class="small">初始设置（PEFT 的默认：r = 8，只加在 q 和 v 上，不重算）要 ${GB(st.tot)}，而且只训练 ${big(st.trained)} 个参数。</p>`+
        `<p class="small">显存里的大块要靠两个设置压下来。QLoRA 把冻结的权重存成 4 位。只用 LoRA 的话，权重留在 BF16，就算重算也${nfit(lo)}。重算把激活压下来，QLoRA 不重算同样${nfit(noRc)}。所以上面列出的方案都是 QLoRA 加重算。</p>`+
        `<p class="small">剩下的是 LoRA 参数，加在哪里都是每个 16 字节，所以要挑参数刚好超过 100M 的组合。加在全部 7 个矩阵上，秩取 8 就够了。只加在 q 和 v 上，或者注意力的 4 个矩阵上，秩要大得多，参数多出约${fracZ}，显存也就多一点。</p>`);
  all+=tx(`<p class="small muted">Note: this challenge counts only memory. It does not check how well a rank of 8 or 4-bit weights train; that needs an evaluation on your task. A real run also needs some memory for the CUDA context, the allocator and a temporary BF16 copy of the matrix being used, which this lab leaves out; with 3 GB to spare, check it on the real card.</p>`,
        `<p class="small muted">注意：这个挑战只算显存，没有考虑秩 8 和 4 位权重对训练效果的影响，那要在你的任务上评测。真实训练还要给 CUDA 上下文、显存分配器，以及正在用的那个矩阵的临时 BF16 副本留显存，这个 Lab 没有算。最省的方案只剩 3 GB 左右的余量，要在真卡上确认。</p>`);
  h+=`<details class="lora-allp" id="lora-allp"${allOpen?' open':''}><summary>${esc(tx(`See all settings that pass (${rows.length})`,`看全部可行的设置（${rows.length} 种）`))}</summary>${all}</details>`;
  return h;
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`Eight short parts. See how LoRA trains two small matrices beside a frozen one, predict how few parameters that is, then find out which memory LoRA saves, which it does not, and what QLoRA and recomputation add. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先看 LoRA 怎样在冻结的矩阵旁边训练两个小矩阵，再猜它要训练的参数有多少，然后找出 LoRA 省了哪块显存、没省哪块，以及 QLoRA 和重算还能省什么。第一次学从这里开始。`),
  sandbox:tx('Four real situations, from LoRA on the 8B on one H100 to the 70B. Pick one to see what fills the GPU.','四个真实情况，从在一张 H100 上用 LoRA 微调 8B，到 70B。选一个，看显存都被什么占了。'),
  challenge:tx(`Fine-tune ${MOD['70b'].name} on one 48 GB GPU with 2,048-token sequences. Choose the method, the rank, the matrices and recomputation so that it fits, trains at least 100M parameters and uses as little memory as possible.`,
               `在一张 48 GB 的卡上微调 ${MOD['70b'].name}，每条序列 2,048 个 token。选方法、秩、加在哪些矩阵上、要不要重算：要放得下，要训练的参数至少 100M 个，显存尽量少。`)
});
/* LoRA's own activations (x·A: r values per token per adapted matrix), left out of the chart: the base case at its largest (the 70B,
   rank 16, all 7, 4,096 tokens, FP32 as PEFT computes lora_A in the adapter dtype) and the largest case (70B, rank 256, all 7, the longest
   sequence, BF16) */
const XA={base:4096*MOD['70b'].L*7*16*4, max:SEQS[SEQS.length-1]*MOD['70b'].L*7*RANKS[RANKS.length-1]*2};
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> every number on this page is counted from the model’s shapes and the library defaults below; ${J('math','“Show the math”')} works it out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Parameters: a LoRA on a matrix with ${DIN()} inputs and ${DOUT()} outputs adds ${A.ui.math('r \\times (d_{in} + d_{out})')} parameters. PEFT’s lora_A is ${A.ui.math('r \\times d_{in}')} and lora_B is ${A.ui.math('d_{out} \\times r')} (${cite(SRC.peftLayer,'layer.py')}).</li>`+
 `<li>Bytes per parameter: full fine-tuning 18 (as ${labLink('training-memory')}). LoRA: every frozen parameter 2 (BF16); every LoRA parameter 16, because ${DEF('PEFT',TIP.peft())} casts B and A from BF16 up to FP32 by default (<code>autocast_adapter_dtype=True</code>, ${cite(SRC.peftMap,'mapping_func.py')}): the FP32 weight, its FP32 gradient and Adam’s m and v; an FP32 weight needs no separate FP32 copy.</li>`+
 `<li>QLoRA: the 7 matrices of every layer in NF4 with double quantization: 4 bits per weight, one 8-bit quantization constant per block of 64 weights, one FP32 constant per 256 of those, ${fixd(NF4,4)} bytes per parameter (${fixd(NF4_NODQ,4)} without double quantization; ${DEF('bitsandbytes',TIP.bnb())} ${cite(SRC.bnbF,'functional.py')}, ${cite(SRC.bnbOps,'ops.py')}). The embedding table, the LM head and the RMSNorm weights stay BF16: transformers replaces only <code>nn.Linear</code> modules and keeps the output embedding out (${cite(SRC.tfBnb,'bitsandbytes.py')}, ${cite(SRC.tfBase,'base.py')}). Matrix multiplications run in BF16 on weights turned back from 4 bits (${cite(SRC.bnbM,'modules.py')}).</li>`+
 `<li>Activations and logits exactly as ${labLink('training-memory')} counts them. LoRA’s activations are counted as equal to full fine-tuning’s, because the backward pass still runs through every frozen layer. Real runs differ a little both ways: a frozen matrix without a LoRA need not keep its input for a weight gradient, so a little less is kept; PEFT casts the input of every matrix with a LoRA to B and A’s precision, which can keep one more copy of it. Neither changes the conclusion: LoRA does not save the activations, and long sequences still need recomputation. The peak is at the end of the forward pass; with recomputation one layer’s activations come back during the backward pass, less than the logits freed before it.</li></ul></div>`+
 `<div><div class="eyebrow">Left out</div><ul>`+
 `<li>The CUDA context, the allocator’s cached memory and temporary buffers.</li>`+
 `<li>QLoRA’s temporary BF16 copy of the matrix in use: at most ${GB(2*MOD['70b'].d*MOD['70b'].f)}, the 70B’s largest.</li>`+
 `<li>LoRA’s own small activations: every token keeps r numbers (x·A) for every matrix with a LoRA. At rank 16 and 4,096 tokens they are under ${fixd(Math.ceil(XA.base/1e8)/10,1)} GB. With the rank and the sequence at their maximum they reach about ${fint(Math.round(XA.max/1e9))} GB for the 70B even in BF16, and the chart then reads low.</li>`+
 `<li>The per-matrix constants of double quantization, under 1 MB in all. PEFT’s default dropout is 0, so there is no dropout mask.</li>`+
 `<li>Compute per token (full fine-tuning 6N, or 8N with recomputation; LoRA about 4N, or about 6N with recomputation) leaves out attention’s own arithmetic, the LoRA matrices’ arithmetic (6 × the LoRA parameters per token, under 1% here) and the time spent turning 4-bit weights back into BF16.</li>`+
 `<li>Quality: this page counts parameters and memory only. Part ${pn('remember')} quotes what the LoRA and QLoRA READMEs report, with the models they tested; how well a given rank, set of matrices or 4-bit weights train on your task needs your own evaluation.</li>`+
 `<li>A GPU’s “80 GB” is read as 80 × 10⁹ bytes (24 and 48 GB likewise).</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>PEFT: ${cite(SRC.peftCfg,'config.py')} (r = 8, lora_alpha = 8, B initialised to 0), ${cite(SRC.peftConst,'constants.py')} (Llama: q_proj and v_proj), ${cite(SRC.peftLayer,'layer.py')}, ${cite(SRC.peftMap,'mapping_func.py')}.</li>`+
 `<li>bitsandbytes ${cite(SRC.bnbF,'functional.py')}, ${cite(SRC.bnbOps,'ops.py')}, ${cite(SRC.bnbM,'modules.py')}; transformers ${cite(SRC.tfBnb,'bitsandbytes.py')}; ${cite(SRC.qlora,'QLoRA README')}; ${cite(SRC.lora,'LoRA README')}; torchtune ${cite(SRC.tt,'README')} and its ${cite(SRC.ttLora,'LoRA')} and ${cite(SRC.ttQlora,'QLoRA')} configs.</li>`+
 `<li>Papers, cited by title only: LoRA: Low-Rank Adaptation of Large Language Models (arXiv 2106.09685); QLoRA: Efficient Finetuning of Quantized LLMs (arXiv 2305.14314).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上的所有数字，都由模型的形状和下面这些库的默认设置算出。${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>参数：一个有 ${DIN()} 个输入、${DOUT()} 个输出的矩阵，加 LoRA 后多 ${A.ui.math('r \\times (d_{in} + d_{out})')} 个参数。PEFT 的 lora_A 是 ${A.ui.math('r \\times d_{in}')}，lora_B 是 ${A.ui.math('d_{out} \\times r')}（${cite(SRC.peftLayer,'layer.py')}）。</li>`+
 `<li>每个参数的字节：全量微调 18 字节（和${labLinkZ('training-memory')}一样）。LoRA 的每个冻结参数 2 字节（BF16）。每个 LoRA 参数 16 字节，分别是 FP32 权重、FP32 梯度，以及 Adam 的 m 和 v，因为 ${DEF('PEFT',TIP.peft())} 默认把 B 和 A 从 BF16 升到 FP32（<code>autocast_adapter_dtype=True</code>，${cite(SRC.peftMap,'mapping_func.py')}）。权重本来就是 FP32，不用再存一份 FP32 副本。</li>`+
 `<li>QLoRA 把每层 7 个矩阵用 NF4 加双重量化存起来。每个权重 4 位，每 64 个权重一个 8 位的量化常数，每 256 个量化常数再共用一个 FP32 常数，平均每个参数 ${fixd(NF4,4)} 字节（不开双重量化是 ${fixd(NF4_NODQ,4)}；${DEF('bitsandbytes',TIP.bnb())} 的 ${cite(SRC.bnbF,'functional.py')}、${cite(SRC.bnbOps,'ops.py')}）。embedding 表、LM head 和 RMSNorm 的参数还是 BF16：transformers 只替换 <code>nn.Linear</code>，并且把输出层排除在外（${cite(SRC.tfBnb,'bitsandbytes.py')}、${cite(SRC.tfBase,'base.py')}）。矩阵乘法在 BF16 里做，用的是从 4 位还原出来的权重（${cite(SRC.bnbM,'modules.py')}）。</li>`+
 `<li>激活和 logits 的算法和${labLinkZ('training-memory')}完全一样。LoRA 的激活按和全量微调一样多算，因为反向还是要穿过每一个冻结的层。实际会有一点出入。冻结、又没加 LoRA 的矩阵，不用为求梯度留下输入，能少存一些。PEFT 会把每个加了 LoRA 的矩阵的输入转成 B 和 A 的精度，可能多存一份。这两点都不改变结论：LoRA 省不了激活，长序列还是要靠重算。峰值在前向结束的那一刻；重算时，反向过程中会有一层的激活短暂回来，但比之前释放的 logits 小。</li></ul></div>`+
 `<div><div class="eyebrow">没有算的</div><ul>`+
 `<li>CUDA 上下文、显存分配器缓存的部分和临时缓冲区。</li>`+
 `<li>QLoRA 正在用的那个矩阵的临时 BF16 副本，最大是 70B 的 ${GB(2*MOD['70b'].d*MOD['70b'].f)}。</li>`+
 `<li>LoRA 自己的少量激活：每个 token 在每个加了 LoRA 的矩阵上留下 r 个数值（x·A）。秩 16、4,096 个 token 时不到 ${fixd(Math.ceil(XA.base/1e8)/10,1)} GB。秩和序列都拉到最大时，70B 的这部分按 BF16 算也有约 ${fint(Math.round(XA.max/1e9))} GB，这时图里的数偏小。</li>`+
 `<li>双重量化每个矩阵自带的常数，加起来不到 1 MB。PEFT 默认的 dropout 是 0，没有 dropout 的 mask。</li>`+
 `<li>每个 token 的计算（全量微调 6N、重算时 8N，LoRA 约 4N、重算时约 6N）没有算注意力自己的计算、LoRA 矩阵自己的计算（每个 token 6 × LoRA 参数量，这里不到 1%），也没有算把 4 位权重还原成 BF16 的时间。</li>`+
 `<li>效果：这一页只算参数和显存。第 ${pn('remember')} 部分引用了 LoRA 和 QLoRA 的 README 报告的效果，并写明了它们测的是哪些模型。某个秩、某组矩阵或 4 位权重在你的任务上效果怎样，要自己评测。</li>`+
 `<li>GPU 的“80 GB”按 80 × 10⁹ 字节算（24 GB、48 GB 同理）。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>PEFT 的 ${cite(SRC.peftCfg,'config.py')}（r = 8，lora_alpha = 8，B 初始化为 0）、${cite(SRC.peftConst,'constants.py')}（Llama 默认只改 q_proj 和 v_proj）、${cite(SRC.peftLayer,'layer.py')}、${cite(SRC.peftMap,'mapping_func.py')}。</li>`+
 `<li>bitsandbytes 的 ${cite(SRC.bnbF,'functional.py')}、${cite(SRC.bnbOps,'ops.py')}、${cite(SRC.bnbM,'modules.py')}；transformers 的 ${cite(SRC.tfBnb,'bitsandbytes.py')}；${cite(SRC.qlora,'QLoRA README')}；${cite(SRC.lora,'LoRA README')}；torchtune 的 ${cite(SRC.tt,'README')}，以及它的 ${cite(SRC.ttLora,'LoRA')} 和 ${cite(SRC.ttQlora,'QLoRA')} 配置。</li>`+
 `<li>下面两篇论文只引了标题。LoRA: Low-Rank Adaptation of Large Language Models（arXiv 2106.09685）。QLoRA: Efficient Finetuning of Quantized LLMs（arXiv 2305.14314）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('count the parameters and the GPU memory of a LoRA or QLoRA fine-tune from a config, and say which part each setting shrinks.','从 config 算出 LoRA 或 QLoRA 微调要训练的参数和显存，并说清每个设置压的是哪一块。');
const EXITS=()=>{
  const b=compute(S8({})), m70=MOD['70b'], m405=MOD['405b'];
  const l70=loraOf(m70,16,'all'), l405=loraOf(m405,16,'all');
  const sh405=shapes(m405), lin405=m405.L*MATS.reduce((a,k)=>a+sh405[k][0]*sh405[k][1],0), w405=lin405*NF4+B_BF16*(m405.P-lin405);
  const q70=compute(S6('qlora')), w70=val(q70,'q4')+val(q70,'fb'), w70r=fint(Math.round(w70/1e10)*10);
  return {
 explain:[
  {q:tx('Full fine-tuning of the 8B needed about 169 GB; LoRA about 41 GB. Where did the saving come from?','全量微调 8B 要约 169 GB，LoRA 只要约 41 GB。省下的显存是从哪里来的？'),
   o:[tx('The frozen weights no longer need gradients or optimizer state: 16 of every parameter’s 18 bytes go away, and only B and A carry them.','冻结的权重不再需要梯度和优化器状态：每个参数 18 字节里的 16 字节没有了，只有 B 和 A 还要带着它们。'),
      tx('LoRA replaces the 8B’s weights with B and A, so the weights shrink from 16 GB to well under 1 GB.','LoRA 用 B 和 A 换掉了 8B 的权重，所以权重从 16 GB 缩到了不到 1 GB。'),
      tx('LoRA computes gradients only for B and A, so the forward pass keeps far fewer activations.','LoRA 只给 B 和 A 算梯度，所以前向留下的激活少得多。')],a:0,
   n:[null,tx('The frozen weights are all still in memory, 2 bytes each in BF16: the forward pass uses every one of them.','冻结的权重全都还在显存里，BF16 下每个 2 字节：前向的每一层都要用它们。'),
     tx('The backward pass still runs through every layer to reach B and A, so the activations hardly shrink.','反向还是要穿过每一层才能到 B 和 A，激活几乎没有变少。')],
   why:tx('Full fine-tuning keeps for every parameter a BF16 weight (2 bytes), an FP32 gradient, an FP32 copy and Adam’s m and v (16 bytes). LoRA freezes the weights: they keep their 2 bytes, and only the LoRA parameters carry the other 16. The activations stay about the same.','全量微调给每个参数存 BF16 权重（2 字节），以及 FP32 梯度、FP32 权重副本和 Adam 的 m、v（16 字节）。LoRA 冻住了权重：它们只剩 2 字节，另外那 16 字节只有 LoRA 的参数才有。激活几乎不变。')},
  {q:tx('How does QLoRA use its 4-bit weights in the forward pass?','QLoRA 在前向时怎么用它的 4 位权重？'),
   o:[tx('Each matrix is stored in 4 bits and turned back into BF16 right before it is used; the multiplication runs in BF16.','每个矩阵存成 4 位，用到它之前先还原成 BF16，乘法在 BF16 里做。'),
      tx('The multiplication runs directly on the 4-bit numbers, which is why QLoRA also trains faster.','乘法直接在 4 位的数上做，所以 QLoRA 训练得也更快。'),
      tx('Only B and A are stored in 4 bits; the frozen weights stay in BF16.','存成 4 位的只是 B 和 A，冻结的权重还是 BF16。')],a:0,
   n:[null,tx('The 4 bits are only how the weights are stored. They are turned back into BF16 before the multiplication, so QLoRA saves memory, not compute.','4 位只是权重的存法。做乘法前，它们会先还原成 BF16，所以 QLoRA 省显存，不省计算。'),
     tx('It is the other way round: the frozen linear weights go into 4 bits, and B and A are trained in higher precision.','正好反过来：存成 4 位的是冻结的线性层权重，B 和 A 用更高的精度训练。')],
   why:tx('QLoRA separates the storage type from the compute type: the frozen weights are stored in NF4, about 0.52 bytes each with their quantization constants, and each matrix is turned back into BF16 for its multiplication.','QLoRA 把存储用的格式和计算用的格式分开。冻结的权重用 NF4 存，连同量化常数每个约 0.52 字节。每个矩阵做乘法前，先还原成 BF16。')}],
 transfer:[
  {q:tx(`On this page, a LoRA of rank 16 on all 7 matrices gave ${m70.name} ${big(l70)} trained parameters. ${m405.name} has about 1.6 times as many layers, and its matrices’ sides are about twice as long. With the same LoRA, about how many parameters does it train?`,
        `这一页里，秩 16 的 LoRA 加在全部 7 个矩阵上，${m70.name} 要训练 ${big(l70)} 个参数。${m405.name} 的层数约是它的 1.6 倍，矩阵的边长约是它的 2 倍。同样的 LoRA，405B 大约要训练多少参数？`),
   o:[tx('About 400M (about 2 times)','约 400M（约 2 倍）'),tx('About 600M (about 3 times)','约 600M（约 3 倍）'),tx('About 1.3B (about 6 times)','约 1.3B（约 6 倍）')],a:1,
   why:tx(`B and A grow with the sides of a matrix, not with its area: twice the sides give about twice the LoRA parameters per layer, and 1.6 times the layers give about 3 times in all, ${big(l405)} (${pctS(l405/m405.P)} of the 405B’s ${big(m405.P)}). The weights themselves grow with the area, about 4 times per matrix.`,
          `B 和 A 跟矩阵的边长成正比，不跟面积成正比：边长约 2 倍，每层的 LoRA 参数就约 2 倍；层数再乘 1.6，一共约 3 倍，也就是 ${big(l405)} 个（占 405B 全部 ${big(m405.P)} 个的 ${pctS(l405/m405.P)}）。权重本身跟面积成正比，每个矩阵约 4 倍。`)},
  {q:tx(`On this page, QLoRA brought ${m70.name}’s frozen weights down to about ${w70r} GB: ${GB(val(q70,'q4'))} of 4-bit weights plus ${GB(val(q70,'fb'))} for the BF16 embedding table and LM head. ${m405.name} has about 5.8 times as many parameters as the 70B. Can one 80 GB H100 fine-tune it with QLoRA?`,
        `这一页里，QLoRA 把 ${m70.name} 冻结的权重压到了约 ${w70r} GB：4 位权重 ${GB(val(q70,'q4'))}，加上 BF16 的 embedding 表和 LM head ${GB(val(q70,'fb'))}。${m405.name} 的参数约是 70B 的 5.8 倍。一张 80 GB 的 H100 能用 QLoRA 微调它吗？`),
   o:[tx('No: the frozen weights alone take over 200 GB','不能：光冻结的权重就超过 200 GB'),tx('Yes, if every layer is recomputed','能，只要每一层都重算'),tx('Yes, with a small rank such as 4','能，只要秩取得够小，比如 4')],a:0,
   why:tx(`The frozen weights grow with the parameters: about 5.8 × ${w70r} GB is over 200 GB (exactly ${GB(w405)}), before any LoRA parameter or activation. Recomputation shrinks only the activations and a small rank only B and A, so neither helps. torchtune’s README lists QLoRA of the 405B on 8 A100s.`,
          `冻结的权重和参数量成正比：约 5.8 × ${w70r} GB，超过 200 GB（准确地说是 ${GB(w405)}），这还没算 LoRA 参数和激活。重算只压激活，小的秩只压 B 和 A，都帮不上。torchtune 的 README 里，405B 的 QLoRA 用了 8 张 A100。`)}],
 practice:[
  {id:'stats-tests',d:(()=>{const xab=A.ui.math('x(AB)'), xa=A.ui.math('(xA)B'), ba=A.ui.math('BA'); return tx(`Q1 and Q2: why a low-rank update ${xab} should be computed as ${xa}, and why a LoRA update ${ba} has rank at most r.`,`第 1、2 题：低秩更新 ${xab} 为什么要按 ${xa} 的顺序算，以及 LoRA 的更新 ${ba} 为什么秩最多是 r。`);})()},
  {id:'distributed-training',d:tx('The memory plan: count the bytes per parameter of a 70B model trained with Adam in mixed precision, the case this lab starts from, before LoRA removes most of them.','显存方案：数一数 70B 模型用 Adam 混合精度训练时每个参数占多少字节。这正是这个 Lab 的起点，LoRA 去掉的就是其中大部分。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'LoRA',title:l&&l.q?L(l.q):tx('How does LoRA fine-tune an 8B model on one GPU?','LoRA 怎么在一张卡上微调 8B 模型？'),
    minutes:12,fid:'Analytical',fidPop:FIDPOP(),descs:DESCS(),mode,
    primer:tx('Primer: training memory','前置知识：训练显存'),primerPop:tx(`<p>This lab builds on three other labs:</p><ul><li>The 18 bytes per parameter, the activations and recomputation: ${labLink('training-memory')}</li><li>Adam’s m and v: ${labLink('optimizers')}</li><li>BF16 and 4-bit formats: ${labLink('number-formats')}</li></ul>`,
      `<p>这个 Lab 接着三个 Lab 往下讲：</p><ul><li>每个参数 18 字节、激活和重算：${labLink('training-memory')}</li><li>Adam 的 m 和 v：${labLink('optimizers')}</li><li>BF16 和 4 位格式：${labLink('number-formats')}</li></ul>`)});
}
function stageHTML(){
  return `<div class="panel stage lora-stage" id="lora-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="lora-st-title"></div><div class="meter-sub" id="lora-st-sub"></div></div><div class="stage-actions"><span class="status" id="lora-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph lora-graph" id="lora-graph"><div class="lora-vis" id="lora-vis">`+
    `<div class="lora-memp"><div class="lora-cap" id="lora-mem-cap"></div><div class="lora-mem" id="lora-mem"></div><div class="lora-leg" id="lora-mem-leg"></div></div>`+
    `<div class="lora-matp" id="lora-matp"><div class="lora-cap" id="lora-mat-cap"></div><div id="lora-mat-segw">${matSeg('lora-st-mat',false)}</div><div class="lora-mat" id="lora-mat"></div><p class="lora-matnote" id="lora-mat-note"></p></div>`+
   `</div></div>`+
   `<div class="metrics lora-metrics" id="lora-metrics"></div>`+
   `<details class="math" id="lora-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="lora-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="lora-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="lora-lock" hidden></p>`+
   `<div class="lockable" id="lora-ctl-m-wrap">${A.ui.seg({id:'lora-ctl-m',key:'m',label:tx('Model','模型'),options:[['8b',MOD['8b'].name],['70b',MOD['70b'].name]],value:S.m})}</div>`+
   `<div class="lockable" id="lora-ctl-gpu-wrap">${A.ui.seg({id:'lora-ctl-gpu',key:'gpu',label:tx('GPU','GPU'),options:GPUK.map(k=>[k,k==='g80'?'80 GB H100':k==='g48'?'48 GB':'24 GB']),value:S.gpu,note:tx('A GPU’s “80 GB” is read as 80 × 10⁹ bytes, and the 24 and 48 GB cards likewise.','GPU 的“80 GB”按 80 × 10⁹ 字节算，24 GB 和 48 GB 的卡同理。')})}</div>`+
   `<div class="lockable" id="lora-ctl-meth-wrap">${A.ui.seg({id:'lora-ctl-meth',key:'meth',label:tx('Method','方法'),options:METHS.map(k=>[k,methName(k)]),value:S.meth})}</div>`+
   `<div class="lockable" id="lora-ctl-r-wrap">${rCtl('lora-ctl-r',false)}</div>`+
   `<div class="lockable" id="lora-ctl-tg-wrap">${tgCtl('lora-ctl-tg',false)}</div>`+
   `<div class="lockable" id="lora-ctl-seq-wrap">${seqCtl('lora-ctl-seq',false)}</div>`+
   `<div class="lockable" id="lora-ctl-rc-wrap">${rcCtl('lora-ctl-rc',false)}</div>`+
   `<div class="lockable" id="lora-ctl-note"><details class="lora-asm" id="lora-asm"><summary>${tx('Estimation assumptions','估算假设')}</summary><p class="ctl-note" id="lora-asm-lr">${tx(`Each LoRA parameter is counted as 16 bytes: ${DEF('PEFT',TIP.peft())} (Hugging Face’s fine-tuning library) keeps B and A in FP32 by default, plus an FP32 gradient and Adam’s m and v. If B and A were kept in BF16, the weight and the gradient would take 2 bytes each. LoRA parameters are so few that the total changes little.`,`每个 LoRA 参数按 16 字节算：${DEF('PEFT',TIP.peft())}（Hugging Face 的微调库）默认把 B 和 A 存成 FP32，再加 FP32 梯度和 Adam 的 m、v。如果 B 和 A 改用 BF16 存，权重和梯度各只要 2 字节。LoRA 参数很少，总数变化不大。`)}</p>`+
     `<p class="ctl-note" id="lora-asm-q">${tx(`QLoRA’s ${fixd(NF4,3)} bytes per frozen linear parameter: the 4-bit weight is 0.5 bytes, and the rest is its share of the quantization constants. ${DEF('bitsandbytes',TIP.bnb())}, the quantization library QLoRA uses, keeps one constant per block of 64 weights and compresses them by double quantization.`,`QLoRA 每个冻结的线性层参数按 ${fixd(NF4,3)} 字节算：4 位的权重是 0.5 字节，多出来的是它分摊的量化常数。QLoRA 用的量化库是 ${DEF('bitsandbytes',TIP.bnb())}，它每 64 个权重存一个量化常数，再用双重量化把这些常数压小。`)}</p>`+
     `<p class="ctl-note" id="lora-asm-lg">${tx(`Logits are counted as one FP32 copy (4 bytes each), as in ${labLink('training-memory')}: an assumption, since frameworks differ.`,`logits 按一份 FP32（每个 4 字节）算，和${labLinkZ('training-memory')}一样。这是假设，各个框架不完全一样。`)}</p></details></div>`+
   `<div class="lockable" id="lora-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    """h = W x + (alpha / r) B A x, with W frozen."""
    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)       # frozen W
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.empty(r, d_in))   # FP32
        self.B = nn.Parameter(torch.zeros(d_out, r))  # B = 0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        h = (x.to(self.A.dtype) @ self.A.T) @ self.B.T  # (xA)B
        return self.base(x) + (h * self.scale).to(x.dtype)

    @torch.no_grad()
    def merge(self):          # fold B A into W after training
        dw = (self.B @ self.A) * self.scale
        self.base.weight += dw.to(self.base.weight.dtype)

def lora_params(mats, r):     # mats: (d_in, d_out) per matrix
    return sum(r * (i + o) for i, o in mats)

d, kv, f = 4096, 1024, 14336            # Llama-3.1-8B
layer = [(d, d), (d, kv), (d, kv), (d, d),  # q k v o
         (d, f), (d, f), (f, d)]            # gate up down
n = 32 * lora_params(layer, 16)
print(n, n * 16 / 1e9)        # 41943040 0.67108864 (GB)`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="lora-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="lora-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A LoRA layer around a frozen nn.Linear, its merge into W, and the parameter count for the 8B.','把一个冻结的 nn.Linear 包成 LoRA 层、训练后合并进 W，以及 8B 的 LoRA 参数量。')}</p><pre class="code" id="lora-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="lora-prod-card"><h3>${A.t('cProd')}</h3><ul class="lora-prod">`+
   tx(`<li>${DEF('PEFT',TIP.peft())}, Hugging Face’s parameter-efficient fine-tuning library: its <code>LoraConfig</code> defaults to r = 8, lora_alpha = 8 and lora_dropout = 0 (${cite(SRC.peftCfg,'config.py')}); for Llama, with no target_modules given, it adapts <code>q_proj</code> and <code>v_proj</code> (${cite(SRC.peftConst,'constants.py')}). The forward line is <code>result = result + lora_B(lora_A(dropout(x))) * scaling</code>, with <code>scaling = lora_alpha / r</code>, or <code>lora_alpha / math.sqrt(r)</code> with <code>use_rslora</code> (${cite(SRC.peftLayer,'layer.py')}). B is initialised to zeros, so “without further training, the LoRA adapter will be a no-op” (${cite(SRC.peftCfg,'config.py')}).</li>`+
      `<li>QLoRA in transformers: <code>BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)</code>, as the QLoRA README shows (${cite(SRC.qlora,'README')}). Its 65B script trains r = 64, lora_alpha = 16 on all linear layers, with gradient checkpointing (${cite(SRC.qlora65,'finetune_guanaco_65b.sh')}).</li>`+
      `<li>${DEF('torchtune',TIP.tt())}, PyTorch’s own fine-tuning library: its single-GPU configs for Llama 3.1 8B use rank 8 and alpha 16 on q, v, the output projection and the MLP, with activation checkpointing (${cite(SRC.ttLora,'8B_lora_single_device.yaml')}, ${cite(SRC.ttQlora,'8B_qlora_single_device.yaml')}). Its README reports a peak of 16.2 ${GIB()} for LoRA and 7.4 ${GIB()} for QLoRA on one RTX 4090, 2 sequences of 2,048 tokens per step (${cite(SRC.tt,'README')}).</li>`+
      `<li>Full fine-tuning can fit one GPU too, by giving something up. torchtune’s README lists the 8B fully fine-tuned on one RTX 4090 at a peak of 18.9 ${GIB()}; its config keeps only BF16 weights, uses bitsandbytes’ 8-bit paged AdamW and updates the weights inside the backward pass (${cite(SRC.tt,'README')}, ${cite(SRC.ttFull,'8B_full_single_device.yaml')}). LoRA gives up nothing on the optimizer, and what it saves after training is only B and A: the LoRA README says this “vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment” (${cite(SRC.lora,'README')}).</li>`+
      `<li>Serving: B·A can be merged into W after training; the LoRA README says this works “without introducing inference latency” (${cite(SRC.lora,'README')}).</li>`,
      `<li>${DEF('PEFT',TIP.peft())} 是 Hugging Face 的参数高效微调库，它的 <code>LoraConfig</code> 默认 r = 8、lora_alpha = 8、lora_dropout = 0（${cite(SRC.peftCfg,'config.py')}）。对 Llama，不指定 target_modules 时只改 <code>q_proj</code> 和 <code>v_proj</code>（${cite(SRC.peftConst,'constants.py')}）。前向那一行是 <code>result = result + lora_B(lora_A(dropout(x))) * scaling</code>，其中 <code>scaling = lora_alpha / r</code>。打开 <code>use_rslora</code> 时，scaling 换成 <code>lora_alpha / math.sqrt(r)</code>（${cite(SRC.peftLayer,'layer.py')}）。B 初始化为 0，所以“without further training, the LoRA adapter will be a no-op”（不训练，B 和 A 就什么都不改，${cite(SRC.peftCfg,'config.py')}）。</li>`+
      `<li>在 transformers 里用 QLoRA：<code>BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)</code>，和 QLoRA 的 README 写的一样（${cite(SRC.qlora,'README')}）。它训练 65B 的脚本用 r = 64、lora_alpha = 16，加在所有线性层上，并打开 gradient checkpointing（${cite(SRC.qlora65,'finetune_guanaco_65b.sh')}）。</li>`+
      `<li>${DEF('torchtune',TIP.tt())} 是 PyTorch 官方的微调库，它给 Llama 3.1 8B 的单卡配置用秩 8、alpha 16，加在 q、v、输出投影和 MLP 上，并打开 activation checkpointing（${cite(SRC.ttLora,'8B_lora_single_device.yaml')}、${cite(SRC.ttQlora,'8B_qlora_single_device.yaml')}）。它的 README 里，一张 RTX 4090 上的峰值：LoRA 16.2 ${GIB()}，QLoRA 7.4 ${GIB()}，每步 2 条 2,048 个 token 的序列（${cite(SRC.tt,'README')}）。</li>`+
      `<li>全量微调也能挤进一张卡，但要在别处让步。torchtune 的 README 里，8B 全量微调在一张 RTX 4090 上的峰值是 18.9 ${GIB()}。它的配置只存 BF16 权重，用 bitsandbytes 的 8 位分页 AdamW，并在反向里直接更新权重（${cite(SRC.tt,'README')}、${cite(SRC.ttFull,'8B_full_single_device.yaml')}）。LoRA 不用在优化器上让步，训练完要存的也只有 B 和 A。LoRA 的 README 说，这样“vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment”（大大减少为各个任务适配的大模型要占的存储，部署时也能高效地切换任务，${cite(SRC.lora,'README')}）。</li>`+
      `<li>上线前，可以把训练好的 B·A 合并进 W。LoRA 的 README 说，这样做“without introducing inference latency”（不增加推理延迟，${cite(SRC.lora,'README')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.paLora,'LoRA: Low-Rank Adaptation of Large Language Models')}<span>${tx('Hu et al., 2021 (arXiv 2106.09685): freeze the weights, train a low-rank B·A beside each matrix.','Hu 等，2021（arXiv 2106.09685）：冻住权重，在每个矩阵旁边训练一个低秩的 B·A。')}</span></li>`+
   `<li>${cite(SRC.paQlora,'QLoRA: Efficient Finetuning of Quantized LLMs')}<span>${tx('Dettmers et al., 2023 (arXiv 2305.14314): NF4, double quantization and paged optimizers, LoRA on a 4-bit base model.','Dettmers 等，2023（arXiv 2305.14314）：NF4、双重量化和分页优化器，在 4 位的基础模型上做 LoRA。')}</span></li>`+
   `<li>${cite(SRC.paRs,'A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA')}<span>${tx(`Kalajdzievski, 2023 (arXiv 2312.03732): the ${A.ui.math('\\frac{α}{\\sqrt{r}}')} scaling behind PEFT’s use_rslora.`,`Kalajdzievski，2023（arXiv 2312.03732）：PEFT 的 use_rslora 用的 ${A.ui.math('\\frac{α}{\\sqrt{r}}')} 缩放。`)}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="lora-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="lora-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="lora-app m2" id="lora-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="lora-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see where one step’s memory goes.','选一个场景，看看一步的显存都花在哪里。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Pick a scenario','选一个应用场景'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="lora-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Fine-tune the 70B on one 48 GB GPU, with at least 100M trained parameters`,'在一张 48 GB 的卡上微调 70B，要训练的参数至少 100M 个')}</h3>`+
   `<p>${tx(`Your team has one GPU with 48 GB and wants to fine-tune ${MOD['70b'].name} on sequences of 2,048 tokens. The plan must fit, and it must train at least 100M parameters. Among the plans that do, the one with the least memory wins.`,
            `你们只有一张 48 GB 的卡，要微调 ${MOD['70b'].name}，每条序列 2,048 个 token。方案要放得下，要训练的参数至少 100M 个。达标的方案里，显存越少越好。`)}</p>`+
   A.ui.seg({id:'lora-ic-chm',key:'chm',label:tx('Method','方法'),options:METHS.map(k=>[k,methName(k)]),value:S.meth,inline:true})+
   `<div id="lora-ch-lw">`+rCtl('lora-ic-chr',true)+A.ui.seg({id:'lora-ic-cht',key:'cht',label:tx('Which matrices get a LoRA','LoRA 加在哪些矩阵上'),options:TGTS.map(k=>[k,tgShort(k)]),value:S.tg,inline:true})+`</div>`+
   A.ui.seg({id:'lora-ic-chc',key:'chc',label:tx('Recomputation','重算'),options:[['0',tx('Off','不重算')],['1',tx('Every layer','每一层都重算')]],value:String(S.rc),inline:true})+
   `<div class="goal mono lora-goal" id="lora-ch-g1"><span>${tx('GPU memory','显存')}</span><span><b id="lora-ch-p"></b>${tx(' (goal: at most 48 GB)','（目标：不超过 48 GB）')}</span></div>`+
   `<div class="goal mono lora-goal" id="lora-ch-g2"><span>${tx('Trained parameters','要训练的参数')}</span><span><b id="lora-ch-t"></b>${tx(' (goal: at least 100M)','（目标：至少 100M 个）')}</span></div>`+
   submit+`<div id="lora-ch-res" class="lora-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},lowrank:{en:'Pick another matrix to continue.',zh:'换一个矩阵看看，才能继续。'},
    memory:{en:'Pick LoRA to continue.',zh:'选 LoRA 才能继续。'},rank:{en:'Change the rank or the matrices to continue.',zh:'改一下秩或加在哪些矩阵上，才能继续。'},
    long:{en:'Make the sequence 16,384 tokens or longer and turn on recomputation to continue.',zh:'把序列拉到 16,384 个 token 或更长，再打开重算，才能继续。'},qlora:{en:'Pick QLoRA to continue.',zh:'选 QLoRA 才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#lora-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#lora-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    $$('[data-lever]',rail).forEach(b=>{const [k,x]=b.dataset.lever.split(':'); let on=false, res='';
      if(k==='l3'){on=x===S.meth; res=l3Tried.has(x)||x==='full'?lv3Res(x):'';}
      else if(k==='l6'){on=x===l6Sel; res=l6Tried.has(x)?lv6Res(x):'';}
      b.setAttribute('aria-pressed',String(on)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=res;});
    if(box){const html=s.obs?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    $$('[data-nav="showme"]',rail).forEach(sm=>{sm.hidden=!!observed[step];});
    $$('.lora-remb',rail).forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.rem===remView)); b.textContent=remBtn(b.dataset.rem);});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#lora-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.meth,S.r,S.tg,S.rc), p=$('#lora-ch-p',rail), t=$('#lora-ch-t',rail); if(!p) return;
    p.textContent=GB(e.tot); t.textContent=big(e.trained);
    const g1=$('#lora-ch-g1',rail), g2=$('#lora-ch-g2',rail); g1.classList.toggle('met',e.fits); g1.classList.toggle('lora-miss',!e.fits); g2.classList.toggle('met',e.enough); g2.classList.toggle('lora-miss',!e.enough);
    $$('.seg-ctl[data-key="chm"]',rail).forEach(g=>A.ui.setPressed(g,S.meth)); $$('.seg-ctl[data-key="cht"]',rail).forEach(g=>A.ui.setPressed(g,S.tg)); $$('.seg-ctl[data-key="chc"]',rail).forEach(g=>A.ui.setPressed(g,String(S.rc)));
    const lw=$('#lora-ch-lw',rail); if(lw) lw.hidden=S.meth==='full';
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#lora-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
/* parts 1-3: the memory chart is still the full fine-tuning baseline; the matrix panel and the trained-parameter card show the LoRA.
   Part 1 draws only the idea (the two paths); the share of trained parameters stays hidden until the prediction in part 3 (R3.8). */
const loraView=()=>mode==='guided'&&(isP('mission')||isP('lowrank')||isP('predict'));
const ideaView=()=>mode==='guided'&&isP('mission');
const hidden0=()=>loraView()&&!pred;
/* which methods the memory chart shows: one new thing at a time (R3.13) */
function colsShown(){
  if(mode!=='guided') return METHS;
  const id=STEPS[step].id;
  if(id==='mission'||id==='lowrank'||id==='predict') return ['full'];
  if(id==='memory') return l3Tried.has('lora')?['full','lora']:['full'];
  if(id==='qlora') return l6Tried.has('qlora')||l6Tried.has('qnorc')?METHS:['full','lora'];
  if(id==='remember') return METHS;
  return ['full','lora'];
}
const BLK={w:{en:'BF16 weights',zh:'BF16 权重'},g:{en:'FP32 gradients',zh:'FP32 梯度'},c:{en:'FP32 copy',zh:'FP32 权重副本'},mv:{en:'Adam’s m and v',zh:'Adam 的 m 和 v'},
  fw:{en:'frozen BF16 weights',zh:'冻结的 BF16 权重'},q4:{en:'4-bit weights',zh:'4 位权重'},fb:{en:'BF16 embedding + LM head',zh:'BF16 的 embedding 和 LM head'},
  lr:{en:'LoRA parameters (with gradients and optimizer state)',zh:'LoRA 参数（含梯度和优化器状态）'},lrS:{en:'LoRA parameters',zh:'LoRA 参数'},a:{en:'activations',zh:'激活'},l:{en:'logits',zh:'logits'}};
/* the shorter names beside the column */
const BLKS={lr:BLK.lrS,fb:{en:'embedding + LM head',zh:'embedding、LM head'},fw:{en:'frozen weights',zh:'冻结的权重'}};
const blkTip=(k,c,v)=>({w:tx(`BF16 weights: ${GB(v)}, 2 bytes for each of the ${big(c.m.P)} parameters.`,`BF16 权重：${GB(v)}，${big(c.m.P)} 个参数，每个 2 字节。`),
  g:tx(`FP32 gradients: ${GB(v)}, 4 bytes per parameter.`,`FP32 梯度：${GB(v)}，每个参数 4 字节。`),
  c:tx(`FP32 copy of the weights: ${GB(v)}, 4 bytes per parameter.`,`FP32 权重副本：${GB(v)}，每个参数 4 字节。`),
  mv:tx(`Adam’s m and v: ${GB(v)}, 4 bytes each per parameter.`,`Adam 的 m 和 v：${GB(v)}，每个参数各 4 字节。`),
  fw:tx(`Frozen BF16 weights: ${GB(v)}, 2 bytes per parameter; no gradient, no optimizer state.`,`冻结的 BF16 权重：${GB(v)}，每个参数 2 字节，没有梯度，也没有优化器状态。`),
  q4:tx(`4-bit weights of the 7 matrices of every layer: ${GB(v)}, ${big(c.lin)} parameters at ${fixd(NF4,3)} bytes each (NF4 with its quantization constants).`,`每层 7 个矩阵的 4 位权重：${GB(v)}，${big(c.lin)} 个参数，每个 ${fixd(NF4,3)} 字节（NF4 加上量化常数）。`),
  fb:tx(`The embedding table, the LM head and the RMSNorm weights, kept in BF16: ${GB(v)}.`,`embedding 表、LM head 和 RMSNorm 的参数，留在 BF16：${GB(v)}。`),
  lr:tx(`LoRA parameters: ${GB(v)}, ${big(c.lp)} parameters at 16 bytes (FP32 weight, gradient, Adam’s m and v).`,`LoRA 参数：${GB(v)}，${big(c.lp)} 个参数，每个 16 字节（FP32 的权重、梯度、Adam 的 m 和 v）。`),
  a:tx(`Activations kept for the backward pass: ${GB(v)}${S.rc?', each layer’s input only (recomputation)':''}.`,`留给反向的激活：${GB(v)}${S.rc?'，重算时每层只留输入':''}。`),
  l:tx(`Logits in FP32: ${GB(v)}, ${fint(S.seq)} tokens × 128,256 × 4 bytes.`,`FP32 的 logits：${GB(v)}，${fint(S.seq)} 个 token × 128,256 × 4 字节。`)}[k]);
const colH=()=>innerWidth<=760?230:262;
function yMaxOf(cs,cap){const mx=Math.max(...cs.map(c=>c.tot)); return Math.min(cap*2.2,Math.max(cap*1.2,mx*1.08));}
function drawMem(){
  const host=$('#lora-mem',app); if(!host) return;
  const shown=colsShown(), cur=shown.includes(S.meth)?S.meth:shown[0], cs={}; shown.forEach(me=>{cs[me]=compute({...S,meth:me});});
  const others=shown.filter(me=>me!==cur), mainOn=shown.includes(cur), LH=13;
  const c=compute(S), W=Math.max(270,Math.min(360,host.clientWidth||300)), H=colH()+LH*others.length, top=16, bot=colH()-22, cap=c.cap, ymax=yMaxOf(Object.values(cs),cap), Y=v=>bot-Math.min(1,v/ymax)*(bot-top);
  const sw=20, sg=10, mx=8+others.length*(sw+sg)+(others.length?8:0), cw=46, lx=mx+cw+8;
  let svg=`<svg class="lora-memsvg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(memAria(cs,c))}">`+
    `<defs><pattern id="lora-p-g" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="lora-pbg"/><line x1="0" y1="0" x2="0" y2="6" class="lora-pln"/></pattern>`+
    `<pattern id="lora-p-m" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" class="lora-pbg"/><circle cx="2.5" cy="2.5" r="1.1" class="lora-pdot"/></pattern>`+
    `<pattern id="lora-p-q" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" class="lora-pq"/><line x1="0" y1="0" x2="0" y2="5" class="lora-pqln"/></pattern>`+
    `<pattern id="lora-p-l" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="4" class="lora-pbg"/><line x1="1" y1="0" x2="1" y2="4" class="lora-pln"/></pattern>`+
    `<pattern id="lora-p-o" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(135)"><line x1="0" y1="0" x2="0" y2="6" class="lora-poln"/></pattern></defs>`;
  const capY=Y(cap);
  const col=(me,x,w,main,i)=>{const cc=cs[me]; let y=bot, out='';
    cc.blocks.forEach(([k,v])=>{if(v<=0) return; const h=Math.min(v,Math.max(0,ymax-(bot-y)/(bot-top)*ymax))/ymax*(bot-top); if(h<=0) return; const yy=y-h;
      out+=`<rect x="${x}" y="${yy.toFixed(2)}" width="${w}" height="${Math.max(0.6,h).toFixed(2)}" class="lora-b-${k}"${main?` data-tipx="${esc(blkTip(k,cc,v))}" tabindex="0"`:''}/>`; y=yy;});
    const clipped=cc.tot>ymax*1.0001;
    if(cc.tot>cap) out+=`<rect x="${x}" y="${Y(cc.tot).toFixed(2)}" width="${w}" height="${(capY-Y(cc.tot)).toFixed(2)}" class="lora-over"/>`;
    if(clipped) out+=`<path d="M${x-2} ${top+5} l${w/4+1} -5 l${w/4+1} 5 l${w/4+1} -5 l${w/4+1} 5" class="lora-brk"/>`;
    out+=`<rect x="${x}" y="${Y(cc.tot).toFixed(2)}" width="${w}" height="${(bot-Y(cc.tot)).toFixed(2)}" class="lora-colbox${main?' main':''}" data-tipx="${esc(colTip(me,cc))}" tabindex="${main?-1:0}"/>`;
    /* the main column's total sits above it; each narrow column's total gets a line of its own under the axis, starting at its left edge */
    let ty=main?Math.max(10,Y(cc.tot)-5):bot+14+LH*(i+1);
    if(main&&ty>capY-3&&ty<capY+11) ty=capY-3; /* never let the GPU line run through the main total */
    out+=`<text x="${main?x+w/2:x-1}" y="${ty.toFixed(1)}" class="lora-tot${main?' lora-mid':''}${cc.fits?'':' bad'} lora-halo"${main?'':` data-col="${me}"`}>${esc(main&&clipped?`${fint(Math.round(cc.tot/1e9))} GB ↑`:GB(cc.tot))}</text>`;
    out+=`<text x="${x+w/2}" y="${bot+14}" class="lora-ax lora-mid${main?' main':''}">${esc(me==='full'?tx('full','全量'):methName(me))}</text>`;
    return out;};
  others.forEach((me,i)=>{svg+=col(me,8+i*(sw+sg),sw,false,i);});
  if(mainOn) svg+=col(cur,mx,cw,true,0);
  /* labels beside the main column (R5.9), pushed apart */
  if(mainOn){const labs=[]; let y=bot;
    cs[cur].blocks.forEach(([k,v])=>{if(v<=0) return; const h=Math.min(v,Math.max(0,ymax-(bot-y)/(bot-top)*ymax))/ymax*(bot-top), yy=y-h; labs.push({y:Math.max(top,yy+h/2),t:`${L(BLKS[k]||BLK[k])} ${GB(v)}`}); y=yy;});
    /* the GPU's label stays on its line; block labels move away from it on either side, then apart from each other */
    const capL={y:capY-8,t:S.gpu==='g80'?'H100 80 GB':gpuShort(S.gpu),cap:true}, minG=12;
    const below=labs.filter(l=>l.y>capL.y).sort((a,b)=>a.y-b.y), above=labs.filter(l=>l.y<=capL.y).sort((a,b)=>b.y-a.y);
    let prev=capL.y; below.forEach(l=>{if(l.y-prev<minG) l.y=prev+minG; prev=l.y;});
    prev=capL.y; above.forEach(l=>{if(prev-l.y<minG) l.y=prev-minG; prev=l.y;});
    if(above.length&&above[above.length-1].y<top+4){let lim=top+4; for(let i=above.length-1;i>=0;i--){if(above[i].y<lim) above[i].y=lim; lim=above[i].y+minG;}}
    labs.push(capL);
    labs.forEach(l=>{svg+=`<text x="${lx}" y="${(l.y+4).toFixed(1)}" class="${l.cap?'lora-capl':'lora-bl'}">${esc(l.t)}</text>`;});}
  /* the GPU line across every column */
  const x1=4, x2=mx+cw+4;
  svg+=`<line x1="${x1}" x2="${x2}" y1="${capY.toFixed(2)}" y2="${capY.toFixed(2)}" class="lora-capline"/>`+(mainOn?'':`<text x="${x2+4}" y="${(capY-4).toFixed(1)}" class="lora-capl">${esc(S.gpu==='g80'?'H100 80 GB':gpuShort(S.gpu))}</text>`);
  if(mainOn&&!cs[cur].fits){const yy=Math.max(top+12,(Math.max(top,Y(cs[cur].tot))+capY)/2+4); svg+=`<text x="${mx+cw/2}" y="${yy.toFixed(1)}" class="lora-oomt lora-halo lora-mid">${esc(tx('no fit','放不下'))}</text>`;}
  svg+=`</svg>`;
  host.innerHTML=svg;
  /* H7 (the owner, 2026-10-09): a label longer than the room right of the main column widens the picture's coordinates, so the whole
     picture scales down a little instead of the label running past its right edge (900 px, English: “frozen weights 141.1 GB” by 5 px) */
  const sv=host.querySelector('svg'); let need=W;
  if(sv) sv.querySelectorAll('text').forEach(t=>{try{const bb=t.getBBox(); if(bb.width) need=Math.max(need,bb.x+bb.width+2);}catch(e){}});
  if(sv&&need>W+0.5) sv.setAttribute('viewBox',`0 0 ${need.toFixed(1)} ${H}`);
  $('#lora-mem-cap',app).textContent=tx(`GPU memory at the peak of one step (${c.m.name}, ${fint(S.seq)} tokens)`,`一步的峰值显存（${c.m.name}，${fint(S.seq)} 个 token）`);
  drawLegend(shown);
}
function colTip(me,c){return tx(`${methName(me)}: ${GB(c.tot)} in all; ${c.fits?`fits the ${GB(c.cap)}`:`more than the ${GB(c.cap)}`}.`,`${methName(me)}：一共 ${GB(c.tot)}；${c.fits?`放得下 ${GB(c.cap)}`:`超过了 ${GB(c.cap)}`}。`);}
function memAria(cs,c){return Object.keys(cs).map(me=>colTip(me,cs[me])).join(' ')+tx(` Current method: ${methName(S.meth)}.`,`当前方法：${methName(S.meth)}。`);}
function drawLegend(shown){
  const host=$('#lora-mem-leg',app), ks=[];
  const add=k=>{if(!ks.includes(k)) ks.push(k);};
  shown.forEach(me=>{if(me==='full'){['w','g','c','mv'].forEach(add);} else if(me==='lora'){['fw','lr'].forEach(add);} else ['q4','fb','lr'].forEach(add);}); ['a','l'].forEach(add);
  const sw=k=>`<i class="sw lora-sw-${k}" aria-hidden="true"></i>`;
  const tip=k=>({w:tx('Full fine-tuning: the BF16 weights, 2 bytes per parameter.','全量微调：BF16 权重，每个参数 2 字节。'),g:tx('Full fine-tuning: FP32 gradients, 4 bytes per parameter.','全量微调：FP32 梯度，每个参数 4 字节。'),c:tx('Full fine-tuning: the FP32 copy of the weights, 4 bytes per parameter.','全量微调：FP32 权重副本，每个参数 4 字节。'),mv:tx('Full fine-tuning: Adam’s m and v, 4 bytes each per parameter.','全量微调：Adam 的 m 和 v，每个参数各 4 字节。'),
    fw:TIP.frozen(),q4:TIP.nf4(),fb:tx('QLoRA keeps the embedding table, the LM head and the RMSNorm weights in BF16.','QLoRA 把 embedding 表、LM head 和 RMSNorm 的参数留在 BF16。'),lr:tx('B and A, 16 bytes per parameter: FP32 weight, gradient, Adam’s m and v.','B 和 A，每个参数 16 字节：FP32 的权重、梯度，以及 Adam 的 m 和 v。'),a:TIP.act(),l:tx('Logits: one FP32 score per vocabulary entry for every token, 4 bytes each.','logits：每个 token 对词表里每一项一个 FP32 得分，每个 4 字节。')}[k]);
  host.innerHTML=ks.map(k=>`<span data-tipx="${esc(tip(k))}">${sw(k)}${esc(L(BLK[k]))}</span>`).join('')+
    `<span data-tipx="${esc(tx('The dashed line is the GPU’s memory; the red hatching is the part of a column above it.','虚线是这张 GPU 的显存，红色斜纹是一栏里超出它的部分。'))}"><i class="sw lora-sw-cap" aria-hidden="true"></i>${esc(tx('GPU memory','GPU 显存'))}</span>`+
    `<span data-tipx="${esc(tx('The part of a column above the GPU’s memory.','一栏里超出 GPU 显存的部分。'))}"><i class="sw lora-sw-over" aria-hidden="true"></i>${esc(tx('does not fit','放不下'))}</span>`;
}
/* part 1: the idea only (review codex-b9v3 应修 2). The input x goes up two paths, through the frozen W and through A then B, and the two
   results are added into the output h: the layout of the LoRA paper's first figure (x at the bottom, W on the left, A narrowing to a
   waist and B widening again on the right). No shapes and no rank here: part 2 brings them. */
function drawIdea(){
  const host=$('#lora-mat',app), note=$('#lora-mat-note',app); if(!host) return;
  const W=246, H=210, xl=64, xr=184, cx=124, cy=42;
  const T=(x,y,t,cls,anchor)=>`<text x="${x}" y="${y}" class="lora-il${cls?' '+cls:''}"${anchor?` text-anchor="${anchor}"`:''}>${esc(t)}</text>`;
  const tipX=tx('The input x: the vector this layer feeds into the matrix.','输入 x：这一层送进这个矩阵的向量。');
  const tipH=tx('The output h: what W gives, plus what the path through A and B adds.','输出 h：W 算出的结果，加上 A、B 这条路的结果。');
  const tipW=tx('W: a matrix the model already has. It is frozen: the forward pass uses it, but training never changes it.','W：模型原有的矩阵。它被冻住，前向要用它，训练却从不改它。');
  const tipA=tx('A: a new small matrix, trained. It squeezes the input into a short list of numbers.','A：新加的小矩阵，要训练。它把输入压成很短的一组数。');
  const tipB=tx('B: a new small matrix, trained. It expands that list to as many numbers as W x has.','B：新加的小矩阵，要训练。它把这组数展开成和 W x 一样长。');
  const tipP=tx('The two results are added here.','两条路的结果在这里相加。');
  const tip=t=>` data-tipx="${esc(t)}" tabindex="0"`;
  let s=`<svg class="lora-ideasvg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx('The input x goes through the frozen W, and beside it through A and then B; the two results are added into the output h. Only A and B are trained.','输入 x 走两条路，一条经过冻结的 W，另一条先经过 A、再经过 B。两条路的结果相加，得到输出 h。只训练 A 和 B。'))}">`+
    `<defs><marker id="lora-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8 z" class="lora-arrh"/></marker></defs>`;
  /* output h (top) and input x (bottom) */
  s+=`<rect x="34" y="4" width="180" height="16" rx="2" class="lora-vec"${tip(tipH)}/>`+T(cx,16,tx('output h','输出 h'),'','middle');
  s+=`<rect x="34" y="190" width="180" height="16" rx="2" class="lora-vec"${tip(tipX)}/>`+T(cx,202,tx('input x','输入 x'),'','middle');
  /* the two paths up from x */
  s+=`<path d="M${xl} 189 V158" class="lora-iln" marker-end="url(#lora-arr)"/><path d="M${xr} 189 V146" class="lora-iln" marker-end="url(#lora-arr)"/>`;
  /* left: the frozen W */
  s+=`<rect x="${xl-42}" y="70" width="84" height="84" class="lora-w"${tip(tipW)}/>`+T(xl,108,'W','b','middle')+T(xl,126,tx('frozen','冻结'),'','middle');
  /* right: A narrows to a waist, B widens again; both trained */
  s+=`<path d="M${xr-30} 142 L${xr+30} 142 L${xr+6} 110 L${xr-6} 110 Z" class="lora-ba"${tip(tipA)}/>`+T(xr,135,'A','in','middle');
  s+=`<path d="M${xr} 110 V104" class="lora-iln"/>`;
  s+=`<path d="M${xr-6} 104 L${xr+6} 104 L${xr+30} 72 L${xr-30} 72 Z" class="lora-ba"${tip(tipB)}/>`+T(xr,90,'B','in','middle');
  s+=T(xr-12,111,tx('trained','要训练'),'s','end');
  /* the results meet at + and go up to h */
  s+=`<path d="M${xl} 69 V${cy} H${cx-11}" class="lora-iln" marker-end="url(#lora-arr)"/><path d="M${xr} 71 V${cy} H${cx+11}" class="lora-iln" marker-end="url(#lora-arr)"/>`;
  s+=T(xl+6,60,'W x','m')+T(xr+6,60,'B A x','m');
  s+=`<circle cx="${cx}" cy="${cy}" r="9" class="lora-plus"${tip(tipP)}/>`+T(cx,cy+4,'+','b','middle');
  s+=`<path d="M${cx} ${cy-10} V22" class="lora-iln" marker-end="url(#lora-arr)"/>`;
  s+=`</svg>`;
  host.innerHTML=s;
  $('#lora-mat-cap',app).textContent=tx('One matrix with LoRA: the input takes two paths','一个矩阵加上 LoRA：输入走两条路');
  note.textContent=tx('Only A and B are trained; W stays frozen.','只训练 A 和 B，W 冻住不变。');
}
/* the matrix picture: W, and beside it B (d_out × r) and A (r × d_in), the outer-product layout. Every block is labelled with its shape,
   rows × columns (the owner's review of 2026-10-08: B's was missing): A's label above A, B's at the bottom left under B, W's to the right
   of W (or under it when W is too wide for that) with the label of B·A (the dashed outline inside W, as large as W) under it; so
   d_out × r times r × d_in visibly gives d_out × d_in. The picture's accessible name says the same shapes.
   A matrix that the current setup gives no LoRA (S.tg leaves it out) draws only its frozen W and says so (review codex-b9v3 应修 6). */
const hasLora=()=>TGT[S.tg].includes(S.mat);
const shp=(a,b)=>`${fint(a)} × ${fint(b)}`;
/* a label's width in the picture's 11 px type: about 6.7 px per monospaced character (the labels hold no CJK) */
const lblW=s=>String(s).length*6.7;
function drawMat(){
  const host=$('#lora-mat',app), note=$('#lora-mat-note',app); if(!host) return;
  if(ideaView()){drawIdea(); return;}
  const mm=loraView()?'lora':S.meth, m=MOD[S.m], mi=matInfo(S.m,S.mat,S.r), full=mm==='full', q4=mm==='qlora', on=!full&&hasLora();
  const big0=Math.max(m.d,m.f), W=240, side=176, sc=side/big0;
  const ww=Math.max(6,mi.din*sc), wh=Math.max(6,mi.dout*sc), t=on?Math.max(3,3*Math.log2(S.r)-3):0, g=on?5:0;
  const ox=14+t+g, oy=26+t+g, LH=14, sw=13;
  const wL=`W ${shp(mi.dout,mi.din)}`, baL=`B·A ${shp(mi.dout,mi.din)}`, bL=`B ${shp(mi.dout,S.r)}`, aL=`A ${shp(S.r,mi.din)}`;
  /* W's labels (W, and B·A under it) go right of W from its top when they fit, else under W; B's label goes under everything on the left */
  const rx=ox+ww+8, right=rx+Math.max(lblW(wL),on?sw+lblW(baL):0)<=W-2;
  const wx=right?rx:ox, wy=right?oy+10:oy+wh+LH, wEnd=wy+(on?LH:0);
  const by=on?Math.max(oy+wh,right?wEnd+3:wEnd)+LH:0, Hh=Math.ceil(Math.max(oy+wh,on?by:wEnd)+6);
  const wState=full?tx('all trained','全部要训练'):!on?(q4?tx(`frozen, stored in 4 bits; this setup puts no LoRA on ${S.mat}`,`冻结，存成 4 位。现在的设置没有给 ${S.mat} 加 LoRA`):tx(`frozen; this setup puts no LoRA on ${S.mat}`,`冻结。现在的设置没有给 ${S.mat} 加 LoRA`)):q4?tx('frozen, stored in 4 bits','冻结，存成 4 位'):tx('frozen','冻结');
  const shapesTxt=on?tx(` Shapes: W ${shp(mi.dout,mi.din)}, B ${shp(mi.dout,S.r)}, A ${shp(S.r,mi.din)}.`,`形状：W 是 ${shp(mi.dout,mi.din)}，B 是 ${shp(mi.dout,S.r)}，A 是 ${shp(S.r,mi.din)}。`)
                    :tx(` Shape: W ${shp(mi.dout,mi.din)}.`,`形状：W 是 ${shp(mi.dout,mi.din)}。`);
  let svg=`<svg class="lora-matsvg" width="${W}" height="${Hh}" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${esc(matText(mi,full,q4,on)+shapesTxt)}">`;
  svg+=`<rect x="${ox}" y="${oy}" width="${ww.toFixed(1)}" height="${wh.toFixed(1)}" class="lora-w${q4?' q4':''}${full?' tr':''}" data-tipx="${esc(tx(`W of ${S.mat}: ${shp(mi.dout,mi.din)} (outputs × inputs), ${big(mi.W)} parameters, ${wState}.`,`${S.mat} 的 W：${shp(mi.dout,mi.din)}（输出 × 输入），${big(mi.W)} 个参数，${wState}。`))}" tabindex="0"/>`;
  svg+=`<text x="${wx.toFixed(1)}" y="${wy.toFixed(1)}" class="lora-ml" data-lbl="W">${esc(wL)}</text>`;
  if(on){
    svg+=`<rect x="${14}" y="${oy}" width="${t.toFixed(1)}" height="${wh.toFixed(1)}" class="lora-ba" data-tipx="${esc(tx(`B: ${shp(mi.dout,S.r)} (outputs × r), trained, starts at 0.`,`B：${shp(mi.dout,S.r)}（输出 × r），要训练，初始全是 0。`))}" tabindex="0"/>`;
    svg+=`<rect x="${ox}" y="${26}" width="${ww.toFixed(1)}" height="${t.toFixed(1)}" class="lora-ba" data-tipx="${esc(tx(`A: ${shp(S.r,mi.din)} (r × inputs), trained.`,`A：${shp(S.r,mi.din)}（r × 输入），要训练。`))}" tabindex="0"/>`;
    svg+=`<text x="${ox}" y="${20}" class="lora-ml" data-lbl="A">${esc(aL)}</text>`;
    svg+=`<text x="14" y="${by.toFixed(1)}" class="lora-ml" data-lbl="B">${esc(bL)}</text>`;
    const baTip=tx(`B·A: ${shp(mi.dout,mi.din)}, the same shape as W; its rank is at most r = ${S.r}.`,`B·A：${shp(mi.dout,mi.din)}，和 W 的形状一样，秩最多是 r = ${S.r}。`);
    svg+=`<rect x="${(ox+1.5).toFixed(1)}" y="${(oy+1.5).toFixed(1)}" width="${Math.max(2,ww-3).toFixed(1)}" height="${Math.max(2,wh-3).toFixed(1)}" class="lora-bxa" data-tipx="${esc(baTip)}"/>`;
    /* B·A's label, with a small dashed square like its outline */
    svg+=`<rect x="${(wx+0.5).toFixed(1)}" y="${(wEnd-8.5).toFixed(1)}" width="9" height="9" class="lora-bxs" aria-hidden="true"/>`;
    svg+=`<text x="${(wx+sw).toFixed(1)}" y="${wEnd.toFixed(1)}" class="lora-ml lora-mlba" data-lbl="BA">${esc(baL)}</text>`;
  }
  svg+=`</svg>`;
  host.innerHTML=svg;
  $('#lora-mat-cap',app).textContent=full?tx('One matrix, drawn to scale','一个矩阵，按比例画'):!on?tx(`${S.mat} has no LoRA in this setup (drawn to scale)`,`现在的设置没给 ${S.mat} 加 LoRA（按比例画）`):tx('One matrix and its B and A, drawn to scale (B and A thickened)','一个矩阵和它的 B、A，按比例画（B 和 A 加粗了）');
  note.innerHTML=matText(mi,full,q4,on,true);
  note.classList.toggle('lora-nolora',!full&&!on);
}
function matText(mi,full,q4,on,html){
  const n=v=>html?bigH(v):esc(v>999999?big(v):fint(v));
  if(full) return tx(`${S.mat}: W has ${n(mi.W)} parameters, and full fine-tuning trains all of them.`,`${S.mat}：W 有 ${n(mi.W)} 个参数，全量微调要训练全部。`);
  if(!on) return tx(`${S.mat}: W has ${n(mi.W)} parameters, ${q4?'frozen in 4 bits':'frozen'}. This setup puts no LoRA on ${S.mat}: it has no B and A, and training leaves it exactly as it was.`,
                    `${S.mat}：W 有 ${n(mi.W)} 个参数，${q4?'冻结，存成 4 位':'冻结'}。现在的设置没有给 ${S.mat} 加 LoRA，它旁边没有 B 和 A，训练时完全不变。`);
  return tx(`${S.mat}: W has ${n(mi.W)} parameters, ${q4?'frozen in 4 bits':'frozen'}. B and A together have ${n(mi.BA)}, ${pctS(mi.ratio)} of W.`,`${S.mat}：W 有 ${n(mi.W)} 个参数，${q4?'冻结，存成 4 位':'冻结'}。B 和 A 一共 ${n(mi.BA)} 个，是 W 的 ${pctS(mi.ratio)}。`);
}
function statusOf(){
  if(mode==='challenge'){const e=chEval(S.meth,S.r,S.tg,S.rc); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the plan fits in 48 GB and trains at least 100M parameters. Now: ${GB(e.tot)}, ${big(e.trained)} trained.`,`方案放得下 48 GB、要训练的参数至少 100M 个时，徽章是绿色。现在：${GB(e.tot)}，训练 ${big(e.trained)} 个。`)};}
  const c=compute(S);
  if(!c.fits) return {cls:'bad',txt:tx('Out of memory','放不下'),tip:tx(`This setup needs ${GB(c.tot)}; ${gpuName(S.gpu)} has ${GB(c.cap)}.`,`这个设置要 ${GB(c.tot)}；${gpuName(S.gpu)}只有 ${GB(c.cap)}。`)};
  return {cls:'ok',txt:tx('Fits','放得下'),tip:tx(`This setup needs ${GB(c.tot)}, within ${GB(c.cap)}.`,`这个设置要 ${GB(c.tot)}，在 ${GB(c.cap)} 以内。`)};
}
function drawHead(){
  const t=$('#lora-st-title',app), sub=$('#lora-st-sub',app), st=$('#lora-status',app), c=compute(S);
  t.textContent=tx(`Does this fine-tune fit ${gpuName(S.gpu)}?`,`这次微调，${gpuName(S.gpu)}放得下吗？`);
  sub.textContent=tx(`${c.m.name}, ${S.meth==='full'?'full fine-tuning':`${methName(S.meth)}: rank r = ${S.r} on ${tgName(S.tg)}`}, ${fint(S.seq)}-token sequence, ${S.rc?'every layer recomputed':'no recomputation'}`,
    `${c.m.name}，${S.meth==='full'?'全量微调':`${methName(S.meth)}：秩 r = ${S.r}，加在${tgName(S.tg)}上`}，每条序列 ${fint(S.seq)} 个 token，${S.rc?'每一层都重算':'不重算'}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const c=compute(S), hide=hidden0(), lv=loraView(), cl=lv?compute(S8({})):c;
  const tp={k:'tp',label:lv?(ideaView()?tx('With LoRA, trained parameters','改用 LoRA，要训练的参数'):tx('With a rank-16 LoRA, trained parameters','如果加秩 16 的 LoRA，要训练的参数')):tx('Trained parameters','要训练的参数'),v:hide?'—':big(cl.trained),sub:hide?(isP('predict')?tx('shown once you answer','作答后揭晓'):tx(`shown in part ${pn('predict')}`,`第 ${pn('predict')} 部分揭晓`)):tx(`${pctS(cl.share)} of all ${big(cl.m.P)}`,`占全部 ${big(cl.m.P)} 个的 ${pctS(cl.share)}`),
    tip:hide?(isP('predict')?tx('Hidden until you answer.','作答之前先隐藏。'):tx(`Hidden until you answer in part ${pn('predict')}.`,`在第 ${pn('predict')} 部分作答之前先隐藏。`)):(!lv&&S.meth==='full'?tx(`Full fine-tuning trains all ${fint(c.m.P)} parameters.`,`全量微调训练全部 ${fint(c.m.P)} 个参数。`):tx(`${c.m.L} layers, rank ${S.r}, and the inputs plus outputs of ${tgName(S.tg)} in each layer give ${fint(cl.lp)}. The formula is in “Show the math”.`,`${c.m.L} 层、秩 ${S.r}，再乘上每层${tgName(S.tg)}的输入数加输出数，一共 ${fint(cl.lp)} 个。公式在“展开公式”里。`))};
  const mem={k:'mem',label:tx('GPU memory','显存合计'),v:GB(c.tot),sub:c.fits?tx(`${GB(c.room)} left of ${GB(c.cap)}`,`共 ${GB(c.cap)}，还剩 ${GB(c.room)}`):tx(`${GB(-c.room)} more than ${GB(c.cap)}`,`比 ${GB(c.cap)} 多 ${GB(-c.room)}`),
    tip:tx(`The sum of the blocks in the chart: ${c.blocks.map(([k,v])=>`${L(k==='lr'?BLK.lrS:BLK[k])} ${G2(v)}`).join(' + ')} = ${S2(c.blocks.map(b=>b[1]))}.`,`图里各块之和：${c.blocks.map(([k,v])=>`${L(k==='lr'?BLK.lrS:BLK[k])} ${G2(v)}`).join(' + ')} = ${S2(c.blocks.map(b=>b[1]))}。`)};
  const cp={k:'cp',label:tx('Compute per token','每个 token 的计算'),v:(S.meth==='full'?'':tx('about ','约 '))+`${c.fl}N`,sub:S.meth==='full'?(S.rc?tx('full fine-tuning’s 6N, plus 2N to recompute','全量微调 6N，加上重算的 2N'):tx('2N forward, 4N backward','前向 2N，反向 4N')):(S.rc?tx(`${methName(S.meth)}’s 4N, plus 2N to recompute`,`${methName(S.meth)} 约 4N，加上重算的 2N`):tx('the frozen weights get no gradients','冻结的权重不求梯度')),
    tip:TIP.n4()+tx(` N = ${big(c.m.P)} here.`,`这里 N = ${big(c.m.P)}。`)};
  if(mode==='challenge') return [mem,tp];
  /* parts 1-3 meet only the idea and the shapes: no compute card yet (review codex-b9v3 应修 2) */
  if(lv) return [tp,mem];
  return [tp,mem,cp];
}
function drawMetrics(){
  const host=$('#lora-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&m.k==='mem'){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:1})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(userAct||prev!==m.v){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4) */
function mathHTML(){
  const c=compute(S), m=c.m, t=T_, mi=matInfo(S.m,S.mat,S.r);
  /* H2 (the owner, 2026-10-09): a few short blocks, each a heading, one lead line, the named formula, the reader's numbers under it and
     the result, with a hairline between blocks (H5). B is d_out × r and A is r × d_in (the owner's review of 2026-10-08) */
  const rows=[], H=x=>`<div class="lora-mh">${x}</div>`, BY=tx('bytes','字节'), TG=v=>G2(v).replace(' GB','\\,\\text{GB}');
  if(!ideaView()) rows.push(H(tx('LoRA parameters of one matrix','一个矩阵的 LoRA 参数'))+
    `<p>${tx(`B has ${A.ui.math('d_{out} \\times r')} parameters and A has ${A.ui.math('r \\times d_{in}')}.`,`B 有 ${A.ui.math('d_{out} \\times r')} 个参数，A 有 ${A.ui.math('r \\times d_{in}')} 个。`)}</p>`+
    EQ(tx('LoRA parameters','LoRA 参数'),['d_{out} \\times r + r \\times d_{in}',`${fint(mi.dout)} \\times ${S.r} + ${S.r} \\times ${fint(mi.din)}`,fint(mi.BA)])+
    `<p>${tx(`${S.mat} has ${DIN()} = ${fint(mi.din)} inputs and ${DOUT()} = ${fint(mi.dout)} outputs, and the rank is ${t('r','r')} = ${S.r}. W, for comparison, has ${fint(mi.dout)} × ${fint(mi.din)} = ${fint(mi.W)}.`,`${S.mat} 有 ${DIN()} = ${fint(mi.din)} 个输入、${DOUT()} = ${fint(mi.dout)} 个输出，秩 ${t('r','r')} = ${S.r}。作为对比，W 有 ${fint(mi.dout)} × ${fint(mi.din)} = ${fint(mi.W)} 个参数。`)}</p>`);
  const sh=c.sh, per=TGT[S.tg].reduce((a,k)=>a+sh[k][0]+sh[k][1],0);
  if(hidden0()) rows.push(H(tx('All LoRA parameters','全部 LoRA 参数'))+`<p class="muted">${(isP('predict')?tx('Shown once you answer.','你作答之后给出。'):tx(`Shown in part ${pn('predict')}, after your prediction.`,`在第 ${pn('predict')} 部分、你作答之后给出。`))}</p>`);
  else rows.push(H(tx('All LoRA parameters','全部 LoRA 参数'))+
    `<p>${tx(`Every layer adapts the same matrices, so add up their inputs and outputs and multiply by the rank and the number of layers:`,`每一层都在同样的矩阵上加 LoRA，所以把这些矩阵的输入数和输出数加起来，再乘以秩和层数：`)}</p>`+
    EQ(tx('all LoRA parameters','全部 LoRA 参数'),['L \\times r \\times \\sum (d_{in} + d_{out})',`${m.L} \\times ${S.r} \\times ${fint(per)}`,fint(c.lpCfg)])+
    `<p>${tx(`${t('L','L')} = ${m.L} layers. The sum runs over ${t('tg',tgName(S.tg))}: ${fint(per)} per layer. That is ${pctS(c.lpCfg/m.P)} of the model’s ${fint(m.P)} parameters.`,`层数 ${t('L','L')} = ${m.L}。求和对${t('tg',tgName(S.tg))}，每层 ${fint(per)}。这占模型 ${fint(m.P)} 个参数的 ${pctS(c.lpCfg/m.P)}。`)}</p>`);
  /* the memory of each block: its parameters times the bytes each one keeps */
  const mem=(nm,n,b,v)=>EQ(nm,[`${n} \\times ${b}\\,\\text{${BY}}`,v]);
  const wl=S.meth==='full'?`<p>${tx('Full fine-tuning keeps 18 bytes for every parameter: the BF16 weight (2), the FP32 gradient (4), the FP32 copy of the weight and Adam’s m and v (4 each).','全量微调给每个参数存 18 字节：BF16 权重 2 字节，FP32 梯度 4 字节，FP32 的权重副本和 Adam 的 m、v 各 4 字节。')}</p>`+mem(tx('model states','模型状态'),fint(m.P),18,TG(B_FULL*m.P)):
    S.meth==='lora'?`<p>${tx('The frozen weights stay in BF16, 2 bytes each. Each LoRA parameter keeps 16 bytes: the FP32 weight, its gradient, m and v.','冻结的权重存成 BF16，每个 2 字节。每个 LoRA 参数存 16 字节：FP32 的权重、梯度、m 和 v。')}</p>`+mem(tx('frozen weights','冻结的权重'),fint(m.P),2,TG(2*m.P))+mem(tx('LoRA parameters','LoRA 参数'),fint(c.lp),16,TG(16*c.lp)):
    `<p>${tx('QLoRA stores the linear layers’ weights in 4 bits. Each one takes half a byte, plus its share of the quantization constants: one 1-byte constant per 64 weights, and one 4-byte constant per 64 × 256 weights.','QLoRA 把线性层的权重存成 4 位。每个权重占半个字节，再加上它分摊的量化常数：每 64 个权重共用一个 1 字节的常数，每 64 × 256 个权重再共用一个 4 字节的常数。')}</p>`+
      EQ(tx('bytes per 4-bit weight','每个 4 位权重的字节'),['0.5 + \\frac{1}{64} + \\frac{4}{64 \\times 256}',fixd(NF4,6)])+
      `<p>${tx('The embedding table and the LM head stay in BF16, 2 bytes each; each LoRA parameter keeps 16 bytes.','embedding 表和 LM head 仍是 BF16，每个 2 字节。每个 LoRA 参数存 16 字节。')}</p>`+
      mem(tx('4-bit weights','4 位权重'),fint(c.lin),fixd(NF4,6),TG(val(c,'q4')))+mem(tx('BF16 weights','BF16 权重'),fint(c.eo+c.nrm),2,TG(val(c,'fb')))+mem(tx('LoRA parameters','LoRA 参数'),fint(c.lp),16,TG(16*c.lp));
  rows.push(H(tx(`Weights and trained parameters (${methName(S.meth)})`,`权重和要训练的参数（${methName(S.meth)}）`))+wl);
  rows.push(H(tx(`Activations and logits (as in ${labLink('training-memory')})`,`激活和 logits（和${labLinkZ('training-memory')}一样）`))+
    `<p>${S.rc?tx('With every layer recomputed, each layer keeps only its input: 2d bytes per token.','每一层都重算时，每层只存它的输入：每个 token 2d 字节。'):tx('Each layer keeps what its backward pass needs, a fixed number of bytes per token.','每一层存下反向要用的东西，每个 token 占固定的字节数。')}</p>`+
    EQ(tx('activations','激活'),[`\\text{${tx('tokens','token 数')}} \\times \\text{${tx('layers','层数')}} \\times \\text{${tx('bytes per token per layer','每个 token 每层的字节')}}`,`${fint(S.seq)} \\times ${m.L} \\times ${fint(S.rc?c.aI:c.aL)}`,TG(c.acts)])+
    `<p>${tx(`The logits are ${t('seq',fint(S.seq))} tokens × 128,256 vocabulary entries × 4 bytes = ${G2(c.logits)}.`,`logits 是 ${t('seq',fint(S.seq))} 个 token × 词表的 128,256 项 × 4 字节 = ${G2(c.logits)}。`)}</p>`);
  rows.push(H(tx('Peak, at the end of the forward pass','峰值（前向结束时）'))+`<p>${tx('Every block above is in memory at once:','上面每一块同时都在显存里：')}</p>`+
    EQ(tx('peak','峰值'),[c.blocks.map(b=>TG(b[1])).join(' + '),S2(c.blocks.map(b=>b[1])).replace(' GB','\\,\\text{GB}')])+`<p>${tx(`${gpuName(S.gpu)} holds ${GB(c.cap)}.`,`${gpuName(S.gpu)}有 ${GB(c.cap)}。`)}</p>`);
  /* compute per token, term by term (the owner's review of 2026-10-08: 6N here, 8N in other labs; the setting decides), shown from the
     part that brings the compute card */
  if(!loraView()){const lo=S.meth!=='full', parts=[['2N',tx('forward','前向')],[lo?'2N':'4N',tx('backward','反向')]].concat(S.rc?[['2N',tx('recompute','重算')]]:[]);
    rows.push(H(tx(`Compute per token (${methName(S.meth)}, ${S.rc?'every layer recomputed':'no recomputation'})`,`每个 token 的计算（${methName(S.meth)}，${S.rc?'每一层都重算':'不重算'}）`))+
      `<p>${lo?tx('LoRA gives the frozen weights no gradient, so its backward pass is only about 2N. Full fine-tuning’s backward pass is 4N: 6N in all without recomputation, 8N with it.','LoRA 不给冻结的权重算梯度，反向只要约 2N。全量微调的反向是 4N，不重算一共 6N，重算一共 8N。')
              :tx('Full fine-tuning computes gradients for the inputs and for the weights, 4N in the backward pass. LoRA gives the frozen weights no gradient: about 4N without recomputation, about 6N with it.','全量微调的反向要给输入和权重都算梯度，是 4N。LoRA 不给冻结的权重算梯度，不重算约 4N，重算约 6N。')}</p>`+
      EQ(tx('operations per token','每个 token 的运算'),[parts.map(p=>`\\text{${p[1]}}`).join(' + '),parts.map(p=>p[0]).join(' + '),[lo?'\\approx':'=',`${c.fl}N`]])+
      `<p>${tx(`${t('N','N')} = ${big(m.P)} is the number of parameters.`,`${t('N','N')} = ${big(m.P)}，是模型的参数量。`)}</p>`);}
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge') Object.assign(S,{m:CH.m,gpu:CH.gpu,seq:CH.seq});
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawMem(); drawMat(); drawMetrics();
  const mb=$('#lora-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the method, the rank, the matrices and recomputation, in the task card. Everything else is fixed: ${MOD['70b'].name}, one 48 GB GPU, 2,048-token sequences.`,`挑战中只能在任务卡里改方法、秩、加在哪些矩阵上和重算。其余设置固定：${MOD['70b'].name}，一张 48 GB 的卡，每条序列 2,048 个 token。`);
  if(mode!=='guided') return '';
  return tx('In this part you change the setting in the card. You can change every setting in Scenarios.','这一部分在卡片里改设置。在应用场景里可以改所有设置。');
}
function syncControls(){
  const setR=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  setR('seq',Math.max(0,SEQS.indexOf(S.seq)),fint(S.seq)); setR('r',Math.max(0,RANKS.indexOf(S.r)),String(S.r));
  [['m',S.m],['meth',S.meth],['tg',S.tg],['rc',String(S.rc)],['gpu',S.gpu],['mat',S.mat]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,v)));
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#lora-controls',app).classList.toggle('locked',ch);
  ['#lora-ctl-m-wrap','#lora-ctl-gpu-wrap','#lora-ctl-meth-wrap','#lora-ctl-seq-wrap','#lora-ctl-rc-wrap','#lora-ctl-reset-wrap'].forEach(id=>hide(id,g||ch));
  hide('#lora-ctl-r-wrap',g||ch||S.meth==='full'); hide('#lora-ctl-tg-wrap',g||ch||S.meth==='full');
  hide('#lora-mat-segw',ideaView());
  /* the assumptions block shows only what the current part has introduced (review codex-b9v3 应修 2) */
  hide('#lora-ctl-note',ch||(g&&step<sIdx('memory'))); hide('#lora-asm-q',g&&step<sIdx('qlora'));
  const vis=$('#lora-vis',app); if(vis) vis.classList.toggle('lora-idea',ideaView());
  const ln=$('#lora-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const c=compute(S);
  A.announce(tx(`${methName(S.meth)}: ${GB(c.tot)} on a ${GB(c.cap)} GPU, ${fitW(c)}.`,`${methName(S.meth)}：${GB(c.tot)}，GPU 有 ${GB(c.cap)}，${fitW(c)}。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  if(!SEQS.includes(S.seq)) S.seq=4096; if(!RANKS.includes(S.r)) S.r=16; if(!TGT[S.tg]) S.tg='all'; S.rc=S.rc?1:0; if(!MOD[S.m]||S.m==='405b') S.m='8b'; if(!GPUS[S.gpu]) S.gpu='g80'; if(!METHS.includes(S.meth)) S.meth='full'; if(!MATS.includes(S.mat)) S.mat='q';
  matSeen.add(S.mat);
  if(isP('rank')&&(S.r!==16||S.tg!=='all')) r4Moved=true;
  if(isP('long')&&!compute(S).fits) seqOOM=true;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setL3(me){l3Tried.add(me); setState({meth:me});}
function setL6(v){l6Sel=v; l6Tried.add(v); setState(L6.find(o=>o.v===v).s);}
function setRem(k){if(!REMS[k]) return; remView=k; S={...REMS[k]()}; renderRail(); if(innerWidth<=760) A.ui.reveal($('#lora-stage',app));}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter){const keepMat=S.mat; S={...s.enter()}; S.mat=keepMat;}
  if(s.id==='qlora'){l6Sel='lora';}
  if(s.id==='memory'&&!l3Tried.has('lora')) S.meth='full';
  if(s.id==='remember') remView='';
  const st=$('#lora-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#lora-rail .step.current',app)||$('#lora-rail',app));}
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
    const st=$('#lora-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    /* the reader has scrolled on past the lab (the page below it fills most of the screen, e.g. the Connect card): leave the page there.
       Part 8's recap got shorter in this pass (the owner's review of 2026-10-08), so the end of the lab now comes into view sooner. */
    if((st.closest('.bench')||st.parentElement).getBoundingClientRect().bottom<innerHeight/2) return;
    const card=$('#lora-rail .step.current',app)||$('#lora-rail .railcard',app)||$('#lora-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#lora-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#lora-insight',app)); else revealBox($('#lora-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; allOpen=false; S={...chState(CH.start.meth,CH.start.r,CH.start.tg,CH.start.rc),mat:S.mat};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s,mat:S.mat};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#lora-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[['8b','70b'].indexOf(s.m),METHS.indexOf(s.meth),RANKS.indexOf(s.r),TGTS.indexOf(s.tg),SEQS.indexOf(s.seq),s.rc,GPUK.indexOf(s.gpu),MATS.indexOf(s.mat)].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==8||p.some(x=>!Number.isInteger(x))) return null;
  const [mi,me,ri,ti,si,rc,gi,ai]=p, mk=['8b','70b'][mi];
  if(!mk||!METHS[me]||!RANKS[ri]||!TGTS[ti]||!SEQS[si]||(rc!==0&&rc!==1)||!GPUK[gi]||!MATS[ai]) return null;
  return {m:mk,meth:METHS[me],r:RANKS[ri],tg:TGTS[ti],seq:SEQS[si],rc,gpu:GPUK[gi],mat:MATS[ai]};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+11), pk=a=>a[Math.floor(r()*a.length)];
  const s={m:pk(['8b','70b']),meth:pk(['lora','qlora']),r:pk([8,16,32,64]),tg:pk(TGTS),seq:pk([2048,4096,8192]),rc:pk([0,1]),gpu:'g80',mat:'q'};
  return {seed,s,c:compute(s)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), s=d.s, c=d.c, qd=$('#lora-drill-q',app), a=$('#lora-drill-a',app);
  qd.innerHTML=tx(`You fine-tune <b>${c.m.name}</b> (${bigH(c.m.P)} parameters, ${c.m.L} layers, hidden size ${fint(c.m.d)}, FFN ${fint(c.m.f)}, KV heads ${c.m.KV} × 128) with <b>${methName(s.meth)}</b>: rank <b>r = ${s.r}</b> on <b>${tgName(s.tg)}</b>, one sequence of <b>${fint(s.seq)} tokens</b>, ${s.rc?'<b>every layer recomputed</b>':'<b>no recomputation</b>'}. How many parameters are trained, and does one 80 GB H100 hold the step?`,
                  `你用 <b>${methName(s.meth)}</b> 微调 <b>${c.m.name}</b>（${bigH(c.m.P)} 个参数，${c.m.L} 层，隐藏维度 ${fint(c.m.d)}，FFN ${fint(c.m.f)}，KV head ${c.m.KV} × 128）：秩 <b>r = ${s.r}</b>，加在<b>${tgName(s.tg)}</b>上，一条 <b>${fint(s.seq)} 个 token</b> 的序列，${s.rc?'<b>每一层都重算</b>':'<b>不重算</b>'}。要训练多少参数？一张 80 GB 的 H100 放得下这一步吗？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const per=TGT[s.tg].reduce((x,k)=>x+c.sh[k][0]+c.sh[k][1],0), wb=s.meth==='lora'?[['fw',val(c,'fw')]]:[['q4',val(c,'q4')],['fb',val(c,'fb')]];
  a.innerHTML=tx(`<p><b>${big(c.lp)} trained parameters; the step needs about ${GB(c.tot)}: it ${c.fits?'fits':'does not fit'} the H100.</b></p>`+
    `<p>LoRA: ${c.m.L} × ${s.r} × ${fint(per)} = ${fint(c.lp)} parameters × 16 bytes = ${G2(16*c.lp)}.</p>`+
    `<p>${s.meth==='lora'?`Frozen weights: ${fint(c.m.P)} × 2 bytes = ${G2(val(c,'fw'))}.`:`4-bit weights: ${fint(c.lin)} × ${fixd(NF4,4)} = ${G2(val(c,'q4'))}; embedding table and LM head in BF16: ${G2(val(c,'fb'))}.`}</p>`+
    `<p>Activations: ${fint(s.seq)} × ${c.m.L} × ${fint(s.rc?c.aI:c.aL)} bytes = ${G2(c.acts)}. Logits: ${fint(s.seq)} × 128,256 × 4 = ${G2(c.logits)}.</p>`+
    `<p>Total: ${wb.map(x=>G2(x[1])).join(' + ')} + ${G2(16*c.lp)} + ${G2(c.acts)} + ${G2(c.logits)} = ${S2(c.blocks.map(b=>b[1]))}.</p>`,
    `<p><b>要训练 ${big(c.lp)} 个参数。这一步约要 ${GB(c.tot)}，H100 ${c.fits?'放得下':'放不下'}。</b></p>`+
    `<p>LoRA：${c.m.L} × ${s.r} × ${fint(per)} = ${fint(c.lp)} 个参数，× 16 字节 = ${G2(16*c.lp)}。</p>`+
    `<p>${s.meth==='lora'?`冻结的权重：${fint(c.m.P)} × 2 字节 = ${G2(val(c,'fw'))}。`:`4 位权重：${fint(c.lin)} × ${fixd(NF4,4)} = ${G2(val(c,'q4'))}；BF16 的 embedding 表和 LM head：${G2(val(c,'fb'))}。`}</p>`+
    `<p>激活：${fint(s.seq)} × ${c.m.L} × ${fint(s.rc?c.aI:c.aL)} 字节 = ${G2(c.acts)}。logits：${fint(s.seq)} × 128,256 × 4 = ${G2(c.logits)}。</p>`+
    `<p>合计：${wb.map(x=>G2(x[1])).join(' + ')} + ${G2(16*c.lp)} + ${G2(c.acts)} + ${G2(c.logits)} = ${S2(c.blocks.map(b=>b[1]))}。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={r:{en:'Rank r',zh:'秩 r'},tg:{en:'Which matrices get a LoRA',zh:'LoRA 加在哪些矩阵上'},seq:{en:'Tokens per sequence',zh:'每条序列的 token 数'},mat:{en:'Matrix in the chart',zh:'图里看哪个矩阵'}};
const TERM={
 N:()=>tx(`N is the number of parameters: ${fint(MOD[S.m].P)} for ${MOD[S.m].name}, from its config.`,`N 是参数量：${MOD[S.m].name} 有 ${fint(MOD[S.m].P)} 个，来自它的 config。`),
 L:()=>tx(`The number of layers: ${MOD[S.m].L} here.`,`层数：这里是 ${MOD[S.m].L}。`),
 r:()=>tx(`The rank r: B has r columns and A has r rows. ${S.r} here.`,`秩 r：B 有 r 列，A 有 r 行。这里是 ${S.r}。`),
 tg:()=>tx(`Each matrix’s inputs plus its outputs, added up over the matrices that get a LoRA in one layer (${tgName(S.tg)}).`,`一层里加了 LoRA 的那些矩阵（${tgName(S.tg)}），每个矩阵的输入数加输出数，全部加起来。`),
 seq:()=>tx(`Tokens processed at once: one sequence of ${fint(S.seq)} tokens.`,`一次处理的 token 数：一条 ${fint(S.seq)} 个 token 的序列。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div>${jumpAct(k)?`<div class="act">${esc(jumpAct(k))}</div>`:''}`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return (k==='r'||k==='tg')?'':tx('Fixed during the challenge: change the plan in the task card.','挑战中固定不变：请在任务卡里改方案。');
  if(mode!=='guided') return '';
  if((k==='r'||k==='tg')&&isP('rank')) return '';
  if(k==='seq'&&isP('long')) return '';
  if(k==='mat'&&!ideaView()) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='fig') return tx('Click to see the two paths in the chart.','点击看图里的两条路。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return k in TERM?tx('Comes from the model’s config; nothing to change here.','来自模型的 config，这里不能改。'):'';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({r:$('#lora-ic-r',app)||$('#lora-ic-chr',app)||$('#lora-ctl-r',app),tg:$('#lora-ic-tg',app)||$('#lora-ic-cht',app)||$('#lora-ctl-tg',app),seq:$('#lora-ic-seq',app)||$('#lora-ctl-seq',app),mat:$('#lora-ic-mat',app)||$('#lora-st-mat',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#lora-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#lora-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='fig'){const f=$('#lora-matp',app); A.tip.hide(); if(f){A.ui.reveal(f); A.ui.flash(f);} return;}
  if(!CTLN[k]){if(src) A.tip.show(src,`<div>${esc(jumpAct(k)||tx('Comes from the model’s config.','来自模型的 config。'))}</div>`); return;}
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S=mode==='sandbox'?{...PRESETS[0].s}:{...BASE}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.meth,S.r,S.tg,S.rc).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#lora-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#lora-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='seq') bySettings(e.target,()=>setState({seq:SEQS[v]}));
    else if(k==='r') bySettings(e.target,()=>setState({r:RANKS[v]}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#lora-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#lora-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>{const sm=s.showMe; if(sm.l3) setL3(sm.l3); else if(sm.l6) setL6(sm.l6); else setState(sm);}); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; wantObs=true; renderRail(); wantObs=false; const fb=$(`#lora-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&mode==='guided'){const [k,x]=d.lever.split(':'); const f=()=>byLearner(()=>{if(k==='l3') setL3(x); else if(k==='l6') setL6(x);});
      if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='mat'){const inRail=!!b.closest('#lora-rail'); byLearner(()=>setState({mat:d.v})); if(inRail&&isP('lowrank')){wantObs=true; revealSoon(); wantObs=false;} return;}
      if(k==='m') bySettings(b,()=>setState({m:d.v}));
      else if(k==='meth') bySettings(b,()=>setState({meth:d.v}));
      else if(k==='tg') bySettings(b,()=>setState({tg:d.v}));
      else if(k==='rc') bySettings(b,()=>setState({rc:+d.v}));
      else if(k==='gpu') bySettings(b,()=>setState({gpu:d.v}));
      else if(k==='chm') byLearner(()=>setState({meth:d.v}));
      else if(k==='cht') byLearner(()=>setState({tg:d.v}));
      else if(k==='chc') byLearner(()=>setState({rc:+d.v}));
      if(b.closest('#lora-rail')&&mode==='guided'){wantObs=true; revealSoon(); wantObs=false;}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; sbTouched=true; setState({...p.s,mat:S.mat}); requestAnimationFrame(()=>{if(app) revealBox($('#lora-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#lora-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
  el.addEventListener('toggle',e=>{if(e.target&&e.target.id==='lora-allp') allOpen=e.target.open;},true);
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the choices tried and
   the matrices looked at, the challenge's result, the drill; the memory is recomputed from these */
const SK=Object.keys(BASE), okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o), pickS=o=>Object.fromEntries(SK.map(k=>[k,o[k]]));
const cfgOf=x=>{const c=okObj(x)?parse(token(x)):null; return c?pickS(c):null;};
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:pickS(S),pred,observed:{...observed},saved:saved?pickS(saved):null,matSeen:[...matSeen],l3Tried:[...l3Tried],l6Tried:[...l6Tried],l6Sel,r4Moved,seqOOM,remView,
    chSubmitted,chReveal,allOpen,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.meth,S.r,S.tg,S.rc).ok};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...(cfgOf(o.S)||BASE)};
    if(mode==='challenge') S=pickS({...chState(S.meth,S.r,S.tg,S.rc),mat:S.mat});
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    matSeen=new Set(['q',...arr(o.matSeen,v=>MATS.includes(v)),S.mat]); l3Tried=new Set(['full',...arr(o.l3Tried,v=>METHS.includes(v))]);
    l6Sel=L6.some(x=>x.v===o.l6Sel)?o.l6Sel:'lora'; l6Tried=new Set(['lora',...arr(o.l6Tried,v=>L6.some(x=>x.v===v)),l6Sel]);
    r4Moved=o.r4Moved===true; seqOOM=o.seqOOM===true; remView=typeof o.remView==='string'&&Object.prototype.hasOwnProperty.call(REMS,o.remView)?o.remView:'';
    saved=mode==='challenge'?cfgOf(o.saved):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; allOpen=chSubmitted&&o.allOpen===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)) sbTouched=true; S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#lora-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#lora-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&step===0){S={...STEPS[0].enter(),mat:S.mat};}
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#lora-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#lora-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#lora-vis',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#lora-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), c=compute(S);
      status.className='status '+s.cls; status.textContent=s.txt;
      const mx=Math.max(c.tot,c.cap)*1.05;
      vis.innerHTML=`<div class="lora-dk"><i class="w" style="width:${(c.blocks.filter(b=>['w','fw','q4','fb'].includes(b[0])).reduce((a,b)=>a+b[1],0)/mx*100).toFixed(1)}%"></i><i class="s" style="width:${(c.blocks.filter(b=>['g','c','mv','lr'].includes(b[0])).reduce((a,b)=>a+b[1],0)/mx*100).toFixed(1)}%"></i><i class="a" style="width:${((c.acts+c.logits)/mx*100).toFixed(1)}%"></i><b style="left:${(c.cap/mx*100).toFixed(1)}%"></b></div>`;
      read.innerHTML=esc(tx(`${c.m.name}, ${methName(S.meth)}: ${GB(c.tot)}; the GPU has ${GB(c.cap)}`,`${c.m.name}，${methName(S.meth)}：${GB(c.tot)}；GPU 有 ${GB(c.cap)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__lora=Object.freeze({
  compute:o=>{const c=compute({...BASE,...o}); return deep({...c,m:{...c.m},g:{...c.g}});}, loraOf:(mk,r,tg)=>loraOf(MOD[mk],r,tg), matInfo,
  chEval:(me,r,tg,rc)=>deep(chEval(me,r,tg,rc)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), chPass:()=>deep(chPass()),
  chParts:(me,r,tg,rc)=>deep(chParts({meth:me,r,tg,rc})), hasLora:()=>hasLora(),
  drill:s=>{const d=makeDrill(s); return deep({s:d.s,tot:d.c.tot,lp:d.c.lp,fits:d.c.fits});},
  consts:Object.freeze({MOD:deep(MOD),GPUS:deep(GPUS),RANKS:RANKS.slice(),SEQS:SEQS.slice(),NF4,NF4_NODQ,CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),CODE}),
  state:()=>({...S,mode,step,pred,observed:{...observed},matSeen:[...matSeen],l3Tried:[...l3Tried],l6Tried:[...l6Tried],l6Sel,r4Moved,seqOOM,remView,chSubmitted,chReveal,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),explainA:e.explain.map(q=>q.a)});}
});
})();
