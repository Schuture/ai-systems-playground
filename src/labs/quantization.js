/* quantization (world inference): BF16 → FP8 → FP4: what do you actually lose?
   Fidelity: Live. A small linear layer (256 inputs, 128 outputs, 32,768 weights) is generated with a seed and quantized in the browser
   with the real formats and real rounding (round to nearest, ties to even, saturating); real-scale memory and step times follow the
   batch-10 brief (S2, S3, S7).
   - Weights: w = 0.02 · z0 / √((z1² + … + z6²) / 6) (Student-t with 6 degrees of freedom; each z is a sum of 12 uniforms − 6), then one
     weight in 256 (on average) is multiplied by 20 to 60: the lab's planted outliers. Stored in BF16 (the baseline).
   - Inputs: 1,000 vectors of 256 numbers (z as above); 4 channels are 20 times larger (activation outlier channels, as LLM.int8 and
     SmoothQuant describe for activations).
   - Schemes: FP8 E4M3 with one scale per tensor for weights and one per input vector for activations (W8A8); INT8 / INT4 symmetric
     (−127…127 / −7…7) with one FP16 scale per row (256) or per block of 128, 32, 16; MXFP4 (E2M1, 32 values share 2^(⌊log2 amax⌋ − 2));
     NVFP4 (E2M1, 16 values share an E4M3 scale amax ÷ 6 ÷ g, with g = tensor amax ÷ (448 × 6)). Bits per weight count the scales
     (per-row and per-tensor scales add under 0.01 bit and are left out).
   - Measures: relative error of the layer's output (Frobenius, over the 1,000 inputs); the share of inputs whose largest output is
     the same as BF16's ("same pick as BF16").
   - Real scale: weights = parameters × bits ÷ 8 (every parameter in the format); 80 GiB × 0.9 − 2 GiB = 70 GiB per H100; KV cache
     128 KiB (8B) / 320 KiB (70B) per token in BF16, half in FP8. Step time (8B): max(FLOPs ÷ (989.5 TFLOP/s × 50%), bytes ÷
     (3.35 TB/s × 60%)); linear layers at 1,979 TFLOP/s when weights and activations are FP8, attention at the BF16 rate.
   Words (EN / ZH): quantization 量化; scale 缩放系数; block 块; outlier 离群值; weight-only 只量化权重; weight and activation 权重和激活都量化;
   same pick as BF16 和 BF16 选的一样; part 部分. window.__qz is a read-only hook for tests/quantization_numbers.py and _flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='quantization';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const M8='Llama‑3.1‑8B', M70='Llama‑3.1‑70B', M405='Llama‑3.1‑405B', GPU='H100';

/* =====================================================================
   engine: formats and rounding
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const FMT={bf16:{m:7,emin:-126,max:(2-2**-7)*2**127},fp16:{m:10,emin:-14,max:65504},e4m3:{m:3,emin:-6,max:448},e2m1:{m:1,emin:0,max:6}};
const fl2=a=>{let e=Math.floor(Math.log2(a)); if(2**e>a) e--; if(2**(e+1)<=a) e++; return e;};
const rne=v=>{const lo=Math.floor(v), d=v-lo; return d>0.5?lo+1:(d<0.5?lo:(lo%2===0?lo:lo+1));};
/* round x to the nearest value of float format f (ties to even); above the largest value it saturates */
function qf(x,f){const F=FMT[f], a=Math.abs(x); if(a===0) return 0; let ex=fl2(a); if(ex<F.emin) ex=F.emin;
  const q=2**(ex-F.m); let v=rne(a/q)*q; if(v>F.max) v=F.max; return x<0?-v:v;}
const E2M1=[0,0.5,1,1.5,2,3,4,6];

/* the small layer */
const O=128, D=256, NX=1000, NU=6, POUT=1/256, OLO=20, OHI=60, NCH=4, CHF=20, SIG=0.02, SEED=1, CH_SEED=2;
const GEN={};
function gen(seed){
  if(GEN[seed]) return GEN[seed];
  const r=mulberry(seed*7919+13), nW=O*D, W=new Float64Array(nW), hit=new Uint8Array(nW);
  const z=()=>{let s=0; for(let i=0;i<12;i++) s+=r(); return s-6;};
  for(let k=0;k<nW;k++){const z0=z(); let c=0; for(let i=0;i<NU;i++){const t=z(); c+=t*t;}
    let w=SIG*z0/Math.sqrt(c/NU); const u=r(), f=OLO+(OHI-OLO)*r(); if(u<POUT){w*=f; hit[k]=1;} W[k]=qf(w,'bf16');}
  const ch=[]; for(let i=0;i<64;i++){const v=r(); if(ch.length<NCH){const c=Math.floor(v*D); if(!ch.includes(c)) ch.push(c);}}
  const X=new Float64Array(NX*D), big=new Uint8Array(D); ch.forEach(c=>big[c]=1);
  for(let n=0;n<NX;n++) for(let j=0;j<D;j++) X[n*D+j]=z()*(big[j]?CHF:1);
  return (GEN[seed]={W,hit,X,ch,Q:{},E:{},Y:null});
}
/* schemes of the small layer: format, block (weights per scale), levels, bits per weight with the scales */
const SCH={
 bf16:{k:'bf16',bits:16,name:{en:'BF16 (no quantization)',zh:'BF16（不量化）'},short:'BF16'},
 fp8:{k:'fp8',bits:8,act:true,name:{en:'FP8 weights (one scale per tensor) and activations (one per token)',zh:'FP8 权重（每个张量一个缩放系数）和激活（每个 token 一个）'},short:'FP8'},
 int8:{k:'int8',bits:8,blk:D,lv:127,name:{en:'INT8, one scale per row',zh:'INT8，每行一个缩放系数'},short:'INT8'},
 'int4-row':{k:'int4-row',bits:4,blk:D,lv:7,name:{en:'INT4, one scale per row (256)',zh:'INT4，每行（256 个）一个缩放系数'},short:{en:'INT4 per row',zh:'INT4 每行'}},
 'int4-128':{k:'int4-128',bits:4.125,blk:128,lv:7,name:{en:'INT4, one scale per 128',zh:'INT4，每 128 个一个缩放系数'},short:{en:'INT4 per 128',zh:'INT4 每 128 个'}},
 'int4-32':{k:'int4-32',bits:4.5,blk:32,lv:7,name:{en:'INT4, one scale per 32',zh:'INT4，每 32 个一个缩放系数'},short:{en:'INT4 per 32',zh:'INT4 每 32 个'}},
 'int4-16':{k:'int4-16',bits:5,blk:16,lv:7,name:{en:'INT4, one scale per 16',zh:'INT4，每 16 个一个缩放系数'},short:{en:'INT4 per 16',zh:'INT4 每 16 个'}},
 mx:{k:'mx',bits:4.25,blk:32,name:{en:'MXFP4 (FP4, 32 share a power-of-two scale)',zh:'MXFP4（FP4，每 32 个共用一个 2 的幂缩放系数）'},short:'MXFP4'},
 nv:{k:'nv',bits:4.5,blk:16,name:{en:'NVFP4 (FP4, 16 share an FP8 scale)',zh:'NVFP4（FP4，每 16 个共用一个 FP8 缩放系数）'},short:'NVFP4'}};
const ALLK=['bf16','fp8','int8','int4-row','int4-128','int4-32','int4-16','mx','nv'];
const CHK=['bf16','fp8','int8','int4-row','int4-128','mx','nv'];
const sname=k=>L(SCH[k].name), sshort=k=>{const s=SCH[k].short; return typeof s==='string'?s:L(s);};
function quantW(seed,k){
  const G=gen(seed); if(G.Q[k]) return G.Q[k];
  const W=G.W, out=new Float64Array(W.length), sc=SCH[k];
  if(k==='bf16') out.set(W);
  else if(k==='fp8'||k==='nv'){let am=0; for(let i=0;i<W.length;i++) am=Math.max(am,Math.abs(W[i]));
    if(k==='fp8'){const s=am/448; for(let i=0;i<W.length;i++) out[i]=qf(W[i]/s,'e4m3')*s;}
    else {const g=am/(448*6); for(let b=0;b<W.length;b+=16){let ab=0; for(let i=b;i<b+16;i++) ab=Math.max(ab,Math.abs(W[i]));
      const s=qf(ab/6/g,'e4m3')*g; for(let i=b;i<b+16;i++) out[i]=qf(W[i]/s,'e2m1')*s;}}}
  else if(k==='mx'){for(let b=0;b<W.length;b+=32){let ab=0; for(let i=b;i<b+32;i++) ab=Math.max(ab,Math.abs(W[i]));
      const s=2**(fl2(ab)-2); for(let i=b;i<b+32;i++) out[i]=qf(W[i]/s,'e2m1')*s;}}
  else {const bs=sc.blk, lv=sc.lv; for(let b=0;b<W.length;b+=bs){let ab=0; for(let i=b;i<b+bs;i++) ab=Math.max(ab,Math.abs(W[i]));
      const s=qf(ab/lv,'fp16'); for(let i=b;i<b+bs;i++) out[i]=Math.max(-lv,Math.min(lv,rne(W[i]/s)))*s;}}
  return (G.Q[k]=out);
}
function outputs(X,W){const Y=new Float64Array(NX*O);
  for(let n=0;n<NX;n++){const xo=n*D; for(let o=0;o<O;o++){const wo=o*D; let s=0; for(let j=0;j<D;j++) s+=X[xo+j]*W[wo+j]; Y[n*O+o]=s;}}
  return Y;}
function actFP8(X){const Xq=new Float64Array(X.length);
  for(let n=0;n<NX;n++){let am=0; for(let j=0;j<D;j++) am=Math.max(am,Math.abs(X[n*D+j])); const s=am/448; for(let j=0;j<D;j++) Xq[n*D+j]=qf(X[n*D+j]/s,'e4m3')*s;}
  return Xq;}
const argmaxRow=(Y,n)=>{let b=0, v=-Infinity; for(let o=0;o<O;o++){const y=Y[n*O+o]; if(y>v){v=y; b=o;}} return b;};
/* the measures: output error, weight error, same pick as BF16 */
function evalK(seed,k){
  const G=gen(seed); if(G.E[k]) return G.E[k];
  if(!G.Y) G.Y=outputs(G.X,G.W);
  const Wq=quantW(seed,k), Y=k==='bf16'?G.Y:outputs(SCH[k].act?actFP8(G.X):G.X,Wq);
  let se=0, s0=0, we=0, w0=0, same=0;
  for(let i=0;i<Y.length;i++){const d=Y[i]-G.Y[i]; se+=d*d; s0+=G.Y[i]*G.Y[i];}
  for(let i=0;i<Wq.length;i++){const d=Wq[i]-G.W[i]; we+=d*d; w0+=G.W[i]*G.W[i];}
  for(let n=0;n<NX;n++) if(argmaxRow(Y,n)===argmaxRow(G.Y,n)) same++;
  return (G.E[k]={k,oerr:Math.sqrt(se/s0),werr:Math.sqrt(we/w0),same,agree:same/NX,bits:SCH[k].bits});
}
/* the two blocks the stage shows: a block of 32 with no planted outlier, and one whose single outlier is about 50 times the others */
let BLK=null;
function blocks(){
  if(BLK) return BLK; const G=gen(SEED); let clean=null, best=null;
  for(let r=0;r<O;r++) for(let b=0;b<D/32;b++){const s=r*D+b*32; let h=0, hi=-1; for(let j=0;j<32;j++) if(G.hit[s+j]){h++; hi=j;}
    if(h===0&&!clean) clean={r,b};
    if(h===1&&hi<16){const oth=[]; for(let j=0;j<32;j++) if(j!==hi) oth.push(Math.abs(G.W[s+j])); oth.sort((x,y)=>x-y);
      const ratio=Math.abs(G.W[s+hi])/oth[15]; if(!best||Math.abs(ratio-50)<Math.abs(best.ratio-50)) best={r,b,ratio,i:hi};}}
  return (BLK={clean,outlier:best});
}
const BSZ={'int4-row':D,'int4-128':128,'int4-32':32,'int4-16':16,mx:32,nv:16,int8:D};
/* the 32 weights of one block of the stage, quantized with scheme k: values, segments (blocks that share a scale) and their grids */
function view(bsel,k){
  const G=gen(SEED), B=blocks()[bsel], st=B.r*D+B.b*32, Wq=quantW(SEED,k), ent=[];
  for(let j=0;j<32;j++) ent.push({j,w:G.W[st+j],q:Wq[st+j],hit:!!G.hit[st+j]});
  const segs=[], bs=BSZ[k];
  if(bs){const rs=B.r*D; for(let a=Math.floor((st-rs)/bs)*bs+rs; a<st+32; a+=bs){
      let am=0, ai=a; for(let i=a;i<a+bs;i++) if(Math.abs(G.W[i])>am){am=Math.abs(G.W[i]); ai=i;}
      let s, lv; if(k==='mx'){s=2**(fl2(am)-2); lv=E2M1;} else if(k==='nv'){let ta=0; for(let i=0;i<G.W.length;i++) ta=Math.max(ta,Math.abs(G.W[i])); const g=ta/(448*6); s=qf(am/6/g,'e4m3')*g; lv=E2M1;}
      else {const L2=SCH[k].lv; s=qf(am/L2,'fp16'); lv=[]; for(let q=0;q<=L2;q++) lv.push(q);}
      segs.push({from:Math.max(0,a-st),to:Math.min(32,a+bs-st),s,lv,amax:am,setter:ai-st,size:bs});}}
  const oi0=bsel==='outlier'?B.i:-1;
  segs.forEach(sg=>{sg.zeros=0; sg.hasO=oi0>=sg.from&&oi0<sg.to; for(let j=sg.from;j<sg.to;j++){const e=ent[j]; if(j!==oi0&&e.w!==0&&e.q===0) sg.zeros++;}});
  const oi=bsel==='outlier'?B.i:-1;
  const zeros=ent.filter(e=>e.j!==oi&&e.w!==0&&e.q===0).length;
  let se=0, s0=0; ent.forEach(e=>{se+=(e.q-e.w)**2; s0+=e.w*e.w;});
  return {ent,segs,oi,zeros,berr:Math.sqrt(se/s0),r:B.r,b:B.b,ratio:B.ratio||0};
}

/* real scale (brief S1–S3, S7) */
const GiB=2**30, KiB=1024, UGPU=80*0.9-2;   // GiB per H100 for weights and the KV cache
const MOD={'8b':{n:M8,P:8030261248,kv:128*KiB},'70b':{n:M70,P:70553706496,kv:320*KiB},'405b':{n:M405,P:405853388800,kv:504*KiB}};
const RBITS={bf16:16,fp8:8,int8:8,'int4-row':4,'int4-128':4.125,'int4-32':4.5,'int4-16':5,mx:4.25,nv:4.5};
const wGiB=(m,k)=>MOD[m].P*RBITS[k]/8/GiB;
function gpus(m,k){const w=wGiB(m,k); for(const n of [1,2,4,8]) if(w/n<=UGPU) return {n,w,room:UGPU-w/n}; return {n:0,w,room:0};}
const KVB={bf16:1,fp8:0.5};
const kvTok8=(k,kv)=>Math.floor((UGPU*GiB-MOD['8b'].P*RBITS[k]/8)/(MOD['8b'].kv*KVB[kv]));
/* the 8B step model (capstone/engine.js) */
const LY=32, DM=4096, KVH=8, HD=128, FFN=14336, VOC=128256;
const P_LAYER=2*DM*DM+2*DM*KVH*HD+3*DM*FFN+2*DM, P_LAYERS=LY*P_LAYER+DM, P_HEAD=VOC*DM;
const FS=989.5e12*0.5, BS=3.35e12*0.6, CTX=2048, BIG=64, PRE=2048;
const attn=(a,b)=>2*LY*DM*(b*(b+1)-a*(a+1));
const ACT8=k=>!!(SCH[k]&&SCH[k].act);
function stepT(k,kv,kind){
  const wb=RBITS[k]/8, fl=ACT8(k)?FS*2:FS, kvb=MOD['8b'].kv*KVB[kv];
  let lin, att, kvt;
  if(kind==='d1'||kind==='d64'){const B=kind==='d1'?1:BIG; lin=B*(2*P_LAYERS+2*P_HEAD); att=B*attn(CTX,CTX+1); kvt=B*CTX;}
  else {lin=PRE*2*P_LAYERS+2*P_HEAD; att=attn(0,PRE); kvt=PRE;}
  const tc=lin/fl+att/FS, tw=wb*(P_LAYERS+P_HEAD)/BS, tkv=kvb*kvt/BS, tm=tw+tkv;
  return {tc,tw,tkv,tm,t:Math.max(tc,tm),mem:tm>=tc,flops:lin+att,bytes:wb*(P_LAYERS+P_HEAD)+kvb*kvt};
}
const KINDS=['d1','d64','pre'];
/* the challenge: Llama-3.1-70B on one H100, 16 users × 8,192 tokens; the small model's second layer must pick like BF16 on ≥ 90% */
const CH_USERS=16, CH_CTX=8192, CH_BAR=0.90;
function chEval(k,kv){const w=wGiB('70b',k), kvg=CH_USERS*CH_CTX*MOD['70b'].kv*KVB[kv]/GiB, need=w+kvg, e=evalK(CH_SEED,k);
  return {k,kv,w,kvg,need,fits:need<=UGPU,agree:e.agree,same:e.same,ok:need<=UGPU&&e.agree>=CH_BAR-1e-12};}
const chAll=()=>{const r=[]; CHK.forEach(k=>['bf16','fp8'].forEach(kv=>r.push(chEval(k,kv)))); return r;};

/* =====================================================================
   state
   ===================================================================== */
const BASE={sch:'bf16',blk:'32',kv:'bf16',model:'8b',view:'serve',bsel:'outlier'};
let S={...BASE}, mode='guided', step=0, root=null, app=null, annT=0, saved=null, lastHash='';
let pred=null, predShown=false, observed={}, tried={}, sbTouched=false, userAct=false, wantObs=false, lastM=null, actN=0;
let chSubmitted=false, chReveal=false, chW='bf16', chKV='bf16', drillSeed=1, drillOpen=false;
const keyOf=s=>{s=s||S; return s.sch==='int4'?'int4-'+s.blk:s.sch;};
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const SIX=id=>STEPS.findIndex(s=>s.id===id);

/* the rounding replay: phase 0 originals, 1 the scale and its grid, 2 each weight moves to the nearest grid value, 3 what was lost */
const SPEEDS={slow:2.4,med:1.2,fast:0.5}; let speed='med';
let AN={on:false,raf:0,start:0,ph:3,t:1,key:'',paused:false,acc:0}, watched=new Set(), nPlays=0;
const animKey=()=>S.bsel+'|'+keyOf();
const hidePred=()=>isP('predict')&&!predShown;

/* =====================================================================
   formatting and small helpers
   ===================================================================== */
const fix=(v,d)=>Number(v).toFixed(d);
const fint=n=>A.fmt.int(n);
const pc=(v,d)=>fix(v*100,d==null?1:d)+'%';
const ms=v=>fix(v*1e3,v*1e3<10?2:1);
const gib=v=>fix(v,1)+' GiB';
const bitsT=b=>String(+b.toFixed(3));
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J_=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const DEF=(text,tip)=>A.ui.def(text,tip);
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):esc(id);
const TD=(k,x)=>`<span class="t qz-td" tabindex="0" data-ctl="${k}">${x}</span>`;
const M=t=>A.ui.math(t);
const EQN=rows=>rows.some(([l])=>/\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="qz-eqs${EQN(rows)?' qz-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="qz-eq-l">${l?M(l):''}</span><span class="qz-eq-e">${M(rel||'=')}</span><span class="qz-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="qz-mb"><p>${lead}</p>${body}${res?`<p class="qz-mb-res">${res}</p>`:''}</div>`;
const SRC={
 nf:'https://github.com/NVIDIA/TransformerEngine/blob/d0b4b32c8e9046246a7ea20025eb06eaba3d7a6b/docs/features/low_precision_training/nvfp4/nvfp4.rst',
 mx:'https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/README.md',
 vfp8:'https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/features/quantization/fp8.md',
 vkv:'https://github.com/vllm-project/vllm/blob/ac8c2130a8a6ecc4940f6c5433cb2ba15e983aa7/docs/features/quantization/quantized_kvcache.md',
 vq:'https://github.com/vllm-project/vllm/blob/ac8c2130a8a6ecc4940f6c5433cb2ba15e983aa7/docs/features/quantization/README.md',
 meta:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/quantization/loader.py',
 metaq:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/quantize_impls.py',
 ds:'https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md',
 mo:'https://github.com/NVIDIA/Model-Optimizer/blob/6a1068746dd0e746fe4bcb79152b86bdcf58cb41/docs/source/guides/_choosing_quant_methods.rst',
 awq:'https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/README.md',
 sq:'https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/README.md',
 bnb:'https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/docs/source/reference/nn/linear8bit.mdx',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 pInt8:'https://arxiv.org/abs/2208.07339', pSq:'https://arxiv.org/abs/2211.10438', pAwq:'https://arxiv.org/abs/2306.00978', pGptq:'https://arxiv.org/abs/2210.17323'};

/* tooltips for terms (R2.8) */
const TIP={
 act:()=>tx('Activations: the numbers each token carries into a layer, the input the layer multiplies by its weights.','激活：每个 token 进入一层时带着的那串数，也就是这一层要和权重相乘的输入。'),
 quant:()=>tx('Quantization stores numbers (weights, activations or the KV cache) in fewer bits: each number is rounded to the nearest value the smaller format can hold.','量化：用更少的位存数（权重、激活或 KV cache），每个值舍入到这种格式能表示的最近值。'),
 scale:()=>tx('A scale: a block of numbers is divided by it and then rounded, so the block fits the format’s range; when the numbers are used, they are multiplied back. How big a block is depends on the scheme: a whole tensor, a row, or a few dozen to a few hundred numbers. The lab number-formats uses the same word.','缩放系数：一块数先除以它，再舍入，让这块数落进格式的范围；用的时候再乘回来。一块多大由方案定：整个张量、每一行，或者几十到上百个挨着的值。数值格式 Lab 也这么叫。'),
 block:()=>tx('A block: the numbers that share one scale. They sit next to each other in one row of the weight matrix.','块：共用一个缩放系数的那些数。它们在权重矩阵的同一行里挨着。'),
 outlier:()=>tx('An outlier: a number much larger than the others in its block. It sets the block’s scale, so the other numbers are left with only a few grid values.','离群值：比同一块里其他数大很多的数。它决定这块的缩放系数，于是其他数只剩几个格子可用。'),
 int4:()=>tx('INT4 stores small whole numbers. This lab uses the symmetric form, −7 to 7; an asymmetric form also stores a zero point per block, which this lab leaves out.','INT4 存的是很小的整数。这个 Lab 用对称的形式，只用 −7 到 7；非对称的形式每块还多存一个零点，这个 Lab 不用它。'),
 small:()=>tx('The small layer: a linear layer with 256 inputs and 128 outputs, generated with a seed in your browser. Its weights have heavy tails, and a few are planted dozens of times larger (this lab’s assumption; Settings has the numbers). Read each output as the score of one possible next token.','小模型的一层：一个在你的浏览器里按种子生成的线性层，256 个输入、128 个输出。权重的分布尾巴较重，还有少数权重被放大了几十倍（这是这个 Lab 的假设，具体的数在自由调参里）。可以把每个输出看成一个候选的下一个 token 的得分。'),
 same:()=>tx('Same pick as BF16: on 1,000 sample inputs, how often the layer’s largest output (read as “which next token”) is the same as with BF16 weights. It is this lab’s stand-in for answer quality; real models are checked on eval sets.','和 BF16 选的一样：在 1,000 个样本输入上，这一层最大的那个输出（看成“选哪个下一个 token”）和 BF16 权重时一样的比例。这是这个 Lab 代替回答质量的指标；真实模型要在评测集上检查。'),
 oerr:()=>tx('Output error: how far the layer’s outputs move from the BF16 outputs, over all 1,000 inputs, as a share of their size (the root of the summed squared differences over the root of the summed squares).','输出误差：在全部 1,000 个输入上，这一层的输出离 BF16 的输出有多远，占输出本身大小的比例（差的平方和开根号，除以输出平方和开根号）。'),
 wonly:()=>tx('Weight-only quantization (W4A16 for 4-bit weights and 16-bit activations): the weights are stored in fewer bits and turned back into BF16 right before each multiply. It saves memory and reading time, not arithmetic.','只量化权重（4 位权重、16 位激活时叫 W4A16）：权重用更少的位存，每次相乘之前再还原成 BF16。省的是显存和读的时间，不省计算。'),
 wact:()=>tx('Weight-and-activation quantization (W8A8 for 8 bits each): the activations are rounded to FP8 too, so the multiply itself runs in FP8 on the GPU’s FP8 units. Only then does the H100’s FP8 arithmetic rate help.','权重和激活都量化（各 8 位时叫 W8A8）：激活也舍入成 FP8，乘法本身在 GPU 的 FP8 单元上做。只有这样，H100 的 FP8 算力才用得上。'),
 mxfp4:()=>tx(`MXFP4 (OCP Microscaling): 4-bit floats FP4 E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives); every 32 values share one 8-bit scale that is a power of two, so each value takes ${M('4 + 8/32 = 4.25')} bits.`,`MXFP4（OCP Microscaling）：4 位浮点 FP4 E2M1（0、0.5、1、1.5、2、3、4、6 和它们的负数）；每 32 个值共用一个 8 位、必须是 2 的幂的缩放系数，所以每个值占 ${M('4 + 8/32 = 4.25')} 位。`),
 nvfp4:()=>tx(`NVFP4 (NVIDIA): FP4 E2M1 values; every 16 values share one FP8 E4M3 scale, and the whole tensor has one FP32 scale on top, so each value takes ${M('4 + 8/16 = 4.5')} bits.`,`NVFP4（NVIDIA）：FP4 E2M1 的数；每 16 个值共用一个 FP8 E4M3 缩放系数，整个张量再有一个 FP32 缩放系数，所以每个值占 ${M('4 + 8/16 = 4.5')} 位。`),
 usable:()=>tx('70 GiB per H100, as in the serving labs: nine tenths of 80 GiB (vLLM’s gpu_memory_utilization) minus the runtime’s overhead. It holds the weights and the KV cache; Settings has the formula.','每张 H100 按 70 GiB 算，和推理服务的几个 Lab 一样：80 GiB 的九成（vLLM 的 gpu_memory_utilization），再减去运行时的开销。权重和 KV cache 都放在这里面；算式在自由调参里。'),
 stepm:()=>tx('A step takes the longer of its arithmetic and its memory reads, because the GPU computes while it reads. Arithmetic runs at 50% of the H100’s dense peak (989.5 TFLOP/s in BF16, 1,979 in FP8) and reads at 60% of 3.35 TB/s: both are this lab’s assumptions, as in the lab prefill-decode.','一步的耗时取计算和读显存中较长的那个，因为 GPU 一边读一边算。计算按 H100 稠密峰值的 50% 算（BF16 989.5 TFLOP/s，FP8 1,979），读显存按 3.35 TB/s 的 60% 算。两者都是这个 Lab 的假设，和 prefill-decode 那个 Lab 一样。')};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'keep',t:{en:'They are stored about as well as in the block without the outlier',zh:'存得和没有离群值的那块差不多准'}},
  {v:'step',t:{en:'Each moves about one grid step: worse, but each keeps a value',zh:'每个偏差大约一个格子：差一些，但都还有值'}},
  {v:'zero',t:{en:'Most of them become 0',zh:'大部分变成 0'}}];
const PRED_RIGHT='zero';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const P1=[['bf16',{en:'BF16',zh:'BF16'}],['fp8',{en:'FP8',zh:'FP8'}],['int4-128',{en:'4-bit',zh:'4 位'}]];
const P4=['int4-row','int4-128','int4-32','int4-16','mx','nv'];
const P5={bf16:{t:{en:'BF16 everywhere (initial setup)',zh:'全部 BF16（初始设置）'},d:{en:'Weights, activations and KV cache in BF16.',zh:'权重、激活和 KV cache 都是 BF16。'},s:{sch:'bf16',kv:'bf16'}},
  w4:{t:{en:'4-bit weights only',zh:'只把权重量化到 4 位'},d:{en:'INT4 with one scale per 128 weights; activations and multiplies stay BF16.',zh:'INT4，每 128 个权重一个缩放系数；激活和乘法还是 BF16。'},s:{sch:'int4',blk:'128',kv:'bf16'}},
  fp8:{t:{en:'FP8 weights and activations',zh:'权重和激活都用 FP8'},d:{en:'The multiplies run in FP8; the KV cache stays BF16.',zh:'乘法用 FP8 做；KV cache 还是 BF16。'},s:{sch:'fp8',kv:'bf16'}},
  fp8kv:{t:{en:'FP8 weights and activations, FP8 KV cache',zh:'权重和激活用 FP8，KV cache 也用 FP8'},d:{en:'Each token’s keys and values take 64 KiB instead of 128 KiB.',zh:'每个 token 的 key 和 value 从 128 KiB 变成 64 KiB。'},s:{sch:'fp8',kv:'fp8'}}};
const P5K=['bf16','w4','fp8','fp8kv'];
const p5Now=()=>P5K.find(k=>{const s=P5[k].s; return s.sch===S.sch&&s.kv===S.kv&&(s.sch!=='int4'||s.blk===S.blk);})||'';
const SCN=[['8b',M8],['70b',M70],['405b',M405]];

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'serve',
  lead:()=>tx('Every decode step reads every weight once. Fewer bytes per weight make each step faster. What does the rounding lose?','decode 每一步都要把权重全读一遍。每个权重的字节少了，每一步就快了。舍入丢掉了什么？'),
  enter:()=>({...BASE,view:'serve'}),always:true,
  body:()=>{const b=stepT('bf16','bf16','d1');
   return tx(`<p>${M8} runs on one ${GPU}. In every decode step the GPU reads nearly all of its weights from memory once, whether the step makes 1 token or 64 (the lab ${labLink('prefill-decode')}).</p><p>In BF16 every weight is 2 bytes. With one request a step takes about ${ms(b.t)} ms, nearly all of it reading weights.</p><p>${DEF('Quantization',TIP.quant())} stores numbers in fewer bits: the weights, the ${DEF('activations',TIP.act())} or the KV cache. Each number is rounded to the nearest value the smaller format can hold. Fewer bytes per weight make every step faster and leave more memory for the KV cache. What the rounding loses is what this lab is about.</p><p>Pick a weight format and watch the step times in the chart. “4-bit” is INT4: each weight is stored in 4 bits, and every 128 weights also share one 16-bit scale (the next part explains it), which adds 16/128 = 0.125 bits per weight. So the “Bits per weight” card shows 4.125.</p><p>The chart has two more rows: one decode step for 64 requests together, and a prefill of 2,048 tokens at once. Each row takes the longer of reading memory and computing. Prefill spends its time computing, so the weight format does not change it; part 5 explains this.</p>`,
             `<p>一张 ${GPU} 跑 ${M8}。decode 的每一步，GPU 都要把几乎全部权重从显存读一遍，不管这一步算 1 个 token 还是 64 个（见 ${labLink('prefill-decode')}）。</p><p>BF16 每个权重占 2 字节。只有 1 个请求时，一步大约 ${ms(b.t)} 毫秒，几乎全花在读权重上。</p><p>${DEF('量化',TIP.quant())}就是用更少的位存数：权重、${DEF('激活',TIP.act())}或 KV cache。每个值舍入到新格式能表示的最近值。每个权重的字节少了，每一步就快了，显存也多出来放 KV cache。舍入会丢掉什么，就是这个 Lab 要讲的。</p><p>选一种权重格式，看看图里每一步的耗时怎么变。“4 位”指 INT4：每个权重存 4 位，每 128 个权重再共用一个 16 位的缩放系数（下一部分讲），摊到每个权重上多 16/128 = 0.125 位。所以右边“每个权重的位数”那张卡显示 4.125。</p><p>图里还有两行：64 个请求一起 decode 的一步，和一次处理 2,048 个 token 的 prefill。每一行的耗时取读显存和计算中较长的那个。prefill 的时间花在计算上，所以换权重格式它不变，第 5 部分细讲。</p>`)+p1Ctl();},
  obs:()=>obsMission()},
 {id:'round',title:{en:'Rounding one block',zh:'舍入一块权重'},view:'block',
  lead:()=>tx('Divide the block by a scale, round each weight to the grid, multiply back.','一块权重先除以缩放系数，再舍入到格子上，用的时候乘回来。'),
  enter:()=>({...BASE,view:'block',bsel:'clean',sch:'int4',blk:'32'}),expect:()=>watched.has('clean|int4-32'),showMe:'play',label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Llama has far too many weights to draw one by one on the page, so from here a small model stands in: ${DEF('a small layer',TIP.small())} with 256 inputs and 128 outputs, generated by the page from a fixed seed. Take 32 weights that sit next to each other in one row of it. Together they form one ${DEF('block',TIP.block())}, to be stored in 4 bits as ${DEF('INT4',TIP.int4())}.</p><p>INT4 holds only whole numbers from −7 to 7.</p><p>These weights all lie within ±${fix(view('clean','int4-32').segs[0].amax,3)}: rounded as they are, every one would become 0.</p><p>So the block is first stretched to INT4’s range, rounded, and shrunk back. Quantization does three things:</p><ol class="qz-steps3"><li>Divide the block by a ${DEF('scale',TIP.scale())} s, chosen so the weight with the largest absolute value lands on 7 (or −7), the edge of INT4’s range.</li><li>Round each result to the nearest whole number from −7 to 7.</li><li>When the weight is used, multiply it back by s. The stored weight is w<sub>q</sub>:</li></ol><div class="fx">${M('w_q = s \\cdot \\operatorname{round}(w / s)')}</div><div class="fx">${M('s = \\max |w| \\,/\\, 7')}</div>`,
              `<p>Llama 的权重太多，没法在页面里逐个画出来，所以下面用一个小模型代替：${DEF('小模型的一层',TIP.small())}只有 256 个输入、128 个输出，是页面按固定种子生成的。取这一层里同一行中挨着的 32 个权重，它们组成一${DEF('块',TIP.block())}，要用 4 位的 ${DEF('INT4',TIP.int4())} 存。</p><p>INT4 只能存 −7 到 7 的整数。</p><p>这些权重都在 ±${fix(view('clean','int4-32').segs[0].amax,3)} 以内，直接舍入会全部变成 0。</p><p>所以先把整块放大到 INT4 的范围，舍入以后再缩回来。量化做三件事：</p><ol class="qz-steps3"><li>把这块权重除以一个${DEF('缩放系数',TIP.scale())} s。s 选得让绝对值最大的那个权重正好落在 7（或 −7）上，也就是 INT4 的边上。</li><li>把每个结果舍入到 −7 到 7 之间最近的整数。</li><li>用这个权重的时候，再乘回 s。存下来的权重记作 w<sub>q</sub>：</li></ol><div class="fx">${M('w_q = s \\cdot \\operatorname{round}(w / s)')}</div><div class="fx">${M('s = \\max |w| \\,/\\, 7')}</div>`),
  obs:()=>obsRound()},
 {id:'predict',title:{en:'Predict: one large weight',zh:'先猜：一个很大的权重'},view:'block',gate:()=>predShown,
  lead:()=>tx('A block of 32 where one weight is about 50 times the others. What happens to the rest in 4 bits?','一块 32 个权重，其中一个是其他权重的 50 倍左右。用 4 位存，其他的会怎样？'),
  enter:()=>({...BASE,view:'block',bsel:'outlier',sch:'int4',blk:'32'}),
  body:()=>{const v=view('outlier','int4-32'), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   const go=pred&&!predShown?`<p class="small qz-saved">${tx('Saved. You can still change it before you round the block.','已记下。舍入这块之前，还可以改。')}</p><div class="stepnav"><button type="button" class="btn sm" data-act="pred-go">${tx('Round this block','舍入这一块')}</button></div>`:'';
   return tx(`<p>Here is another block of 32 from the same layer.</p><p>One of its weights is especially large, about ${fix(Math.round(v.ratio/10)*10,0)} times the middle-sized one of the other 31: an ${DEF('outlier',TIP.outlier())}. The chart shows the block before rounding; the outlier reaches past the top.</p><p>The scale puts the weight with the largest absolute value on 7, and every weight is rounded to a whole number in INT4’s range.</p><p><b>What happens to most of the other 31 weights?</b></p>`,
             `<p>这是同一层里的另一块，也是 32 个权重。</p><p>其中一个权重特别大，大约是其余 31 个里中等大小那个的 ${fix(Math.round(v.ratio/10)*10,0)} 倍，这叫${DEF('离群值',TIP.outlier())}。图里是舍入之前的这一块，离群值超出了图的上沿。</p><p>缩放系数让绝对值最大的权重落在 7 上，每个权重都舍入到 INT4 范围里的整数。</p><p><b>其余 31 个权重里，大部分会怎样？</b></p>`)+ch+go;},
  obs:()=>predShown?obsPredict():''},
 {id:'blocks',title:{en:'Smaller blocks',zh:'块更小'},view:'block',
  lead:()=>tx('An outlier spoils only its own block. Smaller blocks cost one more scale each.','离群值只影响它自己那一块。块越小，越准，代价是每块多存一个缩放系数。'),
  enter:()=>({...BASE,view:'block',bsel:'outlier',sch:'int4',blk:'32'}),expect:()=>tried.p4&&(tried.p4.has('int4-16')||tried.p4.has('nv')),showMe:()=>setP4('int4-16'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>A scale is shared only inside its block, so an outlier spoils only its own block. With 16 weights per block, the half without the outlier gets its own, much smaller scale. The cost is one more 16-bit scale per block, so each weight takes more bits:</p><div class="fx">${M('\\text{bits per weight} = 4 + \\frac{16}{\\text{weights per block}}')}</div><p>Two 4-bit float formats work the same way: they store FP4 values in small blocks, each with an 8-bit scale. Which values FP4 can hold is in the lab ${labLink('number-formats')}.</p>`,
              `<p>缩放系数只在自己那一块里共用，所以离群值只影响它自己那一块。每 16 个权重一块时，不含离群值的那一半有自己的、小得多的缩放系数。代价是每块多存一个 16 位的缩放系数，每个权重占的位数就多了：</p><div class="fx">${M('\\text{每个权重的位数} = 4 + \\frac{16}{\\text{每块的权重数}}')}</div><p>两种 4 位浮点格式也是这样做的：它们按小块存 FP4 的值，每块配一个 8 位的缩放系数。FP4 能表示哪些值，见 ${labLink('number-formats')}。</p>`)+p4Ctl(),
  obs:()=>obsBlocks()},
 {id:'compute',title:{en:'Weights only, or activations too',zh:'只量化权重，还是连激活一起'},view:'serve',
  lead:()=>tx('Weight-only saves reading time; FP8 weights and activations also halve the arithmetic time.','只量化权重省的是读显存的时间；权重和激活都用 FP8，计算时间也减半。'),
  enter:()=>{tried.p5=new Set(['bf16']); return {...BASE,view:'serve'};},expect:()=>tried.p5&&tried.p5.has('w4')&&tried.p5.has('fp8'),showMe:()=>setP5(tried.p5.has('w4')?'fp8':'w4'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>${DEF('Activations',TIP.act())} are the numbers each token carries into a layer, the input the layer multiplies by its weights. ${DEF('Weight-only quantization',TIP.wonly())} stores the weights in 4 bits and turns them back into BF16 right before each multiply. It reads fewer bytes but does the same BF16 arithmetic. ${DEF('Weight-and-activation quantization',TIP.wact())} rounds the activations to FP8 too, so the multiplies run on the H100’s FP8 units, at twice the BF16 rate.</p><p>The chart shows three kinds of step for ${M8}: a decode step with 1 request, a decode step with 64 requests, and a prefill.</p><p>The prefill reads one 2,048-token prompt; every decode request already has 2,048 tokens of context.</p><p>Why one step waits on memory and another on arithmetic is in the lab ${labLink('roofline')}.</p>`,
              `<p>${DEF('激活',TIP.act())}是每个 token 进入一层时带着的那串数，也就是这一层要和权重相乘的输入。${DEF('只量化权重',TIP.wonly())}：权重用 4 位存，每次相乘之前再还原成 BF16。读的字节少了，算的还是 BF16。${DEF('权重和激活都量化',TIP.wact())}：激活也舍入成 FP8，乘法就在 H100 的 FP8 单元上做，速度是 BF16 的两倍。</p><p>图里是 ${M8} 的三种步：1 个请求的 decode 步、64 个请求的 decode 步，以及 prefill。</p><p>prefill 处理一个 2,048 个 token 的 prompt；每个 decode 请求都已经有 2,048 个 token 的上下文。</p><p>为什么有的步在等显存、有的在等计算，见 ${labLink('roofline')}。</p>`)+p5Ctl(),
  obs:()=>obsCompute()},
 {id:'scale',title:{en:'Real models on H100s',zh:'真实的模型放上 H100'},view:'serve',
  lead:()=>tx('How many H100s each model needs in each format, and how much room is left for the KV cache.','每个模型在每种格式下要几张 H100，还剩多少地方放 KV cache。'),
  enter:()=>{tried.p6=new Set(['8b']); return {...BASE,view:'serve',sch:'fp8'};},expect:()=>tried.p6&&tried.p6.has('405b'),showMe:()=>setModel('405b'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>An ${GPU} is sold as 80 GB; this site counts it as 80 ${DEF('GiB',A.terms&&A.terms.gib?A.terms.gib():'GiB = 2³⁰ bytes')}.</p><p>As in the serving labs, ${DEF('70 GiB of each',TIP.usable())} holds the weights and the KV cache.</p><p>A model too large for one GPU is split over several with tensor parallelism: every weight matrix is cut into n pieces, and each of the n GPUs holds one piece, 1/n of the weights. What the KV cache needs per token is in the lab ${labLink('kv-cache')}.</p><p>Pick a model and see how many ${GPU}s its weights need in each format, scales included. The lines marked 1×70, 2×70 and so on are the memory that 1, 2, 4 or 8 cards have together.</p>`,
              `<p>每张 ${GPU} 标称 80 GB，本站按 80 ${DEF('GiB',A.terms&&A.terms.gib?A.terms.gib():'GiB = 2³⁰ 字节')} 计。</p><p>和推理服务的几个 Lab 一样，每张卡有 ${DEF('70 GiB',TIP.usable())} 放权重和 KV cache。</p><p>一张卡放不下的模型，用张量并行分到几张卡上：每个权重矩阵切成 n 份，n 张卡每张存其中一份，也就是 1/n 的权重。KV cache 每个 token 占多少，见 ${labLink('kv-cache')}。</p><p>选一个模型，看看它的权重在每种格式下（算上缩放系数）要几张 ${GPU}。图里标着 1×70、2×70 等的线，是 1、2、4、8 张卡加起来能用的显存。</p>`)+p6Ctl(),
  obs:()=>obsScale()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:'block',
  lead:()=>tx('Count the scales, keep blocks small, and know which step each scheme speeds up.','算上缩放系数，块要小，还要知道每种方案让哪种步变快。'),
  enter:()=>({...BASE,view:'block',bsel:'outlier',sch:'int4',blk:'16'}),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function p1Ctl(){const k=keyOf(); return `<div class="qz-ic" id="qz-ic-p1">`+A.ui.seg({id:'qz-ic-p1s',key:'p1',label:tx('Weight format','权重格式'),options:P1.map(([v,t])=>[v,L(t)]),value:P1.some(p=>p[0]===k)?k:'bf16',inline:true})+`</div>`;}
function p4Res(k){const v=view('outlier',k); return tx(`${v.zeros} of 31 → 0`,`31 个里 ${v.zeros} 个 → 0`);}
const P4D={'int4-row':{en:'One scale for the whole row of 256 weights. A real model’s row has thousands of weights, so the scale’s share is negligible: counted as 4 bits.',zh:'整行 256 个权重共用一个缩放系数。真实模型的一行有几千个权重，缩放系数分摊下来可以不计，这里记作 4 位。'},'int4-128':{en:'One 16-bit scale per 128 weights.',zh:'每 128 个权重一个 16 位缩放系数。'},'int4-32':{en:'One 16-bit scale per 32 weights.',zh:'每 32 个权重一个 16 位缩放系数。'},'int4-16':{en:'One 16-bit scale per 16 weights.',zh:'每 16 个权重一个 16 位缩放系数。'},mx:{en:'FP4 values; 32 share one 8-bit scale that must be a power of two.',zh:'FP4 的值；32 个共用一个 8 位、必须是 2 的幂的缩放系数。'},nv:{en:'FP4 values; 16 share one FP8 scale, plus one FP32 scale for the whole tensor.',zh:'FP4 的值；16 个共用一个 FP8 缩放系数，整个张量再加一个 FP32 缩放系数。'}};
function p4Ctl(){const k=keyOf();
  return `<p class="small qz-lead">${tx('Pick how the row is split into blocks, and watch how many of the 31 small weights survive. The table under the block lists every scheme side by side: bits per weight, output error of the whole layer, and how often it picks the same as BF16.','选一种分块方式，看看 31 个小权重还剩多少。这一块下面的表把每种方案并排列出：每个权重占几位、整层输出偏了多少、多少输入选得和 BF16 一样。')}</p><div class="qz-lv" id="qz-ic-p4">`+A.ui.levers({aria:tx('How the weights are split into blocks, pick one','权重怎么分块，选一个'),value:k,
   options:P4.map(v=>({v,title:`${sshort(v)} · ${bitsT(SCH[v].bits)} ${tx('bits','位')}`,desc:L(P4D[v]),result:(tried.p4&&tried.p4.has(v))||v===k?p4Res(v):''}))})+`</div>`;}
function p5Ctl(){const k=p5Now();
  return `<p class="small qz-lead">${tx('Pick a scheme and watch which of the three steps gets faster.','选一种方案，看看三种步里哪一种变快了。')}</p><div class="qz-lv" id="qz-ic-p5">`+A.ui.levers({aria:tx('What is quantized, pick one','量化什么，选一个'),value:k,
   options:P5K.map(v=>({v,title:L(P5[v].t),desc:L(P5[v].d),result:(tried.p5&&tried.p5.has(v))||v===k?p5Res(v):''}))})+`</div>`;}
const p5Res=v=>{const s=P5[v].s, k=s.sch==='int4'?'int4-'+s.blk:s.sch; return tx(`decode ${ms(stepT(k,s.kv,'d1').t)} ms`,`decode ${ms(stepT(k,s.kv,'d1').t)} 毫秒`);};
function p6Ctl(){return `<div class="qz-ic" id="qz-ic-p6">`+A.ui.seg({id:'qz-ic-p6s',key:'p6',label:tx('Model (Llama‑3.1)','模型（Llama‑3.1）'),options:SCN.map(([v,n])=>[v,n.replace('Llama‑3.1‑','')]),value:S.model,inline:true})+`</div>`;}

function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsMission(){const k=keyOf(), b=stepT('bf16','bf16','d1'), r=stepT(k,'bf16','d1');
  if(k==='bf16') return tx(`<p><b>In BF16, a decode step with one request takes ${ms(b.t)} ms.</b> Reading the weights takes almost all of it; the arithmetic would take far less.</p>`,`<p><b>BF16 时，1 个请求的 decode 步要 ${ms(b.t)} 毫秒。</b>几乎全花在读权重上，计算本身用的时间少得多。</p>`);
  return tx(`<p><b>With ${k==='fp8'?'FP8 weights':'4-bit weights'} the step takes ${ms(r.t)} ms, not ${ms(b.t)} ms.</b></p><p>The GPU reads fewer bytes per weight, so each step finishes sooner.</p><p>Every weight was rounded on the way. The next part shows what that rounding does.</p>`,
            `<p><b>${k==='fp8'?'权重用 FP8':'权重用 4 位'}时，同样一步只要 ${ms(r.t)} 毫秒，而不是 ${ms(b.t)} 毫秒。</b></p><p>每个权重要读的字节少了，所以每一步结束得更早。</p><p>但每个权重都被舍入过。下一部分看看舍入做了什么。</p>`);}
function obsRound(){const v=view('clean','int4-32');
  return tx(`<p><b>The grid is the set of values this block can store: whole multiples of s, from −7s to 7s, the thin lines in the chart. Every weight moved to the nearest grid value, at most half a grid step away.</b></p><p>The grid steps are smaller than most of the weights; only the ${v.zeros} smallest weights rounded to 0.</p><p>The error and pick cards on the right are for the whole layer stored this way: how far its outputs move, and on how many of 1,000 inputs its largest output is the same one as in BF16.</p>`,
            `<p><b>格子就是这一块能存的值：s 的整数倍，从 −7s 到 7s，图里的细线。每个权重都挪到了最近的格子上，最多偏半格。</b></p><p>格子的间距比大多数权重小，只有 ${v.zeros} 个特别小的权重舍入成 0。</p><p>右边的“输出误差”和“和 BF16 选的一样”算的是整层都这样存时的结果：这一层的输出偏了多少，以及 1,000 个输入里，这一层最大的那个输出和不量化时是同一个的比例。</p>`);}
function obsPredict(){const v=view('outlier','int4-32'), z=v.zeros;
  return verdict()+tx(`<p><b>${z===31?'All 31':`${z} of the 31`} other weights became 0.</b></p><p>The outlier set the scale, so one grid step is about 7 times a typical weight here. A weight smaller than half a step rounds to 0, and ${z===31?'every':'almost every'} other weight in this block is that small. What those weights added to the layer’s output is gone; only the outlier survives (in red in the chart).</p>`,
            `<p><b>其余 31 个权重${z===31?'全部':`里有 ${z} 个`}变成了 0。</b></p><p>离群值决定了缩放系数，所以这里一个格子大约是普通权重的 7 倍。比半个格子小的权重都舍入成 0，而这一块里${z===31?'每个':'几乎每个'}别的权重都这么小。这些权重对这一层输出的贡献没有了，只剩离群值（图里的红色）。</p>`);}
function obsBlocks(){const k=keyOf(), v=view('outlier',k);
  const so=v.segs.find(sg=>sg.hasO), sr=v.segs.filter(sg=>!sg.hasO), zr=sr.reduce((a,sg)=>a+sg.zeros,0);
  let head, why=[];
  if(k==='int4-32') head=tx(`With one scale per 32, all 31 small weights become 0.`,`每 32 个一块时，31 个小权重全部变成 0。`);
  else if(v.zeros===31) head=tx(`${sshort(k)}: all 31 small weights become 0 here too.`,`${sshort(k)}：31 个小权重也全部变成 0。`);
  else head=tx(`${sshort(k)}: only ${v.zeros} of the 31 small weights round to zero.`,`${sshort(k)}：31 个小权重里只有 ${v.zeros} 个变成 0。`);
  if(k==='int4-16') why=[tx(`In the outlier’s block of 16, the other ${so.zeros} all become 0.`,`离群值所在的那 16 个里，另外 ${so.zeros} 个都变成了 0。`),
    tx(`The other half has its own, much smaller scale: only the ${zr} smallest become 0, and the rest keep their values.`,`另一半有自己的、小得多的缩放系数，只有 ${zr} 个特别小的变成 0，其余的都留下了值。`)];
  else if(k==='nv') why=[tx(`In the outlier’s block of 16, ${so.zeros} become 0.`,`离群值所在的那 16 个里，有 ${so.zeros} 个变成 0。`),
    tx(`FP4’s grid is finer near 0 than INT4’s, so ${v16o()-so.zeros} more survive there than with INT4.`,`FP4 在 0 附近的格子比 INT4 细，所以这里比 INT4 多留下 ${v16o()-so.zeros} 个。`),
    tx(`The other half has its own scale: only the ${zr} smallest become 0.`,`另一半有自己的缩放系数，只有 ${zr} 个特别小的变成 0。`)];
  else if(k==='mx') why=[tx('MXFP4’s scale can only be a power of two (such as 1/8 or 1/16), so it cannot be exactly what the outlier needs; it takes the nearest power of two below.','MXFP4 的缩放系数只能是 2 的幂（比如 1/8、1/16），不能正好取离群值需要的那个数，只能取比它小的、最近的那个 2 的幂。'),
    tx('32 weights share that scale, and the outlier still sets it for the whole block, so the small weights nearly all become 0.','32 个权重共用这个缩放系数，离群值仍然决定这一整块，所以小权重还是几乎都变成 0。'),
    tx('FP4 can store only a few values near the top, so the outlier 0.55 is stored as 0.5.','FP4 在大数附近能存的值很稀，离群值 0.55 也只能存成 0.5。')];
  else if(k==='int4-row'||k==='int4-128') why=[tx('A bigger block shares its scale with more weights, so more of them sit next to some large weight.','块越大，共用一个缩放系数的权重越多，和某个大权重在同一块的权重也越多。')];
  return `<p><b>${head}</b></p>`+why.map(x=>`<p>${x}</p>`).join('');}
const v16o=()=>view('outlier','int4-16').segs.find(sg=>sg.hasO).zeros;
function obsCompute(){const k=p5Now()||'bf16', s=P5[k].s, kk=s.sch==='int4'?'int4-'+s.blk:s.sch, b={}, r={};
  KINDS.forEach(x=>{b[x]=stepT('bf16','bf16',x); r[x]=stepT(kk,s.kv,x);});
  const x=(a,c)=>a/c;
  if(k==='bf16') return tx(`<p><b>In BF16, decode waits on memory reads; the long prefill waits on arithmetic.</b></p><p>A decode step with one request takes ${ms(b.d1.t)} ms.</p><p>The 2,048-token prefill takes ${ms(b.pre.t)} ms.</p>`,`<p><b>BF16 时，decode 在等读显存，长的 prefill 在等计算。</b></p><p>1 个请求的 decode 步要 ${ms(b.d1.t)} 毫秒。</p><p>2,048 个 token 的 prefill 要 ${ms(b.pre.t)} 毫秒。</p>`);
  if(k==='w4') return tx(`<p><b>With one request, 4-bit weights make the decode step about ${fix(x(b.d1.t,r.d1.t),1)} times as fast; the prefill does not get faster at all.</b></p><p>That is this lab’s step model, which counts memory reads and multiplies only, not the time to turn 4-bit weights back into BF16.</p><p>Decode waits on reading the weights, and about a quarter as many bytes are read. The prefill waits on arithmetic, which is still BF16. With many requests the KV cache is read too, and it did not shrink.</p>`,
            `<p><b>1 个请求时，4 位权重让 decode 步快到约 ${fix(x(b.d1.t,r.d1.t),1)} 倍，prefill 却一点没变快。</b></p><p>这是按这个 Lab 的步耗时模型算的：它只算读显存和乘法，不算把 4 位还原成 BF16 的时间。</p><p>decode 在等读权重，要读的字节只剩四分之一左右。prefill 在等计算，而计算还是 BF16。请求多时还要读 KV cache，它没有变小。</p>`);
  if(k==='fp8') return tx(`<p><b>FP8 weights and activations make the one-request decode step and the prefill about twice as fast.</b></p><p>The prefill drops from ${ms(b.pre.t)} ms to ${ms(r.pre.t)} ms.</p><p>The 64-request decode step gets only about a quarter faster: most of its time goes to reading the KV cache, which is still BF16.</p>`,
            `<p><b>权重和激活都用 FP8 时，1 个请求的 decode 步和 prefill 都快到约 2 倍。</b></p><p>prefill 从 ${ms(b.pre.t)} 毫秒降到 ${ms(r.pre.t)} 毫秒。</p><p>64 个请求的 decode 步只快了约四分之一：它的时间大半花在读 KV cache 上，而 KV cache 还是 BF16。</p>`);
  return tx(`<p><b>An FP8 KV cache makes the 64-request decode step about a third faster.</b></p><p>It drops from ${ms(stepT('fp8','bf16','d64').t)} ms to ${ms(r.d64.t)} ms.</p><p>The one-request step barely changes, because it had little KV cache to read in the first place.</p><p>With many requests and long contexts, a step reads more KV cache than weights, so halving the KV cache matters most there.</p>`,
            `<p><b>KV cache 也用 FP8 后，64 个请求的 decode 步快了约三分之一。</b></p><p>这一步从 ${ms(stepT('fp8','bf16','d64').t)} 毫秒降到 ${ms(r.d64.t)} 毫秒。</p><p>1 个请求的步几乎没变，因为它要读的 KV cache 本来就很少。</p><p>请求多、上下文长时，一步读的 KV cache 比权重还多，所以把它减半效果最大。</p>`);}
function obsScale(){const m=S.model, P=MOD[m], b=gpus(m,'bf16'), f=gpus(m,'fp8'), n=gpus(m,'nv');
  if(m==='8b') return tx(`<p><b>${P.n}’s weights fit on one ${GPU} in BF16, FP8 and NVFP4 alike.</b></p><p>What quantization buys here is room: with FP8 weights one ${GPU} holds the KV cache of ${A.fmt.int(kvTok8('fp8','bf16'))} tokens instead of ${A.fmt.int(kvTok8('bf16','bf16'))}.</p>`,
    `<p><b>${P.n} 的权重不管用 BF16、FP8 还是 NVFP4，都放得进一张 ${GPU}。</b></p><p>量化在这里换来的是空间：FP8 权重时，一张 ${GPU} 能放 ${A.fmt.int(kvTok8('fp8','bf16'))} 个 token 的 KV cache，而不是 ${A.fmt.int(kvTok8('bf16','bf16'))} 个。</p>`);
  if(m==='70b') return tx(`<p><b>${P.n}’s weights need ${b.n} ${GPU}s in BF16, but fit on one in FP8 or NVFP4.</b></p><p>In FP8 only ${gib(f.room)} is left for the KV cache. In 4 bits about half of the GPU is free for it.</p>`,
    `<p><b>${P.n} 的权重用 BF16 要 ${b.n} 张 ${GPU}，用 FP8 或 NVFP4 一张就放得下。</b></p><p>FP8 时只剩 ${gib(f.room)} 放 KV cache。4 位时大约一半的显存都能放 KV cache。</p>`);
  return tx(`<p><b>In BF16, ${P.n}’s weights do not fit even on 8 ${GPU}s.</b></p><p>NVFP4 needs ${n.n} ${GPU}s; FP8 just fits on one machine with ${f.n}.</p><p>Meta’s own FP8 release quantizes only the MLP weights of all but the first and last layers (${cite(SRC.meta,'Meta’s FP8 loader')}), and keeps 81.3B parameters in BF16.</p><p>So with Meta’s release each of the 8 GPUs holds 56.7 GiB; with every weight in FP8, as in this chart, it would be ${gib(f.w/8)}.</p>`,
    `<p><b>${P.n} 的权重用 BF16 时 8 张 ${GPU} 也放不下。</b></p><p>NVFP4 要 ${n.n} 张，FP8 正好放得进一台 ${f.n} 卡机器。</p><p>Meta 自己发布的 FP8 版本只量化除第一层和最后一层以外的 MLP 权重（${cite(SRC.meta,'Meta 的 FP8 加载代码')}），有 81.3B 个参数留在 BF16。</p><p>所以用 Meta 的版本，8 张卡每张存 56.7 GiB；如果像图里这样全部权重都用 FP8，每张是 ${gib(f.w/8)}。</p>`);}
function remember(){
  const nv70=wGiB('70b','nv'), r70=wGiB('70b','int4-row'), dd=(+fix(nv70,1)-(+fix(r70,1))).toFixed(1);
  return tx(`<ol class="qz-rem"><li><b>Count the scales.</b> Besides its 4-bit value, each NVFP4 weight carries a share of its block’s scale: 4.5 bits in all.</li><li><b>One large weight sets its block’s scale.</b> When a block holds a weight dozens of times larger than the rest, INT4 rounds nearly all the others to 0. With smaller blocks, the part without it gets its own scale and most values survive. That is why common 4-bit formats use small blocks.</li><li><b>Weight-only 4-bit speeds up decode, not prefill.</b> Decode waits on reading the weights, so fewer bytes make it faster. Prefill waits on arithmetic, and the multiplies are still BF16. FP8 weights and activations speed up both.</li></ol>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “4 bits is 4 bits.” The scales are stored too, so each weight takes a little more.</p><p>In NVFP4, ${M70} takes ${gib(nv70)}: ${dd} GiB more than its values alone.</p><p>✗ “Quantization only saves memory.” Decode reads every weight once per step, so fewer bytes per weight make decode faster too.</p><p>✗ “Fewer bits is always fine.” In this lab, INT4 with one scale per row changes the layer’s pick for ${pc(1-evalK(SEED,'int4-row').agree)} of the inputs; NVFP4 for ${pc(1-evalK(SEED,'nv').agree)}. Before a real deployment, an eval set must confirm that the answers did not get noticeably worse.</p></div>${A.shell.toScenarios()}`,
    `<ol class="qz-rem"><li><b>要算上缩放系数。</b>NVFP4 每个权重除了 4 位的值，还分摊一份缩放系数，一共 4.5 位。</li><li><b>一个大权重决定它那一块的缩放系数。</b>一块里有一个大几十倍的权重时，用 INT4 存，其余权重几乎全部舍入成 0。块小一些，不含它的那部分有自己的缩放系数，大部分值留了下来。所以常用的 4 位格式都用小块。</li><li><b>只把权重量化到 4 位，decode 变快，prefill 不变。</b>decode 在等读权重，字节少了就快。prefill 在等计算，而乘法还是 BF16。权重和激活都用 FP8 时，decode 和 prefill 都会变快。</li></ol>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “4 位就是 4 位。”缩放系数也要存，每个权重要多占一点。</p><p>${M70} 用 NVFP4 是 ${gib(nv70)}，比只算值的 ${gib(r70)} 多出 ${dd} GiB。</p><p>✗ “量化只省显存。”decode 每一步都要把每个权重读一遍，所以每个权重的字节少了，decode 也会变快。</p><p>✗ “位数越少越好，没有代价。”在这个 Lab 里，INT4 每行一个缩放系数时，这一层有 ${pc(1-evalK(SEED,'int4-row').agree)} 的输入选得和 BF16 不一样；NVFP4 是 ${pc(1-evalK(SEED,'nv').agree)}。真实部署前，要在评测集上确认量化后的回答质量没有明显下降。</p></div>${A.shell.toScenarios()}`);}

/* =====================================================================
   scenarios, header texts, capability layer
   ===================================================================== */
const PRESETS=[
 {id:'init',t:{en:`${M8} in BF16`,zh:`${M8}，BF16`},d:{en:'The initial setup: nothing quantized',zh:'初始设置：什么都不量化'},s:{sch:'bf16',kv:'bf16',model:'8b',view:'serve'}},
 {id:'vfp8',t:{en:'FP8 weights and activations',zh:'FP8 权重和激活'},d:{en:'Weights: one scale per tensor; activations: one per token',zh:'权重每个张量一个缩放系数，激活每个 token 一个'},s:{sch:'fp8',kv:'bf16',model:'8b',view:'block'}},
 {id:'g128',t:{en:'4-bit weights in groups of 128',zh:'4 位权重，每 128 个一组'},d:{en:'INT4 as AWQ’s examples run it; activations stay BF16',zh:'AWQ 示例里用的 INT4；激活还是 BF16'},s:{sch:'int4',blk:'128',kv:'bf16',model:'8b',view:'block'}},
 {id:'nvfp4',t:{en:'NVFP4 weights',zh:'NVFP4 权重'},d:{en:'FP4 in blocks of 16 with an FP8 scale',zh:'FP4，每 16 个一块，配 FP8 缩放系数'},s:{sch:'nv',kv:'bf16',model:'8b',view:'block'}},
 {id:'kv8',t:{en:'Many long chats: FP8 KV cache',zh:'很多长对话：KV cache 用 FP8'},d:{en:`${M8}, FP8 weights and KV cache, 64 requests`,zh:`${M8}，权重和 KV cache 都用 FP8，64 个请求`},s:{sch:'fp8',kv:'fp8',model:'8b',view:'serve'}},
 {id:'405',t:{en:`${M405} on one machine`,zh:`一台机器跑 ${M405}`},d:{en:'8 H100s, FP8 weights',zh:'8 张 H100，FP8 权重'},s:{sch:'fp8',kv:'fp8',model:'405b',view:'serve'}}];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k])&&(p.s.sch!=='int4'||S.blk===p.s.blk));
function scenHTML(i){const p=PRESETS[i], k=keyOf(), e=evalK(SEED,k), b=stepT('bf16','bf16','d1'), r=stepT(k,S.kv,'d1');
  if(p.id==='init') return tx(`<p><b>Nothing is rounded: the layer’s output is the reference.</b></p><p>A one-request decode step of ${M8} takes ${ms(b.t)} ms. Pick another scenario to see what quantization changes.</p>`,`<p><b>什么都没舍入：这一层的输出就是参照。</b></p><p>${M8} 1 个请求的 decode 步要 ${ms(b.t)} 毫秒。选别的场景，看看量化改变了什么。</p>`);
  if(p.id==='vfp8') return tx(`<p><b>FP8 with one scale per tensor for the weights and one per token for the activations: output error ${pc(e.oerr)}, same pick on ${pc(e.agree)} of the inputs.</b></p><p>FP8 is a floating-point format: the closer a value is to 0, the denser the values it can store. So even with a large outlier in the tensor, the small weights still have a fine grid near them and do not all become 0.</p><p>This combination is the lab’s own choice. vLLM’s docs say FP8 quantization gives “a 2x reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy”. vLLM’s online mode (<code>--quantization fp8</code>) gives the activations one scale per tensor too, and the same docs say “latency improvements are limited in this mode” (${cite(SRC.vfp8,'vLLM’s FP8 docs')}).</p>`,
    `<p><b>权重每个张量一个缩放系数、激活每个 token 一个的 FP8：输出误差 ${pc(e.oerr)}，${pc(e.agree)} 的输入选得和 BF16 一样。</b></p><p>FP8 是浮点格式：数越接近 0，能存的值排得越密。所以即使张量里有很大的离群值，小权重附近的格子照样很细，不会都变成 0。</p><p>这种组合是这个 Lab 自己选的。vLLM 的文档说，把模型量化成 FP8，显存需求减半，吞吐最多提高到 1.6 倍，对准确率的影响很小。vLLM 的在线模式（<code>--quantization fp8</code>）给激活也只用每个张量一个缩放系数，同一份文档说这种模式的延迟改善有限（${cite(SRC.vfp8,'vLLM 的 FP8 文档')}）。</p>`);
  if(p.id==='g128') return tx(`<p><b>Output error ${pc(e.oerr)}; the layer keeps its pick on ${pc(e.agree)} of the inputs.</b></p><p>This lab plants a few very large weights (outliers) in the layer on purpose; with 128 weights per group, about 2 in 5 groups contain one.</p><p>AWQ and GPTQ are post-training quantization methods: they first run the model on a small batch of text (calibration data), then use it to choose the scales (AWQ) or correct the rounding (GPTQ). This lab leaves that step out (${cite(SRC.pAwq,'AWQ')}, ${cite(SRC.pGptq,'GPTQ')}).</p>`,
    `<p><b>INT4 每 128 个一组：输出误差 ${pc(e.oerr)}，${pc(e.agree)} 的输入选得和 BF16 一样。</b></p><p>这个 Lab 故意在这一层里放了一些特别大的权重（离群值）。每 128 个一组时，大约四成的组里有一个。</p><p>AWQ 和 GPTQ 是训练后量化方法：先用一小批文本（校准数据）跑一遍模型，再据此挑缩放系数（AWQ）或修正舍入（GPTQ）。这个 Lab 没有做这一步（${cite(SRC.pAwq,'AWQ 论文')}、${cite(SRC.pGptq,'GPTQ 论文')}）。</p>`);
  if(p.id==='nvfp4') return tx(`<p><b>NVFP4: output error ${pc(e.oerr)}, same pick on ${pc(e.agree)} of the inputs.</b></p><p>With blocks of 16, an outlier affects at most the 15 others in its block.</p><p>An ${GPU} has no FP4 arithmetic, so here it is weight-only: a decode step with one request takes ${ms(r.t)} ms.</p>`,
    `<p><b>NVFP4：输出误差 ${pc(e.oerr)}，${pc(e.agree)} 的输入选得和 BF16 一样。</b></p><p>每 16 个一块，一个离群值最多影响同一块的另外 15 个。</p><p>${GPU} 没有 FP4 的计算单元，所以这里只量化权重：1 个请求的 decode 步要 ${ms(r.t)} 毫秒。</p>`);
  if(p.id==='kv8') {const a=stepT('fp8','bf16','d64'), c=stepT('fp8','fp8','d64'); return tx(`<p><b>With weights and activations already in FP8, an FP8 KV cache as well makes the decode step of 64 requests with 2,048 tokens each about a third faster.</b></p><p>It drops from ${ms(a.t)} ms to ${ms(c.t)} ms.</p><p>One ${GPU} also holds twice as many KV cache tokens. vLLM turns it on with <code>kv_cache_dtype="fp8"</code> (${cite(SRC.vkv,'vLLM’s KV cache docs')}).</p>`,
    `<p><b>权重和激活已经是 FP8 时，再把 KV cache 也改成 FP8，64 个请求、每个 2,048 个 token 的 decode 步快了约三分之一。</b></p><p>这一步从 ${ms(a.t)} 毫秒降到 ${ms(c.t)} 毫秒。</p><p>一张 ${GPU} 能放的 KV cache token 数也翻倍。vLLM 用 <code>kv_cache_dtype="fp8"</code> 打开它（${cite(SRC.vkv,'vLLM 的 KV cache 量化文档')}）。</p>`);}
  const g=gpus('405b','fp8'), gb=gpus('405b','bf16');
  return tx(`<p><b>In FP8, ${M405}’s weights fit on one machine with 8 ${GPU}s.</b></p><p>The chart assumes every weight is FP8: ${gib(g.w)} in all, ${gib(g.w/8)} on each GPU.</p><p>Each GPU has ${gib(g.room)} left for the KV cache.</p><p>In BF16 the weights take ${gib(gb.w)}, more than 8 GPUs hold.</p><p>Meta’s actual FP8 release keeps 81.3B parameters in BF16; with it, each GPU holds 56.7 GiB (${cite(SRC.meta,'Meta’s FP8 loader')}).</p>`,
    `<p><b>FP8 时，${M405} 的权重放得进一台 8 卡机器。</b></p><p>图里假设全部权重都用 FP8：一共 ${gib(g.w)}，分到 8 张卡上每张 ${gib(g.w/8)}。</p><p>每张卡还剩 ${gib(g.room)} 放 KV cache。</p><p>BF16 时权重要 ${gib(gb.w)}，8 张卡放不下。</p><p>Meta 真正发布的 FP8 版本有 81.3B 个参数留在 BF16，用它的话每张卡存 56.7 GiB（${cite(SRC.meta,'Meta 的 FP8 加载代码')}）。</p>`);}
const DESCS=()=>({
 guided:tx('Seven parts: why fewer bits speed up decode, how a block of weights is rounded, what one large weight does, smaller blocks, weights only or activations too, and real models on H100s.','七个部分，依次讲为什么位数少了 decode 会变快，一块权重怎么舍入，一个很大的权重会怎样，更小的块，只量化权重还是连激活一起，以及真实的模型放上 H100。'),
 sandbox:tx('Pick a common scheme and see what it does to the small layer and to the step times.','选一种常见的方案，看看它对小模型这一层和每一步的耗时有什么影响。'),
 challenge:tx(`Serve ${M70} on one H100 for 16 users with 8K tokens each, while the small layer keeps its pick.`,`用一张 H100 给 16 个用户跑 ${M70}，每人 8K 个 token，而且小模型的那一层还要选得和 BF16 一样。`)});
const OUTCOME=()=>tx('say what a quantization scheme loses and saves, count its bits with the scales, and pick weight and KV formats that fit a model on fewer GPUs.','说清一种量化方案丢了什么、省了什么，算出带缩放系数的位数，并选出让模型用更少 GPU 装下的权重和 KV 格式。');
const FIDPOP=()=>tx(
 `<p><b>Run in your browser:</b> a small linear layer (256 inputs, 128 outputs) generated with a seed, rounded with the real formats (round to nearest, ties to even; values above a format’s largest are clipped to it) and run on 1,000 seeded inputs. Its weights have heavy tails (Student-t with 6 degrees of freedom), and about 1 weight in 256 is planted 20 to 60 times larger: this lab’s assumption; how many large weights a real matrix has depends on the model. Four of the 256 input channels are 20 times larger, as LLM.int8() and SmoothQuant describe for activations.</p><p><b>Formula:</b> memory of ${M8}, ${M70} and ${M405} (every parameter in the format, scales counted; real recipes often keep the embedding table and the LM head in BF16, and vLLM’s FP8 leaves <code>lm_head</code> out). Step times of ${M8} on one H100 with the step model of the serving labs; linear layers run at the FP8 rate only when weights and activations are FP8, attention always at the BF16 rate. The GPUs’ communication under tensor parallelism is left out.</p><p><b>Simplified:</b> INT4 is symmetric (−7 to 7) with an FP16 scale; GPTQ and AWQ add calibration, which this lab leaves out. “Same pick as BF16” stands in for quality; real models are checked on eval sets.</p><p><b>Sources:</b> ${cite(SRC.nf,'Transformer Engine, NVFP4')}; ${cite(SRC.mx,'microxcaling')}; ${cite(SRC.vfp8,'vLLM, FP8')}; ${cite(SRC.vkv,'vLLM, FP8 KV cache')}; ${cite(SRC.meta,'Meta llama-models, FP8 loader')}; ${cite(SRC.ds,'DeepSeek‑V3 weights')}; ${cite(SRC.mo,'NVIDIA Model Optimizer')}; ${cite(SRC.h100,'NVIDIA H100')}.</p>`,
 `<p><b>在浏览器里跑的：</b>小模型的一层线性层（256 个输入、128 个输出），按种子生成，用真实的格式舍入（舍入到最近，一样近时取偶数；超过格式最大值的截到最大值），在 1,000 个按种子生成的输入上运行。权重的尾巴较重（自由度为 6 的 Student-t 分布），另外大约每 256 个权重里有 1 个被放大 20 到 60 倍：这是这个 Lab 的假设；真实矩阵里有多少大权重，取决于模型。256 个输入通道里有 4 个大 20 倍，这是 LLM.int8() 和 SmoothQuant 对激活的描述。</p><p><b>按公式算的：</b>${M8}、${M70}、${M405} 的显存（每个参数都按这种格式算，算上缩放系数；实际的做法常把 embedding 表和 LM head 留在 BF16，vLLM 的 FP8 不量化 <code>lm_head</code>）。${M8} 在一张 H100 上每一步的耗时，用推理服务几个 Lab 的步耗时模型；只有权重和激活都是 FP8 时，线性层才按 FP8 的速度算，attention 一直按 BF16 的速度算。张量并行时卡和卡之间的通信没有算。</p><p><b>简化：</b>INT4 用对称形式（−7 到 7），缩放系数用 FP16；GPTQ、AWQ 还会用校准数据，这个 Lab 没有做。“和 BF16 选的一样”代替回答质量；真实模型要在评测集上检查。</p><p><b>来源：</b>${cite(SRC.nf,'Transformer Engine，NVFP4')}；${cite(SRC.mx,'microxcaling')}；${cite(SRC.vfp8,'vLLM，FP8')}；${cite(SRC.vkv,'vLLM，FP8 KV cache')}；${cite(SRC.meta,'Meta llama-models，FP8 loader')}；${cite(SRC.ds,'DeepSeek‑V3 权重')}；${cite(SRC.mo,'NVIDIA Model Optimizer')}；${cite(SRC.h100,'NVIDIA H100')}。</p>`);
const PRIPOP=()=>tx(`<p>BF16 stores a number in 16 bits, FP8 E4M3 in 8 and FP4 E2M1 in 4. Fewer bits mean fewer values to round to: FP4 has only 0, 0.5, 1, 1.5, 2, 3, 4 and 6 (and their negatives). The lab ${labLink('number-formats')} shows these grids.</p><p>A decode step reads every weight once, so with few requests it waits on memory reads, not arithmetic (the lab ${labLink('prefill-decode')}).</p>`,
 `<p>BF16 用 16 位存一个值，FP8 E4M3 用 8 位，FP4 E2M1 用 4 位。位数越少，能舍入到的值越少：FP4 只有 0、0.5、1、1.5、2、3、4、6（和它们的负数）。${labLink('number-formats')} 这个 Lab 画了这些格子。</p><p>decode 每一步都要把每个权重读一遍，所以请求少时它在等读显存，而不是等计算（见 ${labLink('prefill-decode')}）。</p>`);
const EXITS=()=>{
  const g=evalK(SEED,'int4-128'), r=evalK(SEED,'int4-row');
  const mf=wGiB('405b','mx'), T2U=8192*MOD['8b'].kv*KVB.fp8/GiB, T2R=UGPU-wGiB('8b','int4-128'), T2a=+fix(T2R,1), T2b=+fix(T2U,2);
  return {core:()=>chSubmitted, explain:[
  {q:tx('Why does INT4 with one scale per row lose more than INT4 with one scale per 128 weights?','为什么 INT4 每行一个缩放系数，比每 128 个一个丢得多？'),
   o:[tx('A row holds more weights, so it more often holds a large one; that weight sets the scale for all of them, and the small ones round to 0.','一行里的权重更多，更容易有一个大权重；它决定全行的缩放系数，小权重就舍入成 0。'),
      tx('A row has more weights, so the rounding errors add up over more numbers.','一行的权重更多，舍入误差在更多的数上累加。'),
      tx('A scale per row is stored in fewer bits than a scale per 128 weights.','每行一个的缩放系数，存的位数比每 128 个一个的少。')],a:0,
   n:[null,tx('Each weight is rounded on its own; its error does not depend on how many share the scale, only on how big that scale is.','每个权重是单独舍入的；它的误差不取决于多少个值共用缩放系数，只取决于缩放系数有多大。'),tx('Both scales are 16 bits here. What differs is how many weights must fit under the one largest weight.','这里两种缩放系数都是 16 位。不同的是有多少个权重要和那个最大的权重共用一个缩放系数。')],
   why:tx(`In this lab the layer’s output error is ${pc(r.oerr)} with one scale per row and ${pc(g.oerr)} with the smaller blocks.`,`在这个 Lab 里，每行一个缩放系数时输出误差是 ${pc(r.oerr)}，块更小时是 ${pc(g.oerr)}。`)},
  {q:tx('Why do 4-bit weights stored as INT4 speed up decode but not a long prefill on an H100?','为什么在 H100 上，INT4 存的 4 位权重让 decode 变快，却不让长的 prefill 变快？'),
   o:[tx('The prefill reads the weights once per token, so it reads them 2,048 times and the saving is lost.','prefill 每个 token 读一次权重，要读 2,048 次，省下的就没了。'),
      tx('Decode waits on reading the weights, and there are fewer bytes; the prefill waits on arithmetic, which still runs in BF16.','decode 在等读权重，字节变少了；prefill 在等计算，而计算还是 BF16。'),
      tx('Turning 4-bit weights back into BF16 takes longer than the prefill itself.','把 4 位权重还原成 BF16，比 prefill 本身还慢。')],a:1,
   n:[tx('A prefill step reads the weights once for all its tokens, like a decode step. It is slow because of the arithmetic for 2,048 tokens.','prefill 的一步和 decode 一样，所有 token 只读一次权重。它慢是因为要为 2,048 个 token 做计算。'),null,tx('The lab counts no time for that; real kernels spend a little, but the prefill is limited by the BF16 multiplies.','这个 Lab 没有算这部分时间；真实的 kernel 会花一点，但 prefill 受限于 BF16 乘法。')],
   why:tx('Only FP8 weights and activations put the multiplies on the faster FP8 units.','只有权重和激活都是 FP8，乘法才会在更快的 FP8 单元上做。')}],
 transfer:[
  {q:tx(`${M405} has 405.9B parameters, stored in MXFP4.<br>Counting the scales, about how much memory do its weights take?`,`${M405} 有 405.9B 个参数，用 MXFP4 存。<br>算上缩放系数，它的权重大约占多少显存？`),
   o:[tx('About 189 GiB','约 189 GiB'),tx('About 201 GiB','约 201 GiB'),tx('About 378 GiB','约 378 GiB')],a:1,
   why:tx(`MXFP4 shares one 8-bit scale among 32 values, so each weight takes ${M('4 + 8/32 = 4.25')} bits, and the weights take ${M(`405.9 \\times 4.25 / 8 = ${fix(MOD['405b'].P*4.25/8/1e9,1)}`)} GB, which is ${gib(mf)}. Counting only the 4-bit values gives ${gib(wGiB('405b','int4-row'))}.`,`MXFP4 每 32 个值共用一个 8 位缩放系数，每个权重占 ${M('4 + 8/32 = 4.25')} 位，权重一共 ${M(`405.9 \\times 4.25 / 8 = ${fix(MOD['405b'].P*4.25/8/1e9,1)}`)} GB，也就是 ${gib(mf)}。只按 4 位算是 ${gib(wGiB('405b','int4-row'))}。`)},
  {q:tx(`One ${GPU} has 70 GiB usable; ${M8}’s weights in INT4 take ${gib(wGiB('8b','int4-128'))} of it.<br>Each user’s KV cache in FP8 takes ${fix(T2U,2)} GiB. About how many users fit?`,`一张 ${GPU} 能用 70 GiB，${M8} 的权重用 INT4 存时占其中 ${gib(wGiB('8b','int4-128'))}。<br>每个用户的 KV cache 用 FP8 存，是 ${fix(T2U,2)} GiB。大约能放几个用户？`),
   o:[tx('About 66','约 66 个'),tx('About 132','约 132 个'),tx('About 264','约 264 个')],a:1,
   why:tx(`After the weights, ${M(`70 - ${fix(wGiB('8b','int4-128'),1)} = ${T2a}`)} GiB is left for the KV cache: ${M(`${T2a} / ${fix(T2U,2)} = ${fix(T2a/T2b,1)}`)}, about ${Math.floor(T2a/T2b)} users. With a BF16 KV cache, half as many fit.`,`权重放下后，${M(`70 - ${fix(wGiB('8b','int4-128'),1)} = ${T2a}`)} GiB 留给 KV cache：${M(`${T2a} / ${fix(T2U,2)} = ${fix(T2a/T2b,1)}`)}，约 ${Math.floor(T2a/T2b)} 个用户。KV cache 用 BF16 时只能放一半。`)}],
 practice:[{id:'matmul-roofline',d:tx('The “Lower precision” paragraph: why FP8 arithmetic alone helps only when the bytes shrink too.','“Lower precision”那一段：为什么只有字节也变少时，FP8 计算才真正有用。')},
           {id:'inference-serving',d:tx('A GPU’s memory split between one replica’s weights and the KV cache: what a smaller weight format frees.','一张 GPU 的显存分给一份模型的权重和 KV cache：更小的权重格式能腾出多少。')}]};};

/* =====================================================================
   challenge
   ===================================================================== */
function chResultHTML(){
  if(!chSubmitted) return '';
  const e=chEval(chW,chKV), all=chAll(), pass=all.filter(x=>x.ok);
  const lab=x=>tx(`weights ${sshort(x.k)}, KV cache ${x.kv==='fp8'?'FP8':'BF16'}`,`权重 ${sshort(x.k)}，KV cache ${x.kv==='fp8'?'FP8':'BF16'}`);
  let h;
  if(e.ok) h=tx(`<p class="okline"><b>Goal met: it fits, and the other layer keeps its pick.</b></p><p>The weights and the KV cache take ${gib(e.need)} of the 70 GiB.</p><p>The other layer picks like BF16 on ${pc(e.agree)} of the inputs.${pass.length===1?' It is the only setup that meets both conditions.':''}</p>`,`<p class="okline"><b>达标：放得下，另一层也选得一样。</b></p><p>权重和 KV cache 一共 ${gib(e.need)}，没有超过 70 GiB。</p><p>另一层 ${pc(e.agree)} 的输入选得和 BF16 一样。${pass.length===1?'这是唯一同时满足两个条件的设置。':''}</p>`);
  else {const why=!e.fits?tx(`it needs ${gib(e.need)}, more than the 70 GiB one ${GPU} has for weights and KV cache`,`它要 ${gib(e.need)}，超过一张 ${GPU} 放权重和 KV cache 的 70 GiB`):tx(`it fits (${gib(e.need)}), but the layer picks like BF16 on only ${pc(e.agree)} of the inputs`,`它放得下（${gib(e.need)}），但这一层只有 ${pc(e.agree)} 的输入选得和 BF16 一样`);
    h=tx(`<p class="badline"><b>Not met: ${why}.</b></p>`,`<p class="badline"><b>没达标：${why}。</b></p>`);
    if(!chReveal) h+=tx(`<p>Hint: both the weights and the KV cache must shrink, and the weight format must keep an outlier from wiping out many neighbours.</p><p><button type="button" class="linkbtn" data-act="reveal">Show the answer</button></p>`,`<p>提示：权重和 KV cache 都要变小，而且权重格式要让一个离群值影响不到太多邻居。</p><p><button type="button" class="linkbtn" data-act="reveal">直接看答案</button></p>`);}
  if(e.ok||chReveal){
    const rows=all.map(x=>({label:lab(x),value:x.need,text:tx(`${gib(x.need)}, same pick ${pc(x.agree)}`,`${gib(x.need)}，选得一样 ${pc(x.agree)}`),ok:x.ok,me:x.k===chW&&x.kv===chKV,tags:x.k==='bf16'&&x.kv==='bf16'?[{text:tx('initial setup','初始设置')}]:[]}));
    h+=A.ui.cmpChart({title:tx(`Memory on one ${GPU} for ${M70} and 16 users × 8K tokens`,`${M70} 加 16 个用户 × 8K 个 token，一张 ${GPU} 上要的显存`),aria:tx('Memory per setup','每种设置要的显存'),rows,goal:UGPU,goalLabel:tx('70 GiB usable','可用 70 GiB'),max:180,brief:true,okText:tx('fits and keeps the pick','放得下，选得也一样'),noText:tx('misses a condition','差一个条件')});
    const best=pass[0];
    if(best) h+=tx(`<p><b>Answer: ${lab(best)}.</b></p><p>In FP8 the KV cache of the 16 users takes ${gib(best.kvg)}, twice that in BF16.</p><p>So the weights must be 4-bit to fit next to it.</p><p>Of those formats, only NVFP4’s blocks of 16 keep the pick at ${pc(best.agree)}.</p><p>MXFP4 and INT4 per 128 fit too, but their larger blocks let one outlier zero out more weights, and they stay below 90%.</p><p class="small muted">Caveat: the 90% bar is this lab’s stand-in. On an H100, NVFP4 runs weight-only (no FP4 arithmetic), and a real deployment would check quality on an eval set.</p>`,
                     `<p><b>答案：${lab(best)}。</b></p><p>16 个用户的 KV cache 用 FP8 是 ${gib(best.kvg)}，用 BF16 要翻倍。</p><p>所以权重必须用 4 位格式才放得下。</p><p>这些格式里，只有 NVFP4 每 16 个一块，这一层 ${pc(best.agree)} 的输入还选得一样。</p><p>MXFP4 和 INT4 每 128 个一块也放得下，但块更大，一个离群值会让更多权重变成 0，没到 90%。</p><p class="small muted">注意：90% 这个门槛是这个 Lab 的替代指标。在 H100 上，NVFP4 只量化权重（没有 FP4 计算单元），真实部署还要在评测集上检查质量。</p>`);
  }
  return h;
}
function challengeHTML(){
  const submit=`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Serve ${M70} on one ${GPU} for 16 users`,`用一张 ${GPU} 给 16 个用户跑 ${M70}`)}</h3>`+
   tx(`<p>${M70} must serve 16 users at once on one ${GPU}, each with 8,192 tokens of context.</p><p>Its weights and the KV cache of all the users must fit in the GPU’s 70 GiB.</p><p>Quality must hold too. This lab cannot actually run the 70B, so another layer of the small model stands in: stored in the weight format you pick, it must pick the same output as BF16 on at least 90% of 1,000 inputs.</p><p>The other layer’s large weights sit elsewhere and differ in size, so the guided parts’ percentages are only a guide. Look for the format in which one large weight affects the fewest neighbours.</p>`,
      `<p>一张 ${GPU} 要同时给 16 个用户跑 ${M70}，每人 8,192 个 token 的上下文。</p><p>权重和所有用户的 KV cache 都要放进这张卡的 70 GiB。</p><p>质量也要守住。这个 Lab 没法真的跑 70B，所以用小模型的另一层代替：用你选的权重格式存这一层，1,000 个输入里至少 90% 要选得和 BF16 一样。</p><p>另一层的大权重在别的位置，大小也不同，所以引导部分那一层的百分比只能当参考。要看哪种格式让一个大权重影响的邻居最少。</p>`)+
   `<p class="small qz-lead">${tx('Pick a weight format and a KV cache format, then submit.','选一种权重格式和一种 KV cache 格式，再提交。')}</p>`+
   `<div class="qz-lv" id="qz-ic-chw">`+A.ui.levers({aria:tx('Weight format, pick one','权重格式，选一个'),value:chW,options:CHK.map(v=>({v,title:sname(v),desc:'',result:`${bitsT(RBITS[v])} ${tx('bits','位')}`}))})+`</div>`+
   A.ui.seg({id:'qz-ic-chkv',key:'chkv',label:tx('KV cache','KV cache'),options:[['bf16','BF16'],['fp8','FP8']],value:chKV,inline:true})+
   `<div class="goal mono qz-goal" id="qz-ch-g1"><span>${tx('Memory on the GPU','这张卡上的显存')}</span><span><b id="qz-ch-mem"></b>${tx(' of 70 GiB',' / 70 GiB')}</span></div>`+
   `<div class="qz-ch-sub">${submit}<p class="small muted">${tx('Submitting runs the other layer with your weight format and says whether both conditions hold.','提交后会用你选的权重格式跑另一层，并告诉你两个条件是否都满足。')}</p></div>`+
   `<div id="qz-ch-res" class="qz-ch-res" aria-live="polite"></div></div>`;
}
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see what its scheme loses in the small layer and what it saves per step.','选一个场景，看看它的方案在小模型的这一层上丢了什么，每一步又省了什么。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="qz-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change the scheme, the KV cache, the model or the picture under ${J_('controls','Settings')}.`,`想换方案、KV cache、模型或图，在“${J_('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}

/* =====================================================================
   layout
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Quantization','量化'),title:l?L(l.q):tx('BF16 → FP8 → FP4: what do you actually lose?','BF16 → FP8 → FP4，到底丢了什么？'),
    minutes:15,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: number formats','前置知识：数值格式'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage qz-stage" id="qz-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="qz-st-title"></div><div class="meter-sub" id="qz-st-sub"></div></div><div class="stage-actions"><span class="status" id="qz-status"></span></div></div>`+
   `<div class="stage-graph graph qz-graph" id="qz-graph">`+
    `<div class="qz-phone" id="qz-phone"></div>`+
    `<div class="qz-blockv" id="qz-blockv"><div class="qz-cap" id="qz-b-cap"></div><div class="qz-tp" id="qz-tp"></div><svg class="qz-svg" id="qz-svg" role="img" aria-label=""></svg><div class="legend-row qz-legend" id="qz-b-leg"></div><div class="qz-note small" id="qz-b-note"></div></div>`+
    `<div class="qz-chart" id="qz-chart"></div>`+
    `<div class="qz-serve" id="qz-serve"><div class="qz-steps" id="qz-steps"></div><div class="qz-mem" id="qz-mem"></div></div>`+
   `</div>`+
   `<div class="metrics qz-metrics" id="qz-metrics"></div>`+
   `<details class="math" id="qz-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="qz-mathbody"></div></details></div>`;
}
const P_STEP=P_LAYERS+P_HEAD;
const GIBT=()=>A.terms&&A.terms.gib?A.terms.gib():tx('GiB = 2³⁰ bytes','GiB = 2³⁰ 字节');
/* decode with one request when the LM head stays in BF16: how much slower than the lab's step (S10) */
const headSlow=k=>{const r=stepT(k,'bf16','d1'), tm=(RBITS[k]/8*P_LAYERS+2*P_HEAD+MOD['8b'].kv*CTX)/BS; return Math.max(r.tc,tm)/r.t-1;};
const headRange=()=>{const v=['fp8','int8','int4-row','int4-128','int4-32','int4-16','mx','nv'].map(headSlow); return [Math.round(Math.min(...v)*100),Math.round(Math.max(...v)*100)];};
const ASSUME=()=>{const [lo,hi]=headRange(); return [
 [tx('Small layer','小模型的一层'),tx('256 inputs, 128 outputs; weights Student-t (6 degrees of freedom) × 0.02; about 1 in 256 weights 20 to 60 times larger; 4 input channels 20 times larger; seed 1 (the challenge: seed 2)','256 个输入、128 个输出；权重服从自由度为 6 的 Student-t 分布再乘 0.02；大约每 256 个权重里有 1 个大 20 到 60 倍；4 个输入通道大 20 倍；种子 1（挑战题用种子 2）')],
 [tx('INT4 and INT8','INT4 和 INT8'),tx(`symmetric, −7 to 7 and −127 to 127, with an FP16 scale ${M('s = \\max |w| \\,/\\, 7')} (INT8: ${M('\\max |w| \\,/\\, 127')}), where ${M('\\max |w|')} is the block’s largest absolute value`,`对称：−7 到 7、−127 到 127；缩放系数用 FP16，${M('s = \\max |w| \\,/\\, 7')}（INT8 是 ${M('\\max |w| \\,/\\, 127')}），其中 ${M('\\max |w|')} 是这块绝对值最大的数`)],
 [tx('GPU memory','显存'),tx(`counted in ${DEF('GiB',GIBT())}; per H100 ${M('80 \\times 0.9 - 2 = 70')} GiB for the weights and the KV cache, as in the serving labs`,`按 ${DEF('GiB',GIBT())} 计；每张 H100 有 ${M('80 \\times 0.9 - 2 = 70')} GiB 放权重和 KV cache，和推理服务的几个 Lab 一样`)],
 [tx('Step time','每一步的耗时'),tx(`the longer of arithmetic at 50% of the dense peak and memory reads at 60% of 3.35 TB/s; decode requests have 2,048 tokens of context. The LM head is counted in the chosen format too; real setups often keep it in BF16, which makes decode ${lo}%–${hi}% slower here, with the same conclusions`,`计算按稠密峰值的 50%、读显存按 3.35 TB/s 的 60%，取较长的那个；decode 请求都有 2,048 个 token 的上下文。LM head 也按所选格式算；实际常把它留在 BF16，这里的 decode 会慢 ${lo}%–${hi}%，结论不变`)]];};
function controlsHTML(){
  return `<aside class="panel controls" id="qz-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="qz-lock" hidden></p>`+
   `<div class="lockable" id="qz-ctl-sch-wrap">`+A.ui.seg({id:'qz-ctl-sch',key:'sch',label:tx('Weight format','权重格式'),options:[['bf16','BF16'],['fp8','FP8'],['int8','INT8'],['int4','INT4'],['mx','MXFP4'],['nv','NVFP4']],value:S.sch,note:tx('FP8 rounds the activations too; the others are weight-only.','FP8 连激活也舍入；其余几种只量化权重。')})+`</div>`+
   `<div class="lockable" id="qz-ctl-blk-wrap">`+A.ui.seg({id:'qz-ctl-blk',key:'blk',label:tx('INT4: weights per scale','INT4：每个缩放系数管几个权重'),options:[['row',tx('row (256)','整行（256）')],['128','128'],['32','32'],['16','16']],value:S.blk,note:tx('Each scale is 16 bits, so smaller blocks cost more bits per weight.','每个缩放系数 16 位，块越小，每个权重分摊的位数越多。')})+`</div>`+
   `<div class="lockable" id="qz-ctl-kv-wrap">`+A.ui.seg({id:'qz-ctl-kv',key:'kv',label:tx('KV cache','KV cache'),options:[['bf16','BF16'],['fp8','FP8']],value:S.kv,note:tx('FP8 halves the bytes per token: 64 KiB for the 8B.','FP8 让每个 token 的字节减半：8B 是 64 KiB。')})+`</div>`+
   `<div class="lockable" id="qz-ctl-model-wrap">`+A.ui.seg({id:'qz-ctl-model',key:'model',label:tx('Model (memory chart)','模型（显存图）'),options:SCN.map(([v,n])=>[v,n.replace('Llama‑3.1‑','')]),value:S.model})+`</div>`+
   `<div class="lockable" id="qz-ctl-view-wrap">`+A.ui.seg({id:'qz-ctl-view',key:'view',label:tx('Picture','图'),options:[['block',tx('a block of weights','一块权重')],['serve',tx('step time and memory','耗时和显存')]],value:S.view})+`</div>`+
   `<div class="qz-grp"><div class="qz-gh">${tx('Values this lab assumes','本 Lab 假设的值')}</div><ul class="qz-asm">${ASSUME().map(([k,v])=>`<li><b>${esc(k)}</b> ${v}</li>`).join('')}</ul></div>`+
   `<div class="lockable" id="qz-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

def int4_blocks(w, block=128):
    """Symmetric INT4, one FP16 scale per block of a row."""
    b = w.reshape(w.shape[0], -1, block)
    s = (np.abs(b).max(-1, keepdims=True) / 7).astype(np.float16)
    s = s.astype(np.float64)
    q = np.clip(np.round(b / s), -7, 7)       # ties to even
    return (q * s).reshape(w.shape)          # multiply back

def bits_per_weight(bits, scale_bits, block):
    return bits + scale_bits / block

rng = np.random.default_rng(0)
w = rng.standard_t(6, size=(128, 256)) * 0.02
w[3, 100] = 0.55                             # one outlier
x = rng.standard_normal((1000, 256))
y = x @ w.T
for block in (256, 128, 32, 16):
    yq = x @ int4_blocks(w, block).T
    err = np.linalg.norm(yq - y) / np.linalg.norm(y)
    same = (yq.argmax(1) == y.argmax(1)).mean()
    print(block, bits_per_weight(4, 16, block),
          round(err, 3), same)`;
const CODE_OUT=`# 256 4.0625 0.189 0.768   one scale per row
# 128 4.125 0.163 0.801
# 32 4.5 0.118 0.862
# 16 5.0 0.098 0.883       smaller blocks: less error, more bits`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="qz-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="qz-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('INT4 with one FP16 scale per block, its bits per weight, and what it does to a layer’s output.','INT4 每块一个 FP16 缩放系数、它每个权重占的位数，以及它对一层输出的影响。')}</p><pre class="code" id="qz-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="qz-prod">`+
   tx(`<li>vLLM: <code>--quantization fp8</code> quantizes every linear layer but <code>lm_head</code> to FP8 E4M3 with a per-tensor scale and computes a dynamic per-tensor scale for the activations (${cite(SRC.vfp8,'docs/features/quantization/fp8.md')}). <code>kv_cache_dtype="fp8"</code> stores the KV cache in FP8 (${cite(SRC.vkv,'quantized_kvcache.md')}).</li>`+
      `<li>Meta’s FP8 405B: ${cite(SRC.meta,'models/llama3/quantization/loader.py')} converts the MLP weights (w1, w3, w2) of all but the first and last layers; ${cite(SRC.metaq,'models/quantize_impls.py')} quantizes activations per row at run time and multiplies with <code>f8f8bf16_rowwise</code>.</li>`+
      `<li>DeepSeek‑V3 ships FP8 E4M3 weights with one scale per 128 × 128 block (${cite(SRC.ds,'README_WEIGHTS.md')}).</li>`+
      `<li>NVIDIA’s Model Optimizer rates INT4 weight-only AWQ “High” for small-batch and “Low” for large-batch performance, and FP8 “Very Low” in accuracy loss (${cite(SRC.mo,'_choosing_quant_methods.rst')}). NVFP4’s two-level scales are in ${cite(SRC.nf,'Transformer Engine’s NVFP4 docs')}.</li>`,
      `<li>vLLM：<code>--quantization fp8</code> 把除 <code>lm_head</code> 以外的每个线性层量化成 FP8 E4M3，每个张量一个缩放系数，激活在运行时算出每个张量一个缩放系数（${cite(SRC.vfp8,'vLLM 的 FP8 文档')}）。<code>kv_cache_dtype="fp8"</code> 让 KV cache 用 FP8 存（${cite(SRC.vkv,'vLLM 的 KV cache 量化文档')}）。</li>`+
      `<li>Meta 的 FP8 405B：${cite(SRC.meta,'models/llama3/quantization/loader.py')} 只转换除第一层和最后一层以外的 MLP 权重（w1、w3、w2）；${cite(SRC.metaq,'models/quantize_impls.py')} 在运行时按行量化激活，再用 <code>f8f8bf16_rowwise</code> 相乘。</li>`+
      `<li>DeepSeek‑V3 发布的是 FP8 E4M3 权重，每 128 × 128 一块配一个缩放系数（见它的${cite(SRC.ds,'权重说明')}）。</li>`+
      `<li>NVIDIA Model Optimizer 的${cite(SRC.mo,'选型指南')}把只量化权重的 INT4 AWQ 评为：小 batch 性能高，大 batch 性能低；FP8 的准确率损失“很低”。NVFP4 的两级缩放系数见 ${cite(SRC.nf,'Transformer Engine 的 NVFP4 文档')}。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.pInt8,'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale')}<span>${tx('Dettmers et al., 2022, arXiv 2208.07339','Dettmers 等，2022，arXiv 2208.07339')}</span></li>`+
   `<li>${cite(SRC.pSq,'SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models')}<span>${tx('Xiao et al., 2022, arXiv 2211.10438','Xiao 等，2022，arXiv 2211.10438')}</span></li>`+
   `<li>${cite(SRC.pAwq,'AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration')}<span>${tx('Lin et al., 2023, arXiv 2306.00978','Lin 等，2023，arXiv 2306.00978')}</span></li>`+
   `<li>${cite(SRC.pGptq,'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers')}<span>${tx('Frantar et al., 2022, arXiv 2210.17323','Frantar 等，2022，arXiv 2210.17323')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="qz-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="qz-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="qz-app m2" id="qz-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="qz-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
const WHY={round:{en:'Watch the three steps play to continue, or press Show me.',zh:'看完三步演示才能继续，或者点“演示给我看”。'},predict:{en:'Pick an answer and round the block to continue.',zh:'选一个答案，再舍入这一块，才能继续。'},
  blocks:{en:'Try blocks of 16 or NVFP4 to continue.',zh:'试一下每 16 个一块或 NVFP4，才能继续。'},compute:{en:'Try 4-bit weights only and FP8 weights and activations to continue.',zh:'试一下“只把权重量化到 4 位”和“权重和激活都用 FP8”，才能继续。'},scale:{en:`Look at ${M405} to continue.`,zh:`看一下 ${M405} 才能继续。`}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#qz-rail',app); if(!rail) return null; rail.innerHTML=railFull(); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#qz-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||(s.id==='predict'?predShown:(s.expect?!!observed[step]||s.id==='blocks'||s.id==='compute'||s.id==='scale':true)));
    if(s.id==='mission') $$('.seg-ctl[data-key="p1"]',rail).forEach(g=>A.ui.setPressed(g,P1.some(p=>p[0]===keyOf())?keyOf():'bf16'));
    if(s.id==='blocks') $$('#qz-ic-p4 [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===keyOf())); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=(tried.p4&&tried.p4.has(v))||v===keyOf()?p4Res(v):'';});
    if(s.id==='compute') $$('#qz-ic-p5 [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===p5Now())); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=(tried.p5&&tried.p5.has(v))||v===p5Now()?p5Res(v):'';});
    if(s.id==='scale') $$('.seg-ctl[data-key="p6"]',rail).forEach(g=>A.ui.setPressed(g,S.model));
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#qz-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):'';
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chW,chKV), m=$('#qz-ch-mem',rail); if(!m) return;
    m.textContent=gib(e.need); const g1=$('#qz-ch-g1',rail); g1.classList.toggle('met',e.fits); g1.classList.toggle('qz-over',!e.fits);
    $$('#qz-ic-chw [data-lever]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lever===chW)));
    $$('.seg-ctl[data-key="chkv"]',rail).forEach(x=>A.ui.setPressed(x,chKV));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#qz-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const curStep=()=>mode==='guided'?STEPS[step].id:'';
const viewNow=()=>mode==='challenge'?'serve':(mode==='guided'?STEPS[step].view:S.view);
const SVGNS='http://www.w3.org/2000/svg';
/* the phase the block picture shows, and how far phase 2 has moved (0…1) */
function dispPhase(){if(hidePred()) return {ph:0,t:0}; if(AN.on||AN.paused) return {ph:AN.ph,t:AN.t}; return watched.has(animKey())?{ph:3,t:1}:{ph:0,t:0};}
function drawBlock(){
  const svg=$('#qz-svg',app), box=$('#qz-blockv',app); if(!svg||box.hidden) return;
  const k=keyOf(), v=view(S.bsel,k), {ph,t}=dispPhase();
  const Wd=Math.max(280,Math.round(svg.clientWidth||box.clientWidth||520)), H=innerWidth<=480?210:236, ml=46, mr=8, mt=30, mb=30;
  const others=v.ent.filter(e=>e.j!==v.oi).map(e=>Math.abs(e.w)), ymax=Math.max(...others)*1.3||0.05;
  const cw=(Wd-ml-mr)/32, X=j=>ml+cw*(j+0.5), Y=y=>mt+(H-mt-mb)*(0.5-Math.max(-ymax,Math.min(ymax,y))/(2*ymax));
  let g=`<rect x="${ml}" y="${mt}" width="${Wd-ml-mr}" height="${H-mt-mb}" class="qz-plot"/>`;
  // y axis
  const tk=+(ymax*0.8).toPrecision(1); [-tk,0,tk].forEach(y=>{g+=`<line x1="${ml}" x2="${Wd-mr}" y1="${Y(y)}" y2="${Y(y)}" class="${y===0?'qz-zero':'qz-ax'}"/><text x="${ml-4}" y="${Y(y)+4}" class="qz-yl" text-anchor="end">${y===0?'0':(y>0?'':'−')+Math.abs(y)}</text>`;});
  // block segments and their grids
  const showGrid=ph>=1&&k!=='bf16';
  v.segs.forEach((sg,i)=>{const x0=ml+cw*sg.from, x1=ml+cw*sg.to;
    if(v.segs.length>1||sg.size<=32) g+=`<rect x="${x0+1}" y="${mt-14}" width="${x1-x0-2}" height="10" class="qz-seg${i%2?' qz-seg2':''}"/>`;
    if(showGrid){const vals=[]; sg.lv.forEach(l=>{vals.push(l*sg.s); if(l) vals.push(-l*sg.s);});
      const gap=vals.length>1?Math.min(...vals.filter(x=>x>0))/(2*ymax)*(H-mt-mb):99;
      if(gap>=3) vals.forEach(y=>{if(Math.abs(y)<=ymax) g+=`<line x1="${x0+1}" x2="${x1-1}" y1="${Y(y)}" y2="${Y(y)}" class="qz-grid"/>`;});}
  });
  // weights
  v.ent.forEach(e=>{const x=X(e.j), isO=e.j===v.oi, setter=v.segs.some(sg=>sg.setter===e.j);
    const big=Math.abs(e.w)>ymax, y0=Y(e.w), qv=ph>=2?(ph===2?e.w+(e.q-e.w)*t:e.q):e.w, y1=Y(qv);
    const red=isO&&ph>=1&&v.zeros>=8, zero=ph>=3&&e.q===0&&e.w!==0&&!isO;
    if(ph>=2&&Math.abs(y1-y0)>0.5) g+=`<line x1="${x}" x2="${x}" y1="${y0}" y2="${y1}" class="qz-mv${red?' qz-red':''}"/>`;
    g+=`<circle cx="${x}" cy="${y0}" r="${Math.min(4.2,cw*0.3)}" class="qz-orig${red?' qz-red':''}${setter&&ph>=1?' qz-set':''}"/>`;
    if(ph>=2) g+=zero?`<path d="M${x-3.5} ${y1-3.5}L${x+3.5} ${y1+3.5}M${x-3.5} ${y1+3.5}L${x+3.5} ${y1-3.5}" class="qz-x"/>`:`<circle cx="${x}" cy="${y1}" r="${Math.min(3.2,cw*0.24)}" class="qz-q${red?' qz-red':''}"/>`;
    if(big) g+=`<text x="${x}" y="${mt-18}" text-anchor="middle" class="qz-ol${red?' qz-redt':''}">↑ ${fix(e.w,2)}</text>`;
  });
  g+=`<text x="${ml}" y="${H-10}" class="qz-xl">${esc(tx('32 weights next to each other in one row','同一行里挨着的 32 个权重'))}</text>`;
  svg.setAttribute('viewBox',`0 0 ${Wd} ${H}`); svg.setAttribute('width',Wd); svg.setAttribute('height',H); svg.innerHTML=g;
  svg.setAttribute('aria-label',tx('A block of 32 weights before and after rounding','一块 32 个权重舍入前后'));
  // caption, phases, legend, note
  const cap=$('#qz-b-cap',app), sgs=v.segs, s0=sgs.find(sg=>sg.from<=(v.oi>=0?v.oi:0)&&sg.to>(v.oi>=0?v.oi:0))||sgs[0];
  cap.textContent=hidePred()?tx('The block before rounding: which weights survive 4 bits?','舍入之前的这一块：用 4 位存，哪些权重留得下？'):(k==='bf16'?tx('The block in BF16: nothing is rounded yet','BF16 的这一块：还没有舍入'):tx(`The block in ${sname(k)}: where does each weight end up?`,`用“${sname(k)}”存这一块：每个权重落到哪里？`));
  const leg=$('#qz-b-leg',app), lh=`<span class="lg"><i class="qz-sw qz-sw-o"></i>${tx('weight before rounding','舍入前的权重')}</span><span class="lg"><i class="qz-sw qz-sw-q"></i>${tx('stored after rounding','舍入后存下的值')}</span><span class="lg"><i class="qz-sw qz-sw-x">×</i>${tx('rounded to 0','舍入成 0')}</span><span class="lg"><i class="qz-sw qz-sw-g"></i>${tx('grid: values the block can store','格子：这一块能存的值')}</span><span class="lg"><i class="qz-sw qz-sw-r"></i>${tx('the outlier that sets the scale','决定缩放系数的离群值')}</span><span class="lg"><i class="qz-sw qz-sw-b"></i>${tx('blocks that share one scale','共用一个缩放系数的块')}</span>`;
  if(leg._h!==lh){leg.innerHTML=lh; leg._h=lh;}
  const note=$('#qz-b-note',app); let nt='';
  if(!hidePred()&&k!=='bf16'&&ph>=1&&s0){const inView=s0.setter>=0&&s0.setter<32;
    nt=tx(`Scale of the block${sgs.length>1?' with the outlier':''}: ${s0.s.toPrecision(3)} (${s0.size===D?'the whole row':`${s0.size} weights`}, set by its largest weight, ${fix(s0.amax,3)}${inView?'':', outside this view'}).`,`${sgs.length>1?'含离群值那块的':'这一块的'}缩放系数：${s0.s.toPrecision(3)}（${s0.size===D?'整行':`${s0.size} 个权重`}，由其中最大的权重 ${fix(s0.amax,3)} 决定${inView?'':'，它不在图里这 32 个之中'}）。`);
    if(ph>=3) nt+=tx(` ${v.zeros} of the ${v.oi>=0?31:32} ${v.oi>=0?'other ':''}weights became 0.`,`${v.oi>=0?'其余 31 个':'32 个'}权重里有 ${v.zeros} 个变成 0。`);
    if(k==='fp8'||k==='int8') nt+=tx(' The grid is too fine to draw at this size.','格子太细，这个尺寸下画不出来。');}
  if(note.textContent!==nt) note.textContent=nt;
  drawTransport();
}
const PHN=[{en:'1 Find the scale',zh:'1 定缩放系数'},{en:'2 Round to the grid',zh:'2 舍入到格子'},{en:'3 What is lost',zh:'3 丢了什么'}];
function drawTransport(){
  const el=$('#qz-tp',app); if(!el) return;
  const k=keyOf(), off=k==='bf16'||hidePred(), {ph}=dispPhase();
  const btn=AN.on?tx('❚❚ Pause','❚❚ 暂停'):(AN.paused?tx('▶ Continue','▶ 继续'):(watched.has(animKey())?tx('▶ Play again','▶ 再放一遍'):tx('▶ Play','▶ 播放')));
  const h=off?'':`<div class="qz-phs" role="group" aria-label="${esc(tx('Steps of the rounding','舍入的步骤'))}">${PHN.map((p,i)=>`<button type="button" class="qz-ph${ph===i+1?' on':''}" data-ph="${i+1}" aria-pressed="${ph===i+1}" data-tipx="${esc(tx('Shows this step.','显示这一步。'))}">${esc(L(p))}</button>`).join('')}</div>`+
    `<div class="qz-tbtns"><button type="button" class="btn sm" data-act="play" id="qz-play" data-tipx="${esc(tx('Plays the three steps of the rounding on this block.','把这一块的三步舍入播放一遍。'))}">${btn}</button><button type="button" class="btn ghost sm" data-act="replay" id="qz-replay" data-tipx="${esc(tx('Plays the rounding again from the start.','从头再播放一遍舍入。'))}">${tx('Replay','重放')}</button><div class="qz-speed"><span class="qz-speed-l">${tx('Speed','速度')}</span>${`<div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="speed">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${speed===v}">${x}</button>`).join('')}</div>`}</div></div>`;
  if(el._h!==h){el.innerHTML=h; el._h=h;}
}
function drawChart(){
  const el=$('#qz-chart',app); if(!el||el.hidden) return;
  const k=keyOf(), rows=ALLK.map(x=>evalK(SEED,x)), mx=Math.max(...rows.map(r=>r.oerr))*1.08;
  const h=`<div class="qz-cap">${tx('How far does each scheme move the small layer’s output, and how many bits does it store per weight?','每种方案让小模型这一层的输出偏多少，每个权重又占几位？')}</div>`+
   `<div class="qz-crows" role="list">`+rows.map(r=>`<div class="qz-crow${r.k===k?' me':''}" role="listitem" data-tipx="${esc(tx(`${sname(r.k)}: ${bitsT(r.bits)} bits per weight; output error ${pc(r.oerr)}; weights off by ${pc(r.werr)}; same pick as BF16 on ${pc(r.agree)} of 1,000 inputs.`,`${sname(r.k)}：每个权重 ${bitsT(r.bits)} 位；输出误差 ${pc(r.oerr)}；权重偏了 ${pc(r.werr)}；1,000 个输入里 ${pc(r.agree)} 选得和 BF16 一样。`))}">`+
     `<span class="qz-cl">${esc(sshort(r.k))}${r.k===k?` <em>${tx('your pick','你的选择')}</em>`:''}</span><span class="qz-cb">${esc(bitsT(r.bits))}</span><span class="qz-ct"><i style="width:${(r.oerr/mx*100).toFixed(1)}%"></i></span><span class="qz-cv">${pc(r.oerr)}</span><span class="qz-ca">${pc(r.agree)}</span></div>`).join('')+`</div>`+
   `<div class="qz-chead small muted"><span>${tx('Columns: scheme, bits per weight with scales, output error (bar), same pick as BF16. Per-row schemes count as 4 and 8 bits: a real model’s row has thousands of weights, so its scale adds under 0.01 bit.','各列：方案、算上缩放系数后每个权重的位数、输出误差（条）、和 BF16 选的一样的比例。每行一个缩放系数的方案记作 4 位、8 位：真实模型的一行有几千个权重，缩放系数分摊下来不到 0.01 位。')}</span></div>`;
  if(el._h!==h){el.innerHTML=h; el._h=h;}
}
const KN={d1:{en:'Decode, 1 request',zh:'decode，1 个请求'},d64:{en:'Decode, 64 requests',zh:'decode，64 个请求'},pre:{en:'Prefill, 2,048 tokens',zh:'prefill，2,048 个 token'}};
function drawSteps(){
  const el=$('#qz-steps',app); if(!el) return;
  const k=keyOf(), kv=S.kv;
  const h=`<div class="qz-cap">${tx(`How long does one step of ${M8} take on one ${GPU}, and what is it waiting on?`,`一张 ${GPU} 跑 ${M8}，一步要多久，在等什么？`)}</div>`+KINDS.map(x=>{const b=stepT('bf16','bf16',x), r=stepT(k,kv,x), mx=Math.max(b.tm,b.tc)*1.05;
    const w=v=>(v/mx*100).toFixed(1)+'%';
    return `<div class="qz-sk"><div class="qz-skh"><b>${esc(L(KN[x]))}</b><span>${tx(`${ms(r.t)} ms`,`${ms(r.t)} ms`)}${k==='bf16'&&kv==='bf16'?'':`<em>${tx(`BF16: ${ms(b.t)} ms`,`BF16：${ms(b.t)} ms`)}</em>`}</span></div>`+
     `<div class="qz-skr${r.mem?' long':''}"><span class="qz-skl">${tx('read memory','读显存')}</span><span class="qz-skt"><i class="qz-w" style="width:${w(r.tw)}"></i><i class="qz-kv" style="width:${w(r.tkv)}"></i></span><span class="qz-skv">${ms(r.tm)}</span></div>`+
     `<div class="qz-skr${r.mem?'':' long'}"><span class="qz-skl">${tx('compute','计算')}</span><span class="qz-skt"><i class="qz-c" style="width:${w(r.tc)}"></i></span><span class="qz-skv">${ms(r.tc)}</span></div></div>`;}).join('')+
   `<div class="legend-row qz-legend"><span class="lg" data-tipx="${esc(tx('Time to read the weights once.','把权重读一遍的时间。'))}"><i class="qz-sw qz-sw-w"></i>${tx('reading weights','读权重')}</span><span class="lg" data-tipx="${esc(tx('Time to read the KV cache of every request in the step.','读这一步里每个请求的 KV cache 的时间。'))}"><i class="qz-sw qz-sw-kv"></i>${tx('reading the KV cache','读 KV cache')}</span><span class="lg" data-tipx="${esc(TIP.stepm())}"><i class="qz-sw qz-sw-c"></i>${tx('arithmetic','计算')}</span><span class="lg"><i class="qz-sw qz-sw-long"></i>${tx('the longer one sets the step time','较长的那个决定一步的耗时')}</span></div>`;
  if(el._h!==h){el.innerHTML=h; el._h=h;}
}
const MEMK=['bf16','fp8','int4-128','mx','nv'];
function drawMem(){
  const el=$('#qz-mem',app); if(!el) return;
  const m=mode==='challenge'?'70b':S.model, P=MOD[m], bw=wGiB(m,'bf16'), mx=Math.max(bw,UGPU*(m==='8b'?1:m==='70b'?2:8))*1.04, k=mode==='challenge'?chW:keyOf();
  const lines=(m==='8b'?[1]:m==='70b'?[1,2]:[1,2,4,8]).filter(n=>UGPU*n<=mx);
  const h=`<div class="qz-cap">${tx(`How many ${GPU}s do ${P.n}’s weights need in each format?`,`${P.n} 的权重在每种格式下要几张 ${GPU}？`)}</div><div class="qz-mrows">`+MEMK.map(x=>{const g=gpus(m,x);
    return `<div class="qz-mrow${x===k?' me':''}"><span class="qz-ml">${esc(sshort(x))}</span><span class="qz-mt">${lines.map(n=>`<b class="qz-gl" style="left:${(UGPU*n/mx*100).toFixed(2)}%"></b>`).join('')}<i style="width:${(g.w/mx*100).toFixed(1)}%"></i></span><span class="qz-mv2">${gib(g.w)}</span><span class="qz-mn">${g.n?tx(`${g.n} GPU${g.n>1?'s':''}, ${gib(g.room)} left each`,`${g.n} 张卡，每张剩 ${gib(g.room)}`):tx('over 8 GPUs','8 张卡放不下')}</span></div>`;}).join('')+`</div>`+
   `<div class="qz-gls small muted">${lines.map(n=>`<span style="left:${(UGPU*n/mx*100).toFixed(2)}%">${n}×70</span>`).join('')}</div>`+
   `<div class="legend-row qz-legend"><span class="lg"><i class="qz-sw qz-sw-w"></i>${tx('weights, all GPUs together','权重，所有卡加起来')}</span><span class="lg" data-tipx="${esc(TIP.usable())}"><i class="qz-sw qz-sw-gl"></i>${tx('usable memory of 1, 2, 4 or 8 H100s (70 GiB each)','1、2、4、8 张 H100 能用的显存（每张 70 GiB）')}</span></div>`;
  if(el._h!==h){el.innerHTML=h; el._h=h;}
}
const memOnly=()=>mode!=='challenge'&&viewNow()==='serve'&&(isP('scale')||(mode==='sandbox'&&S.model!=='8b'));
function statusOf(){
  const k=keyOf();
  if(mode==='challenge'){const e=chEval(chW,chKV); return chSubmitted?(e.ok?{cls:'ok',short:tx('Goal met','达标'),txt:tx('Goal met','达标')}:{cls:'bad',short:tx('Not met','没达标'),txt:tx('Not met','没达标')}):{cls:'',short:tx('Not submitted','未提交'),txt:''};}
  if(hidePred()) return {cls:'',short:tx('Predict first','先预测'),txt:tx('Predict first','先预测')};
  if(memOnly()){const P=MOD[S.model], g=gpus(S.model,'fp8'); return {cls:'',short:'',txt:tx(`${P.n} in FP8: ${g.n?g.n+' '+GPU+(g.n>1?'s':''):'more than 8 '+GPU+'s'}`,`${P.n} 用 FP8：${g.n?g.n+' 张 '+GPU:'8 张 '+GPU+' 也不够'}`)};}
  if(viewNow()==='serve'){const r=stepT(k,S.kv,'d1'); return {cls:'',short:tx(`Decode (1 request) ${ms(r.t)} ms`,`decode 一步（1 个请求）${ms(r.t)} ms`),txt:tx(`Decode step with one request: ${ms(r.t)} ms`,`1 个请求的 decode 步：${ms(r.t)} ms`)};}
  const e=evalK(SEED,k); return {cls:'',short:tx(`layer output error ${pc(e.oerr)}`,`整层输出误差 ${pc(e.oerr)}`),txt:tx(`${sname(k)}: the layer’s output error ${pc(e.oerr)}`,`${sname(k)}：整层输出误差 ${pc(e.oerr)}`)};
}
function drawHead(){
  const v=viewNow(), t=$('#qz-st-title',app), sub=$('#qz-st-sub',app), st=$('#qz-status',app), s=statusOf();
  const k=keyOf();
  let tt, ss;
  if(v==='block'){tt=AN.on?tx('Playing: rounding one block of 32 weights','正在播放：舍入一块 32 个权重'):tx('One block of the small layer, rounded','小模型这一层的一块权重，舍入之后'); ss=hidePred()?tx('INT4, one scale per 32 weights: not rounded yet','INT4，每 32 个权重一个缩放系数：还没舍入'):sname(k);}
  else if(mode==='challenge'){tt=tx(`${M70} on one ${GPU}`,`一张 ${GPU} 跑 ${M70}`); ss=tx('Weights in each format; the KV cache of 16 users × 8K comes on top','每种格式的权重；16 个用户 × 8K 的 KV cache 另外再加');}
  else if(memOnly()){const P=MOD[S.model]; tt=tx(`${P.n}’s weights: how many ${GPU}s each format needs`,`${P.n} 的权重：每种格式要几张 ${GPU}`); ss=tx(`Scales included; 70 GiB usable per ${GPU}`,`算上缩放系数；每张 ${GPU} 能用 70 GiB`);}
  else {const f=sshort(k)+(ACT8(k)?tx(' (weights and activations)','（权重和激活）'):''), withMem=mode==='sandbox';
    tt=withMem?tx(`Step time and memory: ${f}`,`每一步的耗时和显存：${f}`):tx(`Step time: ${f}`,`每一步的耗时：${f}`); ss=tx(`KV cache in ${S.kv==='fp8'?'FP8':'BF16'}; decode requests have 2,048 tokens of context`,`KV cache 用 ${S.kv==='fp8'?'FP8':'BF16'}；decode 请求都有 2,048 个 token 的上下文`);}
  if(t.textContent!==tt) t.textContent=tt; if(sub.textContent!==ss) sub.textContent=ss;
  st.className='status '+s.cls; st.textContent=s.short; st.hidden=!s.short;
}
function metricsDef(){
  const ch=mode==='challenge', k=ch?chW:keyOf(), kv=ch?chKV:S.kv, e=evalK(ch?CH_SEED:SEED,k), r1=stepT(k,kv,'d1'), r64=stepT(k,kv,'d64'), p1=isP('mission'), hide=ch?!chSubmitted:(hidePred()||p1);
  return [
   {id:'bits',lab:tx('Bits per weight','每个权重的位数'),v:SCH[k].bits,txt:bitsT(SCH[k].bits),dp:3,sub:tx('scales included','算上缩放系数'),tip:tx('The bits each weight takes, counting its share of the block’s scale.','每个权重占的位数，算上它分摊的那份缩放系数。')},
   {id:'oerr',lab:tx('Output error','输出误差'),v:e.oerr*100,txt:hide?'—':pc(e.oerr),dp:1,pts:true,sub:ch?tx('the other layer (challenge)','另一层（挑战题）'):p1?tx('a small layer: next part','小模型的一层，下一部分讲'):tx('small layer, 1,000 inputs','小模型这一层，1,000 个输入'),tip:TIP.oerr(),hide},
   {id:'same',lab:tx('Same pick as BF16','和 BF16 选的一样'),v:e.agree*100,txt:hide?'—':pc(e.agree),dp:1,pts:true,sub:tx('in this lab','在这个 Lab 里'),tip:TIP.same(),hide},
   {id:'d1',lab:tx('Decode step, 1 request','decode 一步，1 个请求'),v:r1.t*1e3,txt:ms(r1.t)+' ms',dp:2,sub:ch?tx('8B timing, not part of the challenge','8B 的耗时，挑战不看'):M8,tip:TIP.stepm()},
   {id:'d64',lab:tx('Decode step, 64 requests','decode 一步，64 个请求'),v:r64.t*1e3,txt:ms(r64.t)+' ms',dp:1,sub:ch?tx('8B timing, not part of the challenge','8B 的耗时，挑战不看'):M8,tip:tx('64 requests with 2,048 tokens of context each: the step reads the weights once and every request’s KV cache.','64 个请求，每个有 2,048 个 token 的上下文：这一步把权重读一遍，再读每个请求的 KV cache。')}];
}
function drawMetrics(){
  const el=$('#qz-metrics',app), ms_=metricsDef();
  if(!el._built){el.innerHTML=ms_.map(m=>`<div class="metric" id="qz-m-${m.id}" data-tipx=""><div class="k"></div><div class="v mono qz-mval"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); el._built=true;}
  const last=lastM; lastM={};
  ms_.forEach(m=>{const c=$('#qz-m-'+m.id,app); c.dataset.tipx=m.tip; c.querySelector('.k').textContent=m.lab; c.querySelector('.qz-mval').textContent=m.txt; c.querySelector('.sub').textContent=m.sub;
    const d=c.querySelector('.d'); let dt='';
    if(userAct&&last&&last[m.id]!=null&&!m.hide&&last[m.id]!==m.v) dt=A.fmt.delta(last[m.id],m.v,{dp:m.dp,pts:!!m.pts})||'';
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d._n=actN; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(!last||m.hide||(userAct&&d._n!==actN)){clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent=''; d._n=0;}
    lastM[m.id]=m.hide?null:m.v;});
}
function mathHTML(){
  const k=keyOf()==='bf16'?'int4-32':keyOf(), v=view(S.bsel,k.startsWith('int4')?k:'int4-32'), sg=v.segs.find(s=>s.from<=Math.max(0,v.oi)&&s.to>Math.max(0,v.oi))||v.segs[0];
  const cand=v.ent.filter(e=>e.j>=sg.from&&e.j<sg.to&&e.j!==v.oi&&e.w!==0).sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)), sm=cand[0]||v.ent[0], s=sg.s, S4=s.toPrecision(4), ws=sm.w/s, q=Math.max(-7,Math.min(7,rne(ws)));
  const bits=SCH[keyOf()].bits, kk=keyOf(), r=stepT(kk,S.kv,'d1'), m=S.model, P=MOD[m], wg=wGiB(m,kk);
  let h=MB(tx('<b>Scale and rounding.</b> The weight with the largest absolute value sets the scale; each weight is divided by it, rounded to a whole number from −7 to 7 and multiplied back. One weight of the block shown (INT4):','<b>缩放系数和舍入。</b>绝对值最大的权重决定缩放系数；每个权重除以它，舍入到 −7 到 7 的整数，再乘回来。图里这一块的一个权重（INT4）：'),
    EQ([['s','\\frac{\\max |w|}{7}'],['',`\\frac{${fix(sg.amax,5)}}{7} \\approx ${S4}`]])+
    EQ([['w / s',`\\frac{${fix(sm.w,5)}}{${S4}} \\approx ${fix(ws,2)}`],['w_q','s \\cdot \\operatorname{round}(w / s)'],['',`${S4} \\times \\operatorname{round}(${fix(ws,2)})`],['',`${S4} \\times ${q} = ${fix(q*s,5)}`]]),
    tx(`A weight smaller than half a grid step, ${M('s / 2')}, becomes 0.`,`比半个格子（${M('s / 2')}）还小的权重会变成 0。`));
  h+=MB(tx('<b>Bits per weight.</b> Each block of B weights stores one scale of S bits next to its 4-bit values (INT4 per 128: B = 128, S = 16):','<b>每个权重的位数。</b>每块 B 个权重，除了 4 位的值，还存一个 S 位的缩放系数（INT4 每 128 个：B = 128，S = 16）：'),
    EQ([[tx('\\text{bits per weight}','\\text{每个权重的位数}'),'4 + \\frac{S}{B}'],['','4 + \\frac{16}{128} = 4.125']]),tx(`NVFP4: ${M('4 + 8/16 = 4.5')}. MXFP4: ${M('4 + 8/32 = 4.25')}. Your pick: ${bitsT(bits)} bits.`,`NVFP4：${M('4 + 8/16 = 4.5')}。MXFP4：${M('4 + 8/32 = 4.25')}。你选的方案：${bitsT(bits)} 位。`));
  h+=MB(tx(`<b>Weights in memory.</b> Parameters times bits per weight, over 8 bits per byte (${P.n}, ${sshort(kk)}):`,`<b>权重占的显存。</b>参数量乘以每个权重的位数，再除以每字节 8 位（${P.n}，${sshort(kk)}）：`),
    EQ([[tx('\\text{bytes}','\\text{字节}'),tx('P \\times \\frac{\\text{bits}}{8}','P \\times \\frac{\\text{位数}}{8}')],['',`${P.P} \\times \\frac{${bitsT(RBITS[kk])}}{8}`],['',`${fix(P.P*RBITS[kk]/8/1e9,1)}\\ \\text{GB} = ${fix(wg,1)}\\ \\text{GiB}`]]),'');
  const pst=fix(P_STEP/1e9,3), kvg=fix(MOD['8b'].kv*KVB[S.kv]*CTX/1e9,3), byt=fix(r.bytes/1e9,3), lin=2*P_STEP, att=attn(CTX,CTX+1), fl=fix((lin+att)/1e9,3);
  h+=MB(tx(`<b>Step time.</b> A decode step of ${M8} with one request and 2,048 tokens of context takes the longer of the memory reads, ${M('t_m')}, and the arithmetic, ${M('t_c')}. Decode does not read the whole embedding table (each token takes one row of it), so the step reads ${pst} billion parameters, plus the KV cache. Bytes in GB and FLOPs in GFLOP, over TB/s and TFLOP/s, give ms:`,
                `<b>每一步的耗时。</b>${M8} 1 个请求、2,048 个 token 上下文的 decode 步，取读显存时间 ${M('t_m')} 和计算时间 ${M('t_c')} 中较长的那个。decode 不读整张 embedding 表（每个 token 只取其中一行），所以读的是 ${pst}B 个参数，再加上 KV cache。字节以 GB 为单位、FLOPs 以 GFLOP 为单位，除以 TB/s 和 TFLOP/s，得到的就是毫秒：`),
    EQ([[tx('\\text{bytes}','\\text{字节}'),tx('P_{\\text{step}} \\times \\frac{\\text{bits}}{8} + \\text{KV}','P_{\\text{step}} \\times \\frac{\\text{位数}}{8} + \\text{KV}')],['',`${pst} \\times \\frac{${bitsT(RBITS[kk])}}{8} + ${kvg} = ${byt}`],
        ['t_m',tx('\\frac{\\text{bytes}}{\\text{BW} \\times 60\\%}','\\frac{\\text{字节}}{\\text{带宽} \\times 60\\%}')],['',`\\frac{${byt}}{3.35 \\times 0.6} = ${ms(r.tm)}\\ \\text{ms}`]])+
    EQ([['\\text{FLOPs}',tx('2 P_{\\text{step}} + \\text{attention}','2 P_{\\text{step}} + \\text{attention}')],['',`2 \\times ${pst} + ${fix(att/1e9,3)} = ${fl}`],
        ['t_c',ACT8(kk)?tx('\\frac{\\text{linear}}{1979 \\times 50\\%} + \\frac{\\text{attention}}{989.5 \\times 50\\%}','\\frac{\\text{线性层}}{1979 \\times 50\\%} + \\frac{\\text{attention}}{989.5 \\times 50\\%}'):tx('\\frac{\\text{FLOPs}}{\\text{peak} \\times 50\\%}','\\frac{\\text{FLOPs}}{\\text{峰值} \\times 50\\%}')],
        ['',ACT8(kk)?`\\frac{${fix(lin/1e9,3)}}{1979 \\times 0.5} + \\frac{${fix(att/1e9,3)}}{989.5 \\times 0.5} = ${ms(r.tc)}\\ \\text{ms}`:`\\frac{${fl}}{989.5 \\times 0.5} = ${ms(r.tc)}\\ \\text{ms}`]])+
    EQ([['t','\\max(t_m, t_c)'],['',`\\max(${ms(r.tm)}, ${ms(r.tc)}) = ${ms(r.t)}\\ \\text{ms}`]]),
    tx(`Peak: ${ACT8(kk)?'the linear layers at the FP8 rate, 1,979 TFLOP/s; attention at the BF16 rate, 989.5 TFLOP/s':'989.5 TFLOP/s (BF16)'}; BW: 3.35 TB/s.`,`峰值：${ACT8(kk)?'线性层按 FP8 的 1,979 TFLOP/s，attention 按 BF16 的 989.5 TFLOP/s':'989.5 TFLOP/s（BF16）'}；带宽：3.35 TB/s。`));
  return h;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const v=viewNow(), bv=$('#qz-blockv',app), ch=$('#qz-chart',app), sv=$('#qz-serve',app);
  bv.hidden=v!=='block'; sv.hidden=v!=='serve';
  ch.hidden=v!=='block'||(mode==='guided'&&step<SIX('blocks'));
  const memFirst=mode==='challenge'||isP('scale')||(mode==='sandbox'&&S.model!=='8b');
  sv.classList.toggle('qz-memfirst',memFirst);
  $('#qz-mem',app).hidden=mode==='guided'&&(isP('mission')||isP('compute'));
  $('#qz-steps',app).hidden=mode==='challenge'||memOnly();
  $('#qz-metrics',app).hidden=isP('scale');
  drawHead(); drawBlock(); drawChart(); if(v==='serve'){drawSteps(); drawMem();} drawMetrics();
  const mb=$('#qz-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  {const ht=mode+'|'+token(); if(ht!==lastHash){lastHash=ht; A.hash.write(token());}} A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you pick the weight and KV cache formats in the task card. Fixed: ${M70}, one H100, 16 users × 8,192 tokens.`,`挑战中只能在任务卡里选权重格式和 KV cache 格式。固定的是：${M70}、一张 H100、16 个用户 × 8,192 个 token。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  const m={mission:tx('In this part you pick the weight format in the card. You can change everything in Scenarios.','这一部分只在卡片里选权重格式。在应用场景里什么都可以改。'),
    round:tx('Fixed in this part: INT4 with one scale per 32 weights, on a block without an outlier. You can change them in Scenarios.','这一部分固定用 INT4、每 32 个权重一个缩放系数，舍入一块没有离群值的权重。在应用场景里可以改。'),
    predict:tx('Fixed in this part: INT4 with one scale per 32 weights, on the block with the outlier.','这一部分固定用 INT4、每 32 个权重一个缩放系数，舍入有离群值的那一块。'),
    blocks:tx('In this part you pick how the row is split into blocks in the card. You can change everything in Scenarios.','这一部分只在卡片里选怎么分块。在应用场景里什么都可以改。'),
    compute:tx(`In this part you pick what is quantized in the card. Fixed: ${M8} on one H100.`,`这一部分只在卡片里选量化什么。固定的是：一张 H100 跑 ${M8}。`),
    scale:tx('In this part you pick the model in the card. You can change everything in Scenarios.','这一部分只在卡片里选模型。在应用场景里什么都可以改。'),
    remember:tx('Fixed in this part: INT4 with one scale per 16 weights. You can change it in Scenarios.','这一部分固定用 INT4、每 16 个权重一个缩放系数。在应用场景里可以改。')};
  return m[id]||'';
}
function syncControls(){
  $$('#qz-controls .seg-ctl[data-key="sch"]',app).forEach(g=>A.ui.setPressed(g,S.sch));
  $$('#qz-controls .seg-ctl[data-key="blk"]',app).forEach(g=>A.ui.setPressed(g,S.blk));
  $$('#qz-controls .seg-ctl[data-key="kv"]',app).forEach(g=>A.ui.setPressed(g,S.kv));
  $$('#qz-controls .seg-ctl[data-key="model"]',app).forEach(g=>A.ui.setPressed(g,S.model));
  $$('#qz-controls .seg-ctl[data-key="view"]',app).forEach(g=>A.ui.setPressed(g,S.view));
  $$('.seg-ctl[data-key="speed"]',app).forEach(g=>A.ui.setPressed(g,speed));
  const c=$('#qz-controls',app), g=mode!=='sandbox';
  c.classList.toggle('locked',mode==='challenge');
  const hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  ['#qz-ctl-sch-wrap','#qz-ctl-kv-wrap','#qz-ctl-model-wrap','#qz-ctl-view-wrap','#qz-ctl-reset-wrap'].forEach(id=>hide(id,g));
  hide('#qz-ctl-blk-wrap',g||S.sch!=='int4');
  const ln=$('#qz-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); if(s.txt) A.announce(s.txt);},400);}

/* =====================================================================
   the rounding replay (R3.9, R3.20)
   ===================================================================== */
function stopAnim(){if(AN.raf) cancelAnimationFrame(AN.raf); AN.on=false; AN.raf=0;}
function playAnim(from){
  stopAnim(); const key=animKey(); if(keyOf()==='bf16') return;
  if(A.reduceMotion()){watched.add(key); AN={on:false,raf:0,start:0,ph:3,t:1,key,paused:false,acc:0}; render(); return;}
  nPlays++; AN={on:true,raf:0,start:0,ph:from||1,t:0,key,paused:false,acc:((from||1)-1)*SPEEDS[speed]*1000};
  let last=0;
  const tick=ts=>{if(!AN.on||!app){AN.raf=0; return;}
    if(last) AN.acc+=Math.min(100,ts-last); last=ts;
    const per=SPEEDS[speed]*1000, p=AN.acc/per, ph=1+Math.floor(p);
    if(ph>3){AN.on=false; AN.raf=0; AN.ph=3; AN.t=1; watched.add(AN.key); render(); return;}
    const t=p-Math.floor(p), changed=ph!==AN.ph; AN.ph=ph; AN.t=ph===2?t:(ph>2?1:0);
    if(changed) render(); else drawBlock();
    AN.raf=requestAnimationFrame(tick);};
  AN.raf=requestAnimationFrame(tick); render();
}
/* show a setting: play when its picture was never watched, else show the end at once (R3.20) */
function showKey(){const key=animKey(); if(viewNow()!=='block'||keyOf()==='bf16'||hidePred()){stopAnim(); AN.paused=false; return;}
  if(AN.key===key&&(AN.on||AN.paused)) return;
  stopAnim(); AN.paused=false;
  if(watched.has(key)){AN.ph=3; AN.t=1; AN.key=key; return;}
  playAnim(1);}
function phaseJump(ph){stopAnim(); AN={...AN,on:false,paused:true,ph,t:ph>=2?1:0,key:animKey()}; if(ph===3){watched.add(animKey()); AN.paused=false;} render();}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p);
  if(!['bf16','fp8','int8','int4','mx','nv'].includes(S.sch)) S.sch='bf16'; if(!['row','128','32','16'].includes(S.blk)) S.blk='32';
  if(!KVB[S.kv]) S.kv='bf16'; if(!MOD[S.model]) S.model='8b'; if(!['block','serve'].includes(S.view)) S.view='serve'; if(!['clean','outlier'].includes(S.bsel)) S.bsel='outlier';
  showKey(); render();}
const fromKey=k=>k.startsWith('int4-')?{sch:'int4',blk:k.slice(5)}:{sch:k};
function byLearner(fn){userAct=true; actN++; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setP1(k){setState(fromKey(k));}
function setP4(k){(tried.p4=tried.p4||new Set()).add(k); setState(fromKey(k));}
function setP5(k){(tried.p5=tried.p5||new Set()).add(k); setState({...P5[k].s});}
function setModel(m){(tried.p6=tried.p6||new Set()).add(m); setState({model:m});}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='predict'&&!predShown) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='blocks'||id==='compute') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the options in the card.','向下滚到卡片里的选项。'))}">${tx('Pick below ↓','在下面选 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#qz-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.id==='blocks'&&!tried.p4) tried.p4=new Set(['int4-32']);
  lastM=null; showKey(); renderRail();
}
function revealStep(){A.ui.reveal($('#qz-rail .step.current',app)||$('#qz-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2, R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#qz-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#qz-rail .step.current',app)||$('#qz-rail .railcard',app)||$('#qz-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#qz-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#qz-insight',app)); else revealBox($('#qz-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; stopAnim(); AN.paused=false;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chW='bf16'; chKV='bf16';}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  lastM=null;
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  showKey(); renderRail();
}
const token=(s=S)=>[s.sch,s.blk,s.kv,s.model,s.view].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [sch,blk,kv,model,vw]=p;
  if(!['bf16','fp8','int8','int4','mx','nv'].includes(sch)||!['row','128','32','16'].includes(blk)||!KVB[kv]||!MOD[model]||!['block','serve'].includes(vw)) return null;
  return {...BASE,sch,blk,kv,model,view:vw};
}

/* =====================================================================
   interview drill (seeded): weights memory and GPUs for a model in a format
   ===================================================================== */
const DRM=[['8b',M8],['70b',M70],['405b',M405]], DRK=['fp8','int4-128','mx','nv'];
function makeDrill(seed){const r=mulberry(seed*104729+7), m=DRM[Math.floor(r()*3)], k=DRK[Math.floor(r()*4)], kv=r()<0.5?'bf16':'fp8', users=[8,16,32][Math.floor(r()*3)];
  return {m:m[0],name:m[1],k,kv,users};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#qz-drill-q',app), a=$('#qz-drill-a',app), P=MOD[d.m];
  const f8=d.k==='fp8';
  qd.innerHTML=tx(`You serve <b>${d.name}</b> on H100s with 70 GiB usable each, with ${f8?'<b>FP8 weights and activations</b> (one scale per tensor for the weights)':`<b>${sname(d.k)}</b> weights`}. <b>${d.users}</b> users each keep 8,192 tokens of KV cache in <b>${d.kv==='fp8'?'FP8':'BF16'}</b>. How many H100s (1, 2, 4 or 8) do you need?`,
                  `你用 H100 跑 <b>${d.name}</b>，每张卡能用 70 GiB，${f8?'<b>权重和激活都用 FP8</b>（权重每个张量一个缩放系数）':`权重用 <b>${sname(d.k)}</b>`}。<b>${d.users}</b> 个用户，每人 8,192 个 token 的 KV cache，用 <b>${d.kv==='fp8'?'FP8':'BF16'}</b> 存。要几张 H100（1、2、4 或 8）？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const w=wGiB(d.m,d.k), kvg=d.users*8192*P.kv*KVB[d.kv]/GiB, need=w+kvg; let n=0; for(const x of [1,2,4,8]) if(need/x<=UGPU){n=x; break;}
  const dEq=EQ([[tx('\\text{weights}','\\text{权重}'),`\\frac{${P.P} \\times ${bitsT(RBITS[d.k])}}{8 \\times 2^{30}} = ${fix(w,1)}\\ \\text{GiB}`],[tx('\\text{KV cache}','\\text{KV cache}'),`${d.users} \\times 8192 \\times ${P.kv*KVB[d.kv]/KiB}\\ \\text{KiB} = ${fix(kvg,1)}\\ \\text{GiB}`]]);
  a.innerHTML=tx(`<p><b>${n?`${n} H100${n>1?'s':''}`:'More than 8 H100s'}: ${fix(need,1)} GiB in all${n?`, ${fix(need/n,1)} GiB each`:''}.</b></p>${dEq}<p>${f8?'FP8 takes 8 bits per weight: one scale per tensor adds almost nothing.':`Count the scales in the bits (${bitsT(RBITS[d.k])} bits, not 4).`} Add every user’s KV cache before you divide by the GPUs. Tensor parallelism splits both the weights and the KV cache evenly.</p>`,
                 `<p><b>${n?`${n} 张 H100`:'8 张 H100 也不够'}：一共 ${fix(need,1)} GiB${n?`，每张 ${fix(need/n,1)} GiB`:''}。</b></p>${dEq}<p>${f8?'FP8 每个权重 8 位，每个张量只有一个缩放系数，分摊下来几乎为 0。':`位数要算上缩放系数（${bitsT(RBITS[d.k])} 位，不是 4 位）。`}再加上每个用户的 KV cache，然后才除以卡数。张量并行把权重和 KV cache 都平均分到每张卡上。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const TERM={};
const CTLN={sch:{en:'Weight format',zh:'权重格式'},blk:{en:'INT4: weights per scale',zh:'INT4：每个缩放系数管几个权重'},kv:{en:'KV cache',zh:'KV cache'}};
const termTipHTML=k=>{if(!TERM[k]) return ''; const a=jumpAct(k); return TERM[k]()+(a?`<div class="act">${esc(a)}</div>`:'');};
function ctlLockMsg(k){if(mode==='sandbox') return ''; return tx('Fixed here; you can change it in Scenarios.','这里固定不变；在应用场景里可以改。');}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#qz-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#qz-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=$({sch:'#qz-ctl-sch',blk:'#qz-ctl-blk',kv:'#qz-ctl-kv'}[k],app); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){lastM=null; bySettings(btn,()=>setState({...BASE})); return;}
  if(a==='pred-go'){predShown=true; observed[step]=true; byLearner(()=>{renderRail(); showKey(); render();}); revealSoon(); return;}
  if(a==='play'){if(AN.on){stopAnim(); AN.paused=true; render(); return;} if(AN.paused){const from=AN.ph; playAnim(from); return;} playAnim(1); return;}
  if(a==='replay'){playAnim(1); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chW,chKV).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#qz-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#qz-rail .step.current .qz-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#qz-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#qz-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#qz-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; lastM=null; byLearner(()=>{if(s.showMe==='play'){const sp=speed; playAnim(1); speed=sp;} else s.showMe();}); revealSoon(); wantObs=false;} return;}
    if(d.pred){if(predShown) return; pred=d.pred; renderRail(); const fb=$(`#qz-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.ph){phaseJump(+d.ph); return;}
    if(d.lever!=null){
      if(b.closest('#qz-ic-p4')){const f=()=>byLearner(()=>setP4(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
      if(b.closest('#qz-ic-p5')){const f=()=>byLearner(()=>setP5(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
      if(b.closest('#qz-ic-chw')){if(chW!==d.lever){chW=d.lever; chSubmitted=false; chReveal=false;} byLearner(()=>render()); return;}
    }
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='speed'){speed=d.v; render(); return;}
      if(k==='p1'){byLearner(()=>setP1(d.v)); revealSoon(); return;}
      if(k==='p6'){byLearner(()=>setModel(d.v)); revealSoon(); return;}
      if(k==='chkv'){if(chKV!==d.v){chKV=d.v; chSubmitted=false; chReveal=false;} byLearner(()=>render()); return;}
      if(k==='sch'||k==='blk'||k==='kv'||k==='model'||k==='view') bySettings(b,()=>setState({[k]:d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; lastM=null; byLearner(()=>setState({...BASE,...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#qz-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#qz-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
const sOf=s=>({sch:s.sch,blk:s.blk,kv:s.kv,model:s.model,view:s.view,bsel:s.bsel});
const keyOk=k=>typeof k==='string'&&(['bf16','fp8','int8','mx','nv'].includes(k)||/^int4-(row|128|32|16)$/.test(k));
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>{const tr={}; for(const k in tried) if(tried[k] instanceof Set) tr[k]=[...tried[k]];
    return {v:1,mode,step,S:sOf(S),pred,predShown,observed:{...observed},tried:tr,watched:[...watched],speed,saved:saved?sOf(saved):null,chW,chKV,chSubmitted,chReveal,
      sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(chW,chKV).ok};},
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=sOf({...p,bsel:['clean','outlier'].includes(o.S.bsel)?o.S.bsel:BASE.bsel}); pred=PRED.some(x=>x.v===o.pred)?o.pred:null; predShown=o.predShown===true; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const OK={p4:x=>P4.includes(x),p5:x=>P5K.includes(x),p6:x=>!!MOD[x]};
    tried={}; if(o.tried&&typeof o.tried==='object') for(const k in OK) if(Array.isArray(o.tried[k])) tried[k]=new Set(o.tried[k].filter(OK[k]));
    watched.clear(); if(Array.isArray(o.watched)) o.watched.forEach(k=>{const q=typeof k==='string'?k.split('|'):[]; if(q.length===2&&['clean','outlier'].includes(q[0])&&keyOk(q[1])) watched.add(k);});
    if(SPEEDS[o.speed]) speed=o.speed;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sOf({...sv,bsel:['clean','outlier'].includes(o.saved.bsel)?o.saved.bsel:BASE.bsel}):null;
    chW=mode==='challenge'&&CHK.includes(o.chW)?o.chW:'bf16'; chKV=mode==='challenge'&&KVB[o.chKV]?o.chKV:'bf16';
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true; resumed=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!resumed?parse(tok):null; resumed=false;   /* after restore() the address names the saved settings: nothing to apply */ if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    if(mode==='guided'&&!p&&step===0) Object.assign(S,STEPS[0].enter());
    el.innerHTML=viewHTML(); app=$('#qz-app',el); lastM=null; lastHash='';
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#qz-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    fillRail();
    showKey();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); stopAnim(); root=null; app=null;},
  onLang(){if(root){const d=$('#qz-math',app), open=d&&d.open; const an={...AN}; this.mount(root,null); if(open) $('#qz-math',app).open=true; if(an.on&&!AN.on) playAnim(an.ph);}},
  onResize(){if(app){drawBlock();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#qz-blockv',app).hidden?$('#qz-serve',app):$('#qz-blockv',app)),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#qz-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), k=keyOf();
      status.className='status '+s.cls; status.textContent=s.short; status.hidden=!s.short;
      if(mode==='challenge'){const e=chEval(chW,chKV); vis.innerHTML=`<div class="qz-dk"><i style="width:${Math.min(100,e.need/180*100).toFixed(1)}%"></i><b style="left:${(UGPU/180*100).toFixed(1)}%"></b></div>`;
        read.innerHTML=esc(tx(`${sshort(chW)} weights and ${chKV==='fp8'?'FP8':'BF16'} KV cache need ${gib(e.need)} of 70 GiB`,`${sshort(chW)} 权重加 ${chKV==='fp8'?'FP8':'BF16'} KV cache 要 70 GiB 里的 ${gib(e.need)}`)); return;}
      if(hidePred()){vis.innerHTML=''; read.innerHTML=esc(tx('Predict first: the block is not rounded yet','先预测：这一块还没舍入')); return;}
      if(memOnly()){const P=MOD[S.model], g=gpus(S.model,'fp8'); vis.innerHTML='';
        read.innerHTML=esc(tx(`${P.n}’s weights in FP8: ${g.n?`${g.n} ${GPU}${g.n>1?'s':''}`:`more than 8 ${GPU}s`}`,`${P.n} 的权重用 FP8：${g.n?`${g.n} 张 ${GPU}`:`8 张 ${GPU} 也不够`}`)); return;}
      if(viewNow()==='serve'){const r=stepT(k,S.kv,'d1'), b=stepT('bf16','bf16','d1'); vis.innerHTML=`<div class="qz-dk"><i style="width:${(r.t/b.t*100).toFixed(1)}%"></i></div>`; status.textContent=`${ms(r.t)} ms`;
        read.innerHTML=esc(tx(`${sshort(k)} weights: decode step, 1 request (BF16: ${ms(b.t)} ms)`,`${sshort(k)} 权重：1 个请求的 decode 步（BF16 是 ${ms(b.t)} ms）`)); return;}
      const e=evalK(SEED,k), v=view(S.bsel,k);
      vis.innerHTML=`<div class="qz-dk"><i style="width:${Math.min(100,e.oerr/0.3*100).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(tx(`${sshort(k)}: ${v.zeros} weights of the block become 0`,`${sshort(k)}：这一块有 ${v.zeros} 个权重变成 0`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__qz=Object.freeze({
  evalK:(seed,k)=>deep(evalK(seed,k)), view:(b,k)=>{const v=view(b,k); return deep({...v,ent:v.ent.map(e=>[e.w,e.q,e.hit?1:0])});}, blocks:()=>deep(blocks()),
  gen:seed=>{const G=gen(seed); return {W:Array.from(G.W.slice(0,64)),ch:G.ch.slice(),hits:Array.from(G.hit).reduce((a,b)=>a+b,0),sumW:Array.from(G.W).reduce((a,b)=>a+b,0),sumX:Array.from(G.X).reduce((a,b)=>a+b,0)};},
  qsum:(seed,k)=>{const q=quantW(seed,k); let s=0, s2=0; for(let i=0;i<q.length;i++){s+=q[i]; s2+=Math.abs(q[i]);} return [s,s2];},
  step:(k,kv,kind)=>deep(stepT(k,kv,kind)), wGiB, gpus:(m,k)=>deep(gpus(m,k)), kvTok8, chEval:(k,kv)=>deep(chEval(k,kv)), chAll:()=>deep(chAll()), drill:s=>deep(makeDrill(s)),
  exits:()=>{const e=EXITS(); return deep({transfer:e.transfer.map(q=>({q:q.q,o:q.o,a:q.a,why:q.why}))});},
  consts:Object.freeze({O,D,NX,NU,POUT,OLO,OHI,NCH,CHF,SIG,SEED,CH_SEED,UGPU,CH_BAR,CH_USERS,CH_CTX,RBITS:{...RBITS},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,BASE:{...BASE},PRESETS:PRESETS.map(p=>p.id),ALLK:ALLK.slice(),CHK:CHK.slice()}),
  state:()=>({...S,key:keyOf(),mode,step,pred,predShown,observed:{...observed},anim:{on:AN.on,ph:AN.ph,paused:AN.paused,key:AN.key},watched:[...watched],plays:nPlays,speed,chW,chKV,chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
