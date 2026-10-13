/* number-formats (the root "ML basics", world roots): Why does training use BF16 and not FP16?
   Fidelity: Formula. Every stored value is the exact round-to-nearest-even of the shown number into the format's bit layout.
   - Formats: FP32 1-8-23, BF16 1-8-7, FP16 1-5-10 (IEEE: the all-ones exponent is Inf / NaN); FP8 E4M3 1-4-3 (OCP "E4M3FN": no Inf,
     only S.1111.111 is NaN, so the top exponent holds numbers up to 448); FP8 E5M2 1-5-2 (IEEE-like); FP4 E2M1 1-2-1 (OCP MX: no Inf, no
     NaN, values 0, 0.5, 1, 1.5, 2, 3, 4, 6).
   - quant(x, f): quantum 2^(max(floor(log2|x|), emin) − m), ties to even; a result above the largest value overflows; a non-zero value
     that rounds to 0 is lost. An optional scale (FP8 / FP4 rows): stored = round(x ÷ scale) × scale, scale = block max ÷ format max.
   - Recipes (part 5, the challenge): the gradient is multiplied by the loss scale, stored in the gradient format, divided back in FP32;
     the update goes to the FP32 master copy (exact FP32 sum) or to the 16-bit weight (rounded again).
   Words (EN / ZH): sign 符号位; exponent 指数位; mantissa 尾数位; range 范围; gap 间隔; lost 丢失 (rounded to 0, or an update rounded away);
   overflow 溢出; master copy 主副本; loss scaling 损失缩放 (loss scaling); part / 部分.
   window.__nf is a read-only hook for tests/number-formats_numbers.py and tests/number-formats_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='number-formats';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const FM={
 fp32:{id:'fp32',n:'FP32',e:8,m:23,b:127,k:'ieee'},
 bf16:{id:'bf16',n:'BF16',e:8,m:7,b:127,k:'ieee'},
 fp16:{id:'fp16',n:'FP16',e:5,m:10,b:15,k:'ieee'},
 e4m3:{id:'e4m3',n:'FP8 E4M3',e:4,m:3,b:7,k:'fn'},
 e5m2:{id:'e5m2',n:'FP8 E5M2',e:5,m:2,b:15,k:'ieee'},
 fp4:{id:'fp4',n:'FP4 E2M1',e:2,m:1,b:1,k:'all'}};
Object.values(FM).forEach(f=>{f.emin=1-f.b; f.emax=(f.k==='ieee'?2**f.e-2:2**f.e-1)-f.b; f.top=f.k==='fn'?2-2*2**-f.m:2-2**-f.m;
  f.max=f.top*2**f.emax; f.minN=2**f.emin; f.minS=2**(f.emin-f.m); f.gap1=2**-f.m; f.bits=1+f.e+f.m;});
const ALL=['fp32','bf16','fp16','e4m3','e5m2','fp4'], MAIN=['fp32','bf16','fp16'], SMALL={e4m3:1,e5m2:1,fp4:1};
const fl2=a=>{let e=Math.floor(Math.log2(a)); if(2**e>a) e--; if(2**(e+1)<=a) e++; return e;};
/* round-to-nearest-even of x (a float64) into format f; sc: an optional scale (stored = round(x ÷ sc) × sc) */
function quant(x,fid,sc){
  const f=FM[fid], s=sc||1, y=x/s, a=Math.abs(y), neg=y<0?1:0;
  if(a===0) return {f:fid,x,v:0,kind:'zero',S:neg,E:0,M:0,sc:s};
  let ex=fl2(a); if(ex<f.emin) ex=f.emin;
  const q=2**(ex-f.m), n=a/q, lo=Math.floor(n), d=n-lo, k=d>0.5?lo+1:(d<0.5?lo:(lo%2===0?lo:lo+1)), v=k*q;
  if(v>f.max){if(f.k==='all') return {f:fid,x,v:(neg?-f.max:f.max)*s,kind:'over',S:neg,E:2**f.e-1,M:2**f.m-1,sc:s,clamp:true};   // FP4: no Inf or NaN, OCP MX saturates to ±6
    return {f:fid,x,v:f.k==='ieee'?(neg?-Infinity:Infinity):NaN,kind:'over',S:neg,E:2**f.e-1,M:f.k==='fn'?2**f.m-1:0,sc:s};}
  if(v===0) return {f:fid,x,v:0,kind:'lost',S:neg,E:0,M:0,sc:s};
  let E,M,kind;
  if(v<f.minN){E=0; M=Math.round(v/f.minS); kind='sub';}
  else {const ev=fl2(v); E=ev+f.b; M=Math.round((v/2**ev-1)*2**f.m); kind='norm';}
  return {f:fid,x,v:(neg?-v:v)*s,kind,S:neg,E,M,sc:s};
}
const relErr=r=>r.kind==='lost'||r.kind==='over'?null:(r.x===0?0:Math.abs(r.v-r.x)/Math.abs(r.x));
const bad=r=>r.kind==='lost'||r.kind==='over';
/* the block scale for FP8 / FP4 rows: the block's largest value is blk × the value; scale = that ÷ the format's largest value */
const BLKS=[0,10,100,1000];
const scaleOf=(x,fid,blk)=>blk&&SMALL[fid]?Math.abs(x)*blk/FM[fid].max:1;
const store=(x,fid,blk)=>quant(x,fid,scaleOf(x,fid,blk));
/* OCP MX's shared scale (Scenario 4's comparison): the largest power of two ≤ the block's largest value ÷ the element type's largest power of two */
const MXE={e4m3:8,fp4:2};
const mxStore=(x,fid,blk)=>quant(x,fid,2**(fl2(Math.abs(x)*blk)-MXE[fid]));
/* slider stops: the value 1-2-5 from 1e-10 to 5e5; the update 1-3 from 1e-5 to 1e-2 */
const XS=[]; for(let d=-10;d<=5;d++) [1,2,5].forEach(m=>XS.push(parseFloat(`${m}e${d}`)));
const US=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2];
const X_W=XS.indexOf(0.02), X_G=XS.indexOf(1e-8), U_0=US.indexOf(3e-4);
/* recipes: weight and gradient formats, loss scale, FP32 master copy */
const LSV=65536;
const RC={fp32:{w:'fp32',g:'fp32',ls:1,ms:false},fp16:{w:'fp16',g:'fp16',ls:1,ms:false},fp16ls:{w:'fp16',g:'fp16',ls:LSV,ms:false},
  fp16m:{w:'fp16',g:'fp16',ls:1,ms:true},fp16lsm:{w:'fp16',g:'fp16',ls:LSV,ms:true},bf16:{w:'bf16',g:'bf16',ls:1,ms:false},bf16m:{w:'bf16',g:'bf16',ls:1,ms:true}};
const fr=Math.fround;
/* one training step on the stated tensors: the smallest gradient (lost?), the largest (overflow?), the update (kept?) */
function recipe(r,T){
  const g=quant(T.g*r.ls,r.g), gUsed=bad(g)||g.v===0?0:fr(fr(g.v)/r.ls), gBig=quant(T.gmax*r.ls,r.g);
  const step=w=>{const a=r.ms?fr(w):quant(w,r.w).v; return [a,r.ms?fr(a+fr(T.u)):quant(a+T.u,r.w).v];};
  const [w0,w1]=step(T.w), [v0,v1]=T.w2!=null?step(T.w2):[0,1];
  const kept=w1!==w0, kept2=v1!==v0, sixteen=FM[r.w].bits===16&&FM[r.g].bits===16;
  const bytes=FM[r.w].bits/8+FM[r.g].bits/8+(r.ms?4:0), extras=(r.ls>1?1:0)+(r.ms?1:0);
  const ok=sixteen&&gBig.kind!=='over'&&!bad(g)&&g.v!==0&&kept&&kept2;
  return {r,g,gUsed,gBig,over:gBig.kind==='over',lost:bad(g)||g.v===0,w0,w1,kept,v0,v1,kept2,sixteen,bytes,extras,ok};
}
/* the weight after n updates: the FP32 master, its 16-bit copy, and a 16-bit weight with no master copy */
function series(r,T,n){
  const m=[fr(T.w)], c=[quant(m[0],r.w).v], o=[quant(T.w,r.w).v];
  for(let k=1;k<=n;k++){m.push(fr(m[k-1]+fr(T.u))); c.push(quant(m[k],r.w).v); o.push(quant(o[k-1]+T.u,r.w).v);}
  return {m,c,o};
}
/* The lab assumes each Adam step moves a weight by exactly the learning rate, whatever its size: the typical size of an Adam step
   (Kingma and Ba 2015, §2.1: m̂/√v̂ is usually about ±1 or smaller, at most (1 − β1)/√(1 − β2) ≈ 3.2 for the default β).
   Part 5 / Scenarios: an RMSNorm gain of 1.0 at the peak learning rate 3e-4 (Llama 2 7B's peak).
   The challenge: Llama 2 7B late in pretraining, its cosine schedule down to 10% of the peak (3e-5): weights near 0.02 and
   RMSNorm gains near 1.0, both moved by 3e-5. BF16 loses it on both (half gaps 6.1e-5 and 3.9e-3); FP16 keeps it on 0.02 (half
   gap 7.6e-6) but loses it on 1.0 (4.9e-4). */
const T5=u=>({g:1e-8,gmax:1e-3,w:1,u});
const LR_PEAK=3e-4, LR_END=+(LR_PEAK*0.1).toPrecision(12);   // 10% of the peak: 3e-5
const CHT={g:1e-8,gmax:1e-3,w:0.02,u:LR_END,w2:1};
const CHO=['fp32','fp16','fp16lsm','bf16','bf16m'];
const chEval=id=>recipe(RC[id],CHT);
function chBest(){let b=null; CHO.forEach(id=>{const e=chEval(id); if(e.ok&&(!b||e.extras<b.extras||(e.extras===b.extras&&e.bytes<b.bytes))) b={id,extras:e.extras,bytes:e.bytes};}); return b;}
const LV5=['fp16','fp16ls','fp16m','fp16lsm'];

/* =====================================================================
   formatting
   ===================================================================== */
const NBH='‑', MINUS='−';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fint=A.fmt.int;
const sci=(v,sig)=>{const [m,e]=Math.abs(v).toExponential(sig-1).split('e'); return (v<0?MINUS:'')+m+'e'+(+e<0?NBH+(-e):(+e));};
/* an input value as typed: 1e‑8, 2e‑10, 0.02, 500,000 */
const fin=x=>{if(x>=1e-3&&x<1e6) return x>=1000?fint(x):String(+x.toPrecision(6)); return sci(x,1);};
/* a stored value: 8 significant digits in plain decimal (trailing zeros dropped), else 4 in scientific notation */
function fv(v){if(v===0) return '0'; if(v===Infinity) return 'Inf'; if(v===-Infinity) return MINUS+'Inf'; if(Number.isNaN(v)) return 'NaN';
  const a=Math.abs(v); if(a>=1e-3&&a<1e6){const s=a.toPrecision(8); const t=s.indexOf('.')>=0?s.replace(/0+$/,'').replace(/\.$/,''):s; const [i,dd]=t.split('.'); return (v<0?MINUS:'')+i.replace(/\B(?=(\d{3})+(?!\d))/g,',')+(dd?'.'+dd:'');}
  return sci(v,4);}
const fpc=e=>{if(e===0) return tx('exact','精确'); const p=e*100; if(p<0.01) return '<0.01%'; return (p<0.1?p.toPrecision(2):(p<10?p.toFixed(2):p.toFixed(1)))+'%';};
const fe=r=>bad(r)?(r.kind==='over'?tx('overflow','溢出'):tx('lost','丢失')):fpc(relErr(r));
const pow2=e=>A.ui.math(`2^{${e<0?'-'+(-e):e}}`);   // typeset (R6.29, H1 2026-10-09)
const fname=fid=>FM[fid].n;
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
const M8='Llama‑3.1‑8B', M2='Llama 2 7B';

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={xi:X_W,ui:U_0,view:'a',f16:'fp16',ls:0,ms:0,blk:0};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seenX=new Set(), seenU=new Set(), lv='fp16', lvTried=new Set(['fp16']), remView='a';
let saved=null, chR='fp32', chSubmitted=false, chReveal=false, sbTouched=false;
let drillSeed=2, drillOpen=false, annT=null, userAct=false, wantObs=false;
const shownPrev={};
const X=()=>XS[S.xi], U=()=>US[S.ui];
const curRecipe=()=>{if(mode==='challenge') return RC[chR]; if(mode==='guided') return RC[lv]; const f=S.f16; return {w:f,g:f,ls:S.ls?LSV:1,ms:!!S.ms};};
const recipeName=r=>{const b=r.w==='fp32'?tx('All FP32','全部 FP32'):FM[r.w].n; const p=[b]; if(r.ls>1) p.push(tx('loss scaling','loss scaling')); if(r.ms) p.push(tx('FP32 master copy','FP32 主副本')); return p.join(' + ');};

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 sign:()=>tx('The sign bit: 0 for a positive number, 1 for a negative one.','符号位：正数是 0，负数是 1。'),
 exp:()=>tx('The exponent bits set roughly how large the number is: which power of two it sits near. More exponent bits reach both larger numbers and numbers closer to 0: they set the range.',
            '指数位决定这个数大概在哪个量级，也就是靠近 2 的几次幂。指数位越多，能表示的数就越大，也能越接近 0：指数位决定范围。'),
 man:()=>tx('The mantissa bits pick a point between that power of two and the next one: m bits split the stretch into 2ᵐ equal steps. More mantissa bits give finer steps: they set the precision.',
            '尾数位在这个 2 的幂和下一个 2 的幂之间选一个点：m 个尾数位把这一段分成 2ᵐ 个相等的台阶。尾数位越多，台阶越细：它们决定精度。'),
 float:()=>tx('A floating-point number is stored in three parts. The sign bit says plus or minus, the exponent bits say which power of two the number sits near, and the mantissa bits say where it lies between that power and the next. FP32 uses 32 bits, BF16 and FP16 use 16, FP8 8 and FP4 4.',
              '浮点数分三部分存一个数：符号位定正负，指数位定它靠近 2 的几次幂，尾数位定它在这个幂和下一个幂之间的哪个位置。FP32 用 32 位，BF16 和 FP16 用 16 位，FP8 用 8 位，FP4 用 4 位。'),
 rne:()=>tx('Round to nearest, ties to even: the stored number is the representable number closest to the true one; exactly halfway, the one whose last mantissa bit is 0. It is the default rounding of IEEE 754 and of PyTorch casts.',
            '就近舍入、平局取偶：存下的是离真实值最近的可表示数；正好在中间时，取尾数最后一位是 0 的那个。这是 IEEE 754 和 PyTorch 类型转换的默认舍入方式。'),
 sub:()=>tx('Subnormal numbers are the numbers below a format’s smallest normal number that it can still store. The closer they get to 0, the fewer significant bits they keep, one bit less each time the value halves, until a value rounds to 0. How: a normal number has a leading 1 in front of its mantissa, so all its mantissa bits count. A subnormal number has the exponent field 0 and no leading 1, so its steps stay as wide as at the smallest normal number, and the smaller it is, the fewer bits count.',
            '次正规数（subnormal）是比最小的正规数还小、但格式仍然存得下的数。越接近 0，它们留下的有效位越少：数值每小一半，就少一位，最后舍入成 0。原理：正规数的尾数前面默认有一个 1，所有尾数位都有效。次正规数的指数字段是 0，前面也没有这个 1，所以台阶和最小的正规数那里一样宽；数越小，有效的位越少。'),
 grad:()=>tx('A gradient says how much the loss changes when one parameter changes a little. The optimizer turns it into this step’s update: SGD multiplies it by the learning rate; Adam first divides it by the typical size of its recent values, so a step is usually about one learning rate.',
             '梯度说明一个参数变一点时 loss 变多少。优化器用它算出这一步的更新量：SGD 直接乘上学习率；Adam 先把它除以最近梯度的典型大小，所以一步通常约为一个学习率。'),
 upd:()=>tx('The update is what one optimizer step adds to a weight. SGD’s step is the learning rate times the gradient. Adam first divides the gradient by its typical recent size, so a typical Adam step is about the learning rate, almost regardless of the gradient’s size. At a learning rate of 3e‑4, for example, one step changes a weight by about 3e‑4. Why that is so: see the lab Optimizers.',
            '更新量是优化器一步加到权重上的量。SGD 的一步是学习率乘以梯度。Adam 会先把梯度除以它最近的典型大小，所以 Adam 典型的一步约为一个学习率，基本不看梯度有多大。比如学习率是 3e‑4 时，一步让权重变化约 3e‑4。为什么会这样，见“优化器”这个 Lab。'),
 updRule:()=>tx('A format loses an update smaller than half its gap at that weight, and the gap grows with the weight: BF16 loses updates below about 0.2 to 0.4% of the weight, FP16 below about 0.025 to 0.05%. The guided parts add the update to an RMSNorm gain, which starts at 1.0; most other weights are near 0.02, where the gaps are about 50 times finer.',
                '比这个权重处间隔的一半还小的更新，会被格式丢掉；间隔随权重变大：BF16 丢掉小于权重约 0.2% 到 0.4% 的更新，FP16 丢掉小于约 0.025% 到 0.05% 的。分步引导里，更新加在 RMSNorm 的 gain 上，它的初始值是 1.0；别的权重多在 0.02 左右，那里的间隔约细 50 倍。'),
 gain:()=>`${kt('rmsnorm')}${tx(' These learned scales are called gains here; they start at 1.0.','这里把这些学到的系数叫 gain，它们的初始值是 1.0。')}`,
 mixed:()=>tx('Mixed precision training stores and multiplies most tensors in a 16-bit format and keeps a few in FP32, where 16 bits are not enough.','混合精度训练把大多数张量用 16 位格式存储和相乘，只在 16 位不够用的地方保留 FP32。'),
 mx:()=>tx('The OCP Microscaling (MX) formats, a standard of the Open Compute Project: each block of 32 numbers in FP8, FP6 or FP4 shares one 8-bit power-of-two scale (E8M0).','OCP Microscaling（MX）格式是 Open Compute Project 的一个标准：FP8、FP6 或 FP4 的数每 32 个一块，共用一个 8 位的 2 的幂缩放系数（E8M0）。'),
 ls:()=>tx('Loss scaling multiplies the loss by S before the backward pass, so every gradient comes out S times larger and stays inside FP16’s range; the gradients are divided by S in FP32 before the update. Here S = 65,536 = 2¹⁶, PyTorch GradScaler’s starting scale: it is a setting, not computed.',
           'loss scaling（损失缩放）在反向传播之前把 loss 乘以 S，于是每个梯度都放大 S 倍，留在 FP16 的范围里；更新之前再在 FP32 里除以 S。这里 S = 65,536 = 2¹⁶，是 PyTorch GradScaler 的初始值：它是设定的，不是算出来的。'),
 master:()=>tx('An FP32 master copy keeps every weight in FP32 next to the 16-bit copy used in the matmuls. The optimizer adds each update to the FP32 copy, then rounds it to 16 bits for the next forward pass.',
               'FP32 主副本在 16 位的副本旁边，用 FP32 再存一份每个权重。优化器把每次更新加到 FP32 副本上，再把它舍入成 16 位，给下一次前向传播用。'),
 illus:()=>tx('Illustrative sizes: weights near 0.02, gradients from 1e‑9 to 1e‑3, Adam updates of about the learning rate (3e‑4 at its peak), and activation outliers of 1e3 to 1e4 in large models. They differ from model to model.',
              '示意的大小：权重在 0.02 左右，梯度从 1e‑9 到 1e‑3，Adam 更新量约一个学习率（峰值时 3e‑4），大模型里的激活值离群点在 1e3 到 1e4。不同模型各不相同。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'kept',t:{en:'About 1e‑8, slightly rounded',zh:'约 1e‑8，略有舍入'}},
  {v:'zero',t:{en:'Exactly 0: the gradient is lost',zh:'正好是 0：梯度丢了'}},
  {v:'inf',t:{en:'Infinity: FP16 overflows',zh:'无穷大：FP16 溢出'}}];
const PRED_RIGHT='zero';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const preLock=()=>mode==='guided'&&step<SIX('range');
const xCtl=(id,inline)=>A.ui.range({id,key:'x',label:tx('Value to store','要存的数'),min:0,max:XS.length-1,value:S.xi,out:fin(X()),inline});
const uCtl=(id,inline)=>A.ui.range({id,key:'u',label:tx('Update added to a weight of 1.0','更新量（加到大小为 1.0 的权重上）'),min:0,max:US.length-1,value:S.ui,out:fin(U()),inline});
const LVT={fp16:{en:'FP16 only (initial setup)',zh:'只用 FP16（初始设置）'},fp16ls:{en:'+ loss scaling',zh:'+ loss scaling'},fp16m:{en:'+ FP32 master copy',zh:'+ FP32 主副本'},fp16lsm:{en:'+ both',zh:'+ 两个都用'}};
const LVD={fp16:{en:'gradient and weight stored in FP16',zh:'梯度和权重都存成 FP16'},fp16ls:{en:'the loss × 65,536 (GradScaler’s starting value) before the backward pass',zh:'反向传播之前把 loss 乘以 65,536（GradScaler 的初始值）'},
  fp16m:{en:'updates go to an FP32 copy of the weight',zh:'更新加到权重的 FP32 副本上'},fp16lsm:{en:'loss scaling and an FP32 master copy',zh:'loss scaling 加 FP32 主副本'}};
const lvRes=id=>{const e=recipe(RC[id],T5(3e-4)); return tx(`gradient ${e.lost?'✗':'✓'}, update ${e.kept?'✓':'✗'}`,`梯度 ${e.lost?'✗':'✓'}，更新 ${e.kept?'✓':'✗'}`);};
const lvCtl=()=>`<div class="nf-lv" id="nf-ic-lv">`+A.ui.levers({aria:tx('The fix, pick one','修法，选一种'),value:lv,options:LV5.map(id=>({v:id,title:L(LVT[id]),desc:L(LVD[id]),result:esc(lvTried.has(id)||id===lv?lvRes(id):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'a',focus:'bits',
  lead:()=>tx('One weight, three formats: what does each keep, and what would a tiny gradient keep?','同一个权重，三种格式：各自存下什么？一个很小的梯度又会剩下什么？'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>You train ${M8} in ${DEF('mixed precision',TIP.mixed())}. Every number is stored as a ${DEF('floating-point number',TIP.float())}: a ${DEF('sign',TIP.sign())}, ${DEF('exponent bits',TIP.exp())} and ${DEF('mantissa bits',TIP.man())}. The ${J('bits','chart')} shows a typical weight of 0.02 stored in FP32, BF16 and FP16.</p><p>BF16 and FP16 both take 2 bytes; they only split the 16 bits differently. Training also stores two much smaller numbers. One is a ${DEF('gradient',TIP.grad())} of 1e‑8. The other is an ${DEF('update',TIP.upd())} of 0.0003, added to a weight of 1.0 such as an ${DEF('RMSNorm gain',TIP.gain())}. The update is 0.0003 because at a learning rate of 3e‑4, one Adam step changes a weight by about the learning rate. Which formats keep these two numbers?</p>`,
              `<p>你在用${DEF('混合精度',TIP.mixed())}训练 ${M8}。每个数都存成一个${DEF('浮点数',TIP.float())}，由三部分组成：一个${DEF('符号位',TIP.sign())}、几个${DEF('指数位',TIP.exp())}和几个${DEF('尾数位',TIP.man())}。${J('bits','图里')}是一个大小为 0.02 的典型权重，分别存成 FP32、BF16 和 FP16 的样子。</p><p>BF16 和 FP16 都占 2 字节，只是 16 位的分法不同。训练时还要存两个小得多的数。一个是 1e‑8 的${DEF('梯度',TIP.grad())}。另一个是 0.0003 的${DEF('更新量',TIP.upd())}，要加到一个大小为 1.0 的权重上，比如 ${DEF('RMSNorm 的 gain',TIP.gain())}。更新量取 0.0003，是因为学习率为 3e‑4 时，Adam 一步大约让权重变化一个学习率。哪些格式留得住这两个数？</p>`)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},view:'a',focus:'bits',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: a gradient of 1e‑8 stored in FP16 becomes what?','先猜：1e‑8 的梯度存成 FP16 会变成什么？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>FP16 has 5 exponent bits and 10 mantissa bits. The backward pass produces a gradient of 1e‑8 and stores it in FP16. What is stored?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>FP16 有 5 个指数位和 10 个尾数位。反向传播算出一个 1e‑8 的梯度，存成 FP16。存下的是什么？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'range',title:{en:'Exponent bits set the range',zh:'指数位决定范围'},view:'a',focus:'range',
  lead:()=>tx('Drag the value down to 1e‑8 and up past 65,504: watch where each strip ends.','把数拖到 1e‑8，再拖过 65,504：看每条范围在哪里结束。'),
  enter:()=>({...BASE}),expect:()=>seenX.has(X_G),showMe:{xi:X_G},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Each strip in the chart is the ${J('range','range')} one format can store: its left end is the smallest number the format can store, its right end the largest (the axis is a log scale).</p><p>Almost all numbers a format can store are <b>normal numbers</b>: each keeps the same number of significant digits. The ${DEF('hatched stretch',TIP.sub())} at the left end of each strip holds the <b>subnormal numbers</b>: numbers below the smallest normal number that the format can still store. A number below the smallest normal number does not turn into 0 at once: the closer it gets to 0, the fewer significant digits it keeps, until it rounds to 0.</p>`,
              `<p>图里每一条横条是一种格式能存的${J('range','范围')}：左端是它能存的最小的数，右端是最大的数（横轴是对数刻度）。</p><p>格式能存的数里，绝大多数是<b>正规数</b>：每个正规数的有效数字一样多。每条左端${DEF('斜纹那一段',TIP.sub())}是<b>次正规数</b>（subnormal）：比最小的正规数还小、但还存得下的数。数小于最小的正规数时，不会一下子变成 0，而是越接近 0，留下的有效数字越少，最后舍入成 0。</p>`)+xCtl('nf-ic-x',true),
  obs:()=>obsRange()},
 {id:'prec',title:{en:'Mantissa bits set the precision',zh:'尾数位决定精度'},view:'b',focus:'zoom',
  lead:()=>tx('Add an update to a weight of 1.0: find one that BF16 loses and FP16 keeps.','给大小为 1.0 的权重加上更新量：找一个 BF16 丢掉、FP16 留住的更新量。'),
  enter:()=>({...BASE,view:'b'}),expect:()=>[...seenU].some(i=>{const u=US[i]; return quant(1+u,'bf16').v===1&&quant(1+u,'fp16').v!==1;}),showMe:{ui:US.indexOf(1e-3)},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Near 1.0, a format can only store the ${J('zoom','ticks in the chart')}: the ${DEF('gap',(tx(`The gap between 1.0 and the next number a format can store is ${A.ui.math('2^{-m}')} for m mantissa bits: `,`1.0 和下一个能存的数之间的间隔是 ${A.ui.math('2^{-m}')}（m 是尾数位数）：`)+tx('BF16 2⁻⁷ = 0.0078125, FP16 2⁻¹⁰ ≈ 0.00098, FP32 2⁻²³ ≈ 1.2e‑7. Near a weight of size w, the gap is about w times the gap near 1.0.','BF16 是 2⁻⁷ = 0.0078125，FP16 是 2⁻¹⁰ ≈ 0.00098，FP32 是 2⁻²³ ≈ 1.2e‑7。在大小为 w 的权重附近，间隔大约是 1.0 附近的 w 倍。')))} between neighbours is ${A.ui.math('2^{-m}')} for m mantissa bits. The sum 1.0 + update is ${DEF('rounded',TIP.rne())} to the nearest tick.</p>`,
              `<p>在 1.0 附近，一种格式只能存${J('zoom','图里这些刻度')}上的数。相邻两个刻度的${DEF('间隔',(tx(`The gap between 1.0 and the next number a format can store is ${A.ui.math('2^{-m}')} for m mantissa bits: `,`1.0 和下一个能存的数之间的间隔是 ${A.ui.math('2^{-m}')}（m 是尾数位数）：`)+tx('BF16 2⁻⁷ = 0.0078125, FP16 2⁻¹⁰ ≈ 0.00098, FP32 2⁻²³ ≈ 1.2e‑7. Near a weight of size w, the gap is about w times the gap near 1.0.','BF16 是 2⁻⁷ = 0.0078125，FP16 是 2⁻¹⁰ ≈ 0.00098，FP32 是 2⁻²³ ≈ 1.2e‑7。在大小为 w 的权重附近，间隔大约是 1.0 附近的 w 倍。')))}是 ${A.ui.math('2^{-m}')}，m 是尾数位数。1.0 加上更新量之后，结果要${DEF('舍入',TIP.rne())}到最近的刻度上。</p>`)+uCtl('nf-ic-u',true),
  obs:()=>obsPrec()},
 {id:'fix',title:{en:'The fixes: loss scaling and a master copy',zh:'两个修法：loss scaling 和主副本'},view:'c',focus:'recipe',
  lead:()=>tx('Start from FP16 only, then add each fix: which one saves the gradient, which the update?','从只用 FP16 出发，再加上每个修法：哪个救回梯度，哪个救回更新？'),
  enter:()=>{lv='fp16'; lvTried=new Set(['fp16']); return {...BASE,view:'c'};},expect:()=>lvTried.has('fp16lsm'),showMe:()=>setLv('fp16lsm'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Take one training step with everything stored in FP16: the gradient is 1e‑8, and an update of 0.0003 is added to a weight of size 1.0. Two fixes are standard: ${DEF('loss scaling',TIP.ls())} and an ${DEF('FP32 master copy',TIP.master())}. Pick a fix and see whether the gradient and the update survive. To use both fixes, pick “+ both”.</p>`,
              `<p>看一个全用 FP16 存的训练步。它的梯度是 1e‑8，更新量 0.0003 要加到一个大小为 1.0 的权重上。常用的修法有两个：${DEF('loss scaling',TIP.ls())} 和 ${DEF('FP32 主副本',TIP.master())}。选一种修法，看看梯度和更新能不能保住。想两个修法都用，就选“+ 两个都用”。</p>`)+lvCtl(),
  obs:()=>obsFix()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>remView,focus:()=>({a:'range',b:'zoom',c:'recipe'}[remView]||''),
  lead:()=>tx('Exponent bits set the range, mantissa bits the precision; BF16 trades precision for FP32’s range.','指数位决定范围，尾数位决定精度；BF16 用精度换来 FP32 的范围。'),
  enter:()=>{remView='a'; return {...BASE,xi:X_G};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
const focusOf=s=>typeof s.focus==='function'?s.focus():(s.focus||'');
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction for 1e‑8, “${pt}”, was right.</span></p>`,`<p><span class="okc">你对 1e‑8 的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>For 1e‑8 you predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>对 1e‑8，你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
/* part 3: the value against the three ranges */
function obsRange(){
  const x=X(), r=MAIN.map(f=>quant(x,f)), [r32,rb,rh]=r, h=FM.fp16;
  if(S.xi===X_W) return tx('<p><b>At 0.02 all three formats keep the weight.</b> Drag the value down to 1e‑8 to check your prediction.</p>','<p><b>0.02 时三种格式都留住了这个权重。</b>把数拖到 1e‑8，核对你的预测。</p>');
  let res;
  if(rh.kind==='lost') res=tx(`<p><b>${fin(x)} in FP16 is 0: the value is lost. BF16 stores ${fv(rb.v)} and FP32 ${fv(r32.v)}.</b></p>`,`<p><b>${fin(x)} 存成 FP16 是 0：这个数丢了。BF16 存下 ${fv(rb.v)}，FP32 存下 ${fv(r32.v)}。</b></p>`);
  else if(rh.kind==='over') res=tx(`<p><b>${fin(x)} in FP16 overflows to Inf: FP16 stops at ${fint(h.max)}. BF16 stores ${fv(rb.v)}.</b></p>`,`<p><b>${fin(x)} 存成 FP16 溢出成 Inf：FP16 最大只到 ${fint(h.max)}。BF16 存下 ${fv(rb.v)}。</b></p>`);
  else if(rh.kind==='sub') res=tx(`<p><b>${fin(x)} in FP16 is ${fv(rh.v)}, off by ${fe(rh)}. It sits among FP16’s subnormal numbers, where few significant digits are left. BF16 stores ${fv(rb.v)} (off by ${fe(rb)}).</b></p>`,`<p><b>${fin(x)} 存成 FP16 是 ${fv(rh.v)}，误差 ${fe(rh)}。它落在 FP16 的次正规数里，有效数字已经不多了。BF16 存下 ${fv(rb.v)}（误差 ${fe(rb)}）。</b></p>`);
  else res=tx(`<p><b>${fin(x)} fits all three: FP16 is off by ${fe(rh)}, BF16 by ${fe(rb)}.</b></p>`,`<p><b>${fin(x)} 三种格式都放得下：FP16 误差 ${fe(rh)}，BF16 误差 ${fe(rb)}。</b></p>`);
  const why=tx(`<p>Why: FP16 has only 5 exponent bits, so its smallest positive number is about 6e‑8 (already a subnormal one). 1e‑8 is less than half of it, so it can only round to 0. BF16 has 8 exponent bits, like FP32, and reaches down to about 1e‑40, so 1e‑8 is well inside its range.</p>`,
               `<p>原因：FP16 只有 5 个指数位，能存的最小正数约为 6e‑8（已经是次正规数）。1e‑8 不到它的一半，只能舍入成 0。BF16 和 FP32 一样有 8 个指数位，能存到 1e‑40 这个量级，1e‑8 远在它的范围之内。</p>`);
  return res+(seenX.has(X_G)?verdict():tx('<p class="small">Drag it to 1e‑8 to check your prediction.</p>','<p class="small">拖到 1e‑8，核对你的预测。</p>'))+why;
}
/* part 4: 1.0 + update in the three formats */
function obsPrec(){
  const u=U(), b=quant(1+u,'bf16'), h=quant(1+u,'fp16'), lostB=b.v===1, lostH=h.v===1;
  const hb=FM.bf16.gap1/2, hh=FM.fp16.gap1/2;
  let res;
  if(lostB&&lostH) res=tx(`<p><b>1.0 + ${fin(u)} is 1 in both BF16 and FP16: the update is lost. FP32 keeps ${fv(quant(1+u,'fp32').v)}.</b></p>`,`<p><b>1.0 + ${fin(u)} 在 BF16 和 FP16 里都是 1：更新丢了。FP32 留下 ${fv(quant(1+u,'fp32').v)}。</b></p>`);
  else if(lostB) res=tx(`<p><b>1.0 + ${fin(u)} is 1 in BF16, so the update is lost; FP16 keeps it as ${fv(h.v)}.</b></p>`,`<p><b>1.0 + ${fin(u)} 在 BF16 里是 1，更新丢了；FP16 把它存成 ${fv(h.v)}，留住了。</b></p>`);
  else res=tx(`<p><b>1.0 + ${fin(u)} survives in both: BF16 stores ${fv(b.v)}, FP16 ${fv(h.v)}.</b></p>`,`<p><b>1.0 + ${fin(u)} 两种都留住了：BF16 存成 ${fv(b.v)}，FP16 存成 ${fv(h.v)}。</b></p>`);
  const why=tx(`<p>Why: an update smaller than half the gap rounds back to the old weight. BF16 has only 7 mantissa bits: an update below about ${fixd(hb,4)} added to 1.0 still gives 1.0. FP16 has 10 mantissa bits, so an update is lost only below about ${fixd(hh,5)}. So BF16 is the less precise of the two.</p>`,
               `<p>原因：比半个间隔还小的更新，会舍入回原来的权重。BF16 只有 7 个尾数位：更新量小于约 ${fixd(hb,4)} 时，加到 1.0 上还是 1.0。FP16 有 10 个尾数位，更新量要小于约 ${fixd(hh,5)} 才会丢。所以 BF16 的精度比 FP16 低。</p>`);
  const hint=!lostB||lostH?tx('<p class="small">Find an update that BF16 loses but FP16 keeps.</p>','<p class="small">找一个 BF16 丢掉、FP16 留住的更新量。</p>'):'';
  return res+hint+why;
}
/* part 5: the chosen fix */
function obsFix(){
  const e=recipe(RC[lv],T5(3e-4));
  if(lv==='fp16') return tx(`<p><b>FP16 only: the gradient is stored as 0 and the update rounds away, so this weight never moves.</b> Pick a fix.</p>`,`<p><b>只用 FP16：梯度存成 0，更新也被舍入掉，这个权重永远不动。</b>选一个修法。</p>`);
  if(lv==='fp16ls') return tx(`<p><b>Loss scaling saves the gradient: 1e‑8 × 65,536 is stored as ${fv(e.g.v)}, and dividing in FP32 gives back ${fv(e.gUsed)}. The update is still lost.</b></p><p>Why: after scaling, the gradient is back inside FP16’s range. But the weight is still FP16, and 0.0003 is less than half its gap at 1.0.</p>`,
                                `<p><b>loss scaling 救回了梯度：1e‑8 × 65,536 存成 ${fv(e.g.v)}，在 FP32 里除回去得到 ${fv(e.gUsed)}。更新仍然丢了。</b></p><p>原因：放大以后，梯度回到了 FP16 的范围里。可是权重还是 FP16，而 0.0003 不到 1.0 处间隔的一半。</p>`);
  if(lv==='fp16m') return tx(`<p><b>The FP32 master copy saves the update: it becomes ${fv(e.w1)}. The gradient is still stored as 0.</b></p><p>Why: FP32’s gap at 1.0 is only about 1.2e‑7, so every update is kept. But the master copy only holds the weights; it does nothing for the gradient’s range.</p>`,
                               `<p><b>FP32 主副本救回了更新：权重变成 ${fv(e.w1)}。梯度仍然存成 0。</b></p><p>原因：FP32 在 1.0 处的间隔只有约 1.2e‑7，每个更新都留得下。但主副本只管权重，管不了梯度的范围。</p>`);
  return tx(`<p><b>Both fixes: the gradient comes back as ${fv(e.gUsed)} and the update makes the master copy ${fv(e.w1)}.</b></p><p>Why: each fix covers one failure. In the chart, the master copy climbs with every update while its FP16 copy follows in steps; a weight kept only in FP16 stays at 1.0. This is the recipe of Micikevicius et al. (2018).</p>`,
            `<p><b>两个修法都用：梯度回来了，是 ${fv(e.gUsed)}；更新让主副本变成 ${fv(e.w1)}。</b></p><p>原因：每个修法各管一种故障。图里，主副本每次更新都往上走，它的 FP16 副本一级一级跟上；只用 FP16 存的权重停在 1.0。这就是 Micikevicius 等（2018）的做法。</p>`);
}
/* part 6 */
function remBtn(k){const on=remView===k;
  const tip={a:tx('Shows each format’s range in the chart, with a gradient of 1e‑8.','在图里显示每种格式能存的范围，以及 1e‑8 这个梯度。'),b:tx('Shows the storable numbers near 1.0 in the chart, with an update of 0.001.','在图里显示 1.0 附近能存的数，更新量是 0.001。'),c:tx('Shows FP16 with both fixes in the chart: loss scaling and an FP32 master copy.','在图里显示 FP16 同时用 loss scaling 和 FP32 主副本的情况。')}[k];
  return ` <button type="button" class="linkbtn nf-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const b8=quant(1e-8,'bf16');
  const h=FM.fp16;
  return tx(`<ul class="nf-remlist"><li><b>Before storing a tensor in FP16, compare its smallest values with FP16’s smallest number, about ${sci(h.minS,1)}.</b> Anything smaller rounds to 0, so a gradient that small vanishes and its weight is not updated. BF16 has 8 exponent bits, like FP32, so it keeps gradients far smaller than that.${remBtn('a')}</li>`+
    `<li><b>Before adding an update to a 16-bit weight, look at how far apart the two storable numbers next to that weight are: the update must be more than half of that gap to survive.</b> Fewer mantissa bits mean a wider gap, so BF16 (7 mantissa bits) drops updates that FP16 (10 mantissa bits) still keeps. A weight that loses every update never moves, however long it trains.${remBtn('b')}</li>`+
    `<li><b>Pick the fix by the failure: a gradient too small to store needs loss scaling; an update below half a gap needs an FP32 master copy.</b> Loss scaling multiplies the loss, and with it every gradient, by a large factor before the backward pass, which lifts tiny gradients into FP16’s range. The master copy adds each update to an FP32 copy of the weight, where small updates survive. Mixed-precision training in FP16 uses both.${remBtn('c')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “BF16 is more precise than FP16.” It is less precise: it has 7 mantissa bits against FP16’s 10, so it loses small updates that FP16 keeps. BF16 wins on range, not precision.</p><p>✗ “Fewer bits only cost accuracy, never range.” FP16 gave up 3 exponent bits, so very small gradients round to 0, and its largest number is only 65,504.</p><p>✗ “FP8 is one format.” The standard FP8 has two: E4M3 has finer steps and a small range (largest 448); E5M2 has a large range and coarser steps (largest 57,344). Both need a scale per tensor or per block.</p></div>${A.shell.toScenarios()}`,
    `<ul class="nf-remlist"><li><b>把张量存成 FP16 之前，先拿它最小的值和 FP16 最小的数（约 ${sci(h.minS,1)}）比一比。</b>比它还小的数会被舍入成 0，这样的梯度就消失了，对应的权重也得不到更新。BF16 和 FP32 一样有 8 个指数位，比这小得多的梯度也存得下。${remBtn('a')}</li>`+
    `<li><b>把更新加到 16 位的权重上之前，先看这个权重附近相邻两个能存的数隔多远：更新要大于这个间隔的一半，才留得住。</b>尾数位越少，间隔越宽，所以 BF16（7 个尾数位）会丢掉 FP16（10 个尾数位）还留得住的更新。一个权重如果每次更新都被丢掉，训练再久它也不会动。${remBtn('b')}</li>`+
    `<li><b>哪种故障用哪种修法：梯度太小存不下，用 loss scaling；更新不到半个间隔，用 FP32 主副本。</b>loss scaling 在反向传播之前把 loss 乘上一个大数，每个梯度也跟着放大，太小的梯度就回到了 FP16 的范围里。主副本把每次更新加到权重的 FP32 副本上，小的更新也留得住。用 FP16 做混合精度训练时，两个修法都要用。${remBtn('c')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “BF16 比 FP16 更精确。”恰好相反。BF16 只有 7 个尾数位，FP16 有 10 个，所以 BF16 会丢掉 FP16 留得住的小更新。BF16 赢在范围，不在精度。</p><p>✗ “位数少了只损失精度，不影响范围。”FP16 少了 3 个指数位，所以很小的梯度会被舍入成 0，它最大的数也只有 65,504。</p><p>✗ “FP8 是一种格式。”标准的 FP8 有两种：E4M3 台阶细、范围小（最大 448），E5M2 范围大、台阶粗（最大 57,344）。两种都要按张量或按块配一个缩放系数。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 mp:'https://arxiv.org/abs/1710.03740', bf:'https://arxiv.org/abs/1905.12322', f8:'https://arxiv.org/abs/2209.05433',
 ds:'https://arxiv.org/abs/2412.19437', llama:'https://arxiv.org/abs/2407.21783', l2:'https://arxiv.org/abs/2307.09288', adam:'https://arxiv.org/abs/1412.6980',
 gs:'https://github.com/pytorch/pytorch/blob/main/torch/amp/grad_scaler.py', ac:'https://github.com/pytorch/pytorch/blob/main/torch/amp/autocast_mode.py',
 te:'https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py',
 mg:'https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer.py',
 mx:'https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf'
};
const PRESETS=[
 {id:'bf16',t:{en:'Pretrain an LLM in BF16',zh:'用 BF16 预训练大模型'},d:{en:'The usual format for large models today',zh:'如今大模型的常用格式'},s:{view:'a',f16:'bf16',ls:0,ms:0,ui:U_0,xi:X_G,blk:0}},
 {id:'fp16',t:{en:'FP16 with dynamic loss scaling',zh:'FP16 加动态 loss scaling'},d:{en:'GPUs without BF16, such as the V100',zh:'不支持 BF16 的 GPU，比如 V100'},s:{view:'c',f16:'fp16',ls:1,ms:1,ui:U_0,xi:X_G,blk:0}},
 {id:'fp8',t:{en:'FP8 training with fine-grained scaling',zh:'FP8 训练，细粒度缩放'},d:{en:'DeepSeek‑V3’s linear layers in FP8',zh:'DeepSeek‑V3 的线性层用 FP8'},s:{view:'a',xi:XS.indexOf(1e-4),blk:2,f16:'bf16',ls:0,ms:1,ui:U_0}},
 {id:'inf',t:{en:'FP8 and FP4 weights for inference',zh:'推理用 FP8 和 FP4 权重'},d:{en:'Fewer bytes per weight to serve a model',zh:'部署时每个权重占更少的字节'},s:{view:'a',xi:X_W,blk:1,f16:'bf16',ls:0,ms:1,ui:U_0}}
];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k]));
function scenHTML(i){
  const p=PRESETS[i];
  if(p.id==='bf16'){const x=X(), b=quant(x,'bf16'), lost=ALL.filter(id=>bad(quant(x,id))).map(fname), ls=tx(lost.length<3?lost.join(' and '):lost.slice(0,-1).join(', ')+' and '+lost[lost.length-1],lost.join('、'));
    return tx(`<p><b>BF16 keeps FP32’s 8 exponent bits. So even without loss scaling, a gradient of ${fin(x)} is stored as ${fv(b.v)}${lost.length?`. In ${ls}, it becomes 0`:''}.</b></p><p>${cite(SRC.bf,'Kalamkar et al. (2019)')} trained in BF16 without loss scaling, and the models came out as good as with FP32. Llama 3 405B was trained in BF16. Its gradients were accumulated in FP32 across micro-batches (${cite(SRC.llama,'Llama 3 paper')}, Section 3.3.2).</p>`,
              `<p><b>BF16 保留了 FP32 的 8 个指数位。所以即使不用 loss scaling，${fin(x)} 的梯度也能存成 ${fv(b.v)}${lost.length?`。换成 ${ls}，它会变成 0`:''}。</b></p><p>${cite(SRC.bf,'Kalamkar 等（2019）')}用 BF16 训练，不用 loss scaling，训练出的模型效果和 FP32 一样好。Llama 3 405B 用 BF16 训练。它的梯度在多个 micro-batch 之间用 FP32 累加（${cite(SRC.llama,'Llama 3 论文')}第 3.3.2 节）。</p>`);}
  if(p.id==='fp16'){const e=recipe({w:'fp16',g:'fp16',ls:LSV,ms:true},T5(U()));
    return tx(`<p><b>FP16 training needs loss scaling. With a scale of 65,536, the gradient of 1e‑8 is stored as ${fv(e.g.v)} and comes back as ${fv(e.gUsed)}. The FP32 master copy keeps the update.</b></p><p>PyTorch’s ${cite(SRC.gs,'<code>GradScaler</code>')} starts the scale at 65,536. When a gradient overflows, it halves the scale and skips that step. After 2,000 steps without an overflow, it doubles the scale. The recipe is from ${cite(SRC.mp,'Micikevicius et al. (2018)')}.</p>`,
              `<p><b>FP16 训练离不开 loss scaling。缩放系数是 65,536 时，1e‑8 的梯度存成 ${fv(e.g.v)}，除回去得到 ${fv(e.gUsed)}。FP32 主副本留住了更新。</b></p><p>PyTorch 的 ${cite(SRC.gs,'<code>GradScaler</code>')} 一开始把缩放系数设为 65,536。一旦有梯度溢出，它就把系数减半，并跳过这一步。连续 2,000 步没有溢出，它就把系数加倍。这个做法来自 ${cite(SRC.mp,'Micikevicius 等（2018）')}。</p>`);}
  if(p.id==='fp8'){const x=X(), a=store(x,'e4m3',BLKS[S.blk]), n=quant(x,'e4m3');
    return tx(`<p><b>FP8 E4M3 alone cannot hold ${fin(x)}: it rounds to ${fv(n.v)}. With a scale for its block, it is stored as ${fv(a.v)} (off by ${fe(a)}).</b></p><p>${cite(SRC.ds,'DeepSeek‑V3')} runs the matmuls of its linear layers in FP8 E4M3 (Section 3.3). Activations get one scale per 1 × 128 tile, and weights one per 128 × 128 block. Attention, the output head and the embedding stay in BF16 or FP32. Transformer Engine’s hybrid recipe instead uses E4M3 in the forward pass and E5M2 for gradients. Here the block’s largest value is ${fint(BLKS[S.blk])} times this one (${J('controls','Settings')}).</p>`,
              `<p><b>只用 FP8 E4M3 存不下 ${fin(x)}：它会被舍入成 ${fv(n.v)}。给它所在的块配一个缩放系数后，它存成 ${fv(a.v)}（误差 ${fe(a)}）。</b></p><p>${cite(SRC.ds,'DeepSeek‑V3')} 线性层里的矩阵乘法用 FP8 E4M3（第 3.3 节）。激活值每 1 × 128 个数配一个缩放系数，权重每 128 × 128 一块配一个。attention、输出层和 embedding 留在 BF16 或 FP32。Transformer Engine 的 hybrid 做法则是前向传播用 E4M3，梯度用 E5M2。这里，块里最大的数是这个数的 ${fint(BLKS[S.blk])} 倍（见“${J('controls','自由调参')}”）。</p>`);}
  const blk=BLKS[S.blk], x=X(), a=store(x,'fp4',blk), n=quant(x,'fp4'), e=store(x,'e4m3',blk), mx=blk?mxStore(x,'fp4',blk):a;
  const mxT=bad(mx)?tx('0, so the value is lost','0，这个数丢了'):tx(`${fv(mx.v)} (off by ${fe(mx)})`,`${fv(mx.v)}（误差 ${fe(mx)}）`);
  return tx(`<p><b>A weight of ${fin(x)} stored in FP4 E2M1 with no scale becomes ${fv(n.v)}. With an exact scale for its block, FP4 stores ${fv(a.v)} (off by ${fe(a)}) and FP8 E4M3 stores ${fv(e.v)}.</b></p><p>An FP8 weight takes 1 byte and an FP4 weight only half a byte, so serving a model in them cuts the weights’ memory by half to three quarters compared with 16-bit weights (the precision buttons of the lab ${labLink('kv-cache')} offer both). FP4 can store only 8 sizes, from 0 to 6, so its blocks must be small: the ${DEF('OCP MX',TIP.mx())} ${cite(SRC.mx,'spec')} shares one scale among 32 values. Here the scale is exact: the block’s largest value divided by 6. MX rounds the scale down to a power of two, so FP4 stores ${mxT}.</p>`,
            `<p><b>大小为 ${fin(x)} 的权重，不缩放直接存成 FP4 E2M1，会变成 ${fv(n.v)}。给它所在的块配一个精确的缩放系数后，FP4 存成 ${fv(a.v)}（误差 ${fe(a)}），FP8 E4M3 存成 ${fv(e.v)}。</b></p><p>FP8 权重每个占 1 字节，FP4 只占半个字节，和 16 位的权重比，部署时权重的显存能省一半到四分之三（${labLink('kv-cache')}这个 Lab 的精度按钮里也有这两种）。FP4 只能表示 8 种大小（0 到 6），所以块要很小：${DEF('OCP MX',TIP.mx())} ${cite(SRC.mx,'规范')}让 32 个数共用一个缩放系数。这里的缩放系数是精确的，等于块里最大的数除以 6。MX 会把缩放系数向下取成 2 的幂，于是 FP4 存成 ${mxT}。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const CHN={fp32:{en:'All FP32 (initial setup)',zh:'全部 FP32（初始设置）'},fp16:{en:'All FP16',zh:'全部 FP16'},fp16lsm:{en:'FP16 + loss scaling + FP32 master copy',zh:'FP16 + loss scaling + FP32 主副本'},
  bf16:{en:'All BF16',zh:'全部 BF16'},bf16m:{en:'BF16 + FP32 master copy',zh:'BF16 + FP32 主副本'}};
const CHD={fp32:{en:'weights and gradients 4 bytes each',zh:'权重和梯度各 4 字节'},fp16:{en:'weights and gradients 2 bytes each',zh:'权重和梯度各 2 字节'},fp16lsm:{en:'loss scale 65,536; updates go to FP32',zh:'loss scale 65,536；更新加到 FP32 上'},
  bf16:{en:'weights and gradients 2 bytes each',zh:'权重和梯度各 2 字节'},bf16m:{en:'updates go to FP32',zh:'更新加到 FP32 上'}};
const chChecks=e=>[[tx('Weights and gradients in 16 bits','权重和梯度用 16 位'),e.sixteen],[tx('No overflow (largest gradient 1e‑3)','不溢出（最大的梯度 1e‑3）'),!e.over],
  [tx('No gradient lost (smallest 1e‑8)','梯度不丢（最小的 1e‑8）'),!e.lost],[tx(`No update lost (${fin(CHT.u)} on a weight of 0.02)`,`更新不丢（大小为 0.02 的权重加上 ${fin(CHT.u)}）`),e.kept],
  [tx(`No update lost (${fin(CHT.u)} on a gain of 1.0)`,`更新不丢（大小为 1.0 的 gain 加上 ${fin(CHT.u)}）`),e.kept2]];
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chR), best=chBest(), nm=id=>esc(L(CHN[id]));
  let h;
  if(me.ok&&chR===best.id) h=`<p class="okline nf-mt"><b>`+tx(`Goal met: ${nm(chR)} stores weights and gradients in 16 bits and loses nothing, with one extra part.</b> And that is the fewest: FP16 also works, but only with loss scaling as a second extra part.</p>`,`达成目标：${nm(chR)} 用 16 位存权重和梯度，什么都没丢，只多了一个部件。</b>而且这已经是最少的：FP16 也行，但还要再加 loss scaling。</p>`);
  else if(me.ok) h=`<p class="okline nf-mt"><b>`+tx(`Goal met: nothing is lost and weights and gradients are 16-bit.</b> But one extra part is enough.</p><p>${nm(best.id)} needs no loss scaling, because BF16 already holds the gradient of 1e‑8.</p>`,`达成目标：什么都没丢，权重和梯度都是 16 位。</b>不过只加一个部件就够了。</p><p>${nm(best.id)}不需要 loss scaling，因为 BF16 本来就存得下 1e‑8 的梯度。</p>`);
  else {const miss=chChecks(me).filter(c=>!c[1]).map(c=>c[0]);
    h=`<p class="badline nf-mt"><b>`+tx(`Not yet. ${miss.length>1?'These checks fail':'This check fails'}: ${miss.join('; ')}.</b></p>`,`还没达到。没做到的检查：${miss.join('；')}。</b></p>`)+
      tx('<p class="small">Hint: compare the smallest gradient with each format’s smallest number. Then check whether the update is more than half the gap between the storable numbers near each weight.</p>','<p class="small">提示：先把最小的梯度和每种格式最小的数比一比，再看更新量是否大于每个权重附近间隔的一半。</p>')+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);}
  if(!chReveal) return h;
  const rows=CHO.map(id=>({id,e:chEval(id)}));
  const xp=n=>tx(`, ${n} extra part${n===1?'':'s'}`,`，多 ${n} 个部件`);
  h+=A.ui.cmpChart({title:tx(`${M2}: bytes per parameter for weights, gradients and the master copy (the goal is 16-bit weights and gradients with nothing lost)`,`${M2} 的权重、梯度和主副本每个参数占的字节（目标是权重和梯度用 16 位，什么都不丢）`),
    aria:tx('Bytes per parameter: ','每个参数占的字节：')+rows.filter(x=>x.e.ok||x.id===chR||x.id==='fp32').map(x=>`${L(CHN[x.id])}${xp(x.e.extras)}${tx(': ','：')}${x.e.bytes}`).join(tx('; ','；')),brief:true,
    hiddenText:n=>tx(n===1?'The other recipe misses the goal and is not shown.':`The other ${n} recipes miss the goal and are not shown.`,`其余 ${n} 种做法没达到目标，图里没有列出。`),
    rows:rows.map(x=>({label:`${x.e.ok?'✓ ':''}${L(CHN[x.id])}${xp(x.e.extras)}`,value:x.e.bytes,text:tx(`${x.e.bytes} bytes`,`${x.e.bytes} 字节`),ok:x.e.ok,me:x.id===chR,
      tags:(x.id==='fp32'?[{text:tx('initial setup','初始设置')}]:[]).concat(x.id===chR?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    max:12,okText:tx('✓ meets the goal','✓ 达成目标'),noText:tx('misses it','没达成'),legend:true});
  const bm=chEval('bf16m'), u=fin(CHT.u);
  h+=`<details class="nf-whyd"><summary>${tx('Why: what each recipe loses, and what the passing ones cost','原因：每种做法丢了什么，达标的做法花了什么')}</summary>`+
    tx(`<p class="small">The initial setup, all FP32, loses nothing but takes 4 bytes for each weight and each gradient. All FP16 and all BF16 halve that, but they lose updates: the assumed step of one learning rate, ${u}, is less than half of BF16’s gap at 0.02 and less than half of either format’s gap at 1.0. FP16 also loses the gradient of 1e‑8.</p><p class="small">The two passing recipes take ${bm.bytes} bytes per parameter, the same as all FP32: the FP32 master copy costs back what the 16-bit weights and gradients save. The gain is elsewhere: activations stored in 16 bits take half the memory, and the matmuls run on 16-bit inputs.</p>`,
       `<p class="small">初始设置全部 FP32，什么都不丢，但每个权重、每个梯度都占 4 字节。全部 FP16 和全部 BF16 把它减半，却会丢更新。假设的一步是一个学习率，也就是 ${u}。它不到 BF16 在 0.02 处间隔的一半，也不到两种格式在 1.0 处间隔的一半。FP16 还丢了 1e‑8 的梯度。</p><p class="small">两个达标的做法每个参数都占 ${bm.bytes} 字节，和全部 FP32 一样：FP32 主副本把 16 位权重和梯度省下的又花了回去。好处在别处：激活值用 16 位存，显存减半；矩阵乘法的输入也是 16 位。</p>`)+`</details>`;
  h+=tx(`<p class="small muted">Note: this counts weights, gradients and the master copy only; Adam’s two running averages (m and v, 4 bytes each per parameter) add more in every recipe. Gradient sizes differ from model to model: check your own model’s gradient range (for example, how often GradScaler skips a step) before you drop loss scaling or the master copy.</p>`,
        `<p class="small muted">注意：这里只算权重、梯度和主副本；Adam 的两个滑动平均（m 和 v，每个参数各 4 字节）在每种做法里都要再加。梯度的大小因模型而异：去掉 loss scaling 或主副本之前，先看看自己模型的梯度范围（比如 GradScaler 多久跳过一步）。</p>`);
  return h;
}

/* =====================================================================
   copy the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===6?'Six':fint(STEPS.length)} short parts. Predict first, then see how exponent bits set the range, how mantissa bits set the precision, and how loss scaling and an FP32 master copy fix what 16 bits lose. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看指数位怎样决定范围、尾数位怎样决定精度，以及 loss scaling 和 FP32 主副本怎样补回 16 位丢掉的东西。第一次学从这里开始。`),
  sandbox:tx('Four real uses, from BF16 pretraining to FP4 weights for inference. Pick one to see its formats at work on a gradient, an update or a weight.',
             '四个真实用途，从 BF16 预训练到推理用的 FP4 权重。选一个，看它的格式怎样处理一个梯度、一次更新或一个权重。'),
  challenge:tx(`Train ${M2} late in pretraining with 16-bit weights and gradients: its learning rate has decayed to ${fin(CHT.u)}. Pick the recipe that loses no gradient of 1e‑8 and no update, with the fewest extra parts.`,
               `在预训练后期用 16 位的权重和梯度训练 ${M2}：学习率已经降到 ${fin(CHT.u)}。选一个做法，不丢 1e‑8 的梯度，也不丢更新，而且多加的部件最少。`)
});
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> every stored value on this page is the exact round-to-nearest-even of the shown number into the format’s bits; ${J('math','“Show the math”')} decodes them.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>FP32 (1 sign, 8 exponent, 23 mantissa bits), BF16 (1, 8, 7), FP16 (1, 5, 10), FP8 E4M3 (1, 4, 3; no Inf, largest 448), FP8 E5M2 (1, 5, 2) and FP4 E2M1 (1, 2, 1; largest 6), with subnormal numbers.</li>`+
 `<li>One training step: the gradient (times the loss scale) is stored in 16 bits and divided back in FP32; the update is added to the 16-bit weight or to the FP32 master copy.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The tensor sizes (weights near 0.02, gradients 1e‑9 to 1e‑3, activation outliers 1e3 to 1e4) are illustrative, not measured on ${M8}. The update is an assumption: exactly one learning rate per step, the typical size of an Adam step (${cite(SRC.adam,'Kingma and Ba, 2015')}, Section 2.1). A real step depends on each weight’s m, v and ε: noisy gradients make it smaller, which loses even more of it, and one large gradient after many near-zero ones can make it up to about 3 times larger.</li>`+
 `<li>A value too large for E4M3 becomes NaN, or 448 if the cast saturates; FP4 has no Inf or NaN, so OCP MX clamps it to 6. The block scale here is exact; DeepSeek‑V3 derives it from each tile’s largest value (for a few activations it rounds it to a power of two), and OCP MX rounds it to a power of two.</li>`+
 `<li>FP16 and BF16 matmuls accumulate in FP32 (FP8 tensor cores on Hopper keep about 14 bits, so DeepSeek‑V3 moves partial sums to FP32 every 128 elements), and real training adds noise from batches; neither changes which single value survives a format.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${cite(SRC.mp,'Micikevicius et al., 2018')} (FP16, loss scaling, FP32 master copy); ${cite(SRC.bf,'Kalamkar et al., 2019')} (BF16); ${cite(SRC.f8,'Micikevicius et al., 2022')} (FP8 E4M3 and E5M2); ${cite(SRC.mx,'OCP MX v1.0')} (FP4 E2M1, blocks of 32).</li>`+
 `<li>${cite(SRC.ds,'DeepSeek‑V3 report')} (FP8 training, Section 3.3); ${cite(SRC.llama,'Llama 3 paper')} (BF16, FP32 gradient accumulation, Section 3.3.2); ${cite(SRC.l2,'Llama 2 paper')} (peak learning rate 3e‑4, cosine decay to 10%, Section 2.2); PyTorch ${cite(SRC.gs,'<code>grad_scaler.py</code>')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上每个存下的数，都是把显示的数按就近舍入（平局取偶）精确转换成这种格式的位得到的；${J('math','“展开公式”')}里有解码。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>FP32（1 个符号位、8 个指数位、23 个尾数位）、BF16（1、8、7）、FP16（1、5、10）、FP8 E4M3（1、4、3；没有 Inf，最大 448）、FP8 E5M2（1、5、2）和 FP4 E2M1（1、2、1；最大 6），包括次正规数。</li>`+
 `<li>一步训练：梯度（乘上 loss scale）存成 16 位，再在 FP32 里除回去；更新加到 16 位的权重上，或者加到 FP32 主副本上。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>张量的大小（权重约 0.02，梯度 1e‑9 到 1e‑3，激活值离群点 1e3 到 1e4）是示意的，不是在 ${M8} 上测的。更新量是一个假设：每一步正好一个学习率，也就是 Adam 一步的典型大小（${cite(SRC.adam,'Kingma 和 Ba，2015')}，第 2.1 节）。真实的一步取决于每个权重的 m、v 和 ε：梯度噪声大时它更小，丢得更多；梯度长时间接近 0 之后突然来一个大梯度，它最大可到约 3 倍。</li>`+
 `<li>E4M3 放不下的大数会变成 NaN；如果转换时做饱和，就变成 448。FP4 没有 Inf 和 NaN，OCP MX 把它截到 6。这里块的缩放系数按精确值算；DeepSeek‑V3 用每个小块最大的数来定它（少数激活值会取成 2 的幂），OCP MX 则把它取成 2 的幂。</li>`+
 `<li>FP16 和 BF16 的矩阵乘法用 FP32 累加（Hopper 的 FP8 tensor core 只保留约 14 位，所以 DeepSeek‑V3 每 128 个元素把部分和移到 FP32），真实训练还有 batch 带来的噪声；这两点都不改变单个数能不能被一种格式存下。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${cite(SRC.mp,'Micikevicius 等，2018')}（FP16、loss scaling、FP32 主副本）；${cite(SRC.bf,'Kalamkar 等，2019')}（BF16）；${cite(SRC.f8,'Micikevicius 等，2022')}（FP8 E4M3 和 E5M2）；${cite(SRC.mx,'OCP MX v1.0')}（FP4 E2M1，32 个数一块）。</li>`+
 `<li>${cite(SRC.ds,'DeepSeek‑V3 技术报告')}（FP8 训练，第 3.3 节）；${cite(SRC.llama,'Llama 3 论文')}（BF16，FP32 梯度累加，第 3.3.2 节）；${cite(SRC.l2,'Llama 2 论文')}（学习率峰值 3e‑4，余弦衰减到 10%，第 2.2 节）；PyTorch ${cite(SRC.gs,'<code>grad_scaler.py</code>')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>Each number in a model takes a fixed number of bytes: BF16 takes 2, FP8 1 and FP4 half a byte. ${labLink('kv-cache')} counts memory with these sizes. This lab looks inside those bytes: what the bits mean, and which numbers each format can hold.</p>`,
                    `<p>模型里的每个数都占固定的字节：BF16 占 2 字节，FP8 占 1 字节，FP4 占半个字节。${labLink('kv-cache')}就按这些大小算显存。这个 Lab 看这些字节里面：每一位是什么意思，每种格式能存哪些数。</p>`);
const OUTCOME=()=>tx('predict whether a value survives a number format, and choose formats, loss scaling and a master copy for a training run.','判断一个数存进某种数值格式后还剩什么，并为一次训练选好格式、loss scaling 和主副本。');
const EXITS=()=>{
  const e5=FM.e5m2, h=FM.fp16, w4=quant(4+0.01,'bf16');
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('A gradient of 1e‑8 becomes 0 in FP16 but survives in BF16. Why?','1e‑8 的梯度存成 FP16 变成 0，存成 BF16 却留住了。为什么？'),
   o:[tx('FP16’s 5 exponent bits stop at about 6e‑8, and 1e‑8 is less than half of that, so it rounds to 0; BF16 has FP32’s 8 exponent bits and reaches far below 1e‑8.','FP16 的 5 个指数位最小只到约 6e‑8，1e‑8 不到它的一半，所以舍入成 0；BF16 有 FP32 的 8 个指数位，能到比 1e‑8 小得多的数。'),
      tx('BF16 has more mantissa bits than FP16, so it keeps more digits of a small number; FP16 runs out of digits and stores 0 instead.','BF16 的尾数位比 FP16 多，所以小数能多留几位；FP16 位数不够，只好存成 0。'),
      tx('FP16 cuts the bits it cannot hold instead of rounding them, so a small value drops to 0; BF16 rounds to the nearest number and keeps it.','FP16 把放不下的位直接截掉而不是舍入，所以小值掉成 0；BF16 舍入到最近的数，所以留住了。')],a:0,
   n:[null,tx('BF16 has fewer mantissa bits than FP16: 7 against 10. Its advantage is the exponent: 8 bits against 5.','BF16 的尾数位比 FP16 少：7 对 10。它的优势在指数位：8 对 5。'),
     tx('Both formats round to the nearest number. The rounding is not the problem: FP16 simply has no number near 1e‑8 except 0.','两种格式都是就近舍入。问题不在舍入：FP16 在 1e‑8 附近除了 0 没有别的数。')],
   why:tx(`The exponent field sets the range. FP16’s 5 bits stop at 2⁻²⁴ ≈ ${fv(h.minS)}, and 1e‑8 is less than half of that, so it rounds to 0. BF16’s 8 bits reach ${fv(FM.bf16.minS)}.`,`指数字段决定范围。FP16 的 5 位最小到 2⁻²⁴ ≈ ${fv(h.minS)}，1e‑8 不到它的一半，所以舍入成 0。BF16 的 8 位能到 ${fv(FM.bf16.minS)}。`)},
  {q:tx('1.0 + 0.0003 stored in BF16 is still 1.0. Why does a 16-bit weight lose a small update?','1.0 + 0.0003 存成 BF16 还是 1.0。为什么 16 位的权重会丢掉小的更新？'),
   o:[tx('BF16 can store only numbers one notch apart, about 0.0078 near 1.0; an update below half a notch rounds back to the old weight.','BF16 能存的数是一格一格的，1.0 附近一格约 0.0078；更新不到半格，加完又舍入回原来的权重。'),
      tx('BF16 cannot hold a small change like 0.0003 at all, so the update itself becomes 0 before it reaches the weight.','BF16 根本存不下 0.0003 这样的小改动，所以更新量在加到权重上之前就成了 0。'),
      tx('The sum 1.0003 overflows BF16’s 8 exponent bits, so the cast falls back to the old value of the weight.','1.0003 这个和超出了 BF16 的 8 个指数位，所以转换退回到权重原来的值。')],a:0,
   n:[null,tx('BF16 has FP32’s range: 0.0003 on its own is stored as about 0.0003. The update survives alone; it is lost only when added to 1.0.','BF16 的范围和 FP32 一样：0.0003 单独存是约 0.0003。更新量本身留得住，加到 1.0 上才丢。'),
     tx('BF16 reaches about 3.4e38, like FP32, so 1.0003 cannot overflow. The loss is about precision, not range.','BF16 能到约 3.4e38，和 FP32 一样，1.0003 不会溢出。丢掉更新是精度的问题，不是范围的问题。')],
   why:tx('BF16 has only 7 mantissa bits: near 1.0 one notch is about 0.0078, so an update needs at least half a notch, about 0.0039, to survive. A Llama 2 7B step moves a weight by about one learning rate, 3e‑4 to 3e‑5, far less than half a notch, so every update is rounded away.','BF16 只有 7 个尾数位，1.0 附近一格约 0.0078，更新至少要半格（约 0.0039）才留得住。Llama 2 7B 一步只让权重变化约一个学习率（3e‑4 到 3e‑5），远小于半格，所以每次更新都被舍入掉。')}],
 transfer:[
  {q:tx(`FP16 (5 exponent bits, 10 mantissa bits) reaches down to 2⁻²⁴ ≈ ${fv(h.minS)}. FP8 E5M2 has the same 5 exponent bits but only 2 mantissa bits. About how small is its smallest positive number?`,`FP16（5 个指数位、10 个尾数位）最小能到 2⁻²⁴ ≈ ${fv(h.minS)}。FP8 E5M2 也有 5 个指数位，但只有 2 个尾数位。它最小的正数大约是多少？`),
   o:[tx('About 6e‑8, the same as FP16','约 6e‑8，和 FP16 一样'),tx('About 1.5e‑5','约 1.5e‑5'),tx('About 6e‑5','约 6e‑5'),tx('About 1e‑38','约 1e‑38')],a:1,
   why:tx(`Both have the same exponent bits, so the same smallest normal number: 2⁻¹⁴ ≈ ${fv(e5.minN)}. Below it the subnormal numbers reach further down, by a factor of 2 for each mantissa bit. FP16 has 10 mantissa bits and reaches 2¹⁰ times further down; E5M2 has only 2, so it reaches only 4 times further, down to 2⁻¹⁶ ≈ ${fv(e5.minS)}.`,`两者的指数位一样多，所以最小的正规数也一样，都是 2⁻¹⁴ ≈ ${fv(e5.minN)}。再往下要靠次正规数：每个尾数位能让最小的数再小一半。FP16 有 10 个尾数位，能再往下 2¹⁰ 倍；E5M2 只有 2 个，只能再往下 4 倍，所以最小到 2⁻¹⁶ ≈ ${fv(e5.minS)}。`)},
  {q:tx('In BF16, any update below about 0.0039 added to 1.0 still gives 1.0. What is the smallest update that still changes a BF16 weight of 4.0?','在 BF16 里，小于约 0.0039 的更新加到 1.0 上，结果还是 1.0。要改变一个大小为 4.0 的 BF16 权重，更新量至少要多大？'),
   o:[tx('About 0.001','约 0.001'),tx('About 0.004','约 0.004'),tx('About 0.016','约 0.016'),tx('About 0.06','约 0.06')],a:2,
   why:tx(`At least about 0.016. Near 4.0, neighbouring BF16 numbers are 0.03125 apart, 4 times the gap near 1.0 (4.0 is 4 times larger, so the gap is too); an update must be more than half of it to survive. For example, 4.0 + 0.01 is still stored as ${fv(w4.v)}.`,`至少要约 0.016。4.0 附近相邻两个 BF16 数相差 0.03125，是 1.0 附近的 4 倍（4.0 是 1.0 的 4 倍，间隔也是 4 倍）；更新要超过它的一半才留得住。比如 4.0 + 0.01 存下来仍是 ${fv(w4.v)}。`)}],
 practice:[
  {id:'cs-fundamentals',d:tx('Questions 9 and 8: give the exponent and mantissa bits, the largest value and the gap at 1 for FP32, FP16 and BF16, say why training prefers BF16 and what loss scaling prevents; then define machine epsilon.','第 9、8 题：给出 FP32、FP16、BF16 的指数位、尾数位、最大值和 1 处的间隔，说明训练为什么偏好 BF16、loss scaling 防的是什么；再定义机器精度。')},
  {id:'matmul-roofline',d:tx('The follow-up “Lower precision”: why FP8 compute alone moves the roofline’s ridge, and why it pays off only when the stored bytes also drop to 1 per number.','延伸“更低的精度”：为什么只把计算换成 FP8 会移动 roofline 的拐点，为什么只有存储也降到每个数 1 字节时才划算。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Number formats','数值格式'),title:l?L(l.q):tx('Why does training use BF16 and not FP16?','训练为什么用 BF16，不用 FP16？'),
    minutes:10,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: bytes per number','前置知识：每个数占几个字节'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage nf-stage" id="nf-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="nf-st-title"></div><div class="meter-sub" id="nf-st-sub"></div></div>`+
   `<div class="stage-actions"><div class="seg-ctl nf-view" id="nf-view" role="group" aria-label="${esc(tx('What the chart shows','图里显示什么'))}"><button type="button" data-view="a" aria-pressed="true">${tx('A value','一个数')}</button><button type="button" data-view="b" aria-pressed="false">${tx('An update','一次更新')}</button><button type="button" data-view="c" aria-pressed="false">${tx('A training step','一步训练')}</button></div><span class="status" id="nf-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="nf-legend"></div>`+
   `<div class="stage-graph graph nf-graph" id="nf-graph">`+
    `<div class="nf-phone" id="nf-phone"></div>`+
    `<div class="nf-a" id="nf-a"><div class="nf-cap" id="nf-a-cap"></div><div class="nf-rows" id="nf-rows"></div><div class="legend-row nf-key" id="nf-key-rows"></div><div class="nf-range" id="nf-range"></div><div class="legend-row nf-key" id="nf-key-range"></div></div>`+
    `<div class="nf-b" id="nf-b" hidden><div class="nf-cap" id="nf-b-cap"></div><div class="nf-zoom" id="nf-zoom"></div><div class="legend-row nf-key" id="nf-key-zoom"></div></div>`+
    `<div class="nf-c" id="nf-c" hidden><div class="nf-cap" id="nf-c-cap"></div><div class="nf-paths" id="nf-paths"></div><div class="legend-row nf-key" id="nf-key-paths"></div><div class="nf-sub" id="nf-chart-cap"></div><div class="nf-chart" id="nf-chart"></div><div class="legend-row nf-key" id="nf-key-chart"></div></div>`+
   `</div>`+
   `<div class="metrics nf-metrics" id="nf-metrics"></div>`+
   `<details class="math" id="nf-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="nf-mathbody"></div></details></div>`;
}
const noteX=()=>mode==='sandbox'?tx('Shown in “A value”: the number each format stores, in steps of 1, 2 and 5.','用在“一个数”里：每种格式要存的数，按 1、2、5 的台阶走。'):tx('Shown in the chart: the number each format stores, in steps of 1, 2 and 5.','用在图里：每种格式要存的数，按 1、2、5 的台阶走。');
const noteU=()=>mode==='sandbox'?tx('Shown in “An update” and “A training step”: what one optimizer step adds to a weight of 1.0, an RMSNorm gain.','用在“一次更新”和“一步训练”里：优化器一步加到一个大小为 1.0 的权重（比如 RMSNorm 的 gain）上的量。'):tx('Shown in the chart: what one optimizer step adds to a weight of 1.0, an RMSNorm gain.','用在图里：优化器一步加到一个大小为 1.0 的权重（比如 RMSNorm 的 gain）上的量。');
function controlsHTML(){
  const seg=(id,key,label,opts,v,note)=>`<div class="lockable" id="${id}-wrap">`+A.ui.seg({id,key,label,options:opts,value:v,note})+`</div>`;
  return `<aside class="panel controls" id="nf-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="nf-lock" hidden></p>`+
   `<div class="lockable" id="nf-ctl-x-wrap">`+xCtl('nf-ctl-x',false).replace(/<\/div>$/,`<div class="ctl-note" id="nf-note-x">${noteX()}</div></div>`)+`</div>`+
   `<div class="lockable" id="nf-ctl-u-wrap">`+uCtl('nf-ctl-u',false).replace(/<\/div>$/,`<div class="ctl-note" id="nf-note-u">${noteU()}</div></div>`)+`</div>`+
   seg('nf-ctl-f16','f16',tx('16-bit format for weights and gradients','权重和梯度用的 16 位格式'),[['fp16','FP16'],['bf16','BF16']],S.f16,tx('Used in “A training step”.','用在“一步训练”里。'))+
   seg('nf-ctl-ls','ls',tx('Loss scaling','loss scaling'),[['0',tx('off','不用')],['1','65,536']],String(S.ls),tx('The loss, and so every gradient, is multiplied by 65,536 (2¹⁶, PyTorch GradScaler’s starting value) and divided back in FP32. It is a setting, not computed: GradScaler lowers it when gradients overflow and raises it when they do not.','loss 乘以 65,536（2¹⁶，PyTorch GradScaler 的初始值），每个梯度也就放大这么多，再在 FP32 里除回去。它是设定值，不是算出来的：梯度溢出时 GradScaler 把它调小，不溢出时调大。'))+
   seg('nf-ctl-ms','ms',tx('FP32 master copy of the weights','权重的 FP32 主副本'),[['0',tx('off','不用')],['1',tx('on','用')]],String(S.ms),tx('On: the update is added to an FP32 copy and the 16-bit weight is rounded from it. It costs 4 more bytes per parameter.','用：更新加到一份 FP32 副本上，16 位权重从它舍入得到。每个参数多占 4 字节。'))+
   seg('nf-ctl-blk','blk',tx('Scale per block (FP8 and FP4 rows)','按块缩放（FP8 和 FP4 那几行）'),BLKS.map((b,i)=>[String(i),b?tx(`largest ${fint(b)}×`,`最大 ${fint(b)} 倍`):tx('off','不用')]),String(S.blk),
     tx('Assumed: the largest value in the value’s block is this many times the value. The scale maps that largest value onto the format’s largest (448 for E4M3, 6 for FP4), as DeepSeek‑V3 does per tile. Real blocks hold 16 to 128 numbers, and how far apart they are depends on the tensor; outliers make it worse.','假设：这个数所在的块里，最大的数是它的这么多倍。缩放系数把这个最大的数对到格式的最大值上（E4M3 是 448，FP4 是 6），DeepSeek‑V3 就是这样按小块做的。真实的块有 16 到 128 个数，它们相差多远取决于张量；有离群点时更糟。'))+
   `<div class="nf-grp lockable" id="nf-ctl-assumed"><div class="nf-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('Tensor sizes in the range chart','范围图里的张量大小')}</span><div class="ctl-note">${TIP.illus()} ${tx('They are typical orders of magnitude, chosen to show where each format runs out; a real model’s spread depends on its size, normalization and training stage.','它们是常见的数量级，用来显示每种格式在哪里不够用；真实模型的分布取决于规模、归一化和训练阶段。')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('The update','更新量')}</span><div class="ctl-note">${TIP.upd()} ${TIP.updRule()}</div></div></div>`+
   `<div class="lockable" id="nf-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import struct

def bits32(x):                       # FP32 bit pattern of x
    return struct.unpack('>I', struct.pack('>f', x))[0]

def f32(b):                          # FP32 value of a bit pattern
    return struct.unpack('>f', struct.pack('>I', b))[0]

def to_bf16(x):                      # FP32 -> BF16, nearest, ties to even
    b = bits32(x)
    lsb = (b >> 16) & 1              # last bit BF16 keeps
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return f32(b)                    # (NaN and Inf not handled here)

def show(x):
    b = bits32(x)
    s, e, m = b >> 31, (b >> 23) & 0xFF, b & 0x7FFFFF
    print(f"{x:.8g}: sign {s} exp {e:08b} ({e - 127:+d}) "
          f"mant {m:023b}")

show(0.02)
print(to_bf16(0.02))                 # the weight in BF16
w = 1.0
print(to_bf16(w + 0.0003) == w)      # True: the update is lost
print(to_bf16(w + 0.01))             # 1.0078125, one BF16 step
master = w
for _ in range(20):                  # a higher-precision master copy
    master += 0.0003
print(round(master, 4), to_bf16(master))
# 0.02: sign 0 exp 01111001 (-6) mant 01000111101011100001010
# 0.02001953125
# True
# 1.0078125
# 1.006 1.0078125`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="nf-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="nf-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Read a float’s bits, round FP32 to BF16 the way hardware does, and watch an update vanish until a higher-precision master copy adds it up (Python floats are 64-bit; FP32 works the same way here).','读出一个浮点数的位，像硬件一样把 FP32 舍入成 BF16，看一次更新消失，直到一份更高精度的主副本把它累加起来（Python 的 float 是 64 位；这里换成 FP32 也一样）。')}</p><pre class="code" id="nf-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="nf-prod">`+
   tx(`<li>PyTorch’s ${cite(SRC.gs,'<code>torch/amp/grad_scaler.py</code>')}: <code>GradScaler</code> multiplies the loss by the scale (<code>init_scale=2.0**16</code>), unscales the gradients before the optimizer step, skips the step when it finds Inf or NaN and halves the scale (<code>backoff_factor=0.5</code>), and doubles it after <code>growth_interval=2000</code> clean steps.</li>`+
      `<li>${cite(SRC.ac,'<code>torch/amp/autocast_mode.py</code>')}: <code>autocast</code> runs matmuls in FP16 or BF16 while the parameters stay FP32, so the parameters themselves are the master copy.</li>`+
      `<li>Transformer Engine’s ${cite(SRC.te,'<code>transformer_engine/common/recipe/__init__.py</code>')} defines the FP8 recipes: <code>Format.HYBRID</code> uses E4M3 in the forward pass and E5M2 for gradients, and <code>DelayedScaling</code> sets each tensor’s scale from the largest values of recent steps.</li>`+
      `<li>Megatron-LM’s ${cite(SRC.mg,'<code>megatron/core/optimizer/optimizer.py</code>')} keeps FP32 main parameters for BF16 or FP16 models and copies them back after each step.</li>`,
      `<li>PyTorch 的 ${cite(SRC.gs,'<code>torch/amp/grad_scaler.py</code>')}：<code>GradScaler</code> 把 loss 乘以缩放系数（<code>init_scale=2.0**16</code>），在优化器那一步之前把梯度除回去；发现 Inf 或 NaN 时跳过这一步，把系数减半（<code>backoff_factor=0.5</code>）；连续 <code>growth_interval=2000</code> 步正常就加倍。</li>`+
      `<li>${cite(SRC.ac,'<code>torch/amp/autocast_mode.py</code>')}：<code>autocast</code> 让矩阵乘法用 FP16 或 BF16 算，参数本身留在 FP32，所以参数就是主副本。</li>`+
      `<li>Transformer Engine 的 ${cite(SRC.te,'<code>transformer_engine/common/recipe/__init__.py</code>')} 定义了 FP8 的做法：<code>Format.HYBRID</code> 在前向传播用 E4M3、梯度用 E5M2；<code>DelayedScaling</code> 按最近几步的最大值给每个张量定缩放系数。</li>`+
      `<li>Megatron-LM 的 ${cite(SRC.mg,'<code>megatron/core/optimizer/optimizer.py</code>')} 为 BF16 或 FP16 模型保留 FP32 的 main 参数，每一步之后再拷回去。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.mp,'Mixed Precision Training')}<span>${tx('Micikevicius et al., ICLR 2018: FP16 storage with an FP32 master copy of the weights, loss scaling, and FP32 accumulation','Micikevicius 等，ICLR 2018：FP16 存储，加上权重的 FP32 主副本、loss scaling 和 FP32 累加')}</span></li>`+
   `<li>${cite(SRC.bf,'A Study of BFLOAT16 for Deep Learning Training')}<span>${tx('Kalamkar et al., 2019: BF16 keeps FP32’s range; without loss scaling, the models train as well as with FP32','Kalamkar 等，2019：BF16 保留 FP32 的范围；不用 loss scaling，训练出的模型效果也和 FP32 一样好')}</span></li>`+
   `<li>${cite(SRC.f8,'FP8 Formats for Deep Learning')}<span>${tx('Micikevicius et al., 2022: the E4M3 and E5M2 formats, with E4M3 giving up Inf to reach 448, and scaling per tensor','Micikevicius 等，2022：E4M3 和 E5M2 两种格式，E4M3 放弃 Inf 换来最大 448，并按张量缩放')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="nf-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="nf-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="nf-app m2" id="nf-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="nf-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see which formats can store its numbers.','选一个场景，看看哪些格式存得下那里的数。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="nf-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const u=fin(CHT.u), pk=fin(LR_PEAK);
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Train ${M2} in 16 bits with no gradient or update lost`,`用 16 位训练 ${M2}，梯度和更新都不丢`)}</h3>`+
   `<p>${tx(`You train ${M2} late in pretraining. Its learning rate peaked at ${pk} and decays to 10% of that (${cite(SRC.l2,'Llama 2 paper')}, Section 2.2), so it is now ${u}.`,
            `你在训练 ${M2}，已到预训练后期。学习率的峰值是 ${pk}，之后降到峰值的 10%（${cite(SRC.l2,'Llama 2 论文')}第 2.2 节），所以现在是 ${u}。`)}</p>`+
   `<p>${tx(`How far does one step move a weight? With plain SGD, the step is the learning rate times the gradient. But ${M2} is trained with AdamW, which, like Adam, divides each gradient by the typical size of that weight’s recent gradients. So a typical step moves a weight by about the learning rate, almost regardless of the gradient’s size (the lab ${labLink('optimizers')} shows why). Assume exactly that here: each step moves a weight by ${u}. Most weights are near 0.02, the ${DEF('RMSNorm gains',TIP.gain())} are near 1.0, and the gradients run from 1e‑8 to 1e‑3.`,
            `一步会让权重变多少？用普通的 SGD，一步是学习率乘以梯度。可是 ${M2} 用 AdamW 训练，它和 Adam 一样，会把梯度除以这个权重最近梯度的典型大小。所以一步的大小基本不看梯度有多大，典型的一步约为一个学习率（原因见${labLink('optimizers')}这个 Lab）。这里就假设每一步让权重变化 ${u}。大多数权重约为 0.02，${DEF('RMSNorm 的 gain',TIP.gain())} 约为 1.0，梯度从 1e‑8 到 1e‑3。`)}</p>`+
   `<p>${tx('Goal: store weights and gradients in 16 bits with no overflow, no gradient lost to zero and no update lost. Among these recipes, pick the one with the fewest extra parts (an FP32 master copy, loss scaling).',
            '目标：权重和梯度用 16 位存，不溢出，梯度不丢成 0，更新也不丢。在这些做法里，选多加的部件（FP32 主副本、loss scaling）最少的一个。')}</p>`+
   `<div class="nf-chlv" id="nf-ic-ch">`+A.ui.levers({aria:tx('The recipe, pick one','做法，选一种'),value:chR,options:CHO.map(id=>({v:id,title:L(CHN[id]),desc:L(CHD[id]),result:''}))})+`</div>`+
   `<div class="nf-checks" id="nf-ch-g"></div>`+
   submit+`<div id="nf-ch-res" class="nf-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},range:{en:'Drag the value to 1e‑8 to continue.',zh:'把数拖到 1e‑8 才能继续。'},
    prec:{en:'Find an update BF16 loses and FP16 keeps to continue.',zh:'找到一个 BF16 丢掉、FP16 留住的更新量才能继续。'},fix:{en:'Try “+ both” to continue.',zh:'试一下“+ 两个都用”才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#nf-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="nf-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="nf-why" hidden></p>');}
  A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#nf-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]||(s.id==='range'&&S.xi!==X_W)||(s.id==='prec'&&S.ui!==U_0)||(s.id==='fix'&&lv!=='fp16'))&&(!s.gate||s.gate());
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvTried.has(v)||v===lv?lvRes(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.nf-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#nf-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chR), g=$('#nf-ch-g',rail); if(!g) return;
    $$('[data-lever]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lever===chR)));
    g.innerHTML=chChecks(e).map(([t,ok])=>`<div class="goal mono nf-goal${ok?' met':' nf-over'}"><span>${esc(t)}</span><b>${ok?'✓':'✗'}</b></div>`).join('')+
      `<div class="goal mono nf-goal"><span>${tx('Extra parts','多加的部件')}</span><b>${e.extras}</b></div>`;
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#nf-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'c':S.view);
const rowsNow=()=>mode==='sandbox'?ALL:MAIN;
const showRange=()=>!(mode==='guided'&&step<SIX('range'));
const blkNow=()=>mode==='sandbox'?BLKS[S.blk]:0;
function bitsHTML(r,bw){
  const f=FM[r.f], cell=(b,cls)=>`<i class="nf-bit ${cls}" style="width:${bw}px">${b}</i>`;
  const E=Math.max(0,r.E), M=Math.max(0,r.M), eb=E.toString(2).padStart(f.e,'0'), mb=M.toString(2).padStart(f.m,'0');
  return `<span class="nf-fld">${cell(r.S,'nf-sgn')}</span><span class="nf-fld">${[...eb].map(b=>cell(b,'nf-exp')).join('')}</span><span class="nf-fld">${[...mb].map(b=>cell(b,'nf-man')).join('')}</span>`;
}
const sg=v=>v<0?MINUS+(-v):String(v);
const fxv=x=>{const s=fin(x); return parseFloat(s.replace(/\u2011/g,'-').replace(/,/g,''))===x?s:fv(x);};
function decodeTip(r){
  const f=FM[r.f], x=fxv(r.x);
  if(r.kind==='over') return tx(`${x} is larger than ${f.n}’s largest number, ${fv(f.max)}: it overflows${f.k==='ieee'?' to Inf (exponent all ones, mantissa 0)':(f.k==='fn'?'; E4M3 has no Inf, so the cast gives NaN, or 448 if it saturates':`; FP4 has no Inf or NaN, so the cast clamps it to ${fv(r.v)}, keeping the sign (OCP MX), off by ${fpc(Math.abs(r.v-r.x)/Math.abs(r.x))}`)}.`,
    `${x} 比 ${f.n} 最大的数 ${fv(f.max)} 还大：它溢出了${f.k==='ieee'?'，变成 Inf（指数全 1，尾数 0）':(f.k==='fn'?'；E4M3 没有 Inf，转换结果是 NaN，饱和转换则是 448':`；FP4 没有 Inf 和 NaN，转换时把它截到 ${fv(r.v)}，保留符号（OCP MX），误差 ${fpc(Math.abs(r.v-r.x)/Math.abs(r.x))}`)}。`);
  if(r.kind==='lost') return tx(`${x} is less than half of ${f.n}’s smallest number, ${fv(f.minS)}, so it rounds to 0: every bit is 0.`,`${x} 不到 ${f.n} 最小的数 ${fv(f.minS)} 的一半，所以舍入成 0：每一位都是 0。`);
  const sc=r.sc!==1?tx(` It was divided by the block’s scale ${fv(r.sc)} first and multiplied back after.`,` 它先除以块的缩放系数 ${fv(r.sc)}，存完再乘回去。`):'';
  if(r.kind==='sub') return tx(`${x} is stored as ${fv(r.v)}, a subnormal number: exponent field 0, so its value is ${A.ui.math(`\\frac{${r.M}}{2^{${f.m}}} \\times 2^{${sg(f.emin)}}`)}.${sc}`,`${x} 存成 ${fv(r.v)}，是次正规数：指数字段是 0，所以数值是 ${A.ui.math(`\\frac{${r.M}}{2^{${f.m}}} \\times 2^{${sg(f.emin)}}`)}。${sc}`);
  return tx(`${x} is stored as ${fv(r.v)}: exponent field ${r.E} − bias ${f.b} = ${sg(r.E-f.b)}, mantissa ${r.M} of ${2**f.m}, so its value is ${A.ui.math(`\\left(1 + \\frac{${r.M}}{${2**f.m}}\\right) \\times 2^{${sg(r.E-f.b)}}`)}.${sc}`,
    `${x} 存成 ${fv(r.v)}：指数字段 ${r.E} − 偏移 ${f.b} = ${sg(r.E-f.b)}，尾数 ${r.M}（共 ${2**f.m} 档），所以数值是 ${A.ui.math(`\\left(1 + \\frac{${r.M}}{${2**f.m}}\\right) \\times 2^{${sg(r.E-f.b)}}`)}。${sc}`);
}
function drawA(){
  const x=X(), ids=rowsNow(), host=$('#nf-rows',app), W=host.clientWidth||520, bw=Math.max(7,Math.min(17,Math.floor((W-14)/32)));
  $('#nf-a-cap',app).textContent=tx(`How each format stores ${fin(x)}`,`每种格式怎样存 ${fin(x)}`);
  const blk=blkNow();
  host.innerHTML=ids.map(id=>{const r=store(x,id,blk), f=FM[id], b=bad(r), sc=blk&&SMALL[id];
    return `<div class="nf-row${b?' nf-bad':''}" data-f="${id}"><div class="nf-rh"><span class="nf-fn" tabindex="0" data-tipx="${esc(tx(`${f.n}: ${f.bits} bits, 1 sign bit, ${f.e} exponent bits and ${f.m} mantissa bit${f.m===1?'':'s'}.`,`${f.n}：${f.bits} 位，1 个符号位、${f.e} 个指数位和 ${f.m} 个尾数位。`))}">${f.n}${sc?`<em>${tx(' scaled',' 已缩放')}</em>`:''}</span>`+
      `<span class="nf-val mono">${fv(r.v)}</span><span class="nf-err mono${b?' nf-bad':''}">${fe(r)}</span></div>`+
      `<div class="nf-bits" tabindex="0" data-tipx="${esc(decodeTip(r))}">${bitsHTML(r,bw)}</div></div>`;}).join('');
  const rg=$('#nf-range',app); rg.hidden=!showRange(); if(showRange()) rg.innerHTML=rangeHTML(ids,x,blk);
}
/* the range strips: a log axis from 1e-12 to 1e7; FP32 and BF16 continue past both ends */
const AX0=-12, AX1=7, pos=v=>Math.max(0,Math.min(1,(Math.log10(v)-AX0)/(AX1-AX0)));
const pc=p=>(p*100).toFixed(3)+'%';
function rangeHTML(ids,x,blk){
  const band=(lo,hi,t,tip,cls)=>`<span class="nf-band ${cls||''}" style="left:${pc(pos(lo))};width:${pc(pos(hi)-pos(lo))}" tabindex="0" data-tipx="${esc(tip)}"><em>${esc(t)}</em></span>`;
  const tick=(v,t,tip,cls)=>`<span class="nf-tmark ${cls||''}" style="left:${pc(pos(v))}" tabindex="0" data-tipx="${esc(tip)}"><em>${esc(t)}</em></span>`;
  const marks=`<div class="nf-tens"><div class="nf-trow">`+band(1e-9,1e-3,tx('gradients','梯度'),tx('Gradients: often from about 1e‑9 to 1e‑3 (illustrative).','梯度：常在约 1e‑9 到 1e‑3 之间（示意）。'))+tick(0.02,tx('weights','权重'),tx('Weights: typically near 0.02 (illustrative).','权重：通常在 0.02 左右（示意）。'))+
    `</div><div class="nf-trow">`+tick(3e-4,tx('Adam updates','Adam 更新量'),TIP.upd())+`</div>`+
    `<div class="nf-trow">`+band(1e3,1e4,tx('activation outliers','激活离群点'),tx('Activations: most are near 1, with outliers of 1e3 to 1e4 in large models (illustrative).','激活值：多数在 1 左右，大模型里有 1e3 到 1e4 的离群点（示意）。'),'nf-right')+`</div></div>`;
  const strips=ids.map(id=>{const f=FM[id], r=store(x,id,blk), sc=blk&&SMALL[id]?scaleOf(x,id,blk):1, lo=f.minS*sc, mn=f.minN*sc, hi=f.max*sc, b=bad(r);
    const l=pos(lo), m=pos(mn), h=pos(hi), more0=lo<10**AX0, more1=hi>10**AX1;
    const tip=tx(`${f.n}${sc!==1?' with the block scale':''}: subnormal numbers from ${fv(lo)}, normal numbers from ${fv(mn)} up to ${fv(hi)}.`,`${f.n}${sc!==1?'（按块缩放后）':''}：次正规数从 ${fv(lo)} 开始，正规数从 ${fv(mn)} 到 ${fv(hi)}。`);
    return `<div class="nf-strip" data-f="${id}"><span class="nf-sl">${f.n}</span><span class="nf-st" tabindex="0" data-tipx="${esc(tip)}">`+
      `<i class="nf-sub" style="left:${pc(l)};width:${pc(Math.max(0,m-l))}"></i><i class="nf-norm" style="left:${pc(m)};width:${pc(Math.max(0,h-m))}"></i>`+
      (more0?`<b class="nf-more nf-left${pos(x)<0.24?' nf-short':''}">◂<span> ${fv(lo)}</span></b>`:'')+(more1?`<b class="nf-more nf-right${pos(x)>0.76?' nf-short':''}"><span>${fv(hi)} </span>▸</b>`:'')+
      `<b class="nf-dot${b?' nf-bad':''}" style="left:${pc(pos(x))}"></b></span></div>`;}).join('');
  const axis=[-12,-8,-4,0,4].map(e=>`<span style="left:${pc((e-AX0)/(AX1-AX0))}">${e===0?'1':'1e'+(e<0?NBH+(-e):e)}</span>`).join('');
  return `<div class="nf-sub2">${tx('Range of each format on a log scale; the line is the value','每种格式的范围（对数刻度）；竖线是这个数')}</div>`+marks+`<div class="nf-strips">${strips}<b class="nf-vline" style="left:calc(var(--nf-sl) + (100% - var(--nf-sl)) * ${pos(x).toFixed(5)})"></b></div><div class="nf-axis">${axis}</div>`;
}
/* the update view: number lines from 1.0 to 1.0125 */
const Z0=1, Z1=1.0125;
function drawB(){
  const u=U(), host=$('#nf-zoom',app), zp=v=>(v-Z0)/(Z1-Z0);
  $('#nf-b-cap',app).textContent=tx(`The weight 1.0 plus an update of ${fin(u)}: the numbers each format can store near 1.0`,`大小为 1.0 的权重加上 ${fin(u)} 的更新：每种格式在 1.0 附近能存的数`);
  host.innerHTML=MAIN.map(id=>{const f=FM[id], g=f.gap1, r=quant(1+u,id), lost=r.v===1, dense=g<0.0002;
    const ticks=dense?`<i class="nf-dense"></i>`:Array.from({length:Math.floor((Z1-Z0)/g+1e-9)+1},(_,k)=>`<i class="nf-zt" style="left:${pc(zp(1+k*g))}"></i>`).join('');
    const half=dense?'':`<i class="nf-zh" style="left:${pc(zp(1+g/2))}"></i>`;
    const tip=dense?tx(`FP32 can store a number every ${fv(g)} near 1.0, too close to draw; 1.0 + ${fin(u)} is stored as ${fv(r.v)}.`,`FP32 在 1.0 附近每隔 ${fv(g)} 就有一个数，密得画不出来；1.0 + ${fin(u)} 存成 ${fv(r.v)}。`)
      :tx(`${f.n} can store a number every ${fv(g)} near 1.0 (the ticks); the dashed tick is halfway, ${fv(1+g/2)}. 1.0 + ${fin(u)} is ${lost?'below it, so it rounds back to 1':`stored as ${fv(r.v)}`}.`,`${f.n} 在 1.0 附近每隔 ${fv(g)} 有一个数（刻度）；虚线刻度在一半处，${fv(1+g/2)}。1.0 + ${fin(u)} ${lost?'在它下面，所以舍入回 1':`存成 ${fv(r.v)}`}。`);
    return `<div class="nf-zr${lost?' nf-bad':''}"><div class="nf-zhd"><span class="nf-fn">${f.n}</span><span class="nf-val mono">${fv(r.v)}</span><span class="nf-err mono${lost?' nf-bad':''}">${lost?tx('update lost','更新丢了'):tx('kept','留住了')}</span></div>`+
      `<div class="nf-zl" tabindex="0" data-tipx="${esc(tip)}">${ticks}${half}<b class="nf-zx" style="left:${pc(zp(1+u))}"></b><b class="nf-zs${lost?' nf-bad':''}" style="left:${pc(zp(r.v))}"></b></div></div>`;}).join('')+
    `<div class="nf-axis nf-zaxis">${[1,1.005,1.01].map(v=>`<span style="left:${pc(zp(v))}">${fixd(v,3)}</span>`).join('')}</div>`;
}
/* the recipe view: the gradient's path, the update's, and the weight over 40 updates */
const NSTEP=40;
function tensorsNow(){return mode==='challenge'?CHT:T5(mode==='guided'?3e-4:U());}
function drawC(){
  const r=curRecipe(), T=tensorsNow(), e=recipe(r,T), gF=FM[r.g].n, wF=FM[r.w].n, hc=$('#nf-paths',app);
  $('#nf-c-cap',app).textContent=tx(`One training step with ${recipeName(r)}`,`用 ${recipeName(r)} 训练一步`);
  const st=(t,ok,tip)=>`<span class="nf-step${ok===false?' nf-bad':(ok?' nf-ok':'')}" tabindex="0" data-tipx="${esc(tip)}">${t}</span>`, ar='<span class="nf-ar">→</span>';
  const g=[st(tx(`gradient ${fin(T.g)}`,`梯度 ${fin(T.g)}`),null,tx(`The smallest gradient in this step: ${fin(T.g)}.`,`这一步里最小的梯度：${fin(T.g)}。`))];
  if(r.ls>1) g.push(ar,st(`× ${fint(r.ls)}`,null,TIP.ls()));
  g.push(ar,st(tx(`stored in ${gF}: ${fv(e.g.v)}`,`存成 ${gF}：${fv(e.g.v)}`),!e.lost,decodeTip(e.g)));
  if(r.ls>1) g.push(ar,st(tx(`÷ ${fint(r.ls)} in FP32`,`在 FP32 里 ÷ ${fint(r.ls)}`),null,tx('Dividing back in FP32 keeps the small value, since FP32 reaches down to 1.4e‑45.','在 FP32 里除回去，小值也留得住，因为 FP32 能到 1.4e‑45。')));
  g.push(ar,st(tx(`used: ${fv(e.gUsed)}`,`用到的：${fv(e.gUsed)}`),!e.lost,tx('The gradient the optimizer sees.','优化器看到的梯度。')));
  const u=[st(tx(`weight ${fv(T.w)} + ${fin(T.u)}`,`权重 ${fv(T.w)} + ${fin(T.u)}`),null,TIP.upd()),ar,
    st(r.ms?tx('added in the FP32 master copy','加到 FP32 主副本上'):tx(`added to the ${wF} weight`,`加到 ${wF} 权重上`),null,r.ms?TIP.master():tx(`The weight lives only in ${wF}; the sum is rounded to ${wF} again.`,`权重只存成 ${wF}；相加之后再舍入成 ${wF}。`)),ar,
    st(tx(`result ${fv(e.w1)}`,`结果 ${fv(e.w1)}`),e.kept,e.kept?tx(`The weight moved from ${fv(e.w0)} to ${fv(e.w1)}.`,`权重从 ${fv(e.w0)} 变成 ${fv(e.w1)}。`):tx(`The sum rounded back to ${fv(e.w0)}: the update is lost.`,`相加的结果舍入回 ${fv(e.w0)}：更新丢了。`))];
  const big=[st(tx(`gradient ${fin(T.gmax)}`,`梯度 ${fin(T.gmax)}`),null,tx(`The largest gradient in this step: ${fin(T.gmax)}.`,`这一步里最大的梯度：${fin(T.gmax)}。`))];
  if(r.ls>1) big.push(ar,st(`× ${fint(r.ls)} = ${fv(T.gmax*r.ls)}`,null,TIP.ls()));
  big.push(ar,st(e.over?tx(`overflows ${gF}: ${fv(e.gBig.v)}`,`超出 ${gF}：${fv(e.gBig.v)}`):tx(`stored in ${gF}: ${fv(e.gBig.v)}`,`存成 ${gF}：${fv(e.gBig.v)}`),!e.over,decodeTip(e.gBig)));
  let u2='';
  if(T.w2!=null) u2=`<div class="nf-pline"><span class="nf-pl">${tx('Update on a gain','gain 上的更新')}</span>`+[st(tx(`gain ${fv(T.w2)} + ${fin(T.u)}`,`gain ${fv(T.w2)} + ${fin(T.u)}`),null,TIP.gain()),ar,
    st(r.ms?tx('added in the FP32 master copy','加到 FP32 主副本上'):tx(`added to the ${wF} gain`,`加到 ${wF} 的 gain 上`),null,r.ms?TIP.master():tx(`The gain lives only in ${wF}; the sum is rounded to ${wF} again.`,`gain 只存成 ${wF}；相加之后再舍入成 ${wF}。`)),ar,
    st(tx(`result ${fv(e.v1)}`,`结果 ${fv(e.v1)}`),e.kept2,e.kept2?tx(`The gain moved from ${fv(e.v0)} to ${fv(e.v1)}.`,`gain 从 ${fv(e.v0)} 变成 ${fv(e.v1)}。`):tx(`The sum rounded back to ${fv(e.v0)}: the update is lost.`,`相加的结果舍入回 ${fv(e.v0)}：更新丢了。`))].join('')+`</div>`;
  hc.innerHTML=`<div class="nf-pline"><span class="nf-pl">${tx('Smallest gradient','最小的梯度')}</span>${g.join('')}</div><div class="nf-pline"><span class="nf-pl">${tx('Largest gradient','最大的梯度')}</span>${big.join('')}</div><div class="nf-pline"><span class="nf-pl">${tx(T.w2!=null?'Update on a weight':'Update',T.w2!=null?'权重上的更新':'更新')}</span>${u.join('')}</div>`+u2;
  $('#nf-chart-cap',app).textContent=tx(`The weight after each of ${NSTEP} updates of ${fin(T.u)}`,`${NSTEP} 次 ${fin(T.u)} 的更新后，权重每一步的值`);
  $('#nf-chart',app).innerHTML=chartSVG(r,T);
}
function chartSVG(r,T){
  const s=series(r,T,NSTEP), host=$('#nf-chart',app), W=Math.max(260,Math.round(host.clientWidth||480)), H=150, pl=64, pr=8, pt=10, pb=22;
  const all=(r.ms?s.m.concat(s.c):[]).concat(s.o,[T.w,T.w+NSTEP*T.u]), lo=Math.min(...all), hi=Math.max(...all), pad=(hi-lo)*0.06||T.u;
  const y0=lo-pad, y1=hi+pad, X=k=>pl+(W-pl-pr)*k/NSTEP, Y=v=>pt+(H-pt-pb)*(1-(v-y0)/(y1-y0));
  const path=(a,stairs)=>a.map((v,k)=>k===0?`M${X(0).toFixed(1)} ${Y(v).toFixed(1)}`:(stairs?`H${X(k).toFixed(1)}V${Y(v).toFixed(1)}`:`L${X(k).toFixed(1)} ${Y(v).toFixed(1)}`)).join('');
  const exact=`<path d="M${X(0)} ${Y(T.w).toFixed(1)}L${X(NSTEP)} ${Y(T.w+NSTEP*T.u).toFixed(1)}" class="nf-lx"/>`;
  const lines=`<path d="${path(s.o,true)}" class="nf-lo"/>`+(r.ms?`<path d="${path(s.m,false)}" class="nf-lm"/><path d="${path(s.c,true)}" class="nf-lc"/>`:'');
  const yl=(v)=>`<text x="${pl-6}" y="${(Y(v)+3.5).toFixed(1)}" text-anchor="end">${esc(fv(v))}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(tx(`Weight after each update, from ${fv(T.w)}; the exact sum ends at ${fv(T.w+NSTEP*T.u)}.`,`每次更新后的权重，从 ${fv(T.w)} 开始；精确的和最后是 ${fv(T.w+NSTEP*T.u)}。`))}"><line x1="${pl}" y1="${H-pb}" x2="${W-pr}" y2="${H-pb}" class="nf-ax"/><line x1="${pl}" y1="${pt}" x2="${pl}" y2="${H-pb}" class="nf-ax"/>${exact}${lines}`+
    yl(T.w)+yl(T.w+NSTEP*T.u)+`<text x="${pl}" y="${H-6}">${tx('update 0','第 0 次')}</text><text x="${W-pr}" y="${H-6}" text-anchor="end">${tx(`update ${NSTEP}`,`第 ${NSTEP} 次`)}</text></svg>`;
}
function drawStage(){
  const v=viewNow();
  $('#nf-a',app).hidden=v!=='a'; $('#nf-b',app).hidden=v!=='b'; $('#nf-c',app).hidden=v!=='c';
  if(v==='a') drawA(); else if(v==='b') drawB(); else drawC();
}
function statusOf(){
  const v=viewNow();
  if(v==='c'){const e=recipe(curRecipe(),tensorsNow()), n=(e.lost?1:0)+(e.kept?0:1)+(e.over?1:0);
    if(mode==='challenge') return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),tip:tx('Green when weights and gradients are 16-bit, no gradient overflows, and no gradient or update is lost.','权重和梯度是 16 位、梯度不溢出、梯度和更新都没丢时，徽章就是绿色。')};
    return {cls:n?'bad':'ok',txt:n?tx(`${n} of 3 checks fail`,`3 项检查有 ${n} 项失败`):tx('All 3 checks pass','3 项检查都通过'),tip:tx('How many of the step’s three checks fail: the smallest gradient is not lost, the largest gradient does not overflow, the update is kept.','这一步的 3 项检查里有几项失败：最小的梯度不丢、最大的梯度不溢出、更新留住。')};}
  if(v==='b'){const n=MAIN.filter(id=>quant(1+U(),id).v===1).length;
    return {cls:n?'bad':'ok',txt:n?tx(`Lost in ${n} of 3`,`3 种里丢了 ${n} 种`):tx('Kept in all','都留住了'),tip:tx('In how many of FP32, BF16 and FP16 the update rounds away.','FP32、BF16、FP16 里有几种把这次更新舍入掉了。')};}
  const blk=blkNow(), ids=rowsNow(), rs=ids.map(id=>store(X(),id,blk)), nl=rs.filter(r=>r.kind==='lost').length, no=rs.filter(r=>r.kind==='over').length, k=ids.length;
  const txt=nl&&no?tx(`Lost or overflows in ${nl+no} of ${k}`,`${k} 种里 ${nl+no} 种丢失或溢出`):(nl?tx(`Lost in ${nl} of ${k}`,`${k} 种里丢了 ${nl} 种`):(no?tx(`Overflows in ${no} of ${k}`,`${k} 种里 ${no} 种溢出`):tx('All keep it','都存得下')));
  return {cls:nl+no?'bad':'ok',txt,tip:tx('How many of the formats in the chart lose the value (round it to 0) or overflow (it is larger than their largest number).','图里的格式有几种把这个数丢成 0（舍入成 0），或者溢出（它比格式最大的数还大）。')};
}
function drawHead(){
  const v=viewNow(), t=$('#nf-st-title',app), sub=$('#nf-st-sub',app), st=$('#nf-status',app);
  t.textContent=v==='a'?(showRange()?tx('Bits and range: one value in each format','位和范围：同一个数存成每种格式'):tx('Bits: one value in each format','位：同一个数存成每种格式')):(v==='b'?tx('Precision: a small update on a weight of 1.0','精度：给大小为 1.0 的权重加一个小更新'):tx('Mixed precision: one training step','混合精度：一步训练'));
  sub.textContent=v==='a'?tx(`value ${fin(X())}${blkNow()?`, FP8 and FP4 scaled per block (largest ${fint(blkNow())}×)`:''}`,`数值 ${fin(X())}${blkNow()?`，FP8 和 FP4 按块缩放（最大 ${fint(blkNow())} 倍）`:''}`)
    :(v==='b'?tx(`update ${fin(U())}`,`更新量 ${fin(U())}`):(mode==='challenge'?tx(`${M2}: weights 0.02 and gains 1.0, update ${fin(CHT.u)}, gradients 1e‑8 to 1e‑3`,`${M2}：权重约 0.02，gain 约 1.0，更新 ${fin(CHT.u)}，梯度 1e‑8 到 1e‑3`):tx(`weight 1.0, update ${fin(tensorsNow().u)}, gradients 1e‑8 to 1e‑3`,`权重为 1.0，更新 ${fin(tensorsNow().u)}，梯度 1e‑8 到 1e‑3`)));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const vw=$('#nf-view',app); vw.hidden=mode!=='sandbox'; $$('button',vw).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===S.view)));
}
/* R5.9: each sub-chart has its own key, right under it: the bit rows, the range chart, the zoom, the recipe's steps, the weight chart */
function keysHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`, v=viewNow();
  if(v==='a') return {rows:it('nf-sw-s',tx('sign bit','符号位'),TIP.sign())+it('nf-sw-e',tx('hatched: exponent bits','斜纹：指数位'),TIP.exp())+it('nf-sw-m',tx('plain: mantissa bits','空白：尾数位'),TIP.man())+it('nf-sw-bad',tx('red: lost or overflow','红色：丢失或溢出'),tx('Red marks a value a format cannot hold: it rounds to 0 or overflows.','红色表示这种格式存不下这个数：舍入成 0 或溢出。')),
    range:showRange()?it('nf-sw-sub',tx('hatched stretch: subnormal','斜纹段：次正规数'),TIP.sub())+it('nf-sw-v',tx('line: the value','竖线：这个数'),tx('The vertical line is the value being stored.','竖线是正在存的这个数。'))+it('nf-sw-band',tx('bands: typical tensors (illustrative)','色带：典型张量（示意）'),TIP.illus()):''};
  if(v==='b') return {zoom:it('nf-sw-t',tx('tick: a number the format can store','刻度：格式能存的数'),tx('Each tick is a number the format can store near 1.0.','每个刻度是这种格式在 1.0 附近能存的一个数。'))+it('nf-sw-h',tx('dashed tick: halfway','虚线刻度：一半处'),tx('Below the halfway tick, 1.0 + update rounds down to 1.0.','在虚线刻度以下，1.0 + 更新量会舍入回 1.0。'))+
    it('nf-sw-x',tx('triangle: 1.0 + update, exact','三角：1.0 + 更新量的精确值'),tx('The exact sum before rounding.','舍入之前的精确和。'))+it('nf-sw-d',tx('dot: what is stored','圆点：存下的数'),tx('The number the format stores after rounding; red when it is still 1.0.','舍入后存下的数；仍是 1.0 时为红色。'))};
  const r=curRecipe();
  return {paths:it('nf-sw-ok',tx('green: kept','绿色：留住了'),tx('A step that keeps the value.','留住了数值的一步。'))+it('nf-sw-bad',tx('red: lost','红色：丢了'),tx('A step where the value is lost or overflows.','数值丢失或溢出的一步。')),
    chart:it('nf-sw-lx',tx('dotted: exact sum','点线：精确的和'),tx('The weight if every update were added exactly.','每次更新都精确加上时的权重。'))+
    (r.ms?it('nf-sw-lm',tx('thick line: FP32 master copy','粗线：FP32 主副本'),TIP.master())+it('nf-sw-lc',tx(`stairs: its ${FM[r.w].n} copy`,`台阶：它的 ${FM[r.w].n} 副本`),tx('The 16-bit copy rounded from the master copy for the next forward pass.','从主副本舍入得到的 16 位副本，给下一次前向传播用。')):'')+
    it('nf-sw-lo',tx(`thick dashed: a weight kept only in ${FM[r.w].n}`,`粗虚线：只用 ${FM[r.w].n} 存的权重`),tx('A weight kept only in 16 bits, for comparison: each update is added to it and rounded on its own.','只用 16 位存的权重，用来对比：每次更新加到它上面，单独舍入。'))};
}
/* metric cards: ▲▼ only after the learner's own change, from the values as shown (R6.6); scientific values get no badge */
const numOf=v=>{const s=String(v).replace(/,/g,''); if(/e/.test(s)) return NaN; const m=s.match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?\d+\.(\d+)/); return m?m[1].length:0;};
/* a card shows at most 6 significant digits (R7.8 at 390 px); its tooltip keeps the full value */
const fvc=v=>{if(!isFinite(v)||v===0) return fv(v); const a=Math.abs(v); if(a>=1e-3&&a<1e6){const t=a.toPrecision(6), u=t.indexOf('.')>=0?t.replace(/0+$/,'').replace(/\.$/,''):t, [i,d]=u.split('.'); return (v<0?MINUS:'')+i.replace(/\B(?=(\d{3})+(?!\d))/g,',')+(d?'.'+d:'');} return fv(v);};
const errSub=r=>bad(r)?(r.kind==='over'?tx('overflow','溢出'):tx('lost','丢失')):(relErr(r)===0?tx('exact','精确'):tx(`error ${fe(r)}`,`误差 ${fe(r)}`));
function metricsDef(){
  const v=viewNow();
  if(v==='c'){const r=curRecipe(), e=recipe(r,tensorsNow());
    return [{k:'g',label:tx('Gradient the optimizer uses','优化器用到的梯度'),v:fvc(e.gUsed),sub:e.lost?tx('lost','丢了'):tx('kept','留住了'),tip:tx(`The smallest gradient, ${fin(tensorsNow().g)}, after storage${r.ls>1?' and unscaling':''}: ${fv(e.gUsed)}.`,`最小的梯度 ${fin(tensorsNow().g)} 存下${r.ls>1?'并除回去':''}之后是 ${fv(e.gUsed)}。`)},
      {k:'w',label:tx('Weight after the update','更新后的权重'),v:fvc(e.w1),sub:e.kept?tx('moved','动了'):tx('unchanged','没动'),tip:tx(`The weight after one update: ${fv(e.w1)}${e.kept?'':', the same as before'}.`,`一次更新后的权重：${fv(e.w1)}${e.kept?'':'，和之前一样'}。`)},
      {k:'bytes',label:tx('Bytes per parameter','每个参数的字节'),v:String(e.bytes),sub:r.ms?tx('weights + gradients + master copy','权重 + 梯度 + 主副本'):tx('weights + gradients','权重 + 梯度'),tip:tx(`${FM[r.w].bits/8} for the weight, ${FM[r.g].bits/8} for the gradient${r.ms?' and 4 for the FP32 master copy':''}; Adam’s m and v are not counted.`,`权重 ${FM[r.w].bits/8} 字节，梯度 ${FM[r.g].bits/8} 字节${r.ms?'，FP32 主副本 4 字节':''}；不算 Adam 的 m 和 v。`)}];}
  if(v==='b'){const u=U();
    return MAIN.map(id=>{const r=quant(1+u,id), lost=r.v===1; return {k:'b'+id,label:tx(`${FM[id].n}: 1.0 + update`,`${FM[id].n}：1.0 + 更新量`),v:fvc(r.v),sub:lost?tx('update lost','更新丢了'):tx('kept','留住了'),tip:tx(`1.0 + ${fin(u)} stored in ${FM[id].n}: ${fv(r.v)}.`,`1.0 + ${fin(u)} 存成 ${FM[id].n}：${fv(r.v)}。`)};});}
  const x=X(), blk=blkNow();
  if(blk) return ['e4m3','e5m2','fp4'].map(id=>{const r=store(x,id,blk); return {k:'s'+id,label:tx(`${FM[id].n} with a block scale`,`${FM[id].n}（按块缩放）`),v:fvc(r.v),sub:errSub(r),tip:decodeTip(r)};});
  return MAIN.map(id=>{const r=quant(x,id); return {k:'a'+id,label:tx(`${FM[id].n} stores`,`${FM[id].n} 存成`),v:fvc(r.v),sub:errSub(r),tip:decodeTip(r)};});
}
function drawMetrics(){
  const host=$('#nf-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&(a===0||b===0||Math.max(a/b,b/a)<=100)) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: decode each row; the gap and the halfway mark; the recipe's arithmetic */
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each block is one lead line in
   words, the quantity's name and its formula, the reader's numbers under it, then the result. The rows form a grid, so each number
   sits under the term it replaces and the “=” signs line up (R6.29). A cell is TeX, or {h:html}; inputs keep their terms (R1.4) */
const NMG=rows=>{const n=Math.max(...rows.map(r=>r.length));
  return `<div class="nf-mg" style="grid-template-columns:repeat(${n},auto)">`+rows.map(r=>r.map((c,i)=>{const w=i===r.length-1&&r.length<n;
    return `<span class="nf-mc${i===0?' nf-mn':''}${w?' nf-mw':''}${c&&c.h!=null?' nf-mh':''}"${w?` style="grid-column:${i+1}/-1"`:''}>${c==null||c===''?'':(c.h!=null?c.h:A.ui.math(c))}</span>`;}).join('')).join('')+'</div>';};
const NMB=(lead,...grids)=>`<div class="nf-mb"><p class="nf-ml">${lead}</p>${grids.map(NMG).join('')}</div>`;
const NTX=x=>'\\text{'+String(x)+'}';
function mathHTML(){
  const v=viewNow(), t=T_, H=h=>({h});
  const head=NMB(tx('A floating-point number is a sign bit s, an exponent field E and a mantissa M. The format has m mantissa bits and a bias b:','一个浮点数由符号位 s、指数字段 E 和尾数 M 组成。格式有 m 个尾数位，偏移（bias）是 b：'),
    [[NTX(tx('value','数值')),'=','(-1)^s \\times \\left(1 + \\frac{M}{2^m}\\right) \\times 2^{E - b}'],[NTX(tx('subnormal (E = 0)','次正规数（E = 0）')),'=','(-1)^s \\times \\frac{M}{2^m} \\times 2^{1 - b}']]);
  if(v==='a'){const x=X(), blk=blkNow();
    return head+rowsNow().map(id=>{const r=store(x,id,blk), f=FM[id];
      return `<div class="nf-mb"><p class="nf-ml"><b>${f.n}</b> (m = ${f.m}, b = ${f.b})${tx(': ','：')}${decodeTip(r)}</p></div>`;}).join('')+`<div class="nf-mb"><p class="nf-ml">${tx(`The ${t('x','value')} is rounded to the nearest number the format can store. Exactly halfway, it goes to the one whose last mantissa bit is 0.`,`${t('x','这个数')}舍入到格式能存的最近的数。正好在中间时，取尾数最后一位是 0 的那个。`)}</p></div>`;}
  if(v==='b'){const u=U();
    return NMB(tx(`Near 1.0 a format with m mantissa bits can only store steps of one gap. An ${t('u','update')} smaller than half a gap is lost:`,`在 1.0 附近，有 m 个尾数位的格式只能按一个间隔一格一格地存。${t('u','更新量')}不到半个间隔，就会被舍掉：`),
      [[NTX(tx('gap at 1.0','1.0 处的间隔')),'=','2^{-m}']])+
      MAIN.map(id=>{const f=FM[id], g=f.gap1, r=quant(1+u,id), c=u<g/2?'<':(u===g/2?'=':'>');
      return NMB(`<b>${f.n}</b> (m = ${f.m})${tx(': ','：')}${tx(r.v===1?'the update is lost.':'the update is kept.',r.v===1?'更新被舍掉了。':'更新留下了。')}`,
        [[NTX(tx('gap','间隔')),'=',`2^{-${f.m}}`,'=',H(fv(g))],[NTX(tx('update','更新量')),c,'\\frac{'+NTX(tx('gap','间隔'))+'}{2}'],['','',H(`${fin(u)} ${c} ${fv(g/2)}`)],['1 + '+NTX(tx('update','更新量')),'\\to',H(fv(r.v))]]);}).join('');}
  const r=curRecipe(), T=tensorsNow(), e=recipe(r,T), gF=FM[r.g].n;
  let h=NMB(r.ls>1?tx(`The gradient is multiplied by the loss scale before it is stored in ${gF}, and divided back in FP32 before it is used:`,`梯度先乘上 loss scale，再存成 ${gF}；用之前在 FP32 里除回去：`):tx(`The gradient is stored in ${gF} as it is:`,`梯度直接存成 ${gF}：`),
    [[NTX(tx('gradient','梯度')),'=',H(fin(T.g))]].concat(r.ls>1?[[NTX(tx('× loss scale','乘上 loss scale')),'=',H(`${fin(T.g)} × ${t('ls',fint(r.ls))}`),'=',H(fv(T.g*r.ls))]]:[],
      [[NTX(tx(`stored in ${gF}`,`存成 ${gF}`)),'=',H(fv(e.g.v))]],r.ls>1?[[NTX(tx('÷ loss scale, in FP32','在 FP32 里除回去')),'=',H(`${fv(e.g.v)} ÷ ${fint(r.ls)}`),'=',H(fv(e.gUsed))]]:[]));
  h+=NMB(tx(`The largest gradient must fit too; ${gF}’s largest number is ${fv(FM[r.g].max)}:`,`最大的梯度也要存得下；${gF} 最大的数是 ${fv(FM[r.g].max)}：`),
    [[NTX(tx('largest gradient','最大的梯度')),'=',H(fin(T.gmax))]].concat(r.ls>1?[[NTX(tx('× loss scale','乘上 loss scale')),'=',H(`${fin(T.gmax)} × ${fint(r.ls)}`),'=',H(fv(T.gmax*r.ls))]]:[],[[NTX(tx(`stored in ${gF}`,`存成 ${gF}`)),'=',H(fv(e.gBig.v))]]));
  const wT=r.ms?t('ms',tx('FP32 master copy','FP32 主副本')):FM[r.w].n;
  h+=NMB(tx(`The update is added to the weight and stored in the `,`更新量加到权重上，存进`)+wT+tx(':','：'),
    [[NTX(tx('weight','权重')),'=',H(`${fv(e.w0)} + ${t('u',fin(T.u))}`),'\\to',H(`${fv(e.w1)}${e.kept?'':tx(' (unchanged)','（没变）')}`)]].concat(T.w2!=null?[[NTX(tx('gain','gain')),'=',H(`${fv(e.v0)} + ${t('u',fin(T.u))}`),'\\to',H(`${fv(e.v1)}${e.kept2?'':tx(' (unchanged)','（没变）')}`)]]:[]));
  h+=NMB(tx('Each parameter keeps its weight and its gradient, plus the FP32 master copy if there is one:','每个参数要存权重和梯度；有 FP32 主副本的话，再加上它：'),
    [[NTX(tx('bytes per parameter','每个参数的字节')),'=',H(`${FM[r.w].bits/8} + ${FM[r.g].bits/8}${r.ms?' + 4':''}`),'=',H(String(e.bytes))]]);
  return h;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  if(isP('range')) seenX.add(S.xi); if(isP('prec')) seenU.add(S.ui);
  drawHead();
  const gr=$('#nf-graph',app); gr.dataset.focus=mode==='guided'?focusOf(STEPS[step]):'';
  drawStage(); drawMetrics();
  const lg=$('#nf-legend',app); if(lg){lg.innerHTML=''; lg.hidden=true;}
  const ks=keysHTML(); ['rows','range','zoom','paths','chart'].forEach(k=>{const el=$('#nf-key-'+k,app); if(!el) return; const h=ks[k]||''; if(el._h!==h){el.innerHTML=h; el._h=h;} el.hidden=!h;});
  const mb=$('#nf-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you pick only the recipe, in the task card. Everything else is fixed: ${M2}, weights near 0.02 and gains of 1.0, an update of ${fin(CHT.u)} (the learning rate late in pretraining), gradients from 1e‑8 to 1e‑3, a loss scale of 65,536.`,`挑战中只能在任务卡里选做法。其余固定：${M2}，权重约 0.02，gain 是 1.0，更新 ${fin(CHT.u)}（预训练后期的学习率），梯度 1e‑8 到 1e‑3，loss scale 65,536。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'||id==='predict') return tx('Fixed in this part: the value 0.02 in FP32, BF16 and FP16. Part 3 lets you change the value.','这一部分固定：数值 0.02，格式 FP32、BF16、FP16。第 3 部分可以改这个数。');
  if(id==='range') return tx('In this part you change only the value, in the card or here. You can change everything in Scenarios.','这一部分只改这个数，在卡片里或这里改。在应用场景里可以改所有设置。');
  if(id==='prec') return tx('In this part you change only the update, in the card or here. You can change everything in Scenarios.','这一部分只改更新量，在卡片里或这里改。在应用场景里可以改所有设置。');
  if(id==='fix') return tx('In this part you pick one fix in the card: a gradient of 1e‑8, an update of 0.0003 on a weight of 1.0. You can change everything in Scenarios.','这一部分只在卡片里选一个修法：梯度是 1e‑8，大小为 1.0 的权重加上 0.0003 的更新。在应用场景里可以改所有设置。');
  return tx('Fixed in this part: each “Show in the chart” button in the card sets its own example. You can change everything in Scenarios.','这一部分固定：卡片里每个“在图里看”按钮都会设好自己的例子。在应用场景里可以改所有设置。');
}
function syncControls(){
  $$('input[data-key="x"]',app).forEach(i=>{if(+i.value!==S.xi) i.value=S.xi; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fin(X()); i.setAttribute('aria-valuetext',fin(X()));});
  $$('input[data-key="u"]',app).forEach(i=>{if(+i.value!==S.ui) i.value=S.ui; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fin(U()); i.setAttribute('aria-valuetext',fin(U()));});
  [['f16',S.f16],['ls',String(S.ls)],['ms',String(S.ms)],['blk',String(S.blk)]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,v)));
  const c=$('#nf-controls',app), ch=mode==='challenge', g=mode==='guided', sb=mode==='sandbox';
  c.classList.toggle('locked',ch);
  const hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  hide('#nf-ctl-x-wrap',!(sb||isP('range'))); hide('#nf-ctl-u-wrap',!(sb||isP('prec')));
  ['#nf-ctl-f16-wrap','#nf-ctl-ls-wrap','#nf-ctl-ms-wrap','#nf-ctl-blk-wrap','#nf-ctl-reset-wrap'].forEach(id=>hide(id,!sb));
  hide('#nf-ctl-assumed',ch);
  const nx=$('#nf-note-x',app), nu=$('#nf-note-u',app); if(nx&&nx.textContent!==noteX()) nx.textContent=noteX(); if(nu&&nu.textContent!==noteU()) nu.textContent=noteU();
  const ln=$('#nf-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
  void g;
}
/* the mini result bar's one sentence: the value, the format that matters and its condition (R3.2) */
function dockRead(v){
  const sv=(r,err)=>r.kind==='lost'?tx('0 (lost)','0（丢了）'):(r.kind==='over'?tx(`${fvc(r.v)} (overflow)`,`${fvc(r.v)}（溢出）`):(err&&relErr(r)>0?tx(`${fvc(r.v)} (${fe(r)} off)`,`${fvc(r.v)}（误差 ${fe(r)}）`):fvc(r.v)));
  if(v==='a'){const x=X(), blk=blkNow();
    if(blk) return tx(`${fin(x)}, block scale: E4M3 ${sv(store(x,'e4m3',blk))}, FP4 ${sv(store(x,'fp4',blk),true)}`,`${fin(x)} 按块缩放：E4M3 ${sv(store(x,'e4m3',blk))}，FP4 ${sv(store(x,'fp4',blk),true)}`);
    return tx(`${fin(x)}: FP16 ${sv(quant(x,'fp16'))}, BF16 ${sv(quant(x,'bf16'))}`,`${fin(x)}：FP16 ${sv(quant(x,'fp16'))}，BF16 ${sv(quant(x,'bf16'))}`);}
  if(v==='b'){const u=U(); return tx(`1.0 + ${fin(u)}: BF16 ${fv(quant(1+u,'bf16').v)}, FP16 ${fv(quant(1+u,'fp16').v)}`,`1.0 + ${fin(u)}：BF16 ${fv(quant(1+u,'bf16').v)}，FP16 ${fv(quant(1+u,'fp16').v)}`);}
  const e=recipe(curRecipe(),tensorsNow());
  return tx(`gradient ${fvc(e.gUsed)}, weight ${fvc(e.w1)}`,`梯度 ${fvc(e.gUsed)}，权重 ${fvc(e.w1)}`);
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(`${$('#nf-st-sub',app).textContent}: ${s.txt}`);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  if(preLock()&&(p.xi!=null)) p={...p,xi:S.xi};
  Object.assign(S,p);
  if(!(S.xi>=0&&S.xi<XS.length)) S.xi=X_W; if(!(S.ui>=0&&S.ui<US.length)) S.ui=U_0;
  if(!['a','b','c'].includes(S.view)) S.view='a'; if(!['fp16','bf16'].includes(S.f16)) S.f16='fp16';
  S.ls=S.ls?1:0; S.ms=S.ms?1:0; if(!(S.blk>=0&&S.blk<BLKS.length)) S.blk=0;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(id){lv=id; lvTried.add(id); render();}
function setRem(k){remView=k; S={...BASE,view:k,xi:X_G,ui:US.indexOf(1e-3)}; if(k==='c') lv='fp16lsm'; renderRail(); if(innerWidth<=760) A.ui.reveal($('#nf-stage',app));}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='predict'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='fix') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the four options in the card.','向下滚到卡片里的四个选项。'))}">${tx('Pick a fix below ↓','在下面选修法 ↓')}</button>`;
  if(id==='range') return xCtl('nf-ph-x',true);
  if(id==='prec') return uCtl('nf-ph-u',true);
  return '';
}
function renderPhone(){const el=$('#nf-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  const st=$('#nf-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg=(innerWidth<=760?{range:'#nf-ph-x',prec:'#nf-ph-u'}:{})[s.id]||{range:'#nf-ic-x',prec:'#nf-ic-u',fix:'#nf-ic-lv'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#nf-rail .step.current',app)||$('#nf-rail',app));}
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
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#nf-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#nf-rail .step.current',app)||$('#nf-rail .railcard',app)||$('#nf-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#nf-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#nf-insight',app)); else revealBox($('#nf-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chR='fp32';}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#nf-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[S.xi,S.ui,S.view,S.f16==='bf16'?'b':'h',S.ls,S.ms,S.blk].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const xi=+p[0], ui=+p[1], view=p[2], f16=p[3]==='b'?'bf16':(p[3]==='h'?'fp16':null), ls=+p[4], ms=+p[5], blk=+p[6];
  if(!Number.isInteger(xi)||xi<0||xi>=XS.length||!Number.isInteger(ui)||ui<0||ui>=US.length||!['a','b','c'].includes(view)||!f16||![0,1].includes(ls)||![0,1].includes(ms)||!Number.isInteger(blk)||blk<0||blk>=BLKS.length) return null;
  return {xi,ui,view,f16,ls,ms,blk};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DF=['bf16','fp16','e4m3'];
function makeDrill(seed){const rnd=mulberry(seed*104729+7), f=DF[Math.floor(rnd()*DF.length)];
  const lo={bf16:-3,fp16:-3,e4m3:-1}[f], d=lo+Math.floor(rnd()*3), m=1+Math.floor(rnd()*89)/10, x=parseFloat(`${m.toFixed(1)}e${d}`);
  return {seed,f,x,r:quant(x,f)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), f=FM[d.f], qd=$('#nf-drill-q',app), a=$('#nf-drill-a',app);
  qd.innerHTML=tx(`Store <b>${fin(d.x)}</b> in <b>${f.n}</b> (${f.e} exponent bits, ${f.m} mantissa bits, round to nearest even). What is stored, and how far off is it?`,
                  `把 <b>${fin(d.x)}</b> 存成 <b>${f.n}</b>（${f.e} 个指数位，${f.m} 个尾数位，就近舍入、平局取偶）。存下的是多少，误差多大？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const r=d.r, ex=Math.max(fl2(Math.abs(d.x)),f.emin), q=2**(ex-f.m), n=d.x/q;
  a.innerHTML=tx(`<p><b>${fv(r.v)}, off by ${fe(r)}.</b></p><p>${fin(d.x)} lies between ${A.ui.math(`2^{${sg(ex)}}`)} and ${A.ui.math(`2^{${sg(ex+1)}}`)}, where ${f.n}’s step is ${A.ui.math(`2^{${sg(ex)} − ${f.m}}`)} = ${fv(q)}. ${fin(d.x)} ÷ ${fv(q)} = ${n.toFixed(3)}, which rounds to ${Math.round(r.v/q)}, so the stored value is ${Math.round(r.v/q)} × ${fv(q)} = ${fv(r.v)}.</p>`,
    `<p><b>${fv(r.v)}，误差 ${fe(r)}。</b></p><p>${fin(d.x)} 在 ${A.ui.math(`2^{${sg(ex)}}`)} 和 ${A.ui.math(`2^{${sg(ex+1)}}`)} 之间，${f.n} 在这里的台阶是 ${A.ui.math(`2^{${sg(ex)} − ${f.m}}`)} = ${fv(q)}。${fin(d.x)} ÷ ${fv(q)} = ${n.toFixed(3)}，舍入成 ${Math.round(r.v/q)}，所以存下 ${Math.round(r.v/q)} × ${fv(q)} = ${fv(r.v)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={x:{en:'Value to store',zh:'要存的数'},u:{en:'Update',zh:'更新量'},ls:{en:'Loss scaling',zh:'loss scaling'},ms:{en:'FP32 master copy',zh:'FP32 主副本'}};
const TERM={x:()=>`<div>${esc(tx('The value being stored; each format rounds it to its nearest representable number.','正在存的这个数；每种格式把它舍入到最近的可表示数。'))}</div>`,
  u:()=>`<div>${esc(TIP.upd())}</div>`, ls:()=>`<div>${esc(TIP.ls())}</div>`, ms:()=>`<div>${esc(TIP.master())}</div>`};
const termTipHTML=k=>TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return k==='ls'?tx('Fixed during the challenge: the loss scale is 65,536 in the recipe that uses it.','挑战中固定不变：用 loss scaling 的那个做法，loss scale 是 65,536。'):(k==='x'||k==='u'?tx(`Fixed during the challenge: weights near 0.02 and gains of 1.0, an update of ${fin(CHT.u)}, gradients from 1e‑8 to 1e‑3.`,`挑战中固定不变：权重约 0.02，gain 是 1.0，更新 ${fin(CHT.u)}，梯度 1e‑8 到 1e‑3。`):'');
  if(mode!=='guided') return '';
  if(k==='x'&&isP('range')) return ''; if(k==='u'&&isP('prec')) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='bits'||k==='range'||k==='zoom') return tx('Click to scroll to it in the chart.','点击滚到图里的这一处。');
  if(k==='fid') return tx('Click to jump to the notes at the end of the page.','点击跳到页末的“简化、假设和来源”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({x:$('#nf-ic-x',app)||$('#nf-ctl-x',app),u:$('#nf-ic-u',app)||$('#nf-ctl-u',app),ls:$('#nf-ctl-ls',app),ms:$('#nf-ctl-ms',app)}[k]);
function jump(k,src){
  if(!app) return;
  const sm=A.reduceMotion()?'auto':'smooth';
  if(k==='math'){const d=$('#nf-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:sm}); A.ui.flash(d); return;}
  if(k==='bits'||k==='range'||k==='zoom'){const el={bits:'#nf-rows',range:'#nf-range',zoom:'#nf-zoom'}[k], e=$(el,app); A.tip.hide(); const t=e&&!e.hidden&&e.getClientRects().length?e:$('#nf-graph',app); t.scrollIntoView({block:'center',behavior:sm}); A.ui.flash(t); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#nf-controls',app)); return;}
  if(k==='fid'){const b=$(`[data-pop="${ID}-fid"]`,app), p=document.getElementById(`${ID}-fid`); A.tip.hide(); if(p&&p.hidden&&b) b.click(); const el=p||b; if(el){el.scrollIntoView({block:'center',behavior:sm}); A.ui.flash(el);} return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...BASE,view:S.view}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chR).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#nf-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#nf-ic-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#nf-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#nf-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    const sb=mode==='sandbox';
    if(k==='x') bySettings(e.target,()=>setState(sb?{xi:v,view:'a'}:{xi:v}));
    else if(k==='u') bySettings(e.target,()=>setState(sb&&S.view!=='c'?{ui:v,view:'b'}:{ui:v}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#nf-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#nf-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#nf-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('fix')){const f=()=>byLearner(()=>setLv(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.lever!=null&&mode==='challenge'&&RC[d.lever]){byLearner(()=>{chR=d.lever; render();}); return;}
    if(d.view!=null&&b.closest('#nf-view')){byLearner(()=>setState({view:d.view})); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='f16') bySettings(b,()=>setState({f16:d.v,view:'c'}));
      else if(k==='ls') bySettings(b,()=>setState({ls:+d.v,view:'c'}));
      else if(k==='ms') bySettings(b,()=>setState({ms:+d.v,view:'c'}));
      else if(k==='blk') bySettings(b,()=>setState({blk:+d.v,view:'a'}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#nf-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#nf-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-13, under the rule of 2026-10-12 07:13 UTC): four text-and-formula sections read
   before the guided parts: why fewer bits (memory, bytes moved, 16-bit matmul speed); the floating-point formula with e exponent bits
   and m mantissa bits, FP32 worked through and −0.8125 encoded bit by bit; the largest number, the smallest normal number and the gap
   2^(k−m) (range and precision); rounding to nearest even, overflow, underflow and a small number added to a large one. The letters
   follow the lab's own “Show the math” (E the exponent field, M the mantissa, m the mantissa bits; the bias is written out, since b is
   the PPO primer's baseline). No FP16 or BF16 bound or gap is computed here: parts 3 and 4 let the reader find them; loss scaling and
   the master copy stay with part 5. The H100 speed-up is given only as a ratio in words (about 15: the datasheet's BF16 tensor-core peak over
   its FP32 peak), since tests/number-formats_numbers.py keeps the lab free of a throughput figure (review A11) (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=labLink(id); return tx(a,`“${a}”`);};
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return h.startsWith('<div')?h:`<p>${h}</p>`;}).join('')});
/* formulas on their own line: separate formulas side by side, wrapping under each other in a narrow column, so nothing scrolls sideways */
const prFx=(...xs)=>()=>`<div class="nf-prfx">${xs.map(x=>`<span>${prM(typeof x==='function'?x():x)}</span>`).join('')}</div>`;
/* −0.8125 in FP32, drawn in the lab's own bit cells (sign, exponent, mantissa), each field named under it */
const prBits=()=>{const fld=(bits,c,lab)=>`<span class="nf-prb"><span class="nf-fld">${[...bits].map(b=>`<i class="nf-bit ${c}">${b}</i>`).join('')}</span><em>${lab}</em></span>`;
  return `<div class="nf-prbits" role="img" aria-label="${esc(tx('−0.8125 in FP32: sign bit 1, exponent bits 01111110, mantissa bits 101 followed by 20 zeros','−0.8125 存成 FP32：符号位 1，指数位 01111110，尾数位 101 后面跟 20 个 0'))}">`+
    fld('1','nf-sgn',tx('sign ','符号位 ')+prM('s = 1'))+fld('01111110','nf-exp',tx('exponent ','指数位 ')+prM('E = 126'))+
    fld('10100000000000000000000','nf-man',tx('mantissa ','尾数位 ')+prM('M / 2^{23} = 0.625'))+`</div>`;};
if(A.registerPrimer) A.registerPrimer(ID,{min:8,sections:[
 {h:()=>tx('Number formats: fewer bits per number, less memory, fewer bytes to move, faster arithmetic','数值格式：每个数少用几位，省显存、省带宽、算得更快'),parts:[prP(
   ()=>tx(`A large model has billions to hundreds of billions of parameters, and in training each parameter also has a gradient; the intermediate results each layer computes and hands to the next, called activations, are kept in training until the backward pass has used them (see the basics of ${prLK('backprop')}, where they are the saved tensors). All these numbers sit in the GPU’s memory, and how many bits each takes is set by its number format. FP32, PyTorch’s default floating-point format, takes 32 bits, or 4 bytes, per number: a billion parameters take 4 GB for a single copy.`,
          `大模型有几十亿到几千亿个参数，训练时每个参数还有一个梯度；每一层算出来、交给下一层的中间结果，叫激活值，训练时要留到反向传播用完（见${prLK('backprop')}的知识梳理，那里叫 saved tensors）。这些数都放在 GPU 的显存里，每个数占几位，由它的数值格式决定。PyTorch 默认的浮点格式 FP32，每个数占 32 位，也就是 4 字节：10 亿个参数，只存一份就要 4 GB。`),
   ()=>tx(`Cutting each number from 32 bits to 16 has three benefits. First, the same numbers take half the memory. Second, at every step the GPU reads the numbers it needs from memory into its compute units and writes the results back, and the number of bytes it can move per second (the memory bandwidth) has a limit; with half the bytes per number, moving the same numbers takes half the time (how memory bandwidth limits a GPU’s speed: see ${prLK('roofline')}). Third, the GPU’s units built for matrix multiplication are much faster on 16-bit numbers: the H100, NVIDIA’s data-center GPU, computes in BF16, a 16-bit floating-point format, on these units at a peak about 15 times what it reaches in FP32 on its ordinary units. So large models today are trained with their matrix multiplications mostly in 16-bit formats, with 16-bit and 32-bit formats mixed within one run, which is called mixed-precision training (see ${prLK('training-memory')}); when a model is deployed its weights are often cut further, to 8 or 4 bits.`,
          `每个数从 32 位减到 16 位，有三个好处。第一，同样多的数只占一半显存。第二，GPU 每算一步，都要把用到的数从显存读进计算单元，再把结果写回显存，而每秒能搬运的字节数（显存带宽）有上限；每个数的字节减半，同样多的数，搬运的时间也减半（显存带宽怎样限制 GPU 的速度，见${prLK('roofline')}）。第三，GPU 里专做矩阵乘法的计算单元，算 16 位的数快得多：NVIDIA 的数据中心 GPU H100 用这些单元算 BF16 这种 16 位浮点格式，峰值约是它用普通计算单元算 FP32 的 15 倍。所以如今训练大模型，矩阵乘法大多用 16 位的格式算，一次训练里 16 位和 32 位的格式混着用，这叫混合精度训练（见${prLK('training-memory')}）；部署模型时，权重还常常压到 8 位、4 位。`),
   ()=>tx(`But fewer bits can tell fewer numbers apart: 32 bits have ${prM('2^{32}')} patterns and 16 bits only ${prM('2^{16}')}, and each pattern stands for at most one number. So any format can store only finitely many numbers: any other number has to be stored as the storable number nearest to it, and a number too large or too close to 0 cannot be stored at all. Which numbers are lost when the bits are halved? That depends on how a format spends its bits.`,
          `可是位数少了，能区分的数也就少了：32 位一共有 ${prM('2^{32}')} 种组合，16 位只有 ${prM('2^{16}')} 种，每种组合至多表示一个数。所以一种格式能存的数只有有限多个：别的数只能存成离它最近的一个能存的数，太大或太接近 0 的数根本存不下。位数减半以后，丢掉的是哪些数？这取决于格式怎样分配这些位。`))]},
 {h:()=>tx('The three parts of a floating-point number: sign, exponent and mantissa bits (FP32 as the example)','浮点数的三部分：符号位、指数位、尾数位（以 FP32 为例）'),parts:[prP(
   ()=>tx('The numbers in a model differ hugely in size: some are large, some very close to 0, and gradients are often several orders of magnitude smaller than weights. If the storable numbers were laid out like integers, at equal gaps across a fixed interval (fixed-point numbers), the gap would have to be tiny for small numbers to stay apart, and then covering large numbers too would take far too many bits. Floating-point numbers do what scientific notation does: a number is written as “a few significant digits × the base to some power”, only with base 2. The power carries the size and the number of significant digits is fixed, so large and small numbers get about the same relative precision.',
          '模型里的数大小相差很远：有的很大，有的非常接近 0，梯度常常比权重小好几个数量级。要是像整数那样，在一段固定的区间里按相同的间隔排开能存的数（这叫定点数），要让很小的数也分得清，间隔就得很小，再要覆盖很大的数，位数就远远不够。浮点数用的是科学计数法的办法：一个数写成“几位有效数字 × 底数的几次方”，只是底数取 2。数的大小交给指数，有效数字的位数固定，所以大数和小数的相对精度差不多。'),
   ()=>tx(`A floating-point format has 1 sign bit, ${prM('e')} exponent bits and ${prM('m')} mantissa bits. The sign bit ${prM('s')} sets plus or minus. The exponent bits are read as an unsigned integer ${prM('E')}, and the mantissa bits as an unsigned integer ${prM('M')}, from 0 to ${prM('2^{m} - 1')}; ${prM('E')} minus a fixed offset, the bias, is the true power of 2. Together they stand for the number`,
          `一种浮点格式有 1 个符号位、${prM('e')} 个指数位和 ${prM('m')} 个尾数位。符号位 ${prM('s')} 决定正负。指数位读成一个无符号整数 ${prM('E')}，尾数位读成一个无符号整数 ${prM('M')}（0 到 ${prM('2^{m} - 1')}）；${prM('E')} 减去一个固定的偏移（bias），才是真正的指数。三者表示的数是`),
   prFx(()=>tx('\\text{value} = (-1)^{s} \\times \\left(1 + \\frac{M}{2^{m}}\\right) \\times 2^{E - \\text{bias}}','\\text{数值} = (-1)^{s} \\times \\left(1 + \\frac{M}{2^{m}}\\right) \\times 2^{E - \\text{bias}}'),'\\text{bias} = 2^{e-1} - 1'),
   ()=>tx(`Written in binary, ${prM('1 + M / 2^{m}')} is “1.” followed by the ${prM('m')} mantissa bits. Why a leading 1 that is not stored? Any nonzero binary number can, as in scientific notation, have its point moved to just after its first 1, becoming “1.something × 2 to some power”; the leading digit is then always 1 and need not be stored, so ${prM('m')} mantissa bits give ${prM('m + 1')} significant binary digits. The power must be able to go negative, to reach numbers below 1, but ${prM('E')} is unsigned; that is why the bias is subtracted: whenever ${prM('E')} is below the bias, the power is negative. The bias is about half the range of ${prM('E')}, ${prM('2^{e-1} - 1')}, so that there are about as many negative powers as positive ones, and numbers below 1 and above 1 span about as many orders of magnitude. Two codes of ${prM('E')} are kept for other uses; the numbers the formula gives with any other ${prM('E')} are called normal numbers. All 0s is kept for 0 and the subnormal numbers (some numbers that are not 0 but closer to 0 than every normal number), all 1s for infinity (Inf) and NaN (not a number, the result of 0 divided by 0, for example).`,
          `${prM('1 + M / 2^{m}')} 写成二进制，就是“1.”后面依次跟着 ${prM('m')} 个尾数位。为什么开头有一个不存的 1？任何非零的二进制数，都能像科学计数法那样把小数点移到第一个 1 后面，写成“1.几位 × 2 的几次方”；最高位总是 1，不必存，所以 ${prM('m')} 个尾数位实际给出 ${prM('m + 1')} 个有效二进制位。指数要能取负数，才能表示比 1 小的数，可 ${prM('E')} 是无符号整数，所以要减去偏移：${prM('E')} 比偏移小，指数就是负的。偏移取 ${prM('E')} 的取值范围的一半左右，也就是 ${prM('2^{e-1} - 1')}，这样负的指数和正的指数差不多一样多，比 1 小的数和比 1 大的数，能覆盖的数量级也差不多。${prM('E')} 全是 0 和全是 1 的两个编码另有用途，${prM('E')} 取其余的值时，公式表示的数叫正规数。全 0 留给 0 和次正规数（比所有正规数都更接近 0、又不是 0 的一些数），全 1 留给无穷大 Inf 和 NaN（not a number，比如 0 除以 0 的结果）。`),
   ()=>tx(`FP32 has 1 sign bit, 8 exponent bits and 23 mantissa bits, and its bias is ${prM('2^{7} - 1 = 127')}. Take −0.8125. It is negative, so ${prM('s = 1')}. In binary it is ${prM('-0.1101_{2} = -1.101_{2} \\times 2^{-1}')}, so ${prM('E - 127 = -1')}, ${prM('E = 126')}, which is 01111110 in 8 bits. The 101 after “1.” fills the first three mantissa bits, and the other 20 are 0:`,
          `FP32 有 1 个符号位、8 个指数位、23 个尾数位，偏移是 ${prM('2^{7} - 1 = 127')}。比如把 −0.8125 存成 FP32：它是负数，${prM('s = 1')}；写成二进制是 ${prM('-0.1101_{2} = -1.101_{2} \\times 2^{-1}')}，所以 ${prM('E - 127 = -1')}，${prM('E = 126')}，写成 8 位是 01111110；“1.”后面的 101 依次填进尾数位的前三位，其余 20 位都是 0：`),
   prBits,
   ()=>tx(`Back into the formula: ${prM('M / 2^{23} = 0.101_{2} = 0.625')}, and ${prM('(-1)^{1} \\times (1 + 0.625) \\times 2^{126 - 127} = -0.8125')}.`,
          `代回公式：${prM('M / 2^{23} = 0.101_{2} = 0.625')}，${prM('(-1)^{1} \\times (1 + 0.625) \\times 2^{126 - 127} = -0.8125')}。`))]},
 {h:()=>tx('The largest number, the smallest normal number and the gap between neighbours (range and precision)','最大的数、最小的正规数和相邻两数的间隔（范围与精度）'),parts:[prP(
   ()=>tx(`With this formula one can work out which numbers a format can store. Two things matter: how small and how large the storable numbers go, the range; and how far apart two neighbouring storable numbers are, the gap, where a smaller gap means higher precision. The range depends mainly on the exponent bits. A normal number’s ${prM('E')} is at most ${prM('2^{e} - 2')} (all 1s, ${prM('2^{e} - 1')}, is kept for Inf and NaN), and then the power ${prM('E - \\text{bias} = 2^{e} - 2 - (2^{e-1} - 1) = 2^{e-1} - 1')} equals the bias exactly; with every mantissa bit 1 as well, ${prM('1 + M / 2^{m} = 2 - 2^{-m}')}, and that is the largest number. ${prM('E')} is at least 1 (all 0s is kept for 0 and the subnormal numbers), and with every mantissa bit 0 that gives the smallest normal number:`,
          `有了这个公式，就能算出一种格式能存哪些数。要看两件事：能存的数从多小到多大，叫范围；相邻两个能存的数相隔多远，叫间隔，间隔越小，精度越高。范围主要由指数位决定。正规数的 ${prM('E')} 最大是 ${prM('2^{e} - 2')}（全 1 的 ${prM('2^{e} - 1')} 留给了 Inf 和 NaN），这时指数 ${prM('E - \\text{bias} = 2^{e} - 2 - (2^{e-1} - 1) = 2^{e-1} - 1')}，正好等于偏移；尾数位再全取 1，${prM('1 + M / 2^{m} = 2 - 2^{-m}')}，就是最大的数。${prM('E')} 最小是 1（全 0 留给了 0 和次正规数），尾数位全取 0，就是最小的正规数：`),
   prFx(()=>tx('\\text{largest finite number} = (2 - 2^{-m}) \\times 2^{\\text{bias}}','\\text{最大的有限数} = (2 - 2^{-m}) \\times 2^{\\text{bias}}'),()=>tx('\\text{smallest normal number} = 2^{1 - \\text{bias}}','\\text{最小的正规数} = 2^{1 - \\text{bias}}')),
   ()=>tx(`The factor ${prM('2 - 2^{-m}')} lies between 1 and 2, so the size of both bounds is set by the bias, and the bias ${prM('2^{e-1} - 1')} depends only on the number of exponent bits ${prM('e')}: each extra exponent bit roughly doubles the bias, and so roughly doubles the number of orders of magnitude from the smallest normal number to the largest number. FP32’s bias is 127: its largest finite number is about ${prM('3.4 \\times 10^{38}')}, and its smallest normal number is ${prM('2^{-126}')}, about ${prM('1.2 \\times 10^{-38}')}.`,
          `因子 ${prM('2 - 2^{-m}')} 在 1 和 2 之间，所以两个界的量级都由偏移决定，偏移 ${prM('2^{e-1} - 1')} 又只取决于指数位的个数 ${prM('e')}：指数位每多 1 个，偏移大约翻一倍，从最小的正规数到最大的数，跨过的数量级也大约翻一倍。FP32 的偏移是 127：最大的有限数约为 ${prM('3.4 \\times 10^{38}')}，最小的正规数是 ${prM('2^{-126}')}，约为 ${prM('1.2 \\times 10^{-38}')}。`),
   ()=>tx(`The gap depends on the mantissa bits. Fix ${prM('E')} and write the power as ${prM('k = E - \\text{bias}')}: ${prM('M')} runs from 0 to ${prM('2^{m} - 1')}, and each step of 1 in ${prM('M')} adds ${prM('2^{k} / 2^{m}')} to the number, giving ${prM('2^{m}')} equally spaced numbers from ${prM('2^{k}')} up to just below ${prM('2^{k+1}')}. So between ${prM('2^{k}')} and ${prM('2^{k+1}')}, two neighbouring storable numbers differ by`,
          `间隔由尾数位决定。固定 ${prM('E')}，记指数 ${prM('k = E - \\text{bias}')}：${prM('M')} 从 0 取到 ${prM('2^{m} - 1')}，每加 1，数就加 ${prM('2^{k} / 2^{m}')}，给出从 ${prM('2^{k}')} 起、到 ${prM('2^{k+1}')} 之前等距排开的 ${prM('2^{m}')} 个数。所以在 ${prM('2^{k}')} 和 ${prM('2^{k+1}')} 之间，相邻两个能存的数相差`),
   prFx(()=>tx('\\text{gap} = 2^{k} \\times 2^{-m} = 2^{k-m}','\\text{间隔} = 2^{k} \\times 2^{-m} = 2^{k-m}')),
   ()=>tx(`Each time a power of 2 is passed, the gap doubles: the larger the number, the wider the gap. But the gap divided by the numbers in that stretch always lies between ${prM('2^{-(m+1)}')} and ${prM('2^{-m}')}, however large or small they are. So the mantissa bits set the relative precision. FP32’s gap between 1 and 2 is ${prM('2^{-23}')}, about ${prM('1.2 \\times 10^{-7}')}, roughly 7 significant decimal digits.`,
          `每跨过一个 2 的幂，间隔就翻一倍：数越大，间隔越宽。但不管数是大是小，间隔除以这一段里的数，总在 ${prM('2^{-(m+1)}')} 和 ${prM('2^{-m}')} 之间。所以尾数位定的是相对精度。FP32 在 1 和 2 之间的间隔是 ${prM('2^{-23}')}，约为 ${prM('1.2 \\times 10^{-7}')}，大致相当于 7 位十进制有效数字。`),
   ()=>tx('With the total number of bits fixed, every bit given to the exponent is taken from the mantissa: the range grows and the gaps get coarser; give fewer to the exponent, and the gaps get finer while the range shrinks. There is more than one 16-bit floating-point format, FP16 and BF16 for example, and each strikes its own balance between range and precision.',
          '总位数一定时，多给指数位，就得少给尾数位：范围变大，间隔变粗；少给指数位，间隔变细，范围变小。16 位的浮点格式不止一种，比如 FP16 和 BF16，各自在范围和精度之间做了不同的取舍。'))]},
 {h:()=>tx('Every result is rounded (round to nearest, overflow and underflow)','每次运算的结果都要舍入（就近舍入、溢出和下溢）'),parts:[prP(
   ()=>tx(`An operation on two storable numbers usually has an exact result that is not storable: the exact product of two FP32 numbers, with 24 significant binary digits each, can have 48. The rule of floating-point arithmetic is that the result is the exact value, rounded to the nearest number the format can store; write the rounded result of an exact value ${prM('z')} as ${prM('\\mathrm{fl}(z)')}. When the exact value lies exactly halfway between two storable numbers, it goes to the one whose lowest mantissa bit is 0 (ties to even), so that ties round up and down about equally often and the errors do not keep piling up in one direction. This is the default rounding of IEEE 754, the common standard for floating-point numbers.`,
          `两个能存的数做一次运算，精确的结果往往不是能存的数：两个 FP32 数（各有 24 个有效二进制位）相乘，精确的积最多有 48 位。浮点运算的规则是：结果等于先求出精确值，再舍入到格式能存的最近的数；精确值 ${prM('z')} 舍入后的结果记作 ${prM('\\mathrm{fl}(z)')}。精确值恰好落在两个能存的数正中间时，取尾数最低位是 0 的那个（平局取偶），这样平局时向上、向下舍入的机会差不多，误差不会总往一个方向攒。这是浮点数的通用标准 IEEE 754 规定的默认舍入方式。`),
   ()=>tx(`When the exact value lies within the range of the normal numbers, the rounding error is at most half a gap, and a gap is at most ${prM('2^{-m}')} times the number itself, so`,
          `精确值在正规数的范围里时，舍入的误差最多是半个间隔，而间隔不超过数本身的 ${prM('2^{-m}')} 倍，所以`),
   prFx('\\mathrm{fl}(z) = z \\, (1 + \\delta), \\quad |\\delta| \\le 2^{-(m+1)}'),
   ()=>tx('Every single operation can add a relative error this large, and the fewer the mantissa bits, the larger it is.',
          '每做一次运算，结果都可能带上这么大的相对误差；尾数位越少，这个误差越大。'),
   ()=>tx(`An exact value outside the range follows other rules. One so large that it still exceeds the largest finite number after rounding becomes Inf: an overflow. One so close to 0 that it is not even half the smallest positive number the format can store (the subnormal numbers can be stored too, so this smallest positive number is smaller than the smallest normal number) has 0 as its nearest storable number and becomes 0: an underflow. Inf in a further operation, such as Inf − Inf, gives NaN, and any arithmetic operation involving NaN gives NaN. So one overflow can be carried all the way down and turn the loss into NaN too (see ${prLK('softmax-numerics')}).`,
          `精确值超出范围时另有规则。大到舍入以后仍超过最大的有限数，结果就是 Inf，这叫溢出（overflow）；太接近 0，连格式能存的最小正数的一半都不到（次正规数也存得下，所以这个最小正数比最小的正规数还小），离它最近的能存的数就是 0，结果存成 0，这叫下溢（underflow）。Inf 再参加运算，比如 Inf − Inf，得到 NaN；NaN 参加的算术运算，结果都是 NaN。所以一处溢出可以一路传下去，让 loss 也变成 NaN（见${prLK('softmax-numerics')}）。`),
   ()=>tx(`One kind of loss stays inside the range: adding a small number to a large one. Add ${prM('\\Delta')} to a normal number ${prM('a')}: ${prM('a + \\Delta')} is rounded to the nearest storable number, and if ${prM('|\\Delta|')} is less than half the gap next to ${prM('a')} on ${prM('\\Delta')}’s side, ${prM('a + \\Delta')} is closer to ${prM('a')} than to any other storable number, and the result is ${prM('a')} itself. As the previous section showed, the gap divided by the numbers in its stretch always lies between ${prM('2^{-(m+1)}')} and ${prM('2^{-m}')} (when ${prM('a')} is exactly a power of 2, the gap on its lower side is the smallest, ${prM('2^{-(m+1)} \\, |a|')}). So whatever ${prM('a')} is:`,
          `还有一种丢失，结果并没有超出范围：大数加小数。往一个正规数 ${prM('a')} 上加一个 ${prM('\\Delta')}，${prM('a + \\Delta')} 要舍入到最近的能存的数；${prM('|\\Delta|')} 不到 ${prM('a')} 在 ${prM('\\Delta')} 那一侧的间隔的一半，${prM('a + \\Delta')} 离 ${prM('a')} 就比离别的能存的数都近，结果就是 ${prM('a')} 本身。上一节说过，间隔除以这一段里的数，总在 ${prM('2^{-(m+1)}')} 和 ${prM('2^{-m}')} 之间（${prM('a')} 正好是 2 的幂时，它下面一侧的间隔最小，是 ${prM('2^{-(m+1)} \\, |a|')}）。所以不管 ${prM('a')} 是多少：`),
   prFx('|\\Delta| < 2^{-(m+2)} \\, |a| \\;⇒\\; \\mathrm{fl}(a + \\Delta) = a','|\\Delta| > 2^{-(m+1)} \\, |a| \\;⇒\\; \\mathrm{fl}(a + \\Delta) \\neq a'),
   ()=>tx(`A ${prM('\\Delta')} between the two bounds may or may not be lost, depending on where ${prM('a')} sits in its stretch and on the sign of ${prM('\\Delta')}.`,
          `落在两者之间的 ${prM('\\Delta')} 丢不丢，要看 ${prM('a')} 在它那一段里的位置和 ${prM('\\Delta')} 的正负。`),
   ()=>tx(`In the first case ${prM('\\Delta')} is not 0, yet ${prM('a')} has not changed at all. Training is full of such additions: every step adds a small update to each weight, and many small contributions are added into the same gradient. Whether these small amounts survive depends on how many times smaller they are than what they are added to, and on how many mantissa bits the format has.`,
          `在第一种情况下，${prM('\\Delta')} 不是 0，加完 ${prM('a')} 却一点没变。训练里这样的加法到处都是：每一步都把一个小的更新加到权重上，许多份小的贡献加进同一个梯度里。这些小量能不能留下，取决于它比被加的数小多少倍，以及格式有几个尾数位。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S as it was (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chR,chSubmitted,chReveal,sbTouched,seenX:[...seenX],seenU:[...seenU],
    lv,lvTried:[...lvTried],remView,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(chR).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse([x.xi,x.ui,x.view,x.f16==='bf16'?'b':(x.f16==='fp16'?'h':'x'),x.ls,x.ms,x.blk].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chR=mode==='challenge'&&CHO.includes(o.chR)?o.chR:'fp32';
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    const ints=(a,n)=>Array.isArray(a)?a.filter(x=>Number.isInteger(x)&&x>=0&&x<n):[];
    seenX=new Set(ints(o.seenX,XS.length)); seenU=new Set(ints(o.seenU,US.length));
    lv=LV5.includes(o.lv)?o.lv:'fp16'; lvTried=new Set(['fp16']); if(Array.isArray(o.lvTried)) o.lvTried.forEach(x=>{if(LV5.includes(x)) lvTried.add(x);}); lvTried.add(lv);
    remView=['a','b','c'].includes(o.remView)?o.remView:'a';
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:2; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#nf-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#nf-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    fromSnap=false;
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#nf-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#nf-math',app).open=true;}},
  onResize(){if(app) drawStage();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='a'?$('#nf-rows',app):(v==='b'?$('#nf-zoom',app):$('#nf-paths',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#nf-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.txt; vis.innerHTML='';
      read.innerHTML=esc(dockRead(v));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o,(k,v)=>typeof v==='number'&&!isFinite(v)?String(v):v));
window.__nf=Object.freeze({
  quant:(x,f,sc)=>deep(quant(x,f,sc)), store:(x,f,blk)=>deep(store(x,f,blk)), mx:(x,f,blk)=>deep(mxStore(x,f,blk)), recipe:(id,T)=>deep(recipe(RC[id],T||CHT)), series:(id,T,n)=>deep(series(RC[id],T,n)),
  chEval:id=>deep(chEval(id)), chBest, drill:s=>deep(makeDrill(s)), fmt:Object.freeze({fv,fin,fpc}), exits:()=>deep(EXITS()), code:CODE,
  consts:Object.freeze({FM:deep(FM),XS:XS.slice(),US:US.slice(),X_W,X_G,U_0,LSV,BLKS:BLKS.slice(),CHT:{...CHT},LR_PEAK,CHO:CHO.slice(),LV5:LV5.slice(),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),BASE:{...BASE}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},lv,lvTried:[...lvTried],remView,chR,chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
