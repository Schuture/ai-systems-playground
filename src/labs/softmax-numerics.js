/* softmax-numerics (the roots, ML basics, world roots): Why does a naive softmax return NaN?
   Fidelity: Toy run. The page really computes in FP32, BF16 and FP16: every operation's double result is rounded to the format
   (FP32 with Math.fround; FP16 and BF16 with an exact round-to-nearest-even, overflow to Inf, subnormals kept).
   - Naive softmax: e_i = exp(z_i), s = e_1 + … + e_n (left to right, rounded after each addition), p_i = e_i ÷ s.
   - Stable softmax: m = max z, the same with z_i − m. Exact reference: the float64 softmax of the same inputs.
   - Cross-entropy: minus the log of the naive probability, minus the log of the max-subtracted probability, and log-sum-exp
     (the max, plus the log of the sum, minus the right token's score).
   - Online softmax: blocks of 2; running max m and sum s; when a block raises the max, s × exp(m_old − m_new) first.
   Words (EN / ZH): score; softmax; overflow / 溢出; underflow / 下溢; safe window / 安全区间; NaN; Inf; part / 部分.
   window.__sm is a read-only hook for tests/softmax-numerics_numbers.py and tests/softmax-numerics_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='softmax-numerics';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine: number formats
   ===================================================================== */
/* mant: stored mantissa bits; emin: exponent of the smallest normal number; emax: of the largest */
const FMT={
 fp32:{id:'fp32',name:'FP32',bytes:4,mant:23,emin:-126,emax:127},
 bf16:{id:'bf16',name:'BF16',bytes:2,mant:7,emin:-126,emax:127},
 fp16:{id:'fp16',name:'FP16',bytes:2,mant:10,emin:-14,emax:15}};
Object.values(FMT).forEach(F=>{
  F.max=(2-2**-F.mant)*2**F.emax;            // the largest finite number
  F.ovf=(2-2**-(F.mant+1))*2**F.emax;        // a result at or above this rounds to Inf
  F.sub=2**(F.emin-F.mant);                   // the smallest subnormal number
  F.hi=Math.log(F.ovf);                       // exp(z) is Inf above this z
  F.lo=(F.emin-F.mant-1)*Math.LN2;            // exp(z) rounds to 0 below this z (half the smallest subnormal)
});
const FIDS=['fp32','bf16','fp16'];
/* round a double to the format: round to nearest, ties to even; overflow to ±Inf; subnormals kept */
function rnd(f,x){
  if(f==='fp32') return Math.fround(x);
  const F=FMT[f]; if(!isFinite(x)||x===0) return x;
  const a=Math.abs(x); if(a>=F.ovf) return x>0?Infinity:-Infinity;
  let e=Math.floor(Math.log2(a)); if(2**e>a) e--; else if(2**(e+1)<=a) e++;
  e=Math.max(e,F.emin);
  const q=2**(e-F.mant), r=a/q, fl=Math.floor(r), d=r-fl;
  const n=d>0.5?fl+1:(d<0.5?fl:(fl%2===0?fl:fl+1)), v=n*q;
  return x<0?-v:v;
}
const EXP=(f,x)=>rnd(f,Math.exp(x));
function sumOf(f,e){let s=0; for(const v of e) s=rnd(f,s+v); return s;}
function naive(z,f){const zz=z.map(v=>rnd(f,v)), e=zz.map(v=>EXP(f,v)), s=sumOf(f,e), p=e.map(v=>rnd(f,v/s)); return {zz,e,s,p};}
function stable(z,f){const zz=z.map(v=>rnd(f,v)), m=Math.max(...zz), d=zz.map(v=>rnd(f,v-m)), e=d.map(v=>EXP(f,v)), s=sumOf(f,e), p=e.map(v=>rnd(f,v/s)); return {zz,m,d,e,s,p};}
function exactP(z){const m=Math.max(...z); if(!isFinite(m)) return z.map(()=>NaN); const e=z.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s);}
const hasNaN=p=>p.some(v=>Number.isNaN(v));
/* the largest error against the exact chances, in percentage points (NaN when the result has a NaN) */
function errPP(p,ref){if(hasNaN(p)||hasNaN(ref)) return NaN; return Math.max(...p.map((v,i)=>Math.abs(v-ref[i])))*100;}
/* cross-entropy for the right token y: three ways */
function ceAll(z,y,f){
  const n=naive(z,f), st=stable(z,f), nl=v=>rnd(f,-Math.log(v));
  const lse=rnd(f,st.m+rnd(f,Math.log(st.s))), ll=rnd(f,lse-st.zz[y]);
  const m=Math.max(...z), ex=m+Math.log(z.reduce((a,v)=>a+Math.exp(v-m),0))-z[y];
  return {naive:nl(n.p[y]),soft:nl(st.p[y]),lse:ll,lseRaw:lse,exact:ex,py:n.p[y],pyS:st.p[y],m:st.m,s:st.s,n,st};
}
/* online softmax: one block at a time; each step keeps the running max m and the running sum s */
function online(z,B,f){
  const steps=[]; let m=-Infinity, s=0;
  for(let k=0;k<z.length;k+=B){
    const blk=z.slice(k,k+B).map(v=>rnd(f,v)), mn=Math.max(m,...blk), first=m===-Infinity;
    const sc=first?0:EXP(f,rnd(f,m-mn)), s0=s, s1=first?0:rnd(f,s*sc), terms=blk.map(v=>EXP(f,rnd(f,v-mn)));
    let s2=s1; for(const t of terms) s2=rnd(f,s2+t);
    steps.push({k:k/B,blk,mOld:m,mNew:mn,sc,s0,s1,s2,terms,raised:mn>m&&!first,first});
    m=mn; s=s2;
  }
  const p=z.map(v=>rnd(f,EXP(f,rnd(f,rnd(f,v)-m))/s)), two=stable(z,f).p;
  return {steps,m,s,p,two,diff:Math.max(...p.map((v,i)=>Math.abs(v-two[i])))};
}

/* =====================================================================
   data
   ===================================================================== */
/* the mission: a model sure of the next token; the slider moves every logit together (the largest one sits at TOP) */
const LM={toks:['Paris','the','a','located','home','not'],z:[92,89,87,86,84,82],prompt:'The capital of France is'};
/* part 4 and a scenario: one query's attention scores for six earlier tokens, kept in FP16 */
const AT={toks:['The','cat','sat','on','the','mat'],z:[8,13.5,9,11,9.75,12.25]};
/* part 5 and a scenario: a training step; the right next token, “not”, sits GAP below the top logit */
const CE={toks:['Paris','the','a','located','home','not'],top:[8,5.5,3,1,-0.5],y:5};
/* part 6: eight scores arriving in four blocks of two */
const ON={toks:['The','old','man','saw','the','dog','run','home'],z:[2,4.5,3,7.5,6,1.5,9,5.5],B:2};
/* a scenario: a padding row of attention scores, every key masked (the part 4 tokens) */
const MV={e9:-1e9,inf:-Infinity};
/* the challenge: scores the lab never showed, kept in FP16 */
const CHS={toks:['Once','upon','a','time','there','was'],z:[9.25,13,8,13.5,10.5,12.25]};
const CH={goal:0.1};
const CHO=[
 {id:'fp32',f:'fp32',meth:'naive',bytes:4,store:'fp32',ch:1,t:{en:'Store and compute in FP32',zh:'改用 FP32 存储和计算'},d:{en:'Keep an FP32 copy of the scores, then the same exp, sum and divide',zh:'把得分另存一份 FP32，再照样 exp、求和、相除'}},
 {id:'max',f:'fp16',meth:'stable',bytes:2,store:'fp16',ch:1,t:{en:'Subtract the max, stay in FP16',zh:'减去最大值，仍用 FP16'},d:{en:'Every score minus the largest one, then exp, sum and divide in FP16',zh:'每个得分先减去最大的那个，再在 FP16 里 exp、求和、相除'}},
 {id:'bf16',f:'bf16',meth:'naive',bytes:2,store:'bf16',ch:1,t:{en:'Store the scores in BF16',zh:'改存成 BF16'},d:{en:'The same 2 bytes, with about FP32’s largest number',zh:'同样 2 字节，能存的最大数和 FP32 差不多'}},
 {id:'half',f:'fp16',meth:'naive',bytes:2,store:'fp16',ch:1,xf:v=>v/2,t:{en:'Divide the scores by 2',zh:'把得分除以 2'},d:{en:'Halve every score first, so the largest is 6.75',zh:'先把每个得分减半，最大的变成 6.75'}},
 {id:'clamp',f:'fp16',meth:'naive',bytes:2,store:'fp16',ch:1,xf:v=>Math.min(v,11),t:{en:'Clamp the scores at 11',zh:'把得分截到 11'},d:{en:'Every score above 11 becomes 11, just inside FP16’s limit',zh:'超过 11 的得分都变成 11，刚好在 FP16 的上限以内'}},
 {id:'both',f:'fp32',meth:'stable',bytes:2,store:'fp16',ch:2,t:{en:'Subtract the max, with FP32 arithmetic',zh:'减去最大值，并用 FP32 计算'},d:{en:'The scores stay FP16 in memory; the max, each exp and the sum are kept in FP32, as real GPU code does',zh:'得分在内存里仍是 FP16；最大值、每个 exp 和总和都放在 FP32 里，实际的 GPU 代码就是这样做的'}}];
const CH0={id:'start',f:'fp16',meth:'naive',bytes:2,store:'fp16',ch:0,t:{en:'Naive softmax in FP16',zh:'朴素 softmax，FP16'}};
function chEval(o){const z=CHS.z.map(o.xf||(v=>v)), r=o.meth==='stable'?stable(z,o.f):naive(z,o.f), ex=exactP(CHS.z), err=errPP(r.p,ex);
  return {o,p:r.p,s:r.s,err,ok:!Number.isNaN(err)&&err<CH.goal,ex};}
/* the smallest change (R4.5): fewest changes, then fewest bytes per stored score */
function chBest(){let b=null; CHO.forEach(o=>{const e=chEval(o); if(!e.ok) return; if(!b||o.ch<b.ch||(o.ch===b.ch&&o.bytes<b.bytes)) b=o;}); return b;}

/* =====================================================================
   formatting
   ===================================================================== */
const MINUS='−';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const SUP={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
const sup=n=>String(n).split('').map(c=>SUP[c]||c).join('');
/* a plain number: up to 2 decimals, a real minus sign */
const fN=(v,d)=>{if(Number.isNaN(v)) return 'NaN'; if(v===Infinity) return 'Inf'; if(v===-Infinity) return MINUS+'Inf'; const s=fixd(Math.abs(v),d==null?(Number.isInteger(v)?0:2):d); return (v<0&&+s.replace(/,/g,'')!==0?MINUS:'')+s;};
/* a logit with as many decimals as it has: 92, 13.5, 12.25 */
const fZ=v=>{if(!Number.isFinite(v)) return fN(v); const a=Math.abs(v); if(a>=1e6){const e=Math.log10(a); return Number.isInteger(e)?(v<0?MINUS:'')+'1e'+e:fE(v);} return fN(v,Number.isInteger(v)?0:(Number.isInteger(v*10)?1:2));};
/* a value that may be huge or tiny: 4 significant digits, or a × 10ⁿ */
function fE(x,sig){sig=sig||3; if(Number.isNaN(x)) return 'NaN'; if(x===Infinity) return 'Inf'; if(x===-Infinity) return MINUS+'Inf'; if(x===0) return '0';
  const a=Math.abs(x), neg=x<0?MINUS:'';
  if(a>=0.001&&a<1e5){const d=Math.max(0,sig-1-Math.floor(Math.log10(a))); return neg+fixd(a,Math.min(d,6));}
  let e=Math.floor(Math.log10(a)), m=a/10**e; let ms=m.toFixed(sig-1); if(+ms>=10){e++; ms=(a/10**e).toFixed(sig-1);}
  return `${neg}${ms} × 10${sup(e)}`;}
/* a chance as a percent */
const fP=x=>{if(Number.isNaN(x)) return 'NaN'; if(x===0) return '0%'; const v=x*100; if(v>=99.95&&x<1) return '>99.9%'; if(x>=1) return '100%'; if(v<0.01) return '<0.01%'; if(v<1) return fixd(v,2)+'%'; return fixd(v,1)+'%';};
/* an error in percentage points */
const fPP=x=>Number.isNaN(x)?'NaN':(x===0?'0':(x>=0.001?fE(x,2):fE(x,2)));
const ptsT=x=>Number.isNaN(x)?'NaN':tx(`${fPP(x)} pts`,`${fPP(x)} 个百分点`);
const fF=f=>FMT[f].name;
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):esc(id);
/* Chinese: a link to a lab with a space only on a side where its name is in Latin letters (结构总览和采样, 见 FlashAttention 这样的) */
const linkZ=id=>{const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[id], nm=l&&l.s?A.L(l.s):id; return (/^[A-Za-z0-9]/.test(nm)?' ':'')+labLink(id)+(/[A-Za-z0-9]$/.test(nm)?' ':'');};
const labNm=id=>{const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[id]; return l&&l.s?A.L(l.s):id;};
const math=s=>A.ui.math?A.ui.math(s):esc(s);
/* an exponent, typeset with the kit (R6.29, H1 2026-10-09): e^{z − m} as MathML, in text and in hovers (data-tipx takes HTML) */
const eX=x=>math('e^{'+String(x).replace(/−/g,'-')+'}');
/* the nouns follow the data set (R2.8): a language model's logits give chances; an attention row's scores give attention weights */
const attnNow=()=>{const v=viewNow(); return v!=='ch'&&v!=='on'&&v!=='ce'&&(S.cs==='attn'||S.cs==='mask');};   /* part 6 and the challenge speak of probabilities (owner point 1, B-1) */
const NOUN=()=>attnNow()?{z:tx('score','得分'),zs:tx('scores','得分'),p:tx('attention weight','注意力权重'),ps:tx('attention weights','注意力权重'),top:tx('Largest score','最大的得分')}
  :{z:tx('score','得分'),zs:tx('scores','得分'),p:tx('probability','概率'),ps:tx('probabilities','概率'),top:tx('Largest score','最大的得分')};
/* part 3: the naive softmax agrees with the exact chances when its largest error is below OKE points; okR() is the range of
   largest scores (half steps) where it does, from 88.5 down to −94 (the gate, the hint, the status and the card all use it) */
const zw=w=>/[A-Za-z]/.test(w)?` ${w} `:w;
const OKE=0.01;
/* the naive softmax is off when its error reaches OKE while the max-subtracted one stays far smaller: a range failure, not rounding */
const naiveOff=r=>!hasNaN(r.n.p)&&!(r.errN<OKE)&&r.errN>10*r.errS;
let okRange=null;
function okR(){if(okRange) return okRange;
  const ok=t=>{const z=LM.z.map(v=>v-Math.max(...LM.z)+t), e=errPP(naive(z,'fp32').p,exactP(z)); return !Number.isNaN(e)&&e<OKE;};
  let hi=130; while(hi>-130&&!ok(hi)) hi-=0.5; let lo=hi; while(lo-0.5>=-130&&ok(lo-0.5)) lo-=0.5; return okRange={lo,hi};}
/* part 5: the largest gap at which the naive loss is still finite (103) */
let ceMax=null;
function ceOK(){if(ceMax!=null) return ceMax; let g=0; while(g<150&&Number.isFinite(ceAll(CE.top.concat([CE.top[0]-(g+1)]),CE.y,'fp32').naive)) g++; return ceMax=g;}

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={cs:'lm',fmt:'fp32',top:92,gap:120,mv:'e9'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seenOK=false, seenLow=false, seenF={fp32:true}, seenFin=false, onStep=0, remView='shift';
let saved=null, chPick='start', chTried=new Set(), chSubmitted=false, chReveal=false, sbTouched=false;
let drillSeed=1, drillOpen=false, annT=null, userAct=false, wantObs=false;
const shownPrev={};

/* the logits now on the stage, with their tokens */
function caseNow(){
  if(mode==='challenge') return {toks:CHS.toks,z:CHS.z.slice(),cs:'ch'};
  if(S.cs==='attn'){const mx=Math.max(...AT.z); return {toks:AT.toks,z:AT.z.map(v=>v-mx+S.top),cs:'attn'};}
  if(S.cs==='ce') return {toks:CE.toks,z:CE.top.concat([CE.top[0]-S.gap]),y:CE.y,cs:'ce'};
  if(S.cs==='mask') return {toks:AT.toks,z:AT.z.map(()=>MV[S.mv]),cs:'mask'};
  const mx=Math.max(...LM.z); return {toks:LM.toks,z:LM.z.map(v=>v-mx+S.top),cs:'lm'};
}
function compute(){const c=caseNow(), f=S.fmt, n=naive(c.z,f), st=stable(c.z,f), ex=exactP(c.z);
  return {c,f,F:FMT[f],n,st,ex,errN:errPP(n.p,ex),errS:errPP(st.p,ex),imax:c.z.indexOf(Math.max(...c.z))};}

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 fp:()=>tx('A number format stores every number in a fixed number of bits. FP32 (32 bits, 4 bytes) keeps about 7 significant digits; its largest number is about 3.4 × 10³⁸ and its smallest above 0 about 1.4 × 10⁻⁴⁵. Anything larger becomes Inf (infinity); anything much smaller becomes 0.',
           '数值格式用固定的位数存每个数值。FP32（32 位，4 字节）大约保留 7 位有效数字；它能存的最大数约 3.4 × 10³⁸，大于 0 的最小数约 1.4 × 10⁻⁴⁵。更大的数变成 Inf（无穷大），小得多的数变成 0。'),
 nan:()=>tx('NaN (“not a number”) is what a computer returns when a result has no meaningful value, such as Inf ÷ Inf or 0 ÷ 0. Any arithmetic with a NaN gives NaN again, so one NaN spreads through everything computed from it.',
            'NaN（not a number，“不是数”）是计算机给无定义结果的值，比如 Inf ÷ Inf 或 0 ÷ 0。任何和 NaN 一起算的结果还是 NaN，所以一个 NaN 会传到所有由它算出的数里。'),
 inf:()=>tx('Inf is how a number format writes a result too large to store: in FP32, anything from about 3.4 × 10³⁸ up.','Inf（无穷大）是数值格式对“大到存不下”的结果的写法：在 FP32 里，大约 3.4 × 10³⁸ 以上都是 Inf。'),
 win:()=>{const F=FMT[S.fmt], N=NOUN(); return tx(`The safe window of exp in ${F.name}: above ${fN(F.hi,2)}, the exp of a ${N.z} is larger than ${F.name}’s largest number and becomes Inf; below ${fN(F.lo,2)}, it is smaller than half its smallest number and becomes 0.`,
   `${F.name} 里 exp 的安全区间：${N.z}大于 ${fN(F.hi,2)} 时，它的 exp 超过 ${F.name} 能存的最大数，变成 Inf；小于 ${fN(F.lo,2)} 时，它的 exp 小于最小数的一半，变成 0。`);},
 exact:()=>{const N=NOUN(); return tx(`The exact ${N.ps}: the same softmax computed in float64, with about 16 significant digits, which is exact for every digit shown here.`,`精确的${N.ps}：用 float64（约 16 位有效数字）算同一个 softmax，在这里显示的每一位上都是准确的。`);},
 exactLoss:()=>tx('The exact loss: the same cross-entropy computed in float64, with about 16 significant digits, which is exact for every digit shown here.','精确的 loss：用 float64（约 16 位有效数字）算同一个 cross-entropy，在这里显示的每一位上都是准确的。'),
 pts:()=>{const N=NOUN(); return tx(`pts = percentage points, the gap between two percentages. The exact ${N.ps} are the same softmax computed in float64.`,`个百分点是两个百分数之差。精确的${N.ps}是用 float64 算的同一个 softmax。`);},
 fmts:()=>tx('FP16 and BF16 both use 2 bytes. FP16 keeps 11 significant bits but its largest number is 65,504. BF16’s largest number is about FP32’s, but it keeps only 8 significant bits.','FP16 和 BF16 都占 2 字节。FP16 保留 11 位有效二进制位，但最大只能存 65,504。BF16 能存的最大数和 FP32 差不多，却只保留 8 位有效二进制位。'),
 /* the term “every step rounded to …” in Show the math: what that format keeps (A11) */
 fmtNow:f=>({fp32:tx('Every input, exp, addition and division is rounded to FP32: 4 bytes, about 7 significant digits, largest number about 3.4 × 10³⁸.','每个输入、每次 exp、加法和除法都取整到 FP32：4 字节，大约 7 位有效数字，最大约 3.4 × 10³⁸。'),
   bf16:tx('Every input, exp, addition and division is rounded to BF16: 2 bytes, 8 significant bits, and a largest number about FP32’s.','每个输入、每次 exp、加法和除法都取整到 BF16：2 字节，8 位有效二进制位，能存的最大数和 FP32 差不多。'),
   fp16:tx('Every input, exp, addition and division is rounded to FP16: 2 bytes, 11 significant bits, largest number 65,504.','每个输入、每次 exp、加法和除法都取整到 FP16：2 字节，11 位有效二进制位，最大 65,504。')})[f],
 lse:()=>tx('Log-sum-exp works inside the log and never computes the probability first, so it never takes the log of 0. Part 5 writes its formula on its own line.',
            'log-sum-exp 在 log 里面算，不先算出概率，所以不会对 0 取 log。第 5 部分把它的算式单独写成一行。'),
 ce:()=>tx('Cross-entropy is the training loss for the next token: minus the log of the probability the model gave the right token. A sure, right model has a loss near 0; a model that gives the right token a tiny probability has a large loss.',
           'cross-entropy（交叉熵）是预测下一个 token 的训练 loss：模型给正确 token 的概率取 log，再取负。有把握又答对时 loss 接近 0；给正确 token 的概率很小时 loss 很大。'),
 attn:()=>tx('An attention score says how well one token’s query matches an earlier token’s key. Softmax turns a row of scores into attention weights that add up to 100%.','attention 得分表示一个 token 的 query 和前面某个 token 的 key 有多匹配。softmax 把一行得分变成加起来是 100% 的注意力权重。'),
 temp:()=>{const a=LM.z.map(v=>fN(v/10,1)).join(', '), b=LM.z.map(v=>fZ(v)).join(', ');
   return tx(`Temperature divides every score before softmax (the lab ${labNm('sampling')} shows it). Here the model’s scores are ${a}; divided by 0.1, they become ${b}.`,`temperature 在 softmax 之前把每个得分除以它（“${labNm('sampling')}”这个 Lab 讲 temperature）。这里模型的得分是 ${a.replace(/, /g,'、')}；除以 0.1 以后变成 ${b.replace(/, /g,'、')}。`);},
 /* A26: the kit's logit definition names the LM head; say what it is for a reader who has not met it yet */
 logit:()=>kt('logit')+tx(' The LM head is the model’s last matrix: it gives one score for each token of the vocabulary.','LM head 是模型的最后一个矩阵：它给词表里的每个 token 一个得分。'),
 /* the kit's softmax text writes e^score; tooltips are plain text, so this lab writes exp() (R0.1) */
 sm:()=>tx('Softmax turns a list of scores into probabilities that add up to 100%; a higher score always gets a larger probability.',
           'softmax 把一串得分变成加起来是 100% 的概率；得分越高，概率越大。'),
 kernel:()=>tx('A kernel is a program that runs on the GPU. An attention kernel computes the scores, their softmax and the mix of the value vectors, for many rows at once.','kernel 是在 GPU 上运行的程序。attention 的 kernel 一次为很多行算出得分、softmax，以及按注意力权重混合的 value 向量。'),
 sram:()=>tx('SRAM is the small, fast memory on the GPU chip, next to its compute units: an A100 has 192 KB of it per streaming multiprocessor, against 40 to 80 GB of slower HBM, the GPU’s main memory (Dao et al., 2022).','SRAM 是 GPU 芯片上、紧挨着计算单元的小而快的内存：A100 每个 SM（流式多处理器）有 192 KB，而 GPU 的主存 HBM 有 40 到 80 GB，但慢得多（Dao 等，2022）。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'same',t:{en:'Nothing: every probability stays the same',zh:'不变：每个概率都和原来一样'}},
  {v:'flat',t:{en:'They move toward equal',zh:'各个概率趋向相等'}},
  {v:'sharp',t:{en:'They move apart: the top token gains',zh:'差距拉大：最高的 token 分到更多'}}];
const PRED_RIGHT='same';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
/* parts 1 and 2 show no answer to the prediction (R3.8): no error against the exact chances, no identity in Show the math */
const early=()=>isP('mission')||isP('predict');
const topCtl=(id,inline)=>A.ui.range({id,key:'top',label:tx('Largest score','最大的得分'),min:-260,max:260,value:Math.round(S.top*2),out:fZ(S.top),inline,
  note:inline?'':tx('Moves every score by the same amount; their gaps stay the same.','所有得分一起移动同样多，彼此的差距不变。')});
const fmtCtl=(id,inline)=>A.ui.seg({id,key:'fmt',label:tx('Number format','数值格式'),options:FIDS.map(f=>[f,`${fF(f)} (${FMT[f].bytes} B)`]),value:S.fmt,inline,
  note:inline?'':tx('Every operation is rounded to this format: the inputs, each exp, each addition, each division.','每一步运算都按这个格式取整：输入、每次 exp、每次加法、每次除法。')});
const gapCtl=(id,inline)=>A.ui.range({id,key:'gap',label:tx('Right token below the top score by','正确 token 比最大得分低'),min:0,max:150,value:S.gap,out:fN(S.gap),inline});
const onCtl=()=>`<div class="inline-ctl sm-onctl" id="sm-ic-on"><span class="lbl">${tx('Read the scores block by block','一块一块地读得分')}</span><div class="sm-onbtns"><button type="button" class="btn ghost sm" data-on="-1" aria-label="${esc(tx('Back one block','退回一块'))}">◂</button><button type="button" class="btn sm" data-on="1" aria-label="${esc(tx('Next block','下一块'))}">▸</button><span class="mono sm-onpos"></span></div></div>`;
/* formulas in tooltips (R6.29): the kit renders data-tip strings as HTML, so these carry MathML from A.ui.math */
const FXM=t=>`<div class="fx">${math(t)}</div>`;
A.addStrings({en:{
  sm_temp:`<div>Temperature T divides every score before softmax:</div>${FXM('p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}')}<div>A small T stretches the gaps between the scores, so the top token takes more of the probability; a large T flattens them. Here T = 0.1 turns the model’s scores 9.2, 8.9, 8.7, 8.6, 8.4 and 8.2 into 92, 89, 87, 86, 84 and 82.</div>`,
  sm_ce:`<div>Cross-entropy is the training loss of one prediction: minus the log of the probability the model gave the right answer y.</div>${FXM('\\text{loss} = -\\log p_y')}${FXM('p_y = \\frac{e^{z_y}}{\\sum_j e^{z_j}}')}<div>A sure, right model has a loss near 0; the smaller p<sub>y</sub>, the larger the loss.</div>`,
  sm_sm:`<div>Softmax turns a list of scores into probabilities that add up to 100%; a higher score always gets a larger probability:</div>${FXM('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}`,
  sm_on:`<div>Online softmax reads the scores one block at a time. It keeps the running max m, the largest score so far, and the running sum ℓ, measured against m. When a block brings a larger maximum, the old sum is rescaled once, then the block is added:</div>${FXM('m_{new} = \\max(m, \\max_{block} z)')}${FXM('ℓ \\leftarrow ℓ \\cdot e^{m - m_{new}} + \\sum_{block} e^{z - m_{new}}')}`
},zh:{
  sm_temp:`<div>temperature T 在 softmax 之前把每个得分都除以 T：</div>${FXM('p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}')}<div>T 小，得分之间的差距被拉大，最高的 token 分到更多概率；T 大，各个概率变得更平均。这里 T = 0.1，模型的得分 9.2、8.9、8.7、8.6、8.4、8.2 就变成 92、89、87、86、84、82。</div>`,
  sm_ce:`<div>cross-entropy（交叉熵）是一次预测的训练 loss：模型给正确答案 y 的概率，取 log 再取负。</div>${FXM('\\text{loss} = -\\log p_y')}${FXM('p_y = \\frac{e^{z_y}}{\\sum_j e^{z_j}}')}<div>模型有把握又答对时，loss 接近 0；p<sub>y</sub> 越小，loss 越大。</div>`,
  sm_sm:`<div>softmax 把一串得分变成加起来是 100% 的概率；得分越高，概率越大：</div>${FXM('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}`,
  sm_on:`<div>online softmax 一块一块地读得分。它边读边记两个值：累计最大值 m，也就是目前读到的最大得分；累计和 ℓ，也就是以 m 为准的总和。某一块带来更大的最大值时，先把旧的累计和缩放一次，再加上这一块：</div>${FXM('m_{\\text{新}} = \\max(m, \\max_{\\text{这一块}} z)')}${FXM('ℓ \\leftarrow ℓ \\cdot e^{m - m_{\\text{新}}} + \\sum_{\\text{这一块}} e^{z - m_{\\text{新}}}')}`
}});
/* a term whose hover is one of the strings above (dotted underline, help cursor, R1.3) */
const DEFK=(text,key)=>`<span class="def" tabindex="0" data-tip="${key}">${text}</span>`;
const FXL=t=>`<p class="sm-fx">${math(t)}</p>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'sm',
  lead:()=>tx('A sampler’s softmax in FP32 gets scores up to 92 and prints NaN.','采样器用 FP32 算 softmax，最大的得分是 92，结果输出 NaN。'),
  enter:()=>({...BASE}),
  body:()=>{const r=compute(), top=fZ(Math.max(...LM.z));
   return tx(`<p>${DEFK('Softmax','sm_sm')} turns a list of scores into probabilities that add up to 100%:</p>${FXL('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}<p>You write the sampler of a chat app. After “${esc(LM.prompt)}”, the model gives each candidate next token a score (LLMs call these scores ${DEF('logits',TIP.logit())}). The sampler runs at a ${DEFK('temperature','sm_temp')} of 0.1, which divides every score by 0.1, so the largest score becomes ${top}. Your softmax runs in ${DEF('FP32',TIP.fp())}.</p><p>It prints ${DEF('NaN',TIP.nan())} for “Paris” and “the”, and 0 for the other four. The same softmax with the largest score subtracted first gives “Paris” ${fP(r.st.p[0])} (the right column in the chart). Where does the NaN come from, and why does the subtraction fix it?</p>`,
             `<p>${DEFK('softmax','sm_sm')} 把一串得分变成加起来是 100% 的概率：</p>${FXL('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}<p>你在给一个聊天应用写采样器。模型读完“${esc(LM.prompt)}”，给每个候选的下一个 token 打一个得分（LLM 里把这种得分叫 ${DEF('logit',TIP.logit())}）。采样器的 ${DEFK('temperature','sm_temp')} 设成 0.1，也就是把每个得分除以 0.1，于是最大的得分变成 ${top}。softmax 用 ${DEF('FP32',TIP.fp())} 计算。</p><p>结果“Paris”和“the”的概率是 ${DEF('NaN',TIP.nan())}，另外四个是 0。同样的 softmax，如果先让每个得分减去最大的那个，就给出“Paris” ${fP(r.st.p[0])}（图里右边一列）。NaN 是从哪来的？为什么减一下就好了？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},view:'sm',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: subtract the same number from every score; what happens to the probabilities?','先猜一猜：每个得分都减去同一个值，概率会怎样？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Forget the computer for a moment and think of exact math. If you subtract the same number, say 92, from every score, what happens to the probabilities softmax gives?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>先不管计算机，只想精确的数学。每个得分都减去同一个值，比如 92，softmax 给出的概率会怎样？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'shift',title:{en:'Move every score together',zh:'把所有得分一起移动'},view:'sm',
  lead:()=>tx('Drag the largest score down until the naive softmax works; then far below 0.','把最大的得分往下拖，直到朴素算法能算出结果；再拖到远低于 0 的地方。'),
  enter:()=>({...BASE}),expect:()=>seenOK,showMe:{top:60},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The slider moves all six scores by the same amount, so the gaps between them never change. The shaded band in the chart is where FP32 can compute exp without getting Inf or 0: the ${DEF('safe window',TIP.win())}.</p>`,
              `<p>滑块让六个得分一起移动同样多，它们之间的差距始终不变。图里的阴影带是 FP32 能算出 exp、不会得到 Inf 或 0 的范围，也就是${DEF('安全区间',TIP.win())}。</p>`)+topCtl('sm-ic-top',true),
  obs:()=>obsShift()},
 {id:'fmt',title:{en:'FP16 has a tiny window',zh:'FP16 的安全区间很窄'},view:'sm',
  lead:()=>tx('Switch these six scores to FP16, then to BF16.','把这六个得分换成 FP16，再换成 BF16。'),
  enter:()=>{seenF={fp32:true}; return {...BASE,cs:'attn',top:13.5};},expect:()=>!!seenF.fp16,showMe:{fmt:'fp16'},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>A model also uses softmax inside: ${DEF('attention',TIP.attn())} turns each row of scores into weights with it, and those scores are often kept in 2-byte formats. Here one row has six scores, up to 13.5. Pick a ${DEF('format',TIP.fmts())}:</p>`,
              `<p>模型内部也用 softmax：${DEF('attention',TIP.attn())} 用它把一行得分变成权重，而这些得分常常存成 2 字节的格式。这里一行有六个得分，最大的是 13.5。选一个${DEF('格式',TIP.fmts())}：</p>`)+fmtCtl('sm-ic-fmt',true),
  obs:()=>obsFmt()},
 {id:'ce',title:{en:'A loss that stays finite',zh:'不会变成 Inf 的 loss'},view:'ce',
  lead:()=>tx('The right answer’s score sits 120 below the top score: drag it up until the naive loss is finite.','正确答案的得分比最大的得分低 120：把它往上拖，直到朴素算法的 loss 变成有限值。'),
  enter:()=>({...BASE,cs:'ce',gap:120}),expect:()=>seenFin,showMe:{gap:90},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Training measures how good each prediction is with ${DEFK('cross-entropy','sm_ce')}:</p>${FXL('\\text{loss} = -\\log p_y')}<p>${math('p_y')} is the probability the model gives the right answer y. Here the text goes on with “not”, but the model gave “not” a score far below its top score.</p>`,
              `<p>训练时用 ${DEFK('cross-entropy','sm_ce')} 衡量每一次预测的好坏：</p>${FXL('\\text{loss} = -\\log p_y')}<p>${math('p_y')} 是模型给正确答案 y 的概率。这里正确的下一个 token 是“not”，可模型给它的得分远低于最大的得分。</p>`)+gapCtl('sm-ic-gap',true),
  obs:()=>obsCe()},
 {id:'online',title:{en:'Softmax one block at a time',zh:'一块一块地算 softmax'},view:'on',
  lead:()=>tx('Read eight scores in four blocks, and keep a running max and a running sum.','分四块读八个得分，边读边记最大值和总和。'),
  enter:()=>{onStep=observed[step]?NB+1:0; return {...BASE};}/* R3.20: blocks already read to the end stay read (◂ steps back) */,expect:()=>onStep>=ON.z.length/ON.B+1,showMe:()=>setOn(ON.z.length/ON.B+1),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>A long row of scores often cannot be read at once: a GPU’s ${DEF('fast on-chip memory (SRAM)',TIP.sram())} holds only a small block at a time. The softmax with the max subtracted would then read the row twice: once to find the max, once more to add up.</p>`+
     `<p>${DEFK('Online softmax','sm_on')} reads it once. It keeps two numbers as it goes: the running max m, the largest score so far, and the running sum ℓ. When a block brings a larger max, it does not start over: it rescales the old sum once, then adds the new block. At the end the result is exactly the one two passes give. FlashAttention computes attention this way: ${labLink('flash-attention')}.</p><p>After each block, the running sum is updated like this:</p>${FXL('ℓ \\leftarrow ℓ \\cdot e^{m_{old} - m_{new}} + \\sum_{block} e^{z - m_{new}}')}`,
     `<p>一行得分很长时，往往没法一次读完：GPU 的${DEF('片上高速存储（SRAM）',TIP.sram())}一次只放得下一小块。这时，先减最大值的 softmax 就得把整行读两遍，一遍找最大值，一遍求和。</p>`+
     `<p>${DEFK('online softmax','sm_on')} 只读一遍。它边读边记两个值：累计最大值 m，也就是目前读到的最大得分；累计和 ℓ，也就是目前为止的总和。读到更大的最大值时，它不用从头再来，只要把旧的累计和缩放一次，再加上新的这一块。读完以后，结果和读两遍完全一样。FlashAttention 算 attention 用的就是这个办法，见 ${labLink('flash-attention')} 这个 Lab。</p><p>每读完一块，累计和这样更新：</p>${FXL('ℓ \\leftarrow ℓ \\cdot e^{m_{\\text{旧}} - m_{\\text{新}}} + \\sum_{\\text{这一块}} e^{z - m_{\\text{新}}}')}`)+onCtl(),
  obs:()=>obsOn()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>({shift:'sm',ce:'ce',on:'on'}[remView]||'sm'),
  lead:()=>tx('Subtract the max; take the log before the exp; rescale the running sum when the max grows.','先减最大值；先取 log，不要先算概率；最大值变大时缩放累计和。'),
  enter:()=>{remView='shift'; return {...BASE};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
/* part 3: the result at the learner's largest score, then the verdict, then why */
function obsShift(){
  const r=compute(), F=r.F, top=fZ(S.top), K=okR(), why=tx(`<p>Why: ${math('e^{z - c} = e^{z} \\times e^{-c}')}. When every score loses the same c, every term above and below the fraction line is multiplied by the same ${eX('−c')}, and it cancels in the division. So subtracting the max changes nothing, and afterwards the largest exponent is exactly 0, which cannot overflow.</p>`,
    `<p>原因：${math('e^{z - c} = e^{z} \\times e^{-c}')}。每个得分都减去 c，分子和分母里的每一项就都乘上同一个 ${eX('−c')}，相除时正好约掉。所以减去最大值不改变结果，而减完以后最大的指数正好是 0，不会溢出。</p>`);
  /* before the learner has seen the naive softmax agree, no verdict: say where it does (A2) */
  const after=seenOK?verdict()+why:tx(`<p class="small">Drag it to a value between ${fZ(K.lo)} and ${fZ(K.hi)}: there the naive softmax works.</p>`,`<p class="small">把它拖到 ${fZ(K.lo)} 到 ${fZ(K.hi)} 之间：在这个范围里，朴素算法能算出结果。</p>`);
  if(S.top>F.hi) return tx(`<p><b>With a largest score of ${top}, ${eX(top)} is larger than the largest number FP32 can hold, so it becomes Inf. The sum is Inf too, and Inf ÷ Inf is NaN.</b> Drag it below ${fN(F.hi,2)}.</p>`,`<p><b>最大的得分是 ${top} 时，${eX(top)} 超过了 FP32 能存的最大数，变成 Inf。总和也是 Inf，而 Inf ÷ Inf 得到 NaN。</b>把它拖到 ${fN(F.hi,2)} 以下。</p>`);
  if(hasNaN(r.n.p)) return tx(`<p><b>With a largest score of ${top}, every ${eX('z')} is smaller than the smallest number FP32 can hold, so each becomes 0. The sum is 0, and 0 ÷ 0 is NaN.</b> The column with the max subtracted still gives “Paris” ${fP(r.st.p[0])}.</p>`,`<p><b>最大的得分是 ${top} 时，每个 ${eX('z')} 都小于 FP32 能存的最小数，都变成了 0。总和是 0，而 0 ÷ 0 得到 NaN。</b>先减最大值的那一列仍然给“Paris” ${fP(r.st.p[0])}。</p>`)+after;
  if(r.errN>=OKE){const a=fP(r.n.p[0]), b=fP(r.st.p[0]);
    return tx(`<p><b>With a largest score of ${top}, the naive softmax gives finite probabilities, but they are off by up to ${ptsT(r.errN)}${a!==b?`: “Paris” gets ${a} instead of ${b}`:''}.</b> This far down, ${eX('z')} is so small that FP32 keeps only a few of its digits, and the smallest ones become 0.</p>`,
              `<p><b>最大的得分是 ${top} 时，朴素算法算出了概率，但最多差了 ${ptsT(r.errN)}${a!==b?`：“Paris”得到 ${a}，应该是 ${b}`:''}。</b>得分这么低时，${eX('z')} 非常小，FP32 只能保留它的几位数字，最小的几个直接变成 0。</p>`)+after;}
  const res=tx(`<p><b>With a largest score of ${top}, the naive softmax works, and its probabilities match the column with the max subtracted to every digit shown: “Paris” gets ${fP(r.n.p[0])}.</b> Every score moved by ${fZ(S.top-Math.max(...LM.z))}, and no probability changed.</p>`,
        `<p><b>最大的得分是 ${top} 时，朴素算法能算了，给出的概率和先减最大值的那一列每一位都一样：“Paris”得到 ${fP(r.n.p[0])}。</b>每个得分都移动了 ${fZ(S.top-Math.max(...LM.z))}，没有一个概率变了。</p>`);
  const low=seenLow?'':tx(`<p class="small">Now drag it below ${fN(F.lo,2)}.</p>`,`<p class="small">再把它拖到 ${fN(F.lo,2)} 以下。</p>`);
  return res+verdict()+why+low;
}
/* part 4: the format */
function obsFmt(){
  const r=compute(), F=r.F, z=r.c.z, over=z.filter(v=>v>F.hi).length;
  if(S.fmt==='fp32') return tx(`<p><b>In FP32 both softmaxes work: every score is far inside the window, and both are within ${ptsT(Math.max(r.errN,r.errS))} of the exact weights.</b> Pick FP16.</p>`,`<p><b>FP32 里两种算法都没问题：每个得分都远在安全区间以内，算出的权重离精确值都不到 ${ptsT(Math.max(r.errN,r.errS))}。</b>选 FP16 试试。</p>`);
  if(S.fmt==='fp16') return tx(`<p><b>FP16’s window ends at ${fN(F.hi,2)}. ${['None','One','Two','Three','Four','Five','Six'][over]} of the six scores ${over===1?'is':'are'} above it, and the naive softmax returns NaN. With the max subtracted, FP16 stays within ${ptsT(r.errS)} of the exact weights.</b></p><p>Why: the largest number FP16 can hold is 65,504, and ${eX('11.1')} is already larger. After the subtraction every exponent is at most 0, so every ${eX('z − m')} is at most 1. ${DEF('Kernels',TIP.kernel())} still keep the max and the sum in FP32, because a long FP16 sum loses digits <span class="sm-nw">(${J('fid','see the notes at the end of the page')})</span>.${seenF.bf16?'':' Now try BF16.'}</p>`,
    `<p><b>FP16 的安全区间到 ${fN(F.hi,2)} 就结束了。六个得分里有 ${over} 个超过它，朴素算法返回 NaN。先减最大值，FP16 算出的权重离精确值不到 ${ptsT(r.errS)}。</b></p><p>原因：FP16 最大只能存 65,504，而 ${eX('11.1')} 就已经比它大。减去最大值以后，每个指数都不大于 0，每个 ${eX('z − m')} 都不超过 1。不过 ${DEF('kernel',TIP.kernel())} 仍把最大值和总和放在 FP32 里，因为很长的 FP16 求和会丢掉数字<span class="sm-nw">（${J('fid','见页末的“简化、假设和来源”')}）</span>。${seenF.bf16?'':'再试试 BF16。'}</p>`);
  /* A20: the max-subtracted one can come out a little less accurate; say why when it does */
  const closer=r.errN<r.errS?tx(' Subtracting the max prevents overflow, not rounding error, so here it is not more accurate. Which of the two rounds closer depends on the inputs.','减去最大值防的是溢出，不是取整误差，所以这里它并不更准。两种算法谁的误差小，要看具体的输入。'):'';
  return tx(`<p><b>BF16’s largest number is about FP32’s, so nothing overflows in BF16. But BF16 keeps only 8 significant bits: the naive softmax is off by ${ptsT(r.errN)}, and the one with the max subtracted by ${ptsT(r.errS)}.</b></p><p>Why: BF16 trades digits for range.${closer} The lab ${labLink('number-formats')} compares the formats.</p>`,
            `<p><b>BF16 能存的最大数和 FP32 差不多，所以不会溢出。但它只有 8 位有效二进制位：朴素算法差 ${ptsT(r.errN)}，先减最大值的差 ${ptsT(r.errS)}。</b></p><p>原因：BF16 用精度换来了范围。${closer}${linkZ('number-formats')}这个 Lab 专门比较这些格式。</p>`);
}
/* part 5: cross-entropy */
function obsCe(){
  const r=compute(), c=ceAll(r.c.z,CE.y,S.fmt), g=fN(S.gap), at=S.gap?tx(`${g} below the top`,`比最大的得分低 ${g} `):tx('level with the top','和最大的得分一样高');
  if(c.naive===Infinity) return tx(`<p><b>With “not” ${g} below the top score, its probability is about ${fE(Math.exp(-c.exact))}, far below the smallest number FP32 can hold. It becomes 0, and ${math('-\\log 0 = \\text{Inf}')}. Subtracting the max does not help, because the probability itself is too small. Log-sum-exp gives the right loss, ${fN(c.lse,2)}.</b> Drag the gap down to ${ceOK()} or less.</p>`,
                                         `<p><b>“not”比最大的得分低 ${g} 时，它的概率约是 ${fE(Math.exp(-c.exact))}，远小于 FP32 能存的最小数，于是变成 0，而 ${math('-\\log 0 = \\text{Inf}')}。先减最大值也没用，因为是概率本身太小。log-sum-exp 给出正确的 loss：${fN(c.lse,2)}。</b>把差距拖到 ${ceOK()} 或更小。</p>`);
  const off=Math.abs(c.naive-c.exact)>=0.005, at120=ceAll(CE.top.concat([CE.top[0]-120]),CE.y,'fp32');
  return tx(`<p><b>With “not” ${at}, the naive loss is finite again: ${fN(c.naive,2)}${off?`, while the exact loss is ${fN(c.exact,2)}`:', the same as log-sum-exp'}.</b> At 120 below, it was Inf, while log-sum-exp gave ${fN(at120.lse,2)}.</p><p>Why: log-sum-exp never computes the probability first. It works inside the log from the start:</p>${FXL('\\text{loss} = m + \\log \\sum_j e^{z_j - m} - z_y')}<p>Here m is the largest score and ${math('z_y')} the score of the right answer, “not”. Every term is a normal-sized number.</p>`,
            `<p><b>“not”${at}时，朴素算法的 loss 又是有限值了：${fN(c.naive,2)}${off?`，精确的 loss 是 ${fN(c.exact,2)}`:'，和 log-sum-exp 一样'}。</b>低 120 时它是 Inf，而 log-sum-exp 给出 ${fN(at120.lse,2)}。</p><p>原因：log-sum-exp 不先算概率，从一开始就在 log 里算。</p>${FXL('\\text{loss} = m + \\log \\sum_j e^{z_j - m} - z_y')}<p>其中 m 是最大的得分，${math('z_y')} 是正确答案“not”的得分。式子里每一项都是大小正常的数。</p>`);
}
/* part 6: online softmax, block by block, in words (R6.27): what happened to m and s, never arithmetic to follow */
const NB=ON.z.length/ON.B;
function onNow(){return online(ON.z,ON.B,'fp32');}
const maxTok=(o,k)=>{const read=ON.z.slice(0,k*ON.B), m=Math.max(...read); return ON.toks[read.indexOf(m)];};
function obsOn(){
  const o=onNow();
  if(onStep===0) return tx('<p><b>Nothing read yet: there is no max so far (m = −Inf), and the running sum ℓ is 0.</b> Press ▸ to read the first block.</p>','<p><b>还没开始读：还没有最大值（m = −Inf），累计和 ℓ 是 0。</b>按 ▸ 读第一块。</p>');
  if(onStep<=NB){const st=o.steps[onStep-1], b=onStep, m1=fN(st.mNew,1), m0=fN(st.mOld,1), tk=maxTok(o,b);
    if(st.first) return tx(`<p><b>Block 1: the max so far is ${m1} (“${tk}”). Each of the block’s two scores adds one piece to the running sum ℓ. The closer a score is to the max, the longer its piece, and the max’s own piece is exactly 1.</b></p>`,`<p><b>第 1 块：目前的最大值是 ${m1}（“${tk}”）。这一块的两个得分各在累计和 ℓ 里占一段。得分越接近最大值，这一段越长，最大值自己那段正好是 1。</b></p>`);
    if(st.raised) return tx(`<p><b>Block ${b} brings a larger score, ${m1} (“${tk}”), so the max rises from ${m0} to ${m1}. The old running sum was measured against ${m0}, so it is rescaled once (× ${fE(st.sc,3)}) to be measured against ${m1}. Then the new block is added.</b> The old pieces shrink, and no score is read again.</p>`,
                                    `<p><b>第 ${b} 块里有更大的得分 ${m1}（“${tk}”），最大值从 ${m0} 变成 ${m1}。旧的累计和是以 ${m0} 为准的，所以先整体缩放一次（× ${fE(st.sc,3)}），换成以 ${m1} 为准，再加上新的这一块。</b>旧的几段都变短了，已读的得分不用重读。</p>`);
    return tx(`<p><b>Block ${b} has no score above ${fN(st.mNew,1)}, so the max stays and the old running sum needs no change. The two new pieces are simply added.</b></p>`,`<p><b>第 ${b} 块的得分都不超过 ${fN(st.mNew,1)}，最大值不变，旧的累计和也不用动，新的两段直接加上。</b></p>`);}
  return tx(`<p><b>All four blocks read: m = ${fN(o.m,1)} and ℓ = ${fE(o.s,4)}. Each probability is its score’s piece divided by ℓ, and the result equals two passes (one for the max, one for the sum) ${o.diff===0?'bit for bit':`to ${fE(o.diff)}`}.</b></p><p>Why: after every block, ℓ is exactly this sum over the scores read so far, the same sum the second pass computes:</p>${FXL('ℓ = \\sum_{\\text{read}} e^{z - m}')}<p>Rescaling only moves the old pieces to the new max; it never approximates.</p>`,
            `<p><b>四块都读完了：m = ${fN(o.m,1)}，ℓ = ${fE(o.s,4)}。每个概率就是这个得分的那一段除以 ℓ，结果和读两遍（一遍找最大值、一遍求和）${o.diff===0?'每一位都相同':`最多差 ${fE(o.diff)}`}。</b></p><p>原因在于，每读完一块，ℓ 都正好等于下面这个和，读两遍时求和那一遍算的也是它：</p>${FXL('ℓ = \\sum_{\\text{已读}} e^{z - m}')}<p>缩放只是把旧的几段换成以新的最大值为准，没有任何近似。</p>`);
}
/* part 7: three points with this lab's examples; a button puts that part's picture back in the chart */
function remBtn(k){const on=remView===k;
  const tip={shift:tx('Shows the scores up to 92 in the chart, with the naive softmax’s NaN and the column with the max subtracted.','在图里显示最高 92 的得分、朴素算法的 NaN，以及先减最大值的那一列。'),ce:tx('Shows the losses in the chart when the right token, “not”, is 120 below the top score.','在图里显示正确 token“not”比最大的得分低 120 时的 loss。'),on:tx('Shows the eight scores in the chart, read in four blocks with a running max and sum.','在图里显示分四块读完的八个得分，以及累计的最大值和总和。')}[k];
  return ` <button type="button" class="linkbtn sm-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const st=stable(LM.z,'fp32'), c=ceAll(CE.top.concat([CE.top[0]-120]),CE.y,'fp32'), o=onNow(), r2=o.steps[1];
  const top=fZ(Math.max(...LM.z));
  return tx(`<ul class="sm-remlist"><li><b>Subtract the largest score before exp.</b> It changes no probability, and every ${eX('z − m')} becomes at most 1, so no exp can overflow, however large the scores get. That is why PyTorch’s softmax and FlashAttention always do it.${remBtn('shift')}</li>`+
    `<li><b>For a loss, take the log first; do not compute the probability first.</b> When the right token’s score is far below the top one, its probability rounds to 0, and the log of 0 makes the loss Inf. Log-sum-exp never computes that probability, so the loss stays finite and training can still learn from the wrong guess.${remBtn('ce')}</li>`+
    `<li><b>Reading block by block, keep a running max and a running sum; when the max grows, rescale the old sum once.</b> So each block is read only once: a row too long for fast memory still takes one pass, and the result equals the two-pass softmax. FlashAttention rescales its output the same way, so it never stores the whole row of scores.${remBtn('on')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Subtracting the max changes the probabilities.” Softmax depends only on the gaps between scores, so moving every score by the same amount changes no probability. Subtracting the max is such a move.</p><p>✗ “NaN means the model’s parameters are broken.” The arithmetic alone can give NaN: one score large enough makes its exp overflow FP32 to Inf, and Inf ÷ Inf is NaN. The model itself may be fine.</p><p>✗ “Online softmax is an approximation.” It computes the same sum as two passes, only added block by block, so the result is exact up to rounding.</p></div>${A.shell.toScenarios()}`,
    `<ul class="sm-remlist"><li><b>取 exp 之前，先减去最大的得分。</b>这样做不改变任何概率，而且每个 ${eX('z − m')} 都不超过 1，所以得分再大，exp 也不会溢出。PyTorch 的 softmax 和 FlashAttention 都是这样做的。${remBtn('shift')}</li>`+
    `<li><b>算 loss 时先取 log，不要先算概率。</b>正确 token 的得分比最大的得分低很多时，它的概率会被舍入成 0，而 ${math('\\log 0')} 会让 loss 变成 Inf。log-sum-exp 不先算这个概率，所以 loss 一直是有限的，训练还能从这次猜错里学到东西。${remBtn('ce')}</li>`+
    `<li><b>一块一块地读时，边读边记最大值和总和；最大值变大，就把旧的总和缩放一次。</b>这样一整行只要读一遍，即使它长得放不进高速存储，结果也和读两遍的 softmax 一样。FlashAttention 用同样的办法缩放它的输出，所以不用存下整行得分。${remBtn('on')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “减去最大值会改变概率。”softmax 只看得分之间的差，所以所有得分一起加上或减去同一个数值，概率都不变。减去最大值就是这样的移动。</p><p>✗ “出现 NaN，说明模型的参数坏了。”只靠计算本身也会出 NaN。只要有一个得分大到它的 exp 超出 FP32 的范围，这个 exp 就会变成 Inf，而 Inf ÷ Inf 得到 NaN。这时模型本身可能没有任何问题。</p><p>✗ “online softmax 是近似算法。”它和读两遍的算法算的是同一个总和，只是分块加起来，所以结果是精确的，最多差一点舍入。</p></div>${A.shell.toScenarios()}`);
}
/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 ptc:'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SoftMax.cpp',
 ptk:'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp',
 ptu:'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu',
 fa:'https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h',
 onl:'https://arxiv.org/abs/1805.02867',
 fla:'https://arxiv.org/abs/2205.14135',
 ieee:'https://en.wikipedia.org/wiki/Half-precision_floating-point_format'
};
const PRESETS=[
 {id:'samp',t:{en:'A sampler at a low temperature',zh:'低 temperature 的采样器'},d:{en:'Scores divided by a small temperature reach 92, in FP32',zh:'得分除以很小的 temperature 后高达 92，用 FP32'},s:{cs:'lm',fmt:'fp32',top:92}},
 {id:'attn',t:{en:'Attention scores in FP16',zh:'FP16 的 attention 得分'},d:{en:'One token’s six scores, up to 13.5, kept in 2 bytes each',zh:'一个 token 的六个得分，最高 13.5，每个占 2 字节'},s:{cs:'attn',fmt:'fp16',top:13.5}},
 {id:'loss',t:{en:'A training loss for a wrong guess',zh:'猜错时的训练 loss'},d:{en:'The right token’s score sits 120 below the top one',zh:'正确 token 的得分比最大的得分低 120'},s:{cs:'ce',fmt:'fp32',gap:120}},
 {id:'pad',t:{en:'A padding row, every token masked',zh:'整行都被 mask 的 padding'},d:{en:'Masked scores set to −1e9 or to −Inf',zh:'被 mask 的得分设成 −1e9 或 −Inf'},s:{cs:'mask',fmt:'fp32'}}];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k]));
function scenHTML(i){
  const p=PRESETS[i], r=compute();
  if(p.id==='samp') return tx(`<p><b>The sampler’s temperature is 0.1, so every score is divided by 0.1, and the top score, 9.2, becomes 92. In FP32 the naive softmax returns NaN for “Paris” and “the”. With the max subtracted first, “Paris” gets ${fP(r.st.p[0])}.</b></p><p>Any code path that divides scores before softmax can push them past FP32’s window, which is why samplers and PyTorch’s softmax subtract the max.</p>`,
                                 `<p><b>采样器的 temperature 是 0.1，每个得分都要除以 0.1，最高的 9.2 就变成了 92。在 FP32 里，朴素 softmax 给“Paris”和“the”算出 NaN。先减去最大值再算，“Paris”得到 ${fP(r.st.p[0])}。</b></p><p>任何在 softmax 之前放大得分的代码，都可能把它们推出 FP32 的安全区间，所以采样器和 PyTorch 的 softmax 都先减最大值。</p>`);
  if(p.id==='attn') return tx(`<p><b>This row of attention scores is stored in FP16. Its top score, 13.5, is past FP16’s window, which ends at ${fN(FMT.fp16.hi,2)}, so the naive softmax returns NaN. With the max subtracted, its attention weights are within ${ptsT(r.errS)} of the exact ones.</b></p><p>FlashAttention also subtracts the max so far as it reads. Even with FP16 or BF16 inputs, it keeps the max and the sum in FP32 registers (${cite(SRC.fa,'softmax.h')}).</p>`,
                                 `<p><b>这一行 attention 得分存成 FP16，最高的是 13.5，超出了 FP16 的安全区间（到 ${fN(FMT.fp16.hi,2)} 为止），所以朴素 softmax 返回 NaN。先减最大值，算出的注意力权重离精确值不到 ${ptsT(r.errS)}。</b></p><p>FlashAttention 也是边读边减去目前的最大值。即使输入是 FP16 或 BF16，它也把最大值和总和放在 FP32 寄存器里（${cite(SRC.fa,'softmax.h')}）。</p>`);
  if(p.id==='loss'){const c=ceAll(r.c.z,CE.y,S.fmt);
    return tx(`<p><b>A model can be sure of the wrong token, for example on a mislabeled example or in a diverging run. With the right token’s score 120 below the top score, its probability underflows to 0 in FP32, and ${math('-\\log 0 = \\text{Inf}')}. Log-sum-exp gives ${fN(c.lse,2)}.</b></p><p>An Inf loss turns every gradient into NaN in the backward pass of that step. PyTorch’s cross-entropy calls log_softmax, which computes ${math('z_i - m - \\log \\sum_j e^{z_j - m}')}, with m the largest score (${cite(SRC.ptu,'SoftMax.cu')}).</p>`,
              `<p><b>模型有时很有把握地选错 token，比如遇到标错的样本，或者训练发散的时候。正确 token 的得分比最大的得分低 120 时，它的概率在 FP32 里下溢成 0，而 ${math('-\\log 0 = \\text{Inf}')}。log-sum-exp 给出 ${fN(c.lse,2)}。</b></p><p>loss 一旦是 Inf，这一步反向传播得到的梯度就全是 NaN。PyTorch 的 cross-entropy 调用 log_softmax，它算的是 ${math('z_i - m - \\log \\sum_j e^{z_j - m}')}，m 是最大的得分（${cite(SRC.ptu,'SoftMax.cu')}）。</p>`);}
  const e9=S.mv==='e9';
  return tx(`<p><b>When every token of a row is masked, every ${eX('z')} is 0 and the naive softmax gives 0 ÷ 0 = NaN. ${e9?'If the max is subtracted first, a mask value of −1e9 gives every token the same weight, 1/6. There is no NaN, but the row means nothing':'With −Inf, subtracting the max gives NaN too, because the max is −Inf and −Inf − (−Inf) is NaN'}.</b> Switch the mask value under ${J('controls','Settings')}.</p><p>When only some tokens are masked, both values give them an attention weight of exactly 0. FlashAttention guards the fully masked row: when the max is −Inf it subtracts 0 instead (${cite(SRC.fa,'softmax.h')}). In FP16, −1e9 itself rounds to −Inf.</p>`,
            `<p><b>一行里所有 token 都被 mask 时，每个 ${eX('z')} 都是 0，朴素 softmax 得到 0 ÷ 0 = NaN。${e9?'如果先减最大值，用 −1e9 做 mask 时，每个 token 的权重都是 1/6。这样不会出现 NaN，但这一行的结果没有意义':'如果用 −Inf 做 mask，先减最大值也会得到 NaN，因为最大值本身就是 −Inf，而 −Inf − (−Inf) 是 NaN'}。</b>在“${J('controls','自由调参')}”里切换 mask 的值。</p><p>只有部分 token 被 mask 时，两种值都让它们的注意力权重正好是 0。FlashAttention 专门处理整行被 mask 的情况：最大值是 −Inf 时改减 0（${cite(SRC.fa,'softmax.h')}）。在 FP16 里，−1e9 本身就会取整成 −Inf。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chOpt=id=>id==='start'?CH0:CHO.find(o=>o.id===id);
/* the brief chart's rows (the owner's review of batch 8): [CH0, ...CHO] in order, only the fixes that meet the goal, the pick and the
   initial setup, as A.ui.cmpChart({brief:true}) keeps them */
const chShown=()=>[CH0].concat(CHO).map(chEval).filter(e=>e.ok||e.o.id===chPick||e.o.id==='start');
/* a bar past 55% of the track carries its label inside its end (R7.8) */
function chChartFix(box){
  const rows=chShown(), lg=x=>Number.isNaN(x)?9.4:Math.max(0.25,Math.min(9.4,Math.log10(Math.max(x,1e-12))+7));
  let nan=false; $$('.cmp-row',box).forEach((el,i)=>{const e=rows[i]; if(!e) return; const n=Number.isNaN(e.err); nan=nan||n; el.classList.toggle('sm-cmp-in',lg(e.err)/9.6>0.55); el.classList.toggle('sm-cmp-nan',n);});
  const leg=$('.cmp-leg',box); if(nan&&leg&&!$('.sm-leg-nan',leg)) leg.insertAdjacentHTML('beforeend',`<span><i class="sm-leg-nan"></i>${esc(tx('NaN: no result','NaN：没有结果'))}</span>`);
}
const chLabel=o=>`${L(o.t)} (${o.bytes} B)`;
function chResultHTML(){
  if(!chSubmitted) return '';
  const o=chOpt(chPick), e=chEval(o), best=chBest();
  let h;
  const good=`<p class="okline sm-mt"><b>${tx(`Goal met: the largest error is ${ptsT(e.err)}, below 0.1 points.`,`达成目标：最大误差是 ${ptsT(e.err)}，低于 0.1 个百分点。`)}</b>`;
  if(o.id==='start') h=tx('<p class="badline sm-mt"><b>Not yet: you kept the naive softmax in FP16, which returns NaN.</b> Pick a fix first.</p>','<p class="badline sm-mt"><b>还没达到：你保留了 FP16 的朴素 softmax，它返回 NaN。</b>先选一个办法。</p>');
  else if(e.ok&&o.id===best.id) h=good+tx(' And that is the smallest change: the scores stay in FP16, 2 bytes each, and the only change is one subtraction before exp.</p>','而且这是改动最少的做法：得分仍是 FP16，每个 2 字节，唯一的改动是在 exp 之前做一次减法。</p>');
  else if(e.ok&&o.bytes===best.bytes) h=good+tx(' But a smaller change exists.</p>','不过还有改动更少的做法。</p>')+tx(`<p>This is what real GPU code does, and it keeps 2 bytes per stored score. It makes two changes, though, and one is enough here: subtracting the max alone, with FP16 arithmetic, also meets the goal. The FP32 sum matters on long rows (see the note below the chart).</p>`,`<p>实际的 GPU 代码就是这样做的，每个存储的得分也只占 2 字节。它改了两处，而这里改一处就够：只减去最大值、仍用 FP16 计算，也能达标。FP32 的总和在很长的一行里才要紧（见图下面的说明）。</p>`);
  else if(e.ok) h=good+tx(' But a smaller change exists.</p>','不过还有改动更少的做法。</p>')+tx(`<p>Subtracting the max alone, in FP16, also meets the goal and stores 2 bytes per score. Your fix stores ${o.bytes}.</p>`,`<p>只减去最大值、仍用 FP16，也能达标，而且每个得分只存 2 字节。你的办法要存 ${o.bytes} 字节。</p>`);
  else {
    const why={bf16:tx('BF16’s largest number is about FP32’s, so nothing overflows, but with 8 significant bits the probabilities come out too coarse.','BF16 能存的最大数和 FP32 差不多，所以不会溢出；但它只有 8 位有效二进制位，算出的概率太粗。'),
      half:tx('Halving the scores keeps every exp finite, but softmax of the halved scores is a different, flatter set of probabilities.','把得分减半，每个 exp 都不会溢出；但减半后的 softmax 是另一组更平的概率。'),
      clamp:tx(`Clamping gives three scores the value 11, and three ${eX('11')} (about 59,900 each) already add up past 65,504: the sum is Inf and every probability comes out 0.`,`截断后有三个得分变成 11，三个 ${eX('11')}（每个约 59,900）加起来就超过 65,504：总和是 Inf，每个概率都变成 0。`)}[o.id]||'';
    h=tx(`<p class="badline sm-mt"><b>Not yet: the largest error is ${ptsT(e.err)}, not below 0.1 points.</b> ${why}</p>`,`<p class="badline sm-mt"><b>还没达到：最大误差是 ${ptsT(e.err)}，没有低于 0.1 个百分点。</b>${why}</p>`)+
      tx('<p class="small">Hint: which fixes still give the probabilities of the exact softmax, and which of them stores 2 bytes per score?</p>','<p class="small">提示：哪些办法仍然给出精确 softmax 的概率？其中哪个每个得分只存 2 字节？</p>')+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(!chReveal) return h;
  const rows=[CH0].concat(CHO).map(chEval), lg=x=>Number.isNaN(x)?9.4:Math.max(0.25,Math.min(9.4,Math.log10(Math.max(x,1e-12))+7));
  h+=A.ui.cmpChart({title:tx('Largest error against the exact probabilities for the six scores, in percentage points (log scale; goal: below 0.1; B = bytes per stored score)','六个得分算出的概率和精确值的最大误差，单位是百分点（对数刻度，目标是低于 0.1，B 是每个存储得分的字节数）'),
    aria:tx('Largest error: ','最大误差：')+chShown().map(e=>`${chLabel(e.o)}${tx(': ','：')}${ptsT(e.err)}`).join(tx('; ','；')),brief:true,
    rows:rows.map(e=>({label:`${e.ok?'✓ ':''}${chLabel(e.o)}`,value:lg(e.err),text:fPP(e.err),ok:e.ok,me:e.o.id===chPick,
      tags:(e.o.id==='start'?[{text:tx('initial setup','初始设置')}]:[]).concat(e.o.id===chPick?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:6,goalLabel:tx('goal: below 0.1','目标：低于 0.1'),max:9.6,okText:tx('✓ meets the goal','✓ 达到目标'),noText:tx('misses it','没达到'),goalText:tx('goal: below 0.1','目标：低于 0.1')});
  h+=tx(`<p class="small">In the initial setup the naive softmax in FP16 returns NaN: ${eX('13.5')}, ${eX('13')} and ${eX('12.25')} are all above 65,504. Subtracting the max leaves the exact result unchanged, so the only error left is FP16’s rounding, ${ptsT(chEval(chOpt('max')).err)}. An FP32 copy of the scores is even closer but stores 4 bytes per score; halving and clamping change the probabilities themselves.</p>`,
        `<p class="small">初始设置下，FP16 的朴素 softmax 返回 NaN：${eX('13.5')}、${eX('13')} 和 ${eX('12.25')} 都超过 65,504。减去最大值不改变精确的结果，剩下的误差只有 FP16 的取整，${ptsT(chEval(chOpt('max')).err)}。把得分另存一份 FP32 更准，但每个得分要存 4 字节；减半和截断改变的是概率本身。</p>`);
  let s=0; for(let i=0;i<4096;i++) s=rnd('fp16',s+1);
  h+=tx(`<p class="small muted">Note: six scores are a short row. A row in a real model can have thousands, and an FP16 running sum loses digits as it grows: adding 1 to it 4,096 times stops at ${fN(s)}. That is why real GPU code keeps the max and the sum in FP32 even when it reads FP16 scores: the option “Subtract the max, with FP32 arithmetic”, still 2 bytes per stored score.</p>`,
        `<p class="small muted">注意：六个得分是很短的一行。真实模型里，一行可以有几千个得分，而 FP16 的累计和越大，丢的数字越多：给它加 4,096 次 1，结果停在 ${fN(s)}。所以即使读进来的是 FP16 得分，实际的 GPU 代码也把最大值和总和放在 FP32 里：就是“减去最大值，并用 FP32 计算”这个选项，每个存储的得分仍是 2 字节。</p>`);
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Predict first, then move the scores until the naive softmax breaks, try FP16, keep a loss finite, and compute softmax one block at a time. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再移动得分直到朴素 softmax 出错，试试 FP16，让 loss 不变成 Inf，最后一块一块地算 softmax。第一次学从这里开始。`),
  sandbox:tx('Four real cases, from a sampler at a low temperature to a padding row that is fully masked. Pick one to see where the naive math breaks.',
             '四个真实情况，从低 temperature 的采样器到整行都被 mask 的 padding。选一个，看朴素的算法在哪里出错。'),
  challenge:tx('A row of six FP16 scores, from 8 to 13.5, makes the naive softmax return NaN. Pick the smallest fix that brings every probability within 0.1 percentage points of the exact one.',
               '一行六个得分存成 FP16（8 到 13.5），朴素 softmax 算出 NaN。选出改动最小的办法，让每个概率和精确值相差不到 0.1 个百分点。')
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> this page really computes in FP32, BF16 and FP16. Every operation’s result is rounded to the chosen format, the way the hardware rounds it; ${J('math','“Show the math”')} lists the values.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>FP32 rounding is JavaScript’s <code>Math.fround</code>. FP16 and BF16 use an exact round-to-nearest-even: FP16 keeps 11 significant bits up to 65,504, BF16 keeps 8 bits, and its largest number is about FP32’s; larger results become Inf, and numbers below the smallest normal one keep fewer bits (subnormals).</li>`+
 `<li>The inputs, each exp, each addition of the sum (left to right), each subtraction, division and log are rounded. Exp and log are computed in float64 first and then rounded: close to what a correctly rounded library gives, though it can differ in the last bit.</li>`+
 `<li>The exact probabilities and attention weights are the same softmax in float64.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The scores are made up for this lab; the arithmetic on them is real.</li>`+
 `<li id="sm-fid-sum">Real kernels add in another order (trees of partial sums), and PyTorch adds FP16 and BF16 values in FP32 before it rounds the result. The order changes the last digits, not where exp overflows. A long FP16 running sum does lose digits: adding 1 to it 4,096 times stops at 2,048.</li>`+
 `<li>The window edges are where ${eX('z')} reaches the midpoint above the largest number (Inf) or falls to half the smallest subnormal (0); a library’s exp can differ in the last bit right at an edge.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Stable softmax and log_softmax: PyTorch ${cite(SRC.ptc,'<code>SoftMax.cpp</code>')}, ${cite(SRC.ptk,'<code>cpu/SoftMaxKernel.cpp</code>')} and ${cite(SRC.ptu,'<code>cuda/SoftMax.cu</code>')}.</li>`+
 `<li>Online softmax: ${cite(SRC.onl,'Milakov and Gimelshein, 2018')}; its use in FlashAttention, ${cite(SRC.fla,'Dao et al., 2022')} and ${cite(SRC.fa,'<code>softmax.h</code>')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>这一页真的用 FP32、BF16 和 FP16 计算。每一步运算的结果都按所选格式取整，和硬件的取整方式一样；${J('math','“展开公式”')}列出了各个值。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>FP32 的取整用 JavaScript 的 <code>Math.fround</code>。FP16 和 BF16 用精确的“就近取整，平局取偶”：FP16 保留 11 位有效二进制位，最大 65,504；BF16 保留 8 位，能存的最大数和 FP32 差不多。更大的结果变成 Inf，比最小正规数还小的数保留的位数更少（次正规数）。</li>`+
 `<li>输入、每次 exp、求和时的每次加法（从左到右）、每次减法、除法和 log 都取整。exp 和 log 先用 float64 算，再取整：和正确取整的数学库给出的结果很接近，但最后一位可能不同。</li>`+
 `<li>精确的概率和注意力权重，是同一个 softmax 的 float64 结果。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>这些得分是为这个 Lab 编的；用它们做的运算是真实的。</li>`+
 `<li id="sm-fid-sum">真实的 kernel 用别的顺序相加（分组求部分和），PyTorch 把 FP16 和 BF16 的值放在 FP32 里相加，最后才取整。顺序只影响最后几位，不影响 exp 在哪里溢出。很长的 FP16 累计和确实会丢数字：给它加 4,096 次 1，结果停在 2,048。</li>`+
 `<li>区间的边界是 ${eX('z')} 达到最大数和下一个 2 的幂之间的中点（变成 Inf），或降到最小次正规数的一半（变成 0）的地方；在边界上，数学库的 exp 可能在最后一位上不同。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>稳定的 softmax 和 log_softmax：PyTorch 的 ${cite(SRC.ptc,'<code>SoftMax.cpp</code>')}、${cite(SRC.ptk,'<code>cpu/SoftMaxKernel.cpp</code>')} 和 ${cite(SRC.ptu,'<code>cuda/SoftMax.cu</code>')}。</li>`+
 `<li>online softmax：${cite(SRC.onl,'Milakov 和 Gimelshein，2018')}；FlashAttention 里的用法见 ${cite(SRC.fla,'Dao 等，2022')} 和 ${cite(SRC.fa,'<code>softmax.h</code>')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A model gives every possible next token a score (LLMs call it a ${DEF('logit',TIP.logit())}). ${DEFK('Softmax','sm_sm')} turns the scores into probabilities that add up to 100%:</p>${FXL('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}<p>The labs ${labLink('architecture')} and ${labLink('sampling')} show where the scores come from and how a token is picked from the probabilities.</p>`,
                    `<p>模型给每个可能的下一个 token 打一个得分（LLM 里叫 ${DEF('logit',TIP.logit())}）。${DEFK('softmax','sm_sm')} 把这些得分变成加起来是 100% 的概率：</p>${FXL('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}<p>得分从哪来、怎样按概率选出 token，见${linkZ('architecture')}和${linkZ('sampling')}这两个 Lab。</p>`);
const OUTCOME=()=>tx('explain why a naive softmax returns NaN, and write a softmax, a cross-entropy and an online softmax that never overflow.','讲清朴素 softmax 为什么算出 NaN，并写出不会溢出的 softmax、cross-entropy 和 online softmax。');
const EXITS=()=>{
  const q1=naive([10.5,10.5],'fp16'), e105=EXP('fp16',10.5), sc=Math.exp(3-5), ns=2.5*sc;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why may a softmax subtract the largest score from every score before it exponentiates?','softmax 为什么可以在取 exp 之前，先让每个得分减去最大的那个？'),
   o:[tx('Subtracting m multiplies every exp, above and below the fraction line, by the same factor, and that factor cancels in the division, so every probability stays the same.','减去 m，等于把分子和分母里的每个 exp 都乘上同一个因子。相除时这个因子约掉了，所以每个概率都不变。'),
      tx('After the subtraction the largest score is 0 and drops out of the sum, so only the other scores are left, and none of them can overflow.','减完之后最大的得分变成 0，从总和里去掉了，只剩其他的得分，它们都不会溢出。'),
      tx('The subtraction changes the probabilities a little, but the change is smaller than FP32’s rounding, so no one can see it in the result.','减法会让概率变一点点，但变化比 FP32 的取整误差还小，所以结果里看不出来。')],a:0,
   n:[null,tx(`The largest score does not drop out: ${eX('0')} = 1, so it adds 1 to the sum and keeps the largest probability.`,`最大的得分并没有去掉：${eX('0')} = 1，它给总和加 1，仍然拿到最大的概率。`),
     tx(`In exact math the change is exactly zero, not small: the factor ${eX('−m')} cancels completely.`,`在精确的数学里，变化正好是 0，而不是很小：因子 ${eX('−m')} 被完全约掉。`)],
   why:tx(`Softmax only depends on the gaps between scores. Subtracting the max keeps every gap and puts the largest exponent at exactly 0, so every ${eX('z − m')} is at most 1.`,`softmax 只取决于得分之间的差距。减去最大值保留了每个差距，并让最大的指数正好是 0，所以每个 ${eX('z − m')} 都不超过 1。`)},
  {q:tx(`In online softmax, why is the running sum multiplied by ${math('e^{m_{old} - m_{new}}')} when a block raises the max?`,`online softmax 里，某一块让最大值变大时，为什么要把累计和乘以 ${math('e^{m_{\\text{旧}} - m_{\\text{新}}}')}？`),
   o:[tx('The terms already in the sum were measured against the old max. The factor measures each of them against the new max instead, the same reference as the new block’s terms.','总和里已有的项，是以旧的最大值为准算的。乘上这个因子，每一项都改成以新的最大值为准，和新一块的项用同一个基准。'),
      tx('It shrinks the old terms so that the sum stays below FP32’s largest number; without it the sum would overflow after a few blocks.','它把旧的项缩小，让总和不超过 FP32 能存的最大数；不乘的话，几块之后总和就会溢出。'),
      tx('It corrects the rounding of the earlier blocks: each block adds a small error, and the factor removes it before the next block.','它修正前面几块的取整：每一块都带来一点误差，这个因子在下一块之前把误差消掉。')],a:0,
   n:[null,tx(`Every term is at most 1 after its own max is subtracted, so the sum of a few blocks cannot overflow; without the factor the old terms would simply be too large by ${math('e^{m_{new} - m_{old}}')}.`,`每一项减去自己的最大值后都不超过 1，几块加起来不会溢出；不乘这个因子，旧的项只是大了 ${math('e^{m_{\\text{新}} - m_{\\text{旧}}}')} 倍。`),
     tx('The factor is exact math, not a rounding fix, so the result equals a max pass followed by a sum pass.','这个因子是精确的数学，不是在修正取整，所以结果和先找最大值、再求和的两遍算法一样。')],
   why:tx(`All terms of one sum must share one reference max. The factor moves the old terms to the new max, so after every block ${math('ℓ = \\sum e^{z - m}')} over the scores read so far.`,`同一个总和里的所有项必须用同一个最大值作基准。这个因子把旧的项换到新的最大值上，所以每读完一块，都有 ${math('ℓ = \\sum e^{z - m}')}，求和范围是已读的得分。`)}],
 transfer:[
  {q:tx(`In FP16, whose largest number is 65,504, a naive softmax gets two scores, both 10.5. Each ${eX('10.5')} is about ${fN(e105)}, finite. What does it return?`,`FP16 最大只能存 65,504。一个朴素 softmax 拿到两个得分，都是 10.5。每个 ${eX('10.5')} 约 ${fN(e105)}，是有限值。它返回什么？`),
   o:[tx('50% and 50%','50% 和 50%'),tx('0 and 0','0 和 0'),tx('NaN and NaN','NaN 和 NaN')],a:Number.isNaN(q1.p[0])?2:(q1.p[0]===0?1:0),
   why:tx(`Each exp is finite, but their sum, ${fN(e105)} + ${fN(e105)}, passes 65,504 and becomes Inf. Then each probability is ${fN(e105)} ÷ Inf = 0. The sum has to fit in the format too; subtracting the max (both become 0, ${eX('0')} = 1) gives 50% each.`,
          `每个 exp 都是有限值，但它们的和 ${fN(e105)} + ${fN(e105)} 超过 65,504，变成 Inf。于是每个概率是 ${fN(e105)} ÷ Inf = 0。总和也必须存得下；先减最大值（两个都变成 0，${eX('0')} = 1）就得到各 50%。`)},
  {q:tx('An online softmax has read some blocks: the running max m is 3 and the running sum ℓ is 2.5. The next block’s largest score is 5. Before the new terms are added, what happens to the running sum?','一个 online softmax 已经读了几块：累计最大值 m 是 3，累计和 ℓ 是 2.5。下一块里最大的得分是 5。加上新的项之前，累计和要怎样变？'),
   o:[tx('It stays: the old terms do not depend on the new max','不变：旧的项和新的最大值无关'),tx('It is multiplied by e², growing to about 18.5','乘以 e²，变大到约 18.5'),tx('It is multiplied by e⁻² (about 0.135), shrinking to about 0.34','乘以 e⁻²（约 0.135），缩小到约 0.34')],a:2,
   why:tx(`Every old term must now be measured against the new max, 5, so the whole running sum is multiplied once by:</p>${FXL(`e^{3 - 5} = e^{-2} \\approx ${fixd(sc,3)}`)}<p>That gives about ${fixd(ns,2)}. Then the new block’s terms are added.`,
          `旧的每一项都要改成以新的最大值 5 为准，所以整个累计和乘一次：</p>${FXL(`e^{3 - 5} = e^{-2} \\approx ${fixd(sc,3)}`)}<p>结果约是 ${fixd(ns,2)}。然后再加上新一块的各项。`)}],
 practice:[
  {id:'cross-entropy',d:tx('Part 1: a cross-entropy that stays finite and correct for scores as large as 1,000. Use log-sum-exp, the way this lab computes the loss.','第 1 部分：写一个得分大到 1,000 也有限且正确的 cross-entropy。像这个 Lab 算 loss 那样，用 log-sum-exp。')},
  {id:'streaming-entropy',d:tx('Parts 2 to 4: the entropy of a softmax, first with the max subtracted, then block by block, then in one pass over chunks with a running max, like this lab’s online softmax.','第 2 到 4 部分：softmax 的熵，先减最大值算，再分块算，最后带着累计最大值一遍读完所有块，和这个 Lab 里的 online softmax 一样。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Softmax & numerics','softmax 与数值'),title:l?L(l.q):tx('Why does a naive softmax return NaN?','朴素的 softmax 为什么会算出 NaN？'),
    minutes:10,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: scores and softmax','前置知识：得分和 softmax'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage sm-stage" id="sm-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="sm-st-title"></div><div class="meter-sub" id="sm-st-sub"></div></div><span class="status" id="sm-status" tabindex="0"></span></div>`+
   `<div class="legend-row" id="sm-legend"></div>`+
   `<div class="stage-graph graph sm-graph" id="sm-graph">`+
    `<div class="sm-phone" id="sm-phone"></div>`+
    `<div class="sm-nl" id="sm-nl"></div>`+
    `<div class="sm-tab" id="sm-tab"></div>`+
    `<div class="sm-ce" id="sm-ce" hidden></div>`+
    `<div class="sm-on" id="sm-on" hidden></div>`+
   `</div>`+
   `<p class="sm-scope small muted" id="sm-scope" tabindex="0" data-tipx="${esc(tx('The scores are made up for this lab. Every value computed from them is real FP32, BF16 or FP16 arithmetic, rounded after every operation.','这些得分是为这个 Lab 编的。由它们算出的每个值，都是真实的 FP32、BF16 或 FP16 运算，每一步都取整。'))}"></p>`+
   `<div class="metrics sm-metrics" id="sm-metrics"></div>`+
   `<details class="math" id="sm-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="sm-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="sm-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint small muted" id="sm-hint" hidden></p>`+
   `<p class="ctl-lock" id="sm-lock" hidden></p>`+
   `<div id="sm-ctl-top-wrap">${topCtl('sm-ctl-top',false)}</div>`+
   `<div id="sm-ctl-gap-wrap">${gapCtl('sm-ctl-gap',false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The right next token is “not”; the other five scores stay at 8, 5.5, 3, 1 and −0.5.','正确的下一个 token 是“not”；另外五个得分固定为 8、5.5、3、1 和 −0.5。')}</div></div>`)}</div>`+
   `<div id="sm-ctl-mv-wrap">${A.ui.seg({id:'sm-ctl-mv',key:'mv',label:tx('Masked scores set to','被 mask 的得分设成'),options:[['e9','−1e9'],['inf','−Inf']],value:S.mv,note:tx('Every token of this row is masked, as in a padding row.','这一行的每个 token 都被 mask，就像 padding 的一行。')})}</div>`+
   `<div id="sm-ctl-fmt-wrap">${fmtCtl('sm-ctl-fmt',false)}</div>`+
   `<div class="sm-grp" id="sm-ctl-z"><div class="sm-gh">${tx('Assumed values','假设值')}</div><div class="ctl"><span class="lbl" id="sm-zlab"></span><div class="ctl-note" id="sm-zlist"></div></div></div>`+
   `<div id="sm-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np
np.set_printoptions(suppress=True)

def softmax_naive(z):
    e = np.exp(z)                   # Inf above ~88.7 in float32
    return e / e.sum()

def softmax(z):
    e = np.exp(z - z.max())         # every exponent <= 0
    return e / e.sum()

def cross_entropy(z, y):            # log-sum-exp: never log(0)
    m = z.max()
    return m + np.log(np.exp(z - m).sum()) - z[y]

def softmax_online(blocks):         # m and l (the sum ℓ) in one pass
    m, l = np.float32(-np.inf), np.float32(0)
    for b in blocks:
        m_new = max(m, b.max())
        l = l * np.exp(m - m_new) + np.exp(b - m_new).sum()
        m = m_new
    return m, l                     # softmax = exp(z - m) / l

f32 = np.float32
z = np.array([92, 89, 87, 86, 84, 82], dtype=f32)
with np.errstate(over="ignore", invalid="ignore"):
    print(softmax_naive(z))
print(softmax(z).round(4))
zc = np.array([8, 5.5, 3, 1, -0.5, -112], dtype=f32)
print(cross_entropy(zc, 5))
m, l = softmax_online(np.split(z, 3))
print(np.abs(np.exp(z - m) / l - softmax(z)).max())
# [nan nan  0.  0.  0.  0.]
# [0.9439 0.047  0.0064 0.0023 0.0003 0.    ]
# 120.08612
# 0.0`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="sm-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="sm-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s three fixes in NumPy float32, checked against the naive softmax.','这个 Lab 的三个办法，用 NumPy float32 实现，并和朴素 softmax 对照。')}</p><pre class="code" id="sm-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="sm-prod">`+
   tx(`<li>PyTorch’s CUDA kernels in ${cite(SRC.ptu,'<code>aten/src/ATen/native/cuda/SoftMax.cu</code>')} find the row’s max, then write <code>exp(input - max_input) / sum</code> for softmax and <code>input - max_input - logsum</code> for log_softmax, with the max and the sum in a wider accumulator type (FP32 for FP16 inputs).</li>`+
      `<li>The CPU versions are in ${cite(SRC.ptc,'<code>aten/src/ATen/native/SoftMax.cpp</code>')} and ${cite(SRC.ptk,'<code>cpu/SoftMaxKernel.cpp</code>')}; <code>cross_entropy</code> is built on log_softmax, so it never takes the log of a 0 probability.</li>`+
      `<li>FlashAttention’s ${cite(SRC.fa,'<code>csrc/flash_attn/src/softmax.h</code>')} keeps <code>row_max</code> and <code>row_sum</code> in FP32; when a block raises the max it multiplies <code>row_sum</code> and the output so far by <code>exp2f((max_prev - max_cur) * scale)</code>, and a row whose max is −Inf subtracts 0 instead of −Inf.</li>`,
      `<li>PyTorch 的 CUDA kernel 在 ${cite(SRC.ptu,'<code>aten/src/ATen/native/cuda/SoftMax.cu</code>')} 里先求一行的最大值，softmax 写成 <code>exp(input - max_input) / sum</code>，log_softmax 写成 <code>input - max_input - logsum</code>；最大值和总和放在更宽的累加类型里（FP16 输入用 FP32）。</li>`+
      `<li>CPU 版本在 ${cite(SRC.ptc,'<code>aten/src/ATen/native/SoftMax.cpp</code>')} 和 ${cite(SRC.ptk,'<code>cpu/SoftMaxKernel.cpp</code>')}；<code>cross_entropy</code> 建立在 log_softmax 之上，所以从不对为 0 的概率取 log。</li>`+
      `<li>FlashAttention 的 ${cite(SRC.fa,'<code>csrc/flash_attn/src/softmax.h</code>')} 把 <code>row_max</code> 和 <code>row_sum</code> 放在 FP32 里；某一块让最大值变大时，它把 <code>row_sum</code> 和已算出的输出乘以 <code>exp2f((max_prev - max_cur) * scale)</code>；最大值是 −Inf 的一行改减 0，而不是 −Inf。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.onl,'Online normalizer calculation for softmax')}<span>${tx('Milakov and Gimelshein, 2018: one pass over the scores with a running max and a rescaled running sum, instead of a pass for the max and a pass for the sum','Milakov 和 Gimelshein，2018：带着累计最大值和缩放后的累计和，只读一遍得分，而不是先读一遍求最大值、再读一遍求和')}</span></li>`+
   `<li>${cite(SRC.fla,'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness')}<span>${tx('Dao, Fu, Ermon, Rudra and Ré, 2022: attention computed block by block with the same running max and sum, so the full row of scores is never stored','Dao、Fu、Ermon、Rudra、Ré，2022：用同样的累计最大值和总和一块一块地算 attention，整行得分从不完整存下来')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="sm-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="sm-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="sm-app m2" id="sm-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="sm-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see whether the naive softmax breaks there.','选一个场景，看看朴素 softmax 在那里会不会出错。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="sm-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function chLevers(){return `<div class="sm-lv" id="sm-ic-ch">`+A.ui.levers({aria:tx('The fix, pick one','办法，选一个'),value:chPick,options:CHO.map(o=>({v:o.id,title:L(o.t),desc:`${L(o.d)}${tx(`; ${o.bytes} bytes per stored score`,`；每个存储的得分 ${o.bytes} 字节`)}`,result:esc(chTried.has(o.id)?chRes(o):'')}))})+`</div>`;}
const chRes=o=>{const e=chEval(o); return e.ok?tx('meets the goal','达标'):(Number.isNaN(e.err)?'NaN':tx('misses','没达标'));};
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Stop an FP16 row of scores from giving NaN','让一行 FP16 得分的 softmax 不再算出 NaN')}</h3>`+
   `<p>${tx('A piece of GPU code stores a row of six scores in FP16, from 8 to 13.5, and computes the probabilities with the naive softmax. It returns NaN. Pick the smallest fix that brings every probability within 0.1 percentage points of the exact one. Try a fix and see how far the largest error drops.','一段 GPU 代码把一行六个得分存成 FP16（最小 8，最大 13.5），直接用朴素 softmax 算概率，结果是 NaN。请选出改动最小的办法，让每个概率和精确值相差不到 0.1 个百分点。选一个办法试试，看看最大误差降到多少。')}</p>`+
   chLevers()+
   `<div class="goal mono sm-goal" id="sm-ch-g1"><span>${tx('Largest error','最大误差')}</span><span><b id="sm-ch-err"></b>${tx(' (goal: below 0.1 pts)','（目标：低于 0.1 个百分点）')}</span></div>`+
   `<div class="goal mono sm-goal" id="sm-ch-g2"><span>${tx('Bytes per stored score','每个存储得分的字节数')}</span><span><b id="sm-ch-b"></b></span></div>`+
   submit+`<div id="sm-ch-res" class="sm-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},shift:{en:`Drag the largest score to a value between ${fZ(okR().lo)} and ${fZ(okR().hi)} to continue.`,zh:`把最大的得分拖到 ${fZ(okR().lo)} 到 ${fZ(okR().hi)} 之间才能继续。`},
    fmt:{en:'Pick FP16 to continue.',zh:'选 FP16 才能继续。'},ce:{en:`Drag the gap down to ${ceOK()} or less to continue.`,zh:`把差距拖到 ${ceOK()} 或更小才能继续。`},online:{en:'Read all four blocks and press ▸ once more to continue.',zh:'读完四块，再按一次 ▸ 才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#sm-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="sm-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="sm-why" hidden></p>');}
  A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#sm-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const moved={shift:S.top!==92,fmt:S.fmt!=='fp32'||!!seenF.fp16,ce:S.gap!==120||seenFin,online:onStep>0}[s.id];
    const show=!!s.obs&&(!s.expect||!!observed[step]||!!moved)&&(!s.gate||s.gate());
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.sm-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('.sm-onpos',app).forEach(e=>{e.textContent=onPos();});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#sm-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const o=chOpt(chPick), e=chEval(o), w=$('#sm-ch-err',rail), b=$('#sm-ch-b',rail); if(!w) return;
    w.textContent=ptsT(e.err); b.textContent=tx(`${o.bytes} (${fF(o.store)})`,`${o.bytes}（${fF(o.store)}）`);
    const g1=$('#sm-ch-g1',rail); g1.classList.toggle('met',e.ok); g1.classList.toggle('sm-over',!e.ok);
    $$('[data-lever]',rail).forEach(bt=>{const v=bt.dataset.lever; bt.setAttribute('aria-pressed',String(v===chPick)); const rr=bt.querySelector('.lv-r'); if(rr) rr.textContent=chTried.has(v)?chRes(chOpt(v)):'';});
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#sm-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; chChartFix(res); if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':(S.cs==='ce'?'ce':'sm'));
/* the axis of the number line: fixed per format, so the picture never rescales while the slider moves */
const AXIS={fp32:[-140,140],bf16:[-140,140],fp16:[-24,18]};
function axisPos(f,v){const [a,b]=AXIS[f]; if(v===-Infinity||v<a) return {x:0,off:-1}; if(v===Infinity||v>b) return {x:100,off:1}; return {x:(v-a)/(b-a)*100,off:0};}
const pc=x=>x.toFixed(3)+'%';
/* one row of dots on the number line */
function dotRow(f,vals,toks,kind,F){
  const allLow=vals.every(v=>v<F.lo);
  return vals.map((v,i)=>{const p=axisPos(f,v), over=v>F.hi, low=v<F.lo, bad=kind==='raw'&&(over||allLow||Number.isNaN(v)), hollow=low&&!bad;
    const ez=EXP(f,v), t=toks[i];
    const N=NOUN(), tip=kind==='raw'?tx(`“${t}”: ${N.z} ${fZ(v)}. In ${F.name}, its exp is ${fE(ez)}${over?', above the largest number':(low?', below the smallest number':'')}.`,`“${t}”：${N.z} ${fZ(v)}。在 ${F.name} 里，它的 exp 是 ${fE(ez)}${over?'，超过能存的最大数':(low?'，小于能存的最小数':'')}。`)
      :tx(`“${t}”: ${N.z} minus the max = ${fZ(v)}. In ${F.name}, its exp is ${fE(ez)}.`,`“${t}”：得分减去最大值 = ${fZ(v)}。在 ${F.name} 里，它的 exp 是 ${fE(ez)}。`);
    return `<i class="sm-dot${bad?' sm-bad':''}${hollow?' sm-hol':''}${p.off?' sm-off':''}" style="left:${pc(p.x)}" tabindex="0" data-tipx="${esc(tip)}"></i>`;}).join('');
}
function drawNL(r,c,f){
  const F=FMT[f], el=$('#sm-nl',app), [a,b]=AXIS[f], w0=axisPos(f,F.lo).x, w1=axisPos(f,F.hi).x, z0=axisPos(f,0).x;
  const st=stable(c.z,f), mx=Math.max(...c.z), mp=axisPos(f,mx);
  const N=NOUN(), cap=tx(`Where the ${N.zs} sit, and the safe window of exp in ${F.name}`,`得分在哪里，以及 ${F.name} 里 exp 的安全区间`);
  const lab=Number.isFinite(mx)?`${esc(c.toks[c.z.indexOf(mx)])} ${fZ(mx)}`:fN(mx);
  const html=`<div class="sm-cap">${esc(cap)}</div><div class="sm-line">`+
    `<div class="sm-band" style="left:${pc(w0)};width:${pc(w1-w0)}" tabindex="0" data-tipx="${esc(TIP.win())}"></div>`+
    `<div class="sm-zero" style="left:${pc(z0)}"></div>`+
    `<div class="sm-lrow" data-r="raw"><span class="sm-rl">${esc(N.zs)}</span>${dotRow(f,c.z,c.toks,'raw',F)}<em class="sm-mx${mp.x>80?' sm-l':(mp.x<20?' sm-r':'')}" style="left:${pc(mp.x)}">${lab}${mp.off>0?' ▸':(mp.off<0?' ◂':'')}</em></div>`+
    `<div class="sm-lrow" data-r="st"><span class="sm-rl">${tx('minus the max','减去最大值')}</span>${dotRow(f,st.d,c.toks,'st',F)}</div>`+
    `<div class="sm-ticks"><span class="sm-w" style="left:${pc(w0)}">${fN(F.lo,2)}</span><span class="sm-z" style="left:${pc(z0)}">0</span><span class="sm-w" style="left:${pc(w1)}">${fN(F.hi,2)}</span><span class="sm-e0">${fN(a)}</span><span class="sm-e1">${fN(b)}</span></div></div>`;
  if(el._h!==html){el.innerHTML=html; el._h=html;}
}
/* the table: one row per token; e^z, then two columns of chances */
function drawTab(r,c,f,ch){
  const el=$('#sm-tab',app), F=FMT[f];
  let colA, colB, hA, hB, sumL, ez, sA, sB;
  if(ch){const o=chOpt(chPick), e=chEval(o); colA=e.p; colB=e.ex; sA=e.s; sB=1; hA=tx('your fix','你的办法'); hB=tx('exact','精确'); ez=CHS.z.map(v=>fN(v,2)); sumL=tx(`sum used by your fix: ${fE(e.s)}`,`你的办法用到的总和：${fE(e.s)}`);}
  else {colA=r.n.p; colB=r.st.p; sA=r.n.s; sB=r.st.s; hA=tx('naive','朴素'); hB=tx('max subtracted','先减最大值'); ez=r.n.e.map(v=>fE(v)); sumL=tx(`sum of ${eX('z')} = ${fE(r.n.s)}; after subtracting the max, sum = ${fE(r.st.s,4)}`,`${eX('z')} 之和为 ${fE(r.n.s)}；减去最大值后，和为 ${fE(r.st.s,4)}`);}
  const cell=(p,i,which,sm)=>{const nan=Number.isNaN(p), w=nan?100:Math.max(0,Math.min(1,p))*100, why=sm===Infinity?'Inf ÷ Inf':(sm===0?'0 ÷ 0':'');
    const tip=nan?(why?tx(`“${c.toks[i]}”: NaN, because the division was ${why}.`,`“${c.toks[i]}”：NaN，因为这次除法是 ${why}。`):tx(`“${c.toks[i]}”: NaN, because a step before the division was already NaN (−Inf − (−Inf)).`,`“${c.toks[i]}”：NaN，因为除法之前的一步已经是 NaN（−Inf − (−Inf)）。`))
      :tx(`“${c.toks[i]}”, ${which}: ${fP(p)}.`,`“${c.toks[i]}”，${which}：${fP(p)}。`);
    return `<span class="sm-c${nan?' sm-nan':''}" tabindex="0" data-tipx="${esc(tip)}"><i style="width:${w.toFixed(2)}%"></i><b class="mono">${fP(p)}</b></span>`;};
  const head=`<div class="sm-tr sm-th"><span>${tx('token','token')}</span><span>${ch?tx('score','得分'):eX('z')}</span><span>${esc(hA)}</span><span>${esc(hB)}</span></div>`;
  const rows=c.toks.map((t,i)=>{const over=!ch&&r.n.e[i]===Infinity;
    return `<div class="sm-tr"><span class="sm-tk">${esc(t)}</span><span class="sm-ez mono${over?' sm-bad':''}">${ez[i]}</span>${cell(colA[i],i,hA,sA)}${cell(colB[i],i,hB,sB)}</div>`;}).join('');
  const N=NOUN(), key=`<div class="sm-key">${legIt('sm-sw-bar',N.p,tx(`Each bar is a token’s ${N.p}, from 0 to 100%.`,`每根条是一个 token 的${N.p}，从 0 到 100%。`))+legIt('sm-sw-nan',tx('NaN','NaN'),TIP.nan())}</div>`;
  const html=head+rows+key+`<div class="sm-sum small mono">${sumL}</div>`;
  if(el._h!==html){el.innerHTML=html; el._h=html;}
}
/* cross-entropy: three ways to the same loss */
function drawCE(r,c,f){
  const el=$('#sm-ce',app), k=ceAll(c.z,CE.y,f), MAX=160;
  const row=(lab,v,tip)=>{const inf=!Number.isFinite(v), w=inf?100:Math.max(0,Math.min(1,v/MAX))*100;
    return `<div class="sm-lr"><span class="sm-ll">${lab}</span><span class="sm-lb${inf?' sm-nan':''}" tabindex="0" data-tipx="${esc(tip)}"><i style="width:${w.toFixed(2)}%"></i><em class="sm-lx" style="left:${pc(Math.min(1,k.exact/MAX)*100)}"></em><b class="mono">${fN(v,2)}</b></span></div>`;};
  const html=`<div class="sm-cap">${esc(tx(`The loss for the right token “not”, three ways (bar scale 0 to ${MAX})`,`正确 token“not”的 loss，三种算法（条的刻度 0 到 ${MAX}）`))}</div>`+
    row(math('-\\log p')+tx(', p naive','，p 用朴素算法'),k.naive,tx(`The naive softmax gives “not” a probability p of ${fE(k.py)}, so ${math('-\\log p')} = ${fN(k.naive,2)}.`,`朴素 softmax 给“not”的概率 p 是 ${fE(k.py)}，所以 ${math('-\\log p')} = ${fN(k.naive,2)}。`))+
    row(math('-\\log p')+tx(', max subtracted first','，p 先减最大值再算'),k.soft,tx(`With the max subtracted, the probability p of “not” is ${fE(k.pyS)}, so ${math('-\\log p')} = ${fN(k.soft,2)}.`,`先减最大值时，“not”的概率 p 是 ${fE(k.pyS)}，所以 ${math('-\\log p')} = ${fN(k.soft,2)}。`))+
    row(tx('log-sum-exp','log-sum-exp'),k.lse,tx(`Log-sum-exp works inside the log and never computes the probability, so it never takes the log of 0. Its loss here: ${fN(k.lse,2)}.`,`log-sum-exp 在 log 里面算，不先算出概率，所以不会对 0 取 log。这样算出的 loss 是 ${fN(k.lse,2)}。`))+
    `<div class="sm-key">${legIt('sm-sw-inf',tx('Inf loss','loss 为 Inf'),tx('A red hatched bar is a loss of Inf.','红色斜纹的条表示 loss 是 Inf。'))+legIt('sm-sw-ex',tx('thin mark: exact loss','细竖线：精确的 loss'),TIP.exactLoss())}</div>`+
    `<div class="sm-sum small mono">${esc(tx(`exact loss: ${fN(k.exact,4)} (the thin mark on each bar)`,`精确的 loss：${fN(k.exact,4)}（每条上的细竖线）`))}</div>`;
  if(el._h!==html){el.innerHTML=html; el._h=html;}
}
/* online softmax, drawn (R6.27): the running sum s is a bar with one piece per score read, each of length e^(z − m) against the
   current max m. When a block raises the max, the old bar (dashed) sits above the rescaled one, so the old pieces visibly shrink
   once and nothing is read again; the learner never has to do the arithmetic */
const ONSC=1.5;   /* bar scale: the longest sum, before or after a rescale, is 1.32 */
function onBars(k){   /* k = blocks read (1..NB): the pieces now, and before the rescale when block k raised the max */
  const st=onNow().steps[k-1], n=k*ON.B, pc=(m,upto)=>ON.z.slice(0,upto).map((z,i)=>({i,tok:ON.toks[i],z,e:EXP('fp32',rnd('fp32',rnd('fp32',z)-m)),nw:i>=n-ON.B}));
  return {m:st.mNew,mOld:st.mOld,s:st.s2,s0:st.s0,sc:st.sc,raised:st.raised,first:st.first,now:pc(st.mNew,n),before:st.raised?pc(st.mOld,n-ON.B):null};
}
function onPos(){return onStep===0?tx('before block 1','第 1 块之前'):(onStep<=NB?tx(`block ${onStep} of ${NB}`,`第 ${onStep} / ${NB} 块`):tx('all blocks read, divided by ℓ','读完所有块，已除以 ℓ'));}
function drawOn(){
  const el=$('#sm-on',app), o=onNow(), done=onStep>NB, k=Math.min(onStep,NB), b=k?onBars(k):null;
  const mi=b?b.now.reduce((a,p)=>p.z>a.z?p:a).i:-1;
  const cells=ON.z.map((v,i)=>{const bk=Math.floor(i/ON.B)+1, st=bk<=onStep?'sm-rd':'sm-td', cur=bk===onStep&&!done;
    return `<span class="sm-oc ${st}${cur?' sm-cur':''}${i===mi?' sm-omx':''}${i%ON.B===0?' sm-bs':''}" tabindex="0" data-tipx="${esc(tx(`Score of “${ON.toks[i]}”: ${fN(v,1)}, in block ${bk}.${i===mi?' The largest score read so far: m.':''}`,`“${ON.toks[i]}”的得分：${fN(v,1)}，在第 ${bk} 块。${i===mi?'这是目前读到的最大得分 m。':''}`))}"><em>${esc(ON.toks[i])}</em><b class="mono">${fN(v,1)}</b><s>${i===mi?'m':''}</s></span>`;}).join('');
  const W=e=>(e/ONSC*100).toFixed(2)+'%';
  const bar=(pcs,m,old)=>`<span class="sm-sbar${old?' sm-sold':''}">${pcs.map(p=>`<i class="sm-pc${!old&&p.nw&&!done?' sm-pn':''}" style="width:${W(p.e)}" tabindex="0" data-tipx="${esc(p.z===m?tx(`“${p.tok}”: score ${fN(p.z,1)}, the max itself; its piece is exactly 1.`,`“${p.tok}”：得分 ${fN(p.z,1)}，就是最大值，这一段正好是 1。`):tx(`“${p.tok}”: score ${fN(p.z,1)}, ${fN(m-p.z,1)} below the max ${fN(m,1)}; its piece is ${fE(p.e,3)} long.`,`“${p.tok}”：得分 ${fN(p.z,1)}，比最大值 ${fN(m,1)} 低 ${fN(m-p.z,1)}，这一段长 ${fE(p.e,3)}。`))}">${p.e/ONSC>=0.3?`<em>${esc(p.tok)}</em>`:''}</i>`).join('')}</span>`;
  const row=(lab,pcs,m,s,old)=>`<div class="sm-srow${old?' sm-srow-old':''}"><span class="sm-sl">${lab}</span>${bar(pcs,m,old)}<b class="mono">ℓ = ${s===0?'0':fE(s,3)}</b></div>`;
  let sum;
  if(!b) sum=row(tx('m = −Inf','m = −Inf'),[],0,0,false);
  else{
    const now=row(`m = ${fN(b.m,1)}`,b.now,b.m,b.s,false);
    sum=b.raised&&!done?row(tx(`before: m = ${fN(b.mOld,1)}`,`之前：m = ${fN(b.mOld,1)}`),b.before,b.mOld,b.s0,true)+
        `<div class="sm-sres small">${tx(`↓ m rises to ${fN(b.m,1)}: every old piece × ${fE(b.sc,3)}, then the new block is added`,`↓ m 升到 ${fN(b.m,1)}：旧的每一段都 × ${fE(b.sc,3)}，再加上新的一块`)}</div>`+now:now;
  }
  const skey=`<div class="sm-key">${legIt('sm-sw-pn',tx('pieces added by this block','这一块加上的段'),tx('Each score read adds one piece to the running sum. Filled pieces come from the block just read; outlined ones from earlier blocks.','每读一个得分，累计和就多一段。实心的段是刚读的这一块加上的，空心的段来自之前读的块。'))+legIt('sm-sw-old',tx('the sum before m rose','m 变大之前的和'),tx('Dashed: the sum before the max rose. Every old piece is multiplied by the same factor once, and the pieces are not read again.','虚线：最大值变大之前的和。旧的每一段都乘上同一个因子，只乘一次，已读的得分不用重读。'))}</div>`;
  const sumbox=`<div class="sm-sumbox"><div class="sm-scap small">${tx(`The running sum ℓ: each score read adds one piece. The closer the score is to the largest score so far, m, the longer its piece; the max’s own piece is 1.`,`累计和 ℓ：每读一个得分，就加上一段。得分越接近目前的最大得分 m，这一段越长，最大值自己那段是 1。`)}</div>${skey}${sum}</div>`;
  const fin=done?`<div class="sm-cap">${esc(tx('The probabilities of the eight tokens: online, and with a max pass before the sum pass','八个 token 的概率：online 算法，以及先找最大值、再求和的两遍算法'))}</div><div class="sm-tr sm-or sm-th"><span>token</span><span>${esc(tx('online: m and ℓ in one pass','online：一遍得到 m 和 ℓ'))}</span><span>${esc(tx('max pass, then sum pass','先找最大值，再求和'))}</span></div>`+ON.toks.map((t,i)=>`<div class="sm-tr sm-or"><span class="sm-tk">${esc(t)}</span><span class="sm-c" tabindex="0" data-tipx="${esc(tx(`“${t}”: ${fP(o.p[i])} online.`,`“${t}”：online 算出 ${fP(o.p[i])}。`))}"><i style="width:${(o.p[i]*100).toFixed(2)}%"></i><b class="mono">${fP(o.p[i])}</b></span><span class="sm-c" tabindex="0" data-tipx="${esc(tx(`“${t}”: ${fP(o.two[i])} with a max pass, then a sum pass.`,`“${t}”：先找最大值、再求和，算出 ${fP(o.two[i])}。`))}"><i style="width:${(o.two[i]*100).toFixed(2)}%"></i><b class="mono">${fP(o.two[i])}</b></span></div>`).join('')+
      `<div class="sm-sum small mono">${esc(tx(`largest difference: ${o.diff===0?'0':fE(o.diff)}`,`最大差值：${o.diff===0?'0':fE(o.diff)}`))}</div>`
    :`<p class="small muted sm-onwait">${esc(tx('The probabilities appear once every block is read and the sum is final.','读完所有块、总和确定以后，才能算出每个概率。'))}</p>`;
  const html=`<div class="sm-cap">${esc(tx(`Eight scores in FP32, read two at a time (${onPos()})`,`八个 FP32 得分，每次读两个（${onPos()}）`))}</div><div class="sm-cells">${cells}</div>${sumbox}${fin}`;
  if(el._h!==html){el.innerHTML=html; el._h=html;}
}
function draw(){
  const v=viewNow(), r=compute(), c=r.c, f=mode==='challenge'?chOpt(chPick).f:S.fmt;
  const nl=$('#sm-nl',app), tab=$('#sm-tab',app), ce=$('#sm-ce',app), on=$('#sm-on',app);
  nl.hidden=v==='on'; tab.hidden=!(v==='sm'||v==='ch'); ce.hidden=v!=='ce'; on.hidden=v!=='on';
  if(v==='on'){drawOn(); return;}
  if(v==='ch'){drawNL(r,{toks:CHS.toks,z:CHS.z.map(chOpt(chPick).xf||(x=>x))},f); drawTab(r,c,f,true); return;}
  drawNL(r,c,S.fmt);
  if(v==='ce') drawCE(r,c,S.fmt); else drawTab(r,c,S.fmt,false);
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chOpt(chPick)); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):(Number.isNaN(e.err)?'NaN':tx('Not yet','还没达到')),
    tip:tx(`Green when the largest error against the exact probabilities is below 0.1 percentage points. Now: ${ptsT(e.err)}.`,`和精确概率的最大误差低于 0.1 个百分点时，徽章就是绿色。现在是 ${ptsT(e.err)}。`)};}
  if(v==='on'){const done=onStep>NB; return {cls:'neutral',txt:done?tx('Same result','结果相同'):onPos(),tip:tx('How far the online softmax has read: four blocks of two scores, then the final division.','online softmax 读到了哪里：四块，每块两个得分，最后做一次除法。')};}
  const r=compute();
  if(v==='ce'){const k=ceAll(r.c.z,CE.y,S.fmt), bad=!Number.isFinite(k.naive); return {cls:bad?'bad':'neutral',txt:bad?tx('Naive loss Inf','朴素 loss 为 Inf'):tx('All finite','都是有限值'),
    tip:tx('Red when the naive loss is Inf: the right token’s probability rounded to 0 before its log was taken.','朴素的 loss 为 Inf 时是红色：正确 token 的概率在取 log 之前就取整成了 0。')};}
  const nan=hasNaN(r.n.p), zero=!nan&&r.n.p.every(p=>p===0), off=!zero&&naiveOff(r), N=NOUN();
  return {cls:nan||zero||off?'bad':'neutral',txt:nan?tx('Naive: NaN','朴素：NaN'):(zero?tx('Naive: all 0','朴素：全是 0'):(off?tx(`Naive: ${ptsT(r.errN)} off`,`朴素：差 ${ptsT(r.errN)}`):tx('Naive works','朴素能算'))),
    tip:tx(`The naive softmax’s result in ${fF(S.fmt)}: red when it has a NaN, when every ${N.p} is 0, or when its error reaches ${OKE} points while the error with the max subtracted stays far smaller.`,`朴素 softmax 在 ${fF(S.fmt)} 里的结果：有 NaN、每个${N.p}都是 0，或者它的误差达到 ${OKE} 个百分点、而先减最大值的误差小得多时，是红色。`)};
}
function drawHead(){
  const v=viewNow(), t=$('#sm-st-title',app), sub=$('#sm-st-sub',app), st=$('#sm-status',app), F=FMT[S.fmt];
  const title=v==='ch'?tx('The challenge row: six FP16 scores','挑战题的一行：六个 FP16 得分'):(v==='on'?tx('Online softmax: the max and the sum in one pass over four blocks','online softmax：读一遍四块，得到最大值和总和'):(v==='ce'?tx('Cross-entropy for a right token far below the top','正确 token 远低于最大值时的 cross-entropy'):
    (S.cs==='attn'?tx('Softmax of six attention scores','六个 attention 得分的 softmax'):(S.cs==='mask'?tx('Softmax of a fully masked row','整行被 mask 时的 softmax'):tx(`Softmax after “${LM.prompt}”`,`“${LM.prompt}”之后的 softmax`)))));
  t.textContent=title;
  sub.textContent=v==='ch'?tx(`your fix: ${L(chOpt(chPick).t)}`,`你的办法：${L(chOpt(chPick).t)}`):(v==='on'?tx('FP32, blocks of 2 scores','FP32，每块 2 个得分'):tx(`${F.name}: exp is finite from ${fN(F.lo,2)} to ${fN(F.hi,2)}`,`${F.name}：exp 在 ${fN(F.lo,2)} 到 ${fN(F.hi,2)} 之间是有限值`));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
/* one legend item; the top row keys the picture right under it, every other key sits right under the bars it labels (R5.9) */
const legIt=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
function legendHTML(){
  const it=legIt, v=viewNow(), N=NOUN(), at=attnNow();
  if(v==='on') return it('sm-sw-cur',tx('block being read','正在读的块'),tx('The block the online softmax read last; solid cells are read, dashed ones are still to come.','online softmax 刚读的那一块；实线格子是读过的，虚线格子还没读到。'))+
    it('sm-sw-omx','m',tx('The cell marked m holds the largest score read so far.','标着 m 的格子是目前读到的最大得分。'));
  const hurt=v==='sm'&&naiveOff(compute()), zz='得分';
  const nl=it('sm-sw-band',tx('safe window of exp','exp 的安全区间'),TIP.win())+it('sm-sw-dot',tx(`a ${N.z}`,at?'一个得分':'一个得分'),tx(`Each dot is one ${N.z}; the upper row as given, the lower row minus the largest one.`,`每个点是一个${zz.trimEnd()}；上面一行是原样的，下面一行减去了最大的那个。`))+
    it('sm-sw-bad',tx('exp overflows or all underflow','exp 溢出或全部下溢'),tx(`A red dot is a ${N.z} whose exp is Inf, or every ${N.z} when all of their exps round to 0.`,`红点表示这个${zz}的 exp 是 Inf；所有 exp 都取整成 0 时，每个点都是红的。`))+
    (hurt?it('sm-sw-hol',tx('rounds to 0','取整成 0'),tx(`A hollow dot is a ${N.z} whose exp rounds to 0. Here that is not harmless: so many round to 0 that the naive ${N.ps} are off, by the amount on the error card.`,`空心点表示这个${zz}的 exp 取整成 0。这里并非无害：取整成 0 的太多，朴素的${N.ps}已经不准，差多少见误差卡片。`))
      :it('sm-sw-hol',tx('rounds to 0, harmless','取整成 0，无害'),tx(`A hollow dot is a ${N.z} whose exp rounds to 0 while a larger one is fine: its ${N.p} is simply 0.`,`空心点表示这个${zz}的 exp 取整成 0，而更大的${zz}没问题：它的${N.p}就是 0。`)));
  return nl;   /* the bars' keys sit right under the bars: drawTab, drawCE, drawOn (R5.9) */
}
/* metric cards: ▲▼ only after the learner's own change, from the values as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').replace(/−/g,'-').match(/^[<>]?(-?\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>−-]?\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=viewNow(), N=NOUN();
  if(v==='ch'){const o=chOpt(chPick), e=chEval(o);
    return [{k:'err',label:tx('Largest error','最大误差'),v:ptsT(e.err),sub:tx('goal: below 0.1 pts','目标：低于 0.1 个百分点'),tip:tx(`The largest gap between your fix’s probabilities and the exact ones: ${ptsT(e.err)}. `,`你的办法算出的概率和精确值之间最大的差：${ptsT(e.err)}。`)+TIP.pts()},
      {k:'bytes',label:tx('Bytes per stored score','每个存储得分的字节数'),v:String(o.bytes),sub:fF(o.store),tip:o.store===o.f?tx(`Your fix stores each score in ${fF(o.store)}, ${o.bytes} bytes.`,`你的办法把每个得分存成 ${fF(o.store)}，${o.bytes} 字节。`):tx(`Your fix stores each score in ${fF(o.store)}, ${o.bytes} bytes, and does the arithmetic in ${fF(o.f)}.`,`你的办法把每个得分存成 ${fF(o.store)}，${o.bytes} 字节，运算用 ${fF(o.f)}。`)},
      {k:'tw',label:tx('Top probability','最大的概率'),v:fP(e.p[3]),sub:tx(`exact ${fP(e.ex[3])}`,`精确值 ${fP(e.ex[3])}`),tip:tx(`The probability of “time”, the largest score, with your fix.`,`用你的办法算出的“time”（得分最大）的概率。`)}];}
  if(v==='on'){const o=onNow(), s=onStep>0?o.steps[Math.min(onStep,NB)-1]:null;
    return [{k:'m',label:tx('Running max m','累计最大值 m'),v:s?fN(s.mNew,1):MINUS+'Inf',sub:onPos(),tip:tx('The largest score read so far.','到目前为止读到的最大得分。')},
      {k:'s',label:tx('Running sum ℓ','累计和 ℓ'),v:s?fE(s.s2,4):'0',sub:tx('sum of ','')+eX('z − m')+tx('',' 之和'),tip:tx(`The sum of ${eX('z − m')} over every score read so far, with m the running max.`,`已读的每个得分的 ${eX('z − m')} 之和，m 是累计最大值。`)},
      {k:'diff',label:tx('Online vs two passes','online 和两遍算法'),v:onStep>NB?(o.diff===0?'0':fE(o.diff)):'…',sub:tx('largest difference','最大差值'),tip:onStep>NB?tx('The largest difference between the probabilities from online softmax and from two passes.','online 算出的概率和两遍算法算出的概率，最多差多少。'):tx('Known once every block is read.','读完所有块之后才知道。')}];}
  const r=compute();
  if(v==='ce'){const k=ceAll(r.c.z,CE.y,S.fmt);
    return [{k:'gap',label:tx('Right token below the top','正确 token 低于最大值'),v:fN(S.gap),sub:tx('score gap','得分差距'),tip:tx(`“not” has score ${fN(r.c.z[CE.y],1)}, ${fN(S.gap)} below the top score 8.`,`“not”的得分是 ${fN(r.c.z[CE.y],1)}，比最大的得分 8 低 ${fN(S.gap)}。`)},
      {k:'naive',label:tx('Naive loss','朴素的 loss'),v:fN(k.naive,2),sub:math('-\\log p'),tip:tx(`${math('-\\log p')} for the naive probability p of “not”: ${fN(k.naive,2)}.`,`“not”的朴素概率记作 p，${math('-\\log p')} = ${fN(k.naive,2)}。`)},
      {k:'lse',label:tx('Log-sum-exp loss','log-sum-exp 的 loss'),v:fN(k.lse,2),sub:tx(`exact ${fN(k.exact,2)}`,`精确值 ${fN(k.exact,2)}`),tip:TIP.lse()}];}
  const nanN=hasNaN(r.n.p), zz='得分';
  const top={k:'top',label:N.top,v:fZ(Math.max(...r.c.z)),sub:tx(`window ends at ${fN(r.F.hi,2)}`,`安全区间到 ${fN(r.F.hi,2)}`),tip:tx(`The largest ${N.z} on the stage. Its exp overflows ${r.F.name} above ${fN(r.F.hi,2)}.`,`图里最大的${zz.trimEnd()}。它的 exp 在 ${r.F.name} 里超过 ${fN(r.F.hi,2)} 就溢出。`)},
    sum={k:'sum',label:tx(`Sum of ${eX('z')}, naive`,`朴素的 ${eX('z')} 之和`),v:fE(r.n.s),sub:r.F.name,tip:tx(`The naive softmax adds the exp of every ${N.z} in ${r.F.name}: ${fE(r.n.s)}.`,`朴素 softmax 在 ${r.F.name} 里把每个${zz}的 exp 加起来：${fE(r.n.s)}。`)};
  if(early()) return [top,sum,
    {k:'pn',label:tx('Naive: “Paris”','朴素：“Paris”'),v:fP(r.n.p[0]),sub:tx('its probability','它的概率'),tip:tx(`The naive softmax’s probability for “Paris”: ${fP(r.n.p[0])}.`,`朴素 softmax 给“Paris”的概率：${fP(r.n.p[0])}。`)},
    {k:'ps',label:tx('Max subtracted: “Paris”','先减最大值：“Paris”'),v:fP(r.st.p[0]),sub:tx('its probability','它的概率'),tip:tx(`The probability for “Paris” with the largest score subtracted first: ${fP(r.st.p[0])}.`,`先减去最大的得分时“Paris”的概率：${fP(r.st.p[0])}。`)}];
  return [top,sum,
    {k:'en',label:tx('Naive: largest error','朴素：最大误差'),v:ptsT(r.errN),sub:tx(`vs the exact ${N.ps}`,`和精确${N.ps}比`),tip:nanN?tx('The naive result has a NaN, so it has no error to measure: it is not a result at all.','朴素的结果里有 NaN，没有误差可言：它根本不是结果。'):tx(`The largest gap between the naive ${N.ps} and the exact ones: ${ptsT(r.errN)}. `,`朴素${N.ps}和精确${N.ps}之间最大的差：${ptsT(r.errN)}。`)+TIP.pts()},
    {k:'es',label:tx('Max subtracted: largest error','先减最大值：最大误差'),v:ptsT(r.errS),sub:tx(`vs the exact ${N.ps}`,`和精确${N.ps}比`),tip:tx(`The largest gap between the ${N.ps} with the max subtracted and the exact ones: ${ptsT(r.errS)}. `,`先减最大值算出的${N.ps}和精确${N.ps}之间最大的差：${ptsT(r.errS)}。`)+(!nanN&&r.errN<r.errS?tx('Here the naive one is a little closer: subtracting the max fixes the range, not the rounding. ','这里朴素的反而更接近一点：减去最大值解决的是范围，不是取整。'):'')+TIP.pts()}];
}
/* logits and gaps change by a difference (a percent of a logit means nothing); chances by percentage points; errors by their
   change in points, also across powers of ten; no badge when the data set changes, since the two values belong to different rows (R6.6) */
const ABS={top:1,gap:1,m:1}, PTS={tw:1,pn:1,ps:1}, ERR={en:1,es:1,err:1};
const SUPD={'⁻':'-','⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'};
const numX=v=>{const m=String(v).replace(/,/g,'').replace(/−/g,'-').match(/^(-?\d+(?:\.\d+)?)(?: × 10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?/); if(!m) return NaN; let x=parseFloat(m[1]); if(m[2]) x*=10**parseInt(m[2].split('').map(c=>SUPD[c]).join(''),10); return x;};
function drawMetrics(){
  const host=$('#sm-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join(), ds=viewNow()+'|'+(mode==='challenge'?'ch':S.cs);
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  if(host._ds!==ds){host._ds=ds; defs.forEach(m=>{shownPrev[m.k]=null;});}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').innerHTML=m.label; el.querySelector('.v').textContent=m.v; {const sb=el.querySelector('.sub'); if(/<math/.test(m.sub)) sb.innerHTML=m.sub; else sb.textContent=m.sub;} el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&!/Inf|NaN|…|<|>/.test(prev+m.v)&&(ERR[m.k]||!/×/.test(prev+m.v))){
      if(ERR[m.k]){const a=numX(prev), b=numX(m.v); if(isFinite(a)&&isFinite(b)&&a!==b) dt=`${b>a?'▲ +':'▼ −'}${fPP(Math.abs(b-a))}${tx(' pts vs last',' 个百分点（比上次）')}`;}
      else {const a=numOf(prev), b=numOf(m.v), dp=Math.max(decOf(prev),decOf(m.v));
        if(isFinite(a)&&isFinite(b)&&a!==b){if(ABS[m.k]) dt=`${b>a?'▲':'▼'} ${fN(Math.abs(b-a),dp)}${tx(' vs last','（比上次）')}`; else if(PTS[m.k]) dt=A.fmt.delta(a,b,{dp:1,pts:true})||''; else if(a>=0&&b>=0) dt=A.fmt.delta(a,b,{dp})||'';}}}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each block is one lead line in
   words, the quantity's name and its formula, the reader's numbers under it, then the result. The rows form a grid, so each number
   sits under the term it replaces and the “=” signs line up (R6.29). A cell is TeX, or {h:html}; inputs keep their terms (R1.4) */
/* the last cell of a short row spans the rest (not a fixed column), so the rows still line up when a label moves above them (a grid
   marked .nm puts its labels on lines of their own when the stage is narrow) */
const SMG=rows=>{const n=Math.max(...rows.map(r=>r.length));
  return `<div class="sm-mg${rows.nm?' sm-nm':''}" style="--n:${n};--n1:${n-1}">`+rows.map(r=>r.map((c,i)=>{const w=i===r.length-1&&r.length<n;
    return `<span class="sm-mc${i===0?' sm-mn':''}${w?' sm-mw':''}${c&&c.h!=null?' sm-mh':''}"${w?` style="grid-column:span ${n-i}"`:''}>${c==null||c===''?'':(c.h!=null?c.h:math(c))}</span>`;}).join('')).join('')+'</div>';};
const SMB=(lead,...grids)=>`<div class="sm-mb"><p class="sm-ml">${lead}</p>${grids.map(SMG).join('')}</div>`;
const TX=x=>'\\text{'+String(x)+'}';
/* a number as TeX: the minus sign and Inf / NaN stay readable */
const TN=x=>{const s=String(x); return /^[−-]?[0-9.,]+$/.test(s)?s.replace('−','-'):TX(s);};
function mathHTML(){
  const v=viewNow(), t=T_, N=NOUN();
  if(v==='on'){const o=onNow();
    return SMB(tx('Online softmax reads the scores one block at a time. For each block it updates the running max m and the running sum ℓ; when the max grows, the sum so far is rescaled to the new max first:','online softmax 一块一块地读得分。每读一块，就更新累计最大值 m 和累计和 ℓ；最大值变大时，先把已有的累计和换成以新的最大值为准：'),
        [['m_{\\text{new}}','=',`\\max\\left(m,\\ \\max_{\\text{${tx('block','这一块')}}} z\\right)`],['ℓ','\\leftarrow',`ℓ \\cdot e^{m - m_{\\text{new}}} + \\sum_{\\text{${tx('block','这一块')}}} e^{z - m_{\\text{new}}}`]])+
      /* each block on two rows, m and then ℓ, so a block's numbers fit the box without scrolling sideways (the owner's review of
         2026-10-09, R6.37); the block's label goes above them when the stage is narrow */
      SMB(tx('The four blocks, in the order they are read:','四块依次读进来时的数：'),
        Object.assign(o.steps.flatMap((s,i)=>[[{h:`<b>${tx(`Block ${i+1}`,`第 ${i+1} 块`)}</b>`},{h:'m'},'=',{h:`${s.first?MINUS+'Inf':fN(s.mOld,1)} → ${fN(s.mNew,1)}`}],
          ['',{h:'ℓ'},'=',{h:`${s.first?'0':`${fE(s.s0,4)} × ${fE(s.sc,4)}`} + ${s.terms.map(x=>fE(x,4)).join(' + ')}`},'=',{h:fE(s.s2,4)}]]),{nm:1}))+
      SMB(tx(`At the end, each ${N.p} is its own term divided by ℓ:`,`最后，每个${N.p}都是自己那一项除以 ℓ：`),
        [['p','=','\\frac{e^{z - m}}{ℓ}'],['','=',`\\frac{e^{z - ${TN(fN(o.m,1))}}}{${TN(fE(o.s,4))}}`]]);}
  const r=compute(), c=v==='ch'?{toks:CHS.toks,z:CHS.z}:r.c, f=v==='ch'?chOpt(chPick).f:S.fmt, F=FMT[f];
  const n=naive(c.z,f), st=stable(c.z,f), top=c.z.indexOf(Math.max(...c.z)), tk=`\\text{“${c.toks[top]}”}`;
  let h=SMB(tx(`Softmax turns the scores z into ${N.ps}; here `,`softmax 把得分 z 变成${N.ps}；这里`)+t('fmt',tx(`every step is rounded to ${F.name}`,`每一步都按 ${F.name} 取整`))+tx('.','。')+(early()?'':tx(' Subtracting the largest score m from every score gives the same result:','每个得分都先减去最大的得分 m，结果不变：')),
    early()?[[{h:t('fx',tx('softmax','softmax'))},'p_i','=','\\frac{e^{z_i}}{\\sum_j e^{z_j}}']]:[[{h:t('fx',tx('softmax','softmax'))},'p_i','=','\\frac{e^{z_i}}{\\sum_j e^{z_j}}'],['','','=','\\frac{e^{z_i - m}}{\\sum_j e^{z_j - m}}']]);
  h+=SMB(tx(`The naive softmax takes each exp as it is. For the largest score, ${fZ(n.zz[top])}:`,`朴素算法直接算每个 exp。以最大的得分 ${fZ(n.zz[top])} 为例：`),
    [['e^{z}','=',`e^{${TN(fZ(n.zz[top]))}}`,'=',TN(fE(n.e[top],4))],[TX(tx('sum','总和')),'=','\\sum_j e^{z_j}','=',TN(fE(n.s,4))],
     [`p_{${tk}}`,'=',`\\frac{${TN(fE(n.e[top],4))}}{${TN(fE(n.s,4))}}`,'=',TX(fP(n.p[top]))]]);
  if(!early()) h+=SMB(tx(`With the max subtracted, m = ${fZ(st.m)}, the largest exponent is 0:`,`先减去最大的得分 m = ${fZ(st.m)}，最大的指数就是 0：`),
    [['e^{z - m}','=',`e^{${TN(fZ(st.d[top]))}}`,'=',TN(fE(st.e[top],4))],[TX(tx('sum','总和')),'=','\\sum_j e^{z_j - m}','=',TN(fE(st.s,4))],
     [`p_{${tk}}`,'=',`\\frac{${TN(fE(st.e[top],4))}}{${TN(fE(st.s,4))}}`,'=',TX(fP(st.p[top]))]]);
  h+=SMB(tx(`Where ${F.name} runs out. Its largest number is ${fE(F.max,4)}, so ${eX('z')} is Inf above the log of it. Its smallest number above 0 is ${fE(F.sub,3)}, so ${eX('z')} rounds to 0 below about ${fN(F.lo,2)}.`,`${F.name} 的范围在哪里用完：能存的最大数是 ${fE(F.max,4)}，z 超过它的对数时 ${eX('z')} 是 Inf。大于 0 的最小数是 ${fE(F.sub,3)}，z 低于约 ${fN(F.lo,2)} 时 ${eX('z')} 取整成 0。`)+sumFit(v,c,F),
    [[TX(tx('largest z','z 的上限')),'=','\\log\\left('+TX(tx('largest number','最大数'))+'\\right)'],['','=',`\\log(${TN(fE(F.max,4))})`],['','\\approx',TN(fN(F.hi,2))]]);
  if(v==='ce'){const k=ceAll(c.z,CE.y,f);
    h+=SMB(tx('Log-sum-exp gets the loss without computing the probability first. m is the largest score and z_y the score of “not”:','log-sum-exp 不先算概率，直接算出 loss。m 是最大的得分，z_y 是“not”的得分：'),
      [[TX('loss'),'=','m','+','\\log \\sum_j e^{z_j - m}','-','z_y'],['','=',TN(fN(k.m,1)),'+',TN(fE(Math.log(k.s),4)),'-',`(${TN(fN(c.z[CE.y],1))})`],['','=',TN(fN(k.lse,4))]]);}
  return h;
}
/* B4: the sum must fit too. From this largest value on, the naive sum is Inf although every exp is finite (half steps never land there) */
const fitTop=(z,F)=>{const mx=Math.max(...z); return F.hi-Math.log(z.reduce((a,v)=>a+Math.exp(v-mx),0));};
function sumFit(v,c,F){
  if(v!=='sm'||early()||!(S.cs==='lm'||S.cs==='attn')) return '';
  const t=fitTop(c.z,F), N=NOUN();
  return tx(` The sum must fit too: with these gaps it is Inf from a largest ${N.z} of about ${fN(t,2)}, before any single exp overflows.`,`总和也必须存得下：按这些差距，最大的得分到约 ${fN(t,2)} 时，总和就已经是 Inf，这时单个 exp 还没有溢出。`);
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  if(isP('shift')){const r=compute(); if(!hasNaN(r.n.p)&&r.errN<OKE) seenOK=true; if(S.top<FMT.fp32.lo) seenLow=true;}
  if(isP('fmt')) seenF[S.fmt]=true;
  if(isP('ce')){const k=ceAll(caseNow().z,CE.y,S.fmt); if(Number.isFinite(k.naive)) seenFin=true;}
  drawHead();
  const gr=$('#sm-graph',app); gr.dataset.view=viewNow();
  draw(); drawMetrics();
  const lg=$('#sm-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#sm-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* Settings (R4.2): in each guided part only that part's control is live; the challenge changes only the fix, in the card */
function lockNote(){
  if(mode==='challenge') return esc(tx('In the challenge you pick only the fix, in the task card. Everything else is fixed: the six FP16 scores from 8 to 13.5.','挑战中只能在任务卡里选办法。其余固定：六个 FP16 得分，从 8 到 13.5。'));
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='shift') return esc(tx('Fixed in this part: FP32 and the six scores’ gaps. You move the largest score.','这一部分固定：FP32，以及六个得分之间的差距。你移动最大的得分。'));
  if(id==='fmt') return esc(tx('Fixed in this part: the six attention scores, up to 13.5. You pick the format.','这一部分固定：六个 attention 得分，最高 13.5。你选格式。'));
  if(id==='ce') return esc(tx('Fixed in this part: FP32 and the five other scores. You move the right token.','这一部分固定：FP32 和另外五个得分。你移动正确的 token。'));
  if(id==='online') return esc(tx('Fixed in this part: eight FP32 scores in four blocks of two. You step through the blocks in the card.','这一部分固定：八个 FP32 得分，分成四块，每块两个。你在卡片里逐块推进。'));
  const sc=`<button type="button" class="linkbtn" data-mode-go="sandbox">${tx('Scenarios','应用场景')}</button>`;
  const what=id==='remember'&&remView==='ce'?tx('FP32, with the right token “not” 120 below the top score','FP32，正确 token“not”比最大的得分低 120')
    :(id==='remember'&&remView==='on'?tx('eight FP32 scores, all four blocks read','八个 FP32 得分，四块都已读完'):tx('FP32, with scores up to 92','FP32，得分最高 92'));
  return tx(`Fixed in this part: ${esc(what)}. You can change the settings in ${sc}.`,`这一部分固定：${esc(what)}。在${sc}里可以改这些设置。`);
}
/* Settings → Assumed values: what each data set is, and why these values (R2.7) */
function zNote(){
  const v=viewNow(), list=(toks,z)=>toks.map((t,i)=>`“${t}” ${fZ(z[i])}`).join(tx(', ','，'));
  if(v==='on') return tx(`Made up for this lab: eight scores, ${list(ON.toks,ON.z)}, read in four blocks of two.`,`为这个 Lab 编的：八个得分，${list(ON.toks,ON.z)}，分成四块，每块两个。`);
  if(v==='ch') return tx(`Made up for this lab: one row of six scores, ${list(CHS.toks,CHS.z)}, stored in FP16.`,`为这个 Lab 编的：一行六个得分，${list(CHS.toks,CHS.z)}，存成 FP16。`);
  const c=caseNow();
  if(c.cs==='mask') return tx(`Made up for this lab: a padding row, the scores of all six tokens masked, each set to ${fZ(MV[S.mv])}.`,`为这个 Lab 编的：一行 padding，六个 token 的得分都被 mask，每个都设成 ${fZ(MV[S.mv])}。`);
  if(c.cs==='attn') return tx(`Made up for this lab: one query’s attention scores for six earlier tokens, ${list(c.toks,c.z)}. One token, “cat”, matches clearly.`,`为这个 Lab 编的：一个 query 对前面六个 token 的 attention 得分，${list(c.toks,c.z)}。其中“cat”明显最匹配。`);
  if(c.cs==='ce') return tx(`Made up for this lab: ${list(c.toks,c.z)}. The right next token is “not”, ${fN(S.gap)} below the top score.`,`为这个 Lab 编的：${list(c.toks,c.z)}。正确的下一个 token 是“not”，比最大的得分低 ${fN(S.gap)}。`);
  return tx(`Made up for this lab: ${list(c.toks,c.z)}. A real model computes them from the input; “Paris” leads “the” by 3, the shape of a confident prediction.`,`为这个 Lab 编的：${list(c.toks,c.z)}。真实模型根据输入算出它们；“Paris”比“the”高 3，这是有把握的预测常见的样子。`);
}
function syncControls(){
  $$('input[data-key="top"]',app).forEach(i=>{const v=Math.round(S.top*2); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fZ(S.top); i.setAttribute('aria-valuetext',fZ(S.top));});
  $$('input[data-key="gap"]',app).forEach(i=>{if(+i.value!==S.gap) i.value=S.gap; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fN(S.gap);});
  $$('.seg-ctl[data-key="fmt"]',app).forEach(g=>A.ui.setPressed(g,S.fmt));
  $$('.seg-ctl[data-key="mv"]',app).forEach(g=>A.ui.setPressed(g,S.mv));
  const g=mode==='guided', id=g?STEPS[step].id:'', sb=mode==='sandbox', N=NOUN(), at=attnNow();
  const hide=(sel,h)=>{const e=$(sel,app); if(e) e.hidden=h;};
  hide('#sm-ctl-top-wrap',!((sb&&(S.cs==='lm'||S.cs==='attn'))||id==='shift'));
  hide('#sm-ctl-gap-wrap',!((sb&&S.cs==='ce')||id==='ce'));
  hide('#sm-ctl-mv-wrap',!(sb&&S.cs==='mask'));
  hide('#sm-ctl-fmt-wrap',!(sb||id==='fmt'));
  hide('#sm-ctl-z',mode==='challenge');
  hide('#sm-ctl-reset-wrap',!sb);
  const set=(sel,txt)=>{const e=$(sel,app); if(e&&e.textContent!==txt) e.textContent=txt;};
  set('#sm-ctl-top-wrap label > span:first-child',N.top);
  set('#sm-ctl-top-wrap .ctl-note',at?tx('Moves every score by the same amount; their gaps stay the same.','所有得分一起移动同样多，彼此的差距不变。'):tx('Moves every score by the same amount; their gaps stay the same.','所有得分一起移动同样多，彼此的差距不变。'));
  hide('#sm-hint',!sb); if(sb) set('#sm-hint',tx('Pick a scenario in the card, or set your own case here.','在卡片里选一个应用场景，或者在这里设置你自己的情况。'));
  set('#sm-scope',at?tx('Made-up scores; real arithmetic in the chosen format.','编的得分；在所选格式里真实地计算。'):tx('Made-up scores; real arithmetic in the chosen format.','编的得分；在所选格式里真实地计算。'));
  set('#sm-zlab',tx('The scores','得分'));
  if(mode!=='challenge') set('#sm-zlist',zNote());
  const ln=$('#sm-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln._h!==msg){ln.innerHTML=msg; ln._h=msg;}
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.txt);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  S.top=Math.max(-130,Math.min(130,Math.round(S.top*2)/2)); S.gap=Math.max(0,Math.min(150,Math.round(S.gap)));
  if(!FMT[S.fmt]) S.fmt='fp32'; if(!MV[S.mv]) S.mv='e9'; if(!['lm','attn','ce','mask'].includes(S.cs)) S.cs='lm';
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setOn(k){onStep=Math.max(0,Math.min(NB+1,k)); render();}
function setRem(k){if(!['shift','ce','on'].includes(k)) return; remView=k; if(k==='shift') S={...BASE}; else if(k==='ce') S={...BASE,cs:'ce',gap:120}; else {S={...BASE}; onStep=NB+1;} renderRail(); if(innerWidth<=760) A.ui.reveal($('#sm-stage',app));}
/* phones: the part's one control at the top of the chart, so the control and what it changes share one screen */
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='predict'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='shift') return topCtl('sm-ph-top',true);
  if(id==='fmt') return fmtCtl('sm-ph-fmt',true);
  if(id==='ce') return gapCtl('sm-ph-gap',true);
  if(id==='online') return `<div class="sm-onbtns"><button type="button" class="btn ghost sm" data-on="-1" aria-label="${esc(tx('Back one block','退回一块'))}">◂</button><button type="button" class="btn sm" data-on="1" aria-label="${esc(tx('Next block','下一块'))}">▸</button><span class="mono sm-onpos"></span></div>`;
  return '';
}
function renderPhone(){const el=$('#sm-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...BASE,...s.enter()};
  const st=$('#sm-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#sm-rail .step.current',app)||$('#sm-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), as labs/sampling.js does */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#sm-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#sm-rail .step.current',app)||$('#sm-rail .railcard',app)||$('#sm-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#sm-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#sm-insight',app)); else revealBox($('#sm-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chPick='start'; chTried=new Set();}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#sm-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const CSI=['lm','attn','ce','mask'];
const token=()=>[CSI.indexOf(S.cs),FIDS.indexOf(S.fmt),Math.round(S.top*2),S.gap,S.mv].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [ci,fi,t2,gap]=p.slice(0,4).map(Number), mv=p[4];
  if(!(ci>=0&&ci<4)||!(fi>=0&&fi<3)||!Number.isInteger(t2)||Math.abs(t2)>260||!Number.isInteger(gap)||gap<0||gap>150||!MV[mv]) return null;
  return {cs:CSI[ci],fmt:FIDS[fi],top:t2/2,gap,mv};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const rnd_=mulberry(seed*7919+13), top=Math.round(9+rnd_()*6), z=[top,top-1-Math.round(rnd_()*2),top-3-Math.round(rnd_()*3)];
  return {seed,z,n:naive(z,'fp16'),st:stable(z,'fp16'),ex:exactP(z)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#sm-drill-q',app), a=$('#sm-drill-a',app);
  qd.innerHTML=tx(`Three scores, <b>${d.z.join(', ')}</b>, go through a naive softmax in FP16 (largest number 65,504). What comes out, and what does subtracting the max give?`,
                  `三个得分 <b>${d.z.join('、')}</b> 用 FP16（最大能存 65,504）跑朴素 softmax。结果是什么？先减最大值又得到什么？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const ez=d.n.e.map(v=>fE(v,4)), nanN=hasNaN(d.n.p), zero=!nanN&&d.n.p.every(p=>p===0);
  a.innerHTML=tx(`<p><b>Naive: ${d.n.p.map(fP).join(', ')}. Max subtracted: ${d.st.p.map(fP).join(', ')}.</b></p><p>In FP16, ${eX('z')} = ${ez.join(', ')}; the sum is ${fE(d.n.s,4)}. ${nanN?'An Inf divided by an Inf sum is NaN.':(zero?'Each exp fits, but the sum passes 65,504 and becomes Inf, so every probability is 0.':'Everything fits, so the naive softmax works.')} With the max ${d.z[0]} subtracted: ${eX('0')} = 1, ${eX(fN(d.z[1]-d.z[0]))} = ${fE(d.st.e[1],4)}, ${eX(fN(d.z[2]-d.z[0]))} = ${fE(d.st.e[2],4)}; the sum ${fE(d.st.s,4)}.</p>`,
                 `<p><b>朴素：${d.n.p.map(fP).join('，')}。先减最大值：${d.st.p.map(fP).join('，')}。</b></p><p>FP16 里 ${eX('z')} = ${ez.join('，')}；总和 ${fE(d.n.s,4)}。${nanN?'Inf 除以为 Inf 的总和，得到 NaN。':(zero?'每个 exp 都存得下，但总和超过 65,504 变成 Inf，所以每个概率都是 0。':'都存得下，所以朴素 softmax 能算。')}减去最大值 ${d.z[0]}：${eX('0')} = 1，${eX(fN(d.z[1]-d.z[0]))} = ${fE(d.st.e[1],4)}，${eX(fN(d.z[2]-d.z[0]))} = ${fE(d.st.e[2],4)}；总和 ${fE(d.st.s,4)}。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const TERM={
 fx:()=>`<div>${esc(TIP.sm())}</div><div class="fx">${math('p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}</div>`,
 fmt:()=>`<div>${esc(TIP.fmtNow(viewNow()==='ch'?chOpt(chPick).f:S.fmt))}</div>`
};
const termTipHTML=k=>TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math'||k==='fx') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='fid') return tx('Click to jump to the notes at the end of the page.','点击跳到页末的“简化、假设和来源”。');
  if(k==='fmt'){if(mode==='challenge') return tx('Fixed during the challenge: the format follows the fix you pick.','挑战中固定不变：格式跟着你选的办法走。'); if(mode==='guided'&&!isP('fmt')) return tx('Fixed in this part; part 4 lets you change the format.','这一部分固定不变；第 4 部分可以改格式。'); return tx('Click to jump to “Number format”.','点击跳到“数值格式”。');}
  return '';
}
function jump(k,src){
  if(!app) return;
  if(k==='math'||k==='fx'){const d=$('#sm-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#sm-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='fid'){const b=$(`[data-pop="${ID}-fid"]`,app), p=document.getElementById(`${ID}-fid`); A.tip.hide(); if(p&&p.hidden&&b) b.click();
    const nd=document.getElementById('sm-fid-sum'), el=nd&&nd.getClientRects().length?nd:(p||b); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='fmt'){const el=$('#sm-ic-fmt',app)||$('#sm-ctl-fmt',app); if((mode==='guided'&&!isP('fmt'))||mode==='challenge'||!el||!el.getClientRects().length){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;} A.ui.jumpToEl(el);}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{const keep=S.cs, p=PRESETS.find(x=>x.s.cs===keep); S={...BASE,...(p?p.s:{})}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chOpt(chPick)).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#sm-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#sm-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* a change under Settings never moves the page (R3.2) */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#sm-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='top') bySettings(e.target,()=>setState({top:v/2}));
    else if(k==='gap') bySettings(e.target,()=>setState({gap:v}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#sm-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#sm-rail')&&!b.closest('[data-exits]')&&d.on==null) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#sm-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.on!=null){byLearner(()=>setOn(onStep+(+d.on))); A.tip.quiet(); return;}
    if(d.lever!=null&&mode==='challenge'){chPick=d.lever; chTried.add(d.lever); byLearner(()=>render()); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='fmt'){bySettings(b,()=>setState({fmt:d.v})); if(b.closest('#sm-rail')){wantObs=true; revealSoon(); wantObs=false;}}
      else if(k==='mv') bySettings(b,()=>setState({mv:d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>{S={...BASE,...p.s}; render();}); requestAnimationFrame(()=>{if(app) revealBox($('#sm-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#sm-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S as it was (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chPick,chTried:[...chTried],chSubmitted,chReveal,sbTouched,
    seenOK,seenLow,seenF:{...seenF},seenFin,onStep,remView,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(chOpt(chPick)).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'&&typeof x.top==='number'?parse([CSI.indexOf(x.cs),FIDS.indexOf(x.fmt),Math.round(x.top*2),x.gap,x.mv].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    const ch=mode==='challenge', okPick=x=>CHO.some(c=>c.id===x);
    chPick=ch&&(o.chPick==='start'||okPick(o.chPick))?o.chPick:'start'; chTried=new Set(ch&&Array.isArray(o.chTried)?o.chTried.filter(okPick):[]);
    chSubmitted=ch&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    seenOK=o.seenOK===true; seenLow=o.seenLow===true; seenFin=o.seenFin===true; seenF={fp32:true};
    if(o.seenF&&typeof o.seenF==='object') for(const f in o.seenF) if(FIDS.includes(f)&&o.seenF[f]===true) seenF[f]=true;
    onStep=Number.isInteger(o.onStep)&&o.onStep>=0&&o.onStep<=NB+1?o.onStep:0; remView=['shift','ce','on'].includes(o.remView)?o.remView:'shift';
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#sm-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#sm-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&step===0&&!fromSnap){const s=STEPS[0]; S={...BASE,...s.enter()};}
    fromSnap=false;
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#sm-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#sm-math',app).open=true;}},
  onResize(){if(app) draw();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='on'?$('#sm-on',app):(v==='ce'?$('#sm-ce',app):$('#sm-tab',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#sm-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.txt;
      if(v==='ch'){const e=chEval(chOpt(chPick)); vis.innerHTML=miniBar(e.p); read.innerHTML=esc(tx(`Largest error ${ptsT(e.err)} (goal below 0.1)`,`最大误差 ${ptsT(e.err)}（目标低于 0.1）`)); return;}
      if(v==='on'){const o=onNow(); vis.innerHTML=miniBar(onStep>NB?o.p:[]); read.innerHTML=esc(onStep>NB?tx(`All blocks read: largest difference from two passes ${o.diff===0?'0':fE(o.diff)}`,`读完四块：和两遍算法最大差 ${o.diff===0?'0':fE(o.diff)}`):tx(`Online softmax: ${onPos()}`,`online softmax：${onPos()}`)); return;}
      const r=compute();
      if(v==='ce'){const k=ceAll(r.c.z,CE.y,S.fmt); vis.innerHTML=''; read.innerHTML=esc(tx(`Gap ${fN(S.gap)}: naive loss ${fN(k.naive,2)}, log-sum-exp ${fN(k.lse,2)}`,`差距 ${fN(S.gap)}：朴素 loss ${fN(k.naive,2)}，log-sum-exp ${fN(k.lse,2)}`)); return;}
      vis.innerHTML=miniBar(r.n.p); const N=NOUN(), fm=S.fmt==='fp32'?'':tx(`, ${fF(S.fmt)}`,`（${fF(S.fmt)}）`);
      read.innerHTML=esc(tx(`${N.top} ${fZ(Math.max(...r.c.z))}${fm}: naive ${fP(r.n.p[r.imax])}, max subtracted ${fP(r.st.p[r.imax])}`,`${N.top} ${fZ(Math.max(...r.c.z))}${fm}：朴素 ${fP(r.n.p[r.imax])}，先减最大值 ${fP(r.st.p[r.imax])}`));}
  }
});
/* the mini result bar's picture: the chances laid end to end, or a red block for NaN */
function miniBar(p){if(!p.length) return `<div class="sm-dk"></div>`; if(hasNaN(p)) return `<div class="sm-dk sm-nan"></div>`; let x=0; return `<div class="sm-dk">`+p.map(v=>{if(v<=0) return ''; const s=`<i style="left:${(x*100).toFixed(2)}%;width:${(v*100).toFixed(2)}%"></i>`; x+=v; return s;}).join('')+`</div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o,(k,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v));
window.__sm=Object.freeze({
  rnd, naive:(z,f)=>deep(naive(z,f)), stable:(z,f)=>deep(stable(z,f)), exact:z=>deep(exactP(z)), ce:(z,y,f)=>deep((({n,st,...x})=>x)(ceAll(z,y,f))), online:(z,B,f)=>deep(online(z,B,f)), onBars:k=>deep(onBars(k)), ONSC,
  win:()=>deep(Object.fromEntries(FIDS.map(f=>[f,{hi:FMT[f].hi,lo:FMT[f].lo,max:FMT[f].max,sub:FMT[f].sub}]))),
  chEval:id=>{const e=chEval(chOpt(id)); return deep({err:e.err,ok:e.ok,p:e.p,s:e.s,ex:e.ex});}, chBest:()=>chBest().id,
  drill:s=>deep((({n,st,...x})=>({...x,p:n.p,ps:st.p,s:n.s}))(makeDrill(s))), exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});},
  fmt:Object.freeze({fE,fP,fN,fPP}), okR:()=>({...okR()}), ceOK:()=>ceOK(), fitTop:(z,f)=>fitTop(z,FMT[f]),
  consts:Object.freeze({LM:deep(LM),AT:deep(AT),CE:deep(CE),ON:deep(ON),CHS:deep(CHS),CHO:CHO.map(o=>({id:o.id,f:o.f,meth:o.meth,bytes:o.bytes,store:o.store,ch:o.ch})),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),CODE}),
  state:()=>({...S,mode,step,pred,observed:{...observed},seenOK,seenLow,seenF:{...seenF},seenFin,onStep,remView,chPick,chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
