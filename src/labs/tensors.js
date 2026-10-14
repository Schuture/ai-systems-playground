/* tensors (root "ML basics", world roots): How do you turn a Python loop into one vector operation?
   Fidelity: Toy run. The page computes every shape, every small array and every byte count in JS:
   - broadcasting: shapes lined up from the right; on each axis the sizes are equal, or one is 1 (or missing) and is stretched to the
     other; otherwise NumPy raises an error (the lab names the axis). Small 2-D arrays carry visible numbers (a[i][j] = 10i + j;
     b = 100, 200, … in reading order) and their sum is computed by index mapping (a size-1 axis always reads index 0);
   - einsum with two inputs: a letter in an input and missing from the output is summed; multiply-adds = the product of every distinct
     letter's size; the 4 × 3 by 3 × 2 example (a[i][k] = i + k, b[k][j] = 2k + j + 1) is computed;
   - 1-nearest neighbour at real sizes, FP32 (4 bytes a number): the largest intermediate array and the Python iterations of a double
     loop, one broadcast subtraction, chunks of C query rows, and the expansion ‖x − y‖² = ‖x‖² − 2x·y + ‖y‖² (one matmul). No timings.
   One H100 holds 80 GB, read as 80 GiB (site rule, A.terms.gib).
   window.__ts is a read-only hook for tests/tensors_numbers.py and tests/tensors_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='tensors';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const KiB=1024, MiB=KiB*1024, GiB=MiB*1024, TiB=GiB*1024, PiB=TiB*1024, F32=4, H100=80*GiB;
const parseShape=k=>String(k).split('x').map(Number);
/* broadcasting: one entry per axis of the result, left to right; k counts from the right (0 = the last axis) */
function bshape(a,b){
  const n=Math.max(a.length,b.length), ax=[];
  for(let p=0;p<n;p++){const k=n-1-p, i=a.length-1-k, j=b.length-1-k, x=i>=0?a[i]:null, y=j>=0?b[j]:null, xa=x==null?1:x, ya=y==null?1:y;
    let out=null, bad=false;
    if(xa===ya) out=xa; else if(xa===1) out=ya; else if(ya===1) out=xa; else bad=true;
    ax.push({p,k,x,y,out,bad,sa:!bad&&xa===1&&out>1,sb:!bad&&ya===1&&out>1});}
  const bads=ax.filter(c=>c.bad);
  return {a:a.slice(),b:b.slice(),n,ax,ok:!bads.length,err:bads.length?bads[bads.length-1]:null,out:bads.length?null:ax.map(c=>c.out)};
}
const prod=v=>v.reduce((p,x)=>p*x,1);
/* the small arrays with visible numbers (a is always 2-D here; b is 1-D or 2-D) */
const valA=(i,j)=>10*i+j;
const arrA=s=>[...Array(s[0])].map((_,i)=>[...Array(s[1])].map((_,j)=>valA(i,j)));
const arrB=s=>s.length===1?[[...Array(s[0])].map((_,j)=>100*(j+1))]:[...Array(s[0])].map((_,i)=>[...Array(s[1])].map((_,j)=>100*(i*s[1]+j+1)));
/* a + b by index mapping: an axis of size 1 (or a missing one) always reads index 0, so nothing is copied */
function bsum(a,b){const r=bshape(a,b); if(!r.ok||r.n!==2||a.length!==2) return null;
  const [R,C]=r.out, A2=arrA(a), B2=arrB(b), br=b.length===2?b[0]:1, bc=b[b.length-1];
  return [...Array(R)].map((_,i)=>[...Array(C)].map((_,j)=>A2[a[0]===1?0:i][a[1]===1?0:j]+B2[br===1?0:i][bc===1?0:j]));}
/* einsum with two inputs: up to 4 letters each, no letter twice in one array; summed = in an input, not in the output */
function einsum(spec,sizes){
  const m=/^([a-z]{1,4}),([a-z]{1,4})->([a-z]{0,4})$/.exec(spec); if(!m) return {ok:false};
  const [,x,y,z]=m; if([x,y,z].some(s=>new Set(s).size!==s.length)) return {ok:false};
  const all=[...new Set((x+y).split(''))]; if([...z].some(c=>!all.includes(c))) return {ok:false};
  const sz=c=>sizes[c], summed=all.filter(c=>!z.includes(c)), cols=[...z].concat(summed);
  const outShape=[...z].map(sz);
  return {ok:true,spec,x,y,z,cols,summed,sizes:{...sizes},shapeA:[...x].map(sz),shapeB:[...y].map(sz),outShape,
    macs:prod(all.map(sz)),outN:prod(outShape),perOut:prod(summed.map(sz))};
}
/* the 4 × 3 by 3 × 2 example */
const MA=[...Array(4)].map((_,i)=>[...Array(3)].map((_,k)=>i+k)), MB=[...Array(3)].map((_,k)=>[...Array(2)].map((_,j)=>2*k+j+1));
const MC=MA.map(r=>[0,1].map(j=>r.reduce((s,v,k)=>s+v*MB[k][j],0)));
const ATT={b:1,h:32,q:2048,k:2048,d:128};
const ES=[
 {id:'mm',spec:'ik,kj->ij',sizes:{i:4,k:3,j:2},t:{en:'Matrix multiply (initial setup)',zh:'矩阵乘法（初始设置）'},d:{en:'the 4 × 3 and 3 × 2 arrays in the chart, every number shown',zh:'图里的 4 × 3 和 3 × 2 两个数组，所有数值都写出来了'}},
 {id:'sc',spec:'bhqd,bhkd->bhqk',sizes:ATT,t:{en:'Attention scores',zh:'注意力得分'},d:{en:'two arrays, both of shape (1, 32, 2048, 128)',zh:'两个数组，形状都是 (1, 32, 2048, 128)'}},
 {id:'ov',spec:'bhqk,bhkd->bhqd',sizes:ATT,t:{en:'Attention output',zh:'注意力输出'},d:{en:'shapes (1, 32, 2048, 2048) and (1, 32, 2048, 128)',zh:'形状 (1, 32, 2048, 2048) 和 (1, 32, 2048, 128)'}},
 {id:'bm',spec:'bij,bjk->bik',sizes:{b:8,i:64,j:32,k:16},t:{en:'A batch of matrix multiplies',zh:'一批矩阵乘法'},d:{en:'8 separate matrix multiplies at once, 64 × 32 by 32 × 16',zh:'一次算完 8 对互不相干的矩阵乘法，每对是 64 × 32 乘 32 × 16'}}];
const esOf=id=>ES.find(e=>e.id===id)||ES[0];
const esR=id=>{const e=esOf(id); return einsum(e.spec,e.sizes);};
/* 1-nearest neighbour: N queries (test vectors), M training vectors, D numbers each, FP32 */
function nn(N,M,D){
  return {N,M,D,
    loop:{id:'loop',bytes:D*F32,iters:N*M,macs:N*M*D},
    bc:{id:'bc',bytes:N*M*D*F32,iters:0,macs:N*M*D},
    mm:{id:'mm',bytes:N*M*F32,iters:0,macs:N*M*D+(N+M)*D}};
}
const fits=b=>b<=H100;
/* the challenge: a search service, sizes the guided parts never show */
const CH={N:10000,M:200000,D:768,maxIters:1000,start:'loop'};
const CHO=[{id:'loop'},{id:'bc'},{id:'c1',C:1},{id:'c32',C:32},{id:'c256',C:256},{id:'mm'}];
function chEval(id){
  const o=CHO.find(x=>x.id===id)||CHO[0], {N,M,D}=CH;
  let bytes, iters;
  if(o.id==='loop'){bytes=D*F32; iters=N*M;} else if(o.id==='bc'){bytes=N*M*D*F32; iters=0;} else if(o.id==='mm'){bytes=N*M*F32; iters=0;}
  else {bytes=o.C*M*D*F32; iters=Math.ceil(N/o.C);}
  const fit=fits(bytes), few=iters<=CH.maxIters;
  return {id:o.id,C:o.C||0,bytes,iters,fit,few,ok:fit&&few,written:o.id==='mm'?N*M*F32:N*M*D*F32};
}
/* the best: among the options that meet the goal, the smallest largest array */
function chBest(){let b=null; CHO.forEach(o=>{const e=chEval(o.id); if(e.ok&&(!b||e.bytes<b.bytes)) b=e;}); return b;}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int, big=n=>A.fmt.big?A.fmt.big(n):fint(n), BIGH=n=>A.ui.big&&n>=1e6?A.ui.big(n):fint(n);
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
/* memory: GiB from 1 GiB up (3 significant digits, more than 6 digits as M), MiB and KiB below, so a 2 KiB array is not “0.000002 GiB” */
function fB(b){
  const s3=v=>v>=100?fint(Math.round(v)):fixd(v,v>=10?1:2);
  if(b>=GiB){const g=b/GiB; return (g>=1e6?big(Math.round(g)):s3(g))+' GiB';}
  if(b>=MiB) return s3(b/MiB)+' MiB';
  if(b>=KiB){const k=b/KiB; return (Number.isInteger(k)?fint(k):s3(k))+' KiB';}
  return fint(b)+tx(' bytes',' 字节');
}
/* decimal units with 3 significant digits, as on a vendor's spec sheet (B6) */
const s3d=v=>v>=100?fint(Math.round(v)):fixd(v,v>=10?1:2);
const fGB=b=>b>=1e12?s3d(b/1e12)+' TB':(b>=1e9?s3d(b/1e9)+' GB':(b>=1e6?s3d(b/1e6)+' MB':fint(b)+tx(' bytes',' 字节')));
/* a size with its other unit once: "1.86 GiB (2.00 GB)", "2 KiB (2,048 bytes)", "200 bytes" (A4: never the same value twice) */
const fBB=b=>b<KiB?fB(b):(b<1e6?`${fB(b)}${tx(` (${fint(b)} bytes)`,`（${fint(b)} 字节）`)}`:`${fB(b)}${tx(` (${fGB(b)})`,`（${fGB(b)}）`)}`);
const shp=a=>'('+a.join(', ')+(a.length===1?',':'')+')';
/* NumPy's own spelling in its error message: no space after the comma (B5) */
const npShp=a=>'('+a.join(',')+(a.length===1?',':'')+')';
const shK=k=>shp(parseShape(k));
const NS=' ';
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
/* R6.12: the first GiB on the page carries the definition (with what a GB is) */
const gib=t=>String(t).replace(/ GiB$/,' '+DEF('GiB',kt('gib')));
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const axisName=k=>k===0?tx('the last axis','最后一个轴'):(k===1?tx('the second axis from the right','右数第二个轴'):tx(`axis ${k+1} from the right`,`右数第 ${k+1} 个轴`));

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const N_ST=[1000,2000,5000,10000,20000,50000], M_ST=[2000,10000,50000,100000,500000], D_ST=[50,128,256,512,1024,2048];
const SH_A=['3x1','2x3','4x3','32x4096'], SH_B=['1x4','3','4','2x1','4096','32x1','32'];
const BASE={view:'nn',a:'3x1',b:'1x4',es:'mm',N:10000,M:50000,D:512};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seenErr=false, esTried=new Set(['mm']), dMoved=false, remView='bc', saved=null, chPick=CH.start, chSubmitted=false, chReveal=false, sbTouched=false;
let drillSeed=1, drillOpen=false, annT=null, userAct=false, wantObs=false;
const shownPrev={};

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 tensor:()=>tx('A tensor is an array of numbers with a shape: a list of sizes, one per axis. A shape of (2, 5) means 2 rows of 5 numbers; (32, 4096) is 32 rows of 4,096. NumPy calls it an ndarray, PyTorch a tensor.',
               'tensor（张量）就是带形状的数组：形状是一串大小，每个轴一个。形状 (2, 5) 表示 2 行、每行 5 个数值；(32, 4096) 是 32 行、每行 4,096 个数值。NumPy 叫它 ndarray，PyTorch 叫它 tensor。'),
 axis:()=>tx('An axis is one direction of an array: a matrix has two, rows and columns. Its size is how many entries the array has along it.',
             '轴是数组的一个方向：矩阵有两个轴，行和列。轴的大小，就是数组沿这个方向有多少项。'),
 bcast:()=>(mode==='guided'&&step<SIX('bcast'))
   ?tx('Broadcasting is the rule NumPy and PyTorch use to combine two arrays of different shapes in one operation, without a Python loop. Part 3 shows the rule.',
       'broadcasting（广播）是 NumPy 和 PyTorch 的一条规则：形状不同的两个数组，也能直接做一次运算，不用 Python 循环。第 3 部分细讲。')
   :tx('Broadcasting combines two arrays of different shapes: the shapes are lined up from the right; on each axis the sizes must be equal, or one must be 1 (a missing axis counts as 1), and a 1 is read again for every position of the other size. Nothing is copied.',
       'broadcasting（广播）让形状不同的两个数组直接运算。两个形状从右边对齐；每个轴上，两个大小要么相等，要么有一个是 1（缺的轴算 1）。大小为 1 的那一边会被重复读取，次数等于另一边的大小。整个过程不复制数据。'),
 einsum:()=>tx('einsum (Einstein summation) names every axis of every array with a letter. A letter that appears in the inputs but not after the arrow is summed over; every other letter becomes an axis of the output.',
               'einsum（爱因斯坦求和）给所有数组的每个轴各起一个字母名。出现在输入里、箭头后面却没有的字母，就对它求和；其余字母成为输出的轴。'),
 nn:()=>tx(`1-nearest neighbour (1-NN) gives each test vector the label of the training vector closest to it. The distance here is the squared Euclidean distance ${A.ui.math('{‖x - y‖}^2')} (its formula is in part 5).`,
           `1 近邻（1-NN）：给每个测试向量找出离它最近的训练向量，用那个训练向量的标签。这里的距离是欧氏距离的平方 ${A.ui.math('{‖x - y‖}^2')}（公式见第 5 部分）。`),
 fp32:()=>tx('FP32 is the 32-bit floating-point format: 4 bytes per number, NumPy’s float32 and PyTorch’s default. NumPy’s own default is float64, 8 bytes, which doubles every array.',
             'FP32 是 32 位浮点格式：每个数值 4 字节，即 NumPy 的 float32，也是 PyTorch 的默认格式。NumPy 自己默认用 float64，每个数值 8 字节，数组都会大一倍。'),
 /* part 4 and scenario 3: the attention words, from the kit's shared definitions (R2.8), each with this lab's sizes */
 token:()=>kt('token')+tx(' This lab uses a sequence of 2,048 tokens.',' 这个 Lab 用的是 2,048 个 token 的一段序列。'),
 qkv:()=>kt('qkv')+tx(' In this lab each of them has 128 numbers per head.',' 在这个 Lab 里，它们在每个 head 里各有 128 维。'),
 head:()=>kt('head')+tx(' Llama‑3.1‑8B, whose sizes this lab uses, runs 32 heads in each layer.',' 这个 Lab 用 Llama‑3.1‑8B 的大小，它每层有 32 个 head。'),
 aweight:()=>kt('aweight'),
 batch:()=>tx('A batch is a group of independent inputs, here sequences of tokens, computed together in one call. The letter b counts them; this example has a batch of 1.',
              'batch 是一起计算的一组互不相干的输入，这里是几段 token 序列。字母 b 数的就是它们；这个例子的 batch 是 1。'),
 score:()=>tx('A score is the dot product of one query vector and one key vector. The larger it is, the better that key matches the query.',
              '得分是一个 query 向量和一个 key 向量的点积。得分越大，这个 key 和这个 query 越匹配。'),
 /* A6: why the challenge limits the loop (R2.7: what it stands for, why it is set, what decides it, its usual size) */
 loop:()=>tx('The limit of 1,000 loop runs is this challenge’s setting. It stands for the fixed cost of each run: Python interprets the loop body again and launches at least one GPU operation, and that cost does not shrink when a run does less arithmetic. A few hundred runs add little next to the arithmetic; millions of runs spend most of the time on that cost. 1,000 is a round cut-off between the two.',
             '循环不超过 1,000 次是这道挑战题的设定。它代表每次循环的固定开销：Python 要重新解释一遍循环体，还至少要启动一次 GPU 运算；每次循环算得再少，这份开销也不会变小。跑几百次时，这份开销比起计算本身不算什么；跑到 1M 次以上，大部分时间都花在它上面。1,000 是两者之间取的一个整数界限。'),
 /* A9: the price of the expansion */
 prec:()=>tx(`When x and y are close, ${A.ui.math('{‖x‖}^2 + {‖y‖}^2')} and ${A.ui.math('2\\, x \\cdot y')} are large and nearly equal, and FP32 keeps only about 7 significant digits of each, so their difference loses most of its digits (catastrophic cancellation): a distance can come out slightly negative, and two near neighbours can swap. scikit-learn’s euclidean_distances uses the same expansion, clamps the result at 0 and computes FP32 inputs in FP64 chunks.`,
             `x 和 y 很接近时，${A.ui.math('{‖x‖}^2 + {‖y‖}^2')} 和 ${A.ui.math('2\\, x \\cdot y')} 都很大，又几乎相等。FP32 每个数值只保留约 7 位有效数字，两者一减，大部分有效数字就丢了（catastrophic cancellation）。于是距离可能算成一个很小的负数，两个很近的近邻也可能排反。scikit-learn 的 euclidean_distances 用的是同一个展开式；它把结果截到不小于 0，并把 FP32 的输入分块转成 FP64 再算。`),
 inter:()=>tx('An intermediate array is one the code creates on the way to the answer and throws away afterwards, such as the differences before they are squared and summed.',
              '中间数组是代码在算出答案的途中创建、用完就丢掉的数组，比如还没平方、求和之前的那些差值。'),
 h100:()=>tx('An NVIDIA H100 SXM has 80 GB of memory (the H100 NVL has 94 GB); this site reads the 80 GB as 80 GiB. Every array the code creates must fit in it at the same time.',
             'NVIDIA H100 SXM 有 80 GB 显存（H100 NVL 是 94 GB）；本站把这 80 GB 按 80 GiB 计。代码同时创建的所有数组都要放得进去。'),
 mac:()=>tx('A multiply-add is one multiplication followed by one addition into a running sum, the unit of work of a matrix multiply.',
            '一次乘加就是一次乘法加一次累加，是矩阵乘法的基本工作量单位。'),
 view:()=>tx('NumPy reads a stretched axis with a stride of 0: the same memory again for every position. np.broadcast_to returns such a read-only view without copying.',
             'NumPy 用步长 0 读取被拉伸的轴：每个位置都重新读同一块内存。np.broadcast_to 返回的就是这样一个只读视图，不复制数据。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'err',t:{en:'An error',zh:'报错'}},{v:'34',t:{en:'(3, 4)',zh:'(3, 4)'}},{v:'31',t:{en:'(3, 1)',zh:'(3, 1)'}},{v:'14',t:{en:'(1, 4)',zh:'(1, 4)'}}];
const PRED_RIGHT='34';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const hideOut=()=>mode==='guided'&&step<=SIX('predict');
const segA=(id,inline,list)=>A.ui.seg({id,key:'a',label:tx('Shape of a','a 的形状'),options:(list||SH_A).map(k=>[k,shK(k)]),value:S.a,inline});
const segB=(id,inline,list)=>A.ui.seg({id,key:'b',label:tx('Shape of b','b 的形状'),options:(list||SH_B).map(k=>[k,shK(k)]),value:S.b,inline});
const rng=(id,key,label,st,v,inline)=>A.ui.range({id,key,label,min:0,max:st.length-1,value:Math.max(0,st.indexOf(v)),out:fint(v),inline});
const esLevers=()=>`<div class="ts-lv" id="ts-ic-es">`+A.ui.levers({aria:tx('einsum strings, pick one','einsum 字符串，选一个'),value:S.es,
  options:ES.map(e=>({v:e.id,title:L(e.t),desc:`${e.spec}${tx(': ','：')}${L(e.d)}`,result:esc(esTried.has(e.id)||e.id===S.es?esRes(e.id):'')}))})+`</div>`;
const esRes=id=>{const r=esR(id); return tx(`sums ${r.summed.join(', ')}`,`对 ${r.summed.join('、')} 求和`);};
/* part 5 (R6.29, review 2026-10-03): each way of writing the distance, what each term computes, its shape and its memory at the current
   sizes, next to the formula; the bars of ‖x‖² are grouped ({‖x‖}^2) so both stand the same height */
const TERMS=[
 {k:'bcd',w:'bc',t:{en:'x\u00a0−\u00a0y for every pair, in every one of the D numbers',zh:'每一对向量在每一维上的差 x\u00a0−\u00a0y'},sh:'(N, M, D)',b:(N,M,D)=>N*M*D*F32},
 {k:'bcs',w:'bc',t:{en:'squared and summed over D: the distances',zh:'平方后对 D 求和，得到距离'},sh:'(N, M)',b:(N,M)=>N*M*F32},
 {k:'xx',w:'mm',m:'{‖x‖}^2',t:{en:'one number per test vector',zh:'每个测试向量一个数值'},sh:'(N, 1)',b:N=>N*F32},
 {k:'xy',w:'mm',m:'2\\, x \\cdot y',t:{en:'one dot product per pair, all from one matrix multiply X @ Yᵀ',zh:'每一对一个点积，全部由一次矩阵乘法 X @ Yᵀ 算出'},sh:'(N, M)',b:(N,M)=>N*M*F32},
 {k:'yy',w:'mm',m:'{‖y‖}^2',t:{en:'one number per training vector',zh:'每个训练向量一个数值'},sh:'(1, M)',b:(N,M)=>M*F32}];
const termBytes=k=>{const t=TERMS.find(x=>x.k===k); return t.b(S.N,S.M,S.D);};
function termsHTML(){
  const row=t=>{const b=t.b(S.N,S.M,S.D); return `<div class="ts-tr${fits(b)?'':' ts-over'}" data-term="${t.k}"><span class="ts-tt">${t.m?A.ui.math(t.m)+tx(': ','：'):''}${L(t.t)}</span><b class="mono ts-tb">${fB(b)}</b><span class="mono ts-ts">${tx('shape','形状')} ${t.sh}</span></div>`;};
  return `<div class="ts-terms" id="ts-terms">`+
   `<div class="ts-tw"><div class="ts-th">${tx('Broadcast','广播写法')} <code>((X[:, None] - Y[None])**2).sum(-1)</code></div><p class="ts-fx">${A.ui.math('{‖x − y‖}^2 = \\sum_{d=1}^{D} (x_d − y_d)^2')}</p>${TERMS.filter(t=>t.w==='bc').map(row).join('')}</div>`+
   `<div class="ts-tw"><div class="ts-th">${tx('Expansion: one matrix multiply','展开写法：一次矩阵乘法')}</div><p class="ts-fx">${A.ui.math('{‖x − y‖}^2 = {‖x‖}^2 − 2\\, x \\cdot y + {‖y‖}^2')}</p>${TERMS.filter(t=>t.w==='mm').map(row).join('')}`+
   `<div class="ts-tnote">${tx('The three terms broadcast to (N, M) when they are added: the distances, with no (N, M, D) array on the way.','三项相加时广播成 (N, M)，就是距离；途中没有 (N, M, D) 的数组。')}</div></div></div>`;}
/* the sliders sit in the same card: update the sizes in place, never re-render the card while a slider moves */
function updTerms(){const box=app&&$('#ts-terms',app); if(!box) return;
  TERMS.forEach(t=>{const r=box.querySelector(`[data-term="${t.k}"]`); if(!r) return; const b=t.b(S.N,S.M,S.D), el=r.querySelector('.ts-tb'), s=fB(b); if(el.textContent!==s) el.textContent=s; r.classList.toggle('ts-over',!fits(b));});}
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'nn',focus:'nn',
  lead:()=>tx('Find the nearest of 50,000 training vectors for each of 10,000 test vectors: three ways to write it.','为 10,000 个测试向量各找出最近的训练向量（共 50,000 个）：有三种写法。'),
  enter:()=>({...BASE}),
  body:()=>{const r=nn(BASE.N,BASE.M,BASE.D), tip=tx(`The three largest arrays: ${fB(r.loop.bytes)} for the double loop, which runs ${big(r.loop.iters)} times; ${fBB(r.bc.bytes)} for the broadcast; ${fBB(r.mm.bytes)} for the matrix multiply.`,`三种写法最大的数组：两层循环 ${fB(r.loop.bytes)}，要跑 ${big(r.loop.iters)} 次；广播 ${fBB(r.bc.bytes)}；矩阵乘法 ${fBB(r.mm.bytes)}。`);
   return tx(`<p>You have 10,000 test vectors and 50,000 training vectors, each a list of 512 numbers. For every test vector you want the training vector closest to it. This is ${DEF('1-nearest neighbour (1-NN)',TIP.nn())}, a classic interview task.</p>`+
     `<p>There are three ways to write it, and ${J('nn','the bars in the chart',tip)} compare them. A double loop is the easiest to write, but it runs ${BIGH(r.loop.iters)} Python iterations. One ${DEF('broadcast',TIP.bcast())} subtraction has no loop, but it first builds an ${DEF('intermediate array',TIP.inter())} of about a terabyte, too big for one ${DEF('H100',TIP.h100())}. One matrix multiply has no loop either, and its largest array stays under 2 ${DEF('GiB',kt('gib'))}.</p>`+
     `<p>All three sizes follow from the shapes of the arrays, so you can work them out before you run any code.</p>`,
     `<p>你手上有 10,000 个测试向量和 50,000 个训练向量，每个向量 512 维。任务是给每个测试向量找出离它最近的训练向量。这就是 ${DEF('1 近邻（1-NN）',TIP.nn())}，一道经典的面试题。</p>`+
     `<p>这件事有三种写法，${J('nn','图里的横条',tip)}把它们放在一起比。两层循环最好写，但要跑 ${BIGH(r.loop.iters)} 次 Python 迭代。一次${DEF('广播',TIP.bcast())}减法没有循环，却要先建一个大约 1 TB 的${DEF('中间数组',TIP.inter())}，一张 ${DEF('H100',TIP.h100())} 放不下。一次矩阵乘法同样没有循环，最大的数组还不到 2 ${DEF('GiB',kt('gib'))}。</p>`+
     `<p>这些大小都能从数组的形状直接算出来，不用先运行代码。</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},view:'bc',focus:'bc',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: what does (3, 1) + (1, 4) give?','先猜：(3, 1) + (1, 4) 会得到什么？'),
  enter:()=>({...BASE,view:'bc',a:'3x1',b:'1x4'}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>A ${DEF('tensor',TIP.tensor())} is an array with a shape. In the chart, a has shape (3, 1): 3 rows of 1 number. b has shape (1, 4): 1 row of 4 numbers. What does a + b give?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>${DEF('tensor（张量）',TIP.tensor())}就是带形状的数组。图里的 a 形状是 (3, 1)：3 行，每行 1 个数值。b 的形状是 (1, 4)：1 行，有 4 个数值。a + b 会得到什么？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'bcast',title:{en:'Broadcasting lines shapes up from the right',zh:'广播：形状从右边对齐'},view:'bc',focus:'bc',
  lead:()=>tx('Pick shapes for a and b: find a pair that does not broadcast.','给 a 和 b 选形状：找一对不能广播的。'),
  enter:()=>({...BASE,view:'bc',a:'3x1',b:'1x4'}),expect:()=>seenErr,showMe:{a:'4x3',b:'4'},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Broadcasting lets two arrays of different shapes work together without a loop. Its rule: line the two shapes up from the right and compare them one ${DEF('axis',TIP.axis())} at a time.</p><p>On each axis the two sizes must be equal, or one of them must be 1. A 1 is stretched to the other size (dashed in the chart), and a missing axis on the left counts as 1. Stretching copies nothing: ${DEF('the same value is read again',TIP.view())}.</p><p>Pick a shape for a and one for b, and look for a pair that does not broadcast:</p>`,
              `<p>广播让形状不同的两个数组直接运算，不用写循环。它的规则是：把两个形状从右边对齐，一个${DEF('轴',TIP.axis())}一个轴地比。</p><p>每个轴上，两个大小要么相等，要么其中一个是 1。是 1 的那一边拉伸成另一边的大小（图里的虚线）。左边缺的轴算作 1。拉伸并不复制数据，只是${DEF('把同一个值重复读几遍',TIP.view())}。</p><p>给 a 和 b 各选一个形状，试着找出一对不能广播的：</p>`)+segA('ts-ic-a',true,SH_A.slice(0,3))+segB('ts-ic-b',true,SH_B.slice(0,4)),
  obs:()=>obsBc()},
 {id:'einsum',title:{en:'einsum names every axis',zh:'einsum 给每个轴起名字'},view:'es',focus:'es',
  lead:()=>tx('A letter in the inputs and missing from the output is summed.','输入里有、输出里没有的字母，就被求和。'),
  enter:()=>{esTried=new Set(['mm']); return {...BASE,view:'es',es:'mm'};},expect:()=>esTried.has('sc'),showMe:{es:'sc'},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>${DEF('einsum',TIP.einsum())} gives every axis of every array a letter and follows one rule: a letter that is in the inputs but not after the arrow is summed.</p><p>Take ik,kj->ij. k is in both inputs and missing after the arrow, so each output number multiplies along k and adds the products up: that is a matrix multiply. i and j are written after the arrow, so they become the output’s two axes.</p><p>The two attention strings come from a language model (${labLink('mha')} explains them); here only their letters matter.</p>`,
              `<p>${DEF('einsum',TIP.einsum())} 给输入和输出的每个轴各起一个字母名，然后只用一条规则：输入里有、箭头后面没有的字母，就对它求和。</p><p>以 ik,kj->ij 为例。k 在两个输入里都有，箭头后面却没有，所以每个输出数值都是沿 k 逐个相乘、再加起来，这正是矩阵乘法。i 和 j 写在箭头后面，就成了输出的两个轴。</p><p>两个注意力的字符串取自语言模型（${labLink('mha')}里细讲），这里只看它们的字母。</p>`)+`<p class="small muted ts-one">${tx('Pick an einsum string and see which letters it sums over.','选一个 einsum 字符串，看看它对哪些字母求和。')}</p>`+esLevers(),
  obs:()=>obsEs()},
 {id:'nn',title:{en:'The 1-NN rewrite',zh:'改写 1-NN'},view:'nn',focus:'nn',
  lead:()=>tx('Change N, M and D: which bars move?','改 N、M 和 D：哪些横条在动？'),
  enter:()=>{dMoved=false; return {...BASE,view:'nn'};},expect:()=>dMoved,showMe:{D:1024},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>To find the nearest neighbour, you first need the distance between every test vector and every training vector. The two ways below compute the same distances; they differ in how big an array they keep on the way.</p><p>The broadcast way pairs every row of X with every row of Y. <code>X[:, None]</code> puts its size-1 axis in the middle, the place where Y’s M rows go; <code>Y[None]</code> puts its 1 in front, where X’s N rows go. So cell (i, j) of the difference is xᵢ − yⱼ: row i of X minus row j of Y.</p>`,
              `<p>要找最近邻，先得算出每个测试向量和每个训练向量之间的距离。下面两种写法算出的距离完全一样，差别在于中途要存下多大的数组。</p><p>广播写法让 X 的每一行和 Y 的每一行两两配对。<code>X[:, None]</code> 把大小为 1 的轴插在中间，这个位置是留给 Y 的 M 行的；<code>Y[None]</code> 把 1 插在最前面，留给 X 的 N 行。所以差值的第 (i, j) 格就是 xᵢ − yⱼ：X 的第 i 行减 Y 的第 j 行。</p>`)+termsHTML()+
    `<p>${tx('Change the sizes and watch which terms follow D:','改改这几个大小，看哪一项跟着 D 变：')}</p>`+
    rng('ts-ic-N','N',tx('N, test vectors','N，测试向量数'),N_ST,S.N,true)+rng('ts-ic-M','M',tx('M, training vectors','M，训练向量数'),M_ST,S.M,true)+rng('ts-ic-D','D',tx('D, numbers per vector','D，每个向量的维数'),D_ST,S.D,true),
  obs:()=>obsNn()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>remView,focus:()=>remView,
  lead:()=>tx('Shapes line up from the right; einsum sums the letters that disappear; size every intermediate array first.','形状从右边对齐；einsum 对消失的字母求和；先算中间数组有多大。'),
  enter:()=>{remView='bc'; return {...BASE,view:'bc',a:'4x3',b:'3'};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
const focusOf=s=>typeof s.focus==='function'?s.focus():(s.focus||'');
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
/* part 3: the result for the current pair, the verdict on the prediction's pair, then why (R3.10) */
function bcLine(r){
  const a=shp(r.a), b=shp(r.b);
  if(!r.ok){const e=r.err; return tx(`<p><b>${a} + ${b}: an error.</b> On ${axisName(e.k)}, a has ${e.x} and b has ${e.y}. Neither is 1, so NumPy raises “operands could not be broadcast together”.</p>`,
                                     `<p><b>${a} + ${b}：报错。</b>在${axisName(e.k)}上，a 是 ${e.x}，b 是 ${e.y}，两个都不是 1，所以 NumPy 报错“operands could not be broadcast together”。</p>`);}
  return tx(`<p><b>${a} + ${b} gives ${shp(r.out)}.</b></p>`,`<p><b>${a} + ${b} 得到 ${shp(r.out)}。</b></p>`);
}
function bcWhy(r){
  if(!r.ok){const e=r.err, rows=r.a[0];
    if(r.b.length===1&&r.b[0]===rows) return tx(`<p>Why: a 1-D b is lined up from the right, so it meets a’s last axis, the columns, not the rows. To add one number to each row, b needs shape (${rows}, 1).</p>`,`<p>原因：一维的 b 从右边对齐，对上的是 a 的最后一个轴，也就是列，不是行。想给每一行加一个值，b 的形状要是 (${rows}, 1)。</p>`);
    return tx(`<p>Why: only an axis of size 1 can be stretched. Neither ${e.x} nor ${e.y} is 1, so neither can become the other.</p>`,`<p>原因：只有大小为 1 的轴能被拉伸。${e.x} 和 ${e.y} 都不是 1，谁也变不成谁。</p>`);}
  const st=r.ax.filter(c=>c.sa||c.sb);
  if(!st.length) return tx('<p>Why: every axis already has the same size, so nothing is stretched.</p>','<p>原因：每个轴的大小本来就一样，不用拉伸。</p>');
  const unit=k=>k===0?tx('column','列'):(k===1?tx('row','行'):tx('slice','块'));
  const parts=st.map(c=>c.sa?tx(`a has only one ${unit(c.k)}, and NumPy reads it ${fint(c.out)} times.`,`a 只有 1 ${unit(c.k)}，NumPy 把这一${unit(c.k)}读 ${fint(c.out)} 遍。`)
    :(c.y==null?tx(`b has no ${axisName(c.k).replace(/^the /,'')}, so that axis counts as 1 and b is a single ${unit(c.k)}. NumPy reads that ${unit(c.k)} ${fint(c.out)} times.`,`b 没有${axisName(c.k)}，这个轴算作 1，所以 b 只有 1 ${unit(c.k)}。NumPy 把这一${unit(c.k)}读 ${fint(c.out)} 遍。`)
    :tx(`b has only one ${unit(c.k)}, and NumPy reads it ${fint(c.out)} times.`,`b 只有 1 ${unit(c.k)}，NumPy 把这一${unit(c.k)}读 ${fint(c.out)} 遍。`)));
  return tx(`<p>Why: ${parts.join(' ')} The dashed numbers in the chart are the values read again.</p>`,`<p>原因：${parts.join('')}图里虚线框中的数，就是重复读到的值。</p>`);
}
function obsBc(){
  const r=bshape(parseShape(S.a),parseShape(S.b)), predPair=S.a==='3x1'&&S.b==='1x4';
  return bcLine(r)+(predPair?verdict():'')+bcWhy(r)+(seenErr?'':tx('<p class="small">Now find a pair that does not broadcast.</p>','<p class="small">再找一对不能广播的。</p>'));
}
function obsEs(){
  const r=esR(S.es), e=esOf(S.es), sum=r.summed.join(', ');
  if(S.es==='mm') return tx(`<p><b>ik,kj->ij: the output has shape (4, 2), and k is summed: ${r.macs} ${DEF('multiply-adds',TIP.mac())}.</b> Each output number takes a row of input 1 and a column of input 2, multiplies them number by number and adds the products. For example, the top-left 13 = 0 × 1 + 1 × 3 + 2 × 5.</p><p>Now pick the attention scores.</p>`,
                             `<p><b>ik,kj->ij 的输出形状是 (4, 2)，对 k 求和，一共 ${r.macs} 次${DEF('乘加',TIP.mac())}。</b>每个输出数值都取输入 1 的一行和输入 2 的一列，逐个相乘再相加。比如左上角的 13 = 0 × 1 + 1 × 3 + 2 × 5。</p><p>接下来选“注意力得分”。</p>`);
  const head=tx(`<p><b>${esc(e.spec)}: the output has shape ${shp(r.outShape)}, and ${esc(sum)} is summed: ${big(r.macs)} ${DEF('multiply-adds',TIP.mac())}.</b></p>`,`<p><b>${esc(e.spec)} 的输出形状是 ${shp(r.outShape)}，对 ${esc(sum)} 求和，一共 ${big(r.macs)} 次${DEF('乘加',TIP.mac())}。</b></p>`);
  if(S.es==='sc') return head+tx(`<p>Why: only d is missing after the arrow, so only d is summed. b, h, q and k are all written after the arrow, so they become the output’s four axes.</p>`,
                                 `<p>原因：箭头后面只少了 d，所以只对 d 求和。b、h、q、k 都写在箭头后面，成了输出的四个轴。</p>`);
  if(S.es==='ov') return head+tx(`<p>Why: this time the letter missing after the arrow is k, so k is summed. b, h, q and d are written after the arrow and stay as the output’s axes.</p>`,
                                 `<p>原因：这次箭头后面少的是 k，所以对 k 求和。b、h、q、d 写在箭头后面，留作输出的轴。</p>`);
  return head+tx('<p>Why: only j disappears after the arrow, so only j is summed. b is still after the arrow, so it is not summed: the 8 pairs of matrices are multiplied separately, all in one call.</p>',
                 '<p>原因：箭头后面只少了 j，所以只对 j 求和。b 还在箭头后面，所以不求和：8 对矩阵各乘各的，一次做完。</p>');
}
function obsNn(){
  const r=nn(S.N,S.M,S.D), b0=nn(BASE.N,BASE.M,BASE.D);
  const fitTxt=x=>fits(x.bytes)?tx('which fits in one H100','一张 H100 放得下'):tx('which does not fit in one H100','一张 H100 放不下');
  const res=tx(`<p><b>At N = ${fint(S.N)}, M = ${fint(S.M)} and D = ${fint(S.D)}, the broadcast needs ${fB(r.bc.bytes)}, ${fitTxt(r.bc)}; the expansion needs ${fB(r.mm.bytes)}, ${fitTxt(r.mm)}.</b></p>`,
               `<p><b>N = ${fint(S.N)}、M = ${fint(S.M)}、D = ${fint(S.D)} 时，广播写法要 ${fB(r.bc.bytes)}，${fitTxt(r.bc)}；展开写法要 ${fB(r.mm.bytes)}，${fitTxt(r.mm)}。</b></p>`);
  if(S.D===BASE.D) return res+tx('<p class="small">Now change D, the numbers per vector, and watch which bar moves.</p>','<p class="small">现在改一下 D（每个向量的维数），看哪根横条在动。</p>');
  return res+tx(`<p>Why: D is exactly the axis that gets added up. The broadcast stores all D differences of every pair before it adds them, so its array is proportional to D. The expansion adds D up inside the matrix multiply and stores one number per pair, so D does not change its largest array.</p><p>Both ways do the same number of ${DEF('multiply-adds',TIP.mac())}: the expansion saves memory, not arithmetic.</p><p class="small">The price is ${DEF('precision',TIP.prec())}: when two vectors are very close, this subtraction loses digits in FP32. Clamp the distances at 0, and recompute the closest few exactly when their order matters.</p>`,
                `<p>原因：D 正是要被加掉的那个轴。广播写法先把每一对的 D 个差值全部存下来，再相加，所以它的数组和 D 成正比。展开写法在矩阵乘法内部就把 D 加掉了，每一对只存一个数值，所以 D 再大，它最大的数组也不变。</p><p>两种写法的${DEF('乘加',TIP.mac())}次数一样多：展开写法省的是显存，不是计算。</p><p class="small">代价是${DEF('精度',TIP.prec())}：两个向量很接近时，FP32 下这个减法会丢掉有效数字。所以要把距离截到不小于 0；最近的几个候选要分先后时，再精确算一遍。</p>`);
}
/* part 6: three lines, each with this lab's example and a button that puts it back in the chart (R6.23) */
const REMS=['bc','es','nn'];
function remBtn(k){const on=remView===k;
  const tip={bc:tx('Shows (4, 3) + (3,) in the chart.','在图里显示 (4, 3) + (3,)。'),es:tx('Shows the attention scores in the chart.','在图里显示注意力得分。'),nn:tx(`Shows in the chart how much memory each way of writing 1-NN needs, for ${fint(BASE.N)} test vectors and ${fint(BASE.M)} training vectors of ${BASE.D} numbers.`,`在图里显示 ${fint(BASE.N)} 个测试向量、${fint(BASE.M)} 个训练向量（每个 ${BASE.D} 维）时，1-NN 每种写法要多少显存。`)}[k];
  return ` <button type="button" class="linkbtn ts-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const r=nn(BASE.N,BASE.M,BASE.D);
  return tx(`<ul class="ts-remlist"><li><b>Broadcasting lines shapes up from the right; an axis of size 1, or a missing one, is stretched.</b> So a row of shape (D,) is added to every row of an (N, D) array, with no loop and no copy. A column must keep its axis of size 1, as in (N, 1). Written as (N,), it lines up with the last axis instead: it raises an error, or it quietly pairs with the wrong numbers.${remBtn('bc')}</li>`+
    `<li><b>In einsum, a letter that is in the inputs but not after the arrow is summed.</b> So the letters after the arrow name the result’s axes, and every other letter is summed away. One einsum string can write a matrix multiply, a batched one or attention scores, with no transpose or reshape first.${remBtn('es')}</li>`+
    `<li><b>Before you run it, work out how big every intermediate array is.</b> One vectorized line can build an array far bigger than its inputs and its result: an (N, M, D) array of differences is D times the size of the (N, M) distances you want. Rewriting the distance as the expansion, squared lengths minus twice the dot products, keeps the largest array as small as the result.${remBtn('nn')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Broadcasting copies the input.” It does not: a stretched value is only read again, once for each row it meets. When a broadcast takes a lot of memory, the result itself is big: (N, 1, D) minus (1, M, D) gives ${A.ui.math('N \\times M \\times D')} numbers.</p><p>✗ “einsum is always slower than matmul.” einsum is a way to write the operation; its speed depends on how the library runs it. PyTorch turns a two-input einsum into the same batched matrix multiply that @ uses. NumPy’s np.einsum runs its own loop by default, which is slower; with optimize=True it uses matrix multiplies too.</p><p>✗ “Vectorizing is only about speed.” Removing a loop can cost memory: the double loop holds one (D,) difference at a time, while the broadcast holds all ${A.ui.math('N \\times M \\times D')} differences at once. The expansion removes the loop and still keeps its largest array at (N, M).</p></div>${A.shell.toScenarios()}`,
    `<ul class="ts-remlist"><li><b>广播把形状从右边对齐，大小为 1 的轴和缺的轴会被拉伸。</b>所以形状为 (D,) 的一行，可以直接加到 (N, D) 数组的每一行上，不用循环，也不复制。一列数则要留着大小为 1 的轴，写成 (N, 1)。写成 (N,) 的话，它会对上最后一个轴，要么报错，要么悄悄和错的数配对。${remBtn('bc')}</li>`+
    `<li><b>einsum 里，输入里有、箭头后面没有的字母，就被求和。</b>所以箭头后面的字母就是结果的轴，其余字母都被求和掉。一条 einsum 字符串就能写出矩阵乘法、批量矩阵乘法或注意力得分，不用先转置或改形状。${remBtn('es')}</li>`+
    `<li><b>运行之前，先算一算每个中间数组有多大。</b>一行向量化的代码，中途可能建出比输入和结果都大得多的数组。比如 (N, M, D) 的差值数组，就是想要的 (N, M) 距离的 D 倍大。把距离改写成展开式（两个长度的平方减去两倍点积），最大的数组就只和结果一样大。${remBtn('nn')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “广播会复制输入。”不会。被拉伸的值只是被重复读取，每遇到一行就读一次。广播写法要是很占显存，原因在结果本身。比如 (N, 1, D) 减 (1, M, D)，结果就有 ${A.ui.math('N \\times M \\times D')} 个数值。</p><p>✗ “einsum 总是比矩阵乘法慢。”einsum 只是一种写法，快慢要看库怎么执行它。PyTorch 把两个输入的 einsum 变成和 @ 一样的批量矩阵乘法。NumPy 的 np.einsum 默认用自己的循环，会慢一些；传 optimize=True 后，它也改用矩阵乘法。</p><p>✗ “向量化只是为了快。”去掉循环也可能多占显存。两层循环每次只存一个 (D,) 的差值，广播写法却要一次存下全部 ${A.ui.math('N \\times M \\times D')} 个差值。展开写法同样去掉了循环，最大的数组却只有 (N, M)。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 bdoc:'https://numpy.org/doc/stable/user/basics.broadcasting.html',
 brst:'https://github.com/numpy/numpy/blob/main/doc/source/user/basics.broadcasting.rst',
 npes:'https://github.com/numpy/numpy/blob/main/numpy/_core/einsumfunc.py',
 tfun:'https://github.com/pytorch/pytorch/blob/main/torch/functional.py',
 tlin:'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Linear.cpp',
 harris:'https://www.nature.com/articles/s41586-020-2649-2',
 vdw:'https://arxiv.org/abs/1102.1523',
 einops:'https://openreview.net/forum?id=oapKSVM2bcj',
 skl:'https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/pairwise.py',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 graphs:'https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/'
};
const PRESETS=[
 {id:'bias',t:{en:'Add a bias to a batch of activations',zh:'给一批激活加上 bias'},d:{en:'A layer adds the same 4,096 numbers to each of 32 rows',zh:'一层网络给 32 行里的每一行加上同样的 4,096 个数值'},s:{view:'bc',a:'32x4096',b:'4096'}},
 {id:'norm',t:{en:'Normalize each row',zh:'把每一行归一化'},d:{en:'Divide every row by its own sum, one number per row',zh:'每一行除以自己的和，每行一个数值'},s:{view:'bc',a:'32x4096',b:'32x1'}},
 {id:'att',t:{en:'Attention scores with einsum',zh:'用 einsum 算注意力得分'},d:{en:'Every query against every key in each head',zh:'每个 head 里，每个 query 和每个 key 比一次'},s:{view:'es',es:'sc'}},
 {id:'knn',t:{en:'1-NN at interview sizes',zh:'面试题规模的 1-NN'},d:{en:'2,000 test vectors, 2,000 training vectors, 50 numbers each',zh:'2,000 个测试向量，2,000 个训练向量，每个 50 维'},s:{view:'nn',N:2000,M:2000,D:50}}
];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k]));
/* A12: scenario 2 divides; the stage names the operator of the scenario (the rule is the same for + − × /) */
const OP=()=>mode==='sandbox'&&presetNow()>=0&&PRESETS[presetNow()].id==='norm'?'/':'+';
function scenHTML(i){
  const p=PRESETS[i];
  if(p.id==='bias') return tx(`<p><b>A layer adds the same bias, shape (4096,), to each of 32 rows of activations, shape (32, 4096): the result is (32, 4096).</b></p><p>Both last axes are 4,096. The bias has no first axis, so it counts as 1 and stretches to 32. PyTorch’s linear layers add their bias this way, with no loop over the rows and no copy of the bias.</p>`,
                                   `<p><b>一层网络要给 32 行激活（形状 (32, 4096)）的每一行加上同一个 bias（形状 (4096,)），结果是 (32, 4096)。</b></p><p>两个形状的最后一个轴都是 4,096。bias 没有第一个轴，算作 1，被拉伸到 32。PyTorch 的线性层就是这样加 bias 的：不用循环每一行，也不复制 bias。</p>`);
  if(p.id==='norm') return tx(`<p><b>x / x.sum(axis=1, keepdims=True) divides (32, 4096) by (32, 1): each row gets one sum, which divides the 4,096 numbers of that row.</b></p><p>Without keepdims=True the sums have shape (32,). Lined up from the right, they meet the axis of 4,096, and NumPy raises an error. A square array is worse: (32, 32) / (32,) runs without an error, but x[i, j] is divided by the sum of row j, not of row i.</p>`,
                                   `<p><b>x / x.sum(axis=1, keepdims=True) 用 (32, 1) 去除 (32, 4096)：每一行先算出一个和，再用它去除这一行的 4,096 个数值。</b></p><p>如果不写 keepdims=True，和的形状是 (32,)。它从右边对齐，对上的是 4,096 那个轴，于是报错。数组是方的时候更麻烦：(32, 32) / (32,) 不报错，但 x[i, j] 除的是第 j 行的和，不是第 i 行的和。</p>`);
  if(p.id==='att'){const r=esR('sc');
    return tx(`<p><b>einsum('bhqd,bhkd->bhqk', q, k) compares every query with every key in each head: shape ${shp(r.outShape)}, ${big(r.macs)} multiply-adds.</b></p><p>The sizes are Llama‑3.1‑8B’s: 32 ${DEF('heads',TIP.head())} of 128 numbers each, here for 2,048 ${DEF('tokens',TIP.token())}. This einsum computes the dot product inside each score:</p><p class="ts-fx">${A.ui.math('q \\cdot k = \\sum_{d=1}^{128} q_d \\, k_d')}</p><p>Dividing it by √128 gives the score that ${labLink('mha')} uses; ${labLink('flash-attention')} computes the scores in tiles.</p>`,
              `<p><b>einsum('bhqd,bhkd->bhqk', q, k) 在每个 head 里把每个 query 和每个 key 比一次：形状 ${shp(r.outShape)}，一共 ${big(r.macs)} 次乘加。</b></p><p>大小取自 Llama‑3.1‑8B：32 个 ${DEF('head',TIP.head())}，每个 head 128 维；序列长 2,048 个 ${DEF('token',TIP.token())}。einsum 这一行算的是得分里的点积：</p><p class="ts-fx">${A.ui.math('q \\cdot k = \\sum_{d=1}^{128} q_d \\, k_d')}</p><p>点积再除以 √128，才是${labLink('mha')}里的得分；怎样分块计算，见 ${labLink('flash-attention')}。</p>`);}
  const r=nn(S.N,S.M,S.D);
  return tx(`<p><b>The interview problem “NumPy 1-NN as an Affine Layer” goes up to n and m of 2,000 and d of 50. At those sizes the broadcast needs ${fB(r.bc.bytes)}, which fits, but it is already 50 times the expansion’s ${fB(r.mm.bytes)}.</b></p><p>The ratio is always D, here 50, so small inputs hide the problem. An interviewer asks for the shape of every intermediate array; the way that still runs when the sizes grow is the expansion, at the price of ${DEF('a small error in the distance of very close pairs',TIP.prec())}.</p><p class="small muted">These sizes are FP32. NumPy’s default float64 takes 8 bytes a number: ${fB(2*r.bc.bytes)} and ${fB(2*r.mm.bytes)}, the same ratio.</p>`,
            `<p><b>面试题“NumPy 1-NN 写成一层 Wx+b”的最大规模是 n、m 各 2,000，d 为 50。这时广播写法要 ${fB(r.bc.bytes)}，放得下，但已经是展开写法（${fB(r.mm.bytes)}）的 50 倍。</b></p><p>两者之比总是 D，这里是 50，所以输入小的时候看不出问题。面试官会追问每个中间数组的形状。规模变大以后，还能跑的是展开写法，代价是${DEF('两个很近的向量之间的距离会有一点误差',TIP.prec())}。</p><p class="small muted">这些大小按 FP32 算。NumPy 默认用 float64，每个数值 8 字节，两者分别是 ${fB(2*r.bc.bytes)} 和 ${fB(2*r.mm.bytes)}，比例不变。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const CHT={loop:{en:'Double loop over pairs (initial setup)',zh:'两层循环遍历每一对（初始设置）'},bc:{en:'Broadcast everything at once',zh:'一次广播全部'},c1:{en:'Broadcast in chunks of 1 query',zh:'按 1 个查询一块广播'},
  c32:{en:'Broadcast in chunks of 32 queries',zh:'按 32 个查询一块广播'},c256:{en:'Broadcast in chunks of 256 queries',zh:'按 256 个查询一块广播'},mm:{en:'Matmul expansion',zh:'矩阵乘法展开'}};
/* B1: the double loop keeps a running minimum, so its largest array really is one (D,) difference */
const CHD={loop:{en:'for i, for j: dist = ((X[i] - Y[j])**2).sum(), keep the smallest',zh:'for i、for j：dist = ((X[i] - Y[j])**2).sum()，保留最小的'},bc:{en:'((X[:, None] - Y[None])**2).sum(-1)',zh:'((X[:, None] - Y[None])**2).sum(-1)'},
  c1:{en:'a loop over the queries, the broadcast inside',zh:'循环每个查询，里面做广播'},c32:{en:'a loop over blocks of 32 queries',zh:'循环每 32 个查询一块'},c256:{en:'a loop over blocks of 256 queries',zh:'循环每 256 个查询一块'},
  mm:{en:'(X**2).sum(1)[:, None] - 2 * X @ Y.T + (Y**2).sum(1), no loop',zh:'(X**2).sum(1)[:, None] - 2 * X @ Y.T + (Y**2).sum(1)，没有循环'}};
const itersTxt=n=>n===0?tx('no Python loop','没有 Python 循环'):tx(`${n>=1e6?big(n):fint(n)} ${n===1?'run':'runs'} of the loop`,`循环 ${n>=1e6?big(n):fint(n)} 次`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chPick), best=chBest();
  let h;
  if(me.ok&&me.id===best.id) h=tx(`<p class="okline ts-mt"><b>Goal met, with the smallest array: ${fB(me.bytes)} and ${itersTxt(me.iters)}.</b></p><p>The expansion never builds the differences: it sums over the 768 numbers inside the matrix multiply and keeps one number per pair. Its price is ${DEF('precision for very close pairs',TIP.prec())}: clamp the distances at 0, and recompute the closest few exactly when ties matter.</p>`,
                                  `<p class="okline ts-mt"><b>达成目标，而且数组最小：${fB(me.bytes)}，${itersTxt(me.iters)}。</b></p><p>展开写法根本不建差值数组。768 维的求和在矩阵乘法内部就完成了，每一对只留下一个数值。代价是${DEF('两个很近的向量之间的距离会有一点误差',TIP.prec())}。所以要把距离截到不小于 0；最近的几个候选要分先后时，再精确算一遍。</p>`);
  else if(me.ok) h=tx(`<p class="okline ts-mt"><b>Goal met: ${fB(me.bytes)} at the largest and ${itersTxt(me.iters)}.</b> But the matmul expansion meets it with a smaller array, ${fB(best.bytes)}, and no loop.</p><p>Chunks only cap the peak: over all the runs they still write the whole ${A.ui.math('N \\times M \\times D')} difference, ${fB(me.written)}, ${fint(Math.round(me.written/best.written))} times the expansion’s ${fB(best.written)}.</p>`,
                      `<p class="okline ts-mt"><b>达成目标：最大的数组 ${fB(me.bytes)}，${itersTxt(me.iters)}。</b>不过矩阵乘法展开也能达标，数组更小（${fB(best.bytes)}），还不用循环。</p><p>分块只是压低了峰值。所有轮次加起来，它还是要写出全部 ${A.ui.math('N \\times M \\times D')} 个差值，一共 ${fB(me.written)}，是展开写法（${fB(best.written)}）的 ${fint(Math.round(me.written/best.written))} 倍。</p>`);
  else {
    const why=!me.fit?tx(`<b>Not yet: the largest array is ${fB(me.bytes)}, more than the 80 GiB of one H100.</b>`,`<b>还没达到：最大的数组是 ${fB(me.bytes)}，超过一张 H100 的 80 GiB。</b>`)
      :tx(`<b>Not yet: the array fits (${fB(me.bytes)}), but the Python loop runs ${me.iters>=1e6?big(me.iters):fint(me.iters)} times, more than 1,000.</b>`,`<b>还没达到：数组放得下（${fB(me.bytes)}），但 Python 循环要跑 ${me.iters>=1e6?big(me.iters):fint(me.iters)} 次，超过 1,000 次。</b>`);
    h=`<p class="badline ts-mt">${why}</p>`+tx('<p class="small">Hint: a bigger chunk means fewer loop runs, but more differences held at once. Is there a way to write the distances that never holds the D axis at all?</p>',
       '<p class="small">提示：块越大，循环跑得越少，但每一块要存的差值越多。有没有一种写法，根本不用存下 D 这个轴？</p>')+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(!chReveal) return h;
  const rows=CHO.map(o=>chEval(o.id)), lg=b=>Math.log2(b/KiB);
  h+=A.ui.cmpChart({title:tx(`Largest intermediate array when finding the nearest of 200,000 stored vectors for 10,000 queries, D 768, FP32 (log scale; goal: at most 80 GiB and 1,000 loop runs)`,`为 10,000 个查询在 200,000 个存好的向量里找最近的一个（D 768，FP32），最大的中间数组（对数刻度，目标是不超过 80 GiB、循环不超过 1,000 次）`),
    brief:true,hiddenText:n=>tx(`The other ${n} ways miss at least one target and are not shown.`,`其余 ${n} 种写法至少差一个目标，图里没有列出。`),
    aria:tx('Largest intermediate array: ','最大的中间数组：')+rows.filter(e=>e.ok||e.id===chPick||e.id===CH.start).map(e=>`${L(CHT[e.id])}${tx(': ','：')}${fB(e.bytes)}, ${itersTxt(e.iters)}${e.ok?tx(', meets the goal','，达标'):''}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:(e.ok?'✓ ':'')+L(CHT[e.id]),value:lg(e.bytes),text:`${fB(e.bytes)}${tx(', ','，')}${e.iters===0?tx('no loop','无循环'):tx(`${e.iters>=1e6?big(e.iters):fint(e.iters)} runs`,`${e.iters>=1e6?big(e.iters):fint(e.iters)} 次`)}`,ok:e.ok,me:e.id===chPick,
      tags:(e.id===CH.start?[{text:tx('initial setup','初始设置')}]:[]).concat(e.id===chPick?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:lg(H100),goalLabel:tx('80 GiB: one H100','80 GiB：一张 H100'),max:lg(64*TiB),okText:tx('✓ meets both targets','✓ 两个目标都达到'),noText:tx('misses at least one','至少差一个'),goalText:tx('one H100: 80 GiB','一张 H100：80 GiB')});
  h+=tx(`<p class="small">The double loop, the initial setup, needs only ${fB(chEval('loop').bytes)} but runs ${big(chEval('loop').iters)} times. Broadcasting everything at once removes the loop, but its array is ${DEF('far larger than one H100',`It holds all ${A.ui.math('N \\times M \\times D')} differences at once: ${fB(chEval('bc').bytes)}.`)}. Chunks sit in between: 1 query loops too often, 256 queries no longer fit, and 32 meets the goal. The expansion needs neither the loop nor the D axis.</p>`,
        `<p class="small">初始设置的两层循环只要 ${fB(chEval('loop').bytes)}，却要跑 ${big(chEval('loop').iters)} 次。一次广播全部，没有循环，可数组${DEF('远超一张 H100 的显存',`它要同时存下全部 ${A.ui.math('N \\times M \\times D')} 个差值，一共 ${fB(chEval('bc').bytes)}。`)}。分块介于两者之间：1 个查询一块，循环太多；256 个一块，放不下；32 个一块，达标。展开写法既不用循环，也不用存 D 这个轴。</p>`);
  h+=tx('<p class="small muted">Note: this lab counts only the largest single array. A real run also holds the inputs, the output and short-lived copies (squaring the differences makes a second array of the same size), so an array near 80 GiB would not run either. Measure the peak with torch.cuda.max_memory_allocated() before you ship.</p>',
        '<p class="small muted">注意：这里只算了最大的那个数组。真实运行时，显存里还有输入、输出和临时数组（对差值求平方，会再建一个同样大的数组），所以接近 80 GiB 的数组其实也跑不起来。上线之前，用 torch.cuda.max_memory_allocated() 量一下峰值。</p>');
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===6?'Six':fint(STEPS.length)} short parts. Predict a shape first, then see how broadcasting lines shapes up, how einsum names and sums axes, and how one identity shrinks 1-NN’s memory. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测一个形状；再看广播怎样对齐形状，einsum 怎样给轴起名、求和；最后看一个恒等式怎样把 1-NN 的显存降下来。第一次学，从这里开始。`),
  sandbox:tx('Four everyday uses, from adding a bias to the 1-NN interview problem. Pick one to see its shapes and what they cost.','四个常见用法，从加 bias 到 1-NN 面试题。选一个，看它的形状和开销。'),
  challenge:tx('A search service: 10,000 queries against 200,000 stored vectors of 768 numbers, FP32, on one H100. Choose how to write the distances so that the largest array fits in 80 GiB and the Python loop runs at most 1,000 times.',
               '一个检索服务要在 200,000 个存好的 768 维向量里，为 10,000 个查询各找出最近的一个（FP32，一张 H100）。选一种算距离的写法，让最大的数组放得进 80 GiB，Python 循环不超过 1,000 次。')
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> the page computes every shape, every small array and every byte count in your browser; ${J('math','“Show the math”')} works them out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The rule NumPy and PyTorch use to combine arrays of different shapes, and einsum with two inputs, letters named as in NumPy.</li>`+
 `<li>The numbers in the small arrays are made up (10i + j and 100, 200, …) so that you can follow each one; the results are computed from them.</li>`+
 `<li>1-NN memory: the largest intermediate array of each way to write it, in FP32 (4 bytes a number), and how often its Python loop runs.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>Only the largest single array is counted. A real run also holds the inputs, the output and short-lived copies; the peak is a few times the largest array.</li>`+
 `<li>No timings: the lab counts loop runs, multiply-adds and bytes, not seconds. The GPU’s 80 GB is read as 80 GiB.</li>`+
 `<li>Attention sizes are Llama‑3.1‑8B’s: 32 heads of 128 numbers, at 2,048 tokens. The model computes keys and values for only 8 heads and shares each set among 4 of the 32 heads; this lab counts the scores as the 32 heads compute them.</li>`+
 `<li>The loop limit in the challenge, 1,000 runs, is a setting that stands for the fixed cost of each run of a Python loop.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Broadcasting rules: NumPy’s ${cite(SRC.bdoc,'broadcasting guide')} (${cite(SRC.brst,'source')}); PyTorch follows the same rules.</li>`+
 `<li>einsum: NumPy ${cite(SRC.npes,'<code>numpy/_core/einsumfunc.py</code>')}; PyTorch ${cite(SRC.tfun,'<code>torch/functional.py</code>')} and ${cite(SRC.tlin,'<code>aten/src/ATen/native/Linear.cpp</code>')}.</li>`+
 `<li>The 1-NN task: a standard interview question (see “Go further” in the challenge). The expansion’s precision loss: scikit-learn’s ${cite(SRC.skl,'<code>sklearn/metrics/pairwise.py</code>')} (<code>euclidean_distances</code>).</li>`+
 `<li>The H100 SXM’s 80 GB: NVIDIA’s ${cite(SRC.h100,'H100 page')}. The fixed cost of launching GPU work from Python: PyTorch’s ${cite(SRC.graphs,'post on CUDA Graphs')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>页面上的每个形状、每个小数组、每个字节数都由你的浏览器算出；${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>NumPy 和 PyTorch 让形状不同的数组直接运算的规则，以及两个输入的 einsum，字母写法和 NumPy 一样。</li>`+
 `<li>小数组里的数是编的（10i + j 和 100、200……），方便你逐个跟着看；结果由它们算出。</li>`+
 `<li>1-NN 的显存：每种写法的最大中间数组，按 FP32（每个数值 4 字节）算，以及它的 Python 循环跑几次。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>只算最大的那个数组。真实运行时还要放输入、输出和临时副本，峰值是最大数组的几倍。</li>`+
 `<li>不计时：这个 Lab 数的是循环次数、乘加次数和字节，不是秒。GPU 的 80 GB 按 80 GiB 计。</li>`+
 `<li>注意力的大小取自 Llama‑3.1‑8B：32 个 head，每个 128 维，2,048 个 token。这个模型只为 8 个 head 算 key 和 value，每一份由 32 个 head 中的 4 个共用；这个 Lab 按 32 个 head 实际算的次数计得分。</li>`+
 `<li>挑战题里循环不超过 1,000 次是一个设定，代表 Python 循环每跑一次的固定开销。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>广播规则：NumPy 的${cite(SRC.bdoc,'广播指南')}（${cite(SRC.brst,'源文件')}）；PyTorch 用同样的规则。</li>`+
 `<li>einsum：NumPy 的 ${cite(SRC.npes,'<code>numpy/_core/einsumfunc.py</code>')}；PyTorch 的 ${cite(SRC.tfun,'<code>torch/functional.py</code>')} 和 ${cite(SRC.tlin,'<code>aten/src/ATen/native/Linear.cpp</code>')}。</li>`+
 `<li>1-NN 任务：一道常见面试题（见挑战题里的“延伸练习”）。展开式的精度损失：scikit-learn 的 ${cite(SRC.skl,'<code>sklearn/metrics/pairwise.py</code>')}（<code>euclidean_distances</code>）。</li>`+
 `<li>H100 SXM 的 80 GB：NVIDIA 的 ${cite(SRC.h100,'H100 页面')}。从 Python 启动 GPU 运算的固定开销：PyTorch 关于 ${cite(SRC.graphs,'CUDA Graphs 的文章')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>You need Python loops and lists, and what a matrix is: a grid of numbers with rows and columns. Matrix multiplication takes a row of the first matrix and a column of the second, multiplies them number by number and adds the products.</p><p>This is the first lab of the ML basics; ${labLink('backprop')} builds on it.</p>`,
                    `<p>你需要会 Python 的循环和列表，知道矩阵是什么：一张有行有列的数表。矩阵乘法取第一个矩阵的一行、第二个矩阵的一列，逐个相乘再加起来。</p><p>这是 ML 基础的第一个 Lab；${labLink('backprop')}在它的基础上继续。</p>`);
const OUTCOME=()=>tx('predict the shape of a broadcast or an einsum, and rewrite a loop so that no intermediate array outgrows memory.','预测一次广播或 einsum 的结果形状，并把循环改写成中间数组不会撑爆显存的写法。');
const EXITS=()=>{
  const x=bshape([8,1,6,1],[7,1,5]), s1=esR('sc'), s4=einsum('bhqd,bhkd->bhqk',{...ATT,q:8192,k:8192}), ratio=s4.macs/s1.macs;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does (3, 1) + (1, 4) give a (3, 4) result without copying either input?','为什么 (3, 1) + (1, 4) 得到 (3, 4) 的结果，却不复制任何一个输入？'),
   o:[tx('NumPy first repeats the smaller array in memory until it matches the larger one, builds two 3 × 4 copies, and then adds them number by number.','NumPy 先在内存里把小的数组复制到和大的一样大，建好两个 3 × 4 的副本，再逐个相加。'),
      tx('The shapes line up from the right; on each axis a size of 1 is read again for every position of the other size, so a’s column and b’s row are reused across the grid.','形状从右边对齐。在每个轴上，大小为 1 的那一边，会按另一边的大小被重复读取，所以 a 的那一列和 b 的那一行在整个网格里反复用到。'),
      tx('NumPy multiplies the two shapes axis by axis, 3 × 1 rows and 1 × 4 columns, so any two shapes of the same length can be added.','NumPy 把两个形状逐轴相乘，3 × 1 行、1 × 4 列，所以长度相同的任意两个形状都能相加。')],a:1,
   n:[tx('Broadcasting never builds the stretched copies: NumPy reads a stretched axis with a stride of 0, the same memory again. Only the result is a new 3 × 4 array.','广播不会真的建出拉伸后的副本。NumPy 读被拉伸的轴时，步长是 0，也就是一遍遍读同一块内存。只有结果是新建的 3 × 4 数组。'),null,
     tx('Sizes are never multiplied: they must be equal or one must be 1. (4, 3) + (4,), for example, fails on the last axis, 3 against 4.','大小从来不相乘：两个大小要么相等，要么有一个是 1。比如 (4, 3) + (4,)，最后一个轴上是 3 对 4，会报错。')],
   why:tx('Broadcasting aligns shapes from the right and treats a 1 (or a missing axis) as “the same value for every position”. The result has the full shape, (3, 4), but the inputs stay as they are.','广播从右边对齐形状，把大小为 1 的轴（或缺的轴）当作“每个位置都是同一个值”。结果有完整的形状 (3, 4)，两个输入保持原样。')},
  {q:tx('In bhqd,bhkd->bhqk, why is d summed while b and h are kept, although all three letters appear in both inputs?','在 bhqd,bhkd->bhqk 里，b、h、d 都出现在两个输入里，为什么只对 d 求和，b 和 h 却留下了？'),
   o:[tx('einsum always sums the last letter of each input, which is d in both, and keeps every letter that comes before it in the string.','einsum 总是对每个输入的最后一个字母求和，这里两个都是 d，排在它前面的字母都留下。'),
      tx('einsum sums the letter with the largest size; d is 128 while b is 1 and h is 32, so d is the one that gets summed.','einsum 对大小最大的那个字母求和；d 是 128，b 是 1，h 是 32，所以被求和的是 d。'),
      tx('A letter is summed when it is missing after the arrow; b and h are written in the output, so each batch element and head keeps its own scores, and only d disappears.','字母在箭头后面没有，才被求和；b 和 h 写在输出里，所以每个 batch 元素、每个 head 保留自己的得分，只有 d 消失了。')],a:2,
   n:[tx('Position does not matter: in bhqk,bhkd->bhqd the last letters are k and d, yet only k is summed; d stays in the output.','位置不起作用：在 bhqk,bhkd->bhqd 里，两个输入的最后一个字母是 k 和 d，被求和的只有 k，d 留在了输出里。'),
     tx('Size does not matter: q and k are 2,048, larger than d, and both are kept because they are in the output.','大小不起作用：q 和 k 都是 2,048，比 d 大，却都留下了，因为它们在输出里。'),null],
   why:tx('The output string decides: every letter written after the arrow becomes an axis of the result, and every other letter is summed. So ik,kj->ij is a matrix multiply and bij,bjk->bik is a batch of them.','看箭头后面就知道：写在箭头后面的字母，都成为结果的轴；没写的字母，都被求和。所以 ik,kj->ij 是矩阵乘法，bij,bjk->bik 是一批矩阵乘法。')}],
 transfer:[
  {q:tx('What is the shape of an array of shape (8, 1, 6, 1) plus one of shape (7, 1, 5)?','形状 (8, 1, 6, 1) 的数组加上形状 (7, 1, 5) 的数组，结果是什么形状？'),
   o:[tx('An error','报错'),'(8, 1, 6, 5)','(8, 7, 6, 5)','(7, 6, 5)'],a:x.ok?['(8, 1, 6, 5)','(8, 7, 6, 5)','(7, 6, 5)'].indexOf(shp(x.out))+1:0,
   /* A27: what each wrong pick gets wrong, before the derivation */
   n:[tx('No axis fails here: on every axis the two sizes are equal or one of them is 1, and a missing axis counts as 1.','这里没有哪个轴对不上：每个轴上两个大小要么相等，要么有一个是 1，缺的轴也算 1。'),
      tx('(8, 1, 6, 5) keeps a 1 on the second axis, where the other array has 7; a 1 stretches to the other size, 7.','(8, 1, 6, 5) 在第二个轴上留着 1，可另一边的数组那里是 7；1 会被拉伸成另一边的大小 7。'),null,
      tx('(7, 6, 5) drops the first axis: the shorter shape gets a 1 in front, and that 1 stretches to 8, so the 8 stays.','(7, 6, 5) 丢掉了第一个轴：短的形状在前面补 1，这个 1 被拉伸成 8，所以 8 留下。')],
   why:tx(`Line them up from the right: 1 and 5 give 5; 6 and 1 give 6; 1 and 7 give 7; 8 and a missing axis give 8. The result is ${x.ok?shp(x.out):'an error'}. This is the example in NumPy’s own broadcasting guide.`,`从右边对齐：1 和 5 得 5；6 和 1 得 6；1 和 7 得 7；8 和缺的轴得 8。结果是 ${x.ok?shp(x.out):'报错'}。这正是 NumPy 广播指南里的例子。`)},
  {q:tx(`The attention scores bhqd,bhkd->bhqk take ${big(s1.macs)} multiply-adds at 2,048 tokens. At 8,192 tokens, everything else the same, about how many times as many?`,`用 bhqd,bhkd->bhqk 算注意力得分，2,048 个 token 时要 ${big(s1.macs)} 次乘加。上下文变成 8,192 个 token，其余不变，乘加次数大约是原来的几倍？`),
   o:[tx('About 4 times','约 4 倍'),tx('About 16 times','约 16 倍'),tx('About 64 times','约 64 倍')],a:[4,16,64].indexOf(ratio),
   n:[tx('4 times counts only the queries. The keys grow too: each of the 4 times as many queries is scored against 4 times as many keys.','4 倍只算了 query 变多。key 也变多了：4 倍的 query，每一个都要和 4 倍的 key 算得分。'),null,
      tx('64 times would need d to grow as well. Each head still has 128 numbers per vector; only q and k grow.','64 倍要连 d 也变大才行。每个 head 的向量还是 128 维，变大的只有 q 和 k。')],
   why:tx(`About ${ratio} times. The work of the attention scores grows with the square of the context length, because every token’s q is scored against every token’s k. Four times the tokens gives 4 × 4 = ${ratio} times as many q–k pairs: ${big(s4.macs)} multiply-adds instead of ${big(s1.macs)}. Only attention grows this way; the rest of the model (the MLP and the projections) grows in step with the number of tokens.`,
          `约 ${ratio} 倍。注意力得分的计算量随上下文长度的平方增长，因为每个 token 的 q 都要和每个 token 的 k 算一个得分。token 变成 4 倍，q 和 k 的配对就变成 4 × 4 = ${ratio} 倍，乘加次数从 ${big(s1.macs)} 变成 ${big(s4.macs)}。按平方增长的只有 attention 这一步；模型的其余部件（MLP 和各个投影矩阵），计算量只和 token 数成正比。`)}],
 practice:[
  {id:'numpy-1nn',d:tx('Part 1: write 1-NN with no Python loop and state the shape of every intermediate array; the reference answer is this lab’s expansion, with a keepdims trap.','Part 1：不用 Python 循环写 1-NN，并写出每个中间数组的形状；参考答案就是这个 Lab 的展开写法，还有一个 keepdims 的坑。')},
  {id:'qkv-einsum',d:tx('Part 1: compute the attention scores and the weighted sum of the values with two einsum calls, then add a mask.','Part 1：用两次 einsum 算出注意力得分和对 value 的加权和，再加上 mask。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Tensors & shapes','张量与形状'),title:l?L(l.q):tx('How do you turn a Python loop into one vector operation?','怎么把 Python 循环变成一次向量运算？'),
    minutes:10,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: loops and matrices','前置知识：循环和矩阵'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage ts-stage" id="ts-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="ts-st-title"></div><div class="meter-sub" id="ts-st-sub"></div></div>`+
   `<div class="stage-actions"><div class="seg-ctl ts-view" id="ts-view" role="group" aria-label="${esc(tx('What the chart shows','图里显示什么'))}"><button type="button" data-view="bc" aria-pressed="false">${tx('Broadcasting','广播')}</button><button type="button" data-view="es" aria-pressed="false">einsum</button><button type="button" data-view="nn" aria-pressed="false">1-NN</button></div><span class="status" id="ts-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="ts-legend"></div>`+
   `<div class="stage-graph graph ts-graph" id="ts-graph"><div class="ts-phone" id="ts-phone"></div><div class="ts-bc" id="ts-bc"></div><div class="ts-es" id="ts-es" hidden></div><div class="ts-nn" id="ts-nn" hidden></div></div>`+
   `<p class="ts-scope small muted" tabindex="0" data-tipx="${esc(tx('Every shape, number and byte count on this page is computed in your browser by the rules shown. The small arrays use made-up numbers so you can follow them; no model and no GPU run here.','这一页的每个形状、每个数值和每个字节数，都由你的浏览器按图里的规则算出。小数组用的是编的数，方便你跟着看；这里没有运行模型，也没有用 GPU。'))}">${tx('Computed here: shapes, small arrays and byte counts; no timings.','在页面里算出：形状、小数组和字节数；不计时。')}</p>`+
   `<div class="metrics ts-metrics" id="ts-metrics"></div>`+
   `<details class="math" id="ts-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ts-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="ts-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ts-lock" hidden></p>`+
   `<div class="ts-grp lockable" id="ts-g-bc"><div class="ts-gh">${tx('Broadcasting: the shapes of a and b','广播：a 和 b 的形状')}</div>`+segA('ts-ctl-a',false)+segB('ts-ctl-b',false)+
    `<div class="ctl-note">${tx('Each shape lists the size of every axis: (32, 4096) is 32 rows of 4,096 numbers.','形状列出每个轴的大小：(32, 4096) 就是 32 行，每行 4,096 个数值。')}</div></div>`+
   `<div class="ts-grp lockable" id="ts-g-es"><div class="ts-gh">einsum</div>`+A.ui.seg({id:'ts-ctl-es',key:'es',label:tx('String','字符串'),options:ES.map(e=>[e.id,e.spec]),value:S.es})+
    `<div class="ctl-note">${tx('The attention sizes follow Llama‑3.1‑8B: 32 heads, 128 numbers per head, 2,048 tokens, a batch of 1. These sizes change only the number of multiply-adds, not which letter is summed.','注意力的大小按 Llama‑3.1‑8B 设定：32 个 head，每个 head 128 维，2,048 个 token，batch 为 1。这些大小只影响乘加次数，不影响哪个字母被求和。')}</div></div>`+
   `<div class="ts-grp lockable" id="ts-g-nn"><div class="ts-gh">${tx('1-NN sizes','1-NN 的规模')}</div>`+
    rng('ts-ctl-N','N',tx('N, test vectors','N，测试向量数'),N_ST,S.N,false)+rng('ts-ctl-M','M',tx('M, training vectors','M，训练向量数'),M_ST,S.M,false)+rng('ts-ctl-D','D',tx('D, numbers per vector','D，每个向量的维数'),D_ST,S.D,false)+
    `<div class="ctl-note">${(()=>{const r=nn(2000,2000,50); return tx(`Every number is ${DEF('FP32',TIP.fp32())}, 4 bytes: NumPy’s float32 and PyTorch’s default. NumPy’s own default is float64, 8 bytes, which doubles every bar (at the interview sizes, ${fB(2*r.bc.bytes)} and ${fB(2*r.mm.bytes)}); half precision halves them. No ratio changes. The GPU line is one H100, 80 GB read as 80 GiB.`,`每个数值按 ${DEF('FP32',TIP.fp32())} 算，占 4 字节，对应 NumPy 的 float32，也是 PyTorch 的默认格式。NumPy 自己默认用 float64，占 8 字节，每根横条都会翻倍（面试题规模下是 ${fB(2*r.bc.bytes)} 和 ${fB(2*r.mm.bytes)}）；用半精度则减半。横条之间的比例不变。图里那条粗线是一张 H100，80 GB 按 80 GiB 计。`);})()}</div></div>`+
   `<div class="lockable" id="ts-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

# 1. Broadcasting: a bias row added to every row
a = np.arange(12.0).reshape(4, 3)       # shape (4, 3)
bias = np.array([100.0, 200.0, 300.0])  # shape (3,)
print((a + bias).shape)                 # (4, 3)

# 2. einsum: attention scores, d is summed
rng = np.random.default_rng(0)
q = rng.standard_normal((1, 2, 5, 8))   # b, h, q, d
k = rng.standard_normal((1, 2, 6, 8))   # b, h, k, d
s = np.einsum("bhqd,bhkd->bhqk", q, k, optimize=True)
print(s.shape, np.allclose(s, q @ k.swapaxes(-1, -2)))

# 3. 1-NN: the expansion keeps (N, M), never (N, M, D)
def nn_loop(X, Y):                      # one (D,) difference at a time
    out = []
    for x in X:
        best, arg = np.inf, -1
        for j, y in enumerate(Y):
            dist = ((x - y) ** 2).sum()
            if dist < best:
                best, arg = dist, j
        out.append(arg)
    return np.array(out)

def nn_matmul(X, Y):
    d = ((X ** 2).sum(1)[:, None]          # (N, 1)
         - 2 * X @ Y.T                     # (N, M)
         + (Y ** 2).sum(1)[None, :])       # (1, M)
    return d.argmin(axis=1)

X = rng.standard_normal((50, 16))   # N = 50 test vectors
Y = rng.standard_normal((80, 16))   # M = 80 training vectors
print((nn_loop(X, Y) == nn_matmul(X, Y)).all())
# (4, 3)
# (1, 2, 5, 6) True
# True`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ts-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ts-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A broadcast, the attention scores with einsum, and 1-NN with the expansion, checked against the loop on small data.','一次广播、用 einsum 算注意力得分、用展开写法做 1-NN，并在小数据上和循环版核对。')}</p><pre class="code" id="ts-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ts-prod">`+
   tx(`<li>NumPy’s ${cite(SRC.bdoc,'broadcasting guide')} states the rule this lab uses: compare shapes from the trailing axis; sizes are compatible when equal or when one is 1. Its source is ${cite(SRC.brst,'<code>doc/source/user/basics.broadcasting.rst</code>')}.</li>`+
      `<li>${cite(SRC.npes,'<code>numpy/_core/einsumfunc.py</code>')}: with the default <code>optimize=False</code>, <code>np.einsum</code> runs its own C loop (<code>c_einsum</code>); with <code>optimize=True</code> it chooses an order and runs each two-input step as a (batched) matrix multiply.</li>`+
      `<li>PyTorch’s ${cite(SRC.tfun,'<code>torch/functional.py</code>')} defines <code>torch.einsum</code>; the work happens in ATen’s ${cite(SRC.tlin,'<code>aten/src/ATen/native/Linear.cpp</code>')}, which turns each two-input contraction into a permute, a reshape and a batched matrix multiply (<code>bmm</code>).</li>`+
      `<li>scikit-learn’s <code>euclidean_distances</code> in ${cite(SRC.skl,'<code>sklearn/metrics/pairwise.py</code>')} uses this lab’s expansion and warns that it “potentially suffers from catastrophic cancellation”: it clamps the result at 0 and computes FP32 inputs in FP64 chunks.</li>`,
      `<li>NumPy 的${cite(SRC.bdoc,'广播指南')}写明了这个 Lab 用的规则：从最后一个轴开始比较形状；两个大小相等或有一个是 1 时兼容。源文件是 ${cite(SRC.brst,'<code>doc/source/user/basics.broadcasting.rst</code>')}。</li>`+
      `<li>${cite(SRC.npes,'<code>numpy/_core/einsumfunc.py</code>')}：默认 <code>optimize=False</code> 时，<code>np.einsum</code> 用自己的 C 循环（<code>c_einsum</code>）；<code>optimize=True</code> 时它先选计算顺序，再把每一步两个输入的运算当作（批量）矩阵乘法来做。</li>`+
      `<li>PyTorch 的 ${cite(SRC.tfun,'<code>torch/functional.py</code>')} 定义了 <code>torch.einsum</code>；真正的计算在 ATen 的 ${cite(SRC.tlin,'<code>aten/src/ATen/native/Linear.cpp</code>')} 里，它把每次两个输入的缩并变成转置、reshape 和一次批量矩阵乘法（<code>bmm</code>）。</li>`+
      `<li>scikit-learn 的 <code>euclidean_distances</code>（${cite(SRC.skl,'<code>sklearn/metrics/pairwise.py</code>')}）用的就是这个 Lab 的展开式，文档提醒它可能有 catastrophic cancellation（相近的大数相减丢掉有效数字）：它把结果截到不小于 0，FP32 的输入分块转成 FP64 来算。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.harris,'Array programming with NumPy')}<span>${tx('Harris et al., Nature 585, 2020: the array model behind NumPy, with shapes, strides, broadcasting and vectorized operations','Harris 等，Nature 585，2020：NumPy 背后的数组模型，包括形状、步长、广播和向量化运算')}</span></li>`+
   `<li>${cite(SRC.vdw,'The NumPy array: a structure for efficient numerical computation')}<span>${tx('van der Walt, Colbert and Varoquaux, 2011: strides and views, the reason a broadcast reads instead of copying','van der Walt、Colbert、Varoquaux，2011：步长和视图，这就是广播只读不复制的原因')}</span></li>`+
   `<li>${cite(SRC.einops,'Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation')}<span>${tx('Rogozhnikov, ICLR 2022: einsum-style names for every axis of reshapes and reductions too','Rogozhnikov，ICLR 2022：把 einsum 式的轴命名也用到 reshape 和归约上')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ts-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ts-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ts-app m2" id="ts-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ts-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how its shapes work together.','选一个场景，看看那里的形状怎么一起运算。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ts-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Find the nearest of ${fint(CH.M)} stored vectors for each of ${fint(CH.N)} queries: one H100, at most ${fint(CH.maxIters)} loop runs`,`在 ${fint(CH.M)} 个存好的向量里，为 ${fint(CH.N)} 个查询各找出最近的一个：一张 H100，循环不超过 ${fint(CH.maxIters)} 次`)}</h3>`+
   `<p>${tx(`A search service stores ${fint(CH.M)} vectors of ${CH.D} numbers. For each of ${fint(CH.N)} new queries it must find the nearest stored vector, in FP32 with PyTorch on one H100 (80 GB, read as 80 GiB). Your colleague wrote a double loop. Choose a way to write it that meets two conditions: the largest intermediate array fits in 80 GiB, and the Python loop runs at most ${DEF(`${fint(CH.maxIters)} times`,TIP.loop())}.`,
            `一个检索服务存了 ${fint(CH.M)} 个向量，每个 ${CH.D} 维。来了 ${fint(CH.N)} 个新查询，要为每个查询找出离它最近的那个向量。计算用 PyTorch，FP32，在一张 H100 上跑（80 GB，按 80 GiB 计）。同事写的是两层循环。请选一种写法，满足两个条件：最大的中间数组放得进 80 GiB；Python 循环不超过 ${DEF(`${fint(CH.maxIters)} 次`,TIP.loop())}。`)}</p>`+
   `<div class="ts-lv" id="ts-ic-ch">`+A.ui.levers({aria:tx('Ways to write it, pick one','写法，选一种'),value:chPick,options:CHO.map(o=>({v:o.id,title:L(CHT[o.id]),desc:L(CHD[o.id]),result:''}))})+`</div>`+
   `<div class="goal mono ts-goal" id="ts-ch-g1"><span>${tx('Largest intermediate array','最大的中间数组')}</span><span><b id="ts-ch-b"></b>${tx(' (goal: at most 80 GiB)','（目标：不超过 80 GiB）')}</span></div>`+
   `<div class="goal mono ts-goal" id="ts-ch-g2"><span>${tx('Python loop runs','Python 循环次数')}</span><span><b id="ts-ch-i"></b>${tx(' (goal: at most 1,000)','（目标：不超过 1,000 次）')}</span></div>`+
   `<p class="small ts-goalwhy">${tx('Why a loop limit: every run of a Python loop pays a fixed cost, Python’s own work and at least one launch of work on the GPU, however little that run computes.','为什么要限制循环次数：Python 循环每跑一次，都有一份固定开销，包括 Python 自己的工作和至少一次 GPU 运算的启动。这一次算得再少，这份开销也省不掉。')}</p>`+
   submit+`<div id="ts-ch-res" class="ts-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},bcast:{en:'Find a pair that does not broadcast to continue.',zh:'找到一对不能广播的形状才能继续。'},
    einsum:{en:'Pick the attention scores to continue.',zh:'选“注意力得分”才能继续。'},nn:{en:'Change D to continue.',zh:'改一下 D 才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#ts-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="ts-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="ts-why" hidden></p>');}
  A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#ts-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.gate||s.gate());
    $$('#ts-ic-es [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===S.es)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=esTried.has(v)||v===S.es?esRes(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.ts-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ts-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chPick), b=$('#ts-ch-b',rail), it=$('#ts-ch-i',rail); if(!b) return;
    b.textContent=fB(e.bytes); it.textContent=e.iters===0?tx('0 (no loop)','0（没有循环）'):(e.iters>=1e6?big(e.iters):fint(e.iters));
    const g1=$('#ts-ch-g1',rail), g2=$('#ts-ch-g2',rail);
    g1.classList.toggle('met',e.fit); g1.classList.toggle('ts-over',!e.fit); g2.classList.toggle('met',e.few); g2.classList.toggle('ts-over',!e.few);
    $$('#ts-ic-ch [data-lever]',rail).forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.lever===chPick)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#ts-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':S.view);
/* broadcasting: axis boxes aligned right, then the small arrays with ghost copies */
function axCell(v,stretch,miss,bad,who){
  if(miss&&stretch) return `<span class="ts-ax ts-ghost" data-tipx="${esc(tx(`${who} has no axis here: it counts as 1 and is read ${fint(stretch)} times.`,`${who} 在这里没有轴：算作 1，被读 ${fint(stretch)} 次。`))}" tabindex="0">(1) → ${stretch}</span>`;
  if(miss) return `<span class="ts-ax ts-none" data-tipx="${esc(tx(`${who} has no axis here; it counts as 1.`,`${who} 在这里没有轴，算作 1。`))}" tabindex="0">–</span>`;
  if(stretch) return `<span class="ts-ax ts-ghost" data-tipx="${esc(tx(`${who} has size 1 on this axis: the same value is read ${fint(stretch)} times, not copied.`,`${who} 在这个轴上大小为 1：同一个值被读 ${fint(stretch)} 次，不复制。`))}" tabindex="0">1 → ${stretch}</span>`;
  return `<span class="ts-ax${bad?' ts-bad':''}" data-tipx="${esc(bad?tx(`${who} has ${fint(v)} on this axis, and the other array a different size that is not 1.`,`${who} 在这个轴上是 ${fint(v)}，另一边的数组是不同的大小，也不是 1。`):tx(`${who} has ${fint(v)} ${v===1?'entry':'entries'} along this axis.`,`${who} 沿这个轴有 ${fint(v)} 项。`))}" tabindex="0">${v}</span>`;
}
const SMALL=s=>s.every(v=>v<=6);
function gridHTML(vals,R,C,realR,realC,cap){
  let h=`<div class="ts-gw"><div class="ts-gcap">${cap}</div><div class="ts-grid" style="grid-template-columns:repeat(${C},minmax(0,auto))">`;
  for(let i=0;i<R;i++) for(let j=0;j<C;j++){const real=i<realR&&j<realC, v=vals[realR===1?0:i][realC===1?0:j];
    h+=`<span class="ts-c${real?'':' ts-ghost'}">${v}</span>`;}
  return h+`</div></div>`;
}
function drawBC(){
  const a=parseShape(S.a), b=parseShape(S.b), r=bshape(a,b), hid=hideOut(), box=$('#ts-bc',app);
  const n=r.n, lab=(t,s)=>`<span class="ts-rl">${t} <span class="mono">${s}</span></span>`;
  let h=`<div class="ts-cap">${tx(`a ${OP()} b, shapes lined up from the right`,`a ${OP()} b：形状从右边对齐`)}</div><div class="ts-axes" style="grid-template-columns:minmax(0,auto) repeat(${n},minmax(0,1fr))">`;
  h+=`<span></span>`+r.ax.map(c=>`<span class="ts-axh">${c.k===0?tx('last axis','最后一个轴'):tx(`${c.k+1}th from right`,`右数第 ${c.k+1} 个`).replace('2th','2nd').replace('3th','3rd')}</span>`).join('');
  h+=lab('a',shp(a))+r.ax.map(c=>axCell(c.x,hid?0:(c.sa?c.out:0),c.x==null,c.bad&&!hid,'a')).join('');
  h+=lab('b',shp(b))+r.ax.map(c=>axCell(c.y,hid?0:(c.sb?c.out:0),c.y==null,c.bad&&!hid,'b')).join('');
  h+=`<span class="ts-rl ts-rout">a ${OP()} b</span>`+r.ax.map(c=>hid?`<span class="ts-ax ts-q">?</span>`:(c.bad?`<span class="ts-ax ts-bad ts-x" tabindex="0" data-tipx="${esc(tx(`${fint(c.x)} and ${fint(c.y)}: neither is 1, so these axes cannot be lined up.`,`${fint(c.x)} 和 ${fint(c.y)}：两个都不是 1，这两个轴对不上。`))}">${c.x} ≠ ${c.y}</span>`:`<span class="ts-ax ts-out">${c.out}</span>`)).join('');
  h+=`</div>`;
  if(!hid&&!r.ok) h+=`<p class="ts-err">${tx(`Error on ${axisName(r.err.k)}: ${r.err.x} and ${r.err.y}. NumPy: “operands could not be broadcast together with shapes ${npShp(a)} ${npShp(b)}”.`,`${axisName(r.err.k)}出错：${r.err.x} 和 ${r.err.y}。NumPy 报错“operands could not be broadcast together with shapes ${npShp(a)} ${npShp(b)}”。`)}</p>`;
  else if(!hid) h+=`<p class="ts-res">${tx(`Result: ${shp(r.out)}, ${fint(prod(r.out))} numbers.`,`结果：${shp(r.out)}，${fint(prod(r.out))} 个数值。`)}</p>`;
  /* the arrays with their numbers, when they are small */
  if(SMALL(a)&&SMALL(b)){
    const A2=arrA(a), B2=arrB(b), b2=b.length===2?b:[1,b[0]];
    if(!hid&&r.ok){const [R,C]=r.out, sum=bsum(a,b);
      h+=`<div class="ts-nums">`+gridHTML(A2,R,C,a[0],a[1],'a')+`<span class="ts-op">+</span>`+gridHTML(B2,R,C,b2[0],b2[1],'b')+`<span class="ts-op">=</span>`+gridHTML(sum,R,C,R,C,'a + b')+`</div>`;}
    else h+=`<div class="ts-nums">`+gridHTML(A2,a[0],a[1],a[0],a[1],'a')+`<span class="ts-op">+</span>`+gridHTML(B2,b2[0],b2[1],b2[0],b2[1],'b')+(hid?`<span class="ts-op">=</span><div class="ts-gw"><div class="ts-gcap">a + b</div><div class="ts-qbox">?</div></div>`:'')+`</div>`;
  }
  if(box._h!==h){box.innerHTML=h; box._h=h;}
}
/* A2: what each letter of each string stands for: a short name under the letter and a full sentence on hover */
function letterInfo(id,c,n){const N=fint(n);
  const att={b:[tx('batch','batch'),tx(`b is the batch axis: how many sequences are computed together. Here there is ${N}.`,`b 是 batch 这个轴，表示一起计算的序列有几段。这里只有 ${N} 段。`)],
    h:[tx('head','head'),tx(`h is the head axis: ${N} heads, each computed on its own.`,`h 是 head 这个轴：一共 ${N} 个 head，各算各的。`)],
    q:[tx('query','query'),tx(`q is the token position of the query: ${N} positions.`,`q 是 query 所在的 token 位置，一共 ${N} 个。`)],
    k:id==='ov'?[tx('key','key'),tx(`k is the token position of the value: ${N} positions.`,`k 是 value 所在的 token 位置，一共 ${N} 个。`)]
               :[tx('key','key'),tx(`k is the token position of the key: ${N} positions.`,`k 是 key 所在的 token 位置，一共 ${N} 个。`)],
    d:[tx('dims','维度'),id==='ov'?tx(`d is the dimension of each value vector in a head: ${N} numbers.`,`d 是每个 head 里 value 向量的维度，${N} 维。`):tx(`d is the dimension of each query and key vector in a head: ${N} numbers.`,`d 是每个 head 里 query 和 key 向量的维度，${N} 维。`)]};
  const mat={i:[tx('row','行'),tx(`i is the row of input 1 and of the output: ${N} rows.`,`i 是输入 1 和输出的行：一共 ${N} 行。`)],
    j:[tx('column','列'),tx(`j is the column of input 2 and of the output: ${N} columns.`,`j 是输入 2 和输出的列：一共 ${N} 列。`)],
    k:[tx('shared','共有'),tx(`k is input 1’s column and input 2’s row: ${N} of each, multiplied pairwise and added up.`,`k 是输入 1 的列，也是输入 2 的行：各 ${N} 个，两两相乘再加起来。`)]};
  const bm={b:[tx('pair','第几对'),tx(`b says which of the ${N} pairs of matrices: each pair is multiplied on its own.`,`b 表示第几对矩阵：一共 ${N} 对，每一对单独相乘。`)],
    i:[tx('row','行'),tx(`i is the row of each first matrix and of each result: ${N} rows.`,`i 是每个第一个矩阵和每个结果的行：一共 ${N} 行。`)],
    j:[tx('shared','共有'),tx(`j is the first matrix’s column and the second matrix’s row: ${N} of each, multiplied pairwise and added up.`,`j 是第一个矩阵的列，也是第二个矩阵的行：各 ${N} 个，两两相乘再加起来。`)],
    k:[tx('column','列'),tx(`k is the column of each second matrix and of each result: ${N} columns.`,`k 是每个第二个矩阵和每个结果的列：一共 ${N} 列。`)]};
  const t=(id==='sc'||id==='ov'?att:(id==='bm'?bm:mat))[c]; return t?{s:t[0],t:t[1]}:{s:'',t:''};}
function drawES(){
  const r=esR(S.es), box=$('#ts-es',app), mm=S.es==='mm';
  const has=(s,c)=>s.includes(c), cell=(s,c)=>has(s,c)?`<td class="mono">${fint(r.sizes[c])}</td>`:`<td class="ts-no">–</td>`;
  const th=c=>{const li=letterInfo(S.es,c,r.sizes[c]); return `<th class="${r.summed.includes(c)?'ts-sum':''}" tabindex="0" data-tipx="${esc(li.t)}"><span class="ts-lchar">${c}</span><span class="ts-lname">${esc(li.s)}</span></th>`;};
  let h=`<div class="ts-cap">einsum('${esc(r.spec)}')${tx(': ','：')}${tx('one column per letter','每个字母一列')}</div><div class="ts-ltw"><table class="ts-lt"><thead><tr><th></th>${r.cols.map(th).join('')}</tr></thead><tbody>`;
  h+=`<tr><th>${tx('input 1','输入 1')} <span class="mono">${r.x}</span></th>${r.cols.map(c=>cell(r.x,c)).join('')}</tr>`;
  h+=`<tr><th>${tx('input 2','输入 2')} <span class="mono">${r.y}</span></th>${r.cols.map(c=>cell(r.y,c)).join('')}</tr>`;
  h+=`<tr class="ts-out"><th>${tx('output','输出')} <span class="mono">${r.z}</span></th>${r.cols.map(c=>r.summed.includes(c)?`<td class="ts-x" data-l="${c}" tabindex="0" data-tipx="${esc(tx(`${c} is in the inputs but not after the arrow, so the output has no ${c} axis: each output number adds up ${fint(r.sizes[c])} products along ${c}.`,`${c} 在输入里有，箭头后面没有，所以输出没有 ${c} 这个轴：每个输出数值都是沿 ${c} 的 ${fint(r.sizes[c])} 个乘积加起来的。`))}">${tx('summed away','求和后去掉').split(' ').map(w=>`<span>${w}</span>`).join(' ')}</td>`:`<td class="mono">${fint(r.sizes[c])}</td>`).join('')}</tr></tbody></table></div>`;
  h+=`<p class="ts-res">${tx(`Shapes ${shp(r.shapeA)} and ${shp(r.shapeB)} give ${shp(r.outShape)}: ${fint(r.perOut)} multiply-adds for each of ${r.outN>=1e6?big(r.outN):fint(r.outN)} output numbers.`,`形状 ${shp(r.shapeA)} 和 ${shp(r.shapeB)} 得到 ${shp(r.outShape)}：${r.outN>=1e6?big(r.outN):fint(r.outN)} 个输出数值，每个要 ${fint(r.perOut)} 次乘加。`)}</p>`;
  if(mm){const g=(M,cap,tip)=>`<div class="ts-gw"><div class="ts-gcap">${cap}</div><div class="ts-grid" style="grid-template-columns:repeat(${M[0].length},minmax(0,auto))">${M.map((row,i)=>row.map((v,j)=>tip?`<span class="ts-c ts-out" tabindex="0" data-tipx="${esc(tip(i,j))}">${v}</span>`:`<span class="ts-c">${v}</span>`).join('')).join('')}</div></div>`;
    h+=`<div class="ts-nums">`+g(MA,tx('input 1 (i, k): 4 × 3','输入 1 (i, k)：4 × 3'))+`<span class="ts-op">@</span>`+g(MB,tx('input 2 (k, j): 3 × 2','输入 2 (k, j)：3 × 2'))+`<span class="ts-op">=</span>`+
      g(MC,tx('output (i, j): 4 × 2','输出 (i, j)：4 × 2'),(i,j)=>tx(`Row ${i} times column ${j}, summed over k: ${MA[i].map((v,k)=>`${v} × ${MB[k][j]}`).join(' + ')} = ${MC[i][j]}.`,`第 ${i} 行乘第 ${j} 列，对 k 求和：${MA[i].map((v,k)=>`${v} × ${MB[k][j]}`).join(' + ')} = ${MC[i][j]}。`))+`</div>`;}
  if(box._h!==h){box.innerHTML=h; box._h=h;}
}
/* 1-NN: rows on one log axis, 1 KiB to 1 PiB; the H100 line; red only when an array does not fit */
const LG0=10, LG1=50, xOf=b=>Math.max(0,Math.min(1,(Math.log2(Math.max(b,KiB))-LG0)/(LG1-LG0)));
function nnRows(){
  if(viewNow()==='ch'){const ids=[CH.start].concat(chPick!==CH.start?[chPick]:[]);
    return ids.map(id=>{const e=chEval(id); return {id,label:L(CHT[id]).replace(/ \(initial setup\)|（初始设置）/,''),code:L(CHD[id]),bytes:e.bytes,iters:e.iters,tag:id===chPick?tx('your pick','你的选择'):tx('initial setup','初始设置')};});}
  const r=nn(S.N,S.M,S.D);
  return [{id:'loop',label:tx('Double loop','两层循环'),code:L(CHD.loop),bytes:r.loop.bytes,iters:r.loop.iters,shape:'(D,)'},
    {id:'bc',label:tx('One broadcast subtraction','一次广播减法'),code:'((X[:, None] - Y[None])**2).sum(-1)',bytes:r.bc.bytes,iters:0,shape:'(N, M, D)'},
    {id:'mm',label:tx('Matmul expansion','矩阵乘法展开'),code:'(X**2).sum(1)[:, None] - 2 * X @ Y.T + (Y**2).sum(1)',bytes:r.mm.bytes,iters:0,shape:'(N, M)'}];
}
/* A3: what each row's code builds, in its tooltip */
function nnHow(id){
  if(id==='loop') return tx('For each pair it builds one (D,) difference, adds up its squares and keeps the smallest distance so far.','它对每一对向量建一个 (D,) 的差值，把平方加起来，只保留目前最小的距离。');
  if(id==='bc') return tx('X[:, None] has shape (N, 1, D) and Y[None] has shape (1, M, D), so the difference has shape (N, M, D), every difference of every pair at once.','X[:, None] 的形状是 (N, 1, D)，Y[None] 是 (1, M, D)，所以差值的形状是 (N, M, D)，一次放下每一对的每一个差值。');
  if(id==='mm') return tx('X @ Y.T has shape (N, M), one dot product per pair, and the two row norms, (N, 1) and (1, M), broadcast to it.','X @ Y.T 的形状是 (N, M)，每一对一个点积；两个行范数 (N, 1) 和 (1, M) 广播成同样的形状。');
  if(id==='c1'||id==='c32'||id==='c256'){const C=chEval(id).C; return tx(`Each run takes ${C} ${C===1?'query':'queries'} and broadcasts them against all M stored vectors: an array of shape (${C}, M, D).`,`每一轮取 ${C} 个查询，和全部 M 个存好的向量做广播：数组的形状是 (${C}, M, D)。`);}
  return '';
}
/* A3, R6.25: the inserted axes of the broadcast version, drawn as part 3's boxes; each column is named by what it
   indexes, and a small table shows the pairs the broadcast makes (round 3: why None goes in the middle) */
function nnShapeHTML(){
  const box=(t,cls,tip)=>`<span class="ts-ax${cls?' '+cls:''}" tabindex="0" data-tipx="${esc(tip)}">${t}</span>`;
  const rl=(code,sh)=>`<span class="ts-rl"><code class="ts-code1">${code}</code> <span class="mono">${sh}</span></span>`;
  const n=tx(`N = ${fint(S.N)} test vectors.`,`N = ${fint(S.N)} 个测试向量。`), m=tx(`M = ${fint(S.M)} training vectors.`,`M = ${fint(S.M)} 个训练向量。`), d=tx(`D = ${fint(S.D)} numbers per vector.`,`D = ${fint(S.D)}，每个向量的维数。`);
  const head=[tx('which test vector','哪个测试向量'),tx('which training vector','哪个训练向量'),tx('which dimension','第几维')];
  const v=(c,i)=>A.ui.math(`${c}_{${i}}`), cell=(i,j)=>`<span class="ts-pc">${A.ui.math(`x_{${i}} - y_{${j}}`)}</span>`;   // typeset (R6.29, H1 2026-10-09)
  let pg=`<span class="ts-ph"></span>`+[1,2,3].map(j=>`<span class="ts-ph">${v('y',j)}</span>`).join('')+`<span class="ts-ph">⋯</span>`;
  [1,2,3].forEach(i=>{pg+=`<span class="ts-ph">${v('x',i)}</span>`+[1,2,3].map(j=>cell(i,j)).join('')+`<span class="ts-pc ts-pe">⋯</span>`;});
  pg+=`<span class="ts-ph">⋮</span>`+`<span class="ts-pc ts-pe">⋮</span>`.repeat(3)+`<span class="ts-pc ts-pe">⋱</span>`;
  return `<div class="ts-nnshape"><div class="ts-sub2">${tx('The broadcast version, lined up from the right. None inserts an axis of size 1 (dashed); where it goes decides which rows pair up.','广播写法，从右边对齐。None 插入一个大小为 1 的轴（虚线）；插在哪里，决定了谁和谁配对。')}</div>`+
    `<div class="ts-axes" style="grid-template-columns:minmax(0,auto) repeat(3,minmax(0,1fr))"><span></span>`+head.map(t=>`<span class="ts-axh">${t}</span>`).join('')+
    rl('X[:, None]','(N, 1, D)')+box('N','',n)+box('1 → M','ts-ghost',tx(`X[:, None] puts its size-1 axis in the middle, the place of the training vectors: each row of X is read once for each of the ${fint(S.M)} rows of Y.`,`X[:, None] 把大小为 1 的轴放在中间，也就是训练向量的位置：X 的每一行都对 Y 的 ${fint(S.M)} 行各读一次。`))+box('D','',d)+
    rl('Y[None]','(1, M, D)')+box('1 → N','ts-ghost',tx(`Y[None] puts its size-1 axis in front, the place of the test vectors: each row of Y is read once for each of the ${fint(S.N)} rows of X.`,`Y[None] 把大小为 1 的轴放在最前面，也就是测试向量的位置：Y 的每一行都对 X 的 ${fint(S.N)} 行各读一次。`))+box('M','',m)+box('D','',d)+
    `<span class="ts-rl ts-rout">${tx('difference','差值')} <span class="mono">(N, M, D)</span></span>`+box('N','ts-out',n)+box('M','ts-out',m)+box('D','ts-out',tx(`D = ${fint(S.D)}: this axis is summed only after the whole (N, M, D) array exists.`,`D = ${fint(S.D)}：要等整个 (N, M, D) 数组建好，才对这个轴求和。`))+`</div>`+
    `<div class="ts-pair"><div class="ts-pgrid" role="img" aria-label="${esc(tx('Cell (i, j) of the difference is x_i minus y_j.','差值的第 (i, j) 格是 x_i 减 y_j。'))}">${pg}</div>`+
    `<p class="ts-pcap small">${tx(`Cell (i, j) of the difference is ${A.ui.math('x_i - y_j')}: one row of X minus one row of Y, ${fint(S.D)} numbers. There are ${A.ui.math('N \\times M')} such cells.`,`差值的第 (i, j) 格是 ${A.ui.math('x_i - y_j')}：X 的一行减 Y 的一行，有 ${fint(S.D)} 个数值。这样的格一共有 ${A.ui.math('N \\times M')} 个。`)}</p></div></div>`;
}
/* the axis picture sits above the bars from part 5 on; the bars' legend then moves down to the bars (review v6 TS-S7) */
const nnPic=()=>viewNow()==='nn'&&!(mode==='guided'&&step===0);
function drawNN(){
  const box=$('#ts-nn',app), rows=nnRows(), ch=viewNow()==='ch', hx=xOf(H100)*100;
  const N=ch?CH.N:S.N, M=ch?CH.M:S.M, D=ch?CH.D:S.D;
  let h=`<div class="ts-cap">${tx(`Largest intermediate array of each way, N ${fint(N)}, M ${fint(M)}, D ${fint(D)}, FP32 (log scale)`,`每种写法的最大中间数组：N ${fint(N)}，M ${fint(M)}，D ${fint(D)}，FP32（对数刻度）`)}</div>`+
    `<div class="ts-sub2 ts-xy">${ch?tx(`X holds the ${fint(N)} queries, shape (N, D); Y holds the ${fint(M)} stored vectors, shape (M, D).`,`X 是 ${fint(N)} 个查询，形状 (N, D)；Y 是 ${fint(M)} 个存好的向量，形状 (M, D)。`)
      :tx(`X holds the ${fint(N)} test vectors, shape (N, D); Y holds the ${fint(M)} training vectors, shape (M, D).`,`X 是 ${fint(N)} 个测试向量，形状 (N, D)；Y 是 ${fint(M)} 个训练向量，形状 (M, D)。`)}</div>`;
  if(nnPic()) h+=nnShapeHTML()+`<div class="legend-row ts-nnleg">${legendHTML()}</div>`;
  h+=`<div class="ts-nnrows">`;
  rows.forEach(x=>{const over=!fits(x.bytes), tiny=x.bytes<KiB, w=xOf(x.bytes)*100, beyond=x.bytes>2**LG1;
    const tip=tx(`${x.label}: ${nnHow(x.id)} Its largest array holds ${fBB(x.bytes)}, ${over?'more than':'within'} the 80 GiB of one H100.${tiny?' That is under 1 KiB, where the axis starts, so its bar is a stub.':''} ${x.iters?`Its Python loop runs ${x.iters>=1e6?big(x.iters):fint(x.iters)} times.`:'It has no Python loop.'}`,
                 `${x.label}：${nnHow(x.id)}最大的数组 ${fBB(x.bytes)}，${over?'超过':'没超过'}一张 H100 的 80 GiB。${tiny?'它小于刻度最左端的 1 KiB，所以只画了一小截。':''}${x.iters?`它的 Python 循环跑 ${x.iters>=1e6?big(x.iters):fint(x.iters)} 次。`:'它没有 Python 循环。'}`);
    h+=`<div class="ts-nnr${over?' ts-over':''}" data-id="${x.id}" tabindex="0" data-tipx="${esc(tip)}"><div class="ts-nnh"><b>${esc(x.label)}</b>${x.tag?`<span class="cmp-tag">${esc(x.tag)}</span>`:''}${x.shape?`<span class="mono ts-shp">${x.shape}</span>`:''}</div>`+
      `<code class="ts-code1">${esc(x.code)}</code>`+
      `<div class="ts-trk"><i class="ts-bar" style="width:${tiny?'3px':w.toFixed(2)+'%'}"></i><b class="ts-hl" style="left:${hx.toFixed(2)}%"></b><span class="ts-bv mono${w>50?' ts-in':''}" style="left:${tiny?'3px':w.toFixed(2)+'%'}">${beyond?'▸ ':''}${fB(x.bytes)}${tiny?tx(', under 1 KiB','，小于 1 KiB'):''}${over?tx(', does not fit','，放不下'):''}</span></div>`+
      `<div class="ts-it small">${x.iters?tx(`Python loop: ${x.iters>=1e6?big(x.iters):fint(x.iters)} runs`,`Python 循环：${x.iters>=1e6?big(x.iters):fint(x.iters)} 次`):tx('no Python loop','没有 Python 循环')}</div></div>`;});
  h+=`<div class="ts-axis" aria-hidden="true">${['1 KiB','1 MiB','1 GiB','1 TiB','1 PiB'].map((t,i)=>`<span style="left:${(i*25)}%">${t}</span>`).join('')}<span class="ts-hlab" style="left:${hx.toFixed(2)}%">${tx('one H100: 80 GiB','一张 H100：80 GiB')}</span></div></div>`;
  if(box._h!==h){box.innerHTML=h; box._h=h;}
}
function drawStage(){
  const v=viewNow(), isNN=v==='nn'||v==='ch';
  $('#ts-bc',app).hidden=v!=='bc'; $('#ts-es',app).hidden=v!=='es'; $('#ts-nn',app).hidden=!isNN;
  if(v==='bc') drawBC(); else if(v==='es') drawES(); else drawNN();
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chPick); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Both targets met','两个目标都达到'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the largest array fits in 80 GiB and the Python loop runs at most 1,000 times. Now: ${fB(e.bytes)} and ${itersTxt(e.iters)}.`,`最大的数组不超过 80 GiB、Python 循环不超过 1,000 次时，徽章是绿色。现在是 ${fB(e.bytes)}，${itersTxt(e.iters)}。`)};}
  if(v==='bc'){if(hideOut()) return pred?{cls:'neutral',txt:tx('Answer saved','已记下'),short:tx('Saved','已记下'),tip:tx('Your answer is saved; part 3 shows the result.','你的答案已记下；第 3 部分给出结果。')}:{cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The result is hidden until part 3, after your prediction.','结果先隐藏，你预测之后，第 3 部分再给出。')};
    const r=bshape(parseShape(S.a),parseShape(S.b));
    return r.ok?{cls:'neutral',txt:tx(`Result ${shp(r.out)}`,`结果 ${shp(r.out)}`),short:shp(r.out),tip:tx(`a ${shp(r.a)} ${OP()} b ${shp(r.b)} broadcast to ${shp(r.out)}.`,`a ${shp(r.a)} ${OP()} b ${shp(r.b)} 广播成 ${shp(r.out)}。`)}
      :{cls:'bad',txt:tx('Error','报错'),short:tx('Error','报错'),tip:tx(`a ${shp(r.a)} and b ${shp(r.b)} do not broadcast: ${r.err.x} and ${r.err.y} on ${axisName(r.err.k)}.`,`a ${shp(r.a)} 和 b ${shp(r.b)} 不能广播：${axisName(r.err.k)}上是 ${r.err.x} 和 ${r.err.y}。`)};}
  if(v==='es'){const r=esR(S.es); return {cls:'neutral',txt:tx(`Sums ${r.summed.join(', ')}`,`对 ${r.summed.join('、')} 求和`),short:tx(`Sums ${r.summed.join(', ')}`,`对 ${r.summed.join('、')} 求和`),tip:tx(`In ${r.spec}, ${r.summed.join(', ')} ${r.summed.length===1?'is':'are'} in the inputs and not in the output, so ${r.summed.length===1?'it is':'they are'} summed.`,`在 ${r.spec} 里，${r.summed.join('、')} 在输入里、不在输出里，所以被求和。`)};}
  const r=nn(S.N,S.M,S.D), f=fits(r.bc.bytes);
  return {cls:f?'neutral':'bad',txt:f?tx('Broadcast fits','广播放得下'):tx('Broadcast too big','广播放不下'),short:f?tx('Fits','放得下'):tx('Too big','放不下'),
    tip:tx(`The broadcast difference needs ${fB(r.bc.bytes)}; one H100 holds 80 GiB. The matmul expansion needs ${fB(r.mm.bytes)}.`,`广播出来的差值要 ${fB(r.bc.bytes)}；一张 H100 有 80 GiB。矩阵乘法展开要 ${fB(r.mm.bytes)}。`)};
}
function drawHead(){
  const v=viewNow(), t=$('#ts-st-title',app), sub=$('#ts-st-sub',app), st=$('#ts-status',app);
  t.textContent=v==='ch'?tx('The challenge: 10,000 queries, 200,000 stored vectors','挑战题：10,000 个查询，200,000 个存好的向量'):(v==='bc'?tx(`Broadcasting: a ${OP()} b`,`广播：a ${OP()} b`):(v==='es'?tx('einsum: letters name the axes','einsum：字母给轴命名'):tx('1-NN: three ways to write it','1-NN：三种写法')));
  sub.textContent=v==='bc'?tx(`a ${shK(S.a)}, b ${shK(S.b)}`,`a ${shK(S.a)}，b ${shK(S.b)}`):(v==='es'?esOf(S.es).spec:(v==='ch'?tx('D 768, FP32, one H100','D 768，FP32，一张 H100'):tx(`N ${fint(S.N)}, M ${fint(S.M)}, D ${fint(S.D)}, FP32`,`N ${fint(S.N)}，M ${fint(S.M)}，D ${fint(S.D)}，FP32`)));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const vw=$('#ts-view',app); vw.hidden=mode!=='sandbox'; $$('button',vw).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===S.view)));
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`, v=viewNow();
  if(v==='bc') return it('ts-sw-ax',tx('box: one axis and its size','方框：一个轴和它的大小'),tx('Each box is one axis of the array; the number is its size. Rows are lined up from the right.','每个方框是数组的一个轴，里面的数是它的大小。各行从右边对齐。'))+
    it('ts-sw-gh',tx('dashed: stretched, read again','虚线：被拉伸，重复读取'),tx('A dashed box or number is a size-1 (or missing) axis stretched to the other size: the same value read again, not a copy.','虚线的方框或数，是大小为 1（或缺失）的轴被拉伸成另一边的大小：重复读同一个值，不是副本。'))+
    it('ts-sw-bad',tx('red: sizes that do not match','红色：大小对不上'),tx('Red marks the axis where the two sizes differ and neither is 1: NumPy raises an error.','红色标出两个大小不同、又都不是 1 的那个轴：NumPy 会报错。'));
  if(v==='es') return it('ts-sw-col',tx('column: one letter, one axis','列：一个字母，一个轴'),tx('Each column is one letter of the string; a row shows that array’s size for it, or “–” if the array does not have that axis.','每一列是字符串里的一个字母；每一行写出那个数组在这个轴上的大小，没有这个轴就写“–”。'))+
    it('ts-sw-sum',tx('grey with a slash: summed away','灰格加斜线：求和后去掉'),tx('A grey cell with a slash is a letter that is in the inputs but not after the arrow: it is summed, and the output has no such axis.','灰格加斜线的那一列，字母在输入里有、箭头后面没有：它被求和，输出里没有这个轴。'));
  return it('ts-sw-bar',tx('bar: largest array, log scale','横条：最大的数组，对数刻度'),tx('Each bar is the largest intermediate array that way of writing creates. The axis is logarithmic: each mark is 1,024 times the one before.','每根横条是这种写法创建的最大中间数组。刻度是对数的：每一格是前一格的 1,024 倍。'))+
    it('ts-sw-hl',tx('thick line: one H100, 80 GiB','粗线：一张 H100，80 GiB'),tx('The thick vertical line is the memory of one H100, 80 GB read as 80 GiB.','粗竖线是一张 H100 的显存，80 GB 按 80 GiB 计。'))+
    it('ts-sw-over',tx('red: does not fit','红色：放不下'),tx('A red bar is an array larger than one H100’s memory: that way of writing cannot run on it.','红色的横条比一张 H100 的显存还大：这种写法在它上面跑不起来。'));
}
/* metric cards: ▲▼ only after the learner's own change (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const unitOf=v=>String(v).replace(/^[\d.,]+\s*/,'');
function metricsDef(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chPick);
    return [{k:'big',label:tx('Largest array','最大的数组'),v:fB(e.bytes),sub:tx('goal: at most 80 GiB','目标：不超过 80 GiB'),tip:tx(`The largest intermediate array of “${L(CHT[e.id])}”: ${fB(e.bytes)} (${fGB(e.bytes)}).`,`“${L(CHT[e.id])}”最大的中间数组：${fB(e.bytes)}（${fGB(e.bytes)}）。`)},
      {k:'it',label:tx('Python loop runs','Python 循环次数'),v:e.iters>=1e6?big(e.iters):fint(e.iters),sub:tx('goal: at most 1,000','目标：不超过 1,000'),tip:tx(`How many times the Python loop of “${L(CHT[e.id])}” runs: ${fint(e.iters)}.`,`“${L(CHT[e.id])}”的 Python 循环跑 ${fint(e.iters)} 次。`)},
      {k:'wr',label:tx('Written in total','一共写出'),v:fB(e.written),sub:e.id==='mm'?tx(`only the ${A.ui.math('N \\times M')} dot products`,`只有 ${A.ui.math('N \\times M')} 个点积`):tx(`every ${A.ui.math('N \\times M \\times D')} difference, over all runs`,`全部 ${A.ui.math('N \\times M \\times D')} 个差值，所有轮次加起来`),
        tip:tx(`Every way except the expansion writes all ${A.ui.math('N \\times M \\times D')} differences, one pair or one chunk at a time: ${fB(chEval('bc').written)} in total, however small its largest array is. The expansion writes only the ${A.ui.math('N \\times M')} dot products, ${fB(chEval('mm').written)}.`,
               `除了展开的写法，每种写法都要写出全部 ${A.ui.math('N \\times M \\times D')} 个差值，一次一对或一次一块：一共 ${fB(chEval('bc').written)}，不管它最大的数组有多小。展开的写法只写出 ${A.ui.math('N \\times M')} 个点积，${fB(chEval('mm').written)}。`)}];}
  if(v==='bc'){const r=bshape(parseShape(S.a),parseShape(S.b)), hid=hideOut();
    const st=r.ok?r.ax.filter(c=>c.sa||c.sb).length:0;
    return [{k:'out',label:tx('Result shape','结果形状'),v:hid?'?':(r.ok?shp(r.out):tx('error','报错')),sub:`a ${OP()} b`,tip:hid?tx('Hidden until part 3, after your prediction.','先隐藏，你预测之后，第 3 部分再给出。'):(r.ok?tx(`a ${OP()} b has shape ${shp(r.out)}.`,`a ${OP()} b 的形状是 ${shp(r.out)}。`):tx(`a ${shp(r.a)} and b ${shp(r.b)} do not broadcast.`,`a ${shp(r.a)} 和 b ${shp(r.b)} 不能广播。`))},
      {k:'n',label:tx('Numbers in the result','结果里的数'),v:hid||!r.ok?'–':fint(prod(r.out)),sub:tx('a new array','一个新数组'),tip:hid?tx('Hidden until part 3, after your prediction.','先隐藏，你预测之后，第 3 部分再给出。'):(r.ok?tx(`The result is a new array of ${fint(prod(r.out))} numbers; a and b stay as they are, with ${fint(prod(r.a))} and ${fint(prod(r.b))} numbers.`,`结果是一个新数组，有 ${fint(prod(r.out))} 个数值；a 和 b 保持原样，分别有 ${fint(prod(r.a))} 个和 ${fint(prod(r.b))} 个数值。`):tx('No result: the shapes do not broadcast.','没有结果：形状不能广播。'))},
      {k:'st',label:tx('Axes stretched','被拉伸的轴'),v:hid||!r.ok?'–':String(st),sub:tx('read again, not copied','重复读取，不复制'),tip:hid?tx('Hidden until part 3, after your prediction.','先隐藏，你预测之后，第 3 部分再给出。'):!r.ok?tx('No result: the shapes do not broadcast.','没有结果：形状不能广播。'):tx(`${st} ${st===1?'axis is':'axes are'} stretched from 1 (or a missing axis) to the other size.`,`有 ${st} 个轴从 1（或缺失）被拉伸成另一边的大小。`)}];}
  if(v==='es'){const r=esR(S.es);
    return [{k:'out',label:tx('Output shape','输出形状'),v:shp(r.outShape),sub:r.z,tip:tx(`The output keeps the letters after the arrow, ${r.z}: shape ${shp(r.outShape)}.`,`输出保留箭头后面的字母 ${r.z}：形状 ${shp(r.outShape)}。`)},
      {k:'sum',label:tx('Summed letters','被求和的字母'),v:r.summed.join(', '),sub:tx('in the inputs, not the output','输入有、输出没有'),tip:tx(`${r.summed.join(', ')}: in the inputs and missing from the output.`,`${r.summed.join('、')}：输入里有，输出里没有。`)},
      {k:'mac',label:tx('Multiply-adds','乘加次数'),v:r.macs>=1e6?big(r.macs):fint(r.macs),sub:tx('product of every letter’s size','每个字母的大小相乘'),tip:tx(`One multiply-add for every combination of all the letters: ${r.cols.map(c=>fint(r.sizes[c])).join(' × ')} = ${fint(r.macs)}.`,`所有字母的每一种组合各一次乘加：${r.cols.map(c=>fint(r.sizes[c])).join(' × ')} = ${fint(r.macs)}。`)}];}
  const r=nn(S.N,S.M,S.D);
  return [{k:'bc',label:tx('Broadcast: largest array','广播：最大的数组'),v:fB(r.bc.bytes),sub:'(N, M, D)',tip:tx(`The broadcast difference holds ${A.ui.math('N \\times M \\times D')} = ${fint(S.N*S.M*S.D)} numbers of 4 bytes: ${fB(r.bc.bytes)}.`,`广播出来的差值有 ${A.ui.math('N \\times M \\times D')} = ${fint(S.N*S.M*S.D)} 个数值，每个 4 字节：${fB(r.bc.bytes)}。`)},
    {k:'mm',label:tx('Matmul: largest array','矩阵乘法：最大的数组'),v:fB(r.mm.bytes),sub:'(N, M)',tip:tx(`The matmul expansion holds ${A.ui.math('N \\times M')} = ${fint(S.N*S.M)} numbers of 4 bytes: ${fB(r.mm.bytes)}.`,`矩阵乘法展开有 ${A.ui.math('N \\times M')} = ${fint(S.N*S.M)} 个数值，每个 4 字节：${fB(r.mm.bytes)}。`)},
    {k:'it',label:tx('Double loop: Python runs','两层循环：Python 迭代'),v:big(r.loop.iters),sub:A.ui.math('N \\times M'),tip:tx(`The double loop runs its body ${A.ui.math('N \\times M')} = ${fint(r.loop.iters)} times.`,`两层循环的循环体要跑 ${A.ui.math('N \\times M')} = ${fint(r.loop.iters)} 次。`)}];
}
function drawMetrics(){
  const host=$('#ts-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join()+viewNow();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.classList.toggle('ts-long',String(m.v).length>11); {const sb=el.querySelector('.sub'); if(/<math/.test(m.sub)) sb.innerHTML=m.sub; else sb.textContent=m.sub;} el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&unitOf(prev)===unitOf(m.v)){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&!/\d[MB]$/.test(String(m.v))) dt=A.fmt.delta(a,b,{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows; terms wired to their controls (R1.4) */
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each block is one lead line in
   words, then rows that line up: the quantity's name, its formula, the reader's numbers, the result (R6.29). A cell is TeX, or
   {h:html}; inputs keep their terms (R1.4) */
/* the last cell of a short row spans the rest (not a fixed column), so the rows still line up when the names move above them */
const TMG=rows=>{const n=Math.max(...rows.map(r=>r.length));
  return `<div class="ts-mg" style="--n:${n};--n1:${n-1}">`+rows.map(r=>r.map((c,i)=>{const w=i===r.length-1&&r.length<n;
    return `<span class="ts-mc${i===0?' ts-mn':''}${w?' ts-mw':''}${c&&c.h!=null?' ts-mh':''}"${w?` style="grid-column:span ${n-i}"`:''}>${c==null||c===''?'':(c.h!=null?c.h:A.ui.math(c))}</span>`;}).join('')).join('')+'</div>';};
const TMB=(lead,...grids)=>`<div class="ts-mb"><p class="ts-ml">${lead}</p>${grids.map(g=>typeof g==='string'?g:TMG(g)).join('')}</div>`;
/* a chain grid (name = formula, then a row per “=” step) stacks its names above the steps at narrow widths */
const TMC=rows=>TMG(rows).replace('<div class="ts-mg"','<div class="ts-mg ts-chain"');
const TTX=x=>'\\text{'+String(x)+'}';
function mathHTML(){
  const v=viewNow(), H=h=>({h});
  if(v==='bc'){const r=bshape(parseShape(S.a),parseShape(S.b));
    if(hideOut()) return `<p>${tx('The rule is shown in part 3, after your prediction.','你预测之后，第 3 部分再显示规则。')}</p>`;
    const miss=tx('missing (1)','缺（算 1）');
    return TMB(tx('Compare the axes from the right. Equal sizes stay; a 1, or a missing axis, takes the other size:','从右边逐个比较轴。大小相等的保留；是 1 或者缺这个轴的，取另一边的大小：'),
        [[H(''),H('<b>a</b>'),H('<b>b</b>'),'',H(`<b>${tx('result','结果')}</b>`)]].concat(r.ax.slice().reverse().map(c=>[H(axisName(c.k)),H(c.x==null?miss:String(c.x)),H(c.y==null?miss:String(c.y)),'\\to',H(c.bad?tx('error','报错'):String(c.out))])))+
      TMB(tx('So the result is:','所以结果是：'),[[H(`<b>${r.ok?`${T_('a',shp(r.a))} + ${T_('b',shp(r.b))} → ${shp(r.out)}`:tx(`${shp(r.a)} + ${shp(r.b)}: error`,`${shp(r.a)} + ${shp(r.b)}：报错`)}</b>`)]]);}
  if(v==='es'){const r=esR(S.es);
    return TMB(`${T_('es',esc(r.spec))}${tx(': the size of each letter',' 里每个字母的大小')}`,[r.cols.map(c=>H(`${c} = ${fint(r.sizes[c])}`))])+
      TMB(tx('The letters after the arrow are the output’s axes; every other letter is summed over:','箭头后面的字母是输出的轴，其余字母都被求和掉：'),
        [[TTX(tx('output','输出')),H(r.z||'∅'),'\\to',H(`${shp(r.outShape)} = ${fint(r.outN)} ${tx('numbers','个数值')}`)],
         [TTX(tx('summed','求和')),H(r.summed.join(', ')||'∅'),'\\to',H(`${fint(r.perOut)} ${tx('multiply-adds per output number','次乘加（每个输出数值）')}`)]])+
      TMB(tx('Each multiply-add takes one value of every letter:','每次乘加取每个字母的一个值：'),[[TTX(tx('multiply-adds','乘加次数')),'=',H(r.cols.map(c=>fint(r.sizes[c])).join(' × ')),'=',H(`<b>${fint(r.macs)}</b>`)]]);}
  const ch=v==='ch', N=ch?CH.N:S.N, M=ch?CH.M:S.M, D=ch?CH.D:S.D, tN=ch?fint(N):T_('N',fint(N)), tM=ch?fint(M):T_('M',fint(M)), tD=ch?fint(D):T_('D',fint(D));
  /* each quantity is a short chain broken at “=” (H2): its name and formula, the reader's numbers, the bytes, then the size in GB and GiB */
  const res=b=>{const v=[`${fint(b)} ${tx('bytes','字节')}`].concat(b>=1e6?[fGB(b)]:[]).concat(b>=KiB?[fB(b)]:[]); v[v.length-1]=`<b>${v[v.length-1]}</b>`;
    return [['','=',H(v[0])]].concat(v.length>1?[['','=',H(v.slice(1).join(' = '))]]:[]);};
  const chain=(name,f,nums,b)=>[[name,'=',f],['','=',H(nums)]].concat(res(b));
  let h=TMB(ch?tx(`N = ${tN} queries, M = ${tM} stored vectors, D = ${tD} numbers each. FP32 takes 4 bytes a number; 1 GiB = 1,073,741,824 bytes.`,`N = ${tN} 个查询，M = ${tM} 个存好的向量，每个向量 D = ${tD} 维。FP32 每个数值 4 字节；1 GiB = 1,073,741,824 字节。`)
      :tx(`N = ${tN} test vectors, M = ${tM} training vectors, D = ${tD} numbers each. FP32 takes 4 bytes a number; 1 GiB = 1,073,741,824 bytes.`,`N = ${tN} 个测试向量，M = ${tM} 个训练向量，每个向量 D = ${tD} 维。FP32 每个数值 4 字节；1 GiB = 1,073,741,824 字节。`)+
      tx(' The largest array each way builds:',' 每种写法中途建出的最大数组：'),
    TMC([...chain(TTX(tx('double loop, one difference at a time','两层循环，一次一个差值')),'D \\times 4',`${tD} × 4`,D*F32),
    ...chain(TTX(tx('broadcast, all differences','广播，全部差值')),'N \\times M \\times D \\times 4',`${tN} × ${tM} × ${tD} × 4`,N*M*D*F32),
    ...chain(TTX(tx('matmul expansion, all dot products','矩阵乘法展开，全部点积')),'N \\times M \\times 4',`${tN} × ${tM} × 4`,N*M*F32)]));
  h+=TMB(tx('The work is the same every way; only the double loop runs it one pair at a time in Python:','每种写法的计算量都一样；只有两层循环要在 Python 里一对一对地算：'),
    TMC([[TTX(tx('Python iterations of the double loop','两层循环的 Python 迭代')),'=','N \\times M'],['','=',H(fint(N*M))],
    [TTX(tx('multiply-adds, every way','乘加次数，每种写法都是')),'=','N \\times M \\times D'],['','=',H(fint(N*M*D))],
    [TTX(tx('extra for the norms, expansion only','展开写法另加的范数')),'=','(N + M) \\times D'],['','=',H(fint((N+M)*D))]]));
  if(ch){const e=chEval(chPick); if(e.C) h+=TMB(tx(`In chunks of C = ${e.C} queries, the largest array is one chunk’s differences, and the loop runs once per chunk:`,`每块 C = ${e.C} 个查询时，最大的数组是一块的差值，循环每块跑一次：`),
    TMC([...chain(TTX(tx('one chunk','一块')),'C \\times M \\times D \\times 4',`${fint(e.C)} × ${fint(M)} × ${fint(D)} × 4`,e.bytes),
    [TTX(tx('loop runs','循环次数')),'=','\\left⌈\\frac{N}{C}\\right⌉'],['','=',H(`⌈${fint(N)} ÷ ${fint(e.C)}⌉`)],['','=',H(`<b>${fint(e.iters)}</b>`)]]));}
  return h;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  if(isP('bcast')&&!bshape(parseShape(S.a),parseShape(S.b)).ok) seenErr=true;
  drawHead(); drawStage();
  const gr=$('#ts-graph',app); gr.dataset.focus=mode==='guided'?focusOf(STEPS[step]):'';
  drawMetrics();
  const lg=$('#ts-legend',app), lh=nnPic()?'':legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;} lg.hidden=!lh;
  const mb=$('#ts-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail(); updTerms();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return esc(tx(`In the challenge you choose only the way to write it, in the task card. Everything else is fixed: ${fint(CH.N)} queries, ${fint(CH.M)} stored vectors of ${CH.D} numbers, FP32, one H100.`,`挑战中只能在任务卡里选写法。其余固定：${fint(CH.N)} 个查询，${fint(CH.M)} 个存好的 ${CH.D} 维向量，FP32，一张 H100。`));
  if(mode!=='guided') return '';
  /* A11: "Scenarios" is a link to that mode (R1.10) */
  const sc=`<button type="button" class="linkbtn" data-mode-go="sandbox">${tx('Scenarios','应用场景')}</button>`;
  const id=STEPS[step].id;
  if(id==='bcast') return tx(`In this part you pick the two shapes; the other settings are fixed. You can change everything in ${sc}.`,`这一部分只选两个形状，其余设置固定。在${sc}里可以改全部设置。`);
  if(id==='einsum') return tx(`In this part you pick the einsum string; the other settings are fixed. You can change everything in ${sc}.`,`这一部分只选 einsum 字符串，其余设置固定。在${sc}里可以改全部设置。`);
  if(id==='nn') return tx(`In this part you change N, M and D; the other settings are fixed. You can change everything in ${sc}.`,`这一部分只改 N、M 和 D，其余设置固定。在${sc}里可以改全部设置。`);
  return tx(`Fixed in this part: the chart shows the part’s own example. You can change the settings in ${sc}.`,`这一部分固定：图里显示这一部分自己的例子。在${sc}里可以改这些设置。`);
}
function syncControls(){
  const setR=(key,st,v)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{const x=Math.max(0,st.indexOf(v)); if(+i.value!==x) i.value=x; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fint(v); i.setAttribute('aria-valuetext',fint(v));});
  setR('N',N_ST,S.N); setR('M',M_ST,S.M); setR('D',D_ST,S.D);
  $$('.seg-ctl[data-key="a"]',app).forEach(g=>A.ui.setPressed(g,S.a)); $$('.seg-ctl[data-key="b"]',app).forEach(g=>A.ui.setPressed(g,S.b)); $$('.seg-ctl[data-key="es"]',app).forEach(g=>A.ui.setPressed(g,S.es));
  const g=mode==='guided', id=g?STEPS[step].id:'';
  const hide=(sel,h)=>{const e=$(sel,app); if(e) e.hidden=h;};
  hide('#ts-g-bc',mode==='challenge'||(g&&id!=='bcast')); hide('#ts-g-es',mode==='challenge'||(g&&id!=='einsum')); hide('#ts-g-nn',mode==='challenge'||(g&&id!=='nn')); hide('#ts-ctl-reset-wrap',mode!=='sandbox');
  $('#ts-controls',app).classList.toggle('locked',mode==='challenge');
  const ln=$('#ts-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln._h!==msg){ln.innerHTML=msg; ln._h=msg;}
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.tip);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  const D0=S.D; Object.assign(S,p);
  if(!SH_A.includes(S.a)) S.a=BASE.a; if(!SH_B.includes(S.b)) S.b=BASE.b; if(!ES.some(e=>e.id===S.es)) S.es=BASE.es;
  if(!N_ST.includes(S.N)) S.N=BASE.N; if(!M_ST.includes(S.M)) S.M=BASE.M; if(!D_ST.includes(S.D)) S.D=BASE.D; if(!['bc','es','nn'].includes(S.view)) S.view='nn';
  if(isP('nn')&&S.D!==D0) dMoved=true;
  if(isP('einsum')) esTried.add(S.es);
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setRem(k){if(!REMS.includes(k)) return; remView=k; S={...BASE,view:k,...(k==='bc'?{a:'4x3',b:'3'}:{}),...(k==='es'?{es:'sc'}:{})}; renderRail(); if(innerWidth<=760) A.ui.reveal($('#ts-stage',app));}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='predict'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the four answers in the card.','向下滚到卡片里的四个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='bcast') return segA('ts-ph-a',true,SH_A.slice(0,3))+segB('ts-ph-b',true,SH_B.slice(0,4));
  if(id==='einsum') return `<button type="button" class="btn ghost sm" data-act="to-es" data-tipx="${esc(tx('Scrolls down to the four strings in the card.','向下滚到卡片里的四个字符串。'))}">${tx('Pick a string below ↓','在下面选字符串 ↓')}</button>`;
  if(id==='nn') return rng('ts-ph-D','D',tx('D, numbers per vector','D，每个向量的维数'),D_ST,S.D,true);
  return '';
}
function renderPhone(){const el=$('#ts-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  const st=$('#ts-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#ts-rail .step.current',app)||$('#ts-rail',app));}
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
    const st=$('#ts-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ts-rail .step.current',app)||$('#ts-rail .railcard',app)||$('#ts-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#ts-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ts-insight',app)); else revealBox($('#ts-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chPick=CH.start;}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#ts-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[S.view,SH_A.indexOf(S.a),SH_B.indexOf(S.b),ES.findIndex(e=>e.id===S.es),N_ST.indexOf(S.N),M_ST.indexOf(S.M),D_ST.indexOf(S.D)].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7||!['bc','es','nn'].includes(p[0])) return null;
  const [ia,ib,ie,iN,iM,iD]=p.slice(1).map(Number), ok=(i,arr)=>Number.isInteger(i)&&i>=0&&i<arr.length;
  if(!ok(ia,SH_A)||!ok(ib,SH_B)||!ok(ie,ES)||!ok(iN,N_ST)||!ok(iM,M_ST)||!ok(iD,D_ST)) return null;
  return {view:p[0],a:SH_A[ia],b:SH_B[ib],es:ES[ie].id,N:N_ST[iN],M:M_ST[iM],D:D_ST[iD]};
}

/* =====================================================================
   interview drill: seeded shapes, the answer from bshape (every line redoes by hand)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+3), pick=xs=>xs[Math.floor(r()*xs.length)];
  const n=3, base=[...Array(n)].map(()=>pick([2,3,4,5,6,8])), a=base.map(v=>r()<0.35?1:v), b0=base.map(v=>r()<0.4?1:v), drop=Math.floor(r()*2), b=b0.slice(drop);
  if(r()<0.35){const k=Math.floor(r()*b.length); b[k]=b[k]===1?pick([2,3,4]):b[k]+1;}
  return {seed,a,b,r:bshape(a,b)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#ts-drill-q',app), an=$('#ts-drill-a',app);
  qd.innerHTML=tx(`x has shape <b>${shp(d.a)}</b> and y has shape <b>${shp(d.b)}</b>. What is the shape of x + y, or on which axis does it fail?`,`x 的形状是 <b>${shp(d.a)}</b>，y 的形状是 <b>${shp(d.b)}</b>。x + y 的形状是什么？如果报错，错在哪个轴？`);
  an.hidden=!drillOpen; if(!drillOpen) return;
  an.innerHTML=`<p><b>${d.r.ok?tx(`${shp(d.r.out)}.`,`${shp(d.r.out)}。`):tx(`An error on ${axisName(d.r.err.k)}: ${d.r.err.x} and ${d.r.err.y}.`,`报错，在${axisName(d.r.err.k)}上：${d.r.err.x} 和 ${d.r.err.y}。`)}</b></p>`+
    `<p>${tx('From the right:','从右边起：')} ${d.r.ax.slice().reverse().map(c=>`${c.x==null?tx('missing','缺'):c.x} ${tx('and','和')} ${c.y==null?tx('missing','缺'):c.y} → ${c.bad?tx('error','报错'):c.out}`).join(tx('; ','；'))}.</p>`;
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={N:{en:'N, test vectors',zh:'N，测试向量数'},M:{en:'M, training vectors',zh:'M，训练向量数'},D:{en:'D, numbers per vector',zh:'D，每个向量的维数'},a:{en:'Shape of a',zh:'a 的形状'},b:{en:'Shape of b',zh:'b 的形状'},es:{en:'String',zh:'字符串'}};
const TERM={
 N:()=>`<div>${esc(tx(`N is the number of test vectors: ${fint(S.N)}. Every array of the broadcast and the matmul grows with it.`,`N 是测试向量的数量：${fint(S.N)}。广播和矩阵乘法的数组都随它变大。`))}</div>`,
 M:()=>`<div>${esc(tx(`M is the number of training vectors: ${fint(S.M)}. Every array of the broadcast and the matmul grows with it.`,`M 是训练向量的数量：${fint(S.M)}。广播和矩阵乘法的数组都随它变大。`))}</div>`,
 D:()=>`<div>${esc(tx(`D is how many numbers each vector has: ${fint(S.D)}. It is the axis that gets summed, so only the broadcast array grows with it.`,`D 是每个向量的维数：${fint(S.D)}。它是被求和的那个轴，所以只有广播的数组随它变大。`))}</div>`,
 a:()=>`<div>${esc(tx(`The shape of a: ${shK(S.a)}.`,`a 的形状：${shK(S.a)}。`))}</div>`,
 b:()=>`<div>${esc(tx(`The shape of b: ${shK(S.b)}.`,`b 的形状：${shK(S.b)}。`))}</div>`,
 es:()=>`<div>${esc(TIP.einsum())}</div>`
};
const termTipHTML=k=>TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: the sizes are 10,000 queries, 200,000 stored vectors and 768 numbers.','挑战中固定不变：10,000 个查询，200,000 个存好的向量，每个 768 维。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if((k==='N'||k==='M'||k==='D')&&id==='nn') return '';
  if((k==='a'||k==='b')&&id==='bcast') return '';
  if(k==='es'&&id==='einsum') return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='nn') return tx('Click to scroll to the bars in the chart.','点击滚到图里的横条。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({N:$('#ts-ic-N',app)||$('#ts-ctl-N',app),M:$('#ts-ic-M',app)||$('#ts-ctl-M',app),D:$('#ts-ic-D',app)||$('#ts-ctl-D',app),a:$('#ts-ic-a',app)||$('#ts-ctl-a',app),b:$('#ts-ic-b',app)||$('#ts-ctl-b',app),es:$('#ts-ic-es',app)||$('#ts-ctl-es',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#ts-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='nn'){const el=$('#ts-nn',app); A.tip.hide(); const t=el&&!el.hidden&&el.getClientRects().length?el:$('#ts-graph',app); t.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(t); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#ts-controls',app)); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>setState({...BASE,view:S.view})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chPick).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#ts-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-es'){const q=$('#ts-ic-es',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ts-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#ts-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
/* a change under Settings also switches the chart to that group's view in Scenarios */
const viewFor=k=>({a:'bc',b:'bc',es:'es',N:'nn',M:'nn',D:'nn'}[k]);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value, st={N:N_ST,M:M_ST,D:D_ST}[k]; if(!st) return;
    bySettings(e.target,()=>setState({[k]:st[v],...(mode==='sandbox'?{view:viewFor(k)}:{})}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#ts-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ts-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>setState({...s.showMe})); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ts-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&b.closest('#ts-ic-es')&&isP('einsum')){const f=()=>byLearner(()=>setState({es:d.lever})); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.lever!=null&&b.closest('#ts-ic-ch')&&mode==='challenge'){byLearner(()=>{chPick=d.lever; render();}); return;}
    if(d.view!=null&&b.closest('#ts-view')){byLearner(()=>setState({view:d.view})); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='a'||k==='b'||k==='es') bySettings(b,()=>setState({[k]:d.v,...(mode==='sandbox'?{view:viewFor(k)}:{})}));
      if(b.closest('#ts-rail')&&mode==='guided'){wantObs=true; revealSoon(); wantObs=false;}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#ts-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ts-rail',app)); return;}
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
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chPick,chSubmitted,chReveal,sbTouched,seenErr,esTried:[...esTried],
    dMoved,remView,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(chPick).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse([x.view,SH_A.indexOf(x.a),SH_B.indexOf(x.b),ES.findIndex(e=>e.id===x.es),N_ST.indexOf(x.N),M_ST.indexOf(x.M),D_ST.indexOf(x.D)].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chPick=mode==='challenge'&&CHO.some(x=>x.id===o.chPick)?o.chPick:CH.start;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    seenErr=o.seenErr===true; esTried=new Set(['mm']); if(Array.isArray(o.esTried)) o.esTried.forEach(x=>{if(ES.some(e=>e.id===x)) esTried.add(x);});
    dMoved=o.dMoved===true; remView=REMS.includes(o.remView)?o.remView:'bc';
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#ts-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ts-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    fromSnap=false;
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#ts-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#ts-math',app).open=true;}},
  onResize(){},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='bc'?$('#ts-bc .ts-axes',app):(v==='es'?$('#ts-es .ts-lt',app):$('#ts-nn .ts-nnrows',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#ts-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      if(v==='bc'){const r=bshape(parseShape(S.a),parseShape(S.b)), hid=hideOut();
        vis.innerHTML=`<span class="ts-dkx" aria-hidden="true">${r.ax.map(c=>`<i class="${c.bad&&!hid?'ts-bad':''}"></i>`).join('')}</span>`;
        read.innerHTML=esc(hid?(pred?tx('Your answer is saved; part 3 shows the result','答案已记下，第 3 部分给出结果'):tx('Predict what a + b gives first','先预测 a + b 会得到什么')):(r.ok?tx(`a ${shp(r.a)} ${OP()} b ${shp(r.b)} gives ${shp(r.out)}`,`a ${shp(r.a)} ${OP()} b ${shp(r.b)} 得到 ${shp(r.out)}`):tx(`a ${shp(r.a)} ${OP()} b ${shp(r.b)} fails on ${axisName(r.err.k)}`,`a ${shp(r.a)} ${OP()} b ${shp(r.b)} 在${axisName(r.err.k)}上报错`))); return;}
      if(v==='es'){const r=esR(S.es); vis.innerHTML=`<span class="ts-dkx" aria-hidden="true">${r.cols.map(c=>`<i class="${r.summed.includes(c)?'ts-sum':''}"></i>`).join('')}</span>`; read.innerHTML=esc(tx(`Output ${shp(r.outShape)}, ${r.summed.join(', ')} summed, ${r.macs>=1e6?big(r.macs):fint(r.macs)} multiply-adds`,`输出 ${shp(r.outShape)}，对 ${r.summed.join('、')} 求和，${r.macs>=1e6?big(r.macs):fint(r.macs)} 次乘加`)); return;}
      const rows=nnRows(); vis.innerHTML=`<div class="ts-dkb">`+rows.map(x=>`<i class="${fits(x.bytes)?'':'ts-over'}" style="width:${(xOf(x.bytes)*100).toFixed(1)}%"></i>`).join('')+`</div>`;
      if(v==='ch'){const e=chEval(chPick); read.innerHTML=esc(tx(`${L(CHT[e.id])}: largest array ${fB(e.bytes)}, ${itersTxt(e.iters)}`,`${L(CHT[e.id])}：最大的数组 ${fB(e.bytes)}，${itersTxt(e.iters)}`)); return;}
      const r=nn(S.N,S.M,S.D), f=fits(r.bc.bytes); read.innerHTML=esc(tx(`Broadcast ${f?'fits':'does not fit'} one H100: ${fB(r.bc.bytes)} at ${fint(S.N)} × ${fint(S.M)} × ${fint(S.D)} (matmul ${fB(r.mm.bytes)})`,`广播${f?'放得下':'放不下'}一张 H100：${fint(S.N)} × ${fint(S.M)} × ${fint(S.D)} 时要 ${fB(r.bc.bytes)}（矩阵乘法 ${fB(r.mm.bytes)}）`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__ts=Object.freeze({
  bshape:(a,b)=>deep(bshape(a,b)), bsum:(a,b)=>deep(bsum(a,b)), arrA:s=>deep(arrA(s)), arrB:s=>deep(arrB(s)),
  einsum:(spec,sizes)=>deep(einsum(spec,sizes)), es:id=>deep(esR(id)), MA:deep(MA), MB:deep(MB), MC:deep(MC),
  nn:(N,M,D)=>deep(nn(N,M,D)), chEval:id=>deep(chEval(id)), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)),
  fmt:Object.freeze({B:fB,GB:fGB,shp}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transferO:e.transfer.map(q=>q.o),transferW:e.transfer.map(q=>q.why)});},
  terms:(N,M,D)=>TERMS.map(t=>({k:t.k,w:t.w,sh:t.sh,b:t.b(N,M,D)})),
  consts:Object.freeze({N_ST:N_ST.slice(),M_ST:M_ST.slice(),D_ST:D_ST.slice(),SH_A:SH_A.slice(),SH_B:SH_B.slice(),ES:deep(ES.map(e=>({id:e.id,spec:e.spec,sizes:e.sizes}))),
    CH:deep(CH),CHO:deep(CHO),BASE:{...BASE},H100,STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s})))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},seenErr,esTried:[...esTried],dMoved,remView,chPick,chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
