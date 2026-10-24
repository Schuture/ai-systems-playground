/* flash-attention (the trunk "How LLMs work", world models): How does FlashAttention compute exact attention with less memory traffic?
   Fidelity: Toy run. The browser runs one head of attention on a 12-token example (head size 4, seeded Q, K, V) twice: the standard
   way (the scores S written to HBM and read back, the weights P written and read back, then the output O) and tile by tile with the online softmax
   (FlashAttention-2's loop order: one Q tile on chip, the K and V tiles stream past). The real-size numbers are formulas:
   - HBM traffic per head in BF16 (2 bytes), as the FlashAttention papers count it: standard = Q, K, V read once, S and P each written
     and read once, O written once; FlashAttention = Q read once, O written once, K and V read once per Q tile (⌈Nq/Br⌉ passes).
   - On-chip memory for one tile pair: Q tile (BF16), K and V tiles (BF16), the score tile (FP32), the output accumulator (FP32), the
     running max and sum (FP32); the budget is 227 KiB, what one thread block can use on an H100.
   Words (EN / ZH): score 得分 (as in #mha), attention weight 注意力权重, tile, HBM, SRAM, running max m / running sum ℓ (累计最大值 /
   累计和), rescale 缩放, Q tile / K/V tile, part / 部分, step (the toy's ▸) / 步.
   window.__fa is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='flash-attention';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine: the toy (computed for real) and the real-size formulas
   ===================================================================== */
const TOY={N:12,d:4,B:4,seed:9};
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function toyMats(){const r=mulberry(TOY.seed*7919+17), m=()=>Array.from({length:TOY.N},()=>Array.from({length:TOY.d},()=>Math.round((r()*3-1.5)*10)/10));
  return {Q:m(),K:m(),V:m()};}
const MATS=toyMats();
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
/* standard attention: the whole S, the whole P, then O */
function toyStd(){const {Q,K,V}=MATS, sc=Math.sqrt(TOY.d);
  const S=Q.map(q=>K.map(k=>dot(q,k)/sc));
  const P=S.map(row=>{const m=Math.max(...row), e=row.map(v=>Math.exp(v-m)), z=e.reduce((a,b)=>a+b,0); return e.map(v=>v/z);});
  const O=P.map(p=>V[0].map((_,c)=>p.reduce((s,w,j)=>s+w*V[j][c],0)));
  return {S,P,O};}
/* FlashAttention: Q tile i stays on chip; K/V tile j streams past; per row a running max m, a running sum ℓ and an unnormalized output.
   One entry per tile pair (i, j), with the state after it */
function toyTiled(){const {Q,K,V}=MATS, B=TOY.B, T=TOY.N/B, sc=Math.sqrt(TOY.d), steps=[], O=Array.from({length:TOY.N},()=>new Array(TOY.d).fill(0));
  for(let i=0;i<T;i++){let m=new Array(B).fill(-Infinity), l=new Array(B).fill(0), acc=Array.from({length:B},()=>new Array(TOY.d).fill(0));
    for(let j=0;j<T;j++){
      const s=[], f=[], tmax=[], mOld=m.slice();
      for(let r=0;r<B;r++){const q=Q[i*B+r], row=[]; for(let c=0;c<B;c++) row.push(dot(q,K[j*B+c])/sc); s.push(row); tmax.push(Math.max(...row));}
      const mNew=m.map((x,r)=>Math.max(x,tmax[r]));
      for(let r=0;r<B;r++){const fr=m[r]===-Infinity?0:Math.exp(m[r]-mNew[r]), p=s[r].map(v=>Math.exp(v-mNew[r]));
        f.push(fr); l[r]=l[r]*fr+p.reduce((a,b)=>a+b,0);
        acc[r]=acc[r].map((x,c)=>x*fr+p.reduce((t,w,k)=>t+w*V[j*B+k][c],0));}
      m=mNew;
      const last=j===T-1; if(last) for(let r=0;r<B;r++) O[i*B+r]=acc[r].map(x=>x/l[r]);
      steps.push({i,j,s,tmax,mOld,m:m.slice(),f,l:l.slice(),acc:acc.map(a=>a.slice()),loadQ:j===0,writeO:last,
        moved:(j===0?B*TOY.d:0)+2*B*TOY.d+(last?B*TOY.d:0)});}}
  return {steps,O};}
const STD=toyStd(), TIL=toyTiled();
const MAXDIFF=STD.O.reduce((a,row,i)=>Math.max(a,...row.map((x,c)=>Math.abs(x-TIL.O[i][c]))),0);
/* the standard path, one ▸ per step: what moves, which way, how many numbers */
const NT=TOY.N, DT=TOY.d;
const STD_STEPS=[
 {k:'start',n:0},
 {k:'readQK',n:2*NT*DT,dir:'in',mats:['Q','K']},
 {k:'writeS',n:NT*NT,dir:'out',mats:['S']},
 {k:'readS',n:NT*NT,dir:'in',mats:['S']},
 {k:'writeP',n:NT*NT,dir:'out',mats:['P']},
 {k:'readPV',n:NT*NT+NT*DT,dir:'in',mats:['P','V']},
 {k:'writeO',n:NT*DT,dir:'out',mats:['O']}];
const STD_TOTAL=STD_STEPS.reduce((a,s)=>a+s.n,0), FA_TOTAL=TIL.steps.reduce((a,s)=>a+s.moved,0);
const stdSoFar=k=>STD_STEPS.slice(0,k+1).reduce((a,s)=>a+s.n,0);
const faSoFar=k=>TIL.steps.slice(0,k).reduce((a,s)=>a+s.moved,0);
const SP_TOY=4*NT*NT;
/* the H100 and the models (shared numbers: labs/prefill-decode.js; LAB_UX_GUIDE gold numbers) */
const HW={bw:3.35e12,flops:989.5e12,sfu:3.9e12,smem:227*1024,smemSM:228*1024};
const EL=2;                                 // BF16: 2 bytes a number
const NS=[1024,2048,4096,8192,16384,32768,65536,131072];
const TILES=[32,64,128,256];
/* one head's HBM traffic in bytes. Nq query rows against N keys (prefill: Nq = N; decode: Nq = 1) */
/* floor: Q, K and V read once and O written once, the least any kernel moves (what FlashAttention approaches when the L2 cache serves
   its re-reads of K and V); the papers' count charges every re-read to HBM, so FlashAttention's real traffic lies between floor and fa */
function traffic(N,d,Br,Nq){Nq=Nq==null?N:Nq;
  const stdBase=EL*(2*Nq*d+2*N*d), sp=EL*4*Nq*N, passes=Math.ceil(Nq/Br), faBase=EL*2*Nq*d, kv=EL*passes*2*N*d, kv1=EL*2*N*d;
  return {N,Nq,d,Br,std:stdBase+sp,stdBase,sp,s:EL*Nq*N,fa:faBase+kv,faBase,kv,kv1,floor:faBase+kv1,passes,ratio:(stdBase+sp)/(faBase+kv)};}
/* on-chip memory for one tile pair, bytes */
function sram(Br,Bc,d){const p={q:EL*Br*d,k:EL*Bc*d,v:EL*Bc*d,s:4*Br*Bc,o:4*Br*d,ml:8*Br}; const total=p.q+p.k+p.v+p.s+p.o+p.ml; return {...p,total,fits:total<=HW.smem};}
/* time for one head at full speed, seconds (part 3's answer) */
function times(N,d){return {qk:2*N*N*d/HW.flops,pv:2*N*N*d/HW.flops,exp:N*N/HW.sfu,sp:EL*4*N*N/HW.bw};}
const LLAMA={d:128,H:32}, L1B={d:64,H:32}, L70={H:64}, GEMMA={d:256};
/* the challenge: a model and a length the guided parts never use */
const CH={d:L1B.d,N:32768,start:{Br:64,Bc:64},goal:1.5*A.GiB};
function chEval(Br,Bc){const t=traffic(CH.N,CH.d,Br), m=sram(Br,Bc,CH.d); return {Br,Bc,t,m,fits:m.fits,low:t.fa<=CH.goal,ok:m.fits&&t.fa<=CH.goal};}
const chChanges=(Br,Bc)=>(Br!==CH.start.Br?1:0)+(Bc!==CH.start.Bc?1:0);
function chBest(){let b=null; TILES.forEach(Br=>TILES.forEach(Bc=>{const e=chEval(Br,Bc); if(!e.ok) return; const c=chChanges(Br,Bc);
  if(!b||c<b.c||(c===b.c&&e.m.total<b.m)) b={Br,Bc,c,m:e.m.total};})); return b;}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const MiB=A.MiB, GiB=A.GiB, KiB=A.KiB;
/* bytes in one unit for a pair shown side by side (R6.12): MiB below 10 GiB (whole, or one decimal under 10 MiB), else GiB, one decimal */
const unitOf=max=>max>=10*GiB?'G':'M';
function fB(b,u){u=u||unitOf(b); if(u==='G') return fixd(b/GiB,1)+' GiB'; const m=b/MiB; return (m<10?fixd(m,2):fint(m))+' MiB';}
const fB2=(b)=>fixd(b/GiB,2)+' GiB';                    // the challenge: GiB with two decimals, so 1.01 and 1.50 compare
const fS=b=>b<MiB?fK(b):(b<10*GiB&&b%MiB===0?fint(b/MiB)+' MiB':fB(b,unitOf(b)));   // S: KiB for one decode row; whole MiB when exact
const fLay=b=>b>=GiB?fint(Math.round(b/GiB))+' GiB':fB(b,'M');           // a layer's S: whole GiB from 1 GiB up, as the mission's 4 GiB
const fPart=(b,u)=>b<KiB?`${fint(b)} ${tx('bytes','字节')}`:(b<MiB?fK(b):fT(b,u));   // a part of a sum in a tooltip
const fT=(b,u)=>mode==='challenge'?fB2(b):fB(b,u);      // traffic on the stage: GiB with two decimals in the challenge, as its goal
const fK=b=>{const k=b/KiB; return (Math.abs(k-Math.round(k))<1e-9?fint(k):fixd(k,2))+' KiB';};
const fUs=s=>fint(Math.round(s*1e6));
const fTok=n=>A.fmt.tok(n);
const sci=x=>{if(x===0) return '0'; const e=Math.floor(Math.log10(x)), m=x/10**e; return `${fixd(m,1)} × 10<sup>${e<0?'−'+(-e):e}</sup>`;};
const SUPD={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
const sciT=x=>{if(x===0) return '0'; const e=Math.floor(Math.log10(x)), m=x/10**e; return `${fixd(m,1)} × 10${String(e).replace(/[-0-9]/g,c=>SUPD[c])}`;};   // plain text with superscripts (tooltips, metric cards)
const mx=t=>A.ui.math?A.ui.math(t):esc(t);              // a formula typeset with the kit (R0.1)
const f2=x=>fixd(x,2), f3=x=>fixd(x,3), f4=x=>fixd(x,4);
const NB='‑';                                          // no-break hyphen in model names
const M8=`Llama${NB}3.1${NB}8B`, M1=`Llama${NB}3.2${NB}1B`, M70=`Llama${NB}3.1${NB}70B`, MG=`Gemma${NB}2${NB}9B`, MGPT=`GPT${NB}2`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):id;
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const pl=(n,one,many)=>n===1?one:many;

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={d:128,ni:3,Br:128,Bc:128,dec:false,view:'real',path:'std'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let tstep=0, playing=false, speed='med', raf=0, lastT=0, acc=0;
let lv=128, lvTried=new Set([128]), seenN=new Set([3]), saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false;
let doneStd=false, doneFa=false, drillSeed=1, drillOpen=false, annT=null, wantObs=false;
const shownPrev={};
const SPEEDS={slow:24000,med:14000,fast:8000};       // ms for a whole run, whatever its number of steps: Fast plays all 9 tiles in 8 s (R3.9)
const stepMs=()=>Math.round(SPEEDS[speed]/Math.max(1,stepsN()));
const N_=()=>NS[S.ni];

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 hbm:()=>tx('HBM (high-bandwidth memory) is the GPU’s main memory, next to the compute chip: 80 GB on an H100, read and written at up to 3.35 TB/s. Every matrix that does not fit on the chip lives here.',
            'HBM（high-bandwidth memory，高带宽显存）是 GPU 的主显存，挨着计算芯片：H100 上有 80 GB，读写速度最高 3.35 TB/s。芯片上放不下的矩阵都放在这里。'),
 sram:()=>tx('SRAM here means the fast memory on the compute chip itself. An H100 has 132 streaming multiprocessors (SMs), each with 228 KiB of shared memory; one thread block, the group of threads that works on one tile, can use up to 227 KiB of it. It is much faster than HBM and far smaller: 228 KiB per SM against 80 GB.',
             '这里的 SRAM 指计算芯片自己身上的高速存储。H100 有 132 个流式多处理器（SM），每个有 228 KiB 共享内存；一个 thread block（算一个 tile 的那一组线程）最多能用其中 227 KiB。它比 HBM 快得多，也小得多：每个 SM 228 KiB，HBM 有 80 GB。'),
 score:()=>tx('A score is one token’s query times another token’s key, divided by the square root of the head size d:','得分是一个 token 的 query 乘另一个 token 的 key，再除以 head 维数 d 的平方根：')+FXM(tx('\\text{score} = \\frac{q \\cdot k}{\\sqrt{d}}','\\text{得分} = \\frac{q \\cdot k}{\\sqrt{d}}'))+tx(`S holds one score for every pair of tokens, so it has ${mx('N \\times N')} of them.`,`S 为每一对 token 存一个得分，所以一共 ${mx('N \\times N')} 个。`),
 pmat:()=>kt('aweight')+tx(` P holds the attention weights: the softmax of each row of S, ${mx('N \\times N')} numbers again.`,` P 存的就是注意力权重：S 每一行做 softmax 的结果，同样是 ${mx('N \\times N')} 个数值。`),
 tile:()=>tx('A tile is a block of consecutive rows of a matrix. FlashAttention cuts Q into tiles of Br rows and K and V into tiles of Bc rows, so that one tile of each fits in SRAM at once.',
             'tile 是矩阵里连续几行组成的一块。FlashAttention 把 Q 切成每个 Br 行的 tile，把 K 和 V 切成每个 Bc 行的 tile，让每种各一个 tile 能同时放进 SRAM。'),
 online:()=>tx(`The online softmax computes a softmax in one pass over the scores. Each row keeps its largest score so far (the running max m) and the running sum ℓ of ${mx('e^{\\text{score} - m}')}. When a new score is larger than m, the sum is first multiplied by ${mx('e^{m_{\\text{old}} - m_{\\text{new}}}')}, so it always counts every score against the current max.`,
               'online softmax 只读一遍得分就算出 softmax。每一行记两个值：累计最大值 m，也就是目前最大的得分；累计和 ℓ，也就是目前为止的总和。新得分比 m 大时，先把累计和缩放一次，换成以新的最大值为准。'),
 membound:()=>tx('Memory-bound: the time goes into moving bytes, not into arithmetic. The chip could compute faster than HBM can deliver the numbers.',
                 '显存带宽受限（memory-bound）：时间花在搬字节上，而不是花在计算上。芯片算得比 HBM 送数据还快。'),
 count:()=>tx('Here every Q tile reads K and V again from HBM, as in the FlashAttention paper’s algorithm. An H100’s 50 MB L2 cache probably serves many of these re-reads, so the real traffic lies between the floor of reading Q, K and V and writing O once, and this count. Unfused standard attention also moves S a few more times, so the real gap is likely larger. Measured on GPT‑2 medium (A100, forward and backward), the FlashAttention paper reports 40.3 GB of HBM traffic for standard attention and 4.4 GB for FlashAttention, about 89% less (Figure 2).',
              '这里假设每个 Q tile 都从 HBM 重读 K 和 V，和 FlashAttention 论文的算法一样。H100 的 50 MB L2 cache 很可能送出其中不少重读，所以真实流量在 Q、K、V 各读一次、O 写一次的下限和这里算出的值之间。未融合的标准 attention 还会多搬几次 S，所以真实差距可能更大。FlashAttention 论文在 A100 上实测 GPT‑2 medium 的前向加反向：标准 attention 读写 HBM 40.3 GB，FlashAttention 4.4 GB，少约 89%（图 2）。'),
 countPre:()=>tx('Bytes per head from a formula, as in the FlashAttention paper: every matrix that crosses between HBM and the chip, 2 bytes a number. Part 5 says how real traffic may differ.',
                 '每个 head 的字节数是公式算出的，和 FlashAttention 论文一样：在 HBM 和芯片之间来回的每个矩阵都算上，每个数值 2 字节。第 5 部分会讲真实流量可能有什么不同。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'qk',t:{en:'The matrix multiply QKᵀ that computes the scores',zh:'算得分的矩阵乘法 QKᵀ'}},
  {v:'move',t:{en:'Moving S and P between HBM and the chip',zh:'在 HBM 和芯片之间搬运 S 和 P'}},
  {v:'exp',t:{en:'The exp of every score inside the softmax',zh:'softmax 里给每个得分算 exp'}}];
const PRED_RIGHT='move';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const LVD={32:{en:'many small tiles, little SRAM',zh:'很多小 tile，占的 SRAM 少'},64:{en:'half as many Q tiles as 32',zh:'Q tile 的数量是 32 行时的一半'},128:{en:'the current tiles (initial setup)',zh:'现在用的 tile（初始设置）'},256:{en:'the fewest passes over K and V',zh:'扫 K 和 V 的遍数最少'}};
function lvRes(b){const t=traffic(8192,128,b), m=sram(b,b,128); return m.fits?fB(t.fa,'M'):tx('does not fit','放不下');}
const lvCtl=()=>`<div class="fa-lv" id="fa-ic-lv">`+A.ui.levers({aria:tx('Tile size, pick one','tile 大小，选一种'),value:lv,
  options:TILES.map(b=>({v:b,title:tx(`${b} × ${b} tiles`,`${b} × ${b} 的 tile`),desc:L(LVD[b]),result:esc(lvTried.has(b)||b===lv?lvRes(b):'')}))})+`</div>`;
const nCtl=(id,inline)=>A.ui.range({id,key:'N',label:tx('Context length','上下文长度'),min:0,max:NS.length-1,value:S.ni,out:fTok(N_())+tx(' tokens',' token'),inline});
/* the toy's player: ◂ ▸, play / pause, speed, scrubber (R3.9); the card and the stage hold the same one */
function playerHTML(id){const n=stepsN();
  return `<div class="fa-player" id="${id}"><div class="fa-pbtns"><button type="button" class="btn ghost sm" data-act="back" aria-label="${esc(tx('One step back','后退一步'))}">◂</button>`+
    `<button type="button" class="btn ghost sm fa-play" data-act="play" aria-pressed="${playing}">${playing?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Play','▶ 播放')}</button>`+
    `<button type="button" class="btn ghost sm" data-act="fwd" aria-label="${esc(tx('One step forward','前进一步'))}">▸</button>`+
    `<span class="fa-speed" data-tipx="${esc(tx(`Slow plays the whole run in ${SPEEDS.slow/1000} s, Medium in ${SPEEDS.med/1000} s and Fast in ${SPEEDS.fast/1000} s.`,`走完一遍：慢速 ${SPEEDS.slow/1000} 秒，中速 ${SPEEDS.med/1000} 秒，快速 ${SPEEDS.fast/1000} 秒。`))}" tabindex="0">`+
    `<span class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="speed">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${speed===v}">${x}</button>`).join('')}</span></span></div>`+
    `<label class="fa-scrub"><span class="fa-sl" data-fa-out="stepno"></span><input type="range" data-key="tstep" min="0" max="${n}" step="1" value="${tstep}" aria-label="${esc(tx('Step shown in the chart','图里显示的步'))}"></label></div>`;}
/* formulas written out (Yixiong's review): hovers built as HTML strings with the kit's MathML, one formula per line */
const FXM=t=>`<div class="fx">${mx(t)}</div>`;
const FXL=t=>`<p class="fa-fx">${mx(t)}</p>`;
const DEFK=(text,key)=>`<span class="def" tabindex="0" data-tip="${key}">${text}</span>`;
A.addStrings({en:{
  fa_score:`<div>A score says how well one token’s query matches another token’s key:</div>${FXM('S_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d}}')}<div>${mx('q_i')} is token i’s query, ${mx('k_j')} is token j’s key, and d is the head size, 128 here. S holds one score for every pair of tokens: ${mx('N \\times N')} in all.</div>`,
  fa_sm:`<div>Softmax turns each row of S into attention weights that add up to 1:</div>${FXM('P_{ij} = \\frac{e^{S_{ij}}}{\\sum_k e^{S_{ik}}}')}<div>The output then mixes the values with these weights:</div>${FXM('O = P V')}`,
  fa_on:`<div>Online softmax computes a softmax while it reads the scores one tile at a time. Each row keeps its largest score so far, m, and a running sum, ℓ: the softmax’s denominator so far, the sum of ${mx('e^{S - m}')}. When a tile brings a larger score, what was added so far is rescaled once:</div>${FXM('m_{\\text{new}} = \\max(m, \\max_j S_{ij})')}${FXM('ℓ \\leftarrow ℓ \\cdot e^{m - m_{\\text{new}}} + \\sum_j e^{S_{ij} - m_{\\text{new}}}')}<div>The result equals the ordinary softmax exactly.</div>`
},zh:{
  fa_score:`<div>得分衡量一个 token 的 query 和另一个 token 的 key 有多匹配：</div>${FXM('S_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d}}')}<div>${mx('q_i')} 是第 i 个 token 的 query，${mx('k_j')} 是第 j 个 token 的 key，d 是 head 的维数，这里是 128。S 为每一对 token 存一个得分，一共 ${mx('N \\times N')} 个。</div>`,
  fa_sm:`<div>softmax 把 S 的每一行变成加起来等于 1 的注意力权重：</div>${FXM('P_{ij} = \\frac{e^{S_{ij}}}{\\sum_k e^{S_{ik}}}')}<div>输出再用这些权重给 value 加权求和：</div>${FXM('O = P V')}`,
  fa_on:`<div>online softmax 一个 tile 一个 tile 地读得分，边读边算 softmax。每一行记两个值：累计最大值 m，也就是目前最大的得分；累计和 ℓ，也就是 softmax 的分母（各 ${mx('e^{S - m}')} 之和）到目前为止的值。某个 tile 带来更大的得分时，把已经加起来的部分整体缩放一次：</div>${FXM('m_{\\text{新}} = \\max(m, \\max_j S_{ij})')}${FXM('ℓ \\leftarrow ℓ \\cdot e^{m - m_{\\text{新}}} + \\sum_j e^{S_{ij} - m_{\\text{新}}}')}<div>结果和普通的 softmax 完全相同。</div>`
}});
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'real',
  lead:()=>tx(`${M8} reads an 8K-token prompt on an H100: where do its scores go?`,`${M8} 在 H100 上读一个 8K token 的 prompt：它的得分放在哪？`),
  enter:()=>({...BASE}),
  body:()=>{const t=traffic(8192,128,128), lay=fB(t.s*32,'G').replace('.0 ',' ');
   const f3=FXL(tx('\\text{scores}\\ S = \\frac{Q K^{T}}{\\sqrt{d}}','\\text{得分}\\ S = \\frac{Q K^{T}}{\\sqrt{d}}'))+FXL(tx('\\text{attention weights}\\ P = \\text{softmax}(S)','\\text{注意力权重}\\ P = \\text{softmax}(S)'))+FXL(tx('\\text{output}\\ O = P V','\\text{输出}\\ O = P V'));
   return tx(`<p>${M8} reads an 8K-token document on an H100 (the prefill), and each ${DEF('head',kt('head'))} computes attention in three steps:</p>${f3}`+
     `<p>S holds a ${DEFK('score','fa_score')} for every pair of tokens, so at 8K tokens one head’s S takes ${fB(t.s,'M')}. P, from the ${DEFK('softmax','fa_sm')}, is as large.</p>`+
     `<p>${DEF('HBM',TIP.hbm())}, the GPU’s memory, is large and slow; ${DEF('SRAM',TIP.sram())}, on the chip, is small and fast, a few hundred KB per compute unit, and all arithmetic happens on the chip. S and P do not fit in SRAM, so standard attention writes them out to HBM and reads them back (the hatched part of the bar).</p>`+
     `<p>FlashAttention never writes S or P to HBM, yet its output is exactly the same. How, and what does it save?</p>`,
     `<p>${M8} 在 H100 上读一份 8K token 的文档（prefill），每个 ${DEF('head',kt('head'))} 分三步算 attention：</p>${f3}`+
     `<p>S 为每对 token 存一个${DEFK('得分','fa_score')}，8K token 时一个 head 的 S 就有 ${fB(t.s,'M')}；${DEFK('softmax','fa_sm')} 得出的 P 一样大。</p>`+
     `<p>${DEF('HBM',TIP.hbm())}（显存）又大又慢；芯片上的 ${DEF('SRAM',TIP.sram())} 又小又快，每个计算单元几百 KB，计算只在芯片上做。SRAM 放不下 S 和 P，标准 attention 把它们写到 HBM 再读回来（图中斜纹）。</p>`+
     `<p>FlashAttention 从不把 S 和 P 写到 HBM，输出却完全一样。它怎么做到，省下了什么？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},view:'real',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess: what takes most of standard attention’s time at 8K tokens?','先猜：8K token 时，标准 attention 的时间大多花在哪？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>For one head at 8K tokens, standard attention multiplies matrices, takes the exp of every score, and moves S and P through HBM. On an H100, which of these takes the most time?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>8K token 时，一个 head 的标准 attention 要做矩阵乘法，要给每个得分算 exp，还要经 HBM 搬运 S 和 P。在 H100 上，哪一项最花时间？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'std',title:{en:'Standard attention, step by step',zh:'标准 attention，一步一步看'},view:'toy',path:'std',
  lead:()=>tx('Step through 12 tokens and count what crosses between HBM and the chip.','用 12 个 token 一步步走，数一数在 HBM 和芯片之间来回搬了多少。'),
  enter:()=>({...BASE,view:'toy',path:'std'}),expect:()=>doneStd,showMe:()=>{toEnd();},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The chart runs standard attention for real on a small example: 12 tokens, a head size of 4. Each ▸ is one step. The chip computes S, but S does not stay there: it goes out to HBM and comes back for the softmax. P does the same.</p>`,
              `<p>图里用一个小例子真的跑一遍标准 attention：12 个 token，head 维数 4。每按一次 ▸ 走一步。芯片算出 S 以后，S 并不留在芯片上：它先写到 HBM，做 softmax 时再读回来。P 也一样。</p>`)+playerHTML('fa-ic-play'),
  obs:()=>obsStd()},
 {id:'fa',title:{en:'FlashAttention, tile by tile',zh:'FlashAttention，一个 tile 一个 tile 地算'},view:'toy',path:'fa',
  lead:()=>tx('A Q tile stays on chip, K and V tiles stream past, and S is never written.','一个 Q tile 留在芯片上，K 和 V 的 tile 依次经过，S 从不写出去。'),
  enter:()=>({...BASE,view:'toy',path:'fa'}),expect:()=>doneFa,showMe:()=>{toEnd();},label:{en:'Result',zh:'结果'},
  body:()=>{const up=FXL('m_{\\text{new}} = \\max(m, \\max_j S_{ij})')+FXL('ℓ \\leftarrow ℓ \\cdot e^{m - m_{\\text{new}}} + \\sum_j e^{S_{ij} - m_{\\text{new}}}')+FXL('o \\leftarrow o \\cdot e^{m - m_{\\text{new}}} + \\sum_j e^{S_{ij} - m_{\\text{new}}} v_j');
   const upZ=FXL('m_{\\text{新}} = \\max(m, \\max_j S_{ij})')+FXL('ℓ \\leftarrow ℓ \\cdot e^{m - m_{\\text{新}}} + \\sum_j e^{S_{ij} - m_{\\text{新}}}')+FXL('o \\leftarrow o \\cdot e^{m - m_{\\text{新}}} + \\sum_j e^{S_{ij} - m_{\\text{新}}} v_j');
   return tx(`<p>FlashAttention cuts Q, K and V into ${DEF('tiles',TIP.tile())} of 4 rows: one Q tile stays in ${DEF('SRAM',TIP.sram())} while the K and V tiles stream past, and each 4 × 4 tile of scores is used at once and dropped, so S never goes to HBM.</p>`+
     `<p>The hard part is softmax: it needs a whole row of scores, and each tile sees only part of it. FlashAttention solves it with an ${DEFK('online softmax','fa_on')}: each row keeps a running max m and a running sum ℓ, and when the max rises, the sum and the output built so far are rescaled once, so the result is exact. Part 6 of the lab ${labLink('softmax-numerics')} works it through block by block.</p>`+
     `<p>After each K/V tile, row i updates like this:</p>${up}<p>At the end, the row’s output is ${mx('O = \\frac{o}{ℓ}')}.</p>`,
     `<p>FlashAttention 把 Q、K、V 切成每块 4 行的 ${DEF('tile',TIP.tile())}：一个 Q tile 留在 ${DEF('SRAM',TIP.sram())} 里，K 和 V 的 tile 依次进来，4 × 4 的得分用完就丢，S 从不写到 HBM。</p>`+
     `<p>难处在 softmax：它要用一整行得分，每个 tile 却只有一部分。FlashAttention 用 ${DEFK('online softmax','fa_on')} 解决：每一行记下累计最大值 m 和累计和 ℓ，最大值变大时，就把累计和与已经算出的输出整体缩放一次，结果是精确的。逐块怎么算，见 ${labLink('softmax-numerics')}这个 Lab 第 6 部分。</p>`+
     `<p>每算完一个 K/V tile，第 i 行这样更新：</p>${upZ}<p>最后这一行的输出是 ${mx('O = \\frac{o}{ℓ}')}。</p>`)+playerHTML('fa-ic-play');},
  obs:()=>obsFa()},
 {id:'len',title:{en:'Longer context',zh:'上下文变长'},view:'real',
  lead:()=>tx('Drag the context from 1K up to 128K tokens: what does FlashAttention save, and why half?','把上下文从 1K 拖到 128K：FlashAttention 省下了什么？为什么正好省一半？'),
  enter:()=>({...BASE}),expect:()=>seenN.has(NS.length-1),showMe:{ni:NS.length-1},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>FlashAttention moves about half as much as standard attention: it trades the round trips of S and P for one re-read of K and V per Q tile. Count the 128 scores between one Q tile (128 rows) and one token of the context:</p>`+
     `<ul class="fa-pair"><li><b>Standard attention</b> writes them out as S, reads them back, writes them out as P and reads them back: 4 × 128 = 512 numbers.</li>`+
     `<li><b>FlashAttention</b> keeps S and P in SRAM; instead, the Q tile ${DEF('reads that token’s key and value from HBM again',TIP.count())}, 128 numbers each: 128 + 128 = 256 numbers.</li></ul>`+
     `<p>Exactly half, because the head size happens to equal the Q tile’s rows: both are 128. Drag the context length:</p>`,
     `<p>FlashAttention 搬的量大约只有标准 attention 的一半：它用“每个 Q tile 重读一遍 K 和 V”，换掉了“S 和 P 写出去再读回来”。拿一个 Q tile（128 行）和上下文里一个 token 之间的 128 个得分来数：</p>`+
     `<ul class="fa-pair"><li><b>标准 attention</b> 把它们作为 S 写出去、读回来，再作为 P 写出去、读回来：4 × 128 = 512 个数值。</li>`+
     `<li><b>FlashAttention</b> 把 S 和 P 留在 SRAM 里，代价是这个 Q tile 要${DEF('从 HBM 重读这个 token 的 key 和 value',TIP.count())}，各 128 维：128 + 128 = 256 个数值。</li></ul>`+
     `<p>正好一半，是因为 head 的维数恰好等于 Q tile 的行数，都是 128。拖动上下文长度：</p>`)+nCtl('fa-ic-N',true),
  obs:()=>obsLen()},
 {id:'tile',title:{en:'Choose a tile size',zh:'选 tile 大小'},view:'real',gauge:true,
  lead:()=>tx('Bigger tiles mean fewer passes over K and V, but they must fit on chip.','tile 越大，扫 K 和 V 的遍数越少，但它们必须放得进芯片。'),
  enter:()=>{lv=128; lvTried=new Set([128]); return {...BASE};},expect:()=>lv!==128,showMe:()=>setLv(256),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Each Q tile reads all of K and V once, so a Q tile twice as tall halves those passes. But the Q tile, the K and V tiles, the tile of scores and the output being built must all fit together in the 227 ${DEF('KiB',kt('kib'))} of SRAM one thread block can use (a thread block is the group of GPU threads that handles one Q tile).</p><p>The tiles are square, for ${M8} at 8K tokens. Pick a tile size and see how much it moves through HBM and whether it fits in SRAM.</p>`,
              `<p>每个 Q tile 都要把 K 和 V 整个读一遍，所以 Q tile 高一倍，读 K 和 V 的遍数就少一半。但 Q tile、K/V tile、得分 tile 和正在累加的输出，要一起放进一个 thread block 能用的 227 ${DEF('KiB',kt('kib'))} SRAM（thread block 是负责一个 Q tile 的一组 GPU 线程）。</p><p>这里的 tile 都是正方形，模型是 ${M8}，上下文 8K token。选一种 tile 大小，看看它要经 HBM 搬多少数据，SRAM 又放不放得下。</p>`)+lvCtl(),
  obs:()=>obsTile()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:'real',
  lead:()=>tx('Standard attention moves S and P through HBM; FlashAttention keeps them in SRAM, with the same output.','标准 attention 经 HBM 搬运 S 和 P；FlashAttention 把它们留在 SRAM 里，输出不变。'),
  enter:()=>({...BASE}),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const stepsN=()=>S.path==='fa'?TIL.steps.length:STD_STEPS.length-1;

function verdict(){
  const pt=pred?esc(predT(pred)):'', rt=esc(predT(PRED_RIGHT));
  if(!pred) return tx(`<p>You skipped the prediction. The right answer is “${rt}”.</p>`,`<p>你跳过了预测。正确答案是“${rt}”。</p>`);
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsStd(){
  /* shown from step 0, so the rail does not grow under a sticky stage at the first ▸ (R7.9) */
  if(!doneStd&&tstep===0) return tx('<p class="fa-runl"><b>Step 0 of 6: nothing has crossed between HBM and the chip yet.</b> Press ▸ to take the first step.</p>','<p class="fa-runl"><b>第 0 步（共 6 步）：HBM 和芯片之间还什么都没搬。</b>按 ▸ 走第一步。</p>');
  if(!doneStd) return tx(`<p class="fa-runl"><b>Step ${tstep} of ${STD_STEPS.length-1}: ${fint(stdSoFar(tstep))} numbers have crossed between HBM and the chip so far.</b> Go on to the last step.</p>`,
                         `<p class="fa-runl"><b>第 ${tstep} 步（共 ${STD_STEPS.length-1} 步）：到目前为止，在 HBM 和芯片之间搬了 ${fint(stdSoFar(tstep))} 个数值。</b>继续走到最后一步。</p>`);
  const t=times(8192,128);
  return tx(`<p><b>The 12-token example moved ${fint(STD_TOTAL)} numbers between HBM and the chip, and ${fint(SP_TOY)} of them were S and P going out and coming back.</b></p>`,
            `<p><b>这个 12 token 的例子在 HBM 和芯片之间搬了 ${fint(STD_TOTAL)} 个数值，其中 ${fint(SP_TOY)} 个是 S 和 P 写出去又读回来。</b></p>`)+verdict()+
    tx(`<p>Why: at 8K tokens one head moves ${fB(traffic(8192,128,128).sp,'M')} of S and P, which takes about ${fUs(t.sp)} µs even at the H100’s full 3.35 TB/s. The multiplies ${mx('QK^{T}')} and ${mx('P \\cdot V')} need about ${fUs(t.qk)} µs each at full tensor-core speed, and the exps about ${fUs(t.exp)} µs. So standard attention waits on memory: it is ${DEF('memory-bound',TIP.membound())}.</p>`,
       `<p>原因：8K token 时，一个 head 要搬 ${fB(traffic(8192,128,128).sp,'M')} 的 S 和 P，即使按 H100 满速 3.35 TB/s 也要约 ${fUs(t.sp)} µs。矩阵乘法 ${mx('QK^{T}')} 和 ${mx('P \\cdot V')} 按 tensor core 满速各只要约 ${fUs(t.qk)} µs，算 exp 也约 ${fUs(t.exp)} µs。所以标准 attention 在等显存：它是${DEF('显存带宽受限',TIP.membound())}的。</p>`);
}
function rowRise(){const st=TIL.steps.filter(s=>s.i===0), k=st.findIndex((s,j)=>j>0&&s.f[0]<1); return {j:k, from:st[k].mOld[0], to:st[k].m[0], f:st[k].f[0]};}
function obsFa(){
  if(!doneFa&&tstep===0) return tx('<p class="fa-runl"><b>Tile pair 0 of 9 (3 Q tiles, each paired with 3 K/V tiles): nothing has moved yet.</b></p>','<p class="fa-runl"><b>第 0 对 tile（3 个 Q tile 各配 3 个 K/V tile，共 9 对）：还什么都没搬。</b></p>');
  if(!doneFa) return tx(`<p class="fa-runl"><b>Tile pair ${tstep} of ${TIL.steps.length}: ${fint(faSoFar(tstep))} numbers have crossed so far, and no score has left the chip.</b> Go on to the last tile pair.</p>`,
                        `<p class="fa-runl"><b>第 ${tstep} 对 tile（共 ${TIL.steps.length} 对）：到目前为止搬了 ${fint(faSoFar(tstep))} 个数值，没有一个得分离开芯片。</b>继续走到最后一对。</p>`);
  const r=rowRise();
  return tx(`<p><b>FlashAttention moved ${fint(FA_TOTAL)} numbers instead of ${fint(STD_TOTAL)}, and its output matches standard attention’s: the largest difference is ${sci(MAXDIFF)}, a rounding difference in the last bits.</b></p>`+
     `<p>Why: it reads Q once and writes O once, but reads K and V once for each of the 3 Q tiles; S and P never leave the chip. The rescale keeps the softmax exact: row 1’s max rose from ${f2(r.from)} to ${f2(r.to)} at K/V tile ${r.j+1}, so its running sum and partial output were multiplied by ${f2(r.f)} first, as ${J('row','the table under the chart')} shows. This is the online softmax at work.</p>`,
     `<p><b>FlashAttention 搬了 ${fint(FA_TOTAL)} 个数值，标准 attention 搬了 ${fint(STD_TOTAL)} 个；两者的输出一致：最大的差是 ${sci(MAXDIFF)}，只是最后几位的舍入不同。</b></p>`+
     `<p>原因：Q 只读一遍，O 只写一遍，K 和 V 则被 3 个 Q tile 各读一遍；S 和 P 从不离开芯片。缩放让 softmax 保持精确：第 1 行的最大值在第 ${r.j+1} 个 K/V tile 从 ${f2(r.from)} 升到 ${f2(r.to)}，所以它的累计和与已经算出的输出都先乘上 ${f2(r.f)}，再加上这个 tile（见${J('row','图下的表')}）。</p>`);
}
function obsLen(){
  const N=N_(), t=traffic(N,128,128), u=unitOf(t.std), b8=traffic(8192,128,128), big=traffic(NS[NS.length-1],128,128), k=N/8192;
  const PART={2:{en:'half',zh:'一半'},4:{en:'a quarter',zh:'四分之一'},8:{en:'an eighth',zh:'八分之一'}};
  let res;   // the change against the 8K start (R1.9)
  if(N===8192) res=tx(`<p><b>Back at 8K tokens: standard attention moves ${fB(b8.std,'M')} for one head, and FlashAttention ${fB(b8.fa,'M')}.</b></p>`,
                      `<p><b>回到 8K token：一个 head 的标准 attention 要搬 ${fB(b8.std,'M')}，FlashAttention 要搬 ${fB(b8.fa,'M')}。</b></p>`);
  else if(k>1) res=tx(`<p><b>From 8K to ${fTok(N)} tokens, ${fint(k)} times as long, one head’s standard attention moves ${fB(t.std,u)}, about ${fint(Math.round(t.std/b8.std))} times as much as at 8K. FlashAttention moves ${fB(t.fa,u)}, still about half of standard attention’s.</b></p>`,
                      `<p><b>上下文从 8K 拉长到 ${fTok(N)}（${fint(k)} 倍），一个 head 的标准 attention 要搬 ${fB(t.std,u)}，约是 8K 时的 ${fint(Math.round(t.std/b8.std))} 倍；FlashAttention 要搬 ${fB(t.fa,u)}，仍然约是标准 attention 的一半。</b></p>`);
  else {const q=PART[Math.round(1/k)];
    res=tx(`<p><b>At ${fTok(N)} tokens, ${q.en} of the 8K context, one head’s standard attention moves ${fB(t.std,u)}; at 8K it moved about ${fint(Math.round(b8.std/t.std))} times as much. FlashAttention moves ${fB(t.fa,u)}, still about half of standard attention’s.</b></p>`,
           `<p><b>上下文缩短到 ${fTok(N)}（8K 的${q.zh}），一个 head 的标准 attention 只搬 ${fB(t.std,u)}，8K 时是它的约 ${fint(Math.round(b8.std/t.std))} 倍；FlashAttention 搬 ${fB(t.fa,u)}，仍然约是标准 attention 的一半。</b></p>`);}
  if(!seenN.has(NS.length-1)) return res+tx('<p class="small">Drag it up to 128K.</p>','<p class="small">把它拖到 128K。</p>');
  return res+tx(`<p>Why: the number of scores grows with the square of the context length, while each score costs a fixed amount: 4 numbers (8 bytes) in standard attention, and about 2 (4 bytes) in FlashAttention once its re-reads are spread over the scores. So FlashAttention stays at about half.</p>`,
     `<p>原因是，得分的数量随上下文长度的平方增长，每个得分要搬的量却是固定的：标准 attention 是 4 个数值（8 字节），FlashAttention 把重读摊到每个得分上，约 2 个（4 字节）。所以 FlashAttention 始终约是一半。</p>`);
}
function obsTile(){
  const t=traffic(8192,128,lv), m=sram(lv,lv,128), std=traffic(8192,128,128).std, t128=traffic(8192,128,128).fa;
  /* real kernels' tiles (Dao-AILab/flash-attention: flash_fwd_launch_template.h, hopper/tile_size.h), so the budget is not taken for theirs (B6) */
  const real=tx('Real kernels hold the scores and the output in registers and use 128 × 64 tiles at head size 128 on an A100 (FlashAttention‑2), 128 × 176 on an H100 (FlashAttention‑3, without a causal mask).',
                '真实的 kernel 把得分和输出放在寄存器里：head 维数 128 时，A100 上的 FlashAttention‑2 用 128 × 64 的 tile，H100 上的 FlashAttention‑3 用 128 × 176（无因果 mask）。');
  if(lv===128) return tx(`<p><b>Tiles of 128 rows: FlashAttention moves ${fB(t.fa,'M')}, half of standard attention’s ${fB(std,'M')}, and the tiles take ${fK(m.total)} of the 227 KiB.</b> Pick another size to compare.</p>`,
                         `<p><b>128 行的 tile：FlashAttention 搬 ${fB(t.fa,'M')}，是标准 attention ${fB(std,'M')} 的一半；这对 tile 占 227 KiB 里的 ${fK(m.total)}。</b>选一个别的大小来比较。</p>`);
  if(!m.fits) return tx(`<p><b>Tiles of ${lv} rows would move only ${fB(t.fa,'M')}, but one pair needs ${fK(m.total)} of SRAM, more than the 227 KiB a thread block can use: in this lab’s budget the kernel cannot run with them.</b></p><p>Why: the score tile grows with ${mx('B_r \\times B_c')}, so doubling both sides makes it 4 times larger. The largest square tile that fits this budget is 128 rows. ${real}</p>`,
                        `<p><b>${lv} 行的 tile 只要搬 ${fB(t.fa,'M')}，但一对 tile 要 ${fK(m.total)} 的 SRAM，超过一个 thread block 能用的 227 KiB：按这个 Lab 的预算，kernel 用这个大小跑不起来。</b></p><p>原因：得分 tile 的大小是 ${mx('B_r \\times B_c')}，两边都翻倍，它就变成 4 倍。这个预算下放得下的最大正方形 tile 是 128 行。${real}</p>`);
  const more=t.fa>std*1.02, same=!more&&t.fa>std*0.98, bps=4*128/lv, x=fint(Math.round(t.fa/t128));
  const res=tx(`<p><b>Tiles of ${lv} rows fit easily (${fK(m.total)}). FlashAttention now moves ${fB(t.fa,'M')}: about ${x} times the ${fB(t128,'M')} of 128-row tiles, and ${more?'more than':(same?'about the same as':'less than')} standard attention’s ${fB(std,'M')}.</b></p>`,
               `<p><b>${lv} 行的 tile 很容易放下（${fK(m.total)}）。FlashAttention 现在要搬 ${fB(t.fa,'M')}：约是 128 行 tile 时 ${fB(t128,'M')} 的 ${x} 倍，${more?'比':(same?'和':'比')}标准 attention 的 ${fB(std,'M')}${more?'还多':(same?'差不多':'少')}。</b></p>`);
  return res+tx(`<p>Why: with ${lv}-row tiles there are ${fint(t.passes)} Q tiles, and each reads all of K and V again, ${fint(bps)} bytes per score, ${bps>8?`${fint(bps/8)} times the 8 bytes of S and P`:'as many as the 8 bytes of S and P'}. ${real}</p>`,
                `<p>原因：${lv} 行的 tile 一共有 ${fint(t.passes)} 个 Q tile，每个都要把 K 和 V 整个再读一遍，每个得分 ${fint(bps)} 字节，${bps>8?`是 S 和 P 的 8 字节的 ${fint(bps/8)} 倍`:'和 S、P 的 8 字节一样多'}。${real}</p>`);
}
function remember(){
  const t=times(8192,128), r=rowRise(), m128=sram(128,128,128), m256=sram(256,256,128), t8=traffic(8192,128,128);
  return tx(`<ul class="fa-remlist"><li><b>Standard attention is slow at long context because it moves S and P through the large, slow HBM, not because of the arithmetic.</b> S and P hold one number for every pair of tokens, so they grow with the square of the context and soon outweigh everything else that moves. At 8K tokens, moving them takes about ${fUs(t.sp)} µs per head, while ${mx('QK^{T}')} and ${mx('P \\cdot V')} take about ${fUs(t.qk)} µs each.</li>`+
    `<li><b>FlashAttention keeps S and P in the small, fast SRAM and still gets the same output, thanks to the online softmax</b> (the lab ${labLink('softmax-numerics')} explains it). Whenever a new tile brings a larger score, the row’s running sum and the output built so far are rescaled once, so the result is exact, not an approximation.</li>`+
    `<li><b>The tile size trades SRAM for passes over K and V.</b> Bigger tiles mean fewer passes and less traffic, so pick the largest tile that still fits in the SRAM of one thread block. For ${M8} that is 128 rows, because tiles of 256 rows would need ${fK(m256.total)}, more than the 227 KiB a thread block can use.</li>`+
    `<li><b>What it buys: one fused kernel instead of three, so S and P never make the round trip through HBM.</b> Measured, GPT‑2 medium’s attention takes 7.3 ms instead of 41.7 ms (forward and backward on an A100, ${cite(SRC.fa1,'FlashAttention paper')}), and FlashAttention‑3 reaches up to 740 TFLOP/s on an H100, 75% of its peak (${cite(SRC.fa3,'Shah et al., 2024')}). It also never stores S, which at 128K tokens would take ${fB(traffic(131072,128,128).s,'G')} for one head.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “FlashAttention approximates attention.” It computes the same softmax, so its output differs from standard attention’s only by rounding.</p><p>✗ “FlashAttention does fewer FLOPs.” It does the same multiplies and exps; in training it even computes S a second time in the backward pass instead of storing it. It is faster because it fuses the whole attention into one kernel, so S and P never leave the chip: standard attention waits on HBM, FlashAttention on its arithmetic.</p><p>✗ “FlashAttention makes the KV cache smaller.” K and V keep their size: FlashAttention only avoids storing S and P. Sharing keys and values across query heads (GQA) or compressing them into a small latent vector (MLA) is what shrinks the cache: see ${labLink('gqa-mla')}.</p></div>${A.shell.toScenarios()}`,
    `<ul class="fa-remlist"><li><b>长上下文时标准 attention 慢，是因为它要经又大又慢的 HBM 搬运 S 和 P，不是因为计算。</b>S 和 P 给每一对 token 存一个数值，所以它们随上下文长度的平方增长，很快就比别的数据都大。8K token 时，一个 head 搬 S 和 P 约要 ${fUs(t.sp)} µs，而 ${mx('QK^{T}')} 和 ${mx('P \\cdot V')} 各只要约 ${fUs(t.qk)} µs。</li>`+
    `<li><b>FlashAttention 把 S 和 P 留在又小又快的 SRAM 里，靠 online softmax 仍然得到同样的输出</b>（${labLink('softmax-numerics')}这个 Lab 讲了 online softmax）。每当新的 tile 带来更大的得分，这一行的累计和与已经算出的输出就整体缩放一次，所以结果是精确的，不是近似。</li>`+
    `<li><b>tile 大小是用 SRAM 换扫 K 和 V 的遍数。</b>tile 越大，扫的遍数越少，搬的数据也越少。所以要选一个 thread block 的 SRAM 放得下的最大 tile。${M8} 最大只能用 128 行的 tile，因为 256 行的 tile 要 ${fK(m256.total)}，超过了一个 thread block 能用的 227 KiB。</li>`+
    `<li><b>它换来的是：一个融合的 kernel 代替三个，S 和 P 不再经 HBM 来回。</b>实测中，GPT‑2 medium 的 attention 从 41.7 ms 降到 7.3 ms（A100 上的前向加反向，${cite(SRC.fa1,'FlashAttention 论文')}）；FlashAttention‑3 在 H100 上最高达到 740 TFLOP/s，是峰值的 75%（${cite(SRC.fa3,'Shah 等，2024')}）。它也从不存 S，而 128K token 时一个 head 的 S 就要 ${fB(traffic(131072,128,128).s,'G')}。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “FlashAttention 是近似计算。”它算的是同一个 softmax，所以输出和标准 attention 只差舍入。</p><p>✗ “FlashAttention 少算了 FLOPs。”乘法和 exp 一样多；训练时它甚至在反向传播里把 S 再算一遍，而不是存下来。它快，是因为它把整个 attention 融合成一个 kernel，S 和 P 从不离开芯片：标准 attention 在等 HBM，FlashAttention 在等自己的计算。</p><p>✗ “FlashAttention 让 KV cache 变小。”K 和 V 的大小不变：FlashAttention 只是不存 S 和 P。让几个 query head 共用 key 和 value（GQA），或把它们压缩成一个小的 latent 向量（MLA），才能缩小 cache：见 ${labLink('gqa-mla')}。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 fa1:'https://arxiv.org/abs/2205.14135', fa2:'https://arxiv.org/abs/2307.08691', fa3:'https://arxiv.org/abs/2407.08608',
 fwd:'https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h',
 hop:'https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h',
 vllm:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py',
 sdpa:'https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html',
 tune:'https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html',
 fdec:'https://crfm.stanford.edu/2023/10/12/flashdecoding.html'
};
const PRESETS=[
 {id:'gpt2',t:{en:`${MGPT} with a 1K-token context`,zh:`${MGPT}，1K token 的上下文`},d:{en:'A small model with heads of 64 numbers',zh:'小模型，每个 head 64 维'},s:{d:64,ni:0,Br:128,Bc:128,dec:false}},
 {id:'l8',t:{en:`${M8} reads an 8K document`,zh:`${M8} 读一份 8K 的文档`},d:{en:'The prefill of a long prompt',zh:'一个长 prompt 的 prefill'},s:{d:128,ni:3,Br:128,Bc:128,dec:false}},
 {id:'long',t:{en:`${M8} reads a 128K document`,zh:`${M8} 读一份 128K 的文档`},d:{en:'The longest context the model supports',zh:'模型支持的最长上下文'},s:{d:128,ni:7,Br:128,Bc:128,dec:false}},
 {id:'dec',t:{en:'One decode step at 8K',zh:'8K 时的一个 decode 步'},d:{en:'Writing one new token after an 8K prompt',zh:'读完 8K prompt 后写一个新 token'},s:{d:128,ni:3,Br:128,Bc:128,dec:true}}
];
const presetNow=()=>S.view==='toy'?-1:PRESETS.findIndex(p=>p.s.d===S.d&&p.s.ni===S.ni&&p.s.Br===S.Br&&p.s.Bc===S.Bc&&p.s.dec===S.dec);
const curT=()=>traffic(N_(),S.d,S.Br,S.dec?1:null);
function scenHTML(i){
  const p=PRESETS[i], t=traffic(NS[p.s.ni],p.s.d,p.s.Br,p.s.dec?1:null), u=unitOf(t.std);
  const head=tx(`<p><b>One head reads and writes ${fB(t.std,u)} of HBM with standard attention and ${fB(t.fa,u)} with FlashAttention.</b></p>`,`<p><b>一个 head 读写 HBM 的量：标准 attention 是 ${fB(t.std,u)}，FlashAttention 是 ${fB(t.fa,u)}。</b></p>`);
  if(p.id==='gpt2') return head+tx(`<p>${MGPT}’s heads are 64 numbers wide, so each re-read of K and V costs half as much as for ${M8}. Measured, the gap is larger: on GPT${NB}2 medium the FlashAttention paper reports ${DEF('about 89% less HBM traffic',tx('FlashAttention paper (Dao et al., 2022), Figure 2, measured on an A100 for GPT‑2 medium (1K tokens, head size 64, 16 heads, batch 64), forward and backward: 40.3 GB of HBM reads and writes for standard attention and 4.4 GB for FlashAttention; 41.7 ms against 7.3 ms.','FlashAttention 论文（Dao 等，2022）图 2 在 A100 上测了 GPT‑2 medium 的前向加反向（1K token，head 维数 64，16 个 head，batch 64）。标准 attention 读写 HBM 40.3 GB，FlashAttention 4.4 GB；耗时 41.7 ms 对 7.3 ms。'))} with FlashAttention (${cite(SRC.fa1,'Dao et al., 2022')}). It measures forward and backward: in the backward pass standard attention moves its ${mx('N \\times N')} matrices again, while FlashAttention recomputes S on the chip.</p>`,
                                   `<p>${MGPT} 的 head 是 64 维，每次重读 K 和 V 的代价只有 ${M8} 的一半。实测的差距更大：FlashAttention 论文在 GPT${NB}2 medium 上测得，用 FlashAttention 时 ${DEF('HBM 流量少约 89%',tx('FlashAttention paper (Dao et al., 2022), Figure 2, measured on an A100 for GPT‑2 medium (1K tokens, head size 64, 16 heads, batch 64), forward and backward: 40.3 GB of HBM reads and writes for standard attention and 4.4 GB for FlashAttention; 41.7 ms against 7.3 ms.','FlashAttention 论文（Dao 等，2022）图 2 在 A100 上测了 GPT‑2 medium 的前向加反向（1K token，head 维数 64，16 个 head，batch 64）。标准 attention 读写 HBM 40.3 GB，FlashAttention 4.4 GB；耗时 41.7 ms 对 7.3 ms。'))}（${cite(SRC.fa1,'Dao 等，2022')}）。它测的是前向加反向：反向传播里标准 attention 又要搬一遍 ${mx('N \\times N')} 的矩阵，FlashAttention 则在芯片上重算 S。</p>`);
  if(p.id==='l8') return head+tx(`<p>Standard attention moves S and P through HBM; FlashAttention re-reads K and V for each of its ${fint(t.passes)} Q tiles instead, half the bytes per score with tiles of 128 rows. An H100’s 50 MB L2 cache probably serves many of these re-reads, so the real traffic lies between ${fB(t.floor)} (Q, K and V read once, O written once) and this count.</p>`,
                                 `<p>标准 attention 经 HBM 搬运 S 和 P；FlashAttention 改为让 ${fint(t.passes)} 个 Q tile 各重读一遍 K 和 V，用 128 行的 tile 时，每个得分的字节数是一半。H100 的 50 MB L2 cache 很可能送出其中不少重读，所以真实流量在 ${fB(t.floor)}（Q、K、V 各读一次，O 写一次）和这里算出的值之间。</p>`);
  if(p.id==='long'){const fl=4*t.N*t.N*t.d*LLAMA.H;
    return head+tx(`<p>Standard attention would need ${fB(t.s,'G')} to hold one head’s S, and ${fLay(t.s*LLAMA.H)} for a layer’s 32 heads, far more than an H100’s 80 GB: it cannot even allocate S, so long-context serving needs a fused kernel such as FlashAttention. FlashAttention removes the traffic of S and P, not the arithmetic: at 128K one layer’s attention still does about ${fint(Math.round(fl/1e12))} TFLOP, about ${fixd(fl/HW.flops,2)} s even at the H100’s full speed (half with the causal mask). ${J('math','“Show the math”')} has the product.</p>`,
                     `<p>标准 attention 光放一个 head 的 S 就要 ${fB(t.s,'G')}，一层 32 个 head 要 ${fLay(t.s*LLAMA.H)}，远超 H100 的 80 GB。它连 S 都分配不下，所以长上下文的服务离不开 FlashAttention 这类融合的 kernel。FlashAttention 省掉的是 S 和 P 的流量，不是计算：128K 时一层的 attention 仍要做约 ${fint(Math.round(fl/1e12))} TFLOP 的运算，H100 满速也要约 ${fixd(fl/HW.flops,2)} 秒（有因果 mask 时减半）。算式见${J('math','“展开公式”')}。</p>`);}
  return head+tx(`<p>FlashAttention changes almost nothing here: one new token gives one row of ${fint(NS[p.s.ni])} scores (${fK(t.s)}), and both paths must read all of K and V, the KV cache. Decode is limited by reading the KV cache. With one query row per head, one thread block per Q tile gives few blocks: at batch size 1, ${M8}’s 32 heads give only 32 thread blocks for the H100’s 132 SMs. Flash${NB}Decoding (${cite(SRC.fdec,'Dao et al., 2023')}) splits the keys across thread blocks and combines their partial results with the same rescale.</p>`,
                 `<p>这里 FlashAttention 几乎没有改变什么：一个新 token 只有一行 ${fint(NS[p.s.ni])} 个得分（${fK(t.s)}），两条路径都得把 K 和 V（也就是 KV cache）整个读一遍。decode 的瓶颈是读 KV cache。每个 head 只有一行 query，而一个 Q tile 对应一个 thread block，所以 block 很少：batch 为 1 时，${M8} 的 32 个 head 只给 H100 的 132 个 SM 提供 32 个 thread block。Flash${NB}Decoding（${cite(SRC.fdec,'Dao 等，2023')}）把 key 分给多个 thread block，再用同样的缩放把它们的部分结果合起来。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chSet=(Br,Bc)=>tx(`Q tile ${Br} rows, K/V tile ${Bc} rows`,`Q tile ${Br} 行，K/V tile ${Bc} 行`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.Br,S.Bc), best=chBest(), c=chChanges(S.Br,S.Bc);
  let h;
  const good=`<p class="okline fa-mt">`+tx(`<b>Goal met: the tiles take ${fK(me.m.total)} of the 227 KiB, and one head moves ${fB2(me.t.fa)}.</b>`,`<b>达成目标：这对 tile 占 227 KiB 里的 ${fK(me.m.total)}，一个 head 搬 ${fB2(me.t.fa)}。</b>`);
  if(me.ok&&S.Br===best.Br&&S.Bc===best.Bc) h=good+tx(' And that is the smallest change: only the Q tile grew.</p>','而且这是改动最少的做法：只把 Q tile 变大了。</p>');
  else if(me.ok) h=good+tx(' But you changed more than needed.</p>','不过你改得比需要的多。</p>')+tx(`<p>The K/V tile does not change the traffic, so a Q tile of ${best.Br} rows with the K/V tile left at ${best.Bc} rows meets the goal too.</p>`,`<p>K/V tile 不影响流量。所以 Q tile 用 ${best.Br} 行、K/V tile 保持 ${best.Bc} 行，也能达标。</p>`);
  else h=(!me.fits?tx(`<p class="badline fa-mt"><b>Not yet: this tile pair needs ${fK(me.m.total)}, more than the 227 KiB one thread block can use, so in this lab’s budget the kernel cannot run.</b></p>`,`<p class="badline fa-mt"><b>还没达到：这对 tile 要 ${fK(me.m.total)}，超过一个 thread block 能用的 227 KiB，按这个 Lab 的预算，kernel 跑不起来。</b></p>`)
      :tx(`<p class="badline fa-mt"><b>Not yet: the tiles fit, but one head moves ${fB2(me.t.fa)}, more than 1.50 GiB.</b></p>`,`<p class="badline fa-mt"><b>还没达到：tile 放得下，但一个 head 要搬 ${fB2(me.t.fa)}，超过 1.50 GiB。</b></p>`))+
    tx(`<p class="small">Hint: the traffic is mostly K and V read again, once for each Q tile, so only the Q tile’s height changes it. The K/V tile only takes SRAM. With heads of 64 numbers, every tile is half the size it was for ${M8}.</p>`,
       `<p class="small">提示：流量主要是每个 Q tile 重读一遍 K 和 V，所以只有 Q tile 的高度会改变它。K/V tile 只占 SRAM。head 是 64 维时，每个 tile 只有 ${M8} 时的一半大。</p>`)+
    (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const rows=[]; TILES.forEach(Br=>TILES.forEach(Bc=>rows.push(chEval(Br,Bc))));
  const shown=rows.filter(e=>e.ok||(e.Br===S.Br&&e.Bc===S.Bc)||(e.Br===CH.start.Br&&e.Bc===CH.start.Bc));
  h+=A.ui.cmpChart({title:tx(`One head’s HBM traffic in GiB for ${M1} at 32K tokens on an H100, with the SRAM the tile pair needs (goal: fits in 227 KiB and at most 1.50 GiB)`,`${M1} 在 H100 上读 32K token 时一个 head 的 HBM 流量（GiB），并注明这对 tile 要的 SRAM（目标：放得进 227 KiB，且不超过 1.50 GiB）`),
    brief:true,hiddenText:n=>tx(`The other ${n} tile pairs miss the goal or do not fit, and are not shown.`,`其余 ${n} 对 tile 没达标或放不下，图里没有列出。`),
    aria:tx('One head’s HBM traffic: ','一个 head 的 HBM 流量：')+shown.map(e=>`${chSet(e.Br,e.Bc)}${tx(': ','：')}${fB2(e.t.fa)}${e.fits?'':tx(', does not fit','，放不下')}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:tx(`${e.ok?'✓ ':''}${e.Br} × ${e.Bc} (SRAM ${fK(e.m.total)}${e.fits?'':', does not fit'})`,`${e.ok?'✓ ':''}${e.Br} × ${e.Bc}（SRAM ${fK(e.m.total)}${e.fits?'':'，放不下'}）`),value:e.t.fa/GiB,text:fB2(e.t.fa),ok:e.ok,me:e.Br===S.Br&&e.Bc===S.Bc,
      tags:(e.Br===CH.start.Br&&e.Bc===CH.start.Bc?[{text:tx('initial setup','初始设置')}]:[]).concat(e.Br===S.Br&&e.Bc===S.Bc?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:CH.goal/GiB,goalLabel:tx('goal: 1.50 GiB','目标：1.50 GiB'),max:11,okText:tx('✓ fits and meets the goal','✓ 放得下且达标'),noText:tx('misses the goal or does not fit','没达标或放不下'),goalText:tx('goal: at most 1.50 GiB','目标：不超过 1.50 GiB')});
  const st=chEval(CH.start.Br,CH.start.Bc), m2=sram(256,128,CH.d);
  h+=tx(`<p class="small">In the initial setup (64 × 64), one head moves ${fB2(st.t.fa)}: K and V are read again for each of ${fint(st.t.passes)} Q tiles.</p><p class="small">Why a Q tile of ${best.Br} rows works: it cuts the passes to ${fint(traffic(CH.N,CH.d,best.Br).passes)}, and with heads of 64 numbers the pair still fits (${fK(sram(best.Br,best.Bc,CH.d).total)}). With a K/V tile of 128 rows it would not: the score tile doubles, and the pair needs ${fK(m2.total)}.</p>`,
        `<p class="small">初始设置（64 × 64）下，一个 head 搬 ${fB2(st.t.fa)}：${fint(st.t.passes)} 个 Q tile 各重读一遍 K 和 V。</p><p class="small">Q tile 用 ${best.Br} 行为什么行：遍数降到 ${fint(traffic(CH.N,CH.d,best.Br).passes)}，而 head 是 64 维，这对 tile 仍放得下（${fK(sram(best.Br,best.Bc,CH.d).total)}）。K/V tile 改成 128 行就不行了：得分 tile 翻倍，这对 tile 要 ${fK(m2.total)}。</p>`);
  h+=tx('<p class="small muted">Note: this counts every re-read of K and V as HBM traffic, as the FlashAttention paper does. An H100’s 50 MB L2 cache probably serves many of these re-reads, so the real traffic lies between the floor of reading Q, K and V and writing O once, and this count. Real kernels also keep the score tile and the output in registers and choose tiles by measuring speed: at head size 64, FlashAttention‑3 runs 192 × 192 tiles on an H100 (without a causal mask), which this lab’s budget rejects. Benchmark the tile sizes on your GPU before you trust a count like this one.</p>',
        '<p class="small muted">注意：这里和 FlashAttention 论文一样，把每次重读 K 和 V 都算成 HBM 流量。H100 的 50 MB L2 cache 很可能送出其中不少重读，所以真实流量在 Q、K、V 各读一次、O 写一次的下限和这里算出的值之间。真实的 kernel 还把得分 tile 和输出放在寄存器里，靠实测速度来选 tile。例如 head 维数 64 时，H100 上的 FlashAttention‑3 用 192 × 192 的 tile（无因果 mask），这个 Lab 的预算会判它放不下。相信这类估算之前，先在自己的 GPU 上实测各种 tile 大小。</p>');
  return h;
}

/* =====================================================================
   copy the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===7?'Seven':fint(STEPS.length)} short parts. Predict first, then step through standard attention and FlashAttention on a 12-token example, see how the traffic grows with the context, and choose a tile size. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再在一个 12 token 的例子上一步步走完标准 attention 和 FlashAttention，看流量怎样随上下文增长，最后选 tile 大小。第一次学从这里开始。`),
  sandbox:tx(`Four real cases, from ${MGPT} at 1K tokens to ${M8} at 128K and one decode step. Pick one to see each path’s HBM traffic.`,
             `四个真实情况，从 ${MGPT} 的 1K token 到 ${M8} 的 128K，再加一个 decode 步。选一个，看两条路径各搬多少 HBM 流量。`),
  challenge:tx(`${M1} reads 32K tokens on an H100. Choose the Q tile and the K/V tile so that they fit in the 227 KiB of SRAM one thread block can use and one head moves at most 1.50 GiB of HBM traffic.`,
               `${M1} 在 H100 上读 32K token。选好 Q tile 和 K/V tile，让它们放得进一个 thread block 能用的 227 KiB SRAM，并且一个 head 的 HBM 流量不超过 1.50 GiB。`)
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser runs one head of attention on 12 tokens with a head size of 4, both the standard way and tile by tile, and compares the outputs. The real-size traffic and SRAM numbers are formulas; ${J('math','“Show the math”')} works them out.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Standard attention as three steps (FlashAttention paper, Algorithm 0). It computes the scores ${mx('S = \\frac{QK^{T}}{\\sqrt{d}}')} and writes S to HBM. It reads S back and writes the weights ${mx('P = \\text{softmax}(S)')}. It reads P and V and writes the output ${mx('O = PV')}.</li>`+
 `<li>FlashAttention in FlashAttention‑2’s order: one Q tile on chip, every K and V tile streamed past it, a running max and sum per row, O written once. Q, K and V of the example come from a seeded generator (mulberry32, seed 71,288 = 9 × 7,919 + 17).</li>`+
 `<li>HBM traffic per head in BF16, 2 bytes a number, as in the FlashAttention paper’s algorithm: every re-read of K and V for a new Q tile counts as an HBM read. The stage, the parts, the scenarios, the challenge and the home tile all count this way.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>No causal mask: ${M8}’s attention is causal, and FlashAttention then skips the tiles above the diagonal, about half of its work and traffic; unfused code still writes the whole S.</li>`+
 `<li>Real eager code moves S more times than three steps (a scale, the mask, a softmax in FP32), and an H100’s 50 MB L2 cache probably serves many of FlashAttention’s re-reads of K and V, so its real traffic lies between reading Q, K and V and writing O once and this count. Both likely make the real gap larger than shown. The inference: under this count, FlashAttention with Q tiles of 128 rows does fewer than 128 FLOPs for each byte moved from HBM, so 3.35 TB/s would cap it below 430 TFLOP/s; FlashAttention‑3 reports up to 740 TFLOP/s in FP16 on an H100 (75% of its peak) with Q tiles of 128 rows, so most re-reads cannot come from HBM. Measured on GPT‑2 medium (A100, forward and backward), the FlashAttention paper reports 40.3 GB of HBM traffic for standard attention and 4.4 GB for FlashAttention.</li>`+
 `<li>The on-chip budget adds all tiles together. Real kernels keep K and V tiles in shared memory (often two copies, to load the next while computing) and the score tile and the output in registers, another 256 KiB per SM.</li>`+
 `<li>The times this lab shows assume full speed for each kind of work: 989.5 TFLOP/s for matrix multiplies (dense BF16), 3.35 TB/s for HBM, and 3.9 × 10<sup>12</sup> exps a second (FlashAttention‑3 paper).</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${cite(SRC.fa1,'Dao et al., 2022')} (FlashAttention: Algorithm 0, the IO analysis, GPT‑2 measurements); ${cite(SRC.fa2,'Dao, 2023')} (FlashAttention‑2: the loop order); ${cite(SRC.fa3,'Shah et al., 2024')} (FlashAttention‑3: H100 speed, exp throughput).</li>`+
 `<li>H100: 228 KiB of shared memory per SM, up to 227 KiB for one thread block (NVIDIA ${cite(SRC.tune,'Hopper tuning guide')}); 3.35 TB/s and 989.5 TFLOP/s dense BF16 as in the other labs.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器在 12 个 token、head 维数 4 的例子上，用标准方法和分块方法各跑一遍单个 head 的 attention，并比较输出。真实规模的流量和 SRAM 数字是公式算出的；${J('math','“展开公式”')}里有算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>标准 attention 分三步（FlashAttention 论文的算法 0）。第一步算出得分 ${mx('S = \\frac{QK^{T}}{\\sqrt{d}}')}，写到 HBM。第二步读回 S，算出权重 ${mx('P = \\text{softmax}(S)')} 并写出。第三步读回 P 和 V，写出输出 ${mx('O = PV')}。</li>`+
 `<li>FlashAttention 按 FlashAttention‑2 的顺序：一个 Q tile 留在芯片上，K 和 V 的每个 tile 依次经过它，每一行记着累计最大值和累计和，O 只写一次。例子里的 Q、K、V 来自带种子的生成器（mulberry32，种子 71,288 = 9 × 7,919 + 17）。</li>`+
 `<li>每个 head 的 HBM 流量按 BF16 算，每个数值 2 字节，和 FlashAttention 论文的算法一样：每个新的 Q tile 重读 K 和 V，都算一次 HBM 读取。图、各部分、应用场景、挑战和首页方块都这样算。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>没有因果 mask：${M8} 的 attention 是因果的，这时 FlashAttention 会跳过对角线上方的 tile，约省一半的计算和流量；未融合的代码仍然写出整个 S。</li>`+
 `<li>真实的 eager 代码搬 S 的次数比三步多（缩放、mask、FP32 的 softmax）。H100 的 50 MB L2 cache 很可能送出 FlashAttention 对 K、V 的不少重读，所以它的真实流量在 Q、K、V 各读一次、O 写一次的下限和这里算出的值之间。两者都可能让真实差距比这里更大。推断的依据是，按这里的算法，Q tile 为 128 行时 FlashAttention 每从 HBM 搬运 1 字节做不到 128 次运算，3.35 TB/s 会把它限制在 430 TFLOP/s 以下；而 FlashAttention‑3 论文报告，在 H100 上用 128 行的 Q tile、FP16 最高达到 740 TFLOP/s（峰值的 75%），所以大部分重读不可能来自 HBM。FlashAttention 论文在 A100 上实测 GPT‑2 medium 的前向加反向：标准 attention 读写 HBM 40.3 GB，FlashAttention 4.4 GB。</li>`+
 `<li>芯片上的预算把所有 tile 加在一起算。真实的 kernel 把 K、V 的 tile 放在共享内存里（常放两份，算这一个 tile 时预取下一个），把得分 tile 和输出放在寄存器里，每个 SM 另有 256 KiB 寄存器。</li>`+
 `<li>这个 Lab 里的耗时都假设每种工作按满速进行：矩阵乘法 989.5 TFLOP/s（稠密 BF16），HBM 3.35 TB/s，每秒 3.9 × 10<sup>12</sup> 次 exp（FlashAttention‑3 论文）。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${cite(SRC.fa1,'Dao 等，2022')}（FlashAttention：算法 0、IO 分析、GPT‑2 实测）；${cite(SRC.fa2,'Dao，2023')}（FlashAttention‑2：循环顺序）；${cite(SRC.fa3,'Shah 等，2024')}（FlashAttention‑3：H100 速度、每秒能算多少次 exp）。</li>`+
 `<li>H100：每个 SM 228 KiB 共享内存，一个 thread block 最多 227 KiB（NVIDIA ${cite(SRC.tune,'Hopper 调优指南')}）；3.35 TB/s 和稠密 BF16 989.5 TFLOP/s，和其他 Lab 一致。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>{const f3=FXL(tx('\\text{scores}\\ S = \\frac{Q K^{T}}{\\sqrt{d}}','\\text{得分}\\ S = \\frac{Q K^{T}}{\\sqrt{d}}'))+FXL(tx('\\text{attention weights}\\ P = \\text{softmax}(S)','\\text{注意力权重}\\ P = \\text{softmax}(S)'))+FXL(tx('\\text{output}\\ O = P V','\\text{输出}\\ O = P V'));
  return tx(`<p>A head computes attention in three steps: a ${DEFK('score','fa_score')} for every pair of tokens, a ${DEFK('softmax','fa_sm')} over each row of scores, and a mix of the values by the resulting weights:</p>${f3}<p>${labLink('mha')} builds this up.</p><p>A softmax can also be computed in one pass, keeping a running max and a running sum: the online softmax, in part 6 of ${labLink('softmax-numerics')}. Whether a kernel waits on memory or on arithmetic: ${labLink('roofline')}.</p>`,
            `<p>一个 head 分三步算 attention：给每一对 token 打一个${DEFK('得分','fa_score')}，对每一行得分做 ${DEFK('softmax','fa_sm')}，再按得到的权重混合 value：</p>${f3}<p>${labLink('mha')}从头讲这些。</p><p>softmax 也可以只读一遍就算出来，边读边记累计最大值和累计和，这叫 online softmax，见 ${labLink('softmax-numerics')}这个 Lab 的第 6 部分。一个 kernel 是在等显存还是在等计算，见${labLink('roofline')}。</p>`);};
const OUTCOME=()=>tx('compute how many bytes standard attention and FlashAttention move through HBM for a head, and choose tile sizes that fit on chip.','算出一个 head 的标准 attention 和 FlashAttention 各要经 HBM 搬多少字节，并选出放得进芯片的 tile 大小。');
const EXITS=()=>{
  const s8=traffic(8192,128,128).s, t=times(8192,128), fpb=(4*128)/(EL*4), ridge=HW.flops/HW.bw;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('FlashAttention never sees a whole row of scores at once, yet its output equals the softmax of the whole row. Why?','FlashAttention 从来看不到一整行得分，输出却等于整行的 softmax。为什么？'),
   o:[tx('Each row keeps its max and running sum as it goes; when the max grows, what has been added so far is rescaled once, so the end result equals reading the whole row at once.','每一行边算边记最大值和累计和；最大值变大时，把已经加起来的部分缩放一次，所以最后和一次看完整行的结果一样。'),
      tx('Each tile drops the scores far below its own max, which carry almost no weight, so the tiles together give the softmax up to a tiny error that rounding hides.','每个 tile 丢掉比自己的最大值小很多的得分，它们几乎没有权重，所以各 tile 合起来给出的 softmax 只有很小的误差，被舍入掩盖了。'),
      tx('Each tile computes its own softmax and output, and the tiles’ outputs are averaged with weights set by the number of keys in each tile, which gives the same result.','每个 tile 各自算 softmax 和输出，再按每个 tile 里有几个 key 来加权平均各 tile 的输出，结果一样。')],a:0,
   n:[null,tx('Nothing is dropped: FlashAttention is exact, and its output matches standard attention’s up to rounding.','什么都没丢：FlashAttention 是精确的，它的输出和标准 attention 只差舍入。'),
     tx('Averaging per-tile softmaxes by key count is not the softmax: a tile with larger scores must count for more. The weights come from each tile’s sum of exps, which is what the running sum tracks.','按 key 的数目平均各 tile 的 softmax 不等于整行的 softmax：得分大的 tile 应该占更多。权重来自每个 tile 的 exp 之和，这正是累计和记录的。')],
   why:tx(`Each time the max grows, the old running sum and output are rescaled once to the new max. That step is exact, because for every score already added:</p>${FXL('e^{S_{ij} - m_{\\text{new}}} = e^{S_{ij} - m_{\\text{old}}} \\times e^{m_{\\text{old}} - m_{\\text{new}}}')}<p>After the last tile, the output divided by the running sum is exactly softmax(S)·V.`,`每次最大值变大，旧的累计和与输出都缩放一次，换成以新的最大值为准。这一步是精确的，因为对已经加进去的每个得分都有：</p>${FXL('e^{S_{ij} - m_{\\text{新}}} = e^{S_{ij} - m_{\\text{旧}}} \\times e^{m_{\\text{旧}} - m_{\\text{新}}}')}<p>最后一个 tile 之后，输出除以累计和正好是 softmax(S)·V。`)},
  {q:tx('Why is standard attention memory-bound at 8K tokens on an H100, although its matrix multiplies are large?','8K token 时，标准 attention 的矩阵乘法很大，为什么在 H100 上它还是受显存带宽限制？'),
   o:[tx(`Standard attention does only about ${fint(fpb)} FLOPs for each byte it moves from HBM, while an H100 needs about ${fint(Math.round(ridge))} per byte to avoid waiting on memory.`,`标准 attention 每从 HBM 搬运 1 字节只做约 ${fint(fpb)} 次运算，而 H100 要每字节做到约 ${fint(Math.round(ridge))} 次才不必等显存。`),
      tx('Its three steps are separate kernels, and the roughly 5 µs of launch time before each one, not the bytes moved, is what holds it back.','它的三步是三个单独的 kernel，每个启动前约要 5 µs；拖慢它的是这段启动时间，而不是搬运的字节。'),
      tx(`The exps run on special-function units with about 1/${fint(Math.round(HW.flops/HW.sfu/50)*50)} of the tensor cores’ throughput, so the softmax’s arithmetic, not the bytes moved, is what holds it back.`,`exp 在特殊函数单元上算，每秒能算的次数只有 tensor core 的约 1/${fint(Math.round(HW.flops/HW.sfu/50)*50)}；拖慢它的是 softmax 的计算，而不是搬运的字节。`)],a:0,
   n:[null,tx(`Launch time is real but small next to the bytes: moving S and P, ${fB(EL*4*8192*8192,'M')} at 3.35 TB/s, takes about ${fUs(t.sp)} µs by itself, against about ${fUs(t.qk)} µs for ${mx('QK^{T}')}.`,`启动开销确实存在，但比起搬字节很小：光是按 3.35 TB/s 搬 S 和 P 的 ${fB(EL*4*8192*8192,'M')} 就要约 ${fUs(t.sp)} µs，${mx('QK^{T}')} 只要约 ${fUs(t.qk)} µs。`),
     tx(`The special-function units are slow, but at 8K one head’s ${fint(Math.round(8192*8192/1e6))} million exps take about ${fUs(t.exp)} µs, about as long as ${mx('QK^{T}')} and much less than the ${fUs(t.sp)} µs of moving S and P.`,`特殊函数单元确实慢，但 8K 时一个 head 的约 ${fint(Math.round(8192*8192/1e6)*100)} 万次 exp 只要约 ${fUs(t.exp)} µs，和 ${mx('QK^{T}')} 差不多，远少于搬 S 和 P 的 ${fUs(t.sp)} µs。`)],
   why:tx(`Roofline: a kernel waits on memory when it does fewer FLOPs for each byte moved from HBM than the GPU’s compute divided by its bandwidth. For the H100 that is:</p>${FXL(`\\frac{\\text{989.5 TFLOP/s}}{\\text{3.35 TB/s}} \\approx ${fint(Math.round(ridge))}`)}<p>Standard attention does ${fint(4*128)} FLOPs per score and moves 8 bytes for it, about ${fint(fpb)} FLOPs for each byte moved. That is why the S and P traffic takes about ${fUs(t.sp)} µs per head at 8K, and ${mx('QK^{T}')} and ${mx('P \\cdot V')} about ${fUs(t.qk)} µs each.`,`roofline：一个 kernel 每从 HBM 搬运 1 字节做的运算次数，如果少于 GPU 的算力除以带宽，它就要等显存。H100 的这个比值是：</p>${FXL(`\\frac{\\text{989.5 TFLOP/s}}{\\text{3.35 TB/s}} \\approx ${fint(Math.round(ridge))}`)}<p>标准 attention 每个得分做 ${fint(4*128)} 次运算、搬 8 字节，每搬运 1 字节约做 ${fint(fpb)} 次运算。所以 8K 时一个 head 搬 S 和 P 约要 ${fUs(t.sp)} µs，${mx('QK^{T}')} 和 ${mx('P \\cdot V')} 各约 ${fUs(t.qk)} µs。`)}],
 transfer:[
  {q:tx(`${M8} has 32 heads per layer, and at 8K tokens one layer’s S takes ${fB(s8*32,'G').replace('.0 ',' ')}. ${M70} has 64 ${DEF('query heads',kt('head')+tx(' Each query head has its own queries, so its own scores and its own S.',' 每个 query head 有自己的 query，所以有自己的得分和自己的 S。'))} of the same size. How much does one layer’s S take at 8K?`,`${M8} 每层 32 个 head，8K token 时一层的 S 占 ${fB(s8*32,'G').replace('.0 ',' ')}。${M70} 有 64 个同样大小的 ${DEF('query head',kt('head')+tx(' Each query head has its own queries, so its own scores and its own S.',' 每个 query head 有自己的 query，所以有自己的得分和自己的 S。'))}。8K 时它一层的 S 占多少？`),
   o:[tx('About 2 GiB','约 2 GiB'),tx('About 4 GiB','约 4 GiB'),tx('About 8 GiB','约 8 GiB'),tx('About 16 GiB','约 16 GiB')],a:2,
   why:tx(`Every query head has its own S, 8,192 × 8,192 scores × 2 bytes = ${fB(s8,'M')}. Twice the heads, twice the S: 64 × ${fB(s8,'M')} = ${fB(64*s8,'G').replace('.0 ',' ')}. Sharing keys and values (8 ${DEF('KV heads',tx('A KV head is one set of keys and values. With grouped-query attention several query heads share one KV head: Llama‑3.1‑8B has 32 query heads and 8 KV heads.','一个 KV head 就是一组 key 和 value。用 grouped-query attention 时，几个 query head 共用一个 KV head：Llama‑3.1‑8B 有 32 个 query head、8 个 KV head。'))} in both models) does not help here: each query head still has its own scores.`,`每个 query head 都有自己的 S：8,192 × 8,192 个得分 × 2 字节 = ${fB(s8,'M')}。head 多一倍，S 就多一倍：64 × ${fB(s8,'M')} = ${fB(64*s8,'G').replace('.0 ',' ')}。共用 key 和 value（两个模型都是 8 个 ${DEF('KV head',tx('A KV head is one set of keys and values. With grouped-query attention several query heads share one KV head: Llama‑3.1‑8B has 32 query heads and 8 KV heads.','一个 KV head 就是一组 key 和 value。用 grouped-query attention 时，几个 query head 共用一个 KV head：Llama‑3.1‑8B 有 32 个 query head、8 个 KV head。'))}）在这里帮不上忙：每个 query head 仍有自己的得分。`)},
  {q:tx(`${MG} has heads of 256 numbers, twice as wide as ${M8}’s 128. At 8K tokens, one ${M8} head’s S takes ${fB(s8,'M')}. How much does one ${MG} head’s S take?`,`${MG} 的 head 是 256 维，是 ${M8} 的 128 维的两倍宽。8K token 时，${M8} 一个 head 的 S 占 ${fB(s8,'M')}。${MG} 一个 head 的 S 占多少？`),
   o:[tx('About 64 MiB','约 64 MiB'),tx('About 128 MiB','约 128 MiB'),tx('About 256 MiB','约 256 MiB')],a:1,
   why:tx(`S holds one score per pair of tokens, 8,192 × 8,192 × 2 bytes = ${fB(s8,'M')}, whatever the head size. A wider head makes each score cost more multiplies (${mx('q \\cdot k')} over 256 numbers), and makes Q, K and V bigger, but S stays the same size.`,`S 为每一对 token 存一个得分，8,192 × 8,192 × 2 字节 = ${fB(s8,'M')}，和 head 的维数无关。head 更宽，每个得分要多做乘法（${mx('q \\cdot k')} 有 256 维），Q、K、V 也更大，但 S 一样大。`)}],
 practice:[
  {id:'attention-kv-cache',d:tx('Part 3: attention in blocks with an online softmax. Derive how the running max, the running sum and the output change from one block to the next, prove it by induction, and match the direct result for every block size.','第 3 部分：用 online softmax 分块算 attention。推导累计最大值、累计和与输出从一块到下一块怎样更新，用归纳法证明，并让每种块大小都和直接计算的结果一致。')},
  {id:'qkv-einsum',d:tx(`Part 1: scaled dot-product attention with einsum, the standard path of this lab; the follow-up on complexity says why its ${mx('N \\times N')} scores dominate the memory at long context.`,`第 1 部分：用 einsum 写 scaled dot-product attention，也就是这个 Lab 的标准路径；关于复杂度的追问讲了为什么长上下文时 ${mx('N \\times N')} 的得分占满了显存。`)}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'FlashAttention',title:l?L(l.q):tx('How does FlashAttention compute exact attention with less memory traffic?','FlashAttention 怎么在结果不变的情况下少读写显存？'),
    minutes:15,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: attention and the online softmax','前置知识：attention 和 online softmax'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage fa-stage" id="fa-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="fa-st-title"></div><div class="meter-sub" id="fa-st-sub"></div></div>`+
   `<div class="stage-actions"><div class="seg-ctl fa-view" id="fa-view" role="group" aria-label="${esc(tx('What the chart shows','图里显示什么'))}"><button type="button" data-view="real" aria-pressed="true">${tx('Real size','真实规模')}</button><button type="button" data-view="toy" aria-pressed="false">${tx('12-token example','12 token 的例子')}</button></div>`+
   `<div class="seg-ctl fa-pathsw" id="fa-pathsw" role="group" aria-label="${esc(tx('Which path the example runs','例子跑哪条路径'))}"><button type="button" data-path="std" aria-pressed="true">${tx('Standard','标准')}</button><button type="button" data-path="fa" aria-pressed="false">FlashAttention</button></div>`+
   `<span class="status" id="fa-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="fa-legend"></div>`+
   `<div class="stage-graph graph fa-graph" id="fa-graph">`+
    `<div class="fa-toy" id="fa-toy" hidden>`+
     `<div class="fa-lane fa-hbm"><div class="fa-lh"><b>HBM</b> <span id="fa-hbm-sub"></span></div><div class="fa-mats" id="fa-hbm"></div></div>`+
     `<div class="fa-xfer" id="fa-xfer" aria-live="polite"></div>`+
     `<div class="fa-lane fa-chip"><div class="fa-lh"><b>SRAM</b> <span id="fa-chip-sub"></span></div><div class="fa-mats" id="fa-chipm"></div></div>`+
     `<div class="fa-row" id="fa-row"></div>`+
     `<div class="fa-cnt" id="fa-cnt"></div>`+
     `<div id="fa-stplay"></div>`+
    `</div>`+
    `<div class="fa-real" id="fa-real">`+
     `<div class="fa-cap" id="fa-real-cap"></div><div class="fa-bars" id="fa-bars"></div><p class="fa-room" id="fa-room"></p>`+
     `<div class="fa-curve" id="fa-curve" hidden></div>`+
     `<div class="fa-gauge" id="fa-gauge" hidden></div>`+
    `</div>`+
   `</div>`+
   `<p class="fa-scope small muted" id="fa-scope" tabindex="0"></p>`+
   `<div class="metrics fa-metrics" id="fa-metrics"></div>`+
   `<details class="math" id="fa-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="fa-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="fa-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="fa-lock" hidden></p>`+
   `<div class="lockable" id="fa-ctl-wrap">`+
   A.ui.seg({id:'fa-ctl-d',key:'d',label:tx('Head size d','head 维数 d'),options:[[64,tx(`64 (as ${MGPT})`,`64（和 ${MGPT} 一样）`)],[128,tx(`128 (as ${M8})`,`128（和 ${M8} 一样）`)]],value:S.d,note:tx('Numbers per query, key and value in one head. It sets the size of Q, K, V and their tiles, not the size of S.','一个 head 里每个 query、key、value 的维数。它决定 Q、K、V 和它们的 tile 有多大，不决定 S 有多大。')})+
   A.ui.range({id:'fa-ctl-N',key:'N',label:tx('Context length','上下文长度'),min:0,max:NS.length-1,value:S.ni,out:fTok(N_())+tx(' tokens',' token'),note:tx('Tokens in the prompt. S has one score for every pair of them.','prompt 里的 token 数。S 为它们的每一对存一个得分。')})+
   A.ui.seg({id:'fa-ctl-q',key:'dec',label:tx('Queries','query 行数'),options:[['0',tx('prefill: every token','prefill：每个 token')],['1',tx('decode: one new token','decode：一个新 token')]],value:S.dec?'1':'0',note:tx('Prefill computes a query for every token of the prompt; decode computes one, for the token being written.','prefill 给 prompt 的每个 token 算一个 query；decode 只为正在写的那个 token 算一个。')})+
   A.ui.seg({id:'fa-ctl-Br',key:'Br',label:tx('Q tile (Br rows)','Q tile（Br 行）'),options:TILES.map(b=>[b,String(b)]),value:S.Br,note:tx('Rows of Q kept on chip at once. Every Q tile reads all of K and V once, so a taller tile means fewer passes.','一次留在芯片上的 Q 行数。每个 Q tile 都把 K 和 V 读一遍，所以 tile 越高，遍数越少。')})+
   A.ui.seg({id:'fa-ctl-Bc',key:'Bc',label:tx('K/V tile (Bc rows)','K/V tile（Bc 行）'),options:TILES.map(b=>[b,String(b)]),value:S.Bc,note:tx('Rows of K and V streamed past at a time. It changes the SRAM a tile pair needs, not the traffic.','一次经过的 K、V 行数。它改变一对 tile 要的 SRAM，不改变流量。')})+
   `<div class="fa-grp"><div class="fa-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('2 bytes a number (BF16)','每个数值 2 字节（BF16）')}</span><div class="ctl-note">${tx('Q, K, V, S, P and O are counted in BF16, the usual format for inference. On chip, the score tile and the output being built are counted in FP32 (4 bytes), as kernels accumulate in FP32.','Q、K、V、S、P 和 O 都按 BF16 算，这是推理常用的格式。芯片上的得分 tile 和正在累加的输出按 FP32（4 字节）算，因为 kernel 用 FP32 累加。')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('227 KiB of SRAM for one tile pair','一对 tile 可用 227 KiB SRAM')}</span><div class="ctl-note">${tx('The most shared memory one thread block can use on an H100: 228 KiB per SM minus 1 KiB that CUDA reserves (NVIDIA’s Hopper tuning guide, which writes KB for 1,024 bytes). Real kernels also use registers, so this budget is a simplification.','H100 上一个 thread block 最多能用的共享内存：每个 SM 228 KiB，减去 CUDA 预留的 1 KiB（NVIDIA 的 Hopper 调优指南，文中的 KB 指 1,024 字节）。真实的 kernel 还会用寄存器，所以这个预算是简化。')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('3.35 TB/s of HBM bandwidth','HBM 带宽 3.35 TB/s')}</span><div class="ctl-note">${tx('The H100 SXM’s peak, used only to turn bytes into time. Real kernels reach a large share of it, not all.','H100 SXM 的峰值，只用来把字节数换算成时间。真实的 kernel 能达到它的大部分，达不到全部。')}</div></div></div>`+
   `<button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

def naive(Q, K, V):                     # S and P are N x N
    S = Q @ K.T / np.sqrt(Q.shape[1])
    P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V

def tiled(Q, K, V, Br=4, Bc=4):         # never builds S
    N, d = Q.shape
    O = np.zeros_like(Q)
    for i in range(0, N, Br):           # one Q tile on chip
        q = Q[i:i + Br]
        m = np.full(len(q), -np.inf)    # running max per row
        l = np.zeros(len(q))            # running sum per row
        acc = np.zeros((len(q), d))     # output, not divided yet
        for j in range(0, N, Bc):       # K and V tiles stream past
            s = q @ K[j:j + Bc].T / np.sqrt(d)
            m_new = np.maximum(m, s.max(1))
            f = np.exp(m - m_new)       # rescale when max grows
            p = np.exp(s - m_new[:, None])
            l = l * f + p.sum(1)
            acc = acc * f[:, None] + p @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = acc / l[:, None]
    return O

rng = np.random.default_rng(0)
Q, K, V = (rng.normal(size=(12, 4)) for _ in range(3))
print(np.abs(naive(Q, K, V) - tiled(Q, K, V)).max())
# about 1e-16: the same output, only the rounding differs`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="fa-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="fa-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Standard attention and the tiled version with an online softmax, in NumPy: the same output.','用 NumPy 写的标准 attention 和带 online softmax 的分块版本：输出相同。')}</p><pre class="code" id="fa-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="fa-prod">`+
   tx(`<li>Dao-AILab’s ${cite(SRC.fwd,'<code>csrc/flash_attn/src/flash_fwd_kernel.h</code>')}: the FlashAttention‑2 forward kernel. Each thread block takes one Q tile of one head, loops over the K and V tiles, and keeps the running max and sum per row in registers.</li>`+
      `<li>${cite(SRC.hop,'<code>hopper/flash_fwd_kernel_sm90.h</code>')}: FlashAttention‑3 for H100, which overlaps loading the next K and V tiles, the matrix multiplies and the softmax.</li>`+
      `<li>PyTorch’s ${cite(SRC.sdpa,'<code>torch.nn.functional.scaled_dot_product_attention</code>')} picks a fused FlashAttention kernel when the inputs allow it; vLLM’s ${cite(SRC.vllm,'<code>vllm/v1/attention/backends/flash_attn.py</code>')} runs prefill and decode through FlashAttention on supported GPUs.</li>`,
      `<li>Dao-AILab 的 ${cite(SRC.fwd,'<code>csrc/flash_attn/src/flash_fwd_kernel.h</code>')}：FlashAttention‑2 的前向 kernel。每个 thread block 负责一个 head 的一个 Q tile，循环处理 K 和 V 的各个 tile，把每一行的累计最大值和累计和放在寄存器里。</li>`+
      `<li>${cite(SRC.hop,'<code>hopper/flash_fwd_kernel_sm90.h</code>')}：给 H100 的 FlashAttention‑3，让预取下一个 K、V tile、矩阵乘法和 softmax 同时进行。</li>`+
      `<li>PyTorch 的 ${cite(SRC.sdpa,'<code>torch.nn.functional.scaled_dot_product_attention</code>')} 在输入允许时选用融合的 FlashAttention kernel；vLLM 的 ${cite(SRC.vllm,'<code>vllm/v1/attention/backends/flash_attn.py</code>')} 在支持的 GPU 上用 FlashAttention 跑 prefill 和 decode。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.fa1,'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness')}<span>${tx(`Dao, Fu, Ermon, Rudra and Ré, NeurIPS 2022: tiling plus the online softmax; standard attention needs ${mx('Θ(N d + N^2)')} HBM accesses, FlashAttention ${mx('Θ\\left(\\frac{N^2 d^2}{M}\\right)')} for SRAM size M`,`Dao、Fu、Ermon、Rudra、Ré，NeurIPS 2022：分块加 online softmax；标准 attention 要 ${mx('Θ(N d + N^2)')} 次 HBM 访问，FlashAttention 要 ${mx('Θ\\left(\\frac{N^2 d^2}{M}\\right)')}，M 是 SRAM 大小`)}</span></li>`+
   `<li>${cite(SRC.fa2,'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning')}<span>${tx('Dao, ICLR 2024: the outer loop over Q tiles, one thread block per Q tile, fewer non-matmul operations','Dao，ICLR 2024：外层循环改为遍历 Q tile，一个 Q tile 一个 thread block，减少矩阵乘法以外的运算')}</span></li>`+
   `<li>${cite(SRC.fa3,'FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision')}<span>${tx('Shah, Bikshandi, Zhang, Thakkar, Ramani and Dao, NeurIPS 2024: H100 kernels that overlap data movement, matmuls and the softmax; up to 740 TFLOP/s in FP16','Shah、Bikshandi、Zhang、Thakkar、Ramani、Dao，NeurIPS 2024：让数据搬运、矩阵乘法和 softmax 重叠的 H100 kernel；FP16 最高 740 TFLOP/s')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="fa-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="fa-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="fa-app m2" id="fa-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="fa-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much one head reads and writes in HBM.','选一个场景，看看一个 head 要在 HBM 里读写多少数据。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="fa-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Fit the tiles on an H100, at most 1.50 GiB per head at 32K tokens','在 H100 上放下 tile，32K token 时每个 head 不超过 1.50 GiB')}</h3>`+
   `<p>${tx(`${M1}, whose heads have 64 numbers, reads a 32K-token document on an H100. Choose the Q tile and the K/V tile to meet two goals. First, one tile pair fits in the 227 KiB of SRAM a thread block can use. Second, one head reads and writes at most 1.50 GiB of HBM, with every re-read of K and V counted as HBM traffic.`,
            `${M1} 在 H100 上读一份 32K token 的文档，它的每个 head 是 64 维。请选好 Q tile 和 K/V tile，同时满足两个目标。第一，一对 tile 放得进一个 thread block 能用的 227 KiB SRAM。第二，一个 head 读写 HBM 的量不超过 1.50 GiB，每次重读 K 和 V 都算作从 HBM 读。`)}</p>`+
   A.ui.seg({id:'fa-ic-chBr',key:'chBr',label:tx('Q tile (Br rows)','Q tile（Br 行）'),options:TILES.map(b=>[String(b),String(b)]),value:String(S.Br),inline:true})+
   A.ui.seg({id:'fa-ic-chBc',key:'chBc',label:tx('K/V tile (Bc rows)','K/V tile（Bc 行）'),options:TILES.map(b=>[String(b),String(b)]),value:String(S.Bc),inline:true})+
   `<div class="goal mono fa-goal" id="fa-ch-g1"><span>${tx('SRAM for one tile pair','一对 tile 的 SRAM')}</span><span><b id="fa-ch-m"></b>${tx(' (goal: at most 227 KiB)','（目标：不超过 227 KiB）')}</span></div>`+
   `<div class="goal mono fa-goal" id="fa-ch-g2"><span>${tx('HBM traffic, one head','一个 head 的 HBM 流量')}</span><span><b id="fa-ch-t"></b>${tx(' (goal: at most 1.50 GiB)','（目标：不超过 1.50 GiB）')}</span></div>`+
   submit+`<div id="fa-ch-res" class="fa-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},std:{en:'Step to the last step to continue.',zh:'走到最后一步才能继续。'},fa:{en:'Step to the last tile pair to continue.',zh:'走到最后一对 tile 才能继续。'},
    len:{en:'Drag the context to 128K to continue.',zh:'把上下文拖到 128K 才能继续。'},tile:{en:'Pick another tile size to continue.',zh:'选一个别的 tile 大小才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#fa-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="fa-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="fa-why" hidden></p>');}
  A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#fa-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]||s.id==='std'||s.id==='fa'||(s.id==='len'&&seenN.size>1)||(s.id==='tile'));
    $$('[data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvTried.has(v)||v===lv?lvRes(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.fa-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#fa-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(S.view==='toy'?`<p class="small">${tx('The 12-token example has fixed sizes: 12 tokens, head size 4, tiles of 4 rows. Switch the chart to Real size to see a scenario; changing a setting switches it back.','12 token 的例子大小固定：12 个 token，head 维数 4，tile 每个 4 行。把图切回“真实规模”才能看应用场景；改任何设置也会切回去。')}</p>`:(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:''));
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.Br,S.Bc), m=$('#fa-ch-m',rail), t=$('#fa-ch-t',rail); if(!m) return;
    m.textContent=fK(e.m.total); t.textContent=fB2(e.t.fa);
    const g1=$('#fa-ch-g1',rail), g2=$('#fa-ch-g2',rail);
    g1.classList.toggle('met',e.fits); g1.classList.toggle('fa-over',!e.fits); g2.classList.toggle('met',e.low); g2.classList.toggle('fa-over',!e.low);
    $$('.seg-ctl[data-key="chBr"]',rail).forEach(g=>A.ui.setPressed(g,String(S.Br))); $$('.seg-ctl[data-key="chBc"]',rail).forEach(g=>A.ui.setPressed(g,String(S.Bc)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#fa-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?STEPS[step].view:(mode==='challenge'?'real':S.view);
const pathNow=()=>mode==='guided'?(STEPS[step].path||S.path):S.path;
const faShown=()=>mode!=='guided'||step>=SIX('fa')&&(step>SIX('fa')||doneFa);   // FlashAttention's bar from part 4's end on (R3.8)
/* a cell's shade: grey levels only (R5.6) */
const shade=(v,lo,hi)=>{const x=Math.max(0,Math.min(1,(v-lo)/(hi-lo))); return `background:color-mix(in srgb, var(--ink-2) ${Math.round(12+70*x)}%, var(--surface))`;};
function gridHTML(rows,cols,valAt,opt){opt=opt||{};
  let h=`<div class="fa-grid${opt.cls?' '+opt.cls:''}" style="--c:${cols}">`;
  for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){const v=valAt(r,c); h+=`<i${v==null?' class="fa-e"':` style="${shade(v,opt.lo==null?-1.5:opt.lo,opt.hi==null?1.5:opt.hi)}"`}${opt.hl&&opt.hl(r,c)?' data-h="1"':''}></i>`;}
  return h+`</div>`;}
function matFig(name,sub,inner,cls,tip){return `<figure class="fa-m${cls?' '+cls:''}" tabindex="0" data-tipx="${esc(tip)}"><figcaption><b>${name}</b>${sub?` <span>${sub}</span>`:''}</figcaption>${inner}</figure>`;}
/* the toy at step k of a path: the HBM lane, the chip lane, the transfer line and the row table (HTML strings) */
function toyParts(path,k){
  const {Q,K,V}=MATS, B=TOY.B, n=path==='fa'?TIL.steps.length:STD_STEPS.length-1;
  let hbm='', chip='', xfer='', row='';
  if(path==='std'){
    const st=STD_STEPS[k], has=x=>({S:k>=2,P:k>=4,O:k>=6})[x], on=x=>st.mats&&st.mats.includes(x);
    const mt=(x,rows,cols,val,lo,hi,tip)=>matFig(x,'',has(x)===false?gridHTML(rows,cols,()=>null,{cls:'fa-empty'}):gridHTML(rows,cols,val,{lo,hi}),(on(x)?' fa-on fa-'+st.dir:'')+(has(x)===false?' fa-none':''),tip);
    hbm=mt('Q',NT,DT,(r,c)=>Q[r][c],null,null,tx('Q: one query of 4 numbers per token, 12 tokens.','Q：每个 token 一个 4 维的 query，共 12 个 token。'))+
        mt('K',NT,DT,(r,c)=>K[r][c],null,null,tx('K: one key of 4 numbers per token.','K：每个 token 一个 4 维的 key。'))+
        mt('V',NT,DT,(r,c)=>V[r][c],null,null,tx('V: one value of 4 numbers per token.','V：每个 token 一个 4 维的 value。'))+
        mt('O',NT,DT,(r,c)=>STD.O[r][c],-1,1,tx('O: the output, one row per token; written in the last step.','O：输出，每个 token 一行；最后一步才写出。'))+
        mt('S',NT,NT,(r,c)=>STD.S[r][c],-2,2,tx('S: 12 × 12 scores. Standard attention writes it here in step 2 and reads it back in step 3.','S：12 × 12 个得分。标准 attention 在第 2 步把它写到这里，第 3 步再读回去。'))+
        mt('P',NT,NT,(r,c)=>STD.P[r][c],0,0.5,tx('P: 12 × 12 attention weights. Written in step 4, read back in step 5.','P：12 × 12 个注意力权重。第 4 步写出，第 5 步读回。'));
    const opT={start:tx('nothing yet','还没开始'),readQK:tx('computes ','算 ')+mx('S = QK^{T} \\div \\sqrt{4}'),writeS:tx('sends S out','把 S 送出去'),readS:tx('softmax of each row of S','对 S 的每一行做 softmax'),writeP:tx('sends P out','把 P 送出去'),readPV:tx('computes ','算 ')+mx('O = P \\times V'),writeO:tx('sends O out','把 O 送出去')}[st.k];
    const big=st.k==='readQK'||st.k==='writeS'?STD.S:(st.k==='readS'||st.k==='writeP'||st.k==='readPV'?STD.P:null);
    chip=`<div class="fa-op">${opT}</div>`+(big?matFig(st.k==='readQK'||st.k==='writeS'?'S':'P',tx('on chip for now','暂时在芯片上'),gridHTML(NT,NT,(r,c)=>big[r][c],st.k==='readQK'||st.k==='writeS'?{lo:-2,hi:2}:{lo:0,hi:0.5}),'',tx('At real size this matrix does not fit on the chip: 8,192 × 8,192 scores are 128 MiB, the SRAM of one thread block is 227 KiB. That is why standard attention sends it to HBM.','真实规模时这个矩阵放不进芯片：8,192 × 8,192 个得分是 128 MiB，一个 thread block 的 SRAM 只有 227 KiB。所以标准 attention 要把它送到 HBM。')):'');
    const what={start:tx('Q, K and V wait in HBM.','Q、K、V 在 HBM 里等着。'),readQK:tx(`▼ Read Q and K: ${fint(st.n)} numbers.`,`▼ 读入 Q 和 K：${fint(st.n)} 个数值。`),writeS:tx(`▲ Write S: ${fint(st.n)} numbers.`,`▲ 写出 S：${fint(st.n)} 个数值。`),readS:tx(`▼ Read S back: ${fint(st.n)} numbers.`,`▼ 读回 S：${fint(st.n)} 个数值。`),
      writeP:tx(`▲ Write P: ${fint(st.n)} numbers.`,`▲ 写出 P：${fint(st.n)} 个数值。`),readPV:tx(`▼ Read P back, and V: ${fint(st.n)} numbers.`,`▼ 读回 P，再读 V：${fint(st.n)} 个数值。`),writeO:tx(`▲ Write O: ${fint(st.n)} numbers.`,`▲ 写出 O：${fint(st.n)} 个数值。`)}[st.k];
    xfer=`<span class="fa-xs">${tx(`Step ${k} of ${n}`,`第 ${k} 步（共 ${n} 步）`)}</span> ${esc(what)}`;
  } else {
    const cur=k>0?TIL.steps[k-1]:null, done=TIL.steps.slice(0,k);
    const oRows=new Set(); done.forEach(s=>{if(s.writeO) for(let r=0;r<B;r++) oRows.add(s.i*B+r);});
    const inQ=r=>cur&&Math.floor(r/B)===cur.i, inKV=r=>cur&&Math.floor(r/B)===cur.j;
    hbm=matFig('Q','',gridHTML(NT,DT,(r,c)=>Q[r][c],{hl:r=>inQ(r)&&cur.loadQ}),cur&&cur.loadQ?' fa-on fa-in':'',tx('Q in HBM. A Q tile (4 rows) is read once, when its turn comes.','HBM 里的 Q。每个 Q tile（4 行）轮到时只读一次。'))+
        matFig('K','',gridHTML(NT,DT,(r,c)=>K[r][c],{hl:r=>inKV(r)}),cur?' fa-on fa-in':'',tx('K in HBM. Every Q tile reads all of K again, one tile of 4 rows at a time.','HBM 里的 K。每个 Q tile 都要把 K 整个再读一遍，一次一个 4 行的 tile。'))+
        matFig('V','',gridHTML(NT,DT,(r,c)=>V[r][c],{hl:r=>inKV(r)}),cur?' fa-on fa-in':'',tx('V in HBM, read together with K.','HBM 里的 V，和 K 一起读。'))+
        matFig('O','',gridHTML(NT,DT,(r,c)=>oRows.has(r)?TIL.O[r][c]:null,{lo:-1,hi:1,hl:r=>cur&&cur.writeO&&inQ(r)}),cur&&cur.writeO?' fa-on fa-out':'',tx('O in HBM: a Q tile’s 4 output rows are written once, after its last K/V tile.','HBM 里的 O：一个 Q tile 的 4 行输出，在它的最后一个 K/V tile 之后写一次。'))+
        `<div class="fa-never" tabindex="0" data-tipx="${esc(tx('FlashAttention never writes S or P to HBM: each 4 × 4 tile of scores is used on the chip and dropped.','FlashAttention 从不把 S 和 P 写到 HBM：每个 4 × 4 的得分 tile 在芯片上用完就丢。'))}"><b>S, P</b><span>${tx('never written','从不写出')}</span></div>`;
    const map=`<figure class="fa-m fa-map" tabindex="0" data-tipx="${esc(tx('The 12 × 12 grid of scores, tile by tile: the outlined tile is on the chip now; the shaded ones were computed, used and dropped. None of them was stored.','12 × 12 的得分网格，一个 tile 一个 tile 地算：带粗边框的 tile 正在芯片上；带阴影的已经算过、用完、丢掉。没有一个被存下来。'))}"><figcaption><b>${tx('Score tiles','得分 tile')}</b> <span>${tx('used, then dropped','用完就丢')}</span></figcaption><div class="fa-grid fa-mapg" style="--c:3">`+
      [0,1,2].map(i=>[0,1,2].map(j=>{const idx=i*3+j+1; return `<i class="${idx===k?'fa-now':(idx<k?'fa-past':'')}"></i>`;}).join('')).join('')+`</div></figure>`;
    if(cur){const s=cur.s, qs=Q.slice(cur.i*B,cur.i*B+B), ks=K.slice(cur.j*B,cur.j*B+B), vs=V.slice(cur.j*B,cur.j*B+B);
      chip=matFig(`Q tile ${cur.i+1}`,'',gridHTML(B,DT,(r,c)=>qs[r][c]),'',tx('The Q tile stays on the chip while every K and V tile passes.','这个 Q tile 留在芯片上，等每一个 K、V tile 经过。'))+
        matFig(tx(`K, V tile ${cur.j+1}`,`K、V tile ${cur.j+1}`),'',`<div class="fa-two">${gridHTML(B,DT,(r,c)=>ks[r][c])}${gridHTML(B,DT,(r,c)=>vs[r][c])}</div>`,'',tx('The K and V tile that just arrived from HBM.','刚从 HBM 读进来的 K、V tile。'))+
        matFig(tx('Scores','得分'),'4 × 4',gridHTML(B,B,(r,c)=>s[r][c],{lo:-2,hi:2}),'',tx('This tile’s 16 scores, ','这个 tile 的 16 个得分 ')+mx('q \\cdot k \\div \\sqrt{4}')+tx(': used now, then dropped.','：现在用，用完就丢。'))+
        matFig(tx('Output so far','输出半成品'),'',gridHTML(B,DT,(r,c)=>cur.acc[r][c]/cur.l[r],{lo:-1,hi:1}),'',tx('Each row’s output so far, divided by its running sum: it becomes the final output after the last K/V tile.','每一行到目前为止的输出，除以它的累计和：最后一个 K/V tile 之后就是最终输出。'))+map;}
    else chip=`<div class="fa-op">${esc(tx('nothing yet','还没开始'))}</div>`+map;
    const what=!cur?tx('Q, K and V wait in HBM.','Q、K、V 在 HBM 里等着。')
      :tx(`${cur.loadQ?`▼ Read Q tile ${cur.i+1} (${B*DT} numbers) and`:'▼ Read'} K, V tile ${cur.j+1} (${2*B*DT} numbers); 16 scores used on the chip.${cur.writeO?` ▲ Write O rows ${cur.i*B+1}–${cur.i*B+B} (${B*DT} numbers).`:''}`,
          `${cur.loadQ?`▼ 读入 Q tile ${cur.i+1}（${B*DT} 个数值）和`:'▼ 读入'} K、V tile ${cur.j+1}（${2*B*DT} 个数值）；16 个得分在芯片上用掉。${cur.writeO?`▲ 写出 O 的第 ${cur.i*B+1}–${cur.i*B+B} 行（${B*DT} 个数值）。`:''}`);
    const pair=cur?tx(` (Q tile ${cur.i+1} × K/V tile ${cur.j+1})`,`（Q tile ${cur.i+1} × K/V tile ${cur.j+1}）`):'';
    xfer=`<span class="fa-xs">${tx(`Tile pair ${k} of ${n}`,`第 ${k} 对 tile（共 ${n} 对）`)}${pair}</span> ${esc(what)}`;
    /* one row's running max and sum: the first row of the Q tile on chip; after the last tile pair, row 1, which the result names */
    const ii=cur&&k<n?cur.i:0, hist=TIL.steps.filter(s=>s.i===ii&&TIL.steps.indexOf(s)<k), r0=ii*B;
    row=`<div class="fa-rh">${tx(`Row ${r0+1}: running max m and running sum ℓ, one line per K/V tile`,`第 ${r0+1} 行：累计最大值 m 和累计和 ℓ，每个 K/V tile 一行`)}</div>`+
      `<table class="fa-rt"><thead><tr><th>${tx('K/V tile','K/V tile')}</th><th>${tx('tile max','tile 内最大')}</th><th>m</th><th>${tx('rescale','缩放')}</th><th>ℓ</th></tr></thead><tbody>`+
      [0,1,2].map(j=>{const s=hist.find(x=>x.j===j); if(!s) return `<tr class="fa-e"><td>${j+1}</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>`;
        const rose=j>0&&s.f[0]<1, fac=j===0?'—':(rose?`× ${f2(s.f[0])}`:tx('none','不缩放'));
        return `<tr${rose?' class="fa-up"':''}><td>${j+1}</td><td>${f2(s.tmax[0])}</td><td>${f2(s.m[0])}${rose?' ▲':''}</td><td>${fac}</td><td>${f3(s.l[0])}</td></tr>`;}).join('')+`</tbody></table>`;
  }
  return {hbm,chip,xfer,row};
}
/* R7.9: the chip lane and the transfer line keep the height of their tallest step, so ▸ does not move under the pointer */
const toyMin={};
function reserveToy(path){
  const ch=$('#fa-chipm',app), xf=$('#fa-xfer',app), key=`${path}|${innerWidth}|${A.lang()}`;
  if(!toyMin[key]){const n=path==='fa'?TIL.steps.length:STD_STEPS.length-1, kc=ch.innerHTML, kx=xf.innerHTML; let hc=0, hx=0;
    ch.style.minHeight=''; xf.style.minHeight='';
    for(let k=0;k<=n;k++){const t=toyParts(path,k); ch.innerHTML=t.chip; xf.innerHTML=t.xfer; hc=Math.max(hc,ch.offsetHeight); hx=Math.max(hx,xf.offsetHeight);}
    ch.innerHTML=kc; xf.innerHTML=kx; if(hc>0&&hx>0) toyMin[key]={hc,hx}; else return;}
  ch.style.minHeight=toyMin[key].hc+'px'; xf.style.minHeight=toyMin[key].hx+'px';
}
function drawToy(){
  const path=pathNow(), n=stepsN(); tstep=Math.max(0,Math.min(n,tstep));
  $('#fa-hbm-sub',app).textContent=tx('off the chip: large and slow','芯片外：大而慢');
  $('#fa-chip-sub',app).textContent=tx('on the chip: small and fast','芯片上：小而快');
  const {hbm,chip,xfer,row}=toyParts(path,tstep);
  $('#fa-hbm',app).innerHTML=hbm; $('#fa-chipm',app).innerHTML=chip; $('#fa-xfer',app).innerHTML=xfer;
  reserveToy(path);
  const rw=$('#fa-row',app); rw.innerHTML=row; rw.hidden=!row;
  const sn=path==='std'?stdSoFar(tstep):faSoFar(tstep), spn=path==='std'?STD_STEPS.slice(0,tstep+1).filter(s=>['writeS','readS','writeP','readPV'].includes(s.k)).reduce((a,s)=>a+(s.k==='readPV'?NT*NT:s.n),0):0;
  $('#fa-cnt',app).innerHTML=`<span class="fa-cl">${tx('Numbers moved between HBM and the chip','HBM 和芯片之间搬运的数值')}</span><span class="fa-cbar"><i style="width:${(sn/STD_TOTAL*100).toFixed(2)}%"></i>${path==='std'?`<i class="fa-sp" style="width:${(spn/STD_TOTAL*100).toFixed(2)}%"></i>`:''}</span><b class="mono">${fint(sn)}</b>`+
    (path==='std'?`<div class="fa-ckey">${legIt('fa-sw-sp',tx('hatched: S and P written and read back','斜纹：S 和 P 写出又读回'),tx('In the counter, the hatched part is the numbers of S and P written and read back.','计数条里斜纹的部分，是 S 和 P 写出又读回的数值。'))}</div>`:'');
  const ph=$('#fa-stplay',app); if(!ph.firstChild||ph._n!==n){ph.innerHTML=playerHTML('fa-st-play')+'<p class="fa-done" hidden></p>'; ph._n=n; A.ui.initRanges(ph);}
  /* phones: after the last step the stage's own ▸ does not scroll the page away; a link takes the learner to the result (R3.2, R1.11) */
  const dn=$('.fa-done',ph), showDone=mode==='guided'&&tstep>=n&&innerWidth<=760&&!!STEPS[step].obs;
  if(dn){dn.hidden=!showDone; if(showDone) dn.innerHTML=`<button type="button" class="linkbtn" data-act="toresult">${tx(`Done: the result is in part ${step+1} ↓`,`走完了：结果在第 ${step+1} 部分 ↓`)}</button>`;}
  syncPlayer();
}
function syncPlayer(){
  const n=stepsN(), path=pathNow();
  $$('input[data-key="tstep"]',app).forEach(i=>{i.max=n; if(+i.value!==tstep) i.value=tstep; A.ui.setRange(i); i.setAttribute('aria-valuetext',path==='std'?tx(`step ${tstep} of ${n}`,`第 ${tstep} 步，共 ${n} 步`):tx(`tile pair ${tstep} of ${n}`,`第 ${tstep} 对 tile，共 ${n} 对`));});
  $$('[data-fa-out="stepno"]',app).forEach(e=>{e.textContent=path==='std'?tx(`Step ${tstep} / ${n}`,`第 ${tstep} / ${n} 步`):tx(`Tile pair ${tstep} / ${n}`,`第 ${tstep} / ${n} 对`);});
  $$('[data-act="play"]',app).forEach(b=>{b.textContent=playing?tx('❚❚ Pause','❚❚ 暂停'):(tstep>=n?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放')); b.setAttribute('aria-pressed',String(playing));
    b.dataset.tipx=playing?tx('Pause the run.','暂停播放。'):(tstep>=n?tx('The run is over: play it again from the start.','已经放完：从头再放一遍。'):tx('Play the run from the step shown.','从图里显示的这一步接着放。'));});
  $$('[data-act="back"]',app).forEach(b=>{b.disabled=tstep<=0;}); $$('[data-act="fwd"]',app).forEach(b=>{b.disabled=tstep>=n;});
  $$('.seg-ctl[data-key="speed"]',app).forEach(g=>A.ui.setPressed(g,speed));
}
/* the real-size view: the two bars, the room S would need, the curve over context length (part 5), the on-chip gauge */
function barRow(k,label,parts,total,max,u,tip,ghost){
  const segs=parts.map(p=>`<i class="fa-seg ${p.c}" style="width:${(p.v/max*100).toFixed(3)}%"></i>`).join('');
  return `<div class="fa-brow${ghost?' fa-ghost':''}" data-k="${k}" tabindex="0" data-tipx="${esc(tip)}"><span class="fa-bl">${label}</span><span class="fa-btrk">${segs}</span><span class="fa-bv mono">${ghost?'':fT(total,u)}</span></div>`;}
function realParams(){
  if(mode==='challenge') return {N:CH.N,d:CH.d,Br:S.Br,Bc:S.Bc,dec:false,cap:tx(`${M1}, 32K tokens, one head`,`${M1}，32K token，一个 head`)};
  if(mode==='guided'){const id=STEPS[step].id; const Br=id==='tile'?lv:128;
    return {N:id==='len'?N_():8192,d:128,Br,Bc:Br,dec:false,cap:tx(`${M8}, ${fTok(id==='len'?N_():8192)} tokens, one head`,`${M8}，${fTok(id==='len'?N_():8192)} token，一个 head`)};}
  return {N:N_(),d:S.d,Br:S.Br,Bc:S.Bc,dec:S.dec,cap:tx(`one head of size ${S.d}, ${fTok(N_())} tokens, ${S.dec?'one decode step':'prefill'}`,`一个 head（维数 ${S.d}），${fTok(N_())} token，${S.dec?'一个 decode 步':'prefill'}`)};
}
function drawReal(){
  const p=realParams(), t=traffic(p.N,p.d,p.Br,p.dec?1:null), u=unitOf(t.std), max=Math.max(t.std,t.fa)*1.02, show=faShown();
  $('#fa-real-cap',app).textContent=tx(`HBM traffic for ${p.cap}, BF16`,`${p.cap}的 HBM 流量（BF16）`);
  const stdTip=tx(`Standard attention: Q, K and V read and O written once (${fPart(t.stdBase,u)}), plus S and P each written and read once (${fPart(t.sp,u)}).`,`标准 attention：Q、K、V 各读一次，O 写一次（${fPart(t.stdBase,u)}），再加 S 和 P 各写一次、读一次（${fPart(t.sp,u)}）。`);
  const faTip=t.passes===1?tx(`FlashAttention: Q, K and V read and O written once (${fPart(t.floor,u)}); with one Q tile, K and V are not read again.`,`FlashAttention：Q、K、V 各读一次，O 写一次（${fPart(t.floor,u)}）；只有一个 Q tile，K 和 V 不用重读。`)
    :tx(`FlashAttention: Q, K and V read and O written once (${fPart(t.floor,u)}), the least any kernel moves, plus K and V read again for the other ${fint(t.passes-1)} Q tiles (${fPart(t.kv-t.kv1,u)}). Here those re-reads count as HBM traffic. An H100’s 50 MB L2 cache probably serves many of them, so the real traffic lies between ${fPart(t.floor,u)} and the full ${fPart(t.fa,u)}.`,`FlashAttention：Q、K、V 各读一次，O 写一次（${fPart(t.floor,u)}），这是任何 kernel 最少要搬的；再加其余 ${fint(t.passes-1)} 个 Q tile 重读 K 和 V（${fPart(t.kv-t.kv1,u)}）。这里把这些重读都算作 HBM 流量。H100 的 50 MB L2 cache 很可能送出其中不少，所以真实流量在 ${fPart(t.floor,u)} 和整条的 ${fPart(t.fa,u)} 之间。`);
  $('#fa-bars',app).innerHTML=barRow('std',tx('Standard','标准'),[{c:'fa-base',v:t.stdBase},{c:'fa-sp',v:t.sp}],t.std,max,u,stdTip)+
    (show?barRow('fa','FlashAttention',[{c:'fa-base',v:t.floor},{c:'fa-kv',v:t.kv-t.kv1}],t.fa,max,u,faTip):barRow('fa','FlashAttention',[],0,max,u,tx('FlashAttention’s traffic appears once you have stepped through it in part 4.','FlashAttention 的流量在第 4 部分走完之后显示。'),true).replace('</span></div>',`${tx('after part 4','第 4 部分之后')}</span></div>`))+barKeyHTML(show&&t.passes>1);
  const room=$('#fa-room',app);
  room.innerHTML=tx(`To hold S, standard attention needs <b>${fS(t.s)}</b> of HBM for this head${p.dec||p.d!==128?'':` and ${fLay(t.s*32)} for a layer of 32 heads`}; FlashAttention stores no ${mx('N \\times N')} matrix.`,`为了放 S，标准 attention 这个 head 要占 <b>${fS(t.s)}</b> HBM${p.dec||p.d!==128?'':`，一层 32 个 head 共 ${fLay(t.s*32)}`}；FlashAttention 不存 ${mx('N \\times N')} 的矩阵。`);
  room.hidden=mode==='challenge';
  const cv=$('#fa-curve',app), showCurve=isP('len'); cv.hidden=!showCurve; if(showCurve) cv.innerHTML=curveSVG(p,cv.clientWidth);
  const gg=$('#fa-gauge',app), showG=mode==='challenge'||isP('tile')||mode==='sandbox'; gg.hidden=!showG; if(showG) gg.innerHTML=gaugeHTML(sram(p.Br,p.Bc,p.d),p);
}
/* part 5: traffic against context length, both axes in powers of 2 (each x tick doubles N, labels on every other tick; each step up is 4 times the bytes) */
/* the width of a curve's name, measured in the page's own font, so the right margin always holds it (R5.9) */
const labW=t=>{try{const c=labW.c||(labW.c=document.createElement('canvas').getContext('2d')); c.font=`600 11.5px ${getComputedStyle(document.body).fontFamily}`; return Math.ceil(c.measureText(t).width);}catch(e){return 7*t.length;}};
function curveSVG(p,cw){
  const W=Math.round(Math.max(300,Math.min(640,cw||520))), H=W<420?190:180, l=50, r=9+Math.max(labW(tx('Standard','标准')),labW('FlashAttention'))+6, tp=12, b=26, xs=i=>l+(W-l-r)*i/(NS.length-1);
  const pts=NS.map(N=>traffic(N,128,128)), lo=Math.log2(pts[0].fa), hi=Math.log2(pts[pts.length-1].std), ys=v=>tp+(H-tp-b)*(1-(Math.log2(v)-lo)/(hi-lo));
  const line=(k,cls)=>`<polyline class="${cls}" points="${pts.map((x,i)=>`${xs(i).toFixed(1)},${ys(x[k]).toFixed(1)}`).join(' ')}"/>`;
  const ticks=[4*MiB,64*MiB,GiB,16*GiB].filter(v=>Math.log2(v)>=lo&&Math.log2(v)<=hi).map(v=>`<g><line x1="${l}" x2="${W-r}" y1="${ys(v).toFixed(1)}" y2="${ys(v).toFixed(1)}" class="fa-cv-grid"/><text x="${l-4}" y="${(ys(v)+3.5).toFixed(1)}" text-anchor="end">${v>=GiB?fixd(v/GiB,0)+' GiB':fixd(v/MiB,0)+' MiB'}</text></g>`).join('');
  const xt=NS.map((N,i)=>`<line class="fa-cv-tick" x1="${xs(i).toFixed(1)}" x2="${xs(i).toFixed(1)}" y1="${H-b}" y2="${H-b+5}"/>`+(i%2===1?`<text x="${xs(i).toFixed(1)}" y="${H-8}" text-anchor="middle">${fTok(N)}</text>`:'')).join('');
  const i=S.ni, cur=pts[i];
  return `<div class="fa-cvcap">${tx('One head’s HBM traffic against context length: each tick doubles the context','一个 head 的 HBM 流量随上下文长度的变化：往右一个刻度，上下文翻一倍')}</div>`+
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx(`Traffic rises 4 times per doubling for both paths; at ${fTok(NS[i])}: standard ${fB(cur.std)}, FlashAttention ${fB(cur.fa)}.`,`两条路径都是上下文每翻一倍、流量变成 4 倍；${fTok(NS[i])} 时：标准 ${fB(cur.std)}，FlashAttention ${fB(cur.fa)}。`))}">${ticks}${xt}`+
    line('std','fa-cv-std')+line('fa','fa-cv-fa')+`<line class="fa-cv-now" x1="${xs(i).toFixed(1)}" x2="${xs(i).toFixed(1)}" y1="${tp}" y2="${H-b}"/>`+
    `<circle class="fa-cv-std" cx="${xs(i).toFixed(1)}" cy="${ys(cur.std).toFixed(1)}" r="4"/><circle class="fa-cv-fa" cx="${xs(i).toFixed(1)}" cy="${ys(cur.fa).toFixed(1)}" r="4"/>`+
    `<text class="fa-cv-lab fa-cv-lstd" x="${(W-r+9).toFixed(1)}" y="${(ys(pts[pts.length-1].std)+1).toFixed(1)}">${esc(tx('Standard','标准'))}</text>`+
    `<text class="fa-cv-lab fa-cv-lfa" x="${(W-r+9).toFixed(1)}" y="${(ys(pts[pts.length-1].fa)+9).toFixed(1)}">FlashAttention</text></svg>`;
}
function gaugeHTML(m,p){
  const max=Math.max(m.total,HW.smem)*1.04, segs=[['q',m.q,tx('Q tile','Q tile')],['k',m.k+m.v,tx('K and V tiles','K、V tile')],['s',m.s,tx('score tile (FP32)','得分 tile（FP32）')],['o',m.o+m.ml,tx('output, m, ℓ (FP32)','输出、m、ℓ（FP32）')]];
  const tip=tx(`One tile pair with Br = ${p.Br} and Bc = ${p.Bc} rows, head size ${p.d}: Q tile ${fK(m.q)}, K and V tiles ${fK(m.k+m.v)}, score tile ${fK(m.s)}, output with m and ℓ ${fK(m.o+m.ml)}; ${fK(m.total)} in all, against the 227 KiB one thread block can use. This lab’s budget puts everything in shared memory; real kernels keep the score tile and the output in registers, so they fit larger tiles.`,
               `一对 tile，Br = ${p.Br} 行、Bc = ${p.Bc} 行，head 维数 ${p.d}：Q tile ${fK(m.q)}，K、V tile ${fK(m.k+m.v)}，得分 tile ${fK(m.s)}，输出加 m、ℓ ${fK(m.o+m.ml)}；共 ${fK(m.total)}，上限是一个 thread block 能用的 227 KiB。这个 Lab 的预算把所有东西都放在共享内存里；真实的 kernel 把得分 tile 和输出放在寄存器里，所以放得下更大的 tile。`);
  return `<div class="fa-gcap">${tx(`SRAM for one tile pair (${p.Br} × ${p.Bc}): ${fK(m.total)} of 227 KiB`,`一对 tile（${p.Br} × ${p.Bc}）的 SRAM：${fK(m.total)}，上限 227 KiB`)} <b class="${m.fits?'fa-fit':'fa-nofit'}">${m.fits?tx('fits','放得下'):tx('does not fit','放不下')}</b></div>`+
    `<div class="fa-gtrk${m.fits?'':' fa-over'}" tabindex="0" data-tipx="${esc(tip)}">${segs.map(([c,v])=>`<i class="fa-g-${c}" style="width:${(v/max*100).toFixed(3)}%"></i>`).join('')}<b class="fa-glim" style="left:${(HW.smem/max*100).toFixed(3)}%"><em>227 KiB</em></b></div>`+
    `<div class="fa-gkey">${segs.map(([c,,lb])=>`<span><i class="fa-g-${c}"></i>${esc(lb)}</span>`).join('')}<span tabindex="0" data-tipx="${esc(tx('The dashed mark is the 227 KiB one thread block can use; a red outline means the tiles do not fit.','虚线标记是一个 thread block 能用的 227 KiB；红色边框表示 tile 放不下。'))}"><i class="fa-g-lim"></i>${esc(tx('dashed mark: 227 KiB limit','虚线标记：227 KiB 上限'))}</span></div>`;
}
function statusOf(){
  const v=viewNow();
  if(mode==='challenge'){const e=chEval(S.Br,S.Bc); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Both targets met','两个目标都达到'):(!e.fits?tx('Does not fit','放不下'):tx('Too much traffic','流量太多')),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when one tile pair fits in 227 KiB and one head moves at most 1.50 GiB. Now: ${fK(e.m.total)} and ${fB2(e.t.fa)}.`,`一对 tile 放得进 227 KiB、一个 head 不超过 1.50 GiB 时，徽章就是绿色。现在是 ${fK(e.m.total)} 和 ${fB2(e.t.fa)}。`)};}
  if(v==='toy'){const path=pathNow(), n=stepsN(), end=tstep>=n;
    if(path==='std') return {cls:'neutral',txt:end?tx('S and P went out and back','S 和 P 写出又读回'):tx(`Step ${tstep} of ${n}`,`第 ${tstep} 步，共 ${n} 步`),short:tx(`Step ${tstep} / ${n}`,`第 ${tstep} / ${n} 步`),tip:tx('Where the 12-token example is in standard attention’s six steps.','12 token 的例子走到标准 attention 六步里的哪一步。')};
    return {cls:end?'ok':'neutral',txt:end?tx('Same output','输出相同'):tx(`Tile pair ${tstep} of ${n}`,`第 ${tstep} 对 tile，共 ${n} 对`),short:end?tx('Same output','输出相同'):tx(`Pair ${tstep} / ${n}`,`第 ${tstep} / ${n} 对`),
      tip:end?tx(`After the last tile pair, FlashAttention’s output equals standard attention’s; the largest difference is ${sciT(MAXDIFF)}.`,`最后一对 tile 之后，FlashAttention 的输出等于标准 attention 的；最大的差是 ${sciT(MAXDIFF)}。`):tx('Where the 12-token example is among FlashAttention’s nine tile pairs.','12 token 的例子走到 FlashAttention 九对 tile 里的哪一对。')};}
  const p=realParams(), t=traffic(p.N,p.d,p.Br,p.dec?1:null);
  if(!faShown()) return {cls:'neutral',txt:tx(`S: ${fS(t.s)} per head`,`S：每个 head ${fS(t.s)}`),short:tx(`S ${fS(t.s)}`,`S ${fS(t.s)}`),tip:tx(`One head’s S at ${fTok(p.N)} tokens: ${fTok(p.N)} × ${fTok(p.N)} scores × 2 bytes.`,`${fTok(p.N)} token 时一个 head 的 S：${fTok(p.N)} × ${fTok(p.N)} 个得分 × 2 字节。`)};
  const m=sram(p.Br,p.Bc,p.d);
  if(!m.fits) return {cls:'bad',txt:tx('Tiles do not fit','tile 放不下'),short:tx('Does not fit','放不下'),tip:tx(`One tile pair needs ${fK(m.total)}, more than 227 KiB, so FlashAttention cannot run with these tiles.`,`一对 tile 要 ${fK(m.total)}，超过 227 KiB，FlashAttention 用这个大小跑不起来。`)};
  const rt=t.std/t.fa;
  if(rt<1.05&&rt>0.95) return {cls:'neutral',txt:tx('About the same traffic','流量差不多'),short:tx('About the same','差不多'),tip:tx(`Standard attention moves ${fT(t.std)} and FlashAttention ${fT(t.fa)} for one head: both read all of K and V, and S is one row.`,`一个 head 用标准 attention 搬 ${fT(t.std)}，用 FlashAttention 搬 ${fT(t.fa)}：两者都要把 K 和 V 读一遍，而 S 只有一行。`)};
  const cmp=rt>=1?tx(`standard moves ${fixd(rt,1)} times as much`,`标准是 FlashAttention 的 ${fixd(rt,1)} 倍`):tx(`FlashAttention moves ${fixd(1/rt,1)} times as much, because each of its ${fint(t.passes)} Q tiles reads K and V again`,`FlashAttention 是标准的 ${fixd(1/rt,1)} 倍，因为它的 ${fint(t.passes)} 个 Q tile 各要重读一遍 K 和 V`);
  return {cls:'neutral',txt:tx('S and P stay on chip','S 和 P 留在芯片上'),short:tx('S, P on chip','S、P 在芯片上'),
    tip:tx(`FlashAttention never writes S or P to HBM. One head moves ${fB(t.std)} with standard attention and ${fB(t.fa)} with FlashAttention: ${cmp}. These numbers count every re-read of K and V as HBM traffic. An H100’s 50 MB L2 cache probably serves many of these re-reads, so FlashAttention’s real traffic lies between ${fB(t.floor)} (Q, K and V read once, O written once) and this count.`,
           `FlashAttention 从不把 S 和 P 写到 HBM。一个 head 用标准 attention 搬 ${fB(t.std)}，用 FlashAttention 搬 ${fB(t.fa)}：${cmp}。这里把每次重读 K 和 V 都算作 HBM 流量。H100 的 50 MB L2 cache 很可能送出其中不少重读，所以 FlashAttention 的真实流量在 ${fB(t.floor)}（Q、K、V 各读一次，O 写一次）和这里算出的值之间。`)};
}
const legIt=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
/* R5.9: legends sit beside what they label. The top row keys the toy's matrices, drawn right under it; the bar segments' key sits right
   under the two bars, the counter's under the counter, the 227 KiB mark's under the gauge; the curves are named at their right ends */
function legendHTML(){
  if(viewNow()!=='toy') return '';
  return legIt('fa-sw-cell',tx('cell: one number, stronger shade = larger','格子：一个数值，颜色越重越大'),tx('Each small square is one number of the matrix; the stronger its shade, the larger the number.','每个小方格是矩阵里的一个数值；颜色越重，数值越大。'))+
    legIt('fa-sw-on',tx('thick outline: moved in this step','粗边框：这一步搬运的'),tx('A thick outline marks what crossed between HBM and the chip in the step shown; ▼ is a read into the chip, ▲ a write to HBM.','粗边框标出当前这一步在 HBM 和芯片之间搬运的东西；▼ 是读进芯片，▲ 是写回 HBM。'))+
    legIt('fa-sw-none',tx('dashed: not written yet','虚线：还没写出'),tx('A dashed matrix has not been written to HBM yet.','虚线的矩阵还没写到 HBM。'));
}
/* the key of the two bars' segments, right under the bars */
function barKeyHTML(rereads){
  return `<div class="fa-bkey">`+legIt('fa-sw-base',tx('Q, K, V read once, O written once','Q、K、V 各读一次，O 写一次'),tx('The solid part of each bar: reading the inputs once and writing the output once, the least any kernel moves.','每条的实心部分：输入各读一次、输出写一次，这是任何 kernel 最少要搬的。'))+
    legIt('fa-sw-sp',tx('S and P written and read','S 和 P 写出又读回'),tx('Hatched: standard attention writes S and P to HBM and reads them back, 8 bytes per score in BF16.','斜纹：标准 attention 把 S 和 P 写到 HBM 再读回来，BF16 下每个得分 8 字节。'))+
    (rereads?legIt('fa-sw-kv',tx('K and V read again per Q tile','每个 Q tile 重读 K 和 V'),tx('Dotted: FlashAttention reads K and V again for every further Q tile. These re-reads count as HBM traffic here. An H100’s 50 MB L2 cache probably serves many of them, so FlashAttention’s real traffic lies between the end of the solid part and the end of the bar.','点纹：FlashAttention 每多一个 Q tile 就再读一遍 K 和 V。这里把这些重读都算作 HBM 流量。H100 的 50 MB L2 cache 很可能送出其中不少，所以 FlashAttention 的真实流量在实心部分的末端和整条的末端之间。')):'')+`</div>`;
}
function metricsDef(){
  const v=viewNow();
  if(v==='toy'){const path=pathNow(), endF=tstep>=TIL.steps.length;
    return [{k:'ms',label:tx('Moved, standard','标准搬运'),v:path==='std'?fint(stdSoFar(tstep)):fint(STD_TOTAL),sub:path==='std'?tx(`so far, of ${fint(STD_TOTAL)}`,`到目前，共 ${fint(STD_TOTAL)}`):tx('the whole run','整次运行'),tip:tx(`Numbers that standard attention moves between HBM and the chip for the 12-token example: ${fint(STD_TOTAL)} in all, ${fint(SP_TOY)} of them S and P.`,`12 token 的例子里，标准 attention 在 HBM 和芯片之间搬运的数值：共 ${fint(STD_TOTAL)} 个，其中 ${fint(SP_TOY)} 个是 S 和 P。`)},
      {k:'mf',label:tx('Moved, FlashAttention','FlashAttention 搬运'),v:path==='fa'?fint(faSoFar(tstep)):(faShown()?fint(FA_TOTAL):'—'),sub:path==='fa'?tx(`so far, of ${fint(FA_TOTAL)}`,`到目前，共 ${fint(FA_TOTAL)}`):(faShown()?tx('the whole run','整次运行'):tx('in part 4','第 4 部分')),tip:tx('Numbers that FlashAttention moves for the same example: Q once, O once, K and V once per Q tile.','同一个例子里 FlashAttention 搬运的数值：Q 一次，O 一次，每个 Q tile 读一遍 K 和 V。')},
      {k:'on',label:tx('Scores on chip at once','同时在芯片上的得分'),v:path==='std'?fint(NT*NT):fint(TOY.B*TOY.B),sub:path==='std'?tx('the whole S','整个 S'):tx('one tile','一个 tile'),tip:tx('Standard attention handles all 144 scores of S at once; FlashAttention only one 4 × 4 tile at a time.','标准 attention 一次处理 S 的全部 144 个得分；FlashAttention 一次只处理一个 4 × 4 的 tile。')},
      (()=>{const shown=path==='fa'&&endF||faShown()&&path==='std';   // part 4's result, not before its run ends (R3.8)
        return {k:'diff',label:tx('Largest output difference','输出最大的差'),v:shown?sciT(MAXDIFF):'—',sub:tx('standard vs tiled','标准与按 tile 计算'),
          tip:shown?tx(`The largest gap between any number of the two outputs: ${sciT(MAXDIFF)}, rounding in the last bits of 64-bit numbers.`,`两种输出里任意一个数值的最大差：${sciT(MAXDIFF)}，是 64 位浮点数最后几位的舍入。`)
            :(path==='fa'?tx('Appears after the last tile pair, when both outputs are complete.','走完最后一对 tile、两种输出都算完后显示。'):tx('Appears when FlashAttention has run in part 4.','第 4 部分跑完 FlashAttention 后显示。'))};})()];}
  const p=realParams(), t=traffic(p.N,p.d,p.Br,p.dec?1:null), u=unitOf(t.std), m=sram(p.Br,p.Bc,p.d), show=faShown();
  const out=[{k:'std',label:tx('HBM traffic, standard','标准的 HBM 流量'),v:fT(t.std,u),sub:tx('one head','一个 head'),tip:tx(`Bytes standard attention moves through HBM for one head: ${fT(t.std,u)}. Unfused code moves S a few more times.`,`一个 head 用标准 attention 经 HBM 搬运的字节：${fT(t.std,u)}。未融合的代码还会多搬几次 S。`)},
    {k:'fa',label:tx('HBM traffic, FlashAttention','FlashAttention 的 HBM 流量'),v:show?fT(t.fa,u):'—',sub:show?tx(`Q tiles of ${p.Br} rows`,`Q tile ${p.Br} 行`):tx('in part 4','第 4 部分'),tip:show?(t.passes===1?tx(`With one Q tile, K and V are read only once, with no re-reads, so ${fT(t.fa,u)} is already the least any kernel moves.`,`只有一个 Q tile，K 和 V 只读一次，没有重读，所以 ${fT(t.fa,u)} 已经是最少要搬的量。`):tx(`FlashAttention moves ${fT(t.fa,u)} for one head with Q tiles of ${p.Br} rows, every re-read of K and V counted as HBM traffic. An H100’s 50 MB L2 cache probably serves many of these re-reads, so the real traffic lies between ${fT(t.floor)} (Q, K and V read once, O written once) and ${fT(t.fa,u)}.`,`Q tile 为 ${p.Br} 行时，一个 head 用 FlashAttention 搬 ${fT(t.fa,u)}，每次重读 K 和 V 都算 HBM 流量。H100 的 50 MB L2 cache 很可能送出其中不少重读，所以真实流量在 ${fT(t.floor)}（Q、K、V 各读一次，O 写一次）和 ${fT(t.fa,u)} 之间。`)):tx('FlashAttention’s traffic appears once you have stepped through it in part 4.','FlashAttention 的流量在第 4 部分走完之后显示。')},
    {k:'s',label:tx('S for one head','一个 head 的 S'),v:mode==='challenge'?fB2(t.s):fS(t.s),sub:tx('stored by standard','标准方法要存'),tip:tx(`The room S takes in HBM for one head: ${p.dec?'1':fTok(p.N)} × ${fTok(p.N)} scores × 2 bytes.`,`一个 head 的 S 在 HBM 里占的空间：${p.dec?'1':fTok(p.N)} × ${fTok(p.N)} 个得分 × 2 字节。`)}];
  if(mode!=='guided'||isP('tile')) out.push({k:'sram',label:tx('SRAM, one tile pair','一对 tile 的 SRAM'),v:fK(m.total),sub:tx('of 227 KiB','上限 227 KiB'),tip:tx(`On-chip memory for one Q tile of ${p.Br} rows and one K/V tile of ${p.Bc} rows: ${fK(m.total)}.`,`${p.Br} 行的 Q tile 加 ${p.Bc} 行的 K/V tile 在芯片上占的内存：${fK(m.total)}。`)});
  return out;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
const unitStr=v=>(String(v).match(/[A-Za-z]+$/)||[''])[0];
function drawMetrics(){
  const host=$('#fa-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join()+viewNow()+pathNow();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&viewNow()!=='toy'&&prev!=null&&prev!==m.v&&unitStr(prev)===unitStr(m.v)){   /* steps of the example are not setting changes (R6.6) */const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each block is one lead line in
   words, the quantity's name and its formula, the reader's numbers under it, then the result. The rows form a grid, so each number
   sits under the term it replaces and the “=” signs line up (R6.29). A cell is TeX, or {h:html} for the numbers, whose factors keep
   their hovers and jumps (R1.4) */
const FMC=(c,i,cl)=>`<span class="fa-mc${cl||''}${c&&c.h!=null?' fa-mh':''}">${c==null||c===''?'':(c.h!=null?c.h:mx(c))}</span>`;
/* a long sum (FS: its formula row and the reader's numbers under it) is cut before each “+” into pieces; each piece keeps its numbers
   under its terms, and the pieces wrap onto the next line, indented under the first term, when the box is narrow (the owner's review
   of 2026-10-09: a row the reader has to scroll sideways is not read). The rows after the sum (“= result”) follow on lines of their
   own, their “=” under the sum's */
const FS=(...rows)=>({seg:rows});
const FSG=(rows,rest)=>{const m=rows[0].length, cut=[1]; for(let i=2;i<m;i++) if(rows[0][i]==='+') cut.push(i); cut.push(m);
  const pc=(rs,a,b,cl)=>`<span class="fa-sg${cl||''}" style="grid-template-columns:repeat(${b-a},auto)">`+rs.map(r=>r.slice(a,b).map(c=>FMC(c)).join('')).join('')+'</span>';
  return `<div class="fa-mg" style="--n:2;--n1:1">${FMC(rows[0][0],0,' fa-mn fa-sn')}<span class="fa-mc fa-sw">`+
    cut.slice(0,-1).map((a,k)=>pc(rows,a,cut[k+1])).join('')+rest.map(r=>pc([r],1,r.length,' fa-sl')).join('')+'</span></div>';};
/* the last cell of a short row spans the rest (not a fixed column), so the rows still line up when the name moves above them */
const FMG=rows=>{const k=rows.findIndex(r=>r.seg); if(k>=0) return FSG(rows[k].seg,rows.slice(k+1));
  const n=Math.max(...rows.map(r=>r.length));
  return `<div class="fa-mg" style="--n:${n};--n1:${n-1}">`+rows.map(r=>r.map((c,i)=>{const w=i===r.length-1&&r.length<n;
    return `<span class="fa-mc${i===0?' fa-mn':''}${w?' fa-mw':''}${c&&c.h!=null?' fa-mh':''}"${w?` style="grid-column:span ${n-i}"`:''}>${c==null||c===''?'':(c.h!=null?c.h:mx(c))}</span>`;}).join('')).join('')+'</div>';};
const FMB=(lead,...grids)=>`<div class="fa-mb"><p class="fa-ml">${lead}</p>${grids.map(FMG).join('')}</div>`;
const FH=h=>({h});
function mathHTML(){
  const v=viewNow(), BY=tx('bytes','字节'), D_=(x,tip)=>DEF(String(x),tip), X=x=>'\\text{'+x+'}';
  if(v==='toy'){const k=tstep, path=pathNow(), cur=path==='fa'&&k>0?TIL.steps[k-1]:null, faDone=mode!=='guided'||doneFa;
    const n12=D_(NT,tx('N = 12 tokens in the example','N = 12：例子里的 token 数')), d4=D_(DT,tx('d = 4: the head size of the example','d = 4：例子的 head 维数')), NUM=tx('numbers','个数值');
    let h=FMB(tx('Standard attention reads Q, K and V once and writes O once. It also writes S and P out and reads them back. N is the number of tokens (12 here) and d the head size (4).','标准 attention 把 Q、K、V 各读一次，把 O 写一次，还要把 S 和 P 各写出一次、读回一次。N 是 token 数（这里是 12），d 是 head 维数（4）。'),
      [FS([X(tx('numbers moved, standard','标准方法搬运的数值')),'=','4 N d','+','4 N^2'],
       ['','=',FH(`${D_(4,tx(`Q, K and V read and O written: four ${mx('N \\times d')} matrices, each moved once`,`读 Q、K、V，写 O：四个 ${mx('N \\times d')} 的矩阵，各搬一次`))} × ${n12} × ${d4}`),'+',FH(`${D_(4,tx(`S written and read, P written and read: four moves of ${mx('N \\times N')} numbers`,`S 写出又读回，P 写出又读回：四次 ${mx('N \\times N')} 个数值`))} × ${n12}<sup>2</sup>`)],
       ['','=',FH(fint(4*NT*DT)),'+',FH(fint(4*NT*NT))]),['','=',FH(`<b>${fint(STD_TOTAL)}</b> ${NUM}`)]]);
    /* FlashAttention's total and the outputs' difference are part 4's results (R3.8) */
    if(faDone) h+=FMB(tx(`FlashAttention never writes S or P out. It reads Q and writes O once, but each Q tile reads all of K and V again. ${mx('B_r')} is the rows in a Q tile (4 here), so there are 12 ÷ 4 = 3 Q tiles.`,`FlashAttention 不把 S 和 P 写出去。它读一次 Q、写一次 O，但每个 Q tile 都要把 K 和 V 重读一遍。${mx('B_r')} 是一个 Q tile 的行数（这里是 4），所以一共 12 ÷ 4 = 3 个 Q tile。`),
      [FS([X(tx('numbers moved, FlashAttention','FlashAttention 搬运的数值')),'=','2 N d','+','\\frac{N}{B_r} \\times 2 N d'],
       ['','=',FH(`${D_(2,tx(`Q read and O written: two ${mx('N \\times d')} matrices, each once`,`读 Q、写 O：两个 ${mx('N \\times d')} 的矩阵，各一次`))} × ${n12} × ${d4}`),'+',FH(`${D_(3,tx('12 ÷ 4 = 3 Q tiles, each reading all of K and V','12 ÷ 4 = 3 个 Q tile，每个都把 K 和 V 读一遍'))} × ${D_(2,tx(`K and V: two ${mx('N \\times d')} matrices`,`K 和 V：两个 ${mx('N \\times d')} 的矩阵`))} × ${n12} × ${d4}`)],
       ['','=',FH(fint(2*NT*DT)),'+',FH(fint(3*2*NT*DT))]),['','=',FH(`<b>${fint(FA_TOTAL)}</b> ${NUM}`)]]);
    if(path==='fa'||faDone) h+=FMB(tx(`The online softmax keeps three things for each row i while the tiles of scores arrive: m, the largest score so far; ℓ, the running sum; and o, the output so far, not yet divided by ℓ. Each new tile of scores ${mx('S_{ij}')} updates them (${mx('v_j')} is token j’s value):`,`online softmax 在得分 tile 一个个到来时，给每一行 i 记三样东西：m 是目前最大的得分，ℓ 是累计和，o 是目前的输出（还没除以 ℓ）。每来一个新的得分 tile ${mx('S_{ij}')}，就这样更新它们（${mx('v_j')} 是第 j 个 token 的 value）：`),
      [['m_{\\text{new}}','=','\\max\\left(m,\\ \\max_j S_{ij}\\right)'],['ℓ','\\leftarrow','ℓ \\cdot e^{m - m_{\\text{new}}} + \\sum_j e^{S_{ij} - m_{\\text{new}}}'],
       ['o','\\leftarrow','o \\cdot e^{m - m_{\\text{new}}} + \\sum_j e^{S_{ij} - m_{\\text{new}}} v_j']])+
      FMB(tx('After the last tile, the row’s output is the output so far divided by the running sum:','读完最后一个 tile，这一行的输出就是目前的输出除以累计和：'),[['O','=','\\frac{o}{ℓ}']]);
    else h+=`<p class="muted">${tx('FlashAttention’s count and the difference between the two outputs appear after part 4.','FlashAttention 的计数和两种输出的差，在第 4 部分之后显示。')}</p>`;
    if(cur){const r=0, fr=cur.f[r], mo=cur.mOld[r]===-Infinity?'−∞':f4(cur.mOld[r]);
      h+=FMB(tx(`Tile pair ${k}, row ${cur.i*TOY.B+1}: the new tile’s largest score is ${f4(cur.tmax[r])}.`,`第 ${k} 对 tile，第 ${cur.i*TOY.B+1} 行：新 tile 里最大的得分是 ${f4(cur.tmax[r])}。`),
        [['m_{\\text{new}}','=','\\max\\left(m,\\ \\max_j S_{ij}\\right)'],['','=',FH(`max(${mo}, ${f4(cur.tmax[r])})`)],['','=',FH(f4(cur.m[r]))]].concat(cur.j===0?[]:
        [[X(tx('rescale factor','缩放系数')),'=','e^{m - m_{\\text{new}}}'],['','=',FH(`e<sup>${f4(cur.mOld[r])} − ${f4(cur.m[r])}</sup>`)],['','=',FH(f4(fr))]]),
        [['ℓ','=',FH(`${f4(cur.l[r])}${cur.j===0?tx(' (first tile)','（第一个 tile）'):''}`)]]);}
    if(path==='fa'&&k>=TIL.steps.length||faDone&&path==='std') h+=FMB(tx('The two outputs differ only by rounding:','两种输出只差舍入：'),[[X(tx('largest difference','最大的差')),'=',FH(sci(MAXDIFF))]]);
    return h;}
  const p=realParams(), t=traffic(p.N,p.d,p.Br,p.dec?1:null), m=sram(p.Br,p.Bc,p.d), fN=fint(p.N), fNq=fint(t.Nq);
  const tN=T_('N',fN), tNq=p.dec?T_('Nq',fNq):T_('N',fNq), td=T_('d',p.d), tBr=T_('Br',p.Br), tBc=T_('Bc',p.Bc), b2=T_('b2','2'), b4=T_('b4','4'), b8=T_('b8','8');
  const pre=mode==='guided'&&step<SIX('fa');      // tiles are taught in part 4 (R2.8)
  const sym=tx(`${mx('N_q')} is the number of query tokens, N the tokens in the context and d the head size. Each number takes 2 bytes.`,`${mx('N_q')} 是 query 的 token 数，N 是上下文的 token 数，d 是 head 维数。每个数值占 2 字节。`);
  let h=FMB(tx('Standard attention moves Q, K, V and O once, and S and P (one number per query and key) out and back. ','标准 attention 把 Q、K、V、O 各搬一次，还要把 S 和 P（每个 query 和 key 一个数值）写出去再读回来。')+sym,
    [FS([X(tx('bytes moved, standard','标准方法搬运的字节')),'=','2 \\times \\Big(','2 N_q d','+','2 N d','+','4 N_q N','\\Big)'],
     ['','=',FH(`${b2} × (`),FH(`2 × ${tNq} × ${td}`),'+',FH(`2 × ${tN} × ${td}`),'+',FH(`4 × ${tNq} × ${tN}`),FH(')')]),
     ['','=',FH(`${fint(t.std)} ${BY} = <b>${fT(t.std)}</b>`)]]);
  if(!pre){
    h+=FMB(tx(`FlashAttention reads Q and writes O once, and reads all of K and V once for each Q tile of ${mx('B_r')} rows. The number of Q tiles is the number of passes over K and V:`,`FlashAttention 读一次 Q、写一次 O，每个 ${mx('B_r')} 行的 Q tile 都把 K 和 V 读一遍。Q tile 有几个，就要扫 K、V 几遍：`),
      [[X(tx('passes','遍数')),'=','\\left⌈\\frac{N_q}{B_r}\\right⌉'],['','=',FH(`⌈${tNq} ÷ ${tBr}⌉`)],['','=',FH(fint(t.passes))]],
      [FS([X(tx('bytes moved, FlashAttention','FlashAttention 搬运的字节')),'=','2 \\times \\Big(','2 N_q d','+',X(tx('passes','遍数'))+' \\times 2 N d','\\Big)'],
       ['','=',FH(`${b2} × (`),FH(`2 × ${tNq} × ${td}`),'+',FH(`${fint(t.passes)} × 2 × ${tN} × ${td}`),FH(')')]),
       ['','=',FH(`${fint(t.fa)} ${BY} = <b>${fT(t.fa)}</b> ${tx('(every re-read of K and V counted)','（K、V 的每次重读都算上）')}`)]]);
    h+=FMB(tx('No kernel can move less than reading Q, K and V once and writing O once:','任何 kernel 都至少要把 Q、K、V 各读一次、O 写一次：'),
      [FS([X(tx('least bytes','最少的字节')),'=','2 \\times \\Big(','2 N_q d','+','2 N d','\\Big)'],['','=',FH(`${b2} × (`),FH(`2 × ${tNq} × ${td}`),'+',FH(`2 × ${tN} × ${td}`),FH(')')]),['','=',FH(fT(t.floor))]]);}
  h+=FMB(tx('S alone holds one score for every query and key:','光是 S，就要给每个 query 和 key 存一个得分：'),
    [[X(tx('size of S','S 的大小')),'=','N_q \\times N \\times 2'],['','=',FH(`${tNq} × ${tN} × ${b2}`)],['','=',FH(`${fint(t.s)} ${BY} = ${mode==='challenge'?fB2(t.s):fS(t.s)}`)]]);
  if(mode==='sandbox'&&!p.dec&&p.N===131072&&p.d===128) h+=FMB(tx('The arithmetic of one layer’s 32 heads, and its time at the H100’s 989.5 TFLOP/s:','一层 32 个 head 的 attention 运算量，以及按 H100 的 989.5 TFLOP/s 算要多久：'),
    [[X(tx('FLOPs','运算量')),'=','4 N^2 d \\times 32'],['','=',FH(`4 × ${tN}<sup>2</sup> × ${td} × 32`)],['','=',FH(`${fint(Math.round(4*p.N*p.N*p.d*32/1e12))} TFLOP ≈ ${fixd(4*p.N*p.N*p.d*32/HW.flops,2)} s`)]]);
  if(!pre) h+=FMB(tx(`One Q tile and one K/V tile, with their scores, output and m and ℓ, must fit in a thread block’s SRAM. ${mx('B_c')} is the rows in a K/V tile. The scores, the output, m and ℓ take 4 bytes per number.`,`一个 Q tile、一个 K/V tile，连同它们的得分、输出和 m、ℓ，要放得进一个 thread block 的 SRAM。${mx('B_c')} 是一个 K/V tile 的行数。得分、输出和 m、ℓ 每个数值占 4 字节。`),
    [FS([X(tx('SRAM for a tile pair','一对 tile 的 SRAM')),'=','2 B_r d','+','2 \\times 2 B_c d','+','4 B_r B_c','+','4 B_r d','+','8 B_r'],
     ['','=',FH(`${b2} × ${tBr} × ${td}`),'+',FH(`${b2} × ${T_('kv','2')} × ${tBc} × ${td}`),'+',FH(`${b4} × ${tBr} × ${tBc}`),'+',FH(`${b4} × ${tBr} × ${td}`),'+',FH(`${b8} × ${tBr}`)]),
     ['','=',FH(`${fint(m.total)} ${BY} = ${fK(m.total)} ${m.fits?'≤':'>'} 227 KiB`)]]);
  return h;
}
const preTiles=()=>mode==='guided'&&step<SIX('fa');   // parts 1–3: tiles and passes over K and V are not taught yet (R2.8)
function drawHead(){
  const v=viewNow(), t=$('#fa-st-title',app), sub=$('#fa-st-sub',app), st=$('#fa-status',app);
  if(v==='toy'){t.textContent=tx('12-token example, one head, head size 4','12 token 的例子，一个 head，head 维数 4');
    sub.textContent=pathNow()==='std'?tx('Standard attention: S and P go out to HBM and back','标准 attention：S 和 P 写到 HBM 再读回来'):tx('FlashAttention: one tile of scores at a time, on the chip','FlashAttention：一次算一个得分 tile，都在芯片上');}
  else {const p=realParams(); t.textContent=mode==='challenge'?tx('Tiles for 32K tokens on an H100','H100 上 32K token 的 tile'):tx('How many bytes cross HBM, per head','每个 head 经 HBM 搬运多少字节');
    sub.textContent=mode==='challenge'?tx(`${M1}, head size 64, BF16`,`${M1}，head 维数 64，BF16`):(preTiles()?tx('H100, BF16','H100，BF16'):tx(`H100, BF16, Q tiles of ${p.Br} rows`,`H100，BF16，Q tile ${p.Br} 行`));}
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const vw=$('#fa-view',app); vw.hidden=mode!=='sandbox'; $$('button',vw).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===S.view)));
  const ps=$('#fa-pathsw',app); ps.hidden=!(mode==='sandbox'&&S.view==='toy'); $$('button',ps).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.path===S.path)));
  const sc=$('#fa-scope',app);
  if(v==='toy'){sc.textContent=tx('A 12-token example computed for real; at real size the same steps move far more bytes.','一个真实算出的 12 token 例子；真实规模下，同样的步骤搬运的字节多得多。'); sc.dataset.tipx=tx('Q, K and V are random numbers from a fixed seed. The steps and the counts are exactly those of the real computation, at a small size.','Q、K、V 是固定种子生成的随机数。步骤和计数与真实计算完全一样，只是规模小。');}
  else if(preTiles()){sc.textContent=tx('Traffic comes from a formula.','流量是用公式算出的。'); sc.dataset.tipx=TIP.countPre();}
  else if(traffic(realParams().N,realParams().d,realParams().Br,realParams().dec?1:null).passes===1){sc.textContent=tx('One Q tile: K and V are read once, with no re-reads.','只有一个 Q tile：K 和 V 只读一次，没有重读。'); sc.dataset.tipx=tx('Bytes per head from a formula, 2 bytes a number. With one Q tile, FlashAttention reads each key and value once, the least any kernel moves.','每个 head 的字节数是用公式算的，每个数值 2 字节。只有一个 Q tile 时，FlashAttention 每个 key 和 value 只读一次，这已经是任何 kernel 最少要搬的量。');}
  else {sc.textContent=tx('Every re-read of K and V counts as HBM traffic here. The L2 cache probably serves many of them.','这里把每次重读 K 和 V 都算作 HBM 流量。L2 cache 很可能送出其中不少。'); sc.dataset.tipx=TIP.count();}
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!TILES.includes(S.Br)) S.Br=CH.start.Br; if(!TILES.includes(S.Bc)) S.Bc=CH.start.Bc;}
  const v=viewNow();
  if(mode==='guided'){const s=STEPS[step]; if(s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  $('#fa-toy',app).hidden=v!=='toy'; $('#fa-real',app).hidden=v!=='real';
  if(v==='toy') drawToy(); else drawReal();
  drawMetrics();
  const lg=$('#fa-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;} lg.hidden=!lh;
  const mb=$('#fa-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the two tile sizes, in the task card. Everything else is fixed: ${M1} (head size 64), 32K tokens, prefill, H100.`,`挑战中只能在任务卡里改两个 tile 大小。其余固定：${M1}（head 维数 64），32K token，prefill，H100。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  /* parts 3 and 4 run the small example, not the real-size model (GPT-6 review of v2, finding 4) */
  if(STEPS[step].view==='toy') return id==='fa'
    ? tx(`Fixed in this part: the small example, ${NT} tokens with a head size of ${DT}, cut into Q and K/V tiles of ${TOY.B} rows. Part ${SIX('len')+1} goes back to ${M8} at full size.`,`这一部分固定：小例子，${NT} 个 token，head 维数 ${DT}，切成每个 ${TOY.B} 行的 Q 和 K/V tile。第 ${SIX('len')+1} 部分回到真实大小的 ${M8}。`)
    : tx(`Fixed in this part: the small example, ${NT} tokens with a head size of ${DT}. Part ${SIX('len')+1} goes back to ${M8} at full size.`,`这一部分固定：小例子，${NT} 个 token，head 维数 ${DT}。第 ${SIX('len')+1} 部分回到真实大小的 ${M8}。`);
  if(id==='len') return tx(`Fixed in this part: ${M8} (head size 128), prefill, Q and K/V tiles of 128 rows. The context length is in the card. You can change everything in Scenarios.`,`这一部分固定：${M8}（head 维数 128），prefill，Q 和 K/V tile 各 128 行。上下文长度在卡片里改。在应用场景里可以改全部设置。`);
  if(id==='tile') return tx(`Fixed in this part: ${M8}, 8K tokens, prefill. The tile size is in the card. You can change everything in Scenarios.`,`这一部分固定：${M8}，8K token，prefill。tile 大小在卡片里选。在应用场景里可以改全部设置。`);
  if(preTiles()) return tx(`Fixed in this part: ${M8} (head size 128), 8K tokens, prefill. You can change everything in Scenarios.`,`这一部分固定：${M8}（head 维数 128），8K token，prefill。在应用场景里可以改全部设置。`);
  return tx(`Fixed in this part: ${M8} (head size 128), 8K tokens, prefill, tiles of 128 rows. You can change everything in Scenarios.`,`这一部分固定：${M8}（head 维数 128），8K token，prefill，tile 各 128 行。在应用场景里可以改全部设置。`);
}
function syncControls(){
  $$('input[data-key="N"]',app).forEach(i=>{if(+i.value!==S.ni) i.value=S.ni; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl'); const out=o&&o.querySelector('output'); if(out) out.textContent=fTok(N_())+tx(' tokens',' token'); i.setAttribute('aria-valuetext',fTok(N_())+tx(' tokens',' token'));});
  $$('.seg-ctl[data-key="d"]',app).forEach(g=>A.ui.setPressed(g,String(S.d)));
  $$('.seg-ctl[data-key="dec"]',app).forEach(g=>A.ui.setPressed(g,S.dec?'1':'0'));
  $$('#fa-controls .seg-ctl[data-key="Br"]',app).forEach(g=>A.ui.setPressed(g,String(S.Br)));
  $$('#fa-controls .seg-ctl[data-key="Bc"]',app).forEach(g=>A.ui.setPressed(g,String(S.Bc)));
  const c=$('#fa-controls',app), lock=mode!=='sandbox';
  c.classList.toggle('locked',lock);
  const w=$('#fa-ctl-wrap',app); if(w) w.hidden=lock;
  const ln=$('#fa-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.tip);},400);}

/* =====================================================================
   the player: ◂ ▸, play / pause, speed, scrubber (time-based, R3.9); the last step is held (R7.10)
   ===================================================================== */
function markDone(){const n=stepsN(); if(tstep>=n){if(pathNow()==='std') doneStd=true; else doneFa=true;}}
function setStep(k){tstep=Math.max(0,Math.min(stepsN(),k)); markDone(); render();}
function toEnd(){stopPlay(); setStep(stepsN()); wantObs=true; revealSoon(); wantObs=false;}
function stopPlay(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0;}
function startPlay(wait){
  if(A.reduceMotion()){toEnd(); return;}
  if(tstep>=stepsN()) tstep=0;
  playing=true; lastT=0; acc=Math.max(0,stepMs()-300-(wait||0));   // the first step comes 0.3 s after Play (after the scroll, below)
  const tick=ts=>{if(!playing||!app) return; if(!lastT) lastT=ts; acc+=ts-lastT; lastT=ts;
    if(acc>=stepMs()){acc=0; byLearner(()=>setStep(tstep+1)); if(tstep>=stepsN()){stopPlay(); render(); if(mode==='guided'&&!playPhone){wantObs=true; revealSoon(); wantObs=false;} return;}}
    raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick); syncPlayer();
}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  if(mode==='sandbox'&&S.view==='toy'&&!('view' in p)&&['d','ni','dec','Br','Bc'].some(k=>k in p)){p={...p,view:'real'}; stopPlay();}   // the example has fixed sizes: a setting change shows the real size (R1.9)
  Object.assign(S,p);
  if(!NS[S.ni]) S.ni=3; if(![64,128].includes(S.d)) S.d=128; if(!TILES.includes(S.Br)) S.Br=128; if(!TILES.includes(S.Bc)) S.Bc=128;
  if(isP('len')) seenN.add(S.ni);
  render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(b){lv=b; lvTried.add(b); render();}
function enterStep(i){
  stopPlay(); wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.view==='toy'){S.path=s.path; tstep=(s.path==='std'?doneStd:doneFa)?stepsN():0;}
  if(s.id==='len'){S.ni=3; seenN=new Set([3,...(observed[step]?[NS.length-1]:[])]);}
  const st=$('#fa-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#fa-rail .step.current',app)||$('#fa-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#fa-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#fa-insight',app)); else revealBox($('#fa-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#fa-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#fa-rail .step.current',app)||$('#fa-rail .railcard',app)||$('#fa-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function setMode(m){
  if(m===mode) return;
  stopPlay();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,Br:CH.start.Br,Bc:CH.start.Bc};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided'){S={...BASE}; tstep=doneStd?stepsN():0;}}   // R3.20: a run watched to the end shows its end
  A.shell.setMode(app,m,DESCS());
  const st=$('#fa-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.d,s.ni,s.Br,s.Bc,s.dec?1:0,s.view==='toy'?'t':'r',s.path==='fa'?'f':'s'].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const [d,ni,Br,Bc,dec]=p.slice(0,5).map(Number);
  if(![64,128].includes(d)||!Number.isInteger(ni)||ni<0||ni>=NS.length||!TILES.includes(Br)||!TILES.includes(Bc)||![0,1].includes(dec)||!['t','r'].includes(p[5])||!['s','f'].includes(p[6])) return null;
  return {d,ni,Br,Bc,dec:!!dec,view:p[5]==='t'?'toy':'real',path:p[6]==='f'?'fa':'std'};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function makeDrill(seed){const r=mulberry(seed*104729+7), pk=a=>a[Math.floor(r()*a.length)];
  const N=pk([2048,4096,16384,32768]), d=pk([64,128]), Br=pk([64,128]); return {seed,N,d,Br,t:traffic(N,d,Br)};}
function renderDrill(){
  if(!app) return; const q=makeDrill(drillSeed), qd=$('#fa-drill-q',app), a=$('#fa-drill-a',app), t=q.t;
  qd.innerHTML=tx(`One attention head, <b>${fint(q.N)}</b> tokens, head size <b>${q.d}</b>, BF16. How many bytes does standard attention move through HBM, and how many does FlashAttention with Q tiles of <b>${q.Br}</b> rows, counting every pass over K and V?`,
                  `一个 attention head，<b>${fint(q.N)}</b> 个 token，head 维数 <b>${q.d}</b>，BF16。标准 attention 经 HBM 搬运多少字节？Q tile 为 <b>${q.Br}</b> 行、每次扫 K 和 V 都算在内时，FlashAttention 搬运多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>Standard ${fB(t.std)} and FlashAttention ${fB(t.fa)}: standard moves ${fixd(t.ratio,1)} times as much.</b></p>`+
    `<p>Standard: Q, K, V read and O written, 4 × ${fint(q.N)} × ${q.d} numbers; S and P each written and read, 4 × ${fint(q.N)}² numbers; × 2 bytes = ${fint(t.std)} bytes.</p>`+
    `<p>FlashAttention: Q read and O written, 2 × ${fint(q.N)} × ${q.d}; K and V read once per Q tile, ${fint(t.passes)} × 2 × ${fint(q.N)} × ${q.d}; × 2 bytes = ${fint(t.fa)} bytes. Per score, K and V again cost 4 × ${q.d} ÷ ${q.Br} = ${fixd(4*q.d/q.Br,0)} bytes, against 8 for S and P.</p>`,
    `<p><b>标准 ${fB(t.std)}，FlashAttention ${fB(t.fa)}：标准是 FlashAttention 的 ${fixd(t.ratio,1)} 倍。</b></p>`+
    `<p>标准：读 Q、K、V，写 O，4 × ${fint(q.N)} × ${q.d} 个数值；S 和 P 各写一次、读一次，4 × ${fint(q.N)}² 个数值；× 2 字节 = ${fint(t.std)} 字节。</p>`+
    `<p>FlashAttention：读 Q、写 O，2 × ${fint(q.N)} × ${q.d}；每个 Q tile 读一遍 K 和 V，${fint(t.passes)} × 2 × ${fint(q.N)} × ${q.d}；× 2 字节 = ${fint(t.fa)} 字节。每个得分重读 K、V 的代价是 4 × ${q.d} ÷ ${q.Br} = ${fixd(4*q.d/q.Br,0)} 字节，S 和 P 是 8 字节。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={N:{en:'Context length',zh:'上下文长度'},d:{en:'Head size d',zh:'head 维数 d'},Br:{en:'Q tile',zh:'Q tile'},Bc:{en:'K/V tile',zh:'K/V tile'}};
const TERM={
 N:()=>`<div>${esc(tx(`N = ${fint(realParams().N)}: tokens in the context, the keys every query is scored against.`,`N = ${fint(realParams().N)}：上下文里的 token 数，每个 query 都要和这么多 key 打分。`))}</div>`,
 Nq:()=>`<div>${esc(tx('Nq = 1: query rows. Decode computes a query only for the new token; in prefill Nq = N.','Nq = 1：query 的行数。decode 只为新 token 算 query；prefill 时 Nq = N。'))}</div>`,
 d:()=>`<div>${esc(tx(`d = ${realParams().d}: the head size, numbers per query, key and value.`,`d = ${realParams().d}：head 维数，每个 query、key、value 的维数。`))}</div>`,
 Br:()=>`<div>${esc(tx(`Br = ${realParams().Br}: rows of Q in one tile. Every Q tile reads all of K and V once.`,`Br = ${realParams().Br}：一个 Q tile 的行数。每个 Q tile 都把 K 和 V 读一遍。`))}</div>`,
 Bc:()=>`<div>${esc(tx(`Bc = ${realParams().Bc}: rows of K, and of V, in one tile. It changes the SRAM, not the traffic.`,`Bc = ${realParams().Bc}：一个 K tile（或 V tile）的行数。它改变 SRAM，不改变流量。`))}</div>`,
 b2:()=>`<div>${esc(tx('2 bytes a number, in BF16.','每个数值 2 字节（BF16）。'))}</div>`,
 b4:()=>`<div>${esc(tx('4 bytes a number, in FP32: kernels accumulate the scores and the output in FP32.','每个数值 4 字节（FP32）：kernel 用 FP32 累加得分和输出。'))}</div>`,
 b8:()=>`<div>${esc(tx('8 bytes a row: the running max m and the running sum ℓ, 4 bytes each.','每一行 8 字节：累计最大值 m 和累计和 ℓ，各 4 字节。'))}</div>`,
 kv:()=>`<div>${esc(tx('2 tiles: one of K and one of V.','2 个 tile：K 一个，V 一个。'))}</div>`
};
function ctlLockMsg(k){
  if(mode==='sandbox') return '';
  if(mode==='challenge') return k==='Br'||k==='Bc'?'':tx(`Fixed during the challenge: ${M1}, head size 64, 32K tokens.`,`挑战中固定不变：${M1}，head 维数 64，32K token。`);
  if(k==='N'&&isP('len')) return '';
  if((k==='Br'||k==='Bc')&&isP('tile')) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='row') return tx('Click to jump to the table of row 1 under the chart.','点击跳到图下第 1 行的表。');
  if(k==='gauge') return tx('Click to jump to the SRAM gauge in the chart.','点击跳到图里的 SRAM 刻度条。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  if(mode==='challenge') return tx(`Click to jump to “${L(CTLN[k])}” in the task card.`,`点击跳到任务卡里的“${L(CTLN[k])}”。`);
  if(isP('tile')) return tx('Click to jump to the tile sizes in the card.','点击跳到卡片里的 tile 大小。');
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const termTipHTML=k=>{if(!TERM[k]) return ''; const a=jumpAct(k); return TERM[k]()+(a?`<div class="act">${esc(a)}</div>`:'');};
const ctlEl=k=>({N:$('#fa-ic-N',app)||$('#fa-ctl-N',app),d:$('#fa-ctl-d',app),Br:$('#fa-ic-chBr',app)||$('#fa-ic-lv',app)||$('#fa-ctl-Br',app),Bc:$('#fa-ic-chBc',app)||$('#fa-ic-lv',app)||$('#fa-ctl-Bc',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#fa-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#fa-controls',app)); return;}
  if(k==='row'||k==='gauge'){A.tip.hide(); const el=$(k==='row'?'#fa-row':'#fa-gauge',app); if(el&&!el.hidden) A.ui.jumpToEl(el); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#fa-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
/* phones: the stage's own player does not scroll the page to the card at the end; the stage offers a link instead (R1.11) */
const fromStagePhone=b=>!!(b&&b.closest('#fa-stage'))&&innerWidth<=760;
let playPhone=false;
/* phones: the drawing sits above the card, so the card's ▶ first scrolls the drawing (HBM, SRAM and the row table) under the bars,
   then plays; the result is revealed at the end as before (round-4 reader, B1 and B4: the run played out of sight) */
function stageToView(btn){
  if(innerWidth>760||!btn||!btn.closest('#fa-rail')||A.reduceMotion()) return false;
  const g=$('#fa-graph',app); if(!g) return false;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:50)+8, r=g.getBoundingClientRect();
  if(Math.abs(r.top-top)<8||(r.top>=top&&r.bottom<=innerHeight)) return false;
  window.scrollBy({top:r.top-top,behavior:'smooth'});
  return true;
}
function act(a,btn){
  if(a==='fwd'){stopPlay(); A.tip.quiet(); byLearner(()=>setStep(tstep+1)); if(tstep>=stepsN()&&mode==='guided'&&!fromStagePhone(btn)){wantObs=true; revealSoon(); wantObs=false;} return;}
  if(a==='toresult'){wantObs=true; revealObs(); wantObs=false; return;}
  if(a==='back'){stopPlay(); A.tip.quiet(); byLearner(()=>setStep(tstep-1)); return;}
  if(a==='play'){if(playing){stopPlay(); syncPlayer();} else {playPhone=fromStagePhone(btn); startPlay(stageToView(btn)?600:0);} return;}
  if(a==='reset'){bySettings(btn,()=>{S={...BASE,view:S.view,path:S.path}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.Br,S.Bc).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#fa-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='N') bySettings(e.target,()=>setState({ni:v}));
    else if(k==='tstep'){stopPlay(); byLearner(()=>setStep(v));}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#fa-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#fa-rail')&&!b.closest('[data-exits]')&&!d.act) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#fa-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('tile')){const f=()=>byLearner(()=>setLv(+d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.view!=null&&b.closest('#fa-view')){if(d.view===S.view) return; stopPlay(); byLearner(()=>setState({view:d.view})); return;}
    if(d.path!=null&&b.closest('#fa-pathsw')){if(d.path===S.path) return; stopPlay(); S.path=d.path; tstep=(d.path==='std'?doneStd:doneFa)?stepsN():0; byLearner(()=>setState({path:d.path})); return;}   // R3.20
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='speed'){speed=d.v; syncPlayer(); return;}
      if(k==='d') bySettings(b,()=>setState({d:+d.v}));
      else if(k==='dec') bySettings(b,()=>setState({dec:d.v==='1'}));
      else if(k==='Br') bySettings(b,()=>setState({Br:+d.v}));
      else if(k==='Bc') bySettings(b,()=>setState({Bc:+d.v}));
      else if(k==='chBr') byLearner(()=>setState({Br:+d.v}));
      else if(k==='chBc') byLearner(()=>setState({Bc:+d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s,view:'real'})); requestAnimationFrame(()=>{if(app) revealBox($('#fa-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#fa-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
const sOf=s=>({d:s.d,ni:s.ni,Br:s.Br,Bc:s.Bc,dec:s.dec,view:s.view,path:s.path});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:sOf(S),pred,observed:{...observed},tstep,speed,lv,lvTried:[...lvTried],seenN:[...seenN],doneStd,doneFa,saved:saved?sOf(saved):null,
    chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.Br,S.Bc).ok}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=sOf(p); const st=STEPS[step]; if(mode==='guided'&&st.view==='toy') S.path=st.path;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    doneStd=o.doneStd===true; doneFa=o.doneFa===true; if(SPEEDS[o.speed]) speed=o.speed;
    tstep=Number.isInteger(o.tstep)?Math.max(0,Math.min(stepsN(),o.tstep)):0;
    lv=TILES.includes(o.lv)?o.lv:128; lvTried=new Set([128,...(Array.isArray(o.lvTried)?o.lvTried.filter(b=>TILES.includes(b)):[])]);
    seenN=new Set([3,...(Array.isArray(o.seenN)?o.seenN.filter(i=>Number.isInteger(i)&&i>=0&&i<NS.length):[])]);
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sOf(sv):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true; resumed=true;},
  mount(el,tok){
    wantObs=false; root=el; stopPlay();
    const p=tok&&!resumed?parse(tok):null; resumed=false;   /* after restore() the address names the saved settings: nothing to apply */ if(p){S={...p}; mode='sandbox'; sbTouched=true; tstep=0;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#fa-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#fa-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&step===0){const s=STEPS[0]; Object.assign(S,s.enter());}
    fillRail();
    render();
  },
  leave(){stopPlay(); clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#fa-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#fa-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; return viewNow()==='toy'?$('#fa-cnt',app):$('#fa-bars',app);},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#fa-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(viewNow()==='toy'){const path=pathNow(), sn=path==='std'?stdSoFar(tstep):faSoFar(tstep);
        vis.innerHTML=`<div class="fa-dk"><i style="width:${(sn/STD_TOTAL*100).toFixed(1)}%"></i></div>`;
        read.innerHTML=esc(path==='std'?tx(`Standard attention, step ${tstep} of ${stepsN()}: ${fint(sn)} numbers moved`,`标准 attention，第 ${tstep} 步（共 ${stepsN()} 步）：搬了 ${fint(sn)} 个数值`):tx(`FlashAttention, tile pair ${tstep} of ${stepsN()}: ${fint(sn)} numbers moved`,`FlashAttention，第 ${tstep} 对 tile（共 ${stepsN()} 对）：搬了 ${fint(sn)} 个数值`)); return;}
      const p=realParams(), t=traffic(p.N,p.d,p.Br,p.dec?1:null), u=unitOf(t.std), show=faShown();
      vis.innerHTML=`<div class="fa-dk"><i style="width:100%"></i></div>`+(show?`<div class="fa-dk"><i style="width:${(t.fa/t.std*100).toFixed(1)}%"></i></div>`:'');
      read.innerHTML=esc(show?tx(`${fTok(p.N)} tokens, one head: standard ${fT(t.std,u)}, FlashAttention ${fT(t.fa,u)}`,`${fTok(p.N)} token，一个 head：标准 ${fT(t.std,u)}，FlashAttention ${fT(t.fa,u)}`)
        :tx(`${fTok(p.N)} tokens, one head: standard attention moves ${fT(t.std,u)}`,`${fTok(p.N)} token，一个 head：标准 attention 搬 ${fT(t.std,u)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__fa=Object.freeze({
  traffic:(N,d,Br,Nq)=>deep(traffic(N,d,Br,Nq)), sram:(Br,Bc,d)=>deep(sram(Br,Bc,d)), times:(N,d)=>deep(times(N,d)),
  toy:()=>deep({Q:MATS.Q,K:MATS.K,V:MATS.V,stdO:STD.O,tilO:TIL.O,maxdiff:MAXDIFF,stdTotal:STD_TOTAL,faTotal:FA_TOTAL,spToy:SP_TOY,
    steps:TIL.steps.map(s=>({i:s.i,j:s.j,tmax:s.tmax,mOld:s.mOld.map(x=>x===-Infinity?null:x),m:s.m,f:s.f,l:s.l,moved:s.moved})),stdSteps:STD_STEPS.map(s=>({k:s.k,n:s.n}))}),
  chEval:(Br,Bc)=>deep(chEval(Br,Bc)), chBest, chChanges, drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({NS:NS.slice(),TILES:TILES.slice(),HW:{...HW},CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),SPEEDS:{...SPEEDS}}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transferO:e.transfer.map(q=>q.o),explainA:e.explain.map(q=>q.a),explainO:e.explain.map(q=>q.o),why:e.explain.map(q=>q.why).concat(e.transfer.map(q=>q.why))});},
  state:()=>({...S,mode,step,pred,observed:{...observed},tstep,playing,speed,lv,lvTried:[...lvTried],seenN:[...seenN],doneStd,doneFa,chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
