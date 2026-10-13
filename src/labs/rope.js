/* rope (the trunk "How LLMs work", after "Multi-head attention"): How does a model know where each token is?
   Fidelity: Live, shown as "Toy run". The vectors are made up (one head of 128 numbers per query and key, set by hand so that the
   head looks for the token right before it); the speeds are real: θᵢ = base^(−2i/128) for the 64 pairs of a 128-number head, with
   the bases of Llama‑2‑7B (10,000), Llama‑3‑8B and Llama‑3.1‑8B (500,000) and Code Llama 7B (1,000,000), and Llama‑3.1‑8B's
   "llama3" scaling as transformers computes it (_compute_llama3_parameters: factor 8, low_freq_factor 1, high_freq_factor 4,
   original length 8,192). Pairs are neighbours, numbers 2i and 2i + 1, as in Meta's Llama code and the RoFormer paper.
   The story (review of preview v3, R6.25): 1 the problem on "dog bites man"; 2 one head: 128 numbers → 64 pairs → 64 arrows, and
   "added at the input" versus "turned inside attention"; 3 RoPE on "dog bites man": what turns, by how much, what for; 4 predict, then
   move both tokens with RoPE on; 5 one chart for base, speed and text length; 6 the same chart past the training length; 7 remember.
   Positions count from 1 ("dog" 1, "bites" 2, "man" 3). Parts 1–5 use Llama‑3‑8B's speeds (no scaling), so the chart matches the
   formula; part 6 adds Llama‑3.1‑8B's scaling. window.__rp is a read-only hook for tests/rope_numbers.py and tests/rope_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='rope';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const NBH='‑', NBS=' ', MINUS='−';
const M2=`Llama${NBH}2${NBH}7B`, M3=`Llama${NBH}3${NBH}8B`, M31=`Llama${NBH}3.1${NBH}8B`, MCL=`Code${NBS}Llama${NBS}7B`;

/* =====================================================================
   engine
   ===================================================================== */
const HD=128, NP=HD/2, TAU=2*Math.PI, R2D=180/Math.PI, D2R=Math.PI/180;
const SC31=Object.freeze({factor:8,lo:1,hi:4,old:8192});        // Llama‑3.1‑8B's rope_scaling (config.json)
const THC=new Map();
/* the 64 speeds, radians per token: base^(−2i/128); with the llama3 scaling, pairs whose full turn takes more than 8,192 tokens turn
   8 times slower, pairs under 2,048 tokens keep their speed, and those in between are blended (transformers' formula) */
function thetas(base,sc){
  const key=base+'|'+(sc?'l3':'-'); if(THC.has(key)) return THC.get(key);
  const out=[];
  for(let i=0;i<NP;i++){let f=Math.pow(base,-2*i/HD);
    if(sc){const w=TAU/f;
      if(w>sc.old/sc.lo) f=f/sc.factor;
      else if(w>=sc.old/sc.hi){const s=(sc.old/w-sc.lo)/(sc.hi-sc.lo); f=(1-s)*f/sc.factor+s*f;}}
    out.push(f);}
  THC.set(key,out); return out;}
const thOf=(base,sc)=>thetas(base,sc?SC31:null);
const TH5=thetas(500000,null);                                  // Llama‑3‑8B: the head of parts 1–5
const TH0=TH5[0], TH0D=TH0*R2D;                                  // pair 0 of every Llama head: base⁰ = 1 radian (57.3°) per token
/* the made-up head: 128 numbers per query and key, pair i = numbers 2i and 2i + 1. Pair 0 is drawn large in parts 2–4. The other 63
   pairs follow simple deterministic formulas (lengths 0.22 to 0.48, angles spread by the golden angle); the keys of the nouns are set
   one pair-speed ahead of the query of "bites" (β = α + θᵢ), so this head looks for the token right before "bites" */
const frac=x=>x-Math.floor(x);
function mkVec(p0,lenf,angf){const v=new Array(HD); for(let i=0;i<NP;i++){const la=i===0?p0:[lenf(i),angf(i)]; v[2*i]=la[0]*Math.cos(la[1]*D2R); v[2*i+1]=la[0]*Math.sin(la[1]*D2R);} return v;}
const ALPHA=i=>(i*137.508)%360;
const QV={bites:mkVec([1.5,-47],i=>0.22+0.26*frac(i*0.618034),ALPHA),
  dog:mkVec([1.0,80],i=>0.22+0.26*frac(i*0.7071+0.1),i=>(i*97.3+20)%360),
  man:mkVec([1.1,110],i=>0.22+0.26*frac(i*0.5772+0.6),i=>(i*61.1+200)%360)};
const nounK=p0=>mkVec(p0,i=>0.22+0.26*frac(i*0.414214+0.3),i=>ALPHA(i)+TH5[i]*R2D);
const KV={dog:nounK([1.3,30]),man:nounK([1.2,40]),bites:mkVec([0.9,150],i=>0.22+0.26*frac(i*0.3183+0.45),i=>(i*151.7+75)%360)};
const P0={q:{bites:[1.5,-47],dog:[1.0,80],man:[1.1,110]},k:{dog:[1.3,30],man:[1.2,40],bites:[0.9,150]}};
const WORD={dog:{en:'dog',zh:'狗'},bites:{en:'bites',zh:'咬'},man:{en:'man',zh:'人'}};
const ORD={a:['dog','bites','man'],b:['man','bites','dog']};                 // positions 1, 2, 3
const turnXY=(x,y,a)=>{const c=Math.cos(a),s=Math.sin(a); return [x*c-y*s,x*s+y*c];};
/* one pair's share of the score: the query (word at position m) and the key (word at position n), each turned by its position */
function share(qw,kw,i,m,n,rope){const q=QV[qw], k=KV[kw]; let [x,y]=[q[2*i],q[2*i+1]], [u,v]=[k[2*i],k[2*i+1]];
  if(rope){[x,y]=turnXY(x,y,m*TH5[i]); [u,v]=turnXY(u,v,n*TH5[i]);} return x*u+y*v;}
/* the head's score q·k: the 64 pairs' shares added up */
function score(qw,kw,m,n,rope){let s=0; for(let i=0;i<NP;i++) s+=share(qw,kw,i,m,n,rope); return s;}
function grid(ord,rope){return ORD[ord].map((qw,a)=>ORD[ord].map((kw,b)=>score(qw,kw,a+1,b+1,rope)));}
const wrapDeg=d=>{let x=((d+180)%360+360)%360-180; if(x===-180) x=180; return x;};   // an angle in (−180°, 180°]
/* pair 0 of the query of "bites" at m and of the key of "dog" at n: both directions, the angle between, the share, the head's score */
function pair0(m,n,rope){const qa=P0.q.bites[1]+(rope?m*TH0D:0), ka=P0.k.dog[1]+(rope?n*TH0D:0);
  return {m,n,rope,qa,ka,between:wrapDeg(qa-ka),share:share('bites','dog',0,m,n,rope),head:score('bites','dog',m,n,rope)};}
/* turns over a text of L tokens for every pair; with a training length T (and the training speeds), the turns met in training */
function turnsData(th,Lx,thT,T){
  const rows=th.map((f,i)=>{const t=Lx*f/TAU, tt=T!=null?T*thT[i]/TAU:null;
    return {i,f,t,tt,lam:TAU/f,under:t<1,beyond:tt!=null&&tt<1&&t>tt*(1+1e-9)};});
  const nU=rows.filter(r=>r.under).length, nB=rows.filter(r=>r.beyond).length, s=rows[NP-1];
  return {rows,nUnder:nU,nWrapped:NP-nU,nBeyond:nB,slow:s,ratio:s.tt?s.t/s.tt:null,first:rows.findIndex(r=>r.under),Lx,T};}
const MODELS={
 l2:{id:'l2',name:M2,base:10000,sc:false,train:4096},
 l3:{id:'l3',name:M3,base:500000,sc:false,train:8192},
 l31:{id:'l31',name:M31,base:500000,sc:true,train:131072},
 cl:{id:'cl',name:MCL,base:1000000,sc:false,train:16384}};
/* guided parts' settings */
const SHIFT_MAX=100, L5=[4096,8192,16384,32768,65536,131072,262144], L6=[8192,16384,32768,65536,131072,262144], B5=[10000,500000];
const p5Data=()=>turnsData(thetas(base5,null),L5[l5],null,null);
const p6Data=()=>turnsData(thOf(500000,sc6),L6[l6],thetas(500000,null),8192);
/* scenarios: the settings */
const SB_BASES=[10000,100000,500000,1000000], CTX=[2048,4096,8192,16384,32768,65536,131072,262144], TRAIN=[4096,8192,16384,131072];
const sbData=s=>turnsData(thOf(s.base,s.sc),s.ctx,thOf(s.base,s.sc),s.train);
/* the challenge: a model with 128-number heads, texts of 1M tokens; the smallest base whose slowest pair stays under one full turn */
const CH=Object.freeze({L:1048576,bases:Object.freeze([10000,100000,500000,1000000]),start:10000});
const chTurns=b=>CH.L*thetas(b,null)[NP-1]/TAU;
const chLam=b=>TAU/thetas(b,null)[NP-1];
const chOK=b=>chTurns(b)<1;
const chBest=()=>CH.bases.find(chOK);
const chData=b=>turnsData(thetas(b,null),CH.L,null,null);

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sc2=v=>(v<0&&fixd(Math.abs(v),2)!=='0.00'?MINUS:'')+fixd(Math.abs(v),2);   // a score: 0.44, −1.36
const n3=v=>(v<0&&fixd(Math.abs(v),3)!=='0.000'?MINUS:'')+fixd(Math.abs(v),3);   // one number of a vector: −1.097
const degS=a=>fixd(a*R2D,1)+'°';                                // radians → “57.3°”
const dS=d=>(d<0?MINUS:'')+fixd(Math.abs(d),1)+'°';             // degrees → “−19.7°”
const tokK=n=>n>=1048576&&n%1048576===0?(n/1048576)+'M':(n>=1024&&n%1024===0?(n/1024)+'K':fint(n));
const bigT=n=>n>999999?A.fmt.big(n):fint(Math.round(n));
const turnsS=t=>tx(`${fixd(t,2)} turns`,`${fixd(t,2)} 圈`);
/* turns as a number: 2 decimals from 0.1 up, else 3 significant digits (0.0512, 0.00320), the same on cards, tooltips and in the math (R6.10) */
const tS=t=>t>=0.1?fixd(t,2):t.toPrecision(3);
/* turns as a person reads them: under one turn in degrees, else in turns (3 significant digits past 1,000) */
const turnS=t=>t<1?fixd(t*360,1)+'°':(t<1000?turnsS(t):tx(`${bigT(t)} turns`,`${bigT(t)} 圈`));
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
const expT=(v,d)=>{if(v>=0.01) return fixd(v,d<3?3:d); const [m,e]=v.toExponential(d).split('e'); return `${m} \\times 10^{${+e}}`;};   // the same in TeX, for A.ui.math
const expS=(v,d)=>{if(v>=0.01) return fixd(v,d<3?3:d); const [m,e]=v.toExponential(d).split('e'); return `${m} × 10${String(+e).split('').map(c=>SUP[c]||c).join('')}`;};
const tokN=n=>tx(n===1?'1 token':`${tokK(n)} tokens`,`${tokK(n)} 个 token`);
const tokW=n=>tx(n===1?'1 token':`${fint(n)} tokens`,`${fint(n)} 个 token`);   // exact, for tooltips
const posN=n=>tx(n===1?'1 position':`${fint(n)} positions`,`${fint(n)} 个位置`);
const w=k=>L(WORD[k]);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<span class="linkbtn rp-lk" role="link" tabindex="0" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</span>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
/* a link to another guided part (R1.10); its hover says where it goes */
const PL=(i,t)=>J('p'+i,t);
const ONE_M=()=>tx('Token lengths on this site count K as 1,024, so 1M tokens are 1,024 × 1,024.','本站的 token 长度按 K = 1,024 算，所以 1M 个 token 是 1,024 × 1,024 个。');

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let mode='guided', step=0, pred=null, observed={}, root=null, app=null, annT=null, userAct=false, wantObs=false, sbTouched=false;
let headView='128', rope3=false, shift=0, l5=1, base5=500000, l6=0, sc6=false, p5Last=null, p5Prev={l5:1,base5:500000};
const seenView=new Set(['128']);
let SB={base:500000,sc:true,ctx:131072,train:131072};
let chBase=CH.start, chSubmitted=false, chReveal=false;
const chTried=new Set([CH.start]);
let drillSeed=3, drillOpen=false;
const shownPrev={}, shownRaw={};

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 rope:()=>tx('RoPE (rotary position embedding) turns each query and key, not the value, by an angle set by the token’s position, inside every attention layer.',
             'RoPE（旋转位置编码）在每一层的注意力里，按 token 的位置把 query 和 key 转一个角度（value 不转）。'),
 pairing:()=>tx('Which two numbers form a pair is a fixed choice. Meta’s Llama code and the RoFormer paper pair neighbours, numbers 2i and 2i + 1, as drawn here. Hugging Face’s Llama code pairs number j with number j + 64 (rotate_half) and reorders the rows of Wq and Wk when it converts Meta’s weights, so the scores come out the same. The model learns Wq and Wk around whichever pairing it is trained with.',
                '哪两个数配成一对，是事先定好的。Meta 的 Llama 代码和 RoFormer 论文把相邻的第 2i 维和第 2i + 1 维配对，这里就这样画。Hugging Face 的 Llama 代码把第 j 维和第 j + 64 维配对（rotate_half），转换 Meta 的权重时会重排 Wq 和 Wk 的行，所以得分一样。模型训练时用哪种配对，Wq 和 Wk 就围绕它学。'),
 score:()=>tx(`The score of a query and a key is their dot product ${MX('q \\cdot k')}: multiply the two vectors number by number and add up the 128 products, as in multi-head attention. The model then divides every score by ${MX('\\sqrt{128}')}, which does not change which key scores highest.`,
              `query 和 key 的得分是它们的点积 ${MX('q \\cdot k')}：两个向量逐维相乘，再把 128 个乘积加起来，和多头注意力里一样。模型之后会把每个得分都除以 ${MX('\\sqrt{128}')}，这不会改变哪个 key 得分最高。`),
 shadow:()=>tx('The dot product of two arrows is the length of one arrow’s shadow on the other times the other’s length: |q| × |k| × cos(the angle between them). Arrows pointing the same way give a large positive share, opposite ways a negative one.',
               '两个箭头的点积，等于一个箭头投在另一个方向上的影子长度，乘以另一个箭头的长度：|q| × |k| × cos(夹角)。方向一致时得分大且为正，方向相反时为负。'),
 radian:()=>tx(`A radian is the angle whose arc is as long as the radius: ${MX('\\frac{180°}{\\pi} \\approx 57.3°')}. One full turn is ${MX('2\\pi \\approx 6.28')} radians.`,
               `1 弧度是弧长等于半径时的角度：${MX('\\frac{180°}{\\pi} \\approx 57.3°')}。一整圈是 ${MX('2\\pi \\approx 6.28')} 弧度。`),
 base:()=>tx('The RoPE base, rope_theta in config.json. Each pair turns a little slower than the one before, and the base sets how slow the last pair is: Llama 2 uses 10,000, Llama 3 and 3.1 use 500,000, Code Llama 1,000,000. Pair i turns θᵢ radians per token:',
             'RoPE base，config.json 里叫 rope_theta。每一对都比前一对转得慢一点，base 决定最后一对有多慢：Llama 2 用 10,000，Llama 3 和 3.1 用 500,000，Code Llama 用 1,000,000。第 i 对每个 token 转 θᵢ 弧度：'),
 seen:()=>tx('The angles a pair met in training go from 0 up to where it gets at the training length. A pair that wrapped around in training has met every angle.',
             '一对数在训练时见过的角度，从 0 到它在训练长度处转到的角度。训练时就转过一整圈的数对，每个角度都见过。'),
 head:()=>kt('head')+tx(' Llama‑3‑8B has 32 query heads per layer, each with 128 numbers per query and key.',' Llama‑3‑8B 每层有 32 个 query head，每个 query 和 key 有 128 维。'),
 emb:()=>kt('embedding'),
 absPos:()=>tx('The original Transformer (Vaswani et al., 2017) adds a fixed sine-and-cosine vector for each position to the input embeddings; GPT‑2 and BERT learn one vector per position and add it the same way. After that the position lives inside the numbers of every token’s vector.',
               '最早的 Transformer（Vaswani 等，2017）给每个位置一个固定的正弦、余弦向量，加到输入的 embedding 上；GPT‑2 和 BERT 为每个位置学一个向量，也这样加上去。之后，位置就混在每个 token 的向量的数里。'),
 theta:()=>tx('θᵢ is how far pair i turns per position: 57.3° (1 radian) for pair 0, and a little less for each later pair (part 5).',
              'θᵢ 是第 i 对每个位置转过的角度：第 0 对是 57.3°（1 弧度），后面的一对比一对小一点（第 5 部分）。'),
 mn:()=>tx('m is the position of the query (“bites”), n the position of the key (“dog”); m − n is their distance.',
           'm 是 query（“咬”）的位置，n 是 key（“狗”）的位置；m − n 是它们的距离。'),
 head1:()=>tx('In this lab the head is made up to look for the token just before “bites”. A real head learns what to look for; RoPE is what lets a head see “just before” at all.',
              '这个 Lab 里的 head 是虚构的，设成去找“咬”前面的那个 token。真实的 head 要自己学会找什么；RoPE 让 head 能看出“前面一个”。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'same',t:{en:'It stays exactly the same',zh:'完全不变'}},
  {v:'little',t:{en:'It changes a little',zh:'变一点'}},
  {v:'lot',t:{en:'It changes a lot',zh:'变很多'}}];
const PRED_RIGHT='same';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const HV=[['128',{en:'128 numbers',zh:'128 个数值'}],['pairs',{en:'64 pairs',zh:'64 对'}],['arrows',{en:'64 arrows',zh:'64 个箭头'}]];
const headCtl=()=>A.ui.seg({id:'rp-ic-hv',key:'hv',label:tx('Show the query of “bites” as','把“咬”的 query 画成'),options:HV.map(([v,t])=>[v,L(t)]),value:headView,inline:true});
const ropeCtl=()=>A.ui.seg({id:'rp-ic-rope',key:'rope3',label:tx('RoPE','RoPE'),options:[['off',tx('off','关')],['on',tx('on','开')]],value:rope3?'on':'off',inline:true});
const shiftCtl=()=>A.ui.range({id:'rp-ic-shift',key:'shift',label:tx('Both tokens moved later by','两个 token 一起往后挪'),min:0,max:SHIFT_MAX,value:shift,out:posN(shift),inline:true});
const l5Ctl=()=>A.ui.range({id:'rp-ic-l5',key:'l5',label:tx('Text length L','文本长度 L'),min:0,max:L5.length-1,value:l5,out:tokN(L5[l5]),inline:true});
const b5Ctl=()=>A.ui.seg({id:'rp-ic-b5',key:'b5',label:tx('RoPE base','RoPE base'),options:B5.map(b=>[b,tx(`${fint(b)} (Llama${NBS}${b===10000?'2':'3'})`,`${fint(b)}（Llama${NBS}${b===10000?'2':'3'}）`)]),value:base5,inline:true});
const l6Ctl=()=>A.ui.range({id:'rp-ic-l6',key:'l6',label:tx('Text length L','文本长度 L'),min:0,max:L6.length-1,value:l6,out:tokN(L6[l6]),inline:true});
const s6Ctl=()=>A.ui.seg({id:'rp-ic-s6',key:'s6',label:tx('Slow pairs','慢的数对'),options:[['0',tx('as trained','和训练时一样')],['1',tx(`8 times slower (Llama${NBH}3.1’s scaling)`,`慢 8 倍（Llama${NBH}3.1 的缩放）`)]],value:sc6?'1':'0',inline:true});
const STEPS=[
 {id:'mission',view:'order',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('“Dog bites man” or “man bites dog”: without positions, attention gives both orders the same scores.','“狗咬人”还是“人咬狗”：没有位置信息时，注意力给两者的得分一样。'),
  body:()=>tx(`<p>${M3} reads “dog bites man”. In each ${DEF('head',TIP.head())}, the query of “bites” meets the key of every token, and each meeting gives a ${DEF('score',TIP.score())}, ${MX('q \\cdot k')}, as in the lab ${labLink('mha')}.</p><p>But the ${DEF('embedding table',TIP.emb())} gives “dog” the same row wherever it stands. So without position information, the row of “bites” scores each word the same in “dog bites man” and in “man bites dog” (${J('stage','the two grids in the chart')}): its highest score goes to “dog” both times, and the head cannot tell who bit whom.</p><p>The next parts open one head and show how RoPE puts the positions into every score.</p>`,
              `<p>${M3} 读到“狗咬人”。在每个 ${DEF('head',TIP.head())} 里，“咬”的 query 和每个 token 的 key 相遇，每次相遇得到一个${DEF('得分',TIP.score())} ${MX('q \\cdot k')}，和${labLink('mha')}这个 Lab 里一样。</p><p>可是不管“狗”站在哪里，${DEF('embedding 表',TIP.emb())}给它的都是同一行。所以没有位置信息时，“咬”那一行给每个词的得分，在“狗咬人”和“人咬狗”里都一样（见${J('stage','图里的两个表')}）：最高分两次都给了“狗”，这个 head 分不出谁咬了谁。</p><p>接下来几部分打开一个 head，看 RoPE 怎样把位置放进每个得分。</p>`)},
 {id:'head',view:'head',title:{en:'One head: 128 numbers, 64 pairs',zh:'一个 head：128 维，64 对'},focusCtl:'#rp-ic-hv',
  lead:()=>tx('RoPE groups a head’s 128 numbers into 64 pairs and turns each pair; it adds nothing to the embedding.','RoPE 把 head 的 128 维两两配成 64 对，逐对转动；它不往 embedding 上加东西。'),
  enter:()=>{}, expect:()=>seenView.has('arrows'), showMe:()=>setHV('arrows'), always:true, label:{en:'Observed',zh:'观察到'},
  why:()=>tx('Click “64 arrows” to go on.','点“64 个箭头”，才能进入下一部分。'),
  body:()=>tx(`<p>In one head of ${M3}, the query of “bites” is a vector of 128 numbers, and so is each key. RoPE groups them into ${DEF('64 pairs of neighbours',TIP.pairing())}: a turn needs a plane, and two numbers are the x and y of an arrow. Any fixed pairing works, because the model learns Wq and Wk with it.</p><p>The score ${MX('q \\cdot k')} adds up the products of the 128 numbers, so it is also the sum of the 64 pairs’ own dot products: turning a pair changes only that pair’s share. Keys are split the same way; the value is never turned.</p>`,
              `<p>在 ${M3} 的一个 head 里，“咬”的 query 是一个 128 维的向量，每个 key 也是。RoPE 把这些数${DEF('按相邻两个配成 64 对',TIP.pairing())}：转动需要一个平面，两个数正好是一个箭头的 x 和 y。怎么配对都行，只要固定下来，因为模型的 Wq 和 Wk 就是按这种配对学出来的。</p><p>得分 ${MX('q \\cdot k')} 是把 128 维逐维相乘再相加，所以它也等于 64 对各自点积的总和：转动一对，只改变这一对贡献的那一份。key 也这样配对；value 从来不转。</p>`)+headCtl()+
    tx(`<p>You may know the older way (${J('cmp','the comparison in the chart')}): the original Transformer, GPT‑2 and BERT ${DEF('add a position vector',TIP.absPos())} to each token’s embedding, once, at the input. RoPE adds nothing there; it turns q and k inside every layer’s attention, right before the dot product. ${PL(4,'Part 4')} shows what this buys when both tokens move. The cost is one small step in every layer: each key is turned once, when it is written to the ${labLink('kv-basics','KV cache')}.</p>`,
       `<p>你可能学过另一种做法（见${J('cmp','图里的对比')}）：最早的 Transformer、GPT‑2 和 BERT 在输入处${DEF('给每个 token 的 embedding 加一个位置向量',TIP.absPos())}，只加一次。RoPE 不在那里加任何东西；它在每一层的注意力里，在点积之前转动 q 和 k。两个 token 一起挪动时这样做有什么好处，${PL(4,'第 4 部分')}会看到。代价是每一层多一个小步骤：每个 key 在写进 ${labLink('kv-basics','KV cache')} 时转一次。</p>`),
  obs:()=>obsHead()},
 {id:'rope',view:'pair2',title:{en:'RoPE on “dog bites man”',zh:'“狗咬人”上的 RoPE'},focusCtl:'#rp-ic-rope',
  lead:()=>tx('Each arrow turns by its position × 57.3°; the angle between query and key then changes with the order.','每个箭头转过“位置 × 57.3°”；query 和 key 的夹角于是随顺序变化。'),
  enter:()=>{rope3=false;}, expect:()=>rope3, showMe:()=>{rope3=true;}, always:true, label:{en:'Result',zh:'结果'},
  why:()=>tx('Switch RoPE on to go on.','打开 RoPE，才能进入下一部分。'),
  body:()=>tx(`<p>Follow pair 0 of the query of “bites” and of the key of “dog”, the fastest pair of every Llama head: it turns θ = 57.3° (1 ${DEF('radian',TIP.radian())}) per position. RoPE turns each arrow by its token’s position × θ; the arcs outside the dials show how far.</p><p>In “dog bites man” the key ends up θ behind the query; in “man bites dog”, where “dog” sits at position 3, θ ahead. So the pair’s dot product, the ${DEF('shadow',TIP.shadow())} of one arrow on the other, changes. Switch RoPE on:</p>`,
              `<p>跟着“咬”的 query 和“狗”的 key 的第 0 对看，它是每个 Llama head 里转得最快的一对：每个位置转 θ = 57.3°（1 ${DEF('弧度',TIP.radian())}）。RoPE 把每个箭头转过“它的位置 × θ”，表盘外的弧线画出转了多少。</p><p>在“狗咬人”里，key 最后比 query 少转 θ；在“人咬狗”里，“狗”在位置 3，key 比 query 多转 θ。所以这一对的点积（一个箭头投在另一个上的${DEF('影子',TIP.shadow())}）变了。打开 RoPE：</p>`)+ropeCtl(),
  obs:()=>obsRope()},
 {id:'shift',view:'shift',title:{en:'Move both tokens 100 positions later',zh:'两个 token 一起往后挪 100 个位置'},focusCtl:'#rp-ic-shift',
  lead:()=>tx('With RoPE on, move “dog bites man” 100 positions later: does the score change?','RoPE 开着，把“狗咬人”往后挪 100 个位置：得分会变吗？'),
  enter:()=>{shift=0;}, gate:()=>!!pred, expect:()=>shift!==0, showMe:()=>{shift=100;}, label:{en:'Result',zh:'结果'},
  why:()=>pred?tx('Move both tokens to go on.','把两个 token 一起挪动，才能进入下一部分。'):tx('Pick an answer to go on.','选一个答案，才能进入下一部分。'),
  body:()=>shiftBody(),
  obs:()=>obsShift()},
 {id:'speeds',view:'turns',title:{en:'Fast pairs, slow pairs and the base',zh:'快的数对、慢的数对和 base'},focusCtl:'#rp-ic-l5',
  lead:()=>tx('The base sets each pair’s speed; the text length sets how many turns it makes.','base 决定每一对的转速，文本长度决定它转几圈。'),
  enter:()=>{l5=1; base5=500000; p5Last=null; p5Prev={l5:1,base5:500000};}, expect:()=>l5!==1||base5!==500000, showMe:()=>{p5Prev={l5,base5}; l5=L5.indexOf(131072); base5=10000; p5Last='both';}, label:{en:'Observed',zh:'观察到'},
  why:()=>tx('Drag the text length or switch the base to go on.','拖动文本长度或换一个 base，才能进入下一部分。'),
  body:()=>tx(`<p>In “dog bites man”, “dog” is one token back, and the fast pair 0 tells the order. Were it 5,000 tokens back, pair 0 would have wrapped around about 800 times and could no longer pin that distance down. Slower pairs carry such distances: each pair turns a little slower than the one before, and the ${T_('baseT','base')} sets how slow the last one is.</p><p>The chart shows how many full turns each pair makes over the whole text. Under the line, a pair gives every distance in the text its own angle; above it, the pair wraps around. Drag the text length, then switch the base.</p>`,
              `<p>在“狗咬人”里，“狗”只在前面一个 token，快的第 0 对就分得清顺序。如果它在前面 5,000 个 token，第 0 对已经转过约 800 圈，单靠它就定不准这个距离。这样的距离要靠慢的数对：每一对都比前一对转得慢一点，${T_('baseT','base')} 决定最后一对有多慢。</p><p>图里画出每一对在整段文本上转过几整圈。在线下面，文本里的每个距离都给这一对一个不同的角度；在线上面，这一对会转过一整圈。拖动文本长度，再换一个 base。</p>`)+l5Ctl()+b5Ctl(),
  obs:()=>obsSpeeds()},
 {id:'train',view:'turns',title:{en:'Longer than the training texts',zh:'比训练文本更长'},focusCtl:'#rp-ic-l6',
  lead:()=>tx('Past the training length, the slow pairs reach angles the model never met.','超过训练长度，慢的数对会转到模型从没见过的角度。'),
  enter:()=>{l6=0; sc6=false;}, expect:()=>l6!==0, showMe:()=>{l6=L6.indexOf(131072);}, always:true, label:{en:'Result',zh:'结果'},
  why:()=>tx('Drag the text length up to go on.','把文本长度往上拖，才能进入下一部分。'),
  body:()=>tx(`<p>${M3} was trained on texts of up to 8K tokens. If a question about the dog came 100,000 tokens after “dog bites man”, the slow pairs would have to turn much further than they ever did in training. The ${DEF('hollow marks',TIP.seen())} show how far each pair turned there: pairs above the line wrapped around in training and met every angle; pairs below it only met the angles up to their mark.</p><p>Run it on longer texts, then try Llama‑3.1’s scaling:</p>`,
              `<p>${M3} 训练时的文本最长 8K 个 token。如果一个关于这只狗的问题出现在“狗咬人”之后 100,000 个 token，慢的数对就得转到比训练时远得多的角度。${DEF('空心的记号',TIP.seen())}标出每一对在那个长度上转到哪里：线上面的数对训练时就转过一整圈，每个角度都见过；线下面的只见过它的记号以下的角度。</p><p>让它读更长的文本，再试试 Llama‑3.1 的缩放：</p>`)+l6Ctl()+s6Ctl(),
  obs:()=>obsTrain()},
 {id:'remember',view:'order',rope:true,title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('What turns, what moving both tokens showed, and why the base matters for long texts.','什么在转，一起挪动两个 token 说明了什么，以及 base 为什么对长文本重要。'),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const cur=()=>STEPS[step];
function viewNow(){if(mode==='sandbox'||mode==='challenge') return 'turns'; return cur().view;}
const ropeOn=()=>mode==='guided'&&!!cur().rope;
function setHV(v){headView=v; seenView.add(v);}

/* part 2 */
function obsHead(){
  const q=QV.bites;
  if(headView==='128') return tx(`<p><b>The query of “bites” in this head: 128 numbers, from ${n3(q[0])} and ${n3(q[1])} at the start.</b> Group them into pairs.</p>`,`<p><b>这个 head 里“咬”的 query：128 个数值，开头是 ${n3(q[0])} 和 ${n3(q[1])}。</b>把它们两两配对。</p>`);
  if(headView==='pairs') return tx(`<p><b>Each bracket joins numbers 2i and 2i + 1: pair 0 is numbers 0 and 1, (${n3(q[0])}, ${n3(q[1])}); pair 63 is numbers 126 and 127.</b> Now draw each pair as an arrow.</p>`,`<p><b>每个括号把第 2i 和第 2i + 1 个数值连成一对：第 0 对是第 0、1 个数值（${n3(q[0])}，${n3(q[1])}）；第 63 对是第 126、127 个数值。</b>现在把每一对画成一个箭头。</p>`);
  return tx(`<p><b>Each pair is an arrow: its first number is x, its second y.</b> Pair 0 of “bites” is (${n3(q[0])}, ${n3(q[1])}): an arrow of length 1.5 pointing at −47°. RoPE turns every such arrow of q and k by an angle; the arrow’s length, its content, stays.</p>`,
            `<p><b>每一对是一个箭头：第一个数值是 x，第二个是 y。</b>“咬”的第 0 对是（${n3(q[0])}，${n3(q[1])}）：一个长 1.5、指向 −47° 的箭头。RoPE 把 q 和 k 的每个这样的箭头转一个角度；箭头的长度（也就是内容）不变。</p>`);
}
/* the row of "bites" in one sentence: the key with the highest score */
const topOf=(ord,rope)=>{const g=grid(ord,rope)[1], j=g.indexOf(Math.max(...g)); return {w:ORD[ord][j],v:g[j],j};};
/* part 3 */
function obsRope(){
  const a=pair0(2,1,rope3), b=pair0(2,3,rope3);
  if(!rope3) return tx(`<p><b>RoPE off: both sentences give the same arrows, the same shadow and the same score, ${sc2(a.head)}.</b> Switch RoPE on.</p>`,`<p><b>RoPE 关着：两句话里箭头一样、影子一样，得分也一样，都是 ${sc2(a.head)}。</b>打开 RoPE。</p>`);
  const ta=topOf('a',true), tb=topOf('b',true), t0=topOf('a',false);
  return tx(`<p><b>With RoPE on, the row of “bites” gives its highest score to the token just before “bites”: “${w(ta.w)}” (${sc2(ta.v)}) in “dog bites man”, “${w(tb.w)}” (${sc2(tb.v)}) in “man bites dog”.</b> Without RoPE it went to “${w(t0.w)}” (${sc2(t0.v)}) both times. Softmax then gives that token the largest attention weight.</p>`+
            `<p>What turns: the query and the key, every pair of their numbers. By how much: the token’s position × the pair’s speed (the arcs). What for: the angle between them now changes with the order, so the head can tell who bit whom.</p><p class="small muted">${esc(TIP.head1())}</p>`,
            `<p><b>RoPE 开着时，“咬”那一行的最高分给了“咬”前面的那个 token：“狗咬人”里是“${w(ta.w)}”（${sc2(ta.v)}），“人咬狗”里是“${w(tb.w)}”（${sc2(tb.v)}）。</b>没有 RoPE 时，两次都给了“${w(t0.w)}”（${sc2(t0.v)}）。softmax 于是给这个 token 最大的注意力权重。</p>`+
            `<p>什么在转：query 和 key，它们的每一对数。转多少：token 的位置 × 这一对的转速（见弧线）。有什么用：两者的夹角现在随顺序变化，所以这个 head 分得出谁咬了谁。</p><p class="small muted">${esc(TIP.head1())}</p>`);
}
/* part 4: the prediction first; the slider and the stage's second dial appear after it (R3.8) */
function shiftBody(){
  const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
  return tx(`<p>RoPE stays on. Put 100 tokens of other text in front of “dog bites man”: “bites” moves from position 2 to 102 and “dog” from 1 to 101; their content, the vectors before the turn, is as before. What happens to the head’s score of bites → dog?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it after you move the tokens.</p>':''}`,
            `<p>RoPE 一直开着。在“狗咬人”前面插入 100 个 token 的别的文字：“咬”从位置 2 挪到 102，“狗”从 1 挪到 101；它们的内容（转动之前的向量）和原来一样。这个 head 里“咬 → 狗”的得分会怎样？</p>${ch}${pred?'<p class="muted small">已记下。挪动之后，也还可以改。</p>':''}`)+(pred?shiftCtl():'');
}
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsShift(){
  if(!pred) return '';
  const a=pair0(2,1,true), b=pair0(2+shift,1+shift,true), mn=DEF('m − n',TIP.mn());
  if(!shift) return tx(`<p><b>As written, “bites” at 2 and “dog” at 1: pair 0’s arrows are ${dS(Math.abs(a.between))} apart, and the head’s score is ${sc2(a.head)}.</b> Move both tokens later.</p>`,
                       `<p><b>原文里“咬”在 2、“狗”在 1：第 0 对的两个箭头相差 ${dS(Math.abs(a.between))}，这个 head 的得分是 ${sc2(a.head)}。</b>把两个 token 一起往后挪。</p>`);
  return tx(`<p><b>Both moved ${posN(shift)} later: every arrow turned much further, yet the angle between query and key stayed ${dS(Math.abs(b.between))}, so the score stays ${sc2(b.head)}.</b></p>`,`<p><b>两个 token 都往后挪了 ${posN(shift)}：每个箭头都多转了很多，但 query 和 key 的夹角还是 ${dS(Math.abs(b.between))}，所以得分还是 ${sc2(b.head)}。</b></p>`)+verdict()+
    tx(`<p>Why: RoPE turns the query and the key by the same extra angle, and the score sees only the angle between them. With the content unchanged, the effect of position depends only on the distance ${mn}: the head sees “dog is one before bites” wherever the sentence sits. The curve shows this head’s score at every distance; moving both tokens never moves you along it.</p><p>With a position vector added at the input instead, the moved tokens would get different vectors, and their score would change. This is one reason models switched to RoPE.</p>`,
       `<p>原因：RoPE 让 query 和 key 多转同样的角度，而得分只看它们之间的夹角。内容不变时，位置的影响只取决于距离 ${mn}：句子放在哪里，这个 head 看到的都是“狗在咬前面一个位置”。曲线画出这个 head 在每个距离上的得分；两个 token 一起挪动，不会让你在曲线上移动。</p><p>如果改成在输入处加位置向量，挪动之后两个 token 的向量就变了，得分也会变。这是模型改用 RoPE 的原因之一。</p>`);
}
/* part 5: what the learner changed, compared with the start, and the reason for that one change (R1.9) */
function obsSpeeds(){
  const d=p5Data(), Lx=L5[l5], d0=turnsData(thetas(500000,null),8192,null,null), dL=l5!==1, dB=base5!==500000;
  const what=tx(`${dL?`Text ${tokK(8192)} → ${tokK(Lx)}`:`Text ${tokK(Lx)}`}, ${dB?`base ${fint(500000)} → ${fint(base5)}`:`base ${fint(base5)}`}`,`文本 ${dL?`${tokK(8192)} → ${tokK(Lx)}`:tokK(Lx)}，base ${dB?`${fint(500000)} → ${fint(base5)}`:fint(base5)}`);
  const head=tx(`<p><b>${what}: ${d0.nUnder} → ${d.nUnder} of the 64 pairs under one turn.</b></p>`,`<p><b>${what}：不满一圈的数对 ${d0.nUnder} → ${d.nUnder} 对。</b></p>`);
  if(!p5Last) return head;
  const up=p5Last==='l5'?l5>p5Prev.l5:null, bigger=p5Last==='b5'?base5>p5Prev.base5:null;
  const whyL=up===null&&p5Last!=='both'?'':(up===false?tx('The whole line moved down: a shorter text lowers every dot by the same factor, so more pairs stay under one turn.','整条线往下移：文本变短，每个点都按同样的倍数降低，于是更多数对保持不满一圈。')
    :tx('The whole line moved up: a longer text lifts every dot by the same factor, so more pairs wrap around.','整条线往上移：文本变长，每个点都按同样的倍数升高，于是更多数对转过一整圈。'));
  const whyB=bigger===null&&p5Last!=='both'?'':(bigger?tx('The line swung down about pair 0: a larger base slows the slow pairs, so more of them stay under one turn. Llama 3 raised the base from Llama 2’s 10,000 to 500,000; its paper says the larger base better supports longer contexts.','线绕着第 0 对往下转：base 变大，慢的数对更慢，于是更多数对保持不满一圈。Llama 3 把 base 从 Llama 2 的 10,000 提高到 500,000；它的论文说，更大的 base 能更好地支持长上下文。')
    :tx('The line swung up about pair 0: a smaller base speeds up the slow pairs, so more of them wrap around. Llama 2 used 10,000; Llama 3 raised it to 500,000, and its paper says the larger base better supports longer contexts.','线绕着第 0 对往上转：base 变小，慢的数对变快，于是更多数对转过一整圈。Llama 2 用 10,000；Llama 3 把它提高到 500,000，它的论文说，更大的 base 能更好地支持长上下文。'));
  return head+(p5Last==='both'?`<p>${whyL}</p><p>${whyB}</p>`:`<p>${p5Last==='l5'?whyL:whyB}</p>`);
}
/* part 6 */
function obsTrain(){
  const d=p6Data(), Lx=L6[l6], k=d.ratio?fint(Math.round(d.ratio)):'1', both=d.rows.filter(r=>r.under&&r.beyond).length;
  if(!d.nBeyond&&sc6&&l6>0) return tx(`<p><b>At ${tokN(Lx)}, with the scaling, every pair stays within the angles it met in training.</b> A slow pair now needs 8 times as many tokens to turn as far, so up to 64K tokens it stays in range. Drag the text length further.</p>`,`<p><b>${tokN(Lx)}时，用了缩放，每一对都停在训练时见过的角度以内。</b>慢的数对现在要多 8 倍的 token 才转到同样的角度，所以到 64K 个 token 为止都不超出。把文本长度再往上拖。</p>`);
  if(!d.nBeyond) return tx(`<p><b>At ${tokN(Lx)}, every pair stays within the angles it met in training.</b> Drag the text length up.</p>`,`<p><b>${tokN(Lx)}时，每一对都停在训练时见过的角度以内。</b>把文本长度往上拖。</p>`);
  if(sc6) return tx(`<p><b>Still ${d.nBeyond} pairs past the angles met in 8K training, but the slowest turns only ${k} times as far instead of ${fint(Math.round(Lx/8192))}.</b></p><p>Why: the fast pairs had already met every angle, so slowing them would only crowd neighbours together; the 6 pairs in between (29 to 34) are slowed partly, 1.2 to 5.3 times. This is ${M3} with Llama‑3.1’s scaling, before any long-context training. ${M31} was then trained on long texts: see ${J('sc3','the scenario “Llama‑3.1‑8B, a 128K-token text”')}.</p>`,
                     `<p><b>仍有 ${d.nBeyond} 对超出 8K 训练时见过的角度，但最慢的一对只转到训练时的 ${k} 倍，而不是 ${fint(Math.round(Lx/8192))} 倍。</b></p><p>原因：快的数对训练时就见过每个角度，把它们也放慢，只会让相邻的 token 挤在一起；中间的 6 对（第 29 到 34 对）只放慢一部分，1.2 到 5.3 倍。这是用了 Llama‑3.1 缩放、但还没做长文本训练的 ${M3}。${M31} 之后又在长文本上训练过：见${J('sc3','应用场景“Llama‑3.1‑8B 读 128K 个 token 的文本”')}。</p>`);
  return tx(`<p><b>At ${tokN(Lx)}, ${d.nBeyond} of the 64 pairs turn past every angle they met in training, the slowest ${k} times as far.</b></p><p>Why: the model learned what those angles mean only up to 8K tokens, so past it its attention can break down. Staying under one turn is not enough: ${both} of these pairs still stay under one turn, yet reach angles never met in training.</p><p>${M31}’s config.json slows those pairs 8 times (the “llama3” scaling in Meta’s code), and the Llama 3 paper reports long-context training up to 128K tokens. Try the scaling.</p>`,
            `<p><b>${tokN(Lx)}时，64 对里有 ${d.nBeyond} 对转过了训练时见过的所有角度，最慢的一对转到训练时的 ${k} 倍。</b></p><p>原因：模型只在 8K 个 token 以内学过这些角度的含义，超过这个长度，它的注意力可能失灵。不满一圈还不够：这些数对里有 ${both} 对仍然不满一圈，却转到了训练时没见过的角度。</p><p>${M31} 的 config.json 把这些数对的转速降到 1/8（Meta 代码里的“llama3”缩放），Llama 3 的论文报告了最长 128K 个 token 的长文本训练。试试这个缩放。</p>`);
}
function remember(){
  return tx(`<ul><li><b>What turns, and where.</b> RoPE turns every pair of numbers of q and k by position × the pair’s speed, inside every layer’s attention. The embedding and v are not touched. So position changes only the scores, never the content each token hands over.</li>`+
    `<li><b>The order now matters.</b> With RoPE the row of “bites” favours the token just before it (${J('stage','the grids')}): “dog” in one sentence, “man” in the other, because “dog” is one before “bites” in the first and one after it in the second. Moving both tokens later leaves the score as it was, so a head reads only how far apart two tokens are (${PL(4,'part 4')}).</li>`+
    `<li><b>For a long text, check the slowest pairs.</b> They must not reach angles the model never met in training, or its attention can break down. To read texts longer than its training texts, a model has to slow those pairs and train on long texts, as ${M31} does (${PL(6,'part 6')}).</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “RoPE adds a position vector to the embedding.” That is what the original Transformer, GPT‑2 and BERT do. RoPE adds nothing: it turns q and k in every layer (${PL(2,'part 2')}).</p>`+
    `<p>✗ “With RoPE, a score depends on where the two tokens sit in the text.” With the content unchanged, it depends on the positions only through their distance, so the same two words score the same wherever the sentence sits in the text.</p>`+
    `<p>✗ “Any base works for any length.” The smaller the base, the faster every pair turns. On a long enough text even the slowest pair goes round, and two far-apart distances get the same angle. So the longer the text, the larger the base must be (${PL(5,'part 5')}).</p></div>${A.shell.toScenarios()}`,
    `<ul><li><b>什么在转，在哪里转。</b>RoPE 在每一层的注意力里，把 q 和 k 的每一对数转过“位置 × 这一对的转速”。embedding 和 v 都不动。所以位置只影响得分，不改动每个 token 交出去的内容。</li>`+
    `<li><b>顺序起作用了。</b>有了 RoPE，“咬”那一行偏向它前面的那个 token（见${J('stage','图里的表')}）：一句里是“狗”，另一句里是“人”，因为“狗”在第一句里在“咬”前面一个位置，在第二句里在后面一个位置。两个 token 一起往后挪，得分不变，所以 head 读到的只是两者相距多远（${PL(4,'第 4 部分')}）。</li>`+
    `<li><b>长文本要看最慢的数对。</b>它们不能转到模型训练时没见过的角度，否则注意力可能失灵。要读比训练文本更长的文本，模型就得放慢这些数对，并在长文本上训练，${M31} 就是这么做的（${PL(6,'第 6 部分')}）。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “RoPE 往 embedding 上加了一个位置向量。”这是最早的 Transformer、GPT‑2 和 BERT 的做法。RoPE 什么也不加：它在每一层里转动 q 和 k（${PL(2,'第 2 部分')}）。</p>`+
    `<p>✗ “有了 RoPE，得分取决于两个 token 在文本里的位置。”内容不变时，得分和位置的关系只体现在两者的距离上。所以同样两个词，不管这句话放在文本的哪里，得分都一样。</p>`+
    `<p>✗ “什么 base 都适合任何长度。”base 越小，每一对都转得越快。文本足够长时，连最慢的一对也会转满一圈，相差很远的两个距离就会得到同样的角度。所以文本越长，base 就要越大（${PL(5,'第 5 部分')}）。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real models on real text lengths, one plain phrase each (R2.1)
   ===================================================================== */
const PRESETS=[
 {id:'l2',t:{en:`${M2}, 4K-token texts`,zh:`${M2} 读 4K 个 token 的文本`},d:{en:'Meta’s 2023 model, on texts as long as its training texts',zh:'Meta 2023 年的模型，文本和训练时一样长'},s:{base:10000,sc:false,ctx:4096,train:4096}},
 {id:'l3',t:{en:`${M3}, 8K-token texts`,zh:`${M3} 读 8K 个 token 的文本`},d:{en:'Llama 3 with its larger base, on texts as long as its training texts',zh:'base 更大的 Llama 3，文本和训练时一样长'},s:{base:500000,sc:false,ctx:8192,train:8192}},
 {id:'l3x',t:{en:`${M3}, a 32K-token text`,zh:`${M3} 读 32K 个 token 的文本`},d:{en:'4 times longer than its training texts',zh:'比训练时的文本长 4 倍'},s:{base:500000,sc:false,ctx:32768,train:8192}},
 {id:'l31',t:{en:`${M31}, a 128K-token text`,zh:`${M31} 读 128K 个 token 的文本`},d:{en:'Llama 3 with its slow pairs slowed down, trained on texts up to 128K tokens',zh:'把慢的数对放慢的 Llama 3，训练文本最长 128K 个 token'},s:{base:500000,sc:true,ctx:131072,train:131072}},
 {id:'cl',t:{en:`${MCL}, 16K tokens of code`,zh:`${MCL} 读 16K 个 token 的代码`},d:{en:'Llama 2 fine-tuned on 16K-token code with a much larger base',zh:'在 16K token 的代码上微调过、base 大得多的 Llama 2'},s:{base:1000000,sc:false,ctx:16384,train:16384}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.base===SB.base&&p.s.sc===SB.sc&&p.s.ctx===SB.ctx&&p.s.train===SB.train);
function scenHTML(i){
  const p=PRESETS[i], d=sbData(p.s), name=esc(L(p.t)), both=d.rows.filter(r=>r.under&&r.beyond).length;
  const head=d.nBeyond?tx(`<p><b>${name}: ${d.nBeyond} slow pairs reach angles the model never met in training.</b></p>`,`<p><b>${name}：有 ${d.nBeyond} 对慢的数对转到了模型训练时没见过的角度。</b></p>`)
    :tx(`<p><b>${name}: every pair stays within the angles met in training.</b></p>`,`<p><b>${name}：每一对都停在训练时见过的角度以内。</b></p>`);
  let why='';
  if(p.id==='l2') why=tx(`<p>Base 10,000 is small, but a 4K text is short: the slowest pair turns only ${turnS(d.slow.t)}, exactly the angles the model met in training.</p>`,`<p>base 10,000 很小，但 4K 的文本很短：最慢的一对只转了 ${turnS(d.slow.t)}，正是模型训练时见过的角度。</p>`);
  if(p.id==='l3') why=tx(`<p>With base 500,000 the slow pairs barely move over 8K tokens: the slowest turns ${turnS(d.slow.t)}. Llama 3’s paper says the larger base helps it support longer contexts.</p>`,`<p>base 500,000 时，慢的数对在 8K 个 token 上几乎不动：最慢的只转了 ${turnS(d.slow.t)}。Llama 3 的论文说，更大的 base 有助于支持更长的上下文。</p>`);
  if(p.id==='l3x') why=tx(`<p>Past its 8K training texts, the slowest pair turns ${fint(Math.round(d.ratio))} times as far as in training. Staying under one turn is not enough: ${both} of these pairs stay under one turn, yet reach angles the model never met. Reaching those unseen angles is what can break a model past its training length; ${M31}’s scaling and long-context training fix it.</p>`,`<p>超过 8K 的训练长度，最慢的一对转到训练时的 ${fint(Math.round(d.ratio))} 倍。不满一圈还不够：这些数对里有 ${both} 对不满一圈，却转到了模型没见过的角度。超过训练长度时，转到这些没见过的角度，才可能让模型失灵；${M31} 的缩放和长文本训练解决的正是这个问题。</p>`);
  if(p.id==='l31') why=tx(`<p>The “llama3” scaling in its config.json makes the slow pairs turn 8 times slower, and long-context training up to 128K tokens (reported in the Llama 3 paper) covers every angle they reach: the slowest turns ${turnS(d.slow.t)}.</p>`,`<p>它的 config.json 里的“llama3”缩放把慢的数对的转速降到 1/8，最长 128K 个 token 的长文本训练（Llama 3 的论文里报告过）覆盖了它们转到的每个角度：最慢的一对转了 ${turnS(d.slow.t)}。</p>`);
  if(p.id==='cl') why=tx(`<p>Code Llama raised Llama 2’s base from 10,000 to 1,000,000 and fine-tuned on 16K-token sequences; its paper reports that this also reduces the bias toward short-distance attention.</p>`,`<p>Code Llama 把 Llama 2 的 base 从 10,000 提高到 1,000,000，并在 16K token 的序列上微调；它的论文说，这也减少了注意力偏向近距离的倾向。</p>`);
  return head+why;
}

/* =====================================================================
   challenge
   ===================================================================== */
const chRes=b=>chTried.has(b)?turnsS(chTurns(b)):'';
const chLevers=()=>A.ui.levers({aria:tx('The base, pick one','base，选一个'),value:chBase,options:CH.bases.map(b=>({v:b,title:tx(`Base ${fint(b)}`,`base ${fint(b)}`),desc:b===CH.start?tx('initial setup','初始设置'):'',result:esc(chRes(b))}))});
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chBase, best=chBest(), tm=chTurns(me), below=CH.bases[CH.bases.indexOf(best)-1];
  let h;
  if(chOK(me)&&me===best) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: with base ${fint(me)}, the slowest pair turns ${turnsS(tm)} over 1M tokens.</b> And it is the smallest base that works: base ${fint(below)} gives ${turnsS(chTurns(below))}.</p>`,
                               `<p class="okline" style="margin-top:12px"><b>达成目标：base ${fint(me)} 时，最慢的一对在 1M 个 token 上转 ${turnsS(tm)}。</b>而且这是能达标的最小 base：base ${fint(below)} 会转 ${turnsS(chTurns(below))}。</p>`);
  else if(chOK(me)) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: with base ${fint(me)}, the slowest pair turns ${turnsS(tm)} over 1M tokens.</b> But a smaller base works too: base ${fint(best)} gives ${turnsS(chTurns(best))}.</p>`,
                         `<p class="okline" style="margin-top:12px"><b>达成目标：base ${fint(me)} 时，最慢的一对在 1M 个 token 上转 ${turnsS(tm)}。</b>不过更小的 base 也行：base ${fint(best)} 转 ${turnsS(chTurns(best))}。</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: with base ${fint(me)}, the slowest pair makes ${fixd(tm,2)} full turns over 1M tokens.</b></p><p class="small">Hint: the slowest pair turns about ${MX('\\frac{1}{\\text{base}}')} radians per token. Multiply by 1M tokens, 1,024 × 1,024, and compare with one full turn, ${MX('2\\pi \\approx 6.28')} radians.</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：base ${fint(me)} 时，最慢的一对在 1M 个 token 上转过 ${fixd(tm,2)} 整圈。</b></p><p class="small">提示：最慢的一对每个 token 大约转 ${MX('\\frac{1}{\\text{base}}')} 弧度。乘上 1M 个 token（1,024 × 1,024），再和一整圈 ${MX('2\\pi \\approx 6.28')} 弧度比较。</p>`)
    +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the owner's review of batch 8 (2026-10-07): the chart keeps the bases that pass, the pick and the initial setup (brief); a note
     under it counts the bases left out, all of which wrap around */
  const seen=CH.bases.filter(b=>chOK(b)||b===me||b===CH.start);
  h+=A.ui.cmpChart({title:tx('Tokens the slowest pair needs for one full turn, with heads of 128 numbers (goal: more than the 1M-token text)','128 维 head 的最慢一对转满一圈需要多少个 token（目标：比 1M token 的文本更长）'),
    aria:tx('Tokens for one full turn of the slowest pair: ','最慢一对转满一圈需要的 token：')+seen.map(b=>`${fint(b)}${tx(': ','：')}${bigT(chLam(b))}`).join(tx('; ','；')),
    rows:CH.bases.map(b=>({label:tx(`Base ${fint(b)}: ${fixd(chTurns(b),2)} turns over 1M`,`base ${fint(b)}：1M 上转 ${fixd(chTurns(b),2)} 圈`),value:chLam(b),text:bigT(chLam(b)),ok:chOK(b),me:b===me,
      tags:(b===CH.start?[{text:tx('initial setup','初始设置')}]:[]).concat(b===me?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,hiddenText:n=>tx(n===1?'The other base wraps around within 1M tokens and is not shown.':`The other ${n} bases wrap around within 1M tokens and are not shown.`,`其余 ${n} 种 base 在 1M 个 token 内会转过一整圈，图里没有列出。`),
    goal:CH.L,goalLabel:tx('goal: 1M','目标 1M'),max:7.2e6,okText:tx('more than 1M tokens: under one turn','超过 1M 个 token：不满一圈'),noText:tx('1M or fewer: wraps around','不到 1M 个 token：转过一整圈'),goalText:tx('goal: 1M tokens','目标：1M 个 token')});
  /* base 100,000 may be left out of the chart (brief), so the sentence names it in words and does not point at its row */
  h+=tx(`<p class="small">Initial setup, base 10,000: the slowest pair wraps around ${fint(Math.floor(chTurns(10000)))} times over 1M tokens. A larger base slows it down. With base 100,000 it still wraps around about twice, while with base ${fint(best)} it stays under half a turn.</p>`,
        `<p class="small">初始设置是 base 10,000，最慢的一对在 1M 个 token 上转过 ${fint(Math.floor(chTurns(10000)))} 圈多。base 越大，它转得越慢。base 100,000 时它仍会转过大约两圈，base ${fint(best)} 时连半圈都不到。</p>`);
  h+=tx(`<p class="small">Why: the slowest pair turns about ${MX('\\frac{1}{\\text{base}}')} radians per token. One full turn is ${MX('2\\pi')} radians, so it needs about ${MX('2\\pi \\times \\text{base}')} tokens for it: the tokens grow in step with the base.</p>`,
        `<p class="small">原因：最慢的一对每个 token 约转 ${MX('\\frac{1}{\\text{base}}')} 弧度。一整圈是 ${MX('2\\pi')} 弧度，所以它转满一圈大约要 ${MX('2\\pi \\times \\text{base}')} 个 token，和 base 成正比。</p>`);
  h+=tx(`<p class="small muted">Note: this lab checks only that the slowest pair does not go round. ${M3} already has base 500,000, yet it was trained on 8K-token texts; to read 128K tokens, ${M31} also slowed its slow pairs and trained on long texts (${PL(6,'part 6')}). Test a model at the length you plan before you trust it there.</p>`,
        `<p class="small muted">注意：这个 Lab 只检查最慢的一对会不会转满一圈。${M3} 的 base 已经是 500,000，但它训练时的文本只有 8K 个 token；为了读 128K 个 token，${M31} 还放慢了慢的数对，并在长文本上训练（${PL(6,'第 6 部分')}）。要在某个长度上用一个模型，先在这个长度上测一测它。</p>`);
  return h;
}

/* =====================================================================
   copy the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===7?'Seven':fint(STEPS.length)} short parts. Open one head to see what RoPE turns, watch it tell “dog bites man” from “man bites dog”, see what happens when both tokens move, and how the base, the speeds and the text length fit together. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。打开一个 head 看 RoPE 转的是什么，看它怎样分清“狗咬人”和“人咬狗”、两个 token 一起挪动时会怎样，以及 base、转速和文本长度怎样联系在一起。第一次学从这里开始。`),
  sandbox:tx('Five real cases, from Llama 2 on its 4K-token texts to Llama 3.1 on a 128K-token text. Pick one to see how many turns each pair makes over the text.',
             '五个真实情况，从读 4K token 文本的 Llama 2，到读 128K token 文本的 Llama 3.1。选一个，看每一对在整段文本上转几圈。'),
  challenge:tx('A new model with 128-number heads will read texts of 1M tokens. Pick the smallest RoPE base whose slowest pair turns less than one full turn over the whole text.',
               '一个 head 为 128 维的新模型要读 1M 个 token 的文本。选出最小的 RoPE base，让最慢的一对在整段文本上转不满一圈。')
});
const SRC={
 c31:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 c3:'https://huggingface.co/unsloth/llama-3-8b/blob/main/config.json',
 c2:'https://huggingface.co/NousResearch/Llama-2-7b-hf/blob/main/config.json',
 ccl:'https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json',
 cmis:'https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json',
 cqw:'https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json',
 meta2:'https://github.com/meta-llama/llama/blob/main/llama/model.py',
 meta3:'https://github.com/meta-llama/llama3/blob/main/llama/model.py',
 hfl:'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',
 hfr:'https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py',
 hfg:'https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py',
 hfb:'https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py',
 vb:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/base.py',
 vl3:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/llama3_rope.py',
 vinit:'https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/__init__.py',
 attn:'https://arxiv.org/abs/1706.03762',
 roformer:'https://arxiv.org/abs/2104.09864',
 pi:'https://arxiv.org/abs/2306.15595',
 yarn:'https://arxiv.org/abs/2309.00071',
 codellama:'https://arxiv.org/abs/2308.12950',
 llama3:'https://arxiv.org/abs/2407.21783',
 metaScale:'https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py'
};
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser turns made-up vectors with the real speeds of Llama heads. The vectors and scores are an illustration; the speeds, bases and lengths come from the models’ config files.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>One head of 128 numbers per query and key, in 64 pairs of neighbours (numbers 2i and 2i + 1, as in Meta’s Llama code); the score ${MX('q \\cdot k')} is the sum of the 64 pairs’ dot products.</li>`+
 `<li>The 64 speeds ${A.ui.math('θ_i = \\text{base}^{-2i/128}')} for the bases in the configs: ${M2} 10,000, ${M3} and ${M31} 500,000, ${MCL} 1,000,000; ${M31}’s “llama3” scaling as transformers computes it (factor 8, low_freq_factor 1, high_freq_factor 4, original length 8,192): pairs whose full turn takes more than 8,192 tokens turn 8 times slower, pairs under 2,048 tokens keep their speed, and the 6 pairs in between (29 to 34) are slowed partly, 1.2 to 5.3 times.</li>`+
 `<li>Exact rotations of each pair by position × θᵢ; q and k are turned, v and the embedding are not.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The vectors are made up by hand: pair 0 is drawn large, the other 63 pairs follow simple formulas, and the keys of “dog” and “man” are set so that this head looks for the token right before “bites”. Positions count from 1.</li>`+
 `<li>One head, no causal mask, and no division by ${MX('\\sqrt{128}')}, which scales every score the same way. Parts 1 to 5 use ${M3}’s speeds without scaling, so the chart matches the formula.</li>`+
 `<li>“Angles met in training” takes each pair on its own, from 0 to where it gets at the training length. The challenge’s rule, the slowest pair under one full turn, is this lab’s simplified check. Real long-context methods also train at the long length: position interpolation (Chen et al., 2023) squeezes every position into the trained range, YaRN (Peng et al., 2023) and ${M31}’s “llama3” scaling slow only the slow pairs, and Code Llama raised the base.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>config.json of ${cite(SRC.c2,M2)} (no rope_theta, so the default 10,000 of Meta’s ${cite(SRC.meta2,'llama/model.py')}), ${cite(SRC.c3,M3)}, ${cite(SRC.c31,M31)}, ${cite(SRC.ccl,MCL)}, ${cite(SRC.cmis,'Mistral‑7B')} (rope_theta 10,000), ${cite(SRC.cqw,'Qwen2.5‑7B')} (rope_theta 1,000,000).</li>`+
 `<li>Pairs: Meta’s ${cite(SRC.meta3,'llama3 model.py')} pairs neighbours; Hugging Face’s ${cite(SRC.hfl,'modeling_llama.py')} pairs j with j + 64 (rotate_half). Added positions: Vaswani et al., ${cite(SRC.attn,'Attention Is All You Need')} (2017, sinusoidal); ${cite(SRC.hfg,'GPT‑2')} and ${cite(SRC.hfb,'BERT')} in transformers (learned).</li>`+
 `<li>Su et al., ${cite(SRC.roformer,'RoFormer')} (2021); ${cite(SRC.llama3,'The Llama 3 Herd of Models')} (2024): base 500,000 “to better support longer contexts”; long-context pre-training from 8K to 128K in six stages on about 800B tokens (reported for the 405B); ${cite(SRC.codellama,'Code Llama')} (2023): base 1,000,000, 16K-token sequences; the llama3 scaling: Meta’s ${cite(SRC.metaScale,'llama-models model.py')} (apply_scaling, “Values obtained from grid search”) and Hugging Face ${cite(SRC.hfr,'modeling_rope_utils.py')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-30.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器用 Llama head 的真实转速转动几个虚构的向量。向量和得分只是演示；转速、base 和长度来自模型的配置文件。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>一个 head 的 query 和 key 各有 128 维，按相邻两个数配成 64 对（第 2i 和第 2i + 1 个，和 Meta 的 Llama 代码一样）；得分 ${MX('q \\cdot k')} 是 64 对各自点积的总和。</li>`+
 `<li>64 种转速 ${A.ui.math('θ_i = \\text{base}^{-2i/128}')}，base 取自配置：${M2} 10,000，${M3} 和 ${M31} 500,000，${MCL} 1,000,000；${M31} 的“llama3”缩放按 transformers 的算法计算（factor 8，low_freq_factor 1，high_freq_factor 4，原始长度 8,192）：转一整圈要超过 8,192 个 token 的数对慢 8 倍，不到 2,048 个 token 的保持原速，中间的 6 对（第 29 到 34 对）只放慢一部分，1.2 到 5.3 倍。</li>`+
 `<li>每一对数按“位置 × θᵢ”精确旋转；转 q 和 k，不转 v，也不动 embedding。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>向量是手工虚构的：第 0 对画得最大，其余 63 对按简单的公式生成，“狗”和“人”的 key 设成让这个 head 去找“咬”前面的那个 token。位置从 1 开始数。</li>`+
 `<li>只有一个 head，没有因果 mask，也不除以 ${MX('\\sqrt{128}')}（它让每个得分同比例缩小）。第 1 到第 5 部分用 ${M3} 不缩放的转速，图和公式对得上。</li>`+
 `<li>“训练时见过的角度”对每一对单独算：从 0 到它在训练长度处转到的角度。挑战题的规则（最慢的一对不满一圈）是这个 Lab 的简化判断。真实的长上下文方法还要在长文本上训练：位置插值（Chen 等，2023）把每个位置都压进训练过的范围，YaRN（Peng 等，2023）和 ${M31} 的“llama3”缩放只放慢慢的数对，Code Llama 则调大了 base。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${cite(SRC.c2,M2)} 的 config.json（没有 rope_theta，所以用 Meta ${cite(SRC.meta2,'llama/model.py')} 的默认值 10,000）、${cite(SRC.c3,M3)}、${cite(SRC.c31,M31)}、${cite(SRC.ccl,MCL)}、${cite(SRC.cmis,'Mistral‑7B')}（rope_theta 10,000）、${cite(SRC.cqw,'Qwen2.5‑7B')}（rope_theta 1,000,000）。</li>`+
 `<li>配对：Meta 的 ${cite(SRC.meta3,'llama3 model.py')} 把相邻两个数配对；Hugging Face 的 ${cite(SRC.hfl,'modeling_llama.py')} 把第 j 个和第 j + 64 个配对（rotate_half）。加在输入上的位置：Vaswani 等，${cite(SRC.attn,'Attention Is All You Need')}（2017，正弦）；transformers 里的 ${cite(SRC.hfg,'GPT‑2')} 和 ${cite(SRC.hfb,'BERT')}（学出来的）。</li>`+
 `<li>Su 等，${cite(SRC.roformer,'RoFormer')}（2021）；${cite(SRC.llama3,'The Llama 3 Herd of Models')}（2024）：base 500,000，“更好地支持长上下文”；长文本预训练分 6 个阶段从 8K 到 128K，约 800B 个 token（这是 405B 的数据）；${cite(SRC.codellama,'Code Llama')}（2023）：base 1,000,000，16K token 的序列；llama3 缩放：Meta 的 ${cite(SRC.metaScale,'llama-models model.py')}（apply_scaling，注释写“Values obtained from grid search”）和 Hugging Face ${cite(SRC.hfr,'modeling_rope_utils.py')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-30。</p></div></div>`);
const PRIPOP=()=>tx(`<p>In each head, Wq and Wk turn every token’s vector into a query and a key. The query of one token and the key of another give a score, ${MX('q \\cdot k')}, and softmax turns a row of scores into attention weights: the lab ${labLink('mha')} builds this up.</p><p>Without position information this score is the same wherever the two tokens stand; the lab ${labLink('architecture')} shows why attention alone does not see the order.</p>`,
                    `<p>每个 head 里，Wq 和 Wk 把每个 token 的向量变成 query 和 key。一个 token 的 query 和另一个 token 的 key 得到一个得分 ${MX('q \\cdot k')}，softmax 再把一行得分变成注意力权重：${labLink('mha')}这个 Lab 从头讲了这些。</p><p>没有位置信息时，不管两个 token 站在哪里，这个得分都一样；${labLink('architecture')}这个 Lab 讲了为什么单靠注意力看不出顺序。</p>`);
const OUTCOME=()=>tx('explain what RoPE turns, predict what moving two tokens does to their score, and estimate how many turns each pair makes over a text for a given base.','讲清 RoPE 转的是什么，预测两个 token 一起挪动时得分会怎样，并估算给定 base 时每一对在一段文本上转几圈。');
const EXITS=()=>{
  const th2=thetas(10000,null), a4=4096*th2[NP-1], a16=16384*th2[NP-1];
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('A query and a key keep their content and both move 100 positions later. What happens to their score, and why?','一个 query 和一个 key 内容不变，一起往后挪 100 个位置。它们的得分会怎样？为什么？'),
   o:[tx('Each is turned by its own position, and the score sees only the angle between them, which grows with m − n; moving both turns both by the same extra angle, so that angle stays.','两者各按自己的位置转动，而得分只看它们之间的夹角，这个夹角随 m − n 变化；一起挪动让两者多转同样的角度，所以夹角不变。'),
     tx('RoPE adds a position vector to each token’s embedding, and every layer learns to subtract the two positions, so a move by the same amount cancels out.','RoPE 往每个 token 的 embedding 上加一个位置向量，每一层再学会把两个位置相减，所以挪动同样多就抵消了。'),
     tx('The score does change, but positions 100 tokens later are so close together in angle that the change is too small to see.','得分其实变了，只是往后 100 个位置的角度彼此很接近，变化小到看不出来。')],a:0,
   n:[null,tx('RoPE adds nothing to the embedding: it turns q and k inside every layer’s attention, and the property comes from the rotation itself, not from learning.','RoPE 不往 embedding 上加东西：它在每一层的注意力里转动 q 和 k，这个性质来自旋转本身，不是学出来的。'),
      tx('Each arrow turns 100 × θ further, which is a lot for a fast pair; the score stays the same exactly, not approximately, because only the angle between the arrows counts.','每个箭头多转 100 × θ，对快的数对来说这可不小；得分是严格不变，不是近似不变，因为只有两个箭头的夹角算数。')],
   why:tx('For one pair, the score is |q| × |k| × cos(φq − φk + (m − n) × θ), where φq and φk are the directions of the two arrows before the turn and m, n the positions of the query and the key: the positions appear only as m − n. Every pair of the head behaves this way, so the whole score does too.','对一对数，得分 = |q| × |k| × cos(φq − φk + (m − n) × θ)，其中 φq 和 φk 是两个箭头转动之前的方向，m、n 是 query 和 key 的位置：位置只以 m − n 的形式出现。head 里每一对都是这样，所以整个得分也是。')},
  {q:tx('Why does RoPE turn the 64 pairs of a head at 64 different speeds?','RoPE 为什么让一个 head 的 64 对数以 64 种不同的速度转动？'),
   o:[tx('A fast pair changes a lot between neighbours but comes round again every few tokens, and a slow pair changes little per token but keeps counting over long distances, so together they tell near and far distances apart.','快的数对在相邻 token 之间变化大，但每隔几个 token 就转回来；慢的数对每个 token 变化小，却能在很长的距离上一直计数，合起来近的和远的距离都分得清。'),
     tx('Each pair belongs to a different head, so each head specialises in one distance: fast heads look at neighbours and slow heads look far back.','每一对属于不同的 head，所以每个 head 专看一种距离：快的 head 看邻居，慢的 head 看很远的地方。'),
     tx('The speeds are learned in training, so each pair ends up turning at the speed its words need, fast for common words and slow for rare ones.','转速是训练时学出来的，所以每一对最后按它的词需要的速度转：常见词转得快，罕见词转得慢。')],a:0,
   n:[null,tx('All 64 speeds live inside every head: each head’s query and key have 128 numbers, and pair i of every head turns at the same θᵢ.','64 种转速都在每一个 head 里：每个 head 的 query 和 key 都有 128 维，每个 head 的第 i 对都按同一个 θᵢ 转。'),
      tx('The speeds are fixed by the base and the head size alone (the formula is in “Show the math”); RoPE has no parameters to learn.','转速只由 base 和 head 的维数决定（公式见“展开公式”）；RoPE 没有要学的参数。')],
   why:tx(`Pair 0 turns 1 radian per token and wraps around every 6.28 tokens, while even with base 10,000 the slowest pair needs ${fint(Math.round(TAU/th2[NP-1]))} tokens for one full turn. Near distances show up on the fast pairs, far ones on the slow pairs.`,`第 0 对每个 token 转 1 弧度，每 6.28 个 token 转过一整圈；而最慢的一对，即使 base 只有 10,000，也要 ${fint(Math.round(TAU/th2[NP-1]))} 个 token 才转过一整圈。近的距离体现在快的数对上，远的体现在慢的数对上。`)}],
 transfer:[
  {q:tx(`${M2} has base 10,000; over its 4K-token training texts, its slowest pair turns ${degS(a4)}. Run it on a 16K-token text without any change: about how far does its slowest pair turn?`,`${M2} 的 base 是 10,000；在 4K token 的训练文本上，它最慢的一对转 ${degS(a4)}。不做任何改动，让它读 16K token 的文本：最慢的一对大约转多少？`),
   o:[tx(`About the same, ${fint(Math.round(a4*R2D))}°`,`差不多，约 ${fint(Math.round(a4*R2D))}°`),tx(`About twice as far, ${fint(Math.round(2*a4*R2D))}°`,`约两倍，${fint(Math.round(2*a4*R2D))}°`),tx(`About 4 times as far, ${fint(Math.round(a16*R2D))}°`,`约 4 倍，${fint(Math.round(a16*R2D))}°`),tx('More than one full turn','超过一整圈')],a:2,
   go:()=>openSettings({base:10000,sc:false,ctx:16384,train:4096}),goLabel:tx('Open in Scenarios','在应用场景里打开'),
   why:tx(`The angle is position × θ, so it grows in step with the length: 16K is 4 times 4K, so 4 × ${degS(a4)} = ${degS(a16)}, still under a full turn but 4 times past anything the model met in training.`,`角度 = 位置 × θ，所以它和长度成正比：16K 是 4K 的 4 倍，所以是 4 × ${degS(a4)} = ${degS(a16)}，还不到一整圈，但已经是模型训练时见过的最大角度的 4 倍。`)},
  {q:tx(`Applied to ${M2}, position interpolation (Chen et al., 2023) would run it on 16K-token texts by dividing every position by 4. Between two neighbouring tokens, how far would the fastest pair then turn?`,`把位置插值（Chen 等，2023）用在 ${M2} 上，让它读 16K token 的文本，办法是把每个位置都除以 4。这样相邻两个 token 之间，最快的一对转多少？`),
   o:[tx(`${fixd(TH0D,1)}°, as before`,`${fixd(TH0D,1)}°，和原来一样`),tx(`About ${fint(Math.round(TH0D/4))}°, a quarter`,`约 ${fint(Math.round(TH0D/4))}°，四分之一`),tx(`About ${fint(Math.round(TH0D*4))}°, 4 times`,`约 ${fint(Math.round(TH0D*4))}°，4 倍`)],a:1,
   why:tx(`Neighbours are now 1/4 position apart, so the fastest pair turns 57.3° ÷ 4 = ${fixd(TH0D/4,1)}° between them. Every angle stays inside the range met in training, but neighbours look more alike on the fast pairs: this is why YaRN and ${M31}’s scaling leave the fast pairs alone.`,`相邻 token 现在只差 1/4 个位置，所以最快的一对在它们之间转 57.3° ÷ 4 = ${fixd(TH0D/4,1)}°。每个角度都留在训练时见过的范围里，但在快的数对上，相邻 token 更难分开：所以 YaRN 和 ${M31} 的缩放不动快的数对。`)}],
 practice:[
  {id:'llm-fundamentals',d:tx('Question 4 and the follow-up on long-context extrapolation: prove with one pair of numbers that the score depends on m and n only through m − n, extend it to every pair, and contrast RoPE with a position vector added to the input.','第 4 题和“长上下文外推”那条延伸：用一对数证明得分只通过 m − n 依赖 m 和 n，推广到所有数对，并和加在输入上的位置向量作对比。')},
  {id:'attention-kv-cache',d:tx('The follow-up “RoPE applied to cached keys”: why each key can be turned once, when it is written into the cache.','延伸“对缓存中的键应用 RoPE”：为什么每个 key 写进缓存时转一次就够了。')}]};
};
/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'RoPE',title:l?L(l.q):tx('How does a model know where each token is?','模型怎么知道每个 token 在哪个位置？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: attention scores','前置知识：注意力得分'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage rp-stage" id="rp-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="rp-st-title"></div><div class="meter-sub" id="rp-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="rp-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="rp-legend"></div>`+
   `<div class="stage-graph graph rp-graph" id="rp-graph"><div class="rp-vis" id="rp-vis"></div></div>`+
   `<p class="rp-scope small muted" id="rp-scope" tabindex="0"></p>`+
   `<div class="metrics rp-metrics" id="rp-metrics"></div>`+
   `<details class="math" id="rp-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="rp-mathbody"></div></details></div>`;
}
function controlsHTML(){
  const seg=(id,key,label,opts,v,note)=>`<div class="lockable">`+A.ui.seg({id,key,label,options:opts,value:v,note,inline:false})+`</div>`;
  return `<aside class="panel controls" id="rp-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint small muted" id="rp-hint">${tx('Pick a scenario in the card, or set your own case here.','在卡片里选一个场景，或在这里设定你自己的情况。')}</p>`+
   `<p class="ctl-lock" id="rp-lock" hidden></p>`+
   seg('rp-ctl-base','sbase',tx('RoPE base','RoPE base'),SB_BASES.map(b=>[b,fint(b)]),SB.base,tx('The number that sets the 64 speeds (the formula is in “Show the math”), called rope_theta in config.json. A larger base makes the slow pairs slower and lowers the slow end of the chart.','决定 64 种转速的那个数（公式见“展开公式”），config.json 里叫 rope_theta。base 越大，慢的数对越慢，图里慢的那一端越低。'))+
   seg('rp-ctl-sc','ssc',tx('Slow pairs','慢的数对'),[['0',tx('as the base gives','按 base 算')],['1',tx('8 times slower (llama3)','慢 8 倍（llama3）')]],SB.sc?'1':'0',tx(`${M31}’s “llama3” scaling: pairs whose full turn takes more than 8,192 tokens turn 8 times slower, pairs under 2,048 tokens keep their speed, and the 6 pairs in between (29 to 34) are slowed partly, 1.2 to 5.3 times.`,`${M31} 的“llama3”缩放：转一整圈要超过 8,192 个 token 的数对，转速降到 1/8；不到 2,048 个 token 的保持原速；中间的 6 对（第 29 到 34 对）只放慢一部分，1.2 到 5.3 倍。`))+
   `<div class="lockable">`+A.ui.range({id:'rp-ctl-ctx',key:'sctx',label:tx('Text length','文本长度'),min:0,max:CTX.length-1,value:CTX.indexOf(SB.ctx),out:tokN(SB.ctx),inline:false,note:tx('How many tokens the model reads at once: every pair makes more turns over a longer text.','模型一次读多少个 token：文本越长，每一对转的圈数越多。')})+`</div>`+
   seg('rp-ctl-train','strain',tx('Longest training text','最长的训练文本'),TRAIN.map(n=>[n,tokK(n)]),SB.train,tx('Assumed from the model: the longest texts it was trained on (Llama 2 4K, Llama 3 8K, Code Llama 16K, Llama 3.1 128K after its long-context training). It sets the hollow marks: how far each pair turned in training.','按模型设定：它训练时最长的文本（Llama 2 是 4K，Llama 3 是 8K，Code Llama 是 16K，Llama 3.1 经过长文本训练后是 128K）。它决定空心记号：每一对在训练时转到哪里。'))+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

def speeds(base=500_000.0, d=128):
    # theta_i = base^(-2i/d): one speed per pair
    return base ** (-np.arange(0, d, 2) / d)

def rope(x, pos, theta):
    # turn each pair (x[2i], x[2i+1]) by pos * theta_i
    a = pos * theta
    c, s = np.cos(a), np.sin(a)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=128), rng.normal(size=128)
th = speeds()
s1 = rope(q, 5, th) @ rope(k, 2, th)      # query at 5, key at 2
s2 = rope(q, 105, th) @ rope(k, 102, th)  # both 100 later
print(np.isclose(s1, s2))                 # same score?
turns = 131_072 * th[-1] / (2 * np.pi)    # slowest pair, 128K
print(f"slowest pair over 128K: {turns:.3f} turns")
# slowest pair over 128K: 0.051 turns`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="rp-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="rp-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('RoPE in NumPy: the 64 speeds, the turn of each pair of neighbours, and a numerical check of what moving both tokens does to the score (part 4 asks you first).','用 NumPy 实现 RoPE：64 种转速、每一对相邻数的旋转，再用数值检查两个 token 一起挪动时得分会怎样（第 4 部分会先让你预测）。')}</p><pre class="code" id="rp-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="rp-prod">`+
   tx(`<li>Meta’s ${cite(SRC.meta3,'<code>llama/model.py</code>')} (Llama 3): <code>apply_rotary_emb</code> views q and k as complex numbers made of neighbouring pairs, multiplies them by the precomputed turns, and the turned keys are then written to <code>cache_k</code>, once per token.</li>`+
      `<li>Hugging Face transformers, ${cite(SRC.hfl,'<code>models/llama/modeling_llama.py</code>')}: <code>apply_rotary_pos_emb</code> turns <code>query_states</code> and <code>key_states</code>; the values are left as they are. Its <code>rotate_half</code> pairs number j with number j + 64 instead of neighbours, and converted checkpoints reorder the rows of Wq and Wk, so the scores match Meta’s.</li>`+
      `<li>${cite(SRC.hfr,'<code>modeling_rope_utils.py</code>')}: <code>_compute_llama3_parameters</code> turns the config’s rope_scaling into the 64 scaled speeds. vLLM: ${cite(SRC.vb,'<code>rotary_embedding/base.py</code>')} (<code>RotaryEmbedding</code>, a cos and sin cache for every position), ${cite(SRC.vl3,'<code>llama3_rope.py</code>')} and <code>get_rope()</code> in ${cite(SRC.vinit,'<code>__init__.py</code>')}.</li>`,
      `<li>Meta 的 ${cite(SRC.meta3,'<code>llama/model.py</code>')}（Llama 3）：<code>apply_rotary_emb</code> 把 q 和 k 的相邻两个数看成一个复数，乘上预先算好的旋转，转好的 key 再写进 <code>cache_k</code>，每个 token 只转一次。</li>`+
      `<li>Hugging Face transformers 的 ${cite(SRC.hfl,'<code>models/llama/modeling_llama.py</code>')}：<code>apply_rotary_pos_emb</code> 转动 <code>query_states</code> 和 <code>key_states</code>，value 不动。它的 <code>rotate_half</code> 把第 j 维和第 j + 64 维配对，而不是相邻两维；转换 checkpoint 时会重排 Wq 和 Wk 的行，所以得分和 Meta 的一样。</li>`+
      `<li>${cite(SRC.hfr,'<code>modeling_rope_utils.py</code>')}：<code>_compute_llama3_parameters</code> 把配置里的 rope_scaling 变成 64 个缩放后的转速。vLLM：${cite(SRC.vb,'<code>rotary_embedding/base.py</code>')}（<code>RotaryEmbedding</code>，为每个位置缓存 cos 和 sin）、${cite(SRC.vl3,'<code>llama3_rope.py</code>')}，以及 ${cite(SRC.vinit,'<code>__init__.py</code>')} 里的 <code>get_rope()</code>。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.roformer,'RoFormer: Enhanced Transformer with Rotary Position Embedding')}<span>${tx('Su et al., 2021: the paper that introduced RoPE, turning q and k by angles set by their positions','Su 等，2021：提出 RoPE 的论文，按位置给 q 和 k 转一个角度')}</span></li>`+
   `<li>${cite(SRC.pi,'Extending Context Window of Large Language Models via Positional Interpolation')}<span>${tx('Chen et al., 2023: scale the positions down to the trained range; LLaMA models reach 32,768 tokens within 1,000 steps of fine-tuning','Chen 等，2023：把位置按比例缩进训练过的范围；LLaMA 模型微调不到 1,000 步就能读 32,768 个 token')}</span></li>`+
   `<li>${cite(SRC.yarn,'YaRN: Efficient Context Window Extension of Large Language Models')}<span>${tx('Peng et al., 2023: slow the slow pairs, leave the fast ones, with 10 times fewer tokens than earlier methods','Peng 等，2023：只放慢慢的数对、不动快的，所需 token 比之前的方法少 10 倍')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="rp-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="rp-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="rp-app m2" id="rp-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="rp-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many turns each pair makes over the whole text.','选一个场景，看看每一对在整段文本上转几圈。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="rp-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Pick the smallest base for 1M-token texts','为 1M token 的文本选最小的 base')}</h3>`+
   `<p>${tx(`A new model with heads of 128 numbers (64 pairs), like ${M3}, will read texts of up to ${DEF('1M tokens',ONE_M())}. Pick the smallest RoPE base, from 10,000, 100,000, 500,000 and 1,000,000, whose slowest pair turns less than one full turn over the whole 1M tokens.`,
            `一个新模型的 head 有 128 维（64 对），和 ${M3} 一样，它要读最长 ${DEF('1M 个 token',ONE_M())} 的文本。从 10,000、100,000、500,000 和 1,000,000 里选出最小的 RoPE base，让最慢的一对在整整 1M 个 token 上转不满一圈。`)}</p>`+
   `<p class="small muted">${tx('This lab uses a simple rule: it keeps the slowest pair from giving two far-apart distances the same angle.','这个 Lab 用一条简单的规则：它保证最慢的一对不会给两个相距很远的距离同一个角度。')}</p>`+
   `<div class="rp-lv" id="rp-ic-ch">${chLevers()}</div>`+
   `<div class="goal mono" id="rp-ch-goal"><span>${tx('The slowest pair over 1M tokens','最慢的一对在 1M 个 token 上')}</span><span><b id="rp-ch-now"></b>${tx(' / under 1 turn',' / 不到 1 圈')}</span></div><div class="progress" id="rp-ch-prog" aria-hidden="true"><i></i></div>`+
   submit+`<div id="rp-ch-res" class="rp-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=cur();
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=cur(); return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
/* why Next is greyed out: a line under the buttons and on the phone's part bar, and the same text as the button's tooltip (A13) */
function nextWhy(){if(mode!=='guided'||!isNextDisabled()) return ''; const s=cur(); return s.why?s.why():'';}
function railExtras(rail){
  if(mode!=='guided') return;
  const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
  if(nav) nav.insertAdjacentHTML('afterend','<p class="rp-why rp-why-off" aria-live="polite"></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="rp-why rp-why-off"></p>');
  const sm=$('[data-nav="showme"]',rail); if(sm) sm.dataset.tipx=tx('“Show me” does this part for you: it sets what the card asks for, so you can see the result.','“演示给我看”替你完成这一部分：按卡片的要求设好，让你直接看到结果。');
}
function renderRail(){const rail=$('#rp-rail',app); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; railExtras(rail); A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#rp-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=cur(), box=$('[data-obs]',rail);
    if(s.expect&&s.expect()&&(!s.gate||s.gate())) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step])&&(!s.gate||s.gate());
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.rp-why',rail).forEach(p=>{if(why){p.textContent=why; p.classList.remove('rp-why-off');} else p.classList.add('rp-why-off');});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step]||(!!s.gate&&!s.gate());
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#rp-insight',rail); if(ins){const html=c>=0?`<div class="eyebrow">${tx(`Scenario ${c+1}`,`场景 ${c+1}`)}</div>`+scenHTML(c):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the five scenarios above.','从上面五个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const now=$('#rp-ch-now',rail); if(!now) return;
    const t=chTurns(chBase), ok=t<1; now.textContent=turnsS(t);
    $('#rp-ch-prog i',rail).style.width=Math.min(100,t*100).toFixed(1)+'%';
    $('#rp-ch-goal',rail).classList.toggle('met',ok); $('#rp-ch-prog',rail).classList.toggle('met',ok); $('#rp-ch-goal',rail).classList.toggle('rp-over',!ok); $('#rp-ch-prog',rail).classList.toggle('rp-over',!ok);
    $$('#rp-ic-ch [data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===chBase)); const r=b.querySelector('.lv-r'); if(r) r.textContent=chRes(v);});
    if(chSubmitted&&ok) chReveal=true;
    const res=$('#rp-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing: SVG laid out in real pixels (the stage's width), redrawn on every change
   ===================================================================== */
const f1=v=>Math.round(v*10)/10;
/* box: the width a text must fit in (R7.8; the flow test compares it with getComputedTextLength) */
const TX=(x,y,s,cls,anchor,box)=>`<text x="${f1(x)}" y="${f1(y)}"${anchor?` text-anchor="${anchor}"`:''} class="${cls||'rp-tx'}"${box?` data-box="${f1(box)}"`:''}>${esc(s)}</text>`;
const PT=(cx,cy,r,a)=>[cx+r*Math.cos(a),cy-r*Math.sin(a)];
function arrowSVG(cx,cy,r,a,cls,hs){const [x,y]=PT(cx,cy,r,a), h=hs||Math.max(4,Math.min(9,Math.abs(r)*0.3)), b1=a+Math.PI*0.86, b2=a-Math.PI*0.86;
  return `<line x1="${f1(cx)}" y1="${f1(cy)}" x2="${f1(x)}" y2="${f1(y)}" class="${cls}"/><path d="M${f1(x)} ${f1(y)} L${f1(x+h*Math.cos(b1))} ${f1(y-h*Math.sin(b1))} L${f1(x+h*Math.cos(b2))} ${f1(y-h*Math.sin(b2))} Z" class="${cls} rp-hd"/>`;}
function arcSVG(cx,cy,r,a0,a1,cls){const d=a1-a0; if(Math.abs(d)<1e-4) return '';
  if(Math.abs(d)>=TAU-1e-6) return `<circle cx="${f1(cx)}" cy="${f1(cy)}" r="${f1(r)}" class="${cls}"/>`;
  const [x0,y0]=PT(cx,cy,r,a0), [x1,y1]=PT(cx,cy,r,a1);
  return `<path d="M${f1(x0)} ${f1(y0)} A${f1(r)} ${f1(r)} 0 ${Math.abs(d)>Math.PI?1:0} ${d>0?0:1} ${f1(x1)} ${f1(y1)}" class="${cls}"/>`;}
const svgOpen=(W,H,label)=>`<svg class="rp-svg" viewBox="0 0 ${f1(W)} ${f1(H)}" width="${f1(W)}" height="${f1(H)}" role="img" aria-label="${esc(label)}">`;
const visW=()=>{const v=$('#rp-vis',app); return Math.max(280,Math.min(720,Math.round((v&&v.clientWidth)||560)));};

/* parts 1 and 7: the head's scores in the two orders; the row of "bites" is outlined and its highest score has the thick outline (A2) */
const COND_HEAD=()=>tx('Shown here: a made-up head that looks one token back; no causal mask.','演示条件：一个虚构的 head，设成往前看一个 token；没有因果 mask。');
function svgOrder(W){
  const rope=ropeOn(), two=W>=440, lab=84, cs=Math.max(42,Math.min(58,Math.floor(((two?W/2-12:W)-lab-8)/3))), gw=cs*3, gh=34+18+cs*3;
  const H=two?gh+6:gh*2+14;
  let s=svgOpen(W,H,tx('The head’s scores in “dog bites man” and in “man bites dog”','这个 head 在“狗咬人”和“人咬狗”里的得分'));
  const block=(ord,x0,y0)=>{const g=grid(ord,rope), words=ORD[ord], top=topOf(ord,rope);
    let h=TX(x0+lab+gw/2,y0+14,tx(ord==='a'?'“dog bites man”':'“man bites dog”',ord==='a'?'“狗咬人”':'“人咬狗”'),'rp-tx rp-b','middle');
    h+=TX(x0+lab+gw/2,y0+30,tx('keys →','key →'),'rp-tx rp-mut','middle');
    words.forEach((kw,n)=>{h+=TX(x0+lab+n*cs+cs/2,y0+46,w(kw),'rp-tx','middle',cs-2);});
    words.forEach((qw,m)=>{const y=y0+52+m*cs;
      h+=TX(x0+lab-6,y+cs/2+4,w(qw)+(rope?` (${m+1})`:''),'rp-tx','end',lab-18);
      words.forEach((kw,n)=>{const v=g[m][n], hot=m===1&&n===top.j, x=x0+lab+n*cs;
        const tip=tx(`In “${ord==='a'?'dog bites man':'man bites dog'}”, the query of “${w(qw)}”${rope?` at position ${m+1}`:''} and the key of “${w(kw)}”${rope?` at position ${n+1}`:''} score ${sc2(v)}: the sum of their 64 pairs’ dot products.${hot?' This is the highest score in the row of “bites”.':''}`,
                     `在“${ord==='a'?'狗咬人':'人咬狗'}”里，“${w(qw)}”${rope?`（位置 ${m+1}）`:''}的 query 和“${w(kw)}”${rope?`（位置 ${n+1}）`:''}的 key 得分 ${sc2(v)}：它们 64 对各自点积的总和。${hot?'这是“咬”那一行的最高分。':''}`);
        h+=`<g class="rp-cellg" data-tipx="${esc(tip)}"><rect x="${f1(x+1)}" y="${f1(y+1)}" width="${cs-2}" height="${cs-2}" rx="3" class="rp-cell${hot?' rp-hot':''}" fill-opacity="${(0.05+0.03*Math.min(10,Math.abs(v))).toFixed(2)}"/>`+TX(x+cs/2,y+cs/2+4,sc2(v),'rp-tx rp-num'+(hot?' rp-b':''),'middle',cs-4)+`</g>`;});});
    h+=`<rect x="${f1(x0+lab-2)}" y="${f1(y0+52+cs-2)}" width="${f1(gw+4)}" height="${f1(cs+4)}" rx="5" class="rp-rowbox"/>`;
    h+=`<text transform="translate(${f1(x0+11)} ${f1(y0+52+gw/2)}) rotate(-90)" text-anchor="middle" class="rp-tx rp-mut">${esc(tx('queries →','query →'))}</text>`;
    return h;};
  if(two){s+=block('a',0,0)+block('b',W/2+6,0);}
  else{s+=block('a',(W-lab-gw)/2,0)+block('b',(W-lab-gw)/2,gh+14);}
  const ta=topOf('a',rope), tb=topOf('b',rope);
  return {cap:tx('The head’s scores in the two orders','这个 head 在两种顺序里的得分'),svg:s+'</svg>',
    note:`<b>${rope?tx(`With RoPE (the position in brackets), the row of “bites” favours the token just before it: “${w(ta.w)}” in one order, “${w(tb.w)}” in the other.`,`有 RoPE（括号里是位置）时，“咬”那一行偏向它前面的那个 token：一种顺序里是“${w(ta.w)}”，另一种里是“${w(tb.w)}”。`)
      :tx(`Without positions, the row of “bites” gives its highest score to “${w(ta.w)}” in both orders.`,`没有位置时，“咬”那一行在两种顺序里都把最高分给了“${w(ta.w)}”。`)}</b> <span class="rp-cond">${rope?COND_HEAD():tx('Shown here: one head of 128 made-up numbers per query and key, no causal mask.','演示条件：一个 head，每个 query 和 key 都是 128 维的虚构向量，没有因果 mask。')}</span>`};
}
/* part 2: one head's query of "bites", drawn in place: 128 numbers in rows; in "pairs" a bracket joins numbers 2i and 2i + 1 under
   them; in "arrows" each pair's arrow (x = its first number, y = its second) sits under its bracket. The rows keep one height in
   all three views, so nothing moves while the learner switches. Below: "added at the input" next to "turned inside attention". */
function svgHead(W){
  const q=QV.bites, cols=W>=480?32:16, rows=HD/cols, cw=Math.max(8,Math.min(17,Math.floor((W-8)/cols))), pw=2*cw, x0=Math.round((W-cw*cols)/2);
  const ch=14, r=Math.max(6,Math.min(13,pw/2-2)), rowH=ch+27+2*r, H=rows*rowH;
  const pairsOn=headView!=='128', arrowsOn=headView==='arrows';
  let s=svgOpen(W,H,tx('The query of “bites” in one head: 128 numbers grouped into 64 pairs, each drawn as an arrow','一个 head 里“咬”的 query：128 维配成 64 对，每一对画成一个箭头'));
  for(let rr=0;rr<rows;rr++){const y=rr*rowH+2, yb=y+ch+3;
    for(let k=0;k<cols;k++){const j=rr*cols+k, v=q[j], x=x0+k*cw, pi=j>>1;
      const tip=tx(`Number ${j} of the query of “bites”: ${n3(v)}${pairsOn?`, the ${j%2?'second':'first'} number of pair ${pi}`:''}.`,`“咬”的 query 的第 ${j} 个数值：${n3(v)}${pairsOn?`，是第 ${pi} 对的第 ${j%2?'二':'一'}个数值`:''}。`);
      s+=`<g class="rp-cellg" data-tipx="${esc(tip)}"><rect x="${f1(x)}" y="${f1(y)}" width="${cw-1}" height="${ch}" rx="1.5" class="rp-numc${pairsOn&&pi===0?' rp-p0':''}" fill-opacity="${(0.07+0.75*Math.min(1,Math.abs(v)/1.1)).toFixed(2)}"/></g>`;
      if(!pairsOn&&j%8===0) s+=TX(x+cw/2,yb+13,String(j),j===0?'rp-tx rp-b rp-sm':'rp-tx rp-mut rp-sm','middle');}
    if(!pairsOn) continue;
    for(let k=0;k<cols;k+=2){const pi=(rr*cols+k)/2, xa=x0+k*cw+1, xb=x0+(k+2)*cw-2, xc=(xa+xb)/2;
      s+=`<path d="M${f1(xa)} ${f1(yb)} L${f1(xa)} ${f1(yb+4)} L${f1(xb)} ${f1(yb+4)} L${f1(xb)} ${f1(yb)}" class="rp-brk${pi===0?' on':''}"/>`;
      if(pi%4===0) s+=TX(xc,yb+15,String(pi),pi===0?'rp-tx rp-b rp-sm':'rp-tx rp-mut rp-sm','middle');
      if(arrowsOn){const cy=yb+20+r, xy=[q[2*pi],q[2*pi+1]], len=Math.hypot(xy[0],xy[1]), a=Math.atan2(xy[1],xy[0]);
        const tip=tx(`Pair ${pi} of the query of “bites”: (${n3(xy[0])}, ${n3(xy[1])}), an arrow of length ${fixd(len,2)} pointing at ${dS(a*R2D)}.`,`“咬”的 query 的第 ${pi} 对：（${n3(xy[0])}，${n3(xy[1])}），一个长 ${fixd(len,2)}、指向 ${dS(a*R2D)} 的箭头。`);
        s+=`<g class="rp-mini" data-tipx="${esc(tip)}"><circle cx="${f1(xc)}" cy="${f1(cy)}" r="${f1(r)}" class="rp-dial${pi===0?' rp-on':''}"/>`+arrowSVG(xc,cy,(r-1.5)*(pi===0?1:0.45+0.55*len/1.5),a,pi===0?'rp-q':'rp-qs',pi===0?4:2.6)+`</g>`;}}}
  const box=(t,pos)=>`<div class="rp-box${pos?' rp-posbox':''}">${esc(t)}</div>`, op=t=>`<div class="rp-op" aria-hidden="true">${esc(t)}</div>`;
  const cmp=`<div class="rp-cmp" id="rp-cmp"><div class="rp-cmp-t">${tx('Where the position goes','位置放在哪里')}</div><div class="rp-cmp-g">`+
    `<div class="rp-col"><div class="rp-col-h">${tx('Added at the input','加在输入上')}</div><div class="rp-col-s">${tx('the original Transformer (sine and cosine), GPT‑2 and BERT (learned)','最早的 Transformer（正弦和余弦）、GPT‑2 和 BERT（学出来的）')}</div>`+
      `<div class="rp-row">`+box(tx('embedding row of “dog”','“狗”的 embedding 行'))+op('+')+box(tx('vector of position 1','位置 1 的向量'),true)+`</div>`+
      op(tx('↓ once, before layer 1','↓ 只加一次，在第 1 层之前'))+box(tx('every layer: q, k and v carry the position, mixed into their numbers','每一层：q、k、v 的数里都混着位置'))+`</div>`+
    `<div class="rp-col"><div class="rp-col-h">${tx('Turned inside attention: RoPE','在注意力里转动：RoPE')}</div><div class="rp-col-s">${tx('Llama, Mistral, Qwen','Llama、Mistral、Qwen')}</div>`+
      box(tx('embedding row of “dog”, nothing added','“狗”的 embedding 行，什么也不加'))+op(tx('↓ in every layer','↓ 每一层'))+box(tx('from the layer’s input x: q = Wq·x, k = Wk·x, v = Wv·x','由这一层的输入 x 算出 q = Wq·x，k = Wk·x，v = Wv·x'))+op('↓')+
      `<div class="rp-box rp-posbox">${tx(`turn every pair of q and k by ${MX('\\text{position} \\times')} ${DEF(MX('\\theta_i'),TIP.theta())}, then score ${MX('q \\cdot k')}. v is not turned`,`把 q 和 k 的每一对转过 ${MX('\\text{位置} \\times')} ${DEF(MX('\\theta_i'),TIP.theta())}，再算得分 ${MX('q \\cdot k')}。v 不转`)}</div>`+`</div></div></div>`;
  const cap={'128':tx('128 numbers of the query of “bites” (the labels give each number’s index)','“咬”的 query 的 128 个数值（标签是数值的编号）'),
    pairs:tx('64 pairs: numbers 2i and 2i + 1 form pair i (the labels give each pair’s index)','64 对：第 2i 和第 2i + 1 个数值是第 i 对（标签是对的编号）'),
    arrows:tx('Each pair as an arrow: x is its first number, y its second','每一对画成一个箭头：x 是第一个数值，y 是第二个')}[headView];
  return {cap,svg:s+'</svg>'+cmp};
}
/* one pair's plane: the query (solid) and the key (dashed), the angle between, and the query's shadow on the key's line; with RoPE, an arc
   outside the dial for each arrow's turn (A4): solid for the query, dashed for the key, drawn when the turn is less than a full turn */
function turnArc(cx,cy,r,a0,a1,cls){
  if(a1-a0<1e-4||a1-a0>=TAU-1e-6) return '';
  const [x,y]=PT(cx,cy,r,a1), t=a1+Math.PI/2, h=5, bx=x-h*Math.cos(t), by=y+h*Math.sin(t), nx=3*Math.cos(t+Math.PI/2), ny=-3*Math.sin(t+Math.PI/2);
  return arcSVG(cx,cy,r,a0,a1,cls)+`<path d="M${f1(x)} ${f1(y)} L${f1(bx+nx)} ${f1(by+ny)} L${f1(bx-nx)} ${f1(by-ny)} Z" class="${cls} rp-ah"/>`;}
function pairPanel(cx,cy,R,p){
  const lq=R*0.86, lk=R*0.86*P0.k.dog[0]/P0.q.bites[0], qa=p.qa*D2R, ka=p.ka*D2R, proj=lq*Math.cos((p.qa-p.ka)*D2R);
  let s=`<circle cx="${f1(cx)}" cy="${f1(cy)}" r="${f1(R)}" class="rp-dial"/>`;
  if(p.rope){const q0=P0.q.bites[1]*D2R, k0=P0.k.dog[1]*D2R;
    s+=turnArc(cx,cy,R+5,q0,q0+p.m*TH0,'rp-turnq')+turnArc(cx,cy,R+11,k0,k0+p.n*TH0,'rp-turnk');
    s+=arrowSVG(cx,cy,lq,q0,'rp-c')+arrowSVG(cx,cy,lk,k0,'rp-c');}
  const [fx,fy]=PT(cx,cy,proj,ka), [qx,qy]=PT(cx,cy,lq,qa);
  s+=`<line x1="${f1(cx)}" y1="${f1(cy)}" x2="${f1(fx)}" y2="${f1(fy)}" class="rp-shadow"/><line x1="${f1(qx)}" y1="${f1(qy)}" x2="${f1(fx)}" y2="${f1(fy)}" class="rp-drop"/>`;
  s+=arcSVG(cx,cy,R*0.3,ka,ka+p.between*D2R,'rp-arc rp-arcb');
  s+=arrowSVG(cx,cy,lk,ka,'rp-k')+arrowSVG(cx,cy,lq,qa,'rp-q');
  return s;}
/* where: {en:'In “dog bites man”', zh:'在“狗咬人”里'}; one colon per sentence (R6.14) */
const panelTip=(p,where)=>tx(`${where.en}, pair 0 of the query of “bites”${p.rope?` (position ${p.m}, turned ${p.m} × 57.3°)`:''} points at ${dS(wrapDeg(p.qa))} and the key of “dog”${p.rope?` (position ${p.n}, turned ${p.n} × 57.3°)`:''} at ${dS(wrapDeg(p.ka))}: ${dS(Math.abs(p.between))} apart, so this pair gives ${sc2(p.share)}. The whole head, all 64 pairs, gives ${sc2(p.head)}.`,
  `${where.zh}，“咬”的 query 的第 0 对${p.rope?`（位置 ${p.m}，转过 ${p.m} × 57.3°）`:''}指向 ${dS(wrapDeg(p.qa))}，“狗”的 key${p.rope?`（位置 ${p.n}，转过 ${p.n} × 57.3°）`:''}指向 ${dS(wrapDeg(p.ka))}，相差 ${dS(Math.abs(p.between))}，所以这一对给出 ${sc2(p.share)}。整个 head（64 对）给出 ${sc2(p.head)}。`);
/* the three lines under a dial: each arrow's position and turn, then the angle between and pair 0's share (A4, A16) */
function dialLines(s,cx,y,W,p,last){
  const sm=W<480?' rp-sm':'', bx=W/2-2;
  if(!p.rope) return s+TX(cx,y,tx('no positions, no turn','没有位置，不转'),'rp-tx'+sm,'middle',bx)+TX(cx,y+16,tx(`${dS(Math.abs(p.between))} apart`,`相差 ${dS(Math.abs(p.between))}`),'rp-tx'+sm,'middle',bx)+TX(cx,y+32,last,'rp-tx rp-b'+sm,'middle',bx);
  const big=p.m*TH0>=TAU;
  const qu=W<480?(x=>x):(x=>`“${x}”`);   /* on a phone the words go without quotes, so the two panels' lines keep a gap */
  return s+TX(cx,y,big?tx(`${qu('bites')} at ${p.m}`,`“咬”在 ${p.m}`):tx(`${qu('bites')} at ${p.m}: +${p.m} × 57.3°`,`“咬”在 ${p.m}：+${p.m} × 57.3°`),'rp-tx'+sm,'middle',bx)
    +TX(cx,y+16,big?tx(`${qu('dog')} at ${p.n}`,`“狗”在 ${p.n}`):tx(`${qu('dog')} at ${p.n}: +${p.n} × 57.3°`,`“狗”在 ${p.n}：+${p.n} × 57.3°`),'rp-tx'+sm,'middle',bx)
    +TX(cx,y+32,last,'rp-tx rp-b'+sm,'middle',bx);}
/* part 3: the row of "bites" in each sentence, as bars under its dial (A2); the highest is outlined */
function rowBars(cx,y,W,ord,rope){
  const g=grid(ord,rope)[1], top=topOf(ord,rope), half=W/2-12, lw=W<480?34:44, vw=34, x0=cx-half/2, bw=half-lw-vw, X=v=>Math.max(1,Math.max(0,v)/10*bw);
  let s='';
  [0,2].forEach((j,k)=>{const yy=y+k*20, v=g[j], kw=ORD[ord][j], tip=tx(`In “${ord==='a'?'dog bites man':'man bites dog'}”, the query of “bites” and the key of “${w(kw)}”${rope?` (position ${j+1})`:''} score ${sc2(v)}${j===top.j?', the highest in the row of “bites”':''}.`,`在“${ord==='a'?'狗咬人':'人咬狗'}”里，“咬”的 query 和“${w(kw)}”${rope?`（位置 ${j+1}）`:''}的 key 得分 ${sc2(v)}${j===top.j?'，是“咬”那一行的最高分':''}。`);
    s+=`<g class="rp-cellg" data-tipx="${esc(tip)}">`+TX(x0+lw-5,yy+11,w(kw),'rp-tx'+(W<480?' rp-sm':''),'end',lw-6)+`<rect x="${f1(x0+lw)}" y="${f1(yy)}" width="${f1(X(v))}" height="14" rx="2" class="rp-hbar${j===top.j?' rp-top':''}"/>`+TX(x0+lw+X(v)+4,yy+11,sc2(v),'rp-tx'+(j===top.j?' rp-b':'')+(W<480?' rp-sm':''),null,vw)+`</g>`;});
  return s;}
function svgPair2(W){
  const R=Math.max(50,Math.min(62,W/4-24)), cy=R+38, xs=[W/4,3*W/4], A0=pair0(2,1,rope3), B0=pair0(2,3,rope3), yb=cy+R+64, H=yb+50;
  let s=svgOpen(W,H,tx('Pair 0 of the query of “bites” and the key of “dog” in both sentences','两句话里“咬”的 query 和“狗”的 key 的第 0 对'));
  [[A0,tx('“dog bites man”','“狗咬人”'),xs[0],{en:'In “dog bites man”',zh:'在“狗咬人”里'},'a'],[B0,tx('“man bites dog”','“人咬狗”'),xs[1],{en:'In “man bites dog”',zh:'在“人咬狗”里'},'b']].forEach(([p,name,cx,where,ord])=>{
    s+=TX(cx,14,name,'rp-tx rp-b','middle');
    s+=`<g class="rp-big" tabindex="0" data-tipx="${esc(panelTip(p,where))}">`+pairPanel(cx,cy,R,p)+`</g>`;
    s=dialLines(s,cx,cy+R+17,W,p,tx(`pair 0: ${sc2(p.share)}`,`第 0 对：${sc2(p.share)}`));
    s+=rowBars(cx,yb+10,W,ord,rope3);});
  s+=TX(W/2,yb,tx('row of “bites”, whole head (64 pairs):','“咬”那一行，整个 head（64 对）：'),'rp-tx rp-b'+(W<480?' rp-sm':''),'middle',W-4);
  const ta=topOf('a',rope3), tb=topOf('b',rope3);
  return {cap:tx('Pair 0 of “bites” and “dog”, in both sentences','两句话里，“咬”和“狗”的第 0 对'),svg:s+'</svg>',
    note:`<b>${rope3?tx(`With RoPE, the row of “bites” favours the token just before it: “${w(ta.w)}” in one order, “${w(tb.w)}” in the other.`,`有 RoPE 时，“咬”那一行偏向它前面的那个 token：一种顺序里是“${w(ta.w)}”，另一种里是“${w(tb.w)}”。`):tx(`Without RoPE, both sentences give the same arrows, and the row of “bites” favours “${w(ta.w)}” both times.`,`没有 RoPE 时，两句话里箭头一样，“咬”那一行两次都偏向“${w(ta.w)}”。`)}</b> <span class="rp-cond">${COND_HEAD()}</span>`};
}
/* part 4: as written and moved later, RoPE on; the second dial waits for the prediction (R3.8) */
function svgShift(W){
  const R=W<600?48:54, cy=R+38, xs=[W/4,3*W/4], A0=pair0(2,1,true), B0=pair0(2+shift,1+shift,true), open=!!pred, moved=open&&shift>0;
  const yb=cy+R+56, py=yb+20, ph=84, H=moved?py+ph+42:yb;
  let s=svgOpen(W,H,tx('Pair 0 as written and moved later, with RoPE on','RoPE 开着：第 0 对在原文里和往后挪以后'));
  [[A0,tx('as written','原文里'),xs[0],true,{en:'As written',zh:'原文里'}],[B0,shift?tx(`both moved ${shift} later`,`一起往后挪 ${shift}`):tx('not moved yet','还没挪'),xs[1],open,{en:shift?`Moved ${shift} later`:'Not moved yet',zh:shift?`一起往后挪 ${shift} 以后`:'还没挪时'}]].forEach(([p,name,cx,show,where])=>{
    s+=TX(cx,14,name,'rp-tx rp-b','middle',W/2-8);
    if(!show){s+=`<g class="rp-mask" data-tipx="${esc(tx('This dial is shown once you answer the question in this part.','回答这一部分的问题之后，才显示这个表盘。'))}"><circle cx="${f1(cx)}" cy="${f1(cy)}" r="${f1(R)}" class="rp-maskc"/>`+TX(cx,cy-3,tx('shown after','预测之后'),'rp-tx rp-mut','middle',2*R-12)+TX(cx,cy+12,tx('you predict','才显示'),'rp-tx rp-mut','middle',2*R-12)+`</g>`; return;}
    s+=`<g class="rp-big" tabindex="0" data-tipx="${esc(panelTip(p,where))}">`+pairPanel(cx,cy,R,p)+`</g>`;
    s=dialLines(s,cx,cy+R+17,W,p,tx(`${dS(Math.abs(p.between))} apart, head ${sc2(p.head)}`,`相差 ${dS(Math.abs(p.between))}，head ${sc2(p.head)}`));});
  if(moved){
    /* the head's score against the distance m − n: the content fixed, the positions only through their difference */
    const x0=42, x1=W-12, X=d=>x0+(d+8)/16*(x1-x0), pts=[]; let lo=1e9, hi=-1e9;
    for(let k=0;k<=160;k++){const d=-8+k*0.1, v=score('bites','dog',d,0,true); pts.push([d,v]); lo=Math.min(lo,v); hi=Math.max(hi,v);}
    lo=Math.floor(lo-0.5); hi=Math.ceil(hi+0.5); const Y=v=>py+8+(hi-v)/(hi-lo)*(ph-16);
    s+=TX(2,py-10,tx('The head’s score against the distance m − n','这个 head 的得分和距离 m − n'),'rp-tx rp-b',null,W-4); /* x 2, not 0: a CJK glyph starts 1px left of x */
    s+=`<line x1="${x0}" y1="${f1(py+ph)}" x2="${f1(x1)}" y2="${f1(py+ph)}" class="rp-axis"/>`;
    [-8,-4,0,4,8].forEach(d=>{s+=`<line x1="${f1(X(d))}" y1="${f1(py+ph)}" x2="${f1(X(d))}" y2="${f1(py+ph+4)}" class="rp-axis"/>`+TX(X(d),py+ph+15,(d<0?MINUS:'')+Math.abs(d),'rp-tx rp-mut','middle');});
    [lo,hi].forEach(v=>{s+=TX(x0-5,Y(v)+4,String(v),'rp-tx rp-mut','end');});
    s+=`<text transform="translate(10 ${f1(py+ph/2)}) rotate(-90)" text-anchor="middle" class="rp-tx rp-mut rp-sm">${esc(tx('score','得分'))}</text>`;
    s+=`<path d="${pts.map((p,k)=>(k?'L':'M')+f1(X(p[0]))+' '+f1(Y(p[1]))).join('')}" class="rp-curve"/>`;
    const vp=A0.head, vm=pair0(2,3,true).head;
    s+=`<g class="rp-pt" data-tipx="${esc(tx(`Distance +1, “dog” one token before “bites”: ${sc2(vp)}, as written and after the move alike.`,`距离 +1，“狗”在“咬”前面一个位置：${sc2(vp)}，挪动前后都一样。`))}"><circle cx="${f1(X(1))}" cy="${f1(Y(vp))}" r="5" class="rp-dot"/></g>`;
    s+=`<g class="rp-pt" data-tipx="${esc(tx(`Distance −1, “dog” one token after “bites”, as in “man bites dog”: ${sc2(vm)}.`,`距离 −1，“狗”在“咬”后面一个位置，也就是“人咬狗”：${sc2(vm)}。`))}"><circle cx="${f1(X(-1))}" cy="${f1(Y(vm))}" r="5" class="rp-dotw"/></g>`;
    s+=TX(X(1)+8,Y(vp)-6,sc2(vp),'rp-tx rp-b')+TX(X(-1),Y(vm)+18,sc2(vm),'rp-tx','middle');
    s+=TX((x0+x1)/2,py+ph+30,tx('distance m − n: + means “dog” comes first','距离 m − n：正数表示“狗”在前'),'rp-tx rp-mut'+(W<480?' rp-sm':''),'middle',2*Math.min((x0+x1)/2,W-(x0+x1)/2)-4);}
  const note=(moved?`<b>${tx(`RoPE: ${sc2(A0.head)} before and after.`,`RoPE：挪动前后都是 ${sc2(A0.head)}。`)}</b> ${tx('With a position vector added at the input, the moved tokens would get new vectors and a new score.','如果在输入处加位置向量，挪动后两个 token 的向量变了，得分也会变。')} `:'')+`<span class="rp-cond">${COND_HEAD()}</span>`;
  return {cap:tx('Pair 0 of “bites” and “dog”, RoPE on: as written and moved later','RoPE 开着：“咬”和“狗”的第 0 对，原文里和往后挪以后'),svg:s+'</svg>',note};
}
/* parts 5 and 6, Scenarios, the challenge: for every pair, how many full turns it makes over the text (log scale); the line at one
   turn splits the pairs that wrap around from those that give every distance its own angle; hollow marks: the turns at the training
   length; a thick stem: past the angles met in training */
function turnsCfg(){
  if(mode==='challenge') return {d:chData(chBase),cap:tx(`Turns of each pair over 1M tokens, base ${fint(chBase)}`,`base ${fint(chBase)} 时，每一对在 1M 个 token 上转几圈`)};
  if(mode==='sandbox') return {d:sbData(SB),cap:tx(`Turns of each pair over ${tokN(SB.ctx)}, base ${fint(SB.base)}${SB.sc?', slow pairs 8 times slower':''}`,`base ${fint(SB.base)}${SB.sc?'、慢的数对慢 8 倍':''}时，每一对在 ${tokN(SB.ctx)}上转几圈`)};
  if(cur().id==='train') return {d:p6Data(),cap:tx(`${M3}${sc6?' with Llama‑3.1’s scaling':''}: turns of each pair over ${tokN(L6[l6])}, with hollow marks at the 8K training length`,`${sc6?'用了 Llama‑3.1 缩放的 ':''}${M3}：每一对在 ${tokN(L6[l6])}上转几圈，空心记号标出 8K 训练长度上的圈数`)};
  return {d:p5Data(),cap:tx(`Turns of each pair over ${tokN(L5[l5])}, base ${fint(base5)}`,`base ${fint(base5)} 时，每一对在 ${tokN(L5[l5])}上转几圈`)};
}
const DECADE={'-4':'0.0001','-2':'0.01','2':'100','4':'10,000'};
/* a speed in degrees per token, 3 significant digits: 57.3, 0.00662, 0.000141 */
const degP=v=>v>=10?fixd(v,1):String(+v.toPrecision(3));
const lamS=lam=>lam<100?fixd(lam,1):bigT(lam);
function ptTip(r,d){
  const lamT=mode==='challenge'?tx(` and needs ${lamS(r.lam)} tokens for one full turn`,`，转一整圈要 ${lamS(r.lam)} 个 token`):'';
  const base=tx(`Pair ${r.i} turns ${expS(r.f,3)} radians (${degP(r.f*R2D)}°) per token${lamT}. Over ${tokW(d.Lx)} it makes ${tS(r.t)} turns: ${r.under?'under one turn, so every distance up to the text length gives it its own angle':'it wraps around'}.`,
    `第 ${r.i} 对每个 token 转 ${expS(r.f,3)} 弧度（${degP(r.f*R2D)}°）${lamT}。在 ${tokW(d.Lx)}上它转 ${tS(r.t)} 圈：${r.under?'不满一圈，所以文本长度以内的每个距离都给它一个不同的角度':'它会转过一整圈'}。`);
  if(r.tt==null) return base;
  return base+(r.tt>=1?tx(' In training it wrapped around, so it met every angle.',' 训练时它就转过一整圈，每个角度都见过。')
    :(r.beyond?tx(` In training it made only ${tS(r.tt)} turns, so it is now past every angle it met.`,` 训练时它只转了 ${tS(r.tt)} 圈，现在已经超出了它见过的所有角度。`):tx(` In training it made ${tS(r.tt)} turns, and it stays within that.`,` 训练时它转了 ${tS(r.tt)} 圈，现在仍在这个范围内。`)));
}
/* the previous state of the chart, kept as faint rings after the learner's own change (A6) */
let turnsLast=null, turnsPrev=null;
function svgTurns(W){
  const {d,cap}=turnsCfg(), T=d.T, H=Math.round(Math.max(270,Math.min(330,W*0.56))), ml=W<480?58:68, mr=8, mt=10, mb=56, Y0=-4, Y1=6;
  const pw=W-ml-mr, X=i=>ml+(i+0.5)/NP*pw, Y=t=>{const v=Math.log10(Math.max(t,1e-4)); return mt+(Y1-Math.min(v,Y1))/(Y1-Y0)*(H-mt-mb);};
  const sig=[mode,mode==='guided'?step:'',d.Lx,d.rows[NP-1].f].join('|'), turns=d.rows.map(r=>r.t);
  if(!turnsLast||turnsLast.sig!==sig){if(turnsLast&&userAct&&turnsLast.ctx===mode+(mode==='guided'?step:'')) turnsPrev=turnsLast.turns; else if(!userAct) turnsPrev=null; turnsLast={sig,turns,ctx:mode+(mode==='guided'?step:'')};}
  let s=svgOpen(W,H,cap);
  s+=`<rect x="${ml}" y="${mt}" width="${f1(pw)}" height="${f1(Y(1)-mt)}" class="rp-wrapbg"/>`;
  for(let e=Y0;e<=Y1;e++){const y=Y(Math.pow(10,e)); s+=`<line x1="${ml}" y1="${f1(y)}" x2="${f1(W-mr)}" y2="${f1(y)}" class="${e===0?'rp-one':'rp-grid'}"/>`;
    if(DECADE[e]) s+=TX(ml-5,y+4,DECADE[e],'rp-tx rp-mut rp-sm','end');}
  s+=TX(ml-5,Y(1)+4,tx('1 turn','1 圈'),'rp-tx rp-b rp-sm','end');
  s+=`<text transform="translate(11 ${f1(mt+(H-mt-mb)/2)}) rotate(-90)" text-anchor="middle" class="rp-tx rp-mut rp-sm">${esc(tx('full turns over the text (log)','在整段文本上转几圈（对数）'))}</text>`;
  s+=TX(ml+6,Y(1)-6,tx('↑ wraps around','↑ 转过一整圈'),'rp-tx rp-mut rp-sm')+TX(ml+6,Y(1)+14,tx('↓ under one turn','↓ 不满一圈'),'rp-tx rp-mut rp-sm');
  if(turnsPrev) turnsPrev.forEach((t,i)=>{s+=`<circle cx="${f1(X(i))}" cy="${f1(Y(t))}" r="3" class="rp-prev"/>`;});
  if(T!=null) d.rows.forEach(r=>{const x=X(r.i); if(r.beyond) s+=`<line x1="${f1(x)}" y1="${f1(Y(r.tt))}" x2="${f1(x)}" y2="${f1(Y(r.t))}" class="rp-stem"/>`;
    s+=`<circle cx="${f1(x)}" cy="${f1(Y(r.tt))}" r="3" class="rp-trn"/>`;});
  const colw=pw/NP;
  d.rows.forEach(r=>{const x=X(r.i);
    s+=`<g class="rp-pt" data-tipx="${esc(ptTip(r,d))}"><rect x="${f1(x-colw/2)}" y="${mt}" width="${f1(colw)}" height="${f1(H-mt-mb)}" class="rp-hit"/><circle cx="${f1(x)}" cy="${f1(Y(r.t))}" r="3.2" class="rp-dot2"/></g>`;});
  const ya=H-mb;
  s+=`<line x1="${ml}" y1="${f1(ya)}" x2="${f1(W-mr)}" y2="${f1(ya)}" class="rp-axis"/>`;
  [0,16,32,48,63].forEach(i=>{s+=`<line x1="${f1(X(i))}" y1="${f1(ya)}" x2="${f1(X(i))}" y2="${f1(ya+4)}" class="rp-axis"/>`+TX(X(i),ya+15,String(i),'rp-tx rp-mut rp-sm','middle');});
  s+=TX(ml,ya+30,tx('← fast pairs','← 快的数对'),'rp-tx rp-mut rp-sm')+TX((ml+W-mr)/2,ya+30,tx('pair i','第 i 对'),'rp-tx rp-mut rp-sm','middle')+TX(W-mr,ya+30,tx('slow pairs →','慢的数对 →'),'rp-tx rp-mut rp-sm','end');
  const spd=v=>W<480?`${degP(v)}°/token`:tx(`${degP(v)}° per token`,`每个 token ${degP(v)}°`);
  s+=TX(ml,ya+45,spd(d.rows[0].f*R2D),'rp-tx rp-sm','start',(W-ml-mr)/2-4)+TX(W-mr,ya+45,spd(d.rows[NP-1].f*R2D),'rp-tx rp-sm','end',(W-ml-mr)/2-4);
  return {cap,svg:s+'</svg>'};
}
function drawVis(){
  const v=$('#rp-vis',app); if(!v) return; const W=visW(), view=viewNow();
  const o=view==='order'?svgOrder(W):(view==='head'?svgHead(W):(view==='pair2'?svgPair2(W):(view==='shift'?svgShift(W):svgTurns(W))));
  const html=`<div class="rp-cap">${esc(o.cap)}</div>${o.svg}${o.note?`<p class="rp-note">${o.note}</p>`:''}`;
  if(v._h!==html){v.innerHTML=html; v._h=html;}
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="rp-sw ${cls}"></i><span>${esc(txt)}</span></span>`, view=viewNow();
  if(view==='order') return it('rp-sw-cell',tx('cell: the score of a query (row) and a key (column)','格子：一个 query（行）和一个 key（列）的得分'),tx('Each cell is the score '+MX('q \\cdot k')+' of the query of its row and the key of its column, over all 128 numbers. Darker cells have larger scores, positive or negative.','每个格子是它那一行的 query 和那一列的 key 的得分 '+MX('q \\cdot k')+'（全部 128 维）。颜色越深，得分的绝对值越大。'))+
    it('rp-sw-row',tx('dashed box: the row of “bites”','虚线框：“咬”那一行'),tx('The dashed box marks the row of “bites”: its query against every key of the sentence.','虚线框标出“咬”那一行：它的 query 和句子里每个 key 的得分。'))+
    it('rp-sw-hot',tx('thick outline: its highest score','粗框：这一行的最高分'),tx('The thick outline marks the highest score in the row of “bites”: the token this head attends to most.','粗框标出“咬”那一行的最高分：这个 head 最关注的那个 token。'));
  if(view==='head') return it('rp-sw-cell',tx('cell: one number','格子：一个数值'),tx('Each cell is one of the 128 numbers of the query of “bites”; darker cells hold larger numbers, positive or negative.','每个格子是“咬”的 query 的 128 个数值之一；颜色越深，数值的绝对值越大。'))+
    (headView!=='128'?it('rp-sw-brk',tx('bracket: a pair','括号：一对'),tx('A bracket joins two neighbouring numbers into a pair; the number under it is the pair’s index.','括号把相邻的两个数连成一对；下面的数字是这一对的编号。')):'')+
    (headView==='arrows'?it('rp-sw-q',tx('arrow: a pair as (x, y)','箭头：一对数值当作 (x, y)'),tx('Each dial draws one pair as an arrow: its first number is x, its second y. The thick dial is pair 0, which the next parts follow.','每个表盘把一对数值画成箭头：第一个数值是 x，第二个是 y。粗圈的是第 0 对，后面几部分跟着它看。')):'')+
    it('rp-sw-pos',tx('thick box: where the position enters','粗框：位置从这里进来'),tx('In the comparison, the thick box is the step where the position enters: added to the embedding once, or turned into q and k in every layer.','对比图里，粗框是位置进入的那一步：在输入处加到 embedding 上一次，或者在每一层转进 q 和 k。'));
  if(view==='pair2'||view==='shift') return it('rp-sw-q',tx('solid: query of “bites”','实线：“咬”的 query'),tx('The solid arrow is pair 0 of the query of “bites”, turned by its position when RoPE is on.','实线箭头是“咬”的 query 的第 0 对；RoPE 开着时按它的位置转过。'))+
    it('rp-sw-k',tx('dashed: key of “dog”','虚线：“狗”的 key'),tx('The dashed arrow is pair 0 of the key of “dog”, turned by its position when RoPE is on.','虚线箭头是“狗”的 key 的第 0 对；RoPE 开着时按它的位置转过。'))+
    it('rp-sw-c',tx('dotted: before the turn','点线：转动之前'),tx('The dotted arrows are the same two pairs before RoPE turns them.','点线箭头是 RoPE 转动之前的这两对数。'))+
    it('rp-sw-shadow',tx('thick gray: this pair’s share','粗灰线：这一对的那份得分'),TIP.shadow())+
    it('rp-sw-arc',tx('arc: the angle between','弧线：两者的夹角'),tx('The arc shows the angle between the two arrows.','弧线表示两个箭头之间的夹角。'))+
    ((view==='pair2'&&rope3)||view==='shift'?it('rp-sw-turn',tx('outer arcs: each arrow’s turn','外圈弧线：每个箭头转了多少'),tx('The arcs outside the dial run from each arrow before the turn to the same arrow after it: solid for the query, dashed for the key. Each is the token’s position × 57.3°.','表盘外的弧线从转动之前的箭头画到转动之后的箭头：实线是 query，虚线是 key。每一段都是 token 的位置 × 57.3°。')):'')+
    (view==='pair2'?it('rp-sw-bar',tx('bars: the row of “bites”, the highest outlined','横条：“咬”那一行，最高的那条描边'),tx('Under each dial, the bars are the whole head’s scores in the row of “bites”: its query against the key of each noun. The outlined bar is the highest.','每个表盘下面的横条，是整个 head 在“咬”那一行的得分：它的 query 和每个名词的 key。描边的那条最高。')):'')+
    (view==='shift'&&pred&&shift>0?it('rp-sw-dot',tx('dots: distance +1 and −1','圆点：距离 +1 和 −1'),tx('On the curve, the filled dot is distance +1 (“dog” one token before “bites”), the hollow dot distance −1 (“dog” one token after).','曲线上，实心圆点是距离 +1（“狗”在“咬”前面一个位置），空心圆点是距离 −1（“狗”在后面一个位置）。')):'');
  const {d}=turnsCfg();
  return it('rp-sw-dot2',tx('dot: turns over the text','圆点：在整段文本上转几圈'),tx('Each dot is one pair: how many full turns it makes over the whole text, on a log scale, fast pairs on the left.','每个圆点是一对：它在整段文本上转几整圈（对数刻度），快的数对在左边。'))+
    (turnsPrev?it('rp-sw-prev',tx('faint rings: before your last change','细线圆圈：上一次改动之前'),tx('The faint rings show where each pair was before your last change, so you can see how far the dots moved.','细线圆圈标出上一次改动之前每一对的位置，可以看出圆点移动了多少。')):'')+
    it('rp-sw-one',tx('line: one full turn','横线：一整圈'),tx('Pairs below the line turn less than once over the text, so every distance up to the text length gives them their own angle. Pairs above it wrap around.','线下面的数对在整段文本上转不满一圈，所以文本长度以内的每个距离都给它们一个不同的角度。线上面的会转过一整圈。'))+
    it('rp-sw-wrap',tx('shaded: wraps around','阴影：转过一整圈'),tx('In the shaded band a pair turns more than once over the text: it still tells near neighbours apart, but repeats over long distances.','阴影里的数对在整段文本上不止转一圈：它仍能分辨相邻的 token，但距离远了就会重复。'))+
    (d.T!=null?it('rp-sw-trn',tx('hollow: turns at the training length','空心：训练长度上转的圈数'),TIP.seen())+
      it('rp-sw-stem',tx('thick stem: past the angles met in training','粗竖线：超出训练时见过的角度'),tx('A thick stem joins a pair’s training mark to where it is now, for a pair that never wrapped around in training: it now turns further than it ever did.','粗竖线把一对数的训练记号和它现在的位置连起来，只画在训练时没转过一整圈的数对上：它现在转得比训练时任何时候都远。')):'');
}

/* metric cards: ▲▼ only after the learner's own change, from the numbers as shown (R6.6) */
function metricsDef(){
  const view=viewNow();
  if(view==='order'){const rope=ropeOn(), a=grid('a',rope)[1][0], b=grid('b',rope)[1][2], df=Math.abs(+sc2(a).replace(MINUS,'-')-(+sc2(b).replace(MINUS,'-')));
    return [{k:'sa',label:tx('bites → dog in “dog bites man”','“狗咬人”里的“咬 → 狗”'),v:sc2(a),sub:rope?tx('bites at 2, dog at 1','“咬”在 2，“狗”在 1'):tx('no positions','没有位置'),tip:tx(`The head’s score of the query of “bites” and the key of “dog” in “dog bites man”: ${sc2(a)}.`,`“狗咬人”里“咬”的 query 和“狗”的 key 在这个 head 里的得分：${sc2(a)}。`)},
      {k:'sb',label:tx('bites → dog in “man bites dog”','“人咬狗”里的“咬 → 狗”'),v:sc2(b),sub:rope?tx('bites at 2, dog at 3','“咬”在 2，“狗”在 3'):tx('no positions','没有位置'),tip:tx(`The head’s score of the query of “bites” and the key of “dog” in “man bites dog”: ${sc2(b)}.`,`“人咬狗”里“咬”的 query 和“狗”的 key 在这个 head 里的得分：${sc2(b)}。`)},
      {k:'sd',nod:!rope,label:tx('Difference between the orders','两种顺序的差'),v:sc2(df),sub:rope?tx('RoPE on','有 RoPE'):tx('the head cannot tell','head 分不出'),tip:tx(`The two scores differ by ${sc2(df)}: ${rope?'with RoPE the order matters.':'without positions the order cannot matter.'}`,`两个得分相差 ${sc2(df)}：${rope?'有了 RoPE，顺序就起作用了。':'没有位置，顺序就不起作用。'}`)}];}
  if(view==='head') return [];   // part 2: the drawing and the comparison need the stage's height; its numbers are in the tooltips
  if(view==='pair2'){const a=pair0(2,1,rope3), b=pair0(2,3,rope3), df=Math.abs(numOf(sc2(a.head))-numOf(sc2(b.head)));
    const sh=p=>tx(`pair 0: ${sc2(p.share)}`,`第 0 对：${sc2(p.share)}`);
    const tp=(p,nm)=>tx(`The head’s score of bites → dog in “${nm.en}”, all 64 pairs: ${sc2(p.head)}. Pair 0 alone gives 1.5 × 1.3 × cos(${dS(p.between)}) = ${sc2(p.share)}.`,`“${nm.zh}”里这个 head 给“咬 → 狗”的得分（全部 64 对）：${sc2(p.head)}。单看第 0 对：1.5 × 1.3 × cos(${dS(p.between)}) = ${sc2(p.share)}。`);
    return [{k:'pa',label:tx('bites → dog in “dog bites man”','“狗咬人”里的“咬 → 狗”'),v:sc2(a.head),sub:sh(a),tip:tp(a,{en:'dog bites man',zh:'狗咬人'})},
      {k:'pb',label:tx('bites → dog in “man bites dog”','“人咬狗”里的“咬 → 狗”'),v:sc2(b.head),sub:sh(b),tip:tp(b,{en:'man bites dog',zh:'人咬狗'})},
      {k:'pd',label:tx('Difference between the orders','两种顺序的差'),v:sc2(df),sub:rope3?tx('RoPE on','RoPE 开'):tx('RoPE off: the head cannot tell','RoPE 关：head 分不出'),tip:tx(`The head’s two scores differ by ${sc2(df)}: ${rope3?'with RoPE the order matters.':'without positions the order cannot matter.'}`,`这个 head 的两个得分相差 ${sc2(df)}：${rope3?'有了 RoPE，顺序就起作用了。':'没有位置，顺序就不起作用。'}`)}];}
  if(view==='shift'){const b=pair0(2+shift,1+shift,true), hid=!pred;
    return [{k:'dm',nod:true,label:tx('Distance m − n','距离 m − n'),v:String(b.m-b.n),sub:`${b.m} − ${b.n}`,tip:tx(`m is the position of the query (“bites”, at ${b.m}) and n the position of the key (“dog”, at ${b.n}): the distance m − n is ${b.m-b.n}.`,`m 是 query（“咬”，在 ${b.m}）的位置，n 是 key（“狗”，在 ${b.n}）的位置：距离 m − n 是 ${b.m-b.n}。`)},
      {k:'ab',label:tx('Angle between, pair 0','第 0 对的夹角'),v:hid?'?':dS(Math.abs(b.between)),sub:hid?tx('after your prediction','预测之后显示'):tx('after the move','挪动以后'),tip:hid?tx('The angle between the two arrows after the move stays hidden until you answer.','挪动以后两个箭头的夹角，你作答之前先隐藏。'):tx(`The query points at ${dS(wrapDeg(b.qa))} and the key at ${dS(wrapDeg(b.ka))}: ${dS(Math.abs(b.between))} apart.`,`query 指向 ${dS(wrapDeg(b.qa))}，key 指向 ${dS(wrapDeg(b.ka))}：相差 ${dS(Math.abs(b.between))}。`)},
      {k:'sc',label:tx('The head’s score','这个 head 的得分'),v:hid?'?':sc2(b.head),sub:hid?tx('after your prediction','预测之后显示'):tx('64 pairs added up','64 对加起来'),tip:hid?tx('The head’s score after the move stays hidden until you answer.','挪动以后这个 head 的得分，你作答之前先隐藏。'):tx(`The head’s score of bites → dog with “bites” at ${b.m} and “dog” at ${b.n}: ${sc2(b.head)}.`,`“咬”在 ${b.m}、“狗”在 ${b.n} 时，这个 head 里“咬 → 狗”的得分：${sc2(b.head)}。`)}];}
  const {d}=turnsCfg();
  const tV=tS;   // turns: 2.41, 0.41, then 3 significant digits: 0.0512, 0.00320 (S15)
  const mU={k:'nu',label:tx('Pairs under one turn','不满一圈的数对'),v:tx(`${d.nUnder} of 64`,`${d.nUnder} / 64`),sub:tx('below the line','线下面'),tip:tx(`${d.nUnder} of the 64 pairs turn less than once over ${tokN(d.Lx)}; the other ${d.nWrapped} wrap around.`,`在 ${tokN(d.Lx)}上，64 对里有 ${d.nUnder} 对转不满一圈，另外 ${d.nWrapped} 对会转回原处。`)};
  const mS={k:'sl',raw:d.slow.t,label:tx('Turns of the slowest pair','最慢的一对转的圈数'),v:tV(d.slow.t),sub:d.slow.t<1?tx(`${fixd(d.slow.t*360,1)}° over ${tokN(d.Lx)}`,`在 ${tokN(d.Lx)}上转 ${fixd(d.slow.t*360,1)}°`):tx(`over ${tokN(d.Lx)}`,`在 ${tokN(d.Lx)}上`),tip:tx(`Pair 63 turns ${expS(d.slow.f,3)} radians per token; over ${tokW(d.Lx)} that is ${tS(d.slow.t)} turns, ${turnS(d.slow.t)}.`,`第 63 对每个 token 转 ${expS(d.slow.f,3)} 弧度；${tokW(d.Lx)}就是 ${tS(d.slow.t)} 圈，即 ${turnS(d.slow.t)}。`)};
  const mL={k:'lam',label:tx('Tokens for one turn of the slowest','最慢的一对转一圈要的 token'),v:bigT(d.slow.lam),sub:tx('tokens for one full turn','个 token 转一整圈'),tip:tx(`Pair 63 needs ${MX('\\frac{2\\pi}{\\theta_{63}}')} = ${fint(Math.round(d.slow.lam))} tokens for one full turn.`,`第 63 对转满一圈要 ${MX('\\frac{2\\pi}{\\theta_{63}}')} = ${fint(Math.round(d.slow.lam))} 个 token。`)};
  const mB=d.T!=null?{k:'nb',label:tx('Pairs past their training angles','超出训练角度的数对'),v:tx(`${d.nBeyond} of 64`,`${d.nBeyond} / 64`),sub:tx(`trained on ${tokN(d.T)}`,`训练长度 ${tokN(d.T)}`),tip:tx(`${d.nBeyond} of the 64 pairs turn further over ${tokN(d.Lx)} than they ever did in training on ${tokN(d.T)}.`,`64 对里有 ${d.nBeyond} 对在 ${tokN(d.Lx)}上转得比在 ${tokN(d.T)}的训练中任何时候都远。`)}:null;
  const mT={k:'tl',nod:true,label:tx('Text length L','文本长度 L'),v:tokK(d.Lx),sub:tx('tokens in the text','文本里的 token 数'),tip:tx(`The text is ${tokW(d.Lx)} long.`,`文本长 ${tokW(d.Lx)}。`)};
  if(mode==='challenge') return [{k:'cb',nod:true,label:tx('RoPE base','RoPE base'),v:fint(chBase),sub:tx('your pick','你的选择'),tip:tx(`The base you picked. With it, pair 63, the slowest, turns ${expS(chData(chBase).slow.f,3)} radians per token (the formula is in “Show the math”).`,`你选的 base。用它时，最慢的第 63 对每个 token 转 ${expS(chData(chBase).slow.f,3)} 弧度（公式见“展开公式”）。`)},{...mS,raw:chTurns(chBase),label:tx('Turns of the slowest pair','最慢的一对转的圈数'),v:tV(chTurns(chBase)),sub:tx('over 1M tokens; goal: under 1','在 1M 个 token 上；目标：不到 1')},mL];
  if(mode==='guided'&&cur().id==='speeds') return [mT,mU,mS];
  if(mode==='guided'&&cur().id==='train') return [mT,mB,{...mS,sub:d.nBeyond?tx(`${fint(Math.round(d.ratio))}× its training turns`,`训练时的 ${fint(Math.round(d.ratio))} 倍`):tx('within training','在训练范围内')}];
  return [mU,mS,mB];
}
const numOf=v=>{const m=String(v).replace(/,/g,'').replace(MINUS,'-').match(/^-?\d+(?:\.\d+)?/); return m?parseFloat(m[0]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function drawMetrics(){
  const host=$('#rp-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  host.hidden=!defs.length;
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!m.nod&&prev!=null&&prev!==m.v&&prev!=='?'&&m.v!=='?'){const exact=m.raw!=null&&shownRaw[m.k]!=null, a=exact?shownRaw[m.k]:numOf(prev), b=exact?m.raw:numOf(m.v), sameU=String(prev).replace(/[-−\d.,\s]/g,'')===String(m.v).replace(/[-−\d.,\s]/g,'');
      if(isFinite(a)&&isFinite(b)&&sameU&&a>=0&&b>=0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v; shownRaw[m.k]=m.raw!=null?m.raw:null;});
}
function statusOf(){
  const view=viewNow();
  if(mode==='challenge'){const t=chTurns(chBase), ok=t<1;
    return {cls:ok?'ok':'bad',txt:ok?tx(`Under one turn: ${turnsS(t)}`,`不满一圈：${turnsS(t)}`):tx(`Wraps around: ${turnsS(t)}`,`转过一整圈：${turnsS(t)}`),short:ok?tx('Under 1 turn','不满一圈'):tx('Wraps around','转过一整圈'),
      tip:tx(`The badge turns green when the slowest pair turns less than one full turn over the 1M-token text; now base ${fint(chBase)}.`,`最慢的一对在 1M token 的文本上转不满一圈时，徽章变成绿色；现在是 base ${fint(chBase)}。`)};}
  if(view==='order'){const r=ropeOn(); return {cls:'neutral',txt:r?tx('Scores depend on the order','得分随顺序变化'):tx('Same scores in both orders','两种顺序得分相同'),short:r?tx('Order matters','顺序起作用'):tx('Same scores','得分相同'),
    tip:r?tx('With RoPE, the score of bites → dog is different in the two sentences.','有了 RoPE，“咬 → 狗”在两句话里得分不同。'):tx('Without positions, every score of one sentence also appears in the other, with the same words.','没有位置时，一句话里的每个得分，同样的两个词在另一句里也一样。')};}
  if(view==='head'){const t={'128':tx('128 numbers','128 个数值'),pairs:tx('64 pairs','64 对'),arrows:tx('64 arrows','64 个箭头')}[headView];
    return {cls:'neutral',txt:t,short:t,tip:tx(`The query of “bites” is drawn now as ${t}; the control in the card switches between 128 numbers, 64 pairs and 64 arrows.`,`“咬”的 query 现在画成${t}；卡片里的按钮在 128 个数值、64 对和 64 个箭头之间切换。`)};}
  if(view==='pair2') return {cls:'neutral',txt:rope3?tx('RoPE on: scores differ','RoPE 开：得分不同'):tx('RoPE off: same scores','RoPE 关：得分相同'),short:rope3?tx('RoPE on','RoPE 开'):tx('RoPE off','RoPE 关'),
    tip:rope3?tx('With RoPE on, pair 0 and the whole head score bites → dog differently in the two sentences.','RoPE 开着时，第 0 对和整个 head 在两句话里给“咬 → 狗”的得分不同。'):tx('With RoPE off, the two sentences give the same arrows and the same scores.','RoPE 关着时，两句话里箭头一样，得分也一样。')};
  if(view==='shift'){if(!pred) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The moved dial and its score stay hidden until you answer the question in this part.','回答这一部分的问题之前，挪动后的表盘和得分先隐藏。')};
    const b=pair0(2+shift,1+shift,true); return {cls:'neutral',txt:tx(`Score ${sc2(b.head)}`,`得分 ${sc2(b.head)}`),short:tx(`Score ${sc2(b.head)}`,`得分 ${sc2(b.head)}`),tip:tx(`The head’s score of bites → dog with “bites” at ${b.m} and “dog” at ${b.n}.`,`“咬”在 ${b.m}、“狗”在 ${b.n} 时，这个 head 里“咬 → 狗”的得分。`)};}
  const {d}=turnsCfg();
  if(d.T!=null&&(mode==='sandbox'||cur().id==='train')) return {cls:'neutral',txt:d.nBeyond?tx(`${d.nBeyond} pairs past training`,`${d.nBeyond} 对超出训练`):tx('Within training','在训练范围内'),short:d.nBeyond?tx('Past training','超出训练'):tx('Within training','在训练内'),
    tip:tx(`${d.nBeyond} of the 64 pairs turn past every angle they met in training on ${tokN(d.T)}.`,`64 对里有 ${d.nBeyond} 对转过了 ${tokN(d.T)}训练中见过的所有角度。`)};
  return {cls:'neutral',txt:tx(`${d.nUnder} of 64 under one turn`,`${d.nUnder} / 64 对不满一圈`),short:tx(`${d.nUnder} under 1 turn`,`${d.nUnder} 对不满一圈`),tip:tx(`${d.nUnder} of the 64 pairs turn less than once over ${tokN(d.Lx)}; the other ${d.nWrapped} wrap around.`,`在 ${tokN(d.Lx)}上，64 对里有 ${d.nUnder} 对转不满一圈，另外 ${d.nWrapped} 对会转回原处。`)};
}
function drawHead(){
  const view=viewNow(), t=$('#rp-st-title',app), sub=$('#rp-st-sub',app), st=$('#rp-status',app);
  const TT={order:[tx(`${M3}: one head`,`${M3}：一个 head`),ropeOn()?tx('The scores with RoPE, in both orders','有 RoPE 时两种顺序的得分'):tx('The scores without position information, in both orders','没有位置信息时两种顺序的得分')],
    head:[tx(`${M3}: one head`,`${M3}：一个 head`),tx('What RoPE turns: 128 numbers in 64 pairs','RoPE 转的是什么：128 维，64 对')],
    pair2:[tx('Pair 0: 57.3° per position','第 0 对：每个位置 57.3°'),tx('“dog bites man” and “man bites dog”','“狗咬人”和“人咬狗”')],
    shift:[tx('Pair 0, RoPE on','第 0 对，RoPE 开着'),tx('“dog bites man”, moved later','“狗咬人”往后挪')]};
  if(view==='turns'){
    const m=mode==='challenge'?tx('A model with 128-number heads, texts of 1M tokens','head 为 128 维的模型，1M token 的文本'):(mode==='sandbox'?(presetNow()>=0?L(PRESETS[presetNow()].t):tx('Your settings','你的设置')):(cur().id==='train'?(sc6?tx(`${M3} with Llama‑3.1’s scaling, trained on 8K`,`用了 Llama‑3.1 缩放的 ${M3}，训练长度 8K`):M3):tx(`Heads of 128 numbers, base ${fint(base5)}`,`head 为 128 维，base ${fint(base5)}`)));
    t.textContent=tx('Base, speed and text length in one chart','base、转速和文本长度在一张图上'); sub.textContent=m;}
  else {t.textContent=TT[view][0]; sub.textContent=TT[view][1];}
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const sc=$('#rp-scope',app); if(sc){const early=mode==='guided'&&step<2;
    sc.textContent=early?tx('Made-up vectors in one head of 128 numbers, as in Llama‑3‑8B.','向量是虚构的：一个 head，128 维，和 Llama‑3‑8B 一样。'):tx('Made-up vectors, real speeds: the speeds are Llama’s.','向量是虚构的，转速是真的：转速来自 Llama。');
    sc.dataset.tipx=early?tx('The vectors and the scores come from a head made up by hand, with 128 numbers per query and key, as in Llama‑3‑8B. From part 3 on, the turns use Llama’s real speeds.','向量和得分来自一个手工虚构的 head，每个 query 和 key 有 128 维，和 Llama‑3‑8B 一样。从第 3 部分起，转动用的是 Llama 的真实转速。')
      :tx('The vectors and the scores come from a head made up by hand: 128 numbers per query and key. The speeds come from the Llama models’ config files: pair 0 turns 1 radian per token, and each later pair a little less.','向量和得分来自一个手工虚构的 head：每个 query 和 key 有 128 维。转速来自 Llama 模型的配置文件：第 0 对每个 token 转 1 弧度，后面的一对比一对少一点。');}
}
/* Show the math: every line redoes from the numbers it shows (R6.6), every term has a tooltip (R1.4) */
const mathBase=()=>mode==='challenge'?chBase:(mode==='sandbox'?SB.base:(mode==='guided'&&cur().id==='train'?500000:base5));
const mathScaled=()=>(mode==='sandbox'&&SB.sc)||(mode==='guided'&&cur().id==='train'&&sc6);
/* the owner's decisions of 2026-10-09 (H1, H2): each block says in words what it computes, gives the formula typeset (A.ui.math), then the
   numbers of the current setting (terms wired to their tooltips) and the result, one idea per block, no run-on “= … ; … =” lines */
const MX=t=>A.ui.math(t);
/* the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): the formula is set as its name with “=”,
   then the right-hand side, which goes on its own line under the name when the two do not fit side by side. The right-hand side is cut
   before each + and − outside brackets into terms that wrap onto the next line under its first term; a term still wider than the box
   is cut again before ×. Every piece starts with an empty letter (\mathrm{}), and the name ends with one after its “=”, so an operator at
   the start or end of a piece (+, ×, Σ, =) keeps the spacing it has inside one formula (Chromium gives an mrow's first or last operator
   no space; an empty {} or a space does not count) */
const FWcut=(t,ops)=>{const out=[]; let d=0, last=0;
  for(let i=0;i<t.length;i++){const c=t[i];
    if(c==='{'||c==='(') d++; else if(c==='}'||c===')') d--;
    else if(d===0&&i>last&&ops.some(o=>t.startsWith(o,i))){out.push(t.slice(last,i)); last=i+1;}}
  out.push(t.slice(last)); return out;};
const FW=tex=>{const [name,...rest]=FWcut(tex,[' = ',' \\approx ']); if(!rest.length) return A.ui.math(tex);
  const rel=rest[0].startsWith('=')?'=':'\\approx', rhs=rest.join(' ').slice(rel.length+1);
  const piece=x=>`<span class="rp-fp">${A.ui.math('\\mathrm{} '+x)}</span>`;
  return `<span class="rp-fwn">${A.ui.math(name+' '+rel+' \\mathrm{}')}</span><span class="rp-fwr">${FWcut(rhs,[' + ',' - ']).map(g=>`<span class="rp-fg">${FWcut(g,[' \\times ']).map(piece).join('<wbr>')}</span>`).join('<wbr>')}</span>`;};
/* two formulas side by side (“q = (…), pair i = (…)”) wrap as two: the second goes onto the next line when they do not fit */
const FW2=tex=>{const p=tex.split(', \\quad '); return p.length<2?FW(tex):p.map((x,k)=>`<span class="rp-fp${k<p.length-1?' rp-fq':''}">${MX(x+(k<p.length-1?',':''))}</span>`).join('<wbr>');};
/* a space at the edge of a \text{…} is dropped by MathML, so “turns over L” read “turns overL”: a medium space stands beside it */
const SPX=t=>t.replace(/\\text\{([^}]*) \}/g,'\\text{$1 }\\:').replace(/\\text\{ ([^}]*)\}/g,'\\:\\text{ $1}');
const FL=f=>{const h=FW2(SPX(f)); return `<div class="${h.includes('rp-fwn')?'rp-fw ':''}rp-math-line">${h}</div>`;};
const MB=(lead,f,body)=>`<div class="rp-mb">${lead?`<div class="rp-mlead">${lead}</div>`:''}${[].concat(f||[]).map(FL).join('')}${body}</div>`;
const PAIR0F=()=>tx('\\text{pair 0} = |q| \\times |k| \\times \\cos(\\text{angle between})','\\text{第 0 对} = |q| \\times |k| \\times \\cos(\\text{夹角})');
function mathHTML(){
  const view=viewNow(), t=T_;
  const q=P0.q.bites, k=P0.k.dog, TH=t('th0',`${fixd(TH0D,2)}°`), LQ=t('lq',String(q[0])), LK=t('lk',String(k[0]));
  if(view==='order'){const rope=ropeOn();
    if(!rope){const tot=score('bites','dog',0,0,false);
      return MB(tx('Without RoPE, the score is the dot product of the query and the key: the 128 products of their numbers, added up.','不加 RoPE 时，得分就是 query 和 key 的点积：两者的 128 维逐维相乘，再加起来。'),
        tx('\\text{score} = q \\cdot k = q_0 k_0 + q_1 k_1 + \\ldots + q_{127} k_{127}','\\text{得分} = q \\cdot k = q_0 k_0 + q_1 k_1 + \\ldots + q_{127} k_{127}'),
        `<p>${tx('bites → dog, in both orders','“咬 → 狗”，两种顺序都是')}: <b>${sc2(tot)}</b></p>`);}
    const line=(nm,m,n)=>{const s0=share('bites','dog',0,m,n,true), tot=score('bites','dog',m,n,true), rest=tot-s0;
      return `<p class="rp-msub">${nm}</p><p>${tx('pair 0','第 0 对')} = ${LQ} × ${LK} × cos(${t('d0',dS(q[1]-k[1]))} ${m-n>=0?'+':'−'} ${t('mn',String(Math.abs(m-n)))} × ${TH}) = ${sc2(s0)}</p><p>${tx('pairs 1 to 63','第 1 到 63 对')} = ${sc2(rest)}</p><p>${tx('total','合计')} = <b>${sc2(tot)}</b></p>`;};
    return MB(tx('With RoPE, the score adds up the 64 pairs of numbers, each pair read as an arrow.','加了 RoPE，得分就是 64 对数各自的贡献加起来，每一对看成一个箭头。'),
        tx('\\text{score} = \\sum_{i=0}^{63} (q_{2i} k_{2i} + q_{2i+1} k_{2i+1})','\\text{得分} = \\sum_{i=0}^{63} (q_{2i} k_{2i} + q_{2i+1} k_{2i+1})'),'')+
      MB(tx('A pair’s share is the two arrows’ lengths times the cosine of the angle between them. RoPE turns the query by '+MX('m \\times \\theta_0')+' and the key by '+MX('n \\times \\theta_0')+', so the angle between them changes by '+MX('(m - n) \\times \\theta_0')+'.','一对的贡献，是两个箭头的长度相乘，再乘以它们夹角的余弦。RoPE 把 query 转 '+MX('m \\times \\theta_0')+'，把 key 转 '+MX('n \\times \\theta_0')+'，所以夹角变了 '+MX('(m - n) \\times \\theta_0')+'。'),
        PAIR0F(),
        line(tx('“dog bites man”, m = 2, n = 1','“狗咬人”，m = 2，n = 1'),2,1)+line(tx('“man bites dog”, m = 2, n = 3','“人咬狗”，m = 2，n = 3'),2,3)+
        `<p class="small">${tx('Pairs 1 to 63 of the key of “dog” are set θᵢ ahead of the query’s pair i, so at distance +1 they line up. Pair 0 is drawn by hand.','“狗”的 key 的第 1 到 63 对，比 query 的第 i 对多转 θᵢ，所以在距离 +1 时方向一致。第 0 对是手画的。')}</p>`);}
  if(view==='head'){const v=QV.bites, s0=share('bites','dog',0,0,0,false), tot=score('bites','dog',0,0,false);
    return MB(tx('A query of 128 numbers is read as 64 pairs. Each pair is an arrow in a plane: its length and its direction.','128 维的 query 按每两个数一对，分成 64 对。每一对是平面上的一个箭头，有长度，也有方向。'),
        tx('q = (q_0, q_1, \\ldots, q_{127}), \\quad \\text{pair } i = (q_{2i}, q_{2i+1})','q = (q_0, q_1, \\ldots, q_{127}), \\quad \\text{第 } i \\text{ 对} = (q_{2i}, q_{2i+1})'),
        `<p>${tx('pair 0 of “bites”','“咬”的第 0 对')} = (${n3(v[0])}, ${n3(v[1])}) = ${LQ} × (cos ${t('q0a',dS(q[1]))}, sin ${t('q0a',dS(q[1]))})</p>`)+
      MB(tx('The score adds up what every pair contributes.','得分是每一对的贡献加起来。'),null,
        `<p>${tx('score (bites → dog)','得分（“咬 → 狗”）')} = ${sc2(s0)} (${tx('pair 0','第 0 对')}) + ${sc2(tot-s0)} (${tx('pairs 1 to 63','第 1 到 63 对')}) = <b>${sc2(tot)}</b></p>`);}
  if(view==='pair2'){const a=pair0(2,1,rope3), b=pair0(2,3,rope3);
    if(!rope3) return MB(tx('Without RoPE, the angle between the two arrows is the same in both sentences, so pair 0 gives the same share.','不加 RoPE 时，两个箭头的夹角在两句里一样，所以第 0 对的贡献也一样。'),
        PAIR0F(),
        `<p>${tx('angle between','夹角')} = ${t('q0a',dS(q[1]))} − ${t('k0a',dS(k[1]))} = ${dS(a.between)}</p><p>${tx('pair 0','第 0 对')} = ${LQ} × ${LK} × cos(${dS(a.between)}) = <b>${sc2(a.share)}</b>${tx(' in both sentences','，两句都是')}</p>`);
    return MB(tx('RoPE turns pair 0 by '+MX('\\theta_0')+' = 1 radian for every position, so an arrow’s angle grows with its position.','RoPE 让第 0 对每过一个位置转 '+MX('\\theta_0')+' = 1 弧度，所以箭头的角度随位置增大。'),
        tx('\\text{angle at position } m = \\text{angle without RoPE} + m \\times \\theta_0','\\text{位置 } m \\text{ 上的角度} = \\text{不加 RoPE 时的角度} + m \\times \\theta_0'),
        `<p>θ₀ = 1 ${tx('radian','弧度')} = ${TH}</p>`+
        `<p>${tx('query of “bites” at 2','“咬”的 query 在 2')} = ${t('q0a',dS(q[1]))} + ${t('rope3','2')} × ${TH} = ${dS(a.qa)}</p>`+
        `<p>${tx('key of “dog” at 1 (“dog bites man”)','“狗”的 key 在 1（“狗咬人”）')} = ${t('k0a',dS(k[1]))} + ${t('rope3','1')} × ${TH} = ${dS(a.ka)}</p>`+
        `<p>${tx('key of “dog” at 3 (“man bites dog”)','“狗”的 key 在 3（“人咬狗”）')} = ${t('k0a',dS(k[1]))} + ${t('rope3','3')} × ${TH} = ${dS(b.ka)}${tx(', that is ','，即 ')}${dS(wrapDeg(b.ka))}</p>`)+
      MB(tx('The two sentences now give pair 0 different angles between the arrows, so different shares.','两句里第 0 对的夹角不同了，贡献也就不同。'),
        PAIR0F(),
        `<p>${tx('“dog bites man”','“狗咬人”')}: ${LQ} × ${LK} × cos(${dS(a.between)}) = <b>${sc2(a.share)}</b></p><p>${tx('“man bites dog”','“人咬狗”')}: ${LQ} × ${LK} × cos(${dS(b.between)}) = <b>${sc2(b.share)}</b></p>`+
        `<p>${tx('whole head, 64 pairs','整个 head，64 对')}: <b>${sc2(a.head)}</b> ${tx('and','和')} <b>${sc2(b.head)}</b></p>`);}
  if(view==='shift'){if(!pred) return `<p class="muted">${tx('The lines appear after your prediction.','预测之后显示算式。')}</p>`;
    const b=pair0(2+shift,1+shift,true);
    return MB(`${tx('Move both words: the query at m turns '+MX('m \\times \\theta_0')+', the key at n turns '+MX('n \\times \\theta_0')+'. ','两个词一起挪：位置 m 上的 query 转 '+MX('m \\times \\theta_0')+'，位置 n 上的 key 转 '+MX('n \\times \\theta_0')+'。')}${t('phi','φ')}${tx(' is an arrow’s direction before the turn. In the angle between them, m and n appear only as '+MX('m - n')+'.',' 是箭头转动之前的方向。两者的夹角里，m 和 n 只以 '+MX('m - n')+' 出现。')}`,
        tx('\\text{angle between} = (\\phi_q + m\\,\\theta_0) - (\\phi_k + n\\,\\theta_0) = \\phi_q - \\phi_k + (m - n)\\,\\theta_0','\\text{夹角} = (\\phi_q + m\\,\\theta_0) - (\\phi_k + n\\,\\theta_0) = \\phi_q - \\phi_k + (m - n)\\,\\theta_0'),
        `<p>${tx('angle between','夹角')} = ${t('d0',dS(q[1]-k[1]))} + (${t('shift',String(b.m))} − ${t('shift',String(b.n))}) × ${TH} = <b>${dS(b.between)}</b></p><p>${tx(`So the arrows are ${dS(Math.abs(b.between))} apart.`,`所以两个箭头相差 ${dS(Math.abs(b.between))}。`)}</p>`)+
      MB(tx('Every pair works the same way with its own speed θᵢ, so moving both words by the same amount changes no angle and no score.','每一对都一样，只是转速 θᵢ 不同。所以两个词挪同样多，哪个夹角都不变，得分也不变。'),
        tx('\\text{angle between}_i = \\phi_{q,i} - \\phi_{k,i} + (m - n)\\,\\theta_i','\\text{夹角}_i = \\phi_{q,i} - \\phi_{k,i} + (m - n)\\,\\theta_i'),
        `<p>${tx('whole head','整个 head')} = <b>${sc2(b.head)}</b></p>`);}
  const {d}=turnsCfg(), base=mathBase(), scl=mathScaled();
  const bk=mode==='challenge'?'chbase':(mode==='sandbox'?'sbase':(cur().id==='train'?'base5f':'b5'));
  const lk=mode==='challenge'?'chL':(mode==='sandbox'?'sctx':(cur().id==='train'?'l6':'l5'));
  let h=MB(tx('Pair i turns θᵢ radians per token. Each later pair turns a little slower, and the base sets how slow the last ones get.','第 i 对每个 token 转 θᵢ 弧度。越往后的数对转得越慢，base 决定最后几对慢到什么程度。')+(scl?tx(' Llama 3.1’s “llama3” scaling then makes the slow pairs 8 times slower.',' Llama 3.1 的“llama3”缩放再让慢的数对慢 8 倍。'):''),
    '\\theta_i = \\text{base}^{-2i/128}',
    `<p>base = ${t(bk,fint(base))}</p>`);
  h+=MB(tx('Over a text of L tokens, pair i turns '+MX('L \\times \\theta_i')+' radians. One full turn is '+MX('2\\pi')+' radians.','在 L 个 token 的文本上，第 i 对一共转 '+MX('L \\times \\theta_i')+' 弧度。转一整圈是 '+MX('2\\pi')+' 弧度。'),
    tx('\\text{turns over } L = \\frac{L \\times \\theta_i}{2\\pi}','\\text{在 } L \\text{ 上转的圈数} = \\frac{L \\times \\theta_i}{2\\pi}'),
    `<p>${tx('pair 63','第 63 对')} = ${t(lk,fint(d.Lx))} × ${t('th63',expS(d.slow.f,4))} ÷ ${t('tau',fixd(TAU,4))} = <b>${tS(d.slow.t)}</b> ${tx('turns','圈')}</p>`);
  if(mode==='challenge') h+=MB(tx('The slowest pair needs this many tokens for one full turn.','最慢的一对转一整圈要这么多个 token。'),
    tx('\\text{tokens for one turn} = \\frac{2\\pi}{\\theta_{63}}','\\text{转一圈要的 token 数} = \\frac{2\\pi}{\\theta_{63}}'),
    `<p>= ${t('tau',fixd(TAU,4))} ÷ ${t('th63',expS(d.slow.f,4))} = <b>${fint(Math.round(d.slow.lam))}</b> ${tx('tokens','个 token')}</p>`);
  h+=MB(tx('A pair that turns less than once over the text never comes back to an angle it had.','在整段文本上转不满一圈的数对，不会转回到出现过的角度。'),
    tx('\\text{under one turn: } L \\times \\theta_i < 2\\pi','\\text{不满一圈：} L \\times \\theta_i < 2\\pi'),
    `<p>${tx('pairs under one turn','不满一圈的数对')}: <b>${d.nUnder}</b> ${tx('of 64','/ 64')}${d.nUnder?tx(`, pairs ${d.first} to 63`,`，第 ${d.first} 到第 63 对`):''}</p>`);
  if(d.T!=null) h+=MB(tx('In training the model only saw texts up to the training length, so compare how far the slowest pair turns there.','训练时模型只见过不超过训练长度的文本，所以拿最慢的一对在训练长度上转了多少来比。')+(scl&&mode==='guided'?tx(' This uses the unscaled θᵢ.',' 这里用不缩放的 θᵢ。'):''),
    tx('\\text{turns at the training length} = \\frac{\\text{training length} \\times \\theta_i}{2\\pi}','\\text{训练长度上的圈数} = \\frac{\\text{训练长度} \\times \\theta_i}{2\\pi}'),
    `<p>${tx('training length','训练长度')} = ${t(mode==='sandbox'?'strain':'train8',fint(d.T))}</p><p>${tx('pair 63','第 63 对')}: ${tS(d.slow.tt)} ${tx('turns','圈')}</p>${d.ratio&&d.slow.beyond?`<p>${tx('now ÷ in training','现在 ÷ 训练时')} = ${tS(d.slow.t)} ÷ ${tS(d.slow.tt)} = <b>${fixd(d.ratio,1)}</b></p>`:''}`);
  return h;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=cur(); if(s&&s.expect&&s.expect()&&(!s.gate||s.gate())) observed[step]=true;}
  drawHead(); drawVis(); drawMetrics();
  const lg=$('#rp-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#rp-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  const gr=$('#rp-graph',app); gr.dataset.view=viewNow();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  const rng=(key,val,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==val) i.value=val; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  rng('shift',shift,posN(shift)); rng('l5',l5,tokN(L5[l5])); rng('l6',l6,tokN(L6[l6])); rng('sctx',CTX.indexOf(SB.ctx),tokN(SB.ctx));
  const segs={sbase:String(SB.base),ssc:SB.sc?'1':'0',strain:String(SB.train),hv:headView,rope3:rope3?'on':'off',b5:String(base5),s6:sc6?'1':'0'};
  Object.entries(segs).forEach(([k,v])=>{$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,v));});
  const lock=mode!=='sandbox', c=$('#rp-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#rp-lock',app); ln.hidden=!lock; $('#rp-hint',app).hidden=lock;
  $$('#rp-controls .lockable',app).forEach(e=>{e.hidden=lock;});
  if(lock) ln.textContent=mode==='challenge'?tx('In the challenge you change only the base, in the task card. Everything else is fixed: heads of 128 numbers, texts of 1M tokens, no scaling.','挑战中只能在任务卡里改 base。其余固定：head 为 128 维，文本 1M 个 token，不缩放。')
    :tx('In the guided parts, each part has its own settings in its card. Every setting is open in Scenarios.','分步引导里，每一部分要调的设置都在它的卡片里。所有设置都在“应用场景”里开放。');
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.txt);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=cur(); if(s.enter) s.enter();
  const st=$('#rp-stage',app); if(st) st.scrollTop=0;
  renderRail();
  if(s.focusCtl){const el=$(s.focusCtl,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#rp-rail .step.current',app)||$('#rp-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), as labs/kv-basics.js does */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#rp-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#rp-rail .step.current',app)||$('#rp-rail .railcard',app)||$('#rp-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#rp-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#rp-insight',app)); else revealBox($('#rp-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  mode=m;
  if(m==='challenge'){chBase=CH.start; chSubmitted=false; chReveal=false; chTried.clear(); chTried.add(CH.start);}
  if(m==='sandbox') sbTouched=false;
  A.shell.setMode(app,m,DESCS());
  const st=$('#rp-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[SB.base,SB.sc?1:0,CTX.indexOf(SB.ctx),TRAIN.indexOf(SB.train)].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==4) return null;
  const [b,s,c,t]=p.map(Number);
  if(!SB_BASES.includes(b)||(s!==0&&s!==1)||!Number.isInteger(c)||!Number.isInteger(t)||!(c>=0&&c<CTX.length)||!(t>=0&&t<TRAIN.length)) return null;
  return {base:b,sc:s===1,ctx:CTX[c],train:TRAIN[t]};
}

/* =====================================================================
   interview drill: seeded, from the engine (never the challenge's 128 numbers at 1M tokens)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DR_D=[64,128,256], DR_B=[10000,100000,1000000], DR_L=[4096,32768,262144], DR_I=[0,8,16];
function makeDrill(seed){const rnd=mulberry(seed*7919+17), pick=a=>a[Math.floor(rnd()*a.length)];
  const d=pick(DR_D), b=pick(DR_B), Lx=pick(DR_L), i=pick(DR_I), np=d/2, th=Math.pow(b,-2*i/d), ts=Math.pow(b,-(d-2)/d);
  return {seed,d,b,L:Lx,i,np,th,ts,lam:TAU/th,lamS:TAU/ts,aS:Lx*ts};}
function renderDrill(){
  if(!app) return; const r=makeDrill(drillSeed), qd=$('#rp-drill-q',app), a=$('#rp-drill-a',app);
  qd.innerHTML=tx(`A model’s attention heads have <b>${r.d}</b> numbers per query and key, and its RoPE base is <b>${fint(r.b)}</b>. How many tokens does pair <b>${r.i}</b> need for one full turn? And over a text of <b>${tokK(r.L)}</b> tokens (${fint(r.L)}), how far does the slowest pair turn?`,
                  `一个模型的注意力 head 里，每个 query 和 key 有 <b>${r.d}</b> 维，RoPE base 是 <b>${fint(r.b)}</b>。第 <b>${r.i}</b> 对转满一圈要多少个 token？在一段 <b>${tokK(r.L)}</b>（${fint(r.L)}）个 token 的文本上，最慢的一对转了多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const sw=turnS(r.aS/TAU);
  a.innerHTML=tx(`<p><b>Pair ${r.i}: ${bigT(r.lam)} tokens for one full turn; the slowest pair turns ${sw}.</b></p>`+
    `<p>A head of ${r.d} numbers has ${r.np} pairs, numbered 0 to ${r.np-1}. Pair i turns θᵢ radians per token:</p><div class="rp-math-line">${MX(`\\theta_i = \\text{base}^{-2i/${r.d}}`)}</div><div class="rp-math-line">${MX(`\\theta_{${r.i}} = ${fint(r.b)}^{-${2*r.i}/${r.d}} = ${expT(r.th,4)}`)}</div><p>One full turn is ${MX('2\\pi')} radians, so it takes this many tokens:</p><div class="rp-math-line">${MX(`\\frac{2\\pi}{\\theta_{${r.i}}} = ${fint(Math.round(r.lam))}`)}</div>`+
    `<p>The slowest is pair ${r.np-1}:</p><div class="rp-math-line">${MX(`\\theta_{${r.np-1}} = ${fint(r.b)}^{-${r.d-2}/${r.d}} = ${expT(r.ts,4)}`)}</div><p>Over ${fint(r.L)} tokens it turns ${fint(r.L)} × ${expS(r.ts,4)} = ${fixd(r.aS,4)} radians, that is ${sw}${r.aS>=TAU?`, since one full turn is ${MX('2\\pi')} = ${fixd(TAU,4)} radians`:''}.</p>`,
    `<p><b>第 ${r.i} 对转满一圈要 ${bigT(r.lam)} 个 token；最慢的一对转了 ${sw}。</b></p>`+
    `<p>${r.d} 维的 head 有 ${r.np} 对，编号 0 到 ${r.np-1}。第 i 对每个 token 转 θᵢ 弧度：</p><div class="rp-math-line">${MX(`\\theta_i = \\text{base}^{-2i/${r.d}}`)}</div><div class="rp-math-line">${MX(`\\theta_{${r.i}} = ${fint(r.b)}^{-${2*r.i}/${r.d}} = ${expT(r.th,4)}`)}</div><p>一整圈是 ${MX('2\\pi')} 弧度，所以转满一圈要的 token 数是：</p><div class="rp-math-line">${MX(`\\frac{2\\pi}{\\theta_{${r.i}}} = ${fint(Math.round(r.lam))}`)}</div>`+
    `<p>最慢的是第 ${r.np-1} 对：</p><div class="rp-math-line">${MX(`\\theta_{${r.np-1}} = ${fint(r.b)}^{-${r.d-2}/${r.d}} = ${expT(r.ts,4)}`)}</div><p>在 ${fint(r.L)} 个 token 上，它转了 ${fint(r.L)} × ${expS(r.ts,4)} = ${fixd(r.aS,4)} 弧度，也就是 ${sw}${r.aS>=TAU?`，因为一整圈是 ${MX('2\\pi')} = ${fixd(TAU,4)} 弧度`:''}。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const TERM={
 shift:()=>tx(`The positions of “bites” (2 + ${shift}) and “dog” (1 + ${shift}) after both moved ${shift} later.`,`两个 token 一起往后挪 ${shift} 之后，“咬”在 2 + ${shift}，“狗”在 1 + ${shift}。`),
 rope3:()=>tx('The positions of “bites” (2) and “dog” (1 in “dog bites man”, 3 in “man bites dog”): RoPE turns each by its position × 57.3°.','“咬”的位置（2）和“狗”的位置（“狗咬人”里是 1，“人咬狗”里是 3）：RoPE 把每一个转过“位置 × 57.3°”。'),
 b5:()=>tx(`The base picked in this part: ${fint(base5)}.`,`这一部分选的 base：${fint(base5)}。`),
 l5:()=>tx(`The text length picked in this part: ${fint(L5[l5])} tokens.`,`这一部分选的文本长度：${fint(L5[l5])} 个 token。`),
 l6:()=>tx(`The text length picked in this part: ${fint(L6[l6])} tokens.`,`这一部分选的文本长度：${fint(L6[l6])} 个 token。`),
 base5f:()=>tx(`${M3}’s RoPE base, 500,000, from its config.json (rope_theta).`,`${M3} 的 RoPE base 500,000，来自它的 config.json（rope_theta）。`),
 train8:()=>tx(`${M3} was trained on texts of up to 8,192 tokens.`,`${M3} 训练时的文本最长 8,192 个 token。`),
 sbase:()=>tx(`The RoPE base set under Settings: ${fint(SB.base)}.`,`“自由调参”里设的 RoPE base：${fint(SB.base)}。`),
 sctx:()=>tx(`The text length set under Settings: ${fint(SB.ctx)} tokens.`,`“自由调参”里设的文本长度：${fint(SB.ctx)} 个 token。`),
 strain:()=>tx(`The longest training text set under Settings: ${fint(SB.train)} tokens.`,`“自由调参”里设的最长训练文本：${fint(SB.train)} 个 token。`),
 chbase:()=>tx(`The base you picked in the task card: ${fint(chBase)}.`,`你在任务卡里选的 base：${fint(chBase)}。`),
 chL:()=>tx('The challenge’s text: 1M tokens, 1,048,576.','挑战题的文本：1M 个 token，即 1,048,576 个。'),
 baseT:()=>TIP.base(),
 q0a:()=>tx('−47.0°: the direction of pair 0 of the query of “bites” before the turn, made up for this lab.','−47.0°：“咬”的 query 的第 0 对在转动之前的方向，是这个 Lab 虚构的。'),
 k0a:()=>tx('30.0°: the direction of pair 0 of the key of “dog” before the turn, made up for this lab.','30.0°：“狗”的 key 的第 0 对在转动之前的方向，是这个 Lab 虚构的。'),
 th0:()=>tx('57.30°: how far pair 0 turns per position, 1 radian, in every Llama head.','57.30°：第 0 对每个位置转过的角度，也就是 1 弧度，每个 Llama head 都一样。'),
 lq:()=>tx('1.5: the length of pair 0 of the query of “bites”, its content; the turn does not change it.','1.5：“咬”的 query 的第 0 对的长度，也就是它的内容；转动不改变它。'),
 lk:()=>tx('1.3: the length of pair 0 of the key of “dog”, its content; the turn does not change it.','1.3：“狗”的 key 的第 0 对的长度，也就是它的内容；转动不改变它。'),
 d0:()=>tx('−77.0°: the angle between the two arrows before the turn, −47.0° − 30.0°.','−77.0°：转动之前两个箭头的夹角，−47.0° − 30.0°。'),
 mn:()=>TIP.mn(),
 phi:()=>tx('φ: the direction of an arrow before the turn; φq for the query’s pair i, φk for the key’s.','φ：箭头在转动之前的方向；φq 是 query 的第 i 对，φk 是 key 的第 i 对。'),
 th63:()=>{const b=mathBase(), sc=mathScaled(), f=thOf(b,sc)[NP-1]; return tx(`θ₆₃ = ${expS(f,4)}: how far pair 63, the slowest, turns per token, in radians, at base ${fint(b)}${sc?' with the llama3 scaling':''}.`,`θ₆₃ = ${expS(f,4)}：最慢的第 63 对每个 token 转过的弧度，base ${fint(b)}${sc?'，用了 llama3 缩放':''}。`);},
 tau:()=>tx('The radians in one full turn.','一整圈的弧度。')
};
/* terms that carry a typeset formula under their text (R0.1) */
const TERMFX={tau:()=>'2\\pi \\approx 6.2832', baseT:()=>'θ_i = \\text{base}^{-2i/128}', th63:()=>'θ_{63} = \\text{base}^{-126/128}', phi:()=>['q_i \\cdot k_i = |q_i| \\, |k_i| \\cos Δ_i','Δ_i = φ_q - φ_k + (m - n)\\,θ_i']};
const CONST_T=['baseT','q0a','k0a','th0','lq','lk','d0','mn','phi','th63','tau'];
const termText=k=>{const tm=TERM[k]; return tm?tm():'';};
const termTipHTML=k=>{if(!TERM[k]) return ''; const fx=TERMFX[k]&&A.ui.math?[].concat(TERMFX[k]()).map(f=>`<div class="fx">${A.ui.math(f)}</div>`).join(''):'', act=jumpAct(k);
  return `<div>${esc(termText(k))}</div>${fx}${act?`<div class="act">${esc(act)}</div>`:''}`;};
const CTLN={shift:{en:'Both tokens moved later by',zh:'两个 token 一起往后挪'},rope3:{en:'RoPE',zh:'RoPE'},b5:{en:'RoPE base',zh:'RoPE base'},l5:{en:'Text length L',zh:'文本长度 L'},l6:{en:'Text length L',zh:'文本长度 L'},
  sbase:{en:'RoPE base',zh:'RoPE base'},sctx:{en:'Text length',zh:'文本长度'},strain:{en:'Longest training text',zh:'最长的训练文本'},chbase:{en:'the base options in the task card',zh:'任务卡里的 base 选项'}};
function jumpAct(k){
  const pm=/^p([1-7])$/.exec(k);
  if(pm) return tx(`Click to open part ${pm[1]}: ${L(STEPS[+pm[1]-1].title)}.`,`点击打开第 ${pm[1]} 部分：${L(STEPS[+pm[1]-1].title)}。`);
  if(k==='sc3') return tx(`Click to open the scenario “${L(PRESETS[3].t)}”.`,`点击打开应用场景“${L(PRESETS[3].t)}”。`);
  if(k==='controls') return mode==='sandbox'?tx('Click to jump to Settings.','点击跳到“自由调参”。'):tx('Click to jump to Settings; its settings open in Scenarios.','点击跳到“自由调参”；它的设置在“应用场景”里开放。');
  if(k==='stage') return tx('Click to scroll to the chart.','点击滚到图。');
  if(k==='cmp') return tx('Click to scroll to the comparison in the chart.','点击滚到图里的对比。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='base5f') return tx(`Fixed in this part: ${M3}’s base.`,`这一部分里固定不变：${M3} 的 base。`);
  if(k==='train8') return tx(`Fixed in this part: ${M3}’s training length.`,`这一部分里固定不变：${M3} 的训练长度。`);
  if(k==='chL') return tx('Fixed during the challenge: texts of 1M tokens.','挑战中固定不变：1M 个 token 的文本。');
  if(!CTLN[k]) return '';
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({shift:$('#rp-ic-shift',app),rope3:$('#rp-ic-rope',app),b5:$('#rp-ic-b5',app),l5:$('#rp-ic-l5',app),l6:$('#rp-ic-l6',app),chbase:$('#rp-ic-ch',app),
  sbase:$('#rp-ctl-base',app),sctx:$('#rp-ctl-ctx',app),strain:$('#rp-ctl-train',app)}[k]);
/* open Scenarios with these settings (the scenario link in part 6, "Open in Scenarios" in the capability card) */
function openSettings(set){
  if(!app) return; A.tip.hide();
  const go=()=>{byLearner(()=>{SB={...set}; sbTouched=true; render();});};
  if(mode!=='sandbox'){setMode('sandbox'); go();} else go();
  A.ui.reveal($('#rp-rail',app));
}
function jump(k,src){
  if(!app) return;
  const pm=/^p([1-7])$/.exec(k);
  if(pm){A.tip.hide(); const i=+pm[1]-1; if(mode!=='guided'){step=i; setMode('guided');} else enterStep(i); revealStep(); return;}
  if(k==='sc3'){openSettings(PRESETS[3].s); return;}
  if(k==='math'){const d=$('#rp-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='stage'||k==='cmp'){const el=k==='cmp'?($('#rp-cmp',app)||$('#rp-graph',app)):$('#rp-graph',app); A.tip.hide(); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='controls'){const c=$('#rp-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(CONST_T.includes(k)){if(src) A.tip.show(src,termTipHTML(k)); return;}
  if(['base5f','train8','chL'].includes(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.offsetParent!==null){A.tip.hide(); A.ui.jumpToEl(el); return;}
  if(src) A.tip.show(src,`<div>${esc(termText(k)||jumpAct(k))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{SB={...PRESETS[3].s}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chOK(chBase)) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#rp-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* a change under Settings never moves the page (R3.2): scroll anchoring is paused and the control is put back where it was */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#rp-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
/* on a phone the drawing sits above the card: after a change in parts 2, 3, 5 and 6 (a click, or the end of a drag), bring the drawing
   back into view so the learner sees what changed (A12) */
const PHONE_SHOW=['hv','rope3','l5','b5','l6','s6'];
function phoneShow(k){
  if(innerWidth>760||!PHONE_SHOW.includes(k)||!app) return false;
  requestAnimationFrame(()=>{const g=$('#rp-graph',app); if(!g) return; const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
    const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:0)+8, r=g.getBoundingClientRect();
    if(r.top>=top&&r.bottom<=innerHeight) return;
    window.scrollBy({top:r.top-top,behavior:A.reduceMotion()?'auto':'smooth'});});
  return true;
}
/* a change in the card on a phone keeps the control under the finger while the stage above it changes height (R3.2) */
const inCard=(el,f)=>innerWidth<=760?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='shift') inCard(e.target,()=>{shift=Math.max(0,Math.min(SHIFT_MAX,v)); render();});
    else if(k==='l5') inCard(e.target,()=>{const n=Math.max(0,Math.min(L5.length-1,v)); if(n!==l5){p5Prev={l5,base5}; l5=n; p5Last='l5';} render();});
    else if(k==='l6') inCard(e.target,()=>{l6=Math.max(0,Math.min(L6.length-1,v)); render();});
    else if(k==='sctx') bySettings(e.target,()=>{SB={...SB,ctx:CTX[Math.max(0,Math.min(CTX.length-1,v))]}; render();});});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#rp-rail')){if(phoneShow(k)) return; wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#rp-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=cur(); if(s.showMe){wantObs=true; byLearner(()=>{s.showMe(); render();}); revealSoon(); wantObs=false;} return;}
    if(d.pred){const first=!pred; pred=d.pred; if(first) renderRail(); else render(); $$('#rp-rail [data-pred]',app).forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.pred===pred)));
      const fb=$(`#rp-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&b.closest('#rp-ic-ch')){byLearner(()=>{chBase=+d.lever; chTried.add(chBase); render();}); return;}
    if(d.v!=null&&b.closest('.seg-ctl')){const k=b.closest('.seg-ctl').dataset.key;
      if(k==='hv'){inCard(b,()=>{setHV(d.v); render();}); if(!phoneShow(k)) revealSoon(); return;}
      if(k==='rope3'){inCard(b,()=>{rope3=d.v==='on'; render();}); if(!phoneShow(k)) revealSoon(); return;}
      if(k==='b5'){inCard(b,()=>{if(+d.v!==base5){p5Prev={l5,base5}; base5=+d.v; p5Last='b5';} render();}); if(!phoneShow(k)) revealSoon(); return;}
      if(k==='s6'){inCard(b,()=>{sc6=d.v==='1'; render();}); if(!phoneShow(k)) revealSoon(); return;}
      if(b.closest('#rp-controls')){bySettings(b,()=>{if(k==='sbase') SB={...SB,base:+d.v}; else if(k==='ssc') SB={...SB,sc:d.v==='1'}; else if(k==='strain') SB={...SB,train:+d.v}; render();}); return;}}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>{SB={...p.s}; render();}); requestAnimationFrame(()=>{if(app) revealBox($('#rp-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#rp-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
  /* the in-text links are spans (so they wrap like text): Enter or Space follows them, as on a button */
  el.addEventListener('keydown',e=>{const t=e.target; if((e.key==='Enter'||e.key===' ')&&t&&t.classList&&t.classList.contains('rp-lk')){e.preventDefault(); t.click();}});
}

/* =====================================================================
   registration
   ===================================================================== */
let resizeT=0;
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps it, address token included (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...SB},pred,observed:{...observed},sbTouched,headView,seenView:[...seenView],rope3,shift,l5,base5,l6,sc6,p5Last,p5Prev:{...p5Prev},
    chBase,chTried:[...chTried],chSubmitted,chReveal,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chOK(chBase)}),
  restore(o){if(!o||o.v!==1) return;
    const p=o.S&&typeof o.S==='object'?parse([o.S.base,o.S.sc?1:0,CTX.indexOf(o.S.ctx),TRAIN.indexOf(o.S.train)].join('_')):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; SB={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    sbTouched=o.sbTouched===true; headView=HV.some(h=>h[0]===o.headView)?o.headView:'128';
    seenView.clear(); seenView.add('128'); if(Array.isArray(o.seenView)) o.seenView.forEach(x=>{if(HV.some(h=>h[0]===x)) seenView.add(x);}); seenView.add(headView);
    const idx=(v,arr,d)=>Number.isInteger(v)&&v>=0&&v<arr.length?v:d;
    rope3=o.rope3===true; shift=typeof o.shift==='number'&&o.shift>=0&&o.shift<=SHIFT_MAX?o.shift:0;
    l5=idx(o.l5,L5,1); base5=B5.includes(o.base5)?o.base5:500000; l6=idx(o.l6,L6,0); sc6=o.sc6===true;
    p5Last=['l5','b5','both'].includes(o.p5Last)?o.p5Last:null;
    p5Prev=o.p5Prev&&typeof o.p5Prev==='object'?{l5:idx(o.p5Prev.l5,L5,1),base5:B5.includes(o.p5Prev.base5)?o.p5Prev.base5:500000}:{l5:1,base5:500000};
    const ch=mode==='challenge'; chBase=ch&&CH.bases.includes(o.chBase)?o.chBase:CH.start;
    chTried.clear(); chTried.add(CH.start); if(ch&&Array.isArray(o.chTried)) o.chTried.forEach(x=>{if(CH.bases.includes(x)) chTried.add(x);}); chTried.add(chBase);
    chSubmitted=ch&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){SB={...p}; mode='sandbox'; sbTouched=true;}
    fromSnap=false;
    el.innerHTML=viewHTML(); app=$('#rp-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#rp-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    const rail=$('#rp-rail',app); rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):''); railExtras(rail); A.ui.initRanges(rail);
    render();
    requestAnimationFrame(()=>{if(app) drawVis();});   // the stage's width is known once it is laid out
  },
  leave(){clearTimeout(annT); clearTimeout(resizeT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#rp-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#rp-math',app).open=true;}},
  onResize(){if(app){clearTimeout(resizeT); resizeT=setTimeout(()=>{if(app) drawVis();},60);}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#rp-graph',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#rp-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), view=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      vis.innerHTML=dockVis(view);
      read.innerHTML=esc(dockRead(view));}
  }
});
/* the mini result bar: a small picture of the result (A12): the row of "bites" in both orders, pair 0's two arrows, or the 64 dots
   with the one-turn line */
function dockVis(view){
  const S=26, o=`<svg class="rp-dk" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}" aria-hidden="true">`;
  if(view==='order'){const r=ropeOn(), rows=['a','b'].map(k=>grid(k,r)[1]); let h='';
    rows.forEach((g,k)=>{const top=g.indexOf(Math.max(...g)); g.forEach((v,j)=>{h+=`<rect x="${2+j*8}" y="${4+k*10}" width="6" height="7" rx="1" class="rp-dkc${j===top?' on':''}"/>`;});});
    return o+h+'</svg>';}
  if(view==='turns'){const {d}=turnsCfg(), Y=t=>2+(6-Math.max(-4,Math.min(6,Math.log10(Math.max(t,1e-4)))))/10*22;
    return o+`<line x1="1" y1="${f1(Y(1))}" x2="25" y2="${f1(Y(1))}" class="rp-dkone"/><polyline points="${d.rows.map(r=>`${f1(1+r.i*24/63)},${f1(Y(r.t))}`).join(' ')}" class="rp-dkline"/></svg>`;}
  const r=11, x=13, y=13; let qa=P0.q.bites[1]*D2R, ka=null;
  if(view==='pair2'){const p=pair0(2,1,rope3); qa=p.qa*D2R; ka=p.ka*D2R;}
  else if(view==='shift'){const p=pair0(2+shift,1+shift,true); qa=p.qa*D2R; ka=p.ka*D2R;}
  return o+`<circle cx="${x}" cy="${y}" r="${r}" class="rp-dial"/>`+(ka!=null?`<line x1="${x}" y1="${y}" x2="${f1(x+(r-3)*Math.cos(ka))}" y2="${f1(y-(r-3)*Math.sin(ka))}" class="rp-dkk"/>`:'')+`<line x1="${x}" y1="${y}" x2="${f1(x+(r-2)*Math.cos(qa))}" y2="${f1(y-(r-2)*Math.sin(qa))}" class="rp-hand"/></svg>`;
}
function dockRead(view){
  if(view==='order'){const r=ropeOn(), a=grid('a',r)[1][0], b=grid('b',r)[1][2]; return tx(`bites → dog: ${sc2(a)} vs ${sc2(b)}`,`“咬 → 狗”：${sc2(a)} 和 ${sc2(b)}`);}
  if(view==='head') return tx('The query of “bites”: 128 numbers in 64 pairs','“咬”的 query：128 维，64 对');
  if(view==='pair2'){const a=pair0(2,1,rope3), b=pair0(2,3,rope3); return tx(`RoPE ${rope3?'on':'off'}: ${sc2(a.head)} in “dog bites man”, ${sc2(b.head)} in “man bites dog”`,`RoPE ${rope3?'开':'关'}：“狗咬人”里 ${sc2(a.head)}，“人咬狗”里 ${sc2(b.head)}`);}
  if(view==='shift'){if(!pred) return tx('Predict first: does moving both tokens change the score?','先预测：两个 token 一起挪，得分会变吗？'); const b=pair0(2+shift,1+shift,true); return tx(`bites at ${b.m}, dog at ${b.n}: score ${sc2(b.head)}`,`“咬”在 ${b.m}、“狗”在 ${b.n}：得分 ${sc2(b.head)}`);}
  const {d}=turnsCfg();
  if(mode==='challenge') return tx(`Base ${fint(chBase)} over 1M tokens: the slowest pair turns ${turnsS(chTurns(chBase))}`,`base ${fint(chBase)}，1M 个 token：最慢的一对转 ${turnsS(chTurns(chBase))}`);
  if(d.T!=null&&d.nBeyond) return tx(`${tokN(d.Lx)}: ${d.nBeyond} pairs past training`,`${tokN(d.Lx)}：${d.nBeyond} 对超出训练`);
  if(d.T!=null) return tx(`${tokN(d.Lx)}: every pair within training`,`${tokN(d.Lx)}：每一对都在训练范围内`);
  return tx(`${tokN(d.Lx)}: ${d.nUnder} of 64 pairs under one turn`,`${tokN(d.Lx)}：${d.nUnder} / 64 对不满一圈`);
}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const pick=d=>deep({nUnder:d.nUnder,nWrapped:d.nWrapped,nBeyond:d.nBeyond,first:d.first,slow:d.slow,ratio:d.ratio,turns:d.rows.map(r=>r.t),tt:d.rows.map(r=>r.tt)});
window.__rp=Object.freeze({
  thetas:(b,sc)=>thetas(b,sc?SC31:null).slice(), qv:w_=>QV[w_].slice(), kv:w_=>KV[w_].slice(), share, score, grid:(o,r)=>deep(grid(o,r)), pair0:(m,n,r)=>deep(pair0(m,n,r)),
  turns:(b,sc,Lx,T,trainSc)=>pick(turnsData(thOf(b,sc),Lx,T!=null?thOf(b,trainSc):null,T!=null?T:null)),
  p5:(b,Lx)=>pick(turnsData(thetas(b,null),Lx,null,null)), p6:(Lx,sc)=>pick(turnsData(thOf(500000,sc),Lx,thetas(500000,null),8192)),
  sb:s=>pick(sbData(s)), chTurns, chLam, chBest, drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({HD,NP,SC31:{...SC31},P0:deep(P0),L5:L5.slice(),L6:L6.slice(),B5:B5.slice(),SB_BASES:SB_BASES.slice(),CTX:CTX.slice(),TRAIN:TRAIN.slice(),
    CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),MODELS:deep(MODELS),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,SHIFT_MAX}),
  fmt:Object.freeze({sc2,degS,turnS,tokK,bigT}),
  state:()=>({mode,step,pred,observed:{...observed},headView,seen:[...seenView],rope3,shift,l5,base5,l6,sc6,SB:{...SB},chBase,chSubmitted,chReveal,view:viewNow(),stageBusy:stageT!==0})
});

})();
