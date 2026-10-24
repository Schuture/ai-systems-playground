/* sampling (the trunk "How LLMs work", world models): What do temperature and top-p actually do?
   Fidelity: Toy run. A hand-made next-word table (illustrative, not a real model's logits), three words deep, priced exactly by formula,
   and a seeded sampler for "run it 20 times".
   - One step: p = softmax(z ÷ T); then top-k keeps the k likeliest words and renormalizes; then top-p keeps the smallest set of likeliest
     words whose chances add up to at least p and renormalizes (the order of Hugging Face's generate and of vLLM's sampler). T = 0 is
     greedy: all of the chance goes to the top logit (vLLM: "Zero means greedy sampling").
   - The first step is the home tile's (teasers/set-b.js, SP): the same six words, the same logits, the same top-p rule (not said on the page:
     a learner who opens the lab directly has not seen the tile, review A20).
   - A run draws three random numbers u1, u2, u3 (mulberry32, seeded); each picks the word whose segment it lands in on the cumulative bar
     of that step's final chances, in list order. Lists are stored likeliest first, so list order is the sorted order.
   Words (EN / ZH): temperature (温度 on first use, then temperature); top-p; top-k; logit; greedy / 贪心; run / 次运行; part / 部分.
   window.__sp is a read-only hook for tests/sampling_numbers.py and tests/sampling_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='sampling';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const nd=(en,zh,z,c)=>({w:{en,zh},z,c:c||null});
/* the lunch prompt: the first step is the tile's (logits 2.0 … −1.0); each word has its own list of next words, each of those its own list */
const PROMPT={en:'For lunch today I want',zh:'我今天午饭想吃'};
const TREE=[
 nd('noodles','面条',2.0,[nd('with','加',2.0,[nd('beef','牛肉',1.5),nd('an egg','鸡蛋',1.0)]),nd('and','和',1.2,[nd('dumplings','饺子',1.3),nd('tea','茶',0.5)]),nd('without','不要',0.5,[nd('chili','辣椒',1.2),nd('onions','葱',0.8)])]),
 nd('rice','米饭',1.5,[nd('with','配',2.1,[nd('curry','咖喱',1.4),nd('chicken','鸡肉',1.1)]),nd('and','和',1.3,[nd('soup','汤',1.6),nd('fish','鱼',0.7)]),nd('without','不要',0.2,[nd('meat','肉',1.0),nd('sauce','酱',0.9)])]),
 nd('dumplings','饺子',1.0,[nd('with','蘸',2.4,[nd('vinegar','醋',2.0),nd('chili oil','辣油',0.8)]),nd('and','和',1.0,[nd('noodles','面条',1.0),nd('soup','汤',0.9)]),nd('without','不要',0.1,[nd('garlic','蒜',0.9),nd('meat','肉',0.5)])]),
 nd('hot pot','火锅',0.3,[nd('with','配',1.9,[nd('beer','啤酒',1.3),nd('rice','米饭',0.5)]),nd('and','和',1.0,[nd('noodles','面条',0.8),nd('tea','茶',0.6)]),nd('without','不要',0.8,[nd('chili','辣椒',1.6),nd('tripe','毛肚',0.2)])]),
 nd('salad','沙拉',-0.5,[nd('with','加',2.0,[nd('chicken','鸡肉',1.5),nd('tuna','金枪鱼',0.6)]),nd('and','和',1.3,[nd('soup','汤',1.2),nd('bread','面包',0.9)]),nd('without','不要',0.6,[nd('dressing','沙拉酱',1.4),nd('onions','洋葱',0.4)])]),
 nd('a burger','汉堡',-1.0,[nd('and','和',4.5,[nd('fries','薯条',2.5),nd('a cola','可乐',1.0)]),nd('with','加',1.0,[nd('cheese','芝士',1.4),nd('bacon','培根',0.9)]),nd('without','不要',0.5,[nd('onions','洋葱',1.1),nd('sauce','酱',0.6)]),nd('or','或',0.2,[nd('pizza','披萨',1.2),nd('fried chicken','炸鸡',1.0)])])];
const Z1=TREE.map(x=>x.z);
const N_LEAVES=TREE.reduce((a,x)=>a+x.c.reduce((b,y)=>b+y.c.length,0),0);   // 38 sentences, three words deep
const PEAK=5;                                            // part 4's peaked list: the word after “a burger”
/* the challenge: a prompt the lab never showed, seven words, three of them wrong */
const CHP={en:'Another word for ‘happy’ is',zh:'‘开心’的近义词是'};
const CHW=[{w:{en:'glad',zh:'高兴'},z:2.8},{w:{en:'cheerful',zh:'快乐'},z:2.0},{w:{en:'joyful',zh:'愉快'},z:1.3},{w:{en:'pleased',zh:'喜悦'},z:0.8},
  {w:{en:'sad',zh:'伤心'},z:0.2,bad:true},{w:{en:'angry',zh:'生气'},z:-0.3,bad:true},{w:{en:'afraid',zh:'害怕'},z:-0.8,bad:true}];
const CHZ=CHW.map(x=>x.z);
const CH={Ts:[0.5,0.7,1.0,1.3],Ps:[0.8,0.9,0.95,1.0],start:{T:1.0,P:1.0},wrongMax:0.01,minShare:0.05,minWords:3};

function softmax(z){const m=Math.max(...z), e=z.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s);}
/* one step under the settings. raw: after temperature (T = 0: all on the top logit); q: after top-k, renormalized; fin: after top-p,
   renormalized. sk: the raw chance top-k keeps, so top-p's line sits at p × sk on the raw bar */
function dist(z,T,K,P){
  const n=z.length, raw=T===0?z.map((_,i)=>i===0?1:0):softmax(z.map(v=>v/T));
  const kk=K>0&&K<n?K:n, keepK=raw.map((_,i)=>i<kk), sk=raw.reduce((a,v,i)=>a+(keepK[i]?v:0),0), q=raw.map((v,i)=>keepK[i]?v/sk:0);
  let keep;
  if(T===0) keep=raw.map(v=>v>0);                       // greedy: only the top word can be picked
  else if(P>=1) keep=keepK.slice();
  else {keep=raw.map(()=>false); let c=0; for(let i=0;i<n;i++){if(!keepK[i]) break; if(c<P){keep[i]=true; c+=q[i];}}}
  const s2=q.reduce((a,v,i)=>a+(keep[i]?v:0),0), fin=q.map((v,i)=>keep[i]?v/s2:0);
  const cum=[]; raw.reduce((a,v)=>{a+=v; cum.push(a); return a;},0);
  const nk=keep.filter(Boolean).length, cutK=raw.reduce((a,v,i)=>a+(keepK[i]?0:v),0), cutAll=raw.reduce((a,v,i)=>a+(keep[i]?0:v),0);
  return {n,T,K,P,greedy:T===0,raw,q,fin,keep,keepK,kk,sk,s2,cum,nk,cutK,cutAll,pLine:P<1&&T>0?P*sk:null,kLine:kk<n&&T>0?sk:null};
}
/* inverse CDF: the word whose segment u lands in, the final chances laid end to end in list order */
function pick(fin,u){let c=0, last=0; for(let i=0;i<fin.length;i++){if(fin[i]<=0) continue; last=i; c+=fin[i]; if(u<c) return i;} return last;}
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const RUNS=new Map();
/* 20 runs, three random numbers each (the first word, the second, the third) */
function runsOf(seed){if(RUNS.has(seed)) return RUNS.get(seed); const r=mulberry(seed*7919+17), a=[]; for(let i=0;i<20;i++) a.push([r(),r(),r()]); RUNS.set(seed,a); return a;}
/* the whole state: the first word, the tree three words deep, the 20 runs */
const CC=new Map();
function compute(s){
  const key=[s.T,s.K,s.P,s.seed].join('|'); if(CC.has(key)) return CC.get(key);
  const d1=dist(Z1,s.T,s.K,s.P);
  const tree=TREE.map((a,i)=>{const d2=dist(a.c.map(x=>x.z),s.T,s.K,s.P);
    return {i,p:d1.fin[i],d:d2,kids:a.c.map((b,j)=>{const d3=dist(b.c.map(x=>x.z),s.T,s.K,s.P), pj=d1.fin[i]*d2.fin[j];
      return {j,p:pj,d:d3,leaves:b.c.map((c,k)=>({k,p:pj*d3.fin[k]}))};})};});
  const walk=u=>{const i=pick(d1.fin,u[0]), j=pick(tree[i].d.fin,u[1]), k=pick(tree[i].kids[j].d.fin,u[2]); return {i,j,k,p:tree[i].kids[j].leaves[k].p};};
  const R=runsOf(s.seed), runs=R.map(walk), first=runs.map(r=>r.i), counts=Z1.map((_,i)=>first.filter(x=>x===i).length);
  let best={p:-1}, possible=0;
  tree.forEach(a=>a.kids.forEach(b=>b.leaves.forEach(c=>{if(c.p>0) possible++; if(c.p>best.p) best={i:a.i,j:b.j,k:c.k,p:c.p};})));
  const sents=new Set(runs.map(r=>r.i+'.'+r.j+'.'+r.k));
  const r={s:{...s},d1,tree,runs,first,counts,distinct:counts.filter(Boolean).length,best,possible,distinctS:sents.size,walk};
  CC.set(key,r); return r;
}
/* the challenge: the three wrong words' chances added up, and the good words that keep 5% or more */
function chEval(T,P){const d=dist(CHZ,T,0,P), wrong=CHW.reduce((a,x,i)=>a+(x.bad?d.fin[i]:0),0), n5=d.fin.filter((v,i)=>!CHW[i].bad&&v>=CH.minShare).length;
  return {T,P,d,wrong,n5,ok:wrong<CH.wrongMax&&n5>=CH.minWords};}
/* the fewest changes first (R4.5): how many of the two settings moved from the initial setup, then how hard top-p cuts */
const chChanges=(T,P)=>(T!==CH.start.T?1:0)+(P!==CH.start.P?1:0);
function chBest(){let b=null; CH.Ts.forEach(T=>CH.Ps.forEach(P=>{const e=chEval(T,P); if(!e.ok) return; const c=chChanges(T,P);
  if(!b||c<b.c||(c===b.c&&P>b.P)||(c===b.c&&P===b.P&&Math.abs(T-1)<Math.abs(b.T-1))) b={T,P,c};})); return b;}
/* part 4: two lists at temperature 1, the same rule on both */
const LV=[{id:'none',K:0,P:1,t:{en:'No cut (initial setup)',zh:'不截断（初始设置）'}},{id:'k3',K:3,P:1,t:{en:'Top-k 3',zh:'top-k 3'}},{id:'p9',K:0,P:0.9,t:{en:'Top-p 0.9',zh:'top-p 0.9'}}];
const lvOf=id=>LV.find(v=>v.id===id)||LV[0];
const flatD=id=>{const v=lvOf(id); return dist(Z1,1,v.K,v.P);};
const peakD=id=>{const v=lvOf(id); return dist(TREE[PEAK].c.map(x=>x.z),1,v.K,v.P);};
/* part 5: your run. The first random number is yours to move; the second and third stay fixed */
const U23=[0.30,0.60], U1_0=0.20, U1_SHOW=0.97;

/* =====================================================================
   formatting (percent with enough digits to see a change, R6.10; prose rounded, R6.20)
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const NS=' ';
const fP=x=>{if(x<=0) return '0%'; const v=x*100; if(v>=99.95&&x<1) return '>99.9%'; if(x>=1) return '100%'; if(v<0.01) return '<0.01%'; if(v<1) return fixd(v,2)+'%'; return fixd(v,1)+'%';};
/* two chances that round alike get one more decimal, so the change stays visible (R6.10) */
const fP2=(a,b)=>{if(fP(a)!==fP(b)||a<=0||b<=0||a>=1||b>=1) return [fP(a),fP(b)]; const d=a*100<1||b*100<1?3:2; return [fixd(a*100,d)+'%',fixd(b*100,d)+'%'];};
const fR=x=>{const v=x*100; if(x<=0) return '0%'; if(v>=99.5&&x<1) return tx('almost 100%','接近 100%'); if(v>=9.5) return fint(Math.round(v))+'%'; if(v>=0.95) return fixd(v,1)+'%'; if(v>=0.095) return fixd(v,1)+'%'; return tx('under 0.1%','不到 0.1%');};
const fT=T=>T===0?tx('0 (greedy)','0（贪心）'):fixd(T,1);
const fTs=T=>fixd(T,1);
const fPp=P=>P>=1?tx('1.00 (off)','1.00（不截断）'):fixd(P,2);
const fK=K=>K>0?String(K):tx('off','不限');
const word=(x)=>L(x.w);
const pl=(n,one,many)=>n===1?one:many;
const wordN=n=>tx(n===1?'1 word':`${n} words`,`${n} 个词`);
/* top‑p and top‑k with a no-break hyphen, in labels that must not break across lines (R7.8) */
const nbh=s=>String(s).replace(/(top)-([pk])/gi,'$1‑$2');
/* a chance as shown inside a product that must redo by hand (R6.6): 4 significant digits of a percent (43.69%, 2.175%, 0.1234%),
   so a product of shown factors matches Show the math, which uses 4 significant digits too */
const pcs=x=>{const v=x*100, d=v>=10?2:(v>=1?3:(v>=0.1?4:5)), r=Math.round(v*10**d)/10**d; return {v:r,s:fixd(r,d)+'%'};};
/* 4 significant digits, for the factors and the product in Show the math */
const sig4=x=>{if(x===0) return '0'; const d=Math.max(0,3-Math.floor(Math.log10(Math.abs(x)))); return fixd(x,Math.min(d,7));};
const sentOf=(r,path)=>{const a=TREE[path.i], b=a.c[path.j], c=b.c[path.k]; return A.lang()==='zh'?word(a)+word(b)+word(c):[word(a),word(b),word(c)].join(' ');};
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={T:1.0,P:1.0,K:0,seed:1,view:'a'};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seenT0=false, seenHigh=false, lv='none', lvTried=new Set(['none']), u1=U1_0, u1Moved=false, remView='runs';
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false;
let drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false;
const shownPrev={};
const R=()=>compute(S);
const B0=()=>compute({...BASE});            // the mission's setup: temperature 1, no cut, seed 1

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 /* before part 2 is answered the definition stops short of temperature 0, which is part 2's question (R3.8) */
 temp:()=>(mode==='guided'&&!pred)
   ?tx('Temperature is the number every logit is divided by before softmax. Below 1 it stretches the gaps between the logits, so the likeliest word gains; above 1 it shrinks them, so the chances move toward equal. What happens at 0 is part 2’s question.',
       'temperature（温度）是做 softmax 之前每个 logit 要除以的数。小于 1 时 logit 之间的差距被拉大，最可能的词分到更多；大于 1 时差距被缩小，各个词的概率趋向相等。等于 0 时会怎样，是第 2 部分要回答的问题。')
   :tx('Temperature is the number every logit is divided by before softmax. Below 1 it stretches the gaps between the logits, so the likeliest word gains; above 1 it shrinks them, so the chances move toward equal; at 0 engines skip the draw and always take the top word (greedy decoding).',
       'temperature（温度）是做 softmax 之前每个 logit 要除以的数。小于 1 时 logit 之间的差距被拉大，最可能的词分到更多；大于 1 时差距被缩小，各个词的概率趋向相等；等于 0 时引擎不再抽签，总是取最高的那个词（贪心解码）。'),
 topp:()=>tx('Top-p (nucleus sampling) sorts the words by chance, keeps the smallest set of the likeliest ones whose chances add up to at least p, and scales those up so they add up to 100% again. Top-p 1 keeps every word.',
             'top-p（nucleus sampling，核采样）把词按概率从高到低排好，留下概率加起来至少为 p 的最少几个词，再把它们按比例放大，使总和回到 100%。top-p 为 1 时一个词也不去掉。'),
 topk:()=>tx('Top-k keeps only the k likeliest words and scales their chances up so they add up to 100% again. It keeps the same number of words whatever their chances.',
             'top-k 只留下最可能的 k 个词，再把它们的概率按比例放大，使总和回到 100%。不管这些词的概率是多少，它总是留下同样多的词。'),
 greedy:()=>tx('Greedy decoding always picks the word with the highest logit, so the same prompt always gives the same answer. It is the limit of temperature going to 0.',
               '贪心解码（greedy decoding）总是选 logit 最高的那个词，所以同一个 prompt 总是得到同一个回答。它就是 temperature 趋向 0 的极限。'),
 draw:()=>tx('A draw: the app takes a random number between 0 and 1 and lays the words’ chances end to end from 0 to 1; the word whose stretch the number lands in is picked. Here the random numbers come from a fixed seed, so the page shows the same runs every time.',
             '抽一个词：程序取一个 0 到 1 之间的随机数，把各个词的概率首尾相接排成 0 到 1；随机数落在哪个词的那一段，就选哪个词。这里的随机数来自固定的种子（seed），所以页面每次都给出同样的几次运行。'),
 run:()=>tx('A run is one answer to the prompt: the app draws the first word, adds it to the prompt, and the model scores the next word again.',
            '一次运行就是对 prompt 的一次回答：程序先抽出第一个词，把它接到 prompt 后面，模型再给下一个词打分。'),
 illus:()=>tx('The scores are made up for this lab. Every chance, cut and draw on the page is computed from them exactly. A real model scores its whole vocabulary, often more than 100,000 tokens (Llama 3 has 128,256), and a word may take several tokens.',
              '这些分数是为这个 Lab 编的。页面上的每个概率、每次截断和每次抽取都由它们精确算出。真实模型给整个词表打分（常常超过 100,000 个 token，Llama 3 有 128,256 个），一个词也可能要好几个 token。'),
 scaled:()=>tx('Each logit divided by the temperature. Softmax turns these numbers, not the logits, into the chances: the bigger the gaps between them, the more lopsided the chances.',
               '每个 logit 除以 temperature 之后的数。softmax 把这些数（而不是原来的 logit）变成概率：它们之间差得越多，概率就越悬殊。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'one',t:{en:'1: every run writes the same word',zh:'1 个：每次都写同一个词'}},
  {v:'few',t:{en:'2 or 3: a few different words',zh:'2 到 3 个：少数几个不同的词'}},
  {v:'many',t:{en:'4 to 6: about as many as at temperature 1',zh:'4 到 6 个：和 temperature 为 1 时差不多'}}];
const PRED_RIGHT='one';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const preLock=()=>mode==='guided'&&!pred&&step<=1;       // before the prediction, temperature stays at 1 (R3.8)
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const tIdx=T=>Math.round(T*10), pIdx=P=>Math.round(P*20);
const tCtl=(id,inline)=>A.ui.range({id,key:'T',label:tx('Temperature','temperature'),min:0,max:20,value:tIdx(S.T),out:fT(S.T),inline});
const u1Ctl=(id,note)=>A.ui.range({id,key:'u1',label:tx('First random number','第一个随机数'),min:0,max:99,value:Math.round(u1*100),out:fixd(u1,2),inline:true,
  note:note?tx(`The second and third stay at ${fixd(U23[0],2)} and ${fixd(U23[1],2)}.`,`第二个和第三个随机数固定为 ${fixd(U23[0],2)} 和 ${fixd(U23[1],2)}。`):''});
const LVD={none:{en:'every word stays in the draw',zh:'每个词都留在抽签里'},k3:{en:'keeps the 3 likeliest words on each list',zh:'每个分布都留下最可能的 3 个词'},p9:{en:'keeps the likeliest words until their chances add up to at least 0.9',zh:'从最可能的词开始留，直到概率加起来至少达到 0.9'}};
const lvDesc=id=>L(LVD[id]);
/* the option's result on its title row (R3.6): what it keeps on the flat list, then on the peaked one */
const lvRes=id=>{const f=flatD(id), p=peakD(id); return tx(`keeps ${f.nk} of 6, ${p.nk} of 4`,`平的留 ${f.nk} 个，尖的留 ${p.nk} 个`);};
const lvCtl=()=>`<div class="sp-lv" id="sp-ic-lv">`+A.ui.levers({aria:tx('The rule, pick one','规则，选一个'),value:lv,options:LV.map(v=>({v:v.id,title:L(v.t),desc:lvDesc(v.id),result:esc(lvTried.has(v.id)||v.id===lv?lvRes(v.id):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'strip',view:'a',
  lead:()=>{const b=B0(); return tx(`The same prompt, 20 runs, ${b.distinct} different first words: where does the variety come from?`,`同一个 prompt 跑 20 次，写出 ${b.distinct} 个不同的第一个词：变化从哪来？`);},
  enter:()=>({...BASE}),
  body:()=>{const b=B0();
   return tx(`<p>You send a chat app the same prompt 20 times: “${esc(PROMPT.en)} …”. The ${J('strip','20 runs in the chart')} start with ${b.distinct} different words.</p><p>The model is not what varies: on every run it gives each possible next word the same ${DEF('logit',kt('logit'))}, and ${DEF('softmax',kt('softmax'))} turns them into the same chances (the bars). Then the app ${DEF('draws',TIP.draw())} one word at random with those chances, so each ${DEF('run',TIP.run())} can start differently.</p><p>Two settings shape that draw: ${T_('T','temperature')} reshapes the chances, and ${T_('P','top-p')} cuts off the unlikely words. What does each one do?</p>`,
             `<p>你把同一个 prompt 发给聊天应用 20 次：“${esc(PROMPT.zh)}……”。${J('strip','图里的 20 次运行')}用了 ${b.distinct} 个不同的词开头。</p><p>变的不是模型：每次运行，它给每个可能的下一个词的 ${DEF('logit',kt('logit'))} 都一样，${DEF('softmax',kt('softmax'))} 把它们变成的概率也一样（图里的横条）。然后程序按这些概率随机${DEF('抽一个词',TIP.draw())}，所以每次${DEF('运行',TIP.run())}的开头都可能不同。</p><p>决定这次抽取的有两个设置：${T_('T','temperature（温度）')}改变各个词的概率，${T_('P','top-p')} 去掉不太可能的词。它们各自做了什么？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'strip',view:'a',gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: at temperature 0, how many different first words in 20 runs?','先猜：temperature 为 0 时，20 次运行会写出几个不同的第一个词？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Now the temperature goes from 1 down to 0, and the app runs the same prompt 20 times again. How many different first words will the 20 runs write?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>现在把 temperature 从 1 调到 0，程序把同一个 prompt 再跑 20 次。这 20 次运行会写出几个不同的第一个词？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'temp',title:{en:'Temperature reshapes the chances',zh:'temperature 改变概率的形状'},focus:'bars',view:'a',
  lead:()=>tx('Drag the temperature down to 0 and up to 2: watch the bars and the 20 runs.','把 temperature 拖到 0，再拖到 2：看横条和 20 次运行。'),
  enter:()=>({...BASE}),expect:()=>seenT0,showMe:{T:0},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>${T_('T','Temperature')} divides every logit before softmax (the column ${A.ui.math('\\div T')} in the chart). The model’s logits stay the same; only the division changes.</p>${FXL('Chance of word i',SOFT_TEX)}<p class="small">${M('z_i')} is word i’s logit and ${M('T')} the temperature. The sum in the denominator runs over every word, so the chances add up to 100%.</p>`,
              `<p>做 softmax 之前，每个 logit 都先除以 ${T_('T','temperature')}（图里 ${A.ui.math('\\div T')} 那一列）。模型给的 logit 不变，变的只是这次除法。</p>${FXL('第 i 个词的概率',SOFT_TEX)}<p class="small">${M('z_i')} 是第 i 个词的 logit，${M('T')} 是 temperature。分母把每个词的这一项都加起来，所以概率加起来是 100%。</p>`)+tCtl('sp-ic-T',true),
  obs:()=>obsTemp()},
 {id:'cut',title:{en:'Top-p against top-k',zh:'top-p 和 top-k 比一比'},focus:'duo',view:'duo',
  lead:()=>tx('One rule keeps a fixed number of words, the other a fixed share of the chance.','一个规则留下固定数量的词，另一个留下固定份额的概率。'),
  enter:()=>{lv='none'; lvTried=new Set(['none']); return {...BASE};},expect:()=>lv!=='none',showMe:()=>setLv('p9'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Lists of next words differ in shape. The chances for the first word are flat: four lunches are all good choices. After “a burger” the chances are peaked: “and” (as in “a burger and fries”) takes almost all of the chance.</p><p>${T_('K','Top-k')} keeps a fixed number of the likeliest words; ${T_('P','top-p')} keeps the likeliest words until their chances add up to at least p.</p><p>Pick a rule and see how many words it keeps on the flat list and on the peaked list.</p>`,
              `<p>下一个词的概率分布形状各不相同。第一个词的概率是平的：四种午饭都说得通。“汉堡”之后的概率是尖的：“和”（比如“汉堡和薯条”）几乎拿走了全部概率。</p><p>${T_('K','top-k')} 留下固定数量的最可能的词。${T_('P','top-p')} 从最可能的词开始留，直到概率加起来至少达到 p。</p><p>选一个规则，看看平的和尖的分布上各留下几个词。</p>`)+lvCtl(),
  obs:()=>obsCut()},
 {id:'tree',title:{en:'One early pick changes the rest',zh:'早抽的一个词改变后面的一切'},focus:'tree',view:'b',
  lead:()=>tx('Each pick joins the prompt, and the next word is drawn from its own list.','抽中的词接进 prompt，下一个词从它自己的列表里抽。'),
  enter:()=>{u1=U1_0; u1Moved=false; return {...BASE,view:'b'};},expect:()=>u1Moved&&runPath().i!==0,showMe:()=>setU1(U1_SHOW),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The app adds each word it picks to the prompt, and the model scores the next word again. So the sentence grows like a ${J('tree','tree')}: every first word has its own list of second words, and every second word its own list of third words. The height of a block is the chance of the sentence so far.</p><p>Move where your run’s first random number lands; its second and third random numbers stay the same:</p>`,
              `<p>程序把抽中的每个词接到 prompt 后面，模型再给下一个词打分。所以句子像一棵${J('tree','树')}一样长出来：每个第一个词都有自己的一列第二个词，每个第二个词又有自己的一列第三个词。每一块的高度，就是写到这里的这半句话的概率。</p><p>移动你这次运行的第一个随机数；第二个和第三个随机数不变：</p>`)+u1Ctl('sp-ic-u1',true),
  obs:()=>obsTree()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:()=>({runs:'strip',temp:'bars',cut:'duo',tree:'tree'}[remView]||''),view:()=>({cut:'duo',tree:'b'}[remView]||'a'),
  lead:()=>tx('The model gives chances and a draw picks; temperature reshapes; top-p and top-k cut; early picks steer the rest.','模型给概率，随机抽来选；temperature 改变形状；top-p 和 top-k 截断；先抽的词决定后面。'),
  enter:()=>{remView='runs'; return {...BASE};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
const focusOf=s=>typeof s.focus==='function'?s.focus():(s.focus||'');
/* the verdict on the prediction (about temperature 0), shown once the learner has reached 0 */
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. At temperature 0 the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。temperature 为 0 时，正确答案是“${rt}”。</p>`);
}
/* how many times wider (or narrower) the gaps between the logits get when they are divided by T */
const gapX=T=>{const k=1/T; return Math.abs(k-Math.round(k))<1e-9?String(Math.round(k)):fixd(k,1);};
/* part 3: the result at the learner's temperature, then (at 0) the verdict, then why, pointing at the column of the logits divided by T (R6.25) */
function obsTemp(){
  const r=R(), b=B0(), top=word(TREE[0]), p=r.d1.fin[0], p0=b.d1.fin[0];
  if(S.T===0) return tx(`<p><b>At temperature 0, all 20 runs write “${esc(top)}”.</b></p>`,`<p><b>temperature 为 0 时，20 次运行都写“${esc(top)}”。</b></p>`)+verdict()+
    tx(`<p>Why: a logit cannot be divided by 0 (the column ${A.ui.math('\\div T')} is empty), so engines treat temperature 0 as ${DEF('greedy decoding',TIP.greedy())}. They always take the word with the highest logit, so nothing is left to the draw.${seenHigh?'':' Now drag it above 1.'}</p>`,
       `<p>原因：logit 不能除以 0（${A.ui.math('\\div T')} 那一列是空的），所以引擎把 temperature 0 当作${DEF('贪心解码',TIP.greedy())}，总是取 logit 最高的词，抽签不再起作用。${seenHigh?'':'再把它拖到 1 以上试试。'}</p>`);
  if(S.T===1) return tx(`<p><b>Temperature 1: the chances are the model’s own, and “${esc(top)}” gets about ${fR(p)}.</b> Drag it down to 0 to check your prediction.</p>`,`<p><b>temperature 为 1：概率就是模型原本给的，“${esc(top)}”约占 ${fR(p)}。</b>把它拖到 0，核对你的预测。</p>`);
  const low=S.T<1, rare=r.d1.fin[4]+r.d1.fin[5], rare0=b.d1.fin[4]+b.d1.fin[5];
  const res=low?tx(`<p><b>At temperature ${fTs(S.T)}, “${esc(top)}” gets about ${fR(p)} instead of about ${fR(p0)} at 1: the likeliest word gains.</b></p>`,`<p><b>temperature 为 ${fTs(S.T)} 时，“${esc(top)}”约占 ${fR(p)}，temperature 为 1 时约占 ${fR(p0)}：最可能的词分到更多。</b></p>`)
    :tx(`<p><b>At temperature ${fTs(S.T)}, the bars flatten: the two least likely words together get about ${fR(rare)} instead of about ${fR(rare0)} at 1.</b></p>`,`<p><b>temperature 为 ${fTs(S.T)} 时，横条变平：最不可能的两个词加起来约占 ${fR(rare)}，temperature 为 1 时约占 ${fR(rare0)}。</b></p>`);
  const why=low?tx(`<p>Why: dividing by the temperature makes every gap between the logits ${gapX(S.T)} times as wide (the column ${A.ui.math('\\div T')}), and softmax turns bigger gaps into more lopsided chances. No word is removed: every word keeps some chance.</p>`,`<p>原因：除以这个 temperature，logit 之间的每个差距都变成原来的 ${gapX(S.T)} 倍（${A.ui.math('\\div T')} 那一列），softmax 把更大的差距变成更悬殊的概率。没有哪个词被去掉：每个词都还有一点概率。</p>`)
    :tx(`<p>Why: dividing by the temperature shrinks every gap between the logits to ${fixd(1/S.T,2)} of its size (the column ${A.ui.math('\\div T')}), so the chances move toward equal and unlikely words are drawn more often.</p>`,`<p>原因：除以这个 temperature，logit 之间的每个差距都缩到原来的 ${fixd(1/S.T,2)}（${A.ui.math('\\div T')} 那一列），各个词的概率趋向相等，不太可能的词被抽中得更多。</p>`);
  return res+(seenT0?verdict():tx('<p class="small">Drag it down to 0 to check your prediction.</p>','<p class="small">把它拖到 0，核对你的预测。</p>'))+why;
}
/* part 4: the same rule on the flat list and on the peaked one; where top-k still helps (R6.26) */
function obsCut(){
  if(lv==='none') return tx('<p><b>No cut: every word on both lists can be picked, however unlikely.</b> Pick a rule to compare.</p>','<p><b>不截断：两个分布上的每个词都可能被抽中，不管它多不可能。</b>选一个规则来比较。</p>');
  const f=flatD(lv), p=peakD(lv);
  if(lv==='k3'){const hot=word(TREE[3]);
    return tx(`<p><b>Top-k 3 keeps 3 words on both lists. On the flat list it cuts “${esc(hot)}”, a fine lunch with about ${fR(f.raw[3])}; after “a burger” it still keeps two unlikely words.</b></p><p>Why: top-k counts words, not chance, and the right number of words differs from list to list.</p><p>Top-k still has a job as a cap: when a model is very unsure, top-p can keep thousands of tokens of a large vocabulary, and top-k bounds that number. When it samples, Hugging Face’s generate applies top-k 50 unless the model’s ${cite(SRC.hfcfg,'generation config')} sets another value.</p>`,
              `<p><b>top-k 3 在两个分布上都留下 3 个词。平的分布里，它去掉了“${esc(hot)}”，这顿午饭本来约有 ${fR(f.raw[3])} 的概率；“汉堡”之后，它却还留着两个不太可能的词。</b></p><p>原因：top-k 数的是词的数量，不是概率，而每个分布该留几个词各不相同。</p><p>不过 top-k 仍然有用，可以当上限：模型很没把握时，top-p 可能在很大的词表里留下几千个 token，top-k 能限制这个数量。Hugging Face 的 generate 采样时默认用 top-k 50，除非模型的${cite(SRC.hfcfg,'生成配置')}另设了值。</p>`);}
  return tx(`<p><b>Top-p 0.9 keeps ${f.nk} words on the flat list but only ${p.nk} after “a burger”.</b></p><p>Why: it counts chance, not words. After “a burger”, “and” alone already has more than 90%, so it stops there; the flat list needs ${f.nk} words to get there (the running totals under each list). So top-p keeps more words when the model is unsure and fewer when it is sure.</p>`,
            `<p><b>top-p 0.9 在平的分布里留下 ${f.nk} 个词，在“汉堡”之后只留下 ${p.nk} 个。</b></p><p>原因：它数的是概率，不是词的数量。“汉堡”之后，“和”一个词就超过了 90%，所以到它就停；平的分布要 ${f.nk} 个词才凑够（见每个分布下面的累计）。所以模型没把握时 top-p 留下更多词，有把握时留下更少。</p>`);
}
/* part 5: your run, three words deep */
function obsTree(){
  const r=R(), me=runPath(), base=r.walk([U1_0,U23[0],U23[1]]), a=TREE[me.i];
  if(me.i===base.i) return tx(`<p><b>The first random number, ${fixd(u1,2)}, lands on “${esc(word(a))}”, and your run writes “${esc(sentOf(r,me))}”.</b> Move it until it lands on another first word.</p>`,
                              `<p><b>第一个随机数 ${fixd(u1,2)} 落在“${esc(word(a))}”上，你这次运行写出“${esc(sentOf(r,me))}”。</b>移动它，直到落在另一个第一个词上。</p>`);
  return tx(`<p><b>The first random number now lands on “${esc(word(a))}”, a chance of about ${fR(r.d1.fin[me.i])}, and the whole sentence changes: “${esc(sentOf(r,me))}” instead of “${esc(sentOf(r,base))}”.</b></p><p>Why: the second and third random numbers did not change, but they were drawn from ${esc(word(a))}’s own lists of next words, so what followed “${esc(word(TREE[base.i]))}” can no longer appear. That is why two runs of one prompt drift further apart with every word.</p>`,
            `<p><b>第一个随机数现在落在“${esc(word(a))}”上，它约有 ${fR(r.d1.fin[me.i])} 的概率，整句话都变了：“${esc(sentOf(r,me))}”，而不是“${esc(sentOf(r,base))}”。</b></p><p>原因：第二个和第三个随机数没变，但它们是从“${esc(word(a))}”自己的下一个词列表里抽的，所以“${esc(word(TREE[base.i]))}”后面的那些词不会再出现。这就是同一个 prompt 的两次运行越写越不一样的原因。</p>`);
}
/* part 6: four lines, each a rule and what follows from it (G2, 2026-10-08); a button puts that part's picture back in the chart (R6.23) */
const REMS=['runs','temp','cut','tree'];
/* the first random number the learner last saw land on another word in part 5, or Show me's */
const u1Seen=()=>{const b=B0(), base=b.walk([U1_0,U23[0],U23[1]]); return u1Moved&&b.walk([u1,U23[0],U23[1]]).i!==base.i?u1:U1_SHOW;};
function remBtn(k){const on=remView===k;
  const tip={runs:tx('Puts the 20 runs at temperature 1 back in the chart.','在图里重新显示 temperature 1 时的 20 次运行。'),
    temp:tx('Shows the chart at temperature 0.5, with the logits divided by it.','在图里显示 temperature 0.5 时的概率，以及除以它之后的 logit。'),
    cut:tx('Shows the flat list and the peaked list with top-p 0.9 and their running totals.','在图里显示平的和尖的两个分布，以及 top-p 0.9 下它们的累计概率。'),
    tree:tx('Shows the tree of sentences with your run.','在图里显示句子树和你这次运行。')}[k];
  return ` <button type="button" class="linkbtn sp-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  /* G2 (the owner, 2026-10-08): each line keeps its rule and says what follows from it; no detail of an earlier part to recall.
     The buttons still put each part's picture back in the chart. */
  return tx(`<ul class="sp-remlist"><li><b>The model only gives chances; a random draw picks the word.</b> So the same prompt can give a different answer on every run, with the logits unchanged.${remBtn('runs')}</li>`+
    `<li><b>Temperature reshapes the chances before any cut.</b> Below 1 the likeliest word gains, and at 0 every run writes the same word. Above 1 the chances move toward equal.${remBtn('temp')}</li>`+
    `<li><b>Top-p counts chance, top-k counts words.</b> So top-p keeps many words when the chances are flat and few when they are peaked. Top-k keeps the same number on every list, so it can cut a good word or keep a bad one.${remBtn('cut')}</li>`+
    `<li><b>Early picks steer the rest.</b> Every later word is drawn from the list that the earlier words chose, so one different early pick can change the whole sentence.${remBtn('tree')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Temperature 0 is still random.” In this lab, temperature 0 is greedy: every run writes the likeliest word. A real server can still vary slightly at temperature 0, for a reason that has nothing to do with sampling (${J('fid','see the notes at the end of the page')}).</p><p>✗ “Top-p 0.9 keeps 90% of the words.” It keeps the likeliest words until they hold at least 90% of the chance. On a flat list that can be most of the words, on a peaked list a single word.</p><p>✗ “A higher temperature makes answers better.” It makes them more varied, and it also hands more chance to unlikely words, so odd words turn up more often.</p></div>${A.shell.toScenarios()}`,
    `<ul class="sp-remlist"><li><b>模型只给概率，选哪个词靠随机抽。</b>所以同一个 prompt 每次运行都可能写出不同的回答，哪怕 logit 一点没变。${remBtn('runs')}</li>`+
    `<li><b>temperature 在截断之前改变概率的形状。</b>小于 1 时，最可能的词分到更多。等于 0 时，每次运行都写同一个词。大于 1 时，各个词的概率趋向相等。${remBtn('temp')}</li>`+
    `<li><b>top-p 数概率，top-k 数词。</b>所以概率平的时候，top-p 留下很多词，尖的时候只留几个。top-k 在每个分布上都留同样多的词，可能去掉好词，也可能留下差词。${remBtn('cut')}</li>`+
    `<li><b>先抽的词决定后面。</b>后面每个词都是从前面的词选定的那个列表里抽的，所以前面抽到一个不同的词，整句话都可能变样。${remBtn('tree')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “temperature 为 0 时还是随机的。”在这个 Lab 里，temperature 0 就是贪心，每次运行都写最可能的那个词。真实的服务在 temperature 0 时仍可能有细微差别，但原因和采样无关（${J('fid','见页末的“简化、假设和来源”')}）。</p><p>✗ “top-p 0.9 留下 90% 的词。”它从最可能的词开始留，直到留下的概率至少达到 90%。分布平的时候可能要留下大部分词，尖的时候可能只留一个。</p><p>✗ “temperature 越高，回答越好。”它让回答更多样，也把更多概率分给了不太可能的词，怪词就更常出现。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real uses, typical settings (sources where a default is documented)
   ===================================================================== */
const SRC={
 hfproc:'https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py',
 hfutil:'https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py',
 hfcfg:'https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py',
 vsp:'https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py',
 vsam:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py',
 vtk:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py',
 holtz:'https://arxiv.org/abs/1904.09751',
 fan:'https://aclanthology.org/P18-1082/',
 codex:'https://arxiv.org/abs/2107.03374',
 oai:'https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api',
 llamaMeta:'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/generation_config.json',
 llama:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/generation_config.json',
 tm:'https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/'
};
const PRESETS=[
 {id:'extract',t:{en:'Pull fields out of an invoice',zh:'从发票里提取字段'},d:{en:'A program reads the answer, so every run must give the same one',zh:'程序要读这个回答，所以每次运行都得一样'},s:{T:0,P:1,K:0}},
 {id:'code',t:{en:'Code completion',zh:'代码补全'},d:{en:'Finish the line a developer is typing; the likeliest code is usually right',zh:'补全开发者正在写的那一行；最可能的代码通常就是对的'},s:{T:0.2,P:0.95,K:0}},
 {id:'chat',t:{en:'Chat assistant',zh:'聊天助手'},d:{en:'Answer a user naturally: some variety, but no odd words',zh:'自然地回答用户：有些变化，但不要怪词'},s:{T:0.6,P:0.9,K:0}},
 {id:'brain',t:{en:'Brainstorm names',zh:'头脑风暴起名'},d:{en:'Suggest many different names for a product; variety matters most',zh:'给产品想很多不同的名字；多样性最重要'},s:{T:1.2,P:0.95,K:0}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.T===S.T&&p.s.P===S.P&&p.s.K===S.K);
/* each scenario opens with a full sentence: who uses which settings, then what that leaves on the lunch prompt (R6.21) */
function scenHTML(i){
  const p=PRESETS[i], r=compute({...S,...p.s}), top=esc(word(TREE[0])), n=r.d1.nk;
  const left=tx(`On the lunch prompt, that leaves ${n===1?'only 1 word':`${n} words`} that can be picked.`,`在午饭这个 prompt 上，这样${n===1?'只剩 1 个词':`还剩 ${n} 个词`}可能被抽中。`);
  if(p.id==='extract') return tx(`<p><b>Apps that pull fields out of an invoice usually use temperature 0 (greedy). ${left}</b></p><p>At temperature 0 the engine always takes the top word, so the same invoice gives the same fields on every run, which the program reading them needs. OpenAI’s ${cite(SRC.oai,'prompt-engineering guide')} says temperature 0 is best “for most factual use cases such as data extraction”. A real server can still differ slightly between runs; ${J('fid','Fidelity')} says why.</p>`,
                                 `<p><b>从发票里提取字段的应用，通常用 temperature 0（贪心）。${left}</b></p><p>temperature 为 0 时，引擎总是取最高的那个词，同一张发票每次都给出同样的字段，读取它们的程序正需要这一点。OpenAI 的${cite(SRC.oai,'提示词指南')}说，对数据提取这类大多数事实性用途，temperature 0 最好。真实的服务在不同运行之间仍可能有细微差别，原因见${J('fid','页末的“简化、假设和来源”')}。</p>`);
  if(p.id==='code') return tx(`<p><b>Code completion often uses a low temperature with top-p, for example temperature 0.2 and top-p 0.95. ${left}</b></p><p>A low temperature keeps the draws close to the likeliest code: here “${top}” gets about ${fR(r.d1.fin[0])}. ${cite(SRC.codex,'Chen et al. (2021)')} found temperature 0.2 best for pass@1 (one try must pass the tests) and 0.8 best for pass@100, with a 679M-parameter Codex model on HumanEval, sampling with top-p 0.95. This is a common choice.</p>`,
                              `<p><b>代码补全常用较低的 temperature 加 top-p，比如 temperature 0.2、top-p 0.95。${left}</b></p><p>低 temperature 让抽取贴近最可能的代码：这里“${top}”约占 ${fR(r.d1.fin[0])}。${cite(SRC.codex,'Chen 等（2021）')}用 679M 参数的 Codex 模型在 HumanEval 上测试，发现 pass@1（只试一次就要通过测试）时 temperature 0.2 最好，pass@100 时 0.8 最好，采样都用 top-p 0.95。这是常见的选择。</p>`);
  if(p.id==='chat') return tx(`<p><b>Chat assistants often use a moderate temperature with top-p: Meta ships Llama 3.1 8B Instruct with temperature 0.6 and top-p 0.9. ${left}</b></p><p>A moderate temperature keeps some variety, and top-p drops the unlikely words at the bottom of the list. The defaults are in the model’s ${cite(SRC.llamaMeta,'generation_config.json')} (${cite(SRC.llama,'readable copy')}).</p>`,
                              `<p><b>聊天助手常用适中的 temperature 加 top-p：Meta 发布的 Llama 3.1 8B Instruct 默认用 temperature 0.6、top-p 0.9。${left}</b></p><p>适中的 temperature 保留一些变化，top-p 去掉列表末尾那些不太可能的词。这组默认值写在模型的 ${cite(SRC.llamaMeta,'generation_config.json')} 里（${cite(SRC.llama,'可直接查看的副本')}）。</p>`);
  return tx(`<p><b>Brainstorming often uses a higher temperature, for example 1.2 with top-p 0.95. ${left}</b></p><p>Variety matters more than the single best answer, so a higher temperature spreads the chance, and “${top}” falls to about ${fR(r.d1.fin[0])}. Top-p 0.95 still cuts the least likely word. This is a common choice.</p>`,
            `<p><b>头脑风暴常用较高的 temperature，比如 1.2 加 top-p 0.95。${left}</b></p><p>这里多样性比单个最好的回答更重要，所以较高的 temperature 把概率摊开，“${top}”降到约 ${fR(r.d1.fin[0])}。top-p 0.95 仍然去掉最不可能的那个词。这是常见的选择。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chW=i=>L(CHW[i].w);
const chSet=(T,P)=>tx(`temperature ${fTs(T)}, top-p ${fPp(P)}`,`temperature ${fTs(T)}，top-p ${fPp(P)}`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.T,S.P), best=chBest(), c=chChanges(S.T,S.P);
  let h;
  const good=`<p class="okline sp-mt">`+(me.wrong<=0?tx(`<b>Goal met: top-p cuts all three wrong words, and ${me.n5} good words keep at least 5% each.</b>`,`<b>达成目标：top-p 去掉了全部三个错词，有 ${me.n5} 个好词各保留至少 5%。</b>`)
    :tx(`<b>Goal met: the three wrong words together are picked ${fP(me.wrong)} of the time, and ${me.n5} good words keep at least 5% each.</b>`,`<b>达成目标：三个错词加起来被抽中的概率是 ${fP(me.wrong)}，有 ${me.n5} 个好词各保留至少 5%。</b>`));
  if(me.ok&&S.T===best.T&&S.P===best.P) h=good+tx(' And that is the smallest change: only top-p moved, and it still keeps all four good words.</p>','而且这是改动最少的做法：只动了 top-p，还留住了全部四个好词。</p>');
  else if(me.ok&&c===1) h=good+`</p>`+tx(`<p>But you cut more than needed: top-p ${fixd(best.P,2)} also meets the goal and keeps “${esc(chW(3))}” too.</p>`,`<p>不过你截得比需要的多：top-p ${fixd(best.P,2)} 也能达标，还能留住“${esc(chW(3))}”。</p>`);
  else if(me.ok) h=good+`</p>`+tx(`<p>But one change is enough: keep the temperature at ${fTs(best.T)} and set only top-p to ${fixd(best.P,2)}.</p>`,`<p>不过改一处就够了：temperature 保持 ${fTs(best.T)}，只把 top-p 设成 ${fixd(best.P,2)}。</p>`);
  else h=(me.wrong>=CH.wrongMax?tx(`<p class="badline sp-mt"><b>Not yet: the three wrong words together are picked ${fP(me.wrong)} of the time, not below 1%.</b></p>`,`<p class="badline sp-mt"><b>还没达到：三个错词加起来被抽中的概率是 ${fP(me.wrong)}，没有低于 1%。</b></p>`)
      :tx(`<p class="badline sp-mt"><b>Not yet: only ${me.n5===1?'1 good word keeps':`${me.n5} good words keep`} at least 5%, not 3.</b></p>`,`<p class="badline sp-mt"><b>还没达到：只有 ${me.n5} 个好词保留至少 5%，不到 3 个。</b></p>`))+
      tx(`<p class="small">Hint: a lower temperature shrinks the wrong words’ chances, but it also shrinks the chances of “${esc(chW(2))}” and “${esc(chW(3))}”. Top-p cuts the unlikely words at the bottom of the list and scales the kept words up without changing how they compare. Which top-p stops right after the last good word?</p>`,
         `<p class="small">提示：降低 temperature 会让错词的概率变小，可“${esc(chW(2))}”和“${esc(chW(3))}”的概率也跟着变小。top-p 去掉列表末尾那些不太可能的词，再把留下的词按比例放大，它们彼此之间的比例不变。哪个 top-p 刚好在最后一个好词之后停下？</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const rows=[]; CH.Ts.forEach(T=>CH.Ps.forEach(P=>rows.push(chEval(T,P))));
  /* a pass is marked on the row itself (✓, green label, green outline), since a passing bar is 0% wide (R5.4) */
  /* the owner's review of batch 8 (2026-10-07): brief, so the chart keeps the settings that meet the goal, the pick and the initial
     setup, and a note under it counts the rest */
  const isMe=e=>e.T===S.T&&e.P===S.P, isStart=e=>e.T===CH.start.T&&e.P===CH.start.P;
  h+=A.ui.cmpChart({title:tx(`Chance of picking one of the three wrong words after “${CHP.en}” (goal: below 1%, with at least 3 good words at 5% or more)`,`“${CHP.zh}”之后抽中三个错词之一的概率（目标：低于 1%，且至少 3 个好词各有 5% 以上）`),
    aria:tx('Chance of the wrong words: ','抽中错词的概率：')+rows.filter(e=>e.ok||isMe(e)||isStart(e)).map(e=>`${chSet(e.T,e.P)}${tx(': ','：')}${fP(e.wrong)}${e.ok?tx(', meets both targets','，两个目标都达到'):''}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:nbh(tx(`${e.ok?'✓ ':''}T ${fTs(e.T)}, top-p ${e.P>=1?'off':fixd(e.P,2)}: ${e.n5} good at 5%+`,`${e.ok?'✓ ':''}T ${fTs(e.T)}，top-p ${e.P>=1?'不截断':fixd(e.P,2)}：${e.n5} 个好词 ≥ 5%`)),value:e.wrong*100,text:fP(e.wrong),ok:e.ok,me:e.T===S.T&&e.P===S.P,
      tags:(e.T===CH.start.T&&e.P===CH.start.P?[{text:tx('initial setup','初始设置')}]:[]).concat(e.T===S.T&&e.P===S.P?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,goal:1,goalLabel:tx('goal: below 1%','目标：低于 1%'),max:14,okText:tx('✓ meets both targets','✓ 两个目标都达到'),noText:tx('misses at least one','至少差一个'),goalText:tx('goal: below 1%','目标：低于 1%')});
  const st=chEval(CH.start.T,CH.start.P), g4=dist(CHZ,1,0,1).cum[3];
  h+=tx(`<p class="small">In the initial setup every word can be picked, and the three wrong ones together get ${fR(st.wrong)}.</p><p class="small">A lower temperature alone never works: by the time the wrong words fall below the goal, fewer than three good words keep 5%.</p>`,
        `<p class="small">初始设置下每个词都可能被抽中，三个错词加起来占 ${fR(st.wrong)}。</p><p class="small">只降低 temperature 永远不行。等错词降到目标以下时，保留 5% 以上的好词已经不到三个了。</p>`);
  h+=tx(`<p class="small">Why top-p ${fixd(best.P,2)} works: the four good words already hold ${fR(g4)} of the chance (the last running total under the chart’s bar), so the cut stops right after them.</p><p class="small">A higher temperature pushes more words inside the cut: at temperature 1.3 the same top-p lets “${esc(chW(4))}” in.</p>`,
        `<p class="small">top-p ${fixd(best.P,2)} 为什么行：四个好词加起来已经占 ${fR(g4)}（图里概率条下面最后一个累计），截断正好停在它们后面。</p><p class="small">temperature 更高时，更多的词挤进截断线以内：temperature 为 1.3 时，同样的 top-p 会放进“${esc(chW(4))}”。</p>`);
  h+=tx('<p class="small muted">Note: here the wrong words sit below the good ones. In a real model a wrong word can rank high, and then no sampling setting removes it; and the same top-p keeps a different number of words for every prompt. Check the settings on examples from your own app before you ship them.</p>',
        '<p class="small muted">注意：这里的错词都排在好词后面。真实模型里，错词也可能排在前面，那时任何采样设置都去不掉它；而且同一个 top-p 在每个 prompt 上留下的词数都不一样。上线之前，要拿自己应用里的例子检查这些设置。</p>');
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===6?'Six':fint(STEPS.length)} short parts. Predict first, then see how temperature reshapes the chances, how top-p and top-k cut them, and how one early pick changes the whole sentence. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看 temperature 怎样改变概率的形状，top-p 和 top-k 怎样截断，以及早抽的一个词怎样改变整句话。第一次学从这里开始。`),
  sandbox:tx('Four real uses, from pulling fields out of an invoice to brainstorming names. Pick one to see its typical settings and what they do to the draws.',
             '四个真实用途，从提取发票字段到头脑风暴起名。选一个，看它的常用设置对抽取有什么影响。'),
  challenge:tx(`A new prompt, “${CHP.en}”. Choose the temperature and top-p so that the three wrong words together are picked less than 1% of the time, while at least 3 of the good words keep a chance of at least 5% each.`,
               `换一个新的 prompt“${CHP.zh}”，选好 temperature 和 top-p，让三个错词加起来被抽中的概率低于 1%，同时至少 3 个好词各保留至少 5% 的概率。`)
});
/* the Fidelity popover can be opened before part 2 is answered, so it never says what temperature 0 does (R3.8) */
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> every chance on this page is computed exactly from a small table of scores made up for this lab, and every draw comes from a seeded random number generator; ${J('math','“Show the math”')} works them out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>One step: the chances ${A.ui.math('\\text{softmax}(z / T)')}, where z are the logits and T the temperature, then top-k, then top-p, each followed by rescaling to 100%. This is the order Hugging Face’s generate and vLLM’s sampler use. Temperature 0 is a special case, shown in part 3.</li>`+
 `<li>Top-p keeps the smallest set of likeliest words whose chances add up to at least p; top-k keeps the k likeliest.</li>`+
 `<li>A run draws three random numbers and picks, at each step, the word whose stretch of the cumulative chances each one lands in. The random numbers come from mulberry32 seeded with ${A.ui.math('\\text{seed} \\times 7919 + 17')}; each run takes the next three numbers. The 20 runs use seed 1 until you ask for new draws.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The scores are made up for this lab, and each list has 2 to 7 words, not a whole vocabulary, which often has more than 100,000 tokens (Llama 3 has 128,256). A word or phrase such as “hot pot” counts as one step here; a real tokenizer may split it.</li>`+
 `<li id="sp-fid-nd">The model gives the same logits on every run. On a real server, two runs with the same settings and the same random numbers can still differ: the batch a request shares with others changes the order of floating-point sums in the kernels (${cite(SRC.tm,'Thinking Machines, 2025')}). That changes the logits, not the sampling rules shown here.</li>`+
 `<li>Engines draw in other ways. vLLM hands top-k and top-p to FlashInfer’s sampler on NVIDIA GPUs where it is available; otherwise, or when a request sets its own seed, it divides the chances by random exponential noise and takes the largest. Each word is still picked with exactly the same chance as with the method on this page.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Top-p: ${cite(SRC.holtz,'Holtzman et al., ICLR 2020')}. Top-k: ${cite(SRC.fan,'Fan, Lewis and Dauphin, ACL 2018')}, who sampled from the 10 likeliest words.</li>`+
 `<li>The order and the rules: Hugging Face ${cite(SRC.hfproc,'<code>logits_process.py</code>')} and ${cite(SRC.hfutil,'<code>utils.py</code>')}, with the top-k default in ${cite(SRC.hfcfg,'<code>configuration_utils.py</code>')}; vLLM ${cite(SRC.vsp,'<code>sampling_params.py</code>')} and ${cite(SRC.vtk,'<code>topk_topp_sampler.py</code>')}.</li>`+
 `<li>Typical settings in Scenarios: OpenAI’s ${cite(SRC.oai,'prompt-engineering guide')} (data extraction); ${cite(SRC.codex,'Chen et al., 2021')} (code); Meta’s ${cite(SRC.llamaMeta,'Llama 3.1 8B Instruct generation_config.json')} (chat; ${cite(SRC.llama,'readable copy')}).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-30. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>这一页上的每个概率，都由一张为这个 Lab 编的小分数表精确算出；每次抽取都来自带种子的随机数生成器。${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>一步：先算概率 ${A.ui.math('\\text{softmax}(z / T)')}（z 是 logit，T 是 temperature），再做 top-k，再做 top-p，每次截断后都按比例放大回 100%。Hugging Face 的 generate 和 vLLM 的采样器都是这个顺序。temperature 为 0 是特殊情况，第 3 部分会讲到。</li>`+
 `<li>top-p 留下概率加起来至少为 p 的最少几个最可能的词；top-k 留下最可能的 k 个词。</li>`+
 `<li>一次运行抽三个随机数；每一步，随机数落在累计概率的哪一段，就选那个词。随机数来自 mulberry32 生成器，初始值为 ${A.ui.math('\\text{种子} \\times 7919 + 17')}；每次运行依次取接下来的三个随机数。20 次运行用种子 1，直到你要求重新抽。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>分数是本 Lab 编的；每个列表只有 2 到 7 个词，而不是整个词表（常常超过 100,000 个 token，Llama 3 有 128,256 个）。“火锅”这样的词在这里算一步；真实的 tokenizer 可能把它切开。</li>`+
 `<li id="sp-fid-nd">模型每次运行给出的 logit 都一样。真实的服务上，设置相同、随机数也相同的两次运行仍可能不同：一个请求和别的请求拼在同一个 batch 里，会改变 kernel 里浮点数相加的顺序（${cite(SRC.tm,'Thinking Machines，2025')}）。变的是 logit，不是这里讲的采样规则。</li>`+
 `<li>引擎抽取的方法各不相同。在支持 FlashInfer 采样器的 NVIDIA GPU 上，vLLM 把 top-k 和 top-p 交给它做；否则，或者请求自带种子时，它把概率除以随机的指数噪声，再取最大的那个。换一种抽法，每个词被抽中的概率和这页的方法完全一样。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>top-p：${cite(SRC.holtz,'Holtzman 等，ICLR 2020')}。top-k：${cite(SRC.fan,'Fan、Lewis 和 Dauphin，ACL 2018')}，他们从最可能的 10 个词里抽。</li>`+
 `<li>顺序和规则：Hugging Face 的 ${cite(SRC.hfproc,'<code>logits_process.py</code>')} 和 ${cite(SRC.hfutil,'<code>utils.py</code>')}，top-k 的默认值在 ${cite(SRC.hfcfg,'<code>configuration_utils.py</code>')}；vLLM 的 ${cite(SRC.vsp,'<code>sampling_params.py</code>')} 和 ${cite(SRC.vtk,'<code>topk_topp_sampler.py</code>')}。</li>`+
 `<li>应用场景里的常用设置：OpenAI 的${cite(SRC.oai,'提示词指南')}（数据提取）；${cite(SRC.codex,'Chen 等，2021')}（代码）；Meta 的 ${cite(SRC.llamaMeta,'Llama 3.1 8B Instruct generation_config.json')}（聊天；${cite(SRC.llama,'可直接查看的副本')}）。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-30。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A model ends with the ${DEF('LM head',kt('lmhead'))}, which gives every token in the vocabulary a score, a ${DEF('logit',kt('logit'))}. ${DEF('Softmax',kt('softmax'))} turns the logits into chances that add up to 100%. One token is picked, added to the input, and the model runs again for the next one.</p><p>The lab ${labLink('architecture')} shows this loop from the start. This lab looks only at the pick.</p>`,
                    `<p>模型最后是 ${DEF('LM head',kt('lmhead'))}，它给词表里每个 token 一个得分，也就是 ${DEF('logit',kt('logit'))}。${DEF('softmax',kt('softmax'))} 把 logit 变成加起来是 100% 的概率。选出一个 token，接到输入后面，模型再跑一遍，算下一个。</p><p>${labLink('architecture')}这个 Lab 从头讲这个循环。这个 Lab 只看“选”这一步。</p>`);
const OUTCOME=()=>tx('predict how temperature, top-k and top-p reshape next-token chances, and pick settings for a given use.','预测 temperature、top-k 和 top-p 怎样改变下一个 token 的概率，并为具体用途选出设置。');
const XA=[0.45,0.25,0.15,0.10,0.05], XP=0.75;
const EXITS=()=>{
  const xd=dist(XA.map(v=>Math.log(v)),1,0,XP), xk=xd.nk;                // the chances as logits (log p): softmax gives them back
  const t2=softmax([2/2,0]), t1=softmax([2,0]);
  const b=B0(), sal=b.d1.fin[4];
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('At temperature 0, all 20 runs of the lunch prompt write the same first word. Why?','temperature 为 0 时，午饭这个 prompt 每次运行都写同一个第一个词。为什么？'),
   o:[tx('Dividing the logits by a temperature near 0 stretches every gap, so the top word’s chance goes to 100%; at 0 engines simply take the top word.','logit 除以接近 0 的 temperature，每个差距都被拉得很大，最高那个词的概率趋向 100%；等于 0 时引擎直接取最高的词。'),
      tx('Temperature sets how much random noise the model adds while it computes; at 0 there is no noise, so the model computes the same logits and the same word every time.','temperature 决定模型计算时加进多少随机噪声；等于 0 时没有噪声，所以模型每次算出同样的 logit、同样的词。'),
      tx('Temperature sets how many of the likeliest words stay in the draw, like top-k; at 0 only one word stays, so the draw has a single option.','temperature 像 top-k 一样决定留下几个最可能的词；等于 0 时只留一个，所以抽签只有一个选项。')],a:0,
   n:[null,tx('The model is not what varies: it gives the same logits on every run, at any temperature. Temperature only changes how those logits are turned into chances for the draw.','变的不是模型：不管 temperature 是多少，它每次运行给出的 logit 都一样。temperature 只改变这些 logit 怎样变成抽签用的概率。'),
     tx(`At any temperature above 0, temperature never removes a word; it reweights them all. At temperature 1 “${word(TREE[4])}” still has ${fP(sal)}, and even at 0.1 it keeps a tiny chance.`,`只要 temperature 大于 0，它就不会去掉任何词，只是重新分配概率。temperature 为 1 时“${word(TREE[4])}”仍有 ${fP(sal)}，就算是 0.1 它也还有一点点概率。`)],
   why:tx(`The chances are ${A.ui.math('\\text{softmax}(z / T)')}, where z are the logits. Halving T doubles every gap between them, and each doubling makes the top word’s share more lopsided. Engines treat T = 0 as greedy decoding: they take the word with the highest logit without drawing.`,`概率是 ${A.ui.math('\\text{softmax}(z / T)')}，其中 z 是各个词的 logit。T 减半，logit 之间的每个差距就翻倍，最高那个词的份额也更悬殊。引擎把 T = 0 当作贪心解码：不抽签，直接取 logit 最高的词。`)},
  {q:tx('Why does top-p 0.9 keep 4 of the 6 lunch words but only 1 of the 4 words after “a burger”?','top-p 0.9 在 6 个午饭词里留下 4 个，在“汉堡”之后的 4 个词里却只留下 1 个。为什么？'),
   o:[tx('Top-p keeps the likeliest words until their chances add up to 0.9: after “a burger” the top word alone is above 0.9, while the flat lunch list needs four words to get there.','top-p 从最可能的词开始留，直到概率加起来达到 0.9：“汉堡”之后，最高那个词自己就超过 0.9，而平的午饭分布要四个词才够。'),
      tx('Top-p keeps every word whose own chance is above 1 − 0.9 = 10%: after “a burger” only one word is above 10%, while four lunch words are above it.','top-p 留下自身概率高于 1 − 0.9 = 10% 的每个词：“汉堡”之后只有一个词高于 10%，午饭词里有四个高于它。'),
      tx('Top-p keeps about 90% of the words, counted from the top: it rounds 90% of each list down, and the burger list is short, so it keeps fewer.','top-p 从上往下留下大约 90% 的词：每个列表的 90% 向下取整，“汉堡”那个列表短，所以留得少。')],a:0,
   n:[null,tx(`That rule would keep only three lunch words: “${word(TREE[3])}” has about ${fR(b.d1.fin[3])}, below 10%, yet top-p 0.9 keeps it, because the first three add up to only ${fR(b.d1.cum[2])}.`,`按这个规则，午饭词只会留下三个：“${word(TREE[3])}”约占 ${fR(b.d1.fin[3])}，低于 10%，可 top-p 0.9 留下了它，因为前三个加起来只有 ${fR(b.d1.cum[2])}。`),
     tx('Top-p counts chance, not words: 90% of 6 would be 5 lunch words and 90% of 4 would be 3 burger words, not 4 and 1.','top-p 数的是概率，不是词的数量：6 的 90% 是 5 个午饭词，4 的 90% 是 3 个，而不是 4 和 1。')],
   why:tx('Top-p fixes the share of the chance it keeps, not the number of words, so the set grows when the model is unsure and shrinks when it is sure. Top-k does the opposite: it keeps the same number of words on every list.','top-p 固定的是留下的概率份额，不是词的数量，所以模型没把握时留下的词多，有把握时留下的少。top-k 正好相反：每个列表都留下同样多的词。')}],
 transfer:[
  {q:tx(`A model gives the next token these chances: A 45%, B 25%, C 15%, D 10%, E 5%. With top-p ${fixd(XP,2)}, which tokens can still be picked?`,`模型给下一个 token 的概率是：A 45%，B 25%，C 15%，D 10%，E 5%。用 top-p ${fixd(XP,2)} 时，哪些 token 还可能被抽中？`),
   o:[tx('A only','只有 A'),tx('A and B','A 和 B'),tx('A, B and C','A、B 和 C'),tx('A, B, C and D','A、B、C 和 D')],a:xk-1,
   why:tx(`Add the chances from the top until the total reaches ${fixd(XP,2)}: A gives 45%, A and B give 70%, still below 75%, so C joins and the total is 85%. The kept set is A, B and C, rescaled to ${fixd(0.45/0.85*100,1)}%, ${fixd(0.25/0.85*100,1)}% and ${fixd(0.15/0.85*100,1)}%.`,
          `从最高的开始累加，直到达到 ${fixd(XP,2)}：A 是 45%，A 加 B 是 70%，还不到 75%，所以 C 也留下，总和 85%。留下 A、B、C，放大后分别是 ${fixd(0.45/0.85*100,1)}%、${fixd(0.25/0.85*100,1)}% 和 ${fixd(0.15/0.85*100,1)}%。`)},
  {q:tx(`Two tokens have logits 2 and 0. At temperature 1 the first gets ${fixd(t1[0]*100,0)}%. At temperature 2, about what share does it get?`,`两个 token 的 logit 是 2 和 0。temperature 为 1 时，第一个占 ${fixd(t1[0]*100,0)}%。temperature 为 2 时，它大约占多少？`),
   o:[tx('About 50%','约 50%'),tx('About 73%','约 73%'),tx('About 88%','约 88%'),tx('About 98%','约 98%')],a:1,
   why:tx(`Dividing by 2 halves the gap between the logits, from 2 to 1. The first token then gets ${A.ui.math(`\\frac{e^1}{e^1 + e^0} = \\frac{2.718}{3.718} = ${fixd(t2[0]*100,1)}\\%`)}: still the favourite, but less lopsided than ${fixd(t1[0]*100,0)}%.`,
          `除以 2 让两个 logit 的差距从 2 减半到 1。第一个 token 占 ${A.ui.math(`\\frac{e^1}{e^1 + e^0} = \\frac{2.718}{3.718} = ${fixd(t2[0]*100,1)}\\%`)}：它仍然最可能，但没有 ${fixd(t1[0]*100,0)}% 那么悬殊。`)}],
 practice:[
  {id:'llm-fundamentals',d:tx('Question 11: define greedy, temperature, top-k and top-p sampling, each as a change to the next-token distribution, and find the exact set top-p keeps for a five-token example.','第 11 题：把贪心、temperature、top-k 和 top-p 采样分别定义成对下一个 token 分布的一次变换，并算出一个五个 token 的例子里 top-p 留下的准确集合。')},
  {id:'cross-entropy',d:tx('Part 1: a softmax that stays finite for logits as large as 1,000. Dividing logits by a small temperature makes them that large, so a sampler needs the same trick.','第 1 部分：写一个 logit 大到 1,000 也不溢出的 softmax。logit 除以很小的 temperature 就会变得这么大，所以采样器也需要这个技巧。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Sampling','采样'),title:l?L(l.q):tx('What do temperature and top-p actually do?','Temperature 和 top-p 到底做了什么？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: logits and softmax','前置知识：logit 和 softmax'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage sp-stage" id="sp-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="sp-st-title"></div><div class="meter-sub" id="sp-st-sub"></div></div>`+
   `<div class="stage-actions"><div class="seg-ctl sp-view" id="sp-view" role="group" aria-label="${esc(tx('What the chart shows','图里显示什么'))}"><button type="button" data-view="a" aria-pressed="true">${tx('Next word','下一个词')}</button><button type="button" data-view="b" aria-pressed="false">${tx('Sentence tree','句子树')}</button></div><span class="status" id="sp-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="sp-legend"></div>`+
   `<div class="stage-graph graph sp-graph" id="sp-graph">`+
    `<div class="sp-phone" id="sp-phone"></div>`+
    `<div class="sp-a" id="sp-a"><div class="sp-cap" id="sp-a-cap"></div><div class="sp-rows" id="sp-rows"></div>`+
     `<div class="sp-cum" id="sp-cum"><div class="sp-sub" id="sp-cum-cap"></div><div class="sp-bar sp-cumbar" id="sp-cumbar" tabindex="0"></div><div class="sp-marks" id="sp-cum-marks"></div></div>`+
     `<div class="sp-cum" id="sp-draw"><div class="sp-sub" id="sp-draw-cap"></div><div class="sp-bar sp-drawbar" id="sp-drawbar" tabindex="0"></div></div>`+
     `<div class="sp-strip" id="sp-strip"><div class="sp-sub" id="sp-strip-cap"></div><div class="sp-chips" id="sp-chips"></div></div></div>`+
    `<div class="sp-duo" id="sp-duo" hidden><div class="sp-duo-p" id="sp-duo-f"></div><div class="sp-duo-p" id="sp-duo-k"></div></div>`+
    `<div class="sp-b" id="sp-b" hidden><div class="sp-cap" id="sp-b-cap"></div><div class="sp-tree" id="sp-tree"></div><p class="sp-sent" id="sp-sent"></p></div>`+
   `</div>`+
   `<p class="sp-scope small muted" tabindex="0" data-tipx="${esc(TIP.illus())}">${tx('Illustrative scores; every chance is computed from them exactly.','示意的分数；每个概率都由它们精确算出。')}</p>`+
   `<div class="metrics sp-metrics" id="sp-metrics"></div>`+
   `<details class="math" id="sp-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="sp-mathbody"></div></details></div>`;
}
/* every logit of the tree, one line per first word (S6: the assumed values behind every chance on the page) */
function zListHTML(){
  const z=v=>fixd(v,1), sep=tx(', ','，'), sep2=tx('; ','；'), op=tx(' (','（'), cl=tx(')','）');
  const li=TREE.map(a=>`<li><b>${esc(word(a))} ${z(a.z)}</b> → ${a.c.map(b=>`${esc(word(b))} ${z(b.z)}${op}${b.c.map(c=>`${esc(word(c))} ${z(c.z)}`).join(sep)}${cl}`).join(sep2)}</li>`).join('');
  return `<details class="sp-zlist"><summary>${tx('All the logits in the tree','句子树里的全部 logit')}</summary><p>${tx('Each first word with its logit, then the words that can follow it, then the words that can follow those.','每个第一个词和它的 logit，后面是能接在它后面的词，括号里是能接在那些词后面的词。')}</p><ul>${li}</ul></details>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="sp-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="sp-lock" hidden></p>`+
   `<div class="lockable" id="sp-ctl-T-wrap">`+tCtl('sp-ctl-T',false).replace(/<\/div>$/,`<div class="ctl-note">${tx('Every logit is divided by it before softmax. 0 means greedy: always the top word.','做 softmax 之前每个 logit 都除以它。0 表示贪心：总是取最高的词。')}</div></div>`)+`</div>`+
   `<div class="lockable" id="sp-ctl-P-wrap">`+A.ui.range({id:'sp-ctl-P',key:'P',label:tx('Top-p','top-p'),min:1,max:20,value:pIdx(S.P),out:fPp(S.P),note:tx('Keeps the likeliest words until their chances add up to at least this. 1.00 keeps every word.','从最可能的词开始留，直到概率加起来至少达到这个值。1.00 表示一个也不去掉。')})+`</div>`+
   `<div class="lockable" id="sp-ctl-K-wrap">`+A.ui.seg({id:'sp-ctl-K',key:'K',label:tx('Top-k','top-k'),options:[[0,tx('off','不限')],[1,'1'],[2,'2'],[3,'3'],[4,'4'],[5,'5']],value:S.K,note:tx('Keeps only this many of the likeliest words. Hugging Face and vLLM apply it before top-p.','只留下这么多个最可能的词。Hugging Face 和 vLLM 都先做它，再做 top-p。')})+`</div>`+
   `<div class="lockable sp-seedrow" id="sp-ctl-seed"><div class="ctl"><span class="lbl">${tx('The 20 runs','20 次运行')} <output class="mono" id="sp-seed-out"></output></span>`+
    `<button type="button" class="btn ghost sm" data-act="reseed">${tx('New draws','重新抽')}</button>`+
    `<div class="ctl-note">${tx(`The random numbers come from a seed, so the page shows the same 20 runs every time: mulberry32 seeded with ${A.ui.math('\\text{seed} \\times 7919 + 17')}, each run taking the next three numbers. This is assumed so that every result can be checked; a new seed gives 20 new runs with the same chances.`,`随机数来自一个种子，所以页面每次都给出同样的 20 次运行：生成器是 mulberry32，初始值为 ${A.ui.math('\\text{种子} \\times 7919 + 17')}，每次运行依次取接下来的三个随机数。这样设定是为了每个结果都能核对；换一个种子，就得到 20 次新的运行，概率不变。`)}</div></div></div>`+
   `<div class="sp-grp lockable" id="sp-ctl-z"><div class="sp-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('The logits','logit')}</span><div class="ctl-note">${tx('These are made up for this lab; a real model would compute them from the prompt, over its whole vocabulary. They are chosen so that a few words lead and many unlikely words follow, the usual shape of a next-token distribution.','这些是为这个 Lab 编的；真实模型会根据 prompt 为整个词表算出它们。选这些数，是为了让少数几个词领先、后面跟着许多不太可能的词，这是下一个 token 分布常见的形状。')}</div>${zListHTML()}</div></div>`+
   `<div class="lockable" id="sp-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

def probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    z = np.asarray(logits, dtype=float)
    if temperature == 0:              # greedy: all on the top logit
        p = np.zeros_like(z)
        p[np.argmax(z)] = 1.0
        return p
    z = z / temperature               # T < 1 sharpens, T > 1 flattens
    p = np.exp(z - z.max())           # softmax, stable for any T
    p /= p.sum()
    order = np.argsort(-p, kind="stable")      # likeliest first
    if top_k > 0:                     # keep the k likeliest
        p[order[top_k:]] = 0.0
        p /= p.sum()
    if top_p < 1.0:                   # smallest set reaching top_p
        cum = np.cumsum(p[order])
        n = int(np.searchsorted(cum, top_p)) + 1
        p[order[n:]] = 0.0
        p /= p.sum()
    return p

def sample(p, rng):                   # where the random number lands
    return int(np.searchsorted(np.cumsum(p), rng.random(), "right"))

logits = [2.0, 1.5, 1.0, 0.3, -0.5, -1.0]   # the lunch prompt
for t in (0, 0.5, 1.0):
    print(t, np.round(probs(logits, t, top_p=0.9), 3))
rng = np.random.default_rng(1)   # NumPy's draws, not the page's runs
print([sample(probs(logits, 1.0), rng) for _ in range(20)])
# 0 [1. 0. 0. 0. 0. 0.]
# 0.5 [0.665 0.245 0.09  0.    0.    0.   ]
# 1.0 [0.464 0.281 0.171 0.085 0.    0.   ]
# [1, 4, 0, 4, 0, 0, 2, 0, 1, 0, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0]`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="sp-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="sp-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s rules: temperature, then top-k, then top-p, then one draw.','这个 Lab 的规则：先 temperature，再 top-k，再 top-p，然后抽一次。')}</p><pre class="code" id="sp-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="sp-prod">`+
   tx(`<li>vLLM’s ${cite(SRC.vsp,'<code>vllm/sampling_params.py</code>')} holds the settings of each request: <code>temperature</code> (“Zero means greedy sampling”; below 1e-5 it also resets top-p and top-k), <code>top_p</code>, <code>top_k</code> (0 or −1 keeps every token) and <code>seed</code>.</li>`+
      `<li>Its sampler, ${cite(SRC.vsam,'<code>vllm/v1/sample/sampler.py</code>')}, divides the logits by the temperature. On NVIDIA GPUs where FlashInfer’s sampler is available, ${cite(SRC.vtk,'<code>vllm/v1/sample/ops/topk_topp_sampler.py</code>')} hands top-k and top-p to it, and it draws from the same chances without sorting. Otherwise, or when a request sets its own seed, the PyTorch path sorts the chances in ascending order and drops every token whose running total is at most 1 − p, always keeping the top one.</li>`+
      `<li>Hugging Face’s ${cite(SRC.hfproc,'<code>generation/logits_process.py</code>')} does the same with <code>TemperatureLogitsWarper</code> (the temperature must be strictly positive; greedy is <code>do_sample=False</code>), <code>TopKLogitsWarper</code> and <code>TopPLogitsWarper</code>; ${cite(SRC.hfutil,'<code>generation/utils.py</code>')} applies them in that order. When it samples, it applies top-k 50 unless the model’s generation config sets another value (${cite(SRC.hfcfg,'<code>configuration_utils.py</code>')}).</li>`,
      `<li>vLLM 的 ${cite(SRC.vsp,'<code>vllm/sampling_params.py</code>')} 保存每个请求的设置：<code>temperature</code>（“Zero means greedy sampling”；低于 1e-5 时还会重置 top-p 和 top-k）、<code>top_p</code>、<code>top_k</code>（0 或 −1 表示保留全部 token）和 <code>seed</code>。</li>`+
      `<li>它的采样器 ${cite(SRC.vsam,'<code>vllm/v1/sample/sampler.py</code>')} 先把 logit 除以 temperature。在支持 FlashInfer 采样器的 NVIDIA GPU 上，${cite(SRC.vtk,'<code>vllm/v1/sample/ops/topk_topp_sampler.py</code>')} 把 top-k 和 top-p 交给它，它不排序，按同样的概率抽。否则，或者请求自带种子时，走 PyTorch 路径：把概率从小到大排序，去掉累计和不超过 1 − p 的每个 token，最高的那个总会留下。</li>`+
      `<li>Hugging Face 的 ${cite(SRC.hfproc,'<code>generation/logits_process.py</code>')} 用 <code>TemperatureLogitsWarper</code>（temperature 必须大于 0；贪心是 <code>do_sample=False</code>）、<code>TopKLogitsWarper</code> 和 <code>TopPLogitsWarper</code> 做同样的事；${cite(SRC.hfutil,'<code>generation/utils.py</code>')} 按这个顺序调用它们。采样时它默认用 top-k 50，除非模型的生成配置另设了值（${cite(SRC.hfcfg,'<code>configuration_utils.py</code>')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.holtz,'The Curious Case of Neural Text Degeneration')}<span>${tx('Holtzman, Buys, Du, Forbes and Choi, ICLR 2020: nucleus (top-p) sampling, which keeps the smallest set of likeliest tokens that holds most of the chance and cuts the many unlikely tokens below it','Holtzman、Buys、Du、Forbes、Choi，ICLR 2020：核采样（top-p），保留占了大部分概率的最少几个最可能的 token，去掉下面那一大批不太可能的 token')}</span></li>`+
   `<li>${cite(SRC.fan,'Hierarchical Neural Story Generation')}<span>${tx('Fan, Lewis and Dauphin, ACL 2018: stories sampled from the 10 likeliest words at each step (top-k)','Fan、Lewis、Dauphin，ACL 2018：每一步从最可能的 10 个词里抽（top-k）来写故事')}</span></li>`+
   `<li>${cite(SRC.codex,'Evaluating Large Language Models Trained on Code')}<span>${tx('Chen et al., 2021: with a 679M-parameter Codex model on HumanEval, temperature 0.2 was best for pass@1 and 0.8 for pass@100, sampling with top-p 0.95','Chen 等，2021：在 HumanEval 上用 679M 参数的 Codex 模型测试，pass@1 时 temperature 0.2 最好，pass@100 时 0.8 最好，采样都用 top-p 0.95')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="sp-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="sp-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="sp-app m2" id="sp-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="sp-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how its usual settings change the words that get drawn.','选一个场景，看看它常用的设置怎样改变抽到的词。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="sp-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Keep three good words, drop the wrong ones','留住三个好词，去掉错词')}</h3>`+
   `<p>${tx(`A writing assistant suggests the next word after “${esc(CHP.en)}”. The model’s chances are in the chart; three of its seven words are wrong (✗). Choose a temperature and a top-p so that the three wrong words together are picked less than 1% of the time, while at least 3 of the good words keep a chance of at least 5% each:`,
            `一个写作助手要给出“${esc(CHP.zh)}”之后的下一个词。模型给的概率在图里；七个词里有三个是错的（✗）。选一个 temperature 和一个 top-p，让三个错词加起来被抽中的概率低于 1%，同时至少 3 个好词各保留至少 5% 的概率：`)}</p>`+
   A.ui.seg({id:'sp-ic-chT',key:'chT',label:tx('Temperature','temperature'),options:CH.Ts.map(T=>[String(T),fTs(T)]),value:String(S.T),inline:true})+
   A.ui.seg({id:'sp-ic-chP',key:'chP',label:tx('Top-p','top-p'),options:CH.Ps.map(P=>[String(P),P>=1?tx('1.00 (off)','1.00（不截断）'):fixd(P,2)]),value:String(S.P),inline:true})+
   `<div class="goal mono sp-goal" id="sp-ch-g1"><span>${tx('Chance of the wrong words','错词被抽中的概率')}</span><span><b id="sp-ch-w"></b>${tx(' (goal: below 1%)','（目标：低于 1%）')}</span></div>`+
   `<div class="goal mono sp-goal" id="sp-ch-g2"><span>${tx('Good words at 5% or more','概率 5% 以上的好词')}</span><span><b id="sp-ch-n"></b>${tx(' (goal: at least 3)','（目标：至少 3 个）')}</span></div>`+
   submit+`<div id="sp-ch-res" class="sp-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
/* why Next is greyed out, under the buttons and on the phone's part bar (R1.1, R2.4) */
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},temp:{en:'Drag the temperature to 0 to continue.',zh:'把 temperature 拖到 0 才能继续。'},
    cut:{en:'Pick a rule to continue.',zh:'选一个规则才能继续。'},tree:{en:'Move the first random number onto another word to continue.',zh:'把第一个随机数移到另一个词上才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return nbh(railHTML())+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#sp-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="sp-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="sp-why" hidden></p>');}
  A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#sp-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step]||(s.id==='temp'&&S.T!==1)||(s.id==='cut'&&lv!=='none')||(s.id==='tree'&&u1Moved))&&(!s.gate||s.gate());
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvTried.has(v)||v===lv?lvRes(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=nbh(html); box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    /* once shown, the line keeps its height when Next is enabled (visibility, not display), so nothing below it moves mid-drag (re-review N2) */
    $$('.sp-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#sp-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=nbh(html); ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.T,S.P), w=$('#sp-ch-w',rail), n=$('#sp-ch-n',rail); if(!w) return;
    w.textContent=fP(e.wrong); n.textContent=String(e.n5);
    const g1=$('#sp-ch-g1',rail), g2=$('#sp-ch-g2',rail), o1=e.wrong<CH.wrongMax, o2=e.n5>=CH.minWords;
    g1.classList.toggle('met',o1); g1.classList.toggle('sp-over',!o1); g2.classList.toggle('met',o2); g2.classList.toggle('sp-over',!o2);
    $$('.seg-ctl[data-key="chT"]',rail).forEach(g=>A.ui.setPressed(g,String(S.T))); $$('.seg-ctl[data-key="chP"]',rail).forEach(g=>A.ui.setPressed(g,String(S.P)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#sp-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=nbh(html); res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':S.view);
const isB=()=>viewNow()==='b';
/* the column of logits divided by the temperature (R6.25): on the lunch prompt from part 3 on, and in Scenarios */
const showScaled=()=>viewNow()==='a'&&!(mode==='guided'&&step<SIX('temp'));
const pc=x=>(x*100).toFixed(3)+'%';
const aan=s=>/^(?:8|11|18|8\d)(?:\D|$)/.test(s)?'an':'a';
const sumTo=(a,i)=>a.slice(0,i).reduce((x,y)=>x+y,0);
/* the list on the stage: the lunch prompt's first word, or the challenge's words */
function listNow(){
  if(mode==='challenge'){const d=dist(CHZ,S.T,0,S.P); return {words:CHW.map(x=>L(x.w)),bad:CHW.map(x=>!!x.bad),z:CHZ,d,prompt:L(CHP)};}
  const r=R(); return {words:TREE.map(x=>L(x.w)),bad:TREE.map(()=>false),z:Z1,d:r.d1,prompt:L(PROMPT)};
}
/* z ÷ T as the column shows it: one decimal when every value is exact at one decimal, else two; a dash at T = 0 */
function scaledFmt(z,T){if(T===0) return z.map(()=>'—'); const v=z.map(x=>x/T), d=v.every(x=>Math.abs(x*10-Math.round(x*10))<1e-6)?1:2; return v.map(x=>fixd(x,d));}
const cutRule=d=>{const k=d.kk<d.n, p=d.keepK.some((x,i)=>x&&!d.keep[i]); return k&&p?tx('top-k and top-p','top-k 和 top-p'):(k?'top-k':'top-p');};
function rowsHTML(list,opt){
  opt=opt||{};
  const d=list.d, sc=!!opt.sc, zs=scaledFmt(list.z,d.T), T=fTs(d.T);
  const head=`<div class="sp-hd${sc?' sc':''}"><span>${tx('word','词')}</span><span>logit</span>${sc?`<span class="sp-zh" tabindex="0" data-tipx="${esc(TIP.scaled())}">${A.ui.math('\\div T')}</span>`:''}<span>${tx('chance of being picked','被抽中的概率')}</span><span></span></div>`;
  return head+list.words.map((w,i)=>{const cut=!d.keep[i]&&!d.greedy, pre=d.raw[i], fin=d.fin[i], z=fixd(list.z[i],1), zz=d.T===1||d.T===0?z:`${z} ÷ ${T} = ${zs[i]}`;
    let tip;
    if(d.greedy) tip=d.keep[i]?tx(`“${w}” has the highest logit, ${z}. At temperature 0 it gets all of the chance: greedy decoding.`,`“${w}”的 logit 最高，是 ${z}。temperature 为 0 时它拿到全部概率：这就是贪心解码。`)
        :tx(`“${w}” has logit ${z}, below the top word’s. At temperature 0 it gets no chance at all.`,`“${w}”的 logit 是 ${z}，比最高的那个低。temperature 为 0 时它一点概率也没有。`);
    else if(cut) tip=tx(`At temperature ${T}, softmax gives “${w}” (logit ${zz}) ${fP(pre)}, but ${!d.keepK[i]?'top-k':'top-p'} cuts it, so it cannot be picked.`,`temperature 为 ${T} 时，softmax 算出“${w}”（logit ${zz}）的概率是 ${fP(pre)}，但 ${!d.keepK[i]?'top-k':'top-p'} 把它去掉了，所以它不会被抽中。`);
    else if(d.nk<d.n){const [a,b]=fP2(pre,fin); tip=tx(`At temperature ${T}, softmax gives “${w}” (logit ${zz}) ${a} before the cut; ${cutRule(d)} cuts other words and scales the kept ones up to 100%, so it has ${b}.`,`temperature 为 ${T} 时，softmax 算出“${w}”（logit ${zz}）截断前的概率是 ${a}；${cutRule(d)} 去掉了别的词，把留下的词按比例放大回 100%，所以它现在是 ${b}。`);}
    else {const v=fP(fin); tip=tx(`At temperature ${T}, softmax gives “${w}” (logit ${zz}) ${aan(v)} ${v} chance of being picked.`,`temperature 为 ${T} 时，softmax 算出“${w}”（logit ${zz}）被抽中的概率是 ${v}。`);}
    return `<div class="sp-row${sc?' sc':''}${cut?' cut':''}${list.bad[i]?' bad':''}" data-i="${i}" tabindex="0" data-tipx="${esc(tip)}"><span class="sp-w">${list.bad[i]?'✗ ':''}${esc(w)}</span><span class="sp-z mono">${z}</span>${sc?`<span class="sp-zs mono">${zs[i]}</span>`:''}`+
      `<span class="sp-trk"><i class="sp-pre" style="width:${(pre*100).toFixed(3)}%"></i><i class="sp-fin" style="width:${(fin*100).toFixed(3)}%"></i>${cut?`<i class="sp-hatch" style="width:${(pre*100).toFixed(3)}%"></i>`:''}</span>`+
      `<span class="sp-v mono">${cut?tx('cut','去掉'):fP(fin)}</span></div>`;}).join('');
}
/* the chances laid end to end: one segment per word (a label only where it fits, see fitLabels) */
const segsHTML=(d,list)=>{let x=0; return d.raw.map((p,i)=>{const s=`<i class="sp-seg${d.keep[i]?'':' cut'}${list.bad[i]?' bad':''}" style="left:${pc(x)};width:${pc(p)}">${p>=0.02?`<span>${esc(list.words[i])}</span>`:''}</i>`; x+=p; return s;}).join('');};
const mkP=x=>x>=0.995&&x<1?'>99%':fR(x);
/* the running total at the end of each word top-p keeps; ✓ where it reaches p (R6.25). data-pr: which labels win a crowded row */
function pMarks(d){let h=''; for(let i=0;i<d.nk;i++){const last=i===d.nk-1; h+=`<span class="sp-mk${last?' on':''}" data-pr="${last?3:(i===d.nk-2?2.5:1+i/100)}" style="left:${pc(d.cum[i])}">${mkP(Math.min(1,d.cum[i]/d.sk))}${last?' ✓':''}</span>`;} return h;}
/* the challenge: the running total at the end of each good word, and the three wrong words at the end */
function chMarks(list){const d=list.d, g=list.bad.map((b,i)=>b?-1:i).filter(i=>i>=0), lastG=g[g.length-1];
  return g.map(i=>`<span class="sp-mk${i===lastG?' on':''}" data-pr="${i===lastG?3:1+i/100}" style="left:${pc(d.cum[i])}">${mkP(d.cum[i])}</span>`).join('')+`<span class="sp-mk bad" data-pr="2">✗✗✗</span>`;}
function drawA(){
  const box=$('#sp-a',app), duo=$('#sp-duo',app), b=$('#sp-b',app), v=viewNow();
  box.hidden=!(v==='a'||v==='ch'); duo.hidden=v!=='duo'; b.hidden=v!=='b';
  if(v==='duo') drawDuo(); else if(v==='b') drawTree(); else drawBars(v==='ch');
  fitLabels();
}
function drawBars(ch){
  const list=listNow(), d=list.d;
  $('#sp-a-cap',app).textContent=tx(`The next word after “${list.prompt}”`,`“${list.prompt}”之后的下一个词`);
  const rows=$('#sp-rows',app), rh=rowsHTML(list,{sc:showScaled()}); if(rows._h!==rh){rows.innerHTML=rh; rows._h=rh;}
  /* the words laid end to end after the temperature, with top-k's and top-p's lines (R1.9b); in the challenge, the three top-p choices */
  $('#sp-cum-cap',app).textContent=ch?tx('The chances laid end to end; under the bar, the running total after each good word','概率首尾相接；条下面是每个好词之后的累计')
    :(d.pLine!=null||d.kLine!=null?tx('The chances laid end to end: a word stays if its stretch starts before the line','概率首尾相接：某个词的那一段从线的左边开始，这个词就留下'):tx('The chances laid end to end (no cut)','概率首尾相接（不截断）'));
  const lines=(d.kLine!=null?`<b class="sp-kline${d.kLine<0.35?' r':''}" style="left:${pc(d.kLine)}"><em>${nbh(`top-k ${d.kk}`)}</em></b>`:'')+(d.pLine!=null?`<b class="sp-pline${d.pLine>0.65?' l':''}" style="left:${pc(d.pLine)}"><em>${nbh(`top-p ${fixd(d.P,2)}`)}</em></b>`:'');
  const ot=ch?CH.Ps.filter(P=>P<1&&P!==S.P).map(P=>`<b class="sp-otick" style="left:${pc(P)}"><em>${fixd(P,2)}</em></b>`).join(''):'';
  $('#sp-cum',app).classList.toggle('ch',ch);
  const cb=$('#sp-cumbar',app); cb.innerHTML=segsHTML(d,list)+ot+lines;
  cb.dataset.tipx=d.pLine!=null?tx(`After the temperature, the words’ chances laid end to end from 0 to 100%. Top-p ${fixd(d.P,2)} keeps every word whose stretch starts before the line${d.kLine!=null?`, drawn at ${fixd(d.P,2)} of what top-k keeps`:''}: ${d.nk} of ${d.n}.`,`经过 temperature 后，各个词的概率从 0 到 100% 首尾相接。top-p ${fixd(d.P,2)} 留下每个从线左边开始的词${d.kLine!=null?`（这条线画在 top-k 留下的概率的 ${fixd(d.P,2)} 处）`:''}：${d.n} 个里留下 ${d.nk} 个。`)
    :(d.kLine!=null?tx(`After the temperature, the words’ chances laid end to end. Top-k ${d.kk} keeps the words left of the dotted line.`,`经过 temperature 后，各个词的概率首尾相接。top-k ${d.kk} 留下点线左边的词。`):tx('After the temperature, the words’ chances laid end to end from 0 to 100%. Nothing is cut.','经过 temperature 后，各个词的概率从 0 到 100% 首尾相接。没有截断。'))+
    (ch?tx(' The short dotted ticks above the bar are the three top-p choices.',' 条上方的短点线是 top-p 的三个选项。'):'');
  cb.setAttribute('role','img'); cb.setAttribute('aria-label',cb.dataset.tipx);
  const mk=$('#sp-cum-marks',app), mh=ch?chMarks(list):(d.pLine!=null?pMarks(d):'');
  mk.innerHTML=mh;
  if(mh){const g4=d.cum[list.bad.lastIndexOf(false)];
    mk.tabIndex=0; mk.dataset.tipx=ch?tx(`The running total of the chances from the top, at the end of each good word: the four good words hold ${mkP(g4)} together, and the three wrong words (✗) share the rest.`,`从最高的词开始累加概率，标在每个好词的末尾：四个好词加起来占 ${mkP(g4)}，其余的归三个错词（✗）。`)
      :tx(`The running total of the chances from the top, at the end of each word top-p keeps. Top-p ${fixd(d.P,2)} stops at the first word that brings it to at least ${fixd(d.P,2)} (✓).`,`从最高的词开始累加概率，标在 top-p 留下的每个词的末尾。top-p ${fixd(d.P,2)} 停在让累计至少达到 ${fixd(d.P,2)} 的第一个词（✓）。`);}
  else {mk.removeAttribute('tabindex'); delete mk.dataset.tipx;}
  /* the draw: the final chances end to end, and where each run's random number landed (ticks above the bar only) */
  const draw=$('#sp-draw',app), strip=$('#sp-strip',app); draw.hidden=ch; strip.hidden=ch;
  if(ch) return;
  const r=R();
  $('#sp-draw-cap',app).textContent=tx(`What the draw uses, after the cut: each tick is one run’s random number`,`抽签用的概率（截断之后）：每条细线是一次运行的随机数`);
  let x=0; const dsegs=d.fin.map((p,i)=>{if(p<=0) return ''; const s=`<i class="sp-seg" style="left:${pc(x)};width:${pc(p)}">${p>=0.02?`<span>${esc(list.words[i])}</span>`:''}</i>`; x+=p; return s;}).join('');
  const ticks=runsOf(S.seed).map((u,k)=>`<b class="sp-tick" style="left:${(u[0]*100).toFixed(3)}%" data-k="${k}"></b>`).join('');
  const db=$('#sp-drawbar',app); db.innerHTML=dsegs+ticks;
  db.dataset.tipx=tx(`The chances that remain after the cut, laid end to end. Each of the 20 runs drew a random number between 0 and 1 (a thin tick); the word under the tick is that run’s first word.`,`截断后留下的概率首尾相接。20 次运行各抽一个 0 到 1 之间的随机数（一条细线）；细线落在哪个词上，这次运行的第一个词就是它。`);
  db.setAttribute('role','img'); db.setAttribute('aria-label',db.dataset.tipx);
  $('#sp-strip-cap',app).textContent=tx(`20 runs, seed ${S.seed}: the first word each one wrote (${r.distinct} different)`,`20 次运行（种子 ${S.seed}）：每次写出的第一个词（${r.distinct} 个不同）`);
  const chips=r.first.map((i,k)=>`<span class="sp-chip" data-k="${k}">${esc(list.words[i])}</span>`).join('');
  const ce=$('#sp-chips',app); if(ce._h!==chips){ce.innerHTML=chips; ce._h=chips;}
}
/* part 4: under each list, its chances laid end to end with the rule drawn on them: top-p's running total and line, or top-k's bracket (R6.25) */
function stripHTML(list,capOn){
  const d=list.d, isPc=d.pLine!=null, isK=!isPc&&d.kLine!=null;
  const line=isPc?`<b class="sp-pline${d.pLine>0.65?' l':''}" style="left:${pc(d.pLine)}"><em>${nbh(`top-p ${fixd(d.P,2)}`)}</em></b>`:'';
  const marks=isPc?pMarks(d):(isK?`<span class="sp-kbr" style="width:${pc(d.sk)}"><em>${esc(tx(`${d.kk} words: ${fR(d.sk)}`,`${d.kk} 个词：${fR(d.sk)}`))}</em></span>`:'');
  const cap=isPc?tx(`Running total from the top: top-p stops at the first word that brings it to at least ${fR(d.P)}`,`从最高的开始累加：top-p 停在让累计至少达到 ${fR(d.P)} 的第一个词`)
    :(isK?tx(`Top-k keeps the first ${d.kk} words, whatever they add up to`,`top-k 留下前 ${d.kk} 个词，不管它们加起来是多少`):tx('The chances laid end to end: every word stays','概率首尾相接：每个词都留下'));
  const tip=isPc?tx(`The chances laid end to end, likeliest first. Top-p ${fixd(d.P,2)} keeps every word whose stretch starts before the dashed line: ${d.nk} of ${d.n}. The numbers under the bar are the running totals.`,`概率从高到低首尾相接。top-p ${fixd(d.P,2)} 留下每个从虚线左边开始的词：${d.n} 个里留下 ${d.nk} 个。条下面的数是累计。`)
    :(isK?tx(`The chances laid end to end, likeliest first. Top-k ${d.kk} keeps the first ${d.kk} words, which hold ${fR(d.sk)} together.`,`概率从高到低首尾相接。top-k ${d.kk} 留下前 ${d.kk} 个词，它们加起来占 ${fR(d.sk)}。`):tx('The chances laid end to end, likeliest first. Nothing is cut.','概率从高到低首尾相接。没有截断。'));
  return `<div class="sp-dstrip"><div class="sp-bar sp-mini" tabindex="0" role="img" aria-label="${esc(tip)}" data-tipx="${esc(tip)}">${segsHTML(d,list)}${line}</div><div class="sp-marks${isK?' kb':''}">${marks}</div>${capOn?`<div class="sp-sub">${esc(cap)}</div>`:''}</div>`;
}
function drawDuo(){
  const mkL=(words,z,d)=>({words,bad:words.map(()=>false),z,d}), pk=TREE[PEAK];
  const f=mkL(TREE.map(x=>L(x.w)),Z1,flatD(lv)), p=mkL(pk.c.map(x=>L(x.w)),pk.c.map(x=>x.z),peakD(lv));
  const cap=(t,d)=>`<div class="sp-cap">${esc(t)}</div><div class="sp-keep mono">${esc(tx(`${d.nk} of ${d.n} kept`,`${d.n} 个里留下 ${d.nk} 个`))}</div>`;
  const fh=cap(tx(`Flat: the next word after “${PROMPT.en}”`,`平的：“${PROMPT.zh}”之后的下一个词`),f.d)+`<div class="sp-rows">${rowsHTML(f)}</div>`+stripHTML(f,true);
  const ph=cap(tx(`Peaked: the next word after “${PROMPT.en} a burger”`,`尖的：“${PROMPT.zh}汉堡”之后的下一个词`),p.d)+`<div class="sp-rows">${rowsHTML(p)}</div>`+stripHTML(p,false);
  const a=$('#sp-duo-f',app), b=$('#sp-duo-k',app); if(a._h!==fh){a.innerHTML=fh; a._h=fh;} if(b._h!==ph){b.innerHTML=ph; b._h=ph;}
}
/* the tree, three words deep: one column per word, a block's height is the chance of the sentence so far; your run (part 5) or the
   likeliest sentence is outlined; lines mark where the random numbers landed. Chances in the tooltips are products of the factors as
   shown (R6.6); a block label that a run's tag would cover is left out (R7.8) */
function drawTree(){
  const r=R(), host=$('#sp-tree',app), H=innerWidth<=760?300:340, lab=h=>h*H>=14, run=runMode();
  const me=run?runPath():null, path=me||r.best, us=me?[runU1(),U23[0],U23[1]]:null, sp=A.lang()==='zh'?'':' ';
  $('#sp-b-cap',app).textContent=tx(`Every way “${PROMPT.en} …” can go, three words deep: a block’s height is its chance`,`“${PROMPT.zh}……”三个词以内的每一种写法：每一块的高度就是它的概率`);
  const hd=[tx('1st word','第 1 个词'),tx('2nd word','第 2 个词'),tx('3rd word','第 3 个词')];
  const runLeaf={}; r.runs.forEach(x=>{const k=x.i+'.'+x.j+'.'+x.k; runLeaf[k]=(runLeaf[k]||0)+1;});
  const B=[[],[],[]], tags=[null,null,null], mark=[0,0,0], hit=[null,null,null];
  let y1=0;
  r.tree.forEach(a=>{if(a.p<=0) return; const w1=L(TREE[a.i].w), on1=path.i===a.i, s1=pcs(a.p);
    B[0].push({y:y1,h:a.p,t:w1,on:on1,tip:tx(`A run starts with “${w1}” ${s1.s} of the time.`,`一次运行以“${w1}”开头的概率是 ${s1.s}。`)});
    if(on1){if(!lab(a.p)) tags[0]=[y1,w1]; if(us){mark[0]=us[0]; mark[1]=y1+a.p*us[1]; hit[0]={w:w1,from:sumTo(r.d1.fin,a.i),p:a.p,after:''};}}
    let y2=y1;
    a.kids.forEach(b=>{if(b.p<=0) return; const w2=L(TREE[a.i].c[b.j].w), on2=on1&&path.j===b.j, f2=pcs(a.d.fin[b.j]), s2=pcs(s1.v*f2.v/1e4);
      B[1].push({y:y2,h:b.p,t:w2,on:on2,tip:tx(`A run starts “${w1} ${w2}” ${s2.s} of the time: “${w1}” is picked ${s1.s} of the time, and after it “${w2}” ${f2.s} of the time.`,`一次运行以“${w1}${w2}”开头的概率是 ${s2.s}：先抽中“${w1}”的概率是 ${s1.s}，之后再抽中“${w2}”的概率是 ${f2.s}。`)});
      if(on2){if(!lab(b.p)) tags[1]=[y2,w2]; if(us){mark[2]=y2+b.p*us[2]; hit[1]={w:w2,from:sumTo(a.d.fin,b.j),p:a.d.fin[b.j],after:w1};}}
      let y3=y2;
      b.leaves.forEach(c=>{if(c.p<=0) return; const w3=L(TREE[a.i].c[b.j].c[c.k].w), on3=on2&&path.k===c.k, n=runLeaf[a.i+'.'+b.j+'.'+c.k]||0, f3=pcs(b.d.fin[c.k]), s3=pcs(s1.v*f2.v*f3.v/1e6), s=w1+sp+w2+sp+w3;
        /* the three words' own chances, the same factors as Show the math, so the two give the same 4 digits (re-review N3) */
        B[2].push({y:y3,h:c.p,t:w3,on:on3,n:run?0:n,tip:tx(`A run writes “${s}” ${s3.s} of the time: “${w1}” is picked ${s1.s} of the time, then “${w2}” ${f2.s}, then “${w3}” ${f3.s}.${n?` ${n} of the 20 runs (seed ${S.seed}) wrote it.`:''}`,`一次运行写出“${s}”的概率是 ${s3.s}：先抽中“${w1}”的概率是 ${s1.s}，再抽中“${w2}”的概率是 ${f2.s}，再抽中“${w3}”的概率是 ${f3.s}。${n?`20 次运行（种子 ${S.seed}）里有 ${n} 次写成这样。`:''}`)});
        if(on3){if(!lab(c.p)) tags[2]=[y3,w3]; if(us) hit[2]={w:w3,from:sumTo(b.d.fin,c.k),p:b.d.fin[c.k],after:w1+sp+w2};}
        y3+=c.p;});
      y2+=b.p;});
    y1+=a.p;});
  /* a tag sits just above its thin block; the label of any block whose text would fall under it is left out */
  const band=l=>{if(!tags[l]) return null; const bot=tags[l][0]*H-2; return [bot-17,bot];};
  /* a random number's line, and its label just above it, never run through a block label (re-review N1; struck-through text means
     "cut" in this lab): the label moves just below the line; or above the line's label; or between the block's top and the line, where
     fitLabels leaves it out if it meets the line's label; or it is left out (the block's tooltip and the sentence below still name it) */
  const LH=14, ly=us?mark.map(m=>m*H):null;
  const cols=B.map((col,l)=>{const tb=band(l); return col.map(e=>{let show=lab(e.h), pos='', cls='';
    const top=e.y*H, bot=(e.y+e.h)*H, c=(top+bot)/2;
    if(show&&tb&&c+7>tb[0]&&c-7<tb[1]) show=false;
    if(show&&ly){const z0=ly[l]-16, z1=ly[l]+4;
      if(c+7>z0&&c-7<z1){
        if(z1+LH<=bot-1) {pos=z1-top; cls='sp-mv';}
        else if(z0-LH>=top+1) {pos=z0-LH-top; cls='sp-mv';}
        else if(ly[l]-3-LH>=top+0.5) {pos=ly[l]-3-LH-top; cls='sp-mv sp-mvu';}
        else show=false;}}
    return `<div class="sp-blk${e.on?' on':''}" style="top:${(e.y*100).toFixed(4)}%;height:${(e.h*100).toFixed(4)}%" tabindex="0" data-tipx="${esc(e.tip)}">${show?`<span${cls?` class="${cls}" style="top:${pos.toFixed(1)}px"`:''}>${esc(e.t)}</span>`:''}${e.n&&show?`<em class="sp-n">×${e.n}</em>`:''}</div>`;}).join('');});
  const ord=[tx('1st','第 1 个'),tx('2nd','第 2 个'),tx('3rd','第 3 个')], ordL=[tx('first','第 1 个'),tx('second','第 2 个'),tx('third','第 3 个')];
  const f3=x=>x<=0?'0':(x>=0.9999995?'1':fixd(x,3));
  const mk=l=>{if(!us||!hit[l]) return ''; const h=hit[l], a=f3(h.from), b=f3(h.from+h.p);
    const tip=l===0?tx(`The first random number of your run (${fixd(us[0],2)}): it lands in “${h.w}”, which covers ${a} to ${b} of the first word’s chances.`,`你这次运行的第 1 个随机数（${fixd(us[0],2)}）落在“${h.w}”那一段：它占第一个词概率里 ${a} 到 ${b} 的范围。`)
      :tx(`The ${ordL[l]} random number of your run (${fixd(us[l],2)}): it lands in “${h.w}”, which covers ${a} to ${b} of the chances after “${h.after}”.`,`你这次运行的${ordL[l]}随机数（${fixd(us[l],2)}）落在“${h.w}”那一段：它占“${h.after}”之后的概率里 ${a} 到 ${b} 的范围。`);
    return `<b class="sp-u" style="top:${(mark[l]*100).toFixed(3)}%"><em tabindex="0" data-tipx="${esc(tip)}"><i class="sp-ord">${ord[l]} </i>${fixd(us[l],2)}</em></b>`;};
  const tg=l=>tags[l]?`<b class="sp-tag" style="top:${(tags[l][0]*100).toFixed(3)}%">${esc(tags[l][1])}</b>`:'';
  /* a column with a tag: on a phone the random number's label drops its “1st” (the column head says which word it is) so the two fit side by side */
  host.innerHTML=cols.map((c,l)=>`<div class="sp-col" data-l="${l}"><div class="sp-colh">${hd[l]}</div><div class="sp-colb${tags[l]?' tagged':''}" style="height:${H}px">${c}${mk(l)}${tg(l)}</div></div>`).join('');
  const sent=$('#sp-sent',app), ps=path.p;
  sent.innerHTML=(me?tx('Your run: ','你这次运行：'):tx('The likeliest sentence: ','最可能的一句：'))+`<span class="sp-pr">${esc(L(PROMPT))}</span>${A.lang()==='zh'?'':' '}<b>${esc(sentOf(r,path))}</b>`+tx(` (${fP(ps)} of all runs)`,`（所有运行里占 ${fP(ps)}）`);
}
/* labels that do not fit whole are left out, never cut (R7.8): segment labels against the segment's final width (the bars animate),
   tree labels against their block; then the running totals, two rows at most */
function fitLabels(){
  if(!app) return;
  const segs=[]; $$('.sp-bar',app).forEach(bar=>{const W=bar.clientWidth; if(!W) return; $$('.sp-seg>span',bar).forEach(s=>segs.push([s,parseFloat(s.parentElement.style.width)/100*W-1.5]));});
  const blks=$$('#sp-tree .sp-blk>span',app).filter(s=>s.parentElement.getClientRects().length);
  segs.forEach(([s])=>{s.hidden=false;}); blks.forEach(s=>{s.hidden=false;});
  const hs=segs.map(([s,w])=>s.scrollWidth>w+0.5), hb=blks.map(s=>s.scrollWidth>s.clientWidth+0.5);
  segs.forEach(([s],i)=>{s.hidden=hs[i];}); blks.forEach((s,i)=>{s.hidden=hb[i];});
  /* a segment label that top-p's dashed line (or top-k's dotted one) would cross is left out (re-review N4); the rows name the word */
  $$('.sp-bar',app).forEach(bar=>{const W=bar.clientWidth; if(!W) return; const xs=$$('.sp-pline,.sp-kline',bar).map(b=>parseFloat(b.style.left)/100*W); if(!xs.length) return;
    $$('.sp-seg>span',bar).forEach(s=>{if(s.hidden) return; const g=s.parentElement, l=parseFloat(g.style.left)/100*W, w=parseFloat(g.style.width)/100*W, sw=s.scrollWidth, a=l+(w-sw)/2, b=a+sw;
      if(xs.some(x=>x>a-2&&x<b+2)) s.hidden=true;});});
  /* a label placed between a block's top and a random number's line shares the line's label row: left out where the two meet */
  $$('#sp-tree .sp-blk>span.sp-mvu',app).forEach(s=>{if(s.hidden) return; const u=s.closest('.sp-colb').querySelector('.sp-u em'); if(!u) return;
    const a=s.getBoundingClientRect(), t=document.createRange(); t.selectNodeContents(s); const tr=t.getBoundingClientRect(), b=u.getBoundingClientRect();
    if(tr.right+3>b.left&&tr.left<b.right+3&&a.bottom>b.top&&a.top<b.bottom) s.hidden=true;});
  $$('.sp-marks',app).forEach(m=>{const items=$$('.sp-mk',m); m.classList.remove('two'); if(!items.length||!m.getClientRects().length) return;
    items.forEach(e=>{e.hidden=false; e.classList.remove('lo');});
    const R=items.map(e=>{const r=e.getBoundingClientRect(); return {e,l:r.left,r:r.right,p:+e.dataset.pr||0};}), rows=[[],[]]; let two=false;
    R.sort((a,b)=>b.p-a.p).forEach(x=>{for(let k=0;k<2;k++){if(rows[k].every(y=>x.r+3<=y.l||x.l>=y.r+3)){rows[k].push(x); if(k){x.e.classList.add('lo'); two=true;} return;}} x.e.hidden=true;});
    m.classList.toggle('two',two);});
  /* the challenge's top-p choices: a label under the chosen top-p's own label, or on top of another choice's, is left out */
  const pl=$('#sp-cumbar .sp-pline em',app), taken=pl&&pl.getClientRects().length?[pl.getBoundingClientRect()]:[];
  $$('#sp-cumbar .sp-otick em',app).forEach(e=>{e.hidden=false; const r=e.getBoundingClientRect(); if(!r.width) return; e.hidden=taken.some(q=>r.right+3>q.left&&r.left<q.right+3); if(!e.hidden) taken.push(r);});
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.T,S.P); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Both targets met','两个目标都达到'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the three wrong words together are picked less than 1% of the time and at least 3 good words keep 5% or more. Now: ${fP(e.wrong)} and ${e.n5} good words.`,`三个错词加起来被抽中的概率低于 1%，且至少 3 个好词各有 5% 以上时，徽章就是绿色。现在是 ${fP(e.wrong)} 和 ${e.n5} 个好词。`)};}
  if(v==='duo'){const f=flatD(lv), p=peakD(lv); return {cls:'neutral',txt:tx(`${f.nk} + ${p.nk} kept`,`留下 ${f.nk} + ${p.nk} 个词`),short:tx(`${f.nk} + ${p.nk} kept`,`留下 ${f.nk} + ${p.nk}`),
    tip:tx(`How many words the rule in the card keeps: ${f.nk} of the 6 first words, and ${p.nk} of the 4 words after “a burger”.`,`卡片里选的规则留下几个词：第一个词的 6 个候选里留 ${f.nk} 个，“汉堡”之后的 4 个词里留 ${p.nk} 个。`)};}
  const r=R();
  if(v==='b') return {cls:'neutral',txt:tx(`${r.possible} of ${N_LEAVES} sentences possible`,`${N_LEAVES} 句里有 ${r.possible} 句可能写出`),short:tx(`${r.possible} possible`,`可写出 ${r.possible} 句`),
    tip:tx(`${r.possible} of the ${N_LEAVES} three-word endings in the tree can still be written with these settings: every word on the way must survive the cut.`,`树里 ${N_LEAVES} 种三个词的结尾，在这组设置下还有 ${r.possible} 种写得出来：一路上每个词都要没被截掉。`)};
  const d=r.d1;
  return {cls:'neutral',txt:tx(`${d.nk} of 6 words can be picked`,`6 个词里有 ${d.nk} 个可能被抽中`),short:tx(`${d.nk} of 6 kept`,`留下 ${d.nk} / 6`),
    tip:d.greedy?tx(`Only “${word(TREE[0])}” can be picked: at temperature 0 the top word gets all of the chance.`,`只有“${word(TREE[0])}”可能被抽中：temperature 为 0 时，最高的词拿到全部概率。`)
      :(d.nk===d.n?tx('All 6 words can still be picked: top-k and top-p are off, so nothing is cut.','6 个词都还可能被抽中：top-k 和 top-p 都没开，什么都没去掉。')
        :tx(`${d.nk} of the 6 words can still be picked: ${cutRule(d)} cuts the other ${d.n-d.nk}.`,`6 个词里还有 ${d.nk} 个可能被抽中：${cutRule(d)} 去掉了另外 ${d.n-d.nk} 个。`))};
}
function drawHead(){
  const v=viewNow(), t=$('#sp-st-title',app), sub=$('#sp-st-sub',app), st=$('#sp-status',app), hT=S.T===0?'0':fixd(S.T,1);   // “(greedy)” is on the slider and in the result (O3)
  t.textContent=v==='ch'?tx('The challenge prompt: 7 words, 3 of them wrong','挑战题的 prompt：7 个词，其中 3 个是错的'):(v==='b'?tx('The sentence tree','句子树'):(v==='duo'?tx('One rule, two lists of next words','同一个规则，两个下一个词的分布'):tx('The lunch prompt: 6 possible next words','午饭 prompt：6 个可能的下一个词')));
  sub.textContent=nbh(v==='duo'?tx(`temperature 1.0, rule: ${L(lvOf(lv).t)}`,`temperature 1.0，规则：${L(lvOf(lv).t)}`):tx(`temperature ${hT}, top-p ${S.P>=1?'off':fixd(S.P,2)}${v==='ch'?'':`, top-k ${fK(S.K)}`}`,`temperature ${hT}，top-p ${S.P>=1?'不截断':fixd(S.P,2)}${v==='ch'?'':`，top-k ${fK(S.K)}`}`));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=nbh(s.txt); st.dataset.tipx=s.tip;
  const vw=$('#sp-view',app); vw.hidden=mode!=='sandbox'; $$('button',vw).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===S.view)));
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(nbh(txt))}</span></span>`, v=viewNow();
  if(v==='b'){const run=runMode();
    return it('sp-sw-blk',tx('block: a sentence so far, height = chance','方块：写到这里的半句话，高度 = 概率'),tx('Each block is one way the sentence can go up to that word; its height is the chance that a run writes it.','每一块是句子写到这个词为止的一种写法；它的高度就是一次运行写成这样的概率。'))+
    it('sp-sw-on',tx('thick outline: '+(run?'your run':'the likeliest sentence'),'粗边框：'+(run?'你这次运行':'最可能的一句')),tx('The outlined blocks are the words of the sentence written below the tree.','加粗边框的方块，就是树下面写出的那句话的各个词。'))+
    (run?it('sp-sw-u',tx('line: one of your run’s random numbers (1st, 2nd, 3rd)','横线：你这次运行的一个随机数（第 1、2、3 个）'),tx('In each column, the line marks where that step’s random number landed inside the block of the words before it; its label gives the number.','每一列里，横线标出这一步的随机数落在前面那些词的方块里的什么位置；旁边的标签写着这个数。'))
      :`<span data-tipx="${esc(tx('A “×4” next to a sentence means that 4 of the 20 runs wrote it. A sentence that no run wrote has no mark.','句子旁的“×4”表示 20 次运行里有 4 次写成这句。没有哪次运行写出的句子不标。'))}" tabindex="0"><em class="sp-n sp-sw-n">×N</em><span>${esc(tx(`: N of the 20 runs (seed ${S.seed}) wrote this sentence`,`：20 次运行（种子 ${S.seed}）里有 N 次写成这句`))}</span></span>`);}
  const bars=it('sp-sw-fin',tx('bar: chance of being picked','横条：被抽中的概率'),tx('The solid bar is each word’s chance of being picked with the current settings.','实心条是在当前设置下每个词被抽中的概率。'))+
    it('sp-sw-pre',tx('dashed outline: before the cut','虚线框：截断之前'),tx('The dashed outline is the word’s chance after the temperature, before top-k and top-p. A word that stays grows past it, because the chances that remain are scaled up to 100%.','虚线框是经过 temperature、还没做 top-k 和 top-p 时这个词的概率。留下的词会超出虚线框，因为剩下的概率要按比例放大回 100%。'))+
    it('sp-sw-cut',tx('hatched: cut','斜纹：被去掉'),tx('A hatched word was cut by top-k or top-p and cannot be picked.','斜纹表示这个词被 top-k 或 top-p 去掉了，不会被抽中。'));
  if(v==='duo') return bars;
  const dd=listNow().d;
  const lines=(dd.pLine!=null?it('sp-sw-pl',tx('dashed line: top-p','虚线：top-p'),tx('On the bar of chances laid end to end, the dashed line is where top-p stops: every word whose stretch starts before it stays.','在首尾相接的概率条上，虚线是 top-p 停下的地方：从它左边开始的每个词都留下。')):'')+
    (dd.kLine!=null?it('sp-sw-kl',tx('dotted line: top-k','点线：top-k'),tx('The dotted line marks the end of the k likeliest words that top-k keeps.','点线标出 top-k 留下的最可能的 k 个词在哪里结束。')):'');
  if(v==='ch') return bars+lines+it('sp-sw-ot',tx('short dotted ticks: the top-p choices','短点线：top-p 的选项'),tx('The short dotted ticks above the bar of chances laid end to end mark the top-p choices in the card: 0.80, 0.90 and 0.95.','首尾相接的概率条上方的短点线，标出卡片里 top-p 的选项：0.80、0.90 和 0.95。'))+
    it('sp-sw-bad',tx('✗ wrong word','✗ 错词'),tx('A word marked ✗ is a wrong answer to this prompt. The goal adds up the chances of all three ✗ words.','标 ✗ 的词是这个 prompt 的错误答案。目标把三个 ✗ 词的概率加在一起算。'));
  return bars+lines+it('sp-sw-tick',tx('thin tick: one run’s random number','细线：一次运行的随机数'),tx('Each thin tick is the random number one of the 20 runs drew; the word under it is that run’s first word.','每条细线是 20 次运行中某一次抽到的随机数；它下面的词就是这次运行的第一个词。'));
}
/* metric cards: ▲▼ only after the learner's own change, from the values as shown, in points for chances (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.T,S.P);
    return [{k:'wrong',label:tx('Chance of the wrong words','错词被抽中的概率'),v:fP(e.wrong),sub:tx('goal: below 1%','目标：低于 1%'),pts:true,tip:tx(`“${chW(4)}”, “${chW(5)}” and “${chW(6)}” together are picked ${fP(e.wrong)} of the time: their chances after the temperature and top-p, added up.`,`“${chW(4)}”“${chW(5)}”“${chW(6)}”三个错词加起来被抽中的概率是 ${fP(e.wrong)}：它们经过 temperature 和 top-p 之后的概率之和。`)},
      {k:'n5',label:tx('Good words at 5% or more','概率 5% 以上的好词'),v:String(e.n5),sub:tx('goal: at least 3','目标：至少 3 个'),tip:tx(`${e.n5} of the 4 good words keep a chance of at least 5% each with these settings.`,`按当前设置，4 个好词里有 ${e.n5} 个各保留至少 5% 的概率。`)},
      {k:'kept',label:tx('Words that can be picked','可能被抽中的词'),v:tx(`${e.d.nk} of 7`,`${e.d.nk} / 7`),sub:tx('after top-p','top-p 之后'),tip:tx(`${e.d.nk} of the 7 words can still be picked after top-p.`,`7 个词里，经过 top-p 后还有 ${e.d.nk} 个可能被抽中。`)}];}
  if(v==='duo'){const f=flatD(lv), p=peakD(lv);
    return [{k:'kf',label:tx('Kept, flat list','平的分布留下'),v:tx(`${f.nk} of 6`,`${f.nk} / 6`),sub:tx('the first word','第一个词'),tip:tx(`The rule keeps ${f.nk} of the 6 first words.`,`规则在第一个词的 6 个候选里留下 ${f.nk} 个。`)},
      {k:'kp',label:tx('Kept, peaked list','尖的分布留下'),v:tx(`${p.nk} of 4`,`${p.nk} / 4`),sub:tx('after “a burger”','“汉堡”之后'),tip:tx(`The rule keeps ${p.nk} of the 4 words after “a burger”.`,`规则在“汉堡”之后的 4 个词里留下 ${p.nk} 个。`)},
      {k:'cf',label:tx('Chance cut, flat list','平的分布去掉的概率'),v:fP(f.cutAll),sub:tx('before the cut','按截断前算'),pts:true,tip:tx(`The chance the cut words had before the cut, on the flat list: ${fP(f.cutAll)}.`,`平的分布里，被去掉的词在截断前加起来的概率：${fP(f.cutAll)}。`)},
      {k:'cp',label:tx('Chance cut, peaked list','尖的分布去掉的概率'),v:fP(p.cutAll),sub:tx('before the cut','按截断前算'),pts:true,tip:tx(`The chance the cut words had before the cut, after “a burger”: ${fP(p.cutAll)}.`,`“汉堡”之后，被去掉的词在截断前加起来的概率：${fP(p.cutAll)}。`)}];}
  const r=R();
  if(v==='b'){const me=runMode()?runPath():null;
    const m=[{k:'best',label:tx('Likeliest sentence','最可能的一句'),v:fP(r.best.p),sub:sentOf(r,r.best),pts:true,tip:tx(`The likeliest three-word ending, “${sentOf(r,r.best)}”: the product of its three words’ chances, ${fP(r.best.p)}.`,`最可能的三个词的结尾“${sentOf(r,r.best)}”：三个词的概率相乘，${fP(r.best.p)}。`)},
      {k:'poss',label:tx('Sentences possible','可能写出的句子'),v:tx(`${r.possible} of ${N_LEAVES}`,`${r.possible} / ${N_LEAVES}`),sub:tx('three words deep','三个词以内'),tip:tx(`${r.possible} of the ${N_LEAVES} endings in the tree can still be written.`,`树里 ${N_LEAVES} 种结尾里，还能写出来的有 ${r.possible} 种。`)}];
    if(me) m.push({k:'mine',label:tx('Your sentence','你这一句'),v:fP(me.p),sub:tx('chance of all three words','三个词的概率相乘'),pts:true,tip:tx(`The chance that a run writes your sentence, “${sentOf(r,me)}”: ${fP(me.p)}.`,`一次运行写出你这一句“${sentOf(r,me)}”的概率：${fP(me.p)}。`)});
    else m.push({k:'ds',label:tx('Different sentences, 20 runs','20 次运行写出的不同句子'),v:String(r.distinctS),sub:tx(`seed ${S.seed}`,`种子 ${S.seed}`),tip:tx(`The 20 runs (seed ${S.seed}) wrote ${r.distinctS} different three-word ${r.distinctS===1?'ending':'endings'}.`,`20 次运行（种子 ${S.seed}）写出了 ${r.distinctS} 种不同的三个词的结尾。`)});
    return m;}
  return [{k:'top',label:tx('Top word’s chance','最可能那个词的概率'),v:fP(r.d1.fin[0]),sub:word(TREE[0]),pts:true,tip:tx(`The chance that “${word(TREE[0])}”, the word with the highest logit, is picked: ${fP(r.d1.fin[0])}.`,`logit 最高的“${word(TREE[0])}”被抽中的概率：${fP(r.d1.fin[0])}。`)},
    {k:'kept',label:tx('Words that can be picked','可能被抽中的词'),v:tx(`${r.d1.nk} of 6`,`${r.d1.nk} / 6`),sub:tx('after top-k and top-p','top-k 和 top-p 之后'),tip:tx(`${r.d1.nk} of the 6 words can still be picked after top-k and top-p.`,`第一个词的 6 个候选里，经过 top-k 和 top-p 后还有 ${r.d1.nk} 个可能被抽中。`)},
    {k:'cut',label:tx('Chance cut','被去掉的概率'),v:fP(r.d1.cutAll),sub:tx('before the cut','按截断前算'),pts:true,tip:r.d1.cutAll>0?tx(`The chance the cut words had after the temperature, before the cut: ${fP(r.d1.cutAll)}. The words that stay share it out.`,`被去掉的词经过 temperature、截断之前加起来的概率：${fP(r.d1.cutAll)}。留下的词把它分掉。`):tx('No word is cut, so no chance is moved to the others.','没有词被去掉，所以没有概率要重新分配。')},
    {k:'runs',label:tx('Different first words','不同的第一个词'),v:String(r.distinct),sub:tx(`in 20 runs, seed ${S.seed}`,`20 次运行，种子 ${S.seed}`),tip:tx(`The 20 runs with seed ${S.seed} wrote ${r.distinct} different first ${r.distinct===1?'word':'words'}.`,`用种子 ${S.seed} 的 20 次运行写出了 ${r.distinct} 个不同的第一个词。`)}];
}
function drawMetrics(){
  const host=$('#sp-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=nbh(m.label); el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=nbh(m.sub); el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every number redoes from the ones shown (4 digits, R6.6), every term has a tooltip (R1.4); a term never breaks (R7.8).
   H2 (the owner's decision of 2026-10-09): a few short blocks, like a page of a textbook. Each block has a lead line in words, the
   quantity's name and its formula typeset (A.ui.math), the formula again with the current numbers, and the result; a chain breaks at
   “=” onto lines whose “=” signs line up. Each word's numbers sit in a small table under the term they replace. */
const f4=x=>fixd(x,4);
const nw=s=>`<span class="sp-nw">${s}</span>`;
const M=t=>A.ui.math(t);
const tt=s=>`\\text{${String(s).replace(/[{}\\]/g,'')}}`;                  // a word inside a formula
/* a chain of equalities: the left side once, then each “= …” on its own line, every “=” in one column */
const EQ=(lhs,rhs)=>`<div class="sp-eq sp-math-line">`+rhs.map((r,i)=>`<span class="sp-eq-l">${i?'':M(lhs)}</span><span class="sp-eq-r">${M('= '+r)}</span>`).join('')+`</div>`;
/* a formula on its own line, its name in words on the left (R6.29) */
const FXL=(name,tex)=>`<div class="sp-fxl sp-math-line"><span class="sp-fxn">${name}</span>${M(tex)}</div>`;
const BLK=(lead,body)=>`<div class="sp-mblk"><p class="sp-mlead">${lead}</p>${body}</div>`;
const SOFT_TEX='p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}';
/* in the interview drill the numbers are the drill's own, so its terms are not wired to the settings */
let mPlain=false;
const TM=(k,x)=>mPlain?x:T_(k,x);
/* block 1: softmax with temperature, the formula and what each symbol is */
function mathSoftHead(T){
  if(T===0) return BLK(tx('<b>Temperature 0.</b> A logit cannot be divided by 0, so engines skip softmax and give all of the chance to the word with the highest logit (greedy decoding).',
                          '<b>temperature 为 0。</b>logit 不能除以 0，所以引擎跳过 softmax，把全部概率给 logit 最高的词（贪心解码）。'),'');
  return BLK(tx('<b>From logits to chances.</b> Each logit is divided by the temperature, and softmax turns the results into chances that add up to 100%.',
                '<b>从 logit 到概率。</b>每个 logit 先除以 temperature，softmax 再把结果变成加起来是 100% 的概率。'),
    FXL(TM('fx',tx('chance of word i','第 i 个词的概率')),SOFT_TEX)+
    `<p class="sp-msym">${tx(`${M('z_i')} is word i’s logit, and ${M('T')} is the temperature, now ${TM('T',fixd(T,1))}. The sum in the denominator runs over every word on the list, so the chances add up to 100%.`,
                             `${M('z_i')} 是第 i 个词的 logit，${M('T')} 是 temperature，现在是 ${TM('T',fixd(T,1))}。分母把列表里每个词的这一项都加起来，所以概率加起来是 100%。`)}</p>`);
}
/* each word's numbers: logit, e^(z/T), softmax chance and, after a cut, the final chance */
function mathTable(list,T){
  const d=list.d, cut=d.kLine!=null||d.pLine!=null, e=list.z.map(z=>Math.exp(z/T)), sum=e.reduce((a,b)=>a+b,0);
  const th=`<tr><th>${tx('word','词')}</th><th>logit ${M('z_i')}</th><th>${M('e^{z_i / T}')}</th><th>${M('p_i')}</th>${cut?`<th>${tx('after the cut','截断后')}</th>`:''}</tr>`;
  const tr=list.words.map((w,i)=>`<tr${cut&&!d.keep[i]?' class="sp-cutrow"':''}><td>${esc(w)}</td><td>${fixd(list.z[i],1)}</td><td>${f4(e[i])}</td><td>${f4(e[i]/sum)}</td>${cut?`<td>${d.keep[i]?f4(d.fin[i]):tx('cut','去掉')}</td>`:''}</tr>`).join('');
  const tot=`<tr class="sp-sumrow"><td>${tx('sum','合计')}</td><td></td><td>${f4(sum)}</td><td>1.0000</td>${cut?'<td>1.0000</td>':''}</tr>`;
  return `<div class="sp-mtab-w"><table class="sp-mtab">${th}${tr}${tot}</table></div>`;
}
/* the top word worked out: its chance from the formula, with the numbers in the table */
function mathTop(list,T){
  const e=list.z.map(z=>Math.exp(z/T)), sum=e.reduce((a,b)=>a+b,0), w=list.words[0];
  return `<p>${tx(`For “${esc(w)}”, the word with the highest logit:`,`以 logit 最高的“${esc(w)}”为例：`)}</p>`+
    EQ(`p_{${tt(w)}}`,[`\\frac{e^{${fixd(list.z[0],1)} / ${fixd(T,1)}}}{${f4(sum)}}`,`\\frac{${f4(e[0])}}{${f4(sum)}}`,f4(e[0]/sum)]);
}
/* the cut: which words stay, then the kept chances scaled back up to 100% */
function mathCut(list,T){
  const d=list.d, out=[], w=list.words[0], kept=(n,arr)=>arr.slice(0,n);
  if(d.kLine!=null){const ps=kept(d.kk,d.raw);
    out.push(BLK(tx(`<b>${TM('K',`Top-k ${d.kk}`)}.</b> Keep the ${d.kk} likeliest words and scale their chances up so they add up to 100% again.`,`<b>${TM('K',`top-k ${d.kk}`)}。</b>留下最可能的 ${d.kk} 个词，再把它们的概率按比例放大，让总和回到 100%。`),
      EQ(tx('\\text{kept}','\\text{留下的合计}'),[ps.map(f4).join(' + '),f4(d.sk)])+
      FXL(tx('chance after top-k','top-k 之后的概率'),tx('\\frac{p_i}{\\text{kept}}','\\frac{p_i}{\\text{留下的合计}}'))+
      `<p>${tx(`For “${esc(w)}”:`,`以“${esc(w)}”为例：`)}</p><div class="sp-fxl sp-math-line">${M(`\\frac{${f4(d.raw[0])}}{${f4(d.sk)}} = ${f4(d.q[0])}`)}</div>`));}
  if(d.pLine!=null){const qs=[]; let c=0; for(let i=0;i<d.n;i++){if(!d.keepK[i]) break; c+=d.q[i]; qs.push(d.q[i]); if(c>=d.P) break;}
    const n=qs.length, before=n>1?qs.slice(0,n-1).reduce((a,b)=>a+b,0):0, P=fixd(d.P,2);
    const ak=d.kLine!=null?tx(' (the chances after top-k)','（用 top-k 之后的概率）'):'';
    out.push(BLK(tx(`<b>${TM('P',`Top-p ${P}`)}.</b> Add up the chances${ak} from the likeliest word down and stop at the first word that brings the total to at least ${P}. The words up to and including it stay; the rest are cut.`,
                    `<b>${TM('P',`top-p ${P}`)}。</b>从最可能的词往下累加概率${ak}，在第一个让累计至少达到 ${P} 的词那里停下。到这个词为止的词留下，其余的去掉。`),
      (n>1?`<p>${tx(`The first ${n-1} ${n-1===1?'word holds':'words hold'} only ${f4(before)}, less than ${P}, so word ${n} stays too:`,`前 ${n-1} 个词加起来只有 ${f4(before)}，不到 ${P}，所以第 ${n} 个词也留下：`)}</p>`:
              `<p>${tx(`The likeliest word alone already holds at least ${P}:`,`最可能的那个词自己就至少有 ${P}：`)}</p>`)+
      EQ(n>1?`p_1 + \\cdots + p_{${n}}`:'p_1',[...(n>1?[qs.map(f4).join(' + ')]:[]),`${f4(d.s2)} \\geq ${P}`])+
      FXL(tx('chance after top-p','top-p 之后的概率'),tx('\\frac{p_i}{\\text{kept}}','\\frac{p_i}{\\text{留下的合计}}'))+
      `<p>${tx(`For “${esc(w)}”:`,`以“${esc(w)}”为例：`)}</p><div class="sp-fxl sp-math-line">${M(`\\frac{${f4(d.q[0])}}{${f4(d.s2)}} = ${f4(d.fin[0])}`)}</div>`));}
  return out.join('');
}
/* one list: its table, the top word worked out, then the cut */
function mathList(list,T,opt){
  opt=opt||{};
  if(T===0) return `<p>${tx(`All of the chance goes to “${esc(list.words[0])}”, whose logit, ${fixd(list.z[0],1)}, is the highest.`,`全部概率给了“${esc(list.words[0])}”，它的 logit ${fixd(list.z[0],1)} 最高。`)}</p>`;
  return mathTable(list,T)+(opt.top===false?'':mathTop(list,T))+mathCut(list,T);
}
function mathHTML(){
  const v=viewNow(), hd=s=>`<p class="sp-mh"><b>${esc(s)}</b></p>`;
  if(v==='duo') return mathSoftHead(1)+
    BLK(tx(`<b>The flat list:</b> the next word after “${esc(PROMPT.en)}”`,`<b>平的分布：</b>“${esc(PROMPT.zh)}”之后的下一个词`),mathList({words:TREE.map(x=>L(x.w)),z:Z1,d:flatD(lv)},1))+
    BLK(tx('<b>The peaked list:</b> the next word after “a burger”','<b>尖的分布：</b>“汉堡”之后的下一个词'),mathList({words:TREE[PEAK].c.map(x=>L(x.w)),z:TREE[PEAK].c.map(x=>x.z),d:peakD(lv)},1));
  if(v==='b'){const r=R(), p=runMode()?runPath():r.best, a=r.tree[p.i], b=a.kids[p.j], w1=TREE[p.i], w2=w1.c[p.j];
    /* the product is worked out from the factors as shown (4 significant digits), so it redoes by hand (R6.6) */
    const fs=[a.p,a.d.fin[p.j],b.d.fin[p.k]].map(sig4), prod=fs.reduce((x,y)=>x*parseFloat(y),1);
    return BLK(tx('<b>The chance of a sentence.</b> Each word is drawn from the list that the earlier words chose, so the chance of the whole sentence is the product of its words’ chances.',
                  '<b>一句话的概率。</b>每个词都从前面的词选定的那个列表里抽，所以整句话的概率等于各个词的概率相乘。'),
        FXL(tx('chance of the sentence','这句话的概率'),'P = p_1 \\times p_2 \\times p_3')+
        `<p class="sp-msym">${tx(`${M('p_1')} is the first word’s chance, ${M('p_2')} the second word’s on the list after the first, and ${M('p_3')} the third word’s on the list after the first two.`,
                                 `${M('p_1')} 是第一个词的概率，${M('p_2')} 是第二个词在第一个词之后那个列表里的概率，${M('p_3')} 是第三个词在前两个词之后那个列表里的概率。`)}</p>`+
        `<p>${tx(`For “${esc(sentOf(r,p))}”:`,`以“${esc(sentOf(r,p))}”为例：`)}</p>`+EQ('P',[fs.join(' \\times '),`${sig4(prod)} \\approx ${fP(prod)}`]))+
      mathSoftHead(S.T)+
      BLK(tx('<b>The first word</b>','<b>第一个词</b>'),mathList({words:TREE.map(x=>L(x.w)),z:Z1,d:r.d1},S.T,{top:false}))+
      BLK(tx(`<b>After “${esc(L(w1.w))}”</b>`,`<b>“${esc(L(w1.w))}”之后</b>`),mathList({words:w1.c.map(x=>L(x.w)),z:w1.c.map(x=>x.z),d:a.d},S.T,{top:false}))+
      BLK(tx(`<b>After “${esc(L(w1.w))} ${esc(L(w2.w))}”</b>`,`<b>“${esc(L(w1.w))}${esc(L(w2.w))}”之后</b>`),mathList({words:w2.c.map(x=>L(x.w)),z:w2.c.map(x=>x.z),d:b.d},S.T,{top:false}));}
  const list=listNow();
  let h=mathSoftHead(S.T)+BLK(v==='ch'?tx(`<b>The words after “${esc(CHP.en)}”</b>`,`<b>“${esc(CHP.zh)}”之后的词</b>`):tx(`<b>The next word after “${esc(PROMPT.en)}”</b>`,`<b>“${esc(PROMPT.zh)}”之后的下一个词</b>`),mathList(list,S.T));
  if(v==='ch'){const e=chEval(S.T,S.P), bad=CHW.map((x,i)=>x.bad?i:-1).filter(i=>i>=0);
    h+=BLK(tx('<b>The goal.</b> The three wrong words’ final chances, added up, must stay below 1%.','<b>目标。</b>三个错词的最终概率加起来，要低于 1%。'),
      FXL(tx('chance of a wrong word','抽中错词的概率'),`P_{${tx('\\text{wrong}','\\text{错}')}} = `+bad.map(i=>`p_{${tt(chW(i))}}`).join(' + '))+
      EQ(`P_{${tx('\\text{wrong}','\\text{错}')}}`,[bad.map(i=>f4(e.d.fin[i])).join(' + '),fP(e.wrong)]));}
  return h;
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!CH.Ts.includes(S.T)) S.T=CH.start.T; if(!CH.Ps.includes(S.P)) S.P=CH.start.P; S.K=0;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  if(S.T===0&&isP('temp')) seenT0=true; if(S.T>1&&isP('temp')) seenHigh=true;
  drawHead();
  const gr=$('#sp-graph',app); gr.dataset.focus=mode==='guided'?focusOf(STEPS[step]):'';
  drawA(); drawMetrics();
  const lg=$('#sp-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#sp-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* Settings (R4.2): in the guided parts only the temperature of part 3 can change; everything else is fixed, with a note that says to
   what; the challenge changes only its two settings, in the card */
function lockNote(){
  if(mode==='challenge') return tx(`In the challenge you change only the temperature and top-p, in the task card. Everything else is fixed: the prompt “${CHP.en}” and its seven scores, no top-k.`,`挑战中只能在任务卡里改 temperature 和 top-p。其余设置固定：prompt“${CHP.zh}”和它的七个分数，不用 top-k。`);
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'||id==='predict') return tx('Fixed in this part: temperature 1.0, top-p off, top-k off, seed 1. Part 3 lets you change the temperature.','这一部分固定：temperature 1.0，不用 top-p，不用 top-k，种子 1。第 3 部分可以改 temperature。');
  if(id==='temp') return tx('Fixed in this part: top-p off, top-k off, seed 1. You can change them in Scenarios.','这一部分固定：不用 top-p，不用 top-k，种子 1。在应用场景里可以改。');
  if(id==='cut') return tx('In this part you pick one rule in the card, at temperature 1.0 and seed 1. You can change the settings in Scenarios.','这一部分只在卡片里选一个规则，temperature 为 1.0，种子 1。在应用场景里可以改这些设置。');
  if(id==='tree') return tx('Fixed in this part: temperature 1.0, top-p off, top-k off, seed 1. You move the first random number in the card. You can change the settings in Scenarios.','这一部分固定：temperature 1.0，不用 top-p，不用 top-k，种子 1。第一个随机数在卡片里移动。在应用场景里可以改这些设置。');
  return tx('Fixed in this part: each “Show in the chart” button in the card sets its own example. You can change the settings in Scenarios.','这一部分固定：卡片里每个“在图里看”按钮都会设好自己的例子。在应用场景里可以改这些设置。');
}
function syncControls(){
  $$('input[data-key="T"]',app).forEach(i=>{const v=tIdx(S.T); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fT(S.T); i.setAttribute('aria-valuetext',fT(S.T));});
  $$('input[data-key="P"]',app).forEach(i=>{const v=pIdx(S.P); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fPp(S.P); i.setAttribute('aria-valuetext',fPp(S.P));});
  $$('input[data-key="u1"]',app).forEach(i=>{const v=Math.round(u1*100); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.inline-ctl').querySelector('output'); if(o) o.textContent=fixd(u1,2);});
  $$('.seg-ctl[data-key="K"]',app).forEach(g=>A.ui.setPressed(g,String(S.K)));
  const so=$('#sp-seed-out',app); if(so) so.textContent=tx(`seed ${S.seed}`,`种子 ${S.seed}`);
  const c=$('#sp-controls',app), ch=mode==='challenge', g=mode==='guided', liveT=!g||isP('temp');
  c.classList.toggle('locked',ch);
  const hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  hide('#sp-ctl-T-wrap',!liveT); hide('#sp-ctl-P-wrap',g); hide('#sp-ctl-K-wrap',g); hide('#sp-ctl-seed',g); hide('#sp-ctl-reset-wrap',!liveT);
  const ln=$('#sp-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const v=viewNow();
  if(v==='ch'){const e=chEval(S.T,S.P); A.announce(tx(`Temperature ${fTs(S.T)}, top-p ${fPp(S.P)}: the wrong words ${fP(e.wrong)}, ${e.n5} good words at 5% or more.`,`temperature ${fTs(S.T)}，top-p ${fPp(S.P)}：错词 ${fP(e.wrong)}，${e.n5} 个好词在 5% 以上。`)); return;}
  const r=R(); A.announce(tx(`Temperature ${fT(S.T)}: ${r.d1.nk} of 6 words can be picked; the 20 runs wrote ${r.distinct} different first ${r.distinct===1?'word':'words'}.`,`temperature ${fT(S.T)}：6 个词里 ${r.d1.nk} 个可能被抽中；20 次运行写出 ${r.distinct} 个不同的第一个词。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
const Tstops=[...Array(21).keys()].map(i=>i/10), Pstops=[...Array(20).keys()].map(i=>(i+1)/20);
function setState(p){
  if(preLock()&&p.T!=null) p={...p,T:S.T};
  Object.assign(S,p);
  S.T=Math.round(S.T*10)/10; S.P=Math.round(S.P*20)/20; if(S.T<0||S.T>2) S.T=1; if(S.P<0.05||S.P>1) S.P=1; if(![0,1,2,3,4,5].includes(S.K)) S.K=0;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(id){lv=id; lvTried.add(id); render();}
function setU1(v){u1=Math.max(0,Math.min(0.99,Math.round(v*100)/100)); u1Moved=true; render();}
/* part 6: each line puts its part's example back in the chart; on a phone the chart comes up to show it */
function setRem(k){if(!REMS.includes(k)) return; remView=k; if(k==='cut') lv='p9'; S={...BASE,T:k==='temp'?0.5:1}; renderRail(); if(innerWidth<=760) A.ui.reveal($('#sp-stage',app));}
/* your run: part 5, and part 6's tree line */
const runMode=()=>isP('tree')||(isP('remember')&&remView==='tree');
const runU1=()=>isP('tree')?u1:u1Seen();
function runPath(){return R().walk([runU1(),U23[0],U23[1]]);}
/* phones (A14): the part's one control, or the way to its question, at the top of the chart, so the control and what it changes share
   one screen; hidden above 760 px, where the card sits next to the chart */
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='predict'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='cut') return `<button type="button" class="btn ghost sm" data-act="to-rules" data-tipx="${esc(tx('Scrolls down to the three rules in the card.','向下滚到卡片里的三个规则。'))}">${tx('Pick a rule below ↓','在下面选规则 ↓')}</button>`;
  if(id==='temp') return tCtl('sp-ph-T',true);
  if(id==='tree') return u1Ctl('sp-ph-u1',false);
  return '';
}
function renderPhone(){const el=$('#sp-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  const st=$('#sp-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg=(innerWidth<=760?{temp:'#sp-ph-T',tree:'#sp-ph-u1'}:{})[s.id]||{temp:'#sp-ic-T',cut:'#sp-ic-lv',tree:'#sp-ic-u1'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#sp-rail .step.current',app)||$('#sp-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), as labs/prefill-decode.js does */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#sp-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#sp-rail .step.current',app)||$('#sp-rail .railcard',app)||$('#sp-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#sp-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#sp-insight',app)); else revealBox($('#sp-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,T:CH.start.T,P:CH.start.P,K:0};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE,seed:S.seed,view:'a'};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#sp-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[tIdx(S.T),pIdx(S.P),S.K,S.seed,S.view].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [ti,pi,K,seed]=p.slice(0,4).map(Number), view=p[4];
  if(!Number.isInteger(ti)||ti<0||ti>20||!Number.isInteger(pi)||pi<1||pi>20||![0,1,2,3,4,5].includes(K)||!Number.isInteger(seed)||seed<1||seed>9999||!['a','b'].includes(view)) return null;
  return {T:ti/10,P:pi/20,K,seed,view};
}

/* =====================================================================
   interview drill: seeded, from the engine; every line redoes from the numbers it shows (R6.6)
   ===================================================================== */
const DRILL_T=[0.5,0.8,1.5,2.0], DRILL_P=[0.5,0.6,0.7,0.8];
function makeDrill(seed){const rnd=mulberry(seed*104729+7), pk=a=>a[Math.floor(rnd()*a.length)];
  const z=[3.0,2.0,1.5,0.5].map(v=>Math.round((v+rnd()*0.6-0.3)*10)/10).sort((a,b)=>b-a), T=pk(DRILL_T), P=pk(DRILL_P);
  return {seed,z,T,P,d:dist(z,T,0,P)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#sp-drill-q',app), a=$('#sp-drill-a',app), nm=['A','B','C','D'];
  qd.innerHTML=tx(`A model scores four tokens A, B, C and D with logits <b>${d.z.map(v=>fixd(v,1)).join(', ')}</b>. You sample with temperature <b>${fixd(d.T,1)}</b> and top-p <b>${fixd(d.P,1)}</b>. Which tokens can be picked, and with what chances?`,
                  `模型给 A、B、C、D 四个 token 的 logit 是 <b>${d.z.map(v=>fixd(v,1)).join('、')}</b>。采样用 temperature <b>${fixd(d.T,1)}</b>、top-p <b>${fixd(d.P,1)}</b>。哪些 token 可能被抽中，概率各是多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const kept=nm.filter((_,i)=>d.d.keep[i]);
  /* H2 (2026-10-09): the same short blocks as Show the math, with the drill's own numbers */
  mPlain=true;
  a.innerHTML=`<p><b>${kept.join(tx(', ','、'))}${tx(': ','：')}${kept.map((k,i)=>`${k} ${fP(d.d.fin[i])}`).join(tx(', ','，'))}${tx('.','。')}</b></p>`+
    mathSoftHead(d.T)+BLK(tx('<b>The four tokens</b>','<b>四个 token</b>'),mathList({words:nm,z:d.z,d:d.d},d.T));
  mPlain=false;
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={T:{en:'Temperature',zh:'temperature'},P:{en:'Top-p',zh:'top-p'},K:{en:'Top-k',zh:'top-k'}};
const TERM={
 T:()=>`<div>${esc(TIP.temp())}</div><div class="fx">${A.ui.math('p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}')}</div>`,
 P:()=>`<div>${esc(TIP.topp())}</div><div class="fx">${A.ui.math('\\sum_{i \\in V} p_i \\geq p')}</div><div>${tx(`${A.ui.math('V')} is the set of words top-p keeps, ${A.ui.math('p_i')} the chance of word i, and ${A.ui.math('p')} the top-p setting.`,`${A.ui.math('V')} 是 top-p 留下的词，${A.ui.math('p_i')} 是第 i 个词的概率，${A.ui.math('p')} 是 top-p 的设定值。`)}</div>`,
 K:()=>`<div>${esc(TIP.topk())}</div>`,
 fx:()=>`<div>${esc(tx('Softmax with temperature gives each word its chance of being picked.','带 temperature 的 softmax 给出每个词被抽中的概率。'))}</div><div class="fx">${A.ui.math('p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}')}</div><div>${tx(`${A.ui.math('z_i')} is word i’s logit and ${A.ui.math('T')} the temperature. The sum in the denominator runs over every word, so the chances add up to 100%.`,`${A.ui.math('z_i')} 是第 i 个词的 logit，${A.ui.math('T')} 是 temperature。分母把每个词的这一项都加起来，所以概率加起来是 100%。`)}</div>`
};
const termTipHTML=k=>TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';
/* where a setting cannot change, its term says so and to what it is fixed, and a click does not jump (R1.4) */
function ctlLockMsg(k){
  if(mode==='challenge') return k==='K'?tx('Fixed during the challenge: no top-k.','挑战中固定不变：不用 top-k。'):'';
  if(mode!=='guided'||(k==='T'&&isP('temp'))) return '';
  if(isP('cut')) return tx('Fixed in this part: pick a rule in the card instead.','这一部分固定不变：请在卡片里选规则。');
  if(k==='T'&&step<SIX('temp')) return isP('predict')&&!pred?tx('The temperature can change after you answer part 2.','在第 2 部分作答后，才能改 temperature。'):tx('Fixed at 1.0 in this part; part 3 lets you change it.','这一部分固定为 1.0；第 3 部分可以改。');
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math'||k==='fx') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='strip') return tx('Click to scroll to the 20 runs in the chart.','点击滚到图里的 20 次运行。');
  if(k==='tree') return tx('Click to scroll to the sentence tree in the chart.','点击滚到图里的句子树。');
  if(k==='fid') return tx('Click to jump to the notes at the end of the page.','点击跳到页末的“简化、假设和来源”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  if(mode==='challenge') return tx(`Click to jump to “${L(CTLN[k])}” in the task card.`,`点击跳到任务卡里的“${L(CTLN[k])}”。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({T:$('#sp-ic-T',app)||$('#sp-ic-chT',app)||$('#sp-ctl-T',app),P:$('#sp-ic-chP',app)||$('#sp-ctl-P',app),K:$('#sp-ctl-K',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'||k==='fx'){const d=$('#sp-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='strip'||k==='tree'){const el=$(k==='strip'?'#sp-strip':'#sp-b',app); A.tip.hide(); if(el&&!el.hidden&&el.getClientRects().length){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} else {const g=$('#sp-graph',app); g.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(g);} return;}
  if(k==='controls'){const c=$('#sp-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  /* Fidelity: open the popover and flash the line about real servers at temperature 0 */
  if(k==='fid'){const b=$(`[data-pop="${ID}-fid"]`,app), p=document.getElementById(`${ID}-fid`); A.tip.hide(); if(p&&p.hidden&&b) b.click();
    const nd=document.getElementById('sp-fid-nd'), el=nd&&nd.getClientRects().length?nd:(p||b); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{if(mode==='challenge') setState({T:CH.start.T,P:CH.start.P}); else {const keep=preLock()?S.T:BASE.T; S={...BASE,T:keep,view:S.view}; render();}}); return;}
  if(a==='reseed'){bySettings(btn,()=>setState({seed:S.seed+1})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.T,S.P).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#sp-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-rules'){const q=$('#sp-ic-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#sp-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#sp-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='T') bySettings(e.target,()=>setState({T:Tstops[v]}));
    else if(k==='P') bySettings(e.target,()=>setState({P:Pstops[v-1]}));
    else if(k==='u1') byLearner(()=>setU1(v/100));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#sp-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#sp-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#sp-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('cut')){const f=()=>byLearner(()=>setLv(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    if(d.view!=null&&b.closest('#sp-view')){byLearner(()=>setState({view:d.view})); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='K') bySettings(b,()=>setState({K:+d.v}));
      else if(k==='chT') byLearner(()=>setState({T:+d.v}));
      else if(k==='chP') byLearner(()=>setState({P:+d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#sp-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#sp-rail',app)); return;}
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
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,sbTouched,seenT0,seenHigh,lv,lvTried:[...lvTried],
    u1,u1Moved,remView,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.T,S.P).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'&&typeof x.T==='number'&&typeof x.P==='number'?parse([tIdx(x.T),pIdx(x.P),x.K,x.seed,x.view].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    seenT0=o.seenT0===true; seenHigh=o.seenHigh===true; lv=LV.some(x=>x.id===o.lv)?o.lv:'none';
    lvTried=new Set(['none']); if(Array.isArray(o.lvTried)) o.lvTried.forEach(x=>{if(LV.some(l=>l.id===x)) lvTried.add(x);}); lvTried.add(lv);
    u1=typeof o.u1==='number'&&o.u1>=0&&o.u1<=0.99?Math.round(o.u1*100)/100:U1_0; u1Moved=o.u1Moved===true; remView=REMS.includes(o.remView)?o.remView:'runs';
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#sp-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#sp-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    fromSnap=false;
    fillRail();
    render();
    requestAnimationFrame(fitLabels);
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#sp-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#sp-math',app).open=true;}},
  onResize(){if(app) drawA();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    /* the part of the chart that must stay in view; once it scrolls away the mini result bar shows (R3.2) */
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='duo'?$('#sp-duo-f .sp-rows',app):(v==='b'?$('#sp-tree',app):$('#sp-rows',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#sp-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=nbh(s.short);
      if(v==='ch'){const e=chEval(S.T,S.P); vis.innerHTML=miniBar(e.d.fin,CHW.map(x=>!!x.bad)); read.innerHTML=esc(tx(`Temperature ${fTs(S.T)}, top-p ${fPp(S.P)}: the wrong words ${fP(e.wrong)}, ${e.n5} good words at 5% or more`,`temperature ${fTs(S.T)}，top-p ${fPp(S.P)}：错词 ${fP(e.wrong)}，${e.n5} 个好词在 5% 以上`)); return;}
      if(v==='duo'){const f=flatD(lv), p=peakD(lv); vis.innerHTML=miniBar(f.fin); read.innerHTML=esc(tx(`First word: ${f.nk} of 6 kept; after “a burger”: ${p.nk} of 4`,`第一个词留下 ${f.nk} / 6，“汉堡”之后留下 ${p.nk} / 4`)); return;}
      const r=R();
      if(v==='b'){const me=runMode()?runPath():r.best; vis.innerHTML=miniBar(r.d1.fin); read.innerHTML=esc((runMode()?tx('Your run: ','你这次运行：'):tx('Likeliest: ','最可能：'))+sentOf(r,me)+tx(` (${fP(me.p)})`,`（${fP(me.p)}）`)); return;}
      vis.innerHTML=miniBar(r.d1.fin);
      read.innerHTML=esc(tx(`Temperature ${fT(S.T)}, top-p ${fPp(S.P)}: 20 runs wrote ${r.distinct} different first ${r.distinct===1?'word':'words'}`,`temperature ${fT(S.T)}，top-p ${fPp(S.P)}：20 次运行写出 ${r.distinct} 个不同的第一个词`));}
  }
});
/* the mini result bar's picture: the final chances laid end to end */
function miniBar(fin,bad){let x=0; return `<div class="sp-dk">`+fin.map((p,i)=>{if(p<=0) return ''; const s=`<i class="${bad&&bad[i]?'bad':''}" style="left:${(x*100).toFixed(2)}%;width:${(p*100).toFixed(2)}%"></i>`; x+=p; return s;}).join('')+`</div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__sp=Object.freeze({
  dist:(z,T,K,P)=>deep(dist(z,T,K,P)), pick, runs:seed=>deep(runsOf(seed)),
  compute:s=>{const r=compute({...BASE,...s}); return deep({d1:r.d1,first:r.first,counts:r.counts,distinct:r.distinct,best:r.best,possible:r.possible,distinctS:r.distinctS,runs:r.runs,
    tree:r.tree.map(a=>({i:a.i,p:a.p,fin:a.d.fin,kids:a.kids.map(b=>({j:b.j,p:b.p,fin:b.d.fin,leaves:b.leaves}))}))});},
  walk:(s,u)=>deep(compute({...BASE,...s}).walk(u)),
  chEval:(T,P)=>deep(chEval(T,P)), chBest, chChanges,
  flat:id=>deep(flatD(id)), peak:id=>deep(peakD(id)), drill:s=>deep(makeDrill(s)), exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});},
  fmt:Object.freeze({P:fP,R:fR,pcs,sig4}),
  consts:Object.freeze({Z1:Z1.slice(),CHZ:CHZ.slice(),CHBAD:CHW.map(x=>!!x.bad),TREE:deep(TREE.map(a=>({w:a.w.en,z:a.z,c:a.c.map(b=>({w:b.w.en,z:b.z,c:b.c.map(c=>({w:c.w.en,z:c.z}))}))}))),
    N_LEAVES,PEAK,U23:U23.slice(),U1_0,U1_SHOW,CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),XA:XA.slice(),XP}),
  state:()=>({...S,mode,step,pred,observed:{...observed},seenT0,lv,lvTried:[...lvTried],u1,u1Moved,remView,chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
