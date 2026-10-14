/* pass-k (world eval): Can an agent with 60% pass@1 ship?
   Fidelity: Toy run. 40 coding tasks in three kinds (always pass, pass 60% of the time, never pass), mixed so that pass@1 stays 60%;
   every task is tried n times (10 by default) with seeded random numbers; pass@k and pass^k are estimated from the tries with the
   standard estimators, and the lab also knows each task's true chance, so it can show the true values next to the estimates.
   - pass@k per task: 1 − C(n−c, k)/C(n, k), the unbiased estimator of Chen et al. 2021, as openai/human-eval estimate_pass_at_k computes
     it: 1 − prod(1 − k / arange(n − c + 1, n + 1)), and 1 when n − c < k. pass^k per task: C(c, k)/C(n, k). Both are averaged over tasks.
   - The tries: mulberry32 random numbers, one per task and try; a try of a 60% task passes when its number is below 0.6. The 40 tasks sit
     in four rings of 10; at unevenness level L (0…4) the first 6L tasks always pass and the last 4L never pass, so each step moves one
     ring. Each ring is drawn again (next attempt of the same seed) until its tries pass exactly 60% of the time, and rings 2 and 3 until
     12 of their 20 first tries pass, so the measured pass@1 is exactly 60% at every level and part 1 shows 24 of 40 first tries passing.
   - The challenge is exact (closed form): agent A passes every task 70% of the time; agent B passes 70% of the tasks every time and
     never passes the rest. Calls stop at the first pass.
   - τ-bench: the original leaderboard (sierra-research/tau-bench README), retail domain, tool-calling strategy, gpt-4o:
     pass^1…4 = 0.604, 0.491, 0.430, 0.383.
   Words (EN / ZH): try / 尝试 (一次尝试), task / 任务, always passes / 稳过, sometimes passes / 时好时坏, never passes / 过不了,
   part / 部分. window.__pak is a read-only hook for tests/pass-k_numbers.py and tests/pass-k_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='pass-k';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const NT=40, Q=0.6, NS=[5,10,20], LVMAX=4;
const RING=[[0,1,2,3,4,5,36,37,38,39],[6,7,8,9,10,11,32,33,34,35],[12,13,14,15,16,17,28,29,30,31],[18,19,20,21,22,23,24,25,26,27]];
const split=lv=>({a:6*lv,s:NT-10*lv,z:4*lv});
const kindOf=(i,lv)=>i<6*lv?'a':(i>=NT-4*lv?'z':'s');
const TAU=[0.604,0.491,0.430,0.383];
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* the tries of a task that passes 60% of the time: B[i][j] is true when try j of task i passed */
const TRI=new Map();
function triesOf(seed,n){
  const key=seed+'|'+n; if(TRI.has(key)) return TRI.get(key);
  const want=Math.round(Q*n*10);
  const gen=(r,att)=>{const rnd=mulberry((seed*2654435761+n*97+r*7919+att*104729)>>>0); return RING[r].map(()=>{const row=[]; for(let j=0;j<n;j++) row.push(rnd()<Q); return row;});};
  const sum=g=>g.reduce((a,row)=>a+row.filter(Boolean).length,0), first=g=>g.filter(row=>row[0]).length;
  const G=[null,null,null,null];
  for(let a3=0;a3<4000&&!G[2];a3++){const g3=gen(3,a3); if(sum(g3)!==want) continue; const need=12-first(g3); if(need<0||need>10) continue;
    for(let a2=0;a2<4000;a2++){const g2=gen(2,100000+a3*4000+a2); if(sum(g2)===want&&first(g2)===need){G[3]=g3; G[2]=g2; break;}}}
  for(const r of [1,0]) for(let att=0;att<20000;att++){const g=gen(r,att); if(sum(g)===want){G[r]=g; break;}}
  const B=Array(NT); RING.forEach((ring,r)=>ring.forEach((slot,m)=>{B[slot]=G[r][m];}));
  TRI.set(key,B); return B;
}
/* the two estimators, per task (human-eval's product form for pass@k) */
function pAt(n,c,k){if(k>n) return NaN; if(n-c<k) return 1; let p=1; for(let j=n-c+1;j<=n;j++) p*=1-k/j; return 1-p;}
function pHat(n,c,k){if(k>n) return NaN; if(c<k) return 0; let p=1; for(let j=0;j<k;j++) p*=(c-j)/(n-j); return p;}
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
/* part 1 shows the tasks in a fixed shuffled order (a sorted grid would give away which tasks never pass) */
const SHUF=(()=>{const r=mulberry(7), ix=[...Array(NT).keys()]; for(let i=NT-1;i>0;i--){const k=Math.floor(r()*(i+1)); [ix[i],ix[k]]=[ix[k],ix[i]];} return ix;})();
const CC=new Map();
function compute(s){
  const key=[s.lv,s.n,s.seed].join('|'); if(CC.has(key)) return CC.get(key);
  const n=s.n, B=triesOf(s.seed,n), sp=split(s.lv);
  const tasks=B.map((row,i)=>{const k=kindOf(i,s.lv), t=k==='a'?row.map(()=>true):(k==='z'?row.map(()=>false):row.slice()); return {i,kind:k,t,c:t.filter(Boolean).length,p:k==='a'?1:(k==='z'?0:Q)};});
  const ord={a:0,s:1,z:2};
  const sorted=tasks.slice().sort((x,y)=>ord[x.kind]-ord[y.kind]||y.c-x.c||x.i-y.i);
  const ks=[...Array(n).keys()].map(i=>i+1);
  const at={}, hat={}, firstK={}, tAt={}, tHat={}, nAt={}, nHat={};
  const p1=mean(tasks.map(x=>x.c/n));
  ks.forEach(k=>{at[k]=mean(tasks.map(x=>pAt(n,x.c,k))); hat[k]=mean(tasks.map(x=>pHat(n,x.c,k))); firstK[k]=mean(tasks.map(x=>x.t.slice(0,k).some(Boolean)?1:0));
    tAt[k]=mean(tasks.map(x=>1-Math.pow(1-x.p,k))); tHat[k]=mean(tasks.map(x=>Math.pow(x.p,k))); nAt[k]=1-Math.pow(1-p1,k); nHat[k]=Math.pow(p1,k);});
  const zero=tasks.filter(x=>x.c===0).length, full=tasks.filter(x=>x.c===n).length, first1=tasks.filter(x=>x.t[0]).length;
  const r={s:{...s},n,sp,tasks,sorted,ks,at,hat,firstK,tAt,tHat,nAt,nHat,p1,zero,full,first1};
  CC.set(key,r); return r;
}
/* the challenge, exact: A passes every task 70% of the time; B passes 70% of the tasks every time and never passes the rest */
const CH={p:0.7,goal:0.95,ks:[1,2,3,4,5,6,7,8],ags:['A','B'],start:{ag:'B',k:1}};
function chEval(ag,k){const p=CH.p, f=1-p;
  const done=ag==='A'?1-Math.pow(f,k):p, calls=ag==='A'?(1-Math.pow(f,k))/p:p+f*k;
  return {ag,k,done,calls,ok:done>=CH.goal-1e-12};}
/* minimal: the fewest expected model calls among the options that meet the goal */
function chBest(){let b=null; CH.ags.forEach(ag=>CH.ks.forEach(k=>{const e=chEval(ag,k); if(e.ok&&(!b||e.calls<b.calls-1e-12)) b=e;})); return b;}

/* =====================================================================
   formatting (R6.6: one decimal for shares on the cards and in the results)
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>fixd(x*100,1)+'%';
/* shares that round to 100.0% but are not 100% (the challenge's long retries) */
const fPc=x=>x<1&&x>=0.9995?'>99.9%':fP(x);
/* between two sentences: a space in English, none in Chinese */
const SEP=()=>A.lang()==='zh'?'':' ';
const f3=x=>fixd(x,3);
const NBH='‑';
const GPT='gpt'+NBH+'4o';
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const M=t=>A.ui.math(t);
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="pak-eqs${EQN(rows)?' pak-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="pak-eq-l">${l?M(l):''}</span><span class="pak-eq-e">${M(rel||'=')}</span><span class="pak-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="pak-mb">${lead?`<p>${lead}</p>`:''}${body}${res?`<p class="pak-mb-res">${res}</p>`:''}</div>`;
const NAIVE_TEX=k=>`1 - {(1 - \\text{pass@}1)}^{${k}}`;
const EST5_TEX='1 - \\frac{C(n - c, 5)}{C(n, 5)}';   // a task's pass@5 from its c passes in n tries (the transfer question with n = 3)
/* the kinds of task, by name */
const KN={a:{en:'always pass',zh:'稳过'},s:{en:'sometimes pass',zh:'时好时坏'},z:{en:'never pass',zh:'过不了'}};
const mixText=lv=>{const s=split(lv);
  if(lv===0) return tx('all 40 tasks pass 60% of the time','40 个任务每次都是 60% 的把握');
  if(s.s===0) return tx(`${s.a} tasks always pass, ${s.z} never pass`,`稳过 ${s.a} 个、过不了 ${s.z} 个`);
  return tx(`${s.a} always pass, ${s.s} pass 60% of the time, ${s.z} never pass`,`稳过 ${s.a} 个、时好时坏 ${s.s} 个（每次 60%）、过不了 ${s.z} 个`);};
const LVN=[{en:'all alike',zh:'都一样'},{en:'a little uneven',zh:'差别小'},{en:'in between',zh:'中间'},{en:'very uneven',zh:'差别大'},{en:'all or nothing',zh:'全有或全无'}];

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={lv:2,k:5,n:10,seed:1,tau:0};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seen=new Set(), est='naive', estTried=new Set(['naive']), prod='code', prodTried=new Set(['code']), remView='uneven', moved4=false;
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, chAg=CH.start.ag, chK=CH.start.k;
let drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false;
let shown=0, playing=false, raf=0, spd='med', gridSig='', actPending=false, lastGrid='';   // lastGrid: the tasks of the last grid on screen (R3.20)
const SPEEDS={slow:1.5,med:4,fast:15};          // tries (rows) per second: 10 tries in about 6.7 s, 2.5 s, 0.7 s
const shownPrev={};
/* S25: entering a part, a mode or a scenario drops every ▲▼ badge; the next drawing starts the comparison afresh */
let dropB=false;
const dropBadges=()=>{dropB=true;};
const R=()=>compute(S);

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 at:()=>tx('pass@k: the share of tasks where at least one of k tries passes. It is what a product counts when a check (for example unit tests) can pick the passing try and the user only ever sees that one.',
           'pass@k：k 次尝试里至少有一次通过的任务，占全部任务的比例。产品能自动判断对错（比如有单元测试）、用户只看到通过的那一次时，算的就是它。'),
 hat:()=>tx('pass^k: the share of tasks where all k tries pass. It is what a product counts when the user sees every try, so every one of them has to be right.',
            'pass^k：k 次尝试全部通过的任务，占全部任务的比例。用户看得到每一次尝试、每一次都要对时，算的就是它。'),
 p1:()=>tx('pass@1: the share of tries that pass, averaged over the tasks: how often a single try is right.',
           'pass@1：每个任务的尝试里通过的比例，再对所有任务取平均，也就是一次尝试做对的机会。'),
 tri:()=>tx('A try is one full attempt at a task: the agent writes its answer once, and the task’s unit tests say whether it passed.',
            '一次尝试，就是 agent 把一个任务完整做一遍：写出一份答案，由这个任务的单元测试判断通过没有。'),
 kinds:()=>tx('In this lab every task is one of three kinds: it always passes, it passes 60% of the time, or it never passes. Real tasks can have any chance from 0 to 100%. Three kinds keep the picture readable.',
              '这个 Lab 里的任务只分三种：稳过、每次 60% 的把握（时好时坏）、过不了。真实任务的把握可以是 0 到 100% 之间的任何值。分成三种，是为了让图一眼看得懂。'),
 tau:()=>tx(`τ‑bench sets an agent, given an API’s tools and a policy to follow, in conversations with a user simulated by a language model, in two domains: an airline and a retail shop. The numbers are from its original leaderboard: retail tasks, tool-calling strategy, ${GPT}. Its README warns that these task versions are outdated.`,
            `τ‑bench 让一个 agent 拿着一套 API 工具、照着一份规定，和语言模型扮演的用户对话。场景有两个：一家航空公司和一家网店。这些数来自它最初的排行榜：网店（retail）任务，tool-calling 策略，${GPT}。它的 README 提醒，这一版的任务已经过时。`)
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'low',t:{en:'About 60% (under 70%)',zh:'60% 左右（不到 70%）'}},
  {v:'mid',t:{en:'About 80% (70% to 90%)',zh:'80% 左右（70% 到 90%）'}},
  {v:'high',t:{en:'Close to 100% (90% or more)',zh:'接近 100%（90% 以上）'}}];
const predRight=()=>{const v=compute({...BASE}).at[5]; return v<0.7?'low':(v<0.9?'mid':'high');};
const predT=v=>L(PRED.find(o=>o.v===v).t);
const lvCtl=(id,note)=>A.ui.range({id,key:'lv',label:tx('How uneven the tasks are','任务之间的差别'),min:0,max:LVMAX,value:S.lv,out:L(LVN[S.lv]),inline:true,note:note?mixText(S.lv):''});
const EST=[{id:'naive',t:{en:'Plug pass@1 into '+M(NAIVE_TEX(5)),zh:'代入公式 '+M(NAIVE_TEX(5))},d:{en:'assumes every task is as hard as the average',zh:'假设每个任务都和平均一样难'}},
  {id:'first',t:{en:'Count only the first 5 tries',zh:'只数前 5 次'},d:{en:'a task counts as done if one of its first 5 tries passed; tries 6 to 10 go unused',zh:'前 5 次里有一次过就算做成，后 5 次不用'}},
  {id:'full',t:{en:'Use all 10 tries (as HumanEval does)',zh:'用全部 10 次估（HumanEval 的做法）'},d:{en:'the formula above: every way to pick 5 of the 10 tries, averaged',zh:'就是上面的公式：把从 10 次里挑 5 次的每一种挑法都算上，取平均'}}];
const estVal=id=>{const r=compute({...BASE}); return id==='naive'?r.nAt[5]:(id==='first'?r.firstK[5]:r.at[5]);};
const estCtl=()=>`<div id="pak-ic-est">`+A.ui.levers({aria:tx('Pick a way to compute pass@5','选一种算 pass@5 的方法'),value:est,options:EST.map(e=>({v:e.id,title:L(e.t),desc:L(e.d),result:estTried.has(e.id)||e.id===est?fP(estVal(e.id)):''}))})+`</div>`;
const PROD=[{id:'code',k:5,m:'at',t:{en:'Coding: unit tests check each try, up to 5 tries',zh:'写代码：单元测试判对错，最多试 5 次'},d:{en:'a failed try is retried; the user only sees the try that passed',zh:'没通过就重试，用户只看到通过的那一次'}},
  {id:'support',k:4,m:'hat',t:{en:'Support: the same kind of request comes 4 times',zh:'客服：同一类请求来 4 次'},d:{en:'nothing checks an answer before the user sees it, so all 4 must be right',zh:'没有东西能先替用户判对错，所以 4 次都要对'}}];
const prodOf=id=>PROD.find(p=>p.id===id)||PROD[0];
const prodVal=id=>{const p=prodOf(id), r=compute({...BASE}); return p.m==='at'?r.at[p.k]:r.hat[p.k];};
const prodRes=id=>{const p=prodOf(id); return tx(`look at ${p.m==='at'?'pass@':'pass^'}${p.k}: ${fP(prodVal(id))}`,`看 ${p.m==='at'?'pass@':'pass^'}${p.k}：${fP(prodVal(id))}`);};
const prodCtl=()=>`<div id="pak-ic-prod">`+A.ui.levers({aria:tx('Pick a product','选一个产品'),value:prod,options:PROD.map(p=>({v:p.id,title:L(p.t),desc:L(p.d),result:prodTried.has(p.id)||p.id===prod?esc(prodRes(p.id)):''}))})+`</div>`;
const FX_AT='\\text{pass@}k = \\frac{1}{T} \\sum_{i} \\left[ 1 - \\frac{C(n - c_i, k)}{C(n, k)} \\right]'.replace('\\left[','[').replace('\\right]',']');
const FX_HAT='\\text{pass}^k = \\frac{1}{T} \\sum_{i} \\frac{C(c_i, k)}{C(n, k)}';
const fxLine=t=>`<div class="pak-fx">${M(t)}</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'first',gate:()=>!!pred,skip:true,
  lead:()=>tx('A coding agent passes 60% of tasks on the first try. With up to 5 tries, how many get done?','写代码的 agent 第一次做对 60% 的任务。最多试 5 次，能做成多少？'),
  enter:()=>({...BASE}),
  body:()=>{const r=compute({...BASE}), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Your coding agent gets 60% of tasks right on the first try. The chart has ${NT} tasks, each ${DEF('tried',TIP.tri())} once: ${r.first1} passed.</p><p>Every task has unit tests that tell whether a try is right. So you let the agent try up to 5 times, and a task counts as done as soon as one try passes.</p><p>How many of the tasks get done now?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the result.</p>':''}`,
             `<p>你的写代码 agent 第一次就做对 60% 的任务。图里有 ${NT} 个任务，每个先${DEF('试了',TIP.tri())} 1 次，${r.first1} 个通过了。</p><p>每个任务都有单元测试，能判断一次尝试对不对。所以你让 agent 最多试 5 次，有一次通过就算做成。</p><p>现在能做成多少任务？</p>${ch}${pred?'<p class="muted small">已记下。第 2 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'uneven',title:{en:'The same 60%, very different tasks',zh:'同样是 60%，任务可以很不一样'},view:'grid',
  lead:()=>tx('Drag the slider: the more uneven the tasks, the less retrying rescues.','拖动滑块：任务之间差别越大，重试救回的越少。'),
  enter:()=>{seen=new Set(); return {...BASE};},expect:()=>seen.has(0)&&seen.has(LVMAX),showMe:()=>setState({lv:seen.has(0)?LVMAX:0}),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The same 60% can hide very different tasks. Retrying rescues only the tasks that ${DEF('sometimes pass',TIP.kinds())}; a task that fails every time stays failed however often you retry.</p><p>The slider changes how the ${NT} tasks split into three kinds, while pass@1 stays at 60%. Drag it to both ends:</p>`,
              `<p>同样是 60%，任务之间可以很不一样。重试只救得回${DEF('时好时坏',TIP.kinds())}的任务；每次都做不出的任务，试多少次都没用。</p><p>滑块改的是 ${NT} 个任务怎么分成三种，pass@1 一直是 60%。把它拖到两头各看一次：</p>`)+lvCtl('pak-ic-lv',true),
  obs:()=>obsUneven()},
 {id:'estimate',title:{en:'Estimating pass@k from tries',zh:'从尝试结果估出 pass@k'},view:'grid',
  lead:()=>tx('You only see how often each task was tried and how often it passed.','你只看得到每个任务试了几次、过了几次。'),
  enter:()=>{est='naive'; estTried=new Set(['naive']); return {...BASE};},expect:()=>estTried.has('full'),showMe:()=>setEst('full'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>You cannot see a task’s true chance. You see only that it was tried n times and passed c times. pass@k asks: if you pick k of those n tries at random, what is the chance that at least one passed? Average that over the T tasks:</p>${fxLine(FX_AT)}<p>${M('C(n, k)')} counts the ways to pick k of the n tries. ${M('C(n - c, k)')} counts the picks made only from the ${M('n - c')} failed tries. Their ratio is the chance that all k picked tries failed, and 1 minus it is the chance that at least one passed.</p><p>Here are three ways to compute pass@5. Pick one and see how close its pass@5 comes to the true value.</p>`,
              `<p>你看不到一个任务真实的把握，只看得到它试了 n 次、过了 c 次。pass@k 问的是：从这 n 次里随便挑 k 次，至少有一次通过的概率是多少？把它对 T 个任务取平均：</p>${fxLine(FX_AT)}<p>${M('C(n, k)')} 是从 n 次里挑 k 次的挑法数。${M('C(n - c, k)')} 是只从失败的 ${M('n - c')} 次里挑的挑法数。两者相除，就是挑出的 k 次全都失败的概率。用 1 减去它，就是至少有一次通过的概率。</p><p>下面是算 pass@5 的三种方法。选一种，看看它算出的 pass@5 准不准。</p>`)+estCtl(),
  obs:()=>obsEst()},
 {id:'every',title:{en:'When every time counts: pass^k',zh:'每一次都要对：pass^k'},view:'grid',
  lead:()=>tx('Make the lab’s pass^k curve match τ‑bench’s real one.','让这个 Lab 的 pass^k 曲线贴近 τ‑bench 的真实成绩。'),
  enter:()=>{moved4=false; return {...BASE,lv:0,k:4,tau:1};},expect:()=>moved4,showMe:()=>setState({lv:2}),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>A support agent cannot quietly retry behind the scenes: every answer goes straight to a user. The same kind of request comes from many users, and every one of them must get a right answer. So what counts is whether the same task comes out right all k times. That is ${DEF('pass^k',TIP.hat())}:</p>${fxLine(FX_HAT)}<p>The chance that k tries picked at random from the n all passed is ${M('\\frac{C(c, k)}{C(n, k)}')}.</p><p>${DEF('τ‑bench',TIP.tau())} is a public benchmark for support agents, and its leaderboard reports pass^1 to pass^4 (the diamonds in the chart). Right now every task passes 60% of the time. Drag the slider to bring the lab’s pass^k curve close to the diamonds:</p>`,
              `<p>客服 agent 没法在后台偷偷重试，它的每个回答都直接给到用户。同一类请求会有很多用户来问，每一次都得答对。所以要看同一个任务做 k 次是不是次次都对，这就是 ${DEF('pass^k',TIP.hat())}：</p>${fxLine(FX_HAT)}<p>从 n 次里随便挑 k 次，挑出的全都通过的概率，就是 ${M('\\frac{C(c, k)}{C(n, k)}')}。</p><p>${DEF('τ‑bench',TIP.tau())} 是一个公开的客服 agent 评测，它的排行榜报告了 pass^1 到 pass^4（图里的菱形）。现在每个任务都是 60% 的把握。拖动滑块，让这个 Lab 的 pass^k 曲线贴近这些菱形：</p>`)+lvCtl('pak-ic-lv4',true),
  obs:()=>obsEvery()},
 {id:'decide',title:{en:'Which number decides shipping',zh:'上线该看哪个指标'},view:'grid',
  lead:()=>tx('If a check can pick the passing try, look at pass@k; if not, at pass^k.','能自动判对错，就看 pass@k；不能，就看 pass^k。'),
  enter:()=>{prod='code'; prodTried=new Set(['code']); return {...BASE,k:5};},expect:()=>prodTried.has('support'),showMe:()=>setProd('support'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The same agent can need a different number in a different product. What decides it is whether something can tell, before the user sees it, that a try is right.</p><p>Pick a product and see which number it should look at before shipping.</p>`,
              `<p>同一个 agent 放进不同的产品，要看的指标不一样。关键在于：用户看到之前，有没有东西能判断一次尝试对不对。</p><p>选一个产品，看看它上线前该看哪个指标。</p>`)+prodCtl(),
  obs:()=>obsDecide()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:'grid',
  lead:()=>tx('Same pass@1, different retries; estimate from the tries; pass^k when every time counts.','同样的 pass@1，重试的效果不同；按尝试结果来估；每次都要对时看 pass^k。'),
  enter:()=>{remView='uneven'; return {...BASE,lv:LVMAX};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(predRight()));
  return pred===predRight()?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
const waitLine=()=>tx(`<p class="small muted">The ${NT} tasks are being tried; the result shows once all ${S.n} tries are in.</p>`,`<p class="small muted">${NT} 个任务正在试，试完 ${S.n} 次后，这里给出结果。</p>`);
const done=()=>shown>=S.n;
function obsUneven(){
  if(!done()) return waitLine();
  const r=R(), sp=r.sp, v=fP(r.at[5]);
  let res, why;
  if(S.lv===0){res=tx(`<p><b>With all ${NT} tasks at 60%, up to 5 tries get ${v} of the tasks done.</b></p>`,`<p><b>${NT} 个任务都是 60% 的把握时，最多试 5 次能做成 ${v} 的任务。</b></p>`);
    why=tx('<p>Why: each task has the same 60% chance on every try, so failing all 5 tries is rare. Almost every task gets done.</p>','<p>原因：每个任务每次都有六成把握，5 次全失败的情况很少，所以几乎每个任务都能做成。</p>');}
  else if(sp.s===0){res=tx(`<p><b>With ${sp.a} tasks that always pass and ${sp.z} that never pass, up to 5 tries still get only ${v} done.</b></p>`,`<p><b>${sp.a} 个任务稳过、${sp.z} 个过不了时，最多试 5 次还是只能做成 ${v}。</b></p>`);
    why=tx('<p>Why: the tasks that always pass need no retry, and the tasks that never pass fail again on every retry. Retrying rescues nothing.</p>','<p>原因：稳过的任务用不着重试，过不了的任务重试多少次都失败，所以重试一个任务也救不回来。</p>');}
  else {res=tx(`<p><b>With ${mixText(S.lv)}, up to 5 tries get ${v} of the tasks done.</b></p>`,`<p><b>${mixText(S.lv)}时，最多试 5 次能做成 ${v} 的任务。</b></p>`);
    const cap=NT-sp.z, capP=Math.round(cap/NT*100)+'%';
    why=tx(`<p>Why: retrying rescues almost every task that sometimes passes. The ${sp.z} that never pass stay failed however often they are retried. So at most the other ${cap} tasks, the ones that always or sometimes pass, get done: ${capP} of the ${NT}.</p>`,`<p>原因：时好时坏的任务，重试几乎都能救回来。过不了的 ${sp.z} 个，试多少次都救不回来。所以能做成的，最多只有稳过和时好时坏的那 ${cap} 个任务，也就是 ${capP}。</p>`);}
  const more=(seen.has(0)&&seen.has(LVMAX))?'':(seen.has(0)?tx('<p class="small">Now drag it to the right end.</p>','<p class="small">再把它拖到最右边。</p>'):(seen.has(LVMAX)?tx('<p class="small">Now drag it to the left end.</p>','<p class="small">再把它拖到最左边。</p>'):tx('<p class="small">Drag it to both ends.</p>','<p class="small">把它拖到两头各看一次。</p>')));
  return res+(S.lv===2?verdict():'')+why+more;
}
function obsEst(){
  if(!done()) return waitLine();
  const r=compute({...BASE}), v=fP(estVal(est)), tv=fP(r.tAt[5]);
  if(est==='naive') return tx(`<p><b>Plugging in gives ${v}, yet ${r.zero} of these ${NT} tasks never passed in 10 tries.</b></p><p>Why: the formula treats 60% as every task’s chance. Here 60% is the average of tasks that always pass, sometimes pass and never pass, so the formula overrates what retrying does.</p>`,
                                   `<p><b>代公式得到 ${v}，可这 ${NT} 个任务里有 ${r.zero} 个，10 次一次都没过。</b></p><p>原因：公式把 60% 当成每个任务的把握。这里的 60% 是稳过、时好时坏、过不了三种任务平均出来的，所以公式高估了重试的效果。</p>`);
  if(est==='first') return tx(`<p><b>Counting the first 5 tries: ${v} of the tasks passed at least once in them.</b></p><p>This is unbiased too: redo the eval many times, and it averages out to the true value. But it uses only half the tries, so it swings more with luck. The formula uses all 10 tries. It averages over every way to pick 5 of them, which makes it much steadier.</p>`,
                              `<p><b>只数前 5 次：有 ${v} 的任务在前 5 次里至少过了一次。</b></p><p>这样数也没有偏差：把评测重做很多次，平均下来正好是真实值。可它只用了一半的尝试，结果更容易受运气左右。公式用上了全部 10 次。它把从 10 次里挑 5 次的每一种挑法都算一遍再取平均，所以波动小得多。</p>`);
  return tx(`<p><b>From all 10 tries, pass@5 comes out at ${v}. The true value, which this lab knows because it set each task’s chance, is ${tv}.</b></p><p>This is what the HumanEval evaluation code does: sample n answers per task, count the c that pass, and average the formula over the tasks. With fewer tries than k there is no unbiased estimate, and that code does not report one.</p>`,
            `<p><b>用全部 10 次，估出 pass@5 是 ${v}。这个 Lab 自己设定了每个任务的把握，所以知道真实值是 ${tv}。</b></p><p>HumanEval 的评测代码就是这样做的：每个任务采样 n 个答案，数出通过的 c 个，按公式算出每个任务的值，再取平均。试的次数比 k 少时，没有无偏的估法，那段代码就不报这个数。</p>`);
}
function obsEvery(){
  if(!done()) return waitLine();
  const r=R(), v=fP(r.hat[4]), eq=fP(Math.pow(TAU[0],4));
  const head=tx(`<p><b>On τ‑bench’s leaderboard, ${GPT}’s pass^4 on retail tasks is ${fP(TAU[3])} (the table says ${f3(TAU[3])}). If every task were equally hard, it would be only ${eq}.</b></p>`,
                `<p><b>在 τ‑bench 的排行榜上，${GPT} 做网店任务的 pass^4 是 ${fP(TAU[3])}（表里写 ${f3(TAU[3])}）。如果每个任务一样难，它只会是 ${eq}。</b></p>`);
  if(S.lv===0) return head+tx(`<p>With every task at 60%, the lab’s pass^4 is ${v}, down on the dashed line. Drag the slider to the right.</p>`,`<p>每个任务都是 60% 时，这个 Lab 的 pass^4 是 ${v}，落在虚线上。把滑块往右拖。</p>`);
  const best=bestTauLv(), near=S.lv===best;
  return head+(near?tx(`<p>With ${mixText(S.lv)}, the lab’s pass^4 is ${v}: the closest to τ‑bench.</p>`,`<p>${mixText(S.lv)}时，这个 Lab 的 pass^4 是 ${v}，最贴近 τ‑bench。</p>`)
                   :tx(`<p>With ${mixText(S.lv)}, the lab’s pass^4 is ${v}. ${S.lv<best?'Drag it further right.':'That is too far; drag it back a little.'}</p>`,`<p>${mixText(S.lv)}时，这个 Lab 的 pass^4 是 ${v}。${S.lv<best?'再往右拖一点。':'拖过头了，往回拖一点。'}</p>`))+
    tx('<p>Why: pass^k falls with k only on tasks that sometimes pass. τ‑bench’s curve falls slowly, so many of its tasks must pass every time or almost never.</p>','<p>原因：只有时好时坏的任务，会让 pass^k 随 k 往下掉。τ‑bench 的曲线掉得慢，说明它的任务里有不少是次次都过、或者几乎总失败的。</p>');
}
/* the level whose pass^1…4 is closest to τ-bench's (sum of squared gaps), seed 1, 10 tries */
function bestTauLv(){let b=0, bd=Infinity; for(let lv=0;lv<=LVMAX;lv++){const r=compute({...BASE,lv}); const d=TAU.reduce((a,t,i)=>a+Math.pow(r.hat[i+1]-t,2),0); if(d<bd){bd=d; b=lv;}} return b;}
function obsDecide(){
  if(!done()) return waitLine();
  const r=compute({...BASE}), sp=r.sp;
  if(prod==='code') return tx(`<p><b>A coding product looks at pass@5: ${fP(r.at[5])}.</b></p><p>The unit tests pick the passing try for the user; a failed try costs only one more call. More tries help, but they never rescue the ${sp.z} tasks that never pass: those need a better agent.</p>`,
                              `<p><b>写代码的产品看 pass@5：${fP(r.at[5])}。</b></p><p>单元测试替用户挑出了通过的那一次，失败的尝试只是多花一次调用。多试几次有用，但救不回那 ${sp.z} 个过不了的任务，它们要靠更好的 agent。</p>`);
  return tx(`<p><b>A support product looks at pass^4: only ${fP(r.hat[4])}.</b></p><p>Nothing checks an answer before the user reads it, so every wrong one reaches a user. Retrying does not help here. Find the ${sp.s} tasks the agent only sometimes gets right, learn why it fails on them, and change the prompt or the tools until it passes them every time.</p><p>Report both numbers: pass@k says how far retrying can go, and pass^k says whether users get a right answer every time.</p>`,
            `<p><b>客服产品看 pass^4：只有 ${fP(r.hat[4])}。</b></p><p>用户读到回答之前，没有东西替他判对错，所以每一个错的回答都会被用户看到。这里重试帮不上忙。要找出 agent 时对时错的那 ${sp.s} 个任务，查清它为什么有时失败，再改提示词或工具，让它在这些任务上次次都做对。</p><p>评测报告里两个数都要写：pass@k 说明重试能救到哪里，pass^k 说明用户能不能每次都拿到对的结果。</p>`);
}
/* part 6: each line with its part's example; a button puts that picture back on the stage (R6.23) */
const REMS=['uneven','estimate','every'];
function remBtn(k){const on=remView===k, tip={uneven:tx('Shows the all-or-nothing mix: 24 tasks always pass, 16 never.','在图上显示“全有或全无”：稳过 24 个，过不了 16 个。'),
    estimate:tx('Shows the tasks of the initial setup with the dashed line of the plugged-in formula.','在图上显示初始设置的任务和代公式得到的虚线。'),every:tx('Shows the mix closest to τ‑bench, with its diamonds.','在图上显示最贴近 τ‑bench 的那组任务和它的菱形。')}[k];
  return ` <button type="button" class="linkbtn pak-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const r0=compute({...BASE,lv:0}), r4=compute({...BASE,lv:LVMAX}), b=compute({...BASE}), bt=compute({...BASE,lv:bestTauLv()}), ts=bt.sp;
  return tx(`<ul class="pak-remlist"><li><b>The same pass@1 can mean very different retries.</b> Retrying rescues only the tasks that sometimes pass; a task that never passes stays failed however often you retry. So at the same pass@1, the more uneven the tasks, the less retrying rescues.${remBtn('uneven')}</li>`+
    `<li><b>Estimate pass@k from each task’s own tries.</b> Plugging pass@1 into ${M(NAIVE_TEX('k'))} assumes every task is equally hard, so it overrates retrying. Count each task’s n tries and c passes, and use the formula instead.${remBtn('estimate')}</li>`+
    `<li><b>When users see every answer, look at pass^k, and make the agent right every time on the tasks it only sometimes passes.</b> A real agent’s pass^k falls more slowly than if every task were equally hard. It passes many tasks every time and loses on the ones it only sometimes passes, so those are the tasks to work on.${remBtn('every')}</li></ul>`+
    `<p class="small muted">A long agent task works the same way: every step has to be right, the steps’ chances multiply, and the more steps, the lower the chance that the whole task succeeds. The lab ${labLink('agent-reliability')} covers this.</p>`+
    `<p class="pak-answer"><b>Back to the opening question:</b> the 60% alone cannot say whether the agent can ship. A coding product can let unit tests pick the right try, so it looks at pass@k. With up to 5 tries, this agent gets ${fP(b.at[5])} of the tasks done. A support product shows every answer to a user, so it looks at pass^k. Only ${fP(b.hat[4])} of the tasks are answered right all 4 times.</p>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “pass@k is ${M(NAIVE_TEX('k'))}.” Only when every task is equally hard. When some tasks are much harder than others, the formula overrates retrying.</p><p>✗ “A high pass@10 means a reliable agent.” It only says that one of 10 tries passed: for the same tasks, pass@10 is ${fP(b.at[10])} and pass^10 is ${fP(b.hat[10])}.</p><p>✗ “Trying each task more times in the eval raises pass@k.” More tries in the eval only make the estimate more precise. The true pass@k depends on the agent’s chance on each task and on k, not on how many tries the eval ran (n).</p></div>${A.shell.toScenarios()}`,
    `<ul class="pak-remlist"><li><b>同样的 pass@1，重试的效果可以差很多。</b>重试只救得回时好时坏的任务，每次都过不了的任务，试多少次都没用。所以 pass@1 一样时，任务之间差别越大，重试救回的越少。${remBtn('uneven')}</li>`+
    `<li><b>按每个任务自己的尝试结果来估 pass@k。</b>把 pass@1 代进 ${M(NAIVE_TEX('k'))}，等于假设每个任务一样难，会高估重试的效果。要数出每个任务试了 n 次、过了 c 次，再用公式来估。${remBtn('estimate')}</li>`+
    `<li><b>用户看得到每一次回答时，看 pass^k，并让 agent 在时好时坏的任务上次次做对。</b>真实 agent 的 pass^k，比每个任务一样难时掉得慢。因为很多任务它次次都过，失分集中在时好时坏的那些任务上，要改进的正是这些任务。${remBtn('every')}</li></ul>`+
    `<p class="small muted">很长的 agent 任务也是这样：每一步都要做对，各步的把握连乘起来，步数越多，整件事做成的机会越低。“${labLink('agent-reliability')}”这个 Lab 专门讲这件事。</p>`+
    `<p class="pak-answer"><b>回到开头的问题：</b>pass@1 = 60% 的 agent 能不能上线，光看 60% 答不上来。写代码的产品能用单元测试挑出对的那一次，所以看 pass@k。这个 agent 最多试 5 次，能做成 ${fP(b.at[5])} 的任务。客服的每个回答都直接给到用户，所以看 pass^k。同一类请求来 4 次、次次都答对的任务，只有 ${fP(b.hat[4])}。</p>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “pass@k 就是 ${M(NAIVE_TEX('k'))}。”只有每个任务一样难时才成立。任务有难有易时，这个公式会高估重试的效果。</p><p>✗ “pass@10 很高，说明 agent 很可靠。”它只说明 10 次里至少有一次通过：同一批任务，pass@10 是 ${fP(b.at[10])}，pass^10 只有 ${fP(b.hat[10])}。</p><p>✗ “评测时每个任务多试几次，pass@k 就会变高。”评测时多试几次，只会让估计更准。真实的 pass@k 由 agent 在各个任务上的把握和 k 决定，和每个任务试了几次（n）无关。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 he:'https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py',
 heRead:'https://github.com/openai/human-eval',
 tau:'https://github.com/sierra-research/tau-bench',
 codex:'https://arxiv.org/abs/2107.03374',
 taup:'https://arxiv.org/abs/2406.12045'
};
const PRESETS=[
 {id:'code',t:{en:'Coding agent with unit tests',zh:'有单元测试的写代码 agent'},d:{en:'Tests pick the passing try, so retrying is cheap',zh:'测试能挑出通过的那一次，重试代价小'},s:{lv:2,k:5,tau:0}},
 {id:'support',t:{en:'Support agent, the same requests again and again',zh:'客服 agent，同类请求反复出现'},d:{en:'Every answer reaches a user, so each must be right',zh:'每个回答都会被用户看到，每次都要对'},s:{lv:2,k:4,tau:0}},
 {id:'tau',t:{en:'τ‑bench’s real table',zh:'τ‑bench 的真实成绩'},d:{en:'A published pass^k table for a support agent',zh:'一张公开的客服 agent pass^k 表'},s:{lv:2,k:4,tau:1}},
 {id:'cat',t:{en:'All the failures in one kind of task',zh:'失败都集中在一类任务上'},d:{en:'The agent never passes, say, any database migration',zh:'比如，凡是数据库迁移的任务，agent 一个都过不了'},s:{lv:4,k:5,tau:0}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.lv===S.lv&&p.s.k===S.k&&p.s.tau===S.tau&&S.n===10&&S.seed===1);
function scenHTML(i){
  const p=PRESETS[i], r=compute({...BASE,...p.s});
  if(p.id==='code') return tx(`<p><b>A coding agent with unit tests is judged by pass@k: here up to 5 tries get ${fP(r.at[5])} of the tasks done.</b></p><p>Unit tests pick out the passing try, so a failed try costs only a little time and one more call. That is why code benchmarks such as ${cite(SRC.heRead,'HumanEval')} report pass@k. They sample each task many times, then estimate with the formula the share of tasks where at least one of k tries passes.</p>`,
                                 `<p><b>有单元测试的写代码 agent 看 pass@k：这里最多试 5 次能做成 ${fP(r.at[5])} 的任务。</b></p><p>单元测试能挑出通过的那一次，失败的尝试只多花一点时间和一次调用。所以 ${cite(SRC.heRead,'HumanEval')} 这类代码评测报告的是 pass@k。它们给每个任务采样很多次，再按公式估出“k 次里至少一次通过”的任务占多少。</p>`);
  if(p.id==='support') return tx(`<p><b>A support agent is judged by pass^k: here all 4 answers to the same kind of request are right for only ${fP(r.hat[4])} of the tasks.</b></p><p>Its pass@1 is still 60%, yet over 4 requests of the same kind, ${fP(1-r.hat[4])} of the tasks get at least one wrong answer. Nothing can retry behind the user’s back. To raise pass^k, find the tasks the agent only sometimes gets right, learn why, and change the prompt or the tools until it gets them right every time.</p>`,
                                    `<p><b>客服 agent 看 pass^k：同一类请求来 4 次、4 次都答对的任务，这里只有 ${fP(r.hat[4])}。</b></p><p>它的 pass@1 仍是 60%，可同一类请求来上 4 次，就有 ${fP(1-r.hat[4])} 的任务至少答错一次。在用户面前，没法偷偷重试。要提高 pass^k，得找出 agent 时对时错的任务，查清原因，再改提示词或工具，让它在这些任务上次次做对。</p>`);
  if(p.id==='tau') return tx(`<p><b>On τ‑bench’s leaderboard, ${GPT} has a pass^1 of ${fP(TAU[0])} and a pass^4 of ${fP(TAU[3])} on retail tasks (the diamonds).</b></p><p>If every task were equally hard, pass^4 would be only about ${Math.round(r.nHat[4]*100)}% (the dashed line). The real curve sits near the lab’s mix with ${mixText(S.lv)}: many tasks pass every time and many almost never. The ${cite(SRC.tau,'README')} warns that these task versions are outdated.</p>`,
                                `<p><b>在 τ‑bench 的排行榜上，${GPT} 做网店任务的 pass^1 是 ${fP(TAU[0])}，pass^4 是 ${fP(TAU[3])}（图里的菱形）。</b></p><p>如果每个任务一样难，pass^4 只有 ${Math.round(r.nHat[4]*100)}% 左右（图里的虚线）。真实的曲线贴近这个 Lab 里${mixText(S.lv)}的情况：很多任务次次都过，很多几乎总失败。它的 ${cite(SRC.tau,'README')} 提醒，这一版的任务已经过时。</p>`);
  return tx(`<p><b>When every failure is in one kind of task, retrying does nothing: up to 5 tries still get ${fP(r.at[5])} done, the same as one try.</b></p><p>The ${split(LVMAX).z} tasks that never pass fail for the same reason every time. Find that kind of task among the failed ones, learn why the agent cannot do it, and give it what it lacks (a tool, a prompt, or access). A higher retry count only adds calls.</p>`,
            `<p><b>失败都集中在一类任务上时，重试毫无用处：最多试 5 次还是只做成 ${fP(r.at[5])}，和只试 1 次一样。</b></p><p>那 ${split(LVMAX).z} 个过不了的任务，每次都因为同一个原因失败。要从失败的任务里找出这一类，查清 agent 为什么做不了，再给它补上缺的东西（一个工具、一段提示词或某项权限）。调高重试次数只会多花调用。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const agName=ag=>ag==='A'?tx('Agent A','agent A'):tx('Agent B','agent B');
const chSet=(ag,k)=>tx(`${agName(ag)}, up to ${k} ${k===1?'try':'tries'}`,`${agName(ag)}，最多试 ${k} 次`);
const fC=x=>fixd(x,2);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chAg,chK), best=chBest();
  let h;
  const good=`<p class="okline pak-mt"><b>`+tx(`Goal met: ${chSet(chAg,chK)} gets ${fPc(me.done)} of the tasks done, with ${fC(me.calls)} calls per task on average.`,`达成目标：${chSet(chAg,chK)}，能做成 ${fPc(me.done)} 的任务，平均每个任务调用 ${fC(me.calls)} 次。`)+`</b>`;
  if(me.ok&&me.ag===best.ag&&me.k===best.k) h=good+tx(' And that is the fewest calls: one try fewer misses the goal.</p>','而且调用次数已经最少：再少试一次就不达标。</p>');
  else if(me.ok) h=good+tx(' But fewer calls also meet it.</p>','不过还有更省的做法。</p>')+tx(`<p>Up to ${best.k} tries get ${fP(best.done)} done with ${fC(best.calls)} calls per task. Each extra try is used only when every earlier one failed. That is rare, but each use costs one more call.</p>`,`<p>最多试 ${best.k} 次就能做成 ${fP(best.done)}，平均只调用 ${fC(best.calls)} 次。${me.k-best.k===1?`多出来的第 ${best.k+1} 次尝试，只在前 ${best.k} 次都失败时才会用到`:`第 ${best.k+1} 次及以后的尝试，只在前面全都失败时才会用到`}。用得很少，但每用一次都要多花一次调用。</p>`);
  else h=tx(`<p class="badline pak-mt"><b>Not yet: ${chSet(chAg,chK)} gets ${fP(me.done)} of the tasks done, not 95%.</b></p>`,`<p class="badline pak-mt"><b>还没达到：${chSet(chAg,chK)}，只能做成 ${fP(me.done)} 的任务，不到 95%。</b></p>`)+
      (chAg==='B'?tx('<p class="small">Hint: B’s failures are all in tasks it never passes. How many of those can a retry rescue?</p>','<p class="small">提示：B 的失败全在它从来过不了的任务上。重试能救回其中几个？</p>')
        :tx('<p class="small">Hint: A fails a task only when every try fails. Each extra try makes that rarer.</p>','<p class="small">提示：A 只有每一次都失败，才做不成一个任务。多试一次，这种情况就更少一些。</p>'))+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the owner's review of batch 8 (2026-10-07): the chart keeps the options that meet the goal, the pick and the initial setup (brief) */
  const rows=[]; CH.ags.forEach(ag=>CH.ks.forEach(k=>rows.push(chEval(ag,k))));
  const shown=rows.filter(e=>e.ok||(e.ag===chAg&&e.k===chK)||(e.ag===CH.start.ag&&e.k===CH.start.k));
  h+=A.ui.cmpChart({title:tx('Share of tasks done when both agents get 70% right on the first try, with the average calls per task (goal: at least 95%)','两个 agent 第一次都做对 70% 时，做成的任务占多少，平均每个任务调用几次（目标：至少 95%）'),
    aria:tx('Share of tasks done: ','做成的任务：')+shown.map(e=>`${chSet(e.ag,e.k)}${tx(': ','：')}${fPc(e.done)}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:tx(`${e.ok?'✓ ':''}${e.ag}, ${e.k} ${e.k===1?'try':'tries'}: ${fC(e.calls)} calls`,`${e.ok?'✓ ':''}${e.ag}，最多 ${e.k} 次：调用 ${fC(e.calls)} 次`),value:e.done*100,text:fPc(e.done),ok:e.ok,me:e.ag===chAg&&e.k===chK,
      tags:(e.ag===CH.start.ag&&e.k===CH.start.k?[{text:tx('initial setup','初始设置')}]:[]).concat(e.ag===chAg&&e.k===chK?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,goal:95,goalLabel:tx('goal: 95%','目标：95%'),max:100,okText:tx('✓ meets the goal','✓ 达标'),noText:tx('misses it','不达标'),goalText:tx('goal: 95%','目标：95%')});
  h+=tx(`<p class="small">In the initial setup, B tries once and gets ${fP(chEval('B',1).done)} done. Its failures are in tasks it never passes, so retrying rescues none of them. With any number of tries, B stays at ${fP(CH.p)} and misses the goal.</p><p class="small">A gets past 95% at ${best.k} tries. A fourth try is used only when the first three failed, so it adds few calls, but it still adds some.</p>`,
        `<p class="small">初始设置是 B 只试 1 次，做成 ${fP(chEval('B',1).done)}。B 的失败都在那些从来过不了的任务上，重试一个也救不回来。B 试几次都停在 ${fP(CH.p)}，达不到目标。</p><p class="small">A 试到 ${best.k} 次就超过 95%。第 4 次只在前 3 次都失败时才会用到，多花的调用不多，但还是会多。</p>`);
  h+=tx('<p class="small muted">Note: here the two agents’ chances are given. In a real eval you can only estimate them from tries. Both have the same pass@1, so to tell them apart you have to look at how many times each task passed.</p>',
        '<p class="small muted">注意：这里两个 agent 的把握是直接给定的。真实评测里，你只能从尝试结果去估。两个 agent 的 pass@1 一样，要分清它们，得看每个任务各通过了几次。</p>');
  return h;
}

/* =====================================================================
   copy the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Predict how many tasks retrying gets done, then see how the spread between tasks decides it, how to estimate pass@k from tries, when pass^k is the number that counts, and which number decides shipping. Start here if this is new to you.',
            '一共 6 个部分。先预测重试能做成多少任务，再看任务之间的差别怎样决定答案，怎样从尝试结果估出 pass@k，什么时候该看 pass^k，最后决定上线该看哪个指标。第一次学从这里开始。'),
  sandbox:tx('Four real situations, from a coding agent with unit tests to τ‑bench’s published table. Pick one to see which number counts and how large it is.',
             '四种真实情况，从有单元测试的写代码 agent 到 τ‑bench 公开的成绩表。选一个，看该看哪个指标、它是多少。'),
  challenge:tx('Two agents both pass 70% of tasks on the first try. A coding product needs at least 95% of tasks done in the end. Pick the agent and the most tries, with as few model calls as possible.',
               '两个 agent 第一次都做对 70% 的任务。一个写代码的产品要求最终至少做成 95% 的任务。选 agent 和最多尝试次数，在达标的前提下让调用次数最少。')
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> the ${NT} coding tasks are made up. Whether each try passes is decided by seeded random numbers in your browser. pass@k and pass^k are estimated from those tries with the standard formulas. The lab sets each task’s chance itself, so the page can also work out the true values and set them beside the estimates. ${J('math','“Show the math”')} works every number out.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Each task has a fixed chance of passing a try, and tries are independent of each other.</li>`+
 `<li>pass@k per task is ${M('1 - \\frac{C(n - c, k)}{C(n, k)}')}, the unbiased estimator of Chen et al. (2021), in the product form that HumanEval’s <code>estimate_pass_at_k</code> uses; pass^k per task is ${M('\\frac{C(c, k)}{C(n, k)}')}, the standard estimator of a task’s chance to the power k. Both are averaged over the tasks.</li>`+
 `<li>The challenge is computed exactly from the two agents’ chances; an agent stops at its first passing try.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>In this lab every task is one of three kinds: always passes, passes 60% of the time (sometimes passes), or never passes. Real tasks can have any chance from 0 to 100%. Three kinds are an assumption of this lab, chosen so the picture reads at a glance. They do not change which way each number moves.</li>`+
 `<li>The tries come from mulberry32 random numbers with seed 1. To keep the measured pass@1 at exactly the stated 60%, the lab redraws each group of 10 tasks until its tries pass exactly 60% of the time. The estimates still differ a little from the true values, as real ones do.</li>`+
 `<li>Real tries of the same task are often not independent (the same prompt, the same misunderstanding), which pushes tasks toward always or never passing.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>The estimator: openai/human-eval, ${cite(SRC.he,'<code>human_eval/evaluation.py</code>')} (<code>estimate_pass_at_k</code>); its ${cite(SRC.heRead,'README')} says there is no unbiased estimate of pass@k with fewer samples than k.</li>`+
 `<li>τ‑bench numbers: the original leaderboard in the ${cite(SRC.tau,'sierra-research/tau-bench README')} (retail, tool-calling, ${GPT}), which warns that these task versions are outdated.</li>`+
 `<li>Papers, cited by title only: “Evaluating Large Language Models Trained on Code” (Chen et al., 2021, arXiv 2107.03374); “τ‑bench” (Yao et al., 2024, arXiv 2406.12045). No number on this page is taken from them.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-03. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>这里的 ${NT} 个写代码任务是编出来的。每次尝试过没过，由你浏览器里带种子的随机数决定。pass@k 和 pass^k 都按标准公式，从这些尝试结果估出来。每个任务的把握是这个 Lab 自己设的，所以页面也能算出真实值，和估出来的数对照。页面上每个数怎么算出来，都写在${J('math','“展开公式”')}里。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>每个任务每次尝试通过的把握是固定的，各次尝试互相独立。</li>`+
 `<li>每个任务的 pass@k 是 ${M('1 - \\frac{C(n - c, k)}{C(n, k)}')}，即 Chen 等（2021）的无偏估计，写法和 HumanEval 的 <code>estimate_pass_at_k</code> 一样（连乘形式）；每个任务的 pass^k 是 ${M('\\frac{C(c, k)}{C(n, k)}')}，是“把握的 k 次方”的标准估计。两个都再对任务取平均。</li>`+
 `<li>挑战题按两个 agent 的把握精确计算；agent 一通过就停下，不再试。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>这个 Lab 里的任务只分三种：稳过、每次 60% 的把握（时好时坏）、过不了。真实任务的把握可以是 0 到 100% 之间的任何值。分成三种是这个 Lab 的假设，为的是让图一眼看得懂。它不会改变各个数随设置变化的方向。</li>`+
 `<li>尝试结果来自种子为 1 的 mulberry32 随机数。为了让测出的 pass@1 正好是题目里说的 60%，每 10 个任务一组，反复重抽，直到这一组的尝试正好有 60% 通过。估出来的数和真实值仍有一点差别，真实评测也是这样。</li>`+
 `<li>真实情况下，同一个任务的几次尝试往往不独立（同样的提示词、同样的误解），这会让任务更偏向稳过或者过不了。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>估计公式：openai/human-eval 的 ${cite(SRC.he,'<code>human_eval/evaluation.py</code>')}（<code>estimate_pass_at_k</code>）；它的 ${cite(SRC.heRead,'README')} 说，样本比 k 少时没有 pass@k 的无偏估计。</li>`+
 `<li>τ‑bench 的数：${cite(SRC.tau,'sierra-research/tau-bench 的 README')} 里最初的排行榜（retail，tool-calling，${GPT}）；README 提醒这一版的任务已经过时。</li>`+
 `<li>论文只列标题：“Evaluating Large Language Models Trained on Code”（Chen 等，2021，arXiv 2107.03374）；“τ‑bench”（Yao 等，2024，arXiv 2406.12045）。页面上没有数取自论文。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-03。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>An eval score is measured on a limited set of tasks, so it is a noisy estimate. The lab ${labLink('eval-ci')} covers that. This lab sets it aside and asks one question: when each task may be tried several times, how many more tasks get done?</p>`,
                    `<p>评测分数只在有限的任务上测过，本身是带噪声的估计。“${labLink('eval-ci')}”这个 Lab 专门讲这种噪声。这里先不管它，只看一个问题：同一个任务多试几次，能多做成多少任务？</p>`);
const OUTCOME=()=>tx('choose between pass@k and pass^k for a product, estimate both from repeated tries, and read a real pass^k table.','为一个产品选对 pass@k 还是 pass^k，从多次尝试结果估出它们，并读懂真实的 pass^k 成绩表。');
const EXITS=()=>{
  /* the exact chance that a task at 60% passes at least once in 5 tries: 1 − 0.4⁵ (B1: not the seeded estimate of part 2) */
  const q5=1-Math.pow(1-Q,5);
  return {
 explain:[
  {q:tx('Two agents both pass 60% of tasks on the first try. With up to 5 tries, one gets 99% of its tasks done and the other only 60%. Why?','两个 agent 第一次都做对 60% 的任务。最多试 5 次，一个能做成 99% 的任务，另一个只做成 60%。为什么？'),
   o:[tx('The first has a 60% chance on every task, so 5 tries almost always include a pass; the second’s failures are all in tasks it never passes, and a retry cannot rescue those.','第一个对每个任务都有六成把握，5 次里几乎总有一次通过；第二个的失败全在它从来过不了的任务上，重试救不回来。'),
      tx('The second agent answers more consistently, so each retry repeats the same mistake; raising its sampling temperature would bring it to 99% too.','第二个 agent 每次回答都更一致，重试只会重复同一个错误；把它的采样温度调高，它也能到 99%。'),
      tx('The second agent’s 60% was measured with more noise, so its true pass@1 is lower, and retrying it adds less.','第二个 agent 的 60% 测得噪声更大，真实的 pass@1 更低，所以重试带来的更少。')],a:0,
   n:[null,tx('In this lab a task that never passes has a 0 chance on every try, and no sampling temperature changes that. In real systems a higher temperature makes the answers more varied and now and then rescues a task that always failed. Getting to 99% still needs every task to have a fair chance to begin with.','在这个 Lab 里，过不了的任务每次的把握都是 0，调采样温度也改变不了。真实系统里，调高温度能让回答更多样，偶尔能救回一个原来总失败的任务。可要做到 99%，还得每个任务本来就有不小的把握。'),
     tx('Both pass@1 values are the same 60%, measured the same way. The difference is not noise; it is how the passes spread over the tasks.','两个 pass@1 都是用同样的方法测出的 60%。差别不在噪声，而在通过的次数怎样分布在各个任务上。')],
   why:tx(`pass@k first takes each task’s chance that at least one of k tries passes, then averages over the tasks. When every task passes 60% of the time, each one’s chance of at least one pass in 5 tries is ${fP(q5)}. When 60% of the tasks always pass and the rest never do, the passing tasks need no retry and the failing ones fail every retry, so pass@5 stays at 60%.`,`pass@k 先看每个任务“k 次里至少一次通过”的机会，再对所有任务取平均。如果每个任务都有六成把握，它们 5 次里至少过一次的机会都是 ${fP(q5)}。如果六成任务稳过、其余过不了，稳过的不用重试，过不了的重试多少次都失败，pass@5 就还是 60%。`)},
  {q:tx(`An eval tried each task 10 times. Why not compute pass@k by plugging pass@1 into ${M(NAIVE_TEX('k'))}?`,`评测给每个任务试了 10 次。为什么不能把 pass@1 代进 ${M(NAIVE_TEX('k'))} 来算 pass@k？`),
   o:[tx('The formula assumes every task is as hard as the average. When tasks differ, it overrates retrying, so each task’s pass@k is estimated from its own c passes in n tries and then averaged.','这个公式假设每个任务都和平均一样难。任务难度不同时，它高估了重试的效果，所以要按每个任务自己 n 次里过了 c 次来估，再取平均。'),
      tx('The formula is right; it only needs the pass rate over all 10 tries instead of the first try’s.','公式本身没错，只是要代入 10 次的平均通过率，而不是第一次的。'),
      tx('The formula underrates pass@k, because it ignores tasks that pass several times.','这个公式会低估 pass@k，因为它没算上同一个任务通过好几次的情况。')],a:0,
   n:[null,tx('The pass rate over all 10 tries is the same 60% here; the trouble is the assumption that every task is equally hard.','这里 10 次的平均通过率也是 60%；问题出在“每个任务一样难”这个假设上。'),
     tx('It overrates rather than underrates when tasks differ; and several passes of one task count once anyway, as “at least one pass”.','任务难度不同时，它是高估而不是低估；而且一个任务通过好几次，本来就只算一次“至少一次通过”。')],
   why:tx(`As a task’s chance grows, its chance of at least one pass in k tries rises fast at first and then slowly. So averaging each task’s chance of at least one pass is not the same as plugging in the average chance. Say 60% of the tasks always pass and 40% never do: the average is still 60%, yet pass@5 is only 60%, far below the ${fP(q5)} of tasks all at 60%. Estimating task by task keeps each task’s own c.`,`“k 次里至少一次通过”的机会，随把握先涨得快、后涨得慢，所以先算每个任务的机会再平均，和把平均把握代进公式，结果不一样。比如六成任务次次都过、四成次次都失败，平均把握也是 60%，pass@5 却只有 60%，远低于每个任务都是 60% 时的 ${fP(q5)}。按任务来估，就保留了每个任务自己的 c。`)}],
 transfer:[
  {q:tx('A support agent answers 90% of requests right the first time. The product needs the same kind of request answered right 3 times in a row, so it looks at pass^3. Case 1: every task has a 90% chance. Case 2: 90% of the tasks are always answered right and 10% always wrong. About what is pass^3 in each case? (For reference: every task at 60% gives a pass^4 of about 13%.)','一个客服 agent 第一次就答对 90% 的请求。产品要求同一类请求连续 3 次都答对，也就是看 pass^3。情况一，每个任务都有九成把握。情况二，九成的任务次次答对，一成次次答错。两种情况下，pass^3 大约各是多少？（参照：每个任务都是 60% 时，pass^4 约 13%。）'),
   o:[tx('About 90% in both','两种都约 90%'),tx('About 73% in case 1, 90% in case 2','情况一约 73%，情况二约 90%'),tx('About 90% in case 1, 73% in case 2','情况一约 90%，情况二约 73%'),tx('About 73% in both','两种都约 73%')],a:1,
   why:tx(`In case 1, all 3 tries pass with chance 0.9 × 0.9 × 0.9 = ${fP(Math.pow(0.9,3))}. In case 2, the 90% of tasks that are always right pass all 3 times and the rest never do, so pass^3 stays at 90%.`,`情况一，3 次全过的机会是 0.9 × 0.9 × 0.9 = ${fP(Math.pow(0.9,3))}。情况二，九成的任务 3 次都对，其余的一次也不对，所以 pass^3 还是 90%。`)},
  {q:tx('An eval tried each task only 3 times (n = 3). Can it estimate pass@5?','一次评测给每个任务只试了 3 次（n = 3）。它能估出 pass@5 吗？'),
   o:[tx(`Yes: run the same formula, ${M(EST5_TEX)}, counting a task as 1 when it failed fewer than 5 times`,`能：同一个公式 ${M(EST5_TEX)} 照算，失败不到 5 次的任务算作 1`),tx('No: with only 3 results per task, nothing estimates the chance of at least one pass in 5','不能：每个任务只有 3 次结果，估不出“5 次里至少一次通过”的机会'),tx(`Yes: work out pass@1 first, then plug it into ${M(NAIVE_TEX(5))}, as if every task had that chance`,`能：先算出 pass@1，再代进 ${M(NAIVE_TEX(5))}，当作每个任务都是这个把握`)],a:1,
   why:tx('With only 3 results per task, no way of computing makes the estimate average out to the true chance of at least one pass in 5, so there is no unbiased estimate. HumanEval’s README says the same, and its evaluation code does not report such a k. Run as is, the formula counts every task as 1 and always gives 100%. Plugging into ' +M(NAIVE_TEX(5))+' is right only if every task is equally hard.','只有 3 次结果时，不管怎么算，估出来的数平均下来都对不上“5 次里至少一次通过”的真实机会，所以没有无偏的估法。HumanEval 的 README 也这么说，它的评测代码遇到这种 k 就不报告。照搬公式的话，每个任务都会算成 1，pass@5 永远是 100%。代进 '+M(NAIVE_TEX(5))+'，只有每个任务一样难时才对。')}],
 practice:[
  {id:'agent-harness-eval',d:tx('Deep dive on evaluation: every task runs 3 times and “stability” is the agreement across the repeats. Decide what to report from those runs (pass@1, pass@3 or pass^3) and what a CI gate should compare.','评测那一部分的深入讨论：每个任务跑 3 次，“稳定性”是几次之间的一致程度。决定从这几次运行里报告什么（pass@1、pass@3 还是 pass^3），以及 CI 门槛该比较哪项指标。')},
  {id:'ml-fundamentals',d:tx('Q1: one summary number (accuracy) hides what matters on imbalanced data, as pass@1 hides how the passes spread over tasks. Say which summary misleads, and why.','第 1 题：一个汇总数（准确率）在不平衡的数据上会掩盖关键信息，就像 pass@1 掩盖了通过次数在任务之间的分布。说出哪个汇总数会误导，为什么。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'eval',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'pass@k',title:l?L(l.q):tx('Can an agent with 60% pass@1 ship?','pass@1 = 60% 的 agent 能上线吗？'),
    minutes:10,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: a score is an estimate','前置知识：评测分数是估计值'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
const SPDL={slow:{en:'Slow',zh:'慢'},med:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
function stageHTML(){
  return `<div class="panel stage pak-stage" id="pak-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="pak-st-title"></div><div class="meter-sub" id="pak-st-sub"></div></div><span class="status" id="pak-status" tabindex="0"></span></div>`+
   `<div class="stage-graph graph pak-graph" id="pak-graph">`+
    `<div class="pak-gridbox" id="pak-gridbox"><div class="pak-cap"><span id="pak-grid-cap"></span><span class="pak-leg" id="pak-grid-leg"></span></div>`+
     `<svg class="pak-grid" id="pak-grid" role="img" xmlns="http://www.w3.org/2000/svg"></svg>`+
     `<div class="pak-player" id="pak-player"><div class="pak-prow"><button type="button" class="btn sm pak-play" data-act="play" aria-pressed="false" data-tipx="${esc(tx('Play the tries row by row at the chosen speed; press again to pause. Once every try is shown, the button plays them again from the first.','按选定的速度一行一行放出尝试；再按一次暂停。全部放完后，这个按钮从第 1 次尝试重新播放。'))}"></button>`+
      `<span class="pak-spd" tabindex="0" data-tipx="${esc(tx(`Slow shows ${SPEEDS.slow} tries a second, Medium ${SPEEDS.med} and Fast ${SPEEDS.fast}.`,`慢速每秒显示 ${SPEEDS.slow} 次尝试，中速 ${SPEEDS.med} 次，快速 ${SPEEDS.fast} 次。`))}"><span class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="spd">${Object.keys(SPEEDS).map(v=>`<button type="button" data-v="${v}" aria-pressed="${spd===v}">${esc(L(SPDL[v]))}</button>`).join('')}</span></span></div>`+
      `<div class="pak-scrub"><button type="button" class="btn ghost sm" data-act="r-" aria-label="${esc(tx('One try back','后退一次尝试'))}" data-tipx="${esc(tx('Shows one try fewer, and pauses.','少显示一次尝试，并暂停。'))}">◂</button><input type="range" id="pak-rows" min="0" max="10" step="1" value="0" aria-label="${esc(tx('Tries shown','显示到第几次尝试'))}"><button type="button" class="btn ghost sm" data-act="r+" aria-label="${esc(tx('One try forward','前进一次尝试'))}" data-tipx="${esc(tx('Shows one more try, and pauses.','多显示一次尝试，并暂停。'))}">▸</button><output class="mono" id="pak-rows-out"></output></div></div></div>`+
    `<div class="pak-chartbox" id="pak-chartbox"><div class="pak-cap"><span id="pak-chart-cap"></span></div><svg class="pak-chart" id="pak-chart" role="img" xmlns="http://www.w3.org/2000/svg"></svg><p class="pak-chart-note small muted" id="pak-chart-note"></p></div>`+
   `</div>`+
   `<div class="metrics pak-metrics" id="pak-metrics"></div>`+
   `<details class="math" id="pak-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="pak-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="pak-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="pak-lock" hidden></p>`+
   `<div id="pak-ctl-lv-wrap">`+A.ui.range({id:'pak-ctl-lv',key:'lv',label:tx('How uneven the tasks are','任务之间的差别'),min:0,max:LVMAX,value:S.lv,out:L(LVN[S.lv]),note:tx('Each step right turns 10 tasks that pass 60% of the time into 6 that always pass and 4 that never do. Try each of those 10 tasks once, and 6 pass on average either way, so pass@1 stays at 60%.','每往右拖一格，就有 10 个“每次 60%”的任务变成 6 个稳过、4 个过不了。这 10 个任务各试一次，平均还是过 6 个，所以 pass@1 一直是 60%。')})+`</div>`+
   `<div id="pak-ctl-k-wrap">`+A.ui.range({id:'pak-ctl-k',key:'k',label:tx('k: tries the product allows','k：产品允许的尝试次数'),min:1,max:10,value:S.k,out:String(S.k),note:tx('The k in pass@k and pass^k: how many tries a coding product may make, or how many times a user asks the same thing.','pass@k 和 pass^k 里的 k：写代码的产品最多能试几次，或者一个用户把同一件事问几次。')})+`</div>`+
   `<div id="pak-ctl-n-wrap">`+A.ui.seg({id:'pak-ctl-n',key:'n',label:tx('n: tries per task in the eval','n：评测里每个任务试几次'),options:NS.map(v=>[String(v),String(v)]),value:String(S.n),note:tx('More tries make the estimates more precise; they do not change the true pass@k. With fewer tries than k there is no estimate, so the chart stops at k = n.','试得越多，估得越准，但不改变真实的 pass@k。试的次数比 k 少时没法估，所以图到 k = n 为止。')})+`</div>`+
   `<div id="pak-ctl-tau-wrap">`+A.ui.seg({id:'pak-ctl-tau',key:'tau',label:tx('τ‑bench’s points','τ‑bench 的点'),options:[['0',tx('hide','不显示')],['1',tx('show','显示')]],value:String(S.tau),note:tx('Shows τ‑bench’s pass^1 to pass^4 (retail, tool-calling, '+GPT+') as diamonds.','用菱形显示 τ‑bench 的 pass^1 到 pass^4（retail，tool-calling，'+GPT+'）。')})+`</div>`+
   `<div id="pak-ctl-seed"><div class="ctl"><span class="lbl">${tx('The tries','尝试结果')} <output class="mono" id="pak-seed-out"></output></span><button type="button" class="btn ghost sm" data-act="reseed">${tx('New tries','重新试一遍')}</button>`+
    `<div class="ctl-note">${tx('The tries come from a seed, so the page shows the same ones every time; a new seed gives new tries with the same chances.','尝试结果来自一个种子，所以页面每次给出同样的结果；换一个种子，就是把同样的任务重新试一遍。')}</div></div></div>`+
   `<div class="pak-grp" id="pak-ctl-q"><div class="pak-gh">${tx('Assumed values','假设值')}</div><div class="ctl"><span class="lbl">${tx('Chance of a “sometimes” task','时好时坏的任务每次的把握')} <output class="mono">60%</output></span><div class="ctl-note">${tx('This lab assumes 60%: it makes pass@1 60% at every mix. Real tasks spread over every chance from 0 to 100%; what matters here is only that some tasks sit in between.','本 Lab 假设它是 60%：有了它，不管怎么分，pass@1 都是 60%。真实任务的把握分布在 0 到 100% 之间；这里要紧的只是有一部分任务处在中间。')}</div></div></div>`+
   `<div id="pak-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`from math import comb

def pass_at_k(n, c, k):
    """Chance that k tries picked from n include a pass."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def pass_hat_k(n, c, k):
    """Chance that all k tries picked from n pass."""
    return comb(c, k) / comb(n, k)   # comb(c, k) = 0 if c < k

n, k = 10, 5                         # 10 tries per task in the eval
passes = [10, 10, 10, 7, 6, 6, 4, 0, 0, 0]   # c for 10 tasks
avg = lambda f: sum(f(n, c, k) for c in passes) / len(passes)
print(f"pass@1 = {avg(lambda n, c, k: c / n):.3f}")
print(f"pass@{k} = {avg(pass_at_k):.3f}")
print(f"pass^{k} = {avg(pass_hat_k):.3f}")
# pass@1 = 0.530
# pass@5 = 0.698
# pass^5 = 0.313`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="pak-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="pak-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Both estimators from n tries and c passes per task, then the average over the tasks.','两个估计：每个任务从 n 次尝试、c 次通过算起，再对任务取平均。')}</p><pre class="code" id="pak-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="pak-prod">`+
   tx(`<li>openai/human-eval, ${cite(SRC.he,'<code>human_eval/evaluation.py</code>')}: <code>estimate_pass_at_k(num_samples, num_correct, k)</code> computes ${M('1 - \\frac{C(n - c, k)}{C(n, k)}')} as <code>1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))</code>, which avoids huge binomial coefficients; it returns 1 when ${M('n - c < k')}. <code>evaluate_functional_correctness</code> reports pass@k only for the k that every task has enough samples for.</li>`+
      `<li>sierra-research/tau-bench, ${cite(SRC.tau,'README')}: the leaderboard reports Pass^1 to Pass^4 for each model and strategy, in the airline and retail domains. The README now points to a newer version of the benchmark with fixed tasks.</li>`,
      `<li>openai/human-eval 的 ${cite(SRC.he,'<code>human_eval/evaluation.py</code>')}：<code>estimate_pass_at_k(num_samples, num_correct, k)</code> 算 ${M('1 - \\frac{C(n - c, k)}{C(n, k)}')}，写成 <code>1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))</code>，避免算很大的组合数；${M('n - c < k')} 时直接返回 1。<code>evaluate_functional_correctness</code> 只报告每个任务样本都够的那些 k。</li>`+
      `<li>sierra-research/tau-bench 的 ${cite(SRC.tau,'README')}：排行榜给出每个模型和策略在航空（airline）和网店（retail）两类任务上的 Pass^1 到 Pass^4。README 现在指向修过任务的新版本。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.codex,'Evaluating Large Language Models Trained on Code')}<span>${tx('Chen et al., 2021, arXiv 2107.03374: the paper behind HumanEval and the pass@k estimator from n samples per task','Chen 等，2021，arXiv 2107.03374：HumanEval 的论文，pass@k 的估计方法（每个任务采样 n 次）也出自这里')}</span></li>`+
   `<li>${cite(SRC.taup,'τ‑bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains')}<span>${tx('Yao et al., 2024, arXiv 2406.12045: the paper behind τ‑bench, whose leaderboard reports pass^1 to pass^4; the numbers on this page come from the repository’s README','Yao 等，2024，arXiv 2406.12045：τ‑bench 的论文，它的排行榜报告 pass^1 到 pass^4；页面上的数取自仓库的 README')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="pak-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="pak-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="pak-app m2" id="pak-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="pak-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see which number it should look at.','选一个场景，看看它该看哪个指标。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="pak-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Get 95% of coding tasks done with the fewest calls','用最少的调用，做成 95% 的写代码任务')}</h3>`+
   `<p>${tx('Two agents both pass 70% of tasks on the first try. Agent A has a 70% chance on every task. Agent B passes 70% of the tasks every time and never passes the rest.','两个 agent 第一次都做对 70% 的任务。A 对每个任务都有七成把握；B 有七成的任务次次都过，其余的怎么都过不了。')}</p>`+
   `<p>${tx('Your product writes code. Unit tests check every try, so a failed try can be retried, and the agent stops at the first pass. At least 95% of the tasks must get done in the end. Pick the agent and the most tries k, using as few model calls as possible:','你们的产品写代码。单元测试会检查每一次尝试，没通过可以重试，一通过就停。最终至少要做成 95% 的任务。选一个 agent 和最多尝试次数 k，在达标的前提下，让调用模型的次数尽量少：')}</p>`+
   A.ui.seg({id:'pak-ic-ag',key:'chAg',label:tx('Agent','agent'),options:CH.ags.map(a=>[a,a]),value:chAg,inline:true})+
   A.ui.seg({id:'pak-ic-chk',key:'chK',label:tx('Most tries k','最多尝试次数 k'),options:CH.ks.map(k=>[String(k),String(k)]),value:String(chK),inline:true})+
   `<div class="goal mono pak-goal" id="pak-ch-g1"><span>${tx('Tasks done','做成的任务')}</span><span><b id="pak-ch-d"></b>${tx(' (goal: at least 95%)','（目标：至少 95%）')}</span></div>`+
   `<div class="goal mono pak-goal" id="pak-ch-g2"><span>${tx('Calls per task, on average','平均每个任务调用')}</span><span><b id="pak-ch-c"></b></span></div>`+
   submit+`<div id="pak-ch-res" class="pak-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided') return '';
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},uneven:{en:'Drag the slider to both ends to continue.',zh:'把滑块拖到两头各看一次，才能继续。'},
    estimate:{en:'Pick “Use all 10 tries” to continue.',zh:'选“用全部 10 次估”才能继续。'},every:{en:'Drag the slider to continue.',zh:'拖动滑块才能继续。'},decide:{en:'Pick the support product to continue.',zh:'选客服产品才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#pak-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#pak-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.gate||s.gate());
    $$('#pak-ic-est [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===est)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=estTried.has(v)||v===est?fP(estVal(v)):'';});
    $$('#pak-ic-prod [data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===prod)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=prodTried.has(v)||v===prod?prodRes(v):'';});
    $$('.inline-ctl .ctl-note, .ctl .ctl-note',rail).forEach(n=>{const i=n.parentElement.querySelector('input[data-key="lv"]'); if(i) n.textContent=mixText(S.lv);});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
        if(!box._seen&&done()){box._seen=true; if(wantObs) revealSoon();}}}
    const dis=isNextDisabled(); $$('[data-nav="next"]',rail).forEach(nb=>{if(nb.disabled!==dis) A.ui.setNext?A.ui.setNext(rail,dis,nextWhy()):(nb.disabled=dis);});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#pak-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+(done()?scenHTML(cur):waitLine()):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again&&userAct) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chAg,chK), d=$('#pak-ch-d',rail), c=$('#pak-ch-c',rail); if(!d) return;
    d.textContent=fPc(e.done); c.textContent=tx(`${fC(e.calls)} calls`,`${fC(e.calls)} 次`);
    const g1=$('#pak-ch-g1',rail); g1.classList.toggle('met',e.ok); g1.classList.toggle('pak-over',!e.ok);
    $$('.seg-ctl[data-key="chAg"]',rail).forEach(g=>A.ui.setPressed(g,chAg)); $$('.seg-ctl[data-key="chK"]',rail).forEach(g=>A.ui.setPressed(g,String(chK)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#pak-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const viewNow=()=>mode==='guided'?STEPS[step].view:(mode==='challenge'?'ch':'grid');
const tauOn=()=>mode==='guided'?(isP('every')||(isP('remember')&&remView==='every')):(mode==='sandbox'&&S.tau===1);
/* the k the stage reads at: the product's k in part 5, the part's k elsewhere */
const kNow=()=>isP('decide')?prodOf(prod).k:S.k;
const focusM=()=>isP('every')?'hat':(isP('decide')?prodOf(prod).m:(isP('estimate')||isP('uneven')||isP('mission')?'at':(mode==='sandbox'&&(S.tau===1||(PRESETS[presetNow()]||{}).id==='support')?'hat':'both')));
const SVGNS='http://www.w3.org/2000/svg';
function gridSVG(){
  const svg=$('#pak-grid',app), box=$('#pak-gridbox',app), v=viewNow();
  const W=Math.max(260,Math.round(box.clientWidth||520)), r=R(), n=S.n, first=v==='first';
  const rowsN=first?1:n, LW=A.lang()==='zh'?56:52,   /* zh 56: “第 10 次” crossed the picture's left edge by 6 px at 44 (H7, 2026-10-09) */ cw=(W-LW-4)/NT, sq=Math.max(3.5,Math.min(cw-1.6,11)), rh=Math.max(sq+1.6,Math.min(cw,12.5));
  const top=4, gh=rowsN*rh, bot=first?6:43, H=Math.round(top+gh+bot);   // room for a second line of group names, so the height never jumps
  const cols=first?SHUF.map(i=>r.tasks[i]):r.sorted;
  let s='';
  const lab=j=>tx(j===0?'try 1':`${j+1}`,j===0?'第 1 次':`第 ${j+1} 次`);
  for(let j=0;j<rowsN;j++){
    if(j===0||j===rowsN-1||(rowsN>=10&&(j+1)%5===0)) s+=`<text class="pak-rl" x="${LW-6}" y="${(top+j*rh+rh/2+3.5).toFixed(1)}" text-anchor="end">${esc(j===0?tx('try 1','第 1 次'):tx(`try ${j+1}`,`第 ${j+1} 次`))}</text>`;
    cols.forEach((t,ci)=>{const x=LW+ci*cw+(cw-sq)/2, y=top+j*rh+(rh-sq)/2, vis=first?j<1:j<shown;
      if(!vis){s+=`<rect class="pak-sq wait" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${sq.toFixed(1)}" height="${sq.toFixed(1)}"/>`; return;}
      s+=`<rect class="pak-sq ${t.t[j]?'pass':'fail'}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${sq.toFixed(1)}" height="${sq.toFixed(1)}"/>`;});
  }
  if(!first){
    /* the kinds, named under their columns (R5.9), each with a tooltip */
    const sp=r.sp, groups=[['a',sp.a],['s',sp.s],['z',sp.z]].filter(g=>g[1]>0); let c0=0;
    const gname=(k,cnt)=>S.lv===0?tx(`all ${cnt}: 60% each`,`${cnt} 个都是 60%`):(L(KN[k])+tx(` ${cnt}`,` ${cnt} 个`));
    const gw=(k,cnt)=>Math.max(30,(gname(k,cnt).length*(A.lang()==='zh'?11:6.4))+6);
    /* label centres, pushed apart left to right and kept inside the picture; the middle label drops a line if they still touch */
    const pos=[]; {let cc=0; groups.forEach(([k,cnt])=>{const w=gw(k,cnt), m=LW+(cc+cnt/2)*cw; pos.push({w,x:Math.max(4+w/2,Math.min(W-4-w/2,m)),dy:0}); cc+=cnt;});
      let clash=false; for(let i=1;i<pos.length;i++){const need=pos[i-1].x+pos[i-1].w/2+8+pos[i].w/2; if(pos[i].x<need){pos[i].x=need; if(pos[i].x>W-4-pos[i].w/2) clash=true;}}
      if(clash&&pos.length===3){pos[1].dy=13; let cc2=0; groups.forEach(([k,cnt],i)=>{const m=LW+(cc2+cnt/2)*cw; pos[i].x=Math.max(4+pos[i].w/2,Math.min(W-4-pos[i].w/2,m)); cc2+=cnt;});}
      else if(clash){for(let i=pos.length-2;i>=0;i--){const lim=pos[i+1].x-pos[i+1].w/2-8-pos[i].w/2; if(pos[i].x>lim) pos[i].x=lim;}}}
    groups.forEach(([k,cnt],gi)=>{const x0=LW+c0*cw+1, x1=LW+(c0+cnt)*cw-1, yb=top+gh+5;
      const name=gname(k,cnt);
      const tip=k==='a'?tx(`${cnt} tasks always pass: every one of their ${n} tries passed.`,`${cnt} 个任务稳过：${n} 次尝试次次通过。`):(k==='z'?tx(`${cnt} tasks never pass: none of their ${n} tries passed.`,`${cnt} 个任务过不了：${n} 次尝试一次都没过。`)
        :tx(`${cnt} tasks pass 60% of the time (set by this lab); in these ${n} tries they passed ${Math.min(...r.tasks.filter(x=>x.kind==='s').map(x=>x.c))} to ${Math.max(...r.tasks.filter(x=>x.kind==='s').map(x=>x.c))} times each.`,`${cnt} 个任务每次有 60% 的把握（这个 Lab 设定的）；这 ${n} 次里，每个通过了 ${Math.min(...r.tasks.filter(x=>x.kind==='s').map(x=>x.c))} 到 ${Math.max(...r.tasks.filter(x=>x.kind==='s').map(x=>x.c))} 次。`));
      const tx0=pos[gi].x, ty=yb+18+pos[gi].dy;
      s+=`<g class="pak-gb" data-tipx="${esc(tip)}"><path class="pak-br" d="M${x0.toFixed(1)} ${yb} v4 H${x1.toFixed(1)} v-4"/><rect class="pak-hit" x="${x0.toFixed(1)}" y="${top}" width="${Math.max(2,x1-x0).toFixed(1)}" height="${(gh+bot-2).toFixed(1)}"/><text class="pak-gl" x="${tx0.toFixed(1)}" y="${ty}" text-anchor="middle">${esc(name)}</text></g>`;
      c0+=cnt;});
  }
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width',W); svg.setAttribute('height',H); svg.innerHTML=s;
  const aria=first?tx(`${NT} tasks, one try each: ${r.first1} passed.`,`${NT} 个任务，每个试 1 次：${r.first1} 个通过。`):tx(`${NT} tasks, ${n} tries each: ${mixText(S.lv)}.`,`${NT} 个任务，每个试 ${n} 次：${mixText(S.lv)}。`);
  svg.setAttribute('aria-label',aria);
}
/* the chart against k: pass@k rising, pass^k falling, the dashed "every task equally hard" lines, τ-bench's diamonds, the k line */
function chartSVG(){
  const svg=$('#pak-chart',app), box=$('#pak-chartbox',app), v=viewNow();
  const W=Math.max(260,Math.round(box.clientWidth||520)), H=innerWidth<=760?176:196, zh=A.lang()==='zh';
  const PL=34, PR=zh?78:84, PT=v==='ch'?24:10, PB=24, kmax=v==='ch'?8:S.n, X=k=>PL+(W-PL-PR)*(kmax===1?0:(k-1)/(kmax-1)), Y=p=>PT+(H-PT-PB)*(1-p);
  let s='';
  [0,0.25,0.5,0.75,1].forEach(p=>{s+=`<line class="pak-gridl" x1="${PL}" x2="${W-PR}" y1="${Y(p).toFixed(1)}" y2="${Y(p).toFixed(1)}"/><text class="pak-ax" x="${PL-4}" y="${(Y(p)+3.5).toFixed(1)}" text-anchor="end">${Math.round(p*100)}%</text>`;});
  const kt=[...Array(kmax).keys()].map(i=>i+1).filter(k=>kmax<=10||k===1||k%5===0);
  kt.forEach(k=>{s+=`<text class="pak-ax" x="${X(k).toFixed(1)}" y="${H-8}" text-anchor="middle">${k}</text>`;});
  s+=`<text class="pak-ax" x="${W-PR+16}" y="${H-8}" text-anchor="start">k</text>`;
  const path=(f,ks)=>ks.map((k,i)=>(i?'L':'M')+X(k).toFixed(1)+' '+Y(f(k)).toFixed(1)).join('');
  const ends=[];
  if(v==='ch'){
    const ks=CH.ks.filter(k=>k<=chK);
    s+=`<line class="pak-goalln" x1="${PL}" x2="${W-PR}" y1="${Y(CH.goal).toFixed(1)}" y2="${Y(CH.goal).toFixed(1)}"/>`;
    ends.push({y:Y(CH.goal),t:tx('goal 95%','目标 95%'),c:'g'});
    s+=`<path class="pak-ln at" d="${path(k=>chEval(chAg,k).done,ks)}"/>`+ks.map(k=>`<circle class="pak-dot" cx="${X(k).toFixed(1)}" cy="${Y(chEval(chAg,k).done).toFixed(1)}" r="3.2"/>`).join('');
    const dn=chEval(chAg,chK).done, yd=Y(dn), yg=Y(CH.goal), near=Math.abs(yd-yg)<14;
    const yl=near?(dn>=CH.goal?yd-7:yd+14):yd+3.5;
    s+=`<text class="pak-vl pak-chv" x="${(X(chK)+8).toFixed(1)}" y="${yl.toFixed(1)}">${esc(tx(`${chAg}: ${fPc(dn)}`,`${chAg}：${fPc(dn)}`))}</text>`;
    s+=`<line class="pak-kl" x1="${X(chK).toFixed(1)}" x2="${X(chK).toFixed(1)}" y1="${PT}" y2="${H-PB}"/>`;
  } else {
    const r=R(), ks=r.ks, k0=Math.min(kNow(),S.n), fm=focusM();
    if(v==='first'){
      s+=`<circle class="pak-dot" cx="${X(1).toFixed(1)}" cy="${Y(r.p1).toFixed(1)}" r="3.6"/><text class="pak-vl" x="${(X(1)+7).toFixed(1)}" y="${(Y(r.p1)-6).toFixed(1)}">${fP(r.p1)}</text>`;
      s+=`<line class="pak-kl" x1="${X(5).toFixed(1)}" x2="${X(5).toFixed(1)}" y1="${PT}" y2="${H-PB}"/><text class="pak-q" x="${X(5).toFixed(1)}" y="${(Y(0.8)).toFixed(1)}" text-anchor="middle">?</text>`;
    } else if(!done()){
      s+=`<text class="pak-wait" x="${((PL+W-PR)/2).toFixed(1)}" y="${(Y(0.5)).toFixed(1)}" text-anchor="middle">${esc(tx(`drawn once all ${S.n} tries are in`,`试完 ${S.n} 次后画出`))}</text>`;
    } else {
      const showAt=fm!=='hat'||!isP('every'), showHat=!(mode==='guided'&&step<SIX('every'));
      if(showAt){s+=`<path class="pak-ln ref" d="${path(k=>r.nAt[k],ks)}"/>`; ends.push({y:Y(r.nAt[ks[ks.length-1]]),t:tx('all equal','一样难时'),c:'r'});}
      if(showHat){s+=`<path class="pak-ln ref" d="${path(k=>r.nHat[k],ks)}"/>`; ends.push({y:Y(r.nHat[ks[ks.length-1]]),t:tx('all equal','一样难时'),c:'r'});}
      if(showAt){s+=`<path class="pak-ln at${fm==='hat'?' dim':''}" d="${path(k=>r.at[k],ks)}"/>`+ks.map(k=>`<circle class="pak-dot${fm==='hat'?' dim':''}" cx="${X(k).toFixed(1)}" cy="${Y(r.at[k]).toFixed(1)}" r="2.8"/>`).join('');
        ends.push({y:Y(r.at[ks[ks.length-1]]),t:'pass@k',c:'a'});}
      if(showHat){s+=`<path class="pak-ln hat${fm==='at'?' dim':''}" d="${path(k=>r.hat[k],ks)}"/>`+ks.map(k=>`<rect class="pak-dot sq${fm==='at'?' dim':''}" x="${(X(k)-2.6).toFixed(1)}" y="${(Y(r.hat[k])-2.6).toFixed(1)}" width="5.2" height="5.2"/>`).join('');
        ends.push({y:Y(r.hat[ks[ks.length-1]]),t:'pass^k',c:'h'});}
      if(tauOn()){TAU.forEach((t,i)=>{const k=i+1; if(k>kmax) return; const x=X(k), y=Y(t); s+=`<path class="pak-tau" d="M${x.toFixed(1)} ${(y-4.5).toFixed(1)} l4.5 4.5 l-4.5 4.5 l-4.5 -4.5z"/>`;});
        s+=`<text class="pak-taul" x="${(X(1)+8).toFixed(1)}" y="${(Y(TAU[0])-9).toFixed(1)}">τ‑bench</text>`;}
      /* the k the product uses: a vertical line with the values read at it */
      if(k0>=1&&k0<=kmax){s+=`<line class="pak-kl" x1="${X(k0).toFixed(1)}" x2="${X(k0).toFixed(1)}" y1="${PT}" y2="${H-PB}"/>`;
        const right=X(k0)<(PL+W-PR)/2+40;
        const lab=(p,txt)=>`<text class="pak-vl" x="${(X(k0)+(right?6:-6)).toFixed(1)}" y="${(Y(p)+(p>0.9?12:-5)).toFixed(1)}" text-anchor="${right?'start':'end'}">${esc(txt)}</text>`;
        const la=showAt&&fm!=='hat', lh=showHat&&fm!=='at';
        if(la) s+=lab(r.at[k0],fP(r.at[k0]));
        /* the two values meet (every task always or never passes): one label for both */
        if(lh&&!(la&&Math.abs(Y(r.at[k0])-Y(r.hat[k0]))<13)) s+=lab(r.hat[k0],fP(r.hat[k0]));
        else if(lh&&la&&fP(r.at[k0])!==fP(r.hat[k0])) s+=`<text class="pak-vl" x="${(X(k0)+(right?6:-6)).toFixed(1)}" y="${(Y(r.hat[k0])+12).toFixed(1)}" text-anchor="${right?'start':'end'}">${esc(fP(r.hat[k0]))}</text>`;}
    }
  }
  /* line names at their ends, pushed apart so they never overlap */
  ends.sort((a,b)=>a.y-b.y); const gap=12; for(let i=1;i<ends.length;i++) if(ends[i].y<ends[i-1].y+gap) ends[i].y=ends[i-1].y+gap;
  const over=ends.length?ends[ends.length-1].y-(H-PB-4):0; if(over>0) ends.forEach(e=>{e.y-=over;});
  if(ends.length&&ends[0].y<PT+2){const d=PT+2-ends[0].y; ends.forEach(e=>{e.y+=d;});}
  ends.forEach(e=>{s+=`<text class="pak-end ${e.c}" x="${(e.x!=null?e.x+8:W-PR+6).toFixed(1)}" y="${(e.y+3.5).toFixed(1)}">${esc(e.t)}</text>`;});
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width',W); svg.setAttribute('height',H); svg.innerHTML=s;
  svg.setAttribute('aria-label',chartAria());
}
function chartAria(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chAg,chK); return tx(`${chSet(chAg,chK)}: ${fP(e.done)} of the tasks done; goal 95%.`,`${chSet(chAg,chK)}：做成 ${fP(e.done)} 的任务；目标 95%。`);}
  const r=R(); if(v==='first') return tx('pass@1 is 60%; pass@5 is hidden until you answer.','pass@1 是 60%；pass@5 在你作答前先隐藏。');
  if(!done()) return tx('The chart is drawn once the tries are in.','试完之后再画图。');
  const k=Math.min(kNow(),S.n); return tx(`At k = ${k}: pass@k ${fP(r.at[k])}, pass^k ${fP(r.hat[k])}.`,`k = ${k} 时：pass@k 是 ${fP(r.at[k])}，pass^k 是 ${fP(r.hat[k])}。`);
}
function drawHead(){
  const v=viewNow(), t=$('#pak-st-title',app), sub=$('#pak-st-sub',app), r=R();
  if(v==='ch'){t.textContent=tx('The challenge: how many tasks get done within k tries','挑战题：最多试几次，能做成多少任务'); sub.textContent=tx(`${chSet(chAg,chK)}; ${chAg==='A'?'A has a 70% chance on every task':'B always passes 70% of the tasks and never the other 30%'}`,`${chSet(chAg,chK)}；${chAg==='A'?'A 对每个任务都有七成把握':'B 有七成任务次次都过，三成怎么都过不了'}`);}
  else if(v==='first'){t.textContent=tx(`${NT} coding tasks, one try each`,`${NT} 个写代码的任务，每个先试 1 次`); sub.textContent=tx(`${r.first1} of ${NT} passed on the first try: pass@1 = 60%`,`${NT} 个里有 ${r.first1} 个第一次就通过：pass@1 = 60%`);}
  else {t.textContent=tx(`${NT} tasks, ${S.n} tries each`,`${NT} 个任务，每个试 ${S.n} 次`); sub.textContent=mixText(S.lv);}
  $('#pak-grid-cap',app).textContent=v==='first'?tx('Each column is a task; the row is its first try.','每一列是一个任务，这一行是它的第 1 次尝试。'):tx('Each column is a task, each row a try.','每一列是一个任务，每一行是一次尝试。');
  $('#pak-grid-leg',app).innerHTML=`<span tabindex="0" data-tipx="${esc(tx('A filled square: this try passed the task’s unit tests.','实心方块：这次尝试通过了这个任务的单元测试。'))}"><i class="pak-sw pass"></i>${esc(tx('passed','通过'))}</span><span tabindex="0" data-tipx="${esc(tx('An empty square: this try failed.','空心方块：这次尝试没通过。'))}"><i class="pak-sw fail"></i>${esc(tx('failed','没通过'))}</span>`;
  /* which curves the chart shows: pass@k only before part 4, pass^k only in part 4, both after (as chartSVG draws them) */
  const onlyAt=v==='first'||(mode==='guided'&&step<SIX('every')), onlyHat=isP('every');
  $('#pak-chart-cap',app).textContent=v==='ch'?tx('Share of tasks done within k tries','最多试 k 次时，做成的任务占多少')
    :(onlyAt?tx(`Share of the ${NT} tasks done within k tries`,`最多试 k 次时，${NT} 个任务里有多少能做成`)
      :(onlyHat?tx(`Share of the ${NT} tasks with all k tries passing`,`试 k 次时，${NT} 个任务里有多少次次都过`)
        :tx(`Share of the ${NT} tasks with at least one pass, and with every try passing, in k tries`,`试 k 次时，${NT} 个任务里有多少至少过一次、有多少次次都过`)));
  const dashTip=onlyAt?tx('Dashed: what pass@k would be if every task’s chance equaled pass@1.','虚线：假设每个任务的把握都等于 pass@1 时，pass@k 会是多少。')
    :(onlyHat?tx('Dashed: what pass^k would be if every task’s chance equaled pass@1.','虚线：假设每个任务的把握都等于 pass@1 时，pass^k 会是多少。')
      :tx('Dashed: what pass@k and pass^k would be if every task’s chance equaled pass@1.','虚线：假设每个任务的把握都等于 pass@1 时，pass@k 和 pass^k 会是多少。'));
  const note=$('#pak-chart-note',app);
  note.innerHTML=v==='ch'?esc(tx('Dots: the share this agent gets done with at most 1, 2, … up to your k tries. Dashed line: the 95% goal.','圆点：最多试 1 次、2 次……一直到你选的 k 次时，这个 agent 各能做成多少任务。虚线：目标 95%。'))
    :(v==='first'?esc(tx('The question mark at k = 5 is part 1’s question.','k = 5 处的问号，就是第 1 部分的问题。'))
      :(isP('every')?'':`<span tabindex="0" data-tipx="${esc(TIP.at())}">${esc(tx('pass@k (circles): at least one of k tries passes.','pass@k（圆点）：k 次里至少一次通过。'))}</span>${SEP()}`)+(mode==='guided'&&step<SIX('every')?'':`<span tabindex="0" data-tipx="${esc(TIP.hat())}">${esc(tx('pass^k (squares): all k tries pass.','pass^k（方点）：k 次全部通过。'))}</span>${SEP()}`)+
        `<span tabindex="0" data-tipx="${esc(dashTip)}">${esc(tx('Dashed: if every task were equally hard.','虚线：如果每个任务一样难。'))}</span>`+
        (tauOn()?`${SEP()}<span tabindex="0" data-tipx="${esc(TIP.tau())}">${esc(tx(`Diamonds: τ‑bench, ${GPT}, retail.`,`菱形：τ‑bench，${GPT}，网店任务。`))}</span>`:'')+
        `${SEP()}<span tabindex="0" data-tipx="${esc(tx('The vertical dotted line marks the k the product uses; the values beside it are read there.','竖的点线标出产品用的 k，旁边的数就是在这里读出的值。'))}">${esc(tx('Dotted: the k in use.','点线：用到的 k。'))}</span>`);
  const st=$('#pak-status',app), s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chAg,chK); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达标'):tx('Not yet','还没达标'),tip:tx(`Green when at least 95% of the tasks get done. Now: ${fP(e.done)}.`,`做成的任务至少 95% 时变绿。现在是 ${fP(e.done)}。`)};}
  if(v==='first') return {cls:'neutral',txt:tx('Predict first','先预测'),tip:tx('pass@5 is hidden until you answer in part 1.','在第 1 部分作答前，先隐藏 pass@5。')};
  if(!done()) return {cls:'neutral',txt:tx('Trying…','正在试'),tip:tx('The tasks are being tried; the numbers show once every try is in.','任务正在试；所有尝试结束后再给出数字。')};
  const r=R(), k=Math.min(kNow(),S.n), fm=focusM();
  if(fm==='hat') return {cls:'neutral',txt:`pass^${k} ${fP(r.hat[k])}`,tip:tx(`All ${k} tries pass on ${fP(r.hat[k])} of the tasks (estimated from the tries).`,`按尝试结果估计，${fP(r.hat[k])} 的任务 ${k} 次全部通过。`)};
  return {cls:'neutral',txt:`pass@${k} ${fP(r.at[k])}`,tip:tx(`At least one of ${k} tries passes on ${fP(r.at[k])} of the tasks (estimated from the tries).`,`按尝试结果估计，${fP(r.at[k])} 的任务 ${k} 次里至少通过一次。`)};
}
/* metric cards: ▲▼ only after the learner's own change, from the values as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
function metricsDef(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chAg,chK);
    return [{k:'cd',label:tx('Tasks done','做成的任务'),v:fPc(e.done),sub:tx('goal: at least 95%','目标：至少 95%'),pts:true,tip:tx(`${chSet(chAg,chK)} gets ${fP(e.done)} of the tasks done.`,`${chSet(chAg,chK)}，能做成 ${fP(e.done)} 的任务。`)},
      {k:'cc',label:tx('Calls per task','每个任务的调用'),v:fC(e.calls),sub:tx('on average; stops at the first pass','平均值；一通过就停'),tip:tx(`On average each task costs ${fC(e.calls)} model calls: the agent stops at its first pass, or after ${chK} ${chK===1?'try':'tries'}.`,`平均每个任务调用模型 ${fC(e.calls)} 次：一通过就停，最多试 ${chK} 次。`)},
      {k:'ck',nd:true,label:tx('Most tries','最多尝试次数'),v:String(chK),sub:agName(chAg),tip:tx(`The agent tries a task at most ${chK} ${chK===1?'time':'times'}.`,`agent 对一个任务最多试 ${chK} 次。`)}];}
  const r=R(), k=Math.min(kNow(),S.n), wait=v!=='first'&&!done(), W=x=>wait?'…':x;
  const p1={k:'p1',label:tx('pass@1','pass@1'),v:fP(r.p1),sub:tx('a single try passes','一次尝试就通过'),pts:true,tip:TIP.p1()+SEP()+tx(`Here: ${fP(r.p1)} of all tries passed.`,`这里所有尝试中有 ${fP(r.p1)} 通过。`)};
  if(v==='first') return [p1,{k:'at',label:'pass@5',v:'?',sub:tx('predict first','先预测'),tip:tx('Hidden until you answer in part 1.','在第 1 部分作答前先隐藏。')}];
  const at={k:'at',label:`pass@${k}`,v:W(fP(r.at[k])),sub:tx(`at least one of ${k} tries passes`,`${k} 次里至少一次通过`),pts:true,tip:TIP.at()+SEP()+tx(`Here, estimated from the tries: ${fP(r.at[k])}; the true value is ${fP(r.tAt[k])}.`,`这里按尝试结果估出 ${fP(r.at[k])}；真实值是 ${fP(r.tAt[k])}。`)};
  const hat={k:'hat',label:`pass^${k}`,v:W(fP(r.hat[k])),sub:tx(`all ${k} tries pass`,`${k} 次全部通过`),pts:true,tip:TIP.hat()+SEP()+tx(`Here, estimated from the tries: ${fP(r.hat[k])}; the true value is ${fP(r.tHat[k])}.`,`这里按尝试结果估出 ${fP(r.hat[k])}；真实值是 ${fP(r.tHat[k])}。`)};
  const zero={k:'zero',nd:true,label:tx('Never passed','一次都没过'),v:W(tx(`${r.zero} of ${NT}`,`${r.zero} / ${NT}`)),sub:tx(`tasks that failed all ${S.n} tries`,`${S.n} 次全失败的任务`),tip:tx(`${r.zero} of the ${NT} tasks failed every one of their ${S.n} tries. No number of retries rescues them.`,`${NT} 个任务里有 ${r.zero} 个，${S.n} 次尝试全部失败。重试多少次都救不回它们。`)};
  if(mode==='guided'&&step<SIX('every')) return [p1,at,zero];
  if(isP('every')) return [p1,hat,{k:'tau',label:tx('τ‑bench pass^4','τ‑bench 的 pass^4'),v:fP(TAU[3]),sub:tx(`${GPT}, retail`,`${GPT}，网店任务`),tip:TIP.tau()}];
  return [p1,at,hat,zero];
}
function drawMetrics(){
  const host=$('#pak-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join(), fresh=dropB; dropB=false;
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip; el.classList.toggle('pak-wait',m.v==='…');
    const clr=()=>{clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';};
    if(fresh) shownPrev[m.k]=null;
    /* while the tries play, no badge; the value from before stays the base for the learner's change */
    if(m.v==='…'){clr(); return;}
    let dt=''; const prev=shownPrev[m.k], changed=prev!=null&&prev!==m.v;
    if(userAct&&!m.nd&&changed){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:1}:{})||'';}
    /* a badge only for a metric that this change of the learner's moved; every other badge goes (S25) */
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(userAct||changed||prev==null) clr();
    shownPrev[m.k]=m.v;});
}
/* Show the math: each line redoes from what it shows (R6.6); every term has a tooltip (R1.4) */
function mathHTML(){
  const v=viewNow(), t=T_;
  if(v==='ch'){const e=chEval(chAg,chK), f=fixd(1-CH.p,1);
    /* the j-th try (j = 0, 1, …) runs only when the j earlier ones failed: chance 0.3^j; every term is written out up to k = 4 */
    const term=j=>j===0?'1':(j===1?f:`${f}^{${j}}`);
    const series=chK<=4?[...Array(chK).keys()].map(term).join(' + '):[term(0),term(1),term(2),'\\ldots',term(chK-1)].join(' + ');
    /* k = 7, 8 round to 100.0%: write “1 − 0.3⁸ > 99.9%”, never “= >99.9%” */
    const dv=fPc(e.done), dRow=dv[0]==='>'?['',dv.slice(1).replace('%','\\%'),'>']:['',dv.replace('%','\\%')];
    const kT=tx(`k is the most tries allowed, ${t('chK',String(chK))} here.`,`k 是最多尝试次数，这里是 ${t('chK',String(chK))}。`);
    return chAg==='A'?MB(tx(`<b>Tasks done.</b> Agent A fails a task only if all k tries fail, and each try fails with chance ${f}. ${kT}`,`<b>做成的任务。</b>A 只有 k 次全都失败，才做不成一个任务。每次失败的概率是 ${f}。${kT}`),
        EQ([[tx('\\text{share done}','\\text{做成的比例}'),`1 - ${f}^k`],['',`1 - ${f}^{${chK}}`],dRow]))+
      MB(tx('<b>Calls per task.</b> The first try always runs. Each later try runs only when every earlier one failed, so the average number of calls adds up the chance that each try runs.','<b>每个任务的调用次数。</b>第 1 次一定会试。之后的每一次，只有前面全都失败才会试到。所以平均调用次数，是每一次被试到的概率加起来。'),
        EQ([[tx('\\text{calls}','\\text{调用次数}'),series],['',fC(e.calls)]]))
      :MB(tx(`<b>Tasks done and calls.</b> Agent B passes 70% of tasks on the first try. The other 30% never pass and use all k tries. ${kT}`,`<b>做成的任务和调用次数。</b>B 有七成任务第一次就过。另外三成永远过不了，k 次都会用完。${kT}`),
        EQ([[tx('\\text{share done}','\\text{做成的比例}'),fP(e.done).replace('%','\\%')],[tx('\\text{calls}','\\text{调用次数}'),'0.7 \\times 1 + 0.3 \\times k'],['',`0.7 \\times 1 + 0.3 \\times ${chK}`],['',fC(e.calls)]]));}
  const r=R(), n=S.n, k=Math.min(kNow(),n), ks=k, pc=x=>fP(x).replace('%','\\%');
  if(v==='first') return MB(tx('<b>pass@1</b> is the share of tasks whose first try passed.','<b>pass@1</b> 是第一次尝试就通过的任务占全部任务的比例。'),EQ([['\\text{pass@}1',`\\frac{${r.first1}}{${NT}}`],['',pc(r.first1/NT)]]));
  if(!done()) return `<p>${esc(tx('The tasks are still being tried.','任务还在试。'))}</p>`;
  const sc=r.tasks.filter(x=>x.kind==='s').map(x=>x.c), hist={}; sc.forEach(c=>{hist[c]=(hist[c]||0)+1;});
  const cs=Object.keys(hist).map(Number).sort((a,b)=>b-a), sp=r.sp;
  const per=(f,c)=>f(n,c,ks);
  const termsAt=[], termsHat=[];
  if(sp.a){termsAt.push(`${sp.a} \\times 1`); termsHat.push(`${sp.a} \\times 1`);}
  cs.forEach(c=>{termsAt.push(`${hist[c]} \\times ${fixd(per(pAt,c),4)}`); termsHat.push(`${hist[c]} \\times ${fixd(per(pHat,c),4)}`);});
  if(sp.z){termsAt.push(`${sp.z} \\times 0`); termsHat.push(`${sp.z} \\times 0`);}
  const sumAt=(sp.a+cs.reduce((a,c)=>a+hist[c]*Number(fixd(per(pAt,c),4)),0)), sumHat=(sp.a+cs.reduce((a,c)=>a+hist[c]*Number(fixd(per(pHat,c),4)),0));
  const FX_AT_R=FX_AT.split(' = ')[1], FX_HAT_R=FX_HAT.split(' = ')[1];
  /* the average over the tasks, two terms a line so the sum fits its box (a long sum cannot wrap inside one formula) */
  const avgRows=(name,terms,res)=>{const rows=[]; for(let i=0;i<terms.length;i+=2){const last=i+2>=terms.length, ch=terms.slice(i,i+2).join(' + ')+(last?')':'');
    rows.push(i===0?[name,`\\frac{1}{${NT}} \\times (${ch}`]:['',`+ ${ch}`,' ']);} rows.push(['',res]); return rows;};
  return MB(tx(`<b>The tries.</b> ${t('T',`T = ${NT}`)} tasks, each tried ${t('n','n = '+n)} times; ${t('k','k = '+ks)}. A task that always passes has c = n; one that never passes has c = 0.`,`<b>尝试结果。</b>任务数 ${t('T',`T = ${NT}`)}，每个任务试 ${t('n','n = '+n)} 次，${t('k','k = '+ks)}。稳过的任务 c = n，过不了的 c = 0。`)+
      `<p>${tx('pass@1 is all passes over all tries:','pass@1 是全部通过次数除以全部尝试次数：')}</p>`,
      EQ([['\\text{pass@}1',`\\frac{${Math.round(r.p1*NT*n)}}{${NT*n}}`],['',pc(r.p1)]]))+
    MB(tx(`<b>pass@${ks}.</b> For each task, the chance that at least one of k tries picked from its n passes; then the average over the tasks.`,`<b>pass@${ks}。</b>对每个任务，算从它的 n 次里挑 k 次、至少一次通过的概率，再对所有任务取平均。`),
      EQ([['\\text{pass@}k',FX_AT_R]])+`<p>${tx('Each task that sometimes passes, by its passes c:','时好时坏的任务，按各自的通过次数 c 来算：')}</p>`+
      EQ(cs.map(c=>[`c = ${c}${tx(':','\\text{ 时}')}\\quad 1 - \\frac{C(${n-c}, ${ks})}{C(${n}, ${ks})}`,fixd(per(pAt,c),4)]))+
      `<p>${tx(`Tasks that always pass count 1, tasks that never pass count 0. The average over the ${NT} tasks:`,`稳过的任务算 1，过不了的算 0。对 ${NT} 个任务取平均：`)}</p>`+
      EQ(avgRows(`\\text{pass@}${ks}`,termsAt,pc(sumAt/NT))))+
    MB(tx(`<b>pass^${ks}.</b> For each task, the chance that all k tries picked from its n pass; then the average over the tasks.`,`<b>pass^${ks}。</b>对每个任务，算从它的 n 次里挑 k 次、全部通过的概率，再对所有任务取平均。`),
      EQ([['\\text{pass}^k',FX_HAT_R]].concat(avgRows(`\\text{pass}^{${ks}}`,termsHat,pc(sumHat/NT)))))+
    MB(tx('<b>The dashed lines</b> take every task to be equally hard, with the chance pass@1.','<b>图里的虚线</b>假设每个任务一样难，每次通过的概率都是 pass@1。'),
      EQ([[`\\text{pass@}${ks}`,`1 - {(1 - ${fixd(r.p1,2)})}^{${ks}}`],['',pc(r.nAt[ks])],[`\\text{pass}^{${ks}}`,`${fixd(r.p1,2)}^{${ks}}`],['',pc(r.nHat[ks])]]))+
    MB(tx('<b>True values</b>, from the chances this lab set (1, 0.6 or 0):','<b>真实值</b>，按这个 Lab 设定的把握（1、0.6 或 0）算：'),
      `<p>pass@${ks} = <b>${fP(r.tAt[ks])}</b>${tx(', ','，')}pass^${ks} = <b>${fP(r.tHat[ks])}</b></p>`);
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  syncPlay();
  drawHead(); gridSVG(); chartSVG(); drawMetrics(); drawPlayer();
  const mb=$('#pak-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* a frame of the playback: the grid, the player, and (at the end) everything that waits for the tries */
function frame(){if(!app) return; if(done()){render(); return;} gridSVG(); drawPlayer(); const st=$('#pak-status',app), s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt;}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you choose only the agent and the most tries, in the task card. The two agents’ chances are fixed: 70% on every task for A; 70% of tasks always and the rest never for B.','挑战中只能在任务卡里选 agent 和最多尝试次数。两个 agent 的把握固定：A 每个任务都是 70%；B 七成任务稳过，其余过不了。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return tx(`Fixed in this part: ${NT} tasks, one try each. You can change the settings in Scenarios.`,`这一部分固定：${NT} 个任务，每个试 1 次。在应用场景里可以改设置。`);
  if(id==='uneven'||id==='every') return tx(`Fixed in this part: k = ${S.k}, ${S.n} tries per task, seed 1. The slider is in the card. You can change the rest in Scenarios.`,`这一部分固定：k = ${S.k}，每个任务试 ${S.n} 次，种子 1。滑块在卡片里。其余的在应用场景里可以改。`);
  return tx(`Fixed in this part: ${mixText(S.lv)}; k = ${kNow()}, ${S.n} tries per task, seed 1. You can change them in Scenarios.`,`这一部分固定：${mixText(S.lv)}；k = ${kNow()}，每个任务试 ${S.n} 次，种子 1。在应用场景里可以改。`);
}
function syncControls(){
  $$('input[data-key="lv"]',app).forEach(i=>{if(+i.value!==S.lv) i.value=S.lv; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=L(LVN[S.lv]); i.setAttribute('aria-valuetext',L(LVN[S.lv])+tx(': ','：')+mixText(S.lv));});
  $$('input[data-key="k"]',app).forEach(i=>{if(+i.value!==S.k) i.value=S.k; i.max=String(S.n); A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=String(S.k);});
  $$('.seg-ctl[data-key="n"]',app).forEach(g=>A.ui.setPressed(g,String(S.n)));
  $$('.seg-ctl[data-key="tau"]',app).forEach(g=>A.ui.setPressed(g,String(S.tau)));
  const so=$('#pak-seed-out',app); if(so) so.textContent=tx(`seed ${S.seed}`,`种子 ${S.seed}`);
  const g=mode==='guided', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  ['#pak-ctl-lv-wrap','#pak-ctl-k-wrap','#pak-ctl-n-wrap','#pak-ctl-tau-wrap','#pak-ctl-seed','#pak-ctl-reset-wrap'].forEach(id=>hide(id,g||mode==='challenge'));
  hide('#pak-ctl-q',mode==='challenge'||isP('mission'));
  const ln=$('#pak-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
  $('#pak-gridbox',app).hidden=mode==='challenge';
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; A.announce(chartAria());},400);}

/* =====================================================================
   playback (R3.9): the tries fill in row by row whenever the tasks change; play / pause, ◂ ▸, a slider and three speeds
   ===================================================================== */
const sigNow=()=>{const v=viewNow(); return v==='grid'?[S.lv,S.n,S.seed].join('|'):v;};
/* R3.20: the grids the reader has watched to the end (played or stepped to the last try), by their tasks, not by part or mode. A grid
   watched before (A → B → A, Back to a part, a scenario card, a chip) shows its tries at once; a grid stopped part-way does not count */
const watched=new Set();
const markWatched=()=>{if(viewNow()==='grid'&&shown>=S.n) watched.add(sigNow());};
function syncPlay(){
  const sig=sigNow(); if(sig===gridSig) return; gridSig=sig;
  stopPlay();
  if(viewNow()!=='grid'){shown=S.n; return;}
  /* R3.20: the same tasks as the last grid on screen (the next part, another mode, back from the challenge), or tasks whose grid was watched to the end, show their tries at once; Replay plays them again */
  if(A.reduceMotion()||sig===lastGrid||watched.has(sig)){lastGrid=sig; shown=S.n; return;}
  lastGrid=sig; shown=0; startPlay();
}
function stopPlay(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0;}
function startPlay(){
  stopPlay(); if(!app) return; if(shown>=S.n) shown=0; playing=true; let last=0, acc=shown;
  const tick=ts=>{if(!playing||!app) return; if(!last) last=ts; acc+=Math.min(100,ts-last)/1000*SPEEDS[spd]; last=ts;
    const k=Math.min(S.n,Math.floor(acc)); if(k!==shown){shown=k; if(shown>=S.n){stopPlay(); markWatched(); if(actPending){actPending=false; userAct=true; try{render();} finally{userAct=false;}} else render(); return;} frame();}
    raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick); drawPlayer();
}
function drawPlayer(){
  const p=$('#pak-player',app); if(!p) return; p.hidden=viewNow()!=='grid';
  $$('[data-act="play"]',p).forEach(b=>{b.textContent=playing?tx('❚❚ Pause','❚❚ 暂停'):(shown>=S.n?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放')); b.setAttribute('aria-pressed',String(playing));});
  $$('.seg-ctl[data-key="spd"]',p).forEach(g=>A.ui.setPressed(g,spd));
  const ri=$('#pak-rows',app); if(ri){if(+ri.max!==S.n) ri.max=String(S.n); if(+ri.value!==shown) ri.value=String(shown); A.ui.setRange&&A.ui.setRange(ri);}
  const ro=$('#pak-rows-out',app); if(ro) ro.textContent=tx(`${shown} of ${S.n} tries`,`第 ${shown} / ${S.n} 次`);
}
function setShown(v){stopPlay(); shown=Math.max(0,Math.min(S.n,v)); markWatched(); render();}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  if(!NS.includes(S.n)) S.n=10; S.lv=Math.max(0,Math.min(LVMAX,Math.round(S.lv))); S.k=Math.max(1,Math.min(S.n,Math.round(S.k))); S.tau=S.tau?1:0;
  if(isP('uneven')) seen.add(S.lv);
  if(isP('every')&&S.lv!==0) moved4=true;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;} if(playing&&shown<S.n) actPending=true;}
function setEst(id){est=id; estTried.add(id); render();}
function setProd(id){prod=id; prodTried.add(id); render();}
function setRem(k){if(!REMS.includes(k)) return; remView=k; dropBadges(); S={...BASE,lv:k==='uneven'?LVMAX:(k==='every'?bestTauLv():2),k:k==='every'?4:5}; renderRail(); if(innerWidth<=760) A.ui.reveal($('#pak-stage',app));}
function enterStep(i){
  wantObs=false; actPending=false; dropBadges(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...S,...s.enter()};
  const st=$('#pak-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#pak-rail .step.current',app)||$('#pak-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#pak-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#pak-insight',app)); else revealBox($('#pak-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; actPending=false; dropBadges();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chAg=CH.start.ag; chK=CH.start.k;}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#pak-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.lv,s.k,s.n,s.seed,s.tau].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==5||p.some(x=>!Number.isInteger(x))) return null;
  const [lv,k,n,seed,tau]=p;
  if(lv<0||lv>LVMAX||!NS.includes(n)||k<1||k>n||seed<1||seed>9999||(tau!==0&&tau!==1)) return null;
  return {lv,k,n,seed,tau};
}

/* =====================================================================
   interview drill: seeded, from the engine; every line redoes from what it shows
   ===================================================================== */
function makeDrill(seed){const rnd=mulberry(seed*104729+11), n=[6,8,10][Math.floor(rnd()*3)], k=2+Math.floor(rnd()*2);
  const mid=[1,2].map(()=>1+Math.floor(rnd()*(n-1))).sort((a,b)=>b-a), cs=[n,...mid,0];
  return {seed,n,k,cs,at:cs.map(c=>pAt(n,c,k)),hat:cs.map(c=>pHat(n,c,k))};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#pak-drill-q',app), a=$('#pak-drill-a',app), nm=['A','B','C','D'];
  qd.innerHTML=tx(`An eval tried four tasks A, B, C and D <b>${d.n}</b> times each. They passed <b>${d.cs.join(', ')}</b> times. Estimate pass@${d.k} and pass^${d.k} for this agent.`,
                  `一次评测把 A、B、C、D 四个任务各试了 <b>${d.n}</b> 次，分别通过了 <b>${d.cs.join('、')}</b> 次。估出这个 agent 的 pass@${d.k} 和 pass^${d.k}。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const C=(x,y)=>{if(y<0||y>x) return 0; let r=1; for(let i=1;i<=y;i++) r=r*(x-y+i)/i; return Math.round(r);};
  const cn=C(d.n,d.k), mA=mean(d.at), mH=mean(d.hat);
  const per=tx('\\text{ per task}','\\text{（每个任务）}'), avg=tx('\\text{average}','\\text{平均}');
  a.innerHTML=tx(`<p><b>pass@${d.k} = ${fP(mA)}, pass^${d.k} = ${fP(mH)}.</b></p>`,`<p><b>pass@${d.k} = ${fP(mA)}，pass^${d.k} = ${fP(mH)}。</b></p>`)+
    `<p>${tx(`The ways to pick ${d.k} of the ${d.n} tries:`,`从 ${d.n} 次里挑 ${d.k} 次的挑法数：`)}</p>`+EQ([[`C(${d.n}, ${d.k})`,String(cn)]])+
    `<p>${tx(`pass@${d.k} per task, for tasks ${nm.join(', ')}, then their average:`,`A、B、C、D 每个任务的 pass@${d.k}，再取平均：`)}</p>`+
    EQ([[`\\text{pass@}${d.k}${per}`,`1 - \\frac{C(n - c, ${d.k})}{${cn}}`]].concat(d.cs.map((c,i)=>[`\\text{${nm[i]}}\\quad 1 - \\frac{${C(d.n-c,d.k)}}{${cn}}`,fixd(d.at[i],4)]),[[avg,fixd(mA,4)]]))+
    `<p>${tx(`pass^${d.k} per task, then their average:`,`每个任务的 pass^${d.k}，再取平均：`)}</p>`+
    EQ([[`\\text{pass}^{${d.k}}${per}`,`\\frac{C(c, ${d.k})}{${cn}}`]].concat(d.cs.map((c,i)=>[`\\text{${nm[i]}}\\quad \\frac{${C(c,d.k)}}{${cn}}`,fixd(d.hat[i],4)]),[[avg,fixd(mH,4)]]));
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={lv:{en:'How uneven the tasks are',zh:'任务之间的差别'},k:{en:'k',zh:'k'},n:{en:'n: tries per task',zh:'n：每个任务试几次'},chK:{en:'Most tries k',zh:'最多尝试次数 k'}};
const TERM={
 T:()=>`<div>${esc(tx(`T = ${NT}: the number of tasks in the eval. Every estimate is an average over them.`,`T = ${NT}：评测里的任务数。每个估计都是对它们取平均。`))}</div>`,
 n:()=>`<div>${esc(tx(`n = ${S.n}: how many times the eval tried each task.`,`n = ${S.n}：评测里每个任务试了几次。`))}</div>`,
 k:()=>`<div>${esc(tx(`k = ${Math.min(kNow(),S.n)}: the tries the product allows (pass@k), or the times a user asks the same thing (pass^k).`,`k = ${Math.min(kNow(),S.n)}：产品允许试几次（pass@k），或者用户把同一件事问几次（pass^k）。`))}</div>`,
 chK:()=>`<div>${esc(tx(`k = ${chK}: the most tries you picked in the card.`,`k = ${chK}：你在卡片里选的最多尝试次数。`))}</div>`
};
const termTipHTML=k=>TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return k==='chK'?'':tx('Fixed during the challenge.','挑战中固定不变。');
  if(k==='chK') return tx('This is the challenge’s k.','这是挑战题里的 k。');
  if(mode!=='guided') return k==='T'?tx(`Fixed: the eval has ${NT} tasks.`,`固定不变：评测有 ${NT} 个任务。`):'';
  return k==='T'?tx(`Fixed: the eval has ${NT} tasks.`,`固定不变：评测有 ${NT} 个任务。`):tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  const m=ctlLockMsg(k); if(m) return m;
  if(k==='chK') return tx('Click to jump to “Most tries k” in the task card.','点击跳到任务卡里的“最多尝试次数 k”。');
  return CTLN[k]?tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`):'';
}
const ctlEl=k=>({lv:$('#pak-ctl-lv',app),k:$('#pak-ctl-k',app),n:$('#pak-ctl-n',app),chK:$('#pak-ic-chk',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#pak-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#pak-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='play'){if(playing){stopPlay(); render();} else {if(shown>=S.n) shown=0; startPlay(); frame();} A.tip.quiet&&A.tip.quiet(); return;}
  if(a==='r-'){setShown(shown-1); A.tip.quiet&&A.tip.quiet(); return;}
  if(a==='r+'){setShown(shown+1); A.tip.quiet&&A.tip.quiet(); return;}
  if(a==='reset'){bySettings(btn,()=>setState({...BASE})); return;}
  if(a==='reseed'){bySettings(btn,()=>setState({seed:S.seed+1})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chAg,chK).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#pak-code',app), fl=(t)=>{btn.textContent=t; clearTimeout(btn._t); btn._t=setTimeout(()=>{btn.textContent=A.t('copyCode');},1600);};
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); fl(tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(()=>fl(tx('Copied','已复制')),sel); else sel();}catch(err){sel();}}
}
/* a change under Settings never moves the page (R3.2) */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#pak-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const t=e.target, k=t.dataset&&t.dataset.key;
    if(t.id==='pak-rows'){setShown(+t.value); return;}
    if(!k) return; const v=+t.value;
    if(k==='lv') bySettings(t,()=>setState({lv:v}));
    else if(k==='k') bySettings(t,()=>setState({k:v}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#pak-rail')){wantObs=true; revealSoon(); wantObs=false;}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#pak-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&b.closest('#pak-ic-est')){byLearner(()=>setEst(d.lever)); revealSoon(); return;}
    if(d.lever!=null&&b.closest('#pak-ic-prod')){byLearner(()=>setProd(d.lever)); revealSoon(); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='spd'){spd=d.v; drawPlayer(); return;}
      if(k==='n') bySettings(b,()=>setState({n:+d.v,k:Math.min(S.k,+d.v)}));
      else if(k==='tau') bySettings(b,()=>setState({tau:+d.v}));
      else if(k==='chAg') byLearner(()=>{chAg=d.v; render();});
      else if(k==='chK') byLearner(()=>{chK=+d.v; render();});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; dropBadges(); byLearner(()=>setState({...BASE,...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#pak-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#pak-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the levels, estimates and products tried, the grids
     watched (one still playing counts: a reload shows it finished), the challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chAg,chK,chSubmitted,chReveal,seenLv:[...seen],est,estTried:[...estTried],
    prod,prodTried:[...prodTried],remView,moved4,sbTouched,spd,drillSeed,drillOpen,watched:[...new Set([...watched,...(playing&&viewNow()==='grid'?[sigNow()]:[])])].slice(-100),
    passed:mode==='challenge'&&chSubmitted&&chEval(chAg,chK).ok}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(token(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, ids=a=>a.map(x=>x.id), set=(v,ok,base)=>new Set([...base,...(Array.isArray(v)?v.filter(ok):[])]);
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null;
    chAg=CH.ags.includes(o.chAg)?o.chAg:CH.start.ag; chK=CH.ks.includes(o.chK)?o.chK:CH.start.k; chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    seen=set(o.seenLv,v=>ints(v,0,LVMAX),[]); est=ids(EST).includes(o.est)?o.est:'naive'; estTried=set(o.estTried,v=>ids(EST).includes(v),['naive',est]);
    prod=ids(PROD).includes(o.prod)?o.prod:'code'; prodTried=set(o.prodTried,v=>ids(PROD).includes(v),['code',prod]);
    remView=REMS.includes(o.remView)?o.remView:'uneven'; moved4=bool(o.moved4); sbTouched=bool(o.sbTouched); if(Object.keys(SPEEDS).includes(o.spd)) spd=o.spd;
    drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:3; drillOpen=bool(o.drillOpen);
    watched.clear(); if(Array.isArray(o.watched)) o.watched.forEach(k=>{if(typeof k==='string'&&/^\d+\|\d+\|\d+$/.test(k)) watched.add(k);});
    shown=0; playing=false; lastGrid='';},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#pak-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#pak-rail',app));},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&step===0) S={...S,...STEPS[0].enter()};
    gridSig=''; const wasShown=shown, wasPlaying=playing;
    fillRail();
    render();
    /* a language switch re-mounts: keep the playback where it was */
    if(tok==null&&(wasShown>0||wasPlaying)&&viewNow()==='grid'&&!A.reduceMotion()){shown=wasShown; stopPlay(); if(wasPlaying&&shown<S.n) startPlay(); render();}
  },
  leave(){clearTimeout(annT); stopPlay(); lastGrid=''; root=null; app=null;},
  onLang(){if(root){const d=$('#pak-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#pak-math',app).open=true;}},
  onResize(){if(app){gridSVG(); chartSVG();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; return viewNow()==='ch'?$('#pak-chart',app):$('#pak-grid',app);},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#pak-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.txt.length>14?s.txt.split(' ')[0]:s.txt;
      if(v==='ch'){const e=chEval(chAg,chK); vis.innerHTML=miniBar(e.done); read.innerHTML=esc(tx(`${chSet(chAg,chK)}: ${fP(e.done)} of tasks done, ${fC(e.calls)} calls per task`,`${chSet(chAg,chK)}：做成 ${fP(e.done)}，平均调用 ${fC(e.calls)} 次`)); return;}
      const r=R(), k=Math.min(kNow(),S.n);
      if(v==='first'){vis.innerHTML=miniBar(r.p1); read.innerHTML=esc(tx(`${r.first1} of ${NT} tasks pass on the first try; pass@5 after your prediction`,`${NT} 个任务里 ${r.first1} 个第一次就通过；pass@5 在你预测之后给出`)); return;}
      if(!done()){vis.innerHTML=miniBar(0); read.innerHTML=esc(tx(`Trying: ${shown} of ${S.n} tries done`,`正在试：已试 ${shown} / ${S.n} 次`)); return;}
      const fm=focusM(), val=fm==='hat'?r.hat[k]:r.at[k];
      vis.innerHTML=miniBar(val);
      read.innerHTML=esc(fm==='hat'?tx(`All ${k} tries pass on ${fP(val)} of tasks (pass^${k})`,`${k} 次全过的任务占 ${fP(val)}（pass^${k}）`)
        :tx(`Up to ${k} tries get ${fP(val)} of tasks done (pass@${k})`,`最多试 ${k} 次，做成 ${fP(val)} 的任务（pass@${k}）`));}
  }
});
function miniBar(x){return `<div class="pak-dk"><i style="width:${(Math.max(0,Math.min(1,x))*100).toFixed(1)}%"></i></div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__pak=Object.freeze({
  tries:(seed,n)=>deep(triesOf(seed,n)), pAt, pHat,
  compute:s=>{const r=compute({...BASE,...s}); return deep({sp:r.sp,c:r.tasks.map(x=>x.c),kind:r.tasks.map(x=>x.kind),at:r.at,hat:r.hat,firstK:r.firstK,tAt:r.tAt,tHat:r.tHat,nAt:r.nAt,nHat:r.nHat,p1:r.p1,zero:r.zero,first1:r.first1,sorted:r.sorted.map(x=>x.i)});},
  chEval:(ag,k)=>deep(chEval(ag,k)), chBest:()=>deep(chBest()), bestTauLv, estVal, prodVal, drill:s=>deep(makeDrill(s)), predRight,
  fmt:Object.freeze({P:fP}),
  consts:Object.freeze({NT,Q,NS:NS.slice(),LVMAX,RING:deep(RING),TAU:TAU.slice(),SHUF:SHUF.slice(),CH:deep(CH),BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s})))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},seen:[...seen],est,estTried:[...estTried],prod,prodTried:[...prodTried],remView,moved4,shown,playing,spd,chAg,chK,chSubmitted,chReveal})
});
})();
