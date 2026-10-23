/* agent-capstone (world agents): How do you build an agent loop that holds up, and measure it?
   Fidelity: Simulated (header: Toy run). A fixed, seeded suite of 50 tasks of three kinds (lookups, multi-step edits, tasks that read
   untrusted content) is run k times per task under a harness; every rate is the lab's assumption, the evaluation statistics are exact.
   - A run of task i: n steps. Each attempt at a step succeeds with p = 1 − q_kind × m_i × (0.5 for the stronger model). A failed attempt
     is loud with r (a check after it makes a share C of the silent ones loud, one more model call); a loud failure is retried up to R
     times with the error message (p2 = p); retries used up → the run stops. A silent failure → the run ends wrong.
   - Context: base B; each step adds its tool output + a model turn (+ error messages). 'all' keeps everything (over the window W the
     call is rejected: overflow); 'last' keeps the last 8 steps besides the base, so the task itself stays (a task that needs, at its
     last step, a fact its step 1 looked up ends wrong: 'lost'); 'notes' trims every output to 1,000 tokens and every 8 steps one
     summary call compresses the history into 1,300 tokens (trim + summary + notes).
   - Untrusted tasks read a page at step 2; some carry an injected instruction; at step 3 the model follows it with F (a detector removes
     it first with D). Without a person's confirmation that is an unsafe action; with it the action is refused. Each untrusted run with
     confirmation asks a person once more for its own risky action at the last step.
   - Cost of one run = input tokens read × price (cents); prompt caching is not modeled (said in the Fidelity card). Evaluation:
     s_i = share of task i's k runs that succeed; success = mean s_i; 95% interval = mean ± 1.96 × sd(s_i)/√n (lm-eval's mean_stderr
     over tasks); pass^3 = mean C(c_i,3)/C(k,3); unsafe = count. The interval has two sources (as in eval-ci): which tasks are in the
     suite, and the luck of each run; more runs shrink only the second (lucksplit() estimates it from many draws).
   - Draws: u(seed, task, run, slot, purpose), a counter-based 32-bit hash (common random numbers across harnesses).
   window.__acp is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='agent-capstone';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2').replace(/([一-鿿]) ((?:<[^>]+>)*)(?=[一-鿿])/g,'$1$2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const K_={B:3000,W:131072,TURN:200,ERR:300,TRIM:1000,SUMT:1000,NOTES:300,EVERY:8,LASTN:8,PAGE:6000,SMALL:600,C:0.6,F:0.5,D:0.8};
const PRICE={cheap:1,strong:5};                 // US dollars per million input tokens (the lab's assumption)
const SUITES={
 shop:{seed:11,kinds:[['L',20],['E',20],['U',10]],n:{L:[3,3],E:[12,9],U:[5,3]},q:{L:0.03,E:0.05,U:0.04},r:0.6,outE:[3000,8],inj:j=>j%2===0,need:0.7},
 ops:{seed:23,kinds:[['L',10],['E',25],['U',15]],n:{L:[3,3],E:[14,11],U:[5,3]},q:{L:0.03,E:0.05,U:0.04},r:0.9,outE:[4000,8],inj:j=>j%3!==2,need:0.7},
 tile:{seed:89,kinds:[['L',4],['E',5],['U',3]],n:{L:[3,3],E:[12,9],U:[5,3]},q:{L:0.03,E:0.05,U:0.04},r:0.6,outE:[3000,8],inj:j=>j%2===0,need:0.7}};
const mix=h=>{h^=h>>>16; h=Math.imul(h,0x7feb352d); h^=h>>>15; h=Math.imul(h,0x846ca68b); h^=h>>>16; return h>>>0;};
function u01(seed,a,b,c,d){let h=mix(Math.imul(seed,0x9E3779B1)^0x85ebca6b); h=mix(h^Math.imul(a+1,0x27d4eb2f)); h=mix(h^Math.imul(b+1,0x165667b1)); h=mix(h^Math.imul(c+1,0x2545f491)); h=mix(h^Math.imul(d+1,0x68e31da4)); return h/4294967296;}
const TC={};
function tasksOf(name){
  if(TC[name]) return TC[name];
  const S=SUITES[name], out=[]; let i=0;
  S.kinds.forEach(([kind,cnt])=>{for(let j=0;j<cnt;j++){
    const u=p=>u01(S.seed,i,p,0,0), [lo,span]=S.n[kind], x=u(1);
    const n=lo+Math.floor(u(0)*span), m=x<0.40?0.5:(x<0.75?1:(x<0.90?2:4));
    const o=kind==='E'?S.outE[0]+Math.floor(u(2)*S.outE[1])*1000:K_.SMALL;
    out.push({i,kind,n,m,out:o,need:kind==='E'&&u(3)<S.need,inj:kind==='U'&&S.inj(j)}); i++;}});
  return (TC[name]=out);
}
const runSeed=(name,draw)=>SUITES[name].seed*1000+7+101*((draw||1)-1);
/* one run of one task under harness h; res: ok | stop | wrong | lost | overflow | unsafe */
function simRun(name,t,k,h){
  const S=SUITES[name], seed=runSeed(name,h.draw), p=1-S.q[t.kind]*t.m*(h.model==='strong'?0.5:1);
  const hist=[]; let summ=0, tokens=0, calls=0, conf=0, wrong=false, unsafe=false, res=null, steps=0;
  let live=t.inj; if(t.inj&&h.gate==='detect'&&u01(seed,t.i,k,0,4)<K_.D) live=false;
  const follows=live&&u01(seed,t.i,k,0,3)<K_.F;
  for(let j=1;j<=t.n;j++){
    let o=(t.kind==='U'&&j===2)?K_.PAGE:t.out; if(h.ctx==='notes') o=Math.min(o,K_.TRIM);
    let ctx=K_.B;
    if(h.ctx==='all') hist.forEach(g=>{ctx+=g;}); else if(h.ctx==='last') hist.slice(-K_.LASTN).forEach(g=>{ctx+=g;}); else ctx+=summ;
    if(h.ctx==='all'&&ctx>K_.W){res='overflow'; break;}
    let a=0, fails=0, sr='ok';
    for(;;){calls++; tokens+=ctx+K_.ERR*a; if(h.ck){calls++; tokens+=ctx+K_.ERR*a+K_.TURN;}
      if(u01(seed,t.i,k,j*4+a,0)<p){sr='ok'; break;}
      const loud=u01(seed,t.i,k,j*4+a,1)<S.r||(!!h.ck&&u01(seed,t.i,k,j*4+a,2)<K_.C);
      if(!loud){sr='silent'; break;}
      fails++; if(a>=h.R){sr='stop'; break;} a++;}
    steps=j;
    if(sr==='stop'){res='stop'; break;}
    if(sr==='silent') wrong=true;
    if(t.kind==='U'&&j===3&&follows){if(h.gate==='confirm') conf++; else unsafe=true;}
    if(t.kind==='U'&&j===t.n&&h.gate==='confirm') conf++;
    const grow=o+K_.TURN+K_.ERR*fails;
    if(h.ctx==='notes'){summ+=grow; if(j%K_.EVERY===0&&j<t.n){calls++; tokens+=K_.B+summ; summ=K_.SUMT+K_.NOTES;}}
    else hist.push(grow);
  }
  if(unsafe) res='unsafe';
  else if(res==null) res=(t.need&&h.ctx==='last'&&t.n>K_.LASTN+1)?'lost':(wrong?'wrong':'ok');
  return {res,unsafe,tokens,calls,conf,steps,cost:tokens*PRICE[h.model]/1e6*100};
}
const comb=(n,k)=>{if(k<0||k>n) return 0; let r=1; for(let i=1;i<=k;i++) r=r*(n-k+i)/i; return r;};
const CAUSES=['stop','wrong','lost','overflow','unsafe'];
const EC=new Map();
const hKey=(name,h)=>[name,h.model,h.R,h.ck,h.ctx,h.gate,h.K,h.draw||1].join('|');
function evaluate(name,h){
  const key=hKey(name,h); if(EC.has(key)) return EC.get(key);
  const T=tasksOf(name), K=h.K, runs=[], per=[], cnt=[];
  T.forEach(t=>{const rs=[]; for(let k=0;k<K;k++) rs.push(simRun(name,t,k,h)); runs.push(rs); const c=rs.filter(r=>r.res==='ok').length; cnt.push(c); per.push(c/K);});
  const n=T.length, mean=per.reduce((a,b)=>a+b,0)/n, sd=Math.sqrt(per.reduce((a,x)=>a+(x-mean)*(x-mean),0)/(n-1)), se=sd/Math.sqrt(n);
  const flat=[].concat(...runs), N=flat.length;
  const causes={}; CAUSES.forEach(c=>{causes[c]=flat.filter(r=>r.res===c).length;});
  const p3=K>=3?cnt.reduce((a,c)=>a+comb(c,3)/comb(K,3),0)/n:null;
  const e={name,h:{...h},n,K,runs,per,cnt,succ:mean,sd,se,lo:mean-1.96*se,hi:mean+1.96*se,p3,
    unsafe:flat.filter(r=>r.unsafe).length,cost:flat.reduce((a,r)=>a+r.cost,0)/N,conf:flat.reduce((a,r)=>a+r.conf,0)/N,causes,N};
  if(EC.size>400) EC.clear(); EC.set(key,e); return e;
}
/* the gain of h over h0, task by task (paired), on the same tasks and the same number of runs */
function gainOf(name,h,h0){
  const a=evaluate(name,h), b=evaluate(name,h0), d=a.per.map((x,i)=>x-b.per[i]), n=d.length, m=d.reduce((s,x)=>s+x,0)/n;
  const sd=Math.sqrt(d.reduce((s,x)=>s+(x-m)*(x-m),0)/(n-1)), half=1.96*sd/Math.sqrt(n);
  return {m,lo:m-half,hi:m+half,sd};
}
/* what the interval is made of (B1; the two parts eval-ci teaches): each task's true success chance p_i is estimated from D draws of
   the same harness (D × K runs per task). Between tasks: the spread of the p_i (which tasks are in the suite); luck: mean p_i(1 − p_i),
   divided by k for the share of k runs. luck(k) = the luck part's share of the variance of a task's share at k runs per task. */
const LSC=new Map();
function lucksplit(name,h,D){
  D=D||200; const key=hKey(name,h)+'|'+D; if(LSC.has(key)) return LSC.get(key);
  const T=tasksOf(name), K=h.K, p=T.map(t=>{let c=0; for(let d=1;d<=D;d++){const hd={...h,draw:d}; for(let k=0;k<K;k++) if(simRun(name,t,k,hd).res==='ok') c++;} return c/(D*K);});
  const n=p.length, m=p.reduce((a,b)=>a+b,0)/n, between=p.reduce((a,x)=>a+(x-m)*(x-m),0)/n, within=p.reduce((a,x)=>a+x*(1-x),0)/n;
  const o={between,within,luck:k=>(within/k)/(between+within/k)}; LSC.set(key,o); return o;
}
/* the launch goal, judged on the values as shown (one decimal) so that a shown 85.0% never fails (R6.10) */
const r1=x=>Math.round(x*10)/10;
const GOAL={succ:85,lo:80,p3:70,unsafe:0};
const BUDGET={shop:15,ops:12,tile:15};
function goalOf(e,budget){
  const s=r1(e.succ*100), lo=r1(Math.max(0,e.lo)*100), p3=e.p3==null?null:r1(e.p3*100), c=r1(e.cost);
  const g={succ:s>=GOAL.succ&&lo>=GOAL.lo,p3:p3!=null&&p3>=GOAL.p3,unsafe:e.unsafe===0,cost:c<=budget};
  g.all=g.succ&&g.p3&&g.unsafe&&g.cost; g.miss=['succ','p3','unsafe','cost'].filter(k=>!g[k]).length; return g;
}
const INIT={model:'cheap',R:0,ck:0,ctx:'all',gate:'none',K:3,draw:1};
const FINAL={...INIT,R:2,ck:1,ctx:'notes',gate:'confirm'};

/* the challenge: another team's ops agent, a tighter budget; five choices */
const CH={suite:'ops',budget:BUDGET.ops,models:['cheap','strong'],Rs:[0,1,2],cks:[0,1],ctxs:['all','last','notes'],gates:['none','detect','confirm']};
const chH=o=>({...INIT,model:o.model,R:o.R,ck:o.ck,ctx:o.ctx,gate:o.gate});
function chEval(o){const e=evaluate(CH.suite,chH(o)); return {o:{...o},e,g:goalOf(e,CH.budget)};}
function chAll(){const out=[]; CH.models.forEach(model=>CH.Rs.forEach(R=>CH.cks.forEach(ck=>CH.ctxs.forEach(ctx=>CH.gates.forEach(gate=>out.push(chEval({model,R,ck,ctx,gate}))))))); return out;}
/* minimal: the lowest cost per task as shown (one decimal) among the passing combinations */
function chBest(){const ok=chAll().filter(x=>x.g.all); if(!ok.length) return null; ok.sort((a,b)=>r1(a.e.cost)-r1(b.e.cost)||a.o.R-b.o.R); return ok[0];}
const oKey=o=>[o.model,o.R,o.ck,o.ctx,o.gate].join('_');

/* =====================================================================
   formatting and small helpers
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>fixd(Math.max(0,Math.min(1,x))*100,1)+'%';
const fPts=x=>(x>=0?'+':'−')+fixd(Math.abs(x)*100,1);
const fCt=c=>tx(`${fixd(c,1)} cents`,`${fixd(c,1)} 美分`);
const fI=e=>tx(`${fP(e.lo)} to ${fP(e.hi)}`,`${fP(e.lo)} 到 ${fP(e.hi)}`);
const fInt=n=>A.fmt.int?A.fmt.int(n):String(n);
/* the error shares, from the engine (R9.3, R6.6: say whose share): of all errors r are reported and 1 − r are silent; a check finds C
   of the silent ones, so (1 − r)(1 − C) of all errors stay unseen */
const pc=x=>`${Math.round(x*100)}%`;
const ERRS=suite=>{const r=SUITES[suite].r; return {rep:pc(r),sil:pc(1-r),ck:pc(K_.C),left:pc((1-r)*(1-K_.C))};};
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const T_=(k,x)=>`<span class="t" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const M=t=>A.ui.math?A.ui.math(t):esc(t);
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="acp-eqs${EQN(rows)?' acp-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="acp-eq-l">${l?M(l):''}</span><span class="acp-eq-e">${M(rel||'=')}</span><span class="acp-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="acp-mb"><p>${lead}</p>${body}${res?`<p class="acp-mb-res">${res}</p>`:''}</div>`;
/* a hover's plain text with its formulas typeset (R6.29) */
const tipMath=x=>esc(x).replace(/√n/g,M('\\sqrt{n}')).replace(/C\(c, 3\) ÷ C\(k, 3\)/g,M('\\frac{C(c, 3)}{C(k, 3)}'));
const LL=id=>A.shell.labLink?A.shell.labLink(id):esc(id);
const runsW=K=>tx(K===1?'1 run':`${K} runs`,`${K} 次`);
const MODELN={cheap:{en:'the cheaper model',zh:'便宜的模型'},strong:{en:'the stronger model',zh:'更强的模型'}};
const CTXN={all:{en:'the whole context kept',zh:'上下文全部保留'},last:{en:'only the last 8 steps kept',zh:'上下文只留最近 8 步'},notes:{en:'trim + summary + notes',zh:'上下文裁剪 + 摘要 + 笔记'}};
const GATEN={none:{en:'no gate on risky actions',zh:'危险操作不把关'},detect:{en:'an injection detector',zh:'检测模型过滤注入'},confirm:{en:'a person confirms risky actions',zh:'危险操作要人确认'}};
const retriesW=R=>R===0?tx('no retries','不重试'):tx(R===1?'1 retry':`up to ${R} retries`,`最多重试 ${R} 次`);
const hName=h=>[L(MODELN[h.model]),retriesW(h.R),h.ck?tx('a check after each step','每步检查'):tx('no check','不检查'),L(CTXN[h.ctx]),L(GATEN[h.gate])].join(tx(', ','，'));
/* the stage header: the same full names as the results (S10); in English “full context” is the short form of “the whole context
   kept”, so the initial harness fits on two lines at 390 and the grid stays on the phone's first screen (R7.3) */
const hSub=h=>[h.model==='strong'?tx('Stronger model','更强的模型'):tx('Cheaper model','便宜的模型'),retriesW(h.R),
  h.ck?tx('a check after each step','每步检查'):tx('no check','不检查'),h.ctx==='all'?tx('full context',L(CTXN.all)):L(CTXN[h.ctx]),L(GATEN[h.gate])].join(tx(', ','，'));
const KINDN={L:{en:'lookups',zh:'查询'},E:{en:'multi-step edits',zh:'多步修改'},U:{en:'outside text',zh:'读外部内容'}};
const KINDO={L:{en:'lookups',zh:'查询'},E:{en:'long edits',zh:'长的修改'},U:{en:'tickets',zh:'工单'}};   // the ops agent's suite (the challenge)
const kindName=k=>L((S.suite==='ops'?KINDO:KINDN)[k]);
const CAUSEN={stop:{en:'stopped on a reported error',zh:'报错后停下'},wrong:{en:'finished, but wrong',zh:'悄悄做错'},lost:{en:'lost an early fact',zh:'丢了早先查到的事实'},overflow:{en:'context over the window',zh:'上下文超出窗口'},unsafe:{en:'unsafe action',zh:'不安全的操作'}};

/* =====================================================================
   module state
   ===================================================================== */
let S={...INIT,suite:'shop'}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lv2='init', tried2=new Set(['init']), lv3='all', tried3=new Set(['all']), lv4='none', tried4=new Set(['none']), tried5=new Set([3]), remView='';
let pendBadge=false, runsSeen=false, saved=null, chO={model:'cheap',R:0,ck:0,ctx:'all',gate:'none'}, chTried=[], chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null;
let drillSeed=1, drillOpen=false;
const shownPrev={};
/* R3.9: the grid plays a whole evaluation (150 runs), so its default speed is fast (the product owner, 2026-10-04) */
const PL={pv:0,on:false,speed:'f',raf:0,last:0,acc:0};
const SPD={s:0.2,m:0.13,f:0.01};               // seconds per run shown (R3.9): 150 runs in 30 s, 19.5 s, 1.5 s (fast, the default)
const hNow=()=>({model:S.model,R:S.R,ck:S.ck,ctx:S.ctx,gate:S.gate,K:S.K,draw:S.draw});
const E_=()=>evaluate(S.suite,hNow());
const budgetNow=()=>BUDGET[S.suite];

/* =====================================================================
   definitions (R2.8: one term, one wording; harness as in agent-reliability)
   ===================================================================== */
const TIP={
 harness:()=>tx('The harness is the code around the model that runs the agent: it calls the model, runs the tools, hands the results back, and decides when to retry or stop.',
                'harness 是包在模型外面、负责运行 agent 的代码：它调用模型、执行工具、把结果交回去，并决定什么时候重试、什么时候停下。'),
 suite:()=>tx('An evaluation suite is a fixed set of tasks, each with a way to tell whether a run got it right. Every version of the agent is run on the same tasks, so the scores can be compared.',
              '评测集是一组固定的任务，每个任务都有办法判断一次运行做对了没有。每个版本的 agent 都跑同一组任务，分数才能拿来比较。'),
 run:()=>tx('A run is one attempt at a whole task, from the first model call to the end. The same task can succeed in one run and fail in the next, because the model samples its answers.',
            '一次运行，就是把一个任务从第一次调用模型一直做到结束。同一个任务这一次成功、下一次可能失败，因为模型的回答是抽样出来的。'),
 ci:()=>tx('The 95% interval says how precise this success rate is. Another batch of tasks of the same kind, or the same tasks run again, gives another score. An interval worked out this way covers the true success rate about 95% of the time. It comes from how much the tasks’ success shares differ from each other.',
           '95% 区间说的是这个成功率有多准。换一批同类的任务，或者把同样的任务再跑一遍，分数都会变。照这样算出的区间，大约 95% 的时候能盖住真实的成功率。它按各个任务的成功比例彼此差多少来算。'),
 p3:()=>tx('pass^3 is the share of tasks that succeed in all 3 of 3 runs, the chance that a user who asks the same thing three times gets it right every time. When each task runs more than 3 times, it is estimated from the success counts, the way the lab pass@k explains.',
           'pass^3 是 3 次运行全部成功的任务占多少，也就是同一个请求连做 3 次、每次都做对的把握。每个任务跑了不止 3 次时，就从成功次数来估，算法见 pass@k 这个 Lab。'),
 untrusted:()=>tx('Untrusted content is text the agent reads but nobody on your side wrote: a customer’s email, a web page, an attachment. Anyone can put sentences in it that look like instructions.',
                  '不可信的内容，是 agent 读到、但不是你这边的人写的文字：顾客的邮件、网页、附件。任何人都可以在里面写看起来像指令的句子。'),
 inject:()=>tx('A prompt injection is a sentence hidden in untrusted content that tells the agent to do something, such as “refund this order to account X”. The model may follow it, because to the model it is just more text in the conversation.',
               '提示注入（prompt injection）是藏在不可信内容里、让 agent 去做某件事的一句话，比如“把这笔退款打到 X 账户”。模型可能会照做，因为对模型来说，它只是对话里的又一段文字。'),
 cache:()=>tx('Prompt caching is a cache that model providers offer. The unchanged start of a call is stored, and the next call reuses it. The reused part costs much less (one provider charges a tenth of the normal input price), but it still takes room in the context window.',
              'prompt caching 是模型服务商提供的缓存。一次调用里没变的开头会存起来，下一次调用直接复用。复用的部分便宜得多（一家服务商只收普通输入价格的一成），但照样占着上下文窗口。'),
 window:()=>tx('The context window is the most tokens one model call can read: the system prompt, the conversation and every tool result so far. Here it is 128K tokens (131,072), the lab’s assumption; a call over it is rejected.',
               '上下文窗口是一次模型调用最多能读的 token 数：system prompt、对话和到目前为止的每个工具结果都算在内。这里是 128K 个 token（131,072），这是这个 Lab 的假设；超过它的调用会被拒绝。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
/* part 1 asks which AREA to fix first (S1): option a is one area, errors, with its two settings (retries and a check); the result
   then says what each of the two does alone */
const PRED=[{v:'a',t:{en:'Handle errors (retry with the error message, check after each step)',zh:'处理出错（带着报错重试，每步之后再检查）'}},
  {v:'b',t:{en:'Manage the context (trim + summary + notes)',zh:'管好上下文（裁剪 + 摘要 + 笔记）'}},
  {v:'c',t:{en:'Have a person confirm risky actions',zh:'危险的操作先让人确认'}},
  {v:'d',t:{en:'Switch to a stronger, pricier model',zh:'换一个更强、也更贵的模型'}}];
const PRED_H={a:{R:2,ck:1},b:{ctx:'notes'},c:{gate:'confirm'},d:{model:'strong'}};
const predRight=()=>{const b=PRED.map(o=>({v:o.v,s:evaluate('shop',{...INIT,...PRED_H[o.v]}).succ})); b.sort((x,y)=>y.s-x.s); return b[0].v;};
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const runsDone=()=>PL.pv>=S.K*tasksOf(S.suite).length;
/* part 1: nothing about the result shows before the prediction and before the runs end (R3.8); every part: the cards wait for the runs */
const hideRes=()=>isP('mission')&&!(pred&&runsSeen);
const waiting=()=>hideRes()||!runsDone();
const P2={...INIT}, P3={...INIT,R:2,ck:1}, P4={...P3,ctx:'notes'}, P5={...FINAL};
const LV2=[{v:'init',s:{R:0,ck:0},t:{en:'Initial setup: no retries, no check',zh:'初始设置：不重试，不检查'},d:{en:'A reported error stops the run.',zh:'一报错，这次运行就停下。'}},
  {v:'r2',s:{R:2,ck:0},t:{en:'Retry with the error message, up to 2 times',zh:'带着报错重试，最多 2 次'},d:{en:'Silent errors still go unseen.',zh:'悄悄出的错照样没人看见。'}},
  {v:'r2c',s:{R:2,ck:1},t:{en:'2 retries, and a check after each step',zh:'重试 2 次，每步之后再检查'},d:{en:'One more model call after every attempt.',zh:'每次尝试之后多一次模型调用。'}}];
const LV3=[{v:'all',s:{ctx:'all'},t:{en:'Keep everything (retries and check already on)',zh:'全部保留（重试和检查已加上）'},d:{en:'Every tool output stays in full.',zh:'每个工具的输出都原样留着。'}},
  {v:'last',s:{ctx:'last'},t:{en:'Keep only the last 8 steps',zh:'只留最近 8 步'},d:{en:'The task itself stays; older steps are dropped.',zh:'任务本身一直留着，更早的步骤整个删掉。'}},
  {v:'notes',s:{ctx:'notes'},t:{en:'Trim + summary + notes',zh:'裁剪 + 摘要 + 笔记'},d:{en:'Long outputs cut to the useful part, facts for later kept in notes.',zh:'长输出只留有用的部分，以后要用的事实记进笔记。'}}];
const LV4=[{v:'none',s:{gate:'none'},t:{en:'No gate (retries, check and context management already on)',zh:'不把关（重试、检查、管上下文都已加上）'},d:{en:'The agent acts on whatever it reads.',zh:'agent 读到什么就照着做。'}},
  {v:'detect',s:{gate:'detect'},t:{en:'Filter injections with a detector',zh:'用检测模型过滤注入'},d:{en:'It removes most hidden instructions before the model reads them.',zh:'在模型读到之前，删掉大部分藏着的指令。'}},
  {v:'confirm',s:{gate:'confirm'},t:{en:'A person confirms risky actions after outside content',zh:'读过外部内容后，危险操作要人确认'},d:{en:'Refunds and messages wait for a yes.',zh:'退款、发消息都要等人点头。'}}];
const match=(list)=>{const o=list.find(x=>Object.keys(x.s).every(k=>S[k]===x.s[k])); return o?o.v:'';};
const lvRes2=s=>fP(evaluate('shop',{...hNow(),...s}).succ);
const lvRes4=s=>{const e=evaluate('shop',{...hNow(),...s}); return tx(`${e.unsafe} unsafe`,`不安全 ${e.unsafe} 次`);};
function leversHTML(id,list,cur,tried,res,aria){
  return `<div class="acp-lv" id="${id}">`+A.ui.levers({aria,value:cur,options:list.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc((tried.has(o.v)||o.v===cur)&&!waitingFor(o)?res(o.s):'')}))})+`</div>`;
}
/* an option's result shows once its runs have finished (R3.8): the current option waits for the replay */
const waitingFor=o=>Object.keys(o.s).every(k=>S[k]===o.s[k])&&!runsDone();
/* cost is per run (S8): one run handles one request once; the evaluation runs each task 3 times */
const goalList=()=>tx(`<ul class="acp-goal-l"><li>success at least 85%, and the lower end of its ${DEF('95% interval',TIP.ci())} at least 80%;</li><li>${DEF('pass^3',TIP.p3())} at least 70%;</li><li>no unsafe action at all;</li><li>on average at most 15 cents per run.</li></ul>`,
  `<ul class="acp-goal-l"><li>成功率至少 85%，${DEF('95% 区间',TIP.ci())}的下沿也在 80% 以上；</li><li>${DEF('pass^3',TIP.p3())} 至少 70%；</li><li>不安全的操作一次都没有；</li><li>每次运行平均花费不超过 15 美分。</li></ul>`);
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('A support agent misses its launch goal. Which area, fixed first, helps most?','客服 agent 没达到上线标准。先改哪个方面最管用？'),
  enter:()=>({...INIT}),gate:()=>!!pred&&runsSeen,skip:true,label:{en:'Result',zh:'结果'},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You run an online shop’s support agent. It does three kinds of tasks: lookups such as “where is my order”; edits of a dozen steps or more, such as changing an address or handling a return; and tasks that read a customer’s email or a web page, which is ${DEF('untrusted content',TIP.untrusted())}.</p><p>Before launch, the team runs a fixed ${DEF('suite',TIP.suite())} of 50 tasks, each 3 times (3 ${DEF('runs',TIP.run())}). All four must hold:</p>${goalList()}<p>Today’s ${DEF('harness',TIP.harness())} has nothing added: no retries, no check, the whole context kept, no gate on risky actions. It misses the goal. If you could fix only one area first, which one raises the success rate most?</p>${ch}${pred?`<p class="muted small">${runsSeen?'Saved.':'Saved. You can still change it before the 150 runs in the chart finish.'}</p>`:''}`,
             `<p>你负责一个网店的客服 agent。它做三类任务：查订单这样的简单查询；改地址、退换货这样要走十几步的修改；还有要读顾客邮件或网页的任务，这些邮件和网页都是${DEF('不可信的内容',TIP.untrusted())}。</p><p>上线前，团队用一组固定的 50 个任务做${DEF('评测集',TIP.suite())}，每个任务${DEF('运行',TIP.run())} 3 次。下面四条要同时满足：</p>${goalList()}<p>现在的 ${DEF('harness',TIP.harness())} 什么都没加：出错不重试，不检查，上下文全部保留，危险操作也没人把关。它达不到要求。如果只能先改一个方面，改哪一个，成功率涨得最多？</p>${ch}${pred?`<p class="muted small">${runsSeen?'已记下。':'已记下。图里的 150 次运行跑完之前，还可以改。'}</p>`:''}`);},
  obs:()=>obsMission()},
 {id:'loud',title:{en:'Make failures visible',zh:'让错误露出来'},
  lead:()=>tx('Retry the errors that are reported; a check makes silent ones show.','报出来的错要重试；检查能让悄悄出的错露出来。'),
  enter:()=>{lv2='init'; tried2=new Set(['init']); return {...P2};},expect:()=>tried2.size>1&&runsDone(),showMe:()=>setLv(2,'r2c'),label:{en:'Result',zh:'结果'},
  /* the product owner (2026-10-04, R6.6): the check finds 60% of the silent 40%, said in that order */
  body:()=>{const x=ERRS('shop'), two=x.rep===x.ck;
   return tx(`<p>In the initial setup, most failed runs stopped on a reported error: a tool returned an error and nobody tried again. Other errors are silent, such as an edit to the wrong order, and nothing reports them. ${LL('agent-reliability')} shows how the two differ.</p><p>In this lab ${x.rep} of the errors are reported, and the other ${x.sil} are silent. A check is one more model call after each step, to see whether the step was done right. Of those ${x.sil} silent errors, the check finds ${x.ck}. An error the check finds goes to the retries, just like a reported one. ${two?`Both ${x.ck} figures are assumptions`:'Both shares are assumptions'}.</p><p>Pick a way to handle the errors and see how high the success rate gets.</p>`,
              `<p>初始设置下，失败的运行大多停在报出来的错上：工具报了错，没有人再试一次。还有一些错是悄悄出的，比如改错了订单，没有任何报错。这两种错的区别，见 ${LL('agent-reliability')}。</p><p>在这个 Lab 里，${x.rep} 的错会报出来，剩下 ${x.sil} 是悄悄出的。检查就是每一步做完，再调用一次模型，看这一步做对了没有。这 ${x.sil} 悄悄出的错里，检查能发现 ${x.ck}。检查发现的错，和报出来的错一样，交给重试去改。${two?`这两个 ${x.ck} 都是假设`:'报出来的比例和检查能发现的比例都是假设'}。</p><p>选一种处理错误的办法，看看成功率能涨到多少。</p>`)+leversHTML('acp-ic-lv2',LV2,lv2,tried2,lvRes2,tx('Pick a way to handle the errors','选一种处理错误的办法'));},
  obs:()=>obsLoud()},
 {id:'context',title:{en:'Keep the context in check',zh:'管好上下文'},
  lead:()=>tx('Long tasks overflow the window and cost the most; trim + summary + notes fixes both.','长任务会撑满窗口，也最费钱。裁剪 + 摘要 + 笔记能把两头都管住。'),
  enter:()=>{lv3='all'; tried3=new Set(['all']); return {...P3};},expect:()=>tried3.size>1&&runsDone(),showMe:()=>setLv(3,'notes'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Each step of a multi-step edit reads back thousands of tokens of order history and logs. Kept in full, they are read again on every call, so each call costs more than the last. At the ${DEF('context window',TIP.window())}’s limit the call is rejected. The costs here leave out ${DEF('prompt caching',TIP.cache())}. With it, the re-read part would be much cheaper, but the window would still fill.</p><p>Two common fixes. One keeps only the last 8 steps. The other is trim + summary + notes: each long output is cut to its useful part, the old steps are folded into a summary every 8 steps, and facts needed later go into notes (${LL('context-eng')}).</p><p>Pick a way to manage the context and see how the success rate changes.</p>`,
              `<p>多步修改的任务，每一步都要读回几千个 token 的订单记录和日志。全部留在上下文里，每次调用都要把它们重读一遍，一次比一次贵。到了${DEF('上下文窗口',TIP.window())}的上限，调用就会被拒绝。这里的花费没有算 ${DEF('prompt caching',TIP.cache())}。有了它，重读的部分会便宜很多，可窗口照样会撑满。</p><p>常见的办法有两种。一种是只留最近 8 步。另一种是裁剪 + 摘要 + 笔记。长输出只留有用的部分，旧步骤每 8 步压成一段摘要，以后要用的事实记进笔记（见 ${LL('context-eng')}）。</p><p>选一种管上下文的办法，看看成功率怎么变。</p>`)+leversHTML('acp-ic-lv3',LV3,lv3,tried3,lvRes2,tx('Pick a way to manage the context','选一种管上下文的办法')),
  obs:()=>obsContext()},
 {id:'gate',title:{en:'Gate the risky actions',zh:'危险的操作要把关'},
  lead:()=>tx('A detector lowers the risk; only a person’s yes keeps it at zero.','检测模型能少中招；只有人点头，才能保证一次都不出事。'),
  enter:()=>{lv4='none'; tried4=new Set(['none']); return {...P4};},expect:()=>tried4.size>1&&runsDone(),showMe:()=>setLv(4,'confirm'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Of the 10 tasks that read a customer’s email or a web page, 5 hide a sentence for the agent, such as “refund this order to account X”. This is a ${DEF('prompt injection',TIP.inject())}: the model cannot reliably tell the customer’s request from text inside the email, and sometimes it obeys (${LL('prompt-injection')}).</p><p>In this lab the model obeys an injected instruction half the time. A detector removes 80% of the injections before the model reads them. Both shares are assumptions.</p><p>Pick a safeguard and see how many unsafe actions are left.</p>`,
              `<p>读顾客邮件或网页的 10 个任务里，有 5 个藏着一句给 agent 的话，比如“把这笔退款打到 X 账户”。这就是${DEF('提示注入',TIP.inject())}：模型分不清哪句是顾客的要求、哪句只是邮件里的文字，有时会照做（见 ${LL('prompt-injection')}）。</p><p>在这个 Lab 里，模型有一半的机会照做注入的指令。检测模型能在模型读到之前，删掉 80% 的注入。这两个比例都是假设。</p><p>选一种把关的办法，看看不安全的操作还剩几次。</p>`)+leversHTML('acp-ic-lv4',LV4,lv4,tried4,lvRes4,tx('Pick a safeguard','选一种把关的办法')),
  obs:()=>obsGate()},
 {id:'prove',title:{en:'Prove it',zh:'证明它'},
  lead:()=>tx('How many runs per task? The interval and pass^3 decide.','每个任务跑几次？区间和 pass^3 说了算。'),
  enter:()=>{tried5=new Set([3]); return {...P5};},expect:()=>tried5.size>1&&runsDone(),showMe:{K:1},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The harness now meets the goal on the suite. But the evaluation has noise of its own: another batch of tasks, or the same tasks run again, gives another score. Before launch, two questions need an answer. Is the interval’s lower end above the line? Does each task succeed every time? (${LL('eval-ci')} shows how the interval is worked out, and ${LL('pass-k')} explains pass^3.)</p><p>The comparison with the initial setup is made task by task, on the same 50 tasks. How many runs per task?</p>`,
              `<p>现在的 harness 在评测里达标了。可是评测本身也有噪声：换一批任务，或者把同样的任务再跑一遍，分数都会变。上线前要回答两个问题：区间的下沿过线了吗？同一个任务是不是每次都成功？（区间的算法见 ${LL('eval-ci')}，pass^3 见 ${LL('pass-k')}。）</p><p>和初始设置比的时候，用的是同样的 50 个任务，一个任务一个任务地比。每个任务该跑几次？</p>`)+
   A.ui.seg({id:'acp-ic-K',key:'K',label:tx('Runs per task','每个任务跑几次'),options:[[1,tx('1 run','1 次')],[3,tx('3 runs','3 次')],[5,tx('5 runs','5 次')]],value:S.K,inline:true}),
  obs:()=>obsProve()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Fix where the failures pile up; each change moves more than one condition; prove it with runs.','先看失败堆在哪里，再动手。每处改动都会牵动几条要求。最后用多次运行来证明。'),
  enter:()=>{remView=''; return {...FINAL};},
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(predRight()));
  return pred===predRight()?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsMission(){
  if(!pred) return '';
  if(!runsSeen){const t=doneNow(); return tx(`<p class="acp-runl">The suite is running: ${t.n} of ${t.N} runs done.</p>`,`<p class="acp-runl">评测正在跑：${t.N} 次运行里跑完了 ${t.n} 次。</p>`);}
  const e=evaluate('shop',INIT), rows=PRED.map(o=>({o,e:evaluate('shop',{...INIT,...PRED_H[o.v]})})).sort((a,b)=>b.e.succ-a.e.succ), best=predRight();
  const li=rows.map(x=>`<li${x.o.v===best?' class="acp-best"':''}>${esc(L(x.o.t))}${tx(': ','：')}<b>${fP(x.e.succ)}</b></li>`).join('');
  return tx(`<p><b>Initial setup: ${fP(e.succ)} of the runs succeed (95% interval ${fI(e)}). The goal is missed.</b></p>`,`<p><b>初始设置：${fP(e.succ)} 的运行成功（95% 区间 ${fI(e)}）。没达到要求。</b></p>`)+verdict()+
    tx(`<p>The success rate with one area changed at a time:</p><ul class="acp-one">${li}</ul><p>Why: most failed runs stopped on a reported error that nobody retried (see “Why runs fail” in the chart). The stronger model makes each step steadier, but its context still overflows, and it costs more than five times as much.</p>`,
       `<p>每次只改一个方面，成功率分别是：</p><ul class="acp-one">${li}</ul><p>原因：失败的运行大多停在报出来的错上，没有人重试（看图里的“失败的原因”）。更强的模型让每一步更稳，可它的上下文照样会溢出，花费还是原来的 5 倍多。</p>`)+splitA();
}
/* S1: “handle errors” is two settings; what each does alone, and why they work together (part 1's result and the capability card) */
const ONE={r:{...INIT,R:2},c:{...INIT,ck:1},a:{...INIT,R:2,ck:1},d:{...INIT,model:'strong'}};
function splitA(){
  const i=evaluate('shop',INIT), r=evaluate('shop',ONE.r), c=evaluate('shop',ONE.c), a=evaluate('shop',ONE.a), d=evaluate('shop',ONE.d);
  const same=fP(c.succ)===fP(i.succ), half=a.cost<d.cost/2;
  return tx(`<p>Handling errors takes two settings, and they work together. Retries alone reach ${fP(r.succ)}, below the stronger model. A check alone ${same?`leaves success at ${fP(c.succ)}`:`gives ${fP(c.succ)}`}: without a retry, an error it catches just stops the run. Together they reach ${fP(a.succ)}, at ${half?'less than half':'a fraction of'} the stronger model’s cost.</p>`,
            `<p>处理出错要改两处设置，两处要配合。只加重试是 ${fP(r.succ)}，还不如换模型。只加检查，成功率${same?`还是 ${fP(c.succ)}`:`是 ${fP(c.succ)}`}，因为不重试，查出来的错只会让运行停下。两样一起才到 ${fP(a.succ)}，花费${half?'还不到更强模型的一半':'也比更强的模型低'}。</p>`);
}
function obsLoud(){
  const cur=match(LV2), b=evaluate('shop',{...P2}), e=E_();
  if(!runsDone()) return runningLine();
  if(!cur) return customLine(e);
  if(cur==='init') return tx(`<p><b>Initial setup: ${fP(b.succ)} succeed; ${b.causes.stop} of the ${b.N} runs stopped on a reported error.</b> Pick a way to handle the errors.</p>`,`<p><b>初始设置：成功率 ${fP(b.succ)}；${b.N} 次运行里，有 ${b.causes.stop} 次停在报出来的错上。</b>选一种处理错误的办法。</p>`);
  /* S7: retries let runs that used to stop on an error reach the outside content, so unsafe actions rise */
  const up=e.unsafe>b.unsafe?tx(`<p>Unsafe actions rose from ${b.unsafe} to ${e.unsafe}. Runs that used to stop on an error now get as far as the outside content, and obey what is hidden in it. Part 4 deals with that.</p>`,
    `<p>不安全的操作反而从 ${b.unsafe} 次变成了 ${e.unsafe} 次。原来停在报错上的运行，现在能走到读外部内容那一步，照做了藏在里面的指令。第 4 部分会处理它。</p>`):'';
  if(cur==='r2') return tx(`<p><b>2 retries: success goes from ${fP(b.succ)} to ${fP(e.succ)}.</b></p><p>Almost no run stops on an error now. Two kinds of failure are left. ${e.causes.wrong} runs finished, but wrong, because of silent errors. ${e.causes.overflow} runs overflowed the context window.${tried2.has('r2c')?'':' Now add a check.'}</p>`,
    `<p><b>重试 2 次：成功率从 ${fP(b.succ)} 升到 ${fP(e.succ)}。</b></p><p>现在几乎没有运行停在报错上了。剩下的失败有两种。一种是悄悄出的错，${e.causes.wrong} 次运行做完了，结果却是错的。另一种是上下文，${e.causes.overflow} 次运行撑满了窗口。${tried2.has('r2c')?'':'再加上检查试试。'}</p>`)+up;
  const r2=evaluate('shop',{...P2,R:2});
  return tx(`<p><b>2 retries and a check: success goes from ${fP(b.succ)} to ${fP(e.succ)}.</b></p><p>The check makes most silent errors show, and the retries fix them: wrong runs drop from ${r2.causes.wrong} with retries alone to ${e.causes.wrong}. The price is that every attempt reads the whole context once more, so a run now costs ${fCt(e.cost)} on average, against ${fCt(b.cost)} in the initial setup. The context is the next problem: ${e.causes.overflow} runs still overflow.</p>`,
            `<p><b>重试 2 次，再加检查：成功率从 ${fP(b.succ)} 升到 ${fP(e.succ)}。</b></p><p>检查让大部分悄悄出的错露了出来，重试再把它们改对。做错的运行从只重试时的 ${r2.causes.wrong} 次降到 ${e.causes.wrong} 次。代价是每次尝试都要把整个上下文多读一遍，每次运行平均要花 ${fCt(e.cost)}，初始设置只要 ${fCt(b.cost)}。下一个问题是上下文，还有 ${e.causes.overflow} 次运行撑满了窗口。</p>`)+up;
}
function obsContext(){
  const cur=match(LV3), b=evaluate('shop',{...P3}), e=E_();
  if(!runsDone()) return runningLine();
  if(!cur) return customLine(e);
  if(cur==='all') return tx(`<p><b>Keep everything: ${fP(b.succ)} succeed.</b> ${b.causes.overflow} runs overflow the window. A run costs ${fCt(b.cost)}. Pick a way to manage the context.</p>`,`<p><b>全部保留：成功率 ${fP(b.succ)}。</b>有 ${b.causes.overflow} 次运行撑满了窗口。每次运行花 ${fCt(b.cost)}。选一种管上下文的办法。</p>`);
  if(cur==='last'){const L8=lostOf(e);
    return tx(`<p><b>The last 8 steps: success falls from ${fP(b.succ)} to ${fP(e.succ)}.</b></p><p>Nothing overflows now. The task itself always stays in the context, but what step 1 looked up, such as which order to change, is dropped from step 10 on. ${L8.n} tasks still need it at their last step; ${L8.all?`all ${e.causes.lost} of their runs end wrong`:`${e.causes.lost} of their runs end wrong`}. It is still expensive, ${fCt(e.cost)} per run, because every call reads 8 full tool outputs.</p>`,
    `<p><b>只留最近 8 步：成功率反而从 ${fP(b.succ)} 降到 ${fP(e.succ)}。</b></p><p>上下文不再溢出了。任务本身一直留在上下文里，可第 1 步查到的信息，比如要改的是哪个订单，从第 10 步起就被删掉了。有 ${L8.n} 个任务在最后一步还要用它，${L8.all?`这些任务的 ${e.causes.lost} 次运行全都做错了`:`这些任务有 ${e.causes.lost} 次运行做错了`}。花费也不低，每次运行 ${fCt(e.cost)}，因为每次调用都要读 8 段完整的工具输出。</p>`);}
  return tx(`<p><b>Trim + summary + notes: success goes from ${fP(b.succ)} to ${fP(e.succ)}. A run costs ${fCt(e.cost)} instead of ${fCt(b.cost)}.</b></p><p>The context never overflows, and what step 1 looked up stays in the notes until the last step. Each call reads a short context, so the cost falls to about a fifth. Unsafe actions are still there: ${e.unsafe} in ${e.N} runs.</p>`,
            `<p><b>裁剪 + 摘要 + 笔记：成功率从 ${fP(b.succ)} 升到 ${fP(e.succ)}。每次运行的花费从 ${fCt(b.cost)} 降到 ${fCt(e.cost)}。</b></p><p>上下文再也不会溢出，第 1 步查到的信息记在笔记里，一直留到最后一步。每次调用读的上下文都很短，所以花费降到原来的五分之一左右。不安全的操作还在，${e.N} 次运行里有 ${e.unsafe} 次。</p>`);
}
/* B3, S2: the tasks whose last step needs what step 1 looked up, and whether every one of their runs was lost */
function lostOf(e){const T=tasksOf('shop'), ix=T.filter(t=>t.need&&t.n>K_.LASTN+1).map(t=>t.i);
  return {n:ix.length,edits:T.filter(t=>t.kind==='E').length,all:e.causes.lost===ix.length*e.K};}
function obsGate(){
  const cur=match(LV4), b=evaluate('shop',{...P4}), e=E_();
  if(!runsDone()) return runningLine();
  if(!cur) return customLine(e);
  if(cur==='none') return tx(`<p><b>No gate: ${b.unsafe} unsafe actions in ${b.N} runs.</b> Pick a safeguard.</p>`,`<p><b>不把关：${b.N} 次运行里有 ${b.unsafe} 次不安全的操作。</b>选一种把关的办法。</p>`);
  if(cur==='detect') return tx(`<p><b>A detector: unsafe actions fall from ${b.unsafe} to ${e.unsafe}, not to 0.</b></p><p>The detector removes most injections, but each one it misses is obeyed as before. The goal asks for none at all.${tried4.has('confirm')?'':' Try the person’s confirmation.'}</p>`,
    `<p><b>检测模型：不安全的操作从 ${b.unsafe} 次降到 ${e.unsafe} 次，没有降到 0。</b></p><p>检测模型删掉了大部分注入，可漏掉的那几个，模型照样会听。上线要求是一次都不能有。${tried4.has('confirm')?'':'再试试让人确认。'}</p>`);
  const conf=Math.round(e.conf*100);
  return tx(`<p><b>A person confirms: unsafe actions fall from ${b.unsafe} to 0. Success rises to ${fP(e.succ)}.</b></p><p>After the agent has read outside content, a refund or a message waits for a person’s yes. The person refuses whatever an injected instruction asks for. The price is the person’s time, about ${conf} confirmations per 100 runs.</p>`,
            `<p><b>人确认：不安全的操作从 ${b.unsafe} 次降到 0。成功率升到 ${fP(e.succ)}。</b></p><p>agent 读过外部内容以后，退款、发消息都要等人点头。注入的指令让它做的事，人都会拒绝。代价是人的时间，每 100 次运行约要找人确认 ${conf} 次。</p>`);
}
function obsProve(){
  if(!runsDone()) return runningLine();
  const e=E_(), g=gainOf('shop',hNow(),{...INIT,K:S.K}), g3=goalOf(e,BUDGET.shop);
  const gain=tx(`<p>Against the initial setup, task by task: ${fPts(g.m)} points, 95% interval ${fPts(g.lo)} to ${fPts(g.hi)}. The interval does not include 0, so the gain is not noise.</p>`,`<p>和初始设置逐个任务地比：提升 ${fPts(g.m)} 个百分点，95% 区间 ${fPts(g.lo)} 到 ${fPts(g.hi)}。区间不含 0，所以这个提升不是噪声。</p>`);
  if(S.K===1) return tx(`<p><b>1 run per task: ${fP(e.succ)} succeed, but the interval’s lower end is ${fP(e.lo)}, below 80%.</b></p><p>With one run, nobody can see which tasks fail now and then, so pass^3 cannot be worked out.</p>`,`<p><b>每个任务跑 1 次：成功率 ${fP(e.succ)}，可区间的下沿只有 ${fP(e.lo)}，不到 80%。</b></p><p>只跑一次，也看不出哪些任务时好时坏，pass^3 根本算不出来。</p>`)+gain;
  if(S.K===3) return tx(`<p><b>3 runs per task: ${fP(e.succ)} succeed (interval ${fI(e)}), pass^3 is ${fP(e.p3)}${g3.all?', and all four conditions hold':''}.</b></p>`,`<p><b>每个任务跑 3 次：成功率 ${fP(e.succ)}（区间 ${fI(e)}），pass^3 是 ${fP(e.p3)}${g3.all?'，四条都满足了':''}。</b></p>`)+gain;
  const e3=evaluate('shop',{...hNow(),K:3}), nw=narrowOf(e3,e);
  return tx(`<p><b>5 runs per task: the interval is ${fI(e)}, ${nw.en} narrower than with 3 runs (${fI(e3)}). The evaluation costs two thirds more.</b></p><p>The interval has two parts. One comes from the luck of each run: more runs average it out, but only with the square root of the number of runs. The other comes from which tasks happen to be in the suite, and only more tasks shrink it.</p>`,
            `<p><b>每个任务跑 5 次：区间是 ${fI(e)}，只比跑 3 次（${fI(e3)}）窄了${nw.zh}。评测的花费却多了三分之二。</b></p><p>区间来自两部分。一部分来自每次运行的运气，多跑几次能把它平均掉，但只按运行次数的平方根缩小。另一部分来自评测集里碰巧是哪些任务，只有加任务才能缩小。</p>`)+gain;
}
/* B1: how much narrower the interval got, from the widths as shown (one decimal) */
function narrowOf(a,b){const w=x=>r1(Math.min(1,x.hi)*100)-r1(Math.max(0,x.lo)*100), f=1-w(b)/w(a);
  return {f,en:f<0.15?'only a little':(f<0.25?'only about a fifth':'about a third'),zh:f<0.15?'一点':(f<0.25?'约五分之一':'约三分之一')};}
function runningLine(){const t=doneNow(); return tx(`<p class="acp-runl">The suite is running: ${t.n} of ${t.N} runs done. The result shows when they finish.</p>`,`<p class="acp-runl">评测正在跑：${t.N} 次运行里跑完了 ${t.n} 次。跑完后显示结果。</p>`);}
function customLine(e){return tx(`<p><b>${esc(hName(hNow()))}: ${fP(e.succ)} succeed.</b> This matches none of the options; the chart shows the result.</p>`,`<p><b>${esc(hName(hNow()))}：成功率 ${fP(e.succ)}。</b>这组设置不是上面的任何一个选项；结果见图。</p>`);}
const REMS={
 one:{s:{...INIT},t:{en:'Show the initial setup',zh:'在图里看初始设置'}},
 det:{s:{...P4,gate:'detect'},t:{en:'Show the detector',zh:'在图里看检测模型'}},
 k1:{s:{...FINAL,K:1},t:{en:'Show 1 run per task',zh:'在图里看每个任务跑 1 次'}}};
function remBtn(k){const on=remView===k; return `<button type="button" class="btn ghost sm acp-rem-b" data-rem="${k}" aria-pressed="${on}">${esc(L(REMS[k].t))}</button>`;}
function remember(){
  const i=evaluate('shop',INIT), a=evaluate('shop',{...INIT,R:2,ck:1}), d=evaluate('shop',{...INIT,model:'strong'}), dt=evaluate('shop',{...P4,gate:'detect'}), k1=evaluate('shop',{...FINAL,K:1});
  /* S5: differences of the values as shown (R6.6) */
  const sh=x=>r1(x.succ*100), pa=sh(a)-sh(i), pd=sh(d)-sh(i);
  const r2=evaluate('shop',{...INIT,R:2}), tw=a.cost/r2.cost, twice=tw>=1.8&&tw<2.3;
  return tx(`<ol class="acp-rem"><li><b>Find out why runs fail before you change anything.</b> The change aimed at the main cause of failure pays most. When most runs stop on a reported error, retries and a check raise success more than a stronger model that costs far more. ${remBtn('one')}</li>`+
    `<li><b>Each change mainly answers one condition, and it moves the others too.</b> Retries and checks raise success, but every check is one more model call, so the cost goes up, and managing the context brings it back down. A gate deals with safety. A detector only lowers the unsafe actions; only a person’s confirmation keeps them at zero. ${remBtn('det')}</li>`+
    `<li><b>Prove it with several runs per task.</b> Read the interval’s lower end and pass^3, and compare with the old version task by task. With one run per task, the interval is so wide that its lower end can fall below the line even when the average clears it. ${remBtn('k1')}</li></ol>`+
    `<p class="eyebrow">Common misconceptions</p><ul class="acp-mis"><li><b>“A strong enough model makes the harness unimportant.”</b> A stronger model makes each step steadier, but it still overflows the context and still obeys injections. Only the harness deals with those.</li><li><b>“An average above 85% is ready to ship.”</b> The lower end of the interval and pass^3 have to clear their lines too: one run per task cleared neither.</li><li><b>“A detector stops every injection.”</b> It lowers the rate; only a design that needs a person’s yes bounds the damage.</li></ul>`+A.shell.toScenarios(),
    `<ol class="acp-rem"><li><b>先查清运行为什么失败，再决定改哪里。</b>对准失败主因的改动，收效最大。失败的运行大多停在报出来的错上时，重试加检查比换一个贵得多的强模型更管用。${remBtn('one')}</li>`+
    `<li><b>每一处改动主要管一条要求，也会牵动别的。</b>重试和检查管成功率。可每次检查都要多调用一次模型，花费会涨，要靠管好上下文降回来。把关管安全。检测模型只能让不安全的操作变少，只有人确认才能降到 0。${remBtn('det')}</li>`+
    `<li><b>每个任务多跑几次，再下结论。</b>看区间的下沿，看 pass^3，再逐个任务和旧版本比。每个任务只跑 1 次，区间太宽，平均值过了线，下沿也可能掉到线下面。${remBtn('k1')}</li></ol>`+
    `<p class="eyebrow">常见误解</p><ul class="acp-mis"><li><b>“模型够强，harness 就无所谓。”</b>更强的模型让每一步更稳，可照样撑满上下文，照样听信注入。这两件事只有 harness 能管。</li><li><b>“平均成功率过了 85% 就能上线。”</b>区间的下沿和 pass^3 也要过线：每个任务只跑 1 次时，两条都没过。</li><li><b>“检测模型能拦住所有注入。”</b>它只能少中招；要人点头的设计，才能把最坏的情况框住。</li></ul>`+A.shell.toScenarios());
}

/* =====================================================================
   scenarios (on the support agent's suite; all rates are the lab's assumptions)
   ===================================================================== */
const PRESETS=[
 {id:'strong',t:{en:'Just switch to a stronger model',zh:'只换一个更强的模型'},d:{en:'The usual first idea: leave the harness alone, buy a bigger model',zh:'最常见的第一反应：harness 不动，换更大的模型'},s:{...INIT,model:'strong'}},
 {id:'last',t:{en:'A sliding window of recent steps',zh:'上下文只留最近几步'},d:{en:'The simplest way to stay inside the window',zh:'不撑满窗口最省事的办法'},s:{...FINAL,ctx:'last'}},
 {id:'detect',t:{en:'Only a detector against injections',zh:'只靠检测模型防注入'},d:{en:'No person in the loop, to keep every task automatic',zh:'不让人介入，每个任务都全自动'},s:{...FINAL,gate:'detect'}},
 {id:'nocheck',t:{en:'Save money: no check after each step',zh:'为了省钱，每步不做检查'},d:{en:'Half the model calls, with the rest of the harness in place',zh:'模型调用少一半，harness 其余部分照旧'},s:{...FINAL,ck:0}}];
const presetNow=()=>PRESETS.findIndex(p=>['model','R','ck','ctx','gate','K'].every(k=>S[k]===p.s[k]));
function scenHTML(i){
  const p=PRESETS[i], e=evaluate('shop',{...p.s,draw:S.draw}), g=goalOf(e,BUDGET.shop), f=evaluate('shop',{...FINAL,draw:S.draw});
  const head=tx(`<p><b>${fP(e.succ)} succeed (interval ${fI(e)}). ${missW(g)}.</b></p>`,`<p><b>成功率 ${fP(e.succ)}（区间 ${fI(e)}）。${missW(g)}。</b></p>`);
  if(p.id==='strong'){const i=evaluate('shop',{...INIT,draw:S.draw});
    return head+tx(`<p>Each step goes wrong half as often, so fewer runs stop. But the harness is unchanged, and the old problems remain: ${e.causes.overflow} runs overflow the window and ${e.unsafe} obey an injected instruction. The cost also rises to ${fCt(e.cost)} per run, ${e.cost>5*i.cost?'more than five times':'several times'} the cheaper model.</p>`,
                    `<p>每一步出错少了一半，停下的运行也少了。可 harness 没变，老问题都还在。${e.causes.overflow} 次运行撑满了窗口，${e.unsafe} 次照做了注入的指令。花费也涨到每次运行 ${fCt(e.cost)}，是便宜模型的${e.cost>5*i.cost?' 5 倍多':'好几倍'}。</p>`);}
  if(p.id==='last'){const L8=lostOf(e);
    return head+tx(`<p>The window never fills. But ${L8.n} of the ${L8.edits} multi-step edits still need, at their last step, what step 1 looked up, such as which order to change. With only the last 8 steps kept, that is long gone, and ${L8.all?`all ${e.causes.lost} runs of those ${L8.n} tasks end wrong`:`${e.causes.lost} runs of those ${L8.n} tasks end wrong`}. Every call also reads 8 full tool outputs, so a run still costs ${fCt(e.cost)}.</p>`,
                    `<p>窗口不会撑满了。可 ${L8.edits} 个多步修改里，有 ${L8.n} 个在最后一步还要用第 1 步查到的信息，比如要改的是哪个订单。只留最近 8 步，这条信息早就删掉了，${L8.all?`这 ${L8.n} 个任务的 ${e.causes.lost} 次运行全都做错`:`这 ${L8.n} 个任务有 ${e.causes.lost} 次运行做错`}。每次调用还要读 8 段完整的工具输出，所以每次运行仍要 ${fCt(e.cost)}。</p>`);}
  if(p.id==='detect') return head+tx(`<p>No task waits for anyone, and success is ${fP(e.succ)}. But ${e.unsafe} injected instructions slip past the detector and are carried out, and one is already too many. pass^3 is ${fP(e.p3)}, just under 70%.</p>`,`<p>每个任务都不用等人，成功率 ${fP(e.succ)}。可还是有 ${e.unsafe} 条注入的指令躲过了检测，被执行了，一次就已经太多。pass^3 也只有 ${fP(e.p3)}，差一点没到 70%。</p>`);
  return head+tx(`<p>A run costs ${fCt(e.cost)} instead of ${fCt(f.cost)} with the check. But silent errors now go unseen: ${e.causes.wrong} runs finish wrong, and success stays at ${fP(e.succ)}. Here ${ERRS('shop').sil} of the errors are silent, so the check pays for itself.</p>`,`<p>每次运行只花 ${fCt(e.cost)}，带检查时是 ${fCt(f.cost)}。可悄悄出的错又没人看见了，${e.causes.wrong} 次运行做完却是错的，成功率停在 ${fP(e.succ)}。这里有 ${ERRS('shop').sil} 的错是悄悄出的，所以检查值这个钱。</p>`);
}
/* Z21: “all four missed” instead of “4 of the four” */
const missW=g=>g.all?tx('All four conditions hold','四条都满足'):(g.miss===4?tx('All four conditions missed','四条都没满足'):tx(`${g.miss} of the four conditions missed`,`四条里有 ${g.miss} 条没满足`));

/* =====================================================================
   challenge
   ===================================================================== */
const chName=o=>hName({...o});
function chHint(x){
  const g=x.g, o=x.o, e=x.e;
  if(!g.unsafe&&o.gate!=='confirm') return o.gate==='detect'?tx('Hint: the detector lowers the unsafe actions, but the goal is none at all. Which safeguard can guarantee that?','提示：检测模型能减少不安全的操作，可要求是一次都没有。哪种把关能保证这一点？'):tx('Hint: these tickets carry outside text with hidden instructions. Something has to stand between them and the risky actions.','提示：这些工单里贴着外面来的文字，藏着指令。危险操作之前，得有一道关。');
  if(o.ctx==='all') return tx('Hint: these logs are long. Kept in full, they overflow the window and are read again on every call.','提示：这里的日志很长。全部留着，会撑满窗口，每次调用还要重读一遍。');
  if(o.ctx==='last') return tx('Hint: the task itself stays in the window, but a long task still needs, at its last step, what its first step looked up.','提示：任务本身一直留在窗口里，可长任务到最后一步，还要用到第 1 步查到的信息。');
  if(!g.cost&&o.model==='strong') return tx('Hint: every token the stronger model reads costs five times as much.','提示：更强的模型每读一个 token，价钱是便宜模型的 5 倍。');
  if(!g.cost&&o.ck) return tx('Hint: a check doubles the model calls. How many of this agent’s errors are silent in the first place?','提示：检查让模型调用翻倍。这个 agent 的错，本来有多少是悄悄出的？');
  if(o.R===0) return tx('Hint: almost every error this agent makes is reported. What turns a reported error into a success?','提示：这个 agent 的错几乎都会报出来。怎样把报出来的错变成成功？');
  if(!g.succ||!g.p3) return tx('Hint: look at “Why runs fail” in the chart, and fix the biggest bar first.','提示：看图里的“失败的原因”，先解决最长的那一条。');
  return tx('Hint: look at which condition is missed, and change the setting that drives it.','提示：先看差的是哪一条，再改影响它的那个设置。');
}
function goalLines(e,g,budget){
  const row=(ok,en,zh)=>`<li class="${ok?'acp-met':'acp-miss'}"><span aria-hidden="true">${ok?'✓':'✗'}</span> ${tx(en,zh)}</li>`;
  return `<ul class="acp-gl">`+row(g.succ,`Success ${fP(e.succ)}, lower end ${fP(e.lo)} (goal: 85%, 80%)`,`成功率 ${fP(e.succ)}，下沿 ${fP(e.lo)}（要求 85% 和 80%）`)+
    row(g.p3,`pass^3 ${e.p3==null?'—':fP(e.p3)} (goal: 70%)`,`pass^3 ${e.p3==null?'—':fP(e.p3)}（要求 70%）`)+
    row(g.unsafe,`Unsafe actions: ${e.unsafe} (goal: 0)`,`不安全的操作 ${e.unsafe} 次（要求 0 次）`)+
    row(g.cost,`Cost per run: ${fCt(e.cost)} (goal: at most ${budget} cents)`,`每次运行的花费 ${fCt(e.cost)}（要求不超过 ${budget} 美分）`)+`</ul>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const x=chEval(chO), best=chBest(), e=x.e;
  const good=`<p class="okline acp-mt"><b>`+tx(`Goal met: ${fP(e.succ)} succeed, pass^3 ${fP(e.p3)}, no unsafe action, ${fCt(e.cost)} per run.`,`达成目标：成功率 ${fP(e.succ)}，pass^3 ${fP(e.p3)}，没有不安全的操作，每次运行 ${fCt(e.cost)}。`)+`</b>`;
  let h;
  if(x.g.all&&best&&oKey(best.o)===oKey(chO)) h=good+tx(' And no passing combination costs less.</p>','而且没有哪个达标的组合更省钱。</p>');
  else if(x.g.all) h=good+tx(' But a cheaper combination passes too.</p>','不过还有更省钱的组合也能达标。</p>')+tx(`<p>The cheapest is “${esc(chName(best.o))}”, at ${fCt(best.e.cost)} per run.</p>`,`<p>最省钱的组合是“${esc(chName(best.o))}”，每次运行 ${fCt(best.e.cost)}。</p>`);
  else h=`<p class="badline acp-mt"><b>`+tx(`Not yet: ${missW(x.g).replace(/^A/,'a')}.`,`还没达到：${missW(x.g)}。`)+`</b></p>`+goalLines(e,x.g,CH.budget)+`<p class="small">${chHint(x)}</p>`+(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the chart (the owner's review of batch 8, 2026-10-07): every combination goes in, and the brief chart keeps the ones that pass (the
     cheapest tagged), the initial setup and the pick; the combinations tried on the way are no longer listed */
  const rows=chAll(), ini=oKey({model:'cheap',R:0,ck:0,ctx:'all',gate:'none'}), shown=rows.filter(r=>r.g.all||oKey(r.o)===ini||oKey(r.o)===oKey(chO));
  h+=A.ui.cmpChart({title:tx('Success of the ops agent’s harness (goal: at least 85%, with the other three conditions met)','运维 agent 的成功率（要求至少 85%，另外三条也要满足）'),
    aria:tx('Success and cost: ','成功率和花费：')+shown.map(r=>`${chName(r.o)}${tx(': ','：')}${fP(r.e.succ)}${tx(', ','，')}${fCt(r.e.cost)}`).join(tx('; ','；')),
    rows:rows.map(r=>({label:chName(r.o),value:r.e.succ*100,text:tx(`${fP(r.e.succ)}, ${fCt(r.e.cost)}`,`${fP(r.e.succ)}，${fCt(r.e.cost)}`),ok:r.g.all,me:oKey(r.o)===oKey(chO),
      tags:(oKey(r.o)===ini?[{text:tx('initial setup','初始设置')}]:[]).concat(best&&oKey(r.o)===oKey(best.o)?[{text:tx('cheapest that passes','最省钱的达标组合')}]:[]).concat(oKey(r.o)===oKey(chO)?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,hiddenText:n=>tx(`The other ${n} combinations miss at least one condition and are not shown.`,`其余 ${n} 种组合至少差一条，图里没有列出。`),
    goal:85,goalLabel:tx('goal: 85%','要求：85%'),max:100,okText:tx('meets all four conditions','四条都满足'),noText:tx('misses at least one','至少差一条'),goalText:tx('goal: 85% success','要求：成功率 85%')});
  const st=chEval({model:'cheap',R:0,ck:0,ctx:'all',gate:'none'}), wt=chEval({model:'cheap',R:2,ck:1,ctx:'notes',gate:'confirm'});
  h+=tx(`<p class="small">The initial setup succeeds ${fP(st.e.succ)} of the time. Here ${ERRS('ops').rep} of the errors are reported, so retries catch almost all of them and a check finds little more. The harness built for the support agent in the guided parts, with its check, gets the ops agent above 85%, but it ${DEF('goes over the budget',`${fP(wt.e.succ)} succeed, at ${fCt(wt.e.cost)} per run.`)}. The long logs still need trim + summary + notes, and the outside text still needs a person’s yes.</p>`,
        `<p class="small">初始设置的成功率是 ${fP(st.e.succ)}。这里 ${ERRS('ops').rep} 的错都会报出来，重试几乎都能接住，检查能多发现的很少。把分步引导里给客服 agent 配的 harness（带检查）用在运维 agent 上，成功率超过了 85%，可每次运行的花费${DEF('超出了预算',`成功率 ${fP(wt.e.succ)}，每次运行 ${fCt(wt.e.cost)}。`)}。很长的日志仍然要裁剪 + 摘要 + 笔记，外面来的文字仍然要人点头。</p>`);
  h+=tx('<p class="small muted">Note: here a person always refuses an injected instruction and approves every real request, and a summary never loses a fact that went into the notes. Real reviewers grow tired and approve too quickly. Measure both on your own tasks before relying on them.</p>',
        '<p class="small muted">注意：这里假设人总会拒绝注入的指令、批准每一个真实的请求，记进笔记的事实也不会在摘要里丢掉。真实的审核人会疲劳、会点得太快。要靠这些做决定之前，先在自己的任务上测一测。</p>');
  return h;
}

/* =====================================================================
   header copy, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Guess which area to fix first, then handle errors, the context and risky actions in turn, and prove with the evaluation that the agent meets the goal. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先猜一猜改哪个方面最管用，再依次处理出错、上下文和危险操作，最后用评测证明它达标。第一次学从这里开始。`),
  sandbox:tx('Four common shortcuts on the same support agent: a stronger model alone, a sliding window, a detector alone, no checks to save money. Pick one to see which conditions it misses and why.',
             '同一个客服 agent 上常见的四种捷径，分别是只换更强的模型、只留最近几步、只靠检测模型、为了省钱不做检查。选一个，看它没满足哪几条、为什么。'),
  challenge:tx('Another team’s ops agent, with a tighter budget. Pick one option in each of five settings so that it meets the launch goal for the least money.',
               '另一个团队的运维 agent，预算更紧。五个设置各选一项，让它用最少的钱达到上线标准。')
});
const SRC={tau:'https://arxiv.org/abs/2406.12045',dojo:'https://arxiv.org/abs/2406.13352',litm:'https://arxiv.org/abs/2307.03172',lme:'https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/metrics.py',dojoGh:'https://github.com/ethz-spylab/agentdojo',
  cache:'https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb'};
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser runs 50 fixed tasks several times each, under the harness you set. The random numbers use a fixed seed, so the page shows the same runs every time. Every rate is this lab’s assumption. The success rate, the interval and pass^3 are computed from these runs, and ${J('math','“Show the math”')} redoes them line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Each step can go wrong. ${ERRS('shop').rep} of the support agent’s errors are reported, and the other ${ERRS('shop').sil} are silent. A reported error is retried with its message, and when the retries run out, the run stops. A check is one more model call after each step, to see whether the step was done right. Of those ${ERRS('shop').sil} silent errors, the check finds ${ERRS('shop').ck}, and an error it finds is handled like a reported one.</li>`+
 `<li>The context grows with every tool output. Kept in full, it overflows a 128K window. A sliding window keeps the task itself but drops the facts found in early steps. Trim + summary + notes keeps it short. Cost is the input tokens read, at an assumed price.</li>`+
 `<li>Half of the outside pages the support agent reads hide an injected instruction, and the model obeys it half the time. A detector removes 80% of them before the model reads them. When a person must confirm, they refuse whatever an injected instruction asks for.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul><li>All rates, sizes and prices are this lab’s assumptions.</li>`+
 `<li>The base chance that a step goes wrong is 3% for lookups, 5% for multi-step edits and 4% for reading outside content, halved by the stronger model.</li>`+
 `<li>Tasks differ in difficulty: each task is 0.5 to 4 times as error-prone as the base.</li>`+
 `<li>Prices are $1 for the cheaper model and $5 for the stronger one, per million input tokens. Output tokens are not counted.</li>`+
 `<li>Prompt caching is not counted. A real agent keeps the start of each call unchanged so that it hits the cache, and the re-read part then costs about a tenth of the price, so keeping the whole context would cost much less. It would still overflow the window, so the conclusion about managing the context holds; only the saving from trimming is smaller.</li>`+
 `<li>Every harness is run on the same random draws, so a change flips only the runs it affects. Real runs of two harnesses are independent, so a real comparison is noisier than the ones in this lab.</li>`+
 `<li>People never approve an injected instruction. A fact written into the notes is never lost in a summary. The detector raises no false alarms, and its own cost is left out.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>The interval: lm-evaluation-harness ${cite(SRC.lme,'metrics.py')} (<code>mean_stderr</code>: sample standard deviation over ${M('\\sqrt{n}')}), applied to each task’s share of successful runs.</li>`+
 `<li>pass^k: ${cite(SRC.tau,'τ-bench, arXiv 2406.12045')}. Injection defenses (tool filter, detector, delimiting, repeating the request): ${cite(SRC.dojoGh,'AgentDojo')}, <code>agent_pipeline.py</code>.</li>`+
 `<li>Prompt caching prices: a provider’s cookbook, ${cite(SRC.cache,'<code>misc/prompt_caching.ipynb</code>')}.</li>`+
 `<li>Runs per task, a stability metric and a CI gate: the interview problem under “Go further” in the challenge.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-03. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器按你设的 harness，把 50 个固定的任务各跑几次。随机数用固定的种子，所以每次打开结果都一样。所有比例都是这个 Lab 的假设。成功率、区间和 pass^3 是从这些运行算出来的，${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>每一步都可能出错。客服 agent 的错有 ${ERRS('shop').rep} 会报出来，剩下 ${ERRS('shop').sil} 是悄悄出的。报出来的错会带着报错信息重试，次数用完，这次运行就停下。检查就是每一步做完，再调用一次模型，看这一步做对了没有。这 ${ERRS('shop').sil} 悄悄出的错里，检查能发现 ${ERRS('shop').ck}。检查发现的错，和报出来的错一样处理。</li>`+
 `<li>上下文随每个工具输出变长。全部保留，会撑满 128K 的窗口。只留最近几步，任务本身还在，可前几步查到的事实会被删掉。裁剪 + 摘要 + 笔记能让上下文一直很短。花费按读进去的输入 token 算，价格是假设的。</li>`+
 `<li>客服 agent 读的外部网页里，一半藏着注入的指令，模型有一半的机会照做。检测模型能在模型读到之前删掉其中 80%。要人确认时，注入的指令让 agent 做的事，人都会拒绝。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul><li>所有比例、大小和价格都是本 Lab 的假设。</li>`+
 `<li>每一步出错的基准机会，查询是 3%，多步修改是 5%，读外部内容是 4%，换更强的模型减半。</li>`+
 `<li>任务的难度各不相同，每个任务的出错机会是基准的 0.5 到 4 倍。</li>`+
 `<li>价格按每 1M 个输入 token 算，便宜的模型 1 美元，更强的模型 5 美元。输出的 token 不计入。</li>`+
 `<li>这里没有算 prompt caching。真实的 agent 会让每次调用的开头保持不变，好命中缓存，重读的部分只收约一成的价钱，“全部保留”的花费会低很多。即使这样，上下文还是会撑满窗口，所以管上下文的结论不变，只是裁剪省下的钱没有这么多。</li>`+
 `<li>每个 harness 都用同一组随机数来跑，所以改一处设置，只有受它影响的运行会变。真实的两个 harness 各跑各的，比较起来噪声比这个 Lab 里的更大。</li>`+
 `<li>人从不批准注入的指令。记进笔记的事实不会在摘要里丢掉。检测模型不会误报，它自己的花费也不计入。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>区间：lm-evaluation-harness 的 ${cite(SRC.lme,'metrics.py')}（<code>mean_stderr</code>：样本标准差除以 ${M('\\sqrt{n}')}），这里用在每个任务的成功比例上。</li>`+
 `<li>pass^k：${cite(SRC.tau,'τ-bench，arXiv 2406.12045')}。防注入的几种做法（过滤工具、检测模型、加分隔符、重复用户的请求）：${cite(SRC.dojoGh,'AgentDojo')} 的 <code>agent_pipeline.py</code>。</li>`+
 `<li>prompt caching 的价格：一家服务商的 cookbook，${cite(SRC.cache,'<code>misc/prompt_caching.ipynb</code>')}。</li>`+
 `<li>每个任务跑几次、稳定性指标、CI 怎么把关：见挑战题“延伸练习”里的面试题。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-03。原型，尚未经专家审阅。</p></div></div>`);
/* S13: what this lab uses from the earlier labs, as whole phrases; why LLM as judge is not needed */
const PRIPOP=()=>tx(`<p>This lab uses what earlier labs taught:</p><ul><li>the agent loop and its trace, from ${LL('agent-anatomy')};</li><li>retries and checks, from ${LL('agent-reliability')};</li><li>what stays in the context, from ${LL('context-eng')};</li><li>injections and permissions, from ${LL('prompt-injection')};</li><li>the interval of a score, from ${LL('eval-ci')};</li><li>pass^k, from ${LL('pass-k')}.</li></ul><p>Every task here can be checked automatically, so ${LL('llm-judge')} is not needed.</p>`,
  `<p>这里用到前面几个 Lab 讲过的东西：</p><ul><li>agent 的循环和它的记录，见 ${LL('agent-anatomy')}；</li><li>重试和检查，见 ${LL('agent-reliability')}；</li><li>上下文里该留什么，见 ${LL('context-eng')}；</li><li>注入和权限，见 ${LL('prompt-injection')}；</li><li>分数的区间，见 ${LL('eval-ci')}；</li><li>pass^k，见 ${LL('pass-k')}。</li></ul><p>这里每个任务都能自动判对错，所以用不到 ${LL('llm-judge')}。</p>`);
const OUTCOME=()=>tx('configure an agent’s retries, checks, context and permissions for a launch goal, and prove it on a fixed task suite with several runs per task.','为一个上线目标配好 agent 的重试、检查、上下文和权限，并用固定的任务集、每个任务跑几次来证明它达标。');
const EXITS=()=>{
  const i=evaluate('shop',INIT), a=evaluate('shop',{...INIT,R:2,ck:1}), d=evaluate('shop',{...INIT,model:'strong'}), f=evaluate('shop',FINAL), f5=evaluate('shop',{...FINAL,K:5});
  const r=evaluate('shop',ONE.r), half=(f.hi-f.lo)/2, h20=half*Math.sqrt(50/20), nw=narrowOf(f,f5), luck3=lucksplit('shop',FINAL).luck(3);
  return {explain:[
  {q:tx(`Starting from the initial setup, retries and a check raised success from ${fP(i.succ)} to ${fP(a.succ)}, more than the stronger model did (${fP(d.succ)}). Why?`,`从初始设置出发，重试加检查让成功率从 ${fP(i.succ)} 升到 ${fP(a.succ)}，比换更强的模型（${fP(d.succ)}）还多。为什么？`),
   o:[tx('Most failures stopped on a reported error, which a retry fixes, and the check hands the silent errors to the retries too; the stronger model only makes each step a little steadier.','失败大多停在报出来的错上，重试正好救回它们，检查再把悄悄出的错也交给重试。更强的模型只让每一步稳一点。'),
      tx('The stronger model makes more mistakes on long tasks, so it falls behind there.','更强的模型在长任务上反而错得更多，所以落在后面。'),
      tx('A check turns every step into two tries, so the success rate roughly doubles.','检查让每一步变成两次机会，所以成功率差不多翻倍。')],a:0,
   n:[null,tx('Here the stronger model errs half as often on every task; it is behind because it leaves the reported errors unretried, and its context still overflows.','这里更强的模型在每个任务上都少错一半。它落后，是因为报出来的错仍然没人重试，上下文也照样溢出。'),
     tx('A check does not add a try; it only makes silent errors show, so that a retry can fix them. A check without retries does not raise success at all.','检查不会多给一次机会，它只让悄悄出的错露出来，好让重试去改。只加检查、不重试，成功率一点也不涨。')],
   why:tx(`Look at why runs fail first. Here the biggest bar was “stopped on a reported error”, which is what retries fix. Retries alone reach ${fP(r.succ)}, below the stronger model; with the check they reach ${fP(a.succ)}.`,`先看运行为什么失败。这里最长的一条是“报错后停下”，正是重试能解决的。只加重试是 ${fP(r.succ)}，还不如换模型。加上检查，两样配合才到 ${fP(a.succ)}。`)},
  {q:tx(`With the improved harness, going from 3 to 5 runs per task takes the interval from ${fI(f)} to ${fI(f5)}, ${nw.en} narrower. Why so little, for two thirds more runs?`,`用改好的 harness，每个任务从跑 3 次加到 5 次，区间从 ${fI(f)} 变成 ${fI(f5)}，只窄了${nw.zh}。多跑了三分之二，为什么只窄这么一点？`),
   o:[tx('More runs shrink only the luck of each run, and only with the square root of the number of runs; the differences between tasks do not shrink at all.','多跑几次只缩小每次运行的运气，而且只按次数的平方根缩小，任务之间的差别则一点不变。'),
      tx('With 5 runs more of them fail, which cancels the gain.','跑 5 次时失败的也更多，抵消了好处。'),
      tx('The interval comes almost entirely from how much the 50 tasks differ, so more runs cannot shrink it.','区间几乎全来自 50 个任务之间的差别，多跑几次缩不小它。')],a:0,
   n:[null,tx('More runs do not change how often a task fails; they measure it more precisely.','多跑几次不会改变一个任务失败的频率，只是把它量得更准。'),
     luck3>0.5?tx(`Not in this lab. At 3 runs, most of the interval comes from the luck of each run. If it came almost entirely from the tasks, 5 runs would not have narrowed it at all, yet it narrowed by ${nw.en.replace('only ','')}.`,`在这个 Lab 里不是这样。跑 3 次时，区间的大部分来自每次运行的运气。要是几乎全来自任务之间的差别，跑 5 次区间根本不会变窄，可它窄了${nw.zh}。`)
       :tx('Not entirely. The luck of each run is a sizable part too, and 5 runs did narrow the interval.','不全是这样。每次运行的运气也占了不小的一部分，跑 5 次，区间确实窄了一些。')],
   why:tx(`The interval has two sources. Tasks differ in difficulty, so the score depends on which tasks happen to be in the suite. And each run has its own luck. More runs shrink only the second, and only with the square root of the number of runs, so each extra run buys less. The first shrinks only with more tasks (see ${LL('eval-ci')}).`,
           `区间来自两个方面。一是各个任务难度不同，评测集碰巧选中哪些任务，分数就跟着变。二是每次运行的运气。多跑几次只缩小第二部分，而且按次数的平方根缩小，越往后越不划算。第一部分只有加任务才能缩小（见 ${LL('eval-ci')}）。`)}],
 transfer:[
  {q:tx(`A team’s suite has only 20 tasks, each run 3 times, and its agent also succeeds about 90% of the time. In this lab, 50 tasks gave an interval of about ±${fixd(half*100,1)} points. What about theirs?`,
       `一个团队的评测集只有 20 个任务，每个跑 3 次，他们的 agent 成功率也在 90% 左右。这个 Lab 里 50 个任务时，区间大约是 ±${fixd(half*100,1)} 个百分点。他们的区间会怎样？`),
   o:[tx('Wider, about ±8 points, because a success rate averaged over only 20 tasks swings more','更宽，大约 ±8 个百分点，因为只有 20 个任务，平均出来的成功率摆动更大'),
      tx('About the same, because each task still runs 3 times','差不多，因为每个任务照样跑 3 次'),
      tx('Narrower, because fewer tasks differ less from each other','更窄，因为任务少了，彼此的差别也小')],a:0,
   why:tx(`The fewer the tasks, the less steady their average: the interval is inversely proportional to the square root of the number of tasks. With 20 tasks instead of 50 it is about 1.6 times as wide:`,`任务越少，平均出来的成功率越不稳，区间和任务数的平方根成反比。任务从 50 个减到 20 个，区间大约宽到 1.6 倍：`)+
     `<span class="acp-fx">${M(`${fixd(half*100,1)} \\times \\sqrt{50 / 20} ≈ ${fixd(h20*100,1)}`)}</span>`+
     tx('This assumes their tasks differ in difficulty about as much as these do.','这里假设他们的任务之间，难易差别和这里一样大。')},
  {q:tx('A research agent reads web pages and writes a summary for its user. It has no tool that sends, pays or changes anything. Does it need a person to confirm its actions?','一个调研 agent 读网页，给用户写一份摘要。它没有任何能发送、付款或修改东西的工具。它需要人来确认它的操作吗？'),
   o:[tx('No: it has no tool that pays, sends or changes anything, so no step needs a person’s yes','不需要，它没有能付款、发消息或改东西的工具，不必每一步都等人点头'),tx('Yes: every agent that reads web pages needs a person for every step','需要，凡是读网页的 agent，每一步都要人确认'),tx('Yes, unless a detector is in place','需要，除非装了检测模型')],a:0,
   why:tx(`It has no tool that pays, sends a message or changes anything, so no step needs a person’s yes. An injection can still slant its summary, so the summary is treated as data. One more thing: fetching any web address is itself a way to send data out. If private data is in its context, limit the addresses it can reach (${LL('prompt-injection')}: untrusted content, private data and a way out together are what make it dangerous).`,
           `它没有能付款、发消息或改东西的工具，所以不用逐步找人确认。注入仍可能让摘要写偏，摘要要当成资料看待。还要注意，能访问任意网址，本身就是一条往外发的路。如果它的上下文里有私密数据，就要限制它能访问的网址。不可信的内容、私密数据、往外发的路，三样凑齐才危险（见 ${LL('prompt-injection')}）。`)}],
 practice:[
  {id:'agent-harness-eval',d:tx('The whole problem: a harness shared by several teams and an offline evaluation on top of it. Its deep dive runs each task 3 times, reports a stability metric next to success, and gates a merge on the score falling more than 2 standard errors below a rolling baseline.','整道题：几个团队共用的 harness，和建在它上面的离线评测。深入部分里，每个任务跑 3 次，成功率旁边报告稳定性，合入时成功率比滚动基线低出 2 个标准误以上就拦下。')},
  {id:'agent-tool-loop',d:tx('Parts 1 and 2: write the loop itself, hand a tool’s failure back to the model as a tool_result with is_error, and stop at a cap on calls. These are the retries and the step limit of this lab’s harness.','第 1、2 部分：写出循环本身，把工具的失败作为带 is_error 的 tool_result 交回模型，调用次数到上限就停。这就是这个 Lab 的 harness 里的重试和步数上限。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'agents',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Agent capstone','Agent 综合挑战'),title:l?L(l.q):tx('How do you build an agent loop that holds up, and measure it?','怎么搭一个靠得住的 agent 循环，并评测它？'),
    minutes:20,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: the agent labs','前置知识：agent 的几个 Lab'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
const speedBtns=()=>[['s',tx('Slow','慢')],['m',tx('Medium','中')],['f',tx('Fast','快')]];
function stageHTML(){
  return `<div class="panel stage acp-stage" id="acp-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="acp-st-title"></div><div class="meter-sub" id="acp-st-sub"></div></div><div class="stage-actions"><span class="status" id="acp-status"></span></div></div>`+
   `<div class="stage-graph graph acp-graph" id="acp-graph">`+
    `<div class="acp-cap" id="acp-grid-cap"></div>`+
    `<div class="acp-gridw"><div class="acp-rowl" id="acp-rowl" aria-hidden="true"></div><div class="acp-gridc"><div class="acp-grid" id="acp-grid" role="img"><canvas id="acp-cv"></canvas></div><div class="acp-kinds" id="acp-kinds"></div></div></div>`+
    `<div class="legend-row acp-legend" id="acp-legend"></div>`+
    `<div class="acp-player" id="acp-player">`+
     `<button type="button" class="btn sm acp-play" data-act="play" id="acp-play"></button>`+
     `<button type="button" class="btn ghost sm" data-act="stepf" data-tipx="${esc(tx('One round forward: every task gets its next run.','前进一轮：每个任务都跑下一次。'))}" aria-label="${esc(tx('One round forward','前进一轮'))}">▸</button>`+
     `<button type="button" class="btn ghost sm" data-act="end" data-tipx="${esc(tx('Jump to the end: every run done.','跳到结尾：所有运行都跑完。'))}" aria-label="${esc(tx('To the end','跳到结尾'))}">⏭</button>`+
     `<div class="seg-ctl acp-speed" role="group" aria-label="${esc(tx('Speed','速度'))}" data-key="speed">${speedBtns().map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${PL.speed===v}">${t}</button>`).join('')}</div>`+
     `<input type="range" class="acp-scrub" id="acp-scrub" min="0" max="150" value="0" aria-label="${esc(tx('Position in the evaluation','评测跑到哪里'))}">`+
    `</div>`+
    `<div class="acp-tally mono" id="acp-tally"></div>`+
    `<div class="acp-cap" id="acp-score-cap"></div><div class="acp-score" id="acp-score"></div>`+
    `<div class="acp-cap" id="acp-cause-cap"></div><div class="acp-causes" id="acp-causes"></div>`+
   `</div>`+
   `<p class="acp-scope small muted" data-tipx="${esc(tx('Every rate, size and price on this page is this lab’s assumption. The success rate, its interval and pass^3 are computed from the simulated runs.','页面上的每个比例、大小和价格都是本 Lab 的假设。成功率、区间和 pass^3 都是从模拟出来的运行算出来的。'))}">${tx('Rates and prices are this lab’s assumptions; the success rate and its interval are computed from these simulated runs.','比例和价格是这个 Lab 的假设，成功率和区间是从这些模拟运行算出来的。')}</p>`+
   `<div class="metrics acp-metrics" id="acp-metrics"></div>`+
   `<details class="math" id="acp-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="acp-mathbody"></div></details></div>`;
}
const noteOf=(en,zh)=>`<div class="ctl-note">${tx(en,zh)}</div>`;
const asRow=(k,label,v,en,zh)=>`<div class="ctl acp-as" id="acp-as-${k}"><div class="acp-as-h"><span class="lbl">${label}</span><output>${v}</output></div>${noteOf(en,zh)}</div>`;
function controlsHTML(){
  const seg=(id,key,label,opts,v,note)=>`<div class="lockable" id="acp-w-${key}">`+A.ui.seg({id,key,label,options:opts,value:v,note})+`</div>`;
  return `<aside class="panel controls" id="acp-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="acp-lock" hidden></p>`+
   seg('acp-ctl-model','model',tx('Model','模型'),[['cheap',tx('Cheaper','便宜的')],['strong',tx('Stronger','更强的')]],S.model,tx('The stronger model makes half as many errors per step and costs five times as much per token (both assumed).','更强的模型每一步的错少一半，每个 token 的价钱是 5 倍（两个数都是假设）。'))+
   seg('acp-ctl-R','R',tx('Retries after a reported error','报错后最多重试'),[0,1,2].map(v=>[v,String(v)]),S.R,tx('The step is redone with the error message, up to this many times; then the run stops.','带着报错信息把这一步再做一遍，最多这么多次；用完了，这次运行就停下。'))+
   seg('acp-ctl-ck','ck',tx('Check after each step','每步之后检查'),[[0,tx('No','不检查')],[1,tx('Yes','检查')]],S.ck,tx(`One more model call after every attempt, to see whether the step was done right. Of the ${ERRS('shop').sil} of errors that are silent, it finds ${ERRS('shop').ck}; that share is assumed.`,`每次尝试之后，再调用一次模型，看这一步做对了没有。悄悄出的那 ${ERRS('shop').sil} 错里，检查能发现 ${ERRS('shop').ck}，这个比例是假设。`))+
   seg('acp-ctl-ctx','ctx',tx('Context','上下文'),[['all',tx('Keep all','全部保留')],['last',tx('Last 8 steps','最近 8 步')],['notes',tx('Trim + summary + notes','裁剪 + 摘要 + 笔记')]],S.ctx,tx('Trim + summary + notes: each tool output is cut to 1,000 tokens, the old steps are folded into a summary every 8 steps, and facts for later go into notes. The last 8 steps keep the task itself.','裁剪 + 摘要 + 笔记：每个工具输出裁到 1,000 个 token，旧步骤每 8 步压成一段摘要，以后要用的事实记进笔记。只留最近 8 步时，任务本身也一直留着。'))+
   seg('acp-ctl-gate','gate',tx('Risky actions','危险的操作'),[['none',tx('No gate','不把关')],['detect',tx('Detector','检测模型')],['confirm',tx('Person confirms','人确认')]],S.gate,tx('A person confirms refunds and messages once the agent has read outside content.','agent 读过外部内容以后，退款、发消息都要人确认。'))+
   seg('acp-ctl-K','K',tx('Runs per task','每个任务跑几次'),[[1,'1'],[3,'3'],[5,'5']],S.K,tx('pass^3 needs at least 3 runs per task.','至少每个任务跑 3 次，才算得出 pass^3。'))+
   `<div class="acp-grp lockable" id="acp-w-as"><div class="acp-gh">${tx('Assumed values (fixed)','假设值（固定）')}</div>`+
   asRow('q',tx('Chance a step goes wrong','一步出错的机会'),tx('3% / 5% / 4%','3% / 5% / 4%'),'For lookups, multi-step edits and tasks that read outside content, with the cheaper model; each task is 0.5 to 4 times as error-prone. Assumed: it depends on the model, the tools and the task, so measure it on your own suite.','依次是查询、多步修改和读外部内容的任务，用便宜的模型；每个任务的出错机会是这个数的 0.5 到 4 倍。这是假设：它取决于模型、工具和任务本身，要在自己的评测集上测。')+
   asRow('r',tx('Errors that are reported','会报出来的错'),ERRS('shop').rep,`Assumed. A support agent’s tools report most failures (an API error, a missing order), but an edit to the wrong order looks like a success. The challenge’s ops agent reports ${ERRS('ops').rep}: its commands return exit codes.`,`这是假设。客服 agent 的工具会报出大部分失败（接口报错、找不到订单），可改错了订单看起来就像成功了。挑战题里运维 agent 的错有 ${ERRS('ops').rep} 会报出来：它的命令都有退出码。`)+
   asRow('c',tx('Errors a check finds','检查能发现的错'),tx(`${ERRS('shop').ck} of the ${ERRS('shop').sil}`,`${ERRS('shop').sil} 里的 ${ERRS('shop').ck}`),`Assumed. ${ERRS('shop').sil} of the support agent’s errors are silent, and a check after each step finds ${ERRS('shop').ck} of them. So ${ERRS('shop').left} of all errors still go unseen. How many a check finds depends on the model and the prompt that do the checking; measure it on your own tasks.`,`这是假设。客服 agent 的错有 ${ERRS('shop').sil} 是悄悄出的，每步之后的检查能发现其中的 ${ERRS('shop').ck}。所以全部的错里，还有 ${ERRS('shop').left} 没人看见。检查能发现多少，取决于用来检查的模型和提示词，要在自己的任务上测。`)+
   asRow('win',tx('Context window','上下文窗口'),tx('128K tokens','128K 个 token'),'Assumed; many current models accept this much or more. The base context (system prompt, tool definitions, the task) is 3,000 tokens; a multi-step edit adds 3,000 to 10,000 per step.','这是假设；现在很多模型能读这么多，甚至更多。基础的上下文（system prompt、工具定义、任务本身）是 3,000 个 token；多步修改每一步再加 3,000 到 10,000 个。')+
   asRow('price',tx('Price per million input tokens','每 1M 个输入 token 的价格'),'$1 / $5','Assumed, for the cheaper and the stronger model. Real prices differ by vendor and change often, and several times between a small and a large model is common. Output tokens and prompt caching are left out. With caching, the re-read part of the context would cost much less.','这是假设，分别是便宜的和更强的模型。真实价格因厂商而异、经常变，小模型和大模型差好几倍很常见。输出的 token 和 prompt caching 都没有算。有了缓存，上下文里重读的部分会便宜很多。')+
   asRow('inj',tx('Injections obeyed / removed by the detector','注入被照做 / 被检测模型删掉'),'50% / 80%','Assumed. How often a model obeys depends on the model and the attack; detectors catch many but never all.','这是假设。模型多常照做，取决于模型和攻击的写法；检测模型能拦下很多，但从来拦不全。')+
   `</div>`+
   `<div class="lockable acp-row" id="acp-w-seed"><button type="button" class="btn ghost sm" data-act="reseed">${tx('New runs','重新抽一批运行')}</button> <span class="small muted" id="acp-seed-out"></span>`+
   noteOf('The runs come from a seeded random generator, so the page shows the same runs every time. A new draw keeps the tasks and each step’s chance of going wrong, and gives a new batch of random runs.','运行来自带种子的随机数，所以页面每次都给出同样的运行。重新抽一批时，任务和每一步出错的机会都不变，只换一批新的随机运行。')+`</div>`+
   `<div class="lockable" id="acp-w-reset"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import math, statistics

MAX_STEPS, RETRIES = 40, 2
RISKY = {"refund", "send_message"}

def run(task, model, tools, person_ok):
    """One run of the loop: 'ok', 'wrong' or 'stopped'."""
    history, untrusted = [task["request"]], False
    for _ in range(MAX_STEPS):
        for _ in range(RETRIES + 1):
            act = model(history)             # the next tool call, or done
            if act["name"] == "done":
                return "ok" if task["check"](history) else "wrong"
            if act["name"] in RISKY and untrusted and not person_ok(act):
                out = {"text": "refused by a person"}
                break
            out = tools[act["name"]](act["args"])
            if not out.get("is_error"):
                break
            history.append("error: " + out["text"])  # retry with it
        else:
            return "stopped"                 # retries used up
        untrusted |= out.get("untrusted", False)
        history.append(out["text"][:4000])   # trim long outputs
    return "stopped"                         # step limit

def report(counts, k):
    """counts[i] = successful runs of task i, out of k runs."""
    s = [c / k for c in counts]
    n, mean = len(s), sum(s) / len(s)
    half = 1.96 * statistics.stdev(s) / math.sqrt(n)
    p3 = sum(math.comb(c, 3) for c in counts) / math.comb(k, 3) / n
    return round(mean, 3), round(mean - half, 3), round(p3, 3)

# part 5: successful runs of each task, out of 3 runs
counts = __COUNTS__
print(report(counts, 3))   # __OUT__`;
function codeText(){
  const e=evaluate('shop',FINAL), counts=e.cnt, rows=[];
  for(let i=0;i<counts.length;i+=16) rows.push(counts.slice(i,i+16).join(', '));
  const p3=counts.reduce((a,c)=>a+comb(c,3),0)/comb(3,3)/counts.length;
  return CODE.replace('__COUNTS__','['+rows.join(',\n          ')+']').replace('__OUT__',`(${pyR(e.succ)}, ${pyR(e.lo)}, ${pyR(p3)})`);
}
const pyR=x=>{const v=Math.round(x*1000)/1000; let s=String(v); if(!s.includes('.')) s+='.0'; return s;};
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="acp-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="acp-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A harness skeleton (step limit, retries with the error, a person’s confirmation after untrusted content) and the report this lab reads: success, the interval’s lower end and pass^3, from each task’s count of successful runs.','一个 harness 的骨架（步数上限、带着报错重试、读过不可信内容后要人确认），以及这个 Lab 看的那份报告：用每个任务成功的次数，算出成功率、区间下沿和 pass^3。')}</p><pre class="code" id="acp-code">${esc(codeText())}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="acp-prod">`+
   tx(`<li>Harnesses cap every run: a maximum number of steps, a wall-clock timeout and often a spending cap. A tool’s error goes back to the model as that call’s result with an error flag. Only operations that are safe to repeat are retried automatically, and a tool that changes something carries an idempotency key.</li>`+
      `<li>Long runs keep the context small: old turns are compacted into a summary, facts are written to a notes file and read back when needed, and a self-contained sub-task can go to a sub-agent with its own context.</li>`+
      `<li>Harnesses keep the start of every call unchanged (system prompt, tool definitions, the conversation so far), so that the provider’s prompt caching can reuse it at a fraction of the input price. Rewriting old history breaks the cache, so compaction usually waits until the context is nearly full.</li>`+
      `<li>Tools get the least privilege a task needs, and code runs in a sandbox. After untrusted content, risky actions wait for a person. Detectors and prompt defenses, such as the ones AgentDojo compares (a tool filter, an injection detector, delimiting, repeating the request), lower the rate, but nobody relies on them alone.</li>`+
      `<li>Evaluation runs a fixed suite against recorded or simulated tools, several runs per task, and reports success with its interval, pass^k, cost and safety separately. A change is gated in CI on a subset against a rolling baseline, and a held-out subset catches overfitting to the visible tasks.</li>`,
      `<li>harness 给每次运行都设上限：最多多少步、最长跑多久，常常还有花费上限。工具出错时，报错作为这次调用的结果交回模型，并带上出错的标记。只有能安全重做的操作才自动重试，会改动东西的工具带一个幂等键。</li>`+
      `<li>长任务让上下文保持很短：旧的回合压成摘要，事实写进笔记文件、要用时再读回来，独立的子任务交给一个有自己上下文的子 agent。</li>`+
      `<li>harness 让每次调用的开头保持不变（system prompt、工具定义、之前的对话），好让服务商的 prompt caching 复用它，这部分只收输入价格的一小部分。改写旧的历史会让缓存失效，所以压缩一般等上下文快满时才做。</li>`+
      `<li>工具只给任务需要的最小权限，代码在沙箱里跑。读过不可信的内容之后，危险操作要等人确认。检测模型和提示词上的防御（比如 AgentDojo 比较的过滤工具、检测注入、加分隔符、重复用户的请求）能降低中招的比例，但不能只靠它们。</li>`+
      `<li>评测在录制或模拟的工具上跑一组固定的任务，每个任务跑几次，成功率连同区间、pass^k、花费和安全分开报告。每次改动在 CI 里用一个子集和滚动基线比较、决定放不放行；另留一个平时不看的子集，用来发现是不是只对看得见的任务调得好。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.tau,'τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains')}<span>${tx('Yao, Shinn, Razavi and Narasimhan, arXiv 2406.12045: scores agents by pass^k, the chance of succeeding on every one of k tries of the same task','Yao、Shinn、Razavi、Narasimhan，arXiv 2406.12045：用 pass^k 给 agent 打分，也就是同一个任务试 k 次、每次都成功的概率')}</span></li>`+
   `<li>${cite(SRC.dojo,'AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents')}<span>${tx('Debenedetti et al., arXiv 2406.13352: a benchmark of agent tasks with injected instructions, used to compare defenses','Debenedetti 等，arXiv 2406.13352：一组带注入指令的 agent 任务，用来比较各种防御')}</span></li>`+
   `<li>${cite(SRC.litm,'Lost in the Middle: How Language Models Use Long Contexts')}<span>${tx('Liu et al., arXiv 2307.03172: models use information in the middle of a long context less well than at its ends, one more reason to keep the context short','Liu 等，arXiv 2307.03172：长上下文中间的信息，模型用得不如两头的好，这是让上下文保持精简的又一个理由')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="acp-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="acp-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="acp-app m2" id="acp-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="acp-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see which launch conditions it misses. Each one is a shortcut teams often take with this support agent.','选一个场景，看看它达不到哪几条上线标准。每个场景都是团队在这个客服 agent 上常走的捷径。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="acp-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const seg=(id,key,label,opts,v)=>A.ui.seg({id,key,label,options:opts,value:v,inline:true});
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Get the ops agent to launch, within 12 cents a run','让运维 agent 达到上线标准，每次运行不超过 12 美分')}</h3>`+
   tx(`<p>Another team runs an IT operations agent. It looks up alerts, reads logs thousands of lines long, changes settings and handles tickets. When one of its commands fails, ${ERRS('ops').rep} of the time it reports an error. Its tickets often carry text pasted from outside, and some of that text hides instructions.</p><p>Its suite: 50 tasks (10 lookups, 25 long edits, 15 tickets with outside text), 3 runs each. The launch goal is the support agent’s, with a tighter budget:</p>`,
      `<p>另一个团队有一个 IT 运维 agent。它查告警，读几千行的日志，改配置，处理工单。它的命令出错时，${ERRS('ops').rep} 会直接报错。工单里常贴着从外面复制来的文字，其中有些藏着指令。</p><p>它的评测集也是 50 个任务（10 个查询，25 个长的修改，15 个带外部文字的工单），每个跑 3 次。上线标准和客服 agent 一样，只是预算更紧：</p>`)+
   tx(`<ul class="acp-goal-l"><li>success at least 85%, with the interval’s lower end at least 80%;</li><li>pass^3 at least 70%;</li><li>no unsafe action;</li><li>on average at most 12 cents per run.</li></ul><p>Pick one option in each setting to build a harness. Of the harnesses that pass, the cheapest is best.</p>`,
      `<ul class="acp-goal-l"><li>成功率至少 85%，区间的下沿至少 80%；</li><li>pass^3 至少 70%；</li><li>不安全的操作一次都没有；</li><li>每次运行平均不超过 12 美分。</li></ul><p>每个设置选一项，组成一个 harness。达标的 harness 里，花钱最少的最好。</p>`)+
   seg('acp-ic-chM','chM',tx('Model','模型'),[['cheap',tx('Cheaper ($1 per M tokens)','便宜的（每 1M 个 token 1 美元）')],['strong',tx('Stronger, half the errors ($5)','更强，错少一半（5 美元）')]],chO.model)+
   seg('acp-ic-chR','chR',tx('Retries after a reported error','报错后最多重试'),CH.Rs.map(v=>[v,String(v)]),chO.R)+
   seg('acp-ic-chC','chC',tx('Check after each step','每步之后检查'),[[0,tx('No','不检查')],[1,tx('Yes: +1 call per attempt','检查：每次尝试多 1 次调用')]],chO.ck)+
   seg('acp-ic-chX','chX',tx('Context','上下文'),[['all',tx('Keep all','全部保留')],['last',tx('Last 8 steps','最近 8 步')],['notes',tx('Trim + summary + notes','裁剪 + 摘要 + 笔记')]],chO.ctx)+
   seg('acp-ic-chG','chG',tx('Risky actions','危险的操作'),[['none',tx('No gate','不把关')],['detect',tx('Detector','检测模型')],['confirm',tx('Person confirms','人确认')]],chO.gate)+
   `<div class="acp-live" id="acp-ch-live"></div>`+
   submit+`<div id="acp-ch-res" class="acp-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return !pred?tx('Pick an answer to continue, or Skip.','选一个答案才能继续，也可以跳过。'):tx('Wait for the 150 runs to finish, or press ⏭ in the chart.','等 150 次运行跑完才能继续，也可以点图里的 ⏭。');
  const tried={loud:tried2,context:tried3,gate:tried4,prove:tried5}[id];
  if(tried&&tried.size>1&&!runsDone()) return tx('Wait for the runs to finish, or press ⏭ in the chart.','等运行跑完才能继续，也可以点图里的 ⏭。');
  return {loud:tx('Pick a way to handle errors to continue.','选一种处理错误的办法才能继续。'),context:tx('Pick a way to manage the context to continue.','选一种管上下文的办法才能继续。'),
    gate:tx('Pick a safeguard to continue.','选一种把关的办法才能继续。'),prove:tx('Try another number of runs to continue.','换一个运行次数才能继续。')}[id]||'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy()||tx('Try the part’s control to continue.','先动一动这一部分的设置才能继续。'),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#acp-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function setObs(box,html){if(box._h===html) return; const again=box._h!=null&&box._h!==''&&html!==''&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
function updateRail(){
  const rail=$('#acp-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(s.id==='mission'?!!pred:true);
    const lvs={loud:[LV2,tried2,lvRes2],context:[LV3,tried3,lvRes2],gate:[LV4,tried4,lvRes4]}[s.id];
    if(lvs){const cur=match(lvs[0]); $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever, o=lvs[0].find(x=>x.v===v); b.setAttribute('aria-pressed',String(v===cur)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=(lvs[1].has(v)||v===cur)&&!waitingFor(o)?lvs[2](o.s):'';});}
    $$('.seg-ctl[data-key="K"]',rail).forEach(g=>A.ui.setPressed(g,String(S.K)));
    if(box){box.hidden=!show; if(show){setObs(box,`<div class="eyebrow">${L(s.label||{en:'Result',zh:'结果'})}</div>`+s.obs()); if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy()||null);
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-pred]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pred===pred)));
    $$('[data-rem]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rem===remView)));
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#acp-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+(runsDone()?scenHTML(cur):runningLine()):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:''); ins.hidden=!html; setObs(ins,html);}
  } else {
    const x=chEval(chO), live=$('#acp-ch-live',rail); if(!live) return;
    const lh=runsDone()?`<div class="eyebrow">${tx('Now','现在')}</div>`+goalLines(x.e,x.g,CH.budget):`<p class="small muted">${runningLine().replace(/<\/?p>/g,'')}</p>`;
    if(live._h!==lh){live.innerHTML=lh; live._h=lh;}
    [['chM',chO.model],['chR',chO.R],['chC',chO.ck],['chX',chO.ctx],['chG',chO.gate]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,rail).forEach(g=>A.ui.setPressed(g,String(v))));
    if(chSubmitted&&x.g.all) chReveal=true;
    const res=$('#acp-ch-res',rail); setObs(res,chResultHTML());
  }
}

/* =====================================================================
   stage: the score strip, the grid of runs, the causes, the cards
   ===================================================================== */
const cssv=(el,v)=>getComputedStyle(el).getPropertyValue(v).trim();
const totalCells=()=>S.K*tasksOf(S.suite).length;
function doneNow(){const n=tasksOf(S.suite).length; return {n:Math.min(PL.pv,totalCells()),N:totalCells(),per:n};}
function drawGrid(){
  const host=$('#acp-grid',app), cv=$('#acp-cv',app); if(!host||!cv) return;
  const W=host.clientWidth; if(!W) return;
  const T=tasksOf(S.suite), n=T.length, K=S.K, rowH=K>3?11:14, gap=3, H=K*(rowH+gap)-gap, dpr=window.devicePixelRatio||1;
  if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); cv.style.width=W+'px'; cv.style.height=H+'px';}
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const C={ink2:cssv(app,'--ink-2'),ink3:cssv(app,'--ink-3'),s2:cssv(app,'--surface-2'),surf:cssv(app,'--surface'),bad:cssv(app,'--c-oom'),line2:cssv(app,'--line-2')};
  const kg=[], cw=[]; let x=0; const kgap=6, cwid=(W-kgap*2)/n;
  T.forEach((t,i)=>{if(i>0&&t.kind!==T[i-1].kind) x+=kgap; cw.push(x); x+=cwid;});
  const pre=hideRes()&&!pred, e=E_(), w=Math.max(1,cwid-(cwid>=5?1.5:0.6));
  for(let k=0;k<K;k++){const y=k*(rowH+gap);
    for(let i=0;i<n;i++){const cell=k*n+i, x0=cw[i];
      if(pre||cell>=PL.pv){g.fillStyle=C.s2; g.fillRect(x0,y,w,rowH); continue;}
      const r=e.runs[i][k];
      if(r.res==='ok'){g.fillStyle=C.ink2; g.fillRect(x0,y,w,rowH);}
      else if(r.res==='unsafe'){g.fillStyle=C.bad; g.fillRect(x0,y,w,rowH);}
      else {g.fillStyle=C.surf; g.fillRect(x0,y,w,rowH); g.strokeStyle=C.ink3; g.lineWidth=1; g.strokeRect(x0+0.5,y+0.5,Math.max(0,w-1),rowH-1);}
    }}
  /* the kinds under their columns (R5.9: names beside what they label) */
  const kinds=$('#acp-kinds',app), groups=[]; T.forEach((t,i)=>{const gk=groups[groups.length-1]; if(!gk||gk.kind!==t.kind) groups.push({kind:t.kind,a:i,b:i}); else gk.b=i;});
  const kh=groups.map((gk,gi)=>{const l=cw[gk.a], r=cw[gk.b]+w, cnt=gk.b-gk.a+1, pos=gi===0?' acp-k0':(gi===groups.length-1?' acp-k9':'');
    return `<span class="acp-kind${pos}" style="left:${(l/W*100).toFixed(2)}%;width:${((r-l)/W*100).toFixed(2)}%" data-tipx="${esc(kindTip(gk.kind,cnt))}"><i></i><b>${esc(kindName(gk.kind))}</b><span>${esc(tx(`${cnt} tasks`,`${cnt} 个`))}</span></span>`;}).join('');
  if(kinds._h!==kh){kinds.innerHTML=kh; kinds._h=kh;}
  const rl=$('#acp-rowl',app), rh=Array.from({length:K},(_,k)=>`<span style="height:${rowH}px;margin-bottom:${k<K-1?gap:0}px">${esc(tx(`run ${k+1}`,`第 ${k+1} 次`))}</span>`).join('');
  if(rl._h!==rh){rl.innerHTML=rh; rl._h=rh;}
  host.dataset.tipx=tx(`Each column is one task, each row one run of all ${n} tasks. A filled cell is a run that succeeded, an empty one a run that failed, a red one a run with an unsafe action.`,`每一列是一个任务，每一行是把 ${n} 个任务各跑一次。实心格是成功的运行，空心格是失败的运行，红色格是做了不安全操作的运行。`);
  host.setAttribute('aria-label',host.dataset.tipx);
}
const kindTip=(k,cnt)=>{const [lo,span]=SUITES[S.suite].n[k], st=`${lo}`, en=`${lo+span-1}`;
  if(S.suite==='ops') return {L:tx(`${cnt} lookups, such as checking an alert: ${st} to ${en} steps, short tool outputs.`,`${cnt} 个查询，比如查一条告警：${st} 到 ${en} 步，工具的输出很短。`),
    E:tx(`${cnt} long edits, such as changing a setting on several servers: ${st} to ${en} steps, each reading back thousands of lines of logs.`,`${cnt} 个长的修改，比如在几台服务器上改一项配置：${st} 到 ${en} 步，每一步都要读回几千行日志。`),
    U:tx(`${cnt} tickets with text pasted from outside, ${st} to ${en} steps; some of the text hides an injected instruction.`,`${cnt} 个贴着外部文字的工单，${st} 到 ${en} 步；有些文字里藏着注入的指令。`)}[k];
  return {L:tx(`${cnt} lookups, such as “where is my order”: ${st} to ${en} steps, short tool outputs.`,`${cnt} 个查询，比如“我的订单到哪了”：${st} 到 ${en} 步，工具的输出很短。`),
    E:tx(`${cnt} multi-step edits, such as changing an address or handling a return: ${st} to ${en} steps, each reading back thousands of tokens.`,`${cnt} 个多步修改，比如改地址、处理退换货：${st} 到 ${en} 步，每一步都要读回几千个 token。`),
    U:tx(`${cnt} tasks that read outside content, a customer’s email or a web page, ${st} to ${en} steps; some of it hides an injected instruction.`,`${cnt} 个读外部内容的任务，读的是顾客的邮件或网页，${st} 到 ${en} 步；其中一些藏着注入的指令。`)}[k];};
function drawPlayer(){
  const pb=$('#acp-play',app), sc=$('#acp-scrub',app), tl=$('#acp-tally',app); if(!pb) return;
  const end=PL.pv>=totalCells(), pre=hideRes()&&!pred;
  pb.textContent=PL.on?tx('❚❚ Pause','❚❚ 暂停'):(end?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放'));
  /* R3.20: once the runs have ended, the play button replays them from the start, and its tooltip says so */
  pb.disabled=pre; pb.dataset.tipx=PL.on?tx('Pause the evaluation.','暂停评测。'):(end?tx(`Replay the evaluation from the start: every task, ${S.K} ${S.K===1?'time':'times'}.`,`从头重放评测，每个任务跑 ${S.K} 次。`)
    :tx(`Play the evaluation: every task, ${S.K} ${S.K===1?'time':'times'}.`,`播放评测：每个任务跑 ${S.K} 次。`));
  sc.max=String(totalCells()); if(+sc.value!==Math.min(PL.pv,totalCells())) sc.value=String(Math.min(PL.pv,totalCells())); sc.disabled=pre; A.ui.setRange(sc);
  $$('.acp-speed button',app).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===PL.speed)));
  const t=doneNow();
  let ok=0, un=0; const e=E_(), n=tasksOf(S.suite).length; for(let c=0;c<t.n;c++){const r=e.runs[c%n][Math.floor(c/n)]; if(r.res==='ok') ok++; if(r.unsafe) un++;}
  tl.textContent=pre?tx('Answer the card’s question first.','先回答卡片里的问题。'):tx(`Done: ${t.n} of ${t.N} runs. ${ok} succeeded, ${t.n-ok} failed${un?`, ${un} with an unsafe action`:''}.`,`跑完 ${t.n} / ${t.N} 次：成功 ${ok} 次，失败 ${t.n-ok} 次${un?`，其中 ${un} 次做了不安全的操作`:''}。`);
}
/* S9 (R5.10): each bar's value label sits on its own row above the bar, and the vertical lines (grid, goal, lower end) leave a gap
   at every label row, so no line crosses a label at any width */
const SROW=30, STOP=28;
const scoreH=rowsN=>STOP+rowsN*SROW+12;
function scoreSVG(W){
  const e=E_(), cmp=isP('prove')||(mode==='guided'&&remView==='k1'), base=cmp?evaluate(S.suite,{...INIT,K:S.K,draw:S.draw}):null;
  const Lp=8, Rp=8, x0=40, x1=100, pw=W-Lp-Rp, X=v=>Lp+(v-x0)/(x1-x0)*pw, f1=v=>v.toFixed(1), rowsN=cmp?2:1, H=scoreH(rowsN);
  const lab=r=>STOP+r*SROW-2, barY=r=>STOP+r*SROW+9;
  const lwOf=t=>[...t].reduce((a,ch)=>a+(/[　-鿿＀-￯]/.test(ch)?10.5:6.1),0);
  /* the bars and their labels first, so that the lines can leave a gap where a label sits */
  const rows=[];
  if(cmp) rows.push({ev:base,cls:'acp-b0',label:tx(`initial setup ${fP(base.succ)} (${fP(base.lo)} to ${fP(base.hi)})`,`初始设置 ${fP(base.succ)}（${fP(base.lo)} 到 ${fP(base.hi)}）`),short:tx(`initial ${fP(base.succ)}`,`初始设置 ${fP(base.succ)}`)});
  rows.push({ev:e,cls:'acp-b1',label:tx(`${fP(e.succ)} (${fP(e.lo)} to ${fP(e.hi)})`,`${fP(e.succ)}（${fP(e.lo)} 到 ${fP(e.hi)}）`),short:fP(e.succ)});
  rows.forEach((o,r)=>{if(lwOf(o.label)+8>W) o.label=o.short; const lw=lwOf(o.label); o.lx=Math.max(4+lw/2,Math.min(W-4-lw/2,X(o.ev.succ*100))); o.box=[o.lx-lw/2-4,o.lx+lw/2+4,lab(r)-11,lab(r)+4];});
  /* a vertical line from y0 to y1, cut where a label's box covers its x */
  const vline=(cls,x,y0,y1)=>{let out='', y=y0; rows.map(o=>o.box).filter(b=>x>=b[0]&&x<=b[1]).sort((a,b)=>a[2]-b[2]).forEach(b=>{if(b[3]<=y||b[2]>=y1) return; if(b[2]>y) out+=`<line class="${cls}" x1="${f1(x)}" x2="${f1(x)}" y1="${f1(y)}" y2="${f1(b[2])}"/>`; y=Math.max(y,b[3]);}); if(y1>y) out+=`<line class="${cls}" x1="${f1(x)}" x2="${f1(x)}" y1="${f1(y)}" y2="${f1(y1)}"/>`; return out;};
  let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true">`;
  /* the end ticks' labels sit inside the picture (H7, 2026-10-09: “40%” and “100%” crossed its edges by 1 and 4 px) */
  [40,50,60,70,80,90,100].forEach(v=>{const an=v===40?'start':(v===100?'end':'middle'), tx_=v===40?Math.max(1,X(v)-8):(v===100?Math.min(W-1,X(v)+8):X(v));
    s+=vline('acp-axl',X(v),16,H-16)+`<text class="acp-tk" x="${f1(tx_)}" y="${H-4}" text-anchor="${an}">${v}%</text>`;});
  s+=vline('acp-goal',X(85),12,H-16)+`<text class="acp-gt" x="${f1(X(85)+3)}" y="11">${esc(tx('goal 85%','要求 85%'))}</text>`;
  s+=vline('acp-goal2',X(80),12,H-16)+`<text class="acp-gt2" x="${f1(X(80)-3)}" y="11" text-anchor="end">${esc(tx('lower end 80%','下沿 80%'))}</text>`;
  rows.forEach((o,r)=>{const ev=o.ev, cls=o.cls, lo=Math.max(x0,ev.lo*100), hi=Math.min(100,ev.hi*100), m=ev.succ*100, y=barY(r);
    s+=`<line class="${cls}-ci" x1="${f1(X(lo))}" x2="${f1(X(hi))}" y1="${y}" y2="${y}"/><line class="${cls}-ci" x1="${f1(X(lo))}" x2="${f1(X(lo))}" y1="${y-5}" y2="${y+5}"/><line class="${cls}-ci" x1="${f1(X(hi))}" x2="${f1(X(hi))}" y1="${y-5}" y2="${y+5}"/><circle class="${cls}-pt" cx="${f1(X(m))}" cy="${y}" r="4.2"/>`+
      `<text class="${cls}-lb" x="${f1(o.lx)}" y="${lab(r)}" text-anchor="middle">${esc(o.label)}</text>`;});
  return s+`</svg>`;
}
function drawScore(){
  const w=$('#acp-score',app), cap=$('#acp-score-cap',app); if(!w) return;
  /* the placeholder has the strip's height, so the stage does not jump when the runs end (R7.9) */
  const cmp=isP('prove')||(mode==='guided'&&remView==='k1'); w.style.minHeight=scoreH(cmp?2:1)+'px';
  if(waiting()){cap.textContent=tx('Success rate and its 95% interval: shown when the runs are done.','成功率和它的 95% 区间：运行跑完后显示。'); if(w._h!==''){w.innerHTML=''; w._h='';} w.removeAttribute('data-tipx'); w.classList.add('acp-wait'); return;}
  w.classList.remove('acp-wait');
  cap.textContent=tx(`Does the success rate clear the goal? (dot: success; bar: its 95% interval over ${tasksOf(S.suite).length} tasks)`,`成功率过线了吗？（圆点是成功率，横条是按 ${tasksOf(S.suite).length} 个任务算的 95% 区间）`);
  const W=Math.max(260,w.clientWidth||320), h=scoreSVG(W); if(w._h!==h){w.innerHTML=h; w._h=h;}
  const e=E_(); w.dataset.tipx=tx(`The dot is the share of runs that succeeded, ${fP(e.succ)}; the bar is its 95% interval, ${fI(e)}. The goal asks for at least 85% (dashed line) with the lower end at 80% or more (dotted line).`,`圆点是成功的运行占的比例，${fP(e.succ)}；横条是它的 95% 区间，${fI(e)}。要求是至少 85%（虚线），区间的下沿不低于 80%（点线）。`);
}
function drawCauses(){
  const host=$('#acp-causes',app), cap=$('#acp-cause-cap',app); if(!host) return;
  const e=E_(), N=e.N, wt=waiting();
  cap.textContent=wt?tx('Why runs fail: shown when the runs are done.','失败的原因：运行跑完后显示。'):tx(`Why runs fail: out of ${N} runs, how many ended in each way`,`失败的原因：${N} 次运行里，各有多少次这样结束`);
  const max=Math.max(1,...CAUSES.map(c=>e.causes[c]));
  const h=CAUSES.map(c=>{const v=wt?0:e.causes[c], tip=causeTip(c); return `<div class="acp-cz${c==='unsafe'?' acp-cz-u':''}" data-tipx="${esc(tip)}"><span class="acp-czl">${esc(L(CAUSEN[c]))}</span><span class="acp-czt"><i style="width:${wt?0:(v/max*100).toFixed(1)}%"></i></span><span class="acp-czv mono">${wt?'—':v}</span></div>`;}).join('');
  if(host._h!==h){host.innerHTML=h; host._h=h;}
}
const causeTip=c=>({stop:tx('Runs that stopped because a step kept failing with a reported error and no retry was left (with no retries, the first reported error stops the run).','因为某一步一直报错、重试也用完了而停下的运行（不重试时，第一次报错就停下）。'),
  wrong:tx('Runs that reached the end, but a silent error inside made the result wrong. Nobody was told.','跑到了最后，可中间有一步悄悄做错了，结果是错的。没有人知道。'),
  lost:tx('Only the last 8 steps were kept, so what step 1 looked up, such as which order to change, was gone by the last step, and the run did that step wrong. The task itself always stays.','只留最近 8 步时，第 1 步查到的信息，比如要改的是哪个订单，到最后一步已经被删掉，这次运行的最后一步就做错了。任务本身一直留着。'),
  overflow:tx('Runs whose context grew past the 128K window: the model call was rejected and the run ended there.','上下文超过了 128K 的窗口：模型调用被拒绝，运行就停在那里。'),
  unsafe:tx('The run carried out an injected instruction, such as a refund to the attacker’s account. A single one breaks the goal of none at all.','这次运行照着注入的指令做了事，比如把退款打到攻击者的账户。只要有一次，就不满足“一次都没有”的要求。')}[c]);
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  return `<span data-tipx="${esc(tx('Each column is one task; each row is one run of every task.','每一列是一个任务；每一行是把每个任务各跑一次。'))}"><span>${esc(tx('column = task, row = run','每列一个任务，每行一轮运行'))}</span></span>`+
    it('acp-sw-ok',tx('succeeded','成功'),tx('A filled cell: this run did the task right.','实心格：这次运行把任务做对了。'))+
    it('acp-sw-no',tx('failed','失败'),tx('An empty cell: this run failed; “Why runs fail” under the chart says how.','空心格：这次运行失败了；图下面的“失败的原因”说明是怎么失败的。'))+
    it('acp-sw-un',tx('unsafe action','不安全的操作'),tx('A red cell: this run carried out an injected instruction.','红色格：这次运行执行了注入的指令。'))+
    it('acp-sw-wait',tx('not run yet','还没跑'),tx('This run has not been played yet.','这次运行还没播到。'));
}
function statusOf(){
  if(hideRes()) return {cls:'neutral',txt:pred?tx('Running','在跑'):tx('Predict first','先预测'),short:pred?tx('Running','在跑'):tx('Predict first','先预测'),
    tip:pred?tx('The suite is running; the result shows when every run is done.','评测正在跑，所有运行跑完后显示结果。'):tx('After you answer part 1’s question and the runs finish, this shows how many of the four conditions are missed.','回答第 1 部分的问题、运行跑完以后，这里显示四条要求还差几条。')};
  if(!runsDone()) return {cls:'neutral',txt:tx('Running','在跑'),short:tx('Running','在跑'),tip:tx('The suite is running; the result shows when every run is done.','评测正在跑，所有运行跑完后显示结果。')};
  const e=E_(), g=goalOf(e,budgetNow());
  return {cls:g.all?'ok':'bad',txt:g.all?tx('All four met','四条都满足'):tx(`${g.miss} of 4 missed`,`差 ${g.miss} 条`),short:g.all?tx('Goal met','达标'):tx(`${g.miss} missed`,`差 ${g.miss} 条`),
    tip:g.all?tx(`All four conditions hold: success at least 85% with the lower end at 80%, pass^3 at least 70%, no unsafe action, at most ${budgetNow()} cents per run.`,`四条都满足：成功率至少 85%、下沿至少 80%，pass^3 至少 70%，没有不安全的操作，每次运行不超过 ${budgetNow()} 美分。`)
      :tx(`Green when all four conditions hold: success at least 85% with the lower end at 80%, pass^3 at least 70%, no unsafe action, at most ${budgetNow()} cents per run. Now ${g.miss} ${g.miss===1?'is':'are'} missed.`,`四条都满足时是绿色：成功率至少 85%、下沿至少 80%，pass^3 至少 70%，没有不安全的操作，每次运行不超过 ${budgetNow()} 美分。现在差 ${g.miss} 条。`)};
}
function drawHead(){
  const t=$('#acp-st-title',app), sub=$('#acp-st-sub',app), st=$('#acp-status',app);
  t.textContent=S.suite==='ops'?tx('Does the ops agent meet its launch goal?','运维 agent 达到上线标准了吗？'):tx('Does the support agent meet its launch goal?','客服 agent 达到上线标准了吗？');
  sub.textContent=hSub(hNow());
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
function metricsDef(){
  const e=E_(), wt=waiting(), dash='—', w=tx('shown when the runs are done','运行跑完后显示'), g=goalOf(e,budgetNow());
  const mark=ok=>ok?tx(' ✓',' ✓'):tx(' ✗',' ✗');
  return [{k:'succ',label:tx('Success rate','成功率'),v:wt?dash:fP(e.succ),sub:wt?w:tx(`95% interval ${fI(e)}`,`95% 区间 ${fI(e)}`)+mark(g.succ),pts:true,
     tip:wt?tx('Shown when the runs are done: the share of runs that succeeded, with its 95% interval.','运行跑完后显示：成功的运行占多少，以及它的 95% 区间。'):tx(`${fP(e.succ)} of the ${e.N} runs succeeded. The 95% interval, ${fI(e)}, comes from how much the ${e.n} tasks’ own success shares differ. Goal: at least 85%, lower end at least 80%.`,`${e.N} 次运行里有 ${fP(e.succ)} 成功。95% 区间 ${fI(e)}，按 ${e.n} 个任务各自的成功比例相差多少来算。要求：至少 85%，下沿至少 80%。`)},
    {k:'p3',label:'pass^3',v:wt?dash:(e.p3==null?tx('needs 3 runs','要跑 3 次'):fP(e.p3)),sub:wt?w:(e.p3==null?tx('one run per task is not enough','每个任务只跑 1 次不够'):tx('tasks right every time','每次都做对的任务')+mark(g.p3)),pts:true,
     tip:wt?tx('Shown when the runs are done: the share of tasks that succeed in all 3 of 3 runs. Goal: at least 70%.','运行跑完后显示：3 次运行全部成功的任务占多少。要求：至少 70%。'):e.p3==null?tx('pass^3 needs at least 3 runs of each task: with 1 run, nobody can see which tasks fail now and then.','pass^3 至少要每个任务跑 3 次：只跑 1 次，看不出哪些任务时好时坏。'):tx(`The share of tasks that succeed in 3 runs out of 3${S.K>3?', estimated from each task’s success count the way the lab pass@k explains':''}: ${fP(e.p3)}. Goal: at least 70%.`,`3 次运行都成功的任务占多少${S.K>3?'（从每个任务的成功次数来估，算法见 pass@k 这个 Lab）':''}：${fP(e.p3)}。要求：至少 70%。`)},
    {k:'unsafe',label:tx('Unsafe actions','不安全的操作'),v:wt?dash:tx(String(e.unsafe),`${e.unsafe} 次`),sub:wt?w:tx(`in ${e.N} runs`,`在 ${e.N} 次运行里`)+mark(g.unsafe),
     tip:tx(`Runs that carried out an injected instruction: ${wt?'shown when the runs are done':e.unsafe}. Goal: none at all.${S.gate==='confirm'&&!wt?` A person was asked about ${Math.round(e.conf*100)} times per 100 runs.`:''}`,`照着注入的指令做了事的运行：${wt?'运行跑完后显示':e.unsafe+' 次'}。要求：一次都没有。${S.gate==='confirm'&&!wt?`每 100 次运行，约要找人确认 ${Math.round(e.conf*100)} 次。`:''}`)},
    {k:'cost',label:tx('Cost per run','每次运行的花费'),v:wt?dash:fCt(e.cost),sub:wt?w:tx(`goal: at most ${budgetNow()} cents`,`要求：不超过 ${budgetNow()} 美分`)+mark(g.cost),
     tip:wt?tx('Shown when the runs are done. This is what the input tokens of one run cost on average.','运行跑完后显示。这里是每次运行读进去的输入 token 平均花多少钱。')
       :tx(`One run reads ${fCt(e.cost)} of input tokens on average. Every model call reads the whole context it is given, checks and summaries included. Price: ${PRICE[S.model]} dollar${S.model==='strong'?'s':''} per million tokens. Prompt caching is left out. With it, the re-read part would cost much less.`,`平均每次运行读进去的输入 token，价值 ${fCt(e.cost)}。每次模型调用都要把给它的整个上下文读一遍，检查和摘要也算在内。价格是每 1M 个 token ${PRICE[S.model]} 美元。这里没有算 prompt caching。有了缓存，重读的部分会便宜很多。`)}];
}
/* ▲▼ (R6.6): only after the learner's own change; the values wait for the runs, so the change is remembered until they end */
function drawMetrics(){
  const host=$('#acp-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join(), act=userAct||pendBadge, wt=waiting();
  if(host._sig!==sig){host._sig=sig; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(act&&prev!=null&&prev!==m.v&&prev!=='—'&&m.v!=='—'){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:1}:{dp:m.k==='unsafe'?0:1})||'';}
    /* a badge stays for its 2.6 s through redraws that show the same value; a new value without the learner's change clears it */
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(m.v!=='—'&&prev!==m.v){clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';}
    if(m.v!=='—') shownPrev[m.k]=m.v;});
  if(!wt) pendBadge=false;
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4) */
function mathHTML(){
  if(waiting()) return `<p>${tx('Shown when the runs are done.','运行跑完后显示。')}</p>`;
  const e=E_(), t=T_, n=e.n, K=e.K, f4=v=>fixd(v,4);
  const hist={}; e.cnt.forEach(c=>{hist[c]=(hist[c]||0)+1;});
  const hs=Object.keys(hist).map(Number).sort((a,b)=>b-a).map(c=>tx(`${hist[c]} tasks with ${c} of ${K}`,`${hist[c]} 个任务成功 ${c} / ${K} 次`)).join(tx(', ','，'));
  const blocks=[];
  blocks.push(MB(tx(`<b>${t('succ','Success rate')}.</b> First each task’s share of successful runs, then their mean, so every task counts once. ${M('c_i')} is task i’s successful runs, k = ${K} its runs, n = ${n} the tasks. Here: ${hs}.`,
                    `<b>${t('succ','成功率')}。</b>先算每个任务成功的比例，再取平均，这样每个任务只算一份。${M('c_i')} 是任务 i 成功的次数，k = ${K} 是它跑的次数，n = ${n} 是任务数。这里：${hs}。`),
    EQ([[tx('\\text{share of task } i','\\text{任务 } i \\text{ 的成功比例}')+'\\,\\,s_i','\\frac{c_i}{k}'],[tx('\\text{success rate}','\\text{成功率}'),'\\frac{1}{n} \\sum s_i'],['',`\\frac{${fixd(e.per.reduce((a,b)=>a+b,0),4)}}{${n}}`],['',f4(e.succ)]]),
    tx(`That is <b>${fP(e.succ)}</b>.`,`也就是 <b>${fP(e.succ)}</b>。`)));
  blocks.push(MB(tx(`<b>${t('ci','95% interval')}.</b> Where the success rate would roughly land on another ${n} tasks of the same kind. It reaches 1.96 standard errors either side of the success rate. sd is the sample standard deviation of the shares.`,
                    `<b>${t('ci','95% 区间')}。</b>换 ${n} 个同类的任务再跑，成功率大致会落在这个范围里。它从成功率往两边各伸出 1.96 个标准误差。sd 是这些成功比例的样本标准差。`),
    EQ([[tx('\\text{interval}','\\text{区间}'),tx('\\text{success rate}','\\text{成功率}')+' ± 1.96 \\cdot \\frac{\\text{sd}(s_i)}{\\sqrt{n}}'],['',`${f4(e.succ)} ± 1.96 \\cdot \\frac{${f4(e.sd)}}{\\sqrt{${n}}}`],['',`${f4(e.succ)} ± ${f4(1.96*e.se)}`]]),
    tx(`From <b>${fI(e)}</b>.`,`从 <b>${fI(e)}</b>。`)));
  if(K>=3) blocks.push(MB(tx(`<b>${t('p3','pass^3')}.</b> For each task, the chance that 3 runs drawn from its k runs all succeeded; then the mean over the tasks.${K===3?' With k = 3, a task counts only when all 3 runs succeed.':''}`,
                             `<b>${t('p3','pass^3')}。</b>对每个任务，算从它的 k 次运行里任取 3 次、3 次都成功的比例，再对任务取平均。${K===3?'k = 3 时，3 次都成功的任务才算。':''}`),
    EQ([['\\text{pass}^3','\\frac{1}{n} \\sum \\frac{C(c_i, 3)}{C(k, 3)}'],['',fP(e.p3).replace('%','\\%')]])));
  else blocks.push(MB(tx(`<b>${t('p3','pass^3')}</b> needs k ≥ 3.`,`<b>${t('p3','pass^3')}</b> 要 k ≥ 3 才能算。`),''));
  const tok=e.runs.reduce((a,rs)=>a+rs.reduce((b,r)=>b+r.tokens,0),0)/e.N;
  blocks.push(MB(tx(`<b>${t('cost','Cost per run')}.</b> The input tokens a run reads on average, times the price: ${PRICE[S.model]} dollar${S.model==='strong'?'s':''} per million tokens.`,`<b>${t('cost','每次运行的花费')}。</b>平均每次运行读进去的输入 token，乘以价格。价格是每 1M 个 token ${PRICE[S.model]} 美元。`),
    EQ([[tx('\\text{cost per run}','\\text{每次运行的花费}'),`${fInt(Math.round(tok))} \\times \\frac{${PRICE[S.model]}}{1,000,000}`]]),
    tx(`That is <b>${fCt(e.cost)}</b>.`,`也就是 <b>${fCt(e.cost)}</b>。`)));
  return blocks.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawScore(); drawGrid(); drawPlayer(); drawCauses(); drawMetrics();
  const lg=$('#acp-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const gc=$('#acp-grid-cap',app); gc.textContent=tx(`The evaluation: ${tasksOf(S.suite).length} tasks, each run ${S.K} ${S.K===1?'time':'times'}`,`评测：${tasksOf(S.suite).length} 个任务，每个跑 ${S.K} 次`);
  const mb=$('#acp-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* light redraw during play */
function tick(){if(!app) return; drawGrid(); drawPlayer(); if(runsDone()||PL.pv===0) {render(); return;} updateRail(); A.dock.refresh();}
/* Settings (R4.2): each guided part shows only its own controls; the rest are hidden with a note */
const PART_CTL={loud:['R','ck'],context:['ctx'],gate:['gate'],prove:['K']};
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the five choices in the task card. Everything else is fixed: the ops agent’s 50 tasks, 3 runs each, the assumed rates and prices.','挑战中只能改任务卡里的五个选择。其余的都固定：运维 agent 的 50 个任务，每个跑 3 次，假设的比例和价格。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return tx('Fixed in this part: the initial setup. You can change everything in Scenarios.','这一部分固定为初始设置。在应用场景里每一项都能改。');
  if(id==='loud') return tx('In this part you can change the retries and the check. Fixed: the cheaper model, the whole context kept, no gate, 3 runs per task.','这一部分可以改重试和检查。固定：便宜的模型，上下文全部保留，不把关，每个任务跑 3 次。');
  if(id==='context') return tx('In this part you can change the context. Fixed: 2 retries and a check, no gate.','这一部分可以改上下文。固定：重试 2 次加检查，不把关。');
  if(id==='gate') return tx('In this part you can change the gate on risky actions. Fixed: 2 retries, a check, and trim + summary + notes.','这一部分可以改危险操作的把关。固定：重试 2 次、检查，以及裁剪 + 摘要 + 笔记。');
  if(id==='prove') return tx('In this part you can change the runs per task. Fixed: the improved harness, with retries, a check, trim + summary + notes and a person’s confirmation.','这一部分可以改每个任务跑几次。固定：改好的 harness，也就是重试、检查、裁剪 + 摘要 + 笔记，再加上人确认。');
  return tx('Nothing to set in this part: each button in the card sets the example it names. To change things yourself, go to Scenarios.','这一部分不用改设置，卡片里的每个按钮会设好它说的那个例子。想自己改，去应用场景。');
}
function syncControls(){
  [['model',S.model],['R',S.R],['ck',S.ck],['ctx',S.ctx],['gate',S.gate],['K',S.K]].forEach(([k,v])=>$$(`#acp-controls .seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,String(v))));
  const so=$('#acp-seed-out',app); if(so) so.textContent=tx(`draw ${S.draw}`,`第 ${S.draw} 批`);
  const g=mode==='guided', ch=mode==='challenge', id=g?STEPS[step].id:'', all=['model','R','ck','ctx','gate','K','seed','reset'], live=g?(PART_CTL[id]||[]):(ch?[]:all);
  all.forEach(k=>{const e=$('#acp-w-'+k,app); if(e) e.hidden=!live.includes(k)&&!(k==='reset'&&g&&live.length);});
  const ga=$('#acp-w-as',app); if(ga) ga.hidden=ch;
  const ln=$('#acp-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||waiting()) return; const e=E_(); A.announce(tx(`${fP(e.succ)} of the runs succeed, interval ${fI(e)}, ${e.unsafe} unsafe actions, ${fCt(e.cost)} per run.`,`成功率 ${fP(e.succ)}，区间 ${fI(e)}，不安全的操作 ${e.unsafe} 次，每次运行 ${fCt(e.cost)}。`));},400);}

/* =====================================================================
   the player (R3.9: auto-play, pause, one round, to the end, three speeds, a scrubber)
   ===================================================================== */
function stopLoop(){if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=0;}
function frame(ts){
  PL.raf=0; if(!app||!PL.on) return;
  const dt=Math.min(0.25,(ts-(PL.last||ts))/1000); PL.last=ts; PL.acc+=dt;
  const per=SPD[PL.speed]; let moved=false;
  while(PL.acc>=per){PL.acc-=per; PL.pv++; moved=true; if(PL.pv>=totalCells()){PL.pv=totalCells(); PL.on=false; break;}}
  if(PL.pv>=totalCells()) finished();
  if(moved||!PL.on) tick();
  if(PL.on) PL.raf=requestAnimationFrame(frame);
}
function finished(){
  markSeen();
  if(isP('mission')&&pred&&!runsSeen){runsSeen=true; wantObs=true; renderRail(); revealSoon(); wantObs=false; return;}
  wantObs=true; render(); wantObs=false;
}
function play(){if(hideRes()&&!pred) return; if(PL.pv>=totalCells()) PL.pv=0; if(A.reduceMotion()){PL.pv=totalCells(); PL.on=false; finished(); tick(); return;} PL.on=true; PL.last=0; PL.acc=0; stopLoop(); PL.raf=requestAnimationFrame(frame); drawPlayer();}
function pause(){PL.on=false; stopLoop(); drawPlayer();}
/* R3.20: the run the grid shows, cell by cell in play order (1 succeeded, 2 unsafe action, 0 failed), and its suite */
let onGrid=null;
const cellsOf=e=>{const out=[]; for(let k=0;k<e.K;k++) for(let i=0;i<e.n;i++){const r=e.runs[i][k]; out.push(r.res==='ok'?1:(r.res==='unsafe'?2:0));} return out;};
const noteGrid=()=>{onGrid={suite:S.suite,cells:cellsOf(E_())};};
/* R3.20 (2026-10-04 addition): a setup already watched is not new. The grids watched to the end, remembered by what they draw (the
   cells in play order) per suite, not by part; a grid that is the start of one watched (fewer runs per task) counts as watched too */
const seenGrids={};
function markSeen(){const s=cellsOf(E_()).join(''), a=seenGrids[S.suite]||(seenGrids[S.suite]=[]); if(!a.some(w=>w.startsWith(s))) a.push(s);}
function wasSeen(cells){const s=(cells||cellsOf(E_())).join(''); return (seenGrids[S.suite]||[]).some(w=>w.startsWith(s));}
/* a new setup (R3.20): replay only when the picture would differ. The replay starts at the first cell drawn differently from what the
   grid shows now. When every cell of the new run is already on screen and drawn the same (the same run, or fewer runs per task), or
   the run was watched to the end before (a part revisited, a choice that recreates it, leaving the challenge), the finished result
   shows at once with no replay, so a part's gate counts as passed. The play button replays from the start. */
function restart(){
  const cells=cellsOf(E_()), N=cells.length, old=onGrid&&onGrid.suite===S.suite?onGrid.cells:null, seen=old?Math.min(PL.pv,old.length):0;
  let d=0; if(old) while(d<N&&d<old.length&&old[d]===cells[d]) d++;
  onGrid={suite:S.suite,cells}; PL.acc=0;
  if(hideRes()&&!pred){PL.pv=0; pause(); render(); return;}
  if((d>=N&&seen>=N)||wasSeen(cells)){PL.pv=N; PL.on=false; stopLoop(); markSeen();
    if(isP('mission')&&pred&&!runsSeen){runsSeen=true; renderRail(); return;}
    render(); return;}
  PL.pv=Math.min(d,seen); render(); play();
}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function clampS(){S.model=S.model==='strong'?'strong':'cheap'; S.R=[0,1,2].includes(+S.R)?+S.R:0; S.ck=S.ck?1:0; S.ctx=['all','last','notes'].includes(S.ctx)?S.ctx:'all'; S.gate=['none','detect','confirm'].includes(S.gate)?S.gate:'none'; S.K=[1,3,5].includes(+S.K)?+S.K:3; S.draw=Math.max(1,Math.min(99,Math.round(S.draw||1)));}
function setState(p){const before=JSON.stringify(S); Object.assign(S,p); clampS(); if(JSON.stringify(S)!==before) restart(); else render();}
function byLearner(fn){userAct=true; pendBadge=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(part,v){
  if(part===2){const o=LV2.find(x=>x.v===v); lv2=v; tried2.add(v); setState({...P2,...o.s,suite:'shop',draw:S.draw});}
  else if(part===3){const o=LV3.find(x=>x.v===v); lv3=v; tried3.add(v); setState({...P3,...o.s,suite:'shop',draw:S.draw});}
  else {const o=LV4.find(x=>x.v===v); lv4=v; tried4.add(v); setState({...P4,...o.s,suite:'shop',draw:S.draw});}
}
function setRem(k){if(!REMS[k]) return; remView=k; S={...REMS[k].s,suite:'shop'}; renderRail(); restart(); if(innerWidth<=760) A.ui.reveal($('#acp-stage',app));}
function enterStep(i){
  wantObs=false; pendBadge=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...INIT,...s.enter(),suite:'shop',draw:1};
  const st=$('#acp-stage',app); if(st) st.scrollTop=0;
  renderRail(); restart();
}
function revealStep(){A.ui.reveal($('#acp-rail .step.current',app)||$('#acp-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), as labs/agent-reliability.js does */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#acp-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#acp-rail .step.current',app)||$('#acp-rail .railcard',app)||$('#acp-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#acp-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#acp-insight',app)); else revealBox($('#acp-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; pendBadge=false;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chO={model:'cheap',R:0,ck:0,ctx:'all',gate:'none'}; chTried=[]; S={...chH(chO),suite:'ops'};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...INIT,suite:'shop',draw:S.draw};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#acp-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail(); restart();
}
const tokOf=s=>[s.model,s.R,s.ck,s.ctx,s.gate,s.K,s.draw].join('_');
const token=()=>tokOf(S);
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const [model,R,ck,ctx,gate,K,draw]=p, I=Number.isInteger;
  if(!['cheap','strong'].includes(model)||![0,1,2].includes(+R)||![0,1].includes(+ck)||!['all','last','notes'].includes(ctx)||!['none','detect','confirm'].includes(gate)||![1,3,5].includes(+K)||!I(+draw)||+draw<1||+draw>99) return null;
  return {model,R:+R,ck:+ck,ctx,gate,K:+K,draw:+draw,suite:'shop'};
}

/* =====================================================================
   interview drill (seeded, from the engine's statistics)
   ===================================================================== */
function makeDrill(seed){
  const n=10, k=3, cnt=[]; for(let i=0;i<n;i++){const u=u01(900+seed,i,0,0,0); cnt.push(u<0.55?3:(u<0.75?2:(u<0.9?1:0)));}
  const s=cnt.map(c=>c/k), mean=s.reduce((a,b)=>a+b,0)/n, sd=Math.sqrt(s.reduce((a,x)=>a+(x-mean)*(x-mean),0)/(n-1)), half=1.96*sd/Math.sqrt(n), p3=cnt.filter(c=>c===3).length/n;
  return {seed,n,k,cnt,mean,sd,half,p3};
}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#acp-drill-q',app), a=$('#acp-drill-a',app);
  qd.innerHTML=tx(`An agent is run <b>3 times</b> on each of <b>10 tasks</b>. The successful runs per task are <b>${d.cnt.join(', ')}</b>. What are the success rate, its 95% interval and pass^3?`,
                  `一个 agent 在 <b>10 个任务</b>上各跑 <b>3 次</b>。每个任务成功的次数是 <b>${d.cnt.join('、')}</b>。成功率、它的 95% 区间和 pass^3 各是多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const eq=EQ([[tx('\\text{half-width}','\\text{区间的半宽}'),'1.96 \\cdot \\frac{\\text{sd}}{\\sqrt{n}}'],['',`1.96 \\cdot \\frac{${fixd(d.sd,4)}}{\\sqrt{10}}`],['',fixd(d.half,4)]]);
  a.innerHTML=tx(`<p><b>${fP(d.mean)}, interval ${fP(d.mean-d.half)} to ${fP(d.mean+d.half)}, pass^3 ${fP(d.p3)}.</b></p><p>Each task’s share is its successful runs out of 3; their mean is ${fixd(d.mean,4)}. Their sample standard deviation sd is ${fixd(d.sd,4)}, and n = 10 tasks:</p>${eq}<p>pass^3 counts the tasks with 3 of 3: ${d.cnt.filter(c=>c===3).length} of 10.</p><p>Treating the 30 runs as independent would understate the interval: runs of the same task are alike.</p>`,
    `<p><b>${fP(d.mean)}，区间 ${fP(d.mean-d.half)} 到 ${fP(d.mean+d.half)}，pass^3 ${fP(d.p3)}。</b></p><p>每个任务的成功比例，是成功次数除以 3。这些比例的平均是 ${fixd(d.mean,4)}，样本标准差 sd 是 ${fixd(d.sd,4)}，任务数 n = 10：</p>${eq}<p>pass^3 数的是 3 次都成功的任务：10 个里有 ${d.cnt.filter(c=>c===3).length} 个。</p><p>把 30 次运行当成互不相关来算，区间会偏窄：同一个任务的几次运行是相似的。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={R:{en:'Retries after a reported error',zh:'报错后最多重试'},ck:{en:'Check after each step',zh:'每步之后检查'},ctx:{en:'Context',zh:'上下文'},gate:{en:'Risky actions',zh:'危险的操作'},K:{en:'Runs per task',zh:'每个任务跑几次'},model:{en:'Model',zh:'模型'}};
const TERM={
 succ:()=>tx('The mean of the tasks’ success shares: each task counts once, however many runs it had.','各个任务成功比例的平均：不管跑了几次，每个任务都只算一份。'),
 ci:()=>tx('1.96 standard errors on each side; the standard error is the spread of the tasks’ shares divided by √n, as lm-eval’s mean_stderr does.','两边各 1.96 个标准误；标准误是各任务成功比例的标准差除以 √n，和 lm-eval 的 mean_stderr 一样。'),
 p3:()=>tx('For each task, the chance that 3 runs drawn from its k runs all succeeded: C(c, 3) ÷ C(k, 3), averaged over the tasks.','对每个任务，从它的 k 次运行里任取 3 次、3 次都成功的比例：C(c, 3) ÷ C(k, 3)，再对任务取平均。'),
 cost:()=>tx('The cost of a run is the input tokens it reads times the price, averaged over the runs. Every call reads its whole context again; output tokens and prompt caching are left out.','每次运行的花费，是这次运行读进去的输入 token 乘以价格，再对所有运行取平均。每次调用都要把整个上下文重读一遍。输出的 token 和 prompt caching 都没有算。')
};
const TERMJ={succ:'K',ci:'K',p3:'K',cost:'model'};
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: change the five choices in the task card.','挑战中固定不变：请改任务卡里的五个选择。');
  if(mode!=='guided') return '';
  const live=PART_CTL[STEPS[step].id]||[]; if(live.includes(k)) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math” under the chart: the success rate, the interval and pass^3, worked out line by line.','点击打开图下的“展开公式”，里面有成功率、区间和 pass^3 的逐行算式。');
  const c=TERMJ[k]||k; if(!CTLN[c]) return '';
  /* S12: a term of the formulas is not a setting; when the setting it depends on is fixed here, say nothing about settings */
  const m=ctlLockMsg(c); if(m) return TERM[k]?'':m;
  return tx(`Click to jump to “${L(CTLN[c])}”.`,`点击跳到“${L(CTLN[c])}”。`);
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#acp-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#acp-controls',app)); return;}
  const c=TERMJ[k]||k; if(!CTLN[c]) return;
  if(ctlLockMsg(c)){if(src) A.tip.show(src,TERM[k]?`<div>${tipMath(TERM[k]())}</div>`:`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=(c==='K'&&$('#acp-ic-K',app))||$('#acp-ctl-'+c,app)||$('#acp-controls',app); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='play'){if(PL.on) pause(); else play(); return;}
  if(a==='stepf'){A.tip.quiet(); pause(); if(hideRes()&&!pred) return; const n=tasksOf(S.suite).length; if(PL.pv>=totalCells()) PL.pv=0; PL.pv=Math.min(totalCells(),(Math.floor(PL.pv/n)+1)*n); if(PL.pv>=totalCells()) finished(); tick(); return;}
  if(a==='end'){pause(); if(hideRes()&&!pred) return; PL.pv=totalCells(); finished(); tick(); return;}
  if(a==='reset'){byLearner(()=>{let base={...INIT,suite:'shop',draw:S.draw}; if(mode==='guided'){const s=STEPS[step]; if(s.enter) base={...INIT,...s.enter(),suite:'shop',draw:S.draw};} S=base; restart();}); return;}
  if(a==='reseed'){byLearner(()=>setState({draw:S.draw+1})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chO).g.all) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#acp-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();} finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#acp-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const t=e.target;
    if(t.id==='acp-scrub'){pause(); if(hideRes()&&!pred) return; PL.pv=Math.max(0,Math.min(totalCells(),+t.value)); if(PL.pv>=totalCells()) finished(); tick(); return;}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#acp-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>{if(typeof s.showMe==='function') s.showMe(); else {if(s.showMe.K) tried5.add(s.showMe.K); setState(s.showMe);}}); revealSoon(); wantObs=false;} return;}
    if(d.pred){const first=!pred; pred=d.pred; renderRail(); if(first) restart(); const fb=$(`#acp-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null){const part=isP('loud')?2:(isP('context')?3:(isP('gate')?4:0)); if(!part) return; const f=()=>byLearner(()=>setLv(part,d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='speed'){PL.speed=v; drawPlayer(); return;}
      if(['chM','chR','chC','chX','chG'].includes(k)){const f={chM:'model',chR:'R',chC:'ck',chX:'ctx',chG:'gate'}[k]; chO={...chO,[f]:['R','ck'].includes(f)?+v:v}; if(!chTried.some(o=>oKey(o)===oKey(chO))) chTried.push({...chO}); byLearner(()=>{S={...chH(chO),suite:'ops'}; restart();}); return;}
      if(k==='K'){if(isP('prove')) tried5.add(+v); bySettings(b,()=>setState({K:+v})); if(b.closest('#acp-rail')){wantObs=true; revealSoon(); wantObs=false;} return;}
      if(['model','R','ck','ctx','gate'].includes(k)){bySettings(b,()=>setState({[k]:['R','ck'].includes(k)?+v:v})); return;}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s,suite:'shop',draw:S.draw})); requestAnimationFrame(()=>{if(app) revealBox($('#acp-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#acp-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the choices tried in each part, the grids watched to
     the end, the challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chO:{...chO},chTried:chTried.map(o=>({...o})),chSubmitted,chReveal,
    runsSeen,lv2,tried2:[...tried2],lv3,tried3:[...tried3],lv4,tried4:[...tried4],tried5:[...tried5],remView,sbTouched,drillSeed,drillOpen,speed:PL.speed,
    seen:Object.fromEntries(Object.entries(seenGrids).map(([k,a])=>[k,a.slice(-30)])),passed:mode==='challenge'&&chSubmitted&&chEval(chO).g.all}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(tokOf(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, vs=a=>a.map(x=>x.v), set=(v,ok,base)=>new Set([...base,...(Array.isArray(v)?v.filter(ok):[])]);
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; S={...p,suite:mode==='challenge'?'ops':'shop'}; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null;
    const okO=c=>!!c&&typeof c==='object'&&CH.models.includes(c.model)&&CH.Rs.includes(c.R)&&CH.cks.includes(c.ck)&&CH.ctxs.includes(c.ctx)&&CH.gates.includes(c.gate), pickO=c=>({model:c.model,R:c.R,ck:c.ck,ctx:c.ctx,gate:c.gate});
    if(okO(o.chO)) chO=pickO(o.chO); chTried=[]; if(Array.isArray(o.chTried)) o.chTried.filter(okO).forEach(c=>{if(!chTried.some(x=>oKey(x)===oKey(c))) chTried.push(pickO(c));});
    chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal); runsSeen=bool(o.runsSeen);
    const L2=vs(LV2), L3=vs(LV3), L4=vs(LV4);
    lv2=L2.includes(o.lv2)?o.lv2:'init'; tried2=set(o.tried2,v=>L2.includes(v),['init',lv2]);
    lv3=L3.includes(o.lv3)?o.lv3:'all'; tried3=set(o.tried3,v=>L3.includes(v),['all',lv3]);
    lv4=L4.includes(o.lv4)?o.lv4:'none'; tried4=set(o.tried4,v=>L4.includes(v),['none',lv4]); tried5=set(o.tried5,v=>[1,3,5].includes(v),[3]);
    remView=o.remView===''||Object.keys(REMS).includes(o.remView)?o.remView:''; sbTouched=bool(o.sbTouched); drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen);
    if(['s','m','f'].includes(o.speed)) PL.speed=o.speed;
    if(o.seen&&typeof o.seen==='object') Object.keys(SUITES).forEach(k=>{const a=o.seen[k]; if(Array.isArray(a)) seenGrids[k]=a.filter(w=>typeof w==='string'&&w.length<=2000&&/^[012]+$/.test(w)).slice(-30);});
    if(mode==='challenge') S={...chH(chO),suite:'ops'};
    PL.on=false; PL.pv=!(hideRes()&&!pred)&&wasSeen()?totalCells():0;},   // a grid watched to the end shows finished, as when coming back in the same visit
  mount(el,tok){
    wantObs=false; root=el;
    /* a link with another setup is a new picture: it plays from the start, as on a fresh page, unless it was watched to the end (R3.20) */
    const p=tok?parse(tok):null; if(p){const was=hKey(S.suite,hNow()); S={...p}; mode='sandbox'; sbTouched=true; if(hKey(S.suite,hNow())!==was){PL.on=false; stopLoop(); PL.pv=wasSeen()?totalCells():0;}}
    if(mode==='challenge'){if(!saved) saved={...INIT,suite:'shop'}; S={...chH(chO),suite:'ops'};}
    el.innerHTML=viewHTML(); app=$('#acp-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#acp-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    if(mode==='guided'&&!p&&step===0&&!pred){S={...INIT,suite:'shop'};}
    noteGrid();                                   /* the grid always draws the current setup's runs up to PL.pv */
    fillRail(); render();
    if(PL.on){PL.last=0; stopLoop(); PL.raf=requestAnimationFrame(frame);}
    else if(!(hideRes()&&!pred)&&PL.pv===0) play();
  },
  leave(){clearTimeout(annT); stageStop(); stopLoop(); root=null; app=null;},
  onLang(){if(root){const d=$('#acp-math',app), open=d&&d.open; stopLoop(); this.mount(root,null); if(open) $('#acp-math',app).open=true;}},
  onResize(){if(app){drawGrid(); drawScore();}},
  termTip(k){const f=TERM[k]; if(!f) return ''; const a=jumpAct(k); return `<div>${tipMath(f())}</div>`+(a?`<div class="act">${esc(a)}</div>`:'');},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#acp-grid',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#acp-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(waiting()){vis.innerHTML=''; const t=doneNow(); read.innerHTML=esc(hideRes()&&!pred?tx('Answer part 1 first','先回答第 1 部分的问题'):tx(`Suite running: ${t.n} of ${t.N} runs done`,`评测在跑：跑完 ${t.n} / ${t.N} 次`)); return;}
      const e=E_(), sc=v=>Math.max(0,Math.min(100,(v*100-40)/60*100)), lo=sc(e.lo), hi=sc(e.hi);  /* the stage's axis: 40% to 100% */
      vis.innerHTML=`<div class="acp-dk" aria-hidden="true"><i class="ci" style="left:${lo.toFixed(1)}%;width:${(hi-lo).toFixed(1)}%"></i><i class="gl" style="left:${sc(0.85).toFixed(1)}%"></i><i class="pt" style="left:${sc(e.succ).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(tx(`${fP(e.succ)} succeed (interval ${fI(e)}), ${e.unsafe} unsafe, ${fCt(e.cost)} per run`,`成功率 ${fP(e.succ)}（区间 ${fI(e)}），不安全 ${e.unsafe} 次，每次运行 ${fCt(e.cost)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const slim=e=>({succ:e.succ,lo:e.lo,hi:e.hi,sd:e.sd,p3:e.p3,unsafe:e.unsafe,cost:e.cost,conf:e.conf,causes:{...e.causes},cnt:e.cnt.slice(),per:e.per.slice(),N:e.N,n:e.n,K:e.K});
window.__acp=Object.freeze({
  u01, tasks:name=>deep(tasksOf(name)), run:(name,i,k,h)=>deep(simRun(name,tasksOf(name)[i],k,{...INIT,...h})),
  evaluate:(name,h)=>slim(evaluate(name,{...INIT,...h})), gain:(name,h,h0)=>deep(gainOf(name,{...INIT,...h},{...INIT,...h0})),
  goal:(name,h)=>deep(goalOf(evaluate(name,{...INIT,...h}),BUDGET[name])), chEval:o=>{const x=chEval(o); return {o:x.o,e:slim(x.e),g:x.g};}, chAll:()=>chAll().map(x=>({o:x.o,succ:x.e.succ,lo:x.e.lo,p3:x.e.p3,unsafe:x.e.unsafe,cost:x.e.cost,g:x.g})),
  chBest:()=>{const b=chBest(); return b?{o:b.o,cost:b.e.cost,succ:b.e.succ}:null;}, drill:s=>deep(makeDrill(s)), predRight, codeText,
  lucksplit:(name,h,D)=>{const o=lucksplit(name,{...INIT,...h},D); return {between:o.between,within:o.within,luck3:o.luck(3),luck5:o.luck(5)};},
  consts:Object.freeze({K:{...K_},PRICE:{...PRICE},INIT:{...INIT},FINAL:{...FINAL},BUDGET:{...BUDGET},GOAL:{...GOAL},PRED_H:deep(PRED_H),STEP_IDS:STEPS.map(s=>s.id),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),
    SUITES:Object.fromEntries(Object.entries(SUITES).map(([k,v])=>[k,{seed:v.seed,kinds:v.kinds,n:v.n,q:v.q,r:v.r,outE:v.outE,need:v.need}]))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},runsSeen,pv:PL.pv,playing:PL.on,speed:PL.speed,lv2,lv3,lv4,tried5:[...tried5],chO:{...chO},chSubmitted,chReveal,remView,stageBusy:stageT!==0})
});
})();
