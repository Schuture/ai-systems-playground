/* agent-reliability (world agents): 95% reliable per step, 20 steps: how often does the agent succeed?
   Fidelity: Simulated (header: Toy run). Every per-step rate is the lab's own assumption; the end-to-end numbers follow exactly from
   them by the closed form below. The picture is an illustration: one set of 30 seeded tasks (owner check b7, R3.9). The default seed
   (377) is the one whose 30 tasks come out typical of the formula in every guided part; any other seed is a fair draw ("New draws").
   Replays (R3.20): a new setup replays only the rows whose picture changed, from their first changed cell; the same picture is not
   replayed, a picture already watched to the end shows its end at once, and the play button replays the whole run once it has finished.
   - A task is n steps. A first attempt at a step succeeds with p (q = 1 − p). A failed attempt is loud (the harness sees it) with
     probability r, silent otherwise. A check after every attempt (one more call) makes a share c of the silent failures loud:
     rc = r + (1 − r)c. A loud failure is retried up to R times; a retry succeeds with p2 (with the error message: p2 = p; the same
     call again: p2 = pb, part 5). Retries used up: the run stops. A silent failure: the run goes on, wrong, to the end.
   - P_ok = p + q rc p2 (1 − (q2 rc)^R) / (1 − q2 rc); P_stop = q rc (q2 rc)^R; success = P_ok^n; stopped = 1 − (1 − P_stop)^n;
     calls = (1 + check) × attempts per step × Σ_{j<n} (1 − P_stop)^j.
   - Runs: u(task, step, attempt, purpose) from a counter-based 32-bit hash of the seed (common random numbers: a setting change flips
     only the outcomes near the new threshold). tests/agent-reliability_numbers.py ports it.
   Words (EN / ZH): step / 步 (one model call, usually followed by one tool call); loud / 报出来的错; silent / 悄悄出的错; retry / 重试;
   check / 检查; run / task (a row) = 任务; part / 部分.
   window.__arl is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='agent-reliability';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const RUNS=30;
const BASE={n:20,p:0.95,r:0.7,R:0,ck:0,kind:'i',c:0.8,pb:0.5,seed:377,price:1};
function stepModel(s){
  const p=s.p, q=1-p, p2=s.kind==='b'?s.pb:p, q2=1-p2, rc=s.ck?s.r+(1-s.r)*s.c:s.r, x=q2*rc;
  let geo=0, t=1; for(let i=0;i<s.R;i++){geo+=t; t*=x;}          // Σ_{i<R} x^i
  const ok=p+q*rc*p2*geo, stop=q*rc*Math.pow(x,s.R), att=1+q*rc*geo;
  return {p,q,p2,q2,rc,ok,stop,sil:Math.max(0,1-ok-stop),att};
}
function compute(s){
  const m=stepModel(s), n=s.n, succ=Math.pow(m.ok,n), live=Math.pow(1-m.stop,n);
  let reach=0; for(let j=0;j<n;j++) reach+=Math.pow(1-m.stop,j);
  const calls=(s.ck?2:1)*m.att*reach;
  return {...m,n,succ,wrong:Math.max(0,live-succ),stopped:1-live,reach,calls,cost:calls*(s.price||1),plain:Math.pow(s.p,n)};
}
/* the random numbers: a counter-based hash, one number per (task, step, attempt, purpose: 0 success, 1 loud, 2 caught by the check) */
const mix=h=>{h^=h>>>16; h=Math.imul(h,0x7feb352d); h^=h>>>15; h=Math.imul(h,0x846ca68b); h^=h>>>16; return h>>>0;};
function u01(seed,t,j,a,k){let h=mix(Math.imul(seed,0x9E3779B1)^0x85ebca6b); h=mix(h^Math.imul(t+1,0x27d4eb2f)); h=mix(h^Math.imul(j+1,0x165667b1)); h=mix(h^Math.imul(a*4+k+1,0x2545f491)); return h/4294967296;}
/* one task: cells ok | retry | silent | pale (after a silent failure) | stop | none (after a stop); fails = failed attempts in the cell */
function simRun(s,t){
  const cells=[], p2=s.kind==='b'?s.pb:s.p, per=s.ck?2:1; let doomed=false, stopped=false, calls=0, end=s.n-1, fx=-1;
  for(let j=0;j<s.n;j++){
    if(stopped){cells.push({st:'none',f:0}); continue;}
    let a=0, f=0, res='ok';
    for(;;){calls+=per; if(u01(s.seed,t,j,a,0)<(a===0?s.p:p2)){res='ok'; break;}
      const loud=u01(s.seed,t,j,a,1)<s.r||(!!s.ck&&u01(s.seed,t,j,a,2)<s.c);
      if(!loud){res='silent'; break;}
      f++; if(a>=s.R){res='stop'; break;} a++;}
    if(res==='stop'){stopped=true; end=j; if(fx<0) fx=j; cells.push({st:'stop',f});}
    else if(res==='silent'){if(fx<0) fx=j; cells.push({st:doomed?'pale':'silent',f}); doomed=true;}
    else cells.push({st:doomed?'pale':(f?'retry':'ok'),f});
  }
  return {cells,out:stopped?'stop':(doomed?'wrong':'ok'),calls,end,fx:fx<0?s.n-1:fx};
}
const RC=new Map();
function runsOf(s){const key=[s.n,s.p,s.r,s.R,s.ck,s.kind,s.c,s.pb,s.seed].join('|'); if(RC.has(key)) return RC.get(key);
  const a=[]; for(let t=0;t<RUNS;t++) a.push(simRun(s,t)); if(RC.size>60) RC.clear(); RC.set(key,a); return a;}
function tallyOf(runs){const ok=runs.filter(r=>r.out==='ok').length, wr=runs.filter(r=>r.out==='wrong').length;
  return {n:runs.length,ok,wrong:wr,stop:runs.length-ok-wr,calls:runs.reduce((a,r)=>a+r.calls,0)/Math.max(1,runs.length)};}

/* the challenge: a finance-report agent, 95% per attempt, half its failures silent; four choices */
const CH={p:0.95,pS:0.98,r:0.5,c:0.8,ns:[20,12],Rs:[0,1,2,3],goal:0.9,budget:40,start:{strong:0,ck:0,R:0,n:20}};
const chState=o=>({...BASE,n:o.n,p:o.strong?CH.pS:CH.p,r:CH.r,R:o.R,ck:o.ck,c:CH.c,kind:'i',price:o.strong?2:1});
function chEval(o){const r=compute(chState(o)); return {...o,succ:r.succ,cost:r.cost,okS:r.succ>=CH.goal,okC:r.cost<=CH.budget+1e-9,ok:r.succ>=CH.goal&&r.cost<=CH.budget+1e-9};}
function chAll(){const out=[]; [0,1].forEach(strong=>[0,1].forEach(ck=>CH.Rs.forEach(R=>CH.ns.forEach(n=>out.push(chEval({strong,ck,R,n})))))); return out;}
/* minimal: the fewest calls per task as shown (one decimal) among the passing combinations; ties go to fewer retries */
const r1=x=>Math.round(x*10)/10;
function chBest(){const ok=chAll().filter(e=>e.ok); if(!ok.length) return null; ok.sort((a,b)=>r1(a.cost)-r1(b.cost)||a.R-b.R); return ok[0];}

/* =====================================================================
   formatting
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fP=x=>{const v=x*100; if(x<=0) return '0%'; if(x>=1) return '100%'; if(v>=99.95) return '>99.9%'; if(v<0.05) return '<0.1%'; return fixd(v,1)+'%';};
const fPc=p=>fixd(p*100,1)+'%';                            // a per-step rate as set (95.0%)
const fPs=p=>{const v=p*100; return Math.abs(v-Math.round(v))<1e-9?String(Math.round(v))+'%':fixd(v,1)+'%';};   // in prose: 95%, 99.5%
const fC=x=>fixd(x,1);
const fN=x=>fixd(x,4);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const M=t=>A.ui.math?A.ui.math(t):esc(t);
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="arl-eqs${EQN(rows)?' arl-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="arl-eq-l">${l?M(l):''}</span><span class="arl-eq-e">${M(rel||'=')}</span><span class="arl-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="arl-mb"><p>${lead}</p>${body}${res?`<p class="arl-mb-res">${res}</p>`:''}</div>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const stepsW=n=>tx(n===1?'1 step':`${n} steps`,`${n} 步`);
const retriesW=R=>R===0?tx('no retries','不重试'):tx(R===1?'1 retry':`${R} retries`,`重试 ${R} 次`);

/* =====================================================================
   module state
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lv2='base', tried2=new Set(['base']), lv4='none', tried4=new Set(['none']), lv5='info', tried5=new Set(['info']), remView='';
let runsSeen=false, saved=null, chO={...CH.start}, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null;
let drillSeed=1, drillOpen=false;
const shownPrev={};
/* the player: pv = columns shown (0 … n); sig = what the grid shows (null: nothing yet); fix[i] = columns of row i shown at once in a
   partial replay (n: the row did not change), or null in a full replay. A whole run defaults to fast (R3.9): seconds per run. */
const PL={pv:0,on:false,speed:'f',raf:0,last:0,acc:0,sig:null,fix:null};
const SPD={s:24,m:12,f:6};
/* R3.20 (addition of 2026-10-04): every picture whose end has been on screen, keyed by what the grid draws (each row, cell by cell),
   not by part or mode. Shown again (Back to a part, a scenario or Settings choice that recreates it, leaving the challenge), it shows
   its end at once. */
const SEEN=new Set();
/* the pictures watched before a reload, as short hashes of their keys (a full key is a few thousand characters): round 3 #6 */
const SEENH=new Set(), hkey=k=>{let h=2166136261; for(let i=0;i<k.length;i++){h^=k.charCodeAt(i); h=Math.imul(h,16777619);} return (h>>>0).toString(36);};
const picKey=p=>p.n+':'+p.rows.map(r=>r.join(',')).join('|');
const R_=()=>compute(S);

/* =====================================================================
   definitions (R2.8)
   ===================================================================== */
const TIP={
 step:()=>tx('A step is one turn of the agent: one call to the model, usually followed by one tool call (search the code, edit a file, run the tests); the tool’s result goes back to the model for the next step.',
             '一步就是 agent 的一个回合：调用一次模型，模型通常再调用一次工具（搜代码、改文件、跑测试）；工具的结果交回给模型，接着做下一步。'),
 loud:()=>tx('A loud failure announces itself: a tool returns an error, or a test fails. The harness, the program that runs the agent, sees it and can redo the step.',
             '报出来的错会自己暴露：工具返回错误，或者测试没通过。运行 agent 的程序（harness）看得见它，可以把这一步重做一遍。'),
 silent:()=>tx('A silent failure looks like success: the wrong file was edited, or a number was copied wrong. Nothing reports it, so nothing retries it, and every later step builds on the wrong result.',
               '悄悄出的错看起来和做对了一样：改错了文件，或者把数字抄错了。没有任何东西报告它，也就不会有重试，后面的每一步都在错的结果上接着做。'),
 retry:()=>tx('A retry redoes a step after a loud failure. Here a retry that sees the error message does as well as a first attempt (the lab’s assumption).',
              '重试就是一步报了错之后，把这一步再做一遍。这个 Lab 假设，看得到报错信息的重试和第一次做一样有把握。'),
 harness:()=>tx('The harness is the program around the model that runs the agent: it calls the model, runs the tools, hands the results back, and decides when to retry and when to stop.',
                'harness 是包在模型外面、负责运行 agent 的程序：它调用模型、执行工具、把结果交回去，并决定什么时候重试、什么时候停下。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'a',t:{en:'90 or more',zh:'90 个以上'}},{v:'b',t:{en:'60 to 89',zh:'60 到 89 个'}},{v:'c',t:{en:'30 to 59',zh:'30 到 59 个'}},{v:'d',t:{en:'Fewer than 30',zh:'不到 30 个'}}];
const PRED_RIGHT='c';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
/* parts 1 and 2: one failure state; loud and silent failures, retries and stops come in part 3 (R6.30, R6.31, S1) */
const simple=()=>isP('mission')||isP('chain');
const endOf=r=>simple()?r.fx:r.end;
/* part 1: nothing about the result shows before the prediction and before the runs end (R3.8) */
const hideRes=()=>isP('mission')&&!(pred&&runsSeen);
const LV2=[{v:'base',s:{n:20,p:0.95},t:{en:'Initial setup: 20 steps, 95% each',zh:'初始设置：20 步，每步 95%'},d:{en:'Every step has to be right.',zh:'每一步都得做对。'}},
  {v:'half',s:{n:10,p:0.95},t:{en:'Half the steps: 10',zh:'步数减半：10 步'},d:{en:'Still 95% per step.',zh:'每步还是 95%。'}},
  {v:'steady',s:{n:20,p:0.99},t:{en:'Steadier steps: 99% each',zh:'每步更稳：99%'},d:{en:'Still 20 steps.',zh:'还是 20 步。'}}];
const LV4=[{v:'none',s:{ck:0,R:2},t:{en:'No check, up to 2 retries (initial setup)',zh:'不检查，最多重试 2 次（初始设置）'},d:{en:'Only the errors a tool reports are redone. Nobody sees the silent ones.',zh:'只有报出来的错会重做，悄悄出的错没人发现。'}},
  {v:'ck',s:{ck:1,R:2},t:{en:'A check after each step, up to 2 retries',zh:'每步检查，最多重试 2 次'},d:{en:'The errors the check finds are redone too. Every attempt costs one more call.',zh:'检查发现的错也会重做。每次尝试多一次调用。'}},
  {v:'ck0',s:{ck:1,R:0},t:{en:'A check after each step, no retries',zh:'每步检查，不重试'},d:{en:'An error the check finds is not redone, so the task fails. But the failure is seen.',zh:'检查发现了错也不重做，任务就失败了，但失败被看见了。'}}];
const LV5=[{v:'info',s:{R:2,kind:'i'},t:{en:'Retry with the error message, 2 times (initial setup)',zh:'带着报错信息重试 2 次（初始设置）'},d:{en:'The model reads what went wrong.',zh:'模型看得到哪里错了。'}},
  {v:'blind',s:{R:2,kind:'b'},t:{en:'Retry the same call, 2 times',zh:'原样重试 2 次'},d:{en:'Nothing new goes to the model.',zh:'模型拿不到任何新信息。'}},
  {v:'none',s:{R:0,kind:'i'},t:{en:'No retries',zh:'不重试'},d:{en:'A loud failure stops the task.',zh:'报出来的错直接让任务停下。'}}];
const P3={...BASE,R:0}, P4={...BASE,R:2}, P5={...BASE,R:2};
const lvRes=s=>fP(compute({...S,...s}).succ);
const lvRes4=s=>{const r=compute({...S,...s}); return tx(`${fP(r.succ)}, ${fC(r.calls)} calls`,`${fP(r.succ)}，${fC(r.calls)} 次`);};
function leversHTML(id,list,cur,tried,res,aria){
  return `<div class="arl-lv" id="${id}">`+A.ui.levers({aria,value:cur,options:list.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(tried.has(o.v)||o.v===cur?res(o.s):'')}))})+`</div>`;
}
/* the setting the part's levers match, or '' (R1.8: the pressed state follows the settings) */
const match=(list)=>{const o=list.find(x=>Object.keys(x.s).every(k=>S[k]===x.s[k])); return o?o.v:'';};
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('A 20-step task at 95% per step: how many of 100 tasks succeed?','一个 20 步的任务，每步 95% 的把握：100 个任务能成功几个？'),
  enter:()=>({...BASE}),gate:()=>!!pred&&runsSeen,skip:true,label:{en:'Result',zh:'结果'},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You are building a coding agent. It reads an issue, searches the code, edits files, runs the tests and writes a report. One task takes 20 ${DEF('steps',TIP.step())}.</p><p>The agent gets each step right 95% of the time. A task succeeds only if all 20 steps are right.</p><p>Out of 100 tasks, about how many succeed?</p>${ch}${pred?`<p class="muted small">${runsSeen?'Saved.':`Saved. You can still change it before the ${RUNS} tasks in the chart finish.`}</p>`:''}`,
             `<p>你在做一个修代码的 agent。它读 issue、搜代码、改文件、跑测试，最后写一份报告。一个任务要走 20 ${DEF('步',TIP.step())}。</p><p>这个 agent 每一步有 95% 的把握做对。20 步都做对，任务才算成功。</p><p>100 个任务里，大约能成功多少个？</p>${ch}${pred?`<p class="muted small">${runsSeen?'已记下。':`已记下。图里的 ${RUNS} 个任务跑完之前，还可以改。`}</p>`:''}`);},
  obs:()=>obsMission()},
 {id:'chain',title:{en:'Every step is one more chance to fail',zh:'每多一步，就多一次出错的机会'},
  lead:()=>tx('Success is the per-step chance multiplied n times: fewer steps, or steadier ones?','成功率是每一步的把握连乘：少走几步，还是每步更稳？'),
  enter:()=>{lv2='base'; tried2=new Set(['base']); return {...BASE};},expect:()=>tried2.size>1,showMe:()=>setLv(2,'steady'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>A task succeeds only if every step is right. So the chance that the whole task succeeds is the chance of each step, multiplied together:</p><div class="arl-fx">${M('P_{\\text{task}} = p^{n}')}</div><p>${T_('p','p')} is the chance that one step is right, and ${T_('n','n')} is the number of steps. Every extra step multiplies in one more p, so the success rate drops a little more.</p><p>There are two direct ways to raise it. Pick one and see how high the task success rate gets.</p>`,
              `<p>任务要成功，每一步都得做对。所以把每一步做对的把握乘起来，就是整个任务成功的把握：</p><div class="arl-fx">${M('P_{\\text{任务}} = p^{n}')}</div><p>${T_('p','p')} 是一步做对的把握，${T_('n','n')} 是步数。每多走一步，就要再乘一个 p，成功率就再掉一点。</p><p>有两个直接的办法能把它提上去。选一个，看看任务的成功率能涨到多少。</p>`)+leversHTML('arl-ic-lv2',LV2,lv2,tried2,lvRes,tx('Pick a way to raise the success rate','选一个提高成功率的办法')),
  obs:()=>obsChain()},
 {id:'loud',title:{en:'Loud failures and silent ones',zh:'报出来的错，和悄悄出的错'},
  lead:()=>tx('Retries rescue the failures that announce themselves; silent ones go on to the end.','重试救得回报出来的错；悄悄出的错会一路错到底。'),
  enter:()=>({...P3}),expect:()=>S.R>0,showMe:{R:2},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Some failures announce themselves: a tool returns an error, a test fails. These are ${DEF('loud',TIP.loud())}. Others do not: the wrong file gets edited, a number is copied wrong. These are ${DEF('silent',TIP.silent())}.</p><p>A loud failure can be ${DEF('retried',TIP.retry())}. After a silent one, the later steps carry on from the wrong result. In this lab, 70% of this agent’s failures are loud (an assumption).</p><p>Allow retries first, then lower the share of loud failures:</p>`,
              `<p>有的错会自己报出来：工具返回错误，测试没通过。这叫${DEF('报出来的错',TIP.loud())}。有的错不会：改错了文件，把数字抄错了。这叫${DEF('悄悄出的错',TIP.silent())}。</p><p>报出来的错可以${DEF('重试',TIP.retry())}。悄悄出了错，后面的步骤会在错的结果上接着做。在这个 Lab 里，这个 agent 的错有 70% 会报出来（这是假设）。</p><p>先允许重试，再把报出来的错的比例调低：</p>`)+
   A.ui.seg({id:'arl-ic-R',key:'R',label:tx('Retries per step','每一步最多重试'),options:[0,1,2,3].map(v=>[v,String(v)]),value:S.R,inline:true})+
   A.ui.range({id:'arl-ic-r',key:'r',label:tx('Share of failures that are loud','报出来的错占全部错的'),min:0,max:10,value:Math.round(S.r*10),out:fPs(S.r),inline:true}),
  obs:()=>obsLoud()},
 {id:'check',title:{en:'Make failures show',zh:'让错误露出来'},
  lead:()=>tx('A check after every step costs more calls and buys a higher success rate.','每一步做完检查一次，多花一些调用，换回成功率。'),
  enter:()=>{lv4='none'; tried4=new Set(['none']); return {...P4};},expect:()=>tried4.size>1,showMe:()=>setLv(4,'ck'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Retries only rescue the failures someone can see. So first make the failures show.</p>`+
              `<p>A <b>check</b> is one more model call after a step is done, to judge whether the step is right. For example, the model runs the tests, or reads the output again. A redone step is checked too, so every attempt costs one more call. Here 30% of the failures are silent, and the check finds <span class="arl-cval">${fPs(S.c)}</span> of those (an assumption).</p>`+
              `<p>A <b>retry</b> redoes a step that went wrong, but only when the error is seen. The retries per step are capped; when they run out, the task stops. With a check, an error the check finds counts as seen too.</p><p>Pick a setting and see how much success rate the extra calls buy.</p>`,
              `<p>重试只救得了看得见的错。所以先要让错误露出来。</p>`+
              `<p><b>检查</b>就是一步做完以后，再调用一次模型，判断这一步做得对不对。比如让模型跑一遍测试，或者把输出再看一遍。重做的那一次也要检查，所以每次尝试都多一次模型调用。这个 Lab 假设，出的错里有 30% 是悄悄出的，检查能发现其中的 <span class="arl-cval">${fPs(S.c)}</span>。</p>`+
              `<p><b>重试</b>就是把出错的这一步重做，前提是错被看见了。重做有次数上限，用完了任务就停下。加上检查以后，检查发现的错也算被看见了。</p><p>选一种设置，看看多花的调用能换回多少成功率。</p>`)+leversHTML('arl-ic-lv4',LV4,lv4,tried4,lvRes4,tx('Pick a check setting','选一种检查的设置')),
  obs:()=>obsCheck()},
 {id:'retry',title:{en:'A retry has to change something',zh:'重试要带上新信息'},
  lead:()=>tx('The same call again often fails the same way; give the model the error message.','原样重试，常常原样再错；把报错信息交给模型再试。'),
  enter:()=>{lv5='info'; tried5=new Set(['info']); return {...P5};},expect:()=>tried5.size>1,showMe:()=>setLv(5,'blind'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Retrying the same call often fails the same way. The common practice: when a tool fails, the ${DEF('harness',TIP.harness())} hands the error message back to the model as that tool call’s result, marked as an error. The model sees what went wrong and can fix it on the next turn.</p><p>In this lab a retry with the error message is right 95% of the time, as often as a first attempt; the same call again only 50% (both are assumptions).</p><p>Pick a way to retry and see how the task success rate changes.</p>`,
              `<p>原样重试，常常原样再错。常见的做法是：工具出错时，${DEF('harness',TIP.harness())} 把报错信息作为这次工具调用的结果交回给模型，并标上“出错”。模型看到哪里错了，下一轮就能改。</p><p>在这个 Lab 里，带着报错信息重试，和第一次做一样有 95% 的把握；原样再调用一次，只有 50% 的把握（两个数都是假设）。</p><p>选一种重试的办法，看看任务的成功率怎么变。</p>`)+leversHTML('arl-ic-lv5',LV5,lv5,tried5,lvRes,tx('Pick a way to retry','选一种重试的办法')),
  obs:()=>obsRetry()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Chances multiply; retries rescue only loud failures; make failures show first.','把握要连乘；重试只救报出来的错；先让错误露出来。'),
  enter:()=>{remView=''; return {...P4};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
function obsMission(){
  if(!pred) return '';
  if(!runsSeen){const t=tallyNow(); return tx(`<p>The tasks are running: ${t.n} of ${RUNS} have a result, ${t.ok} succeeded so far.</p>`,`<p>任务正在跑：${RUNS} 个里已经有 ${t.n} 个出了结果，目前成功 ${t.ok} 个。</p>`);}
  const r=R_(), t=tallyOf(runsOf(S)), per100=Math.round(r.succ*100);
  return tx(`<p><b>${t.ok} of the ${RUNS} tasks in the chart succeeded. By the formula, the success rate is ${fP(r.succ)}, about ${per100} in 100 tasks.</b></p>`,`<p><b>图里的 ${RUNS} 个任务成功了 ${t.ok} 个。按公式算，成功率是 ${fP(r.succ)}，也就是 100 个任务里大约 ${per100} 个。</b></p>`)+verdict()+
    tx('<p>Why: a task fails as soon as one of its 20 steps goes wrong. Each step is one more chance to fail, and only about a third of the tasks get past all 20.</p>','<p>原因：20 步里只要有一步做错，任务就失败了。每一步都是一次出错的机会，能躲过全部 20 次的任务，只有三分之一左右。</p>');
}
function obsChain(){
  const r=R_(), b=compute({...BASE}), cur=match(LV2);
  if(!cur) return tx(`<p><b>${stepsW(S.n)} at ${fPc(S.p)} per step: the task succeeds ${fP(r.succ)} of the time.</b> This matches none of the options; the curve in the chart shows every number of steps.</p>`,`<p><b>${stepsW(S.n)}、每步 ${fPc(S.p)}：任务成功率 ${fP(r.succ)}。</b>这组设置不是上面的任何一个选项；图里的曲线给出每一种步数的成功率。</p>`);
  if(cur==='base') return tx(`<p><b>Initial setup: 20 steps at 95% each, the task succeeds ${fP(b.succ)} of the time.</b> Pick a way to raise it.</p>`,`<p><b>初始设置：20 步、每步 95%，任务成功率 ${fP(b.succ)}。</b>选一个办法把它提上去。</p>`);
  const fails=n=>fixd(n,1).replace(/\.0$/,'');
  if(cur==='half') return tx(`<p><b>Half the steps: the success rate goes from ${fP(b.succ)} to ${fP(r.succ)}.</b></p><p>Why: what matters is how many steps go wrong in a task, on average. At 95% per step, 20 steps have ${fails(20*0.05)} failed step on average; 10 steps have ${fails(10*0.05)}.${tried2.has('steady')?'':' Now try steadier steps.'}</p>`,
    `<p><b>步数减半：成功率从 ${fP(b.succ)} 升到 ${fP(r.succ)}。</b></p><p>原因：关键是一个任务里平均有几步会做错。每步 95% 时，20 步里平均有 ${fails(20*0.05)} 步做错，10 步里平均 ${fails(10*0.05)} 步。${tried2.has('steady')?'':'再试试“每步更稳”。'}</p>`);
  return tx(`<p><b>Steadier steps: the success rate goes from ${fP(b.succ)} to ${fP(r.succ)}.</b></p><p>Why: at 99% per step, 20 steps have only ${fails(20*0.01)} failed steps on average, fewer than the ${fails(10*0.05)} of 10 steps at 95%. So making each step steadier helps more than halving the steps here.${tried2.has('half')?'':' Now try half the steps.'}</p>`,
    `<p><b>每步更稳：成功率从 ${fP(b.succ)} 升到 ${fP(r.succ)}。</b></p><p>原因：每步 99% 时，20 步里平均只有 ${fails(20*0.01)} 步做错，比每步 95%、走 10 步时的 ${fails(10*0.05)} 步还少。所以在这里，让每一步更稳比把步数减半更管用。${tried2.has('half')?'':'再试试“步数减半”。'}</p>`);
}
function obsLoud(){
  const r=R_(), b=compute({...P3});
  if(S.R===0) return tx(`<p><b>No retries: the success rate is still ${fP(r.succ)}.</b> Every loud failure stops the task. Allow a retry first.</p>`,`<p><b>不重试：成功率还是 ${fP(r.succ)}。</b>每个报出来的错都让任务停下。先允许重试。</p>`);
  const res=tx(`<p><b>${S.R===1?'1 retry':`${S.R} retries`}, ${fPs(S.r)} of failures loud: the success rate goes from ${fP(b.succ)} to ${fP(r.succ)}.</b></p>`,`<p><b>重试 ${S.R} 次、报出来的错占 ${fPs(S.r)}：成功率从 ${fP(b.succ)} 升到 ${fP(r.succ)}。</b></p>`);
  if(S.r>=1) return res+tx('<p>Why: every failure is loud here, so a retry can catch every one of them; only a step that fails on every try stops the task.</p>','<p>原因：这里每个错都会报出来，重试都接得住；只有每次都失败的那一步才会让任务停下。</p>');
  const why=tx(`<p>Why: a retry only happens when the ${DEF('harness',TIP.harness())} sees the failure. Almost every failure left is silent: ${fP(r.wrong)} of the tasks run to the end on a wrong result (the hatched cell and the faded cells after it).</p>`,`<p>原因：只有 ${DEF('harness',TIP.harness())} 看见了错，才会重试。剩下的失败几乎都是悄悄出的错：${fP(r.wrong)} 的任务带着错一路做到最后（图里的斜纹格，和它后面变淡的格子）。</p>`);
  const lower=S.r<0.7?tx(`<p>With fewer loud failures, fewer can be rescued: at 70% loud the same retries gave ${fP(compute({...P3,R:S.R}).succ)}.</p>`,`<p>报出来的错越少，能救回来的就越少：报出来的错占 70% 时，同样的重试能做到 ${fP(compute({...P3,R:S.R}).succ)}。</p>`)
    :(S.r===0.7?tx('<p>Now lower the share of loud failures.</p>','<p>再把报出来的错的比例调低试试。</p>'):'');
  return res+why+lower;
}
function obsCheck(){
  const r=R_(), b=compute({...P4}), cur=match(LV4);
  if(!cur) return tx(`<p><b>The task succeeds ${fP(r.succ)} of the time, with ${fC(r.calls)} model calls per task.</b> This matches none of the options.</p>`,`<p><b>任务成功率 ${fP(r.succ)}，每个任务 ${fC(r.calls)} 次模型调用。</b>这组设置不是上面的任何一个选项。</p>`);
  if(cur==='none') return tx(`<p><b>No check, up to 2 retries: the task succeeds ${fP(b.succ)} of the time, with ${fC(b.calls)} model calls per task.</b> Pick a setting with a check.</p>`,`<p><b>不检查、最多重试 2 次：任务成功率 ${fP(b.succ)}，每个任务 ${fC(b.calls)} 次模型调用。</b>选一种带检查的设置试试。</p>`);
  if(cur==='ck') return tx(`<p><b>A check after each step, up to 2 retries: the success rate goes from ${fP(b.succ)} to ${fP(r.succ)}, and the model calls per task from ${fC(b.calls)} to ${fC(r.calls)}.</b></p><p>Why: the check finds most of the silent failures, and a found error is redone like a reported one. The price is one more call after every attempt, so the calls about double.</p>`,
    `<p><b>每步检查、最多重试 2 次：成功率从 ${fP(b.succ)} 升到 ${fP(r.succ)}，每个任务的模型调用从 ${fC(b.calls)} 次涨到 ${fC(r.calls)} 次。</b></p><p>原因：悄悄出的错大多被检查发现了，发现的错和报出来的错一样会重做。代价是每次尝试之后都多一次调用，调用差不多翻了一倍。</p>`);
  const n0=compute({...S,ck:0,R:0});
  return tx(`<p><b>A check but no retries: the success rate drops to ${fP(r.succ)}, the same as with no check and no retries.</b></p><p>Why: the check finds the error, but with no retries the step is not redone, so the task just stops.</p><p>What the check buys is that the failures become known. With no check and no retries, ${fP(n0.wrong)} of the tasks finish wrong and nobody knows. With the check, only ${fP(r.wrong)} do; the other failed tasks stop and report the failure. The price: the model calls per task go from ${fC(n0.calls)} to ${fC(r.calls)}.</p>`,
    `<p><b>只检查、不重试：成功率掉到 ${fP(r.succ)}，和不检查、不重试时一样。</b></p><p>原因：检查发现了错，可是不重试，这一步就不会重做，任务只能停下。</p><p>检查的用处，是让失败变得看得见。不检查、不重试时，有 ${fP(n0.wrong)} 的任务做完了却是错的，谁也不知道。加上检查，这样的任务只剩 ${fP(r.wrong)}，其余失败的任务都停了下来，报告了失败。代价是每个任务的模型调用从 ${fC(n0.calls)} 次涨到 ${fC(r.calls)} 次。</p>`);
}
function obsRetry(){
  const r=R_(), b=compute({...P5}), cur=match(LV5);
  if(!cur) return tx(`<p><b>The task succeeds ${fP(r.succ)} of the time.</b> This matches none of the options.</p>`,`<p><b>任务成功率 ${fP(r.succ)}。</b>这组设置不是上面的任何一个选项。</p>`);
  if(cur==='info') return tx(`<p><b>Retries with the error message: the task succeeds ${fP(b.succ)} of the time.</b> Pick another way to retry.</p>`,`<p><b>带着报错信息重试：任务成功率 ${fP(b.succ)}。</b>换一种重试方式试试。</p>`);
  const pk=tx(`<p class="small">Retrying a whole task is a different matter: the lab ${labLink('pass-k')} looks at k tries of the same task.</p>`,`<p class="small">把整个任务重跑几遍是另一回事：${labLink('pass-k')} 那个 Lab 讲同一个任务试 k 次。</p>`);
  if(cur==='blind') return tx(`<p><b>The same call again: the success rate drops from ${fP(b.succ)} to ${fP(r.succ)}.</b></p><p>Why: the model cannot see what went wrong, so half of the retries repeat the mistake. There are as many retries as before, but they rescue fewer failures.</p>`,
    `<p><b>原样重试：成功率从 ${fP(b.succ)} 掉到 ${fP(r.succ)}。</b></p><p>原因：模型看不到哪里错了，一半的重试会再犯同样的错。重试的次数没变，能救回来的错却少了。</p>`)+pk;
  return tx(`<p><b>No retries: the success rate drops from ${fP(b.succ)} to ${fP(r.succ)}.</b> Every loud failure now stops the task.</p>`,`<p><b>不重试：成功率从 ${fP(b.succ)} 掉到 ${fP(r.succ)}。</b>现在每个报出来的错都让任务停下。</p>`)+pk;
}
/* part 6: each line with this lab's example; a button puts it back in the chart (R6.23) */
const REMS={chain:{s:{...BASE,p:0.99}},loud:{s:{...P3,R:2}},check:{s:{...P4,ck:1}}};
function remBtn(k){const on=remView===k;
  const tip={chain:tx('Shows 20 steps at 99% per step in the chart.','在图里显示每步 99%、20 步的任务。'),loud:tx('Shows the agent with 2 retries and no check in the chart.','在图里显示重试 2 次、不检查的 agent。'),check:tx('Shows the agent with a check after each step and 2 retries.','在图里显示每步检查、重试 2 次的 agent。')}[k];
  return ` <button type="button" class="linkbtn arl-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const b=compute({...BASE}), h=compute({...BASE,n:10}), st=compute({...BASE,p:0.99}), l=compute({...P3,R:2}), c=compute({...P4,ck:1});
  return tx(`<ul class="arl-remlist"><li><b>The chances of the steps multiply.</b> Over many steps, a small shortfall per step becomes a big one for the whole task. So a long task needs very steady steps, and a small gain per step raises its success rate a lot.${remBtn('chain')}</li>`+
    `<li><b>Retries rescue only loud failures.</b> A retry needs a failure that someone sees. However many retries you allow, the silent failures stay, and they cap the success rate.${remBtn('loud')}</li>`+
    `<li><b>Make failures show, then retry.</b> A check lets the silent failures be seen, so retries can fix them too. It costs more model calls, and buys back much of the success rate the silent failures took.${remBtn('check')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “95% per step is reliable enough.” Over 20 steps it leaves ${fP(b.succ)}: about 64 of 100 tasks fail.</p><p>✗ “With retries, any failure can be rescued.” Only loud failures trigger a retry; a silent failure is never seen, so no retry ever happens.</p><p>✗ “Splitting a task into more, smaller steps is always safer.” Each step is one more chance to fail: unless every small step is much steadier, more steps mean a lower success rate.</p></div>${A.shell.toScenarios()}`,
    `<ul class="arl-remlist"><li><b>每一步的把握要连乘。</b>步数一多，每步差一点，整个任务就差很多。所以长任务要求每一步都很稳，每步提高一点，成功率就能涨一大截。${remBtn('chain')}</li>`+
    `<li><b>重试只救报出来的错。</b>重试的前提是有人看见了错。重试次数再多，悄悄出的错也还在，成功率就卡在它们留下的上限。${remBtn('loud')}</li>`+
    `<li><b>先让错误露出来，再重试。</b>检查让悄悄出的错被看见，重试才能去改。代价是多花一些模型调用，换回的是悄悄出的错拿走的那部分成功率。${remBtn('check')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “每步 95% 已经够可靠了。”20 步连乘下来只剩 ${fP(b.succ)}：100 个任务里约有 64 个失败。</p><p>✗ “加上重试，什么错都能救回来。”只有报出来的错才会触发重试；悄悄出的错没人看见，重试根本不会发生。</p><p>✗ “把任务拆成更多更小的步骤，总是更稳。”每多一步，就多一次出错的机会；除非每一小步都稳得多，步数越多，成功率越低。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios (all rates are the lab's assumptions)
   ===================================================================== */
const PRESETS=[
 {id:'code',t:{en:'A coding agent fixing a failing test',zh:'修 bug 的 coding agent'},d:{en:'The tests tell it at once when it is wrong: most failures are loud',zh:'测试一跑就知道对不对：大部分错会报出来'},s:{n:20,p:0.95,r:0.9,R:2,ck:0,kind:'i'}},
 {id:'research',t:{en:'A research agent copying numbers from web pages',zh:'从网页上抄数字的调研 agent'},d:{en:'A number copied wrong raises no error: most failures are silent',zh:'数字抄错了也不会报错：大部分错是悄悄出的'},s:{n:15,p:0.95,r:0.2,R:2,ck:0,kind:'i'}},
 {id:'long',t:{en:'A 50-step code migration',zh:'走 50 步的代码迁移'},d:{en:'It runs the tests after every step before it moves on',zh:'每一步做完都跑测试，再往下走'},s:{n:50,p:0.97,r:0.7,R:2,ck:1,kind:'i'}},
 {id:'short',t:{en:'A 5-step meeting-room booking assistant',zh:'只走 5 步的订会议室助手'},d:{en:'A short task, where 90% per step is enough',zh:'任务很短，每步 90% 也够用'},s:{n:5,p:0.90,r:0.8,R:2,ck:0,kind:'i'}}];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k])&&S.c===BASE.c);
function scenHTML(i){
  const p=PRESETS[i], r=compute({...BASE,...p.s});
  if(p.id==='code') return tx(`<p><b>A coding agent, 20 steps, 2 retries: the task succeeds ${fP(r.succ)} of the time.</b></p><p>Its failures are mostly loud: a test fails, a tool returns an error. So the retries catch almost all of them, and few failures slip through silently. Assumed: 95% per step, 90% of failures loud.</p>`,
                                `<p><b>修 bug 的 coding agent：20 步，重试 2 次，任务成功率 ${fP(r.succ)}。</b></p><p>它的错大多会报出来：测试没通过，工具返回错误。所以重试几乎都接得住，悄悄出的错很少。这里假设每步 95%，90% 的错会报出来。</p>`);
  if(p.id==='research') return tx(`<p><b>A research agent, only 15 steps, 2 retries as well: the task succeeds only ${fP(r.succ)} of the time.</b></p><p>Its failures are mostly silent: a number copied wrong raises no error, so no retry ever happens. This is the kind of agent that needs a check most, for example going back to the page to compare the number. Assumed: 95% per step, 20% of failures loud.</p>`,
                                    `<p><b>调研 agent：只有 15 步，也重试 2 次，任务成功率却只有 ${fP(r.succ)}。</b></p><p>它的错大多是悄悄出的：数字抄错了，不会报任何错，重试也就无从发生。这类 agent 最需要检查，比如回到原网页把数字再核对一遍。这里假设每步 95%，20% 的错会报出来。</p>`);
  if(p.id==='long') return tx(`<p><b>A 50-step migration with a check after each step and 2 retries: the task succeeds ${fP(r.succ)} of the time.</b></p><p>Over many steps, silent failures add up. Running the tests after every step turns most of them into loud ones, so a long task can get to the end. The price is about twice the model calls: ${fC(r.calls)} per task. Assumed: 97% per step, 70% of failures loud.</p>`,
                                `<p><b>50 步的代码迁移：每步检查、重试 2 次，任务成功率 ${fP(r.succ)}。</b></p><p>步数一多，悄悄出的错就会积少成多。每一步都跑测试，把大部分错变成看得见的错，长任务才走得完。代价是模型调用差不多翻倍：每个任务 ${fC(r.calls)} 次。这里假设每步 97%，70% 的错会报出来。</p>`);
  const r20=compute({...BASE,...p.s,n:20});
  return tx(`<p><b>A booking assistant, 5 steps at only 90% each, 2 retries: the task still succeeds ${fP(r.succ)} of the time.</b></p><p>Few steps mean few chances to fail. The same assistant on a 20-step task would succeed only ${fP(r20.succ)} of the time. Assumed: 80% of failures loud (the booking system rejects a bad request).</p>`,
            `<p><b>订会议室的助手：5 步，每步只有 90%，重试 2 次，任务成功率也有 ${fP(r.succ)}。</b></p><p>步数少，出错的机会就少。同样的助手放到 20 步的任务里，成功率只有 ${fP(r20.succ)}。这里假设 80% 的错会报出来（预订系统会拒绝不合法的请求）。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chName=o=>{const parts=[o.strong?tx('the 98%-per-step model','每步成功率 98% 的模型'):tx('the 95%-per-step model','每步成功率 95% 的模型'),o.ck?tx('check','检查'):tx('no check','不检查'),retriesW(o.R),stepsW(o.n)]; return parts.join(tx(', ','，'));};
function chHint(e){
  if(!e.okS&&!e.ck&&e.R===0) return tx('Hint: with no retries, every loud failure stops the task. Allow retries first and see how many tasks they rescue.','提示：现在不重试，一报错，任务就停下。先允许重试，看看能救回多少任务。');
  if(!e.okS&&!e.ck&&!e.strong) return tx('Hint: almost every failure left is silent, and retries never see those. How can they be made to show?','提示：剩下的失败几乎都是悄悄出的错，重试根本看不见它们。怎样让它们露出来？');
  if(!e.okS&&e.ck&&e.R===0) return tx('Hint: the check finds the error, but with no retries the step is not redone, so the task just stops.','提示：检查发现了错，可是不重试，这一步就不会重做，任务只能停下。');
  if(!e.okS&&e.strong&&!e.ck) return tx('Hint: the 98%-per-step model makes each step steadier, but its silent failures still go unseen.','提示：每步成功率 98% 的模型让每一步更稳，可它悄悄出的错照样没人看见。');
  if(!e.okC&&e.ck&&e.n===20) return tx('Hint: a check doubles the calls. Fewer steps mean fewer calls.','提示：检查让调用翻倍。步数少了，调用就少了。');
  if(!e.okC&&e.strong) return tx('Hint: each call to the 98%-per-step model counts 2, so its cost grows fast.','提示：每步成功率 98% 的模型调用一次记 2 次，调用成本涨得快。');
  return tx('Hint: look at which target is missed, success or cost, and change the choice that drives it.','提示：先看差的是哪一个目标，成功率还是调用成本，再改影响它的那个选择。');
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const e=chEval(chO), best=chBest();
  const good=`<p class="okline arl-mt"><b>`+tx(`Goal met: ${fP(e.succ)} of the tasks succeed, at a cost of ${fC(e.cost)} calls per task.`,`达成目标：任务成功率 ${fP(e.succ)}，每个任务的调用成本是 ${fC(e.cost)} 次。`)+`</b>`;
  let h;
  if(e.ok&&best&&r1(e.cost)===r1(best.cost)&&e.R===best.R) h=good+tx(' And no passing combination costs less.</p>','而且没有哪个达标的组合成本更低。</p>');
  else if(e.ok) h=good+tx(' But a cheaper combination passes too.</p>','不过还有更省的组合也能达标。</p>')+tx(`<p>The cheapest is “${esc(chName(best))}”, at a cost of ${fC(best.cost)} calls per task (${fP(best.succ)} succeed).</p>`,`<p>最省的组合是“${esc(chName(best))}”，每个任务的调用成本是 ${fC(best.cost)} 次（成功率 ${fP(best.succ)}）。</p>`);
  else h=`<p class="badline arl-mt"><b>`+(!e.okS?tx(`Not yet: ${fP(e.succ)} of the tasks succeed, below 90%.`,`还没达到：任务成功率 ${fP(e.succ)}，不到 90%。`):tx(`Not yet: the cost is ${fC(e.cost)} calls per task, more than 40.`,`还没达到：每个任务的调用成本是 ${fC(e.cost)} 次，超过了 40 次。`))+`</b></p>`+
    `<p class="small">${chHint(e)}</p>`+(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the chart (the owner's review of batch 8, 2026-10-07): every combination goes in, and the brief chart keeps the ones that pass, the
     initial setup and the pick; the failing alternatives the text names are said in words, with their numbers in a hover */
  const cap1=t=>t.charAt(0).toUpperCase()+t.slice(1);
  const all=chAll(), key=o=>[o.strong,o.ck,o.R,o.n].join(), k0=key(CH.start), shown=all.filter(o=>o.ok||key(o)===key(e)||key(o)===k0);
  h+=A.ui.cmpChart({title:tx('Share of the report agent’s tasks that succeed (goal: at least 90%, at a cost of at most 40 calls per task)','报表 agent 的任务成功率（目标：至少 90%，且每个任务的调用成本不超过 40 次）'),
    aria:tx('Success and cost: ','成功率和调用成本：')+shown.map(o=>`${chName(o)}${tx(': ','：')}${fP(o.succ)}${tx(', ','，')}${fC(o.cost)}${tx(' calls',' 次')}`).join(tx('; ','；')),
    rows:all.map(o=>({label:chName(o),value:o.succ*100,text:tx(`${fP(o.succ)}, ${fC(o.cost)} calls`,`${fP(o.succ)}，${fC(o.cost)} 次`),ok:o.ok,me:key(o)===key(e),
      tags:(key(o)===k0?[{text:tx('initial setup','初始设置')}]:[]).concat(key(o)===key(e)?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,hiddenText:n=>tx(`The other ${n} combinations miss at least one target and are not shown.`,`其余 ${n} 种组合至少差一个目标，图里没有列出。`),
    goal:90,goalLabel:tx('goal: 90%','目标：90%'),max:100,okText:tx('meets both targets','两个目标都达到'),noText:tx('misses at least one','至少差一个'),goalText:tx('goal: 90% of tasks','目标：90% 的任务')});
  const st=chEval(CH.start), rOnly=chEval({strong:0,ck:0,R:3,n:20}), ck20=chEval({strong:0,ck:1,R:1,n:20}), s12=[1,2,3].map(R=>chEval({strong:1,ck:0,R,n:12})).sort((a,b)=>b.succ-a.succ)[0];
  h+=tx(`<p class="small">The initial setup, the 95%-per-step model with no check and no retries over 20 steps, succeeds ${fP(st.succ)} of the time. Retries alone ${DEF('stay below 90%',`${cap1(chName(rOnly))}: ${fP(rOnly.succ)} of the tasks succeed.`)}, because half of the failures are silent and retries never see them.</p>`,
        `<p class="small">初始设置用每步成功率 95% 的模型，不检查，不重试，走 20 步，成功率是 ${fP(st.succ)}。只靠重试，成功率${DEF('还是不到 90%',`${chName(rOnly)}，成功率是 ${fP(rOnly.succ)}。`)}，因为一半的错是悄悄出的，重试看不见它们。</p>`);
  h+=tx(`<p class="small">Only a check makes the silent failures show. But at 20 steps, a check with retries ${DEF('goes over the budget',`${cap1(chName(ck20))}: ${fC(ck20.cost)} calls per task, and more retries cost more.`)}. With bigger tools and 12 steps, the same check costs about ${fC(best.cost)} calls. Each call to the 98%-per-step model counts 2, so at 20 steps it is over the budget. At 12 steps without a check, its silent failures ${DEF('keep it below 90%',`At most ${fP(s12.succ)} of the tasks succeed (${chName(s12)}).`)}; with a check, its cost goes over the budget again.</p>`,
        `<p class="small">只有检查能让悄悄出的错露出来。可是在 20 步时，检查加上重试，调用成本${DEF('超出了预算',`${chName(ck20)}，每个任务要 ${fC(ck20.cost)} 次调用，重试越多越贵。`)}。换成更大的工具、合并成 12 步，同样的检查只要约 ${fC(best.cost)} 次。每步成功率 98% 的模型调用一次记 2 次，走 20 步时超出预算。到了 12 步，不加检查，它悄悄出的错让成功率${DEF('达不到 90%',`成功率最多是 ${fP(s12.succ)}（${chName(s12)}）。`)}。加上检查，调用成本又超出了预算。</p>`);
  h+=tx('<p class="small muted">Note: here a check never raises a false alarm, and a merged step is as reliable as a small one. In a real system a bigger step can be harder to get right; measure each step’s success rate on your own tasks before you rely on these numbers.</p>',
        '<p class="small muted">注意：这里假设检查从不误报，合并后的一步和原来的一小步一样有把握。真实系统里，大工具的一步可能更难做对；要靠这些数做决定之前，先用自己的任务测一测每一步的把握。</p>');
  return h;
}

/* =====================================================================
   header copy, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Predict first, then see how much the success rate falls with every step. Then see which failures a retry can rescue, and why a check is worth a few extra calls. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看每多一步，成功率会掉多少。接着看重试能救回哪些错，以及为什么值得多花几次调用来做检查。第一次学从这里开始。`),
  sandbox:tx('Four common agents, which fix a bug, copy numbers from web pages, migrate code in 50 steps, and book a meeting room in 5. Every per-step rate is assumed. Pick one to see how its failures behave and how often it succeeds.',
             '四个常见的 agent，分别在修 bug、从网页抄数字、走 50 步迁移代码、只走 5 步订会议室。每一步的把握都是假设的。选一个，看它的错是什么样、能成功几成。'),
  challenge:tx('A 20-step report agent. Combine a model that is right more often per step, a check, retries and fewer steps so that at least 90 of 100 tasks succeed, at a cost of at most 40 calls per task on average. The 98%-per-step model costs more: each of its calls counts 2.',
               '一个 20 步的报表 agent。把每步成功率更高的模型、检查、重试和更少的步骤组合起来，让 100 个任务至少成功 90 个，同时平均每个任务的调用成本不超过 40 次。每步成功率 98% 的模型更贵，调用一次记 2 次。')
});
const SRC={react:'https://arxiv.org/abs/2210.03629',tau:'https://arxiv.org/abs/2406.12045',swe:'https://arxiv.org/abs/2405.15793'};
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> the chart simulates ${RUNS} tasks with seeded random numbers, as an illustration. The initial seed, ${BASE.seed}, was picked on purpose, because the number of its ${RUNS} tasks that succeed is close to what the formula gives. Every number on the cards comes from a closed form, and ${J('math','“Show the math”')} works it out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>A task is n steps; every step must end right. Each attempt at a step succeeds with the per-step rate; a failure is loud (seen, retried) or silent (unseen, the task goes on, wrong, to the end).</li>`+
 `<li>A loud failure can be retried, up to a cap; when the retries run out, the task stops. A check can follow every attempt: one more call, and part of the silent failures get reported.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>This lab assumes these rates:<ul><li>a step is right 95% of the time;</li><li>70% of the failures are loud;</li><li>of the silent failures, the check finds 80%;</li><li>a retry with the error message is as good as a first attempt;</li><li>the same call again is right only 50% of the time.</li></ul></li>`+
 `<li>Steps fail independently and at the same rate; real steps differ (editing a file is harder than reading one), and one mistake can make the next more likely. A check never raises a false alarm here.</li>`+
 `<li>A model call is the unit of cost; tool calls, tokens and waiting time are not counted.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>No outside rate is used: ${M('p^n')} needs no source. The loop of a step (the model calls a tool and reads its result): ${cite(SRC.react,'ReAct, arXiv 2210.03629')}.</li>`+
 `<li>How a harness (the program that runs the agent) retries and returns tool errors: the two interview problems under “Go further” in the challenge.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-03. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>图里的 ${RUNS} 个任务用带种子的随机数模拟，作为示意。初始的种子 ${BASE.seed} 是特意挑的，因为它抽出的 ${RUNS} 个任务里，成功的任务和公式算的差不多一样多。指标卡上的数字都由公式算出，${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>一个任务有 n 步，每一步都要做对。每次尝试做对的把握是固定的。做错时，要么是报出来的错（看得见，可以重试），要么是悄悄出的错（没人看见，任务带着错一路做到最后）。</li>`+
 `<li>报出来的错可以重试，但次数有上限，用完了任务就停下。每次尝试之后还可以加一次检查：多花一次调用，让一部分悄悄出的错报出来。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>本 Lab 假设下面这些比例：<ul><li>每一步做对的把握是 95%。</li><li>70% 的错会报出来。</li><li>悄悄出的错里，检查能发现 80%。</li><li>带着报错信息重试，和第一次做一样有把握。</li><li>原样重试，只有 50% 的把握。</li></ul></li>`+
 `<li>每一步独立出错，把握都一样；真实的步骤难度不同（改文件比读文件难），一个错也可能让下一步更容易错。这里的检查不会误报。</li>`+
 `<li>成本只算模型调用，不算工具调用、token 和等待时间。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>没有用任何外部的成功率：${M('p^n')} 不需要出处。一步的循环（模型调用工具、读回结果）：${cite(SRC.react,'ReAct，arXiv 2210.03629')}。</li>`+
 `<li>运行 agent 的程序（harness）怎样重试、怎样把工具的错误交回模型：见挑战题“延伸练习”里的两道面试题。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-03。原型，尚未经专家审阅。</p></div></div>`);
/* the success rate with unlimited retries: a step ends right unless one of its failures is silent, p / (1 − q r) per step */
const retryCap=s=>Math.pow(s.p/(1-(1-s.p)*s.r),s.n);
const OUTCOME=()=>tx('estimate an agent’s end-to-end success from its per-step reliability, and decide whether to spend on fewer steps, a stronger model, retries or checks.','从每一步的把握估出 agent 整个任务的成功率，并判断该把钱花在少走几步、更强的模型、重试还是检查上。');
const EXITS=()=>{
  const b=compute({...BASE}), t40=compute({...BASE,n:40,p:0.98}), a=compute({...BASE,n:10,p:0.99}), c=compute({...BASE,n:30,p:0.995}), l=compute({...P3,R:2}), cap=retryCap(P3);
  return {explain:[
  {q:tx(`At 95% per step, a 20-step task succeeds only ${fP(b.succ)} of the time. Why?`,`每步 95%，20 步的任务只有 ${fP(b.succ)} 能成功。为什么？`),
   o:[tx('The task needs all 20 steps right, so the 20 chances of 95% multiply, and the product is about 36%.','任务要 20 步全对，所以 20 个 95% 要乘起来，乘出来只有 36% 左右。'),
      tx('Each wrong step makes the next ones more likely to go wrong, so the errors snowball over 20 steps.','每错一步，后面的步骤就更容易错，20 步下来错误越滚越大。'),
      tx('The model gets worse over a long task, so the later steps are right much less than 95% of the time.','模型越做越差，后面的步骤做对的把握远低于 95%。')],a:0,
   n:[null,tx('Here every step is right 95% of the time whatever happened before; nothing snowballs. The task fails simply because all 20 steps must be right.','这里每一步都有 95% 的把握，和前面发生了什么无关，错误没有越滚越大。任务失败，只是因为 20 步都得做对。'),
     tx('Here the 20th step is right 95% of the time, just like the first. The success rate falls because the chances multiply.','这里第 20 步和第 1 步一样有 95% 的把握。成功率掉下来，是因为把握要连乘。')],
   why:tx(`Success is ${M('p^n')}: each step multiplies in one more 0.95. Twenty of them leave about 0.358.`,`成功率是 ${M('p^n')}：每多一步就再乘一个 0.95，乘 20 次后约剩 0.358。`)},
  {q:tx(`An agent whose failures are 70% loud gets 2 retries. Its success rate rises from ${fP(b.succ)} to ${fP(l.succ)}, not to 100%. Why?`,`一个 agent 的错有 70% 会报出来。给它重试 2 次，成功率从 ${fP(b.succ)} 升到 ${fP(l.succ)}，没有到 100%。为什么？`),
   o:[tx('A retry happens only when a failure is seen; the silent failures are never seen, so those tasks run on, wrong, to the end.','只有看见了错，才会重试；悄悄出的错没人看见，这些任务带着错一路做到最后。'),
      tx('Two retries are too few; with ten retries the success rate would come close to 100%.','重试 2 次太少了；重试 10 次，成功率就会接近 100%。'),
      tx('Every retry brings in new errors of its own, which cancel part of the gain.','每次重试都会带来新的错误，抵消了一部分好处。')],a:0,
   n:[null,tx('More retries cannot rescue a silent failure: it never triggers a retry. At 2 retries, almost every loud failure is already rescued.','再多的重试也救不了悄悄出的错：它根本不会触发重试。重试 2 次时，报出来的错几乎已经全救回来了。'),
     tx('Here a retry is right as often as a first attempt and brings in no new kind of error; the loss comes from the failures nobody sees.','这里重试和第一次做一样有把握，不会带来新的错；损失来自没人看见的错。')],
   why:tx(`30% of this agent’s failures are silent. Retries never see them, so however many retries are allowed, the success rate cannot pass the ceiling they leave (about ${Math.round(cap*100)}% here).`,`这个 agent 的错有 30% 是悄悄出的。重试看不见这些错，所以重试次数再多，成功率也超不过它们留下的上限（这里约 ${Math.round(cap*100)}%）。`)}],
 transfer:[
  {q:tx(`Another agent runs 40 steps at 98% per step, with no retries. About how often does the whole task succeed? (For reference: 20 steps at 95% give ${fP(b.succ)}.)`,`另一个 agent 要走 40 步，每步 98% 的把握，不重试。整个任务大约有多大的把握成功？（参照：每步 95%、20 步时是 ${fP(b.succ)}。）`),
   o:[tx('About 20%','约 20%'),tx('About 45%','约 45%'),tx('About 70%','约 70%'),tx('About 90%','约 90%')],a:1,
   why:tx(`About ${fP(t40.succ)}. Count the failed steps on average: 40 steps at 2% give 0.8, close to the 1 of 20 steps at 5%. So the success rate is only a little higher than ${fP(b.succ)}.`,`约 ${fP(t40.succ)}。数一数平均会错几步：40 步、每步 2%，平均错 0.8 步，和 20 步、每步 5% 时的 1 步差不多。所以成功率只比 ${fP(b.succ)} 高一点。`)},
  {q:tx('Task A takes 10 steps at 99% each; task B takes 30 steps at 99.5% each. No retries. Which succeeds more often?','任务 A 走 10 步，每步 99%；任务 B 走 30 步，每步 99.5%。都不重试。哪个更容易成功？'),
   o:[tx('A','A'),tx('B','B'),tx('About the same','差不多')],a:0,
   why:tx(`A, ${fP(a.succ)} against ${fP(c.succ)}. B’s steps are steadier, but it has three times as many: on average A gets 0.1 steps wrong and B 0.15.`,`A：${fP(a.succ)}，B 是 ${fP(c.succ)}。B 的每一步更稳，可步数是 A 的 3 倍：平均下来，A 错 0.1 步，B 错 0.15 步。`)}],
 practice:[
  {id:'agent-tool-loop',d:tx('Parts 1 and 2: send a tool’s failure back to the model as a tool result marked is_error, and stop after a cap on calls. That is this lab’s retry with the error message, and its stopped task.','第 1、2 部分：把工具的失败作为标了 is_error 的 tool_result 交回模型，并在调用次数到上限时放弃。这就是这个 Lab 里“带着报错信息重试”和“任务停下”。')},
  {id:'agent-harness-eval',d:tx('Deep dive on timeouts, retries and error propagation: retry only the errors that are safe to repeat, hand the real error to the model, and abort a run after repeated identical failures.','深入讨论超时、重试和错误传递：只对可以安全重复的错误重试，把真实的报错交给模型，同一个调用连续失败就终止 run。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'agents',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Agent reliability','Agent 可靠性'),title:l?L(l.q):tx('95% reliable per step, 20 steps: how often does the agent succeed?','每步 95% 可靠，20 步的 agent 还剩多少成功率？'),
    minutes:10,fid:'Simulated',fidPop:FIDPOP(),descs:DESCS(),mode});
}
const speedBtns=()=>[['s',tx('Slow','慢')],['m',tx('Medium','中')],['f',tx('Fast','快')]];
function stageHTML(){
  return `<div class="panel stage arl-stage" id="arl-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="arl-st-title"></div><div class="meter-sub" id="arl-st-sub"></div></div><div class="stage-actions"><span class="status" id="arl-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph arl-graph" id="arl-graph">`+
    `<div class="arl-cap" id="arl-batch"></div>`+
    `<div class="arl-grid" id="arl-grid" tabindex="0" role="img"><canvas id="arl-cv"></canvas></div>`+
    `<div class="legend-row arl-legend" id="arl-legend"></div>`+
    `<div class="arl-player" id="arl-player">`+
     `<button type="button" class="btn sm arl-play" data-act="play" id="arl-play"></button>`+
     `<button type="button" class="btn ghost sm" data-act="stepf" data-tipx="${esc(tx('One step forward: every task takes its next step.','前进一步：每个任务都走下一步。'))}" aria-label="${esc(tx('One step forward','前进一步'))}">▸</button>`+
     `<button type="button" class="btn ghost sm" data-act="end" data-tipx="${esc(tx(`Jump to the end: all ${RUNS} tasks done.`,`跳到结尾：${RUNS} 个任务全部跑完。`))}" aria-label="${esc(tx('To the end','跳到结尾'))}">⏭</button>`+
     `<div class="seg-ctl arl-speed" role="group" aria-label="${esc(tx('Speed','速度'))}" data-key="speed">${speedBtns().map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${PL.speed===v}">${t}</button>`).join('')}</div>`+
     `<input type="range" class="arl-scrub" id="arl-scrub" min="0" max="140" value="0" aria-label="${esc(tx('Position in the replay','播放到哪里'))}">`+
    `</div>`+
    `<div class="arl-tally mono" id="arl-tally"></div>`+
    `<div class="arl-curve" id="arl-curve"><div class="arl-cap" id="arl-curve-cap"></div><div class="arl-svgw" id="arl-svgw" tabindex="0"></div></div>`+
   `</div>`+
   `<p class="arl-scope small muted" tabindex="0" data-tipx="${esc(tx('Every per-step rate on this page is this lab’s assumption. The task’s success follows exactly from those rates.','页面上每一步的把握都是本 Lab 的假设，任务的成功率由这些假设精确算出。'))}">${tx('Assumed per-step rates; every task’s chance follows from them exactly.','每一步的把握是假设的；任务的成功率由它们精确算出。')}</p>`+
   `<div class="metrics arl-metrics" id="arl-metrics"></div>`+
   `<details class="math" id="arl-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="arl-mathbody"></div></details></div>`;
}
const noteOf=(en,zh)=>`<div class="ctl-note">${tx(en,zh)}</div>`;
function controlsHTML(){
  return `<aside class="panel controls" id="arl-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="arl-lock" hidden></p>`+
   `<div class="lockable" id="arl-w-n">`+A.ui.range({id:'arl-ctl-n',key:'n',label:tx('Steps in the task','任务的步数'),min:1,max:50,value:S.n,out:String(S.n),note:tx('How many steps one task takes. Each step is one model call, usually followed by one tool call. Retries and checks make extra model calls.','一个任务要走几步。每一步调用一次模型，模型通常再调用一次工具。重试和检查要另外调用模型。')})+`</div>`+
   `<div class="lockable" id="arl-w-p">`+A.ui.range({id:'arl-ctl-p',key:'p',label:tx('Chance a step is right','一步做对的把握'),min:800,max:999,value:Math.round(S.p*1000),out:fPc(S.p),note:tx('At the initial setup this lab assumes 95%, for “a capable model doing a routine step”. The real chance depends on the model, the tools and how hard the step is, so measure it on your own tasks.','初始设置里，本 Lab 假设它是 95%，代表“能力不错的模型做一步常规操作”。真实的把握取决于模型、工具和这一步有多难，要在自己的任务上测。')})+`</div>`+
   `<div class="lockable" id="arl-w-r">`+A.ui.range({id:'arl-ctl-r',key:'r',label:tx('Share of failures that are loud','报出来的错占全部错的'),min:0,max:10,value:Math.round(S.r*10),out:fPs(S.r),note:tx('Assumed. It is high for work where the tools report errors and tests back it up (writing code, for example). It is low for work where a mistake looks like a normal result (copying numbers, for example).','这是假设。工具会报错、有测试兜底的工作（比如写代码），这个比例就高。做错了看上去也像正常结果的工作（比如抄数字），这个比例就低。')})+`</div>`+
   `<div class="lockable" id="arl-w-R">`+A.ui.seg({id:'arl-ctl-R',key:'R',label:tx('Retries per step','每一步最多重试'),options:[0,1,2,3].map(v=>[v,String(v)]),value:S.R,note:tx('After a loud failure, a step is redone at most this many times. When the retries run out, the task stops. This cap is the lab’s assumption. A real harness (the program that runs the agent) re-sends the same request once or twice by itself after a timeout or a dropped connection. After an error such as a failing test, it hands the error back to the model to fix on its next step. There is no separate retry count then, only the cap on the total number of steps (often called max_steps).','一步报了错，最多再重做几次。次数用完，任务就停下。这个次数是这个 Lab 的假设。真实的 harness（运行 agent 的程序）遇到超时、断线这类错，会把同一个请求自动重发一两次。遇到测试没通过这类错，它会把报错交回模型，让模型在下一步改正。这时没有单独的重试次数，只受总步数上限（常叫 max_steps）的约束。')})+`</div>`+
   `<div class="lockable" id="arl-w-ck">`+A.ui.seg({id:'arl-ctl-ck',key:'ck',label:tx('Check after each step','每一步做完检查'),options:[[0,tx('No','不检查')],[1,tx('Yes','检查')]],value:S.ck,note:tx('After every attempt at a step, one more model call judges whether the step is right. A redone step is checked too, so every attempt costs one more call. An error the check finds is redone, like a reported one.','每一步做完，再调用一次模型，判断这一步做得对不对。重做的那一次也要检查，所以每次尝试多一次调用。检查发现的错，和报出来的错一样会重做。')})+`</div>`+
   `<div class="lockable" id="arl-w-kind">`+A.ui.seg({id:'arl-ctl-kind',key:'kind',label:tx('How a retry is done','怎样重试'),options:[['i',tx('With the error message','带着报错信息')],['b',tx('The same call again','原样再来一次')]],value:S.kind})+`</div>`+
   `<div class="arl-grp lockable" id="arl-w-as"><div class="arl-gh">${tx('Assumed values','假设值')}</div>`+
   `<div id="arl-w-c">`+A.ui.range({id:'arl-ctl-c',key:'c',label:tx('Share of the silent failures the check finds','悄悄出的错里，检查能发现的比例'),min:0,max:10,value:Math.round(S.c*10),out:fPs(S.c),note:tx('Assumed: 80% in the initial setup. Tests catch most wrong edits, but not all. A second look at the output misses some wrong numbers.','这是假设，初始设置是 80%。测试能抓住大部分改错的地方，但不是全部。再看一遍输出，也会漏掉一些抄错的数。')})+`</div>`+
   `<div id="arl-w-pb">`+A.ui.range({id:'arl-ctl-pb',key:'pb',label:tx('A same-call retry is right','原样重试做对的把握'),min:0,max:20,value:Math.round(S.pb*20),out:fPs(S.pb),note:tx(`Assumed: 50%. With nothing new to go on, the model often repeats its mistake. The lower the sampling temperature, the more alike its answers to the same request tend to be (the ${labLink('sampling')} lab explains temperature).`,`假设是 50%。没有新信息，模型常常重复同样的错。采样的 temperature 调得越低，同一个请求得到的回答往往越接近（temperature 在${labLink('sampling')}那个 Lab 里讲）。`)})+`</div></div>`+
   `<div class="lockable arl-row" id="arl-w-seed"><button type="button" class="btn ghost sm" data-act="reseed">${tx('New draws','重新抽')}</button> <span class="small muted" id="arl-seed-out"></span>`+
   noteOf(`The chart’s ${RUNS} tasks are drawn with a fixed random seed, so the page shows the same tasks every time you open it. The initial seed, ${BASE.seed}, was picked on purpose, because the number of its ${RUNS} tasks that succeed is close to what the formula gives. “New draws” picks a new seed and draws ${RUNS} other tasks, with the same per-step chances.`,`图里的 ${RUNS} 个任务是用固定的随机种子抽出来的，所以每次打开页面，看到的任务都一样。初始的种子 ${BASE.seed} 是特意挑的，因为它抽出的 ${RUNS} 个任务里，成功的任务和公式算的差不多一样多。点“重新抽”，会换一个种子，抽出另外 ${RUNS} 个任务，每一步的把握不变。`)+`</div>`+
   `<div class="lockable" id="arl-w-reset"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import random

def step_ok(p, loud, retries):
    """Chance that one step ends right; a loud failure is retried."""
    q = 1 - p
    return p * sum((q * loud) ** k for k in range(retries + 1))

def task_success(n, p, loud=0.7, retries=0):
    return step_ok(p, loud, retries) ** n   # every step must be right

def run_once(n, p, loud, retries, rng):
    wrong = False
    for _ in range(n):
        for _ in range(retries + 1):
            if rng.random() < p:
                break                       # this attempt worked
            if rng.random() >= loud:        # silent: nobody sees it
                wrong = True
                break
        else:
            return "stopped"                # retries used up
    return "wrong" if wrong else "ok"

print(round(task_success(20, 0.95), 3))            # 0.358
print(round(task_success(20, 0.95, 0.7, 2), 3))    # 0.73
rng = random.Random(1)
runs = [run_once(20, 0.95, 0.7, 2, rng) for _ in range(10_000)]
print(runs.count("ok") / len(runs))                # __SIM__`;
const SIMOUT='0.7262';
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="arl-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="arl-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s closed form, and one seeded run of a task.','这个 Lab 的公式，和一次带种子的任务模拟。')}</p><pre class="code" id="arl-code">${esc(CODE.replace('__SIM__',SIMOUT))}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="arl-prod">`+
   tx(`<li>Harnesses put caps on every run: a maximum number of steps (often called <code>max_steps</code>) and a wall-clock timeout. When a cap is hit, the run stops and reports that it gave up, instead of spinning forever.</li>`+
      `<li>A harness retries only the errors that are safe to repeat, such as a timeout or a dropped connection, usually a couple of times with a growing wait. A call with invalid arguments is not retried as it is: the same call fails the same way. A tool that changes something carries an idempotency key, so a retry never does the same change twice.</li>`+
      `<li>A tool’s error goes back to the model as that tool call’s result, with the real error message and a flag such as <code>is_error</code>, so the model can fix its arguments or pick another tool. The same call failing several times in a row ends the run. Both interview problems under “Go further” in the challenge work through these designs.</li>`,
      `<li>harness 给每个 run 设上限：最多走多少步（常叫 <code>max_steps</code>），最长跑多久（wall-clock 超时）。到了上限，run 就停下并报告“放弃”，而不是一直空转。</li>`+
      `<li>harness 只对可以安全重复的错误重试，比如超时、连接断开，通常重试一两次，每次多等一会儿。参数不合法的调用不会原样重试：同样的调用只会同样地失败。会改动东西的工具带一个幂等键（idempotency key），重试时不会把同一个改动做两遍。</li>`+
      `<li>工具出错时，真实的报错信息作为这次工具调用的结果交回模型，并带上 <code>is_error</code> 这样的标记，模型就能改参数或换工具。同一个调用连续失败几次，就终止这个 run。挑战题“延伸练习”里的两道面试题都是在设计这些。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.react,'ReAct: Synergizing Reasoning and Acting in Language Models')}<span>${tx('Yao et al., arXiv 2210.03629: the loop this lab calls a step, in which the model reasons, calls a tool and reads the result before the next call','Yao 等，arXiv 2210.03629：这个 Lab 所说的“一步”的循环，模型先推理、调用工具、读回结果，再做下一次调用')}</span></li>`+
   `<li>${cite(SRC.tau,'τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains')}<span>${tx('Yao, Shinn, Razavi and Narasimhan, arXiv 2406.12045: scores an agent by pass^k, the chance it succeeds on every one of k tries of the same task; on its leaderboard the score falls as k grows','Yao、Shinn、Razavi、Narasimhan，arXiv 2406.12045：用 pass^k 给 agent 打分，也就是同一个任务试 k 次、每次都成功的概率；它的排行榜上，k 越大分数越低')}</span></li>`+
   `<li>${cite(SRC.swe,'SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering')}<span>${tx('Yang et al., NeurIPS 2024, arXiv 2405.15793: the tools an agent is given, its interface to the computer, shape how well it does; this lab’s bigger tools that merge steps are one such choice','Yang 等，NeurIPS 2024，arXiv 2405.15793：给 agent 的工具（它和计算机之间的接口）决定它做得多好；这个 Lab 里合并步骤的大工具就是这样的选择')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="arl-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="arl-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="arl-app m2" id="arl-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="arl-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how often its tasks succeed.','选一个场景，看看它的任务能成功几成。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="arl-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Get the report agent to 90 successes in 100','让报表 agent 100 次至少成功 90 次')}</h3>`+
   tx(`<p>A finance-report agent pulls numbers from several systems, adds them up, checks them and writes a report, in 20 steps. It gets each step right 95% of the time. Half of its failures are silent: a number copied wrong raises no error.</p><p>The goal is that at least 90 of 100 tasks succeed, at a cost of at most 40 calls per task on average.</p><p>The cost counts model calls. A call to the model that is right 95% of the time per step counts 1. The model that is right 98% of the time per step costs twice as much, so each of its calls counts 2. Combine the four choices:</p>`,
      `<p>一个整理财务报表的 agent，要从几个系统里取数、汇总、核对，最后写报告，一共 20 步。它每一步有 95% 的把握做对。它出的错有一半是悄悄出的，比如数字抄错了，不会报错。</p><p>目标是 100 个任务里至少成功 90 个，同时平均每个任务的调用成本不超过 40 次。</p><p>调用成本按模型调用的次数记。每步成功率 95% 的模型调用一次，记 1 次。每步成功率 98% 的模型贵一倍，调用一次记 2 次。下面四个办法可以组合：</p>`)+
   A.ui.seg({id:'arl-ic-chM',key:'chM',label:tx('Model','模型'),options:[[0,tx('Right 95% per step; each call counts 1','每步成功率 95%，调用一次记 1 次')],[1,tx('Right 98% per step, but dearer; each call counts 2','每步成功率 98%，更贵，调用一次记 2 次')]],value:chO.strong,inline:true})+
   A.ui.seg({id:'arl-ic-chC',key:'chC',label:tx('Check after each step','每一步做完检查'),options:[[0,tx('No','不检查')],[1,tx('Yes: one more call per attempt; it finds 80% of the silent failures','检查：每次尝试多调用 1 次，悄悄出的错里能发现 80%')]],value:chO.ck,inline:true})+
   A.ui.seg({id:'arl-ic-chR',key:'chR',label:tx('Retries per step','每一步最多重试'),options:CH.Rs.map(v=>[v,String(v)]),value:chO.R,inline:true})+
   A.ui.seg({id:'arl-ic-chN',key:'chN',label:tx('Steps','步数'),options:[[20,tx('20 steps','20 步')],[12,tx('12 steps, with bigger tools','用更大的工具合并成 12 步')]],value:chO.n,inline:true})+
   `<div class="goal mono arl-goal" id="arl-ch-g1"><span>${tx('Tasks that succeed','任务成功率')}</span><span><b id="arl-ch-s"></b>${tx(' (goal: at least 90%)','（目标：至少 90%）')}</span></div>`+
   `<div class="goal mono arl-goal" id="arl-ch-g2"><span>${tx('Cost per task, in calls','每个任务的调用成本')}</span><span><b id="arl-ch-c"></b>${tx(' (goal: at most 40)','（目标：不超过 40 次）')}</span></div>`+
   submit+`<div id="arl-ch-res" class="arl-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return !pred?tx('Pick an answer to continue, or Skip.','选一个答案才能继续，也可以跳过。'):tx(`Wait for the ${RUNS} tasks to finish, or press ⏭ in the chart.`,`等 ${RUNS} 个任务跑完才能继续，也可以点图里的 ⏭。`);
  return {chain:tx('Pick a way to raise the success rate to continue.','选一个办法才能继续。'),loud:tx('Allow at least 1 retry to continue.','允许至少重试 1 次才能继续。'),
    check:tx('Pick a check to continue.','选一种检查才能继续。'),retry:tx('Pick another way to retry to continue.','换一种重试方式才能继续。')}[id]||'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy()||tx('Try the part’s control to continue.','先动一动这一部分的设置才能继续。'),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#arl-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function setObs(box,html){if(box._h===html) return; const again=box._h!=null&&box._h!==''&&html!==''&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
function updateRail(){
  const rail=$('#arl-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(s.id==='mission'?!!pred:true);
    const lvs={chain:[LV2,tried2,lvRes],check:[LV4,tried4,lvRes4],retry:[LV5,tried5,lvRes]}[s.id];
    if(lvs){const cur=match(lvs[0]); $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever, o=lvs[0].find(x=>x.v===v); b.setAttribute('aria-pressed',String(v===cur)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvs[1].has(v)||v===cur?lvs[2](o.s):'';});}
    $$('.seg-ctl[data-key="R"]',rail).forEach(g=>A.ui.setPressed(g,String(S.R)));
    $$('.arl-cval',rail).forEach(e=>{const v=fPs(S.c); if(e.textContent!==v) e.textContent=v;});
    if(box){box.hidden=!show; if(show){setObs(box,`<div class="eyebrow">${L(s.label||{en:'Result',zh:'结果'})}</div>`+s.obs()); if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy()||null);
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-pred]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pred===pred)));
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#arl-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:''); ins.hidden=!html; setObs(ins,html);}
  } else {
    const e=chEval(chO), s=$('#arl-ch-s',rail), c=$('#arl-ch-c',rail); if(!s) return;
    s.textContent=fP(e.succ); c.textContent=fC(e.cost);
    const g1=$('#arl-ch-g1',rail), g2=$('#arl-ch-g2',rail); g1.classList.toggle('met',e.okS); g1.classList.toggle('arl-over',!e.okS); g2.classList.toggle('met',e.okC); g2.classList.toggle('arl-over',!e.okC);
    [['chM',chO.strong],['chC',chO.ck],['chR',chO.R],['chN',chO.n]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,rail).forEach(g=>A.ui.setPressed(g,String(v))));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#arl-ch-res',rail); setObs(res,chResultHTML());
  }
}

/* =====================================================================
   stage: the grid of tasks (canvas), the player, the curve, the cards
   ===================================================================== */
const cssv=(el,v)=>getComputedStyle(el).getPropertyValue(v).trim();
/* cells narrower than this (px) are drawn without the per-attempt segments (R5.10): 50 steps on a phone, and on a desktop */
const TIGHT=12;
const tightNow=()=>{const h=app&&$('#arl-grid',app), W=h?h.clientWidth:0; return W>0&&(W-20)/S.n<TIGHT;};
const totalCols=()=>S.n;
/* columns of task t on show: in a partial replay an unchanged row is whole, a changed one is whole up to its first change (R3.20) */
const colsOf=t=>PL.fix?Math.max(PL.fix[t]||0,PL.pv):PL.pv;
function tallyNow(){const runs=runsOf(S); return tallyOf(runs.filter((r,t)=>colsOf(t)>=endOf(r)+1));}
function drawGrid(){
  const host=$('#arl-grid',app), cv=$('#arl-cv',app); if(!host||!cv) return;
  const W=host.clientWidth; if(!W) return;
  const n=S.n, rowH=7, gap=2, H=RUNS*(rowH+gap)-gap, mark=16, dpr=window.devicePixelRatio||1;
  if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); cv.style.width=W+'px'; cv.style.height=H+'px';}
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const C={ink:cssv(app,'--ink'),ink2:cssv(app,'--ink-2'),ink3:cssv(app,'--ink-3'),line:cssv(app,'--line'),line2:cssv(app,'--line-2'),s2:cssv(app,'--surface-2'),surf:cssv(app,'--surface'),ok:cssv(app,'--c-ok'),bad:cssv(app,'--c-oom')};
  const cw=(W-mark-4)/n, cg=cw>=6?1.5:(cw>=3?1:0.5), w=cw-cg, tight=cw<TIGHT, simp=simple();
  const pre=hideRes()&&!pred, runs=runsOf(S);
  /* R5.11: the step where a task goes wrong (parts 1 and 2: where it failed; later: a silent error, or a stop) is outlined in red, and
     it appears only when the replay reaches that column; every other cell stays gray. The shapes still tell the kinds apart (R5.6). */
  /* a hollow cell with a thick red outline: a stopped step, or (parts 1 and 2) the step where the task failed */
  const box=(x,y)=>{g.fillStyle=C.surf; g.fillRect(x,y,w,rowH); g.strokeStyle=C.bad; g.lineWidth=1.6; g.strokeRect(x+0.8,y+0.8,Math.max(0,w-1.6),rowH-1.6);};
  for(let i=0;i<RUNS;i++){
    const y=i*(rowH+gap), run=runs[i], failed=run.out!=='ok', cols=colsOf(i);
    for(let j=0;j<n;j++){const x=j*cw;
      if(pre||j>=cols){g.fillStyle=C.s2; g.fillRect(x,y,w,rowH); continue;}
      const c=run.cells[j];
      if(simp&&failed&&j>run.fx) continue;                 // parts 1 and 2: the task has already failed; nothing more is drawn
      if(simp&&failed&&j===run.fx){box(x,y); continue;}
      if(c.st==='none') continue;
      if(c.st==='ok'){g.fillStyle=C.ink2; g.fillRect(x,y,w,rowH); continue;}
      if(c.st==='pale'){g.fillStyle=C.line2; g.fillRect(x,y,w,rowH); continue;}
      if(c.st==='silent'){g.save(); g.beginPath(); g.rect(x,y,w,rowH); g.clip(); g.fillStyle=C.surf; g.fillRect(x,y,w,rowH); g.strokeStyle=C.ink; g.lineWidth=1.2;
        for(let k=-rowH;k<w+rowH;k+=3.2){g.beginPath(); g.moveTo(x+k,y+rowH); g.lineTo(x+k+rowH,y); g.stroke();} g.restore();
        g.strokeStyle=C.bad; g.lineWidth=1.2; g.strokeRect(x+0.6,y+0.6,Math.max(0,w-1.2),rowH-1.2); continue;}
      /* retry and stop: one hollow segment per failed attempt; a retry ends in a solid segment. Narrow cells (R5.10): a retried step is
         one mid-gray cell, a stopped step one hollow cell with a thick outline, so neither looks like a done step or a speck */
      const segs=c.f+(c.st==='retry'?1:0), sw=(w-(segs-1)*1)/segs;
      if(tight||sw<2){if(c.st==='retry'){g.fillStyle=C.ink3; g.fillRect(x,y,w,rowH);} else box(x,y); continue;}
      for(let k=0;k<segs;k++){const sx=x+k*(sw+1), solid=c.st==='retry'&&k===segs-1;
        if(solid){g.fillStyle=C.ink2; g.fillRect(sx,y,sw,rowH);} else {g.fillStyle=C.surf; g.fillRect(sx,y,sw,rowH); g.strokeStyle=c.st==='stop'?C.bad:C.ink3; g.lineWidth=c.st==='stop'?1.6:1; g.strokeRect(sx+0.5,y+0.5,Math.max(0,sw-1),rowH-1);}}
    }
    if(!pre&&cols>=endOf(run)+1){g.font='bold 10px system-ui,sans-serif'; g.textBaseline='middle'; g.fillStyle=run.out==='ok'?C.ok:C.bad; g.fillText(run.out==='ok'?'✓':'✗',W-mark+3,y+rowH/2+0.5);}
  }
  host.dataset.tipx=tx(`The chart runs ${RUNS} tasks as an illustration. Each row is one task, and each cell is one step. A row ends with ✓ when every step was right, and with ✗ otherwise.`,`图里跑了 ${RUNS} 个任务，作为示意。每一行是一个任务，每一格是一步。每一步都做对的行，末尾是 ✓，否则是 ✗。`);
  host.setAttribute('aria-label',host.dataset.tipx);
}
function drawPlayer(){
  const pb=$('#arl-play',app), sc=$('#arl-scrub',app), tl=$('#arl-tally',app), bt=$('#arl-batch',app); if(!pb) return;
  const end=PL.pv>=totalCols(), pre=hideRes()&&!pred;
  pb.textContent=PL.on?tx('❚❚ Pause','❚❚ 暂停'):(end?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放'));
  pb.disabled=pre; pb.dataset.tipx=PL.on?tx('Pause the tasks.','暂停。'):(end?tx(`Play the run again from the start: the ${RUNS} tasks start over from step 1.`,`从头再放一遍：${RUNS} 个任务从第 1 步重新跑。`):tx(`Play the ${RUNS} tasks.`,`播放这 ${RUNS} 个任务。`));
  sc.max=String(totalCols()); if(+sc.value!==PL.pv) sc.value=String(PL.pv); sc.disabled=pre; A.ui.setRange(sc);
  $$('.arl-speed button',app).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===PL.speed)));
  bt.textContent=pre?tx('Answer the card’s question first.','先回答卡片里的问题。'):tx(`${RUNS} tasks, as an illustration`,`${RUNS} 个任务，作为示意`);
  const t=tallyNow();
  const res=tx(`✓ ${t.ok} succeeded, ✗ ${t.n-t.ok} failed.`,`✓ 成功 ${t.ok} 个，✗ 失败 ${t.n-t.ok} 个。`);
  tl.textContent=pre?'':(simple()?(t.n>=RUNS?tx(`All ${RUNS} tasks have a result: `,`${RUNS} 个任务都出了结果：`):tx(`${t.n} of the ${RUNS} tasks have a result: `,`${RUNS} 个任务里，已经有 ${t.n} 个出了结果：`))+res
    :tx(`Done: ${t.n} of ${RUNS} tasks. ✓ ${t.ok} right, ✗ ${t.wrong} wrong at the end, ✗ ${t.stop} stopped.`,`跑完 ${t.n} / ${RUNS} 个任务：✓ 做对 ${t.ok} 个，✗ 做完但错了 ${t.wrong} 个，✗ 中途停下 ${t.stop} 个。`));
}
function curveSVG(W){
  const r=R_(), H=132, Lp=34, Rp=Math.min(118,Math.max(86,W*0.26)), T=10, B=20, pw=Math.max(60,W-Lp-Rp), ph=H-T-B, N=50;
  const X=n=>Lp+(n-1)/(N-1)*pw, Y=v=>T+(1-v)*ph, f1=v=>v.toFixed(1);
  const mine=[], plain=[]; for(let n=1;n<=N;n++){mine.push(compute({...S,n}).succ); plain.push(Math.pow(S.p,n));}
  const same=mine.every((v,i)=>Math.abs(v-plain[i])<1e-9);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true">`;
  [0,0.5,1].forEach(v=>{s+=`<line class="arl-gl" x1="${Lp}" x2="${f1(Lp+pw)}" y1="${f1(Y(v))}" y2="${f1(Y(v))}"/><text class="arl-tk" x="${Lp-4}" y="${f1(Y(v)+3.5)}" text-anchor="end">${Math.round(v*100)}%</text>`;});
  [1,10,20,30,40,50].forEach(n=>{s+=`<text class="arl-tk" x="${f1(X(n))}" y="${H-5}" text-anchor="middle">${n}</text>`;});
  const pts=a=>a.map((v,i)=>f1(X(i+1))+','+f1(Y(v))).join(' ');
  if(!same) s+=`<polyline class="arl-plain" points="${pts(plain)}"/>`;
  s+=`<polyline class="arl-mine" points="${pts(mine)}"/>`;
  /* the names at the line ends (R5.9), spread apart when they meet */
  /* names, not formulas, in the picture (H1); when the two lines meet, “= no retries” goes on a second line so that it stays inside at 390 (H7) */
  const lab=[{y:Y(mine[N-1]),t:tx('your settings','你的设置'),t2:same?tx('= no retries','= 不重试'):'',c:'arl-ln'}];
  if(!same) lab.push({y:Y(plain[N-1]),t:tx('no retries','不重试'),c:'arl-ln arl-ln2'});
  lab.sort((a,b)=>a.y-b.y); for(let i=1;i<lab.length;i++) if(lab[i].y-lab[i-1].y<12) lab[i].y=lab[i-1].y+12;
  lab.forEach(o=>{const y=Math.min(H-B-(o.t2?11:0),o.y)+3.5; s+=`<text class="${o.c}" x="${f1(Lp+pw+5)}" y="${f1(y)}">${esc(o.t)}</text>`+(o.t2?`<text class="${o.c}" x="${f1(Lp+pw+5)}" y="${f1(y+11)}">${esc(o.t2)}</text>`:'');});
  /* the marker at the current number of steps */
  const x=X(S.n), vm=r.succ, vp=r.plain, right=x>Lp+pw*0.62;
  s+=`<line class="arl-mk" x1="${f1(x)}" x2="${f1(x)}" y1="${T-2}" y2="${f1(T+ph)}"/>`;
  if(!same) s+=`<circle class="arl-dot2" cx="${f1(x)}" cy="${f1(Y(vp))}" r="3"/>`;
  s+=`<circle class="arl-dot" cx="${f1(x)}" cy="${f1(Y(vm))}" r="3.6"/>`;
  /* the values at the marker (R7.8): yours above its dot, pⁿ under its dot (above it when there is no room below), never closer than
     one line; each is drawn on a halo of the background, so a line passing under it does not strike it through */
  const top=T+8, bot=T+ph, LH=12;
  let ym=Y(vm)-6, yp=Y(vp)+13;
  if(!same&&yp>bot) yp=Y(vp)-6;
  if(!same&&ym>yp-LH) ym=yp-LH;
  ym=Math.max(top,ym); if(!same&&yp<ym+LH) yp=ym+LH;
  const vals=[{y:ym,t:fP(vm),c:'arl-vl'}].concat(same?[]:[{y:yp,t:fP(vp),c:'arl-vl arl-vl2'}]);
  vals.forEach(o=>{s+=`<text class="${o.c}" x="${f1(right?x-6:x+6)}" y="${f1(o.y)}" text-anchor="${right?'end':'start'}">${o.t}</text>`;});
  s+=`</svg>`;
  return s;
}
function drawCurve(){
  const w=$('#arl-svgw',app), cap=$('#arl-curve-cap',app); if(!w) return;
  if(hideRes()){cap.textContent=tx(`Task success against the number of steps: shown when the ${RUNS} tasks are done.`,`任务成功率随步数的变化，等 ${RUNS} 个任务跑完后显示。`); w.innerHTML=''; w.removeAttribute('data-tipx'); w.classList.add('arl-wait'); return;}
  w.classList.remove('arl-wait');
  cap.textContent=tx(`How does task success fall as the steps go from 1 to 50? (step ${S.n} marked)`,`步数从 1 加到 50，任务成功率掉多快？（标出的是 ${S.n} 步）`);
  const W=Math.max(240,w.clientWidth||300); w.innerHTML=curveSVG(W);
  const r=R_();
  w.dataset.tipx=tx(`The solid line is the task’s success with your settings, for every number of steps from 1 to 50; the dashed line is no retries, ${M('p^n')}: every step right on its first try. At ${stepsW(S.n)}: ${fP(r.succ)} with your settings, ${fP(r.plain)} with no retries.`,`实线是按你的设置、步数从 1 到 50 时任务的成功率。虚线是不重试时的 ${M('p^n')}，也就是每一步都要第一次就做对。${stepsW(S.n)}时：你的设置是 ${fP(r.succ)}，不重试是 ${fP(r.plain)}。`);
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  const row=`<span data-tipx="${esc(tx(`The grid holds ${RUNS} tasks: one row per task, one cell per step.`,`网格里有 ${RUNS} 个任务，每行是一个任务，每格是一步。`))}" tabindex="0"><span>${esc(tx('row = task, cell = step','每行一个任务，每格一步'))}</span></span>`;
  const ok=it('arl-sw-ok',tx('right','做对'),tx('A solid cell: the step was right on its first try.','实心格：这一步第一次就做对了。'));
  const ends=tip=>`<span data-tipx="${esc(tip)}" tabindex="0"><b class="arl-okm">✓</b><b class="arl-badm">✗</b><span>${esc(tx('task right / failed','任务成功 / 失败'))}</span></span>`;
  /* parts 1 and 2: one failure state (R6.30, R6.31) */
  if(simple()) return row+ok+
    it('arl-sw-st',tx('an error: the task has failed','出错（任务已经失败）'),tx('A hollow cell with a thick red outline: the step went wrong. One wrong step is enough to fail the task, so the rest of the row stays empty.','红色粗边框的空心格：这一步做错了。只要错一步，任务就已经失败，所以这一行后面空着。'))+
    ends(tx('At the end of each row: ✓ the task succeeded, ✗ it failed.','每一行的末尾：✓ 任务成功，✗ 任务失败。'));
  const tight=tightNow();
  return row+ok+
    (S.R>0?it('arl-sw-rt'+(tight?' arl-tight':''),tx('loud error, then a retry','报错后重试'),tight
      ?tx('A mid-gray cell: the step failed loudly at first; the harness, the program that runs the agent, had it redone, and the retry was right.','中灰格：这一步先报了错，运行 agent 的程序（harness）让它重做，重做后做对了。')
      :tx('Hollow segments, then a solid one. Each hollow segment is an attempt that failed loudly; the harness, the program that runs the agent, saw it and had the step redone. The solid segment at the end is the retry that was right.','空心段加实心段。每个空心段是一次报了错的尝试，运行 agent 的程序（harness）看见了，就让这一步重做。最后的实心段是做对的那次重试。')):'')+
    it('arl-sw-si',tx('silent error','悄悄出错'),tx('A hatched cell with a red outline: the step went wrong, but nothing reported it, so nobody noticed.','红框的斜纹格：这一步做错了，却没有报错，谁也没发现。'))+
    it('arl-sw-pa',tx('still running, on a wrong result','还在跑，但结果已经错了'),tx('After a silent error the rest of the row fades: the task goes on, but every later step works from the wrong result.','悄悄出错之后，这一行后面的格子变淡：任务还在往下做，可每一步用的都是错的结果。'))+
    (S.R>0?it('arl-sw-st',tx('retries used up: stopped','重试用完：停下'),tx('A hollow segment with a thick red outline means that every attempt at this step failed loudly and the retries ran out, so the task stopped here.','红色粗边框的空心段表示这一步每次尝试都报了错，重试次数也用完了，任务就停在这里。'))
      :it('arl-sw-st',tx('loud error: stopped','报错后停下'),tx('A hollow cell with a thick red outline means that this step failed loudly. With no retries, the task stopped here.','红色粗边框的空心格表示这一步报了错。不重试，任务就停在这里。')))+
    ends(tx('At the end of each row: ✓ the task succeeded, ✗ it failed (wrong at the end, or stopped).','每一行的末尾：✓ 任务成功，✗ 任务失败（做完但错了，或者中途停下）。'));
}
function statusOf(){
  if(mode==='challenge'){const e=chEval(chO); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Both targets met','两个目标都达到'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when at least 90% of the tasks succeed at a cost of at most 40 calls per task. Now: ${fP(e.succ)} succeed, at a cost of ${fC(e.cost)} calls.`,`任务成功率至少 90%、每个任务的调用成本不超过 40 次时，徽章是绿色。现在成功率是 ${fP(e.succ)}，调用成本是 ${fC(e.cost)} 次。`)};}
  if(hideRes()) return {cls:'neutral',txt:pred?tx('Running','在跑'):tx('Predict first','先预测'),short:pred?tx('Running','在跑'):tx('Predict first','先预测'),
    tip:pred?tx(`The ${RUNS} tasks are running. The result shows when they are done.`,`${RUNS} 个任务正在跑，跑完后显示结果。`):tx('Shown after you answer the question in part 1.','回答第 1 部分的问题之后才显示。')};
  const r=R_(); return {cls:r.succ>=0.9?'ok':'neutral',txt:tx(`${fP(r.succ)} succeed`,`成功 ${fP(r.succ)}`),short:tx(`${fP(r.succ)} succeed`,`成功 ${fP(r.succ)}`),
    tip:tx(`By the formula, ${fP(r.succ)} of the tasks succeed with these settings. Green from 90%.`,`按公式算，这组设置下 ${fP(r.succ)} 的任务能成功。90% 以上时是绿色。`)};
}
function drawHead(){
  const t=$('#arl-st-title',app), sub=$('#arl-st-sub',app), st=$('#arl-status',app);
  t.textContent=mode==='challenge'?tx('How many of the report agent’s tasks succeed?','报表 agent 的任务能成功几个？'):tx('How many of these tasks succeed?','这些任务能成功几个？');
  sub.textContent=simple()?tx(`${stepsW(S.n)}, ${fPc(S.p)} per step, no retries, no check`,`${stepsW(S.n)}，每步 ${fPc(S.p)}，不重试，不检查`)
    :tx(`${stepsW(S.n)}, ${fPc(S.p)} per step, ${fPs(S.r)} of failures loud, ${retriesW(S.R)}, ${S.ck?'a check after each step':'no check'}${S.R&&S.kind==='b'?', same-call retries':''}`,
        `${stepsW(S.n)}，每步 ${fPc(S.p)}，报出来的错占 ${fPs(S.r)}，${retriesW(S.R)}，${S.ck?'每步检查':'不检查'}${S.R&&S.kind==='b'?'，原样重试':''}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
function metricsDef(){
  const r=R_(), hid=hideRes(), dash='—', w=tx('shown when the tasks are done','任务跑完后显示'), ch=mode==='challenge';
  const price=ch?chState(chO).price:1, calls=r.calls*price;
  const succ={k:'succ',label:tx('Task success','任务成功率'),v:hid?dash:fP(r.succ),sub:hid?w:tx('every step right','每一步都做对'),pts:true,tip:hid?tx('Shown when the tasks are done: the share of tasks with every step right, by the formula.','任务跑完后显示：按公式算，每一步都做对的任务占多少。'):tx(`By the formula, ${fP(r.succ)} of the tasks end with every step right.`,`按公式算，${fP(r.succ)} 的任务每一步都做对了。`)};
  /* parts 1 and 2: success and failure only; the kinds of failure come in part 3 (S1) */
  if(simple()) return [succ,{k:'fail',label:tx('Tasks that fail','失败的任务'),v:hid?dash:fP(1-r.succ),sub:hid?w:tx('at least one step wrong','至少有一步做错'),pts:true,
    tip:hid?tx('Shown when the tasks are done: the share of tasks with at least one step wrong, by the formula.','任务跑完后显示：按公式算，至少有一步做错的任务占多少。'):tx(`By the formula, ${fP(1-r.succ)} of the tasks get at least one step wrong, and one wrong step fails the task.`,`按公式算，${fP(1-r.succ)} 的任务至少有一步做错。错一步，任务就失败了。`)}];
  const callTip=price>1?tx(`On average a task calls the model ${fC(r.calls)} times. Each call to the 98%-per-step model counts 2, so the cost is ${fC(calls)} calls.`,`平均每个任务调用模型 ${fC(r.calls)} 次。每步成功率 98% 的模型调用一次记 2 次，所以调用成本是 ${fC(calls)} 次。`)
    :tx(`On average a task calls the model ${fC(calls)} times. Every attempt is one call, every check one more, and a task that stopped makes no more calls.`,`平均每个任务调用模型 ${fC(calls)} 次。每次尝试是一次调用，每次检查再加一次，任务停下后就不再调用。`);
  return [succ,
    {k:'wrong',label:tx('Finished, but wrong','做完但错了'),v:hid?dash:fP(r.wrong),sub:hid?w:tx('a silent error inside','中间有悄悄出的错'),pts:true,tip:tx('These tasks ran to the end and reported done, but one step inside went wrong silently. Nobody knows the result is wrong.','这些任务一直做到最后，也报告了完成，可中间有一步悄悄做错了。谁也不知道结果是错的。')},
    {k:'stop',label:tx('Stopped on an error','报错停下'),v:hid?dash:fP(r.stopped),sub:hid?w:(S.R>0?tx('retries used up','重试用完了'):tx('no retries: a loud error stops it','不重试，一报错就停下')),pts:true,
     tip:S.R>0?tx('These tasks kept failing loudly at one step until the retries ran out, so they stopped. At least the harness, the program that runs the agent, knows they failed.','这些任务在某一步一直报错，重试次数用完，就停下了。好在运行 agent 的程序（harness）知道它们失败了。')
       :tx('These tasks failed loudly at one step and, with no retries, stopped there. At least the harness, the program that runs the agent, knows they failed.','这些任务在某一步报了错，又不重试，就停下了。好在运行 agent 的程序（harness）知道它们失败了。')},
    {k:'calls',label:ch?tx('Cost per task','每个任务的调用成本'):tx('Model calls per task','每个任务的模型调用'),v:hid?dash:fC(calls),sub:hid?w:tx('on average, with retries and checks','平均，含重试和检查'),
     tip:hid?tx('Shown when the tasks are done: how many model calls a task makes on average.','任务跑完后显示：平均每个任务调用模型多少次。'):callTip}];
}
function drawMetrics(){
  const host=$('#arl-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&prev!=='—'&&m.v!=='—'){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:1}:{dp:1})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else if(!userAct){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4) */
function mathHTML(){
  if(hideRes()) return `<p>${tx(`Shown when the ${RUNS} tasks are done.`,`${RUNS} 个任务跑完后显示。`)}</p>`;
  const r=R_(), t=T_, inf=S.kind!=='b'||S.R===0, f=fN, simp=simple(), o6=fixd(r.ok,6);
  const x=r.q*r.rc, x2=r.q2*r.rc, keep=fixd(1-r.stop,6), blocks=[];
  blocks.push(MB(tx(`<b>One step.</b> p is the chance that ${t('p','a step is right on its first try')}, ${f(S.p)} here; q is the chance that it fails.`,`<b>一步。</b>p 是${t('p','一步第一次就做对')}的概率，这里是 ${f(S.p)}。q 是它做错的概率。`),
    EQ([['q','1 - p'],['',`1 - ${f(S.p)}`],['',f(r.q)]].concat(simp?[]:S.ck?[[tx('\\text{loud share}','\\text{报出来的比例}')+'\\,\\,r_c','r + (1 - r) \\times c'],['',`${f(S.r)} + ${f(1-S.r)} \\times ${f(S.c)}`],['',f(r.rc)]]:[[tx('\\text{loud share}','\\text{报出来的比例}')+'\\,\\,r_c','r'],['',f(S.r)]]))+
    (simp?'':`<p>${S.ck?tx(`r is the ${t('r','share of failures that are loud')}; the check after each step catches a share c of the silent ones.`,`r 是${t('r','报出来的错')}占的比例。每步之后的检查，能发现悄悄出的错里的 c。`):tx(`r is the ${t('r','share of failures that are loud')}.`,`r 是${t('r','报出来的错')}占的比例。`)}</p>`)));
  const stepName=tx('\\text{step right}','\\text{一步做对}')+'\\,\\,P_{\\text{step}}';
  if(S.R===0) blocks.push(MB(tx('<b>A step ends right</b> only on its first try, since there are no retries.','<b>一步做对</b>只能靠第一次，因为不重试。'),EQ([[stepName,'p'],['',o6]])));
  else if(inf) blocks.push(MB(tx(`<b>A step ends right</b> if one attempt succeeds before the retries run out, and every failed attempt before it was loud. ${t('R',`Up to ${S.R} retries`)}, each as good as a first try.`,`<b>一步做对</b>，要在重试用完之前有一次做对，而且前面每次做错都报了出来。${t('R',`最多重试 ${S.R} 次`)}，每次和第一次一样有把握。`),
    EQ([[stepName,'p \\cdot \\frac{1 - {(q r_c)}^{R+1}}{1 - q r_c}'],['',`${f(S.p)} \\cdot \\frac{1 - ${f(x)}^{${S.R+1}}}{1 - ${f(x)}}`],['',o6]])));
  else blocks.push(MB(tx(`<b>A step ends right</b> on its first try, or on a retry after a loud failure. ${t('kind',`Up to ${S.R} same-call retries`)}, each right with chance ${M('p_2')} = ${f(r.p2)}; ${M('q_2 = 1 - p_2')}.`,`<b>一步做对</b>，要么第一次就对，要么报错后重试做对。${t('kind',`最多原样重试 ${S.R} 次`)}，每次做对的概率是 ${M('p_2')} = ${f(r.p2)}，${M('q_2 = 1 - p_2')}。`),
    EQ([[stepName,'p + q r_c p_2 \\frac{1 - {(q_2 r_c)}^{R}}{1 - q_2 r_c}'],['',`${f(S.p)} + ${f(r.q)} \\times ${f(r.rc)} \\times ${f(r.p2)} \\times \\frac{1 - ${f(x2)}^{${S.R}}}{1 - ${f(x2)}}`],['',o6]])));
  blocks.push(MB(tx(`<b>The task succeeds</b> only if ${t('n',`every one of the ${S.n} steps`)} ends right.`,`<b>任务成功</b>，要 ${t('n',`${S.n} 步`)}全部做对。`),
    EQ([[tx('\\text{task success}','\\text{任务成功率}')+'\\,\\,P_{\\text{task}}','{P_{\\text{step}}}^{n}'],['',`${o6}^{${S.n}}`],['',fixd(r.succ,6)]]),tx(`That is <b>${fP(r.succ)}</b>.`,`也就是 <b>${fP(r.succ)}</b>。`)));
  if(simp) return blocks.join('');
  /* the stop and the wrong ends (S10, R6.29): every formula in full, with the numbers in */
  const stopName=tx('\\text{stop at a step}','\\text{在一步停下}')+'\\,\\,P_{\\text{stop}}';
  const stopRows=S.R===0?[[stopName,'q r_c'],['',`${f(r.q)} \\times ${f(r.rc)}`],['',fixd(r.stop,6)]]
    :inf?[[stopName,'{(q r_c)}^{R+1}'],['',`${f(x)}^{${S.R+1}}`],['',fixd(r.stop,6)]]
    :[[stopName,'q r_c \\cdot {(q_2 r_c)}^{R}'],['',`${f(x)} \\times ${f(x2)}^{${S.R}}`],['',fixd(r.stop,6)]];
  blocks.push(MB(S.R===0?tx('<b>The task stops</b> when a step fails loudly, since there are no retries.','<b>任务停下</b>：一步报了错，又不重试，任务就停下。')
      :tx(`<b>The task stops</b> when ${S.R===1?'both attempts':`all ${S.R+1} attempts`} at a step fail loudly and the retries run out.`,`<b>任务停下</b>：一步的 ${S.R+1} 次尝试都报了错，重试用完，任务就停下。`),
    EQ(stopRows.concat([[tx('\\text{tasks that stop}','\\text{停下的任务}'),'1 - {(1 - P_{\\text{stop}})}^{n}'],['',`1 - ${keep}^{${S.n}}`],['',fixd(r.stopped,4)],
      [tx('\\text{finish, but wrong}','\\text{做完但错了}'),'{(1 - P_{\\text{stop}})}^{n} - P_{\\text{task}}'],['',`${keep}^{${S.n}} - ${fixd(r.succ,6)}`],['',fixd(r.wrong,4)]])),
    tx(`<b>${fP(r.stopped)}</b> of the tasks stop, and <b>${fP(r.wrong)}</b> finish but wrong.`,`<b>${fP(r.stopped)}</b> 的任务停下，<b>${fP(r.wrong)}</b> 做完但错了。`)));
  blocks.push(MB(tx(`<b>${t('ck','Model calls')}</b> per task: the calls per attempt (2 with the check), times the attempts per step, times the steps reached on average.`,`<b>每个任务的${t('ck','模型调用')}</b>：每次尝试的调用次数（加上检查是 2 次），乘以每步平均的尝试次数，再乘以平均走到的步数。`),
    EQ([[tx('\\text{calls}','\\text{调用次数}'),`${S.ck?2:1} \\times ${fixd(r.att,4)} \\times ${fixd(r.reach,3)}`],['',fixd(r.calls,2)]]),tx(`About <b>${fC(r.calls)}</b> calls.`,`约 <b>${fC(r.calls)}</b> 次。`)));
  return blocks.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawGrid(); drawPlayer(); drawCurve(); drawMetrics(); drawLegend();
  const mb=$('#arl-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function drawLegend(){const lg=app&&$('#arl-legend',app); if(!lg) return; const lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}}
/* light redraw during play: the grid, the player and the part-1 card */
function tick(){if(!app) return; drawGrid(); drawPlayer(); if(isP('mission')){updateRail(); drawHead(); drawCurve(); drawMetrics(); const mb=$('#arl-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}} A.dock.refresh();}
/* Settings (R4.2): each guided part shows only its own controls; the rest are hidden with a note */
const PART_CTL={chain:['n','p'],loud:['R','r'],check:['ck','c'],retry:['kind','pb']};
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the four choices in the task card. Everything else is fixed. Half of the failures are loud, and the check finds 80% of the silent ones.','挑战中只能改任务卡里的四个选择。其余的设置固定不变。一半的错会报出来。悄悄出的错里，检查能发现 80%。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return tx('Fixed in this part: 20 steps, 95% per step, no retries, no check. You can change everything in Scenarios.','这一部分固定：20 步，每步 95%，不重试，不检查。在应用场景里每一项都能改。');
  if(id==='chain') return tx('In this part you can change the steps and the per-step chance. Fixed: no retries, no check.','这一部分可以改步数和每一步的把握。固定：不重试，不检查。');
  if(id==='loud') return tx('In this part you can change the retries and the share of loud failures. Fixed: 20 steps, 95% per step, no check.','这一部分可以改重试次数和报出来的错的比例。固定：20 步，每步 95%，不检查。');
  if(id==='check') return tx('In this part you can change the check, and how many of the silent failures it finds. Fixed: 20 steps, 95% per step, 70% loud.','这一部分可以改检查，以及悄悄出的错里检查能发现多少。其余的设置固定不变：20 步，每步 95%，70% 的错会报出来。');
  if(id==='retry') return tx('In this part you can change how a retry is done. Fixed: 20 steps, 95% per step, 70% loud, no check.','这一部分可以改重试的方式。固定：20 步，每步 95%，70% 的错会报出来，不检查。');
  return tx('Fixed in this part: each “Show in the chart” button in the card sets its own example. You can change everything in Scenarios.','这一部分固定：卡片里每个“在图里看”按钮会设好自己的例子。在应用场景里每一项都能改。');
}
function syncControls(){
  const set=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(i.closest('#arl-player')) return; if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl'); const oo=o&&o.querySelector('output'); if(oo) oo.textContent=out; i.setAttribute('aria-valuetext',out);});
  set('n',S.n,String(S.n)); set('p',Math.round(S.p*1000),fPc(S.p)); set('r',Math.round(S.r*10),fPs(S.r)); set('c',Math.round(S.c*10),fPs(S.c)); set('pb',Math.round(S.pb*20),fPs(S.pb));
  [['R',S.R],['ck',S.ck],['kind',S.kind]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,String(v))));
  const so=$('#arl-seed-out',app); if(so) so.textContent=tx(`seed ${S.seed}`,`种子 ${S.seed}`);
  const g=mode==='guided', ch=mode==='challenge', id=g?STEPS[step].id:'', live=g?(PART_CTL[id]||[]):(ch?[]:['n','p','r','R','ck','kind','c','pb','seed','reset']);
  ['n','p','r','R','ck','kind','c','pb','seed','reset'].forEach(k=>{const e=$('#arl-w-'+k,app); if(e) e.hidden=!live.includes(k)&&!(k==='reset'&&g&&live.length);});
  const ga=$('#arl-w-as',app); if(ga) ga.hidden=!live.includes('c')&&!live.includes('pb');
  const ln=$('#arl-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||hideRes()) return; const r=R_(); A.announce(simple()?tx(`${stepsW(S.n)} at ${fPc(S.p)}: ${fP(r.succ)} of the tasks succeed.`,`${stepsW(S.n)}、每步 ${fPc(S.p)}：任务成功率 ${fP(r.succ)}。`)
    :tx(`${stepsW(S.n)} at ${fPc(S.p)}: ${fP(r.succ)} of the tasks succeed, ${fC(r.calls)} model calls per task.`,`${stepsW(S.n)}、每步 ${fPc(S.p)}：任务成功率 ${fP(r.succ)}，每个任务 ${fC(r.calls)} 次模型调用。`));},400);}

/* =====================================================================
   the player (R3.9: auto-play, pause, one step, to the end, three speeds, a scrubber)
   ===================================================================== */
function stopLoop(){if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=0;}
function frame(ts){
  PL.raf=0; if(!app||!PL.on) return;
  const dt=Math.min(0.25,(ts-(PL.last||ts))/1000); PL.last=ts; PL.acc+=dt;
  const col=SPD[PL.speed]/S.n; let moved=false;
  for(;;){if(PL.acc<col) break; PL.acc-=col; PL.pv++; moved=true; if(PL.pv>=totalCols()){PL.pv=totalCols(); PL.on=false; break;}}
  if(PL.pv>=totalCols()) finished();
  if(moved||!PL.on) tick();
  if(PL.on) PL.raf=requestAnimationFrame(frame);
}
function finished(){if(!(hideRes()&&!pred)) SEEN.add(picKey(picNow())); if(isP('mission')&&pred&&!runsSeen){runsSeen=true; wantObs=true; renderRail(); revealSoon(); wantObs=false;}}
/* the play button: from the end it replays the whole run (every row) from step 1 */
function play(){if(hideRes()&&!pred) return; if(PL.pv>=totalCols()){PL.pv=0; PL.fix=null;} if(A.reduceMotion()){PL.pv=totalCols(); PL.on=false; finished(); tick(); return;} PL.on=true; PL.last=0; PL.acc=0; stopLoop(); PL.raf=requestAnimationFrame(frame); drawPlayer();}
function pause(){PL.on=false; stopLoop(); drawPlayer();}
/* what the grid draws for one task, cell by cell (parts 1 and 2 end a failed row at its first failure) */
const drawnRow=(r,simp)=>simp&&r.out!=='ok'?r.cells.map((c,j)=>j<r.fx?'ok0':(j===r.fx?'X':'')):r.cells.map(c=>c.st==='none'?'':c.st+c.f);
const picNow=()=>{const simp=simple(); return {n:S.n,rows:runsOf(S).map(r=>drawnRow(r,simp))};};
/* the whole run from step 1, every row (the first run after the prediction) */
function restart(){PL.fix=null; PL.pv=0; PL.acc=0; if(hideRes()&&!pred){pause(); PL.sig=null; tick(); return;} PL.sig=picNow(); play(); tick();}
/* a new setup (a part, a mode, an option, a setting, a scenario): R3.20. The same picture is not replayed: a finished run stays
   finished, a run in progress goes on. A changed picture replays only its changed rows, each from its first changed cell; the other
   rows show finished at once. */
function refresh(){
  if(hideRes()&&!pred){pause(); PL.fix=null; PL.pv=0; PL.sig=null; tick(); return;}
  const now=picNow(), old=PL.sig; PL.sig=now;
  const first=(a,b)=>{let j=0; while(j<a.length&&a[j]===b[j]) j++; return j;};
  const fix=old&&old.n===now.n?now.rows.map((r,i)=>first(r,old.rows[i])):null;
  if(fix&&fix.every(j=>j>=now.n)){if(PL.pv>=totalCols()) finished(); tick(); return;}
  if(SEEN.has(picKey(now))||SEENH.has(hkey(picKey(now)))){pause(); PL.fix=null; PL.pv=totalCols(); PL.acc=0; finished(); tick(); return;}   // watched before: its end, no replay
  PL.fix=fix; PL.pv=fix?Math.min(...fix.filter(j=>j<now.n)):0; PL.acc=0; play(); tick();
}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function clampS(){S.n=Math.max(1,Math.min(50,Math.round(S.n))); S.p=Math.max(0.8,Math.min(0.999,Math.round(S.p*1000)/1000)); S.r=Math.max(0,Math.min(1,Math.round(S.r*10)/10));
  S.R=[0,1,2,3].includes(S.R)?S.R:0; S.ck=S.ck?1:0; S.kind=S.kind==='b'?'b':'i'; S.c=Math.max(0,Math.min(1,Math.round(S.c*10)/10)); S.pb=Math.max(0,Math.min(1,Math.round(S.pb*20)/20));}
function setState(p){const before=JSON.stringify(S); Object.assign(S,p); clampS(); if(isP('chain')) {const m=match(LV2); if(m){lv2=m; tried2.add(m);}} render(); if(JSON.stringify(S)!==before) refresh();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(part,v){
  if(part===2){const o=LV2.find(x=>x.v===v); lv2=v; tried2.add(v); setState({...BASE,...o.s});}
  else if(part===4){const o=LV4.find(x=>x.v===v); lv4=v; tried4.add(v); setState({...P4,...o.s});}
  else {const o=LV5.find(x=>x.v===v); lv5=v; tried5.add(v); setState({...P5,...o.s});}
}
function setRem(k){if(!REMS[k]) return; remView=k; S={...REMS[k].s}; renderRail(); refresh(); if(innerWidth<=760) A.ui.reveal($('#arl-stage',app));}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...BASE,...s.enter()};
  const st=$('#arl-stage',app); if(st) st.scrollTop=0;
  renderRail(); refresh();
}
function revealStep(){A.ui.reveal($('#arl-rail .step.current',app)||$('#arl-rail',app));}
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
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#arl-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#arl-rail .step.current',app)||$('#arl-rail .railcard',app)||$('#arl-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#arl-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#arl-insight',app)); else revealBox($('#arl-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chO={...CH.start}; S=chState(chO);}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE,seed:S.seed};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#arl-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail(); refresh();
}
const tokOf=s=>[s.n,Math.round(s.p*1000),Math.round(s.r*10),s.R,s.ck,s.kind,Math.round(s.c*10),Math.round(s.pb*20),s.seed].join('_');
const token=()=>tokOf(S);
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==9) return null;
  const [n,pm,r10,R,ck]=p.slice(0,5).map(Number), kind=p[5], [c10,pb20,seed]=p.slice(6).map(Number);
  const I=Number.isInteger; if(!I(n)||n<1||n>50||!I(pm)||pm<800||pm>999||!I(r10)||r10<0||r10>10||![0,1,2,3].includes(R)||![0,1].includes(ck)||!['i','b'].includes(kind)||!I(c10)||c10<0||c10>10||!I(pb20)||pb20<0||pb20>20||!I(seed)||seed<1||seed>9999) return null;
  return {...BASE,n,p:pm/1000,r:r10/10,R,ck,kind,c:c10/10,pb:pb20/20,seed};
}

/* =====================================================================
   interview drill (seeded, from the engine)
   ===================================================================== */
function makeDrill(seed){const rnd=u=>u01(seed+500,0,0,u,0), pk=(a,u)=>a[Math.floor(rnd(u)*a.length)];
  const n=pk([8,10,12,15,25,30],0), p=pk([0.9,0.92,0.95,0.97,0.98],1), r=pk([0.5,0.6,0.8,0.9],2), R=pk([1,2],3);
  return {seed,n,p,r,R,res:compute({...BASE,n,p,r,R})};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#arl-drill-q',app), a=$('#arl-drill-a',app);
  qd.innerHTML=tx(`An agent runs <b>${d.n} steps</b>. Each step is right <b>${fPs(d.p)}</b> of the time; <b>${fPs(d.r)}</b> of its failures are loud and retried up to <b>${d.R===1?'once':`${d.R} times`}</b>; a silent failure goes on to the end. What share of tasks succeed?`,
                  `一个 agent 要走 <b>${d.n} 步</b>。每一步做对的把握是 <b>${fPs(d.p)}</b>；它的错有 <b>${fPs(d.r)}</b> 会报出来，最多重试 <b>${d.R} 次</b>；悄悄出的错会一路带到最后。任务成功率是多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const q=1-d.p, x=q*d.r, sh=v=>String(+v.toFixed(4)), fx=t=>`<div class="arl-fx">${t}</div>`, ok6=fixd(d.res.ok,6);
  const eq=rows=>EQ(rows), stepName=tx('\\text{step right}','\\text{一步做对}')+'\\,\\,P_{\\text{step}}';
  const fStep=eq([[stepName,'p \\cdot \\frac{1 - {(q r)}^{R+1}}{1 - q r}'],['',`${sh(d.p)} \\cdot \\frac{1 - ${sh(x)}^{${d.R+1}}}{1 - ${sh(x)}}`],['',ok6]]),
    fQr=eq([['q r',`${sh(q)} \\times ${sh(d.r)}`],['',sh(x)]]),
    fTask=eq([[tx('\\text{task success}','\\text{任务成功率}')+'\\,\\,P_{\\text{task}}','{P_{\\text{step}}}^{n}'],['',`${ok6}^{${d.n}}`],['',fixd(d.res.succ,4)]]),
    fPlain=eq([[tx('\\text{no retries}','\\text{不重试}'),'p^{n}'],['',`${sh(d.p)}^{${d.n}}`],['',fixd(d.res.plain,4)]]);
  a.innerHTML=tx(`<p><b>${fP(d.res.succ)}.</b></p><p>A step ends right if one attempt succeeds before the retries run out, and every failed attempt before it was loud:</p>${fStep}<p>Here ${M('q r')} is the chance that an attempt fails and the failure is loud:</p>${fQr}<p>All ${d.n} steps must end right for the task to succeed:</p>${fTask}<p>Without retries, a step has to be right on its first try (${fP(d.res.plain)}):</p>${fPlain}`,
    `<p><b>${fP(d.res.succ)}。</b></p><p>一步要算做对，得在重试用完之前有一次做对，而且前面每次做错都报了出来：</p>${fStep}<p>其中 ${M('q r')} 是一次尝试做错、而且报了错的概率：</p>${fQr}<p>${d.n} 步都要做对，任务才成功：</p>${fTask}<p>不重试的话，一步只能第一次就做对（${fP(d.res.plain)}）：</p>${fPlain}`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={n:{en:'Steps in the task',zh:'任务的步数'},p:{en:'Chance a step is right',zh:'一步做对的把握'},r:{en:'Share of failures that are loud',zh:'报出来的错占全部错的'},R:{en:'Retries per step',zh:'每一步最多重试'},ck:{en:'Check after each step',zh:'每一步做完检查'},kind:{en:'How a retry is done',zh:'怎样重试'}};
const TERM={
 p:()=>tx('p: the chance that one attempt at a step is right; its value is this lab’s assumption.','p：一步的一次尝试做对的把握，取值是本 Lab 的假设。'),
 n:()=>tx('n: the number of steps in the task. Each extra step multiplies the success rate by p once more, so it is one more chance to fail.','n：任务的步数。每多一步，成功率就要再乘一个 p，也就多一次出错的机会。'),
 r:()=>tx('Of all the failures, some are reported and seen by the harness (the program that runs the agent). This line is their share. With a check, the failures it finds count too. Only these failures can be retried.','全部的错里，有一部分会报出来，被 harness（运行 agent 的程序）看见。这一行算的是这部分错所占的比例。有检查时，检查发现的错也算在里面。只有这些错能重试。'),
 R:()=>tx('After a loud failure, a step is redone at most this many times. When the retries run out, the task stops.','一步报了错，最多再重做几次。次数用完，任务就停下。'),
 ck:()=>tx('Every attempt is one model call; a check after it is one more.','每次尝试是一次模型调用；之后的检查再加一次。'),
 kind:()=>tx('A same-call retry has nothing new to go on, so it is right less often (assumed 50%).','原样重试没有新信息，做对的把握更低（假设是 50%）。')
};
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: change the four choices in the task card.','挑战中固定不变：请改任务卡里的四个选择。');
  if(mode!=='guided') return '';
  const live=PART_CTL[STEPS[step].id]||[]; if(live.includes(k)||(k==='ck'&&live.includes('ck'))) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({n:'#arl-ctl-n',p:'#arl-ctl-p',r:'#arl-ic-r',R:'#arl-ic-R',ck:'#arl-ctl-ck',kind:'#arl-ctl-kind'}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#arl-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#arl-controls',app)); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=$(ctlEl(k),app)||$(({r:'#arl-ctl-r',R:'#arl-ctl-R'})[k]||'#arl-controls',app); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='play'){if(PL.on) pause(); else play(); return;}
  if(a==='stepf'){A.tip.quiet(); pause(); if(hideRes()&&!pred) return; if(PL.pv>=totalCols()){PL.pv=0; PL.fix=null;} PL.pv++; if(PL.pv>=totalCols()) finished(); tick(); return;}
  if(a==='end'){pause(); if(hideRes()&&!pred) return; PL.pv=totalCols(); finished(); tick(); return;}
  if(a==='reset'){byLearner(()=>{S={...BASE,seed:S.seed}; if(mode==='guided'){const s=STEPS[step]; if(s.enter) S={...BASE,...s.enter()};} render(); refresh();}); return;}
  if(a==='reseed'){byLearner(()=>setState({seed:S.seed+1})); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chO).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#arl-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#arl-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const t=e.target, k=t.dataset&&t.dataset.key;
    if(t.id==='arl-scrub'){pause(); if(hideRes()&&!pred) return; PL.fix=null; PL.pv=Math.max(0,Math.min(totalCols(),+t.value)); if(PL.pv>=totalCols()) finished(); tick(); return;}
    if(!k) return; const v=+t.value;
    const map={n:{n:v},p:{p:v/1000},r:{r:v/10},c:{c:v/10},pb:{pb:v/20}}[k]; if(map) bySettings(t,()=>setState(map));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#arl-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#arl-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){const first=!pred; pred=d.pred; renderRail(); if(first) restart(); const fb=$(`#arl-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null){const part=isP('chain')?2:(isP('check')?4:(isP('retry')?5:0)); if(!part) return; const f=()=>byLearner(()=>setLv(part,d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='speed'){PL.speed=v; drawPlayer(); return;}
      if(k==='R') bySettings(b,()=>setState({R:+v}));
      else if(k==='ck') bySettings(b,()=>setState({ck:+v}));
      else if(k==='kind') bySettings(b,()=>setState({kind:v}));
      else if(['chM','chC','chR','chN'].includes(k)){const f={chM:'strong',chC:'ck',chR:'R',chN:'n'}[k]; chO={...chO,[f]:+v}; byLearner(()=>{S=chState(chO); render(); refresh();});}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...BASE,seed:S.seed,...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#arl-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#arl-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the options tried in each part, the pictures watched to
     the end (hashed), the challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chO:{...chO},chSubmitted,chReveal,runsSeen,
    lv2,tried2:[...tried2],lv4,tried4:[...tried4],lv5,tried5:[...tried5],remView,sbTouched,drillSeed,drillOpen,speed:PL.speed,
    seen:[...new Set([...SEENH,...[...SEEN].map(hkey)])].slice(-200),passed:mode==='challenge'&&chSubmitted&&chEval(chO).ok}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(tokOf(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, vs=a=>a.map(x=>x.v), set=(v,ok,base)=>new Set([...base,...(Array.isArray(v)?v.filter(ok):[])]);
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null;
    const c=o.chO&&typeof o.chO==='object'?o.chO:{}; chO={strong:[0,1].includes(c.strong)?c.strong:CH.start.strong,ck:[0,1].includes(c.ck)?c.ck:CH.start.ck,R:CH.Rs.includes(c.R)?c.R:CH.start.R,n:CH.ns.includes(c.n)?c.n:CH.start.n};
    chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal); runsSeen=bool(o.runsSeen);
    const L2=vs(LV2), L4=vs(LV4), L5=vs(LV5);
    lv2=L2.includes(o.lv2)?o.lv2:'base'; tried2=set(o.tried2,v=>L2.includes(v),['base',lv2]);
    lv4=L4.includes(o.lv4)?o.lv4:'none'; tried4=set(o.tried4,v=>L4.includes(v),['none',lv4]);
    lv5=L5.includes(o.lv5)?o.lv5:'info'; tried5=set(o.tried5,v=>L5.includes(v),['info',lv5]);
    remView=o.remView===''||Object.keys(REMS).includes(o.remView)?o.remView:''; sbTouched=bool(o.sbTouched); drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen);
    if(['s','m','f'].includes(o.speed)) PL.speed=o.speed;
    if(Array.isArray(o.seen)) o.seen.forEach(h=>{if(typeof h==='string'&&/^[0-9a-z]{1,8}$/.test(h)) SEENH.add(h);});
    if(mode==='challenge') S=chState(chO);
    PL.on=false; PL.pv=0; PL.fix=null; PL.sig=null;},   // mount's refresh() shows a picture watched before finished and plays a new one
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'){if(!saved) saved={...BASE}; S=chState(chO);}
    el.innerHTML=viewHTML(); app=$('#arl-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#arl-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    if(mode==='guided'&&!p&&step===0&&!pred){S={...BASE};}
    fillRail(); render();
    /* a first view plays the run; a language switch or a return to the lab shows the same picture as it was (R3.20) */
    stopLoop(); refresh();
    if(PL.on&&!PL.raf){PL.last=0; PL.raf=requestAnimationFrame(frame);}
  },
  leave(){clearTimeout(annT); stageStop(); stopLoop(); root=null; app=null;},
  onLang(){if(root){const d=$('#arl-math',app), open=d&&d.open; stopLoop(); this.mount(root,null); if(open) $('#arl-math',app).open=true;}},
  onResize(){if(app){drawGrid(); drawCurve(); drawLegend();}},
  termTip(k){const f=TERM[k]; return f?`<div>${esc(f())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#arl-grid',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#arl-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(hideRes()){vis.innerHTML=''; const t=tallyNow(); read.innerHTML=esc(pred?tx(`Tasks running: ${t.n} of ${RUNS} have a result`,`任务在跑：${t.n} / ${RUNS} 个已有结果`):tx('Answer part 1 first','先回答第 1 部分的问题')); return;}
      const r=R_(), ch=mode==='challenge', price=ch?chState(chO).price:1, w=v=>(v*100).toFixed(2);
      if(simple()){vis.innerHTML=`<div class="arl-dk"><i class="ok" style="width:${w(r.succ)}%"></i><i class="st" style="width:${w(1-r.succ)}%"></i></div>`;
        read.innerHTML=esc(tx(`${stepsW(S.n)} at ${fPc(S.p)}: ${fP(r.succ)} succeed`,`${stepsW(S.n)}、每步 ${fPc(S.p)}：成功 ${fP(r.succ)}`)); return;}
      vis.innerHTML=`<div class="arl-dk"><i class="ok" style="width:${w(r.succ)}%"></i><i class="wr" style="width:${w(r.wrong)}%"></i><i class="st" style="width:${w(r.stopped)}%"></i></div>`;
      read.innerHTML=esc(ch?tx(`${fP(r.succ)} succeed, cost ${fC(r.calls*price)} calls per task`,`成功 ${fP(r.succ)}，调用成本 ${fC(r.calls*price)} 次`)
        :tx(`${stepsW(S.n)} at ${fPc(S.p)}: ${fP(r.succ)} succeed, ${fC(r.calls)} calls per task`,`${stepsW(S.n)}、每步 ${fPc(S.p)}：成功 ${fP(r.succ)}，每个任务 ${fC(r.calls)} 次调用`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__arl=Object.freeze({
  compute:s=>deep(compute({...BASE,...s})), u01, simRun:(s,t)=>deep(simRun({...BASE,...s},t)), tally:s=>deep(tallyOf(runsOf({...BASE,...s}))),
  chEval:o=>deep(chEval(o)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({BASE:{...BASE},RUNS,SPD:{...SPD},CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,P3:{...P3},P4:{...P4},P5:{...P5},
    LV2:deep(LV2.map(o=>({v:o.v,...o.s}))),LV4:deep(LV4.map(o=>({v:o.v,...o.s}))),LV5:deep(LV5.map(o=>({v:o.v,...o.s}))),CODE,SIMOUT}),
  tight:()=>tightNow(), simple:()=>simple(),
  state:()=>({...S,mode,step,pred,observed:{...observed},runsSeen,pv:PL.pv,playing:PL.on,speed:PL.speed,fix:PL.fix?PL.fix.slice():null,lv2,lv4,lv5,chO:{...chO},chSubmitted,chReveal,remView,stageBusy:stageT!==0})
});
})();
