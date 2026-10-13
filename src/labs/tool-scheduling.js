/* Tool scheduling lab: which tool calls can run in parallel? An agent books a trip (示例) with nine tool calls. The page schedules
   them three ways (one per model call; all in the first model call; in batches by dependency), with an optional lock on the one
   thing two calls write (the trip sheet), a concurrency limit k, and a timeout followed by one retry (with or without an
   idempotency key). Every time is the lab's own assumption; the schedule is computed, deterministic, and recomputed by
   tests/tool-scheduling_numbers.py. Message format and words as in agent-anatomy: model call, tool_use, tool_result, your code. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='tool-scheduling';

/* ---------- engine: pure functions (ENGINE-START) ---------- */
const TO=5;                 // the timeout of one attempt, seconds (the lab's assumption)
const MS=[0.5,1.5,3];       // model call times to choose from, seconds (assumed; 1.5 is the default)
const KS=[1,2,3,0];         // concurrency limit k; 0 = no limit
/* a call: key, the tool's name, a short name, its time in seconds (assumed), the calls whose output it needs, the calls whose
   writes it must see (after), and the one thing it writes that another call also writes (w) */
const TRIP={id:'trip',calls:[
 {k:'cal',fn:'get_calendar',n:{en:'Check calendar',zh:'查日历'},t:0.4,needs:[]},
 {k:'sfl',fn:'search_flights',n:{en:'Search flights',zh:'查航班'},t:2.5,needs:[]},
 {k:'sho',fn:'search_hotels',n:{en:'Search hotels',zh:'查酒店'},t:2.0,needs:[]},
 {k:'wea',fn:'get_weather',n:{en:'Get weather',zh:'查天气'},t:0.6,needs:[]},
 {k:'cmp',fn:'compare_prices',n:{en:'Compare prices',zh:'比价'},t:0.5,needs:[1,2]},
 {k:'bfl',fn:'book_flight',n:{en:'Book flight',zh:'订机票'},t:3.0,needs:[0,4],w:'sheet'},
 {k:'bho',fn:'book_hotel',n:{en:'Book hotel',zh:'订酒店'},t:2.0,needs:[4],w:'sheet'},
 {k:'evt',fn:'add_event',n:{en:'Add to calendar',zh:'写日程'},t:0.4,needs:[5,6]},
 {k:'eml',fn:'send_email',n:{en:'Send email',zh:'发邮件'},t:0.8,needs:[3,5,6]}]};
const CODE={id:'code',calls:[
 {k:'A',fn:'read_file',n:{en:'A read parser.py',zh:'A 读 parser.py'},t:0.3,needs:[]},
 {k:'B',fn:'read_file',n:{en:'B read test_parser.py',zh:'B 读 test_parser.py'},t:0.3,needs:[]},
 {k:'C',fn:'search',n:{en:'C search parse_date',zh:'C 搜 parse_date'},t:1.2,needs:[]},
 {k:'D',fn:'edit_file',n:{en:'D edit parser.py',zh:'D 改 parser.py'},t:0.4,needs:[0,2]},
 {k:'E',fn:'edit_file',n:{en:'E edit test_parser.py',zh:'E 改 test_parser.py'},t:0.4,needs:[1]},
 {k:'F',fn:'run_tests',n:{en:'F run the tests',zh:'F 跑测试'},t:4.0,needs:[],after:[3,4]}]};
const deps=c=>c.needs.concat(c.after||[]);
/* the level of a call: 1 if it waits for nobody, else one more than the deepest call it waits for */
function levels(task){const lv=[]; task.calls.forEach((c,i)=>{lv[i]=1+Math.max(0,...deps(c).map(j=>lv[j]));});
  const n=Math.max(...lv), out=[]; for(let l=1;l<=n;l++) out.push(task.calls.map((c,i)=>i).filter(i=>lv[i]===l)); return out;}
/* which calls each model call asks for: one per model call, all in the first, or by dependency */
function turnsFor(task,sch){const n=task.calls.length, all=[...Array(n).keys()];
  return sch==='seq'?all.map(i=>[i]):sch==='all'?[all]:levels(task);}
/* the turns of an assignment (the model call each call goes out in), empty turns dropped */
function turnsOf(asg){const used=[...new Set(asg)].sort((a,b)=>a-b); return used.map(u=>asg.map((v,i)=>v===u?i:-1).filter(i=>i>=0));}
const r3=v=>Math.round(v*1000)/1000;
/* schedule: each turn is one model call (time m), then your code runs that call's tools. In a turn, calls start in the order
   the model wrote them, as soon as fewer than k run (k 0: no limit) and, with the lock, no call that writes the same thing runs
   or is still waiting ahead of it. A timed-out attempt holds its slot for TO seconds, then one retry runs. After the last turn,
   one more model call writes the answer. */
function sched(task,turns,o){
  const C=task.calls, n=C.length, cs=C.map(()=>null), T=[], bad=[]; let t=0;
  const dur=i=>(o.half===i?C[i].t/2:C[i].t)+(o.fail===i?TO:0);
  turns.forEach((tn,j)=>{const ms=t, t0=r3(t+o.m), pend=tn.slice(), run=[]; let now=t0;
    while(pend.length||run.length){
      for(const i of pend.slice()){
        if(o.k&&run.length>=o.k) break;
        const w=C[i].w, ahead=pend.slice(0,pend.indexOf(i));
        if(o.lock&&w&&(run.some(r=>C[r].w===w)||ahead.some(p=>C[p].w===w))) continue;
        cs[i]={s:now,e:r3(now+dur(i)),turn:j,hang:o.fail===i?r3(now+TO):null}; run.push(i); pend.splice(pend.indexOf(i),1);}
      if(!run.length) break;
      now=Math.min(...run.map(r=>cs[r].e)); for(let q=run.length-1;q>=0;q--) if(cs[run[q]].e<=now+1e-9) run.splice(q,1);}
    const end=Math.max(t0,...tn.map(i=>cs[i].e)); T.push({ms,me:t0,calls:tn.slice(),end}); t=end;});
  const total=r3(t+o.m), turnOf=cs.map(c=>c?c.turn:Infinity);
  C.forEach((c,i)=>{if(!cs[i]) return;
    const miss=c.needs.filter(j=>!(turnOf[j]<turnOf[i])), early=(c.after||[]).filter(j=>!(turnOf[j]<turnOf[i]));
    if(miss.length) bad.push({i,kind:'miss',by:miss,at:cs[i].s});
    if(early.length) bad.push({i,kind:'early',by:early,at:cs[i].s});});
  for(let a=0;a<n;a++) for(let b=a+1;b<n;b++){if(!cs[a]||!cs[b]||!C[a].w||C[a].w!==C[b].w) continue;
    const x=cs[a], y=cs[b]; if(x.s<y.e-1e-9&&y.s<x.e-1e-9){const late=y.e>x.e?b:a; bad.push({i:late,kind:'lost',by:[late===a?b:a],at:cs[late].e});}}
  if(o.fail>=0&&C[o.fail].w&&!o.idem&&cs[o.fail]) bad.push({i:o.fail,kind:'double',by:[],at:cs[o.fail].e});
  /* the critical path: from the last call to finish, back through the call it waited for, then the model call before */
  const crit=new Set();
  for(let j=T.length-1;j>=0;j--){const tn=T[j]; if(!tn.calls.length) continue;
    let cur=tn.calls.reduce((a,b)=>cs[b].e>cs[a].e+1e-9?b:a);
    for(;;){crit.add(cur); if(cs[cur].s<=tn.me+1e-9) break; const p=tn.calls.find(q=>q!==cur&&Math.abs(cs[q].e-cs[cur].s)<1e-9); if(p==null) break; cur=p;}}
  const toolSum=r3(C.reduce((a,c,i)=>a+(cs[i]?cs[i].e-cs[i].s:0),0));
  return {turns:T,cs,total,n:T.length+1,toolSum,bad,crit,m:o.m};}
/* the challenge: every assignment of the six calls to model calls 1 to 6, with empty model calls dropped (4,683 distinct) */
const CH_K=2, CH_M=1.5, CH_GOAL=14;
const chOpt={k:CH_K,lock:1,m:CH_M,fail:-1,half:-1,idem:1};
function chAll(){const seen=new Map(), a=[0,0,0,0,0,0];
  const rec=p=>{if(p===6){const tn=turnsOf(a), key=tn.map(x=>x.join('')).join('|'); if(!seen.has(key)){const R=sched(CODE,tn,chOpt); seen.set(key,{key,turns:tn,total:R.total,n:R.n,right:!R.bad.length,ok:!R.bad.length&&R.total<=CH_GOAL+1e-9});} return;}
    for(let v=1;v<=6;v++){a[p]=v; rec(p+1);}};
  rec(0); const out=[...seen.values()], okp=out.filter(x=>x.ok), best=Math.min(...okp.map(x=>x.total)); out.forEach(x=>{x.best=x.ok&&Math.abs(x.total-best)<1e-9;}); return out;}
/* ENGINE-END */

/* ---------- words and helpers ---------- */
const s1=v=>Math.abs(v*10-Math.round(v*10))>1e-6?v.toFixed(2):v.toFixed(1);
const sec=v=>tx(`${s1(v)} s`,`${s1(v)} 秒`);
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<span class="linkbtn tls-j" data-jump="${k}">${t}</span>`;
const M=t=>`<div class="tls-math-line">${A.ui.math(t)}</div>`;
const MI=t=>A.ui.math(t);
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id];   // a lab that is not live is named in plain text (R6.39)
  if(!l) return esc(id); if(C.labStatus&&C.labStatus(l)!=='live') return esc(L(l.s||{en:id,zh:id}));
  return A.shell.labLink?A.shell.labLink(id):esc(id);};
const CN=(task,i)=>esc(L(task.calls[i].n));
const LG=()=>A.lang()==='zh'?'zh':'en';
const listJ=arr=>A.lang()==='zh'?(arr.length===2?arr.join('和'):arr.join('、')):(arr.length<3?arr.join(' and '):arr.slice(0,-1).join(', ')+' and '+arr[arr.length-1]);
const D_MC=()=>tx('A model call: your code sends the conversation to the model and waits for its reply. Here every model call takes 1.5 s, the lab’s assumption; a real one takes from under a second to many seconds, longer as the conversation grows.','一次模型调用：你的代码把对话发给模型，等它回复。这里每次算 1.5 秒，是本 Lab 的假设；真实的一次从不到 1 秒到很多秒都有，对话越长越慢。');
const D_TU=()=>tx('tool_use is a block in the model’s reply: the name of a tool and its arguments (input), with an id. It is only data; nothing runs until your code runs it.','tool_use 是模型回复里的一块：工具的名字和参数（input），再加一个 id。它只是一段数据，你的代码去执行它之前，什么都没有发生。');
const D_HARN=()=>tx('Your code is the harness: the program around the model that runs the agent. It calls the model, runs the tools, hands the results back, and decides when to retry and when to stop.','你的代码就是 harness：包在模型外面、负责运行 agent 的程序。它调用模型、执行工具、把结果交回去，并决定什么时候重试、什么时候停下。');
const D_SHEET=()=>tx('The trip sheet is one document that collects the confirmations. Each booking tool reads it, adds its line and saves it back. Two saves at the same time: the one saved last overwrites the other’s line.','行程单是一份汇总确认号的文档。每个订票工具都先读出它，加一行，再存回去。两个同时存，后存的会把先存的那行盖掉。');
const D_LOCK=()=>tx('A lock lets one call at a time change a given thing: a second call that wants to change it waits until the first is done. Calls that change nothing never wait for it.','锁：同一时间只让一个调用改某样东西；第二个要改的，等第一个做完再开始。不改它的调用不用等。');
const D_CRIT=()=>tx('The critical path is the chain of waits that sets the total time: in each batch the tool that finishes last (and whatever it queued behind), joined by the model calls between batches.','关键路径是决定总耗时的那条等待链：每一批里最后做完的那个工具（以及它排队等过的调用），再用两批之间的模型调用连起来。');
const D_K=()=>tx('The concurrency limit k: at most k tool calls run at the same moment. A call over the limit starts when one of the running calls finishes, in the order the model wrote them.','并发上限 k：同一时刻最多有 k 个工具调用在跑。超出的，等正在跑的有一个做完再开始，按模型写下它们的顺序。');
const D_TO=()=>tx(`A timeout is how long your code waits for one attempt of a tool call; here ${TO} s, the lab’s assumption. When it passes, your code gives up on that attempt and counts it as failed.`,`超时：你的代码等一次工具调用的最长时间，这里 ${TO} 秒（本 Lab 的假设）。到时还没有回复，就放弃这次尝试，当它失败了。`);
const D_RETRY=()=>tx('A retry sends the same tool call once more after an attempt failed. Here your code retries once, right after the timeout.','重试：一次尝试失败后，把同一个工具调用再发一次。这里你的代码在超时后马上重试一次。');
const D_IDEM=()=>tx('An idempotency key is a unique id your code gives one write and sends again with every retry of it. A service that has already seen the key returns the first result instead of doing the work again (Stripe’s API takes it as the Idempotency-Key header).','幂等键：你的代码给一次写操作配的唯一编号，每次重试都带着同一个。对方见过这个编号，就直接返回第一次的结果，不再做一遍（Stripe 的 API 用 Idempotency-Key 这个请求头接收它）。');

/* ---------- state ---------- */
const DEFAULT={sch:'dep',lock:1,k:0,fail:0,idem:1,m:1.5,half:-1};
const FAILI=[-1,2,5];
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null, pick2='';
let relang=false, acted=false, chA=[1,2,3,4,5,6], chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, userAct=false, annT=0, metKey=null;
const seen=new Set(), shownPrev={}, tried=new Set();
let speed='fast', anim=null, raf=0, lastTs=0;
const RATE={slow:1.0,mid:0.55,fast:0.3};   // real seconds per simulated second: one at a time (27.2 s) plays in about 8 s at 快 (R3.9)
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
const optOf=s=>({k:s.k,lock:s.lock,m:s.m,fail:FAILI[s.fail]||-1,half:s.half,idem:s.idem});
const runOf=s=>sched(TRIP,turnsFor(TRIP,s.sch),optOf(s));
const inCh=()=>mode==='challenge';
const curTask=()=>inCh()?CODE:TRIP;
const curRun=()=>inCh()?sched(CODE,turnsOf(chA),chOpt):runOf(S);
const runKey=()=>inCh()?'code|'+turnsOf(chA).map(x=>x.join('')).join('|'):'trip|'+JSON.stringify([turnsFor(TRIP,S.sch),optOf(S)]);
const seqTotal=m=>sched(TRIP,turnsFor(TRIP,'seq'),{k:0,lock:1,m,fail:-1,half:-1,idem:1}).total;

/* ---------- names of a setup ---------- */
const SCH_N={seq:{en:'One at a time',zh:'一个一个来'},all:{en:'All at once',zh:'一次全发'},dep:{en:'By dependency',zh:'按依赖分批'}};
const kTxt=k=>k?tx(`at most ${k} at once`,`同时最多 ${k} 个`):tx('no limit','并发不限');
function setTxt(s){const parts=[L(SCH_N[s.sch])];
  if(s.sch!=='seq') parts.push(s.lock?tx('one booking at a time on the sheet','行程单加锁'):tx('no lock on the sheet','行程单不加锁'));
  if(s.sch!=='seq') parts.push(kTxt(s.k));
  if(s.fail) parts.push(s.fail===1?tx('hotel search times out once','查酒店超时一次'):(s.idem?tx('flight booking times out, retried with a key','订机票超时，带幂等键重试'):tx('flight booking times out, retried without a key','订机票超时，不带幂等键重试')));
  if(s.half>=0) parts.push(tx(`${L(TRIP.calls[s.half].n)} twice as fast`,`${L(TRIP.calls[s.half].n)}快一倍`));
  if(s.m!==1.5) parts.push(tx(`model calls ${s1(s.m)} s`,`模型调用 ${s1(s.m)} 秒`));
  return parts.join(A.lang()==='zh'?'，':', ');}
/* what went wrong, in words */
/* Chinese text keeps a space where a Latin name (D 改 parser.py) meets a Chinese character */
const CJK=/[\u3400-\u9fff\uff00-\uffef\u3000-\u303f]/;
const zc=(...p)=>p.reduce((a,b)=>{if(!a||!b) return a+b; const x=a[a.length-1], y=b[0]; return a+((CJK.test(x)!==CJK.test(y))&&!/[\s“”（）、，：；。]/.test(x+y)?' ':'')+b;},'');
function badTxt(task,b){const q=task===CODE?(j=>tx(`“${CN(task,j)}”`,`“${CN(task,j)}”`)):(j=>CN(task,j)), c=q(b.i), by=listJ(b.by.map(q));
  if(b.kind==='miss') return tx(`${c} ran before the results of ${by} came back`,zc(c,'在',by,'的结果回来之前就执行了'));
  if(b.kind==='early') return tx(`${c} ran before ${by} had changed the files it reads`,zc(c,'在',by,'改完它要读的文件之前就跑了'));
  if(b.kind==='lost') return tx(`${c} saved the trip sheet over ${by}’s line`,zc(c,'存行程单时盖掉了',by,'写的那行'));
  return tx(`${c} was done twice: two tickets`,zc(c,'做了两次：订了两张票'));}
const resTxt=R=>R.bad.length?tx('wrong','错'):tx('right','对');

/* ---------- guided parts ---------- */
const PRED=['ok','wrong','slow'];
const predTxt=v=>({ok:tx('Much faster, and every result is still right','快很多，结果也都对'),wrong:tx('Much faster, but some calls give wrong results','快很多，但有几个调用的结果是错的'),slow:tx('Hardly faster than one at a time','和一个一个来差不多慢')})[v];
const segIn=(key,label,opts,val)=>A.ui.seg({id:'tls-ic-'+key,key,label,options:opts,value:String(val),inline:true});
const P2C=[5,6,8,3];
const LV3=[{v:'all',p:{sch:'all',lock:0}},{v:'dep0',p:{sch:'dep',lock:0}},{v:'dep1',p:{sch:'dep',lock:1}}];
const LV4=[{v:'-1'},{v:'3'},{v:'1'},{v:'5'}];
const LV6=[{v:'sho',p:{fail:1,idem:1}},{v:'bfl0',p:{fail:2,idem:0}},{v:'bfl1',p:{fail:2,idem:1}}];
const lv3Now=()=>S.sch==='all'&&!S.lock?'all':S.sch==='dep'?'dep'+S.lock:'';
const lv6Now=()=>S.fail===1?'sho':S.fail===2?'bfl'+S.idem:'';
const lvShow=v=>tried.has(String(v))||String(v)===lvCur();
const lvCur=()=>{const id=STEPS[step]&&STEPS[step].id; return id==='batch'?lv3Now():id==='crit'?String(S.half):id==='fail'?lv6Now():'';};
const resOf=p=>{const R=runOf({...cardS(),...p}); return tx(`${s1(R.total)} s, ${R.bad.length?'wrong':'right'}`,`${s1(R.total)} 秒，结果${R.bad.length?'错':'对'}`);};
function lvResTxt(id,v){if(id==='batch'){const o=LV3.find(x=>x.v===v); return o&&lvShow(v)?resOf(o.p):'';}
  if(id==='crit') return v==='-1'||lvShow(v)?sec(runOf({...DEFAULT,half:+v}).total):'';
  if(id==='fail'){const o=LV6.find(x=>x.v===v); return o&&lvShow(v)?resOf(o.p):'';} return '';}
function levers(name,opts,val){return A.ui.levers({aria:name,value:val,options:opts});}
const T_TEX=nm=>`\\text{${nm}} = n \\cdot m + \\sum_{j=1}^{b} w_j`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'Nine tool calls, one per model call.',zh:'9 个工具调用，每次模型调用只要一个。'},
  panels:['graph','gantt'],enter:()=>({...DEFAULT,sch:'seq'}),gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const R=runOf({...DEFAULT,sch:'seq'}), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(v=>`<button type="button" class="choice" data-pred="${v}" aria-pressed="${pred===v}">${esc(predTxt(v))}</button>`).join('')}</div>`;
   return tx(`<p>A travel agent gets a task (an example): “Book my trip to Shanghai on Friday and email me the plan.” It needs nine tool calls: check the calendar, search flights, search hotels, get the weather, compare prices, book the flight, book the hotel, add the trip to the calendar, send the email.</p><p>Right now it asks for one tool at a time: each ${DEF('model call',D_MC())} returns one ${DEF('tool_use',D_TU())}, ${DEF('your code',D_HARN())} runs it and sends the result back, then the next model call follows (the loop ${LL('agent-anatomy')} takes apart). The timeline plays one such run: ${R.n} model calls.</p><p>${LL('agent-anatomy')} showed that tools that do not depend on each other can be asked for in one reply. This lab answers which calls count as independent, and what can go wrong once they go out together.</p><p>In the dependency graph an arrow means “needs this result”: booking the flight needs the calendar (when is Friday free?) and the price comparison (which flight?).</p><p>Suppose the first model call asked for all nine calls and your code ran them all at once. What would happen?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it until you see the result.</p>':''}`,
             `<p>一个出差助手 agent 收到任务（示例）：“帮我订周五去上海的出差，订好后把行程发邮件给我。”它要调用 9 次工具：查日历、查航班、查酒店、查天气、比价、订机票、订酒店、写日程、发邮件。</p><p>现在它一次只要一个工具：每次${DEF('模型调用',D_MC())}只回一个 ${DEF('tool_use',D_TU())}，${DEF('你的代码',D_HARN())}执行完，把结果发回去，再进行下一次模型调用（${LL('agent-anatomy')}拆开讲过这个循环）。时间轴上在播这样跑一遍：一共 ${R.n} 次模型调用。</p><p>${LL('agent-anatomy')}里已经看到，互不依赖的工具可以在一次回复里一起要。这个 Lab 要回答的是：哪些调用算互不依赖，一起要以后又会出什么错。</p><p>依赖图里的箭头表示“要用这个结果”：订机票要用查日历（周五哪段时间有空）和比价（订哪一班）的结果。</p><p>如果第一次模型调用就把 9 个调用全要了，你的代码同时执行它们，会怎样？</p>${ch}${pred?'<p class="muted small">已记下。看到结果之前，还可以改。</p>':''}`);},
  obs:()=>obsMission(),label:{en:'This part’s setup',zh:'这一部分的设置'}},
 {id:'deps',title:{en:'Who waits for whom',zh:'谁要等谁'},lead:{en:'Two reasons to wait: a result, or a shared thing.',zh:'两种等法：等结果，或者等同一样东西。'},
  panels:['graph'],enter:()=>({...DEFAULT,sch:'seq'}),expect:()=>tried.size>1,obsAt:()=>!!pick2,label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick two calls to continue.',zh:'选两个调用再继续。'},
  body:()=>tx(`<p>One call waits for another for one of two reasons.</p><ul><li><b>It needs the other’s result.</b> To book the flight the model must know when Friday is free and which flight is cheapest. Until those results are in the conversation, it cannot write the booking’s arguments.</li><li><b>Both change the same thing.</b> Booking the flight and booking the hotel both write their confirmation into one ${DEF('trip sheet',D_SHEET())}. Run them at the same moment and the later save wipes out the earlier line.</li></ul><p>In the dependency graph, a solid arrow is the first reason; the dashed line between the two bookings is the second.</p><p>Pick a call and see what it waits for.</p>`,
              `<p>一个调用要等另一个，有两种原因。</p><ul><li><b>要用对方的结果。</b>订机票前，模型得知道周五哪段时间有空、哪一班最便宜。这些结果还没进到对话里，它就写不出订票的参数。</li><li><b>两个都要改同一样东西。</b>订机票和订酒店都要把确认号写进同一份${DEF('行程单',D_SHEET())}。两个同时做，后存的那个会把先存的那行盖掉。</li></ul><p>依赖图里，实线箭头是第一种原因；连着两个订票调用的虚线是第二种。</p><p>选一个调用，看看它要等谁。</p>`)+
   segIn('pick2',tx('A call','一个调用'),P2C.map(i=>[TRIP.calls[i].k,L(TRIP.calls[i].n)]),pick2),
  obs:()=>obsDeps()},
 {id:'batch',title:{en:'All at once, or in batches',zh:'一次全发，还是分批'},lead:{en:'Only calls whose inputs are ready go out together.',zh:'输入齐了的调用才能一起发。'},
  panels:['gantt','graph','metrics'],enter:()=>({...DEFAULT,sch:'all',lock:0}),expect:()=>tried.has('dep1'),obsAt:()=>true,label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try “by dependency, with a lock” to continue.',zh:'试一下“按依赖分批，行程单加锁”再继续。'},
  body:()=>{const R=runOf({...DEFAULT,sch:'all',lock:0}), ok=pred==='wrong';
   const v=pred?tx(`<p class="${ok?'okline':'badline'}"><b>${ok?'Right.':'Not quite.'}</b> ${ok?'':`You predicted “${esc(predTxt(pred))}”; the answer is “${esc(predTxt('wrong'))}”. `}All at once takes only ${sec(R.total)}, but ${R.bad.filter(b=>b.kind==='miss').length} calls ran before the results they need came back, so the model had to guess their arguments.</p>`,`<p class="${ok?'okline':'badline'}"><b>${ok?'对了。':'不对。'}</b>${ok?'':`你预测的是“${esc(predTxt(pred))}”，正确答案是“${esc(predTxt('wrong'))}”。`}一次全发只要 ${sec(R.total)}，可是有 ${R.bad.filter(b=>b.kind==='miss').length} 个调用在它要用的结果回来之前就执行了，模型只能猜它们的参数。</p>`)
     :tx(`<p>The answer: all at once takes ${sec(R.total)}, but ${R.bad.filter(b=>b.kind==='miss').length} calls run before the results they need come back.</p>`,`<p>答案：一次全发要 ${sec(R.total)}，但有 ${R.bad.filter(b=>b.kind==='miss').length} 个调用在它要用的结果回来之前就执行了。</p>`);
   return v+tx(`<p>One reply of the model can hold several tool_use blocks, and your code can run them at the same time. Both Anthropic’s and OpenAI’s APIs allow this by default. In this lab your code waits until every tool of the batch is back and sends all the tool_result blocks in one message, so the next model call waits for the slowest tool of the batch.</p><p>Batching by dependency means each model call asks for every call whose inputs are already in; a call that needs another’s result is asked for in the next model call. A ${DEF('lock',D_LOCK())} on the trip sheet means the two calls that change it never run at once: your code starts the second only when the first is done. Your code can only lock the whole call, so the hotel booking waits for the entire flight booking. If the trip sheet offered an “add one line” operation, the two bookings could run together and queue only for that line.</p><p>Pick a way to run the task and watch the total time and the result.</p>`,
                 `<p>模型的一次回复里可以有好几个 tool_use，你的代码可以同时执行它们。Anthropic 和 OpenAI 的 API 默认都允许这样做。在这个 Lab 里，你的代码等这一批全部回来，再把所有 tool_result 放进同一条消息发回去，所以下一次模型调用要等这一批里最慢的那个工具。</p><p>按依赖分批，是说每次模型调用把输入已经齐了的调用全部要出来；要用别人结果的，留到下一次模型调用再要。给行程单加${DEF('锁',D_LOCK())}，是说两个要改行程单的调用不同时跑：你的代码让后一个等前一个做完再开始。你的代码只能锁住整个调用，所以订酒店要等订机票整个做完。如果行程单本身提供“追加一行”的接口，两个订票就能同时跑，只在写那一行时排队。</p><p>选一种跑法，看看总耗时和结果。</p>`)+
    levers(tx('How to run the task','怎么跑这个任务'),[
     {v:'all',title:tx('All at once','一次全发'),desc:tx('All nine in the first model call, run together.','第一次模型调用就要全部 9 个，一起执行。'),result:lvShow('all')?resOf(LV3[0].p):''},
     {v:'dep0',title:tx('By dependency, no lock','按依赖分批，行程单不加锁'),desc:tx('Batches by dependency; the two bookings run together.','按依赖分批；两个订票调用一起跑。'),result:lvShow('dep0')?resOf(LV3[1].p):''},
     {v:'dep1',title:tx('By dependency, with a lock','按依赖分批，行程单加锁'),desc:tx('Batches by dependency; the hotel booking waits for the flight booking.','按依赖分批；订酒店等订机票做完再开始。'),result:lvShow('dep1')?resOf(LV3[2].p):''}],lv3Now());},
  obs:()=>obsBatch()},
 {id:'crit',title:{en:'The critical path',zh:'关键路径'},lead:{en:'Each batch costs as much as its slowest tool.',zh:'每一批花的时间，由它最慢的那个工具决定。'},
  panels:['gantt','graph','metrics'],enter:()=>({...DEFAULT}),expect:()=>[...tried].filter(x=>x!=='-1').length>1,obsAt:()=>acted,label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Speed up two different tools to continue.',zh:'分别让两个工具变快，再继续。'},
  body:()=>tx(`<p>Your code sends a batch’s results back together, so every model call waits until the previous batch has finished. So the total time is the time of every model call plus, for every batch, the time until its last tool is done:</p>${M(T_TEX('total time'))}<p>Here ${MI('n')} counts the model calls (the last one writes the answer), ${MI('m')} is the time of one, ${MI('b')} is the number of batches and ${MI('w_j')} is how long batch ${MI('j')} takes. The tool that finishes last in each batch, plus any call it queued behind (here the flight booking, which the hotel booking waits for), joined by the model calls, form the ${DEF('critical path',D_CRIT())}, outlined thick in the timeline.</p><p>Make one tool twice as fast and watch the total time.</p>`,
              `<p>你的代码把一批的结果一起发回去，所以每次模型调用都要等上一批全部做完。总耗时因此是：每次模型调用的时间，加上每一批从开始到最后一个工具做完的时间。</p>${M(T_TEX('总耗时'))}<p>${MI('n')} 是模型调用的次数（最后一次给出答案），${MI('m')} 是一次的时间，${MI('b')} 是批数，${MI('w_j')} 是第 ${MI('j')} 批花的时间。每一批里最后做完的工具，加上它排队等过的调用（比如订酒店等着的订机票），再用模型调用连起来，就是${DEF('关键路径',D_CRIT())}，时间轴上用粗框标出。</p><p>选一个工具，让它快一倍，看看总耗时。</p>`)+
   levers(tx('Make one tool twice as fast','让一个工具快一倍'),LV4.map(o=>{const h=+o.v; return {v:o.v,title:h<0?tx('None (initial setup)','都不变（初始设置）'):tx(`${L(TRIP.calls[h].n)}: ${s1(TRIP.calls[h].t)} → ${s1(TRIP.calls[h].t/2)} s`,`${L(TRIP.calls[h].n)}：${s1(TRIP.calls[h].t)} → ${s1(TRIP.calls[h].t/2)} 秒`),result:h<0||lvShow(o.v)?sec(runOf({...DEFAULT,half:h}).total):''};}),String(S.half)),
  obs:()=>obsCrit()},
 {id:'limit',title:{en:'How many at once',zh:'同时最多跑几个'},lead:{en:'A limit only bites in the widest batch.',zh:'上限只在最宽的那一批起作用。'},
  panels:['gantt','graph','metrics'],enter:()=>({...DEFAULT}),expect:()=>tried.size>1,obsAt:()=>acted,label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another limit to continue.',zh:'换一个上限再继续。'},
  body:()=>tx(`<p>The services your tools call often have rate limits, and every open connection or browser tab takes resources, so your code often caps how many tool calls run at once. The ${DEF('concurrency limit k',D_K())} is the most tool calls that run at the same moment; calls over the limit queue in the order the model wrote them, and each starts when a running one finishes.</p><p>The widest batch of this task has four calls: calendar, flights, hotels and weather. Lower k and watch the total time.</p>`,
              `<p>被调用的服务往往有限流，每个连接、每个浏览器标签页也都要占资源，所以你的代码常要给同时跑的工具调用设一个上限。${DEF('并发上限 k',D_K())} 是同一时刻最多在跑的工具调用数；超出的按模型写下它们的顺序排队，等正在跑的有一个做完再开始。</p><p>这个任务最宽的一批有 4 个调用：查日历、查航班、查酒店、查天气。把 k 调小，看看总耗时。</p>`)+
   segIn('k',tx('Concurrency limit k','并发上限 k'),KS.map(k=>[String(k),k?String(k):tx('No limit','不限')]),S.k),
  obs:()=>obsLimit()},
 {id:'fail',title:{en:'Timeouts and retries',zh:'超时和重试'},lead:{en:'Retry reads freely; retry writes with a key.',zh:'读可以放心重试；写要带幂等键。'},
  panels:['gantt','graph','metrics'],enter:()=>({...DEFAULT}),expect:()=>tried.has('bfl0')&&tried.has('bfl1'),obsAt:()=>acted,label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try both flight-booking cases to continue.',zh:'两种订机票超时的情况都试一下，再继续。'},
  body:()=>tx(`<p>A tool call sometimes hangs. Your code gives every attempt a ${DEF('timeout',D_TO())} (here ${TO} s): when no reply has come by then, it gives up on that attempt and makes one ${DEF('retry',D_RETRY())} (why retries matter: ${LL('agent-reliability')}).</p><p>Retrying a call that only reads is safe. A write is different: a timeout only says the reply did not arrive, and the work may already be done. An ${DEF('idempotency key',D_IDEM())} is a unique id your code gives the write and sends again with the retry; a service that has seen it returns the first result instead of acting twice.</p><p>Pick a case and see what happens.</p>`,
              `<p>工具调用有时会卡住不回。你的代码给每次尝试定一个${DEF('超时',D_TO())}（这里 ${TO} 秒）：到时还没有回复，就放弃这次尝试，再${DEF('重试',D_RETRY())}一次（重试为什么重要，见 ${LL('agent-reliability')}）。</p><p>只读的调用，重试没有风险。写的调用不一样：超时只说明回复没到，对方可能已经做完了。${DEF('幂等键',D_IDEM())}是你的代码给这次写操作配的唯一编号，重试时带上同一个；对方见过这个编号，就返回第一次的结果，不会再做一遍。</p><p>选一种情况，看看会怎样。</p>`)+
   levers(tx('A call times out','一个调用超时了'),[
    {v:'sho',title:tx('Hotel search times out','查酒店超时'),desc:tx('A read: the retry searches again.','只读：重试就是再查一次。'),result:lvShow('sho')?resOf(LV6[0].p):''},
    {v:'bfl0',title:tx('Flight booking times out, retried without a key','订机票超时，不带幂等键重试'),desc:tx('The first attempt did book the seat; only the reply was lost.','第一次其实订上了，只是回复丢了。'),result:lvShow('bfl0')?resOf(LV6[1].p):''},
    {v:'bfl1',title:tx('Flight booking times out, retried with a key','订机票超时，带幂等键重试'),desc:tx('Same case; the retry carries the first attempt’s key.','同样的情况；重试带着第一次的幂等键。'),result:lvShow('bfl1')?resOf(LV6[2].p):''}],lv6Now()),
  obs:()=>obsFail()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Graph first, then batches, locks and keys.',zh:'先画依赖图，再分批、加锁、带幂等键。'},
  panels:['gantt','graph','metrics'],enter:()=>({...DEFAULT}),
  body:()=>{const d=runOf(DEFAULT), q=runOf({...DEFAULT,sch:'seq'}), w=runOf({...DEFAULT,half:3});
   return tx(`<ul><li><b>Draw the dependency graph before choosing what goes out together.</b> A call that needs no other call’s result and changes nothing another call changes can go out with the others: in the trip, the calendar, flight, hotel and weather lookups all go out in the first model call. Doing this at every model call takes the trip from ${sec(q.total)} to ${sec(d.total)}, with the same result.</li><li><b>To go faster, shorten the critical path and cut model calls.</b> Each model call waits for the slowest tool of the batch before it; making the weather lookup twice as fast saves ${d.total-w.total>1e-6?sec(d.total-w.total):'nothing'}, because the flight search is slower. Each model call saved saves a whole model call (${sec(DEFAULT.m)} here).</li><li><b>Calls that change things need a lock and an idempotency key.</b> Two bookings saving the same trip sheet at once lose a line; a booking retried after a timeout without a key books twice.</li></ul>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Run everything in parallel; more parallel is always faster.” A call sent before its inputs are in runs on a guess, and a limit above the widest batch adds nothing.</p><p>✗ “The total time is the sum of the tool times.” Tools that run together count once, by the slowest; and model calls take time too: in this lab, by dependency, ${sec(d.n*DEFAULT.m)} of the ${sec(d.total)} is model calls.</p><p>✗ “A call that timed out did not happen.” A timeout only means the reply did not arrive in time; the payment or the email may have gone through.</p></div>`,
             `<ul><li><b>先画依赖图，再决定哪些一起发。</b>一个调用只要不用别人的结果、也不和别人改同一样东西，就能和别人一起发。出差任务里，查日历、查航班、查酒店、查天气在第一次模型调用里就能一起要出来。每一次模型调用都这样把输入已齐的调用一起要出来，任务就从 ${sec(q.total)}降到 ${sec(d.total)}，结果不变。</li><li><b>想变快，就缩短关键路径、减少模型调用。</b>每次模型调用都要等前一批里最慢的工具；让查天气快一倍，总耗时${d.total-w.total>1e-6?`少 ${sec(d.total-w.total)}`:'一点也不少'}，因为查航班比它慢。每少一次模型调用，就少等一整次模型调用（这里 ${sec(DEFAULT.m)}）。</li><li><b>会改东西的调用，要加锁、带幂等键。</b>两个订票调用同时存同一份行程单，会丢掉一行；订票超时后不带幂等键重试，会订出两张票。</li></ul>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “能并行的都并行，越并行越快。”输入还没齐就发出去的调用，参数只能靠猜；并发上限超过最宽的那一批以后，也不会再快。</p><p>✗ “总耗时就是所有工具时间的总和。”一起跑的工具只算最慢的那个；模型调用本身也要时间：在这个 Lab 里，按依赖分批时，${sec(d.total)}里有 ${sec(d.n*S.m)}花在模型调用上。</p><p>✗ “超时了，就说明没做成。”超时只说明回复没按时回来；钱可能已经扣了，邮件可能已经发了。</p></div>`)+A.shell.toScenarios();}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
/* the setup a guided part's card explains (R09): the part's own starting setup, with the settings its card's own controls change
   taken from S. When Settings put the stage on anything else, the card says so in a line of its own (nowHTML) and its explanation
   stays on this setup. Part 2 shows only the dependency graph, which no setting changes. */
const OWN={batch:['sch','lock'],crit:['half'],limit:['k'],fail:['fail','idem']}, SKEYS=['sch','lock','k','fail','idem','m','half'];
const partBase=()=>{const s=STEPS[step]; return {...DEFAULT,...(s.enter?s.enter():{})};};
function cardS(){if(mode!=='guided') return S; const b=partBase(); (OWN[STEPS[step].id]||[]).forEach(k=>{b[k]=S[k];}); return b;}
const offCard=()=>{if(mode!=='guided'||STEPS[step].id==='deps') return false; const c=cardS(); return SKEYS.some(k=>S[k]!==c[k]);};
function nowHTML(){if(!offCard()) return ''; const R=runOf(S), c=cardS(), live=!!anim&&anim.key===runKey();
  const res=live?tx(' It is still playing.','还在播放。'):tx(` ${R.n} model calls, ${sec(R.total)}, the result is ${resTxt(R)}.`,`${R.n} 次模型调用，${sec(R.total)}，结果${resTxt(R)}。`);
  return tx(`<p><b>The timeline now shows another setup than this card.</b> You changed Settings, so the timeline now runs: ${esc(setTxt(S))}.${res}</p><p>The explanation and the numbers in this card are about this part’s setup: ${esc(setTxt(c))}.</p><button type="button" class="btn ghost sm" data-act="card">Back to this part’s setup</button>`,
            `<p><b>时间轴上现在不是这张卡片讲的设置。</b>你在“自由调参”里改了设置，时间轴上现在是：${esc(setTxt(S))}。${res}</p><p>这张卡片里的说明和数字，讲的都是这一部分的设置：${esc(setTxt(c))}。</p><button type="button" class="btn ghost sm" data-act="card">回到这一部分的设置</button>`);}

/* ---------- observations ---------- */
function obsMission(){const R=runOf({...DEFAULT,sch:'seq'});
  return tx(`<p><b>One at a time: ${R.n} model calls (nine ask for a tool, the last one answers) and ${sec(R.total)} in all.</b></p><p>The tools themselves take only ${sec(R.toolSum)}; the rest of the time your code is waiting for the model.</p>`,
            `<p><b>一个一个来：${R.n} 次模型调用（每个工具一次，最后一次给出答案），一共 ${sec(R.total)}。</b></p><p>工具本身只用了 ${sec(R.toolSum)}，其余时间你的代码都在等模型。</p>`);}
function obsDeps(){const i=TRIP.calls.findIndex(c=>c.k===pick2); if(i<0) return '';
  const n=k=>L(TRIP.calls[TRIP.calls.findIndex(c=>c.k===k)].n);
  return ({bfl:tx(`<p><b>${n('bfl')} waits for ${n('cal')} and ${n('cmp')}; ${n('cmp')} in turn waits for ${n('sfl')} and ${n('sho')}.</b></p><p>It also changes the trip sheet, like ${n('bho')}, so the two must not save it at the same moment.</p>`,`<p><b>${n('bfl')}要等${n('cal')}和${n('cmp')}；${n('cmp')}又要等${n('sfl')}和${n('sho')}。</b></p><p>它和${n('bho')}都要改行程单，两个不能同时存。</p>`),
    bho:tx(`<p><b>${n('bho')} only waits for ${n('cmp')}, which waits for both searches.</b></p><p>It changes the trip sheet too, so it must not run at the same moment as ${n('bfl')}.</p>`,`<p><b>${n('bho')}只等${n('cmp')}，${n('cmp')}要等两个查询。</b></p><p>它也要改行程单，所以不能和${n('bfl')}同时跑。</p>`),
    eml:tx(`<p><b>${n('eml')} waits for ${n('bfl')}, ${n('bho')} (the confirmation numbers) and ${n('wea')}.</b></p><p>It sits at the end of the longest chain of waiting: calendar or searches, then the comparison, then the bookings, then the email.</p>`,`<p><b>${n('eml')}要等${n('bfl')}、${n('bho')}（确认号）和${n('wea')}。</b></p><p>它在最长那条等待链的末尾：先查询，再比价，再订票，最后发邮件。</p>`),
    wea:tx(`<p><b>${n('wea')} waits for nobody.</b></p><p>Neither do ${n('cal')}, ${n('sfl')} and ${n('sho')}: these four calls can all go out in the first model call.</p>`,`<p><b>${n('wea')}谁也不用等。</b></p><p>${n('cal')}、${n('sfl')}、${n('sho')}也一样：这 4 个调用在第一次模型调用里就能一起要出来。</p>`)})[pick2]||'';}
function obsBatch(){const s=cardS(), R=runOf(s), q=runOf({...s,sch:'seq',lock:1}), v=lv3Now(), nMiss=R.bad.filter(b=>b.kind==='miss').length;
  if(v==='all') return tx(`<p><b>All at once: ${R.n} model calls and ${sec(R.total)}, but the result is wrong.</b></p><p>${nMiss} calls ran before the results they need came back, so the model had to guess their arguments. For example, it booked the 8 am flight without waiting for the calendar, but there is a meeting that morning. The email went out before there were any confirmation numbers. And the two bookings saved the trip sheet at the same moment.</p>`,
                          `<p><b>一次全发：${R.n} 次模型调用，${sec(R.total)}，但结果是错的。</b></p><p>有 ${nMiss} 个调用在它要用的结果回来之前就执行了，参数只能靠模型猜。比如它没等查日历，就订了早上 8 点的航班，可那天上午有会。邮件发出去时，也还没有确认号。另外，两个订票调用同时存了行程单。</p>`);
  if(v==='dep0'){const b=R.bad.find(x=>x.kind==='lost');
    return tx(`<p><b>By dependency, no lock: ${R.n} model calls and ${sec(R.total)}, but the trip sheet is wrong.</b></p><p>Every call now has its inputs. But the two bookings ran together and ${b?badTxt(TRIP,b):''}: the sheet the email is built from has no hotel.</p>`,
              `<p><b>按依赖分批、不加锁：${R.n} 次模型调用，${sec(R.total)}，但行程单是错的。</b></p><p>现在每个调用的输入都齐了。可两个订票调用一起跑，${b?badTxt(TRIP,b):''}：邮件照着它写，里面没有酒店。</p>`);}
  if(v==='dep1') return tx(`<p><b>By dependency, with a lock: ${R.n} model calls and ${sec(R.total)}, and the result is right.</b></p><p>That is ${sec(q.total-R.total)} less than one at a time, with the same result.</p><p>In this lab every model call takes ${sec(s.m)} however many tools it asks for, so most of the saving comes from ${q.n-R.n} fewer model calls. The price of the lock: the hotel booking waits for the flight booking.</p><p>A real model writes out every tool_use, so a reply that asks for more tools takes longer and the saving is smaller (${LL('agent-anatomy')} shows how a model call grows with what it writes).</p>`,
                           `<p><b>按依赖分批、行程单加锁：${R.n} 次模型调用，${sec(R.total)}，结果对。</b></p><p>比一个一个来少了 ${sec(q.total-R.total)}，结果一样。</p><p>在这个 Lab 里，一次模型调用不管要几个工具都算 ${sec(s.m)}，所以省下的时间大半来自少了 ${q.n-R.n} 次模型调用。加锁的代价是订酒店要等订机票做完。</p><p>真实的模型要把每个 tool_use 都写出来，一次要的工具多，这次调用也会长一些，省下的就没这么多（${LL('agent-anatomy')}讲过模型调用的时间怎么随写出的内容变长）。</p>`);
  return genObs();}
function genObs(){const s=cardS(), R=runOf(s);
  return tx(`<p class="small">${esc(setTxt(s))}.</p><p><b>${R.n} model calls, ${sec(R.total)}, the result is ${resTxt(R)}.</b></p>${R.bad.length?`<p>${badTxt(TRIP,R.bad[0])}.</p>`:''}`,`<p class="small">${esc(setTxt(s))}。</p><p><b>${R.n} 次模型调用，${sec(R.total)}，结果${resTxt(R)}。</b></p>${R.bad.length?`<p>${badTxt(TRIP,R.bad[0])}。</p>`:''}`);}
function obsCrit(){const s=cardS(), R=runOf(s), b=runOf({...s,half:-1}), h=s.half;
  if(h<0) return tx(`<p><b>Initial setup: ${sec(R.total)}.</b> The critical path runs through the flight search, the comparison, both bookings and the email.</p>`,`<p><b>初始设置：${sec(R.total)}。</b>关键路径经过查航班、比价、两个订票调用和发邮件。</p>`);
  const c=TRIP.calls[h], nm=L(c.n), d=b.total-R.total;
  if(h===3) return tx(`<p><b>${nm} from ${sec(c.t)} to ${sec(c.t/2)}: still ${sec(R.total)}.</b></p><p>In its batch the flight search takes ${sec(TRIP.calls[1].t)}, and the next model call waits for that anyway. A call off the critical path can get faster without changing the total.</p>`,`<p><b>${nm}从 ${sec(c.t)}变成 ${sec(c.t/2)}：总耗时还是 ${sec(R.total)}。</b></p><p>同一批里查航班要 ${sec(TRIP.calls[1].t)}，下一次模型调用本来就要等它。不在关键路径上的调用，再快也不改变总耗时。</p>`);
  if(h===1) return tx(`<p><b>${nm} from ${sec(c.t)} to ${sec(c.t/2)}: only ${sec(d)} less.</b></p><p>Now the hotel search, at ${sec(TRIP.calls[2].t)}, is the slowest of the first batch, and the next model call waits for it instead.</p>`,`<p><b>${nm}从 ${sec(c.t)}变成 ${sec(c.t/2)}：总耗时只少了 ${sec(d)}。</b></p><p>现在第一批里最慢的是查酒店（${sec(TRIP.calls[2].t)}），下一次模型调用改成等它。</p>`);
  return tx(`<p><b>${nm} from ${sec(c.t)} to ${sec(c.t/2)}: the total drops just as much.</b></p><p>It is on the critical path, and the hotel booking queues behind it on the trip sheet, so every second it saves comes off the total.</p>`,`<p><b>${nm}从 ${sec(c.t)}变成 ${sec(c.t/2)}：总耗时也整整少了这么多。</b></p><p>它在关键路径上，订酒店又排在它后面等行程单，所以它省下的每一秒都从总耗时里扣掉。</p>`);}
function obsLimit(){const s=cardS(), R=runOf(s), u=runOf({...s,k:0}), one=runOf({...s,sch:'seq',lock:1,k:0});
  if(s.k===1) return tx(`<p><b>k = 1: ${sec(R.total)}, against ${sec(u.total)} with no limit.</b></p><p>Inside each batch the tools now run one by one. It is still faster than one at a time (${sec(one.total)}), because there are still only ${R.n} model calls.</p>`,`<p><b>k = 1：${sec(R.total)}，不设上限是 ${sec(u.total)}。</b></p><p>每一批里的工具也只能一个一个跑了。可它还是比一个一个来（${sec(one.total)}）快，因为模型调用仍然只有 ${R.n} 次。</p>`);
  if(s.k===2) return tx(`<p><b>k = 2: ${sec(R.total)}, only ${sec(R.total-u.total)} more than with no limit.</b></p><p>The first batch starts in the order calendar check, flight search, hotel search, weather, and can run only two at once: the hotel search starts when the calendar check is done, and the weather when the hotel search is done. The weather finishes ${sec(R.total-u.total)} after the flight search, so the next model call waits that much longer.</p>`,`<p><b>k = 2：${sec(R.total)}，只比不限多 ${sec(R.total-u.total)}。</b></p><p>第一批按查日历、查航班、查酒店、查天气的顺序开始，同时只能跑两个：查日历做完，查酒店才开始；查酒店做完，查天气才开始。查天气比查航班晚 ${sec(R.total-u.total)}做完，下一次模型调用就多等这 ${sec(R.total-u.total)}。</p>`);
  if(s.k===3) return tx(`<p><b>k = 3: ${sec(R.total)}, the same as with no limit.</b></p><p>The weather starts when the calendar check is done and still finishes long before the flight search. More than three at once would change nothing here.</p>`,`<p><b>k = 3：${sec(R.total)}，和不限一样。</b></p><p>查天气等查日历做完再开始，也还是比查航班早结束很多。在这里，允许超过 3 个同时跑也不会更快。</p>`);
  return tx(`<p><b>No limit: ${sec(R.total)}.</b> The first batch runs all four lookups at once.</p>`,`<p><b>不限：${sec(R.total)}。</b>第一批的 4 个查询同时跑。</p>`);}
function obsFail(){const s=cardS(), R=runOf(s), b=runOf({...s,fail:0}), v=lv6Now();
  if(v==='sho') return tx(`<p><b>Hotel search times out: ${sec(R.total)} instead of ${sec(b.total)}, and the result is right.</b></p><p>The first attempt hangs; after ${TO} s your code gives up and searches again. The whole first batch waits for it. A search changes nothing, so trying again is safe.</p>`,`<p><b>查酒店超时：${sec(R.total)}，原来是 ${sec(b.total)}，结果对。</b></p><p>第一次尝试卡住了；${TO} 秒后你的代码放弃这次，再查一次。整个第一批都在等它。查询不改任何东西，再试一次很安全。</p>`);
  if(v==='bfl0') return tx(`<p><b>Flight booking retried without a key: ${sec(R.total)}, and two tickets.</b></p><p>The airline booked the seat on the first attempt; only its reply was lost. To the airline the retry is a new booking, so the card is charged twice.</p>`,`<p><b>订机票不带幂等键重试：${sec(R.total)}，订了两张票。</b></p><p>航空公司在第一次尝试时就订上了，只是回复丢了。对它来说，重试是一笔新订单，所以卡被扣了两次钱。</p>`);
  if(v==='bfl1') return tx(`<p><b>Flight booking retried with the key: ${sec(R.total)}, one ticket.</b></p><p>The retry carries the first attempt’s idempotency key, so the airline returns the booking it already made. The time is the same as without a key: the key changes what happens, not how long it takes.</p>`,`<p><b>订机票带幂等键重试：${sec(R.total)}，只有一张票。</b></p><p>重试带着第一次的幂等键，航空公司认出是同一笔，返回已经订好的那张。时间和不带键一样：幂等键改变的是结果，不是耗时。</p>`);
  return genObs();}

/* ---------- scenarios ---------- */
const PRESETS=[
 {k:'init',t:{en:'Trip booking (initial setup)',zh:'出差预订（初始设置）'},d:{en:'By dependency, a lock on the trip sheet, no concurrency limit.',zh:'按依赖分批，行程单加锁，并发不限。'},p:{...DEFAULT}},
 {k:'seq',t:{en:'Parallel tool use turned off',zh:'关掉并行工具调用'},d:{en:'At most one tool_use per reply, as disable_parallel_tool_use or parallel_tool_calls=false gives.',zh:'每次回复最多一个 tool_use，就像设了 disable_parallel_tool_use 或 parallel_tool_calls=false。'},p:{...DEFAULT,sch:'seq'}},
 {k:'k2',t:{en:'A harness with a concurrency limit of 2',zh:'并发上限 2 的 harness'},d:{en:'Your code runs at most 2 tools at once (k = 2 from part 5).',zh:'你的代码同时最多跑 2 个工具（第 5 部分的 k = 2）。'},p:{...DEFAULT,k:2}},
 {k:'slow',t:{en:'A long conversation',zh:'对话很长'},d:{en:'Every model call takes 3 s instead of 1.5 s.',zh:'每次模型调用要 3 秒，而不是 1.5 秒。'},p:{...DEFAULT,m:3}},
 {k:'retry',t:{en:'A booking times out',zh:'订机票超时'},d:{en:'The flight booking hangs once; the retry carries an idempotency key.',zh:'订机票卡住一次；重试带着幂等键。'},p:{...DEFAULT,fail:2,idem:1}}];
const same=(a,b)=>['sch','lock','k','fail','idem','m'].every(x=>a[x]===b[x])&&a.half===-1;
const presetNow=()=>PRESETS.findIndex(p=>same(S,p.p));
function sandboxHTML(){const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how the batches, the limit and a timeout change the total time and the result.','选一个场景，看看分批方式、并发上限和超时怎样改变总耗时和结果。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="tls-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function insightHTML(){const R=runOf(S), i=presetNow(), base=runOf(DEFAULT);
  let h=tx(`<p class="small">${esc(setTxt(S))}.</p><p><b>${R.n} model calls, ${sec(R.total)}, the result is ${resTxt(R)}.</b></p>`,`<p class="small">${esc(setTxt(S))}。</p><p><b>${R.n} 次模型调用，${sec(R.total)}，结果${resTxt(R)}。</b></p>`);
  if(i<0) return h+(R.bad.length?`<p>${badTxt(TRIP,R.bad[0])}${tx('.','。')}</p>`:'')+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  const k=PRESETS[i].k;
  if(k==='init') h+=tx(`<p>Four lookups go out together, then the comparison, then the two bookings one after the other, then the calendar entry and the email: five model calls in all.</p>`,`<p>先一起发 4 个查询，再比价，再一个接一个地订机票和酒店，最后写日程、发邮件：一共 5 次模型调用。</p>`);
  else if(k==='seq') h+=tx(`<p>${R.n-base.n} more model calls than the initial setup and ${sec(R.total-base.total)} longer. The API can return one tool_use per reply when parallel tool use is turned off, and some models may be less likely to ask for several at once even when it is on.</p>`,`<p>比初始设置多 ${R.n-base.n} 次模型调用，慢了 ${sec(R.total-base.total)}。关掉并行工具调用时，API 每次回复最多给一个 tool_use；有的模型即使没关，也可能较少一次要几个工具。</p>`);
  else if(k==='k2') h+=tx(`<p>Only ${sec(R.total-base.total)} slower than with no limit: only the first batch has more than two calls, and the slowest call in it, the flight search, starts at once.</p>`,`<p>只比不限慢 ${sec(R.total-base.total)}：只有第一批超过 2 个调用，而这一批里最慢的查航班一开始就在跑。</p>`);
  else if(k==='slow'){const q=runOf({...S,sch:'seq'}); {const d0=sec(runOf({...DEFAULT,sch:'seq'}).total-base.total); h+=tx(`<p>One at a time would now take ${sec(q.total)}, so batching saves ${sec(q.total-R.total)}.</p><p>With slower model calls, every model call saved is worth more: at 1.5 s per model call the saving was ${d0}.</p>`,`<p>这时一个一个来要 ${sec(q.total)}，分批省下 ${sec(q.total-R.total)}。</p><p>模型调用越慢，少一次越值钱：每次 1.5 秒时只省 ${d0}。</p>`);}}
  else h+=tx(`<p>The booking batch waits ${sec(R.total-base.total)} longer for the timeout, but the key keeps it to one ticket. Without the key the same retry books twice.</p>`,`<p>订票那一批为了超时多等了 ${sec(R.total-base.total)}，但幂等键保证只订一张票。不带键，同样的重试会订出两张。</p>`);
  return h;}

/* ---------- challenge ---------- */
const bestWords=turns=>{const nm=t=>t.map(i=>CODE.calls[i].k), j=xs=>xs.length<3?xs.join(' and '):xs.slice(0,-1).join(', ')+' and '+xs[xs.length-1];
  if(A.lang()==='zh') return turns.map((t,k)=>`${nm(t).join('、')} 放在第${'一二三四五六'[k]}次${k?'':'模型调用'}`).join('，');
  const ps=turns.map((t,k)=>k?`${j(nm(t))} in the ${['first','second','third','fourth','fifth','sixth'][k]}`:`${j(nm(t))} in the first model call`); return ps.length<2?ps[0]:ps.slice(0,-1).join(', ')+' and '+ps[ps.length-1];};
const chDone=()=>inCh()&&chSubmitted;
const chMe=()=>{const R=sched(CODE,turnsOf(chA),chOpt); return {R,turns:turnsOf(chA),total:R.total,n:R.n,right:!R.bad.length,ok:!R.bad.length&&R.total<=CH_GOAL+1e-9};};
const tnTxt=tn=>tn.map(x=>x.map(i=>CODE.calls[i].k).join('')).join(' → ');
function challengeHTML(){
  const rows=CODE.calls.map((c,i)=>A.ui.seg({id:'tls-ch-'+i,key:'ch'+i,label:L(c.n)+tx(` (${s1(c.t)} s)`,`（${s1(c.t)} 秒）`),options:[1,2,3,4,5,6].map(v=>[String(v),String(v)]),value:String(chA[i]),inline:true})).join('');
  const sub=`<div id="tls-ch-sub"${chSubmitted?' hidden':''}><button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, the task runs with your arrangement; then you see the time and whether the result is right.','提交后，按你的安排跑一遍，再告诉你耗时和结果对不对。')}</p></div>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Get a bug-fixing agent done right in at most ${CH_GOAL} s, with at most ${CH_K} tools at once`,`让修 bug 的 agent 在 ${CH_GOAL} 秒内做完、结果要对，同时最多跑 ${CH_K} 个工具`)}</h3>`+
   `<p>${tx(`A coding agent fixes a date-parsing bug (an example); the dependency graph shows which of its tool calls waits for which. The tests read both files the edits change. Your code runs at most ${CH_K} tools at once and each model call takes ${s1(CH_M)} s (the lab’s assumptions). Calls asked for in the same model call start in letter order (A first). For each call, choose the model call that asks for it; one more model call writes the answer at the end.`,`一个写代码的 agent 要修一个日期解析的 bug（示例）；依赖图里画着它的工具调用谁要等谁。跑测试要读两个改过的文件。你的代码同时最多跑 ${CH_K} 个工具，每次模型调用 ${s1(CH_M)} 秒（本 Lab 的假设）。同一次模型调用要的工具，按字母顺序开始（A 最先）。给每个调用选“第几次模型调用要它”；最后还有一次模型调用给出答案。`)}</p>`+
   `<div class="tls-chrows">${rows}</div>`+
   `<div class="goal mono" id="tls-ch-n"><span>${tx('Model calls, with the answer','模型调用次数（含给出答案的那次）')}</span><span><b id="tls-ch-nv"></b></span></div>`+
   `<div class="goal mono" id="tls-ch-t"><span>${tx('Total time','总耗时')}</span><span><b id="tls-ch-tv"></b>${tx(` (goal: at most ${CH_GOAL} s, and a right result)`,`（目标：不超过 ${CH_GOAL} 秒，结果要对）`)}</span></div><div class="progress" id="tls-ch-p" aria-hidden="true"><i></i></div>`+
   `<p class="small muted">${tx(`The ${CH_GOAL} s goal is set for this challenge in this lab.`,`${CH_GOAL} 秒是这个 Lab 为这道题定的目标。`)}</p>`+
   sub+`<div id="tls-ch-res" class="tls-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chDone()) return '';
  const all=chAll(), me=chMe(), okp=all.filter(p=>p.ok), best=okp.filter(p=>p.best)[0], myKey=me.turns.map(x=>x.join('')).join('|');
  const why=[]; if(!me.right) me.R.bad.slice(0,2).forEach(b=>why.push(badTxt(CODE,b))); if(me.total>CH_GOAL+1e-9) why.push(tx(`${sec(me.total)} is over ${CH_GOAL} s`,`${sec(me.total)}，超过 ${CH_GOAL} 秒`));
  let h;
  if(me.ok&&Math.abs(me.total-best.total)<1e-9) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${sec(me.total)}, ${me.n} model calls, the result is right.</b> Among the arrangements you can choose here, none is faster.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：${sec(me.total)}，${me.n} 次模型调用，结果对。</b>在这道题能选的安排里，没有哪种更快。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${sec(me.total)}, the result is right.</b> But putting ${bestWords(best.turns)} is faster: ${sec(best.total)} (the chart writes it ${tnTxt(best.turns)}).</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：${sec(me.total)}，结果对。</b>不过把 ${bestWords(best.turns)}，更快：${sec(best.total)}（下图写作 ${tnTxt(best.turns)}）。</p>`);
  else {h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${why.join('; ')}.</b></p><p class="small">Hint: which calls have all their inputs at the start? Which two must be done before the tests may run?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：${why.join('；')}。</b></p><p class="small">提示：一开始就有全部输入的是哪几个？跑测试之前，哪两个必须已经做完？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(me.ok) chReveal=true;
  if(!chReveal) return h;
  const init='0|1|2|3|4|5', mine=all.find(p=>p.key===myKey);
  const pick=okp.slice(); if(mine&&!mine.ok) pick.push(mine); const ini=all.find(p=>p.key===init); if(ini&&!pick.includes(ini)) pick.push(ini);
  const rows=pick.map(p=>({label:tnTxt(p.turns),value:p.total,text:tx(`${s1(p.total)} s${p.right?'':', wrong'}`,`${s1(p.total)} 秒${p.right?'':'，结果错'}`),ok:p.ok,me:p.key===myKey,best:p.best,
    tags:(p.key===init?[{text:tx('initial setup','初始设置')}]:[]).concat(p.key===myKey?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  rows.sort((a,b)=>a.value-b.value);
  const shown=new Set(pick.map(p=>p.key)).size, other=all.length-shown;
  h+=`<p class="small">${tx('In the chart, letters written together are the tools of one model call; an arrow is the next model call.','图里写在一起的字母，是同一次模型调用要的工具；箭头是下一次模型调用。')}</p>`;
  h+=A.ui.cmpChart({brief:true,title:tx(`How long does each arrangement that is right within ${CH_GOAL} s take?`,`哪些安排在 ${CH_GOAL} 秒内做对了，各要多久？`),aria:tx('Total time by arrangement.','各种安排的总耗时。'),rows,goal:CH_GOAL,goalLabel:tx(`goal: ${CH_GOAL} s`,`目标 ${CH_GOAL} 秒`),max:20,
    hiddenText:()=>tx(`The other ${A.fmt.int(other)} arrangements give a wrong result or take longer and are not shown.`,`其余 ${A.fmt.int(other)} 种安排结果错或太慢，图里没有列出。`),
    okText:tx('right, within the goal','结果对，没超时'),noText:tx('wrong or too slow','结果错或超时'),goalText:tx(`goal: ${CH_GOAL} s`,`目标：${CH_GOAL} 秒`)});
  const k0=sched(CODE,best.turns,{...chOpt,k:0}), one=sched(CODE,turnsOf([1,2,3,4,5,6]),chOpt);
  h+=tx(`<p class="small">Why: A, B and C need nothing, so they go out in the first model call; D needs A and C, E needs B, so both go out in the second; the tests must read the edited files, so F waits for the third. That is the fewest model calls that keeps every input ready. In this lab the tools of one model call start in the order A, B, C, at most ${CH_K} at once, so C waits for A, which costs ${sec(best.total-k0.total)}.</p><p class="small">A real harness could start the slowest, C, first: that batch would take ${sec(CODE.calls[2].t)}, and the whole task ${sec(k0.total)}.</p>`,
        `<p class="small">原因：A、B、C 什么都不用等，放在第一次模型调用；D 要用 A 和 C 的结果，E 要用 B 的，两个放在第二次；跑测试要读改过的文件，F 等到第三次。这是让每个调用的输入都齐了的最少模型调用次数。在这个 Lab 里，同一次模型调用要的工具按 A、B、C 的顺序开始，同时最多 ${CH_K} 个，所以 C 要等 A 做完，多花 ${sec(best.total-k0.total)}。</p><p class="small">真实的 harness 可以先发最慢的 C，这一批就只要 ${sec(CODE.calls[2].t)}，总耗时 ${sec(k0.total)}。</p>`);
  h+=tx(`<p class="small muted">Note: the times are this lab’s assumptions, fixed for every run. Real tool times vary, and a batch waits for its slowest call, so measure your own tools before planning batches.</p>`,`<p class="small muted">注意：这些时间是本 Lab 的假设，每次运行都一样。真实工具的耗时有波动，而一批要等最慢的那个，所以安排分批之前，先测一测你自己的工具。</p>`);
  return h;}

/* ---------- capability layer ---------- */
const OUTCOME=()=>tx('draw the dependency graph of an agent’s tool calls, batch the calls by it, estimate the total time, and retry safely after a timeout.','画出 agent 工具调用的依赖图，按它分批，估出总耗时，并在超时后安全地重试。');
const EXITS=()=>({
 explain:[
  {q:tx('In batches by dependency, the weather lookup becomes twice as fast and the total time does not change at all. Why?','按依赖分批时，查天气快了一倍，总耗时却一点没变。为什么？'),
   o:[tx('It is not on the critical path: in its batch the flight search is slower, and the next model call waits for that anyway.','它不在关键路径上：同一批里查航班更慢，下一次模型调用本来就要等查航班。'),
      tx('Tools that run together share the bandwidth, so one getting faster slows the others down by the same amount.','一起跑的工具共用带宽，一个变快，其余的就慢同样多。'),
      tx('Model calls take most of the time, so tool times never matter.','大部分时间花在模型调用上，所以工具快慢从来都不重要。')],a:0,
   n:[null,tx('The tools here are separate services and do not share anything; speeding the flight search does cut the total.','这里的工具是各自独立的服务，不共用什么；让查航班变快，总耗时确实会变短。'),tx('Tool times matter when the tool is on the critical path: a faster flight booking saves its whole difference.','工具在关键路径上时就很重要：订机票变快多少，总耗时就少多少。')],
   why:tx('Each model call waits for the slowest tool of the batch before it; only the calls on that chain of waits set the total.','每次模型调用都要等前一批里最慢的工具；只有这条等待链上的调用决定总耗时。')},
  {q:tx('A flight booking timed out and your code retried it. Why can that book two seats?','订机票超时了，你的代码重试了一次。为什么这可能订出两个座位？'),
   o:[tx('The timeout only means the reply did not arrive; the airline may have booked the seat, and without the same idempotency key the retry is a new booking.','超时只说明回复没到；航空公司可能已经订上了，重试不带同一个幂等键，就是一笔新订单。'),
      tx('The model sees the error and asks for the booking twice in its next reply.','模型看到错误，在下一次回复里要了两次订票。'),
      tx('A timeout cancels the first request, so the retry and a late copy both arrive.','超时会取消第一个请求，于是重试和一个迟到的副本都到了。')],a:0,
   n:[null,tx('Here your code retries by itself, before the model sees anything; the double booking needs no help from the model.','这里是你的代码自己重试，模型还什么都没看到；订出两张票不需要模型参与。'),tx('Giving up on waiting does not reach the airline: the first request went through on its side.','你的代码不再等，并不会通知航空公司：第一个请求在它那边已经做完了。')],
   why:tx('Retry a read freely; retry a write only with an idempotency key the service checks.','读的调用可以放心重试；写的调用要带一个对方会检查的幂等键再重试。')}],
 transfer:[
  {q:tx('An agent asks five suppliers for a quote; each call takes about 2 s and none needs another. Your code runs at most 3 tools at once. About how long does that batch take?','一个 agent 要向五家供应商要报价，每次调用约 2 秒，互不依赖。你的代码同时最多跑 3 个工具。这一批大约要多久？'),
   o:[tx('About 4 s','约 4 秒'),tx('About 2 s','约 2 秒'),tx('About 10 s','约 10 秒')],a:0,
   why:tx('Three run first, for about 2 s; then the other two, for as long again.','先跑三个，约 2 秒；剩下两个再跑一轮，时间一样。')},
  {q:tx('Two tool calls each add a line to the end of the same notes file: read the file, add the line, write it back. What happens if they run at the same moment?','两个工具调用都要往同一个笔记文件末尾加一行：先读文件，加一行，再写回去。如果它们同时跑，会怎样？'),
   o:[tx('The one that writes last overwrites the other’s line','后写回去的那个会盖掉另一个加的那行'),tx('Both lines end up in the file','两行都会在文件里'),tx('The file is locked and both calls fail','文件被锁住，两个调用都失败')],a:0,
   why:tx('Both read the file before either line was added, so each writes back the old file plus its own line; the later write wins. A lock on the file, or running them one after the other, keeps both lines.','两个都在任何一行加上之前读了文件，各自写回“旧文件 + 自己那行”，后写的赢。给文件加锁，或者让它们一个接一个跑，两行才都在。')}],
 practice:[{id:'agent-tool-loop',d:tx('The tool loop itself, including a model that asks for several tools in one turn and the results going back together.','工具调用循环本身，包括模型在一轮里要几个工具、结果一起发回去。')},
           {id:'agent-harness-eval',d:tx('The harness around it: timeouts, retries and idempotency keys for tools that change things.','外面那层 harness：超时、重试，以及给会改东西的工具配幂等键。')}]});

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Seven short parts. See how slow one call at a time is, predict what happens when everything goes out at once, then batch by dependency, find the critical path, set a concurrency limit and handle timeouts.','一共 7 个部分。先看一个一个来有多慢，预测一次全发会怎样，再按依赖分批、找关键路径、设并发上限，最后处理超时和重试。'),
  sandbox:tx('Pick a scenario and see how batching, a concurrency limit and a timeout change the total time and the result.','选一个场景，看看分批方式、并发上限和超时怎样改变总耗时和结果。'),
  challenge:tx(`Choose which model call asks for each of six tool calls, so that a bug fix comes out right within ${CH_GOAL} s.`,`给 6 个工具调用各选一次模型调用，让修 bug 的任务在 ${CH_GOAL} 秒内做对。`)});
const PIN={asdk:'b4b7916deaf4e5570cd627b1ae7ee4394a4de39f',opy:'ef676dbc199bc09d12a1d051f3b8b2486aa53dc2',acb:'d7265d6ae994ccd8429db0594b000073b2f9ad43',ocb:'5bd6cb40b6b84b69b2032d810610aa85064102df',cx:'4bad6d78e9b50f9fa8bd941f1db012ed491ad2da',stp:'29363094b09a67f8b1b50f577b463edbbb27ec9a'};
const GH=(repo,pin,path)=>`https://github.com/${repo}/blob/${pin}/${path}`;
const SRC=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const U_ATC=GH('anthropics/anthropic-sdk-python',PIN.asdk,'src/anthropic/types/tool_choice_auto_param.py');
const U_ACB=GH('anthropics/claude-cookbooks',PIN.acb,'tool_use/parallel_tools.ipynb');
const U_OCC=GH('openai/openai-python',PIN.opy,'src/openai/types/chat/completion_create_params.py');
const U_ORC=GH('openai/openai-python',PIN.opy,'src/openai/types/responses/response_create_params.py');
const U_OCB=GH('openai/openai-cookbook',PIN.ocb,'examples/How_to_call_functions_with_chat_models.ipynb');
const U_CX=GH('openai/codex',PIN.cx,'codex-rs/core/src/tools/parallel.rs');
const U_STP=GH('stripe/stripe-python',PIN.stp,'stripe/_api_requestor.py');
const U_OBC=GH('openai/openai-python',PIN.opy,'src/openai/_base_client.py');
const U_OCN=GH('openai/openai-python',PIN.opy,'src/openai/_constants.py');
const FIDPOP=()=>tx(`<p><b>Toy run:</b> the page schedules every call from the lab’s assumed times; it is not a recording of a real agent.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>Nine tool calls of an example trip task, each with a fixed time (the lab’s assumption), the results it needs and the trip sheet the two bookings write.</li><li>Each model call takes the same time (1.5 s unless changed) and asks for one batch; your code sends the results of a batch back together, so the next model call waits for the slowest tool.</li><li>A concurrency limit; a lock for each thing that gets changed; a ${TO}-s timeout followed by one retry; idempotency keys.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>Times never vary. Real tool times do, and since a batch waits for its slowest call, variation makes batches wait longer than the averages say.</li><li>A model call takes the same time however long the conversation is (${LL('agent-anatomy')} shows it grows).</li><li>A model call takes the same time however many tools it asks for. A real model writes out every tool_use, so a reply that asks for more takes longer, and one model call fewer saves mostly the fixed cost and the re-reading of the conversation, less than a whole call.</li><li>The lock holds the whole booking call, not only the moment it writes the trip sheet.</li><li>The model always batches as told. Real models sometimes ask for one tool at a time even when parallel calls are allowed: Anthropic’s cookbook says Claude 3.7 Sonnet may be less likely to ask for several tools in one reply, even without disable_parallel_tool_use.</li><li>What goes wrong is described, not computed: a wrong argument, a lost line, a second booking.</li></ul></div></div><p class="small">Sources: Anthropic’s ${SRC(U_ATC,'<code>disable_parallel_tool_use</code>')} and ${SRC(U_ACB,'parallel tools cookbook')}; OpenAI’s ${SRC(U_OCC,'<code>parallel_tool_calls</code>')} and ${SRC(U_OCB,'function calling cookbook')}; Stripe’s ${SRC(U_STP,'idempotency keys')}.</p>`,
  `<p><b>玩具实跑：</b>页面按本 Lab 假设的耗时，排出每个调用什么时候跑；不是真实 agent 的录像。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>一个出差任务（示例）的 9 个工具调用，每个有固定的耗时（本 Lab 的假设）、要用哪些结果，以及两个订票调用都要写的行程单。</li><li>每次模型调用的时间一样（默认 1.5 秒），每次要一批工具；你的代码把一批的结果一起发回去，所以下一次模型调用要等最慢的那个。</li><li>并发上限；每份会被改的东西各配一把锁；${TO} 秒超时后重试一次；幂等键。</li></ul></div><div><div class="eyebrow">简化了什么</div><ul><li>耗时从不波动。真实工具的耗时有波动，而一批要等最慢的那个，所以波动会让每批等得比平均值更久。</li><li>模型调用的时间不随对话变长（${LL('agent-anatomy')}里讲过它会变长）。</li><li>一次模型调用不管要几个工具，都算一样长。真实的模型要把每个 tool_use 都写出来，一次要得多，这次调用也会长一些，所以少一次模型调用省下的，主要是固定开销和重读对话的时间，比整整一次少。</li><li>锁住的是整个订票调用，不只是写行程单的那一下。</li><li>模型总是按要求分批。真实的模型即使允许并行，有时也一次只要一个工具：Anthropic 的 cookbook 说，Claude 3.7 Sonnet 即使没设 disable_parallel_tool_use，也可能较少在一次回复里要几个工具。</li><li>出错只用文字描述，不真的计算：参数猜错、行程单丢了一行、订了第二张票。</li></ul></div></div><p class="small">来源：Anthropic 的 ${SRC(U_ATC,'<code>disable_parallel_tool_use</code>')} 和 ${SRC(U_ACB,'并行工具 cookbook')}；OpenAI 的 ${SRC(U_OCC,'<code>parallel_tool_calls</code>')} 和 ${SRC(U_OCB,'function calling cookbook')}；Stripe 的 ${SRC(U_STP,'幂等键')}。</p>`);
const PRIPOP=()=>tx(`<p>An agent is a loop: your code sends the conversation and the tool descriptions to the model; the model replies with tool_use blocks (which tool, which arguments); your code runs the tools and appends their output as tool_result blocks, then calls the model again. ${LL('agent-anatomy')} takes this loop apart; ${LL('agent-reliability')} covers retries.</p>`,
                    `<p>agent 是一个循环：你的代码把对话和工具说明发给模型；模型回复 tool_use（用哪个工具、参数是什么）；你的代码执行工具，把输出作为 tool_result 接在对话后面，再调用一次模型。${LL('agent-anatomy')}把这个循环一步步拆开；${LL('agent-reliability')}讲了重试。</p>`);
function headerHTML(){return A.shell.header({id:ID,world:'agents',outcome:esc(OUTCOME()),short:tx('Tool scheduling','工具调用调度'),title:tx('Which tool calls can run in parallel?','哪些工具调用可以并行？'),
  minutes:10,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: the agent loop','前置知识：agent 循环'),primerPop:PRIPOP(),descs:DESCS(),mode});}

/* ---------- stage ---------- */
function stageHTML(){
  return `<div class="panel stage tls-stage" id="tls-stage">`+
   `<div class="stage-head tls-head"><div class="tls-st-title" id="tls-st-title"></div><div class="stage-actions"><span class="status" id="tls-status" tabindex="0"></span></div></div>`+
   `<section class="tls-pan" id="tls-pan-gantt"><div class="tls-pt" id="tls-g-title"></div><div class="tls-transport" id="tls-transport"></div>`+
     `<div class="tls-ph" id="tls-g-ph" hidden></div><svg class="tls-gantt" id="tls-gantt" role="img" aria-label=""></svg><div class="legend-row tls-leg" id="tls-g-leg"></div></section>`+
   `<section class="tls-pan" id="tls-pan-graph"><div class="tls-pt" id="tls-d-title"></div><svg class="tls-graph" id="tls-graph" role="img" aria-label=""></svg><div class="legend-row tls-leg" id="tls-d-leg"></div></section>`+
   `<div class="metrics tls-metrics" id="tls-metrics"></div>`+
   `<details class="math" id="tls-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="tls-mathbody"></div></details></div>`;}
function controlsHTML(){
  const seg=(key,label,opts,note)=>`<div class="lockable">`+A.ui.seg({id:'tls-ctl-'+key,key,label,options:opts,value:String(S[key]),note})+`</div>`;
  const times=TRIP.calls.map(c=>`<li><span>${esc(L(c.n))}</span><span class="mono">${sec(c.t)}</span></li>`).join('');
  return `<aside class="panel controls" id="tls-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint">${tx('Watch the timeline: when each call runs, and which calls wait.','看时间轴：每个调用什么时候跑，谁在等。')}</p><p class="ctl-lock" id="tls-lock" hidden></p>`+
   seg('sch',tx('How calls go out','怎么发调用'),['seq','all','dep'].map(v=>[v,L(SCH_N[v])]),tx('One at a time: one tool_use per model call. All at once: all nine in the first model call. By dependency: each model call asks for every call whose inputs are in.','一个一个来：每次模型调用一个 tool_use。一次全发：第一次模型调用就要全部 9 个。按依赖分批：每次模型调用要出所有输入已齐的调用。'))+
   seg('lock',tx('Lock on the trip sheet','行程单的锁'),[['1',tx('On','加锁')],['0',tx('Off','不加锁')]],tx('On: the two bookings never save the trip sheet at the same moment; the second waits for the first.','加锁：两个订票调用不会同时存行程单，后一个等前一个做完。'))+
   seg('k',tx('Concurrency limit k','并发上限 k'),KS.map(k=>[String(k),k?String(k):tx('No limit','不限')]),tx('The most tool calls your code runs at the same moment. Real limits come from the services’ rate limits and your own connections.','你的代码同一时刻最多跑几个工具调用。真实的上限来自被调用服务的限流和你自己的连接数。'))+
   seg('fail',tx('A call times out once','一个调用超时一次'),[['0',tx('None','没有')],['1',tx('Hotel search','查酒店')],['2',tx('Flight booking','订机票')]],tx(`The first attempt hangs; after ${TO} s your code gives up and retries once.`,`第一次尝试卡住；${TO} 秒后你的代码放弃，重试一次。`))+
   seg('idem',tx('Idempotency key on writes','写操作带幂等键'),[['1',tx('With a key','带')],['0',tx('Without','不带')]],tx('With a key, a retried booking returns the first booking instead of making a second one.','带幂等键时，重试的订票会返回第一次的订单，而不是再订一张。'))+
   `<div class="tls-assumed"><div class="ctl-title">${tx('Assumed in this lab','这个 Lab 假设的')}</div>`+
   seg('m',tx('Time of one model call','一次模型调用的时间'),MS.map(v=>[String(v),sec(v)]),tx('The lab’s assumption: a few thousand tokens in and a short reply out take about 1.5 s; a long conversation or a long reply takes longer. Here every model call takes the same time, whatever it asks for.','本 Lab 的假设：读几千个 token、回一段短回复，大约 1.5 秒；对话越长、回复越长就越慢。这里每次模型调用的时间都一样，不管这次要几个工具。'))+
   `<p class="ctl-note">${tx(`Tool times, also assumed (fixed in every run; the timeout is ${TO} s):`,`工具的耗时，也是假设（每次运行都一样；超时是 ${TO} 秒）：`)}</p><ul class="tls-times">${times}</ul></div>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE_PY=`import asyncio, contextlib

async def run_batch(uses, tools, k=3, timeout=5.0):
    """Run the tool_use blocks of one reply at the same time.
    Returns the tool_result blocks for the next model call."""
    slots = asyncio.Semaphore(k)        # at most k tools at once
    locks = {}                          # one lock per thing written

    async def one(use):
        tool = tools[use["name"]]
        lock = (locks.setdefault(tool.writes, asyncio.Lock())
                if tool.writes else contextlib.nullcontext())
        key = use["id"]                 # the same key on every retry
        async with lock:                # writers of one thing queue
            async with slots:
                for attempt in range(2):     # first try + one retry
                    try:
                        out = await asyncio.wait_for(
                            tool.run(use["input"], key), timeout)
                        return {"type": "tool_result",
                                "tool_use_id": use["id"],
                                "content": out}
                    except asyncio.TimeoutError:
                        if tool.writes and not tool.takes_key:
                            break       # a repeat could act twice
        return {"type": "tool_result", "tool_use_id": use["id"],
                "content": "timed out; it may have gone through",
                "is_error": True}

    return await asyncio.gather(*(one(u) for u in uses))`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="tls-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="tls-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('One batch of tool calls in Python: a concurrency limit, a lock for each thing that gets changed, a timeout with one retry, and an idempotency key passed to the tool.','用 Python 跑一批工具调用：并发上限、每份会被改的东西各配一把锁、超时后重试一次，并把幂等键交给工具。')}</p><pre class="code" id="tls-code">${esc(CODE_PY)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="tls-prod">`+
   tx(`<li>Anthropic’s API: ${SRC(U_ATC,'<code>disable_parallel_tool_use</code>')} “Defaults to <code>false</code>. If set to <code>true</code>, the model will output at most one tool use.” The ${SRC(U_ACB,'parallel tools cookbook')}: by default the model may “call more than one of them in a single assistant turn”; one call at a time “required ‘back and forth’”.</li>`+
      `<li>OpenAI’s API: the ${SRC(U_ORC,'Responses API’s')} <code>parallel_tool_calls</code>, “Whether to allow the model to run tool calls in parallel.”; Chat Completions has a ${SRC(U_OCC,'parameter of the same name')}. The ${SRC(U_OCB,'cookbook')}: newer models “can call multiple functions in one turn”.</li>`+
      `<li>Codex CLI, ${SRC(U_CX,'<code>tools/parallel.rs</code>')}: every tool says whether it supports parallel calls; those take a shared read lock and run together, the others take the write lock and run alone. Tools that do not say default to alone. This lab’s lock covers less than Codex’s write lock (only the calls that change the trip sheet), but it holds the whole booking call.</li>`+
      `<li>Stripe’s Python library, ${SRC(U_STP,'<code>_api_requestor.py</code>')}: “IKs should be set for all POST requests”, sent as the <code>Idempotency-Key</code> header. The OpenAI Python SDK, ${SRC(U_OBC,'<code>_base_client.py</code>')}: ${SRC(U_OCN,'2 retries')} by default, waiting 0.5 s and doubling up to 8 s, each wait cut by a random 0–25%.</li>`,
      `<li>Anthropic 的 API：${SRC(U_ATC,'<code>disable_parallel_tool_use</code>')} 默认是 <code>false</code>；设成 <code>true</code> 时，“模型最多输出一个 tool use”（原文 “the model will output at most one tool use”）。${SRC(U_ACB,'并行工具 cookbook')}：默认情况下，模型可以在一轮回复里调用不止一个工具；一次只要一个，就要“来回”好几趟（原文 “required ‘back and forth’”）。</li>`+
      `<li>OpenAI 的 API：${SRC(U_ORC,'Responses API')} 的 <code>parallel_tool_calls</code>：“是否允许模型并行调用工具”（原文 “Whether to allow the model to run tool calls in parallel.”）；Chat Completions 有${SRC(U_OCC,'同名参数')}。${SRC(U_OCB,'cookbook')}：较新的模型“能在一轮里调用多个函数”。</li>`+
      `<li>Codex CLI 的 ${SRC(U_CX,'<code>tools/parallel.rs</code>')}：每个工具声明自己是否支持并行调用；支持的拿共享的读锁、一起跑，不支持的拿写锁、单独跑；没声明的工具默认单独跑。这个 Lab 的锁只锁被改的那样东西（行程单），比 Codex 的写锁范围小；但它锁住的是整个订票调用。</li>`+
      `<li>Stripe 的 Python 库 ${SRC(U_STP,'<code>_api_requestor.py</code>')}：“每个 POST 请求都要带幂等键”（原文 “IKs should be set for all POST requests”），放在 <code>Idempotency-Key</code> 请求头里。OpenAI 的 Python SDK ${SRC(U_OBC,'<code>_base_client.py</code>')}：默认${SRC(U_OCN,'重试 2 次')}，先等 0.5 秒，每次翻倍，最多 8 秒，每次再随机缩短 0–25%。</li>`)+
   `</ul><p class="small muted">${tx('Pinned commits: anthropic-sdk-python b4b7916, claude-cookbooks d7265d6, openai-python ef676db, openai-cookbook 5bd6cb4, codex 4bad6d7, stripe-python 2936309.','固定的 commit：anthropic-sdk-python b4b7916，claude-cookbooks d7265d6，openai-python ef676db，openai-cookbook 5bd6cb4，codex 4bad6d7，stripe-python 2936309。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2312.04511" target="_blank" rel="noopener">An LLM Compiler for Parallel Function Calling</a><span>${tx('Kim et al., 2023: plan the calls as a dependency graph, run what is ready in parallel','Kim 等，2023：把调用规划成依赖图，输入齐了的并行执行')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2210.03629" target="_blank" rel="noopener">ReAct: Synergizing Reasoning and Acting in Language Models</a><span>${tx('Yao et al., 2022: the one-tool-per-step agent loop','Yao 等，2022：每步一个工具的 agent 循环')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="tls-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="tls-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="tls-app m2" id="tls-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="tls-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* which panels the stage shows now */
function ctx(){
  if(mode==='guided') return STEPS[step].panels;
  if(inCh()) return chSubmitted?['graph','gantt','metrics']:['graph','gantt'];
  return ['gantt','graph','metrics'];}

/* ---------- drawing ---------- */
const isWide=()=>{const st=app&&$('#tls-stage',app); return !st||st.clientWidth>=520;};
const clockNow=R=>anim&&anim.key===runKey()?anim.t:R.total;
const playing=()=>!!anim&&!anim.paused;
/* graph layouts, hand-placed: [x, y] of each node's top-left corner; edges that skip a column get a route */
const GL={trip:{wide:{W:600,H:180,w:116,h:36,pos:[[6,4],[6,48],[6,92],[6,136],[164,70],[322,26],[322,114],[478,26],[478,114]],route:{'3-8':[[122,152],[300,172],[462,144]]},two:true},
               narrow:{W:360,H:262,w:82,h:46,pos:[[2,4],[92,4],[184,4],[276,4],[139,72],[60,140],[220,140],[60,212],[220,212]],route:{'3-8':[[352,50],[352,196],[302,222]],'0-5':[[30,100]]}}},
          code:{wide:{W:600,H:136,w:170,h:36,pos:[[4,6],[4,50],[4,94],[215,26],[215,92],[426,59]],route:{},two:true},
               narrow:{W:360,H:192,w:108,h:46,pos:[[4,4],[126,4],[248,4],[60,74],[192,74],[126,142]],route:{}}}};
const estW=t=>[...t].reduce((a,ch)=>a+(/[\u3400-\u9fff]/.test(ch)?12:6.6),0);
function split2(t,maxW){if(estW(t)<=maxW) return [t]; const ws=t.split(' '); if(ws.length<2) return [t]; let best=null;
  for(let i=1;i<ws.length;i++){const a=ws.slice(0,i).join(' '), b=ws.slice(i).join(' '), m=Math.max(estW(a),estW(b)); if(!best||m<best[0]) best=[m,a,b];} return [best[1],best[2]];}
function nodeState(R,i,clock){const c=R.cs[i]; if(!c) return 'idle'; const b=R.bad.find(x=>x.i===i&&clock>=x.at-1e-9); if(b) return 'bad'; if(clock>=c.e-1e-9) return 'done'; if(clock>=c.s-1e-9) return 'run'; return 'idle';}
function drawGraph(){
  const task=curTask(), lay=GL[task.id][isWide()?'wide':'narrow'], {W,H,w,h,pos}=lay, svg=$('#tls-graph',app), R=curRun(), clock=clockNow(R);
  const showState=!(mode==='guided'&&['mission','deps'].includes(STEPS[step].id))&&!(inCh()&&!chSubmitted);
  const sel=mode==='guided'&&STEPS[step].id==='deps'?TRIP.calls.findIndex(c=>c.k===pick2):-1;
  const up=new Set(); if(sel>=0){const walk=i=>deps(task.calls[i]).forEach(j=>{if(!up.has(j)){up.add(j); walk(j);}}); walk(sel);}
  const cx=i=>pos[i][0]+w/2, cy=i=>pos[i][1]+h/2;
  const edgePt=(i,tx2,ty2)=>{const x=cx(i), y=cy(i), dx=tx2-x, dy=ty2-y, sx=dx?Math.abs((w/2)/dx):1e9, sy=dy?Math.abs((h/2)/dy):1e9, s=Math.min(sx,sy); return [x+dx*s,y+dy*s];};
  let g=`<defs><marker id="tls-ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="tls-ahp"/></marker></defs>`;
  task.calls.forEach((c,i)=>{deps(c).forEach(j=>{const r=lay.route[j+'-'+i]||[], first=r[0]||[cx(i),cy(i)], last=r.length?r[r.length-1]:[cx(j),cy(j)];
    const a=edgePt(j,first[0],first[1]), b=edgePt(i,last[0],last[1]), pts=[a,...r,b].map(p=>p.map(v=>v.toFixed(1)).join(',')).join(' ');
    const dashed=(c.after||[]).includes(j), hi=sel>=0&&(i===sel&&up.has(j)||up.has(i)&&up.has(j));
    g+=`<polyline class="tls-edge${dashed?' dash':''}${hi?' hi':''}" points="${pts}" marker-end="url(#tls-ah)"/>`;});});
  if(task.id==='trip'){const a=5,b=6, x=cx(a)+(isWide()?0:0), hi=sel===5||sel===6;
    if(isWide()) g+=`<line class="tls-conf${hi?' hi':''}" x1="${x}" y1="${pos[a][1]+h}" x2="${x}" y2="${pos[b][1]}"/><text class="tls-cl" x="${x-6}" y="${(pos[a][1]+h+pos[b][1])/2+4}" text-anchor="end">${esc(tx('same trip sheet','同一份行程单'))}</text>`;
    else g+=`<line class="tls-conf${hi?' hi':''}" x1="${pos[a][0]+w}" y1="${cy(a)}" x2="${pos[b][0]}" y2="${cy(b)}"/><text class="tls-cl" x="${(pos[a][0]+w+pos[b][0])/2}" y="${cy(a)-6}" text-anchor="middle">${esc(tx('same sheet','同一份行程单'))}</text>`;}
  task.calls.forEach((c,i)=>{const [x,y]=pos[i], st=showState?nodeState(R,i,clock):'idle', cls=['tls-node',st,i===sel?'sel':'',up.has(i)?'up':''].join(' ');
    const nm=L(c.n), one=isWide()&&!lay.two;
    g+=`<g class="${cls}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>`;
    if(one) g+=`<text class="tls-nt" x="${x+8}" y="${y+h/2+4}">${esc(nm)}</text><text class="tls-ns" x="${x+w-6}" y="${y+h/2+4}" text-anchor="end">${esc(sec(c.t))}</text>`;
    else {const ls=isWide()?[nm]:split2(nm,w-8), y0=ls.length>1?y+14:y+(isWide()?15:19), ys=ls.length>1?y+41:y+(isWide()?30:35);
      g+=`<text class="tls-nt" text-anchor="middle">${ls.map((l,k)=>`<tspan x="${x+w/2}" y="${y0+k*13}">${esc(l)}</tspan>`).join('')}</text><text class="tls-ns" x="${x+w/2}" y="${ys}" text-anchor="middle">${esc(sec(c.t))}</text>`;}
    g+=`</g>`;});
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.innerHTML=g;
  svg.setAttribute('aria-label',tx('Dependency graph: an arrow points from a call to a call that waits for it.','依赖图：箭头从一个调用指向等它的调用。'));
  $('#tls-d-title',app).textContent=task.id==='trip'?tx('Which call waits for which? (the trip task, an example)','哪个调用要等哪个？（出差任务，示例）'):tx('Which call waits for which? (the bug fix, an example)','哪个调用要等哪个？（修 bug 的任务，示例）');
  const leg=$('#tls-d-leg',app), lk=A.lang()+task.id+showState; if(leg._k!==lk){leg._k=lk;
    const items=[`<span class="lg" data-tipx="${esc(tx('A solid arrow points from a call to a call that needs its result.','实线箭头从一个调用指向要用它结果的调用。'))}"><i class="sw tls-sw-arr"></i>${tx('needs its result','要用它的结果')}</span>`];
    if(task.id==='trip') items.push(`<span class="lg" data-tipx="${esc(D_SHEET())}"><i class="sw tls-sw-conf"></i>${tx('write the same trip sheet','改同一份行程单')}</span>`);
    else items.push(`<span class="lg" data-tipx="${esc(tx('A dashed arrow points from an edit to the tests that must read the edited file.','虚线箭头从改文件的调用指向要读改过的文件的测试。'))}"><i class="sw tls-sw-dash"></i>${tx('must see its changes','要读它改过的文件')}</span>`);
    if(showState) items.push(`<span class="lg" data-tipx="${esc(tx('A call that is running now in the timeline.','时间轴上正在跑的调用。'))}"><i class="sw tls-sw-run"></i>${tx('running','在跑')}</span><span class="lg" data-tipx="${esc(tx('A call that has finished.','已经做完的调用。'))}"><i class="sw tls-sw-done"></i>${tx('done','做完了')}</span><span class="lg" data-tipx="${esc(tx('A call whose result is wrong, from the moment it goes wrong.','结果错了的调用，从出错的那一刻起标红。'))}"><i class="sw tls-sw-bad"></i>${tx('wrong','出错')}</span>`);
    leg.innerHTML=items.join('');}}
function axisMax(R){const base=inCh()?17.1:seqTotal(S.m); return Math.ceil(Math.max(base,R.total)/5)*5;}
function drawGantt(){
  const svg=$('#tls-gantt',app), ph=$('#tls-g-ph',app), tEl=$('#tls-transport',app);
  if(inCh()&&!chSubmitted){svg.style.display='none'; tEl.hidden=true; ph.hidden=false; ph.textContent=tx('Your arrangement runs here once you submit it.','提交以后，你的安排会在这里跑一遍。'); $('#tls-g-leg',app).innerHTML=''; $('#tls-g-title',app).textContent=tx('When does each call run?','每个调用什么时候跑？'); return;}
  svg.style.display=''; tEl.hidden=false; ph.hidden=true;
  const task=curTask(), R=curRun(), wide=isWide(), W=wide?600:360, LW=wide?124:96, X0=LW+6, X1=W-10, RH=wide?19:20, n=task.calls.length, top=8, H=top+RH*(n+1)+30;
  const T=axisMax(R), X=t=>X0+(X1-X0)*t/T, clock=clockNow(R), Y=r=>top+r*RH;
  let g=`<defs><pattern id="tls-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" class="tls-hbg"/><line x1="0" y1="0" x2="0" y2="5" class="tls-hl"/></pattern></defs>`;
  for(let s=0;s<=T+1e-9;s+=5) g+=`<line class="tls-grid" x1="${X(s)}" y1="${top}" x2="${X(s)}" y2="${Y(n+1)}"/><text class="tls-ax" x="${X(s)}" y="${Y(n+1)+13}" text-anchor="middle">${s}</text>`;
  g+=`<text class="tls-ax" x="${X1}" y="${Y(n+1)+26}" text-anchor="end">${esc(tx('time (s)','时间（秒）'))}</text>`;
  g+=`<text class="tls-rl b" x="${LW}" y="${Y(0)+RH/2+4}" text-anchor="end">${esc(tx('Model','模型'))}</text>`;
  task.calls.forEach((c,i)=>{const nm=L(c.n); g+=`<text class="tls-rl" x="${LW}" y="${Y(i+1)+RH/2+4}" text-anchor="end">${esc(!wide&&nm.length>13?nm.slice(0,12)+'…':nm)}</text>`;});
  const bh=RH-7, clip=(s,e)=>Math.max(0,Math.min(e,clock)-s);
  const mcs=R.turns.map(t=>[t.ms,t.me]).concat([[R.turns.length?R.turns[R.turns.length-1].end:0,R.total]]);
  mcs.forEach(([s,e],j)=>{const wv=clip(s,e); if(wv<=0) return; const x=X(s), ww=X(s+wv)-x;
    g+=`<rect class="tls-mc" x="${x}" y="${Y(0)+3}" width="${Math.max(ww,1)}" height="${bh}" rx="2"/>`;
    if(X(e)-x>=13&&wv>=e-s-1e-9) g+=`<text class="tls-mcn" x="${x+(X(e)-x)/2}" y="${Y(0)+3+bh/2+4}" text-anchor="middle">${j+1}</text>`;});
  task.calls.forEach((c,i)=>{const z=R.cs[i]; if(!z) return; const y=Y(i+1)+3, tn=R.turns[z.turn], badB=R.bad.find(b=>b.i===i), isBad=badB&&clock>=badB.at-1e-9, cr=R.crit.has(i)&&!anim;
    if(z.s>tn.me+1e-9&&clock>tn.me) g+=`<line class="tls-wait" x1="${X(tn.me)}" y1="${y+bh/2}" x2="${X(Math.min(z.s,clock))}" y2="${y+bh/2}"/>`;
    if(z.hang!=null){const w1=clip(z.s,z.hang); if(w1>0) g+=`<rect class="tls-hang" x="${X(z.s)}" y="${y}" width="${X(z.s+w1)-X(z.s)}" height="${bh}" rx="2"/>`;
      const w2=clip(z.hang,z.e); if(w2>0) g+=`<rect class="tls-bar${isBad?' bad':''}${cr?' crit':''}" x="${X(z.hang)}" y="${y}" width="${Math.max(1,X(z.hang+w2)-X(z.hang))}" height="${bh}" rx="2"/>`;}
    else{const wv=clip(z.s,z.e); if(wv>0) g+=`<rect class="tls-bar${isBad?' bad':''}${cr?' crit':''}" x="${X(z.s)}" y="${y}" width="${Math.max(1,X(z.s+wv)-X(z.s))}" height="${bh}" rx="2"/>`;}
    if(isBad) g+=`<text class="tls-x" x="${X(z.e)+4}" y="${y+bh/2+4}">✗</text>`;});
  if(anim&&anim.key===runKey()) g+=`<line class="tls-head-l" x1="${X(clock)}" y1="${top}" x2="${X(clock)}" y2="${Y(n+1)}"/>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.innerHTML=g;
  svg.setAttribute('aria-label',tx(`Timeline: ${R.n} model calls, ${s1(R.total)} s in all.`,`时间轴：${R.n} 次模型调用，一共 ${s1(R.total)} 秒。`));
  $('#tls-g-title',app).textContent=tx('When does each call run? (one row per tool call; the top row is the model)','每个调用什么时候跑？（每行一个工具调用，最上面一行是模型）');
  const leg=$('#tls-g-leg',app), hasW=R.cs.some((z,i)=>z&&z.s>R.turns[z.turn].me+1e-9), hasH=R.cs.some(z=>z&&z.hang!=null), lk=[A.lang(),hasW,hasH,R.bad.length>0].join();
  if(leg._k!==lk){leg._k=lk;
    leg.innerHTML=`<span class="lg" data-tipx="${esc(tx('A model call: your code sends the conversation and waits for the reply. The number counts the model calls.','模型调用：你的代码把对话发出去，等模型回复。数字是第几次模型调用。'))}"><i class="sw tls-sw-mc"></i>${tx('model call','模型调用')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('A tool call your code runs, from start to finish.','你的代码执行的一个工具调用，从开始到结束。'))}"><i class="sw tls-sw-bar"></i>${tx('tool call','工具调用')}</span>`+
     `<span class="lg" data-tipx="${esc(D_CRIT())}"><i class="sw tls-sw-crit"></i>${tx('critical path','关键路径')}</span>`+
     (hasW?`<span class="lg" data-tipx="${esc(tx('The call was asked for but waits: for the lock on the trip sheet, or for a free slot under the concurrency limit.','这个调用已经要了，但在等：等行程单的锁，或者等并发上限下空出一个位置。'))}"><i class="sw tls-sw-wait"></i>${tx('waiting to start','等着开始')}</span>`:'')+
     (hasH?`<span class="lg" data-tipx="${esc(D_TO())}"><i class="sw tls-sw-hang"></i>${tx('hung attempt, until the timeout','卡住的尝试，直到超时')}</span>`:'')+
     (R.bad.length?`<span class="lg" data-tipx="${esc(tx('The call whose result is wrong turns red at the moment it goes wrong.','结果错了的调用，在出错的那一刻变红。'))}"><i class="sw tls-sw-bad"></i>${tx('goes wrong','出错')}</span>`:'')+
     `<span class="lg" data-tipx="${esc(tx('While the run plays, a vertical line marks the current moment.','播放时，竖线标出现在播到的时刻。'))}"><i class="sw tls-sw-head"></i>${tx('now (while playing)','现在（播放时）')}</span>`;}}
function drawTransport(){
  const tEl=$('#tls-transport',app); if(tEl.hidden) return; const want=A.lang()+'|'+(playing()?1:0)+'|'+speed;
  if(tEl._k!==want){tEl._k=want;
    tEl.innerHTML=`<button type="button" class="btn sm" data-act="play" data-tipx="${esc(tx('Play the run; press again to pause.','播放这次运行；再按一次暂停。'))}">${playing()?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Play','▶ 播放')}</button>`+
      A.ui.seg({id:'tls-speed',key:'speed',label:tx('Speed','速度'),options:['slow','mid','fast'].map(k=>[k,L(SPEED_N[k])]),value:speed})+
      `<button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch this run again from the start.','从头再看一遍这次运行。'))}">${tx('Replay','重放')}</button>`+
      A.ui.range({id:'tls-tr-p',key:'prog',label:tx('Progress','进度'),min:0,max:100,value:100,out:'100%',inline:true});
    A.ui.initRanges(tEl);}
  const R=curRun(), i=$('#tls-tr-p-in',app), p=Math.round((anim&&anim.key===runKey()?anim.t/R.total:1)*100); if(i&&+i.value!==p){i.value=p; A.ui.setRange(i);} const o=i&&i.closest('.inline-ctl,.ctl'), out=o&&o.querySelector('output'); if(out) out.textContent=p+'%';}
const MET=[
 {k:'tot',label:()=>tx('Total time','总耗时'),tip:()=>tx('From the task to the answer: every model call plus every batch of tools, as the timeline shows.','从收到任务到给出答案：所有模型调用，加上每一批工具，就是时间轴的长度。')},
 {k:'n',label:()=>tx('Model calls','模型调用'),tip:()=>tx('How many times your code calls the model, the last call (the answer) included.','你的代码调用了几次模型，包括最后给出答案的那次。')},
 {k:'tools',label:()=>tx('Tool time added up','工具时间合计'),tip:()=>tx('The time of every tool call added up, hung attempts included. Calls that run together each count in full, so this sum can be more than the time those calls take on the timeline.','每个工具调用的时间加起来，卡住的尝试也算。一起跑的调用各算各的，所以这个合计可以比这些调用在时间轴上占的时间长。')},
 {k:'res',label:()=>tx('Result','结果'),tip:()=>tx('Right when every call had the results it needs, no two calls saved the same thing at once, and nothing was done twice.','每个调用都拿到了要用的结果、没有两个调用同时存同一样东西、没有哪件事做了两次，结果才算对。')}];
function drawMetrics(){
  const R=curRun(), host=$('#tls-metrics',app), clock=clockNow(R), live=clock<R.total-1e-9, key=runKey(), cfgCh=metKey!=null&&metKey!==key; metKey=key;
  if(!host.children.length||host._lang!==A.lang()){host.innerHTML=MET.map(x=>`<div class="metric" data-k="${x.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); host._lang=A.lang();}
  const nNow=live?R.turns.filter(t=>t.ms<=clock+1e-9).length+(clock>=(R.turns.length?R.turns[R.turns.length-1].end:0)?1:0):R.n;
  const toolNow=live?R.cs.reduce((a,z)=>a+(z?Math.max(0,Math.min(z.e,clock)-z.s):0),0):R.toolSum, badNow=R.bad.filter(b=>clock>=b.at-1e-9);
  const vals={tot:live?sec(clock):sec(R.total),n:String(nNow),tools:sec(toolNow),res:live?(badNow.length?tx('Wrong','错'):tx('Running','还在跑')):(R.bad.length?tx('Wrong','错'):tx('Right','对'))};
  const subs={tot:live?tx('so far','到现在'):tx('whole run','整次运行'),n:live?tx('so far','到现在'):tx(`${R.turns.length} ask for tools, 1 gives the answer`,`${R.turns.length} 次要工具，1 次给出答案`),tools:live?tx('so far','到现在'):tx(`${task().calls.length} calls`,`${task().calls.length} 个调用`),
    res:R.bad.length&&(!live||badNow.length)?tx(`${new Set(R.bad.map(b=>b.i)).size} call${new Set(R.bad.map(b=>b.i)).size>1?'s':''} wrong`,`${new Set(R.bad.map(b=>b.i)).size} 个调用出错`):(live?'':tx('every call right','每个调用都对'))};
  MET.forEach(x=>{const el=host.querySelector(`[data-k="${x.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=x.label(); el.querySelector('.v').textContent=vals[x.k]; el.querySelector('.sub').textContent=subs[x.k]; el.dataset.tipx=x.tip();
    el.classList.toggle('tls-bad',x.k==='res'&&vals.res===tx('Wrong','错'));
    let dt=''; const num={tot:Math.round(R.total*10)/10,n:R.n,tools:Math.round(R.toolSum*10)/10}[x.k];
    if(!live&&userAct&&cfgCh&&num!=null&&shownPrev[x.k]!=null&&shownPrev[x.k]!==num) dt=A.fmt.delta(shownPrev[x.k],num,{dp:x.k==='n'?0:1});
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2800);} else if(!live){d.classList.remove('on'); d.textContent='';}
    if(num!=null&&(!live||shownPrev[x.k]==null)) shownPrev[x.k]=num;});}
const task=()=>curTask();
function drawHead(){
  const st=$('#tls-st-title',app), el=$('#tls-status',app), R=curRun(), live=anim&&anim.key===runKey();
  if(inCh()){st.textContent=chSubmitted?tx(`${live?'Playing':'Played'}: the bug fix with your arrangement, ${tnTxt(turnsOf(chA))}`,`${live?'正在播放':'播放完'}：按你的安排修 bug，${tnTxt(turnsOf(chA))}`):tx('The bug fix (an example): arrange the six calls','修 bug 的任务（示例）：安排 6 个调用');
    const me=chMe(), done=chDone(); el.hidden=false; el.className='status'+(done?(me.ok?' ok':' bad'):''); el.textContent=done?(me.ok?tx('Goal met','达标'):tx('Not met','没达标')):tx('Not submitted','还没提交');
    el.dataset.tipx=tx(`Whether your arrangement is right and takes at most ${CH_GOAL} s. It shows once you submit.`,`你的安排是否结果对、而且不超过 ${CH_GOAL} 秒。提交后才显示。`); return;}
  const gOnly=mode==='guided'&&STEPS[step].id==='deps';
  st.textContent=gOnly?tx('The trip task (an example): who waits for whom','出差任务（示例）：谁要等谁'):tx(`${live?'Playing':'Played'}: the trip task, ${setTxt(S).replace(/^./,x=>x.toLowerCase())}`,`${live?'正在播放':'播放完'}：出差任务，${setTxt(S)}`);
  const hide=gOnly||(mode==='guided'&&STEPS[step].id==='mission'&&!pred);
  el.hidden=hide; if(hide) return;
  el.className='status'+(live?'':(R.bad.length?' bad':' ok'));
  el.textContent=live?tx('Playing','播放中'):tx(`${s1(R.total)} s, ${resTxt(R)}`,`${s1(R.total)} 秒，结果${resTxt(R)}`);
  el.dataset.tipx=tx('The total time of the run and whether its result is right.','这次运行的总耗时，以及结果对不对。');}
function panelsShow(){const p=ctx(); ['gantt','graph'].forEach(x=>{$('#tls-pan-'+x,app).hidden=!p.includes(x);});
  $('#tls-metrics',app).hidden=!p.includes('metrics');
  const gEl=$('#tls-pan-gantt',app), dEl=$('#tls-pan-graph',app), gFirst=p.indexOf('gantt')>=0&&p.indexOf('gantt')<p.indexOf('graph');
  if(gFirst&&gEl.nextElementSibling!==dEl) dEl.parentNode.insertBefore(gEl,dEl); else if(!gFirst&&dEl.nextElementSibling!==gEl) gEl.parentNode.insertBefore(dEl,gEl);}
function mathHTML(){
  const task=curTask(), R=curRun(), blk=(lead,rows)=>`<div class="tls-mb"><p class="tls-mb-lead">${lead}</p><div class="tls-eq">${rows.map(([a,b])=>`<div class="tls-eq-l">${a?MI(a):''}</div><div class="tls-eq-r">${MI(b)}</div>`).join('')}</div></div>`;
  const ws=R.turns.map(t=>r3(t.end-t.me)), nm=tx('total time','总耗时'), U=`\\ \\text{${tx('s','秒')}}`;
  let h=blk(tx(`The total time: every model call, plus each batch from its start until its last tool is done. ${MI('n')} is the number of model calls, ${MI('m')} the time of one model call and ${MI('w_j')} the time batch ${MI('j')} takes.`,`总耗时：每次模型调用的时间，加上每一批从开始到最后一个工具做完的时间。${MI('n')} 是模型调用的次数，${MI('m')} 是一次模型调用的时间，${MI('w_j')} 是第 ${MI('j')} 批花的时间。`),
    [[`\\text{${nm}}`,`= n \\cdot m + \\sum_{j=1}^{${R.turns.length}} w_j`],...(ws.length>5?[['',`= ${R.n} \\times ${s1(R.m)} + (${ws.slice(0,5).map(s1).join(' + ')}`],['',`\\quad + ${ws.slice(5).map(s1).join(' + ')})`]]:[['',`= ${R.n} \\times ${s1(R.m)} + (${ws.map(s1).join(' + ')})`]]),['',`= ${s1(R.n*R.m)} + ${s1(r3(ws.reduce((a,b)=>a+b,0)))} = ${s1(R.total)}${U}`]]);
  /* a batch longer than its slowest tool: name the queue and add it up */
  const q2=i=>task===CODE?tx(`“${CN(task,i)}”`,`“${CN(task,i)}”`):CN(task,i), kk=inCh()?CH_K:S.k; let noted=0;
  R.turns.forEach((tn,j)=>{if(!tn.calls.length) return; let cur=tn.calls.reduce((a,b)=>R.cs[b].e>R.cs[a].e+1e-9?b:a); const chain=[cur];
    while(R.cs[cur].s>tn.me+1e-9){const p=tn.calls.find(q=>q!==cur&&Math.abs(R.cs[q].e-R.cs[cur].s)<1e-9); if(p==null) break; chain.unshift(p); cur=p;}
    if(chain.length<2||R.cs[chain[0]].s>tn.me+1e-9) return;
    const ds=chain.map(i=>r3(R.cs[i].e-R.cs[i].s)), a=chain[chain.length-2], b=chain[chain.length-1], lk=!!task.calls[b].w&&task.calls[a].w===task.calls[b].w;
    const why=lk?tx(`${q2(b)} waits for ${q2(a)} to finish and free the lock on the trip sheet`,`${q2(b)}要等${q2(a)}做完、放开行程单的锁才开始`)
                :tx(`${kk===1?'only one tool runs at a time':`at most ${kk} tools run at once`}, so ${listJ(chain.map(q2))} run one after the other`,`同时最多 ${kk} 个，所以${listJ(chain.map(q2))}只能一个接一个地跑`);
    h+=blk(tx(`Batch ${j+1} takes longer than its slowest tool: ${why}.`,zc(`第 ${j+1} 批比它最慢的工具还长：`,why,'。')),[[`w_{${j+1}}`,`= ${ds.map(s1).join(' + ')} = ${s1(ws[j])}${U}`]]); noted++;});
  if(!noted&&ws.length>1){const jx=ws.indexOf(Math.max(...ws)), tn=R.turns[jx], last=tn.calls.reduce((a,b)=>R.cs[b].e>R.cs[a].e+1e-9?b:a);
    h+=`<div class="tls-mb"><p class="tls-mb-lead">${tx(`The longest batch is batch ${jx+1}: ${q2(last)} finishes last, ${sec(ws[jx])} after the batch starts.`,zc(`最长的是第 ${jx+1} 批：`,q2(last),`最后做完，在这一批开始后 ${sec(ws[jx])}。`))}</p></div>`;}
  if(!inCh()&&S.sch!=='seq'){const q=runOf({...S,sch:'seq'});
    h+=blk(tx('One at a time, for comparison: each batch is one tool, so every tool time is added.','对比一个一个来：每批只有一个工具，所以每个工具的时间都要加上。'),[[`\\text{${nm}}`,`= ${q.n} \\times ${s1(q.m)} + ${s1(q.toolSum)}`],['',`= ${s1(q.total)}${U}`]]);}
  return h;}
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(gEl=>{const k=gEl.dataset.key;
    if(k==='speed') A.ui.setPressed(gEl,speed); else if(k==='pick2') A.ui.setPressed(gEl,pick2); else if(/^ch\d$/.test(k)) A.ui.setPressed(gEl,String(chA[+k.slice(2)])); else if(S[k]!=null) A.ui.setPressed(gEl,String(S[k]));});
  $$('#tls-rail .levers button[data-lever]',app).forEach(b=>{const id=STEPS[step]&&STEPS[step].id, v=b.dataset.lever, on=id==='batch'?v===lv3Now():id==='crit'?v===String(S.half):id==='fail'?v===lv6Now():false; b.setAttribute('aria-pressed',String(on)); const rr=b.querySelector('.lv-r'); if(rr){const t=lvResTxt(id,v); if(rr.textContent!==t) rr.textContent=t;}});
  const lock=inCh(), c=$('#tls-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#tls-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx(`In the challenge you choose in the task card when each call goes out. Fixed: at most ${CH_K} tools at once, started in letter order, ${s1(CH_M)} s per model call, a lock on files, no timeouts.`,`挑战中，你在任务卡里安排每个调用什么时候发。固定的：同时最多 ${CH_K} 个工具，按字母顺序开始，每次模型调用 ${s1(CH_M)} 秒，改文件加锁，没有超时。`);
  $$('#tls-controls .lockable, #tls-controls .tls-assumed',app).forEach(e=>{e.hidden=lock;});}

/* ---------- rail ---------- */
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(){
  const rail=$('#tls-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const live=anim&&anim.key===runKey()&&ctx().includes('gantt'), show=!!s.obs&&(s.obsAt?s.obsAt():(!s.expect||!!observed[step]))&&(!live||!!(box&&box._h));
    if(box){if(!live||!box._h){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; box.hidden=false; if(again) A.ui.obsBump(box);}}}
      if(live&&s.id!=='deps'&&s.id!=='mission'&&box._h) box.classList.add('tls-dim'); else box.classList.remove('tls-dim');}
    const nw=$('[data-now]',rail); if(nw){const h=nowHTML(); if(nw._h!==h){nw.innerHTML=h; nw._h=h;} nw.hidden=!h;}
    A.ui.setNext(rail,isNextDisabled());
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#tls-insight',rail); if(ins){const html=`<div class="eyebrow">${c>=0?esc(L(PRESETS[c].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const me=chMe(), done=chDone(), nv=$('#tls-ch-nv',rail); if(!nv) return;
    nv.textContent=String(me.n);
    $('#tls-ch-tv',rail).textContent=done?sec(me.total):tx('shown after you submit','提交后才知道');
    $('#tls-ch-p i',rail).style.width=(done?Math.min(100,CH_GOAL/Math.max(me.total,CH_GOAL)*100):0)+'%'; $('#tls-ch-t',rail).classList.toggle('met',done&&me.ok); $('#tls-ch-p',rail).classList.toggle('met',done&&me.ok);
    const sub=$('#tls-ch-sub',rail); if(sub) sub.hidden=chSubmitted;
    const res=$('#tls-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body()+'<div class="obs tls-now" data-now hidden></div>',back:step>0,showMe:false,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:s.nextWhy?L(s.nextWhy):undefined,skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#tls-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(inCh()&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  panelsShow(); maybePlay(); drawHead();
  if(!$('#tls-pan-gantt',app).hidden){drawTransport(); drawGantt();}
  if(!$('#tls-pan-graph',app).hidden) drawGraph();
  if(!$('#tls-metrics',app).hidden) drawMetrics();
  const mb=$('#tls-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  $('#tls-math',app).hidden=mode==='guided'&&['mission','deps'].includes(STEPS[step].id)||inCh()&&!chSubmitted;
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();} // the kit drops the settings from the address outside Scenarios, so a reload in another mode is not taken for a shared link
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const R=curRun(); A.announce(tx(`${R.n} model calls, ${s1(R.total)} seconds, result ${resTxt(R)}.`,`${R.n} 次模型调用，${s1(R.total)} 秒，结果${resTxt(R)}。`));},400);}

/* ---------- the run player (R3.9, R3.20): a run plays only when its picture has not been watched; a watched one shows at once ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} anim=null; lastTs=0;}
function wantsGantt(){return ctx().includes('gantt')&&!(inCh()&&!chSubmitted);}
function maybePlay(){if(!wantsGantt()){if(anim) stopAnim(); return;} const k=runKey(); if(anim&&anim.key===k) return; if(anim) stopAnim(); if(seen.has(k)) return; startRun(0);}
function startRun(t0){stopAnim(); const k=runKey(); seen.add(k);
  if(A.reduceMotion()) return;
  anim={key:k,t:t0||0,paused:false}; lastTs=0; raf=requestAnimationFrame(tick);}
function tick(ts){raf=0; if(!anim||anim.paused||!app) return;
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts; const R=curRun();
  anim.t=Math.min(R.total,anim.t+dt/RATE[speed]);
  if(anim.t>=R.total-1e-9){anim=null; render(); return;}
  drawTransport(); drawGantt(); if(!$('#tls-pan-graph',app).hidden) drawGraph(); if(!$('#tls-metrics',app).hidden) drawMetrics(); drawHead(); A.dock.refresh();
  raf=requestAnimationFrame(tick);}

/* ---------- navigation ---------- */
function enterStep(i){
  stopAnim(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter());
  tried.clear(); acted=false; if(s.id==='deps') pick2=''; else tried.add(tryKey());
  renderRail();}
function tryKey(){const id=mode==='guided'?STEPS[step].id:''; return id==='batch'?lv3Now():id==='crit'?String(S.half):id==='fail'?lv6Now():id==='limit'?String(S.k):id==='deps'?pick2:JSON.stringify(S);}
function revealStep(){A.ui.reveal($('#tls-rail .step.current',app)||$('#tls-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, rc=box.getBoundingClientRect();
  if(rc.bottom<=innerHeight-8) return;
  const dy=Math.min(rc.bottom-innerHeight+16,rc.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>requestAnimationFrame(()=>{if(!app) return; if(mode==='guided') revealBox($('#tls-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#tls-insight',app)); else revealBox($('#tls-ch-res',app));});
function setMode(m){
  if(m===mode) return; stopAnim();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chA=[1,2,3,4,5,6];}
  if(m==='sandbox'&&was==='guided') S={...DEFAULT};
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
function applyPreset(i){Object.assign(S,PRESETS[i].p); byLearner(render); revealObs();}
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
const token=()=>[S.sch,S.lock,S.k,S.fail,S.idem,String(S.m).replace('.','p')].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==6) return null; const o={sch:p[0],lock:+p[1],k:+p[2],fail:+p[3],idem:+p[4],m:+p[5].replace('p','.'),half:-1};
  if(!SCH_N[o.sch]||![0,1].includes(o.lock)||!KS.includes(o.k)||![0,1,2].includes(o.fail)||![0,1].includes(o.idem)||!MS.includes(o.m)) return null; return o;}
function act(a,btn){
  if(a==='play'){if(playing()){anim.paused=true; if(raf){cancelAnimationFrame(raf); raf=0;}} else if(anim){anim.paused=false; lastTs=0; raf=requestAnimationFrame(tick);} else startRun(0); render(); return;}
  if(a==='replay'){startRun(0); render(); return;}
  if(a==='reset'){stopAnim(); byLearner(()=>{Object.assign(S,mode==='guided'?partBase():DEFAULT); render();}); return;}   // in a guided part: the setup its card explains (R09)
  if(a==='card'){stopAnim(); byLearner(()=>{Object.assign(S,cardS()); render();}); return;}
  if(a==='submit'){chSubmitted=true; render(); revealObs(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#tls-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}

/* ---------- interview drill (seeded): total time of a batched agent run ---------- */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const rnd=mulberry(seed*7919+13), m=[1,1.5,2][Math.floor(rnd()*3)], nb=2+Math.floor(rnd()*2);
  const batches=[]; for(let j=0;j<nb;j++){const c=1+Math.floor(rnd()*3), b=[]; for(let q=0;q<c;q++) b.push((1+Math.floor(rnd()*30))/10); batches.push(b);}
  const mx=batches.map(b=>Math.max(...b)), total=r3((nb+1)*m+mx.reduce((a,b)=>a+b,0)), seqT=r3((batches.flat().length+1)*m+batches.flat().reduce((a,b)=>a+b,0));
  return {seed,m,batches,mx,total,seqT};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#tls-drill-q',app), a=$('#tls-drill-a',app);
  const list=d.batches.map((b,j)=>b.length>1?tx(`model call ${j+1} asks for ${b.length} tools taking ${b.map(s1).join(', ')} s`,`第 ${j+1} 次模型调用要 ${b.length} 个工具，分别要 ${b.map(s1).join('、')} 秒`):tx(`model call ${j+1} asks for 1 tool taking ${s1(b[0])} s`,`第 ${j+1} 次模型调用要 1 个工具，要 ${s1(b[0])} 秒`)).join(tx('; ','；'));
  q.innerHTML=tx(`Each model call takes ${s1(d.m)} s; your code runs a reply’s tools all at once. ${esc(list)}; one more model call answers. How long does the task take, and how long one tool per model call?`,`每次模型调用要 ${s1(d.m)} 秒；你的代码同时执行一次回复里的全部工具。${esc(list)}；最后还有一次模型调用给出答案。整个任务要多久？每次模型调用只要一个工具的话，又要多久？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const n=d.batches.length+1, all=d.batches.flat();
  a.innerHTML=tx(`<p><b>${sec(d.total)}; one at a time ${sec(d.seqT)}</b></p>`,`<p><b>${sec(d.total)}；一个一个来 ${sec(d.seqT)}</b></p>`)+
    M(`\\text{${tx('total, batched','分批时的总耗时')}} = ${n} \\times ${s1(d.m)} + (${d.mx.map(s1).join(' + ')}) = ${s1(d.total)}\\ \\text{${tx('s','秒')}}`)+tx('<p class="small">The brackets hold the slowest tool of each batch.</p>','<p class="small">括号里是每一批最慢的那个工具。</p>')+M(`\\text{${tx('total, one at a time','一个一个来的总耗时')}} = ${all.length+1} \\times ${s1(d.m)} + ${s1(r3(all.reduce((x,y)=>x+y,0)))} = ${s1(d.seqT)}\\ \\text{${tx('s','秒')}}`)+
    tx(`<p>Batched, each model call waits only for the slowest tool of its batch; one at a time, every tool time adds up and every tool costs a model call of its own.</p>`,`<p>分批时，每次模型调用只等这一批里最慢的工具；一个一个来时，每个工具的时间都要加上，每个工具还要多一次模型调用。</p>`);}

/* ---------- jumps ---------- */
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  return '';}
function jump(k){
  if(!app) return;
  if(k==='controls'){const c=$('#tls-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c);}}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tickS=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tickS,120); return;}
    const st=$('#tls-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#tls-rail .step.current',app)||$('#tls-rail .railcard',app)||$('#tls-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, rc=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=rc.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tickS,A.reduceMotion()?30:150);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='prog'){const R=curRun(), p=+e.target.value/100, key=runKey(); stopAnim(); seen.add(key); if(p<1) anim={key,t:p*R.total,paused:true}; render();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#tls-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!seg.closest('#tls-rail');
      if(k==='speed'){speed=d.v; drawTransport(); syncControls(); return;}
      if(k==='pick2'){pick2=d.v; tried.add(pick2); render(); revealObs(); return;}
      if(/^ch\d$/.test(k)){chA[+k.slice(2)]=+d.v; if(chSubmitted&&!chReveal) chSubmitted=false; byLearner(render); return;}
      const v=k==='sch'?d.v:+d.v;
      byLearner(()=>{S[k]=v; if(mode==='guided'){tried.add(tryKey()); if(inRail) acted=true;} render();});
      if(inRail) revealObs(); return;}
    if(d.lever!=null){const id=STEPS[step].id, v=d.lever;
      const p=id==='batch'?LV3.find(o=>o.v===v).p:id==='crit'?{half:+v}:id==='fail'?LV6.find(o=>o.v===v).p:{};
      acted=true; byLearner(()=>{Object.assign(S,p); tried.add(tryKey()); render();}); revealObs(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#tls-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#tls-rail',app)); return;}
    if(d.act) act(d.act,b);});}
/* the lab state contract (round 3 #6): what the learner did here, saved by the kit and put back once per page load before mount */
let restored=false;
const sOK=o=>{if(!o||typeof o!=='object') return null; const p=parse([o.sch,o.lock,o.k,o.fail,o.idem,String(o.m).replace('.','p')].join('_')); if(!p) return null;
  p.half=Number.isInteger(o.half)&&o.half>=-1&&o.half<TRIP.calls.length?o.half:-1; return p;};
function snap(){const sn=[...seen]; return {v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,pick2,tried:[...tried],acted,
  chA:chA.slice(),chSubmitted,chReveal,speed,drillSeed,drillOpen,seen:sn.slice(-12),passed:mode==='challenge'&&chSubmitted&&chMe().ok};}
function restore(o){if(!o||o.v!==1) return; const s=sOK(o.S); if(!s) return;
  mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; S=s;
  step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.includes(o.pred)?o.pred:null;
  observed={}; if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
  const sv=sOK(o.saved); saved=mode==='challenge'&&sv?sv:null;
  pick2=P2C.some(i=>TRIP.calls[i].k===o.pick2)?o.pick2:'';
  tried.clear(); if(Array.isArray(o.tried)) o.tried.slice(0,40).forEach(x=>{if(typeof x==='string'&&x.length<300) tried.add(x);});
  acted=o.acted===true; speed=RATE[o.speed]?o.speed:'fast';
  chA=Array.isArray(o.chA)&&o.chA.length===6&&o.chA.every(v=>Number.isInteger(v)&&v>=1&&v<=6)?o.chA.slice():[1,2,3,4,5,6];
  chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
  drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;
  seen.clear(); if(Array.isArray(o.seen)) o.seen.slice(-12).forEach(x=>{if(typeof x==='string'&&/^(trip|code)\|/.test(x)&&x.length<400) seen.add(x);});
  stopAnim(); restored=true;}
A.registerLab({
  id:ID, snap, restore,
  mount(el,tok){
    if(raf){cancelAnimationFrame(raf); raf=0;} root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    if(!p&&!relang&&!restored){stopAnim(); if(mode==='guided'){const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());}}   // a state just put back keeps its settings
    restored=false;
    el.innerHTML=viewHTML(); app=$('#tls-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#tls-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    renderRail();
    if(anim&&!anim.paused){lastTs=0; raf=requestAnimationFrame(tick);}},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const dm=$('#tls-math',app), open=dm&&dm.open; relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#tls-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#tls-pan-gantt:not([hidden])',app)||$('#tls-pan-graph',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#tls-stage',app),
    render({status,vis,read}){if(!app) return; const R=curRun(), clock=clockNow(R), T=axisMax(R);
      vis.innerHTML=`<div class="tls-dk"><i style="width:${(Math.min(clock,R.total)/T*100).toFixed(1)}%"></i></div>`;
      if(inCh()){const me=chMe(), done=chDone(); status.className='status'+(done?(me.ok?' ok':' bad'):''); status.textContent=done?(me.ok?tx('Goal met','已达标'):tx('Not yet','未达标')):tx('Not submitted','未提交');
        read.textContent=done?tx(`Your arrangement: ${s1(me.total)} s, ${me.n} model calls, the result is ${resTxt(me.R)}.`,`你的安排：${s1(me.total)} 秒，${me.n} 次模型调用，结果${resTxt(me.R)}。`):tx(`Your arrangement uses ${me.n} model calls.`,`你的安排用 ${me.n} 次模型调用。`); return;}
      const live=clock<R.total-1e-9; status.className='status'+(live?'':(R.bad.length?' bad':' ok')); status.textContent=live?tx('Playing','播放中'):(R.bad.length?tx('Wrong','结果错'):tx('Right','结果对'));
      read.textContent=live?tx(`${s1(clock)} s so far, ${L(SCH_N[S.sch]).toLowerCase()}.`,`已经 ${s1(clock)} 秒：${L(SCH_N[S.sch])}。`):tx(`${L(SCH_N[S.sch])}: ${R.n} model calls, ${s1(R.total)} s.`,`${L(SCH_N[S.sch])}：${R.n} 次模型调用，${s1(R.total)} 秒。`);}}
});
/* read-only hook for tests/tool-scheduling_numbers.py and the flow test */
window.__tls={state:()=>({mode,step,...S,pred,pick2,chA:chA.slice(),chSubmitted,chReveal,lang:LG(),speed}),anim:()=>anim?{t:anim.t,key:anim.key,paused:!!anim.paused}:null,seen:()=>[...seen],
  data:()=>({TRIP,CODE,TO,MS,KS,DEFAULT,FAILI,CH_K,CH_M,CH_GOAL}),
  run:s=>{const R=runOf({...DEFAULT,...s}); return {total:R.total,n:R.n,toolSum:R.toolSum,bad:R.bad,crit:[...R.crit].sort((a,b)=>a-b),cs:R.cs,turns:R.turns};},
  sched:(id,turns,o)=>{const R=sched(id==='code'?CODE:TRIP,turns,o); return {total:R.total,n:R.n,toolSum:R.toolSum,bad:R.bad,crit:[...R.crit].sort((a,b)=>a-b),cs:R.cs};},
  levels:id=>levels(id==='code'?CODE:TRIP),chAll:()=>chAll().map(p=>({key:p.key,total:p.total,n:p.n,right:p.right,ok:p.ok,best:p.best})),drill:makeDrill};
})();
