/* agent-anatomy (world agents): What actually happens inside one agent task?
   Fidelity: Illustrative (header fid 'Conceptual'). A constructed run in the message format of a tool-use API, the format of the
   interview problem coding/llm-agent-tool-loop: system and tools sent with every call; messages of text, tool_use {id, name, input}
   and tool_result {tool_use_id, content, is_error} blocks; stop_reason "tool_use" or "end_turn". Not a recording.
   - Tokens: each block counts ceil(characters / 4) (the lab's stated approximation). A tool_use block counts its JSON; big blocks
     (system prompt, tool definitions, files, logs) have a stated character count and a preview.
   - A call's input = every block sent so far (system + tools + all messages); its output = the model's reply.
   - Time (assumed): model call = t0 + input / prefill speed + output / decode speed; tools take fixed times (run_tests: a setting);
     the tools of one reply run at the same time.
   - Limits, checked before each call (as mini-swe-agent's query() does): calls made >= call limit; tokens read so far + this call's
     input > token budget.
   Words (EN / ZH): model call / 模型调用 (第 N 次调用); your code / 你的代码 = the harness; tool / 工具; part / 部分.
   window.__ana is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='agent-anatomy';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine: the constructed runs
   ===================================================================== */
const TOK=c=>Math.ceil(c/4);                                   // the lab's rule: about 4 characters per token
const TOOLT={read_file:0.1,search:0.3,edit_file:0.1,run_lint:1.2}; // seconds; run_tests is a setting
const SYS={kind:'sys',who:'you',chars:12000,prev:'You are a coding agent working in a Python repository. Use the tools to read files, search the code, edit files and run the tests. Make the smallest change that fixes the failing test, run the tests again, then report what you changed.'};
const TOOLS4={kind:'tools',who:'you',chars:4800,n:4,prev:'read_file(path), search(pattern), edit_file(path, old, new), run_tests(path)'};
const TOOLS5={kind:'tools',who:'you',chars:5800,n:5,prev:'read_file(path), search(pattern), edit_file(path, old, new), run_tests(path), run_lint(path)'};
const use=(name,input)=>({name,input});
const res=(chars,prev,err)=>({chars,prev,err:!!err});
const TASK_FIX='CI is red: tests/test_cart.py::test_member_coupon fails with assert 90.0 == 81.0. Find the bug and fix it.';
const R={
 test:res(5600,'import pytest\nfrom cart import Item, total\n\ndef test_member_coupon():\n    items = [Item(price=100.0, qty=1)]\n    assert total(items, member=True, coupon=0.10) == 81.0\n…'),
 cart:res(13600,'def total(items, member=False, coupon=0.0):\n    subtotal = sum(i.price * i.qty for i in items)\n    discount = 0.10 if member else 0.0\n    if coupon:\n        discount = coupon\n    return round(subtotal * (1 - discount), 2)\n…'),
 grep:res(1400,'cart.py:41:    if coupon:\ncart.py:42:        discount = coupon\ncheckout.py:18:    amount = total(cart.items, member=user.is_member, coupon=code.rate)\n…'),
 ed1:res(600,'Edited cart.py, line 42:\n-        discount = coupon\n+        discount += coupon'),
 fail:res(7200,'FAILED tests/test_cart.py::test_member_coupon - assert 80.0 == 81.0\n1 failed, 11 passed\n…',true),
 ed2:res(600,'Edited cart.py, lines 41-43:\n-    if coupon:\n-        discount += coupon\n-    return round(subtotal * (1 - discount), 2)\n+    return round(subtotal * (1 - discount) * (1 - coupon), 2)'),
 pass:res(800,'12 passed\n…'),
 huge:res(160000,'tests/test_cart.py::test_empty_cart PASSED\nDEBUG cart: subtotal=0.0 member=False coupon=0.0\n…\nFAILED tests/test_cart.py::test_member_coupon - assert 80.0 == 81.0\n…',true),
 code1:res(11,'exit code 1',true), code0:res(11,'exit code 0')
};
const U={
 rt:use('read_file',{path:'tests/test_cart.py'}), rc:use('read_file',{path:'cart.py'}), sc:use('search',{pattern:'coupon'}),
 e1:use('edit_file',{path:'cart.py',old:'        discount = coupon',new:'        discount += coupon'}),
 t:use('run_tests',{path:'tests/'}), t1:use('run_tests',{path:'tests/test_cart.py::test_member_coupon'}),
 tv:use('run_tests',{path:'tests/',flags:'-vv --log-level=DEBUG'}),
 e2:use('edit_file',{path:'cart.py',old:'    if coupon:\n        discount += coupon\n    return round(subtotal * (1 - discount), 2)',new:'    return round(subtotal * (1 - discount) * (1 - coupon), 2)'})
};
const FINAL='Fixed cart.total: a coupon used to replace the member discount. Now the two apply one after the other (100.00 to 90.00 to 81.00). All 12 tests pass.';
/* the rename task (30 calls): 13 files to read and edit */
const RN_FILES=[['app/api.py',5200],['app/forms.py',3800],['app/report.py',7600],['app/billing.py',4400],['app/export.py',6000],['app/cli.py',3200],['app/sync.py',8400],
  ['app/models.py',4800],['app/views.py',5600],['app/tasks.py',3600],['app/admin.py',7000],['app/utils.py',4000],['tests/test_dates.py',6400]];
const TASK_RN='Rename parse_date to parse_iso_date everywhere in the repository, and keep the tests green.';
const RN_CALLS=[[[use('search',{pattern:'parse_date('}),res(1600,'app/api.py:12:    start = parse_date(req.args["from"])\napp/forms.py:30:    return parse_date(value)\n… 13 files')]]]
  .concat(...RN_FILES.map(([f,c])=>[[[use('read_file',{path:f}),res(c,`# ${f}\nfrom app.dates import parse_date\n…`)]],[[use('edit_file',{path:f,old:'parse_date(',new:'parse_iso_date('}),res(400,`Edited ${f}: every parse_date( is now parse_iso_date(`)]]]))
  .concat([[[U.t,res(1000,'214 passed\n…')]],[[use('search',{pattern:'parse_date('}),res(11,'No matches.')]]]);
const FINAL_RN='Renamed parse_date to parse_iso_date in 13 files. All 214 tests pass, and no parse_date( is left.';
/* the challenge's task (12 calls) and its stuck twin */
const TASK_CH='Add retries with a timeout to http_client.get, and add a test for them.';
const CHU={rh:use('read_file',{path:'http_client.py'}),sg:use('search',{pattern:'http_client.get('}),rt:use('read_file',{path:'tests/test_http_client.py'}),
  e1:use('edit_file',{path:'http_client.py',old:'    resp = session.get(url, timeout=timeout)\n    return resp',new:'    for attempt in range(retries + 1):\n        try:\n            return session.get(url, timeout=timeout)\n        except requests.Timeout:\n            if attempt == retries:\n                raise\n            time.sleep(backoff * 2 ** attempt)'}),
  e2:use('edit_file',{path:'tests/test_http_client.py',old:'# retries',new:'def test_get_retries_on_timeout(fake_session):\n    fake_session.fail_times(2)\n    assert get("https://x", retries=2).status_code == 200'}),
  t:use('run_tests',{path:'tests/'}),e3:use('edit_file',{path:'http_client.py',old:'import requests',new:'import time\nimport requests'}),
  l:use('run_lint',{path:'.'}),e4:use('edit_file',{path:'http_client.py',old:'    for attempt in range(retries + 1):',new:'    for attempt in range(retries + 1):  # noqa: B007'}),
  e5:use('edit_file',{path:'tests/test_http_client.py',old:'timeout=0.01',new:'timeout=0.05'})};
const CHR={rh:res(10400,'import requests\n\ndef get(url, timeout=5.0, retries=0, backoff=0.5):\n    resp = session.get(url, timeout=timeout)\n    return resp\n…'),
  sg:res(1600,'app/sync.py:22:    data = http_client.get(feed_url).json()\n…'),rt:res(7200,'from http_client import get\n\ndef test_get_ok(fake_session):\n…'),
  ed:res(800,'Edited http_client.py\n…'),edt:res(600,'Edited tests/test_http_client.py\n…'),
  nameErr:res(4800,"FAILED tests/test_http_client.py::test_get_retries_on_timeout - NameError: name 'time' is not defined\n1 failed, 30 passed\n…",true),
  pass:res(800,'31 passed\n…'),lint:res(1200,'http_client.py:9:9: B007 loop variable not used\n2 warnings\n…',true),ed3:res(400,'Edited http_client.py\n…'),clean:res(120,'All checks passed.'),
  flaky:res(4800,'FAILED tests/test_http_client.py::test_get_retries_on_timeout - TimeoutError\n1 failed, 30 passed\n…',true)};
const FINAL_CH='http_client.get now retries a timed-out request up to `retries` times, waiting longer each time. A new test covers it; tests and lint pass.';
const STUCK_CALLS=[[[CHU.rh,CHR.rh]],[[CHU.sg,CHR.sg]],[[CHU.rt,CHR.rt]],[[CHU.e1,CHR.ed]],[[CHU.e2,CHR.edt]]].concat(...Array.from({length:40},()=>[[[CHU.t,CHR.flaky]],[[CHU.e5,CHR.edt]]]));
const RUNS={
 base:{task:TASK_FIX,tools:TOOLS4,calls:[[[U.rt,R.test]],[[U.rc,R.cart]],[[U.sc,R.grep]],[[U.e1,R.ed1]],[[U.t,R.fail]],[[U.e2,R.ed2]],[[U.t,R.pass]]],final:FINAL},
 par:{task:TASK_FIX,tools:TOOLS4,calls:[[[U.rt,R.test],[U.rc,R.cart],[U.sc,R.grep]],[[U.e1,R.ed1]],[[U.t,R.fail]],[[U.e2,R.ed2]],[[U.t,R.pass]]],final:FINAL},
 terse:{task:TASK_FIX,tools:TOOLS4,calls:[[[U.rt,R.test]],[[U.rc,R.cart]],[[U.sc,R.grep]],[[U.e1,R.ed1]],[[U.t,R.code1]],[[U.t1,R.code1]],[[U.rt,R.test]],[[U.e2,R.ed2]],[[U.t,R.code0]]],final:FINAL},
 huge:{task:TASK_FIX,tools:TOOLS4,calls:[[[U.rt,R.test]],[[U.rc,R.cart]],[[U.sc,R.grep]],[[U.e1,R.ed1]],[[U.tv,R.huge]],[[U.e2,R.ed2]],[[U.t,R.pass]]],final:FINAL},
 long:{task:TASK_RN,tools:TOOLS4,calls:RN_CALLS,final:FINAL_RN},
 chN:{task:TASK_CH,tools:TOOLS5,calls:[[[CHU.rh,CHR.rh]],[[CHU.sg,CHR.sg]],[[CHU.rt,CHR.rt]],[[CHU.e1,CHR.ed]],[[CHU.e2,CHR.edt]],[[CHU.t,CHR.nameErr]],[[CHU.e3,CHR.ed3]],[[CHU.t,CHR.pass]],[[CHU.l,CHR.lint]],[[CHU.e4,CHR.ed3]],[[CHU.l,CHR.clean]]],final:FINAL_CH},
 chS:{task:TASK_CH,tools:TOOLS5,calls:STUCK_CALLS,final:null}
};
/* one run under the given settings: o = {lim, budget, stopOnErr, pre, dec, lat, tt} */
function build(key,o){
  const def=RUNS[key], P={lim:0,budget:0,stopOnErr:false,...TIME0,...o};
  const blocks=[{...SYS,at:-1,call:0},{...def.tools,at:-1,call:0},{kind:'task',who:'you',chars:def.task.length,prev:def.task,at:-1,call:0}];
  blocks.forEach(b=>{b.tok=TOK(b.chars);});
  const calls=[], events=[]; let used=0, n=0, stop=null, uid=0, clock=0;
  const steps=def.calls.concat(def.final?[null]:[]);
  for(const st of steps){
    const inTok=blocks.reduce((a,b)=>a+b.tok,0);
    if(P.lim&&n>=P.lim){stop='limit'; break;}
    if(P.budget&&used+inTok>P.budget){stop='budget'; break;}
    n++; used+=inTok; events.push({type:'send',call:n});
    let reply;
    if(!st) reply=[{kind:'text',who:'model',chars:def.final.length,prev:def.final}];
    else reply=st.map(([u])=>{uid++; const tid='toolu_'+String(uid).padStart(2,'0'); return {kind:'use',who:'model',name:u.name,input:u.input,tid,chars:JSON.stringify({type:'tool_use',id:tid,name:u.name,input:u.input}).length};});
    reply.forEach(b=>{b.tok=TOK(b.chars); b.call=n; b.at=events.length; if(b.kind==='use') b.prev=b.name+' '+JSON.stringify(b.input);});
    const out=reply.reduce((a,b)=>a+b.tok,0), tM=P.lat+inTok/P.pre+out/P.dec;
    events.push({type:'reply',call:n,dur:tM}); blocks.push(...reply);
    const c={i:n,inTok,out,newTok:inTok-(calls.length?calls[calls.length-1].inTok:0),reply,res:[],end:!st,tM,tT:0,start:clock,tools:[]};
    clock+=tM; calls.push(c);
    if(!st){stop='end'; break;}
    const rs=st.map(([u,r],k)=>({kind:'res',who:'tool',chars:r.chars,tok:TOK(r.chars),prev:r.prev,err:r.err,name:u.name,tid:reply[k].tid,call:n}));
    c.tools=st.map(([u])=>({name:u.name,t:u.name==='run_tests'?P.tt:TOOLT[u.name]}));
    c.tT=Math.max(...c.tools.map(t=>t.t)); clock+=c.tT;
    events.push({type:'tool',call:n,dur:c.tT,err:rs.some(r=>r.err)}); rs.forEach(b=>{b.at=events.length-1;}); blocks.push(...rs); c.res=rs;
    if(P.stopOnErr&&rs.some(r=>r.err)){stop='error'; break;}
  }
  if(!stop) stop='end';
  if(stop!=='end') events.push({type:'stop',why:stop,call:n});
  const totalIn=calls.reduce((a,c)=>a+c.inTok,0), totalOut=calls.reduce((a,c)=>a+c.out,0);
  const tModel=calls.reduce((a,c)=>a+c.tM,0), tTool=calls.reduce((a,c)=>a+c.tT,0);
  return {key,calls,blocks,events,stop,done:stop==='end',n,totalIn,totalOut,finalIn:calls.length?calls[calls.length-1].inTok:0,tModel,tTool,time:tModel+tTool,P};
}
const TIME0={pre:10000,dec:50,lat:0.5,tt:3};
/* a file the model asked for: the call that asked for it, the first call that sends its content, and how many calls send it */
function resSends(run,path){const u=run.blocks.find(b=>b.kind==='use'&&b.input&&b.input.path===path); if(!u) return null;
  return {asked:u.call,first:u.call+1,times:Math.max(0,run.calls.length-u.call)};}
/* the position in a replay: p events done (0 = before call 1) */
function at(run,p){
  p=Math.max(0,Math.min(run.events.length,p));
  const ev=run.events.slice(0,p), sends=ev.filter(e=>e.type==='send');
  const call=sends.length?sends[sends.length-1].call:0;
  const vis=run.blocks.filter(b=>b.at<p);
  const readSoFar=run.calls.filter(c=>c.i<=call).reduce((a,c)=>a+c.inTok,0);
  const t=ev.reduce((a,e)=>a+(e.dur||0),0), tm=ev.filter(e=>e.type==='reply').reduce((a,e)=>a+e.dur,0);
  return {p,call,last:p?run.events[p-1]:null,vis,readSoFar,t,tm,tt:t-tm,sendIx:run.events.map((e,i)=>e.type==='send'?i:-1).filter(i=>i>=0)};
}
const sendPos=(run,k)=>{let c=0; for(let i=0;i<run.events.length;i++){if(run.events[i].type==='send'){c++; if(c===k) return i+1;}} return run.events.length;};

/* the challenge: a call limit and a token budget, for the normal run and its stuck twin */
const CH={lims:[10,15,25,50],buds:[50000,100000,150000,300000],start:{lim:50,bud:300000}};
function chEval(lim,bud){const nr=build('chN',{lim,budget:bud}), sr=build('chS',{lim,budget:bud});
  return {lim,bud,ok:nr.done,nStop:nr.stop,nCalls:nr.n,nRead:nr.totalIn,sStop:sr.stop,sCalls:sr.n,cost:sr.totalIn};}
function chAll(){const a=[]; CH.lims.forEach(l=>CH.buds.forEach(b=>a.push(chEval(l,b)))); return a;}
/* among the plans that let the normal run finish, the one where the stuck run reads the fewest tokens (R4.5) */
function chBest(){return chAll().filter(e=>e.ok).sort((a,b)=>a.cost-b.cost||a.lim-b.lim||a.bud-b.bud)[0];}
/* several plans can tie at the lowest cost (once the budget stops the stuck run, a looser call limit changes nothing): all of them
   are minimal; a suggestion keeps the learner's own call limit or budget when it can */
const chMinimal=e=>e.ok&&e.cost===chBest().cost;
function chSuggest(me){const best=chBest(), all=chAll().filter(e=>e.ok&&e.cost===best.cost); return all.find(e=>e.lim===me.lim)||all.find(e=>e.bud===me.bud)||best;}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fT=n=>fint(n);                                             // tokens, written in full (six digits or fewer)
const fK=n=>n>=1e6?A.fmt.big(n):fint(n);
const fS=s=>fixd(s,1);                                           // seconds, one decimal
const fPc=x=>fixd(x*100,0)+'%';
const about=n=>fint(Math.round(n/1000)*1000);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const labLinkZ=id=>{const h=labLink(id); return /^[A-Za-z0-9]/.test(h.replace(/<[^>]+>/g,''))?' '+h:h;};   // after a Chinese character
const M=t=>`<div class="ana-fx">${A.ui.math(t)}</div>`;
const m=t=>A.ui.math(t);   // inline, inside a sentence
/* a formula on its own line with its name in words on the left; in a narrow column the formula moves under its name */
const MN=(en,zh,t)=>`<div class="ana-fx ana-fxn"><span class="ana-fx-nm">${tx(en,zh)}</span>${A.ui.math(t)}</div>`;
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="ana-eqs${EQN(rows)?' ana-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="ana-eq-l">${l?A.ui.math(l):''}</span><span class="ana-eq-e">${A.ui.math(rel||'=')}</span><span class="ana-eq-r">${A.ui.math(r)}</span>`).join('')}</div>`;
/* a long sum, three terms a line: the first line after “=”, the next ones start with “+” under it (a formula does not wrap by itself, R6.33) */
const SUMROWS=(name,terms,per=3)=>{const rows=[]; for(let i=0;i<terms.length;i+=per){const ch=terms.slice(i,i+per).join(' + '); rows.push(i===0?[name,ch]:['',`+ ${ch}`,' ']);} return rows;};
const MB=(lead,body,res)=>`<div class="ana-mb"><p>${lead}</p>${body}${res?`<p class="ana-mb-res">${res}</p>`:''}</div>`;
const toolsW=n=>tx(n===1?'1 tool':`${n} tools`,`${n} 个工具`);
const callsW=n=>tx(n===1?'1 call':`${n} calls`,`${n} 次调用`);

/* =====================================================================
   module state
   ===================================================================== */
const BASE={run:'base',lim:0,...TIME0};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let pos=0, playing=false, spd='mid', raf=0, lastT=0, acc=0, replayDone=false, callK=1, callMax=1, relang=false;
/* R3.20: the runs the reader has watched to the end (played, stepped or scrubbed to the last event, or Show me), in any part or mode, by
   what they draw (the run, its limits and its times), not by part; part 2 shows a run already watched (in Scenarios, say) finished at once */
const watched=new Set();
const runSig=r=>JSON.stringify([r.key,r.P.lim,r.P.budget,r.P.stopOnErr,r.P.pre,r.P.dec,r.P.lat,r.P.tt]);
let fx='std', fxTried=new Set(['std']), par='seq', parTried=new Set(['seq']), remView='who';
let chSubmitted=false, chReveal=false, chO={...CH.start}, saved=null, sbTouched=false, userAct=false, wantObs=false, drillSeed=1, drillOpen=false, annT=null;
const shownPrev={};
const SPD={slow:1.6,mid:0.9,fast:0.2};                            // seconds per event; the base run has 23 events (fast: 4.6 s)
const ROWS_MAX=64;

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 harness:()=>tx('The harness is the code around the model that runs the agent: it calls the model, runs the tools, hands the results back, and decides when to retry or stop.',
                'harness 是包在模型外面、负责运行 agent 的代码：它调用模型、执行工具、把结果交回去，并决定什么时候重试、什么时候停下。'),
 token:()=>A.terms&&A.terms.token?A.terms.token():tx('A token is a piece of text the model reads or writes, often a word or part of one.','token 是模型读写文字的单位，常常是一个词或词的一部分。'),
 rule:()=>tx('This lab counts about 4 characters of English text or code as 1 token, rounded up per block. Real counts depend on the model’s tokenizer and the language, so every token count here is an estimate (see the lab Tokenization cost).',
             '本 Lab 按“大约 4 个字符算 1 个 token”来数英文和代码，每一块向上取整。真实的数取决于模型的分词器和语言，所以这里的每个 token 数都是估计（见 Lab“token 成本”）。'),
 kv:()=>(A.terms&&A.terms.kvcache?A.terms.kvcache():tx('The KV cache keeps the key and value vectors of every token the model has already processed, in every layer, so that when the model generates the next token it reads them from memory instead of computing them again.',
             'KV cache 存着模型已经处理过的每个 token 在每一层的 key 和 value 向量，模型生成下一个 token 时直接从显存里读，不再重算。'))+
    tx(' After a call ends, the next call rebuilds it from the whole conversation sent again; with prompt caching, a provider reuses the part that starts the same way.',
       '一次调用结束后，下一次调用要从重发来的整段对话重新算出它；开了 prompt caching 的服务商，能复用开头相同的那一段。'),
 toolUse:()=>tx('tool_use is a block in the model’s reply: the name of a tool and its arguments (input), with an id. It is only data; nothing runs until your code runs it.',
                'tool_use 是模型回复里的一块：工具的名字和参数（input），再加一个 id。它只是一段数据，你的代码去执行它之前，什么都没有发生。'),
 toolRes:()=>tx('tool_result is the block your code sends back: the tool’s output (content), the id of the tool_use it answers (tool_use_id), and is_error, set to true when the tool failed.',
                'tool_result 是你的代码发回去的一块：工具的输出（content）、它回答的是哪一个 tool_use（tool_use_id），以及 is_error（工具失败时设为 true）。'),
 endTurn:()=>tx('Every reply carries a stop_reason: "tool_use" means the model asks for tools and waits for their results; "end_turn" means it has finished, and the loop ends.',
                '每个回复都带一个 stop_reason：“tool_use”表示模型要调用工具，等结果回来再继续；“end_turn”表示它做完了，循环到此结束。'),
 block:()=>tx('A block (content block) is one piece of a message: a piece of text, a tool_use or a tool_result. A message is made of one or more blocks. Here each row is one block; the system prompt and the tool definitions get a row each too.',
              '块（content block）是消息里的一段内容，比如一段文字、一个 tool_use 或一个 tool_result。一条消息由一块或几块组成。这里每一行画一块，system prompt 和工具说明也各占一行。'),
 sysp:()=>tx('The system prompt is the instruction your code puts at the top of every call: who the model is, what it may do, how to work.','system prompt 是你的代码放在每次调用最前面的说明：模型扮演谁、能做什么、该怎么做。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const runBase=()=>build('base',S);
const PRED=[{v:'small',t:{en:'Under 30,000',zh:'不到 30,000 个'}},{v:'mid',t:{en:'30,000 to 90,000',zh:'30,000 到 90,000 个'}},{v:'big',t:{en:'Over 90,000',zh:'90,000 个以上'}}];
const PRED_RIGHT='mid';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const FX=[{v:'std',t:{en:'Send the error back (initial setup)',zh:'把报错发回去（初始设置）'},d:{en:'The test output, which test failed and the two values, goes back as the tool_result, with is_error set to true.',zh:'把测试的输出（哪个测试没过、断言两边的值）作为 tool_result 发回去，is_error 设为 true。'},run:'base',o:{}},
  {v:'stop',t:{en:'Stop on any tool error',zh:'工具一出错就停下'},d:{en:'Your code treats the failure as a crash: the whole task ends there.',zh:'你的代码把出错当成故障：整个任务到此结束。'},run:'base',o:{stopOnErr:true}},
  {v:'terse',t:{en:'Send back only “exit code 1”',zh:'只发回“exit code 1”'},d:{en:'The result still says the tests failed, but not which test or why.',zh:'结果只说测试没通过，不说是哪个测试、为什么。'},run:'terse',o:{}}];
const fxRun=v=>{const f=FX.find(x=>x.v===v); return build(f.run,{...S,...f.o});};
const fxRes=v=>{const r=fxRun(v); return r.done?tx(`fixed, ${r.n} calls`,`${r.n} 次调用，修好了`):tx(`stopped after call ${r.n}`,`第 ${r.n} 次后停下，没修好`);};
const PAR=[{v:'seq',t:{en:'One tool per call (initial setup)',zh:'一次一个（初始设置）'},d:{en:'Calls 1 to 3 each ask for one thing: the test file, cart.py, then a search for “coupon”.',zh:'第 1 到 3 次调用各要一样东西：测试文件、cart.py、搜索“coupon”。'}},
  {v:'par',t:{en:'Three tools in one call',zh:'一次要 3 个'},d:{en:'The first call asks for all three at once; your code runs them at the same time.',zh:'第 1 次调用就把三样一起要过来；你的代码同时执行它们。'}}];
const parRun=v=>build(v==='par'?'par':'base',S);
const parRes=v=>{const r=parRun(v); return tx(`${fS(r.time)} s, ${r.n} calls`,`${fS(r.time)} 秒，${r.n} 次调用`);};
const levers=(id,opts,val,tried,resFn,aria)=>`<div class="ana-lv" id="${id}">`+A.ui.levers({aria,value:val,options:opts.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(tried.has(o.v)||o.v===val?resFn(o.v):'')}))})+`</div>`;
const callCtl=(id,inline)=>A.ui.range({id,key:'callk',label:tx('Which call','第几次调用'),min:1,max:8,value:callK,out:tx(`call ${callK}`,`第 ${callK} 次`),inline});
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'start',gate:()=>!!pred,skip:true,
  lead:()=>tx('An agent fixes a test with 8 model calls. How many tokens does the model read in all?','agent 调用 8 次模型，修好了一个测试。模型一共读了多少 token？'),
  body:()=>{const r=runBase(), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>A test in your repository is failing. You hand it to a coding agent: it reads the code, edits it, runs the tests, and reports that the bug is fixed.</p><p>In this task the agent calls the model ${r.n} times. The last call sends the model a conversation of about ${about(r.finalIn)} ${DEF('tokens',TIP.token())}.</p><p>Over all ${r.n} calls, how many tokens does the model read in total?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>你们代码仓库里有一个测试没通过。你把它交给一个 coding agent：它读代码、改代码、跑测试，最后报告修好了。</p><p>这个任务里，agent 一共调用了 ${r.n} 次模型。最后一次调用发给模型的对话，大约有 ${about(r.finalIn)} 个 ${DEF('token',TIP.token())}。</p><p>${r.n} 次调用加起来，模型一共读进去多少 token？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'who',title:{en:'Who does what',zh:'谁在做事'},view:'replay',expect:()=>replayDone,showMe:()=>finishReplay(),label:{en:'Result',zh:'结果'},
  lead:()=>tx('The model only says what to do next; your code reads, edits and runs everything.','模型只说下一步做什么；读文件、改代码、跑测试，都是你的代码在做。'),
  body:()=>tx(`<p>An agent is a loop. Your code sends the conversation and the available tools to the model. The model either answers, or says “call this tool with these arguments”.</p><p>Your code runs the tool, appends the result to the conversation and sends it all back. This repeats until the model says it is done. The code that runs this loop is the ${DEF('harness',TIP.harness())}.</p>`,
              `<p>agent 就是一个循环。你的代码把对话和可用的工具发给模型；模型要么直接回答，要么说“我要调用某个工具，参数是这些”。</p><p>你的代码去执行这个工具，把结果接在对话后面，再整段发给模型。如此往复，直到模型说做完了。跑这个循环的那段代码，叫 ${DEF('harness',TIP.harness())}。</p>`),
  obs:()=>obsWho()},
 {id:'memory',title:{en:'The conversation is the memory',zh:'对话就是记忆'},view:'calls',expect:()=>callMax>=8,showMe:()=>setCall(8),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Every call sends the whole conversation again; the model reads the sum of all calls.','每次调用都把整段对话重发一遍；模型读进去的，是每一次的总和。'),
  body:()=>tx(`<p>The model does not remember the previous call. For every call, your code sends the whole conversation again: the system prompt, the tool definitions, the task, and every reply and tool result so far.</p><p>Step through the calls. In the chart, each bar is what one call sends, in ${DEF('tokens',TIP.rule())}.</p>`,
              `<p>模型不记得上一次调用。每次调用，你的代码都要把到目前为止的整段对话重新发一遍：system prompt、工具说明、任务，以及之前每一次的回复和工具结果。</p><p>一次一次往后看。图里每根柱子，是一次调用发出去的 ${DEF('token',TIP.rule())}。</p>`)+callCtl('ana-ic-call',true),
  obs:()=>obsMemory()},
 {id:'fail',title:{en:'When a tool fails',zh:'工具出错的时候'},view:'fail',expect:()=>fxTried.has('stop')&&fxTried.has('terse'),showMe:()=>{setFx('stop'); setFx('terse');},label:{en:'Result',zh:'结果'},
  lead:()=>tx('A failed test goes back as a tool result marked as an error; the model reads it and decides what to fix.','测试没通过，结果照样作为工具结果发回去、标成出错；模型看到报错，自己决定怎么改。'),
  body:()=>tx(`<p>After call 5 your code runs the tests, and one fails. That is not a crash of the loop. It is information for the model: the output goes back as a ${DEF('tool_result',TIP.toolRes())}, with is_error set to true.</p><p>Your code can handle a failed tool in three ways. Pick one and see whether the bug gets fixed.</p>`,
              `<p>第 5 次调用之后，你的代码跑了测试，有一个没通过。这不是循环出了故障，而是给模型的信息：测试的输出照样作为 ${DEF('tool_result',TIP.toolRes())} 发回去，is_error 设为 true。</p><p>工具出错时，你的代码有三种做法。选一种，看看 bug 最后修没修好。</p>`)+levers('ana-ic-fx',FX,fx,fxTried,fxRes,tx('Pick how your code handles a failed tool','选一种工具出错时的做法')),
  obs:()=>obsFail()},
 {id:'time',title:{en:'Where the time goes',zh:'时间花在哪里'},view:'time',expect:()=>parTried.has('par'),showMe:()=>setPar('par'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('In this task model calls take most of the time, and independent tools can be asked for at once and run together.','这个任务的时间大部分花在模型调用上，互不依赖的工具可以一次要齐、同时去跑。'),
  body:()=>tx(`<p>Under this lab’s assumptions, most of a task’s time goes to model calls. In every call the model first reads the whole conversation, then writes its reply one token at a time. Writing is much slower than reading (why: ${labLink('prefill-decode')}).</p>`,
              `<p>在这个 Lab 的假设下，一次任务的时间大部分花在模型调用上。每次调用，模型都要先读完整段对话，再一个一个 token 地写出回复。写比读慢得多，原因见${labLinkZ('prefill-decode')}。</p>`)+
    MN('time of one call','一次调用的时间','t = t_0 + \\frac{N_\\text{in}}{v_\\text{prefill}} + \\frac{N_\\text{out}}{v_\\text{decode}}')+
    tx(`<p>${m('t')} is one model call’s time, and ${m('t_0')} is the fixed cost of every call. ${m('N_\\text{in}')} and ${m('N_\\text{out}')} are the tokens the call reads and writes, and ${m('v_\\text{prefill}')} and ${m('v_\\text{decode}')} are the reading and writing speeds. The tools’ time comes on top; except for the test runs, the tools are fast.</p><p>Calls 1 to 3 ask for three things that do not depend on each other. Pick a way to ask for them and see how long the whole task takes.</p>`,
       `<p>${m('t')} 是一次模型调用的时间，${m('t_0')} 是每次调用固定的开销。${m('N_\\text{in}')} 和 ${m('N_\\text{out}')} 是这次调用读进去、写出来的 token 数，${m('v_\\text{prefill}')} 和 ${m('v_\\text{decode}')} 是读和写的速度。工具的时间另算。除了跑测试，别的工具都很快。</p><p>第 1 到 3 次调用要的三样东西，互不依赖。选一种要工具的方式，看看整个任务要花多少秒。</p>`)+levers('ana-ic-par',PAR,par,parTried,parRes,tx('Pick how many tools a call asks for','选一种要工具的方式')),
  obs:()=>obsTime()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>({who:'replay',memory:'calls',fail:'fail',time:'time'}[remView]||'replay'),
  lead:()=>tx('Your code runs the tools. Every call resends everything. Errors go back to the model. In this lab, model calls take most of the time.','工具由你的代码执行。每次调用都重发全部。报错要交回模型。这个 Lab 里，时间主要花在模型调用上。'),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
function obsWho(){
  if(!replayDone) return '';
  const r=runBase(), nt=r.calls.reduce((a,c)=>a+c.res.length,0);
  return tx(`<p><b>In ${r.n} calls the model replied ${r.n} times: ${r.n-1} times it asked for a tool, and the last time it answered. All ${nt} tool calls (reading, searching, editing, running the tests) were run by your code.</b></p><p>The model never touches a file or a terminal. Its reply is only data: which tool, with which arguments (a ${DEF('tool_use',TIP.toolUse())} block). Your code runs it and sends the output back as a tool_result, with the same id, so the model knows which request it answers. The last reply’s ${DEF('stop_reason',TIP.endTurn())} is end_turn, and the loop stops.</p>`,
            `<p><b>${r.n} 次调用，模型回复了 ${r.n} 次：前 ${r.n-1} 次要求调用工具，最后 1 次给出答案。读文件、搜索、改代码、跑测试，这 ${nt} 次工具调用全是你的代码执行的。</b></p><p>模型从不碰文件和终端。它的回复只是一段数据：用哪个工具、参数是什么（${DEF('tool_use',TIP.toolUse())} 块）。你的代码照着执行，再把输出作为 tool_result 发回去，带上同一个 id，模型就知道这是哪一次请求的结果。最后一次回复的 ${DEF('stop_reason',TIP.endTurn())} 是 end_turn，循环就停了。</p>`);
}
function obsMemory(){
  const r=runBase(), c=r.calls[callK-1], prev=callK>1?r.calls[callK-2]:null;
  let s=callK===1?tx(`<p><b>Call 1 sends ${fT(c.inTok)} tokens: the system prompt, the tool definitions and the task.</b> Step on to call 8.</p>`,`<p><b>第 1 次调用发出 ${fT(c.inTok)} 个 token：system prompt、工具说明和任务。</b>往后看到第 8 次。</p>`)
    :tx(`<p><b>Call ${callK} sends ${fT(c.inTok)} tokens. ${fT(prev.inTok)} of them were already sent by call ${callK-1}; only ${fT(c.newTok)} are new.</b></p>`,`<p><b>第 ${callK} 次调用发出 ${fT(c.inTok)} 个 token，其中 ${fT(prev.inTok)} 个第 ${callK-1} 次已经发过，新的只有 ${fT(c.newTok)} 个。</b></p>`);
  if(callMax<8) return s+(callK>1?tx('<p>Step on to call 8.</p>','<p>继续往后看到第 8 次。</p>'):'');
  const tf=resSends(r,'tests/test_cart.py');
  s+=tx(`<p><b>All ${r.n} calls together: the model reads ${fT(r.totalIn)} tokens, ${fixd(r.totalIn/r.finalIn,1)} times the last call.</b></p>`,`<p><b>${r.n} 次调用加起来，模型一共读进去 ${fT(r.totalIn)} 个 token，是最后一次的 ${fixd(r.totalIn/r.finalIn,1)} 倍。</b></p>`)+verdict();
  s+=tx(`<p>Why: the model does not remember what it was sent, so whatever was sent once goes out again with every later call. The system prompt and the tool definitions go out with all ${r.n} calls. The model asked for the test file in call ${tf.asked}; once your code had read it, the file went out with every call from call ${tf.first} on, ${tf.times} times in all.</p><p>So the tokens read are every bar added up, the whole area of the bars:</p>`,
        `<p>原因是模型不记得之前收到过什么。发过一次的内容，后面每次调用都要再发一遍。system prompt 和工具说明，${r.n} 次调用每次都发。测试文件是模型在第 ${tf.asked} 次调用里要的。你的代码把它读出来以后，从第 ${tf.first} 次调用起每次都发，一共发了 ${tf.times} 遍。</p><p>所以模型读进去的 token，是每根柱子加起来，也就是图里所有柱子的总面积：</p>`)+
    M(tx('\\text{tokens read} = \\sum_{i=1}^{8} \\text{input}_i','\\text{读进去的 token} = \\sum_{i=1}^{8} \\text{输入}_i'))+
    tx(`<p>${m('\\text{input}_i')} is everything call i sends: the system prompt, the tool definitions and every message before it.</p>`,`<p>${m('\\text{输入}_i')} 是第 i 次调用发出的全部内容：system prompt、工具说明，加上它之前的每一条消息。</p>`);
  /* the owner (2026-10-14): resending the conversation between calls is not the same thing as decode reading the KV cache for every token
     inside one call; and the bill counts each call's input plus its output */
  s+=tx(`<p><b>The resending happens between calls.</b> Within one call, the input is computed only once: the model keeps each token’s intermediate results (the ${DEF('KV cache',TIP.kv())}) in GPU memory and uses them directly for every token it then writes, with no second computation of the input and no extra charge. For each call, the provider charges for the input tokens and the output tokens, each at its own price: for this task, ${fT(r.totalIn)} input tokens and ${fT(r.totalOut)} output tokens.</p>`,
        `<p><b>重发发生在两次调用之间。</b>一次调用之内，输入只算一遍：模型把每个 token 的中间结果（${DEF('KV cache',TIP.kv())}）存在显存里，之后每写一个 token 都直接用它，不再重算输入，也不另收钱。服务商每次调用按输入 token 和输出 token 各自的单价收费：这个任务输入共 ${fT(r.totalIn)} 个，输出 ${fT(r.totalOut)} 个。</p>`);
  return s;
}
function obsFail(){
  const r=fxRun(fx), b=runBase();
  if(fx==='std') return tx(`<p><b>Send the error back: the model reads “assert 80.0 == 81.0”, gets the fix right on the next call, and the task is done at call ${b.n}.</b></p><p>The error tells the model two things: which test failed, and by how much. 80 instead of 81 says the two discounts must apply one after the other, not be added.${fxTried.size<3?' Now try the other two ways.':''}</p>`,
                        `<p><b>把报错发回去：模型读到“assert 80.0 == 81.0”，下一次调用就改对了，第 ${b.n} 次调用时任务完成。</b></p><p>报错告诉模型两件事：哪个测试没过，差了多少。是 80 而不是 81，说明两个折扣应该先后相乘，而不是相加。${fxTried.size<3?'再试试另外两种做法。':''}</p>`);
  if(fx==='stop') return tx(`<p><b>Stop on any tool error: the task ends after call ${r.n}. The test still fails, and cart.py keeps the wrong first edit.</b></p><p>A failing test is exactly what the model needs to see. Treating it as a crash gives up just before the fix.</p>`,
                            `<p><b>工具一出错就停下：任务在第 ${r.n} 次调用后结束。测试还是没过，cart.py 里留着第一次改错的那一行。</b></p><p>测试没通过，正是模型最需要知道的事。把它当成故障停下，等于在快要修好的时候放弃。</p>`);
  return tx(`<p><b>Only “exit code 1”: the model does not know which test failed or why. It runs the test again and rereads the test file, and fixes it at call ${r.n} instead of ${b.n}.</b></p><p>The two extra calls, and what every later call resends, add ${fT(r.totalIn-b.totalIn)} tokens read. Send the real error back.</p>`,
            `<p><b>只发回“exit code 1”：模型不知道是哪个测试、差在哪里，只好再跑一次、再读一遍测试文件，第 ${r.n} 次调用才修好，而不是第 ${b.n} 次。</b></p><p>多出来的两次调用，加上之后每一次都要重发的内容，让读进去的 token 多了 ${fT(r.totalIn-b.totalIn)} 个。报错要原样发回去。</p>`);
}
function obsTime(){
  const s=parRun('seq'), p=parRun('par'), sl=build('par',{...S,tt:10}), slow=sl.tTool>sl.tModel;
  if(par==='seq') return tx(`<p><b>One tool per call: the whole task takes about ${fS(s.time)} s, and model calls take ${fS(s.tModel)} s of it (${fPc(s.tModel/s.time)}).</b></p><p>Now ask for the three things at once.</p>`,
                            `<p><b>一次一个：整个任务约 ${fS(s.time)} 秒，其中模型调用占 ${fS(s.tModel)} 秒（${fPc(s.tModel/s.time)}）。</b></p><p>再试试一次要齐三样。</p>`);
  return tx(`<p><b>Three tools in one call: ${p.n} calls instead of ${s.n}; the task takes about ${fS(p.time)} s instead of ${fS(s.time)} s.</b></p><p>Most of the saving is the two model calls that are gone. The three tools run at the same time, so that step waits only for the slowest. The model also reads fewer tokens, ${fT(p.totalIn)} instead of ${fT(s.totalIn)}, because two calls no longer resend the conversation.</p>`,
            `<p><b>一次要 3 个：调用从 ${s.n} 次减到 ${p.n} 次，整个任务约 ${fS(p.time)} 秒，原来约 ${fS(s.time)} 秒。</b></p><p>省下的主要是少了两次模型调用。三个工具同时跑，这一步只等最慢的那个。模型读进去的 token 也少了，从 ${fT(s.totalIn)} 个降到 ${fT(p.totalIn)} 个，因为少了两次重发整段对话。</p>`)+
    tx(`<p>This lab assumes ${m('t_0')} = ${fS(S.lat)} s, ${m('v_\\text{prefill}')} = ${fint(S.pre)} tokens/s for reading and ${m('v_\\text{decode}')} = ${S.dec} tokens/s for writing (${J('controls','Settings')}).${slow?' Slow tests change this: if one test run takes 10 s, the tools take more time than the model calls.':''}</p>`,
       `<p>这个 Lab 假设 ${m('t_0')} = ${fS(S.lat)} 秒，读的速度 ${m('v_\\text{prefill}')} 是每秒 ${fint(S.pre)} 个 token，写的速度 ${m('v_\\text{decode}')} 是每秒 ${S.dec} 个 token（见${J('controls','自由调参')}）。${slow?'测试很慢时就不一样了。如果跑一遍测试要 10 秒，工具花的时间就会比模型调用还多。':''}</p>`);
}
const REMS=['who','memory','fail','time'];
function remBtn(k){const on=remView===k;
  const tip={who:tx('Shows the whole bug-fix run in the chart.','在图里显示修 bug 的整次运行。'),memory:tx('Shows the bug fix at call 8, with every call’s bar.','在图里显示修 bug 的第 8 次调用，和每次调用的柱子。'),
    fail:tx('Shows the run with a failed test, the failed test marked.','在图里显示有一个测试没通过的运行，并标出这个测试。'),time:tx('Shows the time line when the first call asks for three tools.','在图里显示第 1 次调用就要 3 个工具时的时间线。')}[k];
  return ` <button type="button" class="linkbtn ana-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const r=runBase(), p=parRun('par'), nt=r.calls.reduce((a,c)=>a+c.res.length,0), tf=resSends(r,'tests/test_cart.py');
  return tx(`<ul class="ana-remlist"><li><b>The model decides; your code acts.</b> The model only sends tool_use blocks, a tool’s name and its arguments. Every tool runs in your code, so your code decides what reaches the model and when the loop stops.${remBtn('who')}</li>`+
    `<li><b>Every call resends the whole conversation, so cost follows the sum.</b> Each call sends everything before it again. So the more calls, the faster the tokens read add up, far past the length of the last conversation.${remBtn('memory')}</li>`+
    `<li><b>When a tool fails, hand the real error to the model.</b> The error tells the model which test failed and by how much, which is what it needs to fix the code. Stopping on the error gives up just before the fix.${remBtn('fail')}</li>`+
    `<li><b>Under this lab’s assumptions, model calls take most of the time. Have the model ask for independent tools at once.</b> Each model call is slow, so a call saved saves the most time. Asking for independent tools together saves calls, and the tools run at the same time.${remBtn('time')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “The model remembers what it did in the last step.” It does not. Whatever the model has seen has to be sent again in every later call. Without it, the model would not know what the test says.</p><p>✗ “The model runs the commands itself.” The model only returns tool_use data; your code runs the tests. That is why the same test result, handed back in different ways, can end the task in different ways.</p><p>✗ “The token cost depends mostly on how long the last conversation is.” The last conversation is only a small part of the total. The more calls, the more often everything before them is resent.</p></div>${A.shell.toScenarios()}`,
    `<ul class="ana-remlist"><li><b>模型做决定，你的代码去执行。</b>模型发出的只是 tool_use，也就是工具名和参数。每个工具都在你的代码里执行，所以发什么给模型、循环什么时候停，都由你的代码决定。${remBtn('who')}</li>`+
    `<li><b>每次调用都重发整段对话，花费看的是总和。</b>每次调用都把前面的内容再发一遍。所以调用越多，读进去的 token 总数涨得越快，远远超过最后那次对话的长度。${remBtn('memory')}</li>`+
    `<li><b>工具出错时，把报错原样交给模型。</b>报错告诉模型哪个测试没过、差了多少，这正是它改代码需要的。一出错就停下，等于在快要修好的时候放弃。${remBtn('fail')}</li>`+
    `<li><b>在这个 Lab 的假设下，时间主要花在模型调用上。互不依赖的工具，让模型一次要齐。</b>每次模型调用都很慢，所以少一次调用，省下的时间最多。互不依赖的工具一次要齐，能省掉几次调用，工具也能同时跑。${remBtn('time')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “模型会记得上一步做过什么。”它不记得。模型看过的内容，后面每次调用都要再发一遍。不发，模型就不知道测试写了什么。</p><p>✗ “是模型自己在执行命令。”模型只回了 tool_use 这段数据，跑测试的是你的代码。所以同一个测试结果，你的代码交回去的方式不同，任务的结局也就不同。</p><p>✗ “token 花费主要看最后那次对话有多长。”最后那次对话只占总数的一小部分。调用越多，前面的内容被重发的次数就越多。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 loop:'https://github.com/Schuture/Anthropic-Interview-Notes/blob/main/coding/llm-agent-tool-loop/README.md',
 mini:'https://github.com/SWE-agent/mini-swe-agent',
 miniAgent:'https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py',
 miniCfg:'https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/default.yaml',
 cache:'https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb',
 react:'https://arxiv.org/abs/2210.03629',
 swe:'https://arxiv.org/abs/2405.15793'
};
const PRESETS=[
 {id:'lim',t:{en:'A call limit of 5',zh:'调用次数上限是 5'},d:{en:'The same bug fix; your code allows at most 5 model calls',zh:'同一个修 bug 任务，你的代码最多只调用 5 次模型'},s:{run:'base',lim:5}},
 {id:'par',t:{en:'Three tools in one call',zh:'一次要 3 个工具'},d:{en:'The same bug fix; the first call asks for three things at once',zh:'同一个修 bug 任务，第 1 次调用就把三样东西一起要过来'},s:{run:'par',lim:0}},
 {id:'long',t:{en:'A rename across 13 files',zh:'在 13 个文件里改名'},d:{en:'A longer task: 30 model calls',zh:'更长的任务：要调用 30 次模型'},s:{run:'long',lim:0}},
 {id:'huge',t:{en:'A test run with full logs',zh:'跑测试时打开了详细日志'},d:{en:'The same bug fix; one test run returns a very long log',zh:'同一个修 bug 任务，有一次跑测试返回了很长的日志'},s:{run:'huge',lim:0}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.run===S.run&&p.s.lim===S.lim);
const curRun=()=>build(S.run,S);
function scenHTML(i){
  const p=PRESETS[i], r=curRun(), b=build('base',{...S,lim:0});   // the bug fix with no call limit, same speeds
  if(p.id==='lim') return tx(`<p><b>With at most 5 calls, your code stops after call 5: the tests have just failed, and the fix never comes.</b></p><p>A call limit is a stop the model cannot get around: once it is reached, your code makes no more model calls, whatever the model still wants. mini-swe-agent has the same setting, step_limit. Set the limit from how many calls real tasks need; this task needs ${b.n}.</p>`,
                             `<p><b>最多 5 次调用：你的代码在第 5 次调用后停下。这时测试刚刚没通过，修复还没来得及做。</b></p><p>调用次数上限是一种模型绕不过去的停止条件。到了上限，你的代码就不再调用模型，不管模型还想不想要工具。mini-swe-agent 也有这个设置，叫 step_limit。上限要按真实任务需要的调用次数来定，这个任务要 ${b.n} 次。</p>`);
  if(p.id==='par') return tx(`<p><b>Three tools in one call: ${r.n} calls instead of ${b.n}, and ${fT(r.totalIn)} tokens read instead of ${fT(b.totalIn)}.</b></p><p>The three tool_result blocks go back in one message, each with the id of the tool_use it answers. The last call sends as much as before: what is saved is two whole resends of the conversation.</p>`,
                             `<p><b>一次要 3 个工具：调用从 ${b.n} 次减到 ${r.n} 次，读进去的 token 从 ${fT(b.totalIn)} 个减到 ${fT(r.totalIn)} 个。</b></p><p>三个 tool_result 放在同一条消息里发回去，各自带着它回答的那个 tool_use 的 id。最后一次调用发出的和原来一样多，省下的是两次重发整段对话。</p>`);
  if(p.id==='long') return tx(`<p><b>This task makes ${r.n} calls. The last call sends only ${fT(r.finalIn)} tokens, yet the model reads ${fK(r.totalIn)} in all, ${fixd(r.totalIn/r.finalIn,1)} times the last call.</b></p><p>For the bug fix it was only ${fixd(b.totalIn/b.finalIn,1)} times. The longer the task, the more often each early block is resent, so the total grows much faster than the conversation. Long-running agents therefore drop or shorten old tool results (${labLink('context-eng')}).</p>`,
                              `<p><b>这个任务调用了 ${r.n} 次。最后一次只发出 ${fT(r.finalIn)} 个 token，模型却一共读了 ${fK(r.totalIn)} 个，是最后一次的 ${fixd(r.totalIn/r.finalIn,1)} 倍。</b></p><p>修 bug 的任务只有 ${fixd(b.totalIn/b.finalIn,1)} 倍。任务越长，前面的每一块被重发的次数就越多，所以总数比对话本身长得快得多。跑得久的 agent 因此会删掉或缩短旧的工具结果（${labLink('context-eng')}）。</p>`);
  const big=r.blocks.find(x=>x.kind==='res'&&x.tok>20000);
  return tx(`<p><b>With full logs on, one test run returns a log of ${fT(big.tok)} tokens. The ${r.n} calls read ${fK(r.totalIn)} tokens in all, against ${fT(b.totalIn)} without the long log.</b></p><p>Every later call resends the log, and it is most of what each of them sends. One tool’s output can fill the conversation, so harnesses cut long outputs down, or keep only their start and end, before they send them back.</p>`,
            `<p><b>跑测试时打开了详细日志，这一次就返回了 ${fT(big.tok)} 个 token 的日志。${r.n} 次调用一共读了 ${fK(r.totalIn)} 个 token，没有这份长日志时只有 ${fT(b.totalIn)} 个。</b></p><p>之后每次调用都要重发这份日志，它占了每次发出内容的一大半。一个工具的输出就能把对话撑满，所以 harness 会先把太长的输出截短，或者只留开头和结尾，再发回去。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const planName=(l,b)=>tx(`limit ${l} calls, budget ${fT(b)} tokens`,`上限 ${l} 次，预算 ${fT(b)} 个 token`);
const stopWhy=w=>({limit:tx('the call limit','调用次数上限'),budget:tx('the token budget','token 预算')}[w]||'');
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chO.lim,chO.bud), best=chBest(), nr=build('chN',{});
  let h;
  const good=tx(`<p class="okline ana-mt"><b>Goal met: the normal run finishes (${callsW(me.nCalls)}, ${fT(me.nRead)} tokens read). The stuck run is stopped by ${stopWhy(me.sStop)} after ${callsW(me.sCalls)}, having read ${fT(me.cost)} tokens.</b>`,
                `<p class="okline ana-mt"><b>达成目标：正常的运行做完了（${callsW(me.nCalls)}，读了 ${fT(me.nRead)} 个 token）。卡住的运行在 ${callsW(me.sCalls)}后被${stopWhy(me.sStop)}拦下，已经读了 ${fT(me.cost)} 个 token。</b>`);
  if(chMinimal(me)) h=good+tx(' And no setting that lets the normal run finish stops the stuck one sooner.</p>','而且在能让正常的运行做完的设置里，这已经是最省的。</p>');
  else if(me.ok){const sg=chSuggest(me); h=good+tx(' But a tighter setting also works.</p>','不过还有更省的设置。</p>')+tx(`<p>With ${esc(planName(sg.lim,sg.bud))}, the stuck run reads only ${fT(sg.cost)} tokens before it stops.</p>`,`<p>改成${esc(planName(sg.lim,sg.bud))}，卡住的运行只读 ${fT(sg.cost)} 个 token 就停下。</p>`);}
  else h=tx(`<p class="badline ana-mt"><b>Not yet: the normal run is stopped by ${stopWhy(me.nStop)} after ${callsW(me.nCalls)}, before it is done.</b></p>`,`<p class="badline ana-mt"><b>还没达到：正常的运行在 ${callsW(me.nCalls)}后被${stopWhy(me.nStop)}拦下，没能做完。</b></p>`)+
      tx(`<p class="small">Hint: the budget must hold the tokens of all ${nr.n} calls added up, not the size of the last one. Each call sends a bit more than the one before.</p>`,
         `<p class="small">提示：预算要装下 ${nr.n} 次调用读进去的 token 的总和，而不是最后一次有多长。每一次调用都比上一次多发一点。</p>`)+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the owner's review of batch 8 (2026-10-07): the chart keeps the settings that let the normal run finish, the pick and the initial setup (brief) */
  const rows=chAll(), isMe=e=>e.lim===chO.lim&&e.bud===chO.bud, isStart=e=>e.lim===CH.start.lim&&e.bud===CH.start.bud, shown=rows.filter(e=>e.ok||isMe(e)||isStart(e)), mx=Math.max(...shown.map(e=>e.cost));
  h+=A.ui.cmpChart({title:tx('Tokens the stuck run reads before it is stopped (green: the normal run finishes)','卡住的运行停下之前读了多少 token（绿色：正常的运行能做完）'),
    aria:tx('Tokens the stuck run reads before it is stopped: ','卡住的运行停下之前读的 token：')+shown.map(e=>`${planName(e.lim,e.bud)}${tx(': ','：')}${fT(e.cost)}${e.ok?tx(', the normal run finishes','，正常的运行能做完'):''}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:(e.ok?'✓ ':'')+planName(e.lim,e.bud),value:e.cost,text:fK(e.cost),ok:e.ok,me:e.lim===chO.lim&&e.bud===chO.bud,
      tags:(e.lim===CH.start.lim&&e.bud===CH.start.bud?[{text:tx('initial setup','初始设置')}]:[]).concat(e.lim===chO.lim&&e.bud===chO.bud?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,hiddenText:n=>tx(`The other ${n} settings stop the normal run too early and are not shown.`,`其余 ${n} 种设置会把正常的运行拦得太早，图里没有列出。`),
    max:mx*1.05,okText:tx('✓ the normal run finishes','✓ 正常的运行能做完'),noText:tx('it is stopped too early','被拦得太早'),legend:true});
  const st=chEval(CH.start.lim,CH.start.bud);
  h+=tx(`<p class="small">The initial setup (limit 50 calls, budget 300,000 tokens) lets the stuck run read ${fT(st.cost)} tokens.</p><p class="small">Why a budget of ${fT(best.bud)} is cheapest: the normal run needs ${callsW(nr.n)} and reads ${fT(nr.totalIn)} tokens in all. Of the four options, ${fT(best.bud)} is the smallest budget that lets it finish, and ${best.lim} the smallest call limit. With that budget, the stuck run is stopped after ${callsW(best.sCalls)}, whatever the call limit (${best.lim} or more). Each call it is allowed costs more than the one before, because its conversation keeps growing.</p>`,
        `<p class="small">初始设置（上限 50 次，预算 300,000 个 token）下，卡住的运行会读 ${fT(st.cost)} 个 token。</p><p class="small">为什么预算 ${fT(best.bud)} 最省：正常的运行要调用 ${nr.n} 次，一共读 ${fT(nr.totalIn)} 个 token。四个选项里，能让它做完的预算最小是 ${fT(best.bud)}，调用次数上限最小是 ${best.lim}。有了这个预算，卡住的运行调用 ${best.sCalls} 次就会停下，上限设 ${best.lim} 还是更多都一样。卡住的运行对话越来越长，多放它调用一次，就比上一次更贵。</p>`);
  h+=tx('<p class="small muted">Note: real tasks vary. Set the limits from a batch of real runs (for example, so that 95% of normal runs finish), and look again when the prompts or the tools change.</p>',
        '<p class="small muted">注意：真实任务各不相同。上限要按一批真实运行来定（例如让 95% 的正常运行做得完），提示词或工具变了要重新看。</p>');
  return h;
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Predict first, then watch one bug fix by an agent: who does what, how many tokens each call sends, what happens when a tool fails, and where the time goes. Start here if this is new to you.',
            '一共 6 个部分。先预测，再看一个 agent 修 bug 的全过程：谁在做事、每次调用发出多少 token、工具出错时怎么办、时间花在哪里。第一次学从这里开始。'),
  sandbox:tx('Four situations: a tight call limit, three tools in one call, a task of 30 calls, and one huge tool output. Pick one to see how the conversation and the tokens change.',
             '四种情况，分别是调用次数上限太紧、一次要 3 个工具、要调用 30 次的长任务、一个特别长的工具输出。选一个，看对话和 token 怎么变。'),
  challenge:tx('Set a call limit and a token budget for a new task: the normal run must finish, and a stuck run must be stopped as early as possible.',
               '给一个新任务设调用次数上限和 token 预算，让正常的运行能做完，卡住的运行尽早停下。')
});
const FIDPOP=()=>tx(
 `<p><b>Illustrative:</b> this lab constructs one run in the real message format of a tool-use API.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The message format: system and tools sent with every call; tool_use blocks with an id, a name and an input; tool_result blocks with tool_use_id, content and is_error; stop_reason tool_use or end_turn (as in the interview problem under “Go further”).</li>`+
 `<li>Every call sends every block so far, so a call’s input is the sum of all blocks before it; the tokens read are the sum over calls.</li>`+
 `<li>Limits are checked before each call, as mini-swe-agent’s <code>query()</code> checks its step and cost limits.</li></ul></div>`+
 `<div><div class="eyebrow">Assumed</div><ul>`+
 `<li>Tokens: this lab counts about 4 characters per token, per block, rounded up, and counts the blocks’ content only. Real counts depend on the tokenizer, and the message format adds a few tokens per message.</li>`+
 `<li>The sizes of the system prompt (12,000 characters), the tool definitions and the files are the lab’s choices. Real agents vary a lot. mini-swe-agent’s system message and task template come to only about 3,000 characters together. An agent with many tools can spend far more on tool definitions alone.</li>`+
 `<li>Time: each model call takes 0.5 s plus reading at 10,000 tokens/s plus writing at 50 tokens/s; a test run takes 3 s. All are assumptions you can change in Settings. Real APIs vary with load and model size; a slow test suite can make tools the larger share.</li>`+
 `<li>Billing: a provider bills each call for its input tokens plus its output tokens, so a task’s bill adds up all its calls. This lab counts every resent input token as a normal input token. With prompt caching, a provider reads the repeated start of the conversation from a cache and bills it at a lower price (one provider’s cookbook prices a cache read at 0.1 times a normal input token); those tokens are still sent and counted.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>The loop and the message format: the interview problem ${cite(SRC.loop,'coding/llm-agent-tool-loop')}.</li>`+
 `<li>A real minimal agent: ${cite(SRC.miniAgent,'SWE-agent/mini-swe-agent <code>agents/default.py</code>')} (checked 2026-10-03) has <code>step_limit</code>, <code>cost_limit</code> and <code>wall_time_limit_seconds</code>; its README says the trajectory and the messages sent to the model are the same.</li>`+
 `<li>mini-swe-agent’s default config, ${cite(SRC.miniCfg,'<code>config/default.yaml</code>')} (checked 2026-10-03): step_limit and cost_limit are 0, which means no limit; the system and task templates come to about 3,000 characters.</li>`+
 `<li>Prompt caching prices: a provider’s cookbook, ${cite(SRC.cache,'<code>misc/prompt_caching.ipynb</code>')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-03. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>示意：</b>本 Lab 按真实的工具调用 API 消息格式构造了一次运行。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>消息格式：system 和工具说明随每次调用一起发；tool_use 块带 id、name 和 input；tool_result 块带 tool_use_id、content 和 is_error；stop_reason 是 tool_use 或 end_turn（和“延伸练习”里那道面试题一样）。</li>`+
 `<li>每次调用都发出之前的全部内容，所以一次调用的输入是它之前所有块的总和；模型读进去的 token，是每次调用加起来。</li>`+
 `<li>上限在每次调用之前检查，和 mini-swe-agent 的 <code>query()</code> 检查它的步数上限和花费上限一样。</li></ul></div>`+
 `<div><div class="eyebrow">假设</div><ul>`+
 `<li>token：本 Lab 每一块按大约 4 个字符算 1 个 token，向上取整，只数各块的内容。真实的数取决于分词器，消息格式本身每条还会多几个 token。</li>`+
 `<li>system prompt（12,000 个字符）、工具说明和各个文件的长度，是这个 Lab 设定的。真实的 agent 差别很大。mini-swe-agent 的 system 消息和任务模板加起来只有约 3,000 个字符。工具多的 agent，光是工具说明就可能长得多。</li>`+
 `<li>时间：每次模型调用 0.5 秒，加上每秒读 10,000 个 token、每秒写 50 个 token；跑一遍测试 3 秒。这些都是假设，可以在自由调参里改。真实 API 的速度随负载和模型大小变化；测试很慢时，工具的时间也可能占大头。</li>`+
 `<li>计费：服务商对每次调用收它的输入 token 加输出 token，一次任务的账单是所有调用加起来。本 Lab 把每个重发的输入 token 都算作普通输入 token。用了 prompt caching 的服务商，从缓存里读对话里重复的开头，按更低的价格计费（一家服务商的 cookbook 里，读缓存的价格是普通输入 token 的 0.1 倍）；这些 token 照样要发、照样计数。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>循环和消息格式：面试题 ${cite(SRC.loop.replace('README.md','README.zh.md'),'coding/llm-agent-tool-loop')}。</li>`+
 `<li>一个真实的最小 agent：${cite(SRC.miniAgent,'SWE-agent/mini-swe-agent 的 <code>agents/default.py</code>')}（2026-10-03 核对），里面有 <code>step_limit</code>、<code>cost_limit</code> 和 <code>wall_time_limit_seconds</code>；它的 README 说，运行记录和发给模型的消息是同一份。</li>`+
 `<li>mini-swe-agent 的默认配置 ${cite(SRC.miniCfg,'<code>config/default.yaml</code>')}（2026-10-03 核对）里，step_limit 和 cost_limit 都是 0，也就是不限。system 和任务两个模板加起来约 3,000 个字符。</li>`+
 `<li>prompt caching 的价格：一家服务商的 cookbook，${cite(SRC.cache,'<code>misc/prompt_caching.ipynb</code>')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-03。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('read an agent’s trace, tell who did what and where its tokens and time went, and set its call and token limits.','读懂一个 agent 的调用记录，说清谁做了什么、token 和时间花在哪里，并给它设好调用次数和 token 的上限。');
const EXITS=()=>{
  const b=runBase(), lg=build('long',S), tf=resSends(b,'tests/test_cart.py');
  const t20=20*(4000+24000)/2;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx(`Call ${tf.first} already sent the test file to the model. Why does call ${b.n} send it again?`,`第 ${tf.first} 次调用已经把测试文件发给了模型，为什么第 ${b.n} 次调用还要再发一遍？`),
   o:[tx('The model keeps nothing between calls; all it can use is the conversation sent with this call.','模型在两次调用之间什么都不保留；它能用的，只有这一次发过去的对话。'),
      tx('The model remembers the file; sending it again only makes the answer surer, and could be skipped.','模型记得这个文件；再发一遍只是让回答更有把握，可以省掉。'),
      tx('Sending the file again makes the tool read it again, so the model gets the newest version.','再发一遍，工具就会再读一次文件，模型才能拿到最新的版本。')],a:0,
   n:[null,tx('Nothing carries over between calls: without the file in this call’s conversation, the model would not know what the test says.','两次调用之间什么都不会留下：这次的对话里没有这个文件，模型就不知道测试写了什么。'),
     tx('Sending a conversation runs no tool. Tools run only when your code runs them, after a tool_use; the old result is simply sent again.','发送对话不会执行任何工具。工具只在模型发出 tool_use、你的代码去执行时才运行；这里只是把旧的结果再发一遍。')],
   why:tx('Each call is independent: the model sees only what this call sends. That is why the conversation is resent in full, and why the tokens read add up over calls.','每次调用都是独立的：模型只看得见这一次发过去的内容。所以要把整段对话重发一遍，读进去的 token 也就一次次加起来。')},
  {q:tx('The model’s reply is a tool_use for run_tests. Who runs the tests?','模型回复了一个调用 run_tests 的 tool_use。测试是谁跑的？'),
   o:[tx('Your code: it reads the tool_use, calls its own run_tests with the input, and sends the output back as a tool_result.','你的代码：它读到 tool_use，用其中的参数调用自己的 run_tests，再把输出作为 tool_result 发回去。'),
      tx('The model’s server runs the tests and writes the output into its next reply.','模型那边的服务器跑了测试，把输出写进下一次回复里。'),
      tx('The model runs the tests while it writes the tool_use, so the result comes with the reply.','模型在写 tool_use 的同时就跑了测试，所以结果跟着回复一起回来。')],a:0,
   n:[null,tx('Nothing runs on the model’s side: every tool call is run by your code, between two model calls.','模型那边什么都没执行：每一次工具调用，都是你的代码在两次模型调用之间执行的。'),
     tx('A reply with stop_reason tool_use ends there and waits; the result exists only after your code runs the tool and sends it in the next call.','stop_reason 是 tool_use 的回复到此就结束了，在等结果；你的代码执行完工具、在下一次调用里发回去，结果才出现。')],
   why:tx('The model only writes data: a tool name and arguments. Running it, and deciding what to do with a failure, is the harness’s job.','模型只写出数据：工具名和参数。执行它，以及失败了怎么办，都是 harness 的事。')}],
 transfer:[
  {q:tx(`Another agent makes 20 calls. Its first call sends about 4,000 tokens and its last about 24,000, growing evenly. About how many tokens does it read in all? (For reference: in this lab, ${b.n} calls from ${fT(b.calls[0].inTok)} to ${fT(b.finalIn)} read ${fT(b.totalIn)}.)`,
      `另一个 agent 调用 20 次。第 1 次发出约 4,000 个 token，最后一次约 24,000 个，中间均匀增长。它一共读进去大约多少 token？（参照：这个 Lab 里 ${b.n} 次调用、从 ${fT(b.calls[0].inTok)} 涨到 ${fT(b.finalIn)}，一共读了 ${fT(b.totalIn)} 个。）`),
   o:[tx('About 24,000','约 24,000 个'),tx('About 280,000','约 280,000 个'),tx('About 480,000','约 480,000 个')],a:1,
   why:tx(`About ${fT(t20)}: 20 calls of about 14,000 tokens on average, the middle of 4,000 and 24,000. 24,000 is only the last call; 480,000 would mean every call sent as much as the last.`,
          `约 ${fT(t20)} 个：20 次调用，平均每次约 14,000 个 token，也就是 4,000 和 24,000 的中间。24,000 只是最后一次；480,000 等于假设每次都发得和最后一次一样多。`)},
  {q:tx('A tool returns a log of 30,000 tokens, and the agent then makes 10 more calls. About how many tokens does that one log add to what the model reads?','一个工具返回了 30,000 个 token 的日志，之后 agent 又调用了 10 次模型。光这份日志，就让模型多读了大约多少 token？'),
   o:[tx('About 30,000','约 30,000 个'),tx('About 300,000','约 300,000 个'),tx('About 3,000,000','约 3,000,000 个')],a:1,
   why:tx('Every one of the 10 later calls sends the whole conversation, the log included: 10 × 30,000 = 300,000. That is why harnesses cut long outputs before sending them back.','之后的 10 次调用，每一次都要发整段对话，这份日志也在里面：10 × 30,000 = 300,000。所以 harness 会先把太长的输出截短，再发回去。')}],
 practice:[
  {id:'agent-tool-loop',d:tx('Parts 1 and 2: a first agent loop has a few bugs. Fix them, then add a cap on calls. The fixes are the rules this lab showed, for example: put the model’s own reply back into the conversation, and answer every tool_use with a tool_result that carries the same id. When you are done, you have written this lab’s loop yourself.','第 1、2 部分：一个初版的 agent 循环里有几处 bug，先把它们修好，再加上调用次数上限。要修的正是这个 Lab 讲过的规矩，比如模型的回复要原样放回对话里，每个 tool_use 都要有一个带同样 id 的 tool_result。做完，你就亲手写出了这个 Lab 里的循环。')},
  {id:'agent-harness-eval',d:tx('A system design: the harness that runs many teams’ agents. It asks for this lab’s loop, its stop conditions (the challenge), how a tool’s errors reach the model (part 4), and a record of every run’s full trace.','系统设计题：设计一套供多个团队共用的 agent 执行框架（harness）。题目要讲的正是这个 Lab 的循环、停止条件（挑战题）、工具的报错怎么交回模型（第 4 部分），以及怎样记录每次运行的完整轨迹。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'agents',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Inside an agent task','Agent 任务拆解'),title:l?L(l.q):tx('What actually happens inside one agent task?','Agent 一次任务里到底发生了什么？'),
    minutes:12,fid:'Conceptual',fidPop:FIDPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage ana-stage" id="ana-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="ana-st-title"></div><div class="meter-sub" id="ana-st-sub"></div></div><div class="stage-actions"><span class="status" id="ana-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph ana-graph" id="ana-graph">`+
    `<div class="ana-phone" id="ana-phone"></div>`+
    `<div class="ana-loopbox" id="ana-loopbox"><div class="ana-loop" id="ana-loop"></div><div class="ana-cap2" id="ana-cap2" aria-live="polite"></div></div>`+
    `<div class="ana-player" id="ana-player"></div>`+
    `<div class="ana-convbox" id="ana-convbox"><div class="ana-cap" id="ana-conv-cap"></div><div class="ana-leg" id="ana-conv-leg"></div><div class="ana-conv" id="ana-conv"></div></div>`+
    `<div class="ana-barsbox" id="ana-barsbox"><div class="ana-cap" id="ana-bars-cap"></div><div class="ana-leg" id="ana-bars-leg"></div><div class="ana-bars" id="ana-bars"></div></div>`+
    `<div class="ana-timebox" id="ana-timebox" hidden><div class="ana-cap" id="ana-time-cap"></div><div class="ana-leg" id="ana-time-leg"></div><div class="ana-time" id="ana-time"></div></div>`+
    `<div class="ana-chbox" id="ana-chbox" hidden><div class="ana-cap" id="ana-ch-cap"></div><div class="ana-leg" id="ana-ch-leg"></div><div class="ana-chart" id="ana-chart"></div></div>`+
   `</div>`+
   `<div class="metrics ana-metrics" id="ana-metrics"></div>`+
   `<details class="math" id="ana-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ana-mathbody"></div></details></div>`;
}
const NOTE={
 lim:()=>{const nb=build('base',{}).n; return tx(`Your code makes no more model calls once this many have been made. A model can keep asking for tools and never stop by itself. Without a call limit, a token budget or another limit of this kind, nothing stops a stuck run. Whether to set one, and how high, is up to whoever writes the harness. mini-swe-agent, for example, leaves step_limit and cost_limit at 0 in its default config, meaning no limit. This bug fix needs ${nb} calls.`,
            `调用次数到了这个数，你的代码就不再调用模型。模型可能一直要工具，自己停不下来。没有调用次数上限、token 预算这类限制，卡住的运行就没有东西能让它停下。设不设、设多少，由写 harness 的人决定。比如 mini-swe-agent 的 step_limit 和 cost_limit，在默认配置里都是 0，也就是不限。这个修 bug 的任务要调用 ${nb} 次。`);},
 pre:()=>tx('Assumed: how many tokens of input the model reads per second before it starts to reply (prefill). It depends on the model’s size, the hardware and the load. Common values range from a few thousand to around 50,000 tokens per second; 10,000 is used here.',
            '假设值：模型开始回复之前，每秒能读多少个输入 token（prefill）。它取决于模型大小、硬件和负载，常见的从每秒几千个到 50,000 个左右不等，这里取 10,000。'),
 dec:()=>tx(`Assumed: how many tokens the model writes per second (decode). Writing is slower than reading. Reading takes the whole input in one step. Writing takes one more step for every new token, and each step reads all the weights and the KV cache from GPU memory (see ${labLink('prefill-decode')}). Typical hosted speeds are tens to a hundred or so tokens per second; 50 is a middle value.`,
            `假设值：模型每秒写出多少个 token（decode）。写比读慢。读的时候，整段输入在一步里就能读完。写的时候，每写一个 token 都要单独走一步，为这一个 token 把全部权重和 KV cache 从显存里读一遍（见${labLinkZ('prefill-decode')}）。托管模型通常每秒几十到一百多个，这里取中间值 50。`),
 lat:()=>tx('Assumed: the fixed cost of one call, before any token is read: the network round trip and the wait in the provider’s queue. It is often a few tenths of a second to a second; 0.5 s is used here.',
            '假设值：一次调用固定要花的时间，和读多少 token 无关，包括网络往返，以及在服务商队列里的等待。通常是零点几秒到一秒，这里用 0.5 秒。'),
 tt:()=>tx('Assumed: how long one run of the test suite takes. It depends entirely on your project; a small suite takes seconds, a large one minutes. With slow tests, tools can take more time than the model.',
           '假设值：跑一遍测试要多久。这完全取决于你们的项目：小的测试集几秒，大的要几分钟。测试很慢时，工具花的时间可能比模型还多。')
};
const SEGS={lim:[[0,{en:'None',zh:'不设'}],[5,{en:'5',zh:'5'}],[10,{en:'10',zh:'10'}]],pre:[[2000,{en:'2,000',zh:'2,000'}],[10000,{en:'10,000',zh:'10,000'}],[50000,{en:'50,000',zh:'50,000'}]],
  dec:[[25,{en:'25',zh:'25'}],[50,{en:'50',zh:'50'}],[100,{en:'100',zh:'100'}]],lat:[[0.2,{en:'0.2 s',zh:'0.2 秒'}],[0.5,{en:'0.5 s',zh:'0.5 秒'}],[1,{en:'1 s',zh:'1 秒'}]],tt:[[1,{en:'1 s',zh:'1 秒'}],[3,{en:'3 s',zh:'3 秒'}],[10,{en:'10 s',zh:'10 秒'}]]};
const SEGL={lim:{en:'Call limit',zh:'调用次数上限'},pre:{en:'Reading speed (tokens/s)',zh:'读的速度（token/秒）'},dec:{en:'Writing speed (tokens/s)',zh:'写的速度（token/秒）'},lat:{en:'Fixed cost per call',zh:'每次调用的固定开销'},tt:{en:'One test run',zh:'跑一遍测试'}};
const RUNOPT=[['base',{en:'Bug fix',zh:'修 bug'}],['par',{en:'Bug fix, 3 tools at once',zh:'修 bug，一次 3 个工具'}],['long',{en:'Rename in 13 files',zh:'13 个文件里改名'}],['huge',{en:'Bug fix, long test log',zh:'修 bug，长测试日志'}]];
const seg=(k,inline)=>A.ui.seg({id:`ana-ctl-${k}`,key:k,label:L(SEGL[k]),options:SEGS[k].map(([v,t])=>[String(v),L(t)]),value:String(S[k]),note:NOTE[k](),inline});
function controlsHTML(){
  return `<aside class="panel controls" id="ana-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ana-lock" hidden></p>`+
   `<div class="lockable" id="ana-w-run">${A.ui.seg({id:'ana-ctl-run',key:'run',label:tx('Task','任务'),options:RUNOPT.map(([v,t])=>[v,L(t)]),value:S.run})}</div>`+
   `<div class="lockable" id="ana-w-lim">${seg('lim')}</div>`+
   `<div class="ana-grp lockable" id="ana-w-time"><div class="ana-gh">${tx('Assumed speeds','假设的速度')}</div>${seg('pre')}${seg('dec')}${seg('lat')}${seg('tt')}</div>`+
   `<div class="lockable" id="ana-w-reset"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`def run_agent(task, tools, system, run_tool, max_calls=8):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": task}]}]
    for _ in range(max_calls):
        # every call sends the whole conversation again
        r = client.create(system=system, tools=tools,
                          messages=messages)
        if r.stop_reason == "end_turn":
            return r.content[0]["text"]
        messages.append({"role": "assistant",
                         "content": r.content})
        results = []
        for b in r.content:          # your code runs tools
            if b["type"] != "tool_use":
                continue
            out, err = run_tool(b["name"], b["input"])
            results.append({"type": "tool_result",
                            "tool_use_id": b["id"],
                            "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    return f"gave up after {max_calls} calls"

def tokens(blocks):                  # about 4 characters each
    return sum(-(-len(str(b)) // 4) for b in blocks)`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ana-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ana-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The whole loop: call the model, run every tool_use, send each tool_result back with its id, stop at end_turn or at the call limit.','整个循环：调用模型，执行每一个 tool_use，把每个 tool_result 带上 id 发回去，到 end_turn 或调用次数上限时停下。')}</p><pre class="code" id="ana-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ana-prod">`+
   tx(`<li>${cite(SRC.miniAgent,'mini-swe-agent, <code>src/minisweagent/agents/default.py</code>')}: <code>run()</code> puts the system message and the task first; each <code>step()</code> is <code>query()</code>, one model call that appends the reply, then <code>execute_actions()</code>, which runs the actions and appends their outputs. Its only tool is bash.</li>`+
      `<li>The same file stops a run before a model call once <code>step_limit</code> calls are made or <code>cost_limit</code> is spent (<code>LimitsExceeded</code>), and also has <code>wall_time_limit_seconds</code>: the two limits of the challenge, plus a clock. Its default config leaves all three at 0, which means no limit; you set the ones you want.</li>`+
      `<li>Providers offer prompt caching because every call resends the same beginning: the repeated part is read from a cache instead of being processed again, much cheaper and faster (${labLink('prefix-caching')} shows the mechanism inside a server).</li>`,
      `<li>${cite(SRC.miniAgent,'mini-swe-agent 的 <code>src/minisweagent/agents/default.py</code>')}：<code>run()</code> 先放 system 消息和任务；每个 <code>step()</code> 就是 <code>query()</code>（调用一次模型、把回复接在后面）加上 <code>execute_actions()</code>（执行动作、把输出接在后面）。它唯一的工具是 bash。</li>`+
      `<li>同一个文件在每次调用模型之前检查两个上限：调用次数到了 <code>step_limit</code>，或者花费到了 <code>cost_limit</code>，就停下（<code>LimitsExceeded</code>）。另外还有一个时间上限 <code>wall_time_limit_seconds</code>。这就是挑战题里的两个上限，再加一个时钟。默认配置里这三个都是 0，也就是不限，要用哪个就自己设。</li>`+
      `<li>因为每次调用都要重发同样的开头，服务商提供 prompt caching：重复的部分从缓存里读，不再重新处理，便宜得多，也快得多（服务器里面怎么做，见${labLinkZ('prefix-caching')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.react,'ReAct: Synergizing Reasoning and Acting in Language Models')}<span>${tx('Yao et al., 2022 (arXiv 2210.03629): the model alternates between reasoning and acting with tools, the pattern behind this loop.','Yao 等，2022（arXiv 2210.03629）：模型在思考和调用工具之间交替，这个循环就是这种模式。')}</span></li>`+
   `<li>${cite(SRC.swe,'SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering')}<span>${tx('Yang et al., 2024 (arXiv 2405.15793): a coding agent that edits repositories through tools; mini-swe-agent comes from the same team.','Yang 等，2024（arXiv 2405.15793）：一个通过工具修改代码仓库的 coding agent；mini-swe-agent 出自同一个团队。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ana-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ana-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ana-app m2" id="ana-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ana-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many tokens each call sends.','选一个场景，看看每次调用发出多少 token。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ana-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const nr=build('chN',{});
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Stop a stuck agent early, without stopping a good one','让卡住的 agent 尽早停下，又不拦住正常的')}</h3>`+
   `<p>${tx('Your team’s coding agent gets a new kind of task: add retries to an HTTP client and test them. Usually it finishes. Sometimes a flaky test makes it loop: it runs the tests, edits, runs them again, and never ends. Set a call limit and a token budget so that the normal run finishes, and a stuck run is stopped having read as few tokens as possible.',
            '你们团队的 coding agent 接到一类新任务：给一个 HTTP 客户端加上重试，并补上测试。通常它能做完。偶尔遇到一个时好时坏的测试，它就会陷进循环：跑测试、改代码、再跑测试，一直停不下来。请设一个调用次数上限和一个 token 预算：正常的运行要能做完；卡住的运行要尽早停下，停下之前读进去的 token 越少越好。')}</p>`+
   `<ul class="ana-asm small"><li>${tx(`The normal run makes ${nr.n} model calls. Its first call sends ${fT(nr.calls[0].inTok)} tokens and its last ${fT(nr.finalIn)}.`,`正常的运行调用 ${nr.n} 次模型：第 1 次发出 ${fT(nr.calls[0].inTok)} 个 token，最后一次发出 ${fT(nr.finalIn)} 个。`)}</li><li>${tx('Before each call your code checks both limits: calls made so far, and tokens read so far plus this call’s.','每次调用之前，你的代码检查两个上限：已经调用的次数，以及已经读的 token 加上这一次要发的。')}</li></ul>`+
   A.ui.seg({id:'ana-ic-chl',key:'chl',label:tx('Call limit','调用次数上限'),options:CH.lims.map(n=>[String(n),String(n)]),value:String(chO.lim),inline:true})+
   A.ui.seg({id:'ana-ic-chb',key:'chb',label:tx('Token budget (tokens read, all calls together)','token 预算（所有调用读进去的 token 加起来）'),options:CH.buds.map(n=>[String(n),fT(n)]),value:String(chO.bud),inline:true})+
   `<div class="goal mono ana-goal" id="ana-ch-g1"><span>${tx('Normal run','正常的运行')}</span><span><b id="ana-ch-n"></b></span></div>`+
   `<div class="goal mono ana-goal" id="ana-ch-g2"><span>${tx('Stuck run, tokens read before it stops','卡住的运行停下前读的 token')}</span><span><b id="ana-ch-s"></b>${tx(' (fewer is better)','（越少越好）')}</span></div>`+
   submit+`<div id="ana-ch-res" class="ana-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},who:{en:'Let the replay finish to continue (“Fast” takes a few seconds).',zh:'等回放播完才能继续（选“快”只要几秒）。'},
    memory:{en:'Step on to call 8 to continue.',zh:'看到第 8 次调用，才能继续。'},fail:{en:'Try the other two ways to continue.',zh:'另外两种做法都试一下，才能继续。'},time:{en:'Pick “Three tools in one call” to continue.',zh:'选“一次要 3 个”才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#ana-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#ana-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&((s.id==='who'&&replayDone)||s.id==='memory'||s.id==='fail'||s.id==='time'||!!observed[step]);
    const syncLv=(host,cur,tried,fn)=>$$(`#${host} [data-lever]`,rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===cur)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tried.has(v)||v===cur?fn(v):'';});
    syncLv('ana-ic-fx',fx,fxTried,fxRes); syncLv('ana-ic-par',par,parTried,parRes);
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ana-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(chO.lim,chO.bud), n=$('#ana-ch-n',rail), sEl=$('#ana-ch-s',rail); if(!n) return;
    n.textContent=e.ok?tx(`finishes in ${e.nCalls} calls`,`做完了，${e.nCalls} 次调用`):tx(`stopped after ${e.nCalls} calls`,`第 ${e.nCalls} 次调用后被拦下`);
    sEl.textContent=fT(e.cost);
    const g1=$('#ana-ch-g1',rail); g1.classList.toggle('met',e.ok); g1.classList.toggle('ana-over',!e.ok);
    $$('.seg-ctl[data-key="chl"]',rail).forEach(g=>A.ui.setPressed(g,String(chO.lim))); $$('.seg-ctl[data-key="chb"]',rail).forEach(g=>A.ui.setPressed(g,String(chO.bud)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#ana-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':'sb');
/* the run the stage shows, and the replay position */
function stageRun(){
  const v=viewNow();
  if(v==='fail') return fxRun(fx);
  if(v==='time') return parRun(par);
  if(v==='ch') return build('chN',{lim:chO.lim,budget:chO.bud});
  if(v==='sb') return curRun();
  return runBase();
}
function stagePos(run){
  const v=viewNow();
  if(v==='start') return 0;
  if(v==='calls') return sendPos(run,callK);
  if(v==='replay'||v==='sb') return Math.min(pos,run.events.length);
  return run.events.length;
}
const hideTotal=()=>mode==='guided'&&(isP('mission')||isP('who'));
const WHO={you:{en:'your code',zh:'你的代码'},model:{en:'the model',zh:'模型'},tool:{en:'a tool',zh:'工具'}};
function blockLabel(b){
  if(b.kind==='sys') return 'system prompt';
  if(b.kind==='tools') return tx(`tool definitions (${b.n})`,`工具说明（${b.n} 个）`);
  if(b.kind==='task') return tx('task','任务');
  if(b.kind==='use') return 'tool_use: '+b.name;
  if(b.kind==='res') return 'tool_result: '+b.name+(b.err?', is_error':'');
  return tx('answer, end_turn','回答，end_turn');
}
function blockTip(b){
  const head=tx(`${blockLabel(b)}, written by ${L(WHO[b.who])}${b.kind==='res'?' (your code wraps the tool’s output and sends it)':(b.kind==='task'?' (the person’s request, sent by your code)':'')}.`,
                `${blockLabel(b)}。${b.kind==='res'?'工具的输出，由你的代码包好发回去':(b.kind==='task'?'用户的请求，由你的代码发出':(b.who==='model'?'模型写的':'你的代码写的'))}。`);
  const size=tx(`${fint(b.chars)} characters, about ${fT(b.tok)} tokens.`,`${fint(b.chars)} 个字符，约 ${fT(b.tok)} 个 token。`);
  return tx(`${head} ${size}`,`${head}${size}`);
}
function drawConv(run,st){
  const host=$('#ana-conv',app), cap=$('#ana-conv-cap',app), ev=st.last;
  const sending=ev&&ev.type==='send';
  cap.textContent=st.call===0?tx('The conversation before call 1 (one row per block)','第 1 次调用之前的对话（一行是一块）'):(sending?tx(`The conversation call ${st.call} sends (one row per block)`,`第 ${st.call} 次调用发出的对话（一行是一块）`):tx(`The conversation after call ${st.call} (one row per block)`,`第 ${st.call} 次调用之后的对话（一行是一块）`));
  const vis=st.vis, newest=ev&&(ev.type==='reply'||ev.type==='tool')?ev:null;
  const rows=vis.map((b,i)=>{const fresh=newest&&b.call===newest.call&&((newest.type==='reply'&&b.who==='model')||(newest.type==='tool'&&b.kind==='res'));
    return `<div class="ana-row w-${b.who}${b.err?' err':''}${fresh?' new':''}${sending?' sent':''}" tabindex="0" data-tipx="${esc(blockTip(b))}"><i class="ana-sw w-${b.who}"></i><span class="ana-rl">${esc(blockLabel(b))}</span><span class="ana-pv">${esc(String(b.prev||'').split('\n')[0])}</span><b class="ana-rt mono">${fT(b.tok)}</b></div>`;}).join('');
  const stopRow=ev&&ev.type==='stop'?`<div class="ana-row ana-stoprow" tabindex="0" data-tipx="${esc(capOf(run,st)[1])}"><span class="ana-rl">${esc(tx('your code stops','你的代码停下'))}</span><span class="ana-pv">${esc(ev.why==='limit'?tx(`it has made ${run.n} calls, the limit`,`已经调用了 ${run.n} 次，到了上限`):(ev.why==='budget'?tx('the token budget is used up','token 预算用完了'):tx('the tool failed; the run ends','工具出错，运行结束')))}</span></div>`:'';
  const h=rows+stopRow;
  if(host._h!==h){host.innerHTML=h; host._h=h; host.scrollTop=host.scrollHeight;}
  const lg=`<span tabindex="0" data-tipx="${esc(tx('Written by your code: the system prompt, the tool definitions, and the task it was given.','你的代码写的：system prompt、工具说明，以及它接到的任务。'))}"><i class="ana-sw w-you"></i>${esc(tx('your code','你的代码'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('Written by the model: tool_use blocks, and the final answer.','模型写的：tool_use 块，以及最后的回答。'))}"><i class="ana-sw w-model"></i>${esc(tx('the model','模型'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('A tool’s output, sent back by your code as a tool_result.','工具的输出，由你的代码作为 tool_result 发回去。'))}"><i class="ana-sw w-tool"></i>${esc(tx('a tool','工具'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('A red outline: a tool_result with is_error set to true.','红框：is_error 为 true 的 tool_result。'))}"><i class="ana-sw ana-sw-err"></i>${esc(tx('error','出错'))}</span>`+
    `<span class="ana-legn" tabindex="0" data-tipx="${esc(TIP.block())}">${esc(tx('row = one block, number = its tokens','一行 = 一块，数字 = 它的 token 数'))}</span>`;
  const leg=$('#ana-conv-leg',app); if(leg._h!==lg){leg.innerHTML=lg; leg._h=lg;}
}
function capOf(run,st){
  const e=st.last;
  if(!e) return [tx('Ready, nothing sent yet','准备好了，还没发出任何东西'),tx('Your code has the system prompt, the tool definitions and the task.','你的代码手里有 system prompt、工具说明和任务。')];
  const c=run.calls[e.call-1];
  if(e.type==='send'){const nb=st.vis.length; return [tx(`Call ${e.call}: your code → the model`,`第 ${e.call} 次调用：你的代码 → 模型`),tx(`It sends the whole conversation: ${nb} blocks, ${fT(c.inTok)} tokens.`,`发出整段对话：${nb} 块，共 ${fT(c.inTok)} 个 token。`)];}
  if(e.type==='reply'){if(c.end) return [tx(`Call ${e.call}: the model → your code`,`第 ${e.call} 次调用：模型 → 你的代码`),tx('The model answers in text; stop_reason is end_turn, so the loop ends.','模型用文字作答，stop_reason 是 end_turn，循环结束。')];
    const ns=c.reply.map(b=>b.name);
    return [tx(`Call ${e.call}: the model → your code`,`第 ${e.call} 次调用：模型 → 你的代码`),ns.length>1?tx(`It asks for ${ns.length} tools at once: ${ns.join(', ')}.`,`它一次要 ${ns.length} 个工具：${ns.join('、')}。`):tx(`It asks for one tool: ${ns[0]}, with its arguments.`,`它要调用一个工具：${ns[0]}，并给出参数。`)];}
  if(e.type==='tool'){const err=c.res.some(r=>r.err), tk=c.res.reduce((a,r)=>a+r.tok,0), nm=c.res.map(r=>r.name);
    return [tx('Your code → the tools → your code','你的代码 → 工具 → 你的代码'),err?tx(`${nm[0]} failed: the output goes back marked is_error.`,`${nm[0]} 没通过：输出标上 is_error，照样接在后面。`):tx(`It runs ${nm.join(', ')}; the result (${fT(tk)} tokens) is appended.`,`执行 ${nm.join('、')}，结果（${fT(tk)} 个 token）接在对话后面。`)];}
  return [tx('Your code stops','你的代码停下'),e.why==='limit'?tx(`It has made ${run.n} calls, the limit.`,`已经调用了 ${run.n} 次，到了上限。`):(e.why==='budget'?tx('One more call would go over the token budget.','再调用一次，读进去的 token 就会超过预算。'):tx('A tool failed, so the whole task ends here.','工具出错了，整个任务到此结束。'))];
}
function drawLoop(run,st){
  const e=st.last, on=k=>{if(!e) return false; if(e.type==='send') return k==='hm'; if(e.type==='reply') return k==='mh'; if(e.type==='tool') return k==='ht'; return false;};
  const act=w=>{if(!e) return w==='you'; if(e.type==='send'||e.type==='stop') return w==='you'; if(e.type==='reply') return w==='model'; return w==='tool';};
  const node=(w,t,sub)=>`<div class="ana-node w-${w}${act(w)?' on':''}"><b>${esc(t)}</b><span>${esc(sub)}</span></div>`;
  const h=node('model',tx('Model','模型'),tx('decides the next step','决定下一步'))+
    `<div class="ana-arr"><span class="${on('hm')?'on':''}">← <em>${esc(tx('conversation','整段对话'))}</em></span><span class="${on('mh')?'on':''}"><em>${esc(tx('reply','回复'))}</em> →</span></div>`+
    node('you',tx('Your code','你的代码'),tx('the harness: sends, runs tools','即 harness：发送对话、执行工具'))+
    `<div class="ana-arr"><span class="${on('ht')?'on':''}"><em>${esc(tx('run','执行'))}</em> →</span><span class="${on('ht')?'on':''}">← <em>${esc(tx('output','输出'))}</em></span></div>`+
    node('tool',tx('Tools','工具'),tx('files, search, tests','文件、搜索、测试'));
  const host=$('#ana-loop',app); if(host._h!==h){host.innerHTML=h; host._h=h;}
  const [l1,l2]=capOf(run,st), ch=`<b>${esc(l1)}</b><span>${esc(l2)}</span>`; const c2=$('#ana-cap2',app); if(c2._h!==ch){c2.innerHTML=ch; c2._h=ch;}
}
function drawBars(run,st){
  const host=$('#ana-bars',app), cap=$('#ana-bars-cap',app), hide=hideTotal();
  const n=run.calls.length, cur=st.call, mx=Math.max(12000,...run.calls.map(c=>c.inTok));
  cap.textContent=hide||cur<n||viewNow()==='replay'&&!replayDone?tx('Tokens each call sends','每次调用发出的 token'):tx(`Tokens each call sends; all ${n} bars together: ${fK(run.totalIn)}`,`每次调用发出的 token；${n} 根柱子加起来：${fK(run.totalIn)} 个`);
  const many=n>12;
  const cols=run.calls.map((c,k)=>{
    const done=c.i<=cur, you=run.blocks.filter(b=>b.who==='you').reduce((a,b)=>a+b.tok,0);
    const mdl=run.blocks.filter(b=>b.who==='model'&&b.call<c.i).reduce((a,b)=>a+b.tok,0), tl=c.inTok-you-mdl;
    const H=v=>(v/mx*100).toFixed(2);
    const prev=k?run.calls[k-1].inTok:0;
    const tip=tx(`Call ${c.i} sends ${fT(c.inTok)} tokens: ${fT(you)} from your code, ${fT(tl)} of tool output, ${fT(mdl)} written by the model.${k?` ${fT(prev)} of them were already sent by call ${c.i-1}.`:''}`,
                 `第 ${c.i} 次调用发出 ${fT(c.inTok)} 个 token：你的代码写的 ${fT(you)} 个，工具的输出 ${fT(tl)} 个，模型写的 ${fT(mdl)} 个。${k?`其中 ${fT(prev)} 个第 ${c.i-1} 次已经发过。`:''}`);
    return `<div class="ana-col${done?'':' todo'}${c.i===cur?' cur':''}" tabindex="0" data-tipx="${esc(done?tip:tx(`Call ${c.i} has not been sent yet.`,`第 ${c.i} 次调用还没有发出。`))}">`+
      (done?`<i class="s-you" style="height:${H(you)}%"></i><i class="s-tool" style="height:${H(tl)}%;bottom:${H(you)}%"></i><i class="s-model" style="height:${H(mdl)}%;bottom:${H(you+tl)}%"></i>`+(k&&!many?`<em class="ana-prevmk" style="bottom:${H(prev)}%"></em>`:'')+(!many?`<span class="ana-cv" style="bottom:${H(c.inTok)}%">${fK(c.inTok)}</span>`:''):'<i class="s-todo" style="height:100%"></i>')+`</div>`;}).join('');
  const labs=run.calls.map(c=>`<span${c.i===cur?' class="on"':''}>${many&&c.i%5&&c.i!==1?'':c.i}</span>`).join('');
  const h=`<div class="ana-cols" style="--n:${n}">${cols}</div><div class="ana-clabs" style="--n:${n}">${labs}</div><div class="ana-cx">${esc(tx('call','调用'))}</div>`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
  const lg=`<span><i class="ana-sw w-you"></i>${esc(tx('your code','你的代码'))}</span><span><i class="ana-sw w-tool"></i>${esc(tx('tool output','工具的输出'))}</span><span><i class="ana-sw w-model"></i>${esc(tx('the model','模型'))}</span>`+
    (!many?`<span tabindex="0" data-tipx="${esc(tx('The dashed mark on a bar is the height of the bar before it: everything below it was already sent by the previous call.','柱子上的虚线是前一根柱子的高度：虚线以下的部分，上一次调用已经发过了。'))}"><i class="ana-sw ana-sw-prev"></i>${esc(tx('already sent by the call before','上一次已经发过'))}</span>`:'');
  const leg=$('#ana-bars-leg',app); if(leg._h!==lg){leg.innerHTML=lg; leg._h=lg;}
}
function drawTime(run){
  const host=$('#ana-time',app), cap=$('#ana-time-cap',app), T=run.time, X=v=>(v/T*100).toFixed(3);
  cap.textContent=tx(`The whole task takes ${fS(T)} s: model calls ${fS(run.tModel)} s (${fPc(run.tModel/T)}), tools ${fS(run.tTool)} s`,`整个任务 ${fS(T)} 秒：模型调用 ${fS(run.tModel)} 秒（${fPc(run.tModel/T)}），工具 ${fS(run.tTool)} 秒`);
  const mseg=run.calls.map(c=>`<i class="t-model" style="left:${X(c.start)}%;width:${X(c.tM)}%" tabindex="0" data-tipx="${esc(tx(`Call ${c.i}: ${fS(c.tM)} s. Reading ${fT(c.inTok)} tokens and writing ${fT(c.out)}, plus the fixed cost.`,`第 ${c.i} 次调用：${fS(c.tM)} 秒。读 ${fT(c.inTok)} 个 token、写 ${fT(c.out)} 个，加上固定开销。`))}"></i>`).join('');
  const tseg=run.calls.filter(c=>c.tools.length).map(c=>c.tools.map((t,k)=>`<i class="t-tool${c.tools.length>1?' par':''}" style="left:${X(c.start+c.tM)}%;width:${X(t.t)}%;--k:${k};--m:${c.tools.length}" tabindex="0" data-tipx="${esc(tx(`${t.name}: ${fS(t.t)} s${c.tools.length>1?`, at the same time as the other ${c.tools.length-1}`:''}.`,`${t.name}：${fS(t.t)} 秒${c.tools.length>1?`，和另外 ${c.tools.length-1} 个同时跑`:''}。`))}"></i>`).join('')).join('');
  let ticks=''; const stepS=T>40?10:5; for(let s=0;s<=T+1e-9;s+=stepS) ticks+=`<span style="left:${X(s)}%">${s}</span>`;
  /* a label under the first call whose tools run at the same time; the row is always there, so the box keeps its height (R7.9) */
  const pc=run.calls.find(c=>c.tools.length>1), at0=pc?(pc.start+pc.tM)/T:0;
  const pnote=pc?`<span class="ana-tnote${at0>0.6?' end':''}" style="left:${X(pc.start+pc.tM)}%">${esc(tx(`↑ ${pc.tools.length} tools at once, ${fS(pc.tT)} s`,`↑ ${pc.tools.length} 个工具同时跑，用时 ${fS(pc.tT)} 秒`))}</span>`:'';
  const h=`<div class="ana-lane"><span class="ana-ll">${esc(tx('model','模型'))}</span><div class="ana-trk">${mseg}</div></div><div class="ana-lane ana-tl"><span class="ana-ll">${esc(tx('tools','工具'))}</span><div class="ana-trk">${tseg}</div></div><div class="ana-lane ana-tnl"><span class="ana-ll"></span><div class="ana-tnt">${pnote}</div></div><div class="ana-tscale"><span class="ana-ll">${esc(tx('seconds','秒'))}</span><div class="ana-ts">${ticks}</div></div>`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
  const lg=`<span><i class="ana-sw w-model"></i>${esc(tx('a model call','一次模型调用'))}</span><span><i class="ana-sw w-tool"></i>${esc(tx('a tool run (thin rows: at the same time)','一次工具执行（细条：同时跑）'))}</span>`;
  const leg=$('#ana-time-leg',app); if(leg._h!==lg){leg.innerHTML=lg; leg._h=lg;}
}
function drawChallenge(){
  const nr=build('chN',{}), sr=build('chS',{}), e=chEval(chO.lim,chO.bud), host=$('#ana-chart',app);
  const cum=r=>{let a=0; return [0].concat(r.calls.map(c=>(a+=c.inTok)));};
  const cn=cum(nr), cs=cum(sr), X=Math.max(chO.lim,nr.n)+3, Y=Math.max(chO.bud,nr.totalIn)*1.12;
  /* drawn at the box's own width (one SVG unit = one pixel), so the labels keep their 11 px on a phone (R7.4) */
  const W=Math.max(280,Math.min(900,Math.round(host.clientWidth||560))), H=W<480?250:236, l=62, r=12, t=34, b=44, px=i=>l+(W-l-r)*i/X, py=v=>t+(H-t-b)*(1-v/Y);
  const line=(arr,upto)=>arr.slice(0,upto+1).map((v,i)=>`${i?'L':'M'}${px(i).toFixed(1)},${py(Math.min(v,Y*1.5)).toFixed(1)}`).join('');
  const sStop=e.sCalls, nStop=e.nCalls;
  const sAll=cs.slice(0,Math.min(cs.length,X+1));
  let ticksX=''; const sx=X>30?10:5; for(let i=0;i<=X;i+=sx) ticksX+=`<text x="${px(i).toFixed(1)}" y="${H-26}" text-anchor="middle">${i}</text>`;
  let ticksY=''; const sy=Y>200000?100000:50000; for(let v=0;v<=Y;v+=sy) ticksY+=`<text x="${l-6}" y="${(py(v)+4).toFixed(1)}" text-anchor="end">${fT(v)}</text><line x1="${l}" x2="${W-r}" y1="${py(v).toFixed(1)}" y2="${py(v).toFixed(1)}" class="g"/>`;
  const axes=`<text x="${((l+W-r)/2).toFixed(1)}" y="${H-6}" text-anchor="middle" class="ax">${esc(tx('model call','第几次调用'))}</text><text x="${l}" y="13" class="ax">${esc(tx('tokens read so far','累计读入的 token'))}</text>`;
  const svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx('Tokens read so far against model calls, for the normal and the stuck run, with the call limit and the budget','两种运行累计读入的 token 随调用次数的变化，以及调用次数上限和预算'))}"><defs><clipPath id="ana-clip"><rect x="${l}" y="${t-6}" width="${W-l-r}" height="${H-t-b+6}"/></clipPath></defs>${ticksY}${ticksX}${axes}`+
    `<rect x="${l}" y="${py(chO.bud).toFixed(1)}" width="${(px(chO.lim)-l).toFixed(1)}" height="${(py(0)-py(chO.bud)).toFixed(1)}" class="box"/>`+
    `<line x1="${px(chO.lim).toFixed(1)}" x2="${px(chO.lim).toFixed(1)}" y1="${t}" y2="${py(0)}" class="lim"/><line x1="${l}" x2="${W-r}" y1="${py(chO.bud).toFixed(1)}" y2="${py(chO.bud).toFixed(1)}" class="lim"/>`+
    `<g clip-path="url(#ana-clip)"><path d="${line(sAll,sAll.length-1)}" class="s ghost"/><path d="${line(cs,sStop)}" class="s"/>`+
    `<path d="${line(cn,cn.length-1)}" class="n ghost"/><path d="${line(cn,nStop)}" class="n"/></g>`+
    `<circle cx="${px(sStop).toFixed(1)}" cy="${py(Math.min(cs[sStop],Y)).toFixed(1)}" r="4" class="sd"/><circle cx="${px(nStop).toFixed(1)}" cy="${py(cn[nStop]).toFixed(1)}" r="4" class="${e.ok?'nd ok':'nd'}"/>`+
    `<text x="${(px(chO.lim)-4).toFixed(1)}" y="${t-6}" text-anchor="end" class="lab">${esc(tx(`limit ${chO.lim}`,`上限 ${chO.lim} 次`))}</text><text x="${l+4}" y="${(py(chO.bud)-4).toFixed(1)}" class="lab">${esc(tx(`budget ${fT(chO.bud)}`,`预算 ${fT(chO.bud)}`))}</text></svg>`;
  $('#ana-ch-cap',app).textContent=tx('Tokens read so far: a run is stopped where it leaves the box','累计读入的 token：运行走出方框的地方，你的代码就让它停下');
  if(host._h!==svg){host.innerHTML=svg; host._h=svg;}
  host.dataset.tipx=tx(`The normal run ${e.ok?`finishes inside the box after ${e.nCalls} calls, ${fT(e.nRead)} tokens`:`leaves the box and is stopped after ${e.nCalls} calls`}. The stuck run is stopped after ${e.sCalls} calls, having read ${fT(e.cost)} tokens.`,
                      `正常的运行${e.ok?`在方框里做完：${e.nCalls} 次调用，${fT(e.nRead)} 个 token`:`走出了方框，第 ${e.nCalls} 次调用后被拦下`}。卡住的运行在第 ${e.sCalls} 次调用后停下，已经读了 ${fT(e.cost)} 个 token。`);
  host.tabIndex=0;
  const lg=`<span><i class="ana-sw ana-sw-n"></i>${esc(tx('normal run','正常的运行'))}</span><span><i class="ana-sw ana-sw-s"></i>${esc(tx('stuck run','卡住的运行'))}</span><span><i class="ana-sw ana-sw-g"></i>${esc(tx('faint: what it would read without limits','淡线：不设上限会读到哪里'))}</span><span><i class="ana-sw ana-sw-box"></i>${esc(tx('dashed box: your two limits','虚线框：你设的两个上限'))}</span>`;
  const leg=$('#ana-ch-leg',app); if(leg._h!==lg){leg.innerHTML=lg; leg._h=lg;}
}
function drawPlayer(run){
  const pl=$('#ana-player',app); if(!pl) return; const E=run.events.length, p=Math.min(pos,E);
  const sig=[playing,spd,p,E,A.lang()].join('|'); if(pl._sig===sig) return; pl._sig=sig;
  const label=playing?tx('❚❚ Pause','❚❚ 暂停'):(p>=E?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  const st=at(run,p), cnt=tx(`call ${st.call} of ${run.calls.length}`,`第 ${st.call} / ${run.calls.length} 次调用`);
  const scrub=pl.querySelector('input[data-key="scrub"]');
  if(scrub&&pl._built===A.lang()+E){scrub.value=p; A.ui.setRange(scrub); pl.querySelector('[data-act="play"]').textContent=label; $$('.seg-ctl[data-key="spd"]',pl).forEach(g=>A.ui.setPressed(g,spd)); pl.querySelector('.ana-count').textContent=cnt; return;}
  pl._built=A.lang()+E;
  pl.innerHTML=`<button type="button" class="btn ghost sm ana-play" data-act="play" data-tipx="${esc(tx('Play the run at the chosen speed; press again to pause. Once the run has finished, the button plays it again from the first event.','按选定的速度播放这次运行；再按一次暂停。放完后，这个按钮从第一个事件重新播放。'))}">${label}</button>`+
    `<button type="button" class="btn ghost sm ana-stepb" data-act="back1" aria-label="${esc(tx('One event back','后退一步'))}" data-tipx="${esc(tx('One event back.','后退一步。'))}">◂</button><button type="button" class="btn ghost sm ana-stepb" data-act="fwd1" aria-label="${esc(tx('One event forward','前进一步'))}" data-tipx="${esc(tx('One event forward.','前进一步。'))}">▸</button>`+
    A.ui.seg({id:'ana-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd})+
    `<div class="ana-scrub"><input type="range" min="0" max="${E}" step="1" value="${p}" data-key="scrub" aria-label="${esc(tx('Position in the replay','回放的位置'))}"><span class="ana-count mono">${cnt}</span></div>`;
  A.ui.initRanges(pl);
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(chO.lim,chO.bud); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Normal run finishes','正常运行能做完'):tx('Stops too early','拦得太早'),short:e.ok?tx('Finishes','能做完'):tx('Too early','拦得太早'),
    tip:tx(`Green when the normal run finishes within both limits. Now the stuck run reads ${fT(e.cost)} tokens before it stops.`,`正常的运行在两个上限之内做完时，徽章就是绿色。现在卡住的运行停下前读了 ${fT(e.cost)} 个 token。`)};}
  const run=stageRun(), st=at(run,stagePos(run));
  if(mode==='guided'&&isP('mission')) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The replay starts in part 2; the total tokens read are hidden until part 3.','回放从第 2 部分开始；读进去的 token 总数到第 3 部分才揭晓。')};
  const e=st.last;
  if(e&&e.type==='stop') return {cls:'bad',txt:tx('Stopped, not done','停下了，没做完'),short:tx('Stopped','停下了'),tip:L({en:'The run stopped before the model said it was done.',zh:'模型还没说做完，运行就停了。'})};
  if(st.p>=run.events.length&&run.done) return {cls:'neutral',txt:tx('Done: end_turn','做完了：end_turn'),short:tx('Done','做完了'),tip:tx('The last reply has stop_reason end_turn: the model says it is done.','最后一次回复的 stop_reason 是 end_turn：模型说做完了。')};
  return {cls:'neutral',txt:st.call?tx(`Call ${st.call} of ${run.calls.length}`,`第 ${st.call} / ${run.calls.length} 次调用`):tx('Not started','还没开始'),short:st.call?tx(`Call ${st.call}`,`第 ${st.call} 次`):tx('Not started','还没开始'),tip:tx('Which model call the replay is at.','回放进行到第几次模型调用。')};
}
function drawHead(run){
  const v=viewNow(), t=$('#ana-st-title',app), sub=$('#ana-st-sub',app), sEl=$('#ana-status',app);
  const titles={ch:tx('A new task: two runs and your two limits','新任务：两种运行，和你设的两个上限'),sb:{base:tx('Fixing a failing test','修一个没通过的测试'),par:tx('Fixing a failing test, three tools at once','修一个没通过的测试，一次要 3 个工具'),long:tx('Renaming a function across 13 files','在 13 个文件里给一个函数改名'),huge:tx('Fixing a failing test, with full test logs','修一个没通过的测试，测试打开了详细日志')}[S.run]};
  t.textContent=v==='ch'?titles.ch:(mode==='sandbox'?titles.sb:tx('An agent fixes a failing test','一个 agent 修好一个没通过的测试'));
  sub.textContent=v==='ch'?planName(chO.lim,chO.bud):(run.done?tx(`${callsW(run.n)}, ${toolsW(run.calls.reduce((a,c)=>a+c.res.length,0))} run by your code`,`${run.n} 次模型调用，你的代码执行了 ${run.calls.reduce((a,c)=>a+c.res.length,0)} 次工具`):tx(`stopped after ${callsW(run.n)}`,`${run.n} 次调用后停下`));
  const s=statusOf(); sEl.className='status '+s.cls; sEl.textContent=s.txt; sEl.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(run,st){
  if(viewNow()==='ch'){const e=chEval(chO.lim,chO.bud);
    return [{k:'nr',label:tx('Normal run','正常的运行'),v:e.ok?tx('done','做完了'):tx('stopped','被拦下'),sub:callsW(e.nCalls),tip:tx(`The normal run ${e.ok?'finishes':'is stopped'} after ${e.nCalls} calls, having read ${fT(e.nRead)} tokens.`,`正常的运行${e.ok?'做完了':'被拦下了'}：${e.nCalls} 次调用，读了 ${fT(e.nRead)} 个 token。`)},
      {k:'sr',label:tx('Stuck run reads','卡住的运行读了'),v:fT(e.cost),sub:tx(`tokens, ${callsW(e.sCalls)}`,`个 token，${e.sCalls} 次调用`),tip:tx(`The stuck run is stopped by ${stopWhy(e.sStop)} after ${e.sCalls} calls.`,`卡住的运行在第 ${e.sCalls} 次调用后被${stopWhy(e.sStop)}拦下。`)},
      {k:'nt',label:tx('Normal run reads','正常的运行读了'),v:fT(e.nRead),sub:tx('tokens in all','个 token，所有调用加起来'),tip:tx('Every call of the normal run, added up.','正常运行的每次调用加起来。')}];}
  const c=st.call?run.calls[st.call-1]:null, hide=hideTotal();
  return [{k:'calls',label:tx('Model calls','模型调用'),v:String(st.call),sub:tx(`of ${run.calls.length} in this run`,`次，这次运行共 ${run.calls.length} 次`),tip:tx(`${st.call} of the run’s ${run.calls.length} model calls have been sent.`,`这次运行的 ${run.calls.length} 次模型调用里，已经发出了 ${st.call} 次。`)},
    {k:'in',label:tx('This call sends','这次调用发出'),v:c?fT(c.inTok):'0',sub:tx('tokens','个 token'),tip:c?tx(`Call ${c.i} sends ${fT(c.inTok)} tokens: every block in the conversation so far.`,`第 ${c.i} 次调用发出 ${fT(c.inTok)} 个 token：对话里到目前为止的每一块。`):tx('No call has been sent yet.','还没有发出任何调用。')},
    {k:'sum',label:tx('Read so far','累计读入'),v:hide?'—':fT(st.readSoFar),sub:hide?tx('shown in part 3','第 3 部分揭晓'):tx('tokens, all calls so far','个 token（到目前为止所有调用加起来）'),tip:hide?tx('Hidden until part 3.','到第 3 部分才揭晓。'):tx(`The tokens of every call so far, added up: ${fT(st.readSoFar)}.`,`到目前为止每次调用的 token 加起来：${fT(st.readSoFar)} 个。`)},
    {k:'t',label:tx('Time so far','到目前用时'),v:fS(st.t),sub:tx(`s, ${fS(st.tm)} s of it in model calls`,`秒，其中模型 ${fS(st.tm)} 秒`),tip:tx(`${fS(st.tm)} s in model calls and ${fS(st.tt)} s running tools, with the assumed speeds.`,`按假设的速度：模型调用 ${fS(st.tm)} 秒，执行工具 ${fS(st.tt)} 秒。`)}];
}
function drawMetrics(run,st){
  const host=$('#ana-metrics',app), defs=metricsDef(run,st), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!playing&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: lines redo from the numbers they show; every term has a tooltip (R1.4) */
function mathHTML(run,st){
  if(viewNow()==='ch'){const nr=build('chN',{}), e=chEval(chO.lim,chO.bud);
    return MB(tx('<b>When your code stops the loop.</b> Before call i, it stops if either of these holds:','<b>你的代码什么时候停下循环。</b>第 i 次调用之前，满足任一条件就停下：'),
        M(tx('i > \\text{limit} \\quad \\text{or} \\quad \\sum_{j<i} \\text{input}_j + \\text{input}_i > \\text{budget}','i > \\text{上限} \\quad \\text{或} \\quad \\sum_{j<i} \\text{输入}_j + \\text{输入}_i > \\text{预算}')))+
      MB(tx(`<b>The normal run</b> sends these tokens over ${nr.n} calls:`,`<b>正常的运行</b>一共 ${nr.n} 次调用，发出的 token：`),
        EQ(SUMROWS(tx('\\text{tokens read}','\\text{读进去的 token}'),nr.calls.map(c=>fT(c.inTok))).concat([['',fT(nr.totalIn)]])),
        tx(`With this setting, the stuck run makes ${e.sCalls} calls and reads ${fT(e.cost)} tokens.`,`这个设置下，卡住的运行调用 ${e.sCalls} 次，读进去 ${fT(e.cost)} 个 token。`));}
  if(mode==='guided'&&isP('mission')) return `<p>${tx('The formulas appear once the replay starts in part 2.','第 2 部分开始回放后，这里给出公式。')}</p>`;
  const hide=hideTotal(), c=st.call?run.calls[st.call-1]:run.calls[0], blocks=[];
  const seen=run.blocks.filter(b=>b.at<sendPos(run,c.i)-1||b.at<0);
  blocks.push(MB(tx(`<b>Tokens of call ${c.i}.</b> A block of text is about one token per 4 characters. Call ${c.i} sends ${seen.length} blocks.`,`<b>第 ${c.i} 次调用的 token。</b>一块文字大约每 4 个字符算 1 个 token。第 ${c.i} 次调用发出 ${seen.length} 块。`),
    EQ([[tx('\\text{tokens of a block}','\\text{一块的 token}'),tx('\\frac{\\text{characters}}{4}','\\frac{\\text{字符数}}{4}'),'\\approx'],].concat(SUMROWS(tx(`\\text{input}_{${c.i}}`,`\\text{输入}_{${c.i}}`),seen.slice(0,6).map(b=>fT(b.tok)).concat(seen.length>6?['\\ldots']:[])),[['',fT(c.inTok)]]))));
  if(!hide&&(st.call>=run.calls.length||viewNow()!=='calls')){const n=run.calls.length;
    blocks.push(MB(tx(`<b>Tokens read in all.</b> Every call resends what came before, so the model reads the inputs of all ${n} calls added up.`,`<b>一共读进去的 token。</b>每次调用都把前面的内容再发一遍，所以模型读进去的，是 ${n} 次调用的输入加起来。`),
      EQ([[tx('\\text{tokens read}','\\text{读进去的 token}'),tx(`\\sum_{i=1}^{${n}} \\text{input}_i`,`\\sum_{i=1}^{${n}} \\text{输入}_i`)],].concat(SUMROWS('',n<=10?run.calls.map(x=>fT(x.inTok)):run.calls.slice(0,4).map(x=>fT(x.inTok)).concat(['\\ldots',fT(run.finalIn)])),[['',fint(run.totalIn)]])),
      tx(`That is <b>${fK(run.totalIn)}</b> tokens.`,`也就是 <b>${fK(run.totalIn)}</b> 个 token。`)));
    /* part 3 (the owner, 2026-10-14): what the bill counts, each call's input plus its output */
    if(isP('memory')){const k=Math.round(run.totalIn/run.totalOut/10)*10;
      blocks.push(MB(tx('<b>Tokens written, and the bill.</b> The reply of each call is its output. The provider bills every call’s input tokens and output tokens, each at its own price.','<b>写出的 token 和账单。</b>每次调用写出的回复，就是这次的输出。服务商对每次调用的输入 token 和输出 token，各按各的单价收费。'),
        EQ([[tx('\\text{tokens written}','\\text{写出的 token}'),tx(`\\sum_{i=1}^{${n}} \\text{output}_i`,`\\sum_{i=1}^{${n}} \\text{输出}_i`)],].concat(SUMROWS('',run.calls.map(x=>fT(x.out))),[['',fint(run.totalOut)]]))+
        EQ([[tx('\\text{cost}','\\text{费用}'),tx(`\\text{input price} \\times ${fT(run.totalIn)}`,`\\text{输入单价} \\times ${fT(run.totalIn)}`)],['',tx(`+\\ \\text{output price} \\times ${fT(run.totalOut)}`,`+\\ \\text{输出单价} \\times ${fT(run.totalOut)}`),' ']]),
        tx(`The input tokens are about ${k} times the output: the model writes a reply of only a few dozen tokens each time, while each input holds every step before it and grows with every call.`,`输入 token 约是输出的 ${k} 倍：模型每次只写几十个 token 的回复，输入却装着之前的每一步，一次比一次长。`)));}}
  blocks.push(MB(tx(`<b>Time of call ${c.i}.</b> The model first reads the input, then writes the output one token at a time. ${m('t_0')} is the fixed cost of a call (${T_('lat',fS(run.P.lat)+' s')}); reading runs at ${T_('pre',fint(run.P.pre))} tokens/s and writing at ${T_('dec',String(run.P.dec))} tokens/s.`,
                    `<b>第 ${c.i} 次调用的时间。</b>模型先读完输入，再一个一个 token 地写出输出。${m('t_0')} 是每次调用固定的开销（${T_('lat',fS(run.P.lat)+' 秒')}）。读的速度是每秒 ${T_('pre',fint(run.P.pre))} 个 token，写的速度是每秒 ${T_('dec',String(run.P.dec))} 个 token。`),
    EQ([[tx('t_\\text{call}','t_\\text{调用}'),tx('t_0 + \\frac{\\text{input}}{v_\\text{prefill}} + \\frac{\\text{output}}{v_\\text{decode}}','t_0 + \\frac{\\text{输入}}{v_\\text{prefill}} + \\frac{\\text{输出}}{v_\\text{decode}}')],['',`${fS(run.P.lat)} + \\frac{${fT(c.inTok)}}{${fint(run.P.pre)}} + \\frac{${fT(c.out)}}{${run.P.dec}}`],['',fixd(c.tM,2)]]),
    tx(`Call ${c.i} takes <b>${fixd(c.tM,2)} s</b>.`,`第 ${c.i} 次调用要 <b>${fixd(c.tM,2)} 秒</b>。`)));
  return blocks.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const v=viewNow(), run=stageRun(), p=stagePos(run), st=at(run,p);
  drawHead(run);
  const showLoop=v!=='ch', showConv=v!=='ch'&&v!=='time', showBars=v!=='ch', showTime=v==='time'||(mode==='sandbox'), showPlayer=v==='replay'||v==='sb';
  $('#ana-loopbox',app).hidden=!showLoop; $('#ana-convbox',app).hidden=!showConv; $('#ana-barsbox',app).hidden=!showBars; $('#ana-timebox',app).hidden=!showTime; $('#ana-chbox',app).hidden=v!=='ch'; $('#ana-player',app).hidden=!showPlayer;
  if(showLoop) drawLoop(run,st);
  if(showPlayer) drawPlayer(run);
  if(showConv) drawConv(run,st);
  if(showBars) drawBars(run,st);
  if(showTime) drawTime(run);
  if(v==='ch') drawChallenge();
  $('#ana-graph',app).dataset.v=v;
  drawMetrics(run,st);
  const mb=$('#ana-mathbody',app), mh=mathHTML(run,st); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
/* during playback only the stage moves; the rail is refreshed at the end */
function renderFrame(){
  if(!app) return; const run=stageRun(), st=at(run,stagePos(run));
  drawHead(run); drawLoop(run,st); drawPlayer(run); drawConv(run,st); drawBars(run,st); drawMetrics(run,st); A.dock.refresh();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the call limit and the token budget, in the task card. The task and the speeds are fixed.','挑战中只能在任务卡里改调用次数上限和 token 预算，任务和速度都固定。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='time') return tx('In this part you can change the assumed speeds below; the task is fixed: the bug fix.','这一部分可以改下面假设的速度；任务固定为修 bug 的那个。');
  return tx('Fixed in this part: the bug fix, with the assumed speeds. You can change the settings in Scenarios.','这一部分固定：修 bug 的任务，速度用假设值。在应用场景里可以改这些设置。');
}
function syncControls(){
  ['run','lim','pre','dec','lat','tt'].forEach(k=>$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,String(S[k]))));
  const co=tx(`call ${callK}`,`第 ${callK} 次`);
  $$('input[data-key="callk"]',app).forEach(i=>{if(+i.value!==callK) i.value=callK; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=co; i.setAttribute('aria-valuetext',co);});
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#ana-controls',app).classList.toggle('locked',ch);
  hide('#ana-w-run',g||ch); hide('#ana-w-lim',g||ch); hide('#ana-w-reset',g||ch); hide('#ana-w-time',ch||(g&&!isP('time')));
  const ln=$('#ana-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||playing) return; const run=stageRun(), st=at(run,stagePos(run)), [a,b]=capOf(run,st); A.announce(`${a}. ${b}`);},400);}

/* =====================================================================
   replay
   ===================================================================== */
function tick(t){
  raf=0; if(!app||!playing) return;
  const run=stageRun(), E=run.events.length, dt=lastT?Math.min(250,t-lastT):16; lastT=t;
  acc+=dt/1000;
  if(acc>=SPD[spd]){acc=0; pos=Math.min(E,pos+1); renderFrame();}
  if(pos>=E){playing=false; watched.add(runSig(run)); if(mode==='guided'&&isP('who')){replayDone=true; wantObs=true;} render(); if(wantObs){revealSoon(); wantObs=false;} return;}
  raf=requestAnimationFrame(tick);
}
function play(){const run=stageRun(); if(pos>=run.events.length) pos=0; playing=true; lastT=0; acc=SPD[spd]*0.5; if(A.reduceMotion()){finishReplay(); return;} if(!raf) raf=requestAnimationFrame(tick); drawPlayer(run);}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; if(app) drawPlayer(stageRun());}
function finishReplay(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; const run=stageRun(); pos=run.events.length; watched.add(runSig(run)); if(mode==='guided'&&isP('who')) replayDone=true; wantObs=true; render(); revealSoon(); wantObs=false;}
function stepBy(d){pause(); const run=stageRun(); pos=Math.max(0,Math.min(run.events.length,pos+d)); if(pos>=run.events.length) watched.add(runSig(run)); if(pos>=run.events.length&&mode==='guided'&&isP('who')&&!replayDone){finishReplay(); return;} render(); A.tip.quiet();}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){Object.assign(S,p); if(!RUNOPT.some(r=>r[0]===S.run)) S.run='base'; pos=Math.min(pos,curRun().events.length); render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setFx(v){fx=v; fxTried.add(v); render();}
function setPar(v){par=v; parTried.add(v); render();}
function setCall(k){callK=Math.max(1,Math.min(8,k)); callMax=Math.max(callMax,callK); render();}
function setRem(k){if(!REMS.includes(k)) return; remView=k; pause();
  if(k==='who') pos=runBase().events.length; if(k==='memory'){callK=8; callMax=8;} if(k==='fail'){fx='stop'; fxTried.add('stop');} if(k==='time'){par='par'; parTried.add('par');}
  renderRail(); if(innerWidth<=760) A.ui.reveal($('#ana-stage',app));}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='memory') return callCtl('ana-ph-call',true);
  if(id==='fail') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the three ways in the card.','向下滚到卡片里的三种做法。'))}">${tx('Pick a way below ↓','在下面选做法 ↓')}</button>`;
  if(id==='time') return `<button type="button" class="btn ghost sm" data-act="to-par" data-tipx="${esc(tx('Scrolls down to the two options in the card.','向下滚到卡片里的两个选项。'))}">${tx('Pick below ↓','在下面选 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#ana-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; pause(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; S={...BASE,pre:S.pre,dec:S.dec,lat:S.lat,tt:S.tt};
  if(s.id!=='time') Object.assign(S,TIME0);
  const st=$('#ana-stage',app); if(st) st.scrollTop=0;
  if(s.id==='who'){if(!replayDone&&watched.has(runSig(runBase()))) replayDone=true;   // R3.20: the same run already watched to the end (in Scenarios) is shown finished and passes the gate
    if(!replayDone) pos=0; else pos=runBase().events.length;}
  if(s.id==='remember'){remView='who'; pos=runBase().events.length;}
  renderRail();
  if(s.id==='who'&&!replayDone) play();
}
function revealStep(){A.ui.reveal($('#ana-rail .step.current',app)||$('#ana-rail',app));}
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
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#ana-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ana-rail .step.current',app)||$('#ana-rail .railcard',app)||$('#ana-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#ana-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ana-insight',app)); else revealBox($('#ana-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; pause();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chO={...CH.start};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided'){S={...BASE}; pos=curRun().events.length;}}
  A.shell.setMode(app,m,DESCS());
  const st=$('#ana-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const RUNK=RUNOPT.map(r=>r[0]);
const tokOf=s=>[RUNK.indexOf(s.run),SEGS.lim.findIndex(x=>x[0]===s.lim),SEGS.pre.findIndex(x=>x[0]===s.pre),SEGS.dec.findIndex(x=>x[0]===s.dec),SEGS.lat.findIndex(x=>x[0]===s.lat),SEGS.tt.findIndex(x=>x[0]===s.tt)].join('_');
const token=()=>tokOf(S);
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==6||p.some(x=>!Number.isInteger(x)||x<0)) return null;
  const [r,l,pr,dc,lt,t]=p; if(r>=RUNK.length||l>=3||pr>=3||dc>=3||lt>=3||t>=3) return null;
  return {run:RUNK[r],lim:SEGS.lim[l][0],pre:SEGS.pre[pr][0],dec:SEGS.dec[dc][0],lat:SEGS.lat[lt][0],tt:SEGS.tt[t][0]};
}

/* =====================================================================
   interview drill: seeded, from the engine's arithmetic
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed*7919+3), pk=a=>a[Math.floor(r()*a.length)];
  const a=pk([3000,4000,6000]), g=pk([800,1200,1500,2000]), n=pk([10,15,20,25]);
  const last=a+(n-1)*g, total=n*a+g*n*(n-1)/2;
  return {seed,a,g,n,last,total};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#ana-drill-q',app), an=$('#ana-drill-a',app);
  qd.innerHTML=tx(`An agent’s first call sends <b>${fT(d.a)}</b> tokens. After every call the conversation grows by <b>${fT(d.g)}</b> tokens (the reply plus the tool result). The task takes <b>${d.n}</b> calls. How many tokens does the last call send, and how many does the model read in all?`,
                  `一个 agent 第 1 次调用发出 <b>${fT(d.a)}</b> 个 token。每次调用之后，对话增长 <b>${fT(d.g)}</b> 个 token（回复加工具结果）。任务一共调用 <b>${d.n}</b> 次。最后一次调用发出多少 token？模型一共读进去多少？`);
  an.hidden=!drillOpen; if(!drillOpen) return;
  const eq=EQ([[tx('\\text{last call}','\\text{最后一次}'),'a + (n - 1) \\times g'],['',`${fT(d.a)} + (${d.n} - 1) \\times ${fT(d.g)}`],['',fT(d.last)],
                [tx('\\text{all calls}','\\text{所有调用}'),'n \\times a + g \\times \\frac{n(n - 1)}{2}'],['',`${d.n} \\times ${fT(d.a)} + ${fT(d.g)} \\times \\frac{${d.n} \\times ${d.n-1}}{2}`],['',fint(d.total)]]);
  an.innerHTML=tx(`<p><b>The last call sends ${fT(d.last)} tokens; the model reads ${fK(d.total)} in all, ${fixd(d.total/d.last,1)} times the last call.</b></p><p>a is the first call’s tokens, g the growth after each call and n the number of calls.</p>${eq}<p>The first call’s tokens are sent n times, and each growth is resent by every later call.</p>`,
    `<p><b>最后一次发出 ${fT(d.last)} 个 token；模型一共读进去 ${fK(d.total)} 个，是最后一次的 ${fixd(d.total/d.last,1)} 倍。</b></p><p>a 是第 1 次发出的 token 数，g 是每次调用之后对话增长的 token 数，n 是调用次数。</p>${eq}<p>第 1 次的内容被发了 n 遍。之后每一次新增的内容，也会被后面的每一次调用重发。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={lat:{en:'Fixed cost per call',zh:'每次调用的固定开销'},pre:{en:'Reading speed (tokens/s)',zh:'读的速度（token/秒）'},dec:{en:'Writing speed (tokens/s)',zh:'写的速度（token/秒）'}};
const TERM={
 lat:()=>tx(`t₀ is the fixed cost of one call, ${fS(S.lat)} s here (assumed): network and queueing.`,`t₀ 是一次调用的固定开销，这里是 ${fS(S.lat)} 秒（假设值）：网络和排队。`),
 pre:()=>tx(`The reading speed, ${fint(S.pre)} tokens per second here (assumed).`,`读的速度，这里是每秒 ${fint(S.pre)} 个 token（假设值）。`),
 dec:()=>tx(`The writing speed, ${S.dec} tokens per second here (assumed).`,`写的速度，这里是每秒 ${S.dec} 个 token（假设值）。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: the speeds do not change the limits.','挑战中固定不变：速度不影响这两个上限。');
  if(mode==='guided'&&!isP('time')&&CTLN[k]) return tx('Fixed in this part; you can change it in part 5 or in Scenarios.','这一部分固定不变；在第 5 部分或应用场景里可以改。');
  return '';
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
function jump(k,src){
  if(!app) return;
  if(k==='controls'){const c=$('#ana-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=$(`#ana-ctl-${k}`,app); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...BASE}; pos=curRun().events.length; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chO.lim,chO.bud).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(playing) pause(); else play(); A.tip.quiet(); return;}
  if(a==='back1'){stepBy(-1); return;}
  if(a==='fwd1'){stepBy(1); return;}
  if(a==='to-q'){const q=$('#ana-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#ana-ic-fx',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-par'){const q=$('#ana-ic-par',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ana-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#ana-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='callk') setCall(v);   /* stepping through the calls, not a setting: no ▲▼ (R6.6) */
    else if(k==='scrub'){pause(); pos=v; const run=stageRun(); if(v>=run.events.length) watched.add(runSig(run)); if(v>=run.events.length&&mode==='guided'&&isP('who')&&!replayDone){finishReplay(); return;} render();}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#ana-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ana-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; s.showMe(); revealSoon(); wantObs=false;} return;}   /* the lab sets the state: no ▲▼ (R6.6) */
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ana-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null){const host=b.closest('.ana-lv'); if(!host) return; const f=()=>byLearner(()=>host.id==='ana-ic-fx'?setFx(d.lever):setPar(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const sg=b.closest('.seg-ctl[data-key]');
    if(sg&&d.v!=null){const k=sg.dataset.key;
      if(k==='spd'){spd=d.v; acc=0; drawPlayer(stageRun()); return;}
      if(k==='chl'){byLearner(()=>{chO.lim=+d.v; render();}); return;}
      if(k==='chb'){byLearner(()=>{chO.bud=+d.v; render();}); return;}
      if(k==='run'){bySettings(b,()=>{pause(); S.run=d.v; pos=build(S.run,S).events.length; render();}); return;}
      if(['lim','pre','dec','lat','tt'].includes(k)){bySettings(b,()=>setState({[k]:+d.v})); if(k==='lim'&&mode==='sandbox') pos=curRun().events.length, render(); return;}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; sbTouched=true; pause(); Object.assign(S,p.s); pos=curRun().events.length; render();   /* a scenario is not compared with the one clicked before: no ▲▼ (R6.6) */ requestAnimationFrame(()=>{if(app) revealBox($('#ana-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ana-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12 07:13 UTC): three sections read before the guided parts, text and formulas
   only. A model only writes text, so it acts through tool calls; a task of many steps is a loop that appends each reply and its tool
   results to the conversation; the model keeps nothing between calls, so every call reads the whole context, and a longer one costs
   more and is harder to use. They state mechanisms only: no token count, time or number of calls of the guided parts. The loop
   picture was dropped after the first-time reader's check (it only restated the text)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=labLink(id); return tx(a,`“${a}”`);};
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return h.startsWith('<div')?h:`<p>${h}</p>`;}).join('')});
const prFx=t=>()=>`<div class="ana-prfx">${prM(t)}</div>`;   // a formula on its own line
A.registerPrimer(ID,{min:6,sections:[
 {h:()=>tx('The model only writes text, so it acts through tool calls (tool use)','模型只会写字，要做事就靠工具调用（tool use）'),parts:[prP(
   ()=>tx(`A language model does one thing: from all the tokens so far it writes the next token, one after another, until a reply is done (see ${prLK('architecture')} for what a token is and how the next one is computed). Its output is text and nothing else. Much work cannot be done by writing alone: looking up the weather now, finding an order in your company’s database, running a command on a server. The model knows only what it saw in training, but the weather keeps changing; the order sits in your database; the command has to run on a machine. All of it lies outside the model.`,
          `语言模型只做一件事：根据前面的全部 token，写出下一个 token，一个接一个，直到写完一段回复（token 是什么、下一个 token 怎样算出来，见${prLK('architecture')}）。它产出的只有文字。可很多事光写字做不成：查一下现在的天气，在公司的数据库里找一条订单，在服务器上跑一条命令。模型知道的只有训练时见过的东西，可天气一直在变；订单在你的数据库里；命令要在机器上执行。这些都在模型外面。`),
   ()=>tx('The way out is a division of labour: the model only writes, and the acting is done by the program that calls the model, “your code” from here on. Every time your code calls the model, it sends a list of tool definitions along: each tool’s name, what it does, and its parameters with their names and types. The tool definitions are input the model reads, just like the conversation. Models are trained so that when a tool would help, they do not answer directly but write a call in an agreed format: which tool, with which arguments, for example the tool <code>get_weather</code> with the arguments <code>{"city": "Beijing"}</code>. This is tool use, also called function calling.',
          '办法是分工：模型只管写，动手的是调用模型的那段程序，下面叫“你的代码”。你的代码每次调用模型，都附上一份工具说明：每个工具叫什么、做什么、要哪些参数，参数叫什么、是什么类型。工具说明和对话一样，是模型要读的输入。模型经过专门的训练，遇到用得上工具的时候，不直接作答，而是按约定的格式写出一个调用：用哪个工具，参数是什么，比如工具 <code>get_weather</code>，参数 <code>{"city": "北京"}</code>。这就是工具调用（tool use，也叫 function calling）。'),
   ()=>tx('The call is still only text, and writing it makes nothing happen. Your code reads it, runs its own <code>get_weather</code> function with those arguments, and hands the output back to the model as a new message; only after reading it does the model write its answer for the user. One reply can ask for several tools at once, so each call and its result have to be matched up. In a common API message format a message can be made of several blocks: each call the model writes is a tool_use block, holding the tool’s name, the arguments and an id; each result your code hands back is a tool_result block, holding the tool’s output and the id of the tool_use it answers.',
          '写出来的调用仍然只是一段文字，本身不会让任何事发生。你的代码读到它，用这些参数执行自己的 <code>get_weather</code> 函数，再把得到的输出作为一条新消息交还给模型；模型读了输出，才写出给用户的回答。一次回复可以同时要好几个工具，所以每个调用和它的结果都要能对上。在常见的 API 消息格式里，一条消息可以由几块内容组成：模型写出的每个调用是一个 tool_use 块，含工具名、参数和一个 id；你的代码交还的每个结果是一个 tool_result 块，含工具的输出和它回答的那个 tool_use 的 id。'))]},
 {h:()=>tx('The agent loop: call the model, run the tools, append the results','agent 循环：调用模型、执行工具、把结果接回对话'),parts:[prP(
   ()=>tx('One tool call is enough to answer “how warm is it in Beijing now?”. A real task takes many steps, and what each step does depends on what the step before it returned. Tracking down an error on a live service, say: only after reading the log do you know which config file to open; only after reading the config do you know which line to change; and after the change you have to read the log again to know whether it worked. Such steps cannot be planned at the start: the model has to see each result before it decides on the next step.',
          '一次工具调用，够回答“北京现在几度”。可真正的任务要走很多步，而且每一步做什么，要看上一步得到了什么。比如排查一个线上服务的报错：看了日志，才知道该打开哪个配置文件；读了配置，才知道该改哪一行；改完还得再看一次日志，才知道改对没有。这样的步骤没法一开始就排好，模型得每看到一个结果，再决定下一步。'),
   ()=>tx(`So your code puts the model call in a loop; each pass is a round, and each round calls the model once. Write ${prM('C_i')} for everything sent to the model in round ${prM('i')}. In round 1, ${prM('C_1')} holds three things: the system prompt (a short instruction at the very top that tells the model its role and how to work), the tool definitions, and the task the user gave. The model reads ${prM('C_i')} and writes a reply ${prM('y_i')}. If ${prM('y_i')} contains tool calls, your code runs them and gets their results ${prM('o_i')}, then appends the reply and the results to ${prM('C_i')}, in that order, as the next round’s input:`,
          `所以你的代码把调用模型放进一个循环，每转一圈叫一轮，每轮调用一次模型。记第 ${prM('i')} 轮发给模型的全部输入为 ${prM('C_i')}。第 1 轮的 ${prM('C_1')} 有三样：system prompt（放在最前面、交代模型身份和做事规矩的一段说明）、工具说明，以及用户交代的任务。模型读 ${prM('C_i')}，写出回复 ${prM('y_i')}；如果 ${prM('y_i')} 里有工具调用，你的代码就去执行，得到结果 ${prM('o_i')}，再把回复和结果依次接到 ${prM('C_i')} 后面，作为下一轮的输入：`),
   prFx('C_{i+1} = (C_i,\\ y_i,\\ o_i)'),
   ()=>tx(`The model and the tools, plus your code that ties them into this loop, together make an agent; your code’s role in it is called the harness. In each round the model produces only ${prM('y_i')}; sending ${prM('C_i')}, running the tools and building ${prM('C_{i+1}')} are all the harness’s work.`,
          `模型和工具，加上把它们串成循环的你的代码，合起来就是一个 agent；你的代码在其中的角色叫 harness。每一轮里，模型只产出 ${prM('y_i')}；发送 ${prM('C_i')}、执行工具、拼出 ${prM('C_{i+1}')}，都是 harness 的事。`),
   ()=>tx('The loop goes round until a reply contains no tool call. Each reply carries a field, stop_reason, saying why it ended. Its two most common values are tool_use, when the reply stops at a tool call and waits for your code to hand back the results, and end_turn, when the model has finished and the reply is the final answer. The loop itself has only one exit, end_turn, and a model may keep asking for tools without ever stopping, so a harness usually sets limits of its own as well, such as a maximum number of model calls.',
          '循环一轮接一轮地转，直到某一轮的回复里不再有工具调用。每个回复都带一个字段 stop_reason，说明它为什么到此结束。最常见的两个值是：tool_use，回复停在工具调用上，等你的代码交回结果；end_turn，模型写完了，这一轮的回复就是最后的答案。循环本身只有 end_turn 这一个出口，而模型也可能一直要工具、停不下来，所以 harness 通常还会自己设上限，比如最多调用多少次模型。'))]},
 {h:()=>tx('The model keeps nothing (statelessness): every call rereads the whole context','模型不记事（无状态）：每次调用都重读整个上下文'),parts:[prP(
   ()=>tx(`In the loop above, ${prM('C_{i+1}')} contains all of ${prM('C_i')}: every round sends everything before it again, not only what is new. That is because the model keeps nothing from one call to the next; this is called being stateless. Writing a reply can be seen as drawing from a conditional distribution:`,
          `上一节的循环里，${prM('C_{i+1}')} 包含 ${prM('C_i')} 的全部内容：每一轮都把前面的东西整段再发一次，而不是只发新增的那一段。这是因为模型在两次调用之间什么都不留下，这叫无状态（stateless）。模型写出回复，可以看成从一个条件分布里抽样：`),
   prFx('y_i \\sim p_\\theta(\\cdot \\mid\\, C_i)'),
   ()=>tx(`${prM('\\theta')} is the model’s parameters (see ${prLK('architecture')}); they change only during training, never while the model is being called. So the distribution of reply ${prM('i')} depends, apart from the fixed ${prM('\\theta')}, on ${prM('C_i')} alone. Whatever survives from one call to the next lies outside the model, such as the conversation your code keeps; for the model to use it, the next call has to hand it in again as input. Whatever the model “knows” in round ${prM('i')}, the task, the files it has read, its own earlier calls, has to be written in ${prM('C_i')}. The whole input sent to the model is called its context.`,
          `${prM('\\theta')} 是模型的参数（见${prLK('architecture')}），它们只在训练时改变，调用模型的时候不变。所以第 ${prM('i')} 次回复的分布，除了固定的 ${prM('\\theta')}，只取决于 ${prM('C_i')}。两次调用之间留下来的东西都在模型外面，比如你的代码手里的对话记录；要让模型用上，下一次调用就得把它作为输入再交给模型。模型在第 ${prM('i')} 轮“知道”的一切，任务、读过的文件、自己之前的调用，都得写在 ${prM('C_i')} 里。发给模型的这一整段输入，叫上下文（context）。`),
   ()=>tx(`The longer the context, the more each call costs. Without a cache, all of ${prM('C_i')} is this call’s prompt: the longer it is, the longer the time to first token (TTFT; see the basics of ${prLK('prefill-decode')}), and the more the provider charges, since it bills by input tokens. Some servers cache the intermediate results they computed and reuse them for a request that starts the same way (see ${prLK('prefix-caching')}). In the agent loop ${prM('C_{i+1}')} starts with exactly ${prM('C_i')}, so on a cache hit only the newly appended ${prM('y_i')} and ${prM('o_i')} have to be computed, which saves time, and the cached part is usually billed at a lower price. It still counts as input, though, and the model still uses the whole of ${prM('C_{i+1}')}.`,
          `上下文越长，每次调用的代价越大。不计缓存时，${prM('C_i')} 整段就是这次调用的 prompt：它越长，首 token 延迟（TTFT）越长（见${prLK('prefill-decode')}的知识梳理），服务商按输入的 token 数收的费也越多。有的服务器会把算过的中间结果缓存起来，遇到开头相同的请求直接复用（见${prLK('prefix-caching')}）。agent 循环里 ${prM('C_{i+1}')} 的开头正是 ${prM('C_i')}，命中缓存时，只有新接上的 ${prM('y_i')} 和 ${prM('o_i')} 要算，省下了时间，缓存的那段收费通常也低一些。但缓存的那段照样算在输入里，模型用到的仍是整段 ${prM('C_{i+1}')}。`),
   ()=>tx(`A longer context is also harder for the model to use well. In every round the model has to find what this step needs somewhere in all of ${prM('C_i')}: the files it read and the command output it got early on are mostly of no use later, yet all of it stays in ${prM('C_i')}. And one call can read only so many tokens (its context window). So what goes into the context, and in what form, is something to design (see ${prLK('context-eng')}).`,
          `上下文越长，模型也越难用好它。每一轮，模型都得在整段 ${prM('C_i')} 里找出这一步用得上的东西：早先读过的文件、跑过的命令输出，到后面多半用不上，却都还留在 ${prM('C_i')} 里。一次调用能读的 token 数也有上限（上下文窗口）。所以上下文里放什么、以什么形式放，是要专门设计的事（见${prLK('context-eng')}）。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, what was tried and watched, the challenge pick and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chO:{...chO},chSubmitted,chReveal,spd,replayDone,callK,callMax,
    fx,fxTried:[...fxTried],par,parTried:[...parTried],remView,sbTouched,drillSeed,drillOpen,watched:[...watched],
    passed:mode==='challenge'&&chSubmitted&&chEval(chO.lim,chO.bud).ok}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(tokOf(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, FXV=FX.map(x=>x.v), PARV=PAR.map(x=>x.v);
    S={...BASE,...p}; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?{...BASE,...sv}:null;
    const c=o.chO&&typeof o.chO==='object'?o.chO:{}; chO={lim:CH.lims.includes(c.lim)?c.lim:CH.start.lim,bud:CH.buds.includes(c.bud)?c.bud:CH.start.bud};
    chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    if(Object.keys(SPD).includes(o.spd)) spd=o.spd; replayDone=bool(o.replayDone);
    callMax=ints(o.callMax,1,8)?o.callMax:1; callK=ints(o.callK,1,callMax)?o.callK:1;
    fx=FXV.includes(o.fx)?o.fx:'std'; fxTried=new Set(['std',fx,...(Array.isArray(o.fxTried)?o.fxTried.filter(v=>FXV.includes(v)):[])]);
    par=PARV.includes(o.par)?o.par:'seq'; parTried=new Set(['seq',par,...(Array.isArray(o.parTried)?o.parTried.filter(v=>PARV.includes(v)):[])]);
    remView=REMS.includes(o.remView)?o.remView:'who'; sbTouched=bool(o.sbTouched); drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen);
    if(Array.isArray(o.watched)) o.watched.forEach(w=>{try{const a=typeof w==='string'&&w.length<200?JSON.parse(w):null; if(Array.isArray(a)&&a.length===8) watched.add(w);}catch(e){}});
    pos=isP('who')&&!replayDone?0:stageRun().events.length;},   // the stage shows the run finished, as a scenario or a part does; part 2 not yet watched replays from the start
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...BASE,...p}; mode='sandbox'; sbTouched=true; pos=curRun().events.length;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#ana-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ana-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    fillRail();
    render();
    if(isP('who')&&!replayDone&&!playing&&!relang) play();   // a language switch keeps a paused replay paused (R3.20)
  },
  leave(){clearTimeout(annT); stageStop(); pause(); root=null; app=null;},
  onLang(){if(root){const d=$('#ana-math',app), open=d&&d.open, was=playing; pause(); relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#ana-math',app).open=true; if(was) play();}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; return viewNow()==='ch'?$('#ana-chart',app):$('#ana-bars',app);},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#ana-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      if(v==='ch'){const e=chEval(chO.lim,chO.bud); vis.innerHTML=''; read.innerHTML=esc(tx(`${planName(chO.lim,chO.bud)}: the normal run ${e.ok?'finishes':'is stopped'}; the stuck run reads ${fT(e.cost)} tokens`,`${planName(chO.lim,chO.bud)}：正常的运行${e.ok?'做完了':'被拦下'}；卡住的运行读了 ${fT(e.cost)} 个 token`)); return;}
      const run=stageRun(), st=at(run,stagePos(run)), c=st.call?run.calls[st.call-1]:null;
      vis.innerHTML=miniBars(run,st);
      read.innerHTML=esc(c?tx(`Call ${c.i} of ${run.calls.length} sends ${fT(c.inTok)} tokens`,`第 ${c.i} / ${run.calls.length} 次调用发出 ${fT(c.inTok)} 个 token`):tx('Before call 1: nothing sent yet','第 1 次调用之前：还没有发出任何东西'));}
  }
});
function miniBars(run,st){const mx=Math.max(12000,...run.calls.map(c=>c.inTok));
  return `<div class="ana-dk">${run.calls.map(c=>`<i style="height:${c.i<=st.call?(c.inTok/mx*100).toFixed(1):0}%"></i>`).join('')}</div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const slim=r=>({key:r.key,n:r.n,stop:r.stop,done:r.done,totalIn:r.totalIn,totalOut:r.totalOut,finalIn:r.finalIn,tModel:r.tModel,tTool:r.tTool,time:r.time,
  calls:r.calls.map(c=>({i:c.i,inTok:c.inTok,out:c.out,tM:c.tM,tT:c.tT,tools:c.tools,end:c.end})),blocks:r.blocks.map(b=>({kind:b.kind,who:b.who,chars:b.chars,tok:b.tok,call:b.call,err:!!b.err,name:b.name||null,input:b.input||null,tid:b.tid||null})),events:r.events.map(e=>({type:e.type,call:e.call}))});
window.__ana=Object.freeze({
  build:(k,o)=>deep(slim(build(k,o||{}))), sends:(k,path,o)=>deep(resSends(build(k,o||{}),path)), chEval:(l,b)=>deep(chEval(l,b)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)),
  runDefs:()=>deep(Object.fromEntries(Object.entries(RUNS).map(([k,d])=>[k,{task:d.task,tools:{chars:d.tools.chars},final:d.final,calls:d.calls.map(st=>st.map(([u,r])=>({name:u.name,input:u.input,chars:r.chars,err:r.err})))}]))),
  consts:Object.freeze({SYS:{chars:SYS.chars},TIME0:{...TIME0},TOOLT:{...TOOLT},CH:deep(CH),PRED_RIGHT,STEP_IDS:STEPS.map(s=>s.id),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),SPD:{...SPD}}),
  state:()=>({...S,mode,step,pred,observed:{...observed},pos,playing,spd,replayDone,callK,callMax,fx,fxTried:[...fxTried],par,parTried:[...parTried],remView,chO:{...chO},chSubmitted,chReveal,stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});}
});
})();
