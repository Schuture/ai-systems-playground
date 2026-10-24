/* context-eng (world agents): What goes into an agent’s context, and what gets cut?
   Fidelity: Simulated (header: Toy run). A seeded long task; the sizes, the share a summary keeps and the trim size are the lab's
   assumptions (Fidelity card, Settings notes). The policies are run step by step on that task, so every number follows from them.
   - Every model call sends the whole context: C = H (system prompt, tool definitions, task) + S (summary, if any) + every kept item
     (one per step: the model's turn, and the tool's output or the user's message). Tokens read in total = Σ C over the calls.
   - Policies: keep everything; keep the last N steps (N counts the main agent's turns); summarize every M steps (the old items become
     one summary; a fact in it survives each summary with p); trim long tool outputs to `cap` tokens (the part the agent knew it needed);
     notes (a fact the agent knew mattered is written outside the context and fetched when needed); a sub-agent (the sub-task runs in
     its own context; the main context gets a short result). A step that looks back at an earlier output no longer in the context
     first re-reads it (one more call). A call whose context is over the window fails: the run stops there.
   - The cost with prompt caching (an assumption, priced as the Anthropic cookbook states): in each call, the part that repeats the
     previous call's beginning, piece by piece, is a cache read at 0.1 times the input price, the rest a cache write at 1.25 times. A cut
     that changes the beginning (a sliding window at every step, a summary every M steps) loses the cache from that point on.
   Words (EN / ZH): context / 上下文; window / 窗口; tokens read in total / 一共读进去的 token (agent-anatomy's words); cost with prompt
   caching / 花费 (in US dollars at an assumed $3 per million input tokens / 美元，按每 1M 个输入 token 3 美元的假设价格); step / 步;
   model call / 模型调用; trim / 修剪; summary / 摘要; notes / 笔记; sub-agent / 子 agent; re-read / 重新读; fact / 事实; part / 部分.
   window.__cte is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='context-eng';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
/* cr, cw: prompt caching as the Anthropic cookbook prices it (the lab's assumption): a cache read costs 0.1 times the input price,
   a cache write 1.25 times. usd: the assumed input price, US dollars per million tokens (the base input price of several models on the
   Claude API pricing page, platform.claude.com/docs/en/about-claude/pricing, read with curl on 2026-10-04); the cost is shown in dollars, rounded, because nobody needs the exact value (R6.20) */
const K={rr:120,fetch:150,write:60,S:2000,Hs:3000,R:800,cr:0.1,cw:1.25,usd:3};
const rng=(a,b)=>{const o=[]; for(let i=a;i<=b;i++) o.push(i); return o;};
const longOf=(list,kind,lo,hi)=>Object.fromEntries(list.map(i=>[i,[kind,lo,hi]]));
const TASKS={
 code:{n:60,H:5000,seed:7,m0:150,m1:300,o0:200,o1:1300,conv:0,sub:[21,32],
  long:{...longOf([5,9,14,37,44,50],'file',7000,10000),...longOf([18,41,55],'log',4000,6000),...longOf([22,24,26,28,30],'page',9000,13000)},
  refs:{8:5,12:9,16:14,20:18,35:14,39:37,43:41,46:9,47:44,53:50,58:55,59:37},
  facts:[{id:'a',at:3,need:48,known:1},{id:'b',at:12,need:52,known:1},{id:'c',at:27,need:56,known:1}]},
 research:{n:60,H:4000,seed:5,m0:150,m1:300,o0:200,o1:800,conv:0,sub:[31,45],
  long:longOf([4,7,10,13,16,19,22,25,32,34,36,38,40,42,44,48,51,54,57],'page',6000,10000),
  refs:{11:10,17:16,26:25,49:48,58:57,52:4},
  facts:[{id:'a',at:2,need:55,known:1},{id:'b',at:10,need:50,known:1},{id:'c',at:38,need:58,known:1}]},
 chat:{n:120,H:2500,seed:3,m0:200,m1:400,o0:30,o1:170,conv:1,sub:null,long:{},refs:{},
  facts:[{id:'a',at:2,need:115,known:1}]},
 ops:{n:80,H:6000,seed:11,m0:150,m1:300,o0:200,o1:1000,conv:0,sub:[25,40],
  long:{...longOf([6,11,15,20,44,53,58,63,69,74],'log',5000,8000),48:['log',6000,9000],...longOf([26,28,30,32,34,36,38,40],'dash',6000,9000)},
  refs:{9:6,13:11,17:15,22:20,46:44,50:48,55:53,60:58,65:63,71:69,76:74,57:20,68:44},
  facts:[{id:'a',at:2,need:72,known:1},{id:'b',at:15,need:61,known:1},{id:'c',at:48,need:66,known:0}]}
};
/* the random numbers: a counter-based hash, one number per (task seed, step, purpose: 0 the model's turn, 1 the output, 2 a long output) */
const mix=h=>{h^=h>>>16; h=Math.imul(h,0x7feb352d); h^=h>>>15; h=Math.imul(h,0x846ca68b); h^=h>>>16; return h>>>0;};
function u01(seed,i,k){let h=mix(Math.imul(seed,0x9E3779B1)^0x85ebca6b); h=mix(h^Math.imul(i+1,0x27d4eb2f)); h=mix(h^Math.imul(k+1,0x165667b1)); return h/4294967296;}
const SC={};
function stepsOf(id){
  if(SC[id]) return SC[id];
  const T=TASKS[id], st=[null];
  for(let i=1;i<=T.n;i++){
    let o=T.o0+Math.floor(u01(T.seed,i,1)*T.o1), kind=T.conv?'msg':'small';
    const lg=T.long[i]; if(lg){kind=lg[0]; o=lg[1]+Math.floor(u01(T.seed,i,2)*(lg[2]-lg[1]));}
    st.push({m:T.m0+Math.floor(u01(T.seed,i,0)*T.m1),o,kind,ref:T.refs[i]||0});
  }
  return (SC[id]=st);
}
const BASE={task:'code',W:131072,hist:'all',N:10,M:20,trim:0,notes:0,sub:0,p:0.8,cap:1500};
/* one run of a policy on a task: every main model call with its context, its parts and where each fact is */
function simulate(s){
  const T=TASKS[s.task], st=stepsOf(s.task), W=s.W, useSub=!!s.sub&&!!T.sub, sa=useSub?T.sub[0]:0, sb=useSub?T.sub[1]:-1;
  const osz=i=>s.trim&&!T.conv?Math.min(st[i].o,s.cap):st[i].o;
  const keeps=(f,i)=>!!f.known||!s.trim||!!T.conv||st[i].o<=s.cap;      // trimming keeps only what the agent knew it needed
  let items=[], summ=0, sid=0, uid=0, prev=null, read=0, subRead=0, cost=0, hit=0, subCost=0, subHit=0, peak=0, over=0, turn=0, nre=0, nsum=0, nsub=0;
  const inS={}, calls=[], avail={};
  const ctx=()=>{let cv=0, tl=0; for(const x of items){cv+=x.cv; tl+=x.tl;} return {tot:T.H+summ+cv+tl,seg:[T.H,summ,cv,tl]};};
  const hasStep=j=>items.some(x=>x.steps.includes(j)), hasFact=id=>items.some(x=>x.fs.includes(id));
  const where=i=>{const o={}; for(const f of T.facts) o[f.id]=i<f.at?'pre':(hasFact(f.id)?'raw':(s.notes&&f.known?'note':(inS[f.id]?'summ':'gone'))); return o;};
  /* what a call sends, in order: the head, the summary, the kept items, each with its identity; the cache covers the pieces that
     repeat the previous call's beginning, up to the first piece that differs */
  const order=()=>{const q=[['H',T.H]]; if(summ) q.push(['S'+sid,summ]); for(const x of items) q.push([x.id,x.cv+x.tl]); return q;};
  const call=(k,i,x)=>{const c=ctx(), r={k,step:i,ctx:c.tot,seg:c.seg,fs:where(i),...(x||{})}; calls.push(r); if(c.tot>W){r.k='over'; over=i; return false;}
    const q=order(); let h=0; if(prev) for(let a=0;a<q.length&&a<prev.length&&q[a][0]===prev[a][0]&&q[a][1]===prev[a][1];a++) h+=q[a][1];
    prev=q; r.hit=h; r.cost=K.cr*h+K.cw*(c.tot-h); cost+=r.cost; hit+=h; read+=c.tot; if(c.tot>peak) peak=c.tot; return true;};
  for(let i=1;i<=T.n;i++){
    if(useSub&&i>sa&&i<=sb) continue;
    turn++; const x=st[i], j=x.ref;
    if(j&&!hasStep(j)){if(!call('re',i,{ref:j})) break; nre++; items.push({id:++uid,turn,cv:K.rr,tl:osz(j),steps:[j],fs:T.facts.filter(f=>f.at===j&&keeps(f,j)).map(f=>f.id)});}
    let extra=0; const now={};
    for(const f of T.facts) if(f.need===i){now[f.id]=hasFact(f.id)?1:(s.notes&&f.known?1:(inS[f.id]?Math.pow(s.p,inS[f.id]):0)); if(s.notes) extra+=K.fetch;}
    if(!call('st',i,{deleg:useSub&&i===sa,need:Object.keys(now)})) break;
    Object.assign(avail,now);
    if(s.notes) extra+=K.write*T.facts.filter(f=>f.at===i&&f.known).length;
    /* the sub-agent only appends, so each of its calls repeats the whole previous one; its first call writes everything */
    if(useSub&&i===sa){let sc=0, pc=0; for(let k=sa;k<=sb;k++){const c=K.Hs+sc; subRead+=c; subHit+=pc; subCost+=K.cr*pc+K.cw*(c-pc); pc=c; nsub++; sc+=st[k].m+osz(k);} read+=subRead; cost+=subCost; hit+=subHit;
      items.push({id:++uid,turn,cv:x.m+extra,tl:K.R,steps:rng(sa,sb),fs:T.facts.filter(f=>f.at>=sa&&f.at<=sb&&f.known).map(f=>f.id)});}
    else items.push({id:++uid,turn,cv:x.m+extra+(T.conv?x.o:0),tl:T.conv?0:osz(i),steps:[i],fs:T.facts.filter(f=>f.at===i&&keeps(f,i)).map(f=>f.id)});
    if(s.hist==='last') items=items.filter(it=>it.turn>turn-s.N);
    if(s.hist==='sum'&&i%s.M===0&&i<T.n){if(!call('sum',i)) break; nsum++; for(const f of T.facts) if(hasFact(f.id)||inS[f.id]) inS[f.id]=(inS[f.id]||0)+1; summ=K.S; sid=++uid; items=[];}
  }
  for(const f of T.facts) if(!(f.id in avail)) avail[f.id]=0;
  let all=1; for(const f of T.facts) all*=avail[f.id];
  const sure=T.facts.filter(f=>avail[f.id]>=1).length;
  return {calls,over,peak,read,subRead,cost,hit,subCost,subHit,nre,nsum,nsub,ncalls:calls.filter(c=>c.k!=='over').length+nsub,avail,succ:over?0:all,sure,nf:T.facts.length,W,task:s.task};
}
const RC=new Map();
const keyOf=s=>[s.task,s.W,s.hist,s.N,s.M,s.trim,s.notes,s.sub,s.p,s.cap].join('|');
function run(s){const k=keyOf(s); if(RC.has(k)) return RC.get(k); const r=simulate(s); if(RC.size>80) RC.clear(); RC.set(k,r); return r;}
/* the running totals at the play head (pv calls shown) */
function liveOf(r,pv){
  let read=0, cost=0, hit=0, peak=0, n=0, sub=0, nre=0, nsum=0; const T=TASKS[r.task];
  for(let i=0;i<Math.min(pv,r.calls.length);i++){const c=r.calls[i]; if(c.k==='over') break; read+=c.ctx; cost+=c.cost; hit+=c.hit; n++; if(c.ctx>peak) peak=c.ctx; if(c.k==='re') nre++; if(c.k==='sum') nsum++; if(c.deleg){read+=r.subRead; cost+=r.subCost; hit+=r.subHit; sub=r.nsub;}}
  const lastStep=pv>0?r.calls[Math.min(pv,r.calls.length)-1].step:0;
  const checked=T.facts.filter(f=>f.need<=lastStep&&(f.id in r.avail)&&!(r.over&&f.need>=r.over));
  return {read,cost,hit,peak,calls:n+sub,nre,nsum,nsub:sub,lastStep,checked};
}
/* the challenge: an on-call agent chasing an outage, 80 steps, three facts; the third only turns out to matter later */
const CH={hists:[['all',0],['last',5],['last',10],['last',20],['last',30],['sum',10],['sum',20]],start:{h:0,trim:0,notes:0,sub:0}};
const chState=o=>{const [hist,v]=CH.hists[o.h]; return {...BASE,task:'ops',hist,N:hist==='last'?v:10,M:hist==='sum'?v:20,trim:o.trim,notes:o.notes,sub:o.sub};};
function chEval(o){const r=run(chState(o)); const okW=!r.over, okF=r.sure===r.nf; return {...o,read:r.read,cost:r.cost,peak:r.peak,over:r.over,avail:r.avail,sure:r.sure,okW,okF,ok:okW&&okF};}
function chAll(){const out=[]; CH.hists.forEach((_,h)=>[0,1].forEach(trim=>[0,1].forEach(notes=>[0,1].forEach(sub=>out.push(chEval({h,trim,notes,sub})))))); return out;}
/* minimal: the lowest cost with prompt caching, as shown (3 significant digits) */
const r3=x=>+x.toPrecision(3);
function chBest(){const ok=chAll().filter(e=>e.ok); if(!ok.length) return null; ok.sort((a,b)=>a.cost-b.cost); return ok[0];}

/* =====================================================================
   formatting
   ===================================================================== */
const fI=n=>A.fmt.int(n);
const fB=n=>A.fmt.big(n);
const fBig=n=>n>=1e6&&A.ui.big?A.ui.big(n):fI(n);
/* tokens read and the cost share one unit in a view (R6.12): M, with two decimals below 1M; no exact value on hover (the owner, 2026-10-07) */
const fM=n=>n>=1e6?A.fmt.big(Math.round(n)):(n/1e6).toFixed(2)+'M';
/* the cost in US dollars at the assumed price, 2 significant digits: $0.71, $1.1, $13 */
const fD=n=>{const d=n*K.usd/1e6; return d<=0?'$0':'$'+(d>=100?String(Math.round(d)):d.toPrecision(2));};
/* comparisons in words, never two exact values (R6.20): “about 2.5 times”, “more than 2 times”, “about a quarter” */
const xAbout=q=>tx(`about ${q.toFixed(1)} times`,`约 ${q.toFixed(1)} 倍`);
const xMore=q=>tx(`more than ${Math.floor(q)} times`,`${Math.floor(q)} 倍多`);
const FR=[[2,'half','一半'],[3,'a third','三分之一'],[4,'a quarter','四分之一'],[5,'a fifth','五分之一']];
const xPart=q=>{const f=FR.reduce((a,b)=>Math.abs(1/b[0]-q)<Math.abs(1/a[0]-q)?b:a); return tx(`about ${f[1]}`,`约${f[2]}`);};
const fP=x=>{const v=x*100; if(x<=0) return '0%'; if(x>=1) return '100%'; return (v<10?v.toFixed(1):String(Math.round(v)))+'%';};
const fW=W=>A.fmt.tok(W);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const M=t=>`<div class="cte-fx">${A.ui.math?A.ui.math(t):esc(t)}</div>`;
const Mi=t=>A.ui.math(t);   // inline, inside a sentence
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="cte-eqs${EQN(rows)?' cte-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="cte-eq-l">${l?Mi(l):''}</span><span class="cte-eq-e">${Mi(rel||'=')}</span><span class="cte-eq-r">${Mi(r)}</span>`).join('')}</div>`;
/* a long sum, a few terms a line (the call's four parts, with their names, one a line): the first line after “=”, the next ones start with “+” under it (a formula does not wrap by itself, R6.33) */
const SUMROWS=(name,terms,per=2)=>{const rows=[]; for(let i=0;i<terms.length;i+=per){const ch=terms.slice(i,i+per).join(' + '); rows.push(i===0?[name,ch]:['',`+ ${ch}`,' ']);} return rows;};
const MB=(lead,body,res)=>`<div class="cte-mb"><p>${lead}</p>${body}${res?`<p class="cte-mb-res">${res}</p>`:''}</div>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const stepW=n=>tx(n===1?'step 1':`step ${n}`,`第 ${n} 步`);

/* =====================================================================
   module state
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lv2='', tried2=new Set(), lv3='all', tried3=new Set(['all']), lv4='slide', tried4=new Set(['slide']), lv5='all', tried5=new Set(['all']), remView='';
let runsSeen=false, saved=null, chO={...CH.start}, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, hl='';
let drillSeed=1, drillOpen=false, wantDelta=false;   // a learner's change waits for the end of its run to show ▲▼ (R6.6)
const shownPrev={};
const PL={pv:0,on:false,speed:'f',raf:0,last:0,acc:0};   // R3.9: a whole run of many calls defaults to fast
const SPD={s:30,m:15,f:5};          // seconds for a whole run, whatever its number of calls (R3.9: time-based)
const R_=()=>run(S);
/* R3.20: what a run shows on screen; a change that leaves it the same does not replay it */
const sigOf=r=>JSON.stringify([r.task,r.calls,r.nsub,r.subRead]);
const shownSig=()=>PL.pv>0||PL.on?sigOf(R_()):'';
/* R3.20: the runs the reader has watched to the end, by what they draw (not by part); shown again, a watched run shows its end at once */
const watched=new Set();
/* the runs watched before a reload, as short hashes of their pictures (a picture is a few thousand characters): round 3 #6 */
const WH=new Set(), hkey=k=>{let h=2166136261; for(let i=0;i<k.length;i++){h^=k.charCodeAt(i); h=Math.imul(h,16777619);} return (h>>>0).toString(36);}, wasWatched=sg=>watched.has(sg)||WH.has(hkey(sg));

/* =====================================================================
   words and definitions (R2.8)
   ===================================================================== */
const TIP={
 step:()=>tx('A step is one turn of the agent: one call to the model, usually followed by one tool call (read a file, run the tests, open a web page); the tool’s result goes back to the model for the next step.',
             '一步就是 agent 的一个回合：调用一次模型，模型通常再调用一次工具（读文件、跑测试、打开网页）；工具的结果交回给模型，接着做下一步。'),
 context:()=>tx('The context is everything sent to the model in one call: the system prompt, the tool definitions, the task, and the conversation so far, tool outputs included. The model remembers nothing between calls; it sees only this.',
                '上下文就是一次调用发给模型的全部内容：system prompt、工具定义、任务，以及到目前为止的对话，工具的输出也在里面。模型在两次调用之间什么都不记得，它只看得见这些。'),
 window:()=>tx('The context window is the most tokens the model can take in one call. A call with a longer context fails. 128K is 131,072 tokens.',
               '上下文窗口是模型一次调用最多能读的 token 数。上下文比它长，这次调用就会失败。128K 就是 131,072 个 token。'),
 read:()=>tx('Every call reads its whole context again, so the tokens read in total are the sum of the contexts of all the calls. Without prompt caching, this is what you pay for and wait for. With caching, see the cost.',
             '每次调用都把整个上下文重新读一遍，所以一共读进去的 token，是所有调用的上下文加起来。没有 prompt caching 时，花的钱和等的时间都跟着它走。有缓存时的花费，看“花费”那一项。'),
 cost:()=>tx(`What the run costs with prompt caching, in US dollars. The input price is assumed to be $${K.usd} per million tokens. In each call, the part that repeats the previous call’s beginning is read from the cache at 0.1 times that price. The rest is written to the cache at 1.25 times. The two multipliers are the ones in Anthropic’s cookbook. Other providers work differently: OpenAI, for example, caches prompts over 1,024 tokens automatically and discounts the cached part.`,
             `有 prompt caching 时，这次运行花多少钱，单位是美元。这里假设输入原价是每 1M 个 token ${K.usd} 美元。每次调用里，和上一次调用开头相同的部分从缓存读，按原价的 0.1 倍算。其余部分要写进缓存，按原价的 1.25 倍算。这两个倍数是 Anthropic cookbook 写的。别的服务商做法不同，比如 OpenAI 会自动缓存超过 1,024 个 token 的 prompt，命中缓存的部分按折扣价算。`),
 cache:()=>tx('Prompt caching: the provider keeps the beginning of the prompts it has seen. When the next call starts with exactly the same tokens, that part is read from the cache, cheaper and faster. In this lab a cache read costs 0.1 times the input price and a cache write 1.25 times (the prices in Anthropic’s cookbook). Change one piece near the beginning, and everything after it has to be written again.',
              'prompt caching 就是服务商把见过的 prompt 的开头存起来。下一次调用的开头和它一模一样时，这部分直接从缓存读，又便宜又快。在这个 Lab 里，从缓存读按原价的 0.1 倍算，写进缓存按 1.25 倍算（Anthropic cookbook 的价格）。开头只要有一段变了，从那里往后都得重新写。'),
 trim:()=>tx('Trimming keeps only the part of a long tool output that the agent needs now, for example the matching lines instead of the whole file. In this lab a trimmed output keeps at most 1,500 tokens, and it keeps what the agent knew it needed at that moment.',
             '修剪就是工具的长输出只留下这一步要用的部分，比如只留搜到的那几行，而不是整个文件。在这个 Lab 里，修剪后最多留 1,500 个 token，留下的是 agent 当时就知道要用的内容。'),
 last:()=>tx('A sliding window keeps only the last N steps of the conversation; everything older is dropped. The system prompt, the tools and the task always stay.',
             '滑动窗口只留最近 N 步的对话，更早的全部删掉。system prompt、工具定义和任务一直留着。'),
 summary:()=>tx('A summary is also called compaction. When the context is nearly full, the model writes the conversation so far as a short summary, which replaces it. To compare runs, this lab summarizes every M steps. A summary keeps the gist, but any one fact can be left out. Here each fact survives one summary with 80%, and two summaries with 64%. There is no accepted figure; 80% is chosen so that the loss after two or three summaries shows.',
                '摘要也叫压缩（compaction）。上下文快满时，让模型把到目前为止的对话写成一段短摘要，替换掉原来的对话。为了好比较，这个 Lab 固定每 M 步压缩一次。摘要留得住大意，可某一个具体的事实可能被漏掉。这里假设每压一次，一个事实留下来的把握是 80%，压两次就只剩 64%。这个把握没有公认的数，取 80% 是为了让两三次压缩后的损失看得出来。'),
 notes:()=>tx('Notes: when the agent sees a fact it knows it will need, it writes it to a notes file outside the context, and reads it back when it needs it. Only the few tokens of the write and the read enter the context. A fact whose importance shows only later never gets written down.',
              '记笔记：agent 看到以后要用的事实，就把它写进上下文之外的一个笔记文件，用的时候再读回来。进上下文的只有写和读的那几十个 token。当时看不出要紧的事，就不会被记下来。'),
 sub:()=>tx('A sub-agent is a second agent that gets one self-contained sub-task. It works in its own, fresh context; only its short result (about 800 tokens here) comes back into the main agent’s context.',
            '子 agent 是另一个 agent，专门做一件独立的子任务。它在自己全新的上下文里做，只把一段简短的结果（这里约 800 个 token）带回主 agent 的上下文。'),
 reread:()=>tx('A step that needs an earlier tool output that is no longer in the context has to fetch it again: one more model call, and the output enters the context again.',
               '要用的旧输出已经不在上下文里了，agent 只好再读一遍：多一次模型调用，那段输出也重新进了上下文。')
};
const FACT={
 code:{a:{s:{en:'① keep public names',zh:'① 不改 public 函数名'},l:{en:'At step 3 the user adds a rule: do not rename any public function. It is needed at step 48, when the agent wants to rename parse_date.',zh:'第 3 步，用户补充了一条要求：不要改任何 public 函数的名字。第 48 步要用：agent 想给 parse_date 改个名字。'}},
       b:{s:{en:'② port 5433',zh:'② 端口 5433'},l:{en:'At step 12, in a config file: the test database listens on port 5433. It is needed at step 52, to run the integration tests.',zh:'第 12 步，在配置文件里查到：测试数据库的端口是 5433。第 52 步跑集成测试时要用。'}},
       c:{s:{en:'③ UTC by default',zh:'③ 默认按 UTC'},l:{en:'At step 27, in the library’s docs: a time without a time zone is read as UTC. It is needed at step 56, to write the fix.',zh:'第 27 步，在日期库的文档里查到：不带时区的时间一律当成 UTC。第 56 步写修复时要用。'}}},
 research:{a:{s:{en:'① 2024 or later',zh:'① 只用 2024 年后的资料'},l:{en:'At step 2 the user asks for sources from 2024 or later only. It is needed at step 55, when the agent picks its sources.',zh:'第 2 步，用户要求只用 2024 年以后的资料。第 55 步挑资料时要用。'}},
       b:{s:{en:'② a price',zh:'② 一个价格'},l:{en:'At step 10 the agent finds a price on a web page. It is needed at step 50, in the comparison table.',zh:'第 10 步，agent 在一个网页上查到一个价格。第 50 步做对比表时要用。'}},
       c:{s:{en:'③ a limit',zh:'③ 一个限制'},l:{en:'At step 38, in the sub-task, the agent finds a usage limit. It is needed at step 58, in the report.',zh:'第 38 步，子任务里查到一个用量上限。第 58 步写报告时要用。'}}},
 chat:{a:{s:{en:'① answer in Chinese',zh:'① 一律用中文回答'},l:{en:'At turn 2 the user asks for every answer in Chinese. It still holds at turn 115.',zh:'第 2 轮，用户要求每个回答都用中文。到第 115 轮，这条要求仍然有效。'}}},
 ops:{a:{s:{en:'① no database restart',zh:'① 不重启数据库'},l:{en:'At step 2 the user says: do not restart the database. It is needed at step 72.',zh:'第 2 步，用户说：不要重启数据库。第 72 步要用。'}},
      b:{s:{en:'② started 03:07',zh:'② 03:07 开始'},l:{en:'At step 15, in a log: the outage started at 03:07. It is needed at step 61, for the timeline.',zh:'第 15 步，从日志里查到故障在 03:07 开始。第 61 步写时间线时要用。'}},
      c:{s:{en:'③ reload at 03:05',zh:'③ 03:05 配置重载'},l:{en:'At step 48, one line in a long log: the config was reloaded at 03:05. Nobody can tell yet that it matters; at step 66 it turns out to be the cause.',zh:'第 48 步，一份长日志里有一行：03:05 配置重新加载过。当时看不出它要紧；到第 66 步才发现，它就是原因。'}}}
};
const TASKN={code:{en:'a coding agent fixing a date bug, 60 steps',zh:'修日期 bug 的 coding agent，60 步'},research:{en:'a research agent reading web pages, 60 steps',zh:'读网页的调研 agent，60 步'},
  chat:{en:'a chat assistant, 120 turns',zh:'聊天助手，120 轮'},ops:{en:'an on-call agent chasing an outage, 80 steps',zh:'排查线上故障的运维 agent，80 步'}};
function polName(s){
  const T=TASKS[s.task], p=[];
  const tn=!!T.conv;   // a chat counts turns, not steps
  p.push(s.hist==='all'?tx('keep everything','什么都不删'):(s.hist==='last'?tx(`keep the last ${s.N} ${tn?'turns':'steps'}`,`只留最近 ${s.N} ${tn?'轮':'步'}`):tx(`summarize every ${s.M} ${tn?'turns':'steps'}`,`每 ${s.M} ${tn?'轮':'步'}压缩一次`)));
  if(s.trim) p.push(tx('trim long outputs','修剪长输出'));
  if(s.notes) p.push(tx('notes','记笔记'));
  if(s.sub&&T.sub) p.push(tx('sub-agent','子 agent'));
  return p.join(tx(', ','，'));
}

/* =====================================================================
   guided parts
   ===================================================================== */
/* R3.21: the card gives what the estimate needs (the head, the average a step adds, the long outputs); the ranges are wide enough
   that the estimate from those facts (about 43 steps) and the run (step 38) land in the same one */
const PRED=[{v:'a',t:{en:'All 60 steps fit',zh:'60 步都放得下'}},{v:'b',t:{en:'Between step 50 and 59',zh:'第 50 到 59 步之间'}},{v:'c',t:{en:'Between step 30 and 49',zh:'第 30 到 49 步之间'}},{v:'d',t:{en:'Before step 30',zh:'第 30 步之前'}}];
const predOf=over=>!over?'a':(over>=50?'b':(over>=30?'c':'d'));
/* part 1's facts, all from the task: the head, the average a step adds over the 60 steps (to 2 significant digits), the most a short
   step adds, the long outputs (how many, their range, where the web pages are) and the estimate (window − head) ÷ average */
function missionFacts(){
  const T=TASKS.code, st=stepsOf('code'); let sum=0, smallHi=0, smallLo=1e9; const pages=[], lg=Object.values(T.long);
  for(let i=1;i<=T.n;i++){const x=st[i]; sum+=x.m+x.o; if(T.long[i]){if(T.long[i][0]==='page') pages.push(i);} else {smallHi=Math.max(smallHi,x.m+x.o); smallLo=Math.min(smallLo,x.m+x.o);}}
  const avg=sum/T.n, avgR=+avg.toPrecision(2);
  return {H:T.H,n:T.n,avg,avgR,nLong:Object.keys(T.long).length,lo:Math.min(...lg.map(v=>v[1])),hi:Math.max(...lg.map(v=>v[2])),smallLo,smallHi,pages,est:Math.floor((BASE.W-T.H)/avgR)};
}
const PRED_RIGHT=predOf(run({...BASE}).over);
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const runDone=()=>PL.pv>=R_().calls.length;
const hideRes=()=>isP('mission')&&!(pred&&runsSeen);
/* part 2: which kind fills the context, at the last call that fit */
const LV2=[{v:'head',i:0,t:{en:'System prompt, tools and task',zh:'system prompt、工具和任务'},d:{en:'Sent at every call, never cut.',zh:'每次都发，从来不删。'}},
  {v:'conv',i:2,t:{en:'The conversation',zh:'对话'},d:{en:'The model’s turns and the user’s messages.',zh:'模型的回合，和用户的消息。'}},
  {v:'tool',i:3,t:{en:'Tool outputs',zh:'工具的输出'},d:{en:'Files, test logs, web pages.',zh:'文件、测试日志、网页。'}}];
function lastFit(){const r=run({...BASE}); const ok=r.calls.filter(c=>c.k!=='over'); return ok[ok.length-1];}
const share2=v=>{const c=lastFit(), o=LV2.find(x=>x.v===v); return c.seg[o.i]/c.ctx;};
const LV3=[{v:'all',s:{hist:'all',trim:0},t:{en:'Keep everything (initial setup)',zh:'什么都不删（初始设置）'},d:{en:'Every tool output stays in full, so the context grows every step.',zh:'工具的输出原样留着，上下文一步比一步长。'}},
  {v:'trim',s:{hist:'all',trim:1},t:{en:'Trim long tool outputs',zh:'修剪工具的长输出'},d:{en:'Keep at most 1,500 tokens of each.',zh:'每段最多留 1,500 个 token。'}},
  {v:'slide',s:{hist:'last',N:10,trim:1},t:{en:'Trim, and keep only the last 10 steps',zh:'修剪，再只留最近 10 步'},d:{en:'Older steps are dropped.',zh:'更早的步骤全部删掉。'}}];
const LV4=[{v:'slide',s:{hist:'last',N:10,trim:1,notes:0},t:{en:'Trim, last 10 steps (initial setup)',zh:'修剪，只留最近 10 步（初始设置）'},d:{en:'Older steps are dropped, and the three facts go with them.',zh:'早先的步骤被删掉，三个事实也跟着没了。'}},
  {v:'sum',s:{hist:'sum',M:20,trim:1,notes:0},t:{en:'Trim, summarize every 20 steps',zh:'修剪，每 20 步压缩成摘要'},d:{en:'Old steps become a summary.',zh:'旧的步骤变成一段摘要。'}},
  {v:'notes',s:{hist:'sum',M:20,trim:1,notes:1},t:{en:'Trim, summarize every 20 steps, and notes',zh:'修剪，每 20 步压缩，再记笔记'},d:{en:'The same summaries; each fact also goes into a notes file.',zh:'摘要照旧，每个事实另外写进笔记文件。'}}];
const LV5=[{v:'all',s:{hist:'all',trim:0,notes:0,sub:0},t:{en:'Keep everything (initial setup)',zh:'什么都不删（初始设置）'},d:{en:'The agent reads the docs itself, and the pages stay in its context.',zh:'agent 自己读文档，网页都留在它的上下文里。'}},
  {v:'sub',s:{hist:'all',trim:0,notes:0,sub:1},t:{en:'Hand the docs to a sub-agent',zh:'把查文档交给子 agent'},d:{en:'Steps 21 to 32 run in its own context.',zh:'第 21 到 32 步在它自己的上下文里做。'}}];
const P3={...BASE}, P4={...BASE,hist:'last',N:10,trim:1}, P5={...BASE};
const resOf=s=>{const r=run({...S,...s}); return r.over?tx(`full at step ${r.over}`,`第 ${r.over} 步塞满`):tx(`${r.sure} of ${r.nf} facts`,`事实 ${r.sure} / ${r.nf}`)+(r.sure<r.nf&&r.succ>0?tx(`, ${fP(r.succ)}`,`，${fP(r.succ)}`):'');};
const resOf4=s=>{const r=run({...S,...s}); return r.sure===r.nf?tx('all 3 for sure','3 个都一定在'):(r.succ>0?tx(`all 3: ${fP(r.succ)}`,`全在 ${fP(r.succ)}`):tx(`${r.sure} of ${r.nf} facts`,`事实 ${r.sure} / ${r.nf}`));};
const resOf5=s=>{const r=run({...S,...s}); return r.over?tx(`full at step ${r.over}`,`第 ${r.over} 步塞满`):tx(`fits, ${fP(r.peak/r.W)} of the window at most`,`放得下，最多占窗口的 ${fP(r.peak/r.W)}`);};
function leversHTML(id,list,cur,tried,res,aria){
  return `<div class="cte-lv" id="${id}">`+A.ui.levers({aria,value:cur,options:list.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(tried.has(o.v)||o.v===cur?res(o.s||o.v):'')}))})+`</div>`;
}
const match=list=>{const o=list.find(x=>Object.keys(x.s).every(k=>S[k]===x.s[k])); return o?o.v:'';};
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('Keep everything in a 128K window: at which step is it full?','什么都不删，128K 的窗口会在第几步塞满？'),
  enter:()=>({...BASE}),gate:()=>!!pred&&runsSeen,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   const f=missionFacts();
   return tx(`<p>A coding agent fixes a date bug in a large project. It reads files, runs the tests and looks things up in the docs: 60 ${DEF('steps',TIP.step())} in all.</p><p>At every step it sends its whole ${DEF('context',TIP.context())} to the model again, and it cuts nothing. The model has a ${DEF('128K window',TIP.window())}.</p><p>The system prompt, tools and task take about ${fI(f.H)} tokens. After that, each step adds about ${fI(f.avgR)} tokens on average. Most steps add a few hundred to two thousand. But in ${f.nLong} of the ${f.n} steps the agent reads a whole file, test log or web page, which adds ${fI(f.lo)} to ${fI(f.hi)} at once.</p><p>At which step does the context no longer fit?</p>${ch}${pred?`<p class="muted small">${runsSeen?'Saved.':'Saved. You can still change it before the run in the chart ends.'}</p>`:''}`,
             `<p>一个 coding agent 在一个大项目里修日期相关的 bug。它要读文件、跑测试、查文档，一共 60 ${DEF('步',TIP.step())}。</p><p>每一步，它都把整个${DEF('上下文',TIP.context())}重新发给模型，而且什么都不删。模型的${DEF('窗口是 128K',TIP.window())}。</p><p>开头的 system prompt、工具和任务约 ${fI(f.H)} 个 token。之后每一步平均加进约 ${fI(f.avgR)} 个 token。大多数步骤只加几百到两千个。可 ${f.n} 步里有 ${f.nLong} 步要读整个文件、测试日志或网页，每次就加进 ${fI(f.lo)} 到 ${fI(f.hi)} 个。</p><p>上下文会在第几步放不下？</p>${ch}${pred?`<p class="muted small">${runsSeen?'已记下。':'已记下。图里这次运行结束之前，还可以改。'}</p>`:''}`);},
  obs:()=>obsMission()},
 {id:'fills',title:{en:'What fills the context',zh:'是什么占满了上下文'},
  lead:()=>tx('Not the conversation: the tool outputs. One file or page is thousands of tokens.','占地方的不是对话，是工具的输出：一个文件、一个网页，就是几千个 token。'),
  enter:()=>{lv2=''; tried2=new Set(); hl=''; return {...BASE};},expect:()=>tried2.size>0,showMe:()=>setLv(2,'tool'),
  body:()=>tx(`<p>The model reads its whole context at every step. So the question is what takes up the room.</p><p>Pick a kind of content and see how much of the context it takes. The chart lights it up. The share is at step ${lastFit().step}, the last call that still fit.</p>`,
              `<p>模型每一步都要把整个上下文读一遍。所以要先看清：是什么占了地方。</p><p>选一类内容，看看它占了上下文的多少。图里会把它标出来。比例按第 ${lastFit().step} 步算，这是最后一次还放得下的调用。</p>`)+leversHTML('cte-ic-lv2',LV2,lv2,tried2,v=>fP(share2(v)),tx('Pick a kind of content','选一类内容')),
  obs:()=>obsFills()},
 {id:'trim',title:{en:'Trim and slide',zh:'修剪，和滑动窗口'},
  lead:()=>tx('Keeping only the last steps is the easiest cut, but early rules and found values go with it.','只留最近几步最省事，但早先的要求和查到的数，会被一起删掉。'),
  enter:()=>{lv3='all'; tried3=new Set(['all']); hl=''; return {...P3};},expect:()=>tried3.size>1,showMe:()=>setLv(3,'slide'),
  body:()=>tx(`<p>Two common cuts. ${DEF('Trimming',TIP.trim())} keeps only the part of a long tool output that the step needs. A ${DEF('sliding window',TIP.last())} keeps only the last few steps.</p><p>Three facts in this task are needed much later. The rows under the chart show where each one is.</p><p>Pick a cut and see how many of the three facts are left.</p>`,
              `<p>常见的删法有两种。一种是${DEF('修剪',TIP.trim())}，工具的长输出只留这一步要用的部分。另一种是${DEF('滑动窗口',TIP.last())}，只留最近几步。</p><p>这个任务里有三个事实，要隔很久才用到。图下面的三行显示它们在哪里。</p><p>选一种删法，看看三个事实还剩几个。</p>`)+leversHTML('cte-ic-lv3',LV3,lv3,tried3,resOf,tx('Pick a cut','选一种删法')),
  obs:()=>obsTrim()},
 {id:'notes',title:{en:'Summaries and notes',zh:'摘要和笔记'},
  lead:()=>tx('Compress old steps into a summary; write the facts you will need into notes and fetch them later.','把旧的步骤压成摘要；把以后要用的事实写进笔记，用的时候再取回来。'),
  enter:()=>{lv4='slide'; tried4=new Set(['slide']); hl=''; return {...P4};},expect:()=>tried4.size>1,showMe:()=>setLv(4,'notes'),
  body:()=>tx(`<p>The sliding window lost all three facts. There are two ways to keep them. A ${DEF('summary',TIP.summary())} replaces the old steps with a short text the model writes. ${DEF('Notes',TIP.notes())} store each fact outside the context and bring it back when it is needed.</p><p>Real agents usually summarize when the context is nearly full. To compare runs, this lab summarizes every 20 steps.</p><p>Pick one and see whether all three facts are kept.</p>`,
              `<p>滑动窗口把三个事实都删掉了。想留住它们，有两种办法。一种是写${DEF('摘要',TIP.summary())}，让模型把旧的步骤写成一段短文字，替换掉原文。另一种是记${DEF('笔记',TIP.notes())}，把事实存到上下文之外，用的时候再取回来。</p><p>真实的 agent 多半是上下文快满时才压缩。为了好比较，这里固定每 20 步压缩一次。</p><p>选一种办法，看看三个事实能不能都留住。</p>`)+leversHTML('cte-ic-lv4',LV4,lv4,tried4,resOf4,tx('Pick a way to keep the facts','选一种留住事实的办法')),
  obs:()=>obsNotes()},
 {id:'sub',title:{en:'Hand off to a sub-agent',zh:'交给子 agent'},
  lead:()=>tx('A self-contained sub-task runs in its own context; only its result comes back.','一件独立的子任务交给另一个 agent，在它自己的上下文里做完，只把结果带回来。'),
  enter:()=>{lv5='all'; tried5=new Set(['all']); hl=''; return {...P5};},expect:()=>tried5.size>1,showMe:()=>setLv(5,'sub'),
  body:()=>tx(`<p>Steps 21 to 32 are a sub-task on their own: read five pages of the date library’s docs and find out how it handles time zones. They fill the context with pages the rest of the task never looks at again.</p><p>A ${DEF('sub-agent',TIP.sub())} can do them instead. Pick one and see how much of the window the context takes at most.</p>`,
              `<p>第 21 到 32 步本身是一件独立的子任务：读日期库文档的五个网页，弄清它怎么处理时区。这些网页塞满了上下文，任务后面却再也不看它们。</p><p>可以把这件事交给${DEF('子 agent',TIP.sub())} 去做。选一种做法，看看上下文最多占窗口的多少。</p>`)+leversHTML('cte-ic-lv5',LV5,lv5,tried5,resOf5,tx('Pick who reads the docs','选一种读文档的做法')),
  obs:()=>obsSub()},
 {id:'prod',title:{en:'How Codex and Claude Code do it',zh:'Codex 和 Claude Code 怎么做'},
  lead:()=>tx('Keep the start fixed, compact only when nearly full, and put the essentials back.','开头不变，快满了才压缩，压缩后把要紧的放回来。'),
  enter:()=>{remView=''; hl=''; return {...BASE,hist:'sum',M:20,trim:1,notes:1};},
  body:()=>prodHTML()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Manage the context like a budget: keep the rules, trim outputs, note facts.','像管预算一样管上下文：规则留着，输出修剪，事实记笔记。'),
  enter:()=>{remView=''; hl=''; return {...BASE,hist:'sum',M:20,trim:1,notes:1};},
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
  const r=R_(), lv=liveOf(r,PL.pv);
  if(!runsSeen) return tx(`<p class="cte-runl">The agent is running: step ${lv.lastStep} of 60.</p>`,`<p class="cte-runl">agent 正在跑：60 步里走到了第 ${lv.lastStep} 步。</p>`);
  const c=r.calls[r.calls.length-1], f=missionFacts(), pg=f.pages;
  const fx=M(`\\frac{\\text{${fI(r.W)}} - \\text{${fI(f.H)}}}{\\text{${fI(f.avgR)}}} \\approx ${f.est}`);
  return tx(`<p><b>At step ${r.over} the context would be ${fI(c.ctx)} tokens, more than the ${fI(r.W)} of the window. The call fails and the task stops there, with ${60-r.over+1} steps not done.</b></p>`,
            `<p><b>到第 ${r.over} 步，上下文要有 ${fI(c.ctx)} 个 token，超过了窗口的 ${fI(r.W)} 个。这次调用失败，任务停在这里，还有 ${60-r.over+1} 步没做。</b></p>`)+verdict()+
    tx(`<p>The numbers in the card give a close estimate. Take the window, subtract the head, and divide by what a step adds on average. About ${f.est} steps fit:</p>`,
       `<p>用卡片里的数就能估出来。窗口减去开头，再除以每一步平均加进的 token，大约够放 ${f.est} 步：</p>`)+fx+
    tx(`<p>The run is full even earlier, at step ${r.over}. The ${pg.length} web pages are all read in steps ${pg[0]} to ${pg[pg.length-1]}, so the context grows faster than average in the first half. A few long outputs fill a window that sounds huge.</p>`,
       `<p>实际在第 ${r.over} 步就放不下了，比估的还早。${pg.length} 个网页都是在第 ${pg[0]} 到 ${pg[pg.length-1]} 步读进来的，所以前半程涨得比平均快。几段长输出，就把听上去很大的窗口塞满了。</p>`);
}
function obsFills(){
  if(!lv2) return tx('<p>Pick a kind to see its share.</p>','<p>选一类，看看它占多少。</p>');
  const c=lastFit(), o=LV2.find(x=>x.v===lv2), sh=fP(c.seg[o.i]/c.ctx);
  const head=tx(`<p><b>${esc(L(o.t))}: ${sh} of the ${fI(c.ctx)} tokens sent at step ${c.step}.</b></p>`,`<p><b>${esc(L(o.t))}：第 ${c.step} 步发出的 ${fI(c.ctx)} 个 token 里，占 ${sh}。</b></p>`);
  if(lv2==='tool') return head+tx('<p>Why: one file dump or one web page is thousands of tokens, while a model turn is a few hundred. A handful of long outputs outweighs everything else, so they are the first thing to cut.</p>','<p>原因：读一个文件、打开一个网页，一次就是几千个 token；模型的一个回合只有几百个。几段长输出就压过了其余所有内容，所以要删，先从它们下手。</p>');
  if(lv2==='conv') return head+tx('<p>The model’s own turns are short. Cutting them would save little. Now try tool outputs.</p>','<p>模型自己的回合很短，删它们省不了多少。再看看工具的输出。</p>');
  return head+tx('<p>This part never grows: it is the same at every step. Now try tool outputs.</p>','<p>这一块从不变大，每一步都一样。再看看工具的输出。</p>');
}
function factLine(r,ids){
  const T=TASKS[r.task]; return T.facts.filter(f=>!ids||ids.includes(f.id)).map(f=>L(FACT[r.task][f.id].s)).join(tx(', ','、'));
}
function obsTrim(){
  const r=R_(), cur=match(LV3), b=run({...P3});
  if(!cur) return tx(`<p><b>${esc(polName(S))}: ${r.over?`full at step ${r.over}`:`${r.sure} of ${r.nf} facts still there when needed`}.</b> This matches none of the options.</p>`,`<p><b>${esc(polName(S))}：${r.over?`第 ${r.over} 步塞满`:`用到时还在的事实 ${r.sure} / ${r.nf}`}。</b>这组设置不是上面的任何一个选项。</p>`);
  if(cur==='all') return tx(`<p><b>Initial setup: the context is full at step ${b.over}.</b> Pick a cut.</p>`,`<p><b>初始设置：上下文在第 ${b.over} 步塞满。</b>选一种删法试试。</p>`);
  if(cur==='trim') return tx(`<p><b>Trimmed outputs: all 60 steps fit, with ${fI(r.peak)} tokens at most, ${fP(r.peak/r.W)} of the window. All ${r.nf} facts are there when needed.</b></p><p>Why: the long outputs were most of the context, and each now keeps only its useful part. Nothing else was cut, so the facts stay. But the context still grows at every step.${tried3.has('slide')?'':' Now try the sliding window.'}</p>`,
    `<p><b>修剪长输出：60 步都放得下，上下文最多 ${fI(r.peak)} 个 token，是窗口的 ${fP(r.peak/r.W)}。三个事实用到时都在。</b></p><p>原因：长输出原本占了上下文的大头，现在每段只留有用的部分。别的什么都没删，所以事实都还在。不过，上下文仍然每一步都在变长。${tried3.has('slide')?'':'再试试滑动窗口。'}</p>`);
  const t=run({...P3,trim:1}), half=r.read<t.read/2, q=r.cost/t.cost;
  return tx(`<p><b>Trim and keep the last 10 steps: the context never passes ${fI(r.peak)} tokens, but none of the ${r.nf} facts is there when it is needed.</b></p><p>Why: the user’s rule from step 3, the port found at step 12 and the docs read at step 27 are no longer among the last 10 steps when they are needed, so they were dropped. ${r.nre} old files were needed again later, so the agent had to re-read them.</p><p>It does not save money either. The tokens read ${half?'fall to less than half of trimming alone':'fall below trimming alone'}, yet with ${DEF('prompt caching',TIP.cache())} it costs ${xAbout(q)} as much as trimming alone. Dropping the oldest step at every step changes the beginning of what is sent, so the provider’s cached beginning cannot be reused.</p>`,
    `<p><b>修剪，再只留最近 10 步：上下文一直不超过 ${fI(r.peak)} 个 token，可三个事实到用的时候一个都不在了。</b></p><p>原因：第 3 步用户的要求、第 12 步查到的端口、第 27 步读到的文档，等到要用时都已经不在最近 10 步里，早被删掉了。另外有 ${r.nre} 个旧文件后来又要用，agent 只好重新读一遍。</p><p>钱也没省下来。读进去的 token ${half?'不到只修剪时的一半':'比只修剪时少'}，可有 ${DEF('prompt caching',TIP.cache())} 时，花的钱反而是只修剪时的${xAbout(q)}。每一步都删掉最早的一步，发出的开头就变了，服务商缓存下来的开头用不上。</p>`);
}
function obsNotes(){
  const r=R_(), cur=match(LV4), b=run({...P4});
  if(!cur) return tx(`<p><b>${esc(polName(S))}: ${r.sure} of ${r.nf} facts surely there when needed.</b> This matches none of the options.</p>`,`<p><b>${esc(polName(S))}：用到时一定在的事实 ${r.sure} / ${r.nf}。</b>这组设置不是上面的任何一个选项。</p>`);
  if(cur==='slide') return tx(`<p><b>Initial setup: trim and keep the last 10 steps. None of the 3 facts is there when needed.</b> Pick a way to keep them.</p>`,`<p><b>初始设置：修剪，只留最近 10 步。三个事实用到时一个都不在。</b>选一种办法把它们留住。</p>`);
  const ps=TASKS.code.facts.map(f=>fP(r.avail[f.id])), av=ps.slice(0,-1).join(tx(', ','、'))+tx(' and ',' 和 ')+ps[ps.length-1];
  if(cur==='sum') return tx(`<p><b>With a summary every 20 steps, the chance that each fact is still there when needed is ${av}. The chance that all three are there is only ${fP(r.succ)}.</b></p><p>Why: here each summary keeps a fact with 80%. The rule from step 3 went through two summaries, so only 64% is left. There is no accepted figure for this chance; 80% is chosen so that the loss after two or three summaries shows. A summary keeps the gist, but no single fact is safe in it.</p><p>The cost does fall a lot, to ${xPart(r.cost/b.cost)} of the sliding window’s. Between two summaries every call starts the same way, so prompt caching works.${tried4.has('notes')?'':' Now add notes.'}</p>`,
    `<p><b>每 20 步压缩一次时，三个事实到用的时候还在的把握分别是 ${av}。三个都在的把握只有 ${fP(r.succ)}。</b></p><p>原因：这里假设每压一次，一个事实留下来的把握是 80%。第 3 步的要求经过了两次压缩，只剩 64%。这个把握没有公认的数，取 80% 是为了让两三次压缩后的损失看得出来。摘要留得住大意，却保不住每一个具体的事实。</p><p>花的钱倒是少多了，降到滑动窗口时的${xPart(r.cost/b.cost)}。两次压缩之间，每次调用的开头都一样，prompt caching 用得上。${tried4.has('notes')?'':'再加上笔记试试。'}</p>`);
  const sm=run({...P4,hist:'sum',M:20}), same=fD(sm.cost)===fD(r.cost);
  return tx(`<p><b>Add notes: all 3 facts are there for sure, and the cost ${same?'barely moves':'rises only a little'}.</b></p><p>Why: each fact went into the notes file the moment the agent saw it, and came back when it was needed. Only the few tokens of the write and the read enter the context. This works because the agent knew at once that these facts mattered.</p>`,
    `<p><b>再记笔记：三个事实都一定在，花的钱${same?'几乎没变':'只多了一点'}。</b></p><p>原因：agent 一看到这些事实，就写进了笔记文件，用的时候再读回来。进上下文的只有写和读的那几十个 token。这能行，是因为 agent 当时就知道它们要紧。</p>`);
}
function obsSub(){
  const r=R_(), cur=match(LV5), b=run({...P5});
  if(!cur) return tx(`<p><b>${esc(polName(S))}.</b> This matches none of the options.</p>`,`<p><b>${esc(polName(S))}。</b>这组设置不是上面的任何一个选项。</p>`);
  if(cur==='all') return tx(`<p><b>Initial setup: the agent reads the docs itself, and the context is full at step ${b.over}.</b> Pick the sub-agent.</p>`,`<p><b>初始设置：文档由 agent 自己读，上下文在第 ${b.over} 步塞满。</b>试试子 agent。</p>`);
  const p4=run(REMS.sum.s), p45=run({...REMS.sum.s,sub:1});
  return tx(`<p><b>With a sub-agent, all 60 steps fit: the main context peaks at ${fI(r.peak)} tokens, ${fP(r.peak/r.W)} of the window.</b></p><p>Why: the five pages stay in the sub-agent’s context, which is thrown away when it is done; the main agent gets an 800-token result, with the UTC fact in it. The sub-agent still pays to read its pages; they just take no room in the main agent’s context.</p><p class="small">When the agent also trims, summarizes every 20 steps and takes notes, adding a sub-agent ${p45.cost<p4.cost?'also lowers the cost a little':'does not lower the cost'}.</p>`,
    `<p><b>交给子 agent 之后，60 步都放得下：主 agent 的上下文最多 ${fI(r.peak)} 个 token，是窗口的 ${fP(r.peak/r.W)}。</b></p><p>原因：五个网页都留在子 agent 的上下文里，它做完就整个丢掉；主 agent 只拿回一段 800 个 token 的结果，UTC 那个事实就在里面。子 agent 读这些网页一样要花 token，只是不占主 agent 的上下文。</p><p class="small">如果 agent 同时修剪、每 20 步压缩、再记笔记，再加上子 agent，花的钱${p45.cost<p4.cost?'也少了一点':'并没有变少'}。</p>`);
}
/* part 6: each line with this lab's example; a button puts it back in the chart (R6.23) */
const REMS={trim:{s:{...BASE,trim:1}},slide:{s:{...BASE,hist:'last',N:10,trim:1}},sum:{s:{...BASE,hist:'sum',M:20,trim:1,notes:1}},
  comp:{s:{...BASE,hist:'sum',M:20,trim:1}},subx:{s:{...BASE,hist:'sum',M:20,trim:1,notes:1,sub:1}}};
function remBtn(k){const on=remView===k;
  const tip={trim:tx('Shows the agent with trimmed tool outputs in the chart.','在图里显示修剪了工具输出的 agent。'),slide:tx('Shows the agent that trims and keeps only the last 10 steps.','在图里显示修剪、只留最近 10 步的 agent。'),sum:tx('Shows the agent that trims, summarizes every 20 steps and takes notes.','在图里显示修剪、每 20 步压缩、再记笔记的 agent。'),
    comp:tx('Shows the agent that trims and summarizes every 20 steps.','在图里显示修剪、每 20 步压缩一次的 agent。'),
    subx:tx('Shows the agent that trims, summarizes every 20 steps, takes notes and hands the docs to a sub-agent.','在图里显示修剪、每 20 步压缩、记笔记，再把查文档交给子 agent 的 agent。')}[k];
  return ` <button type="button" class="linkbtn cte-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
/* owner point 4 (R6.31): how Codex and Claude Code manage context, only what their public docs and code say (sources/b8src/context-mgmt,
   claims CC1–CC11 and CX1–CX9), each point tied to a lever the reader already used. Polish round 3 (2026-10-16): the per-claim source
   links moved to the page-end sources (FIDPOP, “Sources”), so the card keeps only the five conclusions */
function prodHTML(){
  return tx(`<p>Codex and Claude Code must fit very long tasks in the window and also keep hitting the cache; their public docs and code (sources at the end of the page) use the same ideas as this lab.</p><ul class="cte-remlist">`+
    `<li><b>Keep the start fixed, and only append.</b> Claude Code puts the system prompt and tools first; each new Codex request is the previous one, unchanged, plus the new items. A sliding window changes the beginning at every step, so it misses the cache and costs more, not less.${remBtn('slide')}</li>`+
    `<li><b>Compact only when nearly full.</b> Codex compacts by default when the context reaches 90% of the window; Claude Code first clears old tool outputs, then summarizes if it needs to. This lab summarizes every 20 steps only so that runs are easy to compare.${remBtn('comp')}</li>`+
    `<li><b>The summary call reads the cache too.</b> Claude Code’s summary request is the conversation with only the instruction to summarize appended, so its beginning is read from the cache. The Σ in the chart marks such a call.</li>`+
    `<li><b>After compacting, put the essentials back.</b> Claude Code re-injects CLAUDE.md, its memory and the plan, and re-reads up to 5 recently changed files; Codex keeps the recent user messages and a handoff summary. This is the same idea as notes.${remBtn('sum')}</li>`+
    `<li><b>Sub-agents work in their own context.</b> A Claude Code sub-agent has its own context and cache and brings back only a summary, so the main agent’s beginning stays the same.${remBtn('subx')}</li></ul>`,
    `<p>Codex 和 Claude Code 跑超长任务，既要放得下，也要多命中缓存；它们公开的文档和代码（出处在页末）里，用的正是这个 Lab 的办法。</p><ul class="cte-remlist">`+
    `<li><b>开头不变，新内容只往后加。</b>Claude Code 把 system prompt 和工具放在最前；Codex 的新请求就是上一次的请求原样加上新内容。滑动窗口每一步都改开头，用不上缓存，反而更贵。${remBtn('slide')}</li>`+
    `<li><b>快满了才压缩。</b>Codex 默认在上下文用到窗口的 90% 时压缩；Claude Code 先清掉旧的工具输出，不够再写摘要。本 Lab 每 20 步压缩一次，只是为了好比较。${remBtn('comp')}</li>`+
    `<li><b>写摘要的那次调用也读缓存。</b>Claude Code 写摘要的请求，只在对话后面加一句总结的要求，开头不变，就能从缓存读。图里标 Σ 的就是这种调用。</li>`+
    `<li><b>压缩之后，把要紧的放回来。</b>Claude Code 重新放进 CLAUDE.md、记忆和计划，再重读最近改过的最多 5 个文件；Codex 留下最近的用户消息和一段交接摘要。这和记笔记是同一个思路。${remBtn('sum')}</li>`+
    `<li><b>子 agent 用自己的上下文。</b>Claude Code 的子 agent 有自己的上下文和缓存，只交回总结，主 agent 的开头不受影响。${remBtn('subx')}</li></ul>`);
}
function remember(){
  const t=run({...BASE,trim:1}), sl=run(REMS.slide.s), sn=run(REMS.sum.s), sm=run({...BASE,hist:'sum',M:20,trim:1}), big=run({...BASE,W:262144});
  const half=sl.read<t.read/2, same=fD(sm.cost)===fD(sn.cost);
  return tx(`<ul class="cte-remlist"><li><b>Trim the long tool outputs first.</b> Tool outputs take most of the room in an agent’s context. Trimming them is the cheapest cut, and it often makes a long task fit on its own.${remBtn('trim')}</li>`+
    `<li><b>Do not change the beginning at every step.</b> With prompt caching, the part of a call that repeats the previous call’s beginning is cheap. A cut that changes the beginning at every step, such as a sliding window, misses the cache. So it can cost more even while it reads fewer tokens.${remBtn('slide')}</li>`+
    `<li><b>Write the facts you will need into notes.</b> Every summary can drop a fact, so a fact kept only in summaries grows less certain over time. A note keeps it for sure, at little extra cost.${remBtn('sum')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “With a big enough window, nothing needs cutting.” With a 256K window this agent fits. But it re-reads its long outputs at every step: even with prompt caching it costs ${xMore(big.cost/t.cost)} as much as trimming alone. And models use what sits in the middle of a long context less well.</p><p>✗ “A summary loses nothing.” Each summary is one more chance to drop a fact: ${fP(BASE.p)} after one, ${fP(BASE.p**2)} after two, ${fP(BASE.p**3)} after three.</p><p>✗ “The more you cut, the better.” Notes only hold the facts the agent knew mattered when it saw them. Some facts turn out to matter only later, such as a log line found to be the cause of an outage many steps on. Keep only the last few steps, and that line is long gone; notes cannot bring it back. The challenge has such a fact.</p></div>${A.shell.toScenarios()}`,
    `<ul class="cte-remlist"><li><b>先修剪工具的长输出。</b>agent 的上下文里，大头是工具的输出。修剪它们最省事，常常单靠这一招，长任务就放得下了。${remBtn('trim')}</li>`+
    `<li><b>别每一步都改开头。</b>有 prompt caching 时，和上一次调用开头相同的部分很便宜。每一步都改开头的删法（比如滑动窗口）用不上缓存。所以它读的 token 少了，花的钱反而可能更多。${remBtn('slide')}</li>`+
    `<li><b>以后要用的事实，写进笔记。</b>每压缩一次，都可能丢掉一个事实，所以只靠摘要，事实越往后越不保险。写进笔记就一定留得住，花的钱也只多一点。${remBtn('sum')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “窗口够大，就什么都不用删。”换成 256K 的窗口，这个 agent 是放得下了。可它每一步都要把长输出再读一遍，就算有 prompt caching，花的钱也是只修剪时的${xMore(big.cost/t.cost)}。而且上下文越长，模型越容易漏看中间的内容。</p><p>✗ “摘要不会丢信息。”每压一次，就多一次丢掉事实的机会：压一次还剩 ${fP(BASE.p)}，两次 ${fP(BASE.p**2)}，三次 ${fP(BASE.p**3)}。</p><p>✗ “删得越多越好。”笔记只记得住 agent 当时就知道要紧的事实。有的事实要到后面才看出要紧，比如一行日志，过了很多步才发现它就是故障的原因。只留最近几步的话，这行日志早就被删掉了，记笔记也救不回来。挑战题里就有这样一个事实。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios (sizes and rates are the lab's assumptions)
   ===================================================================== */
const PRESETS=[
 {id:'code',t:{en:'A coding agent reading large files',zh:'读大文件的 coding agent'},d:{en:'Trims what it reads, summarizes every 20 steps, takes notes',zh:'读到的内容先修剪，每 20 步压缩一次，记笔记'},s:{task:'code',W:131072,hist:'sum',N:10,M:20,trim:1,notes:1,sub:0}},
 {id:'research',t:{en:'A research agent reading many web pages',zh:'读很多网页的调研 agent'},d:{en:'A sub-agent compares the products; a summary every 20 steps',zh:'对比产品交给子 agent，每 20 步压缩一次'},s:{task:'research',W:131072,hist:'sum',N:10,M:20,trim:0,notes:1,sub:1}},
 {id:'chat',t:{en:'A long chat that must keep an early request',zh:'要守住开头要求的长对话'},d:{en:'The app keeps only the last 20 turns',zh:'应用只留最近 20 轮'},s:{task:'chat',W:131072,hist:'last',N:20,M:20,trim:0,notes:0,sub:0}},
 {id:'big',t:{en:'The same coding agent with a window twice as large',zh:'同一个 coding agent，窗口大一倍'},d:{en:'256K, and nothing is cut',zh:'256K，什么都不删'},s:{task:'code',W:262144,hist:'all',N:10,M:20,trim:0,notes:0,sub:0}}];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k])&&S.p===BASE.p&&S.cap===BASE.cap);
function scenHTML(i){
  const p=PRESETS[i], r=run({...BASE,...p.s});
  if(p.id==='code'){const a=run({...BASE}), bg=run({...BASE,...PRESETS[3].s}); return tx(`<p><b>This coding agent trims what it reads, summarizes every 20 steps and takes notes. The context never passes ${fI(r.peak)} tokens, and all 3 facts are there when needed.</b></p><p>Kept whole, its files and logs filled the window at step ${a.over}. Trimmed, each file keeps only the lines the agent searched for.</p><p>With prompt caching it costs ${xPart(r.cost/bg.cost)} of what the same agent costs with a window twice as large and nothing cut. Between two summaries, every call starts the same way, so the cache keeps working.</p>`,
    `<p><b>这个 coding agent 读到的内容先修剪，每 20 步压缩一次，还记笔记。上下文从不超过 ${fI(r.peak)} 个 token，三个事实用到时都在。</b></p><p>如果整份保留，它读的文件和日志在第 ${a.over} 步就塞满了窗口。修剪之后，每个文件只留它搜到的那几行。</p><p>有 prompt caching 时，它花的钱只有窗口大一倍、什么都不删时的${xPart(r.cost/bg.cost)}。两次压缩之间，每次调用的开头都一样，缓存一直用得上。</p>`);}
  if(p.id==='research'){const a=run({...BASE,task:'research'}); return tx(`<p><b>This research agent hands the product comparison to a sub-agent, summarizes every 20 steps and takes notes. All 60 steps fit, with ${fI(r.peak)} tokens of context at most, and all 3 facts are there when needed.</b></p><p>Web pages are long. Kept whole, they filled the window at step ${a.over}. The sub-agent reads the product pages in its own context, and the notes keep the facts the summary might drop.</p>`,
    `<p><b>这个调研 agent 把对比产品交给子 agent，每 20 步压缩一次，还记笔记。60 步都放得下，上下文最多 ${fI(r.peak)} 个 token，三个事实用到时都在。</b></p><p>网页都很长，全部留着的话，第 ${a.over} 步就塞满了窗口。子 agent 在自己的上下文里读产品网页。摘要可能丢掉的事实，由笔记保住。</p>`);}
  if(p.id==='chat'){const a=run({...BASE,task:'chat',hist:'all'}), x=Math.round(r.cost/a.cost); return tx(`<p><b>A chat that keeps the last 20 turns: at turn 115, the request from turn 2 is gone.</b></p><p>The chat never comes close to the window. Kept whole, it peaks at ${fI(a.peak)} tokens. The problem is the cut itself. The request is no longer among the last 20 turns, so it was dropped. Pinned in notes, or next to the system prompt, it holds to the end.</p><p>Keeping 20 turns does not save money either. Each turn drops the oldest one, so the beginning keeps changing and prompt caching cannot help. With caching it costs about ${x} times as much as keeping everything.</p>`,
    `<p><b>只留最近 20 轮的聊天：到第 115 轮，第 2 轮的要求已经不在了。</b></p><p>这段聊天离窗口还远，全部留着，最多也只有 ${fI(a.peak)} 个 token。问题出在删法本身。那条要求已经不在最近 20 轮里，被删掉了。把它记进笔记，或者放在 system prompt 旁边一直留着，就能守到最后。</p><p>只留 20 轮也不省钱。每过一轮，最早的一轮就被删掉，开头一直在变，prompt caching 用不上。有缓存时，它花的钱差不多是全部留着的 ${x} 倍。</p>`);}
  const c=run({...BASE,...PRESETS[0].s});
  return tx(`<p><b>A 256K window: all 60 steps fit, and every fact is there. But even with prompt caching, it costs ${xMore(r.cost/c.cost)} as much as the agent that trims, summarizes every 20 steps and takes notes.</b></p><p>With twice the window these 60 steps fit, but a longer task would still overflow. Each step still re-reads everything: the last call alone reads ${fI(r.peak)} tokens.</p>`,
    `<p><b>256K 的窗口：60 步都放得下，事实也都在。可就算有 prompt caching，花的钱也是修剪、每 20 步压缩再记笔记那种做法的${xMore(r.cost/c.cost)}。</b></p><p>窗口大一倍，这 60 步是放下了，可任务再长一些照样会塞满。每一步还是要把所有内容再读一遍，光最后一次调用就要读 ${fI(r.peak)} 个 token。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const chHistName=h=>{const [k,v]=CH.hists[h]; return k==='all'?tx('keep everything','什么都不删'):(k==='last'?tx(`last ${v} steps`,`最近 ${v} 步`):tx(`summary every ${v}`,`每 ${v} 步摘要`));};
const chName=o=>[chHistName(o.h),o.trim?tx('trim','修剪'):tx('no trim','不修剪'),o.notes?tx('notes','笔记'):tx('no notes','不记笔记'),o.sub?tx('sub-agent','子 agent'):tx('no sub-agent','无子 agent')].join(tx(', ','，'));
function chHint(e){
  const T=TASKS.ops, mis=T.facts.filter(f=>(e.avail[f.id]||0)<1).map(f=>f.id);
  if(e.over) return tx(`Hint: the context fills the window at step ${e.over}. The logs and dashboards are long. Cut something, or move the dashboards out of the main context.`,`提示：上下文在第 ${e.over} 步塞满了窗口。日志和监控面板都很长。删掉一些，或者把看面板这件事移出主上下文。`);
  if(mis.includes('c')&&e.trim) return tx('Hint: trimming keeps only what looked useful when the log was read. At step 48 nobody could tell the reload line mattered.','提示：修剪只留下读日志时看起来有用的部分。第 48 步的时候，谁也看不出那行配置重载要紧。');
  if(mis.includes('c')&&e.notes) return tx('Hint: notes hold only what the agent knew mattered. The reload line at step 48 did not look important then. Something has to keep the raw log until step 66.','提示：笔记只记得下 agent 当时就知道要紧的事实。第 48 步那行配置重载，当时看起来并不要紧。得有办法把原始日志留到第 66 步。');
  if(mis.includes('c')) return tx('Hint: the line from step 48 is needed at step 66, 18 steps later. Does your cut keep the raw log that long?','提示：第 48 步那一行，第 66 步才用到，隔了 18 步。你的删法能把原始日志留这么久吗？');
  if(mis.length) return tx('Hint: the user’s rule from step 2 and the time from step 15 are needed much later. What keeps a fact the agent knew mattered?','提示：第 2 步用户的要求、第 15 步查到的时间，都要隔很久才用到。什么办法能留住 agent 当时就知道要紧的事实？');
  return tx('Hint: look at which target is missed, and change the choice that drives it.','提示：先看差的是哪一个目标，再改影响它的那个选择。');
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const e=chEval(chO), best=chBest();
  const good=`<p class="okline cte-mt"><b>`+tx(`Goal met: the context never passes ${fI(e.peak)} tokens, all 3 facts are there when needed, and with prompt caching it costs about ${fD(e.cost)}.`,`达成目标：上下文从不超过 ${fI(e.peak)} 个 token，三个事实用到时都在，有 prompt caching 时约花 ${fD(e.cost)}。`)+`</b>`;
  let h;
  if(e.ok&&best&&r3(e.cost)===r3(best.cost)) h=good+tx(' And no passing combination costs less.</p>','而且没有哪个达标的组合花得更少。</p>');
  else if(e.ok) h=good+tx(' But a cheaper combination passes too.</p>','不过还有更省的组合也能达标。</p>')+tx(`<p>The cheapest is “${esc(chName(best))}”, at about ${fD(best.cost)}.</p>`,`<p>最省的组合是“${esc(chName(best))}”，约花 ${fD(best.cost)}。</p>`);
  else h=`<p class="badline cte-mt"><b>`+(e.over?tx(`Not yet: the context fills the 128K window at step ${e.over}.`,`还没达到：上下文在第 ${e.over} 步塞满了 128K 窗口。`):tx(`Not yet: only ${e.sure} of the 3 facts are surely there when needed.`,`还没达到：三个事实里，用到时一定在的只有 ${e.sure} 个。`))+`</b></p>`+
    `<p class="small">${chHint(e)}</p>`+(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the chart (the owner's review of batch 8, 2026-10-07): every combination goes in, and the brief chart keeps the ones that pass, the
     initial setup and the pick */
  const rows=chAll(), key=o=>[o.h,o.trim,o.notes,o.sub].join(), shown=rows.filter(o=>o.ok||key(o)===key(e)||key(o)===key(CH.start));
  const mx=Math.max(...shown.map(o=>o.cost));
  h+=A.ui.cmpChart({title:tx('What the on-call agent costs with prompt caching, in US dollars (green: fits and keeps all 3 facts)','运维 agent 有 prompt caching 时花多少美元（绿色：放得下，三个事实都在）'),
    aria:tx('Cost: ','花费：')+shown.map(o=>`${chName(o)}${tx(': ','：')}${fD(o.cost)}`).join(tx('; ','；')),
    rows:rows.map(o=>({label:chName(o),value:o.cost,text:o.over?tx(`${fD(o.cost)}, full at step ${o.over}`,`${fD(o.cost)}，第 ${o.over} 步塞满`):tx(`${fD(o.cost)}, facts ${o.sure}/3`,`${fD(o.cost)}，事实 ${o.sure}/3`),ok:o.ok,me:key(o)===key(e),
      tags:(key(o)===key(CH.start)?[{text:tx('initial setup','初始设置')}]:[]).concat(key(o)===key(e)?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,hiddenText:n=>tx(`The other ${n} combinations fail at least one target and are not shown.`,`其余 ${n} 种组合至少差一项，图里没有列出。`),
    max:mx*1.25,okText:tx('fits, all 3 facts','放得下，三个事实都在'),noText:tx('fails at least one','至少差一项')});
  const t20=chEval({h:3,trim:1,notes:1,sub:1}), n20=chEval({h:3,trim:0,notes:1,sub:0}), s20=chEval({h:6,trim:0,notes:1,sub:1});
  h+=tx(`<p class="small">Trimming fails every time: it cuts the reload line at step 48, which looked useless then. Notes cannot help with that line either. Only a window of at least 20 raw steps keeps it until step 66, and notes keep the two early facts.</p>`,
        `<p class="small">只要修剪就过不了。第 48 步那行配置重载当时看着没用，被剪掉了。笔记也帮不上这一行。只有至少留 20 步原文，才能把它留到第 66 步。前两个事实交给笔记。</p>`);
  h+=tx(`<p class="small">With 20 raw steps the dashboards (steps 25 to 40) make the context large. A sub-agent reads them in its own context and brings back a short result. Without it, the cost is ${xAbout(n20.cost/best.cost)} as much. Trimming would cost only ${xPart(t20.cost/best.cost)} as much, but it loses the reload line.</p>`,
        `<p class="small">留 20 步原文，监控面板（第 25 到 40 步）会让上下文很大。子 agent 在自己的上下文里看面板，只带回一段短结果。不用子 agent，花的钱是用它时的${xAbout(n20.cost/best.cost)}。修剪更便宜，只要它的${xPart(t20.cost/best.cost)}，却丢了那行配置重载。</p>`);
  h+=tx(`<p class="small">Why does the passing setup still cost this much? Keeping the last 20 steps drops the oldest step at every call, so the beginning keeps changing and prompt caching barely helps. Without trimming, a summary every 20 steps with notes and a sub-agent would cost only ${xPart(s20.cost/best.cost)} as much. But the reload line survives a summary with only ${fP(BASE.p)}, so it is not surely there.</p>`,
        `<p class="small">达标的组合为什么还这么贵？只留最近 20 步，每次调用都删掉最早的一步，开头一直在变，prompt caching 几乎用不上。不修剪，改成每 20 步压缩再记笔记，加上子 agent，花的钱只有它的${xPart(s20.cost/best.cost)}。可那行配置重载过一次压缩，只有 ${fP(BASE.p)} 的把握还在，不算一定在。</p>`);
  h+=tx(`<p class="small muted">Note: here a summary keeps each fact with 80%, a trimmed output keeps exactly what the agent knew it needed, and the cache follows Anthropic’s multipliers at $${K.usd} per million input tokens. Real summaries, trims and prices can be better or worse. Test yours on long tasks of your own before you rely on them.</p>`,
        `<p class="small muted">注意：这里假设摘要保住每个事实的把握是 80%，修剪恰好留下 agent 当时知道要用的内容，缓存按 Anthropic 的倍数、每 1M 个输入 token ${K.usd} 美元算。真实的情况可能更好，也可能更差。依赖它们之前，先在自己的长任务上测一测。</p>`);
  return h;
}

/* =====================================================================
   header copy, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length} short parts. Predict when a 60-step agent fills its window, then see what fills it, and what trimming, a sliding window, summaries, notes and a sub-agent each keep and cost. Then see how Codex and Claude Code do it. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测一个 60 步的 agent 什么时候塞满窗口，再看是什么占满了它，以及修剪、滑动窗口、摘要、笔记和子 agent 各自留下什么、花多少，再看 Codex 和 Claude Code 怎么做。第一次学从这里开始。`),
  sandbox:tx('Four real setups: a coding agent reading large files, a research agent reading web pages, a long chat, and a window twice as large. Pick one to see how its context grows and what it keeps.',
             '四个真实的场景，分别是读大文件的 coding agent、读网页的调研 agent、一段很长的聊天，以及大一倍的窗口。选一个，看它的上下文怎么涨、留下了什么。'),
  challenge:tx('An on-call agent chases an outage over 80 steps. Choose how it cuts its context so that it never fills the 128K window and three facts are there when needed, at the lowest cost with prompt caching.',
               '一个运维 agent 要花 80 步排查一次线上故障。选一种删法，让上下文从不塞满 128K 窗口，三个事实用到时都在，有 prompt caching 时的花费越少越好。')
});
const SRC={litm:'https://arxiv.org/abs/2307.03172',swe:'https://arxiv.org/abs/2405.15793',mem:'https://arxiv.org/abs/2310.08560',
  sweh:'https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/agent/history_processors.py',mini:'https://github.com/SWE-agent/mini-swe-agent',
  cache:'https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb',price:'https://platform.claude.com/docs/en/about-claude/pricing',oaic:'https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_Caching101.ipynb',
  codex:'https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs',
  /* owner point 4: Claude Code's docs and the Codex source at commit c2f7fe8 (sources/b8src/context-mgmt) */
  ccp:'https://code.claude.com/docs/en/prompt-caching',cch:'https://code.claude.com/docs/en/how-claude-code-works',ccw:'https://code.claude.com/docs/en/context-window',ccs:'https://code.claude.com/docs/en/sub-agents',
  cxp:'https://openai.com/index/unrolling-the-codex-agent-loop/',cxc:'https://github.com/openai/codex/blob/c2f7fe89d87ce853900d0b5cb1f5dc4863e44d73/codex-rs/core/src/compact.rs',
  cxm:'https://github.com/openai/codex/blob/c2f7fe89d87ce853900d0b5cb1f5dc4863e44d73/codex-rs/protocol/src/openai_models.rs',
  cxk:'https://github.com/openai/codex/blob/c2f7fe89d87ce853900d0b5cb1f5dc4863e44d73/codex-rs/core/src/client.rs'};
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> a fixed random seed generates one long task, which is then run step by step under each way of cutting. ${J('math','“Show the math”')} adds up, term by term, how many tokens each call reads and what it costs.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Every call sends the whole context: the system prompt, tools and task, a summary if there is one, and every kept step. The tokens read in total are the sum over the calls, the sub-agent’s included.</li>`+
 `<li>A call over the window fails and the run stops. A step that needs an old output that was cut re-reads it. Each needed fact is checked at the step that needs it.</li>`+
 `<li>The cost assumes prompt caching. In each call, the part that repeats the previous call’s beginning is read from the cache at 0.1 times the input price, and the rest is written to the cache at 1.25 times. A cut that changes the beginning loses the cache from that point on. The input price is assumed to be $${K.usd} per million tokens, so the cost is in US dollars.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>The sizes are the lab’s assumptions, drawn from a fixed random seed, and they differ by task. Model turns are about 150 to 600 tokens, short outputs about 200 to 1,500, logs 4,000 to 8,000, and files, web pages and dashboards 6,000 to 13,000. A summary is 2,000 tokens and keeps each fact with 80%. A trimmed output keeps at most 1,500 tokens, and a sub-agent’s result is 800.</li>`+
 `<li>A fact the agent can see is assumed to be used. Real models use what sits in the middle of a long context less well (Liu et al. 2023). This lab does not model it, so long contexts look better here than they are.</li>`+
 `<li>The cache multipliers are Anthropic’s. The input price of $${K.usd} per million tokens is this lab’s assumption. It is the base input price of several models on the ${cite(SRC.price,'Claude API pricing page')}. Prices differ a lot between models. Other providers work differently: OpenAI, for example, caches prompts over 1,024 tokens automatically and discounts the cached part. The tokens read in total ignore the cache and count every token the same.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>${cite(SRC.litm,'Lost in the Middle, arXiv 2307.03172')}: models use information in the middle of a long context less well than at its start or end.</li>`+
 `<li>Dropping old tool outputs: SWE-agent’s ${cite(SRC.sweh,'history_processors.py')} keeps the last n observations (5 in its paper) and replaces older ones with “Old environment output: (n lines omitted)”. ${cite(SRC.mini,'mini-swe-agent')} keeps a linear history instead.</li>`+
 `<li>${cite(SRC.cache,'Anthropic cookbook, prompt_caching.ipynb')} gives the cache prices: a cache write costs 1.25 times the base input price, a cache read 0.1 times.</li>`+
 `<li>${cite(SRC.oaic,'OpenAI cookbook, Prompt_Caching101.ipynb')}: caching turns on automatically for prompts longer than 1,024 tokens.</li>`+
 `<li>${cite(SRC.codex,'Codex CLI, compact.rs')}: when the context nears the limit, it summarizes the conversation and keeps the user’s own messages, up to 20,000 tokens. ${cite(SRC.cxm,'openai_models.rs')}: by default, Codex compacts when the context reaches 90% of the window.</li>`+
 `<li>Codex’s requests and compaction, at commit c2f7fe8: ${cite(SRC.cxp,'OpenAI’s post on the Codex agent loop')} and ${cite(SRC.cxk,'client.rs')} (each new request is the previous one, unchanged, plus the new items), ${cite(SRC.cxc,'compact.rs')} (the recent user messages and a handoff summary are kept after compaction).</li>`+
 `<li>Claude Code docs: ${cite(SRC.ccp,'prompt caching')} (requests ordered from least to most changing, changes appended, the summary request and sub-agents reuse or keep the cache), ${cite(SRC.cch,'how it works')} (old tool outputs cleared first, then a summary), ${cite(SRC.ccw,'context window')} (what is put back after compaction), ${cite(SRC.ccs,'sub-agents')} (a sub-agent has its own context and brings back a summary).</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>用固定的随机数种子生成一个长任务，再按每种删法一步一步跑一遍。每次调用读进去多少 token、花多少，${J('math','“展开公式”')}里逐项加给你看。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>每次调用都发出整个上下文，包括 system prompt、工具和任务，有摘要时加上摘要，再加上留着的每一步。一共读进去的 token 是所有调用加起来，子 agent 读的也算。</li>`+
 `<li>上下文超过窗口，这次调用失败，运行停下。要用的旧输出被删了，就重新读一遍。每个要用的事实，在用到它的那一步检查。</li>`+
 `<li>花费按有 prompt caching 的情况算。每次调用里，和上一次调用开头相同的部分从缓存读，按输入原价的 0.1 倍算。其余部分写进缓存，按 1.25 倍算。删法一改动开头，从改动的地方往后，缓存就用不上了。输入原价假设是每 1M 个 token ${K.usd} 美元，所以花费的单位是美元。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>各段的大小是这个 Lab 的假设，由固定的随机数生成，每个任务不同。模型的回合约 150 到 600 个 token，短输出约 200 到 1,500 个，日志 4,000 到 8,000 个，文件、网页和监控面板 6,000 到 13,000 个。摘要 2,000 个 token，每个事实压一次留下的把握是 80%。修剪后最多留 1,500 个 token，子 agent 的结果 800 个。</li>`+
 `<li>假设上下文里看得到的事实，模型都会用上。真实的模型对长上下文中间的内容用得更差（Liu 等，2023）。这个 Lab 没有建模这一点，所以长上下文在这里显得比实际好。</li>`+
 `<li>缓存的倍数用的是 Anthropic 的。输入原价每 1M 个 token ${K.usd} 美元，是这个 Lab 的假设，和 ${cite(SRC.price,'Claude API 价格页')}上几个模型的输入原价一样。不同模型的价格差别很大。别的服务商做法不同，比如 OpenAI 会自动缓存超过 1,024 个 token 的 prompt，命中缓存的部分按折扣价算。“一共读进去的 token”不管缓存，每个 token 都按同样的代价算。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>${cite(SRC.litm,'Lost in the Middle，arXiv 2307.03172')}：长上下文中间的信息，模型用得比开头和结尾差。</li>`+
 `<li>删掉旧的工具输出：SWE-agent 的 ${cite(SRC.sweh,'history_processors.py')} 只留最近 n 个工具输出（论文里是 5 个），更早的换成 “Old environment output: (n lines omitted)”。${cite(SRC.mini,'mini-swe-agent')} 则保留完整的线性历史。</li>`+
 `<li>${cite(SRC.cache,'Anthropic cookbook，prompt_caching.ipynb')} 写明了缓存的价格。写进缓存是输入原价的 1.25 倍，从缓存读是 0.1 倍。</li>`+
 `<li>${cite(SRC.oaic,'OpenAI cookbook，Prompt_Caching101.ipynb')}：超过 1,024 个 token 的 prompt 会自动缓存。</li>`+
 `<li>${cite(SRC.codex,'Codex CLI 的 compact.rs')}：上下文快到上限时，它把对话写成摘要，保留用户自己说过的话，最多 20,000 个 token。${cite(SRC.cxm,'openai_models.rs')} 里，Codex 默认在上下文用到窗口的 90% 时压缩。</li>`+
 `<li>Codex 的请求和压缩（代码取自 commit c2f7fe8）：${cite(SRC.cxp,'OpenAI 讲 Codex agent 循环的文章')}和 ${cite(SRC.cxk,'client.rs')} 讲每次的新请求是上一次的请求原样加上新内容；${cite(SRC.cxc,'compact.rs')} 讲压缩后留下最近的用户消息和一段交接摘要。</li>`+
 `<li>Claude Code 文档。${cite(SRC.ccp,'prompt caching')} 讲请求按变化从少到多排、变化往后追加，以及写摘要的请求和子 agent 怎么保住缓存。${cite(SRC.cch,'how Claude Code works')} 讲先清旧的工具输出、再写摘要。${cite(SRC.ccw,'context window')} 讲压缩后放回什么。${cite(SRC.ccs,'sub-agents')} 讲子 agent 有自己的上下文，只交回总结。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('predict when an agent’s context overflows and what it costs, and choose what to trim, summarize, note or hand off so that the facts it needs are still there.','预测 agent 的上下文什么时候塞满、要花多少钱，并决定哪些该修剪、压缩、记笔记或交给子 agent，让要用的事实到时候还在。');
const PRIMER=()=>tx(`<p>An agent is a loop: it sends the conversation to the model, runs the tool the model asks for, appends the result, and sends it all again. The model remembers nothing between calls; everything it knows is in the context of this call. The lab ${labLink('agent-anatomy')} walks through one such run.</p>`,
                   `<p>agent 就是一个循环：把对话发给模型，执行模型要的工具，把结果接在后面，再整个发一遍。模型在两次调用之间什么都不记得，它知道的一切，都在这一次的上下文里。${labLink('agent-anatomy')} 这个 Lab 会带你一步一步走完这样一次运行。</p>`);
const EXITS=()=>{
  const big=run({...BASE,W:262144}), d2=quad();
  return {explain:[
  {q:tx('With nothing cut, why do the tokens read in total grow much faster than the context?','什么都不删时，为什么一共读进去的 token 比上下文涨得快得多？'),
   o:[tx('Every call re-reads the whole context, so the total is the sum of all the contexts, and each one is longer than the last.','每次调用都把整个上下文重读一遍，总数是所有上下文加起来，而且一次比一次长。'),
      tx('Tool outputs are counted twice: once when they arrive and once when the model answers.','工具的输出被算了两次：进来时一次，模型回答时又一次。'),
      tx('The model’s replies get longer and longer as the task goes on.','任务越往后，模型的回答越写越长。')],a:0,
   n:[null,tx('Each output is counted once per call that sends it, not twice; the growth comes from sending it again at every later call.','每段输出在每次发送它的调用里算一次，不是两次；涨得快，是因为后面每次调用都要再发一遍。'),
     tx('Here the model’s turns stay 150 to 450 tokens all along; the growth comes from re-reading everything at every call.','这里模型的回合一直是 150 到 450 个 token；涨得快，是因为每次调用都要重读全部内容。')],
   why:tx('The context grows by one step each time, and every call reads all of it. Adding up contexts that grow step by step gives a total that grows with the square of the number of steps.','上下文每一步长一截，每次调用又要全部读一遍。把一次比一次长的上下文加起来，总数大约随步数的平方增长。')},
  {q:tx('With a summary every 20 steps, the user’s rule from step 3 was still there only 64% of the time. Why?','每 20 步压缩一次时，第 3 步用户提的要求只有 64% 的把握还在。为什么？'),
   o:[tx('The rule went through two summaries, and each keeps a fact with 80% (the lab’s assumption): 80% × 80% = 64%.','这条要求经过了两次压缩，这个 Lab 假设每次保住一个事实的把握是 80%：80% × 80% = 64%。'),
      tx('A 2,000-token summary has no room for three facts, so one of them is always cut.','2,000 个 token 的摘要放不下三个事实，总有一个会被删。'),
      tx('Each new summary replaces the old one, so everything from before step 20 is gone.','新的摘要会替换旧的，所以第 20 步之前的内容全没了。')],a:0,
   n:[null,tx('Room is not the problem here: each fact is kept or dropped with its own chance at each summary.','这里不是放不下的问题：每压一次，每个事实都按自己的把握留下或丢掉。'),
     tx('The new summary is written from the old summary plus the new steps, so a fact can survive several summaries, with a smaller chance each time.','新摘要是由旧摘要加上新的步骤写成的，所以一个事实可以经过好几次压缩，只是每次留下的把握都更小。')],
   why:tx('A fact in a summary is not safe: each summary is one more chance to drop it. Facts you know you will need belong in notes.','摘要里的事实并不保险：每压一次，就多一次丢掉它的机会。明知以后要用的事实，应该写进笔记。')}],
 transfer:[
  {q:tx(`With nothing cut and a window large enough, the 60-step coding agent reads ${fB(big.read)} tokens in all. The same kind of task runs twice as long, 120 steps. About how many tokens does it read?`,`什么都不删、窗口足够大时，这个 60 步的 coding agent 一共读进去 ${fB(big.read)} 个 token。同样的任务走两倍长，120 步，大约要读多少？`),
   o:[tx('About twice as many','约 2 倍'),tx('About four times as many','约 4 倍'),tx('About eight times as many','约 8 倍')],a:1,
   why:tx(`About ${fB(d2.r120)} tokens, ${d2.x.toFixed(1)} times as many. Twice the steps, and on average each call is about twice as long, so the total is about four times. That is the square growth.`,`约 ${fB(d2.r120)} 个 token，是原来的 ${d2.x.toFixed(1)} 倍。步数翻倍，平均每次调用的上下文也差不多翻倍，所以总数约是 4 倍。这就是平方增长。`)},
  {q:tx('This lab assumes each summary keeps a fact with 80%, so two summaries leave 64%. A chat app summarizes the same way, once every 20 turns. The user made a request at turn 2. About how likely is the request still there at turn 115?','这个 Lab 假设每压缩一次，一个事实留下来的把握是 80%，压两次就只剩 64%。一个聊天应用也这样压缩，每 20 轮压一次。用户在第 2 轮提了一条要求。到第 115 轮，这条要求还在的把握大约是多少？'),
   o:[tx('About 80%','约 80%'),tx('About 50%','约 50%'),tx('About 33%','约 33%'),tx('About 10%','约 10%')],a:2,
   why:tx(`About ${fP(run({...BASE,task:'chat',hist:'sum',M:20}).succ)}. Summaries happen at turns 20, 40, 60, 80 and 100, so the request goes through five of them, and each one keeps it with only 80%.`,`约 ${fP(run({...BASE,task:'chat',hist:'sum',M:20}).succ)}。第 20、40、60、80、100 轮各压一次，这条要求经过了五次压缩，每次都只有 80% 的把握留下来。`)}],
 practice:[
  {id:'agent-harness-eval',d:tx('Deep dive on trajectory recording: rebuild the exact context sent at each step from the stored turns instead of storing the whole, ever-growing context every time. Stored whole at every step, the total would grow like this lab’s tokens read, with the square of the steps.','深入讨论 trajectory 的记录：每一步发出的上下文，由存下来的各个回合拼回来，而不是每一步把整段越来越长的上下文再存一遍。如果每一步都把整段上下文存一遍，存下来的总量就和这个 Lab 里一共读进去的 token 一样，随步数的平方增长。')},
  {id:'chat-app-context',d:tx('Deep dive (b) is about assembling the context: when the history no longer fits the token budget, fold the oldest turns into a running summary instead of dropping them.','深入讨论（b）讲上下文怎么拼装：历史放不进 token 预算时，把最早的几轮并进一段持续更新的摘要，而不是直接删掉。')}]};
};
/* the 120-step version of part 1's task (the same 60 steps twice), nothing cut, no window: for the second new case */
function quad(){
  const T=TASKS.code, st=stepsOf('code'); let ctx=T.H, r60=0, r120=0;
  for(let k=1;k<=120;k++){const x=st[(k-1)%60+1]; if(k<=60) r60+=ctx; r120+=ctx; ctx+=x.m+x.o;}
  return {r60,r120,x:r120/r60};
}

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'agents',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Context engineering','上下文工程'),title:l?L(l.q):tx('What goes into an agent’s context, and what gets cut?','agent 的上下文里该放什么、删什么？'),
    minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: the agent loop','前置知识：agent 循环'),primerPop:PRIMER(),descs:DESCS(),mode});
}
const speedBtns=()=>[['s',tx('Slow','慢')],['m',tx('Medium','中')],['f',tx('Fast','快')]];
function stageHTML(){
  return `<div class="panel stage cte-stage" id="cte-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="cte-st-title"></div><div class="meter-sub" id="cte-st-sub"></div></div><div class="stage-actions"><span class="status" id="cte-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph cte-graph" id="cte-graph">`+
    `<div class="cte-cap" id="cte-cap"></div>`+
    `<div class="legend-row cte-legend" id="cte-legend"></div>`+
    `<div class="cte-chart" id="cte-chart" tabindex="0" role="img"><canvas id="cte-cv"></canvas></div>`+
    `<div class="cte-lanes-h" id="cte-lanes-h"></div>`+
    `<div class="cte-lanes" id="cte-lanes" tabindex="0" role="img"><canvas id="cte-cv2"></canvas></div>`+
    `<div class="legend-row cte-legend" id="cte-legend2"></div>`+
    `<div class="cte-player" id="cte-player">`+
     `<button type="button" class="btn sm cte-play" data-act="play" id="cte-play"></button>`+
     `<button type="button" class="btn ghost sm" data-act="stepf" data-tipx="${esc(tx('One call forward.','前进一次调用。'))}" aria-label="${esc(tx('One call forward','前进一次调用'))}">▸</button>`+
     `<button type="button" class="btn ghost sm" data-act="end" data-tipx="${esc(tx('Jump to the end of the run.','跳到这次运行的结尾。'))}" aria-label="${esc(tx('To the end','跳到结尾'))}">⏭</button>`+
     `<div class="seg-ctl cte-speed" role="group" aria-label="${esc(tx('Speed','速度'))}" data-key="speed">${speedBtns().map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${PL.speed===v}">${t}</button>`).join('')}</div>`+
     `<input type="range" class="cte-scrub" id="cte-scrub" min="0" max="60" value="0" aria-label="${esc(tx('Position in the run','播放到哪里'))}">`+
    `</div>`+
    `<div class="cte-tally mono" id="cte-tally"></div>`+
   `</div>`+
   `<p class="cte-scope small muted" tabindex="0" data-tipx="${esc(tx('The task, the sizes of its steps, the share of facts a summary keeps and the cache prices are the lab’s assumptions. Every number follows from running each way of cutting on that task.','任务、每一步的大小、摘要保住事实的把握、缓存的价格，都是这个 Lab 的假设。页面上的数字，都是把各种删法放在这个任务上跑出来的。'))}">${tx(`The task comes from a fixed random seed, and the size of each step is assumed. Every number on the page comes from running it. Costs assume $${K.usd} per million input tokens.`,`任务由固定的随机数生成，每一步的大小是假设的。页面上的数都是把这个任务跑一遍得出的。花费按假设的价格算，输入每 1M 个 token ${K.usd} 美元。`)}</p>`+
   `<div class="metrics cte-metrics" id="cte-metrics"></div>`+
   `<details class="math" id="cte-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="cte-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="cte-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="cte-lock" hidden></p>`+
   `<div class="lockable" id="cte-w-task">`+A.ui.seg({id:'cte-ctl-task',key:'task',label:tx('Task','任务'),options:[['code',tx('Coding','修 bug')],['research',tx('Research','调研')],['chat',tx('Chat','聊天')]],value:S.task,note:tx('Each task is a list of steps drawn from a fixed random seed, with an assumed size for each step.','每个任务是一串用固定随机数生成的步骤，每一步的大小是假设的。')})+`</div>`+
   `<div class="lockable" id="cte-w-W">`+A.ui.seg({id:'cte-ctl-W',key:'W',label:tx('Context window','上下文窗口'),options:[[131072,'128K'],[262144,'256K']],value:S.W,note:tx('The most tokens one call can take. 128K = 131,072 tokens.','一次调用最多能读的 token。128K = 131,072 个。')})+`</div>`+
   `<div class="lockable" id="cte-w-hist">`+A.ui.seg({id:'cte-ctl-hist',key:'hist',label:tx('Older steps','旧的步骤'),options:[['all',tx('Keep all','全留')],['last',tx('Last N','最近 N 步')],['sum',tx('Summary','摘要')]],value:S.hist})+`</div>`+
   `<div class="lockable" id="cte-w-N">`+A.ui.seg({id:'cte-ctl-N',key:'N',label:tx('N: steps kept','N：留几步'),options:[5,10,20,30].map(v=>[v,String(v)]),value:S.N})+`</div>`+
   `<div class="lockable" id="cte-w-M">`+A.ui.seg({id:'cte-ctl-M',key:'M',label:tx('M: summarize every','M：每几步压缩一次'),options:[10,20,30].map(v=>[v,String(v)]),value:S.M})+`</div>`+
   `<div class="lockable" id="cte-w-trim">`+A.ui.seg({id:'cte-ctl-trim',key:'trim',label:tx('Trim long tool outputs','修剪工具的长输出'),options:[[0,tx('No','不修剪')],[1,tx('Yes','修剪')]],value:S.trim})+`</div>`+
   `<div class="lockable" id="cte-w-notes">`+A.ui.seg({id:'cte-ctl-notes',key:'notes',label:tx('Notes file','笔记文件'),options:[[0,tx('No','不记')],[1,tx('Yes','记笔记')]],value:S.notes})+`</div>`+
   `<div class="lockable" id="cte-w-sub">`+A.ui.seg({id:'cte-ctl-sub',key:'sub',label:tx('Sub-agent for the sub-task','子任务交给子 agent'),options:[[0,tx('No','不用')],[1,tx('Yes','用')]],value:S.sub,note:tx('The coding and research tasks each have one self-contained sub-task; the chat has none.','修 bug 和调研的任务各有一件独立的子任务；聊天没有。')})+`</div>`+
   `<div class="cte-grp lockable" id="cte-w-as"><div class="cte-gh">${tx('Assumed values','假设值')}</div>`+
   `<div id="cte-w-p">`+A.ui.range({id:'cte-ctl-p',key:'p',label:tx('A summary keeps a fact','摘要保住一个事实'),min:5,max:10,value:Math.round(S.p*10),out:fP(S.p),note:tx('Assumed: 80%. A summary is written by the model in about 2,000 tokens. It keeps the gist but can leave out any one detail. The chance depends on the model, the prompt for the summary and how many facts compete for room in it. There is no accepted figure; 80% is chosen so that the loss after two or three summaries shows.','假设是 80%。摘要由模型写成，约 2,000 个 token，留得住大意，但任何一个细节都可能被漏掉。这个把握取决于模型、写摘要用的提示词，以及有多少事实在争抢摘要的篇幅。它没有公认的数，这里取 80%，是为了让两三次压缩后的损失看得出来。')})+`</div>`+
   `<div id="cte-w-cap">`+A.ui.seg({id:'cte-ctl-cap',key:'cap',label:tx('A trimmed output keeps','修剪后一段输出最多留'),options:[[500,'500'],[1500,'1,500'],[3000,'3,000']],value:S.cap,note:tx('Assumed: 1,500 tokens, about a page of search results or a function and its neighbours.','假设是 1,500 个 token，大约是一页搜索结果，或者一个函数和它前后的几行。')})+`</div></div>`+
   `<div class="lockable" id="cte-w-reset"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`def trim(text, cap=1500 * 4):                # about 4 chars a token
    return text if len(text) <= cap else text[:cap] + "\\n[cut]"

class Context:
    """What one agent sends to the model at every call."""
    def __init__(self, head, keep=10):
        self.head, self.keep = head, keep         # never cut
        self.steps, self.summary, self.notes = [], "", {}

    def add_step(self, turn, output):
        self.steps.append((turn, trim(output)))
        if len(self.steps) > self.keep:
            old = self.steps[:-self.keep]
            self.steps = self.steps[-self.keep:]
            self.summary = summarize(self.summary, old)

    def note(self, key, value):                    # kept outside
        self.notes[key] = value

    def messages(self, need=()):
        text = [self.head]
        if self.summary:
            text.append("Summary so far: " + self.summary)
        text += [f"note {k}: {self.notes[k]}" for k in need]
        for turn, output in self.steps:
            text += [turn, output]
        return text

def summarize(summary, old):
    """Stub: a real harness asks the model for this."""
    return (summary + " " + " ".join(t for t, _ in old))[-2000:]

ctx = Context("You fix bugs. Tools: read, test.", keep=2)
ctx.note("port", "5433")
for i in range(5):
    ctx.add_step(f"step {i}", "x" * 9000)
print(len(ctx.steps), len(ctx.messages(need=["port"])))   # 2 7`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="cte-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="cte-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A context manager: trim each output, keep the last steps, fold older ones into a summary, and keep notes outside.','一个上下文管理器：每段输出先修剪，只留最近几步，更早的并进摘要，笔记放在外面。')}</p><pre class="code" id="cte-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="cte-prod">`+
   tx(`<li><b>Clearing old tool outputs.</b> SWE-agent’s history processor ${cite(SRC.sweh,'LastNObservations')} keeps the last n tool outputs and replaces older ones with one line, “Old environment output: (n lines omitted)”; its docstring warns that this breaks prompt caching, because the history changes at every step.</li>`+
      `<li><b>Compaction.</b> When the context nears the window, coding agents commonly ask the model to summarize the conversation so far, replace the old conversation with the summary and carry on. OpenAI’s Codex CLI, for example, keeps the user’s own messages (up to about 20,000 tokens) and replaces the rest with the summary (${cite(SRC.codex,'compact.rs')}).</li>`+
      `<li><b>Prompt caching.</b> Providers store the beginning of the prompts they have seen. A call that starts the same way reads that part at a fraction of the price, 0.1 times in ${cite(SRC.cache,'Anthropic’s cookbook')}, and ${cite(SRC.oaic,'OpenAI')} caches prompts over 1,024 tokens automatically. So agents keep the beginning of the context fixed and add only at the end. A cut that changes the beginning at every step, such as a sliding window, loses the cache.</li>`+
      `<li><b>Notes and sub-agents.</b> Long-running agents write progress and decisions to files outside the context (a to-do list, a notes file) and read them back later; a lead agent hands a self-contained search to sub-agents that each start with a fresh context and return a short report. Both interview problems under “Go further” in the challenge work through these designs.</li>`,
      `<li><b>清掉旧的工具输出。</b>SWE-agent 的历史处理器 ${cite(SRC.sweh,'LastNObservations')} 只留最近 n 个工具输出，更早的换成一行 “Old environment output: (n lines omitted)”；它的文档注释提醒，这样会让 prompt caching 失效，因为历史每一步都在变。</li>`+
      `<li><b>压缩（compaction）。</b>coding agent 常见的做法是，上下文快到窗口上限时，让模型把之前的对话写成摘要，用摘要换掉旧的对话再接着做。比如 OpenAI 的 Codex CLI 会保留用户自己说过的话，最多约 20,000 个 token，其余换成摘要（${cite(SRC.codex,'compact.rs')}）。</li>`+
      `<li><b>Prompt caching。</b>服务商会把见过的 prompt 开头存起来。下一次调用的开头一样，这一段就按很低的价钱读，${cite(SRC.cache,'Anthropic 的 cookbook')} 里是原价的 0.1 倍。${cite(SRC.oaic,'OpenAI')} 会自动缓存超过 1,024 个 token 的 prompt。所以 agent 都尽量让上下文的开头不变，新内容只往后加。每一步都改开头的删法（比如滑动窗口）会让缓存失效。</li>`+
      `<li><b>笔记和子 agent。</b>跑得久的 agent 把进度和决定写进上下文之外的文件（待办清单、笔记文件），之后再读回来；主 agent 把一件独立的查找交给子 agent，每个子 agent 从全新的上下文开始，只交回一份简短的报告。挑战题“延伸练习”里的两道面试题，都在设计这些。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.litm,'Lost in the Middle: How Language Models Use Long Contexts')}<span>${tx('Liu et al., arXiv 2307.03172: the answer’s passage is moved to different positions among many documents; models use it less well in the middle than at the start or the end.','Liu 等，arXiv 2307.03172：把含答案的那段文字放在许多文档中的不同位置，模型对放在中间的用得比开头和结尾差。')}</span></li>`+
   `<li>${cite(SRC.swe,'SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering')}<span>${tx('Yang et al., arXiv 2405.15793: its agent keeps only the last 5 tool outputs in full, the history processor above.','Yang 等，arXiv 2405.15793：它的 agent 只完整保留最近 5 个工具输出，也就是上面那个历史处理器。')}</span></li>`+
   `<li>${cite(SRC.mem,'MemGPT: Towards LLMs as Operating Systems')}<span>${tx('Packer et al., arXiv 2310.08560: memory outside the context window that the agent writes to and reads back on demand, like this lab’s notes.','Packer 等，arXiv 2310.08560：上下文窗口之外的记忆，agent 按需写入、再读回来，就像这个 Lab 的笔记。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="cte-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="cte-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="cte-app m2" id="cte-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="cte-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how its context grows.','选一个场景，看看它的上下文怎么涨。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="cte-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const F=FACT.ops;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Keep the on-call agent inside 128K, with all 3 facts','让运维 agent 的上下文不超过 128K，三个事实都留住')}</h3>`+
   tx(`<p>An on-call agent chases an outage over 80 steps, reading many long logs and dashboards. The model’s window is 128K. These three facts must be there when they are needed:</p>`,`<p>一个运维 agent 要花 80 步排查一次线上故障，中间要读很多很长的日志和监控面板，模型的窗口是 128K。下面三个事实，用到的时候必须还在：</p>`)+
   `<ul class="cte-chfacts">${['a','b','c'].map(k=>`<li>${esc(L(F[k].l))}</li>`).join('')}</ul>`+
   tx(`<p>Goal: the context never fills the window and all three facts are surely there. Within that, the cost with prompt caching should be as low as possible. Combine the four choices:</p>`,`<p>要求：上下文从不塞满窗口，三个事实用到时一定都在。在此前提下，有 prompt caching 时的花费越少越好。四个选择可以组合：</p>`)+
   A.ui.seg({id:'cte-ic-chH',key:'chH',label:tx('Older steps','旧的步骤'),options:CH.hists.map((_,i)=>[i,chHistName(i)]),value:chO.h,inline:true})+
   A.ui.seg({id:'cte-ic-chT',key:'chT',label:tx('Trim long tool outputs','修剪工具的长输出'),options:[[0,tx('No','不修剪')],[1,tx('Yes, to 1,500 tokens','修剪到 1,500 个 token')]],value:chO.trim,inline:true})+
   A.ui.seg({id:'cte-ic-chN',key:'chN',label:tx('Notes file','笔记文件'),options:[[0,tx('No','不记')],[1,tx('Yes','记笔记')]],value:chO.notes,inline:true})+
   A.ui.seg({id:'cte-ic-chS',key:'chS',label:tx('Dashboards (steps 25 to 40)','看监控面板（第 25 到 40 步）'),options:[[0,tx('The agent itself','agent 自己看')],[1,tx('A sub-agent','交给子 agent')]],value:chO.sub,inline:true})+
   `<div class="goal mono cte-goal" id="cte-ch-g1"><span>${tx('Largest context','最大的一次上下文')}</span><span><b id="cte-ch-w"></b>${tx(' (goal: at most 131,072)','（目标：不超过 131,072）')}</span></div>`+
   `<div class="goal mono cte-goal" id="cte-ch-g2"><span>${tx('Facts surely there when needed','用到时一定在的事实')}</span><span><b id="cte-ch-f"></b>${tx(' (goal: 3 of 3)','（目标：3 / 3）')}</span></div>`+
   `<div class="goal mono cte-goal" id="cte-ch-g3"><span>${tx('Cost with prompt caching','有 prompt caching 时的花费')}</span><span><b id="cte-ch-r"></b>${tx(' (as low as possible)','（越少越好）')}</span></div>`+
   submit+`<div id="cte-ch-res" class="cte-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission') return !pred?tx('Pick an answer to continue, or Skip.','选一个答案才能继续，也可以跳过。'):tx('Wait for the run to end, or press ⏭ in the chart.','等这次运行结束才能继续，也可以点图里的 ⏭。');
  return {fills:tx('Pick a kind to continue.','选一类才能继续。'),trim:tx('Pick a cut to continue.','选一种删法才能继续。'),notes:tx('Pick a way to keep the facts to continue.','选一种留住事实的办法才能继续。'),sub:tx('Pick the sub-agent to continue.','选子 agent 才能继续。')}[id]||'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy()||tx('Try the part’s choice to continue.','先试一试这一部分的选项才能继续。'),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#cte-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function setObs(box,html){if(box._h===html) return; const again=box._h!=null&&box._h!==''&&html!==''&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
function updateRail(){
  const rail=$('#cte-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(s.id==='mission'?!!pred:true);
    const lvs={trim:[LV3,tried3,resOf],notes:[LV4,tried4,resOf4],sub:[LV5,tried5,resOf5]}[s.id];
    if(lvs){const cur=match(lvs[0]); $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever, o=lvs[0].find(x=>x.v===v); b.setAttribute('aria-pressed',String(v===cur)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvs[1].has(v)||v===cur?lvs[2](o.s):'';});}
    if(s.id==='fills') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv2)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tried2.has(v)?fP(share2(v)):'';});
    if(box){box.hidden=!show; if(show){setObs(box,`<div class="eyebrow">${tx('Result','结果')}</div>`+s.obs()); if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy()||null);
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-pred]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pred===pred)));
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#cte-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:''); ins.hidden=!html; setObs(ins,html);}
  } else {
    const e=chEval(chO), w=$('#cte-ch-w',rail), f=$('#cte-ch-f',rail), rd=$('#cte-ch-r',rail); if(!w) return;
    w.textContent=e.over?tx(`full at step ${e.over}`,`第 ${e.over} 步塞满`):fI(e.peak); f.textContent=`${e.sure} / 3`; rd.textContent=fD(e.cost);
    const g1=$('#cte-ch-g1',rail), g2=$('#cte-ch-g2',rail); g1.classList.toggle('met',e.okW); g1.classList.toggle('cte-over',!e.okW); g2.classList.toggle('met',e.okF); g2.classList.toggle('cte-over',!e.okF);
    [['chH',chO.h],['chT',chO.trim],['chN',chO.notes],['chS',chO.sub]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,rail).forEach(g=>A.ui.setPressed(g,String(v))));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#cte-ch-res',rail); setObs(res,chResultHTML());
  }
}

/* =====================================================================
   stage: the context of every call (canvas), the fact rows, the player, the cards
   ===================================================================== */
const cssv=(el,v)=>getComputedStyle(el).getPropertyValue(v).trim();
const SEGN=()=>[tx('system prompt, tools, task','system prompt、工具、任务'),tx('summary','摘要'),tx('conversation','对话'),tx('tool outputs','工具输出')];
function colors(){return {ink:cssv(app,'--ink'),ink2:cssv(app,'--ink-2'),ink3:cssv(app,'--ink-3'),line:cssv(app,'--line'),line2:cssv(app,'--line-2'),s2:cssv(app,'--surface-2'),surf:cssv(app,'--surface'),ok:cssv(app,'--c-ok'),bad:cssv(app,'--c-oom')};}
function hatch(g,x,y,w,h,col,step_){g.save(); g.beginPath(); g.rect(x,y,w,h); g.clip(); g.strokeStyle=col; g.lineWidth=1; for(let k=-h;k<w+h;k+=step_){g.beginPath(); g.moveTo(x+k,y+h); g.lineTo(x+k+h,y); g.stroke();} g.restore();}
let geo=null, ro=null;
/* the chart's height comes from the CSS (--cte-ch through its min-height): 210 px, shorter on a wide desktop stage */
const chartH=h=>{const v=parseFloat(getComputedStyle(h).minHeight); return v>=60?v:210;};
function drawChart(){
  const host=$('#cte-chart',app), cv=$('#cte-cv',app); if(!host||!cv) return;
  const Wd=host.clientWidth; if(!Wd) return;
  const H=chartH(host), dpr=window.devicePixelRatio||1, r=R_(), C=colors(), pre=hideRes()&&!pred;
  if(cv.width!==Math.round(Wd*dpr)||cv.height!==Math.round(H*dpr)){cv.width=Math.round(Wd*dpr); cv.height=Math.round(H*dpr); cv.style.width=Wd+'px'; cv.style.height=H+'px';}
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,Wd,H);
  const T_n=TASKS[S.task].n, Lp=40, Rp=8, T=14, B=18, pw=Wd-Lp-Rp, ph=H-T-B, n=Math.max(1,r.calls.length+(r.over?Math.max(0,T_n-r.over):0));   // S7: the axis ends with the run; a run that stopped keeps room for the steps it never reached
  const ymax=Math.max(r.W,...r.calls.map(c=>c.ctx))*1.06, Y=v=>T+ph-Math.min(v,ymax)/ymax*ph, cw=pw/n, bw=Math.max(1,cw*(cw>5?0.78:0.9));
  geo={Lp,cw,n,T,ph};
  g.font='10px ui-monospace,monospace'; g.fillStyle=C.ink3; g.textAlign='right'; g.textBaseline='middle';
  const tick=ymax>200000?65536:32768;
  for(let v=0;v<=ymax;v+=tick){g.strokeStyle=C.line; g.lineWidth=1; g.beginPath(); g.moveTo(Lp,Y(v)+0.5); g.lineTo(Lp+pw,Y(v)+0.5); g.stroke(); g.fillText(v?A.fmt.tok(v):'0',Lp-4,Y(v));}
  const shown=pre?0:Math.min(PL.pv,r.calls.length);
  const segCol=[C.ink,C.surf,C.ink3,C.line2];
  for(let i=0;i<shown;i++){
    const c=r.calls[i], x=Lp+i*cw+(cw-bw)/2; let y0=T+ph;
    if(c.k==='over'){const yt=Y(c.ctx); g.fillStyle=C.bad; g.globalAlpha=0.85; g.fillRect(x,yt,bw,y0-yt); g.globalAlpha=1; continue;}
    c.seg.forEach((v,si)=>{if(!v) return; const hgt=v/ymax*ph, y=y0-hgt; const dim=hl&&LV2.find(o=>o.v===hl).i!==si;
      g.globalAlpha=dim?0.18:1;
      if(si===1){g.fillStyle=C.surf; g.fillRect(x,y,bw,hgt); hatch(g,x,y,bw,hgt,C.ink2,3); g.strokeStyle=C.ink2; g.strokeRect(x+0.5,y+0.5,Math.max(0,bw-1),Math.max(0,hgt-1));}
      else {g.fillStyle=segCol[si]; g.fillRect(x,y,bw,hgt);}
      g.globalAlpha=1; y0=y;});
    if(c.k==='re'||c.k==='sum'||c.deleg){g.fillStyle=C.ink; g.textAlign='center'; g.textBaseline='bottom'; g.font='bold 10px system-ui,sans-serif'; g.fillText(c.k==='re'?'↻':(c.k==='sum'?'Σ':'⇢'),x+bw/2,Y(c.ctx)-1); g.font='10px ui-monospace,monospace';}
  }
  /* the window: a dashed line, named at its right end (R5.9) */
  g.setLineDash([5,4]); g.strokeStyle=C.bad; g.lineWidth=1.4; g.beginPath(); g.moveTo(Lp,Y(r.W)+0.5); g.lineTo(Lp+pw,Y(r.W)+0.5); g.stroke(); g.setLineDash([]);
  g.fillStyle=C.bad; g.textAlign='right'; g.textBaseline='bottom'; g.font='bold 10.5px system-ui,sans-serif'; g.fillText(tx(`window ${fW(r.W)}`,`窗口 ${fW(r.W)}`),Lp+pw-2,Y(r.W)-2);
  /* x ticks: the steps */
  g.font='10px ui-monospace,monospace'; g.fillStyle=C.ink3; g.textAlign='center'; g.textBaseline='alphabetic';
  const every=T_n>90?20:10, sr=r.calls.some(c=>c.deleg)?TASKS[S.task].sub:null; let lastX=-1e9;
  for(let s=every;s<=T_n;s+=every){
    if(sr&&s>sr[0]&&s<=sr[1]) continue;   // steps handed to the sub-agent have no bar of their own
    const idx=r.calls.findIndex(c=>c.step>=s&&c.k!=='re'), i=idx>=0?idx:r.calls.length-1+(r.over?s-r.over:0), x=Lp+i*cw+cw/2;
    if(x-lastX<24||x>Lp+pw) continue; lastX=x; g.fillText(String(s),x,H-4);}
  g.textAlign='left'; g.fillText(tx('step','步'),2,H-4);
  if(pre){g.fillStyle=C.ink3; g.textAlign='center'; g.textBaseline='middle'; g.font='12px system-ui,sans-serif'; g.fillText(tx('Answer the card’s question first.','先回答卡片里的问题。'),Lp+pw/2,T+ph/2);}
  const tip=tx(`Each bar is one model call. Its height is the number of tokens the call sends to the model, stacked by kind. The dashed line is the ${fW(r.W)} window.`,`每根柱子是一次模型调用，柱子的高度是这次调用发给模型的 token 数，按种类分段叠起来。虚线是 ${fW(r.W)} 的窗口。`)+
    (sr?tx(` Steps ${sr[0]} to ${sr[1]} went to the sub-agent and take a single bar here.`,`第 ${sr[0]} 到 ${sr[1]} 步交给了子 agent，在这里只占一根柱子。`):'');
  host.dataset.tipx=tip; host.setAttribute('aria-label',tip);
}
function drawLanes(){
  const host=$('#cte-lanes',app), cv=$('#cte-cv2',app); if(!host||!cv||!geo) return;
  const Wd=host.clientWidth; if(!Wd) return;
  const r=R_(), T=TASKS[S.task], nf=T.facts.length, rowH=18, H=nf*rowH+2, dpr=window.devicePixelRatio||1, C=colors(), pre=hideRes()&&!pred;
  if(cv.width!==Math.round(Wd*dpr)||cv.height!==Math.round(H*dpr)){cv.width=Math.round(Wd*dpr); cv.height=Math.round(H*dpr); cv.style.width=Wd+'px'; cv.style.height=H+'px';}
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,Wd,H);
  const {Lp,cw}=geo, shown=pre?0:Math.min(PL.pv,r.calls.length);
  T.facts.forEach((f,fi)=>{
    const y=fi*rowH+rowH/2+1;
    g.fillStyle=C.ink; g.font='bold 10.5px system-ui,sans-serif'; g.textAlign='right'; g.textBaseline='middle'; g.fillText(['①','②','③'][fi],Lp-6,y);
    g.strokeStyle=C.line; g.lineWidth=1; g.beginPath(); g.moveTo(Lp,y+0.5); g.lineTo(Wd-8,y+0.5); g.stroke();
    for(let i=0;i<shown;i++){const c=r.calls[i], st=c.fs[f.id], x=Lp+i*cw;
      if(c.k==='over'||st==='pre'||st==='gone') continue;
      if(st==='raw'){g.fillStyle=C.ink; g.fillRect(x,y-3,cw+0.5,6);}
      else if(st==='summ'){g.fillStyle=C.ink2; for(let k=0;k<cw;k+=3) g.fillRect(x+k,y-1.5,1.6,3);}
      else if(st==='note'){g.strokeStyle=C.ink2; g.lineWidth=1.4; g.setLineDash([4,3]); g.beginPath(); g.moveTo(x,y-0.5); g.lineTo(x+cw,y-0.5); g.stroke(); g.setLineDash([]);}
    }
    /* the step that needs it: ✓ or ✗ (or the chance) once the run gets there */
    const ni=r.calls.findIndex(c=>c.step===f.need&&c.k!=='re'), reached=ni>=0&&ni<shown, stopped=r.over&&f.need>=r.over&&shown>=r.calls.length;
    if(reached||stopped){const v=r.over&&f.need>=r.over?0:r.avail[f.id], x=Lp+(ni>=0?ni:r.calls.length-1)*cw+cw/2;
      g.font='bold 12px system-ui,sans-serif'; g.textAlign='center'; g.textBaseline='middle';
      if(v>=1){g.fillStyle=C.ok; g.fillText('✓',x,y);} else if(v<=0){g.fillStyle=C.bad; g.fillText('✗',x,y);} else {g.fillStyle=C.ink; g.font='bold 10px ui-monospace,monospace'; g.fillText(fP(v),Math.min(x,Wd-18),y);}}
  });
  host.dataset.tipx=tx('Each row is one fact from the table above. The line shows where the fact is at each call (see the legend below). At the step that needs it, ✓ means it is there and ✗ that it is gone. A fact only in a summary shows the chance that it is still there.','每一行是上面表里的一个事实。线的样子表示每次调用时它在哪里，见下面的图例。在要用它的那一步，✓ 表示它还在，✗ 表示不在了。只在摘要里的事实，写的是它还在的把握。'); host.setAttribute('aria-label',host.dataset.tipx);
  const lh=$('#cte-lanes-h',app); if(lh){const h=factsTable(S.task); if(lh._h!==h){lh.innerHTML=h; lh._h=h;}}
}
/* R2.11: the facts as an aligned list, one per line, the same fields in the same order: the fact, the step where it appears, the step
   that needs it. On the page, not in a tooltip: the kit's tooltip does not keep line breaks. Each fact's own story is on its hover. */
function factsTable(task){
  const T=TASKS[task], at=n=>T.conv?tx(`turn ${n}`,`第 ${n} 轮`):stepW(n);
  return `<table class="cte-ftab"><thead><tr><th scope="col">${tx('Fact','事实')}</th><th scope="col">${tx('Appears at','出现')}</th><th scope="col">${tx('Needed at','要用')}</th></tr></thead><tbody>`+
    T.facts.map(f=>{const x=FACT[task][f.id]; return `<tr><td>${DEF(esc(L(x.s)),L(x.l))}</td><td>${at(f.at)}</td><td>${at(f.need)}</td></tr>`;}).join('')+`</tbody></table>`;
}
function callTip(c,r){
  const T=TASKS[r.task], sg=c.seg, sn=SEGN();
  if(c.k==='over') return tx(`Step ${c.step}: the context would be ${fI(c.ctx)} tokens, more than the ${fI(r.W)} of the window. The call fails and the run stops.`,`第 ${c.step} 步：上下文要有 ${fI(c.ctx)} 个 token，超过了窗口的 ${fI(r.W)} 个。这次调用失败，运行停下。`);
  const parts=sg.map((v,i)=>v?`${sn[i]} ${fI(v)}`:'').filter(Boolean).join(tx(', ','，'));
  const reads=tx(`the model reads ${fI(c.ctx)} tokens (${parts}).`,`模型读进去 ${fI(c.ctx)} 个 token（${parts}）。`);
  const what=c.k==='re'?tx(`Step ${c.step}: an extra call to re-read the output of step ${c.ref}. In it, ${reads}`,`第 ${c.step} 步多出的一次调用，是为了重新读第 ${c.ref} 步的输出。这次${reads}`)
    :(c.k==='sum'?tx(`After step ${c.step}: the call that writes the summary. In it, ${reads}`,`第 ${c.step} 步之后的这次调用，用来写摘要。这次${reads}`)
    :(c.deleg?tx(`Step ${c.step}: this call hands steps ${T.sub[0]} to ${T.sub[1]} to the sub-agent. In it, ${reads}`,`第 ${c.step} 步这次调用，把第 ${T.sub[0]} 到 ${T.sub[1]} 步交给子 agent。这次${reads}`)
    :tx(`Step ${c.step}: ${reads}`,`第 ${c.step} 步，${reads}`)));
  const cache=c.hit?tx(` ${fI(c.hit)} of them repeat the previous call’s beginning and are read from the cache.`,`其中 ${fI(c.hit)} 个和上一次调用的开头相同，从缓存读。`)
    :tx(' Nothing repeats the previous call’s beginning, so all of it is written to the cache.','没有哪一段和上一次调用的开头相同，全部要写进缓存。');
  return what+cache;
}
function onChartMove(e){
  if(!geo||hideRes()) return; const host=$('#cte-chart',app), r=R_(), rc=host.getBoundingClientRect(), i=Math.floor((e.clientX-rc.left-geo.Lp)/geo.cw);
  if(i<0||i>=Math.min(PL.pv,r.calls.length)){return;}
  if(host._i===i) return; host._i=i; A.tip.show(host,`<div>${esc(callTip(r.calls[i],r))}</div>`);
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`, sn=SEGN();
  return it('cte-sw-h',sn[0],tx('Sent at every call and never cut: the system prompt, the tool definitions and the task.','每次调用都发、从不删除：system prompt、工具定义和任务。'))+
    it('cte-sw-s',sn[1],tx('The summary that replaced the older steps.','替换了旧步骤的那段摘要。'))+
    it('cte-sw-c',sn[2],tx('The model’s turns and the user’s messages (and the few tokens of notes written or read).','模型的回合和用户的消息（还有写笔记、读笔记的那几十个 token）。'))+
    it('cte-sw-t',sn[3],tx('What the tools returned: files, test logs, web pages.','工具返回的内容：文件、测试日志、网页。'))+
    `<span data-tipx="${esc(tx('A call over the window fails; the run stops there.','超过窗口的调用会失败，运行在那里停下。'))}" tabindex="0"><i class="sw cte-sw-o"></i><span>${esc(tx('over the window','超出窗口'))}</span></span>`+
    `<span data-tipx="${esc(tx('↻ an extra call to re-read an old output; Σ the call that writes a summary; ⇢ the call that hands a sub-task to a sub-agent.','↻ 为了重新读旧输出多出的一次调用；Σ 写摘要的那次调用；⇢ 把子任务交给子 agent 的那次调用。'))}" tabindex="0"><b>↻ Σ ⇢</b><span>${esc(tx('extra calls','额外的调用'))}</span></span>`;
}
function legend2HTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  return it('cte-sw-raw',tx('in the context','在上下文里'),tx('The fact is in the context, word for word.','这个事实原样在上下文里。'))+
    it('cte-sw-sum',tx('only in the summary','只在摘要里'),tx('The fact went into a summary; at the step that needs it, the percentage is the chance it survived.','这个事实进了摘要；用到它的那一步，写的百分比是它留下来的把握。'))+
    it('cte-sw-note',tx('in the notes file','在笔记文件里'),tx('The fact is in the notes file, outside the context; it is read back when needed.','这个事实在笔记文件里，不占上下文；用到时再读回来。'))+
    `<span data-tipx="${esc(tx('At the step that needs the fact: ✓ it is there, ✗ it is gone (or the run never got there).','在用到这个事实的那一步：✓ 它在，✗ 它不在了（或者运行没走到那一步）。'))}" tabindex="0"><b class="cte-okm">✓</b><b class="cte-badm">✗</b><span>${esc(tx('there when needed / gone','用到时在 / 不在'))}</span></span>`;
}
function statusOf(){
  if(mode==='challenge'){const e=chEval(chO); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Fits, all 3 facts','放得下，三个事实都在'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the context never passes the window and all 3 facts are surely there. Now: ${e.over?`full at step ${e.over}`:`${fI(e.peak)} tokens at most`}, ${e.sure} of 3 facts surely there.`,`上下文从不超过窗口、三个事实都一定在时，徽章是绿色。现在${e.over?`第 ${e.over} 步塞满`:`最多 ${fI(e.peak)} 个 token`}，一定在的事实有 ${e.sure} 个。`)};}
  if(hideRes()) return {cls:'neutral',txt:pred?tx('Running','在跑'):tx('Predict first','先预测'),short:pred?tx('Running','在跑'):tx('Predict first','先预测'),
    tip:pred?tx('The agent is running; the result shows when the run ends.','agent 正在跑，运行结束后显示结果。'):tx('Shown after you answer the question in part 1.','回答第 1 部分的问题之后才显示。')};
  const r=R_(), done=runDone();
  if(!done) return {cls:'neutral',txt:tx('Running','在跑'),short:tx('Running','在跑'),tip:tx('The run is playing; the verdict shows at its end.','这次运行正在播放，结束时给出结论。')};
  if(r.over) return {cls:'bad',txt:tx(`Full at step ${r.over}`,`第 ${r.over} 步塞满`),short:tx('Window full','窗口塞满'),tip:tx(`The context is over the window at step ${r.over}, so the task stops there.`,`第 ${r.over} 步上下文超过了窗口，任务在那里停下。`)};
  const ok=r.sure===r.nf, unsure=!ok&&r.succ>0;   // S3: a fact only in a summary is a chance, said with the same words as the metric
  const all=r.nf===1?tx('the fact','事实还在'):tx(`all ${r.nf} facts`,`${['','一','两','三'][r.nf]||r.nf}个事实都在`);
  return {cls:ok?'ok':'neutral',txt:ok?tx('Fits, facts kept','放得下，事实都在'):(unsure?tx(`Fits; ${all} ${fP(r.succ)}`,`放得下，${all}的把握 ${fP(r.succ)}`):tx(`Fits; ${r.sure} of ${r.nf} facts sure`,`放得下，一定在的事实 ${r.sure} / ${r.nf}`)),
    short:ok?tx('Fits, facts kept','放得下'):(unsure?tx(`All ${fP(r.succ)}`,`全在 ${fP(r.succ)}`):tx(`${r.sure}/${r.nf} facts sure`,`一定在 ${r.sure}/${r.nf}`)),
    tip:tx('Green when the run fits the window and every needed fact is surely there.','运行放得下、每个要用的事实都一定在时，徽章是绿色。')};
}
function drawHead(){
  const t=$('#cte-st-title',app), sub=$('#cte-st-sub',app), st=$('#cte-status',app);
  t.textContent=tx('How many tokens does each call send to the model?','每次调用，发给模型多少个 token？');
  sub.textContent=`${L(TASKN[S.task])}${tx('; ','；')}${tx(`${fW(S.W)} window`,`${fW(S.W)} 窗口`)}${tx('; ','；')}${polName(S)}`;
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const cap=$('#cte-cap',app), r=R_(), lv=liveOf(r,PL.pv), pre=hideRes()&&!pred;
  cap.textContent=pre?tx('Answer the card’s question first.','先回答卡片里的问题。'):(PL.pv>=r.calls.length?(r.over?tx(`The run stopped at step ${r.over}: the context did not fit.`,`运行停在第 ${r.over} 步：上下文放不下了。`):(TASKS[S.task].conv?tx(`The run is done: ${TASKS[S.task].n} turns, ${r.calls.length} calls of the model.`,`运行结束：${TASKS[S.task].n} 轮，模型调用了 ${r.calls.length} 次。`):tx(`The run is done: ${TASKS[S.task].n} steps, ${r.calls.length} calls of the main agent.`,`运行结束：${TASKS[S.task].n} 步，主 agent 调用了 ${r.calls.length} 次。`))):(TASKS[S.task].conv?tx(`Playing: turn ${lv.lastStep} of ${TASKS[S.task].n}.`,`播放中：${TASKS[S.task].n} 轮里的第 ${lv.lastStep} 轮。`):tx(`Playing: step ${lv.lastStep} of ${TASKS[S.task].n}.`,`播放中：${TASKS[S.task].n} 步里的第 ${lv.lastStep} 步。`)));
}
const numOf=v=>{const s=String(v).replace(/[,$]/g,''); const m=s.match(/^(\d+(?:\.\d+)?)(M|B)?/); if(!m) return NaN; return parseFloat(m[1])*(m[2]==='M'?1e6:(m[2]==='B'?1e9:1));};
function metricsDef(){
  const r=R_(), hid=hideRes(), dash='—', done=runDone(), lv=liveOf(r,PL.pv), w=tx('shown after the prediction','预测之后显示'), tn=!!TASKS[r.task].conv;
  const so=done?tx('whole run','整次运行'):tx(`so far, up to ${tn?'turn':'step'} ${lv.lastStep}`,`到第 ${lv.lastStep} ${tn?'轮':'步'}为止`);
  const facts=done?(r.sure===r.nf?`${r.nf} / ${r.nf}`:(r.succ>0?tx(`${fP(r.succ)} all`,`全在 ${fP(r.succ)}`):`${r.sure} / ${r.nf}`)):`${lv.checked.filter(f=>r.avail[f.id]>=1).length} / ${lv.checked.length}`;
  return [{k:'peak',label:tx('Largest context','最大的一次上下文'),v:hid?dash:(done&&r.over?tx('over','超出'):fI(lv.peak)),sub:hid?w:(done&&r.over?tx(`window full at step ${r.over}`,`第 ${r.over} 步塞满`):tx(`tokens, ${fP(lv.peak/r.W)} of the window`,`个 token，窗口的 ${fP(lv.peak/r.W)}`)),
     tip:tx('The most tokens any one call of the main agent sends to the model.','主 agent 某一次调用发给模型的 token 数，取最大的那次。')},
    {k:'facts',label:tx('Facts there when needed','用到时还在的事实'),v:hid?dash:facts,sub:hid?w:(done?(r.sure===r.nf?tx('all of them, for sure','全部都一定在'):(r.succ>0?tx('chance all are there','全部都在的把握'):tx('surely there','一定在的'))):tx('checked so far','已经检查过的')),
     tip:tx('Each fact the task needs later is checked at the step that needs it. A fact only in a summary counts with its chance of surviving.','任务后面要用的每个事实，在用到它的那一步检查。只在摘要里的事实，按它留下来的把握算。')},
    {k:'cost',label:tx('Cost with caching','有缓存时的花费'),v:hid?dash:fD(lv.cost),sub:hid?w:so,
     tip:hid?tx('Shown after the prediction.','预测之后显示。'):TIP.cost()+' '+tx(`${done?'Over the whole run':'So far'}, ${fP(lv.read?lv.hit/lv.read:0)} of the tokens read came from the cache.`,`${done?'整次运行':'到目前为止'}，读进去的 token 有 ${fP(lv.read?lv.hit/lv.read:0)} 是从缓存读的。`)},
    {k:'read',label:tx('Tokens read in total','一共读进去的 token'),v:hid?dash:fM(lv.read),sub:hid?w:so,
     tip:hid?tx('Shown after the prediction.','预测之后显示。'):tx(`Every call reads its whole context, so this is the sum over all calls ${done?'of the whole run':'so far'}${r.subRead?', the sub-agent’s included':''}. Without prompt caching, this is what you pay for.`,`每次调用都读一遍整个上下文，所以这是${done?'整次运行':'到目前为止'}所有调用加起来的数${r.subRead?'，子 agent 读的也算':''}。没有 prompt caching 时，花的钱跟着它走。`)},
    {k:'calls',label:tx('Model calls','模型调用'),v:hid?dash:fI(lv.calls),sub:hid?w:(lv.nre||lv.nsum||lv.nsub?[lv.nre?tx(`${lv.nre} re-reads`,`重新读 ${lv.nre} 次`):'',lv.nsum?tx(`${lv.nsum} summaries`,`摘要 ${lv.nsum} 次`):'',lv.nsub?tx(`${lv.nsub} by the sub-agent`,`子 agent ${lv.nsub} 次`):''].filter(Boolean).join(tx(', ','，')):(tn?tx('one per turn','每轮一次'):tx('one per step','每步一次'))),
     tip:tx(`Every ${tn?'turn':'step'} is one call; re-reading an old output, writing a summary and the sub-agent’s steps add more.`,`每${tn?'轮':'一步'}一次调用。重新读旧输出、写摘要、子 agent 的每一步，都会多出调用。`)}];
}
function drawMetrics(){
  const host=$('#cte-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  const done=runDone(), learner=userAct||wantDelta, r=R_(), pOver=shownPrev._over||0;
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    if(!done){if(d.textContent){clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';} return;}   // a new run: the old ▲▼ no longer applies
    let dt='', note=false; const prev=shownPrev[m.k];
    if(learner&&prev!=null&&prev!==m.v&&m.k!=='facts'){
      /* S5: a run that stopped early is not comparable with a whole run, so say so instead of ▲▼ */
      if(pOver||r.over){note=true; dt=pOver?tx(`last run stopped at step ${pOver}`,`上次只跑到第 ${pOver} 步`):tx(`this run stopped at step ${r.over}`,`这次只跑到第 ${r.over} 步`);}
      else {const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,{dp:m.k==='cost'?2:0})||'';}}
    if(dt){clearTimeout(d._t); clearTimeout(d._t2); d.textContent=dt; d.classList.toggle('cte-dn',note); d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},note?4200:2600);}
    else if(!learner&&prev!==m.v){clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
  if(done){shownPrev._over=r.over||0; wantDelta=false;}
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4) */
function mathHTML(){
  if(hideRes()) return `<p>${tx('Shown when the run ends.','运行结束后显示。')}</p>`;
  const r=R_(), T=TASKS[S.task], last=r.calls.filter(c=>c.k!=='over'), lc=last[last.length-1], main=last.reduce((a,c)=>a+c.ctx,0), mainCost=last.reduce((a,c)=>a+c.cost,0);
  const nm=A.lang()==='zh'?['开头','摘要','对话','工具输出']:['head','summary','conversation','tool outputs'];
  const blocks=[], readN=tx('\\text{tokens read}','\\text{读入的 token}'), fullN=tx('\\text{full-price tokens}','\\text{按原价算的 token}');
  blocks.push(MB(tx(`<b>One call.</b> A call sends the fixed head, the summary if there is one, and every kept step, added up. ${T_('H','H')} is the head and S the summary. ${Mi('m_j')} is step j’s model turn (with the user’s message and any notes), and ${Mi('o_j')} is its tool output; j runs over the kept steps.`,
                    `<b>一次调用。</b>一次调用发出的内容，是固定的开头、摘要（如果有）和留着的每一步加在一起。${T_('H','H')} 是开头，S 是摘要。${Mi('m_j')} 是第 j 步模型的回合（还有用户的消息和笔记），${Mi('o_j')} 是第 j 步工具的输出。j 取留着的每一步。`),
    EQ([[tx('\\text{context of a call}','\\text{一次调用的上下文}')+'\\,\\,C','H + S + \\sum_{j} (m_j + o_j)'],
        ].concat(SUMROWS(tx(`\\text{step ${lc.step}}`,`\\text{第 ${lc.step} 步}`),lc.seg.map((v,i)=>`${fI(v)}\\,\\text{${tx(` (${nm[i]})`,`（${nm[i]}）`)}}`),1),[['',fI(lc.ctx)]])),
    tx(`The last call, at step ${lc.step}, sends <b>${fI(lc.ctx)}</b> tokens.`,`最后一次调用在第 ${lc.step} 步，发出 <b>${fI(lc.ctx)}</b> 个 token。`)));
  blocks.push(MB(tx('<b>Tokens read in all.</b> Every call reads its whole context again.','<b>一共读进去的 token。</b>每次调用都把整个上下文再读一遍。'),
    EQ([[readN,tx('\\sum_{\\text{calls}} C','\\sum_{\\text{调用}} C')]].concat(r.subRead?[['',`${fI(main)}\\,\\text{${tx(' (main agent)','（主 agent）')}} + ${fI(r.subRead)}\\,\\text{${tx(' (sub-agent)','（子 agent）')}}`],['',fI(r.read)]]:[['',fI(r.read)]])),
    r.subRead?tx(`The main agent’s ${last.length} calls and the sub-agent’s ${r.nsub} calls read <b>${fI(r.read)}</b> tokens.`,`主 agent 调用 ${last.length} 次，子 agent 调用 ${r.nsub} 次，一共读进去 <b>${fI(r.read)}</b> 个 token。`)
      :tx(`The ${last.length} calls read <b>${fI(r.read)}</b> tokens.`,`${last.length} 次调用一共读进去 <b>${fI(r.read)}</b> 个 token。`)));
  if(S.hist==='all'){const nn=last.length, dd=Math.round((lc.ctx-T.H)/Math.max(1,nn-1)), ap=nn*T.H+dd*nn*(nn-1)/2;
    blocks.push(MB(tx(`<b>Why it grows so fast.</b> With nothing cut, each call is one step longer than the one before, so the total grows with the square of the steps. n = ${nn} is the calls and ${T_('d','d')} the tokens added per step, ${fI(dd)} on average.`,`<b>为什么涨得这么快。</b>什么都不删时，每次调用都比上一次多一步，所以总数随步数的平方增长。n = ${nn} 是调用次数，${T_('d','d')} 是平均每步加进的 token，这里是 ${fI(dd)}。`),
      EQ([[readN,'n H + d \\cdot \\frac{n (n-1)}{2}','\\approx'],['',`${nn} \\times ${fI(T.H)} + ${fI(dd)} \\times \\frac{${nn} \\times ${nn-1}}{2}`],['',fI(ap)]]),
      tx(`Close to the exact ${fI(main)}; the small gap is because the steps are not all the same size.`,`和精确值 ${fI(main)} 接近。差一点，是因为每一步的大小不一样。`)));}
  blocks.push(MB(tx(`<b>Prompt caching.</b> ${Mi('C_{\\text{hit}}')} is the part of a call that repeats the previous call’s beginning, read at ${T_('cr','0.1')} times the input price. The rest is written to the cache at ${T_('cw','1.25')} times. So each call counts as this many tokens at the full price:`,
                    `<b>Prompt caching。</b>${Mi('C_{\\text{缓存}}')} 是这次调用里和上一次调用开头相同的部分，按输入原价的 ${T_('cr','0.1')} 倍算。其余部分写进缓存，按 ${T_('cw','1.25')} 倍算。所以每次调用折合成按原价算的 token：`),
    EQ([[fullN,tx('0.1 \\cdot C_{\\text{hit}} + 1.25 \\cdot (C - C_{\\text{hit}})','0.1 \\cdot C_{\\text{缓存}} + 1.25 \\cdot (C - C_{\\text{缓存}})')],[tx('\\text{last call}','\\text{最后一次}'),`0.1 \\times ${fI(lc.hit)} + 1.25 \\times ${fI(lc.ctx-lc.hit)}`],['',fI(Math.round(lc.cost))]]),
    (r.subRead?tx(`The main agent’s calls count as ${fI(Math.round(mainCost))} full-price tokens. The sub-agent only adds to its context, so each of its calls repeats the whole previous one: ${fI(Math.round(r.subCost))}. Total <b>${fI(Math.round(r.cost))}</b>.`,`主 agent 的调用一共折合 ${fI(Math.round(mainCost))} 个按原价算的 token。子 agent 的上下文只往后加，每次调用都接着上一次的开头，一共折合 ${fI(Math.round(r.subCost))} 个。合计 <b>${fI(Math.round(r.cost))}</b> 个。`)
      :tx(`All the calls together count as <b>${fI(Math.round(r.cost))}</b> full-price tokens.`,`所有调用加起来，折合 <b>${fI(Math.round(r.cost))}</b> 个按原价算的 token。`))+tx(` ${fP(r.read?r.hit/r.read:0)} of the tokens read come from the cache.`,`读进去的 token 有 ${fP(r.read?r.hit/r.read:0)} 是从缓存读的。`)));
  blocks.push(MB(tx(`<b>Cost.</b> At the assumed input price of $${K.usd} per million tokens:`,`<b>花费。</b>按假设的价格，输入每 1M 个 token ${K.usd} 美元：`),
    EQ([[tx('\\text{cost in dollars}','\\text{花费（美元）}'),`${fI(Math.round(r.cost))} \\times \\frac{${K.usd}}{1,000,000}`],['',fD(r.cost).replace('$',''),'\\approx']])));
  if(S.hist==='sum') blocks.push(MB(tx(`<b>Facts kept.</b> Every summary keeps a fact with chance ${T_('p','p')} = ${S.p.toFixed(1)}, so a fact that went through k summaries is still there with:`,`<b>留下的事实。</b>每次压缩留住一个事实的把握是 ${T_('p','p')} = ${S.p.toFixed(1)}，所以一个事实经过 k 次压缩后，还在的把握是：`),
    EQ([[tx('\\text{chance it is kept}','\\text{还在的把握}')+'\\,\\,P','p^{k}']]),
    T.facts.map(f=>`${L(FACT[S.task][f.id].s)} ${fP(r.avail[f.id])}`).join(tx(', ','，'))+tx('.','。')));
  return blocks.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawChart(); drawLanes(); drawPlayer(); drawMetrics();
  const lg=$('#cte-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const lg2=$('#cte-legend2',app), lh2=legend2HTML(); if(lg2._h!==lh2){lg2.innerHTML=lh2; lg2._h=lh2;}
  const mb=$('#cte-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function drawPlayer(){
  const pb=$('#cte-play',app), sc=$('#cte-scrub',app), tl=$('#cte-tally',app); if(!pb) return;
  const r=R_(), end=PL.pv>=r.calls.length, pre=hideRes()&&!pred;
  pb.textContent=PL.on?tx('❚❚ Pause','❚❚ 暂停'):(end?tx('▶ Replay','▶ 重放'):tx('▶ Play','▶ 播放'));
  pb.disabled=pre; pb.dataset.tipx=PL.on?tx('Pause the run.','暂停。'):(end?tx('Play this run again from the start.','从头再播一遍这次运行。'):tx('Play the run, one model call after another.','一次调用接一次调用地播放这次运行。'));
  sc.max=String(r.calls.length); if(+sc.value!==PL.pv) sc.value=String(Math.min(PL.pv,r.calls.length)); sc.disabled=pre; A.ui.setRange(sc);
  $$('.cte-speed button',app).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===PL.speed)));
  const lv=liveOf(r,PL.pv);
  const tn=!!TASKS[S.task].conv;
  tl.textContent=pre?'':tx(`Up to ${tn?'turn':'step'} ${lv.lastStep}: ${fI(lv.calls)} model call${lv.calls===1?'':'s'}, ${fM(lv.read)} tokens read, about ${fD(lv.cost)} with caching.`,`到第 ${lv.lastStep} ${tn?'轮':'步'}，模型调用了 ${fI(lv.calls)} 次，一共读进去 ${fM(lv.read)} 个 token，有缓存时约花 ${fD(lv.cost)}。`);
}
function tick(){if(!app) return; drawHead(); drawChart(); drawLanes(); drawPlayer(); if(isP('mission')||runDone()){drawMetrics(); updateRail(); const mb=$('#cte-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}} else drawMetrics(); A.dock.refresh();}
/* Settings (R4.2): each guided part shows only its own controls; the rest are hidden with a note */
const PART_CTL={trim:['trim','hist','N','cap'],notes:['hist','M','notes','p'],sub:['sub']};
const ALL_CTL=['task','W','hist','N','M','trim','notes','sub','p','cap','reset'];
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the four choices in the task card. Everything else is fixed: the on-call task, a 128K window, a summary keeps a fact with 80%, a trimmed output keeps 1,500 tokens.','挑战中只能改任务卡里的四个选择。其余设置固定：运维任务，128K 窗口，摘要保住一个事实的把握 80%，修剪后最多留 1,500 个 token。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'||id==='fills') return tx('Fixed in this part: the coding task, a 128K window, nothing cut. You can change everything in Scenarios.','这一部分固定：修 bug 的任务，128K 窗口，什么都不删。在应用场景里每一项都能改。');
  if(id==='trim') return tx('In this part you can change how older steps are kept and the trimming. Fixed: the coding task, 128K, no notes, no sub-agent.','这一部分可以改旧步骤的留法和修剪。固定：修 bug 的任务，128K，不记笔记，不用子 agent。');
  if(id==='notes') return tx('In this part you can change the summary, the notes and what a summary keeps. Fixed: the coding task, 128K, trimmed outputs.','这一部分可以改摘要、笔记和摘要保住事实的把握。固定：修 bug 的任务，128K，输出都修剪。');
  if(id==='sub') return tx('In this part you can change the sub-agent. Fixed: the coding task, 128K, nothing else cut.','这一部分可以改子 agent。固定：修 bug 的任务，128K，别的什么都不删。');
  return tx('Fixed in this part: each “Show in the chart” button in the card sets its own example. You can change everything in Scenarios.','这一部分固定：卡片里每个“在图里看”按钮会设好自己的例子。在应用场景里每一项都能改。');
}
function syncControls(){
  $$('input[data-key="p"]',app).forEach(i=>{const v=Math.round(S.p*10); if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl'); const oo=o&&o.querySelector('output'); if(oo) oo.textContent=fP(S.p); i.setAttribute('aria-valuetext',fP(S.p));});
  [['task',S.task],['W',S.W],['hist',S.hist],['N',S.N],['M',S.M],['trim',S.trim],['notes',S.notes],['sub',S.sub],['cap',S.cap]].forEach(([k,v])=>$$(`#cte-controls .seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,String(v))));
  const g=mode==='guided', ch=mode==='challenge', id=g?STEPS[step].id:'', live=g?(PART_CTL[id]||[]):(ch?[]:ALL_CTL);
  ALL_CTL.forEach(k=>{const e=$('#cte-w-'+k,app); if(e) e.hidden=!live.includes(k)&&!(k==='reset'&&g&&live.length);});
  const subEl=$('#cte-w-sub',app); if(subEl&&!subEl.hidden&&!TASKS[S.task].sub) subEl.hidden=true;
  const ga=$('#cte-w-as',app); if(ga) ga.hidden=!live.includes('p')&&!live.includes('cap');
  const ln=$('#cte-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app||hideRes()||!runDone()) return; const r=R_(); A.announce(r.over?tx(`The context fills the window at step ${r.over}.`,`上下文在第 ${r.over} 步塞满窗口。`):statusOf().txt+tx(`. With caching it costs about ${fD(r.cost)}.`,`。有缓存时约花 ${fD(r.cost)}。`));},400);}

/* =====================================================================
   the player (R3.9: auto-play, pause, one call, to the end, three speeds, a scrubber)
   ===================================================================== */
function stopLoop(){if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=0;}
function frame(ts){
  PL.raf=0; if(!app||!PL.on) return;
  const r=R_(), dt=Math.min(0.25,(ts-(PL.last||ts))/1000); PL.last=ts; PL.acc+=dt;
  const per=SPD[PL.speed]/Math.max(1,r.calls.length); let moved=false;
  while(PL.acc>=per){PL.acc-=per; PL.pv++; moved=true; if(PL.pv>=r.calls.length){PL.pv=r.calls.length; PL.on=false; break;}}
  if(PL.pv>=r.calls.length) finished();
  if(moved||!PL.on) tick();
  if(PL.on) PL.raf=requestAnimationFrame(frame);
}
function finished(){watched.add(sigOf(R_())); if(isP('mission')&&pred&&!runsSeen){runsSeen=true; wantObs=true; renderRail(); revealSoon(); wantObs=false; return;} if(!hideRes()){drawMetrics(); updateRail();}}
function play(){if(hideRes()&&!pred) return; const r=R_(); if(PL.pv>=r.calls.length) PL.pv=0; if(A.reduceMotion()){PL.pv=r.calls.length; PL.on=false; finished(); tick(); return;} PL.on=true; PL.last=0; PL.acc=0; stopLoop(); PL.raf=requestAnimationFrame(frame); drawPlayer();}
function pause(){PL.on=false; stopLoop(); drawPlayer();}
function restart(){PL.pv=0; PL.acc=0; if(hideRes()&&!pred){pause(); tick(); return;} play();}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function clampS(){S.task=TASKS[S.task]&&S.task!=='ops'?S.task:(S.task==='ops'?'ops':'code'); S.W=S.W===262144?262144:131072; S.hist=['all','last','sum'].includes(S.hist)?S.hist:'all';
  S.N=[5,10,20,30].includes(S.N)?S.N:10; S.M=[10,20,30].includes(S.M)?S.M:20; S.trim=S.trim?1:0; S.notes=S.notes?1:0; S.sub=S.sub?1:0; S.p=Math.max(0.5,Math.min(1,Math.round(S.p*10)/10)); S.cap=[500,1500,3000].includes(S.cap)?S.cap:1500;}
function setState(p){const before=JSON.stringify(S), w=shownSig(); Object.assign(S,p); clampS(); if(JSON.stringify(S)===before){render(); return;} wantDelta=userAct; rerun(w,render);}
/* R3.20: replay only when the new run looks different from the one on screen (w: its picture before the change, '' if none was
   shown). The same run stays where it is: a finished one stays finished, and it passes part 1's gate. A different run that the
   reader has already watched to the end shows its end at once; only a run not yet watched plays (and ▶ Replay replays on demand). */
function rerun(w,draw){
  const sg=sigOf(R_()), same=!!w&&w===sg, seen=!same&&wasWatched(sg)&&!(hideRes()&&!pred);
  if(!same){PL.pv=0; PL.acc=0;}
  if(seen){PL.pv=R_().calls.length; PL.on=false; stopLoop();}
  draw();
  if(seen){finished(); tick(); return false;}
  if(!same){restart(); return true;}
  if(runDone()&&isP('mission')&&pred&&!runsSeen){runsSeen=true; renderRail();}
  return false;
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(part,v){
  if(part===2){lv2=v; tried2.add(v); hl=v; render(); return;}
  if(part===3){const o=LV3.find(x=>x.v===v); lv3=v; tried3.add(v); setState({...P3,...o.s});}
  else if(part===4){const o=LV4.find(x=>x.v===v); lv4=v; tried4.add(v); setState({...P4,...o.s});}
  else {const o=LV5.find(x=>x.v===v); lv5=v; tried5.add(v); setState({...P5,...o.s});}
}
function setRem(k){if(!REMS[k]) return; const w=shownSig(); remView=k; S={...REMS[k].s}; rerun(w,renderRail); if(innerWidth<=760) A.ui.reveal($('#cte-stage',app));}
function enterStep(i,w){
  if(w==null) w=shownSig();   // the next part never replays the run the reader just watched (R3.20)
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...BASE,...s.enter()};
  const st=$('#cte-stage',app); if(st) st.scrollTop=0;
  rerun(w,renderRail);
}
function revealStep(){A.ui.reveal($('#cte-rail .step.current',app)||$('#cte-rail',app));}
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
    const st=$('#cte-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#cte-rail .step.current',app)||$('#cte-rail .railcard',app)||$('#cte-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#cte-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#cte-insight',app)); else revealBox($('#cte-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const w=shownSig(), was=mode; mode=m; hl='';
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chO={...CH.start}; S=chState(chO);}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s,p:BASE.p,cap:BASE.cap};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#cte-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step,w); return;}
  rerun(w,renderRail);
}
const TK={code:'c',research:'r',chat:'h',ops:'o'}, KT={c:'code',r:'research',h:'chat',o:'ops'};
const tokOf=s=>[TK[s.task],s.W===262144?256:128,s.hist,s.N,s.M,s.trim,s.notes,s.sub,Math.round(s.p*10),s.cap].join('_');
const token=()=>tokOf(S);
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==10) return null;
  const task=KT[p[0]], W=p[1]==='256'?262144:(p[1]==='128'?131072:0), hist=p[2], [N,Mm,trim,notes,sub,p10,cap]=p.slice(3).map(Number);
  if(!task||task==='ops'||!W||!['all','last','sum'].includes(hist)||![5,10,20,30].includes(N)||![10,20,30].includes(Mm)||![0,1].includes(trim)||![0,1].includes(notes)||![0,1].includes(sub)||!(p10>=5&&p10<=10)||![500,1500,3000].includes(cap)) return null;
  return {...BASE,task,W,hist,N,M:Mm,trim,notes,sub,p:p10/10,cap};
}

/* =====================================================================
   interview drill (seeded): the total read when nothing is cut
   ===================================================================== */
function makeDrill(seed){const pk=(a,k)=>a[Math.floor(u01(seed+900,1,k)*a.length)];
  const H=pk([2000,3000,4000,5000,6000],0), d=pk([800,1000,1500,2000,2500,3000],1), n=pk([20,30,40,50,60],2);
  return {seed,H,d,n,total:n*H+d*n*(n-1)/2,last:H+d*(n-1)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#cte-drill-q',app), a=$('#cte-drill-a',app);
  qd.innerHTML=tx(`An agent’s system prompt, tools and task take <b>${fI(d.H)}</b> tokens. Each step adds <b>${fI(d.d)}</b> tokens to the conversation, and nothing is cut. Over <b>${d.n}</b> steps, how many tokens does the model read in total, and how long is the last call’s context?`,
                  `一个 agent 的 system prompt、工具和任务一共 <b>${fI(d.H)}</b> 个 token。每一步给对话加进 <b>${fI(d.d)}</b> 个 token，什么都不删。走 <b>${d.n}</b> 步，模型一共读进去多少个 token？最后一次调用的上下文有多长？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const eq=EQ([[tx('\\text{call } i','\\text{第 } i \\text{ 次调用}'),'H + (i - 1) \\times d'],
               [tx('\\text{tokens read}','\\text{一共读入}'),'n \\times H + d \\times \\frac{n(n - 1)}{2}'],['',`${d.n} \\times ${fI(d.H)} + ${fI(d.d)} \\times \\frac{${d.n} \\times ${d.n-1}}{2}`],['',fI(d.total)],
               [tx('\\text{last call}','\\text{最后一次}'),'H + (n - 1) \\times d'],['',`${fI(d.H)} + ${d.n-1} \\times ${fI(d.d)}`],['',fI(d.last)]]);
  a.innerHTML=tx(`<p><b>${fB(d.total)} tokens in total; the last context is ${fI(d.last)} tokens.</b></p><p>H is the head, d the tokens each step adds and n the steps.</p>${eq}<p>The total is ${(d.total/d.last).toFixed(1)} times the last context: it grows with the square of the steps.</p>`,
    `<p><b>一共 ${fB(d.total)} 个 token；最后一次的上下文是 ${fI(d.last)} 个。</b></p><p>H 是开头，d 是每一步加进的 token，n 是步数。</p>${eq}<p>总数是最后一次上下文的 ${(d.total/d.last).toFixed(1)} 倍：它随步数的平方增长。</p>`);
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const TERM={
 H:()=>tx(`H is the head that every call sends and never cuts: the system prompt, the tool definitions and the task. In this task it is assumed to be ${fI(TASKS[S.task].H)} tokens.`,`H 是每次调用都要发、从不删除的开头，包括 system prompt、工具定义和任务。这个任务里假设它有 ${fI(TASKS[S.task].H)} 个 token。`),
 cr:()=>tx('A token read from the cache costs 0.1 times the input price. This is the price in Anthropic’s cookbook, an assumption of this lab.','从缓存读一个 token，按输入原价的 0.1 倍算。这是 Anthropic cookbook 里的价格，在这个 Lab 里是假设。'),
 cw:()=>tx('A token written to the cache costs 1.25 times the input price. This is the price in Anthropic’s cookbook, an assumption of this lab.','写进缓存的 token，按输入原价的 1.25 倍算。这是 Anthropic cookbook 里的价格，在这个 Lab 里是假设。'),
 d:()=>tx('d: how many tokens one step adds to the context on average, its model turn and its tool output together.','d：平均每一步给上下文加进多少个 token，模型的回合和工具的输出加在一起。'),
 p:()=>tx('p: the chance that one summary keeps a given fact. Assumed: 80%.','p：一次压缩保住某个事实的把握。假设是 80%。')
};
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='p') return mode==='challenge'?tx('Fixed during the challenge: a summary keeps a fact with 80%.','挑战中固定不变：摘要保住一个事实的把握是 80%。'):tx('Click to jump to “A summary keeps a fact”.','点击跳到“摘要保住一个事实”。');
  if(k==='H'||k==='d') return tx('Set by the task. Pick another task under Settings.','由任务决定，在“自由调参”里可以换任务。');
  if(k==='cr'||k==='cw') return tx('Fixed in this lab.','这个 Lab 里固定不变。');
  return '';
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#cte-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'||k==='H'||k==='d'){A.tip.hide(); A.ui.jumpToEl($('#cte-controls',app)); return;}
  if(k==='cr'||k==='cw'){if(src) A.tip.show(src,`<div>${esc(TERM[k]())}</div>`); return;}
  if(k==='p'){if(mode==='challenge'){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;} const el=$('#cte-ctl-p-in',app)||$('#cte-controls',app); A.ui.jumpToEl(el); return;}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  const r=R_();
  if(a==='play'){if(PL.on) pause(); else play(); return;}
  if(a==='stepf'){A.tip.quiet(); pause(); if(hideRes()&&!pred) return; if(PL.pv>=r.calls.length) PL.pv=0; PL.pv++; if(PL.pv>=r.calls.length) finished(); tick(); return;}
  if(a==='end'){pause(); if(hideRes()&&!pred) return; PL.pv=r.calls.length; finished(); tick(); return;}
  if(a==='reset'){byLearner(()=>{const w=shownSig(); S={...BASE}; if(mode==='guided'){const s=STEPS[step]; if(s.enter) S={...BASE,...s.enter()};} else if(mode==='sandbox') S={...PRESETS[0].s,p:BASE.p,cap:BASE.cap}; rerun(w,render);}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(chO).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#cte-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#cte-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const t=e.target, k=t.dataset&&t.dataset.key;
    if(t.id==='cte-scrub'){pause(); if(hideRes()&&!pred) return; const r=R_(); PL.pv=Math.max(0,Math.min(r.calls.length,+t.value)); if(PL.pv>=r.calls.length) finished(); tick(); return;}
    if(k==='p') bySettings(t,()=>setState({p:+t.value/10}));});
  el.addEventListener('mousemove',e=>{if(e.target.closest&&e.target.closest('#cte-chart')) onChartMove(e);});
  el.addEventListener('mouseleave',e=>{const h=$('#cte-chart',app); if(h) h._i=null;},true);
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#cte-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){const first=!pred; pred=d.pred; renderRail(); if(first){if(watched.has(sigOf(R_()))){PL.pv=R_().calls.length; PL.on=false; stopLoop(); finished(); tick();} else restart();} const fb=$(`#cte-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null){const part=isP('fills')?2:(isP('trim')?3:(isP('notes')?4:(isP('sub')?5:0))); if(!part) return; const f=()=>byLearner(()=>setLv(part,d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='speed'){PL.speed=v; drawPlayer(); return;}
      if(['task','hist'].includes(k)) bySettings(b,()=>setState({[k]:v}));
      else if(['W','N','M','trim','notes','sub','cap'].includes(k)) bySettings(b,()=>setState({[k]:+v}));
      else if(['chH','chT','chN','chS'].includes(k)){const f={chH:'h',chT:'trim',chN:'notes',chS:'sub'}[k]; if(chO[f]===+v) return; chO={...chO,[f]:+v}; byLearner(()=>{const w=shownSig(); S=chState(chO); rerun(w,render);});}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...BASE,...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#cte-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#cte-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the options tried in each part, the runs watched to the
     end (hashed), the challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chO:{...chO},chSubmitted,chReveal,runsSeen,
    lv2,tried2:[...tried2],lv3,tried3:[...tried3],lv4,tried4:[...tried4],lv5,tried5:[...tried5],remView,hl,sbTouched,drillSeed,drillOpen,speed:PL.speed,
    seen:[...new Set([...WH,...[...watched].map(hkey)])].slice(-200),passed:mode==='challenge'&&chSubmitted&&chEval(chO).ok}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'&&TK[s.task]?parse(tokOf(s)):null, md=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided', p=ps(o.S);
    if(!p&&md!=='challenge') return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, vs=a=>a.map(x=>x.v), set=(v,ok,base)=>new Set([...base,...(Array.isArray(v)?v.filter(ok):[])]);
    mode=md; S=p||{...BASE}; step=ints(o.step,0,STEPS.length-1)?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null;
    const c=o.chO&&typeof o.chO==='object'?o.chO:{}, b01=v=>v===0||v===1; chO={h:ints(c.h,0,CH.hists.length-1)?c.h:CH.start.h,trim:b01(c.trim)?c.trim:0,notes:b01(c.notes)?c.notes:0,sub:b01(c.sub)?c.sub:0};
    chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal); runsSeen=bool(o.runsSeen);
    const L2=vs(LV2), L3=vs(LV3), L4=vs(LV4), L5=vs(LV5);
    lv2=L2.includes(o.lv2)?o.lv2:''; tried2=set(o.tried2,v=>L2.includes(v),lv2?[lv2]:[]); hl=L2.includes(o.hl)?o.hl:'';
    lv3=L3.includes(o.lv3)?o.lv3:'all'; tried3=set(o.tried3,v=>L3.includes(v),['all',lv3]);
    lv4=L4.includes(o.lv4)?o.lv4:'slide'; tried4=set(o.tried4,v=>L4.includes(v),['slide',lv4]);
    lv5=L5.includes(o.lv5)?o.lv5:'all'; tried5=set(o.tried5,v=>L5.includes(v),['all',lv5]);
    remView=o.remView===''||Object.keys(REMS).includes(o.remView)?o.remView:''; sbTouched=bool(o.sbTouched); drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen);
    if(['s','m','f'].includes(o.speed)) PL.speed=o.speed;
    if(Array.isArray(o.seen)) o.seen.forEach(h=>{if(typeof h==='string'&&/^[0-9a-z]{1,8}$/.test(h)) WH.add(h);});
    if(mode==='challenge') S=chState(chO);
    PL.on=false; PL.acc=0; PL.pv=!(hideRes()&&!pred)&&wasWatched(sigOf(R_()))?R_().calls.length:0;},   // a run watched to the end shows finished; mount plays a new one
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'){if(!saved) saved={...BASE}; S=chState(chO);}
    el.innerHTML=viewHTML(); app=$('#cte-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#cte-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    if(mode==='guided'&&!p&&step===0&&!pred){S={...BASE};}
    fillRail(); render();
    const h=$('#cte-chart',app); if(h) h.addEventListener('mouseleave',()=>{h._i=null; A.tip.hide();});
    /* the stage changes width without a window resize (the guided flow narrows the bench, and the chart's height follows the stage's
       width): redraw the chart and the rows when the width or the height the CSS asks for changes */
    if(ro) ro.disconnect(); ro=window.ResizeObserver?new ResizeObserver(()=>{if(!app) return; const c=$('#cte-chart',app), cv=$('#cte-cv',app); if(!c||!cv) return;
      if(Math.abs(c.clientWidth-(parseFloat(cv.style.width)||0))>1||Math.abs(chartH(c)-(parseFloat(cv.style.height)||0))>0.5){drawChart(); drawLanes();}}):null;
    if(ro) ro.observe($('#cte-graph',app));
    if(PL.on){PL.last=0; stopLoop(); PL.raf=requestAnimationFrame(frame);}
    else if(!(hideRes()&&!pred)&&PL.pv===0) play();
  },
  leave(){clearTimeout(annT); stageStop(); stopLoop(); if(ro){ro.disconnect(); ro=null;} root=null; app=null;},
  onLang(){if(root){const d=$('#cte-math',app), open=d&&d.open; stopLoop(); this.mount(root,null); if(open) $('#cte-math',app).open=true;}},
  onResize(){if(app){drawChart(); drawLanes();}},
  termTip(k){const f=TERM[k]; return f?`<div>${esc(f())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#cte-chart',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#cte-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(hideRes()){vis.innerHTML=''; read.innerHTML=esc(pred?tx('The agent is running','agent 正在跑'):tx('Answer part 1 first','先回答第 1 部分的问题')); return;}
      const r=R_(), lv=liveOf(r,PL.pv), pk=Math.min(1,lv.peak/r.W);
      vis.innerHTML=`<div class="cte-dk"><i style="width:${(pk*100).toFixed(1)}%"></i></div>`;
      const tn=!!TASKS[r.task].conv;
      read.innerHTML=esc(tx(`Up to ${tn?'turn':'step'} ${lv.lastStep}: the largest context is ${fP(lv.peak/r.W)} of the ${fW(r.W)} window, about ${fD(lv.cost)} with caching`,`到第 ${lv.lastStep} ${tn?'轮':'步'}，最大的上下文占 ${fW(r.W)} 窗口的 ${fP(lv.peak/r.W)}，有缓存时约花 ${fD(lv.cost)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__cte=Object.freeze({
  run:s=>deep(run({...BASE,...s})), steps:id=>deep(stepsOf(id)), u01, chEval:o=>deep(chEval(o)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)), quad:()=>deep(quad()),
  consts:Object.freeze({BASE:{...BASE},K:{...K},TASKS:deep(TASKS),CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,P3:{...P3},P4:{...P4},P5:{...P5},
    LV3:deep(LV3.map(o=>({v:o.v,...o.s}))),LV4:deep(LV4.map(o=>({v:o.v,...o.s}))),LV5:deep(LV5.map(o=>({v:o.v,...o.s}))),CODE}),
  state:()=>({...S,mode,step,pred,observed:{...observed},runsSeen,pv:PL.pv,playing:PL.on,speed:PL.speed,lv2,lv3,lv4,lv5,hl,chO:{...chO},chSubmitted,chReveal,remView,stageBusy:stageT!==0})
});
})();
