/* admission — the capacity capstone (batch 10): how many GPUs a day of traffic needs, and who waits when it is too much.
   A fleet of H100s, one copy of Llama-3.1-8B (BF16) per GPU, behind one gateway queue. Three kinds of requests: paid chats, free chats
   (one free API key belongs to a script), batch jobs. Policies: a queue limit for free chats (HTTP 429), a per-key rate limit (token
   bucket), paid-first priority with preemption of batch jobs (vLLM's recompute), and a busy-share threshold for batch jobs.
   Engine: the serving capstone's step model and continuous batching, copied (not imported), with a seeded Poisson load; the day is
   priced hour by hour from the measured goodput per GPU. Python twin: tests/admission_numbers.py. */
(function(){
'use strict';
/* ENGINE BEGIN */
const ADM=(function(){
const L=32, D=4096, KVH=8, HD=128, FFN=14336, VOCAB=128256;
const P_LAYER=2*D*D+2*D*KVH*HD+3*D*FFN+2*D, P_LAYERS=L*P_LAYER+D, P_HEAD=VOCAB*D, P_TOTAL=P_LAYERS+2*P_HEAD;
const KV_TOK=2*L*KVH*HD*2, W_READ=2*(P_LAYERS+P_HEAD), W_MEM=2*P_TOTAL;
const FS=1979e12/2*0.5, BS=3.35e12*0.6, GiB=2**30, BLOCK=16;
const POOL_T=Math.floor((80*GiB*0.9-W_MEM-2*GiB)/KV_TOK), POOL_B=Math.floor(POOL_T/BLOCK);
const EPS=1e-9, BUDGET=2048, CAP=1024;
const GOALS={paid:{ttft:1.0,gap:0.15}, free:{ttft:5.0,gap:0.15}}, TARGET={paid:0.9, free:0.8};
const LEN={paid:[400,1600,120,320], free:[400,1600,120,320], batch:[8000,16000,300,800]};
/* the lab's day: share of the peak chat rate in each hour (0:00 … 23:00) */
const PROF=[0.30,0.22,0.18,0.15,0.15,0.18,0.25,0.40,0.60,0.78,0.90,0.97,0.95,0.97,1.00,0.98,0.93,0.85,0.75,0.70,0.68,0.62,0.50,0.40];
const PEAK_H=14, WIN=60, NWIN=120, RL=60;
const attn=(a,b)=>2*L*D*(b*(b+1)-a*(a+1));
const blocks=n=>Math.floor((n+BLOCK-1)/BLOCK);
function mulberry(seed){let h=seed>>>0; return ()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};}
const expGap=(rnd,rate)=>-Math.log(1-rnd())/rate;
const ms=x=>Math.floor(x*1000+0.5)/1000;
const uni=(rnd,lo,hi)=>lo+Math.floor(rnd()*(hi-lo+1));
function sortLoad(a){return a.map((s,i)=>[s,i]).sort((x,y)=>x[0].t-y[0].t||x[1]-y[1]).map(x=>x[0]);}

/* phases: [[seconds, {paid, free, script, batch}], …] one after another (requests per second); backlog: batch jobs waiting at 0 s */
function load(phases,seed,backlog=0){
  const out=[];
  ['paid','free','script','batch'].forEach((cls,k)=>{
    const rnd=mulberry(seed*16+k+1), ln=LEN[cls==='script'?'free':cls]; let t0=0;
    for(const [dur,rates] of phases){const r=rates[cls]||0;
      if(r>0){let t=t0; for(;;){t+=expGap(rnd,r); if(t>=t0+dur) break;
        out.push({t:ms(t),cls:cls==='script'?'free':cls,key:cls==='script'?'script':'',P:uni(rnd,ln[0],ln[1]),out:uni(rnd,ln[2],ln[3])});}}
      t0+=dur;}});
  if(backlog){const rnd=mulberry(seed*16+9), ln=LEN.batch; for(let i=0;i<backlog;i++) out.push({t:0,cls:'batch',key:'',P:uni(rnd,ln[0],ln[1]),out:uni(rnd,ln[2],ln[3])});}
  return sortLoad(out);}

/* one engine step: the slower of its arithmetic and its memory traffic (S2). w.dec: requests that decode one token; w.pf: [r, a, b] prompt chunks */
function stepTime(w){let T=0, lg=0, fa=0, kv=0;
  for(const r of w.dec){const c=r.P+r.gen-1; T+=1; lg+=1; fa+=attn(c,c+1); kv+=c;}
  for(const [r,a,b] of w.pf){T+=b-a; lg+=b===r.Pr?1:0; fa+=attn(a,b); kv+=b;}
  return Math.max((2*P_LAYERS*T+2*P_HEAD*lg+fa)/FS,(W_READ+KV_TOK*kv)/BS);}

/* a queue with a moving head (push at the back, take from the front, a preempted job goes back to the front) */
function mkQ(){return {a:[],h:0};}
const qLen=q=>q.a.length-q.h, qPeek=q=>q.a[q.h], qShift=q=>q.a[q.h++], qPush=(q,x)=>q.a.push(x), qFront=(q,x)=>{if(q.h>0) q.a[--q.h]=x; else q.a.unshift(x);};

/* cfg: {n GPUs, qlim (free requests that may wait; null = no limit), rl (requests per minute for the script's key; 0 = off),
        prio (paid → free → batch; else first come, first served), preempt (default: on with prio), window, sample (s)} */
function simulate(spec,cfg){
  const n=cfg.n, qlim=cfg.qlim==null?null:cfg.qlim, rl=cfg.rl||0, prio=!!cfg.prio, preOn=prio&&cfg.preempt!==false, dts=cfg.sample||0.5, win=cfg.window||0;
  const R=spec.map((s,i)=>({id:i,cls:s.cls,key:s.key,arr:s.t,P:s.P,out:s.out,Pr:s.P,doneP:0,gen:0,nt:0,last:null,gmax:0,first:null,fin:null,rej:null,adm:null,pre:0}));
  const G=[]; for(let i=0;i<n;i++) G.push({i,busy:false,t1:0,work:null,run:[],used:0});
  const Q={paid:mkQ(),free:mkQ(),batch:mkQ()};
  let tok=rl, tlast=0, preempt=0, redo=0, nxs=0; const samples=[];
  function pick(){
    if(prio){for(const c of ['paid','free','batch']) if(qLen(Q[c])) return qPeek(Q[c]); return null;}
    let b=null; for(const c of ['paid','free','batch']) if(qLen(Q[c])){const x=qPeek(Q[c]); if(!b||x.arr<b.arr||(x.arr===b.arr&&x.id<b.id)) b=x;}
    return b;}
  function start(g,now){
    let budget=BUDGET; const dec=[], pf=[], run=g.run;
    for(let i=0;i<run.length;i++){const r=run[i]; if(r.first!=null&&r.doneP>=r.Pr&&r.gen<r.out){dec.push(r); budget-=1;}}
    for(let i=0;i<run.length&&budget>0;i++){const r=run[i]; if(r.doneP<r.Pr){const m=Math.min(r.Pr-r.doneP,budget); pf.push([r,r.doneP,r.doneP+m]); budget-=m;}}
    while(budget>0&&run.length<CAP){
      const c=pick(); if(!c) break;
      const need=blocks(c.P+c.out);
      while(g.used+need>POOL_B&&preOn&&c.cls!=='batch'){
        let v=null; for(const r of run) if(r.cls==='batch'&&(!v||r.adm>v.adm||(r.adm===v.adm&&r.id>v.id))) v=r;
        if(!v) break;
        run.splice(run.indexOf(v),1); g.used-=blocks(v.P+v.out);
        const di=dec.indexOf(v); if(di>=0){dec.splice(di,1); budget+=1;}
        for(let i=pf.length-1;i>=0;i--) if(pf[i][0]===v){budget+=pf[i][2]-pf[i][1]; pf.splice(i,1);}
        redo+=v.doneP<v.Pr?v.doneP:v.P+v.gen-1;
        v.Pr=v.P+v.gen; v.doneP=0; v.pre++; preempt++; qFront(Q.batch,v);}
      if(g.used+need>POOL_B) break;
      qShift(Q[c.cls]); run.push(c); g.used+=need; c.adm=now;
      const m=Math.min(c.Pr-c.doneP,budget); pf.push([c,c.doneP,c.doneP+m]); budget-=m;}
    if(!dec.length&&!pf.length){g.busy=false; g.work=null; return;}
    g.busy=true; g.work={dec,pf}; g.t1=now+stepTime(g.work);}
  function finish(g){const t=g.t1, w=g.work, run=g.run;
    const tk=r=>{if(r.last!=null&&t-r.last>r.gmax) r.gmax=t-r.last; r.last=t; r.nt++;};
    for(const [r,a,b] of w.pf){r.doneP=b; if(b===r.Pr){if(r.first==null){r.first=t; r.gen=1;} else r.gen++; tk(r);}}
    for(const r of w.dec){r.gen++; tk(r);}
    let k=0; for(let i=0;i<run.length;i++){const r=run[i]; if(r.first!=null&&r.doneP>=r.Pr&&r.gen>=r.out){r.fin=t; g.used-=blocks(r.P+r.out);} else run[k++]=r;}
    run.length=k; g.work=null; g.busy=false;}
  function sample(upto){while(nxs<=upto&&nxs<=win+EPS){samples.push([nxs,qLen(Q.paid),qLen(Q.free),qLen(Q.batch)]); nxs=Math.round((nxs+dts)*1e6)/1e6;}}
  let nxt=0, guard=0;
  for(;;){
    if(++guard>5e6) throw new Error('runaway simulation');
    const ta=nxt<R.length?R[nxt].arr:Infinity; let tg=Infinity, gb=null;
    for(const g of G) if(g.busy&&g.t1<tg){tg=g.t1; gb=g;}
    if(ta===Infinity&&tg===Infinity) break;
    if(ta<=tg){
      const r=R[nxt++], t=r.arr; sample(t-EPS); let ok=true;
      if(r.key==='script'&&rl){tok=Math.min(rl,tok+(t-tlast)*rl/60); tlast=t; if(tok>=1-1e-12) tok-=1; else {r.rej='rate'; ok=false;}}
      if(ok&&r.cls==='free'&&qlim!=null&&qLen(Q.free)>=qlim){r.rej='queue'; ok=false;}
      if(ok){qPush(Q[r.cls],r); for(const g of G) if(!g.busy) start(g,t);}
    } else {
      const t=tg; sample(t-EPS); finish(gb); start(gb,t);
      for(const h of G) if(!h.busy) start(h,t);}
  }
  sample(win);
  return {R,preempt,redo,samples,cfg};
}

/* nearest-rank percentile, as the serving capstone */
function pct(xs,p){const s=xs.slice().sort((a,b)=>a-b); if(!s.length) return null; const k=Math.max(1,Math.ceil(p/100*s.length-1e-9)); return s[k-1];}
const gapOf=r=>r.gmax;
function metrics(run){
  const out={};
  for(const cls of ['paid','free']){
    const rs=run.R.filter(r=>r.cls===cls); if(!rs.length){out[cls]=null; continue;}
    const lim=rs.filter(r=>r.rej!=='rate'), served=lim.filter(r=>r.rej==null), g=GOALS[cls];
    const ok=served.filter(r=>r.first-r.arr<=g.ttft+EPS&&gapOf(r)<=g.gap+EPS).length;
    out[cls]={n:rs.length,nlim:lim.length,ok,frac:ok/lim.length,p90:served.length?pct(served.map(r=>r.first-r.arr),90):null,
      rejq:rs.filter(r=>r.rej==='queue').length,rejr:rs.length-lim.length,gapmax:served.reduce((m,r)=>Math.max(m,gapOf(r)),0),
      wmax:served.reduce((m,r)=>Math.max(m,r.first-r.arr),0)};}
  const b=run.R.filter(r=>r.cls==='batch');
  out.batch={n:b.length,pre:run.preempt,redo:run.redo,done:b.filter(r=>r.fin!=null&&r.fin<=(run.cfg.window||Infinity)+EPS).length};
  out.end=run.R.reduce((m,r)=>Math.max(m,r.fin||0),0);
  return out;}
const passes=m=>m.paid.frac>=TARGET.paid-1e-9&&m.free.frac>=TARGET.free-1e-9;

/* memoised runs */
const RUNS=new Map();
function runOf(key,mk){let x=RUNS.get(key); if(x) return x; const run=mk(); run.m=metrics(run); run.key=key; RUNS.set(key,run); return run;}

/* goodput per GPU: one GPU, chats with the paid goals, 60 s of seeded arrivals; the highest whole rate with ≥ 90% meeting the goals */
let GP=null;
function goodput(){ if(GP) return GP;
  const ok=r=>runOf('gp|'+r,()=>simulate(load([[WIN,{paid:r}]],1),{n:1,window:WIN})).m.paid.frac>=TARGET.paid-1e-9;
  let lo=1, hi=40; while(hi-lo>1){const mid=(lo+hi)>>1; if(ok(mid)) lo=mid; else hi=mid;}
  GP={g:lo,next:hi}; return GP;}
/* batch jobs per GPU-second: one GPU works through 600 waiting batch jobs */
let BJ=null;
function batchRate(){ if(BJ) return BJ; const run=runOf('bj',()=>simulate(load([[1,{}]],2,600),{n:1})); BJ={n:600,end:run.m.end,per:600/run.m.end}; return BJ;}

/* the day, hour by hour. T: {paid, free, script} at the peak (requests/s); s: {n, rl, th}. jobs is the day's batch capacity: how many
   batch jobs the share θ leaves could process in 24 hours. It is not capped by the jobs handed in (the page's batchOf does that) */
function chatRate(T,h,rl){return PROF[h]*(T.paid+T.free)+(rl?Math.min(T.script,rl/60):T.script);}
function day(T,s){const g=goodput().g, b=batchRate().per, cap=s.n*g, hours=[]; let jobs=0, over=0;
  for(let h=0;h<24;h++){const c=chatRate(T,h,s.rl), u=c/cap, sh=Math.max(0,s.th-u), j=sh*s.n*b*3600; jobs+=j; if(u>1+1e-12) over++; hours.push({h,c,u,sh,j});}
  return {hours,jobs,over,cap,g,b};}
const quietH=()=>{let h=0; for(let i=1;i<24;i++) if(PROF[i]<PROF[h]) h=i; return h;};

/* the two checks of a whole setup: the busiest minute (14:00) and the quietest hour with its batch jobs */
function peakRun(T,s,seed=7){return runOf(['pk',T.paid,T.free,T.script,s.n,s.q,s.rl,s.prio?1:0,seed].join('|'),()=>simulate(load([[WIN,{paid:T.paid,free:T.free,script:T.script}]],seed),{n:s.n,qlim:s.q||null,rl:s.rl,prio:s.prio,window:WIN}));}
function nightRun(T,s,seed=8){const h=quietH(), d=day(T,s), x=d.hours[h];
  return runOf(['nt',T.paid,T.free,T.script,s.n,s.q,s.rl,s.prio?1:0,s.th,seed].join('|'),()=>simulate(load([[NWIN,{paid:T.paid*PROF[h],free:T.free*PROF[h],script:T.script,batch:x.sh*s.n*d.b}]],seed),{n:s.n,qlim:s.q||null,rl:s.rl,prio:s.prio,window:NWIN,sample:1}));}
/* the peak lasts all afternoon, but its run is one minute from an empty queue. A setup whose 14:00 traffic is above capacity and whose
   free queue has no limit would let that queue grow for hours, so it does not pass, whatever the minute shows */
function overNoLimit(T,s){return !s.q&&chatRate(T,PEAK_H,s.rl)>s.n*goodput().g+1e-9;}
function verdict(T,s,jobs){const pk=peakRun(T,s).m, d=day(T,s), over=overNoLimit(T,s), batchOk=d.jobs>=jobs-1e-6, chatOk=passes(pk)&&!over;
  const nt=chatOk&&batchOk?nightRun(T,s).m:null, nightOk=nt?passes(nt):null;
  return {pk,d,nt,over,chatOk,batchOk,nightOk,ok:chatOk&&batchOk&&!!nightOk,gpuh:s.n*24};}
/* how fast a run's GPUs got through requests up to time t: the requests with a first token by t, per second */
function serveRate(run,t){let k=0; for(const r of run.R) if(r.first!=null&&r.first<=t+EPS) k++; return k/t;}

return {simulate,metrics,passes,load,stepTime,pct,gapOf,runOf,goodput,batchRate,day,chatRate,quietH,peakRun,nightRun,overNoLimit,verdict,serveRate,mulberry,attn,blocks,
  GOALS,TARGET,LEN,PROF,PEAK_H,WIN,NWIN,RL,BUDGET,CAP,
  C:{L,D,KVH,HD,FFN,VOCAB,P_LAYER,P_LAYERS,P_HEAD,P_TOTAL,KV_TOK,W_READ,W_MEM,FS,BS,GiB,BLOCK,POOL_T,POOL_B}};
})();
/* ENGINE END */
if(typeof module!=='undefined'&&typeof window==='undefined'){module.exports=ADM; return;}
window.ADM_ENGINE=ADM;
/* UI BEGIN */
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='admission', C=ADM.C;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const zh=()=>A.lang()==='zh';
const DEF=(text,tip)=>A.ui.def(text,tip);
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
const live=id=>{const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[id]; return !!(l&&l.st&&l.st!=='planned');};
const labName=(id,en,zhs)=>live(id)?labLink(id):tx(`the lab “${en}”`,`“${zhs}”这个 Lab`);
const M=t=>A.ui.math(t);
const J_=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const cite=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const M8B='Llama‑3.1‑8B';
const f1=x=>A.fmt.num(x,1), f2=x=>A.fmt.num(x,2), fint=n=>A.fmt.int(Math.round(n));
const fT=x=>x==null?'—':x<1?(zh()?`${Math.round(x*1000)} 毫秒`:`${Math.round(x*1000)} ms`):x<10?(zh()?`${f2(x)} 秒`:`${f2(x)} s`):(zh()?`${fint(x)} 秒`:`${fint(x)} s`);
const fP=x=>`${x>=1?100:x<=0?0:Math.min(99,Math.max(1,Math.round(x*100)))}%`;
const fJobs=n=>n>=1e6?A.fmt.big(n):fint(Math.round(n/1000)*1000);
const fRate=x=>A.fmt.num(x,Math.abs(x-Math.round(x))<1e-9?0:1);
const fHour=h=>`${h}:00`;
/* the day's batch jobs (R14): cap is the capacity, how many jobs the GPUs could process at most; done = min(cap, handed in), so never more
   than were handed in; spare is the capacity left over once every job is done; short is the jobs left undone */
const batchOf=(d,sub)=>({cap:d.jobs,sub,done:Math.min(d.jobs,sub),spare:Math.max(0,d.jobs-sub),short:Math.max(0,sub-d.jobs)});

/* ---------- the lab's own traffic (stated in Settings and the fidelity card) ---------- */
const BASE_T={paid:40,free:85,script:25}, BASE_JOBS=120000;
const CH_T={paid:50,free:100,script:30}, CH_JOBS=150000;
const PRICE=3;
const BASE={n:6,q:0,rl:0,prio:0,th:60};
const cfgOf=s=>({n:s.n,q:s.q,rl:s.rl,prio:!!s.prio,th:s.th/100});
const G=()=>ADM.goodput().g, BJ=()=>ADM.batchRate();
const perH=()=>BJ().per*3600;
const peakRate=(T,rl)=>ADM.chatRate(T,ADM.PEAK_H,rl);
const avgRate=(T,rl)=>{let s=0; for(let h=0;h<24;h++) s+=ADM.chatRate(T,h,rl); return s/24;};
/* a setup in words. lvl names only what the page has taught so far (S6): 'none' parts 1–4 (no rules yet), 'admit' part 5 (the rules
   that are on), 'full' from part 7 on, in Scenarios and in the challenge (every setting; paid first always comes with preemption) */
const setupName=(s,lvl='full')=>{const p=[tx(`${s.n} GPUs`,`${s.n} 张卡`)], j=x=>x.join(tx(', ','，'));
  if(lvl==='none') return j([...p,tx('no rules at all','没有任何规则')]);
  if(lvl==='admit'){if(s.q) p.push(tx(`free queue ≤ ${s.q}`,`免费请求最多排 ${s.q} 个`)); if(s.rl) p.push(tx('rate limit','限流')); if(s.prio) p.push(tx('paid first','付费优先'));
    return j(p.length>1?p:[...p,tx('no rules at all','没有任何规则')]);}
  p.push(s.q?tx(`free queue ≤ ${s.q}`,`免费请求最多排 ${s.q} 个`):tx('no queue limit','不限排队'));
  p.push(s.rl?tx('rate limit','限流'):tx('no rate limit','不限流'));
  p.push(s.prio?tx('paid first, with preemption','付费优先，可以抢占'):tx('first come, first served','先来先服务'));
  p.push(tx(`batch threshold ${s.th}%`,`批处理门槛 ${s.th}%`)); return j(p);};

/* ---------- definitions (S4, S6) ---------- */
const TIP={
 ttft:()=>tx('Time to first token (TTFT): from the moment a request arrives to its first answer token. It includes the time it waits in a queue.','首 token 延迟（TTFT）：从请求到达，到它的第一个回答 token。包含排队的时间。'),
 gap:()=>tx('Gap: once the answer has started, how long the user waits for the next token. A request meets its goal only if its longest gap is short enough.','间隔：回答开始以后，用户等下一个 token 等了多久。达标看最长的那次。'),
 goals:()=>tx('Goal: what one request must get, such as how soon its first token comes and how long the longest gap between two tokens may be. Paid and free chats have different goals, listed in part 1; the paid ones are the serving capstone’s. A free request turned away for a full queue counts as missed; one stopped by its key’s rate limit is left out.',
   '目标：一个请求要做到什么，比如多快拿到首 token、两个 token 之间最多等多久。付费聊天和免费聊天的目标不一样，列在第 1 部分；付费的和推理服务综合挑战一样。因为排队满了被拒绝的免费请求算没达标；被它的 key 的限流挡下的，不算在内。'),
 slo:()=>tx('Service target (SLO): the share of requests that must meet their goals. In this lab it is 90% for paid chats and 80% for free chats.','服务目标（SLO）：要有多少比例的请求达标。这个 Lab 里付费聊天是 90%，免费聊天是 80%。'),
 batch:()=>tx('Batch job: work nobody watches live, such as summarising a long document. Its prompt is much longer than a chat’s. The jobs are handed in together at midnight and only need to be done within a day.','批处理任务：没人在线等着看的活，比如给一篇长文档写摘要。它的 prompt 比聊天长得多。这些任务半夜一起交上来，一天之内做完就行。'),
 gateway:()=>tx('Gateway: every request reaches it first. It hands the request to a GPU that has room (fewer than 1,024 requests running and room for its KV cache); when every GPU is full, the request waits in the gateway’s queue. Whichever GPU frees room first takes the next request.','网关：请求先到这里。它把请求交给有空位的卡（同时跑的请求不到 1,024 个，KV cache 也放得下）；卡都满了，请求就在网关排队。哪张卡先空出位置，就交给哪张卡。'),
 cap:()=>tx('Capacity: how many requests per second all the GPUs together can take while 90% still meet the paid goals, that is, the number of GPUs × the goodput per GPU.','容量：所有卡加起来每秒能接多少个请求、还能让 90% 的请求达到付费的目标，也就是卡数 × 每张卡的达标吞吐。'),
 goodput:()=>tx('Goodput per GPU (DistServe’s definition, as in P/D disaggregation): the highest rate the service can take while at least 90% of requests meet their goals, divided by the number of GPUs. Here it is measured on one GPU with one minute of seeded random chats, in whole requests per second.','每张卡的达标吞吐（goodput per GPU，DistServe 的定义，和 P/D 分离那个 Lab 一样）：至少 90% 的请求达标时，服务每秒最多能接多少个请求，再除以卡数。这里在一张卡上用一分钟带种子的随机聊天请求来量，速率取整数个/秒。'),
 p90:()=>tx('TTFT p90: 90% of the requests got their first token within this time (nearest rank, as in the serving capstone).','首 token 的 p90：90% 的请求在这个时间之内拿到首 token（取最近的排名，和推理服务综合挑战一样）。'),
 gpuh:()=>tx(`GPU hour: one GPU for one hour. GPU hours add up all the GPUs: 4 GPUs for 24 hours are 96 GPU hours. This lab assumes $${PRICE} per GPU hour.`,`GPU 小时：一张卡用一小时。GPU 小时把所有卡加起来：4 张卡跑 24 小时是 96 GPU 小时。本 Lab 假设每 GPU 小时 ${PRICE} 美元。`),
 qlim:()=>tx('Queue limit: at most this many free requests may wait in the gateway. One more gets an immediate HTTP 429 (Too Many Requests) with a Retry-After header saying when to try again. In this lab a turned-away request does not come back.','排队上限：网关里最多排这么多个免费请求。再来一个，网关马上回 HTTP 429（Too Many Requests），带一个 Retry-After 头，告诉客户端过多少秒再试。这个 Lab 里，被拒绝的请求不再回来。'),
 rl:()=>tx('Rate limit: each API key may send at most 60 requests a minute, kept with a token bucket; requests over it get 429.','限流：每个 API key 每分钟最多发 60 个请求，用令牌桶实现，超过的回 429。'),
 bucket:()=>tx('Token bucket: a counter that refills at a fixed rate up to a cap. A request passes if it can take one token. Short bursts up to the cap pass; a steady stream faster than the refill rate does not.','令牌桶：一个按固定速度回涨、有上限的计数器。请求能拿到一个令牌就放行。不超过上限的短暂突发能过；持续比补充速度快的请求流过不去。'),
 prio:()=>tx('Priority: the gateway keeps three queues, paid, free and batch. When a GPU has room it takes from the paid queue first, then free, then batch. Without priority it takes the request that arrived first.','优先级：网关有付费、免费、批处理三个队。卡空出位置时，先从付费队里取，付费的没有了再取免费的，最后才取批处理。不设优先级时，谁先到先取谁。'),
 preempt:()=>tx('Preemption: when a waiting chat cannot fit its KV cache, the GPU stops the batch job that started latest, frees its KV cache and lets the chat in. The job goes back to the front of the batch queue. Later it is recomputed (vLLM’s default, “recompute”): the tokens it already wrote are kept, but the KV cache of its prompt and of those tokens is computed again.','抢占：排队的聊天请求放不下 KV cache 时，卡把最晚开始的那个批处理任务停下，释放它的 KV cache，让聊天请求进来。这个任务回到批处理队的最前面，之后重算（vLLM 默认的 recompute）：它已经写出的 token 保留，但它的 prompt 和这些 token 的 KV cache 要再算一遍。'),
 bcap:()=>tx('The day’s batch capacity: how many batch jobs the GPUs could process at most in 24 hours, adding up the share θ leaves in each hour. The jobs actually done are the capacity or the jobs handed in, whichever is smaller; when the capacity is larger, the rest is spare.','全天批处理容量：把每个小时 θ 留给批处理的那一份加起来，这组卡 24 小时内最多能处理多少批处理任务。实际做完的数量取容量和交上来的任务数里较小的那个；容量比交上来的多时，多出来的就是余量。'),
 th:()=>tx('Batch threshold θ: each hour, the gateway compares how busy the chats keep the fleet (the hour’s chat rate ÷ capacity) with θ. Below θ, it lets batch jobs in until the fleet is θ busy; at or above θ, no batch job starts that hour.','批处理门槛 θ：每个小时，网关看聊天让卡忙到几成（这一小时的聊天速率 ÷ 容量），和 θ 比。低于 θ，就放批处理任务进来，直到卡忙到 θ；达到 θ，这一小时不开始新的批处理任务。'),
 script:()=>tx('The script: one free API key used by a program that sends 25 requests per second around the clock (this lab’s assumption). Its requests count as free requests.','脚本：一个免费 API key 被一段程序拿去用，全天每秒发 25 个请求（这个 Lab 的假设）。它的请求按免费请求算。')
};
const dGoals=t=>DEF(t,TIP.goals()), dTTFT=t=>DEF(t,TIP.ttft()), dGap=t=>DEF(t,TIP.gap());
const SRC={
 sched:'https://github.com/vllm-project/vllm/blob/03a7d3776fc2f520a70c8d2da1724eab6df6d0e1/vllm/config/scheduler.py',
 args:'https://github.com/vllm-project/vllm/blob/03a7d3776fc2f520a70c8d2da1724eab6df6d0e1/vllm/engine/arg_utils.py',
 opt:'https://github.com/vllm-project/vllm/blob/03a7d3776fc2f520a70c8d2da1724eab6df6d0e1/docs/configuration/optimization.md',
 sgl:'https://github.com/sgl-project/sglang/blob/961bcf481d184be65c2a2879e62faa3e9d2fdc65/docs/docs/advanced_features/server_arguments.mdx',
 fc:'https://github.com/llm-d/llm-d/blob/c690e9acdab3f1c426e3e03ad89de99ebc56bc50/docs/architecture/core/router/epp/flow-control.md',
 io:'https://github.com/llm-d/llm-d/blob/c690e9acdab3f1c426e3e03ad89de99ebc56bc50/docs/api-reference/inferenceobjective.md',
 batch:'https://github.com/openai/openai-cookbook/blob/5bd6cb40b6b84b69b2032d810610aa85064102df/examples/data/oai_docs/batch.txt',
 rl:'https://github.com/openai/openai-cookbook/blob/5bd6cb40b6b84b69b2032d810610aa85064102df/examples/How_to_handle_rate_limits.ipynb',
 p_dist:'https://arxiv.org/abs/2401.09670', p_vllm:'https://arxiv.org/abs/2309.06180'
};

/* ---------- the runs the page shows (all from the engine, memoised) ---------- */
const S1=()=>({n:1,q:0,rl:0,prio:false,th:0});
const oneRun=f=>{const r=Math.round(G()*f*10)/10; return ADM.runOf('one|'+r,()=>ADM.simulate(ADM.load([[ADM.WIN,{paid:r}]],1),{n:1,window:ADM.WIN}));};
const longRates=()=>({hi:Math.round(G()*1.2*10)/10,lo:Math.round(G()*0.7*10)/10});
const longRun=()=>{const x=longRates(); return ADM.runOf('long|'+x.hi,()=>ADM.simulate(ADM.load([[600,{paid:x.hi}],[300,{paid:x.lo}]],3),{n:1,window:900,sample:5}));};
const SURGE={n:4,q:200,rl:60,backlog:600,quiet:20};
const surgeRun=k=>ADM.runOf('surge|'+k,()=>ADM.simulate(ADM.load([[SURGE.quiet,{}],[ADM.WIN-SURGE.quiet,{paid:BASE_T.paid,free:BASE_T.free,script:BASE_T.script}]],11,SURGE.backlog),{n:SURGE.n,qlim:SURGE.q,rl:SURGE.rl,prio:k!=='fcfs',preempt:k==='pre',window:ADM.WIN}));
const pk=s=>ADM.peakRun(BASE_T,cfgOf(s));
const P5={none:{n:4,q:0,rl:0,prio:0,th:60},q:{n:4,q:200,rl:0,prio:0,th:60},rl:{n:4,q:200,rl:60,prio:0,th:60},pf:{n:4,q:200,rl:60,prio:1,th:60}};
const P7S=th=>({n:4,q:200,rl:60,prio:1,th});
const LOADS=[0.7,0.9,1.0,1.1];

/* ---------- prediction (part 4) ---------- */
const PRED=[
 {v:'slow20',t:{en:'Every request is about 20% slower; most still get their first token within 1 s',zh:'每个请求慢 20% 左右；大部分还是 1 秒内拿到首 token'}},
 {v:'grows',t:{en:'The wait keeps growing: by the tenth minute a new request waits more than a minute, and almost every request misses',zh:'等待越来越长：到第 10 分钟，新来的请求要等一分多钟，几乎所有请求都超时'}},
 {v:'extra',t:{en:'Only the extra 20% miss their goal; the rest are served on time',zh:'只有多出来的那 20% 超时，其余的照常达标'}},
 {v:'crash',t:{en:'The GPU runs out of memory and the server crashes',zh:'卡的显存不够，服务崩掉'}}];
const PRED_RIGHT='grows';

/* ---------- state ---------- */
let S={...BASE}, mode='guided', step=0, pred=null, predShown=false, root=null, app=null;
let p2n=null, p3f=null, p5k=null, p6k=null, p7t=null, tried={}, saved=null, chSubmitted=false, chReveal=false, annT=null, stageT=0, userAct=false, spdUser=null;
const watched=new Set();
let PLAY={on:false,raf:0,last:0,key:''}, phNow=Infinity, lastRunKey='', nPlays=0;

/* which picture the stage shows: v.day = {T, s, jobs, mark} or null; v.run(); v.what (title) */
function viewOf(){
  if(mode==='guided'){const id=STEPS[step].id;
    if(id==='mission') return fleetView(BASE_T,{...BASE},BASE_JOBS,'none');
    if(id==='capacity') return fleetView(BASE_T,{...BASE,n:p2n||6},BASE_JOBS,'none');
    if(id==='queue'){const f=p3f==null?0.7:p3f; return {kind:'one',run:()=>oneRun(f),f,day:null};}
    if(id==='predict') return predShown?{kind:'long',run:longRun,day:null}:{kind:'one',run:()=>oneRun(1.0),f:1.0,day:null};
    if(id==='admit') return fleetView(BASE_T,P5[p5k||'none'],BASE_JOBS,'admit');
    if(id==='prio'){const k=p6k||'fcfs'; return {kind:'surge',k,run:()=>surgeRun(k),day:null};}
    if(id==='batch'){const s=P7S(p7t||60), v=fleetView(BASE_T,s,BASE_JOBS); v.kind='night'; v.run=()=>ADM.nightRun(BASE_T,cfgOf(s)); return v;}
    return fleetView(BASE_T,P7S(100),BASE_JOBS);}
  if(mode==='challenge') return fleetView(CH_T,{...S},CH_JOBS);
  return fleetView(BASE_T,{...S},BASE_JOBS);}
function fleetView(T,s,jobs,lvl='full'){return {kind:'peak',T,s,jobs,lvl,run:()=>ADM.peakRun(T,cfgOf(s)),day:{T,s,jobs}};}
/* a setup above capacity at 14:00 with no free queue limit: its minute can pass while the afternoon would not (see the engine) */
const overOf=v=>!!(v&&v.kind==='peak'&&v.day&&ADM.overNoLimit(v.day.T,cfgOf(v.day.s)));
const curView=()=>viewOf();
const curRun=()=>curView().run();
const isP=id=>mode==='guided'&&STEPS[step].id===id;
const mech=()=>false;
const spdNow=()=>spdUser||'fast';
const RUN_SECS={slow:30,med:20,fast:10};

/* ---------- the guided parts ---------- */
const lever=(v,title,desc)=>({v,title,desc,result:''});
const lk=(id,t)=>live(id)?labLink(id,t):esc(t);
const STEPS=[
 {id:'mission',title:{en:'A day of traffic, three kinds of users',zh:'一天的流量，三种用户'},lead:{en:'Paid chats, free chats and batch jobs share one fleet.',zh:'付费聊天、免费聊天和批处理任务共用一组卡。'},
  body:()=>tx(`<p>You run a chat service. Every H100 holds one copy of ${M8B} in BF16. Three kinds of requests come in, each with its own ${dGoals('goal')}:</p><ul class="adm-ul"><li><b>Paid chats</b>: the ${dTTFT('first token')} within 1 s, and never more than 150 ms between two tokens (the longest ${dGap('gap')}).</li><li><b>Free chats</b>: the first token within 5 s, and the same 150 ms between tokens.</li><li><b>${DEF('Batch jobs',TIP.batch())}</b>, such as a summary for each of many long documents: nobody waits for them live. They are handed in at midnight and only need to be done within 24 hours.</li></ul><p>Not every chat can make it, so each kind also has a ${DEF('service target',TIP.slo())}: at least 90% of paid requests must meet their goal, and at least 80% of free ones.</p><p>Every request first reaches a ${DEF('gateway',TIP.gateway())}. It hands the request to a GPU with room; when all GPUs are full, the request waits in the gateway’s queue.</p><p>The top chart is this lab’s day of chat traffic: high in the day, low at night, busiest in mid-afternoon. The dashed line is what the GPUs can take together, their ${DEF('capacity',TIP.cap())}. The hatched blocks are batch jobs: they are let in at night, when the GPUs are idle, and part 7 explains how. The band marked “script” is one free API key that a program uses to send requests all day; they count as free requests. Red marks the traffic above capacity.</p><p>The bottom chart zooms into the busiest minute of the afternoon. Here ${BASE.n} GPUs serve the traffic with no rules at all.</p><p>How many GPUs does the traffic need? And when there is too much, who should wait?</p>`,
     `<p>你负责一个聊天服务：每张 H100 放一份 ${M8B}（BF16）。请求有三种，各有各的${dGoals('目标')}：</p><ul class="adm-ul"><li><b>付费聊天</b>：1 秒内拿到${dTTFT('首 token')}，两个 token 之间最多等 150 毫秒（最长的${dGap('间隔')}）。</li><li><b>免费聊天</b>：5 秒内拿到首 token，两个 token 之间同样最多等 150 毫秒。</li><li><b>${DEF('批处理任务',TIP.batch())}</b>：没人在线等，比如给一大批长文档各写一段摘要。半夜交上来，24 小时内做完就行。</li></ul><p>聊天请求不可能个个都做到，所以每种还有一个${DEF('服务目标',TIP.slo())}：付费请求至少 90% 要达标，免费请求至少 80%。</p><p>每个请求先到${DEF('网关',TIP.gateway())}。网关把它交给有空位的卡；卡都满了，请求就在网关排队。</p><p>上面的图是这个 Lab 设定的一天的聊天流量：白天高、夜里低，下午最忙。虚线是这些卡加起来能接住的速率，叫${DEF('容量',TIP.cap())}。图里的斜线是批处理任务：夜里卡闲的时候才放它们进来，第 7 部分细讲。标着“脚本”的那一层，是一个免费 API key 被一段程序拿去全天不停地发请求，这些请求按免费请求算。红色是超过容量、卡接不住的那部分流量。</p><p>下面的图放大看下午最忙的那一分钟。现在是 ${BASE.n} 张卡，没有任何规则。</p><p>这些流量要几张卡？接不住的时候，让谁等？</p>`)},
 {id:'capacity',title:{en:'Size the fleet for the peak',zh:'按高峰配卡'},lead:{en:'GPUs = peak rate ÷ goodput per GPU.',zh:'卡数 = 高峰速率 ÷ 每张卡的达标吞吐。'},
  body:()=>{const g=G(), pr=peakRate(BASE_T,0), av=avgRate(BASE_T,0);
   return tx(`<p>One GPU can take only so many requests per second; beyond that, requests queue. So first measure one GPU: send it chats faster and faster, and find the highest rate at which 90% still meet the paid goals. That is the ${DEF('goodput per GPU',TIP.goodput())}, ${g} requests per second in this lab.</p><p>Labs such as ${lk('quantization','quantization')} and ${lk('spec-decoding','speculative decoding')} all raise this number; the ${lk('pd-disagg','P/D disaggregation')} lab uses it to compare setups. The higher the goodput per GPU, the fewer GPUs the peak needs.</p><p>With it, the number of GPUs follows. The peak rate is the traffic at 14:00, the busiest hour: ${fRate(pr)} chat requests a second, 40 paid, 85 free and the script’s 25.</p>${EQ([['\\text{GPUs}','\\frac{\\text{peak rate}}{\\text{goodput per GPU}}'],['',`\\frac{${fRate(pr)}}{${g}} \\approx ${f1(pr/g)}`]])}<p>Pick a fleet and watch the busiest minute at 14:00.</p>`,
     `<p>一张卡每秒能接的请求有上限；超过上限，请求就要排队。所以先量一张卡：越来越快地给它发聊天请求，找出还能让 90% 的请求达到付费目标的最高速率。这就是${DEF('每张卡的达标吞吐',TIP.goodput())}（goodput per GPU），这个 Lab 里是每秒 ${g} 个请求。</p><p>${lk('quantization','量化')}、${lk('spec-decoding','投机解码')}这类 Lab 提高的都是这个数；${lk('pd-disagg','P/D 分离')} Lab 也用它比较方案。每张卡的达标吞吐越高，高峰要的卡就越少。</p><p>知道了它，卡数就能算出来。高峰速率是最忙的下午 2 点的流量：每秒 ${fRate(pr)} 个聊天请求，其中付费 40 个、免费 85 个，脚本 25 个。</p>${EQ([['\\text{卡数}','\\frac{\\text{高峰速率}}{\\text{每张卡的达标吞吐}}'],['',`\\frac{${fRate(pr)}}{${g}} \\approx ${f1(pr/g)}`]])}<p>选一种卡数，看看下午 2 点最忙的那一分钟。</p>`)+
   A.ui.levers({aria:tx('Fleet size','卡数'),value:p2n==null?'':String(p2n),options:[
     lever('4',tx('4 GPUs: sized for the day’s average','4 张：按全天平均配'),tx(`The day averages about ${fint(av)} requests per second.`,`全天平均每秒约 ${fint(av)} 个请求。`)),
     lever('6',tx('6 GPUs: sized for the peak','6 张：按高峰配'),tx(`${fRate(pr)} ÷ ${g}, rounded up.`,`${fRate(pr)} ÷ ${g}，向上取整。`)),
     lever('7',tx('7 GPUs: the peak plus one spare','7 张：高峰再多留一张'),tx('Room for a busier day than this one.','给比今天更忙的日子留余量。'))]});},
  obs:()=>{const n=p2n, m=pk({...BASE,n}).m, d=ADM.day(BASE_T,cfgOf({...BASE,n})), all=m.paid.frac>=1&&m.free.frac>=1, qh=ADM.quietH();
   if(n===4) return tx(`<p class="badline">With 4 GPUs, the traffic is above capacity from morning to evening (red in the chart). In the busiest minute the queue keeps growing, and only ${fP(m.paid.frac)} of paid requests meet their goals.</p><p>Enough on average is not enough at the peak: the extra requests of the afternoon have nowhere to go but the queue.</p>`,
     `<p class="badline">4 张卡时，从上午到晚上，流量一直超过容量（图里红色的部分）。最忙的那一分钟，排队一直在涨，付费请求只有 ${fP(m.paid.frac)} 达标。</p><p>平均够用不等于高峰够用：下午多出来的请求没有地方去，只能排队。</p>`);
   if(n===6) return tx(`<p class="okline">With 6 GPUs, every hour is below capacity, and ${all?'every request in the busiest minute meets its goal':`in the busiest minute ${fP(m.paid.frac)} of paid requests meet their goals`}.</p><p>The price is the night: in the quietest hour the chats (script included) keep the GPUs only ${fP(d.hours[qh].u)} busy. Batch jobs use a threshold of ${BASE.th}% by default (part 7 explains it), so these GPUs can process at most about ${fJobs(d.jobs)} batch jobs in a day, fewer than were handed in, and the rest are left undone.</p>`,
     `<p class="okline">6 张卡时，每个小时的流量都在容量以下，最忙的那一分钟${all?'所有请求都达标':`付费请求 ${fP(m.paid.frac)} 达标`}。</p><p>代价在夜里：最闲的凌晨 ${qh} 点，聊天（包括脚本）只让卡忙 ${fP(d.hours[qh].u)}。批处理默认用 ${BASE.th}% 的门槛（第 7 部分讲），所以这组卡一天最多只能处理约 ${fJobs(d.jobs)} 个批处理任务，比交上来的少，剩下的做不完。</p>`);
   return tx(`<p class="okline">One more GPU makes the first token a little faster. 6 GPUs already meet every goal, so this one buys only a little speed and room for a busier day.</p><p>It costs 24 more ${DEF('GPU hours',TIP.gpuh())} a day.</p>`,
     `<p class="okline">多一张卡，首 token 又快了一点。6 张卡已经全部达标，所以这张卡换来的只是一点速度，和给更忙的日子留的余量。</p><p>代价是每天多花 24 ${DEF('GPU 小时',TIP.gpuh())}。</p>`);},
  expect:()=>p2n!=null},
 {id:'queue',title:{en:'Near the limit, the queue grows fast',zh:'接近上限时，排队涨得很快'},lead:{en:'One GPU at 70% to 110% of its goodput.',zh:'一张卡，负载从 70% 到 110%。'},
  body:()=>{const g=G();
   return tx(`<p>Look at one GPU. Requests arrive at random, more in some seconds and fewer in others: in a busy second the extra requests queue, and the GPU works them off in the quieter seconds that follow. The closer the average load is to the GPU’s limit, the fewer quiet seconds are left to work the queue off.</p><p>One GPU can take ${g} requests per second within the goals. Pick a load, that is, the requests arriving each second as a share of those ${g}, and see how long requests wait for their first token.</p>`,
     `<p>只看一张卡。请求随机到达，有的一秒来得多，有的一秒来得少：来得多的那一秒，多出来的请求要排队，卡在后面来得少的几秒里把它们消化掉。平均负载越接近卡的上限，能用来消化排队的空闲就越少。</p><p>一张卡每秒能达标地接 ${g} 个请求。选一个负载，也就是每秒来的请求占这 ${g} 个的几成，看看等首 token 要多久。</p>`)+
   A.ui.levers({aria:tx('Load on one GPU','一张卡的负载'),value:p3f==null?'':String(p3f),options:LOADS.map(f=>({v:String(f),title:tx(`${Math.round(f*100)}%: ${fRate(Math.round(g*f*10)/10)} requests/s`,`${Math.round(f*100)}%：每秒 ${fRate(Math.round(g*f*10)/10)} 个`),desc:'',result:''}))});},
  obs:()=>{const f=p3f, m=oneRun(f).m.paid;
   if(f<=0.7+1e-9) return tx(`<p class="okline">At 70%, requests get their first token quickly, and ${m.frac>=1?'all of them meet':`${fP(m.frac)} meet`} their goals. The queue never gets long: there is plenty of quiet time to work it off.</p>`,`<p class="okline">70% 时，请求都很快拿到首 token，${m.frac>=1?'全部达标':`${fP(m.frac)} 达标`}。排队从来不长：有足够的空闲把它消化掉。</p>`);
   if(f<1-1e-9) return tx(`<p class="okline">At 90%, the first token takes almost twice as long as at 70%, but it is still within the goal. There are fewer quiet seconds left to drain the queue.</p>`,`<p class="okline">90% 时，首 token 要等的时间差不多是 70% 时的两倍，但还在目标以内。能用来消化排队的空闲少了。</p>`);
   if(f<1.05) return tx(`<p>At 100%, the first token takes several times as long as at the lightest load, and ${fP(m.frac)} meet their goals. The ${G()} requests a second from part 2 were measured exactly this way: any faster, and fewer than 90% would meet their goals.</p>`,`<p>100% 时，首 token 要等的时间是最轻负载时的好几倍，${fP(m.frac)} 达标。第 2 部分说的每秒 ${G()} 个，就是这样量出来的：再快一点，达标的就不到 90% 了。</p>`);
   return tx(`<p class="badline">At 110%, requests arrive faster than the GPU finishes them, so the queue keeps growing and only ${fP(m.frac)} meet their goals. The longer this lasts, the worse it gets, because the extra requests are never worked off.</p>`,`<p class="badline">110% 时，请求来得比卡做完得快，排队一直在涨，只有 ${fP(m.frac)} 达标。持续越久越糟，因为多出来的请求永远消化不掉。</p>`);},
  expect:()=>p3f!=null, gate:()=>!!tried.q11},
 {id:'predict',title:{en:'Predict: 20% over, for 10 minutes',zh:'猜一猜：超出 20%，持续 10 分钟'},lead:{en:'Nobody is turned away. What happens?',zh:'没有请求被拒绝，会怎样？'},
  body:()=>{const g=G();
   return tx(`<p>The same single GPU can take ${g} requests per second within the goals.</p><p>Now 20% more arrive, for 10 minutes; then traffic falls to seven tenths of what the GPU can take. There are no rules: every request is accepted and waits its turn.</p><p>What happens to the first token of most requests by the end of the 10 minutes?</p>`,
     `<p>还是这一张卡，每秒能达标地接 ${g} 个请求。</p><p>现在请求多来 20%，持续 10 分钟；之后回落到它能接的七成。没有任何规则：每个请求都收下，排队等。</p><p>到 10 分钟结束时，大部分请求的首 token 会怎样？</p>`)+predHTML();},
  obs:()=>predShown?verdictHTML():'', expect:()=>predShown},
 {id:'admit',title:{en:'Say “try again later” early',zh:'早一点说“稍后再试”'},lead:{en:'A queue limit, a rate limit per key, and paid first.',zh:'排队上限、限流，再让付费请求先走。'},
  body:()=>tx(`<p>Back to the busiest minute of the afternoon, this time with only 4 GPUs. For each request that arrives, the gateway decides: hand it to a GPU now, let it wait, or answer at once “try again later”. This is called admission control.</p><p>The gateway can turn away work it cannot do in time, and it can change who goes first:</p><ul class="adm-ul"><li>A ${DEF('queue limit',TIP.qlim())}: at most 200 free requests may wait. The next one gets HTTP 429 at once, with a Retry-After header that tells the client when to try again; paid requests are never refused.</li><li>A ${DEF('rate limit',TIP.rl())} of 60 requests a minute for each API key, kept with a ${DEF('token bucket',TIP.bucket())}.</li><li>${DEF('Priority',TIP.prio())}: when a GPU has room, the gateway takes paid requests first, then free ones.</li></ul><p>One free key belongs to a ${DEF('script',TIP.script())} that sends 25 requests a second. With the rate limit it gets 1 a second through, and the rest are turned away at once.</p><p>Pick the rules and see how many paid and free requests meet their goals.</p>`,
   `<p>回到下午最忙的那一分钟，这次只用 4 张卡。每来一个请求，网关都要决定：现在交给卡、让它排队，还是马上回一个“稍后再试”。这叫准入控制（admission control）。</p><p>网关可以拒绝来不及做的活，也可以改变谁先走：</p><ul class="adm-ul"><li>${DEF('排队上限',TIP.qlim())}：免费请求最多排 200 个。再来一个，网关马上回 HTTP 429，带一个 Retry-After 头，告诉客户端过多少秒再试；付费请求从不拒绝。</li><li>${DEF('限流',TIP.rl())}：每个 API key 每分钟最多 60 个请求，用${DEF('令牌桶',TIP.bucket())}来记。</li><li>${DEF('优先级',TIP.prio())}：卡空出位置时，网关先取付费请求，再取免费的。</li></ul><p>有一个免费 key 是一段${DEF('脚本',TIP.script())}，每秒发 25 个请求。限流以后，它每秒只能过 1 个，其余的马上被拒绝。</p><p>选一组规则，看看付费和免费请求各有多少达标。</p>`)+
   A.ui.levers({aria:tx('Admission rules','准入规则'),value:p5k||'',options:[
     lever('none',tx('No rules','不设规则'),tx('Every request waits its turn.','每个请求都排队等。')),
     lever('q',tx('Free requests: at most 200 waiting','免费请求最多排 200 个'),tx('The 201st gets 429.','第 201 个回 429。')),
     lever('rl',tx('Also a rate limit: 60 requests a minute per key','再加限流：每个 key 每分钟最多 60 个'),tx('The script gets 1 a second through.','脚本每秒只能过 1 个。')),
     lever('pf',tx('Also paid first','再让付费请求先走'),tx('A GPU with room takes paid requests first.','卡空出位置时，先取付费请求。'))]}),
  obs:()=>{const m=pk(P5[p5k]).m, m0=pk(P5.none).m, mq=pk(P5.q).m;
   if(p5k==='none') return tx(`<p class="badline">With no rules, 4 GPUs take every request and the queue grows the whole minute: only ${fP(m.paid.frac)} of paid requests meet their goals. Everyone is late, the paying users too.</p>`,`<p class="badline">不设规则时，4 张卡把每个请求都收下，排队涨了整整一分钟，付费请求只有 ${fP(m.paid.frac)} 达标。所有人都超时，付费用户也一样。</p>`);
   if(p5k==='q') return tx(`<p>The queue limit turns the extra free requests away at once, and those that stay wait less: free requests on time go from ${fP(m0.free.frac)} to ${fP(m.free.frac)}.</p><p>Paid requests still meet their goals only ${fP(m.paid.frac)} of the time: under first come, first served, paid and free requests queue together in arrival order. The free queue now has a limit, but the paid queue has none and keeps growing.</p>`,
     `<p>排队上限让多出来的免费请求马上被拒绝，留下来的等得短了：免费请求达标的比例从 ${fP(m0.free.frac)} 涨到 ${fP(m.free.frac)}。</p><p>付费请求还是只有 ${fP(m.paid.frac)} 达标：先来先服务时，付费请求和免费请求按到达的先后混在一起排。免费请求的排队有了上限，付费请求的排队却没有，还在一直涨。</p>`);
   if(p5k==='rl') return tx(`<p>With the rate limit too, the script gets only 1 request a second through, and far fewer requests reach the queue. Most of the requests in the “free requests turned away” metric are now the script’s: the rate limit stops them, and they do not count against the free target. Only ${fint(m.free.rejq)} free requests are turned away for a full queue, about ${mq.free.rejq&&m.free.rejq/mq.free.rejq<0.35?'a quarter':'a third'} of the previous step (queue limit only). So this time ${fP(m.free.frac)} of free requests meet their goal.</p><p>Paid requests still meet theirs only ${fP(m.paid.frac)} of the time: first come, first served puts them in arrival order with the free requests. Now try paid first.</p>`,
     `<p>再加上限流，脚本每秒只能过 1 个请求，进排队的请求少了一大截。指标里“被拒绝的免费请求”这个数，现在大部分是脚本的请求：它们被限流挡下，不算进免费请求的目标。因为排队满而被拒绝的免费请求只有 ${fint(m.free.rejq)} 个，大约是上一步（只设排队上限）的${mq.free.rejq&&m.free.rejq/mq.free.rejq<0.35?'四分之一':'三分之一'}。所以这次有 ${fP(m.free.frac)} 的免费请求达标。</p><p>付费请求还是只有 ${fP(m.paid.frac)} 达标：先来先服务时，它们和免费请求按到达的先后排。再让付费请求先走试试。</p>`);
   return tx(`<p class="okline">With paid first, ${fP(m.paid.frac)} of paid requests meet their goals.</p><p>The GPUs are still fewer than the peak needs, so the extra traffic falls on free users: when the queue is full, a new free request gets 429 at once. ${fP(m.free.frac)} of free requests meet their goal, still above the target.</p>`,
     `<p class="okline">付费请求先走以后，${fP(m.paid.frac)} 达标。</p><p>卡还是比高峰少，多出来的流量全落在免费用户身上：排队满了，新来的免费请求马上拿到 429。免费请求 ${fP(m.free.frac)} 达标，还在目标以上。</p>`);},
  expect:()=>p5k!=null, gate:()=>!!tried.pf},
 {id:'prio',title:{en:'When memory is full, preempt a batch job',zh:'显存满了，就抢占批处理任务'},lead:{en:'Paid first is half of it; preemption is the other half.',zh:'付费优先只解决一半，另一半靠抢占。'},
  body:()=>tx(`<p>Queue order solves only half of it: a GPU’s memory can be full. Say the GPUs were idle and the gateway let batch jobs in, and their long prompts filled the KV cache. Then traffic jumps to the afternoon peak, and even a paid chat at the front of the queue has no room for its KV cache.</p><p>${DEF('Preemption',TIP.preempt())} stops the batch job that started latest, frees its KV cache and lets the chat in. The job goes back to the batch queue and is recomputed later (vLLM calls this recompute).</p><p>The same 4 GPUs, the same queue limit and rate limit. At the start ${SURGE.backlog} batch jobs are waiting, and soon after the peak traffic arrives. Pick a rule.</p>`,
   `<p>排好顺序只解决了一半：卡的显存可能已经满了。比如卡空着的时候，网关放进来一批批处理任务，它们很长的 prompt 把 KV cache 占满了。这时流量一下子涨到下午高峰的水平，排在最前面的付费请求也没有地方放 KV cache。</p><p>${DEF('抢占',TIP.preempt())}（preemption）把最晚开始的那个批处理任务停下，释放它的 KV cache，让聊天请求进来。这个任务回到批处理队里，之后重算（vLLM 叫 recompute）。</p><p>同样 4 张卡，同样的排队上限和限流。开始时有 ${SURGE.backlog} 个批处理任务在排队，不久高峰流量就到了。选一种规则。</p>`)+
   A.ui.levers({aria:tx('Who goes first','谁先走'),value:p6k||'',options:[
     lever('fcfs',tx('First come, first served','先来先服务'),tx('Whoever arrives first runs first. Here the batch jobs arrived first, so they run first.','谁先到谁先做。这里批处理任务先到，所以先做它们。')),
     lever('prio',tx('Paid first, no preemption','付费优先，不抢占'),tx('Memory that a finished job frees goes to paid requests first.','任务做完空出的显存，先给付费请求。')),
     lever('pre',tx('Paid first, with preemption','付费优先，可以抢占'),tx('When a chat request finds no room for its KV cache, the GPU stops the batch job that started latest and gives its memory to that chat request.','聊天请求放不下 KV cache 时，卡停掉最晚开始的那个批处理任务，把显存腾给这个聊天请求。'))]}),
  obs:()=>{const m=surgeRun(p6k).m, none=m.paid.ok+m.free.ok===0;
   if(p6k==='fcfs') return tx(`<p class="badline">First come, first served puts every chat behind the batch jobs, and ${none?'not one chat':'almost no chat'} meets its goal.</p>`,`<p class="badline">先来先服务时，每个聊天请求都排在这些批处理任务后面，${none?'没有一个':'几乎没有'}聊天请求达标。</p>`);
   if(p6k==='prio') return tx(`<p>Paid requests now go to the front, and ${fP(m.paid.frac)} meet their goals. But the batch jobs hold the memory, so free requests wait for jobs to finish and free their KV cache: only ${fP(m.free.frac)} meet the free goal.</p>`,`<p>付费请求排到了最前面，${fP(m.paid.frac)} 达标。但显存被批处理任务占着，免费请求要等任务做完、腾出 KV cache，只有 ${fP(m.free.frac)} 达到免费的目标。</p>`);
   return tx(`<p class="okline">With preemption, ${fP(m.paid.frac)} of paid and ${fP(m.free.frac)} of free requests meet their goals.</p><p>The cost is ${fint(m.batch.pre)} preemptions: the stopped jobs keep the tokens they wrote, and only their KV cache is computed again. Batch jobs can afford that: their deadline is a day away.</p>`,
     `<p class="okline">加上抢占，付费请求 ${fP(m.paid.frac)} 达标，免费的 ${fP(m.free.frac)}。</p><p>代价是 ${fint(m.batch.pre)} 次抢占：被停下的任务写出的 token 都保留，只是 KV cache 要重算。批处理任务等得起：它们的期限在一天以后。</p>`);},
  expect:()=>p6k!=null, gate:()=>!!tried.pre},
 {id:'batch',title:{en:'Batch jobs fill the quiet hours',zh:'批处理放到流量低的时候'},lead:{en:'The threshold θ decides when batch jobs run.',zh:'门槛 θ 决定批处理什么时候跑。'},
  body:()=>tx(`<p>At midnight this lab hands in ${fJobs(BASE_JOBS)} batch jobs, to be done within 24 hours. At night the GPUs are mostly idle, which is when the jobs should run.</p><p>The gateway uses a ${DEF('threshold θ',TIP.th())}: each hour it checks how busy the chats keep the GPUs. If that is below θ, it lets batch jobs in until the GPUs are θ busy. The higher θ, the more of the GPUs go to batch jobs.</p><p>Still 4 GPUs, with the queue limit, the rate limit, and paid first with preemption. Pick a threshold and see how many batch jobs these GPUs can process at most in 24 hours (the day’s batch capacity), and how many of the jobs handed in are done.</p><p>The bottom chart now shows two minutes of the quietest hour, 3:00, when batch jobs fill the most.</p>`,
   `<p>这个 Lab 里，每天半夜交上来 ${fJobs(BASE_JOBS)} 个批处理任务，24 小时内做完就行。夜里卡大多闲着，正好用来做它们。</p><p>网关用一个${DEF('门槛 θ',TIP.th())}。网关每个小时看一次：聊天请求让卡忙到了几成。不到 θ，就放批处理任务进来，补到卡忙到 θ 为止。θ 越高，分给批处理的卡越多。</p><p>还是 4 张卡，加上排队上限、限流，以及可以抢占的付费优先。选一个门槛，看看这组卡 24 小时内最多能处理多少批处理任务（全天批处理容量），交上来的任务能做完多少。</p><p>下面的图换成最闲的凌晨 3 点里的两分钟：这时批处理任务最多。</p>`)+
   A.ui.levers({aria:tx('Batch threshold','批处理门槛'),value:p7t==null?'':String(p7t),options:[60,80,100].map(t=>({v:String(t),title:tx(`θ = ${t}%`,`θ = ${t}%`),desc:t===100?tx('Batch jobs take whatever the chats leave.','聊天剩下的全给批处理。'):'',result:''}))}),
  obs:()=>{const s=P7S(p7t), d=ADM.day(BASE_T,cfgOf(s)), ok=d.jobs>=BASE_JOBS, nt=ADM.nightRun(BASE_T,cfgOf(s)).m, half=d.jobs<BASE_JOBS/2, b=batchOf(d,BASE_JOBS);
   const head=ok?tx(`<p class="okline">With θ = ${p7t}%, the share the chats leave could process about ${fJobs(b.cap)} batch jobs in a day: that is the day’s batch capacity. It is more than the jobs handed in, so every job is done, and the capacity left over is spare.</p>`,`<p class="okline">θ = ${p7t}% 时，聊天剩下的卡一天最多能处理约 ${fJobs(b.cap)} 个批处理任务，这是全天批处理容量。它比交上来的任务多，所以任务全部做完，多出来的容量是余量。</p>`)
     :tx(`<p class="badline">With θ = ${p7t}%, these GPUs can process at most about ${fJobs(b.cap)} batch jobs in a day, ${half?'less than half of those handed in':'still fewer than were handed in'}, so the rest are left undone. A lower threshold leaves more of the night idle.</p>`,`<p class="badline">θ = ${p7t}% 时，这组卡一天最多只能处理约 ${fJobs(b.cap)} 个批处理任务，${half?'不到交上来的一半':'还是比交上来的少'}，剩下的做不完。门槛越低，夜里闲着的卡越多。</p>`);
   const night=p7t===100?tx(`<p>At night the jobs fill most of the memory; when chats arrive, there are ${fint(nt.batch.pre)} preemptions in these two minutes, and ${nt.paid.frac>=1?'every paid request still meets its goal':`${fP(nt.paid.frac)} of paid requests still meet their goals`}.</p>`,`<p>凌晨 ${ADM.quietH()} 点批处理任务占了大部分显存；聊天请求一来，这两分钟里就抢占了 ${fint(nt.batch.pre)} 次，付费请求${nt.paid.frac>=1?'照样全部达标':`照样 ${fP(nt.paid.frac)} 达标`}。</p>`):'';
   return head+night+tx(`<p>The threshold only decides how much batch work the idle GPUs do; it does not change the cost: 4 GPUs run all day, ${4*24} GPU hours.</p>`,`<p>门槛只决定闲着的卡做多少批处理，不改变花费：4 张卡全天开着，每天 ${4*24} GPU 小时。</p>`);},
  expect:()=>p7t!=null, gate:()=>!!tried.t100},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Size for the peak, decide who waits, fill the quiet hours.',zh:'按高峰配卡，定好谁等，把闲着的卡用上。'},
  body:()=>rememberHTML()}
];
const WHY={capacity:{en:'Pick a fleet to continue.',zh:'选一种卡数才能继续。'},queue:{en:'Try 110% to continue.',zh:'试一下 110% 才能继续。'},predict:{en:'Pick an answer and show the result to continue.',zh:'选一个答案并看结果才能继续。'},
  admit:{en:'Try paid first to continue.',zh:'试一下付费优先才能继续。'},prio:{en:'Try preemption to continue.',zh:'试一下抢占才能继续。'},batch:{en:'Try θ = 100% to continue.',zh:'试一下 θ = 100% 才能继续。'}};
function rememberHTML(){
  const m4=pk({...BASE,n:4}).m, a=pk(P7S(100)).m, d=ADM.day(BASE_T,cfgOf(P7S(100))), sg=surgeRun('pre').m, allP=a.paid.frac>=1, bt=batchOf(d,BASE_JOBS);
  return tx(`<ul class="adm-ul"><li><b>Size the fleet for the peak, not the average.</b> This lab’s afternoon peak is half as high again as the day’s average. Sized for the average with no rules, 4 GPUs leave the afternoon above capacity, the queue keeps growing, and only ${fP(m4.paid.frac)} of paid requests meet their goals. To serve every request on time, divide the peak rate by the goodput per GPU and round up.</li>`+
     `<li><p><b>With fewer GPUs than the peak needs, decide who waits.</b> Paid requests go first. Free requests may wait, but only up to a limit, beyond which they get 429 with a Retry-After at once; a script is held back by a per-key rate limit.</p><p>Even with the rate limit, the afternoon traffic is above the capacity of these GPUs, which is why the afternoon in the chart is still red. The free users absorb the extra: when their queue is full, a new free request gets 429 at once. That keeps ${allP?'every paid request':`${fP(a.paid.frac)} of paid requests`} on time, and ${fP(a.free.frac)} of free requests.</p></li>`+
     `<li><b>Let batch jobs fill what the chats leave.</b> With θ = 100%, the same GPUs could also process about ${fJobs(bt.cap)} batch jobs a day, more than were handed in: every job is done, and the capacity left over is spare. When chats need the memory, a batch job gives back its KV cache and is recomputed later.</li></ul>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Enough on average is enough.” Requests that arrive faster than the GPUs finish them do not spread out by themselves: at 20% over capacity for 10 minutes, one GPU’s queue grows the whole time and almost every request misses.</p><p>✗ “Turning requests away is bad service.” With no limit when the GPUs cannot keep up, the queue keeps growing and every request is late, paid ones too. Give the extra free requests a quick 429 and let paid requests go first, and the requests that stay are answered on time; a turned-away client can retry a little later.</p><p>✗ “Preemption loses the work.” A preempted job keeps every token it wrote; only its KV cache is computed again. For example, when long batch jobs fill the memory and the peak traffic arrives, there are ${fint(sg.batch.pre)} preemptions in one minute, and no answer is lost.</p></div>`+
     `<p class="small">What real systems do: vLLM serves requests first come, first served unless started with the priority policy, and preempts by recomputing (${cite(SRC.sched,'<code>vllm/config/scheduler.py</code>')}, ${cite(SRC.opt,'optimization docs')}). SGLang has <code>--enable-priority-scheduling</code> and <code>--max-queued-requests</code> (${cite(SRC.sgl,'server arguments')}).</p><p class="small">llm-d holds requests at the gateway in priority bands, rejects a request that would go over its queue limits with 429, and can end negative-priority requests in flight (${cite(SRC.fc,'flow control')}).</p><p class="small">OpenAI’s Batch API has a 24-hour completion window, after which unfinished requests expire and are cancelled; it costs 50% less (${cite(SRC.batch,'Batch API guide')}).</p>`,
   `<ul class="adm-ul"><li><b>按高峰配卡，不按平均。</b>这个 Lab 下午的高峰比全天平均多一半。按平均配 4 张卡、又不加任何规则，下午的流量超过容量，排队一直涨，付费请求只有 ${fP(m4.paid.frac)} 达标。要让每个请求都按时，就用高峰速率除以每张卡的达标吞吐，再向上取整。</li>`+
     `<li><p><b>卡比高峰少的时候，先定好谁等。</b>付费请求先走。免费请求可以排队，但有上限，超过的马上回 429 和 Retry-After；不停发请求的脚本，用按 key 的限流挡住。</p><p>限流以后，下午的流量还是比这几张卡的容量多，所以图里下午还是红色。多出来的这部分由免费请求承担：排队满了，新来的免费请求马上拿到 429。这样付费请求${allP?'全部达标':` ${fP(a.paid.frac)} 达标`}，免费的 ${fP(a.free.frac)} 达标。</p></li>`+
     `<li><b>批处理用聊天剩下的卡。</b>门槛开到 100%，同样的卡一天最多还能处理约 ${fJobs(bt.cap)} 个批处理任务，比交上来的多：任务全部做完，多出来的容量是余量。聊天要用显存时，批处理任务让出 KV cache，之后重算。</li></ul>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “平均够了就够了。”来得比卡做得快的请求不会自己散开：超出容量 20%、持续 10 分钟，一张卡的排队一直在涨，几乎所有请求都超时。</p><p>✗ “拒绝请求是坏服务。”卡接不住的时候不设上限，排队一直涨，所有请求都超时，付费的也一样。给多出来的免费请求马上回 429，再让付费请求先走，留下来的请求就能按时拿到回答；被拒绝的客户端可以过一会儿再试。</p><p>✗ “抢占会丢掉做过的活。”被抢占的任务写出的 token 一个都不丢，只是 KV cache 要再算一遍。比如显存被长文档批处理任务占满时，高峰流量一来，一分钟里抢占了 ${fint(sg.batch.pre)} 次，没有丢掉任何回答。</p></div>`+
     `<p class="small">真实系统怎么做：vLLM 默认先来先服务，启动时选 priority 策略才按优先级排；显存不够时用重算的方式抢占（${cite(SRC.sched,'<code>vllm/config/scheduler.py</code>')}、${cite(SRC.opt,'优化文档')}）。SGLang 有 <code>--enable-priority-scheduling</code> 和 <code>--max-queued-requests</code>（${cite(SRC.sgl,'启动参数')}）。</p><p class="small">llm-d 在网关按优先级分档排队，超过排队上限的请求马上回 429，还可以把正在跑的负优先级请求停掉（${cite(SRC.fc,'flow control')}）。</p><p class="small">OpenAI 的 Batch API 完成窗口是 24 小时，没做完的请求会过期取消；价格便宜 50%（${cite(SRC.batch,'Batch API 文档')}）。</p>`)+A.shell.toScenarios();}

function predHTML(){
  return `<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}"${predShown?' disabled':''}>${esc(L(o.t))}</button>`).join('')}</div>`+
   (pred&&!predShown?`<p class="small muted">${tx('Saved. You can still change it before you show the result.','已记下，看结果之前还可以改。')}</p><div class="stepnav"><button type="button" class="btn sm" data-act="pred-show">${tx('Show the result','看结果')}</button></div>`:'');}
function verdictHTML(){
  if(!predShown) return '';
  const right=pred===PRED_RIGHT, r=longRun(), m=r.m.paid, at=t=>(r.samples.find(x=>Math.abs(x[0]-t)<1e-6)||[0,0])[1], q10=at(600), drain=Math.round((at(600)-at(900))/300);
  const ans=tx('the wait keeps growing, and almost every request misses.','等待越来越长，几乎所有请求都超时。'), q=fint(Math.round(q10/100)*100);
  return `<p class="${right?'okline':'badline'}">${right?tx('Your prediction was right. ','你猜对了。'):tx(`Your prediction: “${esc(L(PRED.find(o=>o.v===pred).t))}”. The answer: `,`你的预测是“${esc(L(PRED.find(o=>o.v===pred).t))}”。答案是：`)+ans}</p>`+
   tx(`<p>The queue grows for the whole 10 minutes, to about ${q} requests, and almost no request meets its goal.</p><p>After traffic falls back, the GPU finishes only about ${drain} more requests a second than arrive, so the queue is still not empty 5 minutes later.</p><p>The extra 20% does not hurt only 20% of the requests: with first come, first served, everyone behind the queue waits for everyone in front. The GPU does not crash either: waiting requests sit in the gateway, not in GPU memory.</p>`,
      `<p>排队涨了整整 10 分钟，到最后有约 ${q} 个请求在等，几乎没有请求达标。</p><p>流量回落以后，卡每秒只比来的多做完 ${drain} 个左右，所以 5 分钟后排队还没排完。</p><p>多出来的 20% 伤到的不只是 20% 的请求：先来先服务时，排在后面的每个人都要等前面的人。卡也不会崩：排队的请求在网关里等，不占显存。</p>`);}

/* ---------- scenarios ---------- */
const PRESETS=[
 {id:'avg',s:{n:4,q:0,rl:0,prio:0,th:60},t:{en:'Sized for the average',zh:'按平均配卡'},d:{en:'4 GPUs, no rules.',zh:'4 张卡，不设规则。'}},
 {id:'peak',s:{n:6,q:0,rl:0,prio:0,th:60},t:{en:'Sized for the peak',zh:'按高峰配卡'},d:{en:'6 GPUs, no rules, batch threshold 60% (batch runs at night).',zh:'6 张卡，不设规则，批处理门槛 60%（夜里跑）。'}},
 {id:'lean',s:{n:4,q:200,rl:60,prio:1,th:100},t:{en:'Lean fleet with every rule',zh:'少配卡，规则全开'},d:{en:'4 GPUs, queue limit, rate limit, paid first with preemption, θ = 100%.',zh:'4 张卡，排队上限、限流、付费优先和抢占，θ = 100%。'}},
 {id:'noprio',s:{n:4,q:200,rl:60,prio:0,th:100},t:{en:'All rules, no priority',zh:'规则全开，但不分优先级'},d:{en:'The same, first come, first served.',zh:'同上，但先来先服务。'}},
 {id:'spare',s:{n:7,q:0,rl:60,prio:1,th:80},t:{en:'Spare GPUs, batch to 80%',zh:'多留卡，批处理到 80%'},d:{en:'7 GPUs, rate limit, paid first with preemption, θ = 80%.',zh:'7 张卡，限流、付费优先和抢占，θ = 80%。'}}];
const sameS=(a,b)=>['n','q','rl','prio','th'].every(k=>+a[k]===+b[k]);
const presetNow=()=>PRESETS.findIndex(p=>sameS(p.s,S));
function scenHTML(i){const p=PRESETS[i], v=ADM.verdict(BASE_T,cfgOf(p.s),BASE_JOBS), m=v.pk, allP=m.paid.frac>=1, b=batchOf(v.d,BASE_JOBS);
  const T={
   avg:()=>tx(`<p>Sized for the average, 4 GPUs are above capacity from morning to evening. With no rules the queue keeps growing, and at the peak only ${fP(m.paid.frac)} of paid requests meet their goals: average load is the wrong sizing rule.</p>`,
              `<p>按平均配的 4 张卡，从上午到晚上都超过容量。不设规则，排队一直涨，高峰时只有 ${fP(m.paid.frac)} 的付费请求达标：不能按平均负载配卡。</p>`),
   peak:()=>tx(`<p>Sized for the peak, every chat goal is met: the safest setup for chats. But with a threshold of 60% some GPUs still sit idle at night. These GPUs can process at most about ${fJobs(b.cap)} batch jobs in a day, fewer than were handed in, so the rest are left undone.</p>`,
               `<p>按高峰配卡，聊天全部达标，对聊天最安全。但门槛只有 60%，夜里的卡还有不少闲着。这组卡一天最多只能处理约 ${fJobs(b.cap)} 个批处理任务，比交上来的少，剩下的做不完。</p>`),
   lean:()=>tx(`<p>The GPUs are fewer than the peak needs, so several daytime hours are above capacity (red). Paid requests go first and the extra free requests get 429 at once, so ${allP?'every paid request':`${fP(m.paid.frac)} of paid requests`} and ${fP(m.free.frac)} of free requests meet their goals.</p><p>Batch jobs may use whatever the chats leave, so the day’s batch capacity is about ${fJobs(b.cap)} jobs, more than were handed in: every job is done, and the capacity left over is spare. The day costs ${v.gpuh} GPU hours.</p>`,
               `<p>卡比高峰少，白天有好几个小时超过容量（红色）。付费请求先走，多出来的免费请求马上拿到 429，所以付费请求${allP?'全部达标':` ${fP(m.paid.frac)} 达标`}，免费的 ${fP(m.free.frac)} 达标。</p><p>聊天剩下的卡都可以给批处理用，所以全天批处理容量约 ${fJobs(b.cap)} 个，比交上来的多：任务全部做完，多出来的容量是余量。这一天要 ${v.gpuh} GPU 小时。</p>`),
   noprio:()=>tx(`<p>Without priority, paid requests queue with free ones in order of arrival, and at the peak only ${fP(m.paid.frac)} of paid requests meet their goals. Compare it with the scenario before to see what priority adds to the limits.</p>`,
                 `<p>没有优先级，付费请求和免费请求按到达的先后排，高峰时只有 ${fP(m.paid.frac)} 的付费请求达标。和上一个场景比，就能看出优先级在限流之外还多做了什么。</p>`),
   spare:()=>tx(`<p>Spare GPUs: every goal is met, the peak stays well under capacity, and the day costs ${v.gpuh} GPU hours. It suits a team that would rather pay for headroom.</p>`,
                `<p>多留卡：全部达标，高峰时容量还比流量多一截，每天要 ${v.gpuh} GPU 小时。适合宁可多花钱也要留余量的团队。</p>`)};
  return T[p.id]();}
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see who meets their goals and what the day costs.','选一个场景，看看谁能达标、一天要花多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="adm-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change the GPUs, the rules or the threshold under ${J_('controls','Settings')}.`,`想改卡数、规则或门槛，在“${J_('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;}

/* ---------- challenge: every option is checked in the background, a few runs per tick ---------- */
const CH_START={n:6,q:0,rl:0,prio:0,th:60};
const CH_N=[4,5,6,7], CH_Q=[0,200,50], CH_TH=[60,80,100];
const CH_OPTS=()=>{const o=[]; for(const n of CH_N) for(const q of CH_Q) for(const rl of [0,60]) for(const prio of [0,1]) for(const th of CH_TH) o.push({n,q,rl,prio,th}); return o;};
const chKey=s=>[s.n,s.q,s.rl,s.prio,s.th].join('_');
const chV=s=>ADM.verdict(CH_T,cfgOf(s),CH_JOBS);
const chPick=()=>({n:S.n,q:S.q,rl:S.rl,prio:S.prio,th:S.th});
const CHV=new Map(); let chT=0, chIdx=0;
function chTick(){chT=0; const o=CH_OPTS(); const t0=Date.now(); if(chSubmitted&&!CHV.has(chKey(chPick()))) CHV.set(chKey(chPick()),chV(chPick()));
  while(chIdx<o.length&&Date.now()-t0<60){const s=o[chIdx++]; CHV.set(chKey(s),chV(s));}
  if(chIdx<o.length) chT=setTimeout(chTick,15);
  if(app&&mode==='challenge'&&chSubmitted) updateRail();}
const chReady=()=>chIdx>=CH_OPTS().length;
function chStartBg(){if(!chT&&!chReady()) chT=setTimeout(chTick,200);}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button>`;
  const seg=(key,label,opts,v,note)=>A.ui.seg({id:'adm-ic-'+key,key,label,options:opts,value:String(v),inline:true,note});
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Meet every goal on a busier day with the fewest GPU hours','让一组 H100 在更忙的一天里全部达标，GPU 小时最少')}</h3>`+
   tx(`<p>A new product runs on H100s, one copy of ${M8B} per GPU, all day. Its day has the same shape as before, only busier:</p><ul class="adm-ul"><li>At the peak: ${CH_T.paid} paid and ${CH_T.free} free chat requests per second.</li><li>The script: one free key sending ${CH_T.script} requests per second all day.</li><li>Batch: ${fJobs(CH_JOBS)} jobs handed in at midnight.</li></ul><p>You pass when:</p><ul class="adm-ul"><li>At the peak, 90% of paid and 80% of free requests meet their goals.</li><li>All batch jobs are done within 24 hours.</li><li>At 3:00, the hour with the most batch work, paid and free requests also reach those two shares.</li></ul><p>Use as few GPU hours as you can.</p>`,
      `<p>一个新产品跑在 H100 上，每张卡一份 ${M8B}，全天开着。一天里流量的形状和前面一样，只是更忙：</p><ul class="adm-ul"><li>高峰：每秒 ${CH_T.paid} 个付费、${CH_T.free} 个免费聊天请求。</li><li>脚本：一个免费 key 全天每秒发 ${CH_T.script} 个请求。</li><li>批处理：半夜交上来 ${fJobs(CH_JOBS)} 个任务。</li></ul><p>过关的条件：</p><ul class="adm-ul"><li>高峰时，付费请求 90% 达标、免费请求 80% 达标。</li><li>批处理任务 24 小时内全部做完。</li><li>批处理最多的凌晨 3 点，付费和免费请求也都要达到这两个比例。</li></ul><p>GPU 小时越少越好。</p>`)+
   seg('chn',tx('GPUs, all day','卡数（全天）'),CH_N.map(n=>[String(n),String(n)]),S.n)+
   seg('chq',tx('Free requests that may wait','免费请求最多排几个'),[['0',tx('no limit','不限')],['200','200'],['50','50']],S.q)+
   seg('chr',tx('Rate limit per key','每个 key 限流'),[['0',tx('off','不限')],['60',tx('60 a minute','每分钟 60 个')]],S.rl)+
   seg('chp',tx('Who goes first','谁先走'),[['0',tx('first come','先来先服务')],['1',tx('paid first, with preemption','付费优先，可以抢占')]],S.prio,tx('Paid first also lets a chat preempt the batch job that started latest when memory is full.','付费优先时，显存满了，聊天请求还可以抢占最晚开始的批处理任务。'))+
   seg('cht',tx('Batch threshold θ','批处理门槛 θ'),CH_TH.map(t=>[String(t),t+'%']),S.th)+
   `<div class="goal mono adm-goal" id="adm-ch-g"><span>${tx('GPU hours a day','每天的 GPU 小时')}</span><span><b id="adm-ch-h"></b></span></div>`+
   submit+`<div id="adm-ch-res" class="adm-ch-res" aria-live="polite"></div></div>`;}
/* why a setup misses, as one sentence; s is the setup (for the rate limit) */
function failWhy(v,s){
  if(!v.chatOk){
    if(v.over&&ADM.passes(v.pk)){const r=fRate(v.d.hours[ADM.PEAK_H].c);
      return s.rl?tx(`With the rate limit the peak still brings about ${r} requests a second, more than these GPUs can take, and free requests have no queue limit. The simulated minute does not show it yet, but the peak lasts all afternoon, and the queue keeps growing.`,`限流以后，高峰每秒还有约 ${r} 个请求，比这几张卡的容量多，免费请求的排队又没有上限。这里模拟的一分钟还看不出来，但高峰持续一整个下午，排队会一直涨。`)
        :tx(`The peak brings about ${r} requests a second, more than these GPUs can take, and free requests have no queue limit. The simulated minute does not show it yet, but the peak lasts all afternoon, and the queue keeps growing.`,`高峰每秒约 ${r} 个请求，比这几张卡的容量多，免费请求的排队又没有上限。这里模拟的一分钟还看不出来，但高峰持续一整个下午，排队会一直涨。`);}
    const pf=v.pk.paid.frac<ADM.TARGET.paid-1e-9;
    return pf?tx(`At the peak only ${fP(v.pk.paid.frac)} of paid requests meet their goals: they wait too long in the queue.`,`高峰时只有 ${fP(v.pk.paid.frac)} 的付费请求达标：它们在队里等得太久。`)
      :tx(`At the peak only ${fP(v.pk.free.frac)} of free requests meet their goal: too many wait too long or get 429.`,`高峰时只有 ${fP(v.pk.free.frac)} 的免费请求达标：太多免费请求等得太久，或者拿到了 429。`);}
  if(!v.batchOk){const b=batchOf(v.d,CH_JOBS); return tx(`These GPUs can process at most about ${fJobs(b.cap)} batch jobs in a day, fewer than the ${fJobs(CH_JOBS)} handed in, so the rest are left undone.`,`这组卡一天最多只能处理约 ${fJobs(b.cap)} 个批处理任务，比交上来的 ${fJobs(CH_JOBS)} 个少，剩下的做不完。`);}
  if(v.nt.paid.frac<ADM.TARGET.paid-1e-9) return tx(`In the quietest hour only ${fP(v.nt.paid.frac)} of paid requests meet their goals: they queue behind the batch jobs.`,`凌晨 ${ADM.quietH()} 点只有 ${fP(v.nt.paid.frac)} 的付费请求达标：它们排在批处理任务后面。`);
  return tx(`In the quietest hour only ${fP(v.nt.free.frac)} of free requests meet their goal: the batch jobs hold the memory.`,`凌晨 ${ADM.quietH()} 点只有 ${fP(v.nt.free.frac)} 的免费请求达到免费的目标：显存被批处理任务占着。`);}
function chResultHTML(){
  if(!chSubmitted) return '';
  const pick=chPick(), v=chV(pick);
  if(!chReady()) {chStartBg(); return `<p class="small muted">${tx(`Comparing all settings (${CHV.size}/${CH_OPTS().length})…`,`正在比较所有设置（${CHV.size}/${CH_OPTS().length}）……`)}</p>`;}
  const opts=CH_OPTS(), feas=opts.filter(o=>CHV.get(chKey(o)).ok), best=Math.min(...feas.map(o=>o.n*24)), nb=best/24;
  let head;
  if(!v.ok) head=`<p class="badline"><b>${tx('Not met.','没达标。')}</b>${tx(' ','')}${failWhy(v,pick)}</p>`+(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  else if(v.gpuh<=best) head=`<p class="okline"><b>${tx(`Met, on the fewest GPU hours: ${v.gpuh} a day ($${fint(v.gpuh*PRICE)} at $${PRICE} each).`,`达标，而且 GPU 小时最少：每天 ${v.gpuh}（按每 GPU 小时 ${PRICE} 美元算，${fint(v.gpuh*PRICE)} 美元）。`)}</b></p>`;
  else head=`<p class="okline"><b>${tx(`Met, on ${v.gpuh} GPU hours a day.`,`达标，每天 ${v.gpuh} GPU 小时。`)}</b>${tx(' ','')}${tx(`But ${nb} GPUs are enough too; the cheapest settings are in the chart below.`,`不过用 ${nb} 张卡也能过关，最省的设置在下面的图里。`)}</p>`;
  if(v.ok){const b=batchOf(v.d,CH_JOBS); head+=`<p>${tx(`Batch jobs: these GPUs could process about ${fJobs(b.cap)} in a day (the day’s batch capacity), so all ${fJobs(CH_JOBS)} handed in are done${b.spare>=500?', and the capacity left over is spare':''}.`,`批处理：这组卡一天最多能处理约 ${fJobs(b.cap)} 个（全天批处理容量），所以交上来的 ${fJobs(CH_JOBS)} 个全部做完${b.spare>=500?'，多出来的容量是余量':''}。`)}</p>`;}
  if(!v.ok&&!chReveal) return head;
  const pk0=chKey(pick), st=chKey(CH_START);
  const rows=opts.filter(o=>CHV.get(chKey(o)).ok||chKey(o)===pk0||chKey(o)===st).map(o=>{const x=CHV.get(chKey(o)), me=chKey(o)===pk0, s0=chKey(o)===st;
    return {label:setupName(o),value:x.gpuh,text:tx(`${x.gpuh} GPU h`,`${x.gpuh} GPU 小时`),ok:x.ok,best:x.ok&&x.gpuh===best,me,tags:[...(me?[{text:tx('your pick','你的选择'),me:true}]:[]),...(s0?[{text:tx('initial setup','初始设置')}]:[])]};});
  const v0=CHV.get(st), bo=feas.filter(o=>o.n*24===best);
  const j80=ADM.day(CH_T,cfgOf({...bo[0],th:80})).jobs, alt=CHV.get(chKey({n:nb,q:50,rl:60,prio:0,th:100})), altN=alt&&alt.chatOk&&alt.batchOk&&alt.nt&&!alt.ok?alt.nt.paid.frac:null, pr=fRate(ADM.chatRate(CH_T,ADM.PEAK_H,60));
  const why=tx(`<p>The initial setup misses. ${failWhy(v0,CH_START)}</p><p>The cheapest settings use ${nb} GPUs and need four things at once:</p><ul class="adm-ul"><li><b>The rate limit</b> takes the script’s ${CH_T.script} requests a second down to 1.</li><li><b>A free queue limit.</b> Even with the rate limit the peak brings about ${pr} requests a second, more than these GPUs can take. With a limit, the extra free requests get 429 at once and those in the queue are on time; with no limit, the free queue keeps growing.</li><li><b>θ = 100%.</b> One step lower, θ = 80% can process at most about ${fJobs(j80)} batch jobs a day, fewer than were handed in.</li><li><b>Paid first, with preemption.</b> At night batch jobs fill the memory${altN!=null?`, and with first come, first served only ${fP(altN)} of paid chats meet their goals then`:', and paid chats are on time only because they can preempt'}.</li></ul><p>Caveat: one minute at the peak and two minutes at 3:00 stand for the whole day, and a turned-away request never comes back. A setup above capacity at the peak with no free queue limit counts as a miss, because one minute from an empty queue cannot show its queue growing all afternoon. A real fleet also keeps some headroom for a busier day.</p>`,
     `<p>初始设置没达标。${failWhy(v0,CH_START)}</p><p>最省的设置用 ${nb} 张卡，要同时做到四件事：</p><ul class="adm-ul"><li><b>限流</b>：把脚本每秒 ${CH_T.script} 个请求压到 1 个。</li><li><b>免费请求的排队上限</b>：限流以后，高峰每秒还有约 ${pr} 个请求，比这几张卡的容量多。有上限时，多出来的免费请求马上拿到 429，排在里面的才能按时；不设上限，免费的排队会一直涨。</li><li><b>θ = 100%</b>：低一档的 θ = 80%，一天最多只能处理约 ${fJobs(j80)} 个批处理任务，比交上来的少。</li><li><b>付费优先，可以抢占</b>：凌晨 ${ADM.quietH()} 点批处理任务占满了显存，${altN!=null?`先来先服务的话，只有 ${fP(altN)} 的付费聊天达标`:'付费聊天靠抢占才能按时'}。</li></ul><p>注意：这里用高峰的一分钟和凌晨 3 点的两分钟代表一整天，被拒绝的请求也不会再回来。高峰比容量多、免费请求又不设排队上限的设置，直接判没过，因为从空队列开始的一分钟看不出排队会一直涨。真实的机群还要给更忙的日子留些余量。</p>`);
  return head+A.ui.cmpChart({title:tx('GPU hours a day for the settings that pass','能过关的设置每天要多少 GPU 小时'),aria:tx('GPU hours for each passing setting','每种过关设置的 GPU 小时'),rows,max:Math.max(...CH_N)*24,brief:true,okText:tx('passes every check','全部条件都满足'),noText:tx('misses at least one','至少一条没满足')})+why;}

/* ---------- capability layer ---------- */
const EXITS=()=>{const g=G(), av2=avgRate(BASE_T,0)*200/peakRate(BASE_T,0), nAv=Math.ceil(av2/g-1e-9);
 return {
 explain:[{q:tx('Why does turning some free requests away keep the others on time?','为什么拒绝一部分免费请求，其他请求反而能按时？'),
   o:[tx('When requests arrive faster than the GPUs finish them, the queue only grows; a queue limit stops that growth, so those who are let in wait a bounded time.','请求来得比卡做得快时，排队只会越来越长；排队上限让它不再涨，进来的请求等的时间就有了上限。'),
      tx('Rejected requests free GPU memory that the others then use to run faster.','被拒绝的请求腾出了显存，其他请求就跑得更快。'),
      tx('A 429 makes the GPU switch to a faster mode.','回 429 会让卡切换到更快的模式。')],a:0,
   why:tx('Under overload the backlog grows by the extra rate every second, and everyone behind it waits for it. Capping the queue turns “everyone late” into “a few told to retry, the rest on time”.','过载时，排队每秒都多出那一部分，排在后面的人都要等它。给排队设上限，就把“所有人都超时”变成“少数人被告知稍后再试，其余的按时”。'),
   n:[null,tx('Waiting requests sit in the gateway, not in GPU memory; the GPUs run at the same speed.','排队的请求在网关里，不占显存；卡的速度不变。'),tx('A 429 changes nothing on the GPU; it only tells the client when to come back.','429 不改变卡上的任何东西，只是告诉客户端什么时候再来。')]},
  {q:tx('Why is preempting a batch job cheap for the batch job?','为什么抢占对批处理任务来说代价不大？'),
   o:[tx('It keeps the tokens it wrote and only recomputes its KV cache later; its deadline is hours away, so the delay does not matter.','它写出的 token 都保留，之后只是重算 KV cache；它的期限在几个小时以后，晚一点没关系。'),
      tx('Preempted jobs are dropped and the client resubmits them.','被抢占的任务直接丢掉，由客户端重新提交。'),
      tx('The KV cache is copied to disk and back at no cost.','KV cache 免费地拷到磁盘再拷回来。')],a:0,
   why:tx('Recompute costs GPU time (the prompt and the written tokens go through prefill again) but loses nothing, and batch work has slack to absorb it.','重算要花 GPU 时间（prompt 和已经写出的 token 再做一次 prefill），但什么都不丢，而批处理有足够的余地。'),
   n:[null,tx('vLLM puts a preempted request back in the waiting queue and recomputes it; nothing is resubmitted.','vLLM 把被抢占的请求放回等待队列再重算，不需要重新提交。'),tx('Swapping to CPU memory is another mode with its own cost; vLLM V1 recomputes by default.','换到 CPU 内存是另一种做法，也有代价；vLLM V1 默认用重算。')]}],
 transfer:[{q:tx(`Next quarter the peak grows from 150 to 200 chat requests per second, with the same mix. One GPU’s goodput is ${g} requests per second. About how many GPUs does the peak need, before any rules?`,`下个季度高峰从每秒 150 个聊天请求涨到 200 个，构成不变。一张卡的达标吞吐是每秒 ${g} 个。不加任何规则，高峰大约要几张卡？`),
   o:[tx(`${Math.ceil(200/g)}: 200 ÷ ${g} ≈ ${f1(200/g)}, rounded up`,`${Math.ceil(200/g)} 张：200 ÷ ${g} ≈ ${f1(200/g)}，向上取整`),tx(`${nAv}: the day averages about ${fint(av2)} requests a second, and ${fint(av2)} ÷ ${g} ≈ ${f1(Math.round(av2)/g)}`,`${nAv} 张：全天平均每秒约 ${fint(av2)} 个，${fint(av2)} ÷ ${g} ≈ ${f1(Math.round(av2)/g)}`),tx(`${Math.floor(200/g)}: 200 ÷ ${g}, rounded down`,`${Math.floor(200/g)} 张：200 ÷ ${g}，向下取整`)],a:0,
   why:tx('Divide the peak by the goodput per GPU and round up; rounding down leaves the peak above capacity, and the queue then grows without end.','用高峰除以每张卡的达标吞吐，再向上取整；向下取整会让高峰超过容量，排队就会一直涨。'),
   n:[null,tx('The average hides the afternoon: sized for it, the peak hours are over capacity.','平均值把下午藏起来了：按它配，高峰的几个小时都会超过容量。'),tx(`${Math.floor(200/g)} × ${g} = ${Math.floor(200/g)*g} is below 200: the peak would be over capacity.`,`${Math.floor(200/g)} × ${g} = ${Math.floor(200/g)*g}，比 200 少：高峰会超过容量。`)]},
  {q:tx('A free user’s script suddenly sends 100 requests a second. With a rate limit of 60 a minute per key, how much of that reaches the GPUs?','一个免费用户的脚本突然每秒发 100 个请求。每个 key 每分钟限 60 个时，到达卡上的有多少？'),
   o:[tx('About 1 a second once its bucket is empty; the rest get 429','桶空了以后每秒约 1 个，其余回 429'),tx('All of it, but slower','全部，只是慢一点'),tx('None: the key is blocked','一个都没有：这个 key 被封了')],a:0,
   why:tx('A token bucket refills at the limit’s rate: 60 a minute is one a second. Once the tokens saved in the bucket are used up, the script gets one request a second through.','令牌桶按限额的速度补充，每分钟 60 个就是每秒 1 个。桶里攒下的令牌用完以后，脚本每秒只能过 1 个。'),
   n:[null,tx('Requests over the limit are answered with 429 at once; they do not wait.','超过限额的请求马上回 429，不会排队。'),tx('The limit slows the key down; it does not block it.','限流只是让这个 key 慢下来，不会封掉它。')]}],
 practice:[{id:'inference-serving',d:tx('Tiered quotas, admission control under load and preemption.','分档的配额、过载时的准入控制和抢占。')},{id:'inference-batching',d:tx('Batched GPU serving with a latency target.','带延迟目标的 GPU 批处理服务。')},
  {id:'design-doc-review',d:tx('Review a design for a batched inference service: batching, load balancing, admission.','评审一个批处理推理服务的设计：批处理、负载均衡、准入控制。')}]};};

/* ---------- copy: header ---------- */
const DESCS=()=>({guided:tx('8 parts. First one GPU near its limit, then how the gateway decides who goes first and who waits in front of a fleet.','8 个部分。先看一张接近上限的卡，再看网关怎样决定一组卡前面谁先走、谁等。'),
  sandbox:tx('Five common setups for the same day of traffic; every setting is open.','同一天流量的五种常见配法；所有设置都能改。'),
  challenge:tx('A bigger day: meet every goal on the fewest GPU hours.','更大的一天，全部达标，GPU 小时还要最少。')});
const OUTCOME=()=>tx('work out how many GPUs a day of traffic needs from its peak and the goodput per GPU; when there are too few, decide who waits with queue limits, rate limits, priority and preemption; and run batch work in the quiet hours.','按高峰和每张卡的达标吞吐算出要几张卡；卡不够时，用排队上限、限流、优先级和抢占决定谁等；把批处理放到流量低的时候跑。');
function FIDPOP(){const mg=labName('multi-gpu-serving','Multi-GPU serving','多卡推理服务');
  return tx(`<p><b>Toy run.</b> A seeded simulator of a fleet of H100 SXM GPUs, each holding one copy of ${M8B} in BF16 (no tensor parallelism and no traffic between GPUs; ${mg} covers serving one model on several GPUs). Every step is priced by the step model of the other inference labs, the slower of its arithmetic and its memory traffic; this lab assumes 50% of the dense peak (989.5 TFLOP/s) and 60% of the 3.35 TB/s bandwidth. Each GPU runs continuous batching with chunked prefill: at most 2,048 tokens per step and 1,024 running requests, and room for 450,907 tokens of KV cache (90% of 80 GiB, 2 GiB overhead). A request sets aside room for its prompt and its whole answer when it starts.</p><p>The traffic, the goals of free chats and the batch jobs are this lab’s own, and it assumes $${PRICE} per GPU hour. Requests arrive at random (Poisson, seeded): chats have prompts of 400 to 1,600 tokens and answers of 120 to 320; batch jobs 8,000 to 16,000 and 300 to 800. The busiest minute (14:00) and two minutes of the quietest hour (3:00) are simulated step by step; the other hours are priced from their average rate and the measured goodput per GPU, as a share of the fleet. The 14:00 run has chats only; in the day model, an hour whose busy share is below θ also runs some batch work, which that run leaves out. A turned-away request does not retry. The gateway hands a request to whichever GPU frees room first. This lab preempts only when a chat is admitted and memory is short (a simplification); vLLM’s docs describe preempting requests whenever the KV cache cannot hold every request in the batch. Left out: network and routing time, the time to start or stop GPUs, and requests that change length.</p><p>Sources: ${cite(SRC.sched,'vLLM scheduler config')} (policies fcfs and priority; queue limit with HTTP 503), ${cite(SRC.args,'vLLM engine defaults')} (8,192 tokens per step and 1,024 requests on an H100 with <code>vllm serve</code>; this lab uses 2,048 tokens to keep gaps short), ${cite(SRC.opt,'vLLM optimization docs')} (preemption by recompute), ${cite(SRC.sgl,'SGLang server arguments')}, ${cite(SRC.fc,'llm-d flow control')} (priority bands, 429, eviction), ${cite(SRC.batch,'OpenAI Batch API guide')}, ${cite(SRC.rl,'OpenAI cookbook: rate limits')}.</p>`,
   `<p><b>玩具实跑。</b>一个带种子的模拟器：一组 H100 SXM，每张卡放一份 ${M8B}（BF16）（不做张量并行，卡之间没有通信；一个模型放在几张卡上怎么跑，见 ${mg}）。每一步的耗时用其他推理 Lab 的步模型算，取算力和带宽两者中慢的那个；本 Lab 假设算力用到稠密峰值（989.5 TFLOP/s）的 50%，带宽用到 3.35 TB/s 的 60%。每张卡跑 continuous batching 加 chunked prefill：每一步最多 2,048 个 token、最多 1,024 个请求同时跑，KV cache 能放 450,907 个 token（80 GiB 的 90%，减去 2 GiB 运行时开销）。请求开始时就为它的 prompt 和整段回答留好空间。</p><p>流量、免费聊天的目标和批处理任务都是本 Lab 自己定的；本 Lab 假设每 GPU 小时 ${PRICE} 美元。请求随机到达（泊松分布，带种子）：聊天的 prompt 400 到 1,600 个 token、回答 120 到 320 个；批处理任务 8,000 到 16,000 个和 300 到 800 个。最忙的一分钟（下午 2 点）和最闲那个小时（凌晨 3 点）里的两分钟逐步模拟；其他小时按平均速率和量出来的每张卡的达标吞吐，算出占了卡的几成。下午 2 点的运行里只有聊天请求；日模型里，忙碌程度低于 θ 的小时也会跑一点批处理，这次运行没有算进去。被拒绝的请求不再重试。网关把请求交给最先空出位置的卡。这个 Lab 只在聊天请求进卡、显存不够时抢占（一种简化）；vLLM 的文档说的是，KV cache 放不下这一批所有请求时就会抢占一部分请求。没有算进去的：网络和路由的时间、开关卡的时间、长度会变的请求。</p><p>来源：${cite(SRC.sched,'vLLM 调度配置')}（fcfs 和 priority 两种策略；排队上限回 HTTP 503）、${cite(SRC.args,'vLLM 引擎默认值')}（H100 上用 <code>vllm serve</code> 启动时，每一步 8,192 个 token、1,024 个请求；这个 Lab 用 2,048 个 token，让间隔保持很短）、${cite(SRC.opt,'vLLM 优化文档')}（用重算的方式抢占）、${cite(SRC.sgl,'SGLang 启动参数')}、${cite(SRC.fc,'llm-d flow control')}（按优先级分档、429、停掉负优先级请求）、${cite(SRC.batch,'OpenAI Batch API 文档')}、${cite(SRC.rl,'OpenAI cookbook：rate limits')}。</p>`);}
const PRIPOP=()=>tx(`<p>A GPU serves many requests at once with continuous batching (${labLink('continuous-batching')}), and a request waits for its first token while others are in prefill (${labLink('prefill-decode')}). The goals for chats come from ${labLink('serving-capstone')}; goodput per GPU from ${labLink('pd-disagg')}. KV cache memory is shared in blocks (${labLink('paged-attention')}).</p>`,
  `<p>一张卡用 continuous batching 同时服务很多请求（${labLink('continuous-batching')}）；别的请求在 prefill 时，新请求要等首 token（${labLink('prefill-decode')}）。聊天的目标来自${labLink('serving-capstone')}，每张卡的达标吞吐来自 ${labLink('pd-disagg')}。KV cache 的显存按 block 分给各个请求（${labLink('paged-attention')}）。</p>`);

/* ---------- views ---------- */
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Capacity capstone','容量规划综合挑战'),title:l?L(l.q):tx('How many GPUs does your traffic need, and who should wait?','你的流量要几张卡，谁该排队？'),
    minutes:20,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: serving one GPU','前置知识：一张卡怎么服务请求'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const spdCtl=()=>`<div class="adm-speed"><span class="adm-speed-l">${tx('Speed','速度')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="spd">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${spdNow()===v}">${x}</button>`).join('')}</div></div>`;
const lg=(tip,cls,t)=>`<span data-tipx="${esc(tip)}"><i class="adm-sw ${cls}"></i>${t}</span>`;
function stageHTML(){
  return `<div class="panel stage adm-stage" id="adm-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="adm-st-title"></div><div class="meter-sub" id="adm-st-sub"></div></div><div class="stage-actions"><span class="status" id="adm-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph adm-graph" id="adm-graph">`+
    `<div class="adm-blk" id="adm-dayb"><div class="adm-bh" id="adm-day-h"></div><div class="adm-day" id="adm-day"></div><div class="legend-row adm-legend" id="adm-day-lg"></div></div>`+
    `<div class="adm-blk"><div class="adm-bh" id="adm-q-h"></div><div class="adm-q" id="adm-q"></div><div class="legend-row adm-legend" id="adm-q-lg"></div>`+
    `<div class="adm-run" id="adm-run"><button type="button" class="btn sm" data-act="play" id="adm-play"></button><button type="button" class="btn ghost sm" data-act="replay" id="adm-replay">${tx('Replay','重放')}</button>${spdCtl()}<input type="range" class="adm-prog" id="adm-prog" min="0" max="1000" value="1000" aria-label="${esc(tx('Position in the run','播放位置'))}"></div></div>`+
    `<div class="adm-blk"><div class="adm-bh">${tx('Who met their goals in this run','这次运行里谁达标了')}</div><div class="adm-goals" id="adm-goals"></div></div>`+
   `</div>`+
   `<div class="metrics adm-metrics" id="adm-metrics"></div>`+
   `<details class="math" id="adm-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="adm-mathbody"></div></details></div>`;}
function controlsHTML(){
  const seg=(id,key,label,opts,v,note)=>A.ui.seg({id:'adm-ctl-'+id,key,label,options:opts,value:String(v),note});
  return `<aside class="panel controls" id="adm-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="adm-lock" hidden></p><div id="adm-ctls">`+
   `<p class="ctl-hint small muted">${tx('Watch the red hours in the top chart of the day (traffic above capacity) and the queue lines in the busiest minute below it.','看上面一天流量图里的红色小时（流量超过容量），和下面那一分钟里排队的线。')}</p>`+
   A.ui.range({id:'adm-ctl-n',key:'n',label:tx('GPUs, all day','卡数（全天）'),min:3,max:8,step:1,value:S.n,out:String(S.n),note:tx(`One copy of ${M8B} per H100. Each GPU takes about ${G()} chat requests per second within the paid goals (measured).`,`每张 H100 一份 ${M8B}。每张卡每秒约能接 ${G()} 个聊天请求并达到付费的目标（量出来的）。`)})+
   seg('q','q',tx('Free requests that may wait','免费请求最多排几个'),[['0',tx('no limit','不限')],['500','500'],['200','200'],['50','50']],S.q,tx('Beyond the limit a free request gets HTTP 429 with Retry-After at once. Paid requests are never refused.','超过上限的免费请求马上拿到 HTTP 429 和 Retry-After。付费请求从不拒绝。'))+
   seg('rl','rl',tx('Rate limit per API key','每个 API key 限流'),[['0',tx('off','不限')],['60',tx('60 a minute','每分钟 60 个')]],S.rl,tx(`A token bucket per key. Only the script’s key (${BASE_T.script} requests a second, this lab’s assumption) ever reaches it; ordinary users send far less.`,`每个 key 一个令牌桶。只有脚本那个 key（每秒 ${BASE_T.script} 个，这个 Lab 的假设）会碰到它；普通用户发得少得多。`))+
   seg('prio','prio',tx('Who goes first','谁先走'),[['0',tx('first come','先来先服务')],['1',tx('paid first, with preemption','付费优先，可以抢占')]],S.prio,tx('Paid first also lets a chat preempt the batch job that started latest when memory is full.','付费优先时，显存满了，聊天请求还可以抢占最晚开始的批处理任务。'))+
   seg('th','th',tx('Batch threshold θ','批处理门槛 θ'),[['60','60%'],['80','80%'],['100','100%']],S.th,tx(`Batch jobs run in hours when chats keep the fleet less than θ busy, up to θ. ${fJobs(BASE_JOBS)} jobs a day are handed in at midnight (this lab’s assumption).`,`聊天让卡忙不到 θ 的小时里跑批处理，跑到 θ 为止。每天半夜交上来 ${fJobs(BASE_JOBS)} 个任务（这个 Lab 的假设）。`))+
   `<p class="ctl-note">${tx(`Traffic at 14:00: ${BASE_T.paid} paid and ${BASE_T.free} free chat requests per second, plus the script; the other hours follow the day curve (this lab’s own).`,`下午 2 点的流量：每秒 ${BASE_T.paid} 个付费、${BASE_T.free} 个免费聊天请求，加上脚本；其他小时按日曲线变化（这个 Lab 自己定的）。`)}</p>`+
   `<button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`# A gateway: priority queues, a queue limit, a token bucket
import collections, time

class Bucket:                      # per API key
    def __init__(self, per_min):
        self.cap = self.tok = per_min
        self.rate, self.t = per_min / 60, time.monotonic()
    def take(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        self.tok = min(self.cap, self.tok + dt * self.rate)
        if self.tok >= 1:
            self.tok -= 1
            return True
        return False

class Gateway:
    ORDER = ("paid", "free", "batch")
    def __init__(self, free_limit=200, per_min=60):
        self.q = {c: collections.deque() for c in self.ORDER}
        self.free_limit, self.per_min = free_limit, per_min
        self.buckets = {}
    def submit(self, req):          # -> "queued" or an HTTP error
        b = self.buckets.setdefault(req["key"], Bucket(self.per_min))
        if not b.take():
            return "429 rate limit, Retry-After: 1"
        full = len(self.q["free"]) >= self.free_limit
        if req["cls"] == "free" and full:
            return "429 queue full, Retry-After: 30"
        self.q[req["cls"]].append(req)
        return "queued"
    def next(self):                 # a GPU has room: who goes first?
        for c in self.ORDER:
            if self.q[c]:
                return self.q[c].popleft()

gw = Gateway(free_limit=2)
for i in range(4):
    print(gw.submit({"key": "app", "cls": "free", "id": i}))
print(gw.submit({"key": "vip", "cls": "paid", "id": 9}))
print(gw.next()["id"])`;
const CODE_OUT=`# queued
# queued
# 429 queue full, Retry-After: 30
# 429 queue full, Retry-After: 30
# queued
# 9`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="adm-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="adm-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The admission side of a gateway: a token bucket per key, a limit on waiting free requests, and paid requests taken first.','网关里管准入的那部分：每个 key 一个令牌桶，免费请求排队有上限，付费请求先取。')}</p><pre class="code" id="adm-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="adm-prod">`+
   tx(`<li>vLLM: ${cite(SRC.sched,'<code>vllm/config/scheduler.py</code>')}. <code>policy</code> is <code>"fcfs"</code> by default; <code>"priority"</code> serves lower priority values first, ties by arrival. <code>max_num_queued_reqs</code> rejects new requests with HTTP 503 once that many are in flight. When KV cache runs out, requests are preempted and recomputed (${cite(SRC.opt,'<code>docs/configuration/optimization.md</code>')}).</li><li>SGLang: ${cite(SRC.sgl,'<code>server_arguments.mdx</code>')}. <code>--enable-priority-scheduling</code>, <code>--max-queued-requests</code>, and <code>--priority-scheduling-preemption-threshold</code> (default 10): how much higher a new request’s priority must be to preempt running ones.</li><li>llm-d: ${cite(SRC.fc,'<code>flow-control.md</code>')}. Requests wait at the gateway in priority bands; a request that would go over the queue limits is rejected at once with HTTP 429. With eviction on, a higher band that is blocked can end in-flight requests of negative priority: the lowest priority first, then the most recently dispatched. Priority per workload: ${cite(SRC.io,'<code>InferenceObjective</code>')}.</li><li>OpenAI: ${cite(SRC.batch,'Batch API guide')} (completion window 24h, 50% lower cost) and ${cite(SRC.rl,'how to handle rate limits')} (429, retry with exponential backoff).</li>`,
      `<li>vLLM：${cite(SRC.sched,'<code>vllm/config/scheduler.py</code>')}。<code>policy</code> 默认是 <code>"fcfs"</code>；<code>"priority"</code> 先服务 priority 值小的，相同时按到达先后。<code>max_num_queued_reqs</code>：在途的请求到了这个数，新请求回 HTTP 503。KV cache 不够时，请求被抢占、之后重算（${cite(SRC.opt,'<code>docs/configuration/optimization.md</code>')}）。</li><li>SGLang：${cite(SRC.sgl,'<code>server_arguments.mdx</code>')}。<code>--enable-priority-scheduling</code>、<code>--max-queued-requests</code>，以及 <code>--priority-scheduling-preemption-threshold</code>（默认 10）：新请求的优先级要高出多少，才能抢占正在跑的请求。</li><li>llm-d：${cite(SRC.fc,'<code>flow-control.md</code>')}。请求在网关按优先级分档排队；超过排队上限的请求马上回 HTTP 429。打开 eviction 后，被挡住的高优先级档可以停掉负优先级的在途请求：优先级最低的先停，同样低的先停最晚派出去的。每类负载的优先级在 ${cite(SRC.io,'<code>InferenceObjective</code>')} 里设。</li><li>OpenAI：${cite(SRC.batch,'Batch API 文档')}（完成窗口 24h，价格低 50%）和 ${cite(SRC.rl,'怎么处理限流')}（429，按指数退避重试）。</li>`)+`</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.p_dist,'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving')}<span>${tx('Zhong et al., 2024, arXiv 2401.09670 (goodput)','Zhong 等，2024，arXiv 2401.09670（达标吞吐）')}</span></li>`+
   `<li>${cite(SRC.p_vllm,'Efficient Memory Management for Large Language Model Serving with PagedAttention')}<span>${tx('Kwon et al., 2023, arXiv 2309.06180 (preemption by recompute or swap)','Kwon 等，2023，arXiv 2309.06180（用重算或换出来抢占）')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="adm-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="adm-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="adm-app m2" id="adm-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="adm-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- the stage ---------- */
const runT=run=>run.cfg.window||run.m.end||1;
const niceStep=T=>{for(const s of [5,10,15,20,30,60,120,180,300]) if(T/s<=6) return s; return 600;};
const niceMax=x=>{if(x<=1) return 1; const p=Math.pow(10,Math.floor(Math.log10(x))); for(const k of [1,1.5,2,2.5,3,4,5,6,8,10]) if(k*p>=x) return k*p; return 10*p;};
const clsIn=run=>({paid:run.R.some(r=>r.cls==='paid'),free:run.R.some(r=>r.cls==='free'),batch:run.R.some(r=>r.cls==='batch')});
const scriptOf=(T,rl)=>rl?Math.min(T.script,rl/60):T.script;
function drawDay(v){
  const blk=$('#adm-dayb',app), host=$('#adm-day',app); if(!blk) return; blk.hidden=!v.day; if(!v.day){host.innerHTML=''; return;}
  const {T,s}=v.day, d=ADM.day(T,cfgOf(s)), W=Math.max(260,Math.round(host.clientWidth||600)), H=W<520?132:150, LW=W<520?30:40, top=14, bot=22, pw=W-LW-6, ph=H-top-bot, bw=pw/24;
  const ymax=niceMax(Math.max(d.cap*1.08,...d.hours.map(x=>x.c),...d.hours.map(x=>x.c+x.sh*d.cap))), Y=y=>top+ph*(1-y/ymax), X=h=>LW+h*bw;
  const hl=v.kind==='night'?ADM.quietH():ADM.PEAK_H, out=[];
  out.push(`<defs><pattern id="adm-hatch" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)"><rect width="5" height="5" class="adm-hbg"/><line x1="0" y1="0" x2="0" y2="5" class="adm-hln"/></pattern></defs>`);
  out.push(`<rect class="adm-hl" x="${(X(hl)-1).toFixed(1)}" y="${top-4}" width="${(bw+2).toFixed(1)}" height="${ph+4}"/>`);
  d.hours.forEach(x=>{const h=x.h, bx=X(h)+bw*0.12, w=bw*0.76, pa=ADM.PROF[h]*T.paid, fr=ADM.PROF[h]*T.free, sc=scriptOf(T,s.rl);
    let y0=0; const seg=(val,c)=>{if(val<=0) return; out.push(`<rect class="${c}" x="${bx.toFixed(1)}" y="${Y(y0+val).toFixed(1)}" width="${w.toFixed(1)}" height="${(Y(y0)-Y(y0+val)).toFixed(1)}"/>`); y0+=val;};
    seg(pa,'adm-pa'); seg(fr,'adm-fr'); seg(sc,'adm-sc');
    if(x.c>d.cap) out.push(`<rect class="adm-over" x="${bx.toFixed(1)}" y="${Y(x.c).toFixed(1)}" width="${w.toFixed(1)}" height="${(Y(d.cap)-Y(x.c)).toFixed(1)}"/>`);
    if(x.sh>0) out.push(`<rect class="adm-bt" x="${bx.toFixed(1)}" y="${Y(x.c+x.sh*d.cap).toFixed(1)}" width="${w.toFixed(1)}" height="${(Y(x.c)-Y(x.c+x.sh*d.cap)).toFixed(1)}" fill="url(#adm-hatch)"/>`);});
  out.push(`<line class="adm-cap" x1="${LW}" x2="${LW+pw}" y1="${Y(d.cap).toFixed(1)}" y2="${Y(d.cap).toFixed(1)}"/>`);
  for(const h of [0,6,12,18]) out.push(`<text class="adm-at" x="${(X(h)+bw/2).toFixed(1)}" y="${H-6}" text-anchor="${h===0?'start':'middle'}">${fHour(h)}</text>`);
  out.push(`<text class="adm-at" x="${(LW-4)}" y="${(Y(ymax)+4).toFixed(1)}" text-anchor="end">${fint(ymax)}</text><text class="adm-at" x="${LW-4}" y="${(Y(0)).toFixed(1)}" text-anchor="end">0</text>`);
  out.push(`<line class="adm-ax" x1="${LW}" x2="${LW+pw}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}"/>`);
  host.innerHTML=`<svg class="adm-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(tx(`Chat requests per second for each hour of the day, the capacity of ${s.n} GPUs, ${d.over} hours above it`,`一天每个小时每秒的聊天请求数，${s.n} 张卡的容量，有 ${d.over} 个小时超过容量`))}">${out.join('')}</svg>`;
  $('#adm-day-h',app).textContent=tx(`A day of chat traffic, requests per second each hour (this lab’s own curve); outlined: ${fHour(hl)}, the run below`,`一天的聊天流量，每个小时每秒多少个请求（这个 Lab 自己定的曲线）；框出来的是${hl===ADM.PEAK_H?'下午 2 点':`凌晨 ${hl} 点`}，就是下面那次运行`);
  const lgd=[lg(TIP.ttft(),'pa',tx('paid','付费')),lg(TIP.slo(),'fr',tx('free','免费')),lg(TIP.script(),'sc',tx('the script','脚本')),lg(TIP.cap(),'over',tx('above capacity','超过容量')),lg(TIP.th(),'bt',tx('batch jobs','批处理')),lg(TIP.cap(),'cap',tx(`capacity: ${fint(d.cap)} requests/s`,`容量：每秒 ${fint(d.cap)} 个`))];
  const lh=lgd.join(''), le=$('#adm-day-lg',app); if(le._h!==lh){le.innerHTML=lh; le._h=lh;}}
function missBins(run,T,nb){const a=new Array(nb).fill(0), b=new Array(nb).fill(0), g=ADM.GOALS.paid;
  for(const r of run.R){if(r.cls!=='paid'||r.arr>T) continue; const i=Math.min(nb-1,Math.floor(r.arr/T*nb)); a[i]++; if(r.first==null||r.first-r.arr>g.ttft+1e-9||ADM.gapOf(r)>g.gap+1e-9) b[i]++;}
  return a.map((x,i)=>x&&b[i]/x>0.1);}
function drawQueue(run,v){
  const host=$('#adm-q',app); if(!host) return; const W=Math.max(260,Math.round(host.clientWidth||600)), H=W<520?132:148, LW=W<520?36:46, top=8, bot=34, T=runT(run), X=t=>LW+(W-LW-8)*Math.min(t,T)/T;
  const has=clsIn(run), S_=run.samples, ser=[['paid',1,'adm-lp'],['free',2,'adm-lf'],['batch',3,'adm-lb']].filter(x=>has[x[0]]);
  const ymax=niceMax(Math.max(5,...S_.map(x=>Math.max(...ser.map(s=>x[s[1]]))))), ph=H-top-bot, Y=y=>top+ph*(1-y/ymax);
  const body=[];
  ser.forEach(([c,i,k])=>{body.push(`<polyline class="${k}" points="${S_.map(x=>`${X(x[0]).toFixed(1)},${Y(x[i]).toFixed(1)}`).join(' ')}"/>`);});
  if(has.paid){const nb=60, mb=missBins(run,T,nb), bw=(W-LW-8)/nb; mb.forEach((m,i)=>{if(m) body.push(`<rect class="adm-miss" x="${(LW+i*bw).toFixed(1)}" y="${H-bot+4}" width="${Math.max(1,bw-0.5).toFixed(1)}" height="5"/>`);});}
  const st=niceStep(T), ax=[]; for(let t=0;t<=T+1e-9;t+=st){const x=X(t), e=x>W-24; ax.push(`<line class="adm-ax" x1="${x.toFixed(1)}" y1="${H-bot+11}" x2="${x.toFixed(1)}" y2="${H-bot+14}"/><text class="adm-at" x="${(e?W-1:x).toFixed(1)}" y="${H-6}" text-anchor="${e?'end':t===0?'start':'middle'}">${T>=180?(zh()?`${t/60} 分`:`${t/60} min`):(zh()?`${t} 秒`:`${t} s`)}</text>`);}
  ax.push(`<text class="adm-at" x="${LW-4}" y="${(Y(ymax)+4).toFixed(1)}" text-anchor="end">${fint(ymax)}</text><text class="adm-at" x="${LW-4}" y="${Y(0).toFixed(1)}" text-anchor="end">0</text><line class="adm-ax" x1="${LW}" x2="${W-8}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}"/>`);
  const p=Math.min(phNow,T), cw=Math.max(0,X(p)-LW);
  host.innerHTML=`<svg class="adm-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc($('#adm-q-h',app).textContent)}"><defs><clipPath id="adm-clip"><rect id="adm-cliprect" x="${LW-2}" y="0" width="${(cw+2).toFixed(1)}" height="${H}"/></clipPath></defs>${ax.join('')}<g clip-path="url(#adm-clip)">${body.join('')}</g><line class="adm-ph" id="adm-ph" x1="${X(p).toFixed(1)}" x2="${X(p).toFixed(1)}" y1="0" y2="${H-bot+10}"${PLAY.on?'':' visibility="hidden"'}/></svg>`;
  host._X={LW,W,T};
  const lgd=[]; if(has.paid) lgd.push(lg(TIP.prio(),'lp',tx('paid waiting','排队的付费请求'))); if(has.free) lgd.push(lg(TIP.qlim(),'lf',tx('free waiting','排队的免费请求'))); if(has.batch) lgd.push(lg(TIP.batch(),'lb',tx('batch jobs waiting','排队的批处理任务')));
  if(has.paid) lgd.push(lg(TIP.goals(),'miss',tx('arriving paid requests: more than 1 in 10 missed','这段时间到的付费请求，超过一成没达标')));
  const lh=lgd.join(''), le=$('#adm-q-lg',app); if(le._h!==lh){le.innerHTML=lh; le._h=lh;}}
function moveHead(run){const host=$('#adm-q',app); if(!host||!host._X) return; const {LW,W,T}=host._X, X=t=>LW+(W-LW-8)*Math.min(t,T)/T, p=Math.min(phNow,T);
  const cr=$('#adm-cliprect',app), pl=$('#adm-ph',app); if(cr) cr.setAttribute('width',(Math.max(0,X(p)-LW)+2).toFixed(1)); if(pl){pl.setAttribute('x1',X(p).toFixed(1)); pl.setAttribute('x2',X(p).toFixed(1)); pl.setAttribute('visibility',PLAY.on?'visible':'hidden');}}
function qHead(v,run){const s=v.s;
  if(v.kind==='one') return tx(`Paid chats waiting for 1 H100 at ${fRate(Math.round(G()*v.f*10)/10)} requests per second (${Math.round(v.f*100)}% of its goodput), 60 seconds`,`1 张 H100，每秒 ${fRate(Math.round(G()*v.f*10)/10)} 个付费聊天请求（达标吞吐的 ${Math.round(v.f*100)}%），60 秒里排队的请求`);
  if(v.kind==='long'){const x=longRates(); return tx(`Paid chats waiting for 1 H100: ${fRate(x.hi)} per second for 10 minutes, then ${fRate(x.lo)} for 5`,`1 张 H100 前排队的付费请求：每秒 ${fRate(x.hi)} 个持续 10 分钟，之后每秒 ${fRate(x.lo)} 个持续 5 分钟`);}
  if(v.kind==='surge') return tx(`Requests waiting at the gateway of 4 H100s: ${SURGE.backlog} batch jobs at the start, the peak from ${SURGE.quiet} s`,`4 张 H100 的网关里排队的请求：开始时 ${SURGE.backlog} 个批处理任务，${SURGE.quiet} 秒起是高峰流量`);
  if(v.kind==='night') return tx(`Requests waiting at the gateway during 2 minutes at 3:00, ${s.n} H100s`,`凌晨 3 点的 2 分钟里，${s.n} 张 H100 的网关里排队的请求`);
  return tx(`Requests waiting at the gateway during the busiest minute (14:00), ${s.n} H100s`,`最忙的一分钟（下午 2 点），${s.n} 张 H100 的网关里排队的请求`);}
const RULE={fcfs:{en:'first come, first served',zh:'先来先服务'},prio:{en:'paid first, no preemption',zh:'付费优先，不抢占'},pre:{en:'paid first, with preemption',zh:'付费优先，可以抢占'}};
function whatOf(v){
  if(v.kind==='one') return tx(`1 H100, ${Math.round(v.f*100)}% of its goodput`,`1 张 H100，达标吞吐的 ${Math.round(v.f*100)}%`);
  if(v.kind==='long') return tx('1 H100, 20% over its goodput for 10 minutes','1 张 H100，超出达标吞吐 20%，持续 10 分钟');
  if(v.kind==='surge') return tx(`4 H100s, a surge onto ${SURGE.backlog} batch jobs, ${L(RULE[v.k])}`,`4 张 H100，高峰撞上 ${SURGE.backlog} 个批处理任务，${L(RULE[v.k])}`);
  if(v.kind==='night') return tx(`3:00 on ${setupName(v.s)}`,`凌晨 3 点，${setupName(v.s)}`);
  return tx(`14:00 on ${setupName(v.s,v.lvl)}`,`下午 2 点，${setupName(v.s,v.lvl)}`);}
function goalsHTML(run,v){const m=run.m, play=PLAY.on, rows=[];
  const row=(lab,frac,tgt,txt,cls)=>`<div class="adm-gr${cls?' '+cls:''}"><span class="adm-gl">${lab}</span><span class="adm-gb"><i style="width:${play?0:Math.round(Math.min(1,frac)*100)}%"></i><b style="left:${Math.round(tgt*100)}%"></b></span><span class="adm-gv mono">${play?'—':txt}</span></div>`;
  const okc=(f,t)=>f>=t-1e-9?'ok':'bad';
  if(m.paid) rows.push(row(DEF(tx('Paid chats','付费聊天'),TIP.goals()),m.paid.frac,ADM.TARGET.paid,tx(`${fP(m.paid.frac)} (target 90%)`,`${fP(m.paid.frac)}（目标 90%）`),okc(m.paid.frac,ADM.TARGET.paid)));
  if(m.free) rows.push(row(DEF(tx('Free chats','免费聊天'),TIP.goals()),m.free.frac,ADM.TARGET.free,tx(`${fP(m.free.frac)} (target 80%)`,`${fP(m.free.frac)}（目标 80%）`),okc(m.free.frac,ADM.TARGET.free)));
  if(v.day&&v.lvl!=='admit'){const d=ADM.day(v.day.T,cfgOf(v.day.s)), dm=v.day.jobs, b=batchOf(d,dm);
    rows.push(row(DEF(tx('Batch jobs done in 24 h','24 小时做完的批处理'),TIP.batch()),b.done/dm,1,tx(`${fJobs(b.done)} of ${fJobs(dm)}`,`${fJobs(b.done)} / ${fJobs(dm)}`),d.jobs>=dm-1e-6?'ok':'bad'));}
  if(!play&&overOf(v)&&ADM.passes(m)) rows.push(`<p class="adm-gnote">${tx('The afternoon traffic is above these GPUs’ capacity, and free requests have no queue limit. This minute starts from an empty queue and does not show it yet, but over an afternoon of peak traffic the queue keeps growing.','下午的流量比这几张卡的容量多，免费请求的排队又没有上限。这一分钟从空队列开始，还看不出来；高峰持续一个下午，排队会一直涨。')}</p>`);
  return rows.join('');}
function metricsDef(run,v){
  const m=run.m, play=PLAY.on, w=x=>play?'—':x, out=[];
  out.push({k:'pp90',label:tx('Paid: first token (p90)','付费：首 token（p90）'),v:w(fT(m.paid?m.paid.p90:null)),sub:tx('goal: within 1 s','目标：1 秒内'),tip:TIP.p90()});
  if(v.kind==='one'||v.kind==='long'){
    const qm=run.samples.reduce((a,x)=>Math.max(a,x[1]),0);
    out.push({k:'wmax',label:tx('Longest wait for a first token','等首 token 最久的'),v:w(fT(m.paid.wmax)),sub:tx('one request','某一个请求'),tip:TIP.ttft()});
    out.push({k:'qmax',label:tx('Most requests waiting','排队最多时'),v:w(fint(qm)),sub:tx('at one moment','同一时刻'),tip:TIP.gateway()});
    out.push({k:'n',label:tx('Requests in the run','这次运行的请求'),v:w(fint(m.paid.n)),sub:tx('seeded random arrivals','带种子的随机到达'),tip:TIP.goodput()});
    return out;}
  if(v.kind==='surge'){
    out.push({k:'fp90',label:tx('Free: first token (p90)','免费：首 token（p90）'),v:w(fT(m.free.p90)),sub:tx('goal: within 5 s','目标：5 秒内'),tip:TIP.p90()});
    out.push({k:'pre',label:tx('Preemptions','抢占次数'),v:w(fint(m.batch.pre)),sub:tx('in this run','这次运行里'),tip:TIP.preempt()});
    out.push({k:'redo',label:tx('Tokens recomputed','重算的 token'),v:w(A.fmt.big(m.batch.redo)),sub:tx('KV cache built again','KV cache 重新算一遍'),tip:TIP.preempt()});
    return out;}
  if(v.kind==='night') out.push({k:'pre',label:tx('Preemptions','抢占次数'),v:w(fint(m.batch.pre)),sub:tx('in these 2 minutes','这 2 分钟里'),tip:TIP.preempt()});
  else out.push({k:'rej',label:tx('Free requests turned away','被拒绝的免费请求'),v:w(fint(m.free.rejq+m.free.rejr)),sub:tx(`of ${fint(m.free.n)} free requests`,`共 ${fint(m.free.n)} 个免费请求`),
    tip:tx(`Turned away for a full queue: ${fint(m.free.rejq)}. Stopped by the rate limit: ${fint(m.free.rejr)}, not counted against the free target.`,`排队满了被拒绝 ${fint(m.free.rejq)} 个；限流挡下 ${fint(m.free.rejr)} 个，不算进免费的目标。`)});
  const d=ADM.day(v.day.T,cfgOf(v.day.s)), gh=v.day.s.n*24;
  if(v.lvl!=='admit'){const b=batchOf(d,v.day.jobs), hi=fJobs(b.sub);
    out.push({k:'jobs',label:tx('Batch capacity in a day (at most)','全天批处理容量（最多能处理）'),v:fJobs(b.cap),
      sub:b.short>0?tx(`${hi} handed in, ${fJobs(b.done)} done`,`交上来 ${hi} 个，做完 ${fJobs(b.done)} 个`):tx(`${hi} handed in, all done${b.spare>=500?`; ${fJobs(b.spare)} spare`:''}`,`交上来 ${hi} 个，全部做完${b.spare>=500?`；余量 ${fJobs(b.spare)} 个`:''}`),tip:TIP.bcap()});}
  out.push({k:'gpuh',label:tx('GPU hours a day','每天的 GPU 小时'),v:String(gh),sub:tx(`$${fint(gh*PRICE)} at $${PRICE} each`,`每 GPU 小时 ${PRICE} 美元，共 ${fint(gh*PRICE)} 美元`),tip:TIP.gpuh()});
  return out;}
const numOf=s=>{const m=String(s).replace(/,/g,'').match(/-?[\d.]+/); if(!m) return NaN; const x=parseFloat(m[0]); return /ms|毫秒/.test(s)?x/1000:x;};
const shownPrev={};
function drawMetrics(run,v){
  const host=$('#adm-metrics',app), defs=metricsDef(run,v), sg=defs.map(m=>m.k).join();
  if(host._sig!==sg){host._sig=sg; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); for(const k in shownPrev) delete shownPrev[k];}
  host.classList.toggle('adm-wait',PLAY.on);
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    if(PLAY.on&&m.v==='—') return;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&prev!=='—'){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,{dp:m.k==='pp90'||m.k==='fp90'||m.k==='wmax'?2:0})||'';}
    clearTimeout(d._t);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on');},2600);} else if(userAct){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});}

/* ---------- Show the math ---------- */
const EQ=rows=>`<div class="adm-eqs">${rows.map(([l,r,rel])=>`<span class="adm-eq-l">${l?M(l):''}</span><span class="adm-eq-e">${M(rel||'=')}</span><span class="adm-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="adm-mb"><p>${lead}</p>${body}${res?`<p class="adm-mb-res">${res}</p>`:''}</div>`;
function mathHTML(){
  const v=curView(), fl=v.day?v.day:{T:BASE_T,s:BASE,jobs:BASE_JOBS}, T=fl.T, s=fl.s, g=G(), pr=peakRate(T,0), d=ADM.day(T,cfgOf(s)), b=ADM.batchRate().per, x=longRates(), qh=ADM.quietH(), u3=d.hours[qh];
  const lr=longRun(), mu=ADM.serveRate(lr,600), mu1=Math.round(mu*10)/10, q10=(lr.samples.find(z=>Math.abs(z[0]-600)<1e-6)||[0,0])[1];
  const sh=Math.round(d.hours.reduce((a,z)=>a+z.sh,0)*1000)/1000, b4=Math.round(b*1e4)/1e4, f3=z=>A.fmt.num(z,3), f4=z=>A.fmt.num(z,4), cs=sh*s.n*b4*3600;
  const t=k=>`\\text{${k}}`, Tx=(en,zhs)=>t(tx(en,zhs)), rps=Tx('requests/s','个/秒');
  return MB(tx('<b>How many GPUs the peak needs.</b> The goodput per GPU is measured on one GPU (the highest whole rate at which 90% of chats meet the paid goals). The fleet’s capacity is the GPUs times that, and the peak divided by it, rounded up, is the GPUs the peak needs:','<b>高峰要几张卡。</b>每张卡的达标吞吐在一张卡上量出来（90% 的聊天达到付费目标时，最高的整数速率）。一组卡的容量是卡数乘以它；高峰除以它再向上取整，就是高峰要的卡数：'),
     EQ([[Tx('capacity','容量'),`${s.n} \\times ${g} = ${fint(s.n*g)}\\ ${rps}`],[Tx('GPUs for the peak','高峰要的卡'),`\\frac{${fRate(pr)}}{${g}} \\approx ${f1(pr/g)} \\to ${Math.ceil(pr/g-1e-9)}`]]),
     tx(`${fRate(pr)} requests per second is this traffic at 14:00 without a rate limit: ${T.paid} paid + ${T.free} free + ${T.script} from the script.`,`每秒 ${fRate(pr)} 个是不限流时下午 2 点的流量：${T.paid} 个付费 + ${T.free} 个免费 + 脚本的 ${T.script} 个。`))+
   MB(tx(`<b>How fast the queue grows.</b> When requests arrive at a rate λ above the rate μ at which the GPU gets through them, the queue grows by the difference every second. With its queue full, one GPU gets through about ${f1(mu)} requests a second, measured on the run below, a little more than the goodput of ${g}:`,`<b>排队涨得多快。</b>请求到达的速率 λ 比卡做完的速率 μ 高时，排队每秒多出两者的差。一张卡排满以后，每秒能做完约 ${f1(mu)} 个请求（用下面这次运行量出来的），比达标吞吐 ${g} 略高：`),
     EQ([[Tx('queue after t','t 秒后的排队'),`(\\lambda - \\mu) \\times t`],['',`(${fRate(x.hi)} - ${f1(mu)}) \\times 600 \\approx ${fint(Math.round((x.hi-mu1)*600))}`]]),
     tx(`The run: one GPU 20% over its goodput for 10 minutes. Its queue at 10 minutes is ${fint(q10)}.`,`这次运行：一张卡超出达标吞吐 20%，持续 10 分钟。到第 10 分钟时，排队是 ${fint(q10)} 个。`))+
   MB(tx('<b>The token bucket.</b> Each key’s bucket holds up to 60 tokens and gets 60 back a minute, that is, 1 a second; a request takes one or gets 429:','<b>令牌桶。</b>每个 key 的桶最多放 60 个令牌，每分钟补回 60 个，也就是每秒 1 个；请求拿走一个，拿不到就回 429：'),
     EQ([[Tx('tokens','令牌'),`\\min(60,\\ ${Tx('tokens','令牌')} + \\Delta t \\times 1\\ ${rps})`],[Tx('script let through','脚本放过的'),`\\min(${T.script},\\ 1) = 1\\ ${rps}`]]),'')+
   MB(tx(`<b>Batch jobs in a day.</b> In hour h the chats keep the fleet ${M('u_h')} busy (the hour’s rate ÷ capacity). Below θ, batch jobs take the share ${M('\\theta - u_h')}. One GPU doing only batch jobs finishes ${f4(b)} a second (measured). Added up over the day, that is the day’s batch capacity, the most these GPUs could process. The jobs done are at most the ${fJobs(fl.jobs)} handed in:`,`<b>一天的批处理。</b>第 h 个小时，聊天让卡忙 ${M('u_h')}（这一小时的速率 ÷ 容量）。低于 θ 时，批处理用掉 ${M('\\theta - u_h')} 那一份。一张卡只做批处理，每秒做完 ${f4(b)} 个（量出来的）。全天加起来，就是全天批处理容量，也就是这组卡最多能处理多少。做完的任务最多就是交上来的 ${fJobs(fl.jobs)} 个：`),
     EQ([[Tx('batch capacity','批处理容量'),`\\sum_{h=0}^{23} \\max(0,\\ \\theta - u_h) \\times ${s.n} \\times ${f4(b)} \\times 3600`],['',`${f3(sh)} \\times ${s.n} \\times ${f4(b)} \\times 3600 \\approx ${fJobs(cs)}`],[Tx('done','做完'),`\\min(${fJobs(cs)},\\ ${fJobs(fl.jobs)}) = ${fJobs(Math.min(cs,fl.jobs))}`],...(cs-fl.jobs>=500?[[Tx('spare','余量'),`${fJobs(cs)} - ${fJobs(fl.jobs)} = ${fJobs(cs-fl.jobs)}`]]:[]),[`u_{${qh}}`,`\\frac{${f1(u3.c)}}{${fint(d.cap)}} = ${fP(u3.u)}`]]),
     tx(`For ${setupName(s)}.${s.rl?` The ${f1(u3.c)} in ${M(`u_{${qh}}`)} is the chat rate at ${qh}:00 after the rate limit, which leaves the script 1 request a second.`:''}`,`按${setupName(s)}算。${s.rl?`${M(`u_{${qh}}`)} 里的 ${f1(u3.c)} 是限流以后凌晨 ${qh} 点的聊天速率，脚本只剩每秒 1 个。`:''}`))+
   MB(tx(`<b>What the day costs.</b> The fleet runs all day, and this lab assumes $${PRICE} per GPU hour:`,`<b>一天要花多少。</b>这组卡全天开着，本 Lab 假设每 GPU 小时 ${PRICE} 美元：`),
     EQ([[Tx('GPU hours','GPU 小时'),`${s.n} \\times 24 = ${s.n*24}`],[Tx('cost','花费'),`${s.n*24} \\times ${PRICE} = ${fint(s.n*24*PRICE)}\\ ${Tx('USD','美元')}`]]));}

/* ---------- stage head, status ---------- */
function statusOf(run){const m=run.m;
  if(PLAY.on) return {cls:'',short:tx('playing','播放中')};
  const ok=(!m.paid||m.paid.frac>=ADM.TARGET.paid-1e-9)&&(!m.free||m.free.frac>=ADM.TARGET.free-1e-9);
  if(ok&&overOf(curView())) return {cls:'bad',short:tx('misses if the peak lasts','高峰持续下去就达不到'),over:true};
  return {cls:ok?'ok':'bad',short:ok?tx('chat targets met','聊天达到服务目标'):tx('chat target missed','聊天没达到服务目标')};}
function drawHead(run,v){
  $('#adm-st-title',app).textContent=(PLAY.on?tx('Playing: ','正在播放：'):tx('Whole run: ','整次运行：'))+whatOf(v);
  $('#adm-st-sub',app).textContent=v.day?tx(`${M8B}, one copy per H100. Simulated step by step; requests are seeded random draws.`,`${M8B}，每张 H100 一份。逐步模拟，请求是带种子的随机抽样。`):tx(`${M8B} on one H100, paid chats only. Simulated step by step.`,`${M8B}，一张 H100，只有付费聊天。逐步模拟。`);
  if(v.kind==='surge') $('#adm-st-sub',app).textContent=tx(`${M8B}, one copy per H100, free queue ≤ ${SURGE.q}, rate limit on. Simulated step by step.`,`${M8B}，每张 H100 一份，免费请求最多排 ${SURGE.q} 个，限流打开。逐步模拟。`);
  const st=statusOf(run), b=$('#adm-status',app); b.className='status '+st.cls; b.textContent=st.short;
  b.dataset.tipx=PLAY.on?tx('The run is playing; its numbers show when it ends.','正在播放，播完后显示结果。'):st.over?tx('The afternoon traffic is above capacity and free requests have no queue limit, so over the afternoon their queue keeps growing; see the note under the goals.','下午的流量比容量多，免费请求的排队又没有上限，高峰持续下去，排队会一直涨；见目标下面的说明。'):TIP.slo();
  $('#adm-q-h',app).textContent=qHead(v,run);}
function drawRunCtl(run){
  const pb=$('#adm-play',app); if(!pb) return; const T=runT(run), atEnd=phNow>=T;
  pb.textContent=PLAY.on?tx('❚❚ Pause','❚❚ 暂停'):(atEnd?tx('▶ Play','▶ 播放'):tx('▶ Continue','▶ 继续'));
  pb.dataset.tipx=tx(`Plays the run at the chosen speed: about ${RUN_SECS[spdNow()]} s.`,`按选定的速度播放这次运行：约 ${RUN_SECS[spdNow()]} 秒。`);
  $('#adm-replay',app).dataset.tipx=tx('Plays this run again from the start.','从头再播放一次这次运行。');
  $('#adm-prog',app).value=String(Math.round(1000*Math.min(1,phNow/T)));
  $$('.seg-ctl[data-key="spd"]',app).forEach(g=>A.ui.setPressed(g,spdNow()));}
function render(){
  if(!app) return; const v=curView(), run=v.run();
  if(run.key!==lastRunKey){lastRunKey=run.key; if(!PLAY.on) phNow=Infinity;}
  drawHead(run,v); drawDay(v); drawQueue(run,v); drawRunCtl(run);
  const gl=$('#adm-goals',app), gh=goalsHTML(run,v); if(gl._h!==gh){gl.innerHTML=gh; gl._h=gh;}
  drawMetrics(run,v);
  const md=$('#adm-math',app), mb=$('#adm-mathbody',app); if(md.open){const mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}}
  if(!md._t){md._t=1; md.addEventListener('toggle',()=>{if(md.open) render();});}
  syncControls(); updateRail();
  if(!PLAY.on&&/^#?admission(~|$)/.test(location.hash||'#admission')) A.hash.write(token());   /* the kit keeps the settings in the address only in Scenarios */
  A.dock.refresh(); announce(run);}
function announce(run){clearTimeout(annT); annT=setTimeout(()=>{if(!app||PLAY.on) return; const m=run.m;
  A.announce(m.free?tx(`${fP(m.paid.frac)} of paid and ${fP(m.free.frac)} of free requests meet their goals.`,`付费请求 ${fP(m.paid.frac)} 达标，免费请求 ${fP(m.free.frac)} 达标。`):tx(`${fP(m.paid.frac)} of requests meet their goals; 90% get their first token within ${fT(m.paid.p90)}.`,`${fP(m.paid.frac)} 的请求达标；90% 在 ${fT(m.paid.p90)}内拿到首 token。`));},400);}

/* ---------- the run player (as labs/pd-disagg.js): play / pause, replay, speed, progress; R3.20 ---------- */
function stopPlay(){if(PLAY.raf) cancelAnimationFrame(PLAY.raf); PLAY.on=false; PLAY.raf=0;}
function playFrom(t0,quick){
  stopPlay(); const run=curRun(), T=runT(run); phNow=t0; nPlays++;
  const secs=quick?1.5:RUN_SECS[spdNow()], rate=T/secs;
  PLAY={on:true,raf:0,last:0,key:run.key};
  const tick=ts=>{if(!PLAY.on||!app){PLAY.raf=0; return;}
    if(PLAY.last){phNow+=Math.min(0.1,(ts-PLAY.last)/1000)*rate; if(phNow>=T){phNow=Infinity; watched.add(PLAY.key); PLAY.on=false; PLAY.raf=0; render(); return;} moveHead(run); const pr=$('#adm-prog',app); if(pr) pr.value=String(Math.round(1000*phNow/T));}
    PLAY.last=ts; PLAY.raf=requestAnimationFrame(tick);};
  PLAY.raf=requestAnimationFrame(tick); render();}
/* show the current run: play it only when its picture is new (R3.20) */
function showRun(src){
  const run=curRun();
  if(PLAY.on&&PLAY.key===run.key) return;
  stopPlay();
  if(watched.has(run.key)||A.reduceMotion()||src==='quiet'){phNow=Infinity; watched.add(run.key); render(); return;}
  playFrom(0,src==='settings');}

/* ---------- the rail ---------- */
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!s.expect())||(!!s.gate&&!s.gate());}
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(L(x.lead))})),m2:true,current:step,body:s.body(),back:step>0,showMe:false,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function renderRail(){const rail=$('#adm-rail',app); if(!rail) return; rail.innerHTML=railFull(); render();}
const both=m=>tx(`paid ${fP(m.paid.frac)}, free ${fP(m.free.frac)}`,`付费 ${fP(m.paid.frac)}，免费 ${fP(m.free.frac)}`);
function leverResult(id,v){
  if(id==='capacity') return tried['n'+v]?tx(`paid ${fP(pk({...BASE,n:+v}).m.paid.frac)}`,`付费 ${fP(pk({...BASE,n:+v}).m.paid.frac)}`):'';
  if(id==='queue') return tried['f'+v]?tx(`p90 ${fT(oneRun(+v).m.paid.p90)}`,`p90 ${fT(oneRun(+v).m.paid.p90)}`):'';
  if(id==='admit') return tried['k'+v]?both(pk(P5[v]).m):'';
  if(id==='prio') return tried['p'+v]?both(surgeRun(v).m):'';
  if(id==='batch'){if(!tried['t'+v]) return ''; const b=batchOf(ADM.day(BASE_T,cfgOf(P7S(+v))),BASE_JOBS); return tx(`capacity ${fJobs(b.cap)}, ${fJobs(b.done)} done`,`容量 ${fJobs(b.cap)}，做完 ${fJobs(b.done)}`);}
  return '';}
const leverNow=id=>({capacity:p2n,queue:p3f,admit:p5k,prio:p6k,batch:p7t}[id]);
function updateRail(){
  const rail=$('#adm-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    const show=!!s.obs&&(s.expect?s.expect():true)&&!(PLAY.on&&!watched.has(curRun().key)&&s.id!=='predict');
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${tx('Observed','观察到')}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}}}
    const cur=leverNow(s.id);
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(cur!=null&&String(cur)===v)); const rr=b.querySelector('.lv-r'); if(rr){const t=leverResult(s.id,v); if(rr.textContent!==t) rr.textContent=t;}});
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#adm-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):''; ins.hidden=!html; if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const h=$('#adm-ch-h',rail); if(!h) return; const gh=S.n*24; h.textContent=tx(`${gh} ($${fint(gh*PRICE)})`,`${gh}（${fint(gh*PRICE)} 美元）`);
    [['chn','n'],['chq','q'],['chr','rl'],['chp','prio'],['cht','th']].forEach(([k,f])=>$$(`.seg-ctl[data-key="${k}"]`,rail).forEach(g=>A.ui.setPressed(g,String(S[f]))));
    const res=$('#adm-ch-res',rail), html=chResultHTML(); if(res._h!==html){res.innerHTML=html; res._h=html;}}
}
function syncControls(){
  const c=$('#adm-controls',app); if(!c) return; const lock=$('#adm-lock',app), wrap=$('#adm-ctls',app), free=mode==='sandbox';
  wrap.hidden=!free; lock.hidden=free; lock.textContent=mode==='guided'?tx('Each part sets its own run; every setting is free in Scenarios.','每一部分有自己的运行；在应用场景里所有设置都能改。'):tx('In the challenge, the choices are in the task card.','挑战题的选择都在任务卡里。');
  if(!free) return;
  ['q','rl','prio','th'].forEach(k=>A.ui.setPressed($(`.seg-ctl[data-key="${k}"]`,c),String(S[k])));
  const ni=$('#adm-ctl-n-in',c); if(ni&&document.activeElement!==ni){ni.value=String(S.n); A.ui.setRange(ni);} const no=ni&&ni.closest('.ctl')&&ni.closest('.ctl').querySelector('output'); if(no) no.textContent=String(S.n);}

/* ---------- navigation ---------- */
function enterStep(i){step=Math.max(0,Math.min(STEPS.length-1,i)); renderRail(); showRun(STEPS[step].id==='mission'?'quiet':'');}
function revealStep(){A.ui.reveal($('#adm-rail .step.current',app)||$('#adm-rail',app));}
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2, R9.7), as labs/llm-judge.js */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#adm-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#adm-rail .step.current',app)||$('#adm-rail .railcard',app)||$('#adm-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function setMode(m){
  if(m===mode) return; const was=mode; mode=m; stopPlay();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH_START}; chStartBg();}
  if(m==='sandbox'&&was==='guided') S={...BASE};
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail(); showRun('quiet');}
const token=()=>[S.n,S.q,S.rl,S.prio,S.th].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [n,q,rl,pr,th]=p.map(Number);
  if(!(n>=3&&n<=8&&Number.isInteger(n))||![0,500,200,50].includes(q)||![0,60].includes(rl)||![0,1].includes(pr)||![60,80,100].includes(th)) return null;
  return {n,q,rl,prio:pr,th};}

/* ---------- interview drill ---------- */
let drillSeed=1, drillOpen=false;
function makeDrill(seed){const r=ADM.mulberry(seed*7919+11), g=G(), peak=100+10*Math.floor(r()*17), over=0.1+0.05*Math.floor(r()*4); return {g,peak,over};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#adm-drill-q',app), a=$('#adm-drill-a',app), n=Math.ceil(d.peak/d.g-1e-9), lam=d.g*(1+d.over), grow=(lam-d.g)*60;
  q.innerHTML=tx(`Your chat traffic peaks at ${d.peak} requests per second, and one H100 serving ${M8B} has a goodput of ${d.g} requests per second. How many GPUs do you need? And if one GPU of that fleet gets ${Math.round(d.over*100)}% more than its goodput for a minute, how long is its queue at the end of that minute?`,
                 `你的聊天流量高峰是每秒 ${d.peak} 个请求，一张跑 ${M8B} 的 H100 的达标吞吐是每秒 ${d.g} 个。要几张卡？如果其中一张卡有一分钟收到比达标吞吐多 ${Math.round(d.over*100)}% 的请求，这一分钟结束时它的排队有多长？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>${n} GPUs; about ${fint(grow)} requests waiting.</b></p>${EQ([['\\text{GPUs}',`\\frac{${d.peak}}{${d.g}} \\approx ${f1(d.peak/d.g)} \\to ${n}`],['\\text{queue}',`(${f2(lam)} - ${d.g}) \\times 60 \\approx ${fint(grow)}`]])}<p>The queue estimate counts the goodput as what the GPU finishes; it really finishes a little more, so the real queue is a bit shorter. Either way it does not shrink until the load falls below the GPU’s limit, which is why a fleet keeps headroom or a gateway turns some requests away.</p>`,
                 `<p><b>${n} 张卡；排队约 ${fint(grow)} 个请求。</b></p>${EQ([['\\text{卡数}',`\\frac{${d.peak}}{${d.g}} \\approx ${f1(d.peak/d.g)} \\to ${n}`],['\\text{排队}',`(${f2(lam)} - ${d.g}) \\times 60 \\approx ${fint(grow)}`]])}<p>这里把达标吞吐当作卡做完的速率；卡实际做完的稍多一点，所以真实的排队短一些。不管怎样，负载降到卡的上限以下之前，排队不会变短；所以机群要留余量，或者让网关拒绝一部分请求。</p>`);}

/* ---------- jumps ---------- */
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  return '';}
function jump(k){
  if(!app) return;
  if(k==='math'){const d=$('#adm-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#adm-controls',app));}}

/* ---------- events ---------- */
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
function setS(p,src){Object.assign(S,p); byLearner(()=>{render(); showRun(src||'settings');});}
function act(a,btn){
  if(a==='play'){if(PLAY.on){stopPlay(); render(); return;} const run=curRun(); playFrom(phNow>=runT(run)?0:phNow,false); return;}
  if(a==='replay'){playFrom(0,false); return;}
  if(a==='reset'){setS({...BASE}); return;}
  if(a==='pred-show'){if(!pred) return; predShown=true; byLearner(()=>{renderRail(); showRun();}); requestAnimationFrame(()=>revealBox($('#adm-rail [data-obs]',app))); return;}
  if(a==='submit'){chSubmitted=true; chStartBg(); renderRail(); requestAnimationFrame(()=>revealBox($('#adm-ch-res',app))); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#adm-code',app); try{navigator.clipboard.writeText(pre.textContent);}catch(e){} btn.textContent=tx('Copied','已复制'); setTimeout(()=>{btn.textContent=A.t('copyCode');},1600);}}
function pickLever(id,v){
  if(id==='capacity'){p2n=+v; tried['n'+v]=true;}
  else if(id==='queue'){p3f=+v; tried['f'+v]=true; if(+v>1.05) tried.q11=true;}
  else if(id==='admit'){p5k=v; tried['k'+v]=true; if(v==='rl') tried.rl=true; if(v==='pf') tried.pf=true;}
  else if(id==='prio'){p6k=v; tried['p'+v]=true; if(v==='pre') tried.pre=true;}
  else if(id==='batch'){p7t=+v; tried['t'+v]=true; if(+v===100) tried.t100=true;}
  else return false;
  byLearner(()=>{render(); showRun();}); requestAnimationFrame(()=>revealBox($('#adm-rail [data-obs]',app))); return true;}
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#adm-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.pred&&!predShown){pred=d.pred; renderRail(); const fb=$(`#adm-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&mode==='guided'&&pickLever(STEPS[step].id,d.lever)) return;
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='spd'){spdUser=v; if(PLAY.on) playFrom(phNow,false); else render(); return;}
      const CHK={chn:'n',chq:'q',chr:'rl',chp:'prio',cht:'th'};
      if(CHK[k]) return byLearner(()=>{S[CHK[k]]=+v; render();});
      if(['q','rl','prio','th'].includes(k)) return setS({[k]:+v});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; S={...p.s}; byLearner(()=>{render(); showRun('settings');}); requestAnimationFrame(()=>{if(app) revealBox($('#adm-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#adm-rail',app)); return;}
    if(d.act) act(d.act,b);});
  el.addEventListener('input',e=>{const t=e.target;
    if(t.id==='adm-prog'){stopPlay(); const run=curRun(), T=runT(run); phNow=T*(+t.value)/1000; if(+t.value>=1000){phNow=Infinity; watched.add(run.key);} moveHead(run); drawRunCtl(run); return;}
    if(t.id==='adm-ctl-n-in'){const o=t.closest('.ctl')&&t.closest('.ctl').querySelector('output'); if(o) o.textContent=t.value;}});
  el.addEventListener('change',e=>{const t=e.target; if(t.id==='adm-ctl-n-in') setS({n:+t.value});
    if(t.id==='adm-prog'&&+t.value>=1000){phNow=Infinity; render();}});
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));}

/* ---------- registration ---------- */
/* the lab state contract (round 3 #6): what the learner did here, and putting it back before the first mount after a reload */
const TRIED_OK=new Set(['n4','n6','n7',...LOADS.map(f=>'f'+f),'q11',...Object.keys(P5).map(k=>'k'+k),'rl','pf','pfcfs','pprio','ppre','pre','t60','t80','t100']);
const sOf=x=>x&&typeof x==='object'?parse([x.n,x.q,x.rl,x.prio,x.th].map(Number).join('_')):null;
function snap(){return {v:1,mode,step,S:{...S},pred,observed:predShown,saved:saved?{...saved}:null,p2n,p3f,p5k,p6k,p7t,tried:{...tried},
  watched:[...watched].slice(-300),spd:spdUser,drill:[drillSeed,drillOpen],chSubmitted,chReveal,passed:mode==='challenge'&&chSubmitted&&chV(chPick()).ok};}
function restoreState(o){if(!o||o.v!==1) return;
  mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
  step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
  const s=sOf(o.S); S=s?{...s}:{...BASE};
  if(mode==='challenge'&&!CH_OPTS().some(x=>chKey(x)===chKey(S))) S={...CH_START};
  pred=PRED.some(x=>x.v===o.pred)?o.pred:null; predShown=!!pred&&o.observed===true;
  const sv=mode==='challenge'?sOf(o.saved):null; saved=sv?{...sv}:null;
  p2n=[4,6,7].includes(o.p2n)?o.p2n:null; p3f=LOADS.includes(o.p3f)?o.p3f:null;
  p5k=Object.keys(P5).includes(o.p5k)?o.p5k:null; p6k=['fcfs','prio','pre'].includes(o.p6k)?o.p6k:null; p7t=[60,80,100].includes(o.p7t)?o.p7t:null;
  tried={}; if(o.tried&&typeof o.tried==='object') for(const k in o.tried) if(TRIED_OK.has(k)&&o.tried[k]===true) tried[k]=true;
  watched.clear(); if(Array.isArray(o.watched)) for(const k of o.watched.slice(-300)) if(typeof k==='string'&&k.length<=80&&/^[a-z]+(\|[\w.-]+)*$/.test(k)) watched.add(k);
  spdUser=Object.keys(RUN_SECS).includes(o.spd)?o.spd:null;
  if(Array.isArray(o.drill)){drillSeed=Number.isInteger(o.drill[0])&&o.drill[0]>=1&&o.drill[0]<1e6?o.drill[0]:1; drillOpen=o.drill[1]===true;}
  chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;}

A.registerLab({
  id:ID, snap, restore:restoreState,
  mount(el,tok){
    root=el; const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#adm-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#adm-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    const rail=$('#adm-rail',app); rail.innerHTML=railFull();
    phNow=Infinity; render();
    if(mode==='guided'&&STEPS[step].id!=='mission') showRun();
    if(mode==='challenge') chStartBg(); else setTimeout(()=>{if(app) chStartBg();},1500);},
  leave(){stopPlay(); clearTimeout(annT); stageStop(); clearTimeout(chT); chT=0; root=null; app=null;},
  onLang(){if(root){const d=$('#adm-math',app), open=d&&d.open, wasPlaying=PLAY.on; stopPlay(); this.mount(root,null); if(open){$('#adm-math',app).open=true; render();} if(wasPlaying) showRun();}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#adm-q',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#adm-graph',app),
    render({status,vis,read}){if(!app) return; const run=curRun(), s=statusOf(run), m=run.m;
      status.className='status '+s.cls; status.textContent=s.short;
      vis.innerHTML=`<div class="adm-dk"><i style="width:${PLAY.on?0:(m.paid.frac*100).toFixed(0)}%"></i></div>`;
      read.textContent=PLAY.on?tx('Playing the run; the numbers show when it ends.','正在播放，播完后显示结果。'):(m.free?tx(`Paid ${fP(m.paid.frac)}, free ${fP(m.free.frac)} meet their goals.`,`付费 ${fP(m.paid.frac)}、免费 ${fP(m.free.frac)} 达标。`):tx(`${fP(m.paid.frac)} meet their goals; p90 first token ${fT(m.paid.p90)}.`,`${fP(m.paid.frac)} 达标；首 token p90 ${fT(m.paid.p90)}。`));}}
});

/* ---------- read-only hook for tests ---------- */
window.__adm=Object.freeze({
  engine:ADM, STEP_IDS:STEPS.map(s=>s.id), PRESETS:PRESETS.map(p=>({id:p.id,s:{...p.s}})), CH_T:{...CH_T}, CH_JOBS, CH_START:{...CH_START}, BASE_T:{...BASE_T}, BASE_JOBS, BASE:{...BASE}, P5, SURGE:{...SURGE}, PRED_RIGHT, LOADS:LOADS.slice(),
  view:()=>{const v=curView(); return {kind:v.kind,s:v.s?{...v.s}:null,key:v.run().key};},
  state:()=>({...S,mode,step,pred,predShown,p2n,p3f,p5k,p6k,p7t,tried:{...tried},chSubmitted,chReveal,playing:PLAY.on,ph:phNow,plays:nPlays,watched:[...watched],spd:spdNow(),spdUser,chReady:chReady()}),
  chOpts:()=>CH_OPTS().map(o=>{const v=chV(o); return {...o,ok:v.ok,gpuh:v.gpuh};}),
  exits:()=>{const e=EXITS(); return {explainA:e.explain.map(q=>q.a),transferA:e.transfer.map(q=>q.a)};},
  code:()=>CODE
});
/* UI END */
})();
