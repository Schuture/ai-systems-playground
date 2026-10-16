/* pd-disagg (world inference): Why run prefill and decode on different GPUs?
   Fidelity: Simulated. A seeded multi-GPU serving simulator on the step model of the other inference labs (S2), with the serving
   capstone's goals (S4); real-scale KV transfer sizes and link speeds from datasheet figures (ml-engineering network notes).
   Parts: one GPU stalled by a 32,768-token prefill; chunked prefill; predict 1 prefill + 3 decode GPUs; split pools; the KV transfer;
   the ratio and goodput per GPU on 8 GPUs; what to remember. window.__pdd is a read-only hook for tests/pd-disagg_numbers.py and
   tests/pd-disagg_flow.py; window.PDD_ENGINE is the engine the home tile (teasers/pd-disagg.js) reuses. */
(function(){
'use strict';
/* ENGINE BEGIN — pd-disagg's multi-GPU serving simulator. tests/pd-disagg_numbers.py holds the Python reference and runs both.
   Pure: no DOM. Llama-3.1-8B in BF16, each GPU an independent copy of the model (no tensor parallelism). Every step is priced by
   the step model of the other inference labs (S2):  t = max(FLOPs / (989.5 TFLOP/s × 50%), bytes / (3.35 TB/s × 60%)),
   FLOPs = 2·P_LAYERS·tokens + 2·P_HEAD·logits + attention, bytes = W_READ + 128 KiB × (KV tokens the step reads).
   Colocated GPU: continuous batching with chunked prefill, as the serving capstone's engine: decodes first (1 token of the budget
   each), then prefills in admission order, then new requests first come, first served while budget and KV room (prompt + whole
   answer, set aside at admission) last. Budget 0 = no limit.
   Split: prefill GPUs run only prefill (FCFS, at most PRE_BUDGET tokens per step) and produce the first token; then the request
   waits for a decode GPU with room for prompt + answer (the one with the most room), and its KV cache (128 KiB × prompt tokens)
   crosses the prefill GPU's link (one transfer at a time, FIFO) at LINK bytes/s. The decode GPU takes it in at its next step;
   every step decodes 1 token for every request it holds.
   Router: a new request goes to the colocated (or prefill) GPU with the fewest prompt tokens still to prefill, then the fewest
   requests, then the lowest index. Events at the same moment: step ends (GPU order), transfers landing, arrivals; then idle GPUs
   with work start a step (GPU order). */
const PDD=(function(){
'use strict';
const L=32, D=4096, KVH=8, HD=128, FFN=14336, VOCAB=128256;
const P_LAYER=2*D*D+2*D*KVH*HD+3*D*FFN+2*D, P_LAYERS=L*P_LAYER+D, P_HEAD=VOCAB*D, P_TOTAL=P_LAYERS+2*P_HEAD;
const KV_TOK=2*L*KVH*HD*2, W_READ=2*(P_LAYERS+P_HEAD), W_MEM=2*P_TOTAL;
const FS=1979e12/2*0.5, BS=3.35e12*0.6, GiB=2**30, BLOCK=16;
const POOL_T=Math.floor((80*GiB*0.9-W_MEM-2*GiB)/KV_TOK), POOL_B=Math.floor(POOL_T/BLOCK);
const EPS=1e-9, LINK={nvl:450e9, ib:50e9}, PRE_BUDGET=8192;
const GOALS={chat:{ttft:1.0,gap:0.15}, doc:{ttft:5.0,gap:null}}, TARGET=0.9;
const PER_GPU=60;
const RATES=[]; for(let i=1;i<=16;i++) RATES.push(i*0.5);
const attn=(a,b)=>2*L*D*(b*(b+1)-a*(a+1));
const blocks=n=>Math.floor((n+BLOCK-1)/BLOCK);
function mulberry(seed){let h=seed>>>0; return ()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};}
const expGap=(rnd,rate)=>-Math.log(1-rnd())/rate;
const ms=x=>Math.floor(x*1000+0.5)/1000;
const uni=(rnd,lo,hi)=>lo+Math.floor(rnd()*(hi-lo+1));
function sortLoad(a){return a.map((s,i)=>[s,i]).sort((x,y)=>x[0].t-y[0].t||x[1]-y[1]).map(x=>x[0]);}
/* traffic: chats (prompt 400–1,600 tokens, answer 120–320) and long documents (prompt 16,384–32,768, answer 100–300) */
const MIX={A:{doc:0.1,seed:7}, B:{doc:0.3,seed:9}, C:{doc:0,seed:7}};
function loadMix(kind,n,rate){const m=MIX[kind], rnd=mulberry(m.seed), out=[]; let t=0;
  for(let i=0;i<n;i++){t+=expGap(rnd,rate); const u=rnd();
    if(u<m.doc) out.push({t:ms(t),cls:'doc',own:uni(rnd,16384,32768),out:uni(rnd,100,300)});
    else out.push({t:ms(t),cls:'chat',own:uni(rnd,400,1600),out:uni(rnd,120,320)});}
  return out;}
/* one GPU: 8 chats (6 a second on average), and, with doc, one 32,768-token document at 1.0 s */
function loadOne(doc){const rnd=mulberry(3), out=[]; let t=0;
  for(let i=0;i<8;i++){t+=expGap(rnd,6.0); out.push({t:ms(t),cls:'chat',own:uni(rnd,400,1600),out:uni(rnd,120,320)});}
  if(doc) out.push({t:1.0,cls:'doc',own:32768,out:200});
  return sortLoad(out);}

function stepCost(work){
  let T=0, lg=0, fa=0, kv=0;
  for(const w of work){const r=w.r;
    if(w.k==='d'){const c=r.P+r.gen-1; T+=1; lg+=1; fa+=attn(c,c+1); kv+=c;}
    else {T+=w.b-w.a; lg+=w.b===r.P?1:0; fa+=attn(w.a,w.b); kv+=w.b;}}
  const fl=2*P_LAYERS*T+2*P_HEAD*lg+fa, by=W_READ+KV_TOK*kv;
  return {T,fl,by,tc:fl/FS,tm:by/BS,dt:Math.max(fl/FS,by/BS)};
}
/* cfg: {mode:'co'|'split', n, p (prefill GPUs when split), budget (colocated chunk budget, 0 = no limit), link:'nvl'|'ib'} */
function simulate(spec,cfg){
  const mode=cfg.mode, n=cfg.n, p=mode==='split'?cfg.p:0, budget=cfg.budget||0, bw=LINK[cfg.link||'nvl'];
  const R=spec.map((s,i)=>({id:i,cls:s.cls,arr:s.t,P:s.own,out:s.out,doneP:0,gen:0,toks:[],first:null,fin:null,g:null,dg:null,x0:null,x1:null,need:blocks(s.own+s.out)}));
  const G=[]; for(let i=0;i<n;i++) G.push({i,kind:mode==='co'?'co':(i<p?'pre':'dec'),busy:false,t1:0,work:null,run:[],wait:[],resv:0,link:0,steps:[],peak:0});
  const pend=[], xf=[], order=R.map((r,i)=>i).sort((a,b)=>R[a].arr-R[b].arr||a-b);
  let nxt=0, t=0, guard=0;
  const pendingPf=g=>{let s=0; for(const r of g.run) if(r.first==null) s+=r.P-r.doneP; for(const r of g.wait) s+=r.P; return s;};
  function route(r){let best=null, bk=null;
    for(const g of G){if(g.kind!=='co'&&g.kind!=='pre') continue; const k=[pendingPf(g),g.run.length+g.wait.length,g.i];
      if(!best||k[0]<bk[0]||(k[0]===bk[0]&&(k[1]<bk[1]||(k[1]===bk[1]&&k[2]<bk[2])))){best=g; bk=k;}}
    r.g=best.i; best.wait.push(r);}
  function tryPend(now){
    while(pend.length){const r=pend[0]; let best=null;
      for(const g of G){if(g.kind!=='dec') continue; if(!best||POOL_B-g.resv>POOL_B-best.resv) best=g;}
      if(POOL_B-best.resv<r.need) return;
      pend.shift(); best.resv+=r.need; best.peak=Math.max(best.peak,best.resv); r.dg=best.i;
      const src=G[r.g], s0=Math.max(now,src.link), s1=s0+r.P*KV_TOK/bw; src.link=s1; r.x0=s0; r.x1=s1; xf.push(r);}
  }
  function start(g,now){
    let bud=g.kind==='co'&&budget?budget:(g.kind==='pre'?PRE_BUDGET:1e12);
    const work=[];
    if(g.kind==='dec'){for(const r of g.wait) g.run.push(r); g.wait=[]; for(const r of g.run) work.push({r,k:'d',a:0,b:0});}
    else {
      if(g.kind==='co') for(const r of g.run) if(r.first!=null&&r.gen<r.out){work.push({r,k:'d',a:0,b:0}); bud-=1;}
      for(const r of g.run) if(r.first==null&&r.doneP<r.P&&bud>0){const k=Math.min(r.P-r.doneP,bud); work.push({r,k:'p',a:r.doneP,b:r.doneP+k}); bud-=k;}
      while(g.wait.length&&bud>0){const r=g.wait[0];
        if(g.kind==='co'&&POOL_B-g.resv<r.need) break;
        g.wait.shift(); g.run.push(r);
        if(g.kind==='co'){g.resv+=r.need; g.peak=Math.max(g.peak,g.resv);}
        const k=Math.min(r.P-r.doneP,bud); work.push({r,k:'p',a:r.doneP,b:r.doneP+k}); bud-=k;}
    }
    if(!work.length) return;
    const c=stepCost(work);
    g.busy=true; g.t1=now+c.dt; g.work=work;
    let pt=0, nd=0, doc=false; for(const w of work){if(w.k==='p'){pt+=w.b-w.a; if(w.r.cls==='doc') doc=true;} else nd++;}
    g.steps.push({t0:now,t1:now+c.dt,pt,nd,doc,ids:work.filter(w=>w.k==='p').map(w=>w.r.id)});
  }
  function finish(g){
    const t1=g.t1; g.busy=false;
    for(const w of g.work){const r=w.r;
      if(w.k==='p'){r.doneP=w.b; if(w.b===r.P){r.first=t1; r.toks.push(t1); r.gen=1; if(g.kind==='pre'){g.run.splice(g.run.indexOf(r),1); pend.push(r);}}}
      else {r.gen++; r.toks.push(t1);}}
    for(const r of g.run.slice()) if(r.first!=null&&r.gen>=r.out){r.fin=t1; g.run.splice(g.run.indexOf(r),1); g.resv-=r.need;}
    g.work=null;
  }
  for(;;){
    if(++guard>2000000) throw new Error('runaway simulation');
    const tA=nxt<order.length?R[order[nxt]].arr:Infinity;
    let tG=Infinity; for(const g of G) if(g.busy&&g.t1<tG) tG=g.t1;
    let tX=Infinity; for(const r of xf) if(r.x1<tX) tX=r.x1;
    t=Math.min(tA,tG,tX); if(t===Infinity) break;
    for(const g of G) if(g.busy&&g.t1<=t+EPS) finish(g);
    for(const r of xf.filter(r=>r.x1<=t+EPS)){xf.splice(xf.indexOf(r),1); G[r.dg].wait.push(r);}
    while(nxt<order.length&&R[order[nxt]].arr<=t+EPS){route(R[order[nxt]]); nxt++;}
    tryPend(t);
    for(const g of G) if(!g.busy) start(g,t);
  }
  for(const r of R) if(r.fin==null||r.toks.length!==r.out) throw new Error('request '+r.id+' did not finish');
  let end=0; for(const r of R) end=Math.max(end,r.fin);
  return {R,G,end,cfg};
}
function pct(xs,p){const s=xs.slice().sort((a,b)=>a-b); if(!s.length) return null; const k=Math.max(1,Math.ceil(p/100*s.length-1e-9)); return s[k-1];}
const gapOf=r=>{let g=0; for(let i=1;i<r.toks.length;i++) g=Math.max(g,r.toks[i]-r.toks[i-1]); return g;};
const okOf=r=>{const g=GOALS[r.cls]; return (r.first-r.arr<=g.ttft+EPS)&&(g.gap==null||gapOf(r)<=g.gap+EPS);};
function metrics(run){
  const R=run.R, out={}; let ok=0;
  for(const cls of ['chat','doc']){const rs=R.filter(r=>r.cls===cls); if(!rs.length) continue;
    const tt=rs.map(r=>r.first-r.arr), gp=rs.map(gapOf), k=rs.filter(okOf).length; ok+=k;
    out[cls]={n:rs.length,ok:k,ttft_p90:pct(tt,90),ttft_max:Math.max(...tt),gap_max:Math.max(...gp),gap_p90:pct(gp,90),
      tpot:rs.reduce((s,r)=>s+(r.fin-r.first)/(r.out-1),0)/rs.length};}
  out.all={n:R.length,ok,ok_frac:ok/R.length,end:run.end};
  return out;
}
const RUNS=new Map();
function runOf(kind,cfg,rate){
  const key=[kind,cfg.mode,cfg.n,cfg.p||0,cfg.budget||0,cfg.link||'nvl',rate].join('|');
  if(RUNS.has(key)) return RUNS.get(key);
  const spec=kind==='one'?loadOne(true):kind==='one0'?loadOne(false):loadMix(kind,PER_GPU*cfg.n,rate*cfg.n);
  const run=simulate(spec,cfg); run.m=metrics(run); run.key=key; RUNS.set(key,run);
  if(RUNS.size>400) RUNS.delete(RUNS.keys().next().value);
  return run;
}
/* goodput per GPU (DistServe's definition, S4): the highest rate on the grid (per GPU, 0.5 to 8 requests/s) at which at least 90% of
   requests meet their goals, found by binary search on the grid (the share meeting goals falls as the rate rises); rates are per GPU */
const GP=new Map();
const gpKey=(kind,cfg)=>[kind,cfg.mode,cfg.n,cfg.p||0,cfg.budget||0,cfg.link||'nvl'].join('|');
const goodputReady=(kind,cfg)=>GP.has(gpKey(kind,cfg));
function goodput(kind,cfg){
  const key=[kind,cfg.mode,cfg.n,cfg.p||0,cfg.budget||0,cfg.link||'nvl'].join('|');
  if(GP.has(key)) return GP.get(key);
  let lo=-1, hi=RATES.length; const rows=[];
  while(hi-lo>1){const mid=(lo+hi)>>1, m=runOf(kind,cfg,RATES[mid]).m; rows.push([RATES[mid],m.all.ok_frac]); if(m.all.ok_frac>=TARGET-EPS) lo=mid; else hi=mid;}
  const res={g:lo<0?0:RATES[lo],rows}; GP.set(key,res); return res;
}
return {goodputReady,simulate,metrics,runOf,goodput,loadMix,loadOne,stepCost,pct,gapOf,okOf,attn,blocks,mulberry,GOALS,TARGET,RATES,PER_GPU,MIX,
  C:{L,D,KVH,HD,FFN,VOCAB,P_LAYER,P_LAYERS,P_HEAD,P_TOTAL,KV_TOK,W_READ,W_MEM,FS,BS,GiB,BLOCK,POOL_T,POOL_B,LINK,PRE_BUDGET}};
})();
/* ENGINE END */
window.PDD_ENGINE=PDD;
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='pd-disagg', C=PDD.C;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const zh=()=>A.lang()==='zh';
const DEF=(text,tip)=>A.ui.def(text,tip);
const labLink=(id,t)=>A.shell.labLink?A.shell.labLink(id,t):'';
const labName=id=>{const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[id]; return l&&l.s?L(l.s):id;};
const labRef=id=>{const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; if(!l) return esc(id); return c.labStatus&&c.labStatus(l)==='live'?labLink(id):esc(L(l.s||{en:id,zh:id}));};   // live labs link, others are named (R6.39)
const M=t=>A.ui.math(t);
const J_=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const cite=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const M8B='Llama‑3.1‑8B';
/* number formats */
const f1=x=>A.fmt.num(x,1), f2=x=>A.fmt.num(x,2), fint=n=>A.fmt.int(n);
const fT=x=>x<1?(zh()?`${Math.round(x*1000)} 毫秒`:`${Math.round(x*1000)} ms`):(zh()?`${f2(x)} 秒`:`${f2(x)} s`);
const fS=x=>zh()?`${f1(x)} 秒`:`${f1(x)} s`;
const fP=x=>`${x>=1?100:x<=0?0:Math.min(99,Math.max(1,Math.round(x*100)))}%`;
const fR=g=>g>=8?tx('8 or more','8 或更多'):A.fmt.num(g,1);
const GB=b=>`${A.fmt.num(b/1e9,b<1e9?2:1)} GB`;
const GIB=b=>`${A.fmt.num(b/C.GiB,0)} GiB`;

/* ---------- the settings ---------- */
const BASE={kind:'A',mode:'co',n:8,p:6,budget:0,link:'nvl',rate:3};
const BUDGETS=[0,2048,512,128];
const cfgOf=s=>s.mode==='co'?{mode:'co',n:s.n,budget:s.budget}:{mode:'split',n:s.n,p:Math.min(s.p,s.n-1),link:s.link};
const setupName=(s,short)=>s.mode==='co'
  ?(s.budget?tx(`colocated (budget ${fint(s.budget)})`,`混在一起跑（预算 ${fint(s.budget)}）`):tx('colocated (no chunking)','混在一起跑（不切块）'))
  :tx(`${s.p} prefill + ${s.n-s.p} decode GPUs${s.link==='ib'?', across nodes':''}`,`${s.p} 张卡做 prefill、${s.n-s.p} 张做 decode${s.link==='ib'?'，跨机器传':''}`);
const KIND={A:{en:'chats, 10% long documents',zh:'聊天为主，10% 是长文档'},B:{en:'chats, 30% long documents',zh:'聊天为主，30% 是长文档'},C:{en:'chats only',zh:'只有聊天'},
  one:{en:'8 chats and one 32,768-token document',zh:'8 个聊天请求和 1 篇 32,768 个 token 的文档'},one0:{en:'8 chats',zh:'8 个聊天请求'}};

/* ---------- the guided parts ---------- */
const P4={kind:'A',n:4,rate:1.5};
const CH={kind:'B',n:8,rate:1.5};
const CH_START={mode:'split',p:2,budget:0,link:'nvl'};
const PRED=[{v:'gapT',t:{en:'The longest gap gets much shorter; the documents wait longer for their first token',zh:'最长的间隔短了很多；文档等首 token 等得更久'}},
  {v:'both',t:{en:'Both get better: shorter gaps and earlier first tokens',zh:'两个都变好：间隔更短，首 token 也更早'}},
  {v:'gap',t:{en:'The longest gap gets much shorter; the documents’ first token is unchanged',zh:'最长的间隔短了很多；文档的首 token 不变'}},
  {v:'none',t:{en:'The longest gap stays about the same: the prefill work is still there',zh:'最长的间隔差不多：prefill 的活一点没少'}}];
const PRED_RIGHT='gapT';
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let p3b=null, p5p=1, p6l='nvl', p7p=6, tried={}, saved=null, chSubmitted=false, chReveal=false, annT=null, stageT=0, sbTouched=false, userAct=false;
let spdUser=null;
const watched=new Set();
let PLAY={on:false,raf:0,last:0,ph:0,key:''}, phNow=Infinity, lastRunKey='', prevM=null, nPlays=0;

/* which run the stage shows */
function viewRun(){
  if(mode==='guided'){const id=STEPS[step].id;
    if(id==='mission') return {kind:'one0',s:{...BASE,n:1,mode:'co',budget:0},rate:0};
    if(id==='stall') return {kind:'one',s:{...BASE,n:1,mode:'co',budget:0},rate:0};
    if(id==='chunk') return {kind:'one',s:{...BASE,n:1,mode:'co',budget:p3b==null?0:p3b},rate:0};
    if(id==='predict') return {kind:P4.kind,s:{...BASE,n:4,mode:'co',budget:0},rate:P4.rate};
    if(id==='split') return {kind:P4.kind,s:{...BASE,n:4,mode:'split',p:p5p,link:'nvl'},rate:P4.rate};
    if(id==='kv') return {kind:P4.kind,s:{...BASE,n:4,mode:'split',p:2,link:p6l},rate:P4.rate};
    if(id==='ratio') return {kind:'A',s:{...BASE,n:8,mode:p7p===0?'co':'split',p:p7p||6,budget:p7p===0?512:0,link:'nvl'},rate:3};
    return {kind:'A',s:{...BASE,n:8,mode:'split',p:6,link:'nvl'},rate:3};}
  if(mode==='challenge') return {kind:CH.kind,s:{...BASE,...S,n:CH.n},rate:CH.rate};
  return {kind:S.kind,s:{...S},rate:S.rate};
}
const curRun=()=>{const v=viewRun(); return PDD.runOf(v.kind,cfgOf(v.s),v.rate);};
const isP=id=>mode==='guided'&&STEPS[step].id===id;
const seenP=id=>mode!=='guided'||step>=SIX(id);
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const mech=()=>mode==='guided'&&['mission','stall','chunk'].includes(STEPS[step].id);
const spdNow=()=>spdUser||'fast';   // a whole run plays in about 10 s by default (R3.9)
const RUN_SECS={slow:40,med:20,fast:10};

/* ---------- definitions (S4, S6) ---------- */
const TIP={
 prefill:()=>tx('Prefill: the step that reads the whole prompt and writes its KV cache. Hundreds or thousands of tokens go through the model at once, so the GPU’s arithmetic is the limit.','prefill：读完整个 prompt、写下它的 KV cache 的那一步。几百上千个 token 一起过模型，所以算力是瓶颈。'),
 decode:()=>tx('Decode: each step writes one more token for every request in the batch. Each step reads all the weights and the KV cache for only one token per request, so memory bandwidth is the limit.','decode：每一步给 batch 里的每个请求各写一个 token。每一步都要把全部权重和 KV cache 读一遍，却只算出每个请求的一个 token，所以带宽是瓶颈。'),
 ttft:()=>tx('Time to first token (TTFT): from the moment a request arrives to its first answer token. It includes the time it waits in a queue.','首 token 延迟（TTFT）：从请求到达，到它的第一个回答 token。包含排队的时间。'),
 gap:()=>tx('Gap: once the answer has started, how long the user waits for the next token. A request meets its goal only if its longest gap is short enough.','间隔：回答开始以后，用户等下一个 token 等了多久。达标看最长的那次。'),
 tpot:()=>tx('Time per output token (TPOT): once the answer has started, how long one token takes on average.','每个输出 token 的耗时（TPOT）：回答开始以后，平均多久出一个 token。'),
 goals:()=>tx('Meets its goals: a chat request gets its first token within 1 s and never waits more than 150 ms between two tokens; a long-document request gets its first token within 5 s.','达标：聊天请求首 token 在 1 秒内，并且最长的间隔不超过 150 毫秒；长文档请求首 token 在 5 秒内。'),
 slo:()=>tx('Service target (SLO): at least 90% of requests meet their goals.','服务目标（SLO）：至少 90% 的请求达标。'),
 goodput:()=>tx(`Goodput per GPU (DistServe’s definition): the highest rate, in requests per second, the service can take while at least 90% of requests meet their goals, divided by the number of GPUs. Here the rates tried go up in steps of 0.5 requests per second per GPU. Each rate runs once, with ${PDD.PER_GPU} requests per GPU; the shorter the run, the better a setup that slowly backs up looks.`,`每张卡的达标吞吐（goodput per GPU，DistServe 的定义）：至少 90% 的请求达标时，服务每秒最多能接多少个请求，再除以卡数。这里的速率按每张卡每秒 0.5 个请求一档往上试。每个速率只跑一次，每张卡 ${PDD.PER_GPU} 个请求；运行越短，慢慢堵起来的方案看起来越好。`),
 chunk:()=>tx('Chunked prefill: the prompt is cut into pieces, and each step runs at most a fixed number of tokens (the token budget; every decode counts 1). A long prompt is then prefilled over many steps, and each of those steps also carries every decoding request’s next token.','chunked prefill：把 prompt 切成几段，每一步最多算固定数量的 token（每步的 token 预算，每个 decode 占 1 个）。长 prompt 于是分很多步算完，每一步还顺带给正在 decode 的请求各出一个 token。'),
 cb:()=>tx('Continuous batching: at every step the engine runs all the requests it holds as one batch; new requests join at any step and finished ones leave. So one step can carry a new request’s prefill and other requests’ decodes.','continuous batching：引擎每一步都把手上所有请求放进同一批算，新请求随时加入，做完的随时离开。所以一步里可能既有新请求的 prefill，也有别的请求的 decode。'),
 colo:()=>tx('Colocated: every GPU does both prefill and decode (continuous batching, here with or without chunked prefill).','混在一起跑：每张卡既做 prefill 又做 decode（continuous batching，这里可以加 chunked prefill）。'),
 pd:()=>tx('P/D disaggregation: some GPUs only prefill, the others only decode; the KV cache a prefill GPU computes moves to a decode GPU.','P/D 分离：一部分卡只做 prefill，另一部分卡只做 decode，prefill 算出的 KV cache 传给 decode 的卡。'),
 xfer:()=>tx('KV transfer: moving one request’s KV cache from the prefill GPU to the decode GPU. For Llama‑3.1‑8B it is 128 KiB per prompt token.','KV 传输：把一个请求的 KV cache 从 prefill 的卡搬到 decode 的卡。Llama‑3.1‑8B 每个 prompt token 是 128 KiB。'),
 nvl:()=>tx('NVLink 4 on an H100 SXM: 450 GB/s in one direction (the 900 GB/s often quoted counts both). This lab sends at full speed; a measured copy between two GPUs of a node reaches about 88% of it (ml-engineering, on H200).','H100 SXM 的 NVLink 4：单向 450 GB/s（常说的 900 GB/s 是两个方向加起来）。这个 Lab 按满速传；实测两张卡之间的拷贝大约能到它的 88%（ml-engineering，在 H200 上测）。'),
 ib:()=>tx('Across nodes: a DGX H100 has one 400 Gb/s ConnectX-7 InfiniBand port per GPU, which is 50 GB/s in one direction. This lab sends at full speed.','跨机器：DGX H100 每张卡配一个 400 Gb/s 的 ConnectX-7 InfiniBand 端口，单向 50 GB/s。这个 Lab 按满速传。')
};
const SRC={
 distserve:'https://github.com/hao-ai-lab/hao-ai-lab.github.io/blob/9dbf98dacdbee4133aedc23726f43aad7286b996/content/blogs/distserve/index.md',
 vllm:'https://github.com/vllm-project/vllm/blob/ed7ae705a1f366c2008964af562a7543e91d934e/docs/features/disagg_prefill.md',
 sglang:'https://github.com/sgl-project/sglang/blob/6d8ddaf1d786c43cc84c6292f930c174c07e4b02/docs/docs/advanced_features/pd_disaggregation.mdx',
 dynamo:'https://github.com/ai-dynamo/dynamo/blob/9fcea9ea29c57e50fb6a3ec681be8cd4fcadb1e4/docs/fern/pages/developer-guide/advanced-customizations/conditional-disaggregation.md',
 mooncake:'https://github.com/kvcache-ai/Mooncake/blob/db6f14bd019d7fe08b2497e88046b72384af287f/README.md',
 net:'https://github.com/stas00/ml-engineering/blob/fb75494c0d9e536f048e27be2869a77fe65ad9f4/network/README.md',
 p_dist:'https://arxiv.org/abs/2401.09670', p_split:'https://arxiv.org/abs/2311.18677', p_moon:'https://arxiv.org/abs/2407.00079'
};
const dPre=t=>DEF(t,TIP.prefill()), dDec=t=>DEF(t,TIP.decode()), dTTFT=t=>DEF(t,TIP.ttft()), dGap=t=>DEF(t,TIP.gap());

/* numbers the copy quotes, all from the engine */
const R1=()=>PDD.runOf('one0',{mode:'co',n:1,budget:0},0);
const R2=()=>PDD.runOf('one',{mode:'co',n:1,budget:0},0);
const R3=b=>PDD.runOf('one',{mode:'co',n:1,budget:b},0);
const stallStep=run=>{let s=null; for(const g of run.G) for(const x of g.steps) if(x.nd>0&&x.pt>0&&(!s||x.t1-x.t0>s.t1-s.t0)) s=x; return s;};
const docOf=run=>run.R.find(r=>r.cls==='doc');
const lateChats=run=>{const d=docOf(run); return run.R.filter(r=>r.cls==='chat'&&r.arr>d.arr);};
const R4=()=>PDD.runOf(P4.kind,{mode:'co',n:4,budget:0},P4.rate);
const R5=p=>PDD.runOf(P4.kind,{mode:'split',n:4,p,link:'nvl'},P4.rate);
const R6=l=>PDD.runOf(P4.kind,{mode:'split',n:4,p:2,link:l},P4.rate);
const XB=n=>n*C.KV_TOK, XT=(n,l)=>n*C.KV_TOK/C.LINK[l];
const P7=[1,2,3,4,5,6,7];
const gpA=c=>PDD.goodput('A',c).g;
const P7CFG=()=>[{k:'co0',c:{mode:'co',n:8,budget:0}},{k:'co512',c:{mode:'co',n:8,budget:512}}].concat(P7.map(p=>({k:'p'+p,c:{mode:'split',n:8,p,link:'nvl'}})));
const ok9=run=>tx(`${run.m.all.ok} of ${run.m.all.n} ${run.m.all.ok===1?'meets its':'meet their'} goals`,`${run.m.all.n} 个里 ${run.m.all.ok} 个达标`);

const STEPS=[
 {id:'mission',title:{en:'Two kinds of work on one GPU',zh:'一张卡上的两种活'},lead:{en:'Prefill and decode share every step.',zh:'prefill 和 decode 挤在同一步里。'},
  body:()=>{const r=R1(), w=watchOf(r), front=gapAt(w).gi<=3;   // the setup in a few short sentences (owner review 2026-10-14: “15 秒内就懂这个 lab 的设定”)
   return tx(`<p>One H100 serves ${M8B} to 8 chats. Each request runs ${dPre('prefill')} (its whole prompt in one go), then ${dDec('decode')} (one token per step). With ${DEF('continuous batching',TIP.cb())}, both share the same steps.</p><p>A chat’s goals: its ${dTTFT('first token')} within 1 s, then ${dGap('gaps')} of at most 150 ms between tokens. The service target: ${DEF('at least 90% of requests meet their goals',TIP.slo())}.</p><p>Here a chat waited at most ${fT(r.m.chat.gap_max)} for its next token (the box ${front?'at the start of':'in'} request ${w.id+1}’s row): that step also computed another chat’s prefill, so it took longer. What happens when a long document arrives?</p>`,
             `<p>一张 H100 跑 ${M8B}，服务 8 个聊天。每个请求先做 ${dPre('prefill')}（一次算完整个 prompt），再做 ${dDec('decode')}（一步出一个 token）。用了 ${DEF('continuous batching',TIP.cb())}，两种活挤在同一步里算。</p><p>聊天的目标：1 秒内出${dTTFT('首 token')}，之后两个 token 之间的${dGap('间隔')}不超过 150 毫秒。服务目标：${DEF('至少 90% 的请求达标',TIP.slo())}。</p><p>现在聊天最长等了 ${fT(r.m.chat.gap_max)}（请求 ${w.id+1} 那一行${front?'最前面':''}框出的一段）：那一步里还算了另一个聊天的 prefill，所以这一步变长。来一篇很长的文档会怎样？</p>`);}},
 {id:'stall',title:{en:'A long prompt stalls everyone',zh:'一个长 prompt 卡住所有人'},lead:{en:'A 32,768-token document arrives.',zh:'来了一篇 32,768 个 token 的文档。'},
  body:()=>tx(`<p>At 1.0 s, while the chats decode, a 32,768-token document arrives; its goal is a first token within 5 s. Without chunking, the engine puts its whole prefill into the next step, together with every chat’s next token.</p>`,
              `<p>1.0 秒时，聊天正在 decode，来了一篇 32,768 个 token 的文档，它的目标是 5 秒内出首 token。不切块的话，引擎把它的整个 prefill 放进下一步，和每个聊天的下一个 token 一起算。</p>`),
  obs:()=>{const r=R2(), s=stallStep(r);
   return tx(`<p><b>That step takes ${fT(s.t1-s.t0)} (red in the chart).</b></p><p>Every chat that was decoding waits for it, so the chats’ longest gap jumps from ${fT(R1().m.chat.gap_max)} to ${fT(r.m.chat.gap_max)}, far past the 150 ms goal.</p><p>A step ends only when all the work in it is done. The document’s prefill is a huge amount of arithmetic, so this step is long. ${ok9(r)}: only the document itself.</p>`,
              `<p><b>这一步要 ${fT(s.t1-s.t0)}（图里的红块）。</b></p><p>正在 decode 的聊天请求都得等它，聊天的最长间隔从 ${fT(R1().m.chat.gap_max)}跳到 ${fT(r.m.chat.gap_max)}，远远超过 150 毫秒的目标。</p><p>一步要把里面所有的活都做完才结束。文档的 prefill 要算的量非常大，所以这一步很长。${ok9(r)}，只有文档自己。</p>`);},always:true},
 {id:'chunk',title:{en:'Chunked prefill cuts the long prefill into pieces',zh:'chunked prefill：把长 prefill 切成小块'},lead:{en:'Pick a token budget per step.',zh:'选每步的 token 预算。'},expect:()=>p3b!=null&&p3b>0,
  body:()=>tx(`<p>The usual fix on one GPU is ${DEF('chunked prefill',TIP.chunk())} (${labLink('chunked-prefill')}): each step runs at most a fixed number of tokens, the token budget per step. The document is prefilled over many steps, each also carrying every chat’s next token. Prompts are computed in order of arrival: the next starts only after every chunk of the previous one is done.</p><p class="small">Pick a budget and watch the chats’ longest gap and the document’s first token.</p>`,
              `<p>一张卡上通常用 ${DEF('chunked prefill',TIP.chunk())}（见 ${labLink('chunked-prefill')}）：每一步最多算固定数量的 token，叫每步的 token 预算。文档分很多步做完 prefill，每一步还顺带给每个聊天出一个 token。prompt 按到达顺序算：前一个 prompt 的块全部算完，下一个才开始。</p><p class="small">选一个预算，看看聊天的最长间隔和文档的首 token 怎么变。</p>`)
   +A.ui.levers({aria:tx('Token budget per step','每步的 token 预算'),value:p3b,options:[2048,512,128,64].map(b=>({v:String(b),title:tx(`Budget of ${fint(b)} tokens per step`,`每步的 token 预算 ${fint(b)}`),desc:'',result:tried['b'+b]?tx(`longest gap ${fT(R3(b).m.chat.gap_max)}`,`最长间隔 ${fT(R3(b).m.chat.gap_max)}`):''}))}),
  obs:()=>{const b=p3b, r=R3(b), d=docOf(r), d0=docOf(R2()), late=lateChats(r), wr=watchOf(r), wl=late.some(x=>x.id===wr.id), no=late.length-1, noE=['no','one','two','three'][no]||String(no), noZ=['零','一','两','三'][no]||String(no), slow=d.first-d.arr>d0.first-d0.arr+0.2, gp=r.m.chat.gap_max;
   return tx(`<p><b>With a budget of ${fint(b)} tokens per step, the chats’ longest gap falls to ${fT(gp)}${gp>0.1?', within the goal but close to it':''}.</b></p><p>The document’s first token comes after ${fT(d.first-d.arr)}, ${slow?'much later than without chunking':'about as soon as without chunking'}.</p><p>${wl?`Request ${wr.id+1} in the chart arrives while the document prefills and queues behind it: its first token comes only after ${fT(wr.first-wr.arr)}, past the 1 s goal. The other ${noE} chats that arrive then miss it too.`:`The ${late.length} chats that arrive while the document prefills queue behind it and wait until its whole prefill is done, so all their first tokens miss the 1 s goal.`}</p><p>Chunking keeps the answers already under way from being stuck behind one huge step, but the budget pulls two ways. A large budget makes each step long, so the gaps grow. With a small one, every step still reads all the weights, so the document’s prefill takes many more steps and its first token comes later.</p>`,
              `<p><b>每步的 token 预算是 ${fint(b)} 时，聊天的最长间隔降到 ${fT(gp)}${gp>0.1?'，在目标之内，但已经很接近':''}。</b></p><p>文档的首 token 在 ${fT(d.first-d.arr)}后到达，${slow?'比不切块时晚得多':'和不切块时差不多'}。</p><p>${wl?`图里的请求 ${wr.id+1} 在文档 prefill 期间到达，排在文档后面，${fT(wr.first-wr.arr)}才出首 token，超过 1 秒的目标；同时新来的另外${noZ}个聊天也一样。`:`文档 prefill 期间新来的 ${late.length} 个聊天请求排在文档后面，要等文档的 prefill 全部算完才轮到自己，首 token 都超过了 1 秒的目标。`}</p><p>切块让正在回答的聊天不再被一大步卡住，可预算的大小两头为难。预算大，每一步长，间隔变长；预算小，每一步还是要把全部权重读一遍，文档的 prefill 要多走很多步，首 token 来得更晚。</p>`);}},
 {id:'predict',title:{en:'Predict: give prefill its own GPU',zh:'先猜：给 prefill 单独一张卡'},lead:{en:'4 GPUs: 1 prefill, 3 decode?',zh:'4 张卡：1 张 prefill、3 张 decode？'},gate:()=>!!pred,
  body:()=>{const r=R4(), s=stallStep(R2());
   return tx(`<p>Now 4 H100s, each with its own copy of the model, take 6 requests per second: nine in ten are chats, one in ten a long document. All 4 run colocated (each does both prefill and decode), without chunking. Each time a document arrives, the chats’ longest gap jumps again, up to ${fT(r.m.chat.gap_max)}.</p><p>A hint: a 32,768-token document takes ${fT(s.t1-s.t0)} of prefill on one GPU.</p><p>Next, 1 GPU will only prefill and 3 will only decode. The prefill GPU works through prompts first come, first served, and sends each finished prompt’s KV cache (what its prefill computed, which decode keeps reading) to a decode GPU. What happens to the chats’ longest gap and to the documents’ first token?</p>`+predHTML(),
             `<p>现在 4 张 H100 各放一份模型，每秒来 6 个请求：九成是聊天，一成是长文档。4 张卡都混在一起跑（每张卡既做 prefill 也做 decode），不切块。每来一篇文档，聊天的最长间隔就又跳上去，最长到 ${fT(r.m.chat.gap_max)}。</p><p>提示：一篇 32,768 个 token 的文档，在一张卡上做 prefill 要 ${fT(s.t1-s.t0)}。</p><p>接下来，让 1 张卡只做 prefill，3 张卡只做 decode。prefill 的卡按到达顺序处理 prompt，算完一个，就把它的 KV cache（prompt 算出的中间结果，decode 要接着用）传给一张 decode 的卡。聊天的最长间隔和文档的首 token 会怎样？</p>`+predHTML());}},
 {id:'split',title:{en:'Split the GPUs into two pools',zh:'把卡分成两组'},lead:{en:'Pick how many GPUs prefill.',zh:'选几张卡做 prefill。'},expect:()=>p5p!==1,
  body:()=>tx(`<p>This is ${DEF('P/D disaggregation',TIP.pd())}: the prefill GPUs run only prefill, the decode GPUs only decode, and a request’s KV cache moves from one to the other. A decode step never carries a prefill, so a long prompt can no longer stall anyone’s answer. The price: how many GPUs each pool gets has to follow its own workload.</p>`,
              `<p>这就是 ${DEF('P/D 分离',TIP.pd())}：做 prefill 的卡只做 prefill，做 decode 的卡只做 decode，请求的 KV cache 从前者传到后者。decode 的步里再也没有 prefill，长 prompt 就卡不住谁的回答了。代价是：两组卡各要几张，得按各自的工作量来定。</p>`)
   +`<div class="pdd-verdict" id="pdd-verdict">${verdictHTML()}</div><p class="small">${tx('Pick how the 4 GPUs are split and watch the share of requests that meet their goals.','选一种分法，看看达标的请求占多少。')}</p>`
   +A.ui.levers({aria:tx('Prefill and decode GPUs','prefill 和 decode 的卡数'),value:String(p5p),options:[1,2,3].map(p=>({v:String(p),title:tx(`${p} prefill + ${4-p} decode`,`${p} 张 prefill、${4-p} 张 decode`),desc:'',result:(tried['s'+p]||p===p5p)?fP(R5(p).m.all.ok_frac):''}))}),
  obs:()=>{const r=R5(p5p), g=fT(r.m.chat.gap_max);
   return tx(`<p><b>With ${p5p} prefill GPUs, prompts no longer queue for long: ${fP(r.m.all.ok_frac)} of requests meet their goals.</b></p><p>${p5p===3?`The chats’ longest gap is only ${g}. One decode GPU is enough here because decoding these answers needs little: each decode step reads the weights once for every request it holds.`:`The chats’ longest gap is only ${g}, and each pool now has enough GPUs for its work.`}</p>`,
             `<p><b>${p5p} 张卡做 prefill 时，prompt 不用久等：${fP(r.m.all.ok_frac)} 的请求达标。</b></p><p>${p5p===3?`聊天的最长间隔也只有 ${g}。这里 1 张 decode 的卡就够，因为这些回答 decode 起来不费力：每一步只读一遍权重，就给它手上的每个请求各出一个 token。`:`聊天的最长间隔也只有 ${g}，两组卡的数量都够用了。`}</p>`);}},
 {id:'kv',title:{en:'What moves: the KV cache',zh:'传过去的是 KV cache'},lead:{en:'Same node or across nodes?',zh:'同一台机器，还是跨机器？'},expect:()=>!!tried.ib,
  body:()=>{const doc=32768;
   return tx(`<p>The prefill GPU must send the prompt’s KV cache to the decode GPU: every decode step reads the KV cache of all the earlier tokens, and without it the decode GPU would have to redo the prefill. The prefill GPU computes the first token and sends it to the user; the second waits for the KV cache, so the transfer lands on each answer’s first gap.</p><p>For ${M8B} the KV cache is 128 KiB per token (${labLink('kv-cache')}), so a 32,768-token document’s is ${GIB(XB(doc))} (${GB(XB(doc))}):</p><ul class="pdd-links"><li>${DEF(tx('Same node, NVLink','同一台机器，NVLink'),TIP.nvl())}: 450 GB/s in one direction, ${fT(XT(doc,'nvl'))}.</li><li>${DEF(tx('Across nodes, InfiniBand','跨机器，InfiniBand'),TIP.ib())}: 50 GB/s in one direction, ${fT(XT(doc,'ib'))}.</li></ul><p class="small">Pick a link and watch the chats’ longest gap.</p>`+linkSeg('pdd-ic-link',p6l),
             `<p>prefill 的卡要把 prompt 的 KV cache 传给 decode 的卡：decode 每出一个 token 都要读前面所有 token 的 KV cache，不传就得把 prefill 重算一遍。第一个 token 由 prefill 的卡算出、直接给用户，第二个要等传完才能写，所以传输时间加在每个回答的第一个间隔上。</p><p>${M8B} 每个 token 的 KV cache 是 128 KiB（见 ${labLink('kv-cache')}），一篇 32,768 个 token 的文档就是 ${GIB(XB(doc))}（${GB(XB(doc))}）：</p><ul class="pdd-links"><li>${DEF('同一台机器，NVLink',TIP.nvl())}：单向 450 GB/s，要 ${fT(XT(doc,'nvl'))}。</li><li>${DEF('跨机器，InfiniBand',TIP.ib())}：单向 50 GB/s，要 ${fT(XT(doc,'ib'))}。</li></ul><p class="small">选一种连接，看聊天的最长间隔怎么变。</p>`+linkSeg('pdd-ic-link',p6l));},
  obs:()=>{const a=R6('nvl'), b=R6('ib');
   return tx(`<p><b>Across nodes, the chats’ longest gap grows from ${fT(a.m.chat.gap_max)} to ${fT(b.m.chat.gap_max)}, still within the goal.</b></p><p>With either link, ${fP(b.m.all.ok_frac)} of requests meet their goals.</p><p>A document’s KV cache takes ${fT(XT(32768,'ib'))} to cross, and each prefill GPU sends one at a time, so a chat whose prefill finishes right behind it waits for that transfer too.</p>`,
              `<p><b>跨机器时，聊天的最长间隔从 ${fT(a.m.chat.gap_max)}涨到 ${fT(b.m.chat.gap_max)}，仍在目标之内。</b></p><p>两种连接下，都是 ${fP(b.m.all.ok_frac)} 的请求达标。</p><p>一篇文档的 KV cache 跨机器要 ${fT(XT(32768,'ib'))}；每张 prefill 的卡一次只传一份，紧跟在它后面算完 prefill 的聊天也得等它传完。</p>`);}},
 {id:'ratio',title:{en:'How many GPUs for each pool?',zh:'两组卡各配几张？'},lead:{en:'8 GPUs: goodput per GPU.',zh:'8 张卡：每张卡的达标吞吐。'},expect:()=>!!tried.r,
  // short (owner review 2026-10-14), and the two winners named by what they win: the best split, and colocated with chunks beating every split
  body:()=>tx(`<p>8 H100s; nine requests in ten are chats, one in ten a long document. Setups are compared by ${DEF('goodput per GPU',TIP.goodput())}: the most requests per second each GPU can take while at least 90% of requests still meet their goals. The bar chart under the timeline shows it for every setup.</p><p class="small">n:m means n GPUs prefill and m decode; “colocated (budget 512)” means no groups: every GPU does both, with chunked prefill of at most 512 tokens per step. Pick one to watch it serve 24 requests per second (3 per GPU).</p>`+p7Seg(),
              `<p>8 张 H100，请求九成是聊天、一成是长文档。方案按${DEF('每张卡的达标吞吐',TIP.goodput())}比：至少 90% 的请求达标时，每张卡每秒最多能接几个请求。时间线下面的条形图列出每种方案的结果。</p><p class="small">n:m 表示 n 张卡做 prefill、m 张做 decode；“混在一起跑（预算 512）”是不分组，每张卡两种活都做，每步最多算 512 个 token（chunked prefill）。选一种，看它每秒接 24 个请求（每张卡 3 个）时怎么跑。</p>`+p7Seg()),
  obs:()=>{const done=P7CFG().every(x=>PDD.goodputReady('A',x.c)); if(!done) return `<p>${tx('Computing the sweep…','正在计算……')}</p>`;
   const g=P7.map(p=>gpA({mode:'split',n:8,p,link:'nvl'})), best=Math.max(...g), bp=P7[g.indexOf(best)], co=gpA({mode:'co',n:8,budget:512}), co0=gpA({mode:'co',n:8,budget:0});
   const crun=curRun(), cr=crun.m, cOk=cr.all.ok_frac>=0.9, cGap=cr.chat.gap_max>PDD.GOALS.chat.gap, cw=watchOf(crun), cga=cw?gapAt(cw):{gi:0,g:0}, co512=setupName({mode:'co',budget:512}),
    cLe=p7p===0?`“${co512}”`:`${p7p}:${8-p7p}`, cLz=p7p===0?`“${co512}”`:` ${p7p}:${8-p7p} `;
   const room=tx('In this lab each request sets aside room on a decode GPU for its prompt and its whole answer. When there is none, a request whose prefill is done waits on the prefill GPU.','在这个 Lab 里，每个请求一开始就在 decode 的卡上预留 prompt 加整个回答的 KV cache 空间。没有空位时，算完 prefill 的请求只能在 prefill 的卡上等着。');
   // one key number per paragraph (R6.20): the stage's metric cards and bar chart carry the rest
   const stopE=cga.gi===1?`answers stop for ${fT(cga.g)} after their first token`:`answers stop for up to ${fT(cga.g)}`, stopZ=cga.gi===1?`回答出了第一个 token 后停了 ${fT(cga.g)}`:`回答中途最长停了 ${fT(cga.g)}`;
   const cur=tx(`<p>${cOk?`Your pick, ${cLe}, has ${fP(cr.all.ok_frac)} of requests meet their goals.`:`Your pick, ${cLe}, misses the target. ${cGap?`${DEF('Too few decode GPUs: KV cache fills their memory',room)}, so ${stopE}.`:'Too few prefill GPUs: prompts queue, so first tokens come late.'}`}</p>`,
                `<p>${cOk?`你选的${cLz}让 ${fP(cr.all.ok_frac)} 的请求达标。`:`你选的${cLz}没达标。${cGap?`${DEF('decode 的卡太少，显存被 KV cache 占满',room)}，${stopZ}。`:'prefill 的卡太少，prompt 要排队，首 token 来得晚。'}`}</p>`);
   return cur+tx(`<p><b>Among the splits, ${bp}:${8-bp} is best, at about ${fR(best)} requests per second per GPU; ${co512} reaches ${fR(co)}, more than any split.</b></p><p>Colocated, the decode tokens ride along with prefill chunks almost for free; split, the decode GPUs leave most of their arithmetic idle. In this lab, what the split buys is not throughput but no budget to pick: a long prompt never lands in a decode step.</p>`,
              `<p><b>分法里 ${bp}:${8-bp} 最好，每张卡每秒约 ${fR(best)} 个；${co512}能到 ${fR(co)} 个，比任何分法都多。</b></p><p>混在一起跑时，decode 的 token 搭着 prefill 的块一起算，几乎不多花时间；分开后，decode 卡的算力大半闲着。在这个 Lab 里，分开跑换来的不是吞吐，而是不用挑预算：长 prompt 进不了 decode 的步。</p>`);}},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'When to split, and what it costs.',zh:'什么时候分开，代价是什么。'},
  body:()=>{const s=stallStep(R2()), r5=R5(1), st=fT(s.t1-s.t0);
   return tx(`<div class="obs"><div class="eyebrow">What to remember</div><ol class="pdd-rem"><li><p><b>A long prefill in the same step stalls every answer.</b> Chunked prefill with the right budget keeps the stall within the goal; with its own GPUs, prefill never enters a decode step, so there is no budget to pick.</p><p>A 32,768-token document prefilled in one step took ${st}, and every chat on that GPU waited that long for its next token.</p></li><li><p><b>Size the two pools from their own work.</b> Count prefill tokens per second against what the prefill GPUs can compute, and the KV cache room the answers in flight need against the decode GPUs’ memory.</p><p>For example, on 4 GPUs with one long document in ten, give prefill only 1 GPU.</p><p>The gaps stay short, but every prompt queues for that one GPU, and only ${fP(r5.m.all.ok_frac)} of requests meet their goals.</p></li><li><p><b>The KV transfer is small next to the prefill.</b> A 32,768-token document takes ${st} to prefill.</p><p>Its KV cache, ${GIB(XB(32768))}, crosses a node’s NVLink in ${fT(XT(32768,'nvl'))}.</p><p>Across nodes it takes ${fT(XT(32768,'ib'))}.</p></li></ol></div>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Splitting is always better.” In this lab, colocated serving (budget 512) takes more requests per GPU than any split, and its longest gap is no longer. vLLM’s docs say disaggregated prefill “DOES NOT improve throughput”; it lets you tune the first token and the gap separately and keeps the longest gaps under control. The same docs add that chunked prefill with a proper chunk size (here, the token budget) can do the same, but that the right value is hard to find in practice.</p><p>✗ “The KV transfer is too slow to be worth it.” Count the bytes: a 32,768-token document’s KV cache crosses a node’s NVLink in ${fT(XT(32768,'nvl'))}, far less than its prefill takes. Watch it only across nodes with a tight gap goal: it lengthens the first gap of each answer.</p></div>`+
     `<div class="obs"><div class="eyebrow">What real systems do</div><ul class="pdd-sys"><li>DistServe: puts prefill and decode on different GPUs and compares setups by goodput (${cite(SRC.distserve,'blog')}).</li><li>vLLM: disaggregated prefill is still experimental; prefill and decode run in separate instances, and a KV connector such as NIXL or Mooncake moves the KV cache (${cite(SRC.vllm,'docs')}).</li><li>SGLang: separate prefill and decode servers, so that prefill batches stop interrupting the decode batches in progress (${cite(SRC.sglang,'docs')}).</li><li>NVIDIA Dynamo: when most of a request’s KV cache is already on a decode GPU, that GPU runs the prefill itself (${cite(SRC.dynamo,'conditional disaggregation')}).</li><li>Mooncake, Kimi’s serving platform: its README says that under real workloads it lets Kimi handle 75% more requests while meeting its service targets, without saying compared with what (${cite(SRC.mooncake,'README')}).</li></ul><p>A split also helps where this lab does not reach: the two phases can choose different parallelism (DistServe’s argument), and the longest gaps stay under control with tight goals (vLLM’s). Next you can read ${labRef('multi-gpu-serving')} (one model on several GPUs) and the ${labRef('admission')} (who waits when GPUs run short).</p></div>`+A.shell.toScenarios(),
             `<div class="obs"><div class="eyebrow">要记住的</div><ol class="pdd-rem"><li><p><b>长 prefill 和 decode 在同一步里，会卡住所有回答。</b>chunked prefill 选对预算，就能把卡顿压到目标以内；给 prefill 单独的卡，长 prompt 根本进不了 decode 的步，不用挑预算。</p><p>32,768 个 token 的文档在一步里做 prefill 要 ${st}，这张卡上的每个聊天请求都要等这么久才出下一个 token。</p></li><li><p><b>两组卡各要几张，得按各自的工作量来定。</b>拿每秒要算的 prefill token 数，对比 prefill 那组卡的算力；拿正在生成的回答要的 KV cache 空间，对比 decode 那组卡的显存。</p><p>比如 4 张卡服务一成是长文档的流量，只给 prefill 1 张卡。</p><p>间隔很短，可所有 prompt 都排在这一张卡上，只有 ${fP(r5.m.all.ok_frac)} 的请求达标。</p></li><li><p><b>KV 传输比 prefill 本身快得多。</b>32,768 个 token 的文档，做 prefill 要 ${st}。</p><p>它的 KV cache 有 ${GIB(XB(32768))}，在同一台机器里 ${fT(XT(32768,'nvl'))}传完。</p><p>跨机器要 ${fT(XT(32768,'ib'))}。</p></li></ol></div>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “分开跑总是更好。”在这个 Lab 里，混在一起跑（预算 512）时，每张卡能接的请求比任何一种分法都多，最长的间隔也不比分开跑长。vLLM 的文档写着，P/D 分离“不提高吞吐”（DOES NOT improve throughput）；它的好处是首 token 和间隔可以分开调，并且管得住最长的间隔。同一页也写了，chunked prefill 的块大小（这里的 token 预算）选得合适也能做到，只是实际中很难找到合适的值。</p><p>✗ “KV 传输很慢，所以不划算。”按字节算一下：一篇 32,768 个 token 的文档，KV cache 在同一台机器里 ${fT(XT(32768,'nvl'))}就传完，比算出它的 prefill 短得多。跨机器、间隔目标又很严时才要留意：它会拉长每个回答的第一个间隔。</p></div>`+
     `<div class="obs"><div class="eyebrow">真实系统怎么做</div><ul class="pdd-sys"><li>DistServe：把 prefill 和 decode 放到不同的卡上，并用达标吞吐比较方案（${cite(SRC.distserve,'博客')}）。</li><li>vLLM：P/D 分离还是实验性功能；prefill 和 decode 在不同的实例里跑，KV cache 由 NIXL、Mooncake 这类传输组件搬过去（${cite(SRC.vllm,'文档')}）。</li><li>SGLang：把 prefill 和 decode 放到不同的服务里，免得 prefill 打断正在进行的 decode（${cite(SRC.sglang,'文档')}）。</li><li>NVIDIA Dynamo：一个请求的 KV cache 大多已经在某张 decode 卡上时，直接让那张卡自己做 prefill（${cite(SRC.dynamo,'conditional disaggregation')}）。</li><li>Mooncake（Kimi 的推理平台）：README 说，在 Kimi 的真实流量下，满足服务目标时能处理的请求多了 75%，没写和什么比（${cite(SRC.mooncake,'README')}）。</li></ul><p>分开跑还有这个 Lab 没模拟的好处：两个阶段可以各选不同的并行方式（DistServe 的论点），目标很严时也更容易管住最长的间隔（vLLM 的说法）。接着可以看${labRef('multi-gpu-serving')}（一个模型用好几张卡）和${labRef('admission')}（卡不够时谁等）。</p></div>`+A.shell.toScenarios());}}
];
function predHTML(){
  return `<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`+
   (pred?`<p class="small muted">${tx('Saved. You can still change it before the next part shows the result.','已记下，进入下一部分看结果之前还可以改。')}</p>`:'');
}
function verdictHTML(){
  if(!pred) return '';
  const right=pred===PRED_RIGHT, r=R5(1);
  return `<p class="${right?'okline':'badline'}">${right?tx('Your prediction was right: the longest gap gets much shorter, and the documents wait longer.','你猜对了：最长的间隔短了很多，文档等得更久。'):tx(`Your prediction: “${esc(L(PRED.find(o=>o.v===pred).t))}”. The answer: the longest gap gets much shorter, and the documents wait longer.`,`你的预测是“${esc(L(PRED.find(o=>o.v===pred).t))}”。答案是：最长的间隔短了很多，文档等得更久。`)}</p>`+
   `<p>${tx(`The chats’ longest gap fell to ${fT(r.m.chat.gap_max)}; nine in ten documents got their first token within ${fT(r.m.doc.ttft_p90)}, later than before.`,`聊天的最长间隔降到 ${fT(r.m.chat.gap_max)}；九成文档的首 token 在 ${fT(r.m.doc.ttft_p90)}之内到达，比原来晚。`)}</p>`+
   `<p>${tx('Why: every prompt now queues for this one GPU, first come, first served. A document takes over a second to prefill, and every document and chat behind it waits, so the chats’ first token comes late too.','原因：所有 prompt 现在都排在这 1 张卡上，按到达顺序一个一个算。一篇文档要算 1 秒多，排在它后面的文档和聊天都要等，所以聊天的首 token 也晚了很多。')}</p>`;
}
const linkSeg=(id,v)=>A.ui.seg({id,key:'p6l',label:tx('Link','连接'),options:[['nvl',tx('same node (NVLink)','同一台机器（NVLink）')],['ib',tx('across nodes (InfiniBand)','跨机器（InfiniBand）')]],value:v,inline:true});
const p7Seg=()=>A.ui.seg({id:'pdd-ic-p7',key:'p7',label:tx('Show the run of','显示这种方案的运行'),options:[['0',setupName({mode:'co',budget:512})]].concat(P7.map(p=>[String(p),`${p}:${8-p}`])),value:String(p7p),inline:true});

/* ---------- goodput sweeps run in the background, one setup per tick ---------- */
const gpQ=[]; let gpT=0;
function needGp(kind,c){if(PDD.goodputReady(kind,c)) return true; if(!gpQ.some(x=>x.kind===kind&&JSON.stringify(x.c)===JSON.stringify(c))) gpQ.push({kind,c}); if(!gpT) gpT=setTimeout(gpTick,30); return false;}
function gpTick(){gpT=0; const j=gpQ.shift(); if(!j) return; PDD.goodput(j.kind,j.c); if(gpQ.length) gpT=setTimeout(gpTick,20); else if(app) render();}

/* ---------- scenarios ---------- */
const PRESETS=[
 {id:'co0',s:{kind:'A',mode:'co',n:8,budget:0,rate:3},t:{en:'Colocated (no chunking)',zh:'混在一起跑（不切块）'},d:{en:'8 GPUs, 24 requests/s, 10% long documents',zh:'8 张卡，每秒 24 个请求，10% 是长文档'}},
 {id:'co512',s:{kind:'A',mode:'co',n:8,budget:512,rate:3},t:{en:'Colocated (budget 512)',zh:'混在一起跑（预算 512）'},d:{en:'the same traffic, chunked prefill',zh:'同样的流量，加 chunked prefill'}},
 {id:'split',s:{kind:'A',mode:'split',n:8,p:6,link:'nvl',rate:3},t:{en:'6 prefill + 2 decode GPUs',zh:'6 张卡做 prefill、2 张做 decode'},d:{en:'the same traffic, split in one node',zh:'同样的流量，在一台机器里分开跑'}},
 {id:'ib',s:{kind:'A',mode:'split',n:8,p:6,link:'ib',rate:3},t:{en:'The same split, across nodes',zh:'同样的分法，跨机器传'},d:{en:'KV caches cross InfiniBand at 50 GB/s',zh:'KV cache 走 InfiniBand，每秒 50 GB'}},
 {id:'chat',s:{kind:'C',mode:'co',n:8,budget:0,rate:3},t:{en:'Chats only, no chunking',zh:'只有聊天，不切块'},d:{en:'short prompts, colocated',zh:'prompt 都短，混在一起跑'}}];
const presetNow=()=>PRESETS.findIndex(p=>Object.keys(p.s).every(k=>S[k]===p.s[k])&&(S.mode==='co'||S.p===p.s.p));
function scenHTML(i){const p=PRESETS[i], r=PDD.runOf(p.s.kind,cfgOf({...BASE,...p.s}),p.s.rate), m=r.m;
  const a=fP(m.all.ok_frac), g=fT(m.chat.gap_max), ct=fT(m.chat.ttft_p90), ct5=fT(PDD.runOf('A',{mode:'co',n:8,budget:512},3).m.chat.ttft_p90);
  const T={
   co0:tx(`<p><b>${a} of requests meet their goals.</b></p><p>Each document’s prefill runs in one step, so the chats on that GPU wait up to ${g} for their next token.</p>`,`<p><b>${a} 的请求达标。</b></p><p>每篇文档的 prefill 在一步里做完，同一张卡上的聊天请求要等最多 ${g}才出下一个 token。</p>`),
   co512:tx(`<p><b>${a} meet their goals; the longest gap is ${g}.</b></p><p>Every step stays short, and the decode tokens ride along with the prefill chunks.</p>`,`<p><b>${a} 达标，最长的间隔 ${g}。</b></p><p>每一步都很短，decode 的 token 搭着 prefill 的块一起算。</p>`),
   split:tx(`<p><b>${a} meet their goals; the longest gap is ${g}.</b></p><p>No decode step carries a prefill. But the prefill that 8 GPUs shared now runs on 6, so chats often queue behind a document’s prefill and get their first token much later than when colocated.</p><p>Nine in ten chats get their first token within ${ct}; colocated with chunking, within ${ct5}.</p>`,`<p><b>${a} 达标，最长的间隔 ${g}。</b></p><p>decode 的步里没有 prefill。可是原来 8 张卡分担的 prefill 现在只有 6 张卡做，聊天常常排在文档后面等 prefill，所以聊天的首 token 比混在一起跑时晚了很多。</p><p>九成聊天的首 token 在 ${ct}之内，混在一起跑并切块时是 ${ct5}。</p>`),
   ib:tx(`<p><b>${a} meet their goals; the longest gap is ${g}.</b></p><p>The slower link adds tens of milliseconds to the first gap of each answer. The first gap is longest when a chat’s KV cache crosses right behind a document’s.</p>`,`<p><b>${a} 达标，最长的间隔 ${g}。</b></p><p>连接慢了，每个回答的第一个间隔多出几十毫秒。聊天的 KV cache 排在一篇文档后面传时，第一个间隔最长。</p>`),
   chat:tx(`<p><b>${a} meet their goals; the longest gap is ${g}.</b></p><p>With short prompts, every prefill is small, so colocated serving without chunking is already fine.</p>`,`<p><b>${a} 达标，最长的间隔 ${g}。</b></p><p>prompt 都短，每次 prefill 的量都小，混在一起跑、不切块就够了。</p>`)};
  return T[p.id];}
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see which requests meet their goals.','选一个场景，看看哪些请求能达标。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="pdd-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change the split, the token budget, the link or the rate under ${J_('controls','Settings')}.`,`想改分法、token 预算、连接或请求速率，在“${J_('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;}

/* ---------- challenge ---------- */
const CH_OPTS=()=>{const o=[]; for(const b of [0,2048,512,128]) o.push({mode:'co',p:6,budget:b,link:'nvl'}); for(let p=1;p<8;p++) for(const l of ['nvl','ib']) o.push({mode:'split',p,budget:0,link:l}); return o;};
const chKey=s=>s.mode==='co'?'co'+s.budget:'p'+s.p+s.link;
const chRun=s=>PDD.runOf(CH.kind,cfgOf({...BASE,...s,n:CH.n}),CH.rate);
const chOk=s=>chRun(s).m.all.ok_frac>=PDD.TARGET-1e-9;
const chPick=()=>({mode:S.mode,p:S.p,budget:S.budget,link:S.link});
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Meet the service target on 8 H100s at 12 requests per second, three in ten long documents','让 8 张 H100 在每秒 12 个请求、三成是长文档时达到服务目标')}</h3>`+
   tx(`<p>8 H100s serve ${M8B}, one copy of the model per GPU. They can all sit in one node, or be spread over two nodes.</p><p>A new product sends 12 requests per second; three in ten are long documents.</p><p>A document’s prompt has 16K to 32K tokens.</p><ul class="pdd-sys"><li>A chat needs its first token within 1 s and no gap over 150 ms.</li><li>A document needs its first token within 5 s.</li><li>At least 90% of requests must meet their goals.</li></ul><p>Choose colocated serving with a token budget per step, or split the GPUs into two groups and choose the link. Among the setups that pass, pick the one with the highest goodput per GPU: the highest request rate per GPU at which it still meets the target. The higher it is, the more traffic can grow before the setup stops passing.</p>`,
      `<p>8 张 H100 跑 ${M8B}，每张卡一份模型。可以都放在一台机器里，也可以分到两台机器上。</p><p>一个新产品每秒发来 12 个请求，三成是长文档，prompt 有 16K 到 32K 个 token。</p><ul class="pdd-sys"><li>聊天要在 1 秒内拿到首 token，间隔不超过 150 毫秒。</li><li>文档要在 5 秒内拿到首 token。</li><li>至少 90% 的请求要达标。</li></ul><p>你可以让卡混在一起跑并选每步的 token 预算，也可以把卡分成两组并选连接。在达标的方案里，选每张卡的达标吞吐最高的那个。达标吞吐是这个方案还能达到服务目标时，每张卡每秒最多能接的请求数。它越高，流量涨得再多一些，这个方案也还能达标。</p>`)+
   A.ui.seg({id:'pdd-ic-chm',key:'chm',label:tx('How to serve','怎么跑'),options:[['co',tx('colocated','混在一起跑')],['split',tx('split','分开跑')]],value:S.mode,inline:true})+
   `<div id="pdd-ch-co"${S.mode==='co'?'':' hidden'}>`+A.ui.seg({id:'pdd-ic-chb',key:'chb',label:tx('Token budget per step','每步的 token 预算'),options:[['0',tx('no chunking','不切块')],['2048','2,048'],['512','512'],['128','128']],value:String(S.budget),inline:true})+`</div>`+
   `<div id="pdd-ch-sp"${S.mode==='split'?'':' hidden'}>`+A.ui.seg({id:'pdd-ic-chp',key:'chp',label:tx('Prefill GPUs (the rest decode)','做 prefill 的卡（其余做 decode）'),options:[1,2,3,4,5,6,7].map(p=>[String(p),String(p)]),value:String(S.p),inline:true})+
   A.ui.seg({id:'pdd-ic-chl',key:'chl',label:tx('Link','连接'),options:[['nvl',tx('same node','同一台机器')],['ib',tx('across nodes','跨机器')]],value:S.link,inline:true})+`</div>`+
   `<div class="goal mono pdd-goal" id="pdd-ch-g"><span>${tx('Requests meeting their goals','达标的请求')}</span><span><b id="pdd-ch-ok"></b>${tx(' (goal: at least 90%)','（目标：至少 90%）')}</span></div><div class="progress" id="pdd-ch-pr" aria-hidden="true"><i></i></div>`+
   submit+`<div id="pdd-ch-res" class="pdd-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chSubmitted) return '';
  const pick=chPick(), ok=chOk(pick), r=chRun(pick).m, opts=CH_OPTS(), feas=opts.filter(chOk);
  for(const o of feas) needGp(CH.kind,cfgOf({...BASE,...o,n:CH.n}));
  const ready=feas.every(o=>PDD.goodputReady(CH.kind,cfgOf({...BASE,...o,n:CH.n})));
  const gpOf=o=>ready?PDD.goodput(CH.kind,cfgOf({...BASE,...o,n:CH.n})).g:0;
  let head;
  if(!ok){const ttftBad=r.chat.ttft_p90>PDD.GOALS.chat.ttft||(r.doc&&r.doc.ttft_p90>PDD.GOALS.doc.ttft), gapBad=r.chat.gap_max>PDD.GOALS.chat.gap;
    const hint=pick.mode==='co'&&!pick.budget?tx('Each document’s prefill runs in one step and stalls every chat on that GPU.','每篇文档的 prefill 在一步里做完，卡住同一张卡上的所有聊天。')
      :pick.mode==='co'?tx('With a budget this small, a document’s prefill is cut into many steps, and every step reads all the weights again, so prefill is too slow.','每步的 token 预算这么小，文档的 prefill 要分成很多步，每一步都要把权重重读一遍，所以 prefill 太慢。')
      :ttftBad&&!gapBad?tx('With 30% long documents there is a lot of prefill work; the prefill GPUs cannot keep up, so prompts queue for them.','三成请求是长文档，prefill 的活很多，prefill 的卡忙不过来，prompt 都在排队等它们。')
      :tx('Too few decode GPUs: their memory fills with KV cache. Requests whose prefill is done wait on the prefill GPUs; users already have the first token, but the next tokens wait for room on a decode GPU, so answers stop after the first token.','decode 的卡太少，显存被 KV cache 占满。算完 prefill 的请求只能在 prefill 的卡上等着；用户已经拿到第一个 token，可后面的 token 要等 decode 的卡腾出空间，所以回答停在第一个 token 之后。');
    head=`<p class="badline"><b>${tx(`Not met: ${fP(r.all.ok_frac)} of requests meet their goals.`,`没达标：${fP(r.all.ok_frac)} 的请求达标。`)}</b>${tx(' ','')}${hint}</p>`+(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);}
  else if(!ready) head=`<p class="okline"><b>${tx(`Met: ${fP(r.all.ok_frac)} of requests meet their goals.`,`达标：${fP(r.all.ok_frac)} 的请求达标。`)}</b>${tx(' ','')}${tx('Comparing the setups that pass…','正在比较能达标的方案……')}</p>`;
  else {const best=Math.max(...feas.map(gpOf)), mine=gpOf(pick), bo=feas.filter(o=>gpOf(o)===best);
    const tie=bo.filter(o=>chKey(o)!==chKey(pick)), tn=tie.length?esc(setupName({...tie[0],n:8})):'', edge=mine<=CH.rate+1e-9;
    head=mine>=best?`<p class="okline"><b>${tx(`Met: ${fP(r.all.ok_frac)} of requests meet their goals, and this setup has the highest goodput per GPU${tie.length?` (tied with “${tn}”)`:''}.`,`达标：${fP(r.all.ok_frac)} 的请求达标，而且每张卡的达标吞吐最高${tie.length?`（和“${tn}”并列）`:''}。`)}</b></p>`+
       `<p>${tx(`It still passes when traffic grows to ${fR(mine)} requests per second per GPU; today it is ${fR(CH.rate)}.`,`流量涨到每张卡每秒 ${fR(mine)} 个请求时，它仍能达标；现在是 ${fR(CH.rate)} 个。`)}</p>`
     :`<p class="okline"><b>${tx(`Met: ${fP(r.all.ok_frac)} of requests meet their goals.`,`达标：${fP(r.all.ok_frac)} 的请求达标。`)}</b></p><p>${tx(`But “${esc(setupName({...bo[0],n:8}))}” has a higher goodput per GPU.`,`不过“${esc(setupName({...bo[0],n:8}))}”的达标吞吐更高。`)}</p>`+
       `<p>${tx(`It still passes at ${fR(best)} requests per second per GPU; yours at ${fR(mine)}${edge?', so a little more traffic and yours misses the target':''}.`,`它在每张卡每秒 ${fR(best)} 个请求时仍能达标，你的是 ${fR(mine)} 个${edge?'，流量再多一点就达不到目标了':''}。`)}</p>`;}
  if(!ok&&!chReveal) return head;
  const rows=opts.map(o=>{const m=chRun(o).m, me=chKey(o)===chKey(pick), st=chKey(o)===chKey(CH_START);
    const pass=m.all.ok_frac>=PDD.TARGET-1e-9, g=pass&&ready?gpOf(o):null;
    return {label:setupName({...o,n:8}),value:m.all.ok_frac*100,text:fP(m.all.ok_frac),ok:pass,me,tags:[...(me?[{text:tx('your pick','你的选择'),me:true}]:[]),...(st?[{text:tx('initial setup','初始设置')}]:[]),...(g!=null?[{text:tx(`goodput ${fR(g)} per GPU`,`达标吞吐：每张卡每秒 ${fR(g)} 个`)}]:[])]};});
  const why=tx(`<p>The initial setup, 2 prefill + 6 decode, meets only ${fP(chRun(CH_START).m.all.ok_frac)}.</p><p>With three in ten requests long documents, prompts queue for the two prefill GPUs.</p><p>The highest goodput per GPU comes from colocated serving with a token budget: the decode tokens ride along with the prefill chunks, and all eight GPUs prefill. The splits that pass give prefill most of the GPUs, and the few decode GPUs leave most of their arithmetic idle.</p><p>Caveat: this lab gives each GPU its own copy of the model and lets a step read and compute fully at the same time; check your own engine’s step times before choosing.</p>`,
                `<p>初始设置是 2 张 prefill、6 张 decode，只有 ${fP(chRun(CH_START).m.all.ok_frac)} 达标。</p><p>三成请求是长文档，prompt 都在排队等那两张 prefill 的卡。</p><p>达标吞吐最高的是混在一起跑、加上每步的 token 预算：decode 的 token 搭着 prefill 的块一起算，八张卡都在做 prefill。能达标的分法要把大部分卡给 prefill，剩下的 decode 卡算力大半闲着。</p><p>注意：这个 Lab 让每张卡各放一份模型，并假设一步之内读显存和计算完全重叠；真要选方案，先在自己的引擎上量一量每一步的时间。</p>`);
  return head+A.ui.cmpChart({title:tx('Which setups meet the goal on 8 H100s at 12 requests per second?','8 张 H100、每秒 12 个请求时，哪些方案能达标？'),aria:tx('Share meeting goals for each setup','每种方案的达标占比'),rows,goal:90,goalLabel:tx('goal 90%','目标 90%'),max:100,brief:true})+why;
}

/* ---------- capability layer ---------- */
const EXITS=()=>({
 explain:[{q:tx('Why does a split setup keep the chats’ gaps short when a long document arrives?','为什么分开跑时，来了长文档，聊天的间隔还是很短？'),
   o:[tx('The decode GPUs never run a prefill, so no decode step has to wait for the document’s prefill.','decode 的卡从不做 prefill，所以 decode 的步不用等文档的 prefill。'),
      tx('The document’s prefill gets faster on a GPU of its own.','文档的 prefill 在单独的卡上算得更快。'),
      tx('The KV cache moves to the decode GPU, so the chats need less memory.','KV cache 传到 decode 的卡上，聊天需要的显存变少了。')],a:0,
   why:tx('A colocated step ends only when all the work in it is done; with a long prefill inside, every decoding request waits for it. In a split setup the decode steps hold only decodes.','混在一起跑时，一步要等里面所有的活都做完；步里有一个长 prefill，所有在 decode 的请求都得等它。分开跑时，decode 的步里只有 decode。'),
   n:[null,tx('The prefill takes as long on its own GPU; what changes is who waits for it.','prefill 在单独的卡上一样慢；变的是谁在等它。'),tx('Memory is not what stalls the chats; the long step is.','卡住聊天的不是显存，是那一长步。')]},
  {q:tx('In this lab, colocated serving with a budget of 512 tokens per step takes more requests per GPU than the best split. Why?','在这个 Lab 里，混在一起跑、每步的 token 预算是 512 时，每张卡能接的请求比最好的分法还多。为什么？'),
   o:[tx('A split pays the KV transfer on every token it generates.','分开跑时，每生成一个 token 都要传一次 KV cache。'),
      tx('A colocated step reads the weights once for both the prompt chunk and the decodes; a decode GPU reads all the weights for decode alone.','混在一起跑时，一步读一遍权重，prompt 的块和 decode 一起算；decode 专用的卡每一步都要为 decode 单独读一遍权重。'),
      tx('Prefill GPUs compute more slowly than colocated GPUs.','做 prefill 的卡算得比混在一起跑的卡慢。')],a:1,
   why:tx('Decode is limited by reading the weights, so the decode tokens ride along with a prefill chunk almost for free. A split gives up that sharing; in exchange there is no budget to choose, and the two pools can be tuned separately.','decode 的瓶颈是读权重，所以 decode 的 token 搭着 prefill 的块一起算，几乎不多花时间。分开跑放弃了这种共用，换来的是不用挑预算的大小，以及两组卡可以各自调参。'),
   n:[tx('Each request moves its KV cache once, after its prefill.','每个请求只在 prefill 之后传一次 KV cache。'),null,tx('Every GPU here is the same H100 with the same step model.','这里每张卡都是同样的 H100，用同一个步模型。')]}],
 transfer:[{q:tx(`A team moves its decode GPUs to another node, linked by InfiniBand at a ninth of NVLink’s one-direction speed. Inside one node, a 32,768-token document’s KV cache crossed in about ${fT(XT(32768,'nvl'))}. About how long does it take now?`,`一个团队把 decode 的卡搬到另一台机器，用 InfiniBand 连接，速度是 NVLink 单向的九分之一。在同一台机器里，一篇 32,768 个 token 的文档，KV cache 大约 ${fT(XT(32768,'nvl'))}传完。现在大约要传多久？`),
   o:[tx('About 10 ms: the bytes are the same','约 10 毫秒：字节数没变'),tx('About 90 ms','约 90 毫秒'),tx('About 1.5 s, as long as its prefill','约 1.5 秒，和它的 prefill 一样久')],a:1,
   why:tx(`A ninth of the speed means nine times the time: ${fT(XT(32768,'ib'))}. That is still far shorter than its prefill, so it matters mainly for the first gap of the answer.`,`速度是九分之一，时间就是九倍：${fT(XT(32768,'ib'))}。这还是比它的 prefill 短得多，主要影响回答的第一个间隔。`)},
  {q:tx(`A service on 4 H100s takes only short chats, colocated without chunking. Every request meets its goals, and the chats’ longest gap is ${fT(PDD.runOf('C',{mode:'co',n:4,budget:0},4).m.chat.gap_max)}. Should the team split prefill and decode?`,`一个服务用 4 张 H100，只接短聊天，混在一起跑、不切块。所有请求都达标，聊天的最长间隔是 ${fT(PDD.runOf('C',{mode:'co',n:4,budget:0},4).m.chat.gap_max)}。要不要把 prefill 和 decode 分开？`),
   o:[tx('Yes: a split always takes more requests per GPU.','要：分开跑，每张卡总能接更多请求。'),
      tx('No: every prefill is small, so no step stalls the answers; a split would add two pools to size and gain nothing here.','不要：每次 prefill 都很小，没有哪一步会卡住回答；分开只会多出两组卡要配，这里得不到好处。'),
      tx('Yes: inside one node the KV transfer is free.','要：同一台机器里，KV 传输不花时间。')],a:1,
   why:tx('A split pays off when long prefills would stall the decodes or when the two phases need different tuning. With short prompts the stall is too small to matter.','长 prefill 会卡住 decode，或者两个阶段需要分开调的时候，分开跑才划算。prompt 都短时，卡顿小到可以不管。'),
   n:[tx('With only short chats, colocated serving without chunking still meets the target at 8 requests per second per GPU, the highest rate this lab tries. A split takes no more, and adds two pools to size.','只有短聊天时，在这个 Lab 里混在一起跑、不切块，每张卡每秒 8 个请求（试过的最高速率）也都达标；分开跑不能多接请求，还多了两组卡要配。'),null,tx('Small, not free: about 0.3 ms for 1,000 tokens over NVLink, and it still has to happen once per request.','不是不花时间，只是很少：1,000 个 token 走 NVLink 约 0.3 毫秒，而且每个请求都要传一次。')]}],
 practice:[{id:'inference-serving',d:tx('A chat backend with a time-to-first-token target per tier.','给不同档位设首 token 目标的聊天后端。')},{id:'inference-batching',d:tx('Batched GPU serving with a latency target.','带延迟目标的 GPU 批处理服务。')}]
});

/* ---------- copy: header ---------- */
const DESCS=()=>({guided:tx('8 parts, from one GPU stalled by a long prompt to a split of 8 GPUs.','8 个部分，从一张被长 prompt 卡住的卡，讲到 8 张卡怎么分。'),
  sandbox:tx('Common setups, and every setting.','几种常见的跑法，所有设置都能改。'),challenge:tx('Serve a document-heavy product on 8 H100s and meet the target.','8 张 H100 服务一个长文档多的产品，选一种跑法让它达标。')});
const OUTCOME=()=>tx('tell when splitting prefill and decode onto different GPUs pays off, size the two pools, and price the KV transfer.','判断什么时候值得把 prefill 和 decode 放到不同的卡上，配好两组卡的数量，并算出 KV 传输的代价。');
const FIDPOP=()=>tx(`<p><b>Toy run.</b> A seeded simulator of several H100 SXM GPUs serving ${M8B} in BF16, one copy of the model per GPU (no tensor parallelism; ${labRef('multi-gpu-serving')} covers it). Every step is priced by the step model of the other inference labs: the slower of its arithmetic at 50% of the dense peak (989.5 TFLOP/s) and its memory traffic at 60% of the 3.35 TB/s bandwidth. Both percentages are this lab’s assumptions.</p><p>Requests arrive at random (Poisson, seeded); chats have prompts of 400 to 1,600 tokens and answers of 120 to 320, documents 16,384 to 32,768 and 100 to 300. Colocated GPUs run continuous batching with chunked prefill as the ${labRef('serving-capstone')} does. Prefill GPUs take up to 8,192 tokens per step, first come, first served, and produce the first token; the KV cache then moves to the decode GPU with the most room, one transfer at a time per prefill GPU, at the link’s full speed (NVLink 4: 450 GB/s one way; InfiniBand: one 400 Gb/s NIC per GPU, 50 GB/s). A decode GPU sets aside room for the prompt and the whole answer (vLLM-style memory: 90% of 80 GiB, minus the weights (15.0 GiB) and 2 GiB overhead, leaves room for 450,907 tokens). Left out: a prefill GPU’s own memory limit, routing and network latency, kernel overheads that make small chunks slower in practice.</p><p>Goodput per GPU tries each rate once, with ${PDD.PER_GPU} requests per GPU; the shorter the run, the better a setup that slowly backs up looks.</p><p>Sources: ${cite(SRC.distserve,'DistServe blog')} (goodput), ${cite(SRC.vllm,'vLLM disaggregated prefill')}, ${cite(SRC.net,'ml-engineering network notes')} (NVLink 4, ConnectX-7).</p>`,
 `<p><b>玩具实跑。</b>一个带种子的模拟器：几张 H100 SXM 跑 BF16 的 ${M8B}，每张卡一份模型（不做张量并行，${labRef('multi-gpu-serving')}这个 Lab 讲它）。每一步的时间用其他推理 Lab 的步模型算：算力按稠密峰值 989.5 TFLOP/s 的 50% 算，带宽按 3.35 TB/s 的 60% 算，取两者中较慢的那个。这两个百分比是这个 Lab 的假设。</p><p>请求随机到达（泊松分布，带种子）。聊天的 prompt 400 到 1,600 个 token，回答 120 到 320 个；文档的 prompt 16,384 到 32,768 个，回答 100 到 300 个。混在一起跑的卡按 ${labRef('serving-capstone')}的做法跑 continuous batching 和 chunked prefill。prefill 的卡每步最多算 8,192 个 token，先到先算，并算出第一个 token；然后 KV cache 传给空间最多的那张 decode 卡，每张 prefill 的卡一次只传一份，按连接的满速算（NVLink 4 单向 450 GB/s；InfiniBand 每张卡一个 400 Gb/s 的网卡，50 GB/s）。decode 的卡为 prompt 和整个回答预留空间（按 vLLM 的显存算法：80 GiB 的 90%，减去权重（15.0 GiB）和 2 GiB 开销，能放 450,907 个 token）。没有模拟：prefill 卡自己的显存上限、路由和网络的延迟，以及实际中让小块变慢的 kernel 开销。</p><p>每张卡的达标吞吐：每个速率只跑一次，每张卡 ${PDD.PER_GPU} 个请求；运行越短，慢慢堵起来的方案看起来越好。</p><p>来源：${cite(SRC.distserve,'DistServe 博客')}（goodput），${cite(SRC.vllm,'vLLM 的 P/D 分离文档')}，${cite(SRC.net,'ml-engineering 的网络笔记')}（NVLink 4、ConnectX-7）。</p>`);
const PRIPOP=()=>tx(`<p>Prefill reads the whole prompt in one go and is limited by arithmetic; decode writes one token per step and is limited by memory bandwidth (${labLink('prefill-decode')}). Continuous batching puts both into the same steps (${labLink('continuous-batching')}), and chunked prefill gives each step a token budget (${labLink('chunked-prefill')}).</p>`,
 `<p>prefill 一次读完整个 prompt，瓶颈在算力；decode 一步写一个 token，瓶颈在带宽（见 ${labLink('prefill-decode')}）。continuous batching 把两种活放进同一步（见 ${labLink('continuous-batching')}），chunked prefill 给每一步定一个 token 预算（见 ${labLink('chunked-prefill')}）。</p>`);

/* ---------- views ---------- */
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):'P/D',title:l?L(l.q):tx('Why run prefill and decode on different GPUs?','为什么把 prefill 和 decode 放到不同的 GPU 上？'),
    minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: prefill and decode','前置知识：prefill 和 decode'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const spdCtl=()=>`<div class="pdd-speed"><span class="pdd-speed-l">${tx('Speed','速度')}</span>${`<div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="spd">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${spdNow()===v}">${x}</button>`).join('')}</div>`}</div>`;
function stageHTML(){
  const lg=(tip,cls,t)=>`<span data-lg="${cls}" data-tipx="${esc(tip)}"><i class="pdd-sw ${cls}"></i>${t}</span>`;
  return `<div class="panel stage pdd-stage" id="pdd-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="pdd-st-title"></div><div class="meter-sub" id="pdd-st-sub"></div></div><div class="stage-actions"><span class="status" id="pdd-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph pdd-graph" id="pdd-graph">`+
    // one picture, one clock (owner review 2026-10-14): the GPU lanes, the followed chat's own row under them and the time axis are one
    // SVG; the progress bar below is laid out by drawTimeline so its thumb sits under the playhead
    `<div class="pdd-blk"><div class="pdd-bh" id="pdd-tl-h">${tx('GPU timeline','每张卡的时间线')}</div><div class="pdd-tl" id="pdd-tl"></div>`+
     `<div class="pdd-scrub" id="pdd-scrub"><input type="range" class="pdd-prog" id="pdd-prog" min="0" max="1000" value="1000" aria-label="${esc(tx('Position in the run','播放位置'))}"></div>`+
     `<div class="pdd-run" id="pdd-run"><button type="button" class="btn sm" data-act="play" id="pdd-play"></button><button type="button" class="btn ghost sm" data-act="replay" id="pdd-replay">${tx('Replay','重放')}</button>${spdCtl()}</div></div>`+
    `<div class="legend-row pdd-legend">${lg(tx('A step with prompt tokens in it.','带着 prompt token 的一步。'),'pf',tx('prefill','prefill'))}${lg(tx('Steps that only decode: one token for each request on that GPU.','只做 decode 的步：给这张卡上的每个请求各出一个 token。'),'dc',tx('decode','decode'))}${lg(tx('A step that held decoding requests longer than the 150 ms gap goal.','让正在 decode 的请求等了超过 150 毫秒的一步。'),'st',tx('stalled decodes','decode 被卡住'))}${lg(tx('A KV cache moving from a prefill GPU to a decode GPU.','一份 KV cache 从 prefill 的卡传到 decode 的卡。'),'kv',tx('KV transfer','KV 传输'))}<span id="pdd-lg-w" data-tipx=""><i class="pdd-sw wr"></i><span id="pdd-lg-wt"></span></span></div>`+
    `<p class="pdd-wcap" id="pdd-w-h"></p>`+
    `<div class="pdd-blk" id="pdd-gpb" hidden><div class="pdd-gp" id="pdd-gp"></div></div>`+
   `</div>`+
   `<div class="metrics pdd-metrics" id="pdd-metrics"></div>`+
   `<details class="math" id="pdd-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="pdd-mathbody"></div></details></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="pdd-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="pdd-lock" hidden></p><div id="pdd-ctls">`+
   A.ui.seg({id:'pdd-ctl-kind',key:'kind',label:tx('Traffic','流量'),options:[['A',tx('10% documents','10% 文档')],['C',tx('chats only','只有聊天')]],value:S.kind,note:tx('Chats have prompts of 400 to 1,600 tokens and answers of 120 to 320; documents have prompts of 16,384 to 32,768 tokens and answers of 100 to 300. Seeded random arrivals; the lab’s own mix.','聊天的 prompt 有 400 到 1,600 个 token，回答 120 到 320 个；文档的 prompt 有 16,384 到 32,768 个 token，回答 100 到 300 个。到达时间带种子随机抽样，流量是这个 Lab 自己定的。')})+
   A.ui.seg({id:'pdd-ctl-n',key:'n',label:tx('GPUs','卡数'),options:[['4','4'],['8','8']],value:String(S.n)})+
   A.ui.range({id:'pdd-ctl-rate',key:'rate',label:tx('Requests per second, per GPU','每张卡每秒的请求数'),min:0.5,max:8,step:0.5,value:S.rate,out:String(S.rate),note:tx(`Each run has ${PDD.PER_GPU} requests per GPU.`,`每次运行每张卡 ${PDD.PER_GPU} 个请求。`)})+
   A.ui.seg({id:'pdd-ctl-mode',key:'mode',label:tx('How to serve','怎么跑'),options:[['co',tx('colocated','混在一起跑')],['split',tx('split','分开跑')]],value:S.mode})+
   `<div id="pdd-ctl-cow">`+A.ui.seg({id:'pdd-ctl-budget',key:'budget',label:tx('Token budget per step (chunked prefill)','每步的 token 预算（chunked prefill）'),options:BUDGETS.map(b=>[String(b),b?fint(b):tx('no chunking','不切块')]),value:String(S.budget),note:tx('Every decode counts 1 token of the budget, as vLLM’s max_num_batched_tokens does.','每个 decode 占预算里的 1 个 token，和 vLLM 的 max_num_batched_tokens 一样。')})+`</div>`+
   `<div id="pdd-ctl-spw">`+A.ui.range({id:'pdd-ctl-p',key:'p',label:tx('Prefill GPUs (the rest decode)','做 prefill 的卡（其余做 decode）'),min:1,max:S.n-1,step:1,value:Math.min(S.p,S.n-1),out:String(Math.min(S.p,S.n-1))})+
   A.ui.seg({id:'pdd-ctl-link',key:'link',label:tx('Link for the KV cache','KV cache 走的连接'),options:[['nvl',tx('same node, 450 GB/s','同一台机器，450 GB/s')],['ib',tx('across nodes, 50 GB/s','跨机器，50 GB/s')]],value:S.link,note:tx('Assumed at full speed, one transfer at a time per prefill GPU.','按满速算，每张 prefill 的卡一次只传一份。')})+`</div>`+
   `<button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`# prefill time and KV transfer (Llama-3.1-8B, BF16, H100)
KV_PER_TOK = 2 * 32 * 8 * 128 * 2      # K and V, 32 layers: bytes
LINK = {"nvlink": 450e9, "infiniband": 50e9}  # bytes/s, one way

def transfer_s(prompt_tokens, link):
    return prompt_tokens * KV_PER_TOK / LINK[link]

def prefill_s(prompt_tokens, flops_per_s=989.5e12 * 0.5):
    P, d, layers = 6.98e9, 4096, 32     # params in the layers
    lin = 2 * P * prompt_tokens
    att = 2 * layers * d * prompt_tokens * (prompt_tokens + 1)
    return (lin + att) / flops_per_s

for n in (1_000, 32_768):
    print(n, f"prefill {prefill_s(n):.3f} s",
          f"nvlink {transfer_s(n, 'nvlink')*1e3:.1f} ms",
          f"ib {transfer_s(n, 'infiniband')*1e3:.1f} ms")`;
const CODE_OUT=`# 1000 prefill 0.029 s nvlink 0.3 ms ib 2.6 ms
# 32768 prefill 1.494 s nvlink 9.5 ms ib 85.9 ms`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="pdd-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="pdd-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The two numbers that decide a split: how long a prompt’s prefill takes, and how long its KV cache takes to move.','决定要不要分开跑的两个数：一个 prompt 的 prefill 要多久，它的 KV cache 传过去要多久。')}</p><pre class="code" id="pdd-code">${esc(CODE+'\n'+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="pdd-prod">`+
   tx(`<li>vLLM: ${cite(SRC.vllm,'<code>docs/features/disagg_prefill.md</code>')}. Prefill and decode run in different vLLM instances; a KV connector (NixlConnector, MooncakeConnector, LMCache and others) moves the KV cache. Marked experimental.</li><li>SGLang: ${cite(SRC.sglang,'<code>pd_disaggregation.mdx</code>')}. Separate prefill and decode servers with Mooncake or NIXL as the transfer engine, including multi-node DeepSeek deployments.</li><li>NVIDIA Dynamo: ${cite(SRC.dynamo,'conditional disaggregation')}. The router sends a request through a prefill worker, or straight to a decode worker that runs the prefill itself, which helps when much of a long prompt’s KV cache is already on that worker.</li><li>Mooncake: ${cite(SRC.mooncake,'README')}. A KV-cache-centric disaggregated platform, Kimi’s serving platform.</li>`,
      `<li>vLLM：${cite(SRC.vllm,'<code>docs/features/disagg_prefill.md</code>')}。prefill 和 decode 在不同的 vLLM 实例里跑，由 KV connector（NixlConnector、MooncakeConnector、LMCache 等）传 KV cache。标注为实验性功能。</li><li>SGLang：${cite(SRC.sglang,'<code>pd_disaggregation.mdx</code>')}。prefill 和 decode 是两个服务，用 Mooncake 或 NIXL 做传输引擎，包括多机部署 DeepSeek。</li><li>NVIDIA Dynamo：${cite(SRC.dynamo,'conditional disaggregation')}。路由器让请求先过 prefill worker，或者直接交给 decode worker，由它自己做 prefill；长 prompt 的 KV cache 大多已经在那个 worker 上时，这样更好。</li><li>Mooncake：${cite(SRC.mooncake,'README')}。以 KV cache 为中心的分离式平台，是 Kimi 的推理平台。</li>`)+`</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.p_dist,'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving')}<span>${tx('Zhong et al., 2024, arXiv 2401.09670','Zhong 等，2024，arXiv 2401.09670')}</span></li>`+
   `<li>${cite(SRC.p_split,'Splitwise: Efficient Generative LLM Inference Using Phase Splitting')}<span>${tx('Patel et al., 2023, arXiv 2311.18677','Patel 等，2023，arXiv 2311.18677')}</span></li>`+
   `<li>${cite(SRC.p_moon,'Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving')}<span>${tx('Qin et al., 2024, arXiv 2407.00079','Qin 等，2024，arXiv 2407.00079')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="pdd-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="pdd-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="pdd-app m2" id="pdd-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="pdd-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- the stage ---------- */
const NS='http://www.w3.org/2000/svg';
function segsOf(run){
  if(run.segs) return run.segs;
  const gl=PDD.GOALS.chat.gap;
  run.segs=run.G.map(g=>{const out=[]; for(const s of g.steps){const c=(s.pt>0&&s.nd>0&&s.t1-s.t0>gl)?'s':(s.pt>0?'p':'d'); const l=out[out.length-1];
    if(l&&l.c===c&&c!=='s'&&s.t0-l.t1<1e-6) l.t1=s.t1; else out.push({t0:s.t0,t1:s.t1,c});} return out;});
  return run.segs;}
const watchOf=run=>{let w=null; for(const r of run.R) if(r.cls==='chat'&&(!w||PDD.gapOf(r)>PDD.gapOf(w)+1e-12)) w=r; return w;};
const niceStep=T=>{const c=[0.5,1,2,5,10,20,30]; for(const s of c) if(T/s<=8) return s; return 60;};
/* One picture, one clock (the owner's review of 2026-10-14: the chat's own row had its own time axis, so its tokens ran ahead of the
   GPU lanes, and the progress bar's dark track did not follow its thumb). The lanes, the followed chat's row under them, the playhead,
   the blue dot and the progress bar all read phNow on the same axis X(t). The followed chat (the one with the longest gap) is drawn in
   blue where it is: its prefill GPU, its KV transfer, its decode GPU. */
const THUMB=14;   // the progress bar's thumb is 14 px wide (pd-disagg.css), so its centre can sit exactly under the playhead
function pathOf(run,r){
  const split=r.dg!=null; let p0=r.first;
  for(const s of run.G[r.g].steps) if(s.ids.indexOf(r.id)>=0){p0=s.t0; break;}
  return {split,g:r.g,dg:split?r.dg:r.g,arr:r.arr,p0,p1:r.first,x0:split?r.x0:r.first,x1:split?r.x1:r.first,fin:r.fin};}
/* where the followed chat is at time t: null before it arrives and once it is done; its lane; or, during its KV transfer, both lanes */
function laneAt(w,t){
  if(t<w.arr-1e-9||t>=w.fin-1e-9) return null;
  if(!w.split||t<w.x0) return {lane:w.g};
  if(t<w.x1) return {from:w.g,to:w.dg,f:(t-w.x0)/Math.max(1e-9,w.x1-w.x0)};
  return {lane:w.dg};}
function gapAt(r){let gi=1, g=0; for(let i=1;i<r.toks.length;i++) if(r.toks[i]-r.toks[i-1]>g){g=r.toks[i]-r.toks[i-1]; gi=i;} return {gi,g};}
const px=x=>x.toFixed(1);   // an SVG coordinate
/* polish round 3 (2026-10-16, R7.12): on a wide stage more than four lanes are drawn shorter so part 7 fits a 1440 × 900 screen: the
   lane's height and the gap between lanes are CSS variables on the timeline's box (--pdd-rh, --pdd-gap, set in pd-disagg.css), read
   when it is drawn. Four lanes or fewer, and narrower stages (no variables), keep the heights they had. */
function laneH(host,n){const cs=getComputedStyle(host), rh=parseFloat(cs.getPropertyValue('--pdd-rh')), g=parseFloat(cs.getPropertyValue('--pdd-gap'));
  return n>4&&rh>0?[rh,g>=0?g:4]:[n<=1?30:n<=4?28:18,n<=4?6:4];}
/* the timeline is drawn for its box's width (and the lane heights above): the stage widens without a window resize when the page puts
   the step list beside it or leaves the primer, so it is drawn again when the stage's width changes */
let stRO=null, stW=0;
function watchStage(){if(stRO){stRO.disconnect(); stRO=null;} const st=app&&$('#pdd-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w; const run=curRun(); drawTimeline(run); drawWatch(run);}));
  stRO.observe(st,{box:'border-box'});}
function drawTimeline(run){
  const host=$('#pdd-tl',app); if(!host) return; const W=Math.max(260,Math.round(host.clientWidth||600)), n=run.G.length, lh=laneH(host,n);
  const LW=W<520?50:84, RH=lh[0], GAP=lh[1], T=Math.max(run.end,0.5), X=t=>LW+(W-LW-8)*t/T, rowY=i=>i*(RH+GAP)+2;
  const r=watchOf(run), w=r?pathOf(run,r):null, WY=n*(RH+GAP)+6, AY=WY+(r?46:0), H=AY+24;   // lanes, the chat's row, the axis
  const segs=segsOf(run), parts=[], body=[], lab=[];
  // the legend names only the marks this figure has (a first-time reader, 2026-10-14: no red block or KV line in part 1, so no key for them)
  const has={pf:false,dc:false,st:false,kv:run.R.some(q=>q.x0!=null)}; segs.forEach(a=>a.forEach(x=>{has[x.c==='d'?'dc':x.c==='s'?'st':'pf']=true;}));
  $$('.pdd-legend [data-lg]',app).forEach(e=>{e.hidden=!has[e.dataset.lg];});
  run.G.forEach((g,i)=>{const y=rowY(i), kind=g.kind==='pre'?'prefill':g.kind==='dec'?'decode':'';
    parts.push(`<rect class="pdd-row" x="${LW}" y="${y}" width="${W-LW-8}" height="${RH}"/>`);
    if(kind&&RH<22) parts.push(`<text class="pdd-rl" data-lane="${i}" x="${LW-6}" y="${y+RH/2+4}" text-anchor="end">${LW>=80?'GPU ':''}${i+1} (${g.kind==='pre'?'P':'D'})</text>`);
    else parts.push(`<text class="pdd-rl" data-lane="${i}" x="${LW-6}" y="${y+RH/2+(kind?-3:4)}" text-anchor="end">GPU ${i+1}</text>`+(kind?`<text class="pdd-rk" data-lane="${i}" x="${LW-6}" y="${y+RH/2+10}" text-anchor="end">${kind}</text>`:''));});
  const short=RH<22&&run.G.some(g=>g.kind==='pre'||g.kind==='dec'), th=$('#pdd-tl-h',app);   // short row labels get their key in the header (S9)
  if(th) th.textContent=short?tx('GPU timeline (P is a prefill GPU, D a decode GPU)','每张卡的时间线（P 是做 prefill 的卡，D 是做 decode 的卡）'):tx('GPU timeline','每张卡的时间线');
  run.G.forEach((g,i)=>{const y=rowY(i);
    for(const s of segs[i]){const x0=X(s.t0), wd=Math.max(0.8,X(s.t1)-x0);
      if(s.c==='d') body.push(`<rect class="pdd-dc" x="${px(x0)}" y="${px(y+RH*0.62)}" width="${px(wd)}" height="${px(RH*0.3)}"/>`);
      else body.push(`<rect class="pdd-${s.c==='s'?'st':'pf'}" x="${px(x0)}" y="${y+1}" width="${px(wd)}" height="${RH-2}"/>`);}});
  for(const q of run.R) if(q.x0!=null) body.push(`<line class="pdd-kv" x1="${px(X(q.x0))}" y1="${rowY(q.g)+RH-2}" x2="${px(X(q.x1))}" y2="${rowY(q.dg)+2}"/>`);
  let gi=1;
  if(w){
    // on the lanes: a blue outline while the chat is in a GPU's batch, a dashed line while it waits there, a blue line for its KV transfer
    const box=(i,a,b)=>{let x0=X(a), x1=X(b); if(x1-x0<4){const c=(x0+x1)/2; x0=c-2; x1=c+2;} return `<rect class="pdd-wbox" x="${px(x0)}" y="${rowY(i)+1}" width="${px(x1-x0)}" height="${RH-2}"/>`;};
    const wait=(i,a,b)=>X(b)-X(a)>1?`<line class="pdd-wq" x1="${px(X(a))}" y1="${rowY(i)+RH/2}" x2="${px(X(b))}" y2="${rowY(i)+RH/2}"/>`:'';
    body.push(wait(w.g,w.arr,w.p0));
    if(w.split) body.push(box(w.g,w.p0,w.p1)+wait(w.g,w.p1,w.x0)+`<line class="pdd-wkv" x1="${px(X(w.x0))}" y1="${rowY(w.g)+RH-1}" x2="${px(X(w.x1))}" y2="${rowY(w.dg)+1}"/>`+box(w.dg,w.x1,w.fin));
    else body.push(box(w.g,w.p0,w.fin));
    // its own row: the arrival, the wait for the first token (dashed), each token (a tick) and the longest gap (red past the goal)
    const ga=gapAt(r), bad=ga.g>PDD.GOALS.chat.gap+1e-9; gi=ga.gi;
    parts.push(`<text class="pdd-rk pdd-wlab" x="${LW-6}" y="${WY+18}" text-anchor="end">${tx('request','请求')}</text><text class="pdd-rl pdd-wlab" x="${LW-6}" y="${WY+31}" text-anchor="end">${r.id+1}</text>`);
    body.push(`<line class="pdd-warr" x1="${px(X(r.arr))}" y1="${WY+12}" x2="${px(X(r.arr))}" y2="${WY+30}"/>`+
      `<rect class="pdd-wt" x="${px(X(r.arr))}" y="${WY+16}" width="${px(Math.max(1,X(r.first)-X(r.arr)))}" height="10"/>`+
      `<rect class="${bad?'pdd-wg bad':'pdd-wg'}" x="${px(X(r.toks[gi-1]))}" y="${WY+12}" width="${px(Math.max(2,X(r.toks[gi])-X(r.toks[gi-1])))}" height="18"/>`+
      `<path class="pdd-tk" d="${r.toks.map(t=>`M${px(X(t))} ${WY+14}v14`).join('')}"/>`);
    lab.push(`<text class="pdd-wl" id="pdd-wl-arr" x="${px(X(r.arr))}" y="${WY+42}">${tx('arrives','到达')}</text>`,
      `<text class="pdd-wl" id="pdd-wl-ttft" x="${px(X(r.first)+3)}" y="${WY+42}">${tx('TTFT','首 token')} ${fT(r.first-r.arr)}</text>`,
      `<text class="pdd-wl${bad?' bad':''}" id="pdd-wl-gap" x="${px(X(r.toks[gi])+4)}" y="${WY+10}">${tx('longest gap','最长间隔')} ${fT(ga.g)}</text>`);}
  const st=niceStep(T), ax=[]; for(let t=0;t<=T+1e-9;t+=st){const x=X(t); ax.push(`<line class="pdd-ax" x1="${px(x)}" y1="${AY}" x2="${px(x)}" y2="${AY+4}"/><text class="pdd-at" x="${px(x>W-24?W-1:x)}" y="${AY+16}" text-anchor="${x>W-24?'end':'middle'}">${zh()?`${t} 秒`:`${t} s`}</text>`);}
  host.innerHTML=`<svg class="pdd-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(tx('Timeline of every GPU (prefill blocks, decode steps, KV transfers) and of one chat’s tokens','每张卡的时间线（prefill 块、decode 步、KV 传输）和一个聊天的每个 token'))}"><defs><clipPath id="pdd-clip"><rect id="pdd-cliprect" x="${LW}" y="0" width="0" height="${H}"/></clipPath></defs>${parts.join('')}<g clip-path="url(#pdd-clip)">${body.join('')}</g>${ax.join('')}${lab.join('')}<line class="pdd-ph" id="pdd-ph" x1="${LW}" x2="${LW}" y1="0" y2="${H}" visibility="hidden"/><circle class="pdd-wdot" id="pdd-wdot" r="4" cx="${LW}" cy="0" visibility="hidden"/></svg>`;
  host._X={LW,W,T,RH,GAP,w,r,gi};
  host._dash=!!w&&(X(w.p0)-X(w.arr)>3||(w.split&&X(w.x0)-X(w.p1)>3)||X(r.first)-X(r.arr)>3);   // a dashed wait the reader can see
  if(r) try{   // the chat's labels stay inside the plot: “arrives” and the first token on the bottom line, the longest gap on the top line
    const bb=id=>{const e=$('#'+id,host); return {e,wd:e.getBBox().width};}, a=bb('pdd-wl-arr'), b=bb('pdd-wl-ttft'), c=bb('pdd-wl-gap'), R=W-2;
    let xa=X(r.arr), xb=Math.max(X(r.first)+3,xa+a.wd+8); if(xb+b.wd>R){xb=R-b.wd; xa=Math.min(xa,xb-8-a.wd);} xa=Math.max(LW,xa); xb=Math.max(xb,xa+a.wd+8);
    a.e.setAttribute('x',px(xa)); b.e.setAttribute('x',px(xb));
    let xc=X(r.toks[gi])+4; if(xc+c.wd>R) xc=Math.max(LW,R-c.wd); c.e.setAttribute('x',px(xc));}catch(e){}
  const pr=$('#pdd-prog',app); if(pr){pr.parentNode.style.paddingLeft=`${LW-THUMB/2}px`; pr.style.width=`${px(X(run.end)-LW+THUMB)}px`;}
  moveHead(run);}
/* every frame: the clip (what has happened so far), the playhead, the dot on the lane the chat is on now (that lane's label in blue),
   the chat's labels once reached, and the progress bar's value and dark track; all from phNow */
function moveHead(run){const host=$('#pdd-tl',app); if(!host||!host._X) return;
  const k=host._X, X=t=>k.LW+(k.W-k.LW-8)*t/k.T, rowY=i=>i*(k.RH+k.GAP)+2, ph=Math.min(phNow,run.end), mid=ph<run.end-1e-9, xp=X(ph);
  const cr=$('#pdd-cliprect',host), pl=$('#pdd-ph',host), dot=$('#pdd-wdot',host);
  if(cr) cr.setAttribute('width',px(Math.max(0,xp-k.LW)));
  if(pl){pl.setAttribute('x1',px(xp)); pl.setAttribute('x2',px(xp)); pl.setAttribute('visibility',mid?'visible':'hidden');}
  const at=k.w&&mid?laneAt(k.w,ph):null;
  $$('[data-lane]',host).forEach(e=>{const i=+e.dataset.lane; e.classList.toggle('pdd-on',!!at&&(at.lane===i||at.from===i||at.to===i));});
  if(dot){if(at){const ya=at.lane!=null?rowY(at.lane)+k.RH/2:rowY(at.from)+k.RH-1, yb=at.lane!=null?ya:rowY(at.to)+1, cy=ya+(yb-ya)*(at.f||0);
      dot.setAttribute('cx',px(xp)); dot.setAttribute('cy',px(cy)); dot.setAttribute('visibility','visible');}
    else dot.setAttribute('visibility','hidden');}
  if(k.r){const sh=(id,t)=>{const e=$('#'+id,host); if(e) e.setAttribute('visibility',ph>=t-1e-9?'visible':'hidden');}; sh('pdd-wl-arr',k.r.arr); sh('pdd-wl-ttft',k.r.first); sh('pdd-wl-gap',k.r.toks[k.gi]);}
  const pr=$('#pdd-prog',app); if(pr){pr.value=String(Math.round(1000*Math.min(1,ph/run.end))); A.ui.setRange(pr);}}
/* under the legend: which chat is in blue */
function drawWatch(run){
  const hd=$('#pdd-w-h',app), lw=$('#pdd-lg-w',app), lt=$('#pdd-lg-wt',app); if(!hd) return; const r=watchOf(run);
  if(lw) lw.hidden=!r; if(!r){hd.textContent=''; return;}
  const id=r.id+1;
  const dash=!!($('#pdd-tl',app)||{})._dash;
  hd.textContent=tx(`Request ${id}: the chat that waited longest for a next token. The blue outline shows which GPU it is on${dash?', the dashed line where it queues':''}, and the bottom row shows each of its tokens.`,`请求 ${id}：等下一个 token 等得最久的那个聊天。蓝框是它在哪张卡上，${dash?'虚线是它在排队，':''}最下面一行是它的每个 token。`);
  if(lt) lt.textContent=tx(`request ${id}`,`请求 ${id}`);
  if(lw) lw.dataset.tipx=tx(`Where request ${id} is. Blue outline: it is in that GPU’s batch (prefill, then decode). Blue line: its KV cache moving to the decode GPU. Dashed: it waits. While the run plays, the blue dot is where it is now. Its own row at the bottom: each tick is one of its tokens, the box its longest gap.`,`请求 ${id} 在哪：蓝框是它在那张卡的批里（先 prefill，再 decode），蓝线是它的 KV cache 传到 decode 的卡，虚线是它在等。播放时，蓝点是它此刻的位置。最下面一行是它自己：每条竖线是它的一个 token，框出的是最长的那次间隔。`);}
function gpChartHTML(){
  const show=(mode==='guided'&&isP('ratio'))||mode==='sandbox'; if(!show) return '';
  if(mode==='guided'){const cf=P7CFG(), ready=cf.every(x=>needGp('A',x.c)); if(!ready) return `<p class="small muted pdd-gp-wait">${tx(`Computing goodput per GPU (${cf.filter(x=>PDD.goodputReady('A',x.c)).length} of ${cf.length})…`,`正在计算每张卡的达标吞吐（${cf.filter(x=>PDD.goodputReady('A',x.c)).length} / ${cf.length}）……`)}</p>`;
    const vals=cf.map(x=>gpA(x.c)), best=Math.max(...vals);
    return A.ui.cmpChart({title:tx('How many requests per second can each of 8 H100s take with at least 90% meeting their goals?','8 张 H100，至少 90% 达标时，每张卡每秒能接多少个请求？'),aria:tx('Goodput per GPU for each setup','每种方案每张卡的达标吞吐'),
      rows:cf.map((x,i)=>({label:x.k==='co0'||x.k==='co512'?setupName(x.c):tx(`${x.c.p} prefill + ${8-x.c.p} decode`,`${x.c.p} 张 prefill、${8-x.c.p} 张 decode`),value:vals[i],text:tx(`${fR(vals[i])}/s`,`每秒 ${fR(vals[i])} 个`),best:vals[i]===best,me:(x.k==='co512'&&p7p===0)||(x.c.p===p7p&&x.c.mode==='split')})),max:8,legend:false});}
  const c=cfgOf(S); if(!needGp(S.kind,c)) return `<p class="small muted">${tx('Computing goodput per GPU for this setup…','正在计算这个方案每张卡的达标吞吐……')}</p>`;
  const g=PDD.goodput(S.kind,c);
  return `<p class="small"><b>${tx(`Goodput per GPU: ${fR(g.g)} requests per second`,`每张卡的达标吞吐：每秒 ${fR(g.g)} 个请求`)}</b>${tx(` (${setupName(S)}, ${S.n} GPUs).`,`（${setupName(S)}，${S.n} 张卡）。`)}</p>`;}
function metricsDef(run){
  const m=run.m, playing=partial(run), v=x=>playing?'—':x;
  const out=[{k:'ok',label:tx('Requests meeting goals','达标的请求'),v:v(fP(m.all.ok_frac)),sub:tx('target: at least 90%','服务目标：至少 90%'),tip:TIP.slo()+tx(' ','')+TIP.goals(),pts:true},
   {k:'gap',label:tx('Chats: longest gap','聊天：最长的间隔'),v:v(fT(m.chat.gap_max)),sub:tx('goal: 150 ms or less','目标：不超过 150 毫秒'),tip:TIP.gap()},
   {k:'cttft',label:tx('Chats: first token (p90)','聊天：首 token（p90）'),v:v(fT(m.chat.ttft_p90)),sub:tx('goal: within 1 s','目标：1 秒内'),tip:TIP.ttft()+tx(' ','')+tx('p90: 9 in 10 requests get theirs within this time.','p90：九成请求在这个时间之内拿到。')}];
  if(m.doc) out.push({k:'dttft',label:tx('Documents: first token (p90)','文档：首 token（p90）'),v:v(fT(m.doc.ttft_p90)),sub:tx('goal: within 5 s','目标：5 秒内'),tip:TIP.ttft()+tx(' ','')+tx('p90: 9 in 10 documents get theirs within this time.','p90：九成文档在这个时间之内拿到。')});
  return out;}
const numOf=s=>{const m=String(s).replace(/,/g,'').match(/-?[\d.]+/); if(!m) return NaN; const x=parseFloat(m[0]); return /ms|毫秒/.test(s)?x/1000:/%/.test(s)?x:x;};
const shownPrev={};
function drawMetrics(run){
  const host=$('#pdd-metrics',app), defs=metricsDef(run), sg=defs.map(m=>m.k).join();
  if(host._sig!==sg){host._sig=sg; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  host.classList.toggle('pdd-wait',partial(run));
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    if(partial(run)) return;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&prev!=='—'){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,m.pts?{pts:true,dp:0}:{dp:2})||'';}
    clearTimeout(d._t);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on');},2600);} else if(userAct){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});}
/* Show the math */
const EQ=rows=>`<div class="pdd-eqs">${rows.map(([l,r,rel])=>`<span class="pdd-eq-l">${l?M(l):''}</span><span class="pdd-eq-e">${M(rel||'=')}</span><span class="pdd-eq-r">${M(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="pdd-mb"><p>${lead}</p>${body}${res?`<p class="pdd-mb-res">${res}</p>`:''}</div>`;
function mathHTML(){
  const s=stallStep(R2()), n=32768, lin=2*C.P_LAYERS*n, att=PDD.attn(0,n), fl=lin+2*C.P_HEAD*1+att, tf=fl/C.FS, by=XB(n), bw=C.W_READ+by, tm=bw/C.BS;
  const T=tx('\\text{step time}','\\text{一步的时间}'), Fh=tx('\\text{FLOPs}','\\text{计算量}'), tc=`t_{${tx('\\text{compute}','\\text{算}')}}`, tmm=`t_{${tx('\\text{memory}','\\text{读}')}}`;
  const e12=x=>`${A.fmt.num(x/1e12,1)} \\times 10^{12}`, sec=tx('\\text{s}','\\text{秒}'), msx=tx('\\text{ms}','\\text{毫秒}');
  /* goodput: the setup on the stage when it is a traffic mix, else part 7's 6 + 2 split; computed only while this block is open */
  const v=viewRun(), mix=['A','B','C'].includes(v.kind), gk=mix?v.kind:'A', gc=mix?cfgOf(v.s):{mode:'split',n:8,p:6,link:'nvl'}, gn=gc.n;
  const md=app&&$('#pdd-math',app), ready=PDD.goodputReady(gk,gc)||(md&&md.open&&needGp(gk,gc)), g=ready?PDD.goodput(gk,gc).g:null;
  const gname=mix?setupName(v.s):setupName({mode:'split',n:8,p:6,link:'nvl'});
  return MB(tx('<b>How long one step takes.</b> A step lasts as long as the slower of its arithmetic and its memory traffic; the GPU does both at once. Take the 32,768-token document’s prefill:','<b>一步要多久。</b>一步的时间取算和读两者中较慢的那个，GPU 一边读一边算。以 32,768 个 token 的文档做 prefill 为例：'),
     EQ([[T,`\\max(${tc},\; ${tmm})`],[tc,`\\frac{${Fh}}{989.5 \\times 10^{12} \\times 50\\%}`],[tmm,`\\frac{${tx('\\text{bytes}','\\text{字节}')}}{3.35 \\times 10^{12} \\times 60\\%}`]]),'')+
   MB(tx(`<b>The arithmetic.</b> ${A.fmt.num(C.P_LAYERS/1e9,2)}B is the parameters in the 32 layers, without the embedding table and the LM head:`,`<b>算的量。</b>${A.fmt.num(C.P_LAYERS/1e9,2)}B 是 32 层里的参数，不含 embedding 表和 LM head：`),
     EQ([[tx('\\text{linear layers}','\\text{线性层}'),`2 \\times ${A.fmt.num(C.P_LAYERS/1e9,2)}\\text{B} \\times 32768 = ${e12(lin)}`],['\\text{attention}',`2 \\times 32 \\times 4096 \\times 32768 \\times 32769`],['',e12(att)],[Fh,`${e12(lin)} + ${e12(att)} = ${e12(fl)}`],[tc,`\\frac{${e12(fl)}}{494.75 \\times 10^{12}} = ${f2(tf)}\\ ${sec}`]]),'')+
   MB(tx('<b>The memory traffic.</b> The step reads the weights once and writes the prompt’s KV cache:','<b>读的量。</b>这一步把权重读一遍，再写下 prompt 的 KV cache：'),
     EQ([[tx('\\text{bytes}','\\text{字节}'),`${A.fmt.num(C.W_READ/1e9,1)}\\ \\text{GB} + ${A.fmt.num(by/1e9,2)}\\ \\text{GB} = ${A.fmt.num(bw/1e9,1)}\\ \\text{GB}`],[tmm,`\\frac{${A.fmt.num(bw/1e9,1)}\\ \\text{GB}}{2.01 \\times 10^{12}\\ \\text{B/s}} = ${A.fmt.num(tm*1000,1)}\\ ${msx}`]]),
     tx(`${f2(tf)} s against ${A.fmt.num(tm*1000,1)} ms: the arithmetic decides. With the chats’ decodes in the same step, the step took ${fT(s.t1-s.t0)}.`,`${f2(tf)} 秒对 ${A.fmt.num(tm*1000,1)} 毫秒，由算的量决定。同一步里还有聊天的 decode，这一步一共 ${fT(s.t1-s.t0)}。`))+
   MB(tx('<b>How long the KV transfer takes.</b> The KV cache of the prompt moves at the link’s speed:','<b>KV 传输要多久。</b>prompt 的 KV cache 按连接的速度传：'),
     EQ([[tx('\\text{bytes}','\\text{字节}'),`32768 \\times 131072 = ${A.fmt.num(by/1e9,2)}\\ \\text{GB}`],[tx('\\text{NVLink}','\\text{NVLink}'),`\\frac{${A.fmt.num(by/1e9,2)}\\ \\text{GB}}{450\\ \\text{GB/s}} = ${A.fmt.num(XT(n,'nvl')*1000,1)}\\ ${msx}`],[tx('\\text{InfiniBand}','\\text{InfiniBand}'),`\\frac{${A.fmt.num(by/1e9,2)}\\ \\text{GB}}{50\\ \\text{GB/s}} = ${A.fmt.num(XT(n,'ib')*1000,1)}\\ ${msx}`]]))+
   MB(tx(`<b>Goodput per GPU.</b> The highest rate tried at which at least 90% of requests meet their goals, divided by the GPUs. For ${esc(gname)} on ${gn} GPUs:`,`<b>每张卡的达标吞吐。</b>试过的速率里，至少 90% 达标的最高速率，再除以卡数。以“${esc(gname)}”、${gn} 张卡为例：`),
     EQ([[tx('\\text{goodput per GPU}','\\text{每张卡的达标吞吐}'),`\\frac{\\max\\{\\lambda : ${tx('\\text{share meeting goals}','\\text{达标占比}')}(\\lambda) \\ge 90\\%\\}}{${tx('\\text{GPUs}','\\text{卡数}')}}`]].concat(g==null?[]:[['',`\\frac{${A.fmt.num(g*gn,0)}\\ ${tx('\\text{requests/s}','\\text{个/秒}')}}{${gn}} = ${A.fmt.num(g,1)}`,g>=8?'\\ge':'=']])),
     g==null?tx('Computing for this setup…','正在为这个方案计算……'):g>=8?tx('8 per GPU is the highest rate this lab tries.','每张卡每秒 8 个是这个 Lab 试过的最高速率。'):'');}

/* ---------- stage head, status ---------- */
/* the run is not shown whole: playing, or paused before its end; its numbers wait for the end, so they never get ahead of the picture */
const partial=run=>PLAY.on||phNow<run.end-1e-9;
function statusOf(run){const m=run.m;
  if(PLAY.on) return {cls:'',short:tx('playing','播放中'),txt:''};
  if(partial(run)) return {cls:'',short:tx('paused','已暂停'),txt:''};
  if(mode==='guided'&&isP('predict')&&!pred) return {cls:'',short:tx('colocated','混在一起跑'),txt:''};
  const ok=m.all.ok_frac>=PDD.TARGET-1e-9; return {cls:ok?'ok':'bad',short:ok?tx('target met','达到服务目标'):tx('target missed','没达到服务目标'),txt:''};}
function drawHead(run){
  const v=viewRun(), s=v.s, nreq=run.R.length;
  const what=v.kind==='one'||v.kind==='one0'?tx(`1 H100, ${L(KIND[v.kind])}`,`1 张 H100，${L(KIND[v.kind])}`)
    :tx(`${s.n} H100s, ${setupName(s)}, ${fint(nreq)} requests at ${A.fmt.num(v.rate*s.n,1)} per second (${L(KIND[v.kind])})`,`${s.n} 张 H100，${setupName(s)}，${fint(nreq)} 个请求，每秒 ${A.fmt.num(v.rate*s.n,1)} 个（${L(KIND[v.kind])}）`);
  const one=v.kind==='one'||v.kind==='one0';
  $('#pdd-st-title',app).textContent=(PLAY.on?tx('Playing: ','正在播放：'):partial(run)?tx('Paused: ','已暂停：'):tx('Whole run: ','整次运行：'))+what;
  $('#pdd-st-sub',app).textContent=v.kind==='one0'?tx('Prefill and decode run in the same steps. Time runs left to right.','prefill 和 decode 在同一步里算。时间从左往右。'):one?tx(`${setupName(s)}. Time runs left to right.`,`${setupName(s)}。时间从左往右。`):tx(`${M8B}, one copy per GPU. Time runs left to right.`,`${M8B}，每张卡一份模型。时间从左往右。`);
  const st=statusOf(run), b=$('#pdd-status',app); b.className='status '+st.cls; b.textContent=st.short;
  b.dataset.tipx=PLAY.on?tx('The run is playing; its numbers show when it ends.','正在播放，播完后显示结果。'):partial(run)?tx('The run is paused; its numbers show when it reaches the end.','已暂停，播到最后显示结果。'):TIP.slo();}
function drawRunCtl(run){
  const pb=$('#pdd-play',app); if(!pb) return;
  const atEnd=phNow>=run.end;
  pb.textContent=PLAY.on?tx('❚❚ Pause','❚❚ 暂停'):(atEnd?tx('▶ Play','▶ 播放'):tx('▶ Continue','▶ 继续'));
  pb.dataset.tipx=tx(`Plays the run at the chosen speed: about ${RUN_SECS[spdNow()]} s.`,`按选定的速度播放这次运行：约 ${RUN_SECS[spdNow()]} 秒。`);
  $('#pdd-replay',app).dataset.tipx=tx('Plays this run again from the start.','从头再播放一次这次运行。');
  $$('.seg-ctl[data-key="spd"]',app).forEach(g=>A.ui.setPressed(g,spdNow()));}
function render(){
  if(!app) return; const run=curRun();
  if(run.key!==lastRunKey){lastRunKey=run.key; if(!PLAY.on) phNow=Infinity;}
  drawHead(run); drawTimeline(run); drawWatch(run); drawRunCtl(run); drawMetrics(run);
  const gpb=$('#pdd-gpb',app), gh=gpChartHTML(); gpb.hidden=!gh; const gp=$('#pdd-gp',app); if(gp._h!==gh){gp.innerHTML=gh; gp._h=gh;}
  const mb=$('#pdd-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  const md=$('#pdd-math',app); if(md&&!md._t){md._t=1; md.addEventListener('toggle',()=>{if(md.open) render();});}
  syncControls(); updateRail();
  // the address names the settings in Scenarios only; leaving Scenarios drops them, so a reload in another mode puts back that mode (round 3 #6)
  if((!PLAY.on||mode!=='sandbox')&&/^#?pd-disagg(~|$)/.test(location.hash||'#pd-disagg')) A.hash.write(token());
  A.dock.refresh(); announce(run);}
function announce(run){clearTimeout(annT); annT=setTimeout(()=>{if(!app||partial(run)) return; A.announce(tx(`${fP(run.m.all.ok_frac)} of requests meet their goals; chats’ longest gap ${fT(run.m.chat.gap_max)}.`,`${fP(run.m.all.ok_frac)} 的请求达标；聊天的最长间隔 ${fT(run.m.chat.gap_max)}。`));},400);}

/* ---------- the run player (as labs/ppo.js): play / pause, 重放, speed, progress; R3.20 ---------- */
function stopPlay(){if(PLAY.raf) cancelAnimationFrame(PLAY.raf); PLAY.on=false; PLAY.raf=0;}
function playFrom(t0,quick){
  stopPlay(); const run=curRun(); phNow=t0; nPlays++;
  const secs=quick?1.5:RUN_SECS[spdNow()], rate=run.end/secs;
  PLAY={on:true,raf:0,last:0,key:run.key};
  const tick=ts=>{if(!PLAY.on||!app){PLAY.raf=0; return;}
    if(PLAY.last){phNow+=Math.min(0.1,(ts-PLAY.last)/1000)*rate; if(phNow>=run.end){phNow=Infinity; watched.add(PLAY.key); PLAY.on=false; PLAY.raf=0; render(); return;} moveHead(run);}
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
const WHY={chunk:{en:'Pick a token budget to continue.',zh:'选一个 token 预算才能继续。'},predict:{en:'Pick an answer to continue.',zh:'选一个答案才能继续。'},split:{en:'Try another split to continue.',zh:'换一种分法才能继续。'},
  kv:{en:'Try the link across nodes to continue.',zh:'试一下跨机器传才能继续。'},ratio:{en:'Click a setup (6:2 too) to watch its run and continue.',zh:'点一种方案（6:2 也行）看它的运行，才能继续。'}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(L(x.lead))})),m2:true,current:step,body:s.body(),back:step>0,showMe:false,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function renderRail(){const rail=$('#pdd-rail',app); if(!rail) return; rail.innerHTML=railFull(); render();}
function updateRail(){
  const rail=$('#pdd-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    const show=!!s.obs&&(s.always?!(partial(curRun())&&!watched.has(curRun().key)):(s.expect?s.expect():true));
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${tx('Observed','观察到')}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}}}
    if(s.id==='chunk') $$('[data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===p3b)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=tried['b'+v]?tx(`longest gap ${fT(R3(v).m.chat.gap_max)}`,`最长间隔 ${fT(R3(v).m.chat.gap_max)}`):'';});   // what changes with the budget (S4)
    if(s.id==='split') $$('[data-lever]',rail).forEach(b=>{const v=+b.dataset.lever; b.setAttribute('aria-pressed',String(v===p5p)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=(tried['s'+v]||v===p5p)?fP(R5(v).m.all.ok_frac):'';});
    $$('.seg-ctl[data-key="p6l"]',rail).forEach(g=>A.ui.setPressed(g,p6l)); $$('.seg-ctl[data-key="p7"]',rail).forEach(g=>A.ui.setPressed(g,String(p7p)));
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#pdd-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):''; ins.hidden=!html; if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const r=chRun(chPick()).m, ok=r.all.ok_frac>=PDD.TARGET-1e-9, o=$('#pdd-ch-ok',rail); if(!o) return;
    o.textContent=fP(r.all.ok_frac); const pr=$('#pdd-ch-pr i',rail); if(pr) pr.style.width=Math.min(100,r.all.ok_frac*100/0.9).toFixed(0)+'%';
    $('#pdd-ch-g',rail).classList.toggle('met',ok); $('#pdd-ch-g',rail).classList.toggle('pdd-over',!ok);
    $('#pdd-ch-co',rail).hidden=S.mode!=='co'; $('#pdd-ch-sp',rail).hidden=S.mode!=='split';
    $$('.seg-ctl[data-key="chm"]',rail).forEach(g=>A.ui.setPressed(g,S.mode)); $$('.seg-ctl[data-key="chb"]',rail).forEach(g=>A.ui.setPressed(g,String(S.budget)));
    $$('.seg-ctl[data-key="chp"]',rail).forEach(g=>A.ui.setPressed(g,String(S.p))); $$('.seg-ctl[data-key="chl"]',rail).forEach(g=>A.ui.setPressed(g,S.link));
    const res=$('#pdd-ch-res',rail), html=chResultHTML(); if(res._h!==html){res.innerHTML=html; res._h=html;}}
}
function syncControls(){
  const c=$('#pdd-controls',app); if(!c) return; const lock=$('#pdd-lock',app), wrap=$('#pdd-ctls',app), free=mode==='sandbox';
  wrap.hidden=!free; lock.hidden=free; lock.textContent=mode==='guided'?tx('Each part sets its own run; every setting is free in Scenarios.','每一部分有自己的运行；在应用场景里所有设置都能改。'):tx('In the challenge, the choices are in the task card.','挑战题的选择都在任务卡里。');
  if(!free) return;
  A.ui.setPressed($('.seg-ctl[data-key="kind"]',c),S.kind); A.ui.setPressed($('.seg-ctl[data-key="n"]',c),String(S.n)); A.ui.setPressed($('.seg-ctl[data-key="mode"]',c),S.mode);
  A.ui.setPressed($('.seg-ctl[data-key="budget"]',c),String(S.budget)); A.ui.setPressed($('.seg-ctl[data-key="link"]',c),S.link);
  $('#pdd-ctl-cow',c).hidden=S.mode!=='co'; $('#pdd-ctl-spw',c).hidden=S.mode!=='split';
  const pi=$('#pdd-ctl-p-in',c); if(pi&&document.activeElement!==pi){pi.max=String(S.n-1); pi.value=String(Math.min(S.p,S.n-1)); A.ui.setRange(pi);} const po=pi&&pi.closest('.ctl')&&pi.closest('.ctl').querySelector('output'); if(po) po.textContent=String(Math.min(S.p,S.n-1));
  const ri=$('#pdd-ctl-rate-in',c); if(ri&&document.activeElement!==ri){ri.value=String(S.rate); A.ui.setRange(ri);} const ro=ri&&ri.closest('.ctl')&&ri.closest('.ctl').querySelector('output'); if(ro) ro.textContent=String(S.rate);}

/* ---------- navigation ---------- */
function enterStep(i){step=Math.max(0,Math.min(STEPS.length-1,i)); renderRail(); showRun(STEPS[step].id==='mission'?'quiet':'');}
function revealStep(){A.ui.reveal($('#pdd-rail .step.current',app)||$('#pdd-rail',app));}
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2, R9.7), as labs/llm-judge.js */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#pdd-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#pdd-rail .step.current',app)||$('#pdd-rail .railcard',app)||$('#pdd-rail',app);
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
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...BASE,...CH_START,n:8,kind:CH.kind,rate:CH.rate};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail(); showRun(m==='sandbox'?'quiet':'quiet');}
const token=(s=S)=>[s.kind,s.n,s.mode,s.p,s.budget,s.link,s.rate].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const [kind,n,md,pp,b,link,rate]=p, N=+n, P=+pp, B=+b, Rt=+rate;
  if(!['A','C'].includes(kind)||![4,8].includes(N)||!['co','split'].includes(md)||!(P>=1&&P<N)||!BUDGETS.includes(B)||!['nvl','ib'].includes(link)||!PDD.RATES.includes(Rt)) return null;
  return {kind,n:N,mode:md,p:P,budget:B,link,rate:Rt};}

/* ---------- interview drill ---------- */
let drillSeed=1, drillOpen=false;
function makeDrill(seed){const rnd=PDD.mulberry(seed*7919+5), n=4096*(2+Math.floor(rnd()*14)); return {n};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#pdd-drill-q',app), a=$('#pdd-drill-a',app);
  q.innerHTML=tx(`A ${fint(d.n)}-token prompt for ${M8B} (BF16) is prefilled on one H100 and decoded on another. How many bytes of KV cache move, and how long does that take inside a node (NVLink, 450 GB/s one way) and across nodes (400 Gb/s InfiniBand)? Is that worth worrying about?`,
                 `${M8B}（BF16）的一个 ${fint(d.n)} 个 token 的 prompt，在一张 H100 上做 prefill，在另一张上 decode。要传多少字节的 KV cache？在一台机器里（NVLink，单向 450 GB/s）和跨机器（400 Gb/s 的 InfiniBand）各要多久？值得担心吗？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const by=XB(d.n), pf=(2*C.P_LAYERS*d.n+2*C.P_HEAD+PDD.attn(0,d.n))/C.FS;
  a.innerHTML=tx(`<p><b>${GB(by)}: ${fT(XT(d.n,'nvl'))} over NVLink, ${fT(XT(d.n,'ib'))} across nodes.</b></p>${EQ([['\\text{bytes}',`${d.n} \\times 2 \\times 32 \\times 8 \\times 128 \\times 2 = ${A.fmt.num(by/1e9,2)}\\ \\text{GB}`],['\\text{InfiniBand}',`\\frac{400\\ \\text{Gb/s}}{8} = 50\\ \\text{GB/s}`]])}<p>The prefill itself takes about ${fT(pf)} at 50% of the dense peak, so the transfer adds a little to the first gap of the answer, not to the throughput. It matters when many transfers share one link: each waits for the ones before it.</p>`,
                 `<p><b>${GB(by)}：NVLink 要 ${fT(XT(d.n,'nvl'))}，跨机器要 ${fT(XT(d.n,'ib'))}。</b></p>${EQ([['\\text{字节}',`${d.n} \\times 2 \\times 32 \\times 8 \\times 128 \\times 2 = ${A.fmt.num(by/1e9,2)}\\ \\text{GB}`],['\\text{InfiniBand}',`\\frac{400\\ \\text{Gb/s}}{8} = 50\\ \\text{GB/s}`]])}<p>prefill 本身按稠密峰值的 50% 算要 ${fT(pf)}，所以传输只让回答的第一个间隔长一点，不影响吞吐。要注意的是很多份传输挤在一条连接上的时候：每一份都要等前面的传完。</p>`);}

/* ---------- jumps ---------- */
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  return '';}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#pdd-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#pdd-controls',app));}}

/* ---------- events ---------- */
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setS(p,src){Object.assign(S,p); if(S.p>S.n-1) S.p=S.n-1; byLearner(()=>{render(); showRun(src||'settings');});}
function act(a,btn){
  if(a==='play'){if(PLAY.on){stopPlay(); render(); return;} const run=curRun(); playFrom(phNow>=run.end?0:phNow,false); return;}
  if(a==='replay'){playFrom(0,false); return;}
  if(a==='reset'){setS({...BASE}); return;}
  if(a==='submit'){chSubmitted=true; renderRail(); requestAnimationFrame(()=>revealBox($('#pdd-ch-res',app))); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#pdd-code',app); try{navigator.clipboard.writeText(pre.textContent);}catch(e){} btn.textContent=tx('Copied','已复制'); setTimeout(()=>{btn.textContent=A.t('copyCode');},1600);}}
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#pdd-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#pdd-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('chunk')){p3b=+d.lever; tried['b'+p3b]=true; byLearner(()=>{render(); showRun();}); requestAnimationFrame(()=>revealBox($('#pdd-rail [data-obs]',app))); return;}
    if(d.lever!=null&&isP('split')){p5p=+d.lever; tried['s'+p5p]=true; byLearner(()=>{render(); showRun();}); requestAnimationFrame(()=>revealBox($('#pdd-rail [data-obs]',app))); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v;
      if(k==='spd'){spdUser=v; if(PLAY.on){const run=curRun(); playFrom(phNow,false);} else render(); return;}
      if(k==='p6l'){p6l=v; if(v==='ib') tried.ib=true; byLearner(()=>{render(); showRun();}); return;}
      if(k==='p7'){p7p=+v; tried.r=true; byLearner(()=>{render(); showRun();}); return;}
      if(k==='chm') return byLearner(()=>{S.mode=v; render();});
      if(k==='chb') return byLearner(()=>{S.budget=+v; render();});
      if(k==='chp') return byLearner(()=>{S.p=+v; render();});
      if(k==='chl') return byLearner(()=>{S.link=v; render();});
      if(k==='kind') return setS({kind:v}); if(k==='n') return setS({n:+v}); if(k==='mode') return setS({mode:v});
      if(k==='budget') return setS({budget:+v}); if(k==='link') return setS({link:v});
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; Object.assign(S,{...BASE,...p.s}); byLearner(()=>{render(); showRun('settings');}); requestAnimationFrame(()=>{if(app) revealBox($('#pdd-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#pdd-rail',app)); return;}
    if(d.act) act(d.act,b);});
  el.addEventListener('input',e=>{const t=e.target;
    if(t.id==='pdd-prog'){stopPlay(); const run=curRun(); phNow=run.end*(+t.value)/1000; if(+t.value>=1000){phNow=Infinity; watched.add(run.key);} moveHead(run); drawRunCtl(run); drawHead(run); drawMetrics(run); return;}
    if(t.id==='pdd-ctl-rate-in'){S.rate=+t.value; const o=t.closest('.ctl')&&t.closest('.ctl').querySelector('output'); if(o) o.textContent=String(S.rate);}
    if(t.id==='pdd-ctl-p-in'){S.p=+t.value; const o=t.closest('.ctl')&&t.closest('.ctl').querySelector('output'); if(o) o.textContent=String(S.p);}});
  el.addEventListener('change',e=>{const t=e.target; if(t.id==='pdd-ctl-rate-in'||t.id==='pdd-ctl-p-in') setS({});
    if(t.id==='pdd-prog'&&+t.value>=1000){phNow=Infinity; render();}});
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));}

/* ---------- registration ---------- */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
const sOf=s=>({kind:s.kind,mode:s.mode,n:s.n,p:s.p,budget:s.budget,link:s.link,rate:s.rate});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:sOf(S),pred,p3b,p5p,p6l,p7p,tried:{...tried},saved:saved?sOf(saved):null,chSubmitted,chReveal,sbTouched,spdUser,watched:[...watched],
    drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chOk(chPick())}),
  restore(o){if(!o||o.v!==1||!o.S||typeof o.S!=='object') return;
    const md=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    // the challenge's workload (kind B, its rate) is outside the shared settings: check the learner's choices with the challenge's own fixed
    const p=parse(token(md==='challenge'?{...o.S,kind:'A',n:8,rate:BASE.rate}:o.S)); if(!p) return;
    mode=md; S=sOf(md==='challenge'?{...p,kind:CH.kind,n:CH.n,rate:CH.rate}:p);
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    p3b=[2048,512,128,64].includes(o.p3b)?o.p3b:null; p5p=[1,2,3].includes(o.p5p)?o.p5p:1; p6l=o.p6l==='ib'?'ib':'nvl'; p7p=o.p7p===0||P7.includes(o.p7p)?o.p7p:6;
    tried={}; if(o.tried&&typeof o.tried==='object') for(const k in o.tried) if(/^(b(2048|512|128|64)|s[1-3]|ib|r)$/.test(k)&&o.tried[k]===true) tried[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sOf(sv):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    spdUser=RUN_SECS[o.spdUser]?o.spdUser:null;
    watched.clear(); if(Array.isArray(o.watched)) o.watched.slice(-200).forEach(k=>{if(typeof k==='string'&&/^(A|B|C|one|one0)\|(co|split)\|[1-8]\|[0-7]\|(0|64|128|512|2048)\|(nvl|ib)\|\d+(\.\d+)?$/.test(k)) watched.add(k);});
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true; resumed=true;},
  mount(el,tok){
    root=el; const p=tok&&!resumed?parse(tok):null; resumed=false; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#pdd-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#pdd-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    const rail=$('#pdd-rail',app); rail.innerHTML=railFull();
    phNow=Infinity; render(); watchStage();
    if(mode==='guided'&&STEPS[step].id!=='mission') showRun();
    setTimeout(()=>{if(app) P7CFG().forEach(x=>needGp('A',x.c));},400);},
  leave(){stopPlay(); clearTimeout(annT); stageStop(); if(stRO){stRO.disconnect(); stRO=null;} root=null; app=null;},
  onLang(){if(root){const d=$('#pdd-math',app), open=d&&d.open, wasPlaying=PLAY.on; stopPlay(); this.mount(root,null); if(open) $('#pdd-math',app).open=true; if(wasPlaying) showRun();}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#pdd-tl',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#pdd-graph',app),
    render({status,vis,read}){if(!app) return; const run=curRun(), s=statusOf(run);
      status.className='status '+s.cls; status.textContent=s.short;
      vis.innerHTML=`<div class="pdd-dk"><i style="width:${partial(run)?0:(run.m.all.ok_frac*100).toFixed(0)}%"></i></div>`;
      read.textContent=partial(run)?tx('Playing the run; the numbers show when it ends.','正在播放，播完后显示结果。'):tx(`${fP(run.m.all.ok_frac)} of requests meet their goals; chats’ longest gap ${fT(run.m.chat.gap_max)}.`,`${fP(run.m.all.ok_frac)} 的请求达标；聊天的最长间隔 ${fT(run.m.chat.gap_max)}。`);}}
});

/* ---------- read-only hook for tests ---------- */
window.__pdd=Object.freeze({
  engine:PDD, STEP_IDS:STEPS.map(s=>s.id), PRESETS:PRESETS.map(p=>({id:p.id,s:{...p.s}})), CH:{...CH}, CH_START:{...CH_START}, P4:{...P4}, PRED_RIGHT,
  view:()=>{const v=viewRun(); return {kind:v.kind,cfg:cfgOf(v.s),rate:v.rate};},
  // the chat drawn in blue on the stage: its times and where it runs (tests/pd-disagg_flow.py checks the picture against these, frame by frame)
  watch:()=>{const run=curRun(), r=watchOf(run); if(!r) return null; const w=pathOf(run,r); return {id:r.id,arr:r.arr,first:r.first,fin:r.fin,toks:r.toks.slice(),split:w.split,g:w.g,dg:w.dg,p0:w.p0,x0:w.x0,x1:w.x1,end:run.end,gi:gapAt(r).gi};},
  state:()=>({...S,mode,step,pred,p3b,p5p,p6l,p7p,tried:{...tried},chSubmitted,chReveal,playing:PLAY.on,ph:phNow,plays:nPlays,watched:[...watched],spd:spdNow(),spdUser,gpQueue:gpQ.length}),
  chOpts:()=>CH_OPTS().map(o=>({...o,ok:chOk(o),share:chRun(o).m.all.ok_frac})),
  exits:()=>{const e=EXITS(); return {transferA:e.transfer.map(q=>q.a)};}
});

})();
