/* Product capstone (batch 10, round 2): how to design a ChatGPT-like app end to end.
   Illustrative (Conceptual): a system-design walk-through. One message's path through the components (gateway, conversation store,
   context builder, retrieval, input check, router, inference, output check) with the time each adds before the first visible word
   (the lab's stated assumptions; the model's first token is the serving labs' 1 s goal each pool is sized for), and back-of-envelope
   sizing from daily users to GPUs and dollars. GPUs per pool come from the serving capstone's engine (capstone/engine.js, copied below
   and made to take the model and the GPU count): goodput per replica, DistServe's definition (brief S4).
   tests/product-design_numbers.py recomputes every number with capstone/engine.py set to the same models. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='product-design';

/* ENGINE BEGIN: capstone/engine.js, unchanged except that it is a function of the model (shapes) and the GPU count n
   (tensor parallelism: the step's FLOPs and bytes are spread over n GPUs; the KV pool is n × (80 GiB × 0.9 − 2 GiB) minus the weights),
   and that the serving capstone's fixed loads are left out. */
function pdzEngine(M){
const L=M.L, D=M.D, KVH=M.KVH, HD=M.HD, FFN=M.FFN, VOCAB=M.VOCAB, NG=M.n;
const P_LAYER=2*D*D+2*D*KVH*HD+3*D*FFN+2*D, P_LAYERS=L*P_LAYER+D, P_HEAD=VOCAB*D, P_TOTAL=P_LAYERS+2*P_HEAD;
const KV_TOK=2*L*KVH*HD*2, W_READ=2*(P_LAYERS+P_HEAD), W_MEM=2*P_TOTAL;
const FS=1979e12/2*0.5*NG, BS=3.35e12*0.6*NG, GiB=2**30, BLOCK=16;
const POOL_T=Math.floor((NG*(80*GiB*0.9-2*GiB)-W_MEM)/KV_TOK), POOL_B=Math.floor(POOL_T/BLOCK);
const EPS=1e-9;
const attn=(a,b)=>2*L*D*(b*(b+1)-a*(a+1));
const blocks=n=>Math.floor((n+BLOCK-1)/BLOCK);

function mulberry(seed){let h=seed>>>0; return ()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};}
const expGap=(rnd,rate)=>-Math.log(1-rnd())/rate;
const ms=x=>Math.floor(x*1000+0.5)/1000;
const uni=(rnd,lo,hi)=>lo+Math.floor(rnd()*(hi-lo+1));

function sortLoad(a){return a.map((s,i)=>[s,i]).sort((x,y)=>x[0].t-y[0].t||x[1]-y[1]).map(x=>x[0]);}
const GOALS={chat:{ttft:1.0,gap:0.15}, doc:{ttft:5.0}};
const TARGET=0.9;
const BASELINE={policy:'continuous',cap:32,budget:0,cache:false,cacheBlocks:null};

/* cfg: {policy:'continuous'|'static', cap, budget (0 = no limit), cache (bool), cacheBlocks (max idle cached blocks; null = no limit)} */
function simulate(spec,cfg){
  const R=spec.map((s,i)=>({id:i,cls:s.cls,app:s.app||null,arr:s.t,pre:s.pre||0,own:s.own,out:s.out,P:(s.pre||0)+s.own,
    admit:null,hit:0,doneP:0,gen:0,toks:[],first:null,fin:null,blkOwn:0,resv:0,waits:{},chunks:[],dsteps:[],sharedNew:0}));
  const order=R.map((r,i)=>i).sort((a,b)=>R[a].arr-R[b].arr||a-b);
  const cap=cfg.cap, budget0=cfg.budget||0, cacheOn=!!cfg.cache, policy=cfg.policy, cacheCap=cfg.cacheBlocks==null?null:cfg.cacheBlocks;
  const pblk=new Map();                 // 'app|i' -> {app, i, ref, last}
  const held=new Map();                 // request id -> [keys it holds a reference to]
  let free=POOL_B, evictions=0, t=0, nxt=0, peak=0, prefillTok=0, hitTok=0;
  const steps=[], events=[], mem=[], waiting=[], running=[]; let batch=[];
  const key=(app,i)=>app+'|'+i;
  const idleCount=()=>{let n=0; for(const v of pblk.values()) if(v.ref===0) n++; return n;};
  const reserved=()=>{let s=0; for(const r of running) s+=r.resv; for(const r of batch) if(!running.includes(r)) s+=r.resv; return s;};
  function evict(n){
    const idle=[...pblk.entries()].filter(([k,v])=>v.ref===0).sort((x,y)=>x[1].last-y[1].last||(x[1].app<y[1].app?-1:x[1].app>y[1].app?1:0)||y[1].i-x[1].i);
    let got=0;
    for(const [k,v] of idle.slice(0,n)){pblk.delete(k); free++; got++; evictions++; events.push({t,type:'evict',app:v.app,i:v.i});}
    return got;
  }
  const canHold=n=>free+idleCount()-reserved()>=n;
  function alloc(r,n){ if(n<=0) return; if(free<n) evict(n-free); if(free<n) throw new Error('KV pool overflow'); free-=n; r.resv=Math.max(0,r.resv-n);}
  function lookup(r){ if(!cacheOn||!r.app||!r.pre) return 0; const nb=Math.floor(r.pre/BLOCK); let h=0; while(h<nb&&pblk.has(key(r.app,h))) h++; return h;}
  const needBlocks=(r,hb)=>blocks(r.P+r.out)-hb;
  const hold=(r,k)=>{let a=held.get(r.id); if(!a) held.set(r.id,a=[]); a.push(k);};
  function admit(r,now){
    const hb=lookup(r);
    for(let i=0;i<hb;i++){const v=pblk.get(key(r.app,i)); v.ref++; v.last=now;}
    r.hit=hb*BLOCK; r.doneP=r.hit; r.admit=now; hitTok+=r.hit; r.resv=needBlocks(r,hb);
    running.push(r); events.push({t:now,type:'admit',id:r.id,hit:r.hit});
    for(let i=0;i<hb;i++) hold(r,key(r.app,i));
  }
  function trimCache(){ if(cacheCap==null) return; const n=idleCount(); if(n>cacheCap) evict(n-cacheCap);}
  function releaseAll(r){
    free+=r.blkOwn; r.blkOwn=0; r.resv=0;
    for(const k of held.get(r.id)||[]){const v=pblk.get(k); v.ref--; v.last=t; if(v.ref<0) throw new Error('negative reference count');}
    held.delete(r.id); trimCache();
  }
  const used=()=>POOL_B-free;
  let guard=0;
  for(;;){
    if(++guard>400000) throw new Error('runaway simulation');
    while(nxt<order.length&&R[order[nxt]].arr<=t+EPS){waiting.push(R[order[nxt]]); nxt++;}
    if(!waiting.length&&!running.length){ if(nxt>=order.length) break; t=R[order[nxt]].arr; continue; }
    let budget=budget0?budget0:1e12;
    const work=[], reasons={};
    if(policy==='continuous'){
      for(const r of running) if(r.first!=null&&r.gen<r.out){work.push({r,k:'d',a:0,b:0}); budget-=1;}
      for(const r of running) if(r.first==null&&r.doneP<r.P&&budget>0){const n=Math.min(r.P-r.doneP,budget); work.push({r,k:'p',a:r.doneP,b:r.doneP+n}); budget-=n;}
      let blocked=null;
      for(const r of waiting.slice()){
        if(blocked==null){
          if(running.length>=cap) blocked='cap';
          else if(budget<=0) blocked='budget';
          else if(!canHold(needBlocks(r,lookup(r)))) blocked='memory';
        }
        if(blocked!=null){reasons[r.id]=blocked; continue;}
        waiting.splice(waiting.indexOf(r),1); admit(r,t);
        const n=Math.min(r.P-r.doneP,budget); work.push({r,k:'p',a:r.doneP,b:r.doneP+n}); budget-=n;
      }
    } else {
      if(!batch.length){
        while(waiting.length&&batch.length<cap){const r=waiting[0]; if(!canHold(needBlocks(r,lookup(r)))) break; waiting.shift(); admit(r,t); batch.push(r);}
      }
      for(const r of waiting) reasons[r.id]='batch';
      for(const r of batch){ if(r.fin!=null) continue; if(r.first==null) work.push({r,k:'p',a:r.doneP,b:r.P}); else if(r.gen<r.out) work.push({r,k:'d',a:0,b:0}); }
    }
    if(!work.length) throw new Error('deadlock: requests wait but nothing can run');
    for(const w of work){const r=w.r;
      if(w.k==='p'){
        const sharedEnd=(cacheOn&&r.app)?Math.floor(r.pre/BLOCK)*BLOCK:0;
        if(w.a<sharedEnd){
          for(let i=Math.floor(w.a/BLOCK);i<Math.floor(Math.min(w.b,sharedEnd)/BLOCK);i++){const k=key(r.app,i);
            if(pblk.has(k)){const v=pblk.get(k); v.ref++; v.last=t;}
            else {alloc(r,1); pblk.set(k,{app:r.app,i,ref:1,last:t}); r.sharedNew++;}
            hold(r,k);}
        }
        const ownA=Math.max(w.a,sharedEnd);
        if(w.b>ownA){const nb=blocks(w.b-sharedEnd)-blocks(ownA-sharedEnd); alloc(r,nb); r.blkOwn+=nb;}
      } else {
        const c=r.P+r.gen-1; if(c%BLOCK===0){alloc(r,1); r.blkOwn++;}
      }
    }
    let T=0, logits=0, flAtt=0, kv=0;
    for(const w of work){const r=w.r;
      if(w.k==='d'){const c=r.P+r.gen-1; T+=1; logits+=1; flAtt+=attn(c,c+1); kv+=c;}
      else {T+=w.b-w.a; logits+=w.b===r.P?1:0; flAtt+=attn(w.a,w.b); kv+=w.b;}
    }
    const fl=2*P_LAYERS*T+2*P_HEAD*logits+flAtt, by=W_READ+KV_TOK*kv, tc=fl/FS, tm=by/BS, dt=Math.max(tc,tm), si=steps.length, t1=t+dt;
    steps.push({t0:t,t1,tc,tm,T,fl,by,work:work.map(w=>({id:w.r.id,k:w.k,a:w.a,b:w.b})),nrun:policy==='continuous'?running.length:batch.length});
    for(const r of waiting){const why=reasons[r.id]||'batch'; r.waits[why]=(r.waits[why]||0)+dt;}
    for(let j=nxt;j<order.length;j++){const q=R[order[j]]; if(q.arr<t1-EPS) q.waits.step=(q.waits.step||0)+(t1-q.arr); else break;}
    t=t1;
    for(const w of work){const r=w.r;
      if(w.k==='p'){r.chunks.push([si,w.a,w.b]); prefillTok+=w.b-w.a; r.doneP=w.b; if(w.b===r.P){r.first=t1; r.toks.push(t1); r.gen=1;}}
      else {r.gen++; r.toks.push(t1); r.dsteps.push(si);}
    }
    peak=Math.max(peak,used());
    const finNow=[];
    for(const r of running.slice()) if(r.first!=null&&r.gen>=r.out){r.fin=t1; running.splice(running.indexOf(r),1); finNow.push(r);}
    for(const r of finNow){releaseAll(r); events.push({t:t1,type:'finish',id:r.id});}
    if(policy==='static'&&batch.length&&batch.every(r=>r.fin!=null)) batch=[];
    mem.push({t:t1,used:used(),idle:idleCount(),resv:reserved(),shared:pblk.size});
  }
  return {R,steps,events,peak,evictions,prefillTok,hitTok,mem,cfg};
}

/* nearest-rank percentile: the smallest value with at least p% of the samples at or below it */
function pct(xs,p){const s=xs.slice().sort((a,b)=>a-b); if(!s.length) return null; const k=Math.max(1,Math.ceil(p/100*s.length-1e-9)); return s[k-1];}
const gapOf=r=>{let g=0; for(let i=1;i<r.toks.length;i++) g=Math.max(g,r.toks[i]-r.toks[i-1]); return g;};
function metrics(run,goals=GOALS){
  const R=run.R, out={};
  for(const cls of ['chat','doc']){
    const rs=R.filter(r=>r.cls===cls); if(!rs.length) continue;
    const ttft=rs.map(r=>r.first-r.arr), gaps=rs.map(gapOf), g=goals[cls]||{};
    const ok=rs.map((r,i)=>(ttft[i]<=(g.ttft??1e9)+EPS)&&(gaps[i]<=(g.gap??1e9)+EPS));
    out[cls]={n:rs.length,ttft_p50:pct(ttft,50),ttft_p95:pct(ttft,95),ttft_max:Math.max(...ttft),gap_max:Math.max(...gaps),gap_p95:pct(gaps,95),
      ok:ok.filter(Boolean).length,ok_frac:ok.filter(Boolean).length/rs.length};
  }
  const toks=R.reduce((s,r)=>s+r.out,0), t0=Math.min(...R.map(r=>r.arr)), t1=Math.max(...R.map(r=>r.fin));
  const ok=['chat','doc'].reduce((s,c)=>s+(out[c]?out[c].ok:0),0);
  out.all={n:R.length,ok,ok_frac:ok/R.length,thr:toks/(t1-t0),makespan:t1-t0,peak_gib:run.peak*BLOCK*KV_TOK/GiB,prefill_tok:run.prefillTok,
    hit_tok:run.hitTok,evictions:run.evictions,steps:run.steps.length};
  return out;
}
/* the invariants the lab's tests check on every run */
function check(run){
  const R=run.R, st=run.steps;
  for(let i=1;i<st.length;i++) if(st[i].t0<st[i-1].t1-EPS) throw new Error('time went backwards');
  for(const s of st){const ids=s.work.map(w=>w.id); if(new Set(ids).size!==ids.length) throw new Error('a request twice in one step');
    for(const w of s.work){if(R[w.id].arr>s.t0+EPS) throw new Error('work before arrival'); if(w.k==='d'&&!(R[w.id].first!=null&&R[w.id].first<=s.t0+EPS)) throw new Error('decode before first token');}}
  for(const r of R){
    if(r.fin==null||r.toks.length!==r.out) throw new Error('request '+r.id+' did not finish with its whole answer');
    if(!(r.admit>=r.arr-EPS&&r.first>r.admit&&r.fin>=r.first)) throw new Error('request '+r.id+' out of order');
    if(r.chunks.reduce((s,c)=>s+c[2]-c[1],0)+r.hit!==r.P) throw new Error('request '+r.id+' prompt tokens do not add up');
    for(let i=1;i<r.toks.length;i++) if(!(r.toks[i]>r.toks[i-1])) throw new Error('tokens out of order');
  }
  for(const m of run.mem) if(m.used<0||m.used>POOL_B||m.used+m.resv>POOL_B+1) throw new Error('memory over the pool');
  return true;
}
return {simulate,metrics,check,pct,gapOf,attn,blocks,mulberry,expGap,ms,uni,sortLoad,GOALS,TARGET,BASELINE,
  C:{L,D,KVH,HD,FFN,VOCAB,NG,P_LAYER,P_LAYERS,P_HEAD,P_TOTAL,KV_TOK,W_READ,W_MEM,FS,BS,GiB,BLOCK,POOL_T,POOL_B}};
}
/* ENGINE END */

/* ---------- the lab's model (pure) ----------
   Small model: Llama-3.1-8B on one H100. Large model: Llama-3.1-70B on 4 H100s with tensor parallelism (brief S2: each GPU holds 1/n of
   the weights and of every request's KV cache and does 1/n of the work; the communication between GPUs is left out).
   The load: chat messages, Poisson arrivals at a stated rate, 200 per run, seed 7. Every prompt starts with the same 2,048-token system
   prompt (instructions and tool definitions, the app 'sys' of the engine: with prompt caching on, its KV cache is computed once and
   shared), then the trimmed recent history and the new message, uniform 500–3,500 tokens (+1,536 tokens of retrieved passages when
   retrieval is on); answers uniform 100–500 tokens. Engine settings: at most 256 requests running, a token budget per step of 2,048 (8B)
   or 1,024 (70B, so that a step with a prefill chunk stays under the 150 ms gap goal).
   Goodput per replica (DistServe's definition, brief S4): the highest rate on a grid of 0.25 requests/s at which at least 90% of the
   chat requests meet their goals (first token within 1 s, no gap over 150 ms); found by bisection on the grid, which assumes that
   passing is monotone in the rate (tests/product-design_numbers.py scans the whole grid and checks it). */
const NB='‑';
const MODELS={
  s:{name:`Llama${NB}3.1${NB}8B`,n:1,budget:2048,M:{L:32,D:4096,KVH:8,HD:128,FFN:14336,VOCAB:128256,n:1}},
  l:{name:`Llama${NB}3.1${NB}70B`,n:4,budget:1024,M:{L:80,D:8192,KVH:8,HD:128,FFN:28672,VOCAB:128256,n:4}}};
const WL={sys:2048,own:[500,3500],ret:1536,out:[100,500],outMean:300,n:200,seed:7,cap:256,step:0.25,gmax:128,storeTok:400,bytesTok:4};
/* time each component adds before the first token, in ms: the lab's assumptions (stated in Settings notes, hovers and the Fidelity card) */
const LAT={net:40,gw:5,hist:15,ctx:5,ret:150,inF:40,route:10,chk:20};
const TTFT_GOAL=1.0;           // brief S4: a chat request's first token within 1 s; each pool is sized so that 90% meet it at peak
const CHECK_GPU=0.001;         // GPU seconds per output check (1 ms of one H100), the lab's assumption
const DAY=86400;
const ENG={s:pdzEngine(MODELS.s.M),l:pdzEngine(MODELS.l.M)};
function loadOf(E,rate,ret){const rnd=E.mulberry(WL.seed), out=[]; let t=0, add=ret?WL.ret:0;
  for(let i=0;i<WL.n;i++){t+=E.expGap(rnd,rate); const own=E.uni(rnd,WL.own[0]+add,WL.own[1]+add), o=E.uni(rnd,WL.out[0],WL.out[1]);
    out.push({t:E.ms(t),cls:'chat',app:'sys',pre:WL.sys,own,out:o});}
  return out;}
const RUNS=new Map();
function runAt(m,i,cache,ret){const key=[m,i,cache?1:0,ret?1:0].join('|'); if(RUNS.has(key)) return RUNS.get(key);
  const E=ENG[m], rate=i*WL.step, run=E.simulate(loadOf(E,rate,ret),{policy:'continuous',cap:WL.cap,budget:MODELS[m].budget,cache:!!cache,cacheBlocks:null}), mt=E.metrics(run);
  const R=run.R, ttft=R.map(r=>r.first-r.arr), tpot=R.map(r=>(r.fin-r.first)/(r.out-1));
  const o={rate,ok:mt.chat.ok_frac,ttft90:E.pct(ttft,90),tpot90:E.pct(tpot,90),tpot50:E.pct(tpot,50),gap:mt.chat.gap_max};
  RUNS.set(key,o); return o;}
const POOLS=new Map();
function pool(m,cache,ret){const key=[m,cache?1:0,ret?1:0].join('|'); if(POOLS.has(key)) return POOLS.get(key);
  let lo=0, hi=WL.gmax+1;
  while(hi-lo>1){const mid=(lo+hi)>>1; if(runAt(m,mid,cache,ret).ok>=ENG[m].TARGET-1e-9) lo=mid; else hi=mid;}
  const at=runAt(m,lo,cache,ret), md=MODELS[m];
  const o={m,g:lo*WL.step,gpu:lo*WL.step/md.n,ttft90:at.ttft90,tpot90:at.tpot90,tpot50:at.tpot50,ok:at.ok};
  POOLS.set(key,o); return o;}
/* one message on a quiet service: two identical requests, the second after the first is done, so a cached system prompt is a hit */
const QUIET=new Map();
function quiet(m,cache,ret){const key=[m,cache?1:0,ret?1:0].join('|'); if(QUIET.has(key)) return QUIET.get(key);
  const E=ENG[m], own=(WL.own[0]+WL.own[1])/2+(ret?WL.ret:0), spec=[0,2].map(t=>({t,cls:'chat',app:'sys',pre:WL.sys,own,out:1}));
  const run=E.simulate(spec,{policy:'continuous',cap:WL.cap,budget:MODELS[m].budget,cache:!!cache,cacheBlocks:null}), r=run.R[1];
  const o={ttft:r.first-r.arr,prefill:WL.sys+own-r.hit,hit:r.hit,prompt:WL.sys+own}; QUIET.set(key,o); return o;}

const FILTS=['off','after','whole',32,16,8,4];
const SHARES=[0,0.3,0.5,0.7,0.9];
const EASY=0.7;                // share of messages a small model answers well enough: the lab's assumption (measure it with your own evals)
const DAUS=[1e4,3e4,1e5,3e5,1e6,3e6,1e7];
const MSGS=[5,10,20,40], PEAKS=[2,3,5];
const BASE={dau:1e6,msgs:10,peak:3,stream:1,cache:0,ret:0,share:0,filt:8,price:3};
const BRIEF={vis:1.8,budget:10000};                  // the guided parts' product brief
const CH={dau:2e6,msgs:12,peak:3,ret:0,price:3,vis:1.5,budget:32000}, CH0={stream:1,cache:0,share:0,filt:16};
const checksPer=f=>typeof f==='number'?WL.outMean/f:(f==='whole'||f==='after')?1:0;
function compute(s){
  const lam=s.dau*s.msgs/DAY*s.peak, P={s:pool('s',s.cache,s.ret),l:pool('l',s.cache,s.ret)};
  const rate={s:lam*s.share,l:lam*(1-s.share)};
  const rep={s:rate.s>0?Math.ceil(rate.s/P.s.g-1e-9):0,l:rate.l>0?Math.ceil(rate.l/P.l.g-1e-9):0};
  const gpus={s:rep.s*MODELS.s.n,l:rep.l*MODELS.l.n};
  const cps=checksPer(s.filt), fRaw=lam*cps*CHECK_GPU, fG=cps?Math.ceil(fRaw-1e-9):0;
  const total=gpus.s+gpus.l+fG, costDay=total*24*s.price, msgsDay=s.dau*s.msgs;
  const used=['s','l'].filter(m=>rate[m]>0), tpot=Math.max(...used.map(m=>P[m].tpot90));
  /* the model's first token: the 90th percentile TTFT the engine measures at goodput, the slower of the pools in use */
  const tModel=Math.max(...used.map(m=>P[m].ttft90));
  /* the request's path before the first word reaches the user (seconds) */
  const seg=[]; let t=0; const add=(k,d)=>{seg.push({k,t0:t,d}); t+=d;};
  add('net',LAT.net/1000); add('gw',LAT.gw/1000); add('hist',LAT.hist/1000); add('ctx',LAT.ctx/1000);
  if(s.ret) add('ret',LAT.ret/1000);
  add('inF',LAT.inF/1000);
  if(s.share>0) add('route',LAT.route/1000);
  add('model',tModel);
  const tFirst=t, tEnd=tFirst+(WL.outMean-1)*tpot;
  let hold=0, gen=0, nChk=0;
  const chkOn=s.filt!=='off', before=chkOn&&(s.filt!=='after'||!s.stream);   // sent when done: even “after” is checked before the user sees it
  /* sent when done (no streaming, or the whole answer held for one check): the user waits while the model writes the rest of the
     answer (gen, drawn on the model's row), then for the last check (hold) */
  if(!s.stream||s.filt==='whole'){gen=(WL.outMean-1)*tpot; hold=chkOn?LAT.chk/1000:0;}
  else if(typeof s.filt==='number') hold=(s.filt-1)*tpot+LAT.chk/1000;
  if(chkOn) nChk=typeof s.filt==='number'?Math.ceil(WL.outMean/s.filt):1;
  if(gen>0) add('gen',gen);
  if(hold>0) add('hold',hold);
  add('back',LAT.net/1000);
  const vis=t;
  return {s,lam,P,rate,rep,gpus,fG,fRaw,cps,total,costDay,msgsDay,tModel,gen,perUserMonth:costDay*30/s.dau,per1k:costDay/(msgsDay/1000),
    storeGB:msgsDay*WL.storeTok*WL.bytesTok/1e9,tpot,used,seg,tFirst,tEnd,hold,nChk,vis,before,chkOn,quality:s.share<=EASY+1e-9};}
/* the challenge: every combination of the four allowed choices under the fixed brief */
function chEval(o){const r=compute({...CH,...o}); const okV=r.vis<=CH.vis+1e-9, okB=r.costDay<=CH.budget+1e-9, okS=r.before, okQ=r.quality;
  return {...o,r,okV,okB,okS,okQ,ok:okV&&okB&&okS&&okQ};}
function chAll(){const out=[]; [1,0].forEach(stream=>[0,1].forEach(cache=>SHARES.forEach(share=>FILTS.forEach(filt=>out.push(chEval({stream,cache,share,filt}))))));
  return out;}
function chBest(){const ok=chAll().filter(e=>e.ok); if(!ok.length) return null; return ok.reduce((b,e)=>e.r.costDay<b.r.costDay-1e-9?e:b);}

/* ---------- words, formats ---------- */
const f=(v,d)=>A.fmt.num(v,d);
const I=n=>A.fmt.int(Math.round(n));
const rT=v=>v>=1000?I(v):v.toFixed(1);   // a rate per second: one decimal, whole numbers with separators from 1,000
const sT=v=>f(v,2)+tx(' s',' 秒');
const msT=v=>{const x=v*1000; return (x>=100?f(x,0):f(x,1))+' ms';};
const dur=v=>v>=1?sT(v):msT(v);
const usd=v=>'$'+(v>=100?I(v):v>=1?f(v,2):f(v,3));
const pc=(v,d=0)=>f(v*100,d)+'%';
const bigN=n=>n>=1e6?A.fmt.big(n):I(n);
const dauT=n=>n>=1e6?tx(`${f(n/1e6,n%1e6?1:0)} million`,`${f(n/1e6,n%1e6?1:0)}M`):n>=1e3?tx(`${I(n)}`,`${I(n)}`):I(n);
const M=t=>A.ui.math(t);
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const cite=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const live=id=>{const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; return !!l&&(c.labStatus?c.labStatus(l):l.st)==='live';};
const LL=id=>{const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; if(live(id)&&A.shell.labLink) return A.shell.labLink(id); return esc(l?L(l.s||{en:id,zh:id}):id);};
const shareT=v=>pc(v);
const cpsT=v=>String(+v.toFixed(2));   // checks per answer: 37.5, 18.75, 75 (no trailing zero, never rounded to one decimal)
/* the output check's options have one set of names everywhere: guided options, Settings, the challenge and its results (P25) */
const filtT=v=>({off:tx('No check','不查'),after:tx('Check after all is sent','全部发完再查'),whole:tx('Check all, then send','整段先查再发')}[v]||tx(`Check every ${v} tokens`,`每 ${v} 个 token 查一次`));
const filtName=v=>typeof v==='number'?tx(`check every ${v} tokens`,`每 ${v} 个 token 查一次`):({off:tx('no check','不查'),after:tx('check after all is sent','全部发完再查'),whole:tx('check all, then send','整段先查再发')}[v]);
/* the option names of the output check, full words so the challenge card alone says what each means (R13) */
const filtShort=v=>({off:tx('No check','不查'),after:tx('After all is sent','全部发完再查'),whole:tx('Check all, then send','整段先查再发')}[v]||tx(`Every ${v} tokens`,`每 ${v} 个 token`));
const FILT_L=()=>tx('Output check: when, or every how many tokens','输出检查：什么时候查，或每多少个 token 检查一次');
const FILT_N=()=>tx('“Every N tokens”: each time the answer has N new tokens, that piece is checked, and only then sent to the user.','“每 N 个 token”：回答每写出 N 个 token，先查这一段，查过才发给用户。');
const mName=m=>MODELS[m].name+(MODELS[m].n>1?tx(` on ${MODELS[m].n} H100s`,`（${MODELS[m].n} 张 H100）`):tx(' on one H100','（1 张 H100）'));
/* sources, pinned (sources/b10src/product-design/README.md) */
const GH='https://github.com/';
const SRC={
  aev:GH+'anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/src/anthropic/types/raw_message_stream_event.py',
  afx:GH+'anthropics/anthropic-sdk-python/blob/b4b7916deaf4e5570cd627b1ae7ee4394a4de39f/tests/lib/streaming/fixtures/basic_response.txt',
  ostr:GH+'openai/openai-python/blob/ef676dbc199bc09d12a1d051f3b8b2486aa53dc2/src/openai/_streaming.py',
  oso:GH+'openai/openai-python/blob/ef676dbc199bc09d12a1d051f3b8b2486aa53dc2/src/openai/types/chat/chat_completion_stream_options_param.py',
  lg:GH+'meta-llama/PurpleLlama/blob/172c1074069eb88ec834124272c1b1c4f8893445/Llama-Guard3/8B/MODEL_CARD.md',
  lgr:GH+'meta-llama/PurpleLlama/blob/172c1074069eb88ec834124272c1b1c4f8893445/Llama-Guard3/README.md',
  pg:GH+'meta-llama/PurpleLlama/blob/172c1074069eb88ec834124272c1b1c4f8893445/Llama-Prompt-Guard-2/86M/MODEL_CARD.md',
  rl:GH+'lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/README.md',
  llr:GH+'BerriAI/litellm-docs/blob/c61d25d31ba6b1f279ee8e620a4d100371957fe7/docs/routing.md',
  llf:GH+'BerriAI/litellm-docs/blob/c61d25d31ba6b1f279ee8e620a4d100371957fe7/docs/proxy/reliability.md',
  pc:GH+'anthropics/claude-cookbooks/blob/d7265d6ae994ccd8429db0594b000073b2f9ad43/misc/prompt_caching.ipynb',
  apc:GH+'vllm-project/vllm/blob/ac8c2130a8a6ecc4940f6c5433cb2ba15e983aa7/docs/features/automatic_prefix_caching.md',
  vsrv:GH+'vllm-project/vllm/blob/ac8c2130a8a6ecc4940f6c5433cb2ba15e983aa7/vllm/entrypoints/openai/chat_completion/serving.py',
  rate:GH+'openai/openai-cookbook/blob/0eac1447d4e24d06e47c459ca5e98f248b9413cf/examples/How_to_handle_rate_limits.ipynb',
  tk:GH+'openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/README.md'};

/* ---------- definitions (R2.10, S4, S6) ---------- */
const D_TTFT=()=>tx('Time to first token (TTFT): from the moment a request arrives to its first answer token, queueing included. The serving labs’ goal for a chat request is 1 s.','首 token 延迟（TTFT）：从请求到达，到它的第一个回答 token，包含排队。Serving 各 Lab 给聊天请求定的目标是 1 秒内。');
const D_VIS=()=>tx('Time to the first visible word: from the moment the user presses Send to the moment the first words of the answer appear on screen. It adds the time of every component on the path, the model’s time to first token, and the time the output check holds the first chunk.','首字可见时间：从用户按下发送，到回答的第一个字出现在屏幕上。它等于路上各部件的时间，加上模型的首 token 延迟，再加上输出检查扣住第一段的时间。');
const D_GOODPUT=()=>tx('Goodput (DistServe’s definition): the highest rate, in requests per second, that one replica can take while at least 90% of requests meet their goals (first token within 1 s, no gap between tokens over 150 ms). This lab counts it per replica; a 70B replica has 4 GPUs, so its goodput per GPU is a quarter of that.','达标吞吐（goodput，DistServe 的定义）：至少 90% 的请求达标（首 token 在 1 秒内、token 之间最长的间隔不超过 150 毫秒）时，一份副本每秒最多能接的请求数。这个 Lab 按每份副本算；70B 一份占 4 张卡，每张卡的达标吞吐是它的四分之一。');
const D_MEET=()=>tx('Meeting the goals: at least 90% of requests get their first token within 1 s and never wait more than 150 ms between two tokens.','达标：至少 90% 的请求同时做到两件事，首 token 在 1 秒内出来，token 之间最长的间隔不超过 150 毫秒。');
const D_STREAM=()=>tx('Streaming: the server sends the answer a few tokens at a time while the model is still writing, instead of waiting for the whole answer.','流式输出：模型每写出几个 token，服务端就把它们发给用户，不等整段写完。');
const D_CACHE=()=>tx('Prompt caching (prefix caching): when a new prompt starts with exactly the same tokens as an earlier one, the engine reuses the KV cache it already computed for that beginning and runs prefill only on the rest.','prompt 缓存（prefix caching）：新的 prompt 开头和之前某个 prompt 的开头一字不差时，引擎直接复用已经算好的那一段 KV cache，只对后面的部分做 prefill。');
const D_ROUTE=()=>tx('Model routing: a small classifier reads each message and sends it to a small, cheap model or to a large one, by how hard it looks.','模型路由：一个小分类器先读每条消息，按它看起来有多难，交给便宜的小模型或大模型。');
const D_SAFE=()=>tx('Safety filter (moderation): a small classifier that labels a piece of text safe or unsafe and stops what the service must not process or must not send. It runs twice: on the input before the model, and on the output before the user. Examples are small classifiers such as Llama Guard 3-1B or Prompt Guard 2; an 8B classifier costs more per check.','安全过滤（moderation）：一个小的分类模型，判断一段文字安不安全，拦下不该处理或不该发出的内容。它要跑两次：输入在模型之前查，输出在用户之前查。比如 Llama Guard 3-1B 或 Prompt Guard 2 这样的小分类器；用 8B 的分类器，每次检查会更贵。');
const D_REPL=()=>tx('Replica: one copy of the model that serves requests on its own. The 8B needs 1 GPU per replica. The 70B’s weights fill almost all of 2 GPUs and leave little room for the KV cache, so this lab gives it 4.','副本（replica）：一份能独立服务请求的模型。8B 一份占 1 张卡。70B 的权重放进 2 张卡后，几乎没地方放 KV cache，所以这个 Lab 给它 4 张。');
const D_CHKGPU=()=>tx('The classifier runs many checks together in one batch, so each check uses only about 1 ms of one GPU (this lab’s assumption). Checks per second at peak times 1 ms is the number of GPUs the checks keep busy.','分类器把许多检查拼成一批一起跑，每次检查只占一张卡约 1 ms 的计算（这个 Lab 的假设）。高峰时每秒要查的次数乘以 1 ms，就是检查一直占着的卡数。');
const D_GPUH=()=>tx('GPU hour: GPUs times hours, so 8 GPUs for one hour are 8 GPU hours. Costs here are for all GPUs together.','GPU 小时：卡数 × 小时数，8 张卡跑 1 小时就是 8 个 GPU 小时。这里的花费都是所有卡加起来。');
const D_PEAK=()=>tx('Peak factor: the busiest hour’s request rate divided by the day’s average. The fleet must be sized for the peak, or first tokens come late exactly when most people are online.','高峰倍数：一天里最忙的那个小时的请求速率，除以全天的平均速率。卡要按高峰配，否则人最多的时候首 token 来得最晚。');

/* the components on the path, in order */
const COMP={
  net:{n:{en:'Network in',zh:'网络（去程）'},d:()=>tx(`The message travels from the user’s phone or browser to the service: ${LAT.net} ms, the lab’s assumption. It depends on the distance to the nearest data center.`,`消息从用户的手机或浏览器传到服务端：${LAT.net} ms，这是这个 Lab 的假设，取决于离最近的数据中心有多远。`)},
  gw:{n:{en:'API gateway',zh:'API 网关'},d:()=>tx(`Checks who is sending (the sign-in token) and whether this user is over the rate limit; over it, the gateway answers 429 at once. Assumed ${LAT.gw} ms: one lookup in memory.`,`检查是谁发的（登录凭证），以及这个用户有没有超过限流；超过了就马上回 429。假设 ${LAT.gw} ms：查一次内存。`)},
  hist:{n:{en:'Conversation service',zh:'会话服务'},sn:{en:'Conversations',zh:'会话服务'},d:()=>tx(`Reads the conversation’s recent turns from the session store and saves the new message. Assumed ${LAT.hist} ms: one read from a database or cache.`,`从会话存储里读出这段对话最近几轮，并存下新消息。假设 ${LAT.hist} ms：读一次数据库或缓存。`)},
  ctx:{n:{en:'Context builder',zh:'上下文构建'},d:()=>tx(`Puts the system prompt, the history and the new message together and keeps them within a token budget: the oldest turns are dropped or replaced by a summary written earlier. Assumed ${LAT.ctx} ms.`,`把 system prompt、历史和新消息拼在一起，控制在 token 预算以内：最早的几轮删掉，或换成之前写好的摘要。假设 ${LAT.ctx} ms。`)},
  ret:{n:{en:'Retrieval',zh:'检索'},d:()=>tx(`Turns the question into an embedding, finds the nearest passages in a vector index and reranks them; 3 passages of 512 tokens join the prompt. Assumed ${LAT.ret} ms.`,`把问题变成 embedding，在向量索引里找最近的段落，再 rerank；3 段、每段 512 个 token 加进 prompt。假设 ${LAT.ret} ms。`)},
  inF:{n:{en:'Input safety check',zh:'输入安全检查'},sn:{en:'Input check',zh:'输入检查'},d:()=>tx(`A small classifier reads the new message before the model does and stops what must not be processed. Assumed ${LAT.inF} ms, including the call to the classifier service.`,`一个小分类器在模型之前先读新消息，拦下不该处理的内容。假设 ${LAT.inF} ms，包括调用分类服务的时间。`)},
  route:{n:{en:'Model router',zh:'模型路由'},d:()=>tx(`Decides whether the small model is good enough for this message or the large one is needed. Assumed ${LAT.route} ms: a small classifier.`,`判断这条消息用小模型够不够，还是要交给大模型。假设 ${LAT.route} ms：一个小分类器。`)},
  model:{n:{en:'Inference',zh:'推理'},d:()=>tx(`Queue and prefill until the first token, then decode, one token per step. The bar up to the first token is the 90th percentile time to first token that the serving engine measures for this pool at its goodput (at peak), the slower pool when two are in use; each pool is sized so that this stays within ${f(TTFT_GOAL,0)} s.`,`先排队、做 prefill，直到第一个 token，再 decode，每一步出一个 token。到第一个 token 的这一段，是 Serving 引擎在这组卡的达标吞吐下（高峰时）测到的首 token 延迟的第 90 百分位；用两组卡时取慢的那组。每组卡都按这个值不超过 ${f(TTFT_GOAL,0)} 秒来配。`)},
  gen:{n:{en:'Inference (rest of the answer)',zh:'推理（写完整段回答）'},d:()=>tx('Sent when done: the user waits while the model writes the rest of the answer, one token per step.','写完再发：模型一步一个 token 写完整段回答，用户一直在等。')},
  hold:{n:{en:'Output safety check',zh:'输出安全检查'},sn:{en:'Output check',zh:'输出检查'},d:()=>''},
  back:{n:{en:'Network back',zh:'网络（回程）'},d:()=>tx(`The first words travel back to the user: ${LAT.net} ms, the same as the way in.`,`第一批字传回用户：${LAT.net} ms，和发过来一样。`)},
  log:{n:{en:'Logging and evals',zh:'日志与评测'},sn:{en:'Logs, evals',zh:'日志与评测'},d:()=>tx('After the answer is complete, the conversation, the latency and the token counts are written to logs; a sample is scored offline by evals. None of it is on the path the user waits for.','回答写完以后，对话、延迟和 token 数写进日志，抽一部分离线做评测。这些都不在用户等待的路上。')}};
const ROWS=['net','gw','hist','ctx','ret','inF','route','model','hold','back','log'];
const modelTip=r=>{const two=r.used.length>1;
  return tx(`<p>Queue and prefill until the first token, then decode, one token per step.</p><p>First token: ${msT(r.tModel)}, the 90th percentile the engine measures at peak${two?' on the slower of the two pools':''}; nine messages in ten get theirs no later. Each pool is sized so that this stays within ${f(TTFT_GOAL,0)} s.</p><p>The whole answer is written ${f(r.tEnd,1)} s after Send.</p>`,
    `<p>先排队、做 prefill，到写出第一个 token；之后 decode，每一步写一个 token。</p><p>首 token：${msT(r.tModel)}，是推理引擎在高峰负载下测到的第 90 百分位${two?'（两组卡里取慢的那组）':''}，十条消息里有九条不比它慢。每组卡都按它不超过 ${f(TTFT_GOAL,0)} 秒来配。</p><p>整段回答在按下发送后第 ${f(r.tEnd,1)} 秒写完。</p>`);};
const P_=t=>`<p>${t}</p>`;
const holdTip=r=>{const s=r.s, F=s.filt;
  if(F==='off') return P_(tx('No output check: whatever the model writes reaches the user.','不查输出：模型写什么，用户就看到什么。'));
  if(!s.stream) return P_(typeof F==='number'?tx(`Without streaming the whole answer is written first, each chunk of ${F} tokens checked as it is written; when the last chunk has been checked (${LAT.chk} ms), the whole answer is sent.`,`不流式输出时，先把整段回答写完，每 ${F} 个 token 一段，写完一段查一段；最后一段查完（${LAT.chk} ms），整段一起发出去。`)
    :tx(`Without streaming the whole answer is written first, checked once (${LAT.chk} ms) and then sent.`,`不流式输出时，先把整段回答写完，查一次（${LAT.chk} ms），再发出去。`));
  if(F==='after') return P_(tx('The answer streams to the user unchecked, and the check runs once, after the whole answer has been sent: by then the user has already seen every word.','回答没查就流式发给用户，整段发完以后才查一次：这时用户已经看到了每一个字。'));
  if(F==='whole') return P_(tx(`The whole answer is held back until it is written and checked (${LAT.chk} ms per check), so the user waits for all of it.`,`整段回答先扣着，写完、查完（每次查 ${LAT.chk} ms）才发，所以用户要等整段写完。`));
  return P_(tx(`The server holds each chunk of ${F} tokens until the classifier has checked it (${LAT.chk} ms per check), then sends it. The first words wait for the first ${F} tokens to be written and checked.`,`服务端每攒够 ${F} 个 token 就交给分类器查（每次 ${LAT.chk} ms），查完才发出这一段。所以第一批字要等前 ${F} 个 token 写完、查完。`));};

const DESCS=()=>({
  guided:tx('Eight short parts. Follow one message through a chat app, then decide how it streams, what it caches, which model answers and how its output is checked, and size the GPUs for a million users.','一共 8 个部分。先跟着一条消息走一遍聊天应用，再决定怎么流式输出、缓存什么、交给哪个模型、怎么检查输出，最后给每天 1M 用户配卡。'),
  sandbox:tx('Pick a product and see when its users see the first words, how many GPUs it needs and what it costs per day.','选一个产品，看看用户多久看到第一个字、要多少张卡、每天花多少钱。'),
  challenge:tx(`Choose streaming, caching, routing and the output check for a chat app with ${dauT(CH.dau)} daily users so that the first words appear within ${f(CH.vis,1)} s at peak, every answer is checked before it is shown, and the GPUs cost at most ${usd(CH.budget)} a day.`,`为一个每天 ${dauT(CH.dau)} 用户的聊天应用选择流式输出、缓存、路由和输出检查，让高峰时首字在 ${f(CH.vis,1)} 秒内出现，回答的每一段发出前都查过，每天 GPU 花费不超过 ${usd(CH.budget)}。`)});
const OUTCOME=()=>tx('lay out the components of a chat app, budget the time before the first word, and size its GPUs and cost from its users.','画出聊天应用的各个部件，给首字之前的时间做预算，并从用户数算出要多少卡、花多少钱。');
const PRIPOP=()=>tx(`<p>A model server is measured by its first token: the ${DEF('time to first token',D_TTFT())}. The serving labs set the goal that at least 90% of chat requests get their first token within 1 s and no gap between tokens over 150 ms, and measure how many requests per second one replica can take under that goal: its ${DEF('goodput',D_GOODPUT())}. The lab ${LL('serving-capstone')} builds that engine; this lab reuses it.</p>`,
  `<p>衡量一个模型服务，先看首 token：${DEF('首 token 延迟',D_TTFT())}。Serving 各 Lab 定的目标是至少 90% 的聊天请求在 1 秒内拿到首 token、token 之间最长的间隔不超过 150 毫秒，并测出在这个目标下一份副本每秒能接多少个请求，也就是它的${DEF('达标吞吐',D_GOODPUT())}。“${LL('serving-capstone')}”这个 Lab 搭了这个引擎，这个 Lab 直接用它。</p>`);
const FIDPOP=()=>tx(
 `<p><b>Illustrative:</b> a system-design walk-through. The time of each component is the lab’s assumption; the GPUs come from the serving capstone’s engine.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The latency of every component except inference is a stated assumption (Settings notes and the hovers on each row).</li>`+
 `<li>Inference: the serving capstone’s engine, unchanged except that it takes the model and the GPU count. Each step costs the longer of its compute (989.5 TFLOP/s × 50%) and its memory reads (3.35 TB/s × 60%), both assumptions, with continuous batching, chunked prefill and prefix caching of the shared 2,048-token system prompt.</li>`+
 `<li>Goodput per replica: the highest rate on a 0.25 requests/s grid at which 90% of 200 seeded chat requests meet the goals; each pool gets the fewest replicas that keep its peak rate at or under it. That is this lab’s estimate.</li>`+
 `<li>The time to the first token is counted as the 1 s goal each pool is sized for; the gap between tokens is the 90th percentile the engine measures at goodput.</li></ul></div><div><div class="eyebrow">Simplified</div><ul>`+
 `<li>${MODELS.l.name} runs on 4 H100s with tensor parallelism; the communication between them is left out.</li>`+
 `<li>The fleet is sized for the peak hour and kept all day; scaling down at night would cost less.</li>`+
 `<li>Each output check costs 1 ms of one H100 and takes ${LAT.chk} ms: assumptions for a small classifier whose calls are batched.</li>`+
 `<li>Answer quality is not simulated: the lab assumes 70% of messages are easy enough for the small model.</li>`+
 `<li>Prices: this lab assumes $3 per GPU hour.</li></ul></div></div>`+
 `<div class="eyebrow">Sources</div><ul class="refs">`+
 `<li>${cite(SRC.aev,'anthropic-sdk-python raw_message_stream_event.py')}: the six message stream event types the SDK defines.</li><li>${cite(SRC.ostr,'openai-python _streaming.py')}: a stream ends with <code>data: [DONE]</code>.</li>`+
 `<li>${cite(SRC.lg,'Llama Guard 3-8B model card')}: 14 hazard categories, prompts and responses.</li><li>${cite(SRC.pg,'Llama Prompt Guard 2 model card')}: 86M and 22M classifiers for prompt attacks.</li>`+
 `<li>${cite(SRC.rl,'RouteLLM README')}: routers that cut cost “by up to 85%” while keeping “95% GPT-4 performance” on MT Bench.</li><li>${cite(SRC.llf,'LiteLLM fallbacks')}: fall back to another model group after retries.</li>`+
 `<li>${cite(SRC.pc,'Anthropic prompt caching cookbook')} and ${cite(SRC.apc,'vLLM automatic prefix caching')}.</li><li>${cite(SRC.rate,'OpenAI cookbook: rate limits')}: RPM and TPM limits, retry with exponential backoff.</li><li>${cite(SRC.tk,'tiktoken README')}: about 4 bytes per token.</li></ul>`,
 `<p><b>示意：</b>一次系统设计的推演。每个部件的耗时是这个 Lab 的假设；卡数来自 Serving 综合挑战的引擎。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>除了推理，每个部件的耗时都是写明的假设（见自由调参的说明和每一行的悬停）。</li>`+
 `<li>推理：Serving 综合挑战的引擎，只改成可以换模型和卡数。每一步的耗时取计算（989.5 TFLOP/s × 50%）和读显存（3.35 TB/s × 60%）中较长的那个，两个百分比都是假设；包括 continuous batching、chunked prefill，以及对共用的 2,048 个 token 的 system prompt 做 prefix caching。</li>`+
 `<li>每份副本的达标吞吐：在每秒 0.25 个请求一格的格点上，200 个带种子的聊天请求里至少 90% 达标的最高速率；每一组卡配最少的副本，让高峰速率不超过它。这是这个 Lab 的估算。</li>`+
 `<li>到第一个 token 的时间按每组卡配卡时用的 1 秒目标算；token 之间的间隔用引擎在达标吞吐下测到的第 90 百分位。</li></ul></div><div><div class="eyebrow">简化了什么</div><ul>`+
 `<li>${MODELS.l.name} 用 4 张 H100 做张量并行，卡之间的通信没有算。</li>`+
 `<li>卡按高峰那个小时配，全天都开着；夜里缩容会更便宜。</li>`+
 `<li>每次输出检查占一张 H100 1 ms、耗时 ${LAT.chk} ms：这是对一个小分类器、调用会合批处理的假设。</li>`+
 `<li>没有模拟回答质量：假设 70% 的消息小模型就能答好。</li>`+
 `<li>价格：本 Lab 假设每 GPU 每小时 3 美元。</li></ul></div></div>`+
 `<div class="eyebrow">来源</div><ul class="refs">`+
 `<li>${cite(SRC.aev,'anthropic-sdk-python raw_message_stream_event.py')}：SDK 定义的 6 种流式消息事件。</li><li>${cite(SRC.ostr,'openai-python _streaming.py')}：流以 <code>data: [DONE]</code> 结束。</li>`+
 `<li>${cite(SRC.lg,'Llama Guard 3-8B 模型卡')}：14 类风险，可以查 prompt，也可以查回答。</li><li>${cite(SRC.pg,'Llama Prompt Guard 2 模型卡')}：86M 和 22M 两个识别 prompt 攻击的分类器。</li>`+
 `<li>${cite(SRC.rl,'RouteLLM README')}：路由器能把成本最多降 85%（原文 up to 85%），同时在 MT Bench 上保持 GPT-4 95% 的表现。</li><li>${cite(SRC.llf,'LiteLLM fallbacks')}：重试失败后换到另一组模型。</li>`+
 `<li>${cite(SRC.pc,'Anthropic prompt caching cookbook')} 和 ${cite(SRC.apc,'vLLM automatic prefix caching')}。</li><li>${cite(SRC.rate,'OpenAI cookbook：限流')}：每分钟请求数（RPM）和 token 数（TPM）的上限，用指数退避重试。</li><li>${cite(SRC.tk,'tiktoken README')}：平均每个 token 约 4 字节。</li></ul>`);

/* ---------- state ---------- */
let S={...BASE}, mode='guided', step=0, root=null, app=null, pred=null, saved=null;
let tried={}, observed={}, userAct=false, annT=0, stageT=0, shownPrev={};
let chSubmitted=false, chReveal=false, drill=null, drillOpen=false, ro=null;
/* a choice in the card brings its observation into view once it shows (after the replay), as the kit does for its own levers (P11) */
let obsReveal=false;
/* the message player. One clock (PL.t, seconds after Send) drives the cursor, the bars, the slider and the title (B2). The wall time of a
   play is planned in pieces: before the first word every component is shown for at least DWELL (R7.11: the 5 ms gateway is seen too),
   longer ones in proportion at the axis rate; the first word holds HOLDFW; then the answer goes on streaming to the end of the axis */
const PL={on:false,raf:0,t:0,w:0,last:0,quick:false,sig:'',done:true,plan:null};
let spd='med';
const SPEEDS={slow:12,med:7,fast:3.5};     // seconds per length of the time axis at the axis rate (R3.9: a step-by-step replay defaults to 中)
const DWELL={slow:0.8,med:0.5,fast:0.25};  // the least time each component is shown
const HOLDFW={slow:0.9,med:0.6,fast:0.3};  // the pause on the first word
const QUICK=1.2;                           // Scenarios, Settings and the challenge, switched back and forth: the end within 1.5 s (R3.9)
const watched=new Set();
let nPlays=0;
const R_=()=>compute(S);

/* ---------- guided parts ---------- */
const PRED=['p10','p37','p70'];
const predTxt=v=>({p10:tx('About 10% of before','约为原来的 10%'),p37:tx('About 37% of before','约为原来的 37%'),p70:tx('About 70% of before','约为原来的 70%')}[v]);
const PRED_RIGHT='p37';
const sid=()=>STEPS[step].id;
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const seenPart=id=>mode!=='guided'||step>=SIX(id);
const pick={};
const LEV={
  stream:{key:'stream',opts:[{v:1,t:{en:'Stream the answer',zh:'流式输出'},d:{en:'Send every few tokens as they are written (checked first)',zh:'写出几个 token 就发（先查过）'}},{v:0,t:{en:'Send it when it is done',zh:'写完再发'},d:{en:'Wait for the whole 300-token answer',zh:'等整段 300 个 token 写完'}}],
    res:v=>{const r=compute({...S,stream:v}); return tx(`first word ${sT(r.vis)}`,`首字 ${sT(r.vis)}`);}},
  cache:{key:'cache',opts:[{v:0,t:{en:'No prompt caching',zh:'不缓存'},d:{en:'Every message runs prefill on its whole prompt',zh:'每条消息都对整个 prompt 做 prefill'}},{v:1,t:{en:'Cache the shared system prompt',zh:'缓存共用的 system prompt'},d:{en:'Its 2,048 tokens are computed once and reused',zh:'这 2,048 个 token 只算一次，之后直接复用'}}],
    res:v=>{const r=compute({...S,cache:v}); return tx(`${I(r.total)} GPUs`,`${I(r.total)} 张卡`);}},
  share:{key:'share',opts:SHARES.filter(v=>v!==0.3).map(v=>({v,t:{en:v?`${pc(v)} to the small model`:'Everything to the large model',zh:v?`${pc(v)} 交给小模型`:'全部交给大模型'},d:{en:v===0?'No router':v===0.9?'More than the easy share':v===EASY?'All the easy messages':'Half of the messages',zh:v===0?'不用路由':v===0.9?'超过了简单消息的比例':v===EASY?'所有简单的消息':'一半的消息'}})),
    res:v=>{const r=compute({...S,share:v}); return tx(`${usd(r.costDay)} a day`,`每天 ${usd(r.costDay)}`);}},
  filt:{key:'filt',opts:['after','whole',16,8,4].map(v=>({v,t:()=>filtT(v),d:{en:v==='after'?'The user sees it before it is checked':v==='whole'?'One check, then send everything':`${cpsT(WL.outMean/v)} checks per answer on average`,zh:v==='after'?'用户先看到，后检查':v==='whole'?'查一次，再全部发出':`每个回答平均查 ${cpsT(WL.outMean/v)} 次`}})),
    res:v=>{const r=compute({...S,filt:v}); return v==='after'?tx('unchecked words shown','没查就显示了'):tx(`first word ${sT(r.vis)}`,`首字 ${sT(r.vis)}`);}},
  dau:{key:'dau',opts:[1e5,1e6,1e7].map(v=>({v,t:()=>tx(`${dauT(v)} daily users`,`每天 ${dauT(v)} 用户`),d:{en:'10 messages each, peak 3 times the average',zh:'每人 10 条消息，高峰是平均的 3 倍'}})),
    res:v=>{const r=compute({...S,dau:v}); return tx(`${I(r.total)} GPUs`,`${I(r.total)} 张卡`);}}};
const PART_LEV={stream:'stream',context:'cache',route:'share',safety:'filt',size:'dau'};
const LEAD={
  stream:()=>tx('Pick when the answer is sent and watch when the first word appears in the chart.','选回答什么时候发出去，看看图里第一个字什么时候出现。'),
  cache:()=>tx('Pick whether the shared system prompt is cached and watch the GPUs the same traffic needs.','选要不要缓存共用的 system prompt，看看同样的流量要多少张卡。'),
  share:()=>tx('Pick how much of the traffic the router sends to the small model and watch the GPUs and the daily cost.','选路由把多少流量交给小模型，看看卡数和每天的花费。'),
  filt:()=>tx('Pick how the output is checked and watch when the first word appears, and whether it was checked first (the system prompt stays cached and 70% of messages still go to the small model).','选怎么检查输出，看看第一个字什么时候出现、出现之前查过没有（system prompt 仍然缓存，70% 的消息仍交给小模型）。'),
  /* part 7 keeps the design built so far, said right above its options (P14), so its first two sentences are the method (N2) */
  dau:()=>tx(`Pick the number of daily users and watch the GPUs and the cost (all with the design built so far: the system prompt cached, ${pc(EASY)} of the messages to the small model, the output checked every ${BASE.filt} tokens).`,`选每天的用户数，看看卡数和花费（都用前面搭好的设计：缓存 system prompt，${pc(EASY)} 的消息交给小模型，每 ${BASE.filt} 个 token 查一次）。`)};
const levTried=(k,v)=>!!(tried[k]&&tried[k].has(String(v)));
function levers(id){const k=PART_LEV[id], lv=LEV[k], cur=String(S[k]);
  return `<p>${LEAD[k]()}</p>`+A.ui.levers({aria:tx('Options: pick one','选项：选一个'),value:cur,
    options:lv.opts.map(o=>({v:String(o.v),title:typeof o.t==='function'?o.t():L(o.t),desc:typeof o.d==='function'?o.d():L(o.d),result:levTried(k,o.v)?esc(lv.res(o.v)):''}))}).replace(/data-lever=/g,`data-lkey="${k}" data-lever=`);}
const playBtns=()=>`<div class="pdz-tbtns"><button type="button" class="btn sm" data-act="play" data-card="1">${cardPlayTxt()}</button></div>`;
const midPlay=()=>!PL.on&&!PL.done&&PL.t>0;
/* the stage's play button says what it does (P6): pause, continue from where it stopped, or play from the start; fixed width (R7.14) */
const playTxt=()=>PL.on?tx('❚❚ Pause','❚❚ 暂停'):midPlay()?tx('▶ Continue','▶ 继续'):tx('▶ Play','▶ 播放');
/* part 1's card button: the first time it sends the message */
const cardPlayTxt=()=>PL.on||midPlay()?playTxt():(watched.has(sigOf(R_()))?tx('▶ Replay the message','▶ 重放这条消息'):tx('▶ Send the message','▶ 发送这条消息'));

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'One message: where does the time go before the first word?',zh:'一条消息：第一个字出现之前，时间花在哪里？'},expect:()=>watched.has(sigOf(compute(baseFor('mission')))),showMe:'mission',
  body:()=>{const b=compute(baseFor('mission')), k=f(b.costDay/BRIEF.budget,1);
   return tx(`<p>You are designing a chat app like ChatGPT for ${dauT(BASE.dau)} daily users, with a GPU budget of ${usd(BRIEF.budget)} a day. The design as it stands needs ${usd(b.costDay)} a day, ${k} times the budget: how do you get within the budget and still meet the product team’s two other requirements?</p><ul><li>At the busiest hour, the time from pressing Send to the first word on screen (the ${DEF('time to the first visible word',D_VIS())}, “first word” below) is at most ${f(BRIEF.vis,1)} s.</li><li>Every chunk of an answer passes a safety check before the user sees it.</li></ul>${playBtns()}`,
    `<p>你要设计一个像 ChatGPT 的聊天应用，每天 ${dauT(BASE.dau)} 用户，GPU 的预算是每天 ${usd(BRIEF.budget)}。现在这套设计每天要 ${usd(b.costDay)}，是预算的 ${k} 倍：怎样把花费降到预算以内，又守住产品团队的另外两条要求？</p><ul><li>高峰时，从按下发送到第一个字出现不超过 ${f(BRIEF.vis,1)} 秒（这段时间叫“${DEF('首字可见时间',D_VIS())}”，简称“首字”）；</li><li>回答的每一段，先过安全检查再给用户看。</li></ul>${playBtns()}`);},
  obs:()=>obsMission()},
 {id:'stream',title:{en:'Streaming',zh:'流式输出'},lead:{en:'Send the answer as it is written, or when it is done?',zh:'回答边写边发，还是写完再发？'},expect:()=>levTried('stream',1)&&levTried('stream',0),showMe:'stream',
  body:()=>tx(`<p>The model writes an answer one token per step, about 50 ms a step here, so a 300-token answer takes more than ten seconds to finish. ${DEF('Streaming',D_STREAM())} sends the words as they are written (each chunk is checked first; the safety part shows how), so the user starts reading after the first chunk instead of after the whole answer.</p>${levers('stream')}`,
    `<p>模型一步只写一个 token，这里每一步约 50 ms，所以 300 个 token 的回答要十几秒才写完。${DEF('流式输出',D_STREAM())}写出几个 token 就发出去（发之前先查过，后面的“安全过滤”部分讲怎么查），用户读到第一段就能开始看，不用等整段写完。</p>${levers('stream')}`),
  obs:()=>obsStream()},
 {id:'context',title:{en:'Context and caching',zh:'上下文和缓存'},lead:{en:'Every message resends the conversation: cache the part every prompt shares.',zh:'每条消息都要重发整段对话：把每个 prompt 都有的那部分缓存起来。'},expect:()=>levTried('cache',0)&&levTried('cache',1),showMe:'context',
  body:()=>tx(`<p>Every message starts with the same 2,048-token system prompt, and the model remembers nothing between messages, so each one runs prefill on it again. With ${DEF('prompt caching',D_CACHE())} its KV cache is computed once and reused; with less to compute, each ${DEF('replica',D_REPL())} takes more messages while still ${DEF('meeting the goals',D_MEET())}, so the same traffic needs fewer GPUs (the lab ${LL('prefix-caching')} shows how caching works).</p>${levers('context')}`,
    `<p>每条消息开头都是同一段 2,048 个 token 的 system prompt，模型不记得上一条，每条都要重新 prefill 它。用了 ${DEF('prompt 缓存',D_CACHE())}，它的 KV cache 只算一次、之后直接复用；要算的少了，每份${DEF('副本',D_REPL())}在${DEF('达标',D_MEET())}的前提下能多接消息，同样的流量要的卡就少（原理见“${LL('prefix-caching')}”这个 Lab）。</p>${levers('context')}`),
  obs:()=>obsContext()},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Send 70% of messages to a model 10 times cheaper: what is left of the cost?',zh:'70% 的消息交给便宜 10 倍的模型：花费还剩多少？'},gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(v=>`<button type="button" class="choice" data-pred="${v}" aria-pressed="${pred===v}">${esc(predTxt(v))}</button>`).join('')}</div>`;
   return tx(`<p>So far every message goes to the large model, ${MODELS.l.name}, with the system prompt cached. Many chat messages are easy, though, and a small model such as ${MODELS.s.name} answers them well (this lab assumes ${pc(EASY)} are); per message, it costs about a tenth as much.</p><p>Suppose 70% of the messages go to the small model and the other 30% stay on the large one. The daily cost of all-large is 100%. About how much is it now?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
     `<p>到现在为止，每条消息都交给大模型 ${MODELS.l.name}（system prompt 已缓存）。可是很多聊天消息很简单，${MODELS.s.name} 这样的小模型就能答好（这个 Lab 假设有 ${pc(EASY)}）；按每条消息算，它的花费只有大模型的十分之一左右。</p><p>假设 70% 的消息交给小模型，剩下 30% 仍然交给大模型。全部用大模型时每天的花费算 100%，现在大约是多少？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'route',title:{en:'Routing',zh:'模型路由'},lead:{en:'A router picks the model per message; cost falls, but not to a tenth.',zh:'路由按消息选模型：花费降了，但降不到十分之一。'},expect:()=>levTried('share',EASY)&&levTried('share',0.9),showMe:'route',
  body:()=>tx(`<p>A ${DEF('model router',D_ROUTE())} reads each message: easy ones go to the cheap small model, hard ones to the large model. This lab assumes ${pc(EASY)} of the messages are easy enough for the small model; send more than ${pc(EASY)} to it and hard messages go to a model that answers them badly.</p>${levers('route')}`,
    `<p>${DEF('模型路由',D_ROUTE())}先读每条消息，简单的交给便宜的小模型，难的交给大模型。这个 Lab 假设 ${pc(EASY)} 的消息小模型能答好；交给小模型的超过 ${pc(EASY)}，就有难题交给了答不好的小模型。</p>${levers('route')}`),
  obs:()=>obsRoute()},
 {id:'safety',title:{en:'Safety filters',zh:'安全过滤'},lead:{en:'Check the input before the model, and the output before the user reads it.',zh:'输入在模型之前查，输出在用户读到之前查。'},expect:()=>levTried('filt','after')&&!!tried.filt&&tried.filt.size>=2,showMe:'safety',
  body:()=>tx(`<p>The output check must read the answer before the user does, but a streamed answer leaves in pieces. So the server collects N tokens at a time, has a ${DEF('safety filter',D_SAFE())} check them, and sends the piece only when it passes.</p><p>A smaller N shows the first word sooner but needs more checks per answer. Each check takes about ${LAT.chk} ms, which the user waits, and the checks ${DEF('keep some GPUs busy',D_CHKGPU())}.</p>${levers('safety')}`,
    `<p>输出检查要在用户之前读到回答，可流式输出的回答是一段一段发出去的。所以服务端每攒够 N 个 token，先交给${DEF('安全过滤',D_SAFE())}的分类器查，查过才发出这一段。</p><p>N 越小，第一个字出现得越早，但每个回答要查的次数越多。每次检查约 ${LAT.chk} ms，这是用户多等的时间；检查也要${DEF('占一些卡',D_CHKGPU())}。</p>${levers('safety')}`),
  obs:()=>obsSafety()},
 {id:'size',title:{en:'Sizing: users to GPUs',zh:'估算：从用户数到卡数'},lead:{en:'Daily users → requests per second at peak → GPUs → dollars.',zh:'每天的用户 → 高峰每秒请求数 → 卡数 → 美元。'},expect:()=>!!(tried.dau&&tried.dau.size>=2),showMe:'size',
  body:()=>tx(`<p>Work out the messages per second in the busiest hour (the ${DEF('peak factor',D_PEAK())} is how many times busier it is than the day’s average):</p><div class="pdz-eqs pdz-eq1"><span class="pdz-eq-r">${M(`\\lambda_{${tt('peak')}} = \\frac{${tt('users')} \\times ${tt('messages each')}}{86{,}400} \\times ${tt('peak factor')}`)}</span></div><p>Then give each pool enough ${DEF('replicas',D_REPL())} that none takes more than its ${DEF('goodput',D_GOODPUT())}.</p>${levers('size')}`,
    `<p>先算最忙那个小时每秒有多少条消息（${DEF('高峰倍数',D_PEAK())}是它比全天平均忙几倍）：</p><div class="pdz-eqs pdz-eq1"><span class="pdz-eq-r">${M(`\\lambda_{${tt('高峰')}} = \\frac{${tt('用户数')} \\times ${tt('每人消息数')}}{86{,}400} \\times ${tt('高峰倍数')}`)}</span></div><p>再给每组卡配够${DEF('副本',D_REPL())}，让每份接的消息不超过它的${DEF('达标吞吐',D_GOODPUT())}。</p>${levers('size')}`),
  obs:()=>obsSize()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  body:()=>rememberHTML()}];

/* ---------- observations (R6.20: the trend, one key number, with its baseline) ---------- */
const okSpan=(ok,t)=>`<b class="${ok?'pdz-ok':'pdz-bad'}">${t}</b>`;
function obsMission(){const r=R_(), out=r.vis-r.tModel-r.hold-r.gen;
  return tx(`<p>The first word appears ${okSpan(r.vis<=BRIEF.vis,sT(r.vis))} after Send, within the ${f(BRIEF.vis,1)} s budget.</p><p>Of that, the model’s first token takes ${sT(r.tModel)}. The output check takes ${sT(r.hold+r.gen)}: it waits until the first ${S.filt} tokens are written and checked before it releases the first chunk. All the other rows together take only ${sT(out)}.</p>`,
    `<p>按下发送 ${okSpan(r.vis<=BRIEF.vis,sT(r.vis))}后，第一个字出现，在 ${f(BRIEF.vis,1)} 秒的预算以内。</p><p>其中模型写出第一个 token 占 ${sT(r.tModel)}；输出检查占 ${sT(r.hold+r.gen)}，它要等前 ${S.filt} 个 token 都写完、查过，才放出第一段；其余几行加起来只占 ${sT(out)}。</p>`);}
function obsStream(){const on=compute({...S,stream:1}), off=compute({...S,stream:0}), both=levTried('stream',1)&&levTried('stream',0);
  if(!both) return S.stream?tx(`<p>Streaming: the first word appears ${sT(on.vis)} after Send; the rest of the answer follows chunk by chunk.</p>`,`<p>流式输出：按下发送 ${sT(on.vis)}后出现第一个字，回答的其余部分一段段跟上来。</p>`)
    :tx(`<p>Sent when done: the user stares at an empty screen for ${okSpan(false,sT(off.vis))}, while the whole 300-token answer is written.</p>`,`<p>写完再发：用户对着空白的屏幕等了 ${okSpan(false,sT(off.vis))}，等整段 300 个 token 写完。</p>`);
  return tx(`<p>Streaming shows the first word after ${okSpan(true,sT(on.vis))}; sending the finished answer takes ${okSpan(false,sT(off.vis))}, about ${f(off.vis/on.vis,0)} times as long.</p><p>Either way the model writes the same tokens and the output is checked the same way (the safety part compares the ways to check), so the GPUs are the same: streaming changes only when the words are sent, and costs nothing.</p>`,
    `<p>流式输出 ${okSpan(true,sT(on.vis))}就出现第一个字；写完再发要 ${okSpan(false,sT(off.vis))}，约是前者的 ${f(off.vis/on.vis,0)} 倍。</p><p>两种做法里，模型写的 token 一样多，输出检查的做法也一样（后面的“安全过滤”部分再比较几种查法），所以卡数一样：流式输出只改变了什么时候把字发出去，不多花钱。</p>`);}
function obsContext(){const a=compute({...S,cache:0}), b=compute({...S,cache:1}), both=levTried('cache',0)&&levTried('cache',1);
  if(!both) return tx(`<p>${S.cache?'With the cache':'Without the cache'}, this traffic needs ${I(compute(S).total)} GPUs at peak. Try the other option.</p>`,`<p>${S.cache?'缓存时':'不缓存时'}，这样的流量高峰时要 ${I(compute(S).total)} 张卡。再试另一个选项。</p>`);
  const gx=b.P.l.g/a.P.l.g, enG=gx>2&&gx<3?'more than twice as many':'more', zhG=gx>2&&gx<3?'多一倍多':'更多';
  /* the replicas are sized to the same goals, so the saved compute becomes messages, not a faster first token (P34) */
  return tx(`<p>With the cache, each replica takes ${enG} messages: the same traffic needs ${okSpan(true,I(b.total))} GPUs instead of ${I(a.total)}, ${usd(b.costDay)} a day instead of ${usd(a.costDay)}.</p><p class="small">The replicas are busier, so each token takes a little longer to write: the first word moves from ${sT(a.vis)} to ${sT(b.vis)}, still within the budget.</p>`,
    `<p>缓存以后，每份副本能接${zhG}的消息：卡从 ${I(a.total)} 张降到 ${okSpan(true,I(b.total))} 张，每天从 ${usd(a.costDay)} 降到 ${usd(b.costDay)}。</p><p class="small">副本更忙了，每个 token 写得稍慢，首字从 ${sT(a.vis)}变成 ${sT(b.vis)}，仍在预算以内。</p>`);}
/* the GPUs of a routed design, by pool; they add up to the total, as the fleet draws them (P23) */
function routeParts(r){const k=r.P.l.gpu?r.P.s.gpu/r.P.l.gpu:0;
  return tx(`<ul class="small"><li>the large-model pool: ${I(r.gpus.l)}; it still serves ${pc(1-r.s.share)} of the messages, at about ${f(k,0)} times the cost per message</li><li>the small-model pool: ${I(r.gpus.s)}</li><li>the output checks: ${I(r.fG)}, which routing does not reduce</li></ul>`,
    `<ul class="small"><li>大模型那组 ${I(r.gpus.l)} 张：它仍要服务 ${pc(1-r.s.share)} 的消息，每条消息的花费约是小模型的 ${f(k,0)} 倍</li><li>小模型那组 ${I(r.gpus.s)} 张</li><li>输出检查 ${I(r.fG)} 张：不随路由减少</li></ul>`);}
function predVerdict(){if(!pred) return ''; const r0=compute({...S,share:0}), r7=compute({...S,share:EASY}), x=r7.costDay/r0.costDay, ok=pred===PRED_RIGHT, at7=Math.abs(S.share-EASY)<1e-9;
  const you=tx(`${ok?'✓':'✗'} You predicted “${predTxt(pred)}”.`,`${ok?'✓':'✗'} 你预测“${predTxt(pred)}”。`);
  if(at7) return `<p class="small">${you}${tx(` The result is ${pc(x)}${ok?', closest to your pick':''}.`,`结果是 ${pc(x)}${ok?'，你选的最接近':''}。`)}</p>`;
  return `<p class="small">${you}${tx(` Sending ${pc(EASY)} of the messages to the small model costs ${pc(x)} of before (${usd(r7.costDay)} a day): the GPUs fall from ${I(r0.total)} to ${I(r7.total)}, which are`,`把 ${pc(EASY)} 的消息交给小模型，花费是原来的 ${pc(x)}（每天 ${usd(r7.costDay)}）：卡从 ${I(r0.total)} 张降到 ${I(r7.total)} 张，其中`)}</p>`+routeParts(r7);}
function obsRoute(){const r0=compute({...S,share:0}), r=R_(), v=S.share;
  let h;
  if(v===0) h=tx(`<p>Everything on the large model: ${I(r.total)} GPUs, ${usd(r.costDay)} a day.</p>`,`<p>全部用大模型：${I(r.total)} 张卡，每天 ${usd(r.costDay)}。</p>`);
  else if(v>EASY+1e-9) h=tx(`<p>${pc(v)} to the small model is cheaper still, ${usd(r.costDay)} a day.</p><p>But only ${pc(EASY)} of the messages are easy. The other ${pc(v-EASY)} are hard questions the small model ${okSpan(false,'answers badly')} (the red squares in the chart): the saving is paid in answer quality.</p>`,
    `<p>${pc(v)} 交给小模型还能更便宜，每天 ${usd(r.costDay)}。</p><p>可是只有 ${pc(EASY)} 的消息是简单的，多出来的 ${pc(v-EASY)} 是难题，小模型${okSpan(false,'答不好')}（图里标红的方块）：省下的钱是拿回答质量换的。</p>`);
  else {const inB=r.costDay<=BRIEF.budget+1e-9;
    h=tx(`<p>With ${pc(v)} to the small model the GPUs drop from ${I(r0.total)} to ${okSpan(true,I(r.total))}: ${usd(r.costDay)} a day, ${pc(r.costDay/r0.costDay)} of all-large${inB?`, within the ${usd(BRIEF.budget)} budget`:''}. The ${I(r.total)} GPUs are</p>`,
      `<p>${pc(v)} 交给小模型以后，卡从 ${I(r0.total)} 张降到 ${okSpan(true,I(r.total))} 张，每天 ${usd(r.costDay)}，是全部用大模型的 ${pc(r.costDay/r0.costDay)}${inB?`，在 ${usd(BRIEF.budget)} 的预算以内`:''}。这 ${I(r.total)} 张卡是：</p>`)+routeParts(r);}
  return h+predVerdict();}
function obsSafety(){const r=R_(), F=S.filt;
  if(F==='after') return tx(`<p>The first word appears after ${sT(r.vis)}, the fastest option, but ${okSpan(false,'the answer is checked only after all of it has been sent')}. If the classifier flags it, the user has already seen it.</p>`,`<p>第一个字 ${sT(r.vis)}就出现了，是最快的；可是${okSpan(false,'回答整段发给用户以后才查')}。分类器就算查出问题，用户也已经看到了。</p>`);
  if(F==='whole') return tx(`<p>Every word is checked first, but the user waits for the whole answer: ${okSpan(false,sT(r.vis))}, far over the ${f(BRIEF.vis,1)} s budget.</p>`,`<p>每个字都先查过，可是用户要等整段回答写完：${okSpan(false,sT(r.vis))}，远超 ${f(BRIEF.vis,1)} 秒的预算。</p>`);
  const ok=r.vis<=BRIEF.vis+1e-9;
  return tx((ok?`<p>Checking every ${F} tokens, the first word appears after ${okSpan(true,sT(r.vis))}, within the ${f(BRIEF.vis,1)} s budget.</p>`:`<p>Checking every ${F} tokens, the first chunk is checked only when all ${F} of its tokens are written, so the first word appears after ${okSpan(false,sT(r.vis))}, over the ${f(BRIEF.vis,1)} s budget.</p>`)+`<p>An answer is checked ${cpsT(r.cps)} times on average, and at peak the checks keep ${I(r.fG)} GPUs busy.</p>`,
    (ok?`<p>每 ${F} 个 token 查一次，第一个字 ${okSpan(true,sT(r.vis))}出现，在 ${f(BRIEF.vis,1)} 秒的预算以内。</p>`:`<p>每 ${F} 个 token 查一次，第一段要等这 ${F} 个 token 全写完才查，第一个字到 ${okSpan(false,sT(r.vis))}才出现，超过了 ${f(BRIEF.vis,1)} 秒的预算。</p>`)+`<p>每个回答平均查 ${cpsT(r.cps)} 次，高峰时检查要占 ${I(r.fG)} 张卡。</p>`);}
function obsSize(){const r=R_(), both=tried.dau&&tried.dau.size>=2, rows=[1e5,1e6,1e7].map(d=>compute({...S,dau:d}));
  const k1=rows[1].total/rows[0].total, k2=rows[2].total/rows[1].total, ten=k2>9.5&&k2<10.5;
  return tx(`<p>${dauT(S.dau)} daily users send ${rT(r.lam)} messages per second at peak. That takes ${I(r.total)} GPUs, ${usd(r.costDay)} a day.</p>`+(both?`<p>From ${dauT(1e6)} to ${dauT(1e7)} users the GPUs grow from ${I(rows[1].total)} to ${I(rows[2].total)}, ${ten?'almost exactly ten times':f(k2,1)+' times'}; from ${dauT(1e5)} to ${dauT(1e6)} only ${f(k1,1)} times, because with few users every pool is rounded up to whole replicas and the remainder is a larger share.</p><p>The first word stays at ${sT(r.vis)}: the fleet is sized to the goal, so more users cost GPUs, not latency.</p>`:''),
    `<p>每天 ${dauT(S.dau)} 用户，高峰时每秒 ${rT(r.lam)} 条消息，要 ${I(r.total)} 张卡，每天 ${usd(r.costDay)}。</p>`+(both?`<p>从 ${dauT(1e6)} 到 ${dauT(1e7)} 用户，卡从 ${I(rows[1].total)} 张涨到 ${I(rows[2].total)} 张，${ten?'差不多正好 10 倍':`是 ${f(k2,1)} 倍`}；从 ${dauT(1e5)} 到 ${dauT(1e6)} 只涨了 ${f(k1,1)} 倍，因为用户少时每组卡都要凑整份副本，零头占比大。</p><p>第一个字一直在 ${sT(r.vis)}出现：卡按目标配，用户多了只多花卡，不多花时间。</p>`:''));}
/* part 8: the design's result against the brief first, then the three points, each with the misconception it answers (P16) */
function rememberHTML(){const r=compute(baseFor('remember')), r0=compute({...baseFor('remember'),share:0}), path=r.vis-r.tModel-r.hold-r.gen;   // the other rows of the path (N1)
  return tx(`<p>The design built so far meets all three requirements: the first word ${sT(r.vis)} after Send (budget ${f(BRIEF.vis,1)} s), every chunk of ${r.s.filt} tokens checked before it is sent, and ${usd(r.costDay)} of GPUs a day (budget ${usd(BRIEF.budget)}).</p>`+
    `<ul><li><b>Budget the time to the first visible word; the model’s first token is one line of it.</b> ✗ “Latency is model inference”: of the ${sT(r.vis)}, the model’s first token is ${sT(r.tModel)}; the rest is ${msT(path)} on the other components of the path and ${msT(r.hold)} while the output check holds the first chunk, and a faster first token shortens neither.</li>`+
    `<li><b>Check the output as it streams, before each chunk is sent.</b> ✗ “Run the safety filter once at the end”: with streaming, by the end the user has read the whole answer.</li>`+
    `<li><b>Route by difficulty, not by cost alone.</b> ✗ “The more goes to the small model, the better”: sending the easy messages to the small model cuts the daily cost from ${usd(r0.costDay)} to ${usd(r.costDay)}; send more and hard messages go to the small model, which answers them badly. Measure the share of easy messages on your own traffic with your own eval set (see the lab ${LL('eval-ci')}).</li></ul>${A.shell.toScenarios()}`,
    `<p>这套设计三条要求都满足了：${sT(r.vis)}出第一个字（预算 ${f(BRIEF.vis,1)} 秒），每 ${r.s.filt} 个 token 查过再发，每天 ${usd(r.costDay)}（预算 ${usd(BRIEF.budget)}）。</p>`+
    `<ul><li><b>给首字做预算，模型的首 token 只是其中一项。</b>✗“延迟就是模型推理”：${sT(r.vis)}里首 token 只占 ${sT(r.tModel)}，其余是路上其他部件的 ${msT(path)} 和输出检查扣住第一段的 ${msT(r.hold)}，首 token 再快也省不掉。</li>`+
    `<li><b>输出边流式边查，每一段查过再发。</b>✗“安全过滤放最后查一次就行”：流式输出时，最后再查，用户早就读完了。</li>`+
    `<li><b>路由按难度分，不只看成本。</b>✗“交给小模型的越多越好”：简单消息交给小模型，每天从 ${usd(r0.costDay)} 降到 ${usd(r.costDay)}；再多交，难题就给了答不好的小模型。简单消息的比例要在自己的流量上用评测集量（见“${LL('eval-ci')}”这个 Lab）。</li></ul>${A.shell.toScenarios()}`);}

/* ---------- scenarios ---------- */
const PRESETS=[
 {t:{en:'A new chat app',zh:'刚上线的聊天应用'},d:()=>tx(`${dauT(1e5)} users, one model, cached`,`${dauT(1e5)} 用户，一个模型，有缓存`),p:{dau:1e5,msgs:10,peak:3,stream:1,cache:1,ret:0,share:0,filt:8,price:3}},
 {t:{en:'Consumer assistant at scale',zh:'大规模消费级助手'},d:()=>tx(`${dauT(1e7)} users, routed`,`${dauT(1e7)} 用户，有路由`),p:{dau:1e7,msgs:10,peak:3,stream:1,cache:1,ret:0,share:0.7,filt:8,price:3}},
 {t:{en:'Help-center assistant (example)',zh:'帮助中心助手（示例）'},d:{en:'Lumo Home, with retrieval; 5 messages per user a day, half to the small model',zh:'小鹿家居，带检索；每人每天 5 条消息，一半交给小模型'},p:{dau:3e5,msgs:5,peak:3,stream:1,cache:1,ret:1,share:0.5,filt:8,price:3}},
 {t:{en:'No streaming',zh:'不流式输出'},d:()=>tx(`${dauT(1e6)} users, answer sent when done`,`${dauT(1e6)} 用户，写完再发`),p:{dau:1e6,msgs:10,peak:3,stream:0,cache:1,ret:0,share:0.7,filt:8,price:3}},
 {t:{en:'Everything on the large model',zh:'全部交给大模型'},d:()=>tx(`${dauT(1e6)} users, no cache, no router`,`${dauT(1e6)} 用户，不缓存、不路由`),p:{dau:1e6,msgs:10,peak:3,stream:1,cache:0,ret:0,share:0,filt:8,price:3}}];
const KEYS=['dau','msgs','peak','stream','cache','ret','share','filt','price'];
const presetNow=()=>PRESETS.findIndex(p=>KEYS.every(k=>String(p.p[k])===String(S[k])));
function insightHTML(){const r=R_(), i=presetNow(), okV=r.vis<=BRIEF.vis+1e-9;
  let h=tx(`<p>${dauT(S.dau)} daily users, ${rT(r.lam)} messages per second at peak; the first word appears after ${okSpan(okV,sT(r.vis))}.</p><p>The fleet is ${I(r.total)} GPUs, ${usd(r.costDay)} a day, ${usd(r.perUserMonth)} per user per month.</p>`,
    `<p>每天 ${dauT(S.dau)} 用户，高峰每秒 ${rT(r.lam)} 条消息；第一个字 ${okSpan(okV,sT(r.vis))}后出现。</p><p>要 ${I(r.total)} 张卡，每天 ${usd(r.costDay)}，每个用户每月 ${usd(r.perUserMonth)}。</p>`);
  const note={0:tx('One model keeps the design simple; at this size the large model’s replicas are only a few dozen GPUs.','只用一个模型，设计最简单；在这个规模下，大模型的副本也只占几十张卡。'),
    1:(()=>{const nr=compute({...S,share:0}), nc=compute({...S,cache:0}); return tx(`At this size routing and caching decide most of the bill: without routing it would be ${usd(nr.costDay)} a day, without caching ${usd(nc.costDay)}, against ${usd(r.costDay)} now.`,`到这个规模，账单主要由路由和缓存决定：不路由每天要 ${usd(nr.costDay)}，不缓存要 ${usd(nc.costDay)}，现在是 ${usd(r.costDay)}。`);})(),
    2:tx(`Retrieval adds ${LAT.ret} ms before the model and 1,536 tokens to every prompt, so each replica takes fewer messages. How retrieval finds the passages: the labs ${LL('embeddings')}, ${LL('vector-index')} and ${LL('chunk-rerank')}.`,`检索在模型之前多了 ${LAT.ret} ms，每个 prompt 多了 1,536 个 token，所以每份副本能接的消息变少了。检索怎么找到这几段，见“${LL('embeddings')}”“${LL('vector-index')}”和“${LL('chunk-rerank')}”这几个 Lab。`),
    3:tx('The GPUs are exactly the same as with streaming; only the wait for the first word changes.','卡和流式输出时一模一样，变的只是等第一个字的时间。'),
    4:tx('Without the cache every prompt is computed in full, and every message pays the large model’s price.','不缓存时每个 prompt 都要从头算，每条消息都按大模型的价钱付。')}[i];
  return h+(note?`<p class="small">${note}</p>`:'');}

/* ---------- challenge ---------- */
const chName=e=>[e.stream?tx('streaming','流式'):tx('sent when done','写完再发'),e.cache?tx('cache','缓存'):tx('no cache','不缓存'),tx(`${pc(e.share)} small`,`小模型 ${pc(e.share)}`),filtName(e.filt)].join(tx(', ','，'));
const chMe=()=>chEval({stream:S.stream,cache:S.cache,share:S.share,filt:S.filt});
function chMiss(e){const m=[]; if(!e.okS) m.push(tx('an answer is shown before it is checked','有回答没查就显示了')); if(!e.okQ) m.push(tx(`more than ${pc(EASY)} go to the small model, so some hard messages do`,`交给小模型的超过了 ${pc(EASY)}，有难题交给了小模型`)); if(!e.okV) m.push(tx(`the first word comes after ${sT(e.r.vis)}`,`第一个字要 ${sT(e.r.vis)}才出现`)); if(!e.okB) m.push(tx(`the GPUs cost ${usd(e.r.costDay)} a day`,`GPU 每天要 ${usd(e.r.costDay)}`)); return m.join(tx('; ','；'));}
function chHint(e){
  if(!e.okS) return tx('Hint: which output checks run before the user sees the words?','提示：哪种输出检查是在用户看到字之前查的？');
  if(!e.stream||e.filt==='whole') return tx('Hint: while the user waits for the whole answer, no budget of a second or two can be met.','提示：用户要等整段回答写完时，一两秒的预算不可能达到。');
  if(!e.okQ) return tx(`Hint: only ${pc(EASY)} of the messages are easy enough for the small model.`,`提示：只有 ${pc(EASY)} 的消息小模型能答好。`);
  if(!e.okV) return tx('Hint: the first word waits for the first checked chunk; a smaller chunk is checked sooner.','提示：第一个字要等第一段查完；段越短，查得越早。');
  if(!e.cache) return tx('Hint: every prompt starts with the same system prompt; what if it were computed only once?','提示：每个 prompt 开头都是同一段 system prompt，只算一次会怎样？');
  return tx('Hint: which pool of GPUs is the most expensive per message, and how much of the traffic can leave it?','提示：哪一组卡每条消息最贵？有多少流量可以从它那里移走？');}
function chResultHTML(){
  if(!chSubmitted) return '';
  const all=chAll(), me=chMe(), best=chBest(), same=e=>e.stream===me.stream&&e.cache===me.cache&&e.share===me.share&&String(e.filt)===String(me.filt), isBest=best&&same(best);
  let h;
  if(me.ok&&isBest) h=tx(`<p class="okline"><b>All goals met at the lowest cost: ${usd(me.r.costDay)} a day, first word after ${sT(me.r.vis)}.</b></p>`,`<p class="okline"><b>全部达标，而且花费最低：每天 ${usd(me.r.costDay)}，第一个字 ${sT(me.r.vis)}后出现。</b></p>`);
  else if(me.ok) h=tx(`<p class="okline"><b>All goals met: ${usd(me.r.costDay)} a day.</b></p><p>A cheaper design meets them too: ${esc(chName(best))}, ${usd(best.r.costDay)} a day.</p>`,`<p class="okline"><b>全部达标：每天 ${usd(me.r.costDay)}。</b></p><p>还有更便宜的设计也全部达标：“${esc(chName(best))}”，每天 ${usd(best.r.costDay)}。</p>`);
  else {h=tx(`<p class="badline"><b>Not yet: ${esc(chMiss(me))}.</b></p>`,`<p class="badline"><b>还没达标：${esc(chMiss(me))}。</b></p>`)+`<p class="small">${chHint(me)}</p>`;
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal&&!(me.ok&&isBest)) return h;
  const init=chEval(CH0), rows=all.map(e=>{const isMe=same(e), isInit=e.stream===init.stream&&e.cache===init.cache&&e.share===init.share&&String(e.filt)===String(init.filt);
    return {label:chName(e),value:e.r.costDay,text:usd(e.r.costDay),ok:e.ok,me:isMe,best:best&&e===best,tags:(isInit?[{text:tx('initial setup','初始设置')}]:[]).concat(isMe?[{text:tx('your pick','你的选择'),me:true}]:[])};});
  const mx=Math.max(...rows.filter(x=>x.ok||x.me||x.tags.length).map(x=>x.value),CH.budget)*1.1;
  h+=A.ui.cmpChart({brief:true,title:tx(`GPU cost per day: every design that meets all goals, plus the starting setup and your choice (budget ${usd(CH.budget)})`,`每天的 GPU 花费：所有全部达标的设计，加上初始设置和你的选择（预算 ${usd(CH.budget)}）`),
    aria:tx('Cost per day: ','每天的花费：')+rows.filter(x=>x.ok||x.me).map(x=>`${x.label}${tx(': ','：')}${x.text}`).join(tx('; ','；')),rows,goal:CH.budget,goalLabel:usd(CH.budget),max:mx,
    okText:tx('meets every goal','全部达标'),noText:tx('misses at least one','至少一项没达标'),goalText:tx('budget','预算'),
    hiddenText:n=>tx(`The other ${n} designs miss at least one goal and are not shown.`,`其余 ${n} 种设计至少有一项没达标，图里没有列出。`)});
  if(best){const r=best.r, c0=chEval({...best,cache:0}), f8=chEval({...best,filt:8});
    h+=tx(`<p class="small">Why “${esc(chName(best))}” is the cheapest design that meets every goal:</p><ul class="small"><li>Streaming is required: no one waits for a whole answer within ${f(CH.vis,1)} s. The output must still be checked before it is sent.</li><li>Caching the shared system prompt more than halves the GPUs each pool needs; without it this design costs ${usd(c0.r.costDay)} a day.</li><li>Sending all ${pc(EASY)} easy messages to the small model removes most of the expensive large-model replicas.</li><li>Checking every 8 tokens would be cheaper, but the first word would come after ${sT(f8.r.vis)}, over the goal.</li></ul>`,
      `<p class="small">为什么“${esc(chName(best))}”是最便宜的达标设计：</p><ul class="small"><li>必须流式输出：没人能在 ${f(CH.vis,1)} 秒内等完整段回答。输出也必须查过再发。</li><li>缓存共用的 system prompt，每组卡要的数量少了一半多；不缓存时这个设计每天要 ${usd(c0.r.costDay)}。</li><li>把 ${pc(EASY)} 的简单消息全交给小模型，省掉了大部分昂贵的大模型副本。</li><li>每 8 个 token 查一次更便宜，可第一个字要 ${sT(f8.r.vis)}才出现，超过了目标。</li></ul>`);}
  h+=tx(`<p class="small muted">Note: the latencies and the easy share here are this lab’s assumptions. Before launch, measure each component’s latency on your own stack and the easy share with your own evals, then redo this budget.</p>`,`<p class="small muted">注意：这里的各项耗时和简单消息的比例都是这个 Lab 的假设。上线之前，在自己的系统上量出每个部件的耗时，用自己的评测量出简单消息的比例，再把这份预算算一遍。</p>`);
  return h;}

/* ---------- HTML ---------- */
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'product',outcome:esc(OUTCOME()),short:tx('Design a ChatGPT','设计 ChatGPT 类应用'),title:l?L(l.q):tx('How do you design a ChatGPT-like app end to end?','怎么从头设计一个 ChatGPT 类应用？'),
    minutes:25,fid:'Conceptual',fidPop:FIDPOP(),primer:tx('Primer: serving goals','前置知识：服务目标'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const spdCtl=()=>`<div class="pdz-speed"><span class="pdz-speed-l">${tx('Speed','速度')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="spd">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${spd===v}">${x}</button>`).join('')}</div></div>`;
function stageHTML(){
  return `<div class="panel stage pdz-stage" id="pdz-stage">`+
   `<div class="stage-head"><div class="pdz-st-title" id="pdz-st-title"></div><div class="stage-actions"><span class="status" id="pdz-status" tabindex="0"></span></div></div>`+
   `<div id="pdz-main" class="pdz-main">`+
    `<section class="pdz-blk" id="pdz-trace"><div class="pdz-ch-t" id="pdz-trace-t"></div><div id="pdz-wf"></div>`+
      `<label class="pdz-scrub"><span class="pdz-sr">${tx('Progress','进度')}</span><input type="range" id="pdz-pos" min="0" max="1000" step="1" value="1000" aria-label="${esc(tx('Progress of the message, in seconds after Send','这条消息的进度，按下发送后的秒数'))}"></label>`+
      `<div class="pdz-player" id="pdz-player"><button type="button" class="btn sm" data-act="play" id="pdz-play"></button><button type="button" class="btn ghost sm pdz-ib" data-act="back" aria-label="${esc(tx('Back to the previous component','退回上一个部件'))}">◂</button><button type="button" class="btn ghost sm pdz-ib" data-act="fwd" aria-label="${esc(tx('On to the next component','前进到下一个部件'))}">▸</button><button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Play the message again from Send.','从按下发送开始，把这条消息再播一遍。'))}">${tx('Replay','重放')}</button>${spdCtl()}</div>`+
      `<div class="pdz-leg legend-row" id="pdz-leg1"></div></section>`+
    `<section class="pdz-blk" id="pdz-prompt" hidden></section>`+
    `<section class="pdz-blk" id="pdz-fleet" hidden></section>`+
    `<section class="pdz-blk" id="pdz-sheet" hidden></section>`+
    `<div class="metrics pdz-metrics" id="pdz-metrics"></div>`+
   `</div>`+
   `<details class="math" id="pdz-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="pdz-mathbody"></div></details></div>`;}
const segOpts=(arr,fn)=>arr.map(v=>[String(v),fn(v)]);
function controlsHTML(){const di=Math.max(0,DAUS.indexOf(S.dau));
  return `<aside class="panel controls" id="pdz-controls" aria-label="${esc(A.t('ctrlAria')||'Settings')}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint">${tx('Watch the first-word time in the chart and the GPUs below it.','看图里第一个字出现的时间，和下面的卡数。')}</p><p class="ctl-lock" id="pdz-lock" hidden></p><div id="pdz-ctls">`+
   A.ui.range({id:'pdz-ctl-dau',key:'dau',label:tx('Daily users','每天的用户数'),min:0,max:DAUS.length-1,step:1,value:di,out:dauT(S.dau),note:tx('Users who send at least one message on a given day.','某一天至少发了一条消息的用户。')})+
   A.ui.seg({id:'pdz-ctl-msgs',key:'msgs',label:tx('Messages per user per day','每人每天的消息数'),options:segOpts(MSGS,String),value:String(S.msgs),note:tx('An assumption; measure it from your logs.','这是假设；从自己的日志里量。')})+
   A.ui.seg({id:'pdz-ctl-peak',key:'peak',label:tx('Peak factor','高峰倍数'),options:segOpts(PEAKS,v=>tx(`${v} times`,`${v} 倍`)),value:String(S.peak),note:tx('Busiest hour’s rate divided by the day’s average; an assumption. It is higher when most users live in one time zone.','最忙那个小时的速率除以全天平均；这是假设。用户集中在一个时区时会更高。')})+
   A.ui.seg({id:'pdz-ctl-stream',key:'stream',label:tx('Streaming','流式输出'),options:[['1',tx('On','开')],['0',tx('Off','关')]],value:String(S.stream)})+
   A.ui.seg({id:'pdz-ctl-cache',key:'cache',label:tx('Prompt caching','prompt 缓存'),options:[['0',tx('Off','关')],['1',tx('On','开')]],value:String(S.cache),note:tx('Caches the 2,048-token system prompt that every message starts with.','缓存每条消息开头都有的 2,048 个 token 的 system prompt。')})+
   A.ui.seg({id:'pdz-ctl-ret',key:'ret',label:tx('Retrieval','检索'),options:[['0',tx('Off','关')],['1',tx('On','开')]],value:String(S.ret),note:tx(`Adds ${LAT.ret} ms (embedding, vector search, rerank; assumed) and 3 passages of 512 tokens to each prompt.`,`多 ${LAT.ret} ms（embedding、向量检索、rerank；假设），每个 prompt 多 3 段、每段 512 个 token。`)})+
   A.ui.seg({id:'pdz-ctl-share',key:'share',label:tx('Messages routed to the small model','交给小模型的消息'),options:segOpts(SHARES,v=>pc(v)),value:String(S.share),note:tx(`Small model: ${MODELS.s.name} on one H100. Large: ${MODELS.l.name} on 4. The lab assumes ${pc(EASY)} of messages are easy enough for the small model.`,`小模型：${MODELS.s.name}，1 张 H100。大模型：${MODELS.l.name}，4 张。这个 Lab 假设 ${pc(EASY)} 的消息小模型能答好。`)})+
   A.ui.seg({id:'pdz-ctl-filt',key:'filt',label:FILT_L(),options:FILTS.map(v=>[String(v),filtShort(v)]),value:String(S.filt),note:tx(`“After all is sent” checks once when the answer has been streamed, so the user reads it before it is checked; “check all, then send” holds the answer until it is checked; “every N tokens” checks each piece of N tokens before sending it. Each check: ${LAT.chk} ms and 1 ms of one H100, assumed for a small classifier; an 8B classifier costs more.`,`“全部发完再查”在回答流式发完以后查一次，用户先看到、后检查；“整段先查再发”扣住整段回答，查完再发；“每 N 个 token”是回答每写出 N 个 token，先查这一段再发。每次检查耗时 ${LAT.chk} ms，占一张 H100 1 ms，这是按小分类器的假设；用 8B 的分类器会更贵。`)})+
   A.ui.range({id:'pdz-ctl-price',key:'price',label:tx('Price per GPU hour','每 GPU 小时的价格'),min:1,max:8,step:0.5,value:S.price,out:usd(S.price),note:tx('An H100 rented by the hour; this lab assumes $3 per GPU hour.','按小时租一张 H100 的价格；本 Lab 假设每 GPU 每小时 3 美元。')})+
   `</div><div id="pdz-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`async def stream_reply(conv, msg, send):
    """One chat turn, streamed as server-sent events."""
    if not await input_check(msg):        # e.g. Llama Guard
        return await send({"error": "blocked"})
    prompt = build_context(conv, msg)      # trim history
    model = route(msg)                     # small or large
    # smaller N: first word sooner, more checks
    chunk, N = [], 8
    async for tok in model.generate(prompt):
        chunk.append(tok)
        if len(chunk) == N:                # hold N tokens, then check
            if not await output_check(conv, chunk):
                return await send({"error": "stopped"})
            await send({"delta": "".join(chunk)})
            chunk = []
    if chunk:                              # the last, shorter chunk
        if not await output_check(conv, chunk):
            return await send({"error": "stopped"})
        await send({"delta": "".join(chunk)})
    await send({"done": True})             # like [DONE]
    log_turn(conv, msg)                    # off the hot path`;
function connectHTML(){
  const fl=(url,label,x)=>`<a href="${url}" target="_blank" rel="noopener"><code>${label}</code></a>${x}`;
  return `<section class="wrap connect" aria-labelledby="pdz-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="pdz-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('One streamed turn: check the input, build the context, route, then release the answer in checked chunks of N tokens.','一轮流式对话：先查输入，拼上下文，选模型，再把回答按 N 个 token 一段、查过一段放一段。')}</p><pre class="code" id="pdz-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="pdz-prod">`+
   tx(`<li>${fl(SRC.aev,'anthropic types/raw_message_stream_event.py',': the SDK defines six message stream event types: message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop.')}</li>`+
      `<li>${fl(SRC.ostr,'openai/_streaming.py',': the client stops reading when an event’s data starts with <code>[DONE]</code>; <code>stream_options.include_usage</code> adds a usage chunk before it.')}</li>`+
      `<li>${fl(SRC.vsrv,'vllm/entrypoints/openai/chat_completion/serving.py',': vLLM’s OpenAI-compatible chat server, the streaming path that open-weight deployments use.')}</li>`+
      `<li>${fl(SRC.apc,'vLLM automatic prefix caching',': reuses the KV cache of a shared prefix so a new query can “skip the computation of the shared part”.')}</li>`+
      `<li>${fl(SRC.llr,'LiteLLM routing',' and '+cite(SRC.llf,'fallbacks')+': load balancing across deployments (simple-shuffle by default), retries, then fallback to another model group.')}</li>`+
      `<li>${fl(SRC.lg,'Llama Guard 3 model card',': an 8B classifier for prompts and responses over 14 hazard categories (S1–S14); the 1B version covers 13.')}</li>`+
      `<li>${fl(SRC.pg,'Llama Prompt Guard 2 model card',`: 86M and 22M classifiers that run at the input check and flag prompt injection (instructions hidden in the input that try to make the model do something else; the lab ${LL('prompt-injection')} shows the attack) and jailbreaks (prompts that try to get around the model’s safety rules).`)}</li>`+
      `<li>${fl(SRC.rate,'OpenAI cookbook: rate limits',`: the gateway limits each user or API key to so many requests (RPM) and tokens (TPM) a minute and answers 429 at once over it, so one script cannot take the GPUs sized for real users; the client retries with exponential backoff. Who should wait when the peak is over capacity: the lab ${LL('admission')}.`)}</li>`,
      `<li>${fl(SRC.aev,'anthropic types/raw_message_stream_event.py','：SDK 定义了 6 种消息事件，即 message_start、content_block_start、content_block_delta、content_block_stop、message_delta 和 message_stop。')}</li>`+
      `<li>${fl(SRC.ostr,'openai/_streaming.py','：事件的 data 以 <code>[DONE]</code> 开头时，客户端停止读取；<code>stream_options.include_usage</code> 会在它之前多发一个用量块。')}</li>`+
      `<li>${fl(SRC.vsrv,'vllm/entrypoints/openai/chat_completion/serving.py','：vLLM 的 OpenAI 兼容聊天服务，自己部署开源模型时走的就是这条流式输出的路径。')}</li>`+
      `<li>${fl(SRC.apc,'vLLM automatic prefix caching','：复用共同前缀的 KV cache，新请求可以跳过共同部分的计算（原文 skip the computation of the shared part）。')}</li>`+
      `<li>${fl(SRC.llr,'LiteLLM routing',' 和 '+cite(SRC.llf,'fallbacks')+'：在多个部署之间做负载均衡（默认 simple-shuffle），先重试，再换到另一组模型。')}</li>`+
      `<li>${fl(SRC.lg,'Llama Guard 3 model card','：8B 的分类模型按 14 类风险（S1–S14）检查 prompt 和回答，1B 的版本是 13 类。')}</li>`+
      `<li>${fl(SRC.pg,'Llama Prompt Guard 2 model card',`：86M 和 22M 两个分类器，放在输入检查这一步，识别 prompt 注入（藏在输入里、想让模型改做别的事的指令，“${LL('prompt-injection')}”这个 Lab 讲这种攻击）和越狱（想绕开模型安全规则的提示）。`)}</li>`+
      `<li>${fl(SRC.rate,'OpenAI cookbook: rate limits',`：网关给每个用户或每个 API key 限定每分钟最多多少个请求（RPM）、多少个 token（TPM），超过了马上回 429，免得一个脚本占掉按真实用户配好的卡；客户端用指数退避重试。高峰超出容量时谁该排队，见“${LL('admission')}”这个 Lab。`)}</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2406.18665" target="_blank" rel="noopener">RouteLLM: Learning to Route LLMs with Preference Data</a><span>${tx('Ong et al., 2024 (arXiv 2406.18665)','Ong 等，2024（arXiv 2406.18665）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2312.06674" target="_blank" rel="noopener">Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations</a><span>${tx('Inan et al., 2023 (arXiv 2312.06674)','Inan 等，2023（arXiv 2312.06674）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2401.09670" target="_blank" rel="noopener">DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving</a><span>${tx('Zhong et al., OSDI 2024 (arXiv 2401.09670): the goodput definition','Zhong 等，OSDI 2024（arXiv 2401.09670）：goodput 的定义')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="pdz-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="pdz-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="pdz-app m2" id="pdz-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="pdz-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3><p>${tx('Pick a product and see when its users see the first word, how many GPUs it needs and what it costs per day.','选一个产品，看看用户多久看到第一个字、要多少张卡、每天花多少钱。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios: pick one','应用场景：选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(typeof p.d==='function'?p.d():L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="pdz-insight" aria-live="polite"></div><p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted pdz-sub">${tx('After you submit, you see which goals your design meets and whether a cheaper design meets them all.','提交后告诉你这个设计达到了哪些目标，以及有没有更便宜的设计全部达标。')}</p>`;
  const lam=CH.dau*CH.msgs/DAY*CH.peak;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Design the chat app for ${dauT(CH.dau)} daily users: first word within ${f(CH.vis,1)} s at peak, every answer checked, at most ${usd(CH.budget)} a day`,`为每天 ${dauT(CH.dau)} 用户设计聊天应用：高峰时 ${f(CH.vis,1)} 秒内出现第一个字，回答的每一段都先查过，每天最多 ${usd(CH.budget)}`)}</h3>`+
   tx(`<p>Each user sends ${CH.msgs} messages a day and the busiest hour has ${CH.peak} times the average rate: ${f(lam,0)} messages per second.</p><p>The models are ${MODELS.s.name} on one H100 and ${MODELS.l.name} on 4.</p><p>At most ${pc(EASY)} of the messages may go to the small model: only that many are easy, and more would send hard questions to it.</p><p>Every answer must pass the output check before the user sees it. Find the cheapest design that meets all of this.</p>`,`<p>每个用户每天发 ${CH.msgs} 条消息，最忙的那个小时是平均速率的 ${CH.peak} 倍：每秒 ${f(lam,0)} 条。</p><p>模型是 ${MODELS.s.name}（1 张 H100）和 ${MODELS.l.name}（4 张）。</p><p>交给小模型的消息不超过 ${pc(EASY)}：只有这么多消息是简单的，再多就有难题交给小模型。</p><p>回答的每一段在用户看到之前都要通过输出检查。找出满足这些要求、花费最低的设计。</p>`)+
   A.ui.seg({id:'pdz-ic-stream',key:'stream',label:tx('Streaming','流式输出'),options:[['1',tx('On','开')],['0',tx('Off','关')]],value:String(S.stream),inline:true})+
   A.ui.seg({id:'pdz-ic-cache',key:'cache',label:tx('Prompt caching','prompt 缓存'),options:[['0',tx('Off','关')],['1',tx('On','开')]],value:String(S.cache),inline:true})+
   A.ui.seg({id:'pdz-ic-share',key:'share',label:tx('To the small model','交给小模型'),options:segOpts(SHARES,v=>pc(v)),value:String(S.share),inline:true})+
   A.ui.seg({id:'pdz-ic-filt',key:'filt',label:FILT_L(),options:FILTS.map(v=>[String(v),filtShort(v)]),value:String(S.filt),inline:true,note:FILT_N()})+
   `<div class="pdz-chg" id="pdz-ch-g"></div><div class="pdz-chp"><div class="progress" id="pdz-ch-p" aria-hidden="true"><i></i></div><span class="small mono" id="pdz-ch-n"></span></div>`+
   `<p class="small muted">${tx(`Users, messages, peak, retrieval (off) and the GPU price (${usd(CH.price)} per GPU hour) are fixed in this challenge.`,`这道题里，用户数、消息数、高峰倍数、检索（关）和 GPU 价格（每 GPU 小时 ${usd(CH.price)}）都固定不变。`)}</p>`+submit+`<div id="pdz-ch-res" aria-live="polite"></div></div>`;}

/* ---------- stage drawing ---------- */
let W=640, GEO=null;
const sigOf=r=>r.seg.map(g=>g.k+':'+Math.round(g.d*1000)).join(',')+'|'+r.s.filt+'|'+r.s.stream;
const rowsOf=r=>ROWS.filter(k=>(k!=='ret'||r.s.ret)&&(k!=='route'||r.s.share>0));
const segOfK=(r,k)=>r.seg.find(g=>g.k===k);
/* the axis reaches past the first word and the budget line, and for a streamed answer checked in chunks past the second check, so the
   picture shows the hold, the check and the next chunk being checked (P12) */
function axisMax(r){const goal=goalNow(), c2=r.s.stream&&typeof r.s.filt==='number'?checkTimes(r)[1]||0:0, m=Math.max(r.vis,goal,c2)*1.12; const st=[0.5,1,2,2.5,5,10,20]; for(const s of st){if(m/s<=8) return Math.ceil(m/s)*s;} return Math.ceil(m/20)*20;}
/* the time axis. Part 2 draws both of its options on one axis, so the wait for a whole answer is drawn, and played, longer than the streamed
   one (P3). Part 6 draws its ways of checking on one axis too, so a larger N moves the first word to the right (N3); the whole answer
   checked first goes past that axis, and the picture says where its first word is */
const AX_SAFE=['after',16,8,4];
function axisOf(r){const id=mode==='guided'?sid():'';
  if(id==='stream') return Math.max(axisMax(compute({...r.s,stream:1})),axisMax(compute({...r.s,stream:0})));
  if(id==='safety') return Math.max(...AX_SAFE.map(v=>axisMax(compute({...r.s,filt:v}))));
  return axisMax(r);}
const goalNow=()=>mode==='challenge'?CH.vis:BRIEF.vis;
function tickStep(X){const n=Math.max(4,Math.min(8,Math.floor(W/80))); for(const s of [0.25,0.5,1,2,5,10]) if(X/s<=n) return s; return 20;}
/* when each output check starts (seconds after Send): a chunk is checked as soon as its last token is written; “whole” and “after” check
   once, when the whole answer is written */
function checkTimes(r){const F=r.s.filt; if(!r.chkOn) return [];
  if(typeof F!=='number') return [r.tEnd];
  const out=[]; for(let j=1;j<=Math.ceil(WL.outMean/F);j++) out.push(r.tFirst+(Math.min(j*F,WL.outMean)-1)*r.tpot); return out;}
const textW=s=>[...s].reduce((a,c)=>a+(c.charCodeAt(0)>0x2e00?11:6.6),0);
/* polish round 5 (2026-10-17, B4): on a wide stage the rows are drawn a little shorter, so part 8 fits a 1440 × 900 screen. The row height
   is a CSS variable on the chart (--pdz-rh, set in product-design.css), read when it is drawn; without it (phones, narrower stages) it is
   as before. The stage's width decides it, and the ResizeObserver on #pdz-main draws the chart again when that width changes. */
const cssNum=(el,v,d)=>{const x=el?parseFloat(getComputedStyle(el).getPropertyValue(v)):NaN; return x>0?x:d;};
function wfSVG(r,tau,X){
  const rows=rowsOf(r), narrow=W<480, LW=narrow?96:Math.min(170,Math.round(W*0.27)), DW=narrow?50:108, RH=narrow?24:cssNum(app&&$('#pdz-wf',app),'--pdz-rh',23), TOP=22, x0=LW+6, x1=W-DW-4,
    sx=t=>x0+(x1-x0)*Math.min(Math.max(t,0),X)/X, H=TOP+rows.length*RH+20, goal=goalNow(), over=r.vis>goal+1e-9, play=tau!=null, cur=play?tau:Infinity;
  const chk=LAT.chk/1000, ck=checkTimes(r), F=r.s.filt, hseg=segOfK(r,'hold'), gen=segOfK(r,'gen'), streamN=!!r.s.stream&&typeof F==='number'&&!!hseg;
  /* decode ticks: one per token while they are far enough apart, else one per chunk of N tokens, else none; the legend says which (N7, N9) */
  const px=(x1-x0)*r.tpot/X, N=typeof F==='number'?F:0, tick=px>=4?'tok':N&&N*px>=5?'chunk':'none';
  GEO={x0,x1,W,tick,N};
  /* a bar up to the message's time now. When the first word comes too late, the part of a bar on its path that lies past the budget line is
     red; the part before the line keeps its own look (P19) */
  const bar=(cls,a,b,y,hh,path)=>{const e=Math.min(b,cur,X); if(e<=a) return ''; let s=`<rect class="pdz-bar ${cls}" x="${sx(a)}" y="${y}" width="${Math.max(sx(e)-sx(a),1.5)}" height="${hh}" rx="1.5"/>`;
    if(path&&over&&e>goal+1e-9){const a2=Math.max(a,goal); s+=`<rect class="pdz-bar red" x="${sx(a2)}" y="${y}" width="${Math.max(sx(e)-sx(a2),1.5)}" height="${hh}"/>`;} return s;};
  let h=`<defs><pattern id="pdz-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect class="pdz-hb" width="5" height="5"/><line class="pdz-hp" x1="0" y1="0" x2="0" y2="5"/></pattern></defs>`;
  const gx=sx(goal), vx=sx(r.vis), yb=TOP+rows.length*RH, visIn=r.vis<=X+1e-9;
  h+=`<line class="pdz-goal" x1="${gx}" x2="${gx}" y1="${TOP-6}" y2="${yb}"/>`;
  h+=`<text class="pdz-gl" x="${Math.min(gx,x1)}" y="${TOP-9}" text-anchor="${gx>x1-60?'end':'middle'}">${esc(tx(`budget ${f(goal,1)} s`,`预算 ${f(goal,1)} 秒`))}</text>`;
  rows.forEach((k,i)=>{const y=TOP+i*RH, g=segOfK(r,k), inSeg=x=>!!x&&cur>=x.t0&&cur<x.t0+x.d, on=play&&(inSeg(g)||(k==='model'&&inSeg(gen))), nm=L(COMP[k].n);
    const tip=k==='hold'?holdTip(r):k==='model'?modelTip(r):`<p>${COMP[k].d()}</p>`;
    h+=`<g class="pdz-row${on?' on':''}" data-k="${k}" data-tipx="${esc(`<p><b>${esc(nm)}</b></p>`+tip)}" tabindex="0">`;
    h+=`<rect class="pdz-hit" x="0" y="${y}" width="${W}" height="${RH}"/>`;
    h+=`<text class="pdz-rl" x="${LW}" y="${y+RH/2+4}" text-anchor="end">${esc(narrow&&COMP[k].sn?L(COMP[k].sn):nm)}</text>`;
    const note=(t,cls,x)=>`<text class="pdz-note${cls?' '+cls:''}" x="${x==null?x0:x}" y="${y+RH/2+4}">${esc(t)}</text>`;
    let end=g?g.t0+g.d:0, redRow=false;
    if(k==='log') h+=note(narrow?tx('after the answer','写完后才记录'):tx('logged after the answer; not part of the wait','回答写完后才记录，不占等待时间'));
    else if(k==='hold'&&!r.chkOn) h+=note(tx('no output check','不查输出'),'bad');
    else if(k==='hold'&&F==='after'&&r.s.stream){if(!play||cur>=r.tFirst) h+=note(narrow?tx('checked after it was sent','发完才查'):tx(`checked once, after all of it was sent (at ${f(r.tEnd,1)} s)`,`整段发完以后（${f(r.tEnd,1)} 秒）才查一次`),'bad',x0+6);}
    else if(k==='hold'){
      /* the first chunk: hatched while its N tokens are being written, then its check (dark); every later check appears as the time reaches it (P12) */
      if(streamN){h+=bar('hold',hseg.t0,ck[0],y+5,RH-10,true);
        const lab=tx(`${F} tokens, then a check`,`攒够 ${F} 个 token 再查`), lx=sx(hseg.t0)-6;
        if(!narrow&&(!play||cur>=hseg.t0)&&lx-x0>=textW(lab)+4) h+=`<text class="pdz-note" x="${lx}" y="${y+RH/2+4}" text-anchor="end">${esc(lab)}</text>`;}
      ck.forEach(tc=>{if(tc<X) h+=bar('chk',tc,tc+chk,y+4,RH-8,!!hseg&&tc>=hseg.t0-1e-9&&tc<hseg.t0+hseg.d);});
      redRow=over&&!!hseg&&end>goal+1e-9;}
    else if(k==='model'&&g){h+=bar('mdl',g.t0,r.tFirst,y+5,RH-10,true);
      /* decode: one short tick per token while the ticks are far enough apart, a long tick where a chunk of N tokens is full */
      const dEnd=Math.min(r.tEnd,X); h+=bar('dec',r.tFirst,dEnd,y+8,RH-16,!!gen);
      const lim=Math.min(dEnd,cur);
      if(cur>=r.tFirst&&tick!=='none') for(let i=1;i<=WL.outMean;i++){const t=r.tFirst+(i-1)*r.tpot; if(t>lim+1e-9) break; const full=!!N&&(i%N===0||i===WL.outMean);
        if(tick==='tok'||full) h+=`<line class="pdz-tok${full?' full':''}" x1="${sx(t)}" x2="${sx(t)}" y1="${y+(full?5:8)}" y2="${y+RH-(full?5:8)}"/>`;}
      if(gen) end=gen.t0+gen.d; redRow=over&&end>goal+1e-9;
      const nt=narrow?tx(`done at ${f(r.tEnd,1)} s`,`第 ${f(r.tEnd,1)} 秒写完`):tx(`the answer is done at ${f(r.tEnd,1)} s`,`回答在第 ${f(r.tEnd,1)} 秒写完`), clash=(visIn?[gx,vx]:[gx]).some(x=>x>x1-10-textW(nt)&&x<x1+2);
      if(r.tEnd>X&&!clash&&(!play||cur>=X)) h+=`<text class="pdz-note" x="${x1-2}" y="${y+4}" text-anchor="end">${esc(nt)}</text>`;}
    else if(g){h+=bar('cmp',g.t0,end,y+5,RH-10,true); redRow=over&&end>goal+1e-9;
      /* the first word past the end of the axis (part 6, the whole answer checked first): the row that ends with it says when (N3) */
      if(k==='back'&&!visIn&&(!play||cur>=X-1e-9)){const ls=[tx(`first word at ${sT(r.vis)} →`,`首字在 ${sT(r.vis)} →`),tx(`first word ${sT(r.vis)} →`,`首字 ${sT(r.vis)} →`),`${sT(r.vis)} →`];
        h+=note(ls.find(t=>textW(t)<=x1-x0-8)||ls[2],'bad',x1-2).replace('<text ','<text text-anchor="end" ');}}
    /* the right column: the row's time once the row is done; the model's says what it counts (P18) */
    if(g&&(!play||cur>=end)){const d=k!=='model'?dur(g.d):gen?(narrow?dur(end-g.t0):tx(`whole ${dur(end-g.t0)}`,`整段 ${dur(end-g.t0)}`)):(narrow?dur(g.d):tx(`TTFT ${dur(g.d)}`,`首 token ${dur(g.d)}`));
      h+=`<text class="pdz-dur${redRow?' bad':''}" x="${W-4}" y="${y+RH/2+4}" text-anchor="end">${esc(d)}</text>`;}
    h+=`</g>`;});
  h+=`<line class="pdz-ax" x1="${x0}" x2="${x1}" y1="${yb+2}" y2="${yb+2}"/>`;
  const ts=tickStep(X); for(let t=0;t<=X+1e-9;t+=ts) h+=`<text class="pdz-tk" x="${sx(t)}" y="${yb+15}" text-anchor="${t===0?'start':'middle'}">${t===0?'0':f(t,ts<1?2:0)}</text>`;
  h+=`<text class="pdz-tk" x="${LW}" y="${yb+15}" text-anchor="end">${esc(narrow?tx('s after Send','按下发送后的秒数'):tx('seconds after Send','按下发送后的秒数'))}</text>`;
  if(visIn&&(!play||cur>=r.vis)) h+=`<line class="pdz-vis" x1="${vx}" x2="${vx}" y1="${TOP-6}" y2="${yb+4}"/>`;
  if(play&&cur<X) h+=`<line class="pdz-cur" x1="${sx(cur)}" x2="${sx(cur)}" y1="${TOP-6}" y2="${yb+4}"/>`;
  return `<svg class="pdz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tx(`The path of one message: the first word appears after ${sT(r.vis)}.`,`一条消息的路径：${sT(r.vis)}后出现第一个字。`))}">${h}</svg>`;}
const lg=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="pdz-sw ${cls}"></i><span>${txt}</span></span>`;
function legendHTML(){return lg('pdz-sw-cmp',tx('component at work','部件在工作'),tx('A bar is the time the component adds before the first word.','横条是这个部件在第一个字之前占的时间。'))+
  lg('pdz-sw-mdl',tx('model: queue + prefill','模型：排队 + prefill'),tx('The time the request queues at the model and runs prefill, until the model writes its first token. The GPUs are sized so that 90% of requests get their first token within 1 s.','这一段是请求在模型那里排队、做 prefill，一直到模型写出第一个 token 的时间。卡是按“90% 的请求首 token 在 1 秒内”来配的。'))+
  (()=>{const t=GEO&&GEO.tick, n=GEO&&GEO.N;
    return lg('pdz-sw-dec',t==='tok'?tx('decode: one tick per token, the first at the end of prefill','decode：一道细线一个 token，第 1 个在 prefill 末端'):t==='chunk'?tx(`decode: one tick per chunk of ${n} tokens`,`decode：一道线是一段（${n} 个 token）`):tx('decode: the rest of the answer','decode：写回答的其余部分'),
      t==='chunk'?tx(`The rest of the answer, one token per step. At this scale the ticks would be too close, so only the end of each chunk of ${n} tokens is drawn.`,`回答的其余部分，每一步写一个 token。这个比例下每个 token 一道线太密，只画每段 ${n} 个 token 的末尾。`)
        :tx('The rest of the answer, one token per step. The first token is the model’s first token, at the end of prefill; a long tick marks a full chunk of N tokens.','回答的其余部分，每一步写一个 token。第 1 个 token 就是模型的首 token，在 prefill 末端；长竖线表示攒满了一段 N 个 token。'));})()+
  lg('pdz-sw-hold',tx('held: the first chunk is being written','扣住：第一段还没写完'),tx('The output check can read the first chunk only when all N of its tokens are written; the user waits for that too.','输出检查要等第一段的 N 个 token 都写完才能查，这段时间用户也在等。'))+
  lg('pdz-sw-chk',tx(`one check (${LAT.chk} ms)`,`一次检查（${LAT.chk} ms）`),tx(`The classifier checks one chunk in ${LAT.chk} ms (an assumption). Only a checked chunk is sent to the user.`,`分类器查一段要 ${LAT.chk} ms（假设）。查过的一段才发给用户。`))+
  lg('pdz-sw-goal',tx('budget for the first word','首字预算'),tx('The product’s goal for the time to the first visible word.','产品对首字可见时间的要求。'))+
  lg('pdz-sw-vis',tx('first word on screen','第一个字出现'),D_VIS())+
  lg('pdz-sw-bad',tx('past the budget','超出预算的部分'),tx('The time before the first word that lies past the budget line.','第一个字出现之前、越过预算线以后的那部分时间。'));}
/* Scenarios and Settings fold the prompt bar and the sizing sheet, so the trace and the fleet fit on the stage (P15); a guided part shows them open */
const moreOpen={prompt:false,sheet:false};
const fold=(k,title,body)=>mode==='guided'?`<div class="pdz-ch-t">${title}</div>${body}`:`<details class="pdz-more" data-more="${k}"${moreOpen[k]?' open':''}><summary class="pdz-ch-t">${title}</summary>${body}</details>`;
function promptHTML(r){const own=(WL.own[0]+WL.own[1])/2, ret=r.s.ret?WL.ret:0, tot=WL.sys+own+ret, q=quiet('l',r.s.cache,r.s.ret), q8=quiet('s',r.s.cache,r.s.ret), w=v=>(100*v/tot).toFixed(2)+'%';
  return fold('prompt',tx('One average prompt, in tokens: what prefill must compute','一个平均长度的 prompt（token 数）：prefill 要算哪些'),
   `<div class="pdz-pbar"><span class="pdz-ps sys${r.s.cache?' hit':''}" style="width:${w(WL.sys)}" data-tipx="${esc(tx(`System prompt: 2,048 tokens, the same for every message.${r.s.cache?' Read from the cache, not computed.':''}`,`system prompt：2,048 个 token，每条消息都一样。${r.s.cache?'从缓存读，不用算。':''}`))}" tabindex="0">${tx('system','system')} ${I(WL.sys)}</span>`+
   `<span class="pdz-ps own" style="width:${w(own)}" data-tipx="${esc(tx('Recent history and the new message: 2,000 tokens on average (500 to 3,500).','最近的历史和新消息：平均 2,000 个 token（500 到 3,500）。'))}" tabindex="0">${tx('history + message','历史 + 新消息')} ${I(own)}</span>`+
   (ret?`<span class="pdz-ps ret" style="width:${w(ret)}" data-tipx="${esc(tx('Three retrieved passages of 512 tokens. Prefill computes them too.','检索到的 3 段，每段 512 个 token。prefill 也要算它们。'))}" tabindex="0">${tx('retrieved','检索')} ${I(ret)}</span>`:'')+`</div>`+
   `<div class="pdz-pl"><span><i class="pdz-sw pdz-sw-cmp"></i>${tx('computed in prefill','prefill 要算')}</span><span><i class="pdz-sw pdz-sw-hit"></i>${tx('read from the cache','从缓存读')}</span></div>`+
   `<p class="small">${tx(r.s.share>0?`Prefill computes ${I(q.prefill)} of ${I(q.prompt)} tokens. On a quiet server it takes ${msT(q8.ttft)} on the small 8B model and ${msT(q.ttft)} on the large 70B model.`:`Prefill computes ${I(q.prefill)} of ${I(q.prompt)} tokens. On a quiet server the 70B model takes ${msT(q.ttft)}.`,r.s.share>0?`prefill 要算 ${I(q.prompt)} 个 token 里的 ${I(q.prefill)} 个。在空闲的服务上，小模型 8B 用 ${msT(q8.ttft)}，大模型 70B 用 ${msT(q.ttft)}。`:`prefill 要算 ${I(q.prompt)} 个 token 里的 ${I(q.prefill)} 个。在空闲的服务上，70B 模型用 ${msT(q.ttft)}。`)}</p>`);}
/* the fleet, at one scale everywhere (P1): a square is 4 GPUs, one replica of the 70B model, and a remainder is a narrower square, so the
   squares' area is in proportion to the GPUs. Past SQCAP squares in all, the rest is counted in words, not drawn: the drawn area never
   shrinks as the GPUs grow. Hard messages the small model answers badly (more than the easy share) are its GPUs drawn red (P10) */
const SQ=4, SQCAP=1000;
const hardOf=r=>r.s.share>EASY+1e-9&&r.gpus.s?Math.round(r.gpus.s*(r.s.share-EASY)/r.s.share):0;
function fleetCells(r){const hard=hardOf(r), parts=[['l',r.gpus.l],['s',r.gpus.s-hard],['s hard',hard],['f',r.fG]]; let left=SQCAP; const out={};
  parts.forEach(([cls,n])=>{const u=Math.max(0,Math.min(n/SQ,left)); left-=u; const full=Math.floor(u+1e-9), rem=u-full; let h='';
    for(let i=0;i<full;i++) h+=`<i class="pdz-sq ${cls}"></i>`;
    if(rem>1e-9) h+=`<i class="pdz-sq ${cls} part" style="width:${(11*rem).toFixed(2)}px"></i>`;
    out[cls]={h,u,cut:Math.round(n-u*SQ)};});
  return out;}
function fleetHTML(r){const tot=r.total, c=fleetCells(r), hard=hardOf(r);
  const cut=n=>n>0?`<div class="pdz-fh small muted">${tx(`${I(n)} more GPUs not drawn`,`还有 ${I(n)} 张卡没画出来`)}</div>`:'';
  const grp=(cls,name,n,tip,inner,more)=>n>0?`<div class="pdz-fg" data-tipx="${esc(tip)}" tabindex="0"><div class="pdz-fl"><span>${name}</span><b>${tx(`${I(n)} GPUs`,`${I(n)} 张卡`)}</b></div><span class="pdz-sqs">${inner}</span>${more||''}</div>`:'';
  return `<div class="pdz-ch-t">${tx(`GPUs at peak for ${dauT(r.s.dau)} daily users: ${I(tot)} GPUs, ${usd(r.costDay)} a day`,`每天 ${dauT(r.s.dau)} 用户、高峰时要的卡：${I(tot)} 张，每天 ${usd(r.costDay)}`)}</div>`+
   `<div class="pdz-fkey"><span><i class="pdz-sqk"></i>${tx('= 4 GPUs (one 70B replica); a narrower square is fewer','= 4 张卡（一份 70B 副本）；窄的方块不到 4 张')}</span>${hard?`<span><i class="pdz-sqk hard"></i>${tx('the small model on hard messages','小模型在答难题')}</span>`:''}</div>`+
   grp('l',tx(`Large model, ${MODELS.l.name}, 4 per replica`,`大模型 ${MODELS.l.name}，每份 4 张`),r.gpus.l,tx(`${I(r.rep.l)} replicas for ${rT(r.rate.l)} messages per second; each takes at most ${f(r.P.l.g,2)} per second (its goodput).`,`${I(r.rep.l)} 份副本，一共接每秒 ${rT(r.rate.l)} 条消息；每份最多接每秒 ${f(r.P.l.g,2)} 条（它的达标吞吐）。`),c.l.h,cut(c.l.cut))+
   grp('s',tx(`Small model, ${MODELS.s.name}, 1 per replica`,`小模型 ${MODELS.s.name}，每份 1 张`),r.gpus.s,tx(`${I(r.rep.s)} replicas for ${rT(r.rate.s)} messages per second; each takes at most ${f(r.P.s.g,2)} per second (its goodput).`,`${I(r.rep.s)} 份副本，一共接每秒 ${rT(r.rate.s)} 条消息；每份最多接每秒 ${f(r.P.s.g,2)} 条（它的达标吞吐）。`),
     c.s.h+(c['s hard'].h?`<span class="pdz-sqh">${c['s hard'].h}</span>`:''),cut(c.s.cut+c['s hard'].cut)+(hard?`<div class="pdz-fh small pdz-bad">${tx(`Red: about ${I(hard)} GPUs answer hard messages, the extra ${pc(r.s.share-EASY)} of the messages sent to the small model`,`红色：约 ${I(hard)} 张卡在答难题，也就是多交给小模型的那 ${pc(r.s.share-EASY)} 的消息`)}</div>`:''))+
   grp('f',tx('Output checks','输出检查'),r.fG,tx(`${cpsT(r.cps)} checks per answer on average, each taking 1 ms of one GPU (this lab’s assumption).`,`每个回答平均查 ${cpsT(r.cps)} 次，每次占一张卡 1 ms（这个 Lab 的假设）。`),c.f.h,cut(c.f.cut));}
function sheetHTML(r){const s=r.s, row=(a,b,c)=>`<tr><th scope="row">${a}</th><td>${b}</td><td class="mono">${c}</td></tr>`;
  return fold('sheet',tx('Sizing sheet: from users to dollars','估算表：从用户数到美元'),`<div class="pdz-tw"><table class="pdz-tab"><tbody>`+
   row(tx('Messages per day','每天的消息'),tx(`${dauT(s.dau)} users × ${s.msgs}`,`${dauT(s.dau)} 用户 × ${s.msgs}`),bigN(r.msgsDay))+
   row(tx('At peak, per second','高峰每秒'),tx(`over 86,400 s, × ${s.peak} at the peak`,`分摊到 86,400 秒，高峰再 × ${s.peak}`),rT(r.lam))+
   (r.rate.s>0?row(tx('Small-model replicas','小模型副本'),tx(`${rT(r.rate.s)} per s, ${f(r.P.s.g,2)} per replica, rounded up`,`每秒 ${rT(r.rate.s)} 条，每份接 ${f(r.P.s.g,2)} 条，向上取整`),tx(`${I(r.rep.s)} → ${I(r.gpus.s)} GPUs`,`${I(r.rep.s)} 份 → ${I(r.gpus.s)} 张卡`)):'')+
   (r.rate.l>0?row(tx('Large-model replicas','大模型副本'),tx(`${rT(r.rate.l)} per s, ${f(r.P.l.g,2)} per replica, rounded up`,`每秒 ${rT(r.rate.l)} 条，每份接 ${f(r.P.l.g,2)} 条，向上取整`),tx(`${I(r.rep.l)} → ${I(r.gpus.l)} GPUs`,`${I(r.rep.l)} 份 → ${I(r.gpus.l)} 张卡`)):'')+
   (r.fG?row(tx('Output checks','输出检查'),tx(`${rT(r.lam)} messages per s × ${cpsT(r.cps)} checks each × ${DEF('1 ms of one GPU per check',D_CHKGPU())} = ${r.fRaw.toFixed(1)} GPUs busy all the time on average, rounded up`,`每秒 ${rT(r.lam)} 条 × 每条平均查 ${cpsT(r.cps)} 次 × ${DEF('每次占卡 1 ms',D_CHKGPU())} = 平均 ${r.fRaw.toFixed(1)} 张卡一直在查，向上取整`),tx(`${I(r.fG)} GPUs`,`${I(r.fG)} 张卡`)):'')+
   row(tx('Cost per day','每天花费'),tx(`${I(r.total)} GPUs × 24 h × ${usd(s.price)}`,`${I(r.total)} 张 × 24 小时 × ${usd(s.price)}`),usd(r.costDay))+
   row(tx('Per user per month','每个用户每月'),tx('× 30 days, shared by the daily users','× 30 天，再由每天的用户分摊'),usd(r.perUserMonth))+
   row(tx('Conversation storage','对话存储'),tx('400 tokens × 4 bytes per message','每条消息 400 个 token × 4 字节'),tx(`${f(r.storeGB,1)} GB a day`,`每天 ${f(r.storeGB,1)} GB`))+
   `</tbody></table></div><p class="small muted">${tx(`Each message stores the user’s message (about 100 tokens) and the answer (300 on average), about 4 bytes per token; GB here is 10⁹ bytes. ${f(r.storeGB,1)} GB a day costs far less than the GPUs, so the cost above counts the GPUs only.`,`每条消息存用户的消息（约 100 个 token）和回答（平均 300 个），每个 token 约 4 字节；这里的 GB 是 10⁹ 字节。每天 ${f(r.storeGB,1)} GB 的存储比卡便宜得多，所以上面的花费只算 GPU。`)}</p>`);}
const MET=[
 {k:'vis',label:()=>tx('First word on screen','首字可见时间'),v:r=>sT(r.vis),dp:2,sub:()=>tx(`budget ${f(goalNow(),1)} s`,`预算 ${f(goalNow(),1)} 秒`),tip:()=>D_VIS(),bad:r=>r.vis>goalNow()+1e-9},
 /* part 6 says how many GPUs the checks take: the card under the picture shows it (P12) */
 {k:'gpu',label:()=>tx('GPUs at peak','高峰时的卡数'),v:r=>I(r.total),dp:0,sub:r=>mode==='guided'&&sid()==='safety'?tx(`${I(r.fG)} of them for checks`,`其中输出检查 ${I(r.fG)} 张`):tx(`${rT(r.lam)} messages per second`,`每秒 ${rT(r.lam)} 条消息`),tip:()=>tx('H100s for both model pools and the output checks, sized for the busiest hour.','两组模型的卡加上输出检查的卡，按最忙那个小时配。')},
 {k:'cost',label:()=>tx('GPU cost per day','每天的 GPU 花费'),v:r=>usd(r.costDay),dp:0,sub:r=>tx(`at ${usd(r.s.price)} per GPU hour`,`每 GPU 小时 ${usd(r.s.price)}`),tip:()=>D_GPUH(),bad:r=>mode==='challenge'&&r.costDay>CH.budget+1e-9},
 {k:'user',label:()=>tx('Per user per month','每个用户每月'),v:r=>usd(r.perUserMonth),dp:3,sub:()=>tx('GPU cost only','只算 GPU'),tip:()=>tx('Daily GPU cost times 30, divided by daily users.','每天的 GPU 花费乘以 30，再除以每天的用户数。')}];
const numOf=s=>parseFloat(String(s).replace(/[$,s ]/g,''));
let dBase=null;   // a scenario switch says which scenario the change is from (P30)
/* the ▲▼ lines say what they are compared with once, above the cards (“▲▼ 和“刚上线的聊天应用”比”), and each card only the change
   (“▲ 38 倍”, “▼ −62%”), so a narrow card keeps it on one line (N13; kit polish K2, A.fmt.delta with base:false). The line above the
   cards sits in the gap above them and takes no room (product-design.css), so the cards do not move when it comes and goes with the ▲▼ */
const vsTxt=()=>dBase?tx(`▲▼ compared with ${dBase.en}`,`▲▼ 和${dBase.zh}比`):tx('▲▼ compared with the last setting','▲▼ 和上一次的设置比');
function drawMetrics(r){const host=$('#pdz-metrics',app);
  if(!host.children.length) host.innerHTML=`<div class="pdz-mvs"></div>`+MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  const vs=host.querySelector('.pdz-mvs'), now=performance.now(); let any=false;
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), v=m.v(r), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(r); el.dataset.tipx=m.tip();
    el.classList.toggle('pdz-mbad',!!(m.bad&&m.bad(r)));
    const a=numOf(shownPrev[m.k]), b=numOf(v), dt=userAct&&isFinite(a)&&isFinite(b)?(A.fmt.delta(a,b,{dp:m.dp,base:false})||''):'';
    if(dt){any=true; d.textContent=dt.replace(/ 倍$/,'\u00a0倍'); d.classList.add('on'); d._at=now; clearTimeout(d._t); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2600);}
    else if(!userAct||now-(d._at||0)>50){d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=v;});
  if(any){vs.textContent=vsTxt(); vs.classList.add('on'); vs._at=now; clearTimeout(vs._t); vs._t=setTimeout(()=>{vs.classList.remove('on'); vs.textContent='';},2600);}
  else if(!userAct||now-(vs._at||0)>50){clearTimeout(vs._t); vs.classList.remove('on'); vs.textContent='';}}
const MB=(lead,rows,res)=>`<div class="pdz-mb"><p>${lead}</p><div class="pdz-eqs">${rows.map(([l,r])=>`<span class="pdz-eq-l">${l?M(l):''}</span><span class="pdz-eq-r">${M(r)}</span>`).join('')}</div>${res?`<p class="pdz-mb-res">${res}</p>`:''}</div>`;
const tt=s=>`\\text{${s}}`;
function mathHTML(r){const s=r.s, ms=v=>I(v*1000), parts=r.seg.filter(g=>!['model','gen','hold'].includes(g.k));
  const pre=parts.reduce((a,g)=>a+g.d,0), G=r.gen>0;
  const terms=[tt(tx('way in','去程')),tt('TTFT')].concat(G?[tt(tx('rest of the answer','写完整段'))]:[]).concat(r.hold>0?[tt(tx('hold','扣住'))]:[]).concat([tt(tx('way back','回程'))]);
  const vals=[ms(pre-LAT.net/1000),ms(r.tModel)].concat(G?[ms(r.gen)]:[]).concat(r.hold>0?[ms(r.hold)]:[]).concat([ms(LAT.net/1000)]);
  let h=MB(tx('Time to the first visible word: the components on the way in, the model’s first token, the time the output check holds the first chunk, and the network on the way back.','首字可见时间：去程各部件的时间，加上模型的首 token，加上输出检查扣住第一段的时间，再加上回程的网络。'),
    [[`${tt(tx('first word','首字'))}`,`= ${terms.join(' + ')}`],['',`= (${vals.map(v=>String(v).replace(/,/g,'{,}')).join(' + ')})\\,${tt('ms')} = ${f(r.vis,2)}\\,${tt('s')}`]],
    tx(`Way in: ${parts.filter(g=>g.k!=='back').map(g=>`${L(COMP[g.k].n)} ${ms(g.d)} ms`).join(', ')}. TTFT is the 90th percentile time to first token the engine measures at goodput, ${ms(r.tModel)} ms, within the ${f(TTFT_GOAL,0)} s goal each pool is sized for.`,
      `去程：${parts.filter(g=>g.k!=='back').map(g=>`${L(COMP[g.k].n)} ${ms(g.d)} ms`).join('，')}。TTFT 是引擎在达标吞吐下测到的首 token 延迟的第 90 百分位，${ms(r.tModel)} ms，在每组卡配卡时用的 ${f(TTFT_GOAL,0)} 秒目标以内。`));
  if(r.hold>0&&s.stream&&typeof s.filt==='number') h+=MB(tx('The hold: the first chunk of N tokens is written, then checked.','扣住的时间：第一段 N 个 token 先写完，再查一次。'),
    [[tt(tx('hold','扣住')),`= (N - 1) \\times ${tt('TPOT')} + ${tt(tx('check','检查'))}`],['',`= (${s.filt} - 1) \\times ${f(r.tpot*1000,2)}\\,${tt('ms')} + ${LAT.chk}\\,${tt('ms')} = ${f(r.hold*1000,0)}\\,${tt('ms')}`]],
    tx(`The chunk’s first token is the model’s first token, already counted in TTFT, so the hold waits for only ${s.filt-1} more. TPOT is the 90th percentile of the time per output token the engine measures at goodput, ${f(r.tpot*1000,2)} ms; the check takes ${LAT.chk} ms (an assumption).`,`第一段的第 1 个 token 就是模型的首 token，已经算在 TTFT 里，所以只要再等 ${s.filt-1} 个。TPOT 是引擎在达标吞吐下测到的每个输出 token 耗时的第 90 百分位，${f(r.tpot*1000,2)} ms；每次检查 ${LAT.chk} ms 是假设。`));
  h+=MB(tx('Messages per second at the busiest hour.','最忙那个小时每秒的消息数。'),
    [[`\\lambda_{${tt(tx('peak','高峰'))}}`,`= \\frac{${tt(tx('users','用户数'))} \\times ${tt(tx('messages','消息数'))}}{86{,}400} \\times ${tt(tx('peak factor','高峰倍数'))}`],['',`= \\frac{${I(s.dau).replace(/,/g,'{,}')} \\times ${s.msgs}}{86{,}400} \\times ${s.peak} = ${f(r.lam,2)}`]]);
  const pl=['s','l'].filter(m=>r.rate[m]>0).map(m=>[`${tt(tx(m==='s'?'small model':'large model',m==='s'?'小模型':'大模型'))}`,`= \\operatorname{ceil}(${f(r.lam,2)} \\times ${f(m==='s'?s.share:1-s.share,1)} \\div ${f(r.P[m].g,2)}) \\times ${MODELS[m].n} = ${I(r.gpus[m])}`]);
  h+=MB(tx('GPUs per pool: enough replicas that none takes more than its goodput, times the GPUs per replica.','每组的卡数：配够副本，让每一份接的消息都不超过它的达标吞吐，再乘以每份的卡数。'),
    [[tt(tx('GPUs','卡数')),`= \\operatorname{ceil}(\\lambda \\times ${tt(tx('share','份额'))} \\div g) \\times ${tt(tx('GPUs per replica','每份卡数'))}`]].concat(pl),
    tx('g is the goodput of one replica, from the engine; the share is the part of the messages the pool gets.','g 是一份副本的达标吞吐，由引擎算出；份额是这一组接的消息占的比例。'));
  if(r.fG) h+=MB(tx('GPUs for the output checks: each check takes 1 ms of one GPU (an assumption).','输出检查的卡数：每次检查占一张卡 1 ms（这是假设）。'),
    [[tt(tx('checks','检查')),`= \\operatorname{ceil}(\\lambda \\times ${tt(tx('checks per answer','每个回答的检查次数'))} \\times 0.001)`],['',`= \\operatorname{ceil}(${f(r.lam,2)} \\times ${cpsT(r.cps)} \\times 0.001) = ${I(r.fG)}`]]);
  h+=MB(tx('Cost per day: every GPU, all day, at the hourly price.','每天的花费：所有卡全天开着，按小时价算。'),
    [[`${tt(tx('cost','花费'))}`,`= (${['s','l'].filter(m=>r.rate[m]>0).map(m=>I(r.gpus[m])).concat(r.fG?[I(r.fG)]:[]).join(' + ')}) \\times 24 \\times ${f(s.price,2)} = ${I(r.costDay)}`]],tx('In US dollars per day.','单位是每天的美元。'));
  return h;}

/* ---------- render ---------- */
function panels(){const g=mode==='guided'?sid():'';
  if(mode!=='guided') return {trace:true,prompt:mode==='sandbox',fleet:true,sheet:true};
  return {trace:['mission','stream','safety','remember'].includes(g),prompt:g==='context',fleet:['context','predict','route','size','remember'].includes(g),sheet:g==='size'};}
function curRow(r,t){const g=r.seg.find(x=>t>=x.t0-1e-9&&t<x.t0+x.d-1e-9); return g?g.k:null;}
/* the stage title says the moment on the picture (P6): playing, paused at a time, or the result. While playing it names the component the
   message is at, which changes at most once per DWELL (P5) */
function stageTitle(r){const g=mode==='guided'?sid():'';
  if(!PL.done){const k=curRow(r,PL.t), at=PL.t<1?`${f(PL.t*1000,0)} ms`:sT(PL.t), nm=k?L(W<480&&COMP[k].sn?COMP[k].sn:COMP[k].n):'';
    if(!PL.on) return k?tx(`Paused ${at} after Send, at “${nm}”`,`停在按下发送后 ${at}：消息在“${nm}”`):tx(`Paused ${at} after Send: the first word is on screen`,`停在按下发送后 ${at}：第一个字已出现`);
    if(k) return tx(`Playing: the message is at “${nm}”`,`播放中：消息在“${nm}”`);
    return !r.s.stream||r.s.filt==='whole'?tx(`Playing: the whole answer appears at once, ${sT(r.vis)} after Send`,`播放中：整段回答在按下发送后 ${sT(r.vis)}一起出现`)
      :tx(`Playing: first word at ${sT(r.vis)}; the rest follows chunk by chunk`,`播放中：第一个字在 ${sT(r.vis)}出现，其余一段一段跟上`);}
  if(g==='context') return tx('What every message sends, and the GPUs it takes','每条消息发出去的内容，和它要的卡');
  if(g==='predict'||g==='route') return tx('The GPUs for this traffic, by pool','这样的流量要的卡，按组分');
  if(g==='size') return tx(`Sizing for ${dauT(S.dau)} daily users`,`为每天 ${dauT(S.dau)} 用户估算`);
  return tx('One message, from Send to the first word on screen','一条消息，从按下发送到第一个字出现');}
function traceTitle(r){return tx(`Each row is one component; its bar is when it works. First word after ${sT(r.vis)}.`,`每一行是一个部件，横条是它工作的时间。第一个字在 ${sT(r.vis)}后出现。`);}
function syncControls(){const c=$('#pdz-controls',app); if(!c) return;
  const lock=$('#pdz-lock',c), ctls=$('#pdz-ctls',c), rst=$('#pdz-reset-wrap',c);
  const msg=mode==='challenge'?tx(`Fixed in this challenge: ${dauT(CH.dau)} daily users, ${CH.msgs} messages each, peak ${CH.peak} times the average, no retrieval, $${CH.price} per GPU hour. The four design choices are in the task card.`,`这道挑战题固定：每天 ${dauT(CH.dau)} 用户，每人 ${CH.msgs} 条消息，高峰是平均的 ${CH.peak} 倍，不检索，每 GPU 小时 ${CH.price} 美元。四个设计选择在任务卡里。`)
    :mode==='guided'?tx(`Each part sets its own example: ${dauT(S.dau)} daily users, ${S.msgs} messages each, peak ${S.peak} times the average. Settings take effect in Scenarios.`,`每一部分用自己的例子：每天 ${dauT(S.dau)} 用户，每人 ${S.msgs} 条消息，高峰是平均的 ${S.peak} 倍。自由调参在应用场景里起作用。`):'';
  lock.hidden=!msg; lock.textContent=msg; ctls.hidden=!!msg; rst.hidden=!!msg; const hint=$('.ctl-hint',c); if(hint) hint.hidden=!!msg;
  if(msg) return;
  $$('.seg-ctl[data-key]',c).forEach(g=>{const k=g.dataset.key; if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
  const di=$('#pdz-ctl-dau-in',c); if(di&&document.activeElement!==di){di.value=Math.max(0,DAUS.indexOf(S.dau)); A.ui.setRange(di);} const dout=di&&di.closest('.ctl')&&di.closest('.ctl').querySelector('output'); if(dout) dout.textContent=dauT(S.dau);
  const pi=$('#pdz-ctl-price-in',c); if(pi&&document.activeElement!==pi){pi.value=S.price; A.ui.setRange(pi);} const po=pi&&pi.closest('.ctl')&&pi.closest('.ctl').querySelector('output'); if(po) po.textContent=usd(S.price);}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
const WHY=()=>({mission:tx('Send the message and watch it to the end to continue, or press Show me.','把这条消息播完才能继续，或点“演示给我看”。'),stream:tx('Try both options to continue, or press Show me.','两个选项都试过才能继续，或点“演示给我看”。'),
  context:tx('Try both options to continue, or press Show me.','两个选项都试过才能继续，或点“演示给我看”。'),predict:tx('Pick an answer to continue, or Skip.','选一个答案才能继续，或点“跳过”。'),
  route:tx(`Try ${pc(EASY)} and ${pc(0.9)} to continue, or press Show me.`,`试过 ${pc(EASY)} 和 ${pc(0.9)} 才能继续，或点“演示给我看”。`),safety:tx('Try “Check after all is sent” and one more option to continue, or press Show me.','试过“全部发完再查”和另一个选项才能继续，或点“演示给我看”。'),
  size:tx('Try another number of users to continue, or press Show me.','再试一种用户数才能继续，或点“演示给我看”。')}[STEPS[step].id]||'');
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:WHY(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#pdz-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}
/* the challenge readout: one line per goal, with its mark, and how many of the four are met (P32) */
function chReadHTML(e){const row=(ok,k,v)=>`<span class="pdz-chm ${ok?'pdz-ok':'pdz-bad'}">${ok?'✓':'✗'}</span><span>${k}</span><span class="mono">${v}</span>`;
  return row(e.okV,tx('First word','首字'),tx(`${sT(e.r.vis)} (goal ${f(CH.vis,1)} s)`,`${sT(e.r.vis)}（目标 ${f(CH.vis,1)} 秒）`))+
    row(e.okB,tx('Cost per day','每天花费'),tx(`${usd(e.r.costDay)} (budget ${usd(CH.budget)})`,`${usd(e.r.costDay)}（预算 ${usd(CH.budget)}）`))+
    row(e.okS,tx('Output check','输出检查'),e.okS?tx('checked before it is shown','先查过再显示'):tx('shown before it is checked','没查就显示了'))+
    row(e.okQ,tx('To the small model','交给小模型'),tx(`${pc(e.share)} (at most ${pc(EASY)})`,`${pc(e.share)}（上限 ${pc(EASY)}）`));}
function updateRail(r){const rail=$('#pdz-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()&&!PL.on) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step])&&!PL.on;
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${tx('Observed','观察到')}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again&&userAct) A.ui.obsBump(box);}
      if(obsReveal){obsReveal=false; requestAnimationFrame(revealObs);}}}
    A.ui.setNext(rail,isNextDisabled(),WHY());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-lkey]',rail).forEach(b=>{const key=b.dataset.lkey, on=String(S[key])===b.dataset.lever; b.setAttribute('aria-pressed',String(on)); b.classList.toggle('on',on);
      const rr=b.querySelector('.lv-r'), v=b.dataset.lever; if(rr){const vv=isNaN(+v)?v:+v, h=levTried(key,v)?esc(LEV[key].res(vv)):''; if(rr.innerHTML!==h) rr.innerHTML=h;}});
    $$('[data-act="play"]',rail).forEach(b=>{b.textContent=b.dataset.card?cardPlayTxt():playTxt();});
  } else if(mode==='sandbox'){const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#pdz-insight',rail); if(ins){const html=`<div class="eyebrow">${cur>=0?esc(L(PRESETS[cur].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {const e=chMe(), g=$('#pdz-ch-g',rail); if(!g) return; const gh=chReadHTML(e); if(g._h!==gh){g.innerHTML=gh; g._h=gh;}
    g.classList.toggle('met',e.ok); const met=[e.okV,e.okB,e.okS,e.okQ].filter(Boolean).length;
    $('#pdz-ch-p i',rail).style.width=(met/4*100)+'%'; $('#pdz-ch-p',rail).classList.toggle('met',e.ok); $('#pdz-ch-n',rail).textContent=tx(`${met} of 4 goals met`,`达标 ${met} / 4 项`);
    $$('.seg-ctl[data-key]',rail).forEach(g=>{const k=g.dataset.key; if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
    const res=$('#pdz-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=!!res._h&&!!html; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
/* the badge in the stage's corner: playing and paused say so (P6); a result names what fails, hard messages on the small model too (P10) */
function statusOf(r){
  if(mode==='challenge'){const e=chMe(); return e.ok?['ok',tx('All goals met','全部达标')]:['bad',tx('Not yet','未达标')];}
  if(PL.on) return ['neutral',tx('Playing','播放中')];
  if(!PL.done) return ['neutral',tx('Paused','已暂停')];
  if(!r.before&&r.s.stream) return ['bad',tx('Unchecked output','输出没查过')];
  if(!r.quality) return ['bad',tx('Hard messages on the small model','难题给了小模型')];
  return r.vis<=goalNow()+1e-9?['ok',tx('First word in time','首字按时')]:['bad',tx('First word too late','首字超时')];}
function render(){if(!app) return;
  const r=R_(), pn=panels(); measureW();
  $('#pdz-st-title',app).textContent=stageTitle(r);
  ['trace','prompt','fleet','sheet'].forEach(k=>{$('#pdz-'+k,app).hidden=!pn[k];});
  /* with the message on the stage, the metric cards come right under it and the other blocks follow, so the cards stay in view (P15) */
  $('#pdz-main',app).classList.toggle('pdz-mup',!!(pn.trace&&(pn.prompt||pn.fleet||pn.sheet)));
  /* part 8 (the message and the GPUs, nothing else): on a wide stage the cards sit beside the GPUs, under the message (product-design.css) */
  $('#pdz-main',app).dataset.lay=pn.trace&&pn.fleet&&!pn.prompt&&!pn.sheet?'tf':'';
  if(pn.trace) drawTrace(r);
  if(pn.prompt){const el=$('#pdz-prompt',app), h=promptHTML(r); if(el._h!==h){el.innerHTML=h; el._h=h;}}
  if(pn.fleet){const el=$('#pdz-fleet',app), h=fleetHTML(r); if(el._h!==h){el.innerHTML=h; el._h=h;}}
  if(pn.sheet){const el=$('#pdz-sheet',app), h=sheetHTML(r); if(el._h!==h){el.innerHTML=h; el._h=h;}}
  drawMetrics(r);
  const st=$('#pdz-status',app), [cls,txt]=statusOf(r);
  st.className='status '+cls; st.textContent=txt;
  st.dataset.tipx=mode==='challenge'?tx('Whether your design meets all four goals: first word, budget, checked output, hard messages on the large model.','你的设计是否四项都达标：首字时间、预算、输出查过、难题交给大模型。'):tx('Whether the first word appears within the time budget, every chunk is checked before it is shown, and no hard message goes to the small model. The cost is on the cards below.','第一个字是否在时间预算内出现，回答的每一段是否在显示前查过，以及有没有难题交给了小模型。花费看下面的卡片。');
  const mb=$('#pdz-mathbody',app), mh=mathHTML(r); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail(r);
  if(!PL.on) A.hash.write(token());
  A.dock.refresh(); announce(r);}
/* the width is measured on every drawing, so a play that starts before the layout settles is redrawn at the right size (P2) */
function measureW(){const m=app&&$('#pdz-main',app), mw=m?m.clientWidth:0; if(mw) W=Math.round(Math.max(300,Math.min(800,mw-26)));}
/* the slider lies under the time axis, from 0 to its end, so the slider and the cursor are one scale (P22) */
function alignScrub(){const pos=$('#pdz-pos',app), svg=$('#pdz-wf svg',app); if(!pos||!svg||!GEO) return; const bw=svg.getBoundingClientRect().width; if(!bw) return;
  const k=bw/GEO.W, th=14, ml=Math.round(GEO.x0*k-th/2), w=Math.round((GEO.x1-GEO.x0)*k+th), key=ml+'|'+w;
  if(pos._g!==key){pos._g=key; pos.style.marginLeft=ml+'px'; pos.style.width=w+'px';}}
function drawTrace(r){measureW(); const wf=$('#pdz-wf',app), X=axisOf(r), tau=PL.done?null:PL.t, h=wfSVG(r,tau,X);
  if(wf._h!==h){wf.innerHTML=h; wf._h=h;}
  const tt=$('#pdz-trace-t',app), tl=traceTitle(r); if(tt.textContent!==tl) tt.textContent=tl;
  const lgd=$('#pdz-leg1',app), lh=legendHTML(); if(lgd._h!==lh){lgd.innerHTML=lh; lgd._h=lh;}
  $$('[data-act="play"]',app).forEach(b=>{const x=b.dataset.card?cardPlayTxt():playTxt(); if(b.textContent!==x) b.textContent=x;});
  const pos=$('#pdz-pos',app); if(pos){const t=tau==null?X:tau; if(document.activeElement!==pos){pos.value=Math.round(1000*Math.min(1,t/X)); A.ui.setRange(pos);}
    pos.setAttribute('aria-valuetext',tx(`${sT(t)} after Send`,`按下发送后 ${sT(t)}`)); alignScrub();}}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app||PL.on) return; A.announce(tx(`First word after ${sT(r.vis)}; ${I(r.total)} GPUs, ${usd(r.costDay)} a day.`,`第一个字 ${sT(r.vis)}后出现；${I(r.total)} 张卡，每天 ${usd(r.costDay)}。`));},400);}

/* ---------- the trace player (R3.9, R3.20) ---------- */
/* the wall-clock plan of one play: pieces [t0,t1] of the message's time, each shown over [w0,w1] seconds of the reader's time. Before the first
   word every component gets at least DWELL (R7.11), a longer one its time at the axis rate; the first word holds HOLDFW; then the answer goes
   on at the axis rate to the end of the axis (P21). One rate for every picture of a part, at the reader's speed (P3). Quick: the whole axis
   in QUICK seconds */
function planOf(r,quick){const X=axisOf(r), P=[]; let w=0; const add=(t0,t1,d)=>{P.push({t0,t1,w0:w,w1:w+d}); w+=d;};
  if(quick){add(0,X,QUICK); return {P,X,total:w,fw:QUICK*Math.min(1,r.vis/X)};}
  const rate=X/SPEEDS[spd];
  r.seg.forEach(g=>{const a=Math.min(g.t0,X), b=Math.min(g.t0+g.d,X); if(b>a) add(a,b,Math.max(DWELL[spd],(b-a)/rate));});
  const v=Math.min(r.vis,X), fw=w; add(v,v,HOLDFW[spd]);
  if(X>v) add(v,X,(X-v)/rate);
  return {P,X,total:w,fw};}
const tOfW=(pl,w)=>{for(const p of pl.P) if(w<p.w1) return p.t0+(p.t1-p.t0)*Math.max(0,w-p.w0)/Math.max(1e-9,p.w1-p.w0); return pl.X;};
const wOfT=(pl,t)=>{for(const p of pl.P){if(t<p.t1-1e-9||(p.t1===p.t0&&Math.abs(t-p.t0)<1e-9)) return p.w0+(p.w1-p.w0)*(p.t1>p.t0?Math.max(0,t-p.t0)/(p.t1-p.t0):0);} return pl.total;};
function stopPL(){if(PL.raf) cancelAnimationFrame(PL.raf); PL.on=false; PL.raf=0;}
function frame(r){drawTrace(r); const tl=$('#pdz-st-title',app), x=stageTitle(r); if(tl.textContent!==x) tl.textContent=x;}
/* play from the message's time `from`; again: the same play at a new speed, not counted as a new one */
function playPL(from,quick,again){stopPL(); const r=R_(), pl=planOf(r,quick);
  PL.sig=sigOf(r); PL.plan=pl; PL.quick=!!quick; PL.t=from>=pl.X-1e-9?0:Math.max(0,from); PL.w=wOfT(pl,PL.t); PL.on=true; PL.done=false; PL.last=0; if(!again) nPlays++;
  const tick=ts=>{if(!PL.on||!app){PL.raf=0; return;}
    if(PL.last){PL.w+=Math.min(0.1,(ts-PL.last)/1000); PL.t=tOfW(pl,PL.w);
      if(PL.w>=pl.total-1e-9){PL.t=pl.X; PL.on=false; PL.done=true; PL.raf=0; watched.add(PL.sig); render(); keepStage(); return;}}
    PL.last=ts; frame(r); PL.raf=requestAnimationFrame(tick);};
  PL.raf=requestAnimationFrame(tick); render();}
/* show the message: a picture already watched shows its end at once (R3.20). A guided part plays at the reader's speed (P3); Scenarios,
   Settings and the challenge, where the reader switches back and forth, play the whole axis in QUICK (R3.9) */
function showTrace(src){stopPL(); const r=R_(), sig=sigOf(r);
  if(watched.has(sig)||A.reduceMotion()||src==='quiet'||!panels().trace){PL.t=axisOf(r); PL.done=true; if(src!=='quiet'||watched.has(sig)) watched.add(sig); render(); return;}
  playPL(0,src==='quick');}
/* ◂ ▸ stop at the components' ends: the start, each component's end, the first word, the end of the axis */
function stepBounds(r,X){const b=[0]; r.seg.forEach(g=>{const e=g.t0+g.d; if(e<X-1e-9) b.push(e);}); b.push(X); return b;}

/* ---------- state changes ---------- */
function baseFor(id){const i=SIX(id), b={...BASE};
  if(i>SIX('stream')) b.stream=1;
  if(i>SIX('context')) b.cache=1;
  if(i>SIX('route')) b.share=EASY;
  return b;}
function setState(p){Object.assign(S,p);
  if(mode==='challenge') Object.assign(S,{dau:CH.dau,msgs:CH.msgs,peak:CH.peak,ret:CH.ret,price:CH.price});
  showTrace(mode==='guided'?'lever':'quick');}
function enterStep(i){stopPL(); obsReveal=false; step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step], k=PART_LEV[s.id];
  S=baseFor(s.id); if(k&&pick[k]!=null) S[k]=pick[k];
  if(k&&!tried[k]) tried[k]=new Set([String(S[k])]);
  const st=$('#pdz-stage',app); if(st) st.scrollTop=0;
  renderRail();
  showTrace(s.id==='mission'?'auto':'quiet');}
/* a new part: its card's top comes under the header (A.ui.reveal). On a desktop, where the stage is sticky beside the card, the scroll
   stops where the bench still holds the whole stage: a short card ends the two columns early, and scrolling its top all the way up would
   push the stage up past the header (N4). The card's top stays on screen */
function revealStep(){const card=$('#pdz-rail .step.current',app)||$('#pdz-rail',app), st=$('#pdz-stage',app), bn=$('.bench',app);
  if(!card||!st||!bn||innerWidth<=760||getComputedStyle(st).position!=='sticky'){A.ui.reveal(card); return;}
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:0)+12;
  const c=card.getBoundingClientRect(), vh=innerHeight, T=parseFloat(getComputedStyle(st).top)||70, y0=scrollY;
  const end=bn.getBoundingClientRect().bottom-(parseFloat(getComputedStyle(bn).paddingBottom)||0), yMax=y0+end-T-st.offsetHeight;
  let y=c.top>=top&&c.top<=vh-160?y0:y0+c.top-top;
  if(y>yMax) y=Math.max(yMax,y0+c.top-(vh-160));
  y=Math.max(0,Math.round(y)); if(Math.abs(y-y0)<2) return;
  if(A.tip.quiet) A.tip.quiet(); window.scrollTo({top:y,behavior:A.reduceMotion()?'auto':'smooth'});}
/* desktop only (the stage stays in view beside the card): below the fold, the observation's top comes under the header; when only its end
   is cut off, a short scroll shows the end and keeps its first line on screen */
function revealObs(){const box=app&&$('#pdz-rail [data-obs]',app); if(!box||box.hidden||innerWidth<=760) return;
  const q=box.getBoundingClientRect(), vh=innerHeight, bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:0)+12;
  if(q.top>vh-160) A.ui.reveal(box); else if(q.bottom>vh) window.scrollBy({top:Math.max(0,Math.min(q.bottom-vh+12,q.top-top)),behavior:A.reduceMotion()?'auto':'smooth'});}
/* keepStage (R9.7), as labs/llm-judge.js */
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#pdz-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#pdz-rail .step.current',app)||$('#pdz-rail .railcard',app)||$('#pdz-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function setMode(m){obsReveal=false;
  if(m===mode) return; stopPL();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...BASE,...CH,...CH0}; delete S.vis; delete S.budget;}
  if(m==='sandbox'&&was==='guided') S={...PRESETS[0].p};
  A.shell.setMode(app,m,DESCS());
  const st=$('#pdz-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail(); showTrace('quiet');}
function applyPreset(i){const was=presetNow(); userAct=true; dBase=was>=0&&was!==i?{en:`“${PRESETS[was].t.en}”`,zh:`“${PRESETS[was].t.zh}”`}:null;
  try{S={...PRESETS[i].p}; showTrace('quick');}finally{userAct=false; dBase=null;}}
const token=()=>[DAUS.indexOf(S.dau),S.msgs,S.peak,S.stream,S.cache,S.ret,Math.round(S.share*10),S.filt,Math.round(S.price*10)].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==9) return null;
  const [d,m,pk,st,c,rt,sh,fl,pr]=p, o={dau:DAUS[+d],msgs:+m,peak:+pk,stream:+st,cache:+c,ret:+rt,share:+sh/10,filt:isNaN(+fl)?fl:+fl,price:+pr/10};
  if(o.dau==null||!MSGS.includes(o.msgs)||!PEAKS.includes(o.peak)||![0,1].includes(o.stream)||![0,1].includes(o.cache)||![0,1].includes(o.ret)||!SHARES.includes(o.share)||!FILTS.map(String).includes(String(o.filt))||!(o.price>=1&&o.price<=8)) return null;
  return o;}
function setLever(key,raw){const v=isNaN(+raw)?raw:+raw; pick[key]=v; (tried[key]=tried[key]||new Set()).add(String(v)); S[key]=v; showTrace('lever');}
function showMe(id){
  if(id==='mission'){const X=axisOf(R_()); playPL(!PL.done&&PL.t<X-1e-9?PL.t:0,false); return;}
  const k=PART_LEV[id], seq={stream:[0,1],context:[0,1],route:[0.9,EASY],safety:['after',16,8],size:[1e7,1e6]}[id];
  if(!k||!seq) return; seq.forEach(v=>{(tried[k]=tried[k]||new Set()).add(String(v));}); pick[k]=seq[seq.length-1]; S[k]=pick[k]; renderRail(); showTrace('lever');}
function act(a,btn){
  if(a==='play'){if(PL.on){stopPL(); render(); return;} const X=axisOf(R_()); playPL(!PL.done&&PL.t<X-1e-9?PL.t:0,false); return;}
  if(a==='replay'){playPL(0,false); return;}
  if(a==='back'||a==='fwd'){stopPL(); const r=R_(), X=axisOf(r), b=stepBounds(r,X), t=PL.done?X:PL.t;
    if(a==='back'){const prev=b.filter(x=>x<t-1e-9); PL.t=prev.length?prev[prev.length-1]:0; PL.done=false;}
    else {const nx=PL.done?b[1]:b.find(x=>x>t+1e-9); PL.t=nx==null?X:nx; if(PL.t>=X-1e-9){PL.t=X; PL.done=true; watched.add(sigOf(r));} else PL.done=false;}   // from the whole run, ▸ lands on the first component's end (R7.10)
    A.tip.quiet&&A.tip.quiet(); render(); return;}
  if(a==='reset'){userAct=true; try{S={...PRESETS[0].p}; showTrace('quick');}finally{userAct=false;} return;}
  if(a==='submit'){chSubmitted=true; {const m=chMe(), b=chBest(); if(m.ok&&b&&m.r.costDay<=b.r.costDay+1e-9) chReveal=true;} renderRail(); keepStage(); return;}
  if(a==='reveal'){chReveal=true; renderRail(); keepStage(); return;}
  if(a==='copy'){(navigator.clipboard?navigator.clipboard.writeText(CODE):Promise.reject()).catch(()=>{}); flashBtn(btn,tx('Copied','已复制'),A.t('copyCode')); return;}
  if(a==='drill-new'){drill=makeDrill(((drill&&drill.seed)||0)+1); drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* interview drill: a new sizing case from the engine each time */
function makeDrill(seed){const rnd=ENG.s.mulberry(900+seed), dau=[2e5,5e5,2e6,5e6][Math.floor(rnd()*4)], msgs=[5,10,20][Math.floor(rnd()*3)], peak=[2,3][Math.floor(rnd()*2)], share=[0.5,0.7][Math.floor(rnd()*2)];
  const s={...BASE,dau,msgs,peak,cache:1,share,filt:8}; return {seed,s,r:compute(s)};}
function renderDrill(){if(!app) return; if(!drill) drill=makeDrill(0); const {s,r}=drill, q=$('#pdz-drill-q',app), an=$('#pdz-drill-a',app);
  q.innerHTML=tx(`A chat app has ${dauT(s.dau)} daily users who send ${s.msgs} messages each; the busiest hour runs at ${s.peak} times the average. A router sends ${pc(s.share)} of messages to ${MODELS.s.name} (one H100 takes ${f(r.P.s.g,2)} messages per second within the goals) and the rest to ${MODELS.l.name} (a 4-GPU replica takes ${f(r.P.l.g,2)}). How many model GPUs does the peak need, and what do they cost per day at $${s.price} per GPU hour?`,
    `一个聊天应用每天有 ${dauT(s.dau)} 用户，每人发 ${s.msgs} 条消息；最忙那个小时是平均速率的 ${s.peak} 倍。路由把 ${pc(s.share)} 的消息交给 ${MODELS.s.name}（一张 H100 在达标的前提下每秒接 ${f(r.P.s.g,2)} 条），其余交给 ${MODELS.l.name}（4 张卡的一份副本每秒接 ${f(r.P.l.g,2)} 条）。高峰时模型要多少张卡？每 GPU 小时 ${s.price} 美元，每天花多少？`);
  an.hidden=!drillOpen;
  const mg=r.gpus.s+r.gpus.l;
  an.innerHTML=drillOpen?tx(`<p><b>${I(mg)} model GPUs, ${usd(mg*24*s.price)} a day.</b></p>`,`<p><b>模型要 ${I(mg)} 张卡，每天 ${usd(mg*24*s.price)}。</b></p>`)+
    `<div class="pdz-eqs">`+[[`\\lambda`,`= \\frac{${I(s.dau).replace(/,/g,'{,}')} \\times ${s.msgs}}{86{,}400} \\times ${s.peak} = ${f(r.lam,2)}`],[tt(tx('small','小')),`= \\operatorname{ceil}(${f(r.rate.s,2)} \\div ${f(r.P.s.g,2)}) = ${I(r.gpus.s)}`],[tt(tx('large','大')),`= \\operatorname{ceil}(${f(r.rate.l,2)} \\div ${f(r.P.l.g,2)}) \\times 4 = ${I(r.gpus.l)}`],[tt(tx('cost','花费')),`= ${I(mg)} \\times 24 \\times ${s.price} = ${I(mg*24*s.price)}`]].map(([l,x])=>`<span class="pdz-eq-l">${M(l)}</span><span class="pdz-eq-r">${M(x)}</span>`).join('')+`</div>`+
    `<p class="small">${tx('Round each pool up to whole replicas before multiplying by the GPUs per replica; the output checks would add a few more GPUs.','每一组先向上取整到整份副本，再乘以每份的卡数；输出检查还要再加几张卡。')}</p>`:'';}
/* capability layer */
function EXITS(){const b=compute(baseFor('remember')), r9=compute({...baseFor('remember'),share:0.9}), x10=compute({...baseFor('remember'),dau:BASE.dau*10});
  return {core:()=>chSubmitted,
   explain:[
    {q:tx('Why does streaming not change how many GPUs the app needs?','为什么流式输出不改变应用要的卡数？'),
     o:[tx('The model writes the same tokens either way; streaming only changes when they are sent, so the GPU work per message is the same.','两种做法下模型写的 token 一样多；流式输出只改变什么时候发出去，所以每条消息的 GPU 工作量不变。'),
        tx('Streaming sends fewer tokens, because the user stops reading early.','流式输出发的 token 更少，因为用户会提前停止阅读。'),
        tx('Streaming moves the decode work to the user’s browser.','流式输出把 decode 的工作搬到了用户的浏览器里。')],a:0,
     n:[null,tx('The model writes the whole answer whether or not the user reads it; only a closed connection stops it early.','不管用户读不读，模型都会把整段回答写完；只有连接断开才会提前停。'),tx('Every token is still generated on the GPUs; the browser only shows the text.','每个 token 仍然在 GPU 上生成；浏览器只负责显示文字。')],
     why:tx('Streaming changes the time to the first visible word, not the work: the same tokens are generated, and they are sent in pieces as they appear.','流式输出改变的是首字可见时间，不是工作量：生成的 token 一样多，只是写出来一段就发一段。')},
    {q:tx('Why must the output check hold back each chunk instead of checking once at the end?','为什么输出检查要把每一段扣住，而不是最后查一次？'),
     o:[tx('With streaming, by the end the user has already seen every word; holding each chunk until it passes means nothing unchecked is ever shown.','流式输出时，等到“最后”，整段回答用户都已经看到了；每一段查过再放，才能保证没查过的内容不会显示出来。'),
        tx('A check at the end costs more GPUs than many small checks.','最后查一次比分段查很多次更费卡。'),
        tx('The classifier cannot read more than a few tokens at a time.','分类器一次只能读几个 token。')],a:0,
     n:[null,tx('One check at the end is the cheapest option; it is rejected because it comes too late, not because of its cost.','最后查一次反而最省卡；它被否定是因为太晚，不是因为贵。'),tx('Classifiers such as Llama Guard read the whole conversation; the chunk size is a choice of the service.','Llama Guard 这类分类器能读整段对话；分多大一段是服务端自己的选择。')],
     why:tx('The check has to come between the model and the user. Small chunks show the first words sooner but need more checks; the chunk size trades latency against check cost.','检查必须放在模型和用户之间。段越小，第一个字出现得越早，但查的次数越多；分段的大小是在延迟和检查成本之间取舍。')}],
   transfer:[
    {q:tx(`This app’s daily users grow from ${dauT(BASE.dau)} to ${dauT(BASE.dau*10)}, with everything else unchanged. It needs ${I(b.total)} GPUs now. About how many will it need?`,`这个应用每天的用户从 ${dauT(BASE.dau)} 涨到 ${dauT(BASE.dau*10)}，其他都不变。现在要 ${I(b.total)} 张卡，到时大约要多少？`),
     o:[tx(`About ${I(Math.round(x10.total/10)*10)}`,`约 ${I(Math.round(x10.total/10)*10)} 张`),tx(`About ${I(Math.round(b.total*3/10)*10)}`,`约 ${I(Math.round(b.total*3/10)*10)} 张`),tx(`About ${I(b.total)}: the GPUs were sized with headroom`,`约 ${I(b.total)} 张：配卡时留了余量`)],a:0,
     why:tx(`In this lab it is ${I(x10.total)}: the peak rate grows tenfold and each replica still takes the same goodput, so the GPUs grow about tenfold too. ${x10.total<10*b.total?`At ${dauT(BASE.dau)} users the rounding up to whole replicas is a larger share, so it is a little less than ${I(10*b.total)}.`:''} The first word stays at ${sT(x10.vis)}.`,`在这个 Lab 里是 ${I(x10.total)} 张：高峰速率涨到 10 倍，每份副本能接的还是那么多，所以卡也差不多涨到 10 倍。${x10.total<10*b.total?`${dauT(BASE.dau)} 用户时向上取整多出的零头占比更大，所以比 ${I(10*b.total)} 张还少一点。`:''}第一个字仍在 ${sT(x10.vis)}出现。`)},
    {q:tx(`A product manager proposes routing ${pc(0.9)} of messages to the small model to cut the daily cost from ${usd(b.costDay)} to ${usd(r9.costDay)}. What should you check first?`,`产品经理提议把 ${pc(0.9)} 的消息交给小模型，把每天的花费从 ${usd(b.costDay)} 降到 ${usd(r9.costDay)}。你应该先查什么？`),
     o:[tx('Whether the small model answers that much of the traffic well enough, measured with evals on real messages','用真实消息做评测，看小模型能不能答好这么多的流量'),tx('Whether the gateway can handle the extra requests','网关能不能处理多出来的请求'),tx('Whether streaming still works with the small model','换成小模型以后流式输出还能不能用')],a:0,
     why:tx(`The request rate does not change, and both models stream the same way. The risk is quality: in this lab only ${pc(EASY)} of messages are easy enough, so the extra ${pc(0.9-EASY)} would get worse answers.`,`请求的速率没变，两个模型的流式输出也一样。风险在质量：这个 Lab 假设只有 ${pc(EASY)} 的消息够简单，多交出去的 ${pc(0.9-EASY)} 会得到更差的回答。`)}],
   practice:[{id:'chat-app-context',d:tx('Design the backend of a consumer assistant app: streaming, conversation storage, context, routing, safety filtering and capacity planning.','设计一个消费级助手应用的后端：流式输出、会话存储、上下文、路由、安全过滤和容量规划。')},
     {id:'inference-serving',d:tx('Design an LLM serving system with latency targets and size it for a given load.','设计一个有延迟目标的大模型推理服务，并按给定负载估算规模。')}]};}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  const C={stream:tx('Streaming','流式输出'),cache:tx('Prompt caching','prompt 缓存'),share:tx('Messages routed to the small model','交给小模型的消息'),filt:tx('Output check','输出检查'),dau:tx('Daily users','每天的用户数'),price:tx('Price per GPU hour','每 GPU 小时的价格')};
  if(!C[k]) return '';
  if(mode==='challenge'&&!['stream','cache','share','filt'].includes(k)) return tx(`Fixed during the challenge: “${C[k]}”.`,`挑战中固定不变：${C[k]}。`);
  if(mode==='guided') return tx(`Fixed in this part: “${C[k]}”.`,`这一部分固定：${C[k]}。`);
  return tx(`Click to jump to “${C[k]}”.`,`点击跳到“${C[k]}”。`);}
function jump(k,src){if(!app) return;
  if(k==='controls'){const c=$('#pdz-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  const locked=mode==='guided'||(mode==='challenge'&&!['stream','cache','share','filt'].includes(k));
  if(locked){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=mode==='challenge'?$('#pdz-ic-'+k,app):$('#pdz-ctl-'+k,app)||$('#pdz-ctl-'+k+'-in',app); if(el) A.ui.jumpToEl(el);}
const TERM={stream:()=>D_STREAM(),cache:()=>D_CACHE(),share:()=>D_ROUTE(),filt:()=>D_SAFE(),dau:()=>tx('Daily users: users who send at least one message on a given day.','每天的用户数：某一天至少发了一条消息的用户。'),price:()=>D_GPUH()};

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('toggle',e=>{const d=e.target; if(d&&d.dataset&&d.dataset.more) moreOpen[d.dataset.more]=d.open;},true);
  el.addEventListener('input',e=>{const t=e.target;
    if(t.id==='pdz-pos'){stopPL(); const r=R_(), X=axisOf(r); PL.t=X*(+t.value)/1000; PL.done=PL.t>=X-1e-9; if(PL.done){PL.t=X; watched.add(sigOf(r));} render(); return;}
    const k=t.dataset&&t.dataset.key;
    if(k==='dau'){userAct=true; try{S.dau=DAUS[+t.value]; render();}finally{userAct=false;}}
    if(k==='price'){userAct=true; try{S.price=Math.round(+t.value*2)/2; render();}finally{userAct=false;}}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset, inRail=!!b.closest('#pdz-rail');
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='spd'){spd=d.v; A.ui.setPressed(seg,d.v); if(PL.on&&!PL.quick) playPL(PL.t,false,true); return;}
      const v=k==='filt'?(isNaN(+d.v)?d.v:+d.v):+d.v;
      userAct=true; try{setState({[k]:v});}finally{userAct=false;} if(inRail) keepStage(); return;}
    if(d.lever!=null&&d.lkey){obsReveal=true; userAct=true; try{setLever(d.lkey,d.lever);}finally{userAct=false;} renderRail(); keepStage(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; obsReveal=true; if(s.showMe) showMe(s.showMe); keepStage(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#pdz-rail [data-pred="${pred}"]`,app); if(fb) fb.focus(); keepStage(); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#pdz-rail',app)); return;}
    if(d.act){act(d.act,b); if(inRail&&!b.closest('[data-exits]')&&d.act!=='copy') keepStage();}
  });}
/* the lab state contract (round 3 #6): what the learner did here, saved by the kit and put back once per page load before mount */
const sOK=o=>o&&typeof o==='object'?parse([DAUS.indexOf(o.dau),o.msgs,o.peak,o.stream,o.cache,o.ret,Math.round(o.share*10),o.filt,Math.round(o.price*10)].join('_')):null;
const LEVK=Object.values(PART_LEV), levOK=(k,v)=>!!sOK({...BASE,[k]:v});
function snap(){const tr={}; for(const k in tried) tr[k]=[...tried[k]];
  return {v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,pick:{...pick},tried:tr,chSubmitted,chReveal,
    drillSeed:drill?drill.seed:null,drillOpen,spd,watched:[...watched].slice(-12),passed:mode==='challenge'&&chSubmitted&&chMe().ok};}
function restore(o){if(!o||o.v!==1) return; const md=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
  let s=null; if(md==='challenge'){s={...BASE,...CH,...CH0}; delete s.vis; delete s.budget;   // the challenge's users and messages are its own; only its four choices come back
    if(o.S&&typeof o.S==='object') ['stream','cache','share','filt'].forEach(k=>{if(levOK(k,o.S[k])) s[k]=o.S[k];});} else s=sOK(o.S);
  if(!s) return; mode=md; S=s;
  step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.includes(o.pred)?o.pred:null;
  observed={}; if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
  const sv=sOK(o.saved); saved=mode==='challenge'&&sv?sv:null;
  for(const k in pick) delete pick[k]; if(o.pick&&typeof o.pick==='object') LEVK.forEach(k=>{if(o.pick[k]!=null&&levOK(k,o.pick[k])) pick[k]=o.pick[k];});
  tried={}; if(o.tried&&typeof o.tried==='object') LEVK.forEach(k=>{const a=o.tried[k]; if(Array.isArray(a)) tried[k]=new Set(a.filter(v=>typeof v==='string'&&v.length<20).slice(0,20));});
  chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
  drill=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?makeDrill(o.drillSeed):null; drillOpen=!!drill&&o.drillOpen===true;
  spd=SPEEDS[o.spd]?o.spd:'med';
  watched.clear(); if(Array.isArray(o.watched)) o.watched.slice(-12).forEach(x=>{if(typeof x==='string'&&x.length<600) watched.add(x);});}
A.registerLab({
  id:ID, snap, restore,
  mount(el,tok){stopPL(); root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#pdz-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#pdz-rail',app)); keepStage();},token});
    bind(app); renderRail(); renderDrill();
    /* the stage's width can settle after the first drawing (fonts, the rail, the page's scroll bar): redraw at the new width (P2) */
    if(ro) ro.disconnect(); ro=window.ResizeObserver?new ResizeObserver(()=>{if(!app) return; const w0=W; measureW(); if(W!==w0&&!PL.on) render();}):null; if(ro) ro.observe($('#pdz-main',app));
    if(mode==='guided'&&step===0&&!watched.size) showTrace('auto'); else {PL.t=axisOf(R_()); PL.done=true; render();}},
  leave(){stopPL(); stageStop(); clearTimeout(annT); if(ro){ro.disconnect(); ro=null;} root=null; app=null;},
  onLang(){if(root){const d=$('#pdz-math',app), open=d&&d.open; if(PL.on){stopPL(); const r=R_(); PL.t=axisOf(r); PL.done=true; watched.add(sigOf(r));} this.mount(root,null); if(open) $('#pdz-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(k){const tm=TERM[k]; if(!tm) return ''; return `<div>${tm()}</div><div class="act">${jumpAct(k)}</div>`;},
  jump, jumpAct,
  dock:{
    /* the bar shows only once the stage's title row has gone, so it never covers the stage's title and badge (P15). The whole #pdz-main
       is taller than the screen in part 8 and the scenarios, so with it as the anchor the bar sat on the sticky stage's title. While the
       stage is stuck on a desktop the kit keeps the bar away itself (kit polish K4), so the anchor no longer switches to the whole stage */
    anchor:()=>app&&$('#pdz-stage .stage-head',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#pdz-main',app),
    render({status,vis,read}){if(!app) return; const r=R_(), g=goalNow();
      vis.innerHTML=`<div class="pdz-dk"><i style="width:${Math.min(100,r.vis/(g*1.5)*100).toFixed(1)}%"></i><b style="left:${(100/1.5).toFixed(1)}%"></b></div>`;
      if(mode==='challenge'){const e=chMe(); status.className='status '+(e.ok?'ok':'bad'); status.textContent=e.ok?tx('Goals met','已达标'):tx('Not yet','未达标');
        read.textContent=tx(`First word ${sT(e.r.vis)}, ${usd(e.r.costDay)} a day`,`首字 ${sT(e.r.vis)}，每天 ${usd(e.r.costDay)}`); return;}
      const chk=r.before||!r.s.stream, ok=r.vis<=g+1e-9&&chk&&r.quality; status.className='status '+(ok?'ok':'bad'); status.textContent=ok?tx('In time','按时'):!chk?tx('Unchecked','没查过'):!r.quality?tx('Hard on small model','难题给了小模型'):tx('Too late','超时');
      read.textContent=tx(`First word after ${sT(r.vis)}; ${I(r.total)} GPUs, ${usd(r.costDay)} a day`,`首字 ${sT(r.vis)}；${I(r.total)} 张卡，每天 ${usd(r.costDay)}`);}
  }
});

/* read-only hook for the golden-number tests and the home tile */
window.__pdz=Object.freeze({compute:s=>{const r=compute({...BASE,...s}); return {lam:r.lam,rep:r.rep,gpus:r.gpus,fG:r.fG,total:r.total,costDay:r.costDay,perUserMonth:r.perUserMonth,storeGB:r.storeGB,tpot:r.tpot,vis:r.vis,hold:r.hold,tFirst:r.tFirst,tEnd:r.tEnd,seg:r.seg.map(g=>({k:g.k,t0:g.t0,d:g.d})),before:r.before,quality:r.quality,nChk:r.nChk,tModel:r.tModel,gen:r.gen,cps:r.cps,fRaw:r.fRaw};},
  pool:(m,c,ret)=>({...pool(m,!!c,!!ret)}),runAt:(m,i,c,ret)=>({...runAt(m,i,!!c,!!ret)}),quiet:(m,c,ret)=>({...quiet(m,!!c,!!ret)}),
  chAll:()=>chAll().map(e=>({stream:e.stream,cache:e.cache,share:e.share,filt:e.filt,cost:e.r.costDay,vis:e.r.vis,ok:e.ok,okV:e.okV,okB:e.okB,okS:e.okS,okQ:e.okQ})),
  chBest:()=>{const b=chBest(); return b?{stream:b.stream,cache:b.cache,share:b.share,filt:b.filt,cost:b.r.costDay,vis:b.r.vis}:null;},
  drill:sd=>{const d=makeDrill(sd); return {s:d.s,gs:d.r.gpus.s,gl:d.r.gpus.l,lam:d.r.lam};},
  /* the fleet's drawn area, in squares of SQ GPUs, by group (P1): what fleetHTML draws for this setting */
  fleetUnits:s=>{const r=compute({...BASE,...s}), c=fleetCells(r); return {sq:SQ,cap:SQCAP,l:c.l.u,s:c.s.u+c['s hard'].u,hard:c['s hard'].u,f:c.f.u,units:c.l.u+c.s.u+c['s hard'].u+c.f.u,total:r.total,hardG:hardOf(r)};},
  /* the play plan for the current picture at the current speed (quick: the Scenarios speed) */
  plan:q=>{const p=planOf(R_(),!!q); return {X:p.X,total:p.total,fw:p.fw,P:p.P.map(x=>({...x}))};},
  dwell:Object.freeze({...DWELL}),holdfw:Object.freeze({...HOLDFW}),speeds:Object.freeze({...SPEEDS}),quick:QUICK,
  consts:Object.freeze({LAT,TTFT_GOAL,CHECK_GPU,WL,BASE,BRIEF,CH,CH0,EASY,SHARES,FILTS,DAUS,MSGS,PEAKS,STEP_IDS:STEPS.map(s=>s.id),PRESETS:PRESETS.map(p=>p.p),MODELS:Object.fromEntries(Object.entries(MODELS).map(([k,m])=>[k,{n:m.n,budget:m.budget,M:m.M}]))}),
  state:()=>({...S,mode,step,pred,playing:PL.on,t:PL.t,w:PL.w,done:PL.done,quick:PL.quick,X:axisOf(R_()),total:PL.plan?PL.plan.total:null,plays:nPlays,spd,watched:[...watched],chSubmitted,chReveal,tried:Object.fromEntries(Object.entries(tried).map(([k,v])=>[k,[...v]]))})});
})();
