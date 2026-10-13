/* ENGINE BEGIN — the serving capstone's event/time model. A line-by-line port of capstone/engine.py (the Python reference);
   tests/serving-capstone_engine.py runs both on every load and setting and compares them. Pure: no DOM, no window.
   One H100 SXM serves Llama-3.1-8B in BF16; every engine step is priced by the step model the other inference labs use:
     t = max(FLOPs / (989.5 TFLOP/s × 50%), bytes / (3.35 TB/s × 60%))
     FLOPs = 2·P_LAYERS·tokens + 2·P_HEAD·logits + attention;  bytes = W_READ + 128 KiB × (KV tokens the step reads)
     a decode token at context c: attention attn(c, c+1), reads c tokens of KV, 1 logit;
     a prefill chunk a..b−1: attention attn(a, b), reads b tokens of KV, 1 logit if it is the prompt's last chunk.
   Continuous batching (simplified vLLM v1), at every step boundary:
     1 every running request with its first token and answer tokens left decodes 1 token (1 token of the budget each);
     2 running requests still in prefill continue, in admission order, each taking min(prompt left, budget left);
     3 waiting requests that have arrived are admitted first come, first served, while running < cap, budget is left, and the KV pool
       can hold the prompt beyond its cache hit plus the whole answer (room is set aside, so nothing is ever preempted);
       the first request that cannot be admitted stops admission, and every request behind it waits for the same reason.
     The budget counts every token of the step (vLLM's max_num_batched_tokens); budget 0 = no limit (a whole prompt in one step).
     Every event of a step happens at its end; a request that arrives during a step waits for its end.
   Static batching: a batch of up to cap waiting requests starts only when the whole previous batch is done; its prompts are prefilled
     together in one step, then it decodes until its longest answer is done.
   KV memory in 16-token blocks: a request holds its prompt blocks as they are computed and a new answer block every 16 tokens. With
     prefix caching on, an app's system-prompt blocks are shared: one physical copy with a reference count; requests that compute the
     same block in the same step keep one copy; blocks nobody uses stay cached (idle) within the cache budget, least recently used
     evicted first, and a new allocation evicts idle blocks when the pool runs out. Blocks in use are never evicted. */
const SC_ENGINE=(function(){
'use strict';
const L=32, D=4096, KVH=8, HD=128, FFN=14336, VOCAB=128256;
const P_LAYER=2*D*D+2*D*KVH*HD+3*D*FFN+2*D, P_LAYERS=L*P_LAYER+D, P_HEAD=VOCAB*D, P_TOTAL=P_LAYERS+2*P_HEAD;
const KV_TOK=2*L*KVH*HD*2, W_READ=2*(P_LAYERS+P_HEAD), W_MEM=2*P_TOTAL;
const FS=1979e12/2*0.5, BS=3.35e12*0.6, GiB=2**30, BLOCK=16;
const POOL_T=Math.floor((80*GiB*0.9-W_MEM-2*GiB)/KV_TOK), POOL_B=Math.floor(POOL_T/BLOCK);
const EPS=1e-9;
const attn=(a,b)=>2*L*D*(b*(b+1)-a*(a+1));
const blocks=n=>Math.floor((n+BLOCK-1)/BLOCK);

function mulberry(seed){let h=seed>>>0; return ()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};}
const expGap=(rnd,rate)=>-Math.log(1-rnd())/rate;
const ms=x=>Math.floor(x*1000+0.5)/1000;
const uni=(rnd,lo,hi)=>lo+Math.floor(rnd()*(hi-lo+1));

/* the fixed loads: the same seeded draws, in the same order, as capstone/engine.py */
function loadStall(seed=11){const rnd=mulberry(seed), out=[]; let t=0;
  for(let i=0;i<36;i++){t+=expGap(rnd,4.0); const own=uni(rnd,400,1600), o=uni(rnd,120,320); out.push({t:ms(t),cls:'chat',own,out:o});}
  out.push({t:2.5,cls:'doc',own:32768,out:200}); return sortLoad(out);}
function loadShared(seed=12,n=48,rate=6.0,pre=6144,app='support'){const rnd=mulberry(seed), out=[]; let t=0;
  for(let i=0;i<n;i++){t+=expGap(rnd,rate); const own=uni(rnd,200,800), o=uni(rnd,100,300); out.push({t:ms(t),cls:'chat',app,pre,own,out:o});}
  return out;}
function loadApps(seed=22,n=48,rate=1.5,pre=24576,mix=[['manual',.5],['policy',.25],['codebase',.15],['paper',.1]]){const rnd=mulberry(seed), out=[]; let t=0;
  for(let i=0;i<n;i++){t+=expGap(rnd,rate); const u=rnd(); let acc=0, app=mix[mix.length-1][0];
    for(const [a,p] of mix){acc+=p; if(u<acc){app=a; break;}}
    const own=uni(rnd,100,400), o=uni(rnd,40,120); out.push({t:ms(t),cls:'chat',app,pre,own,out:o});}
  return out;}
function loadChallenge(seed=21,n=64,rate=6.0){const rnd=mulberry(seed), out=[]; let t=0;
  const apps=[['support',4096,.2],['code',8192,.15],['contracts',12288,.1]];
  for(let i=0;i<n;i++){t+=expGap(rnd,rate); const u=rnd(); let acc=0, s=null;
    for(const [a,pre,p] of apps){acc+=p; if(u<acc){s={cls:'chat',app:a,pre,own:uni(rnd,200,800)}; break;}}
    if(!s) s={cls:'chat',own:uni(rnd,300,1500)};
    s.t=ms(t); s.out=uni(rnd,80,240); out.push(s);}
  out.push({t:3.0,cls:'doc',own:24576,out:200}); out.push({t:8.0,cls:'doc',own:16384,out:200});
  return sortLoad(out);}
/* Python's sort is stable: equal times keep their order */
function sortLoad(a){return a.map((s,i)=>[s,i]).sort((x,y)=>x[0].t-y[0].t||x[1]-y[1]).map(x=>x[0]);}
const LOADS={stall:loadStall, shared:loadShared, apps:loadApps, challenge:loadChallenge};
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
return {simulate,metrics,check,pct,gapOf,attn,blocks,mulberry,LOADS,GOALS,TARGET,BASELINE,
  C:{L,D,KVH,HD,FFN,VOCAB,P_LAYER,P_LAYERS,P_HEAD,P_TOTAL,KV_TOK,W_READ,W_MEM,FS,BS,GiB,BLOCK,POOL_T,POOL_B}};
})();
/* ENGINE END */
if(typeof module!=='undefined') module.exports=SC_ENGINE;
